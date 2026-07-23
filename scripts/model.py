"""
model.py
========

Implements the SignLanguageTranslation model.
Components:
1. Transformer Visual Encoder: models temporal dependencies of landmark sequences.
2. VLMapper: projects visual feature dimensions (d_model) to mBART decoder dimensions.
3. mBART Decoder: generates translated English sentences from projected representations.

Diagnostics added in this revision:
- vl_mapper upgraded from a single nn.Linear to a 2-layer MLP bridge (GELU + LayerNorm + Dropout),
  giving the mapper enough capacity to actually reshape visual features into mBART's embedding
  space instead of acting as a near-linear pass-through the decoder can learn to ignore.
- compute_module_grad_norms(): call after loss.backward() + optimizer.unscale_() in the training
  loop to get separate gradient norms for the visual encoder, the vl_mapper, and the decoder's
  cross-attention projections. Used to verify all three are actually receiving gradient signal.
- attach_cross_attn_diagnostic_hook() / detach_cross_attn_diagnostic_hook() / compute_cross_attn_entropy():
  a forward_hook based probe on mbart.model.decoder.layers[0].encoder_attn that captures attention
  weights during a diagnostic forward pass (output_attentions=True) and reports their entropy, so
  we can tell whether the decoder is actually attending to specific visual tokens or just spreading
  attention near-uniformly across the whole sequence (i.e. ignoring the visual context).
"""

from __future__ import annotations

import math
import logging
from typing import Any

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for sequence modeling.
    """
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, SeqLen, d_model)
        return x + self.pe[:, :x.size(1)]


CURATED_FACE_INDICES = [
    # Left Eyebrow (5 points)
    70, 63, 105, 66, 107,
    # Right Eyebrow (5 points)
    336, 296, 334, 293, 300,
    # Eyes (Left & Right - 8 points)
    33, 133, 159, 145,  # Left eye
    263, 362, 386, 374, # Right eye
    # Nose (4 points)
    1, 4, 197, 6,
    # Mouth Outer Outline (14 points)
    61, 185, 40, 37, 0, 267, 269, 291, 321, 314, 17, 84, 181, 91,
    # Mouth Inner (4 points)
    78, 14, 308, 324
]

class LoraLinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with a Low-Rank Adaptation (LoRA) branch.
    """
    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.original_layer = original_layer
        # Freeze original projection
        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA projection matrices
        self.lora_A = nn.Parameter(torch.zeros((in_features, r)))
        self.lora_B = nn.Parameter(torch.zeros((r, out_features)))
        self.scaling = alpha / r

        # Init weights: A is Kaiming uniform, B is zeroed (ensures no modification at start)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original_layer(x)
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B) * self.scaling
        return orig_out + lora_out


class SignTranslationModel(nn.Module):
    """
    Seq2Seq model mapping landmark sequences to English text sentences.
    """
    def __init__(
        self,
        input_dim: int = 1665,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        mbart_model_name: str = "facebook/mbart-large-50",
        use_modality_encoders: bool = False,
        decoder_adapt_mode: str = "frozen",
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        use_auxiliary_ctc: bool = False,
        gloss_vocab_size: int | None = None
    ) -> None:
        super().__init__()
        # Load mBART for translation task
        self.mbart = MBartForConditionalGeneration.from_pretrained(mbart_model_name, local_files_only=True)

        # Freeze base mBART weights by default
        for param in self.mbart.parameters():
            param.requires_grad = False

        self.use_modality_encoders = use_modality_encoders
        if self.use_modality_encoders:
            # Curated 40 points face coordinates indexes mapping
            face_flat_indices = []
            for k in CURATED_FACE_INDICES:
                face_flat_indices.extend([132 + k*3, 132 + k*3 + 1, 132 + k*3 + 2])
            self.register_buffer("face_flat_indices", torch.tensor(face_flat_indices, dtype=torch.long))

            # Modal projection encoders
            self.pose_enc = nn.Linear(133, 128)
            self.face_enc = nn.Linear(120, 64)
            self.left_hand_enc = nn.Linear(64, 128)
            self.right_hand_enc = nn.Linear(64, 128)
            self.concat_proj = nn.Linear(128 + 64 + 128 + 128, d_model)
        else:
            # Standard flat projection
            self.linear_in = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        # 2. Transformer Visual Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.visual_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 3. VLMapper: Map visual encoder dimensions (512) to mBART decoder dimensions (usually 1024)
        # Upgraded to a 2-layer MLP bridge. A single nn.Linear here is expressive enough that the
        # decoder's language-model prior can learn to route around it (it just needs to find an
        # approximately-consistent bias/offset it can ignore); the extra hidden layer + GELU forces
        # the mapper to actually re-represent the visual features rather than pass them through
        # near-linearly, and the LayerNorms keep the scale compatible with mBART's embedding space.
        mbart_dim = self.mbart.config.d_model
        self.vl_mapper = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, mbart_dim),
            nn.LayerNorm(mbart_dim)
        )

        # 4. Auxiliary CTC Classifier Head
        self.ctc_head = None
        if use_auxiliary_ctc and gloss_vocab_size is not None:
            self.ctc_head = nn.Linear(d_model, gloss_vocab_size)
            logger_model = logging.getLogger("model")
            logger_model.info("Auxiliary CTC loss head initialized with vocab size %d", gloss_vocab_size)

        # 5. Decoder Adapt Mode
        if decoder_adapt_mode == "cross_attn_unfrozen":
            for layer in self.mbart.model.decoder.layers:
                for proj in [layer.encoder_attn.q_proj, layer.encoder_attn.k_proj,
                             layer.encoder_attn.v_proj, layer.encoder_attn.out_proj]:
                    for p in proj.parameters():
                        p.requires_grad = True
        elif decoder_adapt_mode == "lora_cross_attn":
            for layer in self.mbart.model.decoder.layers:
                layer.encoder_attn.q_proj = LoraLinear(layer.encoder_attn.q_proj, r=lora_r, alpha=lora_alpha)
                layer.encoder_attn.k_proj = LoraLinear(layer.encoder_attn.k_proj, r=lora_r, alpha=lora_alpha)
                layer.encoder_attn.v_proj = LoraLinear(layer.encoder_attn.v_proj, r=lora_r, alpha=lora_alpha)
                layer.encoder_attn.out_proj = LoraLinear(layer.encoder_attn.out_proj, r=lora_r, alpha=lora_alpha)

        # Diagnostic hook handle placeholder (populated by attach_cross_attn_diagnostic_hook)
        self._cross_attn_hook_handle = None
        self._cross_attn_capture: dict[str, torch.Tensor] = {}

        # Log detailed parameter budget breakdown stats
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = trainable_params + frozen_params

        breakdown = {}
        if self.use_modality_encoders:
            breakdown["modality_input_encoders"] = sum(
                p.numel() for m in [self.pose_enc, self.face_enc, self.left_hand_enc, self.right_hand_enc, self.concat_proj]
                for p in m.parameters() if p.requires_grad
            )
        else:
            breakdown["linear_in"] = sum(p.numel() for p in self.linear_in.parameters() if p.requires_grad)

        breakdown["positional_encoding"] = sum(p.numel() for p in self.pos_encoder.parameters() if p.requires_grad)
        breakdown["transformer_encoder"] = sum(p.numel() for p in self.visual_encoder.parameters() if p.requires_grad)
        breakdown["vl_mapper"] = sum(p.numel() for p in self.vl_mapper.parameters() if p.requires_grad)

        lora_params = 0
        cross_attn_params = 0
        for layer in self.mbart.model.decoder.layers:
            for name, proj in [("q_proj", layer.encoder_attn.q_proj),
                               ("k_proj", layer.encoder_attn.k_proj),
                               ("v_proj", layer.encoder_attn.v_proj),
                               ("out_proj", layer.encoder_attn.out_proj)]:
                if isinstance(proj, LoraLinear):
                    lora_params += sum(p.numel() for p in [proj.lora_A, proj.lora_B] if p.requires_grad)
                else:
                    cross_attn_params += sum(p.numel() for p in proj.parameters() if p.requires_grad)

        if lora_params > 0:
            breakdown["lora_adapters"] = lora_params
        if cross_attn_params > 0:
            breakdown["cross_attn_unfrozen"] = cross_attn_params

        if self.ctc_head is not None:
            breakdown["ctc_head"] = sum(p.numel() for p in self.ctc_head.parameters() if p.requires_grad)

        mbart_other = sum(p.numel() for p in self.mbart.parameters() if p.requires_grad)
        if cross_attn_params > 0:
            mbart_other -= cross_attn_params
        if lora_params > 0:
            mbart_other -= lora_params
        if mbart_other > 0:
            breakdown["mbart_other"] = mbart_other

        # Display the breakdown table
        print("\n" + "=" * 55)
        print("         TRAINABLE PARAMETERS MODULE BREAKDOWN")
        print("=" * 55)
        for mod, num in breakdown.items():
            print(f"  {mod:<30}: {num:>15,}")
        print("-" * 55)
        print(f"  Total Trainable Params        : {trainable_params:>15,}")
        print(f"  Total Frozen Params           : {frozen_params:>15,}")
        print(f"  Total Model Params            : {total_params:>15,}")
        print("=" * 55 + "\n")

    def _project_inputs(self, landmarks: torch.Tensor) -> torch.Tensor:
        if self.use_modality_encoders:
            # landmarks shape: (Batch, SeqLen, 1665)
            # Pose coordinates (0:132) + Pose mask (1662)
            pose_coords = landmarks[:, :, 0:132]
            pose_mask = landmarks[:, :, 1662:1663]
            pose_in = torch.cat([pose_coords, pose_mask], dim=-1) # (B, T, 133)

            # Face subset coordinates (120 dims)
            face_in = landmarks[:, :, self.face_flat_indices]

            # Left Hand coordinates (1536:1599) + Left Hand mask (1663)
            lh_coords = landmarks[:, :, 1536:1599]
            lh_mask = landmarks[:, :, 1663:1664]
            lh_in = torch.cat([lh_coords, lh_mask], dim=-1) # (B, T, 64)

            # Right Hand coordinates (1599:1662) + Right Hand mask (1664)
            rh_coords = landmarks[:, :, 1599:1662]
            rh_mask = landmarks[:, :, 1664:1665]
            rh_in = torch.cat([rh_coords, rh_mask], dim=-1) # (B, T, 64)

            # Apply individual projection encoders with relu non-linearities
            pose_feat = torch.relu(self.pose_enc(pose_in))
            face_feat = torch.relu(self.face_enc(face_in))
            lh_feat = torch.relu(self.left_hand_enc(lh_in))
            rh_feat = torch.relu(self.right_hand_enc(rh_in))

            # Concatenate and run final linear projection
            concat_feat = torch.cat([pose_feat, face_feat, lh_feat, rh_feat], dim=-1)
            x = self.concat_proj(concat_feat)
        else:
            x = self.linear_in(landmarks)
        return x

    def forward(
        self,
        landmarks: torch.Tensor,
        landmark_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor
    ) -> Any:
        """
        Forward pass during training.
        """
        x = self._project_inputs(landmarks)
        x = self.pos_encoder(x)

        padding_mask = (landmark_mask == 0.0)
        visual_features = self.visual_encoder(x, src_key_padding_mask=padding_mask)

        encoder_outputs = self.vl_mapper(visual_features)

        encoder_mask = landmark_mask.long() if landmark_mask.dtype != torch.long else landmark_mask
        outputs = self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
        )

        if self.ctc_head is not None:
            outputs["ctc_logits"] = self.ctc_head(visual_features)

        return outputs

    def generate(
        self,
        landmarks: torch.Tensor,
        landmark_mask: torch.Tensor,
        tokenizer: Any,
        num_beams: int = 4,
        max_length: int = 128,
        **kwargs
    ) -> torch.Tensor:
        """
        Generates translated sentence token IDs using Beam Search.
        """
        x = self._project_inputs(landmarks)
        x = self.pos_encoder(x)

        padding_mask = (landmark_mask == 0.0)
        visual_features = self.visual_encoder(x, src_key_padding_mask=padding_mask)

        encoder_outputs = self.vl_mapper(visual_features)

        encoder_mask = landmark_mask.long() if landmark_mask.dtype != torch.long else landmark_mask
        model_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        generated_ids = self.mbart.generate(
            encoder_outputs=model_outputs,
            attention_mask=encoder_mask,
            num_beams=num_beams,
            max_length=max_length,
            decoder_start_token_id=tokenizer.lang_code_to_id.get(getattr(tokenizer, "tgt_lang", "en_XX")),
            **kwargs
        )
        return generated_ids

    # ------------------------------------------------------------------
    # Diagnostic #1: per-module gradient norms.
    # Call this AFTER loss.backward() and AFTER optimizer.unscale_() so
    # gradients are on the true (unscaled) scale, and BEFORE clip_grad_norm_.
    # ------------------------------------------------------------------
    def compute_module_grad_norms(self) -> dict[str, float]:
        """
        Returns the L2 gradient norm for three functional groups:
          - "visual_enc": the nn.TransformerEncoder reading landmark sequences
          - "vl_mapper":  the visual-to-language bridge (MLP)
          - "cross_attn": the mBART decoder's encoder_attn projections

        A near-zero "cross_attn" norm with healthy "visual_enc"/"vl_mapper"
        norms indicates the decoder's cross-attention isn't being adapted to
        use the visual features.
        """
        def _grad_l2_norm(params) -> float:
            total_sq = 0.0
            for p in params:
                if p is not None and p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
            return total_sq ** 0.5

        visual_enc_norm = _grad_l2_norm(self.visual_encoder.parameters())
        vl_mapper_norm = _grad_l2_norm(self.vl_mapper.parameters())

        cross_attn_params = []
        for layer in self.mbart.model.decoder.layers:
            for proj in [layer.encoder_attn.q_proj, layer.encoder_attn.k_proj,
                         layer.encoder_attn.v_proj, layer.encoder_attn.out_proj]:
                if isinstance(proj, LoraLinear):
                    cross_attn_params.extend([proj.lora_A, proj.lora_B])
                else:
                    cross_attn_params.extend(list(proj.parameters()))
        cross_attn_norm = _grad_l2_norm(cross_attn_params)

        return {
            "visual_enc": visual_enc_norm,
            "vl_mapper": vl_mapper_norm,
            "cross_attn": cross_attn_norm,
        }

    # ------------------------------------------------------------------
    # Diagnostic #2: cross-attention weight entropy probe.
    # ------------------------------------------------------------------
    def attach_cross_attn_diagnostic_hook(self):
        """Registers a forward_hook on layers[0].encoder_attn to capture attn weights."""
        self.detach_cross_attn_diagnostic_hook()
        self._cross_attn_capture = {}

        target_module = self.mbart.model.decoder.layers[0].encoder_attn

        def _capture_hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self._cross_attn_capture["attn_weights"] = output[1].detach()

        self._cross_attn_hook_handle = target_module.register_forward_hook(_capture_hook)
        return self._cross_attn_hook_handle

    def detach_cross_attn_diagnostic_hook(self) -> None:
        """Removes the diagnostic hook if one is currently registered."""
        if self._cross_attn_hook_handle is not None:
            self._cross_attn_hook_handle.remove()
            self._cross_attn_hook_handle = None

    @torch.no_grad()
    def compute_cross_attn_entropy(
        self,
        landmarks: torch.Tensor,
        landmark_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
    ) -> dict[str, float] | None:
        """
        Runs a diagnostic forward pass with output_attentions=True so the
        registered hook can capture layer-0 cross-attention weights, then
        returns their mean entropy.

        Entropy close to log(src_len) => near-uniform => decoder is ignoring
        the visual context. Entropy well below log(src_len) => peaked =>
        decoder is attending to specific visual tokens.
        """
        x = self._project_inputs(landmarks)
        x = self.pos_encoder(x)

        padding_mask = (landmark_mask == 0.0)
        visual_features = self.visual_encoder(x, src_key_padding_mask=padding_mask)

        encoder_outputs = self.vl_mapper(visual_features)
        encoder_mask = landmark_mask.long() if landmark_mask.dtype != torch.long else landmark_mask

        self._cross_attn_capture = {}
        self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs),
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_attentions=True,
            return_dict=True,
        )

        attn_weights = self._cross_attn_capture.get("attn_weights", None)
        if attn_weights is None:
            return None

        # attn_weights: [batch, heads, tgt_len, src_len]
        eps = 1e-12
        probs = attn_weights.clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=-1)
        mean_entropy = entropy.mean().item()

        src_len = attn_weights.size(-1)
        max_entropy = math.log(src_len) if src_len > 1 else 0.0
        normalized_entropy = (mean_entropy / max_entropy) if max_entropy > 0 else 0.0

        return {
            "mean_entropy": mean_entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized_entropy,
        }
