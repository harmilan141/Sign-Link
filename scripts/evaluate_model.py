"""
scripts/evaluate_model.py

Evaluate the best checkpoint on the validation set split and print
BLEU-1/2/3/4, ROUGE-L, and WER in a single table.

Usage:
    python scripts/evaluate_model.py \
        --config configs/A8_lora_optimized.yaml \
        --checkpoint models/A8_lora_optimized/checkpoint_best.pt
"""

import argparse
import sys
import yaml
from pathlib import Path
import numpy as np

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import random_split
# pyrefly: ignore [missing-import]
from transformers import MBart50TokenizerFast
# pyrefly: ignore [missing-import]
from sacrebleu.metrics import BLEU
# pyrefly: ignore [missing-import]
from rouge_score import rouge_scorer
# pyrefly: ignore [missing-import]
import jiwer

# Ensure dataset/dataloader can load submodules correctly
scripts_dir = Path(__file__).parent.resolve()
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

from dataset import SignLanguageDataset
from dataloader import get_dataloader
from model import SignTranslationModel


def compute_bleu_n(hypotheses, references, n: int) -> float:
    bleu = BLEU(max_ngram_order=n, effective_order=True)
    return bleu.corpus_score(hypotheses, [references]).score


def compute_rouge_l(hypotheses, references) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    return 100.0 * sum(scores) / len(scores) if scores else 0.0


def compute_wer(hypotheses, references) -> float:
    clean_refs = [r if r.strip() else "<empty>" for r in references]
    clean_hyps = [h if h.strip() else "<empty>" for h in hypotheses]
    return 100.0 * jiwer.wer(clean_refs, clean_hyps)


def compute_all_metrics(hypotheses, references) -> dict:
    return {
        "BLEU-1": compute_bleu_n(hypotheses, references, 1),
        "BLEU-2": compute_bleu_n(hypotheses, references, 2),
        "BLEU-3": compute_bleu_n(hypotheses, references, 3),
        "BLEU-4": compute_bleu_n(hypotheses, references, 4),
        "ROUGE-L": compute_rouge_l(hypotheses, references),
        "WER": compute_wer(hypotheses, references),
    }


def print_metrics_table(metrics: dict, n_samples: int, title: str = "Validation Results"):
    rows = [(k, f"{v:.2f}") for k, v in metrics.items()]
    label_w = max(len(r[0]) for r in rows) + 2
    value_w = max(len(r[1]) for r in rows) + 2
    width = label_w + value_w + 3

    print()
    print(f" {title}  (n={n_samples})")
    print("=" * width)
    print(f"| {'Metric'.ljust(label_w - 1)}| {'Score'.ljust(value_w - 1)}|")
    print("-" * width)
    for name, val in rows:
        note = "  (lower is better)" if name == "WER" else ""
        print(f"| {name.ljust(label_w - 1)}| {val.ljust(value_w - 1)}|{note}")
    print("=" * width)
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sign Language Translation Checkpoint.")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to best checkpoint file (.pt)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    # Load YAML Configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(
        config.get("model_name", "facebook/mbart-large-50"),
        src_lang=config.get("src_lang", "hi_IN"),
        tgt_lang=config.get("tgt_lang", "en_XX")
    )

    # 2. Load Dataset
    dataset = SignLanguageDataset(
        config["npz_dir"],
        tokenizer,
        use_normalization=config.get("use_normalization", True),
        use_masking=config.get("use_masking", True)
    )

    # 3. Create Validation Split
    val_split = config.get("val_split", 0.1)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    _, val_dataset = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        num_workers=config.get("num_workers", 4)
    )

    # 4. Initialize Model and Load Checkpoint
    sample_item = val_dataset[0]
    input_dim = sample_item["landmarks"].shape[-1]

    model = SignTranslationModel(
        input_dim=input_dim,
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        mbart_model_name=config.get("model_name", "facebook/mbart-large-50"),
        use_modality_encoders=config.get("use_modality_encoders", False),
        decoder_adapt_mode=config.get("decoder_adapt_mode", "frozen"),
        lora_r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16.0),
        use_auxiliary_ctc=config.get("use_auxiliary_ctc", False),
        gloss_vocab_size=None
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 5. Inference loop
    hypotheses, references = [], []
    print(f"Running inference on {len(val_dataset)} validation samples...")

    with torch.no_grad():
        for batch in val_loader:
            landmarks = batch["landmarks"].to(device)
            landmark_mask = batch["landmark_mask"].to(device)

            generated_ids = model.generate(
                landmarks=landmarks,
                landmark_mask=landmark_mask,
                tokenizer=tokenizer,
                num_beams=config.get("regularization", {}).get("generation", {}).get("num_beams", 4),
                max_length=config.get("regularization", {}).get("generation", {}).get("max_new_tokens", 128)
            )

            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            hypotheses.extend(h.strip() for h in decoded)
            references.extend(t.strip() for t in batch["sentences"])

    metrics = compute_all_metrics(hypotheses, references)
    print_metrics_table(metrics, n_samples=len(val_dataset), title=f"Eval: {Path(args.checkpoint).name}")


if __name__ == "__main__":
    main()
