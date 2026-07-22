"""
train.py
========

Training script for Sign Language Translation on GPU remote server.
Supports config files, K-Fold cross-validation, data augmentation,
torso coordinate normalization, CTC auxiliary loss, and metrics reporting (BLEU, ROUGE, WER).
"""

from __future__ import annotations

import argparse
import logging
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys
from pathlib import Path
import numpy as np
import yaml

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torch.utils.data import Subset
from transformers import MBart50TokenizerFast, get_linear_schedule_with_warmup

# Ensure dataset/dataloader can load submodules correctly
scripts_dir = Path(__file__).parent.resolve()
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

from dataset import SignLanguageDataset
from dataloader import get_dataloader
from model import SignTranslationModel

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train")


def build_gloss_vocab_from_subset(train_subset: Subset) -> dict[str, int]:
    """
    Scans subset NPZ annotations to build a vocabulary of unique gloss words.
    Blank/padding maps to index 0, unknown/OOV maps to index 1.
    """
    vocab = {"<blank>": 0, "<unk>": 1}
    dataset = train_subset.dataset
    indices = train_subset.indices
    for idx in indices:
        file_path = dataset.files[idx]
        try:
            with np.load(file_path, allow_pickle=False) as data:
                gloss_str = str(data["gloss"]).upper().strip()
                for word in gloss_str.split():
                    if word and word not in vocab:
                        vocab[word] = len(vocab)
        except Exception:
            pass
    return vocab


def compute_validation_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """
    Computes BLEU-1/2/3/4 (sacrebleu), ROUGE-L (LCS-based), and WER (jiwer).
    """
    import sacrebleu.metrics
    import jiwer
    
    # 1. sacrebleu corpus BLEU scores
    # Refs must be inside list of lists (a list containing references for each system)
    bleu_refs = [references]
    bleu_1 = sacrebleu.metrics.BLEU(max_ngram_order=1).corpus_score(predictions, bleu_refs).score
    bleu_2 = sacrebleu.metrics.BLEU(max_ngram_order=2).corpus_score(predictions, bleu_refs).score
    bleu_3 = sacrebleu.metrics.BLEU(max_ngram_order=3).corpus_score(predictions, bleu_refs).score
    bleu_4 = sacrebleu.metrics.BLEU(max_ngram_order=4).corpus_score(predictions, bleu_refs).score
    
    # 2. Word Error Rate (WER) via jiwer
    clean_preds = [p.lower().strip() for p in predictions]
    clean_refs = [r.lower().strip() for r in references]
    
    if len(clean_refs) > 0:
        # Avoid empty strings causing divisions by zero
        clean_preds = [p if p else " " for p in clean_preds]
        clean_refs = [r if r else " " for r in clean_refs]
        wer = jiwer.wer(clean_refs, clean_preds) * 100.0
    else:
        wer = 100.0
        
    # 3. LCS-based ROUGE-L Score
    rouge_l = 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
        rouge_l = np.mean(rouge_scores) * 100.0
    except ImportError:
        def lcs(x, y):
            m, n = len(x), len(y)
            dp = [[0]*(n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
            
        f_measures = []
        for r, p in zip(clean_refs, clean_preds):
            r_tokens = r.split()
            p_tokens = p.split()
            if not r_tokens or not p_tokens:
                f_measures.append(0.0)
                continue
            lcs_len = lcs(r_tokens, p_tokens)
            prec = lcs_len / len(p_tokens)
            rec = lcs_len / len(r_tokens)
            if prec + rec > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0
            f_measures.append(f1)
        rouge_l = np.mean(f_measures) * 100.0

    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "rouge_l": rouge_l,
        "wer": wer
    }


def validate(model, dataloader, tokenizer, device, loss_fn) -> tuple[float, dict[str, float]]:
    model.eval()
    val_loss = 0.0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            landmarks = batch["landmarks"].to(device)
            landmark_mask = batch["landmark_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100  # ignore pad in loss
            
            decoder_input_ids = input_ids[:, :-1]
            decoder_labels = labels[:, 1:]
            decoder_attention_mask = attention_mask[:, :-1]

            outputs = model(
                landmarks=landmarks,
                landmark_mask=landmark_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask
            )
            logits = outputs.logits
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), decoder_labels.reshape(-1))
            val_loss += loss.item()

            generated_ids = model.generate(
                landmarks=landmarks,
                landmark_mask=landmark_mask,
                tokenizer=tokenizer,
                num_beams=4,
                max_length=128
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(decoded_preds)
            references.extend(batch["sentences"])

    avg_loss = val_loss / len(dataloader)
    metrics = compute_validation_metrics(predictions, references)
    
    # Print sample predictions for validation debug
    for pred, ref in zip(predictions[:3], references[:3]):
        logger.info(f"Ref: {ref} | Pred: {pred}")
        
    return avg_loss, metrics


def train_fold(
    fold_idx: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: dict,
    tokenizer: MBart50TokenizerFast,
    gloss_vocab: dict[str, int] | None,
    device: torch.device,
    amp_device: str
) -> dict[str, float]:
    """Trains a single model fold and returns its best validation metrics."""
    logger.info("Initializing model for Fold %d...", fold_idx + 1)
    
    # Determine input dimension dynamically from dataset features
    sample_item = train_loader.dataset[0]
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
        gloss_vocab_size=len(gloss_vocab) if gloss_vocab else None
    )
    model.to(device)

    # Set up optimizers and learning rate schedulers
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config["lr"], weight_decay=config["weight_decay"])
    
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    scaler = GradScaler(amp_device) if amp_device == "cuda" else GradScaler()
    
    # Standard Loss with Label Smoothing
    loss_ce_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=config.get("label_smoothing", 0.0))
    
    # CTC Loss definition
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True) if config.get("use_auxiliary_ctc", False) else None
    
    best_bleu_4 = 0.0
    best_metrics = {}
    save_dir = Path(config["save_dir"])
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_loader):
            landmarks = batch["landmarks"].to(device)
            landmark_mask = batch["landmark_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            decoder_input_ids = input_ids[:, :-1]
            decoder_labels = labels[:, 1:]
            decoder_attention_mask = attention_mask[:, :-1]

            optimizer.zero_grad()

            with autocast(device_type=amp_device, enabled=(amp_device == "cuda")):
                outputs = model(
                    landmarks=landmarks,
                    landmark_mask=landmark_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask
                )
                
                logits = outputs.logits
                loss_ce = loss_ce_fn(logits.reshape(-1, logits.size(-1)), decoder_labels.reshape(-1))
                
                # Compute Auxiliary CTC Loss if enabled
                if ctc_loss_fn is not None and "ctc_logits" in outputs and gloss_vocab is not None:
                    targets_list = []
                    target_lengths_list = []
                    for g_str in batch["glosses"]:
                        ids = [gloss_vocab.get(w, 1) for w in g_str.upper().strip().split()] # Map unknown words to <unk> at index 1
                        # Handle empty gloss string fallback to blank token
                        if not ids:
                            ids = [0]
                        targets_list.extend(ids)
                        target_lengths_list.append(len(ids))
                        
                    targets = torch.tensor(targets_list, dtype=torch.long, device=device)
                    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long, device=device)
                    input_lengths = batch["landmark_lengths"].to(device)
                    
                    # Logits format: (SeqLen, Batch, Vocab)
                    ctc_logits = outputs["ctc_logits"].transpose(0, 1)
                    log_probs = torch.log_softmax(ctc_logits, dim=-1)
                    
                    loss_ctc = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
                    loss = loss_ce + config.get("lambda_ctc", 0.3) * loss_ctc
                else:
                    loss = loss_ce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

            if (step + 1) % 15 == 0:
                logger.info(
                    "Fold %d | Epoch [%d/%d] Step [%d/%d] | Loss: %.4f",
                    fold_idx + 1,
                    epoch + 1,
                    config["epochs"],
                    step + 1,
                    len(train_loader),
                    loss.item()
                )

        avg_train_loss = train_loss / len(train_loader)
        
        # Validation evaluation
        val_loss, metrics = validate(model, val_loader, tokenizer, device, loss_ce_fn)
        
        logger.info(
            "Fold %d | Epoch %d metrics | Loss: %.4f | BLEU-4: %.2f | BLEU-1: %.2f | ROUGE-L: %.2f | WER: %.2f",
            fold_idx + 1,
            epoch + 1,
            val_loss,
            metrics["bleu_4"],
            metrics["bleu_1"],
            metrics["rouge_l"],
            metrics["wer"]
        )

        checkpoint = {
            "epoch": epoch,
            "fold": fold_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics
        }

        # Save Fold Checkpoints
        if config.get("save_checkpoints", True):
            latest_path = save_dir / f"checkpoint_latest_fold_{fold_idx}.pt"
            torch.save(checkpoint, latest_path)
            
            if metrics["bleu_4"] >= best_bleu_4:
                best_bleu_4 = metrics["bleu_4"]
                best_metrics = metrics.copy()
                best_path = save_dir / f"checkpoint_best_fold_{fold_idx}.pt"
                torch.save(checkpoint, best_path)
                
                # If fold 0, also overwrite default benchmark models
                if fold_idx == 0:
                    torch.save(checkpoint, save_dir / "checkpoint_best.pt")
        else:
            if metrics["bleu_4"] >= best_bleu_4:
                best_bleu_4 = metrics["bleu_4"]
                best_metrics = metrics.copy()

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Enhanced Sign Language Translation Training.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"), help="Path to config YAML file")
    args = parser.parse_args()

    # Load YAML Configuration
    config_path = args.config
    if not config_path.is_file():
        # Fallback to local configs path check
        local_path = Path(__file__).parent / "configs" / "experiment.yaml"
        if local_path.is_file():
            config_path = local_path
        else:
            logger.error("Configuration file not found at %s", config_path)
            sys.exit(1)
            
    logger.info("Loading config file: %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize Directories
    save_dir = Path(config["save_dir"])
    log_dir = Path(config["log_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # File Logging setup
    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    logger.info("Device active: %s", device)

    # Tokenizer Priming (Ablation support)
    src_lang = config.get("src_lang", "hi_IN")
    tgt_lang = config.get("tgt_lang", "en_XX")
    logger.info("Loading tokenizer with src: %s | tgt: %s", src_lang, tgt_lang)
    tokenizer = MBart50TokenizerFast.from_pretrained(
        config.get("model_name", "facebook/mbart-large-50"),
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        local_files_only=True
    )

    # Load complete dataset instance
    dataset = SignLanguageDataset(
        npz_dir=config["npz_dir"],
        tokenizer=tokenizer,
        max_seq_len=512,
        max_tgt_len=128,
        augment_config=None, # Disable augmentations for static/initial dataset analysis
        use_normalization=config.get("use_normalization", True),
        use_masking=config.get("use_masking", True)
    )
    
    if len(dataset) == 0:
        logger.error("Empty preprocessed dataset. Verify preprocessed path in config.")
        return

    # Build Gloss Vocabulary for Auxiliary CTC Loss
    # We log if CTC is enabled, but compile vocabulary dynamically per training fold to prevent leakage.
    use_ctc = config.get("use_auxiliary_ctc", False)
    logger.info("ctc_active: %s", str(use_ctc).lower())

    # Cross-Validation or Simple Train/Val split execution
    if config.get("use_cross_val", False):
        k = config.get("k_folds", 5)
        logger.info("Starting K-Fold validation sweep with %d folds...", k)
        
        # Partition dataset indices deterministically
        indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        
        fold_results = []
        for fold in range(k):
            logger.info("\n" + "=" * 50)
            logger.info("RUNNING CROSS-VALIDATION FOLD %d / %d", fold + 1, k)
            logger.info("=" * 50)
            
            val_idx = folds[fold]
            train_idx = np.concatenate([folds[i] for i in range(k) if i != fold])
            
            # Setup split datasets (training gets augmentations, validation split remains clean)
            train_dataset = Subset(
                SignLanguageDataset(
                    config["npz_dir"], 
                    tokenizer, 
                    augment_config=config.get("augment_config"),
                    use_normalization=config.get("use_normalization", True),
                    use_masking=config.get("use_masking", True)
                ),
                train_idx
            )
            val_dataset = Subset(
                SignLanguageDataset(
                    config["npz_dir"], 
                    tokenizer, 
                    augment_config=None,
                    use_normalization=config.get("use_normalization", True),
                    use_masking=config.get("use_masking", True)
                ),
                val_idx
            )
            
            # Build Gloss Vocabulary for this Fold training partition only (leakage prevention)
            gloss_vocab = None
            if use_ctc:
                logger.info("Building fold-specific training gloss vocabulary...")
                gloss_vocab = build_gloss_vocab_from_subset(train_dataset)
                logger.info("Gloss vocabulary successfully built for Fold %d! Total vocabulary size: %d", fold + 1, len(gloss_vocab))
            
            train_loader = get_dataloader(train_dataset, config["batch_size"], tokenizer.pad_token_id, shuffle=True, num_workers=config["num_workers"])
            val_loader = get_dataloader(val_dataset, config["batch_size"], tokenizer.pad_token_id, shuffle=False, num_workers=config["num_workers"])
            
            best_fold_metrics = train_fold(
                fold_idx=fold,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                tokenizer=tokenizer,
                gloss_vocab=gloss_vocab,
                device=device,
                amp_device=amp_device
            )
            fold_results.append(best_fold_metrics)

        # Aggregate and report final statistics across folds
        logger.info("\n" + "=" * 60)
        logger.info("K-FOLD CROSS VALIDATION SWEEP COMPLETE (K=%d)", k)
        logger.info("=" * 60)
        for metric_name in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "wer"]:
            values = [res[metric_name] for res in fold_results]
            logger.info("  %s: %.2f ± %.2f", metric_name.upper(), np.mean(values), np.std(values))
        logger.info("=" * 60)
        
    else:
        logger.info("Starting standard single-split training run...")
        val_len = int(len(dataset) * config.get("val_split", 0.1))
        train_len = len(dataset) - val_len
        
        # Partition indices deterministically
        indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_idx = indices[:train_len]
        val_idx = indices[train_len:]
        
        train_dataset = Subset(
            SignLanguageDataset(
                config["npz_dir"], 
                tokenizer, 
                augment_config=config.get("augment_config"),
                use_normalization=config.get("use_normalization", True),
                use_masking=config.get("use_masking", True)
            ),
            train_idx
        )
        val_dataset = Subset(
            SignLanguageDataset(
                config["npz_dir"], 
                tokenizer, 
                augment_config=None,
                use_normalization=config.get("use_normalization", True),
                use_masking=config.get("use_masking", True)
            ),
            val_idx
        )
        
        # Build Gloss Vocabulary for training partition only (leakage prevention)
        gloss_vocab = None
        if use_ctc:
            logger.info("Building training split gloss vocabulary...")
            gloss_vocab = build_gloss_vocab_from_subset(train_dataset)
            logger.info("Gloss vocabulary successfully built from training split! Total vocabulary size: %d", len(gloss_vocab))
            
        train_loader = get_dataloader(train_dataset, config["batch_size"], tokenizer.pad_token_id, shuffle=True, num_workers=config["num_workers"])
        val_loader = get_dataloader(val_dataset, config["batch_size"], tokenizer.pad_token_id, shuffle=False, num_workers=config["num_workers"])
        
        best_metrics = train_fold(
            fold_idx=0,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            tokenizer=tokenizer,
            gloss_vocab=gloss_vocab,
            device=device,
            amp_device=amp_device
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        for name, value in best_metrics.items():
            logger.info("  Validation %s: %.2f", name.upper(), value)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
