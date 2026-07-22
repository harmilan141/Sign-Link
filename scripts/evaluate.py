"""
evaluate.py
===========

Evaluation script for Sign Language Translation model.
Loads the best saved checkpoint, evaluates on the validation split,
computes BLEU, ROUGE-L, and chrF metrics, and saves prediction outputs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import random_split
from transformers import MBart50TokenizerFast

from dataset import SignLanguageDataset
from dataloader import get_dataloader
from model import SignTranslationModel

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluate")


def compute_metrics(predictions, references):
    """
    Computes BLEU, ROUGE-L, and chrF scores.
    """
    metrics = {}
    
    # 1. BLEU Score
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        ref_tokens = [[ref.lower().split()] for ref in references]
        pred_tokens = [pred.lower().split() for pred in predictions]
        chencherry = SmoothingFunction()
        metrics["bleu"] = corpus_bleu(ref_tokens, pred_tokens, smoothing_function=chencherry.method1) * 100
    except Exception as e:
        logger.error("Error computing BLEU: %s", e)
        metrics["bleu"] = 0.0

    # 2. ROUGE-L Score
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
        metrics["rougeL"] = np.mean(rouge_scores) * 100
    except Exception as e:
        logger.error("Error computing ROUGE-L: %s", e)
        metrics["rougeL"] = 0.0

    # 3. chrF Score
    try:
        from nltk.translate.chrf_score import corpus_chrf
        ref_list = [[ref] for ref in references]
        metrics["chrf"] = corpus_chrf(ref_list, predictions) * 100
    except Exception as e:
        logger.error("Error computing chrF: %s", e)
        metrics["chrf"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sign Language Translation model.")
    parser.add_argument("--npz-dir", type=Path, default=Path("/workspace/SignLink/preprocessed"), help="Path to preprocessed .npz directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/SignLink/models/checkpoint_best.pt"), help="Path to best checkpoint file")
    parser.add_argument("--output-json", type=Path, default=Path("/workspace/SignLink/predictions.json"), help="Path to save predictions JSON")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--tokenizer-name", type=str, default="facebook/mbart-large-50", help="Pretrained tokenizer name")
    parser.add_argument("--model-name", type=str, default="facebook/mbart-large-50", help="Pretrained mbart model name")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation set split ratio (must match training split)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1. Load Tokenizer
    logger.info("Loading tokenizer %s", args.tokenizer_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(args.tokenizer_name, src_lang="hi_IN", tgt_lang="en_XX")

    # 2. Load Dataset (validation split using exact same seed 42)
    dataset = SignLanguageDataset(args.npz_dir, tokenizer)
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    _, val_dataset = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    logger.info("Validation dataset size: %d", len(val_dataset))
    
    val_loader = get_dataloader(val_dataset, args.batch_size, tokenizer.pad_token_id, shuffle=False, num_workers=2)

    # 3. Load Checkpoint
    if not args.checkpoint.is_file():
        logger.error("Checkpoint file not found: %s", args.checkpoint)
        sys.exit(1)
        
    logger.info("Loading model checkpoint from %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Auto-detect input landmark dimension from dataset
    input_dim = dataset[0]["landmarks"].shape[1]
    logger.info("Input landmark dim: %d", input_dim)

    # Initialize model and load state dict
    model = SignTranslationModel(input_dim=input_dim, mbart_model_name=args.model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    predictions = []
    references = []
    video_names = []
    results = []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in val_loader:
            landmarks = batch["landmarks"].to(device)
            landmark_mask = batch["landmark_mask"].to(device)

            # Generate target translation
            generated_ids = model.generate(
                landmarks=landmarks,
                landmark_mask=landmark_mask,
                tokenizer=tokenizer,
                num_beams=4,
                max_length=128
            )

            # Decode token IDs
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            predictions.extend(decoded_preds)
            references.extend(batch["sentences"])
            video_names.extend(batch["video_names"])

    # 4. Compute Metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(predictions, references)
    logger.info("=" * 40)
    logger.info("Evaluation Metrics:")
    logger.info(f"BLEU-4:  {metrics['bleu']:.2f}")
    logger.info(f"ROUGE-L: {metrics['rougeL']:.2f}")
    logger.info(f"chrF:    {metrics['chrf']:.2f}")
    logger.info("=" * 40)

    # Assemble prediction results for file saving
    for vid, ref, pred in zip(video_names, references, predictions):
        results.append({
            "video_name": vid,
            "reference": ref,
            "prediction": pred
        })

    # Save to file
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "predictions": results
        }, f, indent=4)
        
    logger.info("Saved predictions and metrics to %s", args.output_json)


if __name__ == "__main__":
    main()
