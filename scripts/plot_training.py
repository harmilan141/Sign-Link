"""
scripts/plot_training.py

Plots Training Loss vs Epoch and Validation BLEU-4 vs Epoch from log files
in logs/. Supports JSONL logs (one JSON object per line, e.g.
{"epoch": 3, "train_loss": 1.42, "val_bleu4": 12.7}) or a CSV with the same
columns. Missing metrics per-line are simply skipped for that curve.

Usage:
    python scripts/plot_training.py --logs-dir logs --out logs/training_curves.png
"""

import argparse
import csv
import json
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt


def load_records(logs_dir: Path) -> list[dict]:
    records = []
    for path in sorted(logs_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    for path in sorted(logs_dir.glob("*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                records.append({k: (float(v) if v not in (None, "") else None)
                                 for k, v in row.items()})
    if not records:
        raise FileNotFoundError(f"No .jsonl or .csv log files found in {logs_dir}")
    return records


def extract_series(records: list[dict], key: str) -> tuple[list[float], list[float]]:
    epochs, values = [], []
    for r in records:
        if r.get("epoch") is not None and r.get(key) is not None:
            epochs.append(r["epoch"])
            values.append(r[key])
    # sort by epoch in case log lines arrived out of order
    pairs = sorted(zip(epochs, values))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="logs", type=Path)
    parser.add_argument("--out", default="logs/training_curves.png", type=Path)
    args = parser.parse_args()

    records = load_records(args.logs_dir)
    loss_epochs, loss_vals = extract_series(records, "train_loss")
    bleu_epochs, bleu_vals = extract_series(records, "val_bleu4")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(loss_epochs, loss_vals, color="tab:red", marker="o", markersize=3)
    ax1.set_title("Training Loss vs Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(bleu_epochs, bleu_vals, color="tab:blue", marker="o", markersize=3)
    ax2.set_title("Validation BLEU-4 vs Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BLEU-4")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}  "
          f"({len(loss_epochs)} loss points, {len(bleu_epochs)} BLEU points)")
    plt.show()


if __name__ == "__main__":
    main()
