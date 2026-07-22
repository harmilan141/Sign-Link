import os
import re

logs_dir = "/workspace/SignLink/logs"
configs = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]

print("=" * 60)
print("  ABLATION SWEEP PROGRESS")
print("=" * 60)

for cfg in configs:
    log_path = os.path.join(logs_dir, cfg, "sweep.log")
    if not os.path.isfile(log_path):
        print(f"  [{cfg}]  Not started yet")
        continue

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Find last epoch step line
    last_step = None
    last_epoch_metrics = None
    last_kfold_summary = None

    for line in lines:
        if "Epoch [" in line and "Step [" in line:
            last_step = line.strip()
        if "Epoch" in line and "metrics" in line and "Loss:" in line:
            last_epoch_metrics = line.strip()
        if "K-FOLD CROSS VALIDATION SWEEP COMPLETE" in line:
            last_kfold_summary = True

    if last_kfold_summary:
        # Extract summary metrics
        bleu4 = rouge = wer = "?"
        for line in lines:
            if "BLEU_4:" in line and "Fold" not in line and "metrics" not in line:
                parts = line.split("BLEU_4:")[-1].strip().split()
                if len(parts) >= 3:
                    bleu4 = f"{parts[0]} ± {parts[2]}"
            if "ROUGE_L:" in line and "Fold" not in line and "metrics" not in line:
                parts = line.split("ROUGE_L:")[-1].strip().split()
                if len(parts) >= 3:
                    rouge = f"{parts[0]} ± {parts[2]}"
            if "WER:" in line and "Fold" not in line and "metrics" not in line:
                parts = line.split("WER:")[-1].strip().split()
                if len(parts) >= 3:
                    wer = f"{parts[0]} ± {parts[2]}"
        print(f"  [{cfg}]  ✅ COMPLETE  | BLEU-4: {bleu4} | ROUGE-L: {rouge} | WER: {wer}")
    elif last_epoch_metrics:
        # Extract fold/epoch info from the line
        m_fold = re.search(r"Fold (\d+)", last_epoch_metrics)
        m_epoch = re.search(r"Epoch (\d+)", last_epoch_metrics)
        m_bleu4 = re.search(r"BLEU-4: ([\d.]+)", last_epoch_metrics)
        m_wer = re.search(r"WER: ([\d.]+)", last_epoch_metrics)
        fold = m_fold.group(1) if m_fold else "?"
        epoch = m_epoch.group(1) if m_epoch else "?"
        bleu4 = m_bleu4.group(1) if m_bleu4 else "?"
        wer = m_wer.group(1) if m_wer else "?"
        print(f"  [{cfg}]  🔄 Running   | Fold {fold}/5, Epoch {epoch}/35 | BLEU-4: {bleu4} | WER: {wer}%")
    elif last_step:
        m_fold = re.search(r"Fold (\d+)", last_step)
        m_epoch = re.search(r"Epoch \[(\d+)/", last_step)
        m_step = re.search(r"Step \[(\d+)/(\d+)\]", last_step)
        fold = m_fold.group(1) if m_fold else "?"
        epoch = m_epoch.group(1) if m_epoch else "?"
        step = f"{m_step.group(1)}/{m_step.group(2)}" if m_step else "?"
        print(f"  [{cfg}]  🔄 Training  | Fold {fold}/5, Epoch {epoch}/35, Step {step}")
    else:
        print(f"  [{cfg}]  ⏳ Loading model...")

print("=" * 60)
