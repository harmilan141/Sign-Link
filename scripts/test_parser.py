import numpy as np

log_file_path = "/workspace/SignLink/logs/A0/sweep.log"
results = {}
k_folds = 5

with open(log_file_path, "r", encoding="utf-8") as lf:
    log_content = lf.read()
    
# Parse final cross-validation statistics averages
for line in log_content.splitlines():
    line = line.strip()
    if "Fold" in line or "metrics" in line:
        continue
        
    if "BLEU_1:" in line:
        parts = line.split("BLEU_1:")[-1].strip().split()
        print("BLEU_1 parts:", parts)
        results["bleu_1_mean"] = float(parts[0])
        results["bleu_1_std"] = float(parts[2])
    elif "BLEU_2:" in line:
        parts = line.split("BLEU_2:")[-1].strip().split()
        print("BLEU_2 parts:", parts)
        results["bleu_2_mean"] = float(parts[0])
        results["bleu_2_std"] = float(parts[2])
    elif "BLEU_3:" in line:
        parts = line.split("BLEU_3:")[-1].strip().split()
        print("BLEU_3 parts:", parts)
        results["bleu_3_mean"] = float(parts[0])
        results["bleu_3_std"] = float(parts[2])
    elif "BLEU_4:" in line:
        parts = line.split("BLEU_4:")[-1].strip().split()
        print("BLEU_4 parts:", parts)
        results["bleu_4_mean"] = float(parts[0])
        results["bleu_4_std"] = float(parts[2])
    elif "ROUGE_L:" in line:
        parts = line.split("ROUGE_L:")[-1].strip().split()
        print("ROUGE_L parts:", parts)
        results["rouge_l_mean"] = float(parts[0])
        results["rouge_l_std"] = float(parts[2])
    elif "WER:" in line:
        parts = line.split("WER:")[-1].strip().split()
        print("WER parts:", parts)
        results["wer_mean"] = float(parts[0])
        results["wer_std"] = float(parts[2])
print("Successfully parsed summary metrics!")
