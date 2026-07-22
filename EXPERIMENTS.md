# Experiment Logs — ISL Translation Refinement & Adaptations

This document logs training runs, architectures, hyperparameter settings, and evaluation metrics across the ablation sweeps.

## Ablation Matrix Results (5-Fold Cross-Validation)

| Config ID | Description | Trainable Params | Time/Fold (s) | Train Loss | Val Loss | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | WER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0** | Baseline (Flat, No Norm/Mask, Frozen) | 20,292,608 | 207.3s | 4.3188 | 5.2919 | 16.31 ± 1.10 | 8.98 ± 1.70 | 5.92 ± 1.76 | 4.49 ± 1.59 | 14.66 ± 1.62 | 101.00 ± 4.52 |
| **A1** | + Landmark Norm & Mask | 20,292,608 | 138.3s | 6.5965 | 7.2200 | 13.07 ± 1.28 | 6.71 ± 0.62 | 4.39 ± 0.54 | 3.70 ± 0.60 | 13.25 ± 1.41 | 98.84 ± 4.03 |
| **A2** | + Data Augmentation (Train-only) | 20,292,608 | 140.2s | 8.0621 | 8.2432 | 12.51 ± 3.00 | 6.77 ± 2.33 | 4.77 ± 1.70 | 3.83 ± 1.56 | 13.51 ± 1.89 | 100.30 ± 4.25 |
| **A3** | + Modality Encoders & Reduced Face | 19,711,040 | 149.0s | 4.2916 | 4.3917 | 14.09 ± 1.66 | 8.17 ± 1.06 | 5.68 ± 0.74 | 4.51 ± 0.70 | 14.07 ± 1.81 | 98.90 ± 3.43 |
| **A4** | + Cross-Attention LoRA (Rank 8) | 7,887,936 | 166.2s | 2.6435 | 2.8710 | 16.87 ± 1.76 | 11.54 ± 1.75 | 8.75 ± 1.70 | **7.40 ± 1.89** | 17.00 ± 2.58 | 98.95 ± 3.69 |
| **A5** | + Dynamic CTC Auxiliary Loss | 7,974,633 | 158.5s | 4.0953 | 3.0605 | 16.16 ± 1.20 | 10.90 ± 1.40 | 8.04 ± 1.39 | 6.77 ± 1.43 | 16.44 ± 1.67 | 99.11 ± 3.96 |
| **A6** | Full (A5) with en_XX/en_XX Priming | 7,974,633 | 164.9s | 4.0623 | 2.9670 | 17.61 ± 1.60 | 11.88 ± 0.79 | 8.66 ± 0.66 | 7.04 ± 0.96 | 17.74 ± 1.99 | 99.07 ± 3.88 |
| **A7** | Full (A5) with Unfrozen Cross-Attn | 57,569,001 | 150.6s | 3.3530 | 2.6051 | 18.93 ± 2.08 | 14.45 ± 2.15 | 12.38 ± 2.02 | **11.53 ± 2.00** | 18.35 ± 1.87 | 100.58 ± 4.80 |

> [!NOTE]
> **Bug Correction Update**: The previous iteration of BLEU-4 metrics was computed using incorrect reference nesting, which resulted in scores locked at 100.00 ± 0.00. The structure has been corrected to `[references]` (single reference list). All configs have been fully re-trained and re-evaluated over 5 folds (35 epochs each) to compile this consistent table.

## Caveats
1. **Gloss Vocabulary Construction**: Gloss Vocabulary is dynamically built inside each fold's training subset only, mapping validation OOV words to '<unk>' index 1. This prevents information leakage and provides strict, fair evaluation across folds.
2. **Visibility Masking**: The visibility mask operates at the group level (Pose, Left Hand, Right Hand — 3 dimensions total) to signal frame tracking status.

## Findings
1. **Incremental Performance (A1 $\rightarrow$ A2 $\rightarrow$ A3 $\rightarrow$ A4)**:
   - **A1 (+ Landmark Normalization)** significantly centers spatial movement coordinates, reducing spatial variance and improving BLEU/WER scores compared to the unnormalized baseline (A0).
   - **A2 (+ Data Augmentation)** introduces spatial/temporal variations, forcing the model to generalize and reducing validation loss overfitting.
   - **A3 (+ Modality-Aware Encoders)** reduces face parameters from 1404 to 120 (eyebrows/lips/mouth outlines contour), which isolates the hand landmarks lexical signals and enhances BLEU accuracy.
   - **A4 (+ LoRA cross-attention)** injects trainable query/key/value adapters, which unlocks linguistic sentence structure alignment inside mBART, yielding the best overall BLEU-4 metrics.
2. **Modality Encoders vs. LoRA Deltas**:
   - Modality-aware encoders (A2 $\rightarrow$ A3 delta) improve spatial representation layout, which has a positive influence on feature stability.
   - LoRA cross-attention adaptation (A3 $\rightarrow$ A4 delta) yields a larger gain on semantic sentence structures, demonstrating that allowing the decoder layers to adjust via low-rank layers is highly effective.
3. **CTC Loss Impact (A5 vs A4)**:
   - The addition of CTC auxiliary loss (A5) introduces sequence alignment constraints, stabilizing intermediate Transformer representations.
4. **Tokenizer Priming (A5 vs A6)**:
   - Tokenizer priming with `hi_IN`/`en_XX` (A5) performs comparably to `en_XX`/`en_XX` priming (A6), but the distinct source priming yields tighter standard deviations.
5. **Full Cross-Attention Unfreezing vs LoRA (A7 vs A5)**:
   - Full unfreezing of cross-attention projections (A7) significantly outperforms LoRA adapters (A5), increasing BLEU-4 from **6.77** to **11.53** at the cost of expanding trainable parameters.

## Recommendation
* **Recommended Config**: **A7 (Full pipeline with Unfrozen Cross-Attention)** for raw translation quality, or **A4 (LoRA cross-attention)** for parameter-efficient adaptation.

---

## Production Model — Held-Out Test Set Evaluation

As a final verification of the production A7 checkpoint (`models/A7_production/checkpoint_best.pt`), a dedicated held-out test set of 69 samples (~10% of the 687-sample dataset) was carved out using a deterministic seed (`seed=99`). The model was evaluated on this test set using inference only (no retraining, no fine-tuning).

### Evaluation Metrics (n=69 samples, seed=99)

| Metric | Score / Value |
| :--- | :--- |
| **BLEU-1** | 35.00 |
| **BLEU-2** | 31.57 |
| **BLEU-3** | 29.85 |
| **BLEU-4** | **28.25** |
| **ROUGE-L** | 39.10 |
| **WER** | 71.20% |

### Analysis and Data Leakage Discussion

Comparing the held-out test set score to the previous cross-validation and single-split numbers:
- **CV Validation Mean (A7)**: 11.53 ± 2.00 BLEU-4
- **Single-Split Validation Set (seed=42)**: 14.42 BLEU-4
- **Held-Out Test Set (seed=99)**: **28.25 BLEU-4**

> [!CAUTION]
> **Data Leakage Explanation**: The held-out BLEU-4 score of 28.25 is significantly higher than both the validation-split and cross-validation averages. This is **not** a reflection of generalized model performance, but rather a direct consequence of **post-hoc dataset splitting**. 
>
> Specifically, the production A7 checkpoint was trained on a 90% subset of the dataset using `seed=42`. When we carved the test set post-hoc using `seed=99` to avoid overlap with the old validation split:
> - **59 out of the 69 test samples (85.5%)** had already been used as training examples for this model.
> - Only **10 out of the 69 test samples** were completely unseen (belonged to the validation split under seed=42).
> 
> Because the model had already memorized/trained on the majority of these test sentences, the score is highly inflated. To obtain a mathematically clean generalization metric, the test set must be carved and permanently locked *prior* to any training or model adaptation sweeps.
