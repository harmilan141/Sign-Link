# Sign Link: Project Development Summary & Chronological Log

This document provides a complete, high-level technical summary of all engineering work, model architecture improvements, bug fixes, training sweeps, and deployment integrations completed for **Sign Link — AI-Powered Indian Sign Language Translator**.

---

## 📅 Chronological Phase Log & Architecture Evolution

### Phase 1: Input Preprocessing & Invariance (Task 1)
* **Goal**: Establish spatial-temporal invariance to eliminate background changes, camera distances, and signer positions.
* **Implementation**:
  * **Shoulder Normalization**: Shifted and centered frame coordinate origins relative to the midpoint of the shoulders. Normalized coordinate scales using the distance from the shoulder to the hip midpoint.
  * **Visibility Occlusion Masking**: Added binary visibility masks (+3 dimensions) mapping the tracking status of the Pose, Left Hand, and Right Hand joints to make the model robust against visual occlusions.
  * **Input Feature Dimensions**: Scaled the feature coordinates vector from 1662-dim to **1665-dim** per frame.

### Phase 2: Data Generalization & Augmentations (Task 2)
* **Goal**: Prevent overfitting on the ISL-CSLTR dataset (687 videos) by adding visual variations.
* **Implementation**:
  * **Spatial Transforms**: Random rotation (tilt), random scaling (zoom), and noise coordinate jittering.
  * **Temporal Warpings**: Respeeding sequences from `0.8x` to `1.2x` using linear interpolation, and random frame dropouts.
  * **Symmetric Hand Mirroring**: Joint-wise X-coordinate negations to double the hand dataset size and simulate left-handed signers.

### Phase 3: Modality-Aware Encoders & Reduced Face Mesh (Task 3)
* **Goal**: Focus representation capacities on lexical sign actions and reduce coordinate noise.
* **Implementation**:
  * **Reduced Face Mesh**: Selected a curated subset of 40 expression keypoints (lips, eyebrows, outlines) instead of MediaPipe's full 468-point face mesh, reducing coordinates and filtering out face shape variations.
  * **Separate Linear Encoders**: Split visual inputs into distinct modality encoders: Pose (128-dim), Hands (128-dim), and Face (64-dim) to prevent visual feature dilution.

### Phase 4: Parameter-Efficient Fine-Tuning via LoRA (Task 4)
* **Goal**: Adapt the multilingual language model (mBART-50) without training 611 million parameters.
* **Implementation**:
  * **Cross-Attention LoRA**: Integrated Low-Rank projection adapters ($r=8$, $\alpha=16$) inside query, key, value, and output branches of the decoder's cross-attention blocks.
  * **Parameter Reduction**: Frozen the mBART model and trained only the visual projections and LoRA branches, reducing trainable parameters to just **1.29% (7.8M parameters)**.

### Phase 5: Sequence Alignment & Multi-Task Loss (Task 5)
* **Goal**: Improve alignment between long sign frames and short target sentences to prevent repetition loops.
* **Implementation**:
  * **Auxiliary CTC Head**: Added a linear projection mapping intermediate Transformer states to gloss labels, supervised using Connectionist Temporal Classification (CTC) loss.
  * **Multi-Task Objective**: Combined sequence-to-sequence loss and CTC loss:
    $$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Cross-Entropy}} + 0.1 \times \mathcal{L}_{\text{CTC}}$$

### Phase 6: Language Priming & Leakage Prevention (Task 6 & 7)
* **Goal**: Enhance translation grammar and prevent training-to-validation leaks.
* **Implementation**:
  * **Target Token Priming**: Set the mBART tokenizer source code to `hi_IN` and target to `en_XX`, improving decoder sentence generation.
  * **Dynamic Gloss Vocabulary (A.6)**: Modified validation fold compilation to build vocabulary maps dynamically from the training split only, mapping validation OOV terms to `<unk>` index `1`.

### Phase 7: Server Optimizations & Sweep Matrix Runs
* **Goal**: Resolve memory limitations and execute the 35-epoch ablation sweeps.
* **Implementation**:
  * **Disk Cleanups**: Removed old PyTorch checkpoints and disabled sweep checkpoint writes (`save_checkpoints: False`), recovering **220 GB** of disk space on the NFS share.
  * **Offline Models**: Implemented `local_files_only=True` to load HuggingFace models strictly from cache, resolving internet connection dropouts on the server.
  * **Sweep Run**: Successfully trained all configurations **A0** to **A7** across 5 folds and 35 epochs, documenting the final table in `EXPERIMENTS.md`.

---

## 📊 Final Sweep Results Table (5-Fold Cross-Validation)

| Config ID | Description | Trainable Params | Time/Fold | Train Loss | Val Loss | BLEU-4 | WER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0** | Baseline (Flat, No Norm/Mask, Frozen) | 20,292,608 | 414.1s | 4.9379 | 5.9940 | 53.22 ± 38.34 | 136.25 ± 70.53 |
| **A1** | + Landmark Norm & Mask | 20,292,608 | 220.3s | 5.7614 | 6.4732 | **100.00 ± 0.00** | **102.43 ± 1.71** |
| **A2** | + Data Augmentation (Train-only) | 20,292,608 | 159.1s | 6.6572 | 6.6764 | 76.73 ± 35.02 | 116.12 ± 11.60 |
| **A3** | + Modality Encoders & Reduced Face | 19,711,040 | 163.9s | 4.9352 | 5.6239 | 39.76 ± 35.10 | 99.07 ± 5.31 |
| **A4** | + Cross-Attention LoRA (Rank 8) | 7,887,936 | 166.9s | 3.2635 | 3.4901 | 100.00 ± 0.00 | 108.76 ± 16.14 |
| **A5** | **+ Dynamic CTC Auxiliary Loss** | **7,974,633** | **165.0s** | **4.2594** | **3.0753** | **100.00 ± 0.00** | **105.55 ± 8.93** |
| **A6** | Full (A5) with en_XX/en_XX Priming | 7,974,633 | 150.9s | 3.9946 | 2.9305 | 100.00 ± 0.00 | 102.58 ± 3.34 |
| **A7** | Full (A5) with Unfrozen Cross-Attn | 57,569,001 | 167.1s | 3.3106 | 2.6078 | 100.00 ± 0.00 | 104.23 ± 7.09 |

---

## 🚀 Production Deployment & API Integration
1. **Production Training**: Trained the optimal model architecture (**A5**) on the complete dataset with checkpoint writing enabled, saving the deployable model to:
   * `/workspace/SignLink/models/A5_production/checkpoint_best.pt`
2. **FastAPI Backend Integration**:
   * Updated `ai-service/main.py` with a **Dynamic Architecture Loader** that inspects the checkpoint keys directly. It automatically detects and configures the presence of modality encoders, LoRA parameters, and CTC loss structures.
   * Modified the model and tokenizer initialization methods in `main.py` to use `local_files_only=True` and offline settings, ensuring startup stability.
   * Restarted the AI service, loading the new A5 production model onto the server GPU.
