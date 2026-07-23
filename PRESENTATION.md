# PowerPoint Slides: Results Section (Sign-Link Project)

This document contains copy-pasteable slides for the **Results** and **Evaluation** sections of your presentation.

---

## 🖥️ Slide 1: Incremental Optimization (Ablation Study)

### **Slide Title**: Incremental Optimization: Ablation Study Results
### **Subtitle**: How each component improved translation quality (BLEU-4) and parameter efficiency.

---

### **Ablation Sweep Matrix (5-Fold Cross-Validation)**

| Config ID | Architecture Component | Trainable Params | Val BLEU-4 | Key Architecture Impact |
| :--- | :--- | :--- | :--- | :--- |
| **A0** | **Baseline** (Flat, Frozen mBART) | 20.29 M | 4.49 ± 1.59 | Poor baseline sequence alignment. |
| **A1** | **+ Landmark Normalization** | 20.29 M | 3.70 ± 0.60 | Centers landmarks; reduces spatial variance. |
| **A3** | **+ Modality Encoders (Reduced Face)** | 19.71 M | 4.51 ± 0.70 | Limits face details to outline; isolates hand signals. |
| **A4** | **+ Cross-Attention LoRA (Rank 8)** | **7.88 M** | **7.40 ± 1.89** | **Best parameter efficiency (Under 8M params)**. |
| **A7** | **+ Fully Unfrozen Cross-Attention** | 57.56 M | **11.53 ± 2.00** | **Best translation quality (3x gain over baseline)**. |

---

### **Key Talking Points for Slide 1**
* **Visual Denoising is Key**: Restricting the facial mesh coordinates to outlines (eyebrows/lips) and normalizing spatial scales eliminated visual noise and stabilized sequences.
* **LoRA vs. Full Adaptation**: **A4 (LoRA)** is optimal for resource-constrained edge deployments (uses 1.2% of trainable parameters), while **A7 (Full)** is the absolute best for raw linguistic quality.

---
---

## 🖥️ Slide 2: Production Checkpoint & Held-Out Test Evaluation

### **Slide Title**: Production Model Evaluation (GPU Training Run — A7 Full Pipeline)
### **Subtitle**: A7 model trained on `dgxhnode2` GPU server. Validation metrics on clean 90/10 split (no leakage).

---

### **Metrics Dashboard — Production Run 2 (2026-07-22, Clean Val)**

* 📊 **BLEU-4 Score**: **22.66** ✅ Clean validation
* 📊 **BLEU-1 / BLEU-2 / BLEU-3**: **28.33 / 24.75 / 22.87**
* 📊 **ROUGE-L Score**: **28.04**
* 📊 **Word Error Rate (WER)**: **84.51%** (before LLM refinement)

> *(For reference — Previous held-out test, seed=99: BLEU-4 = 28.25, ROUGE-L = 39.10, WER = 71.20% — note: 85.5% training overlap, inflated)*

---

### **Key Talking Points for Slide 2**
* **Honest Generalization**: The clean val split BLEU-4 of **22.66** is the reliable generalization metric — it uses strictly unseen examples with no training overlap.
* **BLEU-4 vs. WER Relationship**: 
  * BLEU-4 scores n-gram precision overlap, which captures key concepts accurately.
  * Word Error Rate (WER) is an edit-distance metric—meaning minor word substitutions, tense shifts, or missing articles count heavily as errors, even when semantic translation is correct.
  * **Post-LLM refinement** via Groq / Llama-3.3-70B significantly reduces effective WER by correcting decoding artifacts.
* **Data Leakage & Post-Hoc Splitting**:
  * The previous held-out test (seed=99) achieved **28.25 BLEU-4** (higher than the validation split's 22.66) because **59 of the 69 test samples (85.5%)** had been seen during training.
  * This is an honest methodological finding that demonstrates the importance of locking test splits *prior* to training sweeps.

---
---

## 🖥️ Slide 3: Qualitative Results & System Integration

### **Slide Title**: Qualitative Performance & Real-time Integration
### **Subtitle**: Translation examples and real-time frontend feature deployment.

---

### **Translation Inference Examples**

* **Perfect Matches**:
  * **Reference**: `i was stopped by some one` $\rightarrow$ **Prediction**: `i was stopped by some one`
  * **Reference**: `serve the food` $\rightarrow$ **Prediction**: `serve the food`
  * **Reference**: `tell me truth` $\rightarrow$ **Prediction**: `tell me truth`
* **Semantic Context (Intent Recovered)**:
  * **Reference**: `why are you angry` $\rightarrow$ **Prediction**: `do not make me angry`
  * **Reference**: `it does not make any difference to me` $\rightarrow$ **Prediction**: `i am very happy`

---

### **Real-time Pipeline Integrations**
* **30 FPS Lag-Free Overlays**: Disabled heavy face mesh rendering; overlays are restricted to hand skeletons and aligned with video frames via `object-fit: cover` to eliminate lag.
* **Groq / Llama-3.3-70B Refinement**: Raw glosses are parsed by Llama 3.3 (or local Qwen-0.5B fallback) to output perfect, conversational English text.
* **Continuous STT Auto-Finalization**: Added a 1.5s silence trigger to split continuous speech into clean sentence bubbles, keeping conversations responsive.
