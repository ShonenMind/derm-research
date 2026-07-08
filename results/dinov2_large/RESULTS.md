# DINOv2-Large — BCN20000 Benchmark Results

## Overview

DINOv2-Large (307M parameters) trained for 10 epochs on the BCN20000 dermoscopy dataset (12,233 training images, 8 diagnostic categories). Full fine-tuning of backbone + classification head with cosine LR schedule, class-weighted loss, and early stopping.

**Primary metric: Balanced Accuracy = 0.6581**

---

## Results Summary

| Metric | Score |
|--------|-------|
| Balanced Accuracy | 0.6581 |
| Macro F1 | 0.5943 |
| Macro AUROC | 0.9012 |
| Best Epoch | 10 |

---

## Training Curves (`dinov2_large_training_curves.png`)

Three-panel plot showing Loss, Balanced Accuracy, and Macro F1 over 10 epochs for both training and validation sets.

**Key observations:**
- Loss decreases smoothly throughout — no instability or divergence
- Validation balanced accuracy tracks training closely (minimal overfitting)
- The model was still improving at epoch 10 (no early stopping triggered), suggesting additional epochs or a lower learning rate schedule could push performance slightly higher

---

## Confusion Matrix (`dinov2_large_confusion_matrix.png`)

Normalised confusion matrix showing where the model confuses one diagnosis for another. Each row sums to 1.0 — values on the diagonal represent per-class sensitivity (recall).

**Key observations:**
- **NV (0.81)** and **BCC (0.75)** are the best-classified categories — both are visually distinctive and well-represented in training data
- **MEL (0.71)** shows strong melanoma detection — the most clinically important metric
- **BKL (0.45)** is the most confused class — frequently misclassified as NV (0.18), which is expected since both are benign pigmented lesions that overlap in dermoscopic appearance
- **AK ↔ SCC confusion** (AK misclassified as BCC 0.12, SCC 0.10) reflects the clinical reality that these are on the same squamous neoplasia spectrum
- **DF (0.48)** confused with BKL (0.16) and NV (0.12) — dermatofibromas can mimic both in dermoscopy

---

## Per-Class Sensitivity (`dinov2_large_sensitivity.png`)

Bar chart showing recall for each diagnostic category, color-coded by performance tier:
- **Green (≥ 0.65):** MEL, NV, BCC — reliable detection
- **Orange (0.55–0.65):** VASC — moderate performance despite very few training samples
- **Red (< 0.55):** AK, BKL, DF, SCC — classes that need improvement

The dotted line at 0.50 marks chance-level performance for reference. All classes exceed chance, but rare classes with fewer training samples (DF: 168, VASC: 151 in training) show expected performance limitations.

---

## Stratified Analysis (`dinov2_large_stratified.png`)

Performance breakdown by two clinically relevant subgroups:

### By Malignancy Category (left panel)
- **Malignant** (n=1331): Bal. Acc 0.684, F1 0.653
- **Benign** (n=1128): Bal. Acc 0.622, F1 0.561
- **Indeterminate** (n=163): Bal. Acc 0.584, F1 0.510

The model performs ~6% better on malignant lesions, likely because malignant classes (MEL, BCC, SCC) have more distinctive visual features and higher representation in training data.

### By Melanocytic Status (right panel)
- **Melanocytic** (n=1543): Bal. Acc 0.712, F1 0.689
- **Non-Melanocytic** (n=1079): Bal. Acc 0.599, F1 0.540

The ~11% gap suggests DINOv2's self-supervised pretraining captures melanocytic patterns (pigment network, globules) particularly well, while non-melanocytic lesions (which include the heterogeneous AK, BKL, DF, VASC, SCC categories) are harder to distinguish from each other.

---

## Clinical Implications

1. **Melanoma sensitivity of 0.71** is encouraging for a screening tool — the model catches 71% of melanomas, though the 29% miss rate means it cannot yet serve as a standalone diagnostic
2. **BKL-NV confusion** is the dominant error mode and reflects a known challenge in dermoscopy where even experienced clinicians disagree
3. **Class imbalance** drives most of the poor performance on rare classes — data augmentation or oversampling strategies could improve DF/VASC/SCC sensitivity
4. **AUROC of 0.90** indicates strong separability — the model's probability estimates are well-calibrated even when the argmax prediction is wrong, which is useful for triaging uncertain cases to human review
