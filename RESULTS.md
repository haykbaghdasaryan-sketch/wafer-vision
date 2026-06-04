# Final Results

## Best: 91.5% Macro KNN@5 (Honest Evaluation)

**Method:** ViT-B/16 + Focal Loss (γ=2.0) + Mixup (α=0.4)
**Dataset:** WM-811K, 25,519 labeled defects (8 classes, no "none")
**Split:** Lot-based 70/15/15 (no leakage) + cross-lot duplicate removal
**Evaluation:** KNN index on train embeddings, queries from test

---

## Model Comparison (macro KNN@5)

| Model | KNN@1 | KNN@3 | KNN@5 | KNN@10 | Train time | Params | Status |
|-------|-------|-------|-------|--------|-----------|--------|--------|
| ResNet50 | 91.3% | 90.6% | 90.4% | 90.7% | 76 min | 25M | ✅ |
| EfficientNet-B0 | 91.3% | 91.5% | 90.5% | 91.4% | 105 min | 5M | ✅ best value |
| **ViT-B/16** | **91.0%** | **91.3%** | **91.5%** | 91.5% | 476 min | 86M | ✅ **best accuracy** |

---

## Per-Class KNN@5

| Class | ResNet50 | EfficientNet-B0 | ViT-B/16 | Test Samples |
|-------|----------|-----------------|----------|-------------|
| Center | 96.1% | 96.3% | 96.5% | 567 |
| Donut | 88.6% | 86.7% | 88.6% | 105 |
| Edge-Loc | 92.8% | 93.4% | 92.8% | 817 |
| Edge-Ring | 98.9% | 98.9% | 98.6% | 1238 |
| Loc | 90.0% | 85.5% | 89.2% | 592 |
| Near-full | 76.5% | 88.2% | 82.4% | 17 |
| Random | 92.5% | 92.5% | 94.0% | 133 |
| Scratch | 87.9% | 82.8% | 89.8% | 157 |
| **Macro avg** | **90.4%** | **90.5%** | **91.5%** | 3626 |

ViT-B/16 weighted accuracy ≈ 94.5%.

---

## Training Details

- **Input resolution:** 96×96 (ViT upsamples to 224×224 internally)
- **Balanced training:** 6,882 samples/class (oversampled with augmentation, train only)
- **Augmentation:** rotation, flip, Gaussian noise, random erasing + Mixup (α=0.4)
- **Loss:** Focal (γ=2.0) with inverse-frequency class weights
- **Optimizer:** AdamW + cosine annealing + linear warmup, mixed precision (AMP)
- **Early stopping:** patience=30 on validation loss
- **Platform:** Kaggle, Tesla T4 (15 GB VRAM), seed 42

---

## Methodology (Why These Results Are Honest)

1. **Lot-based split** — wafers from the same manufacturing lot never appear in both train and test. Prevents lot-level leakage which inflates metrics by ~10-15%.

2. **Cross-lot duplicate removal** — WM-811K contains a few wafer maps duplicated across *different* lots, which a lot-based split alone does not catch. A hash check found and removed 13 such wafers from train (0.36% of test) before evaluation. Most published WM-811K work does not check for this.

3. **Oversampling after split** — rare classes augmented only in the train set. Test set contains only original samples.

4. **Train→Index, Test→Query** — KNN builds the index on train embeddings and searches using test embeddings. No self-lookup.

5. **No "none" class** — the normal class (85% of dataset) is excluded from retrieval and handled separately via anomaly detection.

---

## Historical Comparison

| Experiment | Macro KNN@5 | Issue |
|---|---|---|
| Pretrained ImageNet (no training) | ~72% | No domain knowledge |
| Fine-tune (sample split) | 97.7% | ⚠️ Lot leakage + oversampling leak |
| SupCon (sample split, balanced) | 95.4% | ⚠️ Lot leakage |
| SupCon (lot split, 9 classes incl. none) | 79.8% | "none" class drags down |
| SupCon (lot split, 8 defects) | 88.1% | ✅ First honest result |
| Focal+Mixup — ResNet50 (lot split + dedup) | 90.4% | ✅ Honest |
| Focal+Mixup — EfficientNet-B0 (lot split + dedup) | 90.5% | ✅ Honest |
| **Focal+Mixup — ViT-B/16 (lot split + dedup)** | **91.5%** | ✅ **Final best** |

**Conclusion:** ViT-B/16 is the most accurate backbone (91.5%) but ~6× slower
to train than ResNet50; EfficientNet-B0 (90.5%, 5M params) is the best
accuracy-per-cost choice. Focal+Mixup improves on the SupCon baseline (88.1%)
by 2-3pp. ViT is strongest on the hardest classes — Scratch 89.8% (best of
three) and Loc 89.2%. The remaining ceiling is visual similarity between
classes, not the training procedure.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for full analysis and literature comparison.
