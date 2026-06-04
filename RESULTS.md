# Final Results

## Best: 92.5% Macro KNN@5 (Honest Evaluation)

**Method:** ViT-B/16 + Focal Loss (γ=2.0) + Mixup (α=0.4)
**Dataset:** WM-811K, 25,519 labeled defects (8 classes, no "none")
**Split:** Lot-based 70/15/15 (no leakage between train/val/test)
**Evaluation:** KNN index on train embeddings, queries from test

---

## Model Comparison (macro KNN@5)

| Model | KNN@1 | KNN@3 | KNN@5 | KNN@10 | Train time | Params | Status |
|-------|-------|-------|-------|--------|-----------|--------|--------|
| ResNet50 | 89.4% | 90.1% | 90.4% | 90.6% | 39 min | 25M | ✅ |
| EfficientNet-B0 | 90.7% | 90.8% | 91.0% | 91.4% | 97 min | 5M | ✅ best value |
| **ViT-B/16** | **91.5%** | **92.5%** | **92.5%** | 91.8% | 425 min | 86M | ✅ **best accuracy** |

---

## Per-Class KNN@5

| Class | ResNet50 | EfficientNet-B0 | ViT-B/16 | Test Samples |
|-------|----------|-----------------|----------|-------------|
| Center | 96.3% | 97.0% | 97.2% | 567 |
| Donut | 86.7% | 86.7% | 93.3% | 105 |
| Edge-Loc | 92.2% | 93.5% | 93.0% | 817 |
| Edge-Ring | 99.2% | 99.0% | 98.8% | 1238 |
| Loc | 86.1% | 85.0% | 89.2% | 592 |
| Near-full | 82.4% | 88.2% | 88.2% | 17 |
| Random | 92.5% | 92.5% | 94.0% | 133 |
| Scratch | 87.9% | 86.0% | 86.6% | 157 |
| **Macro avg** | **90.4%** | **91.0%** | **92.5%** | 3626 |

ViT-B/16 weighted accuracy ≈ 94.8%.

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

2. **Oversampling after split** — rare classes augmented only in the train set. Test set contains only original samples.

3. **Train→Index, Test→Query** — KNN builds the index on train embeddings and searches using test embeddings. No self-lookup.

4. **No "none" class** — the normal class (85% of dataset) is excluded from retrieval and handled separately via anomaly detection.

5. **Programmatic leak checks** — the training script asserts (a) the three lot sets are disjoint and (b) no identical wafer tensor appears in both train and test. Both pass on every run.

---

## Historical Comparison

| Experiment | Macro KNN@5 | Issue |
|---|---|---|
| Pretrained ImageNet (no training) | ~72% | No domain knowledge |
| Fine-tune (sample split) | 97.7% | ⚠️ Lot leakage + oversampling leak |
| SupCon (sample split, balanced) | 95.4% | ⚠️ Lot leakage |
| SupCon (lot split, 9 classes incl. none) | 79.8% | "none" class drags down |
| SupCon (lot split, 8 defects) | 88.1% | ✅ First honest result |
| Focal+Mixup — ResNet50 (lot split) | 90.4% | ✅ Honest |
| Focal+Mixup — EfficientNet-B0 (lot split) | 91.0% | ✅ Honest |
| **Focal+Mixup — ViT-B/16 (lot split)** | **92.5%** | ✅ **Final best** |

**Conclusion:** ViT-B/16 is the most accurate backbone (92.5%) but ~10× slower
to train than ResNet50; EfficientNet-B0 (91.0%, 5M params) is the best
accuracy-per-cost choice. Focal+Mixup improves on the SupCon baseline (88.1%)
by 2-4pp across all backbones. The remaining ceiling is visual similarity
between classes (Scratch ↔ Edge-Loc), not the training procedure.

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for full analysis and literature comparison.
