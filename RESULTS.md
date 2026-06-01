# Final Results

## Best: 90.8% Macro KNN@5 (Honest Evaluation)

**Method:** Focal Loss (γ=2.0) + Mixup (α=0.4) + EfficientNet-B0  
**Dataset:** WM-811K, 25,519 labeled defects (8 classes, no "none")  
**Split:** Lot-based 70/15/15 (no leakage between train/val/test)  
**Evaluation:** KNN index on train embeddings, queries from test  

---

## Per-Class Results

| Class | KNN@5 | Precision@5 | Test Samples |
|-------|-------|-------------|-------------|
| Edge-Ring | 98.7% | 98.7% | 1238 |
| Center | 97.4% | 97.4% | 567 |
| Edge-Loc | 94.0% | 94.0% | 817 |
| Random | 92.5% | 92.5% | 133 |
| Near-full | 88.2% | 88.2% | 17 |
| Loc | 86.0% | 86.0% | 592 |
| Donut | 84.8% | 84.8% | 105 |
| Scratch | 84.7% | 84.7% | 157 |

**Macro-average KNN@5: 90.8%**  
**Weighted accuracy: 94.1%**

---

## Model Comparison

| Model | Macro KNN@5 | Training Time | Status |
|-------|-------------|--------------|--------|
| ResNet50 | 90.7% | 133 min | ✅ |
| EfficientNet-B0 | 90.8% | 155 min | ✅ Best |
| ViT-B/16 | — | OOM | ❌ Needs >15GB VRAM |

---

## Training Details

- **Input resolution:** 96×96
- **Balanced training:** 6,882 samples/class (oversampled with augmentation)
- **Augmentation:** Random rotation, flip, Gaussian noise (σ=0.03)
- **Optimizer:** AdamW, lr=3e-4 (backbone lr=3e-5)
- **Scheduler:** Cosine annealing with 10-epoch warmup
- **Early stopping:** patience=30, best epoch=33 (ResNet), 135 (EfficientNet)
- **Platform:** Kaggle, Tesla T4 (15 GB VRAM)

---

## Methodology (Why These Results Are Honest)

1. **Lot-based split** — wafers from the same manufacturing lot never appear in both train and test. This prevents lot-level leakage which inflates metrics by ~15%.

2. **Oversampling after split** — rare classes augmented only in train set. Test set contains only original samples.

3. **Train→Index, Test→Query** — KNN evaluation builds the nearest-neighbor index on train embeddings and searches using test embeddings. No self-lookup.

4. **No "none" class** — the "normal" class (85% of dataset) is excluded from retrieval evaluation. It's handled separately via anomaly detection.

---

## Historical Comparison

| Experiment | Macro KNN@5 | Issue |
|---|---|---|
| Pretrained ImageNet (no training) | ~72% | No domain knowledge |
| Fine-tune (sample split) | 97.7% | ⚠️ Lot leakage + oversampling leak |
| SupCon (sample split, balanced) | 95.4% | ⚠️ Lot leakage |
| SupCon (lot split, 9 classes) | 79.8% | "none" class drags down |
| SupCon (lot split, 8 defects) | 90.8% | ✅ Honest |
| **Focal+Mixup (lot split, 8 defects)** | **90.8%** | ✅ **Final** |

**Conclusion:** ~91% is the honest ceiling for single-backbone embedding retrieval on WM-811K with proper lot-level split. Focal Loss and Mixup do not improve beyond SupCon — the bottleneck is visual similarity between classes (Loc↔Random, Scratch↔Edge-Loc), not the training procedure.
