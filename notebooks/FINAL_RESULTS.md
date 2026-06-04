# WaferVision — Final Results (Focal Loss + Mixup)

Honest evaluation on the real **WM-811K** dataset. All numbers use a
lot-based split (no leakage), cross-lot duplicate removal, oversampling
applied to the train set only, and KNN retrieval where the index is built on
train embeddings and queried with test embeddings. Evaluation covers the
**8 defect classes** (the `none` class is excluded — it is handled separately
by anomaly detection). Two automated leakage checks run before training.

## Headline

**91.5% macro KNN@5** — ViT-B/16 + Focal(γ=2.0) + Mixup(α=0.4)

## Model comparison

| Model | KNN@1 | KNN@3 | KNN@5 | KNN@10 | Train time |
|-------|-------|-------|-------|--------|-----------|
| ResNet50 + Focal + Mixup | 91.3% | 90.6% | 90.4% | 90.7% | 76 min |
| EfficientNet-B0 + Focal + Mixup | 91.3% | 91.5% | 90.5% | 91.4% | 105 min |
| **ViT-B/16 + Focal + Mixup** | **91.0%** | **91.3%** | **91.5%** | 91.5% | 476 min |

ViT-B/16 is the most accurate; EfficientNet-B0 is the best accuracy-per-cost
(5M params, 105 min). ViT runs with gradient checkpointing + batch=32 + AMP to
fit a single T4 at 224×224.

## Per-class KNN@5

| Class | ResNet50 | EfficientNet-B0 | ViT-B/16 | Test samples |
|-------|----------|-----------------|----------|-------------|
| Center | 96.1% | 96.3% | 96.5% | 567 |
| Donut | 88.6% | 86.7% | 88.6% | 105 |
| Edge-Loc | 92.8% | 93.4% | 92.8% | 817 |
| Edge-Ring | 98.9% | 98.9% | 98.6% | 1238 |
| Loc | 90.0% | 85.5% | 89.2% | 592 |
| Near-full | 76.5% | 88.2% | 82.4% | 17 |
| Random | 92.5% | 92.5% | 94.0% | 133 |
| Scratch | 87.9% | 82.8% | 89.8% | 157 |

**ViT-B/16 macro KNN@5: 91.5%** · weighted accuracy ≈ 94.5%.

## Methodology (why this is honest)

1. **Lot-based split** — wafers from the same manufacturing lot never span
   train and test.
2. **Cross-lot duplicate removal** — 13 wafer maps byte-identical across
   different lots were removed from train (lot-split alone misses these).
3. **Oversample train only** — minority classes augmented in the train set;
   test contains only original wafers.
4. **Train → index, test → query** — no self-lookup in KNN.
5. **No `none` class** — the 85%-majority normal class is excluded from
   retrieval and handled by anomaly detection.

## Interpretation

ViT-B/16 (91.5%) edges out the CNNs but costs ~6× the training time of
ResNet50. The transformer's global self-attention helps most on the hardest
classes — Scratch 89.8% and Loc 89.2%, both best of the three. Focal+Mixup
lifts all backbones ~2-3pp over the earlier SupCon baseline (88.1%). The
remaining ceiling is visual similarity between classes, not the training
procedure.

## How to reproduce

- Run `notebooks/kaggle_focal_mixup.py` (single Kaggle cell, Run All).
- Dataset (Kaggle input): `qingyi/wm811k-wafer-map` → `LSWMD.pkl`.
- Outputs: `focal_mixup_results.json` + per-model `.pth` checkpoints.
- Seed: 42 · Platform: Kaggle Tesla T4 (15 GB).

Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision
