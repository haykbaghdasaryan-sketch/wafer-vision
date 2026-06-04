# WaferVision — Final Results (Focal Loss + Mixup)

Honest evaluation on the real **WM-811K** dataset. All numbers use a
lot-based split (no leakage), oversampling applied to the train set only,
and KNN retrieval where the index is built on train embeddings and queried
with test embeddings. Evaluation covers the **8 defect classes** (the
`none` class is excluded — it is handled separately by anomaly detection).
Two automated leakage checks run before training and pass on every run.

## Headline

**92.5% macro KNN@5** — ViT-B/16 + Focal(γ=2.0) + Mixup(α=0.4)

## Model comparison

| Model | KNN@1 | KNN@3 | KNN@5 | KNN@10 | Train time |
|-------|-------|-------|-------|--------|-----------|
| ResNet50 + Focal + Mixup | 89.4% | 90.1% | 90.4% | 90.6% | 39 min |
| EfficientNet-B0 + Focal + Mixup | 90.7% | 90.8% | 91.0% | 91.4% | 97 min |
| **ViT-B/16 + Focal + Mixup** | **91.5%** | **92.5%** | **92.5%** | 91.8% | 425 min |

ViT-B/16 is the most accurate; EfficientNet-B0 is the best accuracy-per-cost
(5M params, 97 min). ViT runs with gradient checkpointing + batch=32 + AMP to
fit a single T4 at 224×224.

## Per-class KNN@5

| Class | ResNet50 | EfficientNet-B0 | ViT-B/16 | Test samples |
|-------|----------|-----------------|----------|-------------|
| Center | 96.3% | 97.0% | 97.2% | 567 |
| Donut | 86.7% | 86.7% | 93.3% | 105 |
| Edge-Loc | 92.2% | 93.5% | 93.0% | 817 |
| Edge-Ring | 99.2% | 99.0% | 98.8% | 1238 |
| Loc | 86.1% | 85.0% | 89.2% | 592 |
| Near-full | 82.4% | 88.2% | 88.2% | 17 |
| Random | 92.5% | 92.5% | 94.0% | 133 |
| Scratch | 87.9% | 86.0% | 86.6% | 157 |

**ViT-B/16 macro KNN@5: 92.5%** · weighted accuracy ≈ 94.8%.

## Methodology (why this is honest)

1. **Lot-based split** — wafers from the same manufacturing lot never span
   train and test. This removes the lot-level leakage that inflates metrics
   by 10-15%.
2. **Oversample train only** — minority classes are augmented in the train
   set; the test set contains only original samples.
3. **Train → index, test → query** — no self-lookup in KNN.
4. **No `none` class** — the 85%-majority normal class is excluded from
   retrieval and handled by anomaly detection instead.
5. **Programmatic leak checks** — lot disjointness + duplicate-wafer detection
   are asserted at runtime before training begins.

## Interpretation

ViT-B/16 (92.5%) edges out the CNNs but costs ~10× the training time of
ResNet50. The transformer's global self-attention helps most on the
spatially ambiguous classes (Loc → 89.2%, Donut → 93.3%). Focal+Mixup lifts
all backbones ~2-4pp over the earlier SupCon baseline (88.1%). The remaining
ceiling is visual similarity between classes (Scratch ↔ Edge-Loc), not the
training procedure — beating it would require higher-resolution inputs,
local-attention mechanisms, or ensembles.

## How to reproduce

- Run `notebooks/kaggle_focal_mixup.py` (single Kaggle cell, Run All).
- Dataset (Kaggle input): `qingyi/wm811k-wafer-map` → `LSWMD.pkl`.
- Outputs: `focal_mixup_results.json` + per-model `.pth` checkpoints.
- Seed: 42 · Platform: Kaggle Tesla T4 (15 GB).

Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision
