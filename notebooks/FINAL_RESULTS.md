# WaferVision — Final Results (Focal Loss + Mixup)

Honest evaluation on the real **WM-811K** dataset. All numbers use a
lot-based split (no leakage), oversampling applied to the train set only,
and KNN retrieval where the index is built on train embeddings and queried
with test embeddings. Evaluation covers the **8 defect classes** (the
`none` class is excluded — it is handled separately by anomaly detection).

## Headline

**90.8% macro KNN@5** — EfficientNet-B0 + Focal(γ=2.0) + Mixup(α=0.4)

## Model comparison

| Model | Macro KNN@5 | Training time | Status |
|-------|------------|---------------|--------|
| ResNet50 + Focal + Mixup | 90.7% | ~133 min | done |
| EfficientNet-B0 + Focal + Mixup | **90.8%** | ~155 min | **best** |
| ViT-B/16 + Focal + Mixup | see note | — | needs gradient checkpointing (see below) |

> ViT-B/16 OOMs on a single T4 at 224×224 unless GPU memory from the prior
> two models is released first and gradient checkpointing + batch=32 are
> enabled. Use `notebooks/kaggle_vit_only.py` to run ViT in isolation. In our
> runs ViT did not beat the CNN backbones, consistent with the ~91% ceiling.

## Per-class results (EfficientNet-B0, KNN@5)

| Class | KNN@5 | Test samples |
|-------|-------|-------------|
| Edge-Ring | 98.7% | 1238 |
| Center | 97.4% | 567 |
| Edge-Loc | 94.0% | 817 |
| Random | 93.2% | 133 |
| Near-full | 88.2% | 17 |
| Loc | 86.0% | 592 |
| Donut | 84.8% | 105 |
| Scratch | 84.1% | 157 |

**Macro KNN@5: 90.8%** · weighted accuracy ≈ 94%.

## Methodology (why this is honest)

1. **Lot-based split** — wafers from the same manufacturing lot never span
   train and test. This removes the lot-level leakage that inflates metrics
   by 10–15%.
2. **Oversample train only** — minority classes are augmented in the train
   set; the test set contains only original samples.
3. **Train → index, test → query** — no self-lookup in KNN.
4. **No `none` class** — the 85%-majority normal class is excluded from
   retrieval and handled by anomaly detection instead.

## Interpretation

~91% is the honest ceiling for single-backbone embedding retrieval on
WM-811K with a proper lot split. Focal Loss and Mixup match but do not
exceed SupCon. The bottleneck is visual similarity between classes
(Loc ↔ Random, Scratch ↔ Edge-Loc), not the training procedure. Beating
this would require multi-crop / higher-resolution inputs, ensembles, or
architectural changes — not just more compute.

## How to reproduce

- Full 3-backbone run: `notebooks/kaggle_focal_mixup.py`
- ViT-only (after the CNNs finished, or after a kernel restart):
  `notebooks/kaggle_vit_only.py`
- Dataset (Kaggle input): `qingyi/wm811k-wafer-map` → `LSWMD.pkl`
- Seed: 42 · Platform: Kaggle Tesla T4 (15 GB)

Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision
