# Notebooks (Kaggle)

One self-contained script that runs the full Focal + Mixup study on the real
**WM-811K** dataset and produces the final, honest comparison across three
backbones.

| File | What it does | Runtime (T4) |
|------|--------------|--------------|
| `kaggle_focal_mixup.py` | Full run: ResNet50 -> EfficientNet-B0 -> ViT-B/16, each with Focal(gamma=2.0) + Mixup(alpha=0.4) on FULL data. Mixed precision, GPU freed between models, ViT uses gradient checkpointing. Writes `focal_mixup_results.json` after each model. | ~3-6 h |
| `FINAL_RESULTS.md` | Documented final numbers + methodology. | - |

## How to run

1. Kaggle -> **New Notebook**.
2. **Add Data (Input)**: search `wm811k-wafer-map` by qingyi.
3. **Settings**: Accelerator = **GPU T4**, Internet = **ON**.
4. Paste **all** of `kaggle_focal_mixup.py` into one cell -> **Run All**.

That's it. The script clones the repo, loads the data, trains and evaluates
all three backbones, and prints a final comparison table.

## What the script does (stage by stage)

1. **SETUP** — clone repo, install deps, detect GPU, enable AMP.
2. **LOAD** — read `LSWMD.pkl` via `CompatUnpickler` (legacy pandas pickle).
3. **SPLIT** — extract the 8 defect classes, split 70/15/15 **by lot** so no
   manufacturing lot crosses train/val/test (this is what makes the eval honest).
4. **TENSORS** — resize every wafer to 96x96, 3 channels; oversample minority
   classes in the train set only (with augmentation).
5. **TRAIN/EVAL** — shared `train_model` / `evaluate_model`:
   - Focal loss (class-balanced alpha) + Mixup, AdamW, cosine schedule + warmup.
   - Mixed precision (AMP) for speed and memory.
   - Honest KNN: index built on **train** embeddings, queried with **test**.
6. **EXPERIMENTS A/B/C** — ResNet50, EfficientNet-B0, ViT-B/16. GPU memory is
   released between models; each result is saved to `focal_mixup_results.json`.
7. **COMPARISON** — final table + per-class KNN@5 (highlighting Loc / Scratch),
   read back from the JSON so it prints even if a later stage is interrupted.

## Reliability notes

- **Full data**, no subsampling.
- **AMP** roughly halves memory and doubles speed on a T4, so ViT-B/16 at
  224x224 fits. ViT additionally uses gradient checkpointing + batch=32.
- **Embeddings are extracted in fp32 and NaN-sanitized** before KNN, so the
  evaluation never crashes on a stray non-finite value.
- **Incremental saving**: if the Kaggle 12 h limit is hit during the (slowest)
  ViT stage, the ResNet50 + EfficientNet results are already on disk and the
  summary still prints.
- If ViT ever OOMs on a smaller GPU, lower `BATCH_VIT` (e.g. 16) at the top of
  the file and re-run.

## Outputs

- `focal_mixup_results.json` — all metrics (per model + per class).
- `resnet50_focal_mixup_best.pth`, `efficientnet_b0_focal_mixup_best.pth`,
  `vit_b16_focal_mixup_best.pth` — best checkpoints.

Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision
