# Notebooks (Kaggle)

Self-contained scripts for running WaferVision experiments on Kaggle
(Tesla T4, 15 GB). Each file is meant to be pasted into a single Kaggle
cell and run with **Run All**. Add the dataset `qingyi/wm811k-wafer-map`
as an Input first.

| File | What it does | Runtime |
|------|--------------|---------|
| `kaggle_honest_eval.py` | SupCon baseline, lot-based honest eval (8 classes). Produces the 90.8% reference result. | ~30 min |
| `kaggle_focal_mixup.py` | Full 3-backbone run: ResNet50 → EfficientNet-B0 → ViT-B/16, each with Focal(γ=2.0) + Mixup(α=0.4). Frees GPU between models; ViT uses gradient checkpointing + batch=32. | ~6–10 h |
| `kaggle_vit_only.py` | Runs **only** ViT-B/16 + prints the final comparison table. Use this if the CNNs already finished, the ViT cell OOM'd, or the kernel restarted. Rebuilds tensors automatically if they are gone. | ~2–4 h |
| `FINAL_RESULTS.md` | The documented final numbers and methodology. | — |

## Cell-by-cell notes for `kaggle_focal_mixup.py`

The script is a single linear flow with labeled stages:

1. **SETUP** — clones the repo (branch `feat/focal-mixup`), installs deps,
   detects GPU.
2. **LOAD WM-811K** — `CompatUnpickler` reads the old pandas pickle
   (`encoding='latin1'`, module remapping).
3. **PARSE + LOT-BASED SPLIT** — extracts the 8 defect classes and splits
   70/15/15 by `lotName` so no lot crosses splits.
4. **PROCESS TENSORS** — every wafer is resized to 96×96, scaled, and
   repeated to 3 channels. Train minority classes are oversampled with
   augmentation up to the largest class.
5. **EXPERIMENT A/B/C** — trains each backbone via the shared
   `train_model` / `evaluate_model` functions, freeing GPU memory in
   between (`del model; torch.cuda.empty_cache(); gc.collect()`).
6. **COMPARISON + SUMMARY** — robust tables that skip any model that
   OOM'd, so the run always finishes with a clean report.

## Memory tips (T4, 15 GB)

- Train the CNN backbones first, then ViT last.
- ViT at 224×224 needs the prior models released from VRAM; the scripts do
  this automatically.
- If ViT still OOMs, lower `batch_override` (e.g. 16) or increase
  `ckpt_segments` in `ViTWrapper`.

Repo: https://github.com/haykbaghdasaryan-sketch/wafer-vision
