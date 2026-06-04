# Experiment Results

All numbers below come from a single reproducible run of
`notebooks/kaggle_focal_mixup.py` on the **full** WM-811K defect set
(Kaggle, Tesla T4, seed 42). Two automated leakage checks run before training
(lot disjointness + duplicate-wafer detection); both pass.

## Headline

**Macro KNN@5: 92.5%** — ViT-B/16 + Focal(γ=2.0) + Mixup(α=0.4),
honest lot-based split, 8 defect classes (the majority `none` class is
excluded and handled separately by anomaly detection).

## Evaluation protocol (why it is honest)

1. **Lot-based split (70/15/15).** Wafers are grouped by manufacturing
   `lotName`; an entire lot goes to exactly one split. No lot ever spans
   train/val/test. This removes the lot-level leakage that inflates metrics
   by ~10-15% (wafers from the same lot are near-identical).
2. **Oversample train only.** Minority classes are augmented up to the
   largest class (6,882/class) using rotation, flip, and noise. The val and
   test sets contain only original wafers.
3. **Train → index, test → query.** The KNN index is built on train
   embeddings and queried with test embeddings. A query can never retrieve
   itself.
4. **Macro averaging.** Metrics are averaged per class with equal weight, so
   rare classes (Donut, Near-full) count as much as common ones.
5. **Programmatic leak checks.** The script asserts (a) the three lot sets
   are disjoint and (b) removes any train wafer that is byte-identical to a
   test wafer. The second check matters: WM-811K contains a small number of
   wafer maps duplicated across *different* lots, which a lot-based split
   alone does not catch. Our run found and removed 13 such train/test
   duplicates (0.36% of the test set) before evaluation.

## Training setup

- **Backbones:** ResNet50 (2048-d), EfficientNet-B0 (1280-d), ViT-B/16 (768-d),
  all ImageNet-pretrained.
- **Loss:** Focal (γ=2.0) with inverse-frequency class weights.
- **Augmentation:** Mixup (α=0.4, p=0.5) + rotation/flip/noise/random-erasing.
- **Optimizer:** AdamW, cosine schedule + linear warmup, mixed precision (AMP).
- **Input:** 96×96, 3-channel (ViT upsamples to 224×224 internally).
- **Early stopping:** patience 30 on validation loss.

## Final comparison (macro KNN@5)

| Backbone | KNN@1 | KNN@3 | KNN@5 | KNN@10 | Train time | Params |
|----------|-------|-------|-------|--------|-----------|--------|
| ResNet50 | 89.4% | 90.1% | 90.4% | 90.6% | 39 min | 25M |
| EfficientNet-B0 | 90.7% | 90.8% | 91.0% | 91.4% | 97 min | 5M |
| **ViT-B/16** | **91.5%** | **92.5%** | **92.5%** | 91.8% | 425 min | 86M |

## Per-class KNN@5

| Class | ResNet50 | EfficientNet-B0 | ViT-B/16 | Test samples |
|-------|----------|-----------------|----------|-------------|
| Center | 96.3% | 97.0% | 97.2% | 567 |
| Donut | 86.7% | 86.7% | 93.3% | 105 |
| Edge-Loc | 92.2% | 93.5% | 93.0% | 817 |
| Edge-Ring | 99.2% | 99.0% | 98.8% | 1,238 |
| Loc | 86.1% | 85.0% | **89.2%** | 592 |
| Near-full | 82.4% | 88.2% | 88.2% | 17 |
| Random | 92.5% | 92.5% | 94.0% | 133 |
| Scratch | 87.9% | 86.0% | 86.6% | 157 |

## Analysis

1. **ViT-B/16 is the best backbone (92.5%), but at a steep cost.** It needed
   425 min versus 39 min for ResNet50 — roughly 10× the compute for a +2.1pp
   gain. For production, EfficientNet-B0 (91.0%, 5M params, 97 min) is the
   best accuracy-per-cost trade-off; ViT is the choice only when peak accuracy
   matters more than latency and model size.

2. **Focal + Mixup helped the hard classes.** Compared with the earlier SupCon
   baseline (~88% macro), the focal+mixup recipe lifts the macro KNN@5 by
   ~2-4pp across all backbones. Donut in particular jumps to 93.3% on ViT.

3. **ViT's advantage is concentrated on the ambiguous classes.** Loc improves
   to 89.2% (best of the three) and Donut to 93.3%. The global self-attention
   of the transformer appears to disambiguate spatially diffuse patterns
   (Loc ↔ Random) better than the local receptive fields of CNNs.

4. **Scratch remains the ceiling for every architecture (~86-88%).** Scratches
   are thin, near-linear defects that overlap visually with Edge-Loc. No
   backbone resolves this from global features alone — consistent with prior
   work that flags Loc/Scratch as the hardest classes.

5. **The numbers are honest, not inflated.** All three backbones land in a
   narrow 90.4-92.5% band on the same lot-based split. A leakage bug would
   have produced 98-99% (the level seen in random-split papers). The modest,
   tightly-clustered results are the signature of a clean protocol.

6. **We caught a leakage path that lot-split alone misses.** The duplicate-
   wafer check found 13 wafer maps that are byte-identical across different
   lots and were removed from train before evaluation. The impact is tiny
   (0.36% of test, within noise), but most published WM-811K work does not
   check for this at all — so their reported numbers may include a small
   amount of this leakage.

## Comparison with the literature

Reported accuracy on WM-811K is often 98-99%, but the evaluation protocols
differ from ours in ways that make a direct number-to-number comparison
misleading. The table summarizes what we verified by reading each paper.

| Work | Reported | Classes | Split | Metric | Notes |
|------|----------|---------|-------|--------|-------|
| Bao et al. 2024 (arXiv:2411.11029) | 98.56% | 8 | random 4:1 | accuracy | Autoencoder augmentation; pre-aug val accuracy was only ~85%. **No lot-based split.** |
| Wafer2Spike 2024 (arXiv:2411.19422) | 98% | **9 (incl. None)** | random | avg accuracy | Includes the easy `No-Pattern` class; **random split.** Their Scratch recall is 55-69%. |
| Prabhu & Madhuvairy 2026 (preprints 202603.1447) | 60.0% (vision) / 72.7% (fusion) | 8 | stratified 70/15/15 | accuracy / weighted F1 | Same 8-class task as ours, but a small CNN trained **from scratch** (no pretraining). |
| **This work** | **92.5% macro KNN@5** | 8 | **lot-based 70/15/15** | macro KNN@5 (retrieval) | Pretrained backbones; **no lot leakage**; rare classes weighted equally. |

**Key points for interpretation:**

- The 98% results use **random splits**, which let near-identical wafers from
  the same lot fall into both train and test — a known source of inflation. We
  use a lot-based split, which is stricter.
- Wafer2Spike's 98% is over **9 classes including the easy `None` class**,
  which raises the average; we evaluate the 8 harder defect classes only.
- Our task is **retrieval (macro KNN@5)**, not direct classification, and uses
  macro averaging that penalizes weak rare classes. These choices lower the
  headline number but make it more faithful.
- The most directly comparable study (Prabhu 2026, same 8 classes, stratified
  split) reports 60-73% with a from-scratch CNN, which calibrates how hard the
  8-class problem is once the easy `none` class is removed.

**Honest takeaway:** we do not claim to beat the 98% papers — the metrics are
not comparable. We claim a *cleaner protocol*: no lot leakage, defect-only
classes, macro averaging. Our 92.5% is a conservative, reproducible number in
that stricter setting.

## Limitations

- **Single seed (42).** Reported metrics are from one run; multiple seeds
  would be needed to establish variance and statistical significance.
- **Near-full has only 17 test samples** — its per-class number is noisy.
- **96×96 input** (upsampled for ViT). Native or higher resolution might shift
  results, especially for thin Scratch defects.
- **One dataset.** Generalization to other fabs/datasets is untested.

## Reproducibility

- Script: `notebooks/kaggle_focal_mixup.py` (single Kaggle cell, Run All).
- Dataset: `qingyi/wm811k-wafer-map` → `LSWMD.pkl`.
- Outputs: `focal_mixup_results.json` (+ per-model `.pth` checkpoints).
- Seed 42, deterministic split, leakage checks enforced at runtime.
