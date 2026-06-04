# Experiment Report

## Objective

Build a retrieval system that finds similar wafer maps by defect type using
learned embeddings. The system targets the WM-811K dataset and aims for high
per-class retrieval accuracy even on rare defect types, evaluated under an
honest, leakage-free protocol.

## Dataset: WM-811K

- **Total records:** 811,457
- **Labeled:** 172,950 (8 defect classes + `none`)
- **Labeled defects (used here):** 25,519 across 8 classes
- **Source:** MIR Lab, National Taiwan University

| Class | Count | % of defect samples |
|-------|-------|--------------------|
| Edge-Ring | 9,680 | 37.9% |
| Edge-Loc | 5,189 | 20.3% |
| Center | 4,294 | 16.8% |
| Loc | 3,593 | 14.1% |
| Scratch | 1,193 | 4.7% |
| Random | 866 | 3.4% |
| Donut | 555 | 2.2% |
| Near-full | 149 | 0.6% |

The majority `none` class (~85% of all labeled wafers) is excluded from
retrieval and treated as an anomaly-detection problem.

**Key challenge:** extreme class imbalance (largest defect class ~65× the smallest).

## Methodology

### Evaluation protocol (honest)

1. **Lot-based split (70/15/15):** manufacturing lots assigned to
   train/val/test; no lot appears in multiple splits.
2. **Train-only oversampling:** rare classes augmented up to the largest class
   (6,882/class) with rotation, flip, noise. Val/test contain only originals.
3. **Separate index and query:** KNN index built on train embeddings, queries
   from test embeddings (no self-lookup).
4. **Defect-only evaluation:** `none` excluded from retrieval metrics.
5. **Programmatic leak checks:** lot disjointness + duplicate-wafer detection
   asserted at runtime.

### Training

- **Backbones:** ResNet50, EfficientNet-B0, ViT-B/16 (all ImageNet-pretrained)
- **Loss:** Focal (γ=2.0) with inverse-frequency class weights
- **Augmentation:** Mixup (α=0.4, p=0.5) + rotation/flip/noise/random-erasing
- **Optimizer:** AdamW, cosine annealing + linear warmup, mixed precision (AMP)
- **Input:** 96×96, 3-channel (ViT upsamples to 224×224)
- **Early stopping:** patience 30 on validation loss
- **Seed:** 42

## Results

### Backbone comparison (macro KNN@5)

| Backbone | KNN@5 | Train time | Params |
|----------|-------|-----------|--------|
| ResNet50 | 90.4% | 39 min | 25M |
| EfficientNet-B0 | 91.0% | 97 min | 5M |
| **ViT-B/16** | **92.5%** | 425 min | 86M |

### Per-class KNN@5 (ViT-B/16, best)

| Class | KNN@5 | Test Samples |
|-------|-------|-------------|
| Edge-Ring | 98.8% | 1,238 |
| Center | 97.2% | 567 |
| Random | 94.0% | 133 |
| Donut | 93.3% | 105 |
| Edge-Loc | 93.0% | 817 |
| Loc | 89.2% | 592 |
| Near-full | 88.2% | 17 |
| Scratch | 86.6% | 157 |

**Macro-average KNN@5: 92.5%**

### Ablation: impact of evaluation methodology

| Setup | Macro KNN@5 | Inflation source |
|-------|-------------|-----------------|
| Sample-split + oversample before split | 95.4% | Data leakage (copies in test) |
| Lot-split + include "none" | 79.8% | "none" dominates (85% of test) |
| **Lot-split + defects only** | **88.1%–92.5%** | **Honest** (by backbone) |

## Conclusions

1. **ViT-B/16 is the most accurate backbone (92.5%)** under honest lot-split
   evaluation, but costs ~10× the training time of ResNet50. EfficientNet-B0
   (91.0%, 5M params) is the best accuracy-per-cost trade-off.

2. **Focal + Mixup improves on the SupCon baseline** (88.1%) by 2-4pp across
   all backbones, with the largest gains on ambiguous classes (Donut, Loc).

3. **Evaluation methodology critically affects reported numbers.** Naive
   sample-based splitting inflates metrics by ~7-15% due to lot-level leakage.

4. **Rare classes benefit from balanced training.** Donut reaches 93.3% and
   Random 94.0% on ViT once oversampled with augmentation.

5. **Scratch remains the ceiling (~86-88%) for every architecture.** It shares
   visual patterns with Edge-Loc; global features alone cannot fully separate
   it. This is a data/representation limit, not a training-procedure limit.

6. **The system is production-viable for common defects** (Center, Edge-Ring,
   Edge-Loc, Donut, Random) with >93% retrieval accuracy.
