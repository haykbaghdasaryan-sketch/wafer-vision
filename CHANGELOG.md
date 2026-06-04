# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-06-04

### Added
- Focal Loss (γ=2.0) + Mixup (α=0.4) training recipe
- Full 3-backbone comparison on the complete WM-811K defect set (ResNet50,
  EfficientNet-B0, ViT-B/16) via `notebooks/kaggle_focal_mixup.py`
- Mixed precision (AMP) training + ViT gradient checkpointing (fits a single T4)
- Programmatic leakage checks: lot disjointness + cross-lot duplicate-wafer
  removal (found and removed 13 duplicates lot-split alone misses)
- Incremental result saving to `focal_mixup_results.json`
- Literature comparison section in `docs/EXPERIMENTS.md`

### Results
- **ViT-B/16 reaches 91.5% macro KNN@5** (best), EfficientNet-B0 90.5%,
  ResNet50 90.4% — all under honest lot-based split + dedup, 8 defect classes
- ViT is strongest on the hardest classes: Scratch 89.8%, Loc 89.2%
- Focal+Mixup improves on the SupCon baseline (88.1%) by 2-3pp per backbone

## [0.1.0] - 2026-05-31

### Added
- Complete implementation of WaferVision research platform
- 3 backbone architectures: ResNet50, EfficientNet-B0, ViT-B/16
- 4 training modes: pretrained, fine-tune, metric learning (Triplet/SupCon), self-supervised (SimCLR)
- 7+ embedding quality metrics with bootstrap CI
- Nearest-neighbor retrieval engine (<50ms latency on 25K embeddings)
- Anomaly detection (centroid-based + LOF)
- Grad-CAM and Attention Rollout explainability
- UMAP/t-SNE visualization with Plotly WebGL
- 6-page Streamlit web application
- 522 unit tests with pytest + hypothesis
- Hydra/OmegaConf configuration system with Pydantic validation
- CLI pipeline scripts (extract, benchmark, verify, preflight)
- Google Colab / Kaggle training notebooks
- GitHub Actions CI pipeline

### Experiments
- Achieved 88.1% macro KNN@5 on 8 defect classes (honest lot-split evaluation, SupCon)
- SupCon outperforms Triplet loss on balanced data
- Lot-level split prevents ~15% metric inflation from manufacturing lot leakage
