# Changelog

All notable changes to this project will be documented in this file.

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
- Achieved 88.1% macro KNN@5 on 8 defect classes (honest lot-split evaluation)
- SupCon outperforms Triplet loss on balanced data
- Lot-level split prevents ~15% metric inflation from manufacturing lot leakage
