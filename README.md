# 🔬 WaferVision

**Intelligent Wafer Defect Embedding & Retrieval System**

Research-grade platform for wafer defect analysis built on the WM-811K dataset (811,457 wafer maps; 8 defect classes evaluated, plus a `none` class handled by anomaly detection). Combines deep learning feature extraction, metric learning, embedding evaluation, explainability, anomaly detection, and interactive visualization.

## 🎯 Results (Honest Evaluation)

**Macro-average KNN@5: 92.5%** (ViT-B/16 + Focal + Mixup) on 8 defect classes with lot-based split (no data leakage). Two automated leakage checks run before every training run and pass.

### Backbone comparison

| Backbone | KNN@5 | Train time | Params |
|----------|-------|-----------|--------|
| ResNet50 | 90.4% | 39 min | 25M |
| EfficientNet-B0 | 91.0% | 97 min | 5M |
| **ViT-B/16** | **92.5%** | 425 min | 86M |

### Per-class KNN@5 (ViT-B/16, best model)

| Class | KNN@5 | Test Samples |
|-------|-------|-------------|
| Edge-Ring | 98.8% | 1238 |
| Center | 97.2% | 567 |
| Random | 94.0% | 133 |
| Donut | 93.3% | 105 |
| Edge-Loc | 93.0% | 817 |
| Loc | 89.2% | 592 |
| Near-full | 88.2% | 17 |
| Scratch | 86.6% | 157 |

**Evaluation methodology:**
- Split by manufacturing LOT (not by sample) — prevents lot-level leakage
- KNN index built on TRAIN embeddings, queries from TEST
- Oversampling applied ONLY to train set with augmentation (not identical copies)
- No "none" class in retrieval evaluation (handled separately via anomaly detection)
- Programmatic leak checks (lot disjointness + duplicate-wafer detection) enforced at runtime

## 🏗️ Architecture

```
Data Layer         → Data_Engine (load, preprocess, augment, sample)
Model Layer        → Embedding_Engine (ResNet50, EfficientNet-B0, ViT-B/16)
Training Layer     → Training_Engine (fine-tune, triplet, SupCon, SimCLR)
Analysis Layer     → Evaluation_Suite + Retrieval_Engine + Anomaly_Detector
Visualization      → Grad-CAM, UMAP, t-SNE (Plotly WebGL)
Presentation       → Streamlit Web App (6 pages)
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/haykbaghdasaryan-sketch/wafer-vision.git
cd wafer-vision
pip install -e ".[dev]"
```

### Dataset
Download WM-811K from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place `LSWMD.pkl` in `data/raw/`.

### Run Pipeline
```bash
make preprocess    # Preprocess into train/val/test splits
make extract       # Extract embeddings (pretrained ResNet50)
make evaluate      # Compute all metrics
make app           # Launch Streamlit UI at localhost:8501
```

### Training (requires GPU)
```bash
make train BACKBONE=resnet50 MODE=finetune   # Fine-tune with CrossEntropy
make train BACKBONE=resnet50 MODE=metric     # Metric learning (Triplet)
```

### Google Colab / Kaggle
See `colab_full_training.ipynb` for cloud training with T4 GPU.

## 📊 Features

- **3 Backbone Architectures:** ResNet50 (2048D), EfficientNet-B0 (1280D), ViT-B/16 (768D)
- **4 Training Modes:** Pretrained, Fine-tune, Metric Learning (Triplet/SupCon), Self-supervised (SimCLR)
- **7+ Metrics:** KNN Accuracy, Recall@K, Precision@K, MAP, Silhouette, NMI, Separability
- **Retrieval Engine:** <50ms query latency on 25K embeddings
- **Anomaly Detection:** Centroid-based + LOF with configurable thresholds
- **Explainability:** Grad-CAM (CNN) / Attention Rollout (ViT)
- **Visualization:** Interactive UMAP/t-SNE with Plotly WebGL
- **Web App:** 6-page Streamlit dashboard

## 🧪 Testing

```bash
pytest tests/ -v          # 522 unit tests
make lint                 # Ruff linting
make typecheck            # MyPy type checking
```

## 📁 Project Structure

```
wafer-vision/
├── src/                  # Core library (models, training, evaluation, anomaly, viz)
├── app/                  # Streamlit web application (6 pages)
├── configs/              # Hydra YAML configuration files
├── scripts/              # CLI pipeline scripts
├── tests/                # 522 unit + integration tests
├── docs/                 # Architecture, experiments, API reference
├── colab_full_training.ipynb  # Cloud training notebook
└── Makefile              # All pipeline targets
```

## 📈 Experiment History

| Experiment | Setup | Macro KNN@5 | Notes |
|---|---|---|---|
| Pretrained (no training) | ImageNet ResNet50 | ~72% | Baseline |
| Fine-tune (sample split) | CrossEntropy + class weights | 97.7% (biased) | ⚠️ Lot leakage present |
| SupCon (lot split, 8 defects) | Balanced, augmented | 88.1% | First honest result |
| Focal+Mixup — ResNet50 | Lot split, balanced | 90.4% | Honest |
| Focal+Mixup — EfficientNet-B0 | Lot split, balanced | 91.0% | Honest |
| **Focal+Mixup — ViT-B/16** | **Lot split, balanced** | **92.5%** | ✅ **Best, honest** |

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full analysis and comparison with the literature.

## 📜 License

MIT

## 🙏 Acknowledgments

- WM-811K dataset by MIR Lab, National Taiwan University
- PyTorch, torchvision, scikit-learn, Streamlit communities
