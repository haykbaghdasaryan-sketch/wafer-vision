# 🔬 WaferVision

**Intelligent Wafer Defect Embedding & Retrieval System**

Research-grade platform for wafer defect analysis built on the WM-811K dataset (811,457 wafer maps, 9 defect classes). Combines deep learning feature extraction, metric learning, embedding evaluation, explainability, anomaly detection, and interactive visualization.

## 🎯 Results (Honest Evaluation)

**Macro-average KNN@5: 88.1%** on 8 defect classes with lot-level split (no data leakage).

| Class | KNN@5 | P@5 | Test Samples |
|-------|-------|-----|-------------|
| Center | 95.2% | 94.6% | 567 |
| Edge-Loc | 93.0% | 92.8% | 817 |
| Donut | 91.4% | 90.9% | 105 |
| Edge-Ring | 90.9% | 90.9% | 1238 |
| Random | 90.2% | 90.7% | 133 |
| Loc | 86.0% | 85.9% | 592 |
| Scratch | 81.5% | 81.5% | 157 |
| Near-full | 76.5% | 80.0% | 17 |

**Evaluation methodology:**
- Split by manufacturing LOT (not by sample) — prevents lot-level leakage
- KNN index built on TRAIN embeddings, queries from TEST
- Oversampling applied ONLY to train set with augmentation (not identical copies)
- No "none" class in retrieval evaluation (handled separately via anomaly detection)

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
| Fine-tune (30 epochs) | CrossEntropy + class weights | 88.6% (biased) | Lot leakage present |
| Triplet (30 epochs) | Hard negative mining | Similar | Lot leakage present |
| **SupCon (60 epochs, honest)** | **Lot-split, balanced, augmented** | **88.1%** | **Final honest result** |

## 📜 License

MIT

## 🙏 Acknowledgments

- WM-811K dataset by MIR Lab, National Taiwan University
- PyTorch, torchvision, scikit-learn, Streamlit communities
