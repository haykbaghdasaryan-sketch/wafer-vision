# Architecture

## System Overview

WaferVision follows a modular pipeline architecture where data flows through six core engines:

```
Data_Engine → Embedding_Engine → Training_Engine → Evaluation_Suite → Visualization_Engine → Web_Application
```

Retrieval_Engine and Anomaly_Detector consume embeddings as auxiliary subsystems.

## High-Level Data Flow

```
LSWMD.pkl (2 GB)
    │
    ▼
Data_Engine
    ├── RestrictedUnpickler (safe deserialization)
    ├── Validation (pixel values, spatial dims)
    ├── Preprocessing (resize, normalize, 3-channel)
    └── Stratified Split (train 70% / val 15% / test 15%)
    │
    ▼
Training_Engine
    ├── Mode: Fine-tune (CrossEntropy + class weights)
    ├── Mode: Metric Learning (Triplet / SupCon + ProjectionHead)
    ├── Mode: Self-supervised (SimCLR + NT-Xent)
    └── Checkpointing (atomic write, integrity verification)
    │
    ▼
Embedding_Engine
    ├── ResNet50 → 2048D
    ├── EfficientNet-B0 → 1280D
    └── ViT-B/16 → 768D
    │
    ▼
Analysis Layer
    ├── Evaluation_Suite (KNN, Recall, MAP, NMI, Silhouette)
    ├── Retrieval_Engine (brute-force NN, <50ms)
    └── Anomaly_Detector (centroid distance + LOF)
    │
    ▼
Streamlit Web App (6 pages)
```

## Module Structure

```
src/
├── config.py              Pydantic config + Hydra YAML loading
├── exceptions.py          Complete exception hierarchy (19 classes)
├── logging_config.py      Structured JSON logging with run_id
├── registry.py            Generic decorator-based component registry
├── security.py            RestrictedUnpickler, path validation
├── utils.py               Atomic writes, seeding, disk/memory checks
├── data/                  Loading, preprocessing, augmentation, sampling
├── models/                Backbones, projection head, embedding extractor
├── training/              Losses, schedulers, early stopping, checkpoints, trainer
├── evaluation/            Metrics, retrieval, benchmarking
├── anomaly/               Centroid + LOF detection, threshold computation
└── visualization/         Grad-CAM, UMAP, t-SNE, plotting utilities
```

## Key Design Decisions

1. **Registry pattern** for backbones, losses, metrics — extensible without code changes
2. **Atomic writes** for all persistence (checkpoints, embeddings, configs) — prevents corruption
3. **Lot-based splitting** for honest evaluation — prevents manufacturing lot leakage
4. **BalancedBatchSampler** for metric learning — ensures P classes × K samples per batch
5. **OOM recovery** in embedding extraction — auto-halves batch size up to 3 times
6. **Structured logging** with run_id correlation — traceable across components

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch 2.x |
| Metric Learning | SupCon, Triplet (custom implementation) |
| Explainability | pytorch-grad-cam + custom Attention Rollout |
| Dimensionality Reduction | umap-learn, sklearn t-SNE |
| Visualization | Plotly 5.x (WebGL), matplotlib |
| Web Application | Streamlit |
| Configuration | Hydra/OmegaConf + Pydantic |
| Testing | pytest, hypothesis |
