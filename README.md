# WaferVision

Intelligent wafer defect embedding and retrieval system built on the WM-811K dataset.

## Overview

WaferVision is a research-grade platform for intelligent wafer defect analysis combining:
- Deep learning feature extraction (ResNet50, EfficientNet-B0, ViT-B/16)
- Metric learning (Triplet, SupCon, SimCLR)
- Embedding quality evaluation (7+ metrics with statistical confidence)
- Explainability (Grad-CAM, Attention Rollout)
- Anomaly detection (centroid-based, LOF)
- Interactive Streamlit web application (6 pages)

## Quick Start

```bash
# Install dependencies
make install

# Download dataset
make download

# Preprocess data
make preprocess

# Train a model
make train BACKBONE=resnet50 MODE=finetune

# Extract embeddings
make extract

# Run evaluation benchmark
make evaluate

# Launch web application
make app
```

## Development

```bash
# Run tests
make test

# Lint code
make lint

# Type checking
make typecheck

# Format code
make format

# Clean artifacts
make clean
```

## Project Structure

```
wafer-vision/
├── src/              # Core library (data, models, training, evaluation, anomaly, visualization)
├── app/              # Streamlit web application (6 pages)
├── configs/          # Hydra YAML configuration files
├── scripts/          # CLI pipeline scripts
├── tests/            # Unit tests + property-based tests
├── docs/             # Documentation
├── Makefile          # Development targets
└── pyproject.toml    # Project metadata and dependencies
```

## Requirements

- Python 3.10+
- PyTorch 2.x
- GPU with 8 GB VRAM recommended (CPU inference supported)

## License

MIT
