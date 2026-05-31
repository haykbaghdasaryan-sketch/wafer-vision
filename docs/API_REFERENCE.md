# API Reference

## Core Modules

### `src.models`

```python
from src.models import get_backbone, BaseBackbone, ProjectionHead, EmbeddingExtractor

# Get a backbone by name
model = get_backbone("resnet50", pretrained=True, l2_normalize=False)
# Supported: "resnet50" (2048D), "efficientnet_b0" (1280D), "vit_b16" (768D)

# Forward pass
embeddings = model(images)  # (B, 3, H, W) → (B, D)

# Projection head for metric learning
head = ProjectionHead(input_dim=2048, hidden_dim=512, output_dim=128)
projections = head(embeddings)  # (B, D) → (B, 128), L2-normalized

# Embedding extraction with OOM recovery
extractor = EmbeddingExtractor(backbone=model, batch_size=64, device="cuda")
extractor.extract(dataloader, output_path, metadata={...})
```

### `src.training`

```python
from src.training.trainer import Trainer, TrainingConfig
from src.training.losses import TripletMarginLossWithMining, SupervisedContrastiveLoss, NTXentLoss
from src.training.early_stopping import EarlyStopping
from src.training.scheduler import WarmupCosineScheduler

# Training configuration
config = TrainingConfig(
    mode="metric",          # "pretrained" | "finetune" | "metric" | "selfsupervised"
    loss_name="supcon",     # "crossentropy" | "triplet" | "supcon" | "ntxent"
    num_epochs=60,
    learning_rate=5e-5,
    temperature=0.07,       # For SupCon/NT-Xent
    margin=0.3,             # For Triplet
    device="cuda",
)

# Train
trainer = Trainer(config=config, model=model, train_loader=tl, val_loader=vl)
result = trainer.train()  # Returns TrainingResult
```

### `src.evaluation`

```python
from src.evaluation.metrics import MetricsComputer, EmbeddingMetrics
from src.evaluation.retrieval import RetrievalEngine, RetrievalResult
from src.evaluation.benchmark import BenchmarkRunner

# Compute all metrics
mc = MetricsComputer(embeddings, labels, distance_metric="euclidean")
metrics = mc.compute_all(bootstrap_ci=True)
# metrics.knn_accuracy, metrics.recall_at_k, metrics.nmi, etc.

# Retrieval
engine = RetrievalEngine(embeddings, labels, distance_metric="euclidean")
result = engine.query_by_index(idx=0, k=10)  # Returns sorted neighbors

# Benchmark comparison
runner = BenchmarkRunner(results_dir=Path("outputs/metrics"))
runner.add_result("resnet50", "supcon", metrics)
runner.generate_csv(path); runner.generate_bar_chart(path)
```

### `src.anomaly`

```python
from src.anomaly.detector import AnomalyDetector

detector = AnomalyDetector(train_embeddings, train_labels)
anomalies = detector.detect(test_embeddings, n_std=2.0)
# Returns list of AnomalyResult sorted by distance descending
```

### `src.data`

```python
from src.data.loader import WM811KLoader
from src.data.preprocessing import WaferPreprocessor
from src.data.augmentation import WaferAugmentation, SimCLRAugmentation, EvalTransform
from src.data.sampler import BalancedBatchSampler
from src.data.dataset import WaferMapDataset

# Load
loader = WM811KLoader()
records = loader.load(Path("data/raw/LSWMD.pkl"))

# Preprocess
preprocessor = WaferPreprocessor(target_size=64)
splits = preprocessor.create_splits(records, seed=42)

# Dataset
dataset = WaferMapDataset.from_split("data/processed", split_name="train", transform=aug)

# Balanced sampler for metric learning
sampler = BalancedBatchSampler(labels, p_classes=9, k_samples=4)
```

### `src.visualization`

```python
from src.visualization.gradcam import GradCAMVisualizer
from src.visualization.umap_viz import UMAPVisualizer
from src.visualization.tsne_viz import TSNEVisualizer

# Grad-CAM
viz = GradCAMVisualizer(model)
heatmap = viz.compute_explanation(image)  # (H, W), values [0, 1]
overlay = GradCAMVisualizer.create_overlay(wafer_map, heatmap, alpha=0.5)

# UMAP
umap = UMAPVisualizer(n_components=2, cache_dir="outputs/cache")
projection = umap.fit_transform(embeddings, labels)
fig = umap.create_figure(projection, labels, class_names=[...])
```

## Configuration

All config managed via Hydra YAML + Pydantic validation:

```python
from src.config import load_config, WaferVisionConfig

config = load_config(overrides=["training.lr=1e-3", "model.backbone=vit_b16"])
```

See `configs/` directory for all YAML files.
