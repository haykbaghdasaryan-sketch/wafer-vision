"""Help text and tooltip registry.

Centralized registry of help text and tooltips used throughout
the WaferVision Streamlit application.
"""

# Tooltip registry: maps identifier keys to help text strings
TOOLTIPS: dict[str, str] = {
    # Metrics
    "knn_accuracy": (
        "K-Nearest Neighbor Accuracy: Fraction of test samples whose K nearest "
        "neighbors in embedding space share the same class label."
    ),
    "recall_at_k": (
        "Recall@K: Fraction of relevant items (same class) found within the "
        "top-K retrieved results."
    ),
    "precision_at_k": (
        "Precision@K: Fraction of the top-K retrieved results that share the "
        "same class as the query."
    ),
    "silhouette_score": (
        "Silhouette Score: Measures how similar a sample is to its own cluster "
        "vs. other clusters. Range [-1, 1], higher is better."
    ),
    "nmi": (
        "Normalized Mutual Information: Measures agreement between predicted "
        "clusters (K-means) and true labels. Range [0, 1], higher is better."
    ),
    "map_score": (
        "Mean Average Precision: Average precision across all queries, "
        "accounting for the rank of relevant items."
    ),
    "separability_index": (
        "Separability Index: Ratio of inter-class distance to intra-class "
        "distance. Higher means better class separation."
    ),
    # Models
    "resnet50": (
        "ResNet50: 50-layer residual network. Embedding dimension: 2048. "
        "Strong baseline for image classification tasks."
    ),
    "efficientnet_b0": (
        "EfficientNet-B0: Compound-scaled efficient network. Embedding "
        "dimension: 1280. Good accuracy-to-compute ratio."
    ),
    "vit_b16": (
        "ViT-B/16: Vision Transformer with 16×16 patch size. Embedding "
        "dimension: 768. Captures global spatial relationships."
    ),
    # Training modes
    "pretrained": (
        "Pretrained (Zero-Shot): Uses the frozen backbone without any "
        "fine-tuning. Embeddings from the penultimate layer."
    ),
    "finetune": (
        "Fine-tuned: Trains with cross-entropy loss and class weights. "
        "Uses the penultimate layer embeddings."
    ),
    "metric_triplet": (
        "Metric Learning (Triplet): Trains with triplet margin loss and "
        "hard negative mining. Uses projection head embeddings."
    ),
    "metric_supcon": (
        "Metric Learning (SupCon): Trains with supervised contrastive loss. "
        "Uses projection head embeddings."
    ),
    "selfsupervised": (
        "Self-Supervised (SimCLR): Trains with NT-Xent loss on augmented "
        "view pairs. No labels required during training."
    ),
    # Visualization
    "umap": (
        "UMAP (Uniform Manifold Approximation and Projection): "
        "Non-linear dimensionality reduction preserving both local and "
        "global structure. Faster than t-SNE for large datasets."
    ),
    "tsne": (
        "t-SNE (t-distributed Stochastic Neighbor Embedding): "
        "Non-linear dimensionality reduction emphasizing local structure. "
        "Good for revealing clusters."
    ),
    # Anomaly detection
    "anomaly_threshold": (
        "Anomaly Threshold (σ): Number of standard deviations above the mean "
        "distance. Samples exceeding this are flagged as anomalies. "
        "Lower values flag more samples."
    ),
    "anomaly_confidence": (
        "Anomaly Confidence: 1 - (threshold/distance), clamped to [0, 1]. "
        "Higher confidence means the sample is further from normal."
    ),
    # Wafer maps
    "wafer_pixel_values": (
        "Wafer map pixel values: 0 = background (no die), "
        "1 = normal (passing die), 2 = defect (failing die)."
    ),
    "defect_classes": (
        "9 defect pattern classes in WM-811K: Center, Donut, Edge-Loc, "
        "Edge-Ring, Loc, Near-full, Random, Scratch, None."
    ),
    # Rate limiting
    "rate_limit": (
        "Rate limiting: Maximum 10 retrieval queries per minute per session "
        "to prevent resource exhaustion."
    ),
}


def get_tooltip(key: str) -> str:
    """Get tooltip text by key.

    Args:
        key: Tooltip identifier (e.g., "knn_accuracy", "resnet50").

    Returns:
        The tooltip text, or a fallback message if key not found.
    """
    return TOOLTIPS.get(key, f"No help available for '{key}'.")
