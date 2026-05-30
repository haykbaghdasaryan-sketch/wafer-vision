"""Adaptive threshold computation for anomaly detection."""

import numpy as np


def compute_threshold(distances: np.ndarray, n_sigma: float = 2.0) -> float:
    """Compute global threshold: mean + n_sigma * std.

    Args:
        distances: Array of distances, shape (N,).
        n_sigma: Number of standard deviations above mean.
            Range [0.0, 5.0]. Default 2.0.
            n_sigma=0 returns just the mean (flags everything above mean).

    Returns:
        Threshold value as a float.

    Raises:
        ValueError: If distances is empty or n_sigma is out of range.
    """
    if distances.size == 0:
        raise ValueError("distances array must not be empty")
    if n_sigma < 0.0 or n_sigma > 5.0:
        raise ValueError(f"n_sigma must be in [0.0, 5.0], got {n_sigma}")

    mean = float(np.mean(distances))
    std = float(np.std(distances))
    return mean + n_sigma * std


def compute_per_class_thresholds(
    distances: np.ndarray,
    assigned_classes: np.ndarray,
    n_sigma: float = 2.0,
) -> dict[int, float]:
    """Compute per-class thresholds: mean_c + n_sigma * std_c for each class.

    For each class c, the threshold is computed from the distances of samples
    assigned to that class.

    Args:
        distances: Array of distances, shape (N,).
        assigned_classes: Array of class assignments, shape (N,).
        n_sigma: Number of standard deviations above mean per class.
            Range [0.0, 5.0]. Default 2.0.

    Returns:
        Dict mapping class_index (int) -> threshold (float).

    Raises:
        ValueError: If arrays are empty or have mismatched lengths.
    """
    if distances.size == 0:
        raise ValueError("distances array must not be empty")
    if distances.shape[0] != assigned_classes.shape[0]:
        raise ValueError(
            f"distances and assigned_classes must have same length, "
            f"got {distances.shape[0]} and {assigned_classes.shape[0]}"
        )
    if n_sigma < 0.0 or n_sigma > 5.0:
        raise ValueError(f"n_sigma must be in [0.0, 5.0], got {n_sigma}")

    unique_classes = np.unique(assigned_classes)
    thresholds: dict[int, float] = {}

    for cls in unique_classes:
        mask = assigned_classes == cls
        cls_distances = distances[mask]
        mean_c = float(np.mean(cls_distances))
        std_c = float(np.std(cls_distances))
        thresholds[int(cls)] = mean_c + n_sigma * std_c

    return thresholds
