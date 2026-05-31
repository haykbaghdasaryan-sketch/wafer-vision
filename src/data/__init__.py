"""Data engine: loading, preprocessing, augmentation, and sampling."""

from src.data.adapter import DatasetAdapter
from src.data.dataset import WaferMapDataset
from src.data.loader import WM811KLoader, WaferRecord, LoadResult, KNOWN_CLASSES
from src.data.validation import (
    validate_pixel_values,
    validate_spatial_dimensions,
    validate_wafer_map,
    VALID_PIXEL_VALUES,
    MIN_SPATIAL_DIM,
)
from src.data.augmentation import WaferAugmentation, EvalTransform, SimCLRAugmentation, MixupAugmentation
from src.data.sampler import BalancedBatchSampler

__all__ = [
    "DatasetAdapter",
    "WaferMapDataset",
    "WM811KLoader",
    "WaferRecord",
    "LoadResult",
    "KNOWN_CLASSES",
    "BalancedBatchSampler",
    "validate_pixel_values",
    "validate_spatial_dimensions",
    "validate_wafer_map",
    "VALID_PIXEL_VALUES",
    "MIN_SPATIAL_DIM",
    "WaferAugmentation",
    "EvalTransform",
    "SimCLRAugmentation",
    "MixupAugmentation",
]
