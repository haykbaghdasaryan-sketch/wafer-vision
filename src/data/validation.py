"""Wafer map array validators.

Provides validation functions for wafer map arrays, checking pixel values,
spatial dimensions, and structural integrity of individual records.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Valid pixel values for wafer maps:
# 0 = background/no die, 1 = normal die, 2 = defective die
VALID_PIXEL_VALUES = {0, 1, 2}

# Minimum spatial dimensions for a wafer map to be considered valid
MIN_SPATIAL_DIM = 5


def validate_pixel_values(
    wafer_map: np.ndarray, record_index: int
) -> bool:
    """Validate that a wafer map array contains only values in {0, 1, 2}.

    Args:
        wafer_map: 2D numpy array representing the wafer map.
        record_index: Index of the record for logging purposes.

    Returns:
        True if all pixel values are valid, False otherwise.
        Logs a warning if invalid values are detected.
    """
    unique_values = set(np.unique(wafer_map).astype(int))
    invalid_values = unique_values - VALID_PIXEL_VALUES
    if invalid_values:
        logger.warning(
            "Record %d contains invalid pixel values: %s. "
            "Expected only values in {0, 1, 2}.",
            record_index,
            sorted(invalid_values),
        )
        return False
    return True


def validate_spatial_dimensions(
    wafer_map: np.ndarray, record_index: int, min_dim: int = MIN_SPATIAL_DIM
) -> bool:
    """Validate that a wafer map has sufficient spatial dimensions.

    Wafer maps with dimensions smaller than min_dim×min_dim are considered
    to contain insufficient spatial information for meaningful analysis.

    Args:
        wafer_map: 2D numpy array representing the wafer map.
        record_index: Index of the record for logging purposes.
        min_dim: Minimum acceptable dimension (default: 5).

    Returns:
        True if dimensions are sufficient, False otherwise.
        Logs a warning if dimensions are too small.
    """
    height, width = wafer_map.shape[:2]
    if height < min_dim or width < min_dim:
        logger.warning(
            "Record %d has spatial dimensions %dx%d which is below "
            "minimum %dx%d. Record will be discarded.",
            record_index,
            height,
            width,
            min_dim,
            min_dim,
        )
        return False
    return True


def validate_wafer_map(
    wafer_map: Optional[np.ndarray], record_index: int
) -> tuple[bool, str]:
    """Perform full validation of a wafer map array.

    Checks:
    1. The wafer map is not None and is a numpy ndarray.
    2. The wafer map is 2D.
    3. Spatial dimensions are at least 5×5.
    4. All pixel values are in {0, 1, 2}.

    Args:
        wafer_map: The wafer map array to validate.
        record_index: Index of the record for logging purposes.

    Returns:
        Tuple of (is_valid, reason). If valid, reason is empty string.
        If invalid, reason describes the validation failure.
    """
    # Check for None / not an array
    if wafer_map is None:
        logger.warning("Record %d has a None wafer map.", record_index)
        return False, "wafer_map is None"

    if not isinstance(wafer_map, np.ndarray):
        logger.warning(
            "Record %d wafer map is not a numpy array (type: %s).",
            record_index,
            type(wafer_map).__name__,
        )
        return False, f"wafer_map is not ndarray (type: {type(wafer_map).__name__})"

    # Check dimensionality
    if wafer_map.ndim != 2:
        logger.warning(
            "Record %d wafer map has %d dimensions, expected 2.",
            record_index,
            wafer_map.ndim,
        )
        return False, f"wafer_map has {wafer_map.ndim} dimensions, expected 2"

    # Check spatial dimensions
    if not validate_spatial_dimensions(wafer_map, record_index):
        height, width = wafer_map.shape[:2]
        return False, f"spatial dimensions {height}x{width} below minimum 5x5"

    # Check pixel values
    if not validate_pixel_values(wafer_map, record_index):
        unique_vals = sorted(set(np.unique(wafer_map).astype(int)))
        return False, f"invalid pixel values: {unique_vals}"

    return True, ""
