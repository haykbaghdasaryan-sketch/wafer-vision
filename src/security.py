"""Security utilities: restricted unpickling, path validation, upload validation."""

import io
import pickle
from pathlib import Path
from typing import Any, Set

from src.exceptions import ConfigValidationError, DataValidationError


# Allowed types for restricted unpickling.
# Includes both high-level types and internal reconstruction helpers
# that numpy/pandas use during deserialization.
SAFE_PICKLE_TYPES: Set[str] = {
    # Numpy types
    "numpy.ndarray",
    "numpy.dtype",
    "numpy._core.multiarray._reconstruct",  # numpy 2.x array reconstruction
    "numpy.core.multiarray._reconstruct",  # numpy 1.x array reconstruction
    "numpy._core.multiarray.scalar",  # numpy 2.x scalar reconstruction
    "numpy.core.multiarray.scalar",  # numpy 1.x scalar reconstruction
    # Pandas types
    "pandas.core.frame.DataFrame",
    "pandas.core.series.Series",
    "pandas.core.indexes.base.Index",
    "pandas.core.indexes.base._new_Index",
    "pandas.core.indexes.range.RangeIndex",
    "pandas.core.internals.managers.BlockManager",
    "pandas.core.internals.BlockManager",  # older pandas versions
    "pandas.core.internals.blocks.new_block",
    "pandas._libs.internals._unpickle_block",
    "pandas.indexes.base._new_Index",  # older pandas versions
    # Python builtin types
    "builtins.dict",
    "builtins.list",
    "builtins.tuple",
    "builtins.set",
    "builtins.int",
    "builtins.float",
    "builtins.str",
    "builtins.bytes",
    "builtins.bool",
    "builtins.NoneType",
    "builtins.slice",
    "builtins.frozenset",
    "builtins.range",
    "builtins.complex",
    "builtins.bytearray",
    # Collections (sometimes used in pandas internals)
    "collections.OrderedDict",
    # Copyreg reconstruction (used by pickle protocol)
    "copyreg._reconstructor",
    # Numpy scalar types
    "numpy.int64",
    "numpy.float64",
    "numpy.bool_",
    "numpy.object_",
    "numpy.bytes_",
    "numpy.str_",
}


class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows safe types to prevent arbitrary code execution.

    Blocks instantiation of any class not in SAFE_PICKLE_TYPES whitelist.

    Raises:
        pickle.UnpicklingError: If pickle contains unsafe types.
    """

    def find_class(self, module: str, name: str) -> Any:
        """Override to restrict allowed classes.

        Args:
            module: Module containing the class.
            name: Class name.

        Returns:
            The class if safe.

        Raises:
            pickle.UnpicklingError: If class is not in whitelist.
        """
        full_name = f"{module}.{name}"
        if full_name not in SAFE_PICKLE_TYPES:
            raise pickle.UnpicklingError(
                f"Blocked unsafe type: '{full_name}'. "
                f"Only whitelisted types are allowed for deserialization."
            )
        return super().find_class(module, name)


def restricted_loads(data: bytes) -> Any:
    """Safely load pickle data using RestrictedUnpickler.

    Args:
        data: Raw pickle bytes.

    Returns:
        Deserialized Python object (only safe types).

    Raises:
        pickle.UnpicklingError: If unsafe types detected.
    """
    return RestrictedUnpickler(io.BytesIO(data)).load()


def validate_path(path: str, project_root: Path) -> Path:
    """Validate a file path against traversal attacks.

    Rejects paths containing ".." or absolute paths outside project_root.

    Args:
        path: User-provided path string.
        project_root: Root directory of the project.

    Returns:
        Resolved safe Path object.

    Raises:
        ConfigValidationError: If path is unsafe.
    """
    # Reject paths containing ".." components
    if ".." in path:
        raise ConfigValidationError(
            parameter="path",
            value=path,
            valid_range="Must not contain '..' (path traversal)",
        )

    candidate = Path(path)

    # If it's an absolute path, check it's within project_root
    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
            root_resolved = project_root.resolve()
            # Check if resolved path starts with the project root
            resolved.relative_to(root_resolved)
        except ValueError:
            raise ConfigValidationError(
                parameter="path",
                value=path,
                valid_range=f"Must be within project root: {project_root}",
            )
        return resolved

    # Relative path: resolve relative to project_root
    resolved = (project_root / candidate).resolve()
    root_resolved = project_root.resolve()

    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ConfigValidationError(
            parameter="path",
            value=path,
            valid_range=f"Must be within project root: {project_root}",
        )

    return resolved


def validate_upload(
    file_bytes: bytes,
    filename: str,
    max_size_mb: float = 10.0,
    allowed_extensions: set[str] | None = None,
    max_dimensions: tuple[int, int] = (1024, 1024),
) -> None:
    """Validate an uploaded file for size, extension, and dimensions.

    Args:
        file_bytes: Raw file content.
        filename: Original filename for extension check.
        max_size_mb: Maximum allowed file size in megabytes.
        allowed_extensions: Set of allowed file extensions.
            Defaults to {".png", ".jpg", ".jpeg", ".npy"}.
        max_dimensions: Maximum (width, height) for image files.

    Raises:
        DataValidationError: If any validation check fails.
    """
    if allowed_extensions is None:
        allowed_extensions = {".png", ".jpg", ".jpeg", ".npy"}

    # Check file size
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise DataValidationError(
            record_index=-1,
            violation=(
                f"File size {file_size_mb:.2f} MB exceeds maximum allowed "
                f"size of {max_size_mb} MB"
            ),
        )

    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise DataValidationError(
            record_index=-1,
            violation=(
                f"File extension '{ext}' is not allowed. "
                f"Allowed extensions: {sorted(allowed_extensions)}"
            ),
        )

    # Check dimensions for image files
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"}
    if ext in image_extensions:
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(file_bytes))
            width, height = image.size

            max_width, max_height = max_dimensions
            if width > max_width or height > max_height:
                raise DataValidationError(
                    record_index=-1,
                    violation=(
                        f"Image dimensions ({width}x{height}) exceed maximum "
                        f"allowed dimensions ({max_width}x{max_height})"
                    ),
                )
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                record_index=-1,
                violation=f"Failed to read image file: {e}",
            )
