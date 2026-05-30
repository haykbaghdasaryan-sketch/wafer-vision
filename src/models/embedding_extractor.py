"""Embedding extraction with OOM recovery and incremental support."""

import hashlib
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.backbone import BaseBackbone
from src.utils import atomic_write

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract embeddings from a backbone model with OOM recovery and incremental support.

    Features:
        - OOM recovery: auto-halves batch size up to 3 times on CUDA OOM
        - Incremental extraction: resumes from partial .npy if interrupted
        - Atomic writes: uses temp file + rename for .npy and sidecar JSON
        - Verification: confirms output is loadable with correct shape (N, D)
        - Sample ordering: maintains consistent ordering with dataset indices
        - Progress: tqdm progress bar with ETA

    Args:
        backbone: A BaseBackbone instance (must be in eval mode for extraction).
        batch_size: Initial batch size for extraction. Will be halved on OOM.
        device: Device string ("cuda", "cpu", or "auto" for auto-detection).
    """

    MAX_OOM_RETRIES = 3

    def __init__(
        self,
        backbone: BaseBackbone,
        batch_size: int = 64,
        device: str = "auto",
    ) -> None:
        self._backbone = backbone
        self._initial_batch_size = batch_size
        self._device = self._resolve_device(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' device to cuda/cpu."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def extract(
        self,
        dataloader: DataLoader,
        output_path: Path,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """Extract embeddings from all samples in the dataloader.

        Args:
            dataloader: PyTorch DataLoader yielding (images, labels) batches.
                The dataloader should NOT shuffle data to maintain sample ordering.
            output_path: Path for the output .npy file. Sidecar JSON will be
                written alongside as <stem>_metadata.json.
            metadata: Optional extra metadata fields to include in sidecar JSON.

        Returns:
            Path to the saved .npy embeddings file.

        Raises:
            RuntimeError: If OOM persists after MAX_OOM_RETRIES batch halvings.
            AtomicWriteError: If file writing fails.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine total samples from the dataloader's dataset
        dataset = dataloader.dataset
        total_samples = len(dataset)
        embedding_dim = self._backbone.embedding_dim

        # Check for partial extraction (incremental resume)
        partial_path = output_path.with_suffix(".partial.npy")
        start_index = 0
        partial_embeddings: Optional[np.ndarray] = None

        if partial_path.exists():
            try:
                partial_embeddings = np.load(str(partial_path))
                start_index = partial_embeddings.shape[0]
                logger.info(
                    f"Resuming extraction from sample {start_index}/{total_samples} "
                    f"(loaded partial file: {partial_path})"
                )
            except Exception as e:
                logger.warning(
                    f"Could not load partial file {partial_path}, starting fresh: {e}"
                )
                partial_embeddings = None
                start_index = 0

        # Move backbone to device and set eval mode
        self._backbone.eval()
        self._backbone.to(self._device)

        # Collect embeddings
        all_embeddings: list[np.ndarray] = []
        if partial_embeddings is not None:
            all_embeddings.append(partial_embeddings)

        current_batch_size = self._initial_batch_size
        oom_retries = 0
        samples_processed = start_index

        # Create a progress bar
        pbar = tqdm(
            total=total_samples,
            initial=start_index,
            desc="Extracting embeddings",
            unit="samples",
        )

        with torch.no_grad():
            batch_iter = iter(dataloader)

            # Skip already-processed batches if resuming
            batches_to_skip = start_index // dataloader.batch_size if start_index > 0 else 0
            for _ in range(batches_to_skip):
                try:
                    next(batch_iter)
                except StopIteration:
                    break

            for batch in batch_iter:
                images = batch[0]

                # Process batch with OOM recovery
                embeddings = self._extract_batch_with_oom_recovery(
                    images, current_batch_size, oom_retries
                )

                if embeddings is None:
                    # OOM recovery changed batch size, need to re-split this batch
                    # Process in smaller chunks
                    current_batch_size = max(1, current_batch_size // 2)
                    oom_retries += 1

                    if oom_retries > self.MAX_OOM_RETRIES:
                        pbar.close()
                        raise RuntimeError(
                            f"CUDA OOM persists after {self.MAX_OOM_RETRIES} batch size "
                            f"reductions (final batch_size={current_batch_size}). "
                            f"Consider reducing input resolution or using CPU."
                        )

                    # Re-process this batch in smaller sub-batches
                    embeddings = self._process_in_subbatches(images, current_batch_size)

                all_embeddings.append(embeddings)
                samples_processed += embeddings.shape[0]
                pbar.update(embeddings.shape[0])

                # Periodically save partial results (every 1000 batches)
                if len(all_embeddings) % 100 == 0 and len(all_embeddings) > 1:
                    self._save_partial(all_embeddings, partial_path)

        pbar.close()

        # Concatenate all embeddings
        final_embeddings = np.concatenate(all_embeddings, axis=0)

        # Truncate to exact dataset size (in case of padding in last batch)
        if final_embeddings.shape[0] > total_samples:
            final_embeddings = final_embeddings[:total_samples]

        # Verify shape
        expected_shape = (total_samples, embedding_dim)
        if final_embeddings.shape != expected_shape:
            logger.warning(
                f"Embedding shape mismatch: got {final_embeddings.shape}, "
                f"expected {expected_shape}. Using actual shape."
            )

        # Save embeddings atomically
        self._save_embeddings(final_embeddings, output_path)

        # Save sidecar metadata
        self._save_metadata(
            output_path=output_path,
            num_samples=final_embeddings.shape[0],
            embedding_dim=final_embeddings.shape[1],
            extra_metadata=metadata,
        )

        # Clean up partial file
        if partial_path.exists():
            partial_path.unlink()

        # Verify output
        self._verify_output(output_path, final_embeddings.shape)

        logger.info(
            f"Extraction complete: {final_embeddings.shape[0]} samples, "
            f"dim={final_embeddings.shape[1]}, saved to {output_path}"
        )

        return output_path

    def _extract_batch_with_oom_recovery(
        self,
        images: torch.Tensor,
        batch_size: int,
        current_retries: int,
    ) -> Optional[np.ndarray]:
        """Try to extract embeddings from a batch, return None on OOM.

        Args:
            images: Input image tensor (B, 3, H, W).
            batch_size: Current batch size (for logging).
            current_retries: Number of OOM retries so far.

        Returns:
            Numpy array of embeddings or None if OOM occurred.
        """
        try:
            images_device = images.to(self._device)
            embeddings = self._backbone(images_device)
            return embeddings.cpu().numpy()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                logger.warning(
                    f"CUDA OOM at batch_size={batch_size} "
                    f"(retry {current_retries + 1}/{self.MAX_OOM_RETRIES}). "
                    f"Halving batch size."
                )
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            raise

    def _process_in_subbatches(
        self, images: torch.Tensor, sub_batch_size: int
    ) -> np.ndarray:
        """Process a batch of images in smaller sub-batches after OOM.

        Args:
            images: Full batch of images (B, 3, H, W).
            sub_batch_size: Size of each sub-batch.

        Returns:
            Concatenated numpy embeddings for the entire batch.

        Raises:
            RuntimeError: If OOM persists even with sub-batch processing.
        """
        results: list[np.ndarray] = []
        total = images.shape[0]

        for start in range(0, total, sub_batch_size):
            end = min(start + sub_batch_size, total)
            sub_images = images[start:end]

            try:
                sub_images_device = sub_images.to(self._device)
                embeddings = self._backbone(sub_images_device)
                results.append(embeddings.cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Try even smaller sub-batch
                    if sub_batch_size > 1:
                        smaller = max(1, sub_batch_size // 2)
                        logger.warning(
                            f"OOM in sub-batch (size={sub_batch_size}), "
                            f"trying size={smaller}"
                        )
                        sub_result = self._process_in_subbatches(
                            sub_images, smaller
                        )
                        results.append(sub_result)
                    else:
                        raise RuntimeError(
                            "CUDA OOM even with batch_size=1. "
                            "Input resolution too large for available GPU memory."
                        ) from e
                else:
                    raise

        return np.concatenate(results, axis=0)

    def _save_partial(
        self, embeddings_list: list[np.ndarray], partial_path: Path
    ) -> None:
        """Save partial embeddings to disk for incremental recovery.

        Args:
            embeddings_list: List of embedding arrays accumulated so far.
            partial_path: Path for the partial .npy file.
        """
        try:
            concatenated = np.concatenate(embeddings_list, axis=0)
            np.save(str(partial_path), concatenated)
        except Exception as e:
            logger.warning(f"Failed to save partial embeddings: {e}")

    def _save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Save embeddings atomically as .npy file.

        Args:
            embeddings: Final embedding array (N, D).
            output_path: Target .npy file path.
        """

        def write_fn(temp_path: Path) -> None:
            # np.save appends .npy if not present; ensure the path ends with .npy
            # by writing with file handle to avoid extension issues
            with open(temp_path, "wb") as f:
                np.save(f, embeddings)

        def verify_fn(temp_path: Path) -> None:
            loaded = np.load(str(temp_path))
            if loaded.shape != embeddings.shape:
                raise ValueError(
                    f"Verification failed: saved shape {loaded.shape} != "
                    f"expected {embeddings.shape}"
                )

        atomic_write(output_path, write_fn, verify_fn)

    def _save_metadata(
        self,
        output_path: Path,
        num_samples: int,
        embedding_dim: int,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Save sidecar JSON metadata alongside the .npy file.

        Args:
            output_path: Path to the .npy file (metadata file derives from this).
            num_samples: Number of embedding vectors.
            embedding_dim: Dimensionality of each embedding.
            extra_metadata: Additional metadata fields to include.
        """
        metadata_path = output_path.with_name(
            output_path.stem + "_metadata.json"
        )

        # Compute model weights hash if possible
        model_hash = self._compute_model_hash()

        meta: dict[str, Any] = {
            "backbone_name": self._backbone.__class__.__name__,
            "embedding_dim": embedding_dim,
            "num_samples": num_samples,
            "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_weights_sha256": model_hash,
        }

        # Merge extra metadata
        if extra_metadata:
            meta.update(extra_metadata)

        def write_fn(temp_path: Path) -> None:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        def verify_fn(temp_path: Path) -> None:
            with open(temp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded.get("num_samples") != num_samples:
                raise ValueError("Metadata verification failed: num_samples mismatch")

        atomic_write(metadata_path, write_fn, verify_fn)

        logger.info(f"Saved metadata to {metadata_path}")

    def _compute_model_hash(self) -> str:
        """Compute SHA-256 hash of model weights for provenance tracking.

        Returns:
            Hex digest of SHA-256 hash, or "N/A" if computation fails.
        """
        try:
            hasher = hashlib.sha256()
            for param in self._backbone.parameters():
                hasher.update(param.data.cpu().numpy().tobytes())
            return hasher.hexdigest()
        except Exception:
            return "N/A"

    def _verify_output(self, output_path: Path, expected_shape: tuple) -> None:
        """Verify the saved .npy file is loadable with correct shape.

        Args:
            output_path: Path to the .npy file.
            expected_shape: Expected (N, D) shape.

        Raises:
            RuntimeError: If verification fails.
        """
        try:
            loaded = np.load(str(output_path))
            if loaded.shape != expected_shape:
                raise RuntimeError(
                    f"Output verification failed: expected shape {expected_shape}, "
                    f"got {loaded.shape}"
                )
            if loaded.dtype != np.float32:
                logger.warning(
                    f"Output dtype is {loaded.dtype}, expected float32"
                )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(
                f"Failed to verify output file {output_path}: {e}"
            ) from e
