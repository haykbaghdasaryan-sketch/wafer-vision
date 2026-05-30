"""CLI script to verify deterministic results across runs.

Runs embedding extraction twice with the same seed and compares outputs
for exact match to verify reproducibility of the pipeline.

Usage:
    python scripts/verify_reproducibility.py
    python scripts/verify_reproducibility.py --backbone resnet50 --seed 42
"""

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logging_config import setup_logging
from src.models.embedding_extractor import EmbeddingExtractor
from src.models.factory import get_backbone
from src.utils import set_all_seeds

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Verify reproducibility of embedding extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit_b16"],
        help="Backbone architecture to test.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing preprocessed .pt split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility verification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for computation (CPU recommended for determinism).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use for verification.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples to use (for faster verification).",
    )
    return parser.parse_args()


def extract_with_seed(
    backbone_name: str,
    data_dir: Path,
    split: str,
    seed: int,
    batch_size: int,
    device: str,
    output_path: Path,
    max_samples: int | None = None,
) -> np.ndarray:
    """Run a single extraction pass with the given seed.

    Args:
        backbone_name: Name of the backbone model.
        data_dir: Directory with preprocessed data.
        split: Dataset split name.
        seed: Random seed to use.
        batch_size: Batch size for extraction.
        device: Device for computation.
        output_path: Path to save embeddings.
        max_samples: Max samples to process (None for all).

    Returns:
        Numpy array of extracted embeddings.
    """
    from src.data.dataset import WaferMapDataset

    # Set seeds
    set_all_seeds(seed)

    # Load dataset
    dataset = WaferMapDataset.from_split(
        split_dir=data_dir,
        split_name=split,
        transform=None,
    )

    # Subset if requested
    if max_samples and len(dataset) > max_samples:
        dataset = torch.utils.data.Subset(dataset, list(range(max_samples)))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create backbone
    backbone = get_backbone(backbone_name, pretrained=True)

    # Extract
    extractor = EmbeddingExtractor(
        backbone=backbone,
        batch_size=batch_size,
        device=device,
    )

    extractor.extract(
        dataloader=dataloader,
        output_path=output_path,
        metadata={"seed": seed, "run": "reproducibility_check"},
    )

    return np.load(str(output_path))


def main() -> int:
    """Main entry point for reproducibility verification."""
    args = parse_args()

    # Setup logging
    run_id = setup_logging(level="INFO", output="both", log_dir=Path("logs"))
    logger.info("Starting reproducibility verification", extra={"run_id": run_id})

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run 'make preprocess' first to generate preprocessed data.")
        return 1

    print(f"Reproducibility Verification")
    print(f"{'=' * 50}")
    print(f"  Backbone:     {args.backbone}")
    print(f"  Seed:         {args.seed}")
    print(f"  Device:       {args.device}")
    print(f"  Max samples:  {args.max_samples}")
    print()

    checks_passed = 0
    checks_failed = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Check 1: Extraction determinism
        print("Check 1: Embedding extraction determinism")
        print(f"  Running extraction pass 1...")
        start = time.time()
        try:
            embeddings_1 = extract_with_seed(
                backbone_name=args.backbone,
                data_dir=data_dir,
                split=args.split,
                seed=args.seed,
                batch_size=args.batch_size,
                device=args.device,
                output_path=tmp_path / "run1.npy",
                max_samples=args.max_samples,
            )
            t1 = time.time() - start
            print(f"  Pass 1 complete ({t1:.1f}s): shape={embeddings_1.shape}")
        except Exception as e:
            logger.error(f"Pass 1 failed: {e}")
            print(f"  FAILED: {e}")
            return 1

        print(f"  Running extraction pass 2...")
        start = time.time()
        try:
            embeddings_2 = extract_with_seed(
                backbone_name=args.backbone,
                data_dir=data_dir,
                split=args.split,
                seed=args.seed,
                batch_size=args.batch_size,
                device=args.device,
                output_path=tmp_path / "run2.npy",
                max_samples=args.max_samples,
            )
            t2 = time.time() - start
            print(f"  Pass 2 complete ({t2:.1f}s): shape={embeddings_2.shape}")
        except Exception as e:
            logger.error(f"Pass 2 failed: {e}")
            print(f"  FAILED: {e}")
            return 1

        # Compare
        if embeddings_1.shape != embeddings_2.shape:
            print(f"  FAIL: Shape mismatch ({embeddings_1.shape} vs {embeddings_2.shape})")
            checks_failed += 1
        elif np.array_equal(embeddings_1, embeddings_2):
            print(f"  PASS: Exact match (bitwise identical)")
            checks_passed += 1
        else:
            max_diff = np.max(np.abs(embeddings_1 - embeddings_2))
            mean_diff = np.mean(np.abs(embeddings_1 - embeddings_2))
            if max_diff < 1e-6:
                print(f"  PASS: Near-identical (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                checks_passed += 1
            else:
                print(f"  FAIL: Outputs differ (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
                checks_failed += 1

        # Check 2: Seed reset produces same random state
        print("\nCheck 2: Seed reset consistency")
        set_all_seeds(args.seed)
        state_1 = torch.rand(100)
        np_state_1 = np.random.rand(100)

        set_all_seeds(args.seed)
        state_2 = torch.rand(100)
        np_state_2 = np.random.rand(100)

        torch_match = torch.equal(state_1, state_2)
        numpy_match = np.array_equal(np_state_1, np_state_2)

        if torch_match and numpy_match:
            print(f"  PASS: PyTorch and NumPy random states are deterministic")
            checks_passed += 1
        else:
            print(f"  FAIL: Random states differ (torch={torch_match}, numpy={numpy_match})")
            checks_failed += 1

        # Check 3: Model weights are deterministic after reload
        print("\nCheck 3: Model weight determinism")
        set_all_seeds(args.seed)
        model_1 = get_backbone(args.backbone, pretrained=True)
        set_all_seeds(args.seed)
        model_2 = get_backbone(args.backbone, pretrained=True)

        weights_match = True
        for (n1, p1), (n2, p2) in zip(
            model_1.named_parameters(), model_2.named_parameters()
        ):
            if not torch.equal(p1, p2):
                weights_match = False
                print(f"  Mismatch in parameter: {n1}")
                break

        if weights_match:
            print(f"  PASS: Model weights are identical across loads")
            checks_passed += 1
        else:
            print(f"  FAIL: Model weights differ between loads")
            checks_failed += 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Results: {checks_passed} passed, {checks_failed} failed")
    if checks_failed == 0:
        print("All reproducibility checks PASSED.")
        return 0
    else:
        print("Some reproducibility checks FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
