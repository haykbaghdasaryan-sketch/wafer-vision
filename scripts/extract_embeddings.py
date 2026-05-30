"""CLI script for embedding extraction from trained checkpoints.

Extracts embeddings using a specified backbone model and training mode,
saving results as .npy files with sidecar JSON metadata.

Usage:
    python scripts/extract_embeddings.py --backbone resnet50 --mode pretrained
    python scripts/extract_embeddings.py --backbone vit_b16 --mode finetune --batch-size 32
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.dataset import WaferMapDataset
from src.logging_config import setup_logging
from src.models.embedding_extractor import EmbeddingExtractor
from src.models.factory import get_backbone
from src.utils import set_all_seeds

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a trained backbone model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit_b16"],
        help="Backbone architecture to use.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrained",
        choices=["pretrained", "finetune", "metric", "selfsupervised"],
        help="Training mode (determines which checkpoint to load).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing preprocessed .pt split files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/embeddings",
        help="Directory to save extracted embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for extraction (auto-halved on OOM).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for computation.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to extract embeddings from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint file (overrides auto-detection).",
    )
    return parser.parse_args()


def find_checkpoint(backbone: str, mode: str, checkpoint_dir: Path) -> Path | None:
    """Find the best checkpoint for a given backbone and mode.

    Args:
        backbone: Backbone name.
        mode: Training mode.
        checkpoint_dir: Directory to search for checkpoints.

    Returns:
        Path to checkpoint file, or None if not found.
    """
    if not checkpoint_dir.exists():
        return None

    # Look for best checkpoint first, then latest
    patterns = [
        f"{backbone}_{mode}_best.pth",
        f"checkpoint_best.pth",
        f"{backbone}_{mode}_latest.pth",
        f"checkpoint_epoch_*.pth",
    ]

    for pattern in patterns:
        matches = sorted(checkpoint_dir.glob(pattern))
        if matches:
            return matches[-1]  # Latest by name

    return None


def main() -> int:
    """Main entry point for embedding extraction."""
    args = parse_args()

    # Setup logging
    run_id = setup_logging(level="INFO", output="both", log_dir=Path("logs"))
    logger.info(
        "Starting embedding extraction",
        extra={
            "backbone": args.backbone,
            "mode": args.mode,
            "split": args.split,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )

    # Set seeds for reproducibility
    set_all_seeds(args.seed)

    # Verify data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run 'make preprocess' first to generate preprocessed data.")
        return 1

    # Load dataset
    try:
        dataset = WaferMapDataset.from_split(
            split_dir=data_dir,
            split_name=args.split,
            transform=None,  # No augmentation for extraction
        )
        logger.info(f"Loaded {len(dataset)} samples from {args.split} split")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        print(f"ERROR: Failed to load dataset from {data_dir}: {e}")
        return 1

    # Create dataloader (no shuffle to maintain ordering)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Instantiate backbone
    try:
        pretrained = args.mode == "pretrained"
        backbone = get_backbone(args.backbone, pretrained=pretrained)
        logger.info(
            f"Loaded backbone: {args.backbone} "
            f"(embedding_dim={backbone.embedding_dim})"
        )
    except Exception as e:
        logger.error(f"Failed to create backbone: {e}")
        print(f"ERROR: Failed to create backbone '{args.backbone}': {e}")
        return 1

    # Load checkpoint if not pretrained mode
    if args.mode != "pretrained":
        checkpoint_path = (
            Path(args.checkpoint) if args.checkpoint
            else find_checkpoint(args.backbone, args.mode, Path("outputs/checkpoints"))
        )

        if checkpoint_path and checkpoint_path.exists():
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if "model_state_dict" in checkpoint:
                    backbone.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    backbone.load_state_dict(checkpoint, strict=False)
                logger.info(f"Loaded checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load checkpoint {checkpoint_path}: {e}. "
                    "Using pretrained weights instead."
                )
        else:
            logger.warning(
                f"No checkpoint found for {args.backbone}/{args.mode}. "
                "Using pretrained weights."
            )

    # Setup output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{args.backbone}_{args.mode}_{args.split}.npy"
    output_path = output_dir / output_filename

    # Extract embeddings
    extractor = EmbeddingExtractor(
        backbone=backbone,
        batch_size=args.batch_size,
        device=args.device,
    )

    start_time = time.time()
    try:
        saved_path = extractor.extract(
            dataloader=dataloader,
            output_path=output_path,
            metadata={
                "backbone": args.backbone,
                "training_mode": args.mode,
                "split": args.split,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "run_id": run_id,
            },
        )
        elapsed = time.time() - start_time

        # Report results
        embeddings = np.load(str(saved_path))
        logger.info(
            f"Extraction complete in {elapsed:.1f}s: "
            f"shape={embeddings.shape}, saved to {saved_path}"
        )
        print(f"\nExtraction complete!")
        print(f"  Backbone:  {args.backbone}")
        print(f"  Mode:      {args.mode}")
        print(f"  Split:     {args.split}")
        print(f"  Shape:     {embeddings.shape}")
        print(f"  Time:      {elapsed:.1f}s")
        print(f"  Output:    {saved_path}")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"ERROR: Embedding extraction failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
