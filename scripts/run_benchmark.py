"""CLI script for automated benchmarking of all models.

Discovers all .npy embedding files in the outputs directory,
runs MetricsComputer on each, and produces CSV, LaTeX, and bar chart outputs.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --embeddings-dir outputs/embeddings --output-dir outputs/metrics
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.metrics import MetricsComputer
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run benchmarking on all available embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="outputs/embeddings",
        help="Directory containing .npy embedding files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/metrics",
        help="Directory to save benchmark results.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing preprocessed data (for labels).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split that embeddings correspond to.",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric for nearest-neighbor queries.",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip bootstrap confidence interval computation (faster).",
    )
    return parser.parse_args()


def discover_embeddings(embeddings_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover all .npy embedding files and parse their model/mode info.

    Expected filename format: {backbone}_{mode}_{split}.npy
    e.g., resnet50_finetune_test.npy

    Args:
        embeddings_dir: Directory to search.

    Returns:
        List of (model_name, training_mode, file_path) tuples.
    """
    if not embeddings_dir.exists():
        return []

    results = []
    for npy_file in sorted(embeddings_dir.glob("*.npy")):
        # Skip partial files
        if "partial" in npy_file.stem:
            continue

        parts = npy_file.stem.split("_")
        if len(parts) >= 3:
            # Handle backbone names with underscores (e.g., efficientnet_b0)
            # Try to find the mode by matching known modes
            known_modes = {"pretrained", "finetune", "metric", "selfsupervised"}
            known_splits = {"train", "val", "test"}

            model_parts = []
            mode = None
            split = None

            for part in parts:
                if part in known_splits and split is None:
                    split = part
                elif part in known_modes and mode is None:
                    mode = part
                elif split is None and mode is None:
                    model_parts.append(part)

            if model_parts and mode:
                model_name = "_".join(model_parts)
                results.append((model_name, mode, npy_file))
            else:
                logger.warning(
                    f"Could not parse filename: {npy_file.name}. "
                    "Expected format: {{backbone}}_{{mode}}_{{split}}.npy"
                )
        else:
            logger.warning(
                f"Skipping file with unexpected name format: {npy_file.name}"
            )

    return results


def load_labels(data_dir: Path, split: str) -> np.ndarray | None:
    """Load labels for the specified split.

    Args:
        data_dir: Directory containing preprocessed .pt files.
        split: Split name (train/val/test).

    Returns:
        Numpy array of labels, or None if not found.
    """
    import torch

    labels_path = data_dir / f"{split}_labels.pt"
    if not labels_path.exists():
        return None

    labels_tensor = torch.load(labels_path, weights_only=True)
    return labels_tensor.numpy()


def main() -> int:
    """Main entry point for benchmarking."""
    args = parse_args()

    # Setup logging
    run_id = setup_logging(level="INFO", output="both", log_dir=Path("logs"))
    logger.info("Starting benchmark run", extra={"run_id": run_id})

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)

    # Discover embedding files
    embedding_files = discover_embeddings(embeddings_dir)
    if not embedding_files:
        logger.warning(f"No embedding files found in {embeddings_dir}")
        print(f"No embedding files found in: {embeddings_dir}")
        print("Run 'make extract' first to generate embeddings.")
        return 1

    print(f"Found {len(embedding_files)} embedding file(s):")
    for model_name, mode, path in embedding_files:
        print(f"  - {model_name}/{mode}: {path.name}")

    # Load labels
    labels = load_labels(data_dir, args.split)
    if labels is None:
        logger.error(f"Labels not found in {data_dir} for split '{args.split}'")
        print(f"ERROR: Labels file not found: {data_dir}/{args.split}_labels.pt")
        print("Run 'make preprocess' first to generate preprocessed data.")
        return 1

    print(f"\nLabels loaded: {len(labels)} samples, {len(np.unique(labels))} classes")

    # Setup benchmark runner
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(results_dir=output_dir)

    # Evaluate each embedding file
    total_start = time.time()
    successful = 0
    failed = 0

    for model_name, mode, emb_path in embedding_files:
        print(f"\nEvaluating: {model_name}/{mode}...")
        try:
            embeddings = np.load(str(emb_path))

            # Verify shape compatibility with labels
            if embeddings.shape[0] != labels.shape[0]:
                logger.warning(
                    f"Shape mismatch for {model_name}/{mode}: "
                    f"embeddings={embeddings.shape[0]}, labels={labels.shape[0]}. Skipping."
                )
                print(f"  SKIP: Shape mismatch (embeddings={embeddings.shape[0]}, labels={labels.shape[0]})")
                failed += 1
                continue

            # Compute metrics
            start = time.time()
            computer = MetricsComputer(
                embeddings=embeddings,
                labels=labels,
                distance_metric=args.distance_metric,
            )
            metrics = computer.compute_all(bootstrap_ci=not args.no_bootstrap)
            elapsed = time.time() - start

            # Add to benchmark
            runner.add_result(model_name, mode, metrics)

            print(f"  OK ({elapsed:.1f}s) - KNN@5={metrics.knn_accuracy.get(5, 0):.4f}")
            successful += 1

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}/{mode}: {e}")
            print(f"  FAILED: {e}")
            failed += 1

    # Generate outputs
    if successful > 0:
        print(f"\n{'=' * 60}")
        print(f"Generating benchmark reports...")

        csv_path = output_dir / "benchmark_results.csv"
        runner.generate_csv(csv_path)
        print(f"  CSV:     {csv_path}")

        latex_path = output_dir / "benchmark_table.tex"
        runner.generate_latex(latex_path)
        print(f"  LaTeX:   {latex_path}")

        chart_path = output_dir / "benchmark_chart.png"
        runner.generate_bar_chart(chart_path)
        print(f"  Chart:   {chart_path}")

        # Report best per metric
        best_per_metric = runner.get_best_per_metric()
        print(f"\nBest per metric:")
        for metric_name, result in best_per_metric.items():
            print(f"  {metric_name}: {result.model_name}/{result.training_mode}")

        total_elapsed = time.time() - total_start
        print(f"\nBenchmark complete: {successful} evaluated, {failed} failed, {total_elapsed:.1f}s total")
    else:
        print("\nNo models were successfully evaluated.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
