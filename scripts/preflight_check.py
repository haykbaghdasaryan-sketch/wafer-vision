"""CLI script for pre-training validation and environment checks.

Loads configuration and runs all pre-flight checks to verify the
environment is ready for training before expensive computation begins.

Usage:
    python scripts/preflight_check.py
    python scripts/preflight_check.py --config-dir configs
    python scripts/preflight_check.py backbone=vit_b16 training.mode=metric
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, preflight_check
from src.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command-line arguments.

    Returns:
        Tuple of (parsed known args, remaining Hydra-style overrides).
    """
    parser = argparse.ArgumentParser(
        description="Run pre-training validation checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to configs/ directory (auto-detected if not provided).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code on any non-OK check (not just critical ones).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show failures, suppress OK results.",
    )

    # Use parse_known_args to allow Hydra-style overrides
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> int:
    """Main entry point for preflight checks."""
    args, overrides = parse_args()

    # Setup logging
    run_id = setup_logging(level="INFO", output="console")
    logger.info("Running preflight checks", extra={"run_id": run_id})

    # Load configuration
    config_dir = Path(args.config_dir) if args.config_dir else None
    try:
        config = load_config(overrides=overrides if overrides else None, config_dir=config_dir)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        print(f"\nCONFIGURATION ERROR: {e}")
        print("\nFailed to load configuration. Check your config files and overrides.")
        return 1

    # Run preflight checks
    print(f"\nPreflight Checks")
    print(f"{'=' * 60}")
    print(f"  Backbone:  {config.model.backbone}")
    print(f"  Mode:      {config.train.mode}")
    print(f"  Seed:      {config.seed}")
    print()

    results = preflight_check(config)

    # Display results
    critical_failures = []
    warnings = []
    passed = 0

    for check_name, status in results.items():
        is_ok = status.startswith("OK")
        is_critical = check_name in {
            "dataset_path",
            "checkpoint_dir",
            "output_dir",
            "disk_space",
            "pytorch_version",
        }

        if is_ok:
            passed += 1
            if not args.quiet:
                print(f"  [PASS] {check_name}: {status}")
        elif is_critical:
            critical_failures.append((check_name, status))
            print(f"  [FAIL] {check_name}: {status}")
        else:
            warnings.append((check_name, status))
            print(f"  [WARN] {check_name}: {status}")

    # Summary
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Summary: {passed}/{total} passed", end="")
    if warnings:
        print(f", {len(warnings)} warning(s)", end="")
    if critical_failures:
        print(f", {len(critical_failures)} critical failure(s)", end="")
    print()

    # Additional config summary
    print(f"\nConfiguration Summary:")
    print(f"  Data path:       {config.data.pkl_path}")
    print(f"  Target size:     {config.data.target_size}x{config.data.target_size}")
    print(f"  Backbone:        {config.model.backbone}")
    print(f"  Training mode:   {config.train.mode}")
    print(f"  Learning rate:   {config.train.learning_rate}")
    print(f"  Epochs:          {config.train.num_epochs}")
    print(f"  Batch size:      {config.data.batch_size}")
    print(f"  Checkpoint dir:  {config.train.checkpoint_dir}")

    # Determine exit code
    if critical_failures:
        print(f"\nCritical failures detected. Fix these before training:")
        for name, status in critical_failures:
            print(f"  - {name}: {status}")
        return 1

    if args.strict and warnings:
        print(f"\nStrict mode: warnings treated as failures.")
        return 1

    if not critical_failures:
        print(f"\nAll critical checks passed. Ready to train!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
