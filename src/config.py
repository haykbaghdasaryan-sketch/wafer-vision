"""Pydantic config models + Hydra integration.

Provides structured configuration with validation for the WaferVision pipeline.
All hyperparameters are managed through Hydra/OmegaConf YAML files with Pydantic
schema validation, ensuring experiments are reproducible and configuration errors
are caught early before any computation begins.

Usage:
    from src.config import load_config, preflight_check

    config = load_config(overrides=["training.lr=1e-3", "model.backbone=vit_b16"])
    checks = preflight_check(config)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.exceptions import ConfigConflictError, ConfigValidationError


class DataConfig(BaseModel):
    """Data pipeline configuration.

    Controls dataset loading, preprocessing, splitting, and augmentation.
    """

    pkl_path: str = "data/raw/LSWMD.pkl"
    processed_dir: str = "data/processed"
    target_size: int = Field(default=64, ge=32, le=224)
    train_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)
    augmentation_strength: Literal["light", "medium", "heavy"] = "medium"
    batch_size: int = Field(default=64, ge=1, le=1024)
    num_workers: int = Field(default=4, ge=0, le=16)
    noise_std: float = Field(default=0.01, ge=0.001, le=0.1)

    @field_validator("target_size")
    @classmethod
    def validate_target_size(cls, v: int) -> int:
        """Validate resolution in [32, 224]."""
        if not (32 <= v <= 224):
            raise ValueError(f"Target size must be in range [32, 224], got: {v}")
        return v

    @model_validator(mode="after")
    def validate_ratios_sum(self) -> "DataConfig":
        """Validate train + val + test = 1.0 ± 0.01."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0 (±0.01), got {total:.3f}"
            )
        return self


class ModelConfig(BaseModel):
    """Model architecture configuration.

    Controls backbone selection, pretrained weights, and embedding normalization.
    """

    backbone: str = "resnet50"
    pretrained: bool = True
    l2_normalize: bool = False
    embedding_dim: Optional[int] = None
    projection_hidden: int = Field(default=512, ge=64, le=2048)
    projection_output: int = Field(default=128, ge=32, le=512)

    @field_validator("backbone")
    @classmethod
    def normalize_backbone_name(cls, v: str) -> str:
        """Normalize backbone name to lowercase with underscores."""
        normalized = v.lower().replace("-", "_").replace(" ", "_")
        valid_backbones = {"resnet50", "efficientnet_b0", "vit_b16"}
        if normalized not in valid_backbones:
            raise ValueError(
                f"Unsupported backbone: '{v}'. "
                f"Supported: {sorted(valid_backbones)}"
            )
        return normalized


class TrainConfig(BaseModel):
    """Training hyperparameter configuration.

    Controls training mode, optimization, learning rate scheduling,
    loss functions, and metric learning parameters.
    """

    mode: Literal["pretrained", "finetune", "metric", "selfsupervised"] = "finetune"
    num_epochs: int = Field(default=50, ge=1, le=1000)
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1.0)
    weight_decay: float = Field(default=1e-5, ge=0.0, le=1.0)
    optimizer: Literal["adam", "sgd", "lars"] = "adam"
    loss: str = "crossentropy"
    margin: float = Field(default=0.2, ge=0.05, le=1.0)
    temperature: float = Field(default=0.07, ge=0.01, le=1.0)
    patience: int = Field(default=10, ge=1, le=100)
    gradient_clip_max_norm: float = Field(default=1.0, ge=0.1, le=10.0)
    warmup_epochs: int = Field(default=5, ge=0)
    layer_decay_factor: float = Field(default=0.9, ge=0.5, le=1.0)
    p_classes: int = Field(default=9, ge=2, le=9)
    k_samples: int = Field(default=4, ge=2, le=16)
    accumulation_steps: int = Field(default=1, ge=1, le=32)
    enable_class_weights: bool = True
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_every_n: int = Field(default=0, ge=0)
    resume_from: Optional[str] = None
    verbose: bool = False

    @field_validator("loss")
    @classmethod
    def normalize_loss_name(cls, v: str) -> str:
        """Normalize loss name to lowercase."""
        normalized = v.lower().replace("-", "_").replace(" ", "_")
        valid_losses = {"crossentropy", "triplet", "supcon", "ntxent"}
        if normalized not in valid_losses:
            raise ValueError(
                f"Unsupported loss: '{v}'. Supported: {sorted(valid_losses)}"
            )
        return normalized

    @model_validator(mode="after")
    def validate_mode_loss_compatibility(self) -> "TrainConfig":
        """Check that mode and loss are compatible."""
        if self.mode == "metric" and self.loss not in ("triplet", "supcon"):
            raise ValueError(
                f"Metric learning mode requires loss in ['triplet', 'supcon'], "
                f"got '{self.loss}'"
            )
        if self.mode == "selfsupervised" and self.loss != "ntxent":
            raise ValueError(
                f"Self-supervised mode requires loss='ntxent', got '{self.loss}'"
            )
        if self.mode == "finetune" and self.loss != "crossentropy":
            raise ValueError(
                f"Fine-tune mode requires loss='crossentropy', got '{self.loss}'"
            )
        if self.warmup_epochs >= self.num_epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be < "
                f"num_epochs ({self.num_epochs})"
            )
        return self


class EvalConfig(BaseModel):
    """Evaluation configuration.

    Controls metric computation, distance metrics, and bootstrap CI.
    """

    k_values: list[int] = Field(default=[1, 3, 5, 10])
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"
    subsample_size: int = Field(default=10000, ge=100, le=100000)
    bootstrap_iterations: int = Field(default=1000, ge=100, le=10000)
    output_dir: str = "outputs/metrics"

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, v: list[int]) -> list[int]:
        """Ensure k values are positive integers in ascending order."""
        if not v:
            raise ValueError("k_values must not be empty")
        for k in v:
            if k < 1:
                raise ValueError(f"k values must be positive, got {k}")
        return sorted(set(v))


class LoggingConfig(BaseModel):
    """Logging and observability configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    output: Literal["console", "file", "both"] = "both"
    log_dir: str = "logs/"
    json_format: bool = True
    max_file_size_mb: float = Field(default=10.0, ge=1.0)
    backup_count: int = Field(default=5, ge=1, le=20)


class WaferVisionConfig(BaseModel):
    """Root configuration schema combining all sections.

    Provides unified access to all configuration groups with cross-section
    validation for mutual consistency.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    wandb_enabled: bool = False
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dry_run: bool = False

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, v: int) -> int:
        """Ensure seed is valid for all random generators."""
        if not (0 <= v <= 2**32 - 1):
            raise ValueError(f"Seed must be in [0, 2^32-1], got {v}")
        return v


def _resolve_yaml_to_dict(config_dir: Path, overrides: Optional[list[str]] = None) -> dict[str, Any]:
    """Load Hydra/OmegaConf YAML config and apply overrides.

    Args:
        config_dir: Path to the configs/ directory.
        overrides: Hydra-style CLI overrides (e.g., ["training.lr=1e-3"]).

    Returns:
        Merged configuration dictionary.
    """
    from omegaconf import OmegaConf

    # Load base default config
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise ConfigValidationError(
            parameter="config_dir",
            value=str(config_dir),
            valid_range="directory containing default.yaml",
        )

    cfg = OmegaConf.load(default_path)

    # Remove Hydra-specific 'defaults' key since we handle composition manually
    if "defaults" in cfg:
        defaults_list = OmegaConf.to_container(cfg.defaults, resolve=True)
        del cfg["defaults"]

        # Apply composable configs from defaults list
        for item in defaults_list:
            if isinstance(item, dict):
                for group, name in item.items():
                    if group.startswith("_"):
                        continue
                    group_file = config_dir / group / f"{name}.yaml"
                    if group_file.exists():
                        group_cfg = OmegaConf.load(group_file)
                        cfg = OmegaConf.merge(cfg, group_cfg)
            elif isinstance(item, str) and item == "_self_":
                continue

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Resolve all interpolations and convert to plain dict
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return resolved  # type: ignore[return-value]


def _map_yaml_to_pydantic(raw: dict[str, Any]) -> dict[str, Any]:
    """Map raw YAML config keys to Pydantic model field names.

    Handles naming differences between YAML (user-friendly) and Pydantic models.

    Args:
        raw: Raw config dictionary from YAML.

    Returns:
        Dictionary suitable for WaferVisionConfig construction.
    """
    result: dict[str, Any] = {}

    # Data section
    if "data" in raw:
        data = dict(raw["data"])
        # Map 'raw_path' -> 'pkl_path' if present
        if "raw_path" in data:
            data["pkl_path"] = data.pop("raw_path")
        result["data"] = data

    # Model section
    if "model" in raw:
        model = dict(raw["model"])
        # Map 'l2_normalize' or 'normalize_embeddings'
        if "normalize_embeddings" in model:
            model["l2_normalize"] = model.pop("normalize_embeddings")
        result["model"] = model

    # Training section
    if "training" in raw:
        train = dict(raw["training"])
        # Map field names
        if "epochs" in train and "num_epochs" not in train:
            train["num_epochs"] = train.pop("epochs")
        if "early_stopping_patience" in train and "patience" not in train:
            train["patience"] = train.pop("early_stopping_patience")
        if "layerwise_lr_decay" in train and "layer_decay_factor" not in train:
            train["layer_decay_factor"] = train.pop("layerwise_lr_decay")
        if "gradient_accumulation_steps" in train and "accumulation_steps" not in train:
            train["accumulation_steps"] = train.pop("gradient_accumulation_steps")
        if "P_classes" in train and "p_classes" not in train:
            train["p_classes"] = train.pop("P_classes")
        if "K_samples" in train and "k_samples" not in train:
            train["k_samples"] = train.pop("K_samples")
        if "batch_size" in train:
            # batch_size is in DataConfig, but some training configs set it
            if "data" not in result:
                result["data"] = {}
            result["data"]["batch_size"] = train.pop("batch_size")
        # Remove keys not in TrainConfig
        if "simclr_temperature" in train:
            # For self-supervised, simclr_temperature maps to temperature
            if train.get("mode") == "selfsupervised":
                train["temperature"] = train.pop("simclr_temperature")
            else:
                train.pop("simclr_temperature")
        result["train"] = train

    # Evaluation section
    if "evaluation" in raw:
        eval_cfg = dict(raw["evaluation"])
        # Map k values
        if "knn_k_values" in eval_cfg:
            eval_cfg["k_values"] = eval_cfg.pop("knn_k_values")
        if "recall_k_values" in eval_cfg:
            # Merge into k_values
            if "k_values" not in eval_cfg:
                eval_cfg["k_values"] = eval_cfg.pop("recall_k_values")
            else:
                eval_cfg.pop("recall_k_values")
        result["eval"] = eval_cfg

    # Logging section
    if "logging" in raw:
        result["logging"] = dict(raw["logging"])

    # Top-level fields
    if "seed" in raw:
        result["seed"] = raw["seed"]
    if "dry_run" in raw:
        result["dry_run"] = raw["dry_run"]
    if "wandb_enabled" in raw:
        result["wandb_enabled"] = raw["wandb_enabled"]
    if "experiment_id" in raw:
        result["experiment_id"] = raw["experiment_id"]

    # Experiment section (may contain seed)
    if "experiment" in raw:
        exp = raw["experiment"]
        if isinstance(exp, dict):
            if "seed" in exp:
                result["seed"] = exp["seed"]

    return result


def load_config(
    overrides: Optional[list[str]] = None,
    config_dir: Optional[Path] = None,
) -> WaferVisionConfig:
    """Load and validate configuration from Hydra YAML + CLI overrides.

    Steps:
        1. Load base config (configs/default.yaml)
        2. Apply backbone-specific config (from defaults list)
        3. Apply training mode config (from defaults list)
        4. Apply CLI overrides
        5. Validate with Pydantic schema
        6. Return validated config

    Args:
        overrides: Hydra-style CLI overrides (e.g., ["training.learning_rate=1e-3"]).
        config_dir: Path to configs/ directory. Auto-detected if not provided.

    Returns:
        Validated WaferVisionConfig.

    Raises:
        ConfigValidationError: If any value out of valid range.
        ConfigConflictError: If incompatible options specified.
    """
    if config_dir is None:
        # Try common locations
        candidates = [
            Path("configs"),
            Path(__file__).parent.parent / "configs",
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "default.yaml").exists():
                config_dir = candidate
                break
        if config_dir is None:
            raise ConfigValidationError(
                parameter="config_dir",
                value="None",
                valid_range="directory containing configs/default.yaml",
            )

    try:
        raw = _resolve_yaml_to_dict(config_dir, overrides)
    except Exception as e:
        if isinstance(e, (ConfigValidationError, ConfigConflictError)):
            raise
        raise ConfigValidationError(
            parameter="config_file",
            value=str(config_dir),
            valid_range=f"valid YAML configuration: {e}",
        ) from e

    # Map YAML keys to Pydantic model fields
    mapped = _map_yaml_to_pydantic(raw)

    # Validate and create config
    try:
        config = WaferVisionConfig(**mapped)
    except Exception as e:
        # Convert Pydantic validation errors to our custom exceptions
        error_msg = str(e)
        if "mode" in error_msg and "loss" in error_msg:
            raise ConfigConflictError(
                param_a="train.mode",
                param_b="train.loss",
                explanation=error_msg,
            ) from e
        raise ConfigValidationError(
            parameter="config",
            value=str(mapped),
            valid_range=f"valid configuration: {error_msg}",
        ) from e

    return config


def preflight_check(config: WaferVisionConfig) -> dict[str, str]:
    """Run pre-flight checks before training starts.

    Validates runtime environment and resources against the configuration
    to catch issues before expensive computation begins.

    Checks:
        - Dataset path exists and is readable
        - GPU availability (device name, VRAM)
        - Checkpoint path (exists if resuming, directory writable)
        - Output directory (writable)
        - Disk space (>= 500 MB free)
        - Python/PyTorch versions

    Args:
        config: Validated WaferVisionConfig to check against.

    Returns:
        Dict mapping check_name -> "OK" or error description.
    """
    import shutil
    import sys

    results: dict[str, str] = {}

    # Check 1: Dataset path
    dataset_path = Path(config.data.pkl_path)
    if dataset_path.exists() and dataset_path.is_file():
        results["dataset_path"] = "OK"
    elif dataset_path.exists():
        results["dataset_path"] = f"Path exists but is not a file: {dataset_path}"
    else:
        results["dataset_path"] = (
            f"Dataset not found at: {dataset_path}. "
            f"Download it using 'make download'."
        )

    # Check 2: GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_mb = torch.cuda.get_device_properties(0).total_mem / (1024 * 1024)
            results["gpu"] = f"OK - {device_name} ({vram_mb:.0f} MB)"
        else:
            results["gpu"] = "No GPU available - will use CPU (slower training)"
    except ImportError:
        results["gpu"] = "PyTorch not installed"

    # Check 3: Checkpoint directory
    checkpoint_dir = Path(config.train.checkpoint_dir)
    if config.train.resume_from:
        resume_path = Path(config.train.resume_from)
        if resume_path.exists():
            results["checkpoint_resume"] = "OK"
        else:
            results["checkpoint_resume"] = (
                f"Resume checkpoint not found: {resume_path}"
            )
    else:
        results["checkpoint_resume"] = "OK - fresh training"

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results["checkpoint_dir"] = "OK"
    except OSError as e:
        results["checkpoint_dir"] = f"Cannot create checkpoint dir: {e}"

    # Check 4: Output directory
    output_dir = Path(config.eval.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        results["output_dir"] = "OK"
    except OSError as e:
        results["output_dir"] = f"Cannot create output dir: {e}"

    # Check 5: Disk space (>= 500 MB free)
    try:
        disk_usage = shutil.disk_usage(checkpoint_dir.resolve().anchor or ".")
        free_mb = disk_usage.free / (1024 * 1024)
        if free_mb >= 500:
            results["disk_space"] = f"OK - {free_mb:.0f} MB free"
        else:
            results["disk_space"] = (
                f"Low disk space: {free_mb:.0f} MB free (need >= 500 MB)"
            )
    except OSError:
        results["disk_space"] = "Unable to check disk space"

    # Check 6: Python and PyTorch versions
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results["python_version"] = f"OK - {python_version}"

    try:
        import torch

        results["pytorch_version"] = f"OK - {torch.__version__}"
        if torch.cuda.is_available():
            results["cuda_version"] = f"OK - {torch.version.cuda}"
        else:
            results["cuda_version"] = "N/A - CPU mode"
    except ImportError:
        results["pytorch_version"] = "PyTorch not installed"
        results["cuda_version"] = "N/A"

    # Check 7: Log directory
    log_dir = Path(config.logging.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        results["log_dir"] = "OK"
    except OSError as e:
        results["log_dir"] = f"Cannot create log dir: {e}"

    return results
