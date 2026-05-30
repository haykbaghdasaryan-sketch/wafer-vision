"""Structured logging configuration with JSON output and run_id correlation.

Provides:
- setup_logging(): Configure Python logging with structured JSON format,
  file rotation, and run_id correlation.
- StructuredLogger: A structlog-based logger with context binding for
  module name and run_id, supporting resource usage logging.
"""

import logging
import logging.handlers
import uuid
from pathlib import Path
from typing import Any, Optional

import structlog


def setup_logging(
    level: str = "INFO",
    output: str = "both",
    log_dir: Optional[Path] = None,
    json_format: bool = True,
    run_id: Optional[str] = None,
    max_file_size_mb: float = 10.0,
    backup_count: int = 5,
) -> str:
    """Configure structured logging for the application.

    All log messages include: timestamp, level, module, run_id, message,
    and optional structured fields (epoch, batch, memory, duration).

    Args:
        level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        output: Output destination - "console", "file", or "both".
        log_dir: Directory for log files (created if needed). Defaults to ./logs.
        json_format: If True, emit JSON-formatted log lines.
        run_id: Correlation ID. Generated (UUID4) if not provided.
        max_file_size_mb: Max log file size before rotation.
        backup_count: Number of rotated log files to keep.

    Returns:
        The run_id (generated or provided) for correlation.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    if log_dir is None:
        log_dir = Path("logs")

    # Ensure log directory exists for file output
    if output in ("file", "both"):
        log_dir.mkdir(parents=True, exist_ok=True)

    # Get numeric log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on repeated calls
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Build the structlog ProcessorFormatter that renders the final output.
    # This replaces the stdlib formatter so structlog controls all formatting.
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    # Build handlers based on output mode
    if output in ("console", "both"):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if output in ("file", "both"):
        log_file = log_dir / "wafervision.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=int(max_file_size_mb * 1024 * 1024),
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure structlog for use by StructuredLogger.
    # structlog processes events then hands them to stdlib logging,
    # which uses the ProcessorFormatter above for final rendering.
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    return run_id


class StructuredLogger:
    """Logger wrapper that binds structured context to all messages.

    Uses structlog for structured JSON logging with context binding
    (run_id, module name). All log methods accept arbitrary keyword
    arguments that are included as structured fields in the log output.

    Usage:
        logger = StructuredLogger("training", run_id="abc-123")
        logger.info("Epoch complete", epoch=5, loss=0.234, duration_s=45.2)
    """

    def __init__(self, module_name: str, run_id: str) -> None:
        """Initialize a structured logger bound to a module and run.

        Args:
            module_name: Name of the module using this logger.
            run_id: Correlation ID for tracing across components.
        """
        self._module_name = module_name
        self._run_id = run_id
        self._logger = structlog.get_logger(module_name).bind(
            run_id=run_id,
            module=module_name,
        )

    @property
    def module_name(self) -> str:
        """Return the module name this logger is bound to."""
        return self._module_name

    @property
    def run_id(self) -> str:
        """Return the run_id this logger is bound to."""
        return self._run_id

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an informational message with optional structured fields.

        Args:
            message: Log message.
            **kwargs: Additional structured context fields.
        """
        self._logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message with optional structured fields.

        Args:
            message: Log message.
            **kwargs: Additional structured context fields.
        """
        self._logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message with optional structured fields.

        Args:
            message: Log message.
            **kwargs: Additional structured context fields.
        """
        self._logger.error(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message with optional structured fields.

        Args:
            message: Log message.
            **kwargs: Additional structured context fields.
        """
        self._logger.debug(message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log a critical message with optional structured fields.

        Args:
            message: Log message.
            **kwargs: Additional structured context fields.
        """
        self._logger.critical(message, **kwargs)

    def log_resource_usage(self) -> None:
        """Log current RAM and GPU memory usage at DEBUG level.

        Reports ram_used_mb, ram_available_mb, and optionally
        gpu_used_mb and gpu_total_mb if CUDA is available.
        Uses psutil if available, falls back to platform-specific methods.
        """
        import platform

        resource_info: dict[str, Any] = {}

        # Try psutil first (most reliable cross-platform)
        try:
            import psutil

            mem = psutil.virtual_memory()
            resource_info["ram_used_mb"] = round(mem.used / (1024 * 1024), 1)
            resource_info["ram_available_mb"] = round(
                mem.available / (1024 * 1024), 1
            )
            resource_info["ram_total_mb"] = round(mem.total / (1024 * 1024), 1)
        except ImportError:
            # Fallback: use os-level memory info where possible
            if platform.system() == "Windows":
                try:
                    import ctypes

                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(
                        ctypes.byref(stat)
                    )
                    resource_info["ram_total_mb"] = round(
                        stat.ullTotalPhys / (1024 * 1024), 1
                    )
                    resource_info["ram_available_mb"] = round(
                        stat.ullAvailPhys / (1024 * 1024), 1
                    )
                    resource_info["ram_used_mb"] = round(
                        (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 * 1024),
                        1,
                    )
                except Exception:
                    resource_info["ram_info"] = "unavailable"
            else:
                # Linux/macOS fallback via /proc/meminfo
                try:
                    with open("/proc/meminfo") as f:
                        meminfo: dict[str, int] = {}
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2:
                                meminfo[parts[0].rstrip(":")] = int(parts[1])
                    total_kb = meminfo.get("MemTotal", 0)
                    avail_kb = meminfo.get("MemAvailable", 0)
                    resource_info["ram_total_mb"] = round(total_kb / 1024, 1)
                    resource_info["ram_available_mb"] = round(avail_kb / 1024, 1)
                    resource_info["ram_used_mb"] = round(
                        (total_kb - avail_kb) / 1024, 1
                    )
                except Exception:
                    resource_info["ram_info"] = "unavailable"

        # GPU memory via PyTorch CUDA
        try:
            import torch

            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated()
                gpu_mem_total = torch.cuda.get_device_properties(0).total_mem
                resource_info["gpu_used_mb"] = round(
                    gpu_mem_allocated / (1024 * 1024), 1
                )
                resource_info["gpu_total_mb"] = round(
                    gpu_mem_total / (1024 * 1024), 1
                )
        except ImportError:
            pass

        self._logger.debug("Resource usage", **resource_info)
