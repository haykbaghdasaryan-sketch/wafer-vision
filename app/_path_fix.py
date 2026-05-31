"""Path fix for Streamlit multipage app.
Import this at the top of every page to ensure 'src' is importable
and 'app' module conflicts are resolved.
"""
import sys
import importlib.util
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
_app_dir = str(Path(__file__).resolve().parent)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Helper to import from app/ directory without relying on package resolution
def _import_app_module(name):
    """Import a module from the app/ directory by file path."""
    file_path = Path(_app_dir) / f"{name}.py"
    if not file_path.exists():
        # Try subdirectory (e.g., components/__init__.py)
        file_path = Path(_app_dir) / name / "__init__.py"
    spec = importlib.util.spec_from_file_location(f"app_{name}", str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Pre-load common app modules into a namespace pages can use
import types
app_ns = types.SimpleNamespace()
app_ns.error_handler = _import_app_module("error_handler")
app_ns.rate_limiter = _import_app_module("rate_limiter")
app_ns.components = _import_app_module("components")

# Make safe_execute and other common names available
safe_execute = app_ns.error_handler.safe_execute
SessionRateLimiter = app_ns.rate_limiter.SessionRateLimiter

# Components
loading_spinner = getattr(app_ns.components, 'loading_spinner', lambda *a, **kw: None)
wafer_card = getattr(app_ns.components, 'wafer_card', lambda *a, **kw: None)
wafer_gallery = getattr(app_ns.components, 'wafer_gallery', lambda *a, **kw: None)
metric_card = getattr(app_ns.components, 'metric_card', lambda *a, **kw: None)
metric_table = getattr(app_ns.components, 'metric_table', lambda *a, **kw: None)
radar_chart = getattr(app_ns.components, 'radar_chart', lambda *a, **kw: None)
