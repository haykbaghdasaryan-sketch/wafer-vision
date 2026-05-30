"""Generic registry pattern for extensible component lookup.

Provides a thread-safe, decorator-based registry that maps string names
to component classes. Supports case-insensitive name normalization and
config-driven instantiation.
"""

import re
import threading
from typing import Any, Callable, Dict, Generic, List, Type, TypeVar

T = TypeVar("T")


def _normalize_name(name: str) -> str:
    """Normalize a component name to lowercase with underscores.

    Replaces hyphens and spaces with underscores, collapses multiple
    underscores, and lowercases. Does NOT split CamelCase since backbone
    names like "ResNet50" should map to "resnet50" (not "res_net50").

    Args:
        name: Raw component name (e.g., "ResNet50", "EfficientNet-B0").

    Returns:
        Normalized name (e.g., "resnet50", "efficientnet_b0").
    """
    # Replace hyphens and spaces with underscores
    name = name.replace("-", "_").replace(" ", "_")
    # Lowercase the whole thing
    name = name.lower()
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Strip leading/trailing underscores
    name = name.strip("_")
    return name


class Registry(Generic[T]):
    """Thread-safe registry mapping string names to component classes.

    Supports decorator-based registration and config-driven instantiation.

    Example:
        backbone_registry = Registry[BaseBackbone]("backbone")

        @backbone_registry.register("resnet50")
        class ResNet50Backbone(BaseBackbone): ...

        model = backbone_registry.create("resnet50", pretrained=True)

    Raises:
        KeyError: If name not registered. Message includes all registered names.
    """

    def __init__(self, name: str) -> None:
        """Initialize registry.

        Args:
            name: Human-readable registry name for error messages.
        """
        self._name = name
        self._registry: Dict[str, Type[T]] = {}
        self._lock = threading.Lock()

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class under a given name.

        Args:
            name: Lookup key (will be normalized to lowercase + underscore).

        Returns:
            Decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If name is already registered.
        """
        normalized = _normalize_name(name)

        def decorator(cls: Type[T]) -> Type[T]:
            with self._lock:
                if normalized in self._registry:
                    raise ValueError(
                        f"'{normalized}' is already registered in "
                        f"'{self._name}' registry."
                    )
                self._registry[normalized] = cls
            return cls

        return decorator

    def create(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered component by name.

        Args:
            name: Registered component name (case-insensitive, normalized).
            **kwargs: Constructor arguments passed to the component.

        Returns:
            Instantiated component.

        Raises:
            KeyError: If name not found. Lists all registered names.
        """
        normalized = _normalize_name(name)
        with self._lock:
            if normalized not in self._registry:
                registered = sorted(self._registry.keys())
                raise KeyError(
                    f"'{normalized}' is not registered in '{self._name}' "
                    f"registry. Available: {registered}"
                )
            cls = self._registry[normalized]
        return cls(**kwargs)

    def list_registered(self) -> List[str]:
        """Return sorted list of all registered component names."""
        with self._lock:
            return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered (case-insensitive)."""
        normalized = _normalize_name(name)
        with self._lock:
            return normalized in self._registry

    def __repr__(self) -> str:
        """Return string representation of the registry."""
        with self._lock:
            names = sorted(self._registry.keys())
        return f"Registry(name='{self._name}', registered={names})"


# Global registries
backbone_registry: Registry = Registry("backbone")
loss_registry: Registry = Registry("loss")
metric_registry: Registry = Registry("metric")
dataset_registry: Registry = Registry("dataset")
