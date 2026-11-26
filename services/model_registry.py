"""Lazy, fail-soft model registry.

Every ML model is wrapped in a :class:`LazyModel`: it is loaded on first use (thread
safe), and if loading fails (not downloaded, no internet, out of memory, package
missing) the error is logged once as a warning and the check is switched off while
the rest of the engine keeps working. :func:`model_status` feeds ``GET /health``.
"""

import logging
import threading
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from config.settings import CONFIG

logger = logging.getLogger(__name__)

T = TypeVar("T")

STATUS_DISABLED = "disabled"
STATUS_NOT_LOADED = "not_loaded"
STATUS_READY = "ready"
STATUS_UNAVAILABLE = "unavailable"


def get_device() -> str:
    """Return ``"cuda"`` when a GPU is usable and allowed by config, else ``"cpu"``.

    Returns:
        str: Torch device string.
    """
    wanted = str(CONFIG.get("device", "auto")).lower()
    if wanted == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if wanted == "cuda":
            logger.warning("config.json asks for device=cuda but no CUDA GPU/torch build is available; using CPU")
    except ImportError:
        pass
    return "cpu"


def pipeline_device() -> int:
    """Device index for Hugging Face ``pipeline`` (0 = first GPU, -1 = CPU)."""
    return 0 if get_device() == "cuda" else -1


class LazyModel(Generic[T]):
    """A model that loads on first use and fails softly."""

    def __init__(self, key: str, loader: Callable[[], T], description: str = "") -> None:
        """Create a lazy wrapper.

        Parameters:
            key (str): Key in ``config.json["models"]`` (controls enabled/name).
            loader (Callable[[], T]): Function that builds the model.
            description (str): Human description shown in /health.
        """
        self.key = key
        self.description = description or key
        self._loader = loader
        self._lock = threading.Lock()
        self._model: Optional[T] = None
        self._error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Whether config.json enables this model."""
        return bool(CONFIG["models"].get(self.key, {}).get("enabled", False))

    @property
    def status(self) -> str:
        """One of disabled / not_loaded / ready / unavailable."""
        if not self.enabled:
            return STATUS_DISABLED
        if self._model is not None:
            return STATUS_READY
        if self._error is not None:
            return STATUS_UNAVAILABLE
        return STATUS_NOT_LOADED

    def get(self) -> Optional[T]:
        """Return the loaded model, or None when disabled/unavailable.

        Returns:
            Optional[T]: Model instance or None.
        """
        if not self.enabled or self._error is not None:
            return None
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None and self._error is None:
                try:
                    logger.info("Loading model '%s' on %s ...", self.key, get_device())
                    self._model = self._loader()
                    logger.info("Model '%s' ready", self.key)
                except Exception as exc:  # fail soft: switch this check off
                    self._error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Model '%s' unavailable, check switched off: %s", self.key, self._error)
        return self._model

    def info(self) -> Dict[str, Any]:
        """Status dictionary for the health endpoint."""
        spec = CONFIG["models"].get(self.key, {})
        out: Dict[str, Any] = {"status": self.status, "model": spec.get("name", self.description)}
        if self._error:
            out["error"] = self._error[:300]
        return out


_REGISTRY: Dict[str, LazyModel[Any]] = {}


def register(key: str, loader: Callable[[], T], description: str = "") -> LazyModel[T]:
    """Register (or return the existing) lazy model for ``key``.

    Parameters:
        key (str): Config key.
        loader (Callable[[], T]): Builder function.
        description (str): Human description.

    Returns:
        LazyModel[T]: The registered wrapper.
    """
    if key not in _REGISTRY:
        _REGISTRY[key] = LazyModel(key, loader, description)
    return _REGISTRY[key]  # type: ignore[return-value]


def model_status() -> Dict[str, Dict[str, Any]]:
    """Status of every registered model (for ``GET /health``)."""
    return {key: model.info() for key, model in sorted(_REGISTRY.items())}


def warmup(keys: Optional[list] = None) -> None:
    """Load models now (e.g. at server start-up) instead of on the first request.

    Parameters:
        keys (Optional[list]): Subset of model keys; all registered models when None.
    """
    for key, model in _REGISTRY.items():
        if keys is None or key in keys:
            model.get()
