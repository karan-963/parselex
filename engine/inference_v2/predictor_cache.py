"""Per-process predictor cache — load each ML stage once, reuse across resumes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .model_precision import get_precision

T = TypeVar("T")

_CACHE: dict[str, object] = {}


def get_predictor(key: str, factory: Callable[[], T]) -> T:
    """Return a cached predictor for *key*, constructing via *factory* on first use.

    The cache is namespaced by the active model precision (fp32/int8) so both
    variants can coexist across requests without clobbering each other.
    """
    cache_key = f"{key}:{get_precision()}"
    inst = _CACHE.get(cache_key)
    if inst is None:
        inst = factory()
        _CACHE[cache_key] = inst
    return inst  # type: ignore[return-value]
