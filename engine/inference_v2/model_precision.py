"""Runtime model-precision selection (fp32 vs int8) for inference_v2 stages.

This is an abstract, section-agnostic helper: every stage predictor loads its
own fp32 checkpoint as usual, then calls :func:`apply_precision` to optionally
swap in an INT8 (dynamically quantized) variant based on the current request's
precision. The precision is stored in a :class:`contextvars.ContextVar` so it
propagates through a single synchronous pipeline/rerun call without threading a
parameter through every stage function.
"""

from __future__ import annotations

import contextvars
import os
from typing import Literal

import torch
import torch.nn as nn

# Select a CPU quantized backend so dynamic-quantized ops (prepack/linear) work.
# macOS/ARM ships qnnpack; x86 typically has fbgemm.
_SUPPORTED_QENGINES = tuple(getattr(torch.backends.quantized, "supported_engines", ()))
for _engine in ("qnnpack", "fbgemm"):
    if _engine in _SUPPORTED_QENGINES:
        torch.backends.quantized.engine = _engine
        break

Precision = Literal["fp32", "int8"]

VALID_PRECISIONS: tuple[str, ...] = ("fp32", "int8")
DEFAULT_PRECISION: Precision = "fp32"

_PRECISION: contextvars.ContextVar[str] = contextvars.ContextVar(
    "inference_v2_precision", default=DEFAULT_PRECISION
)


def normalize_precision(raw: str | None) -> Precision:
    """Coerce arbitrary input to a supported precision, defaulting to fp32."""
    value = (raw or "").strip().lower()
    if value in ("int8", "int", "quantized", "quant"):
        return "int8"
    return "fp32"


def set_precision(raw: str | None) -> Precision:
    """Set the active precision for the current context; returns the normalized value."""
    precision = normalize_precision(raw)
    _PRECISION.set(precision)
    return precision


def get_precision() -> Precision:
    """Return the active precision for the current context."""
    return normalize_precision(_PRECISION.get())


def int8_path_for(fp32_path: str) -> str:
    """Derive the INT8 checkpoint path from an fp32 checkpoint path.

    ``.../best_model.pt`` -> ``.../best_model_int8.pt``
    """
    root, ext = os.path.splitext(fp32_path)
    return f"{root}_int8{ext or '.pt'}"


def quantize_dynamic_int8(model: nn.Module, inplace: bool = False) -> nn.Module:
    """Apply dynamic INT8 quantization to all Linear layers (CPU inference).

    ``inplace=True`` swaps only the Linear submodules in place, preserving any
    runtime monkey-patches on other modules (e.g. custom attention ``forward``
    bindings) that whole-model copies/pickles would otherwise break.
    """
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8, inplace=inplace
    )


def apply_precision(
    model: nn.Module,
    fp32_path: str,
    device: torch.device,
) -> tuple[nn.Module, torch.device]:
    """Optionally swap *model* for its INT8 variant based on the active precision.

    For fp32 (default) the model and device are returned unchanged. For int8 the
    already-built fp32 *model* is moved to CPU (quantized ops are CPU-only) and
    quantized in place; a prebuilt ``*_int8.pt`` state_dict is loaded on top when
    available. Returns the model and the device its inputs should live on.
    """
    if get_precision() != "int8":
        return model, device

    cpu = torch.device("cpu")
    int8_path = int8_path_for(fp32_path)

    # In-place quantization preserves monkey-patched submodules (custom attention).
    quantized = quantize_dynamic_int8(model.to(cpu), inplace=True)

    if os.path.isfile(int8_path):
        # Trusted local artifact; packed quantized params require weights_only=False.
        state = torch.load(int8_path, map_location=cpu, weights_only=False)
        quantized.load_state_dict(state)

    quantized.eval()
    return quantized, cpu
