from __future__ import annotations

from typing import Any

import torch
from transformers import BitsAndBytesConfig

from constants import normalize_quantization


CUDA_DEVICE_ALIASES = {"cuda", "gpu", "cuda0", "cuda:0"}
ACCELERATE_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0", "sequential"}


def preferred_cuda_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if prefer_bf16:
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def _canonical_device(device: Any) -> Any:
    if device is None:
        return "auto"
    if isinstance(device, str):
        return device.strip().lower()
    return device


def make_device_map(device: Any = "auto", *, quantization: str = "none") -> Any:
    quantization = normalize_quantization(quantization)
    device = _canonical_device(device)

    if quantization == "4bit" and not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit quantization uses bitsandbytes and requires a CUDA GPU. "
            "Run this on Boada/another GPU machine, or set model.quantization: none."
        )

    if isinstance(device, dict):
        return device

    if device in {"", "auto", None}:
        return "auto" if torch.cuda.is_available() else None

    if device == "cpu":
        if quantization == "4bit":
            raise RuntimeError("4-bit quantization cannot run on CPU.")
        return None

    if isinstance(device, str) and device in ACCELERATE_DEVICE_MAPS:
        return device

    if isinstance(device, str) and (device in CUDA_DEVICE_ALIASES or device.startswith("cuda:")):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {device!r}, but CUDA is not available.")
        cuda_device = "cuda:0" if device in CUDA_DEVICE_ALIASES else device
        return {"": cuda_device}

    return device


def build_model_load_kwargs(
    spec: Any,
    *,
    quantization: str = "none",
    device: Any = "auto",
    prefer_bf16: bool = True,
) -> dict[str, Any]:
    quantization = normalize_quantization(quantization)
    kwargs: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code}

    if quantization == "4bit":
        device_map = make_device_map(device, quantization=quantization)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=preferred_cuda_dtype(prefer_bf16),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = device_map
        return kwargs

    use_cuda = torch.cuda.is_available() and _canonical_device(device) != "cpu"
    kwargs["torch_dtype"] = preferred_cuda_dtype(prefer_bf16) if use_cuda else torch.float32
    device_map = make_device_map(device, quantization=quantization)
    if device_map is not None:
        kwargs["device_map"] = device_map
    return kwargs


def generation_device(model: Any) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_context_length(model: Any, tokenizer: Any) -> int | None:
    candidates = [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    for value in candidates:
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
    return None
