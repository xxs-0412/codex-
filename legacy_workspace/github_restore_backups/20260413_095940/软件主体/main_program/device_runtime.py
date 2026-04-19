from __future__ import annotations

import os
from typing import Any

import torch


DEVICE_ENV_VAR = "BEARING_TORCH_DEVICE"
VALID_DEVICE_PREFERENCES = {"auto", "cpu", "cuda", "directml"}


def _normalize_preference(prefer: str | None) -> str:
    value = str(prefer or "auto").strip().lower()
    if value not in VALID_DEVICE_PREFERENCES:
        valid = ", ".join(sorted(VALID_DEVICE_PREFERENCES))
        raise ValueError(f"Unsupported device preference: {prefer!r}. Expected one of: {valid}.")
    return value


def _try_directml_device() -> tuple[Any, str] | None:
    try:
        import torch_directml  # type: ignore
    except Exception:
        return None
    try:
        return torch_directml.device(), "directml"
    except Exception:
        return None


def resolve_training_device(prefer: str = "auto") -> tuple[Any, str]:
    effective_prefer = _normalize_preference(os.getenv(DEVICE_ENV_VAR, prefer))
    if effective_prefer == "cpu":
        return torch.device("cpu"), "cpu"
    if effective_prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("BEARING_TORCH_DEVICE=cuda, but CUDA is unavailable.")
        return torch.device("cuda"), "cuda"
    if effective_prefer == "directml":
        directml_device = _try_directml_device()
        if directml_device is None:
            raise RuntimeError("BEARING_TORCH_DEVICE=directml, but torch-directml is unavailable.")
        return directml_device
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    directml_device = _try_directml_device()
    if directml_device is not None:
        return directml_device
    return torch.device("cpu"), "cpu"


def infer_model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def cpu_state_dict(model: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            state_dict[name] = value.detach().cpu()
        else:
            state_dict[name] = value
    return state_dict


def device_label_from_device(device: Any) -> str:
    if isinstance(device, torch.device):
        return str(device.type)
    label = str(device).strip().lower()
    if label.startswith("privateuseone"):
        return "directml"
    return label or "cpu"
