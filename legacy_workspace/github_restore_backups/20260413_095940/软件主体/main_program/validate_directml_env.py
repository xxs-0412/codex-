from __future__ import annotations

import importlib
import sys

import torch

from device_runtime import resolve_training_device


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "openpyxl",
    "torch",
    "torch_directml",
]
EXPECTED_TORCH_PREFIX = "2.3.1"


def main() -> None:
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")

    for module_name in REQUIRED_MODULES:
        importlib.import_module(module_name)
        print(f"import_ok: {module_name}")

    print(f"torch_version: {torch.__version__}")
    if not str(torch.__version__).startswith(EXPECTED_TORCH_PREFIX):
        raise RuntimeError(f"Expected torch {EXPECTED_TORCH_PREFIX}.*, but found {torch.__version__}.")

    device, device_label = resolve_training_device("directml")
    a = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    b = torch.tensor([3.0, 4.0], dtype=torch.float32, device=device)
    c = (a + b).detach().cpu().tolist()

    print(f"device: {device_label}")
    print(f"directml_tensor_add: {c}")
    print("validate_directml_env: ok")


if __name__ == "__main__":
    main()
