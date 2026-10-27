"""Safe checkpoint helpers. Never deserialize untrusted pickle-based model files."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file


def save_safetensors(state_dict: Mapping[str, torch.Tensor], path: str | Path) -> None:
    """Save a state dict as non-pickle Safetensors; tensors are detached and moved to CPU."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_state = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }
    if not safe_state or len(safe_state) != len(state_dict):
        raise ValueError("Expected a non-empty state_dict containing only string keys and tensors.")
    save_file(safe_state, str(target))


def load_safetensors(path: str | Path, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a Safetensors file. Reject .pt/.pth/.ckpt files rather than unpickling them."""
    source = Path(path)
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Unsafe checkpoint format; only .safetensors files are accepted.")
    if not source.is_file():
        raise FileNotFoundError(source)
    return load_file(str(source), device=device)
