"""Pick the torch device to run on — the single source of truth for CPU/CUDA/MPS selection.

``"auto"`` prefers the fastest available backend (CUDA, then Apple-Silicon Metal/MPS, then CPU);
any explicit spec ("cpu", "cuda", "mps", "cuda:1", ...) is honoured as-is. Importing this module also
enables the MPS CPU fallback so a kernel MPS hasn't implemented degrades to CPU instead of crashing.
"""

from __future__ import annotations

import os

import torch

# Fall back to CPU for ops MPS lacks rather than raising; harmless when MPS is unused.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_device(spec: str = "auto") -> torch.device:
    """Map a device spec to a concrete :class:`torch.device`.

    ``"auto"`` picks the best available backend: CUDA, then Apple-Silicon MPS, then CPU. Any other
    value is passed straight through to :class:`torch.device`.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)
