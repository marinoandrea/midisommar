"""Persist and restore a :class:`FactoredEventModel` — weights plus the configs to rebuild it.

Kept beside the model (not in :mod:`training`) so the realtime/inference commands can load a
checkpoint without importing the trainer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from midisommar.core.device import resolve_device
from midisommar.core.vocab import VocabConfig
from midisommar.model.factored import FactoredEventModel, ModelConfig


def save_model(model: FactoredEventModel, path: str | Path) -> None:
    """Persist weights + the vocab/model configs needed to rebuild the module."""
    torch.save(
        {
            "vocab": vars(model.vocab),
            "config": vars(model.config),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_model(path: str | Path, device: str = "auto") -> FactoredEventModel:
    """Inverse of :func:`save_model`.

    ``device`` selects where to run: ``"auto"`` (CUDA/MPS/CPU), or an explicit ``"cpu"``/``"cuda"``/
    ``"mps"``. Weights are read onto CPU first, then the rebuilt module is moved — loading straight
    onto MPS can be flaky, so load-then-move is the safe path.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model = FactoredEventModel(
        VocabConfig(**checkpoint["vocab"]), ModelConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolve_device(device))
    return model
