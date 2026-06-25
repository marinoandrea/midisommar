"""The factored per-event recurrent model and its checkpoint IO.

``from midisommar.model import FactoredEventModel, load_model``.
"""

from midisommar.model.checkpoint import load_model, save_model
from midisommar.model.factored import FactoredEventModel, ModelConfig

__all__ = ["FactoredEventModel", "ModelConfig", "load_model", "save_model"]
