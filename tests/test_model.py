"""Tests for the factored per-event model: shapes, loss/backprop, and generation validity."""

from __future__ import annotations

import torch

from osc_genai.model import FactoredEventModel, ModelConfig
from osc_genai.repr import Event
from osc_genai.vocab import EventCodec, VocabConfig


def make_model() -> FactoredEventModel:
    return FactoredEventModel(
        VocabConfig(max_dt=16, max_dur=16, velocity_bins=8),
        ModelConfig(embed_dim=16, hidden_size=32, num_layers=1),
    )


def _random_targets(model: FactoredEventModel, batch: int, length: int) -> torch.Tensor:
    sizes = model.vocab.field_sizes
    return torch.stack([torch.randint(0, s, (batch, length)) for s in sizes], dim=-1)


def test_forward_shapes():
    torch.manual_seed(0)
    model = make_model()
    targets = _random_targets(model, batch=2, length=5)
    logits = model(targets)
    assert len(logits) == 4
    for field_logits, size in zip(logits, model.vocab.field_sizes):
        assert field_logits.shape == (2, 5, size)


def test_loss_is_scalar_and_backprops():
    torch.manual_seed(0)
    model = make_model()
    targets = _random_targets(model, batch=2, length=4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    loss = model.loss(targets, mask)
    assert loss.ndim == 0 and loss.item() > 0
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_generate_returns_valid_fields():
    torch.manual_seed(0)
    model = make_model()
    out = model.generate(max_events=10, temperature=1.0)
    assert isinstance(out, list)
    for fields in out:
        assert len(fields) == 4
        for index, size in zip(fields, model.vocab.field_sizes):
            assert 0 <= index < size
        assert fields[0] != model.vocab.eos_pitch  # EOS is the stop signal, never emitted


def test_generate_with_context_runs():
    torch.manual_seed(0)
    model = make_model()
    codec = EventCodec(model.vocab)
    context = codec.encode_sequence([Event(60, 0, 4, 100), Event(62, 4, 4, 100)], add_eos=False)
    out = model.generate(context=context, max_events=8)
    assert isinstance(out, list)
