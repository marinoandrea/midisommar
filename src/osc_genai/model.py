"""Factored per-event recurrent model.

Each note is four independent categorical fields (pitch, dt, dur, velocity); the model embeds each,
concatenates them, runs a GRU over the sequence, and predicts the *next* note with one softmax head
per field. A learned ``start`` vector is the first input, so the model can generate from nothing;
generation stops when the pitch head emits EOS.

This recurrence is deliberately O(1)-per-event at inference (carry the hidden state, feed one note,
emit one note) — the property the live duet needs. The same module will later be conditioned on the
human's stream for anticipatory accompaniment (M3); for now :meth:`generate` supports unconditional
sampling and ``context``-primed continuation (the turn-taking responder).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .vocab import Fields, VocabConfig


@dataclass
class ModelConfig:
    embed_dim: int = 32  # per-field embedding size
    hidden_size: int = 256
    num_layers: int = 1
    dropout: float = 0.0


class FactoredEventModel(nn.Module):
    def __init__(
        self, vocab: VocabConfig | None = None, config: ModelConfig | None = None
    ) -> None:
        super().__init__()
        self.vocab = vocab or VocabConfig()
        self.config = config or ModelConfig()
        sizes = self.vocab.field_sizes  # (pitch, dt, dur, velocity)
        embed = self.config.embed_dim

        self.embeddings = nn.ModuleList([nn.Embedding(size, embed) for size in sizes])
        input_dim = embed * len(sizes)
        self.start = nn.Parameter(torch.zeros(input_dim))
        self.rnn = nn.GRU(
            input_dim,
            self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )
        self.heads = nn.ModuleList([nn.Linear(self.config.hidden_size, size) for size in sizes])

    # -- core -----------------------------------------------------------------------------------
    def embed(self, fields: torch.Tensor) -> torch.Tensor:
        """``(..., 4)`` long field indices -> ``(..., input_dim)`` embedding."""
        parts = [emb(fields[..., i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=-1)

    def _logits(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return [head(hidden) for head in self.heads]

    def forward(self, targets: torch.Tensor) -> list[torch.Tensor]:
        """``targets`` ``(B, L, 4)`` long -> per-field logits, each ``(B, L, vocab_i)``.

        Input at step *t* is the embedding of target *t-1* (a learned ``start`` at *t=0*), so logits
        at *t* predict target *t* — standard teacher forcing.
        """
        batch = targets.shape[0]
        emb = self.embed(targets)  # (B, L, input_dim)
        start = self.start.view(1, 1, -1).expand(batch, 1, -1)
        inp = torch.cat([start, emb[:, :-1, :]], dim=1)  # (B, L, input_dim)
        out, _ = self.rnn(inp)  # (B, L, hidden)
        return self._logits(out)

    def loss(self, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sum of per-field cross-entropies, averaged over masked positions.

        ``mask`` ``(B, L)`` marks valid positions (real events + the terminal EOS). The dt/dur/
        velocity heads are not trained on EOS positions (their values are meaningless there).
        """
        logits = self.forward(targets)
        eos = targets[..., 0] == self.vocab.eos_pitch
        masks = [mask, mask & ~eos, mask & ~eos, mask & ~eos]
        total = torch.zeros((), device=targets.device)
        for i, (field_logits, field_mask) in enumerate(zip(logits, masks)):
            vocab = field_logits.shape[-1]
            ce = F.cross_entropy(
                field_logits.reshape(-1, vocab),
                targets[..., i].reshape(-1),
                reduction="none",
            ).reshape(targets.shape[:2])
            total = total + (ce * field_mask).sum() / field_mask.sum().clamp(min=1)
        return total

    # -- generation -----------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        context: list[Fields] | None = None,
        max_events: int = 64,
        temperature: float = 1.0,
    ) -> list[Fields]:
        """Autoregressively sample events. With ``context`` the model is primed on it first.

        ``temperature <= 0`` is greedy (argmax). Sampling halts at EOS or ``max_events``.
        """
        self.eval()
        device = self.start.device
        context = list(context or [])
        start = self.start.view(1, 1, -1)
        if context:
            ctx = torch.tensor([context], dtype=torch.long, device=device)  # (1, C, 4)
            prime = torch.cat([start, self.embed(ctx)], dim=1)
        else:
            prime = start
        out, hidden = self.rnn(prime)
        last = out[:, -1, :]  # (1, hidden) — predicts the next event

        result: list[Fields] = []
        for _ in range(max_events):
            fields = self._sample(last, temperature)
            if fields[0] == self.vocab.eos_pitch:
                break
            result.append(fields)
            step = self.embed(torch.tensor([[fields]], dtype=torch.long, device=device))
            out, hidden = self.rnn(step, hidden)
            last = out[:, -1, :]
        return result

    def _sample(self, hidden: torch.Tensor, temperature: float) -> Fields:
        indices: list[int] = []
        for field_logits in self._logits(hidden):  # each (1, vocab)
            logits = field_logits.squeeze(0)
            if temperature <= 0:
                indices.append(int(torch.argmax(logits)))
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                indices.append(int(torch.multinomial(probs, 1)))
        return tuple(indices)  # type: ignore[return-value]
