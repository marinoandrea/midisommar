"""Training the factored event model: batching, the teacher-forced loop, and checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .data import augment, load_midi_dir
from .model import FactoredEventModel, ModelConfig
from .repr import DEFAULT_STEPS_PER_BEAT, Event, notes_to_events
from .vocab import EventCodec, Fields, VocabConfig


def collate(sequences: list[list[Fields]], eos: Fields) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad EOS-terminated encoded sequences into a batch.

    Returns ``targets`` ``(B, L, 4)`` long and ``mask`` ``(B, L)`` bool (True for real positions,
    including each sequence's terminal EOS). Padding is the EOS tuple so embeddings stay valid; the
    mask keeps padding out of the loss.
    """
    max_len = max(len(s) for s in sequences)
    targets = torch.tensor(eos, dtype=torch.long).repeat(len(sequences), max_len, 1)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        targets[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return targets, mask


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    grad_clip: float = 1.0
    device: str = "cpu"


def train(
    model: FactoredEventModel,
    event_sequences: list[list[Event]],
    codec: EventCodec | None = None,
    config: TrainConfig | None = None,
    log_every: int = 50,
) -> list[float]:
    """Teacher-forced next-event training; returns the per-epoch mean loss history."""
    codec = codec or EventCodec(model.vocab)
    config = config or TrainConfig()
    device = torch.device(config.device)
    model.to(device)
    model.train()

    encoded = [codec.encode_sequence(seq, add_eos=True) for seq in event_sequences if seq]
    if not encoded:
        raise ValueError("no (non-empty) training sequences")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history: list[float] = []
    for epoch in range(config.epochs):
        order = torch.randperm(len(encoded)).tolist()
        epoch_loss, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = [encoded[i] for i in order[start : start + config.batch_size]]
            targets, mask = collate(batch, codec.eos)
            targets, mask = targets.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = model.loss(targets, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        history.append(epoch_loss / max(1, batches))
        if log_every and (epoch % log_every == 0 or epoch == config.epochs - 1):
            print(f"epoch {epoch:4d}  loss {history[-1]:.4f}")
    return history


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


def load_model(path: str | Path, map_location: str = "cpu") -> FactoredEventModel:
    """Inverse of :func:`save_model`."""
    checkpoint = torch.load(path, map_location=map_location)
    model = FactoredEventModel(
        VocabConfig(**checkpoint["vocab"]), ModelConfig(**checkpoint["config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the factored event model on a MIDI corpus.")
    parser.add_argument("--data-dir", required=True, help=".mid folder (searched recursively)")
    parser.add_argument("--out", default="model.pt", help="checkpoint output path")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--transpose", type=int, default=5, help="augment by +/- this many semitones (0 disables)"
    )
    parser.add_argument("--steps-per-beat", type=int, default=DEFAULT_STEPS_PER_BEAT)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sequences = [s for s in load_midi_dir(args.data_dir) if s]
    print(f"loaded {len(sequences)} non-empty sequence(s) from {args.data_dir}")
    if args.transpose:
        sequences = augment(sequences, semitones=range(-args.transpose, args.transpose + 1))
        print(f"after +/-{args.transpose} semitone transposition: {len(sequences)} sequence(s)")

    event_sequences = [notes_to_events(s, steps_per_beat=args.steps_per_beat) for s in sequences]
    vocab = VocabConfig()
    model = FactoredEventModel(vocab, ModelConfig(hidden_size=args.hidden, num_layers=args.layers))
    train(
        model,
        event_sequences,
        codec=EventCodec(vocab),
        config=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device
        ),
        log_every=1,
    )
    save_model(model, args.out)
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
