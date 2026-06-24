"""Factored per-event representation: clip ``Note`` lists <-> ordered event sequences.

The from-scratch model (M2+) consumes music as a sequence of *events*, one per note, each factored
into independent fields — pitch, onset delta, duration, velocity (and ``part`` for multi-voice
material) — rather than a flat token stream. This mirrors the per-field heads the recurrent model
will predict, and keeps the live ``Note`` tuple (:class:`osc_genai.generate.Note`) as the boundary
type so the OSC/MIDI I/O never has to change.

Time is quantised to a grid of ``steps_per_beat`` (default 4 = sixteenth notes). ``dt`` is the gap
in steps from the previous note's onset, so absolute timing is recovered by accumulation. Encoding
notes already on the grid and decoding them again is exact; off-grid input is snapped (lossy) to the
nearest step. Muted notes (``is_muted``) are dropped on encode — a muted note is "not played" — per
the project's representation decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from .generate import Note

DEFAULT_STEPS_PER_BEAT = 4


@dataclass(frozen=True)
class Event:
    """One note as factored fields. ``dt`` and ``dur`` are in grid steps, not beats."""

    pitch: int  # 0-127
    dt: int  # steps since the previous event's onset (>= 0)
    dur: int  # note length in steps (>= 1)
    velocity: int  # 0-127
    part: int = 0  # voice / instrument index (0 for single-part material)


def _quantize(beats: float, steps_per_beat: int) -> int:
    """Quantise a beat value to the nearest whole grid step."""
    return int(round(beats * steps_per_beat))


def notes_to_events(
    notes: list[Note], steps_per_beat: int = DEFAULT_STEPS_PER_BEAT
) -> list[Event]:
    """Convert clip notes to an onset-ordered event sequence.

    Notes sort by (start, pitch); muted notes are dropped. ``dt`` is the gap from the previous
    note's onset (the first event's ``dt`` is its absolute onset, so timing is fully recoverable).
    Durations quantise to at least one step so no note vanishes.
    """
    ordered = sorted((n for n in notes if not n.mute), key=lambda n: (n.start, n.pitch))
    events: list[Event] = []
    prev_onset = 0
    for note in ordered:
        onset = _quantize(note.start, steps_per_beat)
        events.append(
            Event(
                pitch=note.pitch,
                dt=max(0, onset - prev_onset),
                dur=max(1, _quantize(note.duration, steps_per_beat)),
                velocity=note.velocity,
                part=0,
            )
        )
        prev_onset = onset
    return events


def events_to_notes(
    events: list[Event], steps_per_beat: int = DEFAULT_STEPS_PER_BEAT
) -> list[Note]:
    """Inverse of :func:`notes_to_events`: recover onset times by accumulating ``dt``."""
    notes: list[Note] = []
    onset = 0
    for event in events:
        onset += event.dt
        notes.append(
            Note(
                pitch=event.pitch,
                start=onset / steps_per_beat,
                duration=event.dur / steps_per_beat,
                velocity=event.velocity,
            )
        )
    return notes
