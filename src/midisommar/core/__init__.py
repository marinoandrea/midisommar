"""Core music representation: the ``Note``/``Event`` types and the model-field codec.

Pure data — no torch, no IO. ``from midisommar.core import Note, Event, EventCodec``.
"""

from midisommar.core.event import (
    DEFAULT_STEPS_PER_BEAT,
    PARTNER,
    SELF,
    Event,
    events_to_notes,
    notes_to_events,
)
from midisommar.core.note import Note
from midisommar.core.vocab import MIDI_RANGE, EventCodec, Fields, VocabConfig

__all__ = [
    "Note",
    "Event",
    "notes_to_events",
    "events_to_notes",
    "DEFAULT_STEPS_PER_BEAT",
    "PARTNER",
    "SELF",
    "EventCodec",
    "VocabConfig",
    "Fields",
    "MIDI_RANGE",
]
