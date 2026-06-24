"""Tests for the duet engine's pure parts (human-context capture + conditioning)."""

from __future__ import annotations

import mido

from osc_genai.duet import HumanContext, _context_events


def test_human_context_collects_recent_noteons():
    ctx = HumanContext(size=3)
    for pitch in (60, 62, 64, 65):
        ctx.on_message(mido.Message("note_on", note=pitch, velocity=100))
    assert ctx.snapshot() == [62, 64, 65]  # rolling window of size 3
    assert ctx.note_count == 4


def test_human_context_ignores_offs_and_zero_velocity():
    ctx = HumanContext()
    ctx.on_message(mido.Message("note_on", note=60, velocity=0))
    ctx.on_message(mido.Message("note_off", note=60, velocity=64))
    ctx.on_message(mido.Message("control_change", control=1, value=10))
    assert ctx.snapshot() == []
    assert ctx.note_count == 0


def test_context_events_from_pitches():
    events = _context_events([60, 64, 67])
    assert [e.pitch for e in events] == [60, 64, 67]
    assert all(e.dt == 1 and e.dur == 1 for e in events)
