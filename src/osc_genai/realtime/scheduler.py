"""Anticipatory real-time scheduling: play machine notes ahead, revise them when the human moves.

The duet generates machine notes into the *future* (a lookahead window) and plays from this buffer.
Notes whose onset falls within a short **commit horizon** ahead of the playhead are locked — too
imminent to un-send — while everything beyond it is *revisable*. When the human's context changes,
the engine drops and regenerates the revisable tail (**reconciliation**), so the machine commits to
the near future yet still adapts to where the human is going.

This module is pure timing/bookkeeping (no model, no MIDI), so it is fully unit-testable with an
explicit playhead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scheduled:
    """A machine note placed on the timeline. ``onset``/``dur`` are absolute grid steps."""

    onset: float
    pitch: int
    velocity: int
    dur: float
    channel: int = 0


class AnticipatoryBuffer:
    """Time-ordered buffer of upcoming machine notes with a protected commit horizon."""

    def __init__(self, commit_horizon: float = 2.0) -> None:
        self.commit_horizon = commit_horizon
        self._notes: list[Scheduled] = []  # kept sorted by onset

    def __len__(self) -> int:
        return len(self._notes)

    def horizon_end(self, playhead: float) -> float:
        """Onsets at or before this step are committed (locked); beyond it they are revisable."""
        return playhead + self.commit_horizon

    def last_onset(self, default: float = 0.0) -> float:
        """Onset of the furthest-future note (how far ahead we've generated)."""
        return self._notes[-1].onset if self._notes else default

    def add(self, notes: list[Scheduled]) -> None:
        self._notes.extend(notes)
        self._notes.sort(key=lambda s: s.onset)

    def upcoming(self) -> list[Scheduled]:
        """The currently-scheduled (not-yet-fired) notes — re-fed as history so generation continues
        from the buffer frontier rather than regenerating the same window."""
        return list(self._notes)

    def clear(self) -> None:
        """Drop every buffered note (e.g. when the transport stops and the plan is abandoned)."""
        self._notes.clear()

    def pop_due(self, now: float) -> list[Scheduled]:
        """Remove and return notes whose onset has arrived (``onset <= now``)."""
        due = [s for s in self._notes if s.onset <= now]
        if due:
            self._notes = [s for s in self._notes if s.onset > now]
        return due

    def reconcile(self, playhead: float) -> tuple[float, int]:
        """Drop the revisable tail (onsets beyond the commit horizon).

        Returns ``(resume_step, dropped)`` — the step at which regenerated notes should resume (just
        after the last committed note ends, or the playhead if nothing is committed) and how many
        notes were dropped.
        """
        horizon = self.horizon_end(playhead)
        kept = [s for s in self._notes if s.onset <= horizon]
        dropped = len(self._notes) - len(kept)
        self._notes = kept
        resume = max((s.onset + s.dur for s in kept), default=playhead)
        return resume, dropped
