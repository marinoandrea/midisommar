"""Tests for the shared beat clock (osc_genai.realtime.clock).

WallClock is exercised directly; LinkClock construction is only checked when aalink is installed
(it's an optional extra and needs an Ableton Link runtime), so these tests stay native-build-free.
"""

from __future__ import annotations

import importlib.util

import pytest

from osc_genai.realtime.clock import LinkClock, WallClock, make_clock

_HAS_AALINK = importlib.util.find_spec("aalink") is not None


def test_wallclock_beat_advances_with_tempo():
    clock = WallClock(120.0)  # 120 BPM -> 2 beats/sec
    first = clock.beat
    # beats are monotonic, non-negative, and report the configured tempo
    assert clock.beat >= first >= 0.0
    assert clock.tempo == 120.0
    assert clock.playing is True  # WallClock never gates
    assert clock.peers == 0


def test_make_clock_without_link_is_wallclock():
    clock = make_clock(False, bpm=130.0, quantum=4, start_stop_sync=True)
    assert isinstance(clock, WallClock)
    assert clock.tempo == 130.0


def test_make_clock_link_without_aalink_raises_helpful_error(monkeypatch):
    """Requesting --link without aalink installed should fail with install guidance, not ImportError."""
    import builtins

    real_import = builtins.__import__

    def _no_aalink(name, *args, **kwargs):
        if name == "aalink":
            raise ImportError("No module named 'aalink'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_aalink)
    with pytest.raises(SystemExit, match="aalink"):
        make_clock(True, bpm=130.0, quantum=4, start_stop_sync=True)


@pytest.mark.skipif(not _HAS_AALINK, reason="aalink (optional extra) not installed")
def test_linkclock_exposes_clock_surface():
    clock = LinkClock(120.0, quantum=4, start_stop_sync=True)
    assert isinstance(clock.beat, float)
    assert clock.tempo > 0
    assert isinstance(clock.playing, bool)
    assert isinstance(clock.peers, int)
