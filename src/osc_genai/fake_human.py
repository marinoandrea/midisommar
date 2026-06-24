"""A fake human: loop a MIDI line into the duet's input port, for a mock local setup.

No controller or Ableton input-routing needed — this plays a looping sequence (one of your own
clips, a ``.mid`` file, or a built-in acid pattern) out to the port the duet listens on
(``osc-genai in``), so you can hear/verify the duet respond. Two terminals::

    uv run osc-genai-duet --checkpoint models/acid_v1.pt        # terminal 1 (creates the ports)
    uv run osc-genai-fake-human --from-data data/MIDI           # terminal 2 (drives the input)

Then route ``osc-genai out`` to a synth in Ableton to hear the model's response.
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import random
import time

# python-rtmidi is mido's real-time backend; select it before mido resolves a default.
os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402

from .data import load_midi_dir, load_midi_file  # noqa: E402
from .generate import Note  # noqa: E402

DEFAULT_TARGET = "osc-genai in"

# A built-in one-bar acid loop (16th notes; root E2 with octave/interval jumps) used when no
# data folder or file is given.
_ACID = [
    Note(40, 0.00, 0.25, 110), Note(52, 0.25, 0.25, 90), Note(40, 0.50, 0.25, 110), Note(43, 0.75, 0.25, 90),
    Note(40, 1.00, 0.25, 110), Note(52, 1.25, 0.25, 90), Note(45, 1.50, 0.25, 100), Note(40, 1.75, 0.25, 110),
    Note(40, 2.00, 0.25, 110), Note(55, 2.25, 0.25, 90), Note(40, 2.50, 0.25, 110), Note(43, 2.75, 0.25, 90),
    Note(40, 3.00, 0.25, 110), Note(52, 3.25, 0.25, 90), Note(48, 3.50, 0.25, 100), Note(40, 3.75, 0.25, 110),
]


def loop_length_beats(notes: list[Note]) -> float:
    """Loop length: the last note's end, rounded up to a whole beat (min 1 beat)."""
    end = max((n.start + n.duration for n in notes), default=0.0)
    return max(1.0, float(math.ceil(end)))


def play_loop(out, notes: list[Note], *, bpm: float = 130.0, channel: int = 0,
              seconds: float | None = None) -> None:
    """Loop ``notes`` out ``out`` in real time at ``bpm`` until ``seconds`` elapses (or forever)."""
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    if not notes:
        return
    sec_per_beat = 60.0 / bpm
    length = loop_length_beats(notes)
    start = time.perf_counter()
    pending_off: list[tuple[float, int]] = []

    def flush(now: float) -> None:
        while pending_off and pending_off[0][0] <= now:
            _, pitch = heapq.heappop(pending_off)
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))

    base, stop = 0.0, False
    try:
        while not stop:
            for note in notes:
                on_at = start + (base + note.start) * sec_per_beat
                while True:
                    now = time.perf_counter()
                    flush(now)
                    if seconds is not None and now - start >= seconds:
                        stop = True
                        break
                    if now >= on_at:
                        break
                    wake = on_at if not pending_off else min(on_at, pending_off[0][0])
                    time.sleep(max(0.0, wake - now))
                if stop:
                    break
                out.send(mido.Message("note_on", note=note.pitch, velocity=note.velocity, channel=channel))
                heapq.heappush(pending_off, (start + (base + note.start + note.duration) * sec_per_beat, note.pitch))
            base += length
    finally:
        for _, pitch in pending_off:
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop a MIDI line into the duet's input port.")
    parser.add_argument("--to-port", default=DEFAULT_TARGET, help="port to send into (the duet's input)")
    parser.add_argument("--from-data", default=None, help="folder of .mid files; loops one clip")
    parser.add_argument("--midi", default=None, help="a single .mid file to loop")
    parser.add_argument("--bpm", type=float, default=130.0)
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    parser.add_argument("--seed", type=int, default=0, help="which clip to pick from --from-data")
    parser.add_argument("--virtual", action="store_true", help="create the port instead of connecting")
    args = parser.parse_args()

    if args.midi:
        notes = load_midi_file(args.midi)
    elif args.from_data:
        sequences = [s for s in load_midi_dir(args.from_data) if s]
        random.seed(args.seed)
        notes = random.choice(sequences) if sequences else _ACID
    else:
        notes = _ACID

    if not args.virtual and args.to_port not in mido.get_output_names():
        raise SystemExit(
            f"Port {args.to_port!r} not found. Start the duet first "
            "(uv run osc-genai-duet ...), or pass --virtual to create the port."
        )

    with mido.open_output(args.to_port, virtual=args.virtual) as out:
        print(f"fake human: looping {len(notes)} notes into {args.to_port!r} at {args.bpm} BPM. Ctrl-C to stop.")
        try:
            play_loop(out, notes, bpm=args.bpm, seconds=args.seconds)
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
