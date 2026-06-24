"""Stream a trained model's output as a real-time MIDI note stream — *live* generation.

Unlike the clip path (write a whole clip, then play it), this emits notes one at a time over a
virtual MIDI port at a chosen tempo: the model keeps generating phrases and they play continuously,
seamlessly joined. Route a Live MIDI track's input at the port (``osc-genai out``) with an
instrument to hear it. This is the model-driven *output* half of the live duet; M3 adds the
human-conditioned *input* half (anticipation).

Timing uses a small real-time scheduler: a heap of pending note-offs is flushed as their moment
arrives while we wait for each next onset.
"""

from __future__ import annotations

import argparse
import heapq
import os
import time

# python-rtmidi is mido's real-time backend; select it before mido resolves a default.
os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402

from .repr import DEFAULT_STEPS_PER_BEAT  # noqa: E402
from .train import load_model  # noqa: E402
from .vocab import EventCodec  # noqa: E402

DEFAULT_OUT_PORT = "osc-genai out"


def stream(
    model,
    out: "mido.ports.BaseOutput",
    *,
    bpm: float = 130.0,
    steps_per_beat: int = DEFAULT_STEPS_PER_BEAT,
    temperature: float = 0.95,
    channel: int = 0,
    seconds: float | None = None,
) -> None:
    """Generate forever (or for ``seconds``) and play each note at its real-time moment."""
    codec = EventCodec(model.vocab)
    sec_per_step = (60.0 / bpm) / steps_per_beat
    start = time.perf_counter()
    onset_step = 0.0  # absolute step position of the next onset, relative to start
    pending_off: list[tuple[float, int]] = []  # min-heap of (off_time, pitch)

    def flush_due(now: float) -> None:
        while pending_off and pending_off[0][0] <= now:
            _, pitch = heapq.heappop(pending_off)
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))

    def expired() -> bool:
        return seconds is not None and (time.perf_counter() - start) >= seconds

    try:
        while not expired():
            events = model.generate(temperature=temperature, max_events=64)
            if not events:
                time.sleep(0.05)
                continue
            for fields in events:
                event = codec.decode(fields)
                onset_step += event.dt
                on_at = start + onset_step * sec_per_step
                while True:  # wait for the onset, releasing finished notes meanwhile
                    now = time.perf_counter()
                    flush_due(now)
                    if now >= on_at:
                        break
                    wake = on_at if not pending_off else min(on_at, pending_off[0][0])
                    time.sleep(max(0.0, wake - now))
                out.send(
                    mido.Message("note_on", note=event.pitch, velocity=event.velocity, channel=channel)
                )
                heapq.heappush(pending_off, (start + (onset_step + event.dur) * sec_per_step, event.pitch))
                if expired():
                    break
    finally:
        for _, pitch in pending_off:
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))
        for pitch in range(128):  # belt-and-braces panic
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream live model generation to a MIDI port.")
    parser.add_argument("--checkpoint", required=True, help="trained model (.pt)")
    parser.add_argument("--out-port", default=DEFAULT_OUT_PORT)
    parser.add_argument("--bpm", type=float, default=130.0)
    parser.add_argument("--steps-per-beat", type=int, default=DEFAULT_STEPS_PER_BEAT)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    parser.add_argument(
        "--no-virtual", action="store_true", help="connect to an existing port by name"
    )
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    with mido.open_output(args.out_port, virtual=not args.no_virtual) as out:
        print(
            f"streaming live generation to MIDI port {args.out_port!r} at {args.bpm} BPM "
            f"(temp {args.temperature}). Ctrl-C to stop."
        )
        try:
            stream(
                model,
                out,
                bpm=args.bpm,
                steps_per_beat=args.steps_per_beat,
                temperature=args.temperature,
                seconds=args.seconds,
            )
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
