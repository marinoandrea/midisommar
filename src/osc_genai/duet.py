"""Real-time responsive duet: the model plays a complementary line *with* the human, live.

The human plays into a virtual MIDI input; the model continuously generates short chunks
**conditioned on the human's most recent notes** and plays them out a virtual MIDI output in real
time, re-conditioning every chunk so it follows along. Both parts sound simultaneously.

Scope honesty: the model is trained on *solo* material, so this is *responsive* (it follows the
human's pitch material/register), not yet truly *anticipatory* (jointly predicting the human's
future) — that needs a multi-part / duet-trained model (M4/M5). The real-time plumbing here is the
foundation for it. Reconciliation of already-scheduled notes is approximated by short chunks +
frequent re-conditioning rather than explicit revision.
"""

from __future__ import annotations

import argparse
import heapq
import os
import threading
import time
from collections import deque

# python-rtmidi is mido's real-time backend; select it before mido resolves a default.
os.environ.setdefault("MIDO_BACKEND", "mido.backends.rtmidi")
import mido  # noqa: E402

from .repr import DEFAULT_STEPS_PER_BEAT, Event  # noqa: E402
from .scheduler import AnticipatoryBuffer, Scheduled  # noqa: E402
from .train import load_model  # noqa: E402
from .vocab import EventCodec  # noqa: E402

DEFAULT_IN_PORT = "osc-genai in"
DEFAULT_OUT_PORT = "osc-genai out"


class HumanContext:
    """Thread-safe rolling buffer of the human's recent pitches, fed from a MIDI input port."""

    def __init__(self, size: int = 12) -> None:
        self._pitches: deque[int] = deque(maxlen=size)
        self._lock = threading.Lock()
        self.note_count = 0

    def on_message(self, msg: "mido.Message") -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            with self._lock:
                self._pitches.append(msg.note)
                self.note_count += 1

    def snapshot(self) -> list[int]:
        with self._lock:
            return list(self._pitches)


def _context_events(pitches: list[int]) -> list[Event]:
    """Minimal Events from the human's recent pitches (pitch context; uniform 16th rhythm)."""
    return [Event(pitch=p, dt=1, dur=1, velocity=100) for p in pitches]


def duet(
    model,
    inp: "mido.ports.BaseInput",
    out: "mido.ports.BaseOutput",
    *,
    bpm: float = 130.0,
    steps_per_beat: int = DEFAULT_STEPS_PER_BEAT,
    temperature: float = 0.95,
    chunk_events: int = 8,
    lookahead_steps: float = 8.0,
    commit_horizon: float = 2.0,
    channel: int = 0,
    seconds: float | None = None,
) -> None:
    """Anticipatory duet: generate machine notes *ahead* into a buffer and play from it; when the
    human's recent context changes, reconcile (drop + regenerate the revisable tail) so the machine
    commits to the near future yet adapts to where the human is going.
    """
    codec = EventCodec(model.vocab)
    context = HumanContext()
    stop = threading.Event()

    def pump() -> None:
        for msg in inp:  # blocking; ends when the port closes
            context.on_message(msg)
            if stop.is_set():
                break

    threading.Thread(target=pump, daemon=True).start()

    sec_per_step = (60.0 / bpm) / steps_per_beat
    start = time.perf_counter()
    buffer = AnticipatoryBuffer(commit_horizon=commit_horizon)
    pending_off: list[tuple[float, int]] = []
    last_fingerprint: tuple[int, ...] | None = None

    def generate_from(resume_step: float) -> int:
        """Generate one chunk (conditioned on the human) and append it from ``resume_step``."""
        primer = _context_events(context.snapshot())
        fields = codec.encode_sequence(primer, add_eos=False) if primer else None
        events = model.generate(context=fields, max_events=chunk_events, temperature=temperature)
        onset = resume_step
        scheduled = []
        for field_tuple in events:
            event = codec.decode(field_tuple)
            onset += event.dt
            scheduled.append(Scheduled(onset, event.pitch, event.velocity, float(event.dur)))
        buffer.add(scheduled)
        return len(events)

    def expired() -> bool:
        return seconds is not None and (time.perf_counter() - start) >= seconds

    try:
        while not expired():
            now = time.perf_counter()
            playhead = (now - start) / sec_per_step

            while pending_off and pending_off[0][0] <= now:  # release finished notes
                _, pitch = heapq.heappop(pending_off)
                out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))

            for sched in buffer.pop_due(playhead):  # fire notes whose moment has arrived
                out.send(mido.Message("note_on", note=sched.pitch, velocity=sched.velocity, channel=channel))
                heapq.heappush(pending_off, (start + (sched.onset + sched.dur) * sec_per_step, sched.pitch))

            fingerprint = tuple(context.snapshot())  # reconcile when the human moves
            if fingerprint and fingerprint != last_fingerprint:
                last_fingerprint = fingerprint
                resume, _ = buffer.reconcile(playhead)
                generate_from(max(resume, playhead))

            guard = 0  # keep the lookahead window filled
            while buffer.last_onset(default=playhead) < playhead + lookahead_steps and guard < 16:
                resume = max(buffer.last_onset(default=playhead), playhead)
                if generate_from(resume) == 0:
                    break
                guard += 1

            time.sleep(0.003)
    finally:
        stop.set()
        for _, pitch in pending_off:
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))
        for pitch in range(128):
            out.send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))


def _open_input(name: str):
    """Connect to an existing input port by name (e.g. an IAC bus), else create it virtually."""
    return mido.open_input(name, virtual=name not in mido.get_input_names())


def _open_output(name: str):
    """Connect to an existing output port by name, else create it virtually."""
    return mido.open_output(name, virtual=name not in mido.get_output_names())


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time responsive duet (model plays with you).")
    parser.add_argument("--checkpoint", required=True, help="trained model (.pt)")
    parser.add_argument("--in-port", default=DEFAULT_IN_PORT, help="port you play into")
    parser.add_argument("--out-port", default=DEFAULT_OUT_PORT, help="port the model plays out")
    parser.add_argument("--bpm", type=float, default=130.0)
    parser.add_argument("--steps-per-beat", type=int, default=DEFAULT_STEPS_PER_BEAT)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--chunk-events", type=int, default=8, help="notes generated per chunk")
    parser.add_argument("--lookahead", type=float, default=8.0, help="grid steps kept generated ahead")
    parser.add_argument("--commit-horizon", type=float, default=2.0, help="steps ahead locked from revision")
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    with _open_input(args.in_port) as inp, _open_output(args.out_port) as out:
        print(
            f"duet: listening to YOU on {args.in_port!r}, responding on {args.out_port!r} "
            f"at {args.bpm} BPM. Play something. Ctrl-C to stop."
        )
        try:
            duet(
                model,
                inp,
                out,
                bpm=args.bpm,
                steps_per_beat=args.steps_per_beat,
                temperature=args.temperature,
                chunk_events=args.chunk_events,
                lookahead_steps=args.lookahead,
                commit_horizon=args.commit_horizon,
                seconds=args.seconds,
            )
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
