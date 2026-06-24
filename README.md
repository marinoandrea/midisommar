# osc-genai

A Python foundation for feeding (eventually ML-generated) MIDI into Ableton Live over OSC.

It talks **directly** to [AbletonOSC](https://github.com/ideoforms/AbletonOSC) using
[`python-osc`](https://pypi.org/project/python-osc/) — no `pylive` wrapper. (`pylive` is a
convenience layer over AbletonOSC and, per its own docs, isn't meant for writing MIDI
notes; note-writing goes through AbletonOSC's OSC endpoints regardless, so we use them
directly.)

The full path now works: a **from-scratch generative model** (a factored per-event GRU) learns
from a MIDI corpus and its output is written into Live as clips, plus a real-time MIDI duet engine.
`uv run osc-genai` remains the minimal plumbing demo (a hardcoded C-major scale through the
`generate_notes()` seam); the model and live workflows are documented below.

## How it fits together

```
ableton.py    ->  AbletonOSC client: send commands, block on query replies, read/write clip notes
generate.py   ->  Note tuple + generate_notes() (the minimal "ML magic" seam)
main.py       ->  the plumbing demo: read track 0, generate, create clip, add notes
repr/vocab/model/train/data/generation.py  ->  the from-scratch model pipeline (see below)
live.py       ->  real-time MIDI-port duet engine
```

AbletonOSC listens on UDP **11000** (commands) and replies on UDP **11001**.

## Setup

```bash
uv sync
```

### Install AbletonOSC into Ableton Live (required for the real run)

1. Ableton Live 11 or 12.
2. Clone AbletonOSC into the Remote Scripts directory — on macOS:
   `~/Music/Ableton/User Library/Remote Scripts/AbletonOSC`
   ```bash
   git clone https://github.com/ideoforms/AbletonOSC.git \
     ~/Music/Ableton/User\ Library/Remote\ Scripts/AbletonOSC
   ```
3. In Live: *Preferences → Link/Tempo/MIDI*, set a **Control Surface** to `AbletonOSC`.

## Run

```bash
uv run osc-genai
```

Expected against a live set: prints the track count and track 0's name, then a new clip
appears in slot 0 of track 0 containing the generated notes. Set `FIRE_AFTER_WRITE = True`
in `main.py` to auto-play it.

## Verify without Ableton

A mock that mimics AbletonOSC's ports lets you exercise the full send/receive path with no
Ableton:

```bash
# terminal 1
uv run python scripts/mock_ableton.py
# terminal 2
uv run osc-genai
```

The mock logs the `create_clip` / `add/notes` messages it receives and answers the
track-name and track-count queries.

The mock **refuses to start if a real AbletonOSC is already answering on port 11000** (so an
open Live set can't silently shadow it and receive the test's note writes). Pass
`--recv-port`/`--reply-port` to run it on isolated ports, or `--force` to override. The automated
tests (`uv run pytest`) use private ports (11900/11901) and never touch 11000/11001.

## Model pipeline (train on your own MIDI, generate into Live)

The generative model is built from scratch (no pretrained weights). A note is four factored fields
— pitch, onset-Δ, duration, velocity — and a GRU predicts the next note one at a time (O(1) per
event, which the live duet needs).

```
src/osc_genai/
  repr.py        Note <-> factored Event (grid-quantised; lossless for on-grid input)
  vocab.py       Event <-> model field indices (clamping, velocity bins, EOS)
  model.py       factored per-event GRU + per-field heads; generate()/sample()
  train.py       teacher-forced training loop, checkpoints, the train CLI
  data.py        load .mid / capture Ableton clips + augmentation (transpose, jitter, scale)
  generation.py  model -> Note phrases; the generate-into-Live CLI
```

**Train** on a folder of `.mid` files (searched recursively; transposition augmentation):

```bash
uv run osc-genai-train --data-dir data/MIDI --out models/acid_v1.pt --epochs 40 --transpose 5
```

**Generate** a phrase into a Live clip (optionally primed on a context clip to "respond" to it):

```bash
uv run osc-genai-generate --checkpoint models/acid_v1.pt --track 0 --slot 0 --temperature 0.95
# respond to the clip in track 2 / slot 0, writing the answer to track 0 / slot 1:
uv run osc-genai-generate --checkpoint models/acid_v1.pt \
  --context-track 2 --context-slot 0 --track 0 --slot 1
```

Training data can also be captured straight out of Live with `data.capture_from_ableton(...)`
(via `ableton.get_clip_notes`) instead of `.mid` files.

## Live duet (real-time)

These tools run over **virtual MIDI ports** (not OSC — AbletonOSC is for clip/LOM control, not
low-latency note streams):

* `osc-genai-play` — stream the model's output continuously (one-way live generation).
* `osc-genai-duet` — a real-time duet: generates a complementary line *ahead* (a lookahead buffer),
  plays from it, and **reconciles** (revises the not-yet-played notes) when your recent notes change.
  Tune with `--lookahead`, `--commit-horizon`, `--chunk-events`.
* `osc-genai-live` — the M1 rule-based harmonizer (no model).

```bash
uv run osc-genai-play --checkpoint models/acid_v1.pt   # model plays; route osc-genai out -> a synth
uv run osc-genai-duet --checkpoint models/acid_v1.pt   # model plays *with* you
```

Route in Ableton: enable **`osc-genai out`** as an *Input* (a synth track's MIDI From) and, for the
duet, **`osc-genai in`** as an *Output* (MIDI To from your playing). The macOS **IAC Driver** works too.

### Mock the duet locally (no controller)

`osc-genai-fake-human` loops a MIDI line into the duet's input, so you can test it with no controller
— two terminals:

```bash
uv run osc-genai-duet --checkpoint models/acid_v1.pt   # terminal 1: creates the ports, listens
uv run osc-genai-fake-human --from-data data/MIDI      # terminal 2: loops one of your clips in
```

Route `osc-genai out` to a synth to hear the response (`--midi FILE`, a built-in acid pattern when no
data is given, and `--bpm` are also available).

The *scheduling* is anticipatory (it plays ahead and reconciles); the *model* is still solo-trained,
so it follows rather than jointly predicts you — that deeper anticipation needs duet-trained data
(M4/M5).

## Tweaking

- `TRACK_INDEX` / `CLIP_SLOT` in `main.py` — which track/slot to write (hardcoded to 0/0).
- `generate_notes()` in `generate.py` — replace the hardcoded melody with a model.
