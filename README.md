# Liar's Chatroom

Pipeline for a 48-hour Kaggle competition: given evidence facts and a 4-speaker
group chat, predict the one speaker whose statement contradicts the evidence.

- Plan: `BUILD_MAP.md` · Handoff log: `PROGRESS.md` · Rules for Claude Code: `CLAUDE.md`
- All code reads and writes one canonical JSONL schema (`src/liars/schema.py`).

## Setup (laptop, CPU only)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest
pip install -e ".[dev,nlp]"      # + torch / transformers (needed from Phase 3)
pytest -q
```

Local runs are CPU only (never MPS). Use Python 3.11, not the Mac default 3.14.

## Phase 0 — Scaffold, schema, CLI

What it does:
- `src/liars/schema.py`: `Message`, `Example`, `load_jsonl`, `save_jsonl`,
  `validate(example)`, `validate_file(path)`. Speakers must be exactly A–D,
  label is one of them or `null` (test), evidence and messages are non-empty, ids
  are unique, the liar has at least one message, and `meta.lie_msg_idx` (if
  present) points at the liar's lines.
- `src/liars/config.py`: `load_config(path, overrides)` with `--set a.b=value`
  style overrides (YAML-typed values, `1e-5` reads as a float), `config_hash`,
  `seed_everything` (python, numpy, torch, transformers).
- `src/liars/runlog.py`: `log_run(...)` appends to `data/runs.csv`.
- `src/liars/stats.py`: numbers behind `liars stats`.
- `tests/fixtures/tiny.jsonl`: 32 hand-written chats, 8 per label, each of the 8
  lie types once per label, with honest hedges and honest mentions of the lied-about
  entity.

How to run:
```bash
liars --help
liars validate --in tests/fixtures/tiny.jsonl
liars stats --in tests/fixtures/tiny.jsonl
```
Commands marked `(todo: Phase N)` are placeholders that exit with code 2.

## Phase 1 — Synthetic generator + benchmark sets

What it does: builds a hidden "world" per chat (events, objects, places, people with
day / time / location / owner / price / count / weather / happened / companion), reads
evidence and honest lines off it, and has one speaker state a value the world does
not have. Every message records its claims, so `verify` proves the label is right.
- `src/liars/synth/world.py` — world + `alternative_value` (wrong values for lies).
- `src/liars/synth/programmatic.py` — chat assembly, 8 lie types (number_change,
  time_shift, day_change, location_swap, owner_swap, negation, implicit, two_step),
  hard negatives (honest lines about the lied-about entity, true "not" statements,
  true implied facts, unverifiable facts, hedges, honest two-message pairs).
- `src/liars/synth/verify.py` — exactly one liar, lie contradicts the evidence, no
  honest line contradicts the world, label balance, duplicate rate.
- `src/liars/synth/templates/{default,alt}.yaml` — ALL wording. Format documented at
  the top of `default.yaml`. Liars and honest speakers share the same templates.
- `src/liars/adapters/synthetic.py` — write canonical JSONL, hide labels, readable samples.
- `configs/synth.yaml` — world sizes, difficulty presets, `liar_position`, `speaker_names`.

How to run:
```bash
liars synth --n 5000 --seed 1 --difficulty med --pack default --out data/canonical/x.jsonl
liars synth --n 1000 --seed 3 --difficulty med,hard --pack alt --out data/bench/test.jsonl \
    --hide-labels data/bench/test_labels.csv
liars synth ... --set 'synth.liar_position={A: 0.4, B: 0.2, C: 0.2, D: 0.2}'
```
Readable samples (lie lines marked) go to `data/samples/synth_<pack>_<difficulty>.txt`.

Benchmark sets (regenerate with these exact commands; `data/` is not in git):
```bash
liars synth --n 15000 --seed 101 --difficulty mixed --pack default --id-prefix s1_ --out data/bench/stage1.jsonl
liars synth --n 2000 --seed 202 --difficulty med,hard --pack alt --id-prefix tr_ --set synth.speaker_names=first_names --out data/bench/train.jsonl
liars synth --n 1000 --seed 303 --difficulty med,hard --pack alt --id-prefix te_ --set synth.speaker_names=first_names --out data/bench/test.jsonl --hide-labels data/bench/test_labels.csv
```

## Phase 2 — CV, trainer, baselines, leak check, submit, score

## Phase 3 — Zero-shot NLI

## Phase 4 — Kaggle packaging + first GPU run

## Phase 5 — Open-LLM zero-shot scoring + model benchmark

## Phase 6 — QLoRA open-LLM classifier

## Phase 7 — DeBERTa focus-marked multiple choice

## Phase 8 — Ensemble, error analysis, rules

## Phase 9 — Dress rehearsal

## Phase 10 — Pseudo-labelling (only if allowed)
