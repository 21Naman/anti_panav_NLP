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

## Phase 2 — CV, trainer, baselines, leak check, submit, score

## Phase 3 — Zero-shot NLI

## Phase 4 — Kaggle packaging + first GPU run

## Phase 5 — Open-LLM zero-shot scoring + model benchmark

## Phase 6 — QLoRA open-LLM classifier

## Phase 7 — DeBERTa focus-marked multiple choice

## Phase 8 — Ensemble, error analysis, rules

## Phase 9 — Dress rehearsal

## Phase 10 — Pseudo-labelling (only if allowed)
