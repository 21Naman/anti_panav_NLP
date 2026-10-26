# Liar's Chatroom — standing rules for Claude Code

## The task
48-hour Kaggle competition. Each example = verified evidence facts + a group chat
with 4 speakers (A, B, C, D). Exactly one speaker makes a statement that
contradicts the evidence. Predict that speaker. Metric: accuracy.
Submission: CSV upload only (no notebook needed), max 5 per day. Top teams may be
asked for their code / notebook for review.
The real dataset is released only during the 48 hours; until then everything is
built and tested on our own synthetic data.

## Organizer rules that constrain the code (confirmed 2026-10-26)
- External prediction APIs (Claude, OpenAI, etc.) are NOT allowed for predictions.
  Open models we run ourselves (Kaggle GPU, laptop) ARE allowed.
- CSV upload only: no inference time limit, internet is fine.
- Pseudo-labelling (training on model-generated test predictions): NOT YET
  CONFIRMED. Code for it stays behind a config switch that defaults to off.
- No manual labelling of test data. No sharing code or predictions with other teams.

## The three files that matter
- `CLAUDE.md` (this file): rules and workflow. Loaded automatically every session.
- `PROGRESS.md`: the handoff log and the source of truth for what is done.
  Read it fully at the start of every session.
- `BUILD_MAP.md`: the plan. Do NOT read the whole file. Find the section you need
  with `grep -n "^## " BUILD_MAP.md`, then Read only that section (offset/limit).
  You will usually need: the current phase section, and sometimes "Shared specs".

## Session start (every new session)
1. Read `PROGRESS.md` fully.
2. Run `git status` and `git log --oneline -8`.
3. Tell me in 5 lines or fewer: mode, current phase and its status, anything left
   unfinished, and what you plan to do next.
4. Wait for my go-ahead, unless my message already says what to do.

## Phase protocol (follow exactly)
- Work on ONE phase at a time. Never start the next phase without my explicit
  "yes" / "go" / "next".
- **Start of a phase:** read that phase's section of BUILD_MAP.md. Restate the goal
  and the "Done when" checklist in a few lines. If you need decisions from me,
  ask them all at once, in one message. Otherwise begin.
- **During a phase:** small local commits are fine (`phase N: ...`). If you hit a
  blocker, or want to deviate from BUILD_MAP.md (different library, different
  design, skipped item), stop and ask first.
- **Respect the time-box** given for each phase in BUILD_MAP.md. If you are clearly
  going to overrun it, stop, say what is left, and ask whether to cut scope.
- **End of a phase**, in this order:
  1. Run every "Done when" check yourself and show the real output (trimmed).
  2. Run `pytest -q`.
  3. Append this phase's entry to the Phase log in `PROGRESS.md` using the template
     there, and update the "Current state" and "Key facts" blocks at the top.
  4. Update the README.md section for this phase (what it does, how to run).
  5. Commit locally: `phase N: <one-line summary>`. Do not push unless I ask.
  6. Send me a short summary (what was built, key numbers, open issues), then ask:
     **"Phase N is done. Move on to Phase N+1 (<name>)?"** — and STOP.
- If a check fails and you cannot fix it, mark the phase `partial` in PROGRESS.md,
  explain why, and ask me what to do.
- When I say "next", suggest I start a fresh session (`/clear`) if the context
  is getting long. PROGRESS.md carries everything over.

## How to write PROGRESS.md entries
- Write for a fresh Claude session that has never seen this chat. It must be able
  to continue from PROGRESS.md alone.
- Be concrete: file paths, exact CLI commands, config names, run names, seeds,
  metric numbers, data sizes, timings, GPU hours.
- Record every decision and why, and every deviation from BUILD_MAP.md (also add
  it to the Decisions register table).
- Keep each phase entry to about 40 lines or fewer. No pasted code or long logs;
  point to files instead.
- Never delete old entries. Correct them with a dated note.

## Hard code rules
- All code reads and writes ONLY the canonical JSONL schema in
  `src/liars/schema.py`. Raw formats are handled only in `src/liars/adapters/`.
- Every model implements `models/base.py`: `fit(train, val, cfg)` and
  `predict_proba(examples) -> DataFrame[id, p_A, p_B, p_C, p_D]` (rows sum to 1).
- Every training run uses the fold column from `cv.py`. Two run modes:
  - `train.mode: kfold` — 5 folds; OOF covers every train row.
  - `train.mode: holdout` — train on all folds except `cv.holdout_fold` (default 0),
    predict that fold + test. Used for expensive LLM runs. OOF covers only the
    holdout fold. Early stopping uses a small inner split of the training folds,
    never the holdout fold.
  Both save OOF preds to `data/oof/<run>.parquet` and test preds to
  `data/preds/<run>.parquet`.
- Log every run to `data/runs.csv`: run name, config path, config hash, mode,
  per-fold accuracy, mean accuracy, wall time, GPU type, timestamp.
- All settings live in `configs/*.yaml`. No magic numbers in code. CLI may
  override with `--set key=value`.
- Seed python, numpy, torch and transformers from the config.
- Every model and generator has a `--debug` mode: tiny model, about 32 examples,
  CPU, finishes in under 2 minutes.
- Never call an external LLM API from model, generator or prediction code.
- Never read test labels. Never hand-edit predictions. Never label test data
  manually (competition rule).
- Allowed dependencies: transformers, datasets, peft, bitsandbytes, accelerate,
  sentencepiece, torch, pandas, numpy, pyarrow, scikit-learn, scipy, pyyaml,
  typer, pytest, tqdm. Ask before adding anything else.
- Every new module gets a pytest using `tests/fixtures/tiny.jsonl`. Tests must be
  fast (whole suite under about 60 s on a laptop CPU).
- Kaggle notebooks stay thin: install the repo, set paths, call the CLI. No logic.

## Token and context hygiene
- Don't print whole large files. Use `head`, `tail`, `wc -l`, or Read with limits.
- Run `pytest -q`. Send training logs to a file and show only the tail.
- Don't re-read files you just wrote.
- When summarising results, prefer a small table over prose.

## Modes
`PROGRESS.md` → "Current state" → `Mode` is one of PREP, REHEARSAL, COMPETITION.
- PREP: follow the phases in order.
- REHEARSAL / COMPETITION: follow "Competition-day runbook" in BUILD_MAP.md. Only
  change `src/liars/adapters/`, `configs/`, `src/liars/features/rules.py` and
  `src/liars/synth/templates/` unless I approve something else. Speed matters:
  keep answers short, and always say how long a command will take before running
  anything that takes more than 5 minutes.
- In COMPETITION mode you may read TRAIN examples freely (forensics). For the TEST
  set, look only at schema, counts and length statistics — never read individual
  test chats or reason about who the liar is in them. Predictions on test come
  only from the models and rules, so no one can say the test set was labelled by
  hand or by an outside AI.
