# Liar's Chatroom — Build Map (v2, 2026-10-27)

The plan for building a competition pipeline with Claude Code before the data is
released, then adapting it fast during the 48 hours.

v2 changes: organizers banned external prediction APIs and confirmed CSV-only
submission, so open LLMs run on our own GPUs became the main model, prep was
trimmed, and expensive runs use a single holdout fold.

Claude: don't read this whole file. `grep -n "^## " BUILD_MAP.md` and read only the
section you need (see CLAUDE.md).

---

## How to use this (for me, the human)

### One-time setup
1. Create an empty PRIVATE GitHub repo `liars-chatroom`, clone it, open it in
   Claude Code. (Keep it private: sharing code with other teams is banned.)
2. Put `CLAUDE.md`, `PROGRESS.md` and `BUILD_MAP.md` in the repo root. Commit them.
3. Form a team (up to 4). Every member's Kaggle account has its own weekly GPU
   quota, so 4 members ≈ 4× the compute. Write the accounts in PROGRESS.md.
4. Make a GitHub fine-grained token with read access to the repo and store it as a
   Kaggle Secret named `GH_TOKEN` on every team account (used by the notebooks).

### Running a phase
- Start of a session: Claude Code loads CLAUDE.md automatically. Type:
  > **Continue from PROGRESS.md.**
- To start a phase, type:
  > **Start Phase N.**
- At the end, Claude runs the checks, writes the PROGRESS.md entry, commits, and
  asks whether to move on. Check the "Done when" list yourself before saying yes.
- After saying yes, run `/clear` (or open a new session) and type
  "Continue from PROGRESS.md."

### If something breaks
> Fix this without changing the canonical schema or the CLI interface: <paste error>

### If Claude drifts from the plan
> Stop. Re-read the Phase N section of BUILD_MAP.md and CLAUDE.md's phase protocol.

### If time is short
See "Phase overview → If you have fewer days" below. Cut in that order; never cut
Phases 2 or 9.

---

## Strategy in one page

**Core idea:** a format-agnostic pipeline. Every model scores "how likely is each
speaker the liar". Models are compared and combined on honest held-out
predictions, so we never waste Kaggle submissions on guessing.

**What actually wins synthetic-data competitions:** the test set comes from the
same generator as the train set. The biggest gains come from understanding
*their* generator in the first hours: its templates, lie types, label skews and
quirks. The pipeline exists so the 48 hours go into that and into improvements,
not plumbing.

**Why open LLMs are the main model:** spotting a lie often needs reasoning
("watched the sunset at the beach" when the evidence says it rained all day).
Instruction-tuned 7–9B open LLMs, fine-tuned with QLoRA, have been the strongest
models in recent Kaggle text competitions of this kind. APIs are banned, so we run
them ourselves on Kaggle GPUs. Every team has the same restriction, so whoever has
this pipeline ready gets the most out of their GPU hours.

The key pieces:
1. **Adapter layer.** All code works on one canonical schema. On competition day
   only `src/liars/adapters/competition.py` and `configs/submission.yaml` are new.
2. **Leak check + forensics.** Hour one on the day: find label skews, liar-only
   templates and lie types. If the data is simple and templated, rules may get
   close to 100% and become the main signal.
3. **Model stack** (all output the same 4-way probabilities):
   - TF-IDF baseline (minutes, CPU) — sanity check and leak detector.
   - Zero-shot NLI (no training) — fast baseline and ensemble feature.
   - Zero-shot open LLM (no training) — strong first submission.
   - **QLoRA open LLM (main model)** — fine-tuned on their train set.
   - DeBERTa focus-marked multiple choice — fast, 5-fold, different errors.
   - Day-of rules — features, never overrides.
4. **Holdout fold for expensive runs.** LLM runs train once on 4/5 of the data and
   predict fold 0; cheap models do 5-fold. The ensemble is fit on fold 0, where
   every model has honest predictions.
5. **Permutation as a switch.** Relabelling speakers removes position bias, but if
   their liar positions are skewed, that skew is real signal. The leak check decides.
6. **Synthetic data** for building and testing everything before the event, and as
   optional extra training data on the day if their train set is small.
7. **Pseudo-labelling** only if the organizers allow it (switch, default off).

---

## Shared specs

### Repo layout
```
liars-chatroom/
├── CLAUDE.md  PROGRESS.md  BUILD_MAP.md  README.md
├── pyproject.toml                # package "liars", extras: [nlp], [dev]
├── configs/                      # one YAML per experiment
│   ├── synth.yaml  baselines.yaml  nli_zeroshot.yaml  llm_zeroshot.yaml
│   ├── llm_qlora.yaml  deberta_mc.yaml  ensemble.yaml  submission.yaml
├── src/liars/
│   ├── schema.py                 # Message, Example, load/save, validate
│   ├── config.py                 # load YAML, --set overrides, seed_everything
│   ├── runlog.py                 # data/runs.csv logging
│   ├── prompt.py                 # shared chat -> text formatting + truncation
│   ├── adapters/  synthetic.py  competition.py (written on the day)
│   ├── synth/
│   │   ├── world.py  programmatic.py  verify.py
│   │   └── templates/  default.yaml  alt.yaml   # template packs
│   ├── cv.py  trainer.py  augment.py  leakcheck.py  score.py
│   ├── features/rules.py         # rule framework (+ day-of rules)
│   ├── models/  base.py  baselines.py  nli_zeroshot.py  llm_zeroshot.py
│   │            llm_qlora.py  deberta_mc.py
│   ├── ensemble.py  analysis.py  pseudo.py  submit.py
│   └── cli.py                    # `liars <command>`
├── notebooks/kaggle/             # thin notebooks that call the CLI
├── scripts/                      # build_wheel.sh (fallback install), gpu_info.py
├── docs/KAGGLE.md
├── data/   (gitignored) raw/ canonical/ oof/ preds/ subs/ models/ reports/
│           features/ samples/ cache/ bench/
└── tests/  fixtures/tiny.jsonl
```

### Canonical schema (one JSON object per line)
```json
{
  "id": "chat_0001",
  "evidence": ["The museum closed at 6 PM on Friday.", "It rained all of Saturday."],
  "messages": [
    {"speaker": "A", "text": "I went Friday afternoon, the exhibits were great."},
    {"speaker": "C", "text": "Saturday was so sunny, I spent the day at the beach."}
  ],
  "speakers": ["A", "B", "C", "D"],
  "label": "C",
  "meta": {"source": "synthetic", "pack": "default", "difficulty": "med",
           "lie_type": "implicit", "lie_msg_idx": [1],
           "name_map": {"A": "Priya", "B": "Tom", "C": "Lena", "D": "Raj"}}
}
```
- `label` is `null` for test data.
- `meta` is free-form. Synthetic data fills `lie_type` and `lie_msg_idx` so error
  analysis can break accuracy down by lie type. Real data may leave them empty.
- If raw data uses names, the adapter maps them to A–D in a deterministic order,
  keeps `name_map` in meta, and `submit.py` maps back.

### Prediction files
Parquet with columns `id, p_A, p_B, p_C, p_D` (rows sum to 1). OOF files also have
`fold`. Ensembling, analysis and submission only ever read these files.

### Benchmark sets (made in Phase 1, used by every model phase)
Our stand-in for "their data" during prep, so every model is compared on the same
thing:
- `data/bench/stage1.jsonl` — `default` pack, large, mixed difficulty (pretraining
  pool for DeBERTa / optional LLM stage 1).
- `data/bench/train.jsonl` — `alt` pack, 2k, med+hard, new seed ("their train").
- `data/bench/test.jsonl` — `alt` pack, 1k, labels set to null ("their test").
- `data/bench/test_labels.csv` — hidden labels, read ONLY by `liars score`.
Every model phase reports holdout (fold 0) accuracy on bench train and accuracy on
bench test via `liars score`.

### CLI commands (final set)
`validate, stats, synth, adapt, folds, leakcheck, train, predict, score, rules,
rules-report, ensemble, analyze, pseudo, submit`

---

## Phase overview

| # | Phase | Time-box | Priority |
|---|-------|----------|----------|
| 0 | Scaffold, schema, CLI | 1–2 h | Must |
| 1 | Synthetic generator + benchmark sets (lean) | 3–4 h | Must |
| 2 | CV, trainer, baselines, leak check, submit, score | 2–3 h | Must |
| 3 | Zero-shot NLI | 2 h | Must |
| 4 | Kaggle packaging + first GPU run | 2 h | Must |
| 5 | Open-LLM zero-shot scoring + model benchmark | 3–4 h | Must |
| 6 | QLoRA open-LLM classifier (main model) | 5–8 h | Must |
| 7 | DeBERTa focus-marked multiple choice | 3–4 h | Should |
| 8 | Ensemble, error-analysis report, rules framework | 3 h | Must |
| 9 | Dress rehearsal on an unseen format (timed) | 3–5 h | Must |
| 10 | Pseudo-labelling (switch, default off) | 1–2 h | Only if allowed |

Total Must ≈ 24–33 h of work plus GPU waiting time.

### If you have fewer days
- **≥ 10 days:** do everything in order.
- **6–9 days:** Phase 7 debug-only (code + CPU test, no GPU runs). Phase 1 with
  one pack + a small alt pack.
- **3–5 days:** 0 → 2 → 3 → 4 → 5 → 6 (debug + one real run) → 8 (ensemble +
  submit only) → 9. Skip 7 and 10.
- **< 3 days:** 0 → 2 → 5 → 9, with Phase 1 cut to a 300-line generator just
  good enough to test on.

---

## Phase 0 — Scaffold, schema, CLI (time-box 1–2 h)

**Goal:** an installable, tested skeleton that everything else plugs into.

**Spec**
- Pip-installable package `liars`, src layout, `pyproject.toml`, Python >= 3.10.
  Extras: `[dev]` = pytest; `[nlp]` = torch, transformers, datasets, peft,
  bitsandbytes, accelerate, sentencepiece. Core: pandas, numpy, pyarrow, pyyaml,
  typer, scikit-learn, scipy, tqdm.
- `schema.py`: dataclasses `Message(speaker, text)` and
  `Example(id, evidence, messages, speakers, label, meta)`; `load_jsonl`,
  `save_jsonl`; `validate(example) -> list[str]` (errors) and `validate_file(path)`.
  Checks: exactly 4 unique speakers; label in speakers or None; every message
  speaker in speakers; evidence and messages non-empty; ids unique within a file.
- `config.py`: `load_config(path, overrides)` supporting `--set a.b=value`;
  `seed_everything(seed)`.
- `runlog.py`: `log_run(...)` appending to `data/runs.csv` (columns in CLAUDE.md).
- `tests/fixtures/tiny.jsonl`: 32 hand-written examples, 8 per label, 6–12
  messages each. Lie types: number, time, day, place, owner, negation, implicit,
  two-message lie. Include honest hedges ("I think...") and honest speakers who
  mention the same entity as the liar.
- Typer CLI `liars` with all commands from Shared specs as placeholders.
  Implement now: `liars validate --in FILE` and `liars stats --in FILE` (count,
  label distribution, messages per chat, facts per chat, chars per chat
  p50/p90/p99).
- `.gitignore`: data/, models/, *.parquet, .venv/, __pycache__/, outputs/.
- README.md with a heading per phase.
- Fill "Key facts → Python / env" in PROGRESS.md.

**Done when**
- [ ] `pip install -e ".[dev]"` works in a fresh venv.
- [ ] `liars --help` lists all commands.
- [ ] `liars validate --in tests/fixtures/tiny.jsonl` reports 0 errors.
- [ ] `liars stats --in tests/fixtures/tiny.jsonl` prints sensible numbers.
- [ ] `pytest -q` passes, including tests that bad examples are rejected.

---

## Phase 1 — Synthetic generator + benchmark sets (time-box 3–4 h, lean)

**Goal:** correctly labelled practice chats to build and test every model on, in a
generator whose style can be swapped in about an hour on competition day. This is
a test tool, not a product: stop polishing once the Done-when list passes.

**Why it works:** generate a hidden "world" first (people, places, events, objects
with attributes). Evidence facts and honest lines are derived from it. The liar
flips one attribute. Because the world is known, every label is guaranteed correct.

**Spec**
- `world.py`: seeded `World` with 4–8 entities of types person, place, event,
  object. Attributes: time, day, location, owner, count, price, colour, weather,
  companion (who was with whom), happened (did / didn't happen).
- **Template packs** `src/liars/synth/templates/<pack>.yaml`: ALL surface text lives
  here, not in code — evidence phrasings, honest-message phrasings, lie phrasings,
  small talk, questions, hedges, fillers, names. At least 3 phrasings per fact
  type (5 for the most common ones), including passive voice and synonyms.
  - `default`: neutral, full sentences.
  - `alt`: casual texting — short replies, lowercase, slang, some emojis. Smaller
    than `default` is fine.
  Document the pack format at the top of `default.yaml` so a new pack can be
  written fast on the day.
- `programmatic.py`: `(World, pack, cfg) -> Example`.
  - Evidence: 3–8 facts. Chat: 8–24 messages, interleaved across A–D.
  - Message kinds: true restatement, neutral / unverifiable, small talk, question,
    hedged true statement, and the lie.
  - Lie types (`meta.lie_type`): number_change, time_shift, day_change,
    location_swap, owner_swap, negation, implicit (implies a false fact without
    stating it), two_step (visible only by combining two of the liar's messages).
    `meta.lie_msg_idx` is a list.
  - Hard negatives: honest speakers mention the same entity as the lie with a
    correct or neutral detail; honest speakers state unverifiable facts (not in
    evidence ≠ lie); honest speakers hedge.
  - Each message internally records its claim (entity, attribute, value) so
    `verify.py` can check it. Not in the output except optional debug meta.
- `configs/synth.yaml`: difficulty presets easy / med / hard (chat length, number of
  facts, share of implicit and two_step lies, hard-negative rate);
  `liar_position: uniform` or explicit probabilities for A–D;
  `speaker_names: letters | first_names`.
- `verify.py`: exactly one liar; the lie really conflicts with the World; no honest
  message conflicts; evidence consistent with the World; label balance;
  duplicate-chat rate. Any error is a bug.
- `adapters/synthetic.py` + CLI:
  `liars synth --n N --seed S --difficulty {easy,med,hard,mixed} --pack NAME --out PATH [--hide-labels LABELS.csv]`.
  Runs verify automatically and prints stats. `--hide-labels` writes labels to a
  CSV and nulls them in the JSONL (used for bench test).
- Write 20 random examples, readable (evidence, then chat), to
  `data/samples/synth_<pack>_<difficulty>.txt` so I can read them.
- Generate the benchmark sets from Shared specs: `stage1.jsonl` (~15k default,
  mixed), `train.jsonl` (2k alt, med+hard, new seed), `test.jsonl` +
  `test_labels.csv` (1k alt, another seed).

**Done when**
- [ ] `liars synth --n 5000 --seed 1 --difficulty med --pack default` finishes in under 1 minute.
- [ ] verify reports 0 errors for every difficulty and both packs.
- [ ] Label balance within ±2% when uniform; a skewed `liar_position` setting is respected.
- [ ] All four benchmark files exist; `liars validate` passes on them.
- [ ] I have read the sample files and confirmed they read naturally (Claude asks me).
- [ ] `pytest -q` passes.

**Learning checkpoint:** Claude explains in the PROGRESS entry how a known world
guarantees labels, and why hard negatives matter.

---

## Phase 2 — CV, trainer, baselines, leak check, submit, score (time-box 2–3 h)

**Goal:** the end-to-end train → held-out preds → submission loop, plus the leak
detector for day 1.

**Spec**
- `cv.py`: `assign_folds(examples, k=5, seed)`, stratified by label (and by
  `meta.source` if present). `liars folds --in FILE --k 5` writes
  `data/canonical/folds_<name>.csv`.
- `models/base.py`: abstract `Model` with `fit(train, val, cfg)` and
  `predict_proba(examples) -> DataFrame[id, p_A, p_B, p_C, p_D]`.
- `trainer.py` + `liars train --config CFG [--test FILE] [--debug] [--set ...]`:
  - `train.mode: kfold` — for each fold, fit on the others, predict the held-out
    fold and the test set; average test preds over folds.
  - `train.mode: holdout` — fit once on all folds except `cv.holdout_fold`
    (inner 5% split of the training folds for early stopping), predict the holdout
    fold and test.
  - Optional `train.refit_full: true` (holdout mode only): after the holdout run,
    retrain on 100% of train with the same number of steps and use that model's
    test preds (OOF stays from the holdout run).
  - Save `data/oof/<run>.parquet` and `data/preds/<run>.parquet`; log to
    `data/runs.csv`. Run name = config name + short timestamp.
- `models/baselines.py`: (a) majority class; (b) TF-IDF + logistic regression as a
  conditional 4-way choice: one feature row per speaker (that speaker's text plus
  position and count features), softmax across the 4 speakers.
- `score.py` + `liars score --preds FILE --labels CSV`: accuracy (+ per lie_type if
  the canonical file is given). Only used on synthetic benchmark test.
- `leakcheck.py` + `liars leakcheck --in FILE`, writes
  `data/reports/leakcheck_<name>.md` with:
  - label distribution, and label vs order of first appearance;
  - label vs number of messages, mean message length, first/last message speaker;
  - n-grams most over-represented in liar lines vs honest lines (chi²);
  - the most frequent message templates (numbers and names masked) and whether any
    template is used only by liars;
  - word overlap between each speaker's lines and the evidence, by label;
  - TF-IDF baseline OOF accuracy and its top 30 features.
- `submit.py` + `liars submit --preds FILE --out data/subs/NAME.csv`, driven by
  `configs/submission.yaml`: id column name, label column name, label format
  (letter | int 0–3 | name), speaker order for ints (as listed | first appearance).
  Optional `--sample sample_submission.csv`: match its columns, order and ids.
  Validate: every test id exactly once, valid labels, correct column order.
  Test: a canonical → submission round trip reproduces train labels exactly.

**Done when**
- [ ] `liars folds` → `liars train --config configs/baselines.yaml --test data/bench/test.jsonl` → `liars score` → `liars submit` works on the benchmark sets, in both kfold and holdout mode.
- [ ] `data/runs.csv` has the runs with per-fold accuracies.
- [ ] `liars leakcheck` runs on `data/bench/train.jsonl`. If TF-IDF accuracy is well above chance (say > 35%) or a template appears only in liar lines, our generator leaks: report it and fix it in Phase 1 code before moving on.
- [ ] `pytest -q` passes.

---

## Phase 3 — Zero-shot NLI (time-box 2 h)

**Goal:** a no-training baseline and an ensemble feature. Don't over-polish.

**Spec**
- `models/nli_zeroshot.py`, `configs/nli_zeroshot.yaml`.
- Model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (base variant as
  a speed option). Debug: a tiny NLI cross-encoder. Read the checkpoint's
  `id2label` to find the contradiction index; never hardcode it.
- Scoring modes (config switch), each giving a per-speaker score, then a softmax
  with a temperature:
  - `pairwise`: premise = each fact, hypothesis = each message; speaker score =
    max contradiction prob over their messages × facts.
  - `joint`: premise = all evidence; hypothesis = each message; max over the
    speaker's messages.
  - `joint_concat`: premise = all evidence; hypothesis = all of the speaker's
    messages joined.
- Configurable filter to skip questions and very short messages.
- Batched, fp16 on GPU; cache pair scores to `data/cache/` keyed by text hash.
- `fit` only tunes the temperature and picks the best mode on the training folds.
- Report accuracy per mode, per `meta.lie_type` and per difficulty.

**Done when**
- [ ] `liars train --config configs/nli_zeroshot.yaml --debug` runs on CPU in under 2 minutes.
- [ ] A second run is clearly faster (cache works).
- [ ] A per-mode, per-lie_type table exists for a small CPU subset of bench train.
- [ ] `pytest -q` passes.

**Learning checkpoint:** Claude explains entailment / contradiction / neutral and
why `pairwise` might miss two_step lies.

---

## Phase 4 — Kaggle packaging + first GPU run (time-box 2 h)

**Goal:** any CLI command can run on a Kaggle GPU in a few clicks, and outputs come
back into `data/`.

**Spec**
- Install on Kaggle with internet on (allowed — CSV-only competition):
  `pip install "git+https://${GH_TOKEN}@github.com/<user>/liars-chatroom.git#egg=liars[nlp]"`
  with `GH_TOKEN` read from Kaggle Secrets. Fallback: `scripts/build_wheel.sh`
  builds a wheel to upload as a private Kaggle Dataset.
- Every config accepts a hub id or a local path for models, plus
  `paths.data_dir` / `paths.out_dir` overrides.
- `scripts/gpu_info.py`: prints GPU name, memory, torch / CUDA / bitsandbytes
  versions, and whether 4-bit loading works. Every notebook runs it first.
- `notebooks/kaggle/run_cli.ipynb`: ONE generic notebook. Cells: install, set
  paths, `gpu_info`, then a single cell where I paste the `liars ...` commands,
  then zip `data/oof`, `data/preds`, `data/runs.csv`, `data/reports` into
  `/kaggle/working/outputs.zip`.
- `docs/KAGGLE.md`: uploading the benchmark / competition data as a private
  dataset; notebook settings (T4 x2, internet on, persistence); running long jobs
  with "Save Version → Run All" so they continue when the browser is closed;
  downloading outputs and unzipping into `data/`; merging `runs.csv`; using
  several team accounts at once.
- First real run: NLI zero-shot (`joint` or best mode) on full bench train +
  bench test on a T4.

**Done when**
- [ ] The NLI run finishes on Kaggle and its OOF / preds files are back in `data/`; `liars score` works on them.
- [ ] GPU time for the run is recorded in PROGRESS.md.
- [ ] `docs/KAGGLE.md` is complete enough to follow at 3 AM.

---

## Phase 5 — Open-LLM zero-shot scoring + model benchmark (time-box 3–4 h)

**Goal:** a strong no-training model (our first real submission on the day) and
a decision on which open LLM to fine-tune in Phase 6.

**Spec**
- `prompt.py` (shared with Phase 6): builds the prompt text from an example —
  numbered evidence, the chat as "S: text" lines, and the question "Which speaker
  (A, B, C or D) makes a statement that contradicts the evidence? Answer with one
  letter." Uses the model's chat template. Truncation policy: keep all evidence;
  if too long, drop the messages furthest from any other message about the same
  entity first; log how many examples were truncated. Config: `max_len`.
- `models/llm_zeroshot.py`, `configs/llm_zeroshot.yaml`:
  - Load a local instruct model in 4-bit NF4 (bitsandbytes), fp16 compute (T4 has
    no bf16). Debug: a ~0.5B instruct model on CPU without quantisation.
  - Score: logits at the answer position for the 4 letter tokens (check how each
    letter tokenises, with and without a leading space; use the variant the chat
    template actually produces). Softmax over those 4. No free-text parsing.
  - Optional few-shot examples drawn from the training folds only.
  - Permutation TTA via `augment.tta_variants` (built here; see Phase 7 for the
    full augment spec — implement `permute_speakers` and `tta_variants` now).
  - `fit` only selects few-shot examples and a temperature.
  - Print chats per minute.
- **Model benchmark:** Claude checks the Hugging Face Hub for current open
  instruct models with licences that allow this use and proposes 2–3 candidates
  in the 7–14B range (plus one ~3B for speed). Run each zero-shot on 500 bench
  train examples on a T4. Table: accuracy, accuracy per lie_type, chats/min,
  GPU memory, any fp16 NaN problems.

**Done when**
- [ ] Debug run works on CPU in under 2 minutes.
- [ ] A test checks that TTA maps permuted predictions back to the right original speaker.
- [ ] The benchmark table is in PROGRESS.md, and "LLM benchmark winner" in Key facts is filled.
- [ ] Holdout + bench test accuracy for the winner are logged.
- [ ] `pytest -q` passes.

---

## Phase 6 — QLoRA open-LLM classifier, the main model (time-box 5–8 h)

**Goal:** the strongest single model, with known GPU cost, ready to fine-tune on
their data the moment it arrives.

**Spec**
- `models/llm_qlora.py`, `configs/llm_qlora.yaml`. Base: the Phase 5 winner (or the
  best model that fits the time budget). 4-bit NF4, LoRA on attention + MLP
  projections (r=16, alpha=32, dropout 0.05 — all in config).
- Same prompt as Phase 5. Loss = cross-entropy over the 4 letter logits at the
  answer position only (not full-vocab LM loss).
- fp16 compute, gradient checkpointing, paged 8-bit AdamW, gradient accumulation,
  cosine LR with warmup, 1–2 epochs, early stopping on the inner split.
- Permutation augmentation behind `augment.permute_train` (default on in prep,
  decided by the leak check on the day). TTA at inference.
- Default `train.mode: holdout`. Support `train.refit_full`.
- Optional stage 1 via `init_from_run` (continue training saved adapters).
- Save only the LoRA adapters to `data/models/<run>/`.
- Before training: print the token-length distribution and an estimated time per
  epoch on a T4, measured from the first 20 steps. If a run would exceed 4 GPU
  hours, stop and ask me.
- Real runs on Kaggle:
  1. QLoRA on bench train (holdout mode) → holdout + bench test accuracy.
  2. If time allows: stage 1 on ~5k of `stage1.jsonl`, then stage 2 on bench
     train. Record whether stage 1 helped.

**Done when**
- [ ] Debug run works on CPU in under 2 minutes.
- [ ] Real run 1 is done; its accuracy vs Phase 5 zero-shot is in a table in PROGRESS.md.
- [ ] "QLoRA time per epoch on T4" in Key facts is filled, so competition-day runs can be sized.
- [ ] `pytest -q` passes.

**Learning checkpoint:** Claude explains what LoRA trains, why scoring only the 4
letter logits works as a classifier, and why the holdout fold must stay untouched.

---

## Phase 7 — DeBERTa focus-marked multiple choice (time-box 3–4 h, Should)

**Goal:** a fast 5-fold model that makes different mistakes than the LLM — a good
ensemble partner. If short on time, finish debug only.

**Spec**
- `augment.py` (permute and TTA exist from Phase 5):
  - `permute_speakers(example, rng)`: a random bijection over A–D applied to
    messages, speakers, label and name_map; returns the inverse map.
  - `shuffle_evidence(example, rng)`.
  - `tta_variants(example, n)`: n permutations; predictions mapped back with the
    inverse maps and averaged.
  - Switches: `augment.permute_train`, `augment.tta_n` (1 = off),
    `augment.shuffle_evidence`.
- `models/deberta_mc.py` (`AutoModelForMultipleChoice`), `configs/deberta_mc.yaml`:
  - 4 choices, one per speaker X: `text_a` = "Evidence: " + facts joined;
    `text_b` = full chat as "S: text" lines, X's lines wrapped in special tokens
    `<t> ... </t>` (added to the tokenizer, embeddings resized), then
    "Is the marked speaker contradicting the evidence?"
  - `max_len` from config (default 512). Truncation: keep all evidence and all of
    X's lines; drop other speakers' lines furthest from X's lines first. Log the
    truncation count.
  - Init: the NLI DeBERTa-v3 large checkpoint from Phase 3; base for speed; xsmall
    for debug.
  - fp16, gradient accumulation, cosine LR with warmup, LR 1e-5 to 2e-5,
    2–4 epochs, early stopping, optional layer-wise LR decay.
  - Optional two-stage: `--stage 1` = one run on `stage1.jsonl` with a 5% holdout;
    `--stage 2` = 5-fold with `init_from_run`. Stage 1 is a switch, not required.
  - Print the token-length distribution and estimated time per epoch for a T4.
  - Inference with TTA.
- Real run (if time): 5-fold on bench train (with and without stage 1).

**Done when**
- [ ] `--debug` stage 1 then `--debug` stage 2 each run on CPU in under 2 minutes.
- [ ] A test checks that the marked speaker's lines are never dropped by truncation.
- [ ] (If run) bench results are added to the PROGRESS.md model table.
- [ ] `pytest -q` passes.

**Learning checkpoint:** Claude explains why focus marking plus a softmax across 4
choices beats classifying each speaker independently.

---

## Phase 8 — Ensemble, error-analysis report, rules framework (time-box 3 h)

**Goal:** combine models, see errors quickly, and be ready to add day-of rules.

**Spec**
- `ensemble.py` + `liars ensemble --runs r1 r2 ... --method {mean,weighted,stack} --name NAME`:
  - Load OOF and test preds; align OOF on the ids every run has (in practice the
    holdout fold when any holdout-mode run is included); print how many rows that is.
    Assert identical test ids.
  - `weighted`: non-negative weights optimised on OOF log-loss (scipy); report
    accuracy with an inner CV on the aligned rows so the score is honest.
  - `stack`: per-speaker features = each run's p_X, log p_X, rank of X, plus rule
    scores if `data/features/rules.parquet` exists; conditional logit (shared
    weights across speakers, softmax across the 4). Nested CV for an honest score.
  - Default output = best by honest OOF; only pick `stack` if it beats `weighted`
    by more than 0.3 points.
  - Print a comparison table: each run, mean, weighted, stack. Save
    `data/preds/ens_<name>.parquet` (and the matching OOF).
- `analysis.py` + `liars analyze --run RUN [--compare RUN2] --n 50`, writes
  `data/reports/<run>.html` (plain HTML, no server):
  - accuracy overall, per fold, per lie_type, per chat-length bucket, per true label
    (position-bias check); A–D confusion matrix;
  - calibration: accuracy by confidence bucket;
  - 50 wrong examples sorted by confidence: evidence, chat with the true liar's
    lines in red and the predicted liar's in blue, probabilities;
  - with `--compare`: examples where the two runs disagree, and who was right.
- `features/rules.py` — framework now, real rules on the day:
  - `Rule` interface: `score(example) -> {speaker: float}`, plus a registry.
  - Two generic example rules: number mismatch near the same noun; day/time
    mismatch after normalising ("6 PM" == "18:00", "Fri" == "Friday").
  - `liars rules --in FILE --out data/features/rules.parquet` (long format: id,
    speaker, rule, score).
  - `liars rules-report --in FILE`: per rule and per lie_type, coverage (how often
    it fires) and precision (how often it points at the true liar).
- Run the ensemble on all bench runs so far; record the final bench table.

**Done when**
- [ ] `liars ensemble` on the bench runs: honest accuracy ≥ the best single model, with the weight table shown.
- [ ] The HTML report opens and is readable; Claude tells me the file path.
- [ ] `liars rules-report` works on bench train.
- [ ] `pytest -q` passes.

---

## Phase 9 — Dress rehearsal on an unseen format (timed, Must)

**Goal:** prove you can go from a strange raw format to a validated submission in
under 90 minutes, and find what slows you down.

The rehearsal data must be made by someone who is not the session doing the
rehearsal, otherwise the adapter "already knows" the format.

**Step A — generate hidden data (one separate session)**
In a new session, say: **"Start Phase 9, step A."** Claude then:
- Writes a generator in `~/liars_rehearsal_gen/` (OUTSIDE the repo; never committed).
- Writes only these into the repo's `data/raw/`: `train.csv`, `test.csv`,
  `sample_submission.csv`. Hidden labels go to `~/liars_rehearsal_gen/test_labels.csv`.
- Makes the format deliberately different from canonical:
  - columns like `conversation_id, facts` (newline-separated or numbered),
    `dialogue` (lines like "Priya: text", real first names), `liar_name`;
  - sample submission with integer labels 0–3 in order of first appearance;
  - new phrasing, at least one lie type we never implemented, a skewed liar
    position, and a generator quirk that leaks the label weakly (so the leak check
    has something to find);
  - about 3k train and 1k test.
- Writes a scorer `~/liars_rehearsal_gen/score.py SUB.csv`.
- Does NOT describe the format, quirks or code in its reply or in PROGRESS.md. It
  only records "rehearsal data generated" and the scorer path.
- Ends the session.

**Step B — rehearse**
1. Open a fresh session, set `Mode: REHEARSAL` in PROGRESS.md, start a timer.
2. Follow the Competition-day runbook below exactly, prompt by prompt, at least up
   to and including "Main models" (launch them for real on Kaggle).
3. Score submissions with the hidden scorer only after each one is made.

**Step C — post-mortem**
Say: **"Phase 9 post-mortem."** Claude writes in PROGRESS.md: a timeline (when each
runbook step finished), what slowed us down, whether the leak check found the quirk,
the scores, and concrete fixes. Then it applies the fixes (code or runbook text)
after I approve.

**Done when**
- [ ] Raw files → validated first submission in under 90 minutes.
- [ ] The adapter round-trip test passes on the rehearsal data.
- [ ] The leak check found the planted quirk (if not, improve `leakcheck.py`).
- [ ] The QLoRA run was launched from the runbook prompt without code changes.
- [ ] The post-mortem and fixes are recorded; Mode is set back to PREP.

---

## Phase 10 (only if allowed) — Pseudo-labelling

Build it behind `pseudo.enabled: false`. Turn it on only after the organizers
confirm in writing; record their answer in PROGRESS.md.

**Spec**
- `pseudo.py` + `liars pseudo --preds ENSEMBLE_PREDS --test FILE --threshold 0.95 --out FILE`:
  selects test examples whose top probability ≥ threshold, writes them as canonical
  examples with `label` = argmax and `meta.source = "pseudo"`. Refuses to run if
  `pseudo.enabled` is false.
- Trainer support: `train.extra_train: [files]` — extra examples are added to the
  training side only and NEVER used for validation, so held-out scores stay honest.
- Report: how many examples pass the threshold and, on bench data (labels known),
  the accuracy of the pseudo-labels.

**Done when**
- [ ] On the bench setup, pseudo-labelling at 0.95 changes holdout accuracy by a recorded amount, and pseudo-label accuracy is reported.

---

## Competition-day runbook

Use this for the Phase 9 rehearsal and the real event. Before starting, set
`Mode: COMPETITION` (or `REHEARSAL`) in PROGRESS.md. Paste the prompts in order.
Claude logs each step in PROGRESS.md under a "Competition log" heading, with times.

### Submission budget
5 per day × 2 days = about 10 total. Day 1: one format check, then one or two
milestones. Day 2: one or two milestones, and keep the last two for SAFE and BOLD.
Trust honest held-out accuracy over the public leaderboard. If they disagree by a
lot, suspect a train/test difference and investigate before submitting more.

### GPU plan
(Updated 2026-10-27 for a SOLO team.) One Kaggle account = one weekly quota (~30 h, resets
Fri/Sat midnight UTC; check the quota bar). Each session has 2x T4. Up to 2 sessions can run
at once (verified 2026-10-27): they halve wall-clock time but each one uses quota.
Priority order: (1) QLoRA main run, (2) LLM zero-shot safety submission, (3) DeBERTa.
Use both T4s inside one session: `liars predict --gpus 2` splits test prediction.
Size every job with `liars plan` (next sections). Never exceed 4 GPU h per job without asking.

### T+0:00 — Adapt (target: 45 min)
```
COMPETITION MODE. The data is in data/raw/. Only change src/liars/adapters/ and
configs/submission.yaml.
1) Inspect every file in data/raw/: columns, dtypes, 3 full TRAIN rows, text
   lengths, label distribution, how speakers are named and ordered, how evidence
   is stored, and the exact sample_submission format. For test: schema and counts
   only. Report the findings briefly.
2) Write src/liars/adapters/competition.py converting train and test to canonical
   (deterministic speaker -> A-D mapping, kept in meta.name_map), and
   configs/submission.yaml to match sample_submission exactly.
3) liars validate on both outputs. Assert exactly one liar per train example and
   report any violations. Write and run the round-trip test (canonical ->
   submission format reproduces the train labels).
4) liars stats: #train, #test, token lengths p50/p90/p99 for the LLM prompt and
   for DeBERTa, messages per chat, facts per chat. Say whether max_len 512
   (DeBERTa) and the LLM max_len are enough.
```

### T+0:45 — Forensics and leak hunt (target: 30 min)
```
Run liars leakcheck on the competition train. Then read 15 random TRAIN examples
yourself and tell me:
- label vs position / first appearance / message counts: is permutation safe?
- templates or words that only liars use; repeated templates in general;
- the list of lie types you can see (name each, give one example);
- how templated the data looks: could exact rules plausibly reach >95%?
- your recommendation: augment.permute_train on or off.
Record the findings in PROGRESS.md. Don't change any code.
```

### T+1:15 — Launch GPU jobs + first submission (target: 30 min)
Step 1 (you, 1 min): read off train size N, test size M, and "total chars" p50 / p90 from
`liars stats`, then run:
```
liars plan --train N --test M --chars <p50> --chars-p90 <p90>
```
It prints GPU hours per model and the exact `--set` flags, by these rules (measured on a T4,
`configs/sizing.yaml`; tokens ~ 75 + 0.38 x chars):

| Rule | When | Do |
|---|---|---|
| R1 TTA | always (LLMs) | `augment.tta_n=1` (TTA 2 gave no gain on bench; halves prediction). DeBERTa keeps TTA 2 unless it is over its cap |
| R2 both T4s | always | train job without `--test`, then `liars predict --run latest:llm_qlora --in <test> --gpus 2 --test-preds` |
| R3 epochs | QLoRA 2 epochs > 4 h | 1 epoch (bench: inner-val 98.75% after 1 epoch vs 100% best) |
| R4 subsample | 1 epoch still > 4 h | `train.subsample=K` (stratified, training folds only), K from `liars plan`; only if K >= 3000 |
| R5 max_len | still no fit and p50 > 512 tokens | `llm.max_len=512` + subsample; check the truncation count in the log |
| R6 fallback model | still no fit | `llm.model_id=google/gemma-4-E4B-it`, 1 epoch (row in PROGRESS sizing plan) |
| R7 zero-shot | its estimate <= 1 h | run it FIRST (safety submission ~1 h after launch); else skip and submit the baseline |
| R8 DeBERTa | holdout mode, 1 fit, 2 epochs <= 1.5 h | run it; else 1 epoch + TTA 1 if <= 1.5 h; else SKIP |
| R9 truncation | p90 tokens > 1024 | >10% of chats get truncated: expect lower accuracy; ask before going to max_len > 1024 (memory) |
| R10 nothing fits | rules R3-R6 all fail | STOP and ask (raise the budget up to the 12 h session limit, or rely on DeBERTa / zero-shot) |

Worked example (N 5000, M 3000, ~650 chars): QLoRA 12B 1 epoch 2.8 h + test 0.3 h, zero-shot
0.5 h, DeBERTa holdout 1.2 h -> ~4.9 GPU h total.

Step 2 (prompt):
```
liars folds on competition train (k=5). Upload the competition data dataset
(python scripts/kaggle_run.py data --slug liars-comp ...). Using the `liars plan` output,
give me the exact Kaggle push commands (with --datasets liars-comp) and expected times for:
(a) llm_zeroshot (if R7) on holdout fold + test: train job, then predict --gpus 2;
(b) llm_qlora holdout mode, permute_train per forensics, with the plan's --set flags;
(c) deberta_mc in holdout mode (if R8).
Launch (a) and (b) as two parallel sessions, (c) after (a) finishes.
Meanwhile on CPU: liars train with configs/baselines.yaml, then liars submit on
the baseline. Show its OOF accuracy and the validated CSV path.
```
Upload the baseline CSV to confirm the format. When (a) finishes, ensemble it
with NLI if available and submit as the first real milestone. The Kaggle runner re-uploads
the code snapshot by itself when HEAD changed, refuses uncommitted changes, and checks that
every attached dataset is private.

### While GPUs run — rules from their templates
```
Using the forensics, write rules in src/liars/features/rules.py that target their
exact templates (number, time, day, owner, location, negation patterns).
Run liars rules-report on competition train. Keep only rules with precision
>= 0.9 when they fire. Report coverage x precision overall.
```
If the rules alone cover most examples at very high precision, they become the
main signal and models handle the rest.

### Optional — retarget the generator (only if their train is small, < ~5k)
```
Write src/liars/synth/templates/comp.yaml imitating the competition's phrasing,
name style and lie types, based on the forensics. Add missing lie types to the
generator. Set liar_position to match their label distribution. Generate 10–20k
examples, verify them, and show me 5 samples next to 5 real train examples.
Then set up a QLoRA stage-1 run on it followed by stage 2 on their train.
```

### When the main runs land — ensemble
```
liars ensemble over <runs> (+ rules features), method stack vs weighted.
Show the table and submit the best as a milestone.
```

### Every few hours — the improvement loop
```
Run liars analyze on <best run>. Summarise the 3 most common error patterns,
2 examples each (train/holdout only), and propose the smallest fix for each
(prompt wording, truncation, new rule, targeted synthetic data, hyperparameters,
another QLoRA seed). Don't implement until I pick.
```

### Around T+24h — Pseudo-labelling (ONLY if the organizers allowed it)
```
Run liars pseudo on the best ensemble's test preds (threshold 0.95, report how
many pass). Re-run QLoRA with extra_train = the pseudo file. Compare holdout
accuracy with and without.
```

### Final 6 hours — Lock in
```
Build the final ensemble over the runs I list, stack vs weighted. Produce two
submissions: SAFE (best honest held-out score, simplest) and BOLD (best ensemble).
Then write METHODOLOGY.md: data adapter, synthetic data and template packs,
models (open LLMs run on our own GPUs, no external APIs), CV / holdout scheme,
ensemble, scores, use of synthetic / pseudo-labelled data, and exact commands to
reproduce both submissions from a clean clone. Pin package versions in
requirements-lock.txt and record the model ids and revisions used.
```
Choose the final two submissions on the Kaggle page before the deadline (if the
competition asks you to select).

---

## GPU budget notes
- Kaggle gives each account a weekly GPU quota (roughly 30 h; check your quota
  page). Do prep runs in an earlier week so competition week's quota is untouched.
- Prep budget (per account): Phase 4 NLI ~1 h, Phase 5 benchmark ~2–3 h,
  Phase 6 QLoRA ~4–8 h, Phase 7 DeBERTa ~2–4 h, Phase 9 rehearsal ~4–6 h.
- Competition weekend (team total): LLM zero-shot 1–2 h, QLoRA main 3–5 h,
  QLoRA variants 4–10 h, DeBERTa 5-fold 3–5 h, pseudo-label round 3–5 h.
- Kaggle sessions stop after about 12 h; size every run to finish well inside that.

## Fair play
- No external prediction APIs (organizer rule, 2026-10-26). Only open models run on
  our own GPUs make predictions.
- Synthetic training data is our own creation, not test data. State it openly in
  METHODOLOGY.md.
- Never label test data by hand. Claude Code does not read individual test chats.
- Never share code or predictions with other teams; the repo stays private.
