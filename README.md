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

Benchmark sets (regenerate with these exact commands; `data/` is not in git). They were made
before the ambiguity fix, so they need `synth.avoid_ambiguous_anaphors=false` to come out identical:
```bash
A=--set=synth.avoid_ambiguous_anaphors=false
liars synth --n 15000 --seed 101 --difficulty mixed --pack default --id-prefix s1_ $A --out data/bench/stage1.jsonl
liars synth --n 2000 --seed 202 --difficulty med,hard --pack alt --id-prefix tr_ --set synth.speaker_names=first_names $A --out data/bench/train.jsonl
liars synth --n 1000 --seed 303 --difficulty med,hard --pack alt --id-prefix te_ --set synth.speaker_names=first_names $A --out data/bench/test.jsonl --hide-labels data/bench/test_labels.csv
```

Ambiguity fix (2026-10-27): `synth/ambiguity.py` flags "anchor ... it ..." pairs where another
speaker mentions a different evidence item in between, so that the "it" line can be read as
being about that item: an honest line then contradicts the evidence, or the lie reads as true.
With `synth.avoid_ambiguous_anaphors: true` (default now) the generator re-draws those chats and
`verify` treats any left as errors; the verify summary always prints the count. The frozen
bench has 13.4% such chats (train 268/2000, test 132/1000). On them QLoRA scores 84% on test
vs 98.7% on clean chats.

## Phase 2 — CV, trainer, baselines, leak check, submit, score

What it does:
- `cv.py` — stratified folds (label, + meta.source). `liars folds` writes
  `data/canonical/folds_<train stem>.csv`; every run reads it.
- `trainer.py` (`liars train`) — `train.mode: kfold` (OOF on every row, test preds
  averaged over folds) or `holdout` (fit on folds != `cv.holdout_fold`, predict that
  fold + test; models with `uses_val` get an inner split of the training folds;
  `train.refit_full` retrains on 100% for test preds). Writes `data/oof/<run>.parquet`
  (id, fold, label, p_A..p_D), `data/preds/<run>.parquet`, `data/runs/<run>.yaml`, and a
  row in `data/runs.csv`. `--debug` uses the tiny fixture and writes to `data/debug/`.
- `models/base.py` (interface), `models/baselines.py` (majority, TF-IDF+LR scored per
  speaker, softmax over the 4).
- `score.py` (`liars score`) — accuracy, by lie_type / difficulty / true label. Labels
  from a CSV or a labelled canonical JSONL (so OOF files can be scored too).
- `leakcheck.py` (`liars leakcheck`) — `data/reports/leakcheck_<name>.md`: label and
  position skew, one-feature rules, chi² n-grams of liar-speaker lines, masked templates
  (flags liar-only ones), evidence overlap, TF-IDF OOF accuracy + top features, alerts.
- `submit.py` (`liars submit`) — preds -> CSV per `configs/submission.yaml`
  (letter | int [as_listed | first_appearance] | name), optional `--sample` to match
  columns / ids / order, validated before writing. `decode_label` is the inverse.

How to run (bench):
```bash
liars folds --in data/bench/train.jsonl --k 5
liars train --config configs/baselines.yaml --test data/bench/test.jsonl
liars train --config configs/baselines.yaml --set train.mode=holdout --set train.refit_full=true
liars score --preds data/preds/<run>.parquet --labels data/bench/test_labels.csv
liars score --preds data/oof/<run>.parquet --labels data/bench/train.jsonl
liars submit --preds data/preds/<run>.parquet --out data/subs/<name>.csv [--sample sample_submission.csv]
liars leakcheck --in data/bench/train.jsonl
```

## Phase 3 — Zero-shot NLI

What it does: `models/nli_zeroshot.py` scores how strongly each speaker's lines
contradict the evidence with an off-the-shelf NLI cross-encoder (no training).
Modes: `pairwise` (each fact vs each message), `joint` (all evidence vs each message),
`joint_concat` (all evidence vs all of a speaker's messages). Speaker score = max
contradiction probability, then softmax with a temperature. `fit` picks the best mode
and temperature on the training folds and writes a per-mode / per-lie_type /
per-difficulty table to `data/reports/<run>_fold<k>.md`. Pair scores are cached in
`data/cache/nli_<model hash>.parquet`, so re-runs and other folds are nearly free.
The contradiction index is read from the checkpoint's `id2label`.

How to run:
```bash
liars train --config configs/nli_zeroshot.yaml --debug          # xsmall NLI, CPU, tiny fixture
liars train --config configs/nli_zeroshot.yaml --test data/bench/test.jsonl   # large, GPU
liars train --config configs/nli_zeroshot.yaml --set nli.model_id=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

## Phase 4 — Kaggle packaging + first GPU run

What it does: run any `liars` command on a Kaggle T4 from the terminal and merge the
outputs back into `data/`. Full guide: [docs/KAGGLE.md](docs/KAGGLE.md).
- `scripts/kaggle_run.py` — `data` (private data dataset + manifest), `code` (private
  code snapshot), `push` (builds a thin notebook + kernel-metadata.json, `kaggle kernels
  push --accelerator NvidiaTeslaT4`), `status`, `wait`, `fetch` (download + merge),
  `merge` (manual zip), `template`.
- `notebooks/kaggle/run_cli.ipynb` — the generic notebook for the manual flow (generated
  by `kaggle_run.py template`).
- `scripts/gpu_info.py` — GPU, memory, versions, 4-bit load check (first cell of every run).
- `scripts/build_wheel.sh` — wheel + source tarball for the code dataset.

How to run:
```bash
python scripts/kaggle_run.py push --name nli-bench --code dataset \
  --cmd "liars train --config configs/nli_zeroshot.yaml --test data/bench/test.jsonl --set nli.progress=false" --wait
python scripts/kaggle_run.py fetch --name nli-bench
```
Safety (2026-10-27): `push` re-uploads the code snapshot by itself when HEAD changed (and waits
until Kaggle serves it), refuses uncommitted changes (`--allow-dirty` overrides), and checks
that every attached dataset is private. `fetch` retries flaky downloads and warns if the run
used a different commit. `gpu_info.py` also saves `data/logs/pip_freeze_kaggle.txt`.

## Phase 5 — Open-LLM zero-shot scoring + model benchmark

What it does:
- `prompt.py` — numbered evidence + "S: text" chat + the question; truncation keeps all
  evidence and drops first the messages furthest from any other message about the same
  entity (small talk goes first). Shared with Phase 6.
- `llm.py` — load an instruct LLM (4-bit NF4 + fp16 compute on CUDA, fp32 on CPU, never
  MPS), render with the model's chat template, find the A/B/C/D token ids by diffing the
  rendered prompt with and without the letter, score batches (left padding, last-position
  logits only), count fp16 NaN rows, track chats/min and GPU memory.
- `augment.py` — `permute_speakers`, `shuffle_evidence`, `tta_variants` (cyclic shifts
  first so every speaker sits in every letter slot), `map_back`.
- `models/llm_zeroshot.py` — softmax over the 4 letter logits, averaged over TTA
  permutations; `fit` picks few-shot examples and a temperature on the training folds;
  report with chats/min, memory, NaN rows, truncation.

How to run:
```bash
liars train --config configs/llm_zeroshot.yaml --debug                     # Qwen2.5-0.5B on CPU
liars train --config configs/llm_zeroshot.yaml --test data/bench/test.jsonl  # Kaggle T4, holdout
liars train --config configs/llm_zeroshot.yaml --set llm.model_id=Qwen/Qwen3-14B --set augment.tta_n=2
```

## Phase 6 — QLoRA open-LLM classifier

What it does: `models/llm_qlora.py` fine-tunes LoRA adapters (r 16, alpha 32, dropout
0.05) on the attention + MLP projections of the language model of a 4-bit NF4 LLM
(vision/audio towers excluded by `qlora.target_regex`). Same prompt as the zero-shot
model; loss = cross-entropy over the 4 letter logits at the answer position. fp16 with a
gradient scaler (T4: no bf16), gradient checkpointing, paged 8-bit AdamW, cosine schedule
with warmup, early stopping on an inner split of the training folds (never the holdout
fold), speaker-permutation augmentation, TTA at inference, temperature fit on the inner
split. Prints the prompt-length distribution and a measured time per epoch after 20 steps
and aborts above `qlora.max_gpu_hours` (default 4) or on persistent fp16 overflow.
Adapters + `model_meta.json` are saved to `data/models/<run>/fold0` (and `full` with
`train.refit_full`). `train.init_from_run` continues from saved adapters (stage 1 -> 2).

How to run:
```bash
liars train --config configs/llm_qlora.yaml --debug                         # 0.5B on CPU, ~35 s
liars train --config configs/llm_qlora.yaml --test data/bench/test.jsonl --set augment.tta_n=2   # Kaggle
liars predict --run <run> --in data/bench/test.jsonl                         # saved adapters
liars train --config configs/llm_qlora.yaml --set data.train=<stage1 file> ...   # stage 1, then
liars train --config configs/llm_qlora.yaml --set train.init_from_run=<stage1 run>
```

Bench result (Kaggle T4, gemma-4-12B-it, 1520 training chats, 2 epochs, 2.6 GPU h): 97.5%
holdout / 96.8% bench test, vs 84.5% / 84.3% zero-shot. ~0.96 h per epoch per 1500 chats.
Fallback `google/gemma-4-E4B-it` (1 epoch, 0.7 GPU h): 91.75% / 93.0%.

Prediction on both T4s and other competition-day options (2026-10-27):
```bash
liars train --config configs/llm_qlora.yaml --set qlora.epochs=1     # no --test: holdout only
liars predict --run latest:llm_qlora --in <test.jsonl> --gpus 2 --test-preds   # 2 processes, merged
liars train ... --set train.subsample=4000      # stratified subset of the training folds
```
LLM TTA default is 1 (TTA 2 gave no measurable gain); with `augment.tta_n=2` the log shows a
"TTA check" line (first variant vs mean of variants) on the holdout. Zero-shot runs are also
reloadable with `liars predict`.

## Phase 7 — DeBERTa focus-marked multiple choice

What it does: `models/deberta_mc.py` (AutoModelForMultipleChoice, initialised from the
Phase 3 NLI DeBERTa). For each chat, 4 choices (one per speaker X): text_a = "Evidence: "
+ facts, text_b = the chat with X's lines wrapped in `<t> ... </t>` (special tokens added,
embeddings resized) + "Is the marked speaker contradicting the evidence?". Softmax across
the 4 choices. Truncation keeps all evidence and every X line; other speakers' lines
furthest from X's lines go first. fp16 + gradient scaler on CUDA, gradient accumulation,
cosine LR with warmup, early stopping on an inner split, optional layer-wise LR decay,
speaker-permutation augmentation, TTA at inference. Two-stage training via `--stage`:
stage 1 = one holdout run on `stage1.jsonl` (5% holdout) that saves the model, stage 2 =
5-fold starting from `latest:deberta_mc_s1`.

How to run:
```bash
liars train --config configs/deberta_mc.yaml --stage 1 --debug    # CPU, xsmall, ~20 s
liars train --config configs/deberta_mc.yaml --stage 2 --debug    # CPU, 3 folds from stage 1, ~40-80 s
liars folds --in data/bench/stage1.jsonl --k 20                    # 5% holdout for stage 1
liars train --config configs/deberta_mc.yaml --test data/bench/test.jsonl   # 5-fold, GPU
```

Bench result (Kaggle T4, DeBERTa-v3-large, 5-fold, 2 epochs, 2.3 GPU h): 93.7% OOF / 95.8%
bench test. Large needs `deberta.gradient_checkpointing: true` on a T4 (OOM otherwise); ~10 min
per epoch per 1500 chats.

## Phase 8 — Ensemble, error analysis, rules

What it does:
- `ensemble.py` (`liars ensemble`) — aligns OOF files on the ids every run has (the
  holdout fold when a holdout run is included), then compares each run, `mean`,
  `weighted` (non-negative weights on OOF log-loss) and `stack` (conditional logit on
  per-speaker p / log p / rank of each run + rule scores, shared weights, softmax over 4).
  Weighted and stack are scored with an inner K-fold, so their accuracy is honest.
  `--method auto` picks the best of mean/weighted, and stack only if it is > 0.3 points
  better. Writes `data/oof/ens_<name>.parquet` and `data/preds/ens_<name>.parquet`.
- `analysis.py` (`liars analyze`) — `data/reports/<run>.html`: accuracy per fold /
  lie_type / chat length / true label, confusion matrix, calibration, the most confident
  mistakes (true liar red, predicted blue), and disagreements with `--compare`.
- `features/rules.py` — `Rule` interface + registry, two generic rules
  (`number_mismatch`, `day_time_mismatch` with "6 PM" == "18:00", "Fri" == "Friday").
  `liars rules` writes `data/features/rules_<stem>.parquet` (id, speaker, rule, score);
  the stacker picks these up automatically. `liars rules-report` shows coverage and
  precision per rule and lie_type.

How to run:
```bash
liars rules --in data/bench/train.jsonl && liars rules --in data/bench/test.jsonl
liars rules-report --in data/bench/train.jsonl
liars ensemble --runs <run1> <run2> <run3> --name bench --method auto
liars analyze --run <run> --compare <run2> --n 50
```

Bench result: QLoRA + DeBERTa mean = 98.0% holdout (honest) / 97.3% bench test vs QLoRA alone
97.5% / 96.8%. `--method auto` also considers the best single run and treats accuracy gaps under
`ensemble.select_tol` (0.3 points) as ties broken by log-loss.

## Competition-day sizing (`liars plan`)

What it does: `src/liars/sizing.py` turns train size, test size and chat length into GPU hours
per model from speeds measured on a Kaggle T4 (`configs/sizing.yaml`) and applies the runbook
rules (BUILD_MAP "Launch GPU jobs": TTA 1, both T4s for prediction, 1 epoch, subsample, cut
max_len, fallback model; DeBERTa and zero-shot only if they fit their caps).

How to run (numbers from `liars stats` -> "total chars" p50 / p90):
```bash
liars plan --train 5000 --test 3000 --chars 650 --chars-p90 860
```
One-page event checklist: [COMPETITION_CHECKLIST.md](COMPETITION_CHECKLIST.md). Write-up skeleton:
[METHODOLOGY.md](METHODOLOGY.md). Pinned versions: `requirements-lock.txt`.

## Phase 9 — Dress rehearsal

## Phase 10 — Pseudo-labelling (only if allowed)

What it does: `pseudo.py` + `liars pseudo` take a prediction file (normally the best
ensemble's test preds), keep the test chats whose top probability is >= `pseudo.threshold`
(0.95), and write them as canonical examples with `label` = argmax and
`meta.source = "pseudo"`. It refuses to run while `pseudo.enabled` is false (the default,
until the organizers confirm pseudo-labelling is allowed). `train.extra_train: [files]`
adds such files to the TRAINING side only (never the inner validation split, never a
held-out fold), so held-out scores stay honest; the fold log shows `(+N extra)`.

How to run (bench, where it is allowed because the data is our own):
```bash
liars pseudo --preds data/preds/<run>.parquet --test data/bench/test.jsonl \
  --out data/bench/pseudo_qlora.jsonl --labels data/bench/test_labels.csv --set pseudo.enabled=true
liars train --config configs/deberta_mc.yaml --test data/bench/test.jsonl \
  --set train.mode=holdout --set 'train.extra_train=[data/bench/pseudo_qlora.jsonl]'
```
`--labels` (bench only) prints the accuracy of the pseudo-labels. Use the preds of a model
that never saw the holdout fold (a holdout-mode run), so the holdout comparison stays clean.
