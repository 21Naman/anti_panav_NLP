# PROGRESS — Liar's Chatroom

> The handoff log. A new Claude Code session reads this file instead of old chats.
> Claude updates it at the end of every phase (see CLAUDE.md → Phase protocol).

## Current state
- Mode: PREP            <!-- PREP | REHEARSAL | COMPETITION -->
- Current phase: 9 — NOT STARTED (needs the user: separate sessions, timed)
- Last completed phase: 10 (pseudo-labelling, switched off) + prep wrap-up (2026-10-27) — done.
  All prep phases 0–8 and 10 are done; Phase 9 (dress rehearsal) is the only one left.
- Next step: the user starts Phase 9 in a fresh session (BUILD_MAP "Phase 9"). GPU quota is
  saved for it (~21 h left this week). Don't start GPU runs before that.
- Autonomy (2026-10-27): the user is away and said to keep going phase after phase
  without waiting for "next", verifying every Done-when check myself. Stop only for
  things that need the user (gated-model licence, a run > 4 GPU hours, anything
  irreversible beyond the git/Kaggle permissions in Key facts). Items that need the
  user's eyes (e.g. Phase 1 sample read-through) are logged as "pending user".
- Blockers / waiting on: organizers' answer on pseudo-labelling; event start date;
  USER: `git push origin main` is blocked by a macOS Keychain prompt (git-credential-osxkeychain
  waits for a click) — run `git push` once yourself. Until then Kaggle runs use `--code dataset`.
  USER (optional): Kaggle Secrets are not attached to API-pushed kernels (see docs/KAGGLE.md §5).
- Standing reminder: at the end of EVERY phase, remind the user to fill in the event
  dates (until "Event dates" below is filled).
- Last updated: 2026-10-27

## Key facts (Claude keeps this current)
- Event dates (48 h window): UNKNOWN — follow the full plan; remind the user at the end of every phase.
- Days left before the event: unknown
- Team members + Kaggle accounts: SOLO. One Kaggle account: `panavrg` → ONE weekly GPU
  quota (~30 h). Wherever BUILD_MAP.md splits jobs across team accounts, run them one
  after another on this account, by priority: (1) QLoRA main run, (2) LLM zero-shot,
  (3) DeBERTa. Point this out whenever sizing runs.
- Python / env: Python 3.11 (`/opt/homebrew/bin/python3.11`) in `.venv`. The Mac's
  default `python3` is 3.14 — don't use it.
- Repo URL (private): https://github.com/Panavdoesstuff/liars-chatroom
  Git: commit locally only before Phase 4. From Phase 4 on, pushing to origin is fine
  (Kaggle installs the code from GitHub).
- Laptop: Mac (Apple Silicon, arm64), CPU only for --debug runs. NEVER use MPS: force
  CPU for all local runs. bitsandbytes does not work on Mac, so local LLM debug runs
  load the tiny model without quantization.
- Kaggle API: `~/.kaggle/kaggle.json` is set up. The `kaggle` CLI is a developer tool:
  install it in `.venv`, do NOT add it to pyproject dependencies. Phase 4: try running
  notebooks from the terminal with `kaggle kernels push` (GPU + internet enabled in
  kernel-metadata.json) and fetch results with `kaggle kernels output`. If Kaggle
  Secrets don't attach to pushed notebooks, fall back to the manual flow in
  docs/KAGGLE.md and tell the user.
- Kaggle Secrets (account panavrg): `GH_TOKEN` (read-only access to this repo) and
  `HF_TOKEN` (Hugging Face read token). Notebooks read both from Kaggle Secrets.
  Never ask the user to paste tokens; never write them into any file.
- Hugging Face: NO gated-model licences accepted. Phase 5: prefer non-gated models. If
  a gated model looks clearly better, tell the user which model page to open and
  accept, and wait.
- Kaggle GPU: T4 x2 per session (Python 3.13, torch 2.11.0+cu128, transformers 5.16.1, peft 0.20.0, accelerate 1.14.0, bitsandbytes 0.50.2, datasets 4.8.5; fp16 only, no bf16; 4-bit NF4 OK). Up to 2 sessions run at once (verified 2026-10-27).
- Kaggle quota: ~30 h/week (Kaggle docs: "30 hours or sometimes higher"). Resets weekly at the Friday/Saturday boundary, midnight UTC (Kaggle staff + community; not in the official docs) -> next reset Sat 2026-10-31 00:00 UTC. Used this week: ~9.0 h -> ~21 h left. CONFIRM on the Kaggle quota bar (it also shows the time to reset; also shows whether a T4 x2 session counts once, as assumed here).
- GPU hours used in prep (account panavrg): ~9.0 h (Phase 4 NLI 0.17; Phase 5 ~1.6; Phase 6 QLoRA 2.58; Phase 7 DeBERTa 2.32 + 2 failed starts 0.1; Phase 10 pseudo 0.68 collapsed + 0.78; E4B fallback benchmark 0.71)
- Models used (Hugging Face id @ revision, all non-gated; revisions = current `main`, last modified before our runs):
  - google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 (Apache-2.0) — LLM zero-shot + QLoRA main
  - google/gemma-4-E4B-it @ ee0ef6023621cff504d758262d4e04895a5af4a2 (Apache-2.0) — QLoRA fallback
  - MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli @ b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7 (MIT) — NLI zero-shot + DeBERTa MC init
  - Benchmark only: Qwen/Qwen3-14B @ 40c06982, microsoft/phi-4 @ 2db69c1c, Qwen/Qwen3.5-9B @ c2022362, Qwen/Qwen3-8B @ b968826d, Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f
  - Debug/tests: Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760, cross-encoder/nli-deberta-v3-xsmall @ a1508764, trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 @ ce8d0bf2, hf-internal-testing/tiny-random-LlamaForCausalLM @ 9fb19125
  - Package versions: requirements-lock.txt (local, installable) + Kaggle runtime in its header; every Kaggle run now writes data/logs/pip_freeze_kaggle.txt.
- Best result so far: ens_bench_qd = mean(QLoRA, DeBERTa) — 98.0% holdout fold 0 (honest) / 97.3% bench test. Best single: llm_qlora_1027-002454 (Kaggle) — holdout — 97.5% fold 0 / 96.8% bench test — gemma-4-12B-it QLoRA r16, 2 epochs, best step 138/190, TTA 2, T=0.25. (Zero-shot gemma: 84.5% / 84.3%; NLI-large: 71.2% OOF / 73.4% test)
- Stage-1 / pretrained checkpoints: none yet (stage 1 deferred for QLoRA and DeBERTa).
- DeBERTa time on T4: DeBERTa-v3-large MC, 1520 chats, ~225 tokens/choice p50 (max_len 512), grad checkpointing, micro 2 x accum 8: 10.4 min/epoch; one fold (2 epochs + evals + TTA-2 predict of 400 + 1000 chats) = 28 min; 5-fold = 2.32 h; 9.8 GiB peak.
- LLM benchmark winner: google/gemma-4-12B-it (Apache-2.0, not gated) — 82.4% zero-shot on 500 bench chats (no TTA) — 110 chats/min on T4 4-bit (66/min with TTA 2), 10.0 GiB. Fallback: Qwen/Qwen3-14B 69.4% (needs prompt.answer_prefix="Answer:")
- QLoRA fallback on T4: google/gemma-4-E4B-it (run llm_qlora_1027-070253), 1 epoch on 1520 chats: 0.41 h/epoch (15.4 s/step), predict 240 chats/min TTA 1, 9.5 GiB; 91.75% holdout / 93.0% test (12B 2 ep: 97.5 / 96.8; DeBERTa fold 0: 92.75). 0.71 GPU h.
- TTA decision (2026-10-27): LLM default tta_n 1 (no measurable gain from 2; see sizing plan). DeBERTa keeps 2.
- QLoRA time per epoch on T4: google/gemma-4-12B-it — 1520 chats (prompt p50 318 / max 489 tokens) — max_len 1024 — 0.96 h/epoch (36.3 s/step, 16 chats/step). Rule of thumb: ~1600 chats/h at these lengths; time grows ~linearly with prompt length. Whole run (2 epochs + 9 evals + calib + TTA-2 predict of 1400 chats) = 2.58 h. Predict = 51 chats/min with TTA 2. 8.6 GiB peak.
- Template packs available: default (neutral sentences), alt (casual texting) — src/liars/synth/templates/
- Benchmark sets: data/bench/{stage1 (15k default mixed, seed 101), train (2k alt med+hard, seed 202, first_names), test (1k alt, seed 303, labels hidden) + test_labels.csv}. Regenerate: README Phase 1.

## Competition-day sizing plan (2026-10-27)
Tool: `liars plan --train N --test M --chars <p50 total chars> --chars-p90 <p90>` (rules R1–R10 in
BUILD_MAP "Launch GPU jobs"; speeds in configs/sizing.yaml). Prompt tokens ~ 75 + 0.38 x total chars
(bench fit). All numbers: Kaggle T4, measured on bench chats (~320 prompt tokens), linear in tokens.
Model check: predicts the real bench QLoRA run at 2.67 h vs 2.58 h measured.

QLoRA TRAIN hours per epoch (holdout mode trains 76% of N), by mean prompt tokens (~ max_len cap):
| model | N train | 320 | 512 | 768 | 1024 |
|---|---|---|---|---|---|
| gemma-4-12B | 2k | 0.96 | 1.53 | 2.30 | 3.07 |
| gemma-4-12B | 5k | 2.40 | 3.83 | 5.75 | 7.67 |
| gemma-4-12B | 10k | 4.79 | 7.67 | 11.5 | 15.3 |
| gemma-4-12B | 20k | 9.58 | 15.3 | 23.0 | 30.7 |
| fallback gemma-4-E4B | 2k / 5k / 10k / 20k | 0.41 / 1.01 / 2.03 / 4.05 | 0.65 / 1.62 / 3.24 / 6.49 | 0.97 / 2.43 / 4.86 / 9.73 | 1.30 / 3.24 / 6.49 / 13.0 |
Add per job: ~0.1 h setup, inner-val evals (4 per epoch over 4% of N), holdout prediction (20% of N).

PREDICTION hours per 1k chats (LLM scoring; DeBERTa is ~10x cheaper):
| model | TTA, GPUs | 320 | 512 | 768 | 1024 |
|---|---|---|---|---|---|
| gemma-4-12B | TTA 1, 1 GPU | 0.16 | 0.26 | 0.39 | 0.52 |
| gemma-4-12B | TTA 1, 2 GPUs (`--gpus 2`) | 0.08 | 0.13 | 0.20 | 0.26 |
| gemma-4-12B | TTA 2, 1 GPU (old default) | 0.33 | 0.52 | 0.78 | 1.05 |
| gemma-4-12B | TTA 2, 2 GPUs | 0.16 | 0.26 | 0.39 | 0.52 |
| gemma-4-E4B | TTA 1, 1 / 2 GPUs | 0.07 / 0.03 | 0.11 / 0.06 | 0.17 / 0.08 | 0.22 / 0.11 |

TTA 2 vs TTA 1 (QLoRA): not measurable without a new GPU run (only the averaged probabilities were
saved; the 12B does not fit a laptop). Evidence: QLoRA is near-certain (388/400 holdout predictions
>= 0.95; only 10 have a top-2 margin < 0.5, so TTA can change at most a handful), no position bias
(per true label .96–.98), and zero-shot gemma on 108 shared holdout chats: TTA 1 86.1% vs TTA 2
85.2%. Gain < 0.5 points -> TTA 1 is the competition default (halves LLM prediction). The trainer
now logs a free "TTA check" (first variant vs mean) whenever tta_n >= 2, to re-check in Phase 9.

Worked example — their train 5k, test 3k, ~650 chars (~322 tokens):
| job | plan | GPU h |
|---|---|---|
| QLoRA 12B | 2 epochs = 5.35 h > 4 h -> 1 epoch: training job (incl. 1000 holdout chats) | 2.81 |
| QLoRA test prediction | 3000 chats, TTA 1, `--gpus 2` (TTA 2 on 1 GPU would be 0.99) | 0.25 |
| LLM zero-shot | calib 100 + 1000 holdout + 3000 test (2 GPUs), TTA 1 | 0.50 |
| DeBERTa-large | holdout mode, 2 epochs, TTA 2 (recommended) | 1.21 |
| (DeBERTa 5-fold instead) | 2 epochs 5.66 h; 1 epoch 3.48 h — not worth it next to QLoRA | — |
| **total (recommended)** | | **~4.9** (+20% margin) |

When a QLoRA job will not fit in ~4 h, in this order (= rules R2–R6):
1. Both T4s for prediction (`liars predict --gpus 2`, done: 2 processes, merged parquet; tested on CPU
   only — first GPU use in Phase 9). Training itself still uses ONE T4 (DDP not built; ~2x if needed).
2. 1 epoch instead of 2 (bench: inner-val 98.75% after 1 epoch vs 100% best).
3. Subsample the training folds (`train.subsample=K`, stratified; K >= 1500).
4. Cut max_len to 512 (only if p50 > 512 tokens; check the truncation count).
5. Smaller model gemma-4-E4B (91.75% holdout on bench at 1 epoch — use only as a last resort).
6. Two parallel sessions (2 jobs at once) save wall-clock, not quota.

## Organizer answers (2026-10-26, from the SPOC)
- Dataset size: still being finalised. -> Plan for both small (<5k) and large train sets.
- Public / private leaderboard split: still being finalised.
- External prediction APIs (Claude, OpenAI, ...): NOT allowed. Running LLMs ourselves: allowed.
- Pseudo-labelling (training on model-generated test predictions): will be clarified before the event.
- Submission: CSV file only. A notebook is not required, but top teams may be asked for notebook / code for review.

## Still open
- [ ] Pseudo-labelling allowed? (organizers will clarify)
- [ ] Exact event start date / time?
- [ ] Approximate train and test sizes? (being finalised)
- [ ] Public / private split? (being finalised)
- [ ] Is self-generated synthetic training data allowed? (assumed yes — "any ML/NLP approach"; confirm)
- [ ] Any sample of the data format before the event?

Answers (date — answer):
-

## Decisions register
| Date | Phase | Decision | Why |
|------|-------|----------|-----|
| 2026-10-26 | plan | Rules are written on competition day against the real generator; prep builds only the rules framework | Template-specific rules beat generic ones |
| 2026-10-26 | plan | Speaker permutation is a config switch, decided after the leak check | Real label positions may be non-uniform |
| 2026-10-27 | plan | No external prediction APIs anywhere in the code | Organizer rule (2026-10-26) |
| 2026-10-27 | plan | Open-LLM QLoRA is the main model; open-LLM zero-shot is a core phase | APIs banned, own LLMs allowed; fine-tuned LLMs usually beat DeBERTa on reasoning-style text tasks |
| 2026-10-27 | plan | Expensive LLM runs use holdout mode (1 fold), not 5-fold; ensemble weights fit on the holdout fold | GPU budget |
| 2026-10-27 | plan | DeBERTa is a "Should" phase; synthetic stage-1 pretraining is a switch | Useful ensemble member, but less likely to be the best single model |
| 2026-10-27 | plan | Dropped: API backends, LLM-rewritten synthetic data, model-download datasets | APIs banned; CSV-only means internet is available on Kaggle |
| 2026-10-27 | plan | Pseudo-labelling built behind a switch, default off | Not yet confirmed by organizers |
| 2026-10-27 | plan | In COMPETITION mode Claude Code never reads individual test chats | Keeps test predictions clearly model-only (fair play) |
| 2026-10-27 | setup | Solo team, one Kaggle account (panavrg): jobs the plan splits across accounts run sequentially, priority QLoRA main → LLM zero-shot → DeBERTa | Only one weekly GPU quota (~30 h) |
| 2026-10-27 | setup | All local runs force CPU (never MPS); local LLM debug runs load the tiny model unquantized | User instruction; bitsandbytes doesn't work on Mac |
| 2026-10-27 | setup | Local env = Python 3.11 venv `.venv` | Matches user's setup; closer to Kaggle's Python than the Mac default 3.14 |
| 2026-10-27 | setup | `kaggle` CLI installed in `.venv` as a dev tool, not a pyproject dependency; Phase 4 tries `kaggle kernels push` / `output`, manual flow as fallback | User instruction; package deps stay on the allowed list |
| 2026-10-27 | setup | Tokens only via Kaggle Secrets (GH_TOKEN, HF_TOKEN); never in files, never asked for in chat | Security |
| 2026-10-27 | setup | Phase 5 prefers non-gated HF models; ask the user before relying on a gated one | No gated licences accepted yet |
| 2026-10-27 | setup | Commit locally only until Phase 4; push to origin from Phase 4 on | Kaggle installs from GitHub starting in Phase 4 |
| 2026-10-27 | 6 | QLoRA gemma-4-12B-it is the main model (97.5% holdout vs 84.5% zero-shot) | +13 points from 1520 training chats |
| 2026-10-27 | 6 | Stage-1 QLoRA experiment deferred until after the Phase 7 DeBERTa run, and only if the weekly quota allows | Solo quota; bench holdout is already ~saturated (97.5%), so stage 1 can hardly show a gain here |
| 2026-10-27 | 7 | DeBERTa real run = large, 5-fold, 2 epochs; stage-1 variant not run | Solo GPU quota (stage 1 on 15k chats ~ +2 h); bench run already gives timing + ensemble partner |
| 2026-10-27 | 7 | DeBERTa loads fp32 master weights and uses gradient checkpointing by default | transformers 5 kept the fp16 checkpoint dtype (GradScaler crash); large OOMed on T4 at micro batch 2 |
| 2026-10-27 | 8 | `ensemble auto` also considers the best single run; accuracies within `select_tol` (0.3 pts) tie and the lower log-loss wins | 1–2 rows of a 400-row holdout are noise; never ship worse than the best single run |
| 2026-10-27 | 10 | DeBERTa collapse guard: val loss ~ ln 4 and acc <= 0.35 after 20% of steps -> restart once (seed+1, lr x 0.67) | DeBERTa-v3-large collapsed to chance on a real run; the guard recovered the rerun |
| 2026-10-27 | prep | LLM TTA default 1 (QLoRA, zero-shot); DeBERTa keeps 2 | No measurable gain from TTA 2 (zero-shot 86.1% vs 85.2% on 108 shared chats; QLoRA only 10/400 near-ties); halves LLM prediction cost |
| 2026-10-27 | prep | Test prediction split over both T4s (`liars predict --gpus 2`), train jobs run without `--test` | One session has 2 T4s; we used one |
| 2026-10-27 | prep | `liars plan` + runbook rules R1–R10 size every GPU job; QLoRA budget 4 h per training job | Solo quota; decisions readable from N, M and chars at 3 AM |
| 2026-10-27 | prep | Generator re-draws ambiguous "it" chats by default; frozen bench sets NOT regenerated | 13.4% of bench chats are ambiguous; all recorded results refer to the frozen sets and GPU runs cannot be repeated |
| 2026-10-27 | prep | Kaggle push re-uploads the code snapshot when HEAD changed, refuses uncommitted changes, checks datasets are private | Removes the manual `kaggle_run.py code` step and stale-code runs |
| 2026-10-27 | prep+ | Added `line_mil` (line-level MIL checker) as an optional 3rd ensemble member, after the 3 main runs | Different error profile from whole-chat models (per-line view, says which line lies); research (MiniCheck, SummaC, DECODE, CDConv) favours fine-grained checking |
| 2026-10-27 | prep+ | Added `llm_qlora_qwen.yaml` (Qwen3-14B, answer_prefix "Answer:", micro batch 1 x 16) as a second LLM family | Ensemble diversity; Qwen bases won Eedi; 14B is ~1 GiB bigger than gemma-12B on a 15 GiB T4 |
| 2026-10-27 | prep+ | Built the improvement loop (`src/liars/loop/`, `liars loop`, 3 agents, `/improve-loop`); promotion = McNemar + >= 1 point, Bonferroni per round | User asked for a plan->act->test->evaluate->improve agent with branching, recombination, guardrails and recovery; rules from the research in docs/KNOWLEDGE.md |
| 2026-10-27 | prep+ | Ensemble gained `vote` and `caruana` methods; `auto` keeps its old candidates unless `auto_candidates` lists them | Recombination by majority vote was asked for; old ensemble results stay reproducible |

## Phase log
Claude appends one entry per phase using this template:

```
### Phase N — <name>
- Status: done | partial | skipped
- Date:
- Built: (bullets: module -> what it does)
- Files added/changed:
- Commands (exact, copy-pasteable):
- Results: (metrics, timings, GPU hours, data sizes — small table if useful)
- Done-when checks: (each item -> pass/fail)
- Decisions & deviations from BUILD_MAP.md:
- Known issues / TODO for later phases:
- Explain-it-to-me: (2–4 plain sentences on the key idea, for my methodology write-up)
- Commit: <short hash>
```

<!-- Phase entries start below this line -->

### Phase 0 — Scaffold, schema, CLI
- Status: done
- Date: 2026-10-27
- Built:
  - `src/liars/schema.py` -> `Message`, `Example` (from_dict/to_dict, strict: unknown keys
    rejected), `load_jsonl`, `save_jsonl`, `validate(ex) -> list[str]`,
    `validate_file(path) -> FileReport(n_examples, errors)`. `SPEAKERS = ("A","B","C","D")`.
  - `src/liars/config.py` -> `load_config(path, overrides)`, `apply_overrides`
    (`a.b=value`, YAML-typed, creates missing keys), `parse_value`, `config_hash` (sha1, 10
    chars, key-order independent), `seed_everything` (python/numpy/torch/transformers if installed).
  - `src/liars/runlog.py` -> `log_run(...)` appends to `data/runs.csv`; columns
    `run, config_path, config_hash, mode, fold_acc (JSON {fold: acc}), mean_acc, wall_time_s,
    gpu, timestamp (UTC), notes`; `detect_gpu()` returns CUDA name or "cpu" (never MPS).
  - `src/liars/stats.py` -> numbers for `liars stats`.
  - `src/liars/cli.py` -> Typer app `liars`: `validate`, `stats` implemented; synth, adapt,
    folds, leakcheck, train, predict, score, rules, rules-report, ensemble, analyze, pseudo,
    submit are placeholders tagged "(todo: Phase N)" that exit with code 2.
  - `tests/fixtures/tiny.jsonl` -> 32 hand-written chats: 8 per label, Latin square so each
    label gets each of the 8 lie types (number_change, time_shift, day_change, location_swap,
    owner_swap, negation, implicit, two_step) exactly once; 6–11 messages; all 4 speakers talk;
    honest "I think" hedges; honest speakers mention the lied-about entity correctly.
    meta = {source: fixture, difficulty, lie_type, lie_msg_idx}.
- Files added: pyproject.toml, .gitignore, README.md, src/liars/{__init__,schema,config,
  runlog,stats,cli}.py, tests/{conftest,test_schema,test_config,test_runlog,test_cli,
  test_stats}.py, tests/fixtures/tiny.jsonl
- Commands:
  `python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"`;
  `liars validate --in tests/fixtures/tiny.jsonl`; `liars stats --in tests/fixtures/tiny.jsonl`;
  `.venv/bin/pytest -q`
- Results: fresh venv install 10 s. Fixture stats: messages/chat 6–11 (mean 7.7),
  facts/chat 3–4, chat chars p50 267 / p90 305 / p99 384. pytest: 48 passed in 1.6 s.
- Done-when checks: fresh-venv install pass; `liars --help` lists all 15 commands pass;
  validate on tiny -> "32 examples, 0 errors" pass; stats sensible pass; pytest (incl. 13
  bad-example rejection cases + 6 malformed-shape cases) pass.
- Decisions & deviations:
  - Speakers must be exactly the letters A–D (spec said "4 unique speakers"); adapters map
    names to letters, so p_A..p_D always line up.
  - Extra checks beyond spec: liar must have >= 1 message; `meta.lie_msg_idx` must point at
    the liar's lines; unknown top-level keys rejected.
  - `.gitignore` entries anchored to the root (`/data/`, `/models/`) so `src/liars/models/`
    is not ignored.
  - Config loader reads `1e-5` as a float (plain PyYAML reads it as a string).
  - Added `src/liars/stats.py` (not in the layout) to keep cli.py thin.
- Known issues / TODO: `[nlp]` extra includes bitsandbytes, which may not install on the Mac;
  install `[nlp]` on Mac without it if needed (Kaggle is Linux, fine there).
- Explain-it-to-me: every part of the pipeline speaks one data format, so the competition's
  raw format only needs one adapter. The validator catches broken data (wrong speakers, a
  liar who never speaks) before any model sees it. The fixture is a tiny, balanced, fully
  known dataset that every test runs on in seconds.
- Commit: (see git log, "phase 0: ...")

### Phase 1 — Synthetic generator + benchmark sets
- Status: done (one Done-when item pending the user: reading the sample files)
- Date: 2026-10-27
- Built:
  - `synth/world.py` -> seeded World of 4–8 entities (event/object/place/person). Cancelled
    events keep only `happened` + `owner`. Weather is per day (events on one day agree).
    Owner/companion/location values come from name pools that are never entities, so
    facts about different entities cannot clash. `alternative_value` makes wrong values.
  - `synth/pack.py` -> load + `check_pack` (renders every template with dummy values) + value
    rendering (days, 12/24 h times, prices, number words) + style (casing, slang, emoji).
  - `synth/programmatic.py` -> chat assembly. Speaker order is drawn before the liar is
    known; every speaker talks >= 2 times. Lies use the SAME templates as honest claims.
    Honest kinds: restate evidence, unverifiable world fact, true "not" statement, true
    implicit, same-entity hard negative, question, small talk, personal, honest
    anchor+"it" pair. Retries avoid duplicate lines and a speaker repeating a fact.
  - `synth/verify.py` -> per chat: schema, evidence = world, only lie lines are false,
    lie spoken by label, false claim's fact is in the evidence, lie type matches the
    change, two_step 2nd line does not name the entity; per set: label balance, dup rate.
  - `adapters/synthetic.py` -> JSONL writer, `--hide-labels` (label=null; lie_type,
    lie_msg_idx, difficulty moved to the CSV), readable samples with "<-- LIE" marks.
  - CLI `liars synth` (prints verify + stats + validate, writes samples, exit 1 on errors).
- Files: src/liars/synth/{__init__,pack,world,programmatic,verify}.py, templates/{default,alt}.yaml,
  src/liars/adapters/{__init__,synthetic}.py, configs/synth.yaml, tests/test_synth.py;
  cli.py (synth), pyproject (package-data for templates).
- Commands: see README Phase 1 (bench commands are exact).
- Results:
  | check | result |
  |---|---|
  | 5000 med default | 1.8 s wall (generation 1.0 s) |
  | verify, 2000 each of easy/med/hard x default/alt | 0 errors, 0 duplicates |
  | uniform labels | exact 25% each (largest-remainder allocation) |
  | skew {A:.4,B:.2,C:.2,D:.2}, n=5000 | 2000/1000/1000/1000 |
  | bench stage1 / train / test | 15000 / 2000 / 1000, validate 0 errors |
  | bench train lengths | total chars p50 641, p99 986; msgs 10–24 |
  | quick leak look (bench train) | liar msgs 4.17 vs 4.11; liar first 25.5%; first-appearance rank flat; liar line 28.7 vs 27.3 chars |
- Done-when checks: speed pass; verify 0 errors all difficulties/packs pass; balance ±2% and
  skew pass; 4 bench files + validate pass; user read samples **PENDING USER**
  (files: data/samples/synth_default_med.txt, synth_alt_hard.txt, synth_alt_med+hard.txt,
  synth_default_mixed.txt); pytest 70 passed (0.6 s) pass.
- Decisions & deviations:
  - Labels allocated exactly (largest remainder, shuffled) instead of sampled: balance is exact.
  - `--difficulty` also accepts comma lists (`med,hard` = equal mix); used for bench train/test.
  - Neg templates omitted for counts ("there weren't 30 people" is ambiguous).
  - Added `synth/pack.py` (not in layout). Object phrase lives in the pack (`vocab.object_phrase`).
  - Hidden test JSONL drops lie_type / lie_msg_idx / difficulty from meta (lie_msg_idx would
    reveal the liar); they are in test_labels.csv for per-type scoring.
- Known issues / TODO: tense mixes ("is"/"was") within a chat; anchor "it" lines can be
  ambiguous when other entities are discussed in between (mitigated: closer turns preferred);
  liar lines are ~1.4 chars longer on average (lie is always a claim) — Phase 2 leak check
  will quantify. `configs/` is not installed with the package (Phase 4: notebooks must cd
  into a repo checkout or pass --config).
- Explain-it-to-me: we first invent a small hidden world (who owns what, when things
  happen) and only then write the chat from it, so we always know which line is false.
  The evidence is a sample of true world facts; the liar states one wrong value for a fact
  that is in the evidence. Hard negatives (honest people mentioning the same thing,
  hedging, saying true "not" statements, or saying true things the evidence doesn't
  cover) stop a model from winning with shortcuts like "whoever mentions the evidence
  entity" or "whoever says not".
- Commit: (see git log, "phase 1: ...")
- Correction 2026-10-27 (during Phase 2): the Phase 2 leak check found the generator
  leaked (TF-IDF 38.2% OOF; anchors, "got my refund", hedged implicit lies, out-of-range
  lie times like "opens at 12"). Fixed in synth code (see Phase 2 entry); bench sets were
  REGENERATED with the same commands/seeds. Old numbers above (liar line 28.7 vs 27.3
  chars etc.) refer to the old generator.

### Phase 2 — CV, trainer, baselines, leak check, submit, score
- Status: done
- Date: 2026-10-27
- Built: `paths.py` (all outputs under paths.data_dir), `preds.py` (PROB_COLS, make/check/
  save/load preds, softmax, argmax), `cv.py`, `models/{base,__init__,baselines}.py`
  (registry: majority, tfidf_lr), `trainer.py`, `score.py`, `leakcheck.py`, `submit.py`;
  CLI `folds`, `train`, `score`, `leakcheck`, `submit` (predict -> Phase 6 placeholder).
  Configs: `configs/baselines.yaml`, `configs/leakcheck.yaml`, `configs/submission.yaml`.
- Generator fix (Phase 1 code) after the leak check fired:
  - Lie values drawn like true values: wrong times stay in the entity type's hour range,
    wrong weather weighted like real weather, `happened` lies state "went ahead" ~80% of
    the time like honest lines (`pick_target`).
  - Every turn first draws an honest kind; the lie replaces one of the liar's CLAIM turns.
  - "Cover" lines (`cover_rate`, 1.0 for med/hard): each honest speaker also says one TRUE
    line in the lie's form (same attribute / "not" / implied / anchor+"it" pair).
  - Hedges never wrap implicit lines (lie or honest).
- Results (bench, all CPU, seconds):
  | run | mode | held-out acc | bench test acc |
  |---|---|---|---|
  | tfidf_lr (old generator) | kfold | 0.3820 | — (leak) |
  | tfidf_lr baselines_1027-035007 | kfold | 0.2465 | 0.2590 |
  | tfidf_lr baselines_1027-035010 | holdout + refit_full | 0.2700 (fold 0) | 0.2690 |
  | majority baselines_1027-035011 | kfold | 0.2500 | — |
  Leak check after fix: TF-IDF OOF 0.2465, 0 liar-only templates, no alerts; best
  one-feature rule "highest evidence overlap" 0.287 (legit: lies are about evidence).
- Done-when checks: folds -> train -> score -> submit on bench in kfold AND holdout pass
  (letter, int/first_appearance and name submissions validated); runs.csv per-fold accs
  pass; leakcheck runs, leak found and FIXED in generator pass; pytest 87 passed (2.0 s) pass.
- Decisions & deviations:
  - OOF parquet also stores `label` (spec: id, fold, p_*) so ensemble/analysis are self-contained.
  - Inner early-stopping split is used in BOTH modes (only for models with uses_val), never
    the held-out fold, so OOF stays honest in kfold too.
  - `--debug` writes to `<data_dir>/debug/` (own runs.csv) so debug runs never mix with real ones.
  - `liars predict` deferred to Phase 6 (sklearn baselines are not saved).
  - Leak-check alert rules: TF-IDF > 0.35; any one-feature rule > chance + 0.05; label or
    liar-position share > 0.30; template used >= 10 times with liar share >= 0.95.
- Known issues / TODO: the TF-IDF baseline ignores evidence (by design: leak detector).
  `configs/` paths are relative to the repo root.
- Explain-it-to-me: we never judge a model on data it trained on: each chat gets a fold,
  and a model only predicts folds it did not see (kfold), or one held-out fold
  (holdout, for expensive models). The leak check trains a model that cannot see the
  evidence at all; if it still beats chance, the data gives the liar away through style,
  not truth. That caught a real flaw in our own generator, which we fixed by making
  honest speakers use every line form liars use.
- Commit: (see git log, "phase 2: ...")

### Phase 3 — Zero-shot NLI
- Status: done
- Date: 2026-10-27
- Built:
  - `models/nli_zeroshot.py` -> `NLIScorer` (HF cross-encoder, batched, length-sorted, fp16 on
    CUDA only), `PairCache` (data/cache/nli_<sha1(model_id)[:10]>.parquet: key=sha1(premise
    \x1f hypothesis) -> contradiction/entailment/neutral), `label_indices` (reads id2label),
    `speaker_scores` (pairwise / joint / joint_concat; max contradiction per speaker; 0 if no
    scoreable line), `NLIZeroShotModel` (fit = choose mode by accuracy + temperature by NLL on
    training folds; `report()` = per-mode x lie_type / difficulty table).
  - `device.py` -> cuda if available else cpu, never mps.
  - trainer: if a model has `report()`, writes `data/reports/<run>_fold<k>.md`.
  - pyproject: `bitsandbytes; sys_platform == 'linux'` (no Mac support).
- Files: src/liars/{device.py, models/nli_zeroshot.py}, configs/nli_zeroshot.yaml, tests/test_nli.py.
- Commands:
  `liars train --config configs/nli_zeroshot.yaml --debug`
  `head -n 200 data/bench/train.jsonl > data/bench/train_sub200.jsonl; liars folds --in data/bench/train_sub200.jsonl --k 5`
  `liars train --config configs/nli_zeroshot.yaml --set data.train=data/bench/train_sub200.jsonl --set data.test=null --set nli.model_id=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli --set nli.device=cpu --set nli.batch_size=32`
- Results:
  | run | data | time | result |
  |---|---|---|---|
  | debug (xsmall NLI, CPU) cold | tiny fixture | 46 s incl. 283 MB download | 90.0% OOF |
  | debug, 2nd run (cache) | tiny fixture | 5.4 s | same |
  | base NLI, CPU, kfold | 200 bench train | 4 min 48 s (~14k pairs) | 63.5% OOF |
  Per-mode on fold-0 training folds (160 chats): joint_concat 0.638 (chosen, T=0.5),
  joint 0.569, pairwise 0.375. two_step: joint_concat 0.727 vs pairwise 0.364;
  time_shift: joint_concat 0.769 vs joint 0.385; negation: joint 0.824 best.
  Full table: data/reports/nli_zeroshot_1027-035542_fold0.md.
- Done-when checks: debug < 2 min on CPU pass (46 s cold); second run faster pass (5.4 s);
  per-mode/per-lie_type table on CPU subset pass; pytest 94 passed pass.
- Decisions & deviations:
  - Debug model `cross-encoder/nli-deberta-v3-xsmall` (Apache-2.0, 3-class). Large model
    `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MIT) is the default; base
    `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (MIT) for the CPU subset. None gated.
  - Default `train.mode: kfold` for NLI (no training + cache = each pair scored once).
  - Auto-mode candidates order [joint, pairwise, joint_concat]; ties go to the first.
- Known issues / TODO: pairwise is weak here (single facts vs unrelated messages give
  spurious contradictions); maybe use (contradiction - entailment) or only pairs sharing an
  entity — not done (don't over-polish). transformers 5.x warns about torch.jit (harmless).
- Explain-it-to-me: an NLI model reads a premise and a hypothesis and says whether the
  premise makes the hypothesis true (entailment), false (contradiction), or neither
  (neutral). We use "contradiction" as a lie score. Unverifiable chat lines should come out
  neutral, not contradiction. Pairwise mode checks one fact against one message at a time,
  so a two-step lie ("I went to the bbq" + later "it was on Monday") is invisible: the "it"
  message alone has no subject. Joining a speaker's messages (joint_concat) lets the
  model connect them, which is why it scored 0.73 vs 0.36 on two_step lies.
- Commit: (see git log, "phase 3: ...")

### Phase 4 — Kaggle packaging + first GPU run
- Status: done
- Date: 2026-10-27
- Built:
  - `scripts/kaggle_run.py` -> data (private dataset + manifest.json of data/ paths), code
    (private code snapshot: wheel + git-archive tarball + COMMIT), push (thin notebook +
    kernel-metadata.json, `kaggle kernels push --accelerator NvidiaTeslaT4`, private,
    internet on, `--code auto|git|dataset`), status, wait, fetch (download + merge), merge,
    template. Merge never overwrites files; runs.csv rows appended, dedup by run name.
  - Notebook (generated by `build_notebook`, template at notebooks/kaggle/run_cli.ipynb):
    settings -> install (GH_TOKEN clone into /tmp, token stripped from remote and never
    printed; fallback code dataset) -> copy data per manifest -> gpu_info -> run commands
    (timing to data/logs/kaggle_timing.json) -> zip outputs to /kaggle/working/outputs.zip.
  - `scripts/gpu_info.py`, `scripts/build_wheel.sh`, `docs/KAGGLE.md`, tests/test_kaggle_run.py.
  - Kaggle (account panavrg, all private): datasets `liars-bench` (bench train/test +
    folds_train.csv; NO label file), `liars-code` (commit f789370); kernel `liars-nli-bench`.
- Commands:
  `python scripts/kaggle_run.py data data/bench/train.jsonl data/bench/test.jsonl data/canonical/folds_train.csv`
  `python scripts/kaggle_run.py code`
  `python scripts/kaggle_run.py push --name nli-bench --code dataset --cmd "liars train --config configs/nli_zeroshot.yaml --test data/bench/test.jsonl --set nli.progress=false"`
  `python scripts/kaggle_run.py wait --name nli-bench; python scripts/kaggle_run.py fetch --name nli-bench`
- Results (run nli_zeroshot_1026-223959, DeBERTa-v3-large NLI, T4, fp16, kfold, auto mode):
  | metric | value |
  |---|---|
  | GPU time | 551 s command, ~10 min kernel (0.17 GPU h) |
  | mode accuracy (fold-0 fit) | joint 0.711 (chosen, T=0.5), joint_concat 0.681, pairwise 0.515 |
  | OOF bench train (2000) | 0.7120 |
  | bench test (1000) | 0.7340 (med 0.781, hard 0.691) |
  | test by lie_type | number 0.881, negation 0.881, time 0.847, owner 0.760, location 0.735, day 0.624, implicit 0.595, two_step 0.519 |
- Done-when checks: NLI run on Kaggle + outputs merged + `liars score` pass; GPU time
  recorded pass; docs/KAGGLE.md complete (terminal flow, manual flow, troubleshooting,
  findings) pass. pytest 97 passed.
- Decisions & deviations:
  - Code install = git clone of a pinned commit (needs configs/ in the repo, not only the
    package) with a code-dataset fallback, instead of `pip install git+https://TOKEN@...`.
  - Data reaches Kaggle as flat files + manifest.json (robust to Kaggle's archive handling);
    folds file uploaded so Kaggle and laptop runs share identical folds.
  - `paths.data_dir` is the single output root (inputs are explicit file paths), so no
    separate `paths.out_dir`.
  - Kaggle Secrets are not attached to API-pushed kernels -> terminal flow uses
    `--code dataset` (user said: fall back and tell them).
  - git push to GitHub blocked by a macOS Keychain prompt with the user away; the stuck
    push was killed (nothing sent). Local commits only until the user runs `git push`.
- Known issues / TODO: before every Kaggle run with new code: commit, then
  `kaggle_run.py code`. `data/logs/kaggle_timing.json` is overwritten on each fetch (the
  copy under data/kaggle/<name>/ remains). NLI is weakest on two_step (0.52) and implicit
  (0.60) — the LLM phases should target those.
- Explain-it-to-me: Kaggle gives free GPUs but only through notebooks. We keep the
  notebook dumb — it installs our code, copies our data, runs the same `liars` commands we
  run on the laptop, and zips the results — so nothing about a run lives in the notebook.
  A script pushes it from the terminal and pulls the results back into data/.
- Commit: (see git log, "phase 4: ...")

### Phase 5 — Open-LLM zero-shot scoring + model benchmark
- Status: done
- Date: 2026-10-27
- Built:
  - `prompt.py` -> `build_user_text` (intro, numbered evidence, "S: text" chat, question; all
    strings in config `prompt:`), `drop_order` / `fit_to_budget` (keep all evidence; drop
    lines sharing no evidence word with another line first, then the most isolated).
  - `llm.py` -> `load_lm` (4-bit NF4 + fp16 compute + double quant on CUDA; fp32 CPU; falls
    back to AutoModelForImageTextToText for multimodal checkpoints), `render` (chat
    template, `chat_template_kwargs`, list-content fallback), `letter_token_ids` (diff of
    tokenised prompt with/without the letter; " A" after text, "A" after whitespace;
    cached per answer_prefix), `prompt_for` (truncation to llm.max_len), `score_prompts`
    (length-sorted left-padded batches, logits_to_keep=1, NaN rows -> uniform, memo by
    prompt hash, chats/min + max GPU memory), `get_lm` (one model per process).
  - `augment.py` -> permute_speakers, shuffle_evidence, tta_variants (identity + cyclic
    shifts first, then random), map_back.
  - `models/llm_zeroshot.py` -> TTA-averaged softmax of letter logits; fit = few-shot pick +
    temperature (calib_n training-fold examples); report() with speed/memory/NaN/truncation.
  - `config.parse_value`: `--set x=Answer:` stays a string (YAML made it a mapping; cost one
    failed Kaggle run).
- Files: src/liars/{prompt,llm,augment}.py, models/llm_zeroshot.py, configs/llm_zeroshot.yaml,
  tests/{test_augment,test_prompt,test_llm}.py (TTA mapping test = test_tta_maps_permuted_predictions_back).
- Commands: benchmark (per model, Kaggle):
  `liars train --config configs/llm_zeroshot.yaml --set data.train=data/bench/train_sub500.jsonl --set data.test=null --set train.mode=kfold --set llm.calib_n=0 --set augment.tta_n=1 --set llm.model_id=<id>`
  winner: `liars train --config configs/llm_zeroshot.yaml --test data/bench/test.jsonl --set llm.model_id=google/gemma-4-12B-it --set augment.tta_n=2 --set llm.calib_n=100`
- Benchmark (500 bench-train chats = data/bench/train_sub500.jsonl, zero-shot, 4-bit NF4,
  fp16, T4, no TTA, no calibration; all non-gated; Llama-3.1-8B skipped = gated):
  | model | acc | chats/min | GPU GiB | fp16 NaN | day | impl | loc | neg | num | owner | time | 2step |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | google/gemma-4-12B-it | **0.824** | 110 | 9.9 | 0 | .90 | .55 | .89 | .95 | .98 | .91 | .90 | .55 |
  | Qwen/Qwen3-14B + "Answer:" | 0.694 | 104 | 10.8 | 0 | .84 | .33 | .67 | .80 | .88 | .79 | .77 | .55 |
  | microsoft/phi-4 | 0.676 | 115 | 10.4 | 0 | .72 | .39 | .71 | .73 | .83 | .76 | .81 | .52 |
  | Qwen/Qwen3.5-9B | 0.664 | 140 | 8.9 | 0 | .75 | .33 | .73 | .73 | .76 | .77 | .73 | .55 |
  | Qwen/Qwen3-8B + "Answer:" | 0.496 | 183 | 7.0 | 0 | .57 | .28 | .48 | .55 | .66 | .57 | .56 | .34 |
  | Qwen/Qwen3-14B (no prefix) | 0.432 | 110 | 10.8 | 0 | predicts A 76% of the time |
  | Qwen/Qwen3-8B (no prefix) | 0.308 | 201 | 6.9 | 0 | predicts D/A only |
  | Qwen/Qwen3-4B-Instruct-2507 (no prefix) | 0.242 | 333 | 3.7 | 0 | predicts C 99% |
  Winner run llm_zeroshot_1026-232702 (gemma-4-12B-it, holdout, TTA 2, calib 100 -> T=2.0):
  **holdout fold 0: 0.845 (400); bench test: 0.843 (1000)**; test by type: number .97, time
  .94, negation .93, owner .90, location .90, day .86, implicit .62, two_step .62; hard .78,
  med .91; per true label .82–.86 (no position bias); 66 chats/min, 10.0 GiB, 0 NaN; 1577 s.
  NLI-large for comparison: 73.4% test.
- Done-when checks: debug < 2 min on CPU pass (1:47 cold incl. 1 GB download, 17 s warm);
  TTA mapping test pass; benchmark table + winner in Key facts pass; winner holdout + test
  logged pass; pytest 110 passed pass.
- Decisions & deviations:
  - Candidates checked on the Hub 2026-10-27 (licence + gated flag). Newest dense models are
    multimodal checkpoints (gemma-4, Qwen3.5, Ministral-3); loaded via
    AutoModelForImageTextToText fallback, text-only prompts.
  - Benchmark without TTA/calibration (raw, same for all); winner run with TTA 2 (little
    position bias) to save GPU.
  - Qwen3 collapse diagnosed: next-token top-5 = "The", "Based", "Speaker" (prose, not a
    letter), also in fp32 on CPU -> prompt-format artifact, fixed by answer_prefix
    "Answer:". Gemma answers with a letter directly, prefix not needed.
- Known issues / TODO: implicit and two_step lies are the weak spots for every model
  (~0.55–0.62). Gemma-4 is a multimodal wrapper: Phase 6 must put LoRA on the language model
  only and watch fp16 overflow in training (no bf16 on T4). Fallback base: Qwen3-14B.
- Explain-it-to-me: we never let the LLM write an answer. We show it the evidence and the
  chat, stop right where the answer would start, and read how likely it thinks each of
  "A", "B", "C", "D" is as the next token — a 4-way probability with no parsing. Models
  have a bias toward some letters, so we also ask with the speakers renamed (TTA) and
  average after mapping the letters back. A model that "wants" to start with "The answer
  is…" gives meaningless letter scores, which is why Qwen3 needed an "Answer:" prefix.
- Commit: (see git log, "phase 5: ...")

### Phase 6 — QLoRA open-LLM classifier (main model)
- Status: done (stage-1 experiment deferred, see decisions)
- Date: 2026-10-27
- Built: `models/llm_qlora.py` -> 4-bit NF4 base + LoRA (r16/a32/d0.05) on q/k/v/o/gate/up/down
  of the language model only (`qlora.target_regex` skips vision/audio towers); loss = CE over
  the 4 letter logits at the answer position; fp16 autocast + GradScaler, gradient
  checkpointing, paged 8-bit AdamW, cosine + warmup, eval every 25% with early stopping on
  an inner split (never the holdout fold), best LoRA state kept, partial accumulation
  dropped at epoch end, temperature fit on the inner split with TTA; time probe after 20
  steps aborts above `max_gpu_hours` (4) or persistent fp16 overflow; saves adapters +
  model_meta.json; `refit_info`/`refit_full`; `init_from_run` (incl. `latest:<prefix>`).
  `liars predict --run/--model-dir` scores new files from saved adapters.
- Files: src/liars/models/llm_qlora.py, configs/llm_qlora.yaml, tests/test_qlora.py,
  src/liars/{cli,trainer,llm}.py, scripts/kaggle_run.py (data/models in outputs).
- Commands: `liars train --config configs/llm_qlora.yaml --debug` (CPU, 38 s);
  Kaggle: `python scripts/kaggle_run.py push --name qlora-gemma4-12b --code dataset --wait
  --cmd "liars train --config configs/llm_qlora.yaml --test data/bench/test.jsonl --set augment.tta_n=2"`
  then `python scripts/kaggle_run.py fetch --name qlora-gemma4-12b`.
- Results (run llm_qlora_1027-002454, holdout fold 0 = 400 chats, train 1520 / inner val 80):
  | model | holdout | test | day | impl | loc | neg | num | owner | time | 2step | hard | med |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | gemma-4-12B zero-shot (P5) | 0.845 | 0.843 | .86 | .62 | .90 | .93 | .97 | .90 | .94 | .62 | .78 | .91 |
  | gemma-4-12B QLoRA | **0.975** | **0.968** | 1.00 | .96 | .99 | .99 | 1.00 | 1.00 | .96 | .86 | .96 | .98 |
  (per-type columns = bench test; holdout two_step .88, implicit .98.) Per true label
  .96–.97 (no position bias). Val curve: .925 (step 23) -> 1.00 (step 138, best) -> .9875;
  val loss .31 -> .006. 9 skipped steps, 8 of them in the first 23 (GradScaler settling
  from 65536), 0 NaN logit rows, 0 truncated prompts. Timing: 0.96 h/epoch, train 122 min,
  whole kernel 9284 s = 2.58 GPU h; peak 8.6 GiB.
- Done-when checks: debug < 2 min on CPU pass (38 s); real run 1 + table vs zero-shot pass;
  "QLoRA time per epoch on T4" filled pass; pytest pass (126 passed, 26 s).
- Decisions & deviations:
  - Solo quota: this was job #1 in the priority order (QLoRA -> LLM zero-shot -> DeBERTa).
  - Run 2 (stage 1 on 5k stage1.jsonl -> stage 2) not run yet: it costs ~3.2 h + 2.6 h and
    the bench holdout is near ceiling (97.5%), so it can barely show a gain. Revisit after
    Phase 7 if quota allows (the more useful variant: stage 1 only, scored on bench test
    = does training on our default pack transfer to the alt pack?).
  - TTA 2 on Kaggle (config default 4) to save GPU; temperature came out 0.25 (sharpening).
- Known issues / TODO:
  - two_step lies (anchor + later anaphor) remain the weakest type (.86 test).
  - GradScaler starts at 65536 and wastes ~8 optimizer steps (plus their LR schedule
    steps) while it settles; harmless here, worth a `init_scale` knob if runs get short.
  - Notebook log prints every line twice (cosmetic).
  - Val split (80 chats) saturates at 100%; with real data use a bigger inner split
    (`train.inner_val_frac`, 0.05 now) if early stopping looks noisy.
- Explain-it-to-me: LoRA freezes the 12B base model and trains small low-rank "patch"
  matrices (~1% of the size) added to the attention and MLP weights, so it fits on a T4
  in 4-bit. We still read only the logits of "A"/"B"/"C"/"D" at the answer position, and the
  loss is plain 4-way cross-entropy on those — the LLM becomes a 4-class classifier that
  reuses its reading skills. The holdout fold is never used for training, early stopping
  or temperature, so its accuracy is an honest estimate of unseen data; the ensemble
  later fits weights on it, which only works if nothing else has touched it.
- Commit: (see git log, "phase 6: ...")

### Phase 7 — DeBERTa focus-marked multiple choice
- Status: done (stage-1 variant not run, see decisions)
- Date: 2026-10-27
- Built: `models/deberta_mc.py` -> AutoModelForMultipleChoice from the Phase 3 NLI DeBERTa;
  4 choices per chat (text_a = "Evidence: " + facts, text_b = chat with the candidate's lines
  in `<t> ... </t>` + question); `keep_for_choice` truncation never drops the marked
  speaker's lines; fp32 master weights + fp16 autocast + GradScaler, gradient checkpointing,
  cosine + warmup, optional layer-wise LR decay, early stopping on an inner split, TTA,
  temperature fit; time probe aborts above `deberta.max_gpu_hours` (4) counting ALL folds
  left (trainer passes `train._fits_left`); report has peak GPU memory. `--stage 1|2` via
  `stages:` in the config; `init_from_run: latest:<prefix>`.
- Files: src/liars/models/deberta_mc.py, configs/deberta_mc.yaml, tests/test_deberta.py
  (marked-lines tests, latest: resolution, time guard, fp16-checkpoint-loads-as-fp32),
  src/liars/trainer.py, src/liars/cli.py.
- Commands: `liars train --config configs/deberta_mc.yaml --stage 1 --debug` then `--stage 2 --debug`;
  Kaggle: `python scripts/kaggle_run.py push --name deberta-large-5f --code dataset --cmd
  "liars train --config configs/deberta_mc.yaml --test data/bench/test.jsonl --set deberta.epochs=2"`.
- Results (run deberta_mc_1027-032050, DeBERTa-v3-large, 5-fold, 2 epochs, TTA 2):
  | model | OOF (2000) | fold 0 | test | impl | 2step | hard | med |
  |---|---|---|---|---|---|---|---|
  | NLI-large zero-shot (P3) | 0.712 | 0.715 | 0.734 | | | | |
  | gemma-4-12B zero-shot (P5) | — | 0.845 | 0.843 | .62 | .62 | .78 | .91 |
  | DeBERTa-large MC (P7) | **0.937** | 0.928 | **0.958** | .97 | .75 | .94 | .97 |
  | gemma-4-12B QLoRA (P6) | — | **0.975** | **0.968** | .96 | .86 | .96 | .98 |
  Folds .928/.938/.953/.948/.920. Per true label .93–.95 (no position bias). Val acc still
  rising at the last eval (best step = last step in 4 of 5 folds) -> 3 epochs likely helps.
  0 chats truncated (p99 343 tokens/choice). 2.32 GPU h, 9.8 GiB peak.
- Done-when checks: debug stage 1 (19 s) then stage 2 (78 s) on CPU pass; marked-lines test
  pass (test_keep_for_choice_never_drops_marked_lines, test_encoded_choices_keep_all_marked_lines);
  bench results in the table above pass; pytest pass (129 passed, 35 s).
- Decisions & deviations: large not base (timing known now for competition day); 2 epochs
  to fit the solo quota; stage-1 variant skipped (GPU). Two failed starts fixed: (1) "Attempting
  to unscale FP16 gradients" — transformers 5 loads the fp16-stored NLI checkpoint in fp16,
  now `dtype=torch.float32`; (2) CUDA OOM at 14.5 GiB — gradient checkpointing switch, on.
- Known issues / TODO: two_step is DeBERTa's weak spot (.75); it disagrees with QLoRA on 31/400
  holdout chats (QLoRA right 24, DeBERTa 5) — useful but small ensemble gain (Phase 8).
- Explain-it-to-me: each chat becomes 4 versions, one per speaker, with that speaker's lines
  wrapped in marker tokens; the model gives each version one score and a softmax across the
  4 picks the liar. Scoring speakers independently would let all four look "fine" or "bad"
  at once; the shared softmax forces a comparison ("who is MOST contradicting?"), which
  matches the task's exactly-one-liar structure and keeps the context of the other speakers.
- Commit: (see git log, "phase 7: ...")

### Phase 8 — Ensemble, error-analysis report, rules framework
- Status: done
- Date: 2026-10-27
- Built: `ensemble.py` (OOF aligned on common ids; mean / weighted (SLSQP on log-loss) /
  stack (conditional logit on p, log p, rank + rule scores) / single; weighted + stack scored
  by an inner 5-fold, so honest; auto = best of weighted/mean/single with `select_tol` ties
  -> lower log-loss, stack only if > `stack_margin` better). `analysis.py` (`liars analyze`:
  plain HTML, per fold / lie_type / length / true label, confusion, calibration, 50 most
  confident mistakes red/blue, `--compare` disagreements). `features/rules.py` (Rule +
  registry, `number_mismatch`, `day_time_mismatch`; `liars rules`, `liars rules-report`).
- Files: src/liars/{ensemble,analysis}.py, src/liars/features/rules.py, configs/ensemble.yaml,
  src/liars/cli.py, tests/{test_ensemble,test_rules_analysis}.py.
- Commands: `liars rules --in data/bench/train.jsonl && liars rules --in data/bench/test.jsonl`;
  `liars rules-report --in data/bench/train.jsonl`;
  `liars ensemble --runs baselines_1027-035007 nli_zeroshot_1026-223959 llm_zeroshot_1026-232702 llm_qlora_1027-002454 deberta_mc_1027-032050 --name bench_v3`;
  `liars ensemble --runs llm_qlora_1027-002454 deberta_mc_1027-032050 --name bench_qd`;
  `liars analyze --run ens_bench_qd --compare llm_qlora_1027-002454 --n 50 --data data/bench/train.jsonl`.
- Final bench table (holdout = fold 0, 400 chats, honest; test = 1000 bench test):
  | run | holdout | log-loss | test |
  |---|---|---|---|
  | tfidf (leak detector) | .270 | 1.43 | — |
  | NLI-large zero-shot | .715 | 1.01 | .734 |
  | gemma-4-12B zero-shot | .845 | .51 | .843 |
  | DeBERTa-large MC 5-fold | .928 | .40 | .958 |
  | gemma-4-12B QLoRA | .975 | .116 | .968 |
  | ens_bench_v3 (all 5, auto -> weighted; QLoRA .944, DeBERTa .050, NLI .006) | .975 | .084 | .971 |
  | (v3 stack, not picked: +0.25 pts < 0.3 margin) | .9775 | .081 | — |
  | **ens_bench_qd (QLoRA + DeBERTa, auto -> mean)** | **.980** | .122 | **.973** |
  Rules on bench train: day_time_mismatch coverage .354 / precision .542 (day_change .97/.72);
  number_mismatch .298 / .446 (number_change .92/.58) — generic rules are weak features here.
- Done-when checks: ensemble honest acc >= best single pass (v3 .975 = .975; qd .980 > .975),
  weight table shown; HTML reports readable pass: data/reports/ens_bench_qd.html,
  data/reports/llm_qlora_1027-002454.html, data/reports/deberta_mc_1027-032050.html;
  rules-report on bench train pass (data/reports/rules_train.md); pytest pass (129).
- Decisions & deviations: added the `single` candidate + `select_tol` (decisions register).
  The run subset for bench_qd was chosen after seeing v3 (mild selection; test agrees).
- Known issues / TODO:
  - Most remaining QLoRA errors are two_step chats where an honest anaphor ("it was 12 bucks",
    "belongs to harry") follows ANOTHER speaker's line about a different entity, so the natural
    reading is a contradiction (3 of 4 two_step mistakes read). In bench train ~86% of
    anchor->anaphor gaps contain another speaker's entity-bearing line (upper bound on
    ambiguity). Generator fix (not done; bench frozen): a switch that places each pair so the
    gap holds only small talk / same-entity lines, then rerun the leak check.
  - Competition day: check whether errors are label ambiguity before chasing them.
- Explain-it-to-me: every model is scored on the same untouched holdout chats; the ensemble
  learns how much to trust each model from those chats, and we measure it with an inner
  cross-validation so the score is not fitted to itself. A weaker model (DeBERTa) still helps
  when it makes different mistakes: averaging fixed 4 of QLoRA's errors and broke 2.
- Commit: (see git log, "phase 8: ...")

### Phase 10 — Pseudo-labelling (built, switched OFF)
- Status: done (code + bench measurement). Stays OFF until the organizers confirm in writing.
- Date: 2026-10-27
- Built: `pseudo.py` + `liars pseudo` (top probability >= `pseudo.threshold` 0.95 -> canonical
  examples, label = argmax, meta.source = "pseudo"; REFUSES unless `pseudo.enabled: true`;
  `--labels` reports pseudo-label accuracy on bench). `train.extra_train: [files]` adds them to
  the TRAINING side only (never inner val, never a held-out fold); fold log shows "(+N extra)".
  DeBERTa collapse guard (found by this phase, see below).
- Files: src/liars/pseudo.py, configs/pseudo.yaml, src/liars/{trainer,cli}.py, tests/test_pseudo.py.
- Commands: `liars pseudo --preds data/preds/llm_qlora_1027-002454.parquet --test data/bench/test.jsonl
  --out data/bench/pseudo_qlora.jsonl --labels data/bench/test_labels.csv --set pseudo.enabled=true`;
  Kaggle `deberta-pseudo`: `liars train --config configs/deberta_mc.yaml --test data/bench/test.jsonl
  --set deberta.epochs=2 --set train.mode=holdout --set 'train.extra_train=[data/bench/pseudo_qlora.jsonl]'`.
- Results: 970/1000 bench test chats have QLoRA confidence >= 0.95; pseudo-label accuracy 98.0%.
  | DeBERTa-large, holdout fold 0 (400) | acc | implicit | two_step |
  |---|---|---|---|
  | no pseudo (fold 0 of deberta_mc_1027-032050) | 0.9275 | .776 | .765 |
  | + 970 pseudo (deberta_mc_1027-065229) | **0.9650** | 1.000 | .824 |
  20 errors fixed, 5 new. Bench test 0.971 is circular (trained on those chats). 0.78 GPU h.
- Done-when checks: holdout change recorded (+3.75 points) pass; pseudo-label accuracy reported pass.
- Decisions & deviations: measured with DeBERTa, not QLoRA (QLoRA holdout ~saturated at 97.5%,
  ~3.5 GPU h). Labels from QLoRA's holdout-mode run (never saw fold 0) so the holdout stays clean.
  Confounds: labels from a stronger model (partly distillation); the run used the collapse
  restart (training seed 43, lr x 0.67) while the baseline used seed 42.
- Known issues: first attempt (deberta_mc_1027-054303) COLLAPSED (val loss = ln 4, 24% holdout);
  fixed with the collapse guard (`deberta.collapse_*`: restart once with a new seed and lower lr),
  which fired and recovered in the second attempt. 0.68 GPU h lost.
- Explain-it-to-me: when a strong model is very sure about a test chat, we can treat its
  answer as a label and train on that chat too. The holdout chats are never touched, so the
  before/after holdout accuracy is honest. It only helps if the confident labels are right
  (98% here) and it is only allowed if the organizers say so.
- Commit: (see git log, "phase 10: ...")

### Prep wrap-up — sizing, both T4s, ambiguity fix, competition readiness (2026-10-27)
- Status: done (user's 7-point list of 2026-10-27; no Phase 9 work)
- Built:
  - `liars plan` (src/liars/sizing.py, configs/sizing.yaml): GPU hours per model from N, M and
    chars; applies rules R1–R10 (BUILD_MAP runbook "Launch GPU jobs"). `train.subsample`
    (stratified, training folds only). `liars predict --gpus N` (src/liars/multigpu.py: shards,
    one process per GPU via CUDA_VISIBLE_DEVICES, merge + id check), `--test-preds`,
    `--run latest:<prefix>`; zero-shot runs now save model_meta.json so they can be reloaded.
  - TTA: LLM default 1; trainer logs a "TTA check" (first variant vs mean) when tta_n >= 2.
  - Generator: `synth/ambiguity.py` + verify count; `synth.avoid_ambiguous_anaphors: true`
    re-draws ambiguous "it" chats (frozen bench needs `=false` to regenerate identically).
  - Kaggle runner: push re-uploads the code snapshot when HEAD changed and waits for it,
    refuses uncommitted changes (`--allow-dirty` to override), checks every attached dataset is
    private; fetch retries 3x (flaky DNS) and warns if the run used another commit;
    gpu_info.py writes data/logs/pip_freeze_kaggle.txt.
  - requirements-lock.txt, METHODOLOGY.md (skeleton, TODO (event) markers),
    COMPETITION_CHECKLIST.md (one page), docs/PREP_REPORT.md (readable summary).
- Results:
  - E4B fallback benchmark (0.71 GPU h): 91.75% holdout / 93.0% test, 2.4x faster than 12B.
  - Ambiguity: frozen bench 13.4% ambiguous (train 268/2000, test 132/1000). QLoRA test: clean
    98.7% vs ambiguous 84.1% (~2/3 of its test errors); QLoRA+DeBERTa mean: clean 99.2%. New
    generation: 0 ambiguous, same lie-type mix, leak check TF-IDF 25.45%, no alerts.
  - `predict --gpus 2` on CPU (debug QLoRA + zero-shot): identical to 1 process (max |diff| 4.5e-6).
  - Both datasets confirmed private (`isPrivate: true`); kernels pushed with is_private true.
- Done-when checks (user list): 1 PROGRESS + report filled pass; 2 two-T4 prediction tested in
  --debug on CPU pass (no GPU run); 3 runbook rules pass; 4 ambiguity flag + avoid pass (bench not
  regenerated, see decisions); 5 lock + revisions + METHODOLOGY + checklist + auto code refresh +
  private datasets pass; 6 no new GPU runs pass; 7 pytest pass (see commit).
- Decisions: see Decisions register rows dated 2026-10-27 "prep".
- Known issues / TODO for Phase 9: first real use of `predict --gpus 2`, `liars plan`, the auto code
  refresh and the TTA check; copy pip_freeze_kaggle.txt into requirements-lock.txt; training still
  uses one T4 per job (DDP not built).
- Commit: (see git log, "prep: ...")

### Prep extra — line-level MIL model + Qwen QLoRA config (2026-10-27)
- Status: built, NOT yet run (no torch in the session that wrote it; no GPU run)
- Built:
  - `src/liars/models/line_mil.py` (`model.name: line_mil`): scores every chat line (+3 lines
    before, 1 after, scored line in <t>...</t>) against the evidence with an NLI DeBERTa
    cross-encoder; score = contradiction log-odds (starts from the NLI head, `head: contradiction`).
    Per speaker: soft max (tau*logsumexp) over their lines; softmax over A-D; CE on the liar label
    (MIL: no line labels needed). Optional line BCE on chats with meta.lie_msg_idx
    (`line_loss_weight`, stage 1 on synthetic = 0.5). Early stopping, temperature, time probe /
    max_gpu_hours, save/load (so `liars predict --gpus 2` works), `line_scores()` (which line is the
    lie) and `line_hit_rate()` in the fold report.
  - `configs/line_mil.yaml` (holdout default, stages 1/2, debug block with xsmall on CPU).
  - `configs/llm_qlora_qwen.yaml`: llm_qlora with Qwen/Qwen3-14B, answer_prefix "Answer:",
    micro batch 1 x accum 16 (same effective batch 16).
  - Tests: `tests/test_line_mil.py` (6 pure tests passed in the writing session; 4 torch tests need
    the cached xsmall model), `tests/test_qwen_config.py` (passed).
- Commands:
  - `pytest -q tests/test_line_mil.py tests/test_qwen_config.py` then `pytest -q`
  - `liars train --config configs/line_mil.yaml --debug` (CPU, < 2 min expected)
  - Kaggle: `python scripts/kaggle_run.py push --name line-mil --datasets liars-comp --cmd "liars train --config configs/line_mil.yaml --test <test>"`
- Cost estimate (from DeBERTa MC's measured 6.84 min/1k chats at 4x225 tokens): ~3x DeBERTa MC,
  ~20 min per 1k chats per epoch (large), ~7 min (base). Not timed yet.
- Known issues / TODO: first torch run is on the user's laptop (`--debug`); not in `liars plan`
  sizing; priority stays QLoRA -> zero-shot -> DeBERTa -> line_mil / Qwen only if quota allows.
- Explain-it-to-me: instead of asking "who is the liar?" about the whole chat, the model asks
  "does this one line contradict the evidence?" for every line, then the speaker whose worst line
  looks most contradictory is the liar. It learns this from the liar label alone.
- Commit: (user commits; see git log)

### Prep extra — improvement loop (2026-10-27)
- Status: built; dry-run tested end to end in the writing session; no real (GPU) loop run yet
- Built:
  - `src/liars/loop/`: ledger (state.json + append-only ledger.jsonl, resumable), catalog
    (branches + search spaces; checks knobs exist and the first value = config default),
    planner (proposals -> baselines -> one-knob mutations of the top-2 branches -> recombination),
    guardrails (budget x 1.2, per-run cap, allowed override prefixes, no data./paths./pseudo,
    no duplicates, unlabelled test), executor (local / kaggle via kaggle_run.py / dry simulation
    with failure injection), recovery (error classes -> retry / mutate / fail / escalate; systemic
    after 3 experiments), evaluate (sanity tests, acc/nll/brier/ece/by lie type, McNemar +
    bootstrap, min detectable gain), orchestrator (the cycle + report.md), preflight (doctor).
  - `ensemble.py`: `vote`, `caruana` methods (inner-CV scored like weighted).
  - CLI: `liars loop run|status|resume|halt|doctor`.
  - Agents: sources in `agents/agents/` (loop-planner, loop-auditor, loop-researcher) and
    `agents/commands/improve-loop.md`; `bash scripts/install_agents.sh` copies them into `.claude/`
    (remote tools may not write .claude/ directly).
  - Docs: `docs/IMPROVE_LOOP.md` (plan, rules, playbook), `docs/KNOWLEDGE.md` (sourced techniques,
    IDs cited by configs/loop.yaml, build queue).
  - Tests: `tests/test_loop.py` (recovery, guardrails, sanity, promotion stats, vote/caruana,
    branch consistency, dry runs with OOM recovery, escalation + resume, systemic failure, proposals).
- Dry run (1200 synthetic train / 300 test, 8 iterations): 22 experiments, 3 promotions, champion =
  vote ensemble, 16.6 of 18 GPU h (simulated), stopped when nothing fit the remaining budget.
- Commands: `pytest -q tests/test_loop.py`; `liars loop doctor --executor kaggle`; `liars loop run --executor dry --iters 6`.
- Independent review (separate agent): 11 bugs found and fixed (mid-batch escalation lost results;
  failed baseline killed a branch forever; skipped runs re-planned; recovery forgot the failed config;
  '401'/'500' in progress logs misread; fragile Kaggle status/timeout/fetch handling; resume kept error
  counts; proposals could lower estimates; id collisions; one ensemble method failing killed all;
  max_experiments off by one). Regression tests added for each.
- Known issues / TODO: est_gpu_h for qlora_qwen and line_mil are estimates; build queue in
  docs/KNOWLEDGE.md (self-distillation, top-k MIL loss, minimal pairs, vLLM, cascade) not built.
- Explain-it-to-me: the loop keeps a lab notebook (the ledger). Each round it runs the most
  promising untried experiment, checks the result is sane, and only crowns a new best model if
  it wins by more than chance (McNemar test). Different models are also combined by voting.
  When something breaks it retries in a cheaper way, and if it cannot fix it, it stops and asks.
- Commit: (user commits; see git log)


### First real GPU runs on the new account (2026-10-29, Kaggle `namanbhandari597`)
- Account switch: Kaggle user is now `namanbhandari597` (OAuth login, `KAGGLE_USERNAME` in ~/.bashrc);
  laptop is Linux, Python 3.12 venv in `liars-chatroom/.venv`. Datasets `liars-code`, `liars-bench`
  created (private). Runs use `--code dataset` (loop default `loop.kaggle.code: dataset`).
- Two failed starts (~0.2 GPU h): notebook had no internet (account not phone-verified) -> pip install
  failed. Fixed by verifying; the notebook now checks internet first and stops in seconds without it.
- line_mil v1 (`line_mil_1028-195601`, lr 1.5e-5, 2 epochs): 57.5% held-out, still rising -> under-trained. 38 min.
- line_mil v2 (`line_mil_1028-205041`, lr 3e-5, 4 epochs): **96.0% held-out (400), 98.0% test (1000)**,
  lie-line hit rate 97.5%, 67 min, 8.2 GiB. Weakest: two_step 76.5% held-out / 87.8% test. Now the
  config default; loop branch est 1.2 h, prior high, space context_before / epochs / lr first.
- Qwen QLoRA bench run (`qwen-bench`) started in parallel; results pending.
