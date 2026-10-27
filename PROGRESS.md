# PROGRESS — Liar's Chatroom

> The handoff log. A new Claude Code session reads this file instead of old chats.
> Claude updates it at the end of every phase (see CLAUDE.md → Phase protocol).

## Current state
- Mode: PREP            <!-- PREP | REHEARSAL | COMPETITION -->
- Current phase: 7 — code + debug done (wip commit 6798b28); GPU run next
- Last completed phase: 6 (QLoRA gemma-4-12B-it) — done
- Next step: Phase 7 real run (DeBERTa 5-fold on bench train, Kaggle), then Phase 8 ensemble,
  Phase 10 entry. Phase 8 and 10 code is already committed as wip (29352e0, c00f4cf).
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
- Kaggle GPU: T4 x2 (Python 3.13, torch 2.11.0+cu128, transformers 5.16.1, bitsandbytes 0.50.2; fp16 only, no bf16; 4-bit NF4 OK) — weekly quota used so far: ~4.4 h (week of 2026-10-26)
- GPU hours used in prep (account panavrg): 4.4 h (Phase 4 NLI 0.17 h; Phase 5 benchmark + winner ~1.6 h; Phase 6 QLoRA 2.58 h)
- Best result so far: llm_qlora_1027-002454 (Kaggle) — holdout — 97.5% fold 0 / 96.8% bench test — gemma-4-12B-it QLoRA r16, 2 epochs, best step 138/190, TTA 2, T=0.25. (Zero-shot gemma: 84.5% / 84.3%; NLI-large: 71.2% OOF / 73.4% test)
- Stage-1 / pretrained checkpoints: (run name — path / Kaggle dataset name)
- LLM benchmark winner: google/gemma-4-12B-it (Apache-2.0, not gated) — 82.4% zero-shot on 500 bench chats (no TTA) — 110 chats/min on T4 4-bit (66/min with TTA 2), 10.0 GiB. Fallback: Qwen/Qwen3-14B 69.4% (needs prompt.answer_prefix="Answer:")
- QLoRA time per epoch on T4: google/gemma-4-12B-it — 1520 chats (prompt p50 318 / max 489 tokens) — max_len 1024 — 0.96 h/epoch (36.3 s/step, 16 chats/step). Rule of thumb: ~1600 chats/h at these lengths; time grows ~linearly with prompt length. Whole run (2 epochs + 9 evals + calib + TTA-2 predict of 1400 chats) = 2.58 h. Predict = 51 chats/min with TTA 2. 8.6 GiB peak.
- Template packs available: default (neutral sentences), alt (casual texting) — src/liars/synth/templates/
- Benchmark sets: data/bench/{stage1 (15k default mixed, seed 101), train (2k alt med+hard, seed 202, first_names), test (1k alt, seed 303, labels hidden) + test_labels.csv}. Regenerate: README Phase 1.

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
    (`train.inner_frac`) if early stopping looks noisy.
- Explain-it-to-me: LoRA freezes the 12B base model and trains small low-rank "patch"
  matrices (~1% of the size) added to the attention and MLP weights, so it fits on a T4
  in 4-bit. We still read only the logits of "A"/"B"/"C"/"D" at the answer position, and the
  loss is plain 4-way cross-entropy on those — the LLM becomes a 4-class classifier that
  reuses its reading skills. The holdout fold is never used for training, early stopping
  or temperature, so its accuracy is an honest estimate of unseen data; the ensemble
  later fits weights on it, which only works if nothing else has touched it.
- Commit: (see git log, "phase 6: ...")
