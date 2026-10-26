# PROGRESS — Liar's Chatroom

> The handoff log. A new Claude Code session reads this file instead of old chats.
> Claude updates it at the end of every phase (see CLAUDE.md → Phase protocol).

## Current state
- Mode: PREP            <!-- PREP | REHEARSAL | COMPETITION -->
- Current phase: 4 — not started
- Last completed phase: 3 (zero-shot NLI) — done
- Next step: start Phase 4 (Kaggle packaging + first GPU run; git push allowed from Phase 4)
- Autonomy (2026-10-27): the user is away and said to keep going phase after phase
  without waiting for "next", verifying every Done-when check myself. Stop only for
  things that need the user (gated-model licence, a run > 4 GPU hours, anything
  irreversible beyond the git/Kaggle permissions in Key facts). Items that need the
  user's eyes (e.g. Phase 1 sample read-through) are logged as "pending user".
- Blockers / waiting on: organizers' answer on pseudo-labelling; event start date
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
- Kaggle GPU: T4 x2 preferred for 4-bit LLMs — weekly quota used so far:
- GPU hours used in prep (account panavrg):
- Best result so far: nli_zeroshot_1027-035542 — kfold — 63.5% OOF — 200-chat CPU subset of bench train (base DeBERTa NLI, joint_concat). TF-IDF leak detector = 24.65% (chance, by design)
- Stage-1 / pretrained checkpoints: (run name — path / Kaggle dataset name)
- LLM benchmark winner: (model id — zero-shot acc — chats/min on T4)
- QLoRA time per epoch on T4: (model id — examples — max_len — hours)
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
