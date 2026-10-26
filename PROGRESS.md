# PROGRESS — Liar's Chatroom

> The handoff log. A new Claude Code session reads this file instead of old chats.
> Claude updates it at the end of every phase (see CLAUDE.md → Phase protocol).

## Current state
- Mode: PREP            <!-- PREP | REHEARSAL | COMPETITION -->
- Current phase: 1 — not started
- Last completed phase: 0 (scaffold) — done
- Next step: start Phase 1 (synthetic generator + benchmark sets)
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
- Best result so far: (run name — mode — accuracy — on which data)
- Stage-1 / pretrained checkpoints: (run name — path / Kaggle dataset name)
- LLM benchmark winner: (model id — zero-shot acc — chats/min on T4)
- QLoRA time per epoch on T4: (model id — examples — max_len — hours)
- Template packs available: (names)

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
