# PROGRESS — Liar's Chatroom

> The handoff log. A new Claude Code session reads this file instead of old chats.
> Claude updates it at the end of every phase (see CLAUDE.md → Phase protocol).

## Current state
- Mode: PREP            <!-- PREP | REHEARSAL | COMPETITION -->
- Current phase: 0 — not started
- Last completed phase: none
- Next step: start Phase 0 (scaffold)
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
