# Competition checklist (one page, follow top to bottom)

Full prompts: BUILD_MAP.md → "Competition-day runbook". Log every step with its time in
PROGRESS.md → "Competition log". Max 5 submissions per day; keep the last 2 for SAFE + BOLD.

## Before the event (day before)
- [ ] Kaggle quota bar: hours left this week, and the reset time (Fri/Sat midnight UTC).
- [ ] `git push origin main` works; `pytest -q` passes; `git status` clean.
- [ ] PROGRESS.md: Mode = COMPETITION, event dates filled in. Start a fresh Claude session.

## The 48 hours

| When | Step (runbook prompt) | Takes | You run / check |
|---|---|---|---|
| T+0:00 | **Adapt**: data → canonical JSONL | 45 min | `kaggle competitions download -c <comp> -p data/raw` and unzip; then paste the Adapt prompt. Check: `liars validate` 0 errors on train + test; round-trip test passes |
| T+0:45 | **Forensics + leak hunt** | 30 min | Paste the prompt. Check: leak-check verdict, lie types, `augment.permute_train` on/off decided |
| T+1:15 | **Sizing** | 2 min | `liars stats --in <train>` → note N, M, "total chars" p50/p90 → `liars plan --train N --test M --chars P50 --chars-p90 P90` |
| T+1:20 | **Launch GPU jobs** | 15 min | `liars folds --in <train> --k 5`; `python scripts/kaggle_run.py data --slug liars-comp <train> <test> <folds>`; push zero-shot + QLoRA as 2 parallel sessions (`--datasets liars-comp`, flags from `liars plan`) |
| T+1:35 | **Baseline submission** (format check) | 10 min | `liars train --config configs/baselines.yaml`; `liars submit ...`; upload; leaderboard ≈ 25% means the format is right |
| T+2:00 | **Rules** from their templates | 1–2 h | Paste the rules prompt. Keep only rules with precision ≥ 0.9 (`liars rules-report`) |
| ~T+2:30 | **Zero-shot lands** → first milestone | 15 min | `python scripts/kaggle_run.py fetch --name <job>`; `liars score`; submit |
| ~T+4–5 | **QLoRA lands** → ensemble + milestone | 20 min | fetch; then push DeBERTa (if `liars plan` said RUN); `liars ensemble --runs ... --name e1`; `liars analyze --run ens_e1`; submit |
| if quota left | **Extra members** (optional) | 1–3 h GPU | `configs/line_mil.yaml` and/or `configs/llm_qlora_qwen.yaml`, holdout mode, then add the runs to `liars ensemble` |
| every 3–4 h | **Improvement loop** | 30 min | Paste the prompt; pick ONE fix; rerun only what it touches |
| ~T+24 | **Pseudo-labels** (ONLY if organizers said yes in writing) | 3–5 h GPU | `--set pseudo.enabled=true`; compare holdout with/without |
| T+42 | **Lock in**: SAFE + BOLD, METHODOLOGY.md | 2 h | SAFE = best honest holdout, simplest; BOLD = best ensemble. Fill METHODOLOGY.md TODOs; copy `data/logs/pip_freeze_kaggle.txt` into requirements-lock.txt |
| before deadline | **Select** the 2 final submissions on Kaggle | 5 min | If the competition asks you to choose |

## Before every Kaggle push
- [ ] Code committed (the runner refuses uncommitted changes and re-uploads the code itself).
- [ ] Data dataset has the latest canonical files + folds (re-run `kaggle_run.py data` after any adapter change).
- [ ] `liars plan` estimate ≤ 4 h for this job, and quota left ≥ estimate + 20%.
- [ ] `--datasets liars-comp` (not the bench dataset); run name says what it is.

## Before every CSV upload
- [ ] Made by `liars submit` (validates rows, ids and format against sample_submission). Never edit a CSV by hand.
- [ ] Predictions come only from models / rules. No one read individual test chats.
- [ ] Its honest holdout / OOF accuracy is logged, and better than (or a deliberate variant of) the last upload.
- [ ] Submissions used today < 5; file name + holdout score + leaderboard score go into the Competition log.

## If something breaks
- Kernel ERROR: `python scripts/kaggle_run.py fetch --name <job>` and read the `.log` there.
- Out of memory: QLoRA `--set qlora.micro_batch_size=1 --set qlora.grad_accum=16`; DeBERTa keep `gradient_checkpointing: true`.
- DeBERTa stuck at chance: the collapse guard restarts it once by itself; if it fails twice, skip DeBERTa.
- Leaderboard far below holdout: suspect a train/test difference; stop submitting, investigate.
- Run would take > 4 h: STOP and ask (rule R10 in the runbook).
