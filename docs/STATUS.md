# Liar's Chatroom — prep status: done and left

As of 2026-10-27

## Summary

Prep is done except Phase 9, the timed dress rehearsal, which must run in separate sessions
with you. Phases 0–8 and 10 are built, tested (144 tests pass) and committed locally.

Best result on our synthetic benchmark: 98.0% holdout and 97.3% test, from averaging the
fine-tuned Gemma-12B (QLoRA) and DeBERTa. About 21 of the ~30 weekly GPU hours are left, saved
for the rehearsal.

Four things only you can do: fill in the event dates, push to GitHub once, skim the sample
chats, and start Phase 9.

## Done

Every prep phase except Phase 9 is complete, with its Done-when checks run and logged in
PROGRESS.md.

| Phase | What it gives you | Key number |
| --- | --- | --- |
| 0 Scaffold | Installable `liars` package, one data schema, config system, CLI | 144 tests, ~45 s |
| 1 Synthetic data | Chat generator: 8 lie types, 2 writing styles, 3 difficulties, every chat verified | Bench sets: 15k / 2k / 1k chats |
| 2 CV + baselines + leak check | Folds, holdout mode, scorer, submission writer, leak detector | Leak check at chance (25%) |
| 3 NLI zero-shot | Checks each line against the evidence, no training | 73.4% test |
| 4 Kaggle runner | Push GPU jobs and pull results from the terminal | Works without Kaggle Secrets |
| 5 LLM zero-shot | Open LLM picks the liar; 6 models compared | Gemma-4-12B: 84.3% test |
| 6 QLoRA (main model) | Fine-tunes Gemma-4-12B on the A–D answer | 97.5% holdout, 96.8% test, 2.6 GPU h |
| 7 DeBERTa | Marks each speaker in turn, picks the most suspicious | 93.7% all folds, 95.8% test |
| 8 Ensemble + reports + rules | Honest model averaging, HTML error report, rule framework | 98.0% holdout, 97.3% test |
| 10 Pseudo-labelling | Trains on confident test predictions; OFF until allowed | DeBERTa 92.75% → 96.5% holdout |
| Final prep round | `liars plan` sizing, both-T4 prediction, TTA 1 default, ambiguity fix, safer Kaggle runner, checklist, methodology, version lock | Worked example: 5k train + 3k test ≈ 4.9 GPU h |

## Left: Phase 9 dress rehearsal

Phase 9 is the only phase not done: a timed practice run on data in a format we have never
seen, to prove raw files can become a validated submission in under 90 minutes. It needs three
separate sessions so the session doing the rehearsal cannot already know the format.

- [ ] **Step A, hidden data (new session).** Say "Start Phase 9, step A." Claude writes a
  generator outside the repo, puts only `train.csv`, `test.csv` and `sample_submission.csv`
  into `data/raw/`, keeps the labels and a scorer hidden, and ends the session. About 3k train
  and 1k test chats, with a new format, a new lie type, a skewed liar position and a planted
  weak leak.
- [ ] **Step B, rehearse (fresh session).** Set Mode to REHEARSAL, start a timer, and follow
  COMPETITION_CHECKLIST.md prompt by prompt, up to launching the main models on Kaggle for real.
- [ ] **Step C, post-mortem.** Say "Phase 9 post-mortem." Claude logs the timeline, what
  slowed you down, the scores and fixes, then sets Mode back to PREP.

Phase 9 is done when:

- [ ] Raw files reach a validated first submission in under 90 minutes
- [ ] The adapter round-trip test passes on the rehearsal data
- [ ] The leak check finds the planted quirk
- [ ] QLoRA launches from the runbook prompt without code changes
- [ ] The post-mortem and fixes are recorded

It is also the first real GPU test of four new pieces: `liars plan`, `liars predict --gpus 2`,
the automatic code upload, and the TTA check. Expected GPU use: about 4–6 hours.

## Needs you

These can only be done by you; none of them takes more than a few minutes except the rehearsal.

- [ ] **Event dates.** Fill in the 48-hour window in PROGRESS.md → Key facts. Until then the
  plan assumes the full schedule.
- [ ] **Push to GitHub once.** Run `git push origin main` in a terminal. A macOS Keychain
  prompt blocks it for me; Kaggle runs work meanwhile from an uploaded code copy.
- [ ] **Skim the sample chats.** Open `data/samples/*.txt` (lies are marked `<-- LIE`) and say
  whether they look realistic enough. This is the last open item from Phase 1.
- [ ] **Check the Kaggle quota bar.** Confirm about 21 h are left and the reset time
  (expected Sat 31 Oct, 00:00 UTC).
- [ ] **Start Phase 9** when you can sit through a timed run (target: first submission within
  90 minutes).
- [ ] **Optional: Low Power Mode.** Your Mac is on battery in Low Power Mode, which makes local
  runs about 2× slower; plug in for the rehearsal.
- [ ] **Decide:** should stage-1 pretraining get a test run during the rehearsal, or wait until
  we know their train size?

## Open questions for the organizers

These answers change what we run on the day; until they arrive, the plan uses the assumption
in the last column.

| Question | Why it matters | Current assumption |
| --- | --- | --- |
| Is pseudo-labelling allowed? | It added 3.75 points to DeBERTa on the bench | Off (`pseudo.enabled: false`) |
| Exact start date and time? | Sets the GPU quota week and the schedule | Unknown |
| Approximate train and test sizes? | Decides epochs, subsampling and DeBERTa via `liars plan` | Plan for 2k to 20k |
| Public / private leaderboard split? | How much to trust the leaderboard vs our holdout | Trust the holdout |
| Is our own synthetic training data allowed? | Stage-1 pretraining option | Assumed yes; confirm |
| Any sample of the data format before the event? | Lets us write the adapter early | None yet |

## Not done on purpose, and known limits

None of these blocks the rehearsal or the event; each was skipped to save GPU time or because
it only makes sense with the real data.

| Item | Why it is not done | When to revisit |
| --- | --- | --- |
| Stage-1 pretraining on synthetic data (QLoRA, DeBERTa) | ~3–6 GPU h, and the bench is already near 98% so it could not show a gain | If their train set is small (under ~5k) |
| Training on both T4s at once | Only prediction is split; training a single job on 2 GPUs needs bigger changes | If a QLoRA job will not fit in 4 h |
| Measuring TTA 2 vs TTA 1 directly on QLoRA | Needs a GPU run; indirect evidence says the gain is under 0.5 points | Rehearsal logs it for free with `augment.tta_n=2` |
| Regenerating the bench sets without ambiguous chats | Every recorded score refers to the current sets | Not needed; new data is clean by default |
| Real rules for their phrasing | Generic rules are weak (45–54% precision) | On the day, after forensics |
| Full Kaggle package list in requirements-lock.txt | The GPU image list is only written by runs from now on | Copy it in after the rehearsal |
| METHODOLOGY.md results and adapter sections | They describe the real competition data | During the event (marked TODO) |
| Smaller fallback model as a real option | Gemma-4-E4B got only 91.75%; a 12B on fewer chats is better | Last resort only |

## Where things are

All files are in the repo `liars-chatroom` (private, local commits not yet pushed).

| What | File or command |
| --- | --- |
| Handoff log, all numbers and decisions | `PROGRESS.md` |
| The plan and the competition-day runbook | `BUILD_MAP.md` |
| One-page event checklist | `COMPETITION_CHECKLIST.md` |
| Readable prep summary | `docs/PREP_REPORT.md` |
| Write-up skeleton for reviewers | `METHODOLOGY.md` |
| GPU plan for given data sizes | `liars plan --train N --test M --chars P50` |
| Run a job on Kaggle / get results back | `python scripts/kaggle_run.py push …` / `fetch --name …` |
| Error reports (open in a browser) | `data/reports/ens_bench_qd.html`, `data/reports/llm_qlora_1027-002454.html` |
| Pinned versions | `requirements-lock.txt` |
