# Prep report: what was built and what it scored

Written 2026-10-27, at the end of the autonomous prep run (Phases 0–8 and 10).
The detailed handoff log is `PROGRESS.md`; this file is the short, readable version.

## Bottom line

- The pipeline is complete for the prep plan except Phase 9 (the timed dress rehearsal,
  which needs you). Every phase has tests, a `--debug` mode and a README section.
- Best result on our synthetic benchmark: **98.0% holdout / 97.3% bench test**, from the
  average of the fine-tuned Gemma model (QLoRA) and DeBERTa. The best single model,
  QLoRA on `google/gemma-4-12B-it`, gets 97.5% / 96.8%.
- No external APIs anywhere. All predictions come from open models run on Kaggle T4s.
- GPU used this week: **~7.4 h** of the ~30 h weekly quota (account `panavrg`).

## Results on the benchmark

The benchmark is our own synthetic data. Bench train has 2000 chats in the casual "alt"
style. Holdout = fold 0 = 400 chats, never used for training or tuning. Bench test has
1000 chats whose labels the models never see.

| Model | Holdout | Bench test | GPU time | Notes |
|---|---|---|---|---|
| TF-IDF baseline (leak detector) | 27.0% | 26.9% | CPU | Near chance = no surface leak in the data |
| NLI DeBERTa-large, zero-shot | 71.5% | 73.4% | 0.17 h | No training |
| Gemma-4-12B, zero-shot | 84.5% | 84.3% | ~0.4 h | Reads the A/B/C/D answer probabilities |
| DeBERTa-large multiple choice, 5-fold | 92.8% | 95.8% | 2.3 h | 93.7% across all 5 folds |
| **Gemma-4-12B QLoRA** (main model) | **97.5%** | **96.8%** | 2.6 h | 2 epochs on 1520 chats |
| Ensemble: QLoRA + DeBERTa (average) | **98.0%** | **97.3%** | – | Best overall |
| Ensemble: all 5 models (weighted) | 97.5% | 97.1% | – | QLoRA weight 0.94 |

By lie type (bench test, QLoRA): 96–100% on number, time, day, place, owner, negation and
implicit lies. **Two-step lies are the weak spot (86%)**: a speaker names a thing in one
message and gives the wrong detail later with "it" (see "Findings" below).

## What was built, phase by phase

| Phase | What it does | Key outcome |
|---|---|---|
| 0 Scaffold | Installable `liars` package, canonical JSONL schema + validator, config system with `--set` overrides, run log, CLI | Every later piece reads/writes one schema |
| 1 Synthetic generator | World-based chat generator: 8 lie types, 2 style packs (neutral and casual texting), 3 difficulties, every chat checked by a structural verifier | Benchmark sets: stage1 (15k), train (2k), test (1k) |
| 2 CV + baselines + leak check | Stratified folds, k-fold and holdout training modes, TF-IDF baseline, leak check, scorer, submission writer | Leak check caught the first generator leaking (38% from word patterns alone); fixed to chance level |
| 3 NLI zero-shot | Each chat line checked against the evidence with an NLI model | 71% with no training |
| 4 Kaggle runner | One script uploads code + data, pushes a GPU notebook, waits, and pulls results back | Runs launch from the terminal; no clicking in Kaggle |
| 5 LLM zero-shot + benchmark | Asks an open LLM "which speaker contradicts the evidence?" and reads the A–D probabilities; tested 6 open models | Gemma-4-12B won (82% raw, 84% with tuning) |
| 6 QLoRA (main model) | Fine-tunes small adapter weights on the 4-bit LLM, trained only on the A–D answer; time guard aborts runs over 4 GPU hours | 97.5% holdout; ~1 h per epoch per 1500 chats on a T4 |
| 7 DeBERTa | Each speaker's lines are marked in turn and the model picks the most suspicious of the 4 | 93.7%; a cheap second opinion |
| 8 Ensemble + reports + rules | Combines models with honest scoring, HTML error report, framework for hand-written rules | Ensemble 98.0%; reports in `data/reports/*.html` |
| 10 Pseudo-labelling | Turns confident test predictions into extra training data; **off by default** until organizers allow it | See Phase 10 below |

### Phase 10 detail (pseudo-labelling, bench only)

- 970 of 1000 bench test chats had QLoRA confidence ≥ 0.95. Those labels were 98.0% correct.
- Holdout effect on DeBERTa: PHASE10_RESULT

## Problems found and fixed along the way

| Problem | Cause | Fix |
|---|---|---|
| Generator leaked (TF-IDF 38% vs 25% chance) | Liars used telltale phrases and unrealistic values | Honest speakers now use the same phrasing; lie values stay in realistic ranges; re-checked at chance |
| Qwen3 models always answered the same letter | They start answers with "The…", not a letter | Optional "Answer:" prefix; letter scoring made prefix-aware |
| `--set x=Answer:` broke a Kaggle run | YAML read it as a dict | Override parser fixed |
| DeBERTa crashed on GPU (fp16 gradients) | New transformers loads the checkpoint in fp16 | Load in fp32; regression test |
| DeBERTa-large ran out of memory on a T4 | Full fine-tune activations | Gradient checkpointing on by default |
| DeBERTa run got stuck at chance (24%) | Known random collapse of DeBERTa-v3-large | Collapse guard: detects it after 20% of training and restarts with a new seed and lower learning rate |
| `git push` hangs | macOS Keychain prompt | Kaggle uses an uploaded code snapshot instead (works); you still need to push once |
| Kaggle Secrets missing in pushed notebooks | Secrets were not attached to notebooks pushed from the terminal | Not needed: code comes from a private dataset, models are not gated |

## Findings worth knowing for competition day

1. **Fine-tuning is the big win.** Zero-shot Gemma reached 84%. Training on 1520 chats took
   it to 97.5%. With their training data, QLoRA should be the first GPU job.
2. **Most remaining errors are ambiguous chats, not model mistakes.** In 3 of the 4 two-step
   errors I read, an honest "it was 12 bucks" came right after someone else's line about a
   *different* item. A human would read it the same wrong way. On their data, check
   whether errors are label ambiguity before spending time on them.
3. **Averaging two different models helps a little.** DeBERTa is weaker but makes different
   mistakes; averaging fixed 4 of QLoRA's 10 holdout errors and broke 2.
4. **Generic rules are weak.** The number and day/time mismatch rules point at the right
   speaker only ~45–54% of the time when they fire. Rules should be written on the day
   against their actual phrasing.
5. **Timing for sizing runs (T4):** QLoRA Gemma-12B ≈ 1 h per epoch per 1500 chats;
   DeBERTa-large ≈ 10 min per epoch per 1500 chats (28 min per fold incl. prediction).

## What still needs you

1. **Event dates:** still unknown. Fill them in `PROGRESS.md` → Key facts.
2. **Push to GitHub once:** run `git push origin main` in a terminal (the Keychain prompt
   blocks me).
3. **Phase 1 sample read-through:** skim `data/samples/*.txt` (lies are marked `<-- LIE`)
   and say if the chats look realistic enough.
4. **Phase 9 dress rehearsal:** a separate, timed session. Start it when you are ready.
5. **Pseudo-labelling:** stays off until the organizers confirm it is allowed.

## Where things are

- Plan: `BUILD_MAP.md`. Handoff log: `PROGRESS.md`. How to run each part: `README.md`.
- Kaggle workflow: `docs/KAGGLE.md`. Error reports: `data/reports/*.html` (open in a browser;
  start with `ens_bench_qd.html` and `llm_qlora_1027-002454.html`).
- Run log: `data/runs.csv`. Predictions: `data/oof/` (held-out) and `data/preds/` (test).
