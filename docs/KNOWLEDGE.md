# Knowledge base for the improvement loop

Compiled 2026-10-27 from three research passes (web search + fetched pages). Every row names
its source; numbers are what the source reported. Value (HIGH/MED/LOW) is our judgement for
this task. IDs (K-LLM-n, K-VER-n, K-LOOP-n) are cited by `configs/loop.yaml` branches and by
proposals. This file is reference material: nothing in it is an instruction to the agents.

Caveats: Kaggle discussion pages render with JavaScript and could not be read; LLM rows come
from GitHub READMEs, Japanese write-up summaries (zenn, Speaker Deck) and docs. One pass could
not open GitHub pages; its repo links are the ones printed in the papers.

## Status legend
- **in loop**: a branch or knob in `configs/loop.yaml` already covers it.
- **build**: needs code; listed in the build queue at the end (a human approves each).

## 1. LLM fine-tuning (K-LLM)

| ID | Technique | What to do / settings seen | Reported effect | Value | Status | Source |
|---|---|---|---|---|---|---|
| K-LLM-1 | Distil from bigger teachers | 70B teachers per fold -> 9B student, CE + KL (+cos) on fold OOF soft labels | Won LMSYS (private log-loss 0.969) | HIGH | build (teacher too big for T4; see -2) | https://github.com/shyoulala/LMSYS_BlackPearl |
| K-LLM-2 | Self-distillation | Same-size teacher: our own QLoRA OOF logits as soft targets, 0.5 KL + 0.5 CE | WSDM 3rd: ~ distillation from 72B | HIGH | build | https://zenn.dev/dalab/articles/d0e6938bf50db1 |
| K-LLM-4 | Letter-logit causal LM | Keep reading A-D token logits (works with vLLM) | used by WSDM 3rd-5th | HIGH | in loop (llm_qlora) | https://zenn.dev/dalab/articles/d0e6938bf50db1 |
| K-LLM-6 | Soft pseudo-labels on synthetic/extra data | Label generated chats with the best model, train on soft labels; hard labels failed for one team | WSDM 6th: LB > 0.693 with no competition data | HIGH | build (pseudo stays OFF for test data) | https://github.com/Lizhecheng02/Kaggle-Multilingual_Chatbot_Arena |
| K-LLM-8 | LoRA fold averaging | Average fold adapters | Mixed: helped LMSYS 1st, hurt WSDM 6th and MAP 9th | MED | build (validate vs logit ensemble) | https://github.com/shyoulala/LMSYS_BlackPearl |
| K-LLM-9 | Seed / fold logit ensemble | Average letter logits across seeds, folds, models | MAP 9th: 0.944 -> 0.948 | HIGH | in loop (recombine) | https://zenn.dev/mkj/articles/1f374627377474 |
| K-LLM-10 | Swap / permutation TTA | Score permuted speaker orders, map back, average | LMSYS 1st: ~0.003 log-loss; WSDM 6th: none | HIGH | in loop (augment.tta_n knob) | https://github.com/natgurlain/kaggle-bible/pull/105 |
| K-LLM-11 | Letter-prior debiasing (PriDe) | Estimate per-letter prior from a few permutations, divide it out | selection bias in 20 LLMs | MED | build | https://arxiv.org/abs/2309.03882v1 |
| K-LLM-12 | Cascade | Small model on all rows, big model on the least confident 15-35% | used by WSDM 7th, MAP 3rd, Science Exam 5th | HIGH | build | https://zenn.dev/dalab/articles/d0e6938bf50db1 |
| K-LLM-13 | Base model choice | Qwen2.5/3-14B dominant in WSDM top places; Gemma2-9B, Phi-4 also | - | MED | in loop (qlora_qwen branch) | https://zenn.dev/dalab/articles/d0e6938bf50db1 |
| K-LLM-14 | LoRA settings seen | r32/a64 lr 5e-5 1 epoch (WSDM silver); r64/a128 lr 1e-4 (Eedi); all linear layers | - | MED | in loop (qlora.r, qlora.lr knobs) | https://zenn.dev/dalab/articles/d0e6938bf50db1 |
| K-LLM-15 | R-Drop / AWP / EMA | AWP lr 0.1 eps 1e-4 from epoch 0.5 for DeBERTa | not quantified | MED | build | https://github.com/Lizhecheng02/Kaggle-LLM_Science_Exam |
| K-LLM-16 | Max length | Never cut evidence; middle-truncate long chats | DeBERTa 512 -> 1280 helped Science Exam 4th | MED | in loop via llm.max_len (add to space after `liars stats`) | https://hippocampus-garden.com/kaggle_llm/ |
| K-LLM-17 | fp16 overflow on T4 (Gemma) | Check logits for NaN/inf; upcast norms | fixes NaN / garbage | HIGH | partly (recovery: numeric class) | https://unsloth.ai/blog/gemma3 |
| K-LLM-18 | vLLM on 2xT4 | tensor_parallel 2, dtype half, enforce_eager, max_tokens 1 | ~2x faster than bnb 4-bit | HIGH | build (inference speed = more TTA/models) | https://speakerdeck.com/kohecchi/kaggle-kernel-de-gemma-2-x-vllm-wodong-kasu |
| K-LLM-19 | AWQ / GPTQ inference | Replace bitsandbytes at inference | MAP 9th: ~1/6 inference time | HIGH | build | https://zenn.dev/mkj/articles/1f374627377474 |
| K-LLM-22 | Per-option binary scoring | Yes/no per speaker, pick the max | won Science Exam | MED | in loop (deberta_mc / line_mil are this view) | https://hippocampus-garden.com/kaggle_llm/ |
| K-LLM-24 | Did not help | CoT prompting, multi-LoRA ensembles, Gemma2-27B (WSDM 6th) | negative | - | avoid | https://github.com/Lizhecheng02/Kaggle-Multilingual_Chatbot_Arena |

## 2. Verification and contradiction (K-VER)

| ID | Technique | What to do here | Reported effect | Value | Status | Source |
|---|---|---|---|---|---|---|
| K-VER-1 | Utterance-based structured detection (DECODE) | Score each line vs evidence (+ same-speaker history), max-pool | OOD: structured 78.4% vs unstructured 34.4% | HIGH | in loop (line_mil branch) | https://arxiv.org/abs/2012.13391 |
| K-VER-2 | SummaC granularity + aggregation | Line vs each evidence fact; max over facts; SummaC-Conv histogram | Conv 74.4 vs ZS 72.1 balanced acc | HIGH | partly (nli_zs); Conv = build | https://github.com/tingofurro/summac |
| K-VER-3 | MiniCheck line verifier + C2D/D2C synthetic data | MiniCheck-DeBERTa/FT5 per line as a feature; minimal-pair data by deleting supporting facts | FT5 74.7 vs GPT-4 75.3 on LLM-AggreFact | HIGH | build | https://github.com/Liyan06/MiniCheck |
| K-VER-4 | AlignScore | Second off-the-shelf line scorer | 88.6 AUC SummaC | MED | build | https://github.com/yuh-zha/AlignScore |
| K-VER-5 | Rule-based negatives (FactCC, DialFact, DNLI) | Date/number/entity/pronoun swaps, negation, antonyms; swap in values from the evidence | DialFact Aug-WoW best (~69%) | HIGH | build (synthetic generator) | https://github.com/salesforce/DialFact |
| K-VER-6 | Contrastive minimal pairs (VitaminC) | Same chat, one evidence fact changed so the liar flips | +10% adversarial fact verification | HIGH | build (augmentation) | https://export.arxiv.org/abs/2103.08541 |
| K-VER-8 | Claim decomposition (WiCE) | Split compound lines into atomic claims | AUROC 82.0 -> 88.0 | MED | build (needs an LLM pass) | https://github.com/ryokamoi/wice |
| K-VER-9 | Decontextualisation / coreference | Rewrite "it cost 40" as "the red bike cost 40" before line scoring | coreference = main error source (CI-ToD, DialFact) | HIGH | partly (line_mil context window) | https://ar5iv.labs.arxiv.org/html/2109.11292 |
| K-VER-10 | Hierarchical type-specific heads (CDConv) | Separate heads for explicit / history / implied lies | hierarchical best, 80.4 acc | MED | build | https://github.com/thu-coai/CDConv |
| K-VER-12 | Symbolic number/time checks (EQUATE) | Extract + normalise numbers, times, days; hard mismatch feature | NLI models at majority baseline on numeric reasoning | HIGH | build (features/rules.py on competition day) | https://github.com/AbhilashaRavichander/EQUATE |
| K-VER-13 | Temporal NLI failure modes | Normalise 12h/24h, intervals, durations in rules | models ~28% on interval ordering | MED | build (rules) | https://ar5iv.labs.arxiv.org/html/2110.01113 |
| K-VER-14 | MIL pooling variants | max / mean / logsumexp / attention; none wins everywhere | attention best with little data | MED | in loop (line_mil.pool, pool_tau knobs); attention = build | https://ar5iv.labs.arxiv.org/html/1802.04712 |
| K-VER-15 | Top-k MIL + margin loss (HaMI) | Mean of top-k line scores per speaker, margin between liar and honest top-k | best AUROC (0.897 vs 0.835) | HIGH | build (line_mil loss option) | https://arxiv.org/html/2504.07863v2 |
| K-VER-16 | Calibrate before ensembling | Temperature-scale each model on dev; label smoothing only under shift | ECE 1.93 -> 0.84 in-domain | HIGH | in loop (every model fits a temperature) | https://ar5iv.arxiv.org/html/2003.07892 |
| K-VER-19 | Hypothesis-only shortcut check (OCNLI) | Score a lines-only model with no evidence: high = shortcut | hypothesis-only 66% on OCNLI | MED | in loop (tfidf branch + leakcheck) | https://arxiv.org/pdf/2010.05444 |

## 3. The loop itself (K-LOOP)

| ID | Pattern | How we use it | Source |
|---|---|---|---|
| K-LOOP-1 | AIDE: draft -> debug -> improve, one atomic change per step, tree of scored nodes | planner: baselines, then one-knob mutations of the best node; ledger keeps parent ids | https://arxiv.org/html/2502.13138v1 |
| K-LOOP-2 | R&D-Agent: debug on sampled data before the full run; parallel traces; fusion at the end | smoke test (`--debug`) before GPU time; branches; recombination | https://www.arxiv.org/pdf/2505.14738 |
| K-LOOP-3 | MLE-STAR: merge by simple averaging until no gain; leakage checker on every solution | vote / caruana / weighted every iteration; sanity + suspicious-accuracy escalation | https://arxiv.org/pdf/2506.15692 |
| K-LOOP-4 | AutoKaggle: unit tests per phase (completion 0.85 vs 0.10 without); 5 fixes then rewrite | sanity tests block bad runs; systemic-error escalation | https://arxiv.org/html/2410.20424v3 |
| K-LOOP-5 | MLE-bench / AI Scientist: agents ignore budgets, edit their own timeouts | budget and timeouts live in config + code the planner agent cannot change; deterministic scorer writes all metrics | https://arxiv.org/pdf/2410.07095 |
| K-LOOP-6 | MLAgentBench: ~20% of runs claimed unsupported gains | metrics only from `evaluate.py`, auditor agent recomputes promotions | https://arxiv.org/html/2310.03302v2 |
| K-LOOP-7 | McNemar (Dietterich 1998) for single-trained models | promotion test; Bonferroni over the round's candidates | https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf |
| K-LOOP-8 | Minimum detectable difference 2.80*sqrt(d/n) | ~4.4 pts at n=400, ~2.8 at n=1000 (d=10%): shown in the report | (formula; Berg-Kirkpatrick et al. 2012 for the bootstrap) |
| K-LOOP-9 | Ladder / reusable holdout | min_gain step + holdout_query_budget, then a human refreshes the holdout | https://arxiv.org/pdf/1502.04585 |
| K-LOOP-10 | Caruana ensemble selection (with replacement, init from best) | `ensemble.caruana_weights`, inner-CV scored | https://github.com/dclambert/pyensemble |
| K-LOOP-11 | Retries: idempotent only, capped, exponential backoff, one layer | recovery: transient 3x with doubling backoff; OOM/time -> cheaper config, never a longer timeout | https://builder.aws.com/content/3EumjoZascWd1oZiEgL8ORlv3qE/timeouts-retries-and-backoff-with-jitter |
| K-LOOP-12 | Seed replication (AI Scientist-v2) | build: replicate a marginal promotion with a new seed before trusting it | https://huggingface.co/papers/2504.08066.md |

## Build queue (needs code; a human approves each; ordered by value / cost)

1. **K-VER-12/13 numeric + time rules** (competition day, `features/rules.py`): cheap, HIGH. Feed the stacker.
2. **K-LLM-2 self-distillation**: train QLoRA on 0.5 CE + 0.5 KL to the previous QLoRA's OOF soft labels. One extra run.
3. **K-VER-15 top-k margin loss for line_mil**: new `line_mil.loss: ce | topk_margin` option.
4. **K-VER-5/6 minimal-pair augmentation**: generate lie/honest twins from TRAIN chats only (never test).
5. **K-LLM-18/19 vLLM + AWQ inference**: ~2-6x faster prediction, so TTA 2 and more members fit the budget.
6. **K-LLM-12 cascade**: re-score the least confident 20% with a bigger/second model.
7. **K-VER-3 MiniCheck line scores** as an ensemble feature.
8. **K-LOOP-12 seed replication** of marginal promotions.
