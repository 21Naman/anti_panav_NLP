# Improvement loop: plan, architecture, playbook

The loop turns "try things until the score goes up" into an auditable process:
**plan -> act -> test -> evaluate -> improve -> repeat**, with branches that run in parallel and
are recombined, guardrails that block bad or unfair runs, and recovery that retries, makes a
run cheaper, gives up, or stops and asks a human. It does not depend on any particular
dataset: point `loop.data` at any canonical train/test file (+ folds) and it works.

Sources for every rule: `docs/KNOWLEDGE.md` (IDs in brackets below).

## 1. Who does what

| Role | Kind | Does | Cannot |
|---|---|---|---|
| `liars loop run` (orchestrator) | deterministic Python | plans baselines + one-knob mutations, checks guardrails, runs jobs, recovers, scores, promotes, recombines, stops | call any model API; change its own budget |
| `loop-planner` agent | Claude Code subagent (reasoning) | reads report + ledger + error analysis + KNOWLEDGE.md, writes <= 3 justified proposals; code ideas go to the build queue | edit configs, budget, code or results; read test chats |
| `loop-auditor` agent | Claude Code subagent (independent) | recomputes every promotion from the raw files, checks guardrails and budget, halts on any mismatch | fix anything (it only reports / halts) |
| `loop-researcher` agent | Claude Code subagent | adds sourced techniques to KNOWLEDGE.md when ideas run out | change configs or code |
| You | human | approve GPU spend > 4 h, resolve escalations, approve build-queue items, upload submissions | - |

Install once: `bash scripts/install_agents.sh` (sources live in `agents/`, copied to `.claude/`).
Driver: `/improve-loop [dry|local|kaggle] [cycles]` (`.claude/commands/improve-loop.md`) runs
doctor -> status -> planner -> one loop iteration -> auditor -> 5-line report.

## 2. One iteration in detail

1. **Plan** (`planner.py`): first new proposals; else a baseline for each branch without one
   (prior high -> low, cheapest first); else one-knob mutations of the best run of the top-2
   branches, moving a knob to a neighbouring value (AIDE: one atomic change per step
   [K-LOOP-1]). At most `max_concurrent` (2) jobs, within the GPU budget.
2. **Guardrails** (`guardrails.py`): budget (estimate x 1.2 must fit; per-run cap 4 h),
   allowed override prefixes only (never `data.*`, `paths.*`, pseudo-labels unless
   `allow_pseudo`), no duplicate runs, max 60 experiments, test file must be unlabelled.
3. **Smoke test**: a branch's first run is executed once with `--debug` on CPU before any GPU
   time (R&D-Agent [K-LOOP-2]); a config/code error fails there for free.
4. **Act** (`executor.py`): local subprocess, Kaggle job (push -> wait -> fetch, 2 at a time),
   or dry (simulated predictions; tests the whole loop in seconds).
5. **Recovery** (`recovery.py`) on failure, by error class from the log tail:

   | class | action | max retries |
   |---|---|---|
   | transient (network, 5xx, DNS) | same run, backoff 30 s x 2^n | 3 |
   | oom | micro batch / 2, grad accum x 2 (same effective batch) | 2 |
   | time (time probe > cap) | epochs -> 1, then subsample 3000 | 2 |
   | numeric (NaN, overflow, collapse) | lr x 0.5, seed + 1 | 2 |
   | config (KeyError, missing file) | fail: needs a fix, not a retry | 0 |
   | human (uncommitted code, quota, auth, licence) | escalate now | - |
   | unknown | one plain retry | 1 |
   The same class in 3 different experiments escalates as **systemic** [K-LOOP-4].
   Timeouts are never lengthened; runs are made cheaper instead [K-LOOP-5, K-LOOP-11].
6. **Test** (`evaluate.sanity`): valid files, held-out coverage = holdout fold size, accuracy
   >= chance + 5 points, no single-letter collapse, test ids match. Failing runs are
   rejected, never scored [K-LOOP-3, K-LOOP-4].
7. **Evaluate**: accuracy, log-loss, Brier, ECE, per-lie-type accuracy; paired comparison with
   the champion on shared chats.
8. **Improve / promote** (`evaluate.compare`):
   - win: gain >= 1 point AND exact McNemar p <= 0.05 / (candidates this round) [K-LOOP-7]
   - tie+: |gain| <= 0.3 points AND log-loss lower by >= 0.01 AND bootstrap lower bound >= -0.3
   - else kept, not promoted.
   The report prints the minimum detectable gain (2.80 x sqrt(d/n): ~4.4 points at n=400,
   ~2.8 at n=1000 [K-LOOP-8]) so small "gains" are read as noise.
9. **Recombine**: the best run of each branch (up to 5) is combined by majority **vote**,
   **Caruana** selection and **weighted** averaging, each scored honestly (inner CV) and
   promoted by the same rule [K-LOOP-3, K-LOOP-10, K-LLM-9].
10. **Stop** when: patience (3 iterations without a promotion), target accuracy, budget below
    the cheapest run, max iterations, or nothing left to plan. **Escalate** when the holdout
    has been used for 30 comparisons (reusable-holdout risk [K-LOOP-9]), a result looks too
    good (>= 99.5% on >= 300 chats), preflight fails, or recovery says so.

State lives in `data/loop/` (state.json, ledger.jsonl, report.md, proposals.yaml, audit.md,
ESCALATE.md): every run is resumable and every decision is written down with its reason.

## 3. Branches (configs/loop.yaml)

| Branch | Why it is there | Prior | Est. GPU h | Knobs explored |
|---|---|---|---|---|
| qlora_gemma | main model, 97.5% on bench | high | 2.6 | lr, LoRA r, epochs, TTA [K-LLM-14, -10] |
| qlora_qwen | second LLM family = ensemble diversity | high | 3.0 | lr, r [K-LLM-13] |
| deberta | different architecture, cheap partner | med | 1.0 | lr, layer-wise lr decay, TTA |
| line_mil | line-level view generalises OOD | med | 1.5 | pooling, tau, context [K-VER-1, -14] |
| llm_zs | zero-shot, different errors | low | 0.5 | TTA |
| nli_zs | zero-shot NLI per line | low | 0.2 | - |
| tfidf | floor + leak detector | low | 0 | - |

## 4. Competition-day playbook (after the adapter + folds exist)

```bash
liars folds --in data/canonical/comp_train.jsonl --k 5
python scripts/kaggle_run.py data --slug liars-comp data/canonical/comp_train.jsonl \
    data/canonical/comp_test.jsonl data/canonical/folds_comp_train.csv
git commit -am "comp data adapter" && git push       # Kaggle runs use the last commit
liars loop doctor --executor kaggle                   # must print "ready"
liars loop run --executor dry --iters 3 --set loop.ledger_dir=data/loop_dry \
    --set loop.data_dir=data/dry                      # 30 s rehearsal on the real files
/improve-loop kaggle 1                                # first real cycle: 2 baselines (~5.6 GPU h)
```
Then one cycle at a time (each is hours on Kaggle), with a look at the report in between.
Submit the champion with the command the report prints; keep the two final submissions
SAFE (best single run) and BOLD (best ensemble), as in COMPETITION_CHECKLIST.md.

Budget sketch for 18 GPU h: iteration 1 gemma + qwen (5.6 h), 2 deberta + line_mil (2.5 h),
3 llm_zs + nli_zs (0.7 h) and ensembles (CPU), 4-5 mutations of the best two branches
(~2 x 2.6 h each); about 3 h stays as margin for recovery retries.

## 5. Roadmap (build queue, needs your OK per item)

In order of value / cost (details in KNOWLEDGE.md): numeric/time rules for the stacker
[K-VER-12]; QLoRA self-distillation [K-LLM-2]; top-k margin loss for line_mil [K-VER-15];
minimal-pair augmentation from TRAIN chats [K-VER-5/6]; vLLM + AWQ inference [K-LLM-18/19];
uncertainty cascade [K-LLM-12]; MiniCheck line scores as a feature [K-VER-3]; seed
replication of marginal promotions [K-LOOP-12]. Each becomes a new knob or branch, so the
loop can test it like everything else.

## 6. Limits (honest list)

- The dry executor tests the plumbing, not model quality: its "accuracy" is simulated.
- GPU-hour estimates for qlora_qwen and line_mil are not yet timed; the first real runs'
  time probes will correct them (update `est_gpu_h`).
- Holdout-mode branches are compared on the holdout fold only (~20% of train): the minimum
  detectable gain is large; prefer k-fold for cheap branches when n is small.
- One-knob mutation can miss interactions (e.g. LoRA r and alpha together); the planner agent
  can propose combined changes when it has a reason.
