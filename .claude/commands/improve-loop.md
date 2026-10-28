---
description: Run the plan -> act -> test -> evaluate -> improve cycle with the planner and auditor agents
argument-hint: "[dry|local|kaggle] [cycles]"
---

Run the Liar's Chatroom improvement cycle. Executor: `$1` (default `dry`). Cycles: `$2`
(default 1). One cycle:

1. **Doctor.** `liars loop doctor --executor <executor>`. On any ERROR stop and tell me what
   to fix. Show only the ERROR/WARN lines.
2. **Status.** `liars loop status | head -20`. If the status is `escalated`, show
   `data/loop/ESCALATE.md`, stop, and ask me. Never resume an escalated loop yourself.
3. **Plan.** Use the `loop-planner` agent (it writes data/loop/proposals.yaml).
4. **Budget check (kaggle only).** Say how many GPU hours the next iteration may use (proposals'
   est_gpu_h, else branch estimates, max 2 jobs) and the hours left. If it is over 4 h, or the
   loop would go below 2 h left, ask me first.
5. **Act / test / evaluate / improve.** `liars loop run --executor <executor> --iters 1 > data/loop/cycle.log 2>&1`
   and show only the last 25 lines. (Kaggle runs take hours: run it in the background and
   check back; never poll faster than every 10 minutes.)
6. **Audit.** Use the `loop-auditor` agent. If it halts the loop, stop and show me why.
7. **Report.** Five lines: champion (accuracy, n), what was promoted or rejected and why, GPU
   hours left, and the next step.

Repeat for the requested number of cycles, stopping early on `stopped` or `escalated`.

Rules: never edit configs/loop.yaml, src/liars/loop/ or results files during a cycle; never
upload a submission (I do that: the report prints the command); never read test chats.
