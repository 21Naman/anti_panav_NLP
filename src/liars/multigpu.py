"""Split one prediction job across several GPUs (Kaggle T4 x2): one process per GPU.

The input is cut into N contiguous shards; each shard is predicted by a child process
`liars predict` pinned to one GPU with CUDA_VISIBLE_DEVICES (each child loads its own copy
of the model, so the model must fit on one GPU, as QLoRA 4-bit and DeBERTa do); the shard
parquet files are merged and checked (every input id exactly once, input order kept).
On CPU (debug) the children simply run side by side.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from liars.preds import load_preds, merge_shards, split_shards
from liars.schema import load_jsonl, save_jsonl


def predict_sharded(run: str, in_path: Path, model_dir: Path, n: int, data_dir: Path,
                    overrides: Sequence[str], work_dir: Path,
                    log: Callable[[str], None] = print) -> pd.DataFrame:
    examples = load_jsonl(in_path)
    shards = split_shards(examples, n)
    work = Path(work_dir) / "shards" / f"{run}__{in_path.stem}"
    work.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, shard in enumerate(shards):
        shard_in, shard_out = work / f"in_{i}.jsonl", work / f"out_{i}.parquet"
        shard_out.unlink(missing_ok=True)
        save_jsonl(shard, shard_in)
        cmd = [sys.executable, "-m", "liars.cli", "predict", "--run", run, "--in", str(shard_in),
               "--model-dir", str(model_dir), "--out", str(shard_out), "--data-dir", str(data_dir)]
        for o in overrides:
            cmd += ["--set", o]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i))
        log(f"shard {i}: {len(shard)} chats on GPU {i}")
        procs.append((subprocess.Popen(cmd, env=env, stdout=open(work / f"log_{i}.txt", "w"),
                                       stderr=subprocess.STDOUT), shard_out, i))
    t0 = time.time()
    failed = []
    for p, _, i in procs:
        if p.wait() != 0:
            failed.append(i)
    if failed:
        tails = {i: (work / f"log_{i}.txt").read_text()[-1500:] for i in failed}
        raise RuntimeError(f"shard(s) {failed} failed; log tails: {tails}")
    log(f"all {n} shards done in {time.time() - t0:.0f}s; merging")
    return merge_shards([load_preds(out) for _, out, _ in procs], [ex.id for ex in examples])
