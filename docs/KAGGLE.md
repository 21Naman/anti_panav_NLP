# Running `liars` on Kaggle GPUs

Short version: `python scripts/kaggle_run.py push --name NAME --cmd "liars ..." --wait`,
then `python scripts/kaggle_run.py fetch --name NAME`. Everything below explains the
pieces and the manual (browser) fallback.

Account: **panavrg** (solo, one weekly GPU quota of about 30 h). Run jobs one after
another, by priority: QLoRA main run, then LLM zero-shot, then DeBERTa.

---

## 0. One-time setup (already done on the laptop)

- `~/.kaggle/kaggle.json` exists (Kaggle → Settings → API → Create New Token).
- Kaggle CLI is in the venv: `.venv/bin/pip install kaggle` (dev tool, NOT a package dependency).
- Kaggle Secrets on the account (Kaggle → Settings → Secrets): `GH_TOKEN` (read-only on
  the private repo) and `HF_TOKEN` (Hugging Face read). Never paste tokens anywhere else.
- The Kaggle account must be phone-verified, or GPU and Internet stay greyed out.

## 1. Terminal flow (preferred)

Run from the repo root with the venv active (`source .venv/bin/activate`).

**a. Data → private dataset.** Uploads files under `data/` and a `manifest.json` that
records their paths, so the notebook rebuilds the same `data/` tree. Include the folds
file so every run uses identical folds.
```bash
python scripts/kaggle_run.py data data/bench/train.jsonl data/bench/test.jsonl data/canonical/folds_train.csv
# competition day (new dataset name):
python scripts/kaggle_run.py data --slug liars-comp data/canonical/comp_train.jsonl \
    data/canonical/comp_test.jsonl data/canonical/folds_comp_train.csv
```
Never upload hidden label files (e.g. `data/bench/test_labels.csv`). Re-running the
command creates a new version of the dataset.

**b. Code.** The notebook gets the code one of two ways:
- `--code auto` (default): `git clone` with the `GH_TOKEN` secret at the exact commit
  you pushed. Needs `git push` first, and the secret must be attached to the notebook.
- `--code dataset`: the private `liars-code` dataset (a snapshot of the last COMMITTED
  code). Refresh it after every commit you want to run:
  ```bash
  python scripts/kaggle_run.py code
  ```
  The notebook prints `code source: dataset liars-code @ <commit>`; check it.

**c. Push and wait.** One kernel per `--name` (`panavrg/liars-<name>`). Re-pushing the
same name makes a new version. Several `--cmd` run in order.
```bash
python scripts/kaggle_run.py push --name nli-bench --code dataset \
  --cmd "liars train --config configs/nli_zeroshot.yaml --test data/bench/test.jsonl --set nli.progress=false" --wait
python scripts/kaggle_run.py status --name nli-bench     # any time
```
Defaults: accelerator `NvidiaTeslaT4` (T4 x2), Internet on, private. The kernel keeps
running if the laptop sleeps.

**d. Fetch and merge.**
```bash
python scripts/kaggle_run.py fetch --name nli-bench
```
This downloads to `data/kaggle/<name>/` (outputs.zip + the kernel log) and merges into
`data/`: new oof/preds/reports/runs files are copied (existing files never overwritten),
and `runs.csv` rows are appended (deduplicated by run name). Then score locally:
```bash
liars score --preds data/preds/<run>.parquet --labels data/bench/test_labels.csv
liars score --preds data/oof/<run>.parquet --labels data/bench/train.jsonl
```
GPU time per command is in `data/logs/kaggle_timing.json` (overwritten by each fetch;
the copy in `data/kaggle/<name>/` stays). Record it in PROGRESS.md.

## 2. Manual flow (browser), if the terminal flow fails

1. Upload data: Kaggle → Datasets → New Dataset → drop the files → **Private**. Add a
   `manifest.json` like `{"bench__train.jsonl": "bench/train.jsonl"}` (file name in the
   dataset → path under `data/`), or simply use step 1a from the laptop.
2. Kaggle → Code → New Notebook → File → Import Notebook → `notebooks/kaggle/run_cli.ipynb`.
3. Right panel: **Accelerator: GPU T4 x2**, **Internet: on**, Persistence: off.
   Add Input: your data dataset (and `liars-code` if you use the dataset fallback).
4. Add-ons → Secrets → tick **GH_TOKEN** and **HF_TOKEN** for this notebook.
5. Edit the settings cell: `REF` (commit or `main`), `CODE_MODE`, `DATA_DATASETS`, and
   the `COMMANDS` block (one `liars ...` command per line).
6. **Save Version → Save & Run All (Commit)**. This runs in the background; you can close
   the browser. Sessions stop after about 12 h, so size runs to finish well before that.
7. When done: open the version → Output → download `outputs.zip`, then on the laptop:
   ```bash
   python scripts/kaggle_run.py merge --zip ~/Downloads/outputs.zip
   ```

## 3. What the notebook does (thin, no logic)

1. Settings cell (REPO, REF, CODE_MODE, datasets, COMMANDS).
2. Install: secrets → git clone into `/tmp/liars` (not saved as output; the token is removed
   from the git remote and never printed) or the code dataset → `pip install -e .[nlp]`.
3. Copy data datasets into `/tmp/liars/data/` following each `manifest.json`.
4. `python scripts/gpu_info.py`: GPU, memory, library versions, 4-bit load check.
5. Run each command; wall time per command → `data/logs/kaggle_timing.json`.
6. Zip `data/oof`, `data/preds`, `data/runs.csv`, `data/reports`, `data/runs`, `data/logs`
   into `/kaggle/working/outputs.zip`.

Regenerate the committed template after changing the notebook code:
`python scripts/kaggle_run.py template` (a test checks they match).

## 4. Troubleshooting at 3 AM

| Symptom | Fix |
|---|---|
| `HEAD is not pushed to origin` | `git push`, or `kaggle_run.py code` and push with `--code dataset` |
| Log says `Kaggle secret GH_TOKEN: not available` | Secret not attached to this kernel: attach it in the notebook UI (Add-ons → Secrets) once, or use `--code dataset` |
| `dataset 'x' is not attached` | wrong `--datasets` name, or the dataset is still processing (`kaggle datasets status panavrg/x`) |
| `no outputs.zip` on fetch | the run failed: read the `.log` in `data/kaggle/<name>/` |
| Status `error` right after push | usually a bad metadata/dataset name; open the kernel URL printed by push |
| GPU / Internet greyed out | phone-verify the account; check the weekly GPU quota (notebook editor, right panel) |
| Out of GPU memory | lower `batch_size` / `max_len` in the config via `--set` |
| Code on Kaggle is old | `--code dataset` uses the last `kaggle_run.py code` upload; re-run it |

## 5. Findings from the first run (2026-10-27, `liars-nli-bench`)

- **Kaggle Secrets are NOT attached to API-pushed notebooks.** The log said
  `Kaggle secret GH_TOKEN: not available (ConnectionError)` (same for HF_TOKEN). So the
  terminal flow uses `--code dataset` (works, tested). To use secrets instead, open the
  kernel page once (Edit → Add-ons → Secrets → tick both) — untested whether later pushed
  versions keep the attachment. Public Hugging Face models download fine without HF_TOKEN.
- Kaggle image: Python 3.13, torch 2.11.0+cu128, transformers 5.16.1, bitsandbytes 0.50.2,
  2 x Tesla T4 (14.6 GiB, compute 7.5 → no bf16, use fp16). 4-bit NF4 load check OK.
- `pip install -e .[nlp]` on Kaggle only adds what is missing (does not reinstall torch).
- NLI large on bench train (2000, kfold, all 3 modes) + test (1000): 551 s command,
  ~10 min kernel wall time.
