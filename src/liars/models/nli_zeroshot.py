"""Zero-shot NLI: how strongly does each speaker's text contradict the evidence?

Modes (each gives one score per speaker, then softmax with a temperature):
  pairwise     premise = one fact,      hypothesis = one message; max over facts x messages
  joint        premise = all evidence,  hypothesis = one message;  max over messages
  joint_concat premise = all evidence,  hypothesis = all of the speaker's messages joined
`fit` trains nothing: it picks the best mode and temperature on the training folds.
Pair scores are cached on disk (data/cache/), keyed by model id + text hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from liars.device import get_device
from liars.models.base import Model
from liars.paths import get_paths
from liars.preds import make_preds, softmax
from liars.schema import SPEAKERS, Example

MODES = ("pairwise", "joint", "joint_concat")
NLI_LABELS = ("contradiction", "entailment", "neutral")
KEY_SEP = "\x1f"


def _key(premise: str, hypothesis: str) -> str:
    return hashlib.sha1(f"{premise}{KEY_SEP}{hypothesis}".encode("utf-8")).hexdigest()


class PairCache:
    """premise/hypothesis -> (contradiction, entailment, neutral) probabilities."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, np.ndarray] = {}
        self.dirty = False
        if path.exists():
            df = pd.read_parquet(path)
            self.data = dict(zip(df["key"], df[list(NLI_LABELS)].to_numpy()))

    def save(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.data)
        df = pd.DataFrame(np.stack([self.data[k] for k in keys]), columns=list(NLI_LABELS))
        df.insert(0, "key", keys)
        df.to_parquet(self.path, index=False)
        self.dirty = False


def label_indices(id2label: dict, model_id: str = "") -> list[int]:
    """Output index of contradiction / entailment / neutral, read from the checkpoint's
    id2label (never hardcoded: different NLI checkpoints order them differently)."""
    names = {int(i): str(l).lower() for i, l in id2label.items()}
    out = []
    for name in NLI_LABELS:
        hits = [i for i, l in names.items() if l.startswith(name[:6])]
        if len(hits) != 1:
            raise ValueError(f"{model_id}: cannot find '{name}' in id2label {names}")
        out.append(hits[0])
    return out


class NLIScorer:
    def __init__(self, ncfg: dict, cache_dir: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = get_device(ncfg["device"])
        self.fp16 = bool(ncfg["fp16"]) and self.device == "cuda"
        self.batch_size = ncfg["batch_size"]
        self.max_length = ncfg["max_length"]
        model_id = ncfg["model_id"]
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.float16 if self.fp16 else torch.float32
        ).to(self.device).eval()
        self.label_idx = label_indices(self.model.config.id2label, model_id)
        tag = hashlib.sha1(model_id.encode()).hexdigest()[:10]
        self.cache = PairCache(cache_dir / f"nli_{tag}.parquet")
        self.n_computed = 0

    def score(self, pairs: Sequence[tuple[str, str]], progress: bool = False) -> np.ndarray:
        """(n, 3) probabilities in NLI_LABELS order."""
        keys = [_key(p, h) for p, h in pairs]
        todo = sorted({k: (p, h) for k, (p, h) in zip(keys, pairs) if k not in self.cache.data}.items(),
                      key=lambda kv: len(kv[1][0]) + len(kv[1][1]))
        if todo:
            from tqdm import tqdm

            torch = self.torch
            batches = range(0, len(todo), self.batch_size)
            for b in tqdm(batches, disable=not progress, desc="nli pairs"):
                chunk = todo[b : b + self.batch_size]
                enc = self.tok([p for _, (p, _) in chunk], [h for _, (_, h) in chunk],
                               truncation=True, max_length=self.max_length, padding=True,
                               return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    logits = self.model(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[:, self.label_idx]
                for (k, _), p in zip(chunk, probs):
                    self.cache.data[k] = p
            self.cache.dirty = True
            self.n_computed += len(todo)
            self.cache.save()
        if not keys:
            return np.zeros((0, len(NLI_LABELS)))
        return np.stack([self.cache.data[k] for k in keys])


_SCORERS: dict[tuple, NLIScorer] = {}


def get_scorer(ncfg: dict, cache_dir: Path) -> NLIScorer:
    key = (ncfg["model_id"], ncfg["device"], ncfg["fp16"], str(cache_dir))
    if key not in _SCORERS:
        _SCORERS[key] = NLIScorer(ncfg, cache_dir)
    return _SCORERS[key]


def keep_message(text: str, fcfg: dict) -> bool:
    t = text.strip()
    if fcfg["skip_questions"] and t.endswith("?"):
        return False
    return len(t.split()) >= fcfg["min_words"]


def evidence_text(ex: Example, sep: str) -> str:
    return sep.join(ex.evidence)


def speaker_scores(examples: Sequence[Example], mode: str, scorer: NLIScorer, ncfg: dict,
                   progress: bool = False) -> np.ndarray:
    """(n_examples, 4) max contradiction probability per speaker (0 if nothing scored)."""
    fcfg, sep = ncfg["filter"], ncfg["evidence_sep"]
    pairs: list[tuple[str, str]] = []
    owner: list[tuple[int, int]] = []  # (example index, speaker index)
    for i, ex in enumerate(examples):
        ev = evidence_text(ex, sep)
        for j, s in enumerate(SPEAKERS):
            lines = [m.text for m in ex.messages if m.speaker == s and keep_message(m.text, fcfg)]
            if not lines:
                continue
            if mode == "pairwise":
                new = [(f, t) for f in ex.evidence for t in lines]
            elif mode == "joint":
                new = [(ev, t) for t in lines]
            elif mode == "joint_concat":
                new = [(ev, ncfg["message_sep"].join(lines))]
            else:
                raise ValueError(f"unknown NLI mode {mode!r}; use one of {MODES}")
            pairs += new
            owner += [(i, j)] * len(new)
    scores = np.zeros((len(examples), len(SPEAKERS)))
    if pairs:
        contra = scorer.score(pairs, progress)[:, NLI_LABELS.index("contradiction")]
        for (i, j), c in zip(owner, contra):
            scores[i, j] = max(scores[i, j], c)
    return scores


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))


class NLIZeroShotModel(Model):
    def fit(self, train, val, cfg):
        ncfg = cfg["nli"]
        self.ncfg = ncfg
        self.scorer = get_scorer(ncfg, get_paths(cfg).cache)
        y = np.array([SPEAKERS.index(ex.label) for ex in train])
        modes = list(ncfg["modes"]) if ncfg["mode"] == "auto" else [ncfg["mode"]]
        rows, per_mode = [], {}
        for mode in modes:
            s = speaker_scores(train, mode, self.scorer, ncfg, progress=ncfg["progress"])
            pred = s.argmax(axis=1)
            per_mode[mode] = (float((pred == y).mean()), s)
            rows.append(pd.DataFrame({
                "mode": mode, "correct": pred == y,
                "lie_type": [ex.meta.get("lie_type", "?") for ex in train],
                "difficulty": [ex.meta.get("difficulty", "?") for ex in train],
            }))
        self.mode = max(modes, key=lambda m: per_mode[m][0])  # ties -> first in config order
        s = per_mode[self.mode][1]
        self.temperature = min(ncfg["temperatures"], key=lambda t: _nll(softmax(s, t), y))
        self.table = pd.concat(rows, ignore_index=True)
        self.per_mode_acc = {m: a for m, (a, _) in per_mode.items()}

    def predict_proba(self, examples):
        s = speaker_scores(examples, self.mode, self.scorer, self.ncfg, progress=self.ncfg["progress"])
        return make_preds([ex.id for ex in examples], softmax(s, self.temperature))

    def report(self) -> str:
        """Per-mode accuracy (overall, per lie_type, per difficulty) on the fit data."""
        t = self.table
        lines = [f"# NLI zero-shot fit report ({self.ncfg['model_id']})", "",
                 f"chosen mode: **{self.mode}**, temperature {self.temperature}", "",
                 "| mode | accuracy |", "|---|---|"]
        lines += [f"| {m} | {a:.4f} |" for m, a in self.per_mode_acc.items()]
        for col in ("lie_type", "difficulty"):
            pv = t.pivot_table(index=col, columns="mode", values="correct", aggfunc="mean")
            n = t[t["mode"] == t["mode"].iloc[0]].groupby(col).size()
            lines += ["", f"## accuracy by {col}", "",
                      f"| {col} | n | " + " | ".join(str(c) for c in pv.columns) + " |",
                      "|---" * (len(pv.columns) + 2) + "|"]
            for idx, r in pv.iterrows():
                lines.append(f"| {idx} | {int(n[idx])} | " + " | ".join(f"{v:.3f}" for v in r.values) + " |")
        return "\n".join(lines)
