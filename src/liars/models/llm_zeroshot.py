"""Zero-shot open LLM: softmax over the logits of 'A', 'B', 'C', 'D' at the answer position.

`fit` trains nothing: it picks few-shot examples (training folds only) and a temperature
on a small calibration sample of the training folds. Prediction averages over TTA
speaker permutations (augment.tta_n), mapped back to the original speakers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from liars.augment import map_back, tta_variants
from liars.llm import get_lm, prompt_for, score_prompts
from liars.models.base import Model
from liars.preds import make_preds, softmax
from liars.prompt import build_user_text
from liars.schema import SPEAKERS, Example

META_FILE = "model_meta.json"


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))


class LLMZeroShotModel(Model):
    def fit(self, train, val, cfg):
        self.cfg = cfg
        self.lcfg, self.pcfg = cfg["llm"], cfg["prompt"]
        self.tta_n = cfg["augment"]["tta_n"]
        self.h = get_lm(self.lcfg)
        rng = random.Random(cfg["seed"])
        n_shot = self.lcfg["fewshot_n"]
        shots = rng.sample(list(train), min(n_shot, len(train))) if n_shot else []
        self.fewshot = [(build_user_text(ex, self.pcfg), ex.label) for ex in shots]
        shot_ids = {ex.id for ex in shots}
        self.temperature = 1.0
        n_cal = self.lcfg["calib_n"]
        if n_cal:
            pool = [ex for ex in train if ex.id not in shot_ids]
            cal = rng.sample(pool, min(n_cal, len(pool)))
            logits = self._tta_logits(cal)
            y = np.array([SPEAKERS.index(ex.label) for ex in cal])
            self.temperature = min(self.lcfg["temperatures"],
                                   key=lambda t: _nll(self._probs(logits, t), y))
        save_dir = cfg["train"].get("_save_dir")
        if save_dir and self.lcfg.get("save_model", True):
            self.save(Path(save_dir))

    def save(self, path: Path) -> None:
        """Nothing is trained: the temperature + few-shot texts are enough to reload
        (`liars predict`, e.g. split across both T4s)."""
        path.mkdir(parents=True, exist_ok=True)
        meta = {"base_model": self.lcfg["model_id"], "temperature": self.temperature,
                "answer_prefix": self.pcfg.get("answer_prefix", ""),
                "fewshot": [list(x) for x in self.fewshot]}
        (path / META_FILE).write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path: Path, cfg: dict) -> "LLMZeroShotModel":
        m = cls()
        m.cfg, m.lcfg, m.pcfg = cfg, cfg["llm"], cfg["prompt"]
        m.tta_n = cfg["augment"]["tta_n"]
        meta = json.loads((Path(path) / META_FILE).read_text())
        m.h = get_lm(m.lcfg)
        m.fewshot = [tuple(x) for x in meta.get("fewshot", [])]
        m.temperature = meta["temperature"]
        return m

    def _tta_logits(self, examples: Sequence[Example]) -> np.ndarray:
        """(n, tta_n, 4) letter logits in ORIGINAL speaker order."""
        variants = [tta_variants(ex, self.tta_n, self.cfg["seed"]) for ex in examples]
        prompts = [prompt_for(self.h, v, self.pcfg, self.lcfg["max_len"], self.fewshot)
                   for vs in variants for v, _ in vs]
        raw = score_prompts(self.h, prompts, self.lcfg["batch_size"], self.lcfg["progress"])
        raw = raw.reshape(len(examples), self.tta_n, len(SPEAKERS))
        out = np.zeros_like(raw)
        for i, vs in enumerate(variants):
            for j, (_, inverse) in enumerate(vs):
                out[i, j] = map_back(raw[i, j], inverse)
        return out

    @staticmethod
    def _probs(logits: np.ndarray, t: float) -> np.ndarray:
        n, k, s = logits.shape
        return softmax(logits.reshape(n * k, s), t).reshape(n, k, s).mean(axis=1)

    def predict_proba(self, examples):
        logits = self._last_tta_logits = self._tta_logits(examples)  # for the trainer's TTA check
        return make_preds([ex.id for ex in examples], self._probs(logits, self.temperature))

    def report(self) -> str:
        st = self.h.stats
        cpm = 60 * st["n_scored"] / max(st["seconds"], 1e-9) / max(self.tta_n, 1)
        tok = self.h.tokenizer
        letters = {s: tok.convert_ids_to_tokens(i) for s, i in zip(SPEAKERS, self.h.letter_ids)}
        rows = [
            ("model", self.h.model_id), ("device", self.h.device),
            ("4-bit", self.lcfg["load_in_4bit"] and self.h.device == "cuda"),
            ("letter tokens", letters), ("tta_n", self.tta_n), ("few-shot", len(self.fewshot)),
            ("temperature", self.temperature), ("prompts scored", st["n_scored"]),
            ("scoring seconds", round(st["seconds"], 1)), ("chats per minute", round(cpm, 1)),
            ("max GPU memory GiB", round(st["max_mem_gb"], 2)),
            ("non-finite logit rows (fp16 NaN)", st["nan_rows"]),
            ("prompts truncated", st["truncated"]),
        ]
        return "# LLM zero-shot report\n\n| item | value |\n|---|---|\n" + "\n".join(
            f"| {k} | {v} |" for k, v in rows)
