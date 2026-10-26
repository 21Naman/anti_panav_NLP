"""Baselines: majority class, and TF-IDF + logistic regression as a 4-way choice.

The TF-IDF model is mainly a leak detector: it sees only each speaker's own text plus
position / count features, never the evidence. Well above 25% on balanced data means
the surface text gives the liar away.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from liars.models.base import Model
from liars.preds import make_preds, softmax
from liars.schema import SPEAKERS, Example

NOT_SPEAKING_RANK = len(SPEAKERS)  # first-appearance rank for a speaker with no lines


class MajorityModel(Model):
    """Predicts the training label distribution for every chat."""

    def fit(self, train, val, cfg):
        counts = Counter(ex.label for ex in train)
        total = sum(counts.values())
        self.prior = np.array([counts.get(s, 0) / total for s in SPEAKERS])

    def predict_proba(self, examples):
        probs = np.tile(self.prior, (len(examples), 1))
        return make_preds([ex.id for ex in examples], probs)


def speaker_view(ex: Example, speaker: str) -> dict:
    lines = [m.text for m in ex.messages if m.speaker == speaker]
    order = list(dict.fromkeys(m.speaker for m in ex.messages))
    return {
        "text": " \n ".join(lines),
        "n_msgs": len(lines),
        "share": len(lines) / max(1, len(ex.messages)),
        "mean_chars": float(np.mean([len(t) for t in lines])) if lines else 0.0,
        "rank": order.index(speaker) if speaker in order else NOT_SPEAKING_RANK,
        "first": float(bool(ex.messages) and ex.messages[0].speaker == speaker),
        "last": float(bool(ex.messages) and ex.messages[-1].speaker == speaker),
    }


class TfidfLRModel(Model):
    def fit(self, train, val, cfg):
        mcfg = cfg["model"]
        self.feat_cfg = mcfg["features"]
        t = mcfg["tfidf"]
        self.vec = TfidfVectorizer(
            ngram_range=tuple(t["ngram_range"]), min_df=t["min_df"],
            max_features=t["max_features"], sublinear_tf=t["sublinear_tf"],
            lowercase=t["lowercase"],
        )
        rows, y = self._rows(train), []
        for ex in train:
            y += [int(s == ex.label) for s in SPEAKERS]
        X_text = self.vec.fit_transform([r["text"] for r in rows])
        num = self._numeric(rows)
        self.scaler = StandardScaler().fit(num) if num.shape[1] else None
        X = self._stack(X_text, num)
        lr = mcfg["lr"]
        self.clf = LogisticRegression(C=lr["C"], max_iter=lr["max_iter"])
        self.clf.fit(X, y)

    def _rows(self, examples: Sequence[Example]) -> list[dict]:
        return [speaker_view(ex, s) | {"letter": s} for ex in examples for s in SPEAKERS]

    def _numeric_names(self) -> list[str]:
        names = []
        if self.feat_cfg["counts"]:
            names += ["n_msgs", "share", "mean_chars"]
        if self.feat_cfg["position"]:
            names += [f"rank={r}" for r in range(NOT_SPEAKING_RANK + 1)] + ["first", "last"]
        if self.feat_cfg["letter"]:
            names += [f"letter={s}" for s in SPEAKERS]
        return names

    def _numeric(self, rows: list[dict]) -> np.ndarray:
        cols = []
        for r in rows:
            v = []
            if self.feat_cfg["counts"]:
                v += [r["n_msgs"], r["share"], r["mean_chars"]]
            if self.feat_cfg["position"]:
                v += [float(r["rank"] == k) for k in range(NOT_SPEAKING_RANK + 1)]
                v += [r["first"], r["last"]]
            if self.feat_cfg["letter"]:
                v += [float(r["letter"] == s) for s in SPEAKERS]
            cols.append(v)
        return np.asarray(cols, dtype=float).reshape(len(rows), -1)

    def _stack(self, X_text, num):
        if self.scaler is None:
            return X_text
        return sparse.hstack([X_text, sparse.csr_matrix(self.scaler.transform(num))]).tocsr()

    def predict_proba(self, examples):
        rows = self._rows(examples)
        X = self._stack(self.vec.transform([r["text"] for r in rows]), self._numeric(rows))
        scores = self.clf.decision_function(X).reshape(len(examples), len(SPEAKERS))
        return make_preds([ex.id for ex in examples], softmax(scores))

    def top_features(self, n: int) -> pd.DataFrame:
        names = list(self.vec.get_feature_names_out()) + self._numeric_names()
        coef = self.clf.coef_[0]
        order = np.argsort(-np.abs(coef))[:n]
        return pd.DataFrame({"feature": [names[i] for i in order], "coef": coef[order]})
