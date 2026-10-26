"""`liars leakcheck`: find label skews and generator quirks before modelling.

Writes data/reports/leakcheck_<name>.md. Everything here looks only at TRAIN data.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from liars.cv import assign_folds
from liars.models.baselines import TfidfLRModel, speaker_view
from liars.preds import accuracy
from liars.schema import SPEAKERS, Example

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
NUM_RE = re.compile(r"\d")
E_TOKEN, NUM_TOKEN = "<E>", "<NUM>"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        return format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v).replace("|", "\\|") for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def evidence_vocab_freq(examples: Sequence[Example]) -> dict[str, float]:
    df = Counter()
    for ex in examples:
        df.update({t for fact in ex.evidence for t in tokens(fact)})
    return {t: c / len(examples) for t, c in df.items()}


def mask_line(text: str, ev_freq: dict[str, float], thr: float) -> str:
    """Numbers -> <NUM>; evidence content words (names, values) -> <E>."""
    out: list[str] = []
    for t in tokens(text):
        if NUM_RE.search(t):
            t = NUM_TOKEN
        elif 0 < ev_freq.get(t, 0) < thr:
            t = E_TOKEN
        if not (out and t == out[-1] and t in (E_TOKEN, NUM_TOKEN)):
            out.append(t)
    return " ".join(out)


def content_tokens(text: str, ev_freq: dict[str, float], thr: float) -> set[str]:
    """Tokens that are not frequent evidence 'function words'."""
    return {t for t in tokens(text) if ev_freq.get(t, 0) < thr}


def _rule_accuracy(examples: Sequence[Example], score_fn, rng: random.Random) -> float:
    hits = 0
    for ex in examples:
        scores = {s: score_fn(ex, s) + rng.random() * 1e-6 for s in SPEAKERS}  # random ties
        hits += max(scores, key=scores.get) == ex.label
    return hits / len(examples)


def run_leakcheck(examples: Sequence[Example], lcfg: dict, model_cfg: dict, seed: int,
                  name: str) -> tuple[str, dict]:
    if any(ex.label is None for ex in examples):
        raise ValueError("leakcheck needs labelled (train) data")
    n = len(examples)
    rng = random.Random(seed)
    thr = lcfg["evidence_function_word_freq"]
    ev_freq = evidence_vocab_freq(examples)
    alerts: list[str] = []
    md = [f"# Leak check: {name}", "", f"{n} labelled chats.", ""]

    # 1. Labels and position --------------------------------------------------
    labels = Counter(ex.label for ex in examples)
    lab = pd.DataFrame({"label": SPEAKERS, "n": [labels.get(s, 0) for s in SPEAKERS]})
    lab["share"] = lab["n"] / n
    ranks = Counter(list(dict.fromkeys(m.speaker for m in ex.messages)).index(ex.label)
                    for ex in examples)
    rk = pd.DataFrame({"liar first-appearance rank": list(range(len(SPEAKERS))),
                       "n": [ranks.get(r, 0) for r in range(len(SPEAKERS))]})
    rk["share"] = rk["n"] / n
    md += ["## 1. Label distribution and position", "", md_table(lab), "", md_table(rk), ""]
    max_label_share = float(lab["share"].max())
    max_rank_share = float(rk["share"].max())
    if max(max_label_share, max_rank_share) > lcfg["skew_alert_share"]:
        alerts.append(f"label/position skew: max label share {max_label_share:.3f}, "
                      f"max first-appearance-rank share {max_rank_share:.3f} -> permutation "
                      "augmentation would hide real signal; consider augment.permute_train=false")

    # 2. One-feature rules ----------------------------------------------------
    views = {(ex.id, s): speaker_view(ex, s) for ex in examples for s in SPEAKERS}
    ev_tokens = {ex.id: {t for f in ex.evidence for t in content_tokens(f, ev_freq, thr)}
                 for ex in examples}

    def overlap(ex, s):
        toks = content_tokens(views[(ex.id, s)]["text"], ev_freq, thr)
        return len(toks & ev_tokens[ex.id]) / max(1, len(toks))

    rules = {
        "most messages": lambda ex, s: views[(ex.id, s)]["n_msgs"],
        "fewest messages": lambda ex, s: -views[(ex.id, s)]["n_msgs"],
        "longest mean line": lambda ex, s: views[(ex.id, s)]["mean_chars"],
        "shortest mean line": lambda ex, s: -views[(ex.id, s)]["mean_chars"],
        "speaks first": lambda ex, s: views[(ex.id, s)]["first"],
        "speaks last": lambda ex, s: views[(ex.id, s)]["last"],
        "highest evidence overlap": overlap,
        "lowest evidence overlap": lambda ex, s: -overlap(ex, s),
    }
    chance = 1 / len(SPEAKERS)
    rule_rows = [(r, _rule_accuracy(examples, fn, rng)) for r, fn in rules.items()]
    rt = pd.DataFrame(rule_rows, columns=["rule (pick the speaker who ...)", "accuracy"])
    liar_vs = pd.DataFrame([
        ("messages per speaker",
         np.mean([views[(ex.id, ex.label)]["n_msgs"] for ex in examples]),
         np.mean([views[(ex.id, s)]["n_msgs"] for ex in examples for s in SPEAKERS if s != ex.label])),
        ("mean line chars",
         np.mean([views[(ex.id, ex.label)]["mean_chars"] for ex in examples]),
         np.mean([views[(ex.id, s)]["mean_chars"] for ex in examples for s in SPEAKERS if s != ex.label])),
        ("evidence overlap",
         np.mean([overlap(ex, ex.label) for ex in examples]),
         np.mean([overlap(ex, s) for ex in examples for s in SPEAKERS if s != ex.label])),
    ], columns=["feature", "liar", "honest"])
    md += ["## 2. One-feature rules (chance = 0.25 on balanced data)", "", md_table(rt), "",
           md_table(liar_vs), ""]
    for r, acc in rule_rows:
        if acc > chance + lcfg["rule_alert_margin"]:
            alerts.append(f"rule '{r}' gets {acc:.3f} (chance {chance:.2f})")

    # 3. Over-represented n-grams in liar-speaker lines -----------------------
    lines, y = [], []
    for ex in examples:
        for m in ex.messages:
            lines.append(m.text)
            y.append(int(m.speaker == ex.label))
    y = np.array(y)
    base_share = y.mean()
    cv = CountVectorizer(ngram_range=tuple(lcfg["ngram_range"]), min_df=lcfg["ngram_min_df"],
                         lowercase=True, binary=True)
    X = cv.fit_transform(lines)
    scores, _ = chi2(X, y)
    liar_c = np.asarray(X[y == 1].sum(axis=0)).ravel()
    tot_c = np.asarray(X.sum(axis=0)).ravel()
    share = liar_c / np.maximum(tot_c, 1)
    names = cv.get_feature_names_out()
    over = np.where(share > base_share)[0]
    top = over[np.argsort(-np.nan_to_num(scores[over]))][: lcfg["top_n"]]
    ng = pd.DataFrame({"ngram": names[top], "chi2": scores[top], "in liar lines": liar_c[top],
                       "total": tot_c[top], "liar share": share[top]})
    md += [f"## 3. N-grams over-represented in liar-speaker lines (base share {base_share:.3f})",
           "", md_table(ng), ""]

    # 4. Templates (numbers and evidence content words masked) ----------------
    tmpl_total, tmpl_liar = Counter(), Counter()
    for ex in examples:
        for m in ex.messages:
            t = mask_line(m.text, ev_freq, thr)
            tmpl_total[t] += 1
            tmpl_liar[t] += int(m.speaker == ex.label)
    tt = pd.DataFrame([(t, c, tmpl_liar[t] / c) for t, c in tmpl_total.most_common(lcfg["top_n"])],
                      columns=["template", "count", "liar share"])
    flagged = [(t, c, tmpl_liar[t] / c) for t, c in tmpl_total.items()
               if c >= lcfg["template_min_count"] and tmpl_liar[t] / c >= lcfg["liar_only_share"]]
    flagged.sort(key=lambda r: -r[1])
    md += ["## 4. Most frequent templates", "", md_table(tt), "",
           f"### Templates used (almost) only by liars (count >= {lcfg['template_min_count']}, "
           f"liar share >= {lcfg['liar_only_share']})", ""]
    if flagged:
        md.append(md_table(pd.DataFrame(flagged, columns=["template", "count", "liar share"])))
        alerts.append(f"{len(flagged)} template(s) used only by liars, e.g. {flagged[0][0]!r}")
    else:
        md.append("None found.")
    md.append("")

    # 5. TF-IDF baseline OOF (speaker text + position/count features only) ----
    folds = assign_folds(examples, lcfg["k"], seed)
    oof_hits = 0
    for f in sorted(set(folds.values())):
        tr = [ex for ex in examples if folds[ex.id] != f]
        va = [ex for ex in examples if folds[ex.id] == f]
        m = TfidfLRModel()
        m.fit(tr, None, model_cfg)
        oof_hits += accuracy(m.predict_proba(va), pd.Series([ex.label for ex in va])) * len(va)
    tfidf_acc = oof_hits / n
    full = TfidfLRModel()
    full.fit(examples, None, model_cfg)
    md += [f"## 5. TF-IDF baseline: {lcfg['k']}-fold OOF accuracy {tfidf_acc:.4f}", "",
           "Top features by |coefficient| (positive = pushes towards 'liar'):", "",
           md_table(full.top_features(lcfg["top_n"])), ""]
    if tfidf_acc > lcfg["tfidf_alert_acc"]:
        alerts.append(f"TF-IDF baseline OOF accuracy {tfidf_acc:.3f} > {lcfg['tfidf_alert_acc']}")

    md[3:3] = ["## Alerts", ""] + ([f"- {a}" for a in alerts] or ["- none"]) + [""]
    summary = {"n": n, "alerts": alerts, "tfidf_oof_acc": tfidf_acc,
               "rules": dict(rule_rows), "liar_only_templates": len(flagged),
               "max_label_share": max_label_share, "max_rank_share": max_rank_share}
    return "\n".join(md), summary
