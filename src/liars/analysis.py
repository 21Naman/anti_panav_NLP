"""`liars analyze`: plain-HTML error analysis of a run's held-out (OOF) predictions.

Sections: accuracy overall / per fold / per lie_type / per chat length / per true label,
A-D confusion matrix, calibration by confidence, the most confident wrong examples
(true liar's lines red, predicted liar's lines blue), and with --compare the examples
where two runs disagree. Only TRAIN (held-out) predictions are shown, never test chats.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from liars.preds import PROB_COLS, argmax_letter
from liars.schema import SPEAKERS, Example

CSS = """body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
table{border-collapse:collapse;margin:8px 0 20px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}
th{background:#f3f3f3}td:first-child,th:first-child{text-align:left}.ex{border:1px solid #ddd;padding:10px;margin:12px 0}
.true{color:#c62828;font-weight:600}.pred{color:#1565c0;font-weight:600}.both{color:#6a1b9a}
.meta{color:#666;font-size:90%}ol{margin:4px 0}"""


def _table(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or "", *df.columns])
    body = ""
    is_float = {c: df[c].dtype.kind == "f" for c in df.columns}  # per column: rows mix dtypes
    for idx in df.index:
        cells = "".join(f"<td>{fmt.format(df.at[idx, c]) if is_float[c] else html.escape(str(df.at[idx, c]))}</td>"
                        for c in df.columns)
        body += f"<tr><td>{html.escape(str(idx))}</td>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def _acc_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)["correct"].agg(n="size", accuracy="mean")
    g.index.name = col
    return g


def _example_html(ex: Example, row: pd.Series, pred: str, extra: str = "") -> str:
    probs = " ".join(f"{s}={row[f'p_{s}']:.2f}" for s in SPEAKERS)
    lines = []
    for m in ex.messages:
        cls = ("both" if m.speaker == ex.label == pred else "true" if m.speaker == ex.label
               else "pred" if m.speaker == pred else "")
        lines.append(f'<div class="{cls}">{m.speaker}: {html.escape(m.text)}</div>')
    facts = "".join(f"<li>{html.escape(f)}</li>" for f in ex.evidence)
    return (f'<div class="ex"><div class="meta">{html.escape(ex.id)} | true <span class="true">'
            f'{ex.label}</span> | predicted <span class="pred">{pred}</span> | {probs} | '
            f'lie_type {ex.meta.get("lie_type", "?")} {extra}</div><b>Evidence</b><ol>{facts}</ol>'
            f'<b>Chat</b>{"".join(lines)}</div>')


def build_report(run: str, oof: pd.DataFrame, examples: Sequence[Example], acfg: dict, n_wrong: int,
                 compare: tuple[str, pd.DataFrame] | None = None) -> str:
    by_id = {ex.id: ex for ex in examples}
    oof = oof[oof["id"].isin(by_id)].copy()
    oof["pred"] = argmax_letter(oof).to_numpy()
    oof["label"] = [by_id[i].label for i in oof["id"]]
    oof["correct"] = oof["pred"] == oof["label"]
    oof["confidence"] = oof[PROB_COLS].max(axis=1)
    oof["lie_type"] = [by_id[i].meta.get("lie_type", "?") for i in oof["id"]]
    n_msgs = [len(by_id[i].messages) for i in oof["id"]]
    oof["chat_length"] = pd.cut(n_msgs, bins=acfg["length_bins"]).astype(str)
    oof["conf_bucket"] = pd.cut(oof["confidence"], bins=acfg["confidence_bins"]).astype(str)

    parts = [f"<h1>Error analysis: {html.escape(run)}</h1>",
             f"<p>{len(oof)} held-out chats, accuracy <b>{oof['correct'].mean():.4f}</b>.</p>"]
    for col, title in [("fold", "Per fold"), ("lie_type", "Per lie type"),
                       ("chat_length", "Per chat length (messages)"),
                       ("label", "Per true label (position bias check)")]:
        if col in oof.columns:
            parts += [f"<h2>{title}</h2>", _table(_acc_by(oof, col))]
    cm = pd.crosstab(oof["label"], oof["pred"]).reindex(index=list(SPEAKERS), columns=list(SPEAKERS), fill_value=0)
    cm.index.name = "true \\ pred"
    parts += ["<h2>Confusion matrix</h2>", _table(cm, "{}")]
    cal = oof.groupby("conf_bucket", observed=True).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                                        mean_confidence=("confidence", "mean"))
    parts += ["<h2>Calibration (accuracy by confidence)</h2>", _table(cal)]

    wrong = oof[~oof["correct"]].sort_values("confidence", ascending=False).head(n_wrong)
    parts.append(f"<h2>{len(wrong)} most confident mistakes</h2><p class='meta'>"
                 "<span class='true'>red</span> = true liar's lines, <span class='pred'>blue</span> = "
                 "predicted liar's lines.</p>")
    parts += [_example_html(by_id[r["id"]], r, r["pred"]) for _, r in wrong.iterrows()]

    if compare:
        name2, oof2 = compare
        o2 = oof2.set_index("id")
        both = oof[oof["id"].isin(o2.index)].copy()
        both["pred2"] = argmax_letter(o2.loc[both["id"]]).to_numpy()
        dis = both[both["pred"] != both["pred2"]]
        win1 = int((dis["pred"] == dis["label"]).sum())
        win2 = int((dis["pred2"] == dis["label"]).sum())
        parts.append(f"<h2>Disagreements with {html.escape(name2)}</h2><p>{len(both)} shared chats, "
                     f"{len(dis)} disagreements: {html.escape(run)} right {win1}, "
                     f"{html.escape(name2)} right {win2}, neither {len(dis) - win1 - win2}.</p>")
        for _, r in dis.head(n_wrong).iterrows():
            parts.append(_example_html(by_id[r["id"]], r, r["pred"], f"| {html.escape(name2)} predicted {r['pred2']}"))
    return f"<!doctype html><meta charset='utf-8'><title>{html.escape(run)}</title><style>{CSS}</style>" + "".join(parts)


def write_report(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
