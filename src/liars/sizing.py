"""Competition-day GPU sizing: `liars plan --train N --test M --chars P50 [--chars-p90 P90]`.

Turns train/test size and chat length (from `liars stats`: "total chars" p50/p90) into
GPU-hour estimates per model, using speeds measured on a Kaggle T4 (configs/sizing.yaml),
and applies the decision rules from the runbook (BUILD_MAP.md -> Launch GPU jobs):
QLoRA 2 epochs -> 1 epoch -> subsample -> cut max_len -> smaller fallback model;
LLMs use TTA 1 and split test prediction over both T4s; DeBERTa (holdout, 1 fit) and the
zero-shot safety run only if they fit their caps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Plan:
    name: str
    hours: float
    detail: str
    sets: list[str]


def tokens_from_chars(chars: float, s: dict) -> float:
    return s["token_overhead"] + s["tokens_per_char"] * chars


def _n_train(n: int, s: dict) -> float:
    return n * (1 - s["holdout_share"]) * (1 - s["inner_val_share"])


def qlora_train_hours(m: dict, n: int, tokens: float, epochs: int, s: dict,
                      n_sub: int | None = None) -> float:
    """The training job: setup + training + inner-val evals + holdout prediction (one GPU)."""
    scale = tokens / s["ref_tokens"]
    n_tr = min(_n_train(n, s), n_sub) if n_sub is not None else _n_train(n, s)
    n_val = n * (1 - s["holdout_share"]) * s["inner_val_share"]
    ppm = m["pred_chats_per_min"]
    train = n_tr * m["train_s_per_chat"] * scale * epochs / 3600
    evals = s["evals_per_epoch"] * epochs * n_val / ppm * scale / 60
    holdout = n * s["holdout_share"] / ppm * scale / 60
    return s["setup_hours"] + train + evals + holdout


def pred_hours(m: dict, n_test: int, tokens: float, s: dict, gpus: int) -> float:
    """Test prediction (`liars predict --gpus N`), without setup."""
    return n_test / m["pred_chats_per_min"] * tokens / s["ref_tokens"] / 60 / gpus


def qlora_hours(m: dict, n: int, n_test: int, tokens: float, epochs: int, s: dict,
                gpus: int, n_sub: int | None = None) -> float:
    """Training job + test prediction in the same session."""
    return qlora_train_hours(m, n, tokens, epochs, s, n_sub) + pred_hours(m, n_test, tokens, s, gpus)


def max_subsample(m: dict, n: int, tokens: float, epochs: int, s: dict, budget: float) -> int:
    """Largest training subset that keeps the TRAINING job within budget (0 if none)."""
    fixed = qlora_train_hours(m, n, tokens, epochs, s, n_sub=0)
    per_chat = m["train_s_per_chat"] * tokens / s["ref_tokens"] * epochs / 3600
    return max(0, int((budget - fixed) / per_chat))


def zeroshot_hours(m: dict, n: int, n_test: int, tokens: float, s: dict, gpus: int) -> float:
    scale = tokens / s["ref_tokens"]
    ppm = m["pred_chats_per_min"]
    return s["setup_hours"] + (m["calib_chats"] + n * s["holdout_share"]) / ppm * scale / 60 \
        + n_test / ppm * scale / 60 / gpus


def deberta_hours(m: dict, n: int, n_test: int, tokens: float, epochs: int, tta: int, s: dict) -> float:
    """Holdout mode, one fit (the ensemble aligns on the holdout fold anyway)."""
    scale = tokens / s["ref_tokens"]  # DeBERTa's ~225 tokens/choice were measured on the same chats
    train = _n_train(n, s) / 1000 * m["train_min_per_1k_chats"] * scale * epochs / 60
    pred = (n * s["holdout_share"] + n_test) * tta / m["pred_chat_variants_per_min"] * scale / 60
    return s["setup_hours"] + train + pred


def plan(n: int, n_test: int, chars_p50: float, s: dict, chars_p90: float | None = None) -> list[str]:
    r, ms = s["rules"], s["models"]
    budget, gpus = r["budget_hours"], r["gpus"]
    t50 = tokens_from_chars(chars_p50, s)
    t90 = tokens_from_chars(chars_p90, s) if chars_p90 else None
    out = [f"train {n}, test {n_test}: prompt tokens ~{t50:.0f} (p50)"
           + (f", ~{t90:.0f} (p90)" if t90 else "") + f"; budget {budget:.1f} GPU h per QLoRA run"]
    max_len = r["max_len_cap"]
    if t90 and t90 > max_len:
        out.append(f"NOTE: p90 prompt ~{t90:.0f} tokens > max_len {max_len}: >10% of chats get "
                   "truncated (lines far from the others are dropped first; evidence is kept)")

    q12, choice = ms["qlora_12b"], None
    n_all = int(_n_train(n, s))
    steps = []

    def consider(name, m, tokens, epochs, k, sets):
        h = qlora_train_hours(m, n, tokens, epochs, s, k)
        steps.append(f"  {name}: training job {h:.1f} h")
        if h <= budget + 1e-9:
            return Plan(name, h, f"{min(k or n_all, n_all)} training chats", sets)
        return None

    for epochs in (2, 1):
        choice = choice or consider(f"QLoRA 12B, {epochs} epoch(s), all {n_all} chats", q12, t50, epochs,
                                    None, [f"qlora.epochs={epochs}"])
    if choice is None:
        k = max_subsample(q12, n, t50, 1, s, budget)
        if k >= r["min_subsample"]:
            choice = consider(f"QLoRA 12B, 1 epoch, subsample {k}", q12, t50, 1, k,
                              ["qlora.epochs=1", f"train.subsample={k}"])
        else:
            steps.append(f"  QLoRA 12B, 1 epoch, subsample: only {k} chats fit (< {r['min_subsample']})")
    t_used = t50
    if choice is None and t50 > 512:
        t_used = 512.0
        k = max_subsample(q12, n, t_used, 1, s, budget)
        if k >= r["min_subsample"]:
            choice = consider(f"QLoRA 12B, 1 epoch, max_len 512, subsample {min(k, n_all)}", q12, t_used, 1,
                              min(k, n_all), ["qlora.epochs=1", "llm.max_len=512"]
                              + ([f"train.subsample={k}"] if k < n_all else []))
        else:
            steps.append(f"  QLoRA 12B, 1 epoch, max_len 512: only {k} chats fit (< {r['min_subsample']})")
    e4b = ms.get("qlora_e4b") or {}
    if choice is None and e4b.get("train_s_per_chat"):
        t_used = t50
        k = max_subsample(e4b, n, t50, 1, s, budget)
        choice = consider(f"fallback QLoRA gemma-4-E4B, 1 epoch, {min(k, n_all)} chats", e4b, t50, 1,
                          min(k, n_all), ["llm.model_id=google/gemma-4-E4B-it", "qlora.epochs=1"]
                          + ([f"train.subsample={k}"] if k < n_all else []))
    out += steps
    test_h = 0.0
    if choice is None:
        out.append("QLoRA: NOTHING FITS the budget -> ask before deciding (raise the budget up to the "
                   "12 h session limit, cap the holdout, or rely on DeBERTa / zero-shot)")
    else:
        m = e4b if "E4B" in choice.name else q12
        test_h = s["setup_hours"] + pred_hours(m, n_test, t_used, s, gpus)
        out.append(f"QLoRA -> {choice.name}: training job ~{choice.hours:.1f} h; TTA 1")
        out.append("   liars train --config configs/llm_qlora.yaml --set "
                   + " --set ".join(choice.sets + ["augment.tta_n=1"]))
        out.append(f"   test: liars predict --run latest:llm_qlora --in <test> --gpus {gpus} --test-preds"
                   f"  (~{test_h:.1f} h; same session if it stays within {budget:.0f} h, else a second session)")

    zs = zeroshot_hours(ms["zeroshot_12b"], n, n_test, t50, s, gpus)
    out.append(f"LLM zero-shot (safety submission): {zs:.1f} h -> "
               + ("RUN first" if zs <= r["zeroshot_max_hours"] else "SKIP (too slow; submit the baseline instead)"))
    db = ms["deberta_large"]
    d2 = deberta_hours(db, n, n_test, t50, 2, 2, s)
    d1 = deberta_hours(db, n, n_test, t50, 1, 1, s)
    if d2 <= r["deberta_max_hours"]:
        dec = f"RUN holdout mode, 2 epochs, TTA 2 ({d2:.1f} h)"
    elif d1 <= r["deberta_max_hours"]:
        dec = f"RUN holdout mode, 1 epoch, TTA 1 ({d1:.1f} h): --set deberta.epochs=1 --set augment.tta_n=1"
    else:
        dec = f"SKIP (2 epochs {d2:.1f} h, 1 epoch {d1:.1f} h > {r['deberta_max_hours']} h cap)"
    out.append(f"DeBERTa-large (ensemble partner): {dec}")
    total = (choice.hours + test_h if choice else 0) + (zs if zs <= r["zeroshot_max_hours"] else 0) \
        + (d2 if d2 <= r["deberta_max_hours"] else d1 if d1 <= r["deberta_max_hours"] else 0)
    out.append(f"TOTAL planned GPU: ~{total:.1f} h (+ ~20% margin for reruns)")
    return out
