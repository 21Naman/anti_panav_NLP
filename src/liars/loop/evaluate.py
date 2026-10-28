"""Test + evaluate: sanity tests on a run's outputs, metrics, and champion promotion.

Promotion (a small holdout is noisy, and every comparison is another look at it):
  win  : accuracy gain >= min_gain AND exact McNemar p <= alpha (paired on the same chats)
  tie+ : |accuracy gain| <= tie_tol AND log-loss lower by >= min_nll_gain AND the paired
         bootstrap 95% lower bound of the accuracy gain >= -tie_tol
  else : reject
Both runs are compared only on the chats both have predictions for (min_aligned of them).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from liars.preds import PROB_COLS, check_preds
from liars.schema import SPEAKERS

EPS = 1e-6


# ------------------------------------------------------------------ metrics
def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], EPS, None))))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def metrics(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    return {"n": int(len(y)), "acc": float((p.argmax(1) == y).mean()), "nll": nll(p, y),
            "brier": float(((p - np.eye(len(SPEAKERS))[y]) ** 2).sum(1).mean()), "ece": ece(p, y, n_bins)}


def oof_arrays(oof: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = oof["id"].tolist()
    y = np.array([SPEAKERS.index(l) for l in oof["label"]])
    return ids, oof[PROB_COLS].to_numpy(), y


# ------------------------------------------------------------------ sanity tests
@dataclass
class Sanity:
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def sanity(oof: pd.DataFrame | None, test: pd.DataFrame | None, test_ids: list[str] | None,
           scfg: dict, expect_oof_n: int | None = None) -> Sanity:
    s = Sanity()
    if oof is None:
        s.failures.append("no OOF predictions file")
        return s
    errs = check_preds(oof)
    if errs:
        s.failures.append(f"OOF file invalid: {errs}")
        return s
    if "label" not in oof.columns or oof["label"].isna().any():
        s.failures.append("OOF rows without labels")
        return s
    if expect_oof_n is not None and len(oof) != expect_oof_n:
        s.failures.append(f"OOF has {len(oof)} rows, expected {expect_oof_n} (held-out coverage)")
    ids, P, y = oof_arrays(oof)
    acc = float((P.argmax(1) == y).mean())
    chance = 1 / len(SPEAKERS)
    if acc < chance + scfg["min_margin_over_chance"]:
        s.failures.append(f"accuracy {acc:.3f} is not above chance {chance:.2f} + "
                          f"{scfg['min_margin_over_chance']} (collapsed or broken run)")
    pred_share = pd.Series(P.argmax(1)).value_counts(normalize=True).max()
    label_share = pd.Series(y).value_counts(normalize=True).max()
    if pred_share > label_share + scfg["max_pred_share_excess"]:
        s.failures.append(f"predicts one letter {pred_share:.0%} of the time (labels: max {label_share:.0%}): "
                          "position/letter collapse")
    if acc >= scfg["suspicious_acc"] and len(y) >= scfg["suspicious_min_n"]:
        s.warnings.append(f"held-out accuracy {acc:.4f} looks too good: check for leakage "
                          "(labels in the input, duplicates across folds)")
    if test_ids is not None:
        if test is None:
            s.failures.append("no test predictions file")
        else:
            errs = check_preds(test)
            if errs:
                s.failures.append(f"test predictions invalid: {errs}")
            elif sorted(test["id"]) != sorted(test_ids):
                s.failures.append("test prediction ids do not match the test file")
            else:
                tshare = pd.Series(test[PROB_COLS].to_numpy().argmax(1)).value_counts(normalize=True).max()
                if tshare > label_share + scfg["max_pred_share_excess"]:
                    s.failures.append(f"test predictions put {tshare:.0%} on one letter")
                ood = abs(test[PROB_COLS].to_numpy().max(1).mean() - P.max(1).mean())
                if ood > scfg["max_conf_shift"]:
                    s.warnings.append(f"mean confidence differs by {ood:.2f} between held-out and test: "
                                      "possible train/test shift")
    return s


# ------------------------------------------------------------------ comparison
def mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Exact two-sided McNemar test on paired correctness (binomial on discordant pairs)."""
    from scipy.stats import binomtest

    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5).pvalue)


def bootstrap_diff(correct_new: np.ndarray, correct_old: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = correct_new.astype(float) - correct_old.astype(float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@dataclass
class Verdict:
    promote: bool
    rule: str
    detail: dict


def compare(new: pd.DataFrame, champ: pd.DataFrame, pcfg: dict, seed: int = 0) -> Verdict:
    a = new.set_index("id")
    b = champ.set_index("id")
    common = sorted(set(a.index) & set(b.index))
    if len(common) < pcfg["min_aligned"]:
        return Verdict(False, "too_few_aligned", {"aligned": len(common)})
    y = np.array([SPEAKERS.index(l) for l in a.loc[common, "label"]])
    pa, pb = a.loc[common, PROB_COLS].to_numpy(), b.loc[common, PROB_COLS].to_numpy()
    ca, cb = pa.argmax(1) == y, pb.argmax(1) == y
    gain = float(ca.mean() - cb.mean())
    nll_gain = nll(pb, y) - nll(pa, y)
    p = mcnemar_p(ca, cb)
    lo, hi = bootstrap_diff(ca, cb, pcfg["n_boot"], seed)
    det = {"aligned": len(common), "acc_new": float(ca.mean()), "acc_champ": float(cb.mean()),
           "acc_gain": gain, "nll_gain": float(nll_gain), "mcnemar_p": p, "boot95": [lo, hi]}
    if gain >= pcfg["min_gain"] and p <= pcfg["alpha"]:
        return Verdict(True, "significant_accuracy_win", det)
    if abs(gain) <= pcfg["tie_tol"] and nll_gain >= pcfg["min_nll_gain"] and lo >= -pcfg["tie_tol"]:
        return Verdict(True, "tie_with_better_log_loss", det)
    return Verdict(False, "not_better", det)


def min_detectable_gain(n: int, disagree_rate: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest paired accuracy gain detectable with the given power:
    (z_{1-alpha/2} + z_power) * sqrt(disagree_rate / n) (normal approximation; 2.80 * sqrt(d/n)
    at alpha 0.05 / 80% power: ~4.4 points at n=400, ~2.8 at n=1000 with 10% disagreement).
    Shown in the report so a 0.5-point 'gain' on 400 chats is not mistaken for progress."""
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * np.sqrt(max(disagree_rate, 1e-9) / n))
