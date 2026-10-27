"""Pseudo-labelling (only if the organizers allow it; `pseudo.enabled` defaults to false).

Selects test chats whose top ensemble probability is >= threshold and writes them as
canonical training examples (label = argmax, meta.source = "pseudo"). The trainer adds
such files (train.extra_train) to the TRAINING side only, never to validation or held-out
folds, so held-out scores stay honest.
"""

from __future__ import annotations

import copy
from typing import Sequence

import pandas as pd

from liars.preds import PROB_COLS, argmax_letter, check_preds
from liars.schema import Example

PSEUDO_SOURCE = "pseudo"


class PseudoDisabled(RuntimeError):
    pass


def select_pseudo(preds: pd.DataFrame, test: Sequence[Example], threshold: float,
                  enabled: bool) -> list[Example]:
    if not enabled:
        raise PseudoDisabled("pseudo.enabled is false: pseudo-labelling is off until the "
                             "organizers confirm it is allowed (record their answer in PROGRESS.md)")
    errors = check_preds(preds)
    if errors:
        raise ValueError(f"bad prediction file: {errors}")
    by_id = {ex.id: ex for ex in test}
    conf = preds[PROB_COLS].max(axis=1)
    letters = argmax_letter(preds)
    out = []
    for eid, c, letter in zip(preds["id"], conf, letters):
        if c >= threshold and eid in by_id:
            ex = copy.deepcopy(by_id[eid])
            ex.label = letter
            ex.meta = {**ex.meta, "source": PSEUDO_SOURCE, "pseudo_confidence": round(float(c), 4)}
            out.append(ex)
    return out
