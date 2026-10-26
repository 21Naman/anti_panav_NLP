"""Every model implements this interface (see CLAUDE.md)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd

from liars.schema import Example


class Model(ABC):
    #: True if fit() wants a validation split (early stopping). The trainer then
    #: carves an inner split out of the TRAINING folds; it never uses held-out folds.
    uses_val: bool = False

    @abstractmethod
    def fit(self, train: Sequence[Example], val: Sequence[Example] | None, cfg: dict) -> None:
        ...

    @abstractmethod
    def predict_proba(self, examples: Sequence[Example]) -> pd.DataFrame:
        """DataFrame[id, p_A, p_B, p_C, p_D], rows sum to 1, same order as `examples`."""

    def refit_info(self) -> dict:
        """What a refit on 100% of train needs to repeat this fit (e.g. number of steps).
        Passed back as cfg["train"]["_refit"] by the trainer when train.refit_full is on."""
        return {}
