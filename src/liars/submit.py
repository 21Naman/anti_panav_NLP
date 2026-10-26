"""`liars submit`: prediction file -> validated competition CSV.

Driven by configs/submission.yaml:
  id_column, label_column       column names in the CSV
  label_format                  letter | int | name
  int_order                     as_listed | first_appearance  (which speaker is 0..3)
`decode_label` is the inverse, for adapters that turn raw train labels into letters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from liars.preds import argmax_letter, check_preds
from liars.schema import Example

LABEL_FORMATS = ("letter", "int", "name")
INT_ORDERS = ("as_listed", "first_appearance")


def speaker_order(ex: Example, int_order: str) -> list[str]:
    if int_order == "as_listed":
        return list(ex.speakers)
    if int_order == "first_appearance":
        order = list(dict.fromkeys(m.speaker for m in ex.messages))
        return order + [s for s in ex.speakers if s not in order]
    raise ValueError(f"int_order must be one of {INT_ORDERS}, got {int_order!r}")


def encode_label(letter: str, ex: Example, scfg: dict) -> Any:
    fmt = scfg["label_format"]
    if fmt == "letter":
        return letter
    if fmt == "int":
        return speaker_order(ex, scfg["int_order"]).index(letter)
    if fmt == "name":
        name_map = ex.meta.get("name_map")
        if not name_map:
            raise ValueError(f"{ex.id}: label_format=name needs meta.name_map")
        return name_map[letter]
    raise ValueError(f"label_format must be one of {LABEL_FORMATS}, got {fmt!r}")


def decode_label(value: Any, ex: Example, scfg: dict) -> str:
    fmt = scfg["label_format"]
    if fmt == "letter":
        return str(value)
    if fmt == "int":
        return speaker_order(ex, scfg["int_order"])[int(value)]
    if fmt == "name":
        inverse = {v: k for k, v in ex.meta["name_map"].items()}
        return inverse[value]
    raise ValueError(f"label_format must be one of {LABEL_FORMATS}, got {fmt!r}")


def make_submission(preds: pd.DataFrame, test: Sequence[Example], scfg: dict,
                    sample: pd.DataFrame | None = None) -> pd.DataFrame:
    errors = check_preds(preds)
    if errors:
        raise ValueError(f"bad prediction file: {errors}")
    by_id = {ex.id: ex for ex in test}
    letters = dict(zip(preds["id"], argmax_letter(preds)))
    missing = [i for i in by_id if i not in letters]
    if missing:
        raise ValueError(f"{len(missing)} test ids have no prediction (e.g. {missing[:3]})")
    id_col, label_col = scfg["id_column"], scfg["label_column"]
    order = list(sample[id_col].astype(str)) if sample is not None else [ex.id for ex in test]
    sub = pd.DataFrame({
        id_col: order,
        label_col: [encode_label(letters[i], by_id[i], scfg) for i in order],
    })
    if sample is not None:
        sub = sub[list(sample.columns)]
    return sub


def validate_submission(sub: pd.DataFrame, test: Sequence[Example], scfg: dict,
                        sample: pd.DataFrame | None = None) -> list[str]:
    errors = []
    id_col, label_col = scfg["id_column"], scfg["label_column"]
    expected_cols = list(sample.columns) if sample is not None else [id_col, label_col]
    if list(sub.columns) != expected_cols:
        errors.append(f"columns {list(sub.columns)} != expected {expected_cols}")
        return errors
    ids = sub[id_col].astype(str)
    if ids.duplicated().any():
        errors.append(f"{int(ids.duplicated().sum())} duplicate ids")
    test_ids = {ex.id for ex in test}
    if set(ids) != test_ids:
        errors.append(f"id mismatch: {len(set(ids) - test_ids)} extra, "
                      f"{len(test_ids - set(ids))} missing")
    if sample is not None and list(ids) != list(sample[id_col].astype(str)):
        errors.append("row order differs from the sample submission")
    if sub[label_col].isna().any():
        errors.append("empty labels")
    by_id = {ex.id: ex for ex in test}
    for i, v in zip(ids, sub[label_col]):
        if i not in by_id:
            continue
        try:
            decode_label(v, by_id[i], scfg)
        except (KeyError, IndexError, ValueError):
            errors.append(f"invalid label {v!r} for id {i}")
            break
    return errors


def write_submission(sub: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path
