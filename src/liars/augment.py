"""Speaker permutation, evidence shuffling and test-time augmentation (TTA).

Switches (config `augment:`): permute_train, tta_n (1 = off), shuffle_evidence.
"""

from __future__ import annotations

import copy
import random
from typing import Sequence

import numpy as np

from liars.schema import SPEAKERS, Example, Message


def apply_permutation(ex: Example, perm: dict[str, str]) -> Example:
    """Rename speakers with `perm` (old letter -> new letter)."""
    out = copy.deepcopy(ex)
    out.messages = [Message(perm[m.speaker], m.text) for m in ex.messages]
    out.speakers = list(SPEAKERS)
    out.label = perm[ex.label] if ex.label is not None else None
    if "name_map" in ex.meta:
        out.meta["name_map"] = {perm[k]: v for k, v in ex.meta["name_map"].items()}
    return out


def permute_speakers(ex: Example, rng: random.Random) -> tuple[Example, dict[str, str]]:
    """Random bijection over A-D. Returns (new example, inverse map new letter -> old letter)."""
    shuffled = list(SPEAKERS)
    rng.shuffle(shuffled)
    perm = dict(zip(SPEAKERS, shuffled))
    return apply_permutation(ex, perm), {v: k for k, v in perm.items()}


def shuffle_evidence(ex: Example, rng: random.Random) -> Example:
    out = copy.deepcopy(ex)
    rng.shuffle(out.evidence)
    return out


def tta_variants(ex: Example, n: int, seed: int = 0) -> list[tuple[Example, dict[str, str]]]:
    """n variants: the original first, then distinct random permutations (cyclic shifts
    first, so n <= 4 puts every speaker in every letter slot exactly once)."""
    identity = {s: s for s in SPEAKERS}
    out = [(ex, identity)]
    seen = {tuple(SPEAKERS)}
    k = len(SPEAKERS)
    for shift in range(1, k):  # cyclic shifts: balanced letter positions
        if len(out) >= n:
            break
        perm = {SPEAKERS[i]: SPEAKERS[(i + shift) % k] for i in range(k)}
        seen.add(tuple(perm[s] for s in SPEAKERS))
        out.append((apply_permutation(ex, perm), {v: q for q, v in perm.items()}))
    rng = random.Random(f"{seed}-{ex.id}")
    while len(out) < n:
        shuffled = list(SPEAKERS)
        rng.shuffle(shuffled)
        if tuple(shuffled) in seen:
            continue
        seen.add(tuple(shuffled))
        perm = dict(zip(SPEAKERS, shuffled))
        out.append((apply_permutation(ex, perm), {v: q for q, v in perm.items()}))
    return out[:n]


def map_back(scores: Sequence[float], inverse: dict[str, str]) -> np.ndarray:
    """Scores indexed by the variant's letters -> scores indexed by the original letters."""
    out = np.zeros(len(SPEAKERS))
    for j, new in enumerate(SPEAKERS):
        out[SPEAKERS.index(inverse[new])] = scores[j]
    return out
