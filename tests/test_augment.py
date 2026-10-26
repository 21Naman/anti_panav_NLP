import random
from collections import Counter

import numpy as np

from liars.augment import map_back, permute_speakers, shuffle_evidence, tta_variants
from liars.schema import SPEAKERS, validate


def test_permute_is_consistent(tiny):
    ex = tiny[0]
    new, inverse = permute_speakers(ex, random.Random(3))
    assert validate(new) == []
    assert inverse[new.label] == ex.label
    for m_old, m_new in zip(ex.messages, new.messages):
        assert inverse[m_new.speaker] == m_old.speaker and m_new.text == m_old.text


def test_tta_variants_cover_every_slot(tiny):
    ex = tiny[5]
    vs = tta_variants(ex, 4, seed=0)
    assert vs[0][0] is ex
    # with 4 cyclic variants the true liar sits in every letter slot exactly once
    assert Counter(v.label for v, _ in vs) == {s: 1 for s in SPEAKERS}
    assert len(tta_variants(ex, 7, seed=0)) == 7


def test_map_back_round_trip(tiny):
    ex = tiny[2]
    for v, inverse in tta_variants(ex, 6, seed=1):
        scores = np.array([1.0 if s == v.label else 0.0 for s in SPEAKERS])
        back = map_back(scores, inverse)
        assert SPEAKERS[int(back.argmax())] == ex.label


def test_shuffle_evidence_keeps_facts(tiny):
    ex = tiny[1]
    out = shuffle_evidence(ex, random.Random(0))
    assert sorted(out.evidence) == sorted(ex.evidence) and out.messages == ex.messages
