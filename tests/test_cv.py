from collections import Counter

from liars.cv import assign_folds, inner_split, load_folds, save_folds


def test_folds_stratified_and_deterministic(tiny, tmp_path):
    folds = assign_folds(tiny, k=4, seed=1)
    assert set(folds) == {ex.id for ex in tiny}
    assert folds == assign_folds(tiny, k=4, seed=1)
    for f in range(4):
        labels = Counter(ex.label for ex in tiny if folds[ex.id] == f)
        assert set(labels.values()) == {2}  # 8 per label / 4 folds
    p = tmp_path / "folds.csv"
    save_folds(folds, p)
    assert load_folds(p) == folds


def test_inner_split(tiny):
    tr, va = inner_split(tiny, 0.2, seed=0)
    assert len(tr) + len(va) == len(tiny)
    assert not {e.id for e in tr} & {e.id for e in va}
