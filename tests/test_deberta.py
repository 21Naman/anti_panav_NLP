import os
import time
from pathlib import Path

import pytest

from liars.config import load_config
from liars.models.deberta_mc import DebertaMCModel, keep_for_choice
from liars.paths import Paths
from liars.schema import SPEAKERS
from liars.trainer import resolve_model_dir

ROOT = Path(__file__).resolve().parents[1]
XSMALL = "cross-encoder/nli-deberta-v3-xsmall"


def test_keep_for_choice_never_drops_marked_lines(tiny):
    for ex in tiny:
        lens = [10] * len(ex.messages)
        for s in SPEAKERS:
            own = [i for i, m in enumerate(ex.messages) if m.speaker == s]
            budget = 20 + sum(lens[i] + 4 for i in own) + 15  # room for ~1 other line
            keep, dropped = keep_for_choice(ex, s, lens, 20, budget, 4)
            assert set(own) <= set(keep)
            assert dropped == len(ex.messages) - len(keep) and dropped > 0


def test_drops_furthest_other_lines_first(tiny):
    ex = tiny[0]
    s = ex.messages[0].speaker
    lens = [10] * len(ex.messages)
    keep, _ = keep_for_choice(ex, s, lens, 0, 10 * len(ex.messages) - 5, 0)
    dropped = set(range(len(ex.messages))) - set(keep)
    own = [i for i, m in enumerate(ex.messages) if m.speaker == s]
    far = max((i for i in range(len(ex.messages)) if i not in own),
              key=lambda i: (min(abs(i - j) for j in own), -i))
    assert dropped == {far}


@pytest.fixture
def encoder():
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(XSMALL)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    m = DebertaMCModel()
    m.cfg = load_config(ROOT / "configs" / "deberta_mc.yaml", ["deberta.max_len=96"])
    m.dcfg = m.cfg["deberta"]
    m.n_truncated = 0
    from transformers import AutoTokenizer
    m.tok = AutoTokenizer.from_pretrained(XSMALL)
    m.tok.add_special_tokens({"additional_special_tokens": [m.dcfg["mark_open"], m.dcfg["mark_close"]]})
    return m


def test_encoded_choices_keep_all_marked_lines(tiny, encoder):
    m = encoder
    for ex in tiny[:8]:
        choices = m._encode_example(ex)
        assert len(choices) == 4
        for s, ids in zip(SPEAKERS, choices):
            assert len(ids) <= m.dcfg["max_len"]
            text = m.tok.decode(ids)
            for msg in ex.messages:
                if msg.speaker == s:
                    assert f"{m.dcfg['mark_open']} {s}: {msg.text}"[:40] in text.replace("  ", " ") \
                        or msg.text[:25] in text
            assert text.count(m.dcfg["mark_open"]) == sum(msg.speaker == s for msg in ex.messages)
    assert m.n_truncated > 0  # max_len 96 forces dropping other speakers' lines


def test_resolve_latest(tmp_path):
    paths = Paths(tmp_path)
    for i, name in enumerate(["deberta_mc_s1_a", "deberta_mc_s1_b", "other_c"]):
        d = paths.models / name / "fold0"
        d.mkdir(parents=True)
        os.utime(paths.models / name, (time.time() + i, time.time() + i))
    assert resolve_model_dir(paths, "latest:deberta_mc_s1").parent.name == "deberta_mc_s1_b"
    with pytest.raises(FileNotFoundError):
        resolve_model_dir(paths, "latest:nothing")
