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


def test_time_guard_stops_long_runs(tiny):
    try:
        from transformers import AutoModelForMultipleChoice  # noqa: F401
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(XSMALL)
    except Exception as e:
        pytest.skip(f"model unavailable: {e}")
    cfg = load_config(ROOT / "configs" / "deberta_mc.yaml",
                      [f"deberta.model_id={XSMALL}", "deberta.device=cpu", "deberta.max_len=128",
                       "deberta.micro_batch_size=2", "deberta.grad_accum=1", "deberta.time_probe_steps=1",
                       "deberta.max_gpu_hours=1e-9", "augment.tta_n=1"])
    cfg["train"]["_fits_left"] = 5
    m = DebertaMCModel()
    m.log = lambda *a: None
    with pytest.raises(RuntimeError, match="max_gpu_hours"):
        m.fit(tiny[:8], None, cfg)


def test_fp16_checkpoint_loads_as_fp32(tmp_path):
    """The real NLI DeBERTa checkpoints are stored in fp16; training needs fp32 master weights."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        src = AutoModelForSequenceClassification.from_pretrained(XSMALL)
        tok = AutoTokenizer.from_pretrained(XSMALL)
    except Exception as e:
        pytest.skip(f"model unavailable: {e}")
    src.half().save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    m = DebertaMCModel()
    m.dcfg = load_config(ROOT / "configs" / "deberta_mc.yaml", ["deberta.device=cpu"])["deberta"]
    m._load(str(tmp_path), add_marks=True)
    assert {p.dtype for p in m.model.parameters()} == {torch.float32}


@pytest.mark.parametrize("retries", [1, 0])
def test_collapse_guard_restarts(tiny, retries):
    import torch
    from liars.models.deberta_mc import _Collapsed
    cfg = load_config(ROOT / "configs" / "deberta_mc.yaml",
                      ["deberta.device=cpu", f"deberta.collapse_retries={retries}"])
    m = DebertaMCModel()
    m.log = lambda *a: None
    attempts = []

    def fake_train(train, val, cfg, attempt):  # attempt 0 collapses, the restart is healthy
        attempts.append(attempt)
        m.torch, m.device, m.model = torch, "cpu", torch.nn.Linear(1, 1)
        if attempt == 0:
            raise _Collapsed("step 77: val loss 1.3863, acc 0.26")
        m.best_step = 5
    m._train = fake_train
    if retries:
        m.fit(tiny[:4], None, cfg)
        assert attempts == [0, 1] and m.restarts == 1
    else:
        with pytest.raises(RuntimeError, match="collapsed"):
            m.fit(tiny[:4], None, cfg)
        assert attempts == [0]


def test_collapse_rule():
    m = DebertaMCModel()
    m.dcfg = load_config(ROOT / "configs" / "deberta_mc.yaml")["deberta"]
    assert m._collapsed(77, 310, 0.26, 1.3863)        # the real collapsed run (2026-10-27)
    assert not m._collapsed(47, 190, 0.4625, 1.3795)  # slow but healthy start (fold 0)
    assert not m._collapsed(10, 310, 0.25, 1.3863)    # too early to judge
