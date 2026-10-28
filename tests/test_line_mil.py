import math
from pathlib import Path

import numpy as np
import pytest

from liars.config import load_config
from liars.models import get_model_class
from liars.models.line_mil import (LineMILModel, fit_window, line_labels, line_windows, pool_np,
                                   render_window)
from liars.schema import SPEAKERS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "line_mil.yaml"
XSMALL = "cross-encoder/nli-deberta-v3-xsmall"


# ------------------------------------------------------------------ pure helpers
def test_windows_cover_every_line_within_bounds(tiny):
    for ex in tiny:
        items = line_windows(ex, 3, 1)
        assert [it.msg_idx for it in items] == list(range(len(ex.messages)))
        for it in items:
            assert it.msg_idx in it.window
            assert it.window == sorted(it.window)
            assert min(it.window) == max(0, it.msg_idx - 3)
            assert max(it.window) == min(len(ex.messages) - 1, it.msg_idx + 1)
            assert it.speaker == ex.messages[it.msg_idx].speaker


def test_fit_window_keeps_target_and_drops_furthest_first():
    lens = {i: 10 for i in range(6)}
    keep, dropped = fit_window([0, 1, 2, 3, 4, 5], 4, lens, 35)
    assert 4 in keep and dropped == 3 and keep == [3, 4, 5]
    keep, dropped = fit_window([0, 1, 2], 1, lens, 5)  # even over budget, the target stays
    assert keep == [1] and dropped == 2
    assert fit_window([0, 1], 0, lens, 100) == ([0, 1], 0)


def test_render_marks_exactly_the_target(tiny):
    ex = tiny[0]
    text = render_window(ex, [0, 1, 2], 1, "{speaker}: {text}", "<t>", "</t>")
    assert text.count("<t>") == 1 and text.count("</t>") == 1
    m = ex.messages[1]
    assert f"<t> {m.speaker}: {m.text} </t>" in text


def test_line_labels_from_meta(tiny):
    for ex in tiny:
        y = line_labels(ex)
        assert y is not None and y.shape == (len(ex.messages),)
        assert set(np.flatnonzero(y)) == set(ex.meta["lie_msg_idx"])
    ex = tiny[0]
    ex.meta = {}
    assert line_labels(ex) is None  # real competition data: no line labels


def test_pool_np():
    scores = np.array([1.0, 3.0, -2.0, 0.5])
    spk = ["A", "A", "B", "C"]  # D has no lines
    mx = pool_np(scores, spk, "max", 1.0, -20.0)
    assert list(mx) == [3.0, -2.0, 0.5, -20.0]
    assert list(pool_np(scores, spk, "mean", 1.0, -20.0)[:3]) == [2.0, -2.0, 0.5]
    lse = pool_np(scores, spk, "logsumexp", 1.0, -20.0)
    assert 3.0 < lse[0] <= 3.0 + math.log(2) + 1e-9      # soft max: >= max, <= max + tau*log(n)
    sharp = pool_np(scores, spk, "logsumexp", 0.01, -20.0)
    assert abs(sharp[0] - 3.0) < 1e-6                      # small tau -> max
    with pytest.raises(ValueError):
        pool_np(scores, spk, "median", 1.0, -20.0)


def test_config_and_registry():
    cfg = load_config(CONFIG)
    assert cfg["model"]["name"] == "line_mil"
    assert get_model_class("line_mil") is LineMILModel
    assert hasattr(LineMILModel, "load")  # `liars predict` (and --gpus 2) needs it
    for stage in (1, 2):
        assert stage in cfg["stages"]


# ------------------------------------------------------------------ with a tiny model
@pytest.fixture
def xsmall():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(XSMALL)
    except Exception as e:
        pytest.skip(f"model unavailable: {e}")
    return XSMALL


def _model(overrides=()):
    m = LineMILModel()
    m.cfg = load_config(CONFIG, [f"line_mil.model_id={XSMALL}", "line_mil.device=cpu",
                                 "line_mil.gradient_checkpointing=false", *overrides])
    m.dcfg, m.n_truncated, m._cache = m.cfg["line_mil"], 0, {}
    m._load(XSMALL, add_marks=True)
    return m


def test_encoding_one_mark_per_pair_and_budget(tiny, xsmall):
    m = _model(["line_mil.max_len=96", "line_mil.context_before=6"])
    for ex in tiny[:6]:
        ids, spk = m._encode_chat(ex)
        assert len(ids) == len(ex.messages) and spk == [x.speaker for x in ex.messages]
        for i, x in enumerate(ids):
            assert len(x) <= 96
            text = m.tok.decode(x)
            assert text.count("<t>") == 1
            assert ex.messages[i].text[:15] in text
    assert m.n_truncated > 0  # max_len 96 with 6 lines of context must drop some context


def test_contradiction_head_found(xsmall):
    m = _model()
    assert m.model.config.id2label[m.contra_idx].lower().startswith("contra")
    m2 = _model(["line_mil.head=new"])
    assert m2.contra_idx is None and m2.model.config.num_labels == 1


def test_torch_pool_matches_numpy(tiny, xsmall):
    import torch
    m = _model()
    s = torch.tensor([1.0, 3.0, -2.0, 0.5])
    spk = ["A", "A", "B", "C"]
    for method in ("max", "mean", "logsumexp"):
        m.dcfg["pool"] = method
        got = m._pool(s, spk).numpy()
        want = pool_np(s.numpy(), spk, method, m.dcfg["pool_tau"], m.dcfg["empty_speaker_score"])
        assert np.allclose(got, want, atol=1e-5)


def test_debug_train_save_reload_predict(tiny_path, tmp_path, xsmall):
    from liars.preds import PROB_COLS, load_preds
    from liars.schema import load_jsonl
    from liars.trainer import resolve_model_dir, run_training
    from liars.paths import get_paths

    cfg = load_config(CONFIG, [f"paths.data_dir={tmp_path}"])
    out = run_training(cfg, CONFIG, debug=True, log=lambda *a: None)
    preds = load_preds(out["preds_path"])
    assert np.allclose(preds[PROB_COLS].to_numpy().sum(1), 1.0)
    assert len(preds) == len(load_jsonl(tiny_path))
    # reload the saved holdout model and predict again: same numbers
    paths = get_paths({"paths": {"data_dir": str(tmp_path / "debug")}})
    mdir = resolve_model_dir(paths, out["run"])
    run_cfg = load_config(paths.run_configs / f"{out['run']}.yaml")
    m = LineMILModel.load(mdir, run_cfg)
    test = load_jsonl(tiny_path)
    again = m.predict_proba(test)
    assert np.allclose(again[PROB_COLS].to_numpy(), preds[PROB_COLS].to_numpy(), atol=1e-4)
    lines = m.line_scores(test[:3])
    assert len(lines) == sum(len(ex.messages) for ex in test[:3])
    assert lines["p_contra"].between(0, 1).all()
    rate = m.line_hit_rate(test)
    assert rate is not None and 0.0 <= rate <= 1.0
