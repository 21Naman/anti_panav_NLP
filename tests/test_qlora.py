import json
from pathlib import Path

import numpy as np
import pytest

from liars.config import load_config
from liars.preds import PROB_COLS

ROOT = Path(__file__).resolve().parents[1]
TINY_LM = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def _cfg(tmp_path, *extra):
    return load_config(ROOT / "configs" / "llm_qlora.yaml", [
        f"llm.model_id={TINY_LM}", "llm.device=cpu", "llm.load_in_4bit=false", "llm.max_len=400",
        "qlora.gradient_checkpointing=false", "qlora.r=2", "qlora.alpha=4",
        "qlora.micro_batch_size=4", "qlora.grad_accum=1", "qlora.epochs=1",
        "qlora.eval_every_frac=0.5", "qlora.time_probe_steps=1", "augment.tta_n=2",
        f"train._save_dir={tmp_path / 'fold0'}", *extra])


@pytest.fixture
def qlora():
    try:
        from liars.llm import load_lm
        load_lm({"model_id": TINY_LM, "device": "cpu", "load_in_4bit": False})
    except Exception as e:
        pytest.skip(f"tiny model unavailable: {e}")
    from liars.models.llm_qlora import LLMQLoRAModel
    return LLMQLoRAModel


def test_fit_save_load_predict(tiny, tmp_path, qlora):
    cfg = _cfg(tmp_path)
    m = qlora()
    m.fit(tiny[:24], tiny[24:28], cfg)
    p = m.predict_proba(tiny[28:])
    assert np.allclose(p[PROB_COLS].sum(axis=1), 1)
    saved = tmp_path / "fold0"
    assert (saved / "model_meta.json").exists() and any(saved.glob("adapter_*"))
    meta = json.loads((saved / "model_meta.json").read_text())
    assert meta["best_step"] == m.best_step and meta["history"]
    again = qlora.load(saved, cfg).predict_proba(tiny[28:])
    assert np.allclose(again[PROB_COLS].to_numpy(), p[PROB_COLS].to_numpy(), atol=1e-4)
    assert m.refit_info()["steps"] >= 1 and "QLoRA report" in m.report()


def test_refit_runs_fixed_steps_without_val(tiny, tmp_path, qlora):
    cfg = _cfg(tmp_path, "train._refit={steps: 2, temperature: 0.5}")
    m = qlora()
    m.fit(tiny[:16], None, cfg)
    assert m.best_step == 2 and m.temperature == 0.5


def test_time_guard_stops_long_runs(tiny, tmp_path, qlora):
    cfg = _cfg(tmp_path, "qlora.max_gpu_hours=0.0000001")
    with pytest.raises(RuntimeError, match="max_gpu_hours"):
        qlora().fit(tiny[:16], tiny[16:20], cfg)
