import re
from pathlib import Path

import numpy as np
import pytest

import liars.models.llm_zeroshot as lz
from liars.config import load_config
from liars.preds import PROB_COLS
from liars.prompt import build_user_text
from liars.schema import SPEAKERS

ROOT = Path(__file__).resolve().parents[1]
TINY_LM = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


def _cfg(*extra):
    return load_config(ROOT / "configs" / "llm_zeroshot.yaml", list(extra))


class FakeHandle:
    stats = {"n_scored": 0, "seconds": 1.0, "nan_rows": 0, "truncated": 0, "max_mem_gb": 0.0}


def test_tta_maps_permuted_predictions_back(tiny, monkeypatch):
    """Fake LLM: votes for whichever letter speaks a lie line IN THE PERMUTED PROMPT.
    After mapping back, every prediction must be the ORIGINAL liar."""
    lies = {ex.messages[i].text for ex in tiny for i in ex.meta["lie_msg_idx"]}

    def fake_score(h, prompts, batch_size, progress=False):
        out = np.zeros((len(prompts), 4))
        for i, p in enumerate(prompts):
            for line in p.splitlines():
                m = re.match(r"^([ABCD]): (.*)$", line)
                if m and m.group(2) in lies:
                    out[i, SPEAKERS.index(m.group(1))] = 10.0
        return out

    monkeypatch.setattr(lz, "get_lm", lambda lcfg: FakeHandle())
    monkeypatch.setattr(lz, "prompt_for", lambda h, ex, pcfg, max_len, fewshot=(): build_user_text(ex, pcfg))
    monkeypatch.setattr(lz, "score_prompts", fake_score)
    m = lz.LLMZeroShotModel()
    m.fit(tiny, None, _cfg("augment.tta_n=4", "llm.calib_n=0"))
    logits = m._tta_logits(tiny)
    assert logits.shape == (32, 4, 4)
    # every variant, after mapping back, points at the original liar
    assert (logits.argmax(axis=2) == np.array([[SPEAKERS.index(e.label)] for e in tiny])).all()
    p = m.predict_proba(tiny)
    assert (p[PROB_COLS].to_numpy().argmax(1) == [SPEAKERS.index(e.label) for e in tiny]).all()


def test_tiny_real_model_runs(tiny, tmp_path):
    """End-to-end on a 4 MB random Qwen2 (CPU): letter tokens, chat template, scoring."""
    try:
        from liars.llm import load_lm
        load_lm({"model_id": TINY_LM, "device": "cpu", "load_in_4bit": False})
    except Exception as e:  # offline and not cached
        pytest.skip(f"tiny model unavailable: {e}")
    cfg = _cfg(f"llm.model_id={TINY_LM}", "llm.device=cpu", "llm.load_in_4bit=false",
               "llm.calib_n=4", "augment.tta_n=2", "llm.fewshot_n=1", "llm.max_len=150")
    cfg["train"]["_save_dir"] = str(tmp_path / "fold0")
    m = lz.LLMZeroShotModel()
    m.fit(tiny[:12], None, cfg)
    p = m.predict_proba(tiny[12:16])
    assert np.allclose(p[PROB_COLS].sum(axis=1), 1)
    m2 = lz.LLMZeroShotModel.load(tmp_path / "fold0", cfg)  # reload = temperature + few-shot
    assert m2.temperature == m.temperature and m2.fewshot == m.fewshot
    assert np.allclose(m2.predict_proba(tiny[12:16])[PROB_COLS], p[PROB_COLS], atol=1e-5)
    assert len(set(m.h.letter_ids)) == 4
    assert "chats per minute" in m.report()
    assert m.h.stats["truncated"] > 0  # max_len 150 forces truncation with a few-shot example


def test_letter_ids_follow_answer_prefix():
    try:
        from liars.llm import load_lm, set_answer_prefix
        h = load_lm({"model_id": TINY_LM, "device": "cpu", "load_in_4bit": False})
    except Exception as e:
        pytest.skip(f"tiny model unavailable: {e}")
    plain = list(set_answer_prefix(h, ""))
    spaced = list(set_answer_prefix(h, "Answer:"))
    toks = h.tokenizer.convert_ids_to_tokens(spaced)
    assert plain != spaced and all(t.startswith(("Ġ", " ", "▁")) for t in toks)
