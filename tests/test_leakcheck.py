from pathlib import Path

from liars.config import load_config
from liars.leakcheck import mask_line, run_leakcheck
from liars.schema import Message
from liars.synth.pack import load_pack
from liars.synth.programmatic import generate_many

ROOT = Path(__file__).resolve().parents[1]
LCFG = load_config(ROOT / "configs" / "leakcheck.yaml", ["leakcheck.k=3"])
MODEL_CFG = load_config(ROOT / "configs" / "baselines.yaml")
SCFG = load_config(ROOT / "configs" / "synth.yaml")["synth"]


def _data(n=240):
    return [r.example for r in generate_many(n, 5, "med", load_pack("default"), SCFG)]


def test_clean_synthetic_has_no_template_leak():
    md, summary = run_leakcheck(_data(), LCFG["leakcheck"], MODEL_CFG, 0, "clean")
    assert summary["liar_only_templates"] == 0
    assert "## 5. TF-IDF baseline" in md


def test_planted_quirk_is_found():
    data = _data()
    for ex in data:
        i = ex.meta["lie_msg_idx"][-1]
        ex.messages[i] = Message(ex.messages[i].speaker, "for real though, " + ex.messages[i].text)
    md, summary = run_leakcheck(data, LCFG["leakcheck"], MODEL_CFG, 0, "planted")
    assert summary["tfidf_oof_acc"] > 0.35
    assert any("TF-IDF" in a for a in summary["alerts"])
    assert "for real" in md


def test_mask_line():
    freq = {"concert": 0.05, "friday": 0.1, "the": 0.9}
    assert mask_line("The concert is on Friday at 7 PM", freq, 0.2) == "the <E> is on <E> at <NUM> pm"
