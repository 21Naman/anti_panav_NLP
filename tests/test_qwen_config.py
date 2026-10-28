from pathlib import Path

from liars.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


def test_qwen_config_differs_from_gemma_only_where_intended():
    gemma = _flat(load_config(ROOT / "configs" / "llm_qlora.yaml"))
    qwen = _flat(load_config(ROOT / "configs" / "llm_qlora_qwen.yaml"))
    assert set(gemma) == set(qwen)
    diff = {k for k in gemma if gemma[k] != qwen[k]}
    assert diff == {"llm.model_id", "prompt.answer_prefix", "qlora.micro_batch_size", "qlora.grad_accum"}
    assert qwen["llm.model_id"].startswith("Qwen/")
    assert qwen["prompt.answer_prefix"] == "Answer:"
    # same effective batch as gemma
    assert qwen["qlora.micro_batch_size"] * qwen["qlora.grad_accum"] == \
        gemma["qlora.micro_batch_size"] * gemma["qlora.grad_accum"]
    assert qwen["model.name"] == "llm_qlora"
