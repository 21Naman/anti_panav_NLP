from pathlib import Path

from typer.testing import CliRunner

from liars.cli import app
from liars.config import load_config
from liars.sizing import plan, qlora_hours

ROOT = Path(__file__).resolve().parents[1]
S = load_config(ROOT / "configs" / "sizing.yaml")["sizing"]


def test_model_reproduces_the_bench_qlora_run():
    m = dict(S["models"]["qlora_12b"], pred_chats_per_min=51.1)  # bench run: TTA 2, one GPU
    h = qlora_hours(m, 2000, 1000, 318, 2, S, gpus=1)
    assert abs(h - 2.58) / 2.58 < 0.10


def test_rules_escalate_with_size():
    small = "\n".join(plan(2000, 1000, 650, S))
    assert "QLoRA -> QLoRA 12B, 2 epoch(s)" in small
    mid = "\n".join(plan(5000, 3000, 650, S))
    assert "QLoRA -> QLoRA 12B, 1 epoch(s)" in mid
    big = "\n".join(plan(20000, 5000, 650, S))
    assert "subsample" in big and "train.subsample=" in big
    long = "\n".join(plan(20000, 5000, 2500, S, chars_p90=3500))
    assert "truncated" in long


def test_plan_cli():
    out = CliRunner().invoke(app, ["plan", "--train", "5000", "--test", "3000", "--chars", "650"]).output
    assert "TOTAL planned GPU" in out and "DeBERTa" in out
