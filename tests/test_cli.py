import json

from typer.testing import CliRunner

from liars.cli import PLACEHOLDERS, app

runner = CliRunner()


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"], terminal_width=200)
    assert result.exit_code == 0
    for name in ["validate", "stats", *PLACEHOLDERS]:
        assert name in result.output, name


def test_validate_fixture(tiny_path):
    result = runner.invoke(app, ["validate", "--in", str(tiny_path)])
    assert result.exit_code == 0, result.output
    assert "32 examples, 0 errors" in result.output


def test_validate_bad_file(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "x", "evidence": [], "messages": [], "speakers": []}) + "\n")
    result = runner.invoke(app, ["validate", "--in", str(p)])
    assert result.exit_code == 1
    assert "errors" in result.output


def test_stats_fixture(tiny_path):
    result = runner.invoke(app, ["stats", "--in", str(tiny_path)])
    assert result.exit_code == 0, result.output
    assert "examples: 32" in result.output
    assert "A 8 (25.0%)" in result.output
    assert "p99" in result.output


def test_placeholder_exits_nonzero():
    result = runner.invoke(app, ["predict", "--run", "x"])
    assert result.exit_code == 2
