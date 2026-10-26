"""`liars` command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from liars.schema import load_jsonl, validate_file
from liars.stats import compute_stats, format_stats

app = typer.Typer(
    help="Liar's Chatroom pipeline. Every command reads/writes the canonical JSONL schema.",
    no_args_is_help=True,
    add_completion=False,
)

InPath = typer.Option(..., "--in", exists=True, dir_okay=False, help="Canonical JSONL file.")


@app.command()
def validate(
    in_path: Path = InPath,
    max_show: int = typer.Option(20, help="How many errors to print."),
) -> None:
    """Check a canonical JSONL file against the schema."""
    report = validate_file(in_path)
    typer.echo(f"{in_path}: {report.n_examples} examples, {len(report.errors)} errors")
    for err in report.errors[:max_show]:
        typer.echo(f"  {err}")
    if len(report.errors) > max_show:
        typer.echo(f"  ... and {len(report.errors) - max_show} more")
    if report.errors:
        raise typer.Exit(code=1)


@app.command()
def stats(in_path: Path = InPath) -> None:
    """Counts, label distribution and length statistics for a canonical file."""
    examples = load_jsonl(in_path)
    if not examples:
        typer.echo(f"{in_path}: no examples")
        raise typer.Exit(code=1)
    typer.echo(f"file: {in_path}")
    typer.echo(format_stats(compute_stats(examples)))


# Commands that later phases implement. They accept any arguments and exit with
# code 2 so scripts never mistake a placeholder for a successful run.
PLACEHOLDERS: dict[str, tuple[int, str]] = {
    "synth": (1, "Generate verified synthetic chats."),
    "adapt": (9, "Convert raw competition files to canonical JSONL."),
    "folds": (2, "Assign stratified CV folds."),
    "leakcheck": (2, "Look for label leaks and generator quirks."),
    "train": (2, "Train a model from a config (kfold or holdout)."),
    "predict": (2, "Predict with a trained run."),
    "score": (2, "Score predictions against hidden labels."),
    "rules": (8, "Compute rule features."),
    "rules-report": (8, "Coverage and precision of each rule."),
    "ensemble": (8, "Combine runs (mean / weighted / stack)."),
    "analyze": (8, "HTML error-analysis report for a run."),
    "pseudo": (10, "Pseudo-label confident test predictions (off by default)."),
    "submit": (2, "Write a validated submission CSV."),
}


def _add_placeholder(name: str, phase: int, summary: str) -> None:
    @app.command(
        name,
        help=f"(todo: Phase {phase}) {summary}",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def _placeholder(ctx: typer.Context) -> None:
        typer.echo(f"`liars {name}` is not implemented yet (Phase {phase}).", err=True)
        raise typer.Exit(code=2)


for _name, (_phase, _summary) in PLACEHOLDERS.items():
    _add_placeholder(_name, _phase, _summary)


if __name__ == "__main__":
    app()
