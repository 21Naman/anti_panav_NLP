"""`liars` command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from liars.config import load_config
from liars.schema import load_jsonl, validate_file
from liars.stats import compute_stats, format_stats

app = typer.Typer(
    help="Liar's Chatroom pipeline. Every command reads/writes the canonical JSONL schema.",
    no_args_is_help=True,
    add_completion=False,
)

InPath = typer.Option(..., "--in", exists=True, dir_okay=False, help="Canonical JSONL file.")
SetOpt = typer.Option([], "--set", help="Config override key=value (repeatable).")


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


@app.command()
def synth(
    n: int = typer.Option(..., "--n", help="Number of chats."),
    seed: int = typer.Option(..., "--seed"),
    difficulty: str = typer.Option("med", help="easy | med | hard | mixed | list like med,hard"),
    pack: str = typer.Option(None, help="Template pack name or YAML path (default: config)."),
    out: Path = typer.Option(..., "--out", help="Output canonical JSONL."),
    hide_labels: Path = typer.Option(
        None, "--hide-labels", help="Write labels to this CSV and null them in --out."
    ),
    config: Path = typer.Option(Path("configs/synth.yaml"), "--config"),
    set_: list[str] = SetOpt,
    id_prefix: str = typer.Option(None, help="Id prefix (default <pack>_s<seed>_)."),
    samples: bool = typer.Option(True, "--samples/--no-samples", help="Write readable samples."),
) -> None:
    """Generate verified synthetic chats."""
    import time

    from liars.adapters.synthetic import to_examples, write_samples, write_synthetic
    from liars.synth.pack import load_pack
    from liars.synth.programmatic import generate_many, liar_probs
    from liars.synth.verify import verify_records

    cfg = load_config(config, set_)
    scfg = cfg["synth"]
    pack_obj = load_pack(pack or scfg["pack"])
    t0 = time.time()
    records = generate_many(n, seed, difficulty, pack_obj, scfg, id_prefix)
    t_gen = time.time() - t0
    report = verify_records(records, liar_probs(scfg["liar_position"]), cfg["verify"])
    examples = to_examples(records)
    write_synthetic(examples, out, hide_labels)
    typer.echo(f"generated {n} chats (pack={pack_obj.name}, difficulty={difficulty}, "
               f"seed={seed}) in {t_gen:.1f}s -> {out}")
    if hide_labels:
        typer.echo(f"labels hidden -> {hide_labels}")
    typer.echo(report.summary())
    typer.echo(format_stats(compute_stats(examples)))
    file_report = validate_file(out)
    typer.echo(f"validate {out}: {len(file_report.errors)} errors")
    if samples:
        name = f"synth_{pack_obj.name}_{difficulty.replace(',', '+')}.txt"
        path = write_samples(examples, Path(scfg["samples_dir"]) / name, scfg["n_samples"], seed)
        typer.echo(f"samples -> {path}")
    typer.echo(f"total {time.time() - t0:.1f}s")
    if report.errors or file_report.errors:
        raise typer.Exit(code=1)


# Commands that later phases implement. They accept any arguments and exit with
# code 2 so scripts never mistake a placeholder for a successful run.
PLACEHOLDERS: dict[str, tuple[int, str]] = {
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
