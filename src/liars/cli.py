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
    pretty_exceptions_enable=False,  # short plain tracebacks
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


@app.command()
def folds(
    in_path: Path = InPath,
    k: int = typer.Option(5, "--k"),
    seed: int = typer.Option(42, "--seed"),
    out: Path = typer.Option(None, "--out", help="Default: data/canonical/folds_<stem>.csv"),
) -> None:
    """Assign stratified CV folds (by label, and meta.source when present)."""
    from collections import Counter

    from liars.cv import assign_folds, folds_path_for, save_folds
    from liars.paths import get_paths

    examples = load_jsonl(in_path)
    fold_map = assign_folds(examples, k, seed)
    path = out or folds_path_for(in_path, get_paths().canonical)
    save_folds(fold_map, path)
    label = {ex.id: ex.label for ex in examples}
    typer.echo(f"{len(fold_map)} examples, k={k}, seed={seed} -> {path}")
    for f in range(k):
        ids = [i for i, v in fold_map.items() if v == f]
        dist = Counter(label[i] for i in ids)
        typer.echo(f"  fold {f}: {len(ids):>6}  " + "  ".join(f"{s} {dist[s]}" for s in sorted(dist)))


@app.command()
def train(
    config: Path = typer.Option(..., "--config", exists=True, dir_okay=False),
    test: Path = typer.Option(None, "--test", help="Canonical test file (overrides data.test)."),
    debug: bool = typer.Option(False, "--debug", help="Tiny data/model on CPU; outputs to data/debug/."),
    stage: int = typer.Option(None, "--stage", help="Apply the config's stages.<N> block (e.g. 1 or 2)."),
    set_: list[str] = SetOpt,
) -> None:
    """Train a model from a config (train.mode: kfold or holdout)."""
    from liars.trainer import run_training

    cfg = load_config(config, set_)
    run_training(cfg, config, test, debug, log=typer.echo, stage=stage)


@app.command()
def score(
    preds: Path = typer.Option(..., "--preds", exists=True, dir_okay=False),
    labels: Path = typer.Option(..., "--labels", exists=True, dir_okay=False,
                                help="CSV id,label[,lie_type,...] or labelled canonical JSONL."),
) -> None:
    """Accuracy of a prediction file against known labels (synthetic / train only)."""
    from liars.preds import load_preds
    from liars.score import format_score, load_labels
    from liars.score import score as score_fn

    typer.echo(format_score(score_fn(load_preds(preds), load_labels(labels))))


@app.command()
def leakcheck(
    in_path: Path = InPath,
    config: Path = typer.Option(Path("configs/leakcheck.yaml"), "--config"),
    set_: list[str] = SetOpt,
) -> None:
    """Look for label skews, liar-only templates and other generator quirks (TRAIN only)."""
    from liars.leakcheck import run_leakcheck
    from liars.paths import get_paths

    cfg = load_config(config, set_)
    model_cfg = load_config(cfg["baseline_config"])
    examples = load_jsonl(in_path)
    md, summary = run_leakcheck(examples, cfg["leakcheck"], model_cfg, cfg["seed"], in_path.stem)
    out = get_paths(cfg).reports / f"leakcheck_{in_path.stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    typer.echo(f"{summary['n']} chats. TF-IDF OOF accuracy {summary['tfidf_oof_acc']:.4f}; "
               f"liar-only templates: {summary['liar_only_templates']}")
    for rule, acc in summary["rules"].items():
        typer.echo(f"  rule {rule:<26} {acc:.3f}")
    typer.echo("ALERTS:" if summary["alerts"] else "ALERTS: none")
    for a in summary["alerts"]:
        typer.echo(f"  - {a}")
    typer.echo(f"report -> {out}")


@app.command()
def submit(
    preds: Path = typer.Option(..., "--preds", exists=True, dir_okay=False),
    out: Path = typer.Option(..., "--out", help="e.g. data/subs/NAME.csv"),
    test: Path = typer.Option(None, "--test", help="Canonical test file (default: config)."),
    sample: Path = typer.Option(None, "--sample", help="sample_submission.csv to match."),
    config: Path = typer.Option(Path("configs/submission.yaml"), "--config"),
    set_: list[str] = SetOpt,
) -> None:
    """Write a validated submission CSV."""
    import pandas as pd

    from liars.preds import load_preds
    from liars.submit import make_submission, validate_submission, write_submission

    cfg = load_config(config, set_)
    scfg = cfg["submission"]
    test_examples = load_jsonl(test or scfg["test"])
    sample_df = pd.read_csv(sample, dtype={scfg["id_column"]: str}) if sample else None
    sub = make_submission(load_preds(preds), test_examples, scfg, sample_df)
    errors = validate_submission(sub, test_examples, scfg, sample_df)
    if errors:
        for e in errors:
            typer.echo(f"  INVALID: {e}")
        raise typer.Exit(code=1)
    path = write_submission(sub, out)
    dist = sub[scfg["label_column"]].value_counts().sort_index()
    typer.echo(f"valid submission: {len(sub)} rows, columns {list(sub.columns)} -> {path}")
    typer.echo("  predicted label counts: " + "  ".join(f"{k} {v}" for k, v in dist.items()))


@app.command()
def predict(
    run: str = typer.Option(..., "--run", help="Run name (uses data/runs/<run>.yaml + data/models/<run>/)."),
    in_path: Path = InPath,
    model_dir: Path = typer.Option(None, "--model-dir", help="Default: data/models/<run>/full or fold0."),
    out: Path = typer.Option(None, "--out", help="Default: data/preds/<run>__<input stem>.parquet"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Where data/runs/<run>.yaml lives."),
    set_: list[str] = SetOpt,
) -> None:
    """Predict a canonical file with a saved model (e.g. QLoRA adapters)."""
    import yaml

    from liars.config import apply_overrides
    from liars.models import get_model_class
    from liars.paths import get_paths
    from liars.preds import save_preds
    from liars.trainer import resolve_model_dir

    paths = get_paths({"paths": {"data_dir": str(data_dir)}})
    cfg = apply_overrides(yaml.safe_load(open(paths.run_configs / f"{run}.yaml")), set_)
    cls = get_model_class(cfg["model"]["name"])
    if not hasattr(cls, "load"):
        typer.echo(f"model {cfg['model']['name']!r} cannot be reloaded; re-run `liars train` with --test")
        raise typer.Exit(code=1)
    mdir = model_dir or resolve_model_dir(get_paths(cfg), run)
    preds = cls.load(mdir, cfg).predict_proba(load_jsonl(in_path))
    path = out or get_paths(cfg).preds / f"{run}__{in_path.stem}.parquet"
    save_preds(preds, path)
    typer.echo(f"{len(preds)} predictions from {mdir} -> {path}")


# Commands that later phases implement. They accept any arguments and exit with
# code 2 so scripts never mistake a placeholder for a successful run.
PLACEHOLDERS: dict[str, tuple[int, str]] = {
    "adapt": (9, "Convert raw competition files to canonical JSONL."),
    "rules": (8, "Compute rule features."),
    "rules-report": (8, "Coverage and precision of each rule."),
    "ensemble": (8, "Combine runs (mean / weighted / stack)."),
    "analyze": (8, "HTML error-analysis report for a run."),
    "pseudo": (10, "Pseudo-label confident test predictions (off by default)."),
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
