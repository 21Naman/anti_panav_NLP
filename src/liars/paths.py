"""Where outputs go. Everything lives under `paths.data_dir` (default "data")."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = "data"


@dataclass(frozen=True)
class Paths:
    data: Path

    @property
    def oof(self) -> Path:
        return self.data / "oof"

    @property
    def preds(self) -> Path:
        return self.data / "preds"

    @property
    def runs_csv(self) -> Path:
        return self.data / "runs.csv"

    @property
    def run_configs(self) -> Path:
        return self.data / "runs"

    @property
    def reports(self) -> Path:
        return self.data / "reports"

    @property
    def models(self) -> Path:
        return self.data / "models"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def canonical(self) -> Path:
        return self.data / "canonical"

    @property
    def subs(self) -> Path:
        return self.data / "subs"

    @property
    def features(self) -> Path:
        return self.data / "features"


def get_paths(cfg: dict | None = None) -> Paths:
    data_dir = ((cfg or {}).get("paths") or {}).get("data_dir") or DEFAULT_DATA_DIR
    return Paths(Path(data_dir))
