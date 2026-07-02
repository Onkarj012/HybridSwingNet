"""V4 reproducible reporting artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def config_hash(config_path: str | Path) -> str:
    data = Path(config_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def git_state(cwd: str | Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True).strip())
    except Exception:
        commit = "unknown"
        dirty = True
    return {"git_commit": commit, "dirty": dirty}


def write_v4_reports(
    report_dir: str | Path,
    replay: pd.DataFrame,
    scorecard_rows: list[dict[str, Any]],
    calibration: pd.DataFrame | None,
    failure_slices: pd.DataFrame | None,
    experiment: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scorecard_rows).to_csv(out / "oos_scorecard.csv", index=False)
    (failure_slices if failure_slices is not None else pd.DataFrame()).to_csv(out / "failure_slices.csv", index=False)
    (calibration if calibration is not None else pd.DataFrame()).to_csv(out / "calibration_by_regime.csv", index=False)
    replay.to_csv(out / "policy_replay.csv", index=False)
    (out / "experiment_card.json").write_text(json.dumps(experiment, indent=2, sort_keys=True))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        return yaml.safe_load(f) or {}
