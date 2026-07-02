"""Simple file-backed V3 model registry."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelRegistryRecord:
    version_id: str
    git_commit: str
    config_hash: str
    dataset_hash: str
    train_stocks: list[str] = field(default_factory=list)
    val_stocks: list[str] = field(default_factory=list)
    test_stocks: list[str] = field(default_factory=list)
    blind_stocks: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class ModelRegistry:
    def __init__(self, registry_dir: str | Path = "runs/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register(self, record: ModelRegistryRecord) -> Path:
        path = self.registry_dir / f"{record.version_id}.json"
        with path.open("w") as f:
            json.dump(asdict(record), f, indent=2, sort_keys=True)
        return path

    def load(self, version_id: str) -> ModelRegistryRecord:
        with (self.registry_dir / f"{version_id}.json").open("r") as f:
            return ModelRegistryRecord(**json.load(f))

