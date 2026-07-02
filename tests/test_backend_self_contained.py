from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.artifacts import ArtifactRegistry
from backend.app.core.cache import TTLCache
from backend.app.core.settings import get_settings
from backend.app.services.backend_service import StockXpertBackendService


def test_backend_has_no_runtime_repo_imports() -> None:
    forbidden = [
        re.compile(r"^\s*from\s+stockxpert\b", re.MULTILINE),
        re.compile(r"^\s*import\s+stockxpert\b", re.MULTILINE),
        re.compile(r"^\s*from\s+scripts\b", re.MULTILINE),
        re.compile(r"^\s*import\s+scripts\b", re.MULTILINE),
        re.compile(r"sys\.path"),
    ]

    for path in (ROOT / "backend").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not pattern.search(text), f"Forbidden runtime dependency in {path}: {pattern.pattern}"


def test_backend_health_payload_starts_from_local_bundle() -> None:
    settings = get_settings()
    registry = ArtifactRegistry(settings)
    service = StockXpertBackendService(settings=settings, artifacts=registry, cache=TTLCache())
    payload = service.health()

    assert payload["status"] == "ok"
    assert payload["artifacts"]["ready"] is True
