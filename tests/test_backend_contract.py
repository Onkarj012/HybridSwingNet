from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app


def test_backend_openapi_contains_expected_paths() -> None:
    schema = app.openapi()
    paths = schema["paths"]

    assert "/" in paths
    assert "/api/health" in paths
    assert "/api/dashboard" in paths
    assert "/api/recommendations" in paths
    assert "/api/stocks/{ticker}" in paths
    assert "/api/stocks/{ticker}/chart" in paths
    assert "/api/metadata/config" in paths


def test_backend_openapi_exposes_recommendation_schema() -> None:
    schema = app.openapi()
    operation = schema["paths"]["/api/recommendations"]["get"]
    content = operation["responses"]["200"]["content"]["application/json"]["schema"]

    assert content["$ref"].endswith("/RecommendationsResponse")
