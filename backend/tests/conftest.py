import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests that require live local services such as WSL, Ollama, and uvicorn",
    )


def pytest_ignore_collect(collection_path, config):
    if collection_path.name.startswith(".") or collection_path.name.startswith("pytest-cache-files-"):
        return True
    if config.getoption("--run-integration"):
        return False
    return collection_path.name.startswith("test_integration_")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="integration test; run with --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
