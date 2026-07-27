import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.training import orchestrator as orchestrator_module
from app.training.orchestrator import MLOpsOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _require_websockets():
    try:
        import websockets  # noqa: F401
    except ImportError:
        pytest.skip("websockets package is not installed")


class DummyDB:
    def add(self, obj):
        self.obj = obj

    def commit(self):
        pass

    def refresh(self, obj):
        obj.id = 1


@pytest.mark.integration
@pytest.mark.ollama
def test_real_ollama_model_smoke_via_analyze_endpoint():
    try:
        tags_response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        tags_response.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama is not reachable on localhost:11434: {exc}")

    models = tags_response.json().get("models", [])
    model_names = {model.get("name") for model in models}
    if main.MODEL_NAME not in model_names:
        pytest.skip(f"Ollama model {main.MODEL_NAME!r} is not installed")

    with TestClient(main.app) as client:
        response = client.post(
            "/analyze",
            json={"text": "To jest neutralne zdanie testowe."},
            timeout=130,
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data.get("reasoning"), str)
    assert isinstance(data.get("discovered_techniques"), list)


@pytest.fixture
def live_uvicorn_server():
    import uvicorn

    port = _free_port()
    main.orchestrator_instance = None
    config = uvicorn.Config(
        main.app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/training/status", timeout=1)
            if response.status_code == 200:
                break
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail(f"uvicorn server did not become ready: {last_error}")

    yield port

    server.should_exit = True
    thread.join(timeout=10)
    main.manager.active_connections.clear()
    main.orchestrator_instance = None


async def _expect_ready_to_promote_broadcast(port):
    import websockets

    async with websockets.connect(f"ws://127.0.0.1:{port}/ws/training/status") as websocket:
        await asyncio.wait_for(websocket.recv(), timeout=5)

        orchestrator = main.orchestrator_instance
        assert orchestrator is not None
        orchestrator.status = "ready_to_promote"
        orchestrator.new_f1_non_empty = 0.5
        orchestrator.new_exact_match = 0.25
        orchestrator.notify()

        message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))
        assert message["status"] == "ready_to_promote"
        assert message["new_f1_non_empty"] == 0.5
        assert message["new_exact_match"] == 0.25


@pytest.mark.integration
@pytest.mark.websocket
def test_ready_to_promote_notify_broadcasts_on_live_websocket(live_uvicorn_server):
    _require_websockets()

    asyncio.run(_expect_ready_to_promote_broadcast(live_uvicorn_server))


@pytest.mark.integration
def test_training_upload_launches_direct_python_invocation(monkeypatch):
    captured = {}

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            # Should NOT have shell=True
            assert not kwargs.get("shell")

    orchestrator = MLOpsOrchestrator(DummyDB())
    monkeypatch.chdir(BACKEND_ROOT)
    monkeypatch.setattr(orchestrator_module.subprocess, "Popen", FakePopen)
    main.app.dependency_overrides[main.get_orchestrator] = lambda: orchestrator

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/training/upload",
                files={
                    "file": (
                        "tiny.jsonl",
                        b'{"input":"tekst","output":"{\\"discovered_techniques\\":[]}"}\n',
                        "application/jsonl",
                    )
                },
            )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "started", "file": "tiny.jsonl"}
    assert orchestrator.status == "training"
    assert orchestrator.current_run_id == 1
    assert isinstance(captured["cmd"], list)
    assert "app.training.trainer" in captured["cmd"]
    assert "wsl" not in captured["cmd"]
    assert "--backend" in captured["cmd"]


@pytest.mark.integration
def test_training_command_uses_direct_python_invocation():
    """Verify that training uses direct subprocess, not wsl --exec."""
    captured = {}

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["shell"] = kwargs.get("shell", False)

    orchestrator = MLOpsOrchestrator(DummyDB())

    import unittest.mock as mock

    with mock.patch.object(orchestrator_module.subprocess, "Popen", FakePopen):
        orchestrator.start_manual_training("test_file.jsonl")

    # Assertions for Linux/Docker
    assert isinstance(captured["cmd"], list)
    assert "app.training.trainer" in captured["cmd"]
    assert "wsl" not in captured["cmd"]
    assert not captured["shell"]  # Should NOT use shell
