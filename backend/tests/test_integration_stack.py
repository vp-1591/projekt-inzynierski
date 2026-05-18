import asyncio
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.training import orchestrator as orchestrator_module
from app.training.orchestrator import MLOpsOrchestrator, _get_wsl_host_ip, _to_wsl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
WSL_PYTHON = _to_wsl(str(BACKEND_ROOT / ".venv-wsl" / "bin" / "python"))


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _require_wsl():
    if shutil.which("wsl") is None:
        pytest.skip("wsl CLI is not available")

    project_wsl_path = _to_wsl(str(PROJECT_ROOT))
    check = subprocess.run(
        [
            "wsl",
            "--",
            "bash",
            "-lc",
            f"command -v bash >/dev/null && test -d '{project_wsl_path}' && test -x '{WSL_PYTHON}'",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if check.returncode != 0:
        pytest.skip(
            "WSL distro, bash, project mount, or backend/.venv-wsl/bin/python is unavailable"
        )


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


@pytest.mark.integration
@pytest.mark.wsl
def test_wsl_readiness_and_host_ip_resolution():
    _require_wsl()

    host_ip = _get_wsl_host_ip()

    assert host_ip
    assert "\x00" not in host_ip

    probe = subprocess.run(
        [
            "wsl",
            "--exec",
            WSL_PYTHON,
            "-c",
            "import sys; print(sys.version_info[0])",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert probe.returncode == 0
    assert probe.stdout.strip() == "3"


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


async def _post_progress_from_wsl(host_ip, port, stage, value):
    payload = json.dumps({"stage": stage, "value": value})
    code = (
        "import urllib.request; "
        f"data={payload!r}.encode('utf-8'); "
        "req=urllib.request.Request("
        f"'http://{host_ip}:{port}/training/progress', "
        "data=data, headers={'Content-Type':'application/json'}, method='POST'); "
        "print(urllib.request.urlopen(req, timeout=5).read().decode('utf-8'))"
    )
    result = subprocess.run(
        ["wsl", "--exec", WSL_PYTHON, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert '"ok"' in result.stdout


async def _expect_websocket_progress(port, stage, value):
    import websockets

    progress_field = f"{stage}_progress"
    async with websockets.connect(
        f"ws://127.0.0.1:{port}/ws/training/status"
    ) as websocket:
        initial = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))
        assert "status" in initial

        host_ip = _get_wsl_host_ip()
        await _post_progress_from_wsl(host_ip, port, stage, value)

        deadline = time.time() + 8
        while time.time() < deadline:
            message = json.loads(await asyncio.wait_for(websocket.recv(), timeout=5))
            if message.get(progress_field) == value:
                return

        pytest.fail(f"websocket did not receive {progress_field}={value}")


@pytest.mark.integration
@pytest.mark.wsl
@pytest.mark.websocket
@pytest.mark.parametrize("stage", ["training", "evaluation"])
def test_wsl_progress_callback_reaches_live_websocket(live_uvicorn_server, stage):
    _require_wsl()
    _require_websockets()

    asyncio.run(_expect_websocket_progress(live_uvicorn_server, stage, 37))


async def _expect_ready_to_promote_broadcast(port):
    import websockets

    async with websockets.connect(
        f"ws://127.0.0.1:{port}/ws/training/status"
    ) as websocket:
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
@pytest.mark.wsl
def test_training_upload_launches_wsl_command_with_backend_callback(
    monkeypatch
):
    captured = {}

    class FakePopen:
        pid = 4242

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    orchestrator = MLOpsOrchestrator(DummyDB())
    monkeypatch.chdir(BACKEND_ROOT)
    monkeypatch.setattr(orchestrator_module, "_get_wsl_host_ip", lambda: "172.20.0.1")
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
    assert "wsl --exec bash -c" in captured["cmd"]
    assert "--data uploads/tiny.jsonl" in captured["cmd"]
    assert "--backend http://172.20.0.1:8000" in captured["cmd"]
