from unittest.mock import mock_open, patch

from app.training import orchestrator as orchestrator_module
from app.training.orchestrator import MLOpsOrchestrator, BACKEND_URL


class DummyDB:
    def add(self, obj):
        self.obj = obj

    def commit(self):
        pass

    def refresh(self, obj):
        obj.id = 7


def test_read_baseline_metrics_supports_document_level_report(monkeypatch):
    content = "\n".join(
        [
            "Mean Document-Level F1 (excluding empty gold-label docs): 0.2847",
            "Exact-Match Accuracy: 0.7318 (1113/1521)",
        ]
    )
    monkeypatch.setattr(orchestrator_module, "_project_root", lambda: "C:/project")

    with patch("builtins.open", mock_open(read_data=content)):
        metrics = MLOpsOrchestrator(db=None).read_baseline_metrics()

    assert metrics == {"f1": 0.2847, "em": 0.7318}


def test_read_baseline_metrics_supports_current_benchmark_report(monkeypatch):
    content = "\n".join(
        [
            "Mean F1 (Non-empty gold docs): 0.4912",
            "Exact-Match Accuracy: 0.7291 (1109/1521)",
        ]
    )
    monkeypatch.setattr(orchestrator_module, "_project_root", lambda: "C:/project")

    with patch("builtins.open", mock_open(read_data=content)):
        metrics = MLOpsOrchestrator(db=None).read_baseline_metrics()

    assert metrics == {"f1": 0.4912, "em": 0.7291}


def test_start_manual_training_after_deployment_resets_candidate_state(monkeypatch):
    captured = {}

    class FakePopen:
        pid = 1234

        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr(orchestrator_module.os, "makedirs", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator_module.subprocess, "Popen", FakePopen)

    orchestrator = MLOpsOrchestrator(DummyDB())
    orchestrator.status = "deployment_success"
    orchestrator.training_progress = 100
    orchestrator.evaluation_progress = 100
    orchestrator.new_f1_non_empty = 0.42
    orchestrator.new_exact_match = 0.24
    orchestrator.latest_adapter_path = "model/latest"
    orchestrator.deployed_adapter_path = "model/deployed"
    orchestrator.last_deployment_status = "deployment_success"

    with patch("builtins.open", mock_open()):
        assert orchestrator.start_manual_training("uploads/new.jsonl") is True

    assert orchestrator.status == "training"
    assert orchestrator.training_progress == 0
    assert orchestrator.evaluation_progress == 0
    assert orchestrator.new_f1_non_empty == 0.0
    assert orchestrator.new_exact_match == 0.0
    assert orchestrator.latest_adapter_path is None
    assert orchestrator.current_run_id == 7
    assert orchestrator.deployed_adapter_path == "model/deployed"
    assert orchestrator.last_deployment_status == "deployment_success"
    # After refactoring, cmd is a list (not a shell string)
    assert isinstance(captured["cmd"], list)
    assert "--data" in captured["cmd"]
    assert "uploads/new.jsonl" in captured["cmd"]
