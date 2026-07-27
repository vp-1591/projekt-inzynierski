"""Tests for backend/app/training/orchestrator.py — MLOpsOrchestrator.

Covers state transitions, progress updates, finish_training_and_evaluate,
reset_candidate_state, read_baseline_metrics, _monitor_training_process, and
deploy_new_adapter — all without a real database or subprocess.
"""

import asyncio
import threading
from unittest.mock import MagicMock, mock_open, patch

import pytest
from conftest import DummyDB

from app.training import orchestrator as orchestrator_module  # noqa: E402
from app.training.orchestrator import MLOpsOrchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakePopen:
    """Mimics subprocess.Popen just enough for start_manual_training tests."""

    def __init__(self, *args, **kwargs):
        self.pid = 9999
        self.returncode = None  # None means "still running"

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class CrashingPopen:
    """Subprocess that exits with a non-zero returncode."""

    def __init__(self, *args, **kwargs):
        self.pid = 8888
        self.returncode = 1

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def db():
    return DummyDB()


@pytest.fixture
def orch(db):
    return MLOpsOrchestrator(db=db)


# ===========================================================================
# State transitions — start_manual_training
# ===========================================================================


@pytest.mark.parametrize("initial_status", ["idle", "ready_to_promote", "deployment_success", "deployment_error"])
@patch("app.training.orchestrator.os.makedirs", MagicMock())
@patch("builtins.open", mock_open())
def test_start_manual_training_from_startable_statuses(initial_status, orch):
    """start_manual_training accepts all STARTABLE_STATUSES and sets status='training'."""
    orch.status = initial_status

    fake_proc = FakePopen()
    with patch.object(orchestrator_module.subprocess, "Popen", return_value=fake_proc):
        result = orch.start_manual_training("/data/train.jsonl")

    assert result is True
    assert orch.status == "training"


def test_start_manual_training_rejects_when_training(orch):
    """start_manual_training returns False when status='training'."""
    orch.status = "training"
    result = orch.start_manual_training("/data/train.jsonl")
    assert result is False
    assert orch.status == "training"


def test_start_manual_training_rejects_when_evaluating(orch):
    """start_manual_training returns False when status='evaluating'."""
    orch.status = "evaluating"
    result = orch.start_manual_training("/data/train.jsonl")
    assert result is False
    assert orch.status == "evaluating"


def test_start_manual_training_rejects_when_deploying(orch):
    """start_manual_training returns False when status='deploying'."""
    orch.status = "deploying"
    result = orch.start_manual_training("/data/train.jsonl")
    assert result is False
    assert orch.status == "deploying"


@patch("app.training.orchestrator.os.makedirs", MagicMock())
@patch("builtins.open", mock_open())
def test_start_manual_training_calls_popen(orch):
    """Verify subprocess.Popen is invoked with a command list including the trainer module."""
    orch.status = "idle"
    fake_proc = FakePopen()
    with patch.object(orchestrator_module.subprocess, "Popen", return_value=fake_proc) as mock_popen:
        orch.start_manual_training("/data/train.jsonl")

    mock_popen.assert_called_once()
    call_args = mock_popen.call_args
    cmd = call_args[0][0]
    assert isinstance(cmd, list)
    assert "app.training.trainer" in cmd
    assert "--data" in cmd


@patch("app.training.orchestrator.os.makedirs", MagicMock())
@patch("builtins.open", mock_open())
def test_start_manual_training_resets_candidate_state(orch):
    """Starting training from a deployment_success state resets candidate fields."""
    orch.status = "deployment_success"
    orch.training_progress = 100
    orch.evaluation_progress = 100
    orch.new_f1_non_empty = 0.42
    orch.new_exact_match = 0.24
    orch.latest_adapter_path = "/old/adapter"
    orch.current_run_id = 99

    fake_proc = FakePopen()
    with patch.object(orchestrator_module.subprocess, "Popen", return_value=fake_proc):
        result = orch.start_manual_training("/data/train.jsonl")

    assert result is True
    assert orch.training_progress == 0
    assert orch.evaluation_progress == 0
    assert orch.new_f1_non_empty == 0.0
    assert orch.new_exact_match == 0.0
    assert orch.latest_adapter_path is None
    assert orch.current_run_id is not None  # Assigned a new run ID


# ===========================================================================
# _monitor_training_process
# ===========================================================================


def test_monitor_training_process_detects_crash(db):
    """When the subprocess exits with non-zero returncode, status reverts to 'idle'."""
    orch = MLOpsOrchestrator(db=db)
    orch.status = "training"
    orch.training_progress = 50

    fake_proc = MagicMock()
    fake_proc.wait.return_value = 1
    fake_proc.returncode = 1
    mock_log_handle = MagicMock()

    orch._monitor_training_process(fake_proc, run_id=None, log_file="/tmp/test.log", log_handle=mock_log_handle)

    assert orch.status == "idle"
    assert orch.training_progress == 0
    mock_log_handle.close.assert_called_once()


def test_monitor_training_process_closes_log_handle(db):
    """The leaked log handle is closed regardless of process outcome."""
    orch = MLOpsOrchestrator(db=db)
    orch.status = "training"

    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    fake_proc.returncode = 0
    mock_log_handle = MagicMock()

    orch._monitor_training_process(fake_proc, run_id=None, log_file="/tmp/test.log", log_handle=mock_log_handle)

    mock_log_handle.close.assert_called_once()


def test_monitor_training_process_noop_on_success(db):
    """When the subprocess exits with returncode=0, status stays unchanged."""
    orch = MLOpsOrchestrator(db=db)
    orch.status = "training"
    orch.training_progress = 75

    fake_proc = MagicMock()
    fake_proc.wait.return_value = 0
    fake_proc.returncode = 0
    mock_log_handle = MagicMock()

    orch._monitor_training_process(fake_proc, run_id=None, log_file="/tmp/test.log", log_handle=mock_log_handle)

    # Status should NOT change on success — the training callback or
    # finish_training_and_evaluate will handle the transition.
    assert orch.status == "training"
    assert orch.training_progress == 75


def test_monitor_training_process_marks_run_failed(db):
    """When the subprocess crashes, the TrainingRun is marked as 'failed'."""
    orch = MLOpsOrchestrator(db=db)
    orch.status = "training"

    # Create a TrainingRun manually
    from app.db.database import TrainingRun

    run = TrainingRun(status="running")
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id

    fake_proc = MagicMock()
    fake_proc.wait.return_value = 1
    fake_proc.returncode = 1
    mock_log_handle = MagicMock()

    orch._monitor_training_process(fake_proc, run_id=run_id, log_file="/tmp/test.log", log_handle=mock_log_handle)

    updated_run = db.query(TrainingRun).get(run_id)
    assert updated_run.status == "failed"
    assert updated_run.end_time is not None


# ===========================================================================
# update_progress
# ===========================================================================


def test_update_progress_training(orch):
    """update_progress with stage='training' sets training_progress."""
    with patch.object(orch, "notify", MagicMock()):
        orch.update_progress("training", 50)
    assert orch.training_progress == 50
    assert orch.evaluation_progress == 0  # Unchanged


def test_update_progress_evaluation(orch):
    """update_progress with stage='evaluation' sets evaluation_progress."""
    with patch.object(orch, "notify", MagicMock()):
        orch.update_progress("evaluation", 75)
    assert orch.evaluation_progress == 75
    assert orch.training_progress == 0  # Unchanged


# ===========================================================================
# finish_training_and_evaluate
# ===========================================================================


def test_finish_training_and_evaluate_transitions_to_evaluating(orch):
    """Calling finish_training_and_evaluate transitions status to 'evaluating' and sets training_progress=100."""
    orch.status = "training"
    orch.training_progress = 80

    with (
        patch.object(orch, "notify", MagicMock()),
        patch("threading.Thread") as MockThread,
    ):
        orch.finish_training_and_evaluate("/model/adapter")

    assert orch.status == "evaluating"
    assert orch.training_progress == 100
    assert orch.evaluation_progress == 0
    assert orch.latest_adapter_path == "/model/adapter"
    # The benchmark thread must have been spawned
    MockThread.assert_called_once()


def test_finish_training_and_evaluate_rejects_duplicate(orch):
    """Calling finish_training_and_evaluate when not 'training' is a no-op."""
    orch.status = "evaluating"

    with patch.object(orch, "notify", MagicMock()):
        orch.finish_training_and_evaluate("/model/adapter")

    assert orch.status == "evaluating"
    # Should not have started a new benchmark thread


# ===========================================================================
# reset_candidate_state
# ===========================================================================


def test_reset_candidate_state_clears_fields(orch):
    """reset_candidate_state zeroes all per-run candidate fields."""
    orch.training_progress = 90
    orch.evaluation_progress = 50
    orch.new_f1_non_empty = 0.65
    orch.new_exact_match = 0.40
    orch.latest_adapter_path = "/some/path"
    orch.current_run_id = 99

    orch.reset_candidate_state()

    assert orch.training_progress == 0
    assert orch.evaluation_progress == 0
    assert orch.new_f1_non_empty == 0.0
    assert orch.new_exact_match == 0.0
    assert orch.latest_adapter_path is None
    assert orch.current_run_id is None


# ===========================================================================
# read_baseline_metrics
# ===========================================================================


@patch("app.training.orchestrator._project_root", return_value="/fake/root")
def test_file_not_found_returns_zeros(mock_root, orch):
    """Missing baseline report file yields zeros."""
    with patch("builtins.open", side_effect=FileNotFoundError("no such file")):
        result = orch.read_baseline_metrics()

    assert result == {"f1": 0.0, "em": 0.0}


@patch("app.training.orchestrator._project_root", return_value="/fake/root")
def test_malformed_content_returns_zeros(mock_root, orch):
    """A report file with no recognisable patterns yields zeros."""
    bad_content = "This is not a valid benchmark report.\nNo metrics here.\n"
    m = mock_open(read_data=bad_content)
    with patch("builtins.open", m):
        result = orch.read_baseline_metrics()

    assert result == {"f1": 0.0, "em": 0.0}


@patch("app.training.orchestrator._project_root", return_value="/fake/root")
def test_parses_document_level_report(mock_root, orch):
    """Parses 'Mean Document-Level F1 (excluding empty gold-label docs): 0.2847'."""
    content = "Exact-Match Accuracy: 0.1234\nMean Document-Level F1 (excluding empty gold-label docs): 0.2847\n"
    m = mock_open(read_data=content)
    with patch("builtins.open", m):
        result = orch.read_baseline_metrics()

    assert result["f1"] == pytest.approx(0.2847)
    assert result["em"] == pytest.approx(0.1234)


@patch("app.training.orchestrator._project_root", return_value="/fake/root")
def test_parses_current_benchmark_report(mock_root, orch):
    """Parses the fallback 'Mean F1 (Non-empty gold docs): 0.4912' pattern."""
    content = "Exact-Match Accuracy: 0.2100\nMean F1 (Non-empty gold docs): 0.4912\n"
    m = mock_open(read_data=content)
    with patch("builtins.open", m):
        result = orch.read_baseline_metrics()

    assert result["f1"] == pytest.approx(0.4912)
    assert result["em"] == pytest.approx(0.2100)


@patch("app.training.orchestrator._project_root", return_value="/fake/root")
def test_document_level_f1_takes_priority_over_non_empty(mock_root, orch):
    """When both F1 patterns are present, the document-level one wins."""
    content = (
        "Exact-Match Accuracy: 0.30\n"
        "Mean Document-Level F1 (excluding empty gold-label docs): 0.55\n"
        "Mean F1 (Non-empty gold docs): 0.44\n"
    )
    m = mock_open(read_data=content)
    with patch("builtins.open", m):
        result = orch.read_baseline_metrics()

    assert result["f1"] == pytest.approx(0.55)


# ===========================================================================
# Thread safety
# ===========================================================================


def test_concurrent_status_updates_are_consistent(orch):
    """Multiple threads calling update_progress don't corrupt state."""
    errors = []

    def update_training():
        try:
            for i in range(50):
                orch.update_progress("training", i)
        except Exception as e:
            errors.append(e)

    def update_evaluation():
        try:
            for i in range(50):
                orch.update_progress("evaluation", i)
        except Exception as e:
            errors.append(e)

    with patch.object(orch, "notify", MagicMock()):
        threads = [threading.Thread(target=update_training), threading.Thread(target=update_evaluation)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert errors == []
    # The final values should be one of the values written by the threads
    assert orch.training_progress in range(50)
    assert orch.evaluation_progress in range(50)


# ===========================================================================
# deploy_new_adapter
# ===========================================================================


@patch("app.training.orchestrator.os.makedirs", MagicMock())
@patch("builtins.open", mock_open())
def test_deploy_conversion_failure(orch):
    """When the converter subprocess fails, status reverts to 'ready_to_promote'."""
    orch.status = "ready_to_promote"

    # Mock async subprocess: readline returns empty bytes immediately (EOF),
    # wait() is an async no-op
    mock_proc = MagicMock()
    mock_proc.stdout = MagicMock()

    async def _readline_empty():
        return b""

    async def _wait():
        pass

    mock_proc.stdout.readline = _readline_empty
    mock_proc.wait = _wait
    mock_proc.returncode = 1  # Non-zero → conversion failure

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        patch.object(orch, "notify", MagicMock()),
    ):
        result = asyncio.run(orch.deploy_new_adapter("/model/latest"))

    assert result is False
    assert orch.status == "ready_to_promote"
    assert orch.last_deployment_status == "deployment_error"


@patch("app.training.orchestrator.os.makedirs", MagicMock())
@patch("builtins.open", mock_open())
def test_deploy_no_gguf_file(orch):
    """When no GGUF file is found after conversion, status reverts to 'ready_to_promote'."""
    orch.status = "ready_to_promote"

    mock_proc = MagicMock()
    mock_proc.stdout = MagicMock()

    async def _readline_empty():
        return b""

    async def _wait():
        pass

    mock_proc.stdout.readline = _readline_empty
    mock_proc.wait = _wait
    mock_proc.returncode = 0  # Conversion succeeded

    with (
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
        patch("app.training.orchestrator.os.listdir", return_value=[]),
        patch("app.training.orchestrator.os.path.isfile", return_value=False),
        patch.object(orch, "notify", MagicMock()),
    ):
        result = asyncio.run(orch.deploy_new_adapter("/model/latest"))

    assert result is False
    assert orch.status == "ready_to_promote"
    assert orch.last_deployment_status == "deployment_error"
