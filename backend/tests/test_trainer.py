"""Tests for ProgressCallback in app.training.trainer.

ModelTrainer depends on GPU/Unsloth and is not unit-testable here;
only ProgressCallback is covered.
"""

import sys
from unittest.mock import MagicMock, patch

# The trainer module imports unsloth, torch, datasets, trl, transformers at the
# top level.  Mock those heavy / GPU-only dependencies *before* importing the
# module under test, so Python can resolve the import statements without needing
# the real packages installed.

# ---- module-level mocks ----
# Use setdefault so a real package (e.g. `requests`) already in sys.modules is
# not overwritten.

_mock_modules = [
    "unsloth",
    "unsloth.chat_templates",
    "torch",
    "datasets",
    "trl",
    "transformers",
]

for _name in _mock_modules:
    sys.modules.setdefault(_name, MagicMock())

# ---- make TrainerCallback a real class so ProgressCallback can inherit ----
# MagicMock *instances* cannot be used as base classes — the metaclass machinery
# raises StopIteration.  We patch the attribute on the already-registered mock
# module so that `from transformers import TrainerCallback` yields a usable
# class.


class _FakeTrainerCallback:
    """Minimal stand-in for transformers.TrainerCallback."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass


sys.modules["transformers"].TrainerCallback = _FakeTrainerCallback

# Now we can safely import the module under test.
from app.training.trainer import ProgressCallback  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(global_step: int, max_steps: int) -> MagicMock:
    """Create a mock TrainerState with the given step counters."""
    state = MagicMock()
    state.global_step = global_step
    state.max_steps = max_steps
    return state


def _make_control() -> MagicMock:
    """Create a mock TrainerControl (unused but required by the callback signature)."""
    return MagicMock()


def _make_args() -> MagicMock:
    """Create a mock TrainingArguments (unused but required by the callback signature)."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_progress_callback_posts_progress():
    """ProgressCallback posts the correct JSON payload to the backend."""
    state = _make_state(global_step=25, max_steps=100)
    callback = ProgressCallback(backend_url="http://localhost:8000")

    with patch("app.training.trainer.requests") as mock_requests:
        callback.on_log(_make_args(), state, _make_control())

    mock_requests.post.assert_called_once_with(
        "http://localhost:8000/training/progress",
        json={"stage": "training", "value": 25},
        timeout=1,
    )


def test_progress_callback_handles_failure_gracefully():
    """If requests.post raises, the exception is suppressed (printed to stderr)."""
    state = _make_state(global_step=50, max_steps=100)
    callback = ProgressCallback(backend_url="http://localhost:8000")

    with patch("app.training.trainer.requests") as mock_requests:
        mock_requests.post.side_effect = ConnectionError("refused")
        # Should NOT raise -- the callback swallows the error.
        callback.on_log(_make_args(), state, _make_control())

    mock_requests.post.assert_called_once()


def test_progress_callback_skips_zero_max_steps():
    """When max_steps == 0, on_log must NOT make a POST request (division guard)."""
    state = _make_state(global_step=5, max_steps=0)
    callback = ProgressCallback(backend_url="http://localhost:8000")

    with patch("app.training.trainer.requests") as mock_requests:
        callback.on_log(_make_args(), state, _make_control())

    mock_requests.post.assert_not_called()


def test_progress_callback_calculates_percentage():
    """global_step=5, max_steps=10 should yield progress=50."""
    state = _make_state(global_step=5, max_steps=10)
    callback = ProgressCallback(backend_url="http://backend:8000")

    with patch("app.training.trainer.requests") as mock_requests:
        callback.on_log(_make_args(), state, _make_control())

    call_kwargs = mock_requests.post.call_args
    assert call_kwargs[1]["json"]["value"] == 50


def test_progress_callback_rounds_to_int():
    """Progress value must be an int even when the division is not whole."""
    state = _make_state(global_step=1, max_steps=3)
    callback = ProgressCallback(backend_url="http://localhost:8000")

    with patch("app.training.trainer.requests") as mock_requests:
        callback.on_log(_make_args(), state, _make_control())

    call_kwargs = mock_requests.post.call_args
    progress = call_kwargs[1]["json"]["value"]
    assert isinstance(progress, int)
    # int(33.33...) == 33
    assert progress == 33


def test_progress_callback_zero_progress():
    """global_step=0, max_steps=100 should yield progress=0."""
    state = _make_state(global_step=0, max_steps=100)
    callback = ProgressCallback(backend_url="http://localhost:8000")

    with patch("app.training.trainer.requests") as mock_requests:
        callback.on_log(_make_args(), state, _make_control())

    call_kwargs = mock_requests.post.call_args
    assert call_kwargs[1]["json"]["value"] == 0


def test_progress_callback_full_progress():
    """global_step=100, max_steps=100 should yield progress=100."""
    state = _make_state(global_step=100, max_steps=100)
    callback = ProgressCallback(backend_url="http://localhost:8000")

    with patch("app.training.trainer.requests") as mock_requests:
        callback.on_log(_make_args(), state, _make_control())

    call_kwargs = mock_requests.post.call_args
    assert call_kwargs[1]["json"]["value"] == 100
