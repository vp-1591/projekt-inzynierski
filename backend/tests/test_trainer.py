"""Tests for ProgressCallback and format_training_example in app.training.trainer.

ModelTrainer depends on GPU/Unsloth and is not unit-testable here;
only ProgressCallback and the formatting function are covered.
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
    "requests",
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
from app.training.trainer import ProgressCallback, format_training_example  # noqa: E402

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


# ---------------------------------------------------------------------------
# Tests for format_training_example
# ---------------------------------------------------------------------------

_CHAT_TEMPLATE_OUTPUT = "<|im_start|>user\nTest input<|im_end|>\n<|im_start|>assistant\nTest output<|im_end|>"


def _make_mock_tokenizer(template_output=_CHAT_TEMPLATE_OUTPUT):
    """Create a mock tokenizer whose apply_chat_template returns the given string."""
    mock = MagicMock()
    mock.apply_chat_template.return_value = template_output
    return mock


class TestFormatTrainingExample:
    """Tests for the format_training_example function."""

    def test_returns_string_not_dict(self):
        """format_training_example must return a plain string, not a dict.

        TRL's SFTTrainer._prepare_dataset wraps the result in {"text": ...}
        automatically. If this function returns a dict, the wrapper produces
        {"text": {"text": "..."}}, which crashes add_eos with AttributeError.
        """
        mock_tokenizer = _make_mock_tokenizer()
        result = format_training_example(
            {"input": "Test input", "output": "Test output"},
            mock_tokenizer,
        )
        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

    def test_truncates_long_input(self):
        """Input longer than max_input_length should be truncated."""
        mock_tokenizer = _make_mock_tokenizer()
        long_input = "A" * 5000

        format_training_example(
            {"input": long_input, "output": "Output"},
            mock_tokenizer,
            max_input_length=100,
        )

        call_args = mock_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]  # first positional arg
        user_msg = messages[0]
        assert len(user_msg["content"]) == 100, "Input was not truncated to max_input_length"

    def test_preserves_short_input(self):
        """Input shorter than max_input_length should not be truncated."""
        mock_tokenizer = _make_mock_tokenizer()
        short_input = "Short input"

        format_training_example(
            {"input": short_input, "output": "Output"},
            mock_tokenizer,
            max_input_length=3500,
        )

        call_args = mock_tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        user_msg = messages[0]
        assert user_msg["content"] == short_input, "Short input was incorrectly truncated"

    def test_trl_wrapping_does_not_nest(self):
        """Simulate TRL's _prepare_dataset wrapping — must not produce nested dicts.

        TRL wraps the formatting function result in {"text": result}. If result
        is a dict, you get {"text": {"text": "..."}} which crashes add_eos.
        """
        mock_tokenizer = _make_mock_tokenizer()
        example = {"input": "Test", "output": "Response"}

        result = format_training_example(example, mock_tokenizer)

        # Simulate what TRL's _prepare_dataset does
        wrapped = {"text": result}

        assert isinstance(wrapped["text"], str), (
            f"After TRL wrapping, 'text' value is {type(wrapped['text']).__name__}, "
            "not str. This would cause a nested dict and crash add_eos."
        )

    def test_add_eos_works_after_wrapping(self):
        """Simulate the full crash path: format → TRL wrap → add_eos.

        This reproduces the original bug: add_eos calls .endswith() on the
        formatted text. If formatting_func returned a dict, this would raise
        AttributeError: 'dict' object has no attribute 'endswith'.
        """
        eos_token = "<|im_end|>"
        mock_tokenizer = _make_mock_tokenizer(
            f"<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\nResponse{eos_token}"
        )
        example = {"input": "Test", "output": "Response"}

        result = format_training_example(example, mock_tokenizer)

        # Simulate TRL wrapping
        formatted_example = {"text": result}

        # Simulate add_eos — this is the line that crashed in the original bug
        if "text" in formatted_example and not formatted_example["text"].endswith(eos_token):
            formatted_example["text"] += eos_token

        # If we get here without AttributeError, the fix works
        assert formatted_example["text"].endswith(eos_token)
