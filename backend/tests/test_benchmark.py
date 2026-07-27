"""Tests for backend/app/training/benchmark.py.

Covers evaluate_response, format_prompt, and report_progress.
The main() CLI entry point is excluded — it requires GPU/model loading.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# The benchmark module imports heavy training dependencies (torch, unsloth, datasets,
# tqdm) at module level. These are not available in the lightweight dev venv and are
# not needed for the functions under test. Mock them before importing the module.
for _mod in ("torch", "unsloth", "unsloth.FastLanguageModel", "datasets", "tqdm", "tqdm.auto", "tqdm.auto.tqdm"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from app.training.benchmark import evaluate_response, format_prompt, report_progress  # noqa: E402, I001


# ---------------------------------------------------------------------------
# evaluate_response
# ---------------------------------------------------------------------------


class TestEvaluateResponseStrictJson:
    """Phase 1 — structured JSON parsing succeeds."""

    def test_evaluate_response_strict_json_dict(self):
        """Dict input with discovered_techniques key yields 'Strict Success'."""
        response = json.dumps(
            {"reasoning": "Some reasoning", "discovered_techniques": ["STRAWMAN", "EMOTIONAL_CONTENT"]}
        )
        result = evaluate_response(response, ["STRAWMAN", "EMOTIONAL_CONTENT"])

        assert result["parsing_status"] == "Strict Success"
        assert set(result["parsed_tags"]) == {"STRAWMAN", "EMOTIONAL_CONTENT"}

    def test_evaluate_response_strict_json_list(self):
        """Plain JSON list input yields 'Strict Success'."""
        response = json.dumps(["STRAWMAN", "CHERRY_PICKING"])
        result = evaluate_response(response, ["STRAWMAN"])

        assert result["parsing_status"] == "Strict Success"
        assert set(result["parsed_tags"]) == {"STRAWMAN", "CHERRY_PICKING"}

    def test_evaluate_response_strict_dict_missing_key(self):
        """Dict without discovered_techniques yields empty tags but still 'Strict Success'."""
        response = json.dumps({"reasoning": "No techniques found"})
        result = evaluate_response(response, ["STRAWMAN"])

        assert result["parsing_status"] == "Strict Success"
        assert result["parsed_tags"] == []

    def test_evaluate_response_strict_dict_non_list_value(self):
        """Dict with discovered_techniques as a non-list value normalises to empty list."""
        response = json.dumps({"discovered_techniques": "not-a-list"})
        result = evaluate_response(response, [])

        assert result["parsing_status"] == "Strict Success"
        assert result["parsed_tags"] == []


class TestEvaluateResponseRegexRecovery:
    """Phase 2 — regex recovery when strict JSON parsing fails."""

    def test_evaluate_response_regex_recovery(self):
        """Malformed JSON containing an array inside text yields 'Recovered'."""
        # Not valid JSON at top level, but contains ["STRAWMAN"] inside
        response = 'Here is my answer: ["STRAWMAN"] and some trailing text'
        result = evaluate_response(response, ["STRAWMAN"])

        assert result["parsing_status"] == "Recovered"
        assert set(result["parsed_tags"]) == {"STRAWMAN"}

    def test_evaluate_response_regex_recovery_multiple_tags(self):
        """Regex recovery extracts multiple tags from embedded array."""
        response = 'The result is ["STRAWMAN", "EMOTIONAL_CONTENT"] end.'
        result = evaluate_response(response, ["STRAWMAN", "EMOTIONAL_CONTENT"])

        assert result["parsing_status"] == "Recovered"
        assert set(result["parsed_tags"]) == {"STRAWMAN", "EMOTIONAL_CONTENT"}


class TestEvaluateResponseFailedParse:
    """Completely unparseable input falls back to 'Failed'."""

    def test_evaluate_response_failed_parse(self):
        """Garbage string yields 'Failed' parsing status."""
        result = evaluate_response("this is not json or json-like at all", ["STRAWMAN"])

        assert result["parsing_status"] == "Failed"
        assert result["parsed_tags"] == []

    def test_evaluate_response_empty_string(self):
        """Empty string yields 'Failed' with empty predicted tags."""
        result = evaluate_response("", [])

        assert result["parsing_status"] == "Failed"
        assert result["parsed_tags"] == []


class TestEvaluateResponseMarkdownWrapped:
    """Responses wrapped in ```json...``` markdown fences."""

    def test_evaluate_response_markdown_wrapped(self):
        """Markdown-wrapped JSON yields 'Strict Success' after stripping."""
        inner = json.dumps({"reasoning": "Analysis", "discovered_techniques": ["EXAGGERATION"]})
        response = f"```json\n{inner}\n```"
        result = evaluate_response(response, ["EXAGGERATION"])

        assert result["parsing_status"] == "Strict Success"
        assert set(result["parsed_tags"]) == {"EXAGGERATION"}


class TestEvaluateResponseEmptyGroundTruth:
    """Edge case: ground truth has no tags."""

    def test_evaluate_response_empty_ground_truth(self):
        """Empty ground truth sets has_gold_labels to False."""
        result = evaluate_response("garbage", [])

        assert result["has_gold_labels"] is False

    def test_evaluate_response_non_empty_ground_truth(self):
        """Non-empty ground truth sets has_gold_labels to True."""
        result = evaluate_response("garbage", ["STRAWMAN"])

        assert result["has_gold_labels"] is True


class TestEvaluateResponseMetrics:
    """F1 and exact-match calculations."""

    def test_evaluate_response_partial_match_f1(self):
        """Partial overlap between predicted and ground truth yields correct F1."""
        # predicted: {A, B}, ground truth: {A, C}
        # TP=1, FP=1, FN=1 → F1 = 2*1 / (2*1+1+1) = 0.5
        result = evaluate_response(
            json.dumps({"discovered_techniques": ["A", "B"]}),
            ["A", "C"],
        )
        assert result["f1_doc"] == pytest.approx(0.5)

    def test_evaluate_response_f1_no_overlap(self):
        """No overlap → F1 = 0."""
        result = evaluate_response(
            json.dumps({"discovered_techniques": ["X"]}),
            ["Y"],
        )
        assert result["f1_doc"] == 0.0

    def test_evaluate_response_f1_perfect_match(self):
        """Perfect match → F1 = 1.0."""
        result = evaluate_response(
            json.dumps({"discovered_techniques": ["A", "B"]}),
            ["B", "A"],
        )
        assert result["f1_doc"] == 1.0

    def test_evaluate_response_f1_both_empty(self):
        """Both predicted and ground truth empty → F1 = 0.0 (edge case in implementation)."""
        # Empty sets: TP=0, FP=0, FN=0 → the code returns 0.0
        result = evaluate_response("nonsense", [])
        assert result["f1_doc"] == 0.0

    def test_evaluate_response_exact_match_true(self):
        """Same predicted and ground truth sets → exact_match=True."""
        result = evaluate_response(
            json.dumps({"discovered_techniques": ["STRAWMAN"]}),
            ["STRAWMAN"],
        )
        assert result["exact_match"] is True

    def test_evaluate_response_exact_match_false(self):
        """Different predicted and ground truth sets → exact_match=False."""
        result = evaluate_response(
            json.dumps({"discovered_techniques": ["STRAWMAN", "EXAGGERATION"]}),
            ["STRAWMAN"],
        )
        assert result["exact_match"] is False

    def test_evaluate_response_exact_match_order_irrelevant(self):
        """Exact match is order-independent (sets comparison)."""
        result = evaluate_response(
            json.dumps({"discovered_techniques": ["B", "A"]}),
            ["A", "B"],
        )
        assert result["exact_match"] is True


class TestEvaluateResponseRawOutput:
    """The raw_output field preserves the original response text."""

    def test_raw_output_preserved(self):
        """raw_output should be the original response_text unchanged."""
        original = json.dumps({"discovered_techniques": ["X"]})
        result = evaluate_response(original, ["X"])

        assert result["raw_output"] == original

    def test_ground_truth_field(self):
        """ground_truth field lists the provided tags (as strings)."""
        result = evaluate_response("irrelevant", ["A", 2, None])
        # None is filtered out; 2 is cast to str
        assert set(result["ground_truth"]) == {"A", "2"}


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------


class TestFormatPrompt:
    """Tests for format_prompt(example, tokenizer)."""

    @staticmethod
    def _make_tokenizer():
        """Return a mock tokenizer with apply_chat_template."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = (
            "<|im_start|>system\nYou are an expert.\n"
            "<|im_end|>\n<|im_start|>user\nHello\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return tokenizer

    def test_format_prompt_adds_system_and_user_messages(self):
        """System prompt and user content appear in the formatted prompt."""
        tokenizer = self._make_tokenizer()
        example = {
            "input": "Tekst do analizy",
            "output": json.dumps({"reasoning": "ok", "discovered_techniques": ["STRAWMAN"]}),
        }

        result = format_prompt(example, tokenizer)

        # apply_chat_template should have been called with system + user messages
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0] if call_args[0] else call_args[1].get("messages", [])
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles
        # The user message content should be the input text
        user_msg = [m for m in messages if m["role"] == "user"][0]
        assert user_msg["content"] == "Tekst do analizy"
        # The prompt key should be set on the returned example
        assert "prompt" in result

    def test_format_prompt_handles_malformed_output(self):
        """Invalid JSON in output field results in empty tags list."""
        tokenizer = self._make_tokenizer()
        example = {
            "input": "Tekst",
            "output": "this is not json",
        }

        result = format_prompt(example, tokenizer)

        assert result["tags"] == []

    def test_format_prompt_preserves_valid_tags(self):
        """Valid JSON output field with discovered_techniques is extracted correctly."""
        tokenizer = self._make_tokenizer()
        example = {
            "input": "Tekst",
            "output": json.dumps(
                {"reasoning": "Found techniques", "discovered_techniques": ["STRAWMAN", "EMOTIONAL_CONTENT"]}
            ),
        }

        result = format_prompt(example, tokenizer)

        assert result["tags"] == ["STRAWMAN", "EMOTIONAL_CONTENT"]

    def test_format_prompt_strips_markdown_from_output(self):
        """Markdown fences around JSON in output are stripped before parsing."""
        tokenizer = self._make_tokenizer()
        inner = json.dumps({"reasoning": "ok", "discovered_techniques": ["EXAGGERATION"]})
        example = {
            "input": "Tekst",
            "output": f"```json\n{inner}\n```",
        }

        result = format_prompt(example, tokenizer)

        assert result["tags"] == ["EXAGGERATION"]

    def test_format_prompt_applies_chat_template_kwargs(self):
        """apply_chat_template is called with tokenize=False and add_generation_prompt=True."""
        tokenizer = self._make_tokenizer()
        example = {
            "input": "Tekst",
            "output": json.dumps({"reasoning": "ok", "discovered_techniques": []}),
        }

        format_prompt(example, tokenizer)

        _, kwargs = tokenizer.apply_chat_template.call_args
        assert kwargs.get("tokenize") is False
        assert kwargs.get("add_generation_prompt") is True


# ---------------------------------------------------------------------------
# report_progress
# ---------------------------------------------------------------------------


class TestReportProgress:
    """Tests for report_progress(url, value)."""

    @patch("app.training.benchmark.requests.post")
    def test_report_progress_sends_post(self, mock_post):
        """report_progress sends a POST request with the correct payload."""
        mock_post.return_value = MagicMock(status_code=200)

        report_progress("http://localhost:8000", 50)

        mock_post.assert_called_once_with(
            "http://localhost:8000/training/progress",
            json={"stage": "evaluation", "value": 50},
            timeout=1,
        )

    @patch("app.training.benchmark.requests.post")
    def test_report_progress_suppresses_exceptions(self, mock_post):
        """POST failures are silently suppressed — no exception propagates."""
        mock_post.side_effect = ConnectionError("Connection refused")

        # Should not raise
        report_progress("http://localhost:8000", 50)

    @patch("app.training.benchmark.requests.post")
    def test_report_progress_suppresses_timeout(self, mock_post):
        """Timeout exceptions are also suppressed."""
        import requests as _requests

        mock_post.side_effect = _requests.exceptions.Timeout("timed out")

        # Should not raise
        report_progress("http://localhost:8000", 75)

    @patch("app.training.benchmark.requests.post")
    def test_report_progress_url_concatenation(self, mock_post):
        """The URL is concatenated with /training/progress without extra slashes."""
        mock_post.return_value = MagicMock(status_code=200)

        report_progress("http://backend:8000", 10)

        called_url = mock_post.call_args[0][0]
        assert called_url == "http://backend:8000/training/progress"
