"""Tests for backend/app/training/evaluator.py — AutoBenchmarker.

sklearn/scipy have compatibility issues on Python 3.14, so we mock sklearn
at the sys.modules level before importing the module under test, then patch
f1_score with a real implementation.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock sklearn/scipy at import time to avoid the Python 3.14 incompatibility.
for mod in ("sklearn", "sklearn.metrics", "sklearn.utils", "scipy", "scipy.stats"):
    sys.modules.setdefault(mod, MagicMock())

from app.training.evaluator import AutoBenchmarker  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal macro-F1 matching sklearn's f1_score(average='macro', zero_division=0)
# ---------------------------------------------------------------------------


def _macro_f1(y_true, y_pred, zero_division=0.0):
    """Compute macro-averaged F1 matching sklearn's binary-per-label approach.

    Each index is treated as a separate binary classification problem.
    Labels where both y_true=0 and y_pred=0 get F1=zero_division (0.0).
    """
    f1_scores = []
    for yt, yp in zip(y_true, y_pred, strict=True):
        tp = int(yt == 1 and yp == 1)
        fp = int(yt == 0 and yp == 1)
        fn = int(yt == 1 and yp == 0)
        if tp == 0 and fp == 0 and fn == 0:
            f1_scores.append(float(zero_division))
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


# The technique mapping used across all tests (3 classes)
_TECHNIQUE_MAPPING = {
    "STRAWMAN": "Chochoł",
    "EMOTIONAL_CONTENT": "Emocje",
    "CHERRY_PICKING": "Wybiórczość",
}
_ALL_CLASSES = list(_TECHNIQUE_MAPPING.keys())


@pytest.fixture(autouse=True)
def patch_f1_score():
    """Replace sklearn f1_score with our minimal implementation for all tests."""
    from app.training import evaluator as ev_module

    def _fake_f1_score(y_true, y_pred, average="macro", zero_division=0):
        return _macro_f1(y_true, y_pred, zero_division=zero_division)

    with patch.object(ev_module, "f1_score", side_effect=_fake_f1_score):
        yield


@pytest.fixture
def benchmarker():
    return AutoBenchmarker(technique_mapping=_TECHNIQUE_MAPPING)


# ---------------------------------------------------------------------------
# evaluate_response  –  parsing_status
# ---------------------------------------------------------------------------


class TestEvaluateResponse:
    def test_evaluate_response_strict_success(self, benchmarker):
        """Valid JSON dict with discovered_techniques yields 'Strict Success'."""
        response = json.dumps(
            {
                "reasoning": "Tekst stosuje chochoł.",
                "discovered_techniques": ["STRAWMAN", "EMOTIONAL_CONTENT"],
            }
        )
        result = benchmarker.evaluate_response(response, ["STRAWMAN", "EMOTIONAL_CONTENT"])

        assert result["parsing_status"] == "Strict Success"
        assert result["parsed_tags"] == ["STRAWMAN", "EMOTIONAL_CONTENT"]
        assert result["ground_truth"] == ["STRAWMAN", "EMOTIONAL_CONTENT"]
        # With 3 classes: STRAWMAN(1,1), EMOT(1,1), CHERRY(0,0) → F1 = (1+1+0)/3 = 2/3
        assert result["f1_score"] == pytest.approx(2.0 / 3)

    def test_evaluate_response_recovered(self, benchmarker):
        """Malformed JSON where an array can be regex-recovered yields 'Recovered'."""
        response = 'discovered_techniques: ["STRAWMAN", "CHERRY_PICKING"] extra stuff'
        result = benchmarker.evaluate_response(response, ["STRAWMAN", "CHERRY_PICKING"])

        assert result["parsing_status"] == "Recovered"
        assert result["parsed_tags"] == ["STRAWMAN", "CHERRY_PICKING"]

    def test_evaluate_response_failed(self, benchmarker):
        """Completely unparseable garbage yields 'Failed' and empty parsed_tags."""
        result = benchmarker.evaluate_response("blah blah no structure here", ["STRAWMAN"])

        assert result["parsing_status"] == "Failed"
        assert result["parsed_tags"] == []
        # predicted=[], actual=["STRAWMAN"] → y_true=[1,0,0], y_pred=[0,0,0]
        # STRAWMAN: TP=0,FP=0,FN=1 → F1=0; EMOT: 0/0→0; CHERRY: 0/0→0
        # Macro = 0/3 = 0.0
        assert result["f1_score"] == 0.0

    def test_evaluate_response_markdown_wrapped(self, benchmarker):
        """JSON wrapped in ```json...``` fences still parses as 'Strict Success'."""
        inner = json.dumps({"discovered_techniques": ["EMOTIONAL_CONTENT"], "reasoning": "Emocje."})
        response = f"```json\n{inner}\n```"
        result = benchmarker.evaluate_response(response, ["EMOTIONAL_CONTENT"])

        assert result["parsing_status"] == "Strict Success"
        assert result["parsed_tags"] == ["EMOTIONAL_CONTENT"]

    def test_evaluate_response_empty_json_dict(self, benchmarker):
        """Valid JSON dict without discovered_techniques key defaults to empty list."""
        response = json.dumps({"reasoning": "Brak technik."})
        result = benchmarker.evaluate_response(response, [])

        assert result["parsing_status"] == "Strict Success"
        assert result["parsed_tags"] == []

    def test_evaluate_response_bare_json_array_yields_failed(self, benchmarker):
        """A bare JSON array is valid JSON but not a dict, so Phase 1 skips it.
        Phase 2 only runs inside the except block, so the result is 'Failed'."""
        response = '["STRAWMAN", "EMOTIONAL_CONTENT"]'
        result = benchmarker.evaluate_response(response, ["STRAWMAN", "EMOTIONAL_CONTENT"])

        assert result["parsing_status"] == "Failed"
        assert result["parsed_tags"] == []


# ---------------------------------------------------------------------------
# calculate_f1
# ---------------------------------------------------------------------------


class TestCalculateF1:
    def test_calculate_f1_perfect_match(self, benchmarker):
        """All 3 classes correctly predicted → F1 = 1.0."""
        assert (
            benchmarker.calculate_f1(
                ["STRAWMAN", "EMOTIONAL_CONTENT", "CHERRY_PICKING"],
                ["STRAWMAN", "EMOTIONAL_CONTENT", "CHERRY_PICKING"],
            )
            == 1.0
        )

    def test_calculate_f1_partial_match(self, benchmarker):
        """Partial overlap produces a correct macro-F1 between 0 and 1.
        predicted=["STRAWMAN"], actual=["STRAWMAN", "CHERRY_PICKING"]
        y_true=[1,0,1], y_pred=[1,0,0]:
          - STRAWMAN: TP=1,FP=0,FN=0 → F1=1
          - EMOT: TP=0,FP=0,FN=0 → F1=0 (zero_division)
          - CHERRY: TP=0,FP=0,FN=1 → F1=0
          - Macro = (1+0+0)/3 = 1/3"""
        f1 = benchmarker.calculate_f1(["STRAWMAN"], ["STRAWMAN", "CHERRY_PICKING"])
        assert f1 == pytest.approx(1.0 / 3)

    def test_calculate_f1_empty_both(self, benchmarker):
        """Both lists empty means perfect match, so F1 = 1.0."""
        assert benchmarker.calculate_f1([], []) == 1.0

    def test_calculate_f1_empty_predicted_nonempty_actual(self, benchmarker):
        """Empty predicted, non-empty actual.
        y_true=[1,1,0], y_pred=[0,0,0]:
          - STRAWMAN: TP=0,FP=0,FN=1 → F1=0
          - EMOT: TP=0,FP=0,FN=1 → F1=0
          - CHERRY: TP=0,FP=0,FN=0 → F1=0
          - Macro = 0/3 = 0.0"""
        assert benchmarker.calculate_f1([], ["STRAWMAN", "EMOTIONAL_CONTENT"]) == pytest.approx(0.0)

    def test_calculate_f1_ignores_unknown_tags(self, benchmarker):
        """Tags not in technique_mapping are excluded from y_true and y_pred.
        UNKNOWN_TAG is filtered out. After filtering:
        predicted=["STRAWMAN"], actual=["STRAWMAN", "CHERRY_PICKING"]"""
        f1 = benchmarker.calculate_f1(
            ["STRAWMAN", "UNKNOWN_TAG"],
            ["STRAWMAN", "CHERRY_PICKING", "UNKNOWN_TAG"],
        )
        assert f1 == pytest.approx(1.0 / 3)

    def test_calculate_f1_all_predicted_none_actual(self, benchmarker):
        """Non-empty predicted but empty actual.
        y_true=[0,0,0], y_pred=[1,1,0]:
          - All labels: F1=0
          - Macro = 0/3 = 0.0"""
        assert benchmarker.calculate_f1(["STRAWMAN", "EMOTIONAL_CONTENT"], []) == pytest.approx(0.0)

    def test_calculate_f1_coerces_to_strings(self, benchmarker):
        """calculate_f1 coerces non-string items to strings before comparison.
        Integer 1 is not a key in technique_mapping, so it contributes nothing.
        STRAWMAN is present in both predicted and actual:
        y_true=[1,0,0], y_pred=[1,0,0]:
          - STRAWMAN: F1=1; EMOT: F1=0; CHERRY: F1=0
          - Macro = 1/3"""
        f1 = benchmarker.calculate_f1([1, "STRAWMAN"], ["STRAWMAN"])
        assert f1 == pytest.approx(1.0 / 3)
