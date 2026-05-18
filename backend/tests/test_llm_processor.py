from app.llm_processor import normalize_llm_response


def test_normalize_llm_response_accepts_valid_json():
    result = normalize_llm_response(
        '{"reasoning": "Tekst wyolbrzymia problem.", '
        '"discovered_techniques": ["EXAGGERATION"]}'
    )

    assert result["reasoning"] == "Tekst wyolbrzymia problem."
    assert result["discovered_techniques"] == ["EXAGGERATION"]


def test_normalize_llm_response_heals_fuzzy_keys_and_tag_typos():
    result = normalize_llm_response(
        '{"reason": "Uzyto emocjonalnego jezyka.", '
        '"techniques": ["emotio content", "quote"]}'
    )

    assert result["reasoning"] == "Uzyto emocjonalnego jezyka."
    assert set(result["discovered_techniques"]) == {
        "EMOTIONAL_CONTENT",
        "QUOTE_MINING",
    }


def test_normalize_llm_response_recovers_from_non_json_text():
    result = normalize_llm_response(
        'reasoning: fallback text "reasoning": "Regex recovered" '
        'tags: ["cherry picking", "MISLEADING_CLICKBAIT"]'
    )

    assert result["reasoning"] == "Regex recovered"
    assert set(result["discovered_techniques"]) == {
        "CHERRY_PICKING",
        "MISLEADING_CLICKBAIT",
    }
