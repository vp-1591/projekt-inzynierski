"""Reproduce the structural audit cited in the Ukrainian thesis.

By default this script audits the local cleaned training data at:
model/dataset/mipd_train_cot_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ALLOWED_TAGS = {
    "REFERENCE_ERROR",
    "WHATABOUTISM",
    "STRAWMAN",
    "EMOTIONAL_CONTENT",
    "CHERRY_PICKING",
    "FALSE_CAUSE",
    "MISLEADING_CLICKBAIT",
    "ANECDOTE",
    "LEADING_QUESTIONS",
    "EXAGGERATION",
    "QUOTE_MINING",
}

TAG_ALIASES = {
    "MISLEADING_CLICKBAI": "MISLEADING_CLICKBAIT",
}


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[3] / "model" / "dataset" / "mipd_train_cot_clean.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit structural and label-consistency properties of mipd_train_cot_clean.jsonl."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=default_dataset_path(),
        type=Path,
        help="Path to mipd_train_cot_clean.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total = 0
    valid_output_json = 0
    non_empty_reasoning = 0
    reasoning_mentions_all_expected_tags = 0
    unknown_tag_records_after_normalization = 0
    alias_normalized_records = 0
    neutral_examples = 0
    manipulative_examples = 0
    tag_counts: Counter[str] = Counter()

    with args.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            total += 1
            row = json.loads(line)
            try:
                output = json.loads(row["output"])
                valid_output_json += 1
            except (KeyError, TypeError, json.JSONDecodeError):
                continue

            reasoning = str(output.get("reasoning", "")).strip()
            if reasoning:
                non_empty_reasoning += 1

            tags = output.get("discovered_techniques", [])
            if not isinstance(tags, list):
                tags = []

            raw_tags = [str(tag).strip().upper() for tag in tags]
            normalized_tags = [TAG_ALIASES.get(tag, tag) for tag in raw_tags]
            unknown_tags = [tag for tag in normalized_tags if tag not in ALLOWED_TAGS]
            if unknown_tags:
                unknown_tag_records_after_normalization += 1
            if raw_tags != normalized_tags:
                alias_normalized_records += 1

            if normalized_tags:
                manipulative_examples += 1
            else:
                neutral_examples += 1

            tag_counts.update(normalized_tags)
            if all(tag in reasoning.upper() for tag in normalized_tags):
                reasoning_mentions_all_expected_tags += 1

    print(f"records: {total}")
    print(f"valid output JSON: {valid_output_json}/{total} ({valid_output_json / total:.2%})")
    print(f"non-empty reasoning: {non_empty_reasoning}/{total} ({non_empty_reasoning / total:.2%})")
    print(
        "reasoning mentions all expected tags: "
        f"{reasoning_mentions_all_expected_tags}/{total} "
        f"({reasoning_mentions_all_expected_tags / total:.2%})"
    )
    print(f"records with unknown tags after normalization: {unknown_tag_records_after_normalization}")
    print(f"records normalized by tag alias: {alias_normalized_records}")
    print(f"neutral examples: {neutral_examples}")
    print(f"manipulative examples: {manipulative_examples}")
    print("tag counts:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
