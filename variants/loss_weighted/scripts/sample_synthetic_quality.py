from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


FIELDS = [
    "id",
    "target_opinions",
    "text",
    "mentions_all_target_aspects",
    "polarities_match_target",
    "no_extra_aspect_sentiment",
    "natural_review",
    "usable_for_training",
    "notes",
]


def load_examples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def write_quality_csv(examples: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "id": example.get("id", ""),
                    "target_opinions": json.dumps(example.get("gold", {}), ensure_ascii=False, sort_keys=True),
                    "text": example.get("text", ""),
                    "mentions_all_target_aspects": "",
                    "polarities_match_target": "",
                    "no_extra_aspect_sentiment": "",
                    "natural_review": "",
                    "usable_for_training": "",
                    "notes": "",
                }
            )


def write_quality_md(output: Path, sample_csv: Path, sample_size: int) -> None:
    text = f"""# Synthetic Data Manual Quality Check

Sample file: `{sample_csv}`

Review {sample_size} generated examples. Fill each blank judgement column with `yes`, `partial`, or `no`.

Columns:
- `mentions_all_target_aspects`: every aspect in `target_opinions` is expressed in the text.
- `polarities_match_target`: the sentiment for those aspects matches the requested polarity.
- `no_extra_aspect_sentiment`: the review does not clearly add sentiment for aspects outside the JSON.
- `natural_review`: the text is fluent enough to be plausible training data.
- `usable_for_training`: overall manual decision.
- `notes`: short explanation for failures or borderline cases.

Report suggestion:
- Mention sample size.
- Report percentage of `usable_for_training=yes`.
- Mention the most frequent failure mode, if any.
"""
    output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a manual QA sheet for synthetic ABSA examples.")
    parser.add_argument("synthetic_json", help="Path to synt.json produced by the augmentation pipeline.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", help="Output CSV path. Defaults next to synt.json.")
    args = parser.parse_args()

    synthetic_path = Path(args.synthetic_json)
    examples = load_examples(synthetic_path)
    rng = random.Random(args.seed)
    sample_size = min(args.sample_size, len(examples))
    sample = rng.sample(examples, sample_size)

    if args.output:
        output = Path(args.output)
    else:
        output = synthetic_path.parent / f"quality_sample_{sample_size}_seed{args.seed}.csv"
    write_quality_csv(sample, output)
    write_quality_md(output.with_suffix(".md"), output, sample_size)
    print(output)
    print(output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
