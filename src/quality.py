from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from constants import ARTIFACTS_DIR, ensure_dir
from data import load_json, write_json


QUALITY_FIELDS = [
    "source",
    "id",
    "language",
    "gold",
    "text",
    "coherent",
    "labels_correct",
    "extra_aspects",
    "usable_for_training",
    "notes",
]


def _sample_source(source: dict[str, Any], sample_size: int, seed: int) -> list[dict[str, Any]]:
    path = Path(source["path"])
    examples = load_json(path)
    if not isinstance(examples, list):
        raise ValueError(f"Quality source must be a JSON list: {path}")
    stable_offset = sum(ord(char) for char in str(path)) % 10_000
    rng = random.Random(seed + stable_offset)
    sample = rng.sample(examples, min(sample_size, len(examples)))
    name = source.get("name") or path.parent.name
    rows = []
    for example in sample:
        rows.append(
            {
                "source": name,
                "id": example.get("id", ""),
                "language": example.get("language", ""),
                "gold": json.dumps(example.get("gold", {}), ensure_ascii=False, sort_keys=True),
                "text": example.get("text", ""),
                "coherent": "",
                "labels_correct": "",
                "extra_aspects": "",
                "usable_for_training": "",
                "notes": "",
            }
        )
    return rows


def run_quality_sample_config(config: dict[str, Any]) -> Path:
    output_root = ensure_dir(Path(config.get("output_root", ARTIFACTS_DIR / "quality_checks")))
    name = config.get("name", "synthetic_quality_samples")
    sample_size = int(config.get("sample_size", 60))
    seed = int(config.get("seed", 13))
    sources = config.get("sources") or []
    if not sources:
        raise ValueError("quality_sample requires at least one source")

    all_rows = []
    output = output_root / f"{name}.csv"
    if output.exists():
        return output
    for source in sources:
        all_rows.extend(_sample_source(source, sample_size, seed))

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    write_json(
        output_root / f"{name}.json",
        {
            "name": name,
            "sample_size_per_source": sample_size,
            "seed": seed,
            "sources": sources,
            "rows": len(all_rows),
            "csv": str(output),
        },
    )
    (output_root / f"{name}.md").write_text(
        "\n".join(
            [
                "# Synthetic Quality Check",
                "",
                f"CSV: `{output}`",
                "",
                "Fill `coherent`, `labels_correct`, `extra_aspects`, and `usable_for_training` with `yes`, `partial`, or `no`.",
                "Use `notes` for recurring failure modes such as wrong language, missing conflict, or extra sentiment.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return output
