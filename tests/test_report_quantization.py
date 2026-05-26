import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constants import normalize_quantization
from report import aggregate_by_seed, scan_runs


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_quantization_aliases_are_canonical():
    assert normalize_quantization("none") == "none"
    assert normalize_quantization("4bit") == "4bit"
    assert normalize_quantization("4 bit") == "4bit"
    assert normalize_quantization("4-bit") == "4bit"


def test_report_aggregates_quantization_aliases_by_seed(tmp_path):
    for seed, quantization, score in [(13, "4 bit", 60.0), (42, "4bit", 70.0)]:
        run_dir = tmp_path / "runs" / "zero_shot" / f"run-{seed}"
        _write_json(
            run_dir / "run.json",
            {
                "run_id": f"run-{seed}",
                "params": {
                    "task": "zero_shot",
                    "model": "llama32B3",
                    "prompt": "configs/prompts/zero_shot/baseline.yaml",
                    "split": "devel",
                    "shots": 0,
                    "shot_strategy": "first",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": 128,
                    "seed": seed,
                    "quantization": quantization,
                },
            },
        )
        _write_json(
            run_dir / "metrics.json",
            {
                "macro_assignment_compatible": {"f1": score},
                "micro": {"f1": score + 1.0},
                "macro_empty_is_perfect": {"f1": score},
                "json_validity_rate": 100.0,
                "empty_prediction_rate": 0.0,
                "errors": 0,
            },
        )

    rows = scan_runs(tmp_path / "runs")
    aggregate = aggregate_by_seed(rows)

    assert len(rows) == 2
    assert len(aggregate) == 1
    assert aggregate[0]["quantization"] == "4bit"
    assert aggregate[0]["n_runs"] == 2
    assert aggregate[0]["macro_f1_mean"] == 65.0
