from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from constants import RUNS_DIR, ensure_dir, get_model_spec
from data import load_json, write_json
from evaluate import evaluate_examples, save_evaluation
from parsing import normalize_prediction


def _load_predictions(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "predictions.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assignment = Path(run_dir) / "predictions.assignment.json"
    if assignment.exists():
        data = load_json(assignment)
        if isinstance(data, list):
            return data
    raise FileNotFoundError(f"No predictions found in {run_dir}")


def _pairs(example: dict[str, Any], repair_policy: str) -> set[tuple[str, str]]:
    raw = example.get("raw_prediction", example.get("prediction", {}))
    normalized = normalize_prediction(raw, repair_policy=repair_policy)
    return set(normalized.opinions.items())


def _pairs_to_dict(pairs: set[tuple[str, str]], member_pair_lists: list[set[tuple[str, str]]]) -> dict[str, str]:
    by_aspect: dict[str, Counter] = defaultdict(Counter)
    for aspect, polarity in pairs:
        by_aspect[aspect][polarity] += 1
    output: dict[str, str] = {}
    for aspect, counts in by_aspect.items():
        if len(counts) == 1:
            output[aspect] = next(iter(counts))
            continue
        ordered = []
        for member_pairs in member_pair_lists:
            for pair_aspect, pair_polarity in member_pairs:
                if pair_aspect == aspect:
                    ordered.append(pair_polarity)
        output[aspect] = counts.most_common(1)[0][0] if not ordered else ordered[0]
    return output


def _vote(member_pairs: list[set[tuple[str, str]]], mode: str) -> set[tuple[str, str]]:
    if mode == "agreement_then_tiebreak" and len(member_pairs) >= 2:
        if member_pairs[0] == member_pairs[1]:
            return set(member_pairs[0])
        member_pairs = member_pairs[:3]

    counts = Counter(pair for pairs in member_pairs for pair in pairs)
    n_members = len(member_pairs)
    if mode in {"majority", "agreement_then_tiebreak"}:
        return {pair for pair, count in counts.items() if count > n_members / 2}
    if mode == "union_high_recall":
        return set(counts)
    if mode == "intersection_high_precision":
        return {pair for pair, count in counts.items() if count == n_members}
    raise ValueError(f"Unknown ensemble mode: {mode}")


def _member_predictions(member: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    name = member.get("name") or Path(str(member.get("path", ""))).name
    if member.get("path"):
        return name, _load_predictions(member["path"]), str(member["path"])
    if member.get("config"):
        from experiments import run_experiment_config

        run_dir = run_experiment_config(member["config"])
        return name, _load_predictions(run_dir), str(run_dir)
    raise ValueError(f"Ensemble member must define path or config: {member}")


def run_ensemble_config(config: dict[str, Any]) -> Path:
    members = config.get("members") or []
    if len(members) < 2:
        raise ValueError("ensemble_eval requires at least two members")

    mode = config.get("mode", "agreement_then_tiebreak")
    repair_policy = config.get("repair_policy", "none")
    output_root = Path(config.get("output_root", RUNS_DIR))
    base_name = config.get("name", "ensemble_eval")
    suffix = f"{mode}-repair{repair_policy}"
    run_id = f"{base_name}-{suffix}"
    run_dir = ensure_dir(output_root / "ensemble_eval" / run_id)
    if (run_dir / "metrics.json").exists() and (run_dir / "predictions.jsonl").exists():
        return run_dir
    ensure_dir(run_dir / "logs")

    loaded = [_member_predictions(member) for member in members]
    member_names = [item[0] for item in loaded]
    member_paths = [item[2] for item in loaded]
    predictions_by_id: list[dict[str, dict[str, Any]]] = [
        {str(example.get("id", example.get("rid", idx))): example for idx, example in enumerate(predictions)}
        for _, predictions, _ in loaded
    ]
    ids = list(predictions_by_id[0])
    outputs = []
    disagreements = []

    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for example_id in ids:
            base = predictions_by_id[0][example_id]
            member_pair_lists = [
                _pairs(member_predictions[example_id], repair_policy)
                for member_predictions in predictions_by_id
                if example_id in member_predictions
            ]
            voted_pairs = _vote(member_pair_lists, mode)
            output = dict(base)
            output["prediction"] = _pairs_to_dict(voted_pairs, member_pair_lists)
            output["raw_prediction"] = output["prediction"]
            output["strict_prediction"] = output["prediction"]
            output["parse_error"] = None
            output["invalid_items"] = []
            output["repaired_items"] = []
            output["ensemble_members"] = member_names
            output["ensemble_mode"] = mode
            output["repair_policy"] = repair_policy
            outputs.append(output)
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            if len({frozenset(pairs) for pairs in member_pair_lists}) > 1:
                disagreements.append(
                    {
                        "id": example_id,
                        "member_predictions": json.dumps(
                            {
                                name: sorted(list(pairs))
                                for name, pairs in zip(member_names, member_pair_lists)
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "voted": json.dumps(output["prediction"], ensure_ascii=False, sort_keys=True),
                    }
                )

    write_json(run_dir / "predictions.assignment.json", outputs)
    metrics, errors = evaluate_examples(outputs, repair_policy="none")
    save_evaluation(run_dir / "metrics.json", run_dir / "errors.csv", metrics, errors)
    write_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "params": {
                "task": "ensemble_eval",
                "model": config.get("model", "ensemble"),
                "prompt": "ensemble",
                "split": config.get("split", "devel"),
                "seed": config.get("seed", ""),
                "quantization": "",
                "mode": mode,
                "repair_policy": repair_policy,
                "members": member_paths,
            },
            "model_spec": get_model_spec("ensemble").__dict__,
        },
    )

    with (run_dir / "member_disagreements.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "member_predictions", "voted"])
        writer.writeheader()
        writer.writerows(disagreements)
    return run_dir
