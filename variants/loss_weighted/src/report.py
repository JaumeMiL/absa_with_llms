from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data import load_json, write_json
from constants import get_model_spec
from constants import REPORTS_DIR, RUNS_DIR, ensure_dir, normalize_quantization


SUMMARY_FIELDS = [
    "task",
    "run_id",
    "model",
    "reportable",
    "prompt",
    "prompt_name",
    "split",
    "shots",
    "shot_strategy",
    "temperature",
    "top_p",
    "max_new_tokens",
    "seed",
    "quantization",
    "device",
    "adapter",
    "train_data",
    "devel_data",
    "synthetic_data",
    "epochs",
    "learning_rate",
    "max_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "eval_accumulation_steps",
    "bf16",
    "fp16",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "macro_f1",
    "micro_f1",
    "macro_empty_f1",
    "json_validity_rate",
    "empty_prediction_rate",
    "errors",
    "train_loss",
    "eval_loss",
    "train_runtime",
    "train_examples",
    "devel_examples",
    "training_history",
    "training_curve",
    "path",
]

GROUP_FIELDS = [
    "task",
    "model",
    "prompt",
    "split",
    "shots",
    "shot_strategy",
    "temperature",
    "top_p",
    "max_new_tokens",
    "quantization",
    "adapter",
    "train_data",
    "synthetic_data",
    "epochs",
    "learning_rate",
    "max_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "eval_accumulation_steps",
    "bf16",
    "fp16",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
]

METRIC_FIELDS = [
    "macro_f1",
    "micro_f1",
    "macro_empty_f1",
    "json_validity_rate",
    "empty_prediction_rate",
    "errors",
]

BREAKDOWN_FIELDS = [
    "task",
    "run_id",
    "model",
    "prompt_name",
    "shots",
    "shot_strategy",
    "quantization",
    "seed",
    "group_type",
    "group",
    "precision",
    "recall",
    "f1",
    "predicted",
    "expected",
    "correct",
    "path",
]


def _safe_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _get(data: dict[str, Any], dotted: str, default: Any = "") -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _prompt_name(prompt: str) -> str:
    if not prompt:
        return ""
    return Path(str(prompt).replace("\\", "/")).stem


def _task_for_eval(params: dict[str, Any], run_dir: Path) -> str:
    if params.get("adapter"):
        return "finetune_eval"
    task = params.get("task") or run_dir.parent.name
    if task == "finetune":
        return "finetune_eval"
    return task


def _base_row(run_path: Path, params: dict[str, Any]) -> dict[str, Any]:
    model_name = params.get("model", "")
    spec = get_model_spec(model_name) if model_name else None
    prompt = params.get("prompt", "")
    quantization = ""
    if "quantization" in params:
        quantization = normalize_quantization(params.get("quantization"))
    return {
        "run_id": load_json(run_path).get("run_id", run_path.parent.name),
        "task": params.get("task", run_path.parent.parent.name),
        "model": model_name,
        "reportable": bool(spec.reportable) if spec else False,
        "prompt": prompt,
        "prompt_name": _prompt_name(prompt),
        "split": params.get("split", ""),
        "shots": params.get("shots", 0),
        "shot_strategy": params.get("shot_strategy", ""),
        "temperature": params.get("temperature", ""),
        "top_p": params.get("top_p", ""),
        "max_new_tokens": params.get("max_new_tokens", ""),
        "seed": params.get("seed", ""),
        "quantization": quantization,
        "device": params.get("device", ""),
        "adapter": params.get("adapter", ""),
        "train_data": params.get("train_data", ""),
        "devel_data": params.get("devel_data", ""),
        "synthetic_data": params.get("synthetic_data", ""),
        "epochs": params.get("epochs", ""),
        "learning_rate": params.get("learning_rate", ""),
        "max_length": params.get("max_length", ""),
        "per_device_train_batch_size": params.get("per_device_train_batch_size", ""),
        "per_device_eval_batch_size": params.get("per_device_eval_batch_size", ""),
        "gradient_accumulation_steps": params.get("gradient_accumulation_steps", ""),
        "eval_accumulation_steps": params.get("eval_accumulation_steps", ""),
        "bf16": params.get("bf16", ""),
        "fp16": params.get("fp16", ""),
        "lora_r": params.get("lora_r", ""),
        "lora_alpha": params.get("lora_alpha", ""),
        "lora_dropout": params.get("lora_dropout", ""),
        "macro_f1": "",
        "micro_f1": "",
        "macro_empty_f1": "",
        "json_validity_rate": "",
        "empty_prediction_rate": "",
        "errors": "",
        "train_loss": "",
        "eval_loss": "",
        "train_runtime": "",
        "train_examples": "",
        "devel_examples": "",
        "training_history": "",
        "training_curve": "",
        "path": str(run_path.parent),
    }


def _flatten_run(run_path: Path) -> dict[str, Any] | None:
    run = load_json(run_path)
    params = run.get("params", {})
    run_dir = run_path.parent
    row = _base_row(run_path, params)

    metrics_path = run_dir / "metrics.json"
    train_metrics_path = run_dir / "train_metrics.json"

    if metrics_path.exists():
        metrics = load_json(metrics_path)
        row.update(
            {
                "task": _task_for_eval(params, run_dir),
                "macro_f1": _get(metrics, "macro_assignment_compatible.f1"),
                "micro_f1": _get(metrics, "micro.f1"),
                "macro_empty_f1": _get(metrics, "macro_empty_is_perfect.f1"),
                "json_validity_rate": metrics.get("json_validity_rate", ""),
                "empty_prediction_rate": metrics.get("empty_prediction_rate", ""),
                "errors": metrics.get("errors", ""),
            }
        )
        return row

    if train_metrics_path.exists():
        train_metrics = load_json(train_metrics_path)
        train_result = train_metrics.get("train_result", {})
        row.update(
            {
                "task": "finetune_train",
                "adapter": train_metrics.get("adapter_dir", row.get("adapter", "")),
                "train_loss": train_result.get("train_loss", ""),
                "eval_loss": train_result.get("eval_loss", ""),
                "train_runtime": train_result.get("train_runtime", ""),
                "train_examples": train_metrics.get("train_examples", ""),
                "devel_examples": train_metrics.get("devel_examples", ""),
                "training_history": train_metrics.get("training_history", ""),
                "training_curve": train_metrics.get("training_curve", ""),
            }
        )
        return row

    return None


def scan_runs(root: str | Path = RUNS_DIR, *, include_local: bool = False) -> list[dict[str, Any]]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for run_path in root.rglob("run.json"):
        row = _flatten_run(run_path)
        if row is None:
            continue
        if include_local or row["reportable"]:
            rows.append(row)
    rows.sort(key=_sort_key)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    macro = _safe_float(row.get("macro_f1"))
    return (
        row.get("task", ""),
        -(macro if macro is not None else -1.0),
        row.get("model", ""),
        row.get("run_id", ""),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate_by_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _safe_float(row.get("macro_f1")) is None:
            continue
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups[key].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for key, group_rows in groups.items():
        output = {field: value for field, value in zip(GROUP_FIELDS, key)}
        output["prompt_name"] = _prompt_name(str(output.get("prompt", "")))
        output["n_runs"] = len(group_rows)
        output["seeds"] = ",".join(sorted({str(row.get("seed", "")) for row in group_rows if row.get("seed", "") != ""}))
        output["run_ids"] = ";".join(row.get("run_id", "") for row in group_rows)
        for metric in METRIC_FIELDS:
            values = [_safe_float(row.get(metric)) for row in group_rows]
            values = [value for value in values if value is not None]
            if not values:
                output[f"{metric}_mean"] = ""
                output[f"{metric}_std"] = ""
                output[f"{metric}_min"] = ""
                output[f"{metric}_max"] = ""
                continue
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            output[f"{metric}_min"] = min(values)
            output[f"{metric}_max"] = max(values)
        aggregate_rows.append(output)

    aggregate_rows.sort(
        key=lambda row: (
            row.get("task", ""),
            -float(row.get("macro_f1_mean") or -1.0),
            row.get("model", ""),
            row.get("prompt", ""),
        )
    )
    return aggregate_rows


def collect_breakdowns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    breakdowns: list[dict[str, Any]] = []
    for row in rows:
        metrics_path = Path(row["path"]) / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        for group_type, key in [("aspect", "per_aspect"), ("language", "per_language")]:
            for group, values in metrics.get(key, {}).items():
                output = {field: row.get(field, "") for field in BREAKDOWN_FIELDS}
                output.update(
                    {
                        "group_type": group_type,
                        "group": group,
                        "precision": values.get("precision", ""),
                        "recall": values.get("recall", ""),
                        "f1": values.get("f1", ""),
                        "predicted": values.get("predicted", ""),
                        "expected": values.get("expected", ""),
                        "correct": values.get("correct", ""),
                    }
                )
                breakdowns.append(output)
    return breakdowns


def write_summary(
    rows: list[dict[str, Any]],
    output_dir: str | Path = REPORTS_DIR,
    *,
    make_plots: bool = True,
) -> Path:
    output_dir = ensure_dir(Path(output_dir))
    by_task_dir = ensure_dir(output_dir / "by_task")
    details_dir = ensure_dir(output_dir / "details")
    aggregate_rows = aggregate_by_seed(rows)
    breakdown_rows = collect_breakdowns(rows)

    _write_csv(output_dir / "summary.csv", rows, SUMMARY_FIELDS)
    aggregate_fields = GROUP_FIELDS + [
        "prompt_name",
        "n_runs",
        "seeds",
        "run_ids",
    ] + [
        f"{metric}_{suffix}"
        for metric in METRIC_FIELDS
        for suffix in ("mean", "std", "min", "max")
    ]
    _write_csv(output_dir / "summary_by_seed.csv", aggregate_rows, aggregate_fields)

    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        task_agg = [row for row in aggregate_rows if row["task"] == task]
        _write_csv(by_task_dir / f"{task}.csv", task_rows, SUMMARY_FIELDS)
        _write_csv(by_task_dir / f"{task}_by_seed.csv", task_agg, aggregate_fields)

    _write_csv(details_dir / "per_group.csv", breakdown_rows, BREAKDOWN_FIELDS)
    _write_csv(
        details_dir / "per_aspect.csv",
        [row for row in breakdown_rows if row["group_type"] == "aspect"],
        BREAKDOWN_FIELDS,
    )
    _write_csv(
        details_dir / "per_language.csv",
        [row for row in breakdown_rows if row["group_type"] == "language"],
        BREAKDOWN_FIELDS,
    )

    write_json(output_dir / "summary.json", rows)
    write_json(output_dir / "summary_by_seed.json", aggregate_rows)
    write_json(output_dir / "breakdowns.json", breakdown_rows)
    _write_markdown(rows, aggregate_rows, output_dir / "summary.md")

    if make_plots:
        _write_plots(rows, aggregate_rows, breakdown_rows, output_dir)
    return output_dir


def _format_score(value: Any) -> str:
    number = _safe_float(value)
    return "" if number is None else f"{number:.2f}"


def _row_label(row: dict[str, Any]) -> str:
    bits = [
        row.get("task", ""),
        row.get("model", ""),
        row.get("prompt_name") or _prompt_name(str(row.get("prompt", ""))),
    ]
    shots = str(row.get("shots", ""))
    if shots and shots != "0":
        bits.append(f"{shots}shot")
    quantization = row.get("quantization", "")
    if quantization and quantization != "none":
        bits.append(str(quantization))
    return ":".join(str(bit) for bit in bits if bit)


def _write_markdown(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["# ABSA Experiment Summary", ""]
    if not rows:
        lines.append("No reportable runs found.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    counts = Counter(row["task"] for row in rows)
    lines.append("## Coverage")
    lines.append("")
    lines.append("| task | runs |")
    lines.append("|---|---:|")
    for task, count in sorted(counts.items()):
        lines.append(f"| {task} | {count} |")

    lines.append("")
    lines.append("## Best Individual Runs")
    lines.append("")
    lines.append("| task | model | prompt | shots | strategy | quant | seed | macro F1 | micro F1 |")
    lines.append("|---|---|---|---:|---|---|---:|---:|---:|")
    scored = [row for row in rows if _safe_float(row.get("macro_f1")) is not None]
    for row in sorted(scored, key=lambda item: float(item["macro_f1"]), reverse=True)[:20]:
        lines.append(
            f"| {row['task']} | {row['model']} | {row['prompt_name']} | {row['shots']} | "
            f"{row['shot_strategy']} | {row['quantization']} | {row['seed']} | "
            f"{_format_score(row['macro_f1'])} | {_format_score(row['micro_f1'])} |"
        )

    lines.append("")
    lines.append("## Best Seed-Aggregated Settings")
    lines.append("")
    lines.append("| task | model | prompt | shots | strategy | quant | seeds | n | macro F1 mean | std |")
    lines.append("|---|---|---|---:|---|---|---|---:|---:|---:|")
    for row in sorted(aggregate_rows, key=lambda item: float(item.get("macro_f1_mean") or -1.0), reverse=True)[:20]:
        lines.append(
            f"| {row['task']} | {row['model']} | {row['prompt_name']} | {row['shots']} | "
            f"{row['shot_strategy']} | {row['quantization']} | {row['seeds']} | {row['n_runs']} | "
            f"{_format_score(row['macro_f1_mean'])} | {_format_score(row['macro_f1_std'])} |"
        )

    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `summary.csv`: one row per run.")
    lines.append("- `summary_by_seed.csv`: same experimental setting aggregated over seeds.")
    lines.append("- `by_task/`: task-specific tables for zero-shot, few-shot, fine-tuning, and fine-tuned evaluation.")
    lines.append("- `details/per_aspect.csv` and `details/per_language.csv`: ABSA-specific error-analysis tables.")
    lines.append("- `plots/`: task-specific plots and report-ready comparisons.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug(value: Any) -> str:
    text = str(value).replace("\\", "/")
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in text).strip("-")


def _write_plots(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]], breakdown_rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        (output_dir / "plots.txt").write_text("matplotlib not available; plots skipped.\n", encoding="utf-8")
        return

    plots_dir = ensure_dir(output_dir / "plots")
    _plot_top_runs(rows, output_dir, plots_dir, plt)
    _plot_task_aggregates(aggregate_rows, plots_dir, plt)
    _plot_prompt_comparisons(aggregate_rows, plots_dir, plt)
    _plot_few_shot_by_shots(aggregate_rows, plots_dir, plt)
    _plot_best_run_breakdowns(rows, breakdown_rows, plots_dir, plt)


def _barh(path: Path, labels: list[str], values: list[float], plt: Any, *, title: str, xerr: list[float] | None = None) -> None:
    if not labels:
        return
    height = max(4, len(labels) * 0.35)
    plt.figure(figsize=(11, height))
    plt.barh(labels[::-1], values[::-1], xerr=xerr[::-1] if xerr else None, color="#356c8c")
    plt.xlabel("Macro F1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def _plot_top_runs(rows: list[dict[str, Any]], output_dir: Path, plots_dir: Path, plt: Any) -> None:
    scored = [row for row in rows if _safe_float(row.get("macro_f1")) is not None]
    top = sorted(scored, key=lambda row: float(row["macro_f1"]), reverse=True)[:20]
    labels = [f"{_row_label(row)}:{str(row['run_id'])[-8:]}" for row in top]
    values = [float(row["macro_f1"]) for row in top]
    _barh(plots_dir / "macro_f1_top_runs.png", labels, values, plt, title="Top Individual Runs")
    _barh(output_dir / "macro_f1_top_runs.png", labels, values, plt, title="Top Individual Runs")


def _plot_task_aggregates(aggregate_rows: list[dict[str, Any]], plots_dir: Path, plt: Any) -> None:
    for task in sorted({row["task"] for row in aggregate_rows}):
        task_rows = [row for row in aggregate_rows if row["task"] == task and _safe_float(row.get("macro_f1_mean")) is not None]
        top = sorted(task_rows, key=lambda row: float(row["macro_f1_mean"]), reverse=True)[:15]
        labels = [_row_label(row) for row in top]
        values = [float(row["macro_f1_mean"]) for row in top]
        xerr = [float(row.get("macro_f1_std") or 0.0) for row in top]
        _barh(plots_dir / f"{_slug(task)}_macro_f1_by_seed.png", labels, values, plt, title=f"{task}: Seed-Aggregated Macro F1", xerr=xerr)


def _plot_prompt_comparisons(aggregate_rows: list[dict[str, Any]], plots_dir: Path, plt: Any) -> None:
    best_by_prompt: dict[tuple[str, str, str], float] = {}
    for row in aggregate_rows:
        value = _safe_float(row.get("macro_f1_mean"))
        if value is None:
            continue
        key = (row.get("task", ""), row.get("model", ""), row.get("prompt_name", ""))
        best_by_prompt[key] = max(best_by_prompt.get(key, -1.0), value)

    for task in sorted({key[0] for key in best_by_prompt}):
        items = [
            (f"{model}:{prompt}", value)
            for (task_name, model, prompt), value in best_by_prompt.items()
            if task_name == task
        ]
        items = sorted(items, key=lambda item: item[1], reverse=True)
        labels = [label for label, _ in items]
        values = [value for _, value in items]
        _barh(plots_dir / f"{_slug(task)}_prompt_comparison.png", labels, values, plt, title=f"{task}: Best Macro F1 by Prompt")


def _plot_few_shot_by_shots(aggregate_rows: list[dict[str, Any]], plots_dir: Path, plt: Any) -> None:
    few_rows = [
        row for row in aggregate_rows
        if row.get("task") == "few_shot"
        and _safe_float(row.get("macro_f1_mean")) is not None
        and str(row.get("shots", "")).isdigit()
    ]
    if not few_rows:
        return

    best: dict[tuple[str, str, int], float] = {}
    for row in few_rows:
        key = (row.get("model", ""), row.get("quantization", ""), int(row.get("shots", 0)))
        best[key] = max(best.get(key, -1.0), float(row["macro_f1_mean"]))

    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (model, quantization, shots), value in best.items():
        series[(model, quantization)].append((shots, value))

    plt.figure(figsize=(9, 5))
    for (model, quantization), points in sorted(series.items()):
        points = sorted(points)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(xs, ys, marker="o", label=f"{model}:{quantization}")
    plt.xlabel("Few-shot examples")
    plt.ylabel("Best seed-aggregated macro F1")
    plt.title("Few-Shot Scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "few_shot_best_by_shots.png", dpi=170)
    plt.close()


def _plot_best_run_breakdowns(rows: list[dict[str, Any]], breakdown_rows: list[dict[str, Any]], plots_dir: Path, plt: Any) -> None:
    scored = [row for row in rows if _safe_float(row.get("macro_f1")) is not None]
    for task in sorted({row["task"] for row in scored}):
        task_rows = [row for row in scored if row["task"] == task]
        if not task_rows:
            continue
        best = max(task_rows, key=lambda row: float(row["macro_f1"]))
        for group_type in ["aspect", "language"]:
            group_rows = [
                row for row in breakdown_rows
                if row["run_id"] == best["run_id"] and row["group_type"] == group_type
            ]
            group_rows = [row for row in group_rows if _safe_float(row.get("f1")) is not None]
            if not group_rows:
                continue
            group_rows.sort(key=lambda row: float(row["f1"]))
            labels = [row["group"] for row in group_rows]
            values = [float(row["f1"]) for row in group_rows]
            _barh(
                plots_dir / f"{_slug(task)}_best_run_per_{group_type}.png",
                labels,
                values,
                plt,
                title=f"{task}: Per-{group_type} F1 for best run",
            )


def run_summarize(
    *,
    experiments_root: str | Path = RUNS_DIR,
    output_dir: str | Path = REPORTS_DIR,
    include_local: bool = False,
    make_plots: bool = True,
) -> Path:
    rows = scan_runs(experiments_root, include_local=include_local)
    return write_summary(rows, output_dir, make_plots=make_plots)
