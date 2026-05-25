from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from data import load_json, write_json
from constants import get_model_spec
from constants import REPORTS_DIR, RUNS_DIR, ensure_dir


def _flatten_run(run_path: Path) -> dict[str, Any] | None:
    metrics_path = run_path.parent / "metrics.json"
    if not metrics_path.exists():
        return None
    run = load_json(run_path)
    metrics = load_json(metrics_path)
    params = run.get("params", {})
    model_name = params.get("model", "")
    spec = get_model_spec(model_name) if model_name else None
    return {
        "run_id": run.get("run_id", run_path.parent.name),
        "task": params.get("task", run_path.parent.parent.name),
        "model": model_name,
        "reportable": bool(spec.reportable) if spec else False,
        "prompt": params.get("prompt", ""),
        "split": params.get("split", ""),
        "shots": params.get("shots", 0),
        "shot_strategy": params.get("shot_strategy", ""),
        "temperature": params.get("temperature", ""),
        "top_p": params.get("top_p", ""),
        "seed": params.get("seed", ""),
        "quantization": params.get("quantization", ""),
        "adapter": params.get("adapter", ""),
        "macro_f1": metrics["macro_assignment_compatible"]["f1"],
        "micro_f1": metrics["micro"]["f1"],
        "macro_empty_f1": metrics["macro_empty_is_perfect"]["f1"],
        "json_validity_rate": metrics.get("json_validity_rate", ""),
        "empty_prediction_rate": metrics.get("empty_prediction_rate", ""),
        "errors": metrics.get("errors", ""),
        "path": str(run_path.parent),
    }


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
    rows.sort(key=lambda row: (row["task"], -float(row["macro_f1"]), row["run_id"]))
    return rows


def write_summary(
    rows: list[dict[str, Any]],
    output_dir: str | Path = REPORTS_DIR,
    *,
    make_plots: bool = True,
) -> Path:
    output_dir = ensure_dir(Path(output_dir))
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    fieldnames = [
        "task",
        "run_id",
        "model",
        "prompt",
        "shots",
        "shot_strategy",
        "temperature",
        "top_p",
        "seed",
        "quantization",
        "macro_f1",
        "micro_f1",
        "json_validity_rate",
        "empty_prediction_rate",
        "path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = ["# ABSA Experiment Summary", ""]
    if rows:
        lines.append("| task | model | prompt | shots | temp | top_p | seed | macro F1 | micro F1 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            lines.append(
                f"| {row['task']} | {row['model']} | {row['prompt']} | {row['shots']} | "
                f"{row['temperature']} | {row['top_p']} | {row['seed']} | "
                f"{float(row['macro_f1']):.2f} | {float(row['micro_f1']):.2f} |"
            )
    else:
        lines.append("No reportable runs found.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if make_plots:
        _write_plot(rows, output_dir)
    write_json(output_dir / "summary.json", rows)
    return output_dir


def _write_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        (output_dir / "plots.txt").write_text("matplotlib not available; plots skipped.\n", encoding="utf-8")
        return

    top = sorted(rows, key=lambda row: float(row["macro_f1"]), reverse=True)[:20]
    labels = [f"{row['task']}:{row['model']}:{row['run_id'][-8:]}" for row in top]
    values = [float(row["macro_f1"]) for row in top]
    height = max(4, len(top) * 0.35)
    plt.figure(figsize=(10, height))
    plt.barh(labels[::-1], values[::-1], color="#2f7f8f")
    plt.xlabel("Macro F1")
    plt.tight_layout()
    plt.savefig(output_dir / "macro_f1_top_runs.png", dpi=160)
    plt.close()


def run_summarize(
    *,
    experiments_root: str | Path = RUNS_DIR,
    output_dir: str | Path = REPORTS_DIR,
    include_local: bool = False,
    make_plots: bool = True,
) -> Path:
    rows = scan_runs(experiments_root, include_local=include_local)
    return write_summary(rows, output_dir, make_plots=make_plots)
