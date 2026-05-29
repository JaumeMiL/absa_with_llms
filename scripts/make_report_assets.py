from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT = REPORTS / "report_assets"
FIGS = OUT / "figures"
TABLES = OUT / "tables"

ASPECTS = [
    "restaurant_general",
    "restaurant_prices",
    "food_quality",
    "food_prices",
    "food_style_options",
    "drinks_quality",
    "drinks_prices",
    "drinks_style_options",
    "ambience",
    "service",
    "location",
]
POLARITIES = ["positive", "negative", "neutral", "conflict"]

COLORS = {
    "zero": "#6C8EBF",
    "few": "#E6A23C",
    "ft": "#4C9F70",
    "ens": "#9B6BB3",
    "dark": "#263238",
    "muted": "#65727e",
    "grid": "#d9dee5",
    "accent": "#D65F5F",
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

TASK_LABELS = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "finetune_eval": "Fine-tuning",
    "ensemble_eval": "Ensemble",
}


def ensure_dirs() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#cfd7e3",
            "axes.labelcolor": COLORS["dark"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "axes.axisbelow": True,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS / f"{name}.png", dpi=220)
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def norm_str(value: object, default: str = "none") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip()
    return text if text else default


def load_summary() -> pd.DataFrame:
    path = REPORTS / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # The merged Boada/Vast report can contain the same run through two copied
    # folders. Keep one row per run id before any mean, count, or plot.
    df = df.drop_duplicates("run_id", keep="first").copy()
    for col in [
        "macro_f1",
        "micro_f1",
        "macro_empty_f1",
        "json_validity_rate",
        "empty_prediction_rate",
        "temperature",
        "top_p",
        "max_new_tokens",
        "seed",
        "shots",
        "epochs",
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
    ]:
        if col in df.columns:
            df[f"{col}_num"] = to_num(df[col])
    df["repair_policy_norm"] = df.get("repair_policy", pd.Series(index=df.index)).map(
        lambda x: norm_str(x, "none")
    )
    df["constraint_norm"] = df.get("constraint", pd.Series(index=df.index)).map(
        lambda x: norm_str(x, "none")
    )
    df["prompt_name"] = df["prompt_name"].fillna("unknown")
    df["shot_strategy"] = df.get("shot_strategy", pd.Series(index=df.index)).map(
        lambda x: norm_str(x, "")
    )
    df["quantization"] = df["quantization"].fillna("none")
    df["data_regime"] = df.apply(infer_data_regime, axis=1)
    df["lora_target"] = df.apply(infer_lora_target, axis=1)
    df["precision"] = df.apply(infer_precision, axis=1)
    df["ensemble_mode"] = df["run_id"].map(infer_ensemble_mode)
    return df


def infer_data_regime(row: pd.Series) -> str:
    text = " ".join(
        norm_str(row.get(k), "")
        for k in ["run_id", "adapter", "train_data", "synthetic_data", "path"]
    ).lower()
    if "weak_polarity_only" in text:
        return "Weak-polarity only"
    if "weak_polarity_synth" in text or "synthetic_weak_aspects" in text:
        return "GPT + weak Llama synth"
    if "language_fixed" in text:
        return "Llama synthetic fixed"
    if "synthetic_llama32b3_1000" in text:
        return "Llama synthetic"
    if "gpt55" in text or "gpt5.5" in text:
        return "GPT synthetic"
    if "train_only" in text or re.search(r"(^|[/\\_-])train($|[/\\_-])", text):
        return "Original train"
    return "Other/unclear"


def infer_lora_target(row: pd.Series) -> str:
    direct = norm_str(row.get("lora_target_modules"), "")
    if direct and direct != "none":
        return direct
    text = " ".join(norm_str(row.get(k), "") for k in ["run_id", "adapter", "path"]).lower()
    if "loraall_linear" in text or "all_linear" in text:
        return "all_linear"
    if "loraattention_qv" in text or "attention_qv" in text:
        return "attention_qv"
    if "loraattention_all" in text or "attention_all" in text:
        return "attention_all"
    if "loramlp" in text:
        return "mlp"
    return "unspecified"


def infer_precision(row: pd.Series) -> str:
    if str(row.get("bf16")).lower() == "true":
        return "bf16"
    if str(row.get("fp16")).lower() == "true":
        return "fp16"
    text = " ".join(norm_str(row.get(k), "") for k in ["run_id", "path"]).lower()
    if "bf16" in text:
        return "bf16"
    if "fp16" in text:
        return "fp16"
    return "unspecified"


def infer_ensemble_mode(run_id: str) -> str:
    text = str(run_id)
    if "intersection_high_precision" in text:
        return "intersection"
    if "agreement_then_tiebreak" in text:
        return "agreement"
    if "majority" in text:
        return "majority"
    if "union_high_recall" in text:
        return "union"
    return "other"


def path_for_run(row: pd.Series) -> Path:
    path = Path(str(row["path"]))
    if not path.is_absolute():
        path = ROOT / path
    return path


def fnum(value: float | int | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_table(
    df: pd.DataFrame,
    name: str,
    caption: str,
    label: str,
    align: str | None = None,
) -> None:
    align = align or ("l" + "r" * (len(df.columns) - 1))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(v) for v in row.tolist()) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
        "",
    ]
    (TABLES / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def bar_label(ax: plt.Axes, bars, values, fmt="{:.1f}", pad=0.7) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLORS["dark"],
            fontweight="bold",
        )


def plot_best_by_family(df: pd.DataFrame) -> pd.DataFrame:
    tasks = ["zero_shot", "few_shot", "finetune_eval", "ensemble_eval"]
    rows = []
    for task in tasks:
        sub = df[(df["task"] == task) & df["macro_f1_num"].notna()]
        if sub.empty:
            continue
        best = sub.sort_values("macro_f1_num", ascending=False).iloc[0]
        rows.append(
            {
                "Familia": TASK_LABELS[task],
                "Runs uniques": int(sub["run_id"].nunique()),
                "Millor macro-F1": best["macro_f1_num"],
                "Micro-F1": best["micro_f1_num"],
                "Run": best["run_id"],
            }
        )
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = [COLORS["zero"], COLORS["few"], COLORS["ft"], COLORS["ens"]][: len(out)]
    bars = ax.bar(out["Familia"], out["Millor macro-F1"], color=colors, width=0.68, zorder=3)
    ax.set_ylim(0, max(out["Millor macro-F1"]) + 10)
    ax.set_ylabel("Millor macro-F1")
    ax.set_title("Progressio dels millors sistemes per familia experimental")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    bar_label(ax, bars, out["Millor macro-F1"], fmt="{:.1f}", pad=1.0)
    for bar, n in zip(bars, out["Runs uniques"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            3.5,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    ax.text(
        0.01,
        -0.16,
        "Una fila per run_id unic. Aquesta figura es descriptiva: mostra el millor sistema trobat de cada familia.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
    )
    savefig(fig, "01_best_macro_f1_by_family")

    table = out.copy()
    table["Millor macro-F1"] = table["Millor macro-F1"].map(lambda x: fnum(x, 3))
    table["Micro-F1"] = table["Micro-F1"].map(lambda x: fnum(x, 3))
    table = table[["Familia", "Runs uniques", "Millor macro-F1", "Micro-F1"]]
    write_latex_table(
        table,
        "01_best_macro_f1_by_family",
        "Millor resultat obtingut per cada familia experimental. Les files estan deduplicades per run_id.",
        "tab:best-by-family",
    )
    return out


def short_run_label(row: pd.Series) -> str:
    task = TASK_LABELS.get(str(row.get("task")), str(row.get("task")))
    model = str(row.get("model", ""))
    prompt = str(row.get("prompt_name", "")).replace("_", " ")
    shots = row.get("shots_num")
    if task == "Ensemble":
        return short_ensemble_label(str(row.get("run_id", "")))
    if task == "Fine-tuning":
        regime = str(row.get("data_regime", ""))
        target = str(row.get("lora_target", ""))
        temp = row.get("temperature_num")
        top_p = row.get("top_p_num")
        return f"{model} | {regime} | {target} | t={temp:g}, p={top_p:g}"
    if task == "Few-shot":
        return f"{model} | {prompt} | {int(shots)} shots | {row.get('shot_strategy', '')}"
    return f"{model} | {prompt} | q={row.get('quantization', '')}"


def plot_run_landscape(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        df["task"].isin(["zero_shot", "few_shot", "finetune_eval", "ensemble_eval"])
        & df["macro_f1_num"].notna()
        & (df["repair_policy_norm"] == "none")
    ].copy()
    tasks = ["zero_shot", "few_shot", "finetune_eval", "ensemble_eval"]
    colors = {
        "zero_shot": COLORS["zero"],
        "few_shot": COLORS["few"],
        "finetune_eval": COLORS["ft"],
        "ensemble_eval": COLORS["ens"],
    }
    markers = {"llama32B3": "o", "qwen25B3": "s", "ensemble": "D"}
    rng = np.random.default_rng(21)
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    for i, task in enumerate(tasks):
        g = sub[sub["task"] == task]
        for model, mg in g.groupby("model"):
            x = np.full(len(mg), i) + rng.uniform(-0.22, 0.22, len(mg))
            ax.scatter(
                x,
                mg["macro_f1_num"],
                s=28 if task == "ensemble_eval" else 22,
                marker=markers.get(model, "o"),
                color=colors[task],
                alpha=0.42,
                linewidth=0,
                label=f"{TASK_LABELS[task]} / {model}" if i == 0 else None,
                zorder=3,
            )
        best = g.sort_values("macro_f1_num", ascending=False).iloc[0]
        ax.scatter(
            [i],
            [best["macro_f1_num"]],
            s=105,
            color=colors[task],
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
        ax.text(
            i,
            best["macro_f1_num"] + 1.15,
            f"{best['macro_f1_num']:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COLORS["dark"],
        )
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Paisatge complet de runs individuals")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        -0.14,
        "Cada punt es un run_id unic. Els punts amb contorn negre indiquen el millor run de cada familia.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
    )
    savefig(fig, "00_run_landscape_individual_runs")
    return sub


def plot_top_individual_runs(df: pd.DataFrame, n: int = 28) -> pd.DataFrame:
    sub = df[
        df["task"].isin(["zero_shot", "few_shot", "finetune_eval", "ensemble_eval"])
        & df["macro_f1_num"].notna()
        & (df["repair_policy_norm"] == "none")
    ].copy()
    sub = sub.sort_values("macro_f1_num", ascending=False).head(n).copy()
    sub["label"] = sub.apply(short_run_label, axis=1)
    color_map = {
        "zero_shot": COLORS["zero"],
        "few_shot": COLORS["few"],
        "finetune_eval": COLORS["ft"],
        "ensemble_eval": COLORS["ens"],
    }
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    y = np.arange(len(sub))[::-1]
    values = sub["macro_f1_num"].values[::-1]
    labels = sub["label"].values[::-1]
    colors = [color_map[t] for t in sub["task"].values[::-1]]
    ax.hlines(y, 0, values, color="#d7dde6", lw=2.0, zorder=1)
    ax.scatter(values, y, s=70, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(max(0, values.min() - 4), min(100, values.max() + 3))
    ax.set_title(f"Top {n} runs individuals")
    ax.grid(axis="x", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for yi, val in zip(y, values):
        ax.text(val + 0.25, yi, f"{val:.2f}", va="center", fontsize=8.2)
    savefig(fig, "00b_top_individual_runs")
    table = sub[["task", "model", "prompt_name", "macro_f1_num", "micro_f1_num", "run_id"]].copy()
    table = table.rename(
        columns={
            "task": "Familia",
            "model": "Model",
            "prompt_name": "Prompt",
            "macro_f1_num": "Macro-F1",
            "micro_f1_num": "Micro-F1",
            "run_id": "Run",
        }
    )
    table["Familia"] = table["Familia"].map(lambda x: TASK_LABELS.get(x, x))
    for c in ["Macro-F1", "Micro-F1"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table.head(12),
        "00b_top_individual_runs",
        "Top 12 runs individuals deduplicats per run_id.",
        "tab:top-individual-runs",
        align="lllrrl",
    )
    return sub


def plot_zero_prompt_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["task"] == "zero_shot")
        & df["macro_f1_num"].notna()
        & (df["repair_policy_norm"] == "none")
    ].copy()
    sub = sub[~sub["prompt_name"].isin(["test", "baseline", "unknown"])]
    order = (
        sub.groupby("prompt_name")["macro_f1_num"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    stats = (
        sub.groupby("prompt_name")
        .agg(
            Runs=("run_id", "nunique"),
            Mitjana=("macro_f1_num", "mean"),
            Desv=("macro_f1_num", "std"),
            Millor=("macro_f1_num", "max"),
        )
        .loc[order]
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    y = np.arange(len(order))
    means = stats["Mitjana"].values
    bars = ax.barh(y, means, color="#dce7f4", edgecolor=COLORS["blue"], height=0.62, zorder=3)
    rng = np.random.default_rng(7)
    for i, prompt in enumerate(order):
        vals = sub[sub["prompt_name"] == prompt]["macro_f1_num"].values
        jitter = rng.uniform(-0.20, 0.20, size=len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter, s=22, alpha=0.65, color=COLORS["blue"], linewidth=0)
        ax.text(
            means[i] + 0.7,
            i,
            f"{means[i]:.1f}  n={int(stats.loc[i, 'Runs'])}",
            va="center",
            fontsize=8.5,
            color=COLORS["dark"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("_", " ") for p in order])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Zero-shot: rendiment descriptiu per prompt")
    ax.grid(axis="x", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        -0.12,
        "Punts = runs individuals. Barres = mitjana. No es una ablatio causal: alguns prompts tenen sweeps de decoding.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
    )
    savefig(fig, "02_zero_shot_prompt_descriptive")

    table = stats.rename(columns={"prompt_name": "Prompt"})
    for c in ["Mitjana", "Desv", "Millor"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table.sort_values("Mitjana", ascending=False),
        "02_zero_shot_prompt_descriptive",
        "Resultats zero-shot per prompt. Taula descriptiva, no causal, perque alguns prompts tambe inclouen sweeps de decoding.",
        "tab:zero-prompt-descriptive",
        align="lrrrr",
    )
    return stats


def matched_delta(
    data: pd.DataFrame,
    factor: str,
    base_value: object,
    variant_value: object,
    controls: list[str],
) -> pd.DataFrame:
    cols = controls + [factor]
    grouped = (
        data[data["macro_f1_num"].notna()]
        .groupby(cols, dropna=False)
        .agg(macro=("macro_f1_num", "mean"), runs=("run_id", "nunique"))
        .reset_index()
    )

    def value_mask(series: pd.Series, value: object) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            try:
                return np.isclose(series.astype(float), float(value), equal_nan=False)
            except (TypeError, ValueError):
                pass
        return series.astype(str) == str(value)

    base = grouped[value_mask(grouped[factor], base_value)].copy()
    var = grouped[value_mask(grouped[factor], variant_value)].copy()
    base = base[controls + ["macro", "runs"]].rename(
        columns={"macro": "base", "runs": "base_runs"}
    )
    var = var[controls + ["macro", "runs"]].rename(
        columns={"macro": "variant", "runs": "variant_runs"}
    )
    out = base.merge(var, on=controls, how="inner")
    out["delta"] = out["variant"] - out["base"]
    return out


def controlled_zero_tables_and_plot(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["task"] == "zero_shot")
        & (df["repair_policy_norm"] == "none")
        & (df["constraint_norm"] == "none")
    ].copy()
    comparisons = [
        (
            "Qwen vs Llama",
            "model",
            "llama32B3",
            "qwen25B3",
            ["prompt_name", "quantization", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "CoT vs my_prompt",
            "prompt_name",
            "my_prompt",
            "cot",
            ["model", "quantization", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "Definitions vs my_prompt",
            "prompt_name",
            "my_prompt",
            "definitions",
            ["model", "quantization", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "Confusion guard vs my_prompt",
            "prompt_name",
            "my_prompt",
            "confusion_guard",
            ["model", "quantization", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "4bit vs no quant.",
            "quantization",
            "none",
            "4bit",
            ["model", "prompt_name", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
    ]
    rows = []
    for label, factor, base, variant, controls in comparisons:
        comp = matched_delta(sub, factor, base, variant, controls)
        rows.append(
            {
                "Comparacio": label,
                "Parelles": len(comp),
                "Base": comp["base"].mean(),
                "Variant": comp["variant"].mean(),
                "Delta": comp["delta"].mean(),
                "Desv. delta": comp["delta"].std(),
            }
        )
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(out))[::-1]
    deltas = out["Delta"].values[::-1]
    colors = [COLORS["green"] if d >= 0 else COLORS["red"] for d in deltas]
    ax.barh(y, deltas, color=colors, zorder=3)
    ax.axvline(0, color=COLORS["dark"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(out["Comparacio"].values[::-1])
    ax.set_xlabel("Delta macro-F1 aparellat")
    ax.set_title("Zero-shot: efectes controlats")
    ax.grid(axis="x", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    for yi, d, n in zip(y, deltas, out["Parelles"].values[::-1]):
        ax.text(d + (0.35 if d >= 0 else -0.35), yi, f"{d:+.1f} (n={n})", va="center", ha="left" if d >= 0 else "right", fontsize=8.5)
    savefig(fig, "02b_zero_shot_controlled_deltas")

    table = out.copy()
    for c in ["Base", "Variant", "Delta", "Desv. delta"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table,
        "02b_zero_shot_controlled_deltas",
        "Comparacions zero-shot aparellades, controlant la resta de variables disponibles.",
        "tab:zero-controlled",
        align="lrrrrr",
    )
    return out


def zero_definitions_decoding_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["task"] == "zero_shot")
        & (df["prompt_name"] == "definitions")
        & (df["quantization"] == "none")
        & (df["repair_policy_norm"] == "none")
    ].copy()
    base = sub[(sub["temperature_num"] == 0.0) & (sub["top_p_num"] == 1.0)]
    rows = []
    for (temp, top_p), group in sub.groupby(["temperature_num", "top_p_num"], dropna=False):
        if pd.isna(temp) or pd.isna(top_p):
            continue
        merged = base[["model", "seed_num", "macro_f1_num"]].rename(
            columns={"macro_f1_num": "base"}
        ).merge(
            group[["model", "seed_num", "macro_f1_num"]].rename(
                columns={"macro_f1_num": "variant"}
            ),
            on=["model", "seed_num"],
            how="inner",
        )
        if merged.empty:
            continue
        merged["delta"] = merged["variant"] - merged["base"]
        rows.append(
            {
                "Temp.": temp,
                "Top-p": top_p,
                "Parelles": len(merged),
                "Greedy base": merged["base"].mean(),
                "Variant": merged["variant"].mean(),
                "Delta": merged["delta"].mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values(["Temp.", "Top-p"])
    table = out.copy()
    for c in ["Temp.", "Top-p"]:
        table[c] = table[c].map(lambda x: fnum(x, 1))
    for c in ["Greedy base", "Variant", "Delta"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table,
        "03_zero_definitions_decoding",
        "Sweep de decoding amb el prompt definitions, controlant model i seed. La base es temperatura 0 i top-p 1.0.",
        "tab:zero-definitions-decoding",
        align="ccrrrr",
    )
    return out


def plot_few_shot_by_shots(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["task"] == "few_shot")
        & df["macro_f1_num"].notna()
        & (df["repair_policy_norm"] == "none")
    ].copy()
    sub = sub[sub["shots_num"].notna()]
    strategies = ["random", "first", "diverse", "coverage_rerank"]
    colors = {
        "random": COLORS["blue"],
        "first": COLORS["yellow"],
        "diverse": COLORS["red"],
        "coverage_rerank": COLORS["green"],
    }
    stats = (
        sub[sub["shot_strategy"].isin(strategies)]
        .groupby(["shot_strategy", "shots_num"])
        .agg(
            Runs=("run_id", "nunique"),
            Mitjana=("macro_f1_num", "mean"),
            Millor=("macro_f1_num", "max"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    rng = np.random.default_rng(9)
    for strategy in strategies:
        g = stats[stats["shot_strategy"] == strategy].sort_values("shots_num")
        if g.empty:
            continue
        ax.plot(
            g["shots_num"],
            g["Mitjana"],
            marker="o",
            lw=2.4,
            label=strategy.replace("_", " "),
            color=colors[strategy],
        )
        vals = sub[sub["shot_strategy"] == strategy]
        ax.scatter(
            vals["shots_num"] + rng.uniform(-0.25, 0.25, len(vals)),
            vals["macro_f1_num"],
            s=12,
            color=colors[strategy],
            alpha=0.22,
            linewidth=0,
        )
    ax.set_xlabel("Nombre de shots")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Few-shot: efecte del nombre d'exemples")
    ax.set_xticks(sorted(sub["shots_num"].dropna().unique()))
    ax.grid(True, axis="both", zorder=0)
    ax.legend(title="Selector")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        -0.15,
        "Linies = mitjana dels runs disponibles; punts = runs individuals. Les conclusions causals es recolzen en les taules aparellades.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
    )
    savefig(fig, "04_few_shot_macro_by_shots_strategy")
    return stats


def controlled_few_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[
        (df["task"] == "few_shot")
        & (df["repair_policy_norm"] == "none")
        & (df["constraint_norm"] == "none")
    ].copy()
    comparisons = [
        (
            "Qwen vs Llama",
            "model",
            "llama32B3",
            "qwen25B3",
            ["prompt_name", "quantization", "shots_num", "shot_strategy", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "4bit vs no quant.",
            "quantization",
            "none",
            "4bit",
            ["model", "prompt_name", "shots_num", "shot_strategy", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "first vs random",
            "shot_strategy",
            "random",
            "first",
            ["model", "prompt_name", "quantization", "shots_num", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "diverse vs random",
            "shot_strategy",
            "random",
            "diverse",
            ["model", "prompt_name", "quantization", "shots_num", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
        (
            "coverage rerank vs random",
            "shot_strategy",
            "random",
            "coverage_rerank",
            ["model", "prompt_name", "quantization", "shots_num", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        ),
    ]
    rows = []
    for label, factor, base, variant, controls in comparisons:
        comp = matched_delta(sub, factor, base, variant, controls)
        rows.append(
            {
                "Comparacio": label,
                "Parelles": len(comp),
                "Base": comp["base"].mean(),
                "Variant": comp["variant"].mean(),
                "Delta": comp["delta"].mean(),
                "Desv. delta": comp["delta"].std(),
            }
        )
    controlled = pd.DataFrame(rows)
    table = controlled.copy()
    for c in ["Base", "Variant", "Delta", "Desv. delta"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table,
        "04b_few_shot_controlled",
        "Comparacions few-shot aparellades, controlant la resta de variables disponibles.",
        "tab:few-controlled",
        align="lrrrrr",
    )

    rows = []
    random_sub = sub[sub["shot_strategy"] == "random"].copy()
    for a, b in [(1, 2), (2, 5), (5, 10), (10, 20), (20, 30)]:
        comp = matched_delta(
            random_sub,
            "shots_num",
            float(a),
            float(b),
            ["model", "prompt_name", "quantization", "shot_strategy", "temperature_num", "top_p_num", "max_new_tokens_num", "seed_num"],
        )
        rows.append(
            {
                "Increment": f"{a}->{b}",
                "Parelles": len(comp),
                "Base": comp["base"].mean(),
                "Mes shots": comp["variant"].mean(),
                "Delta": comp["delta"].mean(),
            }
        )
    increments = pd.DataFrame(rows)
    table = increments.copy()
    for c in ["Base", "Mes shots", "Delta"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table,
        "04c_few_shot_shot_increments_random",
        "Efecte aparellat d'afegir shots dins de l'estrategia random.",
        "tab:few-shot-increments",
        align="lrrrr",
    )
    return controlled, increments


def plot_augmentation_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_axis_off()
    boxes = [
        ("Specs aspecte-polaritat", "Aspectes validats\\n+ polaritats objectiu", 0.05, 0.54),
        ("Model generador", "Prompt invers\\namb llengua fixada", 0.30, 0.54),
        ("Ressenya sintetica", "Text + JSON\\nformat dataset", 0.55, 0.54),
        ("Dataset augmentat", "train + synth\\nper fine-tuning", 0.80, 0.54),
        ("Control qualitat", "Mostra manual\\n+ round-trip opcional", 0.43, 0.14),
    ]
    for title, body, x, y in boxes:
        rect = FancyBboxPatch(
            (x, y),
            0.16,
            0.22,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.4,
            edgecolor="#7891a8",
            facecolor="#f4f7fb",
        )
        ax.add_patch(rect)
        ax.text(x + 0.08, y + 0.155, title, ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["dark"])
        ax.text(x + 0.08, y + 0.075, body, ha="center", va="center", fontsize=8.8, color=COLORS["muted"])
    for x1, y1, x2, y2 in [
        (0.21, 0.65, 0.30, 0.65),
        (0.46, 0.65, 0.55, 0.65),
        (0.71, 0.65, 0.80, 0.65),
        (0.63, 0.54, 0.51, 0.36),
        (0.51, 0.36, 0.88, 0.54),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.5,
                color="#7891a8",
                connectionstyle="arc3,rad=0.0",
            )
        )
    ax.text(
        0.5,
        0.93,
        "Pipeline d'augmentacio sintetica per ABSA",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["dark"],
    )
    ax.text(
        0.5,
        0.04,
        "La variant final controla llengua, aspectes febles i polaritats minoritaries abans d'entrenar.",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    savefig(fig, "05_synthetic_augmentation_pipeline")


def _load_json_list(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("examples") or data.get("data") or []
    return [row for row in data if isinstance(row, dict)]


def _first_synthetic_match(pattern: str) -> Path | None:
    matches = sorted(p for p in ROOT.rglob("synt.json") if pattern.lower() in str(p).lower())
    return matches[0] if matches else None


def write_synthetic_dataset_audit() -> pd.DataFrame:
    candidates = [
        ("GPT synthetic 1000", ROOT / "artifacts" / "generated_data" / "gpt5.5_1000" / "synt.json"),
        ("Llama synthetic 1000 old", _first_synthetic_match("synthetic_llama32B3_1000-mgl1008")),
        ("Llama synthetic 1000 fixed", _first_synthetic_match("synthetic_llama32B3_1000_language_fixed")),
        ("Weak aspects/polarities 600", _first_synthetic_match("synthetic_weak_aspects_600")),
    ]
    rows = []
    for label, path in candidates:
        if path is None or not path.exists():
            continue
        examples = _load_json_list(path)
        languages = Counter(str(ex.get("language", "other")) for ex in examples)
        polarities = Counter()
        for ex in examples:
            opinions = ex.get("opinions") or ex.get("gold") or {}
            if isinstance(opinions, dict):
                polarities.update(str(value) for value in opinions.values())
        rows.append(
            {
                "Dataset": label,
                "Ex.": len(examples),
                "en": languages.get("en", 0),
                "es": languages.get("es", 0),
                "altre": sum(v for k, v in languages.items() if k not in {"en", "es"}),
                "pos": polarities.get("positive", 0),
                "neg": polarities.get("negative", 0),
                "neu": polarities.get("neutral", 0),
                "conf": polarities.get("conflict", 0),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        write_latex_table(
            out,
            "05b_synthetic_dataset_audit",
            "Auditoria dels datasets sintetics generats. Les columnes de polaritat compten parelles aspecte-polaritat, no exemples.",
            "tab:synthetic-dataset-audit",
            align="lrrrrrrrr",
        )
    return out


def plot_finetune_data_regime(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["task"] == "finetune_eval") & df["macro_f1_num"].notna()].copy()
    keep = [
        "Original train",
        "GPT synthetic",
        "Llama synthetic",
        "Llama synthetic fixed",
        "GPT + weak Llama synth",
        "Weak-polarity only",
    ]
    sub = sub[sub["data_regime"].isin(keep)]
    best = (
        sub.groupby(["model", "data_regime"])
        .agg(Best=("macro_f1_num", "max"), Runs=("run_id", "nunique"))
        .reset_index()
    )
    order = [r for r in keep if r in best["data_regime"].unique()]
    models = [m for m in ["llama32B3", "qwen25B3"] if m in best["model"].unique()]
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    model_colors = {"llama32B3": COLORS["green"], "qwen25B3": COLORS["blue"]}
    for i, model in enumerate(models):
        vals = []
        ns = []
        for regime in order:
            row = best[(best["model"] == model) & (best["data_regime"] == regime)]
            vals.append(float(row["Best"].iloc[0]) if not row.empty else np.nan)
            ns.append(int(row["Runs"].iloc[0]) if not row.empty else 0)
        pos = x + (i - (len(models) - 1) / 2) * width
        bars = ax.bar(pos, vals, width=width, label=model, color=model_colors[model], zorder=3)
        for bar, value, n in zip(bars, vals, ns):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                ax.text(bar.get_x() + bar.get_width() / 2, 2.5, f"n={n}", ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" synthetic", "\nsynth").replace("Original train", "Train\noriginal").replace("GPT + weak", "GPT + weak") for r in order])
    ax.set_ylabel("Millor macro-F1")
    ax.set_ylim(0, max(best["Best"]) + 10)
    ax.set_title("Fine-tuning: millor resultat observat per regim de dades")
    ax.grid(axis="y", zorder=0)
    ax.legend(title="Model")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.01,
        -0.20,
        "Figura descriptiva de millors configuracions. Les comparacions causals s'han de llegir juntament amb les taules controlades.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=COLORS["muted"],
    )
    savefig(fig, "06_finetune_best_by_data_regime")

    table = best.sort_values(["data_regime", "model"]).copy()
    table = table.rename(columns={"model": "Model", "data_regime": "Regim", "Best": "Millor macro-F1"})
    table["Millor macro-F1"] = table["Millor macro-F1"].map(lambda x: fnum(x, 3))
    write_latex_table(
        table[["Regim", "Model", "Runs", "Millor macro-F1"]],
        "06_finetune_best_by_data_regime",
        "Millor resultat observat en fine-tuning per regim de dades i model. Taula descriptiva.",
        "tab:finetune-data-regime",
        align="llrr",
    )
    return best


def finetune_controlled_tables_and_plot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[(df["task"] == "finetune_eval") & df["macro_f1_num"].notna()].copy()
    weak = sub[
        (sub["model"] == "llama32B3")
        & (sub["data_regime"] == "GPT + weak Llama synth")
        & (sub["prompt_name"] == "my_prompt")
        & (sub["quantization"] == "4bit")
        & (sub["lora_target"].isin(["attention_qv", "all_linear"]))
    ].copy()
    rows = []
    for (temp, top_p), group in weak.groupby(["temperature_num", "top_p_num"], dropna=False):
        vals = group.groupby("lora_target")["macro_f1_num"].max()
        if {"attention_qv", "all_linear"}.issubset(set(vals.index)):
            rows.append(
                {
                    "Temp.": temp,
                    "Top-p": top_p,
                    "attention_qv": vals["attention_qv"],
                    "all_linear": vals["all_linear"],
                    "Delta": vals["all_linear"] - vals["attention_qv"],
                }
            )
    lora = pd.DataFrame(rows).sort_values(["Temp.", "Top-p"])

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    labels = [f"t={r['Temp.']:.1f}\\np={r['Top-p']:.1f}" for _, r in lora.iterrows()]
    x = np.arange(len(lora))
    ax.plot(x, lora["attention_qv"], marker="o", lw=2.3, label="attention q/v", color=COLORS["yellow"])
    ax.plot(x, lora["all_linear"], marker="o", lw=2.3, label="all linear", color=COLORS["green"])
    for xi, delta in zip(x, lora["Delta"]):
        ax.text(xi, max(lora.iloc[xi]["attention_qv"], lora.iloc[xi]["all_linear"]) + 0.8, f"+{delta:.1f}", ha="center", fontsize=8.5, fontweight="bold", color=COLORS["green"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(min(lora["attention_qv"].min(), lora["all_linear"].min()) - 3, lora["all_linear"].max() + 5)
    ax.set_title("Ablacio controlada: LoRA q/v vs all-linear")
    ax.grid(axis="y", zorder=0)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    savefig(fig, "07_finetune_lora_target_controlled")

    table = lora.copy()
    for c in ["Temp.", "Top-p"]:
        table[c] = table[c].map(lambda x: fnum(x, 1))
    for c in ["attention_qv", "all_linear", "Delta"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table,
        "07_finetune_lora_target_controlled",
        "Comparacio controlada dels moduls LoRA en Llama amb dades sintetiques focalitzades.",
        "tab:finetune-lora-controlled",
        align="ccrrr",
    )

    best = sub[
        (sub["model"] == "llama32B3")
        & (sub["data_regime"] == "GPT + weak Llama synth")
        & (sub["lora_target"] == "all_linear")
        & (sub["prompt_name"] == "my_prompt")
        & (sub["quantization"] == "4bit")
    ].copy()
    decode = (
        best.groupby(["temperature_num", "top_p_num"])
        .agg(Macro=("macro_f1_num", "max"), Micro=("micro_f1_num", "max"))
        .reset_index()
        .sort_values(["temperature_num", "top_p_num"])
    )
    table = decode.rename(columns={"temperature_num": "Temp.", "top_p_num": "Top-p"}).copy()
    for c in ["Temp.", "Top-p"]:
        table[c] = table[c].map(lambda x: fnum(x, 1))
    for c in ["Macro", "Micro"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table,
        "07b_finetune_best_decoding",
        "Decoding sobre el millor adaptador Llama weak-synthetic all-linear.",
        "tab:finetune-decoding",
        align="ccrr",
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    labels = [f"t={r['temperature_num']:.1f}\np={r['top_p_num']:.1f}" for _, r in decode.iterrows()]
    x = np.arange(len(decode))
    ax.plot(x, decode["Macro"], marker="o", lw=2.6, color=COLORS["green"], label="Macro-F1", zorder=3)
    ax.plot(x, decode["Micro"], marker="s", lw=2.0, color=COLORS["blue"], label="Micro-F1", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_title("Millor adaptador: efecte del decoding")
    ax.grid(axis="y", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    for xi, macro in zip(x, decode["Macro"]):
        ax.text(xi, macro + 0.25, f"{macro:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    savefig(fig, "07b_finetune_best_decoding")
    return lora, decode


def plot_ensemble_comparison(df: pd.DataFrame) -> pd.DataFrame:
    ens = df[(df["task"] == "ensemble_eval") & df["macro_f1_num"].notna()].copy()
    ens = ens.sort_values("macro_f1_num", ascending=False).head(14).copy()
    best_ft = df[(df["task"] == "finetune_eval") & df["macro_f1_num"].notna()][
        "macro_f1_num"
    ].max()
    mode_color = {
        "majority": COLORS["purple"],
        "agreement": COLORS["blue"],
        "intersection": COLORS["red"],
        "union": COLORS["green"],
        "other": COLORS["grey"],
    }
    labels = ens["run_id"].map(short_ensemble_label).tolist()[::-1]
    values = ens["macro_f1_num"].tolist()[::-1]
    colors = [mode_color.get(m, COLORS["grey"]) for m in ens["ensemble_mode"].tolist()[::-1]]
    fig, ax = plt.subplots(figsize=(10.8, 7.2))
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, zorder=3)
    ax.axvline(
        best_ft,
        color=COLORS["dark"],
        linestyle="--",
        lw=1.6,
        label=f"Millor fine-tune ({best_ft:.1f})",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Ensembles: majority s'apropa, intersection perd recall")
    ax.grid(axis="x", zorder=0)
    ax.set_xlim(78, best_ft + 3.2)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    for yi, v in zip(y, values):
        ax.text(v + 0.35, yi, f"{v:.1f}", va="center", fontsize=8.5)
    savefig(fig, "08_ensemble_comparison")

    table = ens[["run_id", "ensemble_mode", "macro_f1_num", "micro_f1_num", "empty_prediction_rate_num"]].copy()
    table = table.rename(
        columns={
            "run_id": "Ensemble",
            "ensemble_mode": "Mode",
            "macro_f1_num": "Macro-F1",
            "micro_f1_num": "Micro-F1",
            "empty_prediction_rate_num": "Pred. buides",
        }
    )
    table["Ensemble"] = table["Ensemble"].map(short_ensemble_label)
    for c in ["Macro-F1", "Micro-F1", "Pred. buides"]:
        table[c] = table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        table,
        "08_ensemble_comparison",
        "Principals ensembles comparats amb el millor fine-tuning.",
        "tab:ensemble-comparison",
        align="llrrr",
    )
    return ens


def short_ensemble_label(run_id: str) -> str:
    text = str(run_id)
    text = text.replace("current_best_ensemble_", "")
    text = text.replace("final_ensemble_best_models-", "final ")
    text = text.replace("-repairnone", "")
    text = text.replace("-repairaliases", " + aliases")
    text = text.replace("agreement_then_tiebreak", "agreement")
    text = text.replace("intersection_high_precision", "intersection")
    text = text.replace("_", " ")
    return text


def prf(pred: int, gold: int, correct: int) -> tuple[float, float, float]:
    precision = 100.0 * correct / pred if pred else 0.0
    recall = 100.0 * correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def read_predictions(row: pd.Series) -> list[dict]:
    path = path_for_run(row) / "predictions.jsonl"
    examples = []
    if not path.exists():
        return examples
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def selected_best_runs(df: pd.DataFrame) -> dict[str, pd.Series]:
    selected = {}
    for task, label in [
        ("zero_shot", "Zero-shot"),
        ("few_shot", "Few-shot"),
        ("finetune_eval", "Fine-tune"),
        ("ensemble_eval", "Ensemble"),
    ]:
        sub = df[(df["task"] == task) & df["macro_f1_num"].notna()].copy()
        sub = sub[sub.apply(lambda r: (path_for_run(r) / "predictions.jsonl").exists(), axis=1)]
        if not sub.empty:
            selected[label] = sub.sort_values("macro_f1_num", ascending=False).iloc[0]
    return selected


def breakdown_from_predictions(examples: list[dict], group: str) -> pd.DataFrame:
    if group == "aspect":
        keys = ASPECTS
        getter = lambda pair: pair[0]
    elif group == "polarity":
        keys = POLARITIES
        getter = lambda pair: pair[1]
    elif group == "language":
        keys = sorted({str(e.get("language", "unknown")) for e in examples})
        getter = None
    else:
        raise ValueError(group)
    counts = {k: [0, 0, 0] for k in keys}
    if group == "language":
        for ex in examples:
            key = str(ex.get("language", "unknown"))
            pred = clean_opinions(ex.get("prediction", ex.get("strict_prediction", {})))
            gold = clean_opinions(ex.get("gold", {}))
            pred_pairs = set(pred.items())
            gold_pairs = set(gold.items())
            counts.setdefault(key, [0, 0, 0])
            counts[key][0] += len(pred_pairs)
            counts[key][1] += len(gold_pairs)
            counts[key][2] += len(pred_pairs & gold_pairs)
    else:
        for ex in examples:
            pred = clean_opinions(ex.get("prediction", ex.get("strict_prediction", {})))
            gold = clean_opinions(ex.get("gold", {}))
            pred_pairs = set(pred.items())
            gold_pairs = set(gold.items())
            ok_pairs = pred_pairs & gold_pairs
            for key in keys:
                counts[key][0] += sum(1 for p in pred_pairs if getter(p) == key)
                counts[key][1] += sum(1 for p in gold_pairs if getter(p) == key)
                counts[key][2] += sum(1 for p in ok_pairs if getter(p) == key)
    rows = []
    for key, (pred, gold, correct) in counts.items():
        precision, recall, f1 = prf(pred, gold, correct)
        rows.append(
            {
                "group": key,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted": pred,
                "expected": gold,
                "correct": correct,
            }
        )
    return pd.DataFrame(rows)


def clean_opinions(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out = {}
    for aspect, polarity in value.items():
        if aspect in ASPECTS and polarity in POLARITIES:
            out[str(aspect)] = str(polarity)
    return out


def plot_error_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = selected_best_runs(df)
    aspect_cols = []
    polarity_cols = []
    language_cols = []
    for label, row in selected.items():
        examples = read_predictions(row)
        a = breakdown_from_predictions(examples, "aspect").set_index("group")
        p = breakdown_from_predictions(examples, "polarity").set_index("group")
        l = breakdown_from_predictions(examples, "language").set_index("group")
        aspect_cols.append(a["f1"].rename(label))
        polarity_cols.append(p["f1"].rename(label))
        language_cols.append(l["f1"].rename(label))
    aspect = pd.concat(aspect_cols, axis=1).loc[ASPECTS]
    polarity = pd.concat(polarity_cols, axis=1).loc[POLARITIES]
    language = pd.concat(language_cols, axis=1)

    expected = breakdown_from_predictions(read_predictions(selected["Fine-tune"]), "aspect").set_index("group")["expected"]
    aspect_plot = aspect.copy()
    ylabels = [f"{a.replace('_', ' ')}  (n={int(expected.get(a, 0))})" for a in aspect_plot.index]
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    im = ax.imshow(aspect_plot.values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(aspect_plot.shape[1]))
    ax.set_xticklabels(aspect_plot.columns)
    ax.set_yticks(np.arange(aspect_plot.shape[0]))
    ax.set_yticklabels(ylabels)
    ax.set_title("Error analysis: F1 per aspecte als millors sistemes")
    for i in range(aspect_plot.shape[0]):
        for j in range(aspect_plot.shape[1]):
            val = aspect_plot.iloc[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="white" if val > 62 else COLORS["dark"], fontsize=8.5, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("F1")
    savefig(fig, "09_error_analysis_per_aspect_heatmap")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(polarity.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(polarity.shape[1]))
    ax.set_xticklabels(polarity.columns)
    ax.set_yticks(np.arange(polarity.shape[0]))
    ax.set_yticklabels(polarity.index)
    ax.set_title("Error analysis: F1 per polaritat")
    for i in range(polarity.shape[0]):
        for j in range(polarity.shape[1]):
            val = polarity.iloc[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="white" if val > 60 else COLORS["dark"], fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("F1")
    savefig(fig, "10_error_analysis_per_polarity_heatmap")

    plot_polarity_confusion(selected["Fine-tune"], read_predictions(selected["Fine-tune"]))

    aspect_table = aspect.reset_index().rename(columns={"group": "Aspecte"})
    aspect_table.insert(1, "Gold", [int(expected.get(a, 0)) for a in aspect.index])
    for c in aspect_table.columns:
        if c not in ["Aspecte", "Gold"]:
            aspect_table[c] = aspect_table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        aspect_table,
        "09_error_analysis_per_aspect",
        "F1 per aspecte dels millors sistemes de cada familia. Gold indica el nombre de parelles gold de l'aspecte al devel.",
        "tab:per-aspect-best-systems",
        align="lrrrrr",
    )

    pol_expected = breakdown_from_predictions(read_predictions(selected["Fine-tune"]), "polarity").set_index("group")["expected"]
    polarity_table = polarity.reset_index().rename(columns={"group": "Polaritat"})
    polarity_table.insert(1, "Gold", [int(pol_expected.get(p, 0)) for p in polarity.index])
    for c in polarity_table.columns:
        if c not in ["Polaritat", "Gold"]:
            polarity_table[c] = polarity_table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        polarity_table,
        "10_error_analysis_per_polarity",
        "F1 per polaritat dels millors sistemes de cada familia.",
        "tab:per-polarity-best-systems",
        align="lrrrrr",
    )

    language_table = language.reset_index().rename(columns={"group": "Llengua"})
    for c in language_table.columns:
        if c != "Llengua":
            language_table[c] = language_table[c].map(lambda x: fnum(x, 3))
    write_latex_table(
        language_table,
        "11_per_language_best_systems",
        "F1 per llengua dels millors sistemes de cada familia.",
        "tab:per-language-best-systems",
        align="lrrrr",
    )
    return aspect, polarity, language


def plot_polarity_confusion(row: pd.Series, examples: list[dict]) -> None:
    rows = POLARITIES + ["not_gold"]
    cols = POLARITIES + ["omitted"]
    mat = pd.DataFrame(0, index=rows, columns=cols, dtype=int)
    for ex in examples:
        pred = clean_opinions(ex.get("prediction", ex.get("strict_prediction", {})))
        gold = clean_opinions(ex.get("gold", {}))
        for aspect in sorted(set(pred) | set(gold)):
            gold_pol = gold.get(aspect, "not_gold")
            pred_pol = pred.get(aspect, "omitted")
            if gold_pol in mat.index and pred_pol in mat.columns:
                mat.loc[gold_pol, pred_pol] += 1
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    im = ax.imshow(mat.values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Polaritat predita")
    ax.set_ylabel("Polaritat gold")
    ax.set_title("Fine-tune millor: matriu d'errors de polaritat")
    threshold = mat.values.max() * 0.55
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat.iloc[i, j])
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > threshold else COLORS["dark"], fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Comptes aspecte-polaritat")
    savefig(fig, "11_best_finetune_polarity_confusion")
    mat.to_csv(TABLES / "11_best_finetune_polarity_confusion.csv")


def write_best_finetune_provenance(df: pd.DataFrame) -> pd.DataFrame:
    eval_rows = df[(df["task"] == "finetune_eval") & df["macro_f1_num"].notna()].copy()
    if eval_rows.empty:
        return pd.DataFrame()
    best_eval = eval_rows.sort_values("macro_f1_num", ascending=False).iloc[0]
    adapter = Path(str(best_eval.get("adapter", ""))).parent.name
    train_rows = df[df["task"].eq("finetune_train")].copy()
    train_row = train_rows[train_rows["run_id"].eq(adapter)]
    if train_row.empty:
        train_row = train_rows[train_rows["run_id"].astype(str).str.contains(re.escape(adapter), regex=True, na=False)]

    rows = [
        ("Best eval run", best_eval["run_id"]),
        ("Macro-F1", fnum(best_eval["macro_f1_num"], 3)),
        ("Micro-F1", fnum(best_eval["micro_f1_num"], 3)),
        ("Adapter run", adapter or "--"),
        ("Data regime label", best_eval["data_regime"]),
    ]
    if not train_row.empty:
        tr = train_row.iloc[0]
        rows += [
            ("Base train data", norm_str(tr.get("train_data"), "--")),
            ("Additional synthetic data", norm_str(tr.get("synthetic_data"), "--")),
            ("Model", norm_str(tr.get("model"), "--")),
            ("Quantization", norm_str(tr.get("quantization"), "--")),
            ("Precision", infer_precision(tr)),
            ("LoRA target", infer_lora_target(tr)),
            ("Epochs", fnum(pd.to_numeric(pd.Series([tr.get("epochs")]), errors="coerce").iloc[0], 1)),
            ("Learning rate", norm_str(tr.get("learning_rate"), "--")),
        ]
    out = pd.DataFrame(rows, columns=["Camp", "Valor"])
    write_latex_table(
        out,
        "12_best_finetune_provenance",
        "Proveniencia del millor fine-tune. El millor model combina GPT synthetic amb dades weak-aspect generades amb Llama.",
        "tab:best-finetune-provenance",
        align="ll",
    )
    return out


def _short_path_label(value: object) -> str:
    text = norm_str(value, "")
    if not text:
        return "--"
    if "gpt5.5_1000" in text:
        return "gpt5.5_1000/train_plus_synth"
    if "synthetic_weak_aspects_600" in text:
        return "synthetic_weak_aspects_600 (Llama)"
    return Path(text).name or text


def write_best_finetune_short_table(df: pd.DataFrame) -> pd.DataFrame:
    eval_rows = df[(df["task"] == "finetune_eval") & df["macro_f1_num"].notna()].copy()
    if eval_rows.empty:
        return pd.DataFrame()
    best_eval = eval_rows.sort_values("macro_f1_num", ascending=False).iloc[0]
    adapter = Path(str(best_eval.get("adapter", ""))).parent.name
    train_rows = df[df["task"].eq("finetune_train")].copy()
    train_row = train_rows[train_rows["run_id"].eq(adapter)]
    tr = train_row.iloc[0] if not train_row.empty else best_eval
    rows = [
        ("Model", norm_str(best_eval.get("model"), "--")),
        ("Dades base", _short_path_label(tr.get("train_data"))),
        ("Dades sint. extra", _short_path_label(tr.get("synthetic_data"))),
        ("Quantitzacio", norm_str(tr.get("quantization"), "--")),
        ("Precisio", infer_precision(tr)),
        ("LoRA target", infer_lora_target(tr)),
        ("Decoding", f"temp={fnum(best_eval.get('temperature_num'), 1)}, top-p={fnum(best_eval.get('top_p_num'), 1)}"),
        ("Macro-F1", fnum(best_eval.get("macro_f1_num"), 3)),
        ("Micro-F1", fnum(best_eval.get("micro_f1_num"), 3)),
    ]
    out = pd.DataFrame(rows, columns=["Camp", "Valor"])
    write_latex_table(
        out,
        "12b_best_finetune_provenance_short",
        "Resum llegible de la configuracio del millor fine-tune.",
        "tab:best-finetune-short",
        align="ll",
    )
    return out


def _short_member_label(path: str) -> str:
    text = path.replace("\\", "/")
    name = Path(text).name
    if "llama_weak_polarity_synth" in text and "t0.1-p0.9" in text:
        return "Llama weak all-linear, t=0.1 p=0.9"
    if "llama_weak_polarity_synth" in text and "t0.1-p1.0" in text:
        return "Llama weak all-linear, t=0.1 p=1.0"
    if "llama_weak_polarity_synth" in text and "t0.0-p1.0" in text:
        return "Llama weak all-linear, greedy"
    if "finetune_gpt55_synth_10ep" in text:
        return "Llama GPT synthetic, 10 epochs"
    if "qwen_weak_only" in text:
        return "Qwen weak-polarity all-linear"
    if "qwen_required" in text:
        return "Qwen GPT synthetic QLoRA"
    if "few_shot" in text and "llama32B3" in text:
        return "Llama few-shot CoT 30 random"
    return name[:72]


def _family_from_member(path: str) -> str:
    if "few_shot" in path:
        return "Few-shot"
    if "finetune_eval" in path:
        return "Fine-tune"
    return "Prediccio"


def _summary_row_for_path(df: pd.DataFrame, member_path: str) -> pd.Series | None:
    target = member_path.replace("\\", "/").strip("/")
    paths = df["path"].fillna("").astype(str).str.replace("\\", "/", regex=False).str.strip("/")
    match = df[paths.eq(target)]
    if match.empty:
        run_name = Path(target).name
        match = df[df["run_id"].eq(run_name)]
    if match.empty:
        match = df[paths.map(lambda value: value.endswith(target) or target.endswith(value))]
    return None if match.empty else match.sort_values("macro_f1_num", ascending=False).iloc[0]


def write_best_ensemble_members(df: pd.DataFrame) -> pd.DataFrame:
    ens = df[(df["task"] == "ensemble_eval") & df["macro_f1_num"].notna()].copy()
    if ens.empty:
        return pd.DataFrame()
    best = ens.sort_values("macro_f1_num", ascending=False).iloc[0]
    run_json = path_for_run(best) / "run.json"
    if not run_json.exists():
        return pd.DataFrame()
    data = json.loads(run_json.read_text(encoding="utf-8"))
    members = data.get("params", {}).get("members", [])
    rows = []
    for member_path in members:
        row = _summary_row_for_path(df, str(member_path))
        rows.append(
            {
                "Membre": _short_member_label(str(member_path)),
                "Familia": _family_from_member(str(member_path)),
                "Macro-F1": fnum(row.get("macro_f1_num") if row is not None else None, 3),
            }
        )
    out = pd.DataFrame(rows)
    write_latex_table(
        out,
        "08b_best_ensemble_members",
        "Membres del millor ensemble, actual_best_mixed7-majority.",
        "tab:best-ensemble-members",
        align="llr",
    )
    return out


def write_notes(
    best: pd.DataFrame,
    zero_controlled: pd.DataFrame,
    few_controlled: pd.DataFrame,
    lora: pd.DataFrame,
) -> None:
    lines = [
        "# Report asset notes",
        "",
        "These figures and tables are generated from `reports/summary.csv`, deduplicated by `run_id`.",
        "Use global figures as descriptive context, and use the controlled tables for causal/ablation claims.",
        "",
        "Recommended interpretation:",
        "",
        "- Best-by-family is descriptive: it shows the strongest run found in each experimental family.",
        "- Zero-shot prompt bars are descriptive because `definitions` contains a decoding sweep; use `02b_zero_shot_controlled_deltas.tex` for controlled claims.",
        "- Few-shot shot/strategy plot is descriptive; use `04b_few_shot_controlled.tex` and `04c_few_shot_shot_increments_random.tex` for controlled claims.",
        "- Fine-tuning data-regime plot is best-observed, not a pure causal ablation; the strongest controlled fine-tuning evidence is the LoRA target comparison.",
        "- Per-aspect and per-polarity figures are recomputed from `predictions.jsonl` of the best system of each family.",
        "- The best fine-tune is not only Llama weak synthetic: its training run uses GPT synthetic train_plus_synth plus an additional weak-aspect Llama synthetic file.",
        "",
        "Top-line numbers:",
    ]
    for _, row in best.iterrows():
        lines.append(f"- {row['Familia']}: {row['Millor macro-F1']:.3f} macro-F1.")
    if not zero_controlled.empty:
        z = zero_controlled.set_index("Comparacio")
        if "CoT vs my_prompt" in z.index:
            lines.append(f"- Controlled zero-shot CoT delta: {z.loc['CoT vs my_prompt', 'Delta']:+.3f} macro-F1.")
    if not few_controlled.empty:
        f = few_controlled.set_index("Comparacio")
        if "Qwen vs Llama" in f.index:
            lines.append(f"- Controlled few-shot Qwen delta: {f.loc['Qwen vs Llama', 'Delta']:+.3f} macro-F1.")
    if not lora.empty:
        lines.append(f"- Controlled all-linear LoRA delta range: +{lora['Delta'].min():.3f} to +{lora['Delta'].max():.3f} macro-F1.")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup_style()
    df = load_summary()

    plot_run_landscape(df)
    plot_top_individual_runs(df)
    best = plot_best_by_family(df)
    plot_zero_prompt_descriptive(df)
    zero_controlled = controlled_zero_tables_and_plot(df)
    zero_definitions_decoding_table(df)
    plot_few_shot_by_shots(df)
    few_controlled, _ = controlled_few_tables(df)
    plot_augmentation_pipeline()
    write_synthetic_dataset_audit()
    plot_finetune_data_regime(df)
    lora, _ = finetune_controlled_tables_and_plot(df)
    plot_ensemble_comparison(df)
    write_best_ensemble_members(df)
    plot_error_analysis(df)
    write_best_finetune_provenance(df)
    write_best_finetune_short_table(df)
    write_notes(best, zero_controlled, few_controlled, lora)

    manifest = {
        "figures": sorted(p.name for p in FIGS.glob("*.png")),
        "tables": sorted(p.name for p in TABLES.glob("*.tex")),
        "source_summary": str(REPORTS / "summary.csv"),
        "deduplicated_runs": int(df["run_id"].nunique()),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest['figures'])} PNG figures and {len(manifest['tables'])} LaTeX tables to {OUT}")


if __name__ == "__main__":
    main()
