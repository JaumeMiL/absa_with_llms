from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ABSA_ROOT


def infer_model_from_name(name: str) -> str:
    lowered = name.lower()
    if "qwen" in lowered:
        return "qwen25B3"
    return "llama32B3"


def infer_quant_from_name(name: str) -> str:
    lowered = name.lower()
    return "4bit" if "q4bit" in lowered or "4bit" in lowered else "none"


def make_row(adapter_dir: Path) -> dict:
    run_name = adapter_dir.parent.name
    model = infer_model_from_name(run_name)
    return {
        "kind": "inference",
        "name": f"final_eval_existing_adapter_{run_name}",
        "split": "devel",
        "output_root": "runs",
        "model": {
            "name": model,
            "quantization": infer_quant_from_name(run_name),
            "device": "auto",
            "adapter": str(adapter_dir).replace("\\", "/"),
        },
        "prompt": {"file": "configs/prompts/zero_shot/my_prompt.yaml"},
        "few_shot": {"shots": 0, "strategy": "first"},
        "decoding": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 192, "constraint": "none"},
        "repair_policy": "none",
        "seed": 13,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an inference JSONL for fine-tuned adapters.")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["runs/finetune"],
        help="Adapter roots to scan. Pass boada/runs/finetune explicitly for full historical re-evaluation.",
    )
    parser.add_argument("--output", default="artifacts/sweeps/final_existing_adapter_eval.jsonl")
    args = parser.parse_args()

    adapters = []
    for root in args.roots:
        root_path = (ABSA_ROOT / root).resolve()
        if root_path.exists():
            adapters.extend(sorted(path for path in root_path.rglob("adapter") if path.is_dir()))

    output = (ABSA_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with output.open("w", encoding="utf-8") as handle:
        for adapter in adapters:
            key = str(adapter)
            if key in seen:
                continue
            seen.add(key)
            handle.write(json.dumps(make_row(adapter), ensure_ascii=False, sort_keys=True) + "\n")
    print(output)
    print(f"{len(seen)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
