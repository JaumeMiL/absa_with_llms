from __future__ import annotations

import argparse
import os

from _bootstrap import ABSA_ROOT
from experiments import load_matrix_row, run_experiment_config


def main():
    parser = argparse.ArgumentParser(description="Run one row from a JSONL sweep file.")
    parser.add_argument("jsonl", help="JSONL produced by scripts/make_sweep.py")
    parser.add_argument(
        "--index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
        help="Row index; defaults to SLURM_ARRAY_TASK_ID when present",
    )
    args = parser.parse_args()

    config = load_matrix_row(args.jsonl, args.index)
    print(config)
    run_dir = run_experiment_config(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
