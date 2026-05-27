from __future__ import annotations

import argparse

from _bootstrap import ABSA_ROOT
from constants  import REPORTS_DIR, RUNS_DIR
from report import run_summarize


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ABSA runs into report tables and plots.")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--include-local", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = run_summarize(
        experiments_root=args.runs_dir,
        output_dir=args.output_dir,
        include_local=args.include_local,
        make_plots=not args.no_plots,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
