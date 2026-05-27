import argparse

from _bootstrap import ABSA_ROOT
from common import expand_experiment_config, write_expanded_matrix


def main():
    parser = argparse.ArgumentParser(description="Expand one YAML experiment matrix to JSONL for Slurm arrays.")
    parser.add_argument("config", help="YAML config with a matrix section")
    parser.add_argument("--output", help="Optional JSONL output path")
    args = parser.parse_args()

    rows = expand_experiment_config(args.config)
    output = write_expanded_matrix(args.config, args.output)
    print(f"{output}")
    print(f"{len(rows)} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
