import argparse
from _bootstrap import ABSA_ROOT
from common import load_config
from experiments import run_experiment_config

# C:/Users/Usuario/AppData/Local/Microsoft/WindowsApps/python3.11.exe c:/Users/Usuario/Documents/Universitat/8e_Quatri/MGL/practicamgl/absa_with_llms/scripts/run_experiment.py .\configs\experiments\synthetic\single_check.yaml

def main():
    parser = argparse.ArgumentParser(description="Run one ABSA YAML experiment config.")
    parser.add_argument("config", help="YAML config under ABSA/configs/experiments")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = run_experiment_config(config)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
