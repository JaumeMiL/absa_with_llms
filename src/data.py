# load data / save data
import json
from pathlib import Path
from .constants import DATASET_DIR

def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)

def write_json(path, data, indent = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")

def load_dataset(split_or_path):
    # Altre cop, si es un path perfecte sino va a buscar l'split a la carpeta de datasets
    candidate = Path(split_or_path)
    if candidate.exists():
        path = candidate
    else:
        name = str(split_or_path)
        if not name.endswith(".json"):
            name = f"{name}.json"
        path = DATASET_DIR / name
    examples = load_json(path)
    if not isinstance(examples, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")
    return examples

def merge_train_with_synthetic(train_path, synthetic_path):
    train = load_dataset(train_path)
    synthetic = load_dataset(synthetic_path)
    return train + synthetic
