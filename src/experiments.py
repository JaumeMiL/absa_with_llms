# Aquest fitxer que es de digui a llegir els fitxers i tal i que cridi al que
# toca en funcio del kind, si hi ha matrix que faci la combinatoria...

# que organitzi basicament

import json
from pathlib import Path
from typing import Any

from synthetic import run_augmentation
from common import get_nested, prompt_label
from inference import run_inference
from constants import ARTIFACTS_DIR, RUNS_DIR
from train import run_training



def write_expanded_matrix(config_path, output_path = None):
    # La idea esq si el config es de sweep, expandegi amb les combinacions no?
    return NotImplementedError("Sweep not impleneted yet")


def run_experiment_config(config):
    kind = config.get("kind")
    if kind == "inference":
        return _run_inference_config(config)
    if kind == "augmentation":
        return _run_augmentation_config(config)
    if kind == "finetune":
        return _run_finetune_config(config)
    raise ValueError(f"Unknown experiment kind: {kind}")


def _run_name(config, task, prompt_file = None):
    # Run name for the outputs :) es la carpeta on es guarda
    name = config.get("name")
    if not name:
        return None
    seed = get_nested(config, "seed")
    model = get_nested(config, "model.name", "model")
    pieces = [name, task, model]
    if prompt_file:
        pieces.append(prompt_label(prompt_file))
    quantization = get_nested(config, "model.quantization")
    if quantization:
        pieces.append(f"q{quantization}")
    shots = get_nested(config, "few_shot.shots")
    if shots:
        pieces.append(f"{shots}shots")
        pieces.append(str(get_nested(config, "few_shot.strategy", "")))
    temperature = get_nested(config, "decoding.temperature")
    top_p = get_nested(config, "decoding.top_p")
    if temperature is not None:
        pieces.append(f"t{temperature}")
    if top_p is not None:
        pieces.append(f"p{top_p}")
    learning_rate = get_nested(config, "training.learning_rate")
    epochs = get_nested(config, "training.epochs")
    if learning_rate is not None:
        pieces.append(f"lr{learning_rate}")
    if epochs is not None:
        pieces.append(f"ep{epochs}")
    if seed is not None:
        pieces.append(f"seed{seed}")
    return "-".join(_slug(piece) for piece in pieces if piece)


def _slug(value):
    # estava en unaltre fitxer tb pero no recordo
    text = str(value)
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "-" for char in text).strip("-")


def _run_inference_config(config: dict[str, Any]) -> Path:
    prompt_file = get_nested(config, "prompt.file", "configs/prompts/zero_shot/baseline.yaml")
    shots = int(get_nested(config, "few_shot.shots", 0) or 0)
    mode = "few" if shots else "zero"
    return run_inference(
        mode=mode,
        model_name=get_nested(config, "model.name", "llama32B3"),
        model_path=get_nested(config, "model.path"),
        prompt_name=prompt_file,
        split=config.get("split", "devel"),
        output_root=config.get("output_root", config.get("output", str(RUNS_DIR))),
        run_name=_run_name(config, "few_shot" if shots else "zero_shot", prompt_file),
        shots=shots,
        shot_strategy=get_nested(config, "few_shot.strategy", "first"),
        temperature=float(get_nested(config, "decoding.temperature", 0.0)),
        top_p=float(get_nested(config, "decoding.top_p", 1.0)),
        max_new_tokens=int(get_nested(config, "decoding.max_new_tokens", 128)),
        seed=int(get_nested(config, "seed", 13)),
        quantization=get_nested(config, "model.quantization", "none"),
        adapter=get_nested(config, "model.adapter"),
        limit=config.get("limit"),
    )

def _run_augmentation_config(config):
    return run_augmentation(
        count=int(config.get("count", 1000)),
        seed=int(get_nested(config, "seed", 13)),
        model_name=get_nested(config, "model.name", "llama32B3"),
        prompt_name=get_nested(config, "prompt.file", "configs/prompts/generation/synthetic_reviews.yaml"),
        output_root=config.get("output_root", str(ARTIFACTS_DIR / "generated_data")),
        run_name=_run_name(config, "augmentation", get_nested(config, "prompt.file")),
        shots=int(get_nested(config, "few_shot.shots", 8)),
        shot_strategy=get_nested(config, "few_shot.strategy", "diverse"),
        temperature=float(get_nested(config, "decoding.temperature", 0.8)),
        top_p=float(get_nested(config, "decoding.top_p", 0.95)),
        max_new_tokens=int(get_nested(config, "decoding.max_new_tokens", 256)),
    )


def _run_finetune_config(config: dict[str, Any]) -> Path:
    # fua igual masses parametres jaja
    return run_training(
        model_name=get_nested(config, "model.name", "llama32B3"),
        model_path=get_nested(config, "model.path"),
        prompt_name=get_nested(config, "prompt.file", "configs/prompts/zero_shot/baseline.yaml"),
        train_data=get_nested(config, "data.train", "train"),
        devel_data=get_nested(config, "data.devel", "devel"),
        synthetic_data=get_nested(config, "data.synthetic"),
        output_root=config.get("output_root", str(RUNS_DIR)),
        run_name=_run_name(config, "finetune", get_nested(config, "prompt.file")),
        quantization=get_nested(config, "model.quantization", "none"), # NO EXISTEIX ENCARA
        seed=int(get_nested(config, "seed", 13)),
        max_length=int(get_nested(config, "training.max_length", 1024)),
        epochs=float(get_nested(config, "training.epochs", 3.0)),
        learning_rate=float(get_nested(config, "training.learning_rate", 2e-5)),
        per_device_train_batch_size=int(get_nested(config, "training.per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(get_nested(config, "training.per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(get_nested(config, "training.gradient_accumulation_steps", 8)),
        eval_accumulation_steps=int(get_nested(config, "training.eval_accumulation_steps", 4)),
        lora_r=int(get_nested(config, "lora.r", 8)),
        lora_alpha=int(get_nested(config, "lora.alpha", 16)),
        lora_dropout=float(get_nested(config, "lora.dropout", 0.05)),
        save_total_limit=int(get_nested(config, "training.save_total_limit", 2)),
        bf16=bool(get_nested(config, "training.bf16", True)),
        fp16=bool(get_nested(config, "training.fp16", False)),
    )
