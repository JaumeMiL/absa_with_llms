from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from data import load_dataset, write_json
from inference import make_run_id, set_seed
from logging_utils import RunLogger
from constants import get_model_spec, resolve_model_path
from constants import RUNS_DIR, ensure_dir
from common import load_prompt, render_extraction_messages
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from transformers import TrainingArguments
from transformers import Trainer


def _ensure_chat_template(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not getattr(tokenizer, "chat_template", None): # el model q estava provant no tenia chat template jsjs
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'].upper() + ':\\n' + message['content'] + '\\n\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}ASSISTANT:\\n{% endif %}"
        )


# DFDSAFASDFA HAURIA DE SER LA MATEIXA QUE LOAD MODEL AND TOKENIZER SEGURAMENT! PASSA QUE
# AQUELLA NO TE SUPORT PER LORA PARAMETRES
def load_training_model_and_tokenizer(model_name, model_path_override, quantization, lora_r, lora_alpha, lora_dropout):
    # TODO: quantization
    spec = get_model_spec(model_name)
    model_path = resolve_model_path(model_name, model_path_override)

    kwargs: dict[str, Any] = {"trust_remote_code": spec.trust_remote_code}

    kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and not spec.local_smoke else torch.float32

    # FIX: Explicitly map to CPU if CUDA is not available
    kwargs["device_map"] = "auto" if torch.cuda.is_available() else {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(spec.lora_targets),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=spec.trust_remote_code)
    tokenizer.truncation_side = "left"
    _ensure_chat_template(tokenizer)
    return model, tokenizer


def build_training_dataset(tokenizer, examples, prompt, max_length):
    # ERA L'OLD tokenize_dataset 
    # L'old tokenize dataset crec que feia algo malament, rollo no 
    # maskejava (posava -100) el prompt
    # no ens interessa predir el prompt lol, només  l'output, i que cross entropy no tingui en compte el prompt
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    eos = tokenizer.eos_token or ""
    for example in examples:
        messages = render_extraction_messages(prompt, example)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = json.dumps(example.get("gold", {}), ensure_ascii=False, sort_keys=True)
        full_text = prompt_text + answer + eos # afegeix EOS token q ns si fa falta
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        labels = list(tokenized["input_ids"])
        cutoff = min(len(prompt_ids), len(labels))
        labels[:cutoff] = [-100] * cutoff
        labels = [
            -100 if mask == 0 or token == tokenizer.pad_token_id else label
            for token, mask, label in zip(tokenized["input_ids"], tokenized["attention_mask"], labels)
        ]
        rows["input_ids"].append(tokenized["input_ids"])
        rows["attention_mask"].append(tokenized["attention_mask"])
        rows["labels"].append(labels)
    return Dataset.from_dict(rows)


def _training_arguments(output_dir, args):
    # abans estava ajuntat a create_trainer
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args["per_device_train_batch_size"],
        "per_device_eval_batch_size": args["per_device_eval_batch_size"],
        "gradient_accumulation_steps": args["gradient_accumulation_steps"],
        "eval_accumulation_steps": args["eval_accumulation_steps"],
        "learning_rate": args["learning_rate"],
        "num_train_epochs": args["epochs"],
        "save_total_limit": args["save_total_limit"],
        "load_best_model_at_end": True,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "label_names": ["labels"],
        "bf16": args["bf16"],
        "fp16": args["fp16"],
        "seed": args["seed"],
        "report_to": [],
    }
    try:
        return TrainingArguments(**kwargs, eval_strategy="epoch")
    except TypeError:
        return TrainingArguments(**kwargs, evaluation_strategy="epoch")


def _write_training_history(run_dir, log_history):
    # PLOTEAR EL TRAINING NO??? que sino es complicat veure si ha anat be tb
    history_path = run_dir / "training_history.csv"
    keys = sorted({key for row in log_history for key in row})
    if keys:
        with history_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(log_history)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    train_points = [(row.get("epoch"), row.get("loss")) for row in log_history if row.get("loss") is not None]
    eval_points = [(row.get("epoch"), row.get("eval_loss")) for row in log_history if row.get("eval_loss") is not None]
    if not train_points and not eval_points:
        return

    plt.figure(figsize=(8, 5))
    if train_points:
        xs, ys = zip(*train_points)
        plt.plot(xs, ys, marker="o", label="train loss")
    if eval_points:
        xs, ys = zip(*eval_points)
        plt.plot(xs, ys, marker="o", label="eval loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curve.png", dpi=160)
    plt.close()


def run_training(
    model_name = "llama32B3",
    prompt_name = "zero_shot/baseline",
    train_data = "train",
    devel_data = "devel",
    synthetic_data = None,
    output_root = RUNS_DIR,
    run_name = None,
    quantization = "none",
    model_path = None,
    seed = 13,
    max_length = 1024,
    epochs = 3.0,
    learning_rate = 2e-5,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps = 8,
    eval_accumulation_steps = 4,
    lora_r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    save_total_limit = 2,
    bf16 = True,
    fp16 = False,
):
    
    set_seed(seed)
    try:
        import torch

        if bf16 and not torch.cuda.is_available():
            bf16 = False
    except Exception:
        pass

    params = {
        "task": "finetune",
        "model": model_name,
        "prompt": prompt_name,
        "train_data": train_data,
        "devel_data": devel_data,
        "synthetic_data": synthetic_data,
        "quantization": quantization,
        "model_path": model_path,
        "seed": seed,
        "max_length": max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_accumulation_steps": eval_accumulation_steps,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "save_total_limit": save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
    }
    run_id = run_name or make_run_id("finetune", params)
    run_dir = ensure_dir(Path(output_root) / "finetune" / run_id)
    ensure_dir(run_dir / "logs")
    logger = RunLogger(run_dir / "logs" / "run.log")
    logger.log("Starting fine-tuning run", run_id=run_id, model=model_name, seed=seed, quantization=quantization)
    write_json(run_dir / "run.json", {"run_id": run_id, "params": params, "model_spec": get_model_spec(model_name).__dict__})

    prompt = load_prompt(prompt_name)
    train_examples = load_dataset(train_data)
    if synthetic_data:
        train_examples = train_examples + load_dataset(synthetic_data)
    devel_examples = load_dataset(devel_data)
    logger.log(
        "Loaded data and prompt",
        train_examples=len(train_examples),
        devel_examples=len(devel_examples),
        prompt=prompt.get("_path"),
        synthetic=synthetic_data or "",
    )

    logger.log("Loading training model")
    model, tokenizer = load_training_model_and_tokenizer(
        model_name,
        model_path_override=model_path,
        quantization=quantization,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    logger.log("Training model loaded")
    train_dataset = build_training_dataset(tokenizer, train_examples, prompt, max_length=max_length)
    eval_dataset = build_training_dataset(tokenizer, devel_examples, prompt, max_length=max_length)
    args = _training_arguments(run_dir / "checkpoints", params)
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    logger.log("Starting trainer.train")
    result = trainer.train()
    logger.log("Trainer finished")
    adapter_dir = ensure_dir(run_dir / "adapter")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    _write_training_history(run_dir, trainer.state.log_history)
    write_json(
        run_dir / "train_metrics.json",
        {
            "train_result": getattr(result, "metrics", {}),
            "train_examples": len(train_examples),
            "devel_examples": len(devel_examples),
            "adapter_dir": str(adapter_dir),
            "tensorboard_dir": str(run_dir / "tensorboard"),
            "training_history": str(run_dir / "training_history.csv"),
            "training_curve": str(run_dir / "training_curve.png"),
        },
    )
    logger.log("Finished fine-tuning run", adapter_dir=adapter_dir)
    return run_dir
