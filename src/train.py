from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from data import load_dataset, write_json
from inference import make_run_id, set_seed
from logging_utils import RunLogger
from constants import get_model_spec, resolve_model_path, normalize_quantization
from constants import RUNS_DIR, ensure_dir
from common import load_prompt, render_extraction_messages
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments
from transformers import Trainer
from modeling import build_model_load_kwargs


LORA_TARGET_POLICIES = {
    "attention_qv": ("q_proj", "v_proj"),
    "attention_all": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "all_linear": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
}


def resolve_lora_target_modules(model_name, policy_or_modules):
    spec = get_model_spec(model_name)
    if not policy_or_modules:
        return list(spec.lora_targets), "attention_qv"
    if isinstance(policy_or_modules, (list, tuple)):
        return list(policy_or_modules), ",".join(policy_or_modules)
    policy = str(policy_or_modules).strip()
    if policy in LORA_TARGET_POLICIES:
        return list(LORA_TARGET_POLICIES[policy]), policy
    if "," in policy:
        modules = [item.strip() for item in policy.split(",") if item.strip()]
        return modules, "custom"
    if spec.local_smoke:
        return list(spec.lora_targets), policy
    raise ValueError(f"Unknown LoRA target_modules policy: {policy_or_modules}")


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


def _as_token_ids(value):
    if hasattr(value, "ids"):
        return list(value.ids)
    if hasattr(value, "input_ids") and not isinstance(value, dict):
        return _as_token_ids(value.input_ids)
    if isinstance(value, dict):
        return _as_token_ids(value["input_ids"])
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


# DFDSAFASDFA HAURIA DE SER LA MATEIXA QUE LOAD MODEL AND TOKENIZER SEGURAMENT! PASSA QUE
# AQUELLA NO TE SUPORT PER LORA PARAMETRES
def load_training_model_and_tokenizer(model_name, model_path_override, quantization, lora_r, lora_alpha, lora_dropout, device="auto", lora_target_modules="attention_qv"):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quantization = normalize_quantization(quantization)
    spec = get_model_spec(model_name)
    model_path = resolve_model_path(model_name, model_path_override)
    target_modules, _ = resolve_lora_target_modules(model_name, lora_target_modules)

    kwargs: dict[str, Any] = build_model_load_kwargs(
        spec,
        quantization=quantization,
        device=device,
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True, **kwargs)
    model.config.use_cache = False
    if quantization == "4bit":
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
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
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for example in examples:
        messages = render_extraction_messages(prompt, example)
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_ids = _as_token_ids(prompt_ids)
        answer = json.dumps(example.get("gold", {}), ensure_ascii=False, sort_keys=True)
        answer_ids = _as_token_ids(tokenizer(answer + eos, add_special_tokens=False))

        # Preserve supervised assistant JSON tokens after truncation. The old
        # full-text left truncation could mask all labels and give nan eval loss.
        if len(answer_ids) >= max_length:
            input_ids = answer_ids[-max_length:]
            labels = list(input_ids)
        else:
            prompt_budget = max_length - len(answer_ids)
            kept_prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
            input_ids = kept_prompt_ids + answer_ids
            labels = [-100] * len(kept_prompt_ids) + list(answer_ids)

        attention_mask = [1] * len(input_ids)
        pad_amount = max_length - len(input_ids)
        if pad_amount > 0:
            input_ids = input_ids + [pad_id] * pad_amount
            attention_mask = attention_mask + [0] * pad_amount
            labels = labels + [-100] * pad_amount

        rows["input_ids"].append(input_ids)
        rows["attention_mask"].append(attention_mask)
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
        "warmup_ratio": args["warmup_ratio"],
        "lr_scheduler_type": args["lr_scheduler_type"],
        "max_grad_norm": args["max_grad_norm"],
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
    lora_target_modules = "attention_qv",
    save_total_limit = 2,
    bf16 = True,
    fp16 = False,
    warmup_ratio = 0.0,
    lr_scheduler_type = "linear",
    max_grad_norm = 1.0,
    early_stopping_patience = None,
    device = "auto",
):
    
    quantization = normalize_quantization(quantization)
    set_seed(seed)
    try:
        import torch

        if bf16 and not torch.cuda.is_available():
            bf16 = False
        elif bf16 and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            bf16 = False
            if not fp16:
                fp16 = True
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
        "device": device,
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
        "lora_target_modules": lora_target_modules,
        "save_total_limit": save_total_limit,
        "bf16": bf16,
        "fp16": fp16,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "max_grad_norm": max_grad_norm,
        "early_stopping_patience": early_stopping_patience,
    }
    run_id = run_name or make_run_id("finetune", params)
    run_dir = ensure_dir(Path(output_root) / "finetune" / run_id)
    if (run_dir / "train_metrics.json").exists() and (run_dir / "adapter").exists():
        return run_dir
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
        lora_target_modules=lora_target_modules,
        device=device,
    )
    logger.log("Training model loaded")
    train_dataset = build_training_dataset(tokenizer, train_examples, prompt, max_length=max_length)
    eval_dataset = build_training_dataset(tokenizer, devel_examples, prompt, max_length=max_length)
    args = _training_arguments(run_dir / "checkpoints", params)
    callbacks = []
    if early_stopping_patience not in {None, "", 0, "0"}:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience)))
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, callbacks=callbacks)
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
