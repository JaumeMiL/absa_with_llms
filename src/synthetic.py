import json
import random
from collections import Counter

from constants import ASPECTS, POLARITIES, GENERATED_DIR, ensure_dir
from common import load_prompt, render_extraction_messages, render_generation_messages
from data import write_json, load_dataset
import time
from typing import Any

from pathlib import Path



from data import load_dataset, write_json
from fewshot import select_shots
from inference import generate_text, load_model_and_tokenizer

from logging_utils import RunLogger

from parsing import normalize_prediction
import numpy as np
import torch

# Genera specs randoms i a partir dallo fa sintetiques
# es podria fer tambe diferent, pillant
# exemples reals del dataset, borrant el text i dient que
# el generi, però no crees més varietat de aspects llavors






POLARITY_WEIGHTS = {
    "positive": 0.66,
    "negative": 0.22,
    "neutral": 0.05,
    "conflict": 0.07,
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
}

WEAK_POLARITY_WEIGHTS = {
    "positive": 0.25,
    "negative": 0.25,
    "neutral": 0.25,
    "conflict": 0.25,
}


def _weighted_distribution_from_counts(counts, default):
    total = sum(counts.values())
    if total <= 0:
        return dict(default)
    return {key: counts.get(key, 0) / total for key in default}


def _policy_distribution(policy, *, train_examples=None, field=None, default=None):
    default = default or {}
    if isinstance(policy, dict):
        total = sum(float(value) for value in policy.values())
        if total <= 0:
            return dict(default)
        return {str(key): float(value) / total for key, value in policy.items()}
    if policy == "balanced":
        keys = list(default)
        return {key: 1.0 / len(keys) for key in keys}
    if policy == "weak_polarity_boost":
        return dict(WEAK_POLARITY_WEIGHTS)
    if policy == "train_distribution" and train_examples is not None and field == "language":
        counts = Counter(example.get("language", "") for example in train_examples)
        return _weighted_distribution_from_counts(counts, default)
    if policy == "train_distribution" and train_examples is not None and field == "polarity":
        counts = Counter()
        for example in train_examples:
            for polarity in (example.get("gold") or {}).values():
                counts[str(polarity)] += 1
        return _weighted_distribution_from_counts(counts, default)
    return dict(default)


def _cycle_from_distribution(distribution, count, seed):
    rng = random.Random(seed)
    quotas = {key: int(round(float(weight) * count)) for key, weight in distribution.items()}
    while sum(quotas.values()) < count:
        key = max(distribution, key=lambda item: distribution[item] - quotas[item] / max(count, 1))
        quotas[key] += 1
    while sum(quotas.values()) > count:
        key = max(quotas, key=lambda item: quotas[item])
        quotas[key] -= 1
    values = [key for key, amount in quotas.items() for _ in range(max(amount, 0))]
    rng.shuffle(values)
    return values

def set_seed(seed):
    random.seed(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # fua ns si s'ha de posar també tecnicament el cuda backend deterministic
        # tampoc fa falta no ni que fos un article cientific

def _slug(value):
    text = str(value).replace("\\", "/").split("/")[-1] # extreu el nom del fitxer si és un fitxer
    # per cada caracter, guarda nomes alfanumerics (si guardo ? al no mdel fitxer per exemple github peta)
    # guions, punts _ ... les coses rares les canvio per un -
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "-" for char in text).strip("-")


def make_run_id(task, params):
    # run id per guardar a la carpeta i tal en funcio de parametres i time
    stamp = time.strftime("%Y%m%d-%H%M%S")
    parts = [task, str(params.get("model", "model")), str(params.get("prompt", "prompt"))]
    if params.get("shots"):
        parts.append(f"{params['shots']}shot")
    if "seed" in params:
        parts.append(f"s{params['seed']}")
    slug = "-".join(_slug(part) for part in parts if part)
    return f"{stamp}-{slug}"


def generate_opinion_specs(
    count = 1000,
    seed = 42,
    polarity_distribution = None,
    target_aspects = None,
    force_target_aspect = False,
):
    rng = random.Random(seed)
    specs = []
    seen = set()
    polarity_distribution = polarity_distribution or POLARITY_WEIGHTS
    polarities = [polarity for polarity in POLARITIES if polarity in polarity_distribution]
    weights = [polarity_distribution[polarity] for polarity in polarities]
    polarity_cycle = _cycle_from_distribution(polarity_distribution, count, seed + 101)
    target_aspects = [aspect for aspect in (target_aspects or []) if aspect in ASPECTS]
    attempts = 0

    while len(specs) < count and attempts < count * 50:

        attempts += 1
        size = rng.choices([1, 2, 3, 4, 5, 6], weights=[8, 18, 30, 25, 14, 5], k=1)[0]
        # Controla quants aspects diferents tindrà aquesta opinio :) pesos son randoms

        aspects = rng.sample(list(ASPECTS), k=size)
        if target_aspects and (force_target_aspect or rng.random() < 0.75):
            target = rng.choice(target_aspects)
            if target not in aspects:
                aspects[-1] = target
        # sampleja de la llista d'aspectes en funcio del nombre que hi ha

        spec = {aspect: rng.choices(polarities, weights=weights, k=1)[0] for aspect in sorted(aspects)}
        if polarity_cycle:
            forced_aspect = rng.choice(target_aspects) if target_aspects else rng.choice(list(spec))
            if forced_aspect not in spec:
                forced_aspect = rng.choice(list(spec))
            spec[forced_aspect] = polarity_cycle[len(specs) % len(polarity_cycle)]
        # per cada aspects, un rating en funcio dels weights (pq positius apareixen mes)

        key = json.dumps(spec, sort_keys=True, ensure_ascii=False) # he de convertir a json per poder guardar al set, no puc posar diccionaris en un set
        if key not in seen:
            seen.add(key)
            specs.append(spec) # aqui força a que tots siguin diferents (i per aixo tb el count*50)
            # pero realment no tindria pq
    if len(specs) < count:
        raise RuntimeError(f"Could only create {len(specs)} unique opinion specs")
    return specs

def _clean_review(text):
    cleaned = text.strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _synthetic_report(synthetic, specs, language_policy, polarity_policy):
    language_counts = Counter(item.get("language", "") for item in synthetic)
    aspect_counts = Counter()
    polarity_counts = Counter()
    for spec in specs:
        for aspect, polarity in spec.items():
            aspect_counts[aspect] += 1
            polarity_counts[polarity] += 1
    warnings = []
    total = max(len(synthetic), 1)
    for language, amount in language_counts.items():
        if amount / total > 0.80:
            warnings.append(f"language {language} is {amount / total:.1%} of synthetic examples")
    return {
        "examples": len(synthetic),
        "language_policy": language_policy,
        "polarity_policy": polarity_policy,
        "language_counts": dict(sorted(language_counts.items())),
        "aspect_counts": dict(sorted(aspect_counts.items())),
        "polarity_counts": dict(sorted(polarity_counts.items())),
        "warnings": warnings,
    }


def _write_synthetic_report(run_dir, report):
    write_json(run_dir / "synthetic_report.json", report)
    lines = [
        "# Synthetic Data Report",
        "",
        f"Examples: {report['examples']}",
        f"Language policy: `{report['language_policy']}`",
        f"Polarity policy: `{report['polarity_policy']}`",
        "",
        "## Languages",
        "",
    ]
    for language, amount in report["language_counts"].items():
        lines.append(f"- `{language}`: {amount}")
    lines += ["", "## Polarities", ""]
    for polarity, amount in report["polarity_counts"].items():
        lines.append(f"- `{polarity}`: {amount}")
    if report["warnings"]:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in report["warnings"])
    (run_dir / "synthetic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_augmentation(
        count = 1000,
        seed = 42,
        model_name = "llama32B3",
        model_path = None,
        quantization = "none",
        device = "auto",
        prompt_name = "generation/synthetic_reviews",
        output_root = GENERATED_DIR,
        run_name = None,
        shots = 8, # té sentit fer few shot per generar no? que tingui exemples
        shot_strategy = "diverse",
        temperature = 0.8,
        top_p = 0.95,
        max_new_tokens = 1024, # nidea
        language_policy = "train_distribution",
        polarity_policy = "train_distribution",
        target_aspects = None,
        force_target_aspect = False,
        # també es podria fer el que feia el profe de després fer inferencia amb el mateix
        # model i així tens els que tenen mes probabilitats d'estar be i tal pero ns
    ):
    set_seed(seed)

    # guardo els parametres pq aixi es mes facil guardar-los despres al log :)
    params = {
        "task": "augment",
        "count": count,
        "seed": seed,
        "model": model_name,
        "model_path": model_path,
        "quantization": quantization,
        "device": device,
        "prompt": prompt_name,
        "shots": shots,
        "shot_strategy": shot_strategy,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "language_policy": language_policy,
        "polarity_policy": polarity_policy,
        "target_aspects": target_aspects or [],
        "force_target_aspect": force_target_aspect,
    }

    run_id = run_name or make_run_id("augment", params)
    run_dir = ensure_dir(Path(output_root) / run_id)
    if (run_dir / "synt.json").exists() and (run_dir / "train_plus_synth.json").exists():
        return run_dir
    ensure_dir(run_dir / "logs")
    logger = RunLogger(run_dir / "logs" / "run.log")

    # tots els logger realment podrien ser prints pero bno
    logger.log("Starting augmentation run", run_id=run_id, count=count, seed=seed)
    # json amb els parametres utiltizats
    write_json(run_dir / "run.json", {"run_id": run_id, "params": params})

    train = load_dataset("train")
    language_distribution = _policy_distribution(
        language_policy,
        train_examples=train,
        field="language",
        default={"es": 0.68, "en": 0.32},
    )
    polarity_distribution = _policy_distribution(
        polarity_policy,
        train_examples=train,
        field="polarity",
        default=POLARITY_WEIGHTS,
    )
    languages = _cycle_from_distribution(language_distribution, count, seed + 17)

    # genera els spects ooleole
    specs = generate_opinion_specs(
        count,
        seed=seed,
        polarity_distribution=polarity_distribution,
        target_aspects=target_aspects,
        force_target_aspect=force_target_aspect,
    )
    write_json(run_dir / "opinion_specs.json", specs)
    logger.log("Wrote opinion specs", specs=len(specs))

    
    prompt = load_prompt(prompt_name)
    shot_examples = select_shots(train, shots, strategy=shot_strategy, seed=seed)
    logger.log("Loading generation model", model=model_name, quantization=quantization, device=device, shots=len(shot_examples))
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        quantization=quantization,
        model_path=model_path,
        device=device,
    )
    logger.log("Generation model loaded")


    synthetic = []
    for idx, opinions in enumerate(specs):
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(specs):
            logger.log("Generating synthetic reviews", done=idx + 1, total=len(specs), kept=len(synthetic))
        target_language = languages[idx] if idx < len(languages) else "es"
        target_language_name = LANGUAGE_NAMES.get(target_language, target_language)
        messages = render_generation_messages(
            prompt,
            opinions,
            shot_examples,
            target_language=target_language,
            target_language_name=target_language_name,
        )
        raw_review = generate_text(model,tokenizer,messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = _clean_review(raw_review)
        item: dict[str, Any] = {
            "id": f"synt_{seed}_{idx:04d}",
            "language": target_language,
            "target_language": target_language,
            "text": text,
            "gold": opinions,
            "source": "synthetic",
            "generation_run_id": run_id,}
        synthetic.append(item)

    write_json(run_dir / "synt.json", synthetic)
    report = _synthetic_report(synthetic, specs, language_policy, polarity_policy)
    _write_synthetic_report(run_dir, report)
    train_plus = train + synthetic
    write_json(run_dir / "train_plus_synth.json", train_plus)
    logger.log("Finished augmentation run", synthetic=len(synthetic), train_plus=len(train_plus))
    return run_dir
