import json
import random

from .constants import ASPECTS, GENERATED_DIR, ensure_dir
from common import load_prompt, render_extraction_messages, render_generation_messages
from data import write_json, load_dataset
import time
from typing import Any

from pathlib import Path



from .constants import ALLOWED_ASPECTS, ALLOWED_POLARITIES
from .data import load_dataset, write_json
from .fewshot import select_shots
from .inference import generate_text, load_model_and_tokenizer

from .logging_utils import RunLogger

from .parsing import normalize_prediction
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


def generate_opinion_specs(count = 1000, seed = 42):
    rng = random.Random(seed)
    specs = []
    seen = {}
    polarities = list(POLARITY_WEIGHTS)
    weights = [POLARITY_WEIGHTS[polarity] for polarity in polarities]
    attempts = 0

    while len(specs) < count and attempts < count * 50:

        attempts += 1
        size = rng.choices([1, 2, 3, 4, 5, 6], weights=[8, 18, 30, 25, 14, 5], k=1)[0]
        # Controla quants aspects diferents tindrà aquesta opinio :) pesos son randoms

        aspects = rng.sample(list(ASPECTS), k=size)
        # sampleja de la llista d'aspectes en funcio del nombre que hi ha

        spec = {aspect: rng.choices(polarities, weights=weights, k=1)[0] for aspect in sorted(aspects)}
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


def run_augmentation(
        count = 1000,
        seed = 42,
        model_name = "llama32B3",
        prompt_name = "generation/synthetic_reviews",
        output_root = GENERATED_DIR,
        run_name = None,
        shots = 8, # té sentit fer few shot per generar no? que tingui exemples
        shot_strategy = "diverse",
        temperature = 0.8,
        top_p = 0.95,
        max_new_tokens = 1024, # nidea
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
        "prompt": prompt_name,
        "shots": shots,
        "shot_strategy": shot_strategy,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens
    }

    run_id = run_name or make_run_id("augment", params)
    run_dir = ensure_dir(Path(output_root) / run_id)
    ensure_dir(run_dir / "logs")
    logger = RunLogger(run_dir / "logs" / "run.log")

    # tots els logger realment podrien ser prints pero bno
    logger.log("Starting augmentation run", run_id=run_id, count=count, seed=seed)
    # json amb els parametres utiltizats
    write_json(run_dir / "run.json", {"run_id": run_id, "params": params})

    # genera els spects ooleole
    specs = generate_opinion_specs(count, seed=seed)
    write_json(run_dir / "opinion_specs.json", specs)
    logger.log("Wrote opinion specs", specs=len(specs))

    
    prompt = load_prompt(prompt_name)
    train = load_dataset("train")
    shot_examples = select_shots(train, shots, strategy=shot_strategy, seed=seed)
    logger.log("Loading generation model", model=model_name, shots=len(shot_examples))
    model, tokenizer = load_model_and_tokenizer(model_name, quantization="none")
    logger.log("Generation model loaded")


    synthetic = []
    for idx, opinions in enumerate(specs):
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == len(specs):
            logger.log("Generating synthetic reviews", done=idx + 1, total=len(specs), kept=len(synthetic))
        messages = render_generation_messages(prompt, opinions, shot_examples)
        raw_review = generate_text(model,tokenizer,messages,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        text = _clean_review(raw_review)
        item: dict[str, Any] = {
            "id": f"synt_{seed}_{idx:04d}",
            "language": "synthetic",
            "text": text,
            "gold": opinions,
            "source": "synthetic",
            "generation_run_id": run_id,}
        synthetic.append(item)

    write_json(run_dir / "synt.json", synthetic)
    train_plus = load_dataset("train") + synthetic
    write_json(run_dir / "train_plus_synth.json", train_plus)
    logger.log("Finished augmentation run", synthetic=len(synthetic), train_plus=len(train_plus))
    return run_dir