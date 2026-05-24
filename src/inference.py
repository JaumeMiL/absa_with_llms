
import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from common import load_prompt, prepare_messages, encode, generate, extract_json
from constants import resolve_model_path, get_model_spec
from data import write_json, load_dataset
from logging_utils import RunLogger
from evaluate import save_evaluation, evaluate_examples
from parsing import normalize_prediction
from fewshot import select_shots
from constants import ensure_dir
from pathlib import Path
from common import render_extraction_messages

# renamed to inference.py :)
import random
import numpy as np
import time

# aixo estava tb a synthetic pero no puc importar-ho per merdes circulars, s'hauria de posar a common tot igual
#  pero common esta ple de merda ja 
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

def get_device(model):
    try:
        device = model.device # vaja suposo q hauria d'anar
        return device
    except Exception:
        pass
    return "cpu"

def generate_text(model, tokenizer, messages, temperature, top_p, max_new_tokens):
    input_ids = tokenizer.apply_chat_template(messages,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt",
                                              truncation=True,
                                            max_length=model.config.max_position_embeddings)
    input_ids = input_ids.to(get_device(model))

    do_sample = temperature is not None and temperature > 0 # que el meu huggingface es queixa sino

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        **( {"temperature": temperature, "top_p": top_p} if do_sample else {} ) # es podrien afegir mes arguments suposo
    }

    with torch.no_grad():
        gen_tokens = model.generate(input_ids, **generation_kwargs)
    
    promptlen = len(input_ids[0])
    # decode obtained tokens back into text
    gen_text = tokenizer.decode(gen_tokens[0][promptlen:], skip_special_tokens=True)
    return gen_text

"""
Com a referencia, antic feia


# ns en quin context necessitem realment la funcio de encode per separat :)
def encode(tokenizer, messages) :
    # convert the sequence of messages to an actual prompt in the format
    # expected by the model.
    # Tokenize and encode the prompt.
    input_ids = tokenizer.apply_chat_template(messages,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to("cuda")
    return input_ids


def generate(model, tokenizer, input_ids):
    # generate likely continuation (assistant answer)
    with torch.no_grad():
        gen_tokens = model.generate(input_ids,
                                    max_new_tokens=256,
                                    pad_token_id=tokenizer.eos_token_id
                                   )
    promptlen = len(input_ids[0])
    # decode obtained tokens back into text
    gen_text = tokenizer.decode(gen_tokens[0][promptlen:], skip_special_tokens=True)
    return gen_text

"""


# NOSE SI POSAR-LO AQUI O EN UN COMMON / BUILDERS o algo aixi pero bno
def load_model_and_tokenizer(model_name, quantization, adapter = None, device = "cpu", model_path = None):
    # Bno aqui hauria de fer lo de la funcio de baix de load_model xd

    model_path = resolve_model_path(model_name)
    spec = get_model_spec(model_name)
    kwargs = {
        "trust_remote_code": spec.trust_remote_code,
    }


    # quantization l'ha fet el chat o sigui que nidea de si esta be
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = device
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and not spec.local_smoke else torch.float32
        kwargs["device_map"] = device if torch.cuda.is_available() and not spec.local_smoke else None

    model = AutoModelForCausalLM.from_pretrained(model_path, use_safetensors=True, **kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=spec.trust_remote_code)
    if tokenizer.pad_token_id is None: # per si de cas ja el té definit, q el codi del profe el canviava sempre
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not getattr(tokenizer, "chat_template", None): # el model q faig servir per proves en local n ote definida chat template :D
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'].upper() + ':\\n' + message['content'] + '\\n\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}ASSISTANT:\\n{% endif %}"
        )

    return model, tokenizer


def run_inference(
    *,
    mode: str,
    model_name: str,
    prompt_name: str,
    split: str,
    output_root: str,
    run_name: str | None = None,
    shots: int = 0,
    shot_strategy: str = "first",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 128,
    seed: int = 13,
    quantization: str = "none",
    adapter: str | None = None,
    model_path: str | None = None,
    limit: int | None = None,
):
    set_seed(seed)
    task = "few_shot" if mode == "few" or shots else "zero_shot"

    # lo mateix q a syntetic, el dict aquest es per poder passar-ho pel nom fàcil i per poder guardar
    params = {
        "task": task,
        "mode": mode,
        "model": model_name,
        "prompt": prompt_name,
        "split": split,
        "shots": shots,
        "shot_strategy": shot_strategy,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "quantization": quantization,
        "adapter": adapter,
        "model_path": model_path,
        "limit": limit,
    }

    run_id = run_name or make_run_id(task, params)
    # les runs es creen a runs/(tipus flow zero_shot)/nom i alla hi ha prediccions, run, logs...
    run_dir = ensure_dir(Path(output_root) / task / run_id)

    # el logg realmet podria ser prints de python pq al boada hi ha el log de slurm pero en local no tinc jeje
    ensure_dir(run_dir / "logs")
    logger = RunLogger(run_dir / "logs" / "run.log")
    logger.log("Starting inference run", run_id=run_id, task=task, model=model_name, split=split, seed=seed)
    # el run.json guarda la configuracio basicament (parameters i model utilitzat)
    write_json(run_dir / "run.json", {"run_id": run_id, "params": params, "model_spec": get_model_spec(model_name).__dict__})

    prompt = load_prompt(prompt_name)
    examples = load_dataset(split)

    # limit pq aixi podia provar amb menys però realment podria petar-se
    if limit is not None:
        examples = examples[:limit]
    
    train = load_dataset("train")

    # IMPORTANT: ÉS EL MATEIX CODI PER ZERO SHOT I FEW SHOT, I AQUI ES ON TRIA ELS SHOTS!
    # NO fa lo d'haver d'executar dos fitxers per separat i guardar el FS.FDF raro aquell
    selected_shots = select_shots(train, shots, strategy=shot_strategy, seed=seed)
    logger.log("Loaded data and prompt", examples=len(examples), shots=len(selected_shots), prompt=prompt.get("_path"))

    logger.log("Loading model", model=model_name, quantization=quantization, adapter=adapter or "")
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        quantization=quantization,
        adapter=adapter,
        model_path=model_path,
    )
    logger.log("Model loaded")

    assignment_examples = []
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as jsonl:
        for idx, example in enumerate(examples):

            # AIXO ES PODRIA CONFIGRAR COM A PARAMETRE PER REPORTAR COM DE FREQUENT EL LOGGING PERÒ BNO
            if idx == 0 or (idx + 1) % 10 == 0 or idx + 1 == len(examples):
                logger.log("Generating predictions", done=idx + 1, total=len(examples))

            # els extraction messages son els de prediccio a partir de text
            # els generaiton messages son els de predir text a partir del gold q fa servir synthetic
            messages = render_extraction_messages(prompt, example, selected_shots)
            raw = generate_text(
                model,
                tokenizer,
                messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            normalized = normalize_prediction(raw)
            output = dict(example) # example es el text inicial + gold
            output["raw_prediction"] = raw # IMPORTANT: ns si raw te el prompt també inclos o no, sino s'hauria de treure
            output["prediction"] = normalized.opinions
            output["parse_error"] = normalized.parse_error
            output["invalid_items"] = normalized.invalid_items or []
            output["run_id"] = run_id
            output["example_index"] = idx
            jsonl.write(json.dumps(output, ensure_ascii=False) + "\n")
            assignment_examples.append(output)

    # guarda els json amb tots els outputs
    write_json(run_dir / "predictions.assignment.json", assignment_examples)

    # avalua a partir d'allo, evaluate examples lelgeix el key de prediction
    metrics, errors = evaluate_examples(assignment_examples)
    save_evaluation(run_dir / "metrics.json", run_dir / "errors.csv", metrics, errors)
    logger.log(
        "Finished inference run",
        macro_f1=f"{metrics['macro_assignment_compatible']['f1']:.2f}",
        micro_f1=f"{metrics['micro']['f1']:.2f}",
        errors=len(errors),
    )
    return run_dir