
import os,sys,time,copy,json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from common import load_prompt, load_dataset, prepare_messages, encode, generate, extract_json
from constants import resolve_model_path, get_model_spec
# renamed to inference.py :)


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
    return NotImplementedError(":)")