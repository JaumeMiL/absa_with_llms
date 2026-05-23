import json
import torch
import yaml
from pathlib import Path
from constants import PROMPT_DIR, resolve_path

# ------------ get system and user prompts -----------------
def _prompt_path(name_or_path):
    # Utilitza Paths de pathlib per poder fer servir .exists
    path = Path(name_or_path)
    if path.exists():
        return path
    if path.suffix:
        resolved = resolve_path(path)
        if resolved.exists():
            return resolved
        return PROMPT_DIR / path

    normalized = str(name_or_path).replace("\\", "/") # windows fa coses rares

    candidates = [
        PROMPT_DIR / f"{normalized}.yaml",
        PROMPT_DIR / f"{normalized}.yml",
        PROMPT_DIR / f"{normalized}.json",
        PROMPT_DIR / "zero_shot" / f"{normalized}.yaml",
        PROMPT_DIR / "few_shot" / f"{normalized}.yaml",
        PROMPT_DIR / "generation" / f"{normalized}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(PROMPT_DIR.rglob(f"{Path(normalized).name}.yaml"))
    # rglob llista subdirectoris recurisvament que matchegin el pattern
    if matches:
        return matches[0]
    return candidates[0]

def load_prompt(name_or_path = "zero_shot/baseline"):
    path = _prompt_path(name_or_path)
    with path.open("r", encoding="utf-8") as handle: # encoding utf-8 sino feia coses rares / em petava a windows
        if path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    prompt = dict(raw)
    prompt["_path"] = str(path)

    if "system" not in prompt and "sysprompt" in prompt: # should work with old jsons :)
        prompt["system"] = prompt["sysprompt"]

    if "user_template" not in prompt and "usrprompt" in prompt:
        prompt["user_template"] = prompt["usrprompt"]
        
    if "system" not in prompt or "user_template" not in prompt:
        raise ValueError(f"Prompt must define system/user_template or sysprompt/usrprompt: {path}")
    
    return prompt

    

def _format_template(template , example = None, **values):
    # Pilla strings com "Hola {nom}, bon dia" i subtitueix {nom}
    example = example or {}
    # Pilla la info de l'exemple ?
    data = {
        "id": example.get("id", ""),
        "rid": example.get("rid", example.get("id", "")),
        "language": example.get("language", ""),
        "text": example.get("text", ""),
        "opinions_json": json.dumps(
            values.get("opinions", example.get("gold", example.get("opinions", {}))),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    # opinions es carrega o be dels values o be del gold de l'exemple o be de si l'exemple diu opinions pq ns
    # values de moment no ho faig servir però es per si vols passar valors esepcifics tambee
    data.update(values)

    rendered = template
    for key, value in data.items():
        rendered = rendered.replace("{{" + key + "}}", str(value)) # substitueix les keys pels valors, si el format es {{}}}
    try:
        rendered = rendered.format(**data) # PYTHON TE UN FORMAT PER DEFECTE QUE ES {}
        # PQ COLLONS FEI ASERVIR {{}}
    
    except (KeyError, ValueError):
        pass

    if "{text}" not in template and "{{text}}" not in template and "<text>{{text}}</text>" not in template:
        if data["text"]:
            rendered = f"{rendered.rstrip()}\n\nTEXT:\n{data['text']}" # a molt males afegeix el text tal qual amb intro
    
    return rendered


def render_user(prompt, example):
    return _format_template(prompt["user_template"], example)


def render_generation_user(prompt, opinions):
    # Example = None
    return _format_template(prompt["user_template"], None, opinions=opinions)


def render_extraction_messages(prompt, example, shots = None):
    # tots els que son render és pillar el template i posar-li realment el text que toca a les variables
    messages = [{"role": "system", "content": prompt["system"]}]
    for shot in shots or []: # shots és none pq sino python fa la merda aquella de passar la mateixa llista sempre a les variables, per si de cas
        messages.append({"role": "user", "content": render_user(prompt, shot)})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(shot.get("gold", {}), ensure_ascii=False, sort_keys=True),
            }
        )
    messages.append({"role": "user", "content": render_user(prompt, example)})
    return messages

       

def render_generation_messages(prompt, opinions, shots = None):
    messages = [{"role": "system", "content": prompt["system"]}]
    for shot in shots or []:
        messages.append({"role": "user", "content": render_generation_user(prompt, shot.get("gold", {}))})
        messages.append({"role": "assistant", "content": shot.get("text", "")})
    messages.append({"role": "user", "content": render_generation_user(prompt, opinions)})
    return messages


# ------------ load dataset (either "train", or "devel") -----------------
def load_dataset(data) :
    with open(f"dataset/{data}.json") as f : 
        examples = json.load(f)
    return examples

# ------------ prepare prompt messages for an example ----------------
def prepare_messages(prompts, question) :
    
    # First, system message
    messages = [{"role": "system", "content": prompts["system"]}]
 
    # then, add the text we want the anwer for, so the model will complete the response
    messages.append({"role": "user", 
                     "content": prompts["user_template"]
                                + "\nTEXT: "
                                + question['text']})
    return messages




# ------------ add review to prompt, tokenize, and encode ----------------
def encode(tokenizer, messages) :
    # convert the sequence of messages to an actual prompt in the format
    # expected by the model.
    # Tokenize and encode the prompt.
    input_ids = tokenizer.apply_chat_template(messages,
                                              tokenize=True,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to("cuda")
    return input_ids
    
    
    
# ------------ generate completion for given tokens ----------------
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

# ------------  # find '}' matching '{' in given position p ----------------
def find_matching_bracket(gen_text, p) :
    q = p+1
    c = 1
    while c!=0 and q<len(gen_text) :  # find '}' matching first '{'
       if gen_text[q]=='{' : c += 1
       elif gen_text[q]=='}' : c -= 1
       q += 1
    return q-1

# ------------ clean up response, extracting just the expected json part ----------------
def extract_json(gen_text):    

    # find first '{' and its matching closed bracket '}'
    p = gen_text.find("{") 
    q = find_matching_bracket(gen_text, p)
           
    # get text between first '{' and its matching '}'
    predic = gen_text[p:q+1].replace('“','"').replace('”','"')
    try :
        # load string as json object
        prediction = json.loads(predic)
    except json.JSONDecodeError as err:
        # json was missing or wrongly formatted
        print(repr(err))
        print(gen_text)
        prediction = {}
    return prediction

