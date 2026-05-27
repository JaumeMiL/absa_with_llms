import json
from dataclasses import dataclass
from typing import Any

from constants import ASPECTS, POLARITIES


ALLOWED_ASPECT_SET = set(ASPECTS)
ALLOWED_POLARITY_SET = set(POLARITIES)

# Aixo es per parsejar les respostes idealment
# rollo per extreure els jsons del text q genera el model 


@dataclass
class NormalizedPrediction:
    opinions: dict[str, str]
    valid_json: bool
    parse_error: str | None = None
    invalid_items: list[str] | None = None


def extract_json_object(raw):
    # Extracts a JSON dictionary from a raw LLM output string.

    # errors rapids fora
    if isinstance(raw, dict):
        return raw, None
    if raw is None:
        return {}, "empty prediction"
    if not isinstance(raw, str):
        return {}, f"prediction is {type(raw).__name__}, not str/dict"

    # borra les tonteries de markdown que fan els models de ```
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]  # Drop the top fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1] # Drop the bottom fence
        text = "\n".join(lines).strip()

    # Intenta directament carregar el json si pot
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
        return {}, f"parsed JSON is {type(parsed).__name__}, not object"
    except json.JSONDecodeError:
        pass  # If it fails, we move to the fallback strategy

    # si hi ha text abans, busca el primer { i intenta decodificar la resta
    start_idx = text.find("{")
    if start_idx != -1:
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(text[start_idx:])
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError:
            pass

    return {}, "could not parse JSON object"

def normalize_prediction(raw) -> NormalizedPrediction:
    # retorna un objecte de "normalized prediction" q basicament te els les opinions i si es un json valid i tal
    parsed, parse_error = extract_json_object(raw)

    valid_json = parse_error is None

    # bno pq nose si el model li dira opinions o gold o que
    if isinstance(parsed.get("opinions"), dict):
        parsed = parsed["opinions"]
    elif isinstance(parsed.get("gold"), dict):
        parsed = parsed["gold"]

    opinions: dict[str, str] = {}
    invalid: list[str] = []
    for aspect, polarity in parsed.items():
        aspect_norm = str(aspect).strip().lower().replace("-", "_").replace(" ", "_") # per si de cas normalitzem q no me'n fio dels models
        polarity_norm = str(polarity).strip().lower()
        if aspect_norm in ALLOWED_ASPECT_SET and polarity_norm in ALLOWED_POLARITY_SET:
            opinions[aspect_norm] = polarity_norm
        else:
            invalid.append(f"{aspect}:{polarity}")
    return NormalizedPrediction(opinions=opinions,valid_json=valid_json,parse_error=parse_error,invalid_items=invalid,)

