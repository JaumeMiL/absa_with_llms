import json
from dataclasses import dataclass
from typing import Any

from .constants import ALLOWED_ASPECTS, ALLOWED_POLARITIES


ALLOWED_ASPECT_SET = set(ALLOWED_ASPECTS)
ALLOWED_POLARITY_SET = set(ALLOWED_POLARITIES)

# Aixo es per parsejar les respostes idealment
# rollo per extreure els jsons del text q genera el model 


@dataclass
class NormalizedPrediction:
    opinions: dict[str, str]
    valid_json: bool
    parse_error: str | None = None
    invalid_items: list[str] | None = None


def extract_json_object(raw):
    # aqui la gracia seria extreure el json del text i retornar-lo maco
    raise NotImplementedError("FDSFFADSFAS tinc son")


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

