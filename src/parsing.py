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
    repaired_items: list[str] | None = None


SAFE_ASPECT_ALIASES = {
    "ambiance": "ambience",
    "atmosphere": "ambience",
    "decor": "ambience",
    "decoration": "ambience",
    "overall": "restaurant_general",
    "overall_experience": "restaurant_general",
    "general": "restaurant_general",
    "restaurant": "restaurant_general",
    "place": "restaurant_general",
    "staff": "service",
    "waiter": "service",
    "waiters": "service",
    "waitress": "service",
    "waitresses": "service",
    "hostess": "service",
    "attention": "service",
    "drink_prices": "drinks_prices",
    "drink_price": "drinks_prices",
    "drink_quality": "drinks_quality",
    "drink_options": "drinks_style_options",
    "drink_style_options": "drinks_style_options",
    "menu_options": "food_style_options",
    "food_options": "food_style_options",
    "food_variety": "food_style_options",
    "portion_size": "food_style_options",
    "portions": "food_style_options",
}

SAFE_POLARITY_ALIASES = {
    "mixed": "conflict",
    "both": "conflict",
    "conflicted": "conflict",
    "positive_negative": "conflict",
}

AMBIGUOUS_ASPECTS = {
    "price",
    "prices",
    "quality",
    "options",
    "style_options",
    "style",
    "food",
    "drinks",
    "drink",
}


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


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def repair_aspect(aspect: str, repair_policy: str = "none") -> str | None:
    if aspect in ALLOWED_ASPECT_SET:
        return aspect
    if repair_policy not in {"aliases", "aliases_fuzzy"}:
        return None
    if aspect in SAFE_ASPECT_ALIASES:
        return SAFE_ASPECT_ALIASES[aspect]
    if repair_policy != "aliases_fuzzy" or aspect in AMBIGUOUS_ASPECTS:
        return None

    distances = sorted((_levenshtein(aspect, allowed), allowed) for allowed in ASPECTS)
    best_distance, best_aspect = distances[0]
    second_distance = distances[1][0] if len(distances) > 1 else 999
    max_distance = 2 if len(aspect) <= 8 else 3
    if best_distance <= max_distance and second_distance - best_distance >= 2:
        return best_aspect
    return None


def repair_polarity(polarity: str, repair_policy: str = "none") -> str | None:
    if polarity in ALLOWED_POLARITY_SET:
        return polarity
    if repair_policy in {"aliases", "aliases_fuzzy"}:
        return SAFE_POLARITY_ALIASES.get(polarity)
    return None

def normalize_prediction(raw, repair_policy: str = "none") -> NormalizedPrediction:
    # retorna un objecte de "normalized prediction" q basicament te els les opinions i si es un json valid i tal
    parsed, parse_error = extract_json_object(raw)

    valid_json = parse_error is None

    # bno pq nose si el model li dira opinions o gold o que
    if isinstance(parsed.get("opinions"), dict):
        parsed = parsed["opinions"]
    elif isinstance(parsed.get("gold"), dict):
        parsed = parsed["gold"]
    elif "aspect" in parsed and "polarity" in parsed:
        parsed = {parsed.get("aspect"): parsed.get("polarity")}

    opinions: dict[str, str] = {}
    invalid: list[str] = []
    repaired: list[str] = []
    for aspect, polarity in parsed.items():
        aspect_norm = str(aspect).strip().lower().replace("-", "_").replace(" ", "_") # per si de cas normalitzem q no me'n fio dels models
        polarity_norm = str(polarity).strip().lower()
        repaired_aspect = repair_aspect(aspect_norm, repair_policy)
        repaired_polarity = repair_polarity(polarity_norm, repair_policy)
        if repaired_aspect in ALLOWED_ASPECT_SET and repaired_polarity in ALLOWED_POLARITY_SET:
            opinions[repaired_aspect] = repaired_polarity
            if repaired_aspect != aspect_norm or repaired_polarity != polarity_norm:
                repaired.append(f"{aspect}:{polarity}->{repaired_aspect}:{repaired_polarity}")
        else:
            invalid.append(f"{aspect}:{polarity}")
    return NormalizedPrediction(
        opinions=opinions,
        valid_json=valid_json,
        parse_error=parse_error,
        invalid_items=invalid,
        repaired_items=repaired,
    )

