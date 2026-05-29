from common import load_prompt, load_dataset, encode, generate, extract_json
import random

# ------------ select shots ------------

def select_shots(train, num_shots, strategy, seed):
    if num_shots <= 0:
        return []
    if strategy == "first":
        return train[:num_shots]
    if strategy == "random":
        rng = random.Random(seed)
        train_list = list(train)
        rng.shuffle(train_list) # shuffle es inplace
        return train_list[:num_shots]
    if strategy == "diverse":
        return select_diverse(train, num_shots, seed=seed)
    if strategy == "coverage_rerank":
        return select_coverage_rerank(train, num_shots, seed=seed)
    raise NotImplementedError(f"Selection of fewshot examples via {strategy} not implemented")

def _features(example):
    feats = {f"lang:{example.get('language', 'unknown')}"}
    for aspect, polarity in (example.get("gold") or {}).items():
        feats.add(f"aspect:{aspect}")
        feats.add(f"polarity:{polarity}")
        feats.add(f"combo:{aspect}:{polarity}")
    return feats

def select_diverse(train, num_shots, seed):
    # diverse rollo seria assegurant-se que totes les possibles combinacions (o el màxim) estan cobertes
    # idk why some comments are in english and others in catalan tho i have multiple personalities
    rng = random.Random(seed)
    indexed = list(enumerate(train))
    rng.shuffle(indexed)
    selected = []
    covered = set() #set de les combinacions cobertes
    remaining = indexed

    while remaining and len(selected) < num_shots:
        best_pos = 0
        best_score = -1
        best_size = -1
        for pos, (_, example) in enumerate(remaining):
            feats = _features(example)
            score = len(feats - covered)
            if score > best_score or (score == best_score and len(feats) > best_size):
                best_pos = pos
                best_score = score
                best_size = len(feats)
        item = remaining.pop(best_pos)
        selected.append(item)
        covered.update(_features(item[1]))

    selected.sort(key=lambda item: item[0])
    return [example for _, example in selected]


WEAK_ASPECTS = {
    "location",
    "food_style_options",
    "drinks_quality",
    "food_prices",
    "restaurant_prices",
}

WEAK_POLARITIES = {"neutral", "conflict"}


def _coverage_score(example, covered):
    feats = _features(example)
    new = feats - covered
    gold = example.get("gold") or {}
    score = 2.0 * len(new) + 0.1 * len(gold)
    for aspect, polarity in gold.items():
        if aspect in WEAK_ASPECTS:
            score += 2.5
        if polarity in WEAK_POLARITIES:
            score += 2.0
        if aspect in WEAK_ASPECTS and polarity in WEAK_POLARITIES:
            score += 2.5
    if len(gold) >= 3:
        score += 0.5
    return score


def select_coverage_rerank(train, num_shots, seed):
    rng = random.Random(seed)
    indexed = list(enumerate(train))
    rng.shuffle(indexed)
    selected = []
    covered = set()
    remaining = indexed

    while remaining and len(selected) < num_shots:
        best_pos = 0
        best_score = float("-inf")
        for pos, (_, example) in enumerate(remaining):
            score = _coverage_score(example, covered)
            if score > best_score:
                best_pos = pos
                best_score = score
        item = remaining.pop(best_pos)
        selected.append(item)
        covered.update(_features(item[1]))

    selected.sort(key=lambda item: item[0])
    return [example for _, example in selected]

