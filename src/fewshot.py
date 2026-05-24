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
    raise NotImplementedError(f"Selection of fewshot examples via {strategy} not implemented")

def select_diverse(train, num_shots, seed):
    # diverse rollo seria assegurant-se que totes les possibles combinacions (o el màxim) estan cobertes
    # idk why some comments are in english and others in catalan tho i have multiple personalities
    raise NotImplementedError("Selection of fewshot examples for diversity not implemented.")

# a tomar pel cul la resta de codi d'abans