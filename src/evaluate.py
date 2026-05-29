from __future__ import annotations

import sys
import json


import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from constants import ASPECTS
from data import load_json, write_json
from parsing import normalize_prediction


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    predicted: int
    expected: int
    correct: int


def _prf(predicted, expected, correct, empty_is_perfect = False):
    if empty_is_perfect and predicted == 0 and expected == 0:
        return PRF(100.0, 100.0, 100.0, predicted, expected, correct)
    precision = 100.0 * correct / predicted if predicted else 0.0
    recall = 100.0 * correct / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, predicted, expected, correct)


def opinion_pairs(opinions):
    return {(aspect, polarity) for aspect, polarity in opinions.items()}


def evaluate_examples(examples, repair_policy = "none"):
    rows = []
    errors = []
    total_pred = total_exp = total_ok = 0
    macro_legacy = []
    macro_empty = []
    per_language = {} # aixo no ho feia el seu stats crec! (bno esq no feai res era una merda jajaj)
    per_aspect = {aspect: [0, 0, 0] for aspect in ASPECTS}
    valid_json_count = 0
    empty_prediction_count = 0
    repaired_items_count = 0

    for idx, example in enumerate(examples):
        # realment inference li passa prediction ja i no caldria normalitzar :) pero bno igualment es elm ateix codi
        raw_prediction = example.get("raw_prediction", example.get("prediction", {}))
        normalized = normalize_prediction(raw_prediction, repair_policy=repair_policy)
        prediction = normalized.opinions
        gold = example.get("gold", example.get("opinions", {})) or {}

        pred_pairs = opinion_pairs(prediction) # passa de dict a set de tuples
        gold_pairs = opinion_pairs(gold)
        ok_pairs = pred_pairs & gold_pairs # interseccio de sets :) millor q iterar per les llistes q es el que feia el profe

        pred_n = len(pred_pairs)
        exp_n = len(gold_pairs)
        ok_n = len(ok_pairs)
        total_pred += pred_n
        total_exp += exp_n
        total_ok += ok_n
        valid_json_count += int(normalized.valid_json)
        empty_prediction_count += int(pred_n == 0)
        repaired_items_count += len(normalized.repaired_items or [])

        legacy = _prf(pred_n, exp_n, ok_n, empty_is_perfect=False) # legacy es la antiga de 100.0 * dsfdsfs pero que no te en compte quan el gold està buit I a més el model també prediu buit (no el conta com a bé)
        improved = _prf(pred_n, exp_n, ok_n, empty_is_perfect=True) # aquest si que te en compte el gold buit
        # retorna class PRF (precision recall ...)
        macro_legacy.append(legacy)
        macro_empty.append(improved)

        # separate into languages jeje
        lang = str(example.get("language", "unknown"))
        per_language.setdefault(lang, [0, 0, 0])
        per_language[lang][0] += pred_n
        per_language[lang][1] += exp_n
        per_language[lang][2] += ok_n

        # separate into ASPECTS
        for aspect in ASPECTS:
            aspect_pred = {(a, p) for a, p in pred_pairs if a == aspect}
            aspect_gold = {(a, p) for a, p in gold_pairs if a == aspect}
            per_aspect[aspect][0] += len(aspect_pred)
            per_aspect[aspect][1] += len(aspect_gold)
            per_aspect[aspect][2] += len(aspect_pred & aspect_gold)

        row = {
            "id": example.get("id", example.get("rid", idx)),
            "language": lang,
            "predicted": pred_n,
            "expected": exp_n,
            "correct": ok_n,
            "precision": legacy.precision,
            "recall": legacy.recall,
            "f1": legacy.f1,
            "valid_json": normalized.valid_json,
            "parse_error": normalized.parse_error or "",
            "invalid_items": ";".join(normalized.invalid_items or []),
            "repaired_items": ";".join(normalized.repaired_items or []),
        }
        rows.append(row)
        if row["parse_error"] or row["invalid_items"]:
            errors.append(row)

    count = max(len(examples), 1)
    metrics = {
        "num_examples": len(examples),
        "micro": asdict(_prf(total_pred, total_exp, total_ok)), # PRF del total -> micro
        "macro_assignment_compatible": {
            "precision": sum(x.precision for x in macro_legacy) / count, # macro: suma i divideix per count
            "recall": sum(x.recall for x in macro_legacy) / count,
            "f1": sum(x.f1 for x in macro_legacy) / count,
        },
        "macro_empty_is_perfect": {
            "precision": sum(x.precision for x in macro_empty) / count,
            "recall": sum(x.recall for x in macro_empty) / count,
            "f1": sum(x.f1 for x in macro_empty) / count,
        },
        "per_language": {lang: asdict(_prf(*counts)) for lang, counts in per_language.items()},
        "per_aspect": {aspect: asdict(_prf(*counts)) for aspect, counts in per_aspect.items()},
        "json_validity_rate": 100.0 * valid_json_count / count, # tb guardem com a metrica quants json son valids
        "empty_prediction_rate": 100.0 * empty_prediction_count / count,
        "repair_policy": repair_policy,
        "repaired_items": repaired_items_count,
        "errors": len(errors),
    }
    metrics["examples"] = rows
    return metrics, errors


def evaluate_file(path, repair_policy = "none"): # permet carregar un fitxer de prediccions
    path = Path(path)
    if path.suffix == ".jsonl": # jsonl l'hem de separar per linies
        examples = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        examples = load_json(path)
    if not isinstance(examples, list):
        raise ValueError(f"Prediction file must contain a JSON list: {path}")
    return evaluate_examples(examples, repair_policy=repair_policy)


def write_errors_csv(path, errors):
    # bno guarda en CSV enlloc de fer un print amb tots els errors
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "language",
        "predicted",
        "expected",
        "correct",
        "precision",
        "recall",
        "f1",
        "valid_json",
        "parse_error",
        "invalid_items",
        "repaired_items",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in errors:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_evaluation(metrics_path, errors_path, metrics, errors):
    write_json(metrics_path, metrics)
    write_errors_csv(errors_path, errors)


def compact_summary(metrics):
    # bno l'output igual q el profe
    macro = metrics["macro_assignment_compatible"]
    micro = metrics["micro"]
    improved = metrics["macro_empty_is_perfect"]
    return (
        "         P     R     F\n"
        f"M.avg  {macro['precision']:5.1f} {macro['recall']:5.1f} {macro['f1']:5.1f}\n"
        f"m.avg  {micro['precision']:5.1f} {micro['recall']:5.1f} {micro['f1']:5.1f}\n"
        f"M.empty {improved['precision']:5.1f} {improved['recall']:5.1f} {improved['f1']:5.1f}\n"
        f"JSON valid: {metrics['json_validity_rate']:.1f}% | empty predictions: {metrics['empty_prediction_rate']:.1f}%"
    )


"""
#------------------------------------
OLD CODE ->
def flatten(d, pref="") :
   '''
   flatten a dictionary into a list of keypath:value
   '''
   if type(d) in [str, int, float, bool]:
      return pref+":"+str(d)
   elif type(d) == list :
      return [flatten(x, pref) for x in d]
   elif type(d) == dict :
      f = []
      for k in d :
         ls = flatten(d[k], pref+f".{k}")
         if type(ls)==list : f.extend(ls)
         else: f.append(ls)
      return f
   else :
      print("ERROR - Cannot flaten '{type(d)}'")
      return None   

#------------------------------------

!!! si per algun motiu hi ha duplicats a pred el codi aquest suma ok doble!

def counts(pred, gold):
   ''' 
   Given a prediction and the expected gold result, 
   count number of predicted fields, 
   number of expected, and number of matches
   '''
   pred = flatten(pred)
   gold= flatten(gold)
   prd = len(pred)  # num. of predicted 
   exp = len(gold)  # num of expected
   ok = 0 # num of matches
   for k in pred :
      if k in gold: 
          ok += 1

   return prd, exp, ok

#------------------------------------
def stats(prd, exp, ok) :
   ''' 
   Given the count of predicted, expected and matches, 
   compute precisio, recall and F1
   '''
   P = 100.0*ok/prd if prd>0 else 0
   R = 100.0*ok/exp if exp>0 else 0
   F = 2*P*R/(P+R) if P+R>0 else 0
   return P, R, F


#------------------------------------
with open(sys.argv[1]) as jf :
   examples = json.load(jf)
   
prd_t, exp_t, ok_t = 0, 0, 0
PM, RM, FM = 0, 0, 0
for ex in examples :

   # get current example stats
   prd, exp, ok = counts(ex["prediction"], ex["gold"])

   # accumulate total counts for micro P/R/F1 later
   prd_t += prd
   exp_t += exp
   ok_t += ok

   # compute P/R/F1 for current example, and print stats
   P, R, F = stats(prd,exp,ok)
   print(f"{ex['id']:35s}\t   {P:5.1f} {R:5.1f} {F:5.1f}")
   
   # acumulate sums for for macro P/R/F1 average later
   PM += P
   RM += R
   FM += F
   

# macro P/R/F1: average of P/R/F1 over all examples
PM /= len(examples)
RM /= len(examples)
FM /= len(examples)

# micro P/R/F1: P/R/F1 over all fields of all examples
Pm, Rm, Fm = stats(prd_t, exp_t, ok_t)

# print averages
print("         P     R     F")
print(f"M.avg  {PM:5.1f} {RM:5.1f} {FM:5.1f}")
print(f"m.avg  {Pm:5.1f} {Rm:5.1f} {Fm:5.1f}")

"""
