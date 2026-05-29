# ABSA Experiment Summary

## Coverage

| task | runs |
|---|---:|
| ensemble_eval | 70 |
| few_shot | 408 |
| finetune_eval | 109 |
| finetune_train | 37 |
| zero_shot | 186 |

## Best Individual Runs

| task | model | prompt | shots | strategy | quant | seed | macro F1 | micro F1 |
|---|---|---|---:|---|---|---:|---:|---:|
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 85.72 | 87.27 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 85.45 | 86.49 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 85.20 | 86.03 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 85.20 | 86.03 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 85.11 | 86.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 85.11 | 86.00 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.97 | 85.62 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.93 | 85.75 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.93 | 85.75 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.93 | 85.75 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 84.93 | 85.75 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 84.93 | 85.75 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 84.89 | 86.35 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 84.89 | 86.35 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.31 | 85.39 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.31 | 85.39 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.12 | 86.17 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.12 | 86.17 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.10 | 86.23 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 84.10 | 86.23 |

## Best Seed-Aggregated Settings

| task | model | prompt | shots | strategy | quant | seeds | n | macro F1 mean | std |
|---|---|---|---:|---|---|---|---:|---:|---:|
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 85.20 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 85.11 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 84.93 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 84.89 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 81.92 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 81.52 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 81.51 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 81.24 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 81.24 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 81.17 | 0.00 |
| finetune_eval | llama32B3 | repeat_review | 0 | first | 4bit | 13 | 1 | 80.82 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 80.79 | 0.00 |
| ensemble_eval | ensemble | ensemble | 0 |  | none | 13 | 62 | 80.78 | 4.04 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 80.66 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 80.00 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 42 | 2 | 79.98 | 0.00 |
| finetune_eval | qwen25B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 79.69 | 0.00 |
| finetune_eval | qwen25B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 79.31 | 0.00 |
| finetune_eval | qwen25B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 78.81 | 0.00 |
| finetune_eval | qwen25B3 | my_prompt | 0 | first | 4bit | 13 | 2 | 78.28 | 0.00 |

## Generated Files

- `summary.csv`: one row per run.
- `summary_by_seed.csv`: same experimental setting aggregated over seeds.
- `by_task/`: task-specific tables for zero-shot, few-shot, fine-tuning, and fine-tuned evaluation.
- `details/per_aspect.csv` and `details/per_language.csv`: ABSA-specific error-analysis tables.
- `plots/`: task-specific plots and report-ready comparisons.
