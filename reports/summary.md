# ABSA Experiment Summary

## Coverage

| task | runs |
|---|---:|
| few_shot | 163 |
| finetune_eval | 5 |
| finetune_train | 14 |
| zero_shot | 76 |

## Best Individual Runs

| task | model | prompt | shots | strategy | quant | seed | macro F1 | micro F1 |
|---|---|---|---:|---|---|---:|---:|---:|
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 81.92 | 83.19 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 76.58 | 77.77 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 76.29 | 77.34 |
| finetune_eval | qwen25B3 | my_prompt | 0 | first | 4bit | 13 | 73.61 | 75.29 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 72.71 | 73.67 |
| few_shot | llama32B3 | cot | 10 | random | 4bit | 13 | 70.91 | 72.65 |
| few_shot | llama32B3 | cot | 10 | random | 4bit | 13 | 70.91 | 72.65 |
| few_shot | llama32B3 | cot | 5 | random | none | 13 | 70.91 | 71.80 |
| few_shot | llama32B3 | cot | 5 | random | none | 13 | 70.71 | 71.54 |
| few_shot | llama32B3 | my_prompt | 5 | random | none | 13 | 70.65 | 71.11 |
| few_shot | llama32B3 | cot | 10 | random | none | 13 | 69.61 | 71.10 |
| few_shot | qwen25B3 | cot | 10 | first | 4bit | 13 | 69.53 | 71.65 |
| few_shot | qwen25B3 | cot | 10 | first | 4bit | 13 | 69.53 | 71.65 |
| few_shot | llama32B3 | cot | 10 | random | none | 13 | 69.48 | 70.64 |
| few_shot | llama32B3 | my_prompt | 5 | random | 4bit | 13 | 68.38 | 68.15 |
| few_shot | llama32B3 | my_prompt | 10 | first | none | 13 | 68.24 | 71.75 |
| few_shot | qwen25B3 | my_prompt | 10 | first | 4bit | 13 | 68.08 | 71.50 |
| few_shot | llama32B3 | cot | 5 | random | 4bit | 13 | 68.07 | 68.78 |
| few_shot | llama32B3 | cot | 5 | random | 4bit | 13 | 68.07 | 68.78 |
| few_shot | qwen25B3 | confusion_guard | 10 | random | none | 13 | 67.94 | 71.94 |

## Best Seed-Aggregated Settings

| task | model | prompt | shots | strategy | quant | seeds | n | macro F1 mean | std |
|---|---|---|---:|---|---|---|---:|---:|---:|
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 81.92 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 76.58 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 76.29 | 0.00 |
| finetune_eval | qwen25B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 73.61 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 72.71 | 0.00 |
| few_shot | llama32B3 | cot | 10 | random | 4bit | 13 | 1 | 70.91 | 0.00 |
| few_shot | llama32B3 | cot | 10 | random | 4bit | 13 | 1 | 70.91 | 0.00 |
| few_shot | llama32B3 | cot | 5 | random | none | 13 | 2 | 70.81 | 0.15 |
| few_shot | llama32B3 | my_prompt | 5 | random | none | 13 | 1 | 70.65 | 0.00 |
| few_shot | llama32B3 | cot | 10 | random | none | 13 | 2 | 69.54 | 0.10 |
| few_shot | qwen25B3 | cot | 10 | first | 4bit | 13 | 1 | 69.53 | 0.00 |
| few_shot | qwen25B3 | cot | 10 | first | 4bit | 13 | 1 | 69.53 | 0.00 |
| few_shot | llama32B3 | my_prompt | 5 | random | 4bit | 13 | 1 | 68.38 | 0.00 |
| few_shot | llama32B3 | my_prompt | 10 | first | none | 13 | 1 | 68.24 | 0.00 |
| few_shot | qwen25B3 | my_prompt | 10 | first | 4bit | 13 | 1 | 68.08 | 0.00 |
| few_shot | llama32B3 | cot | 5 | random | 4bit | 13 | 1 | 68.07 | 0.00 |
| few_shot | llama32B3 | cot | 5 | random | 4bit | 13 | 1 | 68.07 | 0.00 |
| few_shot | qwen25B3 | confusion_guard | 10 | random | none | 13 | 2 | 67.94 | 0.00 |
| few_shot | llama32B3 | my_prompt | 10 | random | none | 13 | 1 | 67.84 | 0.00 |
| few_shot | llama32B3 | my_prompt | 10 | random | 4bit | 13 | 1 | 67.23 | 0.00 |

## Generated Files

- `summary.csv`: one row per run.
- `summary_by_seed.csv`: same experimental setting aggregated over seeds.
- `by_task/`: task-specific tables for zero-shot, few-shot, fine-tuning, and fine-tuned evaluation.
- `details/per_aspect.csv` and `details/per_language.csv`: ABSA-specific error-analysis tables.
- `plots/`: task-specific plots and report-ready comparisons.
