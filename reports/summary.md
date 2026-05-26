# ABSA Experiment Summary

## Coverage

| task | runs |
|---|---:|
| few_shot | 51 |
| finetune_eval | 2 |
| finetune_train | 2 |
| zero_shot | 64 |

## Best Individual Runs

| task | model | prompt | shots | strategy | quant | seed | macro F1 | micro F1 |
|---|---|---|---:|---|---|---:|---:|---:|
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 76.29 | 77.34 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 72.71 | 73.67 |
| few_shot | llama32B3 | cot | 5 | random | none | 13 | 70.91 | 71.80 |
| few_shot | llama32B3 | my_prompt | 5 | random | none | 13 | 70.65 | 71.11 |
| few_shot | llama32B3 | cot | 10 | random | none | 13 | 69.48 | 70.64 |
| few_shot | llama32B3 | my_prompt | 10 | first | none | 13 | 68.24 | 71.75 |
| few_shot | llama32B3 | my_prompt | 10 | random | none | 13 | 67.84 | 70.25 |
| few_shot | llama32B3 | cot | 10 | first | none | 13 | 67.27 | 70.91 |
| few_shot | qwen25B3 | cot | 10 | random | none | 13 | 66.98 | 72.01 |
| few_shot | llama32B3 | my_prompt | 8 | first | none | 13 | 66.91 | 70.47 |
| few_shot | qwen25B3 | cot | 2 | random | none | 13 | 66.72 | 68.39 |
| few_shot | llama32B3 | my_prompt | 8 | first | none | 13 | 66.59 | 69.46 |
| few_shot | qwen25B3 | cot | 10 | first | none | 13 | 66.58 | 69.13 |
| few_shot | qwen25B3 | my_prompt | 10 | random | none | 13 | 66.28 | 71.72 |
| few_shot | qwen25B3 | cot | 5 | random | none | 13 | 65.35 | 66.59 |
| few_shot | qwen25B3 | my_prompt | 2 | random | none | 13 | 65.27 | 68.51 |
| few_shot | llama32B3 | my_prompt | 2 | first | none | 13 | 64.87 | 65.38 |
| few_shot | llama32B3 | cot | 5 | first | none | 13 | 64.40 | 66.30 |
| few_shot | llama32B3 | my_prompt | 1 | first | none | 13 | 64.09 | 63.29 |
| few_shot | qwen25B3 | my_prompt | 2 | first | none | 13 | 63.99 | 66.58 |

## Best Seed-Aggregated Settings

| task | model | prompt | shots | strategy | quant | seeds | n | macro F1 mean | std |
|---|---|---|---:|---|---|---|---:|---:|---:|
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 76.29 | 0.00 |
| finetune_eval | llama32B3 | my_prompt | 0 | first | 4bit | 13 | 1 | 72.71 | 0.00 |
| few_shot | llama32B3 | cot | 5 | random | none | 13 | 1 | 70.91 | 0.00 |
| few_shot | llama32B3 | my_prompt | 5 | random | none | 13 | 1 | 70.65 | 0.00 |
| few_shot | llama32B3 | cot | 10 | random | none | 13 | 1 | 69.48 | 0.00 |
| few_shot | llama32B3 | my_prompt | 10 | first | none | 13 | 1 | 68.24 | 0.00 |
| few_shot | llama32B3 | my_prompt | 10 | random | none | 13 | 1 | 67.84 | 0.00 |
| few_shot | llama32B3 | cot | 10 | first | none | 13 | 1 | 67.27 | 0.00 |
| few_shot | qwen25B3 | cot | 10 | random | none | 13 | 1 | 66.98 | 0.00 |
| few_shot | llama32B3 | my_prompt | 8 | first | none | 13 | 2 | 66.75 | 0.22 |
| few_shot | qwen25B3 | cot | 2 | random | none | 13 | 1 | 66.72 | 0.00 |
| few_shot | qwen25B3 | cot | 10 | first | none | 13 | 1 | 66.58 | 0.00 |
| few_shot | qwen25B3 | my_prompt | 10 | random | none | 13 | 1 | 66.28 | 0.00 |
| few_shot | qwen25B3 | cot | 5 | random | none | 13 | 1 | 65.35 | 0.00 |
| few_shot | qwen25B3 | my_prompt | 2 | random | none | 13 | 1 | 65.27 | 0.00 |
| few_shot | llama32B3 | my_prompt | 2 | first | none | 13 | 1 | 64.87 | 0.00 |
| few_shot | llama32B3 | cot | 5 | first | none | 13 | 1 | 64.40 | 0.00 |
| few_shot | llama32B3 | my_prompt | 1 | first | none | 13 | 1 | 64.09 | 0.00 |
| few_shot | qwen25B3 | my_prompt | 2 | first | none | 13 | 1 | 63.99 | 0.00 |
| zero_shot | llama32B3 | cot | 0 | first | none | 13 | 1 | 63.44 | 0.00 |

## Generated Files

- `summary.csv`: one row per run.
- `summary_by_seed.csv`: same experimental setting aggregated over seeds.
- `by_task/`: task-specific tables for zero-shot, few-shot, fine-tuning, and fine-tuned evaluation.
- `details/per_aspect.csv` and `details/per_language.csv`: ABSA-specific error-analysis tables.
- `plots/`: task-specific plots and report-ready comparisons.
