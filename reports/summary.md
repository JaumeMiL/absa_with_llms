# ABSA Experiment Summary

| task | model | prompt | shots | temp | top_p | seed | macro F1 | micro F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| few_shot | llama32B3 | configs/prompts/zero_shot/my_prompt.yaml | 8 | 0.0 | 1.0 | 13 | 66.91 | 70.47 |
| few_shot | llama32B3 | configs/prompts/zero_shot/my_prompt.yaml | 8 | 0.0 | 1.0 | 13 | 66.59 | 69.46 |
| few_shot | llama32B3 | configs/prompts/zero_shot/baseline.yaml | 8 | 0.0 | 1.0 | 13 | 56.32 | 64.71 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/my_prompt.yaml | 0 | 0.0 | 1.0 | 13 | 76.29 | 77.34 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/cot.yaml | 0 | 0.0 | 1.0 | 13 | 63.44 | 61.14 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/cot.yaml | 0 | 0.0 | 1.0 | 13 | 63.44 | 61.14 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/cot.yaml | 0 | 0.0 | 1.0 | 13 | 61.65 | 61.69 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/cot.yaml | 0 | 0.0 | 1.0 | 13 | 61.65 | 61.69 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/cot.yaml | 0 | 0.0 | 1.0 | 13 | 58.33 | 60.07 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 1.0 | 13 | 55.04 | 57.46 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 1.0 | 13 | 55.04 | 57.46 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 0.5 | 101 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 0.5 | 13 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 0.5 | 42 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 0.9 | 101 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 0.9 | 13 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 0.9 | 42 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 1.0 | 101 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 1.0 | 13 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 1.0 | 42 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 1.0 | 13 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.0 | 1.0 | 13 | 54.44 | 55.50 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.2 | 1.0 | 13 | 52.77 | 53.90 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/my_prompt.yaml | 0 | 0.0 | 1.0 | 13 | 52.56 | 55.05 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/my_prompt.yaml | 0 | 0.0 | 1.0 | 13 | 52.56 | 55.05 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/my_prompt.yaml | 0 | 0.0 | 1.0 | 13 | 52.50 | 49.60 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/negative.yaml | 0 | 0.0 | 1.0 | 13 | 50.82 | 49.90 |
| zero_shot | qwen25B3 | configs/prompts/zero_shot/negative.yaml | 0 | 0.0 | 1.0 | 13 | 50.82 | 49.90 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/definitions.yaml | 0 | 0.2 | 1.0 | 42 | 50.80 | 52.78 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/my_prompt.yaml | 0 | 0.0 | 1.0 | 13 | 49.89 | 47.99 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/my_prompt.yaml | 0 | 0.0 | 1.0 | 13 | 49.89 | 47.99 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/negative.yaml | 0 | 0.0 | 1.0 | 13 | 38.53 | 39.51 |
| zero_shot | llama32B3 | configs/prompts/zero_shot/negative.yaml | 0 | 0.0 | 1.0 | 13 | 38.53 | 39.51 |
