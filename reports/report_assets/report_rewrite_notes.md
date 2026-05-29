# Report Rewrite Notes

Use `report_main_insertions.tex` for the body and `report_annex_insertions.tex` for appendix material. Do not use `05_synthetic_augmentation_pipeline` in the final report.

## Global Fixes

- Replace "Aspect Based Sentiment Analyisis" with "Aspect-Based Sentiment Analysis".
- Remove all inline working comments and jokes from the final `.tex`.
- Use "fine-tuning" consistently, not mixed `finetunning`, `finetune`, and `finetuning`.
- The new best overall system is the ensemble `actual_best_mixed7-majority-repairnone`, with macro-F1 85.718. The best individual fine-tune is still 85.199.
- The best fine-tune is not only Llama synthetic. It uses `gpt5.5_1000/train_plus_synth` plus `synthetic_weak_aspects_600` generated with Llama.
- When claiming an effect, prefer the controlled tables. Use descriptive plots only to show the explored landscape.

## Introduccio / Disseny Experimental

Rewrite this as a proper experimental setup:

- models: Llama 3.2 3B and Qwen 2.5 3B;
- data: train for examples/fine-tuning, devel for all reported scores, no test-set leakage;
- output: closed-list JSON mapping aspect to polarity;
- main metric: assignment-compatible macro-F1 on `(aspect, polarity)` pairs;
- secondary metrics: micro-F1, macro-empty-F1, JSON validity, empty prediction rate;
- evaluator fix: empty-gold examples are treated correctly;
- training fix: loss is masked over prompt tokens and applied only to assistant JSON tokens;
- seeds: greedy runs are effectively deterministic, but not every expensive fine-tune has multi-seed replication.

Recommended body figures/tables here:

- `00_run_landscape_individual_runs`
- `01_best_macro_f1_by_family`

## Prompts

Keep the prompt discussion in experimental setup, not split into many repeated subsections.

Important nuance:

- `definitions` has many runs because it was used as a decoding control sweep.
- `cot` is strongest in zero-shot, but it costs more generated tokens.
- `confusion_guard` and overly restrictive prompts hurt because they make the model too conservative.
- `repeat_review` can be mentioned briefly as an extra prompt-repetition experiment, not a main result.

## Zero-Shot

Rewrite the quantization paragraph. Do not say "4bit is much worse" using raw averages only. Use the controlled row:

- `4bit vs no quant.`: 10 matched pairs, delta -7.645 macro-F1, but high standard deviation.

Main controlled claims:

- CoT vs `my_prompt`: +9.179 macro-F1.
- Definitions vs `my_prompt`: +3.529 macro-F1.
- Confusion guard vs `my_prompt`: -30.296 macro-F1.
- Qwen vs Llama: +2.447 macro-F1 on matched configs.
- Decoding sweep with `definitions`: greedy or very low randomness is best; higher temperature/top-p does not help.

Use:

- figure `02_zero_shot_prompt_descriptive`
- tables `02b_zero_shot_controlled_deltas` and `03_zero_definitions_decoding`

## Few-Shot

Fix the current conclusion saying CoT does not help. The best individual few-shot run is Llama + CoT + 30 random shots, macro-F1 76.358. The more careful claim is:

- CoT is not universally dominant in few-shot the way it is in zero-shot.
- More random shots help consistently in matched comparisons.
- Qwen is stronger on average in matched few-shot configs (+3.444), but the best individual few-shot run is Llama.
- Quantization is almost neutral in matched few-shot configs (+0.249), unlike zero-shot.
- `diverse` and `coverage_rerank` underperform random in matched comparisons, probably because they select dense examples and encourage overprediction.

Use:

- figure `04_few_shot_macro_by_shots_strategy`
- tables `04b_few_shot_controlled` and `04c_few_shot_shot_increments_random`

## Data Augmentation

Fix the language statement. Do not say all final synthetic data was only English or only Spanish. The useful story is:

- the old Llama synthetic dataset had `language: synthetic`, so it was not report-clean;
- the corrected Llama synthetic dataset has about 675 Spanish and 325 English examples;
- the weak-aspect/weak-polarity synthetic set is balanced by language, 300 English and 300 Spanish;
- the weak set is also balanced across positive, negative, neutral, and conflict opinion pairs.

Use table `05b_synthetic_dataset_audit`. Do not use plot 05.

## Fine-Tuning

Rewrite this section around three clear questions:

1. Does synthetic data help?
2. Which LoRA target modules work best?
3. Does decoding matter after fine-tuning?

Careful language:

- `06_finetune_best_by_data_regime` is descriptive because several factors change between bars.
- The strongest controlled fine-tuning claim is LoRA target modules: all-linear beats attention q/v by +3.420 to +4.092 macro-F1 across matched decoding settings.
- The best individual fine-tune uses Llama, QLoRA 4bit, bf16, all-linear LoRA, GPT synthetic plus weak-aspect Llama synthetic, temp 0.1, top-p 0.9.
- Do not overclaim BF16 vs FP16 or QLoRA vs LoRA unless the comparison is controlled. Present those as practical/resource choices and exploratory evidence.

Use:

- figure/table `06_finetune_best_by_data_regime`
- figure/table `07_finetune_lora_target_controlled`
- figure/table `07b_finetune_best_decoding`
- table `12b_best_finetune_provenance_short`

## Ensembles

Rewrite this completely. The current text says ensembles did not improve validation, but after rerunning actual-best ensembles this is false.

Correct claim:

- the best ensemble `actual_best_mixed7-majority-repairnone` reaches macro-F1 85.718 and micro-F1 87.273;
- it slightly beats the best individual fine-tune, 85.199;
- majority voting is the best useful mode;
- intersection can be competitive in one narrow case but is generally risky because it can reduce recall;
- alias repair is not central because outputs are already valid and labels are mostly exact.

Use:

- figure `08_ensemble_comparison`
- tables `08_ensemble_comparison` and `08b_best_ensemble_members`

## Error Analysis

Add this as a real section before conclusions. It strengthens the report a lot.

Main observations:

- Fine-tuning dramatically improves weak aspects: `location`, `drinks_quality`, `drinks_prices`, `drinks_style_options`.
- Some rare aspects remain unstable because devel has very small support, especially `location` with only 4 gold pairs and `drinks_prices` with only 3.
- `neutral` and `conflict` remain the hardest polarities, even after fine-tuning.
- The final ensemble improves negative, neutral, and conflict polarity F1 over the best individual fine-tune, but can lose a little on some rare aspects such as `location`.

Use:

- figure/table `09_error_analysis_per_aspect`
- figure/table `10_error_analysis_per_polarity`

## Conclusions

Update final conclusion:

- progression is zero-shot < few-shot < fine-tuning < best ensemble;
- the best individual model is Llama QLoRA all-linear with GPT + weak Llama synthetic;
- the best overall system is majority voting over actual best models;
- the hardest remaining cases are rare aspects and weak polarities;
- the main limitations are limited multi-seed fine-tuning, partially uncontrolled resource-driven comparisons, and no official test-set evaluation yet.
