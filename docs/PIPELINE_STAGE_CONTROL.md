# Pipeline Stage Control

This project supports stage-by-stage execution control from `medsam_config.json`.

## Stage Overview

- Stage 1: load model
- Stage 2: prepare test datasets
- Stage 3: detect train OOD subset
- Stage 4: OOD subset finetune
- Stage 5: full-data finetune
- Stage 6: baseline evaluation
- Stage 7: model evaluation (OOD-finetuned and/or full-finetuned)
- Stage 8: plotting and comparison outputs

## Config Keys (medsam_config.json)

- `run_stage3_detect_train_ood`
- `run_stage4_ood_finetune`
- `run_stage5_full_finetune`
- `run_stage6_baseline_eval`
- `run_stage7_eval_ood_finetuned`
- `run_stage7_eval_full_finetuned`
- `run_stage8_plotting`

Legacy/global controls still exist:

- `skip_finetune`: if true, Stage 4 and Stage 5 are forced off
- `finetune_only`: stop after finetune stages
- `run_only_stage7`: force skip Stage 3~6 and Stage 8 plotting
- `run_only_stage8`: plotting-only mode from existing summaries

## Important Dependency Notes

- Stage 4 uses OOD subset names from Stage 3 outputs.
- If Stage 3 is disabled, Stage 4 tries to load cached results from:
  - `*_train_ood_detect_results.json`
  - `train_ood_subset_summary.json`
- Stage 8 needs summary data:
  - `summary.json` (full-finetuned evaluation summary)
  - optionally `summary_ood_finetuned.json` for 4-way charts

## Common Config Patterns

### A) Run only OOD finetune (skip full finetune)

```json
{
  "skip_finetune": false,
  "run_stage3_detect_train_ood": true,
  "run_stage4_ood_finetune": true,
  "run_stage5_full_finetune": false,
  "run_stage6_baseline_eval": true,
  "run_stage7_eval_ood_finetuned": true,
  "run_stage7_eval_full_finetuned": false,
  "run_stage8_plotting": true
}
```

### B) Run only full finetune (skip OOD finetune)

```json
{
  "skip_finetune": false,
  "run_stage3_detect_train_ood": false,
  "run_stage4_ood_finetune": false,
  "run_stage5_full_finetune": true,
  "run_stage6_baseline_eval": true,
  "run_stage7_eval_ood_finetuned": false,
  "run_stage7_eval_full_finetuned": true,
  "run_stage8_plotting": true
}
```

### C) Re-evaluate and re-plot only (no finetune)

```json
{
  "skip_finetune": true,
  "run_stage6_baseline_eval": true,
  "run_stage7_eval_ood_finetuned": true,
  "run_stage7_eval_full_finetuned": true,
  "run_stage8_plotting": true
}
```

### D) Plot only from existing summaries

```json
{
  "run_only_stage8": true
}
```

## CLI Overrides

All stage keys can be overridden from CLI:

- `--run-stage4-ood-finetune` / `--skip-stage4-ood-finetune`
- `--run-stage5-full-finetune` / `--skip-stage5-full-finetune`
- and similarly for Stage 3/6/7/8

Check all options:

```bash
python main.py --help
```
