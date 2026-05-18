# MedIMG
MedSAM+OOD+TTA in TN3K,TG3K,TN5000,DDTI

## Bottleneck Profiling

Pipeline now supports detailed bottleneck analysis for model loading, fine-tuning, and evaluation (baseline/OOD/TTA).

- Profiling is enabled by default (no extra flag needed)
- Explicitly enable profiling: `python main.py --profile`
- Disable profiling: `python main.py --no-profile`
- Custom profile output path: `python main.py --profile-output results/modular/bottleneck_profile.json`

Generated report includes:

- section-level timing breakdown
- function/substep timing across data loading, preprocessing, device transfer, inference, TTA, metrics, and output saving
- top bottlenecks ranking
- stage-aware optimization recommendations
- optimization-limit analysis (whether current pipeline is near the practical limit or still has clear headroom)

The profiling report is refreshed during execution after each major stage / dataset-mode block completes.

```shell
conda run --no-capture-output -n medsam python -u main.py --tta-fusion entropy_weighted --compile-dynamic --compile-warmup-batches 1,8
```