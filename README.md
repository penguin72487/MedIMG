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
conda run --no-capture-output -n medsam python -u main.py --tta-fusion entropy_weighted --compile-dynamic --compile-warmup-batches 1,8 --finetune
```

硬體自動分工（建議預設）:

- `--cpu-threads 0`: 自動配置 CPU tensor 執行緒（GPU 模式會保留核心給 DataLoader/I/O）
- `--workers 0`: 微調 DataLoader worker 自動配置
- `--eval-workers 0`: 評估 DataLoader worker 自動配置
- 不指定 `--compile-dynamic` 時，會依裝置自動選擇（CUDA 偏向固定形狀，CPU 偏向 dynamic）

```shell
conda run --no-capture-output -n medsam python -u main.py --finetune --cpu-threads 0 --workers 0 --eval-workers 0 --tta-fusion entropy_weighted
```