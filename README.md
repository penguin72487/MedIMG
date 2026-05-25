# MedIMG
MedSAM+OOD+TTA in TN3K,TG3K,TN5000,DDTI

## 單一設定檔管理參數

現在可透過專案根目錄的 `medsam_config.json` 統一管理主要參數（資料路徑、訓練、評估、TTA、compile 等）。
包含進階執行參數（例如 cache、autobatch、warm cache、precompute）也已整合在同一檔案中。

- 預設執行會自動讀取 `medsam_config.json`
- 也可用 `--config` 指定其他設定檔
- CLI 參數會覆蓋設定檔對應欄位

```shell
conda run --no-capture-output -n medsam python -u main.py
```

```shell
conda run --no-capture-output -n medsam python -u main.py --config /path/to/custom_config.json
```

```shell
conda run --no-capture-output -n medsam python -u main.py --finetune --epochs 200
```

上例中 `--finetune --epochs 200` 會覆蓋 `medsam_config.json` 內的 `skip_finetune` 與 `finetune_epochs`。

### OOD 門檻值設定

- 設定檔欄位: `ood_threshold`
- 可用 CLI 覆蓋: `--ood-threshold`
- 範圍: `0.0 ~ 1.0`（越低越容易判定為 OOD）

```shell
conda run --no-capture-output -n medsam python -u main.py --ood-threshold 0.35
```

或直接在 `medsam_config.json` 設定：

```json
{
	"ood_threshold": 0.35
}
```

### 只跑 Stage 7

若你只想跑 Stage 7/7（略過 Stage 3~6）：

```shell
conda run --no-capture-output -n medsam python -u main.py --stage7-only
```

或在 `medsam_config.json` 設定：

```json
{
	"run_only_stage7": true
}
```

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