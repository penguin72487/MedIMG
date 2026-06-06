# MedIMG
MedSAM+OOD+TTA in TN3K,TG3K,TN5000,DDTI

## 單一設定檔管理參數

現在可透過專案根目錄的 `medsam_config.json` 統一管理主要參數（資料路徑、訓練、評估、TTA、compile 等）。
包含進階執行參數（例如 cache、autobatch、warm cache、precompute）也已整合在同一檔案中。

Pipeline 分步控制（含 OOD fine-tune / full fine-tune 拆分）說明請見：`docs/PIPELINE_STAGE_CONTROL.md`

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

### 三道 OOD 防線設定

目前 OOD 偵測已支援三道防線，並且可獨立開關：

- 防線一（崩塌）
	- `ood_enable_collapse_detection`
	- `ood_collapse_max_prob_threshold`（預設 `0.5`）
	- CLI: `--ood-enable-collapse|--ood-disable-collapse`、`--ood-collapse-max-prob-threshold`
- 防線二（Shannon 熵）
	- `ood_enable_entropy_detection`
	- `ood_entropy_threshold`（預設 `0.5`）
	- `ood_entropy_active_prob_threshold`（預設 `0.05`，定義活躍區域）
	- CLI: `--ood-enable-entropy|--ood-disable-entropy`、`--ood-entropy-threshold`、`--ood-entropy-active-prob-threshold`
- 防線三（形態碎裂 / 連通元件）
	- `ood_enable_fragmentation_detection`
	- `ood_fragment_prob_threshold`（預設 `0.5`）
	- `ood_fragment_min_area`（預設 `80`）
	- `ood_fragment_max_large_components`（預設 `3`）
	- CLI: `--ood-enable-fragmentation|--ood-disable-fragmentation`、`--ood-fragment-prob-threshold`、`--ood-fragment-min-area`、`--ood-fragment-max-large-components`

註：最終判定採聯集規則，只要任一道防線命中即視為 OOD；同時仍保留 `ood_method + ood_threshold` 分數閾值判定作為相容 fallback。

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

若你只想跑 Stage 7/8（略過 Stage 3~6 與 Stage 8 繪圖）：

```shell
conda run --no-capture-output -n medsam python -u main.py --stage7-only
```

或在 `medsam_config.json` 設定：

```json
{
	"run_only_stage7": true
}
```

### 只跑 Stage 8（只繪圖）

若你只想用既有 `summary.json` 產生 Stage 8 圖表：

```shell
conda run --no-capture-output -n medsam python -u main.py --stage8-only
```

或在 `medsam_config.json` 設定：

```json
{
	"run_only_stage8": true
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

python3 main.py --run-clinical-mode