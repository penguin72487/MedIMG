"""
MedSAM 模組化流程入口
用法:
    python main.py [選項]

完整流程: 資料讀取 → 模型微調（可跳過）→ 測試評估 → 結果輸出
"""

import argparse
import os
import sys
from pathlib import Path

from medsam_modular.config import (
    DEFAULT_OUTPUT_DIR_REL,
    default_config_path,
    load_settings,
    settings_to_env,
)


CLI_SETTING_KEYS = {
    "data_root",
    "tn3k_path",
    "tg3k_path",
    "ddti_path",
    "tn5000_path",
    "split_root",
    "model_id",
    "weight_path",
    "image_size",
    "require_compile",
    "compile_dynamic",
    "compile_warmup_batches",
    "skip_finetune",
    "finetune_train_backbone",
    "finetune_epochs",
    "finetune_batch",
    "finetune_lr",
    "finetune_weight_decay",
    "finetune_adamw_beta1",
    "finetune_adamw_beta2",
    "finetune_adamw_eps",
    "finetune_val_ratio",
    "finetune_patience",
    "finetune_min_delta",
    "finetune_grad_accum",
    "finetune_grad_clip",
    "finetune_workers",
    "finetune_max_samples",
    "finetune_only",
    "finetune_use_fused_adamw",
    "run_only_stage7",
    "ood_threshold",
    "eval_workers",
    "eval_batch",
    "cpu_threads",
    "ood_method",
    "tta_fusion",
    "tta_augmentations",
    "tta_chunk_size",
    "tta_fixed_batch",
    "output_dir",
}

# 強制 stdout/stderr 無緩衝，確保終端機即時顯示
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def _resolve_config_path(project_root: Path) -> Path:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", metavar="FILE")
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config:
        return Path(pre_args.config)
    return default_config_path(project_root)


def _ood_threshold_type(value: str) -> float:
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise argparse.ArgumentTypeError("OOD threshold must be within [0.0, 1.0]")
    return threshold


def _parse_args(defaults: dict, config_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSAM 模組化流程：讀取資料、訓練（微調）、測試評估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        default=str(config_path),
        metavar="FILE",
        help="JSON 設定檔路徑（參數統一來源）",
    )

    # ── 資料路徑 ──────────────────────────────────────────────────────────────
    data_group = parser.add_argument_group("資料路徑")
    data_group.add_argument(
        "--data-root",
        default=defaults["data_root"],
        dest="data_root",
        metavar="DIR",
        help="所有資料集的根目錄（TN3K / TG3K / DDTI / TN5000 放在其下）",
    )
    data_group.add_argument("--tn3k-path", default=defaults["tn3k_path"], dest="tn3k_path", metavar="DIR", help="TN3K 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--tg3k-path", default=defaults["tg3k_path"], dest="tg3k_path", metavar="DIR", help="TG3K 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--ddti-path", default=defaults["ddti_path"], dest="ddti_path", metavar="DIR", help="DDTI 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--tn5000-path", default=defaults["tn5000_path"], dest="tn5000_path", metavar="DIR", help="TN5000 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument(
        "--split-root",
        default=defaults["split_root"],
        dest="split_root",
        metavar="DIR",
        help="train/val/test split 文字檔根目錄（預設: <專案根>/splits）",
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model_group = parser.add_argument_group("模型")
    model_group.add_argument(
        "--model-id",
        default=defaults["model_id"],
        dest="model_id",
        help="HuggingFace model ID 或本地路徑",
    )
    model_group.add_argument(
        "--weight-path",
        default=defaults["weight_path"],
        dest="weight_path",
        metavar="FILE",
        help="預訓練或微調後的權重 .pth 檔（空白表示從 HuggingFace 下載）",
    )
    model_group.add_argument(
        "--image-size",
        type=int,
        default=defaults["image_size"],
        dest="image_size",
        metavar="N",
        help="輸入影像解析度（正方形邊長）",
    )
    model_group.add_argument(
        "--require-compile",
        action="store_true",
        dest="require_compile",
        help="若 torch.compile(inductor) 無法啟用則中止",
    )
    model_group.add_argument(
        "--compile-dynamic",
        dest="compile_dynamic",
        action="store_true",
        help="啟用 torch.compile dynamic shape（預設: 自動依裝置決定）",
    )
    model_group.add_argument(
        "--no-compile-dynamic",
        dest="compile_dynamic",
        action="store_false",
        help="停用 torch.compile dynamic shape（覆蓋自動設定）",
    )
    model_group.add_argument(
        "--compile-warmup-batches",
        default=defaults["compile_warmup_batches"],
        dest="compile_warmup_batches",
        metavar="BATCHES",
        help="compile warmup 批次（逗號分隔），如: 1,8",
    )

    # ── 訓練（微調） ──────────────────────────────────────────────────────────
    train_group = parser.add_argument_group("訓練（微調）")
    train_group.add_argument(
        "--skip-finetune",
        action="store_true",
        dest="skip_finetune",
        help="跳過微調，直接使用現有權重評估（預設開啟）",
    )
    train_group.add_argument(
        "--finetune",
        dest="skip_finetune",
        action="store_false",
        help="執行微調（關閉 --skip-finetune）",
    )
    train_group.add_argument("--train-backbone",  action="store_true", dest="finetune_train_backbone", help="微調時同時訓練 image encoder backbone")
    train_group.add_argument("--epochs",          type=int,   default=defaults["finetune_epochs"], dest="finetune_epochs", metavar="N",   help="微調總 epoch 數")
    train_group.add_argument("--batch-size",      type=int,   default=defaults["finetune_batch"], dest="finetune_batch", metavar="N",   help="訓練批次大小")
    train_group.add_argument("--lr",              type=float, default=defaults["finetune_lr"], dest="finetune_lr", metavar="LR",  help="初始學習率")
    train_group.add_argument("--weight-decay",    type=float, default=defaults["finetune_weight_decay"], dest="finetune_weight_decay", metavar="WD",  help="AdamW weight decay")
    train_group.add_argument("--adamw-beta1",     type=float, default=defaults["finetune_adamw_beta1"], dest="finetune_adamw_beta1", metavar="B1",  help="AdamW beta1")
    train_group.add_argument("--adamw-beta2",     type=float, default=defaults["finetune_adamw_beta2"], dest="finetune_adamw_beta2", metavar="B2", help="AdamW beta2")
    train_group.add_argument("--adamw-eps",       type=float, default=defaults["finetune_adamw_eps"], dest="finetune_adamw_eps", metavar="EPS", help="AdamW epsilon")
    train_group.add_argument("--val-ratio",       type=float, default=defaults["finetune_val_ratio"], dest="finetune_val_ratio", metavar="R",   help="若無 val split 則切割訓練集的比例")
    train_group.add_argument("--patience",        type=int,   default=defaults["finetune_patience"], dest="finetune_patience", metavar="N",   help="Early stopping patience")
    train_group.add_argument("--min-delta",       type=float, default=defaults["finetune_min_delta"], dest="finetune_min_delta", metavar="D",   help="Early stopping 最小改善量")
    train_group.add_argument("--grad-accum",      type=int,   default=defaults["finetune_grad_accum"], dest="finetune_grad_accum", metavar="N",   help="梯度累積步數")
    train_group.add_argument("--grad-clip",       type=float, default=defaults["finetune_grad_clip"], dest="finetune_grad_clip", metavar="V",   help="梯度裁剪最大範數")
    train_group.add_argument("--workers",         type=int,   default=defaults["finetune_workers"], dest="finetune_workers", metavar="N",   help="DataLoader worker 數量（0=自動）")
    train_group.add_argument("--max-samples",     type=int,   default=defaults["finetune_max_samples"], dest="finetune_max_samples", metavar="N",   help="每個資料集最多取樣數（0 = 不限）")
    train_group.add_argument(
        "--finetune-only",
        action="store_true",
        dest="finetune_only",
        help="只執行微調並輸出權重，不進行後續測試評估",
    )
    train_group.add_argument(
        "--stage7-only",
        action="store_true",
        dest="run_only_stage7",
        help="只執行 Stage 7/7 測試（略過 Stage 3~6）",
    )
    fused_group = train_group.add_mutually_exclusive_group()
    fused_group.add_argument("--use-fused-adamw", action="store_true", dest="finetune_use_fused_adamw", help="啟用 fused AdamW")
    fused_group.add_argument("--no-fused-adamw", action="store_false", dest="finetune_use_fused_adamw", help="停用 fused AdamW（在某些環境下需要）")

    # ── 評估 ──────────────────────────────────────────────────────────────────
    eval_group = parser.add_argument_group("評估")
    eval_group.add_argument(
        "--ood-threshold",
        type=_ood_threshold_type,
        default=defaults["ood_threshold"],
        dest="ood_threshold",
        metavar="T",
        help="OOD 偵測閾值（0~1；越低越容易判定為 OOD）",
    )
    eval_group.add_argument(
        "--eval-workers",
        type=int,
        default=defaults["eval_workers"],
        dest="eval_workers",
        metavar="N",
        help="評估 DataLoader worker 數量（CPU 平行，0=自動）",
    )
    eval_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=defaults["eval_batch"],
        dest="eval_batch",
        metavar="N",
        help="評估批次大小（0 表示依模式自動）",
    )
    eval_group.add_argument(
        "--cpu-threads",
        type=int,
        default=defaults["cpu_threads"],
        dest="cpu_threads",
        metavar="N",
        help="CPU 運算執行緒數（torch.set_num_threads，0=自動）",
    )
    eval_group.add_argument(
        "--ood-method",
        default=defaults["ood_method"],
        dest="ood_method",
        choices=["entropy", "confidence", "variance"],
        help="OOD 偵測方法",
    )
    
    # TTA parameters
    eval_group.add_argument(
        "--tta-fusion",
        default=defaults["tta_fusion"],
        dest="tta_fusion",
        choices=["mean", "median", "entropy_weighted"],
        help="TTA 融合策略（mean: 平均, median: 中位數, entropy_weighted: 熵加權）",
    )
    eval_group.add_argument(
        "--tta-augmentations",
        default=defaults["tta_augmentations"],
        dest="tta_augmentations",
        metavar="AUGS",
        help="自訂 TTA 增強方式（逗號分隔），如: none,hflip,vflip,rotate_90,rotate_270",
    )
    eval_group.add_argument(
        "--tta-chunk-size",
        type=int,
        default=defaults["tta_chunk_size"],
        dest="tta_chunk_size",
        metavar="N",
        help="TTA 分塊推論大小（預設: 8；12GB 顯卡較穩定。設為 0 可啟用自動調參）",
    )
    eval_group.add_argument(
        "--tta-fixed-batch",
        type=int,
        default=defaults["tta_fixed_batch"],
        dest="tta_fixed_batch",
        metavar="N",
        help="固定 TTA batch 大小（0 表示不額外 padding，較省 VRAM）",
    )

    # ── 輸出 ──────────────────────────────────────────────────────────────────
    out_group = parser.add_argument_group("輸出")
    out_group.add_argument(
        "--output-dir",
        default=defaults["output_dir"],
        dest="output_dir",
        metavar="DIR",
        help="結果輸出目錄（預設: <專案根>/results/modular）",
    )

    parser.set_defaults(
        require_compile=bool(defaults["require_compile"]),
        compile_dynamic=defaults["compile_dynamic"],
        skip_finetune=bool(defaults["skip_finetune"]),
        finetune_train_backbone=bool(defaults["finetune_train_backbone"]),
        finetune_only=bool(defaults["finetune_only"]),
        finetune_use_fused_adamw=bool(defaults["finetune_use_fused_adamw"]),
        run_only_stage7=bool(defaults["run_only_stage7"]),
    )

    return parser.parse_args()


def _apply_env(settings: dict) -> None:
    """將設定檔 + CLI 合併參數轉換成 runner.py 讀取的環境變數。"""

    for key, value in settings_to_env(settings).items():
        if value != "":
            os.environ[key] = value


def main() -> None:
    project_root = Path(__file__).resolve().parent

    # 確保專案根目錄在 sys.path 中，方便從任意目錄執行
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config_path = _resolve_config_path(project_root)
    loaded_defaults, resolved_config_path = load_settings(project_root=project_root, config_path=config_path)
    args = _parse_args(loaded_defaults, resolved_config_path)

    effective_settings = dict(loaded_defaults)
    arg_values = vars(args)
    for key in CLI_SETTING_KEYS:
        if key in arg_values:
            effective_settings[key] = arg_values[key]

    _apply_env(effective_settings)

    output_dir = Path(effective_settings["output_dir"]) if effective_settings["output_dir"] else project_root / DEFAULT_OUTPUT_DIR_REL

    from medsam_modular.runner import main as runner_main  # noqa: PLC0415

    print("=" * 80)
    print("  MedSAM Pipeline")
    print(f"  設定檔: {resolved_config_path}")
    print(f"  模式: {'跳過微調（純評估）' if effective_settings['skip_finetune'] else '微調 + 評估'}")
    print(f"  影像尺寸: {effective_settings['image_size']}")
    print(f"  輸出目錄: {output_dir}")
    print("  即時輸出提示: conda run 請使用 --no-capture-output")
    print("=" * 80)

    runner_main()


if __name__ == "__main__":
    main()
