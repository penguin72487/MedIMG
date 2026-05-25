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


DEFAULT_MODEL_ID = "facebook/sam-vit-base"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_OUTPUT_DIR = Path("results") / "modular"

# 強制 stdout/stderr 無緩衝，確保終端機即時顯示
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSAM 模組化流程：讀取資料、訓練（微調）、測試評估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 資料路徑 ──────────────────────────────────────────────────────────────
    data_group = parser.add_argument_group("資料路徑")
    data_group.add_argument(
        "--data-root",
        default="",
        metavar="DIR",
        help="所有資料集的根目錄（TN3K / TG3K / DDTI / TN5000 放在其下）",
    )
    data_group.add_argument("--tn3k-path", default="", metavar="DIR", help="TN3K 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--tg3k-path", default="", metavar="DIR", help="TG3K 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--ddti-path", default="", metavar="DIR", help="DDTI 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--tn5000-path", default="", metavar="DIR", help="TN5000 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument(
        "--split-root",
        default="",
        metavar="DIR",
        help="train/val/test split 文字檔根目錄（預設: <專案根>/splits）",
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model_group = parser.add_argument_group("模型")
    model_group.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID 或本地路徑",
    )
    model_group.add_argument(
        "--weight-path",
        default="",
        metavar="FILE",
        help="預訓練或微調後的權重 .pth 檔（空白表示從 HuggingFace 下載）",
    )
    model_group.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        metavar="N",
        help="輸入影像解析度（正方形邊長）",
    )
    model_group.add_argument(
        "--require-compile",
        action="store_true",
        help="若 torch.compile(inductor) 無法啟用則中止",
    )
    model_group.add_argument(
        "--compile-dynamic",
        dest="compile_dynamic",
        action="store_true",
        default=None,
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
        default="",
        metavar="BATCHES",
        help="compile warmup 批次（逗號分隔），如: 1,8",
    )

    # ── 訓練（微調） ──────────────────────────────────────────────────────────
    train_group = parser.add_argument_group("訓練（微調）")
    train_group.add_argument(
        "--skip-finetune",
        action="store_true",
        default=True,
        help="跳過微調，直接使用現有權重評估（預設開啟）",
    )
    train_group.add_argument(
        "--finetune",
        dest="skip_finetune",
        action="store_false",
        help="執行微調（關閉 --skip-finetune）",
    )
    train_group.add_argument("--train-backbone",  action="store_true", help="微調時同時訓練 image encoder backbone")
    train_group.add_argument("--epochs",          type=int,   default=1000, metavar="N",   help="微調總 epoch 數")
    train_group.add_argument("--batch-size",      type=int,   default=8,    metavar="N",   help="訓練批次大小")
    train_group.add_argument("--lr",              type=float, default=1e-4, metavar="LR",  help="初始學習率")
    train_group.add_argument("--weight-decay",    type=float, default=1e-3, metavar="WD",  help="AdamW weight decay")
    train_group.add_argument("--adamw-beta1",     type=float, default=0.9,  metavar="B1",  help="AdamW beta1")
    train_group.add_argument("--adamw-beta2",     type=float, default=0.999, metavar="B2", help="AdamW beta2")
    train_group.add_argument("--adamw-eps",       type=float, default=1e-8, metavar="EPS", help="AdamW epsilon")
    train_group.add_argument("--val-ratio",       type=float, default=0.1,  metavar="R",   help="若無 val split 則切割訓練集的比例")
    train_group.add_argument("--patience",        type=int,   default=20,   metavar="N",   help="Early stopping patience")
    train_group.add_argument("--min-delta",       type=float, default=1e-4, metavar="D",   help="Early stopping 最小改善量")
    train_group.add_argument("--grad-accum",      type=int,   default=2,    metavar="N",   help="梯度累積步數")
    train_group.add_argument("--grad-clip",       type=float, default=1.0,  metavar="V",   help="梯度裁剪最大範數")
    train_group.add_argument("--workers",         type=int,   default=0,    metavar="N",   help="DataLoader worker 數量（0=自動）")
    train_group.add_argument("--max-samples",     type=int,   default=0,    metavar="N",   help="每個資料集最多取樣數（0 = 不限）")
    train_group.add_argument(
        "--finetune-only",
        action="store_true",
        help="只執行微調並輸出權重，不進行後續測試評估",
    )
    train_group.add_argument(
        "--no-fused-adamw",
        action="store_true",
        help="停用 fused AdamW（在某些環境下需要）",
    )

    # ── 評估 ──────────────────────────────────────────────────────────────────
    eval_group = parser.add_argument_group("評估")
    eval_group.add_argument(
        "--ood-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="OOD 偵測閾值",
    )
    eval_group.add_argument(
        "--eval-workers",
        type=int,
        default=0,
        metavar="N",
        help="評估 DataLoader worker 數量（CPU 平行，0=自動）",
    )
    eval_group.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        metavar="N",
        help="評估批次大小（0 表示依模式自動）",
    )
    eval_group.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        metavar="N",
        help="CPU 運算執行緒數（torch.set_num_threads，0=自動）",
    )
    eval_group.add_argument(
        "--ood-method",
        default="entropy",
        choices=["entropy", "confidence", "variance"],
        help="OOD 偵測方法",
    )
    
    # TTA parameters
    eval_group.add_argument(
        "--tta-fusion",
        default="entropy_weighted",
        choices=["mean", "median", "entropy_weighted"],
        help="TTA 融合策略（mean: 平均, median: 中位數, entropy_weighted: 熵加權）",
    )
    eval_group.add_argument(
        "--tta-fast",
        action="store_true",
        help="使用快速 TTA 模式（僅翻轉增強，加快速度）",
    )
    eval_group.add_argument(
        "--tta-augmentations",
        default="",
        metavar="AUGS",
        help="自訂 TTA 增強方式（逗號分隔），如: none,hflip,vflip,rotate_90,rotate_270",
    )
    eval_group.add_argument(
        "--tta-chunk-size",
        type=int,
        default=8,
        metavar="N",
        help="TTA 分塊推論大小（預設: 8；12GB 顯卡較穩定。設為 0 可啟用自動調參）",
    )
    eval_group.add_argument(
        "--tta-fixed-batch",
        type=int,
        default=0,
        metavar="N",
        help="固定 TTA batch 大小（0 表示不額外 padding，較省 VRAM）",
    )

    # ── 輸出 ──────────────────────────────────────────────────────────────────
    out_group = parser.add_argument_group("輸出")
    out_group.add_argument(
        "--output-dir",
        default="",
        metavar="DIR",
        help="結果輸出目錄（預設: <專案根>/results/modular）",
    )
    out_group.add_argument(
        "--profile",
        dest="profile",
        action="store_true",
        default=True,
        help="啟用流程瓶頸分析與 profiling 報告輸出",
    )
    out_group.add_argument(
        "--no-profile",
        dest="profile",
        action="store_false",
        help="停用 profiling 報告輸出",
    )
    out_group.add_argument(
        "--profile-output",
        default="",
        metavar="FILE",
        help="profiling JSON 輸出路徑（預設: <output-dir>/bottleneck_profile.json）",
    )

    return parser.parse_args()


def _set_env(key: str, value: str) -> None:
    if value:
        os.environ[key] = value


def _set_env_int(key: str, value: int, *, skip_zero: bool = True) -> None:
    if not skip_zero or value > 0:
        os.environ[key] = str(value)


def _set_env_flag(key: str, enabled: bool) -> None:
    os.environ[key] = "1" if enabled else "0"


def _apply_env(args: argparse.Namespace) -> None:
    """將 CLI 參數轉換成 runner.py 讀取的環境變數。"""

    for key, value in {
        "MEDSAM_DATA_ROOT": args.data_root,
        "MEDSAM_TN3K_PATH": args.tn3k_path,
        "MEDSAM_TG3K_PATH": args.tg3k_path,
        "MEDSAM_DDTI_PATH": args.ddti_path,
        "MEDSAM_TN5000_PATH": args.tn5000_path,
        "MEDSAM_SPLIT_ROOT": args.split_root,
        "MEDSAM_MODEL_ID": args.model_id,
        "MEDSAM_WEIGHT_PATH": args.weight_path,
        "MEDSAM_OUTPUT_DIR": args.output_dir,
        "MEDSAM_OOD_METHOD": args.ood_method,
        "MEDSAM_TTA_FUSION": args.tta_fusion,
        "MEDSAM_TTA_AUGMENTATIONS": args.tta_augmentations,
        "MEDSAM_PROFILE_PATH": args.profile_output,
        "MEDSAM_COMPILE_WARMUP_BATCHES": args.compile_warmup_batches,
    }.items():
        _set_env(key, value)

    for key, value in {
        "MEDSAM_IMAGE_SIZE": args.image_size,
        "MEDSAM_OOD_THRESHOLD": args.ood_threshold,
        "MEDSAM_FINETUNE_EPOCHS": args.epochs,
        "MEDSAM_FINETUNE_BATCH": args.batch_size,
        "MEDSAM_FINETUNE_LR": args.lr,
        "MEDSAM_FINETUNE_WEIGHT_DECAY": args.weight_decay,
        "MEDSAM_FINETUNE_ADAMW_BETA1": args.adamw_beta1,
        "MEDSAM_FINETUNE_ADAMW_BETA2": args.adamw_beta2,
        "MEDSAM_FINETUNE_ADAMW_EPS": args.adamw_eps,
        "MEDSAM_FINETUNE_VAL_RATIO": args.val_ratio,
        "MEDSAM_FINETUNE_PATIENCE": args.patience,
        "MEDSAM_FINETUNE_MIN_DELTA": args.min_delta,
        "MEDSAM_FINETUNE_GRAD_ACCUM": args.grad_accum,
        "MEDSAM_FINETUNE_GRAD_CLIP": args.grad_clip,
        "MEDSAM_FINETUNE_WORKERS": args.workers,
        "MEDSAM_FINETUNE_MAX_SAMPLES": args.max_samples,
        "MEDSAM_EVAL_WORKERS": args.eval_workers,
        "MEDSAM_CPU_THREADS": args.cpu_threads,
    }.items():
        _set_env_int(key, value, skip_zero=False)

    for key, enabled in {
        "MEDSAM_SKIP_FINETUNE": args.skip_finetune,
        "MEDSAM_FINETUNE_TRAIN_BACKBONE": args.train_backbone,
        "MEDSAM_FINETUNE_ONLY": args.finetune_only,
        "MEDSAM_FINETUNE_USE_FUSED_ADAMW": not args.no_fused_adamw,
        "MEDSAM_REQUIRE_COMPILE": args.require_compile,
        "MEDSAM_TTA_FAST": args.tta_fast,
        "MEDSAM_PROFILE": args.profile,
    }.items():
        _set_env_flag(key, enabled)

    if args.compile_dynamic is True:
        os.environ["MEDSAM_COMPILE_DYNAMIC"] = "1"
    elif args.compile_dynamic is False:
        os.environ["MEDSAM_COMPILE_DYNAMIC"] = "0"

    _set_env_int("MEDSAM_TTA_CHUNK_SIZE", args.tta_chunk_size)
    _set_env_int("MEDSAM_TTA_FIXED_BATCH", args.tta_fixed_batch)
    _set_env_int("MEDSAM_EVAL_BATCH", args.eval_batch_size)


def main() -> None:
    args = _parse_args()
    _apply_env(args)

    # 確保專案根目錄在 sys.path 中，方便從任意目錄執行
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    output_dir = Path(args.output_dir) if args.output_dir else project_root / DEFAULT_OUTPUT_DIR

    from medsam_modular.runner import main as runner_main  # noqa: PLC0415

    print("=" * 80)
    print("  MedSAM Pipeline")
    print(f"  模式: {'跳過微調（純評估）' if args.skip_finetune else '微調 + 評估'}")
    print(f"  影像尺寸: {args.image_size}")
    print(f"  輸出目錄: {output_dir}")
    print("  即時輸出提示: conda run 請使用 --no-capture-output")
    print("=" * 80)

    runner_main()


if __name__ == "__main__":
    main()
