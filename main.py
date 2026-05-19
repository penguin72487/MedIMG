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
    data_group.add_argument("--tn3k-path",  default="", metavar="DIR", help="TN3K 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--tg3k-path",  default="", metavar="DIR", help="TG3K 資料集路徑（覆蓋 --data-root）")
    data_group.add_argument("--ddti-path",  default="", metavar="DIR", help="DDTI 資料集路徑（覆蓋 --data-root）")
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
        default="facebook/sam-vit-base",
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
        default=512,
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
        action="store_true",
        help="啟用 torch.compile dynamic shape（預設: CUDA 下關閉以提升固定形狀推論效能）",
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
    train_group.add_argument("--epochs",          type=int,   default=100,  metavar="N",   help="微調總 epoch 數")
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
    train_group.add_argument("--workers",         type=int,   default=4,    metavar="N",   help="DataLoader worker 數量")
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
        default=0,
        metavar="N",
        help="TTA 分塊推論大小（0 表示自動；12GB GPU 建議 2-4）",
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


def _apply_env(args: argparse.Namespace) -> None:
    """將 CLI 參數轉換成 runner.py 讀取的環境變數。"""

    def _set(key: str, val: str) -> None:
        if val:
            os.environ[key] = val

    _set("MEDSAM_DATA_ROOT",           args.data_root)
    _set("MEDSAM_TN3K_PATH",           args.tn3k_path)
    _set("MEDSAM_TG3K_PATH",           args.tg3k_path)
    _set("MEDSAM_DDTI_PATH",           args.ddti_path)
    _set("MEDSAM_TN5000_PATH",         args.tn5000_path)
    _set("MEDSAM_SPLIT_ROOT",          args.split_root)
    _set("MEDSAM_MODEL_ID",            args.model_id)
    _set("MEDSAM_WEIGHT_PATH",         args.weight_path)
    _set("MEDSAM_IMAGE_SIZE",          str(args.image_size))
    _set("MEDSAM_OUTPUT_DIR",          args.output_dir)
    _set("MEDSAM_OOD_THRESHOLD",       str(args.ood_threshold))
    _set("MEDSAM_OOD_METHOD",          args.ood_method)
    _set("MEDSAM_TTA_FUSION",          args.tta_fusion)
    _set("MEDSAM_TTA_AUGMENTATIONS",   args.tta_augmentations)
    _set("MEDSAM_PROFILE_PATH",        args.profile_output)
    _set("MEDSAM_TTA_CHUNK_SIZE",      str(args.tta_chunk_size) if args.tta_chunk_size > 0 else "")
    _set("MEDSAM_TTA_FIXED_BATCH",     str(args.tta_fixed_batch) if args.tta_fixed_batch > 0 else "")
    _set("MEDSAM_COMPILE_WARMUP_BATCHES", args.compile_warmup_batches)

    os.environ["MEDSAM_SKIP_FINETUNE"]              = "1" if args.skip_finetune else "0"
    os.environ["MEDSAM_FINETUNE_TRAIN_BACKBONE"]    = "1" if args.train_backbone else "0"
    os.environ["MEDSAM_FINETUNE_EPOCHS"]            = str(args.epochs)
    os.environ["MEDSAM_FINETUNE_BATCH"]             = str(args.batch_size)
    os.environ["MEDSAM_FINETUNE_LR"]                = str(args.lr)
    os.environ["MEDSAM_FINETUNE_WEIGHT_DECAY"]      = str(args.weight_decay)
    os.environ["MEDSAM_FINETUNE_ADAMW_BETA1"]       = str(args.adamw_beta1)
    os.environ["MEDSAM_FINETUNE_ADAMW_BETA2"]       = str(args.adamw_beta2)
    os.environ["MEDSAM_FINETUNE_ADAMW_EPS"]         = str(args.adamw_eps)
    os.environ["MEDSAM_FINETUNE_VAL_RATIO"]         = str(args.val_ratio)
    os.environ["MEDSAM_FINETUNE_PATIENCE"]          = str(args.patience)
    os.environ["MEDSAM_FINETUNE_MIN_DELTA"]         = str(args.min_delta)
    os.environ["MEDSAM_FINETUNE_GRAD_ACCUM"]        = str(args.grad_accum)
    os.environ["MEDSAM_FINETUNE_GRAD_CLIP"]         = str(args.grad_clip)
    os.environ["MEDSAM_FINETUNE_WORKERS"]           = str(args.workers)
    os.environ["MEDSAM_FINETUNE_MAX_SAMPLES"]       = str(args.max_samples)
    os.environ["MEDSAM_FINETUNE_ONLY"]              = "1" if args.finetune_only else "0"
    os.environ["MEDSAM_FINETUNE_USE_FUSED_ADAMW"]   = "0" if args.no_fused_adamw else "1"
    os.environ["MEDSAM_REQUIRE_COMPILE"]            = "1" if args.require_compile else "0"
    os.environ["MEDSAM_COMPILE_DYNAMIC"]            = "1" if args.compile_dynamic else "0"
    os.environ["MEDSAM_TTA_FAST"]                   = "1" if args.tta_fast else "0"
    os.environ["MEDSAM_PROFILE"]                    = "1" if args.profile else "0"


def main() -> None:
    args = _parse_args()
    _apply_env(args)

    # 確保專案根目錄在 sys.path 中，方便從任意目錄執行
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "results" / "modular"

    from medsam_modular.runner import main as runner_main  # noqa: PLC0415

    print("=" * 80)
    print("  MedSAM Pipeline")
    print(f"  模式: {'跳過微調（純評估）' if args.skip_finetune else '微調 + 評估'}")
    print(f"  影像尺寸: {args.image_size}")
    print(f"  輸出目錄: {args.output_dir or str(project_root / 'results' / 'modular')}")
    print("  即時輸出提示: conda run 請使用 --no-capture-output")
    print("=" * 80)

    runner_main()


if __name__ == "__main__":
    main()
