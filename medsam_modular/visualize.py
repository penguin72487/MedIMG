from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def _fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def _delta_vs_baseline(current: float, baseline: float, higher_is_better: bool) -> float:
    return current - baseline if higher_is_better else baseline - current


def build_comparison_table(all_stats: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    rows = []
    for dataset_name, modes in all_stats.items():
        baseline_dice = modes["baseline"]["mean_dice"]
        baseline_dice_1pct_low = modes["baseline"].get("dice_1pct_low", float("nan"))
        baseline_jaccard = modes["baseline"]["mean_jaccard"]
        baseline_f1 = modes["baseline"]["mean_f1"]
        baseline_sensitivity = modes["baseline"].get("mean_sensitivity", modes["baseline"].get("mean_recall", float("nan")))
        baseline_bce = modes["baseline"].get("mean_bce", float("nan"))
        baseline_ece = modes["baseline"].get("mean_ece", float("nan"))

        ood_dice = modes["ood"]["mean_dice"]
        ood_dice_1pct_low = modes["ood"].get("dice_1pct_low", float("nan"))
        ood_jaccard = modes["ood"]["mean_jaccard"]
        ood_f1 = modes["ood"]["mean_f1"]
        ood_sensitivity = modes["ood"].get("mean_sensitivity", modes["ood"].get("mean_recall", float("nan")))
        ood_bce = modes["ood"].get("mean_bce", float("nan"))
        ood_ece = modes["ood"].get("mean_ece", float("nan"))

        tta_dice = modes["tta"]["mean_dice"]
        tta_dice_1pct_low = modes["tta"].get("dice_1pct_low", float("nan"))
        tta_jaccard = modes["tta"]["mean_jaccard"]
        tta_f1 = modes["tta"]["mean_f1"]
        tta_sensitivity = modes["tta"].get("mean_sensitivity", modes["tta"].get("mean_recall", float("nan")))
        tta_bce = modes["tta"].get("mean_bce", float("nan"))
        tta_ece = modes["tta"].get("mean_ece", float("nan"))

        rows.append(
            {
                "Dataset": dataset_name,
                "Baseline Dice": f"{baseline_dice:.4f}",
                "OOD Dice": f"{ood_dice:.4f}",
                "TTA Dice": f"{tta_dice:.4f}",
                "TTA Dice Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_dice, baseline_dice, True)),
                "Baseline Dice 1% Low": f"{baseline_dice_1pct_low:.4f}",
                "OOD Dice 1% Low": f"{ood_dice_1pct_low:.4f}",
                "TTA Dice 1% Low": f"{tta_dice_1pct_low:.4f}",
                "TTA Dice 1% Low Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_dice_1pct_low, baseline_dice_1pct_low, True)),
                "Baseline Jaccard": f"{baseline_jaccard:.4f}",
                "OOD Jaccard": f"{ood_jaccard:.4f}",
                "TTA Jaccard": f"{tta_jaccard:.4f}",
                "TTA Jaccard Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_jaccard, baseline_jaccard, True)),
                "Baseline F1": f"{baseline_f1:.4f}",
                "OOD F1": f"{ood_f1:.4f}",
                "TTA F1": f"{tta_f1:.4f}",
                "TTA F1 Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_f1, baseline_f1, True)),
                "Baseline Sensitivity": f"{baseline_sensitivity:.4f}",
                "OOD Sensitivity": f"{ood_sensitivity:.4f}",
                "TTA Sensitivity": f"{tta_sensitivity:.4f}",
                "TTA Sensitivity Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_sensitivity, baseline_sensitivity, True)),
                "Baseline BCE (lower is better)": f"{baseline_bce:.4f}",
                "OOD BCE (lower is better)": f"{ood_bce:.4f}",
                "TTA BCE (lower is better)": f"{tta_bce:.4f}",
                "TTA BCE Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_bce, baseline_bce, False)),
                "Baseline ECE (lower is better)": f"{baseline_ece:.4f}",
                "OOD ECE (lower is better)": f"{ood_ece:.4f}",
                "TTA ECE (lower is better)": f"{tta_ece:.4f}",
                "TTA ECE Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_ece, baseline_ece, False)),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric_row(ax, ds: str, all_stats: Dict[str, Dict[str, Dict]], metric_key: str, title: str, *, higher_is_better: bool, y_limits=None) -> None:
    methods = ["baseline", "ood", "tta"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    vals = [all_stats[ds][m].get(metric_key, float("nan")) for m in methods]
    ax.bar([m.upper() for m in methods], vals, color=colors)
    ax.set_title(f"{ds} {title}")
    if y_limits is not None:
        ax.set_ylim(y_limits)
    for idx, value in enumerate(vals):
        if not np.isfinite(value):
            ax.text(idx, 0.02, "N/A", ha="center")
            continue
        finite_vals = [v for v in vals if np.isfinite(v)]
        max_val = max(finite_vals) if finite_vals else 0.0
        offset = 0.02 if higher_is_better else max_val * 0.02 if max_val > 0 else 0.002
        ax.text(idx, value + offset, f"{value:.3f}", ha="center")


def save_comparison_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    metric_rows = [
        ("mean_dice", "Dice", True, (0.0, 1.0)),
        ("dice_1pct_low", "Dice 1% Low", True, (0.0, 1.0)),
        ("mean_jaccard", "Jaccard", True, (0.0, 1.0)),
        ("mean_f1", "F1", True, (0.0, 1.0)),
        ("mean_sensitivity", "Sensitivity", True, (0.0, 1.0)),
        ("mean_bce", "BCE (lower is better)", False, None),
        ("mean_ece", "ECE (lower is better)", False, None),
    ]
    fig, axes = plt.subplots(len(metric_rows), len(datasets), figsize=(6 * len(datasets), 4.5 * len(metric_rows)), squeeze=False)

    for i, ds in enumerate(datasets):
        for row_idx, (metric_key, title, higher_is_better, y_limits) in enumerate(metric_rows):
            _plot_metric_row(
                axes[row_idx, i],
                ds,
                all_stats,
                metric_key,
                title,
                higher_is_better=higher_is_better,
                y_limits=y_limits,
            )

    fig.suptitle("MedSAM Baseline vs OOD vs TTA", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Dice/Jaccard/F1/Sensitivity are higher-is-better; BCE/ECE are lower-is-better.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = output_dir / "performance_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _methods() -> List[str]:
    return ["baseline", "ood", "tta"]


def _method_label(method: str) -> str:
    return {"baseline": "Baseline", "ood": "OOD", "tta": "TTA"}.get(method, method)


def _method_color(method: str) -> str:
    return {"baseline": "#1f77b4", "ood": "#d62728", "tta": "#2ca02c"}.get(method, "#7f7f7f")


def _metric(stats: Dict[str, Any], key: str) -> float:
    value = stats.get(key, float("nan"))
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return a / b


def save_method_overview_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    metrics = [
        ("mean_dice", "Dice"),
        ("dice_1pct_low", "Dice 1% Low"),
        ("mean_f1", "F1"),
        ("mean_sensitivity", "Sensitivity"),
    ]

    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(6 * max(1, len(datasets)), 4.2 * len(metrics)), squeeze=False)
    x = np.arange(len(_methods()))

    for col, dataset in enumerate(datasets):
        for row, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row, col]
            vals = [_metric(all_stats[dataset].get(method, {}), metric_key) for method in _methods()]
            ax.bar(x, vals, color=[_method_color(m) for m in _methods()])
            ax.set_xticks(x, [_method_label(m) for m in _methods()])
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{dataset} {metric_title}")
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
                else:
                    ax.text(i, 0.02, "N/A", ha="center", fontsize=9)

    fig.suptitle("Stage 8: Method Overview (Baseline vs OOD vs TTA)", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "stage8_method_overview.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_delta_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    metrics = [
        ("mean_dice", "Delta Dice", True),
        ("dice_1pct_low", "Delta Dice 1% Low", True),
        ("mean_ece", "Delta ECE", False),
        ("mean_bce", "Delta BCE", False),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(8 + 1.5 * len(datasets), 16), squeeze=False)
    x = np.arange(len(datasets))
    width = 0.35

    for row, (metric_key, title, higher_is_better) in enumerate(metrics):
        ax = axes[row, 0]
        delta_ood = []
        delta_tta = []
        for ds in datasets:
            baseline = _metric(all_stats[ds].get("baseline", {}), metric_key)
            ood = _metric(all_stats[ds].get("ood", {}), metric_key)
            tta = _metric(all_stats[ds].get("tta", {}), metric_key)
            if higher_is_better:
                delta_ood.append(ood - baseline)
                delta_tta.append(tta - baseline)
            else:
                delta_ood.append(baseline - ood)
                delta_tta.append(baseline - tta)

        ax.bar(x - width / 2, delta_ood, width=width, label="OOD vs Baseline", color="#d62728")
        ax.bar(x + width / 2, delta_tta, width=width, label="TTA vs Baseline", color="#2ca02c")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x, datasets)
        ax.set_title(title)
        ax.legend(loc="best")

    fig.suptitle("Stage 8: Improvement Delta vs Baseline", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "stage8_delta_vs_baseline.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_cost_breakdown_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    methods = _methods()
    components: List[Tuple[str, str, str]] = [
        ("avg_data_time_ms", "data", "#8dd3c7"),
        ("avg_inference_time_ms", "inference", "#fb8072"),
        ("avg_ood_time_ms", "ood", "#80b1d3"),
        ("avg_metrics_time_ms", "metrics", "#fdb462"),
        ("avg_post_time_ms", "post", "#b3de69"),
    ]

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * max(1, len(datasets)), 6), squeeze=False)
    for col, ds in enumerate(datasets):
        ax = axes[0, col]
        x = np.arange(len(methods))
        bottoms = np.zeros((len(methods),), dtype=np.float64)
        for metric_key, label, color in components:
            vals = np.array([_metric(all_stats[ds].get(m, {}), metric_key) for m in methods], dtype=np.float64)
            vals = np.nan_to_num(vals, nan=0.0)
            ax.bar(x, vals, bottom=bottoms, color=color, label=label)
            bottoms += vals

        ax.set_xticks(x, [_method_label(m) for m in methods])
        ax.set_ylabel("ms / sample")
        ax.set_title(f"{ds} Cost Breakdown")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Stage 8: Per-sample Cost Breakdown (Stacked)", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "stage8_cost_breakdown.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_quality_throughput_frontier(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    for ds in all_stats:
        for method in _methods():
            stats = all_stats[ds].get(method, {})
            x = _metric(stats, "throughput_samples_per_sec")
            y = _metric(stats, "dice_1pct_low")
            if not np.isfinite(y):
                y = _metric(stats, "mean_dice")
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, s=90, color=_method_color(method), alpha=0.9)
            ax.text(x, y, f" {ds}-{_method_label(method)}", fontsize=8)

    ax.set_xlabel("Throughput (samples/sec)")
    ax.set_ylabel("Quality (Dice 1% Low fallback Dice)")
    ax.set_title("Stage 8: Quality-Throughput Frontier")
    ax.grid(alpha=0.3)

    out_path = output_dir / "stage8_quality_throughput_frontier.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_calibration_ece_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    methods = _methods()

    fig, ax = plt.subplots(figsize=(8 + 1.2 * len(datasets), 6))
    x = np.arange(len(datasets))
    width = 0.24
    for idx, method in enumerate(methods):
        vals = [_metric(all_stats[ds].get(method, {}), "mean_ece") for ds in datasets]
        vals = np.nan_to_num(np.asarray(vals, dtype=np.float64), nan=0.0)
        ax.bar(x + (idx - 1) * width, vals, width=width, color=_method_color(method), label=_method_label(method))

    ax.set_xticks(x, datasets)
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Stage 8: Calibration (ECE)")
    ax.legend(loc="best")

    out_path = output_dir / "stage8_calibration_ece.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ood_detection_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    methods = ["ood", "tta"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.32

    for idx, method in enumerate(methods):
        auroc_vals = [_metric(all_stats[ds].get(method, {}), "ood_auroc") for ds in datasets]
        fpr_vals = [_metric(all_stats[ds].get(method, {}), "ood_fpr95") for ds in datasets]
        auroc_vals = np.nan_to_num(np.asarray(auroc_vals, dtype=np.float64), nan=0.0)
        fpr_vals = np.nan_to_num(np.asarray(fpr_vals, dtype=np.float64), nan=0.0)
        axes[0].bar(x + (idx - 0.5) * width, auroc_vals, width=width, label=_method_label(method), color=_method_color(method))
        axes[1].bar(x + (idx - 0.5) * width, fpr_vals, width=width, label=_method_label(method), color=_method_color(method))

    axes[0].set_title("OOD AUROC (higher is better)")
    axes[0].set_ylim(0.0, 1.0)
    axes[1].set_title("OOD FPR95 (lower is better)")
    axes[1].set_ylim(0.0, 1.0)
    for ax in axes:
        ax.set_xticks(x, datasets)
        ax.legend(loc="best")

    fig.suptitle("Stage 8: OOD Detection Quality")
    out_path = output_dir / "stage8_ood_detection_quality.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_tta_cache_hit_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())

    hits = []
    misses = []
    unc_hits = []
    for ds in datasets:
        eval_cfg = all_stats[ds].get("tta", {}).get("eval_config", {})
        hits.append(float(eval_cfg.get("tta_cache_hits", 0)))
        misses.append(float(eval_cfg.get("tta_cache_misses", 0)))
        unc_hits.append(float(eval_cfg.get("tta_unc_cache_hits", 0)))

    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8 + 1.2 * len(datasets), 6))
    ax.bar(x - width, hits, width=width, label="tta_cache_hits", color="#2ca02c")
    ax.bar(x, misses, width=width, label="tta_cache_misses", color="#d62728")
    ax.bar(x + width, unc_hits, width=width, label="tta_unc_cache_hits", color="#1f77b4")
    ax.set_xticks(x, datasets)
    ax.set_ylabel("count")
    ax.set_title("Stage 8: TTA Cache Hit/Miss")
    ax.legend(loc="best")

    out_path = output_dir / "stage8_tta_cache_hits.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_run_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    except Exception:
        return []
    return []


def _summarize_run(all_stats: Dict[str, Dict[str, Dict]]) -> Dict[str, Any]:
    total_samples = 0.0
    weighted_tput = 0.0
    total_hits = 0.0
    total_misses = 0.0
    total_unc_hits = 0.0

    for ds, modes in all_stats.items():
        tta_stats = modes.get("tta", {})
        n = float(_metric(tta_stats, "num_samples"))
        tput = _metric(tta_stats, "throughput_samples_per_sec")
        if np.isfinite(n) and np.isfinite(tput) and n > 0:
            total_samples += n
            weighted_tput += n * tput

        eval_cfg = tta_stats.get("eval_config", {})
        total_hits += float(eval_cfg.get("tta_cache_hits", 0))
        total_misses += float(eval_cfg.get("tta_cache_misses", 0))
        total_unc_hits += float(eval_cfg.get("tta_unc_cache_hits", 0))

    avg_tput = _safe_div(weighted_tput, total_samples)
    hit_ratio = _safe_div(total_hits, total_hits + total_misses)
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "tta_throughput_samples_per_sec": float(avg_tput),
        "tta_cache_hits": int(total_hits),
        "tta_cache_misses": int(total_misses),
        "tta_unc_cache_hits": int(total_unc_hits),
        "tta_cache_hit_ratio": float(hit_ratio),
    }


def save_cache_throughput_trend_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "stage8_run_history.json"
    history = _load_run_history(history_path)
    history.append(_summarize_run(all_stats))
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    x = np.arange(1, len(history) + 1)
    tputs = [float(item.get("tta_throughput_samples_per_sec", 0.0)) for item in history]
    hit_ratios = [float(item.get("tta_cache_hit_ratio", 0.0)) for item in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, tputs, marker="o", color="#1f77b4", linewidth=2, label="Throughput")
    ax1.set_xlabel("Run Index")
    ax1.set_ylabel("TTA Throughput (samples/sec)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, hit_ratios, marker="s", color="#2ca02c", linewidth=2, label="Cache Hit Ratio")
    ax2.set_ylabel("TTA Cache Hit Ratio", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0.0, 1.0)

    plt.title("Stage 8: Cache Impact on Throughput Across Runs")
    out_path = output_dir / "stage8_cache_throughput_trend.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path, history_path
