from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd


def _fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def _delta_vs_baseline(current: float, baseline: float, higher_is_better: bool) -> float:
    return current - baseline if higher_is_better else baseline - current


def _has_any_metric(all_stats: Dict[str, Dict[str, Dict]], metric_key: str) -> bool:
    for modes in all_stats.values():
        for stats in modes.values():
            try:
                value = float(stats.get(metric_key, float("nan")))
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                return True
    return False


def build_comparison_table(all_stats: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    rows = []
    include_ap50 = _has_any_metric(all_stats, "ap50")
    include_map50_95 = _has_any_metric(all_stats, "map50_95")
    for dataset_name, modes in all_stats.items():
        baseline_dice = modes["baseline"]["mean_dice"]
        baseline_dice_5pct_low = modes["baseline"].get("dice_5pct_low", float("nan"))
        baseline_jaccard = modes["baseline"]["mean_jaccard"]
        baseline_f1 = modes["baseline"]["mean_f1"]
        baseline_sensitivity = modes["baseline"].get("mean_sensitivity", modes["baseline"].get("mean_recall", float("nan")))
        baseline_bce = modes["baseline"].get("mean_bce", float("nan"))
        baseline_ece = modes["baseline"].get("mean_ece", float("nan"))

        ood_dice = modes["ood"]["mean_dice"]
        ood_dice_5pct_low = modes["ood"].get("dice_5pct_low", float("nan"))
        ood_jaccard = modes["ood"]["mean_jaccard"]
        ood_f1 = modes["ood"]["mean_f1"]
        ood_sensitivity = modes["ood"].get("mean_sensitivity", modes["ood"].get("mean_recall", float("nan")))
        ood_bce = modes["ood"].get("mean_bce", float("nan"))
        ood_ece = modes["ood"].get("mean_ece", float("nan"))

        tta_dice = modes["tta"]["mean_dice"]
        tta_dice_5pct_low = modes["tta"].get("dice_5pct_low", float("nan"))
        tta_jaccard = modes["tta"]["mean_jaccard"]
        tta_f1 = modes["tta"]["mean_f1"]
        tta_sensitivity = modes["tta"].get("mean_sensitivity", modes["tta"].get("mean_recall", float("nan")))
        tta_bce = modes["tta"].get("mean_bce", float("nan"))
        tta_ece = modes["tta"].get("mean_ece", float("nan"))

        row = {
            "Dataset": dataset_name,
            "Baseline Dice": f"{baseline_dice:.4f}",
            "OOD Dice": f"{ood_dice:.4f}",
            "TTA Dice": f"{tta_dice:.4f}",
            "TTA Dice Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_dice, baseline_dice, True)),
            "Baseline Dice 5% Low": f"{baseline_dice_5pct_low:.4f}",
            "OOD Dice 5% Low": f"{ood_dice_5pct_low:.4f}",
            "TTA Dice 5% Low": f"{tta_dice_5pct_low:.4f}",
            "TTA Dice 5% Low Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_dice_5pct_low, baseline_dice_5pct_low, True)),
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

        if include_ap50:
            baseline_ap50 = modes["baseline"].get("ap50", float("nan"))
            ood_ap50 = modes["ood"].get("ap50", float("nan"))
            tta_ap50 = modes["tta"].get("ap50", float("nan"))
            row.update(
                {
                    "Baseline AP50": f"{baseline_ap50:.4f}",
                    "OOD AP50": f"{ood_ap50:.4f}",
                    "TTA AP50": f"{tta_ap50:.4f}",
                    "TTA AP50 Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_ap50, baseline_ap50, True)),
                }
            )

        if include_map50_95:
            baseline_map = modes["baseline"].get("map50_95", float("nan"))
            ood_map = modes["ood"].get("map50_95", float("nan"))
            tta_map = modes["tta"].get("map50_95", float("nan"))
            row.update(
                {
                    "Baseline mAP50-95": f"{baseline_map:.4f}",
                    "OOD mAP50-95": f"{ood_map:.4f}",
                    "TTA mAP50-95": f"{tta_map:.4f}",
                    "TTA mAP50-95 Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_map, baseline_map, True)),
                }
            )

        rows.append(row)
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
        ("dice_5pct_low", "Dice 5% Low", True, (0.0, 1.0)),
        ("mean_jaccard", "Jaccard", True, (0.0, 1.0)),
        ("mean_f1", "F1", True, (0.0, 1.0)),
        ("mean_sensitivity", "Sensitivity", True, (0.0, 1.0)),
        ("mean_bce", "BCE (lower is better)", False, None),
        ("mean_ece", "ECE (lower is better)", False, None),
    ]
    if _has_any_metric(all_stats, "ap50"):
        metric_rows.append(("ap50", "AP50", True, (0.0, 1.0)))
    if _has_any_metric(all_stats, "map50_95"):
        metric_rows.append(("map50_95", "mAP50-95", True, (0.0, 1.0)))
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
        "Dice/Jaccard/F1/Sensitivity/AP/mAP are higher-is-better; BCE/ECE are lower-is-better.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = output_dir / "performance_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _methods(all_stats: Optional[Dict[str, Dict[str, Dict]]] = None) -> List[str]:
    if not all_stats:
        return ["baseline", "ood", "tta"]
    method_keys: set = set()
    for modes in all_stats.values():
        if isinstance(modes, dict):
            method_keys.update(str(k) for k in modes.keys())

    four_way = ["baseline", "ood_finetune", "full_finetune", "ood_finetune_tta"]
    if all(m in method_keys for m in four_way):
        return four_way
    return ["baseline", "ood", "tta"]


def _method_label(method: str) -> str:
    return {
        "baseline": "Baseline",
        "ood": "OOD",
        "tta": "TTA",
        "ood_finetune": "OOD-Finetune",
        "full_finetune": "Full-Finetune",
        "ood_finetune_tta": "OOD-Finetune+TTA",
    }.get(method, method)


def _method_color(method: str) -> str:
    return {
        "baseline": "#1f77b4",
        "ood": "#d62728",
        "tta": "#2ca02c",
        "ood_finetune": "#ff7f0e",
        "full_finetune": "#2ca02c",
        "ood_finetune_tta": "#d62728",
    }.get(method, "#7f7f7f")


def merge_stage8_stats(
    full_summary: Dict[str, Dict[str, Dict[str, Any]]],
    ood_finetuned_summary: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not ood_finetuned_summary:
        return full_summary
    return _build_variant_stats(full_summary=full_summary, ood_finetuned_summary=ood_finetuned_summary)


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


def _variant_specs() -> List[Tuple[str, str, str]]:
    return [
        ("baseline", "baseline", "#1f77b4"),
        ("ood_finetune", "ood", "#ff7f0e"),
        ("full_finetune", "full", "#2ca02c"),
        ("ood_finetune_tta", "ood+TTAInference", "#d62728"),
    ]


def _build_variant_stats(
    full_summary: Dict[str, Dict[str, Dict[str, Any]]],
    ood_finetuned_summary: Dict[str, Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    datasets = sorted(set(full_summary.keys()) | set(ood_finetuned_summary.keys()))
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset_name in datasets:
        full_modes = full_summary.get(dataset_name, {})
        ood_modes = ood_finetuned_summary.get(dataset_name, {})
        merged[dataset_name] = {
            "baseline": dict(full_modes.get("baseline", ood_modes.get("baseline", {}))),
            "ood_finetune": dict(ood_modes.get("ood", {})),
            "full_finetune": dict(full_modes.get("ood", {})),
            "ood_finetune_tta": dict(ood_modes.get("tta", {})),
        }
    return merged


def _auto_panel_limits(values: List[float], *, higher_is_better: bool, default_limits: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    finite_vals = [float(v) for v in values if np.isfinite(v)]
    if not finite_vals:
        return default_limits

    vmin = min(finite_vals)
    vmax = max(finite_vals)
    if default_limits is None:
        span = vmax - vmin
        pad = max(span * 0.18, 1e-4)
        lo = vmin - pad
        hi = vmax + pad
        if lo == hi:
            hi = lo + 1e-3
        return (lo, hi)

    lo_bound, hi_bound = default_limits
    span = vmax - vmin
    pad = max(span * 0.18, 5e-4)
    if span < 0.02:
        lo = max(lo_bound, vmin - pad)
        hi = min(hi_bound, vmax + pad)
        if hi - lo < 0.005:
            center = 0.5 * (hi + lo)
            lo = max(lo_bound, center - 0.0025)
            hi = min(hi_bound, center + 0.0025)
        if hi <= lo:
            return default_limits
        return (lo, hi)

    return default_limits


def save_four_way_variant_chart(
    full_summary: Dict[str, Dict[str, Dict[str, Any]]],
    ood_finetuned_summary: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = _build_variant_stats(full_summary=full_summary, ood_finetuned_summary=ood_finetuned_summary)
    datasets = list(all_stats.keys())
    metric_rows = [
        ("mean_dice", "Dice", True, (0.0, 1.0)),
        ("dice_5pct_low", "Dice 5% Low", True, (0.0, 1.0)),
        ("mean_jaccard", "Jaccard", True, (0.0, 1.0)),
        ("mean_f1", "F1", True, (0.0, 1.0)),
        ("mean_sensitivity", "Sensitivity", True, (0.0, 1.0)),
        ("mean_bce", "BCE (lower is better)", False, None),
        ("mean_ece", "ECE (lower is better)", False, None),
    ]
    if _has_any_metric(all_stats, "ap50"):
        metric_rows.append(("ap50", "AP50", True, (0.0, 1.0)))
    if _has_any_metric(all_stats, "map50_95"):
        metric_rows.append(("map50_95", "mAP50-95", True, (0.0, 1.0)))

    fig, axes = plt.subplots(len(metric_rows), len(datasets), figsize=(7.5 * max(1, len(datasets)), 4.6 * len(metric_rows)), squeeze=False)
    variant_specs = _variant_specs()

    for col, dataset_name in enumerate(datasets):
        for row, (metric_key, title, higher_is_better, y_limits) in enumerate(metric_rows):
            ax = axes[row, col]
            labels = [label for _, label, _ in variant_specs]
            colors = [color for _, _, color in variant_specs]
            vals = [all_stats[dataset_name].get(variant_key, {}).get(metric_key, float("nan")) for variant_key, _, _ in variant_specs]
            ax.bar(labels, vals, color=colors)
            ax.set_title(f"{dataset_name} {title}")
            ax.tick_params(axis="x", rotation=18)
            panel_limits = _auto_panel_limits(vals, higher_is_better=higher_is_better, default_limits=y_limits)
            if panel_limits is not None:
                ax.set_ylim(panel_limits)
            for idx, value in enumerate(vals):
                if not np.isfinite(value):
                    y0, y1 = ax.get_ylim()
                    ax.text(idx, y0 + (y1 - y0) * 0.08, "N/A", ha="center", fontsize=9)
                    continue
                finite_vals = [v for v in vals if np.isfinite(v)]
                max_val = max(finite_vals) if finite_vals else 0.0
                y0, y1 = ax.get_ylim()
                offset = max((y1 - y0) * 0.03, 5e-4)
                ax.text(idx, value + offset, f"{value:.3f}", ha="center", fontsize=9)

    fig.suptitle("MedSAM Variant Comparison", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Variants: baseline / ood / full / ood+TTAInference. Each panel auto-zooms when differences are very small. Higher is better except BCE/ECE.",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout()

    out_path = output_dir / "performance_comparison_4way.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_method_overview_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    metrics = [
        ("mean_dice", "Dice"),
        ("dice_5pct_low", "Dice 5% Low"),
        ("mean_f1", "F1"),
        ("mean_sensitivity", "Sensitivity"),
    ]

    methods = _methods(all_stats)
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(6 * max(1, len(datasets)), 4.2 * len(metrics)), squeeze=False)
    x = np.arange(len(methods))

    for col, dataset in enumerate(datasets):
        for row, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row, col]
            vals = [_metric(all_stats[dataset].get(method, {}), metric_key) for method in methods]
            ax.bar(x, vals, color=[_method_color(m) for m in methods])
            ax.set_xticks(x, [_method_label(m) for m in methods])
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{dataset} {metric_title}")
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
                else:
                    ax.text(i, 0.02, "N/A", ha="center", fontsize=9)

    fig.suptitle("Stage 8: Method Overview", fontsize=14)
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
        ("dice_5pct_low", "Delta Dice 5% Low", True),
        ("mean_ece", "Delta ECE", False),
        ("mean_bce", "Delta BCE", False),
    ]

    methods = _methods(all_stats)
    compare_methods = [m for m in methods if m != "baseline"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8 + 1.5 * len(datasets), 16), squeeze=False)
    x = np.arange(len(datasets))
    width = 0.75 / max(1, len(compare_methods))

    for row, (metric_key, title, higher_is_better) in enumerate(metrics):
        ax = axes[row, 0]
        for idx_m, method in enumerate(compare_methods):
            deltas = []
            for ds in datasets:
                baseline = _metric(all_stats[ds].get("baseline", {}), metric_key)
                cur = _metric(all_stats[ds].get(method, {}), metric_key)
                if higher_is_better:
                    deltas.append(cur - baseline)
                else:
                    deltas.append(baseline - cur)
            offset = (idx_m - (len(compare_methods) - 1) / 2.0) * width
            ax.bar(x + offset, deltas, width=width, label=f"{_method_label(method)} vs Baseline", color=_method_color(method))
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
    methods = _methods(all_stats)
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

    methods = _methods(all_stats)
    for ds in all_stats:
        for method in methods:
            stats = all_stats[ds].get(method, {})
            x = _metric(stats, "throughput_samples_per_sec")
            y = _metric(stats, "dice_5pct_low")
            if not np.isfinite(y):
                y = _metric(stats, "mean_dice")
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, s=90, color=_method_color(method), alpha=0.9)
            ax.text(x, y, f" {ds}-{_method_label(method)}", fontsize=8)

    ax.set_xlabel("Throughput (samples/sec)")
    ax.set_ylabel("Quality (Dice 5% Low fallback Dice)")
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
    methods = _methods(all_stats)

    fig, ax = plt.subplots(figsize=(8 + 1.2 * len(datasets), 6))
    x = np.arange(len(datasets))
    width = 0.75 / max(1, len(methods))
    for idx, method in enumerate(methods):
        vals = [_metric(all_stats[ds].get(method, {}), "mean_ece") for ds in datasets]
        vals = np.nan_to_num(np.asarray(vals, dtype=np.float64), nan=0.0)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, color=_method_color(method), label=_method_label(method))

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
    methods = [m for m in _methods(all_stats) if m != "baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.75 / max(1, len(methods))

    for idx, method in enumerate(methods):
        auroc_vals = [_metric(all_stats[ds].get(method, {}), "ood_auroc") for ds in datasets]
        fpr_vals = [_metric(all_stats[ds].get(method, {}), "ood_fpr95") for ds in datasets]
        auroc_vals = np.nan_to_num(np.asarray(auroc_vals, dtype=np.float64), nan=0.0)
        fpr_vals = np.nan_to_num(np.asarray(fpr_vals, dtype=np.float64), nan=0.0)
        offset = (idx - (len(methods) - 1) / 2.0) * width
        axes[0].bar(x + offset, auroc_vals, width=width, label=_method_label(method), color=_method_color(method))
        axes[1].bar(x + offset, fpr_vals, width=width, label=_method_label(method), color=_method_color(method))

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

    methods = _methods(all_stats)
    fig, axes = plt.subplots(1, 3, figsize=(12 + 1.5 * len(datasets), 6), squeeze=False)
    metric_specs = [
        ("tta_cache_hits", "TTA cache hits"),
        ("tta_cache_misses", "TTA cache misses"),
        ("tta_unc_cache_hits", "TTA uncertainty cache hits"),
    ]

    x = np.arange(len(datasets))
    width = 0.75 / max(1, len(methods))
    for ax_idx, (metric_key, title) in enumerate(metric_specs):
        ax = axes[0, ax_idx]
        for idx_m, method in enumerate(methods):
            vals = []
            for ds in datasets:
                eval_cfg = all_stats[ds].get(method, {}).get("eval_config", {})
                vals.append(float(eval_cfg.get(metric_key, 0.0)))
            offset = (idx_m - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=_method_label(method), color=_method_color(method))

        ax.set_xticks(x, datasets)
        ax.set_title(title)
        if ax_idx == 0:
            ax.set_ylabel("count")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Stage 8: TTA Cache Metrics by Variant")

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

    methods = _methods(all_stats)
    tta_method = "ood_finetune_tta" if "ood_finetune_tta" in methods else ("tta" if "tta" in methods else methods[-1])

    for ds, modes in all_stats.items():
        tta_stats = modes.get(tta_method, {})
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


def _to_bool_mask(mask_like: Any) -> np.ndarray:
    arr = np.asarray(mask_like)
    if arr.dtype == np.bool_:
        return arr
    return arr > 0.5


def save_top_bottom_case_comparison_chart(
    dataset_name: str,
    case_entries: List[Dict[str, Any]],
    output_dir: Path,
    *,
    file_tag: str,
) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not case_entries:
        return None

    n_rows = len(case_entries)
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, max(4.0, 3.8 * n_rows)), squeeze=False)
    headers = ["Original Image", "Original Label", "Input Prompt", "Model Output"]

    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=11)

    for row, entry in enumerate(case_entries):
        image_np = np.asarray(entry.get("image"), dtype=np.uint8)
        if image_np.ndim == 2:
            image_np = np.stack([image_np, image_np, image_np], axis=-1)

        gt_mask = _to_bool_mask(entry.get("gt_mask", np.zeros(image_np.shape[:2], dtype=np.uint8)))
        pred_mask = _to_bool_mask(entry.get("pred_mask", np.zeros(image_np.shape[:2], dtype=np.uint8)))
        bbox = entry.get("bbox", None)

        row_title = (
            f"{entry.get('rank_label', '')} | {entry.get('name', '')} | "
            f"Dice={float(entry.get('dice', float('nan'))):.3f} | "
            f"OOD={bool(entry.get('is_ood', False))}"
        )

        # Col 1: original image
        ax0 = axes[row, 0]
        ax0.imshow(image_np)
        ax0.axis("off")
        ax0.set_ylabel(row_title, fontsize=9)

        # Col 2: original label overlay
        ax1 = axes[row, 1]
        ax1.imshow(image_np)
        overlay_gt = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 4), dtype=np.float32)
        overlay_gt[..., 1] = 1.0
        overlay_gt[..., 3] = gt_mask.astype(np.float32) * 0.45
        ax1.imshow(overlay_gt)
        ax1.axis("off")

        # Col 3: input prompt (bbox)
        ax2 = axes[row, 2]
        ax2.imshow(image_np)
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            rect = plt.Rectangle((x1, y1), max(1.0, x2 - x1), max(1.0, y2 - y1), fill=False, linewidth=2.0, edgecolor="#ff0000")
            ax2.add_patch(rect)
        ax2.axis("off")

        # Col 4: output result overlay
        ax3 = axes[row, 3]
        ax3.imshow(image_np)
        overlay_pred = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 4), dtype=np.float32)
        overlay_pred[..., 0] = 1.0
        overlay_pred[..., 1] = 0.6
        overlay_pred[..., 3] = pred_mask.astype(np.float32) * 0.45
        ax3.imshow(overlay_pred)
        ood_score = float(entry.get("ood_score", float("nan")))
        ax3.text(0.02, 0.98, f"OOD score={ood_score:.3f}", transform=ax3.transAxes, va="top", ha="left", fontsize=8, color="white", bbox={"facecolor": "black", "alpha": 0.45, "pad": 2})
        ax3.axis("off")

    fig.suptitle(f"{dataset_name} Top3 vs Bottom3 Cases ({file_tag})", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / f"{dataset_name.lower()}_{file_tag}_top3_bottom3_cases.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_ood_train_test_count_chart(ood_counts: Dict[str, Dict[str, float]], output_dir: Path) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not ood_counts:
        return None

    datasets = list(ood_counts.keys())
    train_ood = np.array([float(ood_counts[d].get("train_ood", 0.0)) for d in datasets], dtype=np.float64)
    test_ood = np.array([float(ood_counts[d].get("test_ood", 0.0)) for d in datasets], dtype=np.float64)
    train_ratio = np.array([float(ood_counts[d].get("train_ood_ratio", 0.0)) for d in datasets], dtype=np.float64)
    test_ratio = np.array([float(ood_counts[d].get("test_ood_ratio", 0.0)) for d in datasets], dtype=np.float64)

    x = np.arange(len(datasets))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(7.0 + 1.5 * len(datasets), 6.2), squeeze=False)

    ax0 = axes[0, 0]
    ax0.bar(x - width / 2, train_ood, width=width, label="Train OOD samples", color="#1f77b4")
    ax0.bar(x + width / 2, test_ood, width=width, label="Test OOD samples", color="#d62728")
    ax0.set_xticks(x, datasets)
    ax0.set_ylabel("OOD sample count")
    ax0.set_title("OOD Sample Count")
    ax0.legend(loc="best")

    ax1 = axes[0, 1]
    ax1.bar(x - width / 2, train_ratio, width=width, label="Train OOD ratio", color="#1f77b4")
    ax1.bar(x + width / 2, test_ratio, width=width, label="Test OOD ratio", color="#d62728")
    ax1.set_xticks(x, datasets)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("OOD ratio")
    ax1.set_title("OOD Ratio")
    ax1.legend(loc="best")

    fig.suptitle("Train/Test OOD Distribution by Dataset", fontsize=14)
    plt.tight_layout()
    out_path = output_dir / "ood_train_test_counts.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
