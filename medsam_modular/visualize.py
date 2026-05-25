from pathlib import Path
from typing import Dict

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
