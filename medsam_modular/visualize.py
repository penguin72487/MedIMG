from pathlib import Path
from typing import Dict

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
        baseline_jaccard = modes["baseline"]["mean_jaccard"]
        baseline_f1 = modes["baseline"]["mean_f1"]
        baseline_bce = modes["baseline"].get("mean_bce", float("nan"))

        ood_dice = modes["ood"]["mean_dice"]
        ood_jaccard = modes["ood"]["mean_jaccard"]
        ood_f1 = modes["ood"]["mean_f1"]
        ood_bce = modes["ood"].get("mean_bce", float("nan"))

        tta_dice = modes["tta"]["mean_dice"]
        tta_jaccard = modes["tta"]["mean_jaccard"]
        tta_f1 = modes["tta"]["mean_f1"]
        tta_bce = modes["tta"].get("mean_bce", float("nan"))

        rows.append(
            {
                "Dataset": dataset_name,
                "Baseline Dice": f"{baseline_dice:.4f}",
                "OOD Dice": f"{ood_dice:.4f}",
                "TTA Dice": f"{tta_dice:.4f}",
                "TTA Dice Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_dice, baseline_dice, True)),
                "Baseline Jaccard": f"{baseline_jaccard:.4f}",
                "OOD Jaccard": f"{ood_jaccard:.4f}",
                "TTA Jaccard": f"{tta_jaccard:.4f}",
                "TTA Jaccard Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_jaccard, baseline_jaccard, True)),
                "Baseline F1": f"{baseline_f1:.4f}",
                "OOD F1": f"{ood_f1:.4f}",
                "TTA F1": f"{tta_f1:.4f}",
                "TTA F1 Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_f1, baseline_f1, True)),
                "Baseline BCE (lower is better)": f"{baseline_bce:.4f}",
                "OOD BCE (lower is better)": f"{ood_bce:.4f}",
                "TTA BCE (lower is better)": f"{tta_bce:.4f}",
                "TTA BCE Delta vs Baseline": _fmt_delta(_delta_vs_baseline(tta_bce, baseline_bce, False)),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric_row(ax, ds: str, all_stats: Dict[str, Dict[str, Dict]], metric_key: str, title: str, *, higher_is_better: bool, y_limits=None) -> None:
    methods = ["baseline", "ood", "tta"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    vals = [all_stats[ds][m][metric_key] for m in methods]
    ax.bar([m.upper() for m in methods], vals, color=colors)
    ax.set_title(f"{ds} {title}")
    if y_limits is not None:
        ax.set_ylim(y_limits)
    for idx, value in enumerate(vals):
        offset = 0.02 if higher_is_better else max(vals) * 0.02 if max(vals) > 0 else 0.002
        ax.text(idx, value + offset, f"{value:.3f}", ha="center")


def save_comparison_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    # Four rows: Dice, Jaccard, F1, BCE
    fig, axes = plt.subplots(4, len(datasets), figsize=(6 * len(datasets), 18), squeeze=False)

    for i, ds in enumerate(datasets):
        _plot_metric_row(axes[0, i], ds, all_stats, "mean_dice", "Dice", higher_is_better=True, y_limits=(0.0, 1.0))
        _plot_metric_row(axes[1, i], ds, all_stats, "mean_jaccard", "Jaccard", higher_is_better=True, y_limits=(0.0, 1.0))
        _plot_metric_row(axes[2, i], ds, all_stats, "mean_f1", "F1", higher_is_better=True, y_limits=(0.0, 1.0))
        _plot_metric_row(axes[3, i], ds, all_stats, "mean_bce", "BCE (lower is better)", higher_is_better=False)

    fig.suptitle("MedSAM Baseline vs OOD vs TTA", fontsize=14)
    fig.text(0.5, 0.01, "Dice / Jaccard / F1 are higher-is-better metrics; BCE is a loss metric and lower is better.", ha="center", fontsize=10)
    plt.tight_layout()

    out_path = output_dir / "performance_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
