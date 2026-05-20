from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def build_comparison_table(all_stats: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    rows = []
    for dataset_name, modes in all_stats.items():
        rows.append(
            {
                "Dataset": dataset_name,
                "Baseline Dice": f"{modes['baseline']['mean_dice']:.4f}",
                "OOD Dice": f"{modes['ood']['mean_dice']:.4f}",
                "TTA Dice": f"{modes['tta']['mean_dice']:.4f}",
                "Baseline Jaccard": f"{modes['baseline']['mean_jaccard']:.4f}",
                "OOD Jaccard": f"{modes['ood']['mean_jaccard']:.4f}",
                "TTA Jaccard": f"{modes['tta']['mean_jaccard']:.4f}",
                "Baseline F1": f"{modes['baseline']['mean_f1']:.4f}",
                "OOD F1": f"{modes['ood']['mean_f1']:.4f}",
                "TTA F1": f"{modes['tta']['mean_f1']:.4f}",
                "Baseline BCE": f"{modes['baseline'].get('mean_bce', float('nan')):.4f}",
                "OOD BCE": f"{modes['ood'].get('mean_bce', float('nan')):.4f}",
                "TTA BCE": f"{modes['tta'].get('mean_bce', float('nan')):.4f}",
            }
        )
    return pd.DataFrame(rows)


def save_comparison_chart(all_stats: Dict[str, Dict[str, Dict]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = list(all_stats.keys())
    methods = ["baseline", "ood", "tta"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    # Two rows: top = Dice, bottom = BCE
    fig, axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 10), squeeze=False)

    for i, ds in enumerate(datasets):
        # Row 0: Dice
        vals_dice = [all_stats[ds][m]["mean_dice"] for m in methods]
        axes[0, i].bar([m.upper() for m in methods], vals_dice, color=colors)
        axes[0, i].set_title(f"{ds} Dice")
        axes[0, i].set_ylim([0.0, 1.0])
        for j, v in enumerate(vals_dice):
            axes[0, i].text(j, v + 0.02, f"{v:.3f}", ha="center")

        # Row 1: BCE (lower is better)
        vals_bce = [all_stats[ds][m].get("mean_bce", 0.0) for m in methods]
        axes[1, i].bar([m.upper() for m in methods], vals_bce, color=colors)
        axes[1, i].set_title(f"{ds} BCE")
        for j, v in enumerate(vals_bce):
            axes[1, i].text(j, v + max(vals_bce) * 0.02 if max(vals_bce) > 0 else 0.002, f"{v:.3f}", ha="center")

    fig.suptitle("MedSAM Baseline vs OOD vs TTA", fontsize=14)
    plt.tight_layout()

    out_path = output_dir / "performance_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
