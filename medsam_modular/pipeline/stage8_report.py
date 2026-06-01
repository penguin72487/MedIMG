"""Stage 8 report and plotting entry points."""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from medsam_modular.visualize import (
    build_comparison_table,
    merge_stage8_stats,
    save_cache_throughput_trend_chart,
    save_cost_breakdown_chart,
    save_delta_chart,
    save_four_way_variant_chart,
    save_method_overview_chart,
    save_ood_detection_chart,
    save_quality_throughput_frontier,
    save_tta_cache_hit_chart,
)


def _all_have_baseline_stats(all_stats: Dict[str, Dict[str, Dict[str, Any]]]) -> bool:
    if not all_stats:
        return False
    for _, modes in all_stats.items():
        baseline = modes.get("baseline")
        if not isinstance(baseline, dict) or not baseline:
            return False
    return True


def run_stage8_plotting(
    *,
    all_stats: Dict[str, Dict[str, Dict[str, Any]]],
    all_stats_ood_finetuned: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
    output_dir: Path,
    project_root: Path,
    profiler: Any,
) -> Tuple[Path, Path, Path, Dict[str, Path], Optional[Path]]:
    from medsam_modular.runner import _timed_log

    comparison_path = output_dir / "comparison_table.csv"
    chart_path = output_dir / "performance_comparison_4way.png"
    top_chart_path = (project_root / "results") / chart_path.name
    stage8_paths: Dict[str, Path] = {}
    stage8_history_path: Optional[Path] = None
    stage8_plot_stats = merge_stage8_stats(
        full_summary=all_stats,
        ood_finetuned_summary=all_stats_ood_finetuned,
    )

    if not _all_have_baseline_stats(all_stats):
        print("\n[Stage 8/8] baseline stats 缺失，略過 comparison table/chart 產生。")
    else:
        with _timed_log("Stage 8: build comparison table"):
            with profiler.section_and_flush("stage.build_comparison"):
                comparison_table = build_comparison_table(all_stats)
        with _timed_log("Stage 8: save comparison CSV"):
            with profiler.section_and_flush("stage.save_comparison_csv"):
                comparison_table.to_csv(comparison_path, index=False)
        with _timed_log("Stage 8: save 4-way comparison chart"):
            with profiler.section_and_flush("stage.save_comparison_chart_4way"):
                chart_path = save_four_way_variant_chart(
                    full_summary=all_stats,
                    ood_finetuned_summary=all_stats_ood_finetuned or {},
                    output_dir=output_dir,
                )

    with _timed_log("Stage 8: save method overview chart"):
        with profiler.section_and_flush("stage.save_stage8_method_overview"):
            stage8_paths["method_overview"] = save_method_overview_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save delta chart"):
        with profiler.section_and_flush("stage.save_stage8_delta"):
            stage8_paths["delta_vs_baseline"] = save_delta_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save cost breakdown chart"):
        with profiler.section_and_flush("stage.save_stage8_cost_breakdown"):
            stage8_paths["cost_breakdown"] = save_cost_breakdown_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save quality-throughput frontier"):
        with profiler.section_and_flush("stage.save_stage8_frontier"):
            stage8_paths["quality_throughput_frontier"] = save_quality_throughput_frontier(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save OOD detection chart"):
        with profiler.section_and_flush("stage.save_stage8_ood_detection"):
            stage8_paths["ood_detection_quality"] = save_ood_detection_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save TTA cache chart"):
        with profiler.section_and_flush("stage.save_stage8_tta_cache"):
            stage8_paths["tta_cache_hits"] = save_tta_cache_hit_chart(stage8_plot_stats, output_dir)
    with _timed_log("Stage 8: save cache throughput trend"):
        with profiler.section_and_flush("stage.save_stage8_cache_throughput_trend"):
            trend_path, history_path = save_cache_throughput_trend_chart(stage8_plot_stats, output_dir)
            stage8_paths["cache_throughput_trend"] = trend_path
            stage8_history_path = history_path

    top_results_dir = project_root / "results"
    top_results_dir.mkdir(parents=True, exist_ok=True)
    top_chart_path = top_results_dir / chart_path.name
    with _timed_log("Stage 8: copy top-level comparison chart"):
        with profiler.section_and_flush("stage.copy_chart"):
            if chart_path.exists() and chart_path.resolve() != top_chart_path.resolve():
                shutil.copy2(chart_path, top_chart_path)

    return comparison_path, chart_path, top_chart_path, stage8_paths, stage8_history_path


__all__ = ["run_stage8_plotting"]
