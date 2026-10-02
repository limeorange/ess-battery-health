#!/usr/bin/env python3
"""저장된 Day 2 결과표에서 보고서 그래프만 다시 렌더링한다."""

from pathlib import Path

import pandas as pd

from day2_analysis import (
    configure_korean_plotting,
    plot_ablation,
    plot_actual_vs_predicted,
    plot_batch_shift,
    plot_charging_error,
    plot_group_error,
    plot_importance,
    plot_model_comparison,
    plot_residuals,
    plot_worst_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "day2"
FIGURES = ROOT / "figures" / "day2"


def main() -> None:
    configure_korean_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    ablation = pd.read_csv(RESULTS / "ablation_results.csv")
    predictions = pd.read_csv(RESULTS / "predictions.csv")
    life_group_error = pd.read_csv(RESULTS / "life_group_error.csv")
    charging_error = pd.read_csv(RESULTS / "charging_error.csv")
    worst = pd.read_csv(RESULTS / "worst_predictions.csv")
    importance = pd.read_csv(RESULTS / "feature_importance.csv")
    batch_shift = pd.read_csv(RESULTS / "batch_shift.csv")

    plot_model_comparison(comparison, FIGURES)
    plot_ablation(ablation, FIGURES)
    plot_actual_vs_predicted(predictions, FIGURES)
    plot_residuals(predictions, FIGURES)
    plot_group_error(life_group_error, FIGURES)
    plot_charging_error(predictions, charging_error, FIGURES)
    plot_worst_predictions(worst, FIGURES)
    plot_importance(importance, FIGURES)
    plot_batch_shift(batch_shift, FIGURES)


if __name__ == "__main__":
    main()
