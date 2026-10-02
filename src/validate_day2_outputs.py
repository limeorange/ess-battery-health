#!/usr/bin/env python3
"""Day 2 제출 산출물의 구조·누수·완결성을 빠르게 검증한다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "day2"
FIGURES = ROOT / "figures" / "day2"
REPORT = ROOT / "reports" / "DAY2_분석_보고서.md"
PLAN = ROOT / "reports" / "DAY2_분석_계획.md"
FORBIDDEN = ("cycle_life", "knee", "eol", "target", "global_cell_id", "batch")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    features = pd.read_csv(RESULTS / "feature_table.csv")
    split = pd.read_csv(RESULTS / "split_assignment.csv")
    manifest = pd.read_csv(RESULTS / "feature_manifest.csv")
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    performance = pd.read_csv(RESULTS / "model_performance.csv")
    gaps = pd.read_csv(RESULTS / "gap_analysis.csv")
    validation = pd.read_csv(RESULTS / "feature_reproduction_validation.csv")
    lock = json.loads((RESULTS / "external_evaluation_lock.json").read_text(encoding="utf-8"))

    require(len(features) == 139, "원본 Cell 수가 139개가 아닙니다.")
    require(features["global_cell_id"].is_unique, "Feature Table에 중복 Cell이 있습니다.")
    expected_splits = {"development": 36, "holdout": 10, "external_test": 39, "additional_test": 44}
    counts = split["split"].value_counts().to_dict()
    for name, expected in expected_splits.items():
        require(counts.get(name) == expected, f"{name} Cell 수가 {expected}개가 아닙니다.")

    selected = manifest.loc[manifest["final_selected"].astype(str).str.lower() == "true", "feature_name"].tolist()
    require(set(selected) == set(lock["selected_features"]), "Manifest와 외부평가 잠금의 최종 Feature가 다릅니다.")
    require(not any(any(token in feature.lower() for token in FORBIDDEN) for feature in selected), "금지 Feature가 선택됐습니다.")
    require(manifest.loc[manifest["feature_name"].isin(selected), "leakage_check"].notna().all(), "누수 검증 기록이 없습니다.")
    require(set(comparison["model"]) == {"Median Baseline", "Linear Regression", "Ridge", "ElasticNet", "Gradient Boosting"}, "필수 모델 비교가 불완전합니다.")
    require(set(comparison["feature_set"]) == {"F0 Capacity", "F1 ΔQ", "F2 ΔQ+Capacity", "F3 +Sensor", "F4 +Charging"}, "F0~F4가 불완전합니다.")
    require(set(performance["dataset"]) == {"Train (Batch 1 CV)", "Batch 1 Hold-out", "Batch 2 Test", "Batch 3 Test"}, "필수 성능 행이 없습니다.")
    require(set(gaps["gap"]) == {"Train-Valid", "Valid-Test", "Target-Test", "Batch2-Batch3"}, "필수 Gap이 없습니다.")
    require((validation["status"] == "통과").all(), "Day 1 Feature 재현 검증이 모두 통과하지 않았습니다.")
    require(lock["batch2_used_for_selection"] is False, "Batch 2가 모델 선택에 사용됐습니다.")
    require(lock["post_batch2_retuning"] is False, "Batch 2 평가 후 재튜닝 기록이 있습니다.")
    require(REPORT.exists() and REPORT.stat().st_size > 10_000, "Day 2 최종 보고서가 없거나 너무 작습니다.")
    require(PLAN.exists(), "Day 2 분석 계획서가 없습니다.")
    figures = sorted(FIGURES.glob("*.png"))
    require(len(figures) == 9, "Day 2 그래프가 9개가 아닙니다.")
    require(all(path.stat().st_size > 20_000 for path in figures), "비어 있거나 손상된 그래프가 있습니다.")

    summary = {
        "status": "PASS",
        "raw_cells": len(features),
        "split_counts": expected_splits,
        "selected_features": selected,
        "selected_model": lock["selected_model"],
        "result_csv_count": len(list(RESULTS.glob("*.csv"))),
        "figure_count": len(figures),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
