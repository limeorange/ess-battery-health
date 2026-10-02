#!/usr/bin/env python3
"""Validate Day 2 v2 improvement artifacts without changing them."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import nbformat
import numpy as np
import pandas as pd


def main() -> None:
    result_dir = ROOT / "results" / "day2_v2"
    comparison = pd.read_csv(result_dir / "nested_cv_comparison.csv", encoding="utf-8-sig")
    benchmark = pd.read_csv(result_dir / "predeclared_candidate_external_performance.csv", encoding="utf-8-sig")
    predictions = pd.read_csv(result_dir / "predeclared_candidate_external_predictions.csv", encoding="utf-8-sig")
    lock = json.loads((result_dir / "v2_external_evaluation_lock.json").read_text(encoding="utf-8"))
    curve_data = np.load(result_dir / "delta_q_curve_data.npz", allow_pickle=False)

    batch2 = benchmark.query("dataset == 'Batch 2 Test'").set_index("candidate")
    checks = {
        "후보 6개 nested CV": len(comparison) == 6,
        "Nested CV 선택 1개": int(comparison["selected_v2"].sum()) == 1,
        "잠금 후보 일치": comparison.loc[comparison["selected_v2"], "candidate"].iloc[0] == lock["selected_candidate"],
        "Batch 2 선택 미사용": lock["batch2_target_used_for_selection"] is False,
        "사후 재튜닝 금지": lock["post_external_retuning_allowed"] is False,
        "ΔQ 곡선 크기": curve_data["cycle100_minus10"].shape == (139, 1000),
        "다중 Cycle 곡선 크기": curve_data["trajectory"].shape == (139, 9000),
        "외부 예측 결측 없음": predictions["prediction"].notna().all(),
        "외부 후보별 평가 3개": len(benchmark) == 18,
        "PLS Batch 2 20%대": 20 <= batch2.loc["ΔQ 곡선 PLS log(y)", "mape_pct"] < 30,
        "PCA-Ridge Batch 2 20%대": 20 <= batch2.loc["ΔQ 곡선 PCA-Ridge log(y)", "mape_pct"] < 30,
        "공식 v1 보존": abs(batch2.loc["v1 F2 Ridge", "mape_pct"] - 36.5603113887257) < 1e-9,
        "추가 보고서 존재": (ROOT / "reports" / "DAY2_추가_성능개선_보고서.md").exists(),
    }

    for notebook_name, required_heading in [
        ("02_feature_engineering.ipynb", "추가 개선 실험"),
        ("03_modeling.ipynb", "사전 정의 후보의 외부 민감도 분석"),
    ]:
        notebook = nbformat.read(ROOT / "notebooks" / notebook_name, as_version=4)
        source = "\n".join("".join(cell.get("source", "")) for cell in notebook.cells)
        errors = [
            output
            for cell in notebook.cells if cell.cell_type == "code"
            for output in cell.get("outputs", []) if output.output_type == "error"
        ]
        checks[f"{notebook_name} v2 설명"] = required_heading in source
        checks[f"{notebook_name} 실행 오류 없음"] = not errors
        checks[f"{notebook_name} 실행 출력 존재"] = any(
            cell.cell_type == "code" and cell.get("execution_count") is not None for cell in notebook.cells
        )

    failed = [name for name, passed in checks.items() if not passed]
    table = pd.DataFrame([{"검사": name, "결과": "통과" if passed else "실패"} for name, passed in checks.items()])
    print(table.to_string(index=False))
    if failed:
        raise AssertionError(f"v2 검증 실패: {failed}")
    print(f"\n모든 v2 검증 통과: {len(checks)}개")


if __name__ == "__main__":
    main()
