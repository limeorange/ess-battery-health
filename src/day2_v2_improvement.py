#!/usr/bin/env python3
"""Day 2 v2: Batch 2 일반화 개선을 위한 사후 추가 실험.

공식 v1 결과는 변경하지 않는다. 원 논문형 초기 Feature, ΔQ(V) 전체 곡선,
다중 cycle ΔQ trajectory를 초기 100 cycle 안에서만 만들고 Batch 1
development nested CV로 후보 하나를 선택한다. 선택과 설정을 파일로 잠근 뒤
Batch 2/3를 평가한다. 이 결과는 공식 blind test의 대체가 아니라 post-hoc
improvement study다.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))
MPL_CACHE = ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.day2_analysis import (
    BATCH_FILES,
    RANDOM_SEED,
    configure_korean_plotting,
    decode_matlab_char,
    deref_scalar,
    flat,
    parse_policy,
    save_csv,
    save_figure,
    valid_metric_mask,
)


RESULT_DIR = ROOT / "results" / "day2_v2"
FIGURE_DIR = ROOT / "figures" / "day2_v2"
REPORT_PATH = ROOT / "reports" / "DAY2_추가_성능개선_보고서.md"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

COMMON_VOLTAGE = np.linspace(2.0, 3.5, 1000)
TRAJECTORY_CYCLES = tuple(range(20, 101, 10))


def value_at_cycle(cycles: np.ndarray, values: np.ndarray, cycle: int, metric: str) -> float:
    mask = np.isfinite(cycles) & np.isclose(cycles, cycle) & valid_metric_mask(metric, values)
    return float(values[np.flatnonzero(mask)[0]]) if mask.any() else np.nan


def window_mean(
    cycles: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
    metric: str,
) -> float:
    mask = (cycles >= start) & (cycles <= end) & valid_metric_mask(metric, values)
    return float(np.mean(values[mask])) if mask.any() else np.nan


def interpolate_curve(voltage: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    mask = np.isfinite(voltage) & np.isfinite(capacity)
    if mask.sum() < 20:
        return np.full(COMMON_VOLTAGE.shape, np.nan)
    v, q = voltage[mask], capacity[mask]
    order = np.argsort(v)
    v, q = v[order], q[order]
    unique_v, unique_idx = np.unique(v, return_index=True)
    q = q[unique_idx]
    result = np.interp(COMMON_VOLTAGE, unique_v, q, left=np.nan, right=np.nan)
    return result


def curve_statistics(prefix: str, curve: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(curve)
    if mask.sum() < 20:
        return {
            f"{prefix}_log10_var": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_abs_area": np.nan,
            f"{prefix}_l2": np.nan,
        }
    values = curve[mask]
    voltage = COMMON_VOLTAGE[mask]
    variance = float(np.var(values, ddof=1))
    return {
        f"{prefix}_log10_var": float(np.log10(variance + 1e-12)),
        f"{prefix}_iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_abs_area": float(np.trapezoid(np.abs(values), voltage)),
        f"{prefix}_l2": float(np.sqrt(np.mean(np.square(values)))),
    }


def extract_v2_features() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract paper-like scalars and multi-cycle ΔQ curves from raw files."""
    rows: list[dict[str, Any]] = []
    curve100_rows: list[np.ndarray] = []
    trajectory_rows: list[np.ndarray] = []

    metric_fields = {
        "qd": "QDischarge",
        "ir": "IR",
        "tavg": "Tavg",
        "tmax": "Tmax",
        "chargetime": "chargetime",
    }

    for batch_name, filename in BATCH_FILES.items():
        print(f"[v2 Feature] {batch_name}: {filename}", flush=True)
        with h5py.File(ROOT / "data" / filename, "r") as handle:
            batch = handle["batch"]
            n_cells = int(batch["cycle_life"].shape[0])
            for cell_index in range(n_cells):
                global_id = f"{batch_name.replace(' ', '')}_{cell_index:03d}"
                summary = handle[batch["summary"][cell_index, 0]]
                cycles = flat(summary["cycle"])
                arrays = {name: flat(summary[field]) for name, field in metric_fields.items()}
                common_length = min([len(cycles), *[len(v) for v in arrays.values()]])
                cycles = cycles[:common_length]
                arrays = {name: value[:common_length] for name, value in arrays.items()}

                qd = arrays["qd"]
                qd_mask = (cycles >= 2) & (cycles <= 100) & valid_metric_mask("qd", qd)
                qd_x, qd_y = cycles[qd_mask], qd[qd_mask]
                qd_cycle2 = value_at_cycle(cycles, qd, 2, "qd")
                if len(qd_y):
                    qd_max_index = int(np.argmax(qd_y))
                    qd_max = float(qd_y[qd_max_index])
                    qd_max_cycle = float(qd_x[qd_max_index])
                else:
                    qd_max, qd_max_cycle = np.nan, np.nan
                if len(qd_y) >= 3:
                    qd_slope, qd_intercept = np.polyfit(qd_x, qd_y, 1)
                else:
                    qd_slope, qd_intercept = np.nan, np.nan

                ir_cycle2 = value_at_cycle(cycles, arrays["ir"], 2, "ir")
                ir_cycle100 = value_at_cycle(cycles, arrays["ir"], 100, "ir")
                policy = decode_matlab_char(handle[batch["policy_readable"][cell_index, 0]])
                policy_info = parse_policy(policy)
                record: dict[str, Any] = {
                    "batch": batch_name,
                    "cell_index": cell_index,
                    "global_cell_id": global_id,
                    "cycle_life": deref_scalar(handle, batch["cycle_life"][cell_index, 0]),
                    "qd_cycle2": qd_cycle2,
                    "qd_cycle100": value_at_cycle(cycles, qd, 100, "qd"),
                    "qd_max_minus_cycle2": qd_max - qd_cycle2,
                    "qd_max_cycle": qd_max_cycle,
                    "qd_linear_slope_2_100": float(qd_slope),
                    "qd_linear_intercept_2_100": float(qd_intercept),
                    "qd_slope_95_100": (
                        float(np.polyfit(cycles[(cycles >= 95) & (cycles <= 100) & valid_metric_mask("qd", qd)],
                                         qd[(cycles >= 95) & (cycles <= 100) & valid_metric_mask("qd", qd)], 1)[0])
                        if ((cycles >= 95) & (cycles <= 100) & valid_metric_mask("qd", qd)).sum() >= 3
                        else np.nan
                    ),
                    "charge_time_mean_2_6": window_mean(cycles, arrays["chargetime"], 2, 6, "chargetime"),
                    "ir_min_2_100": (
                        float(np.min(arrays["ir"][(cycles >= 2) & (cycles <= 100) & valid_metric_mask("ir", arrays["ir"])]))
                        if ((cycles >= 2) & (cycles <= 100) & valid_metric_mask("ir", arrays["ir"])).any()
                        else np.nan
                    ),
                    "ir_cycle100_minus_cycle2": ir_cycle100 - ir_cycle2,
                    "tavg_mean_2_100": window_mean(cycles, arrays["tavg"], 2, 100, "tavg"),
                    "tmax_mean_2_100": window_mean(cycles, arrays["tmax"], 2, 100, "tmax"),
                    **policy_info,
                }

                cycle_to_index = {
                    int(round(cycle)): idx for idx, cycle in enumerate(cycles) if np.isfinite(cycle)
                }
                curves = handle[batch["cycles"][cell_index, 0]]["Qdlin"]
                voltage = flat(handle[batch["Vdlin"][cell_index, 0]])
                interpolated: dict[int, np.ndarray] = {}
                for target_cycle in (10, *TRAJECTORY_CYCLES):
                    idx = cycle_to_index.get(target_cycle)
                    if idx is None or idx >= curves.shape[0]:
                        interpolated[target_cycle] = np.full(COMMON_VOLTAGE.shape, np.nan)
                        continue
                    raw = np.asarray(handle[curves[idx, 0]]).reshape(-1)
                    if raw.dtype.kind != "f" or len(raw) != len(voltage):
                        interpolated[target_cycle] = np.full(COMMON_VOLTAGE.shape, np.nan)
                    else:
                        interpolated[target_cycle] = interpolate_curve(voltage, raw.astype(float))

                base_curve = interpolated[10]
                trajectory_curves = []
                for target_cycle in TRAJECTORY_CYCLES:
                    delta = interpolated[target_cycle] - base_curve
                    record.update(curve_statistics(f"dq{target_cycle}_10", delta))
                    trajectory_curves.append(delta)
                curve100_rows.append(trajectory_curves[-1])
                trajectory_rows.append(np.concatenate(trajectory_curves))
                rows.append(record)

    features = pd.DataFrame(rows)
    curve100 = np.vstack(curve100_rows)
    trajectory = np.vstack(trajectory_rows)
    if not features["global_cell_id"].is_unique:
        raise AssertionError("v2 global_cell_id가 유일하지 않습니다.")
    return features, curve100, trajectory


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mape_pct": 100 * mean_absolute_percentage_error(y_true, prediction),
        "mae": mean_absolute_error(y_true, prediction),
        "rmse": math.sqrt(mean_squared_error(y_true, prediction)),
        "r2": r2_score(y_true, prediction),
        "mean_bias": float(np.mean(prediction - y_true)),
        "over_prediction_pct": 100 * float(np.mean(prediction > y_true)),
    }


def log_target(regressor: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )


@dataclass
class Candidate:
    name: str
    feature_family: str
    X: np.ndarray
    estimator: Any
    grid: dict[str, list[Any]]
    feature_count: int
    rationale: str


def scalar_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def curve_pls_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", PLSRegression(scale=False, max_iter=1000)),
        ]
    )


def curve_pca_ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=RANDOM_SEED)),
            ("model", Ridge()),
        ]
    )


def nested_score(candidate: Candidate, dev_indices: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    X = candidate.X[dev_indices]
    y_dev = y[dev_indices]
    outer = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    rows = []
    for outer_fold, (train_idx, valid_idx) in enumerate(outer.split(X), start=1):
        inner = KFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED + outer_fold)
        search = GridSearchCV(
            clone(candidate.estimator),
            candidate.grid,
            scoring="neg_mean_absolute_percentage_error",
            cv=inner,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X[train_idx], y_dev[train_idx])
        prediction = np.asarray(search.predict(X[valid_idx])).reshape(-1)
        fold_metrics = metrics(y_dev[valid_idx], prediction)
        rows.append(
            {
                "candidate": candidate.name,
                "outer_fold": outer_fold,
                "repeat": (outer_fold - 1) // 5 + 1,
                "fold": (outer_fold - 1) % 5 + 1,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "best_params": json.dumps(search.best_params_, ensure_ascii=False, default=float),
                **fold_metrics,
            }
        )
    folds = pd.DataFrame(rows)
    repeat_means = folds.groupby("repeat")["mape_pct"].mean()
    summary = {
        "nested_cv_mape_mean": float(folds["mape_pct"].mean()),
        "nested_cv_mape_std": float(folds["mape_pct"].std(ddof=1)),
        "nested_cv_mape_se": float(repeat_means.std(ddof=1) / math.sqrt(len(repeat_means))),
        "nested_cv_mae_mean": float(folds["mae"].mean()),
        "nested_cv_rmse_mean": float(folds["rmse"].mean()),
        "nested_cv_r2_mean": float(folds["r2"].mean()),
        "worst_fold_mape": float(folds["mape_pct"].max()),
    }
    return folds, summary


def tune_on_development(candidate: Candidate, dev_indices: np.ndarray, y: np.ndarray) -> GridSearchCV:
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    search = GridSearchCV(
        clone(candidate.estimator),
        candidate.grid,
        scoring="neg_mean_absolute_percentage_error",
        cv=cv,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(candidate.X[dev_indices], y[dev_indices])
    return search


def build_candidates(data: pd.DataFrame, curve100: np.ndarray, trajectory: np.ndarray) -> list[Candidate]:
    f2 = ["delta_q_iqr", "qd_mean", "qd_slope"]
    paper_like = [
        "delta_q_log10_var", "delta_q_min", "qd_cycle2", "qd_cycle100",
        "qd_max_minus_cycle2", "qd_max_cycle", "qd_linear_slope_2_100",
        "qd_linear_intercept_2_100", "qd_slope_95_100", "charge_time_mean_2_6",
        "ir_min_2_100", "ir_cycle100_minus_cycle2", "tavg_mean_2_100", "tmax_mean_2_100",
        "c_rate_stage1", "switch_soc_pct", "c_rate_stage2",
    ]
    trajectory_columns = [
        f"dq{cycle}_10_{stat}"
        for cycle in TRAJECTORY_CYCLES
        for stat in ("log10_var", "iqr", "min", "abs_area", "l2")
    ]
    ridge = scalar_pipeline(Ridge())
    elastic = scalar_pipeline(ElasticNet(max_iter=200000, random_state=RANDOM_SEED))
    pls = curve_pls_pipeline()
    pca_ridge = curve_pca_ridge_pipeline()
    return [
        Candidate(
            "v1 F2 Ridge", "현재 Scalar 기준", data[f2].to_numpy(float), ridge,
            {"model__alpha": list(np.logspace(-3, 3, 7))}, len(f2),
            "공식 v1과 같은 세 Feature의 중첩 CV 기준선",
        ),
        Candidate(
            "논문형 ElasticNet log(y)", "원 논문형 Scalar", data[paper_like].to_numpy(float),
            log_target(elastic),
            {
                "regressor__model__alpha": list(np.logspace(-3, 1, 5)),
                "regressor__model__l1_ratio": [0.1, 0.5, 0.9],
            },
            len(paper_like), "초기 방전·충전시간·저항 Feature와 로그 수명",
        ),
        Candidate(
            "ΔQ 곡선 PLS log(y)", "ΔQ(V) 전체 곡선", curve100,
            log_target(pls),
            {"regressor__model__n_components": [1, 2, 3, 4, 5, 6]},
            curve100.shape[1], "단일 통계량 대신 전압별 ΔQ 형태를 PLS로 압축",
        ),
        Candidate(
            "ΔQ 곡선 PCA-Ridge log(y)", "ΔQ(V) 전체 곡선", curve100,
            log_target(pca_ridge),
            {
                "regressor__pca__n_components": [2, 4, 6, 8],
                "regressor__model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            curve100.shape[1], "비지도 곡선 압축 후 규제 선형회귀",
        ),
        Candidate(
            "다중 Cycle ΔQ PLS log(y)", "ΔQ trajectory", data[trajectory_columns].to_numpy(float),
            log_target(pls),
            {"regressor__model__n_components": [1, 2, 3, 4, 5, 6]},
            len(trajectory_columns), "Cycle 20~100의 ΔQ 변화 진행과정을 저차원으로 압축",
        ),
        Candidate(
            "다중 Cycle 원곡선 PLS log(y)", "ΔQ trajectory 원곡선", trajectory,
            log_target(pls),
            {"regressor__model__n_components": [1, 2, 3, 4]},
            trajectory.shape[1], "9개 ΔQ 원곡선의 전압·시간 정보를 동시에 압축",
        ),
    ]


def build_v2_feature_manifest() -> pd.DataFrame:
    rows = [
        ("qd_cycle2", "원 논문형 용량", "summary.QDischarge", "2", "QD at actual cycle 2", "Ah"),
        ("qd_cycle100", "원 논문형 용량", "summary.QDischarge", "100", "QD at actual cycle 100", "Ah"),
        ("qd_max_minus_cycle2", "원 논문형 용량", "summary.QDischarge", "2~100", "max(QD)-QD(cycle 2)", "Ah"),
        ("qd_max_cycle", "원 논문형 용량", "summary.QDischarge", "2~100", "argmax cycle of QD", "cycle"),
        ("qd_linear_slope_2_100", "원 논문형 용량", "summary.QDischarge", "2~100", "OLS slope of QD~cycle", "Ah/cycle"),
        ("qd_linear_intercept_2_100", "원 논문형 용량", "summary.QDischarge", "2~100", "OLS intercept of QD~cycle", "Ah"),
        ("qd_slope_95_100", "후기 초기구간 용량", "summary.QDischarge", "95~100", "OLS slope of QD~cycle", "Ah/cycle"),
        ("charge_time_mean_2_6", "충전시간", "summary.chargetime", "2~6", "mean(valid charge time)", "min"),
        ("ir_min_2_100", "내부저항", "summary.IR", "2~100", "min(valid IR)", "ohm"),
        ("ir_cycle100_minus_cycle2", "내부저항", "summary.IR", "2,100", "IR(100)-IR(2)", "ohm"),
        ("delta_q_curve_100_10", "ΔQ 전체 곡선", "cycles.Qdlin", "10,100", "Qdlin(100,V)-Qdlin(10,V), 1000-point grid", "Ah"),
        ("delta_q_trajectory", "ΔQ 다중 Cycle", "cycles.Qdlin", "10,20,...,100", "Qdlin(k,V)-Qdlin(10,V), k=20..100", "Ah"),
    ]
    manifest = pd.DataFrame(
        rows, columns=["feature_name", "feature_family", "source_field", "cycle_range", "formula", "unit"]
    )
    manifest["missing_rule"] = "각 CV train fold의 median imputation"
    manifest["transform_rule"] = "Scaler/PCA/PLS를 Pipeline 안에서 fold train에만 fit"
    manifest["leakage_check"] = "실제 cycle 번호 100 이하"
    manifest["official_v1_replacement"] = False
    return manifest


def plot_results(comparison: pd.DataFrame, predictions: pd.DataFrame) -> None:
    ordered = comparison.sort_values("nested_cv_mape_mean", ascending=False)
    colors = ["#1A9AA3" if selected else "#AAB5C0" for selected in ordered["selected_v2"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(ordered["candidate"], ordered["nested_cv_mape_mean"],
            xerr=ordered["nested_cv_mape_std"], color=colors, capsize=4)
    ax.set(title="Batch 1 중첩 교차검증 기반 v2 후보 비교", xlabel="Nested CV MAPE (%)", ylabel="")
    ax.text(0.99, 0.02, "Batch 2·3 Target은 후보 선택에 사용하지 않음",
            transform=ax.transAxes, ha="right", color="#526170")
    save_figure(fig, FIGURE_DIR / "01_v2_후보_중첩CV.png")

    external = predictions[predictions["dataset"].isin(["Batch 2 Test", "Batch 3 Test"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    palette = {"v1": "#9AA5B1", "v2": "#1A9AA3"}
    sns.barplot(data=external, x="dataset", y="ape_pct", hue="version", palette=palette,
                errorbar=None, ax=axes[0])
    axes[0].axhline(30, color="#D95D5D", ls="--", label="목표 30%")
    axes[0].set(title="외부 Batch MAPE 비교", xlabel="평가 데이터", ylabel="MAPE (%)")
    axes[0].legend(title="모델")
    batch2 = external.query("dataset == 'Batch 2 Test'")
    sns.scatterplot(data=batch2, x="cycle_life", y="prediction", hue="version",
                    palette=palette, s=65, ax=axes[1])
    low = min(batch2["cycle_life"].min(), batch2["prediction"].min()) * 0.9
    high = max(batch2["cycle_life"].max(), batch2["prediction"].max()) * 1.05
    axes[1].plot([low, high], [low, high], "--", color="#334155")
    axes[1].set(title="Batch 2 실제값–예측값", xlabel="실제 수명 (cycle)", ylabel="예측 수명 (cycle)")
    save_figure(fig, FIGURE_DIR / "02_v1_v2_외부성능.png")

    batch2_v2 = batch2.query("version == 'v2'").copy()
    fig, ax = plt.subplots(figsize=(10, 5.3))
    sns.histplot(batch2_v2["residual"], bins=12, color="#1A9AA3", edgecolor="white", ax=ax)
    ax.axvline(0, color="#334155", ls="--")
    ax.set(title="v2 Batch 2 잔차 분포", xlabel="잔차 = 예측 − 실제 (cycle)", ylabel="Cell 수")
    save_figure(fig, FIGURE_DIR / "03_v2_Batch2_잔차.png")


def plot_external_benchmark(benchmark: pd.DataFrame) -> None:
    batch2 = benchmark.query("dataset == 'Batch 2 Test'").sort_values("mape_pct", ascending=False)
    colors = ["#1A9AA3" if value < 30 else "#AAB5C0" for value in batch2["mape_pct"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(batch2["candidate"], batch2["mape_pct"], color=colors)
    ax.axvline(30, color="#D95D5D", ls="--", lw=1.5, label="20%대 경계")
    for index, value in enumerate(batch2["mape_pct"]):
        ax.text(value + 0.5, index, f"{value:.2f}%", va="center", fontsize=9)
    ax.set(title="사전 정의 후보의 Batch 2 사후 외부진단", xlabel="Batch 2 MAPE (%)", ylabel="")
    ax.legend()
    ax.text(0.99, 0.98, "후보 선택용 결과가 아닌 post-hoc sensitivity analysis",
            transform=ax.transAxes, ha="right", va="top", color="#526170")
    save_figure(fig, FIGURE_DIR / "04_사전후보_Batch2_외부진단.png")


def benchmark_all_predeclared_candidates(
    candidates: list[Candidate],
    data: pd.DataFrame,
    y: np.ndarray,
    dev_indices: np.ndarray,
    holdout_indices: np.ndarray,
    batch1_indices: np.ndarray,
    batch2_indices: np.ndarray,
    batch3_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every predeclared candidate for diagnosis, never for official selection."""
    performance_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    eval_indices = {
        "Batch 1 Hold-out": holdout_indices,
        "Batch 2 Test": batch2_indices,
        "Batch 3 Test": batch3_indices,
    }
    for candidate in candidates:
        print(f"[v2 Post-hoc external diagnostic] {candidate.name}", flush=True)
        search = tune_on_development(candidate, dev_indices, y)
        holdout_prediction = np.asarray(search.predict(candidate.X[holdout_indices])).reshape(-1)
        final_model = clone(search.best_estimator_)
        final_model.fit(candidate.X[batch1_indices], y[batch1_indices])
        predictions = {
            "Batch 1 Hold-out": holdout_prediction,
            "Batch 2 Test": np.asarray(final_model.predict(candidate.X[batch2_indices])).reshape(-1),
            "Batch 3 Test": np.asarray(final_model.predict(candidate.X[batch3_indices])).reshape(-1),
        }
        for dataset, indices in eval_indices.items():
            pred = predictions[dataset]
            performance_rows.append(
                {
                    "candidate": candidate.name,
                    "feature_family": candidate.feature_family,
                    "dataset": dataset,
                    "n": len(indices),
                    "best_params": json.dumps(search.best_params_, ensure_ascii=False, default=float),
                    **metrics(y[indices], pred),
                }
            )
            for idx, value in zip(indices, pred):
                actual = float(y[idx])
                prediction_rows.append(
                    {
                        "candidate": candidate.name,
                        "dataset": dataset,
                        "global_cell_id": data.iloc[idx]["global_cell_id"],
                        "cycle_life": actual,
                        "prediction": float(value),
                        "residual": float(value - actual),
                        "ape_pct": abs(float(value - actual)) / actual * 100,
                    }
                )
    return pd.DataFrame(performance_rows), pd.DataFrame(prediction_rows)


def write_report(
    comparison: pd.DataFrame,
    performance: pd.DataFrame,
    benchmark: pd.DataFrame,
    selected: Candidate,
    lock: dict[str, Any],
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        formatted = frame.copy()
        for column in formatted.select_dtypes(include=[np.number]).columns:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
        headers = [str(column) for column in formatted.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in formatted.astype(str).itertuples(index=False, name=None):
            lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
        return "\n".join(lines)

    v2_b2 = performance.query("version == 'v2' and dataset == 'Batch 2 Test'").iloc[0]
    v1_b2 = performance.query("version == 'v1' and dataset == 'Batch 2 Test'").iloc[0]
    achieved = float(v2_b2["mape_pct"]) < 30
    batch2_benchmark = benchmark.query("dataset == 'Batch 2 Test'").sort_values("mape_pct")
    best_exploratory = batch2_benchmark.iloc[0]
    text = f"""# Day 2 추가 성능 개선 실험

> 작성자: U094 이수현  
> 작성일: 2026-10-02  
> 성격: 공식 v1 외부평가 이후 수행한 post-hoc improvement study

## 1. 이 실험의 위치

공식 v1의 Feature·모델·Batch 2 MAPE 36.56%는 변경하지 않는다. 이 실험은
Batch 2에서 확인된 체계적 과대 예측을 줄일 수 있는지 검토하는 추가 연구다.
후보와 Hyperparameter는 Batch 1 development nested CV에서만 선택했지만,
실험 동기가 기존 Batch 2 결과에서 출발했으므로 v2 Batch 2 결과를 새로운
완전 blind test라고 주장하지 않는다.

## 2. 추가한 정보 표현

- 원 논문형 초기 Feature와 `log(cycle_life)` ElasticNet
- Cycle 100−10 ΔQ(V) 전체 1,000-point 곡선의 PLS/PCA 표현
- Cycle 20~100의 다중 ΔQ trajectory 통계 및 원곡선
- 모든 Imputer·Scaler·PCA·PLS는 Pipeline 안에서 fold train에만 fit

## 3. Batch 1 nested CV 후보 비교

{markdown_table(comparison[["candidate", "feature_family", "feature_count", "nested_cv_mape_mean", "nested_cv_mape_std", "worst_fold_mape", "selected_v2"]])}

선택 모델은 **{selected.name}**이다. 선택 설정은 `{lock['best_params']}`이며
Batch 2·3 Target은 선택에 사용하지 않았다.

![v2 후보 비교](../figures/day2_v2/01_v2_후보_중첩CV.png)

## 4. 외부 Batch 사후 비교

{markdown_table(performance)}

![v1-v2 외부성능](../figures/day2_v2/02_v1_v2_외부성능.png)

Batch 2 MAPE는 v1 **{v1_b2['mape_pct']:.2f}%**에서 v2 **{v2_b2['mape_pct']:.2f}%**로
{abs(v2_b2['mape_pct'] - v1_b2['mape_pct']):.2f}%p {'감소' if v2_b2['mape_pct'] < v1_b2['mape_pct'] else '증가'}했다.
20%대 목표는 **{'달성' if achieved else '미달성'}**했다.

## 5. 해석 원칙

- 성능 개선 여부와 관계없이 공식 v1 결과를 대체하지 않는다.
- Batch 2를 보고 v2 Feature나 Hyperparameter를 다시 변경하지 않는다.
- 내부 CV 개선이 외부 Batch 개선으로 이어지지 않으면 Batch shift 또는
  `Feature → 수명` 관계 변화가 주된 한계라는 증거로 해석한다.
- 최종 운영에서는 평균 MAPE뿐 아니라 과대 예측률과 평균 편향을 함께 본다.

## 6. 사전 정의 후보 전체의 사후 외부진단

아래 표는 v2 후보를 다시 선택하기 위한 표가 아니다. 잠금 후보가 Batch 2에서
악화된 뒤, 사전에 코드로 정의되어 있던 표현들이 외부 Batch에서 어떻게 행동하는지
확인한 sensitivity analysis다.

{markdown_table(batch2_benchmark[["candidate", "feature_family", "mape_pct", "mae", "rmse", "r2", "mean_bias", "over_prediction_pct"]])}

![사전 후보 외부진단](../figures/day2_v2/04_사전후보_Batch2_외부진단.png)

가장 낮은 Batch 2 MAPE는 **{best_exploratory['candidate']}의 {best_exploratory['mape_pct']:.2f}%**로
20%대에 도달했다. 그러나 이 후보를 Batch 2 결과를 근거로 공식 최종 모델로 승격하면
test-set selection이 된다. 따라서 이 수치는 **성능 개선 가능성의 증거**로만 보고하며,
새로운 독립 Batch에서 재검증되기 전까지 공식 v1을 대체하지 않는다.

중요한 발견은 Batch 1 내부에서 가장 좋았던 논문형 Scalar 모델보다 ΔQ 전체 곡선
모델이 Batch 2에서 훨씬 강했다는 점이다. Batch 1 내부 CV 순위와 Batch 2 순위가
다르므로, 단일 source Batch의 내부검증만으로 다른 실험 Batch의 최적 표현을 선택하기
어렵다는 한계가 확인됐다.

## 7. 재현 파일

- 코드: `src/day2_v2_improvement.py`
- 확장 Feature: `results/day2_v2/extended_feature_table.csv`
- Feature 명세: `results/day2_v2/v2_feature_manifest.csv`
- ΔQ 곡선: `results/day2_v2/delta_q_curve_data.npz`
- Nested CV: `results/day2_v2/nested_cv_comparison.csv`
- 잠금 기록: `results/day2_v2/v2_external_evaluation_lock.json`
- Cell별 예측: `results/day2_v2/v1_v2_predictions.csv`
- 사전 후보 외부진단: `results/day2_v2/predeclared_candidate_external_performance.csv`
- 의사결정 기록: `results/day2_v2/v2_decision_log.csv`
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    configure_korean_plotting()
    v1_features = pd.read_csv(ROOT / "results" / "day2" / "feature_table.csv", encoding="utf-8-sig")
    split = pd.read_csv(ROOT / "results" / "day2" / "split_assignment.csv", encoding="utf-8-sig")

    v2_features, curve100, trajectory = extract_v2_features()
    if list(v2_features["global_cell_id"]) != list(v1_features["global_cell_id"]):
        raise AssertionError("v1/v2 Cell 순서가 일치하지 않습니다.")
    v2_additional = v2_features.drop(columns=["batch", "cell_index", "cycle_life"])
    duplicated_v1_columns = [
        column for column in v2_additional.columns
        if column != "global_cell_id" and column in v1_features.columns
    ]
    v2_additional = v2_additional.drop(columns=duplicated_v1_columns)
    data = v1_features.merge(
        v2_additional, on="global_cell_id", how="left", validate="one_to_one",
    ).merge(split[["global_cell_id", "split"]], on="global_cell_id", validate="one_to_one")
    if not np.allclose(data["cycle_life"], v2_features["cycle_life"], equal_nan=True):
        raise AssertionError("v1/v2 Target이 일치하지 않습니다.")

    save_csv(data, RESULT_DIR / "extended_feature_table.csv")
    save_csv(build_v2_feature_manifest(), RESULT_DIR / "v2_feature_manifest.csv")
    np.savez_compressed(
        RESULT_DIR / "delta_q_curve_data.npz",
        global_cell_id=data["global_cell_id"].to_numpy(str),
        voltage=COMMON_VOLTAGE,
        cycle100_minus10=curve100,
        trajectory=trajectory,
        trajectory_cycles=np.asarray(TRAJECTORY_CYCLES),
    )

    y = data["cycle_life"].to_numpy(float)
    dev_indices = np.flatnonzero(data["split"].eq("development").to_numpy())
    holdout_indices = np.flatnonzero(data["split"].eq("holdout").to_numpy())
    batch1_indices = np.flatnonzero(data["split"].isin(["development", "holdout"]).to_numpy())
    batch2_indices = np.flatnonzero(data["split"].eq("external_test").to_numpy())
    batch3_indices = np.flatnonzero(data["split"].eq("additional_test").to_numpy())
    candidates = build_candidates(data, curve100, trajectory)

    fold_frames, comparison_rows = [], []
    for candidate in candidates:
        print(f"[v2 Nested CV] {candidate.name}", flush=True)
        folds, summary = nested_score(candidate, dev_indices, y)
        fold_frames.append(folds)
        comparison_rows.append(
            {
                "candidate": candidate.name,
                "feature_family": candidate.feature_family,
                "feature_count": candidate.feature_count,
                "rationale": candidate.rationale,
                **summary,
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("nested_cv_mape_mean").reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["candidate"])
    selected = next(candidate for candidate in candidates if candidate.name == selected_name)
    comparison["selected_v2"] = comparison["candidate"].eq(selected_name)
    save_csv(pd.concat(fold_frames, ignore_index=True), RESULT_DIR / "nested_cv_folds.csv")
    save_csv(comparison, RESULT_DIR / "nested_cv_comparison.csv")

    print(f"[v2 Selection] {selected.name}", flush=True)
    search = tune_on_development(selected, dev_indices, y)
    holdout_prediction = np.asarray(search.predict(selected.X[holdout_indices])).reshape(-1)
    holdout_metrics = metrics(y[holdout_indices], holdout_prediction)
    lock_path = RESULT_DIR / "v2_external_evaluation_lock.json"
    proposed_lock = {
        "status": "locked_before_v2_external_prediction",
        "created_at": datetime.now().astimezone().isoformat(),
        "study_type": "post-hoc improvement study; official v1 unchanged",
        "selection_data": "Batch 1 development only",
        "selected_candidate": selected.name,
        "feature_family": selected.feature_family,
        "feature_count": selected.feature_count,
        "best_params": json.loads(json.dumps(search.best_params_, default=float)),
        "nested_cv_mape_pct": float(comparison.iloc[0]["nested_cv_mape_mean"]),
        "batch1_holdout_mape_pct": float(holdout_metrics["mape_pct"]),
        "batch2_target_used_for_selection": False,
        "batch3_target_used_for_selection": False,
        "post_external_retuning_allowed": False,
    }
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock["selected_candidate"] != proposed_lock["selected_candidate"] or lock["best_params"] != proposed_lock["best_params"]:
            raise RuntimeError("기존 v2 외부평가 잠금과 재계산된 선택이 다릅니다.")
    else:
        lock = proposed_lock
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")

    final_model = clone(search.best_estimator_)
    final_model.fit(selected.X[batch1_indices], y[batch1_indices])
    v1_predictions = pd.read_csv(ROOT / "results" / "day2" / "predictions.csv", encoding="utf-8-sig")
    prediction_rows = []
    eval_sets = {
        "Batch 1 Hold-out": (holdout_indices, holdout_prediction),
        "Batch 2 Test": (batch2_indices, np.asarray(final_model.predict(selected.X[batch2_indices])).reshape(-1)),
        "Batch 3 Test": (batch3_indices, np.asarray(final_model.predict(selected.X[batch3_indices])).reshape(-1)),
    }
    for dataset, (indices, prediction) in eval_sets.items():
        for idx, pred in zip(indices, prediction):
            actual = float(y[idx])
            prediction_rows.append(
                {
                    "version": "v2", "candidate": selected.name, "dataset": dataset,
                    "global_cell_id": data.iloc[idx]["global_cell_id"], "cycle_life": actual,
                    "prediction": float(pred), "residual": float(pred - actual),
                    "ape_pct": abs(float(pred - actual)) / actual * 100,
                }
            )
    v2_predictions = pd.DataFrame(prediction_rows)
    v1_compare = v1_predictions[["dataset", "global_cell_id", "cycle_life", "prediction", "residual", "ape_pct"]].copy()
    v1_compare.insert(0, "candidate", "공식 F2 Ridge")
    v1_compare.insert(0, "version", "v1")
    all_predictions = pd.concat([v1_compare, v2_predictions], ignore_index=True)
    save_csv(all_predictions, RESULT_DIR / "v1_v2_predictions.csv")

    performance_rows = []
    for (version, candidate_name, dataset), group in all_predictions.groupby(
        ["version", "candidate", "dataset"], sort=False
    ):
        performance_rows.append(
            {
                "version": version, "candidate": candidate_name, "dataset": dataset, "n": len(group),
                **metrics(group["cycle_life"].to_numpy(), group["prediction"].to_numpy()),
            }
        )
    performance = pd.DataFrame(performance_rows)
    save_csv(performance, RESULT_DIR / "v1_v2_performance.csv")

    benchmark, benchmark_predictions = benchmark_all_predeclared_candidates(
        candidates, data, y, dev_indices, holdout_indices, batch1_indices, batch2_indices, batch3_indices
    )
    save_csv(benchmark, RESULT_DIR / "predeclared_candidate_external_performance.csv")
    save_csv(benchmark_predictions, RESULT_DIR / "predeclared_candidate_external_predictions.csv")
    best_batch2 = benchmark.query("dataset == 'Batch 2 Test'").sort_values("mape_pct").iloc[0]
    locked_v2_batch2_mape = float(
        performance.query("version == 'v2' and dataset == 'Batch 2 Test'").iloc[0]["mape_pct"]
    )
    decision_log = pd.DataFrame(
        [
            {
                "stage": "공식 v1 보존", "decision": "F2 Ridge 유지",
                "evidence": "사전 계획에 따른 최초 Batch 2 평가 36.56%", "uses_batch2_for_selection": False,
            },
            {
                "stage": "v2 잠금 선택", "decision": selected.name,
                "evidence": f"Batch 1 nested CV {comparison.iloc[0]['nested_cv_mape_mean']:.2f}%",
                "uses_batch2_for_selection": False,
            },
            {
                "stage": "v2 잠금 외부평가", "decision": "잠금 유지·재튜닝 금지",
                "evidence": f"Batch 2 MAPE {locked_v2_batch2_mape:.2f}%",
                "uses_batch2_for_selection": False,
            },
            {
                "stage": "사전 후보 외부진단", "decision": "공식 모델로 승격하지 않음",
                "evidence": f"{best_batch2['candidate']}가 Batch 2 {best_batch2['mape_pct']:.2f}%",
                "uses_batch2_for_selection": True,
            },
        ]
    )
    save_csv(decision_log, RESULT_DIR / "v2_decision_log.csv")

    plot_results(comparison, all_predictions)
    plot_external_benchmark(benchmark)
    write_report(comparison, performance, benchmark, selected, lock)
    print(comparison[["candidate", "nested_cv_mape_mean", "nested_cv_mape_std", "selected_v2"]].to_string(index=False))
    print(performance.to_string(index=False))
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
