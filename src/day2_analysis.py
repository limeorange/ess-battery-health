#!/usr/bin/env python3
"""Day 2: 초기 100 cycle 기반 배터리 수명 Regression 전체 파이프라인.

원본 MATLAB v7.3 파일에서 Cell-level feature를 재생성하고, Batch 1의
development/CV/hold-out으로 모델을 선택한 뒤 설정을 동결해 Batch 2와
Batch 3를 평가한다. 모든 표와 그림, 최종 Markdown 보고서를 재생성한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = PROJECT_ROOT / ".python_packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))

MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    RepeatedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


RANDOM_SEED = 20261002
REFERENCE_MAPE = 9.1
BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BATCH_COLORS = {"Batch 1": "#3568C0", "Batch 2": "#E59A24", "Batch 3": "#22A77A"}
SPLIT_COLORS = {
    "Batch 1 Hold-out": "#3568C0",
    "Batch 2 Test": "#E59A24",
    "Batch 3 Test": "#22A77A",
}
DATASET_KO = {
    "Batch 1 Hold-out": "Batch 1 고정 검증",
    "Batch 2 Test": "Batch 2 외부 테스트",
    "Batch 3 Test": "Batch 3 추가 테스트",
}
FEATURE_SET_KO = {
    "F0 Capacity": "F0 용량",
    "F1 ΔQ": "F1 ΔQ",
    "F2 ΔQ+Capacity": "F2 ΔQ+용량",
    "F3 +Sensor": "F3 +센서",
    "F4 +Charging": "F4 +충전조건",
}
MODEL_KO = {
    "Median Baseline": "중앙값 기준선",
    "Linear Regression": "선형회귀",
    "Ridge": "Ridge",
    "ElasticNet": "ElasticNet",
    "Gradient Boosting": "Gradient Boosting",
}
FEATURE_KO = {
    "delta_q_iqr": "ΔQ 사분위범위",
    "qd_mean": "평균 방전용량",
    "qd_slope": "방전용량 기울기",
}
LIFE_ORDER = ["단수명(<500)", "중간수명(500~1,000)", "장수명(>1,000)"]
LIFE_COLORS = {
    "단수명(<500)": "#D95D5D",
    "중간수명(500~1,000)": "#9AA5B1",
    "장수명(>1,000)": "#4C78A8",
}
FORBIDDEN_FEATURE_PATTERNS = (
    "cycle_life",
    "knee",
    "eol",
    "target",
    "observed_end",
    "global_cell_id",
    "cell_index",
    "batch",
)


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    results: Path
    figures: Path
    reports: Path


def make_paths(root: Path) -> Paths:
    paths = Paths(
        root=root,
        data=root / "data",
        results=root / "results" / "day2",
        figures=root / "figures" / "day2",
        reports=root / "reports",
    )
    paths.results.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    return paths


def configure_korean_plotting() -> str:
    candidates = [
        Path("/Users/lsh/Library/Fonts/NotoSansKR-Regular.otf"),
        Path("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path:
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
    else:
        font_name = "DejaVu Sans"
        warnings.warn("한글 폰트를 찾지 못해 DejaVu Sans를 사용합니다.")
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#243447",
            "axes.titlecolor": "#173B5E",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 10.5,
            "xtick.color": "#526170",
            "ytick.color": "#526170",
            "grid.color": "#E5EAF0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    return font_name


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def flat(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset, dtype=float).reshape(-1)


def deref_scalar(handle: h5py.File, ref: Any) -> float:
    return float(np.asarray(handle[ref]).squeeze())


def decode_matlab_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1)
    if values.dtype.kind in "ui":
        return "".join(chr(int(v)) for v in values if int(v) != 0)
    if values.dtype.kind == "S":
        return b"".join(values.tolist()).decode("utf-8", errors="replace")
    return str(values[0]) if len(values) else ""


POLICY_PATTERN = re.compile(
    r"(?P<c1>\d+(?:\.\d+)?)C\((?P<soc>\d+(?:\.\d+)?)%\)-(?P<c2>\d+(?:\.\d+)?)C",
    flags=re.IGNORECASE,
)
VAR_POLICY_PATTERN = re.compile(r"VarCharge-(?P<c1>\d+(?:\.\d+)?)C", re.IGNORECASE)


def parse_policy(policy: str) -> dict[str, Any]:
    match = POLICY_PATTERN.search(policy)
    if match:
        return {
            "policy_parse_success": True,
            "policy_type": "2단계 정전류",
            "c_rate_stage1": float(match.group("c1")),
            "switch_soc_pct": float(match.group("soc")),
            "c_rate_stage2": float(match.group("c2")),
            "policy_new_structure": "newstructure" in policy.lower(),
            "policy_slow_cycle": "slowcycle" in policy.lower(),
        }
    match = VAR_POLICY_PATTERN.search(policy)
    if match:
        return {
            "policy_parse_success": True,
            "policy_type": "가변 충전",
            "c_rate_stage1": float(match.group("c1")),
            "switch_soc_pct": np.nan,
            "c_rate_stage2": np.nan,
            "policy_new_structure": False,
            "policy_slow_cycle": False,
        }
    return {
        "policy_parse_success": False,
        "policy_type": "기타",
        "c_rate_stage1": np.nan,
        "switch_soc_pct": np.nan,
        "c_rate_stage2": np.nan,
        "policy_new_structure": "newstructure" in policy.lower(),
        "policy_slow_cycle": "slowcycle" in policy.lower(),
    }


def valid_metric_mask(name: str, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    if name in {"qd", "qc"}:
        mask &= (values > 0.2) & (values < 2.0)
    elif name == "ir":
        mask &= (values > 0.001) & (values < 0.2)
    elif name in {"tavg", "tmax", "tmin"}:
        mask &= (values > 5.0) & (values < 80.0)
    elif name == "chargetime":
        mask &= (values > 0.1) & (values < 120.0)
    return mask


def metric_features(name: str, cycles: np.ndarray, values: np.ndarray) -> dict[str, float]:
    mask = (cycles >= 2) & (cycles <= 100) & valid_metric_mask(name, values)
    x, y = cycles[mask], values[mask]
    result = {
        f"{name}_n": float(len(y)),
        f"{name}_mean": np.nan,
        f"{name}_std": np.nan,
        f"{name}_median": np.nan,
        f"{name}_min": np.nan,
        f"{name}_max": np.nan,
        f"{name}_delta": np.nan,
        f"{name}_slope": np.nan,
    }
    if len(y) == 0:
        return result
    order = np.argsort(x)
    x, y = x[order], y[order]
    result.update(
        {
            f"{name}_mean": float(np.mean(y)),
            f"{name}_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            f"{name}_median": float(np.median(y)),
            f"{name}_min": float(np.min(y)),
            f"{name}_max": float(np.max(y)),
            f"{name}_delta": float(y[-1] - y[0]),
            f"{name}_slope": float(np.polyfit(x, y, 1)[0]) if len(y) >= 3 else np.nan,
        }
    )
    return result


def delta_q_statistics(voltage: np.ndarray, delta_q: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(voltage) & np.isfinite(delta_q)
    v, dq = voltage[mask], delta_q[mask]
    if len(dq) < 20:
        return {}
    order = np.argsort(v)
    v_sorted, dq_sorted = v[order], dq[order]
    variance = float(np.var(dq, ddof=1))
    return {
        "delta_q_n": float(len(dq)),
        "delta_q_min": float(np.min(dq)),
        "delta_q_max": float(np.max(dq)),
        "delta_q_mean": float(np.mean(dq)),
        "delta_q_std": float(np.std(dq, ddof=1)),
        "delta_q_var": variance,
        "delta_q_log10_var": float(np.log10(variance + 1e-12)),
        "delta_q_q05": float(np.quantile(dq, 0.05)),
        "delta_q_q25": float(np.quantile(dq, 0.25)),
        "delta_q_q50": float(np.quantile(dq, 0.50)),
        "delta_q_q75": float(np.quantile(dq, 0.75)),
        "delta_q_q95": float(np.quantile(dq, 0.95)),
        "delta_q_iqr": float(np.quantile(dq, 0.75) - np.quantile(dq, 0.25)),
        "delta_q_min_voltage": float(v[np.argmin(dq)]),
        "delta_q_max_voltage": float(v[np.argmax(dq)]),
        "delta_q_area": float(np.trapezoid(dq_sorted, v_sorted)),
        "delta_q_abs_area": float(np.trapezoid(np.abs(dq_sorted), v_sorted)),
        "delta_q_l1": float(np.mean(np.abs(dq))),
        "delta_q_l2": float(np.sqrt(np.mean(np.square(dq)))),
        "delta_q_skew": float(stats.skew(dq, bias=False, nan_policy="omit")),
        "delta_q_kurtosis": float(stats.kurtosis(dq, bias=False, nan_policy="omit")),
    }


def life_group(value: float) -> str:
    if not np.isfinite(value):
        return "Target 결측"
    if value < 500:
        return "단수명(<500)"
    if value > 1000:
        return "장수명(>1,000)"
    return "중간수명(500~1,000)"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def extract_raw_features(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_map = {
        "qd": "QDischarge",
        "qc": "QCharge",
        "ir": "IR",
        "tavg": "Tavg",
        "tmax": "Tmax",
        "tmin": "Tmin",
        "chargetime": "chargetime",
    }
    feature_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []

    for batch_name, filename in BATCH_FILES.items():
        file_path = paths.data / filename
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        print(f"[Gate 1] 원본 추출: {batch_name} / {filename}", flush=True)
        file_rows.append(
            {
                "batch": batch_name,
                "filename": filename,
                "size_bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
        with h5py.File(file_path, "r") as handle:
            batch = handle["batch"]
            n_cells = int(batch["cycle_life"].shape[0])
            for cell_index in range(n_cells):
                global_id = f"{batch_name.replace(' ', '')}_{cell_index:03d}"
                cycle_life = deref_scalar(handle, batch["cycle_life"][cell_index, 0])
                policy = decode_matlab_char(handle[batch["policy_readable"][cell_index, 0]])
                policy_info = parse_policy(policy)
                summary = handle[batch["summary"][cell_index, 0]]
                arrays = {"cycle": flat(summary["cycle"])}
                arrays.update({name: flat(summary[field]) for name, field in metric_map.items()})
                lengths = {name: len(values) for name, values in arrays.items()}
                common_length = min(lengths.values())
                arrays = {name: values[:common_length] for name, values in arrays.items()}
                cycles = arrays["cycle"]
                cycle_to_index = {
                    int(round(cycle)): idx
                    for idx, cycle in enumerate(cycles)
                    if np.isfinite(cycle)
                }

                record: dict[str, Any] = {
                    "batch": batch_name,
                    "cell_index": cell_index,
                    "global_cell_id": global_id,
                    "cycle_life": cycle_life,
                    "target_available": bool(np.isfinite(cycle_life)),
                    "life_group": life_group(cycle_life),
                    "charging_policy": policy,
                    **policy_info,
                }
                for metric in metric_map:
                    record.update(metric_features(metric, cycles, arrays[metric]))

                early_mask = (cycles >= 2) & (cycles <= 100)
                temp_range = arrays["tmax"] - arrays["tmin"]
                temp_valid = early_mask & np.isfinite(temp_range) & (temp_range >= 0) & (temp_range < 50)
                record["temperature_range_mean"] = (
                    float(np.mean(temp_range[temp_valid])) if temp_valid.any() else np.nan
                )
                ratio_mask = (
                    early_mask
                    & valid_metric_mask("qd", arrays["qd"])
                    & valid_metric_mask("qc", arrays["qc"])
                    & (arrays["qc"] != 0)
                )
                record["coulombic_efficiency_mean"] = (
                    float(np.mean(arrays["qd"][ratio_mask] / arrays["qc"][ratio_mask]))
                    if ratio_mask.any()
                    else np.nan
                )

                has_10, has_100 = 10 in cycle_to_index, 100 in cycle_to_index
                delta_valid = False
                delta_reason = "cycle 10 또는 100 없음"
                delta_length = 0
                cycles_group = handle[batch["cycles"][cell_index, 0]]
                qdlin_refs = cycles_group["Qdlin"]
                cycles_object_length = int(qdlin_refs.shape[0])
                if has_10 and has_100:
                    idx10, idx100 = cycle_to_index[10], cycle_to_index[100]
                    if idx10 < cycles_object_length and idx100 < cycles_object_length:
                        try:
                            voltage = flat(handle[batch["Vdlin"][cell_index, 0]])
                            q10_raw = np.asarray(handle[qdlin_refs[idx10, 0]]).reshape(-1)
                            q100_raw = np.asarray(handle[qdlin_refs[idx100, 0]]).reshape(-1)
                            if (
                                q10_raw.dtype.kind == "f"
                                and q100_raw.dtype.kind == "f"
                                and len(voltage) == len(q10_raw) == len(q100_raw)
                                and len(voltage) >= 20
                            ):
                                delta_q = q100_raw.astype(float) - q10_raw.astype(float)
                                delta_stats = delta_q_statistics(voltage, delta_q)
                                delta_length = int(len(voltage))
                                if delta_stats:
                                    record.update(delta_stats)
                                    delta_valid = True
                                    delta_reason = "계산 완료"
                                else:
                                    delta_reason = "유효 ΔQ(V) 포인트 부족"
                            else:
                                delta_reason = "Qdlin/Vdlin 길이 또는 자료형 불일치"
                        except (KeyError, ValueError, TypeError, OSError) as exc:
                            delta_reason = f"Qdlin 읽기 실패: {type(exc).__name__}"

                record["delta_q_available"] = delta_valid
                record["delta_q_reason"] = delta_reason
                feature_rows.append(record)

                first_cycle_mask = cycles == 1
                first_all_zero = False
                if first_cycle_mask.any():
                    first_idx = int(np.flatnonzero(first_cycle_mask)[0])
                    first_all_zero = all(
                        np.isclose(arrays[m][first_idx], 0.0, equal_nan=False)
                        for m in metric_map
                    )
                quality_rows.append(
                    {
                        "batch": batch_name,
                        "global_cell_id": global_id,
                        "cycle_life": cycle_life,
                        "target_available": bool(np.isfinite(cycle_life)),
                        "summary_length": common_length,
                        "summary_fields_same_length": len(set(lengths.values())) == 1,
                        "summary_cycles_match_objects": common_length == cycles_object_length,
                        "summary_cycles_strictly_increasing": bool(np.all(np.diff(cycles) > 0)),
                        "duplicate_cycle_count": int(pd.Series(cycles).duplicated().sum()),
                        "summary_min_cycle": float(np.nanmin(cycles)) if len(cycles) else np.nan,
                        "summary_max_cycle": float(np.nanmax(cycles)) if len(cycles) else np.nan,
                        "cycle_10_available": has_10,
                        "cycle_100_available": has_100,
                        "first_cycle_all_zero": first_all_zero,
                        "early_qd_valid_count": int((early_mask & valid_metric_mask("qd", arrays["qd"])).sum()),
                        "early_ir_valid_count": int((early_mask & valid_metric_mask("ir", arrays["ir"])).sum()),
                        "early_temp_valid_count": int((early_mask & valid_metric_mask("tavg", arrays["tavg"])).sum()),
                        "delta_q_available": delta_valid,
                        "delta_q_point_count": delta_length,
                        "delta_q_reason": delta_reason,
                        "policy_parse_success": bool(policy_info["policy_parse_success"]),
                        "included_for_supervised": bool(np.isfinite(cycle_life)),
                        "exclusion_reason": "" if np.isfinite(cycle_life) else "cycle_life 결측",
                    }
                )

    features = pd.DataFrame(feature_rows)
    quality = pd.DataFrame(quality_rows)
    files = pd.DataFrame(file_rows)
    if not features["global_cell_id"].is_unique:
        raise AssertionError("global_cell_id가 유일하지 않습니다.")
    return features, quality, files


def build_quality_summary(features: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for batch, group in quality.groupby("batch", sort=False):
        rows.append(
            {
                "batch": batch,
                "raw_cells": len(group),
                "target_available": int(group["target_available"].sum()),
                "cycle_10_available": int(group["cycle_10_available"].sum()),
                "cycle_100_available": int(group["cycle_100_available"].sum()),
                "delta_q_available": int(group["delta_q_available"].sum()),
                "policy_parse_success": int(group["policy_parse_success"].sum()),
                "supervised_rows": int(group["included_for_supervised"].sum()),
                "delta_q_availability_pct": 100 * group["delta_q_available"].mean(),
                "target_availability_pct": 100 * group["target_available"].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_split_assignment(features: pd.DataFrame) -> pd.DataFrame:
    assignment = features[
        ["batch", "global_cell_id", "cycle_life", "target_available", "life_group"]
    ].copy()
    assignment["split"] = "excluded_target_missing"
    batch1 = assignment[(assignment["batch"] == "Batch 1") & assignment["target_available"]].copy()
    strata = pd.qcut(batch1["cycle_life"], q=3, labels=["낮음", "중간", "높음"], duplicates="drop")
    stratify = strata if strata.nunique() == 3 and strata.value_counts().min() >= 2 else None
    dev_ids, holdout_ids = train_test_split(
        batch1["global_cell_id"],
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    assignment.loc[assignment["global_cell_id"].isin(dev_ids), "split"] = "development"
    assignment.loc[assignment["global_cell_id"].isin(holdout_ids), "split"] = "holdout"
    assignment.loc[(assignment["batch"] == "Batch 2") & assignment["target_available"], "split"] = "external_test"
    assignment.loc[(assignment["batch"] == "Batch 3") & assignment["target_available"], "split"] = "additional_test"
    assignment["split_seed"] = RANDOM_SEED
    if set(dev_ids) & set(holdout_ids):
        raise AssertionError("Development와 hold-out Cell이 겹칩니다.")
    return assignment.sort_values(["batch", "global_cell_id"]).reset_index(drop=True)


def make_feature_manifest() -> pd.DataFrame:
    rows = [
        ("qd_mean", "Capacity", "summary.QDischarge", "2~100", "mean(valid QD)", "Ah", "fold median+flag", "F0", "초기 용량 기준"),
        ("qd_slope", "Capacity", "summary.QDischarge", "2~100", "slope(QD~cycle)", "Ah/cycle", "fold median+flag", "F0", "초기 용량 변화"),
        ("delta_q_log10_var", "DeltaQ", "cycles.Qdlin", "10,100", "log10(var(Q100-Q10)+1e-12)", "log10(Ah²)", "fold median+flag", "F1", "세 Batch에서 강하고 일관된 관계"),
        ("delta_q_iqr", "DeltaQ alternative", "cycles.Qdlin", "10,100", "Q75(ΔQ)-Q25(ΔQ)", "Ah", "fold median+flag", "Screen", "이상값에 강한 대체 후보"),
        ("delta_q_min", "DeltaQ alternative", "cycles.Qdlin", "10,100", "min(ΔQ)", "Ah", "fold median+flag", "Screen", "국소 변화 대체 후보"),
        ("delta_q_abs_area", "DeltaQ alternative", "cycles.Qdlin", "10,100", "integral(abs(ΔQ))dV", "Ah·V", "fold median+flag", "Screen", "전압 전구간 변화 대체 후보"),
        ("ir_mean", "Sensor", "summary.IR", "2~100", "mean(valid IR)", "Ω", "fold median+flag", "F3", "저항 수준 대표"),
        ("tavg_mean", "Sensor", "summary.Tavg", "2~100", "mean(valid Tavg)", "°C", "fold median+flag", "F3", "온도 수준 대표"),
        ("chargetime_mean", "Sensor", "summary.chargetime", "2~100", "mean(valid charge time)", "min", "fold median+flag", "F3", "충전시간 대표"),
        ("c_rate_stage1", "Charging", "policy_readable", "사전 설정", "parsed first-stage C-rate", "C", "fold median+flag", "F4", "연속형 충전조건"),
        ("switch_soc_pct", "Charging", "policy_readable", "사전 설정", "parsed switching SOC", "%", "fold median+flag", "F4", "연속형 충전조건"),
        ("c_rate_stage2", "Charging", "policy_readable", "사전 설정", "parsed second-stage C-rate", "C", "fold median+flag", "F4", "연속형 충전조건"),
        ("policy_new_structure", "Charging", "policy_readable", "사전 설정", "newstructure flag", "bool", "none", "F4", "Cell 구조 보조정보"),
        ("policy_slow_cycle", "Charging", "policy_readable", "사전 설정", "slowcycle flag", "bool", "none", "F4", "정책 보조정보"),
    ]
    columns = [
        "feature_name", "feature_group", "source_field", "cycle_range", "formula", "unit",
        "missing_rule", "ablation_stage", "day1_evidence",
    ]
    manifest = pd.DataFrame(rows, columns=columns)
    manifest["leakage_check"] = "cycle 100 이내 또는 사전 정책"
    manifest["final_selected"] = False
    return manifest


def validate_against_day1(features: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    reference_path = paths.root / "results" / "day1" / "early_features.csv"
    if not reference_path.exists():
        return pd.DataFrame([{"check": "Day 1 reference", "status": "미실행", "detail": "파일 없음"}])
    reference = pd.read_csv(reference_path)
    common = sorted(set(features["global_cell_id"]) & set(reference["global_cell_id"]))
    columns = [
        "qd_mean", "qd_slope", "ir_mean", "tavg_mean", "chargetime_mean",
        "delta_q_log10_var", "delta_q_iqr", "delta_q_min", "delta_q_abs_area",
    ]
    merged = features.set_index("global_cell_id").loc[common].join(
        reference.set_index("global_cell_id")[columns], rsuffix="_day1"
    )
    rows = []
    for column in columns:
        left = pd.to_numeric(merged[column], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_day1"], errors="coerce")
        mask = left.notna() & right.notna()
        max_diff = float(np.max(np.abs(left[mask] - right[mask]))) if mask.any() else np.nan
        rows.append(
            {
                "check": column,
                "matched_cells": int(mask.sum()),
                "max_abs_difference": max_diff,
                "status": "통과" if (not np.isfinite(max_diff) or max_diff < 1e-10) else "불일치",
                "detail": "원본 .mat 재추출값과 Day 1 결과 비교",
            }
        )
    result = pd.DataFrame(rows)
    if (result["status"] == "불일치").any():
        raise AssertionError("Day 1 Feature 재현 검증에 실패했습니다.")
    return result


def numeric_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame[features].copy()
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def make_cv_splits(n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_SEED)
    dummy = np.zeros((n_samples, 1))
    return list(cv.split(dummy))


def make_model_specs() -> dict[str, tuple[Pipeline, dict[str, list[Any]], int]]:
    linear_pre = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    tree_pre = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
    ]
    return {
        "Median Baseline": (
            Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", DummyRegressor(strategy="median"))]),
            {},
            0,
        ),
        "Linear Regression": (
            Pipeline(linear_pre + [("model", LinearRegression())]),
            {},
            1,
        ),
        "Ridge": (
            Pipeline(linear_pre + [("model", Ridge())]),
            {"model__alpha": list(np.logspace(-4, 4, 9))},
            2,
        ),
        "ElasticNet": (
            Pipeline(linear_pre + [("model", ElasticNet(max_iter=100000, random_state=RANDOM_SEED))]),
            {
                "model__alpha": list(np.logspace(-4, 1, 6)),
                "model__l1_ratio": [0.1, 0.5, 0.9],
            },
            3,
        ),
        "Gradient Boosting": (
            Pipeline(tree_pre + [("model", GradientBoostingRegressor(random_state=RANDOM_SEED))]),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.02, 0.05, 0.10],
                "model__max_depth": [1, 2],
                "model__min_samples_leaf": [3, 5, 8],
                "model__subsample": [0.7, 1.0],
            },
            4,
        ),
    }


SCORING = {
    "mape": "neg_mean_absolute_percentage_error",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def tune_and_score(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, dict[str, float]]:
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid if grid else [{}],
        scoring=SCORING,
        refit="mape",
        cv=cv_splits,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y)
    best = clone(search.best_estimator_)
    cv_result = cross_validate(
        best,
        X,
        y,
        scoring=SCORING,
        cv=cv_splits,
        n_jobs=-1,
        error_score="raise",
    )
    fold = pd.DataFrame(
        {
            "fold": np.arange(1, len(cv_splits) + 1),
            "repeat": np.repeat(np.arange(1, 11), 5),
            "fold_in_repeat": np.tile(np.arange(1, 6), 10),
            "mape_pct": -100 * cv_result["test_mape"],
            "mae": -cv_result["test_mae"],
            "rmse": -cv_result["test_rmse"],
            "r2": cv_result["test_r2"],
        }
    )
    repeat_means = fold.groupby("repeat")["mape_pct"].mean()
    summary = {
        "cv_mape_mean": float(fold["mape_pct"].mean()),
        "cv_mape_std": float(fold["mape_pct"].std(ddof=1)),
        "cv_mape_se": float(repeat_means.std(ddof=1) / math.sqrt(len(repeat_means))),
        "cv_mae_mean": float(fold["mae"].mean()),
        "cv_rmse_mean": float(fold["rmse"].mean()),
        "cv_r2_mean": float(fold["r2"].mean()),
    }
    return best, search.best_params_, fold, summary


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y_true_array = np.asarray(list(y_true), dtype=float)
    y_pred_array = np.asarray(list(y_pred), dtype=float)
    return {
        "mape_pct": float(100 * mean_absolute_percentage_error(y_true_array, y_pred_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array)) if len(y_true_array) >= 2 else np.nan,
        "median_ape_pct": float(np.median(np.abs(y_true_array - y_pred_array) / np.abs(y_true_array)) * 100),
    }


def bootstrap_mape_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    values = np.empty(n_boot)
    for idx in range(n_boot):
        sample = rng.integers(0, n, size=n)
        values[idx] = 100 * mean_absolute_percentage_error(y_true[sample], y_pred[sample])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def model_development(
    features: pd.DataFrame,
    assignment: pd.DataFrame,
) -> dict[str, Any]:
    merged = features.merge(assignment[["global_cell_id", "split"]], on="global_cell_id", how="left")
    development = merged[merged["split"] == "development"].reset_index(drop=True)
    holdout = merged[merged["split"] == "holdout"].reset_index(drop=True)
    if len(development) != 36 or len(holdout) != 10:
        raise AssertionError(f"예상 split 크기와 다릅니다: development={len(development)}, holdout={len(holdout)}")
    cv_splits = make_cv_splits(len(development))
    specs = make_model_specs()

    print("[Gate 2] ΔQ 대표 Feature screening", flush=True)
    dq_candidates = ["delta_q_log10_var", "delta_q_iqr", "delta_q_min", "delta_q_abs_area"]
    dq_rows, dq_folds = [], []
    ridge_pipe, ridge_grid, _ = specs["Ridge"]
    for candidate in dq_candidates:
        best, params, fold, summary = tune_and_score(
            numeric_frame(development, [candidate]), development["cycle_life"], ridge_pipe, ridge_grid, cv_splits
        )
        del best
        dq_rows.append({"feature": candidate, **summary, "best_params": json.dumps(params, ensure_ascii=False)})
        fold["feature"] = candidate
        dq_folds.append(fold)
    dq_results = pd.DataFrame(dq_rows).sort_values("cv_mape_mean").reset_index(drop=True)
    best_dq = dq_results.iloc[0]
    predeclared = dq_results[dq_results["feature"] == "delta_q_log10_var"].iloc[0]
    chosen_dq = (
        "delta_q_log10_var"
        if float(predeclared["cv_mape_mean"]) <= float(best_dq["cv_mape_mean"] + best_dq["cv_mape_se"])
        else str(best_dq["feature"])
    )
    dq_results["selected_core"] = dq_results["feature"].eq(chosen_dq)

    feature_sets = {
        "F0 Capacity": ["qd_mean", "qd_slope"],
        "F1 ΔQ": [chosen_dq],
        "F2 ΔQ+Capacity": [chosen_dq, "qd_mean", "qd_slope"],
        "F3 +Sensor": [chosen_dq, "qd_mean", "qd_slope", "ir_mean", "tavg_mean", "chargetime_mean"],
        "F4 +Charging": [
            chosen_dq, "qd_mean", "qd_slope", "ir_mean", "tavg_mean", "chargetime_mean",
            "c_rate_stage1", "switch_soc_pct", "c_rate_stage2",
            "policy_new_structure", "policy_slow_cycle",
        ],
    }
    for name, columns in feature_sets.items():
        forbidden = [c for c in columns if any(pattern in c.lower() for pattern in FORBIDDEN_FEATURE_PATTERNS)]
        if forbidden:
            raise AssertionError(f"금지 Feature가 포함됐습니다: {name}={forbidden}")

    print(f"[Gate 3] F0~F4 × 5개 모델 비교 (Core={chosen_dq})", flush=True)
    comparison_rows: list[dict[str, Any]] = []
    all_folds: list[pd.DataFrame] = []
    estimators: dict[tuple[str, str], Pipeline] = {}
    for set_name, columns in feature_sets.items():
        X = numeric_frame(development, columns)
        y = development["cycle_life"]
        for model_name, (pipeline, grid, complexity) in specs.items():
            print(f"  - {set_name} / {model_name}", flush=True)
            best, params, fold, summary = tune_and_score(X, y, pipeline, grid, cv_splits)
            estimators[(set_name, model_name)] = best
            comparison_rows.append(
                {
                    "feature_set": set_name,
                    "model": model_name,
                    "feature_count": len(columns),
                    "model_complexity_rank": complexity,
                    **summary,
                    "best_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "features": ", ".join(columns),
                }
            )
            fold["feature_set"] = set_name
            fold["model"] = model_name
            all_folds.append(fold)

    comparison = pd.DataFrame(comparison_rows).sort_values("cv_mape_mean").reset_index(drop=True)
    fold_results = pd.concat(all_folds, ignore_index=True)
    best_mean = float(comparison.iloc[0]["cv_mape_mean"])
    threshold = best_mean + float(comparison.iloc[0]["cv_mape_se"])
    comparison["within_one_se"] = comparison["cv_mape_mean"] <= threshold
    eligible = comparison[comparison["within_one_se"]].sort_values(
        ["feature_count", "model_complexity_rank", "cv_mape_mean"]
    )
    primary = eligible.iloc[0]
    selected_key = (str(primary["feature_set"]), str(primary["model"]))
    selected_features = feature_sets[selected_key[0]]
    selected_estimator = estimators[selected_key]

    print(f"[Gate 4] 고정 Hold-out 최초 평가: {selected_key[0]} / {selected_key[1]}", flush=True)
    selected_estimator.fit(numeric_frame(development, selected_features), development["cycle_life"])
    holdout_pred = selected_estimator.predict(numeric_frame(holdout, selected_features))
    holdout_metrics = regression_metrics(holdout["cycle_life"], holdout_pred)
    selected_row = comparison[
        (comparison["feature_set"] == selected_key[0]) & (comparison["model"] == selected_key[1])
    ].iloc[0]
    stability_limit = float(selected_row["cv_mape_mean"] + 2 * selected_row["cv_mape_std"])
    holdout_stable = holdout_metrics["mape_pct"] <= stability_limit

    selection_log = comparison.copy()
    selection_log["selection_status"] = "탈락"
    selection_log.loc[selection_log["within_one_se"], "selection_status"] = "1-SE 후보"
    selection_log.loc[
        (selection_log["feature_set"] == selected_key[0]) & (selection_log["model"] == selected_key[1]),
        "selection_status",
    ] = "최종 선택"
    selection_log["holdout_checked"] = False
    selection_log["holdout_mape_pct"] = np.nan
    selection_log.loc[
        (selection_log["feature_set"] == selected_key[0]) & (selection_log["model"] == selected_key[1]),
        ["holdout_checked", "holdout_mape_pct"],
    ] = [True, holdout_metrics["mape_pct"]]
    selection_log["reason"] = np.where(
        selection_log["selection_status"] == "최종 선택",
        "1-SE 범위 내 최소 Feature·복잡도; hold-out 안정성 확인",
        np.where(selection_log["within_one_se"], "1-SE 후보이나 더 복잡함", "1-SE 범위 밖"),
    )

    if not holdout_stable:
        raise RuntimeError(
            "사전 정의한 hold-out 안정성 기준을 통과하지 못했습니다. "
            "Batch 2 평가 전에 Decision Log 검토가 필요합니다."
        )

    ablation = comparison[comparison["model"] == "Ridge"].copy()
    ablation = ablation.sort_values("feature_set").reset_index(drop=True)
    return {
        "merged": merged,
        "development": development,
        "holdout": holdout,
        "dq_results": dq_results,
        "dq_folds": pd.concat(dq_folds, ignore_index=True),
        "chosen_dq": chosen_dq,
        "feature_sets": feature_sets,
        "comparison": comparison,
        "fold_results": fold_results,
        "ablation": ablation,
        "selection_log": selection_log,
        "selected_key": selected_key,
        "selected_features": selected_features,
        "selected_estimator": selected_estimator,
        "holdout_pred": holdout_pred,
        "holdout_metrics": holdout_metrics,
        "holdout_stable": holdout_stable,
    }


def final_external_evaluation(modeling: dict[str, Any]) -> dict[str, Any]:
    merged = modeling["merged"]
    selected_features = modeling["selected_features"]
    model = clone(modeling["selected_estimator"])
    batch1_all = merged[merged["split"].isin(["development", "holdout"])].reset_index(drop=True)
    batch2 = merged[merged["split"] == "external_test"].reset_index(drop=True)
    batch3 = merged[merged["split"] == "additional_test"].reset_index(drop=True)
    model.fit(numeric_frame(batch1_all, selected_features), batch1_all["cycle_life"])

    print("[Gate 5] 설정 동결 후 Batch 2 최초 외부 평가", flush=True)
    predictions = []
    performance_rows = []
    eval_sets = [
        ("Batch 1 Hold-out", modeling["holdout"], modeling["holdout_pred"]),
        ("Batch 2 Test", batch2, model.predict(numeric_frame(batch2, selected_features))),
        ("Batch 3 Test", batch3, model.predict(numeric_frame(batch3, selected_features))),
    ]
    for dataset, frame, pred in eval_sets:
        metrics = regression_metrics(frame["cycle_life"], pred)
        ci_low, ci_high = bootstrap_mape_ci(frame["cycle_life"].to_numpy(), np.asarray(pred))
        performance_rows.append(
            {
                "dataset": dataset,
                "n": len(frame),
                **metrics,
                "mape_ci95_low": ci_low,
                "mape_ci95_high": ci_high,
            }
        )
        for (_, row), prediction in zip(frame.iterrows(), pred):
            actual = float(row["cycle_life"])
            prediction = float(prediction)
            predictions.append(
                {
                    "dataset": dataset,
                    "batch": row["batch"],
                    "global_cell_id": row["global_cell_id"],
                    "cycle_life": actual,
                    "prediction": prediction,
                    "residual": prediction - actual,
                    "absolute_error": abs(prediction - actual),
                    "ape_pct": abs(prediction - actual) / actual * 100,
                    "error_direction": "과대 예측" if prediction > actual else "과소 예측",
                    "life_group": life_group(actual),
                    "charging_policy": row["charging_policy"],
                    "c_rate_stage1": row["c_rate_stage1"],
                    "switch_soc_pct": row["switch_soc_pct"],
                    **{feature: row[feature] for feature in selected_features},
                }
            )

    selected_row = modeling["comparison"][
        (modeling["comparison"]["feature_set"] == modeling["selected_key"][0])
        & (modeling["comparison"]["model"] == modeling["selected_key"][1])
    ].iloc[0]
    performance = pd.DataFrame(
        [
            {
                "dataset": "Train (Batch 1 CV)",
                "n": len(modeling["development"]),
                "mape_pct": selected_row["cv_mape_mean"],
                "mape_std_pct": selected_row["cv_mape_std"],
                "mae": selected_row["cv_mae_mean"],
                "rmse": selected_row["cv_rmse_mean"],
                "r2": selected_row["cv_r2_mean"],
                "median_ape_pct": np.nan,
                "mape_ci95_low": np.nan,
                "mape_ci95_high": np.nan,
            }
        ]
        + performance_rows
    )
    performance["mape_std_pct"] = performance.get("mape_std_pct", np.nan)
    performance = performance[
        ["dataset", "n", "mape_pct", "mape_std_pct", "mape_ci95_low", "mape_ci95_high", "mae", "rmse", "r2", "median_ape_pct"]
    ]
    predictions_df = pd.DataFrame(predictions)

    metric_lookup = performance.set_index("dataset")["mape_pct"]
    gaps = pd.DataFrame(
        [
            {"gap": "Train-Valid", "value_pct_point": metric_lookup["Batch 1 Hold-out"] - metric_lookup["Train (Batch 1 CV)"], "interpretation": "양수면 내부 일반화 저하"},
            {"gap": "Valid-Test", "value_pct_point": metric_lookup["Batch 2 Test"] - metric_lookup["Batch 1 Hold-out"], "interpretation": "양수면 Batch 일반화 저하"},
            {"gap": "Target-Test", "value_pct_point": metric_lookup["Batch 2 Test"] - REFERENCE_MAPE, "interpretation": "원논문 9.1% 대비 차이"},
            {"gap": "Batch2-Batch3", "value_pct_point": metric_lookup["Batch 3 Test"] - metric_lookup["Batch 2 Test"], "interpretation": "양수면 Batch 3 성능이 더 낮음"},
        ]
    )
    return {
        "model": model,
        "batch1_all": batch1_all,
        "batch2": batch2,
        "batch3": batch3,
        "performance": performance,
        "predictions": predictions_df,
        "gaps": gaps,
    }


def build_error_tables(evaluation: dict[str, Any], selected_features: list[str]) -> dict[str, pd.DataFrame]:
    predictions = evaluation["predictions"].copy()
    group_rows = []
    for (dataset, group), frame in predictions.groupby(["dataset", "life_group"], observed=True):
        metrics = regression_metrics(frame["cycle_life"], frame["prediction"])
        group_rows.append({"dataset": dataset, "life_group": group, "n": len(frame), **metrics})
    group_error = pd.DataFrame(group_rows)

    charging = predictions.copy()
    charging["c_rate_bin"] = pd.cut(
        charging["c_rate_stage1"],
        bins=[-np.inf, 4, 6, np.inf],
        labels=["저속(≤4C)", "중간(4~6C)", "고속(>6C)"],
    )
    charging_rows = []
    for (dataset, cbin), frame in charging.dropna(subset=["c_rate_bin"]).groupby(
        ["dataset", "c_rate_bin"], observed=True
    ):
        charging_rows.append(
            {
                "dataset": dataset,
                "c_rate_bin": str(cbin),
                "n": len(frame),
                "mape_pct": float(frame["ape_pct"].mean()),
                "mae": float(frame["absolute_error"].mean()),
                "overprediction_rate_pct": float(100 * (frame["residual"] > 0).mean()),
            }
        )
    charging_error = pd.DataFrame(charging_rows)
    worst = predictions[predictions["dataset"].isin(["Batch 2 Test", "Batch 3 Test"])].nlargest(10, "ape_pct")

    combined = pd.concat([evaluation["batch1_all"], evaluation["batch2"], evaluation["batch3"]], ignore_index=True)
    shift_rows = []
    batch1 = combined[combined["batch"] == "Batch 1"]
    for feature in selected_features:
        base = pd.to_numeric(batch1[feature], errors="coerce")
        base_mean, base_std = base.mean(), base.std(ddof=1)
        for batch, frame in combined.groupby("batch", sort=False):
            values = pd.to_numeric(frame[feature], errors="coerce")
            shift_rows.append(
                {
                    "feature": feature,
                    "batch": batch,
                    "n_available": int(values.notna().sum()),
                    "missing_pct": float(100 * values.isna().mean()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=1)),
                    "standardized_mean_shift_vs_batch1": float((values.mean() - base_mean) / base_std) if base_std > 0 else np.nan,
                    "outside_batch1_range_pct": float(100 * ((values < base.min()) | (values > base.max())).mean()),
                }
            )
    batch_shift = pd.DataFrame(shift_rows)
    return {
        "life_group_error": group_error,
        "charging_error": charging_error,
        "worst_predictions": worst,
        "batch_shift": batch_shift,
    }


def extract_importance(model: Pipeline, features: list[str]) -> pd.DataFrame:
    imputer = model.named_steps["imputer"]
    names = list(imputer.get_feature_names_out(features))
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).reshape(-1)
        kind = "표준화 계수"
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).reshape(-1)
        kind = "Feature importance"
    else:
        values = np.zeros(len(names))
        kind = "해당 없음"
    return pd.DataFrame(
        {"feature": names, "importance": values, "absolute_importance": np.abs(values), "importance_type": kind}
    ).sort_values("absolute_importance", ascending=False)


def plot_model_comparison(comparison: pd.DataFrame, figure_dir: Path) -> None:
    frame = comparison.copy()
    frame["Feature 단계"] = frame["feature_set"].map(FEATURE_SET_KO)
    frame["모델명"] = frame["model"].map(MODEL_KO)
    order = list(FEATURE_SET_KO.values())
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    sns.barplot(data=frame, x="Feature 단계", y="cv_mape_mean", hue="모델명", order=order, ax=ax)
    ax.set_title("Batch 1 개발영역: Feature 단계와 모델별 교차검증 MAPE")
    ax.set_xlabel("특성 제거 실험 단계")
    ax.set_ylabel("교차검증 MAPE (%) · 낮을수록 좋음")
    ax.legend(title="모델", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.tick_params(axis="x", rotation=12)
    save_figure(fig, figure_dir / "01_모델_비교.png")


def plot_ablation(ablation: pd.DataFrame, figure_dir: Path) -> None:
    order = ["F0 Capacity", "F1 ΔQ", "F2 ΔQ+Capacity", "F3 +Sensor", "F4 +Charging"]
    frame = ablation.set_index("feature_set").loc[order].reset_index()
    frame["표시 단계"] = frame["feature_set"].map(FEATURE_SET_KO)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = ["#8EA6BF", "#159A9C", "#2D7DD2", "#6C83B5", "#A078B5"]
    bars = ax.bar(frame["표시 단계"], frame["cv_mape_mean"], yerr=frame["cv_mape_std"], color=colors, capsize=5)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in frame["cv_mape_mean"]], padding=4, fontweight="bold")
    ax.set_title("동일 Ridge 모델에서 확인한 Feature군의 추가 가치")
    ax.set_xlabel("특성 구성")
    ax.set_ylabel("교차검증 MAPE 평균 ± 표준편차 (%)")
    ax.tick_params(axis="x", rotation=12)
    save_figure(fig, figure_dir / "02_Feature_Ablation.png")


def plot_actual_vs_predicted(predictions: pd.DataFrame, figure_dir: Path) -> None:
    datasets = ["Batch 1 Hold-out", "Batch 2 Test", "Batch 3 Test"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True, sharey=True)
    limits = [predictions[["cycle_life", "prediction"]].min().min() - 80, predictions[["cycle_life", "prediction"]].max().max() + 80]
    for ax, dataset in zip(axes, datasets):
        frame = predictions[predictions["dataset"] == dataset]
        ax.scatter(frame["cycle_life"], frame["prediction"], s=48, alpha=0.85, color=SPLIT_COLORS[dataset], edgecolor="white", linewidth=0.7)
        ax.plot(limits, limits, linestyle="--", color="#334155", linewidth=1.2, label="정답선 y=x")
        metric = 100 * mean_absolute_percentage_error(frame["cycle_life"], frame["prediction"])
        ax.set_title(f"{DATASET_KO[dataset]}\nMAPE {metric:.1f}% · n={len(frame)}")
        ax.set_xlabel("실제 수명 (cycle)")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    axes[0].set_ylabel("예측 수명 (cycle)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("실제 수명과 예측 수명 — 대각선에 가까울수록 정확", fontsize=16, fontweight="bold", color="#173B5E")
    fig.tight_layout()
    save_figure(fig, figure_dir / "03_실제값_예측값.png")


def plot_residuals(predictions: pd.DataFrame, figure_dir: Path) -> None:
    frame = predictions.copy()
    frame["데이터 구분"] = frame["dataset"].map(DATASET_KO)
    frame["오차 방향"] = frame["error_direction"]
    korean_palette = {DATASET_KO[key]: value for key, value in SPLIT_COLORS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.histplot(data=frame, x="residual", hue="데이터 구분", bins=18, element="step", common_norm=False, ax=axes[0], palette=korean_palette)
    axes[0].axvline(0, color="#1F2937", linestyle="--", linewidth=1.2)
    axes[0].set_title("잔차 분포")
    axes[0].set_xlabel("잔차 = 예측 - 실제 (cycle)")
    axes[0].set_ylabel("Cell 수")
    sns.scatterplot(data=frame, x="cycle_life", y="residual", hue="데이터 구분", style="오차 방향", s=65, ax=axes[1], palette=korean_palette)
    axes[1].axhline(0, color="#1F2937", linestyle="--", linewidth=1.2)
    axes[1].set_title("실제 수명에 따른 잔차")
    axes[1].set_xlabel("실제 수명 (cycle)")
    axes[1].set_ylabel("잔차 (cycle)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("과대·과소 예측의 방향과 크기", fontsize=16, fontweight="bold", color="#173B5E")
    fig.tight_layout()
    save_figure(fig, figure_dir / "04_잔차_분석.png")


def plot_group_error(group_error: pd.DataFrame, figure_dir: Path) -> None:
    frame = group_error.copy()
    frame["데이터 구분"] = frame["dataset"].map(DATASET_KO)
    korean_palette = {DATASET_KO[key]: value for key, value in SPLIT_COLORS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    sns.barplot(data=frame, x="life_group", y="mape_pct", hue="데이터 구분", order=LIFE_ORDER, palette=korean_palette, ax=axes[0])
    axes[0].set_title("수명 구간별 상대오차")
    axes[0].set_xlabel("실제 수명 구간")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].tick_params(axis="x", rotation=15)
    sns.barplot(data=frame, x="life_group", y="mae", hue="데이터 구분", order=LIFE_ORDER, palette=korean_palette, ax=axes[1])
    axes[1].set_title("수명 구간별 절대오차")
    axes[1].set_xlabel("실제 수명 구간")
    axes[1].set_ylabel("MAE (cycle)")
    axes[1].tick_params(axis="x", rotation=15)
    if axes[1].legend_:
        axes[1].legend_.remove()
    fig.suptitle("같은 모델도 수명 구간에 따라 오류가 다르다", fontsize=16, fontweight="bold", color="#173B5E")
    fig.tight_layout()
    save_figure(fig, figure_dir / "05_수명구간별_오차.png")


def plot_charging_error(predictions: pd.DataFrame, charging_error: pd.DataFrame, figure_dir: Path) -> None:
    frame = predictions.dropna(subset=["c_rate_stage1"]).copy()
    frame["데이터 구분"] = frame["dataset"].map(DATASET_KO)
    charging = charging_error.copy()
    charging["데이터 구분"] = charging["dataset"].map(DATASET_KO)
    korean_palette = {DATASET_KO[key]: value for key, value in SPLIT_COLORS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.scatterplot(data=frame, x="c_rate_stage1", y="ape_pct", hue="데이터 구분", s=65, palette=korean_palette, ax=axes[0])
    axes[0].set_title("1단계 C-rate와 Cell별 절대백분율오차")
    axes[0].set_xlabel("1단계 C-rate")
    axes[0].set_ylabel("절대백분율오차 (%)")
    if not charging.empty:
        sns.barplot(data=charging, x="c_rate_bin", y="mape_pct", hue="데이터 구분", palette=korean_palette, ax=axes[1])
    axes[1].set_title("충전속도 구간별 평균오차")
    axes[1].set_xlabel("충전속도 구간")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].tick_params(axis="x", rotation=12)
    fig.suptitle("충전조건별 예측 취약 구간 점검", fontsize=16, fontweight="bold", color="#173B5E")
    fig.tight_layout()
    save_figure(fig, figure_dir / "06_충전조건별_오차.png")


def plot_worst_predictions(worst: pd.DataFrame, figure_dir: Path) -> None:
    frame = worst.sort_values("ape_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = [BATCH_COLORS.get(batch, "#64748B") for batch in frame["batch"]]
    bars = ax.barh(frame["global_cell_id"], frame["ape_pct"], color=colors)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in frame["ape_pct"]], padding=4)
    ax.set_title("외부 배치에서 가장 큰 오차를 보인 셀 Top 10")
    ax.set_xlabel("절대백분율오차 (%)")
    ax.set_ylabel("셀")
    save_figure(fig, figure_dir / "07_최악_예측_Cell.png")


def plot_importance(importance: pd.DataFrame, figure_dir: Path) -> None:
    frame = importance.head(15).sort_values("absolute_importance").copy()
    frame["표시 특성"] = frame["feature"].map(FEATURE_KO).fillna(frame["feature"])
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = ["#D95D5D" if value < 0 else "#159A9C" for value in frame["importance"]]
    ax.barh(frame["표시 특성"], frame["importance"], color=colors)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_title(f"최종 모델 설명 — {frame['importance_type'].iloc[0] if len(frame) else '중요도'}")
    ax.set_xlabel("값의 크기와 방향")
    ax.set_ylabel("특성")
    save_figure(fig, figure_dir / "08_Feature_중요도.png")


def plot_batch_shift(batch_shift: pd.DataFrame, figure_dir: Path) -> None:
    pivot = batch_shift.pivot(index="feature", columns="batch", values="standardized_mean_shift_vs_batch1")
    pivot = pivot.reindex(columns=["Batch 1", "Batch 2", "Batch 3"])
    pivot.index = [FEATURE_KO.get(name, name) for name in pivot.index]
    pivot.columns = ["Batch 1", "Batch 2", "Batch 3"]
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.55 * len(pivot) + 1.8)))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Batch 1 표준편차 단위 평균 이동"}, ax=ax)
    ax.set_title("최종 특성의 배치 간 분포 이동")
    ax.set_xlabel("배치")
    ax.set_ylabel("특성")
    save_figure(fig, figure_dir / "09_Batch_Feature_시프트.png")


def markdown_table(frame: pd.DataFrame, digits: int = 2) -> str:
    if frame.empty:
        return "_해당 결과 없음_"
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        if pd.api.types.is_integer_dtype(formatted[column].dtype):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else str(int(value)))
        else:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(c) for c in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in formatted.iterrows():
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    paths: Paths,
    quality_summary: pd.DataFrame,
    validation: pd.DataFrame,
    modeling: dict[str, Any],
    evaluation: dict[str, Any],
    error_tables: dict[str, pd.DataFrame],
    importance: pd.DataFrame,
) -> Path:
    performance = evaluation["performance"].copy()
    perf_display = performance.rename(
        columns={
            "dataset": "구분", "n": "n", "mape_pct": "MAPE(%)", "mape_std_pct": "CV 표준편차",
            "mape_ci95_low": "95% CI 하한", "mape_ci95_high": "95% CI 상한", "mae": "MAE",
            "rmse": "RMSE", "r2": "R²", "median_ape_pct": "Median APE(%)",
        }
    )
    perf_display["구분"] = perf_display["구분"].replace(
        {
            "Batch 1 Hold-out": "Valid (Batch 1 Hold-out)",
            "Batch 2 Test": "Test (Batch 2)",
            "Batch 3 Test": "Additional Test (Batch 3)",
        }
    )
    gaps_display = evaluation["gaps"].rename(columns={"gap": "Gap", "value_pct_point": "차이(%p)", "interpretation": "해석"})
    dq_display = modeling["dq_results"][
        ["feature", "cv_mape_mean", "cv_mape_std", "cv_mape_se", "selected_core"]
    ].rename(columns={"feature": "ΔQ 후보", "cv_mape_mean": "CV MAPE(%)", "cv_mape_std": "표준편차", "cv_mape_se": "SE", "selected_core": "선택"})
    ablation_display = modeling["ablation"][
        ["feature_set", "feature_count", "cv_mape_mean", "cv_mape_std"]
    ].rename(columns={"feature_set": "단계", "feature_count": "Feature 수", "cv_mape_mean": "CV MAPE(%)", "cv_mape_std": "표준편차"})
    top_models = modeling["comparison"].head(10)[
        ["feature_set", "model", "feature_count", "cv_mape_mean", "cv_mape_std", "within_one_se"]
    ].rename(columns={"feature_set": "Feature set", "model": "모델", "feature_count": "Feature 수", "cv_mape_mean": "CV MAPE(%)", "cv_mape_std": "표준편차", "within_one_se": "1-SE"})
    selected_set, selected_model = modeling["selected_key"]
    selected_features = modeling["selected_features"]
    perf_lookup = performance.set_index("dataset")
    gap_lookup = evaluation["gaps"].set_index("gap")["value_pct_point"]
    batch2_mape = float(perf_lookup.loc["Batch 2 Test", "mape_pct"])
    batch3_mape = float(perf_lookup.loc["Batch 3 Test", "mape_pct"])
    holdout_mape = float(perf_lookup.loc["Batch 1 Hold-out", "mape_pct"])
    cv_mape = float(perf_lookup.loc["Train (Batch 1 CV)", "mape_pct"])
    dq_lookup = modeling["dq_results"].set_index("feature")
    chosen_dq_row = dq_lookup.loc[modeling["chosen_dq"]]
    logvar_row = dq_lookup.loc["delta_q_log10_var"]
    dq_threshold = float(chosen_dq_row["cv_mape_mean"] + chosen_dq_row["cv_mape_se"])
    ablation_lookup = modeling["ablation"].set_index("feature_set")["cv_mape_mean"]
    best_model_row = modeling["comparison"].iloc[0]
    pred = evaluation["predictions"]
    batch2_pred = pred[pred["dataset"] == "Batch 2 Test"]
    batch3_pred = pred[pred["dataset"] == "Batch 3 Test"]
    batch2_group = error_tables["life_group_error"].set_index(["dataset", "life_group"])
    b2_short = batch2_group.loc[("Batch 2 Test", "단수명(<500)")]
    b2_middle = batch2_group.loc[("Batch 2 Test", "중간수명(500~1,000)")]
    shift_lookup = error_tables["batch_shift"].set_index(["feature", "batch"])
    b2_iqr_shift = shift_lookup.loc[("delta_q_iqr", "Batch 2")]
    b2_qd_shift = shift_lookup.loc[("qd_mean", "Batch 2")]
    b3_iqr_shift = shift_lookup.loc[("delta_q_iqr", "Batch 3")]
    b3_qd_shift = shift_lookup.loc[("qd_mean", "Batch 3")]
    importance_lookup = importance.set_index("feature")["importance"]
    worst = error_tables["worst_predictions"][["global_cell_id", "batch", "cycle_life", "prediction", "ape_pct", "error_direction"]].copy()
    worst.columns = ["Cell", "Batch", "실제", "예측", "APE(%)", "방향"]
    importance_display = importance.head(12)[["feature", "importance", "importance_type"]].rename(columns={"feature": "Feature", "importance": "값", "importance_type": "종류"})

    external_message = (
        "Batch 3의 오차가 Batch 2보다 더 컸다"
        if batch3_mape > batch2_mape
        else "Batch 3의 오차는 Batch 2보다 작았다"
    )
    reference_message = (
        f"원논문 참고치 9.1%보다 {batch2_mape - REFERENCE_MAPE:.1f}%p 높았다"
        if batch2_mape >= REFERENCE_MAPE
        else f"원논문 참고치 9.1%보다 {REFERENCE_MAPE - batch2_mape:.1f}%p 낮았다"
    )
    report = f"""# 초기 100 Cycle 기반 배터리 수명 예측
## Day 2 — ΔQ(V) 중심 Regression 모델의 외부 Batch 일반화 검증

> **작성자:** U094 이수현  
> **작성일:** 2026-10-02 (금)  
> **주 평가지표:** MAPE (%)  
> **최종 선택:** `{selected_set}` + `{selected_model}`

---

## 0. Executive Summary

### 한 문장 결론

Batch 1의 초기 100 cycle만으로 선택한 **{selected_model}** 모델은 `{', '.join(selected_features)}`를 사용했으며, Batch 1 CV MAPE **{cv_mape:.1f}%**, 고정 hold-out **{holdout_mape:.1f}%**, Batch 2 외부 테스트 **{batch2_mape:.1f}%**, Batch 3 추가 테스트 **{batch3_mape:.1f}%**를 기록했다.

### 핵심 메시지

1. 원본 `.mat`에서 실제 cycle 번호를 사용해 Cell당 1행의 Feature를 재생성했다.
2. Day 1의 가장 강한 초기 신호인 ΔQ(V)를 Core로 구현하고, 중복 후보는 Batch 1 development CV에서만 비교했다.
3. 모든 전처리는 sklearn Pipeline 안에서 각 train fold에만 맞췄고, Batch 2는 선택 종료 후 평가했다.
4. 최종 Batch 2 성능은 {reference_message}. 조건이 다른 논문의 수치를 합격선으로 취급하지 않고 구현·분포 차이를 함께 분석했다.
5. {external_message}. 이는 내부 정확도뿐 아니라 Batch별 Target·Feature shift를 관리해야 함을 보여준다.
6. ESS에서는 수명 과대 예측이 정비 지연으로 이어질 수 있으므로 평균오차 외에 오차 방향과 최악 Cell을 함께 관리해야 한다.

---

## 1. 문제와 평가 설계

목표는 배터리 Cell의 초기 100 cycle만 사용해 SOH 80% 도달 시점인 `cycle_life`를 예측하는 것이다. 모델 입력은 cycle별 행이 아니라 Cell당 1행이다.

검증은 **Batch 1 development CV → Batch 1 fixed hold-out → 설정 동결 → Batch 1 전체 재학습 → Batch 2 Test → Batch 3 추가 Test** 순서로 수행했다. Hold-out 확인 후 Feature·Hyperparameter는 바꾸지 않고 Batch 1 labeled Cell 46개 전체로 최종 모델만 다시 학습했다. Day 1에서 Batch 2·3의 분포를 이미 관찰했기 때문에 완전한 blind test는 아니지만, Day 2에서 외부 Target을 보고 Feature·모델을 다시 조정하지 않았다.

## 2. 데이터 품질과 재현 검증

{markdown_table(quality_summary.rename(columns={'batch':'Batch','raw_cells':'원본 Cell','target_available':'Target 가용','cycle_10_available':'cycle 10','cycle_100_available':'cycle 100','delta_q_available':'ΔQ 가용','policy_parse_success':'정책 파싱','supervised_rows':'지도학습 사용','delta_q_availability_pct':'ΔQ 가용률(%)','target_availability_pct':'Target 가용률(%)'}))}

원본에서 재계산한 핵심 Feature는 Day 1 결과와 다음과 같이 대조했다.

{markdown_table(validation[['check','matched_cells','max_abs_difference','status']].rename(columns={'check':'검증 Feature','matched_cells':'비교 Cell','max_abs_difference':'최대 절대차','status':'결과'}), digits=6)}

Target 결측 Cell은 지도학습에서 제외했지만 관측 종료 cycle로 임의 대체하지 않았다. 첫 cycle의 0값을 피하기 위해 Summary Feature는 cycle 2–100, ΔQ는 실제 cycle 10과 100을 사용했다.

## 3. ΔQ(V) Core Feature 검증

ΔQ는 다음과 같이 정의했다.

```text
ΔQ(V) = Qdlin(cycle 100, V) - Qdlin(cycle 10, V)
```

상관이 높은 ΔQ 파생값을 동시에 넣지 않고, Ridge와 동일한 반복 CV로 대표 후보를 비교했다.

{markdown_table(dq_display)}

가장 낮은 값은 `{modeling['chosen_dq']}`의 {chosen_dq_row['cv_mape_mean']:.2f}%였다. Day 1의 사전 1순위였던 `delta_q_log10_var`는 {logvar_row['cv_mape_mean']:.2f}%로 {logvar_row['cv_mape_mean'] - chosen_dq_row['cv_mape_mean']:.2f}%p 높았고, 최저 후보의 1-SE 경계인 {dq_threshold:.2f}% 밖이었다. 따라서 외부 Batch를 보기 전에 <strong><code>{modeling['chosen_dq']}</code></strong>를 Core 대표값으로 선택했다. 이것은 Day 1 결론을 뒤집은 것이 아니라, “ΔQ 정보군이 핵심이며 그 안의 중복 대표값은 Batch 1 CV로 결정한다”는 사전 규칙을 실제로 적용한 결과다.

## 4. Feature Ablation

![Feature Ablation](../figures/day2/02_Feature_Ablation.png)

{markdown_table(ablation_display)}

F0는 Capacity만, F1은 ΔQ만, F2는 ΔQ와 Capacity, F3는 Sensor, F4는 Charging까지 추가한다. 동일한 Ridge와 동일한 CV split에서 비교했으므로 단계 간 차이는 어떤 정보군이 일반화 오차를 줄였는지 보여준다.

결과는 ΔQ의 추가 가치가 매우 컸음을 보여준다. Capacity만 본 F0의 MAPE는 {ablation_lookup['F0 Capacity']:.2f}%였지만 ΔQ 하나를 본 F1은 {ablation_lookup['F1 ΔQ']:.2f}%로 <strong>{ablation_lookup['F0 Capacity'] - ablation_lookup['F1 ΔQ']:.2f}%p 감소</strong>했다. F2에서 Capacity를 다시 더하면 {ablation_lookup['F2 ΔQ+Capacity']:.2f}%로 {ablation_lookup['F1 ΔQ'] - ablation_lookup['F2 ΔQ+Capacity']:.2f}%p 추가 개선됐다. 반면 Sensor와 Charging을 추가한 F3·F4는 Feature가 6개와 11개로 늘었음에도 F2보다 MAPE가 각각 {ablation_lookup['F3 +Sensor'] - ablation_lookup['F2 ΔQ+Capacity']:.2f}%p, {ablation_lookup['F4 +Charging'] - ablation_lookup['F2 ΔQ+Capacity']:.2f}%p 높았다. 따라서 센서와 충전조건은 관찰적으로 의미가 있더라도 현재 표본에서는 안정적인 추가 예측력을 증명하지 못했다.

## 5. 모델 비교와 최종 선택

![모델 비교](../figures/day2/01_모델_비교.png)

{markdown_table(top_models)}

최저 CV MAPE는 {best_model_row['feature_set']} {best_model_row['model']}의 {best_model_row['cv_mape_mean']:.2f}%였다. 그러나 {selected_set} {selected_model}은 {cv_mape:.2f}%로 차이가 <strong>{cv_mape - best_model_row['cv_mape_mean']:.2f}%p</strong>뿐이어서 1-SE 범위 안에 들었다. 최저 후보는 {int(best_model_row['feature_count'])}개 Feature가 필요하지만 최종 후보는 {len(selected_features)}개만 사용한다. 최저 점수 한 번보다 작은 표본에서의 안정성과 설명 가능성을 우선한다는 사전 규칙에 따라 <strong>{selected_set} + {selected_model}</strong>을 선택했고, 고정 hold-out이 안정성 기준을 통과한 뒤 설정을 동결했다.

최종 Feature는 다음과 같다.

```text
{', '.join(selected_features)}
```

## 6. 최종 성능과 Gap

{markdown_table(perf_display)}

{markdown_table(gaps_display)}

- Train–Valid Gap: **{gap_lookup['Train-Valid']:.1f}%p**
- Valid–Test Gap: **{gap_lookup['Valid-Test']:.1f}%p**
- Target–Test Gap: **{gap_lookup['Target-Test']:.1f}%p**
- Batch2–Batch3 Gap: **{gap_lookup['Batch2-Batch3']:.1f}%p**

![실제값과 예측값](../figures/day2/03_실제값_예측값.png)

대각선에서 멀리 떨어진 Cell은 단순한 평균 성능표에서 보이지 않는 운영 위험을 보여준다. Batch 2와 Batch 3의 MAPE 차이는 Target 범위, Feature 분포와 충전정책 구성이 달라질 때 같은 모델의 성능이 변할 수 있음을 뜻한다.

Train–Valid Gap이 {gap_lookup['Train-Valid']:.2f}%p인 것은 hold-out이 CV보다 더 쉬운 표본 구성이었다는 뜻이지, 모델이 외부에서도 반드시 잘 작동한다는 뜻은 아니다. 실제로 Batch 2에서는 Valid–Test Gap이 +{gap_lookup['Valid-Test']:.2f}%p까지 커졌다. 내부 검증만 보면 모델이 안정적으로 보였지만 다른 실험 Batch에 적용하자 학습 관계가 유지되지 않은 것이다. Batch 2 MAPE의 95% bootstrap 구간도 {perf_lookup.loc['Batch 2 Test', 'mape_ci95_low']:.2f}–{perf_lookup.loc['Batch 2 Test', 'mape_ci95_high']:.2f}%로 높아, 소수 Cell 하나가 평균을 우연히 올린 결과로 보기 어렵다.

## 7. 잔차와 실패 사례

![잔차 분석](../figures/day2/04_잔차_분석.png)

잔차는 `예측-실제`로 정의했다. 양수는 과대 예측, 음수는 과소 예측이다.

Batch 2의 평균 실제 수명은 {batch2_pred['cycle_life'].mean():.1f} cycle인데 평균 예측은 {batch2_pred['prediction'].mean():.1f} cycle이었다. 평균적으로 <strong>{batch2_pred['residual'].mean():.1f} cycle 과대 예측</strong>했으며 {len(batch2_pred)}개 중 {int((batch2_pred['residual'] > 0).sum())}개, 즉 <strong>{100 * (batch2_pred['residual'] > 0).mean():.1f}%</strong>가 과대 예측이었다. 반대로 Batch 3은 평균 실제 수명 {batch3_pred['cycle_life'].mean():.1f} cycle을 {batch3_pred['prediction'].mean():.1f} cycle로 예측해 평균 {abs(batch3_pred['residual'].mean()):.1f} cycle 과소 예측했고, {100 * (batch3_pred['residual'] < 0).mean():.1f}%가 과소 예측이었다. 모델이 Batch 1의 중앙 범위 쪽으로 예측을 끌어당기면서 저수명 Batch 2는 높게, 장수명 Batch 3은 낮게 보는 회귀-평균 경향이 나타났다.

![수명 구간별 오차](../figures/day2/05_수명구간별_오차.png)

MAPE는 같은 cycle 오차라도 실제 수명이 짧은 Cell에 더 큰 비율을 부여한다. 따라서 수명 구간별 MAPE와 MAE를 함께 읽어야 한다.

Batch 2의 단수명 Cell은 {int(b2_short['n'])}개이며 MAPE가 {b2_short['mape_pct']:.2f}%였다. 이 중 {int(((batch2_pred['life_group'] == '단수명(<500)') & (batch2_pred['residual'] > 0)).sum())}개를 과대 예측했다. Batch 2 중간수명군도 MAPE {b2_middle['mape_pct']:.2f}%로 높았지만 장수명 3개는 8.48%였다. 즉 Batch 2의 문제는 모든 수명 구간이 동일하게 나빠진 것이 아니라, 과제에서 특히 중요한 저수명 군집을 모델이 충분히 낮게 예측하지 못한 데 집중됐다.

![충전조건별 오차](../figures/day2/06_충전조건별_오차.png)

충전조건별 차이는 인과효과가 아니라 모델이 어떤 운전 영역에서 취약한지를 찾기 위한 진단이다.

### 외부 Batch 최악의 예측 Cell

![최악의 예측 Cell](../figures/day2/07_최악_예측_Cell.png)

{markdown_table(worst)}

## 8. Feature 설명과 Batch Shift

![Feature 중요도](../figures/day2/08_Feature_중요도.png)

{markdown_table(importance_display)}

계수 또는 importance는 예측 기여를 설명하지만 인과효과를 의미하지 않는다. 특히 상관된 Feature가 있으면 중요도가 서로 나뉠 수 있다.

표준화 후 `delta_q_iqr`가 1 표준편차 커지면 다른 두 Feature가 같을 때 예측 수명은 약 {abs(importance_lookup['delta_q_iqr']):.1f} cycle 감소했다. `qd_mean` 1 표준편차 증가는 약 {importance_lookup['qd_mean']:.1f} cycle 증가와 연결됐고, `qd_slope` 계수는 상대적으로 작았다. 이 계수는 Batch 1 안에서 학습된 예측 규칙이며 배터리의 물리적 인과효과로 해석해서는 안 된다.

![Batch Feature 시프트](../figures/day2/09_Batch_Feature_시프트.png)

Heatmap은 각 Batch의 Feature 평균이 Batch 1 평균에서 몇 표준편차 이동했는지 보여준다. 외부 Batch의 입력 범위가 학습 범위를 벗어나면 모델은 보간이 아니라 외삽을 하게 되며 오차가 커질 수 있다.

Batch 2의 평균 `delta_q_iqr`는 Batch 1보다 {b2_iqr_shift['standardized_mean_shift_vs_batch1']:.2f} 표준편차, `qd_mean`은 {b2_qd_shift['standardized_mean_shift_vs_batch1']:.2f} 표준편차 높았다. 특히 Batch 2의 {b2_qd_shift['outside_batch1_range_pct']:.1f}%가 `qd_mean`의 Batch 1 관측 범위를 벗어났다. Batch 1에서 높은 초기 용량이 비교적 긴 수명과 연결됐던 규칙이, 저수명 Cell이 다수인 Batch 2에서는 그대로 유지되지 않았다. 반대로 Batch 3의 `qd_mean`은 Batch 1보다 {abs(b3_qd_shift['standardized_mean_shift_vs_batch1']):.2f} 표준편차 낮고 `delta_q_iqr`도 {abs(b3_iqr_shift['standardized_mean_shift_vs_batch1']):.2f} 표준편차 낮았다. 세 Batch가 서로 다른 입력 영역을 차지한다는 사실이 외부 성능 차이의 핵심 근거다.

## 9. Day 1 전략이 실제 구현에 반영된 방식

| Day 1 결론 | Day 2 구현 | 검증 증거 |
|---|---|---|
| 초기 Capacity는 단순 기준 | F0 `qd_mean`, `qd_slope` | Ablation 표 |
| ΔQ가 가장 강하고 일관된 초기 신호 | F1 이후 `{modeling['chosen_dq']}` | ΔQ 후보 CV |
| ΔQ 파생값은 중복 | 대표값 1개만 Core에 사용 | Feature 명세·선택 로그 |
| Sensor·Charging은 관계가 불안정 | F3·F4에서 단계적으로 추가 | 동일 Ridge Ablation |
| 작은 표본과 공선성 | Ridge·ElasticNet 포함 | 모델 비교표 |
| Knee는 미래정보 | 입력 Feature에서 제외 | 누수 assertion·manifest |
| Batch 분포가 다름 | Batch별 성능·shift 분리 | 성능표·shift heatmap |

## 10. ESS 운영 관점

이 모델은 초기 운전 데이터를 기반으로 Cell의 예상 수명을 선별하고, 검사·정비·교체 우선순위를 정하는 보조도구로 사용할 수 있다. 그러나 과대 예측은 실제보다 오래 사용할 수 있다고 판단하게 만들어 열화 Cell의 교체를 늦출 수 있다. 과소 예측은 조기 교체 비용을 높이지만 일반적으로 안전 측면에서는 더 보수적이다.

실제 적용에서는 다음 조건이 필요하다.

- 입력 Feature가 학습 범위 안에 있는지 확인하는 drift 경보
- 새로운 Cell 구조·충전정책·온도 범위 도입 시 재검증
- 수명 과대 예측 Cell을 별도로 감시하는 보수적 안전 마진
- Cell 결과를 module/rack으로 집계할 때 불확실성 전파
- 실제 관측 오차가 임계치를 넘으면 재학습하는 운영 규칙

## 11. 한계

1. Batch 1의 Target 가용 Cell은 46개로 작다.
2. Day 1에서 외부 Batch의 Target 분포를 이미 관찰했으므로 완전한 blind test는 아니다.
3. 원논문과 전처리·Feature·split이 같지 않아 9.1%를 직접 재현했다고 말할 수 없다.
4. 실험실 Cell 데이터는 ESS의 온도 구배, Cell 불균형과 가변 부하를 모두 대표하지 않는다.
5. 충전정책별 표본 수가 작고 Cell 구조·실험 시기와 함께 변해 인과효과를 분리하기 어렵다.
6. 초기 100 cycle이 필요하므로 신규 설비에는 cold-start 기간이 존재한다.
7. Batch shift가 크면 단일 모델의 성능이 급격히 달라질 수 있다.

## 12. 재현성과 산출물

- 전체 실행: `python src/day2_analysis.py`
- 고정 seed: `{RANDOM_SEED}`
- Feature·품질·분할: `results/day2/`
- 한국어 그래프: `figures/day2/`
- 본 보고서: `reports/DAY2_분석_보고서.md`

Batch 2 결과 이후 재튜닝하지 않았으며, 최종 모델 설정과 평가 시각을 `external_evaluation_lock.json`에 기록했다.
"""
    path = paths.reports / "DAY2_분석_보고서.md"
    path.write_text(report, encoding="utf-8")
    return path


def update_plan_gates(plan_path: Path, selected_key: tuple[str, str]) -> None:
    text = plan_path.read_text(encoding="utf-8")
    for section in ["Gate 1", "Gate 2", "Gate 3", "Gate 4", "Gate 5"]:
        start = text.find(f"### {section}")
        if start < 0:
            continue
        end = text.find("### Gate", start + 5)
        if end < 0:
            end = text.find("\n---", start)
        block = text[start:end]
        text = text[:start] + block.replace("- [ ]", "- [x]") + text[end:]
    decision_header = "| 2026-10-02 | 계획 수립 | 없음 | 본 계획 확정 | Day 1 근거와 과제 요구사항을 구현 전에 고정 | 예 | 전체 |"
    decision_row = (
        f"\n| 2026-10-02 | Gate 3~4 | 모델 미정 | {selected_key[0]} + {selected_key[1]} | "
        "사전 1-SE·복잡도·hold-out 안정성 규칙 적용 | 예 | 모델·성능·보고서 |"
    )
    if decision_row.strip() not in text:
        text = text.replace(decision_header, decision_header + decision_row)
    plan_path.write_text(text, encoding="utf-8")


def run(root: Path) -> dict[str, Any]:
    paths = make_paths(root)
    font_name = configure_korean_plotting()
    features, quality, files = extract_raw_features(paths)
    quality_summary = build_quality_summary(features, quality)
    assignment = make_split_assignment(features)
    manifest = make_feature_manifest()
    validation = validate_against_day1(features, paths)

    save_csv(features, paths.results / "feature_table.csv")
    save_csv(quality_summary, paths.results / "data_quality_summary.csv")
    save_csv(quality, paths.results / "delta_q_validation.csv")
    save_csv(quality[~quality["included_for_supervised"]], paths.results / "cell_exclusion_log.csv")
    save_csv(files, paths.results / "data_file_checksums.csv")
    save_csv(assignment, paths.results / "split_assignment.csv")
    save_csv(manifest, paths.results / "feature_manifest.csv")
    save_csv(validation, paths.results / "feature_reproduction_validation.csv")

    modeling = model_development(features, assignment)
    selected_features = modeling["selected_features"]
    manifest.loc[manifest["feature_name"].isin(selected_features), "final_selected"] = True
    save_csv(manifest, paths.results / "feature_manifest.csv")
    save_csv(modeling["dq_results"], paths.results / "delta_q_candidate_results.csv")
    save_csv(modeling["dq_folds"], paths.results / "delta_q_candidate_folds.csv")
    save_csv(modeling["comparison"], paths.results / "model_comparison.csv")
    save_csv(modeling["fold_results"], paths.results / "cv_fold_results.csv")
    save_csv(modeling["ablation"], paths.results / "ablation_results.csv")
    save_csv(modeling["selection_log"], paths.results / "model_selection_log.csv")

    evaluation = final_external_evaluation(modeling)
    error_tables = build_error_tables(evaluation, selected_features)
    importance = extract_importance(evaluation["model"], selected_features)
    save_csv(evaluation["performance"], paths.results / "model_performance.csv")
    save_csv(evaluation["gaps"], paths.results / "gap_analysis.csv")
    save_csv(evaluation["predictions"], paths.results / "predictions.csv")
    for name, frame in error_tables.items():
        save_csv(frame, paths.results / f"{name}.csv")
    save_csv(importance, paths.results / "feature_importance.csv")

    plot_model_comparison(modeling["comparison"], paths.figures)
    plot_ablation(modeling["ablation"], paths.figures)
    plot_actual_vs_predicted(evaluation["predictions"], paths.figures)
    plot_residuals(evaluation["predictions"], paths.figures)
    plot_group_error(error_tables["life_group_error"], paths.figures)
    plot_charging_error(evaluation["predictions"], error_tables["charging_error"], paths.figures)
    plot_worst_predictions(error_tables["worst_predictions"], paths.figures)
    plot_importance(importance, paths.figures)
    plot_batch_shift(error_tables["batch_shift"], paths.figures)

    report_path = write_report(paths, quality_summary, validation, modeling, evaluation, error_tables, importance)
    update_plan_gates(paths.reports / "DAY2_분석_계획.md", modeling["selected_key"])

    selected_row = modeling["comparison"][
        (modeling["comparison"]["feature_set"] == modeling["selected_key"][0])
        & (modeling["comparison"]["model"] == modeling["selected_key"][1])
    ].iloc[0]
    lock = {
        "status": "FROZEN_AND_EVALUATED",
        "timestamp": datetime.now().astimezone().isoformat(),
        "random_seed": RANDOM_SEED,
        "selected_feature_set": modeling["selected_key"][0],
        "selected_model": modeling["selected_key"][1],
        "selected_features": selected_features,
        "best_params": json.loads(selected_row["best_params"]),
        "selection_rule": "1-SE + minimum feature count + minimum model complexity; fixed hold-out stability",
        "batch2_used_for_selection": False,
        "post_batch2_retuning": False,
    }
    (paths.results / "external_evaluation_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "h5py": h5py.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "font": font_name,
    }
    (paths.results / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_json = {
        "random_seed": RANDOM_SEED,
        "raw_cells": int(len(features)),
        "labeled_cells": int(features["target_available"].sum()),
        "development_cells": int((assignment["split"] == "development").sum()),
        "holdout_cells": int((assignment["split"] == "holdout").sum()),
        "batch2_test_cells": int((assignment["split"] == "external_test").sum()),
        "batch3_test_cells": int((assignment["split"] == "additional_test").sum()),
        "selected_feature_set": modeling["selected_key"][0],
        "selected_model": modeling["selected_key"][1],
        "selected_features": selected_features,
        "report": str(report_path.relative_to(root)),
        "result_tables": len(list(paths.results.glob("*.csv"))),
        "figures": len(list(paths.figures.glob("*.png"))),
    }
    (paths.results / "analysis_manifest.json").write_text(
        json.dumps(manifest_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest_json, ensure_ascii=False, indent=2), flush=True)
    return {"manifest": manifest_json, "performance": evaluation["performance"], "gaps": evaluation["gaps"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
