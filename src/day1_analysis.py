#!/usr/bin/env python3
"""Day 1 ESS 배터리 수명 EDA 전체 파이프라인.

대용량 MATLAB v7.3 파일을 h5py로 순차 접근해 필요한 필드만 읽고,
과제의 다섯 질문에 대응하는 표·그래프·보고서를 생성한다.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
from scipy import stats  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402


RANDOM_SEED = 42
RNG = np.random.default_rng(RANDOM_SEED)

BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BATCH_ORDER = list(BATCH_FILES)
BATCH_COLORS = {
    "Batch 1": "#2563EB",
    "Batch 2": "#F59E0B",
    "Batch 3": "#10B981",
}
LIFE_COLORS = {
    "단수명(<500)": "#E45756",
    "중간수명(500~1000)": "#9CA3AF",
    "장수명(>1000)": "#4C78A8",
}

FEATURE_LABELS = {
    "delta_q_iqr": "ΔQ 사분위범위",
    "delta_q_std": "ΔQ 표준편차",
    "delta_q_var": "ΔQ 분산",
    "delta_q_log10_var": "ΔQ 로그분산",
    "delta_q_q05": "ΔQ 5% 분위수",
    "delta_q_min": "ΔQ 최솟값",
    "delta_q_q25": "ΔQ 25% 분위수",
    "delta_q_l2": "ΔQ 제곱평균제곱근",
    "delta_q_abs_area": "ΔQ 절대면적",
    "delta_q_l1": "ΔQ 평균절대값",
    "delta_q_area": "ΔQ 부호면적",
    "delta_q_mean": "ΔQ 평균",
    "delta_q_q50": "ΔQ 중앙값",
    "delta_q_q75": "ΔQ 75% 분위수",
    "delta_q_q95": "ΔQ 95% 분위수",
    "chargetime_mean": "평균 충전시간",
    "chargetime_median": "충전시간 중앙값",
    "chargetime_slope": "충전시간 기울기",
    "chargetime_delta": "충전시간 변화량",
    "tavg_mean": "평균온도",
    "tmax_mean": "최고온도 평균",
    "tmin_mean": "최저온도 평균",
    "qd_mean": "평균 방전용량",
    "qd_std": "방전용량 표준편차",
    "qd_slope": "방전용량 기울기",
    "ir_mean": "평균 내부저항",
    "ir_std": "내부저항 표준편차",
    "ir_slope": "내부저항 기울기",
    "c_rate_stage1": "1단계 C-rate",
    "c_rate_stage2": "2단계 C-rate",
    "switch_soc_pct": "전환 SOC",
    "temperature_range_mean": "평균 온도범위",
    "coulombic_efficiency_mean": "평균 쿨롱효율",
}


@dataclass(frozen=True)
class OutputPaths:
    data_dir: Path
    result_dir: Path
    figure_dir: Path
    report_dir: Path


def configure_korean_plotting() -> str:
    """한글 폰트를 명시적으로 등록하고 공통 그래프 스타일을 설정한다."""

    candidates = [
        Path("/Users/lsh/Library/Fonts/NotoSansKR-Regular.otf"),
        Path("/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    ]
    font_path = next((p for p in candidates if p.exists()), None)
    if font_path is None:
        font_name = "DejaVu Sans"
        warnings.warn("한글 폰트를 찾지 못했습니다. DejaVu Sans를 사용합니다.")
    else:
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#1F2937",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    return font_name


def ensure_paths(root: Path) -> OutputPaths:
    paths = OutputPaths(
        data_dir=root / "data",
        result_dir=root / "results" / "day1",
        figure_dir=root / "figures" / "day1",
        report_dir=root / "reports",
    )
    for path in [paths.result_dir, paths.figure_dir, paths.report_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def decode_matlab_char(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1)
    if values.dtype.kind in "ui":
        return "".join(chr(int(v)) for v in values if int(v) != 0)
    if values.dtype.kind == "S":
        return b"".join(values.tolist()).decode("utf-8", errors="replace")
    return str(values[0]) if len(values) else ""


def deref_scalar(handle: h5py.File, ref) -> float:
    return float(np.asarray(handle[ref]).squeeze())


def flat(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset, dtype=float).reshape(-1)


def life_group(value: float) -> str:
    if not np.isfinite(value):
        return "Target 결측"
    if value < 500:
        return "단수명(<500)"
    if value > 1000:
        return "장수명(>1000)"
    return "중간수명(500~1000)"


POLICY_PATTERN = re.compile(
    r"(?P<c1>\d+(?:\.\d+)?)C\((?P<soc>\d+(?:\.\d+)?)%\)-(?P<c2>\d+(?:\.\d+)?)C",
    flags=re.IGNORECASE,
)
VAR_POLICY_PATTERN = re.compile(r"VarCharge-(?P<c1>\d+(?:\.\d+)?)C", re.IGNORECASE)


def parse_policy(policy: str) -> dict[str, float | bool | str]:
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
        c_rate = float(match.group("c1"))
        return {
            "policy_parse_success": True,
            "policy_type": "가변 충전",
            "c_rate_stage1": c_rate,
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
    x = cycles[mask]
    y = values[mask]
    result: dict[str, float] = {
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
    v = voltage[mask]
    dq = delta_q[mask]
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


def _prefix_sums(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    zero = np.array([0.0])
    return tuple(
        np.concatenate([zero, np.cumsum(arr)])
        for arr in (x, y, x * x, y * y, x * y)
    )


def _linear_sse(prefix: tuple[np.ndarray, ...], left: int, right: int) -> float:
    px, py, pxx, pyy, pxy = prefix
    n = right - left
    if n < 2:
        return np.inf
    sx, sy = px[right] - px[left], py[right] - py[left]
    sxx, syy = pxx[right] - pxx[left], pyy[right] - pyy[left]
    sxy = pxy[right] - pxy[left]
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope = 0.0
        intercept = sy / n
    else:
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
    sse = (
        syy
        + slope * slope * sxx
        + n * intercept * intercept
        + 2 * slope * intercept * sx
        - 2 * slope * sxy
        - 2 * intercept * sy
    )
    return float(max(sse, 0.0))


def estimate_knee(cycles: np.ndarray, qd: np.ndarray, cycle_life: float) -> dict[str, float | bool | str]:
    if not np.isfinite(cycle_life) or cycle_life <= 0:
        return {"knee_available": False, "knee_reason": "cycle_life 결측"}
    base_mask = (cycles >= 2) & (cycles <= 10) & valid_metric_mask("qd", qd)
    if base_mask.sum() < 3:
        return {"knee_available": False, "knee_reason": "초기 기준 용량 부족"}
    baseline = float(np.median(qd[base_mask]))
    mask = (
        np.isfinite(cycles)
        & np.isfinite(qd)
        & (qd > 0.5 * baseline)
        & (qd < 1.3 * baseline)
    )
    x, y = cycles[mask], qd[mask] / baseline
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 80:
        return {"knee_available": False, "knee_reason": "유효 열화 곡선 부족"}
    coverage = float(np.max(x) / cycle_life) if cycle_life > 0 else np.nan
    if coverage < 0.8:
        return {
            "knee_available": False,
            "knee_reason": "전체 수명의 80% 미만만 관측",
            "curve_coverage": coverage,
        }

    smooth = pd.Series(y).rolling(9, center=True, min_periods=1).median().to_numpy()
    prefix = _prefix_sums(x, smooth)
    min_segment = max(20, int(len(x) * 0.10))
    candidates = range(min_segment, len(x) - min_segment)
    best_index, best_sse = None, np.inf
    for idx in candidates:
        score = _linear_sse(prefix, 0, idx) + _linear_sse(prefix, idx, len(x))
        if score < best_sse:
            best_index, best_sse = idx, score
    single_sse = _linear_sse(prefix, 0, len(x))
    if best_index is None or not np.isfinite(best_sse):
        return {"knee_available": False, "knee_reason": "구간 회귀 실패"}

    improvement = 1.0 - best_sse / single_sse if single_sse > 0 else 0.0
    pre_slope = float(np.polyfit(x[:best_index], smooth[:best_index], 1)[0])
    post_slope = float(np.polyfit(x[best_index:], smooth[best_index:], 1)[0])
    return {
        "knee_available": True,
        "knee_reason": "계산 완료",
        "curve_coverage": coverage,
        "baseline_qd": baseline,
        "knee_cycle": float(x[best_index]),
        "knee_fraction_of_life": float(x[best_index] / cycle_life),
        "knee_fit_improvement": float(improvement),
        "pre_knee_slope": pre_slope,
        "post_knee_slope": post_slope,
    }


def extract_batches(paths: OutputPaths) -> dict[str, pd.DataFrame]:
    cell_rows: list[dict] = []
    early_rows: list[dict] = []
    quality_rows: list[dict] = []
    degradation_rows: list[dict] = []
    delta_feature_rows: list[dict] = []
    delta_curve_rows: list[dict] = []
    knee_rows: list[dict] = []

    metric_map = {
        "qd": "QDischarge",
        "qc": "QCharge",
        "ir": "IR",
        "tavg": "Tavg",
        "tmax": "Tmax",
        "tmin": "Tmin",
        "chargetime": "chargetime",
    }

    for batch_name, filename in BATCH_FILES.items():
        file_path = paths.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {file_path}")
        print(f"[추출] {batch_name}: {filename}", flush=True)
        with h5py.File(file_path, "r") as handle:
            batch = handle["batch"]
            n_cells = batch["cycle_life"].shape[0]
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
                max_cycle = float(np.nanmax(cycles)) if len(cycles) else np.nan
                min_cycle = float(np.nanmin(cycles)) if len(cycles) else np.nan
                target_gap = max_cycle - cycle_life if np.isfinite(max_cycle) else np.nan
                curve_coverage = (
                    max_cycle / cycle_life
                    if np.isfinite(cycle_life) and cycle_life > 0
                    else np.nan
                )

                cell_record = {
                    "batch": batch_name,
                    "cell_index": cell_index,
                    "global_cell_id": global_id,
                    "cycle_life": cycle_life,
                    "life_group": life_group(cycle_life),
                    "charging_policy": policy,
                    **policy_info,
                }
                cell_rows.append(cell_record)

                early_record = dict(cell_record)
                for metric in metric_map:
                    early_record.update(metric_features(metric, cycles, arrays[metric]))
                early_mask = (cycles >= 2) & (cycles <= 100)
                temp_range = arrays["tmax"] - arrays["tmin"]
                temp_valid = early_mask & np.isfinite(temp_range) & (temp_range >= 0) & (temp_range < 50)
                early_record["temperature_range_mean"] = (
                    float(np.mean(temp_range[temp_valid])) if temp_valid.any() else np.nan
                )
                ratio_mask = (
                    early_mask
                    & valid_metric_mask("qd", arrays["qd"])
                    & valid_metric_mask("qc", arrays["qc"])
                    & (arrays["qc"] != 0)
                )
                early_record["coulombic_efficiency_mean"] = (
                    float(np.mean(arrays["qd"][ratio_mask] / arrays["qc"][ratio_mask]))
                    if ratio_mask.any()
                    else np.nan
                )

                cycles_group = handle[batch["cycles"][cell_index, 0]]
                qdlin_refs = cycles_group["Qdlin"]
                cycles_object_length = int(qdlin_refs.shape[0])
                cycle_to_index = {
                    int(round(cycle)): idx
                    for idx, cycle in enumerate(cycles)
                    if np.isfinite(cycle)
                }
                has_10 = 10 in cycle_to_index
                has_100 = 100 in cycle_to_index
                delta_valid = False
                delta_reason = "cycle 10 또는 100 없음"
                delta_record = dict(cell_record)

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
                                q10 = q10_raw.astype(float)
                                q100 = q100_raw.astype(float)
                                delta_q = q100 - q10
                                delta_stats = delta_q_statistics(voltage, delta_q)
                                if delta_stats:
                                    delta_record.update(delta_stats)
                                    delta_valid = True
                                    delta_reason = "계산 완료"
                                    # 그래프 파일 크기를 줄이되 곡선 형태는 보존한다.
                                    stride = max(1, len(voltage) // 250)
                                    for v, dq in zip(voltage[::stride], delta_q[::stride]):
                                        if np.isfinite(v) and np.isfinite(dq):
                                            delta_curve_rows.append(
                                                {
                                                    "batch": batch_name,
                                                    "global_cell_id": global_id,
                                                    "cycle_life": cycle_life,
                                                    "voltage": float(v),
                                                    "delta_q": float(dq),
                                                }
                                            )
                                else:
                                    delta_reason = "유효 ΔQ(V) 포인트 부족"
                            else:
                                delta_reason = "Qdlin/Vdlin 길이 또는 자료형 불일치"
                        except (KeyError, ValueError, TypeError, OSError) as exc:
                            delta_reason = f"Qdlin 읽기 실패: {type(exc).__name__}"
                delta_record["delta_q_available"] = delta_valid
                delta_record["delta_q_reason"] = delta_reason
                delta_feature_rows.append(delta_record)
                early_record["delta_q_available"] = delta_valid
                if delta_valid:
                    for key, value in delta_record.items():
                        if key.startswith("delta_q_") and key not in {
                            "delta_q_available",
                            "delta_q_reason",
                        }:
                            early_record[key] = value
                early_rows.append(early_record)

                first_cycle_mask = cycles == 1
                first_all_zero = False
                if first_cycle_mask.any():
                    first_idx = int(np.flatnonzero(first_cycle_mask)[0])
                    first_all_zero = all(
                        np.isclose(arrays[m][first_idx], 0.0, equal_nan=False)
                        for m in metric_map
                    )
                duplicate_cycles = int(pd.Series(cycles).duplicated().sum())
                qd_invalid = int((~valid_metric_mask("qd", arrays["qd"])).sum())
                early_window = (cycles >= 2) & (cycles <= 100)
                early_ir_valid_count = int(
                    (early_window & valid_metric_mask("ir", arrays["ir"])).sum()
                )
                early_tmin_invalid_count = int(
                    (early_window & ~valid_metric_mask("tmin", arrays["tmin"])).sum()
                )
                nan_inf_total = int(
                    sum((~np.isfinite(arrays[m])).sum() for m in metric_map)
                )
                quality_rows.append(
                    {
                        "batch": batch_name,
                        "global_cell_id": global_id,
                        "cycle_life": cycle_life,
                        "summary_length": common_length,
                        "summary_min_cycle": min_cycle,
                        "summary_max_cycle": max_cycle,
                        "cycles_object_length": cycles_object_length,
                        "summary_fields_same_length": len(set(lengths.values())) == 1,
                        "summary_cycles_match_objects": common_length == cycles_object_length,
                        "cycles_strictly_increasing": bool(np.all(np.diff(cycles) > 0)),
                        "duplicate_cycle_count": duplicate_cycles,
                        "first_cycle_all_zero": first_all_zero,
                        "invalid_qd_count": qd_invalid,
                        "early_ir_valid_count": early_ir_valid_count,
                        "early_ir_unavailable": early_ir_valid_count < 20,
                        "early_tmin_invalid_count": early_tmin_invalid_count,
                        "nan_inf_total": nan_inf_total,
                        "cycle_10_available": has_10,
                        "cycle_100_available": has_100,
                        "delta_q_available": delta_valid,
                        "delta_q_reason": delta_reason,
                        "target_minus_observed_end": cycle_life - max_cycle,
                        "curve_coverage": curve_coverage,
                        "full_curve_coverage": bool(curve_coverage >= 0.8),
                        "target_available": bool(np.isfinite(cycle_life)),
                        "policy_parse_success": bool(policy_info["policy_parse_success"]),
                    }
                )

                base_mask = (cycles >= 2) & (cycles <= 10) & valid_metric_mask("qd", arrays["qd"])
                baseline = float(np.median(arrays["qd"][base_mask])) if base_mask.any() else np.nan
                if np.isfinite(baseline) and baseline > 0 and np.isfinite(cycle_life):
                    qd_plot_mask = (
                        np.isfinite(cycles)
                        & np.isfinite(arrays["qd"])
                        & (cycles <= cycle_life)
                        & (arrays["qd"] > 0.5 * baseline)
                        & (arrays["qd"] < 1.3 * baseline)
                    )
                    for cyc, qd in zip(cycles[qd_plot_mask], arrays["qd"][qd_plot_mask]):
                        degradation_rows.append(
                            {
                                "batch": batch_name,
                                "global_cell_id": global_id,
                                "cycle_life": cycle_life,
                                "life_group": life_group(cycle_life),
                                "cycle": float(cyc),
                                "qd": float(qd),
                                "soh": float(qd / baseline),
                            }
                        )

                if np.isfinite(cycle_life):
                    target_curve_mask = cycles <= cycle_life
                    knee_result = estimate_knee(
                        cycles[target_curve_mask], arrays["qd"][target_curve_mask], cycle_life
                    )
                else:
                    knee_result = {"knee_available": False, "knee_reason": "cycle_life 결측"}
                knee_rows.append({**cell_record, **knee_result})

        gc.collect()

    frames = {
        "cells": pd.DataFrame(cell_rows),
        "early_features": pd.DataFrame(early_rows),
        "quality": pd.DataFrame(quality_rows),
        "degradation": pd.DataFrame(degradation_rows),
        "delta_features": pd.DataFrame(delta_feature_rows),
        "delta_curves": pd.DataFrame(delta_curve_rows),
        "knees": pd.DataFrame(knee_rows),
    }
    return frames


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = 2000) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    samples = RNG.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if len(valid_idx) == 0:
        return pd.Series(adjusted, index=p_values.index)
    order = valid_idx[np.argsort(values[valid_idx])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0, 1)
    return pd.Series(adjusted, index=p_values.index)


def summarize_analysis(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cells = frames["cells"].copy()
    quality = frames["quality"].copy()
    early = frames["early_features"].copy()
    knees = frames["knees"].copy()

    batch_summary = (
        cells.groupby("batch", observed=True)["cycle_life"]
        .agg(
            cell_count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            max="max",
            skew="skew",
        )
        .reindex(BATCH_ORDER)
        .reset_index()
    )

    rate_rows = []
    for batch_name, group in cells.groupby("batch", observed=True):
        group = group[group["cycle_life"].notna()]
        for group_name in LIFE_COLORS:
            count = int((group["life_group"] == group_name).sum())
            total = len(group)
            low, high = wilson_interval(count, total)
            rate_rows.append(
                {
                    "batch": batch_name,
                    "life_group": group_name,
                    "count": count,
                    "total": total,
                    "rate_pct": 100 * count / total,
                    "ci_low_pct": 100 * low,
                    "ci_high_pct": 100 * high,
                }
            )
    group_rates = pd.DataFrame(rate_rows)

    outlier_rows = []
    for batch_name, group in cells.groupby("batch", observed=True):
        group = group[group["cycle_life"].notna()]
        q1, q3 = group["cycle_life"].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = group[(group["cycle_life"] < low) | (group["cycle_life"] > high)].copy()
        for _, row in flagged.iterrows():
            outlier_rows.append(
                {
                    "batch": batch_name,
                    "global_cell_id": row["global_cell_id"],
                    "cycle_life": row["cycle_life"],
                    "charging_policy": row["charging_policy"],
                    "outlier_direction": "낮음" if row["cycle_life"] < low else "높음",
                    "tukey_low_fence": low,
                    "tukey_high_fence": high,
                }
            )
    outliers = pd.DataFrame(outlier_rows)
    if not outliers.empty:
        diagnostic_columns = [
            "global_cell_id",
            "tavg_mean",
            "tmax_mean",
            "chargetime_mean",
            "c_rate_stage1",
            "delta_q_log10_var",
            "qd_slope",
        ]
        available_columns = [column for column in diagnostic_columns if column in early.columns]
        outliers = outliers.merge(
            early[available_columns], on="global_cell_id", how="left"
        )

    quality_summary = (
        quality.groupby("batch", observed=True)
        .agg(
            cells=("global_cell_id", "count"),
            labeled_cells=("target_available", "sum"),
            missing_target_cells=("target_available", lambda s: int((~s).sum())),
            first_cycle_zero_cells=("first_cycle_all_zero", "sum"),
            cycle_10_available_cells=("cycle_10_available", "sum"),
            cycle_100_available_cells=("cycle_100_available", "sum"),
            delta_q_available_cells=("delta_q_available", "sum"),
            full_curve_cells=("full_curve_coverage", "sum"),
            policy_parse_success_cells=("policy_parse_success", "sum"),
            cells_with_nan_inf=("nan_inf_total", lambda s: int((s > 0).sum())),
            cells_with_duplicate_cycles=("duplicate_cycle_count", lambda s: int((s > 0).sum())),
            summary_object_mismatch=("summary_cycles_match_objects", lambda s: int((~s).sum())),
            early_ir_unavailable_cells=("early_ir_unavailable", "sum"),
            cells_with_early_tmin_invalid=("early_tmin_invalid_count", lambda s: int((s > 0).sum())),
        )
        .reindex(BATCH_ORDER)
        .reset_index()
    )
    for column in [
        "labeled_cells",
        "cycle_10_available_cells",
        "cycle_100_available_cells",
        "delta_q_available_cells",
        "full_curve_cells",
        "policy_parse_success_cells",
    ]:
        quality_summary[column.replace("_cells", "_pct")] = (
            100 * quality_summary[column] / quality_summary["cells"]
        )
    quality_summary["knee_eligible_pct_labeled"] = (
        100 * quality_summary["full_curve_cells"] / quality_summary["labeled_cells"]
    )

    test_rows = []
    grouped_life = [
        cells.loc[cells["batch"] == batch, "cycle_life"].dropna().to_numpy()
        for batch in BATCH_ORDER
    ]
    h_stat, h_p = stats.kruskal(*grouped_life)
    test_rows.append(
        {
            "analysis": "배치별 cycle_life 분포",
            "test": "Kruskal-Wallis",
            "comparison": "Batch 1 vs Batch 2 vs Batch 3",
            "statistic": h_stat,
            "p_value": h_p,
            "effect_or_distance": np.nan,
        }
    )
    for i, left in enumerate(BATCH_ORDER):
        for right in BATCH_ORDER[i + 1 :]:
            x = cells.loc[cells["batch"] == left, "cycle_life"].dropna().to_numpy()
            y = cells.loc[cells["batch"] == right, "cycle_life"].dropna().to_numpy()
            ks = stats.ks_2samp(x, y)
            test_rows.append(
                {
                    "analysis": "배치별 cycle_life 분포",
                    "test": "Kolmogorov-Smirnov",
                    "comparison": f"{left} vs {right}",
                    "statistic": ks.statistic,
                    "p_value": ks.pvalue,
                    "effect_or_distance": wasserstein_distance(x, y),
                }
            )

    numeric_exclusions = {
        "cell_index",
        "cycle_life",
        "policy_parse_success",
        "policy_new_structure",
        "policy_slow_cycle",
        "delta_q_available",
    }
    feature_columns = [
        column
        for column in early.select_dtypes(include=[np.number]).columns
        if column not in numeric_exclusions
        and not column.endswith("_n")
        and early[column].notna().sum() >= 20
    ]
    corr_rows = []
    scopes = ["전체", *BATCH_ORDER]
    for scope in scopes:
        scope_df = early if scope == "전체" else early[early["batch"] == scope]
        for feature in feature_columns:
            subset = scope_df[[feature, "cycle_life"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset) < 8 or subset[feature].nunique() < 3:
                continue
            pearson = stats.pearsonr(subset[feature], subset["cycle_life"])
            spearman = stats.spearmanr(subset[feature], subset["cycle_life"])
            corr_rows.append(
                {
                    "scope": scope,
                    "feature": feature,
                    "n": len(subset),
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_r": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                }
            )
    correlations = pd.DataFrame(corr_rows)
    if not correlations.empty:
        correlations["spearman_p_bh"] = correlations.groupby("scope", observed=True)[
            "spearman_p"
        ].transform(bh_adjust)
        correlations["abs_spearman"] = correlations["spearman_r"].abs()

    policy_rows = []
    for (batch_name, policy), group in cells.groupby(["batch", "charging_policy"], observed=True):
        target = group["cycle_life"].dropna()
        if len(target) == 0:
            continue
        ci_low, ci_high = bootstrap_mean_ci(target)
        policy_rows.append(
            {
                "batch": batch_name,
                "charging_policy": policy,
                "n": len(target),
                "mean_cycle_life": target.mean(),
                "median_cycle_life": target.median(),
                "std_cycle_life": target.std(ddof=1),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )
    policy_summary = pd.DataFrame(policy_rows)

    parsed = cells[["cycle_life", "c_rate_stage1", "c_rate_stage2", "switch_soc_pct", "batch"]]
    for feature in ["c_rate_stage1", "c_rate_stage2", "switch_soc_pct"]:
        subset = parsed[[feature, "cycle_life"]].dropna()
        if len(subset) >= 8 and subset[feature].nunique() >= 3:
            result = stats.spearmanr(subset[feature], subset["cycle_life"])
            test_rows.append(
                {
                    "analysis": "충전 조건과 수명",
                    "test": "Spearman correlation",
                    "comparison": feature,
                    "statistic": result.statistic,
                    "p_value": result.pvalue,
                    "effect_or_distance": np.nan,
                }
            )

    for batch_name in BATCH_ORDER:
        subset = early[
            (early["batch"] == batch_name)
            & early["cycle_life"].notna()
            & early["delta_q_log10_var"].notna()
        ]
        if len(subset) < 12:
            continue
        low_cut = subset["cycle_life"].quantile(0.25)
        high_cut = subset["cycle_life"].quantile(0.75)
        low_values = subset.loc[
            subset["cycle_life"] <= low_cut, "delta_q_log10_var"
        ].to_numpy()
        high_values = subset.loc[
            subset["cycle_life"] >= high_cut, "delta_q_log10_var"
        ].to_numpy()
        test = stats.mannwhitneyu(high_values, low_values, alternative="two-sided")
        rank_biserial = 2 * test.statistic / (len(high_values) * len(low_values)) - 1
        test_rows.append(
            {
                "analysis": "ΔQ 장·단수명 분위수 비교",
                "test": "Mann-Whitney U",
                "comparison": batch_name,
                "statistic": test.statistic,
                "p_value": test.pvalue,
                "effect_or_distance": rank_biserial,
            }
        )

    knee_valid = knees[knees.get("knee_available", False) == True]  # noqa: E712
    if len(knee_valid) >= 8:
        knee_corr = stats.spearmanr(knee_valid["knee_cycle"], knee_valid["cycle_life"])
        test_rows.append(
            {
                "analysis": "knee point와 수명",
                "test": "Spearman correlation",
                "comparison": "knee_cycle vs cycle_life",
                "statistic": knee_corr.statistic,
                "p_value": knee_corr.pvalue,
                "effect_or_distance": np.nan,
            }
        )

    statistical_tests = pd.DataFrame(test_rows)

    # Batch 1을 기준으로 표준화 평균차와 정규화 Wasserstein 거리를 계산한다.
    if correlations.empty:
        top_features: list[str] = []
    else:
        ranked_features = (
            correlations[correlations["scope"] == "전체"]
            .sort_values("abs_spearman", ascending=False)["feature"]
            .drop_duplicates()
            .tolist()
        )
        preferred_features = [
            "delta_q_log10_var",
            "delta_q_min",
            "qd_slope",
            "ir_slope",
            "tavg_mean",
            "tmax_mean",
            "chargetime_mean",
            "c_rate_stage1",
            "switch_soc_pct",
            "temperature_range_mean",
            "coulombic_efficiency_mean",
            "qd_mean",
        ]
        top_features = [
            feature for feature in preferred_features if feature in ranked_features
        ]
        top_features.extend(
            feature for feature in ranked_features if feature not in top_features
        )
        top_features = top_features[:15]
    shift_rows = []
    base = early[early["batch"] == "Batch 1"]
    for feature in top_features:
        base_values = base[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if len(base_values) < 5:
            continue
        base_mean, base_std = np.mean(base_values), np.std(base_values, ddof=1)
        if base_std == 0 or not np.isfinite(base_std):
            continue
        for batch_name in ["Batch 2", "Batch 3"]:
            values = (
                early.loc[early["batch"] == batch_name, feature]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy()
            )
            if len(values) < 5:
                continue
            shift_rows.append(
                {
                    "feature": feature,
                    "comparison": f"{batch_name} - Batch 1",
                    "standardized_mean_difference": (np.mean(values) - base_mean) / base_std,
                    "normalized_wasserstein": wasserstein_distance(base_values, values) / base_std,
                    "batch1_n": len(base_values),
                    "comparison_n": len(values),
                }
            )
    batch_shift = pd.DataFrame(shift_rows)

    return {
        "batch_summary": batch_summary,
        "lifetime_group_rates": group_rates,
        "outlier_cells": outliers,
        "quality_audit": quality_summary,
        "correlation_results": correlations,
        "policy_summary": policy_summary,
        "statistical_tests": statistical_tests,
        "batch_shift": batch_shift,
    }


def build_feature_decisions(
    correlations: pd.DataFrame,
    early: pd.DataFrame,
) -> pd.DataFrame:
    if correlations.empty:
        pooled = pd.DataFrame(columns=["feature", "spearman_r"])
    else:
        pooled = correlations[correlations["scope"] == "전체"].set_index("feature")

    def corr(feature: str) -> float:
        if feature in pooled.index:
            return float(pooled.loc[feature, "spearman_r"])
        return np.nan

    def availability(feature: str) -> float:
        return 100 * early[feature].notna().mean() if feature in early else np.nan

    rows = [
        {
            "feature": "delta_q_log10_var",
            "source": "Qdlin",
            "cycle_range": "10, 100",
            "formula": "log10(var(Q100-Q10)+1e-12)",
            "missing_handling": "두 곡선이 모두 유효한 셀만 계산; pipeline에서 대체",
            "pooled_spearman": corr("delta_q_log10_var"),
            "availability_pct": availability("delta_q_log10_var"),
            "decision": "우선 채택",
            "reason": "초기 열화 형상을 압축하며 원논문 핵심 계열 feature",
            "leakage_risk": "낮음: 100사이클 이내",
        },
        {
            "feature": "delta_q_min",
            "source": "Qdlin",
            "cycle_range": "10, 100",
            "formula": "min(Q100-Q10)",
            "missing_handling": "ΔQ(V) 미가용 셀은 대체 및 결측 flag",
            "pooled_spearman": corr("delta_q_min"),
            "availability_pct": availability("delta_q_min"),
            "decision": "후보 채택",
            "reason": "곡선의 국소 열화 강도 반영",
            "leakage_risk": "낮음",
        },
        {
            "feature": "qd_slope",
            "source": "summary.QDischarge",
            "cycle_range": "2~100",
            "formula": "QD~cycle 선형회귀 기울기",
            "missing_handling": "물리 범위 밖 값 제외 후 계산",
            "pooled_spearman": corr("qd_slope"),
            "availability_pct": availability("qd_slope"),
            "decision": "조건부 후보",
            "reason": "초기 용량 변화 속도를 반영하지만 배치별 상관 방향이 달라 CV·ablation 필요",
            "leakage_risk": "낮음",
        },
        {
            "feature": "ir_slope",
            "source": "summary.IR",
            "cycle_range": "2~100",
            "formula": "IR~cycle 선형회귀 기울기",
            "missing_handling": "비정상·0값 제외",
            "pooled_spearman": corr("ir_slope"),
            "availability_pct": availability("ir_slope"),
            "decision": "조건부 후보",
            "reason": "저항 증가율 후보이나 Batch 2 결측과 배치별 방향 불일치 확인 필요",
            "leakage_risk": "낮음",
        },
        {
            "feature": "tavg_mean",
            "source": "summary.Tavg",
            "cycle_range": "2~100",
            "formula": "초기 평균 온도",
            "missing_handling": "5~80°C 유효 범위",
            "pooled_spearman": corr("tavg_mean"),
            "availability_pct": availability("tavg_mean"),
            "decision": "조건부 후보",
            "reason": "열 스트레스 후보이나 배치별 상관 방향이 달라 Batch 1 CV에서 검증",
            "leakage_risk": "낮음",
        },
        {
            "feature": "tmax_mean",
            "source": "summary.Tmax",
            "cycle_range": "2~100",
            "formula": "초기 최고온도의 평균",
            "missing_handling": "5~80°C 유효 범위",
            "pooled_spearman": corr("tmax_mean"),
            "availability_pct": availability("tmax_mean"),
            "decision": "조건부/중복 검토",
            "reason": "Tavg와 강한 공선성 가능; CV에서 한 변수만 선택 검토",
            "leakage_risk": "낮음",
        },
        {
            "feature": "chargetime_mean",
            "source": "summary.chargetime",
            "cycle_range": "2~100",
            "formula": "초기 평균 충전 시간",
            "missing_handling": "0.1~120분 범위",
            "pooled_spearman": corr("chargetime_mean"),
            "availability_pct": availability("chargetime_mean"),
            "decision": "조건부 후보",
            "reason": "C-rate의 연속형 대리 변수이나 배치별 관계 반전이 커 민감도 분석 필요",
            "leakage_risk": "낮음",
        },
        {
            "feature": "c_rate_stage1 / switch_soc_pct / c_rate_stage2",
            "source": "policy_readable",
            "cycle_range": "실험 설정",
            "formula": "충전 정책 문자열 정규식 파싱",
            "missing_handling": "가변 충전·미파싱 정책에 결측 flag",
            "pooled_spearman": corr("c_rate_stage1"),
            "availability_pct": availability("c_rate_stage1"),
            "decision": "조건부 후보",
            "reason": "미지 정책 일반화에 유리하나 배치·셀구조 교란과 방향 불일치가 큼",
            "leakage_risk": "낮음; 운영 시 사전 가용성 확인",
        },
        {
            "feature": "knee_cycle",
            "source": "전체 QD 곡선",
            "cycle_range": "전체 수명",
            "formula": "2구간 회귀 SSE 최소점",
            "missing_handling": "불완전 곡선은 미계산",
            "pooled_spearman": np.nan,
            "availability_pct": np.nan,
            "decision": "Day 2 입력에서 제외",
            "reason": "설명용으로 유용하지만 미래 전체 수명 정보 사용",
            "leakage_risk": "매우 높음",
        },
        {
            "feature": "cycle_life / EOL 파생값",
            "source": "target",
            "cycle_range": "전체 수명",
            "formula": "target 또는 target 파생",
            "missing_handling": "해당 없음",
            "pooled_spearman": np.nan,
            "availability_pct": 100.0,
            "decision": "제외",
            "reason": "정답 누수",
            "leakage_risk": "매우 높음",
        },
    ]
    decisions = pd.DataFrame(rows)
    lookup_alias = {
        "c_rate_stage1 / switch_soc_pct / c_rate_stage2": "c_rate_stage1",
    }
    for idx, row in decisions.iterrows():
        lookup = lookup_alias.get(row["feature"], row["feature"])
        batch_values = {}
        for batch_name in BATCH_ORDER:
            match = correlations[
                (correlations["scope"] == batch_name)
                & (correlations["feature"] == lookup)
            ]
            batch_values[batch_name] = (
                float(match.iloc[0]["spearman_r"]) if len(match) else np.nan
            )
            decisions.loc[idx, f"{batch_name.lower().replace(' ', '_')}_spearman"] = batch_values[
                batch_name
            ]
        finite_values = [value for value in batch_values.values() if np.isfinite(value)]
        if finite_values:
            signs = np.sign(finite_values)
            decisions.loc[idx, "batch_direction_consistency"] = (
                "동일 방향" if np.all(signs == signs[0]) else "방향 불일치"
            )
        else:
            decisions.loc[idx, "batch_direction_consistency"] = "해당 없음"
    return decisions


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.7)
    ax.grid(axis="x", visible=False)


def plot_quality(summary: pd.DataFrame, figure_dir: Path) -> None:
    metrics = {
        "labeled_pct": "수명 목표값 가용",
        "delta_q_available_pct": "ΔQ(V) 가용",
        "knee_eligible_pct_labeled": "목표값 보유 셀 중 급격 열화점 분석 가능",
        "policy_parse_success_pct": "충전정책 파싱 성공",
    }
    plot_df = summary[["batch", *metrics]].melt(
        id_vars="batch", var_name="metric", value_name="rate"
    )
    plot_df["metric"] = plot_df["metric"].map(metrics)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="metric",
        y="rate",
        hue="batch",
        hue_order=BATCH_ORDER,
        palette=BATCH_COLORS,
        ax=ax,
    )
    ax.set_title("배치별 분석 데이터 가용률")
    ax.set_xlabel("")
    ax.set_ylabel("가용 셀 비율(%)")
    ax.set_ylim(0, 108)
    ax.legend(title="배치", ncol=3, loc="upper center")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)
    clean_axis(ax)
    save_figure(fig, figure_dir / "00_데이터_품질_가용률.png")


def plot_lifetime_distribution(cells: pd.DataFrame, figure_dir: Path) -> None:
    bins = np.arange(150, 2351, 100)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for column, batch_name in enumerate(BATCH_ORDER):
        values = cells.loc[cells["batch"] == batch_name, "cycle_life"].dropna()
        ax = axes[0, column]
        ax.hist(values, bins=bins, color=BATCH_COLORS[batch_name], alpha=0.82, edgecolor="white")
        ax.axvline(values.median(), color="#111827", linestyle="--", linewidth=1.5)
        ax.set_title(f"{batch_name} 수명 분포 (n={len(values)})")
        ax.set_ylabel("배터리 셀 수")
        clean_axis(ax)

        ax = axes[1, column]
        sorted_values = np.sort(values)
        ecdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        ax.step(sorted_values, ecdf, where="post", color=BATCH_COLORS[batch_name], linewidth=2.2)
        ax.axvline(500, color=LIFE_COLORS["단수명(<500)"], linestyle=":", linewidth=1.5)
        ax.axvline(1000, color=LIFE_COLORS["장수명(>1000)"], linestyle=":", linewidth=1.5)
        ax.set_xlabel("수명(cycle)")
        ax.set_ylabel("누적 비율")
        ax.set_xlim(150, 2300)
        ax.set_ylim(0, 1.03)
        clean_axis(ax)
    fig.suptitle("질문 1. 배치별 배터리 수명 분포와 누적분포", fontsize=18, fontweight="bold")
    fig.subplots_adjust(top=0.90)
    save_figure(fig, figure_dir / "01_수명_분포.png")


def plot_lifetime_groups(cells: pd.DataFrame, rates: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    sns.violinplot(
        data=cells,
        x="batch",
        y="cycle_life",
        order=BATCH_ORDER,
        hue="batch",
        palette=BATCH_COLORS,
        inner=None,
        cut=0,
        legend=False,
        ax=axes[0],
    )
    sns.stripplot(
        data=cells,
        x="batch",
        y="cycle_life",
        order=BATCH_ORDER,
        color="#1F2937",
        alpha=0.62,
        size=4,
        jitter=0.18,
        ax=axes[0],
    )
    axes[0].axhline(500, color=LIFE_COLORS["단수명(<500)"], linestyle=":")
    axes[0].axhline(1000, color=LIFE_COLORS["장수명(>1000)"], linestyle=":")
    axes[0].set_title("배치별 수명 분포와 개별 셀")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("수명(cycle)")
    clean_axis(axes[0])

    pivot = rates.pivot(index="batch", columns="life_group", values="rate_pct").reindex(BATCH_ORDER)
    bottom = np.zeros(len(pivot))
    for group_name in LIFE_COLORS:
        values = pivot.get(group_name, pd.Series(0, index=pivot.index)).to_numpy()
        axes[1].bar(
            pivot.index,
            values,
            bottom=bottom,
            label=group_name,
            color=LIFE_COLORS[group_name],
            edgecolor="white",
        )
        for idx, value in enumerate(values):
            if value >= 5:
                axes[1].text(idx, bottom[idx] + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=9)
        bottom += values
    axes[1].set_title("수명 구간 구성비")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("비율(%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(title="수명 구간", loc="upper left", bbox_to_anchor=(1.01, 1))
    clean_axis(axes[1])
    fig.suptitle("질문 1. 이상치와 수명 구간 비교", fontsize=18, fontweight="bold")
    save_figure(fig, figure_dir / "02_수명_구간_비율.png")


def plot_degradation(degradation: pd.DataFrame, figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True, sharey=True)
    for ax, batch_name in zip(axes, BATCH_ORDER):
        batch_df = degradation[degradation["batch"] == batch_name]
        for cell_id, cell_df in batch_df.groupby("global_cell_id", observed=True):
            group = cell_df["life_group"].iloc[0]
            ax.plot(
                cell_df["cycle"],
                cell_df["soh"],
                color=LIFE_COLORS[group],
                linewidth=0.65,
                alpha=0.42,
            )
        ax.axhline(0.8, color="#DC2626", linestyle="--", linewidth=1.5, label="EOL 80%")
        ax.set_title(f"{batch_name} (n={batch_df['global_cell_id'].nunique()})")
        ax.set_xlabel("사이클")
        ax.set_xlim(0, 2300)
        ax.set_ylim(0.68, 1.15)
        clean_axis(ax)
    axes[0].set_ylabel("정규화 방전용량(SOH)")
    handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=label)
        for label, color in LIFE_COLORS.items()
    ]
    handles.append(plt.Line2D([0], [0], color="#DC2626", lw=1.5, ls="--", label="EOL 80%"))
    axes[-1].legend(handles=handles, loc="lower left", fontsize=8)
    fig.suptitle("질문 2. 배치별 방전용량 열화 곡선", fontsize=18, fontweight="bold")
    save_figure(fig, figure_dir / "03_열화_곡선.png")


def plot_representative_degradation(
    cells: pd.DataFrame, degradation: pd.DataFrame, figure_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    quantiles = [(0.25, "하위 25%"), (0.50, "중앙"), (0.75, "상위 25%")]
    colors = ["#E45756", "#9CA3AF", "#4C78A8"]
    for ax, batch_name in zip(axes, BATCH_ORDER):
        batch_cells = cells[(cells["batch"] == batch_name) & cells["cycle_life"].notna()]
        for (quantile, label), color in zip(quantiles, colors):
            target = batch_cells["cycle_life"].quantile(quantile)
            idx = (batch_cells["cycle_life"] - target).abs().idxmin()
            cell_id = batch_cells.loc[idx, "global_cell_id"]
            life = batch_cells.loc[idx, "cycle_life"]
            curve = degradation[degradation["global_cell_id"] == cell_id]
            ax.plot(curve["cycle"], curve["soh"], color=color, linewidth=2.2, label=f"{label}: {life:.0f}회")
        ax.axhline(0.8, color="#DC2626", linestyle="--", linewidth=1.3)
        ax.set_title(batch_name)
        ax.set_xlabel("사이클")
        ax.set_ylim(0.68, 1.15)
        ax.legend(fontsize=8)
        clean_axis(ax)
    axes[0].set_ylabel("정규화 방전용량(SOH)")
    fig.suptitle("질문 2. 수명 분위수 대표 셀의 열화 패턴", fontsize=18, fontweight="bold")
    save_figure(fig, figure_dir / "04_대표_셀_열화.png")


def plot_knee(knees: pd.DataFrame, figure_dir: Path) -> None:
    valid = knees[knees["knee_available"] == True].copy()  # noqa: E712
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for batch_name in BATCH_ORDER:
        subset = valid[valid["batch"] == batch_name]
        axes[0].scatter(
            subset["cycle_life"],
            subset["knee_cycle"],
            s=42,
            alpha=0.72,
            color=BATCH_COLORS[batch_name],
            label=f"{batch_name} (n={len(subset)})",
        )
    limit = max(2000, valid[["cycle_life", "knee_cycle"]].max().max() * 1.03) if len(valid) else 2000
    axes[0].plot([0, limit], [0, limit], color="#94A3B8", linestyle="--", linewidth=1)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_title("급격 열화 시작점과 총수명")
    axes[0].set_xlabel("총수명(cycle)")
    axes[0].set_ylabel("급격 열화 시작점(cycle)")
    axes[0].legend(fontsize=8)
    clean_axis(axes[0])

    sns.boxplot(
        data=valid,
        x="batch",
        y="knee_fraction_of_life",
        order=BATCH_ORDER,
        hue="batch",
        palette=BATCH_COLORS,
        legend=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=valid,
        x="batch",
        y="knee_fraction_of_life",
        order=BATCH_ORDER,
        color="#1F2937",
        alpha=0.55,
        size=4,
        ax=axes[1],
    )
    axes[1].set_title("총수명 대비 급격 열화 시작점 위치")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("총수명 대비 비율")
    clean_axis(axes[1])
    fig.suptitle("질문 2. 2구간 회귀 기반 급격 열화 시작점", fontsize=18, fontweight="bold")
    save_figure(fig, figure_dir / "05_knee_point_분석.png")


def _interpolate_group_curves(group: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    curves = []
    for _, cell in group.groupby("global_cell_id", observed=True):
        cell = cell.sort_values("voltage")
        x = cell["voltage"].to_numpy()
        y = cell["delta_q"].to_numpy()
        unique_x, indices = np.unique(x, return_index=True)
        if len(unique_x) >= 20:
            curves.append(np.interp(grid, unique_x, y[indices]))
    return np.vstack(curves) if curves else np.empty((0, len(grid)))


def plot_delta_q(
    delta_curves: pd.DataFrame, cells: pd.DataFrame, figure_dir: Path
) -> None:
    merged = delta_curves.merge(
        cells[["global_cell_id", "batch", "cycle_life"]],
        on=["global_cell_id", "batch", "cycle_life"],
        how="left",
    )
    voltage_min, voltage_max = merged["voltage"].quantile([0.01, 0.99])
    grid = np.linspace(voltage_min, voltage_max, 250)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True, sharey=True)
    for ax, batch_name in zip(axes, BATCH_ORDER):
        batch_cells = cells[(cells["batch"] == batch_name) & cells["cycle_life"].notna()]
        low_cut = batch_cells["cycle_life"].quantile(0.25)
        high_cut = batch_cells["cycle_life"].quantile(0.75)
        batch_curves = merged[merged["batch"] == batch_name]
        for label, condition, color in [
            ("하위 25%", batch_curves["cycle_life"] <= low_cut, "#E45756"),
            ("상위 25%", batch_curves["cycle_life"] >= high_cut, "#4C78A8"),
        ]:
            curves = _interpolate_group_curves(batch_curves[condition], grid)
            if len(curves) == 0:
                continue
            median = np.nanmedian(curves, axis=0)
            q1, q3 = np.nanquantile(curves, [0.25, 0.75], axis=0)
            ax.plot(grid, median, color=color, linewidth=2.2, label=f"{label} (n={len(curves)})")
            ax.fill_between(grid, q1, q3, color=color, alpha=0.16)
        ax.axhline(0, color="#64748B", linewidth=0.8)
        ax.set_title(batch_name)
        ax.set_xlabel("전압(V)")
        ax.legend(fontsize=8)
        clean_axis(ax)
    axes[0].set_ylabel("ΔQ(V) [Ah]")
    fig.suptitle("질문 3. 100사이클 - 10사이클 ΔQ(V) 비교", fontsize=18, fontweight="bold")
    save_figure(fig, figure_dir / "06_deltaQ_곡선.png")


def plot_delta_correlations(
    early: pd.DataFrame, correlations: pd.DataFrame, figure_dir: Path
) -> None:
    feature = "delta_q_log10_var"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for batch_name in BATCH_ORDER:
        subset = early[early["batch"] == batch_name]
        axes[0].scatter(
            subset[feature],
            subset["cycle_life"],
            color=BATCH_COLORS[batch_name],
            s=42,
            alpha=0.70,
            label=batch_name,
        )
    axes[0].set_title("ΔQ(V) 로그 분산과 배터리 수명")
    axes[0].set_xlabel("log10[Var(ΔQ(V))]")
    axes[0].set_ylabel("수명(cycle)")
    axes[0].legend()
    clean_axis(axes[0])

    delta_features = [
        "delta_q_log10_var",
        "delta_q_min",
        "delta_q_l1",
        "delta_q_abs_area",
        "delta_q_iqr",
    ]
    corr_plot = correlations[
        (correlations["scope"].isin(["전체", *BATCH_ORDER]))
        & (correlations["feature"].isin(delta_features))
    ].copy()
    corr_plot["feature_label"] = corr_plot["feature"].map(
        lambda x: FEATURE_LABELS.get(x, x)
    )
    pivot = corr_plot.pivot(index="feature_label", columns="scope", values="spearman_r")
    pivot = pivot.reindex(columns=["전체", *BATCH_ORDER])
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=axes[1],
    )
    axes[1].set_title("ΔQ(V) 파생변수의 수명 상관")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle("질문 3. 초기 ΔQ(V) 신호의 설명력", fontsize=18, fontweight="bold")
    save_figure(fig, figure_dir / "07_deltaQ_상관.png")


def plot_charging_policy(
    cells: pd.DataFrame, policy_summary: pd.DataFrame, figure_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.45]})
    for batch_name in BATCH_ORDER:
        subset = cells[(cells["batch"] == batch_name) & cells["c_rate_stage1"].notna()]
        axes[0].scatter(
            subset["c_rate_stage1"],
            subset["cycle_life"],
            color=BATCH_COLORS[batch_name],
            s=46,
            alpha=0.72,
            label=batch_name,
        )
    axes[0].set_title("1단계 C-rate와 수명")
    axes[0].set_xlabel("1단계 충전 속도(C-rate)")
    axes[0].set_ylabel("수명(cycle)")
    axes[0].legend()
    clean_axis(axes[0])

    ranked = policy_summary.sort_values(["n", "mean_cycle_life"], ascending=[False, False]).head(16).copy()
    ranked = ranked.sort_values("mean_cycle_life")
    y = np.arange(len(ranked))
    colors = ranked["batch"].map(BATCH_COLORS)
    xerr = np.vstack(
        [
            ranked["mean_cycle_life"] - ranked["bootstrap_ci_low"],
            ranked["bootstrap_ci_high"] - ranked["mean_cycle_life"],
        ]
    )
    axes[1].errorbar(
        ranked["mean_cycle_life"],
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#94A3B8",
        capsize=3,
        linewidth=1.2,
    )
    axes[1].scatter(ranked["mean_cycle_life"], y, c=colors, s=55, zorder=3)
    labels = [f"{p} · {b} · n={n}" for p, b, n in zip(ranked["charging_policy"], ranked["batch"], ranked["n"])]
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_title("표본이 많은 충전 정책별 평균 수명과 부트스트랩 95% 신뢰구간")
    axes[1].set_xlabel("평균 수명(cycle)")
    axes[1].set_ylabel("")
    clean_axis(axes[1])
    fig.suptitle("질문 4. 충전 조건과 배터리 수명", fontsize=18, fontweight="bold")
    save_figure(fig, figure_dir / "08_충전_정책.png")


def plot_early_correlations(
    correlations: pd.DataFrame, early: pd.DataFrame, figure_dir: Path
) -> list[str]:
    pooled = correlations[correlations["scope"] == "전체"].sort_values("abs_spearman", ascending=False)
    top_features = pooled["feature"].drop_duplicates().head(14).tolist()
    corr_plot = correlations[
        correlations["feature"].isin(top_features)
        & correlations["scope"].isin(["전체", *BATCH_ORDER])
    ].copy()
    corr_plot["label"] = corr_plot["feature"].map(lambda x: FEATURE_LABELS.get(x, x))
    pivot = corr_plot.pivot(index="label", columns="scope", values="spearman_r")
    order_labels = [FEATURE_LABELS.get(x, x) for x in top_features]
    pivot = pivot.reindex(index=order_labels, columns=["전체", *BATCH_ORDER])
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Spearman ρ"},
        ax=ax,
    )
    ax.set_title("질문 5. 초기 신호와 수명의 Spearman 상관계수")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, figure_dir / "09_초기_신호_상관.png")

    available = [f for f in top_features if f in early and early[f].notna().sum() >= 20]
    if available:
        matrix = early[available].corr(method="spearman")
        matrix.index = [FEATURE_LABELS.get(x, x) for x in matrix.index]
        matrix.columns = [FEATURE_LABELS.get(x, x) for x in matrix.columns]
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        sns.heatmap(
            matrix,
            mask=mask,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            cbar_kws={"label": "파생변수 간 Spearman ρ"},
            ax=ax,
        )
        ax.set_title("초기 파생변수 간 다중공선성 점검")
        save_figure(fig, figure_dir / "10_다중공선성.png")
    return top_features


def plot_batch_shift(batch_shift: pd.DataFrame, figure_dir: Path) -> None:
    if batch_shift.empty:
        return
    top = (
        batch_shift.assign(abs_smd=batch_shift["standardized_mean_difference"].abs())
        .groupby("feature", observed=True)["abs_smd"]
        .max()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    pivot = (
        batch_shift[batch_shift["feature"].isin(top)]
        .pivot(index="feature", columns="comparison", values="standardized_mean_difference")
        .reindex(top)
    )
    pivot.index = [FEATURE_LABELS.get(x, x) for x in pivot.index]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "표준화 평균차(SMD)"},
        ax=ax,
    )
    ax.set_title("Batch 1 대비 초기 파생변수 분포 이동")
    ax.set_xlabel("")
    ax.set_ylabel("파생변수")
    save_figure(fig, figure_dir / "11_배치_시프트.png")


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 3) -> str:
    view = frame.copy()
    if columns is not None:
        view = view[columns]
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
    headers = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[c]).replace("|", "\\|") for c in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_report(
    paths: OutputPaths,
    frames: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
    decisions: pd.DataFrame,
    font_name: str,
) -> Path:
    batch_summary = summaries["batch_summary"]
    quality = summaries["quality_audit"]
    correlations = summaries["correlation_results"]
    tests = summaries["statistical_tests"]
    outliers = summaries["outlier_cells"]
    cells = frames["cells"]
    knees = frames["knees"]

    if correlations.empty:
        top_corr = correlations
        top_corr_table = pd.DataFrame()
    else:
        top_features = (
            correlations[correlations["scope"] == "Batch 1"]
            .sort_values("abs_spearman", ascending=False)["feature"]
            .head(10)
            .tolist()
        )
        top_corr = correlations[
            (correlations["scope"] == "전체")
            & (correlations["feature"].isin(top_features))
        ]
        top_corr_table = (
            correlations[correlations["feature"].isin(top_features)]
            .pivot(index="feature", columns="scope", values="spearman_r")
            .reindex(index=top_features, columns=["Batch 1", "Batch 2", "Batch 3", "전체"])
            .reset_index()
            .rename(
                columns={
                    "feature": "Feature",
                    "Batch 1": "Batch 1 ρ",
                    "Batch 2": "Batch 2 ρ",
                    "Batch 3": "Batch 3 ρ",
                    "전체": "전체 ρ",
                }
            )
        )
        top_corr_table["Feature"] = top_corr_table["Feature"].map(
            lambda x: FEATURE_LABELS.get(x, x)
        )
    kw = tests[tests["test"] == "Kruskal-Wallis"].iloc[0]
    knee_test = tests[tests["analysis"] == "knee point와 수명"]
    knee_rho = float(knee_test.iloc[0]["statistic"]) if len(knee_test) else np.nan
    delta_row = top_corr[top_corr["feature"] == "delta_q_log10_var"]
    delta_rho = float(delta_row.iloc[0]["spearman_r"]) if len(delta_row) else np.nan
    delta_b1_row = correlations[
        (correlations["scope"] == "Batch 1")
        & (correlations["feature"] == "delta_q_log10_var")
    ]
    delta_b1_rho = (
        float(delta_b1_row.iloc[0]["spearman_r"]) if len(delta_b1_row) else np.nan
    )
    c_rate_test = tests[
        (tests["analysis"] == "충전 조건과 수명") & (tests["comparison"] == "c_rate_stage1")
    ]
    c_rate_rho = float(c_rate_test.iloc[0]["statistic"]) if len(c_rate_test) else np.nan
    c_rate_corr_table = correlations[
        (correlations["feature"] == "c_rate_stage1")
        & (correlations["scope"].isin(["전체", *BATCH_ORDER]))
    ][["scope", "n", "spearman_r", "spearman_p_bh"]].rename(
        columns={
            "scope": "범위",
            "n": "n",
            "spearman_r": "Spearman ρ",
            "spearman_p_bh": "BH 보정 p",
        }
    )
    delta_group_tests = tests[tests["analysis"] == "ΔQ 장·단수명 분위수 비교"][
        ["comparison", "statistic", "p_value", "effect_or_distance"]
    ].rename(
        columns={
            "comparison": "배치",
            "statistic": "Mann-Whitney U",
            "p_value": "p-value",
            "effect_or_distance": "순위양분 상관",
        }
    )
    valid_knees = knees[knees["knee_available"] == True]  # noqa: E712
    early_knee_count = int((valid_knees["knee_fraction_of_life"] < 0.5).sum())
    knee_summary = valid_knees.copy()
    knee_summary["slope_acceleration"] = (
        knee_summary["post_knee_slope"].abs()
        / knee_summary["pre_knee_slope"].abs().replace(0, np.nan)
    )
    knee_summary = (
        knee_summary.groupby("batch", observed=True)
        .agg(
            cells=("global_cell_id", "count"),
            median_knee_cycle=("knee_cycle", "median"),
            median_knee_fraction=("knee_fraction_of_life", "median"),
            median_slope_acceleration=("slope_acceleration", "median"),
        )
        .reindex(BATCH_ORDER)
        .reset_index()
        .rename(
            columns={
                "batch": "배치",
                "cells": "분석 셀 수",
                "median_knee_cycle": "중앙 knee cycle",
                "median_knee_fraction": "수명 대비 중앙 위치",
                "median_slope_acceleration": "knee 이후 기울기 배수",
            }
        )
    )
    outlier_new_structure = (
        int(outliers["charging_policy"].str.contains("newstructure", case=False, na=False).sum())
        if not outliers.empty
        else 0
    )

    batch_table = batch_summary.rename(
        columns={
            "batch": "배치",
            "cell_count": "셀 수",
            "mean": "평균",
            "median": "중앙값",
            "std": "표준편차",
            "min": "최솟값",
            "q1": "Q1",
            "q3": "Q3",
            "max": "최댓값",
            "skew": "왜도",
        }
    )
    quality_table = quality.rename(
        columns={
            "batch": "배치",
            "cells": "셀 수",
            "labeled_cells": "Target 보유 셀",
            "missing_target_cells": "Target 결측 셀",
            "first_cycle_zero_cells": "첫 사이클 0 셀",
            "delta_q_available_pct": "ΔQ 가용률(%)",
            "knee_eligible_pct_labeled": "Target 보유 셀 중 knee 분석률(%)",
            "policy_parse_success_pct": "정책 파싱률(%)",
            "cells_with_nan_inf": "NaN/Inf 셀",
            "early_ir_unavailable_cells": "초기 IR 미가용 셀",
            "cells_with_early_tmin_invalid": "초기 Tmin 이상 셀",
        }
    )
    outlier_table = outliers.rename(
        columns={
            "batch": "배치",
            "global_cell_id": "셀 ID",
            "cycle_life": "수명",
            "charging_policy": "충전 정책",
            "outlier_direction": "방향",
            "tavg_mean": "초기 평균온도(°C)",
            "chargetime_mean": "초기 평균 충전시간",
            "c_rate_stage1": "1단계 C-rate",
            "delta_q_log10_var": "ΔQ 로그분산",
        }
    )
    labeled_cells = int(cells["cycle_life"].notna().sum())
    missing_targets = int(cells["cycle_life"].isna().sum())

    report = f"""# Day 1 배터리 수명 예측 EDA 보고서

## 1. 분석 목적

초기 100사이클 데이터로 `cycle_life`를 예측하는 Day 2 Regression 모델을 설계하기 위해 Batch 1·2·3을 동일한 기준으로 분석했다. 원본 MATLAB v7.3 파일은 메모리에 일괄 적재하지 않고 HDF5 reference를 순차 접근했다. 모든 그래프는 한국어와 물리 단위를 표기했으며 사용 폰트는 `{font_name}`이다.

## 2. 핵심 결론

- 원본 총 {len(cells)}개 셀을 품질 감사했고, target이 있는 {labeled_cells}개 셀을 수명 분석에 사용했다: {', '.join(f'{row.batch} {int(row.cell_count)}개' for row in batch_summary.itertuples())}. `cycle_life`가 결측인 {missing_targets}개 셀은 target 기반 분석에서 제외했으며 기록 길이로 임의 대체하지 않았다.
- 배치별 수명 분포 차이에 대한 Kruskal-Wallis 검정은 H={kw['statistic']:.3f}, p={kw['p_value']:.3g}였다. 이는 배치 간 분포 이동을 모델 일반화 위험으로 관리해야 함을 뜻한다.
- 전체 수명 곡선으로 계산한 knee point와 `cycle_life`의 Spearman ρ는 {knee_rho:.3f}였다. 다만 knee point는 미래 수명 정보를 사용하므로 Day 2 입력 feature에서는 제외한다.
- `ΔQ(V)` 로그 분산과 `cycle_life`의 Spearman ρ는 실제 학습 배치인 Batch 1에서 {delta_b1_rho:.3f}, 전체에서 {delta_rho:.3f}였다. cycle 10·100만 사용하므로 Day 2 핵심 후보로 채택한다.
- 1단계 C-rate와 수명의 전체 Spearman ρ는 {c_rate_rho:.3f}였다. 충전 정책별 표본이 작고 배치·구조가 혼재하므로 인과관계가 아닌 연관성으로 해석한다.
- Batch 2·3 label을 이용한 결과는 외부 분포 감사용이다. 실제 feature·모델 선택은 Batch 1 내부 CV와 hold-out을 중심으로 수행한다.

## 3. 데이터 품질 감사

{markdown_table(quality_table, ['배치', '셀 수', 'Target 보유 셀', 'Target 결측 셀', '첫 사이클 0 셀', '초기 IR 미가용 셀', '초기 Tmin 이상 셀', 'ΔQ 가용률(%)', 'Target 보유 셀 중 knee 분석률(%)', '정책 파싱률(%)'], 1)}

![배치별 데이터 가용률](../figures/day1/00_데이터_품질_가용률.png)

### 관찰

Batch 1의 모든 셀은 첫 사이클의 summary 값이 0이므로 초기 통계는 실제 cycle 2~100의 물리적으로 유효한 값으로 계산했다. Batch 2에는 `cycle_life` 결측 셀과 초기 IR이 전부 0인 셀이 있으며 0은 실제 저항이 아닌 결측 표현으로 처리했다. Target 결측 셀은 기록 길이로 수명을 대체하지 않았고 target 기반 분석과 knee point 분석에서 제외했다.

### 모델링 영향

Day 2 pipeline은 0값을 정상 측정값으로 평균에 포함하지 않아야 한다. IR은 결측 flag와 train-only imputation을 사용하고, ΔQ(V) 미가용 여부도 별도 flag로 보존한다.

## 4. 질문 1 — Cycle Life 분포

{markdown_table(batch_table, ['배치', '셀 수', '평균', '중앙값', '표준편차', '최솟값', 'Q1', 'Q3', '최댓값', '왜도'], 1)}

![수명 분포](../figures/day1/01_수명_분포.png)

![수명 구간 비율](../figures/day1/02_수명_구간_비율.png)

### 관찰

배치별 중앙값·분산·상한이 다르며 Batch 3은 장수명 꼬리가 상대적으로 길다. Tukey 1.5 IQR 기준 이상치는 {len(outliers)}개였으며 삭제하지 않고 별도 사례로 보존했다. 이상치 중 `newstructure` 정책 표기가 있는 셀은 {outlier_new_structure}개였지만 구조 효과와 배치 효과가 겹치므로 이를 단독 원인으로 단정하지 않는다.

{markdown_table(outlier_table, ['배치', '셀 ID', '수명', '충전 정책', '방향', '초기 평균온도(°C)', '초기 평균 충전시간', '1단계 C-rate', 'ΔQ 로그분산'], 3) if not outlier_table.empty else '이상치 없음'}

### 원인 가설

충전 프로토콜, 셀 구조, 실험 시기와 측정 길이 차이가 함께 작용했을 가능성이 있다. 관찰 자료만으로 특정 요인의 인과효과를 단정하지 않는다.

### Day 2 영향

Batch 1만으로 선택한 모델은 Batch 2·3에서 target 및 feature shift를 겪을 수 있으므로 Batch 1 CV, 고정 hold-out, Batch 2 외부 테스트를 분리한다.

## 5. 질문 2 — 방전용량 열화와 knee point

![전체 열화곡선](../figures/day1/03_열화_곡선.png)

![대표 셀 열화곡선](../figures/day1/04_대표_셀_열화.png)

![knee point](../figures/day1/05_knee_point_분석.png)

{markdown_table(knee_summary, digits=3)}

### 관찰

초기 cycle 2~10의 방전용량 중앙값을 셀별 기준 용량으로 두고 SOH를 계산했다. `cycle_life`가 있고 전체 수명의 80% 이상이 관측된 {len(valid_knees)}개 셀에서 9-point rolling median과 2구간 선형회귀로 knee point를 계산했다. 이 중 총수명의 50% 이전이 최적 분할점으로 나온 셀은 {early_knee_count}개로, 자동 탐지 결과의 민감도 사례로 별도 확인해야 한다.

### 원인 가설

초기 완만한 열화 뒤 용량 감소가 가속되는 패턴은 내부 열화 누적과 연관될 수 있으나, 본 분석만으로 전기화학적 인과기전을 확정할 수 없다.

### Day 2 영향 및 feature 결정

knee point와 전체 열화곡선은 현상을 설명하는 탐색적 지표로만 사용한다. 자동 분할점이 전기화학적 knee의 정답이라고 단정하지 않으며, 미래 사이클을 포함하므로 Day 2 예측 feature에서는 명시적으로 제외한다.

## 6. 질문 3 — ΔQ(V) 초기 신호

![ΔQ(V) 곡선](../figures/day1/06_deltaQ_곡선.png)

![ΔQ(V) 상관](../figures/day1/07_deltaQ_상관.png)

{markdown_table(delta_group_tests, digits=3)}

### 관찰

실제 summary cycle 번호로 cycle 10·100의 위치를 찾고, 공통 `Vdlin`에서 `Qdlin(100)-Qdlin(10)`을 계산했다. 배치 내 상·하위 수명 25%의 중앙곡선과 IQR을 비교해 고정 임계값 그룹이 비어 생기는 문제를 피했다.

### Day 2 영향 및 feature 결정

`delta_q_log10_var`, `delta_q_min`, 절대면적 등은 초기 100사이클 내에서 계산할 수 있다. 상관이 높은 ΔQ feature를 모두 넣기보다 Batch 1 CV에서 대표 feature를 선택하거나 규제를 적용한다.

## 7. 질문 4 — 충전 조건과 수명

![충전 정책](../figures/day1/08_충전_정책.png)

{markdown_table(c_rate_corr_table, digits=3)}

### 관찰

충전 정책 문자열에서 1단계 C-rate, 전환 SOC, 2단계 C-rate를 분리했다. 정책별 n과 bootstrap 95% CI를 함께 표시했으며 n=1~2인 정책의 평균은 인과효과가 아닌 탐색적 경향으로만 해석한다.

### Day 2 영향 및 feature 결정

원문 정책 one-hot은 미지 정책에서 취약할 수 있다. 연속형 C-rate·전환 SOC feature를 우선 사용하고, 범주형 정책은 `handle_unknown='ignore'`가 적용된 보조 feature로 검토한다.

## 8. 질문 5 — 초기 신호와 수명

{markdown_table(top_corr_table, digits=3)}

![초기 신호 상관](../figures/day1/09_초기_신호_상관.png)

![다중공선성](../figures/day1/10_다중공선성.png)

![배치 시프트](../figures/day1/11_배치_시프트.png)

### 관찰

Pearson과 Spearman을 함께 계산하고, 다중 검정은 Benjamini-Hochberg 방식으로 보정했다. 온도 계열과 ΔQ 계열처럼 서로 강하게 연관된 feature 군은 대표 변수를 선택하거나 Ridge/ElasticNet 규제를 사용해야 한다.

### 모델링 영향

배치 전체 상관은 배치 간 평균 차이에 의해 부풀려질 수 있다. 따라서 전체 상관과 배치별 상관의 방향·크기를 함께 보고, Day 2의 실제 선택은 Batch 1 CV 안정성을 우선한다.

## 9. 최종 feature 채택표

{markdown_table(decisions.rename(columns={'feature':'Feature', 'source':'원본', 'cycle_range':'사이클 범위', 'formula':'계산식', 'missing_handling':'결측 처리', 'pooled_spearman':'전체 Spearman ρ', 'batch_1_spearman':'Batch 1 ρ', 'batch_2_spearman':'Batch 2 ρ', 'batch_3_spearman':'Batch 3 ρ', 'batch_direction_consistency':'배치 방향성', 'availability_pct':'가용률(%)', 'decision':'결정', 'reason':'근거', 'leakage_risk':'누수 위험'}), ['Feature', '원본', '사이클 범위', '계산식', 'Batch 1 ρ', 'Batch 2 ρ', 'Batch 3 ρ', '배치 방향성', '가용률(%)', '결정', '근거', '누수 위험'], 3)}

## 10. Day 2 모델 전략

1. **분석 단위:** 배터리 셀 1개당 feature 행 1개.
2. **학습·검증:** Batch 1을 fixed hold-out으로 먼저 분리하고 나머지에서 K-fold CV.
3. **외부 테스트:** 모든 선택 종료 후 Batch 2를 한 번 평가. Batch 3은 선택적 추가 테스트.
4. **Baseline:** Batch 1 train 중앙값 예측.
5. **선형 후보:** Ridge/ElasticNet — 작은 표본, 공선성 대응.
6. **비선형 후보:** Gradient Boosting/Random Forest — ΔQ, 온도, 충전조건의 비선형 상호작용 대응.
7. **선택 기준:** Batch 1 CV 평균 MAPE, fold 편차, hold-out Gap, 설명 가능성.
8. **전처리:** imputer·scaler·encoder를 모델과 하나의 sklearn Pipeline으로 구성하고 각 CV train fold에서만 fit.

## 11. 한계

- 배치별 셀 수가 작고 충전 정책별 표본은 더 작아 정책 효과의 인과 추론이 어렵다.
- Batch 2의 8개 셀과 Batch 3의 2개 셀은 `cycle_life`가 결측이므로 수명 분포·상관·knee 분석에서 제외했다. 이들이 중도절단된 장수명 셀이라면 관측된 target 분포에 편향이 남을 수 있다.
- 실험실 셀 데이터는 실제 ESS module/rack의 온도 구배, 셀 불균형, 운영 부하를 완전히 대표하지 않는다.
- Batch 2·3 label은 과제상 분포 설명에 사용했으나 Day 2 모델 튜닝에는 사용하지 않아야 한다.

## 12. 생성 산출물

- 재현 코드: `src/day1_analysis.py`
- 실행 노트북: `notebooks/01_EDA.ipynb`
- 수치 결과: `results/day1/*.csv`
- 한국어 그래프: `figures/day1/*.png`
- 본 보고서: `reports/DAY1_분석_보고서.md`
"""
    report_path = paths.report_dir / "DAY1_분석_보고서.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def save_results(
    paths: OutputPaths,
    frames: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
    decisions: pd.DataFrame,
) -> None:
    save_csv(frames["early_features"], paths.result_dir / "early_features.csv")
    save_csv(frames["delta_features"], paths.result_dir / "delta_q_features.csv")
    save_csv(frames["knees"], paths.result_dir / "knee_points.csv")
    save_csv(frames["quality"], paths.result_dir / "quality_by_cell.csv")
    for name, frame in summaries.items():
        save_csv(frame, paths.result_dir / f"{name}.csv")
    save_csv(decisions, paths.result_dir / "feature_decisions.csv")


def make_plots(
    paths: OutputPaths,
    frames: dict[str, pd.DataFrame],
    summaries: dict[str, pd.DataFrame],
) -> None:
    plot_quality(summaries["quality_audit"], paths.figure_dir)
    plot_lifetime_distribution(frames["cells"], paths.figure_dir)
    plot_lifetime_groups(frames["cells"], summaries["lifetime_group_rates"], paths.figure_dir)
    plot_degradation(frames["degradation"], paths.figure_dir)
    plot_representative_degradation(frames["cells"], frames["degradation"], paths.figure_dir)
    plot_knee(frames["knees"], paths.figure_dir)
    plot_delta_q(frames["delta_curves"], frames["cells"], paths.figure_dir)
    plot_delta_correlations(frames["early_features"], summaries["correlation_results"], paths.figure_dir)
    plot_charging_policy(frames["cells"], summaries["policy_summary"], paths.figure_dir)
    plot_early_correlations(summaries["correlation_results"], frames["early_features"], paths.figure_dir)
    plot_batch_shift(summaries["batch_shift"], paths.figure_dir)


def run_analysis(root: Path | str = PROJECT_ROOT) -> dict[str, str | int]:
    root = Path(root).resolve()
    paths = ensure_paths(root)
    font_name = configure_korean_plotting()
    frames = extract_batches(paths)
    summaries = summarize_analysis(frames)
    decisions = build_feature_decisions(summaries["correlation_results"], frames["early_features"])
    save_results(paths, frames, summaries, decisions)
    make_plots(paths, frames, summaries)
    report_path = make_report(paths, frames, summaries, decisions, font_name)

    manifest = {
        "random_seed": RANDOM_SEED,
        "font": font_name,
        "raw_cell_count": int(len(frames["cells"])),
        "labeled_cell_count": int(frames["cells"]["cycle_life"].notna().sum()),
        "batch_files": BATCH_FILES,
        "report": str(report_path.relative_to(root)),
        "figure_count": len(list(paths.figure_dir.glob("*.png"))),
        "result_table_count": len(list(paths.result_dir.glob("*.csv"))),
    }
    manifest_path = paths.result_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 1 ESS 배터리 EDA 실행")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="프로젝트 루트")
    args = parser.parse_args()
    run_analysis(args.root)


if __name__ == "__main__":
    main()
