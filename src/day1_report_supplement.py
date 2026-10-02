#!/usr/bin/env python3
"""Day 1 보고서 보강용 진단 분석.

기존 EDA가 생성한 cell-level feature 표를 다시 읽어 다음 세 가지를 재현한다.
1) Batch 2의 500 cycle 미만 저수명 군집 비교
2) 충전 조건과 초기 열화 proxy의 Batch별/Batch 보정 상관
3) Batch 1 기준 고상관 feature 묶음과 대표 후보

원본 데이터는 변경하지 않으며, 결과 CSV와 한국어 SVG figure만 추가한다.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "day1"
FIGURES = ROOT / "figures" / "day1"
RNG_SEED = 20261002


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 4 or frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return np.nan, len(frame)
    ranked = frame.rank(method="average")
    return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1])), len(frame)


def permutation_spearman_p(
    x: pd.Series, y: pd.Series, *, n_perm: int = 5000, seed: int = RNG_SEED
) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 4:
        return np.nan
    rx = frame.iloc[:, 0].rank(method="average").to_numpy(float)
    ry = frame.iloc[:, 1].rank(method="average").to_numpy(float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom == 0:
        return np.nan
    observed = abs(float(np.sum(rx * ry) / denom))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        permuted = rng.permutation(ry)
        rho = abs(float(np.sum(rx * permuted) / denom))
        extreme += rho >= observed - 1e-15
    return (extreme + 1) / (n_perm + 1)


def permutation_median_p(
    a: np.ndarray, b: np.ndarray, *, n_perm: int = 10000, seed: int = RNG_SEED
) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    observed = abs(float(np.median(a) - np.median(b)))
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = abs(float(np.median(perm[: len(a)]) - np.median(perm[len(a) :])))
        extreme += diff >= observed - 1e-15
    return (extreme + 1) / (n_perm + 1)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a>b)-P(a<b). 양수면 첫 번째 집단 값이 더 크다."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0, 1)
    return pd.Series(adjusted, index=p_values.index)


def partial_spearman_by_batch(df: pd.DataFrame, x_name: str, y_name: str) -> tuple[float, int]:
    frame = df[["batch", x_name, y_name]].dropna().copy()
    if len(frame) < 6:
        return np.nan, len(frame)
    x = frame[x_name].rank(method="average").to_numpy(float)
    y = frame[y_name].rank(method="average").to_numpy(float)
    dummies = pd.get_dummies(frame["batch"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(frame)), dummies])
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return np.nan, len(frame)
    return float(np.corrcoef(x_resid, y_resid)[0, 1]), len(frame)


def partial_permutation_p(
    df: pd.DataFrame, x_name: str, y_name: str, *, n_perm: int = 5000, seed: int = RNG_SEED
) -> float:
    frame = df[["batch", x_name, y_name]].dropna().copy()
    if len(frame) < 6:
        return np.nan
    x = frame[x_name].rank(method="average").to_numpy(float)
    y = frame[y_name].rank(method="average").to_numpy(float)
    dummies = pd.get_dummies(frame["batch"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(frame)), dummies])
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    denom = np.sqrt(np.sum(x_resid**2) * np.sum(y_resid**2))
    if denom == 0:
        return np.nan
    observed = abs(float(np.sum(x_resid * y_resid) / denom))
    rng = np.random.default_rng(seed)
    extreme = 0
    batches = frame["batch"].to_numpy()
    for _ in range(n_perm):
        permuted = y_resid.copy()
        for batch in np.unique(batches):
            idx = np.flatnonzero(batches == batch)
            permuted[idx] = rng.permutation(permuted[idx])
        rho = abs(float(np.sum(x_resid * permuted) / denom))
        extreme += rho >= observed - 1e-15
    return (extreme + 1) / (n_perm + 1)


LOW_LIFE_FEATURES = {
    "delta_q_log10_var": "ΔQ 로그분산",
    "qd_slope": "초기 QD 기울기",
    "qd_std": "초기 QD 변동성",
    "ir_std": "초기 IR 변동성",
    "tavg_mean": "평균 온도",
    "tmax_mean": "최고 온도",
    "chargetime_mean": "평균 충전시간",
    "c_rate_stage1": "1단계 C-rate",
    "switch_soc_pct": "전환 SOC",
}


def analyze_low_life(df: pd.DataFrame) -> pd.DataFrame:
    b2 = df.loc[(df["batch"] == "Batch 2") & df["cycle_life"].notna()].copy()
    short = b2["cycle_life"] < 500
    rows: list[dict[str, object]] = []
    for idx, (feature, label) in enumerate(LOW_LIFE_FEATURES.items()):
        a = b2.loc[short, feature].to_numpy(float)
        b = b2.loc[~short, feature].to_numpy(float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        rows.append(
            {
                "batch": "Batch 2",
                "comparison": "cycle_life < 500 vs >= 500",
                "feature": feature,
                "feature_ko": label,
                "n_short": len(a),
                "n_other": len(b),
                "median_short": np.median(a) if len(a) else np.nan,
                "median_other": np.median(b) if len(b) else np.nan,
                "cliffs_delta": cliffs_delta(a, b),
                "permutation_p": permutation_median_p(a, b, seed=RNG_SEED + idx),
            }
        )
    out = pd.DataFrame(rows)
    out["p_bh"] = bh_adjust(out["permutation_p"])
    return out.sort_values("cliffs_delta", key=lambda x: x.abs(), ascending=False)


CHARGING_FEATURES = {
    "c_rate_stage1": "1단계 C-rate",
    "switch_soc_pct": "전환 SOC",
    "c_rate_stage2": "2단계 C-rate",
}
DEGRADATION_OUTCOMES = {
    "cycle_life": "Cycle Life",
    "delta_q_log10_var": "ΔQ 로그분산",
    "qd_slope": "초기 QD 기울기",
}


def analyze_charging(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = [("전체", df)] + [(b, df[df["batch"] == b]) for b in ["Batch 1", "Batch 2", "Batch 3"]]
    seed_offset = 0
    for feature, feature_ko in CHARGING_FEATURES.items():
        for outcome, outcome_ko in DEGRADATION_OUTCOMES.items():
            for scope, frame in scopes:
                rho, n = spearman(frame[feature], frame[outcome])
                p = permutation_spearman_p(
                    frame[feature], frame[outcome], seed=RNG_SEED + seed_offset
                )
                seed_offset += 1
                rows.append(
                    {
                        "scope": scope,
                        "feature": feature,
                        "feature_ko": feature_ko,
                        "outcome": outcome,
                        "outcome_ko": outcome_ko,
                        "n": n,
                        "spearman_r": rho,
                        "permutation_p": p,
                    }
                )
            rho, n = partial_spearman_by_batch(df, feature, outcome)
            p = partial_permutation_p(df, feature, outcome, seed=RNG_SEED + seed_offset)
            seed_offset += 1
            rows.append(
                {
                    "scope": "배치 보정",
                    "feature": feature,
                    "feature_ko": feature_ko,
                    "outcome": outcome,
                    "outcome_ko": outcome_ko,
                    "n": n,
                    "spearman_r": rho,
                    "permutation_p": p,
                }
            )
    out = pd.DataFrame(rows)
    out["p_bh_within_scope"] = out.groupby("scope")["permutation_p"].transform(bh_adjust)
    return out


REDUNDANCY_CANDIDATES = [
    "delta_q_log10_var", "delta_q_var", "delta_q_std", "delta_q_iqr",
    "delta_q_l1", "delta_q_abs_area", "delta_q_l2", "delta_q_area", "delta_q_mean",
    "delta_q_min", "delta_q_q05", "delta_q_q25",
    "qd_mean", "qd_median", "qd_min", "qd_max", "qd_std", "qd_delta", "qd_slope",
    "qc_mean", "qc_median", "qc_min", "qc_max", "qc_std", "qc_delta", "qc_slope",
    "ir_mean", "ir_median", "ir_min", "ir_max", "ir_std", "ir_delta", "ir_slope",
    "tavg_mean", "tavg_median", "tavg_min", "tavg_max", "tavg_std", "tavg_delta", "tavg_slope",
    "tmax_mean", "tmax_median", "tmax_min", "tmax_max", "tmax_std", "tmax_delta", "tmax_slope",
    "chargetime_mean", "chargetime_median", "chargetime_min", "chargetime_max",
    "chargetime_std", "chargetime_delta", "chargetime_slope",
    "c_rate_stage1", "switch_soc_pct", "c_rate_stage2",
]

PREFERRED_REPRESENTATIVES = [
    "delta_q_log10_var", "delta_q_min", "delta_q_area", "qd_mean", "qd_slope",
    "qc_mean", "qc_slope", "ir_mean", "ir_std", "tavg_mean", "tavg_slope",
    "tmax_mean", "tmax_slope", "chargetime_mean", "chargetime_slope",
    "c_rate_stage1", "switch_soc_pct", "c_rate_stage2",
]


def analyze_redundancy(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    train = df.loc[(df["batch"] == "Batch 1") & df["cycle_life"].notna()].copy()
    features = [c for c in REDUNDANCY_CANDIDATES if c in train.columns and train[c].notna().sum() >= 30]
    corr = train[features].corr(method="spearman")
    adjacency = {feature: set() for feature in features}
    for i, first in enumerate(features):
        for second in features[i + 1 :]:
            value = corr.loc[first, second]
            if np.isfinite(value) and abs(value) >= threshold:
                adjacency[first].add(second)
                adjacency[second].add(first)

    seen: set[str] = set()
    components: list[list[str]] = []
    for feature in features:
        if feature in seen or not adjacency[feature]:
            continue
        stack = [feature]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacency[current] - seen)
        if len(component) > 1:
            components.append(sorted(component))

    rows: list[dict[str, object]] = []
    for group_index, members in enumerate(sorted(components, key=lambda x: (-len(x), x)), start=1):
        preferred = [feature for feature in PREFERRED_REPRESENTATIVES if feature in members]
        representative = preferred[0] if preferred else members[0]
        pair_values = []
        for i, first in enumerate(members):
            for second in members[i + 1 :]:
                value = abs(float(corr.loc[first, second]))
                if np.isfinite(value):
                    pair_values.append(value)
        rho_target, _ = spearman(train[representative], train["cycle_life"])
        rows.append(
            {
                "group_id": f"R{group_index:02d}",
                "n_features": len(members),
                "representative": representative,
                "members": " | ".join(members),
                "min_abs_pairwise_rho": min(pair_values) if pair_values else np.nan,
                "max_abs_pairwise_rho": max(pair_values) if pair_values else np.nan,
                "representative_target_spearman_batch1": rho_target,
                "selection_basis": "Batch 1 내부 | |Spearman|≥0.90 연결군 | 해석성과 안정성 우선",
            }
        )
    return pd.DataFrame(rows)


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:'Apple SD Gothic Neo','Noto Sans KR','Malgun Gothic',sans-serif;fill:#172B3A}",
        ".title{font-size:28px;font-weight:700}.subtitle{font-size:15px;fill:#526674}",
        ".label{font-size:14px}.small{font-size:12px;fill:#526674}.value{font-size:13px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="56" y="52" class="title">{html.escape(title)}</text>',
    ]


def render_low_life_svg(table: pd.DataFrame, path: Path) -> None:
    plot = table.sort_values("cliffs_delta").reset_index(drop=True)
    width, height = 1120, 610
    left, right, top, bottom = 275, 90, 116, 88
    plot_width = width - left - right
    row_h = (height - top - bottom) / len(plot)
    x = lambda value: left + (float(value) + 1) / 2 * plot_width
    parts = svg_header(width, height, "Batch 2 저수명 군집: 초기 신호의 방향과 크기")
    parts.append('<text x="56" y="80" class="subtitle">Cliff’s δ: 양수는 500 cycle 미만 군집에서 더 큼, 음수는 더 작음 · 기술적 연관이며 인과효과가 아님</text>')
    for tick in [-1, -0.5, 0, 0.5, 1]:
        tx = x(tick)
        color = "#9FB0BC" if tick == 0 else "#E5EBEF"
        stroke = 2 if tick == 0 else 1
        parts.append(f'<line x1="{tx:.1f}" y1="{top-12}" x2="{tx:.1f}" y2="{height-bottom+4}" stroke="{color}" stroke-width="{stroke}"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height-bottom+28}" class="small" text-anchor="middle">{tick:g}</text>')
    for index, row in plot.iterrows():
        cy = top + index * row_h + row_h / 2
        value = float(row["cliffs_delta"])
        x0, x1 = x(0), x(value)
        color = "#E67E22" if value > 0 else "#167D8D"
        parts.append(f'<text x="{left-18}" y="{cy+5:.1f}" class="label" text-anchor="end">{html.escape(str(row["feature_ko"]))}</text>')
        parts.append(f'<rect x="{min(x0,x1):.1f}" y="{cy-10:.1f}" width="{abs(x1-x0):.1f}" height="20" rx="5" fill="{color}" opacity="0.88"/>')
        anchor = "start" if value >= 0 else "end"
        offset = 9 if value >= 0 else -9
        parts.append(f'<text x="{x1+offset:.1f}" y="{cy+5:.1f}" class="value" text-anchor="{anchor}">{value:+.2f}</text>')
    parts.append(f'<text x="{left}" y="{height-24}" class="small">← 저수명 군집에서 더 작음</text>')
    parts.append(f'<text x="{width-right}" y="{height-24}" class="small" text-anchor="end">저수명 군집에서 더 큼 →</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_charging_svg(table: pd.DataFrame, path: Path) -> None:
    pairs = [
        ("c_rate_stage1", "cycle_life"),
        ("c_rate_stage1", "delta_q_log10_var"),
        ("c_rate_stage1", "qd_slope"),
        ("switch_soc_pct", "cycle_life"),
        ("c_rate_stage2", "cycle_life"),
    ]
    scopes = ["전체", "Batch 1", "Batch 2", "Batch 3", "배치 보정"]
    labels = {
        ("c_rate_stage1", "cycle_life"): "1단계 C-rate → 수명",
        ("c_rate_stage1", "delta_q_log10_var"): "1단계 C-rate → ΔQ 로그분산",
        ("c_rate_stage1", "qd_slope"): "1단계 C-rate → QD 기울기",
        ("switch_soc_pct", "cycle_life"): "전환 SOC → 수명",
        ("c_rate_stage2", "cycle_life"): "2단계 C-rate → 수명",
    }
    width, height = 1120, 560
    left, top, cell_w, cell_h = 330, 125, 140, 66
    parts = svg_header(width, height, "충전 조건과 수명·초기 열화 신호의 관계")
    parts.append('<text x="56" y="80" class="subtitle">색과 숫자는 Spearman ρ · 같은 규칙이 Batch마다 반복되는지와 Batch 보정 후 남는지를 함께 확인</text>')
    for col, scope in enumerate(scopes):
        cx = left + col * cell_w + cell_w / 2
        parts.append(f'<text x="{cx}" y="{top-22}" class="label" text-anchor="middle">{html.escape(scope)}</text>')
    for row_index, pair in enumerate(pairs):
        cy = top + row_index * cell_h + cell_h / 2
        parts.append(f'<text x="{left-20}" y="{cy+5}" class="label" text-anchor="end">{html.escape(labels[pair])}</text>')
        for col, scope in enumerate(scopes):
            subset = table[(table["feature"] == pair[0]) & (table["outcome"] == pair[1]) & (table["scope"] == scope)]
            value = float(subset.iloc[0]["spearman_r"]) if len(subset) else np.nan
            x0 = left + col * cell_w
            y0 = top + row_index * cell_h
            if not np.isfinite(value):
                fill, text_value = "#F2F5F7", "—"
            else:
                intensity = 0.15 + 0.75 * abs(value)
                base = (230, 126, 34) if value > 0 else (22, 125, 141)
                fill = "#%02x%02x%02x" % tuple(int(255 - (255 - channel) * intensity) for channel in base)
                text_value = f"{value:+.2f}"
            parts.append(f'<rect x="{x0+8}" y="{y0+7}" width="{cell_w-16}" height="{cell_h-14}" rx="10" fill="{fill}"/>')
            parts.append(f'<text x="{x0+cell_w/2}" y="{cy+5}" class="value" text-anchor="middle">{text_value}</text>')
    legend_y = height - 52
    parts.append(f'<rect x="{left}" y="{legend_y}" width="18" height="18" rx="4" fill="#62A8B3"/><text x="{left+28}" y="{legend_y+14}" class="small">음의 관계</text>')
    parts.append(f'<rect x="{left+150}" y="{legend_y}" width="18" height="18" rx="4" fill="#EDAA6B"/><text x="{left+178}" y="{legend_y+14}" class="small">양의 관계</text>')
    parts.append(f'<text x="{width-70}" y="{legend_y+14}" class="small" text-anchor="end">Batch 보정 = Batch 평균 차이를 제거한 부분 순위상관</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(RESULTS / "early_features.csv", encoding="utf-8-sig")

    low_life = analyze_low_life(features)
    charging = analyze_charging(features)
    redundancy = analyze_redundancy(features)

    low_life.to_csv(RESULTS / "low_life_group_comparison.csv", index=False, encoding="utf-8-sig")
    charging.to_csv(RESULTS / "charging_degradation_correlations.csv", index=False, encoding="utf-8-sig")
    redundancy.to_csv(RESULTS / "feature_redundancy_groups.csv", index=False, encoding="utf-8-sig")

    render_low_life_svg(low_life, FIGURES / "12_저수명_군집_진단.svg")
    render_charging_svg(charging, FIGURES / "13_충전조건_열화상관.svg")

    print(f"low_life_rows={len(low_life)}")
    print(f"charging_rows={len(charging)}")
    print(f"redundancy_groups={len(redundancy)}")


if __name__ == "__main__":
    main()
