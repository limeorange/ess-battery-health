#!/usr/bin/env python3
"""Create the two self-contained, submission-oriented Day 2 notebooks.

The notebooks contain the actual feature and modeling logic as visible code cells.
Expensive raw extraction and full hyperparameter search use explicit switches so a
normal Run All is fast and deterministic while the complete computation remains
editable and reproducible in the notebook itself.
"""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
PIPELINE_SOURCE = ROOT / "src" / "day2_analysis.py"
SOURCE_TEXT = PIPELINE_SOURCE.read_text(encoding="utf-8")
SOURCE_TREE = ast.parse(SOURCE_TEXT)
V2_PIPELINE_SOURCE = ROOT / "src" / "day2_v2_improvement.py"
V2_SOURCE_TEXT = V2_PIPELINE_SOURCE.read_text(encoding="utf-8")
V2_SOURCE_TREE = ast.parse(V2_SOURCE_TEXT)


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def source_for(*names: str) -> str:
    """Copy named functions/classes from the canonical pipeline into a code cell."""
    nodes = {
        node.name: node
        for node in SOURCE_TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    chunks = []
    lines = SOURCE_TEXT.splitlines()
    for name in names:
        node = nodes[name]
        starts = [node.lineno]
        starts.extend(decorator.lineno for decorator in getattr(node, "decorator_list", []))
        chunks.append("\n".join(lines[min(starts) - 1 : node.end_lineno]))
    return "\n\n".join(chunks)


def v2_source_for(*names: str) -> str:
    """Copy named v2 functions/classes into an explanatory notebook cell."""
    nodes = {
        node.name: node
        for node in V2_SOURCE_TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    chunks = []
    lines = V2_SOURCE_TEXT.splitlines()
    for name in names:
        node = nodes[name]
        starts = [node.lineno]
        starts.extend(decorator.lineno for decorator in getattr(node, "decorator_list", []))
        chunks.append("\n".join(lines[min(starts) - 1 : node.end_lineno]))
    return "\n\n".join(chunks)


def notebook(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python", "version": "3.12", "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3", "nbconvert_exporter": "python", "file_extension": ".py",
        },
    }
    return nb


FEATURE_SETUP = r'''
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import hashlib
import json
import re
import sys

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

LOCAL_PACKAGES = ROOT / ".python_packages"
if LOCAL_PACKAGES.exists() and str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))

import h5py
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from scipy import stats
from sklearn.model_selection import train_test_split

DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results" / "day2"
FIGURE_DIR = ROOT / "figures" / "day2"
RANDOM_SEED = 20261002
BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

font_candidates = [
    Path("/Users/lsh/Library/Fonts/NotoSansKR-Regular.otf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
]
font_path = next((path for path in font_candidates if path.exists()), None)
if font_path:
    fm.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

display(HTML("""
<style>
.jp-Notebook { max-width: 1180px; margin: auto; }
.jp-MarkdownOutput h1 { color:#173B5E; border-bottom:3px solid #1A9AA3; padding-bottom:.35rem; }
.jp-MarkdownOutput h2 { color:#173B5E; margin-top:2rem; }
.jp-MarkdownOutput h3 { color:#138A92; }
.jp-MarkdownOutput blockquote { border-left:5px solid #1A9AA3; background:#F1F8F8; padding:.75rem 1rem; }
.dataframe { font-size:13px; }
</style>
"""))

print("프로젝트 루트:", ROOT)
print("원본 데이터 폴더:", DATA_DIR)
'''


MODEL_SETUP = r'''
from pathlib import Path
from typing import Any, Iterable
import json
import math

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULT_DIR = ROOT / "results" / "day2"
RANDOM_SEED = 20261002
REFERENCE_MAPE = 9.1
FORBIDDEN_FEATURE_PATTERNS = (
    "cycle_life", "knee", "eol", "target", "observed_end",
    "global_cell_id", "cell_index", "batch",
)

font_candidates = [
    Path("/Users/lsh/Library/Fonts/NotoSansKR-Regular.otf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
]
font_path = next((path for path in font_candidates if path.exists()), None)
if font_path:
    fm.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")

display(HTML("""
<style>
.jp-Notebook { max-width: 1180px; margin: auto; }
.jp-MarkdownOutput h1 { color:#173B5E; border-bottom:3px solid #1A9AA3; padding-bottom:.35rem; }
.jp-MarkdownOutput h2 { color:#173B5E; margin-top:2rem; }
.jp-MarkdownOutput h3 { color:#138A92; }
.jp-MarkdownOutput blockquote { border-left:5px solid #1A9AA3; background:#F1F8F8; padding:.75rem 1rem; }
.dataframe { font-size:13px; }
</style>
"""))

print("프로젝트 루트:", ROOT)
'''


def build_feature_notebook() -> nbf.NotebookNode:
    helper_source = source_for(
        "Paths", "make_paths", "flat", "deref_scalar", "decode_matlab_char",
        "parse_policy", "valid_metric_mask", "metric_features", "delta_q_statistics",
        "life_group", "sha256_file",
    )
    extraction_source = source_for("extract_raw_features")
    quality_source = source_for(
        "build_quality_summary", "make_split_assignment", "make_feature_manifest", "validate_against_day1"
    )
    v2_feature_source = v2_source_for(
        "value_at_cycle", "window_mean", "interpolate_curve", "curve_statistics", "extract_v2_features"
    )
    return notebook([
        md(r'''
        # 02. Feature Engineering — 원본 `.mat`에서 Cell당 1행 만들기

        **Day 2 Regression · 배터리 Cycle Life 예측**  
        작성자: **U094 이수현** · 작성일: **2026-10-02**

        > 목표: 실제 cycle 번호 기준의 초기 100 cycle만 사용해 미래 수명을 설명할 수 있는 Feature Table을 만든다.

        이 노트북은 결과 CSV를 설명하는 문서가 아니라 **Feature Engineering의 주 실행본**이다. 원본 구조 확인, 유효값 필터, ΔQ(V), Cell 단위 집계, 품질검사, 고정 split과 누수 점검 코드가 모두 포함되어 있다.

        - 빠른 검토: `REBUILD_FROM_RAW = False`로 Run All
        - 원본부터 재현: `REBUILD_FROM_RAW = True` (7GB가 넘는 파일을 읽으므로 시간이 걸림)
        - 개인 실험: 공식 결과를 덮어쓰지 않도록 `SAVE_REBUILT_TABLES = False` 유지

        > **편집 주의:** 이 파일을 직접 보완한 뒤 `src/build_day2_notebooks.py`를 다시 실행하면 수동 변경이 덮어써진다. 앞으로 노트북을 주 작업본으로 사용할 때는 먼저 Git commit이나 별도 백업을 남긴다.
        '''),
        md(r'''
        ## 1. 환경과 데이터 계약

        분석 단위는 cycle 행이 아니라 **Cell당 1행**이다. 같은 Cell의 cycle을 train과 validation에 나누면 동일 배터리 정보가 양쪽에 들어가는 누수가 생긴다.

        - Summary Feature: 실제 cycle 2~100
        - ΔQ(V): `Qdlin(cycle 100) − Qdlin(cycle 10)`
        - Charging Feature: 시험 시작 전에 정해진 프로토콜
        - 금지: cycle 100 이후 값, Knee, EOL, cycle_life 또는 이를 직접 암시하는 값
        '''),
        code(FEATURE_SETUP),
        md(r'''
        ## 2. 원본 MATLAB v7.3 구조 확인

        파일 전체를 메모리에 올리지 않고 `h5py` 참조를 따라 필요한 Cell과 필드만 읽는다. 아래 코드는 Batch 1의 최상위 구조와 첫 Cell의 Summary 필드를 직접 확인한다.
        '''),
        code(r'''
        sample_file = DATA_DIR / BATCH_FILES["Batch 1"]
        with h5py.File(sample_file, "r") as handle:
            batch = handle["batch"]
            first_summary = handle[batch["summary"][0, 0]]
            structure = {
                "최상위 key": list(handle.keys()),
                "batch field": list(batch.keys()),
                "첫 Cell summary field": list(first_summary.keys()),
                "원본 Cell 수": int(batch["cycle_life"].shape[0]),
            }
        for key, value in structure.items():
            print(f"{key}: {value}")
        '''),
        md(r'''
        `cycle_life`, `summary`, `cycles`, `Vdlin`, `policy_readable`은 각 Cell을 가리키는 참조 배열이다. 따라서 단순한 표 읽기가 아니라 Cell 참조를 역참조해야 한다.
        '''),
        md(r'''
        ## 3. 원본 읽기와 Feature 계산 함수

        아래는 실제 구현이다. `metric_features()`는 cycle 2~100만 필터링하고, `delta_q_statistics()`는 ΔQ 곡선을 모델에 넣을 수 있는 통계량으로 바꾼다.
        '''),
        code(helper_source),
        md(r'''
        읽을 때는 다음을 확인한다.

        1. `valid_metric_mask()`가 물리적으로 말이 되지 않는 0값과 극단값을 제거한다.
        2. `metric_features()`가 평균·표준편차·변화량·기울기를 같은 규칙으로 계산한다.
        3. `delta_q_statistics()`가 전압을 정렬한 후 분산, IQR, 최솟값, 절대면적을 계산한다.
        4. 분산에는 `1e-12`를 더한 뒤 로그를 취해 0의 로그 문제를 피한다.
        '''),
        md(r'''
        ## 4. 전체 Cell 추출 로직

        이 함수가 세 Batch를 순회하면서 실제 cycle 번호를 찾고 Cell당 하나의 `record`를 만든다. 배열의 10번째·100번째 위치가 아니라 `cycle_to_index`에서 실제 10과 100을 찾는 부분이 핵심이다.
        '''),
        code(extraction_source),
        md(r'''
        ### 품질 요약·고정 split·Feature Manifest 함수

        원본 재계산 직후 사용할 품질 요약 함수도 먼저 정의한다. Target 결측을 따로 기록하고 Batch 1만 development/hold-out으로 나누며, 모든 Feature의 출처·cycle 범위·공식·단위를 문서화한다.
        '''),
        code(quality_source),
        md(r'''
        ## 5. 원본 재계산 또는 저장 결과 불러오기

        전체 로직은 위에 공개되어 있다. 기본 Run All에서는 동일 코드로 이미 생성한 표를 읽어 시간을 절약한다. 원본 재현이 필요하면 첫 번째 스위치를 `True`로 바꾼다.
        '''),
        code(r'''
        REBUILD_FROM_RAW = False
        SAVE_REBUILT_TABLES = False

        paths = make_paths(ROOT)
        if REBUILD_FROM_RAW:
            feature_table, quality_detail, file_checks = extract_raw_features(paths)
            quality_summary = build_quality_summary(feature_table, quality_detail)
            if SAVE_REBUILT_TABLES:
                feature_table.to_csv(RESULT_DIR / "feature_table.csv", index=False, encoding="utf-8-sig")
                quality_detail.to_csv(RESULT_DIR / "delta_q_validation.csv", index=False, encoding="utf-8-sig")
                quality_summary.to_csv(RESULT_DIR / "data_quality_summary.csv", index=False, encoding="utf-8-sig")
                file_checks.to_csv(RESULT_DIR / "data_file_checksums.csv", index=False, encoding="utf-8-sig")
        else:
            feature_table = pd.read_csv(RESULT_DIR / "feature_table.csv")
            quality_detail = pd.read_csv(RESULT_DIR / "delta_q_validation.csv")
            quality_summary = pd.read_csv(RESULT_DIR / "data_quality_summary.csv")
            file_checks = pd.read_csv(RESULT_DIR / "data_file_checksums.csv")

        print(f"Feature Table: {feature_table.shape[0]}행 × {feature_table.shape[1]}열")
        print(f"고유 Cell: {feature_table['global_cell_id'].nunique()}개")
        display(feature_table[[
            "batch", "global_cell_id", "cycle_life", "qd_mean", "qd_slope",
            "delta_q_log10_var", "delta_q_iqr", "ir_mean", "tavg_mean",
            "c_rate_stage1", "switch_soc_pct"
        ]].head())
        '''),
        md(r'''
        ## 6. ΔQ(V)를 한 Cell에서 직접 확인

        $$\Delta Q(V)=Q_{100}(V)-Q_{10}(V)$$

        아래 코드는 원본의 실제 cycle 번호로 두 곡선을 찾아 직접 뺀다. 왼쪽은 두 Q(V) 곡선, 오른쪽은 그 차이다.
        '''),
        code(r'''
        def load_delta_q_curve(file_path: Path, cell_index: int) -> pd.DataFrame:
            """원본 한 Cell에서 실제 cycle 10·100의 Qdlin과 ΔQ를 반환한다."""
            with h5py.File(file_path, "r") as handle:
                batch = handle["batch"]
                summary = handle[batch["summary"][cell_index, 0]]
                cycles = flat(summary["cycle"])
                cycle_to_index = {
                    int(round(cycle)): idx for idx, cycle in enumerate(cycles) if np.isfinite(cycle)
                }
                if 10 not in cycle_to_index or 100 not in cycle_to_index:
                    raise ValueError("cycle 10 또는 100이 없습니다.")
                curves = handle[batch["cycles"][cell_index, 0]]["Qdlin"]
                idx10, idx100 = cycle_to_index[10], cycle_to_index[100]
                voltage = flat(handle[batch["Vdlin"][cell_index, 0]])
                q10 = flat(handle[curves[idx10, 0]])
                q100 = flat(handle[curves[idx100, 0]])
            if not (len(voltage) == len(q10) == len(q100)):
                raise ValueError("Vdlin과 Qdlin 길이가 다릅니다.")
            return pd.DataFrame({
                "전압": voltage, "cycle 10 Qdlin": q10,
                "cycle 100 Qdlin": q100, "ΔQ": q100 - q10,
            }).dropna()

        dq_example = load_delta_q_curve(DATA_DIR / BATCH_FILES["Batch 1"], cell_index=0)
        example_stats = delta_q_statistics(dq_example["전압"].to_numpy(), dq_example["ΔQ"].to_numpy())

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        axes[0].plot(dq_example["전압"], dq_example["cycle 10 Qdlin"], label="cycle 10", lw=2)
        axes[0].plot(dq_example["전압"], dq_example["cycle 100 Qdlin"], label="cycle 100", lw=2)
        axes[0].set(title="동일 Cell의 방전용량 곡선", xlabel="전압 (V)", ylabel="누적 방전용량 (Ah)")
        axes[0].legend()
        axes[1].plot(dq_example["전압"], dq_example["ΔQ"], color="#D95D5D", lw=2)
        axes[1].axhline(0, color="#334155", ls="--", lw=1)
        axes[1].set(title="cycle 100 − cycle 10의 ΔQ(V)", xlabel="전압 (V)", ylabel="ΔQ (Ah)")
        fig.suptitle("초기 90 cycle 동안 전압-용량 곡선이 이동한 정도", fontsize=15, fontweight="bold")
        plt.tight_layout(); plt.show()

        display(pd.Series({
            "delta_q_log10_var": example_stats["delta_q_log10_var"],
            "delta_q_iqr": example_stats["delta_q_iqr"],
            "delta_q_min": example_stats["delta_q_min"],
            "delta_q_abs_area": example_stats["delta_q_abs_area"],
        }).to_frame("값"))
        '''),
        md(r'''
        ΔQ 곡선 전체를 넣으면 작은 표본에 비해 변수가 너무 많아진다. 곡선을 대표 통계량으로 압축하고, 어느 하나를 Core로 쓸지는 Batch 1 development CV에서만 결정한다.
        '''),
        md(r'''
        ## 7. 품질 요약·고정 split·Feature Manifest 함수

        앞에서 정의한 함수를 실제 Feature Table에 적용하고 저장 결과와 일치하는지 검증한다.
        '''),
        code(r'''
        quality_summary_live = build_quality_summary(feature_table, quality_detail)
        display(quality_summary_live)
        assert len(feature_table) == 139
        assert feature_table["global_cell_id"].is_unique
        assert quality_detail["cycle_10_available"].all()
        assert quality_detail["cycle_100_available"].all()
        assert quality_detail["delta_q_available"].all()
        print("품질검사 통과: 139개 Cell 모두 cycle 10·100과 ΔQ 계산 가능")
        '''),
        md(r'''
        Target이 없는 Cell은 Feature 계산 실패가 아니라 지도학습 정답이 없다는 뜻이다. 관측 종료 cycle로 Target을 대체하지 않고 supervised 평가에서만 제외한다.
        '''),
        code(r'''
        split_live = make_split_assignment(feature_table)
        split_saved = pd.read_csv(RESULT_DIR / "split_assignment.csv")
        split_check = split_live[["global_cell_id", "split"]].merge(
            split_saved[["global_cell_id", "split"]], on="global_cell_id", suffixes=("_재계산", "_저장")
        )
        assert (split_check["split_재계산"] == split_check["split_저장"]).all()
        display(split_live.groupby(["batch", "split"], as_index=False).agg(
            Cell_수=("global_cell_id", "size"), 평균_수명=("cycle_life", "mean")
        ).round(1))
        print("고정 split 재현 통과 · seed =", RANDOM_SEED)
        '''),
        md(r'''
        Batch 1의 46개 Cell만 36개 development와 10개 fixed hold-out으로 나눈다. Batch 2와 Batch 3는 분할·Feature 선택에 참여하지 않는다.
        '''),
        code(r'''
        manifest = make_feature_manifest()
        with open(RESULT_DIR / "external_evaluation_lock.json", encoding="utf-8") as handle:
            lock = json.load(handle)
        manifest["final_selected"] = manifest["feature_name"].isin(lock["selected_features"])
        display(manifest)
        '''),
        md(r'''
        ## 8. Feature 관계와 다중공선성 사전 점검

        ΔQ 후보는 같은 곡선에서 파생되므로 강하게 중복될 수 있다. 대표값 하나만 고르는 이유를 Batch 1 development의 Spearman 상관으로 확인한다.
        '''),
        code(r'''
        development_ids = set(split_live.query("split == 'development'")["global_cell_id"])
        development = feature_table[feature_table["global_cell_id"].isin(development_ids)].copy()
        candidate_features = [
            "delta_q_log10_var", "delta_q_iqr", "delta_q_min", "delta_q_abs_area",
            "qd_mean", "qd_slope", "ir_mean", "tavg_mean", "chargetime_mean"
        ]
        corr = development[candidate_features + ["cycle_life"]].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax)
        ax.set_title("Batch 1 development Feature의 Spearman 상관", fontweight="bold")
        plt.tight_layout(); plt.show()
        display(corr["cycle_life"].drop("cycle_life").sort_values(key=abs, ascending=False).to_frame("Cycle Life와 Spearman ρ"))
        '''),
        md(r'''
        상관은 후보를 이해하는 근거이지 최종 선택 기준 하나로 사용하지 않는다. 실제 선정은 같은 반복 CV에서 예측오차를 비교한다.
        '''),
        md("## 9. 누수 방지 자동검사"),
        code(r'''
        forbidden = ("cycle_life", "knee", "eol", "target", "observed_end", "global_cell_id", "cell_index", "batch")
        candidate_names = manifest["feature_name"].tolist()
        checks = {
            "Cell당 1행": feature_table["global_cell_id"].is_unique,
            "139개 원본 Cell": len(feature_table) == 139,
            "cycle 10·100 기반 ΔQ": set(manifest.loc[manifest["feature_name"].str.startswith("delta_q"), "cycle_range"]) == {"10,100"},
            "금지 Feature 없음": not any(token in name.lower() for name in candidate_names for token in forbidden),
            "Batch 1 split 고정": (split_live["split"] == "development").sum() == 36 and (split_live["split"] == "holdout").sum() == 10,
            "외부 Batch 선택 격리": set(split_live.query("batch != 'Batch 1' and target_available")["split"]) == {"external_test", "additional_test"},
        }
        display(pd.DataFrame([{"검사": key, "결과": "통과" if value else "실패"} for key, value in checks.items()]))
        assert all(checks.values())
        '''),
        md(r'''
        ## 10. Feature Engineering 결론

        - 원본 역참조부터 Cell당 1행 생성까지의 코드가 모두 포함되어 있다.
        - 실제 cycle 2~100만 Summary Feature에 사용하고 ΔQ는 실제 cycle 10과 100으로 계산한다.
        - Target 결측과 Feature 실패를 구분하고 모든 제외 이유를 기록한다.
        - ΔQ 후보의 중복을 확인했으므로 대표값은 하나만 선택한다.
        - Batch 1 development/hold-out을 고정했고 Batch 2·3은 선택 과정에서 격리했다.

        다음 `03_modeling.ipynb`에서 ΔQ screening, F0~F4 Ablation, 다섯 모델의 반복 CV, 1-SE 선택, hold-out과 외부 Batch 평가를 실제 코드로 수행한다.
        '''),
        md(r'''
        ## 11. 추가 개선 실험 — 새로운 정보 표현

        공식 v1 결과를 변경하지 않고 다음 세 표현을 별도 `day2_v2` 산출물로 생성했다.

        1. 원 논문형 초기 Feature: cycle 2·100 용량, 최대용량 변화와 발생 cycle, 선형 기울기·절편, cycle 2~6 충전시간, 내부저항 변화
        2. `ΔQ100-10(V)`의 공통 2.0~3.5V, 1,000-point 원곡선
        3. cycle 20~100의 다중 ΔQ trajectory

        모든 입력은 실제 cycle 100 이하만 사용한다. 아래 셀은 원본 역참조부터 새 Feature 생성까지의 실제 구현이다.
        '''),
        code(r'''
        V2_RESULT_DIR = ROOT / "results" / "day2_v2"
        COMMON_VOLTAGE = np.linspace(2.0, 3.5, 1000)
        TRAJECTORY_CYCLES = tuple(range(20, 101, 10))
        '''),
        code(v2_feature_source),
        code(r'''
        RUN_V2_RAW_EXTRACTION = False

        if RUN_V2_RAW_EXTRACTION:
            v2_features_live, delta_q_curve_live, delta_q_trajectory_live = extract_v2_features()
            print("재추출:", v2_features_live.shape, delta_q_curve_live.shape, delta_q_trajectory_live.shape)
        else:
            v2_features_live = pd.read_csv(V2_RESULT_DIR / "extended_feature_table.csv")
            curve_data = np.load(V2_RESULT_DIR / "delta_q_curve_data.npz", allow_pickle=False)
            delta_q_curve_live = curve_data["cycle100_minus10"]
            delta_q_trajectory_live = curve_data["trajectory"]

        v2_manifest = pd.read_csv(V2_RESULT_DIR / "v2_feature_manifest.csv")

        display(v2_features_live[[
            "global_cell_id", "batch", "cycle_life", "qd_cycle2", "qd_cycle100",
            "qd_max_minus_cycle2", "qd_max_cycle", "charge_time_mean_2_6",
            "ir_cycle100_minus_cycle2", "dq100_10_log10_var"
        ]].head())
        print("ΔQ100-10 원곡선:", delta_q_curve_live.shape)
        print("9개 다중 Cycle ΔQ 원곡선:", delta_q_trajectory_live.shape)
        display(v2_manifest)
        assert delta_q_curve_live.shape == (139, 1000)
        assert delta_q_trajectory_live.shape == (139, 9000)
        '''),
        md(r'''
        이 확장은 Feature 수를 무작정 늘리는 작업이 아니다. 기존 `delta_q_iqr`가 버리던 전압 위치 정보를 보존하고, 원 논문과 현재 구현의 차이를 검증하기 위한 제한된 추가 실험이다. 모델 선택에는 계속 Batch 1 development만 사용한다.
        '''),
    ])


def build_model_notebook() -> nbf.NotebookNode:
    cv_source = source_for(
        "numeric_frame", "make_cv_splits", "make_model_specs", "tune_and_score",
        "protocol_group_robust_score",
    )
    metric_source = source_for("life_group", "regression_metrics", "bootstrap_mape_ci")
    error_source = source_for("build_error_tables", "extract_importance")
    v2_model_source = v2_source_for(
        "metrics", "log_target", "Candidate", "scalar_pipeline", "curve_pls_pipeline",
        "curve_pca_ridge_pipeline", "nested_score", "tune_on_development", "build_candidates",
        "benchmark_all_predeclared_candidates",
    )
    return notebook([
        md(r'''
        # 03. Modeling — 모델 선택부터 외부 Batch 검증까지

        **Day 2 Regression · 배터리 Cycle Life 예측**  
        작성자: **U094 이수현** · 작성일: **2026-10-02**

        > 목표: Batch 1 안에서 Feature와 모델을 선택하고 설정을 동결한 뒤, Batch 2를 최종 외부 테스트로 평가하고 같은 설정으로 Batch 3 일반화를 확인한다.

        이 노트북은 모델링의 **주 실행본**이다. Pipeline, 반복 CV, Hyperparameter grid, ΔQ screening, F0~F4 Ablation, 1-SE 선정, hold-out, 외부 예측, 지표·Gap·오류·분포 이동·시각화 코드가 직접 들어 있다.

        `RUN_FULL_CV_SEARCH = False`에서는 시간이 오래 걸리는 전체 탐색 결과만 저장본에서 읽는다. 탐색 코드는 그대로 보이며 `True`로 바꾸면 같은 계산을 수행한다. 최종 선형회귀 적합과 hold-out·Batch 2·Batch 3 예측은 기본 Run All에서도 실제로 다시 수행한다.

        > **편집 주의:** 이 파일을 직접 보완한 뒤 `src/build_day2_notebooks.py`를 다시 실행하면 수동 변경이 덮어써진다. 앞으로 노트북을 주 작업본으로 사용할 때는 먼저 Git commit이나 별도 백업을 남긴다.
        '''),
        md(r'''
        ## 1. 환경과 입력 데이터

        `02_feature_engineering.ipynb`가 만든 Cell당 1행 Feature Table과 고정 split을 읽는다. 모델 입력에 cycle_life, Cell ID, Batch 이름이 들어가지 않는지 다시 검사한다.
        '''),
        code(MODEL_SETUP),
        code(r'''
        feature_table = pd.read_csv(RESULT_DIR / "feature_table.csv")
        split_assignment = pd.read_csv(RESULT_DIR / "split_assignment.csv")
        data = feature_table.merge(
            split_assignment[["global_cell_id", "split"]],
            on="global_cell_id", how="left", validate="one_to_one"
        )
        development = data.query("split == 'development'").reset_index(drop=True)
        holdout = data.query("split == 'holdout'").reset_index(drop=True)
        batch2 = data.query("split == 'external_test'").reset_index(drop=True)
        batch3 = data.query("split == 'additional_test'").reset_index(drop=True)
        display(pd.DataFrame({
            "구간": ["Batch 1 development", "Batch 1 hold-out", "Batch 2 test", "Batch 3 additional"],
            "Cell 수": [len(development), len(holdout), len(batch2), len(batch3)],
            "평균 수명": [development.cycle_life.mean(), holdout.cycle_life.mean(), batch2.cycle_life.mean(), batch3.cycle_life.mean()],
        }).round(1))
        assert (len(development), len(holdout), len(batch2), len(batch3)) == (36, 10, 39, 44)
        '''),
        md(r'''
        ## 2. CV·Pipeline·모델 후보의 실제 구현

        모든 전처리는 Pipeline 안에 둔다. `SimpleImputer`와 `StandardScaler`는 각 CV train fold에서만 fit되므로 validation 통계가 학습에 새지 않는다.

        모델 후보는 Median Baseline, Linear Regression, Ridge, ElasticNet, 제한된 Gradient Boosting이다. 작은 정형 데이터이므로 딥러닝은 표본 대비 복잡성과 설명 부담이 커서 제외했다.
        '''),
        code(cv_source),
        code(r'''
        SCORING = {
            "mape": "neg_mean_absolute_percentage_error",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
        cv_splits = make_cv_splits(len(development))
        model_specs = make_model_specs()
        print(f"동일하게 재사용할 CV 분할 수: {len(cv_splits)} = 5 folds × 10 repeats")
        for name, (pipeline, grid, complexity) in model_specs.items():
            combinations = int(np.prod([len(values) for values in grid.values()])) if grid else 1
            print(f"\n[{name}] 복잡도 순위={complexity} · 탐색 조합={combinations}")
            print(pipeline)
        '''),
        md(r'''
        `GridSearchCV`와 최종 `cross_validate` 모두 같은 50개 fold를 사용한다. 평균 MAPE뿐 아니라 표준편차, repeat 간 표준오차, MAE, RMSE, R²를 함께 저장한다.
        '''),
        md(r'''
        ## 3. ΔQ 대표 Feature screening

        같은 곡선에서 만든 네 후보를 동시에 넣지 않는다. Batch 1 development만 사용하고 동일한 Ridge·동일한 split에서 하나씩 비교한다.
        '''),
        code(r'''
        RUN_FULL_CV_SEARCH = False
        dq_candidates = ["delta_q_log10_var", "delta_q_iqr", "delta_q_min", "delta_q_abs_area"]

        if RUN_FULL_CV_SEARCH:
            dq_rows, dq_fold_frames = [], []
            ridge_pipe, ridge_grid, _ = model_specs["Ridge"]
            for candidate in dq_candidates:
                _, params, fold, summary = tune_and_score(
                    numeric_frame(development, [candidate]), development["cycle_life"],
                    ridge_pipe, ridge_grid, cv_splits,
                )
                dq_rows.append({"feature": candidate, **summary, "best_params": json.dumps(params, ensure_ascii=False)})
                fold["feature"] = candidate
                dq_fold_frames.append(fold)
            dq_results = pd.DataFrame(dq_rows).sort_values("cv_mape_mean").reset_index(drop=True)
            dq_folds = pd.concat(dq_fold_frames, ignore_index=True)
        else:
            dq_results = pd.read_csv(RESULT_DIR / "delta_q_candidate_results.csv")
            dq_folds = pd.read_csv(RESULT_DIR / "delta_q_candidate_folds.csv")

        best_dq = dq_results.sort_values("cv_mape_mean").iloc[0]
        predeclared = dq_results.query("feature == 'delta_q_log10_var'").iloc[0]
        chosen_dq = (
            "delta_q_log10_var"
            if predeclared["cv_mape_mean"] <= best_dq["cv_mape_mean"] + best_dq["cv_mape_se"]
            else str(best_dq["feature"])
        )
        dq_results["selected_core_recomputed"] = dq_results["feature"].eq(chosen_dq)
        display(dq_results[["feature", "cv_mape_mean", "cv_mape_std", "cv_mape_se", "selected_core_recomputed"]].round(3))
        print("선정된 ΔQ Core:", chosen_dq)
        '''),
        md(r'''
        사전 1순위 `delta_q_log10_var`가 최저 후보의 1-SE 안에 들어오면 계획을 유지하고, 그렇지 않을 때만 최저 후보로 바꾼다. 실제로는 IQR 8.26%, log-variance 9.00%였고 1-SE 밖이어서 IQR을 선택했다. Batch 2를 보기 전의 결정이다.
        '''),
        md(r'''
        ## 4. F0~F4 Feature Ablation과 다섯 모델 전체 비교

        아래 딕셔너리가 Feature 선정 논리를 코드로 명시한다. 이후 모든 조합은 같은 `cv_splits`를 사용한다.
        '''),
        code(r'''
        feature_sets = {
            "F0 Capacity": ["qd_mean", "qd_slope"],
            "F1 ΔQ": [chosen_dq],
            "F2 ΔQ+Capacity": [chosen_dq, "qd_mean", "qd_slope"],
            "F3 +Sensor": [chosen_dq, "qd_mean", "qd_slope", "ir_mean", "tavg_mean", "chargetime_mean"],
            "F4 +Charging": [
                chosen_dq, "qd_mean", "qd_slope", "ir_mean", "tavg_mean", "chargetime_mean",
                "c_rate_stage1", "switch_soc_pct", "c_rate_stage2", "policy_new_structure", "policy_slow_cycle",
            ],
        }
        for set_name, columns in feature_sets.items():
            leaked = [name for name in columns if any(token in name.lower() for token in FORBIDDEN_FEATURE_PATTERNS)]
            if leaked:
                raise AssertionError(f"{set_name}에 미래정보·식별자 포함: {leaked}")
            print(f"{set_name:18s} · {len(columns):2d}개 · {columns}")
        '''),
        code(r'''
        if RUN_FULL_CV_SEARCH:
            comparison_rows, all_fold_frames = [], []
            for set_name, columns in feature_sets.items():
                X_dev, y_dev = numeric_frame(development, columns), development["cycle_life"]
                for model_name, (pipeline, grid, complexity) in model_specs.items():
                    print(f"실행: {set_name} / {model_name}")
                    _, params, fold, summary = tune_and_score(X_dev, y_dev, pipeline, grid, cv_splits)
                    comparison_rows.append({
                        "feature_set": set_name, "model": model_name,
                        "feature_count": len(columns), "model_complexity_rank": complexity,
                        **summary, "best_params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                        "features": ", ".join(columns),
                    })
                    fold["feature_set"], fold["model"] = set_name, model_name
                    all_fold_frames.append(fold)
            comparison = pd.DataFrame(comparison_rows).sort_values("cv_mape_mean").reset_index(drop=True)
            fold_results = pd.concat(all_fold_frames, ignore_index=True)
        else:
            comparison = pd.read_csv(RESULT_DIR / "model_comparison.csv")
            fold_results = pd.read_csv(RESULT_DIR / "cv_fold_results.csv")

        best_mean = float(comparison.iloc[0]["cv_mape_mean"])
        one_se_threshold = best_mean + float(comparison.iloc[0]["cv_mape_se"])
        comparison["within_one_se_recomputed"] = comparison["cv_mape_mean"] <= one_se_threshold
        display(comparison.head(12)[[
            "feature_set", "model", "feature_count", "cv_mape_mean", "cv_mape_std",
            "cv_mae_mean", "cv_rmse_mean", "cv_r2_mean", "best_params", "within_one_se_recomputed"
        ]].round(3))
        '''),
        code(r'''
        stage_order = ["F0 Capacity", "F1 ΔQ", "F2 ΔQ+Capacity", "F3 +Sensor", "F4 +Charging"]
        ablation = comparison.query("model == 'Ridge'").copy()
        ablation["feature_set"] = pd.Categorical(ablation["feature_set"], stage_order, ordered=True)
        ablation = ablation.sort_values("feature_set")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].errorbar(ablation["feature_set"].astype(str), ablation["cv_mape_mean"],
                         yerr=ablation["cv_mape_std"], fmt="o-", capsize=5, color="#1A9AA3", lw=2)
        axes[0].set(title="Feature Ablation: 같은 Ridge로 비교", xlabel="Feature 단계", ylabel="CV MAPE (%)")
        axes[0].tick_params(axis="x", rotation=18)
        axes[1].bar(ablation["feature_set"].astype(str), ablation["feature_count"], color="#3568C0")
        axes[1].set(title="성능과 함께 증가한 Feature 수", xlabel="Feature 단계", ylabel="Feature 수")
        axes[1].tick_params(axis="x", rotation=18)
        fig.suptitle("ΔQ의 기여는 크지만 Sensor·Charging의 추가 이득은 확인되지 않음", fontsize=15, fontweight="bold")
        plt.tight_layout(); plt.show()
        display(ablation[["feature_set", "feature_count", "cv_mape_mean", "cv_mape_std", "features"]].round(3))
        '''),
        md(r'''
        Capacity만 사용한 F0 18.95%에서 ΔQ만 사용한 F1 8.26%로 크게 개선되고, Capacity를 결합한 F2가 7.68%로 추가 개선된다. Sensor와 Charging을 더한 F3·F4는 변수 수가 늘지만 Ridge MAPE가 악화된다.
        '''),
        md(r'''
        ## 5. 충전 프로토콜 그룹 강건성으로 최종 후보 선정

        무작위 CV는 같은 C-rate 그룹의 Cell을 train과 validation에 함께 놓을 수 있다. Batch 2처럼 처음 보는 충전조건으로 이동하는 상황을 모사하기 위해, Batch 1 development의 1단계 C-rate를 하나씩 통째로 제외하는 nested leave-one-group-out 검증을 사용한다. 선택 기준은 **최악 그룹 MAPE 최소화**다.
        '''),
        code(r'''
        if RUN_FULL_CV_SEARCH:
            robust_fold_frames, robust_feature_rows = [], []
            ridge_pipeline, ridge_grid, _ = model_specs["Ridge"]
            for set_name, columns in feature_sets.items():
                folds, summary = protocol_group_robust_score(
                    development, columns, ridge_pipeline, ridge_grid, set_name, "feature_set"
                )
                robust_fold_frames.append(folds)
                robust_feature_rows.append({"feature_set": set_name, **summary})
            robust_feature_results = pd.DataFrame(robust_feature_rows).sort_values(
                ["worst_group_mape_pct", "macro_group_mape_pct", "feature_count"]
            ).reset_index(drop=True)
            robust_feature_results["selected_feature_set"] = False
            robust_feature_results.loc[0, "selected_feature_set"] = True
            preliminary_set = str(robust_feature_results.loc[0, "feature_set"])

            robust_model_rows = []
            for model_name, (pipeline, grid, complexity) in model_specs.items():
                folds, summary = protocol_group_robust_score(
                    development, feature_sets[preliminary_set], pipeline, grid, model_name, "model"
                )
                folds["feature_set"] = preliminary_set
                robust_fold_frames.append(folds)
                robust_model_rows.append({
                    "feature_set": preliminary_set, "model": model_name,
                    "model_complexity_rank": complexity, **summary,
                })
            robust_model_results = pd.DataFrame(robust_model_rows).sort_values(
                ["worst_group_mape_pct", "macro_group_mape_pct", "model_complexity_rank"]
            ).reset_index(drop=True)
            robust_model_results["selected_model"] = False
            robust_model_results.loc[0, "selected_model"] = True
            robust_fold_results = pd.concat(robust_fold_frames, ignore_index=True)
        else:
            robust_feature_results = pd.read_csv(RESULT_DIR / "protocol_robust_feature_results.csv")
            robust_model_results = pd.read_csv(RESULT_DIR / "protocol_robust_model_results.csv")
            robust_fold_results = pd.read_csv(RESULT_DIR / "protocol_robust_fold_results.csv")

        display(robust_feature_results[[
            "feature_set", "feature_count", "pooled_mape_pct", "macro_group_mape_pct",
            "worst_group_mape_pct", "group_mape_std_pct", "selected_feature_set"
        ]].round(3))
        selected_feature_set = str(
            robust_feature_results.loc[robust_feature_results["selected_feature_set"], "feature_set"].iloc[0]
        )
        display(robust_model_results[[
            "model", "pooled_mape_pct", "macro_group_mape_pct",
            "worst_group_mape_pct", "group_mape_std_pct", "selected_model"
        ]].round(3))
        selected_model_name = str(
            robust_model_results.loc[robust_model_results["selected_model"], "model"].iloc[0]
        )
        selected_features = feature_sets[selected_feature_set]
        selected_row = comparison.query(
            "feature_set == @selected_feature_set and model == @selected_model_name"
        ).iloc[0]
        selected_params = json.loads(selected_row["best_params"])
        print("최종 선택:", selected_feature_set, "+", selected_model_name)
        print("Feature:", selected_features)
        print("설정:", selected_params)
        assert selected_feature_set == "F1 ΔQ"
        assert selected_model_name == "Linear Regression"
        assert selected_params == {}
        '''),
        code(r'''
        top = comparison.head(12).sort_values("cv_mape_mean", ascending=False).copy()
        top["후보"] = top["feature_set"] + " · " + top["model"]
        colors_top = ["#1A9AA3" if (fs == selected_feature_set and model == selected_model_name) else "#9AA5B1"
                      for fs, model in zip(top["feature_set"], top["model"])]
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.barh(top["후보"], top["cv_mape_mean"], xerr=top["cv_mape_std"], color=colors_top, capsize=3)
        ax.axvline(one_se_threshold, color="#D95D5D", ls="--", label=f"1-SE 경계 {one_se_threshold:.2f}%")
        ax.set(title="Batch 1 development 모델 비교", xlabel="반복 CV MAPE (%)", ylabel="")
        ax.legend(); plt.tight_layout(); plt.show()
        '''),
        md(r'''
        ## 6. Batch 1 fixed hold-out — 설정 동결 전 최종 점검

        선택된 Pipeline을 새로 만들고 development 36개에만 fit한다. 아래 예측은 저장값이 아니라 노트북에서 실제로 다시 계산된다.
        '''),
        code(metric_source),
        code(r'''
        base_pipeline, _, _ = model_specs[selected_model_name]
        selected_estimator = clone(base_pipeline).set_params(**selected_params)
        selected_estimator.fit(numeric_frame(development, selected_features), development["cycle_life"])
        holdout_prediction = selected_estimator.predict(numeric_frame(holdout, selected_features))
        holdout_metrics = regression_metrics(holdout["cycle_life"], holdout_prediction)
        selected_cv_row = comparison.query("feature_set == @selected_feature_set and model == @selected_model_name").iloc[0]
        stability_limit = selected_cv_row["cv_mape_mean"] + 2 * selected_cv_row["cv_mape_std"]
        display(pd.Series({**holdout_metrics, "사전 안정성 상한": stability_limit,
                           "통과": holdout_metrics["mape_pct"] <= stability_limit}).to_frame("값"))
        assert holdout_metrics["mape_pct"] <= stability_limit
        saved_holdout = pd.read_csv(RESULT_DIR / "predictions.csv").query("dataset == 'Batch 1 Hold-out'")
        saved_holdout = saved_holdout.set_index("global_cell_id").loc[holdout["global_cell_id"]]
        print("저장 예측과 최대 절대차:", np.max(np.abs(holdout_prediction - saved_holdout["prediction"].to_numpy())))
        '''),
        md(r'''
        Hold-out MAPE는 5.28%로 안정성 상한을 통과했다. 최종 모델은 Hyperparameter가 없는 단일 특성 선형회귀다. 개정 동기가 기존 Batch 2 실패에서 출발했다는 점은 숨기지 않고, 실제 적합과 선정에 Batch 2·3 Target을 사용하지 않았다.
        '''),
        md(r'''
        ## 7. 설정 잠금 확인 후 Batch 2·3 외부 평가

        Lock을 검사한 뒤 Batch 1 labeled 46개 전체로 같은 Pipeline을 다시 fit하고 Batch 2·3를 실제로 예측한다.
        '''),
        code(r'''
        with open(RESULT_DIR / "external_evaluation_lock.json", encoding="utf-8") as handle:
            lock = json.load(handle)
        assert lock["batch2_target_used_in_model_fit_or_candidate_scoring"] is False
        assert lock["batch3_target_used_in_model_fit_or_candidate_scoring"] is False
        assert lock["protocol_amendment_motivated_by_prior_batch2_failure"] is True
        assert lock["batch2_is_fresh_blind_test"] is False
        assert lock["post_batch2_target_optimization"] is False
        assert lock["selected_feature_set"] == selected_feature_set
        assert lock["selected_model"] == selected_model_name
        assert set(lock["selected_features"]) == set(selected_features)
        print("외부 평가 잠금 확인:", lock["status"])

        final_model = clone(base_pipeline).set_params(**selected_params)
        batch1_all = data.query("split in ['development', 'holdout']").reset_index(drop=True)
        final_model.fit(numeric_frame(batch1_all, selected_features), batch1_all["cycle_life"])
        evaluation_sets = [
            ("Batch 1 Hold-out", holdout, holdout_prediction),
            ("Batch 2 Test", batch2, final_model.predict(numeric_frame(batch2, selected_features))),
            ("Batch 3 Test", batch3, final_model.predict(numeric_frame(batch3, selected_features))),
        ]
        prediction_rows = []
        for dataset_name, frame, predicted in evaluation_sets:
            for (_, row), y_hat in zip(frame.iterrows(), predicted):
                actual, y_hat = float(row["cycle_life"]), float(y_hat)
                prediction_rows.append({
                    "dataset": dataset_name, "batch": row["batch"], "global_cell_id": row["global_cell_id"],
                    "cycle_life": actual, "prediction": y_hat, "residual": y_hat - actual,
                    "absolute_error": abs(y_hat - actual), "ape_pct": abs(y_hat - actual) / actual * 100,
                    "error_direction": "과대 예측" if y_hat > actual else "과소 예측",
                    "life_group": life_group(actual), "charging_policy": row["charging_policy"],
                    "c_rate_stage1": row["c_rate_stage1"], "switch_soc_pct": row["switch_soc_pct"],
                    **{feature: row[feature] for feature in selected_features},
                })
        predictions = pd.DataFrame(prediction_rows)
        saved_predictions = pd.read_csv(RESULT_DIR / "predictions.csv")
        check = predictions[["dataset", "global_cell_id", "prediction"]].merge(
            saved_predictions[["dataset", "global_cell_id", "prediction"]],
            on=["dataset", "global_cell_id"], suffixes=("_재계산", "_저장")
        )
        print("동결 외부예측 재현 최대 절대차:", np.max(np.abs(check["prediction_재계산"] - check["prediction_저장"])))
        '''),
        md("## 8. MAPE·MAE·RMSE·R²와 모든 Gap 계산"),
        code(r'''
        performance_rows = [{
            "dataset": "Train (Batch 1 CV)", "n": len(development),
            "mape_pct": selected_cv_row["cv_mape_mean"], "mape_std_pct": selected_cv_row["cv_mape_std"],
            "mape_ci95_low": np.nan, "mape_ci95_high": np.nan,
            "mae": selected_cv_row["cv_mae_mean"], "rmse": selected_cv_row["cv_rmse_mean"],
            "r2": selected_cv_row["cv_r2_mean"], "median_ape_pct": np.nan,
        }]
        for dataset_name, frame in predictions.groupby("dataset", sort=False):
            metrics = regression_metrics(frame["cycle_life"], frame["prediction"])
            ci_low, ci_high = bootstrap_mape_ci(frame["cycle_life"].to_numpy(), frame["prediction"].to_numpy())
            performance_rows.append({"dataset": dataset_name, "n": len(frame), "mape_std_pct": np.nan,
                                     "mape_ci95_low": ci_low, "mape_ci95_high": ci_high, **metrics})
        performance = pd.DataFrame(performance_rows)
        metric_lookup = performance.set_index("dataset")["mape_pct"]
        gaps = pd.DataFrame([
            {"gap": "Train-Valid", "value_pct_point": metric_lookup["Batch 1 Hold-out"] - metric_lookup["Train (Batch 1 CV)"]},
            {"gap": "Valid-Test", "value_pct_point": metric_lookup["Batch 2 Test"] - metric_lookup["Batch 1 Hold-out"]},
            {"gap": "Target-Test", "value_pct_point": metric_lookup["Batch 2 Test"] - REFERENCE_MAPE},
            {"gap": "Batch2-Batch3", "value_pct_point": metric_lookup["Batch 3 Test"] - metric_lookup["Batch 2 Test"]},
        ])
        display(performance.round(3)); display(gaps.round(3))
        '''),
        md(r'''
        개정 모델은 Batch 1 반복 CV 8.34%, hold-out 5.28%, Batch 2 24.68%, Batch 3 14.10%를 기록했다. 기존 F2 Ridge 대비 Batch 2 MAPE가 11.88%p 낮아졌지만 Valid–Test Gap은 여전히 +19.41%p이므로 Batch shift가 해결된 것은 아니다.
        '''),
        md("## 9. 실제값–예측값과 잔차"),
        code(r'''
        colors = {"Batch 1 Hold-out": "#3568C0", "Batch 2 Test": "#E59A24", "Batch 3 Test": "#22A77A"}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
        limits = [predictions[["cycle_life", "prediction"]].min().min() * .9,
                  predictions[["cycle_life", "prediction"]].max().max() * 1.05]
        for ax, dataset_name in zip(axes, colors):
            frame = predictions.query("dataset == @dataset_name")
            ax.scatter(frame["cycle_life"], frame["prediction"], s=48, color=colors[dataset_name], edgecolor="white")
            ax.plot(limits, limits, "--", color="#334155", label="정답선 y=x")
            ax.set(title=f"{dataset_name}\nMAPE {frame.ape_pct.mean():.1f}%", xlabel="실제 수명 (cycle)", xlim=limits, ylim=limits)
        axes[0].set_ylabel("예측 수명 (cycle)"); axes[-1].legend()
        fig.suptitle("실제 수명과 예측 수명 — 대각선에서 멀수록 큰 오차", fontsize=15, fontweight="bold")
        plt.tight_layout(); plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        sns.histplot(data=predictions, x="residual", hue="dataset", bins=18, element="step", common_norm=False, ax=axes[0], palette=colors)
        axes[0].axvline(0, color="#334155", ls="--"); axes[0].set(title="잔차 분포", xlabel="잔차 = 예측 − 실제 (cycle)", ylabel="Cell 수")
        sns.scatterplot(data=predictions, x="cycle_life", y="residual", hue="dataset", style="error_direction", s=65, ax=axes[1], palette=colors)
        axes[1].axhline(0, color="#334155", ls="--"); axes[1].set(title="실제 수명에 따른 잔차", xlabel="실제 수명 (cycle)", ylabel="잔차 (cycle)")
        fig.suptitle("과대·과소 예측의 방향과 크기", fontsize=15, fontweight="bold")
        plt.tight_layout(); plt.show()
        display(predictions.groupby("dataset", as_index=False).agg(
            n=("global_cell_id", "size"), 실제_평균=("cycle_life", "mean"), 예측_평균=("prediction", "mean"),
            평균_잔차=("residual", "mean"), 과대예측률=("residual", lambda s: 100 * (s > 0).mean())
        ).round(2))
        '''),
        md(r'''
        Batch 2는 평균 실제 565.7 cycle을 678.4 cycle로 예측했고 39개 중 32개를 과대 예측했다. 기존 모델보다 편향은 줄었지만, ESS에서 정비 지연을 만들 수 있는 과대 예측 위험은 여전히 남아 있다.
        '''),
        md(r'''
        ## 10. 수명·충전조건별 오류, 최악 Cell, Feature shift

        아래 함수는 방금 계산한 `predictions`와 세 Batch Feature에서 오류표를 직접 만든다.
        '''),
        code(error_source),
        code(r'''
        evaluation = {"predictions": predictions, "batch1_all": batch1_all, "batch2": batch2, "batch3": batch3}
        error_tables = build_error_tables(evaluation, selected_features)
        importance = extract_importance(final_model, selected_features)
        life_error = error_tables["life_group_error"]
        charging_error = error_tables["charging_error"]
        worst = error_tables["worst_predictions"]
        batch_shift = error_tables["batch_shift"]
        display(life_error.round(2)); display(charging_error.round(2))
        display(worst[["dataset", "global_cell_id", "cycle_life", "prediction", "ape_pct", "error_direction"]].round(2))
        '''),
        code(r'''
        life_order = ["단수명(<500)", "중간수명(500~1,000)", "장수명(>1,000)"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=life_error, x="life_group", y="mape_pct", hue="dataset", order=life_order, palette=colors, ax=axes[0])
        axes[0].set(title="수명 구간별 상대오차", xlabel="실제 수명 구간", ylabel="MAPE (%)"); axes[0].tick_params(axis="x", rotation=15)
        sns.barplot(data=charging_error, x="c_rate_bin", y="mape_pct", hue="dataset", palette=colors, ax=axes[1])
        axes[1].set(title="충전속도 구간별 오차", xlabel="초기 C-rate 구간", ylabel="MAPE (%)"); axes[1].tick_params(axis="x", rotation=15)
        fig.suptitle("어떤 Cell 집단에서 모델이 취약한가?", fontsize=15, fontweight="bold")
        plt.tight_layout(); plt.show()
        '''),
        code(r'''
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        worst_plot = worst.sort_values("ape_pct")
        axes[0].barh(worst_plot["global_cell_id"], worst_plot["ape_pct"], color="#D95D5D")
        axes[0].set(title="외부 Batch 최악 예측 Cell Top 10", xlabel="절대백분율오차 (%)", ylabel="Cell")
        imp = importance.sort_values("absolute_importance")
        imp_colors = ["#D95D5D" if value < 0 else "#1A9AA3" for value in imp["importance"]]
        axes[1].barh(imp["feature"], imp["importance"], color=imp_colors); axes[1].axvline(0, color="#334155")
        axes[1].set(title="최종 선형회귀의 표준화 계수", xlabel="계수의 크기와 방향", ylabel="Feature")
        fig.suptitle("실패 사례와 모델이 사용한 신호", fontsize=15, fontweight="bold")
        plt.tight_layout(); plt.show(); display(importance.round(3))
        '''),
        code(r'''
        shift_pivot = batch_shift.pivot(index="feature", columns="batch", values="standardized_mean_shift_vs_batch1").reindex(columns=["Batch 1", "Batch 2", "Batch 3"])
        fig, ax = plt.subplots(figsize=(8.5, 4.3))
        sns.heatmap(shift_pivot, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=.5,
                    cbar_kws={"label": "Batch 1 표준편차 단위 평균 이동"}, ax=ax)
        ax.set(title="최종 Feature의 Batch 간 분포 이동", xlabel="Batch", ylabel="Feature")
        plt.tight_layout(); plt.show(); display(batch_shift.round(3))
        '''),
        md(r'''
        Batch 2의 `delta_q_iqr`는 Batch 1 대비 +0.99 SD, Batch 3은 -1.04 SD 이동했다. 최종 모델에서 Batch 2 외삽을 키웠던 Capacity Feature를 제외했지만, ΔQ 신호와 수명의 관계 자체도 Batch별로 달라 완전한 일반화는 달성하지 못했다.
        '''),
        md(r'''
        ## 11. ESS 운영 해석과 한계

        - **과대 예측:** 점검·교체 지연과 안전 여유 감소. Batch 2에서 집중된 위험이다.
        - **과소 예측:** 조기 교체와 예비품·인력 과투입.
        - 현재 모델은 drift 경고, 조건별 재검증, 안전 마진 없이 운영에 직접 배포할 수 없다.
        - Batch 1 labeled Cell은 46개로 작고 실험실 데이터가 실제 ESS의 온도 구배·불균형·가변 부하를 모두 대표하지 않는다.
        - 충전정책은 Batch·실험시기·Cell 구조와 함께 변하므로 C-rate의 인과효과로 단정할 수 없다.
        '''),
        md("## 12. 제출 전 자동검사"),
        code(r'''
        checks = {
            "입력 Feature에 미래정보 없음": not any(token in name.lower() for name in selected_features for token in FORBIDDEN_FEATURE_PATTERNS),
            "최종 Feature는 ΔQ 대표값 1개": selected_features == ["delta_q_iqr"],
            "최종 모델 선형회귀": selected_model_name == "Linear Regression",
            "추가 Hyperparameter 없음": selected_params == {},
            "필수 평가 4구간": set(performance["dataset"]) == {"Train (Batch 1 CV)", "Batch 1 Hold-out", "Batch 2 Test", "Batch 3 Test"},
            "필수 Gap 4개": set(gaps["gap"]) == {"Train-Valid", "Valid-Test", "Target-Test", "Batch2-Batch3"},
            "Batch 2·3 Target 적합·선정 미사용": (
                lock["batch2_target_used_in_model_fit_or_candidate_scoring"] is False
                and lock["batch3_target_used_in_model_fit_or_candidate_scoring"] is False
            ),
            "사후 프로토콜 개정 공개": lock["protocol_amendment_motivated_by_prior_batch2_failure"] is True,
            "예측 결측 없음": predictions["prediction"].notna().all(),
        }
        display(pd.DataFrame([{"검사": name, "결과": "통과" if passed else "실패"} for name, passed in checks.items()]))
        assert all(checks.values())
        print("모든 제출 핵심검사 통과")
        '''),
        md(r'''
        ## 13. 최종 결론

        1. ΔQ screening과 F0~F4 Ablation 후, C-rate 그룹 외삽에서 최악 오차가 가장 작은 `delta_q_iqr` 단일 Feature를 선택했다.
        2. 다섯 모델을 같은 nested group validation으로 비교하고 **Linear Regression**을 선택했다.
        3. Batch 1 CV 8.34%, hold-out 5.28%, Batch 2 24.68%, Batch 3 14.10%였다.
        4. Batch 2의 단수명 Cell 대부분을 과대 예측했으므로 안전 관점에서 직접 배포할 수 없다.
        5. 가장 중요한 결론은 내부 CV뿐 아니라 **Batch drift와 과대 예측 방향을 운영 중 감시해야 한다**는 것이다.

        부족한 Feature나 모델을 추가할 때도 `feature_sets`, `model_specs`, 오류 분석 셀을 직접 수정해 확장할 수 있다.
        '''),
        md(r'''
        ## 14. 추가 개선 실험의 위치

        다음 분석은 최초 F2 Ridge 외부평가 이후 수행한 **post-hoc representation study**다. 최초 F2 Ridge의 Batch 2 MAPE 36.56%와 개정 F1 선형회귀의 24.68%와는 별도로, 곡선 전체 표현의 가능성을 진단한다.

        - 후보 선택: Batch 1 development nested CV만 사용
        - 후보: 논문형 log-ElasticNet, ΔQ 곡선 PLS/PCA-Ridge, 다중 Cycle ΔQ PLS
        - 고정 hold-out: 내부 안정성 보조 확인
        - Batch 2·3: 후보 잠금 이후 평가

        Batch 2 결과를 이용해 최종 후보를 다시 선택하지 않으며, 후보 전체의 외부 결과는 성능 개선 가능성을 확인하는 사후 민감도 분석으로만 제시한다.
        '''),
        code(r'''
        from dataclasses import dataclass
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.decomposition import PCA
        from sklearn.model_selection import KFold

        V2_RESULT_DIR = ROOT / "results" / "day2_v2"
        TRAJECTORY_CYCLES = tuple(range(20, 101, 10))
        '''),
        code(v2_model_source),
        md(r'''
        ## 15. v2 후보와 Nested CV

        `log_target()`은 학습 시 `log(cycle_life)`, 예측 시 `exp()`를 Pipeline과 결합한다. PCA·PLS·Scaler·Imputer는 각 fold의 train에만 fit된다. Outer CV가 성능을 추정하고 inner CV가 component 수와 규제강도를 선택한다.
        '''),
        code(r'''
        v2_data = pd.read_csv(V2_RESULT_DIR / "extended_feature_table.csv")
        v2_curves = np.load(V2_RESULT_DIR / "delta_q_curve_data.npz", allow_pickle=False)
        delta_q_curve = v2_curves["cycle100_minus10"]
        delta_q_trajectory = v2_curves["trajectory"]
        v2_candidates = build_candidates(v2_data, delta_q_curve, delta_q_trajectory)

        RUN_V2_NESTED_SEARCH = False
        if RUN_V2_NESTED_SEARCH:
            y_v2 = v2_data["cycle_life"].to_numpy(float)
            dev_v2 = np.flatnonzero(v2_data["split"].eq("development").to_numpy())
            v2_rows, v2_folds = [], []
            for candidate in v2_candidates:
                folds, summary = nested_score(candidate, dev_v2, y_v2)
                v2_folds.append(folds)
                v2_rows.append({
                    "candidate": candidate.name, "feature_family": candidate.feature_family,
                    "feature_count": candidate.feature_count, **summary,
                })
            v2_comparison = pd.DataFrame(v2_rows).sort_values("nested_cv_mape_mean")
        else:
            v2_comparison = pd.read_csv(V2_RESULT_DIR / "nested_cv_comparison.csv")

        display(v2_comparison[[
            "candidate", "feature_family", "feature_count", "nested_cv_mape_mean",
            "nested_cv_mape_std", "worst_fold_mape", "selected_v2"
        ]].round(3))
        '''),
        md(r'''
        Batch 1 nested CV에서는 논문형 ElasticNet `log(y)`가 7.21%로 가장 낮았다. 이 후보를 잠근 뒤 외부평가했지만 Batch 2 MAPE는 47.39%로 악화됐다. 내부 CV 개선이 다른 Batch의 개선을 보장하지 않는다는 결과다.
        '''),
        md(r'''
        ## 16. 사전 정의 후보의 외부 민감도 분석

        다음 표는 Batch 2를 이용한 모델 선택표가 아니다. v2 코드에 사전 정의되어 있던 각 표현을 같은 방식으로 적합했을 때 외부 Batch에서 어떤 행동을 보였는지 확인한 진단표다.
        '''),
        code(r'''
        v2_locked_performance = pd.read_csv(V2_RESULT_DIR / "v1_v2_performance.csv")
        external_diagnostic = pd.read_csv(V2_RESULT_DIR / "predeclared_candidate_external_performance.csv")
        display(v2_locked_performance.round(3))
        display(external_diagnostic.query("dataset == 'Batch 2 Test'")[[
            "candidate", "feature_family", "mape_pct", "mae", "rmse", "r2",
            "mean_bias", "over_prediction_pct", "best_params"
        ]].sort_values("mape_pct").round(3))
        '''),
        md(r'''
        ### 핵심 발견

        - `ΔQ 곡선 PLS log(y)`: Batch 2 MAPE **25.54%**
        - `ΔQ 곡선 PCA-Ridge log(y)`: Batch 2 MAPE **25.84%**
        - 최초 v1 F2 Ridge(백업): Batch 2 MAPE **36.56%**

        ΔQ 전체 곡선 표현은 Batch 2에서 20%대에 도달했다. 그러나 Batch 1 nested CV에서는 공식 Scalar 모델보다 낮은 순위였다. 따라서 이 모델을 Batch 2 결과만으로 공식 최종 모델로 승격할 수는 없다. 새로운 독립 Batch에서 재검증되기 전까지는 **성능 개선 가능성을 보여주는 사후 결과**로 해석한다.
        '''),
        code(r'''
        locked = json.loads((V2_RESULT_DIR / "v2_external_evaluation_lock.json").read_text(encoding="utf-8"))
        checks_v2 = {
            "공식 v1 결과 보존": (RESULT_DIR / "external_evaluation_lock.json").exists(),
            "v2 선택은 Batch 1 development": locked["selection_data"] == "Batch 1 development only",
            "Batch 2 선택 미사용": locked["batch2_target_used_for_selection"] is False,
            "사후 재튜닝 금지": locked["post_external_retuning_allowed"] is False,
            "초기 100 cycle 표현": delta_q_curve.shape[1] == 1000 and delta_q_trajectory.shape[1] == 9000,
        }
        display(pd.DataFrame([{"검사": key, "결과": "통과" if value else "실패"} for key, value in checks_v2.items()]))
        assert all(checks_v2.values())
        '''),
        md(r'''
        ## 17. 추가 개선 실험 결론

        1. 원 논문형 Scalar 모델은 Batch 1 내부 CV를 개선했지만 Batch 2 일반화는 악화됐다.
        2. ΔQ 전체 곡선을 PLS/PCA로 압축하면 Batch 2 MAPE가 25%대로 감소했다.
        3. 내부 CV 순위와 외부 Batch 순위가 달라 단일 Batch 선택의 한계가 확인됐다.
        4. 25%대 결과는 Batch 2를 본 뒤 확인한 사후 민감도 결과이므로 공식 v1을 대체하지 않는다.
        5. 다음 단계는 ΔQ 곡선 모델을 미사용 독립 Batch에 사전 잠금한 뒤 평가하는 것이다.
        '''),
    ])


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        NOTEBOOK_DIR / "02_feature_engineering.ipynb": build_feature_notebook(),
        NOTEBOOK_DIR / "03_modeling.ipynb": build_model_notebook(),
    }
    for path, nb in outputs.items():
        nbf.validate(nb)
        nbf.write(nb, path)
        print(f"created: {path.relative_to(ROOT)} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
