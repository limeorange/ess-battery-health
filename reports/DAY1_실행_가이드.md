# Day 1 분석 실행 가이드

## 전체 분석 재실행

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
python src/day1_analysis.py
```

현재 로컬 환경에서는 다음 실행기가 검증되었습니다.

```bash
/opt/homebrew/Cellar/jupyterlab/4.2.5_1/libexec/bin/python src/day1_analysis.py
```

## 환경 설치

```bash
python -m pip install -r requirements-day1.txt
```

## 입력

- `data/2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `data/2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `data/2018-04-12_batchdata_updated_struct_errorcorrect.mat`

Extra 가변충전 파일은 `cycle_life`가 없어 target 기반 Day 1 비교에서 제외했습니다.

## 출력

- `results/day1/`: 셀별 feature, 품질 감사, 통계 검정, feature 결정표
- `figures/day1/`: 한국어 분석 그래프 12개
- `reports/DAY1_분석_보고서.md`: 결과와 모델 전략을 연결한 보고서
- `notebooks/01_EDA.ipynb`: 결과 확인 및 전체 파이프라인 재실행 노트북

## 재현성 규칙

- 난수 시드: 42
- 원본 `.mat` 파일은 읽기 전용으로 접근
- Batch 1 → Batch 2 → Batch 3 순차 처리
- 초기 summary feature는 실제 cycle 2~100으로 통일
- `ΔQ(V)`는 실제 cycle 번호 10과 100에서 계산
- `cycle_life` 결측 셀은 기록 길이로 대체하지 않음
- 0으로 기록된 내부저항·온도 등은 물리적 측정값이 아닌 결측으로 처리
