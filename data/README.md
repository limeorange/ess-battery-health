# 데이터 배치 안내

이 디렉토리에는 Severson et al. (2019)의 배터리 수명 실험 MATLAB v7.3 파일을 배치한다. 원본 파일은 수 GB 크기이므로 Git 저장소에 포함하지 않는다.

## 필요한 파일

| 파일 | Day 2 역할 |
|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 — 모델 개발, 반복 CV, 고정 hold-out |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 — 필수 최종 외부 테스트 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 — 추가 일반화 테스트 |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` | Extra — 기본 분석 범위 밖 |

## 데이터 사용 원칙

- 분석 단위는 Cell이며 모델 입력은 Cell당 1행이다.
- Summary Feature는 실제 cycle 2~100, ΔQ(V)는 cycle 10과 100만 사용한다.
- Batch 2·3 Target은 Feature·모델·Hyperparameter 선택에 사용하지 않는다.
- 원본 파일명·크기·SHA-256은 분석 실행 후 `results/day2/data_file_checksums.csv`에 기록된다.

## 출처

Severson et al. (2019), *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy, DOI: `10.1038/s41560-019-0356-8`.
