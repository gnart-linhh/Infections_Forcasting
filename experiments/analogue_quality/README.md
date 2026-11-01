# Experiment 5 — Analogue Quality vs Forecast Accuracy

Bản hiện hành dùng **E2 v4 / `nonconstant_v2`**, output mới
`outputs/experiment_5_nonconstant_v2/`; không resume vào bộ cũ.
[Report v1 lưu riêng](../../outputs/experiment_5/report.md).

Bổ sung phân tích matching quality–forecast error cho **sMOA**, dùng nguyên bộ
paper N=18.387 attempts/family, k=5, m=4.422, GAM/GCV, omit bridge, seed0.
Tập test mô tả giống E2/E4: 50 bang ×50 origins ×4 horizons =10.000 instances.
Không chọn lại hyperparameters từ kết quả này.

[experiment_5.ipynb](experiment_5.ipynb) là notebook chính. Chỉ bốn file nguồn
cần thiết: script, notebook, test và README này. Thiết kế và nhận xét đặt chung
ở đây; bảng/hình/report sinh trong một output folder, không tạo snapshot hay
các phiên bản báo cáo trùng nhau.

## Chạy

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/analogue_quality/experiment_5.py
.venv/bin/python -m unittest discover -s experiments/analogue_quality -v
.venv/bin/python experiments/analogue_quality/experiment_5.py --execute-notebook
```

Runner dùng lại `outputs/experiment_2_nonconstant_v2`, `outputs/epidemic_panel` và memmap
`outputs/paper_embedding_k5_h4_omit_nonconstant_v2` đang có. Mean top-m distance chưa được lưu
trong E2 nên truy vấn lại exact L1, kiểm tra nguyên 10.000 point forecasts và
các min/median/max distances; không sinh synthetic hoặc embedding mới.
Lần chạy sau dùng `cases.csv` đã xác minh checksum, không query lại nếu inputs
và extraction signature không đổi. Notebook chỉ đọc artifact, không chạy model.

## Thiết kế

Mỗi case=state/origin lưu minimum/mean/median/max L1 và features tại origin;
mỗi forecast instance thêm horizon, prediction, actual, absolute error và
normalized absolute error. Absolute error là loss của một instance; MAE là
trung bình loss khi tổng hợp theo nhóm/horizon. Scaled distance=distance/(k×level_scale), với
level_scale=max(mean(last 6 raw observations đã floor 1),1). Normalization của
error vẫn là mean(last 4 floored observations) như E4. Đây là chia distance
sau matching, không đổi neighbor ordering hoặc thuật toán dự báo.

Hai scale này liên quan nhau: tương quan ratios có thể do scale, không tự
chứng minh analogue tốt gây forecast tốt. Báo cả raw/scaled min/mean/median
với raw/normalized errors; Spearman riêng từng horizon; strata phase/case level;
partial rank correlation kiểm soát hai mức ca, growth, volatility, history length
và phase. Phần pooled dùng một mean-error record mỗi case, không nhân sample
size lên bốn lần do lặp horizons. Không dùng naive independent-row p-values.

Bootstrap Spearman chính và risk metrics resample **nguyên bang**, 500 draws,
seed=20261101. Các origins/horizons cùng bang đi cùng cluster; điều này không
loại bỏ mọi phụ thuộc do waves toàn quốc. Strata n<20 hoặc constant input ghi NA.

## Phân nhóm và counterexamples

Ngưỡng quintile lấy một lần từ **training cases**, áp dụng nguyên cho test.
Ties không bị jitter hoặc chia tùy tiện; ngưỡng trùng được gộp, nhóm test có
thể không cân bằng/rỗng. Good match=scaled mean distance ≤training q20;
extreme error=absolute error/origin error scale >1. Export tất cả good-match /
large-error instances; hình minh họa chọn hậu nghiệm bốn cases lỗi lớn ở bốn
bang, không gọi đó là accuracy điển hình hoặc một random sample.

## Kiểm chứng tín hiệu chẩn đoán rủi ro

- Train: cases có cả bốn target dates ≤2020-12-31.
- Test: origins ≥2021-01-01; buffer không dùng để fit/chấm risk diagnostics.
- Training outcomes lấy từ **historical JHU vintage tại first test origin**.
  Chỉ dùng targets đã xảy ra trước cutoff, có mặt trong snapshot trước Monday
  issuance. Không lấy final revisions hoặc outcomes chưa đến hạn để fit risk.
- Diagnostic frozen sau lần test đầu; final truth chỉ đánh giá risk trên test.
- Distance-only dùng log1p(scaled mean distance); context-only dùng các features
  đã biết tại origin; combined thêm đúng distance đó. Logistic fit từng horizon,
  L2=1 và train-only standardization, không chọn feature/penalty theo test.
- AUROC/AP/Brier cùng paired delta AUROC/Brier và state-cluster intervals.
  AP đọc cạnh extreme prevalence; AUROC ranking khác calibration xác suất.

Risk diagnostics dự đoán cờ forecast error, không phải mô hình dự báo số ca
mới. Cần xem distance có tăng held-out performance ngoài context không, không
chỉ nhìn correlation trên toàn mẫu. Đây là một fixed time split, chưa phải
external validation hoặc bằng chứng đủ để đặt cảnh báo vận hành.

Secondary seeds 1/2 của E2 chỉ kiểm tra **median-distance correlation** trên
cùng N/k/m/targets bằng artifacts sẵn có; không bịa mean distance chưa lưu.
Core giữ nguyên floating-point variance-filter behavior đã audit ở E2 để các
forecasts còn đối chiếu được; chưa xác minh Python/R numerical parity.

## Output gọn

`outputs/experiment_5_nonconstant_v2/`: protocol/progress/checksums, cases.csv,
instances.csv.gz, correlations.csv, grouped_errors.csv, counterexamples.csv,
risk_scores.csv, risk_predictions.csv.gz, report.md và figures PNG/SVG.
Code provenance nhúng trong protocol.json, không nhân bản thư viện/source tree.
Các input của E2/E4 không bị ghi đè. `progress.json` chỉ complete sau khi đủ
analysis/report, không dùng log launch làm bằng chứng đã có kết quả.

## Kiểm chứng provenance và kết quả hiện hành

Giữ kiểm tra SHA numerical core với E2; không bỏ guard khi code đã thay đổi.
Runner yêu cầu E2 protocol v4, complete, zero constant histories; xác minh
packed banks/context/group artifacts của cả ba seeds. Cache phải version 2,
`nonconstant_v2`, cùng library SHA và số windows với seed 0. Exact replay kiểm
tra 10.000 points và min/median/max distances của 2.500 cases bằng equality.
Mean distance được tính từ đúng top-m, không thay matching hoặc thresholds.

Notebook chỉ hiển thị khi complete, protocol/filter, source/input/artifact hashes,
cache và số targets/cases đều hợp lệ. Correlations, temporal diagnostics và
figures được sinh cùng lượt; không chép số liệu từ output v1.

Xem [report mới](../../outputs/experiment_5_nonconstant_v2/report.md) và
[notebook](experiment_5.ipynb) cho kết quả đã xác minh.

Lượt chạy mới đã **complete**: exact replay 10.000 forecasts và 2.500 cases;
1.300 train / 1.150 test / 50 embargo. Numerical SHA, input/artifact hashes,
cache `nonconstant_v2` và reference groups của ba seeds đều đã kiểm tra.
Notebook đã chạy đủ 7 code cells.

| Horizon | Spearman scaled mean distance–normalized error | Sau điều chỉnh context |
| --- | ---: | ---: |
| 1 | 0.237 | 0.010 |
| 2 | 0.239 | 0.006 |
| 3 | 0.204 | -0.012 |
| 4 | 0.191 | -0.038 |

Distance có association yếu với lỗi; sau điều chỉnh context, association gần zero.
Incremental held-out AUROC của context+distance so với context-only vẫn nhỏ,
phụ thuộc horizon; không suy ra good match bảo đảm forecast tốt.
Xem paired intervals và Brier trong report trước khi diễn giải tín hiệu rủi ro.

Đối chiếu với v1 xác nhận 10.000 points và toàn bộ min/mean/median distances
không đổi. Các số trên được tính lại từ v4; giữ nguyên bootstrap, temporal split,
ngưỡng good-match/extreme và các tiêu chí matching.
