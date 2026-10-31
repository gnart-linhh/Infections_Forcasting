# Experiment 5 — Analogue Quality vs Forecast Accuracy

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

Runner dùng lại `outputs/experiment_2`, `outputs/epidemic_panel` và memmap
`outputs/paper_embedding_k5_h4_omit` đang có. Mean top-m distance chưa được lưu
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

`outputs/experiment_5/`: protocol/progress/checksums, cases.csv,
instances.csv.gz, correlations.csv, grouped_errors.csv, counterexamples.csv,
risk_scores.csv, risk_predictions.csv.gz, report.md và figures PNG/SVG.
Code provenance nhúng trong protocol.json, không nhân bản thư viện/source tree.
Các input của E2/E4 không bị ghi đè. `progress.json` chỉ complete sau khi đủ
analysis/report, không dùng log launch làm bằng chứng đã có kết quả.

## Kết quả trên dữ liệu hiện có

Đã chạy đủ 10.000 primary instances, exact replay khớp toàn bộ forecasts cũ.
Case split: 1.300 train /1.150 test /50 embargo; first test origin 2021-01-23.
Ngưỡng good match q20=0,034042 (scaled mean L1). Không dùng error/test quantiles
để đặt ngưỡng.

| Horizon | Spearman scaled distance vs normalized error | Sau điều chỉnh context |
| --- | ---: | ---: |
| 1 | 0,237 | 0,010 |
| 2 | 0,239 | 0,006 |
| 3 | 0,204 | −0,012 |
| 4 | 0,191 | −0,038 |

Raw mean L1 với raw absolute error có Spearman khoảng 0,60–0,68; khi chia
scale cho cả distance/error, association giảm còn 0,19–0,24. Vì vậy không đọc
raw correlation cao như bằng chứng độc lập của matching quality.

Distance thấp có xu hướng đi cùng normalized error thấp, nhưng quan hệ không mạnh và
phần lớn association biến mất khi kiểm soát mức ca/growth/volatility/history/
phase trong partial rank analysis. Không diễn giải như causal effect hoặc
bằng chứng không còn bất kỳ nonlinear signal nào.

Trong **337 good-match test cases**, extreme errors lần lượt là
10/26/38/46 ở horizons 1/2/3/4 (2,97% /7,72% /11,28% /13,65%). Good distance
không bảo đảm future giống analogue: South Carolina origin 2021-12-25 có
scaled distance 0,03153, h4 forecast 9.057,5 trong khi final actual 112.585;
Nevada cùng origin distance 0,01603, forecast 6.040 vs actual 65.307.
Hình failure chọn hậu nghiệm các ca sai lớn, không phải một mẫu đại diện.

| Horizon | Test AUROC distance-only | Context-only | Context+distance | Paired delta AUROC 95% CI |
| --- | ---: | ---: | ---: | --- |
| 1 | 0,699 | 0,745 | 0,753 | +0,0082 [0,0031; 0,0143] |
| 2 | 0,657 | 0,661 | 0,660 | −0,0009 [−0,0113; 0,0094] |
| 3 | 0,664 | 0,648 | 0,650 | +0,0012 [−0,0004; 0,0029] |
| 4 | 0,660 | 0,666 | 0,651 | −0,0148 [−0,0250; −0,0062] |

Thêm distance chỉ cải thiện ranking nhẹ ở h1 trong split này; h2/h3 không có
incremental AUROC rõ, h4 giảm. Brier của combined so với context lần lượt
thay đổi gần 0 /−0,00195 /−0,00048 /+0,00238: ranking và chất lượng xác suất
không đồng nghĩa. Brier không phải phép đo calibration thuần túy.
Các CIs không điều chỉnh multiple comparisons; không xem một kết quả tốt
riêng lẻ là đủ để đặt alert threshold vận hành.

Secondary median-distance correlations ở hai banks khác khoảng 0,19–0,24,
nhất quán về quy mô association yếu; không mở rộng kết luận mean-distance
hoặc risk-model performance sang các banks chưa đánh giá mean.

Thông tin rút ra: distance có thể là **một feature chẩn đoán có điều kiện**, nhưng
không dùng đơn lẻ như chứng nhận forecast đáng tin. Cần giữ context và horizon,
đọc paired held-out increments và kiểm chứng trên các thời kỳ khác trước khi
công bố khả năng cảnh báo tổng quát. Đây không phải thay đổi thuật toán sMOA.

Có 7 nhóm hình PNG/SVG, bảng đầy đủ và notebook có outputs; 8 tests kiểm tra
ties/rank metrics, frozen bins/scaling, joins, replay parity và training labels
không dùng final future revisions. Checksum/code provenance ở protocol/progress.
