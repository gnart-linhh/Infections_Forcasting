# Experiment 6 — Probabilistic Forecast Evaluation

Bản hiện hành: **`nonconstant_v2` / UQ `matured`**, dùng full đã hoàn tất
`outputs/paper_full_nonconstant_v2_matured/`. Kết quả hồi cứu cũ giữ nguyên ở
[report cũ](../../outputs/experiment_6/report.md).

Đánh giá phân tầng UQ negative binomial gốc của sMOA, dùng lại **22.400 forecasts, 515.200 quantiles** từ `outputs/paper_full_nonconstant_v2_matured/`. Bộ paper **N=18.387, k=5, m=4.422**, GAM, bỏ bridge; không sinh thêm library, chạy BO hoặc thay phương pháp UQ.

Mở [experiment_6.ipynb](experiment_6.ipynb), chọn kernel `.venv`. Kết quả đầy đủ: [report.md](../../outputs/experiment_6_nonconstant_v2_matured/report.md).

```bash
.venv/bin/python experiments/probabilistic/experiment_6.py
.venv/bin/python -m unittest discover -s experiments/probabilistic -v
.venv/bin/python experiments/probabilistic/experiment_6.py --execute-notebook
```

Notebook chỉ đọc kết quả đã chạy. Bốn source files gồm runner/phân tích/plots, tests, notebook và README; dùng lại origin features và WIS convention hiện có. Output nằm trong một thư mục `outputs/experiment_6_nonconstant_v2_matured/`, không sao chép synthetic bank hay forecast runs.

## Thiết kế

- Tính coverage inclusive `L <= y <= U`, MPIW cho đủ **11 levels: 10%, 20%, …, 90%, 95%, 98%**. Đồng thời lưu coverage theo source `L < y <= U` và số targets khác ở biên.
- WIS giữ đúng **bảy quantiles** `.025,.1,.25,.5,.75,.9,.975`; không thay metric bằng trung bình pinball trên grid 23 quantiles. Giữ raw scoring truth và interval bounds sau median recentering.
- Phân tầng theo horizons 1–4; phase rising/declining/stable; case level low/medium/high; lịch sử quan sát; số calibration pairs. Có cả horizon × từng dimension.
- Phase và case level giữ quy tắc Experiment 4, tính từ **vintage input tại origin**. Phase: mean last2 so với previous2 (pseudocount 1), ngưỡng ±10%; level: mean last4, ngưỡng 1.000 và 10.000 ca/tuần.
- Lịch sử đầy đủ là **30–141 tuần**, nên nhóm ≤26 tuần rỗng. Dùng dải cố định 27–52, 53–104 và ≥105 tuần. Đây là lượng history thực tế, không phải thử nghiệm cắt ngắn history.
- Calibration count bins giữ nguyên: <24, 24–49, 50–99, ≥100 pairs. Với matured, horizons dài có thể có 21–23 eligible pairs ngay sau warmup. Toàn bộ scored forecasts đã MLE sau warmup; không có initial-size period để đánh giá chuyển initial→MLE.
- Quan hệ width–error: Spearman riêng từng horizon, raw/normalized/context-adjusted; normalized dùng mean last4 ở origin. CI bootstrap nguyên bang 500 lần, giữ nguyên chuỗi trong bang; không coi overlapping targets độc lập.

## Phạm vi diễn giải

Full dùng **`matured`**: pair phải có `forecast_origin < current origin`,
`target_end_date <= current origin`, cùng bang/horizon, actual hữu hạn và không âm.
Current-origin forecasts và future targets không vào MLE. Điều kiện switch sang
MLE vẫn theo nguồn: ít nhất 24 stored prior rows, không yêu cầu 24 eligible pairs
riêng cho từng horizon. Negative outcomes bị loại khỏi likelihood nhưng giữ khi scoring.

**Không gọi đây là backtest hoàn toàn causal/real-time**: warmup vẫn dùng snapshot
cố định 15/08/2020 và calibration labels vẫn là final revisions. Cutoff/issuance
trong archive là Chủ nhật/thứ Hai sau nhãn origin thứ Bảy.

Biến động coverage/width theo calibration count có thể đến từ lịch dịch, point forecast, quy mô ca và size; không thể quy thành tác động nhân quả của việc “có thêm dữ liệu”. Quan hệ width/error cũng không đủ xác nhận một risk detector ngoài mẫu. Không thay NB hoặc trộn run giữ bridge vào so sánh.

Runner yêu cầu full `complete`, `nonconstant_v2`, đúng paper parameters và
`matured`; xác minh numerical source snapshot/current core, library/truth/vintage hashes,
target alignment, đủ 23 quantiles và WIS/coverage gốc. Dựng lại từng tập calibration
pairs, counts, negative exclusions, future counts và **tính lại mọi `nb_size` để
đối chiếu chính xác**. Không thay saved quantiles hoặc sinh lại draws.

Chỉ xuất report sau khi 22.400 fits hợp lệ, zero future-target use, và WIS/MAE
cùng các chỉ số theo horizon khớp `05_evaluation/summary.json` của full mới.
Coverage inclusive được tính trực tiếp từ quantiles; source lower-open coverage
được lưu riêng, cùng số targets khác nhau tại biên dưới.

`protocol.json` ghi version/policy/filter, nguồn và input/source SHA;
`summary.json` ghi kiểm chứng, software và artifact hashes. Notebook từ chối
complete cũ, policy/filter sai, source/input/artifact bị thay đổi hoặc thiếu targets.
Các CSV chính: `instances.csv.gz`, `metrics.csv`, `calibration.csv`,
`width_correlations.csv`; sáu bộ hình PNG/SVG. Không tạo thêm source file hoặc
sao chép synthetic/forecast banks. Thiết kế, bootstrap và thresholds giữ nguyên.

## Kết quả hiện hành

Xem [report mới](../../outputs/experiment_6_nonconstant_v2_matured/report.md)
và [notebook](experiment_6.ipynb) sau khi toàn bộ kiểm tra hoàn tất.

Lượt mới đã xác minh đủ **22.400 NB sizes** (equality với saved sizes),
515.200 quantiles và mọi calibration count/negative exclusion; **0** future-target
uses. WIS/MAE overall và từng horizon khớp full reproduction matured.

| Horizon | Coverage 95% inclusive | MPIW 95% (ca) | WIS | MAE |
| --- | ---: | ---: | ---: | ---: |
| Overall | 87.21875% | 39,436.28 | 4,747.38 | 7,384.66 |
| 1 tuần | 91.58929% | 23,848.24 | 2,628.44 | 4,173.93 |
| 2 tuần | 89.96429% | 38,115.76 | 4,008.56 | 6,273.88 |
| 3 tuần | 85.80357% | 45,993.75 | 5,655.20 | 8,808.29 |
| 4 tuần | 81.51786% | 49,787.38 | 6,697.30 | 10,282.54 |

Coverage inclusive **87.21875%**, CI state-bootstrap
**[86.03%; 88.38%]**; source lower-open
**87.16964%**. Inclusive thêm **11 targets**
ở biên dưới; hai quy tắc được lưu và hiển thị riêng.

Đã đối chiếu AST: 13 hàm phân tích/metrics/bootstrap/strata không thay đổi.
9 tests đã qua, gồm boundary convention, matured eligibility, bất biến khi sửa
future outcomes/current forecast, phát hiện NB size sai và từ chối complete cũ.

Lượt E6 mới đã **complete**; notebook đã chạy đủ 7 code cells sau khi vượt qua provenance gate.
