# Experiment 6 — Probabilistic Forecast Evaluation

Đánh giá phân tầng UQ negative binomial gốc của sMOA, dùng lại **22.400 forecasts, 515.200 quantiles** từ `outputs/paper_full/`. Bộ paper **N=18.387, k=5, m=4.422**, GAM, bỏ bridge; không sinh thêm library, chạy BO hoặc thay phương pháp UQ.

Mở [experiment_6.ipynb](experiment_6.ipynb), chọn kernel `.venv`. Kết quả đầy đủ: [report.md](../../outputs/experiment_6/report.md).

```bash
.venv/bin/python experiments/probabilistic/experiment_6.py
.venv/bin/python -m unittest discover -s experiments/probabilistic -v
.venv/bin/python experiments/probabilistic/experiment_6.py --execute-notebook
```

Notebook chỉ đọc kết quả đã chạy. Bốn source files gồm runner/phân tích/plots, tests, notebook và README; dùng lại origin features và WIS convention hiện có. Output nằm trong một thư mục `outputs/experiment_6/`, không sao chép synthetic bank hay forecast runs.

## Thiết kế

- Tính coverage inclusive `L <= y <= U`, MPIW cho đủ **11 levels: 10%, 20%, …, 90%, 95%, 98%**. Đồng thời lưu coverage theo source `L < y <= U` và số targets khác ở biên.
- WIS giữ đúng **bảy quantiles** `.025,.1,.25,.5,.75,.9,.975`; không thay metric bằng trung bình pinball trên grid 23 quantiles. Giữ raw scoring truth và interval bounds sau median recentering.
- Phân tầng theo horizons 1–4; phase rising/declining/stable; case level low/medium/high; lịch sử quan sát; số calibration pairs. Có cả horizon × từng dimension.
- Phase và case level giữ quy tắc Experiment 4, tính từ **vintage input tại origin**. Phase: mean last2 so với previous2 (pseudocount 1), ngưỡng ±10%; level: mean last4, ngưỡng 1.000 và 10.000 ca/tuần.
- Lịch sử đầy đủ là **30–141 tuần**, nên nhóm ≤26 tuần rỗng. Dùng dải cố định 27–52, 53–104 và ≥105 tuần. Đây là lượng history thực tế, không phải thử nghiệm cắt ngắn history.
- Calibration count bins cố định: 24–49, 50–99, ≥100 pairs. Toàn bộ scored forecasts đã MLE sau warmup; không có initial-size period để đánh giá chuyển initial→MLE.
- Quan hệ width–error: Spearman riêng từng horizon, raw/normalized/context-adjusted; normalized dùng mean last4 ở origin. CI bootstrap nguyên bang 500 lần, giữ nguyên chuỗi trong bang; không coi overlapping targets độc lập.

## Phạm vi diễn giải

Lượt `full` dùng **UQ hồi cứu** theo yêu cầu trước đó: final outcomes của forecasts đã phát, kể cả targets chưa đến hạn, cộng frozen warmup từ snapshot 15/08/2020. Đây không phải real-time calibration. Cutoff/issuance trong archive là Chủ nhật/thứ Hai sau nhãn origin thứ Bảy.

Biến động coverage/width theo calibration count có thể đến từ lịch dịch, point forecast, quy mô ca và size; không thể quy thành tác động nhân quả của việc “có thêm dữ liệu”. Quan hệ width/error cũng không đủ xác nhận một risk detector ngoài mẫu. Không thay NB hoặc trộn run giữ bridge vào so sánh.

Runner kiểm tra target alignment, đầy đủ 23 quantiles, WIS/coverage gốc, hashes truth/vintages và generation metadata. Counts/future-label access được dựng lại từ warmup và forecasts trước origin. `protocol.json` ghi quy tắc/source/checksums; `summary.json` ghi software, trạng thái và artifact hashes. Các CSV chính: `instances.csv.gz`, `metrics.csv`, `calibration.csv`, `width_correlations.csv`; sáu bộ hình PNG/SVG.

## Kết quả trên full cohort

| Horizon | Coverage 95% inclusive | MPIW 95% (ca) | WIS | MAE |
| --- | ---: | ---: | ---: | ---: |
| 1 tuần | 91,59% | 23.848,24 | 2.628,44 | 4.173,93 |
| 2 tuần | 90,09% | 38.098,73 | 3.988,48 | 6.273,88 |
| 3 tuần | 86,02% | 46.088,20 | 5.624,65 | 8.808,29 |
| 4 tuần | 81,79% | 49.837,52 | 6.668,32 | 10.282,54 |
| Overall | **87,37%** | **39.468,17** | **4.727,47** | **7.384,66** |

Overall state-bootstrap CI coverage 95% là **[86,17%; 88,51%]**, thấp hơn nominal. Source strict-lower coverage là 87,33%; đổi inclusive thêm 10 targets ở biên dưới. Truth vượt upper trong 10,21% targets, thấp hơn lower trong 2,42%.

- Declining tại origin: coverage 80,70%; rising 91,52%; stable 92,92%.
- Case-count low: 78,61%; riêng tuần 4: 68,01%.
- Width/error có Spearman raw 0,63–0,69; normalized 0,32–0,43; sau điều chỉnh origin context còn 0,10–0,18. Độ rộng có tín hiệu về độ khó nhưng không bảo đảm từng forecast sẽ đúng.
- Nhóm 24–49 calibration pairs coverage 95,98%, nhóm ≥100 pairs 84,10%; đây là liên hệ qua thời gian, không phải bằng chứng thêm data gây calibration kém. Tất cả forecasts đã MLE, không có đối chứng initial→MLE.

Đã kiểm tra 8 tests và toàn bộ input/quantile/WIS/calibration counts của full run. Notebook hiển thị các bảng và sáu figures; report giải thích các nhóm và giới hạn của quy ước hồi cứu.
