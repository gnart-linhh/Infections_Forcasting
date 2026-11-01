# Experiment 4 — Forecast Error Analysis Across Epidemic Conditions

Bản hiện hành dùng **E2 v4 / `nonconstant_v2`**, output riêng
`outputs/experiment_4_nonconstant_v2/`. Kết quả v1 được lưu nguyên ở
[report cũ](../../outputs/experiment_4_omit/report.md).

Phân tích **sMOA Python dùng bộ paper `N=18.387, k=5, m=4.422`**; không dùng
winner BO. Thực nghiệm tập trung rút ra insights về hành vi sai số
của sMOA theo điều kiện dịch. Đây là thực nghiệm bổ sung, không phải hình đã
xuất bản trong paper.

## Thiết kế cố định trước khi chấm điểm

50 bang; 50 origins: mỗi tuần từ 04/04–05/09/2020, sau đó mỗi 4 tuần từ
03/10/2020–01/10/2022; horizons 1–4 tuần. Tổng 2.500 forecast cases,
10.000 targets sMOA nếu không có lỗi fit. Lấy toàn bộ history có sẵn mỗi origin.
Không chọn origins theo mức sai số hay chỉ chọn các đợt dịch đẹp.

History là snapshot **JHU đúng thời điểm**, tổng hợp county → bang và ngày →
tuần Chủ nhật–Thứ bảy, theo cách lấy dữ liệu của LANL: origin Thứ bảy,
cutoff Chủ nhật `origin+1`, phát hành Thứ hai. Không gọi đó là dữ liệu đã có
ngay Thứ bảy. Final truth chỉ dùng làm nhãn chấm điểm. Calendar bắt đầu tháng
4 vì archive index hiện có bắt đầu 31/03/2020; không lấy snapshot tháng 8 gán
ngược cho tháng 3. Nguồn và commit bất biến ghi trong `origins.csv`/provenance.

Các quy tắc nằm trong [conditions.py](conditions.py), đóng băng vào
`protocol.json` **trước khi sinh dự báo**, không dùng quantiles tính trên test:

| Chiều | Quy tắc chỉ dùng history tại origin |
| --- | --- |
| Epidemic phase | `g=(mean(last 2)+1)/(mean(previous 2)+1)-1`; rising nếu `g>10%`, declining nếu `g<-10%`, còn lại stable |
| Case-count level | Mean 4 tuần cuối: low `<1.000`, medium `[1.000,10.000)`, high `≥10.000` ca/tuần/bang |
| Volatility | SD mẫu của `diff(log1p(last 8 weeks))`: low `<0,15`, medium `[0,15;0,40)`, high `≥0,40` |
| Horizon | 1, 2, 3, 4 tuần |
| Data availability | Early: `≤26` tuần history quan sát; later: `>26` tuần |

History được floor tại 1 như forecast sMOA; số tuần âm/zero bị floor vẫn ghi
trong output để kiểm tra revisions. Các ngưỡng là lựa chọn của Experiment 4,
không phải tuyên bố rằng paper đã dùng các nhóm này.

## Metrics và kết quả

- MAE, RMSE, signed bias = prediction − actual; âm nghĩa là dự báo thấp.
- Normalized absolute error = `abs(prediction-actual)/max(mean(last 4 history),1)`.
  Không chia cho future actual hoặc loại các tuần actual bằng zero.
- Extreme error: normalized absolute error **>1**; sensitivity **>0,5** và **>2**.
- Khoảng tin cậy percentile bằng 1.000 bootstrap samples theo **bang**,
  giữ các origins/horizons cùng bang trong một cluster; seed 20261030.

Output:

```text
outputs/experiment_4_nonconstant_v2/
  protocol.json, progress.json, run.lock
  instances.csv             lỗi và đặc trưng từng target, đủ để audit nhóm
  case_status.csv           mọi case, kể cả lỗi; runtime mỗi forecast case
  overall.csv               MAE, RMSE, normalized MAE, bias, extreme rate
  phase.csv, case_level.csv, volatility_level.csv, horizon.csv, history_stage.csv
  phase_horizon.csv, phase_case_level.csv, history_stage_horizon.csv
  group_counts_*.csv        số cases/bang/origins, hiện rõ nhóm rỗng
  extreme_cases.csv         30 origins sai lớn nhất, không lặp 4 horizons
  extreme_profiles.csv      phân bố các extreme instances
  error_quantile_cases.csv  bốn cases tại các error percentiles cố định
  audit/                    đối chiếu cùng cohort với full, kèm hashes
  report.md                 số thực tế và nhận xét tự sinh từ run
  figures/                  PNG + SVG cho 8 nhóm biểu đồ
  checkpoints/              tiếp tục từ origin đã hoàn tất
```

Report trả lời rising so với stable, bias khi tăng dịch, tốc độ tăng lỗi theo
horizon và các đặc trưng chung của extreme errors. Có raw error lẫn normalized
error để tránh kết luận chỉ do bang lớn. Những nhận xét này là mô tả; phase,
case level, volatility và lịch sử có thể gây nhiễu lẫn nhau. `early` đo số tuần
thực có, không phải “ít ca”. Calendar lấy mẫu dày hơn đầu dịch nên pooled MAE
không phải trung bình với trọng số đều theo năm. Không tính WIS vì thí nghiệm
này phân tích point errors; UQ/WIS vẫn ở experiment forecasting riêng.

## Chạy

Từ root repo, dùng `.venv` hiện có; không cần dependency mới cho Experiment 4:

```bash
.venv/bin/python -u experiments/error_analysis/prepare_panel.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/error_analysis/experiment_4.py
```

Tiếp tục cùng inputs/code/rules:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/error_analysis/experiment_4.py --resume
.venv/bin/python -m unittest discover -s experiments/error_analysis -v
.venv/bin/python experiments/notebooks.py experiments/error_analysis/experiment_4.ipynb
```

[Notebook xem kết quả](experiment_4.ipynb) đọc progress, tables và các hình.
Runner yêu cầu E2 v4 đã complete tại `outputs/experiment_2_nonconstant_v2/`.
Trước khi đánh dấu complete, runner kiểm tra SHA numerical core, packed bank,
context và group artifacts; tự tính lại 10.000 forecasts rồi đối chiếu **chính xác**
point forecasts, actuals và toàn bộ target keys với seed 0 của E2 v4.
Notebook xác minh thêm protocol, filter, input/source/artifact hashes và cache;
trạng thái `complete` đơn lẻ không đủ để hiển thị kết quả.

Embedding toàn bộ synthetic library đọc streaming và dùng memmap trên đĩa,
giữ đúng thứ tự/window/median của `forecasting/model.py`, không giảm số curves
hoặc approximate nearest neighbors. Mặc định **bridge=omit**, đồng nhất nhánh
`full`: embed thêm một future difference rồi bỏ difference nối history/future.
UQ không ảnh hưởng point predictions và không chạy ở đây. Cache cần khoảng 1 GB; setup/index được
tách khỏi runtime forecast. Synthetic library thực tế có 55.161 attempts,
54.081 curves được chấp nhận; không sinh bù 1.080 seasonal attempts bị loại.
Python RNG/GAM vẫn có khác biệt so với R/mgcv được ghi trong report.

`prepare_panel.py --offline` chỉ tạo panel các vintage đã cache; runner từ chối
panel thiếu. Nhánh đồng bộ E2 yêu cầu đủ 50 bang và 50 origins, bridge=omit.

## Đối chiếu và cách đọc hình

Đối chiếu E2 nằm trong `progress.json:e2_parity`; không tái dùng audit 6.200
shared targets của các full runs cũ để xác nhận bộ mới. Không đổi lịch,
ngưỡng nhóm, bootstrap hoặc matching để cải thiện điểm.

- `forecast_overview`: CA/TX/NY/FL cố định, toàn bộ scheduled origins, horizon 1/4.
- `error_quantile_trajectories`: percentiles 10/50/90/99 của mean normalized error
  mỗi case qua bốn horizons, chọn hậu nghiệm để thấy nhiều mức lỗi.
- `extreme_trajectories`: **bốn cases sai nhất**, không đại diện forecast accuracy
  điển hình. Giữ hình để phân tích failure modes, không thay bằng các ca đẹp.

Final history được vẽ cạnh as-of input để phát hiện revisions; không đưa final
history vào model. Khác UQ chỉ ảnh hưởng intervals/WIS, không sửa đường point.
Không so pooled MAE giữa hai tập origins khác nhau để kết luận model tốt hơn.

<a id="results"></a>

## Kết quả hiện hành

Đọc [report mới](../../outputs/experiment_4_nonconstant_v2/report.md) và notebook;
chỉ sử dụng số liệu sau khi kiểm tra provenance thành công.

Lượt chạy mới đã **complete**: 2.500/2.500 cases hợp lệ, 0 fit lỗi,
10.000/10.000 point forecasts và actuals khớp chính xác E2 v4 seed 0;
max point difference = 0. Cache có 12.921.078 windows, zero constant histories.

MAE **5,310.91**, RMSE **20,620.47**, normalized MAE **0.7413**, bias **-1,122.47**; extreme rate **17.53%**.

Đã đối chiếu trực tiếp với output omit v1: cả 10.000 points không đổi.
Các chỉ số trùng là kết quả được tính lại, không lấy từ báo cáo cũ.
Các ngưỡng, bootstrap và quy tắc phân nhóm giữ nguyên. Notebook đã chạy đủ 6 code cells.
