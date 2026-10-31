# Experiment 4 — Forecast Error Analysis Across Epidemic Conditions

Phân tích **sMOA Python dùng bộ paper `N=18.387, k=5, m=4.422`**; không dùng
winner BO. Thực nghiệm tập trung rút ra insights về hành vi sai số
của sMOA theo điều kiện dịch. Đây là thực nghiệm bổ sung, không phải hình đã
xuất bản trong paper.

**Đã chạy xong full**: xem [kết quả và nhận xét](#results). sMOA có MAE
5.310,91, RMSE 20.620,47 trên 10.000 targets, bridge=omit; báo cáo phân tích sai số theo
phase, mức ca, volatility, horizon và số tuần history.

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
outputs/experiment_4_omit/
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
Lệnh tái chạy run cũ nằm trong `outputs/experiment_4/launch.json`; bản mới chạy theo lệnh trên,
source snapshot đã kiểm tra nằm ở `outputs/experiment_4_omit/source_snapshot/`.
Chỉ báo hoàn tất khi `progress.json` có `status=complete`; không coi log launch
là kết quả đã chạy xong. Không chạy lại BO.

Embedding toàn bộ synthetic library đọc streaming và dùng memmap trên đĩa,
giữ đúng thứ tự/window/median của `forecasting/model.py`, không giảm số curves
hoặc approximate nearest neighbors. Mặc định **bridge=omit**, đồng nhất nhánh
`full`: embed thêm một future difference rồi bỏ difference nối history/future.
`--bridge-mode keep --output-dir outputs/experiment_4_keep_new` chạy đối chứng
trong thư mục riêng. UQ không ảnh hưởng point predictions và không chạy ở đây. Cache cần khoảng 1 GB; setup/index được
tách khỏi runtime forecast. Synthetic library thực tế có 55.161 attempts,
54.081 curves được chấp nhận; không sinh bù 1.080 seasonal attempts bị loại.
Python RNG/GAM vẫn có khác biệt so với R/mgcv được ghi trong report.

`prepare_panel.py --offline` chỉ tạo panel các vintage đã cache; runner từ chối
panel thiếu trừ khi ghi rõ `--allow-incomplete`. Không dùng chế độ đó để công bố
đã hoàn tất thiết kế 50 origins.

## Audit với full và cách đọc hình

```bash
.venv/bin/python experiments/error_analysis/audit.py
```

Kết quả cũ **keep** giữ nguyên tại `outputs/experiment_4`; kết quả mới **omit**
tại `outputs/experiment_4_omit`. Cùng 10.000 targets, library, GAM, N/k/m và labels;
không thay đổi lịch hay ngưỡng nhóm để cải thiện điểm. Trên 6.200 targets chung,
old keep = full keep; new omit = full omit, point-by-point, max difference 0.
126.450 historical values chung cũng khớp. Audit gồm CSV đối chiếu và hashes.

- `forecast_overview`: CA/TX/NY/FL cố định, toàn bộ scheduled origins, horizon 1/4.
- `error_quantile_trajectories`: percentiles 10/50/90/99 của mean normalized error
  mỗi case qua bốn horizons, chọn hậu nghiệm để thấy nhiều mức lỗi.
- `extreme_trajectories`: **bốn cases sai nhất**, không đại diện forecast accuracy
  điển hình. Giữ hình để phân tích failure modes, không thay bằng các ca đẹp.

Final history được vẽ cạnh as-of input để phát hiện revisions; không đưa final
history vào model. Khác UQ chỉ ảnh hưởng intervals/WIS, không sửa đường point.
Không so pooled MAE giữa hai tập origins khác nhau để kết luận model tốt hơn.

<a id="results"></a>

## Kết quả và audit hiện hành

Bản hiện tại dùng **N=18.387/family, k=5, m=4.422, GAM/GCV, bridge=omit**,
50 bang × 50 origins × 4 horizons = **10.000 targets**, không có fit lỗi.
Output: `outputs/experiment_4_omit/`; notebook
[experiment_4.ipynb](experiment_4.ipynb). Bản keep cũ giữ nguyên tại
`outputs/experiment_4/`, gồm protocol, source snapshot, predictions và report.

### Vì sao hình prediction trông sai nhiều?

**Hình cũ `extreme_trajectories` chọn đúng bốn cases có normalized error lớn
nhất trong 2.500 cases**, mỗi case bốn horizons. Đó là hình failure analysis,
không phải một mẫu đại diện. Caption cũ chưa làm rõ mức độ chọn lọc này.

Có một khác biệt cấu hình thật: Experiment 4 cũ giữ bridge difference, trong
khi nhánh `full` hiện bỏ bridge. Đã thêm lựa chọn bridge rõ vào protocol, đổi
mặc định thành omit và chạy lại trên đúng cùng 10.000 targets. Không đổi
N/k/m, history, labels, schedule, smoothing hoặc ngưỡng chia nhóm.

### Kiểm tra đối chiếu trực tiếp

| Đối chiếu trên cùng state/origin/target | Targets | Point khác nhau | Label khác nhau | Max point difference |
| --- | ---: | ---: | ---: | ---: |
| Experiment 4 cũ keep vs full cũ keep | 6.200 | 0 | 0 | 0 |
| Experiment 4 mới omit vs full hiện tại omit | 6.200 | 0 | 0 | 0 |

31 origins chung ×50 bang ×4 horizons. Toàn bộ 126.450 giá trị history chung
cũng khớp. Kiểm tra target_end_date = origin +7×horizon, tính lại absolute errors
và giữ nguyên labels đều pass. Chưa thấy lỗi lệch một tuần, ghép nhầm bang,
hoặc dùng một bộ prediction khác trong Experiment 4.

Trên cùng 6.200 targets, MAE keep là 7.552,67, omit là 7.400,52. Vì tập test
Experiment 4 và full khác calendar, pooled MAE của cả hai không thể so trực tiếp.
Khác policy UQ không làm thay đổi point forecasts; Experiment 4 chỉ chấm điểm
point, không tính intervals/WIS.

Audit có thể chạy lại bằng `python experiments/error_analysis/audit.py`;
chi tiết ở `outputs/experiment_4_omit/audit/{comparison.json,bridge_comparison.csv,provenance.json}`.

### Kết quả trên toàn bộ cùng 10.000 targets

| Phiên bản | MAE | RMSE | Mean normalized absolute error | Extreme rate (>1) |
| --- | ---: | ---: | ---: | ---: |
| Cũ, bridge keep | 5.479,11 | 20.722,37 | 0,756 | 18,12% |
| Mới, bridge omit | 5.310,91 | 20.620,47 | 0,741 | 17,53% |

MAE giảm khoảng 3,07%; không phải mọi case đều tốt hơn. Median normalized
absolute error bản omit là 0,294, trong khi mean là 0,741: một số sai số lớn
kéo mean lên. Chỉ số này chia cho mean 4 tuần history tại origin (floor 1),
không phải percentage error chia cho actual tương lai.

| Horizon | MAE omit | Extreme rate |
| --- | ---: | ---: |
| 1 tuần | 2.986,59 | 7,00% |
| 2 tuần | 4.953,46 | 15,28% |
| 3 tuần | 6.439,19 | 21,68% |
| 4 tuần | 6.864,39 | 26,16% |

Horizon 4 có MAE khoảng 2,30 lần horizon 1. Các hình chỉ xem horizon ngắn hoặc
origins ít biến động có thể trông tốt hơn hình chọn ca sai nhất ở cả bốn horizons.

### Failure modes thực sự vẫn còn

- **Florida, origin 2021-11-27:** as-of history cuối là 0 (final tuần đó cũng 0),
  sau tuần trước 9.891. Cả keep và omit dự báo 1 ở cả bốn horizons. Actual các
  tuần sau là 19.555, 12.984, 28.841, 124.865. Model floor history ở 1, lấy mức
  cuối làm base và không nhận diện lịch reporting. Đây là hạn chế có thật,
  không phải lỗi vẽ hoặc chắc chắn do revision. Không tự impute zero để sửa điểm.
- **Alabama, origin 2021-12-25:** omit dự báo horizon 4 là 17.465,5, actual
  132.765. Tăng trưởng tương lai vượt xa analog forecast.
- **West Virginia, origin 2020-04-04:** omit horizon 4 là 1.652, actual 175.
  History ngắn/tăng nhanh dẫn đến tiếp tục tăng trong khi actual giảm.

Đó là những ca quan sát hậu nghiệm, không chứng minh mọi outbreak đều thất bại.
Raw point forecasts dùng cùng core Python với full, exact L1 neighbors và
synthetic library hash `374329bac3b4bbc28cfc04b6e5f7ff2edcefe9a56979c45b6344dc5ff1c9d9ff`.

### Insights cập nhật, bản omit

| Phase tại origin | Targets | MAE | Normalized MAE | Bias | Extreme rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rising | 4.520 | 7.268,40 | 1,048 | −655,71 | 27,26% |
| Stable | 1.860 | 4.373,59 | 0,556 | −1.159,65 | 11,72% |
| Declining | 3.620 | 3.348,35 | 0,453 | −1.686,16 | 8,37% |

Rising khó hơn stable ở cả hai thang. Bias rising đã đổi dấu sau khi bỏ bridge;
nhận xét bias dương của bản keep không áp dụng cho bản mới. Rising có 44,78%
instances underprediction: mean bias âm không có nghĩa đa số cases dự báo thấp.
Volatility cao: normalized MAE 1,208/extreme rate 30,40%; thấp: 0,403/8,26%.

Early (≤26 tuần history) có normalized MAE 0,991/extreme rate 26,03%; later
0,624/13,53%. Case-count low có normalized MAE 1,355 trong khi raw MAE chỉ
453,41. Các nhóm tương quan với phase, quy mô bang và calendar; không suy ra
hiệu ứng nhân quả của history hoặc volatility từ bảng mô tả này.

### Hình đã bổ sung

Giữ riêng hình extreme và ghi rõ selection ngay trên hình. Bổ sung:

- `forecast_overview`: bốn bang cố định CA/TX/NY/FL, tất cả origins, horizon 1/4,
  vẽ predictions tại target dates và đối chiếu final weekly series.
- `error_quantile_trajectories`: các examples ở percentile 10/50/90/99 của mean
  normalized error qua bốn horizons mỗi case. Đây cũng là post-hoc selection,
  nhưng cho thấy nhiều mức lỗi; không gọi chúng là random/representative sample.
- Với các trajectories, vẽ cả final history và as-of history để thấy data revisions.
  Chỉ as-of history được dùng cho dự báo.

Có 8 nhóm hình PNG/SVG. Tám tests pass, gồm bridge indexing, embedding/forecast
parity, origin-only features, thresholds, error metrics và quantile selection.
Python RNG/GAM chưa được chứng minh numerical parity với R/mgcv; không điều chỉnh
thuật toán theo các errors vừa xem để ép kết quả gần actual.
