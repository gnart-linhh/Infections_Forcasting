# Bước 4 — benchmark nhiều bang, warmup và historical snapshots

Mở rộng từ California sang nhiều chuỗi độc lập. Mỗi bang dùng cùng synthetic
library, k/h/closest, smoother và lịch Monte Carlo; không tune tham số trên điểm
benchmark. Embedding được tạo một lần và dùng chung; dispersion học riêng cho
từng bang/model/horizon.

## Chuẩn bị manifest

File CSV có một dòng cho mỗi chuỗi, đường dẫn tương đối tính từ nơi đặt manifest:

```csv
series,truth,as_of_truth
California,California.csv,
Texas,Texas.csv,
```

Cột `truth` là final truth để chấm điểm, định dạng bước 2. Nếu có CSV sẵn,
không cần checkout precog hoặc thư viện đọc RData.

Đã xuất final truth của 50 bang vào `outputs/truth_states/`, gồm manifest và
provenance với checksum nguồn. Có thể tự chuyển bộ truth gốc bằng Python:

```bash
# Chỉ cần pyreadr cho chuyển định dạng, không cần R.
.venv/bin/python -m pip install pyreadr
.venv/bin/python experiments/forecasting/prepare_state_truth.py \
  --input-dir /path/to/state_truths \
  --output-dir outputs/truth_states_new
```

Script không sinh hoặc giả lập snapshot lịch sử từ final truth.

## Chạy benchmark final truth

```bash
.venv/bin/python experiments/forecasting/run_benchmark.py \
  --manifest outputs/truth_states/manifest.csv \
  --synthetic outputs/synthetic_demo/curves.csv \
  --history-mode final --probabilistic \
  --warmup-start-date 2020-06-06 \
  --start-date 2020-08-15 --end-date 2022-10-01 \
  --max-origins 0 --negative-truth clip \
  --seed 13 --interval-draws 5000 \
  --output-dir outputs/benchmark_50_states_new
```

- `--warmup-start-date` tạo các dự báo trước ngày chấm điểm để tích lũy outcome
  cho dispersion. Các dự báo này không vào MAE/WIS benchmark. Ghi riêng vào
  `warmup_forecasts.csv` của từng bang. Không chọn ngày warmup bằng kết quả test.
- `--end-date` chặn ngày origin, không chặn ngày target. Với h=4, origin cuối
  2022-10-01 cần truth tới 2022-10-29. Trong dữ liệu gốc có bang có lịch dài hơn,
  nên cần cùng mốc kết thúc để so sánh.
- `--negative-truth clip` chấp nhận số ca tuần âm và chặn dưới 1 theo phép
  `pmax(1, value)` trong luồng gốc. Mặc định `reject` dừng nếu có giá trị âm.
  Không thay giá trị raw trong CSV. 50 file nguồn có 8 giá trị âm thuộc 7 bang;
  đây là một quyết định xử lý dữ liệu có ảnh hưởng đến điểm số.
- `--max-origins 0` dùng tất cả origins được chọn cho từng bang. Dùng số nhỏ
  để smoke test; số này không giới hạn warmup.

Output:

- `series_metrics.csv`: MAE, WIS, coverage, ratio so baseline từng bang và đường
  dẫn output. Ratio = mean score sMOA / mean score persistence trên cùng targets.
- `horizon_metrics.csv`: chỉ số từng bang/horizon.
- `summary.json`: **pooled** score (trọng số theo số dự báo), **mean/median ratio
  theo bang** (mỗi bang trọng số bằng nhau), số bang tốt hơn baseline và kết quả
  theo horizon. Macro ratio khác ratio của hai pooled means. Baseline có score
  bằng 0 bị loại khỏi ratio, đếm riêng; không cộng epsilon.
- `metadata.json`: cấu hình, checksum code/manifest, chỉ ghi `status=complete`
  khi tất cả bang chạy xong.
- `01/`, `02/`, ...: output chuẩn của bước 3, metadata/checksum input và warmup
  riêng từng chuỗi. `series_metrics.csv` ánh xạ số thư mục với tên bang.

Nếu có lỗi ở một bang, command dừng, không tự bỏ bang để báo kết quả đẹp hơn.
Thư mục có kết quả từng phần nhưng chưa có summary/metadata hoàn chỉnh không
được coi là benchmark hoàn tất. Dùng một output directory mới để chạy lại.

## Chế độ historical as-of

Thêm đường dẫn snapshot **thật** vào cột `as_of_truth` cho mỗi bang:

```csv
series,truth,as_of_truth
California,California.csv,California_asof.csv
```

Snapshot archive là CSV dài, một chuỗi, ba cột:

```csv
as_of,target_end_date,value
2020-08-15,2020-01-25,0
2020-08-15,2020-02-01,3
```

Ví dụ trên chỉ minh họa schema; file chạy thực phải chứa đủ mọi tuần từ đầu
chuỗi tới origin, cho **từng origin** trong warmup và scoring. `as_of` là ngày
cutoff thông tin của mô hình. Không được gắn nhãn as_of cho final truth sao chép.
Nếu nguồn tải vào ngày hôm sau, cần kiểm tra quy ước thời điểm công bố trước khi
chuyển thành schema này; chưa tự ánh xạ quy ước Chủ nhật của covidHubUtils gốc.

```bash
.venv/bin/python experiments/forecasting/run_benchmark.py \
  --manifest /path/to/asof_manifest.csv --history-mode as_of \
  --probabilistic --warmup-start-date 2020-06-06 \
  --start-date 2020-08-15 --end-date 2022-10-01 --max-origins 0 \
  --output-dir outputs/benchmark_asof
```

Single-series runner cũng nhận `--as-of-truth /path/to/snapshots.csv` và các
tham số warmup/end-date tương tự. Khi dùng snapshots:

1. History để smooth/match và giá trị cuối dùng tái dựng forecast lấy hoàn toàn
   từ đúng snapshot origin.
2. Outcome của forecast cũ được cập nhật từ **snapshot hiện tại** trước khi fit
   MLE, không dùng giá trị đã chỉnh sửa trong final truth.
3. Final truth chỉ dùng để chấm MAE/WIS.
4. Thiếu đúng vintage, thiếu tuần, trùng tuần hoặc có target sau cutoff thì dừng.
   Không tự fallback sang final truth hoặc dùng snapshot tương lai. Snapshot
   values phải không âm; các giá trị thấp hơn 1 được chặn 1 như bước trước.

**Chưa có snapshot lịch sử thật trong workspace**; chế độ as-of mới được kiểm
chứng bằng fixtures trong tests. Benchmark 50 bang lần này vẫn là final-truth
backtest, chưa phải tái hiện dự báo thời gian thực.

## Kiểm chứng và phạm vi còn thiếu

```bash
.venv/bin/python -m unittest discover -s experiments/forecasting -v
```

17 tests bao gồm kiểm tra khi thay toàn bộ final truth, forecast/quantile dùng
as-of không thay đổi nhưng điểm số thay đổi; warmup không vào tập chấm điểm;
6 origins warmup khiến MLE dùng đúng 6/5/4/3 outcomes ở origin chấm đầu tiên.
Giữ kiểm tra tái lập point forecast và quantile của bước 3 khi không bật tùy chọn mới.

Warmup này tính forecast từ lịch sử của từng ngày, khác quy ước prehistory dùng
một snapshot đóng băng 2020-08-15 trong helper gốc. Synthetic vẫn là bản Python
xấp xỉ với 30 curves, closest=50; smoother vẫn là SciPy spline thay GAM. Chưa
có so sánh COVIDHub ensemble, lịch snapshot thật hoặc thư viện 55.161 curves
của cấu hình lớn. Benchmark này mở rộng độ bao phủ đánh giá, không giải quyết
hết các khác biệt phương pháp.


## Kết quả đã chạy: 50 bang trên final truth

Cấu hình cố định: 30 synthetic curves (10 mỗi nhóm), k=5, h=4, closest=50,
spline, NB 5.000 draws, seed=13. Cùng origins 2020-08-15 tới 2022-10-01;
10 warmup origins/bang từ 2020-06-06. Tổng 5.600 scored origins, 22.400 dự báo,
500 warmup origins. Đã chạy thành công đủ 50 bang, không loại bang nào.

| Chỉ số gộp trên mọi dự báo | sMOA | Persistence + NB |
| --- | ---: | ---: |
| MAE | 8.123,37 | 8.393,78 |
| WIS | 5.057,37 | 5.216,35 |
| Coverage 95% | 83,64% | 86,31% |

| So sánh mỗi bang có trọng số bằng nhau | MAE | WIS |
| --- | ---: | ---: |
| Trung bình ratio sMOA / baseline | 1,0258 | 1,0355 |
| Trung vị ratio | 1,0291 | 1,0387 |
| Số bang sMOA tốt hơn | 20/50 | 19/50 |

Ratio dưới 1 là tốt hơn. sMOA có absolute score gộp thấp hơn nhưng macro ratio
cao hơn 1: lợi ích ở các bang có mức sai số lớn không đồng nghĩa cải thiện ở
đa số bang. Coverage 95% vẫn thiếu. Cấu hình nhỏ này chưa tái hiện chất lượng
được công bố trong bài báo.

File local đầy đủ: `outputs/benchmark_50_states_aligned/`. Snapshot có thể xem
trong repo:

- [Summary, cấu hình và audit dữ liệu](results/benchmark_50_states.json).
- [Bảng chỉ số 50 bang](results/benchmark_50_states.csv).
- [Checksum nguồn final truth](results/state_truth_provenance.json).

Lượt chạy thử trước ở `outputs/benchmark_50_states/` đã dừng do giá trị âm; không
dùng thư mục đó để phân tích. Dùng thư mục `_aligned` đã hoàn tất ở trên.

Điểm gộp đã được tính lại độc lập từ các CSV dự báo để đối chiếu summary;
kiểm tra đủ 448 dự báo/bang, 10 warmup/bang và checksum input/code. Ở origin
chấm đầu tiên, MLE có 10/9/8/7 outcome theo horizon. Các output của bước 3
không bị thay đổi khi dùng các tùy chọn mặc định.
