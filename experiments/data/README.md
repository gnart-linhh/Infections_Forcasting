# Chuẩn bị dữ liệu đầu vào

Các lệnh này chạy khi chuẩn bị nguồn dữ liệu; pipeline forecasting đọc CSV đã
chuẩn bị và không gọi lại chúng ở mỗi lần dự báo.

| Script | Đầu vào → đầu ra |
| --- | --- |
| `fetch_paper_data.py` | Tải snapshot JHU theo commit covidData và forecast COVIDHub thật → CSV, raw cache, provenance (không clone, không R) |
| `prepare_state_truth.py` | State truth `.RData` → CSV từng bang, manifest và provenance; đọc bằng Python/`pyreadr`, không chạy R |
| `prepare_asof.py` | Archive `tdat_list_tot_weekly.csv` → snapshot CSV từng bang bằng streaming |
| `prepare_reference_scores.py` | Original model scores và location table → canonical scores cho paired comparison |

```bash
.venv/bin/python experiments/data/prepare_state_truth.py \
  --input-dir /path/to/state_truths --output-dir outputs/truth_states_new

.venv/bin/python experiments/data/prepare_asof.py \
  --input /path/to/tdat_list_tot_weekly.csv \
  --truth-manifest outputs/truth_states/manifest.csv --output-dir outputs/asof_states

.venv/bin/python experiments/data/prepare_reference_scores.py \
  --input /path/to/scores_tot_w_wis_components.csv \
  --locations /path/to/state_census.csv --output outputs/reference_scores.csv
```

Final truth có ở `outputs/truth_states/`. Dữ liệu historical thật cho Figure 2
đã có ở `outputs/paper_data/`: 4 bang, 4 snapshot (2020-08-15/22/29 và 2020-09-05), và
forecast chính thức baseline/4-week ensemble ở các ngày có incident cases.

```bash
.venv/bin/python experiments/data/fetch_paper_data.py \
  --as-of-dates 2020-08-15 2020-08-22 2020-08-29 2020-09-05
```

Script port cách `covidData` lấy JHU: chọn issue <= Saturday+1 theo danh mục
commit SHA, sum county cumulative theo bang, diff daily, sum Sunday–Saturday,
bỏ tuần chưa đủ. Gắn nhãn as_of=Saturday theo LANL, dù retrieval cutoff là Sunday.
Không chỉnh negative revisions trước aggregation. Raw downloads và SHA được giữ
trong provenance. Các forecast không tồn tại trong archive được ghi 404, không
điền bằng dự báo giả; file có deaths nhưng không có cases cũng không được dùng.

Đây là phần dữ liệu cho Figure 2, chưa phải archive đầy đủ mọi bang/vintage
5,5 GB hay score archive toàn bộ models; các adapter vẫn phục vụ trường hợp đó.
