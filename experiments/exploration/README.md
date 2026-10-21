# Dataset exploration

Khám phá hai bộ dữ liệu trước khi train/backtest:

- Synthetic: `outputs/synthetic_demo/curves.csv`.
- Final truth của 50 bang: các CSV được liệt kê trong `outputs/truth_states/manifest.csv`.

## Notebook

Mở [dataset_exploration.ipynb](dataset_exploration.ipynb), chọn kernel `.venv` của
project rồi Run All. Notebook có bảng và biểu đồ được lưu trực tiếp trong output
cells để có thể xem ngay trong IDE. Có thể sửa `SYNTHETIC`, `MANIFEST`, `OUT`, K/H
ở cell cấu hình để dùng bộ dữ liệu khác.

```bash
.venv/bin/python -m pip install -r experiments/exploration/requirements.txt
.venv/bin/python experiments/exploration/execute_notebook.py
```

Lệnh thứ hai chạy lại và lưu outputs vào chính notebook. Script dùng Python của
environment đang chạy; không cần checkout `precog/` hoặc R.

## Chạy script, không mở Jupyter

```bash
.venv/bin/python experiments/exploration/explore_datasets.py \
  --synthetic outputs/synthetic_demo/curves.csv \
  --manifest outputs/truth_states/manifest.csv \
  --output-dir outputs/dataset_exploration
```

Đầu vào chỉ được đọc. Chạy lại sẽ ghi lại các file EDA dẫn xuất có cùng tên trong
output directory; không sửa CSV gốc hoặc các kết quả backtest.

## Nội dung

1. **Schema/quy mô:** số curve, quan sát, family, scale và cadence.
2. **Synthetic:** thống kê từng curve/nhóm, phân phối peak và độ dài, floor values,
   step sequence, metadata và số cửa sổ embedding thực sự hợp lệ cho K/H.
3. **Truth:** khoảng ngày mỗi bang, thiếu/không hợp lệ, trùng tuần, khoảng cách
   khác 7 ngày, số âm, số 0, peak và các thay đổi lớn nhất giữa hai tuần.
4. **Biểu đồ:** shape từng family; phân phối synthetic; lịch 50 bang; chuỗi raw của
   California/Texas/New York/Florida; quy mô và thời điểm trên các tuần chung.
5. **Hàm ý cho backtest:** không trộn counts/proportion, không coi một monthly
   step là một weekly step, không gọi zero hay thay đổi lớn là lỗi để tự loại bỏ.

## Output

Trong `outputs/dataset_exploration/`:

- `report.md`: báo cáo kèm hình.
- `figures/`: 5 biểu đồ, mỗi hình có cả PNG và SVG để xuất/chia sẻ.
- `synthetic_curves.csv`, `synthetic_groups.csv`: bảng thống kê synthetic.
- `truth_series.csv`: bảng thống kê từng bang.
- `truth_anomalies.csv`: các dòng cần xem xét, có cột flags; bao gồm zero nhưng
  không khẳng định zero là lỗi dữ liệu.
- `largest_weekly_changes.csv`: 20 thay đổi raw lớn nhất, chỉ để kiểm tra.
- `summary.json`, `metadata.json`: số liệu tổng, phiên bản và checksum input/code.

Raw values được giữ nguyên trong thống kê. Plot normalized chỉ để mô tả và ghi
rõ cách biến đổi; không dùng full-dataset normalization để đưa vào forecast.
Biểu đồ lịch chỉ thể hiện có dòng CSV, không khẳng định giá trị hợp lệ. Panel
normalized truth dùng các tuần có mặt ở tất cả bang, clip âm về 0 và chia cho
peak riêng từng bang trên khoảng chung đó. Không nội suy tuần thiếu.

EDA mô tả final truth; không thay thế archive as-of hoặc chứng minh kết quả bài
báo. Kết quả EDA không được dùng để chọn hyperparameter trên tập test.

## Kết quả trên bộ hiện tại

- 30 synthetic curves, 8.641 quan sát: 24 weekly / 6 monthly,
  14 counts / 16 proportion; 1.930 giá trị ở floor `1e-10`.
- Với k=5, h=4: 8.371 embedding windows, khớp runner forecasting.
- 50 bang, 7.267 dòng truth. Khoảng lịch chung: 2020-01-25–2022-10-29;
  Arizona có lịch dài hơn tới 2023-02-25.
- 8 giá trị âm ở 7 bang, 324 số 0; 332 giá trị sẽ thay đổi khi floor ở 1.
  Không phát hiện nonfinite, ngày trùng hoặc khoảng cách khác 7 ngày trong
  từng chuỗi truth hiện có.

Notebook đã chạy toàn bộ 11 code cells, lưu 5 hình trực tiếp trong notebook.
Các số trên mô tả bộ synthetic nhỏ và truth local hiện tại, không đại diện cho
mọi cấu hình sinh dữ liệu hoặc mọi snapshot lịch sử.
