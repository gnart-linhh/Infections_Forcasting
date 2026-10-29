# Dataset exploration

Khám phá hai bộ dữ liệu trước khi train/backtest:

- Synthetic: `outputs/reproduction_full/01_synthetic/curves.csv`.
- Final truth của 50 bang: các CSV được liệt kê trong `outputs/truth_states/manifest.csv`.

## Notebook

Mở [dataset_exploration.ipynb](dataset_exploration.ipynb), chọn kernel `.venv` của
project rồi Run All. Notebook có bảng và biểu đồ được lưu trực tiếp trong output
cells để có thể xem ngay trong IDE. Có thể sửa `SYNTHETIC`, `MANIFEST`, `OUT`, K/H
ở cell cấu hình để dùng bộ dữ liệu khác.

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python experiments/exploration/execute_notebook.py
```

Lệnh thứ hai chạy lại và lưu outputs vào chính notebook. Script dùng Python của
environment đang chạy; không cần checkout `precog/` hoặc R.

## Chạy script, không mở Jupyter

```bash
.venv/bin/python experiments/exploration/explore_datasets.py \
  --synthetic outputs/reproduction_full/01_synthetic/curves.csv \
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

## Quy mô dữ liệu

Notebook mặc định đọc thư viện full ở `outputs/reproduction_full/01_synthetic`.
Quy mô tham chiếu là 18.387 attempts/family × 3 = 55.161 attempts; lượt hiện
có giữ 54.081 curves hợp lệ. Metadata phân biệt attempts với accepted curves.
Notebook đọc CSV theo categorical columns để giảm RAM; timeout một giờ mỗi cell.
Chạy notebook để lấy bảng thống kê và hình của chính bộ dữ liệu đang chọn.
