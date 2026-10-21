# Bước 2 — sMOA point forecast và backtest

Dùng trực tiếp `curves.csv` đã sinh ở bước 1. Pipeline chỉ chạy Python;
không import hoặc gọi code trong checkout `precog/`.

## Chuẩn bị truth

CSV của **một** chuỗi số ca hàng tuần, gồm hai cột:

```csv
target_end_date,value
2020-01-25,0
2020-02-01,3
2020-02-08,3
```

Ngày phải liên tục cách nhau 7 ngày, không trùng; giá trị hữu hạn và không âm.
Theo code bài báo, giá trị được chặn dưới ở 1 trước khi dự báo và tính MAE.

Đã có bản CSV local tại `outputs/truth/California.csv`, chuyển bằng Python từ
truth California đi kèm nguồn tham khảo. Nếu cần chuyển lại một file RData có sẵn,
đây là thao tác tùy chọn, không cần cài R:

```bash
.venv/bin/python -m pip install pyreadr
.venv/bin/python experiments/forecasting/export_truth.py \
  --input /path/to/California.RData --output outputs/truth/California.csv
```

Chỉ cần NumPy và SciPy nếu đã có CSV; dùng cùng environment bước sinh dữ liệu.

## Chạy

```bash
.venv/bin/python experiments/forecasting/run_backtest.py \
  --synthetic outputs/synthetic_demo/curves.csv \
  --truth outputs/truth/California.csv \
  --series California --k 5 --h 4 --closest 50 \
  --max-origins 0 --output-dir outputs/backtest_california
```

Mặc định dùng 8 origins từ 2020-08-15; `--max-origins 0` dùng tất cả mốc đủ
history và đủ 4 tuần tương lai. Output phải là thư mục trống hoặc chưa tồn tại.
`--smoothing none` dùng cho đối chiếu không làm trơn; mặc định `spline` dùng
SciPy cubic smoothing spline với GCV, chỉ fit history tại mỗi origin.

## Thuật toán và output

1. Với từng synthetic curve, tạo cửa sổ gồm `k+1` giá trị history và `h` giá trị
   future. Không nối cửa sổ qua ranh giới hai curve. Chặn synthetic ở `1e-8` và
   loại cửa sổ có history phương sai bằng 0, tương tự `create_embed_matrix` gốc.
2. Lấy sai phân: X có k cột, y có h cột. Giữ ID curve và bước bắt đầu để truy vết.
3. Tại mỗi origin, chỉ làm trơn history đến origin, lấy k sai phân cuối và tính
   khoảng cách L1 với X. Chọn `closest` cửa sổ gần nhất; nếu thư viện nhỏ hơn,
   dùng toàn bộ và ghi số thực dùng trong summary.
4. Lấy trung vị y theo từng horizon, cộng dồn vào **giá trị quan sát cuối cùng**,
   chặn dưới ở 1. Tương lai truth chỉ dùng để chấm điểm.

Output:

- `forecasts.csv`: dự báo, truth, sai số tuyệt đối và baseline persistence.
- `matches.csv`: các curve/window được chọn và khoảng cách theo từng origin.
- `summary.json`: MAE tổng và theo horizon của sMOA và baseline.
- `metadata.json`: tham số, phiên bản, checksum input/code và giới hạn thí nghiệm.

Kiểm tra thuật toán:

```bash
.venv/bin/python -m unittest discover -s experiments/forecasting -v
```

## Kết quả kiểm chứng local

Với `outputs/synthetic_demo` (30 curve, 10 mỗi nhóm), k=5, h=4, closest=50,
thu được 8.371 cửa sổ. Backtest California từ 2020-08-15, 112 origins,
448 dự báo, default spline:

| Horizon | sMOA MAE | Persistence MAE |
| --- | ---: | ---: |
| 1 tuần | 23.683,99 | 26.665,48 |
| 2 tuần | 43.236,04 | 47.166,79 |
| 3 tuần | 60.634,09 | 64.843,19 |
| 4 tuần | 76.514,67 | 81.684,35 |
| Tổng | 51.017,20 | 55.089,95 |

Kết quả đầy đủ local: `outputs/backtest_california_all/`. Đây là cấu hình nhỏ để
kiểm chứng pipeline; không phải kết quả tái hiện đầy đủ bài báo. 8 origins đầu
cho MAE sMOA 9.623,97, baseline 9.102,28; hiệu quả phụ thuộc giai đoạn đánh giá.

## Mức độ tái hiện và bước tiếp theo

Đối chiếu thuật toán với `smoa/R/run_smoa.R` và `smoa/R/smoa_helpers.R` trong
[nguồn LANL](https://github.com/lanl/precog/tree/main/smoa). Công thức embedding,
L1 matching, median differences và cộng dồn giữ theo code gốc. Bản này:

- Dùng thư viện synthetic Python xấp xỉ của bước 1.
- Dùng spline SciPy thay cho `mgcv::gam`; không khẳng định hai smoother tương đương.
- Backtest trên final truth, chưa dùng dữ liệu báo cáo lịch sử theo từng `as_of`.
  Không nhìn tương lai trong vòng backtest, nhưng final truth vẫn có thể chứa
  các chỉnh sửa sau ngày origin; chưa phải mô phỏng dự báo thời gian thực.
- Giữ tất cả cadence/scale trong synthetic library theo luồng embedding code gốc,
  không tự chuyển monthly thành weekly.
- Chưa có khoảng dự báo, online MLE dispersion, WIS hoặc so sánh COVIDHub ensemble.

Sau bước kiểm chứng point forecast, phần kế tiếp là học dispersion từ những
forecast đã có outcome tại origin, sinh quantile forecast và đánh giá WIS;
sau đó mới so sánh đầy đủ với các baseline của bài báo trên dữ liệu as-of.
