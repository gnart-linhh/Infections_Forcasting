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
- Chế độ mặc định chỉ dự báo điểm; bật `--probabilistic` để học dispersion và tính WIS.
  Chưa so sánh COVIDHub ensemble.

Bước dispersion và WIS đã có trong chế độ bên dưới. Phần còn lại để tái hiện đầy
đủ là dữ liệu as-of, prehistory calibration, cấu hình thư viện lớn và so sánh
các model COVIDHub của bài báo.


## Bước 3 — khoảng dự báo, online dispersion và WIS

```bash
.venv/bin/python experiments/forecasting/run_backtest.py \
  --synthetic outputs/synthetic_demo/curves.csv \
  --truth outputs/truth/California.csv \
  --probabilistic --interval-draws 5000 --seed 13 \
  --max-origins 0 --output-dir outputs/backtest_california_wis
```

Dùng cùng dữ liệu và point forecasts của bước 2. `--probabilistic` hỗ trợ h từ
1 tới 4. Các tham số khởi tạo theo horizon là NB size `[20, 10, 6, 0.3]`.
Trong 6 origins đầu, mean lấy point+1; sau đó mean lấy point và size được fit
riêng từng horizon bằng maximum likelihood, tối ưu inverse-size trong (0,1].
Cận số học inverse-size nhỏ nhất là `1e-8`, gần giới hạn Poisson.

Tại origin đang xét, chỉ các forecast từ origins trước có `target_end_date <=
origin` được dùng để học. Giữ quy tắc trong nguồn gốc: khi forecast cũ <100,
mean trong hàm likelihood được thay bằng observed value. Khi có dữ liệu as-of,
observed value này cũng cần là giá trị đã biết tại ngày origin.

Mỗi horizon lấy 5.000 mẫu NB, tính quantile `[0.025, 0.1, 0.25, 0.5, 0.75, 0.9,
0.975]`, rồi dịch trung vị về point forecast. Giữ cách dịch của tác giả, nên cận
dưới có thể âm; không clip lại trước chấm điểm. NumPy RNG có seed riêng theo
ngày origin, horizon và model, giúp giữ kết quả của cùng origin khi đổi số mốc
chạy. Không khớp từng số với Monte Carlo của R.

WIS dùng công thức `2 * mean(pinball loss)` trên bảy quantile, cùng định nghĩa
trong helper gốc. Xuất sharpness, overprediction, underprediction và coverage
50/80/95%; coverage dùng `lower < actual <= upper` theo nguồn.

Output bổ sung:

- `quantiles.csv`: bảng dài của sMOA và persistence.
- `forecasts.csv`: thêm NB size, chế độ initial/mle, số outcome calibration,
  WIS, các thành phần và coverage của mỗi model.
- `summary.json`: trung bình các chỉ số, tổng và theo horizon.

Baseline persistence được học dispersion riêng, dùng cùng lịch calibration và
quy tắc NB; đây là baseline chẩn đoán bổ sung của project, không phải COVIDHub
ensemble. Pipeline chưa nạp forecast prehistory trước `--start-date`, nên lịch
chuyển sang MLE không khớp thí nghiệm đầy đủ của tác giả, vốn nạp early-pandemic
forecasts.

### Kết quả California, cấu hình nhỏ

112 origins, 448 dự báo; 30 synthetic curves, k=5, h=4, closest=50,
spline, 5.000 draws, seed=13. Kết quả local ở
`outputs/backtest_california_wis/`.

| Chỉ số | sMOA | Persistence + NB |
| --- | ---: | ---: |
| MAE | 51.017,20 | 55.089,95 |
| WIS | 31.854,45 | 35.442,88 |
| Coverage 50% | 52,46% | 47,10% |
| Coverage 80% | 75,22% | 74,11% |
| Coverage 95% | 86,61% | 86,16% |

WIS thấp hơn là tốt hơn, nhưng coverage 95% còn dưới mức danh nghĩa. Chưa có
cơ sở kết luận đã tái hiện chất lượng của bài báo từ cấu hình và final truth này.
Snapshot summary và cấu hình tại [results/california_probabilistic.json](results/california_probabilistic.json)
được lưu trong Git; CSV chi tiết vẫn ở outputs local.

10 tests kiểm tra embedding, matching, WIS bằng công thức interval score độc lập,
MLE trên dữ liệu NB có size đã biết, lịch khởi tạo, tái lập RNG và chống sử dụng
outcome tương lai. Kiểm tra end-to-end thay đổi truth sau origin mà quantile
đã phát hành vẫn giống hệt; toàn bộ point forecasts giữ nguyên so với bước 2.

## Bước 4 — mở rộng nhiều bang

Đã có runner benchmark, calibration warmup, giới hạn ngày origin và reader cho
historical snapshots. Xem [BENCHMARK.md](BENCHMARK.md) để chạy 50 bang và hiểu
mức độ tái hiện của dữ liệu final-truth/as-of. Các lệnh bước 2–3 vẫn dùng được.
