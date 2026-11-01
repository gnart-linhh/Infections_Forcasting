# Experiment 0 — tái hiện Figure 2 bằng Python

**Audit 2026-11-01:** code mới dùng `nonconstant_v2`, loại history hoàn toàn phẳng. Các kết quả/outputs/notebook đã lưu bên dưới vẫn thuộc NumPy filter v1; chưa chạy lại full hoặc BO sau sửa. [Đối chiếu R, số lượng windows và phạm vi ảnh hưởng](../../REPRODUCTION_AUDIT.md). Dùng output/cache directory mới; không resume run v1 bằng code mới.

Run chính trên toàn bộ lịch tuần: xem [FULL.md](FULL.md) và
[full.ipynb](full.ipynb). Run này dùng 50 bang × 112 origins, snapshot JHU đầy
đủ cho test chung, paper N=18.387/k=5/m=4.422; BO hiện tại tiếp tục riêng.

Mở [experiment_0.ipynb](experiment_0.ipynb), chọn kernel `.venv`, Run All.
Notebook hiển thị hình paper, hình Python cùng bố cục và MAE/WIS thực tế.

Đã đọc [README gốc](https://github.com/lanl/precog/tree/main/smoa): ngoài
`state_truths`, experiment cần historical truth và forecast/score COVIDHub.
Run cũ dùng final truth chưa đáp ứng phần historical data. Run mới đã tải
snapshot JHU và forecast COVIDHub chính thức, không cần clone repo hay chạy R.

```bash
.venv/bin/python experiments/data/fetch_paper_data.py \
  --as-of-dates 2020-08-15 2020-08-22 2020-08-29 2020-09-05
.venv/bin/python -u experiments/reproduction/experiment_0.py
```

Dữ liệu cache ở `outputs/paper_data/`; kết quả mới ở
`outputs/experiment_0_source/`. Thư viện synthetic đầy đủ được kiểm checksum
và dùng lại từ `outputs/reproduction_full/01_synthetic/curves.csv`.
**Không chạy lại BO để ép tham số gần paper**: nhánh Figure 2 dùng đúng tham
số công bố `num_curves=18387, k=5, closest=4422`.

## Những phần đã sửa so với run cũ

| Phần | Run mới |
| --- | --- |
| History | JHU snapshot lịch sử từ commit SHA do `reichlab/covidData` chỉ định; gộp county thành bang, tổng tuần Chủ nhật–Thứ bảy |
| Nhãn as_of | `get_data_as_of.R` tải Saturday+1 rồi ghi nhãn Saturday; Python làm giống nguồn, không gọi đó là dữ liệu đã có vào Saturday |
| Smoothing | Thin plate regression spline, k=round(n/2), affine null space, curvature penalty, GCV; thay natural smoothing spline khác rank |
| Synthetic | Đủ 55.161 attempts, 54.081 accepted; giữ NumPy RNG khác R |
| Embedding | `diff(X)` / `diff(y)` bỏ bridge đúng riêng nhánh Figure 2 |
| UQ | Initial sizes [20,10,6,6]; MLE và NB 5.000 draws, median recentering |
| Ngày | 4 bang × 6 first-target dates trong source × 4 horizons = 96 forecasts/model |
| Mô hình so sánh | Forecast gốc COVIDhub-baseline và COVIDhub-4_week_ensemble; không thay bằng persistence tự viết |
| Hình | 3 hàng model × 4 cột bang; cùng y-scale, actual cam, point đen, bands xanh 50%/95% |

First-target dates: 2020-03-21, 04-18, 05-23, 06-20, 07-18, 08-15.
Origin = first target − 7 ngày; Monday submission = first target − 5 ngày.
Mỗi tuần giữa các mốc vẫn được chạy để cập nhật UQ.

**Quy tắc hồi cứu của Figure 2:** source fit NB từ mọi forecast đã phát trước
đó cùng final outcomes, kể cả target chưa tới. Runner chỉ bật quy tắc này bằng
policy có tên `figure2_retrospective`; kết quả không được gọi là real-time causal.
Xuất thêm `smoa_matured_uq` giữ nguyên điểm dự báo, chỉ dùng target đã tới hạn.
Forecasting thông thường vẫn mặc định matured outcomes.

## Cách đọc kết quả

- `report.md`, `metrics.csv`: MAE, WIS, độ phủ và số forecasts từng mô hình.
- `paired_metrics.csv`: **chỉ giao các targets có đủ 3 mô hình**; ensemble chỉ
  có 16 targets ở các ngày chọn, nên không so nó với mean trên cả 96 targets.
- `figure_2_python.png/.svg`, `plot_data.csv`: hình và đúng số liệu dựng hình.
- `ablation.csv`: cùng 96 targets, công bố cả final/historical × spline/GAM;
  không chọn biến thể theo lỗi COVID rồi gọi là tái hiện nguồn.
- `protocol.json`, data `provenance.json`: URL, commit, input/code hashes và seeds.

Hình mới đã gần cấu trúc và xu hướng đường cong của Figure 2. Chưa có vector
forecast sMOA gốc để kiểm tra numerical parity; Python RNG/library khác R,
và GCV optimizer chưa được cross-check với fitted-value fixtures của mgcv.
Cơ sở toán của smoother: [mgcv thin plate regression splines](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth.construct.tp.smooth.spec.html).

Run 50 bang/104 BO trials cũ ở `outputs/reproduction_full/` vẫn là kết quả
**legacy final truth + spline**; không đổi nhãn nó thành kết quả historical/GAM.
BO và forecasting mặc định mới dùng GAM; phải dùng thư mục mới khi chạy lại
full, vì source/input checksum đã đổi. Cache Figure 2 có 4 bang × 4 vintage;
run [full mới](FULL.md) đã bổ sung snapshot 50 bang × 112 origins hàng tuần
của test chung. Đây chưa phải toàn bộ file nguồn 5,5 GB cho mọi ngày/bang.
Không khẳng định đã tái hiện Figure 3–5 hay tỷ lệ thắng 78%/76% các models.

```bash
.venv/bin/python -m unittest discover -s tests/reproduction -v
.venv/bin/python -m unittest discover -s tests/forecasting -v
.venv/bin/python -m unittest discover -s experiments/optimization -v
```
