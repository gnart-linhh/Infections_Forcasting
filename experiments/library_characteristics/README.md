# Experiment 3 — Impact of Synthetic Library Characteristics

Lượt hiện hành: **protocol v2 / nonconstant_v2**, output
`outputs/experiment_3_nonconstant_v2/`, dựa trên E2 mới ở
`outputs/experiment_2_nonconstant_v2/`. Cả 36 runs được tính mới; full-mixed
phải khớp chính xác E2 trên 10.000 targets/seed và matching distances, sau khi
kiểm tra source, contexts và bank hashes. Không copy forecasts để gọi là parity.
[Notebook](experiment_3.ipynb) và [report mới](../../outputs/experiment_3_nonconstant_v2/report.md)
chỉ hiển thị kết quả đã kiểm chứng; không mang ranking v1 sang lượt mới.

Giữ **sMOA, k=5, m=4.422, GAM, omit bridge**, khảo sát rollercoaster, wiggle, seasonal và mixed trên **3 banks độc lập** đã sinh trong E2. Không sửa generator, không sinh lại synthetic data, không chạy BO hay UQ.

Mỗi cấu hình dùng chung **50 bang ×50 origins ×4 horizons =10.000 targets** của E2/E4. Tổng 36 runs, 360.000 forecasts. Đây là panel phân tích gồm cả giai đoạn ít history; không phải toàn bộ lịch 112 origins của main `full`.

## Ba nhánh kiểm soát

| Nhánh | Single family | Mixed | Mục đích |
| --- | --- | --- | --- |
| `full` | Toàn bộ accepted từ 18.387 attempts của họ đó | Toàn bộ ba họ, khoảng 54.000 curves | Đánh giá thư viện nguyên bản |
| `equal_curves` | 15.000 curves có valid windows | 5.000/họ, tổng 15.000 | Loại chênh lệch số curves |
| `equal_windows` | 3.000.000 valid windows | 1.000.000/họ, tổng 3.000.000 | Loại chênh lệch số analogue segments |

Giữ m=4.422 trong cả ba nhánh. Cùng số curves vẫn có M khác nhau do độ dài và valid-window filtering, nên cần nhánh thứ ba. Ngược lại, cùng M không đồng nghĩa cùng số curves; báo cả C và M. So sánh single/mixed trong cùng nhánh, không coi chênh lệch giữa các nhánh là tác động thuần của thành phần.

Sampling uniform không replacement, dùng RNG riêng cho seed/họ/đơn vị chọn mẫu, không dùng COVID outcomes. Trong mỗi nhánh kiểm soát, phần mixed của từng họ là subset của single tương ứng. Giữ thứ tự nguồn để tie-breaking exact L1 không thay đổi. Không ép thay thế các seasonal attempts bị generator loại. Dùng original N=18.387 banks, không dùng extensions 22k/30k.

## Chạy và xem kết quả

```bash
# Fresh output; --wait-for-e2 chờ E2 mới hoàn tất cả report và notebook.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/library_characteristics/experiment_3.py \
  --e2-dir outputs/experiment_2_nonconstant_v2 --wait-for-e2 \
  --output-dir outputs/experiment_3_nonconstant_v2 \
  --notebook-path experiments/library_characteristics/experiment_3.ipynb
.venv/bin/python -m unittest discover -s experiments/library_characteristics -v
```

Dùng launch.json/resume_command và source snapshot để tiếp tục job đã khởi
chạy. E3 dừng nếu E2 thất bại hoặc method/filter không khớp. Notebook được
execute sau khi đủ 36 runs và mọi artifact/source checks hoàn tất.

### Điều kiện curves tham gia equal-curves

Sau filter mới, seasonal banks có **29/42/35** accepted curves không còn
window hợp lệ. Equal-curves chọn uniform trong pool có ít nhất một valid
window để ngân sách thực sự là 15.000 (mixed 5.000/họ); vẫn đủ hơn 17.000
seasonal curves hợp lệ mỗi bank. Điều kiện chỉ dùng library, không dùng COVID
errors. Full vẫn đưa mọi accepted curve vào input, báo input_accepted_curves,
eligible_window_curves và zero_window_curves riêng. Mỗi họ/seed có trên bốn
triệu valid windows, nên equal-windows=3 triệu vẫn khả thi.

## Metrics và diễn giải

- MAE/RMSE tính riêng từng seed, báo mean và sample SD qua ba seeds. Mean RMSE khác pooled RMSE. Variability trong nhánh subsampling gồm cả bank RNG và subset RNG; chỉ ba seeds nên chưa kết luận phổ quát.
- Primary normalized NN = min L1 / `(5 × max(mean(last6 origin-input floored counts),1))`; có cả normalized median top-m. Đây là chuẩn hóa khi báo cáo, không đổi neighbor selection. Error normalization dùng mean last4 như E4.
- Phân tích theo horizon, phase, case level và history stage từ đúng vintage input tại origin. Paired delta MAE(single − mixed) và CI bootstrap nguyên bang 500 lần, giữ ba seeds cùng cluster. CI điều kiện trên ba banks, không bao quát mọi generator seeds hoặc mọi phụ thuộc giữa bang.
- Morphology dùng lại EDA: length, scale, cadence, số peaks, relative volatility/CV. Peak rule median3, prominence 10% amplitude, separation 4 original steps; weekly và monthly phải đọc riêng.
- Giữ raw scoring truth và nguyên valid-window rule. Bắt buộc WINDOW_FILTER hiện hành và retained constant histories=0 cho mọi run; uniform legacy filter cũng bị từ chối. Không coi NN thấp là bảo đảm forecast tốt.

Input hashes, design và source được ghi trước scoring trong `protocol.json`; database ghi hoàn tất từng run trong transaction. Ba full-mixed runs tái dùng E2, kiểm tra đủ shared targets/outcomes và checksums. Các cấu hình còn lại chạy exact matching bằng core hiện tại, không thay algorithm.

## Kết quả hiện hành

Chỉ dùng `outputs/experiment_3_nonconstant_v2/` sau progress status=complete,
36 runs/360.000 forecasts, zero constant rows và 30.000 independently matched
E2 reference targets. Report v1 giữ riêng ở `outputs/experiment_3/report.md`;
chưa coi những kết luận mixed/seasonal cũ là kết quả của v2.
