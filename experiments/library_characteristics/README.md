# Experiment 3 — Impact of Synthetic Library Characteristics

Giữ **sMOA, k=5, m=4.422, GAM, omit bridge**, khảo sát rollercoaster, wiggle, seasonal và mixed trên **3 banks độc lập** đã sinh trong E2. Không sửa generator, không sinh lại synthetic data, không chạy BO hay UQ.

Mỗi cấu hình dùng chung **50 bang ×50 origins ×4 horizons =10.000 targets** của E2/E4. Tổng 36 runs, 360.000 forecasts. Đây là panel phân tích gồm cả giai đoạn ít history; không phải toàn bộ lịch 112 origins của main `full`.

## Ba nhánh kiểm soát

| Nhánh | Single family | Mixed | Mục đích |
| --- | --- | --- | --- |
| `full` | Toàn bộ accepted từ 18.387 attempts của họ đó | Toàn bộ ba họ, khoảng 54.000 curves | Đánh giá thư viện nguyên bản |
| `equal_curves` | 15.000 accepted curves | 5.000/họ, tổng 15.000 | Loại chênh lệch số curves |
| `equal_windows` | 3.000.000 valid windows | 1.000.000/họ, tổng 3.000.000 | Loại chênh lệch số analogue segments |

Giữ m=4.422 trong cả ba nhánh. Cùng số curves vẫn có M khác nhau do độ dài và valid-window filtering, nên cần nhánh thứ ba. Ngược lại, cùng M không đồng nghĩa cùng số curves; báo cả C và M. So sánh single/mixed trong cùng nhánh, không coi chênh lệch giữa các nhánh là tác động thuần của thành phần.

Sampling uniform không replacement, dùng RNG riêng cho seed/họ/đơn vị chọn mẫu, không dùng COVID outcomes. Trong mỗi nhánh kiểm soát, phần mixed của từng họ là subset của single tương ứng. Giữ thứ tự nguồn để tie-breaking exact L1 không thay đổi. Không ép thay thế các seasonal attempts bị generator loại. Dùng original N=18.387 banks, không dùng extensions 22k/30k.

## Chạy và xem kết quả

```bash
.venv/bin/python -u experiments/library_characteristics/experiment_3.py
# Tiếp tục các groups chưa xong, không tính lại completed groups:
.venv/bin/python -u experiments/library_characteristics/experiment_3.py --resume
.venv/bin/python -m unittest discover -s experiments/library_characteristics -v
.venv/bin/python experiments/library_characteristics/experiment_3.py --execute-notebook
```

Mở [experiment_3.ipynb](experiment_3.ipynb), chọn `.venv`; notebook chỉ đọc kết quả. [Report](../../outputs/experiment_3/report.md) chứa bảng và nhận xét. Theo dõi `outputs/experiment_3/progress.json` trong khi chạy.

Chỉ bốn file nguồn: runner/phân tích, tests, notebook, README. Một SQLite `results.sqlite` lưu toàn bộ forecasts, matching distances, origin context, morphology từng curve và metadata mỗi run; không xuất thêm CSV forecast trùng dữ liệu. Chỉ xuất các bảng tổng hợp nhỏ và sáu bộ hình PNG/SVG. Scratch embeddings dùng lại một thư mục và xóa sau từng worker; không tạo thêm source snapshots hay clone.

## Metrics và diễn giải

- MAE/RMSE tính riêng từng seed, báo mean và sample SD qua ba seeds. Mean RMSE khác pooled RMSE. Variability trong nhánh subsampling gồm cả bank RNG và subset RNG; chỉ ba seeds nên chưa kết luận phổ quát.
- Primary normalized NN = min L1 / `(5 × max(mean(last6 origin-input floored counts),1))`; có cả normalized median top-m. Đây là chuẩn hóa khi báo cáo, không đổi neighbor selection. Error normalization dùng mean last4 như E4.
- Phân tích theo horizon, phase, case level và history stage từ đúng vintage input tại origin. Paired delta MAE(single − mixed) và CI bootstrap nguyên bang 500 lần, giữ ba seeds cùng cluster. CI điều kiện trên ba banks, không bao quát mọi generator seeds hoặc mọi phụ thuộc giữa bang.
- Morphology dùng lại EDA: length, scale, cadence, số peaks, relative volatility/CV. Peak rule median3, prominence 10% amplitude, separation 4 original steps; weekly và monthly phải đọc riêng.
- Giữ raw scoring truth và nguyên valid-window rule. Một số constant histories được giữ do floating-point rounding của variance; báo constant fraction, không sửa core riêng cho E3. Không coi NN thấp là bảo đảm forecast tốt.

Input hashes, design và source được ghi trước scoring trong `protocol.json`; database ghi hoàn tất từng run trong transaction. Ba full-mixed runs tái dùng E2, kiểm tra đủ shared targets/outcomes và checksums. Các cấu hình còn lại chạy exact matching bằng core hiện tại, không thay algorithm.

## Kết quả đã chạy

Hoàn tất **36/36 runs**, **360.000 forecasts**, **90.000 matching cases**; khảo sát morphology toàn bộ **162.129 curves** của ba banks. Lượt chạy đầu mất khoảng **9 phút 15 giây**, dùng lại banks và E2 contexts; không bao gồm thời gian sinh bank trước đó.

MAE trung bình qua ba seeds:

| Chế độ | Rollercoaster | Wiggle | Seasonal | Mixed |
| --- | ---: | ---: | ---: | ---: |
| Full | 5.422,55 | 5.423,70 | 5.312,89 | **5.309,16** |
| Equal curves | 5.435,43 | 5.440,28 | **5.320,62** | 5.390,65 |
| Equal windows | 5.463,74 | 5.470,10 | **5.353,25** | 5.419,46 |

- Full seasonal−mixed chỉ **+3,73 MAE**, paired CI **[−41,65; +54,17]** chứa 0. Không đủ phân biệt rõ hai cấu hình trên ba banks này.
- Seasonal−mixed ở equal-curves **−70,02**, CI **[−116,25; −15,73]**; equal-windows **−66,21**, CI **[−107,36; −19,57]**. Lợi thế full-mixed không thể quy riêng cho việc trộn ba họ.
- Trong equal-windows, mixed có normalized NN thấp hơn ở cả ba phase nhưng seasonal có MAE thấp hơn: matching gần hơn không bảo đảm forecast tốt hơn.
- Mixed có SD MAE nhỏ nhất ở full (**2,37**) và equal-windows (**1,24**), nhưng không ở equal-curves. Relative SD MAE các cấu hình khoảng **0,023%–0,380%**; chỉ mô tả ba seeds, không xác nhận ranking trên mọi seeds.
- Weekly curves có mean số peaks rollercoaster/wiggle/seasonal khoảng **3,74 / 3,62 / 9,77** theo rule EDA. Seasonal có đuôi volatility rất dài: weekly median khoảng **0,390**, mean **1,796**, p99 **34,43**; không nên diễn giải mean cao thành mọi seasonal curve đều biến động hơn.

RMSE, normalized distances, SD, paired CI, phân tầng và các giới hạn nằm trong report/notebook. Chín tests kiểm tra sampling, nesting, tie-breaking, embedding/forecast parity, seed-score aggregation và database transactions.
