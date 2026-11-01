# Phạm vi tái hiện sMOA bằng Python

Nguồn: [LANL/precog](https://github.com/lanl/precog/tree/main/smoa), commit
`aba1948d228e5380d72907048236601874b53cac`. Figure 2 được đối chiếu thêm với
commit trước publication `2018f2e` (2025-05-22). Python không gọi R hay phụ
thuộc checkout `precog/` khi chạy.

## Đối chiếu source và sửa bộ lọc embedding — 2026-11-01

Đã đọc trực tiếp Git blobs của checkout nguồn tại commit trên; các file R
được đối chiếu không có sửa đổi trong working tree. Những nghi vấn trong bản
review không mặc nhiên là lỗi Python. Nguồn quyết định:
[smoa_helpers.R](https://github.com/lanl/precog/blob/aba1948d228e5380d72907048236601874b53cac/smoa/R/smoa_helpers.R#L91),
[figure_2.R](https://github.com/lanl/precog/blob/aba1948d228e5380d72907048236601874b53cac/smoa/R/figure_2.R#L76),
[run_smoa.R](https://github.com/lanl/precog/blob/aba1948d228e5380d72907048236601874b53cac/smoa/R/run_smoa.R#L98).

| Thành phần | Kết luận từ R và cách xử lý |
| --- | --- |
| Bridge/indexing | `run_smoa.R:98–106` giữ bridge qua `diff(c(last(X),y))`; `figure_2.R:76–83` bỏ bridge qua `diff(y)` với thêm một future level. Python giữ hai chế độ. Với levels `[1,2,4,7,11,16]`, k=2,h=2: query `[1,2]`, keep `[3,4]`, omit `[4,5]`. Omit thực sự bỏ increment kế tiếp; đây là quy ước figure script, không tự sửa theo MAE. |
| Số quan sát | `figure_2.R:155` bỏ qua n≤k; k differences cần k+1 levels. Giữ yêu cầu Python. |
| Point aggregation | `figure_2.R:175–190`: L1, median từng future increment, rồi cumsum, cộng số ca raw cuối và floor 1. Giữ; test dùng ví dụ median-of-paths cho kết quả khác. |
| NB forecast<100 | Có trong `run_smoa.R:260–266` và `figure_2.R:238–248`: chỉ thay mean của likelihood bằng actual, không thay point forecast phát ra. Giữ cả strict threshold `<100`, round outcomes và tối ưu inverse size. Python chặn inverse size ở 1e-8 thay cho biên Poisson tại 0; không khẳng định optimizer bitwise giống R. |
| UQ hồi cứu | Figure 2 dùng final outcomes của các forecasts đã phát trước đó, kể cả chưa tới hạn; `run_smoa.R:253` lọc `target_end_date <= curr_targetend_date`. Giữ policy explicit. Maturity không loại được look-ahead từ frozen warmup hay final revisions. |
| Warmup/MLE | Source kiểm tra tổng số rows `<6*4`, không yêu cầu mỗi horizon có sáu mature labels. Không đổi ngưỡng này. Python thêm fallback initial khi horizon không có eligible labels; đó là xử lý biên, không phải thay điều kiện R. Init Figure 2 `[20,10,6,6]`; main `[20,10,6,0.3]`. |
| Coverage | `run_smoa.R:341–342` dùng `actual>lower & actual<=upper`. Giữ lower-open để đối chiếu source; E6 dùng inclusive coverage theo định nghĩa phân tích và ghi riêng, không đồng nhất hai con số. |
| WIS | Giữ 2×mean(pinball) trên bảy quantiles đối xứng. Tests đối chiếu với weighted interval-score formula. |
| GAM | Python đã có basis/penalty/GCV tests nhưng **chưa có fitted-values fixture độc lập từ mgcv**. Chưa thể kết luận numerical parity với R, kể cả khi rank và mục tiêu toán giống nhau. Audit này không chạy R và không chọn smoothing theo COVID test. |

**Sửa lỗi thực tế:** source helper lọc `RowVar(history)>0`. `np.var` float64
có thể dương cho history gồm sáu giá trị 1e-8 giống hệt nhau. Bộ lọc mới
`nonconstant_v2` so sánh trực tiếp các levels sau floor 1e-8, loại chính xác
history hoàn toàn phẳng và giữ cả biến động một ULP; không dùng tolerance.
[R rowMeans implementation](https://raw.githubusercontent.com/wch/r-source/trunk/src/main/array.c)
dùng accumulator `LDOUBLE`, cho thấy không thể mặc nhiên đồng nhất hành vi
rounding với NumPy. Đây là sửa theo ý nghĩa zero-variance của source;
chưa phải chứng minh bitwise parity với R build/phần cứng của tác giả.

Kiểm đếm **toàn bộ bank seed0** đã lưu của Python: 55.161 attempts,
54.081 accepted, 15.510.156 observations. CSV SHA256
`374329bac3b4bbc28cfc04b6e5f7ff2edcefe9a56979c45b6344dc5ff1c9d9ff`.
Packed seed0 giữ nguyên curves và thứ tự của CSV này; không sinh lại data.

| Embedding k=5,h=4 | Trước lọc | NumPy v1 giữ | Constant bị v1 giữ nhầm | nonconstant_v2 giữ |
| --- | ---: | ---: | ---: | ---: |
| Keep bridge, width=10 | 15.023.427 | 15.023.012 | 2.078.715 | 12.944.297 |
| Omit bridge, width=11 | 14.969.346 | 14.968.931 | 2.047.853 | 12.921.078 |

Không tìm thấy nonconstant window nào bị v1 loại mà v2 giữ trên bank này.
Mỗi họ trong cả ba banks vẫn có trên 4 triệu valid omit windows sau sửa,
nên E3 equal-windows=3 triệu vẫn khả thi. Số 15.642.758 của paper không thể
dùng làm checksum cho bank Python: RNG, accepted curves, độ dài và convention
keep/omit đều ảnh hưởng M. Không chỉnh generator hoặc filter để ép khớp M.

Kiểm tra tác động có kiểm soát: lấy 100 case indices cách đều từ 0 đến 2.499
trong E2 seed0, đủ bốn horizons/case (400 targets), giữ nguyên GAM queries,
raw anchors và labels. Rebuild exact-neighbor index sau khi loại các flat rows:
**0/400 point forecasts thay đổi**, MAE cả hai = **3.209,16**.
Điều này chỉ xác nhận trên mẫu đã kiểm tra; không thay thế một run full và
không chứng minh BO objective/winner hoặc UQ của mọi instance không đổi.

Tất cả builders (in-memory, disk, BO, E2/E3 và EDA) dùng chung predicate.
Cache metadata v2 ghi `window_filter`; cache v1 bị từ chối, không ghi đè.
Full/E4/E1/E5 chọn đường dẫn cache mới; E3 chỉ reuse full-mixed E2 nếu filter
trùng, nếu không tính mới. E5 yêu cầu E2 có cùng source. BO config và selected
parameters được kiểm tra filter trước khi prediction; không resume hoặc dùng
winner cũ như thể đã optimize objective mới.

**Các kết quả ở phần dưới, các README kết quả và notebook đã execute vẫn
thuộc bộ lọc v1.** Tại thời điểm audit ban đầu chưa rerun. Sau yêu cầu tiếp theo, full 22.400
targets và BO 104 evaluations đã khởi chạy ở các outputs nonconstant_v2,
chưa hoàn tất; MAE/WIS cũ và winner N=29.357,k=7,m=5.167 không được đổi nhãn thành
kết quả v2. Dùng output directory mới cho các lượt chạy sau audit, giữ bank,
vintages và artifacts cũ để đối chiếu. E6 có thể tiếp tục phân tích intervals
cũ nhưng phải mang provenance v1; muốn đánh giá pipeline v2 cần prediction/UQ mới.

Sau đối chiếu UQ lần tiếp theo, full mặc định đã đổi sang **matured** để
khớp main `run_smoa.R:253`, còn `experiment_0` giữ retrospective của Figure 2.
Paper chạy lại tại `outputs/paper_full_nonconstant_v2_matured/`; winner tự chờ
ở `outputs/bo_gam_full_nonconstant_v2_matured/`. BO point-MAE không bị restart.
Bản full v2 hồi cứu đã complete được giữ tại `outputs/paper_full_nonconstant_v2/`.
Vẫn giữ omit bridge theo yêu cầu trước, khác main R; frozen warmup và final
revisions vẫn hồi cứu. Chưa gọi toàn bộ pipeline causal hoặc replay nguyên trạng R.

Kiểm chứng sau sửa: **149 tests pass** trên forecasting, reproduction,
synthetic, BO, E1–E6, exploration và pipeline. Hai reproduction notebooks
validate đúng schema; toàn bộ code cells/outputs đã lưu được giữ nguyên,
chỉ thêm ghi chú phiên bản. SHA256 của CSV seed0 và hai packed-bank files
khớp metadata. Không thêm file nguồn hoặc dataset mới trong audit này.

## Kiểm chứng sau khi chuyển full UQ sang main R

Lượt `outputs/paper_full_nonconstant_v2_matured/` đã hoàn tất dự báo đủ
22.400 targets. Đối chiếu timestamps của mọi calibration count và fit lại
600 NB sizes xác nhận chỉ matured pairs được dùng. Point forecasts của cả
50 bang giống byte-for-byte với full v2 hồi cứu. MAE =7.384,66 giữ nguyên;
WIS =4.747,38 và coverage 95% lower-open =87,17% (hồi cứu: 4.727,47 /87,33%).
Default CLI full đã là matured, Figure 2/Experiment 0 vẫn opt-in retrospective.
53 forecasting tests pass. BO point-MAE tiếp tục nguyên trạng; lịch full
winner mới chỉ định matured và cùng paper baseline mới. Không coi maturity
filter đã giải quyết frozen warmup hoặc vintage revisions còn lại.

## Kết quả đã lưu trước sửa bộ lọc (v1)

[Experiment 0](experiments/reproduction/README.md) dùng
`outputs/experiment_0_source/`: 4 bang, 6 origins cố định, horizons 1–4 tuần.
Dữ liệu nằm ở `outputs/paper_data/`, gồm snapshot JHU lịch sử của 4 vintage
và forecast chính thức baseline/4-week ensemble ở các ngày có incident cases.
URL, commit, SHA và quy tắc Saturday+1 retrieval nằm trong provenance.

- Library Python: 18.387 attempts/family × 3 = 55.161 attempts; 54.081 accepted.
- Tham số Figure 2 cố định theo publication: num_curves=18.387, k=5, closest=4.422.
- Smoother: thin plate regression spline, rank round(n/2), curvature penalty, GCV.
- Figure 2 giữ embedding bỏ bridge và NB initialization [20,10,6,6].
- UQ `figure2_retrospective` dùng final outcomes của mọi forecast đã phát trước
  đó, kể cả target chưa tới, như source figure script. Không gọi đây là causal.
- Đối chứng `smoa_matured_uq` giữ cùng points và chỉ fit target đã tới hạn;
  frozen historical snapshot vẫn khác một real-time evaluation đầu năm 2020.
- MAE/WIS trên 96 targets: 10.570,35 / 7.431,19. So với COVIDHub chỉ dùng giao
  targets có dữ liệu; giao cả ba models có 16 targets, không suy ranking toàn kỳ.
- Hình 3 hàng model × 4 cột bang, chung y-scale, bands 50%/95%; giữ đủ target
  cuối 2020-09-05. Hình paper tham chiếu giữ trong output hiện hành `reference/`.

Notebook lưu kết quả thật; report công bố cả final/historical × spline/GAM
ablation. Không chọn biến thể theo lỗi COVID để gọi là phương pháp nguồn.

## Run full legacy được giữ để đối chiếu

Run mới với bộ paper, historical history và GAM ở `outputs/paper_full/`;
[full protocol](experiments/reproduction/FULL.md). Test chung có **50 bang ×
112 origins hàng tuần × 4 horizons = 22.400 targets/model**. Toàn bộ 112 JHU
vintages đã dựng ở `outputs/paper_weekly_archive/`, đủ mỗi tuần từ 15/08/2020
đến 01/10/2022; Arizona-only later dates không thuộc cửa sổ chung. Trạng thái
và số liệu thực tế nằm trong pipeline.json/report, không dùng điểm legacy.

Run full giữ bridge/matured trước đó có MAE 7.943,55, WIS 5.090,15 và coverage
95% là 85,15%; được lưu riêng ở `outputs/paper_full_bridge_matured/`. **Full
hiện tại đổi theo yêu cầu sang bỏ bridge + UQ hồi cứu**, dùng cả final outcomes
của targets chưa đến hạn ở forecasts đã phát trước đó. Không gán điểm cũ cho
quy ước mới. [Kết quả hiện hành](experiments/reproduction/FULL.md#results).

Cả hai run mới đã hoàn tất trên 22.400 targets, **bỏ bridge + UQ hồi cứu**:
paper MAE 7.384,66 / WIS 4.727,47; BO winner MAE 7.443,16 / WIS 4.734,66.
Đây là đánh giá có future labels trong UQ, không phải kết quả real-time causal.

BO với GAM ở `outputs/bo_gam/` đã hoàn tất đủ 104 evaluations, chọn
N=29.357/k=7/m=5.167. Job
`outputs/bo_gam_full/` chỉ bắt đầu khi selection và run paper complete, rồi
chạy winner trên cùng archive, origins, labels, UQ/scoring code và seeds.

`outputs/reproduction_full/` đã hoàn tất 104 BO evaluations, 50 bang,
22.468 forecasts/model, dùng **final truth + spline cũ**. Winner:
num_curves=29.357, k=7, closest=5.167; MAE 7.564,27, WIS 4.765,92;
coverage 95% là 86,78%. Các điểm này không phải kết quả historical/GAM mới.

Run full chứa library đang dùng, checkpoint, dữ liệu dự báo và metadata kiểm
chứng nên được giữ. Source hashes cũ không bị sửa để khớp workspace hiện tại.
BO và forecasting mặc định mới dùng GAM; không resume run legacy bằng source
mới hoặc ghép BO spline với prediction GAM. Chạy lại full cần thư mục output
mới. Figures 3–5 và ranking/calibration full chưa được chạy lại với protocol mới.

## Quy ước thuật toán còn được giữ

- Generator source-structure-v2 dùng LSODA và NumPy RNG; ghi accepted/failed,
  không sinh bù. Seed train/validation/heldout tách vai trò.
- BO: num_curves 10k–30k, k2–7, closest1k–6k; 99 LHS + 5 EI; mọi validation
  origin đủ điều kiện. Mỗi trial resample train/validation; shuffle/cap 50.
  Objective là MAE; independent test chỉ chạy sau chọn best observed trial.
- Optimization/Figure 2 bỏ bridge; `run_smoa.R` nguồn giữ bridge. Full hiện
  chủ động dùng bỏ bridge theo yêu cầu, kết hợp với lịch test 50 bang × 112 tuần.
- Default forecasting UQ dùng matured targets, main NB sizes [20,10,6,0.3].
  Giữ forecast<100 likelihood rule; zero outcomes hợp lệ. Negative labels bị
  loại khỏi NB likelihood và ghi count, nhưng giữ raw values để chấm điểm.
- WIS dùng 7 quantiles [.025,.1,.25,.5,.75,.9,.975]. Full coverage dùng 23
  quantiles / 11 intervals, điều kiện lower < actual <= upper. Cặp .01/.99 là
  98%, dù prose nguồn có nói 99%.
- Exact L1 neighbors, stable ties; atomic checkpoints, source/input hashes và
  process locks. Prediction API không nhận future truth; UQ policy có nhãn riêng.

## Những phần chưa xác nhận tương đương

| Phần | Giới hạn |
| --- | --- |
| RNG / ODE | NumPy/LSODA khác R/deSolve; synthetic curves không trùng từng số |
| GAM | Cùng thin plate/GCV về mô hình toán; chưa có fitted-value fixtures mgcv để kiểm numerical parity |
| BO | Python Gaussian/RBF GP và EI khác hetGP; đã chạy xong 104 trials với GAM, winner được kiểm chứng bằng artifact hashes |
| Historical data | Có Figure 2 và đủ JHU vintages cho 50 bang × 112 tuần của full common test; chưa tải toàn bộ archive gốc 5,5 GB |
| External models | Có baseline/ensemble Figure 2; chưa có archive scores toàn bộ models |
| Original forecasts | Chưa có vector sMOA R gốc; không thể lượng hóa phần trăm giống hay xác nhận full-paper metrics |

`publication_reproduction_complete` vẫn false. Full ngân sách tính toán hoặc
hình tương tự chưa đủ để chứng minh numerical reproduction của toàn paper.
