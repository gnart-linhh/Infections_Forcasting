# Experiment 2 — Sensitivity, Interactions and Computational Trade-offs

Thực nghiệm Python cho sMOA theo thiết kế E2.A/B/C. Reference:
**(N,k,m)=(18387,5,4422)**, GAM/GCV, exact L1 neighbors, **omit bridge**, h=4.
N là **số attempts mỗi family**, không phải số accepted curves hay valid segments M.
Không chạy BO, không chọn lại tham số theo COVID test, không thêm baseline model.

## E2.A — Thay đổi từng tham số

| Tham số | Các mức | Hai tham số còn lại |
| --- | --- | --- |
| N | 500, 2000, 5000, 10000, 18387, 22000, 30000 | k=5, m=4422 |
| k | 3, 5, 7, 9 | N=18387, m=4422 |
| m | 100, 500, 2000, 4422, 8000 | N=18387, k=5 |

Reference được tính một lần mỗi seed, dùng chung cho ba đường sensitivity.
Mỗi job kiểm tra m≤M, không âm thầm giảm m. Thư viện nhỏ lấy các attempt indices
j<N **trong từng family** của cùng largest bank; không lấy N accepted curves
để bù các seasonal attempts bị loại. Prefix của một family không đổi seed khi N đổi.

## E2.B — Factorial và tương tác

Đủ 8 corners của N={1000,5000}, k={3,7}, m={500,2000}, thêm center
(N,k,m)=(2000,5,1000) và paper reference. Các mức này do thí nghiệm đề xuất,
không phải hyperparameters hoặc kết quả từ paper. Tổng cộng **23 cấu hình
riêng biệt ×3 training seeds**, sau khi loại trùng giữa E2.A/B.

Heatmaps k=3/7 chỉ vẽ các corners đã thực sự chạy, cùng color scale; không
nội suy hoặc điền các ô chưa đánh giá. Center/reference có trong bảng riêng.
Heatmap bổ sung cho cả k=3/5/7/9 hiển thị tất cả cấu hình thực sự chạy trong
E2.A/B, gồm center/reference; các tổ hợp chưa chạy để xám, không điền số giả.
Coded effect = 2×mean(product of ±1 factor signs × MAE). Pair interaction
có thêm difference-in-differences = 2×coded effect. Triple interaction chỉ
exploratory; không suy ra causal effects hay statistical significance từ ba seeds.

## Fixed evaluation và early availability

Dùng **đúng cùng panel đã kiểm tra của Experiment 4**:
50 bang ×50 origins ×4 horizons = **10.000 targets/config/seed**.
Origins weekly 2020-04-04–2020-09-05, sau đó mỗi 4 tuần tới 2022-10-01.
Mọi main origin có ít nhất 10 quan sát nên đủ cho k=9; không để k nhỏ có nhiều
samples hơn. Tổng main study: **690.000 point forecasts**.

Input là JHU as-of thực trước phát hành, final truth chỉ làm labels; Saturday
label, Sunday cutoff, Monday issuance. Source URLs/commits ở
`outputs/epidemic_panel/origins.csv`; không dùng final history thay snapshots.
Raw input floor 1 như core; final labels giữ cả zero/negative revisions.
GAM history được fit một lần mỗi case, giữ như nhau giữa k/N/m.

Phần riêng **controlled short-history availability** dùng seed 0, N=18387,
m=4422, first 16 scheduled origins ×50 bang. Chỉ cho thấy last L={4,6,8,10}
quan sát tại chính vintage đó. Đây là kiểm soát lượng history, không giả rằng
đó là vintage sớm hơn hoặc dữ liệu đã có trước thực tế.

- Chỉ so các k hợp lệ **trong cùng L**, cùng cases.
- Core GAM hiện yêu cầu ≥5 observations: required history = max(k+1,5).
- Ghi rõ `insufficient_history` thay vì ép output/fallback; L=4 không có k nào
  chạy với GAM core hiện tại. Không sửa model core để vượt điều kiện này.
- Auxiliary results không được gộp vào main scores hoặc seed stability.

## Seeds, nesting và source parity

Seed labels 0/1/2 tương ứng offsets 0/1.000.000/2.000.000. Với largest N=18387,
scalar generation seed = offset +family_index×18387 +attempt_index+1.
Các namespaces không giao nhau. Trong mỗi seed, mọi N dùng nested prefixes.

Seed 0 đọc nguyên thư viện đang dùng trong reproduction. Metadata của thư viện
cũ ghi source hash khác source hiện tại; vì vậy đã **replay toàn bộ 55.161
attempts** bằng source hiện tại và kiểm tra CSV SHA256 giống hoàn toàn bản gốc.
Không chỉ kiểm vài curves hoặc dựa vào tên version. Replay lưu tại
`outputs/experiment_2/libraries/seed_0_replay`.

Accepted curves seed 0/1/2: 54.081 /54.006 /54.042; mọi bank đều 55.161 attempts.
Packed banks và prefix manifests giữ original order để tie breaking nhất quán.
Reference seed 0 còn được đối chiếu đủ 10.000 point forecasts với
`outputs/experiment_4_omit/instances.csv` ở bước validation cuối.

## Metrics và đo runtime

Accuracy: MAE, RMSE, bias và normalized MAE theo seed/horizon/config. Summary
RMSE là mean của ba seed-specific RMSE, ghi rõ trong report. Paired delta MAE
so với **same-seed reference**, 95% intervals dùng 1000 state-cluster bootstrap
draws, giữ origins/horizons/seeds cùng bang trong cluster. Không dùng paired CI
để tuyên bố cấu hình tốt nhất đã được xác nhận trên một test độc lập mới.

Stability: SD của MAE giữa ba seeds và SD/range của prediction trên từng target.
Matching: minimum/median/max L1 distance, L1/k, unique matched curves, coverage
fraction, IQR của selected future increments, mean-versus-median path gap.
Matched-curve fraction là tỷ lệ accepted curves đã được dùng bởi matches,
không phải prediction-interval coverage hoặc xác suất COVID nằm trong library.
L1 giữa các k có geometry khác nhau; L1/k không hoàn toàn sửa được điều này.
Với nested N và cùng k, kiểm tra minimum distance không tăng trên mọi query.
Forecast accuracy không bắt buộc cải thiện dù matching distance tốt hơn.

Runtime tách prefix loading, embedding và KD-tree setup. Một index phục vụ
mọi m cùng (seed,N,k); chạy maximum-m query chung, dùng exact prefixes. Đo thêm
32 queries cố định, riêng từng m để không gọi tốc độ query m=8000 là tốc độ m=100.
Estimated core latency = mean smoothing + warmed per-m query + median
aggregation; khác measured wall time của cả job. Diagnostics, output I/O và
setup không nằm trong estimated latency.

Peak memory dùng ru_maxrss trong **subprocess riêng từng (seed,N,k)**, gồm load
bank, embedding, KD tree, matching và diagnostics. Vì các m dùng chung job,
đây không phải benchmark peak memory riêng từng m; không gồm generation workers.
Scratch embeddings được xóa sau mỗi job hoàn tất để giới hạn disk usage.
Prefix loader giải nén full largest packed bank trước khi lấy subset nên peak
RSS của N nhỏ vẫn gồm backing bank chung. Group wall time còn gồm diagnostic
I/O và auxiliary short-history ở seed0 full-N; không xếp hạng tốc độ giữa
seeds bằng total wall time này.
Generation+CSV export time đo từ run.lock mtime tới metadata mtime cho largest
bank; seed0 là **fresh replay duration**, không gán thành duration lịch sử cũ.
Các prefix N nhỏ không được sinh riêng, nên không bịa standalone generation time.
Hai bank seed 1/2 được sinh đồng thời, seed0 replay chạy riêng; các duration này
phản ánh system load thực tế khác nhau, không dùng để xếp hạng thuật toán giữa seeds.

K thay đổi M: báo cả N và M; equal-M ablation là tùy chọn chưa thực hiện.
Có thêm numerical validity audit: core `np.var>0` đôi khi giữ các histories
hoàn toàn phẳng do floating-point roundoff. `segment_validity_audit.csv` đếm
riêng số rows này theo mọi seed/N/k. Không âm thầm đổi core giữa sensitivity;
ảnh hưởng k vì vậy bao gồm cả hành vi bộ lọc hiện tại. Strict constant-filter
ablation là một nghiên cứu implementation riêng, chưa chạy trong E2 này.

## Chạy và tiếp tục

**Với output hiện có:** dùng `experiments/sensitivity/extend_n.py` (lệnh ở cuối)
để bổ sung hoặc kiểm tra/resume mức N mới nhất. Protocol cũ được giữ nguyên; không
resume bằng runner đã đổi design. Lệnh runner dưới đây dành cho chạy từ đầu
trong một output directory mới, ví dụ thêm `--output-dir outputs/experiment_2_fresh`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/sensitivity/run_experiment_2.py
```

Runner tự chuẩn bị các bank thiếu. Các thư mục bank có thể được tạo trước bằng
`experiments/synthetic/generate_python.py` với đúng offsets và N=18387. Không
chạy hai generator đồng thời ghi cùng thư mục. Resume cùng code/input/design:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/sensitivity/run_experiment_2.py --resume
.venv/bin/python -m unittest discover -s experiments/sensitivity -v
.venv/bin/python experiments/notebooks.py experiments/sensitivity/experiment_2.ipynb
```

[experiment_2.ipynb](experiment_2.ipynb) đọc các artifacts thực tế và hình;
không sinh lại synthetic hoặc fit lại khi mở notebook. Xem [kết quả và nhận xét](#results)
cho nhận xét thực tế sau khi job hoàn thành. Theo dõi `outputs/experiment_2/progress.json`
và `run.log`; nếu có lỗi xem `failure.json`. Chỉ coi complete sau khi đủ validation/report.

## Tổ chức

- `design.py`: cấu hình/seed/calendar rules cố định trước scoring.
- `library.py`: validate/checksum/replay, packed banks và exact prefix embeddings.
- `engine.py`: isolated group, exact matching, aggregation, auxiliary availability.
- `run_experiment_2.py`: chuẩn bị contexts, freeze protocol/source, chạy/resume jobs.
- `analysis.py`: paired summaries, stability, interactions, figures/report.
- `test_sensitivity.py`: invariants của thiết kế và phương pháp.

Output `outputs/experiment_2/`: `protocol.json`, `progress.json`, `source_snapshot`,
`configurations.csv`, `contexts.npz`, packed banks, `groups/*/{forecasts,matches}.csv.gz`,
`runtime.json` và completion checksums từng group; summary/seed/horizon tables,
factorial effects, generation/runtime/matching tables, short-history availability,
`nested_minimum_distances.csv`, `validation.json`, `report.md`, PNG/SVG figures.

Đây là thí nghiệm sensitivity bổ sung trên fixed COVID panel, không phải bảng
accuracy của original study hoặc một BO mới. Python RNG/GAM vẫn có khác biệt
so với R/mgcv; không điều chỉnh phương pháp sau khi xem test để ép kết quả tốt hơn.

## Mở rộng N — dùng chung script hiện có

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/sensitivity/extend_n.py --target-n 30000
```

Chạy lại cùng lệnh để kiểm tra/resume các job chưa xong. Không dùng runner cũ
với `--resume` để ghi đè protocol ban đầu. Fresh runner vẫn hỗ trợ đầy đủ
23 configs /42 groups /690.000 main targets trong output directory mới.

Mở rộng E2.A tại k=5/m=4422, ba seeds, cùng targets, theo chuỗi
**18.387→22.000→30.000**. Lượt 22k thêm 3.613 attempts/family; lượt 30k chỉ
thêm 8.000 attempts/family. SeedSequence key `[offset,2000202,family,index]`
không phụ thuộc N đích. Không dịch scalar seeds của wiggle/seasonal khi N đổi;
không sinh bù rejections. Prefix cũ được kiểm tra toàn bộ values/metadata/IDs
và giữ nguyên thứ tự để tie breaking ổn định.

Mỗi mức mới chỉ thêm 30.000 forecasts. Lượt 30k giữ nguyên 39 groups đã có.
Protocol gốc và các amendment trước được giữ nguyên; `effective_protocol.json`
trỏ tới amendment mới nhất. `extension_generation.csv` chứa cả hai lần mở rộng.

Không tạo thêm script/notebook hoặc các bản sao before/render/source_snapshot
cho mỗi lượt: bảng, hình, README, RESULTS và notebook được cập nhật tại chỗ.
Chỉ lưu packed bank, seed audit và ba nhóm forecasts cần cho N mới.
Code provenance được nhúng trong protocol.json thay vì nhân bản file nguồn.
Các backup trước của lượt 22k vẫn giữ nguyên, không tạo thêm backup cho 30k.

<a id="results"></a>

## Kết quả sensitivity và interactions

**Đã chạy xong:** 23 cấu hình ×3 independent training banks ×10.000 common
COVID targets = **690.000 point forecasts**; thêm 28.800 auxiliary targets
cho controlled short-history availability. Notebook:
[experiment_2.ipynb](experiment_2.ipynb); artifacts: `outputs/experiment_2/`.

Reference **(N,k,m)=(18387,5,4422)**; N là attempts/family, h=4, GAM/GCV,
exact L1, omit bridge. Cùng 50 bang ×50 origins của Experiment 4, không đổi
calendar hoặc thresholds theo kết quả. Không chạy BO hoặc UQ trong E2.

MAE/RMSE dưới đây là mean của ba seed-specific scores; SD là SD giữa ba seeds,
không phải CI. Reference mean MAE **5.309,16**, RMSE **20.634,96**, MAE seed SD
**2,37**. Seed 0 khớp chính xác đủ 10.000 point forecasts và labels của Experiment 4.

### E2.A — Thay từng tham số

Giữ k=5, m=4422 khi thay N:

| N attempts/family | Mean MAE | MAE seed SD |
| --- | ---: | ---: |
| 500 | 5.709,83 | 17,92 |
| 2.000 | 5.480,73 | 16,49 |
| 5.000 | 5.392,23 | 11,95 |
| 10.000 | 5.345,72 | 2,71 |
| 18.387 | 5.309,16 | 2,37 |
| 22.000 | 5.298,69 | 5,28 |
| 30.000 | 5.280,97 | 3,03 |

Tăng N cải thiện cả matching và mean accuracy trên chính panel này, với lợi
ích giảm dần: 10.000→18.387 chỉ giảm MAE thêm 36,56 ca (khoảng 0,68%).
Bổ sung N=22.000 theo yêu cầu sau lượt đầu, giữ k=5/m=4422 và cùng ba seeds:
mean RMSE **20.610,46**; paired delta MAE so với N=18.387 là **−10,46 ca**
(khoảng **−0,20%**), 95% state-cluster interval **[−17,12; −4,59]**.
Mức tăng library từ 18.387 lên 22.000 do đó vẫn có cải thiện nhỏ trên panel
này, không chứng minh N=22.000 tối ưu toàn cục. Seed SD tăng từ 2,37 lên 5,28;
N lớn hơn không bảo đảm stability cải thiện đơn điệu.

Bổ sung tiếp **N=30.000**, vẫn k=5/m=4422 và cùng targets: mean MAE
**5.280,97**, mean RMSE **20.572,78**, MAE seed SD **3,03**. So với N=22.000,
MAE giảm thêm **17,72 ca (~0,33%)**; so với reference 18.387 giảm **28,19 ca
(~0,53%)**, paired 95% state-cluster interval **[−46,06; −10,78]**.
Đây là kết quả trên fixed panel, không chứng minh global optimum.

Trên 7.500 seed/state/origin queries, không có lần minimum L1 distance tăng
khi mở rộng nested bank. Đây là tính chất của matching trên nested sets,
không chứng minh accuracy luôn đơn điệu trên các test khác.

Giữ N=18387, m=4422 khi thay k:

| k | Mean MAE | MAE seed SD | Paired delta MAE vs k=5 | 95% state-cluster interval |
| --- | ---: | ---: | ---: | --- |
| 3 | 5.354,36 | 1,87 | +45,20 | [−4,44; 98,71] |
| 5 | 5.309,16 | 2,37 | 0 | [0; 0] |
| 7 | 5.276,19 | 3,75 | −32,97 | [−78,36; 12,58] |
| 9 | 5.314,88 | 3,52 | +5,73 | [−48,03; 60,06] |

k=7 có mean MAE thấp nhất trong đường k, nhưng paired interval vẫn chứa 0.
Không khẳng định k=7 tốt hơn ở mọi bang hoặc là lựa chọn đã xác nhận trên
một test độc lập. k=9 không mang thêm accuracy ở điểm trung bình đang xét.

Giữ N=18387, k=5 khi thay m:

| m | Mean MAE | MAE seed SD |
| --- | ---: | ---: |
| 100 | 5.308,98 | 10,85 |
| 500 | 5.247,65 | 5,16 |
| 2.000 | 5.269,58 | 3,09 |
| 4.422 | 5.309,16 | 2,37 |
| 8.000 | 5.347,53 | 1,14 |

Tăng m làm MAE ít dao động giữa các banks hơn trong các mức này, nhưng không
đảm bảo giảm mean MAE. m=500 giảm mean MAE khoảng 1,16% so với reference;
paired interval của delta là [−110,20; −8,08] ca. Các intervals là mô tả trên
fixed panel, không điều chỉnh multiple comparisons hoặc biến một best row
hậu nghiệm thành winner được xác nhận độc lập. m=8000 ổn định hơn về MAE seed
SD nhưng mean MAE cao hơn reference 38,37 ca. m đếm segments, không phải
số curves độc lập; CSV còn báo số unique matched curves.

### E2.B — Tương tác thật sự có ý nghĩa cho diễn giải

Đủ 8 corners của N={1000,5000}, k={3,7}, m={500,2000}; thêm center=(2000,5,1000)
và paper reference. Heatmaps chỉ hiển thị các ô thực sự chạy, cùng color scale;
các tổ hợp chưa chạy ở hình grid tổng để xám.

Một ví dụ **N × m** từ conditional differences:

| k cố định | MAE change khi m 500→2000 tại N=1000 | Cùng change tại N=5000 | Difference-in-differences |
| --- | ---: | ---: | ---: |
| 3 | +112,42 | +18,25 | −94,18 |
| 7 | +159,36 | +71,52 | −87,84 |

Tăng m gây penalty về mean MAE nhỏ hơn khi thư viện lớn hơn trong các corners
này. Vì vậy ảnh hưởng của m không tách rời N. Các tương tác N×k và m×k cũng
có bảng conditional contrasts riêng trong `conditional_interactions.csv`.

Coded triple interaction có mean 1,59, seed SD 10,44, không ổn định về hướng.
Chỉ xem là exploratory với tám corners và ba seeds; không suy ra quan hệ
ba chiều chắc chắn hoặc tác động nhân quả.

Mean MAE thấp nhất trong 23 rows quan sát là **5.244,04** tại
(N,k,m)=(5000,7,500); đây là mô tả hậu nghiệm, không phải BO winner hoặc
khuyến nghị thay tham số production. Center có mean MAE 5.351,79.

### E2.C — Runtime và stability

Với N=18387, m=4422, mean warmed query benchmark qua ba seeds:

| k | Query benchmark ms/case |
| --- | ---: |
| 3 | 1,41 |
| 5 | 4,91 |
| 7 | 38,65 |
| 9 | 127,90 |

k=9 tốn khoảng 26 lần query latency của k=5 trên máy/lượt chạy này, trong khi
mean MAE không cải thiện. Đây là riêng query benchmark, không phải toàn bộ
forecasting/training time. Lượt runner ban đầu, gồm packing, scoring 21 cấu hình và report,
chạy khoảng **36 phút 21 giây**, chưa gồm các lượt sinh/replay libraries trước đó.

Runtime CSV tách prefix loading, embedding, tree setup, shared maximum-m
queries, aggregation và benchmark riêng từng m. Peak RSS đo trong subprocess
riêng theo (seed,N,k); không gọi peak dùng chung này là memory riêng từng m.
Prefix loader giải nén full largest bank trước khi chọn subset, nên peak của
N nhỏ cũng có backing-bank overhead. Seed0 group wall times còn gồm auxiliary
short-history analysis; không so raw job totals giữa seeds như benchmark thuần.
Generation/export timings là observed durations của full banks; seed0 là
fresh replay, seed1/2 được sinh đồng thời. Không bịa standalone generation
time cho các N nhỏ vốn được lấy từ nested prefixes.

Stability có cả MAE SD và per-target prediction SD/range; ba seeds còn ít để
đánh giá tails. Main errors theo từng horizon có trong `scores_by_horizon.csv`.

### Short-history availability

Giữ N=18387, m=4422, seed0; first 16 genuine panel origins ×50 bang. Chỉ cho
model thấy last L quan sát tại chính vintage đó, không giả snapshot có sớm hơn.

- L=4: mọi k đều unavailable vì GAM core cần ít nhất 5 observations.
- L=6: k=3/5 fit được; k=7/9 không đủ history.
- L=8: thêm k=7.
- L=10: cả bốn k fit được.

Có 28.800 forecast targets hợp lệ và 5.600 case/config combinations được ghi
`insufficient_history`. Không tạo dự báo giả để lấp các ô. Chỉ so các k trong
cùng L và cùng cases; không gộp auxiliary scores vào main 690.000 targets.

### Numerical validity audit — giới hạn quan trọng khi đọc k

Core hiện dùng `np.var(history, ddof=1)>0` sau floor 1e-8. Floating-point
roundoff có thể làm variance rất nhỏ dương trên history hoàn toàn phẳng.
Seed0 reference k=5 giữ **2.047.853 rows phẳng /14.968.931 core rows =13,68%**;
audit full-N seed0 cho k=3/7/9 không thấy rows phẳng được giữ như vậy.

E2 **giữ nguyên core** để so đúng các run trước và ghi rõ hiện tượng trong
`segment_validity_audit.csv`/hình riêng. Vì vậy k sensitivity hiện gồm cả
hành vi numerical validity filter, không chỉ hiệu quả window length. Chưa
chạy strict constant-filter ablation hoặc equal-M control; không coi các
k comparisons là đã cô lập hoàn toàn hiệu ứng riêng của window length.

### Kiểm chứng và artifacts

- 10 tests pass: design/OFAT/factorial, nested attempts, exact neighbor prefixes,
  bridge omission/core forecast parity, paired labels, interaction scaling,
  m≤M và numerical validity audit.
- 23 configs ×3 seeds, đủ 690.000 main targets; reference seed0 khớp Experiment 4.
- Minimum M của toàn bộ jobs là 401.711; mọi requested m được kiểm tra theo M
  của chính job. Nested minimum-distance violations và median-distance decrease
  khi tăng m đều bằng 0.
- Ba banks giữ 54.081 /54.006 /54.042 curves từ cùng 55.161 attempts/bank.
  Full seed0 replay CSV SHA256 khớp bitwise thư viện Python reference.
- 11 nhóm hình PNG/SVG; notebook đã thực thi. Inputs/code/artifact checksums,
  source snapshot, group completion hashes, report metadata và validation JSON
  lưu trong output. Python RNG/GAM chưa được chứng minh numerical parity với R.

### Bổ sung N=22.000/30.000 và bảo toàn kết quả cũ

Điểm N=22.000 là amendment sau nghiên cứu ban đầu theo yêu cầu người dùng,
không ghi lại thành mức đã chọn trước khi xem kết quả. Lượt bổ sung chạy
khoảng 4 phút 4 giây, gồm sinh thêm, ba seed forecasts và cập nhật report.

Giữ nguyên 18.387 attempts/family cũ, thêm 3.613 attempts/family/seed bằng
SeedSequence namespace riêng; không chạy generator với scalar-seed stride mới
vì cách đó sẽ thay wiggle/seasonal ban đầu. Các banks 22k giữ
64.709 /64.643 /64.675 curves hợp lệ từ 66.000 attempts/bank. Từng curve value,
metadata, ID và thứ tự prefix cũ được kiểm tra toàn bộ, không chỉ lấy sample.

Chỉ chạy thêm 30.000 targets; 36 groups cũ và mọi giá trị của 21 summary rows
cũ giữ nguyên. Protocol gốc không bị ghi đè. `effective_protocol.json` nối
protocol gốc với `amendments/N22000/protocol.json`; bảng/hình/notebook trước
mở rộng giữ ở `amendments/N22000/before/`. Generation costs của phần thêm
được ghi riêng trong `extension_generation.csv`.

Lượt N=30.000 nối trực tiếp từ các banks 22k, chỉ thêm 8.000 attempts/family/seed;
không sinh lại phần cũ. Ba banks giữ 88.197 /88.175 /88.178 curves hợp lệ từ
90.000 attempts/bank. Chỉ chạy thêm 30.000 targets, giữ nguyên hashes của
39 groups trước đó. Tổng hiện tại: 23 cấu hình, 3 seeds, 690.000 main forecasts.

Dùng lại `extend_n.py`, notebook và mọi bảng/hình hiện có; không tạo thêm
script, notebook, bản sao toàn bộ báo cáo hoặc source snapshot cho lượt 30k.
Chỉ lưu dữ liệu/kết quả cần thiết; code provenance nhúng trong protocol.json.
Lệnh chạy/resume: `python experiments/sensitivity/extend_n.py --target-n 30000`.
