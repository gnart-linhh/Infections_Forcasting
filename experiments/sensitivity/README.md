# Experiment 2 — Sensitivity, Interactions and Computational Trade-offs

Lượt hiện hành: **protocol v4 / nonconstant_v2**, output
`outputs/experiment_2_nonconstant_v2/`. Giữ nguyên 23 cấu hình ×3 seeds ×10.000
targets = **690.000 forecasts**, kể cả N=22.000 và N=30.000. Results v1 chỉ
được lưu để đối chiếu ở `outputs/experiment_2/`, không nhập lại forecasts.
[Notebook](experiment_2.ipynb) và [report mới](../../outputs/experiment_2_nonconstant_v2/report.md)
chỉ đọc output mới sau validation. Chưa công bố ranking v2 khi chưa complete.

Các packed banks và extensions cũ được kiểm tra hashes, budget, seed và mọi
prefix curve/ID rồi link sang output mới; không sinh CSV/bank trùng. Các links
phụ thuộc dữ liệu gốc nên không xóa packed banks nguồn. Mỗi group ghi filter,
constant rows=0, contexts/bank hashes và numerical source; analysis/notebook
kiểm tra lại. Thay k vẫn đồng thời thay M và geometry, chưa có equal-M control.

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
Numerical validity audit bắt buộc retained constant histories =0 cho mọi
N/k/seed với nonconstant_v2. Không dùng variance floating point để giữ nhầm
flat rows. Thứ hạng k=5/k=7 và các kết luận v1 phải được tính lại; không điều
chỉnh filter hoặc các mức tham số để giữ thứ hạng cũ.

## Chạy và tiếp tục

```bash
# Fresh output; không khởi chạy trùng job đang có.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/sensitivity/run_experiment_2.py \
  --reuse-banks-from outputs/experiment_2 \
  --output-dir outputs/experiment_2_nonconstant_v2 \
  --notebook-path experiments/sensitivity/experiment_2.ipynb
.venv/bin/python -m unittest discover -s experiments/sensitivity -v
```

Log/progress/launch.json và frozen source nằm trong output mới. Dùng đúng
resume_command trong launch.json để giữ method/data/source identity. Notebook
được execute tự động sau khi report/hash audit xong; nếu có lỗi, E3 đang chờ
sẽ dừng thay vì dùng kết quả chưa kiểm chứng.

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

## Các mức N mở rộng

N22k/N30k được giữ nguyên từ yêu cầu trước, cùng prefix namespace và rejected
attempts; không thêm mức mới hoặc thay configurations sau khi xem v2 errors.
Các extension files cũ được dùng lại và kiểm tra giá trị/IDs của prefix đầy đủ.

<a id="results"></a>

## Kết quả hiện hành

Đọc `outputs/experiment_2_nonconstant_v2/progress.json`; chỉ kết luận khi
status=complete, notebook_status=complete, validation ghi đúng nonconstant_v2,
constant_history_rows=0 và đủ 690.000 targets. Các bảng/hình/notebook sẽ được
sinh từ chính run này. Report v1 vẫn nằm ở `outputs/experiment_2/report.md`;
không dùng MAE hoặc ranking cũ như kết luận của bộ lọc mới.
