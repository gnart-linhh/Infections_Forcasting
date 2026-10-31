# Infections Forcasting — full Python SMOA reproduction

Pipeline: **synthetic → Bayesian optimization → UQ → prediction → evaluation**.
Chỉ dùng Python; không cần R, SLURM hoặc đưa checkout `precog/` vào repo này.

Run full với bộ paper hiện tại: [hướng dẫn và phạm vi](experiments/reproduction/FULL.md),
[notebook](experiments/reproduction/full.ipynb), outputs `outputs/paper_full/`.
50 bang × **112 origins hàng tuần** × 4 horizons, dùng snapshot lịch sử.
Full hiện **bỏ bridge + UQ hồi cứu có future labels** theo yêu cầu. BO đã hoàn
tất 104 lượt; `outputs/bo_gam_full/` chạy winner trên cùng test và quy ước để
so với bộ paper. Không gọi WIS/coverage của nhánh full này là real-time.

## Experiment 0 — đối chiếu paper

[Experiment 0](experiments/reproduction/README.md) và
[notebook kết quả](experiments/reproduction/experiment_0.ipynb) kiểm tra 4 bang,
6 lịch phát hành, horizon 1–4 tuần, MAE/WIS và đối chiếu hình paper.
**Run mới `outputs/experiment_0_source/`** dùng snapshot JHU lịch sử, forecast
COVIDHub chính thức, thin plate GAM/GCV và đúng bố cục Figure 2. Report có
điểm trên targets chung và ablation dữ liệu/smoothing.

Run full 50 bang cũ (104 BO evaluations, 22.468 forecasts/model) dùng **final
truth + legacy spline**: MAE 7.564,27, WIS 4.765,92; chưa phải kết quả đã sửa.
Không dùng điểm đó để khẳng định đã khớp paper.

## Cấu hình thực nghiệm

| Thành phần | Cấu hình mặc định theo code gốc |
| --- | --- |
| Thư viện synthetic tham chiếu | 18.387 lần sinh/family × 3 families = **55.161** |
| BO search space | `num_curves=10000..30000`, `k=2..7`, `closest=1000..6000` |
| BO budget | **99 LHS + 5 EI = 104 objective evaluations** |
| Synthetic trong mỗi BO trial | Sinh độc lập 30.000–90.000 training curves; validation `floor(num_curves/1000)` mỗi family, shuffle rồi tối đa 50 curves |
| Validation origins | Tất cả origins đủ điều kiện, không subsample |
| Dữ liệu đánh giá mặc định | Đủ 50 bang trong `outputs/truth_states/manifest.csv`; tất cả origins đủ điều kiện từ 2020-08-15 |
| Smoothing | Gaussian thin plate regression spline, rank round(n/2), GCV như mô hình mgcv nguồn |
| UQ | Online negative binomial, 5.000 draws; 23 quantiles cho 11 mức coverage |
| Evaluation | Raw COVID truth, MAE/WIS của sMOA, biểu đồ dự báo/thực tế và prediction intervals từng bang |

18.387 là giá trị hard-coded trong `run_smoa.R` sau BO của tác giả. BO Python
được chạy lại trên miền gốc và có thể chọn số curves khác: prediction dùng
**đúng training bank/k/closest của winner đã thực sự evaluate**, không ép BO
trả lại tham số đã công bố. `01_synthetic` là thư viện đầy đủ ở quy mô tham
chiếu, dùng để kiểm tra generation/EDA; BO tự sinh bank riêng theo mỗi candidate.
Không có thư viện preview hay profile `small`.

Số lượng trên là **số lần sinh** như code gốc; nếu generator trả `None` thì bỏ
mẫu đó và ghi accepted/failed. Không sinh bù âm thầm hoặc báo attempts thành
accepted curves.

## Chạy full

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -u experiments/run_pipeline.py \
  --manifest outputs/truth_states/manifest.csv \
  --protocol source --workers 4 \
  --output-dir outputs/reproduction_gam_full
```

Đây là công việc dài; số worker điều khiển sinh curve bằng các process độc lập,
không giảm số trials, training curves hay origins. 99 LHS objectives được xử
lý tuần tự để không đồng thời giữ 99 embedding matrices trong RAM. EI vẫn là
5 vòng tuần tự. Đây là khác biệt về lịch tính toán so với SLURM gốc.

Xem tiến độ và tiếp tục khi bị ngắt:

```bash
.venv/bin/python experiments/status.py --output-dir outputs/reproduction_gam_full
.venv/bin/python -u experiments/run_pipeline.py \
  --manifest outputs/truth_states/manifest.csv \
  --protocol source --workers 4 \
  --output-dir outputs/reproduction_gam_full --resume
```

Resume yêu cầu cùng cấu hình, dữ liệu và checksum code. Checkpoint giữ completed
trials và RNG của LHS/EI; trial đang dở được chạy tiếp từ bank đã checkpoint,
hoặc sinh lại cùng seed nếu chưa sinh xong bank. Synthetic CSV checkpoint theo
500 attempts, không nhân đôi dòng khi tiếp tục. Mỗi output có process lock để
tránh hai lệnh cùng ghi kết quả.

Nếu chủ động chỉ đánh giá một bang, thay `--manifest` bằng
`--truth outputs/truth_states/California.csv --series California`; synthetic và BO vẫn
full mặc định. Các numeric overrides được ghi `custom` khi khác ngân sách gốc;
fixtures nhỏ chỉ nằm trong tests.

## Output

```text
outputs/reproduction_full/
  run_config.json       cấu hình bất biến, input/code checksums
  pipeline.json         trạng thái thực tế và các bước hoàn tất
  01_synthetic/         curves.csv (đầy đủ), metadata; checkpoint khi đang sinh
  02_optimization/      config, trials, checkpoint RNG, progress, selected bank
  03_uq/                policy; dispersion thực tế fit trước prediction mỗi origin
  04_prediction/        01/, 02/, ... forecasts/quantiles/calibration của từng bang
  05_evaluation/        report.md, MAE/WIS tổng hợp, scores/coverage, báo cáo từng bang
```

Trong `05_evaluation/01/`, `02/`, ... có `report.md`,
`forecast_vs_actual.png/.svg` và `metrics_by_horizon.png/.svg`.
`series_metrics.csv` ánh xạ số thư mục với tên bang. MAE/WIS cũng in ra terminal.
Chỉ báo hoàn tất khi các artifact thực tế được ghi xong; không có bảng điểm
full trước khi job full hoàn thành.

## Dữ liệu COVID thật

README gốc phân biệt ba nguồn dữ liệu:

| Nguồn | Mục đích | Trạng thái local |
| --- | --- | --- |
| `state_truths/*.RData` | Final COVID truth của từng bang | Đã chuyển bằng Python sang `outputs/truth_states/`, có provenance/checksums |
| `tdat_list_tot_weekly.csv` (~5,5 GB theo nguồn) | Archive truth theo từng thời điểm công bố | Đã dựng lại từ JHU 50 bang × 112 vintage hàng tuần cho test chung tại `outputs/paper_weekly_archive/`; giữ cache Figure 2 riêng. Không phải tải toàn bộ file gốc 5,5 GB |
| `scores_tot_w_wis_components.csv` (~100 MB theo nguồn) | Điểm của external models | Đã lấy forecast baseline/ensemble cho Figure 2 và chấm targets chung; chưa có score archive toàn bộ models |

Run `paper_full` dùng **historical snapshots** cho mọi tuần của test chung.
Các lệnh khác dùng manifest không chứa `as_of_truth` vẫn lấy final truth làm
history. Smoother dùng cùng thin plate/GCV, nhưng chưa xác minh numerical parity
với mgcv; RNG/ODE/GP vẫn khác số học với R.
[Audit](REPRODUCTION_AUDIT.md) ghi phạm vi và khác biệt còn lại;
`publication_reproduction_complete` chưa được đặt thành true.

## Tài liệu và kiểm tra

Giao diện chạy và xem kết quả bằng notebook:
[forecasting.ipynb](experiments/forecasting/forecasting.ipynb) →
[evaluation.ipynb](experiments/forecasting/evaluation.ipynb).
Chọn kernel `.venv`; cell cấu hình mặc định đọc job full hiện tại.

Chạy và lưu một notebook bằng cùng môi trường Python:

```bash
.venv/bin/python experiments/notebooks.py experiments/comparison/experiment_1.ipynb
```

Các notebook dùng chung runner này; không có script executor riêng cho từng thư mục.

- [Synthetic generation](experiments/synthetic/README.md).
- [Bayesian optimization](experiments/optimization/README.md).
- [UQ, prediction, evaluation và plots](experiments/forecasting/README.md).
- [Benchmark 50 bang](experiments/forecasting/README.md#benchmark-50-bang).
- [Dataset exploration](experiments/exploration/README.md).
- [Experiment 1: so sánh các mô hình theo độ dài history](experiments/comparison/README.md)
  (nhóm chính hoàn tất 464.800 forecasts: early-stage và fixed-origin history control; Holt/ARIMA chưa chạy do thiếu statsmodels).
- [Experiment 2: parameter sensitivity và interactions](experiments/sensitivity/README.md):
  N/k/m, nested libraries, ba training seeds, factorial 2³, accuracy–stability–runtime.
- [Experiment 3: synthetic library characteristics](experiments/library_characteristics/README.md):
  từng họ so với mixed, ba seeds và controls đồng nhất số curves/số matching windows.
- [Experiment 4: sai số theo điều kiện dịch](experiments/error_analysis/README.md):
  snapshot đúng origin, 50 bang × 50 origins, bộ paper N=18.387/k=5/m=4.422,
  MAE/RMSE/bias/extreme errors và biểu đồ chẩn đoán.
- [Experiment 5: analogue quality và forecast accuracy](experiments/analogue_quality/README.md):
  matching distances, Spearman, good-match failures và held-out risk diagnostics.
- [Experiment 6: probabilistic forecast evaluation](experiments/probabilistic/README.md):
  coverage/MPIW/WIS theo horizon, history và điều kiện dịch; chẩn đoán UQ negative binomial gốc.

```bash
.venv/bin/python -m unittest discover -s experiments/synthetic -v
.venv/bin/python -m unittest discover -s experiments/optimization -v
.venv/bin/python -m unittest discover -s tests/forecasting -v
.venv/bin/python -m unittest discover -s experiments -p test_pipeline.py -v
```

Code và kết quả demo/mini, legacy generator và các snapshots điểm số cũ đã được
dọn. Giữ CSV COVID và provenance đầu vào. `outputs/`, `.venv/`, checkout tham
chiếu `precog/` không commit. Repo đích:
[gnart-linhh/Infections_Forcasting](https://github.com/gnart-linhh/Infections_Forcasting).
