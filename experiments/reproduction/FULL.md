# Full weekly COVID run — paper parameters, then BO winner

Run paper: `outputs/paper_full/`. Giữ **N=18.387 attempts/family, k=5,
m=4.422**, GAM/GCV. Tái dùng synthetic Python đầy đủ và kiểm SHA256: 55.161
attempts, 54.081 accepted curves. Không chạy lại BO hoặc gắn nhãn bộ paper là
winner BO Python. BO ở `outputs/bo_gam/` đã hoàn tất 104 evaluations; winner
N=29.357, k=7, m=5.167 được chạy lại với cùng quy ước mới.

**Quy ước full hiện tại:** bỏ bridge difference và dùng **UQ hồi cứu**, kể cả
outcomes của targets chưa đến hạn trong các forecasts đã phát trước đó.
Chỉ sinh/calibrate/đánh giá/vẽ **sMOA**. Hai bộ paper và BO là hai cấu hình
tham số của cùng mô hình sMOA.
[Run paper hiện hành đã hoàn tất](#results): MAE 7.384,66, WIS 4.727,47.
Run winner BO cũng đã hoàn tất: MAE 7.443,16, WIS 4.734,66; bảng ghép targets
ở `outputs/bo_gam_full/paper_comparison/` và trong notebook.
Run giữ bridge/matured trước đó được lưu riêng ở `outputs/paper_full_bridge_matured/`;
run winner cũ đang dở ở `outputs/bo_gam_full_bridge_matured_partial/`.

## Phạm vi full

**50 bang × 112 origins hàng tuần × 4 horizons = 22.400 targets/model.**
Origins từ 15/08/2020 đến 01/10/2022; targets có thể tới 29/10/2022.
Đây là toàn bộ lịch tuần trong cửa sổ chung có đủ bốn future labels cho cả
50 bang, không subsample 50 origins như Experiment 4. Arizona có truth kéo
dài riêng tới 2023; các tuần chỉ Arizona có dữ liệu không thuộc test chung.

`outputs/paper_weekly_archive/` chứa snapshot JHU cho mọi origin, URL commit
bất biến và checksums, lấy từ index chính thức `reichlab/covidData`. Tái dùng
raw cache đã tải; tải bổ sung các vintage thiếu. County cumulative → daily
differences → Sunday–Saturday state weekly sums. Saturday là tuần quan sát
cuối, cutoff dữ liệu Sunday `origin+1`, issuance Monday, theo code LANL.
Không thay snapshot thiếu bằng final truth. Final raw truth dùng chấm điểm.

## UQ và phương pháp

- Gaussian thin plate GAM/GCV, exact L1 matching, median `diff(y)` **bỏ bridge**.
  Dùng h+1 future levels để lấy đủ h=4 differences, như embedding Figure 2.
- NB initialization sizes `[20,10,6,0.3]`; sau warmup dùng MLE riêng cho mỗi
  horizon trên mọi forecast **đã phát trước origin**, kể cả khi
  `target_end_date > origin`, với final outcomes có sẵn trong dữ liệu hồi cứu.
- Calibration dùng final truth kể cả target chưa mature theo policy hồi cứu;
  prehistory trước 15/08/2020 tính từ snapshot cố định ngày 15/08. **Cả UQ và
  warmup đều hồi cứu**; không gọi WIS/coverage là kết quả real-time. Không replay bản
  ghi first-origin bị lặp trong upstream; negative outcomes loại khỏi NB
  likelihood và ghi số lượng, giữ nguyên trong evaluation.
- Chỉ thay bridge và eligibility UQ theo yêu cầu; NB initialization vẫn là
  `[20,10,6,0.3]` của main runner, không đổi sang initialization riêng của Figure 2.
  `calibration.csv` ghi `calibration_policy` và `future_calibration_n` để kiểm chứng.
- 5.000 NB draws/horizon, seed 13, median recentering; 23 quantiles/11 coverage
  levels, WIS trên 7 quantiles như helper gốc.
- Chạy một luồng BLAS, dùng disk-backed embedding đầy đủ. Mọi neighbor match
  vẫn lưu curve ID/window start/distance; ghi dần ra **matches.csv.gz** nén
  lossless để giảm RAM và dung lượng đĩa, không giảm số neighbors.

Run này dùng Python core của repo. RNG/Monte Carlo/GAM optimizer có thể khác
R; chưa khẳng định numerical parity, khớp toàn bộ figures hay ranking các
external models của paper. Các outputs legacy không được đổi nhãn thành run mới.

## Chạy và xem tiến độ

```bash
.venv/bin/python -u experiments/data/prepare_weekly_archive.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/reproduction/run_full.py \
  --embedding-cache outputs/paper_embedding_k5_h4_omit
.venv/bin/python experiments/status.py --output-dir outputs/paper_full
```

Thêm `--resume` để tiếp tục cùng inputs/source/config. Source đang chạy được
đóng băng trong `source_snapshot/`; command resume chính xác trong `launch.json`.
Notebook [full.ipynb](full.ipynb) đọc tiến độ, MAE/WIS và hình từ outputs thật.

Output `paper_full/`: `run.log`, `pipeline.json`, `run_config.json`,
`03_uq/policy.json`, `04_prediction/01..50/`, `05_evaluation/report.md`,
`scores.csv`, `coverage.csv`, `series_metrics.csv` và hình PNG/SVG từng bang.
Chỉ hoàn tất khi `pipeline.json` có `status=complete`, đủ 22.400 targets/model.

Đợt loại diagnostic model dùng `--reuse-predictions-from` để giữ nguyên từng
giá trị sMOA đã phát, không refit hoặc resample NB. Quy tắc số học UQ và source
model/embedding được kiểm tra trước khi reuse; provenance trỏ tới generation
metadata gốc. Evaluation và mọi hình được dựng lại chỉ cho sMOA. Các lần chạy
mới không dùng tùy chọn reuse sẽ sinh trực tiếp sMOA-only. Không tính thời gian
xử lý/lọc này thành một lần inference mới.

## Chạy lại khi BO hiện tại hoàn tất

Driver chờ run paper hoàn tất, sau đó dùng đúng bank/k/m của
winner trên cùng archive, lịch, seed, UQ/scoring và phần mềm:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/reproduction/run_full.py \
  --selection-dir outputs/bo_gam/02_optimization --wait-for-selection \
  --paper-run outputs/paper_full --output-dir outputs/bo_gam_full
.venv/bin/python experiments/status.py --output-dir outputs/bo_gam_full
```

`waiting_for_bo`/`waiting_for_paper` là đang chờ; chưa có kết quả COVID của winner. Driver
không chọn incumbent tạm thời; chỉ dùng selection đã complete, đúng artifact
hashes và GAM. Nếu BO/paper lỗi thì dừng với trạng thái blocked. Khi chạy xong,
`bo_gam_full/paper_comparison/` có MAE/WIS trên targets ghép chính xác và theo
horizon; không so hai aggregate trên các test khác nhau. BO tối ưu synthetic
validation nên không đảm bảo thắng bộ paper trên dữ liệu COVID thật.

<a id="results"></a>

## Kết quả full hiện hành

Run paper hoàn tất 30/10/2026 22:40 giờ Việt Nam. Output hiện hành: `outputs/paper_full/`.

**Quy ước hiện hành theo yêu cầu:** `bridge_mode=omit`, `uq_policy=figure2_retrospective`.
UQ dùng final outcomes của các forecasts đã phát trước đó, kể cả targets chưa đến hạn. Đây là kết quả **hồi cứu, có future labels trong UQ**, không phải đánh giá real-time causal.

### Bộ paper — chỉ sMOA

**N=18.387 attempts/family, k=5, m=4.422**, cùng Python synthetic bank đầy đủ: 55.161 attempts, 54.081 accepted; GAM/GCV; NB 5.000 draws, seed 13.

**50 bang × 112 origins hàng tuần × 4 horizons = 22.400 targets sMOA**. Origins 15/08/2020–01/10/2022; dữ liệu và targets giữ như run trước.

| Metric | sMOA |
| --- | ---: |
| MAE | 7.384,66 |
| WIS | 4.727,47 |
| Empirical coverage của khoảng 95% | 87,33% |

| Horizon | Targets | MAE | WIS |
| --- | ---: | ---: | ---: |
| 1 tuần | 5.600 | 4.173,93 | 2.628,44 |
| 2 tuần | 5.600 | 6.273,88 | 3.988,48 |
| 3 tuần | 5.600 | 8.808,29 | 5.624,65 |
| 4 tuần | 5.600 | 10.282,54 | 6.668,32 |

### Kiểm chứng quy ước và thời gian

- Thời gian run paper: 420,9 giây, gồm dựng embedding/cache mới, UQ, prediction, evaluation và plots; tái dùng synthetic/raw archive đã có.
- Bỏ bridge bằng cách dựng thêm một future level, lấy `diff(y)` để vẫn có đủ 4 horizons. Đã tái dựng một dự báo từ toàn bộ 4.422 neighbor matches đã lưu.
- Trong 22.400 UQ fits sMOA, có 16.794 fits dùng outcomes chưa đến hạn; tổng 33.564 lượt dùng future outcomes qua các fits. Số này là lượt sử dụng, không phải số targets duy nhất.
- Horizon 1 không có future outcomes từ forecasts đã phát trước đó trên lịch tuần; horizon 2/3/4 có tối đa 1/2/3 outcomes chưa đến hạn mỗi fit. Các outcomes âm bị loại và ghi count.
- Lưu đủ 24.763.200 neighbor matches ở `matches.csv.gz`, nén lossless; 515.200 quantiles sMOA, 11 coverage levels.
- 41 forecasting tests + 2 pipeline tests + 5 reproduction tests pass. Test future-label sensitivity xác nhận future labels đổi retrospective intervals và giữ nguyên point forecasts khi chỉ đổi UQ.
- Giữ NB initialization main `[20,10,6,0.3]` và frozen warmup; không đổi các thành phần khác sang toàn bộ Figure 2. Trong lịch full này đã có đủ warmup để dùng MLE từ origin chấm điểm đầu.

### Winner BO — đã chạy xong cùng quy ước mới

BO đã hoàn tất 104 evaluations, chọn **N=29.357, k=7, m=5.167**. Cả hai run đều bỏ bridge và dùng retrospective UQ trên đúng cùng 22.400 targets, cùng raw truth, software/code hashes và seed.

| Bộ tham số | MAE | WIS | Coverage 95% |
| --- | ---: | ---: | ---: |
| Paper: 18.387 / 5 / 4.422 | 7.384,66 | 4.727,47 | 87,33% |
| BO: 29.357 / 7 / 5.167 | 7.443,16 | 4.734,66 | 88,00% |

Winner BO có MAE cao hơn 58,50 (0,79%) và WIS cao hơn 7,18 (0,15%) trên COVID test này. BO tối ưu synthetic validation nên không bảo đảm thắng bộ paper trên COVID; chênh lệch nhỏ này chưa được kiểm định ý nghĩa thống kê.

| Horizon | MAE paper | MAE BO | WIS paper | WIS BO |
| --- | ---: | ---: | ---: | ---: |
| 1 tuần | 4.173,93 | 4.155,17 | 2.628,44 | 2.599,01 |
| 2 tuần | 6.273,88 | 6.280,01 | 3.988,48 | 3.933,18 |
| 3 tuần | 8.808,29 | 8.889,57 | 5.624,65 | 5.653,61 |
| 4 tuần | 10.282,54 | 10.447,89 | 6.668,32 | 6.752,82 |

Thời gian của lượt sinh dự báo gốc: paper 420,9 giây; BO winner 2.158,8 giây, không tính thời gian chờ hoặc BO optimization. Sau khi loại diagnostic model, giữ nguyên các giá trị sMOA đã sinh và dựng lại evaluation/plots; không coi bước lọc này là inference mới.

Warmup tối thiểu phụ thuộc k: paper có 24 origins/bang, winner k=7 có 22. Các scoring origins và raw inputs tại scoring giữ nguyên; quy tắc UQ giống nhau nhưng số past forecasts dùng calibration có thể khác.

So sánh ghép targets và CSV chi tiết: `outputs/bo_gam_full/paper_comparison/`. Notebook full đã thực thi, đọc cả hai kết quả và bảng so sánh.

### Artifacts và giới hạn

- [Notebook full](full.ipynb); [protocol và cách chạy](FULL.md). Reports/plots từng bang nằm trong `outputs/paper_full/05_evaluation/`.
- Run cũ giữ bridge/matured được lưu tại `outputs/paper_full_bridge_matured/`; run winner cũ đang dở tại `outputs/bo_gam_full_bridge_matured_partial/`. Neighbor audits của các run cũ đã nén lossless.
- MAE/WIS mới không được diễn giải là bằng chứng cải thiện real-time từ việc dùng future labels. Cả embedding lẫn eligibility UQ đã thay đổi; không quy toàn bộ chênh lệch cho một yếu tố.
- Python RNG/ODE/GAM/NB Monte Carlo khác R; chưa kiểm numerical parity hay external-model ranking của toàn paper. `publication_reproduction_complete` vẫn false.

Các artifacts hiện hành chỉ chứa sMOA. Dự báo/quantiles sMOA được giữ nguyên chính xác từ generation trước; evaluation và plots được dựng lại. Provenance: `04_prediction/reuse_provenance.json`.
