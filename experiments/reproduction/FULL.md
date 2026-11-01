# Full weekly COVID run — paper parameters, then BO winner

Full mặc định dùng **UQ matured theo nhánh chính `run_smoa.R`**, giữ
**N=18.387 attempts/family, k=5, m=4.422**, GAM/GCV và **omit bridge** theo yêu cầu trước.
Bản hồi cứu đã hoàn tất nằm riêng ở `outputs/paper_full_nonconstant_v2/`;
không thay các scores của nó bằng scores từ policy mới.

## Đối chiếu UQ với R

README nguồn gọi `run_smoa.R`. Tại commit
`aba1948d228e5380d72907048236601874b53cac`,
[main script dòng 253](https://github.com/lanl/precog/blob/aba1948d228e5380d72907048236601874b53cac/smoa/R/run_smoa.R#L253)
lọc `target_end_date <= curr_targetend_date` trước MLE riêng từng horizon.
[Figure 2 dòng 238](https://github.com/lanl/precog/blob/aba1948d228e5380d72907048236601874b53cac/smoa/R/figure_2.R#L238)
fit trực tiếp toàn bộ earlier-issued forecast rows, không lọc maturity.
Vì vậy full dùng `matured`; `experiment_0` giữ `figure2_retrospective` để
đối chiếu Figure 2. Policy hồi cứu chỉ chạy khi chỉ định rõ.

Chỉ thay UQ eligibility. Bridge vẫn omit theo yêu cầu trước, trong khi main R
keep bridge: **chưa gọi toàn bộ full là replay nguyên trạng main R**. Quy tắc
UQ và initialization `[20,10,6,0.3]` khớp nhánh main.

- MLE chỉ nhận earlier-issued rows có target date ≤ origin; current-issued forecasts không vào prior.
- Source chuyển MLE sau 24 stored rows, không đòi sáu mature labels mỗi horizon.
- Giữ NB mean<100 rule, 5.000 draws, seed13, median recentering, 23 quantiles/11 levels; WIS trên bảy quantiles nguồn.
- R dùng final truth cho labels hiệu chỉnh và warmup từ snapshot cố định 15/08/2020. Python giữ cả hai để đối chiếu nguồn. `matured` **không tự làm cả pipeline causal**, vì revisions và frozen warmup vẫn hồi cứu.
- Point forecasts, matching và MAE không phụ thuộc policy UQ; WIS, dispersion và coverage phải tính lại.

## Trạng thái các lượt nonconstant_v2

- Paper: `outputs/paper_full_nonconstant_v2_matured/`, đã complete đủ 22.400 targets.
- BO vẫn chạy nguyên: `outputs/bo_gam_nonconstant_v2/02_optimization/`, 99 LHS +5 EI, GAM, all validation origins, 4 workers. BO tối ưu point-MAE, không dùng NB/UQ, nên không restart vì đổi policy này.
- Full winner tự chờ: `outputs/bo_gam_full_nonconstant_v2_matured/`.
- Lịch chờ winner hồi cứu cũ đã được đánh dấu superseded, không chạy tiếp.
- Mỗi job có `launch.json`, PID và resume command. Hai full dùng cùng source snapshot tại `outputs/paper_full_nonconstant_v2_matured/source_snapshot/`.

[full.ipynb](full.ipynb) đọc các thư mục matured mới. Khi chưa complete chỉ
xem trạng thái/artifacts hiện có; không trình bày scores hồi cứu như kết quả mới.

## Phạm vi và dữ liệu

**50 bang ×112 origins hàng tuần ×4 horizons =22.400 targets/model**.
Origins 15/08/2020–01/10/2022; targets tới 29/10/2022. Tất cả 112 JHU vintages
đã chuẩn bị ở `outputs/paper_weekly_archive/`; không fallback sang final history.
Dùng raw final truth để scoring. Synthetic Python không sinh lại: 55.161
attempts, 54.081 accepted curves, checksum được kiểm chứng. Bộ lọc
`nonconstant_v2` giữ 12.921.078 omit windows ở k=5,h=4.

Chỉ sinh/calibrate/đánh giá/vẽ sMOA, đủ 1–4 week-ahead forecasts, MAE/WIS và
prediction intervals. Matches lưu lossless trong `matches.csv.gz`.

## Chạy và theo dõi

```bash
.venv/bin/python experiments/status.py --output-dir outputs/paper_full_nonconstant_v2_matured
.venv/bin/python experiments/status.py --output-dir outputs/bo_gam_nonconstant_v2
.venv/bin/python experiments/status.py --output-dir outputs/bo_gam_full_nonconstant_v2_matured
```

Các job đã khởi chạy, không chạy thêm lần trùng. Khi cần tiếp tục, dùng đúng
`resume_command` trong launch.json để giữ frozen source và config.
Ví dụ cho một lượt mới với cùng policy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/reproduction/run_full.py \
  --uq-policy matured --bridge-mode omit \
  --embedding-cache outputs/paper_embedding_k5_h4_omit_nonconstant_v2 \
  --output-dir outputs/paper_full_nonconstant_v2_matured
```

Sau khi BO complete, driver winner dùng cùng UQ matured, archive, lịch,
software, seeds và raw labels. `paper_comparison/` ghép đúng từng target;
không so aggregate giữa các protocols khác nhau. BO tối ưu synthetic validation,
không bảo đảm thắng cấu hình paper trên COVID.

Source/RNG/GAM/NB của Python chưa có numerical parity hoàn toàn với R;
[REPRODUCTION_AUDIT.md](../../REPRODUCTION_AUDIT.md) ghi các khác biệt còn lại.

## Kết quả matured — đối chiếu UQ chính trong R

Đã hoàn tất dự báo **50 bang /22.400 targets** với `nonconstant_v2`, omit bridge,
UQ matured. Full source reproduction vẫn có các khác biệt bridge/RNG/GAM và
warmup/revisions đã khai báo ở trên.

| Metric | Hồi cứu Figure 2 (`paper_full_nonconstant_v2`) | Matured main R (`paper_full_nonconstant_v2_matured`) |
| --- | ---: | ---: |
| MAE | 7.384,66 | 7.384,66 |
| WIS | 4.727,47 | 4.747,38 |
| Coverage 95%, lower-open như R | 87,33% | 87,17% |

Kiểm tra độc lập toàn bộ **22.400 calibration counts** bằng timestamp của
issued/target pairs; không dùng target chưa đến hạn. Fit lại **600 NB sizes**
(ba origins ×bốn horizons ×50 bang) trên đúng matured pairs cho kết quả khớp.
`predictions.csv` của cả 50 bang giống byte-for-byte với bản v2 hồi cứu;
parameters, library hash, raw-truth hashes và origins giống nhau. Thay policy
chỉ ảnh hưởng UQ/WIS/coverage trong phép đối chiếu này.

Nguồn số: `04_prediction/summary.json`, `calibration.csv`,
`warmup_forecasts.csv`, `forecasts.csv` và `predictions.csv` từng bang.
Pipeline đã complete; đủ 50 báo cáo bang và hình tại `05_evaluation/`.
53 forecasting tests pass, gồm kiểm tra default matured và future-label perturbation.

<a id="results"></a>

## Kết quả hồi cứu đã lưu — không phải UQ mặc định hiện tại

Run paper hoàn tất 30/10/2026 22:40 giờ Việt Nam. Output lịch sử: `outputs/paper_full/`.

**Quy ước của bảng kết quả lịch sử này:** `bridge_mode=omit`, `uq_policy=figure2_retrospective`.
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
