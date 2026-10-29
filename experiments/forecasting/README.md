# Bước 3–5 — UQ → prediction → evaluation

Thứ tự toàn project: synthetic (1) → Bayesian selection (2) → UQ (3) →
prediction (4) → evaluation (5). Benchmark nhiều bang thuộc evaluation;
không đặt Bayesian optimization sau benchmark.

## Dùng notebook

Đối chiếu yêu cầu tái hiện paper: mở
[Experiment 0](../reproduction/experiment_0.ipynb). Notebook đó có lịch cố định,
hình paper Figure 2 và Python cùng bố cục, historical JHU, forecast gốc COVIDHub và bảng sai khác.

Mở và chạy lần lượt bằng kernel `.venv` của project:

1. [forecasting.ipynb](forecasting.ipynb): theo dõi BO, xem COVID truth và UQ,
   chạy dự báo full hoặc đọc kết quả pipeline, xem forecast 1–4 tuần của một origin.
2. [evaluation.ipynb](evaluation.ipynb): MAE/WIS tổng hợp và theo horizon,
   actual-vs-forecast diagrams, interval calibration và external-model comparison.

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Chọn **Run All**. Mặc định notebooks đọc kết quả job full hiện tại; khi BO chưa
xong sẽ hiển thị trạng thái pending. Không lấy tham số demo hoặc tạo score giả.
`SELECTED_SERIES` chỉ chọn bang để hiển thị, không giảm quy mô full 50 bang.

Để chạy riêng từ BO winner đã hoàn tất, chọn `FORECAST_MODE="run"` trong
notebook forecasting. Để chấm lại issued predictions, chọn
`EVALUATION_MODE="recompute"` trong notebook evaluation. Các chế độ này ghi vào
`outputs/notebook_forecasting/full` và `outputs/notebook_evaluation/full`.
Nếu evaluation dùng forecast vừa chạy riêng, đổi `PREDICTION_DIR` tương ứng.

Các module `.py` bên dưới chứa hàm dùng chung cho notebook và pipeline tự động.
Notebook trình bày thí nghiệm theo từng cell; thuật toán không bị chép thành
hai bản khác nhau. Kết quả CSV, JSON, report và PNG/SVG vẫn được xuất.

## Các file được giữ

Folder gồm hai notebook, các module dùng chung và README:

| File | Vai trò |
| --- | --- |
| `run_backtest.py` | Chạy một bang theo các forecast origins |
| `run_benchmark.py` | Chạy toàn bộ bang trong manifest, chia sẻ embedding/index |
| `selection.py` | Kiểm tra và áp dụng library/tham số BO winner |
| `model.py`, `smoothing.py`, `neighbors.py` | Embedding, dự báo điểm và tìm exact L1 neighbors |
| `prediction.py` | Phát hành point forecasts và predictive quantiles |
| `uncertainty.py` | Fit NB dispersion, quantiles, WIS và coverage |
| `snapshots.py` | Đọc history as-of và cập nhật outcomes cho calibration |
| `evaluate.py`, `evaluation_report.py` | Chấm điểm, tổng hợp và vẽ biểu đồ |
| `compare_models.py` | So sánh external models trên cùng tập targets khi cung cấp scores |

Các script chuẩn bị dữ liệu nằm ở [experiments/data](../data/README.md).
Tests nằm ở `tests/forecasting/`; hướng dẫn benchmark được gộp bên dưới.
Provenance gốc nằm cùng dữ liệu ở `outputs/truth_states/provenance.json`.

```bash
.venv/bin/python -m unittest discover -s tests/forecasting -v
```

## Chạy sau BO

```bash
.venv/bin/python experiments/forecasting/run_backtest.py \
  --selection-dir outputs/reproduction_full/02_optimization \
  --truth outputs/truth_states/California.csv --series California \
  --probabilistic --coverage-grid full --score-truth raw \
  --calibration-truth final --prehistory-mode frozen \
  --warmup-start-date 1900-01-01 --max-origins 0 \
  --output-dir outputs/prediction_new

.venv/bin/python experiments/forecasting/evaluate.py \
  --prediction-dir outputs/prediction_new \
  --truth outputs/truth_states/California.csv --output-dir outputs/evaluation_new
```

`--selection-dir` kiểm tra hash artifact, áp dụng `k`, `closest` và đúng training
library của BO winner. Không chỉ đọc JSON rồi tiếp tục dùng các default cũ.
Cũng có thể dùng `--synthetic`, `--k`, `--closest` tường minh để chạy ablation.

## Chuẩn bị dữ liệu

Truth: CSV một chuỗi, ngày `YYYY-MM-DD` cách nhau 7 ngày, cột
`target_end_date,value`. Chuyển bộ truth gốc bằng Python:

```bash
.venv/bin/python -m pip install pyreadr  # chỉ cần cho chuyển định dạng
.venv/bin/python experiments/data/prepare_state_truth.py \
  --input-dir /path/to/state_truths --output-dir outputs/truth_states_new
```

Đã có final truth local ở `outputs/truth_states/`. Core pipeline chỉ cần
NumPy/SciPy; evaluation thêm Matplotlib để vẽ (đều nằm trong `requirements.txt`
ở root), không dùng R. Không coi final truth là historical as-of.

Nếu có file nguồn `tdat_list_tot_weekly.csv`, chuyển bằng streaming để tránh nạp
file nhiều GB vào RAM:

```bash
.venv/bin/python experiments/data/prepare_asof.py \
  --input /path/to/tdat_list_tot_weekly.csv \
  --truth-manifest outputs/truth_states/manifest.csv \
  --output-dir outputs/asof_states
```

Mỗi archive có `as_of,target_end_date,value`; reader yêu cầu đúng vintage và
đủ tuần, không fallback sang final truth. Giữ nguyên nhãn `as_of` trong nguồn:
`get_data_as_of.R` tải với `as_of=origin+1` rồi gán nhãn origin, nên đây không
phải bằng chứng thông tin đã có vào cùng ngày origin. Adapter không tự đổi
nhãn để làm dữ liệu có vẻ causal. Archive đầy đủ mọi vintage chưa có; phần Figure 2 thật đã tải ở `outputs/paper_data/` (xem `experiments/reproduction/README.md`).

## Bước 3 — UQ calibration

`uncertainty.calibration_plan` chạy **trước point prediction** của origin hiện
tại; chỉ forecast cũ với target đã tới origin mới vào likelihood.

- 6 origins đầu: NB size `[20,10,6,0.3]`, mean `point+1`.
- Sau đó: fit inverse-size riêng từng horizon, bounds `(0,1]` với cận số học
  `1e-8` thay 0; size là nghịch đảo, mean của predictive NB là point.
- Giữ quy tắc nguồn: forecast cũ <100 thì mean trong likelihood được thay bằng
  outcome. Zero outcomes hợp lệ; không ép chúng thành 1 để fit NB.
- Negative outcomes không thuộc support NB: loại khỏi likelihood, ghi số lượng
  `excluded_negative` trong `calibration.csv`; vẫn dùng giá trị raw để chấm điểm.
  Đây là numerical repair có nhãn, không mô phỏng `-Inf`/lỗi của code R.

`--calibration-truth final` lấy outcome cuối cùng như `forecasts_data$true_values`
trong nguồn. Nó có thể chứa chỉnh sửa sau origin, chưa phải đánh giá causal.
`--calibration-truth history` lấy outcome đang có trong history/snapshot hiện
hành; đây là biến thể causal, không tự gọi là đúng protocol nguồn.

`--prehistory-mode frozen` dùng snapshot `--frozen-as-of-date` (mặc định
2020-08-15) cho warmup trước cutoff, theo ý tưởng helper gốc. Warmup sau cutoff
vẫn cần đúng vintage riêng. Code R còn thêm một forecast row ở chính origin
2020-08-15 rồi lại dự báo origin đó; Python không lặp row đó để tránh nhân đôi
trọng số calibration. `rolling` dùng history của từng origin, khác quy ước nguồn.

## Bước 4 — prediction

`prediction.predict_origin` không có argument actual/truth/future. Nó nhận
history, embedding và UQ plan đã fit:

1. Smooth history bằng thin plate regression spline/GCV, k=round(n/2); lấy k sai phân cuối. `--smoothing spline` là đối chứng legacy, không phải GAM.
2. Khoảng cách L1, chọn closest windows, median từng future-difference horizon.
3. Cộng dồn vào observation cuối, floor point ở 1.
4. Lấy 5.000 mẫu NB/horizon, dịch median về point theo nguồn. Không clip cận âm.

Embedding main runner dùng bridge X→y như `run_smoa.R`. Objective optimization
gốc dùng `diff(y)` không có bridge; cả hai hành vi được giữ riêng, có tests và
metadata, không sửa nhầm một phía để che sai khác trong chính nguồn.

## Bước 5 — evaluation

`evaluate.py` chỉ đọc issued `predictions.csv`, `quantiles.csv` rồi ghép raw final
truth. **Không floor truth về 1 khi chấm điểm.** History vẫn floor 1 như nguồn.
Zero/negative observations vẫn được giữ trong MAE/WIS; không biến chúng thành
case count khác để làm điểm số thay đổi âm thầm.

- WIS: `2 * mean(pinball loss)` trên 7 quantiles
  `[.025,.1,.25,.5,.75,.9,.975]`, cùng helper R.
- Coverage: full 23 quantiles `[.01,.025,.05,.10,...,.95,.975,.99]` → 11 central
  intervals: 98%,95%,90%,80%,...,10%. Điều kiện `lower < actual <= upper`.
- Sharpness, overprediction, underprediction; persistence baseline riêng.
- MAE/WIS theo horizon trong runner, và file score dài cho so sánh model.

### Xem biểu đồ và MAE/WIS từ lần chạy đã có

Không cần chạy lại synthetic, BO hoặc prediction. Ví dụ với California (thư mục `05` trong manifest 50 bang hiện tại), sau khi job full hoàn tất:

```bash
.venv/bin/python experiments/forecasting/evaluate.py \
  --prediction-dir outputs/reproduction_full/04_prediction/05 \
  --truth outputs/truth_states/California.csv \
  --output-dir outputs/reproduction_full/05_evaluation/05
```

Lệnh in bảng **MAE và WIS** của sMOA/persistence ra terminal, đồng thời tạo:

- `report.md`: bảng tổng hợp và theo horizon, kèm hình và protocol dữ liệu/BO.
- `forecast_vs_actual.png` / `.svg`: COVID thực tế (đen), sMOA (xanh), persistence
  (cam nét đứt); vùng 50%/95% là khoảng dự báo của sMOA.
- `metrics_by_horizon.png` / `.svg`: MAE/WIS so với persistence ở từng horizon.
- `summary.json`: trường `mae` tường minh (`abs_error` là alias),
  `by_horizon` và danh sách artifacts.

Mỗi ô đồ thị giữ một horizon cố định; các điểm nối tiếp thuộc các origin khác
nhau. Không nối lẫn dự báo 1–4 tuần thành một đường như thể cùng một lần phát
hành. Điểm tổng hợp lấy trung bình trên origin–horizon; một tuần target có
thể xuất hiện ở nhiều horizon. Cả MAE và WIS càng thấp càng tốt. WIS dùng đúng
7 quantiles của helper nguồn; không đổi thang điểm để vẽ biểu đồ.

Pipeline tự tạo báo cáo này ở bước 5. Nếu chỉ cần bảng điểm, thêm `--no-plots`
để không cần Matplotlib. Nếu prediction không có quantiles, vẫn vẽ dự báo
điểm nhưng WIS ghi `N/A`, không suy ra WIS từ MAE.

`--score-truth floored --coverage-grid standard` chỉ để đối chiếu convention cũ
của project. Evaluator độc lập luôn chấm raw truth; không dùng nó để xác nhận
bảng điểm legacy đã floor. Các output/snapshots cũ đã được dọn; không dùng điểm mini làm kết quả full.

## So sánh external models trên đúng các targets chung

Nếu có `scores_tot_w_wis_components.csv` của nguồn và bảng location:

```bash
.venv/bin/python experiments/data/prepare_reference_scores.py \
  --input /path/to/scores_tot_w_wis_components.csv \
  --locations /path/to/state_census.csv --output outputs/reference_scores.csv

.venv/bin/python experiments/forecasting/compare_models.py \
  --smoa-scores outputs/evaluation_new/scores.csv \
  --reference-scores outputs/reference_scores.csv \
  --output-dir outputs/model_comparison
```

Canonical reference schema:
`series,forecast_origin,target_end_date,horizon,model,abs_error,wis`.
R ghép theo location/target/horizon, không forecast_date; adapter suy origin
bằng target−7*h để biểu diễn cùng key và giữ submission_date gốc riêng.
Python không cho duplicate submissions cùng model/key tự nhân trọng số như một
many-to-many join; cần chọn submission rõ ràng. Không suy rằng inferred origin
là thời điểm thông tin thực tế được biết. Comparator báo số paired/unmatched,
NA/nonfinite bị loại; không so mean trên hai tập targets khác nhau. Cần kiểm
tra reference WIS dùng cùng quantile/scoring convention trước khi diễn giải.

Workspace chưa có score archive/COVIDHub repo tương ứng để chạy so sánh thực;
fixtures kiểm tra adapter/join không được báo thành kết quả COVIDHub.

## Output

- `predictions.csv`: point/baseline đã phát hành, không có truth hay score.
- `quantiles.csv`: 23 quantiles/model/horizon khi chọn full grid.
- `calibration.csv`: NB size/method/outcome count/negative exclusion theo origin.
- `coverage.csv`: 11 mức coverage/model/horizon.
- `matches.csv`: synthetic curve/window IDs và L1 distance.
- `warmup_forecasts.csv`: history calibration, không vào scored metrics.
- `forecasts.csv`: view kết hợp prediction/actual/score cho tương thích và xem IDE.
- `summary.json`, `metadata.json`: tham số, phiên bản, checksum và convention.
- Evaluation riêng có `report.md`, hình PNG/SVG và bảng MAE/WIS theo horizon.

Tổng hợp 50 bang: [Benchmark 50 bang](#benchmark-50-bang).
Bảng audit toàn project: [REPRODUCTION_AUDIT.md](../../REPRODUCTION_AUDIT.md).

Pipeline full mặc định chạy đủ 50 bang và mọi origin; tạo report tổng hợp ở `05_evaluation/report.md` cùng report/PNG/SVG riêng mỗi bang. Không có bảng metric full trước khi job thực sự hoàn tất.

## Benchmark 50 bang

Pipeline full tự chạy tất cả bang trong manifest sau khi BO đủ 104 evaluations,
không dùng bảng điểm mini/legacy. Nếu chỉ chạy benchmark sau một BO đã hoàn tất:

```bash
.venv/bin/python experiments/forecasting/run_benchmark.py \
  --manifest outputs/truth_states/manifest.csv \
  --selection-dir outputs/reproduction_full/02_optimization \
  --history-mode final --probabilistic --max-origins 0 \
  --warmup-start-date 1900-01-01 --prehistory-mode frozen \
  --calibration-truth final --score-truth raw --coverage-grid full \
  --negative-truth clip --output-dir outputs/full_benchmark
```

`--max-origins 0` lấy mọi origin đủ điều kiện, không giới hạn ở 8 origins.
Có thể dùng `--end-date` để chọn cùng mốc kết thúc giữa các bang; đây là cutoff
origin, nên phải có thêm 4 tuần target để chấm đủ horizon. Mặc định không cắt
bớt lịch từng bang. Warmup không vào MAE/WIS scored forecasts.

Embedding và exact L1 neighbor index được tạo một lần, chia sẻ giữa các bang;
UQ dispersion fit riêng theo bang/model/horizon. Không tune trên COVID test
scores. `--resume` kiểm tra config/input/code và tiếp tục từ bang chưa xong;
output bang đang dở được tái tính, bang đã hoàn tất được dùng lại.

### Manifest và dữ liệu

```csv
series,truth,as_of_truth
California,California.csv,
Texas,Texas.csv,
```

Đường dẫn tương đối tính từ manifest. Final truth chuẩn bị bằng
`prepare_state_truth.py` (Python, optional `pyreadr`; không chạy R).
Provenance/checksums: `outputs/truth_states/provenance.json` và bản theo dõi
trong chính metadata đầu vào, không giữ thêm bản sao ở folder forecasting.

Để dùng historical archive thật, điền `as_of_truth` mọi bang và chọn
`--history-mode as_of`. Reader yêu cầu đúng vintage và đủ tuần, không fallback
sang final truth. Adapter: `experiments/data/prepare_asof.py`. Phần archive
Figure 2 đã tải ở `outputs/paper_data/`; archive đủ mọi bang/ngày chưa có.
Final-truth backtest không được gọi là realtime evaluation.

### Điểm số và hình

- `series_metrics.csv`: MAE/WIS/coverage, output subdir và ratio so persistence từng bang.
- `horizon_metrics.csv`: điểm theo bang/horizon.
- `summary.json`: pooled score có trọng số theo số forecasts, mean/median ratio
  mỗi bang có trọng số bằng nhau, số bang tốt hơn baseline.
- `01/`, `02/`, ...: issued predictions, quantiles, calibration và matches;
  ánh xạ thư mục qua `series_metrics.csv`.

Bước 5 của `experiments/run_pipeline.py` gọi `evaluate_benchmark`, tạo
`05_evaluation/report.md`, CSV scores/coverage gộp và báo cáo/PNG/SVG riêng cho
**từng bang**. Mỗi hình tách horizon 1–4 tuần, vẽ actual + point forecasts +
50%/95% predictive intervals; MAE/WIS in trên báo cáo và terminal.

Chấm raw truth, giữ cả zero/negative revisions. Chỉ floor history ở 1 và loại
negative outcomes khỏi NB likelihood với số lượng ghi rõ. WIS dùng 7 quantiles
theo helper nguồn; full coverage dùng 23 quantiles. Persistence là diagnostic
baseline; external COVIDHub models cần score archive thật và paired comparator.

Job full **legacy final truth + spline** đã hoàn tất 104/104 BO evaluations và 50 bang: 22.468
forecasts/model, MAE 7.564,27 và WIS 4.765,92. Coverage 95% = 86,78%; chưa tái
hiện nominal calibration của paper. Chi tiết đối chiếu nằm trong Experiment 0;
các snapshots mini cũ đã bị dọn và không dùng làm kết quả mới.
