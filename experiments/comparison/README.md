# Experiment 1 — Comparative Evaluation under Limited Historical Data

**Đã triển khai nhóm chính theo thiết kế bên dưới.** Trạng thái lượt chạy tại `outputs/experiment_1/progress.json`; chỉ xem là hoàn tất khi `status=complete` và `primary_complete=true`. Holt/ARIMA cần statsmodels và được ghi `dependency_missing` khi chưa có package, không có accuracy estimates thay thế.

Mở [experiment_1.ipynb](experiment_1.ipynb) bằng kernel `.venv` sau khi runner hoàn tất. Notebook chỉ đọc tables/figures/database đã sinh. [Report](../../outputs/experiment_1/report.md) chứa kết quả thực tế; phần còn lại của README giữ specification đã chốt trước scoring.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/comparison/experiment_1.py
# Tiếp tục completed blocks, không tính lại forecasts đã lưu:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -u experiments/comparison/experiment_1.py --resume
.venv/bin/python -m unittest discover -s experiments/comparison -v
.venv/bin/python experiments/comparison/experiment_1.py --execute-notebook
```

Lượt hiện hành ghi console log tại `outputs/experiment_1/run.log`. Không cần cài thêm package cho nhóm chính. Folder chỉ có bốn file nguồn: README, runner/phân tích, tests và notebook.

Mục tiêu là kiểm tra sMOA có đáng dùng khi mới có ít quan sát của một dịch bệnh hay không, và lợi thế/hạn chế thay đổi ra sao khi tăng lượng history. E1 là phân tích bổ sung cho Python implementation, không phải một thí nghiệm đã có nguyên dạng trong paper.

## 1. Những lựa chọn chính

| Thành phần | Quyết định |
| --- | --- |
| Nhóm chính | Persistence, MA(3), Drift, SES, sMOA |
| Nhóm bổ sung | Holt additive linear trend và ARIMA nhỏ, chỉ tại T=16,24 |
| Early-stage E1.A | T=6,8,11 tuần kể từ reporting onset; origin tăng cùng T |
| Controlled availability E1.B | T=6,8,11,16,24, **giữ cùng origin và cùng targets**, chỉ thay phần history được cấp |
| Horizons | 1,2,3,4 tuần, cùng target dates cho mọi mô hình trong một case |
| sMOA | N=18.387 attempts/family, k=5, m=4.422; GAM/GCV; omit bridge; full mixed library |
| Training-bank replicates | Ba banks E2 hiện có, seeds 0/1/2; không ensemble ba forecasts |
| Dữ liệu test | Synthetic weekly-count curves mới, độc lập RNG; COVID historical vintages hiện có |
| Metrics | MAE, RMSE, bias, MASE hợp lệ, normalized MAE và runtime |
| UQ | Không thuộc E1; prediction intervals và WIS đã được đánh giá ở E6 |

Nhóm chính bao phủ bốn giả định đơn giản: giữ mức hiện tại, lấy mức trung bình gần đây, tiếp tục drift, và làm trơn mức bằng exponential smoothing. SES là trường hợp level-only liên quan ETS(A,N,N); không gọi nó là benchmark cho toàn bộ họ ETS. Không thêm seasonal ETS/SARIMA hoặc neural networks trong E1 này: T≤24 không cung cấp một chu kỳ năm của weekly series, còn pretraining neural models sẽ tạo thêm một bài toán về dữ liệu ngoài.

Persistence được giữ **trong benchmark so sánh này**, theo bản đề xuất E1. Không đưa lại Persistence vào full/E3/E4/E6, vốn chỉ phân tích sMOA.

## 2. Model specification: không lựa chọn sau khi xem test errors

Gọi `z[1:T]` là history duy nhất được cấp cho model sau preprocessing chung.

| Model | Quy tắc cố định | History cần để chạy; scope E1 |
| --- | --- | --- |
| Persistence | `forecast(h)=z[T]` | ≥1; chạy mọi T của E1 |
| MA(3) | `forecast(h)=mean(z[T-2:T])`, flat ở cả bốn horizons | ≥3; không chọn window bằng test |
| Drift | `forecast(h)=z[T]+h*(z[T]-z[1])/(T-1)` | ≥2; dùng cả phần history được phép |
| SES | `level[1]=z[1]`; `level[t]=alpha*z[t]+(1-alpha)*level[t-1]`; forecast bằng `level[T]` | Conditional SSE fit trên history, known initialization; chạy mọi T của E1 |
| sMOA | GAM fit trên **toàn bộ T values được cấp**, clip fitted tại 1; lấy 6 fitted levels cuối →5 differences →exact L1 top 4.422 →median future increments | ≥6; không cấp history trước window |

MA(3) là mean của ba quan sát cuối, không phải recursive moving average tự đưa predictions vào history. Drift ở đây là endpoint drift, không phải OLS linear regression hoặc local trend; không thay qua lại giữa các định nghĩa khi thấy kết quả.

SES chỉ ước lượng **alpha∈[0,1]**, giữ initial level là quan sát đầu. Tối thiểu hóa one-step conditional SSE trên các điểm 2..T; không dùng forecast outcomes. Quy tắc triển khai: grid alpha 0,0.01,…,1; refine các local minima bằng bounded scalar optimization, xét cả endpoints; ties theo tolerance tương đối 1e-10 chọn alpha nhỏ hơn. Chuẩn hóa SSE bằng một hằng số tính từ chính history để ổn định số học, không đổi minimizer. Constant histories trả về level đó và ghi `flat_objective`; alpha=1 hợp lệ, lúc đó SES trùng Persistence. Không gọi đây là SES với estimated initial state hoặc tuyên bố numerical parity với default statsmodels.

**Điểm cần sửa trong cách diễn giải sMOA:** matching chỉ có k=5 differences, nhưng query được tạo sau GAM trên T quan sát. T tăng vẫn có thể làm GAM và query thay đổi. Không thể nói sMOA luôn cho cùng kết quả chỉ vì vẫn lấy 6 điểm cuối. Phải lưu query vector và hash của history để kiểm chứng điều này.

Persistence/Drift và các công thức scaled errors dựa trên các định nghĩa chuẩn trong [FPP3 — toolbox](https://otexts.com/fpp3/toolbox.html); SES dùng recurrence và conditional fitting như mô tả trong [FPP3 — exponential smoothing](https://otexts.com/fpp3/exponential-smoothing.html). Các lựa chọn MA window, initialization và numerical search ở trên là quyết định cụ thể của E1.

### Nhánh bổ sung: Holt và ARIMA

Chỉ chạy trên **E1.B tại T=16 và T=24**. Đây là ngưỡng scope do thiết kế chọn để hạn chế một benchmark chỉ có 6–8 points; không phải định lý rằng các mô hình này không thể fit ở T<16. Các ô T=6,8,11 ghi `outside_secondary_scope`, không ghi là optimizer thất bại.

- **Holt:** additive level + additive linear trend, không seasonality, không damping, không automatic ETS search. Dùng implementation được kiểm chứng của statsmodels, `initialization_method='estimated'`, optimize trên history. Ghi initialization, optimizer, bounds, convergence và version.
- **ARIMA:** candidate set cố định `(0,1,0)`, `(1,1,0)`, `(0,1,1)`, đều có drift (`trend='t'` trong statsmodels ARIMA). Giữ d=1 ở mọi candidate; không thêm ADF/auto-differencing hoặc search p,q lớn trên chuỗi ngắn. ARIMA(0,1,0)+drift có thể trùng Drift; vẫn là lựa chọn hợp lệ trong candidate set.
- ARIMA chọn minimum **finite AICc** trong những candidates hội tụ hợp lệ, chỉ dùng history hiện có. Lưu mọi candidate status, order, parameters, effective nobs, degrees of freedom và AICc. Dùng `result.aicc`, không tự đếm thiếu variance/trend parameters. AICc ties chọn model ít parameters hơn, sau đó theo order trong danh sách trên.
- Cố định state-space ML fitting, stationarity/invertibility enforcement, optimizer L-BFGS, maxiter=200 và không random restarts. Mọi retry khác phải là amendment trước khi chạy lại cả cohort, không sửa riêng các cases khó.
- Fit invalid nếu exception, optimizer không hội tụ, parameters/likelihood/forecasts không finite, hoặc AICc không định nghĩa được khi cần selection. Warnings về covariance/boundary phải lưu riêng; không tự coi một forecast rất lớn nhưng finite là invalid chỉ vì nó sai lớn khi so với truth. Không dùng test error để quyết định giữ/bỏ fit.
- Nếu có valid candidates, chọn trong số đó; nếu không còn candidate hợp lệ, ghi `no_valid_candidate` và không có forecast. Không thay bằng Persistence rồi gắn nhãn ARIMA. Thiếu package ghi `dependency_missing`, không phải `fit_failed`.

API/trend conventions và AICc tham chiếu [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html), [ARIMAResults.aicc](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.aicc.html). Nhánh này là robustness analysis, không được quyết định lại mô hình chính sau khi xem outcomes.

## 3. Cùng thông tin mục tiêu, khác prior phải được công khai

Mọi mô hình được cấp đúng cùng history slice, không có quan sát trước window và sau origin. Smoothing chỉ thuộc sMOA; không đưa GAM-smoothed data cho baselines vì làm vậy sẽ thay định nghĩa của chúng.

Preprocessing chung:

1. Input là raw weekly counts từ đúng vintage; `z=max(raw,1)` cho **mọi model**, tương thích sMOA hiện tại. Lưu raw negatives/zeros và số values bị floor.
2. Forecast point của mọi model có lower bound 1, giống current sMOA. Lưu số forecasts bị clip; không đặt upper cap hoặc winsorize các errors lớn.
3. Scoring dùng truth thô: final real counts, synthetic generator output nguyên bản. Giữ zero/negative real corrections trong scoring; không áp floor 1 lên labels.
4. Không dùng log transform, rescale theo future peak, cộng 100 vào synthetic test, hoặc chọn smoothing/MA window từ forecast outcomes. Đây là count-forecast benchmark riêng, không sao chép phép `+100` của synthetic BO objective.

sMOA có **external synthetic prior**; baselines không được pretrain bằng bank đó. Công bằng ở đây là cùng quyền truy cập target history và future embargo, không phải mọi phương pháp có cùng prior. Chi phí tạo/tra cứu prior phải được báo riêng. Giữ đúng mixed bank của paper-parameter run; không chuyển sang seasonal-only vì kết quả E3, không thay bằng BO winner sau khi đã thấy test performance.

Real arm hiện chỉ có COVID-19. Nhiều bang và origins không biến nó thành nhiều bệnh độc lập; kết luận về bệnh mới khác cần dữ liệu bổ sung, không thể suy ra chỉ từ số lượng forecast rows.

## 4. E1.A — Early-stage forecasting

### 4.1 Định nghĩa reporting onset

`tau` là tuần đầu có raw weekly count **≥10**. Đây là ngưỡng reporting onset do E1 quy định, không phải ngày dịch bắt đầu sinh học. Không định nghĩa onset bằng phần trăm của future maximum, vị trí peak, hoặc threshold đã chọn theo accuracy.

Model chỉ nhận `y[tau:tau+T]`, T∈{6,8,11}; origin là `tau+T-1`, target là bốn tuần tiếp theo. Không cấp pre-onset weeks cho GAM, SES hoặc Drift.

**T thay đổi thì cả origin, targets và phase cũng thay đổi.** E1.A trả lời khả năng dự báo theo tuổi reporting episode; không dùng nó một mình để kết luận thêm history gây accuracy tốt hơn. E1.B xử lý câu hỏi giữ nguyên thời điểm dự báo.

### 4.2 Synthetic test: fresh RNG, giữ generator hiện tại

- Dùng nguyên generator của project. Chỉ nhận curves có `time_cadence='weekly'` và `scale='counts'` vào test chính. Monthly steps không được gọi là weeks; proportion curves không được floor toàn bộ thành 1 rồi tính như count forecasts.
- Ba test seeds 0/1/2, **100 eligible curves/family/test seed**: tổng 900 independent curves, cân bằng rollercoaster/wiggle/seasonal.
- Namespace mới `[20261104,310,test_seed,family_index,attempt_index]`. Không trùng BO train/validation/test hoặc scalar training seeds của E2. Audit phải kiểm tra disjoint RNG keys và exact duplicate content hashes, giữ nguồn của mọi rejected attempt. Numeric duplicates được flag và truy nguồn; distinct RNG draws vẫn có thể sinh cùng một curve nên không tự loại duplicate chỉ vì đã thấy output trùng.
- Budget cố định **800 attempts/family/test seed**, tổng 7.200 attempts; không redraw vì forecast sai. Trong mỗi family/test seed, lấy 100 curves đầu theo attempted index đáp ứng metadata, finite values, onset rule và có đủ T_max+4=28 weeks kể từ onset.
- Nếu một cell không đủ 100 curves, dừng trước scoring và ghi thiếu cohort; không âm thầm giảm quota, nới criteria hoặc tăng attempts sau khi đã xem errors.
- Chỉ kiểm tra metadata, data integrity và horizon availability để chọn cohort. Loại do thiếu follow-up length là điều kiện quan sát của cohort, cần báo; không loại theo phase, future magnitude hoặc model performance.
- Test seeds và training-bank seeds **crossed**, không ghép train-seed0 chỉ với test-seed0. Mỗi sMOA bank dự báo cùng toàn bộ 900 curves; baselines chạy một lần trên từng history.
- Chạy cả ba T trên cùng 900 curves. Dự kiến **2.700 history/origin cases**, 10.800 target instances cho mỗi model hoặc mỗi sMOA training bank.

Independent RNG không làm test thành out-of-distribution: test và training vẫn chia sẻ generator families. Synthetic arm kiểm tra hành vi trong cùng mô hình sinh với trajectories mới; real arm cần thiết để đánh giá transfer. Không tuyên bố thắng synthetic đồng nghĩa tốt hơn trên bệnh mới bất kỳ.

### 4.3 Real early-stage: dùng vintages đã có, không frozen future snapshot

Project đã có `outputs/epidemic_panel/` với genuine historical JHU vintages từ **04/04/2020**. Đây là archive mở rộng của project; không có nghĩa paper gốc đã cung cấp cùng giai đoạn trước tháng 8.

Đóng băng onset cho từng bang bằng **vintage đầu 04/04/2020**, tìm tuần đầu raw count ≥10 trong snapshot đó. Chỉ chấm các forecast origins **sau** snapshot đầu này. Như vậy selection metadata đã biết trước mọi forecast, không dùng future final revisions để đặt onset.

Tại origin của từng T, lấy đúng current vintage của bang, chỉ giữ các dates từ frozen onset tới origin. Không lấy values từ snapshot tháng 8 cho dự báo tháng 4; không lấy final truth làm input. Các revisions giữa vintages vẫn được giữ vì đó là thông tin thực có ở mỗi origin.

**Tập real đã kiểm tra inputs/targets trước scoring:**

| T | Bang có đủ vintage/target | Khoảng forecast origins |
| --- | ---: | --- |
| 6 | 50 | 11/04/2020–02/05/2020 |
| 8 | 50 | 25/04/2020–16/05/2020 |
| 11 | 50 | 16/05/2020–06/06/2020 |

Frozen onset nằm trong 07/03/2020–28/03/2020. Cùng 50 bang có đủ cả ba T: **150 history/origin cases, 600 target instances/model hoặc sMOA bank**. Nếu inputs thay đổi, recompute eligibility trước scoring; so sánh qua T trên common-state intersection, không đổi tập bang ngầm.

Origin là nhãn thứ Bảy; archive dùng Sunday cutoff và Monday issuance. Snapshot 04/04 có thông tin cutoff 05/04, trước các issuance của E1.A. Gọi đây là **historical-vintage backtest**, không phải một cuộc đánh giá prospective, hoặc benchmark hoàn toàn đồng nhất với ForecastHub/paper.

## 5. E1.B — Increasing historical availability, fixed-origin control

T∈{6,8,11,16,24}. Với mỗi fixed origin o, model nhận **last T observations ending at o**. Cùng case/horizon có cùng target và actual cho mọi T. Không dùng expanding prefix với origin cũng dịch chuyển rồi gọi là control cho riêng history length.

### Synthetic

Dùng đúng 900 test curves của E1.A; fixed origin mỗi curve là `tau+23` (24 observations kể từ reporting onset). Các T nhỏ hơn là trailing windows của cùng origin này. Không chọn origin quanh peak sau khi xem future curve.

**4.500 history cases, 18.000 target instances/model hoặc sMOA bank**. A và B có thể dùng cùng test curves, nên không gộp chúng thành hai tập independent evidence.

### Real

Dùng calendar E2/E4 đã đóng băng, lấy **common origins có ≥24 weekly records cho đủ cả 50 bang**. Tất cả năm T đều dùng cùng tập state/origin, không thêm early origins chỉ cho T nhỏ.

Kiểm tra hiện tại cho **37 origins, 04/07/2020–01/10/2022**, gồm 50 bang ×37 origins =1.850 fixed-origin cases. Năm T tạo **9.250 history cases, 37.000 target instances/model hoặc sMOA bank**.

Đây là controlled truncation của available calendar history, không gọi T=6 ở năm 2022 là “dịch mới bắt đầu”. T=24 có thể gồm pre-onset zero weeks ở những origins đầu; lưu reporting age và phase để giải thích, không loại những windows đó sau khi thấy errors. Calendar gốc lấy mẫu dày hơn đầu dịch, không phải weekly đều suốt ba năm; giữ nguyên cho tính so sánh với E2/E4.

Thêm một analysis theo từng T/horizon; secondary Holt/ARIMA chỉ có T=16,24 trên đúng common-origin cohort này. Không dùng scores của full-history E2/E3 thay cho truncated-history forecasts: GAM đầu vào khác. Có thể reuse raw inputs, banks và verified embeddings; chỉ reuse prediction nếu history hash, parameters, source và targets khớp hoàn toàn.

## 6. Metrics, missing fits và uncertainty của so sánh

### 6.1 Point metrics

- **MAE, RMSE**, signed bias `prediction−actual`; báo overall và từng horizon, riêng synthetic/real, A/B và T.
- **Normalized MAE** dùng `S=max(mean(last4 z),1)` tại origin. Trong B, S giống nhau qua mọi T vì T_min=6; hữu ích để đọc thay đổi history trên cùng scale. Synthetic còn báo theo family/test seed; real báo theo bang/phase/level, tránh một vài chuỗi scale lớn quyết định mọi nhận xét.
- **MASE:** `D_T=mean(abs(diff(z[1:T])))`, scaled loss=`abs(error)/D_T`. D chỉ dùng history được phép, cùng D cho mọi model trong case. Nếu D=0, ghi MASE undefined và báo n; case vẫn giữ trong MAE/RMSE/bias. Không thêm epsilon để ép định nghĩa MASE.
- MASE denominator thay đổi khi T thay đổi: không dùng riêng trend MASE qua T làm bằng chứng hiệu ứng history. Trong B, ưu tiên raw MAE/RMSE và normalized MAE với S cố định theo origin để so sánh qua T.
- Không dùng MAPE làm metric chính vì zeros/small counts. Không pool synthetic và real thành một raw MAE chung.

### 6.2 Ba banks không phải ba lần nhân sample size

Tính đầy đủ metrics riêng từng sMOA bank, sau đó báo **mean và sample SD của ba seed-specific scores**. Mean RMSE không phải pooled RMSE. Baselines deterministic có một score trên cùng cohort, không tạo ba bản sao rồi báo như ba independent runs.

Để paired với baseline, lấy **mean loss của ba sMOA banks tại từng target**, không lấy mean/median prediction để tạo ensemble. Bootstrap trên losses đó; giữ bảng riêng từng bank để thấy ranking có đổi theo seed không.

### 6.3 Failures và masks

Mỗi model ×case ghi attempted, status, fit diagnostics và runtime, kể cả không có predictions. Bốn horizons là một forecast block; nếu block không finite/không fit hợp lệ, ghi failed block, không lấy riêng horizon thuận lợi.

- Báo availability rate trên **toàn bộ planned cases**.
- Main five-model comparison dùng chung intersection của valid cases trong từng domain/arm/T, yêu cầu đủ cả ba sMOA banks, giữ tất cả dự báo của same case. Không lấy mean loss trên hai banks nếu bank còn lại thất bại. Báo rõ số bị loại và reason; không chỉ xuất bảng errors rồi giấu failures.
- Các đường history-length của B và paired change qua T dùng thêm **common valid intersection qua cả năm T**. Secondary so sánh T=16 với T=24 cũng dùng mask hợp lệ chung của hai T. Có thể báo bảng per-T successful-case như mô tả phụ, nhưng không coi thay đổi case mix là hiệu ứng tăng history.
- Secondary báo cả pairwise masks với sMOA và all-secondary-model intersection. Primary errors khi so với ARIMA phải được tính lại trên đúng ARIMA-valid mask, không so primary all-case MAE với ARIMA successful-case MAE.
- `outside_secondary_scope`, `dependency_missing`, `insufficient_history`, `fit_failed`, `no_valid_candidate` là các trạng thái khác nhau. T<16 của secondary không tính vào optimizer failure rate.
- Không đưa MAE giả hoặc impute Persistence cho missing forecast; không tự kết luận model có n thấp hơn là tốt hơn vì raw MAE nhỏ hơn.

### 6.4 Paired inference và “tương đương”

Bootstrap 1.000 draws, seed=20261104: nguyên **test curve** cho synthetic, nguyên **bang** cho real. Giữ mọi T/horizon, same-case model pairing và training-bank replicates trong cluster. Không resample từng forecast row độc lập. A/B dùng cùng curves/states nên không cộng sample sizes để tăng significance.

Report paired `delta_MAE = MAE_sMOA−MAE_baseline` cùng 95% percentile interval, theo T/horizon. Trong B, thêm paired change T=24 so với T=6 trong từng model để đo thay đổi history trên cùng origins. Real state bootstrap chưa mô hình hóa hết national-wave dependence; synthetic intervals không bao quát mọi possible generator families/seeds.

Đặt trước margin thực tiễn **±5% relative MAE**, là quy ước E1, không phải kết luận của paper. Chỉ dùng nhãn “tương đương thực tiễn trong E1” nếu **toàn bộ paired bootstrap 95% interval** của `(MAE_sMOA−MAE_baseline)/MAE_baseline` nằm trong [−0.05,+0.05]. Đây là descriptive interval criterion, không tuyên bố một formal TOST test. Nếu denominator MAE_baseline=0 thì relative measure undefined; báo absolute errors. CI chứa 0 nhưng rộng không đủ kết luận tương đương.

Mọi phân tầng là exploratory, không có adjustment cho nhiều so sánh. Không chọn một T/horizon/baseline thắng đẹp nhất rồi dùng như kết luận toàn cục.

## 7. Runtime và complexity versus performance

Phân biệt rõ ba loại chi phí của sMOA:

1. **Offline synthetic generation:** bank có sẵn không đồng nghĩa chi phí bằng 0. Dùng provenance/timing có ghi nhận, phân biệt original run với replay; thời gian chưa đo thì ghi unavailable.
2. **Embedding + neighbor-index preparation:** đo rebuild mới khi triển khai; cache reuse có thời gian load riêng. Không trộn cold build với online prediction rồi so với baseline chỉ tính fit.
3. **Online history preprocessing/GAM + search + aggregation:** đo cùng bốn horizons, một CPU worker, cùng thread limits. Baselines tính cả fit/selection và prediction; ARIMA tính tất cả candidate fits, kể cả những candidate không được chọn.

Báo total seconds, amortized seconds/case, peak memory và cold-start cost. Nếu dùng batch matching, label runtime là amortized theo batch; không gọi batch-time/32 là latency đo trực tiếp của mỗi case. Ghi software versions, hardware, worker/thread counts. Timing dưới độ phân giải đáng tin của clock phải được đánh dấu, không diễn giải chênh lệch vài microseconds thành một ranking chắc chắn.

Đọc error và cost cùng nhau. Nếu một baseline nằm trong vùng practical equivalence mà rẻ hơn nhiều, đó là bằng chứng hữu ích cho RQ4. Không gán một utility score tùy ý để trộn MAE, seconds và memory.

## 8. Quy mô và output khi triển khai

| Arm/domain | History cases | Targets/model hoặc mỗi sMOA bank |
| --- | ---: | ---: |
| A — synthetic | 2.700 | 10.800 |
| B — synthetic | 4.500 | 18.000 |
| A — real | 150 | 600 |
| B — real | 9.250 | 37.000 |
| Tổng | **16.600** | **66.400** |

Bốn baselines chạy một lần; sMOA chạy ba training banks trên cùng test. **Planned primary forecast records: 464.800**, không phải 464.800 independent test points. Secondary tại T=16,24 có tối đa thêm 44.000 records, valid count phụ thuộc fit status. Đây là ngân sách E1 tự quy định, không phải số targets hay BO evaluations trong paper.

Để giữ gọn khi triển khai, tối đa bốn source files trong folder này: README hiện tại, một `experiment_1.py` chứa runner/baselines/reporting, một test file và một notebook đọc kết quả. Không tạo module riêng cho mỗi baseline hoặc một script cho mỗi T.

Một output folder `outputs/experiment_1/` sẽ chứa:

- `results.sqlite`: curves/audit metadata, case histories hoặc offsets, as-of provenance, actuals, query vectors, model fit statuses, từng point forecast và timings; lưu dữ liệu chi tiết một lần.
- `protocol.json` và `progress.json`: fixed rules, source/input hashes, seed namespaces, software, run status và cuối cùng các checksums/verification.
- Các bảng nhỏ `scores.csv`, `paired_comparisons.csv`, `availability.csv`, `runtime.csv`, một `report.md` và tối đa sáu bộ hình: A early-stage accuracy; B fixed-origin history curves; horizon heatmaps; paired skill; availability; error–runtime trade-off.
- Không copy các training banks, không xuất thêm forecast CSV trùng SQLite, không tạo notebook rỗng hoặc nhiều bản RESULTS/report. Scratch embeddings được dọn sau từng bank worker.

Trước scoring, chốt protocol/hash và chọn test cohort chỉ bằng các quy tắc trên. Test cases không feedback cho BO, chọn generator, chọn family mix hoặc đổi baseline configurations. Không copy số liệu E2/E3 thành kết quả E1.

## 9. Kiểm tra bắt buộc và tiêu chí kết luận

Tests khi triển khai phải kiểm tra: đúng T values và embargo; onset không dùng future labels; synthetic keys disjoint và quotas; horizon/date alignment; Persistence/MA3/Drift trên series biết đáp án; SES constant history/alpha endpoints; GAM query chỉ dùng permitted history; omit-bridge parity; MASE zero denominator; failed-fit masks; averaging seed losses khác ensemble predictions; cluster bootstrap giữ pairing; database resume không đổi completed forecasts.

| Research question | Bằng chứng cần có |
| --- | --- |
| RQ1 — very short history | A tại T=6,8,11, so sánh paired với cả bốn low-data baselines; báo riêng synthetic/real |
| RQ2 — thêm history | B cùng origins/targets, T=6,8,11,16,24; không suy nhân quả riêng từ A |
| RQ3 — horizon | Metrics/paired differences riêng h=1,2,3,4 trong mỗi T |
| RQ4 — complexity | Practical-equivalence criterion, online cost và offline/index cost tách riêng |
| RQ5 — điều kiện thuận lợi/hạn chế | T/horizon, reporting age, phase/level; failures và seed variability, không chỉ một overall winner |

Hiện `.venv` có NumPy/SciPy/Pandas nhưng **chưa có statsmodels/patsy**. Primary SES được triển khai với SciPy theo specification ở trên; secondary cần dependency phù hợp. Không cài thêm package trong lượt triển khai này. Mọi con số ở bảng feasibility là số cases có dữ liệu, không phải forecast accuracy; đọc actual metrics từ report/CSV sau khi chạy hoàn tất.

## 10. Kết quả lượt chạy đầu

Nhóm chính hoàn tất **464.800 forecasts**, **16.600 cases**, **0 fit failures**; giữ toàn bộ common cohort. Synthetic test đạt đủ 900 curves từ 7.200 attempts, không có content hash trùng training hoặc trùng nhau. Lượt đầu mất **443,61 giây (~7 phút 24 giây)**, gồm fresh-test generation, preparation, baselines, ba sMOA banks và analysis, không gồm sinh training banks trước đây.

Real A, MAE trung bình qua ba sMOA banks:

| T | Persistence | MA(3) | Drift | SES | sMOA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 1.643,22 | 1.661,96 | 2.970,14 | **1.642,44** | 1.927,80 |
| 8 | 1.401,74 | 1.913,12 | 2.552,10 | 1.400,80 | **970,76** |
| 11 | 1.752,18 | 2.068,40 | 2.014,87 | 1.757,09 | **1.420,94** |

T6/T8 paired CI sMOA−Persistence còn chứa 0; T11 delta **−331,24**, CI **[−534,20; −122,96]**. Ở T8 raw MAE tốt hơn nhưng normalized MAE sMOA **0,4466**, SES **0,3379**: chưa thể coi sMOA thắng mọi metric hoặc mọi scale.

Real B, cùng 1.850 fixed origins/cases cho mọi T:

| T | Persistence MAE | sMOA MAE | sMOA normalized MAE |
| --- | ---: | ---: | ---: |
| 6 | 7.460,58 | 6.967,65 | **0,6111** |
| 8 | 7.460,58 | 6.819,98 | 0,6171 |
| 11 | 7.460,58 | **6.509,42** | 0,6185 |
| 16 | 7.460,58 | 6.589,34 | 0,6176 |
| 24 | 7.460,58 | 6.559,52 | 0,6215 |

sMOA có raw MAE thấp nhất trong nhóm chính ở cả năm T của real B; so Persistence giảm khoảng **6,61%–12,75%**. T24−T6 sMOA delta **−408,13**, CI **[−707,19; −187,26]**. Gain không đơn điệu; normalized MAE không cùng xu hướng với raw MAE. Persistence/MA3 đã kiểm tra predictions không đổi qua T, trong khi khoảng **99,16%** longer-history GAM queries khác T6 ngoài tolerance 1e-10.

Synthetic sMOA có raw MAE thấp nhất ở mọi T đã chạy, nhưng normalized losses có đuôi rất dài: riêng B T6, mean **251,70**, median **0,320**, top 1% losses đóng góp **98,18%** tổng normalized loss. Giữ toàn bộ cases và metric specification, không loại extreme errors để làm đẹp kết quả.

Không có overall comparison nào đạt interval criterion tương đương thực tiễn ±5%; điều đó không chứng minh mọi model khác nhau đáng kể. Runtime online sMOA ở real B khoảng **3,6–4,6 ms/case** (GAM charged + batched matching/aggregation), SES khoảng **0,10–0,23 ms/case**; preparation/index **12–14 giây/bank**, peak RSS khoảng **1,64–1,75 GB**. Các microsecond baselines còn rẻ hơn, nhưng timing quá nhỏ không đủ để xếp hạng chi tiết.

**Holt/ARIMA chưa được chấm accuracy**: mỗi model có 5.500 in-scope blocks `dependency_missing`; code adapters chưa được integration-test với statsmodels cài thật trong môi trường này. Không suy ra rằng chúng kém sMOA. Primary được kiểm tra bằng 11 tests; notebook đọc cùng database/tables và sáu bộ hình, không chạy lại models.
