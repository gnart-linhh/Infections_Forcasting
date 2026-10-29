# Bước 2 — full Bayesian optimization

Mặc định chạy **99 Latin hypercube + 5 Expected Improvement = 104 evaluations**.
Không có profile `small`. Miền: `num_curves=10000..30000` mỗi family,
`k=2..7`, `closest=1000..6000`; 3 families, h=4.

```bash
.venv/bin/python -u experiments/optimization/run_optimization.py \
  --workers 4 --seed 13 \
  --output-dir outputs/reproduction_gam_full/02_optimization
```

Lệnh trên chạy riêng bước BO. Thường dùng entrypoint [toàn pipeline](../../README.md)
để tự chuyển selected bank/k/closest sang UQ/prediction/evaluation.

## Smoother đã sửa

Mặc định `--smoothing gam`: thin plate regression spline, k=round(n/2),
curvature penalty và GCV, cùng mô hình toán với `mgcv::gam` trong objective R.
`--smoothing spline` chỉ dành cho đối chiếu legacy; không coi spline cũ là GAM.
Cache/source hashes chứa `smoothing.py`; không resume 104 trials legacy bằng
source mới. Run cũ đã hoàn tất ngân sách nhưng dùng smoother khác; winner đó
không được gắn nhãn là BO đã chạy với GAM.

## Objective và protocol

Mỗi trial sinh independent training bank (30.000–90.000 attempts) và validation
`floor(num_curves/1000)` mỗi family; shuffle, chỉ giữ tối đa 50 accepted curves.
Đánh giá **mọi origin đủ điều kiện**, không lấy 12 origins như bản trước.
Validation cộng 100 theo nguồn. Score trả về là MAE trung bình trên các
forecasts, không phải WIS. UQ trong objective R không ảnh hưởng Score nên không
thực hiện các phép NB/WIS dư thừa đó.

- Embedding `upstream`: `diff(X)` và `diff(y)` bỏ bridge, như `optimize_smoa.R`.
- Runner prediction có bridge như `run_smoa.R`; không gộp hai convention khác nhau.
- Seed namespaces tách role train/validation/heldout và trial/family/index.
- Test độc lập role 2 chỉ sinh sau khi chọn best observed configuration;
  mặc định `floor(best_num_curves/1000)` mỗi family, shuffle/cap 50.
- GP Gaussian/RBF homoscedastic và EI minimization dùng NumPy/SciPy;
  không khẳng định khớp từng số với R/hetGP.

Nguồn đối chiếu, commit `aba1948d228e5380d72907048236601874b53cac`:
[optimize_smoa.R](https://github.com/lanl/precog/blob/aba1948d228e5380d72907048236601874b53cac/smoa/R/optimize_smoa.R),
[parallel_bayesian_optimization.R](https://github.com/lanl/precog/blob/aba1948d228e5380d72907048236601874b53cac/smoa/R/parallel_bayesian_optimization.R).

## Runtime và resume

Curve generation dùng independent processes, giữ thứ tự/seed dù đổi worker.
Objectives LHS xử lý tuần tự để giới hạn RAM; 5 vòng EI vẫn tuần tự. Tìm neighbor
bằng exact L1 cKDTree, `eps=0`, xử lý ties bằng source row index như stable sort;
không dùng approximate neighbors hoặc giảm số window.

```bash
.venv/bin/python -u experiments/optimization/run_optimization.py \
  --workers 4 --seed 13 \
  --output-dir outputs/reproduction_gam_full/02_optimization --resume
```

- `checkpoint.json`: completed history và RNG state của BO, ghi atomic sau mỗi trial.
- `working/`: chỉ bank train/validation của trial hiện tại; bank sinh xong có thể dùng lại.
  Nếu ngắt giữa lúc sinh bank thì trial đó sinh lại cùng seed, không bỏ trial.
- `best_checkpoint/`: chỉ bank của best trial hiện tại, giúp không mất winner khi bị ngắt.
- `audit/*.jsonl.gz`: seed/accepted log chi tiết trên đĩa; không tích hàng triệu dict vào RAM.
- `progress.json`: trial, phase và số attempts/origins đã xử lý.
- `run.lock`: tránh chạy hai process cùng ghi một output.

Resume kiểm tra cấu hình và checksum code. Không dùng lại một library khác
phiên bản chỉ vì đường dẫn vẫn giống. Không giữ bank của toàn bộ 104 trials
trên đĩa; chỉ giữ bank đang chạy, bank tốt nhất và audit đầy đủ.

## Kết quả

Sau khi đủ evaluations và heldout test:

- `trials.csv/.json`: 104 dòng nếu dùng cấu hình full; đủ phase, params, MAE và EI diagnostics.
- `best_params.json`: **best observed** configuration, không phải proposal chưa evaluate.
- `best_library/curves.csv`: đúng bank của trial được chọn, không sinh bank khác thay thế.
- `heldout_forecasts.csv`, `summary.json`, `generation_audit.json`.
- `metadata.json`: `status=complete`, `full_search_budget=true`, artifact/source hashes.

Không khẳng định BO chạy lại sẽ trả `num_curves=18387,k=5,closest=4422` như
constants trong runner R: RNG, smoother và surrogate numerical khác nhau.

Advanced flags cho nghiên cứu (`--embedding-mode pipeline`,
`--validation-protocol fixed`, các range/budget overrides) vẫn có nhãn tường minh.
Miền/budget khác gốc ghi `profile=custom`, `full_search_budget=false`.
Chúng không phải cấu hình mặc định full. Tests dùng fixtures nhỏ, không tạo
kết quả thực nghiệm production.

```bash
.venv/bin/python -m unittest discover -s experiments/optimization -v
```
