# Bước 1 — full synthetic generation bằng Python

Mặc định theo `run_smoa.R`: **18.387 attempts mỗi family × 3 = 55.161 attempts**.
Không có chế độ demo/preview mặc định.

```bash
.venv/bin/python -u experiments/synthetic/generate_python.py \
  --workers 4 --output-dir outputs/reproduction_full/01_synthetic
```

Dùng [entrypoint toàn pipeline](../../README.md) nếu muốn tiếp tục tự động sang
full BO → UQ → prediction → evaluation.

## Generator

1. `sir_rollercoaster`: cộng 3–7 sóng SIR I(t), reproduction
   `1.1+Beta(1,3)*20`, gamma `1/Uniform(1,10)`, Dirichlet initial states;
   cadence 10–52 không hoàn lại.
2. `sir_rollercoaster_wiggle`: nhiều sóng, thêm biến điệu sin.
3. `seasonal`: 5–15 sóng, lặp cách 52 bước, weekly/monthly cadence.
4. Beta observation noise, log-uniform peak .0005–.25, proportion/counts,
   final floor 1e-10. Monthly bins theo `floor((1:n)/4)` như R.

`source-structure-v2` dùng SciPy LSODA và NumPy RNG; không tương đương từng số
với R/deSolve. Invalid curve/`None` được bỏ và log như nguồn; không tự sinh bù.
Metadata phân biệt `attempted_total`, `accepted_total`, `accepted` mỗi family
và `failed`, không báo sai số lượng khi một attempt thất bại.

## Output và resume

- `curves.csv`: chỉ xuất hiện khi thư viện hoàn tất, grouped theo curve ID.
- `curves.csv.partial`, `checkpoint.json`: khi đang sinh; checkpoint mỗi 500 attempts.
- `metadata.json`: status, số lượng, seed, code/data hashes và phiên bản thư viện.
- `run.lock`: khóa process đang ghi.

```bash
.venv/bin/python -u experiments/synthetic/generate_python.py \
  --workers 4 --output-dir outputs/reproduction_full/01_synthetic --resume
```

Resume truncate phần CSV sau byte offset đã checkpoint rồi tiếp tục đúng seed,
không nhân đôi dòng. Worker processes không làm đổi thứ tự hay RNG theo curve.
Seed mặc định `curve_id=1..55161`, theo ý tưởng set.seed(i) của runner R; NumPy
không cho cùng số với R. `step` là bước cadence, không phải ngày lịch.

BO không cố định `num_curves=18387`: ở bước 2 mỗi candidate sinh bank riêng
10k–30k attempts/family. Prediction sử dụng bank của BO winner; thư viện bước
1 dùng để đối chiếu generation ở quy mô công bố và EDA, không giả làm winner.

Nguồn: LANL/precog SMOA, commit `aba1948d228e5380d72907048236601874b53cac`.
`reference/UPSTREAM_LICENSE` lưu BSD-3 License và thông tin bản quyền gốc.
