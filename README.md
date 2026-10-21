# Infections Forcasting

Project Python tái hiện bước sinh synthetic data của **Synthetic Method of
Analogues for Emerging Infectious Disease Forecasting**.

Repo đích: [gnart-linhh/Infections_Forcasting](https://github.com/gnart-linhh/Infections_Forcasting).

## Cấu trúc

```text
experiments/synthetic/
  generator.py          # Mô hình SIR và ba nhóm đường cong
  generate_python.py    # CLI sinh dữ liệu, xuất CSV và metadata
  requirements.txt      # NumPy, SciPy
  reference/            # Thông tin nguồn và bản quyền
outputs/                # Kết quả local, được Git ignore
```

## Chạy

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/synthetic/requirements.txt
.venv/bin/python experiments/synthetic/generate_python.py \
  --num-curves 10 --output-dir outputs/synthetic_demo
```

Xem [hướng dẫn thí nghiệm](experiments/synthetic/README.md) để hiểu phương pháp,
seed và output. Pipeline chỉ cần Python và chạy độc lập với thư mục `precog/`.
Checkout `precog/` cũ được giữ local và Git ignore, không đưa vào repo mới.

## Bước 2: dự báo và đánh giá

Đã có pipeline Python đọc synthetic CSV để tạo cửa sổ analogues, dự báo 1–4 tuần
và backtest với baseline giữ nguyên giá trị tuần cuối:

```bash
.venv/bin/python experiments/forecasting/run_backtest.py \
  --synthetic outputs/synthetic_demo/curves.csv \
  --truth outputs/truth/California.csv \
  --max-origins 0 --output-dir outputs/backtest_california
```

Xem [hướng dẫn backtest](experiments/forecasting/README.md) để chuẩn bị truth CSV,
đọc kết quả và hiểu các khác biệt còn lại so với thí nghiệm gốc.


## Bước 3: khoảng dự báo và WIS

```bash
.venv/bin/python experiments/forecasting/run_backtest.py \
  --truth outputs/truth/California.csv --probabilistic \
  --max-origins 0 --seed 13 --output-dir outputs/backtest_california_wis
```

Học NB dispersion theo horizon từ các forecast đã có outcome, xuất quantile,
WIS và coverage. Chi tiết phương pháp, kết quả California và giới hạn tái hiện
ở [hướng dẫn bước 3](experiments/forecasting/README.md#bước-3--khoảng-dự-báo-online-dispersion-và-wis).

## Bước 4: benchmark 50 bang và historical snapshots

```bash
.venv/bin/python experiments/forecasting/run_benchmark.py \
  --manifest outputs/truth_states/manifest.csv --probabilistic \
  --warmup-start-date 2020-06-06 --end-date 2022-10-01 \
  --max-origins 0 --negative-truth clip \
  --output-dir outputs/benchmark_50_states_new
```

Xem [hướng dẫn benchmark](experiments/forecasting/BENCHMARK.md): manifest CSV,
chỉ số tổng hợp/theo bang, calibration warmup và chế độ đọc snapshot as-of.
Lượt benchmark hiện tại dùng final truth; chưa có archive as-of thật trong workspace.

## Dataset exploration

Mở [notebook exploration](experiments/exploration/dataset_exploration.ipynb) để
xem thống kê, kiểm tra chất lượng dữ liệu và biểu đồ cho synthetic + truth 50 bang.
Cách cài dependency, chạy lại và các file báo cáo: [hướng dẫn EDA](experiments/exploration/README.md).
