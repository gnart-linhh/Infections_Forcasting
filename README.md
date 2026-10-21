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
