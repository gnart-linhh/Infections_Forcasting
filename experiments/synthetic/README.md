# Thí nghiệm 01 — sinh synthetic data bằng Python

Pipeline chạy độc lập, không cần R, SLURM, COVID truth hoặc repo clone `precog/`.

## Cài đặt và chạy

Chạy từ thư mục gốc project:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r experiments/synthetic/requirements.txt
.venv/bin/python experiments/synthetic/generate_python.py \
  --num-curves 10 --output-dir outputs/synthetic_demo
```

`--num-curves` là số lần sinh mỗi nhóm, tổng ba nhóm. Cấu hình số lượng trong
code bài báo là 18.387 mỗi nhóm; nên kiểm tra với số nhỏ trước.
Thư mục output phải trống để tránh ghi đè.

## Output và tái lập

- `curves.csv`: ID, seed, nhóm, cadence, scale, seasonal, step và value.
- `metadata.json`: cấu hình, số mẫu thành công/lỗi, phiên bản Python/NumPy/SciPy
  và checksum module sinh dữ liệu.

Mỗi mẫu dùng RNG riêng, seed bằng `seed_offset + curve_id`. ID bắt đầu từ 1;
nhóm xếp liên tiếp. Đổi số mẫu mỗi nhóm làm đổi seed của nhóm sau.
`step` không phải ngày lịch. Seasonal có thể có cadence weekly hoặc monthly.
Không sinh bù mẫu trả về không hợp lệ.

Đây là bản chuyển phương pháp sang Python, chưa được xác nhận tương đương số học
với kết quả tác giả: RNG, ODE solver, thứ tự lấy mẫu và một số chi tiết
padding/aggregation khác code gốc. Cùng seed cho kết quả tái lập trong cùng
môi trường Python, không có nghĩa khớp từng số với R.

## Phương pháp trong code gốc

Nguồn: `precog/smoa/R/run_smoa.R` (khối Generate Synthetic List for Training)
và `precog/smoa/R/smoa_helpers.R` (`sir`, `gen_curve`).

1. `sir_rollercoaster`: cộng 3–7 sóng SIR, lấy quỹ đạo **I(t)**; tham số reproduction
   `1.1 + Beta(1,3)*20`, gamma nghịch đảo Uniform(1,10), trạng thái ban đầu
   Dirichlet(1000,0.1,0.1); khoảng cách sóng lấy từ 10–52.
2. `sir_rollercoaster_wiggle`: cùng ý tưởng nhiều sóng, thêm biến điệu sin.
3. `seasonal`: lặp sóng SIR cách 52 bước, 5–15 sóng, có thể có xu hướng biên độ;
   ngẫu nhiên chọn cadence weekly/monthly.
4. Thêm beta noise, đổi đỉnh theo log-uniform từ 0.0005 tới 0.25; chọn proportion
   hoặc counts (nhân dân số); chặn giá trị nhỏ nhất ở `1e-10`.

Lưu ý I(t) là số đang nhiễm trong mô hình; incidence trong helper còn được lưu
để lấy thông tin đỉnh, không phải chuỗi chính được cộng thành synthetic curve.
`gen_curve_lhs` có trong helper nhưng không được gọi trong khối sinh dữ liệu của
`run_smoa.R`, nên pipeline này dùng `gen_curve`.

## Nguồn và bản quyền

Tham khảo [LANL/precog SMOA](https://github.com/lanl/precog/tree/main/smoa),
commit local tham chiếu `aba1948d228e5380d72907048236601874b53cac`.
Các đường dẫn R ở phần mô tả trên là vị trí trong nguồn tác giả, không phải
dependency của project này. `generator.py` tách từ Python mini-port local đã có;
giữ phần sinh đường cong, bỏ các hàm forecasting/scoring chưa cần trong bước 1.
`reference/UPSTREAM_README.md` lưu thông tin bản quyền và BSD-3 License gốc.
