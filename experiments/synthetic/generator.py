from __future__ import annotations #Đảm bảo các annotation kiểu dữ liệu (type hints) được xử lý dưới dạng chuỗi (string), giúp tương thích tốt hơn với Python cũ và tránh lỗi khi tham chiếu lớp chưa định nghĩa.

from dataclasses import dataclass 

import numpy as np
from scipy.integrate import solve_ivp #để giải hệ phương trình vi phân thường (ODE) của mô hình SIR.


@dataclass
class SyntheticCurve:
    ts: np.ndarray # Mảng chứa chuỗi thời gian của các ca nhiễm
    curve_type: str # Loại đường cong (ví dụ: seasonal, sir_rollercoaster)
    time_cadence: str # Đơn vị thời gian (hàng tuần - weekly, hàng tháng - monthly)
    scale: str # Thang đo dữ liệu (tỷ lệ - proportion, hoặc số ca tuyệt đối - counts)
    seasonal: bool # Có tính chu kỳ mùa vụ hay không


def sir(beta: float, gamma: float, s0: float, i0: float, r0: float, times: np.ndarray) -> dict[str, np.ndarray]:
    """Solve the SIR system used by the original R helper."""

    #Định nghĩa hệ phương trình vi phân SIR
    def equations(_time: float, y: np.ndarray) -> list[float]:
        s, i, r = y
        return [-beta * i * s, beta * i * s - gamma * i, gamma * i]

    #giải hệ ODE từ thời điểm đầu đến cuối, lấy giá trị tại các mốc times
    sol = solve_ivp(
        equations,
        (float(times[0]), float(times[-1])),
        [s0, i0, r0],
        t_eval=times.astype(float),
        rtol=1e-6, #độ chính xác tương đối
        atol=1e-9, #độ chính xác tuyệt đối 
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    s, i, r = sol.y #Lấy kết quả S, I, R.
    incidence = np.concatenate([[np.nan], -np.diff(s)]) #incidence (số ca mới) ≈ −ΔS (sự giảm của nhóm dễ nhiễm).
    return {"S": s, "I": i, "R": r, "incidence": incidence}

#Chọn thứ tự ngẫu nhiên có trọng số (không lặp lại). Dùng để sắp xếp các đợt sóng theo kích thước hoặc khoảng cách một cách “ngẫu nhiên có thiên hướng”.
def _weighted_choice_order(rng: np.random.Generator, n: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float) #chuyển weight thành mảng numpy số thực
    weights = weights / weights.sum() #chuyển trọng số thành xác suất
    return rng.choice(np.arange(n), size=n, replace=False, p=weights)

#Cắt bỏ phần đầu và cuối của chuỗi thời gian nếu giá trị quá nhỏ (dưới ngưỡng), chỉ giữ lại phần có tín hiệu + một chút padding.
def _trim_small_tails(ts: np.ndarray, threshold: float = 1e-3, pad: int = 10) -> np.ndarray:
    idx = np.flatnonzero(ts > threshold) #tìm và trả về một mảng chứa chỉ số (vị trí) của tất cả các phần tử trong ts lớn hơn mức threshold
    if idx.size == 0:
        return ts
    start = max(0, idx[0] - pad)
    end = min(len(ts), idx[-1] + pad)
    return ts[start:end]

#Sinh một sóng dịch đơn lẻ từ mô hình SIR.
def _single_sir_wave(rng: np.random.Generator, reproduction: float | None = None) -> tuple[np.ndarray, dict[str, float]]:
    reproduction = float(reproduction if reproduction is not None else 1.1 + rng.beta(1, 3) * 20) #Số tái sản xuất cơ bản R₀. Nếu không cho sẵn thì lấy ngẫu nhiên (thường từ khoảng 1.1 đến ~21)
    gamma = 1.0 / rng.uniform(1, 10) #tốc độ hồi phục (thời gian nhiễm từ 1–10 đơn vị thời gian).
    s0, i0, r0 = rng.dirichlet([1000, 0.1, 0.1]) #đảm bảo tổng 3 giá trị này luôn bằng 1. Quần thể S chiếm đa số (trọng số 1000), còn I và R rất nhỏ (0.1)
    beta = (reproduction / s0) * gamma
    times = np.arange(1, 501)

    for _ in range(5):
        out = sir(beta, gamma, s0, i0, r0, times)
        wave = np.maximum(np.asarray(out["I"], dtype=float), 0.0)
        if np.all(np.isfinite(wave)):
            break
        reproduction = float(1.1 + rng.beta(1, 3) * 20)
        gamma = 1.0 / rng.uniform(1, 10)
        s0, i0, r0 = rng.dirichlet([1000, 0.1, 0.1])
        beta = (reproduction / s0) * gamma
    else:
        wave = np.repeat(0.01, len(times))
        out = {"incidence": np.zeros_like(wave)}

    scaled = wave / max(float(np.nanmax(wave)), 1e-12)
    active = np.flatnonzero(scaled > 1e-4)
    if active.size:
        wave = wave[: active[-1] + 1]

    incidence = np.asarray(out["incidence"], dtype=float)
    incidence[0] = 0.0
    params = {
        "pit": float(np.nanargmax(incidence)), #Peak Incidence Time
        "piv": float(np.nanmax(incidence)), #Peak Incidence Value
        "s0": float(s0),
        "i0": float(i0),
        "r0": float(r0),
    }
    return wave, params


def gen_curve(curve_type: str, rng: np.random.Generator | None = None) -> SyntheticCurve | None:
    """Generate one synthetic curve following the R `gen_curve` structure."""

    rng = rng or np.random.default_rng()
    if curve_type not in {"sir_rollercoaster", "sir_rollercoaster_wiggle", "seasonal"}:
        raise ValueError(f"unsupported curve_type: {curve_type}")

    seasonal = curve_type == "seasonal"

    if curve_type in {"sir_rollercoaster", "sir_rollercoaster_wiggle"}:
        nwaves = int(rng.integers(3, 8)) #sinh 3-7 giá trị R0 khác nhau
        reproduction = 1.1 + rng.beta(1, 3, size=nwaves) * 20
        #Sắp xếp thứ tự các sóng theo R₀ tăng dần, giảm dần hoặc ngẫu nhiên.
        order_type = rng.choice(["increasing", "decreasing", "random"])
        if order_type == "increasing":
            reproduction = reproduction[_weighted_choice_order(rng, nwaves, reproduction)[::-1]]
        elif order_type == "decreasing":
            reproduction = reproduction[_weighted_choice_order(rng, nwaves, reproduction)]
        
        #Khoảng cách giữa các sóng (10–52 tuần), ưu tiên khoảng cách nhỏ hơn một chút.
        cadence = rng.integers(10, 53, size=nwaves)
        cadence = cadence[_weighted_choice_order(rng, nwaves, cadence.max() + 1 - cadence)]
        
        #Sinh từng sóng, thêm phần “đệm” nhỏ ở đầu/cuối, và dịch chuyển theo thời gian.
        start = 1
        waves: list[np.ndarray] = []
        for jj in range(nwaves):
            wave, _params = _single_sir_wave(rng, float(reproduction[jj]))
            prefix = rng.uniform(0, np.quantile(wave, 0.01) + 1e-9, size=20)
            suffix = rng.uniform(0, np.quantile(wave, 0.01) + 1e-9, size=20)
            waves.append(np.concatenate([np.zeros(start), prefix, np.nan_to_num(wave), suffix]))
            start += int(cadence[jj])

        #Cộng tất cả sóng lại (chồng lên nhau) rồi chuẩn hóa về [0, 1] và cắt đuôi nhỏ.
        length = max(len(x) for x in waves)
        ts = np.zeros(length)
        for wave in waves:
            ts[: len(wave)] += wave
        ts = ts / max(float(ts.max()), 1e-12)
        ts = _trim_small_tails(ts)

        #Thêm hiệu ứng “lắc lư” bằng cách nhân với một sóng sin biến đổi biên độ.
        if curve_type == "sir_rollercoaster_wiggle":
            mult_squish = rng.uniform(1, 2)
            mult_period = len(ts) / rng.uniform(np.sqrt(2), np.sqrt(10)) ** 2
            sign = rng.choice([-1, 1])
            mult_sin = 1 + mult_squish * (1 + sign * np.sin((np.pi * np.arange(1, len(ts) + 1)) / mult_period))
            ts = np.maximum(0, mult_sin * ts)
            ts = ts / max(float(ts.max()), 1e-12)

        #Thêm phần nền nhỏ ở đầu và cuối. Đơn vị thời gian luôn là weekly.
        ts = np.concatenate(
            [
                rng.uniform(0, np.quantile(ts, 0.011) + 1e-9, size=20),
                ts,
                rng.uniform(0, np.quantile(ts, 0.011) + 1e-9, size=20),
            ]
        )
        time_cadence = "weekly"

    #Sinh 5–15 sóng giống nhau (cùng hình dạng), chỉ khác kích thước.
    else:
        nwaves = int(rng.integers(5, 16))
        wave, _params = _single_sir_wave(rng, float(1.2 + rng.beta(1, 3) * 10))
        
        #70% trường hợp có xu hướng tăng/giảm dần kích thước các sóng theo năm, 30% giữ nguyên kích thước.
        trend = rng.choice([1, 2], p=[0.7, 0.3])
        if trend == 1:
            z = rng.normal(0, 0.5) * np.arange(1, nwaves + 1)
            prop_scale = (np.exp(z) / (1 + np.exp(z)))
            prop_scale = prop_scale / prop_scale[0]
        else:
            prop_scale = np.ones(nwaves)

        #Đặt các sóng cách nhau đúng 52 tuần (1 năm), cắt bỏ năm đầu và nửa năm cuối.
        period = max(52, len(wave))
        ts = np.zeros(nwaves * period)
        for jj in range(nwaves):
            start = 52 * jj
            ts[start : start + len(wave)] += prop_scale[jj] * wave
        ts = ts[: int(52 * nwaves)]
        if ts.max() > 1:
            ts = ts / ts.max()
        ts = ts[52:-26]

        #Ngẫu nhiên chọn weekly hoặc monthly. Nếu monthly thì gộp 4 tuần thành 1 tháng.
        time_cadence = rng.choice(["monthly", "weekly"])
        if time_cadence == "monthly":
            n_months = int(np.ceil(len(ts) / 4))
            ts = np.array([ts[i * 4 : (i + 1) * 4].sum() for i in range(n_months)])
            if ts.max() > 1:
                ts = ts / ts.max()

    #Thêm nhiễu quan sát bằng phân phối Beta (mô phỏng tỷ lệ quan sát được với độ tin cậy alpha)
    ts = np.maximum(ts, 1e-9)
    alpha = float(np.exp(rng.uniform(np.log(5e1), np.log(1e4))))
    # Keep beta parameters strictly positive even when ts is close to one.
    obs = rng.beta(np.maximum(alpha * ts, 1e-9), np.maximum(alpha * (1 - np.minimum(ts, 1 - 1e-9)), 1e-9))
 
    #Chuẩn hóa lại để đỉnh của đường cong nằm trong khoảng 0.05% – 25%.
    peak_incidence = float(np.exp(rng.uniform(np.log(0.0005), np.log(0.25))))
    obs = (obs / max(float(obs.max()), 1e-12)) * peak_incidence

    #Ngẫu nhiên chọn:
    #proportion: giữ nguyên tỷ lệ (0–1)
    #counts: nhân với dân số (từ ~200 nghìn đến 100 triệu) rồi làm tròn thành số nguyên.
    scale = rng.choice(["proportion", "counts"])
    if scale == "counts":
        population = int(round(rng.uniform(np.sqrt(2e5), np.sqrt(1e8)) ** 2))
        obs = np.round(population * obs)

    #Nếu có giá trị không hợp lệ thì trả về None
    if not np.all(np.isfinite(obs)):
        return None

    #Nếu đỉnh xuất hiện quá sớm (< 30 bước) thì thêm một đoạn nền nhỏ ở đầu để đường cong trông tự nhiên hơn.
    if int(np.argmax(obs)) < 30 and not seasonal:
        prefix_len = int(rng.integers(10, 18))
        obs = np.concatenate([rng.uniform(0, np.quantile(obs, 0.05) + 1e-9, size=prefix_len), obs])

    return SyntheticCurve(
        ts=np.maximum(obs.astype(float), 1e-10),
        curve_type=curve_type,
        time_cadence=str(time_cadence),
        scale=str(scale),
        seasonal=seasonal,
    )

