# Phạm vi tái hiện sMOA bằng Python

Nguồn: [LANL/precog](https://github.com/lanl/precog/tree/main/smoa), commit
`aba1948d228e5380d72907048236601874b53cac`. Figure 2 được đối chiếu thêm với
commit trước publication `2018f2e` (2025-05-22). Python không gọi R hay phụ
thuộc checkout `precog/` khi chạy.

## Kết quả hiện hành

[Experiment 0](experiments/reproduction/README.md) dùng
`outputs/experiment_0_source/`: 4 bang, 6 origins cố định, horizons 1–4 tuần.
Dữ liệu nằm ở `outputs/paper_data/`, gồm snapshot JHU lịch sử của 4 vintage
và forecast chính thức baseline/4-week ensemble ở các ngày có incident cases.
URL, commit, SHA và quy tắc Saturday+1 retrieval nằm trong provenance.

- Library Python: 18.387 attempts/family × 3 = 55.161 attempts; 54.081 accepted.
- Tham số Figure 2 cố định theo publication: num_curves=18.387, k=5, closest=4.422.
- Smoother: thin plate regression spline, rank round(n/2), curvature penalty, GCV.
- Figure 2 giữ embedding bỏ bridge và NB initialization [20,10,6,6].
- UQ `figure2_retrospective` dùng final outcomes của mọi forecast đã phát trước
  đó, kể cả target chưa tới, như source figure script. Không gọi đây là causal.
- Đối chứng `smoa_matured_uq` giữ cùng points và chỉ fit target đã tới hạn;
  frozen historical snapshot vẫn khác một real-time evaluation đầu năm 2020.
- MAE/WIS trên 96 targets: 10.570,35 / 7.431,19. So với COVIDHub chỉ dùng giao
  targets có dữ liệu; giao cả ba models có 16 targets, không suy ranking toàn kỳ.
- Hình 3 hàng model × 4 cột bang, chung y-scale, bands 50%/95%; giữ đủ target
  cuối 2020-09-05. Hình paper tham chiếu giữ trong output hiện hành `reference/`.

Notebook lưu kết quả thật; report công bố cả final/historical × spline/GAM
ablation. Không chọn biến thể theo lỗi COVID để gọi là phương pháp nguồn.

## Run full legacy được giữ để đối chiếu

`outputs/reproduction_full/` đã hoàn tất 104 BO evaluations, 50 bang,
22.468 forecasts/model, dùng **final truth + spline cũ**. Winner:
num_curves=29.357, k=7, closest=5.167; MAE 7.564,27, WIS 4.765,92;
coverage 95% là 86,78%. Các điểm này không phải kết quả historical/GAM mới.

Run full chứa library đang dùng, checkpoint, dữ liệu dự báo và metadata kiểm
chứng nên được giữ. Source hashes cũ không bị sửa để khớp workspace hiện tại.
BO và forecasting mặc định mới dùng GAM; không resume run legacy bằng source
mới hoặc ghép BO spline với prediction GAM. Chạy lại full cần thư mục output
mới. Figures 3–5 và ranking/calibration full chưa được chạy lại với protocol mới.

## Quy ước thuật toán còn được giữ

- Generator source-structure-v2 dùng LSODA và NumPy RNG; ghi accepted/failed,
  không sinh bù. Seed train/validation/heldout tách vai trò.
- BO: num_curves 10k–30k, k2–7, closest1k–6k; 99 LHS + 5 EI; mọi validation
  origin đủ điều kiện. Mỗi trial resample train/validation; shuffle/cap 50.
  Objective là MAE; independent test chỉ chạy sau chọn best observed trial.
- Optimization/Figure 2 bỏ bridge; main prediction có bridge theo runner nguồn.
- Default forecasting UQ dùng matured targets, main NB sizes [20,10,6,0.3].
  Giữ forecast<100 likelihood rule; zero outcomes hợp lệ. Negative labels bị
  loại khỏi NB likelihood và ghi count, nhưng giữ raw values để chấm điểm.
- WIS dùng 7 quantiles [.025,.1,.25,.5,.75,.9,.975]. Full coverage dùng 23
  quantiles / 11 intervals, điều kiện lower < actual <= upper. Cặp .01/.99 là
  98%, dù prose nguồn có nói 99%.
- Exact L1 neighbors, stable ties; atomic checkpoints, source/input hashes và
  process locks. Prediction API không nhận future truth; UQ policy có nhãn riêng.

## Những phần chưa xác nhận tương đương

| Phần | Giới hạn |
| --- | --- |
| RNG / ODE | NumPy/LSODA khác R/deSolve; synthetic curves không trùng từng số |
| GAM | Cùng thin plate/GCV về mô hình toán; chưa có fitted-value fixtures mgcv để kiểm numerical parity |
| BO | Python Gaussian/RBF GP và EI khác hetGP; 104 trials với GAM mới chưa chạy lại |
| Historical data | Có phần Figure 2; chưa có archive 5,5 GB cho mọi bang/vintage |
| External models | Có baseline/ensemble Figure 2; chưa có archive scores toàn bộ models |
| Original forecasts | Chưa có vector sMOA R gốc; không thể lượng hóa phần trăm giống hay xác nhận full-paper metrics |

`publication_reproduction_complete` vẫn false. Full ngân sách tính toán hoặc
hình tương tự chưa đủ để chứng minh numerical reproduction của toàn paper.
