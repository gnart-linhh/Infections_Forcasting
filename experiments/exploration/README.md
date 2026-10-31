# Synthetic & Real-World Dataset Exploration

Mở [dataset_exploration.ipynb](dataset_exploration.ipynb), chọn kernel `.venv`, Run All.
Notebook trình bày cả hai phần, bảng kết quả và 12 biểu đồ (PNG/SVG). Dữ liệu đầu vào
chỉ được đọc; không chạy lại BO, prediction hoặc thay đổi kết quả forecasting.

## Chạy

```bash
.venv/bin/python experiments/exploration/explore_datasets.py
.venv/bin/python experiments/notebooks.py experiments/exploration/dataset_exploration.ipynb
.venv/bin/python -m unittest discover -s experiments/exploration -p 'test_*.py' -v
```

Script đầu tính lại toàn bộ EDA. Notebook mặc định đọc artifact hoàn chỉnh đã sinh;
đặt `REBUILD=True` để tính lại từ notebook. Đường dẫn inputs/output có thể chỉnh ở
cell đầu hoặc qua `--synthetic`, `--manifest`, `--archive-dir`, `--population`,
`--output-dir` của CLI. Sau khi đổi input, cần rebuild; notebook không tự xác minh
lại checksum 1 GB dữ liệu ở mỗi lần mở. `metadata.json` lưu checksum lần chạy.

## Synthetic Dataset Exploration

Đọc toàn bộ `outputs/reproduction_full/01_synthetic/curves.csv`, không lấy mẫu
để tính thống kê. Metadata generation phân biệt 55.161 attempts (18.387/family)
với 54.081 accepted curves: rollercoaster 18.387, wiggle 18.387, seasonal 17.307.
Không sinh bù các seasonal attempts bị loại.

- Số curve/quan sát; phân phối độ dài và magnitude (min/median/max/p95/peak).
- Số đỉnh sau median smoothing 3 bước, prominence 10% curve range, minimum
  distance 4 bước. Sensitivity 5%/20%, raw peak count và boundary maxima được lưu.
- Volatility: mean(abs(diff))/mean; CV và SD(diff)/mean; ECDF và boxplots.
- So sánh ba họ, tách scale counts/proportion và cadence weekly/monthly.
- Số embedding windows hợp lệ với k/h cấu hình (mặc định 5/4), keep/omit bridge.

Đỉnh phát hiện trên observations không bằng số thành phần SIR: noise, rounding
và các waves chồng lấn có thể làm thay đổi số đỉnh. Peak counts phụ thuộc độ dài;
có thêm density/100 original steps. Monthly steps không phải weekly steps.
Chỉ các hình minh họa shape dùng ba ví dụ đầu mỗi family/scale/cadence và chia
cho peak riêng; mọi thống kê dùng toàn bộ curves.

## Real-World Dataset Exploration

Final truth: `outputs/truth_states/manifest.csv`, 50 bang, chuyển từ LANL
`state_truths`. So sánh bang/khu vực dùng lịch giao nhau 2020-01-25–2022-10-29;
phần Arizona dài hơn vẫn giữ trong bảng `final_all_records.csv`.

- Phân phối weekly counts, zero, negative revisions, quantiles; case rates/100k.
- Tăng/giảm mạnh: hai tuần liên tiếp, |delta| ≥1.000 và |delta/previous| ≥50%,
  previous >0/current ≥0. Zero→positive ghi riêng, không tự thêm epsilon.
- So sánh 50 bang và bốn Census regions. Region rate = sum(cases)/sum(population).
- Khoảng low/high theo counts (0–100 / ≥10.000) và rate (0–1 / ≥100 per100k).
  Runs ≥3 tuần được đánh dấu sustained; negative revisions không coi là low.
- As-of/final: ghép chính xác state/target; signed revision = final − as-of.
  Lag 0 dùng mỗi state/target một lần. Lag 0–12 dùng cohort có đủ 13 vintages.
  Có absolute difference, signed difference, tỷ lệ thay đổi và symmetric
  relative difference 2|final−asof|/(|final|+|asof|), both-zero→0.

As-of archive: `outputs/paper_weekly_archive`, 112 weekly vintages từ JHU county
snapshots, origin labels 2020-08-15–2022-10-01. Label Saturday, cutoff Sunday,
issuance Monday theo nguồn. Final và as-of có upstream/aggregation khác nhau;
không quy toàn bộ discrepancies thành reporting delay. Các ví dụ revision lớn
được chọn hậu nghiệm và ghi rõ, không coi là đại diện toàn bộ dữ liệu.

Population/regions: bản CSV [state_census_2019.csv](reference/state_census_2019.csv)
được lưu kèm [upstream BSD license](reference/UPSTREAM_LICENSE), từ
[LANL state_census.csv](https://github.com/lanl/precog/blob/main/smoa/data/state_census.csv).
Dùng POPESTIMATE2019 cố định, không phải dân số từng năm. Không cần checkout
`precog/`, R hoặc tải dữ liệu mới để chạy EDA trên các inputs hiện có.

## Kết quả và tổ chức code

Kết quả ở `outputs/dataset_exploration/`:

- `report.md`: liên kết hai báo cáo có hình.
- `synthetic/{curves,groups,examples}.csv`, `synthetic/report.md`, `synthetic/figures/`.
- `real_world/`: common weekly counts, state/region summaries, sharp changes,
  low/high periods, epidemic peaks, matched revision pairs và balanced lag tables;
  `report.md` và `figures/`.
- `summary.json`, `metadata.json`, `progress.json`: số liệu tổng, input/code SHA256,
  phiên bản và trạng thái hoàn thành.

`synthetic_eda.py` xử lý streaming library; `real_world_eda.py` xử lý weekly data
và vintages; `explore_datasets.py` điều phối, vẽ hình, ghi báo cáo. Notebook dùng
cùng artifact và cùng định nghĩa. [kết quả và nhận xét](#results) ghi nhận xét lần chạy.

EDA hồi cứu giữ raw values, không tự loại zero/negative hoặc thay dữ liệu bằng
normalization. Đây không phải forecast evaluation; không dùng full-data
statistics để chọn hyperparameters trên tập test.

<a id="results"></a>

## Kết quả khảo sát dữ liệu

Artifact: `outputs/dataset_exploration/`, hai báo cáo có hình tại
`synthetic/report.md` và `real_world/report.md`. Notebook
[dataset_exploration.ipynb](dataset_exploration.ipynb) chứa bảng và hình đã chạy.
Các con số sau mô tả bộ input hiện có, không khẳng định Python generator tạo
cùng từng observation với R hoặc rằng synthetic giống hoàn toàn COVID thật.

### Synthetic Dataset Exploration

54.081 accepted curves / 15.510.156 observations từ 55.161 attempts:
rollercoaster 18.387, wiggle 18.387, seasonal 17.307. Seasonal có 1.080 attempts
bị loại theo generator, không phải thiếu file hoặc chỉ lấy sample để EDA.

So sánh **cùng weekly/counts**, sau median-3 smoothing, prominence 10%, distance 4:

| Họ | Curves | Median độ dài (bước tuần) | Median số đỉnh | Median mean(abs(diff))/mean |
| --- | ---: | ---: | ---: | ---: |
| Rollercoaster | 9.114 | 263 | 4 | 0,20433 |
| Wiggle | 9.259 | 262 | 4 | 0,20790 |
| Seasonal | 4.390 | 441 | 8 | 0,23420 |

- Rollercoaster và wiggle có phân phối độ dài/số đỉnh gần nhau. Wiggle có median
  biến động cao hơn khoảng 1,75%, nhưng không nên diễn giải là tăng số waves hoặc
  hiệu ứng nhân quả: đây không phải các đường latent được ghép cặp.
- Seasonal dài hơn, nhiều đỉnh lặp lại hơn. Số đỉnh tuyệt đối chịu ảnh hưởng độ
  dài; xem thêm peak density/100 steps và sensitivity prominence 5%/20%.
- Seasonal monthly/counts có median 111 bước, 7 đỉnh, volatility 0,51441; không
  so trực tiếp mức này với weekly volatility vì khác khoảng thời gian mỗi bước.
- Counts/proportion có scale rất khác: median peak weekly/counts khoảng
  205–211 nghìn, weekly/proportion khoảng 0,011–0,012. Không gộp raw magnitudes.
- Tổng windows k=5/h=4: keep bridge 15.023.012, omit bridge 14.968.931, khớp số
  windows của thư viện embedding full. Windows không phải số curves.

### Real-World Dataset Exploration

Giữ đủ 7.267 source records. So sánh dùng 7.250 state-weeks (145 tuần × 50 bang),
2020-01-25–2022-10-29; Arizona có 17 tuần thêm chỉ nằm ở bảng source.

- Phân phối lệch phải: median 4.772, mean 13.268,06 cases/state-week; p95
  50.118,85, max 852.280 (California, 2022-01-15). Có 324 zero và 8 negative
  revisions, được giữ nguyên thay vì coi là missing hoặc clip về zero.
- 591 tăng/giảm mạnh thỏa đồng thời |delta| ≥1.000 và |delta/previous| ≥50%.
  Đây là thay đổi số liệu báo cáo; backlog/revisions có thể góp phần tạo spikes.
- Regional peak rates khác thời điểm và độ cao: Northeast 2.163,55/100k
  (2022-01-08), Midwest 1.515,17 (2022-01-15), South 1.652,63 (2022-01-15),
  West 1.847,23 (2022-01-22). Dùng tổng cases/tổng population 2019.
- Median state rates dao động từ Hawaii 61,02 tới Kentucky 177,94 cases/100k/tuần
  trên cùng lịch. Raw case-count ranking có thể bị chi phối bởi population.
- Ví dụ high-rate run dài: South Carolina ≥100 cases/100k/tuần trong 46 tuần,
  2020-06-20–2021-05-01; Vermont 45 tuần, 2021-08-14–2022-06-18. Tất cả low/high
  runs, cả ngắn lẫn dài, có ngày bắt đầu/kết thúc trong `low_high_periods.csv`.
  Low-rate ví dụ: West Virginia và South Dakota ≤1/100k trong 9 tuần,
  2020-01-25–2020-03-21. Số ca báo cáo thấp đầu kỳ không chứng minh không có
  lây nhiễm. Ngưỡng mô tả; counts và rate runs có thể trùng
  nhau nên không cộng chúng thành số đợt dịch độc lập.

### As-of và final

478.800 state/target/vintage pairs từ 112 vintages. Lag 0 gồm 5.600 state/target
pairs độc nhất: 18,77% khác final, mean signed(final−asof) +133,69 cases,
mean absolute discrepancy 433,94, median 0, p90 311.

Median 0 nhưng mean lớn cho thấy phần lớn pairs giống nhau, còn một số trường
hợp có revisions rất lớn. Ví dụ Tennessee tuần 2022-02-05: as-of 155.239, final
71.531 (−83.708); Colorado tuần 2022-01-08: 60.134 →127.085 (+66.951).
Các trường hợp này chọn hậu nghiệm để kiểm tra, không phải mẫu đại diện.

Balanced cohort giữ **cùng 5.000 state/target pairs tại mọi lag 0–12**:

| Lag (tuần) | Mean absolute discrepancy | Mean symmetric relative difference |
| --- | ---: | ---: |
| 0 | 447,78 cases | 2,347% |
| 4 | 303,97 cases | 1,417% |
| 8 | 251,33 cases | 1,206% |
| 12 | 212,86 cases | 1,018% |

Mean absolute discrepancy giảm khoảng 52,46% từ lag 0 tới 12, nhưng vẫn khác
zero. As-of dựng từ JHU county snapshots; final từ LANL state_truths. Vì có thể
khác upstream/aggregation, đường này mô tả chênh lệch dữ liệu, không đo thuần
reporting delay. Không dùng nó làm forecast accuracy hoặc leakage-free features.

Kiểm tra tính đúng của EDA: 8 tests bao gồm peak scale invariance, count/step
validation, zero/negative/gaps, low/high runs, common calendar/population mapping,
matched joins và balanced revision cohorts. Checksum inputs/code lưu ở metadata.
