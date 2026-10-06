# Kẻ mạo danh - Olympic AI PTIT 2026

Bộ thực hành này hướng dẫn cách giải bài **Kẻ mạo danh (The Impostor)**: khám phá dữ liệu, xây dựng mô hình, so sánh các phương án và tạo tệp nộp bài.

Mỗi mẫu dữ liệu là một cặp gồm hai bức ảnh chân dung (`image_0` và `image_1`), trong đó có đúng một ảnh thật và một ảnh giả mạo. Mô hình cần so sánh các đặc trưng thị giác và dấu vết kỹ thuật số giữa hai bức ảnh để dự đoán nhãn vị trí của ảnh giả: `fake_position` thuộc tập {0, 1}.

Tài liệu chi tiết: [Mô tả bài toán](docs/challenge.md), [Kiến trúc pipeline và mã nguồn](docs/pipeline.md), [Nguồn của kết quả đã chạy](docs/execution.md), [sơ đồ tổng quan](../assets/README.md).

## Bắt đầu từ đâu?

Chọn notebook theo mục đích của bạn:

| Bạn muốn... | Bắt đầu ở đây |
| --- | --- |
| Học cách giải từ đầu | Đọc **8 notebook chính, từ 00 đến 07**, bắt đầu với [bài toán và dữ liệu](notebooks/00_problem_and_data.ipynb). Mỗi bài giải thích vì sao thử một ý tưởng, cách viết code và kết quả thu được. |
| Chạy lại từ dữ liệu đến bài nộp | Mở [pipeline_end_to_end](notebooks/pipeline_end_to_end.ipynb) sau khi cài môi trường và chuẩn bị dữ liệu. Các bước chạy được gom trong một notebook. |
| Tìm hiểu thêm một thử nghiệm | Chọn phụ lục A-D theo chủ đề quan tâm. Phần này là tùy chọn, bạn không cần đọc hết để theo các bài chính. |

Nếu mới bắt đầu, bạn nên đi theo ba chặng học tập:

- **Chặng 1 (00-02):** Hiểu bài toán, khám phá dữ liệu và xây dựng baseline với Logistic Regression.
- **Chặng 2 (03-04):** Huấn luyện mạng CNN pretrained và so sánh các phương pháp đưa ảnh vào mạng (native crop và resampled).
- **Chặng 3 (05-07):** Sàng lọc backbone, phân tích lỗi, đánh giá điều kiện blend mô hình và xuất bài nộp.

Các notebook đã lưu bảng kết quả và hình để bạn đọc trước khi tự chạy. Khi chạy lần lượt các bài, giữ cùng một `KMD_RUN_ID` để bài sau tìm được kết quả của bài trước.

## Cấu trúc thư mục

```text
KeMaoDanh/
├── README.md                 # Hướng dẫn của riêng bài Kẻ mạo danh
├── notebooks/                # 8 bài chính, 4 phụ lục tùy chọn và 1 pipeline tổng
├── src/kmd/                  # Mã nguồn được các notebook import trực tiếp
├── configs/                  # Cấu hình mô hình và development split cố định
├── scripts/                  # Chuẩn bị dữ liệu và đánh giá dự đoán
├── docs/                     # Mô tả bài toán, pipeline và nguồn kết quả
├── data/                     # Dữ liệu cục bộ, không đưa lên Git
├── artifacts/models/         # Checkpoint theo từng lượt chạy, không đưa lên Git
├── outputs/                  # Dự đoán và submission, không đưa lên Git
├── pyproject.toml
└── uv.lock
```

Toàn bộ hướng dẫn cài đặt và chạy bên dưới đều bắt đầu từ thư mục `KeMaoDanh/`. Khi clone repo, bạn cần giữ nguyên cấu trúc thư mục này để notebook tìm thấy gói `kmd`, configs và các file hỗ trợ.

---

## 1. Cài môi trường với uv

Dùng **Python 3.11** và `uv` để cài các phiên bản thư viện trong `uv.lock`:

### Bước 1: Cài đặt uv (nếu máy chưa có)
```bash
# Cài đặt uv trên Linux / macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Hoặc trên Windows (PowerShell):
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Bước 2: Clone repository và đồng bộ thư viện
```bash
git clone https://github.com/Purin1410/Olympic_AI_PTIT_2026_preliminary_round.git
cd Olympic_AI_PTIT_2026_preliminary_round/KeMaoDanh

# Đối với máy có GPU NVIDIA (hỗ trợ CUDA 12.8):
uv sync --locked --extra cu128

# Hoặc đối với máy chỉ có CPU (phục vụ EDA, Logistic Regression và kiểm tra mã nguồn):
# uv sync --locked --extra cpu
```

### Bước 3: Khởi động Jupyter Lab
```bash
# Đối với môi trường GPU:
uv run --locked --extra cu128 jupyter lab

# Đối với môi trường CPU:
# uv run --locked --extra cpu jupyter lab
```

*Lưu ý:* Khi chạy lần đầu tiên các kiến trúc tích chập (DenseNet-121, EfficientNet-B0, EfficientNet-B2, ResNet-18), hệ thống sẽ tự động tải các trọng số ImageNet pretrained từ kho lưu trữ chính thức của torchvision về thư mục cache của máy.

---

## 2. Chuẩn bị dữ liệu

Với tệp `the_imposter.zip` của lớp, chạy lệnh sau để giải nén và kiểm tra dữ liệu:

```bash
# Đối với môi trường GPU:
uv run --locked --extra cu128 python scripts/prepare_official_dataset.py --zip-path the_imposter.zip --dest-dir data

# Đối với môi trường CPU:
# uv run --locked --extra cpu python scripts/prepare_official_dataset.py --zip-path the_imposter.zip --dest-dir data
```

Script tìm các thư mục train/test kể cả khi ZIP có nhiều lớp `data/`, giữ nguyên byte ảnh JPEG và kiểm tra manifest sau khi giải nén.

Nếu lưu trữ dữ liệu tại thư mục ngoài dự án, bạn có thể thiết lập các biến môi trường:
```bash
export DATA_ROOT=/duong/dan/den/data/train
export TEST_ROOT=/duong/dan/den/data/private_test/private_test
```

*Lưu ý:* Nếu không cung cấp tập test, toàn bộ quá trình huấn luyện và đánh giá trên 800 mẫu phát triển vẫn hoàn thành đầy đủ và xuất thông báo `no submission produced (test root not provided)`. Dữ liệu và checkpoint được giữ ngoài Git thông qua `.gitignore`.

---

## 3. Chọn notebook để học hoặc chạy

### Chuỗi 8 notebook chính: 00-07

Các bài nối từ quan sát dữ liệu đến đặt giả thuyết, thử nghiệm và đọc kết quả để quyết định bước tiếp theo.

| STT | Notebook | Nội dung trọng tâm | Đầu vào | Đầu ra chính |
|:---:|---|---|---|---|
| **00** | [00_problem_and_data](notebooks/00_problem_and_data.ipynb) | Bản chất bài toán cặp ảnh, bảo toàn chuỗi `pair_id` có số 0 đầu, phân chia 800 dev và 200 holdout, Macro-F1 so với Accuracy | Train `pairs.csv` | Biểu đồ phân bố và phân tích ví dụ số nhỏ |
| **01** | [01_eda_and_baseline](notebooks/01_eda_and_baseline.ipynb) | Khám phá dữ liệu trên train Fold 0, quy tắc chọn tệp dung lượng nhẹ hơn, ma trận nhầm lẫn, nhận diện phản ví dụ | Train Fold 0 | Phân tích phân bố bytes và ma trận nhầm lẫn |
| **02** | [02_features_and_lr](notebooks/02_features_and_lr.ipynb) | 32 đặc trưng thống kê, vector hiệu $x = f_1 - f_0$, tính đối xứng $p(-x)=1-p(x)$, bảng 9 probe LR trên 3-fold OOF | Train images | Biểu đồ 9 probe LR và các cặp dự đoán đúng/sai |
| **03** | [03_pretrained_and_finetune](notebooks/03_pretrained_and_finetune.ipynb) | Phản ví dụ bố trí không gian, DenseNet-121 + Head, BatchNorm eval, warmup, trực quan `train_step`, Frozen vs Full fine-tune | Train Fold 0 | Checkpoint và bảng đánh giá đối chứng Fold 0 |
| **04** | [04_native_and_resampling](notebooks/04_native_and_resampling.ipynb) | Cùng vùng nhìn 4 crop (224x224) khác thao tác điểm ảnh: Native 1:1 vs Resampled (co 112 rồi phóng 224), lấy trung bình 4 logit cho mỗi ảnh | Train Fold 0 | Đo lường định lượng ảnh hưởng của phép nội suy |
| **05** | [05_backbone_and_selection](notebooks/05_backbone_and_selection.ipynb) | Mốc đối chứng `center60_cap48`, sàng lọc 6 cấu hình, shortlist 2 challenger, xác nhận qua 3 seed, chọn mô hình đơn winner động | Train 3 Folds | Bảng sàng lọc, shortlist và `single_selection.json` |
| **06** | [06_errors_and_blend](notebooks/06_errors_and_blend.ipynb) | Ma trận 4 nhóm đúng/sai giữa winner và native, thử blend 50/50, cổng Selection Gate +0.005 qua 3 seed, bỏ self-blend nếu winner là native | 3 Seeds x 3 Folds | Bảng blend và `submission_decision.json` |
| **07** | [07_private_and_submission](notebooks/07_private_and_submission.ipynb) | Khóa quyết định vào `review_freeze.json`, đối chiếu private nếu có, hai chiến lược: `fold_ensemble` (mặc định) hoặc `refit_all` (1.000 cặp) | Models + Test | Bảng đối chiếu private và tệp `submission.csv` |

### Chạy lại bằng notebook tổng hợp

Notebook [pipeline_end_to_end](notebooks/pipeline_end_to_end.ipynb) chạy các bước sàng lọc mô hình, xác nhận qua seed, thử blend, chốt lựa chọn và xuất submission khi có tập test. Bạn có thể chọn chạy notebook này thay cho việc thực thi lần lượt các bài chính. Nếu đã chạy xong 00-07, bạn không cần chạy thêm bản tổng hợp.

### 4 phụ lục mở rộng (tùy chọn)

Bốn phụ lục mở rộng các thử nghiệm: **A** về biểu diễn ảnh và fine-tuning, **B** về hàm mục tiêu và ghép cặp dữ liệu, **C** về lượng dữ liệu, **D** về tăng cường dữ liệu (augmentation). Bạn có thể chọn từng phần để đọc thêm; các bài chính không yêu cầu chạy phụ lục.

<details>
<summary>Xem nội dung 4 phụ lục</summary>

| Phụ lục | Notebook | Nội dung trọng tâm | Đầu vào | Đầu ra chính |
|:---:|---|---|---|---|
| **A** | [extension_a_representations](notebooks/extension_a_representations.ipynb) | Frozen Embedding + LR, Partial vs Full fine-tuning, Top-2 vs Mean pooling, ResNet-18 trên ảnh RGB vs Gaussian/NPR residual | Train 3 Folds | Bảng đối chứng mở rộng về biểu diễn và kiến trúc |
| **B** | [extension_b_objectives_and_repair](notebooks/extension_b_objectives_and_repair.ipynb) | So ba loss (image, pairwise, mixed) và thử re-pairing với mixed loss, cùng 19 epoch trên DenseNet-121 3-fold OOF | Train 3 Folds | Bảng so sánh hàm mục tiêu và kỹ thuật ghép cặp lại |
| **C** | [extension_c_data_scaling](notebooks/extension_c_data_scaling.ipynb) | Khảo sát quy mô dữ liệu 25%, 50%, 100% train dưới cùng ngân sách cố định 391 bước cập nhật trên EfficientNet-B2 | Các tập con train | Macro-F1 theo lượng dữ liệu |
| **D** | [extension_d_resize_augmentation](notebooks/extension_d_resize_augmentation.ipynb) | Tăng cường dữ liệu co giãn ngẫu nhiên Resize Augmentation 90-100% trên EfficientNet-B2 (đối chứng b2 chuẩn vs b2_resize) | Train 3 Folds | Bảng đánh giá ảnh hưởng của phép co giãn ngẫu nhiên |

</details>

Nếu muốn đưa cả kết quả phụ lục vào bảng đối chiếu private ở bài 07, hãy chạy các phụ lục trước bước khóa quyết định và đặt `KMD_INCLUDE_EXTENSIONS=1`. Nếu chỉ chạy các bài chính, giữ mặc định `0`.

---

## 4. Cách chọn mô hình và nộp bài

Bài 05 so kích thước ảnh, vùng nhìn và backbone với mốc `center60_cap48`: DenseNet-121, center60, 224px, trần 48 epoch, microbatch 8 và batch hiệu dụng 24. Hai ứng viên dẫn đầu được xác nhận cùng ba mốc trên các seed `20260917`, `20260918`, `20260919`; chọn mô hình đơn theo Macro-F1 trung bình, hòa thì theo thứ tự chữ cái.

Bài 06 thử trung bình xác suất 50/50 giữa mô hình đã chọn và `native`. Chỉ dùng blend khi mức tăng Macro-F1 trung bình chưa làm tròn qua ba seed đạt ít nhất **+0.0050** (0,50 điểm phần trăm). Nếu mô hình được chọn đã là `native`, giữ mô hình đó.

Bài 07 lưu quyết định trước khi xem nhãn private. Mặc định `fold_ensemble` dùng trung bình ba mô hình fold. `refit_all` huấn luyện lại trên 1.000 cặp có nhãn với số epoch lấy từ development; 200 cặp từng giữ riêng lúc này cũng tham gia train. Với blend, refit từng nhánh rồi lấy trung bình xác suất. Seed nộp bài là `20260917`.

Xem [các cấu hình đối chứng và luồng chọn mô hình](docs/pipeline.md) để đọc chi tiết. Các bảng development dùng OOF; bảng test dùng ensemble fold hoặc refit theo chiến lược đã chọn.

---

## 5. `from kmd...` lấy mã ở đâu?

Thư viện `kmd` nằm ngay trong repo, tại [src/kmd](src/kmd). Lệnh `uv sync` cài gói này vào môi trường của bài; không cần lấy thêm một repo riêng. Notebook cũng tìm `src/` từ thư mục `KeMaoDanh` để có thể đọc mã trực tiếp. Giữ nguyên cả thư mục khi clone, không chỉ sao chép notebook riêng.

Đọc [core.py](src/kmd/core.py) cho CSV, metric và submission; [models.py](src/kmd/models.py) cho điểm số và hàm mất mát; [training.py](src/kmd/training.py) cho vòng huấn luyện; [pipeline.py](src/kmd/pipeline.py) cho train theo fold và suy diễn; [decision_flow.py](src/kmd/decision_flow.py) cho sàng lọc và cổng blend; [finalization.py](src/kmd/finalization.py) cho bước khóa và refit. Notebook giữ các phép tính chính ở gần phần giải thích, còn những bước lặp lại dùng các hàm chung này.

---

## 6. Đặt dữ liệu ngoài repo và chạy trên Kaggle

Nếu dữ liệu nằm cạnh repo, đặt `DATA_ROOT` và `TEST_ROOT` thành đường dẫn tuyệt đối tới đúng thư mục chứa `pairs.csv` trước khi mở Jupyter.

Ví dụ thiết lập đường dẫn trong một notebook Kaggle đang mở:

```python
import os
os.environ['KMD_PROJECT_ROOT'] = '/kaggle/working/Olympic_AI_PTIT_2026_preliminary_round/KeMaoDanh'
os.environ['DATA_ROOT'] = '/kaggle/input/<dataset>/data/train'
os.environ['TEST_ROOT'] = '/kaggle/input/<dataset>/data/public_test'
```

Môi trường `.venv` của uv và kernel Kaggle đang mở là hai môi trường riêng. Cách dùng đúng lockfile mà không thay kernel hiện tại là gọi qua uv từ một cell shell, sau khi cài uv và đồng bộ thư viện như hướng dẫn ở trên:

```bash
cd /kaggle/working/Olympic_AI_PTIT_2026_preliminary_round/KeMaoDanh
KMD_SMOKE=0 KMD_RUN_ID=kaggle_run KMD_PROFILE=full uv run --locked --extra cu128 jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --output-dir outputs/kaggle notebooks/pipeline_end_to_end.ipynb
```

Để chạy một notebook của repo trong môi trường uv và lưu output, dùng lệnh sau ở cùng thư mục:

```bash
uv run --locked --extra cu128 jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --output-dir outputs/kaggle notebooks/00_problem_and_data.ipynb
```

Nếu mở trực tiếp từng file trong giao diện Kaggle, kernel đó cần có các thư viện tương ứng; chỉ tạo `.venv` không tự đổi kernel. CUDA cần được bật cho các bài CNN; chỉ baseline LR chạy được hoàn toàn bằng CPU. Internet cần được bật khi cài gói hoặc tải trọng số lần đầu.
