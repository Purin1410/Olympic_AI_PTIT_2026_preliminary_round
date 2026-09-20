# Kẻ mạo danh - Olympic AI PTIT 2026

Output và hình hiện có đến từ [lượt chạy đầy đủ ngày 20/09/2026](docs/execution.md).

Kho lưu trữ mã nguồn giải pháp và chuỗi bài giảng thực nghiệm cho bài toán **Kẻ mạo danh (The Impostor)** trong khuôn khổ vòng sơ loại Olympic AI PTIT 2026.

Mỗi mẫu dữ liệu là một cặp gồm hai bức ảnh chân dung (`image_0` và `image_1`), trong đó có đúng một ảnh thật và một ảnh giả mạo. Hệ thống học cách so sánh các đặc trưng thị giác và dấu vết kỹ thuật số giữa hai bức ảnh để dự đoán nhãn vị trí của ảnh giả: `fake_position` thuộc tập {0, 1}.

Tài liệu chi tiết: [Mô tả bài toán](docs/challenge.md), [Kiến trúc pipeline và mã nguồn](docs/pipeline.md), [Bảng ánh xạ 27 trang báo cáo nghiên cứu](docs/report_mapping.md).

---

## 1. Hướng dẫn Cài đặt Môi trường (uv)

Dự án yêu cầu phiên bản đã thử nghiệm **Python 3.11** và sử dụng `uv` để cài các phiên bản thư viện đã khóa trong `uv.lock`:

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

## 2. Chuẩn bị Dữ liệu từ file data.zip được cung cấp cho lớp

Khi nhận được **file data.zip được cung cấp cho lớp**, người học sử dụng kịch bản xử lý dữ liệu an toàn:

```bash
# Đối với môi trường GPU:
uv run --locked --extra cu128 python scripts/prepare_official_dataset.py --zip-path data.zip --dest-dir data

# Đối với môi trường CPU:
# uv run --locked --extra cpu python scripts/prepare_official_dataset.py --zip-path data.zip --dest-dir data
```

Kịch bản tự động xử lý các cấu trúc thư mục lồng nhau thường gặp trong gói dữ liệu của lớp (như `data/data/train` và `data/private_test/private_test`), bảo đảm toàn vẹn từng byte ảnh JPEG gốc và kiểm tra an toàn chống tấn công Zip Slip.

Nếu lưu trữ dữ liệu tại thư mục ngoài dự án, bạn có thể thiết lập các biến môi trường:
```bash
export DATA_ROOT=/duong/dan/den/data/train
export TEST_ROOT=/duong/dan/den/data/private_test/private_test
```

*Lưu ý:* Nếu không cung cấp tập test, toàn bộ quá trình huấn luyện và đánh giá trên 800 mẫu phát triển vẫn hoàn thành đầy đủ và xuất thông báo `no submission produced (test root not provided)`. Dữ liệu và checkpoint được giữ ngoài Git thông qua `.gitignore`.

---

## 3. Danh mục 13 Notebook Thực hành

Mỗi bài nối các bước: `Quan sát -> Đặt câu hỏi -> Can thiệp kiểm soát -> Diễn giải kết quả -> Quyết định tiếp theo`.

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
| **A** | [extension_a_representations](notebooks/extension_a_representations.ipynb) | Frozen Embedding + LR, Partial vs Full fine-tuning, Top-2 vs Mean pooling, ResNet-18 trên ảnh RGB vs Gaussian/NPR residual | Train 3 Folds | Bảng đối chứng mở rộng về biểu diễn và kiến trúc |
| **B** | [extension_b_objectives_and_repair](notebooks/extension_b_objectives_and_repair.ipynb) | Khảo sát 4 biến thể hàm mất mát ở 19 epoch terminal (image, pairwise, mixed, re-pairing) trên DenseNet-121 3-fold OOF | Train 3 Folds | Bảng so sánh hàm mục tiêu và kỹ thuật ghép cặp lại |
| **C** | [extension_c_data_scaling](notebooks/extension_c_data_scaling.ipynb) | Khảo sát quy mô dữ liệu 25%, 50%, 100% train dưới cùng ngân sách cố định 391 gradient updates terminal trên EfficientNet-B2 | Train Subsets | Đường cong tăng trưởng theo quy mô dữ liệu |
| **D** | [extension_d_resize_augmentation](notebooks/extension_d_resize_augmentation.ipynb) | Tăng cường dữ liệu co giãn ngẫu nhiên Resize Augmentation 90-100% trên EfficientNet-B2 (đối chứng b2 chuẩn vs b2_resize) | Train 3 Folds | Bảng đánh giá ảnh hưởng của phép co giãn ngẫu nhiên |
| **⚡** | [pipeline_end_to_end](notebooks/pipeline_end_to_end.ipynb) | Điều phối toàn diện theo giao thức hiện hành: PROFILE baseline/full, sàng lọc, xác nhận 3 seed, thử blend, khóa freeze, xuất bài | Train (+ Test) | Toàn bộ artifacts và tệp `submission.csv` chuẩn |

---

## 4. Chi tiết Phương pháp luận và Kỹ thuật Mô hình hóa

### 4.1. Mốc kiểm soát ngân sách `center60_cap48`
Trong các bài thực hành đầu, cấu hình `center60` chạy trần 24 epoch với microbatch 24. Khi mở rộng sang các kiến trúc 288px hoặc EfficientNet, việc thay đổi đồng thời kích thước ảnh, trần epoch và microbatch sẽ tạo ra yếu tố gây nhiễu kép. Cấu hình `center60_cap48` ra đời nhằm đóng vai trò mốc đối chứng chuẩn: giữ nguyên DenseNet-121, view center60, size 224 nhưng đặt trần 48 epoch, microbatch 8 và tích lũy gradient để giữ batch hiệu dụng bằng 24. Ta so `center60_cap48` với `dense288`, `center72`, `b0_224` hoặc `b2_224` để lần lượt đổi size, vùng nhìn hoặc backbone. Riêng B2 ở 288px (`b2`) được so với `dense288` để chỉ đổi backbone.

### 4.2. Sàng lọc và Lựa chọn Mô hình Đơn Động
Quy trình không mặc định trước mô hình nào sẽ chiến thắng:
1. Huấn luyện 6 cấu hình sàng lọc và 2 mốc tham chiếu (`center60`, `native`) trên seed chính `20260917`.
2. Lập shortlist gồm đúng 2 challenger có điểm Macro-F1 cao nhất trên tập phát triển.
3. Xác nhận shortlist cùng 3 mốc đối chứng trên 3 hạt giống ngẫu nhiên: `20260917`, `20260918`, `20260919`.
4. Mô hình đơn chiến thắng (winner) là mô hình đạt Macro-F1 trung bình cao nhất qua 3 seed (hòa xét theo thứ tự chữ cái).

### 4.3. Phân tích Lỗi và Cổng Blend Selection Gate (+0.0050)
Sau khi chọn được mô hình đơn winner, hệ thống căn chỉnh 800 cặp OOF ở seed `20260917` giữa winner và mô hình `native` để phân tích ma trận 4 góc:
- Cả hai cùng đúng / Cả hai cùng sai.
- Chỉ winner đúng (native sai): rủi ro làm hỏng kết quả khi kết hợp.
- Chỉ native đúng (winner sai): cơ hội sửa sai tiềm năng của blend.

Tỷ lệ kết hợp được cố định ở mức 50/50: $p_{\text{blend}} = 0.5 \cdot p_{\text{winner}} + 0.5 \cdot p_{\text{native}}$.
Ta chỉ chọn blend khi mức tăng Macro-F1 trung bình chưa làm tròn qua 3 seed phải đạt tối thiểu **+0.0050** (+0.50 điểm phần trăm) so với mô hình đơn:
$$\Delta_{\text{mean}} = \frac{1}{3} \sum_{s=1}^{3} (\text{F1}_{\text{blend}, s} - \text{F1}_{\text{winner}, s}) \ge +0.0050$$
Nếu không đạt ngưỡng hoặc nếu winner chính là `native`, hệ thống giữ mô hình đơn và bỏ qua việc kết hợp để tránh tăng độ phức tạp khi suy diễn.

### 4.4. Khóa Quyết định và Hai Chiến lược Nộp bài
- **Khóa trước khi xem private (`review_freeze.json`):** Quyết định chọn phương pháp nộp và danh sách đối chiếu được ghi nhận trước khi mở nhãn kiểm tra, giúp giữ tính khách quan giữa tập phát triển và tập kiểm tra.
- **Chiến lược `fold_ensemble` (mặc định):** Lấy trung bình xác suất dự đoán từ 3 mô hình fold trên tập test. Ba mô hình fold có tập huấn luyện chồng lặp (mỗi fold học trên hai phần ba dữ liệu phát triển, phần train của các fold giao nhau), không phải các mô hình độc lập. Việc lấy trung bình 3 fold là giải pháp tận dụng cả 3 mạng đã huấn luyện, không đảm bảo tự động giảm phương sai trong mọi trường hợp.
- **Chiến lược `refit_all` (tùy chọn):** Bắt đầu lại từ pretrained và huấn luyện trên toàn bộ 1.000 cặp có nhãn (gộp cả 200 cặp holdout). Ngân sách epoch được lấy cố định từ trung vị best epoch của 3 fold ở seed đầu (hoặc 19 epoch cố định cho Native), không dùng tập test để chọn điểm dừng.
  - Nếu phương pháp nộp là mô hình đơn: tạo 1 checkpoint tương ứng.
  - Nếu phương pháp nộp là mô hình kết hợp (`blend_selected_native`): huấn luyện riêng từng thành phần (1 checkpoint cho nhánh winner và 1 checkpoint cho nhánh native, tổng cộng 2 checkpoint) rồi lấy trung bình xác suất khi suy diễn.

---

## 5. `from kmd...` lấy mã ở đâu?

Thư viện `kmd` nằm ngay trong repo, tại [src/kmd](src/kmd). Lệnh `uv sync` cài gói này vào môi trường của bài; không cần lấy thêm một repo riêng. Notebook cũng tìm `src/` từ thư mục `KeMaoDanh` để có thể đọc mã trực tiếp. Giữ nguyên cả thư mục khi clone, không chỉ sao chép notebook riêng.

Đọc [core.py](src/kmd/core.py) cho CSV, metric và submission; [models.py](src/kmd/models.py) cho điểm số và hàm mất mát; [training.py](src/kmd/training.py) cho vòng huấn luyện; [pipeline.py](src/kmd/pipeline.py) cho train theo fold và suy diễn; [decision_flow.py](src/kmd/decision_flow.py) cho sàng lọc và cổng blend; [finalization.py](src/kmd/finalization.py) cho bước khóa và refit. Notebook giữ các phép tính chính ở gần phần giải thích, còn những bước lặp lại dùng các hàm chung này.

---

## 6. Dữ liệu Đặt Cạnh Repo và Môi trường Kaggle

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

---

## 7. Kiểm tra Tĩnh và Giao thức Dòng lệnh

### 7.1. Phân biệt CLI Kế thừa (`run_pipeline.py`) và Giao thức Hiện hành
- Kịch bản `scripts/run_pipeline.py` là **giao thức dòng lệnh cũ (legacy protocol)**. Kịch bản này thực thi 5 kiến trúc CNN cố định và mặc định suy diễn B2 hoặc blend B2+Native theo báo cáo lịch sử. Kịch bản này không áp dụng luồng quyết định động mới (sàng lọc 8 cấu hình, shortlist 3 seed, cổng blend động theo winner, khóa freeze_review và refit 1.000 cặp).
- Để vận hành đúng và đủ toàn bộ giao thức thực nghiệm hiện hành, khuyến nghị sử dụng notebook điều phối `notebooks/pipeline_end_to_end.ipynb`.

### 7.2. Kiểm tra Cấu trúc Tĩnh Dự án (`check_structure.py`)
Kịch bản kiểm tra tĩnh chạy nhanh với thư viện chuẩn Python:

```bash
# Kiểm tra cấu trúc tĩnh của 13 notebook (chấp nhận notebook đã có kết quả thực thi):
python scripts/check_structure.py

# Kiểm tra nghiêm ngặt: yêu cầu mọi cell code phải có execution_count tuần tự 1..N và không có lỗi:
python scripts/check_structure.py --require-executed

# Kiểm tra thư mục bản nháp:
python scripts/check_structure.py --notebook-dir notebooks/revision_draft
```

### 7.3. Kịch bản Xuất bản Notebook Đã Thực thi (`publish_executed_notebooks.py`)
Dành cho người quản trị cập nhật kết quả sau khi hoàn thành lượt chạy trên GPU:

```bash
python scripts/publish_executed_notebooks.py \
    --evidence-dir /duong/dan/den/evidence \
    --archive-dir /duong/dan/ngoai_repo/archive \
    --dry-run
```

- Kiểm tra tệp `execution_summary.json`: đúng 13 notebook duy nhất, trạng thái `passed`, chế độ `full`, `smoke=False` (kiểu boolean), khớp số lượng cell code, và không có lỗi.
- Yêu cầu bắt buộc trường `source_sha256` khớp với mã băm của bản nháp gốc trong `revision_draft`, kiểm tra thứ tự cell code tuần tự 1..N và so khớp từng cell để chống can thiệp mã nguồn.
- Kiểm tra trước xung đột tệp lưu trữ (preflight collision check) và từ chối nếu tệp lưu trữ đích đã tồn tại với nội dung khác.
- Từ chối thư mục lưu trữ nằm trong repository hoặc thư mục evidence nằm trong thư mục notebook chính.
- Sao lưu toàn bộ notebook cũ ra thư mục ngoài trước khi thay thế hoặc dọn dẹp các tệp cũ.
