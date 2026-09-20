# Kẻ mạo danh - Olympic AI PTIT 2026

Repository bài giải và chuỗi bài giảng thực hành thực nghiệm  cho bài toán **Kẻ mạo danh (The Impostor)**.

Mỗi mẫu dữ liệu là một cặp gồm 2 ảnh (`image_0` và `image_1`), trong đó có **đúng một ảnh thật** và **một ảnh giả**. Mô hình học cách so sánh sự sai khác kỹ thuật số giữa 2 ảnh và dự đoán nhãn vị trí ảnh giả: `fake_position` $\in \{0, 1\}$.

Chi tiết đề bài và phương pháp: [Mô tả bài toán](docs/challenge.md), [Pipeline thực nghiệm](docs/pipeline.md).

---

## 1. Hướng dẫn Cài đặt Môi trường

Repository chuẩn hóa môi trường cho cả hai hình thức: **Máy Local** và **Google Colab**.

### A. Cài đặt trên Máy Local
Môi trường dự kiến: **Python 3.11**. Từ thư mục gốc repository:

```bash
cd KeMaoDanh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Với máy có GPU NVIDIA (CUDA 12.8):
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Hoặc máy chỉ có CPU (dành cho EDA & Logistic Regression):
# python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --user --name kmd --display-name "Kẻ mạo danh"
python -m jupyter lab
```

### B. Cài đặt trên Google Colab (với GPU)
Khi chạy trên Google Colab, bạn có thể clone repository của mình hoặc upload trực tiếp thư mục `KeMaoDanh` lên Google Drive để lưu trữ checkpoint bền vững:

```python
# 1. Mount Google Drive để lưu model và kết quả
from google.colab import drive
drive.mount('/content/drive')

# 2. Chuyển vào thư mục làm việc (ví dụ trên Drive)
%cd /content/drive/MyDrive/KeMaoDanh

# 3. Cài đặt các thư viện ghim phiên bản (pinned dependencies)
!pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
!pip install -r requirements.txt
!pip install -e .

# 4. Thiết lập biến môi trường trỏ đến dữ liệu
import os
os.environ['DATA_ROOT'] = '/content/drive/MyDrive/dataset/train'
# Chỉ đặt TEST_ROOT nếu có bộ test:
# os.environ['TEST_ROOT'] = '/content/drive/MyDrive/dataset/test'
```

Sau khi cài hoặc đổi torch trên Colab, restart runtime rồi chạy lại cell mount/đường dẫn. Các notebook dùng cùng thuật toán cho local và Colab; chọn GPU runtime cho bài 02–04 và nhánh full. Lần đầu tải trọng số ImageNet cần mạng. Phiên Colab có thể ngắt; lưu repo và artifacts trên Drive để giữ các fold đã hoàn tất. Fold chạy dở chưa hỗ trợ resume.

---

## 2. Cấu trúc Dữ liệu

Người học tự cung cấp dữ liệu ảnh thật theo cấu trúc:

```text
data/
├── train/
│   ├── pairs.csv           # Cột: pair_id,image_0,image_1,fake_position
│   └── images/...          # Các file ảnh thật và giả
└── test/                   # (Tùy chọn) Bộ dự đoán không có nhãn
    ├── pairs.csv           # Cột: pair_id,image_0,image_1
    └── images/...
```

Ảnh phải là **file nguyên bản**: không recompress hoặc resize cả dataset trước khi trích xuất. Cột `pair_id` được đọc ở định dạng chuỗi (`str`) để giữ nguyên các số `0` ở đầu (ví dụ: `"00295"`).

Nếu đặt dữ liệu ở thư mục khác, thiết lập biến môi trường trước khi khởi động Jupyter Lab:
```bash
export DATA_ROOT=/duong/dan/dataset/train
export TEST_ROOT=/duong/dan/dataset/test
python -m jupyter lab
```

*Nếu không có bộ test, pipeline và notebook vẫn hoàn thành trọn vẹn việc huấn luyện và đánh giá trên tập Development, thông báo trạng thái `no submission produced` mà không gây lỗi.*

---

## 3. Danh mục 7 Notebook Thực hành

Chuỗi bài học được thiết kế tuần tự theo phương pháp tiếp cận khoa học (`Câu hỏi → Ví dụ → Tính toán → Kết quả → Giới hạn → Bước tiếp theo`):

| STT | Notebook | Nội dung trọng tâm | Đầu vào | Đầu ra |
|:---:|---|---|---|---|
| **00** | [00_problem_and_data](notebooks/00_problem_and_data.ipynb) | Bản chất bài toán cặp ảnh, schema nhãn, 800 dev vs 200 holdout, Macro-F1 vs Accuracy | Train `pairs.csv` | Bảng phân tích phân bố và ví dụ số nhỏ |
| **01** | [01_features_and_lr](notebooks/01_features_and_lr.ipynb) | 32 đặc trưng thống kê, quy ước hiệu $x = f_1 - f_0$, tính đối xứng $p(-x)=1-p(x)$, EDA Train Fold 0, 3 biến thể LR | Train images | Bảng so sánh 3 mô hình LR trên Fold 0 |
| **02** | [02_pretrained_and_finetune](notebooks/02_pretrained_and_finetune.ipynb) | DenseNet-121 backbone + Head, BatchNorm eval, Head-only warmup, trực quan hóa `train_step`, Frozen vs Full Fine-tune | Train Fold 0 | Checkpoint và so sánh Validation Fold 0 |
| **03** | [03_native_and_resampling](notebooks/03_native_and_resampling.ipynb) | Vấn đề của Resize, 4 Native crops $224 \times 224$ (1:1), Multi-crop Mean Logits, đối chứng Native vs Resampled ($112 \rightarrow 224$) | Train Fold 0 | Đánh giá đối chứng ảnh hưởng của nội suy |
| **04** | [04_errors_and_blend](notebooks/04_errors_and_blend.ipynb) | EfficientNet-B2 $288 \times 288$ (nhiễu đồng thời), căn chỉnh dự đoán, ma trận lỗi 4 góc, Blend 50/50 B2/Native | Validation Fold 0 | Thống kê số lỗi sửa được vs lỗi mới |
| **05** | [05_validation_and_submission](notebooks/05_validation_and_submission.ipynb) | Đánh giá 3-Fold Cross-Validation (Full 800 OOF), caveats chọn checkpoint, quy trình inference test và xuất CSV chuẩn | 3-Fold Checkpoints + Test | Bảng OOF đầy đủ và file `submission.csv` |
| **⚡** | [pipeline_end_to_end](notebooks/pipeline_end_to_end.ipynb) | **Bộ điều phối tích hợp một lượt:** Chạy nhanh Baseline LR (`PROFILE='baseline'`) hoặc toàn bộ CNN x 3 fold + Blend (`PROFILE='full'`) | Train (+ Test tùy chọn) | Toàn bộ session artifacts và submission |

---

## 4. Lựa chọn Luồng Thực thi

- **Học tập từng bước:** Đi theo thứ tự từ `00` đến `05`.
- **Chạy nhanh Baseline (Fast Baseline):** Mở [pipeline_end_to_end.ipynb](notebooks/pipeline_end_to_end.ipynb) với `PROFILE = 'baseline'`. Dùng CPU để trích xuất đặc trưng, huấn luyện 3-fold LR và tạo submission.
- **Chạy so sánh toàn diện (Full Comparison):** Mở [pipeline_end_to_end.ipynb](notebooks/pipeline_end_to_end.ipynb) với `PROFILE = 'full'` trên máy có GPU CUDA để huấn luyện toàn bộ 5 mô hình CNN x 3 fold và kết hợp Blending.

---

## 5. Chạy bằng Dòng lệnh (CLI)

Từ thư mục `KeMaoDanh/`:

```bash
# 1. Chạy Baseline Logistic Regression (Nhanh trên CPU/GPU):
python scripts/run_pipeline.py --train-root "$DATA_ROOT" --profile baseline

# 2. Chạy Toàn bộ Pipeline CNN + Blend (Yêu cầu GPU):
python scripts/run_pipeline.py --train-root "$DATA_ROOT" --test-root "$TEST_ROOT" --profile full

# 3. Đánh giá file submission với nhãn thật (nếu có):
python scripts/evaluate_predictions.py --labels /path/to/truth.csv --predictions outputs/run_id/submission.csv

# 4. Kiểm tra tính toàn vẹn cấu trúc tĩnh:
python scripts/check_structure.py
```

---

## 6. Kiểm tra Toàn vẹn và Trạng thái Nghiệm thu

Script `check_structure.py` kiểm tra tĩnh toàn bộ cú pháp Python (AST), cấu trúc JSON của 7 notebook, metadata, tính duy nhất của cell ID, và cấu hình split:

```bash
python scripts/check_structure.py
```

## Phiên chạy và phạm vi kiểm tra

Giữ cùng `RUN_ID='lesson_session'` trong bài 01–05. Bài 01 tạo phiên; bài 05 mặc định `METHOD='lr'`, có thể chạy thẳng sau bài 01 trên CPU. Nếu học CNN, chạy lần lượt 02–04 rồi chọn phương pháp ở bài 05. Notebook tích hợp dùng RUN_ID riêng; muốn tái sử dụng các fold từ bài học, đặt cùng RUN_ID và giữ nguyên code, config, split và ảnh.

Phiên lưu hash của ảnh, manifest và code. Fold hoàn chỉnh chỉ được dùng lại khi config, ID, metadata dự đoán và hash checkpoint khớp. Fold thiếu được train; fold dở dang hoặc không khớp sẽ dừng, hãy đặt RUN_ID mới để bắt đầu một thí nghiệm khác. LR được fit lại từ dữ liệu thật khi hoàn tất 3 fold.

Bản này đã được kiểm tra tĩnh và đọc rà soát code/notebook; **chưa chạy notebook, train hay kiểm tra runtime trên GPU**. Không có điểm số tái lập mới được khẳng định. Các con số và ảnh kết quả sẽ xuất hiện khi người học chạy với dữ liệu thật. Dependency được ghim để mô tả môi trường dự kiến, chưa phải xác nhận cài đặt thành công trên mọi máy.
