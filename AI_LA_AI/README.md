# AI LÀ AI

Bộ thực hành này dùng ảnh chân dung để phân loại người thật (**Real, nhãn 0**) và ảnh do AI tạo ra (**Fake, nhãn 1**). Năm notebook đi từ khám phá dữ liệu, xây dựng mô hình và phân tích lỗi đến kết hợp dự đoán, chọn ngưỡng và tạo tệp nộp bài. Các phép tính chính được viết trong cell để bạn có thể theo dõi và sửa khi học.

Nếu mới bắt đầu, hãy đi lần lượt qua bài **01 → 02 → 03**. Bài **00** gom toàn bộ pipeline; bài **04** dành cho các thử nghiệm ablation và cách đọc kết quả.

## Notebook và Google Colab

| Bài | Nội dung | Notebook | Colab |
| --- | --- | --- | --- |
| 00 | Từ dữ liệu đến submission: EDA, hai nhánh RGB/High-pass, kết hợp xác suất và xuất ZIP. | [Pipeline](notebooks/00_pipeline_end_to_end.ipynb) | [Mở Colab](https://colab.research.google.com/drive/1DOdCC4VWEq8qK9juWKeP5lzbGF3xia77#scrollTo=eeb9631b) |
| 01 | EDA, Macro-F1, CNN2, học chuyển giao, crop/FFT và đối chứng Same-FOV. | [EDA, baseline và hình học](notebooks/01_eda_baseline_geometry.ipynb) | [Mở Colab](https://colab.research.google.com/drive/1KnXe0jnp1rbEKRIn-zptR3_6NhVCG-aY#scrollTo=2e6741dd) |
| 02 | Gaussian, residual, High-pass và các bước huấn luyện, validation, TTA. | [High-pass](notebooks/02_forensic_specialist.ipynb) | [Mở Colab](https://colab.research.google.com/drive/150yCSqBAQj9R5PsneU2JHGwh1tGR-E-m#scrollTo=01c04168) |
| 03 | Ensemble, fixes/breaks, ngưỡng quyết định, stacking và kiểm tra định dạng submission. | [Ensemble và submission](notebooks/03_ensemble_threshold_submission.ipynb) | [Mở Colab](https://colab.research.google.com/drive/10wOB3F3pMCgH-_T0gyrlckeBUfQCt7Jj#scrollTo=bcb0957e) |
| 04 | RGB, Haar và trọng số mẫu; so sánh lỗi, đánh giá nhóm và chọn ngưỡng. | [Ablation](notebooks/04_negative_results_and_ablation.ipynb) | [Mở Colab](https://colab.research.google.com/drive/19SWSTvwrJwOVP4TggsYZVNcPumLPdLod#scrollTo=a692a04d) |

## Chạy trên Colab

Mở bài từ bảng trên rồi chạy các cell theo thứ tự. Cell code đầu chuẩn bị môi trường và package; phần dữ liệu tải bộ ảnh theo cấu hình của bài.

Bài 01 và 02 mặc định chạy phần phân tích, minh họa trên CPU. Khi muốn huấn luyện, chọn **Runtime → Change runtime type → T4 GPU** rồi bật cờ tương ứng:

- **Bài 01:** `RUN_BASELINE`, `RUN_E1` và `RUN_E2` mặc định là `False`. `BASELINE_EPOCHS=1` dùng cho lượt thử; E1 gồm ba lượt huấn luyện, E2 gồm bốn lượt để so sánh Native và Resampled trên hai fold.
- **Bài 02:** `RUN_TRAIN=False`. Bật `True` để huấn luyện High-pass; dùng `MODE="load"` khi đã có checkpoint.
- **Bài 03:** `SOURCE="reference"` đọc dự đoán đã lưu. Đổi thành `"learner"` để phân tích kết quả của bạn, với `RUN_ID` khớp bài 00. Phần Stack6 cần hai tệp `stack6_features.csv` và `stack6_provenance.json` trong `reference_artifacts/` trước khi bật `RUN_STACK6`.
- **Bài 00:** mặc định huấn luyện hai nhánh trên GPU, mỗi nhánh 15 epoch. `RUN_OOF=False`; bật cờ này nếu muốn tạo dự đoán OOF cho đủ năm fold.
- **Bài 04:** `MODE="train"`, `FOLDS=[0]`, `EPOCHS=15` chạy ba nhánh ResNet18 trên GPU. Đổi sang `MODE="reference"` để đọc kết quả lịch sử trên CPU.

Các bảng `reference` là dự đoán của những lượt chạy đã lưu. Khi đánh giá mô hình mình vừa huấn luyện, dùng checkpoint và dự đoán do chính lượt đó tạo ra.

## Chạy trên máy cá nhân

Từ thư mục repo, tạo môi trường Python và cài package:

```bash
cd AI_LA_AI
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Sau đó mở một notebook trong `notebooks/`. Các lượt huấn luyện đầy đủ cần GPU CUDA.

## Dữ liệu và cấu trúc thư mục

Bảng chia năm fold cố định nằm ở `assets/splits/train_folds.csv`. Nếu chuẩn bị dữ liệu thủ công, dùng cấu trúc sau:

| Dữ liệu | Thư mục ảnh | Bảng thông tin | Cột cần có |
| --- | --- | --- | --- |
| Train | `data/train/images/` | `data/train/manifest.csv` | `file_name,label` |
| Private Test | `data/test/images/` | `data/test/manifest.csv` | `file_name` |

Các bảng có thể thêm cột `path`. Nếu ảnh nằm ở nơi khác, đặt biến môi trường `AILAAI_DATA_ROOT` trước khi tạo workspace.

`notebooks/` chứa năm bài thực hành; `notebook_sources/` chứa bản Markdown của các cell; `src/ailaai/` chứa code dùng chung; `configs/` lưu cấu hình. Dự đoán tham khảo nằm trong `reference_artifacts/`, còn bundle kết quả lịch sử của bài 04 nằm trong `data/negative_results/`.

## Kết quả và checkpoint

Checkpoint và log được lưu dưới `artifacts/<run_id>/`; bảng, hình và tệp nộp bài nằm dưới `outputs/<run_id>/`. Bài 00 tạo `submission.zip`, bên trong có một tệp `submission.csv`. ZIP ở bài 03 dùng để minh họa và kiểm tra định dạng.

Chạy lại cùng cấu hình sẽ tiếp tục checkpoint còn dở hoặc nạp lượt đã hoàn tất. Khởi động lại kernel vẫn giữ file trong runtime. Nếu xóa runtime hoặc Colab thu hồi phiên, bạn cần nạp lại checkpoint đã lưu ở nơi khác để học tiếp.
