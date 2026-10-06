# AI LÀ AI

Bộ thực hành gồm năm notebook cho bài toán phân loại ảnh chân dung Real/Fake. Bắt đầu với [bài 1](notebooks/01_eda_baseline_geometry.ipynb), rồi học phần High-pass và kết hợp mô hình. [Bản đồ nội dung](READING_NOTEBOOK_MAP.md) nối từng phần Reading với cell và mã nguồn.

| Notebook | Nội dung |
|---|---|
| [01 — EDA, baseline và hình học](notebooks/01_eda_baseline_geometry.ipynb) | Kiểm kê ảnh; dung lượng JPEG; xám/màu; 5 fold; Macro-F1; CNN2; frozen backbone/fine-tuning; crop/FFT và đối chứng Same-FOV. |
| [02 — High-pass](notebooks/02_forensic_specialist.ipynb) | Gaussian, residual, chuẩn hóa; Dataset/DataLoader; AdamW, loss, AMP; một bài tập CPU; huấn luyện, TTA và phân tích lỗi. |
| [03 — Ensemble, ngưỡng và submission](notebooks/03_ensemble_threshold_submission.ipynb) | Ghép theo tên ảnh; fixes/breaks; lỗi còn lại; cross-fit ngưỡng; LR hai nguồn và Stack6; code tạo/đọc lại ZIP. |
| [00 — Pipeline](notebooks/00_pipeline_end_to_end.ipynb) | EDA trước khi học; hai nhánh RGB/High-pass; tùy chọn tạo OOF 5 fold; dự đoán 200 ảnh test và xuất ZIP. |
| [04 — Các hướng thử nghiệm](notebooks/04_negative_results_and_ablation.ipynb) | Haar, Sobel, sample weights; ba nhánh; đánh giá nhóm; calibration/evaluation; đọc lại các kết quả lịch sử trong Phụ lục D. |

## Chạy để học hoặc giảng thử

Mở notebook bằng các liên kết dưới đây, dùng runtime mới rồi chạy các cell theo thứ tự. Cell đầu tải repo và cài package; phần dữ liệu tự tải bộ ảnh [who_is_AI](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view).

| Notebook | Google Colab |
|---|---|
| 01 — EDA, baseline và hình học | [Mở trên Colab](https://colab.research.google.com/github/Purin1410/Olympic_AI_PTIT_2026_preliminary_round/blob/main/AI_LA_AI/notebooks/01_eda_baseline_geometry.ipynb) |
| 02 — High-pass | [Mở trên Colab](https://colab.research.google.com/github/Purin1410/Olympic_AI_PTIT_2026_preliminary_round/blob/main/AI_LA_AI/notebooks/02_forensic_specialist.ipynb) |
| 03 — Ensemble và submission | [Mở trên Colab](https://colab.research.google.com/github/Purin1410/Olympic_AI_PTIT_2026_preliminary_round/blob/main/AI_LA_AI/notebooks/03_ensemble_threshold_submission.ipynb) |
| 00 — Pipeline | [Mở trên Colab](https://colab.research.google.com/github/Purin1410/Olympic_AI_PTIT_2026_preliminary_round/blob/main/AI_LA_AI/notebooks/00_pipeline_end_to_end.ipynb) |
| 04 — Ablation | [Mở trên Colab](https://colab.research.google.com/github/Purin1410/Olympic_AI_PTIT_2026_preliminary_round/blob/main/AI_LA_AI/notebooks/04_negative_results_and_ablation.ipynb) |

Bản notebook từng lưu trong Drive giữ nội dung tại lúc lưu. Để lấy phiên bản vừa cập nhật, mở lại từ liên kết GitHub/Colab phía trên. Nếu chạy trên máy, tải hoặc clone repo và mở notebook trong `AI_LA_AI/notebooks/`.

- **Bài 1:** mặc định `RUN_BASELINE=False`, `RUN_E1=False`, `RUN_E2=False`, chạy EDA và minh họa trên CPU. Bật từng cờ khi muốn train trên T4. `BASELINE_EPOCHS=1` là lượt thử; E1 có 3 fits và E2 có 4 fits, mỗi fit mặc định 15 epoch.
- **Bài 2:** mặc định `RUN_TRAIN=False`; vẫn chạy phép lọc và bài tập CNN nhỏ với 4 ảnh trên CPU. Bật `RUN_TRAIN=True` để học ResNet34 15 epoch; `MODE="load"` để nạp checkpoint.
- **Bài 3:** mặc định đọc dự đoán reference trên CPU. `RUN_STACK6=True` chạy thêm LR trên sáu nguồn khi đã đặt hai tệp Stack6 từ gói ZIP vào `reference_artifacts/`. `SOURCE="learner"` cần cả hai nhánh cùng run_id; nhập run_id của bài 00 trong cell tạo workspace.
- **Bài 00:** mặc định huấn luyện hai nhánh trên T4, mỗi nhánh 15 epoch. `RUN_OOF=True` thêm 10 fits để tạo đủ 2.000 dự đoán OOF. Chỉ bật khi đã dành thời gian cho lượt này.
- **Bài 4:** `MODE="train"` chạy ba nhánh ResNet18 trên T4; `MODE="reference"` đọc kết quả đã lưu trên CPU. Train mặc định một fold; chọn cả năm fold sẽ có 15 fits. Reference mặc định đọc đủ 5 fold, 2.000 ảnh qua `REFERENCE_FOLDS`.

## Dữ liệu và kết quả

Bảng chia cố định nằm ở `assets/splits/train_folds.csv`. Ảnh train ở `data/train/images/`, ảnh Private Test ở `data/test/images/`. Có thể đặt `AILAAI_DATA_ROOT` nếu ảnh nằm ở nơi khác.

Checkpoint và log nằm trong `artifacts/<run_id>/`; bảng và hình nằm trong `outputs/<run_id>/`. Các notebook dùng run_id có hậu tố `reading_v2` để tách lượt mới khỏi bản trước. Khởi động lại kernel vẫn giữ file trong runtime; khi xóa runtime cần lưu checkpoint ra ngoài nếu muốn học tiếp.

`reference_artifacts/` chứa OOF RGB và High-pass. Hai tệp tùy chọn `stack6_features.csv` và `stack6_provenance.json` được cung cấp trong gói ZIP của bài giảng; trước khi bật `RUN_STACK6`, chép chúng vào thư mục này trong runtime. `data/negative_results/` chứa bundle lịch sử cho bài 4. Các bảng reference là kết quả đã lưu, không phải kết quả huấn luyện vừa chạy.

## Sửa code trong bài

Các thuật toán được viết thành cell có thể đọc và sửa. Dòng `# ailaai-source` trỏ đến file/hàm tương ứng trong `src/ailaai/`. Notebook giữ các hàm quản lý file/checkpoint dùng chung để việc học tiếp không phụ thuộc thứ tự thao tác thủ công.

Người biên soạn sửa Markdown ở `notebook_sources/` và Python ở `src/ailaai/`, rồi chạy:

```bash
python scripts/build_notebooks.py --write
python scripts/build_notebooks.py --check
python -m pytest tests -q
```

`--write` đồng bộ cả thân hàm Python và notebook; `--check` phát hiện sai khác giữa các nguồn.

## Phạm vi kiểm tra

Bản bổ sung 06/10/2026 đã được kiểm tra EDA trên đủ 2.000 ảnh, chạy các phần phân tích/reference trên CPU và pipeline với CNN nhỏ. Các lượt đầy đủ ResNet34, E1, E2 và OOF mới chưa được chạy lại trên GPU. Có 18 tests đạt; các thay đổi cấu hình cho lượt CPU và báo cáo chi tiết được giữ trong workspace bài giảng.

[Biên bản Colab trước đó](evidence/colab_runall_20261006/README.md) thuộc phiên bản `ba003c7`; không dùng biên bản đó để khẳng định bản bổ sung đã chạy đủ GPU.
