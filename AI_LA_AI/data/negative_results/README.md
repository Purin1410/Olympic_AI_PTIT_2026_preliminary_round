# Dự đoán tham khảo cho bài 04

Thư mục này chứa kết quả OOF trên 2.000 ảnh train để đọc phần ablation trong bài 04. Khi đặt `MODE="reference"`, notebook đọc các bảng dưới đây trên CPU. Chế độ `train` tải ảnh và huấn luyện ba nhánh riêng.

| Tệp | Nội dung |
| --- | --- |
| `oof_predictions.csv` | Tên ảnh, nhãn, fold, các xác suất và đặc trưng dùng để phân nhóm. |
| `threshold_choices.csv` | Ngưỡng ảnh xám/màu đã chọn trên bốn fold còn lại. |
| `promotion_gates.json` | Kết quả so nhánh có trọng số, nhóm ít biên và đóng góp khi thay nhánh trong stack. |
| `source_manifest.json` | Nguồn từng cột, cách tính đặc trưng và SHA256 của các tệp. |
| Ba tệp PNG | Biểu đồ Wavelet, nhóm ít biên và chọn ngưỡng, tính từ các bảng trên. |

## Xác suất và đặc trưng

`legacy_stack_prob` là xác suất từ Legal7 (`results/stack_crossfit_legal7_c1/oof.csv`). Tệp sau calibration chứa quyết định 0/1, nên dùng xác suất gốc khi thử ngưỡng. Trên 2.000 ảnh, Macro-F1 của ngưỡng cố định là 96,6998%, cross-fit là 96,3996%, còn trung vị ngưỡng áp lại OOF là 96,7498%. Phép áp lại OOF dùng lại nhãn đã tham gia chọn ngưỡng.

`is_gray` dùng ảnh RGB thu nhỏ với `thumbnail((64,64))`, rồi kiểm tra trung bình `max(pixel)-min(pixel) < 0.5`. `edge_ratio` trong bundle lấy từ đặc trưng **Laplacian 64px** đã lưu. Phân vị 25% của ảnh Fake ở bốn fold còn lại cho ngưỡng nhóm ít biên của fold đang xét. Phần huấn luyện trong notebook dùng **Sobel** trên vùng crop; hai phép đo có thang giá trị khác nhau.

Các bảng giữ đủ 2.000 ID và ghép theo `file_name,label,fold`. Khi đọc bảng Wavelet, lưu ý RGB R34 và Wavelet R18 thay cả backbone lẫn biểu diễn. Bảng stack dùng kết quả đã lưu trong `promotion_gates.json`.

## Tạo lại bundle từ dữ liệu nguồn

Từ thư mục `AI_LA_AI/`, với pandas, NumPy, Pillow và Matplotlib:

```bash
python scripts/build_negative_results.py \
  --historical-root /path/to/historical/AI_LA_AI \
  --edge-features /path/to/historical/edge_ratio_laplacian64_c0625.csv \
  --train-dir /path/to/TACVU1/data/train
```

Để xem lại các bảng, mở [bài 04](../../notebooks/04_negative_results_and_ablation.ipynb) và chọn `MODE="reference"`. Nguồn và mã băm của bundle nằm trong [source_manifest.json](source_manifest.json).
