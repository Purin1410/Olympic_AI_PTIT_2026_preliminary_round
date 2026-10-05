# Replay phụ lục 04

Gói train-only gồm 2.000 ID/nhãn/fold chuẩn, xác suất OOF lịch sử, feature biên,
ngưỡng theo fold, gate receipt và ba biểu đồ Matplotlib PNG. Notebook chỉ đọc gói
này, dùng `pandas` và `IPython`; không cần ảnh gốc, GPU, model weights hay mạng.
Trong Colab, đặt `data/negative_results/` dưới `/content` hoặc cạnh notebook.

`source_manifest.json` ghi nguồn từng cột và thông tin các tệp đầu ra. Các phép
ghép khi đóng gói kiểm tra one-to-one trên `file_name,label,fold` và đủ 2.000 ID.
`promotion_gates.json` được sao chép nguyên byte từ receipt của run edge-weighted.

## Nguồn xác suất stack

`legacy_stack_prob` lấy từ `results/stack_crossfit_legal7_c1/oof.csv`. Đây là nguồn
xác suất gốc tái lập **toàn bộ 2.000 quyết định cross-fit** trong
`results/stack7_gray_color_crossfit_calibration/oof.csv`, cùng điểm fixed
96,6998%, cross-fit 96,3996% và median 96,7498%. CSV sau calibration chứa quyết
định 0/1, nên không thể dùng làm xác suất để dò ngưỡng.

Fallback cũ `evidence/replay/legacy_stack_oof.csv` là bản sao của
`results/stack6_no_center80_c1/oof.csv`. Nó có cùng điểm fixed nhưng khác xác
suất: với cờ xám đúng, cross-fit đạt 96,4497% và median đạt 96,6498%, không khớp
receipt Case 3. Vì vậy bundle dùng Legal7 thay cho fallback này.

## Feature lịch sử và phạm vi hình ảnh

`is_gray` được xuất bằng đúng logic `calibrate_stack_gray_color.py`: đổi RGB,
`thumbnail((64,64))`, rồi kiểm tra trung bình `max(pixel)-min(pixel) < 0.5`.
Hash tổng hợp của 2.000 ảnh train dùng để xuất cờ được lưu trong manifest.

`edge_ratio` lấy từ bản lịch sử `edge_ratio_laplacian64_c0625.csv` đã lưu trong
archive metadata. Đây là feature Laplacian, không đổi tên thành Sobel. Phân vị
25% của Fake ở bốn fold còn lại tái lập đủ năm cutoff của gate receipt.
Không thay bằng canonical re-audit 1.998 dòng: audit đó loại 01911.jpg và
01989.jpg từng bị ghi đè bởi contact-sheet. Bundle giữ phạm vi lịch sử 2.000 dòng.

Ba PNG là biểu đồ số liệu được tính từ chính bundle, không phải ảnh ví dụ hay
bằng chứng về nguyên nhân lỗi. G3 đọc quyết định và attribution đã lưu trong
receipt; Replay không chạy lại huấn luyện hoặc fitting stack. Các ngưỡng là
lựa chọn đã lưu, không được tối ưu lại khi mở notebook.

## Tạo lại và kiểm tra

Từ thư mục `AI_LA_AI/`, với môi trường có pandas, NumPy, Pillow và Matplotlib:

```bash
python scripts/build_negative_results.py \
  --historical-root /path/to/historical/AI_LA_AI \
  --edge-features /path/to/historical/edge_ratio_laplacian64_c0625.csv \
  --train-dir /path/to/TACVU1/data/train
python scripts/build_notebooks.py --check
```

Để chạy kiểm thử độc lập trong kernel CPU mới, cài thêm `nbformat`, `nbclient`
và `ipykernel`, rồi chạy:

```bash
python scripts/verify_negative_results.py
```

Verifier sao chép notebook và bundle sang thư mục tạm, chạy mọi code cell, kiểm
tra hashes và ghi kết quả tại `evidence/negative_results/replay_verification.json`.
Notebook phát hành giữ outputs rỗng.
