# Olympic AI PTIT 2026 - Vòng sơ loại

Kho lưu trữ tổng hợp giải pháp và chuỗi bài giảng thực nghiệm cho các bài toán trong vòng sơ loại cuộc thi Olympic AI PTIT 2026. Mỗi bài toán được tổ chức hoàn toàn độc lập về cấu trúc dữ liệu, kiến trúc mô hình và quy trình kiểm định.

---

## Danh mục các bài toán trong kỳ thi

| Bài toán | Nội dung trọng tâm | Tài liệu & Notebooks |
|---|---|---|
| **1. Kẻ mạo danh** | Phân loại nhị phân theo cặp ảnh đối đầu: 32 đặc trưng thống kê số học, mô hình tuyến tính đối xứng, mạng CNN thích nghi, cơ chế cắt vùng nguyên bản Native 1:1, sàng lọc backbone với mốc cap48, xác nhận 3 seed, cổng Blend Selection Gate +0.005, chốt quyết định trước khi mở private và tạo tệp nộp bài | [Hướng dẫn chi tiết bài toán Kẻ mạo danh](KeMaoDanh/README.md) |
| **2. Bài toán tiếp theo** | Kiến trúc mô hình và chuỗi thực nghiệm chuyên sâu cho nhiệm vụ thứ hai trong khuôn khổ vòng sơ loại | Theo dõi cập nhật |

```text
.
├── README.md
├── .gitignore
├── assets/                       # Sơ đồ kiến trúc tổng quan của hệ thống
└── KeMaoDanh/
    ├── README.md                 # Hướng dẫn chi tiết, cài đặt môi trường bằng uv
    ├── notebooks/                # Chuỗi 13 notebook thực hành từ cơ bản đến mở rộng
    ├── scripts/                  # Kịch bản kiểm tra tĩnh, xuất bản và pipeline kế thừa
    ├── src/kmd/                  # Các mô-đun mã nguồn lõi (extractor, models, trainer, pipeline, suite, gate, decision_flow, finalization, teaching)
    ├── configs/                  # Các tệp cấu hình JSON và danh sách phân chia development_split.csv
    ├── docs/                     # Tài liệu đặc tả bài toán, kiến trúc pipeline và bảng ánh xạ 27 trang báo cáo
    ├── data/                     # Dữ liệu hình ảnh (được bỏ qua không commit lên Git)
    ├── artifacts/models/         # Checkpoint và artifacts lưu theo phiên làm việc (không commit)
    └── outputs/                  # Tệp dự đoán xác suất và tệp nộp bài submission.csv (không commit)
```

Sơ đồ tổng quan luồng xử lý: [Pipeline Kẻ mạo danh](assets/README.md).

---

## Điều hướng nhanh cho Bài toán Kẻ mạo danh

Chuỗi 13 notebook thực hành tuân thủ chu trình khoa học: `Quan sát -> Đặt câu hỏi -> Can thiệp kiểm soát -> Diễn giải kết quả -> Quyết định tiếp theo`. Lộ trình gồm hai phần: mạch học tuần tự chính (00 đến 07) và các phụ lục chuyên đề mở rộng (A đến D), cùng notebook điều phối tổng thể.

### 1. Mạch học tuần tự chính (Bắt buộc: Bài 00 đến 07)
- [00. Bản chất bài toán và Cấu trúc dữ liệu](KeMaoDanh/notebooks/00_problem_and_data.ipynb): Cấu trúc cặp ảnh đối đầu, phân rã nhãn `[1-y, y]`, bảo toàn mã chuỗi `pair_id` có số 0 đầu, phân chia 800 cặp development (3 fold) và 200 cặp holdout độc lập, thước đo Macro-F1 so với Accuracy.
- [01. Khám phá dữ liệu và thử quy tắc đơn giản](KeMaoDanh/notebooks/01_eda_and_baseline.ipynb): Khám phá dữ liệu trên phần train của Fold 0, quy tắc chọn tệp dung lượng nhẹ hơn, ma trận nhầm lẫn và nhận diện các phản ví dụ thực tế.
- [02. Đặc trưng thủ công và Logistic Regression đối xứng](KeMaoDanh/notebooks/02_features_and_lr.ipynb): 32 đặc trưng thống kê thuộc 4 nhóm, nguyên lý vector hiệu $x = f_1 - f_0$ bảo toàn tính đối xứng $p(-x) = 1 - p(x)$, chuẩn hóa không trừ tâm, đánh giá bảng 9 probe LR trên 3-fold OOF.
- [03. Mạng CNN Pretrained và Tinh chỉnh](KeMaoDanh/notebooks/03_pretrained_and_finetune.ipynb): Phản ví dụ mảng 2x2 về giới hạn của thống kê phẳng, DenseNet-121 + Linear head xuất điểm từng ảnh, quy tắc BatchNorm eval mode, head warmup và đối chứng Frozen so với Full fine-tuning trên center60 Fold 0.
- [04. Vùng cắt Native và Dấu vết nội suy](KeMaoDanh/notebooks/04_native_and_resampling.ipynb): Đối chứng cùng vùng nhìn 4 crop (224x224) nhưng khác thao tác điểm ảnh (Native giữ nguyên 1:1 so với Resampled co 112 rồi phóng 224), multi-crop mean logit pooling 8 logit, ngân sách 19 epoch cố định trên Fold 0.
- [05. Sàng lọc kích thước, vùng nhìn và backbone rồi chọn mô hình đơn](KeMaoDanh/notebooks/05_backbone_and_selection.ipynb): Thiết lập mốc đối chứng `center60_cap48` (trần 48 epoch, microbatch 8, batch hiệu dụng 24) để tách biệt kích thước và backbone; sàng lọc 6 cấu hình tại seed 20260917; lập shortlist 2 challenger dẫn đầu; xác nhận shortlist cùng 3 mốc qua 3 seed (20260917, 20260918, 20260919); chọn mô hình đơn winner động có F1 trung bình cao nhất (không gán cứng B2).
- [06. Lỗi chồng lấp với native và blend 50/50 theo mô hình đơn đã chọn](KeMaoDanh/notebooks/06_errors_and_blend.ipynb): Căn 800 cặp OOF seed 20260917 giữa winner và native; phân tích ma trận 4 nhóm đúng/sai (cơ hội sửa lỗi so với rủi ro làm hỏng); thử blend xác suất 50/50 cố định; cổng Blend Selection Gate đòi hỏi mức tăng trung bình chưa làm tròn qua 3 seed đạt tối thiểu +0.0050 Macro-F1; tự động bỏ qua self-blend nếu winner là native.
- [07. Từ quyết định phát triển đến bài nộp](KeMaoDanh/notebooks/07_private_and_submission.ipynb): Khóa quyết định phát triển vào `review_freeze.json` trước khi đọc nhãn private; đối chiếu private nếu có cấu hình `PRIVATE_ROOT`, `PRIVATE_LABELS`, `PRIVATE_LABEL_SOURCE`; hai chiến lược nộp bài: mặc định `fold_ensemble` (trung bình xác suất 3 fold có tập train chồng lặp) hoặc tùy chọn `refit_all` (huấn luyện lại trên toàn bộ 1.000 cặp có nhãn với ngân sách epoch chốt từ development, tạo 1 checkpoint cho mô hình đơn hoặc 2 checkpoint nếu là blend); xuất `submission.csv`.

### 2. Các Phụ lục Chuyên đề Mở rộng (Tùy chọn: Extension A đến D)
- [Phụ lục A: Biểu diễn đầu vào và Cơ chế tinh chỉnh](KeMaoDanh/notebooks/extension_a_representations.ipynb): Frozen Embedding kết hợp Logistic Regression; đối chứng Partial so với Full fine-tuning; cơ chế Top-2 pooling so với Mean pooling; mạng ResNet-18 nhận ảnh màu RGB so với Gaussian residual và NPR residual trên 3-fold OOF.
- [Phụ lục B: Hàm mục tiêu huấn luyện và Kỹ thuật ghép cặp lại](KeMaoDanh/notebooks/extension_b_objectives_and_repair.ipynb): Khảo sát 4 biến thể hàm mất mát ở điểm kết thúc 19 epoch (image loss, pairwise loss, mixed loss, re-pairing) trên DenseNet-121 3-fold OOF.
- [Phụ lục C: Khảo sát quy mô dữ liệu dưới ngân sách cập nhật cố định](KeMaoDanh/notebooks/extension_c_data_scaling.ipynb): Đánh giá EfficientNet-B2 288px trên 25%, 50%, 100% dữ liệu train dưới cùng ngân sách cố định 391 gradient updates terminal (smoke: 4 updates).
- [Phụ lục D: Tăng cường dữ liệu bằng phép co giãn ngẫu nhiên](KeMaoDanh/notebooks/extension_d_resize_augmentation.ipynb): Đối chứng trực tiếp mô hình chuẩn `b2` với mô hình áp dụng tăng cường co giãn ngẫu nhiên `b2_resize` (90-100%) trên 3-fold OOF (giữ nguyên B2 làm đối chứng đã đăng ký).

### 3. Bộ điều phối Tổng thể và Giao thức Thực nghiệm
- [pipeline_end_to_end.ipynb](KeMaoDanh/notebooks/pipeline_end_to_end.ipynb): Thực thi toàn bộ quy trình theo giao thức hiện hành từ dữ liệu đến submission. Thiết lập `PROFILE = 'baseline'` cho Logistic Regression nhanh trên CPU/GPU, hoặc `PROFILE = 'full'` để chạy sàng lọc CNN, xác nhận 3 seed, thử blend với native, khóa qua `freeze_review`, hỗ trợ các phụ lục khi đặt `KMD_INCLUDE_EXTENSIONS=1`, và xuất kết quả theo chiến lược `fold_ensemble` hoặc `refit_all`.
- *Lưu ý về CLI cũ:* Kịch bản dòng lệnh `scripts/run_pipeline.py` là giao thức kế thừa (legacy protocol), thực thi 5 kiến trúc cố định và mặc định xuất B2 theo báo cáo lịch sử. Kịch bản này không áp dụng luồng quyết định động mới (sàng lọc 8 cấu hình, shortlist 3 seed, cổng blend động, khóa freeze_review, refit 1.000 cặp). Để vận hành đầy đủ giao thức mới nhất, khuyến nghị sử dụng `pipeline_end_to_end.ipynb`.

---

## Hướng dẫn Khởi chạy Nhanh

Cài đặt trình quản lý `uv` và khởi chạy môi trường dòng lệnh hoặc Jupyter Lab từ thư mục `KeMaoDanh/`:

```bash
cd KeMaoDanh

# Cài đặt thư viện (phiên bản thử nghiệm chuẩn Python 3.11):
uv sync --locked --extra cu128
# (Hoặc máy CPU: uv sync --locked --extra cpu)

# Khởi động Jupyter Lab:
uv run --locked --extra cu128 jupyter lab

# Kiểm tra cấu trúc tĩnh của 13 notebook:
python scripts/check_structure.py

# Kiểm tra nghiêm ngặt toàn bộ cell code đã được thực thi tuần tự 1..N và không có lỗi:
python scripts/check_structure.py --require-executed
```
