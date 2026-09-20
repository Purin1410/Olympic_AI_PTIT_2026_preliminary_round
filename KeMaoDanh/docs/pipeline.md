# Pipeline Thực nghiệm Kẻ mạo danh

Tài liệu này mô tả toàn bộ luồng xử lý từ dữ liệu thô, trích xuất đặc trưng, huấn luyện mô hình học sâu đến tạo file submission trên tập test.

---

## 1. Chuỗi Bài học Thực nghiệm (7 Notebooks)

1. **`00_problem_and_data.ipynb` - Bài toán & Dữ liệu:**
   - Đọc manifest `pairs.csv`, kiểm tra schema, giữ nguyên định dạng chuỗi `pair_id` (bảo toàn các số 0 ở đầu).
   - Tách 800 cặp Development (chia 3-fold CV) và giữ nguyên 200 cặp Holdout độc lập.
   - So sánh định lượng chỉ số Macro-F1 vs Accuracy qua ví dụ số nhỏ.

2. **`01_features_and_lr.ipynb` - Đặc trưng Thủ công & Logistic Regression:**
   - Trực quan hóa 32 đặc trưng thống kê trên một cặp ảnh thật: dung lượng `log_bytes`, ảnh xám, gradient Sobel, Laplacian, residual, mask center vs border.
   - Lấy hiệu sai khác $x = \text{feat}(\text{image}_1) - \text{feat}(\text{image}_0)$.
   - Khám phá dữ liệu (EDA) nghiêm ngặt chỉ trên tập TRAIN của Fold 0.
   - Chuẩn hóa `StandardScaler(with_mean=False)` và `LogisticRegression(fit_intercept=False)` để bảo toàn tính đối xứng $p(-x) = 1 - p(x)$.

3. **`02_pretrained_and_finetune.ipynb` - Pretrained Backbone & Fine-tuning:**
   - Khởi tạo backbone DenseNet-121 và Linear classification head xuất 1 logit cho mỗi ảnh.
   - Đặt BatchNorm ở chế độ `eval()` và áp dụng warmup chỉ cập nhật head.
   - Trực quan hóa bước huấn luyện `train_step` dùng chung với core trainer.
   - So sánh trực tiếp trên Fold 0: Frozen Backbone vs Full Fine-tuning.

4. **`03_native_and_resampling.ipynb` - Native Crops & Dấu vết Nội suy:**
   - Kiểm tra ảnh hưởng của giảm độ phân giải rồi phóng lại cùng vùng ảnh; chưa giả định hướng tác động.
   - Trích xuất 4 patch Native $224 \times 224$ nguyên bản ở tỉ lệ 1:1.
   - Cơ chế Multi-crop: tính trung bình Logits qua 4 crops trước khi tính xác suất cặp.
   - Thí nghiệm đối chứng trên Fold 0: Native vs Resampled ($112 \rightarrow 224$).

5. **`04_errors_and_blend.ipynb` - Phân tích Lỗi & Kết hợp Mô hình:**
   - Huấn luyện EfficientNet-B2 độ phân giải $288 \times 288$ (nhận diện yếu tố nhiễu đồng thời).
   - Căn chỉnh dự đoán validation fold 0 giữa Native và B2.
   - Phân tích ma trận chồng chéo lỗi 4 góc (4-way error overlap).
   - Kết hợp mô hình (Blending) cố định: $p = 0.5 p_{\text{B2}} + 0.5 p_{\text{native}}$, đếm số lỗi sửa được vs số lỗi mới sinh ra.

6. **`05_validation_and_submission.ipynb` - Đánh giá 3-Fold OOF & Submission:**
   - Mở rộng đánh giá ra 3-fold Cross-Validation (toàn bộ 800 mẫu development Out-of-Fold).
   - Phân tích hiện tượng Optimistic Bias khi chọn checkpoint bằng validation.
   - Quy trình Inference an toàn: kiểm tra đủ 3 checkpoints, kiểm tra tách biệt test set, xuất file `submission.csv`.
   - Xử lý mượt mà khi không có tập test: thông báo trạng thái `no submission produced`.

7. **`pipeline_end_to_end.ipynb` - Bộ điều phối Tích hợp:**
   - Cấu hình `PROFILE = 'baseline'` (mặc định): Chạy nhanh 3-fold LR trên CPU/GPU.
   - Cấu hình `PROFILE = 'full'`: Huấn luyện toàn bộ LR + 5 cấu hình CNN x 3 fold + Blend.

---

## 2. Thực thi qua Dòng lệnh (CLI)

Sau khi cài đặt môi trường, từ thư mục `KeMaoDanh/`:

```bash
# Chạy Baseline Logistic Regression nhanh:
python scripts/run_pipeline.py --train-root "$DATA_ROOT" --profile baseline

# Chạy Full Pipeline (LR + 5 CNN x 3 folds + Blend + Test Submission):
python scripts/run_pipeline.py --train-root "$DATA_ROOT" --test-root "$TEST_ROOT" --profile full

# Đánh giá file dự đoán với nhãn có thật:
python scripts/evaluate_predictions.py --labels /path/to/ground_truth.csv --predictions outputs/run_id/submission.csv

# Kiểm tra cấu trúc static check:
python scripts/check_structure.py
```
