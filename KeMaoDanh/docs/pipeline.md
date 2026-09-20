# Kiến trúc Pipeline và Luồng thực nghiệm Kẻ mạo danh

Tài liệu này mô tả kiến trúc phần mềm, cấu trúc các mô-đun lõi trong gói `kmd`, lộ trình học tập qua 13 notebook thực hành và các quy trình vận hành pipeline.

---

## 1. Cấu trúc các mô-đun mã nguồn trong `src/kmd/`

Gói thư viện `kmd` chứa mã nguồn dùng chung cho toàn bộ chuỗi thực nghiệm:

| Mô-đun | Đường dẫn tệp | Chức năng chính |
|---|---|---|
| `kmd.core` | `src/kmd/core.py` | Đọc ghi tệp CSV/JSON bảo toàn kiểu dữ liệu, tính điểm Macro-F1 (`metric`), phân chia fold (`split_fold`), kiểm tra toàn vẹn băm SHA-256 (`sha256`) và xuất tệp nộp bài (`export_submission`). |
| `kmd.config` | `src/kmd/config.py` | Lớp cấu hình `Config` định nghĩa toàn bộ siêu tham số: kích thước ảnh, vùng nhìn, tốc độ học backbone và head, warmup, số epoch tối đa, microbatch, effective_batch. |
| `kmd.presets` | `src/kmd/presets.py` | Định nghĩa các cấu hình mẫu đã đăng ký sẵn (`get_preset_config`, `list_presets`) phục vụ đối chứng có kiểm soát. |
| `kmd.extractor` | `src/kmd/extractor.py` | Trích xuất 32 đặc trưng thống kê số học (dung lượng, màu sắc, texture, tương phản giữa/biên) và danh mục 9 phép thử cắt bỏ đặc trưng (`list_lr_ablations`). |
| `kmd.data` | `src/kmd/data.py` | Lớp `Pairs` kế thừa PyTorch Dataset, nạp ảnh theo cặp, hỗ trợ biến đổi tăng cường dữ liệu và cơ chế tạo 4 vùng cắt nguyên bản (Native crops 1:1). |
| `kmd.models` | `src/kmd/models.py` | Khởi tạo các kiến trúc backbone (DenseNet-121, EfficientNet-B0, EfficientNet-B2, ResNet-18), lớp phân loại tuyến tính Linear head, chế độ huấn luyện (`train_mode`) và hàm mục tiêu BCE (`objective`). |
| `kmd.training` | `src/kmd/training.py` | Vòng lặp tối ưu hóa, hàm cập nhật gradient đơn bước (`train_step`), cơ chế tích lũy gradient (gradient accumulation) và giới hạn chuẩn đạo hàm (gradient clipping). |
| `kmd.pipeline` | `src/kmd/pipeline.py` | Quản lý phiên làm việc (`start_session`), huấn luyện kiểm định chéo (`fit_lr_cv`, `train_cnn_cv`), tập hợp dự đoán OOF (`compare_oof`), kết hợp mô hình (`blend`) và suy diễn test an toàn. |
| `kmd.suite` | `src/kmd/suite.py` | Các bộ thực nghiệm đối chứng chuyên biệt: `run_lr_suite`, `run_crop_comparison_suite`, `run_loss_comparison_suite`, `run_residual_suite`, `run_data_amount_suite`. |
| `kmd.gate` | `src/kmd/gate.py` | Hàm tính toán độ chênh lệch Macro-F1 chưa làm tròn qua 3 seed và kiểm tra ngưỡng tăng trưởng tối thiểu $+0.0050$. |
| `kmd.decision_flow` | `src/kmd/decision_flow.py` | Luồng quyết định thực nghiệm: sàng lọc 6 cấu hình tại seed 20260917 (`run_backbone_screening_suite`), xác nhận shortlist qua 3 seed (`run_single_confirmation_suite`), và đánh giá blend động với mô hình native (`run_selected_blend_suite`). |
| `kmd.finalization` | `src/kmd/finalization.py` | Khóa quyết định trước khi mở private (`freeze_review`), suy diễn dự đoán trên private (`predict_frozen_comparison`), chấm điểm private (`score_private`), và huấn luyện lại trên toàn bộ 1.000 cặp (`refit_all`, `infer_refit`). |
| `kmd.teaching` | `src/kmd/teaching.py` | Tiện ích trực quan hóa: hiển thị cặp ảnh, vẽ lớp phủ vùng cắt (`plot_crop_overlay_512`), phóng to điểm ảnh (`plot_patch_zoom_comparison`), và tra cứu ca dự đoán OOF (`view_oof_case_by_id`). |

---

## 2. Danh mục 13 Notebook Thực hành

Toàn bộ chuỗi bài học được chia thành hai mạch: 8 bài học tuần tự chính (00 đến 07) và 4 bài học phụ lục mở rộng (Extension A đến D), cùng một notebook điều phối tổng thể.

### A. Mạch học tuần tự chính (Bài 00 đến 07)
1. **[00_problem_and_data.ipynb](../notebooks/00_problem_and_data.ipynb) - Bản chất bài toán và Cấu trúc dữ liệu:**
   - Cấu trúc cặp ảnh đối đầu và quy ước nhãn vị trí `fake_position` thuộc {0, 1}.
   - Giữ nguyên vẹn mã chuỗi `pair_id` có các chữ số 0 ở đầu.
   - Phân chia 800 cặp phát triển (3 fold cân bằng) và bảo toàn 200 cặp holdout độc lập.
   - Phân tích sự khác biệt giữa thước đo Macro-F1 và Accuracy.
2. **[01_eda_and_baseline.ipynb](../notebooks/01_eda_and_baseline.ipynb) - Khám phá dữ liệu và thử quy tắc đơn giản:**
   - Khám phá dữ liệu trên phần train của Fold 0 nhằm tránh rò rỉ thông tin kiểm định.
   - Quy tắc chọn tệp có dung lượng bytes nhẹ hơn và ma trận nhầm lẫn tương ứng.
   - Phân tích các phản ví dụ thực tế nơi quy tắc dung lượng bị thất bại.
3. **[02_features_and_lr.ipynb](../notebooks/02_features_and_lr.ipynb) - Kết hợp các phép đo bằng Logistic Regression:**
   - Khám phá 32 đặc trưng thống kê và quy ước vector hiệu $x = f_1 - f_0$.
   - Chứng minh và kiểm tra tính đối xứng $p(-x) = 1 - p(x)$ (chuẩn hóa không trừ tâm, không hệ số chặn).
   - Đánh giá đầy đủ bảng 9 probe LR trên 3-fold OOF đầy đủ qua `run_lr_suite`.
4. **[03_pretrained_and_finetune.ipynb](../notebooks/03_pretrained_and_finetune.ipynb) - Mạng CNN Pretrained và Fine-tuning:**
   - Phản ví dụ mảng 2x2 về giới hạn của thống kê phẳng khi mất thông tin bố cục không gian.
   - Kiến trúc DenseNet-121 kết hợp lớp Linear head xuất điểm số từng ảnh.
   - Nguyên tắc BatchNorm eval mode và giai đoạn Head-only warmup.
   - Trực quan hóa bước học `train_step` và đối chứng Frozen so với Full fine-tuning trên center60 Fold 0.
5. **[04_native_and_resampling.ipynb](../notebooks/04_native_and_resampling.ipynb) - Vùng cắt Native và Dấu vết nội suy:**
   - Sơ đồ lớp phủ vùng Center60 và 4 vùng cắt Native (224x224 giữ nguyên tỷ lệ 1:1).
   - So sánh phóng to ô 64x64 pixel gốc so với pixel qua phép nội suy bilinear.
   - Phép đối chứng kiểm soát: Native so với Resampled (cùng vùng nhìn 4 crop, khác thao tác điểm ảnh).
   - Mỗi ảnh có 4 logit, lấy trung bình riêng thành điểm của ảnh; hiệu hai điểm đi qua sigmoid để dự đoán vị trí ảnh giả.
6. **[05_backbone_and_selection.ipynb](../notebooks/05_backbone_and_selection.ipynb) - Sàng lọc kích thước, vùng nhìn và backbone:**
   - Mốc đối chứng chuẩn `center60_cap48` (trần 48 epoch, microbatch 8, batch hiệu dụng 24).
   - Sàng lọc 6 cấu hình tại seed `20260917` (kèm 2 mốc tham chiếu `center60` và `native`).
   - Lập shortlist gồm 2 challenger dẫn đầu về Macro-F1 trên tập phát triển.
   - Xác nhận shortlist cùng 3 mốc trên 3 seed (`20260917`, `20260918`, `20260919`).
   - Lựa chọn mô hình đơn chiến thắng (winner) động theo Macro-F1 trung bình cao nhất (không gán cứng B2).
7. **[06_errors_and_blend.ipynb](../notebooks/06_errors_and_blend.ipynb) - Phân tích lỗi và Blend 50/50:**
   - Căn 800 cặp OOF seed `20260917` giữa winner và mô hình `native`.
   - Ma trận phân bổ 4 góc: đếm số cặp cùng đúng, cùng sai, chỉ winner đúng, chỉ native đúng.
   - Nhận diện cơ hội sửa sai tiềm năng và nguy cơ làm hỏng của phép kết hợp.
   - Thử nghiệm kết hợp xác suất cố định 50/50: $p_{\text{blend}} = 0.5 \cdot p_{\text{winner}} + 0.5 \cdot p_{\text{native}}$.
   - Áp dụng cổng Blend Selection Gate: yêu cầu mức tăng Macro-F1 trung bình chưa làm tròn qua 3 seed phải đạt ít nhất $+0.0050$.
   - Bỏ qua việc kết hợp nếu mô hình đơn được chọn chính là `native`.
8. **[07_private_and_submission.ipynb](../notebooks/07_private_and_submission.ipynb) - Từ quyết định phát triển đến bài nộp:**
   - Khóa toàn bộ quyết định vào tệp `review_freeze.json` trước khi đọc dữ liệu private.
   - Dự đoán trước, chấm điểm sau: suy diễn toàn bộ danh sách đã khóa rồi mới mở nhãn kiểm tra nếu có cấu hình.
   - Hai chiến lược nộp bài rõ ràng:
     - `fold_ensemble` (mặc định): lấy trung bình xác suất từ 3 mô hình fold có tập huấn luyện chồng lặp.
     - `refit_all` (tùy chọn): huấn luyện lại trên toàn bộ 1.000 cặp có nhãn với ngân sách epoch chốt từ development.
   - Tạo và kiểm tra tệp nộp bài `submission.csv`.

### B. Phụ lục chuyên đề mở rộng (Extension A đến D)
9. **[extension_a_representations.ipynb](../notebooks/extension_a_representations.ipynb) - Biểu diễn đầu vào và Cơ chế tinh chỉnh:**
   - Frozen Embedding 1.024 chiều từ DenseNet-121 kết hợp với bộ phân loại lồi Logistic Regression.
   - Đánh giá 3-fold OOF cho Partial fine-tuning so với Full fine-tuning.
   - So sánh cơ chế Top-2 pooling với Mean pooling trên 4 vùng cắt.
   - Đánh giá ResNet-18 khi nhận ảnh màu RGB so với bản đồ phần dư Gaussian residual và NPR residual.
10. **[extension_b_objectives_and_repair.ipynb](../notebooks/extension_b_objectives_and_repair.ipynb) - Hàm mục tiêu và Ghép cặp lại:**
    - Khảo sát 4 biến thể hàm mất mát ở điểm kết thúc 19 epoch terminal: image loss, pairwise loss, mixed loss, re-pairing.
    - Phân tích ưu nhược điểm của kỹ thuật ghép cặp lại (re-pairing) so với cặp đối đầu tự nhiên.
11. **[extension_c_data_scaling.ipynb](../notebooks/extension_c_data_scaling.ipynb) - Khảo sát quy mô dữ liệu:**
    - Đánh giá 3-fold OOF trên 25%, 50% và 100% dữ liệu huấn luyện dưới cùng ngân sách cố định 391 gradient updates terminal (smoke: 4 updates) trên EfficientNet-B2 288px.
    - Giữ nguyên 100% tập kiểm định 800 mẫu để bảo đảm tính so sánh công bằng.
12. **[extension_d_resize_augmentation.ipynb](../notebooks/extension_d_resize_augmentation.ipynb) - Tăng cường dữ liệu bằng phép co giãn ngẫu nhiên:**
    - Khảo sát ảnh hưởng của phép co giãn ngẫu nhiên Resize Augmentation (90-100%) trong lúc huấn luyện trên EfficientNet-B2 288px.
    - Đối chứng trực tiếp mô hình chuẩn `b2` với mô hình `b2_resize` trên toàn bộ 3-fold OOF (giữ B2 làm đối chứng đã đăng ký).

### C. Bộ điều phối tổng thể
13. **[pipeline_end_to_end.ipynb](../notebooks/pipeline_end_to_end.ipynb) - Điều phối tự động toàn diện:**
    - Chạy từ dữ liệu thật đến bài nộp theo đúng giao thức hiện hành.
    - Hỗ trợ `PROFILE = 'baseline'` (Logistic Regression nhanh trên CPU/GPU).
    - Hỗ trợ `PROFILE = 'full'` (chạy sàng lọc, xác nhận 3 seed, thử blend với native, khóa qua `freeze_review`, hỗ trợ phụ lục khi đặt `KMD_INCLUDE_EXTENSIONS=1`, và xuất bài theo `fold_ensemble` hoặc `refit_all`).

---

## 3. Quy trình Ra Quyết định và Phương pháp luận

Quy trình thực nghiệm vận hành theo nguyên tắc khoa học có kiểm soát:

```text
Quan sát dữ liệu
       │
       ▼
Đặt câu hỏi cụ thể (Kích thước, Vùng nhìn, Kiến trúc)
       │
       ▼
Can thiệp kiểm soát (Mốc center60_cap48, Giữ nguyên siêu tham số khác)
       │
       ▼
Sàng lọc tại seed chính -> Shortlist 2 ứng viên -> Xác nhận 3 seed
       │
       ▼
Chọn mô hình đơn winner có Macro-F1 trung bình cao nhất
       │
       ▼
Phân tích ma trận lỗi với mô hình Native (4 nhóm)
       │
       ▼
Thử Blend 50/50 qua cổng Selection Gate (Ngưỡng tăng trưởng >= +0.0050)
       │
       ▼
Khóa quyết định vào review_freeze.json (Trước khi mở nhãn Private)
       │
       ▼
Xuất bài nộp theo chiến lược: fold_ensemble (mặc định) hoặc refit_all
```

- **Mốc `center60_cap48`:** Sử dụng DenseNet-121, view center60, size 224, nhưng đặt trần 48 epoch, microbatch 8, tích lũy gradient để giữ batch hiệu dụng 24. Cấu hình này giúp cô lập biến số khi so sánh với các mô hình 288px và EfficientNet.
- **Xác nhận 3 Seed:** Ba hạt giống `20260917`, `20260918`, `20260919` đo lường mức độ nhạy cảm do khởi tạo trọng số và thứ tự xáo trộn batch trên cùng 800 cặp phát triển. Seed nộp bài được chọn trước là `20260917`.
- **Cổng Blend Selection Gate (+0.0050):** Chỉ chấp nhận đưa thêm mô hình thứ hai vào đường ống khi mức tăng trưởng trung bình chưa làm tròn qua 3 seed đạt tối thiểu $+0.0050$ Macro-F1 so với mô hình đơn. Nếu mô hình đơn được chọn là `native`, hệ thống tự động bỏ qua việc kết hợp với chính nó.
- **Khóa trước khi xem private:** Quyết định lựa chọn mô hình được ghi nhận vào tệp `review_freeze.json` trước khi thực hiện các phép đo trên dữ liệu private, nhằm giữ tính khách quan giữa tập phát triển và tập kiểm tra.
- **Kỹ thuật về mô hình fold và refit:**
  - Ba mô hình fold có tập huấn luyện chồng lặp (mỗi fold dùng hai phần ba dữ liệu phát triển, phần train của các fold giao nhau), không phải các mô hình độc lập. Việc kết hợp 3 fold là cách tận dụng cả 3 mạng đã huấn luyện, không đảm bảo tự động giảm phương sai trong mọi trường hợp.
  - Chiến lược `refit_all` huấn luyện lại từ pretrained trên toàn bộ 1.000 cặp có nhãn với ngân sách epoch chốt từ development. Nếu quyết định nộp là mô hình đơn thì tạo 1 checkpoint; nếu quyết định nộp là blend (`blend_selected_native`), `refit_all` huấn luyện riêng từng thành phần (1 checkpoint cho nhánh winner và 1 checkpoint cho nhánh native, tổng cộng 2 checkpoint) rồi lấy trung bình xác suất khi suy diễn test.

---

## 4. Chạy toàn bộ pipeline

Notebook [`pipeline_end_to_end.ipynb`](../notebooks/pipeline_end_to_end.ipynb) là điểm vào chính để chạy giao thức hiện hành từ dữ liệu đến tệp nộp bài. Notebook gọi trực tiếp các hàm trong `kmd.decision_flow` và `kmd.finalization`, đồng thời lưu lại các quyết định chọn mô hình trước khi đối chiếu private.

Hai script độc lập được giữ lại vì phục vụ trực tiếp cho người học:

- `scripts/prepare_official_dataset.py`: giải nén và kiểm tra cấu trúc bộ dữ liệu của lớp.
- `scripts/evaluate_predictions.py`: tính Macro-F1 cho tệp dự đoán khi có nhãn tham chiếu.
