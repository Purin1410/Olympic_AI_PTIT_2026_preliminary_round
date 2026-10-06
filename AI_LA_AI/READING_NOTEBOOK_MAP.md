# Đối chiếu Reading và năm notebook

Nguồn bài đọc: `OlympicAI_2026/topic_ai_la_ai/reading.tex` trong workspace bài giảng, chỉ năm section được file này input: `00_intro`, `01_analysis`, `02_build`, `03_evaluation`, `03_appendix`. Các tệp `02_foundations`, `03_build`, `04_evaluation` cùng thư mục là bản khác, không dùng làm nguồn đối chiếu lần này.

Số cell trong bảng đếm từ 1, gồm cả Markdown và code. Cell được chỉ đến là lời dẫn; phần code nằm ngay sau hoặc trong nhóm cell kế tiếp.

| Nội dung Reading | Notebook/cell | Mã và phép kiểm tra |
|---|---|---|
| II.1: dữ liệu và quy ước nhãn | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 5 | data.py: load_train_manifest, load_fold_split |
| II.1.2: 5-fold và kiểm tra rò rỉ | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 17 | StratifiedKFold minh họa; bảng cố định; hash ảnh qua fold |
| II.1.3: confusion matrix, Precision/Recall/F1 | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 19; [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 21 | macro_f1_from_counts; confusion_matrix; precision_recall_fscore_support |
| II.2.1: thống kê và ngưỡng dung lượng | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 12; [NB0](notebooks/00_pipeline_end_to_end.ipynb), cell 16 | teaching.py: image_census, choose_size_rule |
| II.2.2: xám/màu | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 15; [NB0](notebooks/00_pipeline_end_to_end.ipynb), cell 19 | crosstab; quy tắc xám→Real; ảnh mẫu 4 nhóm |
| II.2.3–4: crop và phổ FFT | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 45; [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 47 | transforms.py: native_view, resampled_view; visuals.py: plot_spectra |
| III.1: CNN2 và một bước học | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 23 | teaching.py: BaselineCNN2 (Conv-BN-ReLU-Pool ×2); CE/backward/step |
| III.2: frozen backbone và fine-tuning | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 43 | build_resnet34; fit_fold; RUN_E1 gồm 3 fits |
| III.3: Same-FOV trên folds 0,1 | [NB1](notebooks/01_eda_baseline_geometry.ipynb), cell 51 | RUN_E2 gồm 4 fits; ghép từng ảnh, điểm từng fold và pooled |
| III.4: Gaussian, residual, gain/offset | [NB2](notebooks/02_forensic_specialist.ipynb), cell 7; [NB2](notebooks/02_forensic_specialist.ipynb), cell 11 | forensics.py: gaussian_blur_rgb; highpass_view trong notebook |
| III.5.1: Dataset/DataLoader | [NB2](notebooks/02_forensic_specialist.ipynb), cell 14; [NB0](notebooks/00_pipeline_end_to_end.ipynb), cell 30 | data.py: FaceDataset, make_loader được nhúng thành cell |
| III.5.2: optimizer, loss, AMP và accumulation | [NB2](notebooks/02_forensic_specialist.ipynb), cell 21 | engine.py: train_one_epoch, _optimizer, _view_batch |
| III.5.2: validation, TTA và checkpoint cuối | [NB2](notebooks/02_forensic_specialist.ipynb), cell 23; [NB0](notebooks/00_pipeline_end_to_end.ipynb), cell 41 | engine.py: _validation_predictions, fit_fold, load_run, predict |
| II.1.2/IV.1: tạo OOF hai nhánh | [NB0](notebooks/00_pipeline_end_to_end.ipynb), cell 49 | RUN_OOF: 5 folds × 2 nhánh; kiểm tra mỗi ảnh đúng một validation |
| III.6: mean, overlap, fixes/breaks | [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 9; [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 11 | predictions.py: align_predictions; phép tính trực tiếp theo từng ảnh |
| IV.2: lỗi còn lại và xám/màu | [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 15 | phân rã sai chung/chưa sửa/làm hỏng; ảnh và điểm theo nhóm |
| III.7: ngưỡng và thiên lệch chọn/chấm cùng tập | [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 17; [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 19 | teaching.py: crossfit_threshold; bảng cố định/in-sample/cross-fit |
| III.8: dự đoán và submission | [NB0](notebooks/00_pipeline_end_to_end.ipynb), cell 55; [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 30 | engine.py: predict; submission.py: export/validate và kiểm tra nội dung ZIP |
| Phụ lục C: stacking | [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 22; [NB3](notebooks/03_ensemble_threshold_submission.ipynb), cell 25 | crossfit_stack, crossfit_logit_stack; 6 nguồn OOF có hash; saved-vs-refit |
| IV.3/Phụ lục D: Wavelet | [NB4](notebooks/04_negative_results_and_ablation.ipynb), cell 8; [NB4](notebooks/04_negative_results_and_ablation.ipynb), cell 39 | ablation.py: haar_view, paired_report |
| Phụ lục D: sample weighting và subgroup | [NB4](notebooks/04_negative_results_and_ablation.ipynb), cell 16; [NB4](notebooks/04_negative_results_and_ablation.ipynb), cell 41 | image_features, low_edge_weights; weighted CE trong train_one_epoch |
| Phụ lục D: chọn ngưỡng theo nhóm | [NB4](notebooks/04_negative_results_and_ablation.ipynb), cell 43; [NB4](notebooks/04_negative_results_and_ablation.ipynb), cell 47 | calibration_split, choose_group_thresholds; Legal7 historical replay |

## Các ranh giới cần giữ khi giảng

- EDA tính lại trên ảnh thật; không điền số có sẵn trong Reading. Histogram dung lượng, ngưỡng trên cả tập và ngưỡng chọn từ train fold có phạm vi khác nhau.
- Reading mô tả CNN2 nhận ảnh 64 nhưng caption bảng E1 ghi chung Resize 224. Notebook chọn CNN2 64 và ResNet 224, ghi rõ tại bảng điều kiện. Chưa thể gọi recipe mới là tái tạo E1 lịch sử.
- Preset mặc định có smoothing 0, TTA False và accumulation 2; bảng OOF lịch sử dùng smoothing 0,03/flip TTA. Scheduler hiện có eta_min mặc định 0, trong khi Reading ghi 1e-6. Các khác biệt này được giữ để không tự đổi recipe đã phát hành; khi muốn tái tạo phải chốt cấu hình từ run nguồn.
- Các cell E1, E2 và RUN_OOF đã có mã huấn luyện, nhưng các lượt GPU đầy đủ mới chưa được chạy trong lần bổ sung này.
- Gói ZIP bài giảng có đủ sáu nguồn OOF của Stack6 đã ghép và xác minh hash; bản GitHub có code và cần chép thêm hai tệp tùy chọn vào `reference_artifacts/` trước khi bật `RUN_STACK6`. Tên `center70`/`highpass` trong run Stack6 trỏ tới checkpoint cũ được liệt kê trong provenance, không phải tự động lấy RGB/HP Native358 của phần chính. Mã fit LR mới theo mô tả standardized logits; bảng saved-vs-refit giữ nguyên mọi chênh lệch số.
- NB4 train dùng Haar xếp bốn góc, Sobel và ResNet18 cùng backbone; bundle reference dùng cấu hình Wavelet/Laplacian lịch sử. Ngưỡng của Legal7 được replay từ bundle, tách khỏi bài tập calibration RGB.
- Đoạn GroupDRO/XRM cuối Phụ lục D là một ghi chú thử nghiệm lịch sử, không phải một bài thực hành trong năm notebook. Không có triển khai XRM hoặc lượt tái tạo mới trong gói này.
- `demo_submission.zip` ở NB3 minh họa định dạng. Bài nộp từ ảnh test thuộc NB0. Nhãn Private Test không dùng để chọn mô hình hay ngưỡng.

## Bảo trì source

Các cell có `# ailaai-source` chứa thân hàm Python thật; builder kiểm tra chúng với file gốc trước khi sinh notebook. Thay đổi trực tiếp cell phù hợp cho bài tập. Để duy trì bộ tài liệu, sửa `notebook_sources/*.md` hoặc hàm ở `src/ailaai/`, sau đó chạy `scripts/build_notebooks.py --write` và `--check`.

Các hàm quản lý checkpoint/config/hash vẫn được import từ package. Phần EDA, mô hình, Dataset/DataLoader, tiền xử lý, vòng học, đánh giá, ghép xác suất, ngưỡng và submission đã có code để đọc trong notebook.
