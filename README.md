# Olympic AI PTIT 2026 - Preliminary Round

Repository bài giải và chuỗi bài giảng thực nghiệm có định hướng sư phạm. Mỗi bài toán có cấu trúc thư mục, môi trường và pipeline thực nghiệm độc lập.

## Các bài toán

| Bài toán | Nội dung trọng tâm | Hướng dẫn & Notebooks |
|---|---|---|
| **Kẻ mạo danh** | Xác định ảnh giả trong một cặp: 32 đặc trưng thống kê, Logistic Regression, Fine-tuning CNN (DenseNet-121, EfficientNet-B2), Native Crops, phân tích lỗi và submission | [Hướng dẫn chi tiết](KeMaoDanh/README.md) |


```text
.
├── README.md
├── .gitignore
├── assets/                       # Sơ đồ kiến trúc chung, chia theo bài toán
└── KeMaoDanh/
    ├── README.md                 # Hướng dẫn chi tiết bài toán Kẻ mạo danh
    ├── notebooks/                # 7 notebook bài giảng thực nghiệm tuần tự
    ├── scripts/                  # Chạy pipeline dòng lệnh và kiểm tra cấu trúc
    ├── src/kmd/                  # Các module xử lý lõi (extractor, models, trainer, pipeline)
    ├── configs/                  # File cấu hình JSON và development_split.csv
    ├── docs/                     # Tài liệu bài toán và pipeline
    ├── data/                     # Dữ liệu ảnh
    ├── artifacts/models/         # Checkpoint và artifact của từng phiên chạy
    └── outputs/                  # File kết quả dự đoán và submission
```

Sơ đồ tổng quan: [Pipeline Kẻ mạo danh](assets/README.md).

## Điều hướng Nhanh cho Kẻ mạo danh

1. **Chuỗi Bài giảng Tuần tự (Learning Sequence):**
   - [00 - Bài toán & Cấu trúc Dữ liệu](KeMaoDanh/notebooks/00_problem_and_data.ipynb)
   - [01 - 32 Đặc trưng Thống kê & Logistic Regression](KeMaoDanh/notebooks/01_features_and_lr.ipynb)
   - [02 - Pretrained CNN & Fine-tuning](KeMaoDanh/notebooks/02_pretrained_and_finetune.ipynb)
   - [03 - Native Crops 1:1 & Dấu vết Nội suy](KeMaoDanh/notebooks/03_native_and_resampling.ipynb)
   - [04 - Phân tích Lỗi & Kết hợp Mô hình (Blend 50/50)](KeMaoDanh/notebooks/04_errors_and_blend.ipynb)
   - [05 - Đánh giá 3-Fold Full OOF & Tạo Submission](KeMaoDanh/notebooks/05_validation_and_submission.ipynb)

2. **Chạy nhanh Baseline (Fast Baseline):** Mở [pipeline_end_to_end.ipynb](KeMaoDanh/notebooks/pipeline_end_to_end.ipynb) với `PROFILE = 'baseline'` để chạy 3-fold LR trên CPU/GPU

3. **So sánh Toàn diện (Full Comparison):** Mở [pipeline_end_to_end.ipynb](KeMaoDanh/notebooks/pipeline_end_to_end.ipynb) với `PROFILE = 'full'` để huấn luyện toàn bộ 5 mô hình CNN x 3 fold và tạo bản nộp bài hoàn chỉnh.
