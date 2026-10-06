# AI LÀ AI

Tài liệu và code thực hành cho bài toán phân loại ảnh chân dung thật và ảnh do AI tạo ra. Trong notebook, học viên có thể đọc công thức, theo dõi từng bước xử lý và sửa các phần được dùng trong bài học. Thư mục `src/ailaai/` chứa code dùng chung cho việc đọc dữ liệu, huấn luyện, đánh giá và xuất tệp nộp bài.

## Cài đặt

Nếu chạy trên máy cá nhân, tạo môi trường Python và cài package bằng các lệnh sau:

```bash
cd Olympic_AI_PTIT_2026_preliminary_round/AI_LA_AI
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Nếu dùng Google Colab, mở một notebook trong thư mục `notebooks/`. Với bài 0, 1 và 2, chọn **Runtime → Change runtime type → T4 GPU**, rồi bấm **Run all**. Bài 3 và phụ lục 4 chạy được trên CPU. Cell đầu tự chuẩn bị môi trường; cell dữ liệu tự tải và nhận diện thư mục ảnh.


## Chuẩn bị dữ liệu

Bảng chia dữ liệu huấn luyện thành các fold nằm ở `assets/splits/train_folds.csv`.

Đặt ảnh và bảng thông tin theo cấu trúc sau:

| Dữ liệu | Thư mục ảnh | Bảng thông tin | Các cột cần có |
| --- | --- | --- | --- |
| Huấn luyện | `data/train/images/` | `data/train/manifest.csv` | `file_name,label` |
| Kiểm thử | `data/test/images/` | `data/test/manifest.csv` | `file_name` |

Cả hai bảng có thể thêm cột `path`. Nếu lưu dữ liệu ở nơi khác, đặt biến môi trường `AILAAI_DATA_ROOT` trỏ đến thư mục đó trước khi tạo workspace. Trên Colab, bạn có thể để mặc định: bộ nạp tự xử lý ZIP từ link trong bài, kể cả khi bên trong đã có lớp thư mục `data/`.

## Chọn notebook cho bài học

Bạn có thể học lần lượt từ bài 1 đến bài 3, rồi mở bài 0 để chạy toàn bộ quy trình.

| Notebook | Nội dung |
| --- | --- |
| `01_eda_baseline_geometry.ipynb` | Khám phá dữ liệu, làm quen với mô hình cơ sở và so sánh cách xử lý hình học của ảnh. |
| `02_forensic_specialist.ipynb` | Tìm hiểu bộ lọc High-pass và cách truyền hàm xử lý ảnh vào phần code huấn luyện dùng chung. |
| `03_ensemble_threshold_submission.ipynb` | Phân tích dự đoán trên tập validation, kết hợp hai mô hình và kiểm tra tệp ZIP minh họa. |
| `00_pipeline_end_to_end.ipynb` | Chạy hai nhánh RGB và High-pass, lấy trung bình xác suất và tạo `outputs/<run_id>/submission.zip`. |
| `04_negative_results_and_ablation.ipynb` | Đọc lại ba thử nghiệm chưa đem lại cải thiện như mong đợi và thảo luận khi nào nên dừng một hướng thử nghiệm. |

Phụ lục 4 có 18 cell và tính lại kết quả từ dữ liệu dự đoán đã đóng gói trong `data/negative_results/`. Bài này chạy trên CPU, chỉ cần `pandas` và `IPython`; không cần huấn luyện lại hay tải trọng số mô hình.

Để chạy phụ lục, mở notebook từ thư mục package hoặc `notebooks/`. Nếu dùng Colab, notebook tự lấy gói kết quả từ repo. Thông tin nguồn nằm trong gói dữ liệu.

## Cập nhật notebook từ nguồn Markdown

Các lệnh dưới đây dành cho người biên soạn tài liệu:

```bash
python scripts/build_notebooks.py --write
python scripts/build_notebooks.py --check
python -m compileall src/ailaai/
```

Lệnh `--write` tạo lại notebook từ nguồn Markdown. Lệnh `--check` báo nếu nội dung notebook khác với bản được tạo từ nguồn. Lệnh cuối kiểm tra cú pháp các tệp Python trong package.

Kết quả mỗi lượt chạy được lưu ở `artifacts/<run_id>/` và `outputs/<run_id>/`. Không commit các thư mục kết quả này vào repo.

Bài 1 và 2 mặc định huấn luyện đủ 15 epoch trên fold 0. Khi chạy lại cùng cấu hình, chương trình tiếp tục checkpoint còn dở hoặc nạp kết quả đã hoàn tất. Nếu đổi cấu hình, kết quả được lưu ở thư mục riêng. Private Test được chọn rõ ràng; bộ nạp không tự thay bằng Public Test.

Để học tiếp, checkpoint cần còn trong thư mục `artifacts/`. Khởi động lại kernel vẫn giữ các tệp này. Nếu xóa runtime hoặc Colab thu hồi phiên, bạn cần tải lại checkpoint đã lưu ở nơi khác hoặc huấn luyện từ đầu.

Cả năm notebook đã được [chạy kiểm tra trực tiếp trên Colab ngày 06/10/2026](evidence/colab_runall_20261006/README.md), gồm huấn luyện đủ 15 epoch, tạo ZIP nộp bài và học tiếp từ checkpoint sau khi khởi động lại kernel.
