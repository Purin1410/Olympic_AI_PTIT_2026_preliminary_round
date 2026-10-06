# Kiểm tra năm notebook trên Colab

Ngày 06/10/2026, bản code `ba003c7` đã được chạy trực tiếp trên Google Colab bằng **Run all**. Cả năm notebook hoàn tất, tổng cộng 50 cell code, không có lỗi Python trong các lượt đã kiểm tra.

| Notebook | Thiết bị | Cell code | Kết quả |
| --- | --- | ---: | --- |
| `00_pipeline_end_to_end.ipynb` | T4 GPU | 16 | RGB và High-pass đủ 15 epoch; dự đoán 200 ảnh Private Test; ZIP hợp lệ. |
| `01_eda_baseline_geometry.ipynb` | T4 GPU | 9 | Đủ 15 epoch sau khi khởi động lại kernel và học tiếp từ checkpoint; chạy tới cell cuối. |
| `02_forensic_specialist.ipynb` | T4 GPU | 5 | Chạy từ đầu trong phiên mới, huấn luyện High-pass đủ 15 epoch. |
| `03_ensemble_threshold_submission.ipynb` | CPU | 12 | Đọc 2.000 dự đoán mẫu, tạo ZIP minh họa 200 dòng và phát hiện trường hợp trùng tên ảnh. |
| `04_negative_results_and_ablation.ipynb` | CPU | 8 | Tải tài nguyên, tính lại các bảng và hiển thị hình minh họa. |

Các lượt huấn luyện dùng toàn bộ 2.000 ảnh train, chia fold 0 thành 1.600 ảnh học và 400 ảnh validation. Cấu hình mặc định được giữ nguyên. Bài 3 và 4 phân tích kết quả mẫu có sẵn.

Bài 00 được Run all lần thứ hai trong cùng phiên. Cả hai nhánh nạp lượt đã hoàn tất và tạo lại ZIP hợp lệ, không huấn luyện lại. Với bài 01, kernel được khởi động lại khi đang huấn luyện; lượt sau học tiếp từ epoch 4 và hoàn tất epoch 15. Bài 01 cũng chạy lại hết sau khi tải lại trang.

Trong lúc kiểm tra bài 01, Colab báo không tải được JavaScript để hiển thị output. Training vẫn tiếp tục. Sau khi đóng thông báo, hoàn tất training và tải lại trang, lượt Run all tiếp theo hiển thị được ảnh, vùng cắt và biểu đồ FFT; thông báo không lặp lại.

[verification.json](verification.json) ghi số cell, bộ đếm thực thi và các tình huống đã kiểm tra. [epoch_logs.csv](epoch_logs.csv) lưu số liệu được in trên Colab; log bài 01 trong tệp này bắt đầu từ epoch 4, sau khi khởi động lại kernel. Bằng chứng được đọc từ giao diện Colab; checkpoint và notebook có output chưa được tải về để kiểm tra độc lập.

Để học tiếp sau khi khởi động lại kernel, checkpoint cần còn trong thư mục `artifacts/`. Xóa runtime sẽ xóa các tệp trong phiên đó.
