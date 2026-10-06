# Hồ sơ kiểm tra bài 04 tại phiên bản bac26fc

Ngày: 2026-10-06. Các số cell và kết quả dưới đây thuộc phiên bản này.

## Phạm vi đã chạy

- 11 kiểm tra cục bộ đạt: phép Haar với tín hiệu có hệ số biết trước, kích thước/miền giá trị/gradient, cân bằng tổng trọng số Fake, xử lý trường hợp đặc trưng bằng nhau, chia calibration/evaluation, chọn ngưỡng, lưu trọng số trong cấu hình và nạp checkpoint có trọng số.
- Notebook sinh từ Markdown khớp cả năm bản; kiểm tra cú pháp và khoảng trắng đạt.
- Chế độ `reference`, đủ năm fold: chạy hết 11 cell code trong kernel CPU mới, gồm cell cài đặt chung; 2.000 dự đoán, ba hình PNG, 1.008 ảnh evaluation sau khi tách calibration theo nhãn và xám/màu. Báo cáo: `nb4_e2e_reference_verification.json`.
- Tính đặc trưng từ bộ ảnh thật đã tải: 2.000 ảnh, 1.000 ảnh xám; fold 0 có 1.600 train và 400 validation. Ngưỡng Sobel phân vị 25% của Fake train là 0,0141556910; tổng trọng số 800 ảnh Fake giữ ở 800. Phần validation tách thành 198 calibration và 202 evaluation. Lượt tính đặc trưng CPU mất khoảng 15 giây với hai luồng PyTorch.

## Phần chưa xác minh

Hồ sơ này chưa có kết quả huấn luyện đủ 15 epoch cho ba nhánh bài 04 trên Colab T4.

## Công thức và kết quả

Bài 4 mặc định dùng ResNet18 cho cả ba nhánh, biểu diễn Haar xếp bốn dải trong từng kênh RGB, đặc trưng Sobel và trọng số chọn từ Fake train. Đây là công thức thực hành mới; không gán điểm lịch sử cho nó. Chế độ `reference` giữ dự đoán lịch sử, nhưng phần thử ngưỡng dùng cách tách calibration/evaluation mới được mô tả trong notebook.

Bài này kết thúc ở báo cáo ablation trên dữ liệu có nhãn. Tạo tệp nộp Private Test thuộc bài 0; báo cáo bài 4 không công bố điểm Private Test.
