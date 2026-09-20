# Lượt chạy được lưu trong notebook

Các output hiện có được tạo ngày 20/09/2026 trên dữ liệu thật, với `KMD_SMOKE=0`, `KMD_PROFILE=full` và `KMD_INCLUDE_EXTENSIONS=1`. Cả 13 notebook đã được thực thi bằng kernel mới. Các fold trùng cấu hình dùng lại checkpoint của chính lượt chạy này sau khi kiểm tra mã nguồn, dữ liệu và cấu hình; không dùng checkpoint smoke.

Môi trường chạy: RTX 5060 Ti 16 GB, Python 3.11.16, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, NumPy 2.2.6, pandas 2.3.3 và scikit-learn 1.7.2. Run ID là `lesson_full_v5_20260920`. Cài đặt cho lượt chạy của bạn theo [README](../README.md); đường dẫn máy thực thi xuất hiện trong output chỉ ghi lại nơi kết quả được tạo.

Trên 800 cặp development, EfficientNet-B2 đạt Macro-F1 OOF trung bình ba seed là 97,5833%. Blend 50/50 với native tăng trung bình 0,4583 điểm phần trăm so với B2, dưới ngưỡng 0,5 đã đặt trước. Vì vậy lựa chọn nộp vẫn là B2, seed 20260917, ensemble ba fold. Nhánh tùy chọn `refit_all` chưa được thực thi trong lượt này.

Notebook 07 đối chiếu 35 phương pháp đã khóa trên 100 cặp private. Nhãn tham chiếu lấy từ `pairs_results.csv` trong gói dữ liệu của lớp. Đây là điểm tính lại từ tệp nhãn đó, không phải điểm nộp mới trên leaderboard. Bảng private không được dùng để đổi lựa chọn mô hình. OOF development và dự đoán private bằng ensemble ba fold là hai giao thức khác nhau, như notebook đã giải thích.

Notebook giữ 36 hình nhúng cùng bảng kết quả và log. Bạn có thể đọc trực tiếp mà không cần chạy lại. Kiểm tra cuối xác nhận tất cả code cell đã chạy theo thứ tự, không có output lỗi, ảnh trích xuất khớp ảnh nhúng, và submission public có 100 dòng hợp lệ. Log của 102 fold CNN xác nhận đủ ngân sách hoặc dừng đúng điều kiện early stopping; ba fold LR trên embedding được kiểm tra bằng checksum và dự đoán sau khi nạp lại.
