# Đặc tả Bài toán Kẻ mạo danh

## 1. Định nghĩa bài toán

Trong khuôn khổ cuộc thi Olympic AI PTIT 2026, bài toán Kẻ mạo danh (The Impostor) đặt ra yêu cầu xác định vị trí của khuôn mặt nhân tạo trong một cặp ảnh chân dung:
- Mỗi mẫu dữ liệu gồm một cặp hai ảnh: `image_0` (ảnh ở vị trí bên trái) và `image_1` (ảnh ở vị trí bên phải).
- Ràng buộc cấu trúc bất biến: Trong mỗi cặp luôn có đúng một ảnh thật và một ảnh giả mạo.
- Nhãn vị trí cần suy đoán: `fake_position` nhận giá trị `0` nếu ảnh bên trái là ảnh giả, hoặc nhận giá trị `1` nếu ảnh bên phải là ảnh giả.

Hệ thống không cần giải quyết bài toán phát hiện ảnh giả tuyệt đối với một ngưỡng tĩnh trong mọi điều kiện đời thực. Thay vào đó, bài toán là một cuộc đối đầu so sánh tương đối: giữa hai bức ảnh đang xét, bức ảnh nào chứa nhiều dấu vết kỹ thuật số bất thường hoặc dấu hiệu của thuật toán tạo sinh hơn.

## 2. Định dạng dữ liệu đầu vào

Tệp định danh `pairs.csv` trong tập huấn luyện bao gồm các cột bắt buộc:
- `pair_id`: Chuỗi ký tự định danh cố định gồm 5 chữ số có các số 0 ở đầu (ví dụ: `00578`, `00001`). Chuỗi này bắt buộc phải được đọc dưới dạng chuỗi ký tự (`str`) để tránh mất mát dữ liệu khi ép kiểu.
- `image_0`: Đường dẫn tương đối trỏ đến tệp ảnh thứ nhất trong cặp.
- `image_1`: Đường dẫn tương đối trỏ đến tệp ảnh thứ hai trong cặp.
- `fake_position`: Giá trị nhãn số nguyên thuộc {0, 1}.

Tệp `pairs.csv` trong tập kiểm tra (test) chỉ bao gồm ba cột đầu tiên (`pair_id`, `image_0`, `image_1`).

## 3. Quy định tệp kết quả nộp bài

Tệp nộp bài chính thức `submission.csv` phải có định dạng bảng chuẩn gồm đúng hai cột theo thứ tự:

```text
pair_id,fake_position
00001,0
00002,1
...
```

Yêu cầu kỹ thuật:
- Cột `pair_id` phải giữ nguyên vẹn định dạng chuỗi ký tự ban đầu và thứ tự các dòng của tập kiểm tra.
- Cột `fake_position` chỉ nhận giá trị số nguyên nhị phân là 0 hoặc 1 dựa trên ngưỡng xác suất phán đoán 0.5.

## 4. Thước đo đánh giá Macro-F1

Cuộc thi sử dụng chỉ số Macro-F1 làm thước đo xếp hạng chính thức:

$$\text{Macro-F1} = \frac{F1_0 + F1_1}{2}$$

Trong đó, $F1_0$ và $F1_1$ là chỉ số F1-Score tính độc lập cho từng vị trí lớp (lớp vị trí 0 và lớp vị trí 1):

$$F1_k = \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}$$

Thước đo Macro-F1 xử phạt rất nặng các mô hình chỉ thiên lệch dự đoán về một vị trí để đạt độ chính xác Accuracy cao. Chi tiết phân tích so sánh và công thức tính toán được minh họa tại [00_problem_and_data.ipynb](../notebooks/00_problem_and_data.ipynb).
