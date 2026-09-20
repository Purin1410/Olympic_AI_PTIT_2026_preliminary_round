# Bài toán Kẻ mạo danh

Một mẫu là một cặp ảnh, có đúng một ảnh thật và một ảnh giả. Cần dự đoán vị trí của ảnh giả.
Đầu vào `pairs.csv` cung cấp ID và hai đường dẫn ảnh. Train thêm cột `fake_position` thuộc {0,1}.
Đầu ra là CSV với đúng hai cột `pair_id,fake_position`, một dòng cho mỗi ID đầu vào.

Code mô hình trả `p = P(ảnh phải giả)`. Quy tắc quyết định cố định là `p >= 0.5 → 1`, ngược lại `0`.
Accuracy và Macro-F1 được tính riêng, không mặc định hai chỉ số luôn bằng nhau.

