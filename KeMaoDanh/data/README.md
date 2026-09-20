# Dữ liệu người học tự cung cấp

Đặt dataset gốc trong `train/` và bộ cần dự đoán trong `test/`, hoặc đặt `DATA_ROOT`/`TEST_ROOT`.
Mỗi thư mục chứa `pairs.csv` và các ảnh được CSV tham chiếu bằng đường dẫn tương đối.
Train bắt buộc có `pair_id,image_0,image_1,fake_position`. Test chỉ cần ba cột đầu.
`fake_position=0` là trái giả, `1` là phải giả; không có lớp "cả hai thật".

Code chỉ chọn 800 cặp theo `../configs/development_split.csv`; file này không chứa ảnh hay nhãn.
Dùng đúng bản dataset tương ứng, không tự đổi ID hoặc thay ảnh dưới cùng tên.
Các nhãn/ảnh của 200 cặp ngoài development không được đưa vào fit hay đánh giá.
Ở bước test, nếu CSV có thêm cột nhãn thì cột đó cũng bị bỏ qua.

Không đặt dataset/zip/model trong Git. `.gitignore` bỏ qua mọi nội dung data trừ file hướng dẫn này.
