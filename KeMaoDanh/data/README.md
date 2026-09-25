# Chuẩn bị dữ liệu

Từ thư mục `KeMaoDanh`, giải nén gói `the_imposter.zip` của lớp:

```bash
uv run --locked --extra cu128 python scripts/prepare_official_dataset.py --zip-path the_imposter.zip --dest-dir data
```

Dùng `--extra cpu` thay `--extra cu128` nếu chỉ chạy baseline. Script giữ nguyên byte ảnh, kiểm tra đường dẫn trước khi giải nén và bỏ qua notebook baseline nằm trong ZIP. Sau lệnh trên, gói của lớp tạo cấu trúc:

```text
KeMaoDanh/data/data/
├── train/pairs.csv
├── train/images/...
├── public_test/pairs.csv
├── public_test/images/...
└── private_test/private_test/
    ├── pairs.csv
    └── images/...
```

Notebook tìm dữ liệu ở cấu trúc này, biến thể Windows có thêm một lớp `data/` (`KeMaoDanh/data/data/data/...`), hoặc `KeMaoDanh/data/train`. Mặc định tìm public test trước. Nếu data nằm cạnh repo hoặc ở Kaggle input, đặt đường dẫn tới thư mục chứa manifest:

```bash
export DATA_ROOT=/duong/dan/den/train
export TEST_ROOT=/duong/dan/den/public_test
```

`DATA_ROOT` phải có `pairs.csv` với `pair_id,image_0,image_1,fake_position`; các đường dẫn ảnh tính từ thư mục đó. `TEST_ROOT` dùng `pairs.csv` không cần nhãn. Pipeline không đọc `pairs_results.csv` để suy diễn.

Khi không có test, notebook vẫn huấn luyện và đánh giá OOF nhưng không xuất submission. Nếu đã đặt `TEST_ROOT` mà đường dẫn sai, code báo lỗi để bạn sửa.

800 ID trong `configs/development_split.csv` dùng cho phát triển, chia thành ba fold. 200 cặp giữ lại không được dùng để fit hay tính điểm trong các notebook này. Dữ liệu, ZIP, checkpoint và output đều nằm ngoài phần đưa lên Git.
