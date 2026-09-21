# Olympic AI PTIT 2026 - Vòng sơ loại

Kho lưu trữ tổng hợp lời giải và notebook thực hành cho các bài toán trong vòng sơ loại Olympic AI PTIT 2026. Mỗi bài được đặt trong một thư mục riêng, có README, dữ liệu đầu vào, mã nguồn và quy trình thực nghiệm độc lập.

## Các bài toán

### Kẻ mạo danh

Bài toán phân loại vị trí ảnh giả trong cặp ảnh chân dung. Chuỗi bài học hướng dẫn bạn qua từng bước: khám phá dữ liệu, baseline Logistic Regression, fine-tuning CNN, so sánh vùng cắt ảnh, chọn backbone, phân tích lỗi, thử nghiệm blend mô hình và xuất bài nộp.

Nếu mới học, bạn hãy bắt đầu với 8 notebook chính từ 00 đến 07. Khi đã hiểu cách giải và muốn chạy lại, dùng notebook pipeline tổng hợp. Bốn phụ lục A-D dành cho những thử nghiệm bạn muốn tìm hiểu thêm.

Xem [cách chọn notebook và hướng dẫn chạy bài Kẻ mạo danh](KeMaoDanh/README.md).

### Bài toán thứ hai

Nội dung sẽ được bổ sung trong một thư mục ngang hàng khi hoàn thiện.

## Cấu trúc repository

```text
.
├── README.md
├── assets/       # Tài liệu và sơ đồ dùng chung
└── KeMaoDanh/    # Lời giải và notebook của bài Kẻ mạo danh
```

Mỗi bài toán mới sẽ được thêm thành một thư mục độc lập ở cấp root theo cùng cách tổ chức.
