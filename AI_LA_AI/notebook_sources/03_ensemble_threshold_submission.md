<!-- ailaai-cell:00:markdown -->
# Bài 3: Kết hợp mô hình, khảo sát ngưỡng và kiểm tra tệp nộp bài

Sau khi có dự đoán của RGB và High-pass, ta sẽ so sánh các ảnh mà hai nhánh đoán đúng hoặc sai, rồi thử lấy trung bình xác suất theo tỷ lệ 50/50.

Bài này cũng khảo sát ảnh hưởng của ngưỡng quyết định và minh họa cách tạo, kiểm tra `submission.zip`.

<!-- ailaai-cell:01:code -->
from pathlib import Path
import importlib
import os
import subprocess
import sys

URL = "https://github.com/Purin1410/Olympic_AI_PTIT_2026_preliminary_round.git"
REF = os.environ.get("AILAAI_RELEASE_REF", "main")
# Reuse a local checkout when the notebook is opened from this repository.
TASK = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "src/ailaai").is_dir() and (p / "pyproject.toml").is_file()), None)
if TASK is None:
    REPO = Path("/content" if Path("/content").is_dir() else Path.cwd()) / "Olympic_AI_PTIT_2026_preliminary_round"
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REF, URL, str(REPO)], check=True)
    TASK = REPO / "AI_LA_AI"
else:
    REPO = TASK.parent
if not (TASK / "src/ailaai").is_dir():
    raise FileNotFoundError(f"Không tìm thấy package AI Là AI trong {TASK}.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(TASK / "requirements-colab.txt"), "-e", str(TASK)], check=True)
src_path = str((TASK / "src").resolve())
if src_path not in sys.path:
    sys.path.insert(0, src_path)
importlib.invalidate_caches()
import ailaai
if Path(ailaai.__file__).resolve().parent != TASK.resolve() / "src/ailaai":
    raise RuntimeError("Phiên đang dùng bản ailaai ở thư mục khác. Khởi động lại phiên rồi chạy từ đầu.")
print("Đã sẵn sàng:", TASK)

<!-- ailaai-cell:02:code -->
from ailaai.config import Workspace
from ailaai.resources import check_environment, verify_checkout
ws = Workspace.from_root(TASK, run_id="student_e2e_v1")
verify_checkout(REPO)
check_environment(profile="replay")
SOURCE = "reference"  # đổi thành "learner" để đọc run của bạn
print(ws.summary())

<!-- ailaai-cell:03:markdown -->
## 1. Ghép dự đoán của hai nhánh

Dự đoán ngoại mẫu theo fold (out-of-fold, OOF) được tạo cho ảnh thuộc fold không dùng để huấn luyện mô hình tương ứng. Trước khi so sánh, ghép dự đoán RGB và High-pass theo cùng tên ảnh, nhãn và fold.

Mỗi nhánh xuất xác suất cho lớp Fake, ký hiệu $P(\text{Fake})$. Đặt `SOURCE = "reference"` để đọc kết quả mẫu đã lưu hoặc `"learner"` để phân tích kết quả từ các lượt huấn luyện của bạn.

<!-- ailaai-cell:04:code -->
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from ailaai.resources import prepare_resources
from ailaai.predictions import load_validation_pair, align_predictions
from ailaai.visuals import show_prediction_examples
prepare_resources(ws, TASK / "configs/resources.json", profile="train")
rgb, hp, expected_val = load_validation_pair(ws, source=SOURCE)
m = align_predictions(rgb, hp, expected_rows=expected_val)
print(rgb.summary())
print(hp.summary())
print("Số ảnh đang phân tích:", len(m))

<!-- ailaai-cell:05:markdown -->
## 2. Các nhóm đúng và sai của hai mô hình

Ta chia các ảnh thành bốn nhóm:
1. Cả RGB và High-pass cùng đúng.
2. Chỉ RGB đúng.
3. Chỉ High-pass đúng.
4. Cả hai cùng sai.

Các nhóm này cho biết lỗi của hai nhánh trùng nhau đến đâu. Những ảnh chỉ một nhánh đúng là nơi có thể thử cải thiện bằng cách kết hợp, nhưng kết quả còn phụ thuộc vào xác suất của từng nhánh.

<!-- ailaai-cell:06:code -->
rgb_ok = (m.p_rgb >= 0.5) == m.label
hp_ok = (m.p_hp >= 0.5) == m.label
overlap = pd.Series({
    "Cả hai đúng": int((rgb_ok & hp_ok).sum()),
    "Chỉ RGB đúng": int((rgb_ok & ~hp_ok).sum()),
    "Chỉ High-pass đúng": int((~rgb_ok & hp_ok).sum()),
    "Cả hai sai": int((~rgb_ok & ~hp_ok).sum())})
assert overlap.sum() == len(m)
display(overlap.to_frame("Số ảnh"))

<!-- ailaai-cell:07:markdown -->
## 3. Lấy trung bình xác suất 50/50

Ta tính:

$$p_{\text{Mean}} = 0.5 \cdot p_{\text{RGB}} + 0.5 \cdot p_{\text{HP}}$$

Ví dụ, một ảnh Fake có $p_{\text{RGB}} = 0.48$ và $p_{\text{HP}} = 0.90$. Trung bình là $0.69$, vượt ngưỡng $0.50$ và sửa được quyết định sai của RGB. Ví dụ này minh họa cách kết hợp xác suất, chưa cho biết mô hình đã học dấu vết nào.

Trong kết quả OOF mẫu, cả RGB và High-pass đều có FP = 28. So với RGB, Mean sửa 26 ca (13 Real và 13 Fake), nhưng làm sai thêm 5 ca, đều là Fake. Tỷ lệ sửa/sai thêm là 5,2 : 1; tổng lỗi giảm từ 92 xuống 71.

Xem `fix_mask` và `break_mask` trong bảng dưới để đối chiếu những ca được sửa với những ca sai thêm.

<!-- ailaai-cell:08:code -->
m["p_mean"] = 0.5 * m.p_rgb + 0.5 * m.p_hp
mean_ok = (m.p_mean >= 0.5) == m.label
fix_mask = ~rgb_ok & mean_ok
break_mask = rgb_ok & ~mean_ok
display(pd.DataFrame([
    {"Mô hình": name, "Macro-F1": f1_score(m.label, m[col] >= 0.5,
        labels=[0, 1], average="macro", zero_division=0)}
    for name, col in [("RGB", "p_rgb"), ("High-pass", "p_hp"), ("Mean 50/50", "p_mean")]]))
print("RGB sai → mean đúng:", int(fix_mask.sum()))
print("RGB đúng → mean sai:", int(break_mask.sum()))
assert (~mean_ok).sum() == (~rgb_ok).sum() - fix_mask.sum() + break_mask.sum()
print(confusion_matrix(m.label, m.p_mean >= 0.5, labels=[0, 1]))

<!-- ailaai-cell:09:markdown -->
## 4. Xem ảnh được sửa và ảnh sai thêm

Ta xem hai nhóm ảnh:
- RGB sai nhưng Mean đúng (`fixes`).
- RGB đúng nhưng Mean sai (`breaks`).

Đối chiếu ảnh và xác suất của hai nhánh để xem quyết định thay đổi thế nào. Quan sát ảnh có thể gợi ý câu hỏi để thử tiếp, nhưng chưa đủ để xác định nguyên nhân mô hình dự đoán sai.

<!-- ailaai-cell:10:code -->
show_prediction_examples(m.loc[fix_mask].head(3), title="RGB sai → mean đúng")
show_prediction_examples(m.loc[break_mask].head(3), title="RGB đúng → mean sai")

<!-- ailaai-cell:11:markdown -->
## 5. Khảo sát ngưỡng quyết định

Khi quét ngưỡng từ 0.1 đến 0.9 trên validation, bạn có thể thấy một ngưỡng như 0.52 hoặc 0.48 cho F1 cao hơn mốc 0.50.

Nếu dùng cùng dữ liệu để chọn ngưỡng và báo điểm, kết quả có thể lạc quan hơn khi gặp dữ liệu mới (optimism bias). Muốn quyết định đổi ngưỡng, cần đánh giá lựa chọn đó trên dữ liệu không dùng để chọn ngưỡng.

Bài này giữ ngưỡng mặc định $t = 0.50$. Đây là lựa chọn cho quy trình đang dùng, không phải ngưỡng tốt nhất cho mọi bài toán.

<!-- ailaai-cell:12:code -->
import matplotlib.pyplot as plt
thresholds = np.linspace(0.1, 0.9, 161)
scores = np.array([f1_score(m.label, m.p_mean >= t, labels=[0, 1],
                          average="macro", zero_division=0) for t in thresholds])
i = int(scores.argmax())
print("Ngưỡng tốt nhất trên tập đang khảo sát:", thresholds[i])
print("Điểm dùng cả để chọn và đo ngưỡng:", scores[i])
plt.plot(thresholds, scores)
plt.axvline(0.5, color="gray", linestyle="--", label="Mặc định 0.5")
plt.xlabel("Ngưỡng"); plt.ylabel("Macro-F1 trên tập đang khảo sát"); plt.legend(); plt.show()

<!-- ailaai-cell:13:markdown -->
## 6. Chọn phương án để chạy tiếp

Mức tăng nhỏ, chẳng hạn dưới 0.1%, trên một fold chưa đủ để quyết định dùng phương án phức tạp hơn. Cần xem kết quả có lặp lại và có đáng với chi phí bổ sung không.

Trong kết quả OOF mẫu, Simple Mean 50/50 với RGB và High-pass tại $t = 0.50$ đạt Macro-F1 96.45%. Cách này không cần học thêm tham số kết hợp.

Ta ghi lại phương án đã chọn trước khi dự đoán tập test.

<!-- ailaai-cell:14:code -->
from ailaai.pipeline import save_analysis_note
FINAL_THRESHOLD = 0.5
save_analysis_note(ws, rgb=rgb, highpass=hp, threshold=FINAL_THRESHOLD,
                   method="probability_mean_50_50",
                   note="Giữ phương án này; threshold sweep chỉ để quan sát.")

<!-- ailaai-cell:15:markdown -->
## 7. Tạo và kiểm tra tệp nộp bài

`export_submission` và `validate_submission` kiểm tra các điều kiện sau:
- CSV nằm ở gốc tệp ZIP.
- Có hai cột `file_name,category_id`.
- Đủ số dòng theo danh sách test và không thiếu giá trị.
- Nhãn là số nguyên 0 hoặc 1.
- Tên ảnh khớp danh sách test, không trùng và không thiếu ảnh.

<!-- ailaai-cell:16:code -->
from ailaai.submission import export_submission, validate_submission
DEMO_NAMES = [f"demo_{i:03d}.jpg" for i in range(200)]
demo = pd.DataFrame({"file_name": DEMO_NAMES,
                     "category_id": np.arange(200, dtype="int64") % 2})
display(demo.head())

<!-- ailaai-cell:17:code -->
demo_path = export_submission(demo, ws.output_root / "demo_submission.zip",
                              expected_names=DEMO_NAMES, expected_count=200)
demo_receipt = validate_submission(demo_path, expected_names=DEMO_NAMES,
                                  expected_count=200, expected_frame=demo)
print(demo_receipt.summary())
print("File minh họa; không phải dự đoán test.")

<!-- ailaai-cell:18:markdown -->
## 8. Thử một trường hợp dữ liệu lỗi

Cell dưới tạo hai dòng trùng tên ảnh `file_name` để xem bộ kiểm tra có phát hiện lỗi không.

<!-- ailaai-cell:19:code -->
bad = demo.copy()
bad.loc[1, "file_name"] = bad.loc[0, "file_name"]
try:
    export_submission(bad, ws.output_root / "demo_duplicate.zip",
                      expected_names=DEMO_NAMES, expected_count=200)
except ValueError as exc:
    print("Đã phát hiện lỗi:", exc)
else:
    raise AssertionError("Bộ kiểm tra chưa phát hiện tên ảnh trùng.")

<!-- ailaai-cell:20:markdown -->
## 9. Chuyển sang pipeline đầy đủ

Sau khi xem kết quả kết hợp, cách chọn ngưỡng và cách kiểm tra ZIP, bạn có thể mở `00_pipeline_end_to_end.ipynb` để chạy quy trình từ dữ liệu đến tệp nộp bài.

<!-- ailaai-cell:21:code -->
print("Nguồn:", SOURCE, "| Số ảnh validation:", len(m))
print("Threshold đã ghi:", FINAL_THRESHOLD)
print("Notebook dự đoán test:", TASK / "notebooks/00_pipeline_end_to_end.ipynb")

<!-- ailaai-cell:22:markdown -->
Kết quả phân tích trong bài này dùng dự đoán OOF. `demo_submission.zip` là tệp minh họa cách đóng gói, không phải bài nộp cho Private Test. Để dự đoán trên ảnh test thực tế, dùng `00_pipeline_end_to_end.ipynb`.
