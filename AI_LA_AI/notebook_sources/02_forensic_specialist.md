<!-- ailaai-cell:00:markdown -->
# Bài 2: Xử lý ảnh bằng High-pass và huấn luyện ResNet34

Ở bài 1, ta đã xem tín hiệu tần số thay đổi khi resize ảnh. Bài này thử một cách biểu diễn khác: lấy ảnh gốc trừ đi ảnh làm mờ Gaussian.

Ta sẽ viết `highpass_view`, xem ảnh sau khi lọc và dùng biểu diễn này làm đầu vào cho ResNet34. Kết quả sẽ được so sánh với nhánh RGB ở bài 3.

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
from ailaai.resources import check_environment
RUN_ID = "lesson_nb2_v1"
ws = Workspace.from_root(TASK, run_id=RUN_ID)
check_environment(profile="train")
print(ws.summary())

<!-- ailaai-cell:03:markdown -->
## 1. Cấu hình nhánh High-pass

Để so sánh trên cùng dữ liệu, High-pass dùng vùng cắt $358 \times 358$ và bảng chia fold giống nhánh RGB. Bộ dữ liệu có tại [who_is_AI (Google Drive)](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view?usp=drive_link).

Xem các tham số trong `configs/highpass358.json`: kích thước nhân Gaussian $k = 5$, độ lệch chuẩn $\sigma = 1.0$, hệ số khuếch đại (gain) và độ dịch (offset).

<!-- ailaai-cell:04:code -->
from dataclasses import replace
from ailaai.config import load_config
from ailaai.resources import prepare_resources
from ailaai.data import load_train_manifest, load_fold_split, decode_rgb
from ailaai.visuals import show_views

prepare_resources(ws, TASK / "configs/resources.json", profile="train")
train = load_train_manifest(ws)
fit_rows, val_rows = load_fold_split(train, TASK / "assets/splits/train_folds.csv", fold=0)
cfg = load_config(TASK / "configs/highpass358.json")
print("Fit:", len(fit_rows), "Validation:", len(val_rows))

<!-- ailaai-cell:05:markdown -->
## 2. Lấy ảnh gốc trừ đi ảnh làm mờ

Với ảnh gốc $I$, ta làm mờ bằng Gaussian rồi tính phần dư:

$$R = I - (G_\sigma * I)$$

Ảnh làm mờ giảm các thay đổi nhanh theo không gian. Phần dư làm nổi các thay đổi cục bộ, chẳng hạn chi tiết bề mặt và nhiễu. Các chi tiết này không tự xác định ảnh là thật hay giả.

Ta nhân phần dư với gain, cộng offset 0.5 và đưa giá trị về dải $[0, 1]$. Hàm `highpass_view` xử lý được cả một ảnh dạng CHW và một batch dạng BCHW.

<!-- ailaai-cell:06:code -->
import torch
from ailaai.forensics import gaussian_blur_rgb
from ailaai.transforms import native_view

VIEW_SPEC = dict(cfg.view)
def highpass_view(images):
    crop = native_view(images, crop=VIEW_SPEC["crop"])
    low = gaussian_blur_rgb(crop, kernel_size=VIEW_SPEC["kernel"], sigma=VIEW_SPEC["sigma"])
    return (VIEW_SPEC["gain"] * (crop - low) + VIEW_SPEC["offset"]).clamp(0, 1)

example = decode_rgb(train.iloc[0].path)
rgb = native_view(example, crop=VIEW_SPEC["crop"])
filtered = highpass_view(example)
assert filtered.shape == rgb.shape and torch.isfinite(filtered).all()
show_views(example, {"RGB Native 358": rgb, "High-pass": filtered})

<!-- ailaai-cell:07:markdown -->
## 3. Huấn luyện hoặc nạp mô hình

Hàm `highpass_view` vừa xem được truyền vào `fit_fold`. ResNet34 nhận ảnh High-pass làm đầu vào.

Chọn cách chạy:
- Mặc định `RUN_TRAIN = True` và `MODE = "train"`: huấn luyện đủ 15 epoch trên fold 0 với GPU T4.
- `RUN_TRAIN = False` và `MODE = "train"`: bỏ qua huấn luyện.
- `MODE = "load"`: nạp lượt chạy đã lưu nếu có checkpoint tương ứng.

Khi có lượt chạy, cell hiển thị phần tổng hợp kết quả.

<!-- ailaai-cell:08:code -->
from ailaai.engine import fit_fold, load_run
from ailaai.models import model_factory as build_model
MODEL_SPEC = dict(cfg.model)
def student_model_factory(*, initialize):
    return build_model(MODEL_SPEC["backbone"], MODEL_SPEC["weights"], initialize=initialize)
MODE = "train"
RUN_TRAIN = True
if MODE == "train" and RUN_TRAIN:
    run = fit_fold(ws, cfg, branch="highpass", train_rows=fit_rows, val_rows=val_rows,
                   model_factory=student_model_factory, view_fn=highpass_view,
                   view_spec=VIEW_SPEC, model_spec=MODEL_SPEC)
elif MODE == "load":
    run = load_run(ws, cfg, branch="highpass", train_rows=fit_rows, val_rows=val_rows,
                   model_factory=student_model_factory, view_fn=highpass_view,
                   view_spec=VIEW_SPEC, model_spec=MODEL_SPEC)
else:
    run = None
if run is not None:
    print(run.summary())

<!-- ailaai-cell:09:markdown -->
## 4. Chuẩn bị so sánh với RGB

RGB và High-pass nhận hai cách biểu diễn của cùng vùng ảnh. Điểm riêng của mỗi nhánh chưa cho biết lấy trung bình xác suất có tốt hơn hay không.

Hai nhánh có sai trên cùng những ảnh không? Bài 3 sẽ so sánh các nhóm lỗi và kiểm tra kết quả kết hợp.
