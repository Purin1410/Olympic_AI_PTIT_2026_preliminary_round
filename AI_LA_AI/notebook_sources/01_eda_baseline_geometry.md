<!-- ailaai-cell:00:markdown -->
# Bài 1: Khám phá dữ liệu, Macro-F1 và hình học ảnh

Bài này bắt đầu từ dữ liệu: các lớp có cân bằng không, mô hình được chấm điểm thế nào và cách xử lý kích thước ảnh ảnh hưởng ra sao?

Ta sẽ kiểm tra dữ liệu, tính Macro-F1 từ ma trận nhầm lẫn và theo dõi một bước học của CNN2. Sau đó, ta so sánh phổ FFT của vùng cắt gốc với vùng được tái lấy mẫu theo chuỗi $358 \to 224 \to 358$.

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
RUN_ID = "lesson_nb1_v1"
ws = Workspace.from_root(TASK, run_id=RUN_ID)
check_environment(profile="train")
print(ws.summary())

<!-- ailaai-cell:03:markdown -->
## 1. Nạp và kiểm tra dữ liệu

Bộ dữ liệu có tại [who_is_AI (Google Drive)](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view?usp=drive_link). Phần nạp dữ liệu sử dụng thư mục `data/` và cấu hình nguồn tải của bài.

Ta dùng bảng chia 5 fold cố định ở `assets/splits/train_folds.csv` để các thử nghiệm có cùng cách chia dữ liệu. Trước khi chạy mô hình, kiểm tra số ảnh và tỷ lệ hai lớp Real (0), Fake (1).

<!-- ailaai-cell:04:code -->
from ailaai.data import load_train_manifest, load_fold_split
from ailaai.resources import prepare_resources

prepare_resources(ws, TASK / "configs/resources.json", profile="train")
train = load_train_manifest(ws)
fit_rows, val_rows = load_fold_split(train, TASK / "assets/splits/train_folds.csv", fold=0)
print("Train:", len(train), "Fit:", len(fit_rows), "Validation:", len(val_rows))
display(train.label.value_counts().sort_index().rename(index={0: "Real", 1: "Fake"}))

<!-- ailaai-cell:05:markdown -->
## 2. Cách tính Macro-F1

Bài thi dùng Macro-F1 để đánh giá. Chỉ số này tính $F_1$ cho từng lớp rồi lấy trung bình:

$$\text{Macro-F1} = \frac{F_{1, \text{Real}} + F_{1, \text{Fake}}}{2}$$

Cell dưới tính điểm từ ma trận nhầm lẫn (TP, FP, FN, TN) và đối chiếu với `scikit-learn`.

<!-- ailaai-cell:06:code -->
import numpy as np
from sklearn.metrics import f1_score
from ailaai.metrics import macro_f1

def macro_f1_from_counts(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        cm[int(truth), int(prediction)] += 1
    per_class = []
    for label in (0, 1):
        tp = cm[label, label]
        denominator = 2 * tp + cm[:, label].sum() - tp + cm[label, :].sum() - tp
        per_class.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(per_class))

y = np.array([0, 0, 0, 1])
pred = np.array([0, 0, 0, 0])
print("Macro-F1:", macro_f1_from_counts(y, pred))
assert np.isclose(macro_f1_from_counts(y, pred), macro_f1(y, pred))
assert np.isclose(macro_f1(y, pred), f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))

<!-- ailaai-cell:07:markdown -->
## 3. Theo dõi một bước học của CNN2

Ta dùng CNN2 với hai tầng tích chập để xem dữ liệu đi qua mạng như thế nào: tensor ảnh, các tầng tích chập, gộp thích ứng (adaptive pooling), rồi tầng phân loại.

Sau khi có đầu ra, ta tính loss Cross-Entropy, lan truyền ngược và cập nhật trọng số trong một bước tối ưu.

<!-- ailaai-cell:08:code -->
import torch
from torch import nn

class CNN2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(32, num_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

demo_model = CNN2()
demo_optimizer = torch.optim.AdamW(demo_model.parameters(), lr=1e-3)
demo_logits = demo_model(torch.rand(2, 3, 358, 358))
demo_loss = nn.functional.cross_entropy(demo_logits, torch.tensor([0, 1]))
demo_loss.backward()
demo_optimizer.step()
print("Logits:", tuple(demo_logits.shape), "Loss:", float(demo_loss.detach()))

<!-- ailaai-cell:09:markdown -->
## 4. So sánh cùng vùng nhìn: Same-FOV

Co giãn ảnh (resize) làm thay đổi các chi tiết tần số cao như thế nào?

Ta cắt vùng giữa ảnh (Center Crop), giữ lại khoảng 70% chiều dài mỗi cạnh. Với ảnh gốc $512 \times 512$, vùng cắt là $358 \times 358$ vì $0{,}7 \times 512 = 358{,}4 \approx 358$. Trên các ảnh đã xem, vùng này bao quanh mắt, mũi, miệng và hai má. Đây là lựa chọn từ quan sát, chưa có thí nghiệm xác định kích thước tối ưu.

Hai phiên bản dùng cùng vùng cắt:
- **Native 358:** Giữ nguyên pixel gốc.
- **Resampled 358:** Tái lấy mẫu (resampling) bằng cách thu nhỏ về $224 \times 224$, rồi phóng lại $358 \times 358$.

Cùng vùng nhìn và cùng kích thước đầu ra giúp ta tập trung so sánh ảnh hưởng của phép tái lấy mẫu.

<!-- ailaai-cell:10:code -->
from ailaai.transforms import native_view, resampled_view
from ailaai.data import decode_rgb
sample = train.iloc[0]
image = decode_rgb(sample.path)
native = native_view(image)
resampled = resampled_view(image, crop=358, size=224)
print("Native:", tuple(native.shape), "Resampled:", tuple(resampled.shape))
assert native.shape == resampled.shape == (3, 358, 358)

<!-- ailaai-cell:11:markdown -->
## 5. Quan sát phổ tần số 2D-FFT

Biến đổi Fourier 2D (FFT) biểu diễn các thành phần tần số của ảnh. Biểu đồ chênh lệch phổ (difference spectrum) giúp xem thành phần nào thay đổi sau khi tái lấy mẫu.

Giữ vùng cắt Native bảo toàn pixel gốc. Quan sát FFT cho thấy thay đổi về tín hiệu; để biết ảnh hưởng đến phân loại, ta còn cần so sánh kết quả mô hình.

<!-- ailaai-cell:12:code -->
from ailaai.visuals import show_views, plot_spectra
show_views(image, {"Native 358": native, "Resampled 358": resampled})
plot_spectra(native, resampled, names=("Native 358", "Resampled 358"))

<!-- ailaai-cell:13:markdown -->
## 6. Huấn luyện thử mô hình

Bạn viết hàm tạo mô hình (`student_model_factory`) và hàm xử lý ảnh (`student_view`) trong notebook, rồi truyền chúng vào `fit_fold` để huấn luyện.

Mặc định `RUN_BASELINE = True`: mô hình học đủ 15 epoch trên fold 0. Chọn T4 GPU trong Colab trước khi chạy. Đặt cờ này thành `False` nếu chỉ muốn xem các phần minh họa.

<!-- ailaai-cell:14:code -->
from dataclasses import replace
from ailaai.config import load_config
from ailaai.engine import fit_fold
from ailaai.models import model_factory as build_model

cfg = load_config(TASK / "configs/rgb_native358.json")
MODEL_SPEC = dict(cfg.model)
def student_model_factory(*, initialize):
    return build_model(MODEL_SPEC["backbone"], MODEL_SPEC["weights"], initialize=initialize)
def student_view(batch):
    return native_view(batch)
RUN_BASELINE = True
if RUN_BASELINE:
    run = fit_fold(ws, cfg, branch="rgb", train_rows=fit_rows, val_rows=val_rows,
                   model_factory=student_model_factory, view_fn=student_view,
                   view_spec=dict(cfg.view), model_spec=MODEL_SPEC)
    print(run.summary())

<!-- ailaai-cell:15:markdown -->
## 7. Nhìn lại bài 1

Quy ước nhãn trong các bài là $0 = \text{Real}$ và $1 = \text{Fake}$. Khi đọc kết quả, xem $F_1$ của từng lớp cùng với Macro-F1 để biết mô hình đang gặp khó ở lớp nào.

So sánh FFT cho thấy resize làm thay đổi tín hiệu tần số cao. Điều này chưa đủ để kết luận cách xử lý nào phân loại tốt hơn. Các bài tiếp theo dùng vùng cắt $358 \times 358$ để xây dựng và so sánh các nhánh mô hình.

<!-- ailaai-cell:16:code -->
from ailaai.metrics import classification_report
print(classification_report(y, np.array([0.1, 0.2, 0.4, 0.6])))
