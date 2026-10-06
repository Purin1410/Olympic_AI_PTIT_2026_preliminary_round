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
RUN_ID = "lesson_nb1_reading_v2"
ws = Workspace.from_root(TASK, run_id=RUN_ID)
check_environment(profile="replay")
print(ws.summary())

<!-- ailaai-cell:03:code -->
from pathlib import Path
from typing import Any, Callable, Mapping, Iterable
from dataclasses import dataclass, replace
import hashlib, json, time, os, tempfile, zipfile
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=["#2563A6", "#C17817", "#7A5BA7"])})
from IPython.display import display
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import functional as TF, InterpolationMode
from ailaai.config import TrainConfig, Workspace, write_json
from ailaai.data import decode_rgb
from ailaai.models import attach_optimizer_groups
from ailaai.metrics import classification_report
from ailaai.predictions import PredictionTable
from ailaai.resources import sha256_file
# Các hàm dưới quản lý checkpoint/config; phép học được viết ở các cell tiếp theo.
from ailaai.engine import (RunInfo, _resolved_config, _run_directory, _load_checkpoint,
                          _seed, _save_curves, _checkpoint_state, _callable_digest)
ModelFactory = Callable[..., nn.Module]
ViewFunction = Callable[[torch.Tensor], torch.Tensor]

<!-- ailaai-cell:04:markdown -->
## 1. Nạp và kiểm tra dữ liệu

Bộ dữ liệu có tại [who_is_AI (Google Drive)](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view?usp=drive_link). Phần nạp dữ liệu sử dụng thư mục `data/` và cấu hình nguồn tải của bài.

Ta dùng bảng chia 5 fold cố định ở `assets/splits/train_folds.csv` để các thử nghiệm có cùng cách chia dữ liệu. Trước khi chạy mô hình, kiểm tra số ảnh và tỷ lệ hai lớp Real (0), Fake (1).

<!-- ailaai-cell:05:code -->
from ailaai.data import load_train_manifest, load_fold_split
from ailaai.resources import prepare_resources

prepare_resources(ws, TASK / "configs/resources.json", profile="train")
train = load_train_manifest(ws)
fit_rows, val_rows = load_fold_split(train, TASK / "assets/splits/train_folds.csv", fold=0)
print("Train:", len(train), "Fit:", len(fit_rows), "Validation:", len(val_rows))
display(train.label.value_counts().sort_index().rename(index={0: "Real", 1: "Fake"}))

<!-- ailaai-cell:06:markdown -->
## Khám phá dữ liệu trước khi huấn luyện

Trước hết, ta xem toàn bộ tập train: mỗi lớp có bao nhiêu ảnh, ảnh có cùng kích thước không, dung lượng JPEG phân bố ra sao và có ảnh nào trùng nội dung. Hai phỏng đoán trong Reading II.2 cũng được kiểm tra ở đây: ảnh nhỏ hơn có dễ là Fake hơn không, và ảnh xám có phải đều là Real?

Cột `is_gray` yêu cầu R=G=B trên mọi pixel. Cột `near_gray` cho phép chênh lệch nhỏ giữa các kênh, để ta thấy kết quả có phụ thuộc cách định nghĩa ảnh xám hay không.

<!-- ailaai-cell:07:code -->
# ailaai-source: src/ailaai/teaching.py::image_census
def image_census(rows):
    """Read every image; record size, exact gray flag and duplicate content."""
    records = []
    for row in rows.itertuples():
        path = Path(row.path)
        with Image.open(path) as image:
            original_mode = image.mode
            rgb = np.asarray(image.convert("RGB"))
        delta = rgb.max(axis=2).astype(float) - rgb.min(axis=2).astype(float)
        records.append({"file_name": row.file_name, "label": int(row.label),
                        "path": str(path), "width": rgb.shape[1], "height": rgb.shape[0],
                        "mode": original_mode, "size_kib": path.stat().st_size / 1024,
                        "is_gray": bool((delta == 0).all()),
                        "near_gray": bool(delta.mean() <= 1),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    census = pd.DataFrame(records)
    if census.empty or not census.file_name.is_unique:
        raise ValueError("Image census requires nonempty rows with unique file names.")
    return census

<!-- ailaai-cell:08:code -->
census = image_census(train)
display(census.groupby("label").agg(n=("file_name", "size"), gray=("is_gray", "sum"), near_gray=("near_gray", "sum")))
display(census.groupby(["width", "height", "mode"]).size().rename("Số ảnh").reset_index())
duplicates = census[census.duplicated("sha256", keep=False)]
print("Ảnh thuộc nhóm trùng byte:", len(duplicates))
if not duplicates.empty:
    display(duplicates[["file_name", "label", "sha256"]].head(20))
census.to_csv(ws.output_root / "eda_image_census.csv", index=False)

<!-- ailaai-cell:09:markdown -->
### Xem ảnh của từng nhóm

Mỗi hàng dưới đây ứng với một tổ hợp nhãn và cờ xám/màu. Seed cố định giúp các lần mở notebook chọn lại cùng ảnh. Hãy nhìn vùng mặt, nền và mức độ chi tiết; tên tệp dưới mỗi ảnh giúp ta quay lại kiểm tra khi cần. Nhận xét về vị trí khuôn mặt mới chỉ áp dụng cho những mẫu đang xem.

<!-- ailaai-cell:10:code -->
fig, axes = plt.subplots(4, 3, figsize=(10, 11))
for row_index, (label, gray) in enumerate([(0, False), (0, True), (1, False), (1, True)]):
    group = census[(census.label == label) & (census.is_gray == gray)]
    selected = group.sample(min(3, len(group)), random_state=2026)
    for ax in axes[row_index]: ax.axis("off")
    for ax, row in zip(axes[row_index], selected.itertuples()):
        with Image.open(row.path) as im: ax.imshow(im.convert("RGB"))
        ax.set_title(f'{row.file_name} | {"Fake" if label else "Real"} | {"xám" if gray else "màu"}', fontsize=9)
fig.tight_layout(); plt.show()

<!-- ailaai-cell:11:markdown -->
### Dung lượng JPEG có tách được Real và Fake?

Nếu hai lớp khác nhau về dung lượng, một ngưỡng đơn giản có thể là điểm bắt đầu. Bảng dưới dùng KiB, tức số byte chia cho 1024; histogram cho thấy hai phân bố chồng lấn đến đâu.

Ta chọn ngưỡng theo Accuracy rồi so hai cách đánh giá: chọn và chấm ngay trên cả tập, hoặc chọn trên phần train của fold 0 rồi chấm trên validation. ROC-AUC dùng `-size_kib`, tương ứng với phỏng đoán ảnh Fake nhỏ hơn. Accuracy ở đây cần được phân biệt với Macro-F1 dùng để chấm bài thi.

<!-- ailaai-cell:12:code -->
# ailaai-source: src/ailaai/teaching.py::choose_size_rule
def choose_size_rule(rows):
    """Select a JPEG-size threshold and direction on the supplied training rows."""
    values = np.sort(rows.size_kib.unique())
    candidates = np.r_[values[0] - 1, (values[:-1] + values[1:]) / 2, values[-1] + 1]
    best = None
    for fake_is_small in (True, False):
        for threshold in candidates:
            prediction = (rows.size_kib < threshold) if fake_is_small else (rows.size_kib >= threshold)
            accuracy = float((prediction.astype(int) == rows.label).mean())
            if best is None or accuracy > best["accuracy"]:
                best = {"threshold_kib": float(threshold), "fake_is_small": fake_is_small,
                        "accuracy": accuracy}
    return best

<!-- ailaai-cell:13:code -->
from sklearn.metrics import accuracy_score, roc_auc_score
size_summary = census.groupby("label").size_kib.agg(["mean", "median", "std", "min", "max"])
display(size_summary.rename(index={0: "Real", 1: "Fake"}))
fig, ax = plt.subplots(figsize=(8, 4))
for label, name in [(0, "Real"), (1, "Fake")]:
    ax.hist(census.loc[census.label == label, "size_kib"], bins=30, alpha=.5, label=name)
ax.set(xlabel="Dung lượng JPEG (KiB)", ylabel="Số ảnh", title="Dung lượng: hai lớp chồng lấn đến đâu?")
ax.legend(); plt.show()
all_rule = choose_size_rule(census)
fit_census = census[census.file_name.isin(fit_rows.file_name)]
val_census = census[census.file_name.isin(val_rows.file_name)]
fit_rule = choose_size_rule(fit_census)
val_pred = (val_census.size_kib < fit_rule["threshold_kib"]) if fit_rule["fake_is_small"] else (val_census.size_kib >= fit_rule["threshold_kib"])
print("Chọn + chấm cả tập (mô tả):", all_rule)
print("Chọn train fold 0:", fit_rule, "| Accuracy validation:", accuracy_score(val_census.label, val_pred))
print("ROC-AUC mô tả, score=-size:", roc_auc_score(census.label, -census.size_kib))

<!-- ailaai-cell:14:markdown -->
### Ảnh xám có phải đều là Real?

Bảng chéo đếm trực tiếp ảnh xám và ảnh màu trong mỗi lớp. Sau đó ta thử quy tắc “xám là Real, màu là Fake” và tính cả Accuracy lẫn Macro-F1.

Nếu hai lớp có cùng tỷ lệ ảnh xám, riêng cờ này sẽ không giúp tách lớp. Các thông tin màu khác vẫn có thể hữu ích.

<!-- ailaai-cell:15:code -->
from sklearn.metrics import f1_score
color_table = pd.crosstab(census.label, census.is_gray).reindex(index=[0, 1], columns=[False, True], fill_value=0)
color_table.index = ["Real", "Fake"]; color_table.columns = ["Màu", "Xám"]
display(color_table)
gray_prediction = (~census.is_gray).astype(int)
print("Quy tắc xám→Real, màu→Fake:",
      {"Accuracy": accuracy_score(census.label, gray_prediction),
       "Macro-F1": f1_score(census.label, gray_prediction, average="macro", labels=[0, 1], zero_division=0)})
fig, ax = plt.subplots(figsize=(6, 3.5)); color_table.plot.bar(ax=ax, rot=0)
ax.set(ylabel="Số ảnh", title="Nhãn × xám/màu"); fig.tight_layout(); plt.show()

<!-- ailaai-cell:16:markdown -->
### Chia và kiểm tra năm fold

Các thí nghiệm dùng bảng chia cố định đã phát hành. Mỗi ảnh phải xuất hiện đúng một lần ở validation và không nằm trong train của chính fold đó.

Đoạn `StratifiedKFold` minh họa cách giữ tỷ lệ nhãn khi chia dữ liệu. Biến `demo_fold` chỉ dùng để quan sát; các lượt huấn luyện vẫn đọc bảng trong `assets/splits/train_folds.csv`.

<!-- ailaai-cell:17:code -->
from sklearn.model_selection import StratifiedKFold
fold_table = pd.read_csv(TASK / "assets/splits/train_folds.csv")
assert fold_table.file_name.is_unique and set(fold_table.file_name) == set(train.file_name)
assert set(fold_table.fold) == set(range(5))
display(pd.crosstab(fold_table.fold, fold_table.label))
seen_validation = []
for fold in range(5):
    tr, va = load_fold_split(train, TASK / "assets/splits/train_folds.csv", fold)
    assert set(tr.file_name).isdisjoint(va.file_name)
    seen_validation.extend(va.file_name)
assert len(seen_validation) == len(set(seen_validation)) == len(train)
demo_fold = np.full(len(train), -1)
for fold, (_, ids) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(train, train.label)):
    demo_fold[ids] = fold
print("Demo StratifiedKFold:", np.bincount(demo_fold))
# Nếu hai ảnh trùng byte rơi vào hai fold khác nhau, phải xử lý trước khi diễn giải điểm.
if "census" in globals():
    hash_folds = census.merge(fold_table[["file_name", "fold"]], on="file_name", validate="one_to_one")
    assert not (hash_folds.groupby("sha256").fold.nunique() > 1).any(), "Trùng nội dung qua fold"

<!-- ailaai-cell:18:markdown -->
## 2. Cách tính Macro-F1

Bài thi dùng Macro-F1 để đánh giá. Chỉ số này tính $F_1$ cho từng lớp rồi lấy trung bình:

$$\text{Macro-F1} = \frac{F_{1, \text{Real}} + F_{1, \text{Fake}}}{2}$$

Cell dưới tính điểm từ ma trận nhầm lẫn (TP, FP, FN, TN) và đối chiếu với `scikit-learn`.

<!-- ailaai-cell:19:code -->
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

<!-- ailaai-cell:20:markdown -->
### Ví dụ All-Fake trong Reading II.1.3

Accuracy và F1 từng lớp trả lời các câu hỏi khác nhau. Đoạn dưới dùng nhãn train thật và dự đoán tất cả là Fake; hàng là nhãn thật, cột là dự đoán.

<!-- ailaai-cell:21:code -->
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
all_fake = np.ones(len(train), dtype=int)
cm = confusion_matrix(train.label, all_fake, labels=[0, 1])
precision, recall, f1, support = precision_recall_fscore_support(train.label, all_fake, labels=[0, 1], zero_division=0)
display(pd.DataFrame(cm, index=["Real thật", "Fake thật"], columns=["Đoán Real", "Đoán Fake"]))
display(pd.DataFrame({"Lớp": ["Real", "Fake"], "Precision": precision, "Recall": recall, "F1": f1, "n": support}))
print("Accuracy:", accuracy_score(train.label, all_fake), "| Macro-F1:", f1.mean())

<!-- ailaai-cell:22:markdown -->
## 3. Theo dõi một bước học của CNN2

Ta dùng CNN2 với hai tầng tích chập để xem dữ liệu đi qua mạng như thế nào: tensor ảnh, các tầng tích chập, gộp thích ứng (adaptive pooling), rồi tầng phân loại.

Sau khi có đầu ra, ta tính loss Cross-Entropy, lan truyền ngược và cập nhật trọng số trong một bước tối ưu.

<!-- ailaai-cell:23:code -->
# ailaai-source: src/ailaai/teaching.py::BaselineCNN2,build_resnet34
class BaselineCNN2(nn.Module):
    """The two Conv-BatchNorm-ReLU-Pool blocks printed in Reading III.1."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))
        attach_optimizer_groups(self, self.classifier)

    def forward(self, images):
        return self.classifier(self.features(images))


def build_resnet34(pretrained=True, freeze_backbone=False):
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet34(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model._freeze_backbone = freeze_backbone
    return attach_optimizer_groups(model, model.fc)

<!-- ailaai-cell:24:code -->
demo_model = BaselineCNN2()
demo_optimizer = torch.optim.AdamW(demo_model.parameters(), lr=1e-3, weight_decay=1e-4)
demo_logits = demo_model(torch.rand(2, 3, 64, 64))
demo_labels = torch.tensor([0, 1], dtype=torch.long)
demo_loss = nn.functional.cross_entropy(demo_logits, demo_labels)
demo_optimizer.zero_grad(set_to_none=True)
demo_loss.backward(); demo_optimizer.step()
print("Logits:", tuple(demo_logits.shape), "P(Fake):", demo_logits.softmax(1)[:, 1].detach(), "Loss:", demo_loss.item())

<!-- ailaai-cell:25:markdown -->
### Viết hai phép biến đổi ảnh

Hai hàm cùng lấy vùng crop 358 × 358. Nhánh Resampled đưa vùng đó qua phép nội suy bilinear 358 → 224 → 358, còn Native giữ nguyên pixel.

<!-- ailaai-cell:26:code -->
# ailaai-source: src/ailaai/transforms.py::native_view,resampled_view
def native_view(images: torch.Tensor, crop: int = 358) -> torch.Tensor:
    """Take the centered crop without rescaling its pixels."""
    if images.ndim not in (3, 4):
        raise ValueError("native_view expects CHW or BCHW RGB tensors.")
    return TF.center_crop(images, [crop, crop])


def resampled_view(
    images: torch.Tensor,
    crop: int = 358,
    size: int = 224,
    interpolation: Any = InterpolationMode.BILINEAR,
    antialias: bool = True,
) -> torch.Tensor:
    """Keep the Native crop's field of view while sampling through a smaller grid."""
    cropped = native_view(images, crop=crop)
    small = TF.resize(cropped, [size, size], interpolation=interpolation, antialias=antialias)
    return TF.resize(small, [crop, crop], interpolation=interpolation, antialias=antialias)

<!-- ailaai-cell:27:markdown -->
### Đọc điểm số cùng điều kiện đánh giá

| Phép thử | Dữ liệu đánh giá | Điều kiện |
|---|---|---|
| E1 CNN2 / Frozen / Fine-tuning | Fold 0, 400 ảnh | CNN2 64; ResNet resize 224; không TTA/smoothing |
| E2 Same-FOV | Fold 0 và 1, 800 ảnh | cùng crop và tensor 358; không TTA/smoothing |
| RGB/HP/Mean lịch sử | 5-fold OOF, 2.000 ảnh | smoothing 0,03; flip TTA; terminal epoch 15 |
| Lượt học mặc định hiện tại | Fold 0, 400 ảnh | theo cấu hình in trong notebook; không mặc nhiên là OOF lịch sử |

Có một chỗ chưa thống nhất trong Reading: phần III.1 mô tả CNN2 nhận ảnh 64, còn chú thích bảng E1 ghi chung Resize 224. Notebook dùng 64 cho CNN2 và 224 cho ResNet. Khi so sánh, cần giữ khác biệt này cùng với số fold, TTA và làm mịn nhãn trong bảng trên.

<!-- ailaai-cell:28:markdown -->
### Đọc ảnh thành batch

`FaceDataset` trả về ảnh RGB, nhãn và tên tệp. Ảnh còn ở dải [0, 1]; sau khi ghép thành batch, ta mới crop hoặc lọc High-pass, rồi chuẩn hóa bằng mean/std của ImageNet.

Các hàm dưới được gọi trực tiếp trong phần huấn luyện. Bạn có thể sửa từng bước ngay tại cell để quan sát tác động. Mã tương ứng nằm trong `src/ailaai/data.py`, `models.py` và `engine.py`; phần giải thích là Reading III.5.

Preset bài học để smoothing = 0 và TTA = False. Khi đối chiếu bảng OOF lịch sử, nhớ rằng bảng đó dùng smoothing = 0,03 và flip TTA.

<!-- ailaai-cell:29:code -->
# ailaai-source: src/ailaai/data.py::FaceDataset,make_loader
class FaceDataset(Dataset[tuple[torch.Tensor, int, str]]):
    """Decode images without resizing; the engine applies the injected view."""

    def __init__(self, rows: pd.DataFrame, labeled: bool = True) -> None:
        self.rows = rows.reset_index(drop=True).copy()
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows.iloc[index]
        label = int(row.label) if self.labeled else -1
        return decode_rgb(row.path), label, str(row.file_name)


def make_loader(
    rows: pd.DataFrame,
    batch_size: int,
    labeled: bool = True,
    shuffle: bool = False,
    seed: int = 2026,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a worker-safe loader for notebook-defined transform functions."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(FaceDataset(rows, labeled=labeled), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=generator)

<!-- ailaai-cell:30:markdown -->
### Xử lý ảnh trước khi đưa vào mạng

Ảnh train có thể được lật ngẫu nhiên để tăng dữ liệu. Ảnh validation giữ nguyên thứ tự và không áp dụng phép lật ngẫu nhiên này. Cả hai đều đi qua cùng phép crop/lọc, sau đó mới được chuẩn hóa.

<!-- ailaai-cell:31:code -->
# ailaai-source: src/ailaai/engine.py::_normalize,_view_batch
def _normalize(images: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    mean = images.new_tensor(cfg.normalize_mean).view(1, 3, 1, 1)
    std = images.new_tensor(cfg.normalize_std).view(1, 3, 1, 1)
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"view_fn must return BCHW RGB, received {tuple(images.shape)}")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("view_fn must return finite values in [0, 1] before normalization.")
    return (images - mean) / std


def _view_batch(batch: torch.Tensor, view_fn: ViewFunction, cfg: TrainConfig, augment: bool) -> torch.Tensor:
    if augment:
        probability = float(cfg.augmentation.get("horizontal_flip_probability", 0.0))
        if not 0 <= probability <= 1:
            raise ValueError("horizontal_flip_probability must be in [0, 1].")
        flips = torch.rand(batch.shape[0], device=batch.device) < probability
        batch = torch.where(flips[:, None, None, None], batch.flip(-1), batch)
    viewed = view_fn(batch)
    if not isinstance(viewed, torch.Tensor) or viewed.shape[0] != batch.shape[0]:
        raise ValueError("view_fn must return a tensor with the same batch size.")
    return _normalize(viewed, cfg)

<!-- ailaai-cell:32:markdown -->
### Những tham số nào được cập nhật?

Backbone đã học từ ImageNet, còn head được khởi tạo cho bài toán hai lớp, nên ta có thể đặt learning rate riêng cho hai phần. Nếu khóa backbone, optimizer chỉ cập nhật head. Các lớp BatchNorm trong backbone cũng phải giữ ở chế độ `eval` để thống kê của chúng không tiếp tục đổi.

<!-- ailaai-cell:33:code -->
# ailaai-source: src/ailaai/models.py::optimizer_parameter_groups
def optimizer_parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list[dict[str, Any]]:
    """Return nonempty, disjoint optimizer groups for backbone and classifier."""
    groups = {"backbone": [], "head": []}
    for parameter in model.parameters():
        if parameter.requires_grad:
            group = getattr(parameter, "_ailaai_group", "backbone")
            if group not in groups:
                raise ValueError(f"Unknown optimizer parameter group {group!r}.")
            groups[group].append(parameter)
    if not groups["head"]:
        raise ValueError("Classifier optimizer group must contain trainable parameters.")
    return [{"params": params, "lr": lr} for params, lr in
            [(groups["backbone"], backbone_lr), (groups["head"], head_lr)] if params]

<!-- ailaai-cell:34:code -->
# ailaai-source: src/ailaai/engine.py::_optimizer
def _optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = optimizer_parameter_groups(model, cfg.backbone_lr, cfg.head_lr)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

<!-- ailaai-cell:35:markdown -->
### Theo dõi một epoch huấn luyện

Mạng trả hai logits cho mỗi ảnh. Cross-Entropy nhận các logits này cùng nhãn kiểu `long`; làm mịn nhãn và trọng số từng ảnh cũng được áp dụng tại bước tính loss.

Sau `backward`, optimizer cập nhật tham số. Khi dùng gradient accumulation, vài batch góp gradient trước một lần cập nhật; nhóm cuối có thể ít batch hơn nên phải chia loss theo đúng số batch của nhóm đó. AMP được bật khi chạy trên CUDA.

<!-- ailaai-cell:36:code -->
# ailaai-source: src/ailaai/engine.py::train_one_epoch
def train_one_epoch(model, epoch_train, optimizer, scaler, cfg, view_fn, device, sample_weights=None):
    """One explicit optimization epoch, also usable for a bounded CPU exercise."""
    model.train()
    if getattr(model, "_freeze_backbone", False):
        model.eval()
        model.fc.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    correct = 0
    seen = 0
    accumulation = cfg.accumulation
    for step, (images, target, file_names) in enumerate(epoch_train):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prepared = _view_batch(images, view_fn, cfg, augment=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(prepared)
            if sample_weights is not None:
                losses = nn.functional.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing, reduction="none")
                weights = losses.new_tensor([sample_weights[name] for name in file_names])
                loss = (losses * weights).mean()
            else:
                loss = nn.functional.cross_entropy(logits, target, label_smoothing=cfg.label_smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite.")
        loss_value = float(loss.detach())
        group_start = (step // accumulation) * accumulation
        batches_per_group = min(accumulation, len(epoch_train) - group_start)
        scaler.scale(loss / batches_per_group).backward()
        boundary = (step + 1) % accumulation == 0 or step + 1 == len(epoch_train)
        if boundary:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss_value * len(target)
        correct += int((logits.detach().argmax(1) == target).sum())
        seen += len(target)
    return total_loss, correct, seen

<!-- ailaai-cell:37:markdown -->
### Đánh giá validation và thử flip TTA

`softmax(logits)[:, 1]` cho xác suất Fake. Khi bật TTA, mô hình dự đoán thêm ảnh lật ngang rồi lấy trung bình hai xác suất. Loss validation trong code vẫn được tính trên ảnh gốc.

Toàn bộ bước này chạy với `model.eval()` và không tính gradient. Nhãn validation chỉ dùng để đo loss và Macro-F1.

<!-- ailaai-cell:38:code -->
# ailaai-source: src/ailaai/engine.py::_validation_predictions
def _validation_predictions(
    model: nn.Module,
    loader: DataLoader,
    cfg: TrainConfig,
    view_fn: ViewFunction,
    device: torch.device,
) -> tuple[PredictionTable, float, float]:
    model.eval()
    names: list[str] = []
    labels: list[int] = []
    probabilities: list[float] = []
    losses = 0.0
    with torch.no_grad():
        for images, target, file_names in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prepared = _view_batch(images, view_fn, cfg, augment=False)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(prepared)
                loss = nn.functional.cross_entropy(logits, target)
            probability = logits.float().softmax(dim=1)[:, 1]
            if cfg.tta:
                flipped = _view_batch(images.flip(-1), view_fn, cfg, augment=False)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    flip_probability = model(flipped).float().softmax(1)[:, 1]
                probability = 0.5 * (probability + flip_probability)
            names.extend(file_names)
            labels.extend(target.cpu().tolist())
            probabilities.extend(probability.cpu().tolist())
            losses += float(loss.detach()) * len(target)
    table = PredictionTable(pd.DataFrame({"file_name": names, "label": labels, "prob": probabilities}))
    report = classification_report(labels, probabilities)
    return table, losses / len(labels), float(report["macro_f1"])

<!-- ailaai-cell:39:markdown -->
### Chạy đủ số epoch và lưu kết quả

`fit_fold` ghép các bước vừa viết thành một lượt huấn luyện. Sau mỗi epoch, hàm lưu đường học và checkpoint; ở epoch cuối, nó xuất dự đoán kèm tên ảnh để dùng cho phần so sánh.

Checkpoint giữ cả trạng thái optimizer, scheduler và bộ sinh số ngẫu nhiên để có thể học tiếp sau khi kernel khởi động lại. Nếu chọn `load_run`, cấu hình sẽ được kiểm tra trước khi nạp kết quả.

<!-- ailaai-cell:40:code -->
# ailaai-source: src/ailaai/engine.py::fit_fold,load_run
def fit_fold(
    workspace: Workspace,
    cfg: TrainConfig,
    branch: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    model_factory: ModelFactory,
    view_fn: ViewFunction,
    view_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    sample_weights: Mapping[str, float] | None = None,
    *, allow_cpu: bool = False,
) -> RunInfo:
    """Fit one branch on the provided training rows and validate each epoch."""
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("fit_fold needs a CUDA GPU. Use load_run for a released checkpoint.")
    if branch not in {"rgb", "highpass", "resampled", "wavelet", "edge_weighted"}:
        raise ValueError("Unknown training branch.")
    if train_rows.empty or val_rows.empty or set(train_rows.file_name) & set(val_rows.file_name):
        raise ValueError("Training and validation rows must be nonempty and disjoint.")
    resolved = _resolved_config(workspace, cfg, branch, train_rows, val_rows, model_factory, view_fn, view_spec, model_spec, sample_weights)
    sample_weights = resolved["recipe"].get("sample_weights")
    run_dir = _run_directory(workspace, branch, resolved)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Lượt chạy {branch}: {run_dir}", flush=True)
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists() and not config_path.exists():
        raise ValueError(f"Checkpoint has no saved recipe; move it aside or use a new run_id: {run_dir}")
    if config_path.exists() and json.loads(config_path.read_text()) != resolved:
        raise ValueError(f"Run configuration changed; choose a new run_id. Existing run: {run_dir}")
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if status.get("status") == "complete" and checkpoint_path.is_file():
        print("Đã có lượt chạy hoàn tất; đang nạp kết quả.", flush=True)
        return load_run(workspace, cfg, branch, train_rows, val_rows, model_factory,
                        view_fn, view_spec, model_spec, checkpoint_source=checkpoint_path, sample_weights=sample_weights)
    write_json(config_path, resolved)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _seed(cfg.seed)
    model = model_factory(initialize=not checkpoint_path.is_file()).to(device)
    optimizer = _optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    validation_loader = make_loader(val_rows, cfg.batch_size, labeled=True, shuffle=False, seed=cfg.seed)
    curves_path = run_dir / "curves.csv"
    start_epoch = 0
    records = pd.read_csv(curves_path).to_dict("records") if curves_path.is_file() else []
    if checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        start_epoch = int(state["epoch"])
        records = [record for record in records if int(record["epoch"]) <= start_epoch]
    write_json(run_dir / "run_status.json", {"status": "running", "start_epoch": start_epoch, "config_sha256": sha256_file(config_path)})
    for epoch in range(start_epoch, cfg.epochs):
        tick = time.monotonic()
        epoch_train = make_loader(train_rows, cfg.batch_size, labeled=True, shuffle=True, seed=cfg.seed + epoch)
        total_loss, correct, seen = train_one_epoch(
            model, epoch_train, optimizer, scaler, cfg, view_fn, device, sample_weights)
        scheduler.step()
        predictions, val_loss, val_f1 = _validation_predictions(model, validation_loader, cfg, view_fn, device)
        records.append({"epoch": epoch + 1, "train_loss": total_loss / seen, "train_accuracy": correct / seen,
                        "val_loss": val_loss, "val_macro_f1": val_f1, "seconds": time.monotonic() - tick,
                        "backbone_lr": optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
                        "head_lr": optimizer.param_groups[-1]["lr"]})
        _save_curves(curves_path, records)
        pending_checkpoint = checkpoint_path.with_suffix(".pt.partial")
        torch.save(_checkpoint_state(model, optimizer, scheduler, scaler, epoch + 1), pending_checkpoint)
        pending_checkpoint.replace(checkpoint_path)
        print(f"{branch} | epoch {epoch + 1}/{cfg.epochs} | loss {total_loss / seen:.4f} | "
              f"val F1 {val_f1:.4f} | {records[-1]['seconds']:.1f}s", flush=True)
    if len(records) < cfg.epochs:
        raise RuntimeError("Training ended before the configured terminal epoch.")
    prediction_meta = {
        "source": "learner_run", "run_id": workspace.run_id, "branch": branch,
        "checkpoint_sha256": sha256_file(checkpoint_path), "config_sha256": sha256_file(config_path),
        "split_sha256": resolved["split_sha256"], "validation_ids_sha256": resolved["validation_ids_sha256"],
        "validation_manifest": "validation_manifest.csv",
        "model_spec": dict(model_spec), "view_spec": dict(view_spec), "label_map": {"0": "Real", "1": "Fake"},
    }
    predictions, _, _ = _validation_predictions(model, validation_loader, cfg, view_fn, device)
    predictions = PredictionTable(predictions.rows, prediction_meta)
    predictions.save(run_dir / "val_predictions.csv")
    pd.DataFrame(val_rows).to_csv(run_dir / "validation_manifest.csv", index=False)
    write_json(run_dir / "run_status.json", {"status": "complete", "epochs": cfg.epochs,
                                             "checkpoint_sha256": prediction_meta["checkpoint_sha256"]})
    write_json(workspace.artifact_root / f"{branch}_active.json", {"directory": run_dir.name})
    result = RunInfo(workspace.run_id, branch, run_dir, checkpoint_path, pd.DataFrame(records), predictions, prediction_meta)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def load_run(
    workspace: Workspace,
    cfg: TrainConfig,
    branch: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    model_factory: ModelFactory,
    view_fn: ViewFunction,
    view_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    checkpoint_source: str | Path | None = None,
    sample_weights: Mapping[str, float] | None = None,
) -> RunInfo:
    """Load a completed branch after exact config and callable checks."""
    run, model, device = _load_checkpoint(workspace, cfg, branch, train_rows, val_rows,
                                          model_factory, view_fn, view_spec, model_spec, checkpoint_source, sample_weights)
    if not run.val_predictions.rows.file_name.isin(val_rows.file_name).all() or len(run.val_predictions.rows) != len(val_rows):
        raise ValueError("Stored validation predictions do not cover the current fold.")
    write_json(workspace.artifact_root / f"{branch}_active.json", {"directory": run.run_dir.name})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return run

<!-- ailaai-cell:41:code -->
from dataclasses import replace
from ailaai.config import load_config
BASELINE_EPOCHS = 1  # giữ lựa chọn trong bản bạn tải; đổi 15 cho lượt đầy đủ
cfg = replace(load_config(TASK / "configs/rgb_native358.json"), epochs=BASELINE_EPOCHS)
MODEL_SPEC = dict(cfg.model)

<!-- ailaai-cell:42:markdown -->
### E1: từ CNN2 đến học chuyển giao

Bật `RUN_E1` để chạy ba phương án, mỗi phương án 15 epoch: CNN2 học từ đầu, ResNet34 chỉ học head và ResNet34 fine-tuning toàn bộ mạng. Với phương án khóa backbone, cả BatchNorm được giữ ở `eval`, còn head vẫn ở `train`.

Sau khi chạy, so đường train loss với val loss rồi xem Macro-F1. Các biểu đồ lấy từ lượt vừa thực hiện; mức độ khớp với số liệu lịch sử còn phụ thuộc cấu hình và điều kiện chạy.

<!-- ailaai-cell:43:code -->
RUN_E1 = False
E1_EPOCHS = 15
E1_RESULTS = {}
def resize64(batch): return TF.resize(batch, [64, 64], antialias=True)
def resize224(batch): return TF.resize(batch, [224, 224], antialias=True)
def cnn_factory(*, initialize): return BaselineCNN2()
def frozen_factory(*, initialize): return build_resnet34(pretrained=initialize, freeze_backbone=True)
def full_factory(*, initialize): return build_resnet34(pretrained=initialize, freeze_backbone=False)
if RUN_E1:
    from ailaai.visuals import show_learning_curves
    for name, factory, view, size, backbone, frozen, lr in [
        ("cnn2", cnn_factory, resize64, 64, "cnn2", False, 1e-3),
        ("frozen", frozen_factory, resize224, 224, "resnet34", True, 1.5e-4),
        ("finetune", full_factory, resize224, 224, "resnet34", False, 1.5e-4)]:
        spec = {"backbone": backbone, "weights": None if name == "cnn2" else "IMAGENET1K_V1", "frozen": frozen}
        view_spec = {"name": "resize", "size": size}
        experiment_cfg = replace(cfg, epochs=E1_EPOCHS, model=spec, view=view_spec,
                                 backbone_lr=lr, head_lr=1e-3 if name == "cnn2" else 7.5e-4,
                                 tta=False, label_smoothing=0.0)
        experiment_ws = Workspace.from_root(TASK, run_id=f"{RUN_ID}_e1_{name}")
        result = fit_fold(experiment_ws, experiment_cfg, "rgb", fit_rows, val_rows,
                          factory, view, view_spec, spec)
        E1_RESULTS[name] = result
        display(result.curves); show_learning_curves(result.curves)
    display(pd.DataFrame([{"model": name, **classification_report(r.val_predictions.rows.label, r.val_predictions.rows.prob)}
                          for name, r in E1_RESULTS.items()]))
else:
    print("E1 có đủ source; bật RUN_E1 để chạy 3 fits.")

<!-- ailaai-cell:44:markdown -->
## 4. So sánh cùng vùng nhìn: Same-FOV

Co giãn ảnh (resize) làm thay đổi các chi tiết tần số cao như thế nào?

Ta cắt vùng giữa ảnh (Center Crop), giữ lại khoảng 70% chiều dài mỗi cạnh. Với ảnh gốc $512 \times 512$, vùng cắt là $358 \times 358$ vì $0{,}7 \times 512 = 358{,}4 \approx 358$. Trên các ảnh đã xem, vùng này bao quanh mắt, mũi, miệng và hai má. Đây là lựa chọn từ quan sát, chưa có thí nghiệm xác định kích thước tối ưu.

Hai phiên bản dùng cùng vùng cắt:
- **Native 358:** Giữ nguyên pixel gốc.
- **Resampled 358:** Tái lấy mẫu (resampling) bằng cách thu nhỏ về $224 \times 224$, rồi phóng lại $358 \times 358$.

Cùng vùng nhìn và cùng kích thước đầu ra giúp ta tập trung so sánh ảnh hưởng của phép tái lấy mẫu.

<!-- ailaai-cell:45:code -->
from ailaai.data import decode_rgb
sample = train.iloc[0]
image = decode_rgb(sample.path)
native = native_view(image)
resampled = resampled_view(image, crop=358, size=224)
print("Native:", tuple(native.shape), "Resampled:", tuple(resampled.shape))
assert native.shape == resampled.shape == (3, 358, 358)

<!-- ailaai-cell:46:markdown -->
## 5. Quan sát phổ tần số 2D-FFT

Biến đổi Fourier 2D (FFT) biểu diễn các thành phần tần số của ảnh. Biểu đồ chênh lệch phổ (difference spectrum) giúp xem thành phần nào thay đổi sau khi tái lấy mẫu.

Giữ vùng cắt Native bảo toàn pixel gốc. Quan sát FFT cho thấy thay đổi về tín hiệu; để biết ảnh hưởng đến phân loại, ta còn cần so sánh kết quả mô hình.

<!-- ailaai-cell:47:code -->
from ailaai.visuals import show_views, plot_spectra
show_views(image, {"Native 358": native, "Resampled 358": resampled})
plot_spectra(native, resampled, names=("Native 358", "Resampled 358"))
plt.show()

<!-- ailaai-cell:48:markdown -->
## 6. Huấn luyện nhánh RGB Native

Hàm `student_model_factory` tạo mô hình, còn `student_view` quyết định cách xử lý ảnh. Cả hai được truyền vào `fit_fold` đã định nghĩa phía trên.

Mặc định `RUN_BASELINE = False` để bạn có thể xem EDA và các minh họa trên CPU. Khi muốn huấn luyện, bật cờ này và chọn T4 GPU. `BASELINE_EPOCHS = 1` dành cho lượt thử; đổi thành 15 để chạy đủ. Các thí nghiệm E1 và E2 có cờ bật riêng.

<!-- ailaai-cell:49:code -->
from dataclasses import replace
from ailaai.config import load_config
from ailaai.models import model_factory as build_model

# Dùng cfg và MODEL_SPEC đã chọn ở phần E1.
def student_model_factory(*, initialize):
    return build_model(MODEL_SPEC["backbone"], MODEL_SPEC["weights"], initialize=initialize)
def student_view(batch):
    return native_view(batch)
RUN_BASELINE = False
if RUN_BASELINE:
    run = fit_fold(ws, cfg, branch="rgb", train_rows=fit_rows, val_rows=val_rows,
                   model_factory=student_model_factory, view_fn=student_view,
                   view_spec=dict(cfg.view), model_spec=MODEL_SPEC)
    print(run.summary())

<!-- ailaai-cell:50:markdown -->
### E2: so sánh Native và Resampled trên hai fold

Bật `RUN_E2` để chạy cả hai nhánh trên folds 0 và 1, tổng cộng bốn lượt huấn luyện. Seed, backbone, kích thước tensor, augmentation và epoch cuối được giữ giống nhau.

Hãy xem kết quả từng fold trước khi gộp: thứ tự hai nhánh có thể đảo chiều. Macro-F1 gộp được tính lại từ 800 dự đoán, vì lấy trung bình hai F1 sẽ cho một phép đo khác.

<!-- ailaai-cell:51:code -->
RUN_E2 = False
E2_FOLDS = [0, 1]
E2_EPOCHS = 15
e2_predictions = []
if RUN_E2:
    for fold in E2_FOLDS:
        tr, va = load_fold_split(train, TASK / "assets/splits/train_folds.csv", fold)
        for branch, view, view_spec in [
            ("rgb", native_view, {"name": "native", "crop": 358}),
            ("resampled", resampled_view, {"name": "resampled", "crop": 358, "size": 224})]:
            experiment_cfg = replace(cfg, epochs=E2_EPOCHS, view=view_spec, tta=False, label_smoothing=0.0)
            experiment_ws = Workspace.from_root(TASK, run_id=f"{RUN_ID}_e2_fold{fold}")
            result = fit_fold(experiment_ws, experiment_cfg, branch, tr, va,
                              student_model_factory, view, view_spec, MODEL_SPEC)
            e2_predictions.append(result.val_predictions.rows.assign(fold=fold, branch=branch))
    e2 = pd.concat(e2_predictions, ignore_index=True)
    e2.to_csv(ws.output_root / "e2_val_predictions.csv", index=False)
    report = [{"branch": branch, "fold": fold, "n": len(part), **classification_report(part.label, part.prob)}
              for (branch, fold), part in e2.groupby(["branch", "fold"])]
    report += [{"branch": branch, "fold": "pooled", "n": len(part), **classification_report(part.label, part.prob)}
               for branch, part in e2.groupby("branch")]
    display(pd.DataFrame(report))
    paired = e2.pivot(index=["file_name", "label", "fold"], columns="branch", values="prob").reset_index()
    native_wrong = (paired.rgb >= .5) != paired.label
    resampled_wrong = (paired.resampled >= .5) != paired.label
    print("Chỉ Native sai:", int((native_wrong & ~resampled_wrong).sum()),
          "| Chỉ Resampled sai:", int((resampled_wrong & ~native_wrong).sum()))
else:
    print("E2 có đủ source; bật RUN_E2 để chạy 4 fits.")

<!-- ailaai-cell:52:markdown -->
## 7. Nhìn lại bài 1

Quy ước nhãn trong các bài là $0 = \text{Real}$ và $1 = \text{Fake}$. Khi đọc kết quả, xem $F_1$ của từng lớp cùng với Macro-F1 để biết mô hình đang gặp khó ở lớp nào.

So sánh FFT cho thấy resize làm thay đổi tín hiệu tần số cao. Điều này chưa đủ để kết luận cách xử lý nào phân loại tốt hơn. Các bài tiếp theo dùng vùng cắt $358 \times 358$ để xây dựng và so sánh các nhánh mô hình.

<!-- ailaai-cell:53:code -->
if RUN_BASELINE:
    from ailaai.visuals import show_learning_curves
    display(run.curves); show_learning_curves(run.curves)
    display(pd.DataFrame([classification_report(run.val_predictions.rows.label, run.val_predictions.rows.prob)]))
else:
    print("Đã chạy EDA và các minh họa; baseline chưa được huấn luyện trong lượt này.")

<!-- ailaai-cell:54:markdown -->
## Đọc thêm và xem mã nguồn

[Bản đồ Reading và notebook](../READING_NOTEBOOK_MAP.md) chỉ từng nội dung của bài đọc đến các cell và tệp Python tương ứng. Bài đọc chính là `OlympicAI_2026/topic_ai_la_ai/reading.tex`, các phần II–IV và Phụ lục B–D trong workspace bài giảng.

Dòng `# ailaai-source` ở đầu một số cell chỉ nơi lưu hàm trong `src/ailaai/`. Khi thực hành, bạn có thể sửa ngay trong cell. Khi biên soạn lại tài liệu, sửa tệp Python rồi chạy `python scripts/build_notebooks.py --write` để đồng bộ.

Tài liệu của thư viện:

- PyTorch: [học chuyển giao](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html).
- scikit-learn: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Khi báo kết quả, ghi kèm cấu hình và số ảnh đã đánh giá. Các ví dụ CPU giúp kiểm tra cách tính; số liệu `reference` là kết quả đã lưu. Muốn đánh giá lượt huấn luyện của mình, dùng checkpoint, log và dự đoán do lượt đó tạo ra.
