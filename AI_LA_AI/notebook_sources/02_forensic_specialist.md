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
RUN_ID = "lesson_nb2_reading_v2"
ws = Workspace.from_root(TASK, run_id=RUN_ID)
check_environment(profile="e2e")
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
## 1. Cấu hình nhánh High-pass

Để so sánh trên cùng dữ liệu, High-pass dùng vùng cắt $358 \times 358$ và bảng chia fold giống nhánh RGB. Bộ dữ liệu có tại [who_is_AI (Google Drive)](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view?usp=drive_link).

Xem các tham số trong `configs/highpass358.json`: kích thước nhân Gaussian $k = 5$, độ lệch chuẩn $\sigma = 1.0$, hệ số khuếch đại (gain) và độ dịch (offset).

<!-- ailaai-cell:05:code -->
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

<!-- ailaai-cell:06:markdown -->
### Từ Gaussian 1D đến bộ lọc 2D

Tạo kernel chuẩn hóa tổng bằng 1, tích ngoài tạo kernel 2D; `groups=3` lọc độc lập ba kênh, reflect padding giữ kích thước.

<!-- ailaai-cell:07:code -->
# ailaai-source: src/ailaai/forensics.py::gaussian_blur_rgb
def gaussian_blur_rgb(images: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Apply a channelwise Gaussian blur to CHW or BCHW RGB tensors."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
        squeeze = True
    elif images.ndim == 4:
        squeeze = False
    else:
        raise ValueError("Gaussian blur expects CHW or BCHW RGB tensors.")
    if images.shape[1] != 3 or kernel_size < 1 or kernel_size % 2 == 0 or sigma <= 0:
        raise ValueError("Expected three channels, a positive odd kernel, and positive sigma.")
    axis = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(axis * axis) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
    pad = kernel_size // 2
    blurred = F.conv2d(F.pad(images, (pad, pad, pad, pad), mode="reflect"), kernel, groups=3)
    return blurred.squeeze(0) if squeeze else blurred

<!-- ailaai-cell:08:markdown -->
## 2. Lấy ảnh gốc trừ đi ảnh làm mờ

Với ảnh gốc $I$, ta làm mờ bằng Gaussian rồi tính phần dư:

$$R = I - (G_\sigma * I)$$

Ảnh làm mờ giảm các thay đổi nhanh theo không gian. Phần dư làm nổi các thay đổi cục bộ, chẳng hạn chi tiết bề mặt và nhiễu. Các chi tiết này không tự xác định ảnh là thật hay giả.

Ta nhân phần dư với gain, cộng offset 0.5 và đưa giá trị về dải $[0, 1]$. Hàm `highpass_view` xử lý được cả một ảnh dạng CHW và một batch dạng BCHW.

<!-- ailaai-cell:09:code -->
import torch
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
plt.show()

<!-- ailaai-cell:10:markdown -->
### Xem riêng ảnh mờ, residual và ảnh sau khuếch đại

Residual có giá trị âm; thang màu đối xứng quanh 0 giúp thấy dấu. Không chuẩn hóa min-max từng ảnh trước khi cho mô hình học.

<!-- ailaai-cell:11:code -->
low = gaussian_blur_rgb(rgb, VIEW_SPEC["kernel"], VIEW_SPEC["sigma"])
residual = rgb - low
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
axes[0].imshow(low.permute(1, 2, 0)); axes[0].set_title("Gaussian low-pass")
limit = max(float(residual.abs().max()), 1e-6)
im = axes[1].imshow(residual.mean(0), cmap="coolwarm", vmin=-limit, vmax=limit)
axes[1].set_title("Residual có dấu"); fig.colorbar(im, ax=axes[1], shrink=.7)
axes[2].imshow(filtered.permute(1, 2, 0)); axes[2].set_title("gain × residual + offset")
for ax in axes: ax.axis("off")
fig.tight_layout(); plt.show()
print("Residual range:", float(residual.min()), float(residual.max()))

<!-- ailaai-cell:12:markdown -->
### Đọc điểm số cùng điều kiện đánh giá

| Phép thử | Dữ liệu đánh giá | Điều kiện |
|---|---|---|
| E1 CNN2 / Frozen / Fine-tuning | Fold 0, 400 ảnh | CNN2 64; ResNet resize 224; không TTA/smoothing |
| E2 Same-FOV | Fold 0 và 1, 800 ảnh | cùng crop và tensor 358; không TTA/smoothing |
| RGB/HP/Mean lịch sử | 5-fold OOF, 2.000 ảnh | smoothing 0,03; flip TTA; terminal epoch 15 |
| Lượt học mặc định hiện tại | Fold 0, 400 ảnh | theo cấu hình in trong notebook; không mặc nhiên là OOF lịch sử |

Có một chỗ chưa thống nhất trong Reading: phần III.1 mô tả CNN2 nhận ảnh 64, còn chú thích bảng E1 ghi chung Resize 224. Notebook dùng 64 cho CNN2 và 224 cho ResNet. Khi so sánh, cần giữ khác biệt này cùng với số fold, TTA và làm mịn nhãn trong bảng trên.

<!-- ailaai-cell:13:markdown -->
### Đọc ảnh thành batch

`FaceDataset` trả về ảnh RGB, nhãn và tên tệp. Ảnh còn ở dải [0, 1]; sau khi ghép thành batch, ta mới crop hoặc lọc High-pass, rồi chuẩn hóa bằng mean/std của ImageNet.

Các hàm dưới được gọi trực tiếp trong phần huấn luyện. Bạn có thể sửa từng bước ngay tại cell để quan sát tác động. Mã tương ứng nằm trong `src/ailaai/data.py`, `models.py` và `engine.py`; phần giải thích là Reading III.5.

Preset bài học để smoothing = 0 và TTA = False. Khi đối chiếu bảng OOF lịch sử, nhớ rằng bảng đó dùng smoothing = 0,03 và flip TTA.

<!-- ailaai-cell:14:code -->
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

<!-- ailaai-cell:15:markdown -->
### Xử lý ảnh trước khi đưa vào mạng

Ảnh train có thể được lật ngẫu nhiên để tăng dữ liệu. Ảnh validation giữ nguyên thứ tự và không áp dụng phép lật ngẫu nhiên này. Cả hai đều đi qua cùng phép crop/lọc, sau đó mới được chuẩn hóa.

<!-- ailaai-cell:16:code -->
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

<!-- ailaai-cell:17:markdown -->
### Những tham số nào được cập nhật?

Backbone đã học từ ImageNet, còn head được khởi tạo cho bài toán hai lớp, nên ta có thể đặt learning rate riêng cho hai phần. Nếu khóa backbone, optimizer chỉ cập nhật head. Các lớp BatchNorm trong backbone cũng phải giữ ở chế độ `eval` để thống kê của chúng không tiếp tục đổi.

<!-- ailaai-cell:18:code -->
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

<!-- ailaai-cell:19:code -->
# ailaai-source: src/ailaai/engine.py::_optimizer
def _optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = optimizer_parameter_groups(model, cfg.backbone_lr, cfg.head_lr)
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)

<!-- ailaai-cell:20:markdown -->
### Theo dõi một epoch huấn luyện

Mạng trả hai logits cho mỗi ảnh. Cross-Entropy nhận các logits này cùng nhãn kiểu `long`; làm mịn nhãn và trọng số từng ảnh cũng được áp dụng tại bước tính loss.

Sau `backward`, optimizer cập nhật tham số. Khi dùng gradient accumulation, vài batch góp gradient trước một lần cập nhật; nhóm cuối có thể ít batch hơn nên phải chia loss theo đúng số batch của nhóm đó. AMP được bật khi chạy trên CUDA.

<!-- ailaai-cell:21:code -->
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

<!-- ailaai-cell:22:markdown -->
### Đánh giá validation và thử flip TTA

`softmax(logits)[:, 1]` cho xác suất Fake. Khi bật TTA, mô hình dự đoán thêm ảnh lật ngang rồi lấy trung bình hai xác suất. Loss validation trong code vẫn được tính trên ảnh gốc.

Toàn bộ bước này chạy với `model.eval()` và không tính gradient. Nhãn validation chỉ dùng để đo loss và Macro-F1.

<!-- ailaai-cell:23:code -->
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

<!-- ailaai-cell:24:markdown -->
### Chạy đủ số epoch và lưu kết quả

`fit_fold` ghép các bước vừa viết thành một lượt huấn luyện. Sau mỗi epoch, hàm lưu đường học và checkpoint; ở epoch cuối, nó xuất dự đoán kèm tên ảnh để dùng cho phần so sánh.

Checkpoint giữ cả trạng thái optimizer, scheduler và bộ sinh số ngẫu nhiên để có thể học tiếp sau khi kernel khởi động lại. Nếu chọn `load_run`, cấu hình sẽ được kiểm tra trước khi nạp kết quả.

<!-- ailaai-cell:25:code -->
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

<!-- ailaai-cell:26:markdown -->
### Tạo ResNet và thay tầng phân loại

Khi bắt đầu học, mô hình nạp trọng số ImageNet rồi thay classifier bằng tầng hai đầu ra. Khi đọc checkpoint, ta chỉ cần dựng lại kiến trúc; trọng số sẽ lấy từ checkpoint đó.

<!-- ailaai-cell:27:code -->
from torchvision import models as tv_models
_BUILDERS = {"resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights),
             "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights)}

<!-- ailaai-cell:28:code -->
# ailaai-source: src/ailaai/models.py::model_factory
def model_factory(
    backbone: str = "resnet34",
    weights: str | None = "IMAGENET1K_V1",
    initialize: bool = True,
) -> nn.Module:
    """Build a two-class model, optionally starting from ImageNet weights."""
    if backbone not in _BUILDERS:
        raise ValueError(f"Unsupported backbone {backbone!r}; choose one of {sorted(_BUILDERS)}.")
    builder, enum = _BUILDERS[backbone]
    selected = enum[weights] if initialize and weights else None
    model = builder(weights=selected)
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError(f"The {backbone} builder does not expose the expected linear fc head.")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return attach_optimizer_groups(model, model.fc)

<!-- ailaai-cell:29:code -->
build_model = model_factory

<!-- ailaai-cell:30:markdown -->
### Thử một bước tối ưu và suy luận ngay trên CPU

Dùng CNN nhỏ với 4 ảnh train để theo dõi gradient, trọng số thay đổi và xác suất TTA. Đây là bài tập kiểm tra luồng tensor, không phải kết quả ResNet34 trong Reading. Mô hình demo được tạo riêng, không dùng checkpoint của nhánh High-pass.

<!-- ailaai-cell:31:code -->
# ailaai-source: src/ailaai/teaching.py::BaselineCNN2
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

<!-- ailaai-cell:32:code -->
demo_rows = fit_rows.groupby("label", group_keys=False).head(2).reset_index(drop=True)
demo_cfg = replace(cfg, batch_size=2, accumulation=1, tta=True)
demo = BaselineCNN2()
optimizer = _optimizer(demo, demo_cfg)
scaler = torch.amp.GradScaler("cpu", enabled=False)
loader = make_loader(demo_rows, batch_size=2, shuffle=False)
initial = demo.classifier[-1].weight.detach().clone()
loss_sum, correct, seen = train_one_epoch(demo, loader, optimizer, scaler, demo_cfg, highpass_view, torch.device("cpu"))
assert not torch.equal(initial, demo.classifier[-1].weight)
predictions, val_loss, score = _validation_predictions(demo, loader, demo_cfg, highpass_view, torch.device("cpu"))
print("Bài tập 4 ảnh train:", {"loss": loss_sum/seen, "weight_changed": True, "n": seen})
display(predictions.rows)
del demo, optimizer, scaler

<!-- ailaai-cell:33:markdown -->
## 3. Huấn luyện hoặc nạp mô hình High-pass

ResNet34 nhận đầu vào từ hàm `highpass_view` vừa viết. Chọn cách chạy bằng hai biến dưới đây:

- `RUN_TRAIN = False`, `MODE = "train"`: xem các minh họa và bỏ qua huấn luyện ResNet34. Đây là lựa chọn mặc định.
- `RUN_TRAIN = True`, `MODE = "train"`: huấn luyện 15 epoch trên fold 0 với T4 GPU.
- `MODE = "load"`: đọc checkpoint đã lưu nếu có cấu hình tương ứng.

Nếu đã huấn luyện hoặc nạp được checkpoint, phần tiếp theo hiển thị đường học và dự đoán.

<!-- ailaai-cell:34:code -->
MODEL_SPEC = dict(cfg.model)
def student_model_factory(*, initialize):
    return build_model(MODEL_SPEC["backbone"], MODEL_SPEC["weights"], initialize=initialize)
MODE = "train"
RUN_TRAIN = False
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

<!-- ailaai-cell:35:markdown -->
### Đọc đường học và ca dự đoán sai

Theo dõi train loss/val loss và Macro-F1. Chỉ bảng từ lượt đã chạy mới là kết quả của cấu hình hiện tại.

<!-- ailaai-cell:36:code -->
if run is not None:
    from ailaai.visuals import show_learning_curves, show_prediction_examples
    display(run.curves); show_learning_curves(run.curves)
    pred = run.val_predictions.rows
    display(pd.DataFrame([classification_report(pred.label, pred.prob)]))
    wrong = pred.loc[(pred.prob >= .5) != pred.label].merge(val_rows[["file_name", "path"]], on="file_name", validate="one_to_one")
    display(wrong.head(10))
    selected = wrong.head(3)
    if len(selected):
        fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), squeeze=False)
        for ax, row in zip(axes[0], selected.itertuples()):
            with Image.open(row.path) as image: ax.imshow(image.convert("RGB"))
            ax.set_title(f"{row.file_name} | y={row.label} | P(HP)={row.prob:.3f}"); ax.axis("off")
        fig.tight_layout(); plt.show()

<!-- ailaai-cell:37:markdown -->
## 4. Chuẩn bị so sánh với RGB

RGB và High-pass nhận hai cách biểu diễn của cùng vùng ảnh. Điểm riêng của mỗi nhánh chưa cho biết lấy trung bình xác suất có tốt hơn hay không.

Hai nhánh có sai trên cùng những ảnh không? Bài 3 sẽ so sánh các nhóm lỗi và kiểm tra kết quả kết hợp.

<!-- ailaai-cell:38:markdown -->
## Đọc thêm và xem mã nguồn

[Bản đồ Reading và notebook](../READING_NOTEBOOK_MAP.md) chỉ từng nội dung của bài đọc đến các cell và tệp Python tương ứng. Bài đọc chính là `OlympicAI_2026/topic_ai_la_ai/reading.tex`, các phần II–IV và Phụ lục B–D trong workspace bài giảng.

Dòng `# ailaai-source` ở đầu một số cell chỉ nơi lưu hàm trong `src/ailaai/`. Khi thực hành, bạn có thể sửa ngay trong cell. Khi biên soạn lại tài liệu, sửa tệp Python rồi chạy `python scripts/build_notebooks.py --write` để đồng bộ.

Tài liệu của thư viện:

- PyTorch: [học chuyển giao](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html).
- scikit-learn: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Khi báo kết quả, ghi kèm cấu hình và số ảnh đã đánh giá. Các ví dụ CPU giúp kiểm tra cách tính; số liệu `reference` là kết quả đã lưu. Muốn đánh giá lượt huấn luyện của mình, dùng checkpoint, log và dự đoán do lượt đó tạo ra.
