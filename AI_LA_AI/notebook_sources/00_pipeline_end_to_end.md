<!-- ailaai-cell:00:markdown -->
# Pipeline: Từ dữ liệu đến tệp nộp bài

Notebook này gom các bước từ nạp dữ liệu đến tạo tệp nộp bài trong một phiên Google Colab:
1. Thiết lập môi trường và cấu hình hai nhánh RGB Native 358 và High-pass.
2. Nạp dữ liệu, kiểm tra và chia tập train/validation.
3. Huấn luyện hai mô hình ResNet34 hoặc nạp checkpoint đã lưu.
4. Lấy trung bình xác suất của hai nhánh theo tỷ lệ 50/50.
5. Dự đoán trên 200 ảnh Private Test, tạo và kiểm tra `submission.zip`.

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
RUN_ID = "student_e2e_v1"
ws = Workspace.from_root(TASK, run_id=RUN_ID)
verify_checkout(REPO)
check_environment(profile="e2e")
print(ws.summary())

<!-- ailaai-cell:03:markdown -->
## 1. Cấu hình huấn luyện và mô hình

Ở cell dưới, bạn có thể chọn chế độ cho từng nhánh trong `MODE`: `"train"` để huấn luyện hoặc `"load"` để nạp checkpoint đã lưu.

Các tham số `epochs`, `backbone_lr` và `head_lr` lần lượt quy định số epoch và tốc độ học (learning rate) của phần thân mạng (backbone) và tầng phân loại (head). Cấu hình mặc định dùng ResNet34 với trọng số ImageNet và vùng cắt trung tâm $358 \times 358$, giữ nguyên pixel gốc.

<!-- ailaai-cell:04:code -->
from dataclasses import replace
from ailaai.config import load_config, read_json
e2e_cfg = read_json(TASK / "configs/e2e.json")
MODE = e2e_cfg["default_modes"].copy()
MODEL_SPEC = {"backbone": "resnet34", "weights": "IMAGENET1K_V1"}
rgb_cfg = replace(load_config(TASK / "configs/rgb_native358.json"),
                  epochs=15, backbone_lr=1.5e-4, head_lr=7.5e-4,
                  model=MODEL_SPEC.copy())
hp_cfg = replace(load_config(TASK / "configs/highpass358.json"),
                 epochs=15, backbone_lr=1.5e-4, head_lr=7.5e-4,
                 model=MODEL_SPEC.copy())
assert set(MODE.values()) <= {"train", "load"}
check_environment(profile="train" if "train" in MODE.values() else "infer")
print(rgb_cfg)
print(hp_cfg)

<!-- ailaai-cell:05:markdown -->
## 2. Nạp và kiểm tra dữ liệu

Bộ dữ liệu có tại [who_is_AI (Google Drive)](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view?usp=drive_link). Phần nạp dữ liệu sử dụng thư mục `data/` và cấu hình nguồn tải của bài.

Tập train có 2.000 ảnh với nhãn $0 = \text{Real}$ và $1 = \text{Fake}$. Tập Private Test có 200 ảnh không kèm nhãn.

Trước khi huấn luyện, kiểm tra số ảnh, phân bố nhãn và mở thử vài ảnh để xem dữ liệu đã được đọc đúng chưa.

<!-- ailaai-cell:06:code -->
from ailaai.resources import prepare_resources
from ailaai.data import load_train_manifest, load_test_manifest, validate_data
from ailaai.visuals import show_samples

resources = TASK / "configs/resources.json"
prepare_resources(ws, resources, profile="e2e")
train = load_train_manifest(ws)
test = load_test_manifest(ws)
validate_data(train, test, resource_manifest=resources)
print("Train:", len(train), "Test:", len(test))
display(train.label.value_counts().rename(index={0: "Real", 1: "Fake"}))
display(test[["file_name"]].head())
show_samples(train, label_col="label", n=6)

<!-- ailaai-cell:07:markdown -->
## 3. Chia tập train và validation

Hai nhánh RGB và High-pass dùng cùng bảng chia fold ở `assets/splits/train_folds.csv` để được đánh giá trên cùng các ảnh validation.

Ảnh validation không được dùng để cập nhật trọng số. Ta dùng kết quả trên tập này để theo dõi quá trình học và so sánh các phương án.

<!-- ailaai-cell:08:code -->
from ailaai.data import load_fold_split
fit_rows, val_rows = load_fold_split(
    train, TASK / e2e_cfg["split_path"], fold=e2e_cfg["fold"])
assert set(fit_rows.file_name).isdisjoint(val_rows.file_name)
assert len(fit_rows) + len(val_rows) == len(train)
print("Train:", len(fit_rows), "Validation:", len(val_rows))
display(val_rows.label.value_counts().sort_index())

<!-- ailaai-cell:09:markdown -->
## 4. Hai cách xử lý ảnh: RGB Native và High-pass

Cả hai nhánh đều cắt vùng trung tâm $358 \times 358$:
- `native_view` giữ nguyên pixel của vùng cắt RGB.
- `highpass_view` lấy vùng cắt trừ đi ảnh làm mờ Gaussian, rồi điều chỉnh giá trị về dải $[0, 1]$.

Hai hàm được viết ngay trong notebook và truyền vào phần code huấn luyện dùng chung. Bạn có thể đọc hoặc sửa từng phép xử lý trước khi chạy.

<!-- ailaai-cell:10:code -->
import torch
from torchvision.transforms import functional as TF
from ailaai.data import decode_rgb
from ailaai.visuals import show_views
from ailaai.forensics import gaussian_blur_rgb
VIEW_SPEC = {"crop": rgb_cfg.view["crop"],
             **{k: hp_cfg.view[k] for k in ("kernel", "sigma", "gain", "offset")}}
RGB_VIEW_SPEC = {"crop": rgb_cfg.view["crop"]}
HP_VIEW_SPEC = dict(VIEW_SPEC)
def native_view(x):
    return TF.center_crop(x, [VIEW_SPEC["crop"]] * 2)
def highpass_view(x):
    x = native_view(x)
    low = gaussian_blur_rgb(x, kernel_size=VIEW_SPEC["kernel"], sigma=VIEW_SPEC["sigma"])
    return (VIEW_SPEC["gain"] * (x - low) + VIEW_SPEC["offset"]).clamp(0, 1)
x = decode_rgb(train.iloc[0].path)
for view in (native_view, highpass_view):
    z = view(x)
    assert z.shape == (3, 358, 358) and torch.isfinite(z).all()
    assert 0 <= z.min() and z.max() <= 1
show_views(x, {"RGB Native 358": native_view(x), "High-pass": highpass_view(x)})
rgb_cfg = replace(rgb_cfg, view={"name": "native", "crop": VIEW_SPEC["crop"]})
hp_cfg = replace(hp_cfg, view={"name": "highpass", **VIEW_SPEC})
print("Normalize:", rgb_cfg.normalize_mean, rgb_cfg.normalize_std)

<!-- ailaai-cell:11:markdown -->
## 5. Tạo mô hình

Hàm `model_factory` tạo ResNet34 và thay tầng phân loại cuối (`fc`) bằng tầng có hai đầu ra cho Real và Fake.

Các tham số được chia thành nhóm để AdamW dùng tốc độ học khác nhau: thấp hơn ở backbone đã có trọng số tiền huấn luyện và cao hơn ở tầng phân loại mới.

<!-- ailaai-cell:12:code -->
from torch import nn
from torchvision import models
from ailaai.models import attach_optimizer_groups
MODEL_BUILDERS = {
    "resnet34": (models.resnet34, models.ResNet34_Weights),
    "resnet18": (models.resnet18, models.ResNet18_Weights),
}
def model_factory(*, initialize):
    builder, weight_enum = MODEL_BUILDERS[MODEL_SPEC["backbone"]]
    weights = weight_enum[MODEL_SPEC["weights"]] if initialize else None
    model = builder(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    attach_optimizer_groups(model, head=model.fc)
    return model

<!-- ailaai-cell:13:markdown -->
## 6. Huấn luyện hoặc nạp nhánh RGB

Nếu chọn `"train"`, cell dưới huấn luyện nhánh RGB. AMP (Automatic Mixed Precision) kết hợp các mức độ chính xác số học khi chạy trên GPU để giảm nhu cầu bộ nhớ và hỗ trợ tốc độ tính toán. Nếu chọn `"load"`, cell nạp lượt chạy đã lưu.

Theo dõi loss Cross-Entropy và Macro-F1 trên validation sau mỗi epoch.

<!-- ailaai-cell:14:code -->
from ailaai.engine import fit_fold, load_run
from ailaai.visuals import show_learning_curves
rgb_kwargs = dict(branch="rgb", train_rows=fit_rows, val_rows=val_rows,
                  model_factory=model_factory, view_fn=native_view,
                  view_spec=RGB_VIEW_SPEC, model_spec=MODEL_SPEC)
if MODE["rgb"] == "train":
    rgb_run = fit_fold(ws, rgb_cfg, **rgb_kwargs)
else:
    rgb_run = load_run(ws, rgb_cfg, **rgb_kwargs,
                      checkpoint_source=e2e_cfg["checkpoint_sources"]["rgb"])
show_learning_curves(rgb_run.curves)
print(rgb_run.summary())

<!-- ailaai-cell:15:markdown -->
## 7. Huấn luyện hoặc nạp nhánh High-pass

Nhánh High-pass dùng cùng fold với RGB, nhưng đầu vào là phần chênh lệch giữa ảnh gốc và ảnh làm mờ Gaussian. Cell dưới huấn luyện hoặc nạp mô hình theo chế độ đã chọn.

Quan sát đường cong học của hai nhánh để xem loss và điểm validation thay đổi ra sao.

<!-- ailaai-cell:16:code -->
hp_kwargs = dict(branch="highpass", train_rows=fit_rows, val_rows=val_rows,
                 model_factory=model_factory, view_fn=highpass_view,
                 view_spec=HP_VIEW_SPEC, model_spec=MODEL_SPEC)
if MODE["highpass"] == "train":
    hp_run = fit_fold(ws, hp_cfg, **hp_kwargs)
else:
    hp_run = load_run(ws, hp_cfg, **hp_kwargs,
                     checkpoint_source=e2e_cfg["checkpoint_sources"]["highpass"])
show_learning_curves(hp_run.curves)
print(hp_run.summary())

<!-- ailaai-cell:17:markdown -->
## 8. Lấy trung bình xác suất hai nhánh

Với mỗi ảnh validation, ta tính:

$$p_{\text{Mean}} = 0.5 \cdot p_{\text{RGB}} + 0.5 \cdot p_{\text{HP}}$$

Dùng ngưỡng $t = 0.50$ để chuyển xác suất thành nhãn. So sánh Macro-F1 của từng nhánh với kết quả lấy trung bình để xem việc kết hợp có cải thiện trong lượt chạy này không.

<!-- ailaai-cell:18:code -->
import pandas as pd
from sklearn.metrics import f1_score
from ailaai.predictions import align_predictions
val = align_predictions(rgb_run.val_predictions, hp_run.val_predictions,
                        expected_rows=val_rows)
val["p_mean"] = 0.5 * val.p_rgb + 0.5 * val.p_hp
scores = pd.DataFrame([
    {"model": name, "macro_f1": f1_score(val.label, val[col] >= 0.5,
        labels=[0, 1], average="macro", zero_division=0)}
    for name, col in [("RGB", "p_rgb"), ("High-pass", "p_hp"), ("Mean 50/50", "p_mean")]
])
display(scores)

<!-- ailaai-cell:19:markdown -->
## 9. Chốt cấu hình và lưu thông tin lượt chạy

Trước khi dự đoán Private Test, ta lưu trọng số kết hợp, ngưỡng $t = 0.50$, đường dẫn checkpoint và điểm validation vào `decision.json`. Tệp này ghi lại các lựa chọn đã dùng để tiện xem lại hoặc chạy lại.

<!-- ailaai-cell:20:code -->
from ailaai.pipeline import save_decision
decision = save_decision(
    ws, runs={"rgb": rgb_run, "highpass": hp_run},
    method="probability_mean", weights={"rgb": 0.5, "highpass": 0.5},
    threshold=0.5, label_map={0: "Real", 1: "Fake"},
    test_rows=test, validation_scores=scores)
print(decision.summary())

<!-- ailaai-cell:21:markdown -->
## 10. Dự đoán trên Private Test

Hai mô hình lần lượt dự đoán trên 200 ảnh test. Xác suất của từng nhánh được lưu vào CSV, rồi lấy trung bình 50/50 để tạo nhãn cuối cùng: `0` cho Real và `1` cho Fake.

<!-- ailaai-cell:22:code -->
from ailaai.engine import predict
rgb_test = predict(rgb_run, test, model_factory=model_factory,
                   view_fn=native_view, view_spec=RGB_VIEW_SPEC,
                   model_spec=MODEL_SPEC, decision=decision)
rgb_test.save(ws.output_root / "rgb_test.csv")

<!-- ailaai-cell:23:code -->
hp_test = predict(hp_run, test, model_factory=model_factory,
                  view_fn=highpass_view, view_spec=HP_VIEW_SPEC,
                  model_spec=MODEL_SPEC, decision=decision)
hp_test.save(ws.output_root / "highpass_test.csv")

<!-- ailaai-cell:24:code -->
assert decision.weights == {"rgb": 0.5, "highpass": 0.5}
pred = align_predictions(rgb_test, hp_test, expected_rows=test)
pred["p_mean"] = 0.5 * pred.p_rgb + 0.5 * pred.p_hp
submission = pd.DataFrame({
    "file_name": pred.file_name,
    "category_id": (pred.p_mean >= decision.threshold).astype("int64")})
pred.to_csv(ws.output_root / "mean_test.csv", index=False)
display(submission.head())
print("Số dòng:", len(submission))

<!-- ailaai-cell:25:markdown -->
## 11. Tạo và kiểm tra `submission.zip`

Hàm `export_submission` tạo tệp nộp bài. Hàm `validate_submission` kiểm tra:
- CSV nằm ở gốc tệp ZIP.
- Có đúng 200 dòng, tương ứng với danh sách ảnh test.
- Có hai cột `file_name,category_id`.
- Không thiếu giá trị hoặc trùng tên ảnh; nhãn là số nguyên 0 hoặc 1.

Nếu chạy trên Google Colab, bước cuối gửi tệp `submission.zip` về máy của bạn.

<!-- ailaai-cell:26:code -->
from ailaai.submission import export_submission
zip_path = export_submission(
    submission, ws.output_root / "submission.zip",
    expected_names=test.file_name.tolist(), expected_count=200)
print(zip_path)

<!-- ailaai-cell:27:code -->
from ailaai.submission import validate_submission
receipt = validate_submission(
    zip_path, expected_names=test.file_name.tolist(), expected_count=200,
    expected_frame=submission, receipt_path=ws.output_root / "submission_check.json")
print(receipt.summary())
if "google.colab" in sys.modules:
    from google.colab import files
    files.download(str(zip_path))
