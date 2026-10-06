<!-- ailaai-cell:00:markdown -->
# Bài 4: Chạy thử nghiệm và đọc kết quả ablation

Bài này chạy từ ảnh gốc đến bảng so sánh: RGB, biểu diễn Wavelet Haar và RGB có trọng số cho nhóm Fake ít biên. Sau đó ta thử chọn ngưỡng riêng cho ảnh xám/màu.

**Trên Colab:** chọn **Runtime → Change runtime type → T4 GPU**, rồi bấm **Run all**. Mặc định mỗi nhánh học 15 epoch trên fold 0. Thời gian phụ thuộc GPU và tốc độ tải dữ liệu; cell huấn luyện in tiến độ sau mỗi epoch.

Các thử nghiệm dưới dùng một công thức giảng dạy mới, cùng ResNet18 và cùng cách chia dữ liệu. Điểm số có thể khác những lần thử trong bài đọc. Ta cần xem kết quả vừa chạy trước khi quyết định có giữ một hướng hay không.

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

<!-- ailaai-cell:02:markdown -->
## 1. Chọn cách chạy

Giữ `MODE = "train"` để tải ảnh và huấn luyện thật. Đổi thành `"reference"` nếu chỉ muốn đọc các dự đoán lịch sử trên CPU.

`FOLDS = [0]` chạy một fold với 1.600 ảnh train và 400 ảnh validation. Muốn tạo dự đoán OOF cho đủ 2.000 ảnh, đổi thành `[0, 1, 2, 3, 4]`. Khi đó notebook huấn luyện 15 mô hình: ba nhánh cho mỗi fold.

Checkpoint được lưu sau từng epoch. Chạy lại trong cùng runtime sẽ tiếp tục phần còn thiếu hoặc dùng lại lượt đã hoàn tất. Nếu xóa runtime Colab, cần lưu thư mục `artifacts/` ra ngoài trước.

<!-- ailaai-cell:03:code -->
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display
from ailaai.config import Workspace, load_config, config_with
from ailaai.data import load_train_manifest, load_fold_split, decode_rgb
from ailaai.resources import prepare_resources
from ailaai.engine import fit_fold
from ailaai.models import model_factory as build_model
from ailaai.transforms import native_view
from ailaai.ablation import (haar_view, image_features, low_edge_weights,
                            paired_report, calibration_split, choose_group_thresholds)
from ailaai.metrics import classification_report, macro_f1

MODE = "train"
FOLDS = [0]
EPOCHS = 15
RUN_ID = "lesson_nb4_e2e_v1"
if MODE not in {"train", "reference"}:
    raise ValueError('MODE chỉ nhận "train" hoặc "reference".')
if not FOLDS or len(set(FOLDS)) != len(FOLDS) or not set(FOLDS) <= set(range(5)):
    raise ValueError("FOLDS cần các fold khác nhau trong 0..4.")
if MODE == "train" and not torch.cuda.is_available():
    raise RuntimeError('Chọn T4 GPU rồi chạy lại, hoặc đổi MODE thành "reference" để đọc kết quả trên CPU.')
ws = Workspace.from_root(TASK, run_id=RUN_ID)
cfg = config_with(load_config(TASK / "configs/rgb_native358.json"), epochs=EPOCHS,
                  model={"backbone": "resnet18", "weights": "IMAGENET1K_V1"})
MODEL_SPEC = dict(cfg.model)
print("Chế độ:", MODE, "| folds:", FOLDS, "| epoch mỗi nhánh:", EPOCHS)
print("Kết quả:", ws.output_root)

<!-- ailaai-cell:04:markdown -->
## 2. Tải dữ liệu và kiểm tra cách chia

Bộ ảnh có tại [who_is_AI (Google Drive)](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view?usp=drive_link). Cell dưới tự tải, giải nén và tìm thư mục ảnh. Bảng chia fold cố định giúp các nhánh được so sánh trên cùng ảnh.

Ở chế độ `reference`, notebook dùng CSV đã đóng gói trong repo và không tải ảnh gốc.

<!-- ailaai-cell:05:code -->
if MODE == "train":
    prepare_resources(ws, TASK / "configs/resources.json", profile="train")
    train = load_train_manifest(ws)
    print("Ảnh train:", len(train))
    display(train.label.value_counts().sort_index().rename(index={0: "Real", 1: "Fake"}))
    for fold in FOLDS:
        fit_rows, val_rows = load_fold_split(train, TASK / "assets/splits/train_folds.csv", fold)
        print(f"Fold {fold}: train {len(fit_rows)}, validation {len(val_rows)}")
else:
    reference_dir = TASK / "data/negative_results"
    results = pd.read_csv(reference_dir / "oof_predictions.csv")
    results = results[results.fold.isin(FOLDS)].copy()
    cutoffs = {int(item["fold"]): float(item["cutoff"])
               for item in json.loads((reference_dir / "promotion_gates.json").read_text())["cutoffs"]}
    print("Đang đọc dự đoán lịch sử:", len(results), "ảnh")

<!-- ailaai-cell:06:markdown -->
## 3. RGB và Wavelet nhìn ảnh như thế nào?

Ta cắt vùng giữa ảnh thành 358 × 358, tương ứng khoảng 70% chiều dài mỗi cạnh của ảnh 512 × 512.

Nhánh RGB giữ vùng cắt này. Nhánh Haar phân rã thành bốn dải LL, LH, HL, HH rồi xếp chúng vào bốn góc của từng kênh màu. Các dải chi tiết có thể âm nên được đưa về quanh 0,5; đầu ra vẫn có ba kênh và kích thước 358 × 358.

Cả hai nhánh dùng ResNet18, trọng số ImageNet, cùng seed và lịch học. Như vậy ta không đổi backbone khi thử biểu diễn ảnh. Cách xếp bốn dải ở đây được viết rõ trong `haar_view`; đây là lựa chọn cho bài thực hành, chưa phải công thức tốt nhất.

<!-- ailaai-cell:07:code -->
def student_model_factory(*, initialize):
    return build_model(MODEL_SPEC["backbone"], MODEL_SPEC["weights"], initialize=initialize)

def rgb_view(batch):
    return native_view(batch, crop=358)

def wavelet_view(batch):
    return haar_view(batch, crop=358)

RGB_SPEC = {"name": "native", "crop": 358}
WAVELET_SPEC = {"name": "haar_quadrants", "crop": 358, "version": 1}
if MODE == "train":
    sample = decode_rgb(train.iloc[0].path)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, title, view in zip(axes, ["RGB", "Haar: LL / LH / HL / HH"],
                               [rgb_view(sample), wavelet_view(sample)]):
        ax.imshow(view.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ws.output_root / "input_views.png", dpi=150)
    plt.show()

<!-- ailaai-cell:08:markdown -->
## 4. Chọn nhóm Fake ít biên từ tập train

Ở bài thực hành này, `edge_ratio` là tỷ lệ pixel có độ lớn gradient Sobel vượt 0,08 trên vùng cắt. Ảnh xám được nhận diện qua mức chênh lệch giữa ba kênh màu. Ta tính cả hai từ ảnh đang dùng, không ghép đặc trưng lịch sử vào lượt chạy mới.

Mỗi fold chọn ngưỡng phân vị 25% từ **ảnh Fake của phần train**. Ảnh Fake thấp hơn ngưỡng nhận trọng số 1,5; các ảnh Fake còn lại được giảm trọng số để tổng trọng số lớp Fake giữ nguyên. Ảnh Real vẫn có trọng số 1. Nếu nhiều ảnh có cùng giá trị tại ngưỡng, nhóm được tăng trọng số có thể ít hơn 25%.

Đây là trọng số cho loss của từng ảnh, không phải loss theo pixel. Ngưỡng được chọn trước khi xem kết quả validation.

<!-- ailaai-cell:09:code -->
if MODE == "train":
    features = image_features(train, crop=358, edge_threshold=0.08)
    train = train.merge(features, on="file_name", validate="one_to_one")
    display(features.head())
    print("Ảnh xám:", int(features.is_gray.sum()))
    features.to_csv(ws.output_root / "image_features.csv", index=False)

<!-- ailaai-cell:10:markdown -->
## 5. Huấn luyện ba nhánh

Ta chạy RGB, Haar và RGB có trọng số lần lượt để không giữ nhiều mô hình trên GPU cùng lúc. Validation chỉ dùng để theo dõi; mỗi nhánh lấy checkpoint ở epoch cuối đã định trước, không chọn epoch theo điểm cao nhất.

Learning rate cho backbone là 1,5 × 10⁻⁴, cho head là 7,5 × 10⁻⁴; weight decay là 10⁻⁴. Các thiết lập còn lại nằm trong cấu hình dùng chung. Nếu thay số epoch hoặc công thức, bộ huấn luyện tạo thư mục riêng để tránh nạp nhầm checkpoint.

<!-- ailaai-cell:11:code -->
runs = {}
if MODE == "train":
    frames, cutoffs = [], {}
    for fold in FOLDS:
        fit_rows, val_rows = load_fold_split(train, TASK / "assets/splits/train_folds.csv", fold)
        weights, cutoff = low_edge_weights(fit_rows, multiplier=1.5)
        cutoffs[fold] = cutoff
        fold_ws = Workspace.from_root(TASK, run_id=f"{RUN_ID}_fold{fold}")
        pd.DataFrame({"file_name": list(weights), "weight": list(weights.values())}).to_csv(
            fold_ws.output_root / "train_sample_weights.csv", index=False)
        frame = val_rows[["file_name", "label", "fold", "is_gray", "edge_ratio"]].copy()
        for branch, view_fn, view_spec, sample_weights, column in [
            ("rgb", rgb_view, RGB_SPEC, None, "rgb_prob"),
            ("wavelet", wavelet_view, WAVELET_SPEC, None, "wavelet_prob"),
            ("edge_weighted", rgb_view, RGB_SPEC, weights, "edge_weighted_prob"),
        ]:
            branch_cfg = config_with(cfg, view=view_spec)
            run = fit_fold(fold_ws, branch_cfg, branch, fit_rows, val_rows,
                           student_model_factory, view_fn, view_spec, MODEL_SPEC,
                           sample_weights=sample_weights)
            runs[(fold, branch)] = run
            probability = run.val_predictions.rows[["file_name", "prob"]].rename(columns={"prob": column})
            frame = frame.merge(probability, on="file_name", validate="one_to_one")
            print(run.summary())
        # Cùng RGB baseline cho so sánh trọng số; không thêm một mô hình thứ tư.
        frame["edge_base_prob"] = frame["rgb_prob"]
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    print("Đã huấn luyện xong:", len(runs), "mô hình")
else:
    print("Chế độ reference: bỏ qua huấn luyện.")

if not results.file_name.is_unique or results.empty:
    raise ValueError("Các dự đoán cần có tên ảnh duy nhất và không được rỗng.")
for col in ["rgb_prob", "wavelet_prob", "edge_base_prob", "edge_weighted_prob"]:
    if not np.isfinite(results[col]).all() or not results[col].between(0, 1).all():
        raise ValueError(f"Xác suất không hợp lệ: {col}")
print("Dự đoán validation:", len(results), "| folds:", sorted(results.fold.unique()))

<!-- ailaai-cell:12:markdown -->
## 6. Đọc điểm tổng và số lỗi

Real là 0, Fake là 1. FN là ảnh Fake bị đoán thành Real; FP là ảnh Real bị đoán thành Fake. Ta dùng ngưỡng 0,5 cho bảng so sánh đầu tiên.

Một fold chỉ cho kết quả trên 400 ảnh validation. Nếu chạy đủ năm fold, bảng tổng hợp có 2.000 dự đoán OOF: mỗi ảnh được dự đoán bởi mô hình không học ảnh đó.

<!-- ailaai-cell:13:code -->
metric_rows = []
for branch, column in [("RGB", "rgb_prob"), ("Haar", "wavelet_prob"),
                       ("RGB có trọng số", "edge_weighted_prob")]:
    report = classification_report(results.label, results[column])
    tn, fp, fn, tp = np.asarray(report["confusion_matrix"]).ravel()
    metric_rows.append({"Nhánh": branch, "n": len(results), "Macro-F1": report["macro_f1"],
                        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
metrics = pd.DataFrame(metric_rows)
display(metrics)
if MODE == "reference":
    print("RGB/Wavelet lịch sử thay cả backbone; không suy ra riêng ảnh hưởng của biểu diễn.")

<!-- ailaai-cell:14:markdown -->
## 7. Wavelet: sửa lỗi hay tạo thêm lỗi?

**Fixes** là những ảnh RGB đoán sai nhưng Haar đoán đúng. **Breaks** là những ảnh RGB đoán đúng nhưng Haar lại sai. Hiệu `breaks − fixes` cho biết số lỗi tăng ròng.

Đọc bảng vừa tính trước khi kết luận. Nếu thử nghiệm này tốt hơn RGB, ta ghi nhận kết quả đó; không ép lượt chạy mới phải giống một thử nghiệm thất bại trong bài đọc.

<!-- ailaai-cell:15:code -->
wavelet_report = paired_report(results, "rgb_prob", "wavelet_prob")
display(pd.DataFrame([wavelet_report]))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].bar(metrics["Nhánh"], metrics["Macro-F1"])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Macro-F1")
axes[0].tick_params(axis="x", rotation=15)
axes[1].bar(["Fixes", "Breaks"], [wavelet_report["fixes"], wavelet_report["breaks"]])
axes[1].set_ylabel("Số ảnh")
fig.tight_layout()
fig.savefig(ws.output_root / f"{MODE}_wavelet_comparison.png", dpi=150)
plt.show()

<!-- ailaai-cell:16:markdown -->
## 8. Trọng số mẫu: nhóm mục tiêu có tốt hơn không?

Điểm tổng có thể tăng trong khi nhóm Fake ít biên vẫn sai thêm. Ta dùng ngưỡng đã chọn từ train của từng fold để xem riêng nhóm này và nhóm Real ít biên.

Nếu nhóm mục tiêu quá nhỏ hoặc không có ảnh, bảng ghi số ảnh để ta biết giới hạn của phép so sánh. Cải thiện vài ảnh trên một fold chưa đủ để kết luận hướng này ổn định.

<!-- ailaai-cell:17:code -->
edge_report = paired_report(results, "edge_base_prob", "edge_weighted_prob")
display(pd.DataFrame([edge_report]))
subgroup_rows = []
for fold, frame in results.groupby("fold"):
    for label, name in [(1, "Fake ít biên"), (0, "Real ít biên")]:
        group = frame[(frame.label == label) & (frame.edge_ratio < cutoffs[int(fold)])]
        baseline_errors = int(((group.edge_base_prob >= 0.5).astype(int) != group.label).sum())
        weighted_errors = int(((group.edge_weighted_prob >= 0.5).astype(int) != group.label).sum())
        subgroup_rows.append({"fold": int(fold), "Nhóm": name, "n": len(group),
                              "RGB errors": baseline_errors, "Weighted errors": weighted_errors,
                              "cutoff từ train": cutoffs[int(fold)]})
subgroups = pd.DataFrame(subgroup_rows)
display(subgroups)
fig, ax = plt.subplots(figsize=(8, 3.5))
subgroups.groupby("Nhóm")[["RGB errors", "Weighted errors"]].sum().plot.bar(ax=ax, rot=0)
ax.set_ylabel("Số ảnh sai")
fig.tight_layout()
fig.savefig(ws.output_root / f"{MODE}_edge_subgroups.png", dpi=150)
plt.show()

<!-- ailaai-cell:18:markdown -->
## 9. Chọn ngưỡng trên calibration, chấm trên evaluation

Trong mỗi fold validation, ta tách khoảng một nửa làm calibration, phần còn lại làm evaluation; cách chia giữ các nhóm nhãn và xám/màu. Mô hình không học cả hai phần này.

Ta thử ngưỡng 0,30 đến 0,70 trên calibration. Nhóm thiếu một trong hai lớp giữ ngưỡng 0,5. Sau khi chọn xong, ta so ngưỡng mới với 0,5 **trên cùng phần evaluation**. Không dò lại ngưỡng bằng nhãn evaluation.

Bài thực hành dùng xác suất RGB cho bước này. Phần ngưỡng trong bài đọc dùng một mô hình ghép khác, nên không lấy điểm ở đây làm điểm tái tạo của mô hình đó.

<!-- ailaai-cell:19:code -->
threshold_rows, evaluation_frames = [], []
for fold, frame in results.groupby("fold"):
    calibration, evaluation = calibration_split(frame, seed=cfg.seed + int(fold))
    chosen = choose_group_thresholds(calibration, probability="rgb_prob")
    evaluation["chosen_threshold"] = evaluation.is_gray.map(chosen)
    evaluation["prediction_default"] = (evaluation.rgb_prob >= 0.5).astype(int)
    evaluation["prediction_calibrated"] = (evaluation.rgb_prob >= evaluation.chosen_threshold).astype(int)
    evaluation_frames.append(evaluation)
    threshold_rows.append({"fold": int(fold), "calibration_n": len(calibration),
                           "evaluation_n": len(evaluation), "gray_threshold": chosen[1],
                           "color_threshold": chosen[0]})
thresholds = pd.DataFrame(threshold_rows)
evaluation = pd.concat(evaluation_frames, ignore_index=True)
display(thresholds)
threshold_report = pd.DataFrame([
    {"Ngưỡng": "0.5", "n": len(evaluation),
     "Macro-F1": macro_f1(evaluation.label, evaluation.prediction_default)},
    {"Ngưỡng": "Chọn trên calibration", "n": len(evaluation),
     "Macro-F1": macro_f1(evaluation.label, evaluation.prediction_calibrated)},
])
display(threshold_report)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(threshold_report["Ngưỡng"], threshold_report["Macro-F1"])
ax.set_ylim(0, 1)
ax.set_ylabel("Macro-F1 trên evaluation")
fig.tight_layout()
fig.savefig(ws.output_root / f"{MODE}_threshold_evaluation.png", dpi=150)
plt.show()

<!-- ailaai-cell:20:markdown -->
## 10. Lưu kết quả và quyết định bước tiếp theo

Cell cuối lưu dự đoán, bảng điểm, ngưỡng và danh sách checkpoint của lượt chạy. Các hình phía trên cũng được lưu dưới dạng PNG. Đây là báo cáo ablation trên ảnh có nhãn, không phải điểm Private Test.

Khi đọc kết quả, hãy nêu rõ số fold, số ảnh và số epoch. Với Wavelet, xem fixes/breaks; với trọng số, xem nhóm mục tiêu; với ngưỡng, xem phần evaluation. Nếu muốn thử tiếp, đổi một yếu tố rồi chạy lại cùng cách chia dữ liệu.

<!-- ailaai-cell:21:code -->
output = ws.output_root / MODE
output.mkdir(parents=True, exist_ok=True)
for name, table in [("validation_predictions", results), ("metrics", metrics),
                    ("edge_subgroups", subgroups), ("threshold_choices", thresholds),
                    ("threshold_evaluation", evaluation), ("threshold_metrics", threshold_report)]:
    table.to_csv(output / f"{name}.csv", index=False)
receipt = {"mode": MODE, "folds": FOLDS, "epochs": EPOCHS if MODE == "train" else None,
           "config": cfg.to_dict() if MODE == "train" else None,
           "validation_rows": len(results), "threshold_evaluation_rows": len(evaluation),
           "wavelet": wavelet_report, "edge_weighting": edge_report,
           "cutoffs_from_training": cutoffs,
           "checkpoints": [str(run.checkpoint_path) for run in runs.values()]}
(output / "run_summary.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
print("Đã lưu báo cáo:", output)
print("Đã lưu hình:", ws.output_root)
print("Chạy xong bài 4.")
