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
RUN_ID = "student_e2e_reading_v2"  # cùng RUN_ID với notebook 00
ws = Workspace.from_root(TASK, run_id=RUN_ID)
verify_checkout(REPO)
check_environment(profile="replay")
SOURCE = "reference"  # đổi thành "learner" để đọc run của bạn
print(ws.summary())

<!-- ailaai-cell:03:code -->
from pathlib import Path
from typing import Any, Iterable
from dataclasses import dataclass
import hashlib, os, tempfile, zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ailaai.config import write_json
from ailaai.resources import sha256_file
from ailaai.predictions import PredictionTable
from ailaai.metrics import classification_report

<!-- ailaai-cell:04:markdown -->
### Ghép dự đoán theo tên ảnh

Hai bảng có thể sắp xếp khác nhau. Ta ghép one-to-one theo tên ảnh, đồng thời kiểm tra nhãn, fold và độ phủ dữ liệu trước khi cộng xác suất.

<!-- ailaai-cell:05:code -->
# ailaai-source: src/ailaai/predictions.py::align_predictions
def align_predictions(
    rgb: PredictionTable | pd.DataFrame,
    highpass: PredictionTable | pd.DataFrame,
    expected_rows: pd.DataFrame | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Join by unique file IDs and require exact coverage, labels, folds, and split."""
    rgb_table = rgb if isinstance(rgb, PredictionTable) else PredictionTable(rgb)
    hp_table = highpass if isinstance(highpass, PredictionTable) else PredictionTable(highpass)
    rgb_meta, hp_meta = rgb_table.meta, hp_table.meta
    rgb_split, hp_split = rgb_meta.get("split_sha256"), hp_meta.get("split_sha256")
    if rgb_split and hp_split and rgb_split != hp_split:
        raise ValueError("RGB and High-pass predictions come from different train splits.")
    left = rgb_table.rows.copy()
    right = hp_table.rows.copy()
    shared = sorted(set(left.columns) & set(right.columns) - {"file_name", "prob"})
    left = left.rename(columns={"prob": "p_rgb"})
    right = right.rename(columns={"prob": "p_hp"})
    merged = left.merge(right[["file_name", "p_hp", *shared]], on="file_name", how="outer",
                        validate="one_to_one", indicator=True, suffixes=("", "_hp"))
    if not (merged._merge == "both").all():
        missing = merged.loc[merged._merge != "both", "file_name"].head(5).tolist()
        raise ValueError(f"Prediction file_name coverage differs between branches: {missing}")
    merged = merged.drop(columns="_merge")
    for column in shared:
        other = f"{column}_hp"
        if other in merged:
            if not merged[column].equals(merged[other]):
                raise ValueError(f"RGB and High-pass prediction metadata differ in {column}.")
            merged = merged.drop(columns=other)
    if expected_rows is not None:
        if isinstance(expected_rows, pd.DataFrame):
            expected = expected_rows.copy()
            if "file_name" not in expected:
                raise ValueError("expected_rows must contain file_name.")
        else:
            expected = pd.DataFrame({"file_name": list(expected_rows)})
        expected["file_name"] = expected.file_name.astype(str)
        if not expected.file_name.is_unique:
            raise ValueError("Expected file names are not unique.")
        if set(merged.file_name) != set(expected.file_name):
            missing = sorted(set(expected.file_name) - set(merged.file_name))[:5]
            extra = sorted(set(merged.file_name) - set(expected.file_name))[:5]
            raise ValueError(f"Prediction coverage differs from expected rows; missing={missing}, extra={extra}.")
        if "label" in expected:
            if "label" not in merged:
                raise ValueError("Expected validation rows have labels but predictions do not.")
            labels = expected.set_index("file_name").label
            actual = merged.set_index("file_name").label
            if not labels.sort_index().equals(actual.reindex(labels.index).sort_index()):
                raise ValueError("Prediction labels do not match the requested validation rows.")
        if "fold" in expected and "fold" in merged:
            folds = expected.set_index("file_name").fold
            actual_folds = merged.set_index("file_name").fold
            if not folds.sort_index().equals(actual_folds.reindex(folds.index).sort_index()):
                raise ValueError("Prediction folds do not match the requested validation rows.")
        if "path" in expected:
            merged = merged.merge(expected[["file_name", "path"]], on="file_name", how="left", validate="one_to_one")
    return merged.sort_values("file_name").reset_index(drop=True)

<!-- ailaai-cell:06:markdown -->
## 1. Ghép dự đoán của hai nhánh

Dự đoán ngoại mẫu theo fold (out-of-fold, OOF) được tạo cho ảnh thuộc fold không dùng để huấn luyện mô hình tương ứng. Trước khi so sánh, ghép dự đoán RGB và High-pass theo cùng tên ảnh, nhãn và fold.

Mỗi nhánh xuất xác suất cho lớp Fake, ký hiệu $P(\text{Fake})$. Đặt `SOURCE = "reference"` để đọc kết quả mẫu đã lưu hoặc `"learner"` để phân tích kết quả từ các lượt huấn luyện của bạn.

<!-- ailaai-cell:07:code -->
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from ailaai.resources import prepare_resources
from ailaai.predictions import load_validation_pair
from ailaai.visuals import show_prediction_examples
prepare_resources(ws, TASK / "configs/resources.json", profile="train")
rgb, hp, expected_val = load_validation_pair(ws, source=SOURCE)
m = align_predictions(rgb, hp, expected_rows=expected_val)
print(rgb.summary())
print(hp.summary())
print("Số ảnh đang phân tích:", len(m))

<!-- ailaai-cell:08:markdown -->
## 2. Các nhóm đúng và sai của hai mô hình

Ta chia các ảnh thành bốn nhóm:
1. Cả RGB và High-pass cùng đúng.
2. Chỉ RGB đúng.
3. Chỉ High-pass đúng.
4. Cả hai cùng sai.

Các nhóm này cho biết lỗi của hai nhánh trùng nhau đến đâu. Những ảnh chỉ một nhánh đúng là nơi có thể thử cải thiện bằng cách kết hợp, nhưng kết quả còn phụ thuộc vào xác suất của từng nhánh.

<!-- ailaai-cell:09:code -->
rgb_ok = (m.p_rgb >= 0.5) == m.label
hp_ok = (m.p_hp >= 0.5) == m.label
overlap = pd.Series({
    "Cả hai đúng": int((rgb_ok & hp_ok).sum()),
    "Chỉ RGB đúng": int((rgb_ok & ~hp_ok).sum()),
    "Chỉ High-pass đúng": int((~rgb_ok & hp_ok).sum()),
    "Cả hai sai": int((~rgb_ok & ~hp_ok).sum())})
assert overlap.sum() == len(m)
display(overlap.to_frame("Số ảnh"))

<!-- ailaai-cell:10:markdown -->
## 3. Lấy trung bình xác suất 50/50

Ta tính:

$$p_{\text{Mean}} = 0.5 \cdot p_{\text{RGB}} + 0.5 \cdot p_{\text{HP}}$$

Ví dụ, một ảnh Fake có $p_{\text{RGB}} = 0.48$ và $p_{\text{HP}} = 0.90$. Trung bình là $0.69$, vượt ngưỡng $0.50$ và sửa được quyết định sai của RGB. Ví dụ này minh họa cách kết hợp xác suất, chưa cho biết mô hình đã học dấu vết nào.

Trong kết quả OOF mẫu, cả RGB và High-pass đều có FP = 28. So với RGB, Mean sửa 26 ca (13 Real và 13 Fake), nhưng làm sai thêm 5 ca, đều là Fake. Tỷ lệ sửa/sai thêm là 5,2 : 1; tổng lỗi giảm từ 92 xuống 71.

Xem `fix_mask` và `break_mask` trong bảng dưới để đối chiếu những ca được sửa với những ca sai thêm.

<!-- ailaai-cell:11:code -->
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

<!-- ailaai-cell:12:markdown -->
## 4. Xem ảnh được sửa và ảnh sai thêm

Ta xem hai nhóm ảnh:
- RGB sai nhưng Mean đúng (`fixes`).
- RGB đúng nhưng Mean sai (`breaks`).

Đối chiếu ảnh và xác suất của hai nhánh để xem quyết định thay đổi thế nào. Quan sát ảnh có thể gợi ý câu hỏi để thử tiếp, nhưng chưa đủ để xác định nguyên nhân mô hình dự đoán sai.

<!-- ailaai-cell:13:code -->
show_prediction_examples(m.loc[fix_mask].head(3), title="RGB sai → mean đúng")
show_prediction_examples(m.loc[break_mask].head(3), title="RGB đúng → mean sai")
plt.show()

<!-- ailaai-cell:14:markdown -->
### Lỗi còn lại: sai chung, bỏ lỡ cơ hội và làm hỏng

Ba nhóm dưới phân rã toàn bộ lỗi Mean. Với hai nhánh cùng sai về một phía 0,5, tổ hợp lồi của hai xác suất vẫn ở phía sai tại ngưỡng đó. Quan sát ảnh gợi ý giả thuyết; không xác nhận nguyên nhân sinh ảnh.

<!-- ailaai-cell:15:code -->
both_wrong = ~rgb_ok & ~hp_ok
missed = ~rgb_ok & hp_ok & ~mean_ok
broken = rgb_ok & ~mean_ok
introduced = rgb_ok & hp_ok & ~mean_ok
assert int(introduced.sum()) == 0
print("Cùng sai:", int(both_wrong.sum()), "| HP đúng nhưng Mean chưa sửa RGB:", int(missed.sum()), "| Mean làm hỏng RGB:", int(broken.sum()))
print("Mean sai:", int((~mean_ok).sum()), "=", int(both_wrong.sum()), "+", int(missed.sum()), "+", int(broken.sum()))
assert int((~mean_ok).sum()) == int(both_wrong.sum() + missed.sum() + broken.sum())
display(m.loc[~mean_ok, ["file_name", "label", "p_rgb", "p_hp", "p_mean"]].head(15))
show_prediction_examples(m.loc[both_wrong].head(3), title="Cả RGB và High-pass cùng sai")
from PIL import Image
if "path" in m.columns:
    gray_flags = []
    for row in m.itertuples():
        with Image.open(row.path) as image:
            image = image.convert("RGB"); image.thumbnail((64, 64))
            array = np.asarray(image).astype(float)
        gray_flags.append(bool((array.max(2) - array.min(2)).mean() < .5))
    m["is_gray"] = gray_flags
    display(pd.DataFrame([{"is_gray": flag, "n": len(g), **classification_report(g.label, g.p_mean)}
                          for flag, g in m.groupby("is_gray")]))

<!-- ailaai-cell:16:markdown -->
## 5. Khảo sát ngưỡng quyết định

Khi quét ngưỡng từ 0.1 đến 0.9 trên validation, bạn có thể thấy một ngưỡng như 0.52 hoặc 0.48 cho F1 cao hơn mốc 0.50.

Nếu dùng cùng dữ liệu để chọn ngưỡng và báo điểm, kết quả có thể lạc quan hơn khi gặp dữ liệu mới (optimism bias). Muốn quyết định đổi ngưỡng, cần đánh giá lựa chọn đó trên dữ liệu không dùng để chọn ngưỡng.

Bài này giữ ngưỡng mặc định $t = 0.50$. Đây là lựa chọn cho quy trình đang dùng, không phải ngưỡng tốt nhất cho mọi bài toán.

<!-- ailaai-cell:17:code -->
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=["#2563A6", "#C17817", "#7A5BA7"])})
thresholds = np.linspace(0.1, 0.9, 161)
scores = np.array([f1_score(m.label, m.p_mean >= t, labels=[0, 1],
                          average="macro", zero_division=0) for t in thresholds])
i = int(scores.argmax())
print("Ngưỡng tốt nhất trên tập đang khảo sát:", thresholds[i])
print("Điểm dùng cả để chọn và đo ngưỡng:", scores[i])
plt.plot(thresholds, scores)
plt.axvline(0.5, color="gray", linestyle="--", label="Mặc định 0.5")
plt.xlabel("Ngưỡng"); plt.ylabel("Macro-F1 trên tập đang khảo sát"); plt.legend(); plt.show()

<!-- ailaai-cell:18:markdown -->
### Chọn ngưỡng trên các fold khác

Với đủ năm fold, mỗi lượt chọn ngưỡng từ bốn fold rồi áp dụng sang fold còn lại. Như vậy nhãn của fold đang chấm không tham gia chọn ngưỡng cho chính nó.

Code cần ít nhất hai fold. Đây là cross-fit ở bước chọn ngưỡng trên ma trận OOF đã có; các mô hình base không được huấn luyện lại như trong nested CV.

<!-- ailaai-cell:19:code -->
# ailaai-source: src/ailaai/teaching.py::crossfit_threshold
def crossfit_threshold(frame, probability="p_mean", grid=None):
    """Fit a threshold on other held-out folds, then score only the excluded fold."""
    from sklearn.metrics import f1_score
    if frame.fold.nunique() < 2 or not frame.file_name.is_unique:
        raise ValueError("Cross-fit needs at least two folds and unique file names.")
    grid = np.asarray(grid if grid is not None else np.linspace(.1, .9, 161))
    parts, choices = [], []
    for fold in sorted(frame.fold.unique()):
        calibration = frame[frame.fold != fold]
        evaluation = frame[frame.fold == fold].copy()
        scores = [f1_score(calibration.label, calibration[probability] >= t,
                           labels=[0, 1], average="macro", zero_division=0) for t in grid]
        # Deterministic tie break: prefer the threshold closest to 0.5.
        best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0, atol=1e-12))
        threshold = float(grid[best[np.argmin(abs(grid[best] - .5))]])
        evaluation["chosen_threshold"] = threshold
        evaluation["crossfit_prediction"] = (evaluation[probability] >= threshold).astype(int)
        parts.append(evaluation)
        choices.append({"fold": int(fold), "threshold": threshold,
                        "calibration_n": len(calibration), "evaluation_n": len(evaluation)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(choices)

<!-- ailaai-cell:20:code -->
if "fold" in m and m.fold.nunique() >= 2:
    calibrated, choices = crossfit_threshold(m)
    display(choices)
    display(pd.DataFrame([
        {"Cách đo": "Ngưỡng cố định 0.5", "Macro-F1": f1_score(m.label, m.p_mean >= .5, average="macro")},
        {"Cách đo": "Chọn rồi chấm cùng tập", "Macro-F1": float(scores.max())},
        {"Cách đo": "Cross-fit ngưỡng", "Macro-F1": f1_score(calibrated.label, calibrated.crossfit_prediction, average="macro")}]))
    calibrated.to_csv(ws.output_root / "crossfit_threshold.csv", index=False)
else:
    print("Cần dự đoán từ ít nhất hai fold để chạy cross-fit ngưỡng.")

<!-- ailaai-cell:21:markdown -->
### Mở rộng: để Logistic Regression học cách kết hợp

Thay vì đặt trọng số 50/50, ta đưa hai cột xác suất RGB và High-pass vào Logistic Regression. Ở mỗi lượt meta-CV, bộ kết hợp học trên các fold khác rồi dự đoán fold giữ lại.

Ví dụ đầu dùng hai nguồn đang có trong notebook. Phần tiếp theo dùng đủ sáu nguồn của Clean Stack6 để đối chiếu với Phụ lục C. Hai phép thử có bộ đặc trưng khác nhau nên cần đọc điểm số riêng.

<!-- ailaai-cell:22:code -->
# ailaai-source: src/ailaai/teaching.py::crossfit_stack
def crossfit_stack(frame, columns=("p_rgb", "p_hp")):
    """Meta-CV on a fixed OOF feature matrix; this is not nested base-model CV."""
    from sklearn.linear_model import LogisticRegression
    if frame.fold.nunique() < 2 or not frame.file_name.is_unique:
        raise ValueError("Meta-CV needs at least two folds and unique file names.")
    parts = []
    for fold in sorted(frame.fold.unique()):
        fit = frame[frame.fold != fold]
        heldout = frame[frame.fold == fold].copy()
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=2026)
        meta.fit(fit[list(columns)], fit.label)
        heldout["p_stack"] = meta.predict_proba(heldout[list(columns)])[:, 1]
        parts.append(heldout)
    return pd.concat(parts, ignore_index=True)

<!-- ailaai-cell:23:code -->
if "fold" in m and m.fold.nunique() >= 2:
    stacked = crossfit_stack(m, columns=("p_rgb", "p_hp"))
    display(pd.DataFrame([{"Phương án": "Mean 50/50", **classification_report(m.label, m.p_mean)},
                          {"Phương án": "LR meta-CV hai nguồn", **classification_report(stacked.label, stacked.p_stack)}]))
    stacked.to_csv(ws.output_root / "stack_two_sources.csv", index=False)

<!-- ailaai-cell:24:markdown -->
### Đối chiếu Clean Stack6 với sáu nguồn đã lưu

Bundle `stack6_features.csv` ghép sáu nguồn theo tên ảnh, nhãn và fold: `center70`, `native410`, `preservejitter`, `highpass`, `mean2`, `vgg19bn`. Các tệp gốc và mã kiểm tra nằm trong `stack6_provenance.json`. Đây là những lượt chạy được chỉ rõ trong báo cáo Stack6, không mặc nhiên là nhánh RGB/HP Native358 đang dùng ở phần chính.

Báo cáo gốc dùng Logistic Regression trên logit đã chuẩn hóa. Ta viết lại bước này bên dưới: scaler và LR chỉ học từ các fold khác. Bảng kết quả đặt dự đoán đã lưu cạnh lượt fit lại để kiểm tra; nếu có chênh lệch thì giữ nguyên chênh lệch đó. Báo cáo nguồn chưa xác định đầy đủ chi tiết solver và clipping của code lịch sử.

Đây vẫn là meta-CV trên đặc trưng OOF đã chọn qua quá trình phát triển, không phải đánh giá độc lập của toàn bộ quá trình chọn mô hình. Bật `RUN_STACK6` để chạy trên CPU khi có bundle đi kèm. Nếu mở notebook từ GitHub, lấy `stack6_features.csv` và `stack6_provenance.json` trong gói ZIP bài giảng rồi chép vào `TASK / "reference_artifacts"` trước khi bật cờ này.

<!-- ailaai-cell:25:code -->
# ailaai-source: src/ailaai/teaching.py::crossfit_logit_stack
def crossfit_logit_stack(frame, columns):
    """Fit standardized-logit LR per held-out fold, following the saved Stack6 report."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if frame.fold.nunique() < 2 or not frame.file_name.is_unique:
        raise ValueError("Meta-CV needs multiple folds and unique file names.")
    probability = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Stacking inputs must be finite probabilities.")
    probability = np.clip(probability, 1e-6, 1-1e-6)
    logits = np.log(probability / (1-probability))
    output = frame.copy()
    output["p_stack_refit"] = np.nan
    for fold in sorted(frame.fold.unique()):
        heldout = (frame.fold == fold).to_numpy()
        # Both scaler and LR are fit on calibration folds only.
        meta = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=2026))
        meta.fit(logits[~heldout], frame.label.to_numpy()[~heldout])
        output.loc[heldout, "p_stack_refit"] = meta.predict_proba(logits[heldout])[:, 1]
    return output

<!-- ailaai-cell:26:code -->
RUN_STACK6 = False
if RUN_STACK6:
    import json
    asset = TASK / "reference_artifacts/stack6_features.csv"
    provenance_path = TASK / "reference_artifacts/stack6_provenance.json"
    if not asset.is_file() or not provenance_path.is_file():
        raise FileNotFoundError("Cần hai tệp Stack6 trong reference_artifacts của gói notebook đã bổ sung.")
    provenance = json.loads(provenance_path.read_text())
    assert sha256_file(asset) == provenance["matrix_sha256"]
    features6 = pd.read_csv(asset)
    columns6 = [item["name"] for item in provenance["feature_sources"]]
    refit6 = crossfit_logit_stack(features6, columns6)
    display(pd.DataFrame([{"Nguồn": name, **classification_report(refit6.label, refit6[column])}
                          for name, column in [("Stack6 đã lưu", "p_stack_saved"), ("Fit lại LR trên sáu nguồn", "p_stack_refit")]]))
    print("Chênh lệch xác suất lớn nhất:", float(abs(refit6.p_stack_refit - refit6.p_stack_saved).max()))
    refit6.to_csv(ws.output_root / "stack6_refit_vs_saved.csv", index=False)

<!-- ailaai-cell:27:markdown -->
## 6. Chọn phương án để chạy tiếp

Mức tăng nhỏ, chẳng hạn dưới 0.1%, trên một fold chưa đủ để quyết định dùng phương án phức tạp hơn. Cần xem kết quả có lặp lại và có đáng với chi phí bổ sung không.

Trong kết quả OOF mẫu, Simple Mean 50/50 với RGB và High-pass tại $t = 0.50$ đạt Macro-F1 96.45%. Cách này không cần học thêm tham số kết hợp.

Ta ghi lại phương án đã chọn trước khi dự đoán tập test.

<!-- ailaai-cell:28:code -->
from ailaai.pipeline import save_analysis_note
FINAL_THRESHOLD = 0.5
save_analysis_note(ws, rgb=rgb, highpass=hp, threshold=FINAL_THRESHOLD,
                   method="probability_mean_50_50",
                   note="Giữ phương án này; threshold sweep chỉ để quan sát.")

<!-- ailaai-cell:29:markdown -->
### Đóng gói CSV và đọc lại để kiểm tra

Theo Reading III.8, ZIP chỉ chứa `submission.csv` ở thư mục gốc. CSV phải có đúng thứ tự hai cột, đủ tên ảnh test, nhãn 0 hoặc 1, không trùng và không thiếu giá trị. Các hàm dưới thực hiện việc ghi file và kiểm tra lại nội dung vừa ghi.

<!-- ailaai-cell:30:code -->
# ailaai-source: src/ailaai/submission.py::SubmissionReceipt,_validate_frame
@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    """Validation details for a written submission archive."""

    path: Path
    sha256: str
    row_count: int
    member: str
    columns: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "row_count": self.row_count,
                "member": self.member, "columns": list(self.columns), "valid": True}


def _validate_frame(frame: pd.DataFrame, expected_names: Iterable[str], expected_count: int) -> pd.DataFrame:
    names = list(expected_names)
    if len(names) != expected_count or len(set(names)) != expected_count:
        raise ValueError(f"Expected test names must contain exactly {expected_count} unique entries.")
    if list(frame.columns) != ["file_name", "category_id"]:
        raise ValueError("Submission columns and order must be exactly file_name,category_id.")
    if len(frame) != expected_count:
        raise ValueError(f"Submission must have exactly {expected_count} data rows; found {len(frame)}.")
    result = frame.copy()
    if result.isna().any().any():
        raise ValueError("Submission contains missing values.")
    result["file_name"] = result.file_name.astype(str)
    if not result.file_name.is_unique:
        raise ValueError("Submission contains duplicate file_name values.")
    if set(result.file_name) != set(names):
        missing = sorted(set(names) - set(result.file_name))[:5]
        extra = sorted(set(result.file_name) - set(names))[:5]
        raise ValueError(f"Submission file names differ from test; missing={missing}, extra={extra}.")
    if not result.category_id.astype(str).isin(["0", "1"]).all():
        raise ValueError("category_id must contain literal integer labels 0 or 1.")
    labels = pd.to_numeric(result.category_id, errors="coerce")
    if labels.isna().any() or not np.equal(labels, np.floor(labels)).all() or not labels.isin([0, 1]).all():
        raise ValueError("category_id values must be integer labels 0 or 1.")
    result["category_id"] = labels.astype("int64")
    return result

<!-- ailaai-cell:31:code -->
# ailaai-source: src/ailaai/submission.py::validate_submission,export_submission
def validate_submission(
    zip_path: str | Path,
    expected_names: Iterable[str],
    expected_count: int = 200,
    expected_frame: pd.DataFrame | None = None,
    receipt_path: str | Path | None = None,
) -> SubmissionReceipt:
    """Read a ZIP back and check member, CRC, schema, IDs, labels, and optional values."""
    source = Path(zip_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        with zipfile.ZipFile(source) as archive:
            if archive.testzip() is not None:
                raise ValueError("Submission ZIP CRC check failed.")
            members = archive.namelist()
            if members != ["submission.csv"]:
                raise ValueError("ZIP must contain exactly one root member named submission.csv.")
            with archive.open("submission.csv") as handle:
                returned = pd.read_csv(handle, dtype={"file_name": str, "category_id": str})
    except zipfile.BadZipFile as exc:
        raise ValueError("Submission is not a readable ZIP archive.") from exc
    returned = _validate_frame(returned, expected_names, expected_count)
    if expected_frame is not None:
        expected = _validate_frame(expected_frame, expected_names, expected_count)
        left = returned.sort_values("file_name").reset_index(drop=True)
        right = expected.sort_values("file_name").reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("Read-back submission values do not match the frame that was exported.")
    receipt = SubmissionReceipt(source.resolve(), sha256_file(source), len(returned), "submission.csv", tuple(returned.columns))
    if receipt_path:
        write_json(receipt_path, receipt.summary())
    return receipt


def export_submission(
    frame: pd.DataFrame,
    zip_path: str | Path,
    expected_names: Iterable[str],
    expected_count: int = 200,
) -> Path:
    """Validate and atomically create a ZIP with one root-level submission.csv."""
    expected_list = list(expected_names)
    normalized = _validate_frame(frame, expected_list, expected_count)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("submission.csv", normalized.to_csv(index=False, lineterminator="\n"))
        validate_submission(temporary, expected_list, expected_count, expected_frame=normalized)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination

<!-- ailaai-cell:32:markdown -->
## 7. Tạo và kiểm tra tệp nộp bài

`export_submission` và `validate_submission` kiểm tra các điều kiện sau:
- CSV nằm ở gốc tệp ZIP.
- Có hai cột `file_name,category_id`.
- Đủ số dòng theo danh sách test và không thiếu giá trị.
- Nhãn là số nguyên 0 hoặc 1.
- Tên ảnh khớp danh sách test, không trùng và không thiếu ảnh.

<!-- ailaai-cell:33:code -->
DEMO_NAMES = [f"demo_{i:03d}.jpg" for i in range(200)]
demo = pd.DataFrame({"file_name": DEMO_NAMES,
                     "category_id": np.arange(200, dtype="int64") % 2})
display(demo.head())

<!-- ailaai-cell:34:code -->
demo_path = export_submission(demo, ws.output_root / "demo_submission.zip",
                              expected_names=DEMO_NAMES, expected_count=200)
demo_receipt = validate_submission(demo_path, expected_names=DEMO_NAMES,
                                  expected_count=200, expected_frame=demo)
print(demo_receipt.summary())
print("File minh họa; không phải dự đoán test.")

<!-- ailaai-cell:35:markdown -->
## 8. Thử một trường hợp dữ liệu lỗi

Cell dưới tạo hai dòng trùng tên ảnh `file_name` để xem bộ kiểm tra có phát hiện lỗi không.

<!-- ailaai-cell:36:code -->
bad = demo.copy()
bad.loc[1, "file_name"] = bad.loc[0, "file_name"]
try:
    export_submission(bad, ws.output_root / "demo_duplicate.zip",
                      expected_names=DEMO_NAMES, expected_count=200)
except ValueError as exc:
    print("Đã phát hiện lỗi:", exc)
else:
    raise AssertionError("Bộ kiểm tra chưa phát hiện tên ảnh trùng.")

<!-- ailaai-cell:37:markdown -->
## 9. Chuyển sang pipeline đầy đủ

Sau khi xem kết quả kết hợp, cách chọn ngưỡng và cách kiểm tra ZIP, bạn có thể mở `00_pipeline_end_to_end.ipynb` để chạy quy trình từ dữ liệu đến tệp nộp bài.

<!-- ailaai-cell:38:code -->
print("Nguồn:", SOURCE, "| Số ảnh validation:", len(m))
print("Threshold đã ghi:", FINAL_THRESHOLD)
print("Notebook dự đoán test:", TASK / "notebooks/00_pipeline_end_to_end.ipynb")

<!-- ailaai-cell:39:markdown -->
Kết quả phân tích trong bài này dùng dự đoán OOF. `demo_submission.zip` là tệp minh họa cách đóng gói, không phải bài nộp cho Private Test. Để dự đoán trên ảnh test thực tế, dùng `00_pipeline_end_to_end.ipynb`.

<!-- ailaai-cell:40:markdown -->
## Đọc thêm và xem mã nguồn

[Bản đồ Reading và notebook](../READING_NOTEBOOK_MAP.md) chỉ từng nội dung của bài đọc đến các cell và tệp Python tương ứng. Bài đọc chính là `OlympicAI_2026/topic_ai_la_ai/reading.tex`, các phần II–IV và Phụ lục B–D trong workspace bài giảng.

Dòng `# ailaai-source` ở đầu một số cell chỉ nơi lưu hàm trong `src/ailaai/`. Khi thực hành, bạn có thể sửa ngay trong cell. Khi biên soạn lại tài liệu, sửa tệp Python rồi chạy `python scripts/build_notebooks.py --write` để đồng bộ.

Tài liệu của thư viện:

- PyTorch: [học chuyển giao](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html), [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [AMP](https://docs.pytorch.org/docs/stable/amp.html).
- scikit-learn: [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html), [F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Khi báo kết quả, ghi kèm cấu hình và số ảnh đã đánh giá. Các ví dụ CPU giúp kiểm tra cách tính; số liệu `reference` là kết quả đã lưu. Muốn đánh giá lượt huấn luyện của mình, dùng checkpoint, log và dự đoán do lượt đó tạo ra.
