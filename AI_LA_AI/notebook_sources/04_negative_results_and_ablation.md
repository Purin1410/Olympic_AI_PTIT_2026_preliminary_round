<!-- ailaai-cell:00:markdown -->
# Phụ lục 4: Ba thử nghiệm chưa cải thiện kết quả và cách quyết định dừng

**Thời lượng:** khoảng 15 phút đọc và thảo luận.  
**Cách chạy:** Tính lại kết quả trên CPU từ dự đoán OOF đã lưu; không cần GPU hay huấn luyện lại.

Ta xem ba hướng đã thử: biểu diễn Wavelet 2D, tăng trọng số cho nhóm ảnh Fake ít biên và chọn ngưỡng riêng cho ảnh xám/màu. Mỗi trường hợp đặt ra một câu hỏi khác nhau khi đọc kết quả: có sửa được nhiều lỗi hơn số lỗi mới, có cải thiện đúng nhóm mục tiêu và có đánh giá ngưỡng trên dữ liệu độc lập không?

<!-- ailaai-cell:01:markdown -->
## 1. Quy ước và cách đọc kết quả

Nhãn Real là 0, Fake là 1. FN (false negative) là ảnh Fake bị đoán thành Real; FP (false positive) là ảnh Real bị đoán thành Fake.

Khi so sánh phương án B với baseline A:
- **Fixes:** A sai, B đúng.
- **Breaks:** A đúng, B sai.
- **Lỗi tăng ròng:** $\text{Breaks} - \text{Fixes}$.

Các bảng dưới được tính từ dự đoán OOF của các thử nghiệm đã lưu. Ta sẽ đọc cả điểm tổng lẫn các nhóm ảnh được sửa hoặc sai thêm.

<!-- ailaai-cell:02:markdown -->
## 2. Đọc và kiểm tra dữ liệu đã lưu

Gói `data/negative_results/` gồm:
- `oof_predictions.csv`: 2.000 dòng dự đoán OOF của các nhánh.
- `threshold_choices.csv`: ngưỡng đã chọn theo fold cho ảnh xám/màu.
- `promotion_gates.json`: ba tiêu chí chọn phương án và kết quả đánh giá.
- Ba hình: `wavelet_case.png`, `edge_subgroup_case.png`, `threshold_case.png`.

Bộ ảnh gốc có tại [who_is_AI (Google Drive)](https://drive.google.com/file/d/1g_43_Xn-DWYB-k7Yq4XQTr5ZQXZ0UdXq/view?usp=drive_link).

Trước khi so sánh, cell dưới kiểm tra các dự đoán được ghép theo cùng tên ảnh, nhãn và fold.

<!-- ailaai-cell:03:code -->
from pathlib import Path
import json
import math
import statistics

import importlib.util
import subprocess
import sys
missing = [name for name in ("pandas", "IPython") if importlib.util.find_spec(name) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
import pandas as pd
from IPython.display import display, Image

# Tự lấy gói kết quả khi notebook được mở riêng trên Colab.
import os
import subprocess
import sys
TASK = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "data/negative_results/oof_predictions.csv").is_file()), None)
if TASK is None:
    REPO = Path("/content" if Path("/content").is_dir() else Path.cwd()) / "Olympic_AI_PTIT_2026_preliminary_round"
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch",
                        os.environ.get("AILAAI_RELEASE_REF", "main"),
                        "https://github.com/Purin1410/Olympic_AI_PTIT_2026_preliminary_round.git", str(REPO)], check=True)
    TASK = REPO / "AI_LA_AI"
DATA_DIR = TASK / "data/negative_results"

OOF_PATH = DATA_DIR / "oof_predictions.csv"
THRESHOLD_PATH = DATA_DIR / "threshold_choices.csv"
GATES_PATH = DATA_DIR / "promotion_gates.json"
FIGURE_PATHS = {
    "wavelet": DATA_DIR / "wavelet_case.png",
    "edge": DATA_DIR / "edge_subgroup_case.png",
    "threshold": DATA_DIR / "threshold_case.png",
}
for path in [OOF_PATH, THRESHOLD_PATH, GATES_PATH, *FIGURE_PATHS.values()]:
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu đầu vào bắt buộc: {path.name}")

REQUIRED_COLUMNS = [
    "file_name", "label", "fold", "rgb_prob", "wavelet_prob",
    "edge_base_prob", "edge_weighted_prob", "legacy_stack_prob",
    "is_gray", "edge_ratio",
]
oof = pd.read_csv(OOF_PATH)
thresholds = pd.read_csv(THRESHOLD_PATH)
with GATES_PATH.open(encoding="utf-8") as handle:
    gate_receipt = json.load(handle)

if list(oof.columns) != REQUIRED_COLUMNS:
    raise ValueError(f"OOF header không đúng contract: {list(oof.columns)}")
if len(oof) != 2000 or oof["file_name"].isna().any():
    raise ValueError("OOF cần đúng 2.000 dòng, không thiếu file_name.")
if oof["file_name"].duplicated().any():
    raise ValueError("file_name bị trùng; không thể ghép cặp an toàn.")
if not set(oof["label"].unique()).issubset({0, 1}):
    raise ValueError("label chỉ được nhận 0=Real hoặc 1=Fake.")
if not set(oof["fold"].unique()).issubset({0, 1, 2, 3, 4}):
    raise ValueError("fold phải thuộc {0, 1, 2, 3, 4}.")
if not set(oof["is_gray"].unique()).issubset({0, 1}):
    raise ValueError("is_gray chỉ được nhận 0 hoặc 1.")

probability_columns = [
    "rgb_prob", "wavelet_prob", "edge_base_prob",
    "edge_weighted_prob", "legacy_stack_prob",
]
for column in probability_columns + ["edge_ratio"]:
    oof[column] = pd.to_numeric(oof[column], errors="coerce")
    if oof[column].isna().any() or not oof[column].map(math.isfinite).all():
        raise ValueError(f"{column} có giá trị thiếu hoặc không hữu hạn.")
for column in probability_columns:
    if not oof[column].between(0, 1).all():
        raise ValueError(f"{column} phải nằm trong [0, 1].")

if list(thresholds.columns) != ["fold", "gray_threshold", "color_threshold"]:
    raise ValueError("Header threshold_choices.csv không đúng contract.")
if len(thresholds) != 5 or set(thresholds["fold"]) != {0, 1, 2, 3, 4}:
    raise ValueError("Cần đúng một threshold đã chọn cho mỗi fold 0-4.")
if thresholds["fold"].duplicated().any():
    raise ValueError("threshold_choices.csv có fold trùng.")
for column in ["gray_threshold", "color_threshold"]:
    thresholds[column] = pd.to_numeric(thresholds[column], errors="coerce")
    if thresholds[column].isna().any() or not thresholds[column].between(0, 1).all():
        raise ValueError(f"{column} thiếu hoặc ngoài [0, 1].")

print(f"Bundle: {DATA_DIR}")
print(f"OOF rows: {len(oof)} | folds: {sorted(oof.fold.unique())}")
print("Xác nhận dữ liệu: CSV, ngưỡng, receipt và ba PNG đều hợp lệ 100%.")

<!-- ailaai-cell:04:markdown -->
## 3. Thử nghiệm 1: Biểu diễn Wavelet

Nhánh này phân rã ảnh bằng Wavelet 2D thành bốn dải LL, LH, HL và HH để thử biểu diễn thông tin tần số cho mô hình.

Nhánh Wavelet sửa được bao nhiêu ảnh RGB đoán sai, và làm sai thêm bao nhiêu ảnh? Khi so Wavelet ResNet18 với RGB ResNet34, ta đã thay đổi những yếu tố nào?

<!-- ailaai-cell:05:code -->
def predictions(probabilities, threshold=0.5):
    return (pd.Series(probabilities).astype(float) >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    pairs = list(zip(map(int, y_true), map(int, y_pred)))
    return {
        "TN": sum(y == 0 and p == 0 for y, p in pairs),
        "FP": sum(y == 0 and p == 1 for y, p in pairs),
        "FN": sum(y == 1 and p == 0 for y, p in pairs),
        "TP": sum(y == 1 and p == 1 for y, p in pairs),
    }


def transition_counts(y_true, pred_a, pred_b):
    y = list(map(int, y_true))
    a = list(map(int, pred_a))
    b = list(map(int, pred_b))
    correct_a = [pa == yi for yi, pa in zip(y, a)]
    correct_b = [pb == yi for yi, pb in zip(y, b)]
    fixes = sum((not ca) and cb for ca, cb in zip(correct_a, correct_b))
    breaks = sum(ca and (not cb) for ca, cb in zip(correct_a, correct_b))
    return {"fixes": fixes, "breaks": breaks, "net errors (B − A)": breaks - fixes}


def macro_f1(y_true, y_pred):
    cm = confusion_counts(y_true, y_pred)
    f1_real = 2 * cm["TN"] / max(1, 2 * cm["TN"] + cm["FP"] + cm["FN"])
    f1_fake = 2 * cm["TP"] / max(1, 2 * cm["TP"] + cm["FP"] + cm["FN"])
    return (f1_real + f1_fake) / 2


def paired_report(frame, a_column, b_column, threshold=0.5):
    y = frame["label"].astype(int)
    a = predictions(frame[a_column], threshold)
    b = predictions(frame[b_column], threshold)
    a_correct = a.to_numpy() == y.to_numpy()
    b_correct = b.to_numpy() == y.to_numpy()
    fixes = int((~a_correct & b_correct).sum())
    breaks = int((a_correct & ~b_correct).sum())
    return {
        "Nhánh A": a_column,
        "Nhánh B": b_column,
        "n": len(frame),
        "A Macro-F1 (%)": 100 * macro_f1(y, a),
        "B Macro-F1 (%)": 100 * macro_f1(y, b),
        "A errors": int((~a_correct).sum()),
        "B errors": int((~b_correct).sum()),
        "fixes": fixes,
        "breaks": breaks,
        "net errors (B − A)": breaks - fixes,
        "A confusion": confusion_counts(y, a),
        "B confusion": confusion_counts(y, b),
    }


wavelet_result = paired_report(oof, "rgb_prob", "wavelet_prob")
display(pd.DataFrame([wavelet_result]).T.rename(columns={0: "Kết quả"}))
assert wavelet_result["fixes"] == 30, "Fixes không khớp receipt lịch sử."
assert wavelet_result["breaks"] == 183, "Breaks không khớp receipt lịch sử."
assert wavelet_result["net errors (B − A)"] == 153

<!-- ailaai-cell:06:code -->
display(Image(filename=str(FIGURE_PATHS["wavelet"])))

<!-- ailaai-cell:07:markdown -->
### Kết quả Wavelet và giới hạn của phép so sánh

Wavelet sửa 30 ảnh RGB đoán sai nhưng làm sai thêm 183 ảnh. Tổng lỗi tăng 153; Macro-F1 giảm từ 95,40% xuống 87,75%.

ResNet18 được dùng để thăm dò trước khi đầu tư huấn luyện Wavelet với ResNet34. Nhánh High-pass đã có kết quả ResNet18 94,50% và ResNet34 95,15%, chênh 0,65 điểm phần trăm. Các số này cung cấp bối cảnh cho quyết định dừng nhánh Wavelet, nhưng không dự đoán được mức cải thiện của Wavelet khi đổi backbone.

Phép so sánh Wavelet-R18 với RGB-R34 đổi cả biểu diễn, backbone và batch (16 sang 32). Vì vậy, kết quả cho thấy cấu hình Wavelet đã thử chưa đạt yêu cầu; chưa tách được ảnh hưởng riêng của biểu diễn Wavelet. Muốn kiểm tra riêng yếu tố này, cần giữ các điều kiện còn lại giống nhau.

Khi đọc bảng, hãy xem cả số ca được sửa và số ca sai thêm.

<!-- ailaai-cell:08:markdown -->
## 4. Thử nghiệm 2: Tăng trọng số cho ảnh Fake ít biên

Phân tích lỗi gợi ý thử tập trung vào ảnh Fake có tỷ lệ biên thấp. Phương án này tăng trọng số loss lên $1{,}5\times$ cho nhóm $25\%$ ảnh Fake ít biên nhất trong tập train.

Nếu Macro-F1 trên 2.000 ảnh tăng, số ca bỏ sót ở nhóm muốn cải thiện có giảm theo không?

<!-- ailaai-cell:09:code -->
edge_global = paired_report(oof, "edge_base_prob", "edge_weighted_prob")
display(pd.DataFrame([
    {
        "Recipe": "Baseline",
        "n": edge_global["n"],
        "Macro-F1 (%)": edge_global["A Macro-F1 (%)"],
        "FP": edge_global["A confusion"]["FP"],
        "FN": edge_global["A confusion"]["FN"],
    },
    {
        "Recipe": "Edge subgroup weighting",
        "n": edge_global["n"],
        "Macro-F1 (%)": edge_global["B Macro-F1 (%)"],
        "FP": edge_global["B confusion"]["FP"],
        "FN": edge_global["B confusion"]["FN"],
    },
]).round(4))
assert edge_global["n"] == 2000

<!-- ailaai-cell:10:code -->
cutoff_records = gate_receipt.get("cutoffs")
if not isinstance(cutoff_records, list):
    raise ValueError("promotion_gates.json cần danh sách cutoffs theo fold.")
cutoffs = {int(row["fold"]): float(row["cutoff"]) for row in cutoff_records}
if set(cutoffs) != {0, 1, 2, 3, 4}:
    raise ValueError("Cần cutoff edge-ratio cho đủ năm fold.")

edge_rows = oof.copy()
edge_rows["low_edge_group"] = edge_rows.apply(
    lambda row: float(row["edge_ratio"]) <= cutoffs[int(row["fold"])], axis=1
)
target_fake = edge_rows[(edge_rows.label == 1) & edge_rows.low_edge_group]
target_real = edge_rows[(edge_rows.label == 0) & edge_rows.low_edge_group]
fake_base = predictions(target_fake.edge_base_prob)
fake_weighted = predictions(target_fake.edge_weighted_prob)
real_base = predictions(target_real.edge_base_prob)
real_weighted = predictions(target_real.edge_weighted_prob)

display(pd.DataFrame([
    {
        "Nhóm": "Fake edge-ratio thấp",
        "n": len(target_fake),
        "FN baseline": int((fake_base == 0).sum()),
        "FN weighted": int((fake_weighted == 0).sum()),
    },
    {
        "Nhóm": "Real edge-ratio thấp",
        "n": len(target_real),
        "FP baseline": int((real_base == 1).sum()),
        "FP weighted": int((real_weighted == 1).sum()),
    },
]))
assert len(target_fake) == 248 and len(target_real) == 397
assert int((fake_base == 0).sum()) == 17
assert int((fake_weighted == 0).sum()) == 18
assert int((real_base == 1).sum()) == 10
assert int((real_weighted == 1).sum()) == 11

recorded_gates = gate_receipt.get("gates", gate_receipt)
gate_names = ["G1_mechanism", "G2_collateral", "G3_stack"]
if not all(name in recorded_gates for name in gate_names):
    raise ValueError("Receipt thiếu một trong ba cờ G1/G2/G3.")
stack_info = gate_receipt.get("stack_attribution", {})
recorded_stack_gain = stack_info.get("total_net_error_reduction")

observed_checks = [
    int((fake_weighted == 0).sum()) < int((fake_base == 0).sum()),
    int((real_weighted == 1).sum()) <= int((real_base == 1).sum()),
    recorded_stack_gain is not None and float(recorded_stack_gain) > 0,
]
gate_table = pd.DataFrame([
    {"Gate": name, "Đạt theo receipt": bool(recorded_gates[name]),
     "Quan sát đơn giản": bool(observed)}
    for name, observed in zip(gate_names, observed_checks)
])
display(gate_table)
if any(bool(recorded_gates[name]) for name in gate_names):
    raise ValueError("Receipt không khớp trạng thái đã ghi cho case này.")

<!-- ailaai-cell:11:code -->
display(Image(filename=str(FIGURE_PATHS["edge"])))

<!-- ailaai-cell:12:markdown -->
### Điểm tổng và kết quả ở nhóm mục tiêu

Macro-F1 tăng từ 95,40% lên 95,55%; FN trên toàn bộ dữ liệu giảm từ 64 xuống 61.

Ngưỡng phân vị 25% được tính trên ảnh Fake của 4 fold train, rồi áp dụng cho cả hai lớp ở fold OOF còn lại. Gộp các fold có 248 ảnh Fake và 397 ảnh Real trong nhóm ít biên. Nhóm này không được tạo bằng cách lấy trực tiếp 25% của 1.000 ảnh Fake OOF.

Ở 248 ảnh Fake mục tiêu, FN tăng từ 17 lên 18. Với 397 ảnh Real cùng dải biên thấp, FP tăng từ 10 lên 11. Điểm tổng tăng nhưng hai nhóm đang quan tâm đều có thêm lỗi.

Ba tiêu chí đã đặt ra cho phương án này:
- **G1:** Giảm FN ở nhóm Fake mục tiêu. Chưa đạt: 17 lên 18.
- **G2:** Không tăng FP ở nhóm Real tương ứng. Chưa đạt: 10 lên 11.
- **G3:** Giảm tổng lỗi khi thay nhánh vào mô hình kết hợp. Chưa đạt: mức giảm lỗi ròng bằng 0.

Phương án chưa đạt ba tiêu chí, nên chưa được chọn thay baseline. Khi thử cải thiện một nhóm ảnh cụ thể, cần xem kết quả của nhóm đó cùng với điểm tổng.

<!-- ailaai-cell:13:markdown -->
## 5. Thử nghiệm 3: Chọn ngưỡng riêng cho ảnh xám và ảnh màu

Phép thử dùng cột `legacy_stack_prob` của Stacking Legal7 đã lưu. Với ngưỡng cố định $t = 0.50$, mô hình đạt Macro-F1 96,70% và có 66 lỗi. Đây là kết quả của Legal7, không phải Simple Mean hay Clean Stack6.

Ta thử chọn hai ngưỡng: $t_{\text{gray}}$ cho ảnh xám và $t_{\text{color}}$ cho ảnh màu.

Cặp ngưỡng xám 0.485 và màu 0.510 cho điểm 96,75% khi chấm lại trên OOF. Dữ liệu dùng để chọn ngưỡng và dữ liệu dùng để báo điểm có tách biệt không?

<!-- ailaai-cell:14:code -->
thresholds = thresholds.sort_values("fold").reset_index(drop=True)
threshold_map = thresholds.set_index("fold")
gray_median = statistics.median(thresholds.gray_threshold.tolist())
color_median = statistics.median(thresholds.color_threshold.tolist())

y = oof.label.astype(int)
score = oof.legacy_stack_prob.astype(float)
fold = oof.fold.astype(int)
gray = oof.is_gray.astype(int) == 1

fixed_pred = (score >= 0.5).astype(int)
crossfit_threshold = pd.Series([
    float(threshold_map.loc[int(f), "gray_threshold" if is_gray else "color_threshold"])
    for f, is_gray in zip(fold, gray)
], index=oof.index)
crossfit_pred = (score >= crossfit_threshold).astype(int)
median_threshold = pd.Series([
    gray_median if is_gray else color_median for is_gray in gray
], index=oof.index)
median_pred = (score >= median_threshold).astype(int)

crossfit_transitions = transition_counts(y, fixed_pred, crossfit_pred)
median_transitions = transition_counts(y, fixed_pred, median_pred)

threshold_results = pd.DataFrame([
    {
        "Cách chấm": "Ngưỡng cố định 0,500",
        "Macro-F1 (%)": 100 * macro_f1(y, fixed_pred),
        **confusion_counts(y, fixed_pred),
        "fixes so với fixed": 0,
        "breaks so với fixed": 0,
        "net errors so với fixed": 0,
        "Diễn giải": "Mốc cố định",
    },
    {
        "Cách chấm": "Threshold cross-fit",
        "Macro-F1 (%)": 100 * macro_f1(y, crossfit_pred),
        **confusion_counts(y, crossfit_pred),
        "fixes so với fixed": crossfit_transitions["fixes"],
        "breaks so với fixed": crossfit_transitions["breaks"],
        "net errors so với fixed": crossfit_transitions["net errors (B − A)"],
        "Diễn giải": "Mỗi fold dùng ngưỡng chọn từ bốn fold còn lại",
    },
    {
        "Cách chấm": "Median threshold áp lại toàn OOF",
        "Macro-F1 (%)": 100 * macro_f1(y, median_pred),
        **confusion_counts(y, median_pred),
        "fixes so với fixed": median_transitions["fixes"],
        "breaks so với fixed": median_transitions["breaks"],
        "net errors so với fixed": median_transitions["net errors (B − A)"],
        "Diễn giải": "Tái sử dụng OOF labels để chọn và chấm",
    },
])
display(threshold_results.round(4))
print(f"Median gray threshold: {gray_median:.3f}")
print(f"Median color threshold: {color_median:.3f}")

assert abs(gray_median - 0.485) < 1e-9
assert abs(color_median - 0.510) < 1e-9
assert abs(100 * macro_f1(y, fixed_pred) - 96.6998) < 0.01
assert abs(100 * macro_f1(y, crossfit_pred) - 96.3996) < 0.01
assert abs(100 * macro_f1(y, median_pred) - 96.7498) < 0.01

<!-- ailaai-cell:15:code -->
display(Image(filename=str(FIGURE_PATHS["threshold"])))

<!-- ailaai-cell:16:markdown -->
### So sánh cách chọn và đánh giá ngưỡng

| Cách đánh giá | Macro-F1 | Kết quả |
| --- | --- | --- |
| Giữ $t = 0.50$ | 96,70% | 66 lỗi, mốc tham chiếu của Legal7. |
| Áp ngưỡng trung vị lên lại OOF | 96,75% | Dữ liệu đã góp phần chọn ngưỡng được dùng lại để báo điểm. |
| Đánh giá chéo (cross-fit) | 96,40% | Chọn ngưỡng trên 4 fold, đánh giá ở fold còn lại; thêm 6 lỗi so với mốc 0.50. |

Điểm chấm lại trên dữ liệu chọn ngưỡng có thể lạc quan (optimism bias). Trong phép thử Legal7 này, chính sách tách ngưỡng không tốt hơn ngưỡng cố định khi đánh giá chéo.

Ta giữ $t = 0.50$ cho phương án này. Kết quả không có nghĩa ngưỡng 0.50 luôn tốt nhất; thay đổi ngưỡng cần được đánh giá trên dữ liệu không tham gia chọn ngưỡng.

<!-- ailaai-cell:17:markdown -->
## 6. Những câu hỏi trước khi chọn phương án mới

1. Hai phương án có được đánh giá trên cùng ảnh, nhãn và fold không?
2. Ta thay đổi một yếu tố hay nhiều yếu tố cùng lúc?
3. Nhóm ảnh muốn cải thiện có tốt hơn không?
4. Có bao nhiêu ca được sửa và bao nhiêu ca sai thêm?
5. Dữ liệu đánh giá có tham gia chọn ngưỡng hoặc siêu tham số không?

Nếu phương án chưa đạt tiêu chí, ghi lại kết quả và lý do chưa chọn. Ta có thể giữ baseline để tiếp tục thử hướng khác.
