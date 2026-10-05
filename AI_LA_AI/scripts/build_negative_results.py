"""Package train-only historical OOF and static Matplotlib figures for appendix 04.

This is an authoring tool; the notebook only needs the committed bundle and pandas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["file_name", "label", "fold"]
RUNS = {
    "rgb_prob": "r34_center70_native358_jitter410_s2026",
    "wavelet_prob": "overnight_wavelet_r18_native358_s2026",
    "edge_base_prob": "r34_center70_native358_jitter410_s2026",
    "edge_weighted_prob": "r34_native358_edgeq25_massnorm_w15_s2026",
    # The calibrated stack7 OOF stores hard decisions, not input probabilities.
    # This raw legal7 OOF reproduces every cross-fit decision in that receipt.
    "legacy_stack_prob": "stack_crossfit_legal7_c1",
}
GRAY_LOGIC = "RGB thumbnail((64,64)); mean(max(pixel)-min(pixel)) < 0.5"
BLUE, RED, TEXT = "#245A81", "#AC3E35", "#233342"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_join(left, right, keys):
    if right[keys].isna().any().any() or right.file_name.duplicated().any():
        raise ValueError("Source has missing/duplicate identifiers")
    if len(right) != len(left):
        raise ValueError("Source population differs from the canonical split")
    merged = left.merge(right, on=keys, how="left", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Source IDs, labels or folds disagree with the canonical split")
    return merged.drop(columns="_merge")


def gray_flag(path):
    # Exact logic used by calibrate_stack_gray_color.py (not the other calibrator).
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((64, 64))
        pixels = np.asarray(rgb, dtype=np.int16)
    return int((pixels.max(axis=2) - pixels.min(axis=2)).mean() < 0.5)


def counts(frame, column, threshold=0.5):
    y, pred = frame.label.to_numpy(), frame[column].to_numpy() >= threshold
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tp = int(((y == 1) & pred).sum())
    f1 = (2 * tn / (2 * tn + fp + fn) + 2 * tp / (2 * tp + fp + fn)) / 2
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "macro_f1": f1}


def save_figure(fig, out, name, title, footer):
    fig.suptitle(title, x=0.06, ha="left", fontsize=17, fontweight="bold", color=TEXT)
    fig.text(0.06, 0.025, footer, fontsize=10, color=TEXT, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.81, bottom=0.22, wspace=0.40)
    fig.savefig(out / name, dpi=140, facecolor="white")
    plt.close(fig)


def bars(ax, labels, values, colors, ylabel):
    rectangles = ax.bar(labels, values, color=colors, width=0.55)
    ax.bar_label(rectangles, padding=5, fmt="%.0f")
    ax.set_ylim(0, max(values) * 1.28)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E9ED", linewidth=0.8)


def figures(frame, thresholds, gates, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": TEXT, "axes.labelcolor": TEXT,
                         "axes.spines.top": False, "axes.spines.right": False})
    rgb, wave = counts(frame, "rgb_prob"), counts(frame, "wavelet_prob")
    a = (frame.rgb_prob >= 0.5) == frame.label
    b = (frame.wavelet_prob >= 0.5) == frame.label
    fixes, breaks = int((~a & b).sum()), int((a & ~b).sum())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    bars(axes[0], ["RGB R34", "Wavelet R18"],
         [rgb["fp"] + rgb["fn"], wave["fp"] + wave["fn"]], [BLUE, RED], "Tổng lỗi / 2.000 ảnh")
    axes[0].set_title(f'Macro-F1: {100*rgb["macro_f1"]:.2f}% → {100*wave["macro_f1"]:.2f}%')
    bars(axes[1], ["Sửa đúng (fixes)", "Làm hỏng (breaks)"], [fixes, breaks],
         [BLUE, RED], "Số ảnh chuyển trạng thái")
    axes[1].set_title(f"Tăng ròng {breaks-fixes} lỗi")
    save_figure(fig, out, "wavelet_case.png", "Wavelet: sửa 30 ảnh, làm hỏng 183 ảnh",
                "OOF lịch sử · t = 0,50 · Cấu hình đổi cả biểu diễn, backbone và batch size; chưa cô lập nguyên nhân.")

    cutoff_map = {row["fold"]: row["cutoff"] for row in gates["cutoffs"]}
    low = frame.edge_ratio <= frame.fold.map(cutoff_map)
    fake, real = frame[low & (frame.label == 1)], frame[low & (frame.label == 0)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    bars(axes[0], ["Baseline", "Weighted"],
         [counts(fake, "edge_base_prob")["fn"], counts(fake, "edge_weighted_prob")["fn"]],
         [BLUE, RED], "Fake bị bỏ sót (FN)")
    axes[0].set_title(f"Nhóm mục tiêu: {len(fake)} ảnh Fake ít biên")
    bars(axes[1], ["Baseline", "Weighted"],
         [counts(real, "edge_base_prob")["fp"], counts(real, "edge_weighted_prob")["fp"]],
         [BLUE, RED], "Real bị báo nhầm (FP)")
    axes[1].set_title(f"Tác dụng phụ: {len(real)} ảnh Real ít biên")
    save_figure(fig, out, "edge_subgroup_case.png", "Điểm tổng tăng; nhóm cần sửa lại tăng lỗi",
                "OOF lịch sử · Feature Laplacian 64px · Cutoff từ Fake của 4 fold còn lại · G1 / G2 / G3 đều thất bại.")

    table = thresholds.set_index("fold")
    cross = np.array([table.loc[f, "gray_threshold" if g else "color_threshold"]
                      for f, g in zip(frame.fold, frame.is_gray)])
    median = np.where(frame.is_gray, thresholds.gray_threshold.median(), thresholds.color_threshold.median())
    scores = [100 * counts(frame, "legacy_stack_prob", t)["macro_f1"] for t in [0.5, cross, median]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    axes[0].plot(thresholds.fold, thresholds.gray_threshold, "o-", color=BLUE, label="Ảnh xám")
    axes[0].plot(thresholds.fold, thresholds.color_threshold, "s--", color=RED, label="Ảnh màu")
    axes[0].axhline(0.5, color="#647383", linestyle=":", label="Mốc 0,50")
    axes[0].set(xlabel="Fold kiểm định", ylabel="Ngưỡng chọn trên 4 fold còn lại",
                xticks=range(5), ylim=(0.40, 0.68), title="Median xám 0,485 / màu 0,510")
    axes[0].legend(fontsize=10, frameon=False)
    axes[0].grid(axis="y", color="#E4E9ED")
    rectangles = axes[1].barh(["Fixed 0,50", "Cross-fit", "Median áp lại OOF"], scores, color=[BLUE, RED, "#647383"])
    axes[1].bar_label(rectangles, labels=[f"{v:.4f}%" for v in scores], padding=6)
    axes[1].set(xlim=(0, 120), xticks=[0, 25, 50, 75, 100], xlabel="Macro-F1 (%)",
                title="Cross-fit giảm điểm so với mốc cố định")
    axes[1].invert_yaxis()
    save_figure(fig, out, "threshold_case.png", "Dò ngưỡng: phân biệt chọn ngưỡng và kiểm định",
                "Median áp lại OOF tái sử dụng nhãn đã chọn ngưỡng · Cross-fit chỉ cách ly bước chọn ngưỡng, không train lại stack.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--edge-features", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True, help="train folder containing images/")
    args = parser.parse_args()
    history = args.historical_root.resolve()
    out = ROOT / "data/negative_results"
    sources = []

    def record(path, role):
        sources.append({"role": role, "source": str(path.resolve().relative_to(history))
                        if path.resolve().is_relative_to(history) else str(path.relative_to(ROOT)),
                        "sha256": digest(path)})

    split = ROOT / "assets/splits/train_folds.csv"
    frame = pd.read_csv(split)[KEYS]
    if len(frame) != 2000 or frame.file_name.duplicated().any():
        raise ValueError("Expected 2,000 unique canonical train IDs")
    record(split, "canonical split")
    for column, run in RUNS.items():
        path = history / "results" / run / "oof.csv"
        source = pd.read_csv(path)[KEYS + ["prob"]].rename(columns={"prob": column})
        frame = checked_join(frame, source, KEYS)
        if not np.isfinite(frame[column]).all() or not frame[column].between(0, 1).all():
            raise ValueError(f"Invalid probabilities: {column}")
        record(path, column)
    features = pd.read_csv(args.edge_features)[KEYS + ["edge_ratio"]]
    frame = checked_join(frame, features, KEYS)
    if not np.isfinite(frame.edge_ratio).all():
        raise ValueError("Invalid edge features")
    record(args.edge_features, "historical edge_ratio; do not substitute canonical re-audit")
    frame["is_gray"] = [gray_flag(args.train_dir / "images" / name) for name in frame.file_name]
    image_hash = hashlib.sha256()
    for name in sorted(frame.file_name):
        image_hash.update(f"{name} {digest(args.train_dir / 'images' / name)}\n".encode())

    gate_path = history / "results/r34_native358_edgeq25_massnorm_w15_s2026/promotion_gate.json"
    calibration_path = history / "results/stack7_gray_color_crossfit_calibration/calibration_report.json"
    gates = json.loads(gate_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    thresholds = pd.DataFrame(calibration["choices"]).rename(columns={"gray": "gray_threshold", "color": "color_threshold"})
    record(gate_path, "promotion gates")
    record(calibration_path, "fold threshold choices")
    for script in ["calibrate_stack_gray_color.py", "audit_edge_subgroup_weighted_run.py"]:
        record(history / script, "historical audit logic")
    for run in dict.fromkeys(RUNS.values()):
        config = history / "results" / run / "config.json"
        if config.is_file():
            record(config, "historical run configuration")

    # Verify the original training-only cutoffs, subgroup counts and confusion receipt.
    for receipt in gates["cutoffs"]:
        fold, cutoff = receipt["fold"], receipt["cutoff"]
        observed = frame[(frame.fold != fold) & (frame.label == 1)].edge_ratio.quantile(0.25)
        if not np.isclose(observed, cutoff, atol=1e-12, rtol=0):
            raise ValueError(f"Historical edge feature does not reproduce fold {fold} cutoff")
    for col, receipt in [("edge_base_prob", gates["baseline_native"]), ("edge_weighted_prob", gates["candidate"])]:
        if any(not np.isclose(v, receipt[k], atol=1e-12, rtol=0) for k, v in counts(frame, col).items()):
            raise ValueError(f"Global gate receipt mismatch: {col}")
    calibrated_path = history / "results/stack7_gray_color_crossfit_calibration/oof.csv"
    calibrated = pd.read_csv(calibrated_path)[KEYS + ["prob"]].rename(columns={"prob": "recorded_crossfit"})
    comparison = checked_join(frame, calibrated, KEYS)
    table = thresholds.set_index("fold")
    cross = [table.loc[f, "gray_threshold" if g else "color_threshold"] for f, g in zip(frame.fold, frame.is_gray)]
    if not np.array_equal(frame.legacy_stack_prob.to_numpy() >= cross, comparison.recorded_crossfit.to_numpy()):
        raise ValueError("Gray flags/raw stack probabilities do not reproduce historical cross-fit decisions")
    record(calibrated_path, "hard decisions used only to verify cross-fit replay")

    out.mkdir(parents=True, exist_ok=True)
    columns = KEYS + list(RUNS) + ["is_gray", "edge_ratio"]
    frame[columns].to_csv(out / "oof_predictions.csv", index=False)
    thresholds[["fold", "gray_threshold", "color_threshold"]].to_csv(out / "threshold_choices.csv", index=False)
    shutil.copyfile(gate_path, out / "promotion_gates.json")
    figures(frame, thresholds, gates, out)
    outputs = {p.name: digest(p) for p in sorted(out.iterdir()) if p.suffix in {".csv", ".png"} or p.name == "promotion_gates.json"}
    manifest = {"protocol": "train-only historical CPU replay; no retraining or private evaluation",
                "row_count": len(frame), "gray_logic": GRAY_LOGIC,
                "train_image_hash_protocol": "SHA256 of sorted 'file_name SHA256(file_bytes)\\n' lines",
                "train_images_sha256": image_hash.hexdigest(), "sources": sources, "outputs_sha256": outputs}
    (out / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Packaged {len(frame)} rows; {int(frame.is_gray.sum())} gray; {len(outputs)} replay artifacts → {out}")


if __name__ == "__main__":
    main()
