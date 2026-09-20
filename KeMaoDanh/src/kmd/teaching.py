"""Mô-đun tiện ích trực quan hóa, phát hiện dữ liệu và kiểm tra ca dự đoán cho notebook giảng dạy.

Cung cấp các hàm vẽ lớp phủ vùng cắt, so sánh phóng to pixel, giải quyết đường dẫn dữ liệu
và hiển thị dự đoán Out-Of-Fold (OOF) theo đúng fold kiểm định.
"""
from pathlib import Path
import os
from typing import Optional, Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import functional as TF

from .core import PACKAGE, read_csv, read_json, metric
from .dataset_prep import discover_dataset_roots


def find_task_root() -> Path:
    """Xác định thư mục gốc của bài toán Kẻ mạo danh."""
    cwd = Path.cwd().resolve()
    for cand in [cwd, cwd.parent, cwd / 'KeMaoDanh', cwd.parent / 'KeMaoDanh']:
        if (cand / 'src/kmd').is_dir() and (cand / 'configs').is_dir():
            return cand.resolve()
    if (PACKAGE / 'src/kmd').is_dir():
        return PACKAGE.resolve()
    raise FileNotFoundError(
        "Không tìm thấy thư mục gốc KeMaoDanh chứa 'src/kmd' và 'configs'."
    )


def resolve_data_roots(task_root: Optional[Path] = None) -> Tuple[Path, Optional[Path]]:
    """Xác định đường dẫn data_root (bắt buộc) và test_root (tùy chọn) từ biến môi trường hoặc tự động phát hiện.

    Quy tắc kiểm tra:
    - DATA_ROOT là bắt buộc: nếu biến môi trường được đặt nhưng sai, hoặc không đặt và không tự động
      tìm thấy thư mục chứa pairs.csv, hàm sẽ báo lỗi rõ ràng kèm hướng dẫn khắc phục.
    - TEST_ROOT: nếu người dùng đặt biến môi trường rõ ràng mà đường dẫn không hợp lệ, hàm sẽ báo lỗi ngay;
      nếu không đặt, hệ thống cố gắng tìm trong data/ và cho phép vắng mặt (trả về None).
    """
    root = task_root or find_task_root()
    env_train = os.environ.get('DATA_ROOT')
    env_test = os.environ.get('TEST_ROOT')

    if env_train:
        train_path = Path(env_train).expanduser().resolve()
        if not train_path.is_dir() or not (train_path / 'pairs.csv').is_file():
            raise FileNotFoundError(
                f"Biến môi trường DATA_ROOT='{env_train}' không trỏ tới thư mục hợp lệ chứa 'pairs.csv'. "
                "Vui lòng kiểm tra lại đường dẫn."
            )
    else:
        discovered = discover_dataset_roots(root / 'data')
        cand = discovered.get('train')
        if cand and (cand / 'pairs.csv').is_file():
            train_path = cand
        elif (root / 'data/train/pairs.csv').is_file():
            train_path = (root / 'data/train').resolve()
        elif (root / 'data/data/train/pairs.csv').is_file():
            train_path = (root / 'data/data/train').resolve()
        else:
            raise FileNotFoundError(
                "Không tìm thấy thư mục dữ liệu huấn luyện. "
                "Vui lòng thiết lập biến môi trường DATA_ROOT trỏ đến thư mục train (chứa 'pairs.csv' và ảnh), "
                "hoặc giải nén file data.zip được cung cấp cho lớp vào thư mục 'KeMaoDanh/data/'."
            )

    test_path: Optional[Path] = None
    if env_test:
        test_path = Path(env_test).expanduser().resolve()
        if not test_path.is_dir() or not (test_path / 'pairs.csv').is_file():
            raise FileNotFoundError(
                f"Biến môi trường TEST_ROOT='{env_test}' được thiết lập nhưng không tồn tại hoặc thiếu 'pairs.csv'."
            )
    else:
        discovered = discover_dataset_roots(root / 'data')
        candidates = [
            discovered.get('public_test'),
            discovered.get('private_test'),
            root / 'data/data/private_test/private_test',
            root / 'data/private_test/private_test',
            root / 'data/test',
            root / 'data/public_test',
        ]
        for c in candidates:
            if c and (c / 'pairs.csv').is_file():
                test_path = c.resolve()
                break

    return train_path, test_path


def show_pair_images(
    row: pd.Series,
    data_root: Path | str,
    title: Optional[str] = None,
    p_pred: Optional[float] = None,
) -> None:
    """Hiển thị hai ảnh image_0 và image_1 cạnh nhau kèm nhãn thực tế và dự đoán nếu có."""
    root = Path(data_root)
    p0 = root / row['image_0']
    p1 = root / row['image_1']
    if not p0.is_file() or not p1.is_file():
        raise FileNotFoundError(f"Thiếu ảnh của cặp {row.get('pair_id')}: {p0} hoặc {p1}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    with Image.open(p0) as im0, Image.open(p1) as im1:
        axes[0].imshow(im0.convert('RGB'))
        axes[0].set_title(f"image_0 ({row['image_0']})\n{im0.size[0]}x{im0.size[1]} px | {p0.stat().st_size:,} bytes")
        axes[0].axis('off')

        axes[1].imshow(im1.convert('RGB'))
        axes[1].set_title(f"image_1 ({row['image_1']})\n{im1.size[0]}x{im1.size[1]} px | {p1.stat().st_size:,} bytes")
        axes[1].axis('off')

    header = title or f"Cặp {row.get('pair_id', '')}"
    if 'fake_position' in row and 'Nhãn thực tế:' not in header:
        header += f" | Nhãn thực tế: image_{int(row['fake_position'])} là ảnh giả"
    if p_pred is not None:
        header += f" | p(phải giả) = {p_pred:.3f}"
    fig.suptitle(header, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_crop_overlay_512(image_path: Path | str) -> None:
    """Vẽ vùng nhìn Center60 (307x307) và 4 vùng cắt Native (224x224) trên nền ảnh 512x512."""
    import matplotlib.patches as patches
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp ảnh: {path}")

    with Image.open(path) as img:
        im = img.convert('RGB')
        w, h = im.size

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(im)

    # Vùng cắt Center60: 60% của 512 = 307.2 -> 307x307 đặt ở chính giữa
    cw, ch = int(w * 0.6), int(h * 0.6)
    c_left, c_top = (w - cw) // 2, (h - ch) // 2
    rect_c60 = patches.Rectangle(
        (c_left, c_top), cw, ch,
        linewidth=2.5, edgecolor='crimson', facecolor='none',
        linestyle='--', label=f'Center60: {cw}x{ch} px (sẽ co về 224x224)'
    )
    ax.add_patch(rect_c60)

    # 4 vùng cắt Native 224x224 giữ nguyên tỉ lệ 1:1
    centers = [(0.5, 0.5), (0.38, 0.38), (0.62, 0.38), (0.5, 0.64)]
    colors = ['navy', 'forestgreen', 'darkorange', 'purple']
    labels = [
        'Vùng 1: Trung tâm (0.50, 0.50)',
        'Vùng 2: Trên trái (0.38, 0.38)',
        'Vùng 3: Trên phải (0.62, 0.38)',
        'Vùng 4: Dưới (0.50, 0.64)',
    ]

    for (cx, cy), col, lab in zip(centers, colors, labels):
        px = max(0, min(w - 224, round(cx * w) - 112))
        py = max(0, min(h - 224, round(cy * h) - 112))
        p_rect = patches.Rectangle(
            (px, py), 224, 224,
            linewidth=1.8, edgecolor=col, facecolor='none',
            label=f"{lab} [{px}:{px+224}, {py}:{py+224}]"
        )
        ax.add_patch(p_rect)

    ax.set_title("Vùng center60 và bốn crop native trên ảnh 512x512", fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_patch_zoom_comparison(
    image_path: Path | str,
    zoom_box: Tuple[int, int, int, int] = (80, 80, 144, 144),
) -> None:
    """Hiển thị patch 224x224 gốc so với patch qua nội suy (224 -> 112 -> 224), kèm ô phóng to 64x64."""
    import matplotlib.patches as patches
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp ảnh: {path}")

    with Image.open(path) as img:
        im = img.convert('RGB')
        w, h = im.size
        left = max(0, min(w - 224, round(0.5 * w) - 112))
        top = max(0, min(h - 224, round(0.5 * h) - 112))
        native_patch = im.crop((left, top, left + 224, top + 224))

    # Tái tạo hiệu ứng nội suy bilinear 224 -> 112 -> 224
    resampled_patch = TF.resize(
        TF.resize(native_patch, [112, 112], interpolation=TF.InterpolationMode.BILINEAR, antialias=True),
        [224, 224],
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True
    )

    x0, y0, x1, y1 = zoom_box
    zoom_native = native_patch.crop(zoom_box)
    zoom_resampled = resampled_patch.crop(zoom_box)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))

    axes[0, 0].imshow(native_patch)
    axes[0, 0].set_title("Vùng cắt Native (224x224 pixel gốc)")
    rect0 = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect0)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(resampled_patch)
    axes[0, 1].set_title("Vùng cắt qua nội suy (224 -> 112 -> 224 bilinear)")
    rect1 = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 1].add_patch(rect1)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(zoom_native, interpolation='nearest')
    axes[1, 0].set_title(f"Phóng to ô 64x64 pixel gốc [{x0}:{x1}, {y0}:{y1}]")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(zoom_resampled, interpolation='nearest')
    axes[1, 1].set_title(f"Phóng to ô 64x64 qua nội suy [{x0}:{x1}, {y0}:{y1}]")
    axes[1, 1].axis('off')

    fig.suptitle("Cùng vùng ảnh trước và sau nội suy", fontsize=12)
    plt.tight_layout()
    plt.show()


def view_oof_case_by_id(
    pair_id: str,
    dev_frame: pd.DataFrame,
    session_dir: Path | str,
    data_root: Path | str,
    models: Sequence[str] = ('b2', 'native'),
    seed: int = 20260917,
    smoke: bool = False,
    historical_note: Optional[str] = None,
) -> None:
    """Hiển thị dự đoán Out-Of-Fold thực tế cho một cặp ID cụ thể từ các mô hình hợp lệ.

    Quy tắc kiểm định chặt chẽ:
    - Không tự ý fallback giữa chế độ smoke và chế độ full, hoặc giữa các seed khác nhau.
    - Xác thực độ bao phủ đầy đủ và tính nhất quán metadata của fold trước khi hiển thị.
    - Chỉ hiển thị các mô hình thực sự có tệp OOF hợp lệ trong phiên làm việc hiện tại.
    - Không gắn nhãn kết quả lịch sử chủ quan lên dự đoán của lượt chạy mới mà không có chú thích nguồn.
    """
    session = Path(session_dir)
    target_id = str(pair_id).zfill(5)

    matching = dev_frame.loc[dev_frame.pair_id == target_id]
    if matching.empty:
        raise ValueError(f"Không tìm thấy cặp {target_id} trong tập development.")

    row = matching.iloc[0]
    fold = int(row['inner_fold'])
    true_label = int(row['fake_position'])

    from .pipeline import compare_oof, aligned_predictions
    _, all_predictions = compare_oof(session, models, seed=seed, smoke=smoke)
    preds = {}
    for name, prediction in all_predictions.items():
        aligned = aligned_predictions(prediction, dev_frame)
        preds[name] = float(aligned.loc[aligned.pair_id == target_id, 'p'].item())

    caption = f"Cặp {target_id} (Fold kiểm định: {fold}) | Nhãn thực tế: image_{true_label} là ảnh giả"
    if historical_note:
        print(f"Ghi chú báo cáo: {historical_note}")

    if not preds:
        caption += "\n(Chưa có tệp dự đoán OOF phù hợp cho cặp này trong phiên hiện tại)"
    else:
        caption += "\n[Dự đoán lượt chạy hiện tại]:"
        for m, p in preds.items():
            chosen = int(p >= 0.5)
            status = "Đúng" if chosen == true_label else "Sai"
            caption += f"\n- {m}: p(phải giả)={p:.3f} -> Chọn vị trí {chosen} ({status})"

    if smoke:
        caption = "[SMOKE] " + caption
    show_pair_images(row, data_root, title=caption)
