"""Public reusable suite API for systematic ablation and hypothesis testing.

Provides organized suites matching the experimental sections of the report:
- lr_ablation: All 9 LR probes (groups 0, 1:19, 19:28, 28:32, single & drop)
- candidate_seeds: DenseNet-121 224 (A_full_batch24), DenseNet-288, B2-288 across 3 seeds + Blend gate evaluation
- crop_comparison: Native vs Resampled (partial & full 1:1 crops)
- loss_comparison: Image, Pair, Mixed, Repair loss variants (19-epoch terminal)
- backbone_residual: ResNet-18 with RGB, Gaussian residual, NPR residual
- data_amount: 25%, 50%, 100% training subsets under fixed update budget
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Sequence
import pandas as pd

from .core import read_csv, metric
from .extractor import list_lr_ablations, NAMES
from .presets import canonical_preset_name
from .gate import evaluate_blend_gate


def run_lr_suite(
    frame: pd.DataFrame,
    root: Path | str,
    session: Path | str,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Execute all 9 LR ablations without overwriting, sharing a single pre-extracted feature table."""
    from .pipeline import fit_lr_cv, load_session, aligned_predictions, extract_pair_features
    session = load_session(session, data_root=root)
    dev_frame = frame

    # Extract full 32 features ONCE and save features.csv to avoid repeated disk passes
    x_all, table = extract_pair_features(dev_frame, root, feature_columns=NAMES)
    table.to_csv(session / 'features.csv')

    ablations = list_lr_ablations()
    results = []
    predictions = {}

    for name in ablations:
        pred = fit_lr_cv(dev_frame, root, session, variant=name, precomputed_features=table)
        predictions[f"lr_{name}"] = pred
        aligned = aligned_predictions(pred, dev_frame)
        scores = metric(dev_frame.fake_position, aligned.p)
        results.append({"model": f"lr_{name}", "variant": name, **scores})

    table_df = pd.DataFrame(results)
    table_df.to_csv(session / "lr_ablation_comparison.csv", index=False)
    return table_df, predictions


def run_candidate_seeds_suite(
    frame: pd.DataFrame,
    root: Path | str,
    session: Path | str,
    seeds: Sequence[int] = (20260917, 20260918, 20260919),
    smoke: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train center60 224 (DenseNet-121 batch24), DenseNet-288, and B2-288 across 3 seeds and evaluate Blend gate."""
    from .pipeline import train_cnn_cv, blend, load_session, aligned_predictions
    session = load_session(session, data_root=root)
    results = []
    predictions = {}

    # Candidate architectures to evaluate across all 3 seeds:
    # center60 (DenseNet121 224 A_full_batch24), dense288 (DenseNet121 288), b2 (EfficientNet-B2 288)
    candidate_archs = ("center60", "dense288", "b2")

    for seed in seeds:
        for arch in candidate_archs:
            oof = train_cnn_cv(arch, frame, root, session, folds=(0, 1, 2), seed=seed, smoke=smoke)
            model_key = f"{arch}_smoke_s{seed}" if smoke else f"{arch}_s{seed}"
            predictions[model_key] = oof
            aligned = aligned_predictions(oof, frame)
            scores = metric(frame.fake_position, aligned.p)
            results.append({"model": model_key, "arch": arch, "seed": seed, "smoke": smoke, **scores})

        # Train native for the Blend gate with B2
        nat_oof = train_cnn_cv("native", frame, root, session, folds=(0, 1, 2), seed=seed, smoke=smoke)
        nat_key = f"native_smoke_s{seed}" if smoke else f"native_s{seed}"
        predictions[nat_key] = nat_oof
        aligned_nat = aligned_predictions(nat_oof, frame)
        scores_nat = metric(frame.fake_position, aligned_nat.p)
        results.append({"model": nat_key, "arch": "native", "seed": seed, "smoke": smoke, **scores_nat})

        # Evaluate Blend of B2 and Native
        b2_key = f"b2_smoke_s{seed}" if smoke else f"b2_s{seed}"
        b2_oof = predictions.get(b2_key)
        blend_oof = blend(b2_oof, nat_oof, frame)
        blend_key = f"blend_smoke_s{seed}" if smoke else f"blend_s{seed}"
        predictions[blend_key] = blend_oof
        blend_oof_name = f"blend_smoke_s{seed}_oof.csv" if smoke else f"blend_s{seed}_oof.csv"
        blend_oof.to_csv(session / blend_oof_name, index=False)
        if not smoke and seed == 20260917:
            blend_oof.to_csv(session / "blend_oof.csv", index=False)

        aligned_bld = aligned_predictions(blend_oof, frame)
        scores_bld = metric(frame.fake_position, aligned_bld.p)
        results.append({"model": blend_key, "arch": "blend", "seed": seed, "smoke": smoke, **scores_bld})

    table = pd.DataFrame(results)
    table_name = "candidate_seeds_smoke_comparison.csv" if smoke else "candidate_seeds_comparison.csv"
    table.to_csv(session / table_name, index=False)

    gate = evaluate_blend_gate(session, seeds=seeds, is_smoke=smoke, dev_frame=frame)
    return table, gate


def run_crop_comparison_suite(
    frame: pd.DataFrame,
    root: Path | str,
    session: Path | str,
    smoke: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compare Native vs Resampled crops in both partial and full fine-tune modes."""
    from .pipeline import train_cnn_cv, load_session, aligned_predictions
    session = load_session(session, data_root=root)
    models = ["partial_native", "partial_resampled", "native", "resampled", "top2"]
    results = []
    predictions = {}

    for name in models:
        oof = train_cnn_cv(name, frame, root, session, folds=(0, 1, 2), smoke=smoke)
        model_key = f"{name}_smoke" if smoke else name
        predictions[model_key] = oof
        aligned = aligned_predictions(oof, frame)
        scores = metric(frame.fake_position, aligned.p)
        results.append({"model": model_key, "name": name, "smoke": smoke, **scores})

    table = pd.DataFrame(results)
    table_name = "crop_comparison_smoke.csv" if smoke else "crop_comparison.csv"
    table.to_csv(session / table_name, index=False)
    return table, predictions


def run_loss_comparison_suite(
    frame: pd.DataFrame,
    root: Path | str,
    session: Path | str,
    smoke: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compare image, pairwise, mixed, and re-paired loss objectives at 19 epochs terminal."""
    from .pipeline import train_cnn_cv, load_session, aligned_predictions
    session = load_session(session, data_root=root)
    models = ["loss_image", "loss_pair", "loss_mixed", "repair"]
    results = []
    predictions = {}

    for name in models:
        oof = train_cnn_cv(name, frame, root, session, folds=(0, 1, 2), smoke=smoke)
        model_key = f"{name}_smoke" if smoke else name
        predictions[model_key] = oof
        aligned = aligned_predictions(oof, frame)
        scores = metric(frame.fake_position, aligned.p)
        results.append({"model": model_key, "name": name, "smoke": smoke, **scores})

    table = pd.DataFrame(results)
    table_name = "loss_comparison_smoke.csv" if smoke else "loss_comparison.csv"
    table.to_csv(session / table_name, index=False)
    return table, predictions


def run_residual_suite(
    frame: pd.DataFrame,
    root: Path | str,
    session: Path | str,
    smoke: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compare ResNet-18 with RGB, Gaussian residual, and NPR nearest-neighbor residual."""
    from .pipeline import train_cnn_cv, load_session, aligned_predictions
    session = load_session(session, data_root=root)
    models = ["resnet18_rgb", "resnet18_gaussian", "resnet18_npr"]
    results = []
    predictions = {}

    for name in models:
        oof = train_cnn_cv(name, frame, root, session, folds=(0, 1, 2), smoke=smoke)
        model_key = f"{name}_smoke" if smoke else name
        predictions[model_key] = oof
        aligned = aligned_predictions(oof, frame)
        scores = metric(frame.fake_position, aligned.p)
        results.append({"model": model_key, "name": name, "smoke": smoke, **scores})

    table = pd.DataFrame(results)
    table_name = "residual_comparison_smoke.csv" if smoke else "residual_comparison.csv"
    table.to_csv(session / table_name, index=False)
    return table, predictions


def run_data_amount_suite(
    frame: pd.DataFrame,
    root: Path | str,
    session: Path | str,
    smoke: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compare 25%, 50%, and 100% training subsets with equal 391 updates (or 4 in smoke)."""
    from .pipeline import train_cnn_cv, load_session, aligned_predictions
    session = load_session(session, data_root=root)
    models = ["data_25", "data_50", "data_100"]
    results = []
    predictions = {}

    for name in models:
        oof = train_cnn_cv(name, frame, root, session, folds=(0, 1, 2), smoke=smoke)
        model_key = f"{name}_smoke" if smoke else name
        predictions[model_key] = oof
        aligned = aligned_predictions(oof, frame)
        scores = metric(frame.fake_position, aligned.p)
        results.append({"model": model_key, "name": name, "smoke": smoke, **scores})

    table = pd.DataFrame(results)
    table_name = "data_amount_comparison_smoke.csv" if smoke else "data_amount_comparison.csv"
    table.to_csv(session / table_name, index=False)
    return table, predictions


def run_suite(
    suite_name: str,
    frame: pd.DataFrame,
    root: Path | str,
    session: Path | str,
    smoke: bool = False,
) -> Any:
    """Public entrypoint for executing any registered hypothesis testing suite."""
    s = str(suite_name).strip().lower()
    if s in ("lr", "lr_ablation", "lr_ablations"):
        return run_lr_suite(frame, root, session)
    if s in ("seeds", "candidate_seeds", "three_seeds"):
        return run_candidate_seeds_suite(frame, root, session, smoke=smoke)
    if s in ("crops", "crop_comparison", "native_resampled"):
        return run_crop_comparison_suite(frame, root, session, smoke=smoke)
    if s in ("loss", "loss_comparison", "objectives"):
        return run_loss_comparison_suite(frame, root, session, smoke=smoke)
    if s in ("residual", "residuals", "backbone_residual", "resnet18"):
        return run_residual_suite(frame, root, session, smoke=smoke)
    if s in ("data", "data_amount", "fractions", "subsets"):
        return run_data_amount_suite(frame, root, session, smoke=smoke)
    raise ValueError(
        f"Unknown suite '{suite_name}'. Supported suites: "
        "'lr_ablation', 'candidate_seeds', 'crop_comparison', 'loss_comparison', "
        "'backbone_residual', 'data_amount'"
    )
