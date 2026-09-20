"""Decision gate for blending vs single backbone selection across multiple seeds.

According to report Table 14:
- Evaluates Macro-F1 across 3 unique seeds (20260917, 20260918, 20260919)
- Calculates mean unrounded Macro-F1 delta: mean(F1_blend - F1_b2)
- Requires mean unrounded delta >= +0.005 (+0.50 pp) to accept Blend complexity
- Only renders an official decision on complete non-smoke runs with verified provenance
- Default fallback remains B2 (retaining single model simplicity)
- When smoke or seeds missing, returns decision=None
"""
from pathlib import Path
from typing import Dict, Any, Sequence, Optional
import numpy as np
import pandas as pd

from .core import read_csv, read_json, metric, sha256
from .presets import get_preset_config
from .core import split_fold
from .pipeline import aligned_predictions, validate_fold_checkpoint


EXPECTED_SEEDS = (20260917, 20260918, 20260919)


def summarize_deltas(deltas, threshold=0.005):
    """Apply the predeclared threshold to the unrounded mean of three seed deltas."""
    values = np.asarray(deltas, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Need exactly three finite seed deltas.')
    mean = float(values.mean())
    return mean, mean >= threshold


def compute_seed_delta(
    b2_oof: pd.DataFrame,
    native_oof: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute unrounded Macro-F1 delta and error reductions for one seed using aligned predictions."""
    b2_aligned = aligned_predictions(b2_oof, ground_truth)
    nat_aligned = aligned_predictions(native_oof, ground_truth)

    y = b2_aligned["fake_position"].to_numpy().astype(int)
    p_b2 = b2_aligned["p"].to_numpy().astype(float)
    p_nat = nat_aligned["p"].to_numpy().astype(float)
    p_blend = 0.5 * p_b2 + 0.5 * p_nat

    metric_b2 = metric(y, p_b2)
    metric_blend = metric(y, p_blend)

    # Keep full unrounded floating point precision
    delta_f1 = float(metric_blend["macro_f1"] - metric_b2["macro_f1"])
    error_reduction = int(metric_b2["errors"] - metric_blend["errors"])

    return {
        "b2_macro_f1": float(metric_b2["macro_f1"]),
        "blend_macro_f1": float(metric_blend["macro_f1"]),
        "delta_f1": delta_f1,
        "delta_pp": delta_f1 * 100.0,
        "b2_errors": int(metric_b2["errors"]),
        "blend_errors": int(metric_blend["errors"]),
        "error_reduction": error_reduction,
    }


def evaluate_blend_gate(
    session_dir: Path | str,
    seeds: Sequence[int] = EXPECTED_SEEDS,
    threshold: float = 0.005,
    is_smoke: bool = False,
    dev_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Evaluate whether Blend improves over B2 by at least threshold across all 3 seeds.

    Strictly fails closed: requires verified non-smoke provenance with 3 folds [0, 1, 2]
    and validated checkpoints for all 3 seeds. Cannot decide from loose CSVs.
    Returns decision=None for smoke runs, invalid seeds, or missing/unverified seeds.
    Threshold is in unrounded units (+0.005 = +0.50 pp).
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Gate threshold must be finite and nonnegative.")
    session = Path(session_dir)

    # Validate exactly 3 unique expected seeds
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(seeds) != set(EXPECTED_SEEDS):
        return {
            "decision": None,
            "status": "invalid_seeds",
            "threshold": threshold,
            "mean_delta_unrounded": None,
            "mean_delta_percent": None,
            "gate_passed": False,
            "complete": False,
            "reason": f"Expected exactly 3 unique candidate seeds {list(EXPECTED_SEEDS)}, got {list(seeds)}.",
            "seed_results": [],
            "missing_seeds": list(set(EXPECTED_SEEDS) - set(seeds)),
        }

    if is_smoke:
        return {
            "decision": None,
            "status": "smoke_run_no_conclusion",
            "threshold": threshold,
            "mean_delta_unrounded": None,
            "mean_delta_percent": None,
            "gate_passed": False,
            "complete": False,
            "reason": "Smoke execution mode active. Decisions require complete non-smoke runs.",
            "seed_results": [],
            "missing_seeds": [],
        }

    if dev_frame is None:
        dev_file = session / "development.csv"
        if not dev_file.is_file():
            raise FileNotFoundError(f"Missing development frame at {dev_file}")
        dev_frame = read_csv(dev_file)

    if len(dev_frame) != 800 or not dev_frame["pair_id"].is_unique:
        raise ValueError("development.csv does not contain the canonical 800 unique pairs.")
    for req_col in ['pair_id', 'image_0', 'image_1', 'fake_position', 'inner_fold']:
        if req_col not in dev_frame:
            raise ValueError(f"development.csv missing required metadata column '{req_col}'")

    seed_results = []
    missing_seeds = []
    deltas = []

    for seed in seeds:
        seed_valid = True
        arch_oofs = {}

        for arch in ("b2", "native"):
            oof_candidates = [
                session / f"{arch}_s{seed}_oof.csv",
            ]
            if seed == 20260917:
                oof_candidates.append(session / f"{arch}_oof.csv")
            oof_file = next((p for p in oof_candidates if p.is_file()), None)

            if oof_file is None:
                seed_valid = False
                break

            # Mandatory model index for non-smoke provenance verification
            idx_candidates = [
                session / f"{arch}_s{seed}_models.json",
            ]
            if seed == 20260917:
                idx_candidates.append(session / f"{arch}_models.json")
            idx_file = next((p for p in idx_candidates if p.is_file()), None)

            if idx_file is None:
                return {
                    "decision": None,
                    "status": "missing_provenance",
                    "threshold": threshold,
                    "mean_delta_unrounded": None,
                    "mean_delta_percent": None,
                    "gate_passed": False,
                    "complete": False,
                    "reason": f"Missing mandatory model index for {arch} seed {seed}. Decisions cannot be rendered from loose CSVs.",
                    "seed_results": [],
                    "missing_seeds": [seed],
                }

            records = read_json(idx_file)
            if sorted(r.get("fold") for r in records) != [0, 1, 2]:
                return {
                    "decision": None,
                    "status": "incomplete_fold_records",
                    "threshold": threshold,
                    "mean_delta_unrounded": None,
                    "mean_delta_percent": None,
                    "gate_passed": False,
                    "complete": False,
                    "reason": f"Model index for {arch} seed {seed} does not contain complete folds [0, 1, 2].",
                    "seed_results": [],
                    "missing_seeds": [seed],
                }

            oof_df = read_csv(oof_file)
            try:
                aligned_oof = aligned_predictions(oof_df, dev_frame)
            except Exception as e:
                return {
                    "decision": None,
                    "status": "corrupt_oof_metadata",
                    "threshold": threshold,
                    "mean_delta_unrounded": None,
                    "mean_delta_percent": None,
                    "gate_passed": False,
                    "complete": False,
                    "reason": f"OOF predictions for {arch} seed {seed} failed alignment: {e}",
                    "seed_results": [],
                    "missing_seeds": [seed],
                }

            # Validate each fold checkpoint using validate_fold_checkpoint
            for rec in records:
                fold = rec.get("fold")
                ckpt_rel = rec.get("checkpoint")
                if not ckpt_rel:
                    return {
                        "decision": None,
                        "status": "corrupt_model_record",
                        "threshold": threshold,
                        "mean_delta_unrounded": None,
                        "mean_delta_percent": None,
                        "gate_passed": False,
                        "complete": False,
                        "reason": f"Record for {arch} seed {seed} fold {fold} missing checkpoint path.",
                        "seed_results": [],
                        "missing_seeds": [seed],
                    }
                ckpt_path = session / ckpt_rel
                if not ckpt_path.is_file():
                    return {
                        "decision": None,
                        "status": "missing_checkpoint_file",
                        "threshold": threshold,
                        "mean_delta_unrounded": None,
                        "mean_delta_percent": None,
                        "gate_passed": False,
                        "complete": False,
                        "reason": f"Checkpoint file missing: {ckpt_path}",
                        "seed_results": [],
                        "missing_seeds": [seed],
                    }
                if sha256(ckpt_path) != rec.get("sha256"):
                    return {
                        "decision": None,
                        "status": "checkpoint_hash_mismatch",
                        "threshold": threshold,
                        "mean_delta_unrounded": None,
                        "mean_delta_percent": None,
                        "gate_passed": False,
                        "complete": False,
                        "reason": f"Checkpoint SHA256 mismatch for {arch} seed {seed} fold {fold}.",
                        "seed_results": [],
                        "missing_seeds": [seed],
                    }

                folder = ckpt_path.parent
                c_fold = get_preset_config(arch, seed=seed, fold=fold, smoke=False)
                tr, va = split_fold(dev_frame, fold)
                is_valid, metrics = validate_fold_checkpoint(
                    folder, c_fold, tr, va, session, purpose="teaching_full_fold"
                )
                if not is_valid:
                    return {
                        "decision": None,
                        "status": "checkpoint_validation_failed",
                        "threshold": threshold,
                        "mean_delta_unrounded": None,
                        "mean_delta_percent": None,
                        "gate_passed": False,
                        "complete": False,
                        "reason": f"Checkpoint validation failed for {arch} seed {seed} fold {fold}: {metrics}",
                        "seed_results": [],
                        "missing_seeds": [seed],
                    }

                # Verify matching per-fold saved predictions against assembled OOF
                dev_fold_file = folder / "development.csv"
                dev_fold_df = read_csv(dev_fold_file)
                fold_oof_rows = aligned_oof[aligned_oof.inner_fold == fold].sort_values("pair_id")
                dev_fold_sorted = dev_fold_df.sort_values("pair_id")
                if not np.allclose(fold_oof_rows.p.to_numpy(float), dev_fold_sorted.p.to_numpy(float), atol=1e-6):
                    return {
                        "decision": None,
                        "status": "prediction_mismatch",
                        "threshold": threshold,
                        "mean_delta_unrounded": None,
                        "mean_delta_percent": None,
                        "gate_passed": False,
                        "complete": False,
                        "reason": f"Per-fold predictions mismatch assembled OOF for {arch} seed {seed} fold {fold}.",
                        "seed_results": [],
                        "missing_seeds": [seed],
                    }

            arch_oofs[arch] = aligned_oof

        if not seed_valid:
            missing_seeds.append(seed)
            continue

        res = compute_seed_delta(arch_oofs["b2"], arch_oofs["native"], dev_frame)
        res["seed"] = seed
        seed_results.append(res)
        deltas.append(res["delta_f1"])

    complete = (len(missing_seeds) == 0 and len(seed_results) == len(seeds))
    if not complete:
        mean_d = float(np.mean(deltas)) if deltas else None
        return {
            "decision": None,
            "status": "incomplete_seeds",
            "threshold": threshold,
            "mean_delta_unrounded": mean_d,
            "mean_delta_percent": (mean_d * 100.0) if mean_d is not None else None,
            "gate_passed": False,
            "complete": False,
            "reason": f"Incomplete seeds: required {list(seeds)}, missing {missing_seeds}.",
            "seed_results": seed_results,
            "missing_seeds": missing_seeds,
        }

    # Mean unrounded delta across all 3 evaluated seeds
    mean_delta, gate_passed = summarize_deltas(deltas, threshold)
    mean_delta_percent = mean_delta * 100.0
    decision = "blend" if gate_passed else "b2"

    reason = (
        f"Mean 3-seed unrounded delta is {mean_delta:.6f} ({mean_delta_percent:+.4f} pp). "
        + (f"Meets threshold of {threshold:.4f} (+{threshold * 100:.2f} pp). Selected Blend."
           if gate_passed
           else f"Below threshold of {threshold:.4f} (+{threshold * 100:.2f} pp). Retained B2.")
    )

    return {
        "decision": decision,
        "status": "passed" if gate_passed else "retained_single",
        "threshold": threshold,
        "mean_delta_unrounded": mean_delta,
        "mean_delta_percent": mean_delta_percent,
        "gate_passed": gate_passed,
        "complete": True,
        "reason": reason,
        "seed_results": seed_results,
        "missing_seeds": [],
    }
