"""Train on actual development images, infer unlabeled test, and export CSV.

Supports profiles:
  - baseline: 3-fold Logistic Regression with fresh 32 statistical features.
  - full: 3-fold LR + 5 CNN architectures (frozen, center60, native, resampled, b2) with default final B2 inference.

Supports explicit --final-method:
  - lr: 3-fold Logistic Regression
  - b2: 3-fold EfficientNet-B2 (default for profile=full per report decision)
  - blend: 50/50 probability blend of B2 and Native

Supports --smoke:
  - max 64 train pairs/fold stratified, 2 epochs, warmup 1
  - full 800 dev validation coverage preserved across 3 folds
  - smoke artifacts isolated, no definitive decision conclusions drawn
"""
import argparse
import os
from pathlib import Path
import sys
import json
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kmd.core import PACKAGE, write_json, export_submission, metric
from kmd.pipeline import (
    prepare_development, start_session, fit_lr_cv, train_cnn_cv,
    compare_oof, blend, load_pairs, check_test_separation,
    infer_cnn, infer_lr, validate_run_id,
)
from kmd.gate import evaluate_blend_gate
from kmd.suite import run_suite


def main():
    default_train = os.environ.get('DATA_ROOT')
    default_test = os.environ.get('TEST_ROOT')

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--train-root',
        type=Path,
        default=Path(default_train) if default_train else None,
        required=default_train is None,
        help='Path to train directory containing pairs.csv (or set DATA_ROOT env var)',
    )
    parser.add_argument(
        '--test-root',
        type=Path,
        default=Path(default_test) if default_test else None,
        help='Optional path to test directory containing pairs.csv (or set TEST_ROOT env var)',
    )
    parser.add_argument(
        '--profile',
        choices=['baseline', 'full'],
        default='baseline',
        help="Execution profile: 'baseline' (LR only, fast CPU/GPU) or 'full' (LR + 5 CNNs x 3 folds)",
    )
    parser.add_argument(
        '--final-method',
        choices=['lr', 'b2', 'blend'],
        default=None,
        help="Explicit final inference method: 'lr', 'b2' (default for full), or 'blend'.",
    )
    parser.add_argument(
        '--session-id',
        type=str,
        default='lesson_session',
        help='Explicit session directory name (default: lesson_session)',
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Run fast smoke verification (max 64 train pairs/fold, 2 epochs, full validation)',
    )
    parser.add_argument(
        '--suite',
        choices=['lr_ablation', 'candidate_seeds', 'crop_comparison', 'loss_comparison', 'backbone_residual', 'data_amount'],
        default=None,
        help='Execute an optional systematic hypothesis testing suite',
    )
    parser.add_argument(
        '--gate',
        action='store_true',
        help='Evaluate 3-seed Blend selection gate (requires 3 seeds of B2 and Native)',
    )
    args = parser.parse_args()

    # Determine final inference method: baseline defaults to lr, full defaults to b2
    if args.final_method is not None:
        final_method = args.final_method
    else:
        final_method = 'lr' if args.profile == 'baseline' else 'b2'

    session_id = validate_run_id(args.session_id)
    if args.smoke and session_id == 'lesson_session':
        session_id = 'smoke_session'

    if args.profile == 'full' and not torch.cuda.is_available():
        raise RuntimeError("Profile 'full' requires CUDA GPU acceleration. Use --profile baseline for fast CPU execution.")

    train_root = Path(args.train_root).expanduser().resolve()
    if not train_root.is_dir() or not (train_root / 'pairs.csv').is_file():
        raise FileNotFoundError(f"Missing train dataset directory or pairs.csv at {train_root}")

    frame = prepare_development(train_root)
    session = start_session(frame, train_root, run_id=session_id)
    print(f"Session directory: {session.relative_to(PACKAGE)}")
    if args.smoke:
        print("Running in unified SMOKE mode: max 64 train pairs/fold, 2 epochs, full 800 dev validation.")

    has_test = False
    test = None
    if args.test_root is not None:
        test_root = Path(args.test_root).expanduser().resolve()
        if not test_root.is_dir() or not (test_root / 'pairs.csv').is_file():
            raise FileNotFoundError(f"Specified test directory does not contain pairs.csv: {test_root}")
        test = load_pairs(test_root, labeled=False)
        check_test_separation(test, test_root, session)
        has_test = True
        print(f"Loaded {len(test)} test pairs from {test_root}")

    # Optional hypothesis suite execution
    if args.suite is not None:
        print(f"\n=== Executing Experimental Suite: {args.suite} ===")
        suite_table, _ = run_suite(args.suite, frame, train_root, session, smoke=args.smoke)
        if isinstance(suite_table, pd.DataFrame):
            print(suite_table.to_string(index=False))

    # Baseline step: 3-fold Logistic Regression
    print("\nRunning 3-fold Cross-Validation for Logistic Regression (32 features)...")
    fit_lr_cv(frame, train_root, session)

    models_to_evaluate = ['lr']

    if args.profile == 'full':
        cnn_names = ['frozen', 'center60', 'native', 'resampled', 'b2']
        for name in cnn_names:
            print(f"Training 3-fold CNN: {name} (smoke={args.smoke})...")
            train_cnn_cv(name, frame, train_root, session, smoke=args.smoke)
        models_to_evaluate += cnn_names

    comparison, oof = compare_oof(session, models_to_evaluate, smoke=args.smoke)

    # Compute candidate 50/50 blend in full profile for comparison
    if args.profile == 'full' and 'b2' in oof and 'native' in oof:
        blend_oof = blend(oof['b2'], oof['native'], frame)
        blend_oof_name = 'blend_smoke_s20260917_oof.csv' if args.smoke else 'blend_oof.csv'
        blend_oof.to_csv(session / blend_oof_name, index=False)
        score_blend = metric(frame.fake_position, blend_oof.p)
        blend_row = pd.DataFrame([{'model': 'blend_b2_native', **score_blend}])
        comparison = pd.concat([comparison, blend_row], ignore_index=True)

    comp_name = 'comparison_smoke.csv' if args.smoke else 'comparison.csv'
    comparison.to_csv(session / comp_name, index=False)
    print("\n=== Development Out-Of-Fold Evaluation ===")
    print(comparison.to_string(index=False))

    # Blend gate evaluation if requested or profile=full
    if args.gate or args.profile == 'full':
        gate_res = evaluate_blend_gate(session, is_smoke=args.smoke, dev_frame=frame)
        print("\n=== Blend Selection Gate Evaluation ===")
        dec = gate_res.get('decision')
        print(f"- Decision: {dec.upper() if dec is not None else 'NONE (No Conclusion)'}")
        print(f"- Status: {gate_res.get('status')}")
        print(f"- Reason: {gate_res.get('reason')}")
        mean_d = gate_res.get('mean_delta_unrounded')
        mean_pct = gate_res.get('mean_delta_percent')
        if mean_d is not None and mean_pct is not None:
            print(f"- Mean Unrounded Delta: {mean_d:.6f} ({mean_pct:+.4f} pp)")
        gate_file = "blend_gate_smoke.json" if args.smoke else "blend_gate.json"
        write_json(session / gate_file, gate_res)

    print(f"\nFinal method selected for export/inference: {final_method.upper()}")

    if has_test:
        output_dir = PACKAGE / 'outputs' / session.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if final_method == 'b2':
            print("Inferring test set using 3-fold EfficientNet-B2 (default report method)...")
            prediction = infer_cnn('b2', test, test_root, session, smoke=args.smoke)
        elif final_method == 'blend':
            print("Inferring test set using candidate 50% B2 + 50% Native blend...")
            pred_b2 = infer_cnn('b2', test, test_root, session, smoke=args.smoke)
            pred_native = infer_cnn('native', test, test_root, session, smoke=args.smoke)
            prediction = blend(pred_b2, pred_native, test)
        else:
            print("Inferring test set using 3-fold Logistic Regression...")
            prediction = infer_lr(test, test_root, session)

        prediction.to_csv(output_dir / 'test_probabilities.csv', index=False)
        sub_path = output_dir / 'submission.csv'
        export_submission(prediction, sub_path)
        print(f"\nSaved submission to: {sub_path.relative_to(PACKAGE)}")
    else:
        print("\nNotice: Missing or unprovided test-root. Development training/evaluation completed successfully.")
        print("Status: no submission produced (test root not provided).")


if __name__ == '__main__':
    main()
