"""Train on actual development images, infer unlabeled test, and export CSV.

Supports profiles:
  - baseline: 3-fold Logistic Regression with fresh 32 statistical features.
  - full: 3-fold LR + 5 CNN architectures (frozen, center60, native, resampled, b2) + 50/50 blend.
"""
import argparse
from pathlib import Path
import sys
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kmd.core import PACKAGE, write_json, export_submission, metric
from kmd.pipeline import (prepare_development, start_session, fit_lr_cv, train_cnn_cv,
                          compare_oof, blend, load_pairs, check_test_separation,
                          infer_cnn, infer_lr, validate_run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-root', type=Path, required=True, help='Path to train directory containing pairs.csv')
    parser.add_argument('--test-root', type=Path, default=None, help='Optional path to test directory containing pairs.csv')
    parser.add_argument('--profile', choices=['baseline', 'full'], default='baseline',
                        help="Execution profile: 'baseline' (LR only, fast CPU/GPU) or 'full' (LR + 5 CNNs x 3 folds)")
    parser.add_argument('--session-id', type=str, default='lesson_session', help='Explicit session directory name')
    args = parser.parse_args()

    session_id = validate_run_id(args.session_id)

    if args.profile == 'full' and not torch.cuda.is_available():
        raise RuntimeError("Profile 'full' requires CUDA GPU acceleration. Use --profile baseline for fast CPU execution.")

    train_root = Path(args.train_root).expanduser().resolve()
    if not train_root.is_dir() or not (train_root / 'pairs.csv').is_file():
        raise FileNotFoundError(f"Missing train dataset directory or pairs.csv at {train_root}")

    frame = prepare_development(train_root)
    session = start_session(frame, train_root, run_id=session_id)
    print(f"Session directory: {session.relative_to(PACKAGE)}")

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

    # Baseline step: 3-fold Logistic Regression
    print("Running 3-fold Cross-Validation for Logistic Regression (32 features)...")
    fit_lr_cv(frame, train_root, session)

    models_to_evaluate = ['lr']

    if args.profile == 'full':
        cnn_names = ['frozen', 'center60', 'native', 'resampled', 'b2']
        for name in cnn_names:
            print(f"Training 3-fold CNN: {name}...")
            train_cnn_cv(name, frame, train_root, session)
        models_to_evaluate += cnn_names

    comparison, oof = compare_oof(session, models_to_evaluate)

    # Compute fixed 50/50 blend in full profile on development OOF
    if args.profile == 'full':
        blend_oof = blend(oof['b2'], oof['native'], frame)
        blend_oof.to_csv(session / 'blend_oof.csv', index=False)
        score_blend = metric(frame.fake_position, blend_oof.p)
        blend_row = pd.DataFrame([{'model': 'blend_b2_native', **score_blend}])
        comparison = pd.concat([comparison, blend_row], ignore_index=True)

    comparison.to_csv(session / 'comparison.csv', index=False)
    print("\n=== Development Out-Of-Fold Evaluation ===")
    print(comparison.to_string(index=False))

    if has_test:
        output_dir = PACKAGE / 'outputs' / session.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.profile == 'full':
            print("Inferring test set using 50% B2 + 50% Native blend...")
            pred_b2 = infer_cnn('b2', test, test_root, session)
            pred_native = infer_cnn('native', test, test_root, session)
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
