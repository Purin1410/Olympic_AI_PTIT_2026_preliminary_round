"""Real-data teaching and experimentation pipeline for Kẻ mạo danh."""
from .config import Config
from .core import PACKAGE, metric, pair_table, export_submission, split_fold, read_csv, write_json, read_json
from .pipeline import (
    Runtime, prepare_development, start_session, load_session,
    extract_pair_features, fit_lr_fold, fit_lr_cv, train_cnn_fold,
    train_cnn_cv, complete_method_cv, compare_oof, blend,
    infer_lr, infer_cnn, load_pairs, check_test_separation,
    fold_run_name, find_fold_folder, validate_fold_checkpoint,
)
from .presets import get_preset_config, list_registered_presets, is_registered_preset, canonical_preset_name
from .extractor import NAMES, features, get_lr_features, list_lr_ablations, resolve_lr_ablation_name
from .gate import evaluate_blend_gate
from .suite import run_suite, run_lr_suite
from .dataset_prep import extract_official_zip, discover_dataset_roots, validate_official_layout
from .subsets import sample_train_subset, sample_smoke_train

__all__ = [
    'Config', 'Runtime', 'PACKAGE', 'metric', 'pair_table', 'export_submission', 'split_fold',
    'prepare_development', 'start_session', 'load_session', 'extract_pair_features',
    'fit_lr_fold', 'fit_lr_cv', 'train_cnn_fold', 'train_cnn_cv', 'complete_method_cv',
    'compare_oof', 'blend', 'infer_lr', 'infer_cnn', 'load_pairs', 'check_test_separation',
    'fold_run_name', 'find_fold_folder', 'validate_fold_checkpoint',
    'get_preset_config', 'list_registered_presets', 'is_registered_preset', 'canonical_preset_name',
    'NAMES', 'features', 'get_lr_features', 'list_lr_ablations', 'resolve_lr_ablation_name',
    'evaluate_blend_gate', 'run_suite', 'run_lr_suite',
    'extract_official_zip', 'discover_dataset_roots', 'validate_official_layout',
    'sample_train_subset', 'sample_smoke_train',
]
