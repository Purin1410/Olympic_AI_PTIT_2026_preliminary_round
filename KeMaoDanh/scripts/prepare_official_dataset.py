"""Prepare and validate official competition dataset from ZIP or directory.

Extracts official ZIP while preserving raw image bytes, validates path safety,
resolves nested roots (including data/data/train and extra data/ wrappers),
and strictly isolates test inference from groundtruth files (pairs_results.csv).
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kmd.dataset_prep import extract_official_zip, discover_dataset_roots, validate_official_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=None, help="Path to official dataset ZIP")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1] / "data")),
        help="Destination directory for extraction (default: KeMaoDanh/data)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit overwriting existing files (default: False)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only discover and validate existing dataset structure without extraction",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir).expanduser().resolve()

    if args.zip_path is not None and not args.check_only:
        zip_path = Path(args.zip_path).expanduser().resolve()
        print(f"Extracting official dataset from {zip_path} -> {dest_dir}...")
        result = extract_official_zip(zip_path, dest_dir, overwrite=args.overwrite)
        print(f"Extracted {result['total_files']} files successfully.")

    roots = discover_dataset_roots(dest_dir)
    layout = validate_official_layout(roots)

    print("\n=== Dataset Roots Discovery ===")
    for k, v in roots.items():
        print(f"- {k}: {v}")

    print("\n=== Dataset Validation Report ===")
    print(json.dumps(layout, indent=2))

    if layout.get("train_valid") and layout.get("public_test_valid"):
        print("\nDataset structure is verified and matches official competition specifications.")
    else:
        print("\nNote: Some official sets were not found or differed from standard counts.")


if __name__ == "__main__":
    main()
