"""Evaluate a CSV against explicitly supplied labels; no positional joins."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from kmd.core import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()
    truth, pred = read_csv(args.labels), read_csv(args.predictions)
    for frame in (truth, pred):
        if not {'pair_id', 'fake_position'} <= set(frame) or frame.empty or not frame.pair_id.is_unique:
            raise ValueError('Need nonempty unique pair_id,fake_position columns.')
        if frame.pair_id.isna().any() or not frame.fake_position.isin([0, 1]).all():
            raise ValueError('Missing ID or invalid binary label.')
    if set(truth.pair_id) != set(pred.pair_id):
        raise ValueError('Predictions must cover exactly the label IDs.')
    aligned = pred.set_index('pair_id').loc[truth.pair_id]
    y, p = truth.fake_position.to_numpy(), aligned.fake_position.to_numpy()
    print({'n': len(y), 'macro_f1': float(f1_score(y, p, average='macro', labels=[0, 1], zero_division=0)),
           'accuracy': float(accuracy_score(y, p)), 'errors': int(np.sum(y != p))})


if __name__ == '__main__':
    main()
