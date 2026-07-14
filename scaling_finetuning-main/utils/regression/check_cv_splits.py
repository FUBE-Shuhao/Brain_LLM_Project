#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect cross-validation splits used by the fMRI ridge pipeline.

The paper states that fMRI ridge evaluation used ten-fold cross-validation
over 148 sentences. The old fixed split is included here only as a comparison.

Usage:
  python utils/regression/check_cv_splits.py
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold


N_SENTENCES = 148


def describe_fold(label: str, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    print()
    print(f"== {label} ==")
    print(f"n_folds: {len(folds)}")
    all_test = []
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        all_test.extend(test_idx.tolist())
        print(
            f"fold {i:02d}: "
            f"train={len(train_idx):3d}, test={len(test_idx):2d}, "
            f"test_range={int(test_idx[0]) + 1:3d}-{int(test_idx[-1]) + 1:3d}"
        )

    counts = np.bincount(np.array(all_test, dtype=int), minlength=N_SENTENCES)
    print("test coverage:")
    print(f"  min/max times each sentence appears in test: {counts.min()} / {counts.max()}")
    print(f"  all sentences tested exactly once: {bool(np.all(counts == 1))}")


def main() -> None:
    paper10fold = list(KFold(n_splits=10, shuffle=False).split(np.arange(N_SENTENCES)))
    fixed = [(np.arange(0, 133), np.arange(133, N_SENTENCES))]

    print("Cross-validation split check")
    print(f"n_sentences: {N_SENTENCES}")
    print()
    print("Paper method text says: ten-fold cross-validation, about 90% train and 10% test.")
    print("Figure schematic shows one 133/15 train-test example.")
    print("A single fixed 133/15 split is therefore a schematic/legacy comparison, not the full ten-fold evaluation.")

    describe_fold("paper10fold, current default", paper10fold)
    describe_fold("fixed 133/15, legacy comparison", fixed)

    print()
    print("Conclusion")
    print("  Use CV_MODE=paper10fold for the Figure 4b-style main analysis.")
    print("  Use CV_MODE=fixed only for legacy/debug comparison.")


if __name__ == "__main__":
    main()
