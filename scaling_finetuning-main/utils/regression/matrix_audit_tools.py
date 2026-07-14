# -*- coding: utf-8 -*-
"""
Matrix audit tools for heads_vs_fmri_original_pathfixed.py

Purpose:
1. Check X_all / Y_all row count and orientation.
2. Check sentence -> edge row slices.
3. Check fold-level train/test edge masks.
4. Check every edge row is predicted exactly once in paper10fold.
5. Recompute Pearson r from y_true/y_pred and compare with core corr.

This file does NOT change regression results.
"""

import json
import os
from pathlib import Path

import numpy as np


def _as_array(x, name):
    arr = np.asarray(x)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _word_len(sent):
    """
    words_list.p is usually list[list[str]].
    This helper is intentionally conservative.
    """
    try:
        return len(sent)
    except TypeError:
        raise TypeError(f"Cannot get word length from sentence object: {type(sent)}")


def sentence_edge_counts(words_list, include_diagonal=False):
    counts = []
    for sent in words_list:
        n = _word_len(sent)
        if include_diagonal:
            counts.append(n * (n + 1) // 2)
        else:
            counts.append(n * (n - 1) // 2)
    return np.asarray(counts, dtype=int)


def sentence_edge_slices(words_list, include_diagonal=False):
    counts = sentence_edge_counts(words_list, include_diagonal=include_diagonal)
    slices = []
    start = 0
    for c in counts:
        end = start + int(c)
        slices.append(slice(start, end))
        start = end
    return slices, counts, int(start)


def normalize_x_orientation(x_all, expected_edges):
    """
    Expected final X shape: (n_edges, n_features / n_heads).

    Some older code may keep X as (n_heads, n_edges), so this detects and transposes.
    """
    x_all = _as_array(x_all, "x_all")

    if x_all.ndim != 2:
        raise ValueError(f"x_all must be 2D, got shape={x_all.shape}")

    if x_all.shape[0] == expected_edges:
        return x_all, "rows_are_edges"

    if x_all.shape[1] == expected_edges:
        return x_all.T, "transposed_from_features_by_edges"

    raise ValueError(
        f"x_all does not contain expected edge dimension {expected_edges}. "
        f"Got shape={x_all.shape}"
    )


def normalize_y_orientation(y_all, expected_edges):
    """
    Expected final Y shape: (n_edges, n_vertices).
    """
    y_all = _as_array(y_all, "y_all")

    if y_all.ndim != 2:
        raise ValueError(f"y_all must be 2D, got shape={y_all.shape}")

    if y_all.shape[0] == expected_edges:
        return y_all, "rows_are_edges"

    if y_all.shape[1] == expected_edges:
        return y_all.T, "transposed_from_vertices_by_edges"

    raise ValueError(
        f"y_all does not contain expected edge dimension {expected_edges}. "
        f"Got shape={y_all.shape}"
    )


def make_sentence_folds(n_sentences, cv_mode="paper10fold"):
    """
    Mirrors the intended split checked by check_cv_splits.py.

    paper10fold for 148 sentences:
      fold01 test 1-15
      ...
      fold08 test 106-120
      fold09 test 121-134
      fold10 test 135-148

    fixed:
      test only final 15 sentences, legacy/debug comparison.
    """
    sent_ids = np.arange(n_sentences)

    if cv_mode == "paper10fold":
        test_chunks = np.array_split(sent_ids, 10)
        folds = []
        for test_sents in test_chunks:
            test_sents = np.asarray(test_sents, dtype=int)
            train_sents = np.setdiff1d(sent_ids, test_sents)
            folds.append((train_sents, test_sents))
        return folds

    if cv_mode == "fixed":
        test_sents = sent_ids[-15:]
        train_sents = sent_ids[:-15]
        return [(train_sents, test_sents)]

    raise ValueError(f"Unknown cv_mode={cv_mode!r}")


def edge_indices_from_sentences(sentence_indices, edge_slices):
    parts = []
    for si in sentence_indices:
        sl = edge_slices[int(si)]
        parts.append(np.arange(sl.start, sl.stop, dtype=int))
    if not parts:
        return np.asarray([], dtype=int)
    return np.concatenate(parts)


def pearsonr_cols(y_true, y_pred):
    """
    Column-wise Pearson r.
    Input shape: (n_samples, n_vertices).

    Constant columns get r=0 instead of NaN.
    """
    y_true = _as_array(y_true, "y_true").astype(np.float64, copy=False)
    y_pred = _as_array(y_pred, "y_pred").astype(np.float64, copy=False)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}")

    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)

    num = np.sum(yt * yp, axis=0)
    den = np.sqrt(np.sum(yt * yt, axis=0) * np.sum(yp * yp, axis=0))

    r = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=(den > 0))
    return r


def summary_1d(arr):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return {
            "n": int(arr.size),
            "n_finite": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "positive_ratio": None,
        }

    x = arr[finite]
    return {
        "n": int(arr.size),
        "n_finite": int(finite.sum()),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "positive_ratio": float(np.mean(x > 0)),
    }


class MatrixAuditRecorder:
    """
    Use this inside the real core script.

    Typical use:

        audit = MatrixAuditRecorder(...)
        ...
        audit.record_fold(...)
        ...
        audit.finish(y_true_all=y_all, y_pred_all=y_pred_all, corr_from_core=corr)

    It writes one JSON file and raises AssertionError / ValueError if something is wrong.
    """

    def __init__(
        self,
        words_list,
        x_all,
        y_all,
        cv_mode="paper10fold",
        out_dir="_matrix_audit",
        tag="audit",
        include_diagonal=False,
    ):
        self.words_list = words_list
        self.n_sentences = len(words_list)
        self.cv_mode = cv_mode
        self.out_dir = Path(out_dir)
        self.tag = str(tag)
        self.include_diagonal = include_diagonal

        self.edge_slices, self.edge_counts, self.expected_edges = sentence_edge_slices(
            words_list,
            include_diagonal=include_diagonal,
        )

        self.x_all, self.x_orientation = normalize_x_orientation(x_all, self.expected_edges)
        self.y_all, self.y_orientation = normalize_y_orientation(y_all, self.expected_edges)

        if self.x_all.shape[0] != self.y_all.shape[0]:
            raise AssertionError(
                f"X/Y row mismatch after orientation normalization: "
                f"X={self.x_all.shape}, Y={self.y_all.shape}"
            )

        if not np.isfinite(self.x_all).all():
            bad = int(np.size(self.x_all) - np.isfinite(self.x_all).sum())
            raise AssertionError(f"x_all contains non-finite values: {bad}")

        if not np.isfinite(self.y_all).all():
            bad = int(np.size(self.y_all) - np.isfinite(self.y_all).sum())
            raise AssertionError(f"y_all contains non-finite values: {bad}")

        self.expected_folds = make_sentence_folds(self.n_sentences, cv_mode=cv_mode)

        self.expected_sentence_test_count = np.zeros(self.n_sentences, dtype=int)
        self.expected_edge_test_count = np.zeros(self.expected_edges, dtype=int)

        self.expected_fold_info = []
        for fold_id, (train_sents, test_sents) in enumerate(self.expected_folds, start=1):
            train_edges = edge_indices_from_sentences(train_sents, self.edge_slices)
            test_edges = edge_indices_from_sentences(test_sents, self.edge_slices)

            if np.intersect1d(train_edges, test_edges).size != 0:
                raise AssertionError(f"Expected fold {fold_id}: train/test edge overlap")

            self.expected_sentence_test_count[test_sents] += 1
            self.expected_edge_test_count[test_edges] += 1

            self.expected_fold_info.append(
                {
                    "fold_id": int(fold_id),
                    "train_sentences": int(len(train_sents)),
                    "test_sentences": int(len(test_sents)),
                    "train_edges": int(len(train_edges)),
                    "test_edges": int(len(test_edges)),
                    "test_sentence_range_1based": [
                        int(test_sents.min() + 1),
                        int(test_sents.max() + 1),
                    ],
                    "test_edge_range_0based": [
                        int(test_edges.min()) if len(test_edges) else None,
                        int(test_edges.max()) if len(test_edges) else None,
                    ],
                }
            )

        self.actual_edge_test_count = np.zeros(self.expected_edges, dtype=int)
        self.actual_fold_info = []

        self.report = {
            "tag": self.tag,
            "cv_mode": self.cv_mode,
            "n_sentences": int(self.n_sentences),
            "expected_edges": int(self.expected_edges),
            "sentence_edge_count_min": int(self.edge_counts.min()),
            "sentence_edge_count_median": float(np.median(self.edge_counts)),
            "sentence_edge_count_max": int(self.edge_counts.max()),
            "x_shape_original_normalized": list(self.x_all.shape),
            "y_shape_original_normalized": list(self.y_all.shape),
            "x_orientation": self.x_orientation,
            "y_orientation": self.y_orientation,
            "n_features": int(self.x_all.shape[1]),
            "n_vertices": int(self.y_all.shape[1]),
            "expected_fold_info": self.expected_fold_info,
            "expected_sentence_test_count_min": int(self.expected_sentence_test_count.min()),
            "expected_sentence_test_count_max": int(self.expected_sentence_test_count.max()),
            "expected_edge_test_count_min": int(self.expected_edge_test_count.min()),
            "expected_edge_test_count_max": int(self.expected_edge_test_count.max()),
            "actual_fold_info": self.actual_fold_info,
        }

        if cv_mode == "paper10fold":
            assert self.expected_sentence_test_count.min() == 1
            assert self.expected_sentence_test_count.max() == 1
            assert self.expected_edge_test_count.min() == 1
            assert self.expected_edge_test_count.max() == 1

    def record_fold(
        self,
        fold_id,
        train_edges,
        test_edges,
        x_train=None,
        y_train=None,
        y_pred=None,
    ):
        """
        Call this inside the real CV loop after train_edges/test_edges are known.

        train_edges/test_edges must be edge-row indices, not sentence indices.
        """
        train_edges = np.asarray(train_edges, dtype=int)
        test_edges = np.asarray(test_edges, dtype=int)

        if train_edges.ndim != 1 or test_edges.ndim != 1:
            raise ValueError("train_edges/test_edges must be 1D edge-row index arrays")

        if len(train_edges) == 0 or len(test_edges) == 0:
            raise AssertionError(f"Actual fold {fold_id}: empty train/test edges")

        if train_edges.min() < 0 or test_edges.min() < 0:
            raise AssertionError(f"Actual fold {fold_id}: negative edge index")

        if train_edges.max() >= self.expected_edges or test_edges.max() >= self.expected_edges:
            raise AssertionError(
                f"Actual fold {fold_id}: edge index out of range. "
                f"expected_edges={self.expected_edges}, "
                f"train_max={train_edges.max()}, test_max={test_edges.max()}"
            )

        overlap = np.intersect1d(train_edges, test_edges)
        if overlap.size != 0:
            raise AssertionError(
                f"Actual fold {fold_id}: train/test overlap, n_overlap={overlap.size}"
            )

        self.actual_edge_test_count[test_edges] += 1

        item = {
            "fold_id": int(fold_id),
            "train_edges": int(len(train_edges)),
            "test_edges": int(len(test_edges)),
            "train_edge_minmax": [int(train_edges.min()), int(train_edges.max())],
            "test_edge_minmax": [int(test_edges.min()), int(test_edges.max())],
        }

        if x_train is not None:
            x_train = np.asarray(x_train)
            item["x_train_shape"] = list(x_train.shape)
            if x_train.shape[0] != len(train_edges):
                raise AssertionError(
                    f"Actual fold {fold_id}: x_train rows != train_edges. "
                    f"{x_train.shape[0]} vs {len(train_edges)}"
                )

        if y_train is not None:
            y_train = np.asarray(y_train)
            item["y_train_shape"] = list(y_train.shape)
            if y_train.shape[0] != len(train_edges):
                raise AssertionError(
                    f"Actual fold {fold_id}: y_train rows != train_edges. "
                    f"{y_train.shape[0]} vs {len(train_edges)}"
                )

        if y_pred is not None:
            y_pred = np.asarray(y_pred)
            item["y_pred_shape"] = list(y_pred.shape)
            if y_pred.shape[0] != len(test_edges):
                raise AssertionError(
                    f"Actual fold {fold_id}: y_pred rows != test_edges. "
                    f"{y_pred.shape[0]} vs {len(test_edges)}"
                )

        self.actual_fold_info.append(item)

    def finish(
        self,
        y_true_all,
        y_pred_all,
        corr_from_core=None,
        corr_path=None,
        extra=None,
    ):
        """
        Call after all folds are done.

        y_true_all: usually y_all
        y_pred_all: full out-of-fold prediction matrix, same shape as y_all
        corr_from_core: core script's r vector, optional
        corr_path: path to saved corr.npy, optional
        """
        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)

        y_true_all, y_true_orientation = normalize_y_orientation(y_true_all, self.expected_edges)
        y_pred_all, y_pred_orientation = normalize_y_orientation(y_pred_all, self.expected_edges)

        if y_true_all.shape != y_pred_all.shape:
            raise AssertionError(
                f"Prediction shape mismatch: y_true={y_true_all.shape}, "
                f"y_pred={y_pred_all.shape}"
            )

        if not np.isfinite(y_pred_all).all():
            bad = int(np.size(y_pred_all) - np.isfinite(y_pred_all).sum())
            raise AssertionError(f"y_pred_all contains non-finite values: {bad}")

        self.report["actual_fold_info"] = self.actual_fold_info
        self.report["actual_n_recorded_folds"] = int(len(self.actual_fold_info))
        self.report["actual_edge_test_count_min"] = int(self.actual_edge_test_count.min())
        self.report["actual_edge_test_count_max"] = int(self.actual_edge_test_count.max())
        self.report["y_true_orientation_finish"] = y_true_orientation
        self.report["y_pred_orientation_finish"] = y_pred_orientation

        if self.cv_mode == "paper10fold":
            if self.actual_edge_test_count.min() != 1 or self.actual_edge_test_count.max() != 1:
                raise AssertionError(
                    "Actual fold prediction coverage is wrong: "
                    f"edge_test_count min/max = "
                    f"{self.actual_edge_test_count.min()} / {self.actual_edge_test_count.max()}"
                )

        r_recomputed = pearsonr_cols(y_true_all, y_pred_all)
        self.report["r_recomputed_summary"] = summary_1d(r_recomputed)

        corr_core = None
        if corr_from_core is not None:
            corr_core = np.asarray(corr_from_core)
        elif corr_path is not None:
            corr_core = np.load(corr_path)

        if corr_core is not None:
            corr_core = np.asarray(corr_core).reshape(-1)
            if corr_core.shape != r_recomputed.shape:
                raise AssertionError(
                    f"corr shape mismatch: core={corr_core.shape}, "
                    f"recomputed={r_recomputed.shape}"
                )

            diff = corr_core - r_recomputed
            self.report["corr_core_summary"] = summary_1d(corr_core)
            self.report["corr_core_minus_recomputed"] = {
                "max_abs_diff": float(np.max(np.abs(diff))),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "median_abs_diff": float(np.median(np.abs(diff))),
            }

        if extra is not None:
            self.report["extra"] = extra

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.out_dir / f"{self.tag}.matrix_audit.json"
        out_path.write_text(json.dumps(self.report, indent=2, ensure_ascii=False))

        print("\n[MATRIX AUDIT PASS]")
        print(f"  tag              : {self.tag}")
        print(f"  cv_mode          : {self.cv_mode}")
        print(f"  X shape          : {self.x_all.shape}")
        print(f"  Y shape          : {self.y_all.shape}")
        print(f"  expected edges   : {self.expected_edges}")
        print(
            f"  actual edge cover: "
            f"{self.actual_edge_test_count.min()} / {self.actual_edge_test_count.max()}"
        )
        print(f"  r mean/median    : {self.report['r_recomputed_summary']['mean']:.6f} / "
              f"{self.report['r_recomputed_summary']['median']:.6f}")

        if "corr_core_minus_recomputed" in self.report:
            print(
                "  corr diff maxabs : "
                f"{self.report['corr_core_minus_recomputed']['max_abs_diff']:.12g}"
            )

        print(f"  saved            : {out_path}")
        return self.report
