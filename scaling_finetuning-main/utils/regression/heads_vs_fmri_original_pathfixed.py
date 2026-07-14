#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 截至目前，最后一次修改为2026/7/9
"""
Path-fixed fMRI ridge regression for Reading Brain model-attention analyses.

This version keeps the original repository's file layout and command-line
interface, but makes the default analysis closer to the method described in
the paper:

- fMRI target matrices use summed BOLD values for repeated word transitions.
- Evaluation uses sentence-level 10-fold cross-validation by default.
- The old fixed 133/15 split is still available through CV_MODE=fixed.
- A JSON sidecar is saved next to each .npy result to record run settings.

Usage:
  python heads_vs_fmri_original_pathfixed.py 1 base llama 7B lh 25

Environment overrides:
  PROJECT_DIR=/path/to/scaling_finetuning-main
  DATA_DIR=/path/to/project/Data
  ATTN_DIR=/path/to/project/model_attention
  OUTPUT_ROOT=/path/to/result/root
  N_JOBS=8

Analysis controls:
  CV_MODE=paper10fold        # paper10fold or fixed
  BOLD_AGGREGATION=sum      # sum or last
  FMRI_ZSCORE_AXIS=0        # 0 keeps original code behavior; 1 is vertex time-course z-score
  MASK_ZERO_Y=0             # 1 excludes zero target edges from Pearson r
  SUBTRACT_TRIVIAL_PATTERNS=1
  TRIVIAL_PATTERNS=first_word,previous_word
  ATTN_PREFIX_OVERRIDE=...  # optional, for custom/residual attention filenames
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import joblib.parallel
from joblib import Parallel, delayed
from scipy.stats import ConstantInputWarning, pearsonr, zscore
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# =========================
# Local path configuration
# =========================
PROJECT_DIR = Path(os.environ.get(
    "PROJECT_DIR",
    "/home/wsh/repository/Brain_LLM_Project/scaling_finetuning-main",
))
BRAIN_PROJECT_DIR = Path(os.environ.get(
    "BRAIN_PROJECT_DIR",
    "/home/wsh/repository/Brain_LLM_Project/",
))
ANALYSIS_DIR = Path(os.environ.get("ANALYSIS_DIR", str(PROJECT_DIR / "Analysis")))
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT_DIR / "Data")))
ATTN_DIR = Path(os.environ.get("ATTN_DIR", str(PROJECT_DIR / "model_attention")))
OUTPUT_ROOT = Path(os.environ.get(
    "OUTPUT_ROOT",
    str(BRAIN_PROJECT_DIR / "4_Results_paper_heads_vs_fmri"),
))

N_JOBS = int(os.environ.get("N_JOBS", "-1"))
CV_MODE = os.environ.get("CV_MODE", "paper10fold").strip().lower()
BOLD_AGGREGATION = os.environ.get("BOLD_AGGREGATION", "sum").strip().lower()
FMRI_ZSCORE_AXIS = int(os.environ.get("FMRI_ZSCORE_AXIS", "0"))
MASK_ZERO_Y = os.environ.get("MASK_ZERO_Y", "0").strip().lower() in {"1", "true", "yes", "y"}
SHOW_PROGRESS = os.environ.get("SHOW_PROGRESS", "1").strip().lower() in {"1", "true", "yes", "y"}
PROGRESS_UPDATE_SECONDS = float(os.environ.get("PROGRESS_UPDATE_SECONDS", "1.0"))
SUBTRACT_TRIVIAL_PATTERNS = os.environ.get(
    "SUBTRACT_TRIVIAL_PATTERNS", "1"
).strip().lower() in {"1", "true", "yes", "y"}
TRIVIAL_PATTERNS = tuple(
    p.strip().lower()
    for p in os.environ.get("TRIVIAL_PATTERNS", "first_word,previous_word").split(",")
    if p.strip()
)
ATTN_PREFIX_OVERRIDE = os.environ.get("ATTN_PREFIX_OVERRIDE")

ALPHAS = np.logspace(1, 3, 20)

MODEL_SPECS = {
    "gpt2_base": {"layers": 12, "heads": 12},
    "gpt2_medium": {"layers": 24, "heads": 16},
    "gpt2_large": {"layers": 36, "heads": 20},
    "gpt2_xlarge": {"layers": 48, "heads": 25},
    "llama_7B": {"layers": 32, "heads": 32},
    "llama_13B": {"layers": 40, "heads": 40},
    "llama_30B": {"layers": 60, "heads": 52},
    "llama_65B": {"layers": 80, "heads": 64},
    "alpaca_7B": {"layers": 32, "heads": 32},
    "alpaca_13B": {"layers": 40, "heads": 40},
    "vicuna_7B": {"layers": 32, "heads": 32},
    "vicuna_13B": {"layers": 40, "heads": 40},
    "mistral_7B": {"layers": 32, "heads": 32},
    "gemma_7B": {"layers": 28, "heads": 16},
}

GROUP_TO_ATTN_PREFIX = {
    "base": "rb_p1",
    "instr": "instr_rb_p1",
    "ctrl": "ctrl_rb_p1",
}

EXPECTED_FSAVERAGE5_VERTICES = 10242
RM_SENTENCE_IDXS = [142, 143, 147, 148, 152, 153, 154]
SNTS_LIST = [31, 31, 28, 28, 30]
CUMS = np.cumsum(SNTS_LIST)
LAST_TRIVIAL_PATTERN_DIAGNOSTICS: dict[str, object] = {}


def existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    msg = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of these paths exists:\n{msg}")


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def sentence_id(article: int, sentence: int) -> str:
    return f"t.0{article}.0{sentence}" if sentence < 10 else f"t.0{article}.{sentence}"


def safe_zscore(arr: np.ndarray, axis=None) -> np.ndarray:
    return np.nan_to_num(zscore(arr, axis=axis, nan_policy="omit"))


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class SimpleProgressBar:
    def __init__(self, total: int, label: str, width: int = 32):
        self.total = int(total)
        self.label = label
        self.width = int(width)
        self.done = 0
        self.start = time.time()
        self.last_render = 0.0

    def update(self, n: int) -> None:
        self.done = min(self.total, self.done + int(n))
        self.render()

    def render(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_render < PROGRESS_UPDATE_SECONDS and self.done < self.total:
            return
        self.last_render = now

        frac = self.done / self.total if self.total else 1.0
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self.start
        rate = self.done / elapsed if elapsed > 0 and self.done else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        msg = (
            f"\r{self.label}: [{bar}] {self.done}/{self.total} "
            f"({frac * 100:5.1f}%) elapsed {format_duration(elapsed)} "
            f"ETA {format_duration(eta)}"
        )
        print(msg, end="", flush=True)

    def close(self) -> None:
        self.done = self.total
        self.render(force=True)
        print()


@contextmanager
def joblib_progress(total: int, label: str):
    if not SHOW_PROGRESS:
        yield
        return

    progress = SimpleProgressBar(total=total, label=label)
    old_callback = joblib.parallel.BatchCompletionCallBack

    class ProgressBatchCompletionCallBack(old_callback):
        def __call__(self, *args, **kwargs):
            progress.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = ProgressBatchCompletionCallBack
    try:
        progress.render(force=True)
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        progress.close()


def make_trivial_patterns_by_sentence(
    words_list: list[int],
    pattern_names: tuple[str, ...],
) -> tuple[list[np.ndarray], list[str]]:
    pattern_by_sentence = []
    names = list(pattern_names)
    valid_names = {"first_word", "previous_word", "current_word"}
    unknown = sorted(set(names) - valid_names)
    if unknown:
        raise ValueError(f"Unknown trivial pattern(s): {unknown}. Valid: {sorted(valid_names)}")

    for n_word in words_list:
        n_word = int(n_word)
        tril_idx = np.tril_indices(n_word, k=-1)
        patterns = []

        for name in names:
            pattern = np.zeros((n_word, n_word), dtype=np.float64)
            if name == "first_word":
                pattern[1:, 0] = 1.0
            elif name == "previous_word":
                for row in range(1, n_word):
                    pattern[row, row - 1] = 1.0
            elif name == "current_word":
                np.fill_diagonal(pattern, 1.0)

            patterns.append(pattern[tril_idx[0], tril_idx[1]])

        if patterns:
            pattern_by_sentence.append(np.stack(patterns, axis=1))
        else:
            pattern_by_sentence.append(np.zeros((len(tril_idx[0]), 0), dtype=np.float64))

    return pattern_by_sentence, names


def split_edges_by_sentence(x_edges_by_head: np.ndarray, lengths: list[int]) -> list[np.ndarray]:
    out = []
    start = 0
    for length in lengths:
        end = start + int(length)
        out.append(x_edges_by_head[start:end].T)
        start = end

    if start != x_edges_by_head.shape[0]:
        raise ValueError(f"Split consumed {start} edges, but X has {x_edges_by_head.shape[0]}")
    return out


def trivial_pattern_r2(x_edges_by_head: np.ndarray, patterns: np.ndarray) -> dict[str, float | list[float]]:
    x_centered = x_edges_by_head - x_edges_by_head.mean(axis=0, keepdims=True)
    p_centered = patterns - patterns.mean(axis=0, keepdims=True)
    ss_total = np.sum(x_centered * x_centered, axis=0)
    valid_heads = ss_total > 0
    if not np.any(valid_heads):
        return {"mean": 0.0, "max": 0.0, "per_head": []}

    beta = np.linalg.lstsq(p_centered, x_centered[:, valid_heads], rcond=None)[0]
    resid = x_centered[:, valid_heads] - p_centered @ beta
    ss_resid = np.sum(resid * resid, axis=0)
    r2 = 1.0 - ss_resid / ss_total[valid_heads]
    r2 = np.clip(np.nan_to_num(r2), 0.0, 1.0)
    return {
        "mean": float(np.mean(r2)),
        "max": float(np.max(r2)),
        "per_head": [float(v) for v in r2],
    }


def residualize_trivial_patterns(
    x_by_sentence: list[np.ndarray],
    pattern_by_sentence: list[np.ndarray],
    pattern_names: list[str],
) -> tuple[list[np.ndarray], dict[str, object]]:
    lengths = [x.shape[1] for x in x_by_sentence]

    x_edges_by_head = np.concatenate(x_by_sentence, axis=1).T.astype(np.float64, copy=False)
    patterns = np.concatenate(pattern_by_sentence, axis=0).astype(np.float64, copy=False)

    if x_edges_by_head.shape[0] != patterns.shape[0]:
        raise ValueError(f"Pattern shape mismatch: X={x_edges_by_head.shape}, P={patterns.shape}")

    keep = patterns.std(axis=0) > 0
    kept_names = [name for name, use in zip(pattern_names, keep) if use]
    dropped_names = [name for name, use in zip(pattern_names, keep) if not use]
    patterns = patterns[:, keep]

    diagnostics: dict[str, object] = {
        "requested_patterns": pattern_names,
        "active_patterns": kept_names,
        "dropped_patterns": dropped_names,
        "n_edges": int(x_edges_by_head.shape[0]),
    }

    if patterns.shape[1] == 0:
        diagnostics["skipped"] = "no non-constant trivial pattern columns"
        return x_by_sentence, diagnostics

    diagnostics["r2_before"] = trivial_pattern_r2(x_edges_by_head, patterns)

    x_centered = x_edges_by_head - x_edges_by_head.mean(axis=0, keepdims=True)
    p_centered = patterns - patterns.mean(axis=0, keepdims=True)

    beta = np.linalg.lstsq(p_centered, x_centered, rcond=None)[0]
    x_residual = x_centered - p_centered @ beta

    diagnostics["rank"] = int(np.linalg.matrix_rank(p_centered))
    diagnostics["r2_after"] = trivial_pattern_r2(x_residual, patterns)

    return split_edges_by_sentence(x_residual, lengths), diagnostics


def zscore_x_by_sentence(x_by_sentence: list[np.ndarray]) -> list[np.ndarray]:
    lengths = [x.shape[1] for x in x_by_sentence]
    x_all_edges = np.concatenate(x_by_sentence, axis=1).astype(np.float64, copy=False)
    x_all_edges = safe_zscore(x_all_edges.flatten(), axis=None).reshape(x_all_edges.shape)

    out = []
    start = 0
    for length in lengths:
        end = start + int(length)
        out.append(x_all_edges[:, start:end])
        start = end
    return out


def build_article_major_attention_slots(layer_attn: np.ndarray, n_head: int) -> np.ndarray:
    if layer_attn.shape[0] != len(SNTS_LIST):
        raise ValueError(f"Expected {len(SNTS_LIST)} articles, got attention shape={layer_attn.shape}")
    if layer_attn.shape[2] != n_head:
        raise ValueError(f"Expected {n_head} heads, got attention shape={layer_attn.shape}")

    max_sentences = layer_attn.shape[1]
    slots = []
    for article_idx, n_sentences in enumerate(SNTS_LIST):
        if n_sentences > max_sentences:
            raise ValueError(
                f"Article {article_idx + 1} needs {n_sentences} sentences, "
                f"but attention only has {max_sentences} slots"
            )
        slots.append(layer_attn[article_idx, :n_sentences])

    # Raw attention is article x sentence x head x word x word.
    # The BOLD target is built article-major too, so X must use the same order.
    return np.concatenate(slots, axis=0).transpose(1, 0, 2, 3)


def sentence_fold_indices(n_sentences: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if CV_MODE == "fixed":
        train_idx = np.arange(0, 133)
        test_idx = np.arange(133, n_sentences)
        return [(train_idx, test_idx)]

    if CV_MODE != "paper10fold":
        raise ValueError("CV_MODE must be 'paper10fold' or 'fixed', got: %r" % CV_MODE)

    return list(KFold(n_splits=10, shuffle=False).split(np.arange(n_sentences)))


def load_words() -> tuple[pd.DataFrame, list[int]]:
    words_csv = existing_path(ANALYSIS_DIR / "words.csv", PROJECT_DIR / "words.csv")
    words_list_p = existing_path(ANALYSIS_DIR / "words_list.p", PROJECT_DIR / "words_list.p")

    words = pd.read_csv(words_csv)
    with open(words_list_p, "rb") as f:
        words_list = list(pickle.load(f))

    if len(words_list) != sum(SNTS_LIST):
        raise ValueError(f"Expected {sum(SNTS_LIST)} sentence lengths, got {len(words_list)}")

    total_edges = sum(int(n) * (int(n) - 1) // 2 for n in words_list)
    if total_edges != 7388:
        print(f"Warning: lower-triangle edge count is {total_edges}, not paper-reported 7388.")

    return words, words_list


def load_layer_attention(name: str, size: str, group: str, layer: int, n_head: int) -> tuple[np.ndarray, Path]:
    attn_prefix = ATTN_PREFIX_OVERRIDE or GROUP_TO_ATTN_PREFIX[group]
    layer_attn_path = require_file(
        ATTN_DIR / name / size / f"{attn_prefix}_layer{layer}.npy",
        "Attention file",
    )

    print(f"Loading attention: {layer_attn_path}")
    layer_attn = np.load(layer_attn_path)
    if layer_attn.ndim != 5:
        raise ValueError(f"Attention file should be 5D, got shape={layer_attn.shape}: {layer_attn_path}")
    if layer_attn.shape[2] != n_head:
        raise ValueError(
            f"Attention head mismatch: file has {layer_attn.shape[2]} heads, expected {n_head}. "
            f"Shape={layer_attn.shape}"
        )

    return layer_attn, layer_attn_path


def build_attention_by_sentence(layer_attn: np.ndarray, words_list: list[int], n_head: int) -> list[np.ndarray]:
    global LAST_TRIVIAL_PATTERN_DIAGNOSTICS

    x_all = build_article_major_attention_slots(layer_attn, n_head)

    if x_all.shape[0] != n_head:
        raise ValueError(f"Head mismatch after reshaping: got {x_all.shape[0]}, expected {n_head}")
    if x_all.shape[1] != len(words_list):
        raise ValueError(f"Sentence mismatch after deletion: got {x_all.shape[1]}, expected {len(words_list)}")
    if x_all.shape[-1] < max(words_list):
        raise ValueError(f"Attention word dimension {x_all.shape[-1]} < max sentence length {max(words_list)}")

    x_all = x_all.astype(np.float64, copy=False)

    x_by_sentence = []
    for sid, n_word in enumerate(words_list):
        n_word = int(n_word)
        tril_idx = np.tril_indices(n_word, k=-1)
        x_snt = x_all[:, sid, :n_word, :n_word]
        x_by_sentence.append(x_snt[:, tril_idx[0], tril_idx[1]])

    LAST_TRIVIAL_PATTERN_DIAGNOSTICS = {
        "enabled": bool(SUBTRACT_TRIVIAL_PATTERNS),
        "requested_patterns": list(TRIVIAL_PATTERNS),
    }
    if SUBTRACT_TRIVIAL_PATTERNS:
        pattern_by_sentence, pattern_names = make_trivial_patterns_by_sentence(words_list, TRIVIAL_PATTERNS)
        x_by_sentence, diagnostics = residualize_trivial_patterns(
            x_by_sentence,
            pattern_by_sentence,
            pattern_names,
        )
        diagnostics["enabled"] = True
        LAST_TRIVIAL_PATTERN_DIAGNOSTICS = diagnostics

        before = diagnostics.get("r2_before", {})
        after = diagnostics.get("r2_after", {})
        print(
            "Trivial-pattern residualization: "
            f"active={diagnostics.get('active_patterns', [])}, "
            f"dropped={diagnostics.get('dropped_patterns', [])}, "
            f"R2 mean/max before={before.get('mean', 0.0):.6f}/{before.get('max', 0.0):.6f}, "
            f"after={after.get('mean', 0.0):.6f}/{after.get('max', 0.0):.6f}"
        )
    else:
        print("Trivial-pattern residualization: disabled")

    return zscore_x_by_sentence(x_by_sentence)


def load_events_and_surfaces(subj: str, group: str, hem: str) -> tuple[pd.DataFrame, list[np.ndarray]]:
    surf = []
    events_lst = []

    for article in range(1, 6):
        run = 1
        events_path = require_file(
            DATA_DIR / group / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv",
            "Event file",
        )
        events = pd.read_csv(events_path, delimiter="\t").dropna().reset_index(drop=True)
        run_article = int(str(events.SentenceID.iloc[0]).split(".")[1])

        while run_article != article and run < 5:
            run += 1
            events_path = require_file(
                DATA_DIR / group / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv",
                "Event file",
            )
            events = pd.read_csv(events_path, delimiter="\t").dropna().reset_index(drop=True)
            run_article = int(str(events.SentenceID.iloc[0]).split(".")[1])

        if run_article != article:
            raise RuntimeError(f"Could not find run for article {article} in {subj}")

        events["article"] = article
        events_lst.append(events)

        fmri_path = require_file(
            DATA_DIR / "fsaverage5" / f"{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii",
            "fMRI surface file",
        )
        fmri_gii = nib.load(str(fmri_path))
        fmri_data = np.column_stack([arr.data for arr in fmri_gii.darrays])
        if fmri_data.shape[0] != EXPECTED_FSAVERAGE5_VERTICES:
            raise ValueError(
                f"Expected {EXPECTED_FSAVERAGE5_VERTICES} fsaverage5 vertices, "
                f"got {fmri_data.shape[0]} in {fmri_path}"
            )

        fmri_data = safe_zscore(fmri_data, axis=FMRI_ZSCORE_AXIS)
        surf.append(fmri_data)
        print(f"Article {article}: run={run}, fmri={fmri_path.name}, events={events_path.name}")

    return pd.concat(events_lst).reset_index(drop=True), surf


def article_sentence_count(words: pd.DataFrame, article: int) -> int:
    prefix = f"t.0{article}."
    article_rows = words[words.SentenceID.astype(str).str.startswith(prefix)]
    if article_rows.empty:
        raise ValueError(f"No words rows found for article {article} using prefix {prefix!r}")
    return int(str(article_rows.SentenceID.iloc[-1]).split(".")[-1])


def build_bold_by_sentence(
    v: int,
    events: pd.DataFrame,
    surf: list[np.ndarray],
    words: pd.DataFrame,
    words_list: list[int],
) -> list[np.ndarray]:
    y_by_sentence = []

    for article in range(1, 6):
        n_snts = article_sentence_count(words, article)
        for sent in range(1, n_snts + 1):
            snt_id = sentence_id(article, sent)
            event = events[events.SentenceID.astype(str) == snt_id].reset_index(drop=True)
            sid = sent - 1 if article == 1 else int(CUMS[article - 2]) + sent - 1
            n_word = int(words_list[sid])

            fmri_snt = np.zeros((n_word, n_word), dtype=np.float64)
            tril_idx = np.tril_indices(n_word, k=-1)

            for ind, e in event.iterrows():
                if ind >= len(event) - 1:
                    continue

                row = int(e.CURRENT_FIX_INTEREST_AREA_ID) - 1
                col = int(event.iloc[ind + 1].CURRENT_FIX_INTEREST_AREA_ID) - 1
                if row < 0 or row >= n_word or col < 0 or col >= n_word:
                    continue

                scan = int(np.ceil((event.iloc[ind + 1].onset / 1000 + 5) / 0.4))
                while scan >= surf[article - 1].shape[1]:
                    scan -= 1

                if BOLD_AGGREGATION == "sum":
                    fmri_snt[row, col] += surf[article - 1][v, scan]
                elif BOLD_AGGREGATION == "last":
                    fmri_snt[row, col] = surf[article - 1][v, scan]
                else:
                    raise ValueError("BOLD_AGGREGATION must be 'sum' or 'last', got: %r" % BOLD_AGGREGATION)

            y_by_sentence.append(fmri_snt[tril_idx[0], tril_idx[1]])

    if len(y_by_sentence) != len(words_list):
        raise ValueError(f"BOLD sentence count mismatch: got {len(y_by_sentence)}, expected {len(words_list)}")

    return y_by_sentence


def concat_sentences(items: list[np.ndarray], sentence_indices: np.ndarray) -> np.ndarray:
    return np.concatenate([items[int(i)] for i in sentence_indices], axis=-1)


def process_vertex(
    v: int,
    x_by_sentence: list[np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
    events: pd.DataFrame,
    surf: list[np.ndarray],
    words: pd.DataFrame,
    words_list: list[int],
) -> float:
    y_by_sentence = build_bold_by_sentence(v, events, surf, words, words_list)
    fold_corrs = []

    for train_idx, test_idx in folds:
        x_train = concat_sentences(x_by_sentence, train_idx)
        x_test = concat_sentences(x_by_sentence, test_idx)
        y_train = concat_sentences(y_by_sentence, train_idx)
        y_test = concat_sentences(y_by_sentence, test_idx)

        if x_train.shape[1] != y_train.shape[0]:
            raise ValueError(f"Train shape mismatch: X={x_train.shape}, y={y_train.shape}")
        if x_test.shape[1] != y_test.shape[0]:
            raise ValueError(f"Test shape mismatch: X={x_test.shape}, y={y_test.shape}")

        model = RidgeCV(alphas=ALPHAS).fit(x_train.T, y_train)
        y_pred = model.predict(x_test.T)

        if MASK_ZERO_Y:
            mask = y_test != 0
            if mask.sum() <= 5:
                fold_corrs.append(0.0)
                continue
            y_pred = y_pred[mask]
            y_test = y_test[mask]

        r, _ = pearsonr(y_pred, y_test)
        fold_corrs.append(float(np.nan_to_num(r)))

    return float(np.nanmean(fold_corrs))


def main() -> None:
    if len(sys.argv) != 7:
        print(
            "Usage: python heads_vs_fmri_original_pathfixed.py "
            "<subj_id> <group> <name> <size> <hem> <layer>\n"
            "Example: python heads_vs_fmri_original_pathfixed.py 1 base llama 7B lh 25",
            file=sys.stderr,
        )
        sys.exit(1)

    subj_id = int(sys.argv[1])
    group = sys.argv[2]
    name = sys.argv[3]
    size = sys.argv[4]
    hem = sys.argv[5]
    layer = int(sys.argv[6])

    if hem not in {"lh", "rh"}:
        raise ValueError(f"hem must be lh or rh, got: {hem!r}")
    if group not in GROUP_TO_ATTN_PREFIX:
        raise ValueError(f"group must be one of {sorted(GROUP_TO_ATTN_PREFIX)}, got: {group!r}")
    if FMRI_ZSCORE_AXIS not in {0, 1, -1}:
        raise ValueError(f"FMRI_ZSCORE_AXIS must be 0, 1, or -1, got: {FMRI_ZSCORE_AXIS}")

    subj = "sub-0%d" % subj_id if subj_id < 10 else "sub-%d" % subj_id
    model_key = "%s_%s" % (name, size)
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model/size: {model_key}. Known: {sorted(MODEL_SPECS)}")

    n_layers = MODEL_SPECS[model_key]["layers"]
    n_head = MODEL_SPECS[model_key]["heads"]
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"Invalid layer for {model_key}: {layer}. Valid range: 0..{n_layers - 1}")

    print("=" * 80)
    print("heads_vs_fmri path-fixed run")
    print(f"Project dir     : {PROJECT_DIR}")
    print(f"Data dir        : {DATA_DIR}")
    print(f"Analysis dir    : {ANALYSIS_DIR}")
    print(f"Output root     : {OUTPUT_ROOT}")
    print(f"Target          : subj={subj}, group={group}, model={model_key}, hemi={hem}, layer={layer}, heads={n_head}")
    print(f"CV mode         : {CV_MODE}")
    print(f"BOLD aggregation: {BOLD_AGGREGATION}")
    print(f"fMRI zscore axis: {FMRI_ZSCORE_AXIS}")
    print(f"Mask zero y     : {MASK_ZERO_Y}")
    print(f"Show progress   : {SHOW_PROGRESS}")
    print(f"Subtract trivial patterns: {SUBTRACT_TRIVIAL_PATTERNS}")
    print(f"Trivial patterns: {list(TRIVIAL_PATTERNS)}")
    print("=" * 80)

    words, words_list = load_words()
    folds = sentence_fold_indices(len(words_list))
    print("Folds:", [(len(train), len(test)) for train, test in folds])

    layer_attn, layer_attn_path = load_layer_attention(name, size, group, layer, n_head)
    x_by_sentence = build_attention_by_sentence(layer_attn, words_list, n_head)
    print(f"Built attention by sentence: {len(x_by_sentence)} sentences")

    events, surf = load_events_and_surfaces(subj, group, hem)

    start = time.time()
    with joblib_progress(EXPECTED_FSAVERAGE5_VERTICES, "Vertices"):
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_vertex)(v, x_by_sentence, folds, events, surf, words, words_list)
            for v in range(EXPECTED_FSAVERAGE5_VERTICES)
        )
    corr = np.array(results, dtype=np.float64)
    elapsed = time.time() - start

    out_dir = OUTPUT_ROOT / name / size / group / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group}_subj{subj_id}_corr{hem}.npy"
    np.save(out_path, corr)

    meta = {
        "script": str(Path(__file__).resolve()),
        "subj_id": subj_id,
        "subject": subj,
        "group": group,
        "name": name,
        "size": size,
        "hemisphere": hem,
        "layer": layer,
        "n_heads": n_head,
        "cv_mode": CV_MODE,
        "folds": [{"n_train_sentences": int(len(train)), "n_test_sentences": int(len(test))}
                  for train, test in folds],
        "bold_aggregation": BOLD_AGGREGATION,
        "fmri_zscore_axis": FMRI_ZSCORE_AXIS,
        "mask_zero_y": MASK_ZERO_Y,
        "show_progress": SHOW_PROGRESS,
        "subtract_trivial_patterns": SUBTRACT_TRIVIAL_PATTERNS,
        "trivial_patterns": list(TRIVIAL_PATTERNS),
        "trivial_pattern_diagnostics": LAST_TRIVIAL_PATTERN_DIAGNOSTICS,
        "alphas": ALPHAS.tolist(),
        "attention_path": str(layer_attn_path),
        "output_path": str(out_path),
        "elapsed_seconds": elapsed,
        "summary": {
            "shape": list(corr.shape),
            "mean": float(np.nanmean(corr)),
            "median": float(np.nanmedian(corr)),
            "max": float(np.nanmax(corr)),
            "min": float(np.nanmin(corr)),
        },
    }
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Time taken: {elapsed:.6f} seconds")
    print(f"Saved: {out_path}")
    print(f"Saved metadata: {meta_path}")
    print(f"corr shape: {corr.shape}")
    print(
        f"mean={np.nanmean(corr):.8f}, median={np.nanmedian(corr):.8f}, "
        f"max={np.nanmax(corr):.8f}, min={np.nanmin(corr):.8f}"
    )


if __name__ == "__main__":
    main()
