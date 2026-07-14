#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
heads_vs_fmri.py

Robust ridge-regression script for aligning LLM attention heads with fMRI data.

Call pattern kept compatible with the original project:
    python3 heads_vs_fmri.py <subj_id> <group> <name> <size> <hem> <layer>

Example:
    python3 heads_vs_fmri.py 1 base llama 7B lh 14
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import zscore, pearsonr, ConstantInputWarning
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=ConstantInputWarning)


# ============================================================
# 0. Project paths
# ============================================================
PROJECT_ROOT = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main")
RESULTS_ROOT = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results")
EVENTS_BASE_DIR = Path("/media/i9a2/WindowsData/Wsh/DeepPrep/input")

os.chdir(PROJECT_ROOT)


# ============================================================
# 1. Static model specs
# ============================================================
MODEL_SPECS = {
    "gpt2_base":      {"layers": 12, "heads": 12},
    "gpt2_medium":    {"layers": 24, "heads": 16},
    "gpt2_large":     {"layers": 36, "heads": 20},
    "gpt2_xlarge":    {"layers": 48, "heads": 25},

    "llama_7B":       {"layers": 32, "heads": 32},
    "llama_13B":      {"layers": 40, "heads": 40},
    "llama_30B":      {"layers": 60, "heads": 52},
    "llama_65B":      {"layers": 80, "heads": 64},

    "alpaca_7B":      {"layers": 32, "heads": 32},
    "alpaca_13B":     {"layers": 40, "heads": 40},

    # 注意：原魔改代码里 vicuna_7B 重复写了两次，第二次会覆盖成 40，这是明显错误。
    "vicuna_7B":      {"layers": 32, "heads": 32},
    "vicuna_13B":     {"layers": 40, "heads": 40},

    "mistral_7B":     {"layers": 32, "heads": 32},
    "gemma_7B":       {"layers": 28, "heads": 16},
}

GROUP_TO_ATTN_PREFIX = {
    "base": "rb_p1",
    "instr": "instr_rb_p1",
    "ctrl": "ctrl_rb_p1",
}

EXCLUDE_SUBJECTS = {21, 52}
RM_SENTENCE_IDXS = [142, 143, 147, 148, 152, 153, 154]
SNTS_LIST = [31, 31, 28, 28, 30]
CUMS = np.cumsum(SNTS_LIST)


# ============================================================
# 2. Utilities
# ============================================================
def usage_and_exit() -> None:
    print(
        "Usage: python3 heads_vs_fmri.py <subj_id> <group> <name> <size> <hem> <layer>\n"
        "Example: python3 heads_vs_fmri.py 1 base llama 7B lh 14",
        file=sys.stderr,
    )
    sys.exit(2)


def parse_args():
    if len(sys.argv) != 7:
        usage_and_exit()

    subj_id = int(sys.argv[1])
    group = sys.argv[2]
    name = sys.argv[3]
    size = sys.argv[4]
    hem = sys.argv[5]
    layer = int(sys.argv[6])

    if hem not in {"lh", "rh"}:
        raise ValueError(f"hem must be 'lh' or 'rh', got {hem!r}")

    if group not in GROUP_TO_ATTN_PREFIX:
        raise ValueError(f"Unknown group {group!r}. Valid groups: {sorted(GROUP_TO_ATTN_PREFIX)}")

    key = f"{name}_{size}"
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model spec {key!r}. Please add it to MODEL_SPECS.")

    spec = MODEL_SPECS[key]
    if not (0 <= layer < spec["layers"]):
        raise ValueError(
            f"Layer out of range for {key}: got layer={layer}, "
            f"expected 0..{spec['layers'] - 1}"
        )

    return subj_id, group, name, size, hem, layer, key, spec


def subj_label(subj_id: int) -> str:
    return f"sub-0{subj_id}" if subj_id < 10 else f"sub-{subj_id}"


def safe_zscore(arr: np.ndarray, axis=None) -> np.ndarray:
    return np.nan_to_num(zscore(arr, axis=axis, nan_policy="omit"))


def find_attention_file(name: str, size: str, group: str, layer: int) -> Path:
    """
    Use group-specific prefix if present. For strict original base replication, base is rb_p1.
    """
    attn_dir = PROJECT_ROOT / "model_attention" / name / size
    prefix = GROUP_TO_ATTN_PREFIX[group]
    path = attn_dir / f"{prefix}_layer{layer}.npy"

    if not path.exists():
        # Fallback only for non-base experiments if you intentionally keep a single base attention set.
        fallback = attn_dir / f"rb_p1_layer{layer}.npy"
        raise FileNotFoundError(
            f"Attention file not found: {path}\n"
            f"Fallback candidate also available? {fallback.exists()} -> {fallback}\n"
            "If this is a base run, expected rb_p1_layer*.npy. "
            "If this is instr/ctrl, make sure the corresponding attention files exist."
        )

    return path


def load_attention_matrix(name: str, size: str, group: str, layer: int, words_list, spec) -> np.ndarray:
    """
    Expected raw layer_attn shape:
        [article_or_block, sentence_in_article, head, max_word, max_word]

    Original logic:
        attn = layer_attn.swapaxes(2, 0)
        X = np.array([np.concatenate(i, axis=0) for i in attn])
        X = np.delete(X, rm_id, axis=1)

    Important:
        Do NOT pad to 17.
        The correct check is current_dim >= max(words_list).
        In your debug result, max(words_list)=16 and LLaMA-7B dim=16, so that is correct.
    """
    path = find_attention_file(name, size, group, layer)
    layer_attn = np.load(path)

    if layer_attn.ndim != 5:
        raise ValueError(f"{path} should be 5D, got shape={layer_attn.shape}")

    if layer_attn.shape[-1] != layer_attn.shape[-2]:
        raise ValueError(f"{path} last two dims should be square, got shape={layer_attn.shape}")

    current_dim = layer_attn.shape[-1]
    max_words = max(words_list)

    if current_dim < max_words:
        raise ValueError(
            f"True attention/word mismatch for {name}_{size} layer{layer}: "
            f"attention dim={current_dim}, but max(words_list)={max_words}. "
            "Do not pad here; fix upstream attention extraction."
        )

    raw_heads = layer_attn.shape[2]
    if raw_heads != spec["heads"]:
        raise ValueError(
            f"Head mismatch before swapaxes for {name}_{size} layer{layer}: "
            f"attention has {raw_heads}, expected {spec['heads']}. "
            f"Raw shape={layer_attn.shape}"
        )

    if layer == 0:
        print(
            f"✅ Attention check: {path}\n"
            f"   raw shape={layer_attn.shape}, attention_dim={current_dim}, max_words={max_words}, "
            f"heads={raw_heads}; no padding applied."
        )

    return layer_attn


def build_X(layer_attn: np.ndarray, words_list, spec) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X_train: [n_head, n_train_edges]
        X_test:  [n_head, n_test_edges]
    """
    attn = layer_attn.swapaxes(2, 0)
    X = np.array([np.concatenate(i, axis=0) for i in attn])
    X = np.delete(X, RM_SENTENCE_IDXS, axis=1)

    if X.shape[0] != spec["heads"]:
        raise ValueError(f"X head mismatch: got {X.shape[0]}, expected {spec['heads']}")

    if X.shape[1] != len(words_list):
        raise ValueError(f"Sentence mismatch after rm_id: X has {X.shape[1]}, words_list has {len(words_list)}")

    if X.shape[1] != sum(SNTS_LIST):
        raise ValueError(f"Sentence mismatch after rm_id: X has {X.shape[1]}, expected {sum(SNTS_LIST)}")

    if X.shape[-1] < max(words_list):
        raise ValueError(f"X max word dim={X.shape[-1]} < max(words_list)={max(words_list)}")

    # Keep original author's standardization logic for reproducibility.
    X = safe_zscore(X.flatten(), axis=None).reshape(X.shape)

    X_train, X_test = [], []
    for i in range(X.shape[1]):
        n_word = int(words_list[i])
        tril_idx = np.tril_indices(n_word, k=-1)
        X_snt = X[:, i, :n_word, :n_word]

        if i < 133:
            X_train.append(X_snt[:, tril_idx[0], tril_idx[1]])
        else:
            X_test.append(X_snt[:, tril_idx[0], tril_idx[1]])

    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    total_edges = sum(int(w) * (int(w) - 1) // 2 for w in words_list)
    if X_train.shape[1] + X_test.shape[1] != total_edges:
        raise ValueError(
            f"Lower-triangle edge count mismatch: train+test="
            f"{X_train.shape[1] + X_test.shape[1]}, expected={total_edges}"
        )

    # Paper reports 7,388 for all 148 sentences after removing rm_id.
    if total_edges != 7388:
        print(
            f"⚠️ Warning: lower-triangle total is {total_edges}, not 7388. "
            "If you changed stimuli/words_list this may be expected."
        )

    print(
        f"✅ X built: X_train={X_train.shape}, X_test={X_test.shape}, "
        f"total_edges={X_train.shape[1] + X_test.shape[1]}"
    )

    return X_train, X_test


def load_events_and_surfaces(subj: str, hem: str):
    surf = []
    events_lst = []

    for article in range(1, 6):
        run = 1
        event_file = EVENTS_BASE_DIR / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv"

        if not event_file.exists():
            raise FileNotFoundError(f"Event file not found: {event_file}")

        events = pd.read_csv(event_file, delimiter="\t").dropna().reset_index(drop=True)
        run_article = int(events.SentenceID.iloc[0].split(".")[1])

        while run_article != article and run < 5:
            run += 1
            event_file = EVENTS_BASE_DIR / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv"
            if not event_file.exists():
                raise FileNotFoundError(f"Event file not found: {event_file}")

            events = pd.read_csv(event_file, delimiter="\t").dropna().reset_index(drop=True)
            run_article = int(events.SentenceID.iloc[0].split(".")[1])

        if run_article != article:
            raise RuntimeError(f"Could not find run for article {article} in {subj}")

        events["article"] = article
        events_lst.append(events)

        fmri_file = PROJECT_ROOT / "Data" / "fsaverage5" / (
            f"{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii"
        )
        if not fmri_file.exists():
            raise FileNotFoundError(f"fMRI surface file not found: {fmri_file}")

        fmri_gii = nib.load(str(fmri_file))
        fmri_data = np.column_stack([arr.data for arr in fmri_gii.darrays])
        fmri_data = safe_zscore(fmri_data, axis=0)
        surf.append(fmri_data)

        print(f"✅ Article {article}: run={run}, events={len(events)}, fmri={fmri_data.shape}")

    events_all = pd.concat(events_lst).reset_index(drop=True)
    return events_all, surf


def process_vertex(v, X_train, X_test, events, surf, words, words_list):
    y_train, y_test = [], []

    for article in range(1, 6):
        n_snts = int(
            words[words.SentenceID.str.match(f"t.0{article}")]
            .SentenceID.iloc[-1]
            .split(".")[-1]
        )

        for i in range(1, n_snts + 1):
            snt_id = f"t.0{article}.0{i}" if i < 10 else f"t.0{article}.{i}"
            event = events[events.SentenceID.str.match(snt_id)].reset_index(drop=True)

            sid = i - 1 if article == 1 else CUMS[article - 2] + i - 1
            n_word = int(words_list[sid])

            fmri_snt = np.zeros((n_word, n_word), dtype=np.float32)
            tril_idx = np.tril_indices(n_word, k=-1)

            for ind, e in event.iterrows():
                if ind >= len(event) - 1:
                    continue

                row = int(e.CURRENT_FIX_INTEREST_AREA_ID) - 1
                col = int(event.iloc[ind + 1].CURRENT_FIX_INTEREST_AREA_ID) - 1

                # Keep original author's row=current, col=next logic.
                # Lower triangle therefore corresponds to right-to-left / regressive movement
                # under the paper's row/column convention.
                if not (0 <= row < n_word and 0 <= col < n_word):
                    continue

                scan = int(np.ceil((event.iloc[ind + 1].onset / 1000 + 5) / 0.4))
                while scan >= surf[article - 1].shape[1]:
                    scan -= 1

                fmri_snt[row, col] = surf[article - 1][v, scan]

            if sid < 133:
                y_train.extend(fmri_snt[tril_idx[0], tril_idx[1]])
            else:
                y_test.extend(fmri_snt[tril_idx[0], tril_idx[1]])

    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    if X_train.shape[1] != y_train.shape[0]:
        raise ValueError(f"Train sample mismatch: X={X_train.shape[1]}, y={y_train.shape[0]}")
    if X_test.shape[1] != y_test.shape[0]:
        raise ValueError(f"Test sample mismatch: X={X_test.shape[1]}, y={y_test.shape[0]}")

    model_train = RidgeCV(alphas=np.logspace(1, 3, 20)).fit(X_train.T, y_train)

    y_predict = X_test.T @ model_train.coef_

    mask = y_test != 0

    if mask.sum() > 5:
       r, _ = pearsonr(y_predict[mask], y_test[mask])
    else:
       r = 0

    return float(np.nan_to_num(r))


# ============================================================
# 3. Main
# ============================================================
def main() -> int:
    subj_id, group, name, size, hem, layer, key, spec = parse_args()

    if subj_id in EXCLUDE_SUBJECTS:
        print(f"🚫 sub-{subj_id:02d} is excluded. Terminating safely.")
        return 0

    subj = subj_label(subj_id)

    words = pd.read_csv(PROJECT_ROOT / "Analysis" / "words.csv")
    with open(PROJECT_ROOT / "Analysis" / "words_list.p", "rb") as f:
        words_list = pickle.load(f)

    if len(words_list) != sum(SNTS_LIST):
        raise ValueError(f"words_list has {len(words_list)} sentences, expected {sum(SNTS_LIST)}")

    print(
        f"\n🚀 Start ridge: subj={subj}, model={key}, group={group}, hemi={hem}, layer={layer}\n"
        f"   words_list: n_sentence={len(words_list)}, max_words={max(words_list)}"
    )

    layer_attn = load_attention_matrix(name, size, group, layer, words_list, spec)
    X_train, X_test = build_X(layer_attn, words_list, spec)

    events, surf = load_events_and_surfaces(subj, hem)

    start = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(process_vertex)(v, X_train, X_test, events, surf, words, words_list)
        for v in range(10242)
    )
    corr = np.asarray(results, dtype=np.float32)
    elapsed = time.time() - start

    print(f"✅ Ridge finished. Time taken: {elapsed:.6f} seconds")
    print(f"   corr shape={corr.shape}, mean={np.nanmean(corr):.6f}, max={np.nanmax(corr):.6f}")

    save_dir = RESULTS_ROOT / name / size / group / f"layer{layer}"
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{group}_subj{subj_id}_corr{hem}.npy"
    np.save(save_path, corr)
    print(f"💾 Saved: {save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
