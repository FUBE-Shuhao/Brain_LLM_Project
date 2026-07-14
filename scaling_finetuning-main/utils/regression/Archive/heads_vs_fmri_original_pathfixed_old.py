#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path-fixed version of the original heads_vs_fmri.py.

Goal:
- Keep the original algorithm as much as possible.
- Only adapt paths to the local Brain_LLM_Project layout.
- Keep the original zero-filling behavior in fmri_snt.
- Save one .npy per layer, so it can be checked by the existing layer-wise result scripts.

Usage example:
  python heads_vs_fmri_original_pathfixed.py 1 base llama 7B lh 25

Run all llama-7B-base lh layers for subject 1:
  for L in {0..31}; do python heads_vs_fmri_original_pathfixed.py 1 base llama 7B lh $L; done
"""

import os
import sys
import pickle
import time
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

# =========================
# Local path configuration
# =========================
PROJECT_DIR = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main")
BRAIN_PROJECT_DIR = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project")

# The original script used relative paths under /scratch/.../readbrain:
#   Analysis/words.csv
#   Analysis/words_list.p
#   Analysis/attns/{name}/{size}/p1/rb_p1_layer{layer}.npy
#   Data/{group}/{subj}/func/*events.tsv
#   Data/fsaverage5/*.func.gii
ANALYSIS_DIR = PROJECT_DIR / "Analysis"
DATA_DIR = PROJECT_DIR / "Data"
ATTN_DIR = PROJECT_DIR / "model_attention"

# Save to a separate result root by default, to avoid overwriting your current 4_Results.
# If you want to overwrite/use the normal result folder, run with:
#   OUTPUT_ROOT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results python ...
OUTPUT_ROOT = Path(os.environ.get(
    "OUTPUT_ROOT",
    str(BRAIN_PROJECT_DIR / "4_Results_original_heads_vs_fmri")
))

# Number of parallel jobs. Use N_JOBS=8 python ... if you want to limit CPU usage.
N_JOBS = int(os.environ.get("N_JOBS", "-1"))


def existing_path(*candidates: Path) -> Path:
    """Return the first existing path; raise a clear error otherwise."""
    for p in candidates:
        if p.exists():
            return p
    msg = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"None of these paths exists:\n{msg}")


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

    subj = "sub-0%d" % subj_id if subj_id < 10 else "sub-%d" % subj_id

    model_size_heads = {
        "gpt2_base": 12,
        "gpt2_medium": 16,
        "gpt2_large": 20,
        "gpt2_xlarge": 25,
        "llama_7B": 32,
        "llama_13B": 40,
        "llama_30B": 52,
        "llama_65B": 64,
        "alpaca_7B": 32,
        "alpaca_13B": 40,
        "vicuna_7B": 32,
        "vicuna_13B": 40,
        "mistral_7B": 32,
        "gemma_7B": 16,
    }
    model_key = "%s_%s" % (name, size)
    n_head = model_size_heads[model_key]

    print("=" * 80)
    print("Original heads_vs_fmri path-fixed run")
    print(f"Project dir : {PROJECT_DIR}")
    print(f"Data dir    : {DATA_DIR}")
    print(f"Analysis dir: {ANALYSIS_DIR}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Target      : subj={subj}, group={group}, model={model_key}, hemi={hem}, layer={layer}, heads={n_head}")
    print("=" * 80)

    rm_id = [142, 143, 147, 148, 152, 153, 154]

    # Prefer the original Analysis/ location, but allow fallback to project root if you placed files there.
    words_csv = existing_path(ANALYSIS_DIR / "words.csv", PROJECT_DIR / "words.csv")
    words_list_p = existing_path(ANALYSIS_DIR / "words_list.p", PROJECT_DIR / "words_list.p")

    words = pd.read_csv(words_csv)
    with open(words_list_p, "rb") as f:
        words_list = pickle.load(f)

    snts_list = [31, 31, 28, 28, 30]
    cums = np.cumsum(snts_list)

    layer_attn_path = ATTN_DIR / name / size / f"rb_p1_layer{layer}.npy"
    if not layer_attn_path.exists():
        raise FileNotFoundError(f"Attention file not found: {layer_attn_path}")

    print(f"Loading attention: {layer_attn_path}")
    layer_attn = np.load(layer_attn_path)

    # Original algorithm starts here.
    attn = layer_attn.swapaxes(2, 0)
    X = np.array([np.concatenate(i, axis=0) for i in attn])
    X = np.delete(X, rm_id, axis=1)
#添加了精度修改，先转换为float64
    X = X.astype(np.float64, copy=False)
    X = np.nan_to_num(zscore(X.flatten(), nan_policy="omit")).reshape(X.shape)

    X_train, X_test = [], []
    for i in range(X.shape[1]):
        tril_idx = np.tril_indices(words_list[i], k=-1)
        if i < 133:
            X_snt_train = X[:, i, : words_list[i], : words_list[i]]
            X_train.append(X_snt_train[:, tril_idx[0], tril_idx[1]])
        else:
            X_snt_test = X[:, i, : words_list[i], : words_list[i]]
            X_test.append(X_snt_test[:, tril_idx[0], tril_idx[1]])
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    print(f"X_train shape: {X_train.shape}  # expected roughly n_head x train_lower_tri")
    print(f"X_test  shape: {X_test.shape}   # expected roughly n_head x test_lower_tri")

    surf = []
    events_lst = []
    for article in range(1, 6):
        run = 1
        events_path = DATA_DIR / group / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv"
        events = pd.read_csv(events_path, delimiter="\t")
        events = events.dropna().reset_index(drop=True)
        run_article = int(events.SentenceID[0].split(".")[1])
        while run_article != article and run < 5:
            run += 1
            events_path = DATA_DIR / group / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv"
            events = pd.read_csv(events_path, delimiter="\t")
            events = events.dropna().reset_index(drop=True)
            run_article = int(events.SentenceID[0].split(".")[1])

        events["article"] = [article] * len(events)
        events_lst.append(events)

        fmri_path = DATA_DIR / "fsaverage5" / f"{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii"
        if not fmri_path.exists():
            raise FileNotFoundError(f"fMRI surface file not found: {fmri_path}")
        print(f"Article {article}: using run {run}, fmri={fmri_path.name}, events={events_path.name}")

        fmri_gii = nib.load(str(fmri_path))
        fmri_data = np.column_stack([arr.data for arr in fmri_gii.darrays])
        fmri_data = np.nan_to_num(zscore(fmri_data, axis=0, nan_policy="omit"))
        surf.append(fmri_data)

    events = pd.concat(events_lst).reset_index(drop=True)
    corr = np.zeros(10242)

    def process_vertex(v, X_train, X_test, events, surf, words):
        y_train, y_test = [], []
        for article in range(1, 6):
            n_snts = int(
                words[words.SentenceID.str.match("t.0%d" % article)]
                .SentenceID.iloc[-1]
                .split(".")[-1]
            )
            for i in range(1, n_snts + 1):
                snt_id = "t.0%d.0%d" % (article, i) if i < 10 else "t.0%d.%d" % (article, i)
                event = events[events.SentenceID.str.match(snt_id)]
                sid = i - 1 if article == 1 else cums[article - 2] + i - 1
                fmri_snt = np.zeros((words_list[sid], words_list[sid]))
                tril_idx = np.tril_indices(words_list[sid], k=-1)
                event = event.reset_index(drop=True)
                for ind, e in event.iterrows():
                    if ind < len(event) - 1:
                        row = int(e.CURRENT_FIX_INTEREST_AREA_ID) - 1
                        col = int(event.iloc[ind + 1].CURRENT_FIX_INTEREST_AREA_ID) - 1
                        if row < 0 or row >= words_list[sid] or col < 0 or col >= words_list[sid]:
                            continue
                        scan = int(np.ceil((event.iloc[ind + 1].onset / 1000 + 5) / 0.4))
                        while scan >= surf[article - 1].shape[1]:
                            scan -= 1
                        fmri_snt[row, col] = surf[article - 1][v, scan]
                if sid < 133:
                    y_train.extend(fmri_snt[tril_idx[0], tril_idx[1]])
                else:
                    y_test.extend(fmri_snt[tril_idx[0], tril_idx[1]])

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        model_train = RidgeCV(alphas=np.logspace(1, 3, 20)).fit(X_train.T, y_train)
        y_predict = X_test.T @ model_train.coef_
        corr[v], _ = pearsonr(y_predict, y_test)
        corr[v] = np.nan_to_num(corr[v])
        return corr[v]

    start = time.time()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_vertex)(v, X_train, X_test, events, surf, words) for v in range(10242)
    )
    corr = np.array(results)
    end = time.time()
    elapsed = end - start

    out_dir = OUTPUT_ROOT / name / size / group / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group}_subj{subj_id}_corr{hem}.npy"
    np.save(out_path, corr)

    print(f"Time taken: {elapsed:.6f} seconds")
    print(f"Saved: {out_path}")
    print(f"corr shape: {corr.shape}")
    print(f"mean={np.nanmean(corr):.8f}, median={np.nanmedian(corr):.8f}, max={np.nanmax(corr):.8f}, min={np.nanmin(corr):.8f}")


if __name__ == "__main__":
    main()
