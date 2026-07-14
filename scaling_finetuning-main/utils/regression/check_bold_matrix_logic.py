#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate BOLD-matrix construction choices for the fMRI ridge pipeline.

This script does not run ridge regression. It checks the assumptions used by
heads_vs_fmri_original_pathfixed.py:
  - scan = ceil((next_fixation_onset / 1000 + HRF_DELAY_SECONDS) / TR_SECONDS)
  - repeated word transitions are aggregated with sum, not overwritten by last
  - lower-triangle BOLD vectors have 7,388 entries

Usage from project root:
  python utils/regression/check_bold_matrix_logic.py 4 base lh

Optional environment variables:
  PROJECT_DIR=~/repository/scaling_finetuning-main
  ANALYSIS_DIR=~/repository/scaling_finetuning-main/Analysis
  DATA_DIR=~/repository/scaling_finetuning-main/Data
  TR_SECONDS=0.4
  HRF_DELAY_SECONDS=5.0
  FMRI_ZSCORE_AXIS=0
  SAMPLE_VERTICES=0,1024,4096,8192
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore


EXPECTED_FSAVERAGE5_VERTICES = 10242
EXPECTED_SENTENCE_COUNTS = [31, 31, 28, 28, 30]
EXPECTED_N_SENTENCES = sum(EXPECTED_SENTENCE_COUNTS)
EXPECTED_LOWER_TRI_EDGES = 7388
CUMS = np.cumsum(EXPECTED_SENTENCE_COUNTS)
SENTENCE_ID_RE = re.compile(r"^t\.0([1-5])\.0?([1-9][0-9]*)$")


def existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exists:\n" + "\n".join(str(p) for p in candidates))


def sentence_id(article: int, sentence: int) -> str:
    return f"t.0{article}.0{sentence}" if sentence < 10 else f"t.0{article}.{sentence}"


def normalize_sentence_id(value: object) -> str:
    value = str(value).strip()
    match = SENTENCE_ID_RE.match(value)
    if not match:
        return value
    article = int(match.group(1))
    sentence = int(match.group(2))
    return sentence_id(article, sentence)


def split_sentence_id(sid: str) -> tuple[int, int]:
    sid = normalize_sentence_id(sid)
    match = SENTENCE_ID_RE.match(sid)
    if not match:
        raise ValueError(f"Invalid SentenceID: {sid!r}")
    return int(match.group(1)), int(match.group(2))


def global_sentence_index(article: int, sentence: int) -> int:
    return sentence - 1 if article == 1 else int(CUMS[article - 2]) + sentence - 1


def safe_zscore(arr: np.ndarray, axis=None) -> np.ndarray:
    return np.nan_to_num(zscore(arr, axis=axis, nan_policy="omit"))


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def parse_sample_vertices(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def selected_run_for_article(run_events: dict[int, pd.DataFrame], article: int) -> int:
    for run in range(1, 6):
        first_sid = normalize_sentence_id(run_events[run]["SentenceID"].iloc[0])
        first_article, _ = split_sentence_id(first_sid)
        if first_article == article:
            return run
    raise RuntimeError(f"Could not find selected run for article {article}")


def load_events(data_dir: Path, group: str, subj: str) -> tuple[pd.DataFrame, dict[int, int]]:
    run_events = {}
    for run in range(1, 6):
        path = data_dir / group / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv"
        if not path.exists():
            fail(f"missing event file: {path}")
        events = pd.read_csv(path, delimiter="\t").dropna().reset_index(drop=True)
        for col in ["SentenceID", "CURRENT_FIX_INTEREST_AREA_ID", "onset"]:
            if col not in events.columns:
                fail(f"{path} missing required column {col!r}")
        events["SentenceID"] = events["SentenceID"].map(normalize_sentence_id)
        events["run"] = run
        run_events[run] = events

    article_to_run = {article: selected_run_for_article(run_events, article) for article in range(1, 6)}
    selected = []
    for article, run in article_to_run.items():
        ev = run_events[run].copy()
        ev = ev[ev["SentenceID"].map(lambda sid: split_sentence_id(sid)[0]) == article].copy()
        selected.append(ev)
    return pd.concat(selected, ignore_index=True), article_to_run


def sidecar_tr_candidates(fmri_path: Path) -> list[tuple[Path, object]]:
    candidates = []
    sibling_jsons = [
        fmri_path.with_suffix("").with_suffix(".json"),
        fmri_path.with_suffix(".json"),
    ]
    for path in sibling_jsons:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for key in ["RepetitionTime", "TR", "SamplingFrequency"]:
            if key in data:
                candidates.append((path, {key: data[key]}))
    return candidates


def load_surfaces(data_dir: Path, subj: str, article_to_run: dict[int, int], hem: str, fmri_zscore_axis: int) -> dict[int, np.ndarray]:
    surf = {}
    print()
    print("fMRI files:")
    for article in range(1, 6):
        run = article_to_run[article]
        path = data_dir / "fsaverage5" / f"{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii"
        if not path.exists():
            fail(f"missing fMRI surface file: {path}")
        gii = nib.load(str(path))
        data = np.column_stack([arr.data for arr in gii.darrays])
        if data.shape[0] != EXPECTED_FSAVERAGE5_VERTICES:
            fail(f"{path.name}: expected {EXPECTED_FSAVERAGE5_VERTICES} vertices, got {data.shape[0]}")
        print(f"  article {article}, run {run}: {path.name}, shape={data.shape}")
        tr_sidecars = sidecar_tr_candidates(path)
        if tr_sidecars:
            print(f"    sidecar TR candidates: {tr_sidecars}")
        surf[article] = safe_zscore(data, axis=fmri_zscore_axis)
    ok("all fMRI surfaces loaded and z-scored with the requested axis")
    return surf


def transition_records(events: pd.DataFrame, words_list: list[int], tr_seconds: float, hrf_delay_seconds: float):
    records = []
    per_sentence_counts = []
    duplicate_cells_by_sentence = []
    scan_by_article = defaultdict(list)
    repeated_examples = []

    for article, n_sentences in enumerate(EXPECTED_SENTENCE_COUNTS, start=1):
        for sent in range(1, n_sentences + 1):
            sid = sentence_id(article, sent)
            gidx = global_sentence_index(article, sent)
            n_word = int(words_list[gidx])
            ev = events[events["SentenceID"] == sid].reset_index(drop=True)
            if ev.empty:
                fail(f"missing events for {sid}")

            ids = pd.to_numeric(ev["CURRENT_FIX_INTEREST_AREA_ID"], errors="coerce").astype(int).to_numpy()
            onsets = pd.to_numeric(ev["onset"], errors="coerce").to_numpy(dtype=float)
            cell_counter = Counter()

            for ind in range(len(ev) - 1):
                row = int(ids[ind]) - 1
                col = int(ids[ind + 1]) - 1
                if row < 0 or row >= n_word or col < 0 or col >= n_word:
                    fail(f"{sid}: transition out of range row={row}, col={col}, n_word={n_word}")
                scan = int(np.ceil((float(onsets[ind + 1]) / 1000.0 + hrf_delay_seconds) / tr_seconds))
                records.append((article, sid, gidx, n_word, row, col, scan))
                scan_by_article[article].append(scan)
                cell_counter[(row, col)] += 1

            repeated = {cell: count for cell, count in cell_counter.items() if count > 1}
            duplicate_cells_by_sentence.append(len(repeated))
            per_sentence_counts.append(len(ev) - 1)
            if repeated and len(repeated_examples) < 12:
                repeated_examples.append((sid, n_word, repeated))

    return {
        "records": records,
        "per_sentence_counts": per_sentence_counts,
        "duplicate_cells_by_sentence": duplicate_cells_by_sentence,
        "scan_by_article": scan_by_article,
        "repeated_examples": repeated_examples,
    }


def build_bold_vectors_for_vertex(records, surf: dict[int, np.ndarray], vertex: int, aggregation: str) -> list[np.ndarray]:
    by_sentence = []
    rec_idx = 0
    for article, n_sentences in enumerate(EXPECTED_SENTENCE_COUNTS, start=1):
        for sent in range(1, n_sentences + 1):
            sid = sentence_id(article, sent)
            gidx = global_sentence_index(article, sent)
            n_word = None
            sentence_records = []
            while rec_idx < len(records) and records[rec_idx][1] == sid:
                sentence_records.append(records[rec_idx])
                n_word = records[rec_idx][3]
                rec_idx += 1
            if n_word is None:
                raise RuntimeError(f"No transition records for {sid}; cannot infer n_word")

            mat = np.zeros((n_word, n_word), dtype=np.float64)
            for rec in sentence_records:
                article_i, _, _, _, row, col, scan = rec
                if scan >= surf[article_i].shape[1]:
                    raise RuntimeError(f"scan out of range after validation: article={article_i}, scan={scan}")
                value = surf[article_i][vertex, scan]
                if aggregation == "sum":
                    mat[row, col] += value
                elif aggregation == "last":
                    mat[row, col] = value
                else:
                    raise ValueError(aggregation)
            tril_idx = np.tril_indices(n_word, k=-1)
            by_sentence.append(mat[tril_idx[0], tril_idx[1]])
    return by_sentence


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python utils/regression/check_bold_matrix_logic.py <subj_id> <group> <lh|rh>\n"
            "Example: python utils/regression/check_bold_matrix_logic.py 4 base lh",
            file=sys.stderr,
        )
        sys.exit(2)

    subj_id = int(sys.argv[1])
    group = sys.argv[2]
    hem = sys.argv[3]
    if hem not in {"lh", "rh"}:
        raise ValueError(f"hem must be lh or rh, got {hem!r}")

    subj = f"sub-0{subj_id}" if subj_id < 10 else f"sub-{subj_id}"
    project_dir = Path(os.environ.get("PROJECT_DIR", ".")).expanduser().resolve()
    analysis_dir = Path(os.environ.get("ANALYSIS_DIR", str(project_dir / "Analysis"))).expanduser().resolve()
    data_dir = Path(os.environ.get("DATA_DIR", str(project_dir / "Data"))).expanduser().resolve()
    tr_seconds = float(os.environ.get("TR_SECONDS", "0.4"))
    hrf_delay_seconds = float(os.environ.get("HRF_DELAY_SECONDS", "5.0"))
    fmri_zscore_axis = int(os.environ.get("FMRI_ZSCORE_AXIS", "0"))
    sample_vertices = parse_sample_vertices(os.environ.get("SAMPLE_VERTICES", "0,1024,4096,8192"))

    print("Checking BOLD matrix construction")
    print(f"PROJECT_DIR       : {project_dir}")
    print(f"ANALYSIS_DIR      : {analysis_dir}")
    print(f"DATA_DIR          : {data_dir}")
    print(f"subject/group/hem : {subj} / {group} / {hem}")
    print(f"TR_SECONDS        : {tr_seconds}")
    print(f"HRF_DELAY_SECONDS : {hrf_delay_seconds}")
    print(f"FMRI_ZSCORE_AXIS  : {fmri_zscore_axis}")
    print(f"SAMPLE_VERTICES   : {sample_vertices}")

    words_list_p = existing_path(analysis_dir / "words_list.p", project_dir / "words_list.p")
    with open(words_list_p, "rb") as f:
        words_list = [int(n) for n in pickle.load(f)]
    if len(words_list) != EXPECTED_N_SENTENCES:
        fail(f"words_list length is {len(words_list)}, expected {EXPECTED_N_SENTENCES}")
    edges = sum(n * (n - 1) // 2 for n in words_list)
    if edges != EXPECTED_LOWER_TRI_EDGES:
        fail(f"words_list lower-triangle edges is {edges}, expected {EXPECTED_LOWER_TRI_EDGES}")
    ok("words_list gives 148 sentences and 7,388 lower-triangle entries")

    events, article_to_run = load_events(data_dir, group, subj)
    print()
    print("article -> selected run:")
    for article in range(1, 6):
        print(f"  article {article}: run {article_to_run[article]}")

    rec_info = transition_records(events, words_list, tr_seconds, hrf_delay_seconds)
    records = rec_info["records"]
    if not records:
        fail("no fixation transitions found")

    scan_by_article = rec_info["scan_by_article"]
    print()
    print("scan index summary from ceil((next_onset/1000 + delay) / TR):")
    for article in range(1, 6):
        scans = np.array(scan_by_article[article], dtype=int)
        print(
            f"  article {article}: n={len(scans)}, "
            f"min/median/max={scans.min()} / {np.median(scans):.1f} / {scans.max()}"
        )

    onset_values = pd.to_numeric(events["onset"], errors="coerce").to_numpy(dtype=float)
    print()
    print("onset magnitude:")
    print(f"  min/median/max ms-like values: {onset_values.min():.3f} / {np.median(onset_values):.3f} / {onset_values.max():.3f}")
    if np.median(onset_values) < 100:
        warn("onset median is <100; core formula assumes milliseconds, so verify source units manually")
    else:
        ok("onset values are ms-scale, consistent with onset/1000")

    print()
    print("transition/repetition summary:")
    print(f"  total candidate transitions: {len(records)}")
    print(
        "  transitions per sentence min/median/max: "
        f"{min(rec_info['per_sentence_counts'])} / "
        f"{np.median(rec_info['per_sentence_counts']):.1f} / "
        f"{max(rec_info['per_sentence_counts'])}"
    )
    repeated_sentence_count = sum(1 for n in rec_info["duplicate_cells_by_sentence"] if n > 0)
    repeated_cell_count = sum(rec_info["duplicate_cells_by_sentence"])
    print(f"  sentences with repeated word-transition cells: {repeated_sentence_count}")
    print(f"  repeated cells across all sentences: {repeated_cell_count}")
    if repeated_cell_count > 0:
        ok("repeated transitions exist, so sum and last aggregation are not equivalent")
        print("  repeated examples:")
        for sid, n_word, repeated in rec_info["repeated_examples"]:
            print(f"    {sid}, n_word={n_word}, repeated={dict(repeated)}")
    else:
        warn("no repeated transitions found; sum and last would be equivalent for this subject/events")

    surf = load_surfaces(data_dir, subj, article_to_run, hem, fmri_zscore_axis)
    for article in range(1, 6):
        max_scan = max(scan_by_article[article])
        n_timepoints = surf[article].shape[1]
        if max_scan >= n_timepoints:
            fail(f"article {article}: max_scan={max_scan} exceeds fMRI n_timepoints={n_timepoints}")
    ok("all scan indices are inside fMRI timepoints; no clamp/decrement is needed")

    print()
    print("sample vertex comparison: BOLD_AGGREGATION=sum vs last")
    for vertex in sample_vertices:
        if vertex < 0 or vertex >= EXPECTED_FSAVERAGE5_VERTICES:
            warn(f"skip invalid sample vertex {vertex}")
            continue
        y_sum = np.concatenate(build_bold_vectors_for_vertex(records, surf, vertex, "sum"))
        y_last = np.concatenate(build_bold_vectors_for_vertex(records, surf, vertex, "last"))
        if y_sum.shape[0] != EXPECTED_LOWER_TRI_EDGES:
            fail(f"vertex {vertex}: y_sum length={y_sum.shape[0]}, expected {EXPECTED_LOWER_TRI_EDGES}")
        diff = y_sum - y_last
        changed = int(np.count_nonzero(np.abs(diff) > 1e-12))
        corr = np.corrcoef(y_sum, y_last)[0, 1] if np.std(y_sum) > 0 and np.std(y_last) > 0 else np.nan
        print(
            f"  vertex {vertex}: len={len(y_sum)}, "
            f"nonzero_sum={np.count_nonzero(y_sum)}, nonzero_last={np.count_nonzero(y_last)}, "
            f"changed_cells={changed}, abs_diff_mean={np.mean(np.abs(diff)):.6g}, corr={corr:.6f}"
        )
    ok("sample BOLD vectors have expected 7,388 length")

    print()
    print("Method notes")
    print("  Paper states each BOLD matrix cell is the sum of BOLD signals at transition timepoints.")
    print("  Original backup script used the same +5s and 0.4s scan formula, but overwrote repeated cells.")
    print("  Current script uses BOLD_AGGREGATION=sum, matching the paper statement for repeated transitions.")
    print()
    print("Summary")
    print("  status: OK")


if __name__ == "__main__":
    main()
