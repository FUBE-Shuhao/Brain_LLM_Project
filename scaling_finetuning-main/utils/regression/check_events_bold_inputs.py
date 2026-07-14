#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate events.tsv inputs for BOLD-matrix construction in the fMRI ridge pipeline.

This checks the data assumptions used by heads_vs_fmri_original_pathfixed.py:
  - required event columns exist
  - each article is mapped to the same run-selection logic as the core script
  - SentenceID values match the expected 148 Reading Brain sentences
  - CURRENT_FIX_INTEREST_AREA_ID is a 1-based integer word index within words_list.p
  - onset values look like milliseconds and are ordered within each sentence
  - optional: scan indices from ceil((onset / 1000 + 5) / 0.4) fit inside fMRI .func.gii timepoints

Usage:
  python utils/regression/check_events_bold_inputs.py 4 base
  python utils/regression/check_events_bold_inputs.py 4 base lh

Optional environment variables:
  PROJECT_DIR=/path/to/scaling_finetuning-main
  ANALYSIS_DIR=/path/to/scaling_finetuning-main/Analysis
  DATA_DIR=/path/to/scaling_finetuning-main/Data
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SENTENCE_COUNTS = [31, 31, 28, 28, 30]
EXPECTED_N_SENTENCES = sum(EXPECTED_SENTENCE_COUNTS)
CUMS = np.cumsum(EXPECTED_SENTENCE_COUNTS)
REQUIRED_COLUMNS = ["SentenceID", "CURRENT_FIX_INTEREST_AREA_ID", "onset"]
SENTENCE_ID_RE = re.compile(r"^t\.0([1-5])\.0?([1-9][0-9]*)$")
TR_SECONDS = 0.4
HRF_DELAY_SECONDS = 5.0


def existing_path(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exists:\n" + "\n".join(str(p) for p in candidates))


def sentence_id(article: int, sentence: int) -> str:
    return f"t.0{article}.0{sentence}" if sentence < 10 else f"t.0{article}.{sentence}"


def expected_sentence_ids() -> list[str]:
    ids = []
    for article, n_sentences in enumerate(EXPECTED_SENTENCE_COUNTS, start=1):
        for sentence in range(1, n_sentences + 1):
            ids.append(sentence_id(article, sentence))
    return ids


def normalize_sentence_id(value: object) -> str:
    value = str(value).strip()
    match = SENTENCE_ID_RE.match(value)
    if not match:
        return value
    article = int(match.group(1))
    sentence = int(match.group(2))
    return sentence_id(article, sentence)


def article_from_sentence_id(value: object) -> int:
    sid = normalize_sentence_id(value)
    match = SENTENCE_ID_RE.match(sid)
    if not match:
        raise ValueError(f"Invalid SentenceID: {value!r}")
    return int(match.group(1))


def global_sentence_index(article: int, sentence: int) -> int:
    return sentence - 1 if article == 1 else int(CUMS[article - 2]) + sentence - 1


def split_sentence_id(sid: str) -> tuple[int, int]:
    sid = normalize_sentence_id(sid)
    match = SENTENCE_ID_RE.match(sid)
    if not match:
        raise ValueError(f"Invalid SentenceID: {sid!r}")
    return int(match.group(1)), int(match.group(2))


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[PASS] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def load_fmri_n_timepoints(data_dir: Path, subj: str, run: int, hem: str) -> int:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for fMRI timepoint checks") from exc

    fmri_path = data_dir / "fsaverage5" / f"{subj}_task-read_run-{run}_hemi-{hem}_space-fsaverage5_bold.func.gii"
    if not fmri_path.exists():
        raise FileNotFoundError(f"fMRI surface file not found: {fmri_path}")
    gii = nib.load(str(fmri_path))
    return len(gii.darrays)


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        print(
            "Usage: python utils/regression/check_events_bold_inputs.py <subj_id> <group> [lh|rh]\n"
            "Example: python utils/regression/check_events_bold_inputs.py 4 base lh",
            file=sys.stderr,
        )
        sys.exit(2)

    subj_id = int(sys.argv[1])
    group = sys.argv[2]
    hem = sys.argv[3] if len(sys.argv) == 4 else None
    if hem is not None and hem not in {"lh", "rh"}:
        raise ValueError(f"hem must be lh or rh, got {hem!r}")

    subj = f"sub-0{subj_id}" if subj_id < 10 else f"sub-{subj_id}"
    project_dir = Path(os.environ.get("PROJECT_DIR", ".")).resolve()
    analysis_dir = Path(os.environ.get("ANALYSIS_DIR", str(project_dir / "Analysis"))).resolve()
    data_dir = Path(os.environ.get("DATA_DIR", str(project_dir / "Data"))).resolve()

    words_list_p = existing_path(analysis_dir / "words_list.p", project_dir / "words_list.p")
    with open(words_list_p, "rb") as f:
        words_list = [int(n) for n in pickle.load(f)]

    if len(words_list) != EXPECTED_N_SENTENCES:
        fail(f"words_list length is {len(words_list)}, expected {EXPECTED_N_SENTENCES}")

    print("Checking events/BOLD inputs")
    print(f"PROJECT_DIR : {project_dir}")
    print(f"ANALYSIS_DIR: {analysis_dir}")
    print(f"DATA_DIR    : {data_dir}")
    print(f"subject     : {subj}")
    print(f"group       : {group}")
    print(f"hemisphere  : {hem if hem else '(not checking fMRI timepoints)'}")
    print(f"words_list.p: {words_list_p}")
    print()

    run_events: dict[int, pd.DataFrame] = {}
    for run in range(1, 6):
        path = data_dir / group / subj / "func" / f"{subj}_task-read_run-{run}_events.tsv"
        if not path.exists():
            fail(f"missing event file: {path}")
        events_raw = pd.read_csv(path, delimiter="\t")
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in events_raw.columns]
        if missing_cols:
            fail(f"{path} missing required columns: {missing_cols}; columns={list(events_raw.columns)}")

        events = events_raw.dropna().reset_index(drop=True)
        if events.empty:
            fail(f"{path} is empty after dropna(), matching core script behavior")

        events["SentenceID"] = events["SentenceID"].map(normalize_sentence_id)
        events["run"] = run
        run_events[run] = events

        run_articles = sorted({article_from_sentence_id(sid) for sid in events["SentenceID"]})
        print(
            f"run {run}: rows raw/dropna={len(events_raw)}/{len(events)}, "
            f"articles={run_articles}, first_sentence={events.SentenceID.iloc[0]}"
        )
    ok("all run event files exist and contain required columns")

    article_to_run: dict[int, int] = {}
    selected_events = []
    for article in range(1, 6):
        selected_run = None
        for run in range(1, 6):
            first_article = article_from_sentence_id(run_events[run].SentenceID.iloc[0])
            if first_article == article:
                selected_run = run
                break
        if selected_run is None:
            fail(f"could not find run whose first SentenceID belongs to article {article}")
        article_to_run[article] = selected_run
        ev = run_events[selected_run].copy()
        ev = ev[ev["SentenceID"].map(article_from_sentence_id) == article].copy()
        selected_events.append(ev)

    print()
    print("article -> selected run, using the same first-SentenceID logic as the core script:")
    for article, run in article_to_run.items():
        print(f"  article {article}: run {run}")
    ok("found one selected run for every article")

    events = pd.concat(selected_events, ignore_index=True)
    observed_ids = list(dict.fromkeys(events["SentenceID"]))
    expected_ids = expected_sentence_ids()
    missing = [sid for sid in expected_ids if sid not in observed_ids]
    extra = [sid for sid in observed_ids if sid not in expected_ids]
    if missing or extra:
        print("[FAIL] selected events do not cover the expected 148 sentences")
        print(f"  observed unique count: {len(observed_ids)}")
        print(f"  missing first 30: {missing[:30]}")
        print(f"  extra first 30: {extra[:30]}")
        sys.exit(1)
    ok("selected events cover all expected 148 SentenceID values")

    if observed_ids != expected_ids:
        first_mismatch = None
        for idx, (obs, exp) in enumerate(zip(observed_ids, expected_ids)):
            if obs != exp:
                first_mismatch = (idx, obs, exp)
                break
        warn(f"event SentenceID first-appearance order differs from canonical order: {first_mismatch}")
    else:
        ok("event SentenceID first-appearance order matches canonical order")

    bad_word_indices = []
    bad_onset = []
    nonmonotonic = []
    transition_stats = {
        "sentences_with_events": 0,
        "candidate_transitions": 0,
        "valid_transitions": 0,
        "out_of_range_transitions": 0,
    }
    max_scan_by_article = {article: -1 for article in range(1, 6)}
    event_counts = []

    for sid in expected_ids:
        article, sentence = split_sentence_id(sid)
        gidx = global_sentence_index(article, sentence)
        n_word = words_list[gidx]
        ev = events[events["SentenceID"] == sid].reset_index(drop=True)
        event_counts.append(len(ev))
        if ev.empty:
            fail(f"missing events for {sid}")

        ids = pd.to_numeric(ev["CURRENT_FIX_INTEREST_AREA_ID"], errors="coerce")
        onsets = pd.to_numeric(ev["onset"], errors="coerce")
        if ids.isna().any():
            bad_word_indices.append((sid, "non_numeric_word_index", ids[ids.isna()].index.tolist()))
            continue
        if onsets.isna().any():
            bad_onset.append((sid, "non_numeric_onset", onsets[onsets.isna()].index.tolist()))
            continue

        ids_int = ids.astype(int)
        not_integer = ids.to_numpy() != ids_int.to_numpy()
        if np.any(not_integer):
            bad_word_indices.append((sid, "non_integer_word_index", ids[not_integer].tolist()))

        min_id = int(ids_int.min())
        max_id = int(ids_int.max())
        if min_id < 1 or max_id > n_word:
            bad_word_indices.append((sid, "word_index_out_of_range", n_word, min_id, max_id))

        onset_values = onsets.to_numpy(dtype=float)
        if np.any(np.diff(onset_values) < 0):
            nonmonotonic.append((sid, "onset_decreases_within_sentence", onset_values.tolist()))

        transition_stats["sentences_with_events"] += 1
        for ind in range(len(ev) - 1):
            transition_stats["candidate_transitions"] += 1
            row = int(ids_int.iloc[ind]) - 1
            col = int(ids_int.iloc[ind + 1]) - 1
            if row < 0 or row >= n_word or col < 0 or col >= n_word:
                transition_stats["out_of_range_transitions"] += 1
                continue
            transition_stats["valid_transitions"] += 1
            scan = int(np.ceil((float(onsets.iloc[ind + 1]) / 1000.0 + HRF_DELAY_SECONDS) / TR_SECONDS))
            max_scan_by_article[article] = max(max_scan_by_article[article], scan)

    if bad_word_indices:
        print("[FAIL] bad CURRENT_FIX_INTEREST_AREA_ID entries")
        for item in bad_word_indices[:50]:
            print(" ", item)
        sys.exit(1)
    ok("all CURRENT_FIX_INTEREST_AREA_ID values are 1-based integers within words_list.p lengths")

    if bad_onset:
        print("[FAIL] bad onset entries")
        for item in bad_onset[:50]:
            print(" ", item)
        sys.exit(1)
    ok("all onset values are numeric")

    if nonmonotonic:
        print("[FAIL] onset decreases within at least one sentence")
        for item in nonmonotonic[:20]:
            print(" ", item)
        sys.exit(1)
    ok("onset values are nondecreasing within every sentence")

    onset_all = pd.to_numeric(events["onset"], errors="coerce").to_numpy(dtype=float)
    print()
    print("onset summary:")
    print(f"  min/median/max: {np.min(onset_all):.3f} / {np.median(onset_all):.3f} / {np.max(onset_all):.3f}")
    print(f"  p95          : {np.percentile(onset_all, 95):.3f}")
    if np.nanmedian(onset_all) < 100:
        warn("onset median is <100; values may already be seconds, but core script assumes milliseconds")
    else:
        ok("onset magnitude is consistent with milliseconds, matching core script onset/1000")

    print()
    print("event count per sentence:")
    print(f"  min/median/max: {min(event_counts)} / {np.median(event_counts):.1f} / {max(event_counts)}")
    print("transition stats:")
    for key, value in transition_stats.items():
        print(f"  {key}: {value}")
    if transition_stats["out_of_range_transitions"] != 0:
        fail("some transitions are out of range and would be skipped by core script")
    ok("all fixation-to-next-fixation transitions used by core script are in range")

    print()
    print("max scan index by article from core formula:")
    for article in range(1, 6):
        print(f"  article {article}: max_scan={max_scan_by_article[article]}, selected_run={article_to_run[article]}")

    if hem:
        print()
        print("fMRI timepoint check:")
        for article in range(1, 6):
            run = article_to_run[article]
            n_timepoints = load_fmri_n_timepoints(data_dir, subj, run, hem)
            max_scan = max_scan_by_article[article]
            print(f"  article {article}, run {run}, {hem}: n_timepoints={n_timepoints}, max_scan={max_scan}")
            if max_scan >= n_timepoints:
                fail(
                    f"article {article} max_scan={max_scan} exceeds fMRI timepoints={n_timepoints}; "
                    "core script would clamp by decrementing scan"
                )
        ok("all computed scan indices fit inside fMRI timepoints")

    print()
    print("Summary")
    print("  status: OK")


if __name__ == "__main__":
    main()
