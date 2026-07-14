#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
makesure_result.py

Robust result summarizer for Brain_LLM_Project ridge-regression outputs.

Examples:
    python3 makesure_result.py --model llama --size 7B --group base --hemi lh
    python3 makesure_result.py --model gpt2 --size large --group base --hemi lh
    python3 makesure_result.py --model llama --size 7B --group base --hemi lh --require-complete
    python3 makesure_result.py --model llama --size 7B --group base --hemi lh --save-csv llama_7B_base_lh.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


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

    "vicuna_7B":      {"layers": 32, "heads": 32},
    "vicuna_13B":     {"layers": 40, "heads": 40},

    "mistral_7B":     {"layers": 32, "heads": 32},
    "gemma_7B":       {"layers": 28, "heads": 16},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize layer-wise mean Pearson's r from saved corr*.npy files."
    )

    parser.add_argument(
        "--results-root",
        default="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results",
        help="Root result directory containing model/size/group/layer*/ files."
    )
    parser.add_argument("--model", default="llama", help="Model family, e.g. llama, gpt2, alpaca, vicuna.")
    parser.add_argument("--size", default="7B", help="Model size, e.g. 7B, 13B, large, xlarge.")
    parser.add_argument("--group", default="base", help="Group prefix in result filenames, e.g. base, instr, ctrl.")
    parser.add_argument("--hemi", default="lh", choices=["lh", "rh"], help="Hemisphere suffix in result filenames.")

    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Direct path to the model-size-group result directory. "
            "If omitted, uses results-root/model/size/group."
        )
    )
    parser.add_argument(
        "--include-extra-layers",
        action="store_true",
        help="Do not filter layers by MODEL_SPECS. By default unexpected layers are ignored."
    )
    parser.add_argument(
        "--min-subjects",
        type=int,
        default=1,
        help="Minimum number of subject files required for a layer to be included."
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Only include common subjects across all usable layers. "
            "This avoids comparing complete layers against incomplete layers."
        )
    )
    parser.add_argument(
        "--expected-vertices",
        type=int,
        default=10242,
        help="Expected number of fsaverage5 vertices per hemisphere. Set <=0 to disable this check."
    )
    parser.add_argument(
        "--exclude-subjs",
        default="21,52",
        help="Comma-separated subject IDs to ignore if files are present. Default: 21,52."
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional CSV path for saving layer, mean_r, std_across_subjects, n_subjects, subject_ids."
    )

    return parser.parse_args()


def parse_excluded_subjects(text: str) -> set[int]:
    if not text.strip():
        return set()
    out = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.add(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid subject ID in --exclude-subjs: {item!r}") from exc
    return out


def layer_number_from_dir(path: Path) -> int | None:
    m = re.fullmatch(r"layer(\d+)", path.name)
    return int(m.group(1)) if m else None


def subject_id_from_result_name(filename: str, group: str, hemi: str) -> int | None:
    # Expected format: base_subj1_corrlh.npy / instr_subj12_corrrh.npy
    pattern = rf"^{re.escape(group)}_subj(\d+)_corr{re.escape(hemi)}\.npy$"
    m = re.fullmatch(pattern, filename)
    return int(m.group(1)) if m else None


def discover_layers(base_dir: Path) -> list[int]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {base_dir}")

    layers = []
    for p in base_dir.iterdir():
        if p.is_dir():
            layer = layer_number_from_dir(p)
            if layer is not None:
                layers.append(layer)
    return sorted(set(layers))


def load_layer_subject_means(
    layer_dir: Path,
    group: str,
    hemi: str,
    excluded_subjects: set[int],
    expected_vertices: int,
) -> dict[int, float]:
    subject_means: dict[int, float] = {}

    for file_path in sorted(layer_dir.iterdir()):
        if not file_path.is_file():
            continue

        subj = subject_id_from_result_name(file_path.name, group=group, hemi=hemi)
        if subj is None:
            continue
        if subj in excluded_subjects:
            continue

        try:
            corr_array = np.load(file_path)
        except Exception as exc:
            print(f"⚠️  Failed to load {file_path}: {exc}", file=sys.stderr)
            continue

        if expected_vertices > 0 and corr_array.size != expected_vertices:
            print(
                f"⚠️  Vertex-count warning: {file_path.name} has {corr_array.size} values, "
                f"expected {expected_vertices}. Still using np.nanmean.",
                file=sys.stderr,
            )

        subj_mean = float(np.nanmean(corr_array))
        if not np.isfinite(subj_mean):
            print(f"⚠️  Non-finite mean in {file_path}; skipped.", file=sys.stderr)
            continue

        subject_means[subj] = subj_mean

    return subject_means


def main() -> int:
    args = parse_args()
    excluded_subjects = parse_excluded_subjects(args.exclude_subjs)

    base_dir = Path(args.base_dir) if args.base_dir else (
        Path(args.results_root) / args.model / args.size / args.group
    )

    key = f"{args.model}_{args.size}"
    spec = MODEL_SPECS.get(key)

    print(f"📁 Result directory: {base_dir}")
    print(f"🎯 Target: {key} | group={args.group} | hemi={args.hemi}")

    if spec is None:
        print(
            f"⚠️  Unknown model spec for {key}. "
            "Layer filtering by expected architecture is disabled unless you add it to MODEL_SPECS.",
            file=sys.stderr,
        )
        expected_layers = None
    else:
        expected_layers = spec["layers"]
        print(f"🧠 Expected architecture: {expected_layers} layers, {spec['heads']} heads")

    detected_layers = discover_layers(base_dir)
    if not detected_layers:
        print("❌ No layer directories found.")
        return 1

    print(f"🔎 Detected layer directories: {detected_layers}")

    if expected_layers is not None and not args.include_extra_layers:
        allowed = set(range(expected_layers))
        extra = [l for l in detected_layers if l not in allowed]
        missing = [l for l in range(expected_layers) if l not in detected_layers]

        if extra:
            print(
                f"⚠️  Ignoring unexpected layers for {key}: {extra}. "
                "Use --include-extra-layers only if this is intentional."
            )
        if missing:
            print(f"⚠️  Missing expected layers for {key}: {missing}")

        layers_to_use = [l for l in detected_layers if l in allowed]
    else:
        layers_to_use = detected_layers

    layer_subject_means: dict[int, dict[int, float]] = {}

    for layer in layers_to_use:
        layer_dir = base_dir / f"layer{layer}"
        subj_means = load_layer_subject_means(
            layer_dir=layer_dir,
            group=args.group,
            hemi=args.hemi,
            excluded_subjects=excluded_subjects,
            expected_vertices=args.expected_vertices,
        )

        if len(subj_means) < args.min_subjects:
            print(
                f"⚠️  Layer {layer} skipped: only {len(subj_means)} subjects "
                f"(< min_subjects={args.min_subjects})."
            )
            continue

        layer_subject_means[layer] = subj_means

    if not layer_subject_means:
        print("❌ No usable layer results found after filtering.")
        return 1

    if args.require_complete:
        common_subjects = set.intersection(*(set(v.keys()) for v in layer_subject_means.values()))
        if not common_subjects:
            print("❌ --require-complete requested, but no subject is common to all layers.")
            return 1

        print(f"🔒 Complete-subject mode: using common subjects only: {sorted(common_subjects)}")
        for layer in list(layer_subject_means):
            layer_subject_means[layer] = {
                subj: val
                for subj, val in layer_subject_means[layer].items()
                if subj in common_subjects
            }

    # Warn if subject sets differ across layers.
    subject_sets = {layer: set(vals.keys()) for layer, vals in layer_subject_means.items()}
    unique_sets = {tuple(sorted(s)) for s in subject_sets.values()}
    if len(unique_sets) > 1 and not args.require_complete:
        print(
            "⚠️  Subject sets differ across layers. "
            "For strict comparison, rerun with --require-complete or set --min-subjects appropriately."
        )
        for layer, subjects in sorted(subject_sets.items()):
            print(f"    Layer {layer}: n={len(subjects)}, subjects={sorted(subjects)}")

    layer_stats = {}
    for layer, subj_means in sorted(layer_subject_means.items()):
        vals = np.array(list(subj_means.values()), dtype=float)
        layer_stats[layer] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
            "subjects": sorted(subj_means.keys()),
        }

    print("\n--- Mean Pearson's r per Layer ---")
    for layer, stats in sorted(layer_stats.items()):
        print(
            f"Layer {layer}: {stats['mean']:.5f} "
            f"(n={stats['n']}, std_across_subjects={stats['std']:.5f})"
        )

    golden_layer = max(layer_stats, key=lambda x: layer_stats[x]["mean"])
    best = layer_stats[golden_layer]

    print("\n" + "=" * 50)
    print(
        f"🏆 GOLDEN LAYER: Layer {golden_layer} "
        f"with mean r = {best['mean']:.5f} "
        f"(n={best['n']}) 🏆"
    )
    print("=" * 50)

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["layer", "mean_r", "std_across_subjects", "n_subjects", "subject_ids"])
            for layer, stats in sorted(layer_stats.items()):
                writer.writerow([
                    layer,
                    f"{stats['mean']:.10f}",
                    f"{stats['std']:.10f}",
                    stats["n"],
                    " ".join(map(str, stats["subjects"])),
                ])
        print(f"💾 CSV saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
