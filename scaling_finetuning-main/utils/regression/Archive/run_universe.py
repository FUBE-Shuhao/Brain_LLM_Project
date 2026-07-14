#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
run_universe.py

Robust task launcher for heads_vs_fmri.py.

Key changes:
1. No blind layer scanning as the source of truth.
2. Uses model architecture specs to restrict valid layers.
3. Uses group-specific attention prefixes.
4. Supports clean rerun with --overwrite.
5. Skips excluded fMRI subjects 21 and 52 by default.

Examples:
    python3 run_universe.py
    python3 run_universe.py --models llama --sizes 7B --groups base
    python3 run_universe.py --models llama --sizes 7B --groups base --subjects 1 --overwrite
    python3 run_universe.py --models gpt2 --sizes large --layers 0,1,2
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main")
ATTN_DIR = PROJECT_ROOT / "model_attention"
RESULTS_DIR = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results")
HEADS_SCRIPT = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main/utils/regression/heads_vs_fmri.py")

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

GROUP_TO_ATTN_PREFIX = {
    "base": "rb_p1",
    "instr": "instr_rb_p1",
    "ctrl": "ctrl_rb_p1",
}

DEFAULT_EXCLUDE_SUBJS = {21, 52}


def parse_csv_list(text: str, cast=str):
    if text is None or text == "":
        return []
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def parse_subjects(text: str) -> list[int]:
    """
    Accepts:
        "1"
        "1,2,3"
        "1-52"
        "1-5,8,10"
    """
    out = set()
    for part in parse_csv_list(text, str):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_args():
    p = argparse.ArgumentParser(description="Launch ridge regression jobs robustly.")

    p.add_argument("--models", default="llama", help="Comma-separated model families, e.g. llama,gpt2")
    p.add_argument("--sizes", default="7B", help="Comma-separated sizes, e.g. 7B,13B or large")
    p.add_argument("--groups", default="base", help="Comma-separated groups: base,instr,ctrl")
    p.add_argument("--hemis", default="lh", help="Comma-separated hemispheres: lh,rh")
    p.add_argument("--layers", default="", help="Optional comma-separated layers. Empty means all valid layers.")
    p.add_argument("--subjects", default="1-52", help="Subjects, e.g. 1 or 1,2,3 or 1-52")
    p.add_argument("--exclude-subjs", default="21,52", help="Comma-separated subject IDs to skip.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if result file already exists.")
    p.add_argument("--dry-run", action="store_true", help="Only print planned tasks.")
    p.add_argument("--python-bin", default=sys.executable, help="Python executable to call.")
    p.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one task fails.")

    return p.parse_args()


def get_valid_layers(name: str, size: str, requested_layers: list[int]) -> list[int]:
    key = f"{name}_{size}"
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model spec: {key}. Add it to MODEL_SPECS.")

    n_layers = MODEL_SPECS[key]["layers"]
    valid = list(range(n_layers))

    if requested_layers:
        bad = [x for x in requested_layers if x not in valid]
        if bad:
            raise ValueError(f"Requested invalid layers for {key}: {bad}. Valid: 0..{n_layers - 1}")
        return requested_layers

    return valid


def attention_file_exists(name: str, size: str, group: str, layer: int) -> bool:
    prefix = GROUP_TO_ATTN_PREFIX[group]
    return (ATTN_DIR / name / size / f"{prefix}_layer{layer}.npy").exists()


def main() -> int:
    args = parse_args()

    models = parse_csv_list(args.models)
    sizes = parse_csv_list(args.sizes)
    groups = parse_csv_list(args.groups)
    hemis = parse_csv_list(args.hemis)
    requested_layers = parse_csv_list(args.layers, int)
    subjects = parse_subjects(args.subjects)
    excluded = set(parse_csv_list(args.exclude_subjs, int))

    for g in groups:
        if g not in GROUP_TO_ATTN_PREFIX:
            raise ValueError(f"Unknown group {g!r}. Valid groups: {sorted(GROUP_TO_ATTN_PREFIX)}")

    for h in hemis:
        if h not in {"lh", "rh"}:
            raise ValueError(f"Unknown hemisphere {h!r}. Use lh or rh.")

    print("🚀 === Robust ridge launcher ===")
    print(f"Models={models}, sizes={sizes}, groups={groups}, hemis={hemis}")
    print(f"Subjects={subjects}, excluded={sorted(excluded)}, overwrite={args.overwrite}")
    print("================================")

    tasks = []
    skipped_missing_attention = []

    for name in models:
        for size in sizes:
            key = f"{name}_{size}"
            layers = get_valid_layers(name, size, requested_layers)
            print(f"🧠 {key}: valid layers -> {layers[0]}..{layers[-1]} ({len(layers)} layers)")

            for group in groups:
                for layer in layers:
                    if not attention_file_exists(name, size, group, layer):
                        skipped_missing_attention.append((name, size, group, layer))
                        continue

                    for subj in subjects:
                        if subj in excluded:
                            continue

                        for hemi in hemis:
                            target = (
                                RESULTS_DIR / name / size / group / f"layer{layer}" /
                                f"{group}_subj{subj}_corr{hemi}.npy"
                            )

                            if target.exists() and not args.overwrite:
                                print(f"⏭️  Skip existing: {target}")
                                continue

                            tasks.append((subj, group, name, size, hemi, layer, target))

    if skipped_missing_attention:
        print("\n⚠️ Missing attention files were skipped:")
        for item in skipped_missing_attention[:20]:
            print(f"   {item[0]}_{item[1]} {item[2]} layer{item[3]}")
        if len(skipped_missing_attention) > 20:
            print(f"   ... and {len(skipped_missing_attention) - 20} more")

    print(f"\n✅ Planned runnable tasks: {len(tasks)}")

    if args.dry_run:
        for t in tasks[:50]:
            subj, group, name, size, hemi, layer, target = t
            print(f"DRY: subj={subj}, {name}_{size}, group={group}, hemi={hemi}, layer={layer} -> {target}")
        if len(tasks) > 50:
            print(f"... and {len(tasks) - 50} more")
        return 0

    n_fail = 0

    for idx, (subj, group, name, size, hemi, layer, target) in enumerate(tasks, start=1):
        print(
            f"\n▶ [{idx}/{len(tasks)}] "
            f"sub-{subj:02d} | {name}_{size} | {group} | {hemi} | layer {layer}"
        )

        cmd = [
            args.python_bin,
            str(HEADS_SCRIPT),
            str(subj),
            group,
            name,
            size,
            hemi,
            str(layer),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            n_fail += 1
            print(f"❌ Failed: sub-{subj:02d} | {name}_{size} | {group} | {hemi} | layer {layer}")
            print("----- STDOUT -----")
            print(result.stdout.strip())
            print("----- STDERR -----")
            print(result.stderr.strip())

            if args.stop_on_error:
                return result.returncode
        else:
            # Print tail only to keep logs readable.
            stdout_tail = "\n".join(result.stdout.strip().splitlines()[-8:])
            if stdout_tail:
                print(stdout_tail)
            print(f"✅ Done -> {target}")

    print("\n🏁 All requested tasks finished.")
    if n_fail:
        print(f"⚠️ Failed tasks: {n_fail}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
