#!/usr/bin/env python3
"""Path-fixed author-style trivial-pattern residualization for Reading Brain attention.

This follows the repository author's residue idea:

    attention_head ~= self + prev + start
    residue = attention_head - predicted_trivial_component

Unlike the fMRI ridge script's online residualization, this script works on the
full lower triangle including the diagonal, matching utils/get_residues/
get_model_residues_rb.py. The resulting residue_rb_p1_layer*.npy files can be
used by the fMRI script with:

    ATTN_PREFIX_OVERRIDE=residue_rb_p1
    SUBTRACT_TRIVIAL_PATTERNS=0
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parents[2]))
ATTN_DIR = Path(os.environ.get("ATTN_DIR", PROJECT_DIR / "model_attention"))
LABEL_DIR = Path(os.environ.get("LABEL_DIR", PROJECT_DIR / "golden_attention" / "reading_brain"))
ATTN_PREFIX = os.environ.get("ATTN_PREFIX", "rb_p1")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "residue_rb_p1")

MODEL_SPECS = {
    "gpt2_large": {"layers": 36, "heads": 20},
    "llama_7B": {"layers": 32, "heads": 32},
    "llama_13B": {"layers": 40, "heads": 40},
    "llama_30B": {"layers": 60, "heads": 52},
    "llama_65B": {"layers": 80, "heads": 64},
    "alpaca_7B": {"layers": 32, "heads": 32},
    "alpaca_13B": {"layers": 40, "heads": 40},
    "vicuna_7B": {"layers": 32, "heads": 32},
    "vicuna_13B": {"layers": 40, "heads": 40},
}

LABEL_TYPES = ("self", "prev", "start")


def tril_idx(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(int(n), k=0)


def load_labels() -> list:
    labels = []
    for label_type in LABEL_TYPES:
        path = LABEL_DIR / f"label_{label_type}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Label file not found: {path}")
        with path.open("rb") as f:
            labels.append(pickle.load(f))
    return labels


def parse_layers(spec: str, n_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_layers))
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Layer {layer} outside valid range 0..{n_layers - 1}")
        layers.append(layer)
    if not layers:
        raise ValueError("No layers requested")
    return layers


def fit_predict_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = float(y.mean())
    x_centered = x - x_mean
    y_centered = y - y_mean
    beta = np.linalg.lstsq(x_centered, y_centered, rcond=None)[0]
    pred = x_centered @ beta + y_mean
    ss_total = float(np.sum(y_centered * y_centered))
    ss_resid = float(np.sum((y - pred) ** 2))
    r2 = 0.0 if ss_total == 0 else 1.0 - ss_resid / ss_total
    return pred, float(np.nan_to_num(r2)), beta


def recover_head_residue(
    residue_values: np.ndarray,
    target: np.ndarray,
    sentence_lengths: list[list[int]],
) -> None:
    start = 0
    for article_idx, article_lengths in enumerate(sentence_lengths):
        for sentence_idx, n_words in enumerate(article_lengths):
            n_values = int(n_words) * (int(n_words) + 1) // 2
            values = residue_values[start:start + n_values]
            row_idx, col_idx = tril_idx(n_words)
            target[article_idx, sentence_idx, row_idx, col_idx] = values
            start += n_values
    if start != residue_values.shape[0]:
        raise ValueError(f"Recovered {start} values, but residue has {residue_values.shape[0]}")


def residualize_layer(name: str, size: str, layer: int, n_heads: int, labels: list) -> dict:
    attn_path = ATTN_DIR / name / size / f"{ATTN_PREFIX}_layer{layer}.npy"
    if not attn_path.exists():
        raise FileNotFoundError(f"Attention file not found: {attn_path}")

    layer_attn = np.load(attn_path)
    if layer_attn.ndim != 5:
        raise ValueError(f"Attention should be 5D, got {layer_attn.shape}: {attn_path}")
    if layer_attn.shape[2] != n_heads:
        raise ValueError(f"Expected {n_heads} heads, got {layer_attn.shape[2]}: {attn_path}")

    flattened_labels = []
    flattened_attention = []
    sentence_lengths: list[list[int]] = [[] for _ in range(len(labels[0]))]

    for article_idx in range(len(labels[0])):
        for sentence_idx in range(len(labels[0][article_idx])):
            n_words = len(np.asarray(labels[0][article_idx][sentence_idx]))
            sentence_lengths[article_idx].append(n_words)
            row_idx, col_idx = tril_idx(n_words)

            sentence_attn = layer_attn[article_idx, sentence_idx]
            flattened_attention.append(sentence_attn[:, row_idx, col_idx].T)

            label_cols = []
            for label_data in labels:
                label_mat = np.asarray(label_data[article_idx][sentence_idx], dtype=np.float64)
                label_cols.append(label_mat[row_idx, col_idx])
            flattened_labels.append(np.stack(label_cols, axis=1))

    x_labels = np.concatenate(flattened_labels, axis=0)
    y_attention = np.concatenate(flattened_attention, axis=0)
    if x_labels.shape[0] != y_attention.shape[0]:
        raise ValueError(f"Flatten mismatch: labels={x_labels.shape}, attention={y_attention.shape}")

    residue = np.zeros_like(layer_attn, dtype=layer_attn.dtype)
    r2_scores = []
    coefficients = []
    for head in range(n_heads):
        pred, r2, beta = fit_predict_linear(x_labels, y_attention[:, head])
        r2_scores.append(r2)
        coefficients.append(beta.tolist())
        recover_head_residue(y_attention[:, head] - pred, residue[:, :, head], sentence_lengths)

    out_path = ATTN_DIR / name / size / f"{OUTPUT_PREFIX}_layer{layer}.npy"
    np.save(out_path, residue)
    meta = {
        "script": str(Path(__file__).resolve()),
        "method": "author_style_linear_residue_inclusive_lower_triangle",
        "name": name,
        "size": size,
        "layer": layer,
        "n_heads": n_heads,
        "label_types": list(LABEL_TYPES),
        "include_diagonal": True,
        "attention_path": str(attn_path),
        "output_path": str(out_path),
        "n_edges_inclusive": int(x_labels.shape[0]),
        "r2_mean": float(np.mean(r2_scores)),
        "r2_max": float(np.max(r2_scores)),
        "r2_min": float(np.min(r2_scores)),
        "r2_per_head": [float(v) for v in r2_scores],
        "coefficients_per_head": coefficients,
    }
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: python get_model_residues_rb_pathfixed.py <name> <size> <layer|comma_layers|all>\n"
            "Example: python get_model_residues_rb_pathfixed.py llama 7B 8",
            file=sys.stderr,
        )
        sys.exit(1)

    name = sys.argv[1]
    size = sys.argv[2]
    model_key = f"{name}_{size}"
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model/size: {model_key}. Known: {sorted(MODEL_SPECS)}")

    n_layers = MODEL_SPECS[model_key]["layers"]
    n_heads = MODEL_SPECS[model_key]["heads"]
    layers = parse_layers(sys.argv[3], n_layers)
    labels = load_labels()

    print("=" * 80)
    print("Author-style model attention residue")
    print(f"Project dir  : {PROJECT_DIR}")
    print(f"Attention dir: {ATTN_DIR}")
    print(f"Label dir    : {LABEL_DIR}")
    print(f"Target       : {model_key}, layers={layers}, heads={n_heads}")
    print(f"Labels       : {list(LABEL_TYPES)}")
    print("=" * 80)

    start = time.time()
    metas = []
    for layer in layers:
        meta = residualize_layer(name, size, layer, n_heads, labels)
        metas.append(meta)
        print(
            f"layer {layer}: saved {meta['output_path']}; "
            f"R2 mean/max={meta['r2_mean']:.6f}/{meta['r2_max']:.6f}"
        )

    print(f"Done in {time.time() - start:.2f}s. Mean R2 across requested layers: "
          f"{np.mean([m['r2_mean'] for m in metas]):.6f}")


if __name__ == "__main__":
    main()
