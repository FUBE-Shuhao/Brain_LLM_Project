import os
import re
import glob
import pickle
import numpy as np
import pandas as pd

PROJECT = "/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main"
ATTN_ROOT = os.path.join(PROJECT, "model_attention", "llama")

EXPECTED_EDGES = 7388

SIZES = {
    "7B": 32,
    "13B": 40,
    "30B": 60,
    "65B": 80,
}

def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None found:\n" + "\n".join(paths))

def load_sentence_lengths():
    pkl_path = first_existing([
        os.path.join(PROJECT, "Analysis", "words_list.p"),
        os.path.join(PROJECT, "Data", "words_list.p"),
    ])

    print("=" * 90)
    print("Loading words_list.p")
    print("=" * 90)
    print("path:", pkl_path)

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    print("type:", type(obj))
    print("len:", len(obj))
    print("first 20 raw:", obj[:20])

    if not isinstance(obj, list):
        raise RuntimeError("words_list.p is not a list")

    if not all(isinstance(x, (int, np.integer)) for x in obj):
        raise RuntimeError("words_list.p is not a pure int sentence-length list")

    lens = [int(x) for x in obj]

    edges = sum(n * (n - 1) // 2 for n in lens)

    print("n_sentences:", len(lens))
    print("min/max length:", min(lens), max(lens))
    print("total words:", sum(lens))
    print("total lower-triangle no-diagonal edges:", edges)

    if len(lens) != 148:
        raise RuntimeError(f"Expected 148 sentence lengths, got {len(lens)}")

    if edges != EXPECTED_EDGES:
        raise RuntimeError(f"Expected 7388 edges, got {edges}")

    print("[OK] words_list.p gives exactly 7388 edges")
    return lens

def load_article_sentence_counts():
    """
    这里只用 words.csv 的 SentenceID 来确定 148 个句子如何分到 5 个 article/run。
    不用 words.csv 算句长，因为它现在是 1526 words / 7359 edges，和作者句长表不一致。
    """
    csv_path = first_existing([
        os.path.join(PROJECT, "Analysis", "words.csv"),
        os.path.join(PROJECT, "Data", "words.csv"),
    ])

    print("\n" + "=" * 90)
    print("Loading words.csv only for article sentence counts")
    print("=" * 90)
    print("path:", csv_path)

    df = pd.read_csv(csv_path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    if "SentenceID" not in df.columns:
        raise RuntimeError("words.csv has no SentenceID column")

    sids = df["SentenceID"].astype(str).tolist()

    # 保留首次出现顺序
    unique_sids = []
    seen = set()
    for sid in sids:
        if sid not in seen:
            seen.add(sid)
            unique_sids.append(sid)

    print("unique SentenceID count:", len(unique_sids))
    print("first 10 SentenceID:", unique_sids[:10])

    if len(unique_sids) != 148:
        raise RuntimeError(f"Expected 148 unique SentenceID, got {len(unique_sids)}")

    # 解析 t.01.01 / t.02.03 这种格式
    article_ids = []
    for sid in unique_sids:
        m = re.search(r"t\.(\d+)\.", sid)
        if not m:
            raise RuntimeError(f"Cannot parse article id from SentenceID: {sid}")
        article_ids.append(int(m.group(1)))

    counts = []
    for a in sorted(set(article_ids)):
        counts.append(sum(x == a for x in article_ids))

    print("article sentence counts:", counts)
    print("sum article counts:", sum(counts))

    if len(counts) != 5:
        raise RuntimeError(f"Expected 5 articles, got {len(counts)}")

    if sum(counts) != 148:
        raise RuntimeError(f"Article counts do not sum to 148: {sum(counts)}")

    if max(counts) > 31:
        raise RuntimeError(f"Article has >31 sentences: {counts}")

    return counts

def make_slots(lens, article_counts):
    """
    attention shape 是 (5, 31, n_heads, 16, 16)
    把 148 个句长放回 5 × 31 的 slot 里，padding slot 设为 0。
    """
    slots = np.zeros((5, 31), dtype=int)

    k = 0
    for a, c in enumerate(article_counts):
        for s in range(c):
            slots[a, s] = lens[k]
            k += 1

    if k != 148:
        raise RuntimeError(f"Assigned {k} sentences, expected 148")

    edges = int(sum(n * (n - 1) // 2 for n in slots.reshape(-1) if n > 0))
    print("\n" + "=" * 90)
    print("Sentence slot summary")
    print("=" * 90)
    print("slots shape:", slots.shape)
    print("nonzero slots:", int((slots > 0).sum()))
    print("article counts from slots:", [int((slots[i] > 0).sum()) for i in range(5)])
    print("total words from slots:", int(slots.sum()))
    print("total edges from slots:", edges)

    if edges != EXPECTED_EDGES:
        raise RuntimeError(f"Slot edges expected 7388, got {edges}")

    print("[OK] slots give exactly 7388 edges")
    return slots

def find_layer_files(size):
    root = os.path.join(ATTN_ROOT, size)
    files = glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)

    layer_to_path = {}

    for f in files:
        name = os.path.basename(f)
        m = re.search(r"layer[_-]?(\d+)", name)
        if not m:
            continue

        layer = int(m.group(1))

        # 同层重复时，优先 rb_p1_layer，且路径短者优先
        score = len(f)
        if "rb_p1_layer" in name:
            score -= 1000

        if layer not in layer_to_path or score < layer_to_path[layer][0]:
            layer_to_path[layer] = (score, f)

    return {layer: path for layer, (_, path) in layer_to_path.items()}

def flatten_x(att, slots):
    if att.ndim != 5:
        raise RuntimeError(f"Expected 5D attention, got {att.shape}")

    if not (att.shape[0] == 5 and att.shape[1] == 31 and att.shape[-2:] == (16, 16)):
        raise RuntimeError(f"Unexpected attention shape: {att.shape}")

    n_heads = att.shape[2]

    rows = []
    for a in range(5):
        for s in range(31):
            n = int(slots[a, s])
            if n <= 1:
                continue

            if n > 16:
                raise RuntimeError(f"Sentence length >16 at article {a}, slot {s}: {n}")

            tri = np.tril_indices(n, k=-1)

            # att[a, s] shape: n_heads × 16 × 16
            x = att[a, s, :, :n, :n][:, tri[0], tri[1]].T
            rows.append(x)

    X = np.concatenate(rows, axis=0)
    return X, n_heads

def main():
    lens = load_sentence_lengths()
    article_counts = load_article_sentence_counts()
    slots = make_slots(lens, article_counts)

    for size, expected_layers in SIZES.items():
        print("\n" + "=" * 90)
        print(f"Checking {size}")
        print("=" * 90)

        layer_to_path = find_layer_files(size)
        found_layers = sorted(layer_to_path.keys())
        missing = [i for i in range(expected_layers) if i not in layer_to_path]

        print("found layer count:", len(found_layers))
        if found_layers:
            print("found min/max:", min(found_layers), max(found_layers))
        print("missing layers:", missing)

        shape_set = set()
        head_set = set()
        ok_count = 0
        bad_count = 0

        sample_layers = set()
        if found_layers:
            sample_layers = {
                found_layers[0],
                found_layers[len(found_layers) // 2],
                found_layers[-1],
            }

        for layer in found_layers:
            path = layer_to_path[layer]

            try:
                arr = np.load(path, allow_pickle=False)
                X, n_heads = flatten_x(arr, slots)

                finite = float(np.isfinite(X).mean())
                zero = float((X == 0).mean())
                mean = float(np.nanmean(X.astype(np.float64)))
                std = float(np.nanstd(X.astype(np.float64)))

                shape_set.add(X.shape)
                head_set.add(n_heads)

                is_ok = (X.shape[0] == EXPECTED_EDGES and finite == 1.0)

                if is_ok:
                    ok_count += 1
                else:
                    bad_count += 1

                if layer in sample_layers or not is_ok:
                    print(
                        f"layer {layer:02d}: "
                        f"raw_shape={arr.shape}, X_shape={X.shape}, "
                        f"heads={n_heads}, finite={finite:.6f}, zero={zero:.6f}, "
                        f"mean={mean:.6g}, std={std:.6g}, "
                        f"path={path}"
                    )

            except Exception as e:
                bad_count += 1
                print(f"layer {layer:02d}: ERROR {repr(e)} path={path}")

        print("\nSummary for", size)
        print("OK layers:", ok_count)
        print("BAD layers:", bad_count)
        print("X shapes seen:", sorted(shape_set))
        print("n_heads seen:", sorted(head_set))

        if missing:
            print("[WARNING] missing layers:", missing)

        if bad_count == 0 and not missing:
            print(f"[PASS] {size}: all layers flatten to 7388 × n_heads")
        else:
            print(f"[CHECK] {size}: inspect warnings/errors above")

if __name__ == "__main__":
    main()
