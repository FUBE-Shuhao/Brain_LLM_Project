### 6.9

## 1. 跑 sub-05 到 sub-08 的 fMRIPrep25

```
cat > /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/run_fmriprep25_sub05_08.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

BIDS=/media/i9a2/WindowsData/Wsh/DeepPrep/input
OUT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/output1
WORK=/media/i9a2/WindowsData/Wsh/DeepPrep/work/fmriprep25_output1
TF=/media/i9a2/WindowsData/Wsh/DeepPrep/templateflow
FS_LICENSE=/media/i9a2/WindowsData/Wsh/DeepPrep/license.txt
LOG=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/fmriprep25_sub05_08.log

mkdir -p "$OUT" "$WORK" "$TF"

echo "BIDS=$BIDS"
echo "OUT=$OUT"
echo "WORK=$WORK"
echo "TF=$TF"
echo "FS_LICENSE=$FS_LICENSE"
echo "LOG=$LOG"

docker run --rm -it --network host \
  -v "$BIDS":/data:ro \
  -v "$OUT":/out \
  -v "$WORK":/work \
  -v "$TF":/opt/templateflow \
  -v "$FS_LICENSE":/opt/freesurfer/license.txt:ro \
  -e TEMPLATEFLOW_HOME=/opt/templateflow \
  nipreps/fmriprep:25.0.0 \
  /data /out participant \
  --participant-label 05 06 07 08 \
  --fs-license-file /opt/freesurfer/license.txt \
  --output-spaces MNI152NLin2009cAsym fsaverage5 \
  --nthreads 16 \
  --omp-nthreads 8 \
  --mem-mb 64000 \
  -w /work \
  2>&1 | tee "$LOG"
SH

chmod +x /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/run_fmriprep25_sub05_08.sh

/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/run_fmriprep25_sub05_08.sh
```

------

## 2. fMRIPrep 跑完后检查 sub-05 到 sub-08 输出

```
OUT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/output1

for sid in 05 06 07 08; do
  echo "============================================================"
  echo "sub-$sid"
  echo "============================================================"

  echo "task-read surface func.gii count:"
  find "$OUT/sub-$sid/func" -name "sub-${sid}_task-read*hemi-*bold.func.gii" | wc -l

  echo "task-read MNI BOLD count:"
  find "$OUT/sub-$sid/func" -name "sub-${sid}_task-read*space-MNI152NLin2009cAsym*desc-preproc_bold.nii.gz" | wc -l
done
```

每个 subject 理想输出应该是：

```
task-read surface func.gii count: 10
task-read MNI BOLD count: 5
```

------

## 3. 把 sub-05 到 sub-08 的 surface 文件复制成 regression 需要的命名

这一步会同时写入：

```
Data/fsaverage5_fmriprep25
Data/fsaverage5
```

因为你现在已经把 `Data/fsaverage5` 作为当前实际读取目录，`fsaverage5_fmriprep25` 作为备份目录。这样两个目录内容保持一致。

```
PROJECT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main
OUT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/output1

NEWFS=$PROJECT/Data/fsaverage5_fmriprep25
ACTIVEFS=$PROJECT/Data/fsaverage5

mkdir -p "$NEWFS" "$ACTIVEFS"

for sid in 05 06 07 08; do
  subj="sub-$sid"

  echo "============================================================"
  echo "Copying $subj task-read fsaverage files"
  echo "============================================================"

  for run in 1 2 3 4 5; do
    for pair in "L lh" "R rh"; do
      H=$(echo "$pair" | awk '{print $1}')
      h=$(echo "$pair" | awk '{print $2}')

      src=$(find "$OUT/$subj/func" \
        -name "${subj}_task-read_run-${run}_*hemi-${H}*bold.func.gii" | head -n 1)

      if [ -z "$src" ]; then
        echo "MISSING: $subj run=$run hemi=$H"
        continue
      fi

      dst_name="${subj}_task-read_run-${run}_hemi-${h}_space-fsaverage5_bold.func.gii"

      cp -f "$src" "$NEWFS/$dst_name"
      cp -f "$src" "$ACTIVEFS/$dst_name"

      echo "OK: $dst_name"
    done
  done
done
```

检查复制结果：

```
PROJECT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main

for sid in 04 05 06 07 08; do
  echo "sub-$sid active fsaverage5 count:"
  find "$PROJECT/Data/fsaverage5" -name "sub-${sid}_task-read_run-*_hemi-*_space-fsaverage5_bold.func.gii" | wc -l
done
```

每个都应该是：

```
10
```

------

## 4. 复制 sub-05 到 sub-08 的 events

```
PROJECT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main
BIDS=/media/i9a2/WindowsData/Wsh/DeepPrep/input

for sid in 05 06 07 08; do
  subj="sub-$sid"

  mkdir -p "$PROJECT/Data/base/$subj/func"

  cp -f "$BIDS/$subj/func/${subj}_task-read_run-"*_events.tsv \
    "$PROJECT/Data/base/$subj/func/"

  echo "$subj events copied:"
  ls "$PROJECT/Data/base/$subj/func/${subj}_task-read_run-"*_events.tsv | wc -l
done
```

每个应该是：

```
5
```

------

## 5. 跑 sub-04 到 sub-08 的候选层岭回归

`sub-04` 已经全层跑完了，这个脚本会自动跳过已有结果。`sub-05~08` 会跑候选层。

候选层：

```
7B:  14 17 21 30
13B: 7 35 37 39
30B: 55 57 58 59
65B: 30 60 65 70 78
cat > /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/run_ridge_fmriprep25_sub04_08_candidates.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

PROJECT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main
export OUTPUT_ROOT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_fmriprep25_test

cd "$PROJECT/utils/regression"

echo "PROJECT=$PROJECT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Using active fsaverage5:"
ls -ld "$PROJECT/Data/fsaverage5"

for sid in 4 5 6 7 8; do
  subj=$(printf "sub-%02d" "$sid")

  echo "============================================================"
  echo "Subject $subj"
  echo "============================================================"

  for item in \
    "7B 14" "7B 17" "7B 21" "7B 30" \
    "13B 7" "13B 35" "13B 37" "13B 39" \
    "30B 55" "30B 57" "30B 58" "30B 59" \
    "65B 30" "65B 60" "65B 65" "65B 70" "65B 78"
  do
    size=$(echo "$item" | awk '{print $1}')
    layer=$(echo "$item" | awk '{print $2}')

    out="$OUTPUT_ROOT/llama/$size/base/layer${layer}/base_subj${sid}_corrlh.npy"

    if [ -s "$out" ]; then
      echo "SKIP existing: $subj llama $size lh layer=$layer"
      continue
    fi

    echo "RUN: $subj llama $size lh layer=$layer"
    python3 heads_vs_fmri_original_pathfixed.py "$sid" base llama "$size" lh "$layer"
  done
done
SH

chmod +x /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/run_ridge_fmriprep25_sub04_08_candidates.sh

/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/run_ridge_fmriprep25_sub04_08_candidates.sh
```

------

## 6. 跑完后汇总 sub-04 到 sub-08 的 fMRIPrep25 小样本趋势

```
python3 - <<'PY'
import numpy as np
from pathlib import Path

root = Path("/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_fmriprep25_test/llama")
subjects = [4, 5, 6, 7, 8]
group = "base"
hem = "lh"

candidates = {
    "7B":  [14, 17, 21, 30],
    "13B": [7, 35, 37, 39],
    "30B": [55, 57, 58, 59],
    "65B": [30, 60, 65, 70, 78],
}

print("fMRIPrep25 candidate-layer group sweep | sub-04~08 | lh")
print("=" * 100)

summary = {}

for size, layers in candidates.items():
    print(f"\n{size}")
    print("-" * 100)

    rows = []

    for layer in layers:
        means, medians, top10s, posmeans = [], [], [], []

        for sid in subjects:
            p = root / size / group / f"layer{layer}" / f"{group}_subj{sid}_corr{hem}.npy"

            if not p.exists():
                continue

            r = np.load(p)
            r = r[np.isfinite(r)]

            means.append(float(np.mean(r)))
            medians.append(float(np.median(r)))
            top10s.append(float(np.mean(np.sort(r)[int(0.9 * len(r)):])))
            posmeans.append(float(np.mean(r[r > 0])) if np.any(r > 0) else np.nan)

        if not means:
            print(f"layer {layer:02d}: missing")
            continue

        row = {
            "layer": layer,
            "mean": float(np.mean(means)),
            "std": float(np.std(means)),
            "median": float(np.mean(medians)),
            "top10": float(np.mean(top10s)),
            "posmean": float(np.nanmean(posmeans)),
            "n": len(means),
        }
        rows.append(row)

        print(
            f"layer {layer:02d}: "
            f"mean={row['mean']:.6f}, "
            f"std={row['std']:.6f}, "
            f"median={row['median']:.6f}, "
            f"top10={row['top10']:.6f}, "
            f"posmean={row['posmean']:.6f}, "
            f"n={row['n']}"
        )

    if rows:
        best = max(rows, key=lambda x: x["mean"])
        summary[size] = best
        print(
            f"BEST {size}: "
            f"layer={best['layer']}, "
            f"mean={best['mean']:.6f}, "
            f"std={best['std']:.6f}, "
            f"median={best['median']:.6f}, "
            f"top10={best['top10']:.6f}, "
            f"posmean={best['posmean']:.6f}, "
            f"n={best['n']}"
        )

print("\n" + "=" * 100)
print("Scaling summary | fMRIPrep25 | sub-04~08 | best candidate layer by mean")
print("=" * 100)

for size in ["7B", "13B", "30B", "65B"]:
    if size not in summary:
        print(f"{size}: missing")
        continue
    x = summary[size]
    print(
        f"{size:>4} | "
        f"best_layer={x['layer']:>2} | "
        f"mean={x['mean']:.6f} | "
        f"std={x['std']:.6f} | "
        f"median={x['median']:.6f} | "
        f"top10={x['top10']:.6f} | "
        f"posmean={x['posmean']:.6f} | "
        f"n={x['n']}"
    )
    
PY
```

![image-20260610095939489](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260610095939489.png)

趋势是30B > 13B ≈ 7B > 65B



出现了点不太对，这里算的是候选组的r，让我们来算lun'wen'li