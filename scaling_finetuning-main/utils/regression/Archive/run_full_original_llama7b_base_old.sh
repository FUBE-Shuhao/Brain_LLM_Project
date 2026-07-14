#!/usr/bin/env bash
set -euo pipefail
#該腳本作爲trigger激活heads_vs_fmri_original_pathfixed.py，用於全量運行所有受試者的所有層（注意將for hem in lh rh; do改爲了for hem in lh; do，也就是縣跑了左腦，右腦等論文結束以後也可以跑）
PY=python3

SCRIPT="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main/utils/regression/heads_vs_fmri_original_pathfixed.py"
PROJECT="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main"

EVENT_SRC="/media/i9a2/WindowsData/Wsh/DeepPrep/input"
EVENT_DST="${PROJECT}/Data/base"

OUTPUT_ROOT="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_original_heads_vs_fmri/llama/7B/base"

LOG_DIR="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_original_heads_vs_fmri/logs_llama7b_base"
mkdir -p "$LOG_DIR"

# 论文 fMRI 分析排除 21 和 52
SKIP_SUBJECTS="21 52"

should_skip() {
  local id="$1"
  for s in $SKIP_SUBJECTS; do
    if [ "$id" = "$s" ]; then
      return 0
    fi
  done
  return 1
}

copy_events() {
  local id="$1"
  local subj
  subj=$(printf "sub-%02d" "$id")

  local src_dir="${EVENT_SRC}/${subj}/func"
  local dst_dir="${EVENT_DST}/${subj}/func"

  if [ ! -d "$src_dir" ]; then
    echo "ERROR: event source dir not found: $src_dir"
    exit 1
  fi

  mkdir -p "$dst_dir"

  cp -f "${src_dir}/${subj}_task-read_run-"*_events.tsv "$dst_dir/"

  local n
  n=$(find "$dst_dir" -maxdepth 1 -name "${subj}_task-read_run-*_events.tsv" | wc -l)

  if [ "$n" -lt 5 ]; then
    echo "ERROR: expected 5 event files for $subj, found $n in $dst_dir"
    exit 1
  fi
}

check_surface_files() {
  local id="$1"
  local subj
  subj=$(printf "sub-%02d" "$id")

  for hem in lh; do
    for run in 1 2 3 4 5; do
      local f="${PROJECT}/Data/fsaverage5/${subj}_task-read_run-${run}_hemi-${hem}_space-fsaverage5_bold.func.gii"
      if [ ! -f "$f" ]; then
        echo "ERROR: missing surface file: $f"
        exit 1
      fi
    done
  done
}

for id in $(seq 1 52); do
  if should_skip "$id"; then
    echo "Skipping subject $id according to paper exclusion."
    continue
  fi

  subj=$(printf "sub-%02d" "$id")
  echo "============================================================"
  echo "Preparing $subj"
  echo "============================================================"

  copy_events "$id"
  check_surface_files "$id"

  for hem in lh; do
    for layer in $(seq 0 31); do

      out_file="${OUTPUT_ROOT}/layer${layer}/base_subj${id}_corr${hem}.npy"
      log_file="${LOG_DIR}/subj${id}_${hem}_layer${layer}.log"

      if [ -s "$out_file" ]; then
        echo "SKIP existing: $out_file"
        continue
      fi

      echo "------------------------------------------------------------"
      echo "Running subj=$id hem=$hem layer=$layer"
      echo "Output: $out_file"
      echo "Log   : $log_file"
      echo "------------------------------------------------------------"

      "$PY" "$SCRIPT" "$id" base llama 7B "$hem" "$layer" 2>&1 | tee "$log_file"
    done
  done
done

echo "DONE full run: llama 7B base, subjects 1-52 excluding 21 and 52, lh+rh, layers 0-31."