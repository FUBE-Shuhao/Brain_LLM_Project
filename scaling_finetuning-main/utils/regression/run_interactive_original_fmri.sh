#!/usr/bin/env bash
set -euo pipefail

# Interactive launcher for heads_vs_fmri_original_pathfixed.py.
# Defaults are optimized for the current replication phase: run base attention
# quickly, while still allowing later instr/ctrl runs only when their attention
# files actually exist for the chosen model/size.

PY="${PY:-python3}"
PROJECT="${PROJECT:-/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main}"
SCRIPT="${SCRIPT:-${PROJECT}/utils/regression/heads_vs_fmri_original_pathfixed.py}"
PREPROC_DIR="${PREPROC_DIR:-/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/output1}"
EVENT_SRC="${EVENT_SRC:-/media/i9a2/WindowsData/Wsh/DeepPrep/input}"
RESULT_ROOT="${RESULT_ROOT:-/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_original_heads_vs_fmri}"
N_JOBS="${N_JOBS:--1}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

ask_text() {
  local prompt="$1"
  local default="${2:-}"
  local answer

  if [ -n "$default" ]; then
    read -r -p "${prompt} [${default}]: " answer
    echo "${answer:-$default}"
  else
    read -r -p "${prompt}: " answer
    echo "$answer"
  fi
}

ask_yes_no() {
  local prompt="$1"
  local default="${2:-no}"
  local answer suffix

  case "$default" in
    yes) suffix="[Y/n]" ;;
    no) suffix="[y/N]" ;;
    *) die "ask_yes_no default must be yes or no, got: $default" ;;
  esac

  while true; do
    read -r -p "${prompt} ${suffix}: " answer
    answer="${answer:-$default}"
    answer="${answer,,}"
    case "$answer" in
      y|yes) return 0 ;;
      n|no) return 1 ;;
      *) echo "Please answer yes or no." ;;
    esac
  done
}

supported_sizes_for_model() {
  local model="$1"
  case "$model" in
    gpt2) echo "base medium large xlarge" ;;
    llama) echo "7B 13B 30B 65B" ;;
    alpaca) echo "7B 13B" ;;
    vicuna) echo "7B 13B" ;;
    mistral) echo "7B" ;;
    gemma) echo "7B" ;;
    *) return 1 ;;
  esac
}

layers_for_model_size() {
  local model="$1"
  local size="$2"
  case "${model}_${size}" in
    gpt2_base) echo 12 ;;
    gpt2_medium) echo 24 ;;
    gpt2_large) echo 36 ;;
    gpt2_xlarge) echo 48 ;;
    llama_7B|alpaca_7B|vicuna_7B|mistral_7B) echo 32 ;;
    llama_13B|alpaca_13B|vicuna_13B) echo 40 ;;
    llama_30B) echo 60 ;;
    llama_65B) echo 80 ;;
    gemma_7B) echo 28 ;;
    *) return 1 ;;
  esac
}

prefix_for_group() {
  local group="$1"
  case "$group" in
    base) echo "rb_p1" ;;
    instr) echo "instr_rb_p1" ;;
    ctrl) echo "ctrl_rb_p1" ;;
    *) return 1 ;;
  esac
}

expand_number_list() {
  local text="$1"
  local min="$2"
  local max="$3"
  local out=()
  local part a b i

  text="${text// /}"
  if [ -z "$text" ]; then
    return 1
  fi

  if [ "$text" = "all" ]; then
    for ((i = min; i <= max; i++)); do
      out+=("$i")
    done
  else
    IFS=',' read -r -a parts <<< "$text"
    for part in "${parts[@]}"; do
      if [[ "$part" =~ ^[0-9]+-[0-9]+$ ]]; then
        a="${part%-*}"
        b="${part#*-}"
        if ((a > b)); then
          die "Invalid range '$part': start is greater than end."
        fi
        for ((i = a; i <= b; i++)); do
          if ((i < min || i > max)); then
            die "Value $i is out of range ${min}-${max}."
          fi
          out+=("$i")
        done
      elif [[ "$part" =~ ^[0-9]+$ ]]; then
        i="$part"
        if ((i < min || i > max)); then
          die "Value $i is out of range ${min}-${max}."
        fi
        out+=("$i")
      else
        die "Invalid list item '$part'. Use examples like 1-5,8,10."
      fi
    done
  fi

  printf "%s\n" "${out[@]}" | awk '!seen[$0]++'
}

contains_number() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

subject_label() {
  printf "sub-%02d" "$1"
}

attention_group_available() {
  local model="$1"
  local size="$2"
  local group="$3"
  local prefix

  prefix="$(prefix_for_group "$group")"
  [ -f "${PROJECT}/model_attention/${model}/${size}/${prefix}_layer0.npy" ]
}

copy_events_if_needed() {
  local id="$1"
  local group="$2"
  local subj src_dir dst_dir n

  subj="$(subject_label "$id")"
  src_dir="${EVENT_SRC}/${subj}/func"
  dst_dir="${PROJECT}/Data/${group}/${subj}/func"

  [ -d "$src_dir" ] || die "Event source dir not found: $src_dir"
  mkdir -p "$dst_dir"
  cp -f "${src_dir}/${subj}_task-read_run-"*_events.tsv "$dst_dir/"

  n="$(find "$dst_dir" -maxdepth 1 -name "${subj}_task-read_run-*_events.tsv" | wc -l)"
  [ "$n" -ge 5 ] || die "Expected at least 5 event files for $subj, found $n in $dst_dir"
}

check_events() {
  local id="$1"
  local group="$2"
  local subj dir n

  subj="$(subject_label "$id")"
  dir="${PROJECT}/Data/${group}/${subj}/func"
  [ -d "$dir" ] || die "Event dir not found: $dir"
  n="$(find "$dir" -maxdepth 1 -name "${subj}_task-read_run-*_events.tsv" | wc -l)"
  [ "$n" -ge 5 ] || die "Expected at least 5 event files for $subj, found $n in $dir"
}

copy_fsaverage5_if_needed() {
  local id="$1"
  local subj hem run hemi_upper src dst src_json dst_json

  subj="$(subject_label "$id")"
  mkdir -p "${PROJECT}/Data/fsaverage5"

  for hem in "${HEMIS[@]}"; do
    case "$hem" in
      lh) hemi_upper="L" ;;
      rh) hemi_upper="R" ;;
      *) die "Unknown hemisphere: $hem" ;;
    esac

    for run in 1 2 3 4 5; do
      src="${PREPROC_DIR}/${subj}/func/${subj}_task-read_run-${run}_hemi-${hemi_upper}_space-fsaverage5_bold.func.gii"
      dst="${PROJECT}/Data/fsaverage5/${subj}_task-read_run-${run}_hemi-${hem}_space-fsaverage5_bold.func.gii"

      [ -f "$src" ] || die "Preprocessed fsaverage5 source not found: $src"
      if [ "$OVERWRITE_DATA" = "yes" ] || [ ! -s "$dst" ]; then
        cp -f "$src" "$dst"
      fi

      src_json="${src%.func.gii}.json"
      dst_json="${dst%.func.gii}.json"
      if [ -f "$src_json" ] && { [ "$OVERWRITE_DATA" = "yes" ] || [ ! -s "$dst_json" ]; }; then
        cp -f "$src_json" "$dst_json"
      fi
    done
  done
}

check_surface_files() {
  local subj="$1"
  local hem run f

  for hem in "${HEMIS[@]}"; do
    for run in 1 2 3 4 5; do
      f="${PROJECT}/Data/fsaverage5/${subj}_task-read_run-${run}_hemi-${hem}_space-fsaverage5_bold.func.gii"
      [ -f "$f" ] || die "Missing fsaverage5 surface file: $f"
    done
  done
}

check_attention_file() {
  local model="$1"
  local size="$2"
  local group="$3"
  local layer="$4"
  local prefix f

  prefix="$(prefix_for_group "$group")"
  f="${PROJECT}/model_attention/${model}/${size}/${prefix}_layer${layer}.npy"
  [ -f "$f" ] || die "Missing attention file: $f"
}

format_duration() {
  local seconds="${1:-0}"
  if [ "$seconds" = "unknown" ] || [ -z "$seconds" ]; then
    printf "unknown"
    return
  fi

  seconds=$((seconds < 0 ? 0 : seconds))
  local hours=$((seconds / 3600))
  local minutes=$(((seconds % 3600) / 60))
  local secs=$((seconds % 60))

  if ((hours > 0)); then
    printf "%d:%02d:%02d" "$hours" "$minutes" "$secs"
  else
    printf "%02d:%02d" "$minutes" "$secs"
  fi
}

render_overall_progress() {
  local finished="$1"
  local total="$2"
  local started_at="$3"
  local completed_runs="$4"
  local completed_seconds="$5"
  local current="$6"
  local width=34
  local now elapsed pct filled empty bar eta avg

  now="$(date +%s)"
  elapsed=$((now - started_at))

  if ((total > 0)); then
    pct="$(awk -v f="$finished" -v t="$total" 'BEGIN { printf "%.1f", 100*f/t }')"
    filled="$(awk -v f="$finished" -v t="$total" -v w="$width" 'BEGIN { printf "%d", (t ? int(w*f/t) : w) }')"
  else
    pct="100.0"
    filled="$width"
  fi
  empty=$((width - filled))
  bar="$(printf "%${filled}s" "" | tr ' ' '#')$(printf "%${empty}s" "" | tr ' ' '-')"

  if ((completed_runs > 0)); then
    avg="$(awk -v s="$completed_seconds" -v n="$completed_runs" 'BEGIN { printf "%.0f", s/n }')"
    eta=$((avg * (total - finished)))
  else
    eta="unknown"
  fi

  echo "Overall: [$bar] ${finished}/${total} (${pct}%) elapsed $(format_duration "$elapsed") ETA $(format_duration "$eta")"
  echo "Current: $current"
}

echo "============================================================"
echo "Interactive fMRI ridge regression launcher"
echo "Project       : $PROJECT"
echo "Script        : $SCRIPT"
echo "Preproc source: $PREPROC_DIR"
echo "Event source  : $EVENT_SRC"
echo "Result root   : $RESULT_ROOT"
echo "============================================================"

[ -f "$SCRIPT" ] || die "Regression script not found: $SCRIPT"
[ -d "$PROJECT" ] || die "Project dir not found: $PROJECT"
[ -d "${PROJECT}/Data" ] || die "Data dir not found: ${PROJECT}/Data"
[ -d "${PROJECT}/model_attention" ] || die "model_attention dir not found: ${PROJECT}/model_attention"

subjects_text="$(ask_text "Subjects to run, e.g. 1-52 or 4,5,8" "1-52")"
mapfile -t SUBJECTS < <(expand_number_list "$subjects_text" 1 99)
[ "${#SUBJECTS[@]}" -gt 0 ] || die "No subjects selected."

if ask_yes_no "Use paper exclusion subjects 21 and 52?" "yes"; then
  exclude_text="21,52"
else
  exclude_text="$(ask_text "Subjects to exclude, comma/range format or empty" "")"
fi

EXCLUDED=()
if [ -n "${exclude_text// /}" ]; then
  mapfile -t EXCLUDED < <(expand_number_list "$exclude_text" 1 99)
fi

while true; do
  MODEL="$(ask_text "Which model family? Options: gpt2, llama, alpaca, vicuna, mistral, gemma" "llama")"
  if supported_sizes_for_model "$MODEL" >/dev/null; then
    break
  fi
  echo "Unknown model family: $MODEL"
done

SIZES=()
for size in $(supported_sizes_for_model "$MODEL"); do
  default="no"
  if [ "$size" = "7B" ] || { [ "$MODEL" = "gpt2" ] && [ "$size" = "base" ]; }; then
    default="yes"
  fi
  if ask_yes_no "Run ${MODEL} ${size}?" "$default"; then
    SIZES+=("$size")
  fi
done
[ "${#SIZES[@]}" -gt 0 ] || die "No model sizes selected."

declare -A SIZE_TO_GROUPS
if ask_yes_no "Run base group only for this pass? This is recommended for the current quick replication." "yes"; then
  for size in "${SIZES[@]}"; do
    SIZE_TO_GROUPS["$size"]="base"
  done
else
  for size in "${SIZES[@]}"; do
    selected_groups=()
    echo "Available attention groups for ${MODEL} ${size}:"
    for group in base instr ctrl; do
      if attention_group_available "$MODEL" "$size" "$group"; then
        echo "  $group: available"
        default="no"
        [ "$group" = "base" ] && default="yes"
        if ask_yes_no "Run ${MODEL} ${size} ${group}?" "$default"; then
          selected_groups+=("$group")
        fi
      else
        echo "  $group: missing, skipped"
      fi
    done
    [ "${#selected_groups[@]}" -gt 0 ] || die "No group selected for ${MODEL} ${size}."
    SIZE_TO_GROUPS["$size"]="${selected_groups[*]}"
  done
fi

HEMIS=()
if ask_yes_no "Run left hemisphere (lh)?" "yes"; then
  HEMIS+=("lh")
fi
if ask_yes_no "Run right hemisphere (rh)?" "no"; then
  HEMIS+=("rh")
fi
[ "${#HEMIS[@]}" -gt 0 ] || die "No hemisphere selected."

PREPARE_FSAVERAGE5="no"
OVERWRITE_DATA="no"
if ask_yes_no "Copy/prepare fsaverage5 GIFTI files from PREPROC_DIR into PROJECT/Data/fsaverage5?" "yes"; then
  PREPARE_FSAVERAGE5="yes"
  [ -d "$PREPROC_DIR" ] || die "Preproc source dir not found: $PREPROC_DIR"
  if ask_yes_no "Overwrite existing prepared fsaverage5 files in PROJECT/Data/fsaverage5?" "no"; then
    OVERWRITE_DATA="yes"
  fi
fi

COPY_EVENTS="no"
if ask_yes_no "Copy event TSV files from EVENT_SRC into PROJECT/Data/{group} before running?" "no"; then
  COPY_EVENTS="yes"
  [ -d "$EVENT_SRC" ] || die "Event source dir not found: $EVENT_SRC"
fi

OVERWRITE="no"
if ask_yes_no "Overwrite existing non-empty result .npy files?" "no"; then
  OVERWRITE="yes"
fi

DRY_RUN="no"
if ask_yes_no "Only print the planned jobs without running them?" "no"; then
  DRY_RUN="yes"
fi

N_JOBS="$(ask_text "N_JOBS for joblib inside each regression process (-1 means all cores)" "$N_JOBS")"

declare -A SIZE_TO_LAYERS
for size in "${SIZES[@]}"; do
  n_layers="$(layers_for_model_size "$MODEL" "$size")"
  max_layer=$((n_layers - 1))
  if ask_yes_no "Run all layers for ${MODEL} ${size} (0-${max_layer})?" "yes"; then
    layer_text="all"
  else
    layer_text="$(ask_text "Layers for ${MODEL} ${size}, e.g. 0-3,8,10" "0-${max_layer}")"
  fi
  SIZE_TO_LAYERS["$size"]="$(expand_number_list "$layer_text" 0 "$max_layer" | paste -sd ' ' -)"
done

echo
echo "Preflight checks..."
TASK_COUNT=0
for id in "${SUBJECTS[@]}"; do
  if contains_number "$id" "${EXCLUDED[@]}"; then
    continue
  fi

  subj="$(subject_label "$id")"

  if [ "$PREPARE_FSAVERAGE5" = "yes" ]; then
    copy_fsaverage5_if_needed "$id"
  fi
  check_surface_files "$subj"

  for size in "${SIZES[@]}"; do
    for group in ${SIZE_TO_GROUPS[$size]}; do
      if [ "$COPY_EVENTS" = "yes" ]; then
        copy_events_if_needed "$id" "$group"
      fi
      check_events "$id" "$group"

      for layer in ${SIZE_TO_LAYERS[$size]}; do
        check_attention_file "$MODEL" "$size" "$group" "$layer"
        for hem in "${HEMIS[@]}"; do
          TASK_COUNT=$((TASK_COUNT + 1))
        done
      done
    done
  done
done

[ "$TASK_COUNT" -gt 0 ] || die "No runnable jobs after applying exclusions."

echo "Preflight OK. Planned jobs: $TASK_COUNT"
echo "Subjects : ${SUBJECTS[*]}"
echo "Excluded : ${EXCLUDED[*]:-(none)}"
echo "Model    : $MODEL"
echo "Sizes    : ${SIZES[*]}"
for size in "${SIZES[@]}"; do
  echo "Groups for ${MODEL} ${size}: ${SIZE_TO_GROUPS[$size]}"
  echo "Layers for ${MODEL} ${size}: ${SIZE_TO_LAYERS[$size]}"
done
echo "Hemis    : ${HEMIS[*]}"
echo "Results  : $RESULT_ROOT"
echo

if [ "$DRY_RUN" = "yes" ]; then
  echo "Dry run selected. Planned jobs:"
fi

JOB_INDEX=0
FINISHED_TASKS=0
COMPLETED_RUNS=0
COMPLETED_RUN_SECONDS=0
OVERALL_STARTED_AT="$(date +%s)"
for id in "${SUBJECTS[@]}"; do
  if contains_number "$id" "${EXCLUDED[@]}"; then
    echo "Skipping subject $id according to exclusion list."
    continue
  fi

  for size in "${SIZES[@]}"; do
    for group in ${SIZE_TO_GROUPS[$size]}; do
      for layer in ${SIZE_TO_LAYERS[$size]}; do
        for hem in "${HEMIS[@]}"; do
          JOB_INDEX=$((JOB_INDEX + 1))

          out_file="${RESULT_ROOT}/${MODEL}/${size}/${group}/layer${layer}/${group}_subj${id}_corr${hem}.npy"
          log_dir="${RESULT_ROOT}/logs_${MODEL}_${size}_${group}"
          log_file="${log_dir}/subj${id}_${hem}_layer${layer}.log"

          if [ "$OVERWRITE" = "no" ] && [ -s "$out_file" ]; then
            echo "[$JOB_INDEX/$TASK_COUNT] SKIP existing: $out_file"
            FINISHED_TASKS=$((FINISHED_TASKS + 1))
            render_overall_progress "$FINISHED_TASKS" "$TASK_COUNT" "$OVERALL_STARTED_AT" "$COMPLETED_RUNS" "$COMPLETED_RUN_SECONDS" "skipped subj=$id model=${MODEL}_${size} group=$group hem=$hem layer=$layer"
            continue
          fi

          render_overall_progress "$FINISHED_TASKS" "$TASK_COUNT" "$OVERALL_STARTED_AT" "$COMPLETED_RUNS" "$COMPLETED_RUN_SECONDS" "starting subj=$id model=${MODEL}_${size} group=$group hem=$hem layer=$layer"
          echo "[$JOB_INDEX/$TASK_COUNT] subj=$id model=${MODEL}_${size} group=$group hem=$hem layer=$layer"
          echo "  Output: $out_file"
          echo "  Log   : $log_file"

          if [ "$DRY_RUN" = "yes" ]; then
            continue
          fi

          mkdir -p "$log_dir"
          task_started_at="$(date +%s)"
          if PROJECT_DIR="$PROJECT" \
            DATA_DIR="${PROJECT}/Data" \
            ATTN_DIR="${PROJECT}/model_attention" \
            OUTPUT_ROOT="$RESULT_ROOT" \
            N_JOBS="$N_JOBS" \
              "$PY" "$SCRIPT" "$id" "$group" "$MODEL" "$size" "$hem" "$layer" 2>&1 | tee "$log_file"; then
            task_elapsed=$(($(date +%s) - task_started_at))
            FINISHED_TASKS=$((FINISHED_TASKS + 1))
            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
            COMPLETED_RUN_SECONDS=$((COMPLETED_RUN_SECONDS + task_elapsed))
            render_overall_progress "$FINISHED_TASKS" "$TASK_COUNT" "$OVERALL_STARTED_AT" "$COMPLETED_RUNS" "$COMPLETED_RUN_SECONDS" "finished subj=$id model=${MODEL}_${size} group=$group hem=$hem layer=$layer in $(format_duration "$task_elapsed")"
          else
            echo "FAILED subj=$id model=${MODEL}_${size} group=$group hem=$hem layer=$layer"
            echo "See log: $log_file"
            exit 1
          fi
        done
      done
    done
  done
done

total_elapsed=$(($(date +%s) - OVERALL_STARTED_AT))
echo "DONE. Finished planned interactive ridge-regression run in $(format_duration "$total_elapsed")."
