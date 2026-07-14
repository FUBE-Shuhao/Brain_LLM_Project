2026/6/19第一次用codex进行修改，修改内容为：添加问答机制，包括选用模型名，模型大小，受试者规模等等
添加了归档，把run_universe.py和heads_vs_fmri,py放入Archive

更新了heads_vs_fmri_original_pathfixed.py以及run_full_original_llama7b_base.sh,旧的这两个文件重命名为heads_vs_fmri_original_pathfixed_old.py以及run_full_original_llama7b_base_old.sh




新添加的两个文件叫做run_interactive_original_fmri.sh heads_vs_fmri_original_pathfixed.py


这两轮我们主要把 fMRI 回归复现流程梳理清楚并做了脚本改造。
第一轮做了检查和判断：
看了论文 PDF、structrue.txt、utils/regression 里的相关脚本。
确认 structrue.txt 描述的预处理结果是合理的：50 个被试，排除 21/52，读任务 fsaverage5 GIFTI 共 500 个，即 50 × 5 runs × 2 hemispheres。
确认回归脚本使用的是 Data/fsaverage5/*_space-fsaverage5_bold.func.gii，输出每半球 10242 个顶点。
判断当前项目内 Data/fsaverage5 只有小样本，不是全量数据；全量要依赖你 Linux 工作站里的 /media/.../3_BrainData_Y/output1。
分析了 run_full_original_llama7b_base.sh：原先是硬编码跑 llama 7B base lh 全层全被试。
分析了 run_universe.py：它是另一套 robust launcher，调用 heads_vs_fmri.py 和 /4_Results，不是当前 exact-original/pathfixed 复现流。
给了 utils/regression 文件夹整理建议：保留主脚本、QC 脚本，归档旧版/测试文件，删除空文件和临时脚本。
第二轮做了实际修改：
修改了 [heads_vs_fmri_original_pathfixed.py](/c:/Users/wsh/Downloads/scaling_for_codex/scaling_finetuning-main(1\)/scaling_finetuning-main/utils/regression/heads_vs_fmri_original_pathfixed.py)：
支持环境变量覆盖 PROJECT_DIR / DATA_DIR / ATTN_DIR / OUTPUT_ROOT / N_JOBS。
增加模型/规模/layer 合法性检查。
增加 base/instr/ctrl 到 attention 前缀的映射。
检查 attention shape 和 head 数是否匹配。
检查 fMRI GIFTI 是否为 10242 个 fsaverage5 顶点。
保留原始算法逻辑和输出格式。

重写了 [run_full_original_llama7b_base.sh](/c:/Users/wsh/Downloads/scaling_for_codex/scaling_finetuning-main(1\)/scaling_finetuning-main/utils/regression/run_full_original_llama7b_base.sh)：
从固定 llama 7B base lh 改成交互式 launcher。
可以输入 subjects，例如 1-52、4,5,8。
默认使用论文排除 21,52。
可以选择模型族和规模。
默认推荐只跑 base，适合当前快速复现。
如果不只跑 base，会自动检测对应 instr/ctrl attention 文件是否存在，不存在就跳过，不乱问。
可以选择 lh/rh。
可以 dry-run，只打印计划不跑。
可以选择是否覆盖已有 .npy。
可以指定所有层或部分层。
新增自动从 /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/output1 复制/整理 fsaverage5 GIFTI 到 ${PROJECT}/Data/fsaverage5，并把 hemi-L/R 转成脚本需要的 hemi-lh/rh 文件名。
保留可选复制 events TSV 到 Data/{group} 的功能。

验证情况：
Python 脚本已通过 py_compile。
当前 Windows 环境没有真正可用的 Linux bash/依赖库，所以没有实际运行全量任务；这和你说的“Linux 工作站项目复制到 Windows 里给我看”一致。





使用方法：现在运行会像这样问：
Subjects to run, e.g. 1-52 or 4,5,8 [1-52]:
Use paper exclusion subjects 21 and 52? [Y/n]:

Which model family? Options: gpt2, llama, alpaca, vicuna, mistral, gemma [llama]:
Run llama 7B? [Y/n]:
Run llama 13B? [y/N]:
Run llama 30B? [y/N]:
Run llama 65B? [y/N]:

Run base group only for this pass? This is recommended for the current quick replication. [Y/n]:
你现在就直接回车选 base only。
然后：
Run left hemisphere (lh)? [Y/n]:
Run right hemisphere (rh)? [y/N]:

Copy/prepare fsaverage5 GIFTI files from PREPROC_DIR into PROJECT/Data/fsaverage5? [Y/n]:
Overwrite existing prepared fsaverage5 files in PROJECT/Data/fsaverage5? [y/N]:

Copy event TSV files from EVENT_SRC into PROJECT/Data/{group} before running? [y/N]:
Overwrite existing non-empty result .npy files? [y/N]:
Only print the planned jobs without running them? [y/N]:
N_JOBS for joblib inside each regression process (-1 means all cores) [-1]:

Run all layers for llama 7B (0-31)? [Y/n]:
我建议你第一次这样回答：
Subjects: 1-52
Use exclusion: 回车
Model: 回车  # llama
Run 7B: 回车
Run 13B/30B/65B: no
Run base only: 回车
lh: 回车
rh: no
Copy/prepare fsaverage5: 回车
Overwrite prepared fsaverage5: no
Copy event TSV: 视情况，如果 Data/base 已有就 no
Overwrite result: no
Dry run: yes
N_JOBS: 8 或回车
All layers: 回车




本次只运行了1-52，llama的7B13B30B

ok了 您先用

ok







由于冲突，在31受试者那里停了，根据检查，一个只有log是多余生成。

PROJECT="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main"
RESULT_ROOT="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_original_heads_vs_fmri"

cd "$PROJECT"

taskset -c 0-15 env \
  N_JOBS=16 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PROJECT="$PROJECT" \
  SCRIPT="$PROJECT/utils/regression/heads_vs_fmri_original_pathfixed.py" \
  PREPROC_DIR="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/output1" \
  EVENT_SRC="/media/i9a2/WindowsData/Wsh/DeepPrep/input" \
  RESULT_ROOT="$RESULT_ROOT" \
  bash "$PROJECT/utils/regression/run_interactive_original_fmri.sh"

  用如上命令再次运行，参数为：
  Subjects to run: 31
Use paper exclusion subjects 21 and 52? y

Which model family? llama

Run llama 7B? y
Run llama 13B? y
Run llama 30B? y
Run llama 65B? n

Run base group only? y
Run left hemisphere? y
Run right hemisphere? n

Copy/prepare fsaverage5...? y
Overwrite existing prepared fsaverage5 files...? n

Copy event TSV files...? y
Overwrite existing non-empty result .npy files? n

N_JOBS: 16
Run all layers for llama 30B? y
接下来进行1-30的岭回归
1.生成run_sub01_30_llama7b_base_lh_fmriprep25.sh用于1-30的7B

以上操作日期：2026：7：5



