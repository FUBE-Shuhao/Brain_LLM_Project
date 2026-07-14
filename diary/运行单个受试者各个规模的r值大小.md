### 5.31



发现岭回归结果里部分规模的部分层出现全部为0的情况

![image-20260531142917105](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260531142917105.png)

并且从小规模到大规模出现r值倒吸的现象

在最怀疑的优先级是：

```
1. 65B / 30B attention 文件是否有质量问题或提取口径不同；
2. 当前 fsaverage5 是 Nilearn 从 MNI vol_to_surf 映射，不是论文 mri_vol2surf；
3. 比较的是 raw mean r，不是 normalized r；
4. 只看 sub-01/lh，个体差异可能很大；
5. 65B layer79 已经明确坏掉，说明大模型 attention 文件至少有局部异常。
```

经检查,model_attention文件夹里LLM的原始.npy文件并不全是0和无效值,可能是因为heads_vs_fmri_oringinal_code.py文件里使用的是下三角矩阵,而原始npy的下三角全为0



经排查,npy文件没有错误,但是发现npy文件里采用的是float16 而代码里没有考虑到float16的情况,检查后得知每一个文件都发生了偏差.





在 `heads_vs_fmri_original_pathfixed.py` 里，把这句：

```
X = np.nan_to_num(zscore(X.flatten(), nan_policy='omit')).reshape(X.shape)
```

改成：

```
X = X.astype(np.float64, copy=False)
X = np.nan_to_num(zscore(X.flatten(), nan_policy='omit')).reshape(X.shape)
```



通过float16数据改为float64,让数据格式合法



再次运行small bunch of clean subjects

得到结果

![image-20260602155020084](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260602155020084.png)

总体符合趋势.但是30B和65B的趋势接近甚至下降

接下来开始扩大范围重新运行

```
cd /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main/utils/regression

for sid in $(seq 4 20); do
  subj=$(printf "sub-%02d" "$sid")

  echo "============================================================"
  echo "Preparing clean subject $subj"
  echo "============================================================"

  mkdir -p /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main/Data/base/$subj/func

  cp -f /media/i9a2/WindowsData/Wsh/DeepPrep/input/$subj/func/${subj}_task-read_run-*_events.tsv \
  /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main/Data/base/$subj/func/

  declare -A jobs
  jobs["7B"]=21
  jobs["13B"]=37
  jobs["30B"]=58
  jobs["65B"]=11

  for size in 7B 13B 30B 65B; do
    layer=${jobs[$size]}

    out_file="/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results_original_heads_vs_fmri/llama/$size/base/layer$layer/base_subj${sid}_corrlh.npy"

    if [ -s "$out_file" ]; then
      echo "SKIP existing: sub=$sid size=$size layer=$layer"
      continue
    fi

    echo "Running $subj | llama $size | layer $layer | lh"
    python3 heads_vs_fmri_original_pathfixed.py $sid base llama $size lh $layer
  done

  unset jobs
done
```

