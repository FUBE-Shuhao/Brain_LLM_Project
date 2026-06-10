### 6.7



成功利用原版指令直接得到了sub-04的预处理数据,以及fsaveage5的结果,使用该结果重新进行岭回归,跑完几个规模的所有层



```
BIDS=/media/i9a2/WindowsData/Wsh/DeepPrep/input
OUT=/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/output1
WORK=/media/i9a2/WindowsData/Wsh/DeepPrep/work/fmriprep25_output1
TF=/media/i9a2/WindowsData/Wsh/DeepPrep/templateflow
FS_LICENSE=/media/i9a2/WindowsData/Wsh/DeepPrep/license.txt

docker run --rm -it --network host \
  -v "$BIDS":/data:ro \
  -v "$OUT":/out \
  -v "$WORK":/work \
  -v "$TF":/opt/templateflow \
  -v "$FS_LICENSE":/opt/freesurfer/license.txt:ro \
  -e TEMPLATEFLOW_HOME=/opt/templateflow \
  nipreps/fmriprep:25.0.0 \
  /data /out participant \
  --participant-label 04 \
  --fs-license-file /opt/freesurfer/license.txt \
  --output-spaces MNI152NLin2009cAsym fsaverage5 \
  --nthreads 16 \
  --omp-nthreads 8 \
  --mem-mb 64000 \
  -w /work \
  2>&1 | tee /media/i9a2/WindowsData/Wsh/Brain_LLM_Project/3_BrainData_Y/fmriprep25_sub04_test.log
```

![image-20260607200234995](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260607200234995.png)