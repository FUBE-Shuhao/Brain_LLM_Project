### 5.2

14:50

开始运行fMRIPrep，尝试先运行sub-01受试者。

文件位置：/media/i9a2/WindowsData/Wsh/DeepPrep/input_mini



14:53

运行：sudo docker run -ti --rm -v /media/i9a2/WindowsData/Wsh/DeepPrep/input_mini:/data:ro -v /media/i9a2/WindowsData/Wsh/DeepPrep/output_mini:/out -v /media/i9a2/WindowsData/Wsh/DeepPrep/work:/work -v /media/i9a2/WindowsData/Wsh/DeepPrep/license.txt:/opt/freesurfer/license.txt ghcr.io/nipreps/fmriprep:25.2.5 /data /out participant --participant-label 01 -w /work --fs-license-file /opt/freesurfer/license.txt --stop-on-first-crash --skip-bids-validation



<img src="./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260502145242053.png" alt="image-20260502145242053" style="zoom:50%;" />

持续到15:01运行正常



16：15

运行完毕

该受试者用时1h24min



![image-20260502163225618](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260502163225618.png)

19:40

开始运行01-52受试者





21:20 

第一个受试者运行完毕，但是因为权限问题会被卡住，使用新命令

```
for i in $(seq -w 01 52); do echo "=========================================================="; echo " 🚀 正在启动处理: sub-${i} "; echo "=========================================================="; docker run --rm -v /media/i9a2/WindowsData/Wsh/DeepPrep/input:/data:ro -v /media/i9a2/WindowsData/Wsh/DeepPrep/output:/out -v /media/i9a2/WindowsData/Wsh/DeepPrep/work:/work -v /media/i9a2/WindowsData/Wsh/DeepPrep/license.txt:/opt/freesurfer/license.txt ghcr.io/nipreps/fmriprep:25.2.5 /data /out participant --participant-label ${i} -w /work --fs-license-file /opt/freesurfer/license.txt --stop-on-first-crash --skip-bids-validation --nprocs 24 --omp-nthreads 8 --mem_mb 48000; echo "=========================================================="; echo " ✅ sub-${i} 处理完成，正在清理 work 缓存目录释放空间..."; rm -rf /media/i9a2/WindowsData/Wsh/DeepPrep/work/*; echo " 🗑️ 清理完毕。准备处理下一个..."; echo "=========================================================="; done
```



### 5.3

截止至13:35

处理到sub-12

<img src="./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260503133523191.png" alt="image-20260503133523191" style="zoom:50%;" />
