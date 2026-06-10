### 5.22



更新算法,在循环之前先嗅探出所有应该要跑的路径,并且改善了关于group的健壮性,不会乱报错.

更新后的run_universe.py为:

```python
import os
import subprocess
import sys

# 1. 基础路径配置
PROJECT_ROOT = "/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main"
ATTN_DIR = os.path.join(PROJECT_ROOT, "model_attention")
RESULTS_DIR = "/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results"

# ==========================================
# 2. 🎯 精准控制台
# ==========================================
TARGET_MODELS = ["llama"]          
TARGET_SIZES  = ["13B"]            
TARGET_GROUPS = ["base"]           
TARGET_LAYERS = []                 

EXCLUDE_SUBJS = [21, 52]           

print("🚀 === 精准打击版算力流水线 (极速优化版) 准备就绪 ===")

# ==========================================
# 3. 🔍 预编译执行计划 (AOT Compilation)
# 将所有不依赖 subj 的嗅探操作全部提前！
# ==========================================
print("📡 正在执行全局目录嗅探，预生成任务清单...")
valid_tasks = [] # 存放绝对有效的组合：(name, size, layer, group)

for name in TARGET_MODELS:
    for size in TARGET_SIZES:
        specific_attn_dir = os.path.join(ATTN_DIR, name, size)
        
        if not os.path.exists(specific_attn_dir):
            print(f"  ⚠️ 警告: 找不到目录 {specific_attn_dir}，已跳过。")
            continue
        
        # 全局扫描层数
        detected_layers = []
        for f in os.listdir(specific_attn_dir):
            if f.endswith('.npy') and 'layer' in f:
                try:
                    detected_layers.append(int(f.split('layer')[-1].split('.npy')[0]))
                except ValueError:
                    continue
        
        detected_layers = sorted(list(set(detected_layers)))
        run_layers = [l for l in detected_layers if l in TARGET_LAYERS] if TARGET_LAYERS else detected_layers
        
        # 全局嗅探源文件存在性
        for layer in run_layers:
            for group in TARGET_GROUPS:
                if group == 'base':
                    src_prefix = 'rb_p1'
                elif group == 'instr':
                    src_prefix = 'instr_rb_p1'
                elif group == 'ctrl':
                    src_prefix = 'ctrl_rb_p1'
                else:
                    continue
                
                src_file = os.path.join(specific_attn_dir, f"{src_prefix}_layer{layer}.npy")
                
                # 只有源文件确实存在，才加入最终执行清单
                if os.path.exists(src_file):
                    valid_tasks.append((name, size, layer, group))

print(f"✅ 任务清单预编译完成！共锁定 {len(valid_tasks)} 种有效的子任务组合。")
print("==========================================")

# ==========================================
# 4. ⚡ 极速主循环 (告别多余 I/O)
# ==========================================
for subj in range(1, 53):
    if subj in EXCLUDE_SUBJS:
        print(f"🛡️ [隔离] 主动跳过黑名单受试者: sub-{subj:02d}")
        continue
        
    print(f"\n🧠 开始处理受试者: sub-{subj:02d}")
    
    # 直接遍历事先准备好的有效任务清单
    for name, size, layer, group in valid_tasks:
        
        # 目标文件断点续传判定
        target_file = f"{RESULTS_DIR}/{name}/{size}/{group}/layer{layer}/{group}_subj{subj}_corrlh.npy"
        
        if os.path.exists(target_file):
            print(f"  ⏭️ [跳过] {name}_{size} -> {group} -> Layer {layer} (战果已存在！)")
            continue

        print(f"  ▶ 计算中: {name}_{size} -> {group} -> Layer {layer} ...")
        
        # 构建标准外部调用指令
        cmd = [
            "python3", "heads_vs_fmri.py",
            str(subj), group, name, size, "lh", str(layer)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 容错监控
        if result.returncode != 0:
            print(f"  ❌ 异常: sub-{subj} | {name}_{size} | {group} | Layer {layer}")
            print(f"  错误日志: {result.stderr.strip()}")

print("\n🏁 [任务完成] 所有指定配置的算力倾泻已安全结束！")
```





17:00 开始全新运行,速度大概为一名受试者/小时

![image-20260522171938028](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260522171938028.png)

### 5.22-5.24

依旧正常运行



![image-20260524153702770](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260524153702770.png)

