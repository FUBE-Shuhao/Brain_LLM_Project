### 5.20

![54bca6095c0d8c4fd7e27cc681b5a9fd](./../../Documents/BaiduSyncdisk/笔记/图片文件/54bca6095c0d8c4fd7e27cc681b5a9fd.png)

计算完成。

接下来使用makesure_result.py来求r值

```python
import os
import numpy as np

# Define your base directory
base_dir = "/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/4_Results/gpt2/large"
num_layers = 36 # GPT-2 Large has 36 layers
layer_r_values = {}

for layer_idx in range(num_layers):
    layer_path = os.path.join(base_dir, f"layer{layer_idx}")
    
    # Skip if the folder doesn't exist
    if not os.path.exists(layer_path):
        print(f"Warning: Directory not found for {layer_path}")
        continue

    subj_r_means = []
    
    # Iterate through all subject files in the layer folder
    for file in os.listdir(layer_path):
        if file.startswith("base_subj") and file.endswith("_corrlh.npy"):
            file_path = os.path.join(layer_path, file)
            
            # Load the correlation array for the subject
            corr_array = np.load(file_path)
            
            # Calculate the mean r value for the subject
            # np.nanmean is used to ignore any vertices outside the brain mask (NaNs)
            subj_mean = np.nanmean(corr_array)
            subj_r_means.append(subj_mean)

    # Average the r values across all subjects for this specific layer
    if subj_r_means:
        layer_r_values[layer_idx] = np.mean(subj_r_means)

# Print the results for all 36 layers
print("--- Mean Pearson's r per Layer ---")
for layer, r_val in sorted(layer_r_values.items()):
    print(f"Layer {layer}: {r_val:.5f}")

# Identify and print the golden layer
if layer_r_values:
    golden_layer = max(layer_r_values, key=layer_r_values.get)
    best_r = layer_r_values[golden_layer]
    print("\n" + "="*30)
    print(f"🏆 GOLDEN LAYER: Layer {golden_layer} with r = {best_r:.5f} 🏆")
    print("="*30)
```

