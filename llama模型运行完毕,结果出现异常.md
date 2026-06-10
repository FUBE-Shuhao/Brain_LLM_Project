#### 5.26

计算完毕

![image-20260526115928018](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260526115928018.png)

![image-20260526115906079](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260526115906079.png)





结果完全错误,黄金层居然是第一层,与预期结果严重不符,且r的相关性太低.



更新了之后出现,虽然最佳层在16,然而r值过高



后来发现是因为sanity check有问题,即是因为全脑的0值导致相关性过高,

修改了逻辑,忽略零值

heads_vs_fmri.py中

## 原始版本是：

```
model_train = RidgeCV(alphas=np.logspace(1, 3, 20)).fit(X_train.T, y_train)

y_predict = X_test.T @ model_train.coef_

r, _ = pearsonr(y_predict, y_test)

return float(np.nan_to_num(r))
```

------

## 先改成 “只在非零位置算 r”

直接替换成：

```
model_train = RidgeCV(alphas=np.logspace(1, 3, 20)).fit(X_train.T, y_train)

y_predict = X_test.T @ model_train.coef_

mask = y_test != 0

if mask.sum() > 5:
    r, _ = pearsonr(y_predict[mask], y_test[mask])
else:
    r = 0

return float(np.nan_to_num(r))
```

之后只跑了受试者1



![image-20260526225656185](./../../Documents/BaiduSyncdisk/笔记/图片文件/image-20260526225656185-1779807416941-1.png)

发现mean值异常的低,也就是说之前的值确实是因为0的干扰



之后还发现我们对于预处理数据的使用和原文里的方法并不相同,但是当务之急并不是此事所以暂时忽略,它只会略微影响图片质量.



除此之外,如截图所说,我们的数据维度与预计维度是不符合的



论文明确写的是：lower-triangle 拼接后的总长度是 **7,388 × nhead**。
 你现在是 **7,421**，差了 **33 个 word-pair samples**。



这个情况很可能是我们自己分解test_data.xlsx文件与原作者的分解方式不符合



最紧缺的东西就是原作者的words.csv





采用了一段代码来排查:

```
import pickle
import numpy as np

words_list = pickle.load(open(
    "/media/i9a2/WindowsData/Wsh/Brain_LLM_Project/scaling_finetuning-main/Analysis/words_list.p",
    "rb"
))

edges = [w * (w - 1) // 2 for w in words_list]

print("n sentences:", len(words_list))
print("max words:", max(words_list))
print("total edges:", sum(edges))
print("train edges:", sum(edges[:133]))
print("test edges:", sum(edges[133:]))

print("\n--- sentence-level words and edges ---")
for i, (w, e) in enumerate(zip(words_list, edges)):
    print(f"{i:03d}: words={w:2d}, edges={e:3d}")
```



