## Sort-and-Cluster Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sort-and-cluster（SnC）是 Quamba2 提出的针对 SSM 输入激活 $x_t$ 的 8-bit 量化方法，利用 SSM 的两个关键属性：(1) **channel persistence**——各 channel 的激活幅度在不同输入样本间保持一致；(2) **channel order preserving**——SSD 计算是 channel-wise 的，因此输入 channel 顺序等于输出 channel 顺序。SnC 通过 offline calibration 获取各 channel 的最大值，先对 head 内 channel 按最大值排序，再对 head 聚类（m 组），对每组 head 内的 channel 聚类（n 组），最终使用 $m \times n$ 个 scaling factor 量化 $x_t$。排序后的 head 嵌入被"解耦"（disentangle），使得具有相似激活特性的 head 自然聚在一起，从而提升组内量化精度。Quamba2 默认 m=4, n=4，即每层使用 16 个 scaling factor。该方法的额外开销：offline 时需对权重进行 cluster-aware reordering 以匹配新的 channel/head 顺序，online 时仅需按预计算 index 重排激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline calibration（一次执行）
for each block in model:
    for each calibration sample:
        record abs_max = max(|x_t[c]|) for each channel c
    channel_order = argsort(descending, abs_max)     # 按最大值降序排列
    heads_ordered = rearrange(heads, channel_order)
    head_clusters = kmeans(heads_ordered, m)          # m 组 head
    for each head_cluster:
        channel_clusters = kmeans(channels, n)         # n 组 channel
        for each (hc, cc) pair:
            s[hc][cc] = max_abs_value / 127.0          # INT8 scale

# Online inference（每 token）
x_sorted = x_t[channel_order]                         # 按预计算顺序重排
x_sorted = rearrange(x_sorted, head_clusters, channel_clusters)
x_quant = clamp(round(x_sorted / s), -127, 127)       # 8-bit 量化
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖：(1) calibration 数据集（Quamba2 使用 Pile 随机 512 句）；(2) 聚类算法（k-means 或类似无监督方法）；(3) offline weight reordering kernel（按聚类索引重排 W_in 列、W_out 行、W_conv channel、W_norm 参数）。论文发现 m=4, n=4 在所有实验中足够好，更大的 m/n 收益递减。与 MambaQuant 的 clipping 和 Quamba 的 percentile clipping 相比，SnC 在 Mamba2-8B 上提升约 4%（W8A8 设置下 69.8% vs 64.8% FP16 70.7%）。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
