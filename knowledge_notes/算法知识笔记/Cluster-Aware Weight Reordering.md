## Cluster-Aware Weight Reordering

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cluster-aware weight reordering 是 sort-and-cluster 量化的配套 offline 预处理步骤：由于 sort-and-cluster 改变了激活的 channel/head 顺序（按聚类结果重排），为避免激活和权重的 channel 不匹配，必须对 SSM block 中所有与激活交互的权重矩阵进行对应的 offline 重排。具体重排：(1) input projection weights $W_{in}$ 的**列**按聚类索引排序；(2) causal conv1d weights 的**channel**按聚类索引排序；(3) normalization weights 按聚类索引排序；(4) output projection weights $W_{out}$ 的**行**按聚类索引排序，以恢复正确的输出顺序。因为 SSD 计算保持 channel order（channel order preserving），这些重排保证整个 block 的输出与未重排的 FP16 block 完全等价（compute-invariance）。重排是一次性的 offline 操作，不增加推理延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline reordering（一次执行）
cluster_indices = get_cluster_indices(channel_order, head_clusters, channel_clusters)

# 重排各权重
W_in_reordered = W_in[:, cluster_indices]              # 列重排
W_conv_reordered = W_conv[cluster_indices, :]           # channel 重排
W_norm_reordered = W_norm[cluster_indices]              # normalization 重排
W_out_reordered = W_out[cluster_indices, :]             # 行重排（等价于 P@W_out）
# 其中 P 是 permutation matrix from cluster_indices

# 对于 W4A8/W4A16，权重重排后再进行 GPTQ 优化
W_out_reordered_4bit = GPTQ(W_out_reordered, calib_data)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖 per-group scaling factor 的重新组织（按聚类结果排列），使用 fused dequant+matmul kernel 时需保证 scaling factor layout 与 weight layout 一致以最大化 Tensor Core 加载效率。论文基于 CUTLASS 实现，参考了 Marlin (Frantar et al. 2024) 的 weight 重排策略。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
