## Table-Batched Embedding（TBE，表批式嵌入）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Table-Batched Embedding（TBE，表批式嵌入）是 DLRM 等推荐模型中把"批量的多用户/多特征 embedding 表查表"合并成单个算子批处理的嵌入计算范式（源于 AutoShard [Zha et al., KDD'22] 与 Meta 生产栈）：一次 kernel 调用处理一个 batch 中所有用户的所有（稀疏）特征对多张 embedding 表的查找、求和/拼接，输出稠密特征供 interaction/MLP 使用。稀疏部分特征不规则、embedding 表超单卡容量（MTIA 300 的 150B 参数 DLRM 99% 在稀疏侧），故 TBE 是 memory-bound/instruction-bound，需要 embedding cache、索引 DMA（gather/scatter）与排序加速。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TBE 前向/反向的 kernel 计算过程（MTIA 300）：
```python
# 前向: 每个 (batch, feature) 的 sparse index → 查 embedding 表 → 聚合
for u in range(B):                          # 批量用户
    for f in features(u):                   # 稀疏特征
        idx = indices[u, f]                 # 连续子集映射到单输出索引（打包）
        emb = embed_table_gather(idx, table[f])   # 索引 DMA gather（专用 cache）
        out[u, f] = sum_or_concat(emb)
# 反向: 稀疏索引需排序使"连续子集 → 单一 embedding 表索引"
sorted_idx = radix_sort(indices)            # 硬件 radix sort（桶化+直方图）
grad = scatter_grad(sorted_idx, table_grad) # 索引 DMA scatter
```
MTIA 300 加速手段：dedicated embedding caches、硬件索引 DMA（scatter/gather）、硬件 radix sort（反向排序）；性能 TBE 前向 2.0×/1.6×、反向 2.1×/1.6×（几何均值 vs H100/H200）。注意 skewed 输入（多数索引指向同一特征）使算子变 cache-bound/instruction-bound 而非带宽-bound，性能不随 HBM 带宽线性扩展。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TorchRec 提供 TBE 高层实现（MTIA 300 上经 TorchInductor 编译）；GPU 上 AutoShard 做 embedding 表分片。MTIA 300 侧专用硬件（embedding cache/索引 DMA/radix sort）由编译器生成对应 kernel。使用场景：DLRM 训练/推理的稀疏特征处理（Facebook/Instagram 广告、短视频、好友流推荐）。信息缺口：论文未给出 TBE kernel 的具体缓存容量与索引 DMA 的并发度。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
