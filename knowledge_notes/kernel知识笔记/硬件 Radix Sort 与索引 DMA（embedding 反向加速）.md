## 硬件 Radix Sort 与索引 DMA（embedding 反向加速）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
硬件 Radix Sort 是 MTIA 300 SFU（SIMD 引擎）内的专用排序单元，用于加速 embedding 反向（backward）的稀疏索引重排；索引 DMA（indexed DMA）是 Fabric Interface/Command Processor 支持的按索引列表做 gather/scatter 的 DMA（含字节对齐 tensor 切片 DMA）。动机：DLRM 前向把稀疏 offsets 与 indices 打包成"单个输出索引映射到连续输入子集"，而反向需把稀疏 indices 排序成"连续子集映射到单一 embedding 表索引"——排序是 embedding 反向的关键前置；MTIA 300 用硬件 radix sort 替代软件排序。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
radix sort 的硬件执行（MTIA 300 描述）：从 LS 取元素 → 按位桶化（bucketization）→ 生成直方图（histograms）→ 把桶化元素写回内存。与索引 DMA 组合的 embedding 反向流程：
```python
# 1. 硬件 radix sort: LS 元素 → 桶化 → 直方图 → 写回（SFU）
sorted_idx = radix_sort(indices)            # 连续子集 → 单一 embedding 索引
# 2. 索引 DMA: Command Processor 用 LS 中的索引列表生成读/写序列
grad = indexed_dma_gather(sorted_idx)       # 按索引 gather 梯度
embed_grads = indexed_dma_scatter(grad)     # 按索引 scatter 到表
```
索引 DMA 也用于前向 embedding 查表（scatter/gather），字节对齐切片 DMA 消除 tensor 切片布局变换的软件开销。TBE 前向 2.0×/1.6×、反向 2.1×/1.6×（vs H100/H200）部分归因于此。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：radix sort 为 SFU 内固定功能单元（按位桶化 + 直方图 + 写回）；索引 DMA 在 FI/CP 中由硬件生成读/写序列（索引列表存 LS）。使用场景：DLRM 稀疏侧前向（embedding 查表 gather/scatter）与反向（梯度 scatter + 索引排序）；与 TBE/embedding cache 配套。信息缺口：论文未给出 radix sort 的位宽/基数（radix）与桶数量细节。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
