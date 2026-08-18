## HCLOG（分组位打包压缩组件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LC framework（github.com/burtscher/LC-framework）中的压缩组件之一，属于"分组位打包"类：把 16 KB 数据块划分为固定数量的 32 个子块（sub-chunk），对每个子块计算其最小值前导零个数作为元数据，只存储各元素的有效位（低 bit），从而利用数据中高位全 0 的冗余。LC 框架是跨平台数据压缩工具库，提供多种压缩组件与预处理方法，其中 Reducer 是唯一用于缩短数据序列的组件（含 HCLOG、RLE、RRE、RZE 等）。ENEC 论文用改进的 LC 框架对模型权重做组件组合搜索（Observation 2：HCLOG 变体在多数模型上取得最高压缩比 98%+ 的胜出率），并扩展了 LC 框架——支持不同子块数量的一组 HCLOG 压缩器变体（因为单个 outlier 会迫使整个子块采用更高位宽，可调子块数可缓解）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# HCLOG 分组位打包（LC 框架内）
for sub_chunk in split(block_16KB, 32):
    leading_zeros = min_lz(sub_chunk)        # 子块内元素最小前导零数 = 元数据
    for v in sub_chunk:
        store_low_bits(v, width=16 - leading_zeros)  # 只存有效低位
```
Annotations：该思路与 ENEC 的分组位宽打包同源——都是"按组算一个公共位宽、只存有效位"；ENEC 进一步改成阈值量化（≤m 用 m、否则 n）+ 纯 OR/shift 的 lane folding，去掉乘除/规约。LC 搜索的价值：用少量代表性模型（deepseek-llm-7b-base、Llama-3.1-8B）离线确定哪种轻量压缩组件组合在模型权重上最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：LC 框架的 C++ 压缩组件库；ENEC 论文在 Section II-C 对其扩展（可变子块数 HCLOG 变体）并用它做最优组件搜索。使用：作为压缩组件组合搜索的候选集之一——论文观察到 HCLOG 在模型权重上几乎总是最优（98%+ 情况），这直接启发了 ENEC 采用分组位宽打包路线。局限：经典 HCLOG 的 reduction max 在 Ascend 向量单元上开销大（ENEC 分析占 40%），且单 outlier 拖累整子块位宽——ENEC 用两级阈值量化规避。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
