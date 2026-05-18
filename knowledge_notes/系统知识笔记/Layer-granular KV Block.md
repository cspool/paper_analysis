## Layer-granular KV Block

术语是什么？通过联网搜索让回答具体和精准。

Layer-granular KV Block是eLLM对vLLM PagedAttention的KV block结构的重构。vLLM原生的KV block将某个请求在**所有transformer layer**上的多个连续token的KV数据打包到同一个物理block中（即一个physical block跨所有layer）。eLLM将block改为按**F个连续layer**划分的更小单元：一个physical block只覆盖F个连续layer的token KV数据，而非全部layer。默认F=4，即每个block仅覆盖4个连续transformer layer。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Layer-granular KV Block在eLLM系统架构中的作用：

**动机**：vLLM的all-layer block在token-wise caching场景下存在两个问题：(1) 内部碎片——当请求只释放部分layer的KV时，all-layer block中未被释放的layer空间被浪费；(2) 粗粒度——无法按layer选择性recompute/swap。eLLM需要更细粒度的block来支持"per-layer KV release + recompute"。

**运转流程**（以Llama2-13B, F=4为例）：
1. Llama2-13B共40层，每F=4层一组共10组。每组layer的KV block独立分配和管理。
2. 当某请求r=0.4，决定释放前40%历史token的KV时，对所有10组block统一操作——释放对应token在这些layer group中的physical block。
3. decode到layer i时，scheduler通过map table查找该layer所属group（如layer 5属于group [5-8]）中cached token的physical block ID——仅需查找该group的映射，而非遍历所有layer。
4. F的选择影响trade-off：F过小（如F=1）——映射访问开销大（需维护更多block entry）；F过大（如F=all）——内存碎片增加（类似原vLLM）。论文实验后取F=4为默认。

**与PagedAttention兼容**：layer-granular block保留PagedAttention的page-based allocation机制——每个block仍是固定大小的物理内存页，block table仍是logical-to-physical映射。区别在于logical block的组织维度从"cross-all-layer token chunk"变为"F-layer token chunk"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：在vLLM PagedAttention基础上修改block allocator和block table结构。
1. Block分配器改为按layer group独立分配——为每个F-layer group维护独立的free block list。
2. Map table扩展为包含layer_id维度：`map_table[seq_id][token_id][layer_group_id] → physical_block_id`。
3. Block size保持不变（vLLM默认16 token），仅改变block覆盖的layer范围。
4. 论文报告map table layer lookup overhead < 0.015ms平均，可忽略。

涉及论文标题：
- High Throughput and Low Latency LLM Serving via Adaptive KV Caching
