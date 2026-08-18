## Sub-batch 调度（跨设备子批调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sub-batch 调度把请求 batch 划分为两个（或更多）子批交错执行：一个子批的 attention 在加速器上执行的同时，另一个子批的 FC 在 GPU 上执行。动机：单 batch 内 FC 依赖 attention 的输出（GPU 必须等 attention 完成），两设备存在串行依赖与空闲；拆成子批后跨 batch 并行，提升两设备利用率。该技术源自 AFD 系统（NeuPIMs/NEO 等），CHIME-sys 继承之。代价：子批使单侧 batch 减半，batch 偏小时 GPU 利用率下降——CHIME 实测 HBM-PIM 的 sub-batch 在 OPT-66B+Dolphin trace 上吞吐反而低于 GPU-only。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CHIME 的两子批模型（rank 粒度）：T_GPU_i = t_p(chunk 列表, 已完成 token) + t_batch(∑chunk, 另一子批的 decode 请求数)（GPU 侧 = prefill attention + 批 FC）；T_PIM_i = t_d(各 rank 已完成 token) + t_comm（PIM 侧 = decoding attention + 与另一子批重叠的传输）。时间线上 sub-batch 0 在 GPU 算 FC 时 sub-batch 1 在 PIM 算 attention，交替推进。理想情况两子批等长无气泡；不等长则快的一侧空转，这正是 CHIME 用 alignment-predicting 调度要消除的问题（对齐 T_GPU≈T_PIM）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：运行时维护请求队列，按策略把请求分入各子批，为每子批在各自设备上调度执行；与 continuous batching（vLLM/SGLang）在单设备内混排 prefill/decode 不同，sub-batch 调度面向跨设备的并行。使用方式：AFD 系统的标准吞吐优化；配套手段包括通信计算重叠、延迟预测对齐（CHIME）、负载均衡（KV 按 layer 粒度 interleaved 存放）。局限：加速器侧容量不足时子批可容纳请求更少、放大 batch 缩小问题（CHIME 用 2TB DIMM-PIM 缓解）。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
