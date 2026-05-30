## HCCS (Huawei Cache Coherence System)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HCCS 是华为设计的高速芯片间互联总线，用于 Ascend NPU 服务器内部多芯片之间的缓存一致性通信。在 Ascend 910A 服务器中，每 8 个 NPU 分为两组，每组 4 个 NPU 通过 HCCS 互联，提供 256GB/s 的高带宽。HCCS 支持 cache-coherent 内存访问，使得多个 NPU 可以高效共享和传输数据。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 LocMoE 的 MoE 训练中，HCCS 起到关键作用：
- 节点内 8 个 NPU 通过 HCCS 互联，带宽远高于节点间 RoCE 网络
- Group-wise All-to-All 策略将通信拆分到 TP 域和 EP 域：TP 域内的 All-Gather 操作利用 HCCS 高带宽同步 token 数据
- Locality Loss 的目标是促使 token 优先路由到同节点本地 expert，从而将更多通信从跨节点 RoCE 转移到节点内 HCCS
- 这是 LocMoE 减少 All-to-All 通信时间 5.13% 的关键硬件基础

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HCCS 是 Ascend 服务器内部封闭的互联方案，通过 CANN 驱动和 HCCL 通信库对上提供通信接口。上层框架（如 MindSpore）无需直接操作 HCCS，而是通过 HCCL 的通信原语间接利用其高带宽特性。

涉及论文标题：
- LocMoE: A Low-overhead MoE for Large Language Model Training
