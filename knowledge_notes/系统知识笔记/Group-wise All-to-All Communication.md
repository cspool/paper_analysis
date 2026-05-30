## Group-wise All-to-All Communication

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Group-wise All-to-All 是 MindSpore 框架内置的一种分层集合通信算法，专为 MoE 的 All-to-All 通信优化设计。核心思想是将全量 All-to-All 通信拆分为多个维度并行执行：在 EP (Expert Parallel) 域进行实际 token 交换 → 在 TP (Tensor Parallel) 域通过 All-Gather 同步 token 数据。由于 TP 域通常在同一节点内（利用 HCCS 256GB/s 高带宽），而 EP 域可能跨节点（RoCE 低带宽），这种拆分将部分通信量从低带宽路径转移到高带宽路径。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 LocMoE 的 PanGu-Σ 128 Ascend 910A NPU 训练中（EP=16, TP=8）：

```
# 标准 All-to-All:
# 所有 128 个 device 之间全量交换 token
# 问题：跨节点传输小数据块，带宽利用率低
All_to_All(all_devices_128, tokens)

# Group-wise All-to-All:
# Step 1: 在 TP domain 内 (8 devices per group)
# 将 token 数据按 EP domain 拆分
for ep_rank in range(EP_SIZE=16):
    # 每个 device 只负责其 EP rank 对应部分的 All-to-All
    local_send = prepare_tokens_for_ep(ep_rank)

# Step 2: HCCL All-to-All 在 EP domain (16 devices, 可能跨节点)
# 实际 token 交换仅发生在 EP domain
HCCL_all_to_all(ep_group, local_send)

# Step 3: HCCL All-Gather 在 TP domain (8 devices, 节点内 HCCS)
# 在 TP 域同步完整的 token 数据，利用 HCCS 高带宽
HCCL_all_gather(tp_group, ep_received_data)
```

这种分层设计的优势：
- EP 域通信量减少（仅传输必要部分）
- TP 域通信利用节点内 HCCS 高带宽
- 可与 FFN 计算切片重叠（slice-and-overlap）：通信启动后立即开始 FFN 计算，通过流水线 mask 延迟

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Group-wise All-to-All 是 MindSpore 框架内置的 group-wise exchange 算法实现。用户通过框架的并行策略配置（如设置 TP=8, EP=16）间接启用，无需手动调用。在 LocMoE 中，该方法配合通信-计算重叠进一步减少 All-to-All 通信开销。All-to-All 通信时间占总训练时间的 18.10%（128N）和 28.74%（256N），经过 Group-wise + overlap 优化后降低 5.13%。

涉及论文标题：
- LocMoE: A Low-overhead MoE for Large Language Model Training
