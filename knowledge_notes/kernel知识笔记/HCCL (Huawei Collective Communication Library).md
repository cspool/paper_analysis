## HCCL (Huawei Collective Communication Library)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HCCL 是华为开发的高性能集合通信库，基于 Ascend NPU 平台，类似 NVIDIA NCCL 的角色。提供单节点多卡和多节点多卡的集合通信原语，包括 All-to-All、All-Gather、All-Reduce、Reduce-Scatter、Broadcast 等。支持多种通信算法：ring、mesh、HD (Hierarchical Decomposition)、ring+HD、mesh+HD。通信底层基于 PCI-E、HCCS（节点内）和 RoCE（节点间）高速链路。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 LocMoE 的 PanGu-Σ MoE 训练中，HCCL 执行以下通信模式：

```
# MoE 层的通信-计算流程（使用 HCCL 原语）
# 假设 128 Ascend 910A NPU, EP=16, TP=8

# Phase 1: All-to-All Dispatch（将 token 发送到对应 expert 的设备）
# Group-wise: 按 TP 域拆分 All-to-All
for tp_group in range(num_tp_groups):
    # 每个 TP group 内的 device 负责 EP domain 的部分通信
    local_tokens = tokens[tp_group * local_batch : ...]
    # HCCL All-to-All: 在 EP domain 内交换 token 数据
    dispatched = HCCL.all_to_all(local_tokens, expert_idx, ep_group)
    
# Phase 2: All-Gather in TP domain
# 利用 HCCS 256GB/s 高带宽在 TP 域同步
all_tokens = HCCL.all_gather(dispatched, tp_group)

# Phase 3: Expert FFN 计算（与通信重叠）
# FFN kernel 切片与下一轮 All-to-All 重叠
for micro_batch in split(local_tokens):
    expert_output = expert_ffn_kernel(micro_batch)  # AI Core 执行
    # HCCL All-to-All combine 与计算流水线重叠
```

论文图 2 显示了 HCCL 各通信算子在 64N/128N/256N 下的算法带宽。随着节点数增加，All-to-All 带宽瓶颈加剧（跨节点 RoCE vs 节点内 HCCS 的带宽差异）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HCCL 通过 CANN 驱动层与硬件交互，对上提供标准集合通信 API。在 MindSpore 中，HCCL 通信原语通过框架的通信后端自动调用，用户通过配置并行策略（EP/TP/DP）间接使用。HCCL 的算法选择（ring vs mesh vs HD）可由环境变量或配置文件控制，不同算法在不同通信模式和集群拓扑下有各自的性能优势。

涉及论文标题：
- LocMoE: A Low-overhead MoE for Large Language Model Training
