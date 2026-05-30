## Expert-Sharding Parallelism (ESP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Sharding Parallelism (ESP / 专家分片并行) 是 MoE 训练中当 GPU 数量超过 expert 数量时引入的并行策略。当 P > E 时纯 EP 会导致部分 GPU 闲置。ESP 将每个 expert 的权重沿 hidden dimension 切分到 ESP group 内多张 GPU（类似 MP），使所有 GPU 参与计算。ESP 引入 ESP-AllGather（expert 计算前收集 token 分片）和 ESP-ReduceScatter（expert 计算后聚合输出并切分）两个集合通信操作。当 ESP group 对齐节点内 GPU 数时（如 8 卡 DGX），这两个操作为节点内通信（NVLink），与节点间 AlltoAll 可重叠执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 EP+ESP 组合的 kernel 执行流程（N_ESP=4）：

```
# 输入 tokens 已通过 A2A Dispatch 到达 expert 所在 GPU group
# Step 1: ESP-AllGather (intra-node)
for gpu in ESP_group:
    local_shard = tokens_on_gpu[gpu]
    full_input = AllGather(local_shard)   # [T/N_ESP, M] → [T, M]

# Step 2: Expert Computation (各GPU算自己的权重分片)
# W1 [M, H] 沿 H 维切分为 [M, H/N_ESP]
local_out = full_input @ W1_shard        # [T, H/N_ESP]
local_out = activation(local_out)
local_out = local_out @ W2_shard         # [T, M]

# Step 3: ESP-ReduceScatter (intra-node)
combined = ReduceScatter(local_out)      # 聚合+切分 → [T/N_ESP, M]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DeepSpeed-MoE 和 Tutel 均支持 ESP。FSMoE 通过 ExpertBase 抽象支持，用户设置 N_ESP 参数，调度器自动管理通信 placement。ESP 通信量随 N_ESP 增大而增加，FSMoE 在 N_ESP=N_MP=节点内 GPU 数的最常见配置下重点优化其与 AlltoAll 的协同调度。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
