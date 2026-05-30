## Deterministic ScatterAdd (确定性分散累加 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Deterministic ScatterAdd 是 LongCat-Flash 训练基础设施中的确定性梯度聚合 kernel。ScatterAdd 在 MoE backward pass 中承担关键角色——将各 expert 处理的 token 梯度按原始 token 位置聚合回去。默认 CUDA 实现因 input-output operand count 不匹配（多个 expert 可能向同一 token 位置写入梯度），强制单 compute unit 串行执行，导致最高 50x 减速。

LongCat-Flash 的 Deterministic ScatterAdd 使用 hierarchical reduction algorithm：先将梯度按 token 位置分组，然后在各 processor 间并行规约，再按确定性顺序合并。结果在保证确定性的同时，性能达到与非确定性版本持平。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Deterministic ScatterAdd 原理

// 输入:
//   grad_per_expert: [num_experts, num_tokens_per_expert, d_model]  # 各 expert 产生的梯度
//   token_to_expert_map: [num_experts, num_tokens_per_expert]  # token → expert routing info
//   original_token_order: [batch, seq_len]

// Step 1: 按 destination token 分组 (parallel across processors)
for proc_id in range(num_processors):
    local_buckets = [[] for _ in range(max_token_id)]
    for expert_grad, token_id in my_assigned_range:
        local_buckets[token_id].append(expert_grad)

// Step 2: 每个 processor 内规约
for token_id in local_buckets.keys():
    local_reduced[token_id] = sum(local_buckets[token_id])

// Step 3: Processor 间按确定性顺序合并 (hierarchical reduction)
// 而不是 atomicAdd (非确定性)
sorted_procs = sort_by_id(processors)  // 确定性顺序
for token_id in range(max_token_id):
    result = zeros(d_model)
    for proc_id in sorted_procs:
        result += local_reduced[proc_id][token_id]
    output[token_id] = result
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 与默认 CUDA ScatterAdd（如 `scatter_add_()`）单 compute unit 串行执行对比，hierarchical reduction 将工作分配到所有 processor → 消除 50x 减速。
2. 确定性保证：按 processor ID 升序合并（而非依赖硬件 timing），确保相同输入在不同 run 下 bitwise 一致。
3. 在 LongCat-Flash 中的地位：与 Deterministic FAG 一起构成端到端确定性训练的 backward pass 组件。
4. 通用性：hierarchical reduction 思想不仅适用于 MoE token 梯度聚合，也可用于其他以 ScatterAdd 为 bottleneck 的操作。

涉及论文标题：
- LongCat-Flash Technical Report
