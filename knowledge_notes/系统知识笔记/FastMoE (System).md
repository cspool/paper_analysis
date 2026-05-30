## FastMoE (System)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FastMoE（He et al., 2021）是清华大学 PACMAN 团队开发的 MoE 分布式训练系统，提供了基于 PyTorch 的灵活 expert parallelism 实现。FastMoE 将 MoE 层的关键操作（gate、dispatch、combine、expert compute）模块化，支持自定义 expert 数量和每 token 激活 expert 数。FasterMoE 基于 FastMoE 构建，在其上添加了 dynamic shadowing、smart scheduling 和 topology-aware gate 三项优化。FasterMoE 默认所有优化关闭时行为等同于 FastMoE。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# FastMoE 的 MoE layer 执行流程 (baseline expert parallelism):
# 输入: tokens [B, H] on each worker

# 1. Gate (本地计算):
scores = W_gate @ tokens               # [B, E]
selected = topk(scores, k=2)           # [B, 2] expert indices

# 2. All-to-all Dispatch (跨 worker 通信):
send tokens to target expert workers via NCCL all-to-all

# 3. Expert Compute (本地):
for each expert on this worker:
    output += expert_ffn(received_tokens)

# 4. All-to-all Combine (跨 worker 通信):
return results to origin workers via NCCL all-to-all
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GitHub: https://github.com/laekov/fastmoe。FasterMoE 的优化通过环境变量控制开关（默认关闭 = FastMoE 行为）。FastMoE 支持 custom gate 机制，FasterMoE 的 topology-aware gate 即作为 custom gate 实现。也提供 single-worker 模式用于集成 DeepSpeed ZeRO。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
