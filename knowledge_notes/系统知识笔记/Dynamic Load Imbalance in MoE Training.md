## Dynamic Load Imbalance in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Load Imbalance 是 MoE 分布式训练的核心挑战之一。由于训练数据自然服从偏斜分布，某些 expert 会接收到远超平均数量的 token（"热门 expert"），而其他 expert 接收的 token 极少。在 expert parallelism 下，热门 expert 所在的 worker 计算负载（GeMM batch size B_w）远大于其他 worker，导致部分 worker 计算时间主导整体延迟而其他 worker 空闲。FasterMoE（PPoPP'22）观测到：在 16 expert 的 MoE 模型中，4/16 expert 处理约 20% 的所有 tokens（3.2× 平均值），且此热度分布随 training iteration 动态变化（快速变化的不均匀分布和少量稳定但长期不受欢迎的 expert）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# Expert parallelism 中的负载不均衡示例 (16 experts, 16 GPUs)
# Forward pass of one MoE layer:

# Gate 路由结果 (某 iteration):
# Expert 0: 500 tokens → GPU 0 重载
# Expert 1: 30 tokens  → GPU 1 轻载
# Expert 2: 25 tokens  → GPU 2 轻载
# ... (其余 experts 各 5-20 tokens)

# All-to-all dispatch 后各 GPU 的 batch size:
# B = [500, 30, 25, 15, 10, 8, 12, 8, 12, 10, 7, 5, 8, 6, 5, 10]

# 端到端延迟 (同步模式):
# Lat_e2e = max_w{Lat_comp(B_w)} + max_w{Lat_comm(B_w)}
# = Lat_comp(500) + Lat_comm(500)  ← GPU 0 主导

# GPU 1-15 在 GPU 0 完成前空闲:
# idle_time = Lat_comp(500) - Lat_comp(B_w)  (w=1..15)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

解决方案包括：(1) FasterMoE 的 Dynamic Shadowing——将热门 expert 参数复制到所有 worker 避免 token 传输；(2) GShard 的 auxiliary load balancing loss——在训练 loss 中加入 expert 均衡项；(3) BASE Layer 的 matching 算法——将 tokens 均匀分配到 experts；(4) Tutel 的自适应 expert capacity——动态调整每个 expert 的处理容量。

HarMoEny 将此概念从 training 扩展到 **inference** 场景，并提出根本不同的解决思路：
- **Inference-specific 特征**：Inference 中 expert popularity skew 具有 "batch-to-batch 动态性"——因输入 domain 变化（medical vs programming）导致。HarMoEny 实验表明连续 batch 间 throughput 可下降 37.6%，使得 profiling-based 方案（ExFlow integer programming，8.5-45min）完全不可行。
- **实证发现**：在 GPUs 间负载完全均衡时，all-to-all synchronization 仅占 execution time 的 2%（Section 3），推翻了 "all-to-all 通信是主要瓶颈" 的假设，将问题根源指向 load imbalance。
- **HarMoEny 方案**：通过 online token rebalancing（per-batch, 无需 profiling）+ async expert prefetching 实现 near-perfect 负载均衡，GPU idle time 从 82.6% 降至 2.6%，scheduler 开销仅为 20-31% of layer latency。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- HarMoEny: Efficient Multi-GPU Inference of MoE Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

ZeRO（Zero Redundancy Optimizer, Rajbhandari et al., 2020）是 Microsoft DeepSpeed 中的内存优化技术，通过将 optimizer states、gradients 和 parameters 沿 data parallel 维度切分到不同 worker 来消除数据并行中的内存冗余。分三个阶段：(1) ZeRO-1 仅切分 optimizer states；(2) ZeRO-2 额外切分 gradients；(3) ZeRO-3 切分 optimizer states、gradients 和 parameters 全部三部分，使每 worker 仅持有模型参数的 1/N 分片。FasterMoE 使用 ZeRO-3 作为数据并行 baseline——纯数据并行无法容纳超出单 GPU 内存的大模型，而 ZeRO-3 虽能容纳但引入了大量通信开销（梯度 all-reduce、参数 broadcast/gather），导致 DDL-Roofline 中 R_CC 极低。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# ZeRO Stage 3 in an MoE training forward pass:
# 非 MoE 层 → 数据并行 (ZeRO-3)
# MoE 层 → expert parallelism (FastMoE)

# ZeRO-3 每 iteration 的通信 (非 MoE 层):
# Forward: all-gather parameters → compute → (discard params)
# Backward: all-gather parameters → compute gradients → reduce-scatter gradients
# Optimizer step: each worker updates its own partition

# 通信量对比 (MoE MLP 层, N workers):
# ZeRO-3 (data parallel): all-reduce gradients 2N·α·H²
# Expert parallelism: 4× all-to-all, total B·H per all-to-all
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DeepSpeed GitHub: https://github.com/microsoft/DeepSpeed。FasterMoE 使用 DeepSpeed v0.4.4 的 ZeRO 实现，在 MoE 模型上 ZeRO-3 将模型复制到所有 worker 后进行数据并行训练。FastMoE 的 single-worker 版本与 DeepSpeed ZeRO 集成。实验显示（MoE-GPT 3.42B, 16 GPU）：FasterMoE vs ZeRO-3 加速 6.63× (johnny) / 17.87× (trevor)。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
