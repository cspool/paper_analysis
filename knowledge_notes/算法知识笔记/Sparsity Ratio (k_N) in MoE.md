## Sparsity Ratio (k/N) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparsity Ratio (k/N) 是 MoE 模型架构的关键参数，表示每个 token 激活的 expert 数 (k) 与总 expert 数 (N) 之比。它决定了 MoE 的稀疏激活程度：k/N 越小，稀疏性越高，每个 token 的计算量越低，但每个 expert 接收到的 token 越少。代表性值：Mixtral-8x7B k/N=2/8=0.25、Qwen2-57B k/N=8/64=0.125、DeepSeek-V3 k/N=8/256≈0.03。近期模型趋势向更低的 k/N 发展（保持 k 不变、增大 N），以获得更好的 accuracy-FLOPs trade-off。

LYNX 揭示了 sparsity ratio 的另一维度：它决定了 batch 级别 expert activation 的饱和速度。例如 Qwen2-57B (k/N=0.125)，每个 token 选 8 个 expert，batch 仅需 8 个 diverse requests 即可饱和全部 64 个 expert。对于更低的 k/N（如 DeepSeek-V3 的 0.03），batch 级 expert 饱和更慢，MoE 的稀疏性优势在更大的 batch size 下仍可保持。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Sparsity Ratio 对 decode 阶段 arithmetic intensity 的影响
arithmetic_intensity ∝ (batch_size × k) / N

# 例：Mixtral-8x7B, B=16, k=2, N=8
AI = 16×2/8 = 4  → moderate memory-bound

# 例：Qwen2-57B, B=16, k=8, N=64
AI = 16×8/64 = 2  → strongly memory-bound

# 例：DeepSeek-V3, B=16, k=8, N=256
AI = 16×8/256 = 0.5 → extremely memory-bound
```

LYNX 中 sparsity ratio 还决定了 AffinityBinning 的 α 和 β 参数——k/N 越小（稀疏性越高），bin 划分越细（更大的 β），因为更高的稀疏性意味着更少的 tokens 竞争每个 expert，需要更精细的 confidence 区分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sparsity ratio 是模型架构设计时的固定 choice，在模型训练前确定。当前 MoE 模型通过增大 N（而非减小 k）来降低 k/N——因为 k 太小会导致 training instability and expert collapse，而增大 N 通过 load-balancing loss 可以稳定训练。LYNX 利用 sparsity ratio 仅在模型加载时读取一次来确定 binning 参数，无需运行时调整。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
