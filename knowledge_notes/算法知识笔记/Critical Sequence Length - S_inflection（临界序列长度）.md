## Critical Sequence Length - S_inflection（临界序列长度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

$S_{\text{inflection}}$ 是 MagicDec 提出的临界序列长度概念：对于给定的模型-硬件组合，当输入 context length $S \ge S_{\text{inflection}}$ 时，speculative decoding（SD）在高 batch size 下开始有效（speedup > 1），且 speedup 随 batch size 增大而提升。当 $S < S_{\text{inflection}}$ 时，SD 在大 batch 下失效（speedup < 1），因为推理过程 compute-bound 导致 $T_V/T_T$ 过高。

$S_{\text{inflection}}$ 由两个因素决定：
1. **模型 FLOPS-to-memory ratio**：GQA 模型（如 LLaMA-3.1-8B）有更高的 FLOPS-to-memory 比（因为 KV head 更少），需更长序列才能达到 memory-bound 状态，因此 $S_{\text{inflection}}$ 更高。非 GQA 模型（如 LLaMA-2-7B）的 $S_{\text{inflection}}$ 更低。
2. **GPU FLOPS-to-bandwidth ratio**：H100（高 FLOPS/带宽比）的 $S_{\text{inflection}}$ 低于 A100 和 L40，意味着在 H100 上 SD 更早开始有效。

MagicDec 实验测定 LLaMA-3.1-8B 在 8×A100 上的 $S_{\text{inflection}} \approx 4000$ tokens（Figure 2c）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

S_inflection 的推导基于 roofline 模型。当 batch size 增大时：

- $S < S_{\text{inflection}}$：推理 compute-bound，线性层（MLP/Attention 投影）的计算成为瓶颈。验证 cost $T_V/T_T$ 随 batch 增大显著上升（Figure 2b），因为验证需对所有候选 token 做完整计算。SD 失效，speedup 随 batch 增大而下降。
- $S \ge S_{\text{inflection}}$：推理 memory-bound，KV cache loading 成为主导瓶颈。$T_V/T_T \approx 1$（验证与解码共享 KV）。同时 draft 使用压缩 KV（budget K << S）→ $T_D/T_T \to 0$ → speedup = $\Omega(\gamma,\alpha) > 1$。

```
# S_inflection 的判断逻辑
if S < S_inflection:
    # compute-bound: T_V/T_T 随 B 增大显著上升
    # SD speedup 随 B 增大下降 → 大 batch 应禁用 SD
else:
    # memory-bound: KV bottleneck, T_V/T_T ≈ 1
    # 压缩 KV draft 使 T_D/T_T → 0
    # SD speedup 随 B 增大反而提升 → 大 batch 使用 SD 有利
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

使用方式：在部署长上下文 LLM serving 时，根据模型和 GPU 类型估计 $S_{\text{inflection}}$，决定是否对当前请求启用 SD。对于 LLaMA-3.1-8B + A100，S > 4000 时启用 SD 且不限制 batch size；S < 4000 时仅在小 batch（< 32）启用 SD。H100 的 S_inflection 更低（~2000-3000），L40 更高。MagicDec 框架自动根据 profile 数据确定 S_inflection 并选择最优 drafting 策略。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
