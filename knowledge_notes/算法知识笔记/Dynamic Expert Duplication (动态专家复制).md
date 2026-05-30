## Dynamic Expert Duplication (动态专家复制)

术语是什么？
Dynamic Expert Duplication 是一种 MoE 推理负载均衡技术。在多 GPU Expert Parallelism 设置中，当 token-to-expert 分布倾斜（skewed）时，部分 GPU 上的热门 expert 处理的 token 远超平均，成为 compute 和 communication 的 bottleneck。Expert Duplication 将热门 expert 的权重复制到 underloaded GPU 上，使多个 GPU 共同处理同一 expert 的 token，从而将集中的负载分散到多 GPU。关键在于"动态"——token 分布随输入 batch 变化，需要 predictor 预先预测每层的 expert 激活分布，然后据此决定哪些 expert 需要被复制到哪些 GPU。

从算法pipeline角度拆解术语：
MoE-GPS 的 Expert Duplication 在每层 Transformer Block 的 Attention 之前执行（Algorithm 1）：
```
输入: token_expert_map f (T tokens → E experts), GPU_memory M, 
     初始 placement P, 最大复制数 C_max
输出: 均衡 placement P, token→GPU dispatch d

1. d(t) = min{g | (f(t), g) ∈ P}     // 将 token 分配到持有其 expert 的 GPU
2. L[g] = |{t | d(t) = g}|            // 每 GPU 负载
3. while max(L) - min(L) > 1:         // 不均衡时迭代
4.   g_h = argmax(L); g_c = argmin(L)
5.   Δ = ceil((L[g_h] - L[g_c]) / 2)
6.   e* = most_popular_expert_on(g_h)  // overloaded GPU 上 token 最多的 expert
7.   if (e*, g_c) ∉ P and copies(e*) < C_max and params(e*) ≤ M[g_c]:
8.     copy_weights(e* → g_c)          // NVLink/PCIe 传输 ~47MB (Mixtral expert FP16)
9.     P = P ∪ {(e*, g_c)}
10.    reassign first Δ tokens of e* from g_h to g_c
11. update L[g_h], L[g_c]
return P, d
```
核心贪心策略：每次迭代取负载最大 GPU 上 token 最多的 expert，将其复制到负载最小的 GPU，并转移一半差值 token 到新 GPU。当所有 GPU token 数差异 ≤1 时停止。Expert 复制通信（~0.1ms per expert over NVLink 3.0）可与 Attention 计算重叠。

术语一般如何实现？如何使用？
已有工作：MoE-Prediction [Cong et al. 2024] 首次提出预测 token-to-expert 分布指导 expert placement；Prophet [Wang et al. 2023] 提出细粒度动态 expert duplication 策略；FlexMoE [Nie et al. 2023] 和 FasterMoE [He et al. 2022] 也使用 expert duplication。实现要点：(1) predictor 频率——从每 batch [He et al., Prophet] 到每 10 分钟 [DeepSeek-V3]，权衡 overhead 和有效性；(2) placement 开销——expert 权重传输可通过与 Attention 计算 overlap 隐藏（batch size≥16, seq_len≥2K 时 PCIe 也可隐藏）；(3) 内存容量——每 GPU 的 expert 复制数受显存限制（C_max 参数）。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

---
