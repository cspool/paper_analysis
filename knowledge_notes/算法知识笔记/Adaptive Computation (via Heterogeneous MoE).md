## Adaptive Computation (via Heterogeneous MoE)

术语解释
Adaptive Computation（自适应计算）指模型根据不同输入的复杂度动态分配不同量的计算资源：简单输入使用较少计算，复杂输入使用较多计算。在异构 MoE 中，自适应计算通过不同大小的 expert 自然实现——routing decisions 将 token 发送到不同大小的 expert，简单 token 走小 expert，复杂 token 走大 expert。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
与 traditional MoE 的 conditional computation（条件计算）的区别：
- **Conditional computation**：仅激活 expert 子集（稀疏激活），但所有被激活的 expert 大小相同 → 每个被处理的 token 获得相同计算量
- **Adaptive computation**：不同 token 不仅激活不同的 expert 子集，还被路由到不同大小的 expert → 不同 token 获得不同计算量

AutoMoE 通过异构 expert 设计（variable expert FFN size）实现 adaptive compute：
- 搜索空间允许 FFN intermediate size ∈ {1024, 2048, 3072}
- 同一层中可有不同大小的 expert（如 3 个 expert: sizes [3072, 2048, 1024]）
- Router 的 top-1 路由自然将不同 token 分配到不同大小的 expert
- 该设计等价于 early-exit 风格的 adaptive compute 但通过路由而非层级别实现

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Adaptive Compute via Heterogeneous MoE
# Layer with 4 heterogeneous experts: sizes [3072, 2048, 2048, 1024]

def adaptive_moe_forward(x):
    # x: [batch, d_model], d_model = 512
    
    # Router
    logits = x @ W_router                # [batch, 4]
    expert_idx = argmax(logits, dim=-1)   # [batch], top-1 routing
    
    output = zeros_like(x)
    flops_per_token = []
    
    for token_i, e_idx in enumerate(expert_idx):
        e_ffn_size = expert_sizes[e_idx]  # 3072, 2048, 2048, or 1024
        
        # FFN: W1 ∈ R^{ffn_size × 512}, W2 ∈ R^{512 × ffn_size}
        h = ReLU(x[token_i] @ W1[e_idx].T)  # [ffn_size] — FLOPs 正比于 ffn_size
        out = h @ W2[e_idx].T               # [512]        — FLOPs 正比于 ffn_size
        output[token_i] = out
        
        flops = 2 * 512 * e_ffn_size  # 2 × d × ffn_size
        flops_per_token.append(flops)
    
    # 示例：简单 token "the" → expert 3 (ffn_size=1024) → FLOPs = 2×512×1024 = 1.05M
    #       复杂 token "photosynthesis" → expert 0 (ffn_size=3072) → FLOPs = 3.15M
    # 计算量差异: 3× reduction for simple tokens
    return output, flops_per_token
```

术语一般如何实现？如何使用？
- 异构 expert 尺寸在搜索阶段确定，推理时 router 动态分配
- 也可通过 identity/dummy expert（FFN size=0）实现极端自适应——某些 token 完全跳过 FFN 计算
- AutoMoE 的关键实证发现：70% expert layers 有 ≥2 experts，>75% 含可变 expert 尺寸
- 另一种 adaptive computation 实现：Ada-K（RL-based dynamic K）和 AdaMOE（null experts）
- 与 Mixture-of-Depths (MoD) 互补：MoD 在深度维度自适应（跳过层），异构 MoE 在宽度维度自适应（不同大小 expert）

**Duo-LLM 的自适应计算视角**：Duo-LLM 在每层 FFN 中并排放置一个 big FFN（inner_dim=10240）和一个 small FFN（inner_dim=640，16x smaller），两者共享 attention。训练时以 random routing（p=0.5）确保两个模块互换。推理时的自适应计算通过 router 在 per-token per-layer 粒度决定走 big 还是 small（甚至 skip），实现 token 级别的计算弹性。关键发现：Oracle 最优路由下仅使用 1 个 big layer 的 perplexity 低于所有 12 层都用 big module，证明了精细粒度自适应计算的巨大潜力。

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---
