## Sublinear Token Budget Scaling for Sparse Attention (稀疏注意力Token预算的次线性缩放)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sublinear Token Budget Scaling 是 Sparse Frontier 论文发现的一个关键规律：随着序列长度 L 增长，维持相同 accuracy degradation 所需的 token budget 以**次线性（sublinear）**速率增长，而非线性（linear）或固定（fixed）。具体来说，对于目标相对误差 ≈0.2，所需的 attention budget fraction 从 16K 时的 1/10 降至 32K 时的 1/15 再到 64K 时的 1/20——即翻倍序列长度不需要翻倍 token budget。

理论基础：Herdan's Law（Herdan, 1960）——自然语言中，序列越长，新信息的出现频率越低，允许更高的稀疏度。从信息论角度，更长上下文的 token 级信息密度递减（diminishing marginal information density），使得注意力模式更集中在少数关键 token 上。

实践意义：(1) 当前 production 中常用的固定 budget 方法（如固定 token_budget=4096）是次优的——应该在长度增长时增加预算但不必翻倍；(2) 固定 budget fraction（如总是 10% attention）则是过于保守——实际可以随着长度增长逐步提高 sparsity；(3) 最优预算函数应遵循 budget(L) ∝ L^k, k<1（如 k≈0.7-0.8）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Sublinear Budget Model (基于 Sparse Frontier 的 iso-error 曲线)
# 目标：给定目标 relative error ε，求预算函数 B(L)
# 
# 从 Figure 3 提取的近似 iso-error 点：
#   ε=0.2:  B(16K)≈0.1L, B(32K)≈0.067L, B(64K)≈0.05L
#   ε=0.1:  B(16K)≈0.2L, B(32K)≈0.2L, B(64K)≈0.17L

# Sublinear model fitting:
#   B(L) = c * L^k, where k < 1
#   For ε=0.2: k ≈ log(0.05/0.1) / log(64/32) ≈ log(0.5)/log(2) ≈ -1 → 
#              B ∝ L^0 (constant fraction decreasing!)
# Wait, the fraction itself decreases. Let me clarify:
#   budget_fraction(16K)=1/10, budget_fraction(64K)=1/20
#   token_count(16K)=1600, token_count(64K)=3200
#   So doubling L from 16K→32K→64K only needs 1.33×→1.33× more tokens

# Practice: dynamic budget schedule for serving
def get_token_budget(seq_len, base_budget_16k=1600):
    """次线性 token budget 调度器"""
    # 确保 budget 至少不减少
    return max(base_budget_16k, base_budget_16k * (seq_len/16384)^0.5)
```

术语一般如何实现？如何使用？

实际部署时:(1) 根据任务的 accuracy 要求确定 ε 容忍度；(2) 通过离线 profiling 建立 L→B(L) 查找表（对每种稀疏注意力方法和模型）；(3) serving 时根据实际 prompt length 查表和设置 token budget。Sparse Frontier 建议未来方向：开发可靠的动态 budget 分配机制（目前 dynamic 方法如 FlexPrefill 缺乏鲁棒性），使 sublinear scaling 能自动实现。

涉及论文标题：
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
