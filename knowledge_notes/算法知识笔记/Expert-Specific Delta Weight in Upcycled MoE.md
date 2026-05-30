## Expert-Specific Delta Weight in Upcycled MoE

术语解释
在 upcycled MoE 中，专家权重 W_i = W_base + Δ_i。Δ_i 是训练中学到的相对于共享初始权重的微小偏移。DeRS 论文发现 Δ_i 高度冗余（余弦相似度 > 0.999），可通过稀疏化（drop 90% 元素）或量化（至 2-bit）几乎无损失地压缩。

术语是什么？
- Δ_i = W_i - W_base ∈ R^{d×d_h}，表示专家 i 相对共享知识的偏移
- 冗余程度取决于 dense model 是否经过先验微调：微调过的 delta 冗余度高（可 99% drop / 1-bit），未微调的冗余度低（需保守压缩）
- 独立压缩 Δ_i 而非 W_i 可保护 W_base 中的预训练知识

从算法pipeline角度拆解术语。
```
def analyze_and_compress_deltas(experts, W_base):
    for W_i in experts:
        cos_sim = cosine_similarity(flatten(W_i), flatten(W_base))
        # > 0.999, delta幅值远小于base
        Δ_i = W_i - W_base
    
    for each Δ_i:
        M ~ Bernoulli(p); compact = (1-M)⊙Δ_i/(1-p)  # 稀疏化
        # 或 compact = Quant(Δ_i, k_bits)               # 量化
    return W_base, compact_deltas

def synthesize(W_base, compact_delta_i):
    return W_base + reconstruct(compact_delta_i)
```

术语一般如何实现？如何使用？
- 观察方法：计算 flatten(W_i) 与 flatten(W_base) 的余弦相似度
- 与 LoRA 的关系：LoRA 的 ΔW = A·B 可视为 delta weight 的低秩形式，但应用于不同场景

涉及论文标题：
- DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models
