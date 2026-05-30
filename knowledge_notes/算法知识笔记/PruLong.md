## PruLong

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PruLong 是 Princeton PLI 提出的端到端 KV cache eviction 方法。核心：将 attention heads 二分类为 retrieval（full KV）和 streaming（local W=1024 + sinks S=128）。vs DuoAttention 的三项创新：(1) next-token prediction loss 替代 L2 reconstruction；(2) Hard Concrete Bernoulli masks 消除 train-test gap；(3) natural long-context data 替代 synthetic passkey。冻结模型权重，仅训练 mask parameters，1000 steps 收敛。Recall critical KV footprint 46% vs DuoAttention 58%（-12 points）。

从算法pipeline角度拆解术语。

**PruLong 训练伪代码**：
```
# 参数：log_α_{i,j} (每层每头), λ1, λ2 (Lagrange), τ=2/3, l=-0.1, r=1.0
for step in 1..1000:
    t = min(target, target × step/800)  # sparsity warmup
    for each head (i,j):
        z̃ = HardConcrete(log_α, τ, l, r)  # Bernoulli 采样
        attn = z̃ × Attn_full + (1-z̃) × Attn_streaming
    L_ntp = cross_entropy(logits, labels)
    s = 1 - mean(σ(log_α + log(10)))  # expected sparsity
    L = L_ntp + λ1(s-t) + λ2(s-t)²
    log_α -= ∇L; λ1 += ∇L; λ2 += ∇L

# 离散化：top k% log_α → z=1 (retrieval), 其余 z=0 (streaming)
```

**PruLong vs DuoAttention@70% sparsity**：
| 维度 | DuoAttention | PruLong |
|------|-------------|---------|
| 训练目标 | L2 reconstruction | NTP loss |
| Mask 类型 | Continuous z∈[0,1] | Hard Concrete Bernoulli |
| 训练数据 | Synthetic passkey | Natural long data |
| Recall score | 38.6 | 91.4 |

术语一般如何实现？如何使用？

PyTorch + HuggingFace Transformers，在 Llama-3.1-8B-Instruct 和 ProLong-8B 上评估。训练配置：batch 1,048,576 tokens, seq_len 131,072, model weights frozen。代码：https://github.com/princeton-pli/PruLong

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---
