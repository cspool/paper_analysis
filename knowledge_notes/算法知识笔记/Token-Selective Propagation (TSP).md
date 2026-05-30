## Token-Selective Propagation (TSP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token-Selective Propagation (TSP) 是 FastKV 论文提出的两阶段预填充策略的核心机制。TSP 在 transformer decoder 的中间某层（TSP Layer，如 LLaMA-3.1-8B 的 layer 15），基于最近 window tokens（默认 8 个）的注意力权重计算每个输入 token 的 saliency score，仅将得分最高的 top-R_TSP（默认 20%）token 的 hidden states 传播到后续层。TSP 之前的层保持完整上下文计算，之后的层仅在压缩后的 hidden states 子集上计算注意力。TSP 的动机来自 Layer-dependent Context Dynamics 的观察：早期层的注意力焦点高度不稳定（不同层的 critical token 集合差异大），若过早剪枝会不可逆地丢弃后续层需要的 token；后期层的注意力则趋于稳定（可以安全剪枝）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**TSP 在 FastKV 两阶段预填充中的伪代码：**

```
# 阶段一：完整上下文预填充（Layer 0 到 L_TSP）
for l = 0 to L_TSP:
    X, Att_l, K_X, V_X = layer_l(X)
    K, V = KV_Compress(K_X, V_X, Att_l, R_KV)       # 独立于 TSP 的 KV 压缩
    if l == L_TSP:
        # TSP: 计算 saliency score
        for i in 0..N_I-1:                            # N_I = 输入 token 数
            for h in 0..H-1:                           # H = head 数
                # Eq(1): window tokens 作为 query 的注意力聚合
                S_i^{l,h} = MaxPooling( Σ_{n=0}^{N_obs} Att_l[h, N_I-1-n, i+m] )
            # Eq(2): 跨 head 平均得到 layer-level saliency
            S_i^{TSP_layer} = (1/H) * Σ_h S_i^{TSP_layer,h}
        # 选取 top-R_TSP 关键 token + 所有 window tokens
        I_TSP = TopK(S^{TSP_layer}, N_I * R_TSP)
        I_TSP = I_TSP ∪ {N_I - N_obs, ..., N_I - 1}    # window tokens 强制保留
        x = X[I_TSP]                                   # 仅传播选中的 hidden states

# 阶段二：压缩上下文预填充（TSP Layer+1 到 Last Layer）
for l = L_TSP+1 to L-1:
    x, Att_l, K_x, V_x = layer_l(x)                   # 仅在 x（压缩后）上计算
    K, V = KV_Compress(K_x, V_x, Att_l, R_KV)
```

**TSP 与 GemFilter 的关键差异：** GemFilter 在 filter layer 选择 token 后，从 layer 0 重新开始预填充——早期层被迫使用同一 token 子集。TSP 保留早期层完整上下文，仅在后层（注意力稳定后）剪枝。被 TSP 丢弃的 token 已在早期层的注意力中将语义融合到保留 token 中（Figure 7）。

**TSP Layer 自动选择（Eq 3）：**
```
L_TSP = argmin_{L ≤ L_max} (1/N) Σ_{i=1}^{N} ||H_i - H'_{L,i}||₂²
```
其中 H_i 为完整上下文下最终层 hidden state，H'_{L,i} 为在 L 层应用 TSP 后的 hidden state。通过少量标定数据最小化输出偏差选择最优 TSP 层。

术语一般如何实现？如何使用？

实现集成在 HuggingFace Transformers 的 self-attention 层中，与 FlashAttention-2 兼容。关键实现要点：(1) TSP 的 saliency scoring 仅基于 N_obs=8 个 window token 的注意力行，计算开销极小（128K上下文仅 0.15s，占总预填充 0.88%）；(2) MaxPooling kernel_size=7 用于平滑时间维度的注意力分数；(3) TSP rate 与 KV retention rate 完全解耦独立配置——TSP rate 控制预填充计算量（等于 1 - Σ_{l>TSP}(1-R_TSP)，LLaMA-3.1-8B 下约 60%），KV retention rate 独立控制解码时每层保留的 KV cache 比例（10% 或 20%）；(4) window tokens 强制保留机制确保最新上下文永远可用。

涉及论文标题：
- FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

---
