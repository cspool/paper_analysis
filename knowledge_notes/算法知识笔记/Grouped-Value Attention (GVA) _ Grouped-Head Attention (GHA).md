## Grouped-Value Attention (GVA) / Grouped-Head Attention (GHA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

GVA 和 GHA 是 GTA 论文在方法论演进中引入的两种中间注意力变体，作为从 MHA 到 GTA 的"进化步骤"（evolving patterns of attention mechanisms）。

**Grouped-Value Attention (GVA)**：将 attention weights（从 Q 和 K 计算得出）在 heads 组内共享。同一 group 内的多个 heads 使用相同的 attention distribution（softmax(QK^T)），但各自应用不同的 value 投影。这复用了 attention weight 计算（减少 QK^T 次数），但保留了每个 head 独立的 value transformation。KV cache 仍为 (H + n_k d_h)N，内存节省来自 K 的减少而非 V。

**Grouped-Head Attention (GHA)**：进一步压缩——在 heads 组内共享 Q 和 K 表示（同一 group 内使用相同的 query、key），values 从共享源分别计算。这显著降低了 KV cache 至 (n_k d_h + n_v d_h)N。代价是 reduced diversity in Q/K representations（共享导致 query/key 的多样性下降），限制了模型在复杂任务上捕获细粒度上下文依赖的能力。

GVA 和 GHA 共同阐释了 attention 机制中 **efficiency-expressivity trade-off**：GVA 在较少牺牲表达力的情况下减少计算冗余；GHA 以表达力损失换取更大的内存/计算节省。GTA 通过在 GHA 基础上引入 nonlinear value decoder（从 latent 动态生成 head-specific values），在保持 GHA 级内存效率的同时恢复表达力。

从算法pipeline角度拆解，给出具体例子。

**GVA 计算流程**：
```
# 假设 n_h=12, n_q=3, n_k=3, n_v=12 (MHA 的 V group 数=12)
for g in 0..2:
    Q_g = head_group_queries[g]       # 4 heads 共享
    K_g = K[:, g*64:(g+1)*64]
    A_g = softmax(Q_g @ K_g^T / 8)    # (N,N) → 4 heads 共享
    
    for each head i in group g:
        V_i = head_values[i]           # 独立 V
        O_i = A_g @ V_i                # 共享 attention × 独立 value
```

**GHA 计算流程**：
```
# 假设 n_h=12, n_q=3, n_k=3, n_v=3
for g in 0..2:
    Q_g = Q[:, g*64:(g+1)*64]          # Q 按 group 共享
    K_g = K[:, g*64:(g+1)*64]          # K 按 group 共享
    C_g = V_shared[:, g*64:(g+1)*64]   # 共享 value source
    
    A_g = softmax(Q_g @ K_g^T / 8)     # (N,N)
    
    for each head i in group g:
        V_i = C_g @ W_{i}               # 从共享源派生 head-specific V
        O_i = A_g @ V_i
```

**GTA 在 GHA 上的关键改进（对应 Eq 6）：**
```
# GHA: V_i = C_g @ W_i                  # 线性投影，仅依赖 C_g
# GTA: V_i = C_g @ W_{P,i} ⊙ σ(x_t W_{G,i})  # 非线性 + context-adaptive
# σ 为 Sigmoid，x_t 为当前 token
```

术语一般如何实现？如何使用？

GVA 和 GHA 在论文中主要作为**分析性"跳板"**出现——它们阐明了 attention 效率设计空间中的关键权衡维度（attention sharing vs value sharing vs KV compression），为 GTA 的设计提供了动机。论文未将这些变体作为独立方法发表或开源；实际使用中，工程师可根据具体设备的 memory/compute 瓶颈选择 GVA（memory-rich 场景，V 可保持独立）、GHA（memory-constrained 场景，但 attention 表达力有损）或 GTA（平衡方案，引入 latent + nonlinear decoder）。

涉及论文标题：
- GTA__Grouped-head_latenT_Attention

---
