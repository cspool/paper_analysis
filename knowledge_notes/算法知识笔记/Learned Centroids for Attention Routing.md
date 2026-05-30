## Learned Centroids for Attention Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learned Centroids for Attention Routing 是 Focus 论文的核心机制：在预训练 Transformer 的每个 attention 层中添加少量可学习的 centroid 向量 C ∈ R^{K×d_g}（K 个 group，d_g 维 centroid 空间）和一个轻量投影矩阵 W_g ∈ R^{d×d_g}，用于将 token 分配到语义 group，进而控制哪些 token pair 可以互相关注（routing）。关键设计原则是 separation of routing and attention：centroid 仅决定"谁可以关注谁"（routing 决策），QKV 注意力决定"关注多少"（content 传输）。预训练 QKV 权重完全冻结，centroid 参数低至 148K（d_g=16 时仅占模型 0.1%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Centroid 在 Focus 推理 pipeline 中的角色：

```
# 每个 attention 层
def focus_attention_layer(h, Q_weight, K_weight, V_weight, O_weight, C, W_g):
    # === Routing Phase (centroid-only, 可训练) ===
    # d_g=16 维投影，仅需 148K 参数
    token_repr = h @ W_g.T              # [T, d_g]
    group_scores = token_repr @ C.T     # [T, K]  token-centroid 亲和度
    g = sinkhorn(group_scores / τ)      # [T, K]  双随机 group assignment
    
    # === Content Phase (预训练权重, 冻结) ===
    q = h @ Q_weight.T                  # [T, d_head]
    k = h @ K_weight.T                  # [T, d_head]
    v = h @ V_weight.T                  # [T, d_head]
    
    # === Gated Attention ===
    for i in range(T):
        for j in range(i+1):
            if i - j <= w:              # 局部窗口 → 全注意力
                s_ij = q_i @ k_j.T
            else:                       # 远距离 → 组门控
                gate = sigmoid(λ * g[i] @ g[j].T)  # 同组≈1, 异组≈0
                s_ij = q_i @ k_j.T * gate
    
    attn_out = softmax(s) @ v          # 仅有效 pair 参与 softmax
    return attn_out @ O_weight.T
```

关键设计决策：
- **d_g=16**：token grouping 是低维任务，16 维足矣。d_g=768（全维）vs d_g=16，参数差 50 倍（7.1M vs 148K），PPL 无差异（均为 34.5）
- **K=4 或 8**：group 数量。K=4 时每个 token 属于 2 个 group（top-k=2），约 50% 远距离 pair 被剪枝，2× 加速；K=8 时 8.6× 加速（1M token）
- **λ**：gate steepness 参数，控制 sigmoid 的锐度

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Focus centroid 的使用方式：
- **Composable training**（仅 centroid 训练）：加载预训练模型 → 每层插入 C 和 W_g → 冻结所有原权重 → 仅训练 centroid 参数（4000 steps on PG-19 with GPT-2 124M）
- **Full fine-tuning**（两阶段）：Phase 1 仅 centroid 训练建立 group 结构 → Phase 2 解冻所有权重联合微调
- **Inference**：hard top-k assignment（每个 token 选 top-k 个 group），仅同组 token pair 计算注意力
- 从零训练：centroid + 标准 QKV 权重一起随机初始化、一起训练（Mistral 7B from scratch on 2B tokens）
- 跨架构通用：MHA、GQA、GQA+bias、MHA+QK-norm、interleaved+softcap 五种 attention 架构均适用

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)
