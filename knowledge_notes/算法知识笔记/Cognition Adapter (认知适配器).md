## Cognition Adapter (认知适配器)

术语解释
Cognition Adapter 是 BrainMoE 中用于将多个脑认知专家（brain expert）的 cognition embeddings 适应到下游任务的 Transformer Decoder 模块，通过 multi-head self-attention 混合 expert embeddings 和 task queries，再通过 cross-attention 将原始 FC 矩阵信息注入，最终输出下游分类预测。

术语是什么？
Cognition Adapter 是一个专门设计的 Transformer Decoder 架构，输入为两部分拼接的 token vectors：
- Z̄_{:k} = Z ⊙ P：top-k 个 cognition embeddings（expert 输出 × Router 权重），维度 [k, C_hid]
- Z̄_{k:(k+P)}：随机初始化的 task query embeddings，维度 [P, C_hid]（P=下游分类数）

每层 adapter 执行两个 attention 操作：
1. Multi-head Self-Attention (MHSA)：Q=Z̄α_h, K=Z̄β_h, V=Z̄γ_h，在 expert embeddings 和 task queries 之间混合信息
2. Multi-head Cross-Attention：Q=Iα̂_h (FC matrix), K=Z̄β̂_h, V=Iγ̂_h，将原始脑连接组信息注入到 task representations
3. FFN（MLP）

最后通过 Linear(Z̄[k:]) 仅取 task query 部分输出分类 logits。

从算法pipeline角度拆解术语。
```
# Cognition Adapter Forward
def cognition_adapter(Z_experts, FC_matrix, P_classes, k_top):
    # Z_experts: [N, C_hid] - N个expert的cognition embeddings
    # FC_matrix: [M, M] - functional connectivity矩阵
    
    # Step 1: Router选择top-k experts
    P = Softmax(Linear_router(Z_experts))       # [N]
    topk_idx = TopK(P, k_top)
    Z_topk = Z_experts[topk_idx] * P[topk_idx]  # [k, C_hid]
    
    # Step 2: 拼接task query embeddings
    Q_task = Parameter(randn(P_classes, C_hid))  # [P, C_hid]
    Z_bar = concat([Z_topk, Q_task])             # [k+P, C_hid]
    
    # Step 3: Adapter layers
    for layer in range(num_layers):
        # Self-Attention: expert-task混合
        Z_bar = Z_bar + MHSA(Z_bar, Z_bar, Z_bar)  # Eq(2)
        
        # Cross-Attention: FC矩阵→task表示
        Q_cross = FC_matrix @ alpha_hat           # [M, C_hid]
        K_cross = Z_bar @ beta_hat                # [k+P, C_hid]
        V_cross = FC_matrix @ gamma_hat           # [M, C_hid]
        Z_bar = Z_bar + CrossAttn(Q_cross, K_cross, V_cross)  # Eq(3)
        
        # FFN
        Z_bar = Z_bar + MLP(Z_bar)
    
    # Step 4: 输出分类
    y_pred = Linear(Z_bar[k_top:])               # [P] logits
    return y_pred
```

术语一般如何实现？如何使用？
- 架构选择：BrainMoE 选择 Transformer Decoder 而非简单 MLP，因为 MLP adapter 在高维 latent space（C_hid=2048）下不可扩展
- Cross-attention 设计：将 FC matrix [M,M] 作为 Q 和 V 的 source，使 adapter 能直接访问原始脑连接组信息，避免信息瓶颈
- Router+Adapter 联合训练：expert 参数冻结，仅训练 Router 和 Adapter 参数
- 输入灵活性：不依赖 expert 的内部架构，支持 FC-based 和 BOLD-based expert 混合使用（All-in-one BrainMoE 36 experts）

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---
