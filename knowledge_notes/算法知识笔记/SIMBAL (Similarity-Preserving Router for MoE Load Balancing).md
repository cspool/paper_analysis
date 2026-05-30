## SIMBAL (Similarity-Preserving Router for MoE Load Balancing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SIMBAL（SIMilarity-preserving routers for MoE load BALancing）是一种新的 MoE 负载均衡方法，通过鼓励 Router 权重矩阵 R ∈ R^{D_M×E} 保持 token 间成对相似性来替代传统的 uniform-distribution-based 负载均衡损失。核心思想：如果 Router 矩阵 R 是正交的（R^T R = I），则 Router 的前向映射 x → xR 保留输入 token 间的点积（即角度/相似性）：(x1 R)·(x2 R) ≈ x1·x2。这意味着相似 token 会获得相似的 expert 分布，不同 token 会获得不同分布，从而自然地实现负载均衡（而非强制均匀分布）。

SIMBAL 的辅助损失：L_orth = ||R^T R - I_E||_1（L1 norm of Gram matrix deviation from identity）。该方法属于 loss-based soft constraint，而非显式正交参数化（如 QR 分解）。优势：(1) 不需要 float32→bfloat16 的精度转换，(2) 不需要昂贵的重正交化步骤，(3) 与标准 AdamW 训练兼容。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

SIMBAL 在 MoE Transformer 训练中的完整流程：

```python
# ===== 每 training step =====

# 1. Standard MoE forward pass (Token Choice, top-A routing)
x = attention_output                                # [B, S, D_M]
router_logits = x @ R                                # [B, S, E], R ∈ R^{D_M×E}
router_probs = softmax(router_logits, dim=-1)        # [B, S, E]
topk_vals, topk_ids = topk(router_probs, k=A)        # [B, S, A]

# 2. Expert computation (unchanged from standard MoE)
output = compute_moe_experts(x, topk_vals, topk_ids)  # [B, S, D_M]

# 3. SIMBAL auxiliary loss computation
# Only requires the router weight matrix R
w = R.weight                                          # [E, D_M] (或 [D_M, E])
gram = w @ w.T                                        # [E, E], Gram matrix
L_orth = ||gram - I_E||_1                             # L1 deviation from identity

# 4. Combined loss
L_total = L_lm + lambda_simbal * L_orth  # lambda_simbal typically 0.1 (insensitive)

# 5. Backward: L_orth gradient pulls R toward orthogonality
```

关键特性：
- L_orth 仅依赖 Router 权重，与数据分布无关 → 对 batch size 不敏感
- lambda_simbal 不敏感 (0.01/0.1/1.0 下 perplexity 差异 < 0.03)
- 配合正交初始化 (Saxe et al. 2014) 加速收敛；也可以仅执行少量 router-only SGD steps
- 比显式 QR 分解参数化更高效：后者需要 float32 计算 + 每次迭代重正交化

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- Router 维度特征：D_M >> E（如 D_M=1536, E=32）→ R 是 tall matrix → Gram matrix E×E 极小 → L_orth 计算开销可忽略
- 与 LBL 的对比：LBL 计算每个 expert 的 f_i·P_i 聚合统计量（需要 batch token routing info）→ SIMBAL 只需要 Router weight matrix
- 在 OLMo 开源代码库 (https://github.com/allenai/OLMo) 上实现，loss 通过 AddAuxiliaryLoss autograd trick 或直接相加集成
- 效果：比 LBL 快 36% 收敛（相同 loss 所需 token），expert 冗余（PES）降低 5-8x
- 与推理时 expert pruning 的协同：SIMBAL 产生 less uniform routing → 低 weight expert 更可安全丢弃 → 7.4% throughput improvement

涉及论文标题：
- Load Balancing Mixture of Experts with Similarity Preserving Routers
