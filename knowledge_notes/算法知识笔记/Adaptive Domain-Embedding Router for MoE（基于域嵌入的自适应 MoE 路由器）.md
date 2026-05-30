## Adaptive Domain-Embedding Router for MoE（基于域嵌入的自适应 MoE 路由器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Domain-Embedding Router 是 Nexus (Gritsch et al., 2024) 提出的 MoE 路由器设计。与传统线性路由器 W_r @ x 不同，Nexus router 由两部分组成：(1) 投影层 P_r（2-layer SwiGLU MLP）：将预计算的域嵌入 d_i ∈ R^m 投影为 expert embedding e_i ∈ R^h；(2) 相似度路由：通过 token hidden state x 与各 e_i 的点积计算路由概率 s_i = softmax(x · e_i)。P_r 是 hypernetwork——它以域嵌入 d_i 为条件生成路由器参数（expert embeddings），而非直接存储固定参数。域嵌入 d_i 通过外部 embedding model（如 Cohere Embed v3）对该域训练数据编码后取平均得到，也可通过无监督聚类的 centroid 获得。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Nexus Adaptive Domain-Embedding Router（每个 Transformer block 一个）
# 输入: pre-computed domain_embeddings [m, n_experts], hidden states x [s, h]
# 输出: routed expert indices, gate weights

# Step 1: 投影层将域嵌入映射为专家嵌入
# P_r = 2-layer MLP with SwiGLU: W1 in R^{2h×m}, W2 in R^{h×h}
for i in range(n_experts):
    e_i = P_r(d_i)                          # d_i: [m] → e_i: [h]
    # = W2 @ SwiGLU(W1 @ d_i)
    # SwiGLU(x) = SiLU(W_gate @ x) ⊙ (W_up @ x)

# Step 2: 计算 token 与各专家嵌入的相似度
router_logits = x @ E                        # [s, n_experts], E = [e_1,...,e_n]

# Step 3: Top-K 路由
router_probs = softmax(router_logits)
topk_gates, topk_indices = topk(router_probs, k=1)

# Step 4: MoE 输出
output = shared_expert_ffn(x)               # seed model FFN, always active
output += topk_gates * routed_expert_ffns[topk_indices](x)
```
**添加新 expert（Extension）**：新域 d_new → e_new = P_r(d_new) → 直接 append FFN_new 到 expert 数组。P_r 不变，无需重新训练 router 维度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：投影层 P_r 在每个 Transformer block 中独立训练一个 2-layer SwiGLU MLP。域嵌入在训练前预计算并存储，推理时可预计算 expert embedding 缓存（与输入 x 无关）。
- **与超网络的关系**：P_r 是 hypernetwork——它以域嵌入 d_i 为条件生成路由器参数（e_i），而非直接存储固定路由参数。
- **优势**：(a) Router 参数与 expert 数量解耦——新增 expert 不需要扩大 router；(b) 域语义归纳偏置——即使负载均衡损失很低，域嵌入的内在语义也能维持稳定的路由（Nexus ablation：α=0.0005 时性能不变，线性 router 下降 2%）；(c) 保留域间关系——P_r 投影后 Books-C4、GitHub-SE 的高 cosine similarity 关系被保留。
- **局限**：需要外部 embedding model 预计算域嵌入（论文使用 Cohere Embed v3）；当前仅验证了基于预定义数据源的域划分。
- **代码**：论文未开源，但提供了完整的 PyTorch 伪代码（Figure 2）。

涉及论文标题：
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

---
