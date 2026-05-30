## MoE Router (Top-K Gating Network)

术语是什么？
MoE Router（也称Gating Network/门控网络）是稀疏MoE模型中的核心路由组件，决定每个输入token被分配到哪些expert进行计算。Router通常是一个小型线性层（W_g ∈ R^{d_model × n_expert}），对每个token的hidden state计算logits，经Softmax后得到每个expert的概率分布，然后选择概率最高的top-K个expert（K通常为1-4）。MoE-CAP分析了多种路由变体：(1) 纯Top-K路由（如Mixtral-8x22B, K=2）；(2) Top-K+Shared Experts路由（如Qwen1.5-MoE, K=4+4 shared，DeepSeek-R1, K=8+1 shared）；(3) 高expert数+低K路由（如Switch-C, 128 experts, K=1）。Router的设计直接影响模型稀疏性（activated/total参数比）和负载均衡。

从算法pipeline角度拆解术语：
MoE Router在单个MoE layer中的计算流程：
```
# 输入: hidden_state h ∈ R^{d_model}（单个token）
# W_g ∈ R^{d_model × n_expert}: router权重
# 输出: selected_experts 和对应的gate weights

logits = h @ W_g                          # [n_expert] 原始logits
probs = Softmax(logits)                    # [n_expert] 概率分布
topk_vals, topk_indices = TopK(probs, K)   # 选top-K expert索引和概率

if has_shared_experts:                     # 若有shared experts
    # shared experts始终激活（如Qwen1.5-MoE: 4 shared experts）
    selected_experts = topk_indices ∪ shared_expert_indices

# 归一化gate weights（仅对选中的expert）
gate_weights = Softmax(topk_vals)          # re-normalize

for each selected expert i:
    expert_output_i = gate_weights[i] * FFN_i(h)

output = Σ expert_output_i                 # 加权求和
```
MoE-CAP通过追踪Router输出的topk_indices来记录每层𝟙[l,i]（expert i是否被激活），从而计算S_activated。S-MFU中的k_expert参数也直接来自模型配置的K值和shared expert数。

术语一般如何实现？
HuggingFace Transformers中MoE Router通常实现为`nn.Linear(d_model, n_expert)`后接Top-K选择。主流serving框架（vLLM, SGLang）在MoE layer实现中集成fused routing kernel以降低router开销。MoE-CAP在router后植入probe记录激活模式用于S-MBU计算。Router的负载均衡是主要挑战——部分expert可能被过度使用（load imbalance），导致GPU闲置。

**MoE-Pruner 的补充**：MoE-Pruner (Xie et al., 2024) 首次将 Router 权重显式纳入 MoE expert 层的剪枝度量。其核心洞察是：Router 输出的 Gate 权重向量（经 Softmax 归一化后）反映了"该 expert 对当前 token 的重要性"——若某 expert 对一批 token 的 Gate 权重极低，其权值即使 magnitude 大也应优先剪除。MoE-Pruner 的剪枝度量 S = |W_ij| * ||X_j * Gate_j|| 在 Wanda（S = |W_ij| * ||X_j||）基础上多乘了一个按元素广播的 Gate 权重项，使被 Router 抑制的 expert（Gate ≈ 0）的权值重要性被降低，被 Router 强激活的 expert（Gate ≈ 1）的权值重要性被保留。这是首次将 MoE routing 信息从"选择哪些 expert"扩展到"在 expert 内部哪些权值更重要"的粒度。

**Nexus 的 Adaptive Domain-Embedding Router 变体**：Nexus 提出了一种从根本上不同于线性 router 的路由机制——不学习 W_r ∈ R^{h×n}，而是学习投影函数 P_r 将域嵌入映射为 expert embedding：
```
# Nexus Router（每个 MoE block）
domain_embeddings: [m, n_experts]  # 预计算的域嵌入（如 Cohere Embed v3 编码）
expert_embeddings = P_r(domain_embeddings)  # [h, n_experts], P_r = 2-layer SwiGLU MLP
router_probs = softmax(inputs @ expert_embeddings)  # [batch, seq, n_experts]
index, gate = topk(router_probs, k=1)
```
与线性 router 的关键区别：(1) Router 参数不随 expert 数量增长——P_r 是固定的投影函数，输入为域嵌入 d_i 而非 token x；(2) 路由基于 token 与各域 expert embedding 的点积相似度，具有域语义归纳偏置；(3) 新 expert 加入时仅需计算 d_new → P_r(d_new) = e_new，无需修改 router 参数维度。Nexus 的实验显示此 router 对 load balancing loss 因子不敏感（低至 0.0005 时仍稳定），而线性 router 下降约 2%。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
