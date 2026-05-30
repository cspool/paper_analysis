## BTX (Branch-Train-Mix)

术语是什么？
BTX（Branch-Train-Mix, Sukhbaatar et al. 2024）是一种将多个领域特化的 dense expert LLM 合并为统一 Mixture-of-Experts (MoE) 模型的方法。流程分三步：(1) **Branch**：从 pretrained base model 分支，对各分支在特定领域数据上做 Continual Pretraining (CPT) 得到多个 dense expert；(2) **Train/Merge**：将各 expert 的非 FFN 层（embedding、attention、normalization、head）通过 unweighted averaging 合并为一套共享参数，FFN 层保持独立作为 MoE expert，插入 MLP router 做 token-level top-K routing；(3) **Mix/Fine-tune**：合并后 MoE 在混合所有数据源的数据集上 fine-tune 40B tokens 以训练 router 并恢复因参数干扰导致的性能损失。

从算法pipeline角度拆解术语：
```
输入: base model θ_b, l 个 dense expert [θ₁,...,θₗ]（均从 θ_b 分支 CPT 得到）
输出: fine-tuned MoE model θ_m

// Step 1: 非 FFN 层平均合并
for each non-FFN layer (embedding, attention, norm, head):
    θ_m[layer] = 1/l * Σ_{i=1}^{l} θ_i[layer]     // unweighted averaging

// Step 2: FFN 层保留 + 插入 router
θ_r = random_init_MLP(hidden_dim, l)

// Step 3: MoE layer forward
v = input_token_embedding
FF_MoE(v) = Σ_{i=1}^{K} SoftMax(top-K(θ_r · v)) · FF_i(v)

// Step 4: Fine-tuning 40B tokens on mixed data
```

术语一般如何实现？如何使用？
- 基于标准 PyTorch 分布式训练。BTX 论文使用 8-expert MoE，训练在 64-128 GPUs。
- 局限：(1) 所有 expert 需同架构（从同一 ancestor 分支）；(2) 简单平均忽略参数干扰（sign conflict, magnitude disparity）；(3) fine-tuning 通信开销大。
- MergeME 以 BTX 为 baseline，改进为 Dare/Ties 替代平均、PPL 路由替代 training router、projector 方法支持异构合并。

**Nexus 对 BTX 的改进**：Nexus 在 BTX 的 upcycling 框架基础上做了三处改进：(1) **Router 设计**：BTX 使用标准线性 router W_r ∈ R^{h×n}（从零训练），Nexus 用基于域嵌入投影的自适应 router e_i = P_r(d_i)（P_r 为 2-layer SwiGLU MLP），路由概率 s_i = softmax(x · e_i)；(2) **Shared Expert**：BTX 将 seed model FFN 作为一个普通 routed expert，Nexus 将其作为 always-activated shared expert；(3) **扩展能力**：BTX 不支持扩展，Nexus 可通过计算新域嵌入 → 投影 → appendix FFN 高效添加新 expert。实验显示 Nexus 在 470M/2.8B 规模上分别优于 BTX-style MoE (Linear Router) 2.1%/1.6%（相对），扩展新 Code expert 时相对 gain 18.8%。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

---
