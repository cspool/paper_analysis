## Post-training Expert Pruning via Reconstruction Loss Minimization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Post-training Expert Pruning（后训练专家剪枝）是一种无需训练的 MoE LLM 压缩技术，通过永久移除 MoE 层中贡献较小的 expert 来减少模型参数总量和部署内存需求。该方法属于 expert-level sparsity（专家级稀疏），与传统的 weight-level 剪枝（如 Wanda 的非结构化稀疏、SparseGPT 的 2:4 结构化稀疏）正交。

核心思想：给定一个训练好的 MoE 模型（如 Mixtral 8x7B，每层 8 个 expert），对每一层独立地枚举所有可能的保留 r 个 expert 的组合 C（从 n 个中选 r 个），对每个组合计算"token 重建损失"——即 prune 后 MoE 层输出 F'(x, C) 与原始层输出 F(x) 之间的 Frobenius 范数差异。选择使重建损失最小的组合 C*，丢弃其余 n−r 个 expert。

公式化为逐层优化问题（受 He et al. 2017 channel pruning 启发）：

$$\min_{\mathbf{C}} \|\mathcal{F}'(\boldsymbol{x}, \mathbf{C}) - \mathcal{F}(\boldsymbol{x})\|_F \quad \text{s.t.} \quad \mathbf{C} \subseteq \{\text{expert}_0, \dots, \text{expert}_{n-1}\}, |\mathbf{C}| = r$$

其中 x 为校准集缓存的输入，F(x) 为原始输出，F'(x, C) 为仅保留 C 中 expert 及对应 routing weight 后的输出。

关键实现细节：
- 校准集：task-agnostic 使用 C4（与 Wanda 一致，pre-training 数据覆盖面广）；task-specific（如数学）切换到 MATH training set
- 枚举复杂度：每层 C(n, r) 种组合。Mixtral 8x7B 的 n=8，r=6 时 C(8,6)=28 种；r=4 时 C(8,4)=70 种。Pruning 耗时 r=6 约 30 分钟，r=4 约 90 分钟
- 逐层独立剪枝（layer-wise），而非逐层渐进式剪枝（progressive layer-by-layer）：实验表明 progressive 方式在高速剪枝率下会过拟合小校准集（Tab. 6）
- 剪枝后仅需修改 HuggingFace model config 的 num_experts 字段即可加载部署

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ============ Step 1: 校准数据缓存 ============
calibration_set = sample(C4, num_sequences=128, seq_len=2048)
for each MoE_layer l in [0, 1, ..., L-1]:
    X_l = []   // 输入 token hidden states
    Y_l = []   // 原始输出
    for x in calibration_set:
        x_l = forward_to_layer_l(x)           // 到达第 l 层的 hidden state
        y_l = MoE_layer_l(x_l)                // 8-expert 完整推理
        X_l.append(x_l)
        Y_l.append(y_l)

// ============ Step 2: 逐层枚举搜索 ============
for each layer l:
    best_loss = inf
    best_subset = None
    for each subset C of {expert_0, ..., expert_7} with |C| == r:
        total_loss = 0
        for each (x, y) in zip(X_l, Y_l):
            y_hat = pruned_MoE_layer(x, C)    // 仅 C 中 expert 参与 top-k
            total_loss += ||y_hat - y||_F^2
        avg_loss = total_loss / len(X_l)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_subset = C
    keep[l] = best_subset

// ============ Step 3: 拼接并保存 ============
for each layer l:
    experts[l] = experts[l][keep[l]]
    router_weights[l] = router_weights[l][keep[l]]
model_config.num_experts = r
save_pruned_model(model)
```

**Pruned MoE Layer 推理（推理阶段）**：
```
Input: token x ∈ R^d
Router: l = W_router @ x                // logits ∈ R^r (仅保留的 r 个 expert)
        w = Softmax(l)                  // routing weights ∈ R^r
        {e0, e1} = TopK(w, k=2)        // 从 r 个中选 top-2
        w̃_e0 = w_e0 / (w_e0 + w_e1)
        w̃_e1 = w_e1 / (w_e0 + w_e1)
Expert e0: z0 = W_down_e0 @ (SiLU(W_gate_e0 @ x) ⊙ W_up_e0 @ x)
Expert e1: z1 = W_down_e1 @ (SiLU(W_gate_e1 @ x) ⊙ W_up_e1 @ x)
Output: z = w̃_e0 * z0 + w̃_e1 * z1
```

Annotations:
- 逐层搜索空间：C(n, r) 种组合，Mixtral n=8, r=6 → 28 种/层，32 层共 896 次评估
- 校准数据量：128 sequences × 2048 tokens = 262,144 tokens
- Pruning 后 expert 权重直接删除（deleted），不是 mask（zero-out），因此内存实际减少
- 关键设计选择：layer-wise 独立（非 progressive）是为了避免小校准集导致的过拟合

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **开源实现**：https://github.com/Lucky-Lance/Expert_Sparsity（论文承诺开源）
- **部署方式**：Prune 后仅需修改 HuggingFace model config 中 num_experts 字段，无需修改推理代码
- **Task-Agnostic vs Task-Specific**：通用任务用 C4 校准，领域任务（如数学）切换到领域数据集（如 MATH）。两种校准仅在 Fig. 4 中 4/32 层的 expert 选择相同，差异显著
- **Fine-tuning 恢复精度**：Task-specific prune 后在 MetaMathQA 上 fine-tune 900 steps（lr=2e-5, cosine scheduler, 16×A100），可将 r=6 模型的 GSM8K 从 51.25 恢复到 79.53，接近原模型 81.35
- **限制**：枚举复杂度为 O(C(n, r))，n=32 expert 时 C(32,6)=906,192 组合，不可行。论文承认此局限
- **与 weight pruning 的关系**：正交且可叠加（Sec. A.6），expert pruning 减少 expert 数量，weight pruning 减少单个 expert 内部参数，quantization 进一步减少 bit-width

涉及论文标题：
- Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models

---
