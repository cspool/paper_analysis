## Load Balancing Loss (MoE Auxiliary Loss)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load balancing loss（负载均衡损失）是 MoE 训练中的辅助损失函数，用于激励 Router 在 token 分配时产生均匀的 expert 负载。定义为 L_aux = num_experts × Σ_e f_e × P_e，其中 f_e 是分配给 expert e 的 token 比例，P_e 是 Router 分配给 expert e 的平均概率。最小化此损失鼓励两个目标的均匀性：(1) 实际分配的 token 数（f_e），(2) Router 的 softmax 概率（P_e）。由 Shazeer et al. (2017) 首次提出，Switch Transformer (Fedus et al. 2022) 广泛使用。该损失以权重 α 加到主任务损失上。虽然能改善负载均衡，但 MegaBlocks 论文指出即使使用负载均衡损失，token routing 仍然高度不均衡（Hwang et al. 2022 也证实此事）。

从算法pipeline角度拆解术语：
Load balancing loss 的计算：
```
输入: router_probs (num_tokens, num_experts)  # softmax 输出
      expert_indices (num_tokens,)             # top-k 选择
输出: L_aux (scalar)

# 1. 计算每个 expert 的实际 token 比例
f_e = count(indices == e) / num_tokens

# 2. 计算每个 expert 的平均路由概率
P_e = mean(router_probs[:, e])

# 3. 负载均衡损失
L_aux = num_experts * sum_e (f_e * P_e)

# 4. 总损失
L_total = L_task + α * L_aux  # α 典型值为 0.01
```
L_aux 在负载不均衡时较大（某些 expert f_e 高 P_e 高），在均匀时最小（所有 f_e = 1/E, P_e = 1/E 时 L_aux = 1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 PyTorch 中通过 `torch.scatter_add` 或 `torch.bincount` 统计每个 expert 的 token 数，再计算 f_e 和 P_e 的点积。
- α 通常设为 0.01（Switch Transformer）。过大的 α 会干扰主任务学习，过小则效果不足。
- 除了计算效率价值，负载均衡损失还确保所有 expert 在训练中看到足够的 token 以避免退化——某些 expert 可能长时间收不到 token 而停止接收梯度更新（Zhou et al. 2022）。
- 替代方法：BASE layers (Lewis et al. 2021) 将路由建模为线性分配问题保证完美均衡；Expert Choice Routing (Zhou et al. 2022) 反转路由方向让 expert 选择 token。

**Nexus 的发现：自适应 Router 对负载均衡损失不敏感。** Nexus 的 ablation（Figure 7）对比了 load balancing loss factor α=0.05 和 α=0.0005 的效果。结果：线性 router 的 upcycled MoE 在 α=0.0005 时性能下降约 2%（相对），而 Nexus 的自适应 domain-embedding router 在两个 α 值下性能几乎不变。原因：Nexus router 的 expert embedding 始终基于域表示（e_i = P_r(d_i)），即使负载均衡损失权重极低，域语义本身也能提供稳定的 token 分配——专家嵌入的内在域语义充当了隐式正则化。这使得 Nexus 在实际部署中无需精细调优 load balancing loss 超参数。

**MoLE 的特殊情况：全激活训练的天然均衡性。** MoLE 在所有 experts 始终激活（不做 top-K 稀疏选择）且接收梯度的全激活训练范式下，Router 不会面临 collapse 风险。因此 MoLE 仅使用 language modeling cross-entropy loss，无需任何 auxiliary loss。MoLE 的 ablation（Table 4）显示，添加 load balance loss 和 z-loss 后模型性能反而下降（MoLE-16E 160M: LM loss only AVG 41.9 → +load_balance 41.7 → +z-loss 40.6），因为 auxiliary loss 使优化目标与推理需求不对齐。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning (per-layer prompt-level routing, f_i = proportion of prompts assigned to expert i, p̂_i = mean probability mass, λ_lb=1e-2)
- Mixture-of-Experts with Expert Choice Routing
- Mixture of Lookup Experts
- MoH: Multi-Head Attention as Mixture-of-Head Attention
- MoLA: MoE LoRA with Layer-wise Expert Allocation (per-layer load balancing loss: L_aux = Σ_j N_j · Σ_e f_e^j · P_e^j, where N_j is layer j's expert count; follows Switch Transformers formulation)
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts (adaptive domain-embedding router is robust to low α = 0.0005, while linear router drops ~2%)

**MoH 中的应用**：MoH 将 load balance loss 应用于 attention head 级别的路由（而非 FFN expert）。L_b = Σ_{i=h_s+1}^{h} P_i · f_i，仅对路由头计算（共享头始终激活无需均衡）。P_i = mean(Softmax(W_r·x_t)[i-h_s])，f_i = mean(token选择head i 的指示函数)。β=0.01 对所有任务（ViT/DiT/LLM）通用。

---
