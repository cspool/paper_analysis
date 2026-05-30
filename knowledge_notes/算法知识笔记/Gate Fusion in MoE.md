## Gate Fusion in MoE

术语是什么？
Gate Fusion（门控融合）是 MoE 架构中用于聚合多个专家网络输出信息的机制。标准实现为：一个可学习的门控网络（通常为线性层 + softmax）根据输入样本生成 N 维权重向量，对 N 个专家的输出做加权求和：output = Σ_{e=1}^{N} gate[e] · expert_e(input)。门控融合的关键特性是样本自适应（sample-wise）——不同样本可能激活不同的专家组合。在 M3oE 中，Shared Expert Module 使用 D×T 个独立 gate 为每一对域-任务生成专属的专家融合权重（公式 4），确保不同优化目标能从共享知识中获取不同比例的信息。而对于 Domain/Task Expert Module，则采用基于可训练标量的偏置融合而非门控网络，以降低参数量并实现更显式的解耦控制。

从算法pipeline角度拆解术语：
```
// 标准 Gate Fusion (MMoE)
h = input_embedding
gate_logits = W_gate @ h + b_gate     // shape: (batch, N)
gate_weights = softmax(gate_logits, dim=-1)  // 每行和为1
expert_outputs = stack([expert_i(h) for i in 1..N])  // shape: (batch, N, hidden)
fused = sum(gate_weights[:, :, None] * expert_outputs, dim=1)  // 加权求和

// M3oE Shared Module 的 Gate Fusion
// 有 D×T 个独立 gate，每个负责一对 (d,t)
for d in 1..D, t in 1..T:
    gate_{d,t} = softmax(W_{d,t} @ h_d + b_{d,t})
    S_{d,t}(h_d) = gate_{d,t} · [expert_1(h_d), ..., expert_N(h_d)]
```
注意 M3oE 的 Shared gate 与 domain expert / task expert 模块的融合方式不同：后者使用固定标量权重（β_d, β_t）而非输入依赖的门控，因为域/任务归属在样本级别已确定，无需样本自适应融合。

术语一般如何实现？如何使用？
Gate Fusion 在 PyTorch 中通常通过 `nn.Linear(hidden_dim, num_experts)` + `F.softmax(dim=-1)` 实现，结合 `torch.einsum` 或广播乘法完成加权求和。Gate 融合适用于专家选择依赖输入内容的场景（如 Shared Expert，因为不同样本利用跨域共性知识的程度不同），而对域/任务归属已知的场景，使用可训练标量权重更高效且可解释性更强。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---
