## PR-MoE (Pyramid-Residual Mixture-of-Experts)

术语解释
PR-MoE 是 DeepSpeed-MoE 提出的混合专家架构，将 Pyramid-MoE（深层更多专家）和 Residual-MoE（固定 MLP + 可变专家作为残差修正）结合，在不牺牲模型质量的前提下减少 MoE 参数至 3x。

术语是什么？
PR-MoE 由两个独立验证的设计观察推动：

- **Phenomenon-I（Pyramid-MoE 的基础）**：在 CV 中已知浅层学通用特征、深层学任务特定特征，但在 NLP/MoE 中未被验证。论文通过对比 First-Half-MoE（前一半层含 MoE）和 Second-Half-MoE（后一半层含 MoE），发现后者性能显著优于前者，证明**深层使用 MoE 带来的收益更大**。由此提出 Pyramid-MoE：深层 MoE 层使用更多专家（如 350M+PR-MoE-32/64：前 10 层 32 experts/层，后 2 层 64 experts/层）。

- **Phenomenon-II（Residual-MoE 的基础）**：增加 expert capacity（每 token 激活更多专家，如 Top-2 gating）能提升精度，但 all-to-all 通信量翻倍。论文发现将 Top-2 gating 改为固定 dense MLP + 1 个可变 expert（残差相加），精度等价于 Top-2，但通信量等价于 Top-1（因为仅需传输 1 个 expert 的 token）。这种设计将专家视为对固定 MLP 输出的"误差修正项"。

PR-MoE 是 Pyramid-MoE 和 Residual-MoE 的组合：所有标准 MoE 层替换为 PR-MoE 层。

从算法pipeline角度拆解术语：
```
# PR-MoE Layer 前向计算（per token）
Input:  hidden_states h ∈ R^{M}

# Step 1: Attention（与标准 Transformer 相同）
h = SelfAttention(h)

# Step 2: Residual-MoE
h_mlp = W2_fixed @ GeLU(W1_fixed @ h)           # 固定 dense MLP（所有 token 共享此路径）
                                                    # W1_fixed: [4M, M], W2_fixed: [M, 4M]
gate_logits = W_gate @ h                           # [num_experts]，当前层的 expert 数（可变）
expert_id = argmax(Softmax(gate_logits))           # Top-1 gating
h_expert = W2[expert_id] @ GeLU(W1[expert_id] @ h) # 选中的 expert FFN
h = h + (h_mlp + h_expert)                         # 残差连接：固定 MLP 输出 + 专家输出

# Pyramid-MoE 配置示例（350M+PR-MoE-32/64, 24 layers, 12 MoE layers）：
# Layers  1-10 (MoE layers 1-10): 32 experts per layer
# Layers 11-12 (MoE layers 11-12): 64 experts per layer
# 深层使用 2x 专家
```

关键训练设计：
由于不同层有不同 expert 数，传统单一 expert parallelism degree 不再高效。DeepSpeed 实现 multi-expert + multi-data parallelism：128 GPUs 上，32-expert 层使用 {EP=32, DP=4}，64-expert 层使用 {EP=64, DP=2}，128-expert 层使用 {EP=128, DP=1}。每 GPU 始终保持恰好 1 expert，避免 load imbalance 和 batch size 降低。

术语一般如何实现？如何使用？
- 开源：https://github.com/microsoft/DeepSpeed
- 替换标准 MoE layer 为 PR-MoE layer，API 与 DeepSpeed MoE API 兼容
- 训练时需配置 multi-expert + multi-data parallelism（自动通过 DeepSpeed runtime 处理）
- 推理时 PR-MoE 的固定 MLP 路径可与 attention 等 non-expert 操作一起通过 tensor-slicing 并行

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---
