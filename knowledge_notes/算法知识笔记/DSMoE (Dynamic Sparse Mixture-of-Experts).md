## DSMoE (Dynamic Sparse Mixture-of-Experts)

术语解释
DSMoE (Dynamic Sparse Mixture-of-Experts) 是一种将预训练 Dense LLM 的 FFN 层转化为输入自适应的稀疏 MoE 的方法，由 Lv et al. (2025) 提出。其核心创新在于：(1) 通过矩阵分区保留全部预训练知识；(2) 使用 sigmoid 门控替代 softmax top-k 实现每个 expert 独立激活决策；(3) 通过 Straight-Through Estimator 和稀疏损失实现端到端的动态稀疏模式学习。与剪枝（永久丢弃参数）和传统 MoE（固定 top-k 激活 + load balancing loss）有本质区别。

术语是什么？
DSMoE 由三个正交模块组成，总损失函数为 L = L_LM + (1/(L·N)) · Σ_l Σ_n G(σ(ĥY_n))，无可选的 load balancing loss：

**模块 1 — FFN Partitioning**：将 SwiGLU FFN 的三个权重矩阵 (W_gate, W_up, W_down) 沿 intermediate dimension 划分为 n 个 expert，全部参数保留，所有 expert 输出之和数学等价于原始 FFN 输出。

**模块 2 — STE-Enhanced Sigmoid Gating**：使用 sigmoid 激活（非 softmax）使每个 expert 独立判断是否激活（σ(xY_i) > τ）。与 softmax 的关键区别：sigmoid 输出非归一化，各 expert 决策互不依赖，允许变长 expert 集合。STE 确保所有 expert 门控参数在反向传播中接收梯度。

**模块 3 — L1 Sparse Loss**：惩罚门控激活值，与 STE 门控梯度形成对抗，鼓励模型为不同输入学习不同的稀疏激活模式。

从算法pipeline角度拆解术语：
DSMoE 训练和推理的完整流程：

```
=== DSMoE Training Pipeline ===

# 初始化:
# 1. 加载预训练 Dense LLaMA 模型
# 2. 对每层 FFN 执行 Matrix Partitioning → n 个 expert
# 3. 初始化门控参数 Y ∈ R^{d×n}（随机初始化）
# 4. 不引入 load balancing loss

for each training step:
    for each Transformer layer l:
        # Step 1: Self-Attention（不变）
        x = attention(layer_norm_1(h_prev))
        h_hat = residual(x, h_prev)
        
        # Step 2: DSMoE FFN (替代原 Dense FFN)
        gate_probs = sigmoid(h_hat @ Y)           # [B, n]
        gate_ste = STE(gate_probs, threshold=0.5) # 前向稀疏 + 反向全梯度
        
        # Step 3: Expert 计算（仅激活 expert 参与）
        outputs = []
        for i in range(n):
            if gate_ste[:, i].any():
                o_i = expert_i_swiglu_ffn(h_hat)
                outputs.append(o_i * gate_ste[:, i:i+1])
        
        h = sum(outputs) * (n / num_active)  # 归一化
        
        # Step 4: 收集门控值用于 sparse loss
        layer_gate_values.append(gate_ste)
    
    # Step 5: Loss 计算
    lm_loss = cross_entropy(logits, targets)
    sparse_loss = (1/(L*N)) * sum(g.sum() for g in layer_gate_values)
    total_loss = lm_loss + sparse_loss  # 无 load balancing loss!
    
    # Step 6: 反向传播
    total_loss.backward()
    # STE 确保所有 expert Y_i 接收梯度（即使未激活）
    optimizer.step()
```

```
=== DSMoE Inference Pipeline (per token) ===

输入: token embedding x

for each layer:
    h_hat = attention(x)
    gate_probs = sigmoid(h_hat @ Y)   # [1, 8]
    
    # 硬阈值推理
    active = gate_probs > 0.5         # e.g., [1, 0, 1, 0, 1, 0, 1, 0]
    num_active = active.sum()         # 4 of 8 active
    
    # 仅计算激活 expert
    h = zeros(d)
    for i in range(8):
        if active[i]:
            h += expert_i_ffn(h_hat) * gate_probs[i]
    
    h = h * (8 / num_active)          # 归一化
    x = layer_norm(h + residual)

# 层间激活模式: 形成 "W 形" —— 首尾层高激活、中间层突起、其余层低激活
```

术语一般如何实现？如何使用？
- **训练配置**：论文使用 lr=2e-5, batch_size=32, seq_len=1024, 10B tokens 继续预训练，threshold τ=0.5（可通过 sweep τ∈[0.2,0.8] 调节稀疏度，τ 越大越稀疏）
- **激活参数比例**：LLaMA-7B DSMoE 在 τ=0.5 时激活约 58.46% 参数（3.93B/6.74B），τ=0.8 时降至 52.54%
- **推理加速机制**：结构化 expert 跳过（未激活 expert 的矩阵乘法可直接跳过，无需稀疏计算库）vs SparseGPT 的非结构化稀疏需要专用硬件
- **vs 传统 MoE 的关键差异**：
  | 维度 | 传统 MoE (Switch/LLaMA-MoE) | DSMoE |
  |------|---------------------------|-------|
  | 路由函数 | Softmax top-k | Sigmoid + 阈值 |
  | Expert 激活数 | 固定 k | 变长 (1~n) |
  | Load balancing | 需要 auxiliary loss | 不需要 |
  | Expert 初始化 | 随机初始化 | 从预训练模型分区继承 |
  | 训练目标 | L_LM + L_aux (load balance) | L_LM + L_sparse (L1) |
  | 路由策略 | 竞争性（归一化后选择） | 独立性（每个 expert 独立决策） |
- **层间激活模式（W 形）**：底层（高激活，多维特征处理）→ 中层（突起激活，关键特征转换区）→ 中上层（低激活，特化处理）→ 顶层（高激活，综合决策）。不同输入激活模式不同，体现输入自适应特性
- **局限性**：仅验证到 7B 参数（受计算资源限制），更大模型的扩展行为未知；门控参数 Y 需从随机初始化训练（可能导致训练初期路由不稳定）；训练和推理的 gate 行为不一致（训练 STE vs 推理纯硬阈值）；开源代码未公开

涉及论文标题：
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

---
