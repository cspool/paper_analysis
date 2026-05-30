## Block-wise Memory Allocation for KV Cache Quantization（KV缓存块级内存分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Block-wise Memory Allocation 是 PM-KVQ 提出来解决不同 Transformer block 的 KV Cache 对量化的敏感度差异问题。核心思想是：不为所有 block 分配统一位宽，而是通过一阶泰勒近似评估每个 block 的 KV Cache 敏感度，将内存分配建模为整数规划问题，用 CVXPY 求解器在几秒内给出每个 block 的最优位宽配置。敏感 block（深层 block + 第一层）获得更多内存（更高位宽），不敏感 block 使用更低位宽，在相同总内存预算下最大化模型精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === 预处理阶段：Block-wise Sensitivity Profiling ===
// 校准数据：RedPajama arXiv subset, 512 samples × 2048 tokens

for each calibration_sample:
    for each transformer_block i in [0..N-1]:
        // 前向传播获取 KV Cache 和梯度
        K_i, V_i = block.compute_kv(hidden_states)

        for each candidate_bitwidth b in B:
            // 量化 KV Cache
            K_i_q = asymmetric_group_quantize(K_i, bits=b, group_size=128)
            V_i_q = asymmetric_group_quantize(V_i, bits=b, group_size=128)

            // 一阶泰勒敏感度估计
            G_Ki = grad(loss, K_i)  // 损失对 Key Cache 的梯度
            G_Vi = grad(loss, V_i)

            s_{i,b} = ||G_Ki ⊙ (K_i - K_i_q)||_1 + ||G_Vi ⊙ (V_i - V_i_q)||_1

// === 整数规划求解 ===
variables: x_{i,b} ∈ {0,1}  // 每个 block 选一个位宽

Objective:
min Σ_i Σ_b x_{i,b} · s_{i,b}

Constraints:
Σ_b x_{i,b} = 1, ∀i                       // 每个 block 恰好一个位宽
Σ_i Σ_b x_{i,b} · Mem(Q_b(K_i)+Q_b(V_i)) ≤ M  // 总内存不超过预算

solved_by: CVXPY (几秒内求解, Diamond & Boyd 2016)

// === 推理时使用 ===
// 求解结果为每个block i 分配 Fbit_i
for each decoded_token:
    for each block i:
        target_bitwidth_i = solved_x_{i,b}

        // Progressive Quantization with per-block Fbit
        if block i 使用 progressive quantization:
            从16bit开始逐步降到 Fbit_i
        else:
            直接量化到 Fbit_i
```

从 Paper Figure 3/4 的敏感度分析可知：
- 深层 block 对量化更敏感 → 分配更高位宽（如 4-bit）
- Qwen 模型第一层 block 异常敏感 → 获得最高位宽
- LLaMA 模型各层敏感度相对平滑 → 位宽分配更均匀
- B = {2,4}（7B<模型）或 {4,8}（7B 模型），即每个 block 获得 2-bit 或 4-bit（或 4-bit/8-bit）

术语一般如何实现？如何使用？

实现关键：(1) 一阶泰勒近似需要在校准数据上做一次前向传播记录梯度，对 7B-70B 模型可在数分钟内完成；(2) CVXPY 求解整数规划仅需几秒；(3) 位宽可选集合 B 一般设 2 个候选值（{2,4} 或 {4,8}），限制解空间并保证求解速度；(4) 该策略与渐进式量化正交——每个 block 的 Fbit 由整数规划确定，但推理时仍可渐进降低到 Fbit。

适用场景：(1) 当可用内存不足以将所有 block 统一升级到更高位宽时（如 batch size 减小后单样本内存增加，但不足以统一 4-bit）；(2) 与均匀位宽配合——在大 batch 时使用均匀位宽，小 batch 时切换为 block-wise 分配以利用额外内存。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---
