## INT8 Trainable Attention (SageBwd)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SageBwd 是首个支持训练的 8-bit 量化 attention 方案。不同于已有工作（FlashAttention3 FP8、SageAttention INT8）仅支持推理，SageBwd 同时实现 attention 的前向和反向 INT8 量化计算。前向对 QK^T（per-block INT8）和 PV（P 做 per-token INT8 + V 做 per-block INT8）两个 MatMul 量化。反向涉及 5 个 MatMul（S=QK^T, dV=P^T dO, dP=dO V^T, dQ=dS K, dK=dS^T Q），其中 dP=dO V^T 被识别为精度最关键的操作——其误差通过 FlashAttention 循环沿序列长度累积到 dQ/dK——因此保持 dOV^T 在 FP16，其余 4 个 MatMul 做 INT8 per-block 量化。选择 INT8 而非 FP8 作为训练量化精度，因 INT8 反向梯度精度更高（CosSim 0.9987 vs FP8 0.9880）且硬件支持更广泛（A100/AMD/Ascend 均支持 INT8 Tensor Core）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

前向（Algorithm 2 简化）：
```
输入: Q, K, V ∈ FP16
s_Q, Q̂_i = ψ(Q_i)   // per-block INT8: s_X = max(|X|)/127
s_K, K̂_j = ψ(K_j^T)
s_V, V̂_j = ψ(V_j)
K_m = mean(K); K ← K - K_m  // Smoothing K

// QK^T matmul in INT8
S_ij = MM(Q̂_i, K̂_j) × s_Q × s_K + rowsum(Q_i)K_m^T  (注:论文Algorithm 2实际未用FP4MM而是量化MatMul)

// Online softmax + per-token quant for P
m_ij = max(m_{i,j-1}, rowmax(S_ij))
P̃_ij = exp(S_ij - m_ij)
s_P = exp(rowmax(S_ij) - m_ij) / 127   // per-token scale, reuse softmax max
P̂_ij = P̃_ij / s_P                        // per-token INT8 quantization

// PV matmul
O_ij = diag(e^{m_{i,j-1}-m_ij}) * O_{i,j-1} + MM(P̂_ij, V̂_j) × s_P × s_V
```

反向（Algorithm 3 简化，关键操作）：
```
// 仅 dP = dO V^T 保持 FP16
dP_ij = MM(dO, V_j^T)   // FP16, 不量化

// 其余 4 个 MatMul 做 INT8 per-block
dS_ij = P_ij ∘ (dP_ij - D_i)             // element-wise
s_dS, dŜ_ij = ψ(dS_ij)                    // INT8 per-block
dQ_i += MM(dŜ_ij, K̂_j) × s_dS × s_K       // INT8
dK_j += MM(dŜ_ij^T, Q̂_i) × s_dS × s_Q     // INT8
s_dO, dÔ_i = ψ(dO_i)                      // INT8 per-block
s_P, P̂_ij = ψ(P_ij)                        // INT8 per-block (重算P)
dV_j += MM(P̂_ij^T, dÔ_i) × s_P × s_dO     // INT8
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SageBwd 使用 OpenAI Triton 实现 forward+backward kernel。相比 FP16 FlashAttention，前向实现约 2× 加速，反向 1.2~1.6× 加速，端到端 forward+backward 最高 1.67× 加速（RTX4090）。适用场景：fine-tuning 任务（Qwen2.5、Llama3.2 fine-tune on GSM8K/MMLU/DROP/HELLASWAG 达到 BF16 同等精度），但不适用于 pretraining（收敛速度较慢）。INT8 选择比 FP8 更优：梯度 L1 error 更低（dQ: 0.029 vs 0.070）、硬件支持更广。开源参考：https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
