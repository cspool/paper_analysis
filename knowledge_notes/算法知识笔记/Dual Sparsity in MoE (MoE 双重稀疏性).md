## Dual Sparsity in MoE (MoE 双重稀疏性)

术语解释
MoE 推理中的"双重稀疏性"指 tensor-level sparsity（张量级，即 Top-K 专家选择的不均衡）和 neuron-level sparsity（神经元级，即 SwiGLU FFN 内部激活值分布不均）在不同粒度上同时存在，共同决定计算效率与精度。每个 token 的 FFN 输出由 gating score × activation value 联合调制，两种稀疏性协调利用可在 ~25% drop rate 下仅损失 0.08-0.28% accuracy。

术语是什么？
DualSparse-MoE 论文的核心观察：对 MoE 模型做一次推理前向，沿 expert × neuron 维度可视化 accumulated absolute activation values，发现 (1) Tensor-Level Sparsity（y 轴）：不同 expert 被激活频率极不均衡（高负载 expert 处理大量 token，低负载仅处理极少 token）；(2) Neuron-Level Sparsity（x 轴）：每个 expert 内部 neuron 的 |Swish(x·W₁) ⊙ (x·W₃)| 高度不均（少数 neuron 贡献大部分 output magnitude，大量 neuron 接近零但非硬零，因 SwiGLU 无 ReLU 般硬零）。Tensor-level 用于 coarse-grained expert dropping（1T-Drop/2T-Drop），Neuron-level 用于 fine-grained major/minor reconstruction。Profiling 发现低负载 expert 有大量负 accumulated gate value，而高负载 expert 罕见 → 暗示两种稀疏性存在内在关联。

从算法pipeline角度拆解术语：
```
# Double-sparse observation
For each MoE layer:
  heatmap = zeros(E, d_ffn)
  For each token t:
    s = TopK(Softmax(t·W_g), K)
    For each activated e:
      gate_out = Swish(t·W1_e)
      up_out = t·W3_e
      heatmap[e, :] += |gate_out ⊙ up_out|
  # heatmap rows → tensor-level imbalance
  # heatmap columns within a row → neuron-level imbalance

# Two-level exploitation
L1 (Tensor-level): 1T-Drop via normalized gating score threshold T_drop^1
L2 (Neuron-level): 2T-Drop with T_major^2 < T_minor^2
  - score < T_major^2: skip expert entirely
  - T_major^2 ≤ score ≤ T_minor^2: compute major sub-expert only
  - score > T_minor^2: compute full expert
```

术语一般如何实现？如何使用？
- 观察：在 calibration data (MMLU/C4) 上前向传播，记录 per-expert per-neuron 的 accumulated |gate·up|
- Tensor-level 利用：expert partition 增加专家粒度 → fine-grained dropping；1T-Drop/2T-Drop 按归一化 gating score 阈值丢弃
- Neuron-level 利用：importance profiling → major/minor split → 2T-Drop 的中间档仅计算 major half
- Profiling 选择：Mixtral+OLMoE 适用 accumulated abs gate value；DeepSeek-V2-Lite-Chat 适用 accumulated abs gate-up value
- 跨任务泛化性：gating score 分布在不同 benchmark 间高度一致（图 6c），保证 threshold-based dropping 的泛化性

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
