## DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包括三部分：**(1) Standard MoE for NLG**：将 MoE 架构应用于自回归 NLG 模型（GPT-like Transformer），每两层的前馈层之一替换为 MoE 层（128 experts, top-1 gating），在 300B tokens 训练下实现与 4-5x 参数量 dense 模型相同的质量。**(2) PR-MoE (Pyramid-Residual MoE)**：结合两种创新 —— Pyramid-MoE（深层 MoE 层使用更多 experts，如最后两层 2x experts）+ Residual-MoE（每 token 同时经过固定 MLP 和选定 expert，等价于 Top-2 gating 的精度但仅需 Top-1 通信量），参数效率提升 3x。**(3) MoS (Mixture-of-Students)**：Staged Knowledge Distillation，在预训练早期使用 KD（teacher PR-MoE → student PR-MoE，student 深度减少 12.5%），400K steps 后停用 KD 仅用标准 LM loss 继续训练，解决学生模型容量不足造成的 underfitting。MoS 进一步减少模型大小至 3.7x。实验比较：(a) 350M+MoE-128 vs 350M dense, 1.3B+MoE-128 vs 1.3B/6.7B dense 的 validation loss 和 zero-shot；(b) PR-MoE vs Standard MoE 的参数量和精度对比（Table 4）；(c) 5 种 MoE 架构的 ablation study（Standard-MoE-32/128, Pyramid-MoE, Residual-MoE, PR-MoE）；(d) PR-MoE+MoS vs PR-MoE 直接减层的 zero-shot 对比（Table 5）；(e) Staged KD vs Full KD vs No KD 的 validation loss 对比。

- 硬件平台是什么，配置是什么。
  128 张 NVIDIA A100 GPU（Azure ND A100 instances），通过 NCCL, Mellanox OFED, Sharp, CUDA 优化。使用 DeepSpeed 进行数据和 expert parallel 训练。

- 模型是什么。数据集和bench分别是什么。
  模型：350M (24 layers, 1024 hidden, 16 heads), 1.3B (24 layers, 2048 hidden, 16 heads), 6.7B (32 layers, 4096 hidden, 32 heads) 的 dense baseline；350M+MoE-128 (13B params), 1.3B+MoE-128 (52B params) 的 standard MoE；350M+PR-MoE-32/64 (4B params), 1.3B+PR-MoE-64/128 (31B params) 的 PR-MoE；350M+PR-MoE+L21+MoS (3.5B), 1.3B+PR-MoE+L21+MoS (27B) 的 MoS。训练数据：与 MT-NLG 相同的 300B tokens，sequence length 2K。Benchmarks：6 个 zero-shot 任务 —— LAMBADA（补全预测）、PIQA（常识推理）、BoolQ 和 RACE-h（阅读理解）、TriviaQA 和 WebQs（问答）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源，代码和文档在 DeepSpeed GitHub (https://github.com/microsoft/DeepSpeed) 和 DeepSpeed 官网 (https://www.deepspeed.ai/)。论文 ICML 2022。

  **PR-MoE 算法流程（以 350M+PR-MoE-32/64, 24 layers 为例）**：
  ```
  Input: token embeddings x_1,...,x_S ∈ R^{M}, M=1024

  // 前 10 个 MoE 层使用 32 experts (Pyramid-MoE shallow)
  For layer l ∈ MoE layers 1..10:
    // 每层 MoE 位于两个 Attention 层之间
    h = Attention(x)
    // Residual-MoE: 固定 MLP + 选定 expert 各自处理
    h_mlp = W2_fixed @ GeLU(W1_fixed @ h)        // 固定 dense MLP（所有 token 共享）
    gate_logits = W_gate @ h                        // [32] per token
    expert_id = argmax(gate_logits)                 // Top-1 gating
    h_expert = W2_expert[expert_id] @ GeLU(W1_expert[expert_id] @ h)
    x = x + h_mlp + h_expert                       // 残差连接相加

  // 后 2 个 MoE 层使用 64 experts (Pyramid-MoE deep)
  For layer l ∈ MoE layers 11..12:
    h = Attention(x)
    h_mlp = W2_fixed @ GeLU(W1_fixed @ h)
    gate_logits = W_gate @ h                        // [64] per token
    expert_id = argmax(gate_logits)                 // Top-1 gating
    h_expert = W2_expert[expert_id] @ GeLU(W1_expert[expert_id] @ h)
    x = x + h_mlp + h_expert
  ```

  **MoS Staged KD 训练流程**：
  ```
  // Teacher: 24-layer PR-MoE (1.3B+PR-MoE-64/128, 31B params)
  // Student: 21-layer PR-MoE (1.3B+PR-MoE-64/128, 27B params)

  For training step t = 1..T:
    x = next_batch()
    teacher_logits = TeacherPRMoE(x)         // 教师 soft label
    student_logits = StudentPRMoE(x)         // 学生预测

    L_CE = CrossEntropy(student_logits, y)   // 标准语言模型 loss
    L_KD = KLDiv(student_logits, teacher_logits)  // 蒸馏 loss

    if t <= 400K:
      L = L_CE + α * L_KD                   // 使用 KD loss + CE loss
    else:
      L = L_CE                               // 停用 KD，仅优化标准 LM loss
    optimizer.step(L)
  ```
  关键创新：发现全程 KD 在训练后期（>400K steps）开始伤害精度，原因在于学生容量不足导致 underfitting —— 学生无法同时最小化 L_CE 和 L_KD。Staged KD 通过早期 KD 提供引导、后期放开自主优化来解决此问题。
