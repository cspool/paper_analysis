## PM-KVQ Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PM-KVQ 是一种面向 long-CoT LLM 的 post-training KV Cache 混合精度量化方法，包含三个核心组件：(1) **Progressive Quantization（渐进量化）**——初始以 16-bit 存储 KV Cache，当显存耗尽时通过 bit-width shrinking 逐步将已有 KV Cache 降位宽（16→8→4→2 bit），同时新 token 继续以当前最高位宽存储，等价右移操作 `X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b` 等效于先反量化再量化；(2) **Block-wise Memory Allocation（块级内存分配）**——用一阶 Taylor 近似估计每个 transformer block 的 KV Cache 对 b-bit 量化的敏感度 `s_{i,b} = ||G_{K_i} ⊙ (K_i - Q_b(K_i))||_1 + ||G_{V_i} ⊙ (V_i - Q_b(V_i))||_1`，将 bit-width 分配形式化为 Integer Programming 问题由 CVXPY 求解，为更敏感的 block 分配更高位宽；(3) **Calibration with Positional Interpolation（位置插值校准）**——用短上下文校准数据（2048 tokens）配合 RoPE 位置插值 `cos(s·mθ_i)` 近似长上下文数据分布（s=4 嵌入 8192 上下文），避免因 RoPE 低频通道周期超过校准数据长度导致校准偏差。同时继承 KIVI 的 channel-wise reparameterization（将 Key Cache outlier 迁移到 Query）、首 token INT16 保留和 128 token 滑动窗口。
  - 实验比较：PM-KVQ vs **RotateKV**（uniform bit-width + Hadamard rotation）、**MiKV**（heavy-hitter oracle + mixed precision）、**KIVI**（per-channel Key + per-token Value），在 4-bit（DeepSeek-LLaMA-8B）和 2-bit（其他 LLM）下比较数学推理（AIME-2024/2025 pass@1 和 Voting）和编程（LiveCodeBench pass@1、CMIMC-2025 pass@1 和 Voting）。消融实验：(a) 三种 bit-width shrinking 策略对比（Direct Right Shift / Modified Right Shift / Equivalent Right Shift）；(b) 不同校准长度和位置插值因子对比。

- 硬件平台是什么，配置是什么。
  - 性能评估：8×A100-80G GPU 服务器（fake quantization，非真实量化推理）。单 GPU 设定：DeepSeek-Qwen-7B 用 1×4090-24G、DeepSeek-LLaMA-8B 用 1×4090-24G、DeepSeek-Qwen-14B 用 1×A100-40G、DeepSeek-Qwen-32B 用 1×A100-80G、QwQ-32B 用 1×A100-80G、DeepSeek-LLaMA-70B 用 1×A100-80G。

- 模型是什么。数据集和bench分别是什么。
  - 模型：**DeepSeek-R1-Distill-Qwen-7B/14B/32B**、**DeepSeek-R1-Distill-LLaMA-8B/70B**、**QwQ-32B**。所有模型使用 GQA 或 MHA attention。
  - 校准数据集：**RedPajama arXiv 子集**，512 条样本，每条 2048 tokens。位置插值因子 s=4（有效上下文 8192 tokens），α 参数在 [0,1] 以 grid size 20 搜索最小化 self-attention 重建损失。
  - 评估 Benchmark：**AIME-2024**（30 题）、**AIME-2025**（30 题）——数学竞赛 pass@1 和 Voting（16 次采样）；**CMIMC-2025**——数学竞赛 pass@1 和 Voting（16 次采样）；**LiveCodeBench**（2025.1.1-4.6 题目）——代码生成 pass@1（4 次采样）。采样参数：temperature=0.6, top-p=0.95, max output length=32768 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub **https://github.com/thu-nics/PM-KVQ**。
  - PM-KVQ 算法完整 Pipeline 伪代码：

  ```
  === 阶段 1: 离线校准与块级内存分配 ===
  输入: long-CoT LLM (N 个 transformer blocks), 校准数据 D_cal (512 seqs × 2048 tokens)

  # Step 1.1: 块级敏感度分析
  for i = 1 to N:  # 每个 transformer block
      用 D_cal 做前向传播，计算 KV Cache 和梯度
      for b in B (可选位宽集合，如 {2,4} 或 {4,8}):
          对 K_i, V_i 做 b-bit fake quantization → Q_b(K_i), Q_b(V_i)
          计算敏感度: s_{i,b} = ||G_{K_i} ⊙ (K_i - Q_b(K_i))||_1
                            + ||G_{V_i} ⊙ (V_i - Q_b(V_i))||_1

  # Step 1.2: Integer Programming 求解 bit-width 分配
  构建 ILP:
    minimize  Σ_i Σ_b x_{i,b} · s_{i,b}
    subject to:
      Σ_b x_{i,b} = 1, ∀i   # 每个 block 分配一个 Fbit
      Σ_i Σ_b x_{i,b} · (Mem(Q_b(K_i)) + Mem(Q_b(V_i))) ≤ M  # 显存约束
      x_{i,b} ∈ {0,1}
  调用 CVXPY 求解 → 得到每个 block i 的目标 Fbit b_i^*

  # Step 1.3: 位置插值校准
  对 RoPE 旋转矩阵引入位置缩放因子 s=4:
    [K̃_{m,i}; K̃_{m,i+d/2}] = [cos(s·mθ_i) -sin(s·mθ_i); sin(s·mθ_i) cos(s·mθ_i)] [K_{m,i}; K_{m,i+d/2}]
  用校准数据计算 channel-wise reparameterization factor:
    λ_i = (max_m K_{m,i})^α
  其中 α 在 [0,1] 以 grid size 20 搜索最小化 self-attention 重建损失

  === 阶段 2: 推理时渐进量化 ===
  # 对每个 transformer block i，目标 Fbit = b_i^*
  # 设当前 KV Cache 位宽为 current_bit = 16
  # 预留显存 budget = b_i^* 对应最大上下文长度所需显存

  for each decoding step t:
      # 新 token 的 K, V 以 current_bit 存储
      K_new, V_new = current_token_key_value

      if memory_usage + new_token_memory > memory_budget:
          # 位宽缩减: current_bit 降一档 (16→8, 8→4, 4→2)
          new_bit = current_bit // 2
          # Equivalent Right Shift: 等效于反量化→量化
          X_new_bit = ((2^{2*new_bit} - 2^{new_bit} + 1) * (X_old + 2^{new_bit-1})) >> (3*new_bit)
          # 零点和缩放因子更新: Z_b = Z_{2b}, S_b = (2^b + 1) S_{2b}
          current_bit = new_bit

      # 存储新 token KV 到 cache
      KV_cache.append(K_new, V_new, bit=current_bit)

      # 注意力计算（混合精度）:
      # - 首 token 始终 INT16
      # - 最近 128 token 始终 INT16
      # - 其余按 current_bit × Fbit 的渐进量化结果
      attention_output = mixed_precision_attention(Q, KV_cache)
  ```

  - 位宽缩小策略核心张量计算（Equivalent Right Shift）：
    - **16→8 bit**: `X_8 = ((2^{16} - 2^8 + 1)(X_16 + 2^7)) >> 24`，`Z_8 = Z_16`, `S_8 = (2^8 + 1)S_16`
    - **8→4 bit**: `X_4 = ((2^8 - 2^4 + 1)(X_8 + 2^3)) >> 12`，`Z_4 = Z_8`, `S_4 = (2^4 + 1)S_8`
    - **4→2 bit**: `X_2 = ((2^4 - 2^2 + 1)(X_4 + 2^1)) >> 6`，`Z_2 = Z_4`, `S_2 = (2^2 + 1)S_4`
    - 等效性证明：先将 2b-bit 量化整数反量化为浮点 → 再重新量化为 b-bit，等价于上述整数移位操作

  - 关键超参数：
    - 量化方式：asymmetric group-wise quantization，group size=128
    - Fbit 设置：DeepSeek-LLaMA-8B 用 4-bit，其他 LLM 用 2-bit
    - 可选位宽集合 B：DeepSeek-LLaMA-8B 用 {4, 8}，其他 LLM 用 {2, 4}
    - 首 token：INT16 保留
    - 滑动窗口：最近 128 tokens INT16（继承自 KIVI/SKVQ）

  - 核心结果：PM-KVQ 在 2-bit 下比 KIVI 提升最高 8% pass@1（数学推理），Voting 提升最高 15.56%（7B 级模型）、17.78%（10B-32B 级模型）。70B 级 LLaMA-70B 在 AIME-2024 上 pass@1 从 KIVI 的 51.88% 提升到 64.79%。
