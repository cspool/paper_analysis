## Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出 Sherry，一种 1.25-bit 三值量化方法，核心实现：(1) **3:4 细粒度结构化稀疏**：在每连续 4 个权重中强制恰好 3 个非零（±1），将 4 个权重打包为 5 bits（排列数 N_perm = C(4,3) × 2³ = 32，饱和 5-bit 索引），实现等效 1.25 bit/weight；(2) **Arenas（Annealing Residual Synapse）模块**：训练时注入异构梯度 Y = X·Q(W) + λ_t·X·W，其中 λ_t 为 annealing coefficient 在训练结束时退火至零，防止梯度同质化导致的 weight trapping 和表示坍缩。实验比较：(1) zero-shot 基准精度对比（PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande），baseline 包括 TWN、Spectra、BitNet、TernaryLLM、LLM-QAT、ParetoQ、Tequila；(2) CPU 推理吞吐量（tokens/s）和模型大小（MB）对比（BitNet I2_S、Tequila TL2）。

- 硬件平台是什么，配置是什么。
  训练平台：论文未明确说明 GPU 配置。推理效率评估：Intel i7-14700HX CPU，测量 tokens/s 和模型大小；AngelSlim 框架层面在 Apple M4 和 MediaTek Dimensity 9500 上额外评估了边缘设备推理效率。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3.2 1B 和 3B（BF16 baseline）。数据集与 Benchmark：PIQA、ARC-Easy（ARC-e）、ARC-Challenge（ARC-c）、HellaSwag（HelS）、WinoGrande（WinG），均为 zero-shot 评估。训练数据：论文未明确说明 Sherry 的训练数据集规模和构成。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Tencent/AngelSlim（sherry 分支），复现模型权重：https://huggingface.co/MoraxGeo/Sherry-3B-1.25bit-per-channel

  **算法 Pipeline：Sherry 1.25-bit 三值量化 QAT（结合 3:4 稀疏 + Arenas 模块）：**

  **Step 1 — 量化前向传播（带 3:4 结构化稀疏）：**
  给定权重组 W = [w₁, w₂, w₃, w₄]（连续 4 个权重为一组）：
  ```
  # 标准三值量化函数（per-weight）:
  Q(w_i) = α · sign(w_i) · I[|w_i| ≥ Δ]   # 输出 ∈ {+α, 0, -α}

  # 3:4 结构化稀疏约束（per-group of 4）:
  # 在每组 4 个权重中，保证恰好 1 个为 0，3 个非零（±1）
  # 实现方式：计算 |w_i| 的排序，将最小 |w_i| 的权重量化为 0
  for each group g of 4 weights:
      scores = [|w_1|, |w_2|, |w_3|, |w_4|]
      prune_idx = argmin(scores)
      for i in [0..3]:
          if i == prune_idx:
              Q(w_i) = 0
          else:
              Q(w_i) = α · sign(w_i)
  ```

  **Step 2 — Arenas 模块前向注入（并行于量化路径）：**
  ```
  # 量化路径（3:4 sparsified）:
  Y_q = X · Q(W)   # 矩阵乘法用三值权重 + 3:4 稀疏

  # Arenas 残差路径（注入连续潜权重的异构梯度）:
  Y_res = λ_t · X · W   # W 为全精度潜权重, λ_t 随时间退火

  # 总前向输出:
  Y = Y_q + Y_res = X · Q(W) + λ_t · X · W
  ```

  **Step 3 — 5-bit 打包与推理（training-free，仅推理时）：**
  ```
  # 每组 4 个三值权重（含 3:4 稀疏）打包为 5-bit 索引:
  # N_perm = C(4,3) × 2^3 = 4 × 8 = 32 种可能排列
  # 5 bits 恰好编码 32 种排列 → 完美饱和 5-bit 索引空间

  for each group of 4 ternary weights:
      # 编码：3 个非零位置的组合（C(4,3)=4 种）× 
      #        每个非零位置的符号（±1，2³=8 种）
      group_idx = encode_sparse_ternary(w_0, w_1, w_2, w_3)
      # group_idx ∈ [0, 31], 存储为 5-bit
      # 存储开销：4 weights / 5 bits = 1.25 bits/weight
  ```

  **Step 4 — 训练时 Arenas 退火与梯度流：**
  ```
  # λ_t 调度: 从 λ_0（初始值，论文未给出具体值）退火至 0
  # 梯度流（通过 STE）:
  ∂L/∂W = ∂L/∂Y · (λ_t + ∂Q(W)/∂W)
  # 当 λ_t > 0 时，∂L/∂X 不会坍缩为低秩（Arenas 注入全秩残差信号）
  # 当 λ_t → 0 时，模型收敛为纯三值量化+3:4 稀疏，推理零额外开销
  ```

  **关键设计对比：**
  - 2-bit 打包策略（如 BitNet）：每权重 2 bits → 浪费 0.42 bits（32/16=2x 开销），每 4 权重 8 bits
  - 1.67-bit 打包策略（如 Tequila）：3 权重打包为 5 bits → SIMD 不友好的 3-way pattern
  - Sherry 1.25-bit：4 权重打包为 5 bits → SIMD 友好的 4-way pattern，完美对齐现代 CPU 的 128/256-bit 向量寄存器
