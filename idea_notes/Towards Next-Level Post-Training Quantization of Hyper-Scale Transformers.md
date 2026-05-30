## Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

- baseline方法是什么？
  Baseline 是经典的 block-wise PTQ 方法（BRECQ）和 layer-wise PTQ 方法（AdaRound/OPTQ/Z-FOLD）。
  
  **BRECQ**（block-wise）：将 Transformer block 内的所有层（Q、K、V、O、FFN）联合量化，最小化整个 block 输出重构误差 `E[||f(W_Q,W_K,W_V,...)(X) - f(W_Q,W_K,W_V,...)(X)||^2]`，使用学习方式优化 weight-rounding policy。全栈执行示例：给定一批校准数据 X（B×L×d），BRECQ 对 transformer block 中所有 linear 层同时量化 → 每轮迭代：forward W_Q→Q、W_K→K、W_V→V → 计算 attention `softmax(QK^T/√d)V` → 通过 O 投影和 FFN → 计算与全精度 block 输出的重构误差 → 反向传播更新所有层的量化参数/rounding policy → 重复直至收敛。**缺陷**：(1) 时间复杂度 O(B·L·d_h·max(d, L))，每轮需完整 attention forward pass，OPT-2.7B 需 20+ GPU 小时；(2) 对大模型（≥6.7B）OOM 不可运行；(3) 超参数敏感，对 LLM 未优化。
  
  **AdaRound/OPTQ**（layer-wise）：逐层独立量化，最小化每层输出误差 `E[||Q(W)X - WX||^2]`，Hessian 固定为 `H = 2E[XX^T]`。全栈执行示例：AdaRound 逐层遍历 Transformer → 对每个 linear 层：计算 H=2E[XX^T] → 优化 rounding policy V 最小化 `||WX - W̃X||^2` + rounding regularization → 输出量化后层权重 → 下一层用上层的量化输出作为输入 → 逐层累积误差。**缺陷**：(1) Hessian 仅基于 X 独立计算，未考虑 attention 内部 Q/K/V 之间的跨层依赖；(2) layer-wise 重构目标忽略了 attention output 的整体误差传播；(3) 低比特（INT2）下性能急剧退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  *aespa* 提出 "逐层量化 + attention-wise 重构" 的折中策略：每层单独量化（保留 layer-wise 效率），但损失函数以 attention 输出重构为目标（引入跨层依赖）。
  
  全栈执行示例（对应 paper Algorithm 1 + Table 4）：给定校准数据 X（128 segments × 2048 tokens from C4），aespa 对每个 Transformer block 执行：
  - **算法pipeline层**：先全精度前向一次预计算关键统计量 → `H_xx = E[XX^T]`（d×d）、`H_v = E[X A^T A X^T]`（d×d，含 attention map 信息）、`E[K^TK]`（d_h×d_h）、`E[Q^TQ]`（d_h×d_h）→ 所有后续迭代无需再执行 attention forward。
  - 量化 W_V：用 Z-FOLD 基于 `H_v` 初始化 scale/zero-point → 每轮用 OPTQ 初始 round 或 AdaRound 优化时，直接计算 `loss = tr(ΔW_V·H_v·ΔW_V^T)`（一次矩阵乘+逐元素乘），无需 forward pass。
  - 量化 W_Q：用 Z-FOLD 基于 `H_xx` 初始化 → 每轮 AdaRound 优化：`loss = tr(E[K^TK]·ΔW_Q·H_xx·ΔW_Q^T)`（两次矩阵乘+逐元素乘）。关键：E[K^TK] 引入了 key projection 的信息（跨层依赖），但 K 是固定全精度的（单独量化策略保证）。
  - 量化 W_K：同理，`loss = tr(E[Q^TQ]·ΔW_K·H_xx·ΔW_K^T)`，E[Q^TQ] 引入 query 的信息。
  - 量化 FFN/O-proj：使用 standard layer-wise 目标 `loss = tr(ΔW·H_xx·ΔW^T)`。
  - **kernel调度层**：论文未明确说明（纯算法层方案，kernel 为 PyTorch 标准 matmul）。
  - **Serving调度/编译框架/硬件架构/芯片设计层**：论文未明确说明。
  
  对比 baseline 的关键改进：
  1. **跨层依赖建模**：baseline Hessian `H=2E[XX^T]` 将 Q/K/V 视为独立；aespa 的 `H_v=2E[XA^TAX^T]` 通过 attention map A 直接将 Q 和 K 的信息耦合进 V 的 Hessian，且 W_Q/W_K 的损失函数通过 `E[K^TK]` 和 `E[Q^TQ]` 注入跨投影依赖。
  2. **预计算加速**：baseline 每轮需 O(B·L·d_h·max(d,L)) 的 attention forward；aespa 通过预计算统计量，每轮仅 O(d_h d^2)，且与校准数据量无关。OPT-125M 上 FLOPs 差 28 倍（0.24 vs 6.7 GFLOPS）。
  3. **全数据集梯度估计**：预计算使单次 loss 计算等价于在整个校准集上评估（batch size = 全部 128 segments），梯度估计更准确，收敛更快（2000 轮迭代即可）。
  4. **单独量化可行性验证**（Table 5）：虽然逐层量化，但 attention-wise 重构目标使性能接近 block-wise 联合量化（BRECQ），INT3/4 下几乎无损。
