## RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

- baseline方法是什么？
  - Baseline 方法分为两类：(a) **先 SFT 再 PTQ 的两阶段 pipeline**：先用全精度（FP16/BF16）对预训练 LLM 做 SFT，然后用 PTQ 方法（RTN、GPTQ、QuaRot、SpinQuant）将 SFT 后的模型量化为 W4A4KV4。代表性流程：预训练 Llama 3.1 8B → FP16 SFT 训练（Tulu 3, 2 epoch, AdamW lr=5e-6）→ QuaRot PTQ（Walsh-Hadamard rotation + GPTQ-style weight calibration）或 SpinQuant PTQ（learned rotation + calibration），共需 2.1→1.3h 训练。(b) **纯 QAT 方法**（STE only, 不含 rotation）：直接将 STE 应用于 SFT loss，在 4-bit 量化约束下用梯度近似训练模型参数，不引入旋转矩阵来消除 outlier。
  - Baseline 的全栈执行例子（以 Llama 3.1 8B SFT→QuaRot W4A4KV4 两阶段 pipeline 为例）：
    - **算法pipeline**：预训练 Llama 3.1 8B → FP16 SFT（Tulu 3, AdamW, lr=5e-6, 2 epoch, 100k samples, cos schedule, 8×A100）→ QuaRot PTQ（对所有线性层插入 Walsh-Hadamard 旋转矩阵，吸收 normalization 参数，128 样本校准，逐层 GPTQ-style weight 最优量化参数搜索 + uniform activation/KV cache 量化）。**核心缺陷 1**：SFT 和量化分离导致次优结果——SFT 阶段优化的全精度权重在后续量化时产生严重精度损失（activations 中存在 outlier 值撑大量化范围，增加量化误差），Table 2 中 SFT→QuaRot W4A4KV4 avg=28.46 vs FP SFT avg=42.16（-13.70 gap）。**核心缺陷 2**：STE without rotation 在 4-bit 激活量化时性能严重退化（Table 2 STE avg=17.14），因为激活 outlier 导致 STE 梯度偏差过大（Theorem 4.3 证明预测误差正比于 weight quantization error 的加权和）。
    - **系统框架**：PyTorch 训练 + QuaRot 的 Hadamard CUDA kernel for online rotation。论文未修改 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：使用 fast Hadamard CUDA kernel（来自 QuaRot/QuIP#）处理在线旋转矩阵乘法。量化 matmul 使用标准 PyTorch 模拟量化（fake quantization），无 custom INT4 kernel。论文未说明具体的 GPU kernel 级优化。
    - **硬件架构**：8× NVIDIA A100 GPUs。SFT→QuaRot 训练时间 2.1h (FP SFT) + ~0h (QuaRot PTQ 几乎无训练开销) = 2.1h；peak memory 300GB (SFT) + 0 (PTQ calibration negligible)。但 W4A4KV4 精度损失严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：RoSTE 提出 **QA-SFT（Quantization-Aware Supervised Fine-Tuning）**，在单一训练阶段同时完成量化和微调，核心通过 **Bilevel Optimization + Adaptive Rotation** 解决两个 baseline 缺陷：
    - **对应缺陷 1（两阶段分离导致次优）**：将 SFT 和量化合并为单一优化问题（Formula 8），直接优化 `min_{W,R} L_SFT(m_Q(·; W,R)) s.t. R R^T = I`。简化为 bilevel formulation（Formula 11）：上层 STE 优化量化权重矩阵 W（SFT objective），下层选择旋转矩阵 R（quantization error surrogate loss E(12)）。训练中交替执行 rotation configuration search 和 QAT via STE，使量化误差在训练过程中持续被优化，而非固定于 PTQ 校准时刻。
    - **对应缺陷 2（激活 outlier 导致 STE 性能退化）**：通过 adaptive Walsh-Hadamard rotation 消除激活 outlier（Fig. 3 显示 RoSTE 训练收敛后无激活 outlier，而 STE 仍存在大量 outlier）。Proposition 4.4 证明旋转后的 weight quantization error 从 `O(d·max_i w_i²)` 降至 `O(‖w‖²)`（w.h.p.），将 outlier 主导的量化误差转化为均匀化误差。自适应策略（逐层在 I vs H 之间选择）避免"全部旋转"在某些层引入新异常值的问题（Table 3：No Rotation ROUGE=22.37, Complete Rotation ROUGE=13.09, RoSTE Adaptive=23.07）。
  - 全栈执行例子（RoSTE W4A4KV4 on Llama 3.1 8B, 8×A100）：
    - **算法pipeline**：预训练 Llama 3.1 8B → 修改 normalization layers（吸收 LayerNorm/RMSNorm 参数）→ one-shot rotation configuration search（逐层比较 W4A4 quantization error with/without Walsh-Hadamard rotation, 128 校准样本）→ QA-SFT training（STE + adaptive rotation, AdamW lr sweep {5e-6,1e-6,5e-7}, 2 epoch, 100k Tulu 3 samples, 8×A100, gradient accumulation=16, max seq len=1024, W4A4KV4 asymmetric uniform quantizer, per-token activation + per-channel weight quantization groups, clipping factor sweep {1, 0.95, 0.9}）→ 合并 offline rotations 到权重，保留 online rotations 在 fast Hadamard kernel。训练过程中，每层 linear layer forward: `X_rot = Q_x(X R_i)` (online rotation via Hadamard kernel if R_i not merged), `W_rot = Q_w(R_i^T W_i)`, `output = X_rot · W_rot` (INT4 matmul simulation)。Backward via STE: `∂L/∂W_i ≈ (R_i)^T · gradient_from_upper_layer`。效果：W4A4KV4 avg=31.69（vs best baseline SpinQuant avg=29.13, +2.56），W4A8KV4 avg=37.70（vs best baseline SpinQuant avg=35.02, +2.68），训练开销仅比 STE 多 0.4h（2.8h vs 2.4h, Table 10）。
    - **系统框架**：PyTorch 实现，无 Serving 框架修改。论文未说明具体的训练框架（如 FSDP/DeepSpeed 等）。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：使用 fast Hadamard CUDA kernel（继承自 QuaRot/QuIP#）处理在线旋转矩阵 R_3, R_3^T, R_4 的矩阵乘法，开销可忽略（训练时间 2.8h vs STE 2.4h，仅 +16.7%）。论文未引入新的自定义 kernel。
    - **硬件架构**：8× NVIDIA A100 GPUs。RoSTE training time 2.8h, peak memory 318GB（与 STE 317GB 几乎相同, Table 10）。总成本：2.8h × 8 A100 ≈ 22.4 GPU-hours。
