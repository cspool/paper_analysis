## QERA: an Analytical Framework for Quantization Error Reconstruction

- baseline方法是什么？
  - Baseline 方法：LoftQ（QPEFT 场景）和 ZeroQuant-V2（PTQ 场景）——两者都使用截断 SVD 对权重量化误差 (W - W̃) 进行低秩近似来重建量化误差：C_k = SVD_k(W - W̃)。这种方法最小化的是权重逼近误差 ||W - W̃ - C_k||_F，即 Frobenius 范数下的最优低秩逼近（Eckart-Young-Mirsky 定理）。LoftQ 在此基础上增加了迭代优化（Algorithm 1），交替更新量化权重和低秩项。LQ-LoRA 进一步引入启发式的行列缩放矩阵 D_row, D_col。
  - 全栈执行例子（Baseline: ZeroQuant-V2 / LoftQ on LLaMA-2-7B, 4-bit, rank=32）：
    - **算法pipeline**：对每个线性层权重 W ∈ R^{m×n}，先量化 W_q = q(W)（如 MXINT block size=32），反量化 W̃ = dq(W_q) → 计算权重量化误差 E = W - W̃ → 对 E 做截断 SVD：U, Σ, V^T = SVD(E)，取前 k 个奇异值/向量 → A_k = U_{:,:k}√Σ_{:k,:k}, B_k = √Σ_{:k,:k}V_{:k,:}^T → 前向传播 y = x(W̃ + A_k B_k) → 推理时合并 C_k = A_k B_k 不增推理开销。LoftQ 迭代 T=5 次，每轮用 W - A_k B_k 替代 W 重新量化。关键缺陷：最小化权重误差不等于最小化输出误差。实验证明（Figure 1），LoftQ 迭代数增加时所有层权重误差单调降，但模型输出误差不一定降甚至可能升（如 LoftQ 5-iter vs 3-iter 在 rank k=8 时输出误差更大）；rank 增加也不保证输出误差单调降（如 rank 16 输出误差 > rank 4）。
    - **系统框架**：PyTorch + HuggingFace Transformers + PEFT，GPU 上执行。LoftQ 迭代需反复量化-反量化-SVD，计算开销随迭代数线性增长。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。使用标准 PyTorch CUDA 矩阵乘法和 SVD 算子。
    - **硬件架构**：NVIDIA A100 80GB / A6000 48GB GPU。无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QERA 重新审视 QER 问题的优化目标，指出应最小化层输出误差（Problem 2: min E[||x(W̃ + C_k) - xW||₂²]）而非权重误差（Problem 1: min ||W - W̃ - C_k||_F）。通过严格的数学推导（Theorem 1 和 Theorem 2），给出两个闭式解：(1) QERA-exact：使用输入自相关矩阵 R_{XX} = E[x^T x] 的矩阵平方根对标度化后的权重量化误差做 SVD 再反标度化；(2) QERA-approx：在"不同嵌入维度不相关"假设下，将 R_{XX} 简化为对角矩阵 S²，大幅降低计算开销。两个解都对任意量化函数 q(·) 适用，且 QERA-approx 从理论上解释了 LQER 启发式标度的成功和失败原因。
  - 全栈执行例子（QERA on LLaMA-2-7B, 4-bit, rank=32）：
    - **算法pipeline**：
      1. 校准阶段：对校准集 X 中所有输入 x ∈ R^{1×m} 累积计算：(a) QERA-approx: s_sq[i] = E[x_i²]，构建对角标度矩阵 S = diag(√E[x₁²], ..., √E[x_m²])；(b) QERA-exact: R_{XX} = E[x^T x] ∈ R^{m×m}（FP64 精度累积外积），计算矩阵平方根 R_{XX}^{1/2}（blocked Schur algorithm, CPU, SciPy）。
      2. 量化权重：W_q = q(W), W̃ = dq(W_q)。
      3. 标度化误差：Q = S(W - W̃)（approx）或 Q = R_{XX}^{1/2}(W - W̃)（exact）。
      4. 截断 SVD：U, Σ, V^T = SVD(Q)，取前 k 个分量。
      5. 反标度化：A_k = S^{-1}U_{:,:k}（approx）或 A_k = (R_{XX}^{1/2})^{-1}U_{:,:k}（exact），B_k = Σ_{:k,:k}V_{:k,:}^T。
      6. 前向/推理：y = x(W̃ + A_k B_k)，低秩项可预合并。
      7. 结果：相比 ZeroQuant-V2，4-bit LLaMA-3.1-70B 上 6 个下游任务平均 Δacc = +2.97%，WikiText2 Δppl = -0.38；3-bit LLaMA-2-7B 上 Δppl = -2.65（ZeroQuant-V2 13.00 → QERA-exact 10.67）。QPEFT 中 2-bit RoBERTa-base @ GLUE Δacc = +6.05% vs LoftQ；微调收敛速度更快（Figure 2）。QERA 的模型输出误差随 rank 单调递减（而 LoftQ 不单调），且更多校准样本一致提升 QERA 性能（而 LQER 随机波动）。
    - **系统框架**：PyTorch + Transformers + PEFT + Accelerate（训练/微调），SciPy blocked Schur（矩阵平方根），lm-evaluation-harness（评测）。矩阵平方根计算在 CPU 上执行（FP64），GPU 加速矩阵平方根将是未来优化方向。QERA-approx 初始化时间约 21s-30min（模型规模相关），QERA-exact 约 1.6min-4.9h。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。推理时 QERA 的 A_k, B_k 预合并入 W̃，与 baseline 使用相同的矩阵乘法 kernel，无额外运行时开销。
    - **硬件架构**：NVIDIA A100 80GB GPU × 4（QPEFT）/ A6000 48GB GPU × 8（PTQ），AMD EPYC CPU。无定制硬件。
  - **Baseline 缺陷 → 方法设计映射**：
    - (i) Baseline 最小化权重误差 ||W - W̃ - C_k||_F（Problem 1），但不能保证降低模型输出误差 → QERA 重新定义优化目标为最小化层输出误差 E[||x(W̃ + C_k) - xW||²]（Problem 2），并严格推导出闭式解。
    - (ii) LoftQ 迭代增加和 rank 增加不保证输出误差单调降（Figure 1）→ QERA 的模型输出误差随 rank 单调递减，消除了迭代的不确定性，无需迭代算法。
    - (iii) LQER 启发式标度 S 导致校准样本数增加时模型性能随机波动（Figure 3 purple curve）→ QERA-approx 从理论上推导出正确的 S = diag(√E[x_i²])（而非 LQER 的 E[|x_i|]），使更多校准样本一致提升性能直至收敛（Figure 3 green curve）。
    - (iv) Baseline 在低比特（2/3-bit）下精度崩溃（如 QLoRA 2-bit CoLA Matt=0, LoftQ=3.43）→ QERA-exact 使用完整的 R_{XX} 信息，在 2-bit RoBERTa @ CoLA 上达到 Matt=26.43，3-bit LLaMA-2-7B QERA-exact PPL=10.67 vs ZeroQuant-V2 13.00。
