## MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

- baseline方法是什么？
  Baseline 为 **M-SMoE**（Li et al. 2023），它从传统的"参数合并"视角出发压缩 MoE 模型。以 M-SMoE 压缩 Qwen1.5-MoE-A2.7B（60 experts → 30 experts per layer, layers 10-23）为例说明全栈执行：
  - **算法层**：M-SMoE 的核心流程：(1) 基于 expert 参数相似度聚类（如将 60 个 experts 聚为 30 组）；(2) 簇内对三个权重矩阵 W_D, W_G, W_U 分别做使用频率加权的参数平均，得到合并后的 expert 权重；(3) 路由权重取簇内原始路由权重之和。这一过程等价于 MergeMoE 框架下的 T_1 = [I; I; ...; I]（拼接单位矩阵）, T_2 = T_3 = 加权平均矩阵（式 4），但 T_1 直接从参数平均导出而非优化得到。
  - **系统框架层**：标准 PyTorch 实现，MoE layer 的 forward pass 不变（router → top-K expert selection → expert FFN → weighted sum）。合并后的模型以标准 HuggingFace 格式加载和推理。使用 DCLM 框架评估。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA）。
  - **kernel 调度层**：论文未明确说明。合并后的模型推理使用标准 PyTorch CUDA kernel，无自定义 kernel。
  - **硬件架构层**：NVIDIA H20 96GB GPU。合并和推理无特殊硬件要求。
  - M-SMoE 的核心缺陷在于 T_1, T_2, T_3 未经过量化优化——T_2/T_3 的加权平均权重基于经验选择（使用频率）而非最优性证明，T_1 直接从参数平均等价得到而非优化，导致压缩后的 expert 输出与原始 expert 输出的线性组合间存在可优化的残差。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MergeMoE 将 expert merging 重新解释为**输出合并**视角下的线性优化问题，通过理论分析 + 最小二乘法求解最优压缩矩阵，系统性减少逼近误差。全栈执行路径（以 Qwen1.5-MoE-A2.7B, 60→30 experts, layers 10-23 为例）：
  - **算法层 — 输出合并视角的优化框架**：
    1. **理论重构（§3.2）**：将 merging 过程重新表述为在前向计算中插入矩阵 A（求和矩阵，式 2）和 B（加权矩阵）及维度缩减矩阵 T_1, T_2, T_3（式 3）。合并后 expert 的输出 = W'_D T_1 (σ(T_2 W'_G X) ⊙ (T_3 W'_U X))，目标是最小化与原始 experts 输出线性组合的 Frobenius 误差。
    2. **权重最优性证明（Theorem 1）**：在假设 router logits 与 expert 输出独立的条件下，减少 experts 的 Y(BA-I_N) 误差下界，**严格证明**簇内使用相对频率 f_j / Σ f_k 作为合并权重是最优的——而 M-SMoE 仅凭经验选择此方案。
    3. **T_2/T_3 优化**：改用 W_U 和 W_G 的拼接相似度作为聚类度量（M-SMoE 用整体参数相似度），使 T_2/T_3 的加权平均在更相似的 W_G/W_U 间进行，减少非线性和 Hadamard 积引入的误差。
    4. **T_1 最小二乘优化（式 5-6）**：固定 T_2/T_3 后，通过采样输入 X̂ 计算 P = σ(T_2 W'_G X̂) ⊙ (T_3 W'_U X̂) 和 Q = σ(W'_G X̂) ⊙ (W'_U X̂)，对线性系统 T_1 P = Q 求 Moore-Penrose 伪逆闭式解 T_1 = Q P†。这是 M-SMoE 完全缺失的步骤——M-SMoE 等价于 T_1 = [I; I; ...; I]（仅做拼接，不做维度缩减优化）。
    5. **路由权重更新**：合并后路由权重 = A · 原始路由权重（与 M-SMoE 相同，因求和矩阵 A 由聚类唯一确定）。
  - **系统框架层**：基于 PyTorch，逐层反向遍历执行压缩。使用 torch hooks 获取中间激活 → GPU 内存中 BFloat16 最小二乘计算 → 释放内存。单一 H20 GPU 完成全流程，每层 <1 分钟。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。无自定义 GPU kernel。
  - **硬件架构层**：NVIDIA H20 96GB（合并用 1 卡，评估用 2 卡）。无硬件修改。
  - 对比 baseline 的改进映射（以 Qwen1.5-MoE 压缩为例，Table 2）：
    - **M-SMoE 的 T_1 未优化 → MergeMoE 的最小二乘 T_1**：M-SMoE 中 T_1 = [I; I]（无优化的拼接）→ MergeMoE 通过 QP† 在采样输入上最小化 T_1 P - Q 的残差。Table 2 结果：WinoGrande 70.48 vs 68.98 (+1.50), Hellaswag 71.58 vs 68.87 (+2.71), SQuAD 56.40 vs 54.99 (+1.41), MRPC 74.75 vs 72.30 (+2.45)，在所有 benchmark 上一致优于 M-SMoE。
    - **M-SMoE 的权重选择仅凭经验 → MergeMoE 的理论最优性证明（Theorem 1）**：同等聚类条件下，使用频率加权被证明是误差下界的最优解，使权重分配有理论保证。
    - **M-SMoE 的聚类度量不够精细 → MergeMoE 的 W_U||W_G 拼接距离**：聚类时仅关注与 T_2/T_3 直接相关的 W_U 和 W_G，减少 T_2/T_3 加权平均在非线性激活 σ 和 Hadamard 积 ⊙ 处引入的误差。消融实验（Table 5, w/o merging errors）验证了聚类误差和合并误差的分离。
    - **输入样本的敏感性**：MergeMoE 的最小二乘法存在样本数临界阈值（~32），低于阈值时性能崩溃（random guessing），但高于阈值后性能持续提升（Figure 4）。跨数据集泛化实验（Table 4）表明即使使用单一数据集采样，性能下降也很小（如 Hellaswag: 71.56 self-sourced vs 71.58，仅差 0.02）。
  - 局限性：MergeMoE 比 M-SMoE 慢（因最小二乘法），且在极低样本量下性能崩溃，需要保证足够的输入样本量。
