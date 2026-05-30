## KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

- baseline方法是什么？
  - **Direct VQ（直接向量量化）**：将 VQ（如 GPTVQ、VPTQ、PCDVQ）直接应用于 MoE LLM 的 expert 权重，clustering 权重子向量到共享 codebook。不区分 expert 间的共享/特异性结构，同等对待所有 expert 的权重向量。
  - **Scalar Quantization（RTN, GPTQ）**：逐权重独立量化，在 ≤3 bit 下 representational capability 急剧下降。
  - **MoEQuant**：使用 routing statistics 平衡各 expert 在校准中的贡献，但 ≤4 bit 下性能不满意。
  - **Baseline 全栈执行例子（以 Qwen1.5-MoE-A2.7B 一个 token 推理为例）**：
    - **算法pipeline**：输入 token x → MoE layer gating 选择 top-k expert → 每个 expert MLP 执行 `y_i = W_i x`（W_i 已通过 Direct VQ 量化：将 W_i 按 d=4 分块 → K-means 训练 codebook → 每个子向量存储 codebook index + 共享 codebook）→ 加权聚合 `y = Σ g_i y_i`。Direct VQ 将所有 expert 的权重子向量同等对待，不区分共享和特异性方向。
    - **系统框架**：PyTorch + HuggingFace Transformers 推理管线。量化权重通过 codebook 查表解码为 FP16 进行 forward pass。使用 LM-Evaluation-Harness 评测。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。VQ 解码通过查表 dequantize → FP16 GEMM 实现。
    - **硬件架构**：论文未涉及硬件架构设计，在 NVIDIA RTX A6000 GPU 上运行。
  - **Baseline 的核心缺陷**：
    1. **Expert 间冗余浪费 codebook 容量**：MoE expert 常捕获相似特征模式，权重存在大量跨 expert 冗余。同一层内不同 expert 对相同输入产生高度相似的输出（Fig. 2a）。Direct VQ 将每个 expert 独立量化，导致有限 codebook 资源重复编码相似表示，无法集中编码 expert 的差异化（特异性）信息。
    2. **量化误差经 expert 聚合放大**：量化误差在各层累积产生 biased layer outputs。MoE 架构中多个 expert 的输出通过 gating weights 加权求和，biased outputs 被聚合放大（而非像 dense LLM 中仅线性累积），导致更严重的 distributional shift（Fig. 3 显示 Direct VQ 后 per-channel mean 和 variance 显著偏离 FP16）。
    3. **SQ 在超低比特下的表示瓶颈**：≤3 bit 时 scalar quantization 的离散表示能力不足以覆盖 MoE 大量参数的分布。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **KBVQ-MoE = IDRE + BCOS**：
    - **IDRE（Input-driven Redundancy Elimination）**：
      - **解决缺陷 1**：通过 KLT 将 expert 权重对齐到输入统计方向（而非仅考虑权重自身结构），构建 unified representation `W̄`。然后 SVD 提取 top-k 主导共享分量 `W_share`（保留全精度），将剩余 expert-specific 分量 `W_quant` 做 VQ 量化。KLT 确保提取的共享方向同时是"输入高能量方向"和"跨 expert 高使用率方向"。实验显示 IDRE 后不同 expert 的输出相似度显著降低（Fig. 2b vs 2a），验证了冗余消除有效性。KLT 带来的收益 vs 纯 SVD：WikiText2 perplexity 降低 2+ 点（Table 3）。
    - **BCOS（Bias-Corrected Output Stabilization）**：
      - **解决缺陷 2**：仅对 expert-specific 分量 `W_quant` 做 VQ 量化（共享分量保持全精度，不引入误差），然后通过 channel-wise affine compensation（`s_j = σ_{y_j}/σ_{ŷ_j} - 1`, `b_j = μ_{y_j} - (1+s_j)μ_{ŷ_j}`）校正量化输出。该校正基于 MMSE 准则的闭式最优解（Appendix A.4 证明），使每个 channel 的 mean/variance 与 FP16 严格对齐，消除 distributional shift（Fig. 3 中 KBVQ-MoE 的 mean/variance 与 FP 高度一致）。BCOS 仅引入 2·oc 个参数/层，推理 FLOPs 增加 <0.1%。
  - **论文方法全栈执行例子（以 Qwen1.5-MoE-A2.7B 一个 token 推理为例）**：
    - **算法pipeline**：
      1. **离线校准阶段**：从 RedPajama 采样 256 条校准数据（seq len=4096）→ 逐 MoE layer 收集输入激活 X → 计算 `C_X = X^T X / (B-1)` → KLT 特征分解得 `U_X = U_KLT Λ_KLT^{1/2}` → 各 expert 权重右乘 `U_X` 投影到输入相干空间 → 堆叠所有 expert 的 `W̃^(i)` 成 `W̄` → SVD 截断取 top-k（`k = ic/128`）得 `U_share` 和 `V_k^(i)` → 计算 `W_share^(i)` 和 `W_quant^(i) = W^(i) - W_share^(i)` → 对 `W_quant^(i)` 做 K-means VQ（d=4, 100 iters）训练 codebook → 用 calibration 数据估计 per-channel `μ_y, σ_y, μ_ŷ, σ_ŷ` → 计算 BCOS 参数 `(s, b)`。
      2. **推理阶段**：输入 token x → MoE gating 选 top-k expert → 对每个选中 expert：`y_corr = (1+s) ⊙ ((W_share + W_quant,VQ) x) + b` → 加权聚合。W_share 以 FP16 存储和计算，W_quant,VQ 通过 codebook index 查表解码为 FP16 再做 MatVec。
    - **系统框架**：基于 PyTorch 推理管线。量化权重由 `W_share`（FP16 低秩矩阵）+ `W_quant,VQ`（codebook index + shared codebook）+ `(s, b)`（FP16 per-channel）三部分组成。推理时：index lookup → dequantize → FP16 MatVec → per-channel affine `(1+s)⊙result + b`。评测使用 LM-Evaluation-Harness。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。量化权重解码和校正均为标准 PyTorch 操作。解码速度测试显示 2-bit KBVQ-MoE 实现 1.58× 加速（vs BF16），推理 overhead <0.1% FLOPs。
    - **硬件架构**：论文未涉及硬件架构设计。在 NVIDIA RTX A6000 上完成量化；A100 上完成 MoE 压缩方法对比实验。
  - **关键设计选择**：
    - 为什么 KLT 在 SVD 之前？KLT 使 SVD 的 Gram 矩阵 `S = W̄^T W̄` 的频谱同时反映输入能量（`Λ_X`）和跨 expert 权重能量（`Σ_i W^(i)T W^(i)`），确保提取的共享方向在输入高能量方向上有更大保留。纯 SVD 仅考虑权重结构，忽略输入统计。
    - 为什么 k = ic/128？MoE expert 融合后的 Gram 矩阵 S 呈强低秩性（附录 Fig. 5 显示奇异值快速衰减），功率律近似 `σ_j² ∝ j^{-α}`。k=ic/128 时 ρ_k ≈ 0.6-0.8，继续增大 k 边际收益递减但存储开销线性增加（Table 4, Table 7）。
