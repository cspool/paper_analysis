## Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations

- baseline方法是什么？
  - **NF4（NormalFloat 4-bit）** [QLoRA, Dettmers et al. 2023]：假设网络权重服从 N(0,σ²)，基于 Gaussian 分位数构建 16 个 reconstruction level 的固定码本，声称每个码本点等概率使用（信息论最优）。固定 -1, 0, 1 三个 reconstruction level，用于 block-wise absmax 量化。
  - **AF4（AbnormalFloat 4-bit）** [Yoshida 2023]：分析归一化权重分布对 block size 的依赖，直接最小化归一化权重的 MAE 来获得码本。也固定 -1, 0, 1 三个 level。
  - **Baseline 全栈执行例子**（以 Llama-3.1-8B one token 推理为例）：
    - **算法pipeline**：预训练权重 W → 按 block size I=64 分块 → 每块除以 `w_b^max = max |w_{b,i}|` 归一化到 [-1,1] → NF4/AF4 码本 scalar 量化 → 存储量化索引（4-bit per weight）+ 量化常数 `w_b^max`（BF16）。解码时 `Ŵ_{b,i} = w_b^max * x̂(index)`。
    - **系统框架**：使用 PyTorch + QLoRA 框架（HuggingFace PEFT + bitsandbytes），在 HuggingFace Transformers 模型上加载量化权重。推理时 fused kernel 从 4-bit 索引查表解码为 BF16 进行 forward pass，或直接在量化域执行 LoRA 微调。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。4-bit 解码通过 bitsandbytes 中的 CUDA kernel 实现（dequantize + FP16/BF16 GEMM）。
    - **硬件架构**：论文未涉及硬件架构设计，在 NVIDIA GPU（A100/RTX 3080）上运行。
  - **Baseline 的核心缺陷**：
    1. **归一化权重量化误差不等于端到端权重误差**：NF4 基于 Gaussian 分位数等概率假设（错误），AF4 最小化归一化权重的 MAE，但真正的目标是 `MAE(W, Q(W))`。归一化权重 X 的每个样本在反向缩放时乘以不同的 `w_b^max`，对最终 errors 的贡献不同。大 `w_b^max` 的 block 中的量化误差被放大，但 NF4/AF4 未考虑这一点。
    2. **绝对值归一化浪费一个重建层级**：对于 block-wise absmax normalization，每个 block 实际上只包含 -1 或 +1 中的一个端点，但 NF4/AF4 固定了两个端点 (-1 和 1)，导致一个 layer 被浪费。
    3. **Outlier 破坏归一化分布假设**：少量 outlier weight 导致其所在 block 的 `w_b^max` 异常大，归一化后非 outlier 权重被过度压缩到零附近（underrange），量化器在次优的 rate-distortion 区间运作。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **BOF4/BOF4-S**：推导考虑 `w_b^max` 分布的 EM 算法（modified Lloyd's），最小化真正的端到端量化误差 `MSE(W, Q(W))` 或 `MAE(W, Q(W))`。
    - **解决缺陷 1**：centroid 公式引入 `w_b^max` 的分布权重。MSE centroid 是 `w_b^max` 平方加权的均值（Eq. 6），MAE centroid 是 `w_b^max` 加权的中位数（Eq. 8）。直观理解：block max 大的 block，其归一化权重在重建层级更新中贡献更大权重。
    - **解决缺陷 2**：BOF4-S 用 signed absmax normalization 替换 absmax normalization。归一化后只需固定 1 个端点（`x̂(16)=1`），且归一化权重分布只在 x=1 有离散概率 `1/I`。释放了一个 reconstruction level 给中间区域使用，降低整体量化误差。
  - **OPQ**：outlier 混合精度策略。
    - **解决缺陷 3**：将 outlier（`|w_{b,i}| > σ_b * F_M^{-1}(0.95)`）单独以 BF16+position 存储，替换为零。归一化时 outlier 不计入 `w_b^max` 的计算，使归一化后权重分布与理论 `p_X^cont` 高度吻合，量化器在最优设计点工作（而非 underrange 区间）。
  - **论文方法全栈执行例子**（以 Llama-3.1-8B one token 推理为例）：
    - **算法pipeline**：
      1. 预训练权重 W → 按 I=64 分块
      2. OPQ（可选）：每 block 计算 σ_b，检测 `|w| > σ_b * F_M^{-1}(0.95)` 的 outlier → 存储为 BF16 + 64-bit index → 替换为 0
      3. Block-wise signed absmax normalization：`w_b^max = w_{b, argmax|w|}` → `x_{b,i} = w_{b,i}/w_b^max`
      4. BOF4-S(MSE) 查表量化：`Ŵ_{b,i} = w_b^max * x̂(codebook_index)`
      5. 存储：4-bit 索引 × |W| + BF16 量化常数（B 个）+ OPQ outlier（BF16 + 64-bit position，约 0.96% 额外内存 at I=64/q=0.95）
    - **系统框架**：基于 QLoRA 框架（HuggingFace PEFT + 自定义 BOF4 量化后端替代 bitsandbytes NF4）。同 baseline 使用 PyTorch Transformers，仅替换码本和归一化方式。推理时解码流程与 NF4 一致（查表 dequantize → BF16 GEMM），无额外 kernel 修改。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。解码 4-bit 索引通过自定义 CUDA kernel（复用 bitsandbytes 框架结构），OPQ 在推理时仅需根据 position index 将 BF16 outlier 写回对应位置，开销极小（RTX 4070 Ti Super 上生成 1000 tokens 的额外耗时见图 11，随 I 增大递减）。
    - **硬件架构**：论文未涉及硬件架构设计。
