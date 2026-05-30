## FlatQuant: Flatness Matters for LLM Quantization

- baseline方法是什么？
  Baseline 是 **QuaRot**（Ashkboos et al. 2024）和 **SpinQuant**（Liu et al. 2024c），两者均使用 Hadamard 变换（或学习到的正交旋转矩阵）对权重和激活做预量化变换以消除离群值。此外 per-channel scaling（SmoothQuant, OmniQuant）也作为参考 baseline。

  **QuaRot 全栈执行例子**（LLaMA-2-7B, W4A4, RTX 3090）：

  - **算法pipeline**：校准数据 X → 将 LayerNorm 替换为 RMSNorm → 在模型权重上应用固定 Hadamard 矩阵 H（离线融合到权重中：W' = H^T W H）→ 在线推理时对激活做 Hadamard 变换 X' = X H → per-token/per-channel 对称量化到 INT4 → INT4 GEMM（CUTLASS kernel）。Hadamard 矩阵 H ∈ {+1,-1}^{n×n} 对所有层复用，不考虑逐层特性差异。
  - **系统框架**：基于 HuggingFace + PyTorch 的 PTQ 脚本，无 Serving 框架修改。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：CUTLASS INT4 matmul kernel + FlashInfer KV cache 量化。Hadamard 变换在线计算使用 PyTorch matmul（3 次在线变换：P_a, P_o, P_ug），在线变换带来约 0.26× 端到端减速。
  - **硬件架构**：NVIDIA RTX 3090 GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **Hadamard 变换不考虑逐层特性**：Hadamard 矩阵全局复用，无法针对每个线性层的独特权重和激活分布模式做自适应调整。结果：某些层的权重或激活仍呈现陡峭分散分布（steep and dispersed distributions），残留离群通道。
  2. **变换后平坦度仍然不足**：per-channel scaling 仅调整对角线元素（diag(c)），以牺牲权重量化质量为代价平滑激活；Hadamard 变换虽在通道间重新分配离群值，但对 pivot tokens（前几个 token）的大量离群值无能为力，量化误差在初始 token 和深层累积严重。
  3. **修改 LayerNorm 为 RMSNorm 限制灵活性**：QuaRot 将 LayerNorm 改为 RMSNorm 并将正交变换融合到前层，但 pre-norm 架构的残差连接迫使所有 block 共享同一变换，限制了逐层表达能力。
  4. **在线变换开销较大**：Hadamard 变换为全尺寸矩阵乘法，带来约 0.26× 额外减速。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **FLATQUANT（Fast and Learnable Affine Transformation）**，通过可学习仿射变换增强权重和激活的平坦度：

  **(1) Kronecker 可学习仿射变换替代固定 Hadamard 变换**
  - 对每个线性层学习独立的仿射变换 P = P₁ ⊗ P₂（Kronecker 乘积），而非全局复用 Hadamard H。
  - 内存节省 n/2 倍（取 n₁=n₂=√n 时最优），计算节省 √n/2 倍。LLaMA-2-7B 仅 2.61% FLOPs 开销 + 3.41MB 额外内存。
  - **解决缺陷 1**：逐层定制变换矩阵，通过 MSE 损失（Eq.4）直接优化量化输出保真度，自动适配各层权重/激活分布特征。

  **(2) 可学习逐通道缩放 + 可学习裁剪阈值增强平坦度**
  - 在仿射变换前添加 diag(c) 缩放（可融合到前层消除开销），变换后应用 sigmoid 后的裁剪阈值 α_w, α_a。
  - **解决缺陷 2**：仿射变换先将离群值在通道间重新分配（平滑 pivot tokens），再通过裁剪阈值去除剩余极端值。消融实验证明各组件叠加有效（PPL: RTN baseline 1266.60 → +LT 8.50 → +PS 7.95 → +LCT 6.98）。

  **(3) 保留原始 LayerNorm 保持架构灵活性**
  - 不修改 LayerNorm 为 RMSNorm，保留原始架构。
  - **解决缺陷 3**：各 Transformer block 可独立学习不同的仿射变换 P，不受残差连接约束，提升 expressiveness。

  **(4) Triton 融合 kernel 消除在线变换开销**
  - 将 Q(P₁^T ×₁ X̃ ×₂ P₂) 融合为单 Triton kernel，所有中间结果保持在 SRAM 内。
  - **解决缺陷 4**：kernel 融合后 5 个在线变换仅带来 0.07× 端到端减速（vs QuaRot 的 0.26× 仅 3 个变换），prefill 2.30×/decode 1.76× vs FP16。

  **全栈执行对比（LLaMA-2-7B, W4A4, RTX 3090）**：

  - **算法pipeline**：校准数据（128 segments WikiText-2, 2048 tokens）→ 逐 block 训练 Θ={P₁,P₂,c,α_a,α_w}（AdamW, MSE loss, 15 epochs）→ P^{-1} 融合到权重离线预计算 → 在线推理：X̃ = reshape(X)→ X' = P₁^T X̃ P₂（融合 kernel + 即时量化）→ INT4 GEMM（CUTLASS）。与 QuaRot 的固定 H 相比，FLATQUANT 的逐层 P 通过梯度下降直接最小化量化 MSE。
  - **系统框架**：同 QuaRot，基于 HuggingFace + PyTorch PTQ 脚本。量化后权重和变换矩阵保存为模型文件，推理时加载。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：Triton 融合 kernel（仿射变换 + 量化 单 kernel）+ CUTLASS INT4 matmul + FlashInfer KV cache。关键差异：QuaRot 的 3 个 Hadamard 在线变换（matmul）带来 0.26× 减速，FLATQUANT 的 5 个仿射在线变换（融合 kernel）仅带来 0.07× 减速。
  - **硬件架构**：NVIDIA RTX 3090 GPU，无自定义硬件。

  **关键结果**：
  - LLaMA-3-70B W4A4 RTN：首次 ≤1% 准确率下降（Avg 79.01 vs FP16 79.95），超越 SpinQuant 7.5%
  - LLaMA-3-8B W4A4 RTN WikiText-2 PPL：6.98 vs SpinQuant 7.96（↓12.3%）
  - LLaMA-3-8B W3A3KV3：PPL 10.82 vs QuaRot 686.54（极端低比特场景优势巨大）
  - DeepSeek-R1 W4A4：AIME2024 73.3（接近 FP8 的 79.8）
