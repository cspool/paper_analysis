## Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

- baseline方法是什么？
  - **BRECQ / QDrop（标准 PTQ）→ 直接合并**：先对从同一源模型域自适应得到的不同目标域模型分别进行标准 PTQ（BRECQ：block-wise reconstruction + AdaRound 优化舍入策略；QDrop：block-wise reconstruction + 随机 dropout 激活量化），然后通过 midpoint weight averaging 合并量化模型。由于标准 PTQ 仅关注单模型量化精度，不考虑未来合并的兼容性，量化引入的离散化噪声会增大 loss landscape 的 error barrier，导致合并后模型在 interpolated 点出现显著的性能退化。
  - 全栈执行例子（以 QDrop + merging baseline 为例）：
    - 算法pipeline：源预训练 ResNet-50 θ₀ → SHOT 域自适应分别得到 θ_R→A、θ_R→C、θ_R→P → 各域独立 QDrop 量化（block-wise reconstruction + partial activation dropout，不含 Hessian 正则化，不含距离正则化）→ BN 折叠 → 量化权重 w_tar1_q、w_tar2_q、w_tar3_q → 直接 midpoint averaging w_merged = Σ w_tar_i_q / 3 → 推理。
    - 系统框架/Serving调度：论文未明确说明具体推理框架，仅使用 PyTorch 原生 fake-quantization + block reconstruction pipeline。
    - 编译框架/kernel调度：论文未明确说明，无自定义 kernel 或编译优化。
    - 硬件架构/芯片设计：论文未明确说明具体 GPU 型号或硬件配置。
  - Baseline 核心缺陷：标准 PTQ 不感知"未来合并"目标，量化后的权重与源权重差异大（weight divergence），且 loss surface 仍然尖锐（high curvature → 量化噪声被放大），合并时 error barrier 升高，harmoinc mean 精度显著下降（如 W4A4 下 QDrop 合并模型仅 58.92 mIoU 语义分割，GTA→Cityscapes+IDD）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HDRQ 方法**：通过三个组件在量化阶段即考虑合并兼容性：i) Noise-based Hessian Regularization：以噪声采样替代确定性量化值进行 block reconstruction，隐式正则化 Hessian，使权重到达更平坦的 loss 区域，减少合并时对噪声的敏感性；ii) Weight Distance Regularization：添加 ℓ₂ 正则项最小化量化权重与源权重的距离，利用三角不等式间接约束各域权重间的差异，使合并后的 interpolated 点更接近最优；iii) Noise-Sampling-Based Rounding：合并阶段通过采样噪声+cossim 筛选解决舍入歧义，确保合并取整方向一致。
  - **核心洞察**：从 error barrier 的理论分析出发（扩展 Frankle et al. 2020 的框架到量化噪声），推导出量化后 error barrier 的上界由 (1) Hessian 在 merged point 的大小和 (2) 两模型间距离共同决定。因此控制 Hessian（噪声量化）和距离（距离正则）就能降低合并的 error barrier。
  - **如何解决 Baseline 缺陷**：
    - 针对"量化权重 divergence 大"：距离正则化直接使各域量化权重停在源权重附近，||w_tar1 − w_tar2|| ≤ ||w_src − w_tar1|| + ||w_src − w_tar2||，两域距离被源距离之和 bound。
    - 针对"loss surface 尖锐"：噪声量化使 E[L(ŵ)] ≈ E[L(w) + ½·εᵀ·∇²_w L(w)·ε]，训练过程隐式惩罚 ∇²_w L，引导到平坦区域（Figure 2 可视化验证）。
    - 针对"合并舍入歧义"：当 Δ₁≈Δ₂ 时浮点合并退化为整数歧义，noise sampling rounding 利用 cosine similarity 从多个噪声样本中优选取整方向。
    - 效果：W4A4 语义分割，QDrop merged mIoU 58.92 → HDRQ 63.00（+4.08）；W3A3 Office-Home R→A,C,P，QDrop 62.99 → HDRQ 64.70（+1.71% harmonic mean）。
  - 全栈执行例子：
    - 算法pipeline：源预训练模型 θ₀ → SHOT/HRDA 域自适应得到各目标域 θ_i → BN 折叠 → HDRQ 量化：block-wise reconstruction（Adam LR=0.001, cos annealing, 20000 iter, λ=5e-2），每次迭代 (a) 计算量化值 ŵ = clamp(⌊w/Δ⌉, …)·Δ (b) 噪声采样 ε = w−ŵ, 使用 w+ε 前向传播 (c) loss = reconstruction_L2 + λ·||w_src − (w+ε)||₂² (d) 更新 w (e) 最后 3500 iter 切换到确定性 fake quantization → 得到量化权重 w_i_q → 合并：对每层采样 30 组噪声，每组计算 I_merged^k = ⌊(I₁·Δ+ε₁^k + I₂·Δ+ε₂^k)/(2Δ)⌉，选 cosine similarity 最高的 → w_merged = midpoint_average(w_1_q, w_2_q) → 多目标域统一推理。
    - 系统框架：论文未明确说明具体推理框架或 Serving 系统修改，为标准 PyTorch block-wise reconstruction pipeline。
    - 编译框架/kernel调度：论文未明确说明，无自定义 kernel 实现。
    - 硬件架构：论文未明确说明具体 GPU 型号。
