## APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers

- baseline方法是什么？
  Baseline 是 BRECQ（ICLR 2021）的 block-wise 量化重建框架，以及在此基础上改进的 QDrop（ICLR 2022）。BRECQ 使用 Hessian guided loss 来衡量量化质量，但该 loss 做了两个不准确的假设：(1) 将 Hessian 矩阵近似为 Fisher Information Matrix（FIM），(2) 将 FIM 的对角元近似为输出梯度平方。这两个近似在模型预测分布与真实数据分布不一致时会引入误差，且无法泛化到分类以外的任务（如检测和分割）。此外，ViT 中 post-GELU 激活分布严重不平衡（负值集中在 [-0.17, 0]，正值稀疏且范围可高达 40），导致量化误差极大。
  
  Baseline 全栈执行例子（ViT-S, W3/A3, ImageNet）：
  - 算法pipeline：BRECQ/QDrop block-wise 重建 → 每 block 用 channel-wise 均匀量化权重、layer-wise 均匀量化激活 → 使用 MSE loss 或基于 FIM 近似的 Hessian loss 优化 AdaRound 权重 → 输出量化 block。
  - 系统框架：timm 加载预训练 ViT → 校准集为 1024 张 ImageNet 无标签图 → PyTorch forward/backward 更新 AdaRound 参数 → 无需完整训练集。
  - 编译框架：论文未明确说明（使用标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明（无自定义 kernel，使用标准 PyTorch 量化算子）。
  - 硬件架构：论文未明确说明（在 NVIDIA RTX 4090 GPU 上训练，Intel i5-12400F CPU 上测试 W8A8 推理延迟）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 APHQ-ViT，包含两个核心创新：(1) Average Perturbation Hessian（APH）loss，(2) MLP Reconstruction（MR）。
  
  **APH loss 解决 Hessian 估计不准确**：BRECQ 的 Hessian 依赖 FIM 近似，在输出分布拟合不准时误差大。APHQ-ViT 直接从 Hessian 定义出发，对 block 输出 O 施加微小扰动 ΔO=10^-6，得到 O⁺ 和 O⁻，分别前向传播通过剩余 blocks 计算蒸馏 loss（分类用 KL 散度，检测用 KL+smooth L1），再反向传播得 Jacobian J⁺ 和 J⁻，用有限差分公式 H_i = (J⁺_i - J⁻_i) / (2·ΔO) 直接计算 Hessian 对角元。相比 BRECQ 的近似，APH 无需 FIM 假设，理论上可泛化到检测和分割等多任务。同时对所有校准样本的 Hessian 取平均（H_bar），降低梯度方差、稳定训练（定理 3.2 证明 Var[∂L_APH/∂θ] ≤ Var[∂L_PH/∂θ]）。
  
  **MR 解决 post-GELU 激活量化困难**：将 MLP 中 GELU 替换为 ReLU，然后用 APH 加权的 L2 loss 重建 MLP 输出，使得 ReLU 版本逼近原始 GELU 版本的输出。MR 同时添加 clamp loss：用 99% 分位数截断 ReLU 后的正激活值，约束激活范围。GELU→ReLU 的替换消除了 GELU 负值区域 [-0.17, 0] 造成的密集分布不平衡，clamp 则降低了层间激活范围的巨大差异。论文论证 ReLU 在浅层 MLP（逐层单独重建）中不会出现 dying ReLU 问题，且 ReLU 可被折叠进前一层线性层，加速推理。
  
  论文方法全栈执行例子（ViT-S, W3/A3, ImageNet）：
  - 算法pipeline：block-wise 量化 → 阶段1：计算 block 输出的 APH（正/负扰动 → 蒸馏 loss → Jacobian 差分 → H_bar）→ 阶段2：MR 替换 GELU 为 ReLU → 用 L_Direct + α·L_Clamp 重建 MLP（APH 加权）→ 阶段3：QDrop 量化重建 → 用 L_APH = Σ(Ô_i - O_i)²·H_bar_i 优化 AdaRound 权重 → 输出量化 block。全部使用标准均匀量化器（channel-wise 权重、layer-wise 激活），无需专用硬件。
  - 系统框架：timm 加载预训练 ViT / MMDetection 加载检测模型 → 1024 张 ImageNet 或 256 张 COCO 无标签校准集 → PyTorch 训练约 62-170 min（单 RTX 4090）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（ReLU 替代 GELU 后，在 CPU W8A8 推理上实现了 1.49×-1.75× 的加速，因 ReLU 可被折叠进前一层矩阵乘法）。
  - 硬件架构：论文未明确说明。
