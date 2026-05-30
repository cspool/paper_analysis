## Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：HDRQ（Hessian and Distance Regularizing Quantization），一种面向多目标域自适应的合并友好型 PTQ 方法，包含三个核心组件：i) Noise-based Hessian Regularization：将量化误差建模为均匀噪声 ε∼U[-Δ/2, Δ/2]，向权重中加入采样噪声 w+ε 替代确定性量化值 ŵ 进行重建训练，隐式惩罚损失曲面的尖锐曲率，使权重收敛到更平坦的局部极小值，降低合并时的 error barrier；ii) Weight Distance Regularization：通过最小化量化后权重与源模型权重的 ℓ₂ 距离（||w_src − w_tar||₂），利用三角不等式（||w_tar1 − w_tar2|| ≤ ||w_src − w_tar1|| + ||w_src − w_tar2||）间接控制不同域自适应权重之间的差异，确保合并兼容性；iii) Noise-Sampling-Based Rounding：在合并阶段，对量化权重加入采样噪声后再取整（I_merged = ⌊(I₁·Δ₁+ε₁ + I₂·Δ₂+ε₂)/(Δ₁+Δ₂)⌉），通过 cosine similarity 筛选最优噪声样本，解决浮点域合并与整数域合并间的舍入歧义问题。整体流程：源预训练模型 → 单目标域自适应（HRDA/SHOT）→ HDRQ 量化（block-wise reconstruction 20000 迭代，Adam with cosine annealing LR=0.001，λ=5e-2，最后 3500 迭代切换到 fake quantization）→ 模型合并（midpoint weight averaging + noise sampling rounding）→ 多目标域统一模型。
  - 实验比较：在 Semantic Segmentation（GTA→Cityscapes/IDD，HRDA+ResNet-101）和 Image Classification（Office-Home 四域，SHOT+ResNet-50）两个多目标域自适应任务上，对比 BRECQ 和 QDrop 两种 PTQ baseline，涵盖 W6A6、W4A4、W8A8、W8A4、W4A8、W4A4、W3A3 多种 bit-width。指标为 mIoU（分割）和 Harmonic Mean Accuracy（分类）。额外包含 ablation study（逐组件增量消融：Baseline QDrop → +Noise-based quantization → +Distance regularization）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号和配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型：ResNet-101（语义分割 backbone）+ 简单卷积头（HRDA 架构）；ResNet-50（图像分类 backbone，SHOT 架构）。
  - 数据集/benchmark：
    - 语义分割：源域 GTA 合成数据集（Richter et al., 2016），目标域 Cityscapes（Cordts et al., 2016）和 Indian Driving Dataset / IDD（Varma et al., 2019）。指标 mIoU，30 次采样平均。
    - 图像分类：Office-Home 数据集（Venkateswara et al., 2017），四域（Real/Art/Clipart/Product），一域作源、三域作目标，四种源域配置 R→A,C,P / A→R,C,P / C→R,A,P / P→R,A,C。指标 Harmonic Mean Accuracy。
  - 单目标域自适应方法：HRDA（Hoyer et al., 2022，语义分割），SHOT（Liang et al., 2020，图像分类）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供开源代码链接。发表于 ICML 2025，作者机构为 POSTECH。
  - 算法 pipeline（以 ResNet-50 单层 block-wise reconstruction 为例）：
    1. **源模型域自适应**：对源预训练模型 θ₀ 各自适应到目标域，得到 θ₁（如 Real→Clipart）和 θ₂（如 Real→Product）。
    2. **HDRQ 量化（各域独立执行）**：
       - 对 θ₁ 的每个 block（含 BN 折叠后的卷积层），计算量化步长 Δ = (max(w)−min(w))/(2^b−1)。
       - 噪声量化模拟：w_hat = clamp(⌊w/Δ⌉, −2^{b-1}, 2^{b-1}−1)·Δ，噪声 ε = w − w_hat ∼ U[-Δ/2, Δ/2]，训练使用 w+ε。
       - Block-wise reconstruction：minimize ||F_block(w+ε, x) − F_block(w_orig, x)||₂² + λ·||w_src − (w+ε)||₂²，其中 λ=5e-2 为距离正则化系数。
       - 最后 3500 迭代切换到 fake quantization（确定性 ŵ），学习率衰减到很小。
    3. **合并（Noise Sampling Rounding）**：
       - 采样多组噪声 ε₁^k, ε₂^k ∼ U[-Δ/2, Δ/2]，对每组 k 计算 I_merged^k = ⌊(I₁·Δ₁+ε₁^k + I₂·Δ₂+ε₂^k)/(Δ₁+Δ₂)⌉。
       - 计算 cos_sim(vec(w_merged^k − w_tar), vec(w_tar1 − w_tar2))，选最高相似度的样本。
       - 最终合并权重：w_merged = (w_tar1_quant + w_tar2_quant) / 2（midpoint averaging，此时 step sizes 相同）。
    4. 推理：合并后的统一量化模型直接在各目标域上推理，无需额外的适配器或变换。
  - 张量计算核心（Hessian Regularization 的数值效果）：E[L(ŵ)] ≈ E[L(w) + ½·εᵀ·∇²_w L(w)·ε]，均匀噪声 ε 使损失函数隐式惩罚 ∇²_w L(w) 的大特征值，引导到平坦区域。
