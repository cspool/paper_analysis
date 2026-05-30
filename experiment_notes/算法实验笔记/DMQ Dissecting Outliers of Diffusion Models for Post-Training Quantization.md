## DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  DMQ 提出了针对扩散模型后训练量化（PTQ）的异常值抑制方法，核心包含两个组件：(1) **Learned Equivalent Scaling (LES)**：学习通道级缩放因子 τ ∈ R^{Cin}，通过双向重分布异常值（Y = (X/τ)(τ^T ⊙ W)），最小化量化输出与原始输出的 MSE 来平衡权重和激活之间的量化难度。引入 **Adaptive Timestep Weighting**，基于各时间步的累积损失动态调整权重 λ_{t_i} = (1 - Λ_{t_i}/ΣΛ_{t'})^α，优先优化量化误差小但对最终质量影响关键的早期去噪步。(2) **Power-of-Two Scaling (PTS)**：针对 skip connection 等层中的极端异常值，使用通道级 2 的幂次缩放因子 δ，通过 bit-shift 操作高效处理（Y ≈ s^X s^W · Σ X̃ · (W̃ ≪ δ)），配合 **Voting Algorithm** 从校准集中通过统计共识选择鲁棒的 δ 因子。
  实验比较了 W8A8、W4A8、W4A6 量化配置下的无条件生成（FFHQ、LSUN-Bedroom、LSUN-Church）、条件生成（ImageNet）和文本引导生成（MS-COCO + Stable Diffusion v1.4），对比方法包括 Q-Diffusion、PTQD、EDA-DM、TFMQ-DM。

- 硬件平台是什么，配置是什么。
  论文未明确说明训练/推理所用 GPU 型号。实验使用 PyTorch 框架，基于 LDM（latent-diffusion）和 Stable Diffusion 官方实现。Section E 中自定义 CUDA kernel 在 GPU 上实现了 W4A8 GEMM 的 5.17× 加速（vs PyTorch FP32 GEMM at M=3072）。

- 模型是什么。数据集和bench分别是什么。
  模型：LDM-8（LSUN Church 256×256）、LDM-4（LSUN Bedroom 256×256 / FFHQ 256×256 / ImageNet 256×256）、Stable Diffusion v1.4（text-to-image, 512×512）。数据集：FFHQ 256×256、LSUN-Bedrooms 256×256、LSUN-Churches 256×256、ImageNet 256×256、MS-COCO（text prompts）。评估指标：FID、sFID、IS（条件生成）、LPIPS、SSIM、PSNR、CLIP Score（文本引导生成）。采样使用 DDIM sampler，无条件/条件生成用 20 步，文本引导生成用 50 步。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/LeeDongYeun/dmq。基于 LDM（https://github.com/CompVis/stable-diffusion）和 guided-diffusion（https://github.com/openai/guided-diffusion）构建。

  算法pipeline（后训练量化流程）：
  1. **校准数据收集**：使用 DDIM sampler（无条件/条件 20 步，文本引导 25 步），每步采样 n=256 个数据点，总计 N=T×n 个校准数据点（无条件 5120，类条件 10240，文本引导 6400）
  2. **LES 学习阶段**：逐层优化通道级缩放因子 τ：
     - 前向：X̂ = X/τ, Ŵ = τ^T ⊙ W
     - 量化：X̂_q = MinMaxQ(X̂), Ŵ_q = MinMaxQ(Ŵ)
     - 损失：L_i = ||X_i W - Q(X̂_i) Q(Ŵ)||²
     - 自适应时间步加权：λ_{t_i} = (1 - Λ_{t_i}/ΣΛ_{t'})^α，Λ_t ← 0.95Λ_t + 0.05·E[L_i]
     - 总损失：L = (1/B) Σ λ_{t_i} L_i
     - 迭代次数：4000-6000，batch size=32（无条件/条件）/8（Stable Diffusion）
  3. **权重精炼**：使用 BRECQ 进行 block-wise 权重量化重建（Adaround 自适应舍入）
  4. **PTS 因子选择**（仅 skip connection 层）：
     - 候选选择：对每个校准样本 i 和通道 k，评估 δ ∈ {0,1,...,D}，选最小化量化误差的 δ*_{i,k}
     - 投票：δ_k^{mode} = mode({δ*_{i,k}})，一致性 r_k = Σ1{δ*_{i,k}=δ_k^{mode}}/N
     - 阈值化：若 r_k > κ(=0.85)，δ_k = δ_k^{mode}；否则 δ_k = 0（不缩放）
  5. **推理融合**：
     - LES：τ 融合到权重（τ^T ⊙ W 预计算）和激活 scale（τ ⊙ s^X 预计算），零推理开销
     - PTS：激活量化 X̃ = clamp(⌊X / (2^δ ⊙ τ ⊙ s^X)⌉, l, u)，权重加载时执行 Ŵ_{kj}^{shifted} = Ŵ_{kj} ≪ δ_k，矩阵乘 Y ≈ s^X s^W · Σ X̃ · Ŵ^{shifted}

  关键张量计算示例（W4A8 skip connection 层）：
  - 权重 W ∈ R^{Cin×Cout}，激活 X ∈ R^{B×Cin}
  - LES 融合后：Ŵ = τ^T ⊙ W（预计算），激活量化 scale = τ ⊙ s^X（预计算）
  - PTS 融合后：X̃ = round(X / (2^δ ⊙ τ ⊙ s^X))，Ŵ^{shifted} = Ŵ ≪ δ（bit-shift at kernel load time）
  - 输出：Y ≈ s^X · s^W · (X̃ @ Ŵ^{shifted})
