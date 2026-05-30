## S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

- baseline方法是什么？
  Baseline方法为现有V-DMs PTQ方法（以PTQ4DiT/ViDiT-Q为代表）：使用随机或均匀采样策略从候选池中选取校准样本，在block-wise PTQ优化中对所有token施加均匀权重的量化损失L_quant = (1/n) Σ_j ||θ^f(x_{j,:}) - θ^q(x_{j,:})||²，所有token贡献均等。

  全栈执行例子（CogVideoX-5B W4A6 PTQ on A800）：
  - 算法Pipeline：随机/均匀选取N个校准样本（N≈40，受限于V-DM长token序列的显存约束） → 逐block进行前向传播 → 每block内对所有n=s×t个token计算MSE损失（均匀加权）→ 反向传播更新量化参数（channel-wise scale, rotation matrix, learnable clipping threshold）→ GPTQ weight quantizer逐列补偿误差 → 吸收量化参数输出W4A6模型。
  - 系统框架：PyTorch，单卡A800 GPU，block-wise优化（30样本，15 epochs/layer，AdamW + cosine LR）。
  - 编译框架：论文未明确说明。
  - Kernel调度：部署使用ViDiT-Q/FlatQuant的CUDA kernel进行INT4 weight dequantize和INT6 activation online quantize。
  - 硬件架构：论文未明确说明。

  Baseline存在两个核心缺陷：
  1. **校准数据方差高**：V-DMs的token序列极长（n=s×t，如CogVideoX-5B每帧数千token × 数十帧），在校准预算有限（仅几十个样本）的情况下，随机/均匀采样策略导致量化性能方差极大，不同seed下Imaging Quality波动可达±1.76。这是因为不同prompt和不同timestep的样本对扩散过程和量化过程的信息贡献差异显著，随机采样无法保证覆盖关键样本。
  2. **均匀token权重浪费优化能力**：V-DMs的全空间-时间注意力呈现明显稀疏模式——仅约10%的token拥有高注意力权重，其余90%对最终输出影响微弱。均匀权重的MSE损失将有限校准数据的优化能力浪费在对低影响力token的精确对齐上，而高影响力token的对齐不足导致生成质量下降。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出S²Q-VDiT，包含两个核心组件对应解决baseline的两大缺陷：

  **Hessian-aware Salient Data Selection (SDS)** 解决缺陷1（校准数据方差高）：
  - 关键观察：不同timestep的去噪信息量差异显著（相邻步表示变化大的timestep包含更多独特信息），不同样本对量化扰动的敏感度也不同（Hessian矩阵特征值大的样本扰动敏感）。
  - 同时计算扩散salience C_diff = ||x_t - x_{t-1}||²/||x_t||²（衡量去噪信息量）和量化salience C_quant = ||x_t^T x_t||_2（基于Levenberg-Marquardt Hessian近似衡量量化敏感度），min-max归一化后取乘积C_sample = C̅_diff · C̅_quant作为统一得分。乘积形式由算术-几何平均不等式保证仅当两个维度均高时才得高分，自然惩罚单维度强的样本。
  - 按C_sample降序选Top-N构成校准集，确保既覆盖关键的扩散去噪阶段又包含对量化最敏感的样本，使有限校准样本最大化表征能力和稳定性。

  **Attention-guided Sparse Token Distillation (STD)** 解决缺陷2（均匀token权重浪费优化）：
  - 关键观察：V-DMs各层attention map中大量token的注意力权重极低（<10%的top tokens占总注意力权重的绝大部分），仅小部分token对模型输出有实质影响。
  - 利用每block的多头注意力图A ∈ R^{H×n×n}计算每个token j的全局重要性得分S_j = Σ_{h,i} A_{h,i,j}，经min-max归一化并映射到[λ_min, λ_max]得到λ_j。
  - 将均匀加权损失改为L_quant = (1/n) Σ_j λ_j · ||θ^f(x_{j,:}) - θ^q(x_{j,:})||²，使高影响力token（λ_j→λ_max=1）获得完整优化力度，低影响力token（λ_j→λ_min=0.5）放松对齐约束。λ_min控制松弛程度，0.5为最佳平衡点。

  全栈执行例子（CogVideoX-5B W4A6 on A800，与baseline对比）：
  - 算法Pipeline（S²Q-VDiT新增步骤以→标出）：
    1. → 在候选池中计算每个(x_t, prompt)的C_diff和C_quant → min-max归一化 → 乘积得C_sample → Top-40构成D_calib（替代随机采样）
    2. → 用FP模型对D_calib中每个样本逐block前向传播，预计算并存储每个block的attention map A
    3. 逐block进行量化优化：
       → 从预存attention map中检索当前样本对应block的A → 计算S_j = Σ_{h,i} A_{h,i,j} → 归一化得到λ_j
       → 前向传播FP block和量化block得到θ^f(x)和θ^q(x) → 计算重加权损失 L_quant = (1/n) Σ_j λ_j · ||θ^f(x_{j,:}) - θ^q(x_{j,:})||²（替代均匀加权）
       → 反向传播更新量化参数（diag-balance scale lr=5e-3, rotation matrix lr=5e-3, clipping threshold lr=5e-2）
    4. GPTQ weight quantizer逐列补偿 + 吸收量化参数 → 输出W4A6模型
    5. CUDA部署推理：INT4 weight dequantize + INT6 activation online quantize
  - 系统框架：PyTorch + CUDA，单卡A800 GPU，校准40样本30样本训练15 epochs/layer，AdamW + cosine LR scheduler
  - 编译框架：论文未明确说明。
  - Kernel调度：部署基于ViDiT-Q [62] 和 FlatQuant [47] 的CUDA kernel做INT4/INT6推理，无额外kernel修改。
  - 硬件架构：论文未明确说明。

  Ablation验证（W4A4 CogVideoX-2B）：
  - SDS有效性：SDS vs ATOP(随机timestep+单prompt) → SDS Imaging Quality=52.95±0.69 vs ATOP=51.65±1.76，不仅均值更高且方差更低（0.69 vs 1.76），证明SDS在性能和稳定性上双重优势。
  - DS单独使用：Imaging Quality=52.73±0.98；QS单独使用：52.34±0.85，两者均优于随机采样且分别方差<1，联合使用(SDS)最佳。
  - STD有效性：w/o STD → w/ STD (λ_min=0.5) 在所有VBench维度上均有提升，λ_min在{0.3, 0.5, 0.7}范围内均有效证明鲁棒性。
  - SDS+STD可集成到已有PTQ方法：将SDS和STD应用于PTQ4DiT → Aesthetic Quality从45.49提升至46.89(+SDS)再至47.27(+STD)。
  - W4A6下CogVideoX-5B场景一致性：S²Q-VDiT=46.66，甚至超越FP(45.28)；W4A4下CogVideoX-2B场景一致性34.23，对比最佳baseline仅12.21（近3倍提升）。
