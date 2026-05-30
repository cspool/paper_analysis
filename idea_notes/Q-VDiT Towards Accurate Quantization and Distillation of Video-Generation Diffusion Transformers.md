## Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

- baseline方法是什么？
  - Baseline 方法是将已有的图像 DiT 量化方法（如 ViDiT-Q、PTQ4DiT、Q-DiT 等）直接应用于视频 DiT 模型。这些方法存在两个核心缺陷：(1) 量化过程仅使用标准 PTQ 流程（RTN 量化 + MSE 重建损失），未针对视频生成的高信息密度进行误差补偿，导致剧烈量化信息丢失；(2) 优化目标仅考虑单帧的 MSE 对齐，忽略视频帧间的时空相关性，导致帧间不连贯和整体视频质量下降。
  - 全栈执行例子（以 W3A6 ViDiT-Q 在 Open-SORA 上的推理）：
    - **算法层**：使用 channel-wise weight quantization + dynamic token-wise activation quantization，RTN 量化，weight 从 FP16 → INT3，activation → INT6。对每层 Linear Y=Q̂(X)·Q̂(W)^T，直接使用量化值计算，量化误差 Δ=W−Q̂(W) 被丢弃。
    - **系统框架层**：基于 PyTorch 推理，使用标准 INT 矩阵乘法 kernel。校准阶段用 10 个 prompt 的 50 个去噪步进行 PTQ 校准，损失 L_task = ||S^{FP} − S^{Q}||²，仅按逐帧 MSE 优化量化参数。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：标准 PyTorch 量化推理 kernel，无定制 kernel 或融合优化。FP16 baseline 直接运行，量化模型使用 INT GEMM。
    - **硬件架构层**：运行在 NVIDIA GPU 上（具体型号论文未明确），使用 CUDA 环境。
  - Baseline 缺陷的直接体现：W3A6 下 ViDiT-Q 的 Scene Consistency 仅 11.99（FP 为 39.61），VQA-Technical 仅 10.26（FP 为 53.49），无法生成有意义的连贯视频。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - Q-VDiT 在算法 pipeline 层引入两个互补组件解决 baseline 的量化信息丢失和帧间优化缺失问题：
    1. **TQE (Token-aware Quantization Estimator)**：从信息论角度（Theorem 3.2, H(Δ)≤H(W)），在 token 维度和 feature 维度使用 rank=1 低秩参数 (α,β) 估计量化误差，将 X·W^T 近似为 Q̂(X)·Q̂(W)^T + Δ̂·β。Token-aware 缩放因子 M 按帧区分不同 token 的量化损失程度，修正了 baseline 丢弃量化误差的问题。额外参数仅 d_out+d_in（vs baseline 的 0），推理时通过 LoRunner Kernel 融合，延迟增加 <5%。
    2. **TMD (Temporal Maintenance Distillation)**：在优化目标中增加帧间时序分布 KL 散度项 L_temporal = Σ_i KL(D^{FP}_i || D^{Q}_i)，其中 D_i = softmax([cos_sim(S_i,S_1),...,cos_sim(S_i,S_t)])。该梯度（Eq. 16-18）确保每帧的优化受所有帧共同引导，修正了 baseline 只优化单帧 MSE 的缺陷。
  - 全栈执行例子（以 W3A6 Q-VDiT 在 Open-SORA 上的推理）：
    - **算法层**：对每层 Linear，执行 Y = Q̂(X)·Q̂(W)^T + ((M ⊙ Q̂(X))·α)·β^T。TQE 的 rank=1 低秩分支补偿 token 维度和 feature 维度的量化误差。校准时联合优化 L_total = ||S^{FP}−S^{Q}||² + 100·Σ_i KL(D^{FP}_i || D^{Q}_i)，TMD 项确保帧间分布对齐。
    - **系统框架层**：基于 PyTorch，校准使用与 baseline 相同的数据（10 prompts, 50 steps），但用 TQE 修正前向传播。batch size=4，学习率 lr=1e-6（量化参数）、lr=1e-5（TQE 参数）。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：使用 LoRunner Kernel（来自 SVDQuant）将 TQE 低秩分支（rank=1）与量化 GEMM 融合。Down projection（X→Δ̂）与量化 kernel 融合，Up projection（Δ̂→output）与量化计算 kernel 融合，共享激活张量以消除额外内存访问，kernel 调用次数减半。rank=1 时延迟增加 <5%，远低于 SVDQuant 的 rank=16 配置。
    - **硬件架构层**：在 NVIDIA GPU 上运行（W4A8 时显存节省 2.40×，推理加速 1.35×，Tab. 7）。
  - 效果：W3A6 下 Scene Consistency 从 SOTA 11.99/12.04 提升到 23.40（近翻倍），VQA-Technical 从 29.58 提升到 59.10（翻倍），W4A6 下几乎无损。即使在更高位 W4A8 下 VQA-Aesthetic 达 71.32，超过 FP 模型的 66.91。
