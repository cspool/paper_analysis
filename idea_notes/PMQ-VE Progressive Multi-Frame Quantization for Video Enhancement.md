## PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

- baseline方法是什么？
  Baseline 是传统的 **Post-Training Quantization (PTQ)** 方法，包括：(1) 工业级工具 Quantization API——**OpenVINO** [Gorbachev et al. 2019]、**TensorRT** [Vanholder 2016]、**SNPE** [Ignatov et al. 2018]；(2) 经典统计方法——**MinMax**（全局 min/max 裁剪）、**Percentile**（分位数裁剪）、**NoisyQuant**（噪声偏置增强 PTQ）；(3) 低层视觉专用方法——**DBDC+Pac** [Tu et al. 2023]（校准+蒸馏）、**2DQuant** [Liu et al. 2024]（单边搜索+知识蒸馏）。这些方法存在两个核心缺陷：(a) **无法跨帧分配差异化表示能力**：视频增强模型需从多帧聚合纹理和运动信息，各帧激活分布显著不同（见图 2a），但传统方法对多帧执行统一 per-tensor 量化，忽略了帧间激活分布差异，导致动态范围跨帧不匹配和亚像素空间细节利用不足；(b) **过度依赖全精度教师**：直接用量化方法将高精度网络量化为低精度时，FP32 教师与低比特学生（2bit/4bit）之间存在显著的容量差距，传统方法仅用全精度教师进行知识蒸馏，使低比特学生难以学习高质量映射。

  Baseline 全栈执行例子（以 RSTT 模型在 STVSR 任务上 4-bit 量化为例）：
  - **算法层**：输入 7 帧 LR 视频 → RSTT encoder 提取多级特征字典 → 传统的 per-tensor uniform quantizer：所有帧的 Linear/MatMul 层激活共享同一对 [lb, ub]=[min(all_frames_act), max(all_frames_act)] 或使用 2DQuant 的单边搜索 → 裁剪后做 round((x-lb)/Δ) 量化 + dequantize → decoder 逐级查询特征字典重建 HR 帧 → 若使用 DBDC+Pac，则加入 FP 教师对输出做 L2 蒸馏。
  - **系统框架层**：8×NVIDIA V100 GPU，PyTorch fake quantization，Adam 优化器 lr=2×10^-4，Cosine Annealing 20000 迭代，batch size=8 per GPU。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。方法为纯量化算法，不涉及自定义 kernel 或硬件修改。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出的 **PMQ-VE** 是一个粗-细两阶段量化框架，通过两个核心模块分别解决 baseline 的两个缺陷：
  **(1) BMFQ（Backtracking-based Multi-Frame Quantization）→ 解决跨帧表示能力分配不均**：对多帧激活张量 X∈R^{N×C×H×W} 进行 per-frame 独立量化。为每帧 X_i 独立搜索裁剪边界 (lb_i, ub_i)，采用百分位数初始化（lb∈[p0.1, p10], ub∈[p90, p99.9]）抑制 outlier，再通过回溯搜索（BTBI）在候选空间中递归评估量化误差并剪枝/回溯，高效收敛到每帧最优边界。与 baseline 中对所有帧使用统一量化范围不同，BMFQ 使每帧获得适配其自身激活分布的动态范围。
  **(2) PMTD（Progressive Multi-Teacher Distillation）→ 解决全精度教师与低比特学生之间的容量差距**：采用层次化蒸馏框架。训练低比特模型（如 4-bit）时，同时使用全精度（FP32）教师和中间比特教师（如 INT8）进行监督。损失函数 L_PMTD = (L_INT + α(t)·L_FP) / (1+α(t))，其中 α(t) 随时间线性增长，使训练从中间教师逐步过渡到全精度教师。每个教师损失包含输出级 L2 重建损失和中级 MSE 特征匹配损失。通过渐进过渡，降低低比特模型训练难度，弥合量化误差。

  PMQ-VE 全栈执行例子（以 RSTT 在 STVSR 任务上 4-bit 量化为例）：
  - **算法层**：
    - 粗阶段（BMFQ）：输入 7 帧 LR 视频 → RSTT encoder 提取特征 → 对每层 Linear/MatMul 的激活 X∈R^{N×C×H×W}，BTBI 算法为每帧独立搜索 (lb_i, ub_i)：lb_i 从 p0.1 开始向 p10 回溯搜索，ub_i 从 p99.9 开始向 p90 回溯搜索 → 对每帧执行 clamp+round+dequantize 假量化 → 评估 ||X_i - X̂_i||^2，剪枝低效路径 → 得到每帧最优裁剪边界的量化模型。
    - 精阶段（PMTD）：对 BMFQ 初始化的 4-bit 模型进行蒸馏微调 → 先训练 8-bit 中间模型（用 FP 教师蒸馏）→ 训练 4-bit 模型时，每个迭代：前向得到学生输出 out_4bit → 同时计算与 INT8 教师的 L2 损失和特征 MSE，以及与 FP 教师的 L2 损失和特征 MSE → α(t) 从 0 线性增长至 1，使监督信号从 INT8 逐步过渡到 FP → 通过 STE 反向传播更新量化边界和权重 → 最终得到 4-bit 量化模型 → decoder 重建 HR 帧。
  - **系统框架层**：8×NVIDIA V100 GPU，PyTorch fake quantization，Adam 优化器，Cosine Annealing。粗阶段 batch size=8/GPU（无蒸馏），精阶段 batch size=2/GPU（含蒸馏，显存更大）。数据增强：随机裁剪、旋转、翻转。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。PMQ-VE 为纯量化算法，fake quantization 在 PyTorch 框架内完成，无自定义 CUDA kernel 或硬件修改。
