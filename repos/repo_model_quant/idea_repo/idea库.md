# idea库

## Training Dynamics Impact Post-Training Quantization Robustness

- baseline方法是什么？
  Baseline 是现有的 PTQ 研究视角：Kumar et al. (2024) 和 Ouyang et al. (2024) 建立了量化误差的 scaling law，认为 `δ_PTQ` 随训练数据量增加而增大，即"模型训练数据越多，量化越困难"（quantization degradation increases as models are trained on more data）。由此引申出"量化更有利于 undertrained 模型"的结论。
  
  全栈执行示例：baseline 方法训一个 LLM → 在固定 token 预算（如 100B）用 cosine decay 完成训练 → 收集不同 token 数量处的 checkpoint → 对每个 checkpoint 做 GPTQ 3/4-bit 量化 → 绘制 `δ_PTQ vs training tokens` 曲线 → 观察到随 token 增加量化误差单调上升。
  - **算法pipeline层**：cosine decay 调度，学习率从峰值平滑衰减至零。训练数据越多时，学习率在后期越来越小，模型进入很陡峭的 loss 区域。
  - **系统框架层**：论文未明确说明。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

  Baseline 的缺陷：
  1. **混淆变量**：训练数据规模和优化动态（学习率衰减）被混淆在一起。cosine 调度下数据越多等价于学习率越小，前人将其归因于数据量而非学习率动态。
  2. **checkpoint 收集时机不当**：在未完成 annealing 的阶段收集 checkpoint，无法公平比较不同 token 预算的量化鲁棒性。
  3. **忽略了训练超参数可调性**：假设量化退化不可逆，未探讨通过调整训练超参数来改善 PTQ 鲁棒性的可能。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法通过三个层面解耦和调制训练动态对量化鲁棒性的影响：(1) 在大量开源模型的长训练轨迹上观测量化误差演变；(2) 受控实验分离学习率和数据量；(3) 提出具体的训练干预手段以改善 PTQ 鲁棒性。

  全栈执行示例：
  - **算法pipeline层**：
    - **解耦分析**：用 WSD 调度替换 cosine 调度。WSD 将训练分为恒学习率阶段 + 线性衰减阶段，在恒学习率阶段（长达 11T tokens for SmolLM3）量化误差基本保持稳定，仅在 lr 衰减时激增。同样 token 预算下，WSD 的 `δ_PTQ` 增长显著慢于 cosine。证明关键因素是学习率衰减而非数据量。
    - **学习率干预**：固定所有其他超参数，仅改变峰值学习率（如 1e-3, 3e-3, 6e-3），发现学习率越大 → 衰减后量化误差越小。在相同 validation loss 下，更大的学习率实现更低的量化误差。
    - **Weight averaging (LAWA/model soup)**：沿训练轨迹做 weight averaging 可作为 lr decay 的替代方案。LAWA 在全精度下略逊于 cooldown，但在 3-bit 量化后可以匹配甚至超越 cooldown 的表现。Model soup（多数据混合训练模型平均）量化误差低于任何单个成分。
    - **Hessian 几何分析**：量化误差的机制根源在于 loss landscape 的几何性质。学习率衰减时 Hessian 最大特征值（sharpness）和 trace 均急剧上升，模型进入更尖锐的 loss 区域（更敏感于量化引起的权重扰动）。较大的峰值学习率和 weight averaging 都促进收敛到更平坦的极小值（wider minima），从而提升量化鲁棒性。
  - **系统框架层**：量化使用 GPTQModel + HuggingFace Transformers，评估使用 vLLM 加速推理。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：量化 kernel 使用 GPTQ/AWQ 的 fused dequantization-GEMM kernel（混合精度 kernel 融合去量化和矩阵乘法步骤）。论文未修改 kernel 实现。
  - **硬件架构层**：论文未明确说明。

  对比 baseline 的关键改进：
  1. **穿透混淆变量**：将量化退化的根因从训练数据规模纠正为学习率动态。WSD 实验直接证明：固定的恒学习率阶段（无论多长）不会导致量化退化，退化仅发生在 lr 衰减阶段。
  2. **可行的干预手段**：(a) 选择更大的峰值学习率（在同样模型质量下降低量化误差）；(b) 使用 WSD 代替 cosine 以更好控制末期学习率；(c) 使用 weight averaging/LAWA 作为 lr 衰减的互补甚至替代方案；(d) 训练过程中持续监控 PTQ 误差作为超参数选择的附加指标。
  3. **几何解释**：揭示了学习率动态如何通过 loss landscape 的平坦度（flatness/sharpness）影响量化鲁棒性，为未来的训练设计提供了几何层面的理论指导。


## Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

- baseline方法是什么？
  Baseline 是经典的 block-wise PTQ 方法（BRECQ）和 layer-wise PTQ 方法（AdaRound/OPTQ/Z-FOLD）。
  
  **BRECQ**（block-wise）：将 Transformer block 内的所有层（Q、K、V、O、FFN）联合量化，最小化整个 block 输出重构误差 `E[||f(W_Q,W_K,W_V,...)(X) - f(W_Q,W_K,W_V,...)(X)||^2]`，使用学习方式优化 weight-rounding policy。全栈执行示例：给定一批校准数据 X（B×L×d），BRECQ 对 transformer block 中所有 linear 层同时量化 → 每轮迭代：forward W_Q→Q、W_K→K、W_V→V → 计算 attention `softmax(QK^T/√d)V` → 通过 O 投影和 FFN → 计算与全精度 block 输出的重构误差 → 反向传播更新所有层的量化参数/rounding policy → 重复直至收敛。**缺陷**：(1) 时间复杂度 O(B·L·d_h·max(d, L))，每轮需完整 attention forward pass，OPT-2.7B 需 20+ GPU 小时；(2) 对大模型（≥6.7B）OOM 不可运行；(3) 超参数敏感，对 LLM 未优化。
  
  **AdaRound/OPTQ**（layer-wise）：逐层独立量化，最小化每层输出误差 `E[||Q(W)X - WX||^2]`，Hessian 固定为 `H = 2E[XX^T]`。全栈执行示例：AdaRound 逐层遍历 Transformer → 对每个 linear 层：计算 H=2E[XX^T] → 优化 rounding policy V 最小化 `||WX - W̃X||^2` + rounding regularization → 输出量化后层权重 → 下一层用上层的量化输出作为输入 → 逐层累积误差。**缺陷**：(1) Hessian 仅基于 X 独立计算，未考虑 attention 内部 Q/K/V 之间的跨层依赖；(2) layer-wise 重构目标忽略了 attention output 的整体误差传播；(3) 低比特（INT2）下性能急剧退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  *aespa* 提出 "逐层量化 + attention-wise 重构" 的折中策略：每层单独量化（保留 layer-wise 效率），但损失函数以 attention 输出重构为目标（引入跨层依赖）。
  
  全栈执行示例（对应 paper Algorithm 1 + Table 4）：给定校准数据 X（128 segments × 2048 tokens from C4），aespa 对每个 Transformer block 执行：
  - **算法pipeline层**：先全精度前向一次预计算关键统计量 → `H_xx = E[XX^T]`（d×d）、`H_v = E[X A^T A X^T]`（d×d，含 attention map 信息）、`E[K^TK]`（d_h×d_h）、`E[Q^TQ]`（d_h×d_h）→ 所有后续迭代无需再执行 attention forward。
  - 量化 W_V：用 Z-FOLD 基于 `H_v` 初始化 scale/zero-point → 每轮用 OPTQ 初始 round 或 AdaRound 优化时，直接计算 `loss = tr(ΔW_V·H_v·ΔW_V^T)`（一次矩阵乘+逐元素乘），无需 forward pass。
  - 量化 W_Q：用 Z-FOLD 基于 `H_xx` 初始化 → 每轮 AdaRound 优化：`loss = tr(E[K^TK]·ΔW_Q·H_xx·ΔW_Q^T)`（两次矩阵乘+逐元素乘）。关键：E[K^TK] 引入了 key projection 的信息（跨层依赖），但 K 是固定全精度的（单独量化策略保证）。
  - 量化 W_K：同理，`loss = tr(E[Q^TQ]·ΔW_K·H_xx·ΔW_K^T)`，E[Q^TQ] 引入 query 的信息。
  - 量化 FFN/O-proj：使用 standard layer-wise 目标 `loss = tr(ΔW·H_xx·ΔW^T)`。
  - **kernel调度层**：论文未明确说明（纯算法层方案，kernel 为 PyTorch 标准 matmul）。
  - **Serving调度/编译框架/硬件架构/芯片设计层**：论文未明确说明。
  
  对比 baseline 的关键改进：
  1. **跨层依赖建模**：baseline Hessian `H=2E[XX^T]` 将 Q/K/V 视为独立；aespa 的 `H_v=2E[XA^TAX^T]` 通过 attention map A 直接将 Q 和 K 的信息耦合进 V 的 Hessian，且 W_Q/W_K 的损失函数通过 `E[K^TK]` 和 `E[Q^TQ]` 注入跨投影依赖。
  2. **预计算加速**：baseline 每轮需 O(B·L·d_h·max(d,L)) 的 attention forward；aespa 通过预计算统计量，每轮仅 O(d_h d^2)，且与校准数据量无关。OPT-125M 上 FLOPs 差 28 倍（0.24 vs 6.7 GFLOPS）。
  3. **全数据集梯度估计**：预计算使单次 loss 计算等价于在整个校准集上评估（batch size = 全部 128 segments），梯度估计更准确，收敛更快（2000 轮迭代即可）。
  4. **单独量化可行性验证**（Table 5）：虽然逐层量化，但 attention-wise 重构目标使性能接近 block-wise 联合量化（BRECQ），INT3/4 下几乎无损。

## Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

- baseline方法是什么？
  CADyQ (Content-Aware Dynamic Quantization) 在 SR 模型每一层嵌入可训练的 bit selector，同时对 layer 和 patch 进行 bit-width 自适应分配，依据特征梯度大小测量量化敏感度。全栈执行示例：给定一张 LR 输入图像，CADyQ 将其分割为 patch 送入 SR 模型 → 每个卷积层前，bit selector 根据该层特征梯度 magnitude 为该层+该 patch 选择 bit-width（4/6/8 bit）→ 对应层用选中的 bit 量化激活值（weight 固定 8-bit）→ 逐层累积计算 → 输出 SR 重建。缺陷：(1) 每层的 bit selector 引入额外计算开销，深层网络尤为严重；(2) 逐层独立调整 bit 会打乱原始模型层间关系，降低量化模型的表示能力（t-SNE 可视化证实 CADyQ 量化后层特征分布与原始全精度模型显著偏离）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Granular-DQ 完全放弃 layer sensitivity 考量，转为 patch-wise + layer-invariant 的动态量化范式，基于图像内在的多粒度线索和熵统计。全栈执行示例：给定一张 LR 输入图像 X：
  - **算法pipeline层**：GBC 编码器对 X 提取 D 层多粒度特征（Z_1 细粒度纹理→Z_D 粗粒度结构），经 GroupNorm+AvgPool+Concat+GAP 融合为全局描述 S。线性层 + Gumbel-Softmax 为每个 patch 采样门控分数 p_i，映射到候选 bit {4,6,8}。E2B 在训练集上预计算所有 patch 的像素熵分布 H（Gaussian 核密度估计），插入分位数阈值（0.5, 0.9）将 H 切分为 3 个子区间对应 [4,5,8] bits。对 GBC 分配高 bit 的 patch，根据其熵值落入区间确定最终适配 bit。ATC 用 EMA(γ=0.9997) 动态校准阈值。量化使用 QuantSR 方案，weight 固定 8-bit。
  与 baseline 的核心差异：所有层对同一 patch 使用相同 bit-width，保持了层间关系的一致性（t-SNE 验证 Granular-DQ 量化后特征分布更接近全精度模型）；不需要每层插入 bit selector，仅输入端一个 GBC，计算开销可忽略。
  - **系统框架层**：论文未明确说明（基于 PyTorch 实现，不涉及 serving/编译框架/kernel/硬件层次修改）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

## Task-Specific Zero-shot Quantization-Aware Training for Object Detection

- baseline方法是什么？
  Baseline为task-agnostic ZSQ方法（PSAQ-ViT V2、MimiQ、CLAMP-ViT）以及标准real-data QAT方法（LSQ、LSQ+）。task-agnostic ZSQ方法在数据合成阶段仅使用L_prior = L_BNS（BNS对齐）或L_PSE（Patch Similarity Entropy）生成无类别/无边界框标签的通用图像，QAT阶段仅使用KL散度蒸馏对齐量化模型与全精度模型的输出logits，不引入检测任务特定的训练损失。而LSQ/LSQ+虽然使用真实数据训练，但在有限校准集（如2k样本）下性能严重退化。
  
  全栈执行例子（以YOLOv5-s W6A6 MS-COCO，task-agnostic ZSQ baseline为例）：
  - 算法Pipeline：随机高斯噪声初始化x → 仅用L_prior + L_reg优化合成无标签图像 → 生成120k张task-agnostic校准图像 → LSQ量化全精度网络ϕ(θ)为W6A6 → 仅用KL散度蒸馏L_KD对齐输出logits微调 → 输出量化权重θ^q
  - 系统框架：PyTorch，2× RTX 4090 GPU
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明

  Baseline核心缺陷：(1) task-agnostic合成图像缺乏目标类别、边界框位置和尺寸信息，与检测任务分布不匹配；(2) 检测数据集类别分布不均，随机均匀采样产生的标签导致合成数据分布失实；(3) QAT阶段仅用logits对齐不足以恢复复杂检测网络的性能，缺少对中间特征和检测head的约束。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出首个task-specific ZSQ for object detection框架，两个阶段对应解决baseline缺陷：
  
  - **Idea 1（Adaptive Label Sampling + Task-Specific Data Synthesis）** 解决缺陷1和2：以随机初始化的单目标标签y为起点，交替进行图像优化和目标检测重标注。每固定迭代步，利用预训练teacher对当前输入x做检测推理，以conf > conf_thresh + IOU < iou_thresh的规则增删标签，使得标签y逐步收敛到teacher认可的合理检测目标集合。固定最终标签后重新初始化输入并用task-specific损失L_total = α_prior·L_prior + α_detect·L_detect(ϕ(x),y) + L_reg优化合成图像，使图像中的视觉特征与标签中的目标类别、位置和尺寸对齐。此过程无需任何真实标注，仅靠预训练网络的知识即可重建出与真实数据类别分布和空间分布近似的校准集（仅2k样本，1/60原始大小）。
  
  - **Idea 2（Task-Specific QAT Distillation: L_feat + L_detect）** 解决缺陷3：在QAT阶段不仅使用预测级KL蒸馏L_KD，还引入特征级MSE蒸馏L_feat（对齐teacher和student中间层特征图，防止低比特下误差累积）和task-specific检测损失L_detect（直接利用合成标签中的边界框和类别信息训练量化网络的检测能力）。三项损失联合：L^Q = β_KL·L_KD + β_feat·L_feat + β_detect·L_detect。
  
  全栈执行例子（YOLOv5-s W6A6 MS-COCO）：
  - 算法Pipeline：Gaussian噪声初始化x(160分辨率) + 随机单目标标签y → 循环：teacher(ϕ(θ))前向检测 → 高置信度预测增删标签 → 获得收敛标签y* → 固定y*，重初始化Gaussian噪声x(640分辨率) → 最小化α_prior·L_prior + α_detect·L_detect(ϕ(x), y*) + L_reg 2500次迭代（Adam, lr=1e-2, 余弦退火） → 生成2k张task-specific合成校准集{(ẍ_i, ŷ_i)} → LSQ量化附加到所有内部层 → for each batch: 计算L_KD = KL(z^F(ẍ_i;θ), z^Q(ẍ_i;θ')) + L_feat = MSE(f_l^F, f_l^Q) + L_detect(ϕ^Q(ẍ_i), ŷ_i) → 反向传播更新θ^Q和量化scale s → 输出W6A6 YOLOv5-s，mAP=32.7%（超越full-data LSQ 31.5% +1.2pp，仅用1/60数据）
  - 系统框架：PyTorch，2× RTX 4090 GPU，合成阶段8 GPU生成256张/20分钟
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明

  Ablation验证（YOLOv5-s W4A4）：移除L_detect → mAP从19.0降至16.8（-2.2pp）；同时移除L_feat和L_KD → mAP降至11.8（-7.2pp）。Adaptive Label Sampling vs MultiSample(In-distri.) → mAP+2.3pp@W6A6。Task-agnostic QAT（Ours w/o L_detect）vs 完整方法 → W6A6提升2.3pp@YOLO11-s。

## SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

- baseline方法是什么？
  Baseline方法为HAST + TexQ的合成数据集生成部分（calibration center synthesis + hard sample generation + sample difficulty promotion），微调阶段使用标准ZSQ损失函数 L_ZSQ = KL(q(x_i;θ) || q(x_i;θ^q)) + λ_CE·CE(q(x_i;θ^q), y_i)，即对所有合成样本同时施加KL散度知识蒸馏和交叉熵硬标签损失。全栈执行例子（ResNet-18 W3A3 on ImageNet）：
  - 算法Pipeline：随机初始化高斯噪声样本 → 最小化L_IL + α·L_BNS优化合成样本 → RTN量化预训练模型 → 对所有样本统一使用KL+CE损失微调量化模型 → 输出W3A3量化权重和激活
  - 系统框架：PyTorch + TorchVision，单卡RTX 3090训练
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明
  Baseline存在三个核心缺陷：1）合成数据集含高频噪声，频域分布与真实图像差异显著；2）量化模型基于错误图像区域（off-target patterns）做预测；3）困难样本的硬标签常错误，误导微调。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SYNQ在baseline合成数据集生成后，引入三个合成感知的微调创新改进：
  - Idea 1（低通滤波）：对合成样本x_i在频域应用高斯低通滤波器G，去除高频噪声 → 解决缺陷1。x_i^F = F^{-1}(G ⊙ F(x_i))，G为2D高斯核。滤波后样本的幅度分布更接近真实图像（ImageNet主要集中低频），使量化模型微调更高效。
  - Idea 2（CAM对齐）：在微调损失中加入L_CAM = ||S^θ(x_i^F) - S^θ^q(x_i^F)||_F²，强制量化模型的Grad-CAM显著性图对齐预训练模型 → 解决缺陷2。Grad-CAM通过激活梯度加权定位对预测贡献最大的图像区域，ReLU保留正贡献区域，MSE对齐确保量化模型关注与预训练模型相同的目标区域。
  - Idea 3（困难样本软标签）：定义样本难度δ = 1 - q_{y_i}(x_i;θ)，当δ > τ时仅使用KL散度（软标签），跳过硬标签交叉熵 → 解决缺陷3。避免预训练模型对困难样本的误分类标签误导量化模型。
  总损失：L_SYNQ = (1/N) Σ [KL + 1_{δ≤τ}·λ_CE·CE + λ_CAM·L_CAM]，其中低通滤波在微调前一次性完成。
  
  全栈执行例子（ResNet-18 W3A3 on ImageNet）：
  - 算法Pipeline：随机初始化高斯噪声样本 → 最小化L_IL + α·L_BNS生成合成样本 → FFT → 逐元素乘高斯低通核G → IFFT得到滤波样本x_i^F → RTN量化初始化θ^q → for each epoch/每个x_i^F: 前向传播θ和θ^q → 反向传播计算Grad-CAM梯度 → 计算L_CAM对齐 → 计算δ判定是否施加CE → 聚合KL + 条件CE + L_CAM → 更新θ^q（SGD, momentum=0.9, wd=1e-4, 100 epochs） → 输出W3A3量化模型
  - 系统框架：PyTorch + TorchVision，单卡RTX 3090，5120张合成样本，batch size=16（ImageNet），合成阶段Adam优化器（lr=0.5, 衰减0.1/50步）
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明
  
  Ablation验证（ResNet-18 W3A3）：Baseline 43.63% → +I1 49.43% (+5.80pp) → +I2 48.26% (+4.63pp) → +I3 46.42% (+2.79pp) → 全组合52.02%。低通滤波贡献最大，三项互补。SYNQ兼容任意ZSQ方法（GDFQ、Qimera、AdaDFQ、IntraQ、HAST、TexQ、Genie-PTQ），平均微调时间开销仅17.81%。

## SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

- baseline方法是什么？
  Baseline是uniform-precision PTQ方案（以GPTQ为代表）：对LLM所有权重矩阵使用统一bit-width的group-wise量化（group_size=128），采用基于Hessian矩阵的逐列误差补偿（OBQ延续）。全栈执行例子（以LLaMA-7B 2-bit GPTQ在A800上推理为例）：
  - **算法Pipeline**：输入tokens(2048) → embedding → L层Transformer Block，每Block内：RMSNorm(FP16) → MHA(Q/K/V/O投影+RoPE+Softmax+Attention) → 残差 → RMSNorm(FP16) → FFN(Gate/Up/Down投影+SiLU) → 残差。所有Linear层的权重用INT2统一量化（group_size=128, per-channel scale/zero），Hessian近似H = (1/P) Σ x^T x，逐列OBQ误差补偿。
  - **系统框架**：AutoGPTQ推理（AutoGPTQ），对统一2-bit权重的每个128元素group做dequantize后与FP16 activation做矩阵乘法。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：AutoGPTQ CUDA kernel对统一2-bit权重按group做dequantize+向量点积，warp内32 threads处理128列group，data access pattern一致。
  - **硬件架构**：NVIDIA A800 GPU（Ampere架构），论文未涉及RTL或模拟器修改。
  
  Baseline的两大缺陷：
  1. **统一精度忽略权重重要性差异**：所有权重同等对待，但salient权重对输出loss影响远大于非salient权重（δ_{i,j}=w_{i,j}²/[H⁻¹]_{j,j}²）。尤其在2-bit场景，有限码本容量无法同时容纳重要信息，导致perplexity崩塌（如LLaMA-7B GPTQ 2-bit WikiText2 perplexity高达152.31）。
  2. **element-wise混合精度的硬件不友好性**：现有方法（SpQR、PB-LLM、LLM-MQ）使用非结构化element-wise混合精度，需要额外存储bitmaps或code indices，无法与AutoGPTQ的group-wise packing兼容，导致部署效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SliM-LLM，包含两个核心组件对应解决baseline的两大缺陷：

  **SBA (Salience-Determined Bit Allocation)** 解决缺陷1（统一精度忽略权重差异）：
  - 关键观察：salient权重在channel维度上呈现spatial clustering（因activation outlier channels始终出现在固定位置，由Theorem 1证明：x_{:,p}^* > x_{:,j} → H_{p,p} > H_{j,j} → δ_{:,p} > δ_{:,k}）
  - 依据group内平均salience排序，双指针搜索最优混合精度配置：高salience group给3-bit，等量低salience group给1-bit补偿，其余2-bit（|G₁|=|G₃|约束维持average 2-bit）
  - 优化目标从MSE改为KL divergence（D_KL(xW^T || xŴ_sba^T)），从信息熵角度对齐输出分布而非仅最小化权重差值
  - 优势：group-wise结构化混合精度可直接用AutoGPTQ的packing机制，无需额外bitmap，硬件友好
  
  全栈执行例子（LLaMA-7B 2-bit SliM-LLM对比GPTQ）：
  - **算法Pipeline**：每层Linear权重W → 先按128列分组计算average salience → SBA双指针搜索确定1/2/3-bit group分布 → SQC对每个group内1% salient权重做τ校准→ GPTQ的OBQ逐列误差补偿。关键差异：不同group用不同bit-width（1/2/3-bit），salient group精度更高。
  - **系统框架**：修改版AutoGPTQ，存储时额外记录每个group的bit-width（2-bit/group聚合为整数），weights按各group精度分别pack。推理时按group逐精度dequantize。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：修改版AutoGPTQ CUDA kernel，逐group读取bit-width确定解包方式→dequantize→与shared activation做向量点积。因group内部精度统一，warp内threads的code path和数据访问逻辑仍保持一致。
  - **硬件架构**：NVIDIA A800 GPU，论文未涉及RTL或模拟器修改。

  **SQC (Salience-Weighted Quantizer Calibration)** 进一步解决缺陷1在group内部的残余问题：
  - 即使SBA给高salience group高bit-width，group内部仍有个别稀疏salient元素（约1%）与非salient元素共享量化器参数
  - SQC通过3-σ规则选中这些salient权重（w_s），引入calibration参数τ对scale和zero point做区间搜索[1-λ, 1+λ]（λ=0.1，50个candidate）
  - 优化加权目标: argmin_τ (||w_s - τ·s·Q(w_s,τs,τz)||² + ||w_us - τ·s·Q(w_us,τs,τz)||²)，扩大量化器perception interval的同时w_s和w_us仍共享同一套(τs, τz)，无需额外存储
  - 效果：OPT-1.3B某channel绝对误差从0.0055降至0.0039，salient权重误差显著降低

  协同效果：SBA处理global（group间）salience差异 → 结构化group-wise混合精度 → 硬件友好；SQC处理local（group内）salience差异 → 保护离散的重要权重 → 性能提升。SliM-LLM 2-bit LLaMA-7B WikiText2 PPL=14.58（vs GPTQ 152.31），接近3-bit水平的16×压缩比（6×内存减少），且保持GPU推理可用速度（61.2 vs 83.9 token/s）。

## I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

- baseline方法是什么？
  Baseline是标准PTQ方案：对post-Softmax激活使用log2量化器（LQ），对post-LayerNorm激活使用layer-wise均匀量化器，权重使用channel-wise均匀量化器，采用block-wise reconstruction优化目标。
  
  全栈执行例子（以DeiT-S W3A3在3090 GPU上推理为例）：
  - **算法Pipeline**：输入图像(224×224) → patch embedding → L个Transformer Block，每Block内：LayerNorm(FP32) → MHSA(QKV投影+Softmax+Attention) → 残差连接 → LayerNorm(FP32) → MLP(GELU+FC) → 残差连接。量化点：所有权重(MatMul输入)和激活值做INT3量化，log2量化器处理post-Softmax激活，layer-wise均匀量化器处理post-LayerNorm激活。
  - **系统框架**：PyTorch推理，无Serving框架修改。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：log2量化器的bit-shift操作在GPU上以标准整数运算执行，无自定义kernel。SULQ也通过bit-shifting执行。
  - **硬件架构**：NVIDIA 3090 GPU（Ampere架构），论文未涉及RTL或模拟器修改。
  
  Baseline的两个核心缺陷：
  1. **Quantization Inefficiency**：log2量化器的量化范围无法覆盖全部输入域，大量远离零的值被clamp到相同位置，造成大量化误差。例如3-bit时[8,26]段的值全部被clamp到7。
  2. **Rugged Loss Landscape**：channel-wise权重量化 + layer-wise post-LayerNorm激活量化的组合导致loss landscape粗糙且loss值放大，容易误导优化进入局部极小值。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出I&S-ViT，包含两个核心组件：
  
  **SULQ (Shift-Uniform-Log2 Quantizer)** 解决缺陷1：
  - 在log2变换前引入shift bias η: X_q = UQ(-log₂(X+η), b)
  - 反量化: X̄ = 2^{-round(D-UQ(X_q))} - η
  - 通过均匀量化器完整覆盖输入域（3-bit时8个整数均匀分布在[19,0]范围），同时保持对接近零区域的细粒度bit分配，匹配post-Softmax长尾分布。仅增加一次round和两次加法，可由bit-shifting高效执行。
  
  **SOS (Smooth Optimization Strategy)** 解决缺陷2：
  - Stage 1：全精度权重 + channel-wise量化的post-LayerNorm激活 → loss landscape平滑且loss值低，优化更稳定
  - Stage 2：通过scale reparameterization无损地将channel-wise转为layer-wise量化器（调整LayerNorm的affine参数和下一层权重）
  - Stage 3：量化权重并在全量化状态下微调恢复性能
  
  全栈执行例子（I&S-ViT对比Baseline）：
  - **算法Pipeline**：相同Transformer结构，但post-Softmax激活改用SULQ（shift+log2+uniform），post-LayerNorm激活在Stage 1用channel-wise量化获得平滑landscape，Stage 2通过scale reparameterization无损转layer-wise，Stage 3全量化微调。DeiT-S W3A3从Baseline的3.36%提升至55.78%（+52.42%）。
  - **系统框架**：同Baseline，PyTorch推理。
  - **编译框架**：论文未明确说明。
  - **Kernel调度**：SULQ的bit-shifting操作与标准log2量化器相同硬件效率，无额外kernel修改。
  - **硬件架构**：同Baseline，单张3090 GPU，论文未涉及RTL或模拟器修改。

## Scheduling Weight Transitions for Quantization-Aware Training

- baseline方法是什么？
  Baseline 方法是标准 QAT + 传统 LR 调度：使用梯度优化器（SGD/Adam/AdamW）搭配手动设定的 LR 调度策略（step decay 或 cosine annealing）更新全精度潜权重，间接训练量化权重。潜权重 `w^{t+1} = w^t - μ^t·g^t`，LR μ^t 按预设 schedule 衰减。

  **Baseline 全栈执行例子（以 ResNet-20 W2A2 在 CIFAR-100 上使用 SGD + step LR decay 为例）：**
  - **算法 Pipeline**：前向传播中，全精度潜权重 w 经 quantizer（normalize → round → de-normalize）变为 2-bit 量化权重 w_q → 用 w_q 计算卷积输出和交叉熵 loss → 反向传播时用 STE 将 ∂L/∂w_q 梯度原样回传到潜权重 w → SGD 优化器用当前 LR μ^t 更新 w。LR μ^t 按 step decay 每 100 epoch 除以 5，后期 LR 极小，但潜权重已聚集在 transition point（如零值）附近，即使小 LR 也能推动大量权重越过 transition point，导致量化权重在训练后期发生剧烈振荡（effective step size 不收敛），batch normalization 统计量不稳定，最终测试精度下降。
  - **Serving 框架**：论文未明确说明。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：论文未明确说明。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：(a) QAT 中量化权重的改变量（effective step size）与 LR 相关性弱——量化权重仅在潜权重越过 quantizer 的 transition point 时才改变离散级别，而潜权重是否越过 transition point 受其分布而非仅受 LR 控制；(b) 训练后期潜权重倾向于在 transition point 附近聚集，即使 LR 极小也能导致大量 transitions，造成训练不稳定和精度退化；(c) 手动 LR 调度无法显式控制量化权重的"粗到细"优化进程，与全精度训练中 LR 直接控制 weight update magnitude 的本质不同。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 TR（Transition Rate）调度技术：放弃调度 LR，改为调度目标 TR（target transition rate），并使用 TALR（Transition-Adaptive Learning Rate）自适应调整潜权重的更新步长，使得实际 TR 跟随目标 TR。

  **论文方法全栈执行例子（ResNet-20 W2A2 在 CIFAR-100 上使用 SGDT + cosine target TR decay）：**
  - **算法 Pipeline**（每迭代步 t）：
    1. 前向/反向传播与 baseline 相同（quantizer → STE → gradient g^t）。
    2. 计算当前 TR `k^t = Σᵢ I[w_d^t(i) ≠ w_d^{t-1}(i)] / N`（跨所有量化权重计数离散级别变化的占比）。
    3. 用 momentum=0.99 估计 running TR `K^t = mK^{t-1} + (1-m)k^t`，平滑掉单步噪声。
    4. 按加法规则调整 TALR `U^t = max(0, U^{t-1} + η(R^t - K^t))`，其中 R^t 是目标 TR（由 cosine scheduler 从初始值 λ√b_w 衰减到零）。当 K^t < R^t 时 U^t 增大（鼓励更多 transition），反之减小。
    5. 以 TALR 代替 LR 更新潜权重 `w^{t+1} = w^t - U^t·g^t`。
    与 baseline 的关键区别：TALR 不是手动预设的 schedule，而是实时反馈控制——当潜权重向 transition point 聚集、transitions 天然容易发生时，running TR K^t 会自然升高，TALR 自动降低以抑制 transition。这解决了 baseline LR 无法感知潜权重分布的问题。训练后期 U^t 趋近于零，即使潜权重已聚集在 transition point 附近也不会产生振荡。
  - **Serving 框架**：论文未明确说明。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：论文未明确说明。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  方法如何解决 Baseline 缺陷：
  - 缺陷 (a)：TR 调度直接控制量化权重层面的 effective step size，因为量化权重的 effective step size ≈ δ^t·I[transition occurred]（要么为 0，要么等于相邻量化级别间距 δ^t），所以控制 transition 数量等价于控制 effective step size。
  - 缺陷 (b)：TALR 通过负反馈机制自适应调整——当潜权重聚集在 transition point 附近时，即使小步长也能引发大量 transition，TALR 检测到 TR 超标后自动降低步长，从而抑制训练后期的振荡。
  - 缺陷 (c)：通过调度 target TR（而非 LR），实现了对量化权重的"粗到细"控制——初期高 target TR 允许充分探索，后期 target TR 衰减到零保证收敛稳定。对多种 scheduler（step/cosine）、多种优化器（SGD/Adam/AdamW/NAdam/Adamax/RMSProp/Adagrad）和多种任务（分类/检测）均有效。训练开销仅增加约 2%。

## SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

- baseline方法是什么？
  Baseline 方法分为三类：(1) **单独剪枝**：SparseGPT（基于逐层 Hessian 逆的稀疏回归）、Wanda（基于 weight × activation 幅度的简单剪枝）、Magnitude Pruning；(2) **单独量化**：OPTQ（基于 OBS 的逐层量化）、AWQ（激活感知权重量化，scale 显著 channel）、OmniQuant（可学习 clipping + channel scaling）、AffineQuant（等价仿射变换）、Group AbsMax；(3) **联合剪枝+量化**：JSQ（仅支持 8-bit，低位宽精度差）、L²QER（仅量化的一-shot 低秩适配，与稀疏结合时精度显著下降）。

  **Baseline 全栈执行例子（以 Wanda + Group AbsMax 为例）：**
  - **算法 Pipeline**：Wanda 逐行计算 weight × activation norm 重要性分数 → 保留 top 50% 权重（2:4 模式每 4 个保留 2 个）→ Group AbsMax 以 group size 128 对剩余权重做 4-bit 量化 → 输出稀疏量化模型。两种误差 E_Q 和 E_S 各自独立累积，不做联合补偿。
  - **Serving 框架**：论文未明确说明（实验使用 Sparse Marlin + vLLM 仅用于 SLiM 自身的加速比评估，未修改框架调度逻辑）。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：Sparse Marlin CUDA kernel 实现 2:4 稀疏 × 4-bit 量化矩阵乘法（数千行 CUDA，仅支持有限 GPU 架构）；无自定义 kernel 调度优化。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：(a) 量化误差和稀疏误差独立累积，无联合补偿机制；(b) 均匀量化 scaling factor 选择使用 AbsMax 对 outlier 敏感，Grid Search 次优且昂贵；(c) 低秩适配（如 L²QER）初始化基于 weight norm 而非对模型输出的实际影响，需要昂贵重训练；(d) 联合稀疏+量化时精度显著下降，尤其是 4-bit + 2:4 稀疏场景；(e) Group Quantization 增加反量化开销和 GPU kernel 实现复杂度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SLiM 通过三个协同设计的组件解决 baseline 缺陷：

  1. **SLiM-Quant**：将非凸量化 MSE 优化问题通过概率化重表述为凸问题，在权重直方图上做数值积分 + 多网格搜索，高效找到全局最优 uniform scaling factor α*。避免了 Grid Search 的次优性，且 uniform quantization 比 group quantization 去除了反量化开销（实测 6% 加速）。激活感知变体 SLiM-Quant^O 对 1% 最高显著性 channel 做 weight-activation scaling 互换，结合联合显著性 saliency = |diag(x_mean) × W|，进一步降低输出误差。

  2. **Wanda 剪枝** 在量化权重上施加 2:4 半结构化稀疏，直接产生硬件可加速的稀疏模式。

  3. **SLiM-LoRA**：提出满足可逆性和可加性的 saliency 函数 F(W) = diag(x)W（x 为校准集输入平均绝对值），使得低秩适配器可通过 SVD 数学推导闭式解，无需迭代训练。关键创新：(a) 可加性允许将适配器的显著性从压缩误差中隔离——F(-(E_Q + E_S)) = F(W^C - W)，SVD 分解后通过逆变换直接得到 L, R；(b) 显著性加权确保适配器优先修正对输出影响最大的权重通道，而非均匀最小化 Frobenius 范数（对比 Naive-LoRA）；(c) 无需重训练，one-shot 完成，比 L²QER 更好地处理联合稀疏+量化误差。

  **SLiM 全栈执行例子：**
  - **算法 Pipeline**：加载预训练权重 W → SLiM-Quant：构建权重直方图 f_abs → 多网格搜索 α* 最小化 E_quant + E_clip → W^Q = round(clip(W/α*)) × 2^{q-1} → Wanda 在 W^Q 上施加 2:4 稀疏 → 计算 E_C = W^C - W → 构建显著性矩阵 S_C = diag(x_mean)E_C → SVD(S_C) 取 rank r=0.1d → 逆显著性变换 L = diag(1/x_mean)L̃, R = R̃ → 可选：对 L, R 做 AbsMax group quantization（group size 128, 4-bit）→ 可选：在 C4 (300K tokens) 上 PEFT 微调（冻结 W^C，仅更新 L, R，使用 STE 处理量化适配器）。
  - **Serving 框架**：论文未明确说明（推理加速使用 Sparse Marlin kernel + vLLM，但未修改调度逻辑）。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：量化稀疏矩阵乘法使用 Sparse Marlin CUDA kernel；低秩适配器乘法使用 Dense Quantized Marlin（适配器量化时）或标准 PyTorch kernel；PEFT 微调阶段使用 Triton 自定义量化/反量化 kernel 降低 STE 开销。无自定义 kernel 调度优化。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  对比 Baseline，SLiM 的核心突破在于：(a) 将量化+稀疏+低秩适配三个组件通过统一显著性函数协同优化，而非各自独立处理；(b) SLiM-LoRA 的可逆可加显著性设计使低秩适配器获得闭式解，消除重训练需求；(c) SLiM-Quant 的概率化 uniform quantization 在保持硬件友好性的同时达到 group quantization 精度；(d) 进一步量化适配器 + PEFT 微调形成完整压缩-补偿-精调闭环。

## SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

- baseline方法是什么？
  - Baseline 方法分为两类：(a) **未量化的 ShardedDP 训练**：使用 Megatron-LM 的 distributed optimizer，BF16/FP32（weights/gradients）混合精度训练，每个 iteration 执行一次 AllGather（BF16 weights）和一次 ReduceScatter（FP32 gradients）。在全精度通信下，GPT-1.3B on 32 A100 的梯度通信耗时 379.3ms（Table 4）。随着模型增大和 GPU 数量增加，通信开销成为严重瓶颈。(b) **ZeRO++ / QSDP 风格的 4-bit 通信量化**：ZeRO++ 对权值直接做 INT4 group-wise 量化（qW），并对梯度使用两次 all-to-all（intra + inter node）各 4-bit 量化（ULq）。**核心缺陷**：直接 4-bit 量化权值 (qW) 导致显著精度损失（Table 1: GPT-125M qW 即使 group_size=32 也有 >4% loss 增加 vs baseline 2.29392）；两次级联的 4-bit 梯度量化 (ULq) 导致误差累积（Fig. 5: GPT-125M ULq loss 与 baseline 有显著 gap）；ZeRO++ 缺乏理论收敛保证，QSDP 限于特定 quantizer（random shift）和较强假设（Polyak-Łojasiewicz condition）。
  - Baseline 的全栈执行例子（以 Megatron-LM + ZeRO++ 风格全 4-bit 量化训练 GPT-1.3B on 32 A100 为例）：
    - **算法pipeline**：ZeRO++/QSDP Forward: `AllGather(QuantizeINT4(w_main[p])) → dequantize → ForwardPass(w_model, input) → Backward: Gradient(w_model, output) → QuantizeINT4(g_model) → IntraAlltoAll(INT4) → Dequantize → LocalReduce → QuantizeINT4(reduced) → InterAlltoAll(INT4) → Dequantize → ReduceScatter(INT4) → Optimizer(w_main, g_main)`。**核心缺陷 1**：权值直接 4-bit 量化→量化误差 ∝ max(|w|)/15（每 2048 group），但 |w| 范围大导致大误差（Fig. 4a 直方图显示权重分布宽且有长尾）。**核心缺陷 2**：两次连续 4-bit 量化引入误差传播→`g_final = Q4(Q4(g + ε₁) + ε₂)` ≈ g + 2ε，累积误差比单次量化加倍。**核心缺陷 3**：biased compressor (如 ternary quantizer) 直接用于权值压缩时，收敛失败（Counterexample 4.1 证明 ternary quantizer 直接量化 w 使 SGD 收敛到初始点而非最优解）。
    - **系统框架**：Megatron-LM 训练框架 + NCCL 集体通信。AllGather（权值分发）+ ReduceScatter（梯度归约）。论文未修改 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 NCCL ring-based all-gather/reduce-scatter 通信 kernel。量化/反量化使用标准 CUDA element-wise kernel（非融合）。GPU kernel 层面无特殊优化。
    - **硬件架构**：NVIDIA A100/H800 GPU 集群，NVLink/NVSwitch intra-node（高带宽），100Gbps Slingshot / 3.2Tbps InfiniBand inter-node（相对低带宽）。ZeRO++ 方法下，gradient comm time 约 45ms（Table 4），但由于 qW 误差导致 validation loss 不可接受。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：SDP4Bit 提出两种核心技术创新解决 baseline 的精度-通信权衡困境：
    - **对应缺陷 1（权值直接量化误差大）**：提出 **Quantization on Weight Differences (qWD)** —— 不量化权值本身，而是量化两次迭代间的权值差值 `Δw = w_t - w_{t-1}`。推理基础：(a) 经验上差值分布更均匀且范围更窄（Fig. 4b），INT4 量化误差更小；(b) 理论上 `‖q(Δw)-Δw‖/‖w‖ ≲ ‖q(w)-w‖/‖w‖`（量化误差相对权重本身的比值更小）；(c) 理论保证：weight diff quantization 兼容 arbitrary biased compressor，收敛率与标准 SGD 相同（Theorem 4.1: O(1/√T)），而 QSDP 需要 Polyak-Łojasiewicz condition 和特定 quantizer。
    - **对应缺陷 2（梯度 4-bit 连续量化误差累积）**：提出 **Two-Level Gradient Smooth Quantization (TLq-HS)** —— 采用 INT8（intra-node）+ INT4（inter-node）分级精度，并施加 Hadamard Transform 平滑 gradient outlier。作用：(a) INT8 intra-node 量化误差远小于 INT4，降低第一级误差；(b) Hadamard Transform（32×32）将 outlier 的信息分散到邻近元素（Fig. 6 直方图对比），使 gradients 趋于平滑，quantization error 大幅降低。
    - **对应缺陷 3（引入压缩带来的计算 overhead）**：通过算法-系统协同优化消除 overhead：(a) Hadamard kernel fusion（CUDA kernel 融合 transform + quantize/dequantize，overhead < 0.3%）；(b) operation pruning（利用 H·H=I 和分配律裁剪冗余 transform，6次→2次）；(c) buffer reuse（复用 Megatron-LM model weights buffer 避免额外显存分配）。
    - **对应缺陷 4（缺乏收敛保证）**：Theorem 4.1 在 smoothness 和 bounded variance 假设下证明 SDP4Bit 达到 O(1/√T) 收敛率，不需要 Polyak-Łojasiewicz 条件，支持 arbitrary biased compressor for weight compression。
  - 全栈执行例子（SDP4Bit GPT-6.7B on 128 H800，TP=4, PP=1）：
    - **算法pipeline**：每 iteration: Forward: `d[p] = w_main[p] - w_model[p] → d_q = QuantizeINT4_wqGroup(d[p], group_size=2048) → d_q_global = AllGather(d_q) → dequantize → w_model += d_deq → output = ForwardPass(w_model, input)` → Backward: `g_model = Gradient(w_model, output) → g_hat = Hadamard(g_model) → qg8 = QuantizeINT8(g_hat, group_size=512) → IntraAlltoAll(qg8) via NVSwitch → dequantize(received) → local_reduce → g_hat_red = Hadamard(local_reduced) → qg4 = QuantizeINT4(g_hat_red, group_size=128) → InterAlltoAll(qg4) via InfiniBand → dequantize → final_reduce → g_final = Hadamard(final_reduced) → Optimizer(w_main[p], g_final)`。效果：GPT-6.7B 最终 validation loss 与 baseline 几乎重合（0.24% 差异），18B on 128 H800 加速 4.08×，13B on 128 A100 Slingshot 加速 2.68×。
    - **系统框架**：Megatron-LM（Distributed Optimizer 模式，维持完整 model weights）+ NCCL（all-gather / all-to-all 集体通信）。通过 `--quantized-weights`, `--quantized-gradients`, `--hadamard-transform` 三个开关控制 SDP4Bit 启用。梯度 all-to-all pipeline 通过 `--gradient-alltoall-pipeline` 控制 chunk 并发度以实现 intra/inter-node 通信重叠。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：Tensor parallelism 通信（all-reduce for TP）保持全精度不变。Fused Hadamard + (de)quantize CUDA kernels：每个 thread block 处理一个 quantization group，在 shared memory 中完成 Hadamard transform 和 quantization，只需一次 global memory read 和一次 write，消除中间 global memory traffic。AlltoAll pipeline：梯度按 GPU local rank 分成 `pipeline_chunks` 份，intra-node 和 inter-node all-to-all 以流水线方式重叠执行（不同 chunk 的 intra 和 inter 通信在不同 CUDA stream 上运行）。
    - **硬件架构**：128× NVIDIA H800 (TP=4) 或 128× A100 (TP=8+PP=2)。SDP4Bit E2E training throughput: GPT-18B 59.2 TFLOPs (H800) vs baseline 14.5 TFLOPs (+4.08×)。gradient communication time: 45.8ms (SDP4Bit) vs 379.3ms (baseline GPT-1.3B on 32 A100)。Hadamard kernel fusion 将 grad comm time 从 64.6ms 降至 45.8ms（-29% vs 未融合版本），且 (de)quantization throughput 几乎不受 Hadamard 影响（Table 5: 301.8 vs 305.6 GB/s 量化，差异 < 1%）。

## RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

- baseline方法是什么？
  - Baseline 方法分为两类：(a) **先 SFT 再 PTQ 的两阶段 pipeline**：先用全精度（FP16/BF16）对预训练 LLM 做 SFT，然后用 PTQ 方法（RTN、GPTQ、QuaRot、SpinQuant）将 SFT 后的模型量化为 W4A4KV4。代表性流程：预训练 Llama 3.1 8B → FP16 SFT 训练（Tulu 3, 2 epoch, AdamW lr=5e-6）→ QuaRot PTQ（Walsh-Hadamard rotation + GPTQ-style weight calibration）或 SpinQuant PTQ（learned rotation + calibration），共需 2.1→1.3h 训练。(b) **纯 QAT 方法**（STE only, 不含 rotation）：直接将 STE 应用于 SFT loss，在 4-bit 量化约束下用梯度近似训练模型参数，不引入旋转矩阵来消除 outlier。
  - Baseline 的全栈执行例子（以 Llama 3.1 8B SFT→QuaRot W4A4KV4 两阶段 pipeline 为例）：
    - **算法pipeline**：预训练 Llama 3.1 8B → FP16 SFT（Tulu 3, AdamW, lr=5e-6, 2 epoch, 100k samples, cos schedule, 8×A100）→ QuaRot PTQ（对所有线性层插入 Walsh-Hadamard 旋转矩阵，吸收 normalization 参数，128 样本校准，逐层 GPTQ-style weight 最优量化参数搜索 + uniform activation/KV cache 量化）。**核心缺陷 1**：SFT 和量化分离导致次优结果——SFT 阶段优化的全精度权重在后续量化时产生严重精度损失（activations 中存在 outlier 值撑大量化范围，增加量化误差），Table 2 中 SFT→QuaRot W4A4KV4 avg=28.46 vs FP SFT avg=42.16（-13.70 gap）。**核心缺陷 2**：STE without rotation 在 4-bit 激活量化时性能严重退化（Table 2 STE avg=17.14），因为激活 outlier 导致 STE 梯度偏差过大（Theorem 4.3 证明预测误差正比于 weight quantization error 的加权和）。
    - **系统框架**：PyTorch 训练 + QuaRot 的 Hadamard CUDA kernel for online rotation。论文未修改 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：使用 fast Hadamard CUDA kernel（来自 QuaRot/QuIP#）处理在线旋转矩阵乘法。量化 matmul 使用标准 PyTorch 模拟量化（fake quantization），无 custom INT4 kernel。论文未说明具体的 GPU kernel 级优化。
    - **硬件架构**：8× NVIDIA A100 GPUs。SFT→QuaRot 训练时间 2.1h (FP SFT) + ~0h (QuaRot PTQ 几乎无训练开销) = 2.1h；peak memory 300GB (SFT) + 0 (PTQ calibration negligible)。但 W4A4KV4 精度损失严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：RoSTE 提出 **QA-SFT（Quantization-Aware Supervised Fine-Tuning）**，在单一训练阶段同时完成量化和微调，核心通过 **Bilevel Optimization + Adaptive Rotation** 解决两个 baseline 缺陷：
    - **对应缺陷 1（两阶段分离导致次优）**：将 SFT 和量化合并为单一优化问题（Formula 8），直接优化 `min_{W,R} L_SFT(m_Q(·; W,R)) s.t. R R^T = I`。简化为 bilevel formulation（Formula 11）：上层 STE 优化量化权重矩阵 W（SFT objective），下层选择旋转矩阵 R（quantization error surrogate loss E(12)）。训练中交替执行 rotation configuration search 和 QAT via STE，使量化误差在训练过程中持续被优化，而非固定于 PTQ 校准时刻。
    - **对应缺陷 2（激活 outlier 导致 STE 性能退化）**：通过 adaptive Walsh-Hadamard rotation 消除激活 outlier（Fig. 3 显示 RoSTE 训练收敛后无激活 outlier，而 STE 仍存在大量 outlier）。Proposition 4.4 证明旋转后的 weight quantization error 从 `O(d·max_i w_i²)` 降至 `O(‖w‖²)`（w.h.p.），将 outlier 主导的量化误差转化为均匀化误差。自适应策略（逐层在 I vs H 之间选择）避免"全部旋转"在某些层引入新异常值的问题（Table 3：No Rotation ROUGE=22.37, Complete Rotation ROUGE=13.09, RoSTE Adaptive=23.07）。
  - 全栈执行例子（RoSTE W4A4KV4 on Llama 3.1 8B, 8×A100）：
    - **算法pipeline**：预训练 Llama 3.1 8B → 修改 normalization layers（吸收 LayerNorm/RMSNorm 参数）→ one-shot rotation configuration search（逐层比较 W4A4 quantization error with/without Walsh-Hadamard rotation, 128 校准样本）→ QA-SFT training（STE + adaptive rotation, AdamW lr sweep {5e-6,1e-6,5e-7}, 2 epoch, 100k Tulu 3 samples, 8×A100, gradient accumulation=16, max seq len=1024, W4A4KV4 asymmetric uniform quantizer, per-token activation + per-channel weight quantization groups, clipping factor sweep {1, 0.95, 0.9}）→ 合并 offline rotations 到权重，保留 online rotations 在 fast Hadamard kernel。训练过程中，每层 linear layer forward: `X_rot = Q_x(X R_i)` (online rotation via Hadamard kernel if R_i not merged), `W_rot = Q_w(R_i^T W_i)`, `output = X_rot · W_rot` (INT4 matmul simulation)。Backward via STE: `∂L/∂W_i ≈ (R_i)^T · gradient_from_upper_layer`。效果：W4A4KV4 avg=31.69（vs best baseline SpinQuant avg=29.13, +2.56），W4A8KV4 avg=37.70（vs best baseline SpinQuant avg=35.02, +2.68），训练开销仅比 STE 多 0.4h（2.8h vs 2.4h, Table 10）。
    - **系统框架**：PyTorch 实现，无 Serving 框架修改。论文未说明具体的训练框架（如 FSDP/DeepSpeed 等）。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：使用 fast Hadamard CUDA kernel（继承自 QuaRot/QuIP#）处理在线旋转矩阵 R_3, R_3^T, R_4 的矩阵乘法，开销可忽略（训练时间 2.8h vs STE 2.4h，仅 +16.7%）。论文未引入新的自定义 kernel。
    - **硬件架构**：8× NVIDIA A100 GPUs。RoSTE training time 2.8h, peak memory 318GB（与 STE 317GB 几乎相同, Table 10）。总成本：2.8h × 8 A100 ≈ 22.4 GPU-hours。

## QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

- baseline方法是什么？
  - Baseline 方法：(a) **纯量化方法**（Q-VDiT, ViDiT-Q, QuaRot, SmoothQuant, PTQ4DiT, Q-DiT）：对 video DiT 做 PTQ 量化（W6A6 或 W4A8），使用 channel-wise weight 量化 + token-wise dynamic activation 量化，block-wise 校准。保持 100% attention density，仅通过量化压缩模型存储和计算。(b) **纯稀疏化方法**（SVG/SparseVideoGen, DiTFastAttn/DFT, Jenga）：保持 FP16 权重，仅通过空间-时间 attention mask 裁剪冗余 attention 计算（通常 15%-40% density），减小 attention 计算量。(c) **Naive 量化+稀疏化组合**（QuaRot+SVG, Q-VDiT+SVG 等）：直接将现有量化方法和稀疏 attention mask 叠加使用，不做任何协同优化。
  - Baseline 的全栈执行例子（以 Q-VDiT+SVG naive 组合在 W4A8 + 15% density 下推理 HunyuanVideo-13B 为例）：
    - **算法pipeline**：HunyuanVideo-13B FP16 → Q-VDiT PTQ 校准（temporal distillation + block-wise 量化优化，W4A8）→ 推理时每步：Q_quant = Q(X)Q(W_q)^T, K_quant = Q(X)Q(W_k)^T, V_quant = Q(X)Q(W_v)^T → softmax(Q_quant K_quant^T/√d_k ⊙ M_SVG)V_quant → 输出。**核心缺陷**：量化噪声 ϵ 注入 QK dot product 产生系统性偏差 δ，稀疏 mask M 删除部分 attention connection，两者叠加产生 amplified attention shift（Δ_total = Δ_sparse + Δ_quant + O(||ϵ||·||M||_0)），实际测量的 attention MSE 为 0.685（远大于单独量化 0.216 和单独稀疏化 0.134 之和），导致视频质量严重退化（HunyuanVideo-13B W4A8 15% density: Q-VDiT+SVG 仅 16.66 PSNR vs FP 基线无限值）。
    - **系统框架**：PyTorch + CUTLASS INT matmul kernel。论文未修改 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：CUTLASS INT8 Tensor Core GEMM 处理量化 matmul + 标准 cuBLAS FP16 attention kernel 处理稀疏 attention（masking 后仍走 dense GEMM 路径，无稀疏硬件加速支持）。
    - **硬件架构**：单 NVIDIA A800 80GB。Q-VDiT+SVG W4A8 15% density: DiT time 687s (1.84× speedup vs FP 1264s); 存储 6.50GB (3.68× 压缩 vs FP 23.88GB); 但 PSNR 仅 16.66，视频质量不可接受。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuantSparse 提出两大核心技术解决 "amplified attention shift"：(1) **MSAD（Multi-Scale Salient Attention Distillation）**——在校准阶段通过双尺度蒸馏对抗量化导致的 attention 偏差：Global Guidance 用平均池化下采样 Q/K（stride s=128）后在低分辨率上蒸馏全局 attention 结构（内存从 O(L²) 降至 O(L̃²)，s=128 时仅 0.14GB），Local Guidance 利用 token saliency 呈重尾分布的特性（<10% tokens 占据大部分 attention mass），仅对 top-k=256 salient queries 做高分辨率 attention 蒸馏（内存约 O(kL)）。两者结合在几乎无额外校准开销下（+0.8% GPU memory, +1.6% time）大幅对齐量化 attention 与 FP attention。(2) **SSAR（Second-Order Sparse Attention Reparameterization）**——针对量化后一阶残差 Δ_quant^(t)（即全 attention 与量化稀疏 attention 之差）因量化噪声 ϵ^(t) 的时间变化而违反时间不变性假设的问题，发现二阶残差 Δ̃_quant^(t)=Δ_quant^(t)−Δ_quant^(t-1) 具有显著更高的时间稳定性（因 diffusion process 中 ϵ^(t) 为缓变随机过程，相邻时间步噪声分布相似，ϵ^(t)−ϵ^(t-1) 近似平稳）。在推理时缓存参考步的一阶+二阶残差并复用（仅需额外一次矩阵加法），用 SVD 投影到 top-r=16 主成分进一步抑制时间方差（几乎无额外开销）。
  - 全栈执行例子（QuantSparse W4A8 + 15% density on HunyuanVideo-13B, A800）：
    - **算法pipeline**：HunyuanVideo-13B FP16 → 校准阶段：block-wise PTQ + MSAD 蒸馏（Global: Q̃,K̃=AvgPool(Q,K,s=128)→MSE(Ã_FP||Ã_quant)；Local: top-k=256 salient queries→MSE(A_FP[I]||A_quant[I])）→ AdamW 优化量化参数 → 吸收量化参数得 M_quant → 推理阶段：每步计算量化稀疏 attention A_s,q^(t)=softmax(Q_quant K_quant^T/√d_k ⊙ M_SVG)V_quant → 在 cache interval τ=5 内复用缓存残差 Δ_quant^(t_ref)+Δ̃_quant^(t_ref)（一阶+二阶，一次加法叠加于稀疏输出）→ 超过 interval 时刷新缓存：计算全 attention→更新 Δ_quant 和 Δ̃_quant→SVD 取 r=16 主成分→更新缓存。**直接修复 baseline 缺陷**：MSAD 在校准阶段将 quantized attention 对齐到 FP attention（attention MSE 从 0.685 大幅降低），SSAR 在推理阶段通过二阶残差校正恢复稀疏化丢失的低值但关键的 attention connection。效果：HunyuanVideo-13B W4A8 15% density → PSNR 20.88（vs Q-VDiT pure quant 16.85, vs Q-VDiT+SVG naive 16.66），VQA 81.19（vs FP 81.23），几乎无损。
    - **系统框架**：论文未修改 Serving 框架，在 PyTorch 层面实现。可组合 SageAttention（W8A8 attention 量化）+ TeaCache（latent caching）进一步加速至 2.47×。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：CUTLASS INT matmul kernel 处理量化后的 INT8 GEMM。稀疏 attention mask 与量化权重/激活的联合计算无专用 kernel（masking 后走 dense GEMM）。SSAR 残差校正仅需矩阵加法，`torch.add` 开销可忽略（+0.2% DiT time）。额外显存开销：cached residuals ~2× attention output size（一阶+二阶合并存储），Wan2.1-14B 多 ~2.1GB（26.04→28.14GB）。
    - **硬件架构**：单 NVIDIA A800 80GB GPU，CUDA 12.4。HunyuanVideo-13B W4A8 15% density: 存储 6.49GB（3.68× 压缩 vs FP 23.88GB），显存 27.02GB（1.32× 节省 vs FP 35.79GB），DiT time 671s（1.88× 加速 vs FP 1264s）。Wan2.1-14B W4A8 15% density: 存储 7.00GB（3.80× 压缩），DiT time 2315s（1.74× 加速 vs FP 4031s）。

## Quamba2 A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

- baseline方法是什么？
  - Baseline 方法：(a) **Quamba**（Chiang et al. 2025）：仅支持 W8A8，仅 Mamba1 backbone。使用 percentile clipping 量化 SSM 输入 x_t，对 output projection input 应用 online Hadamard transform 消除 outlier。不支持 4-bit 权重、不支持 Mamba2、不做 head-to-toe 量化（embedding 和 lm_head 保持 FP16）。(b) **MambaQuant**（Xu et al. 2025）：支持 W8A8 和 W4A8，但仅 Mamba1 backbone。使用 variance-aligned rotation 方法量化。W4A8 下性能显著下降（Mamba1-2.8B W4A8 58.5% avg vs FP16 62.2%）。
  - Baseline 在模型推理全栈的执行例子（以 Quamba W8A8 Mamba2-8B 为例）：
    - **算法pipeline**：FP16 Mamba2-8B → 收集 calibration stats → percentile clipping 量化 x_t（在线执行）→ online Hadamard transform on output proj input → W8A8 per-tensor/per-channel 量化 weights → embedding/lm_head 保持 FP16。量化粒度粗（per-tensor/channel），导致 Mamba2-8B W8A8 仅 64.8% avg accuracy（vs FP16 70.7%），差距 5.9%。
    - **系统框架**：论文未明确说明 Serving 框架集成。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 INT8 GEMM kernel + online FWHT kernel + percentile clipping kernel（在线排序+截断，引入额外延迟）。无 4-bit kernel 支持。
    - **硬件架构**：NVIDIA A5000 GPU。W8A8 Mamba2-8B TPOT = 14.12ms, TTFT = 124.01ms。FP16 embedding/lm_head 阻止在 Orin Nano 8G 部署（OOM）。
  - Baseline 的核心缺陷：(1) **bit-width 单一**：Quamba 仅 W8A8，MambaQuant W4A8 精度差，无法覆盖不同部署场景（W4A8 大 batch 高吞吐 vs W4A16 单用户低延迟）；(2) **Mamba2 精度差**：Quamba 的 clipping+Hadamard 在 Mamba2 上效果差（W8A8 仅 64.8% vs FP16 70.7%），因为未利用 SSD 的 channel order preserving 和 activation persistence 特性；(3) **不做全模型量化**：embedding/lm_head 保持 FP16 导致显存瓶颈，无法在边缘设备部署；(4) **SSM 输入参数量化粗糙**：per-tensor/channel 量化 x_t/B_t/C_t 对 SSM 线性递归的误差极度敏感。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Quamba2 提出两大核心技术：(1) **Sort-and-cluster** 量化 x_t：利用 SSM 的 **channel order preserving**（SSD 计算 channel-wise，输入 channel 顺序=输出 channel 顺序）和 **channel persistence**（各 channel 激活幅度在不同输入间保持一致），先 offline 校准 channel max → 排序 channel → 聚类 head（m 组）→ 每组 head 内再聚类 channel（n 组）→ 共 m×n 个 scaling factor 精细量化 x_t 到 8-bit，配合 offline cluster-aware weight reordering 保证计算正确性。(2) **Per-state-group quantization** 量化 B_t/C_t：利用 **state persistence**（B 和 C 中激活的 state group 在时间步和样本间一致），对每组 state group 使用独立 scaling factor，大幅提升小数值 group 的量化精度。
  - 全栈执行例子（Quamba2 W4A8 Mamba2-8B on A5000）：
    - **算法pipeline**：FP16 Mamba2-8B → Pile 512 句 calibration → (a) 记录 x 各 channel max，排序，对 head 聚类 m=4 组，每组内 channel 聚类 n=4 组 → 16 个 scale per layer 量化 x_t；(b) 记录 B/C state group 激活模式 → per-state-group scale；(c) offline cluster-aware reorder: W_in 列/W_out 行/W_conv channel/W_norm 全部按 cluster indices 重排；(d) offline Hadamard fusion: W_in^H=W_in@H^T, W_out^H=H@W_out@H^T；(e) GPTQ 优化 4-bit weights per-group；(f) W4AX 进化搜索自动选择每层 W4A8/W4A16。效果：W4A8 Mamba2-8B 69.1% avg（vs FP16 70.7%, -1.6%），W4A16 69.8%（-0.9%）。
    - **系统框架**：集成 vLLM，替换所有 projection/SSD/conv/embed 层为 Quamba2 量化 kernel。支持 head-to-toe 量化使 Mamba2-8B 部署在 Orin Nano 8G（13 TPS）。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：W4A8 input proj: 4-bit weights + 8-bit acts → per-group dequant → INT8 Tensor Core matmul → fused output scale → INT8 output。SSD kernel: 8-bit A/B/C/x/cached states 加载 → INT8 scan → 写回 8-bit cached states（HBM traffic 减半）。FWHT kernel: 内联 scaling factor s_y 避免额外计算。W4A16 gen: 4-bit weights → dequant → FP16 matmul（memory-bound, TPOT 7.58ms）。W4A8 gen: 8-bit INT8 matmul（TPOT 7.43ms + state 压缩带来更大 batch 支持）。
    - **硬件架构**：NVIDIA A5000 GPU 24GB (cloud), Orin Nano 8G (edge)。Quamba2 支持所有 bit-width 在 roofline model frontier：小 batch→W4A16 memory-bound 最优，中 batch→W4A8 平衡，大 batch→W8A8 compute-bound 吞吐最高。4× memory reduction（15.7GB→1.4GB for Mamba2-2.7B W4A8），1.3× prefilling speedup，3× generation speedup。

## QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

- baseline方法是什么？
  - Baseline：BF16 LoRA RL 训练——使用 BF16 精度的预训练 LLM 权重，仅训练 LoRA adapter（rank=32），通过 GRPO/DAPO 进行 RL。BF16 LoRA 的缺陷：(1) **rollout 速度慢**：BF16 模型在推理时需要 15.2GB（7B），受限于显存带宽和计算量，rollout 吞吐量仅 115.4 tokens/s（batch=2, H100）；(2) **内存占用大**：7B 模型 15.2GB，32B 模型 62.3GB，无法在单 H100 80GB 上做 batch≥4 的 RL 训练（OOM）；(3) **收敛慢**：需 500+ steps 才能看到 reward 上升（BigMath 上的 7B 模型），因为 BF16 模型采样熵低，探索不足；(4) **对高学习率敏感**：LR>5e-6 就训练不稳定/崩溃；(5) **QLoRA (NF4) 更慢**：虽然内存更小（5.7GB），但 NF4 的 unpack+lookup table 操作使 rollout 比 BF16 还慢 0.7×−0.8×。
  - 全栈执行例子（Baseline: BF16 LoRA + GRPO on Qwen2.5-7B-Instruct, single H100）：
    - **算法pipeline**：预训练 Qwen2.5-7B-Instruct BF16 权重 → 添加 LoRA adapter (rank=32, α=32) → policy model 与 reference model (BF16 副本) 并存于 GPU → rollout 阶段 BF16 FP16 GEMM 推理 → 生成 G=8 个候选 → 计算 reward + group advantage → KL 散度约束 → AdamW-8bit 更新 LoRA → 每步约 600ms rollout + 200ms 其他计算，总 BF16 模型 15.2GB + LoRA + optimizer state ≈ 30GB+。
    - **系统框架**：论文未明确说明特定 Serving 框架用于 baseline rollout，但暗示使用 vLLM 作为 rollout 引擎；训练框架为自研 GRPO/DAPO 实现。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：BF16 推理使用标准 cuBLAS GEMM kernels，无特殊优化。NF4 baseline（QLoRA）使用 NF4→FP16 转换的 dequant kernel + cuBLAS GEMM。
    - **硬件架构**：NVIDIA H100 80GB GPU，Tensor Cores 执行 BF16×BF16 GEMM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QeRL = NVFP4 权重量化 + LoRA + Adaptive Quantization Noise (AQN)。解决 Baseline 缺陷的方式：(1) **NVFP4 加速 rollout**：用 NVFP4 量化权重（7B 仅 5.9GB，~61% 显存节省），结合 Marlin kernel 的 NVFP4×BF16 快速矩阵乘法，实现 1.2×−1.5× rollout 加速（batch=8 时从 1641→2091 tokens/s）；(2) **内存节省支持更大模型**：32B 仅 20.7GB，首次实现在单 H100 80GB 上训练 32B 模型（BF16 需 62.3GB，OOM）；(3) **量化噪声增强探索**：核心发现——量化误差 Δϵ = Q(θ)−θ 系统性增加采样熵 H(π(|q))，使输出分布更"平坦"，鼓励探索更广的 token 空间，从而在 BigMath 上仅需 ~200 steps 即可看到 reward 快速增长（vs BF16 LoRA 的 500+ steps）；(4) **AQN 动态噪声调度**：静态量化噪声对后期训练不利→引入 AQN，通过 RMSNorm 注入可控高斯噪声 Z_noisy∼N(0,σ²I)，按指数衰减 σ(k)=σ_start×(σ_end/σ_start)^((k-1)/(K-1)) 从探索过渡到利用，使 7B QeRL+AQN 在 GSM8K 达到 90.8%（vs BF16 LoRA 88.1%）；(5) **高学习率鲁棒性**：量化噪声的稳定化效应使 QeRL 能在 LR=3e-5 下稳定训练，reward 增长速率接近 BF16 LoRA 的 2 倍。
  - 全栈执行例子（QeRL: NVFP4+LoRA+AQN + GRPO on Qwen2.5-7B-Instruct, single H100）：
    - **算法pipeline**：Qwen2.5-7B BF16 → AWQ calibration (OpenThoughts-114k) → NVFP4 量化：\tilde{W} (4-bit) + S_FP32 + S_E4M3(E4M3, block=16) → 添加 LoRA adapter (rank=32, α=32) → QeRL pipeline：step 1. 判断 stage k，计算 σ(k)（stage 0=0 无额外噪声）；step 2. AQN 注入：Z_noisy∼N(0,σ²I) → w_noise = Z_noisy+w → RMSNorm_noise(x)=w_noise⊙x/√(mean(x²)+δ)（等价于乘法噪声 (Z_noise/w+I)⊙\hat{W}）；step 3. rollout：policy model（NVFP4 量化+AQN 噪声）生成 G=8 个候选，Marlin kernel 执行 NVFP4×BF16 GEMM；step 4. 计算 group advantage A_i = (r_i−mean(r))/std(r)；step 5. 仅更新 LoRA A,B（量化权重冻结）。模型 5.9GB+LoRA+optimizer≈~12GB，远小于 BF16 baseline。
    - **系统框架**：vLLM 作为 rollout 引擎（memory utilization=0.30 for 7B），训练框架为自研 GRPO/DAPO 实现。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：Marlin kernel 加速 NVFP4×BF16 推理——kernel 从 global memory 加载 4-bit packed weights 到 shared memory → dequant: ŵ=S_FP32×S_E4M3[block]×unpack_4bit(w̃) → BF16 GEMM → 加 LoRA 输出。AQN 噪声经 RMSNorm 注入，不破坏 kernel 的 NVFP4×BF16 op 路径。
    - **硬件架构**：NVIDIA H100 80GB GPU，Tensor Cores 执行 BF16 GEMM + Marlin 优化的 NVFP4 dequant+compute。

## QT-DoG: Quantization-aware Training for Domain Generalization

- baseline方法是什么？
  - Baseline 方法：标准 ERM (Empirical Risk Minimization) 在 DomainBed 框架下训练 ResNet-50。ERM 使用 Adam optimizer (lr=5e-5, no weight decay)，batch size=32 per-domain，在多源域上最小化交叉熵损失，每 300 步在源域验证集上选择最优模型。ERM 的核心缺陷是：标准优化器（SGD/Adam）往往收敛到损失景观中的尖锐极小值（sharp minima），导致对源域过拟合、OOD 泛化能力差。此外，ERM 在训练过程中 OOD 性能高度不稳定（Figure 4 所示），模型选择不可靠。
  - 全栈执行例子（Baseline: ERM + ResNet-50 on PACS, 全精度 FP32）：
    - **算法pipeline**：ImageNet 预训练 ResNet-50 → 对每个目标域训练 5000 步，Adam 优化交叉熵损失 → 每 300 步验证选模 → 推理时单次前向传播，所有参数 FP32，模型体积 25M × 4 bytes = ~100MB，推理延迟 34.28ms (AMD EPYC 7302 CPU)。
    - **系统框架**：DomainBed 框架 (Gulrajani & Lopez-Paz, 2021)，PyTorch 1.10.0。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 PyTorch CUDA 卷积和全连接 kernel，FP32 精度。
    - **硬件架构**：单张 NVIDIA A100 GPU。无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QT-DoG 将量化感知训练（QAT）作为隐式正则化器引入 ERM 训练流程。核心洞见：量化将连续权重空间离散化为有限集合，等价于在权重上引入均匀分布的加性噪声 Δ ∈ [−s/2, +s/2]（Figure 2 验证 KL 散度仅 0.0009）。通过二阶 Taylor 展开 L(w+Δ) ≈ L(w) + ∇L(w)^T Δ + ½Δ^T H Δ 可知：在尖锐极小值区域（H 特征值大），小扰动 Δ 导致损失急剧上升，促使优化器"逃离"尖锐区域，收敛到平坦极小值（H 特征值小），Δ 的影响可忽略。平坦极小值区域对应低复杂度网络，对输入扰动不敏感，OOD 泛化更强。单模型 QT-DoG (7-bit) 在 PACS 上 87.8% vs ERM 84.7%，同时模型体积压缩 4.6x。EoQ 集成 5 个 7-bit 量化模型，总参数量仅 1.1x 全精度单模型，却在全部 5 个 DomainBed 数据集上超越所有 DG 方法（Avg 68.4% vs DiWA 68.0% / EoA 68.0%）。
  - 全栈执行例子（QT-DoG + LSQ 7-bit on ResNet-50, PACS）：
    - **算法pipeline**：
      1. 前 2000 步：标准 ERM 全精度训练（预热阶段，学习良好的权重初始化）。
      2. 第 2000 步起：对除最后一层外的所有卷积/全连接层启用 LSQ 量化——
         ```
         for weight W in layers[:-1]:
           s = learnable_per_channel_scaling  # QAT 学习最优量化步长
           W_bar = round(clip(W/s, -64, 63))  # 7-bit signed: Q_N=64, Q_P=63
           W_q = W_bar * s                     # 量化权重（INT7 存储，FP 计算）
         # Forward: y = Conv(x, W_q)  # 使用量化权重前向
         # Backward: STE 梯度直通 round() 操作
         ```
      3. 量化噪声 Δ = W_q - W 在每次前向时注入，迫使优化器寻找对 Δ 不敏感的平坦区域。
      4. 继续训练至 5000 步（DomainNet 15000 步），每 300 步验证选模。
      5. 推理：直接使用 INT7 量化权重，前向传播等价于标准模型但权重精度降低。
      6. EoQ：独立训练 5 个模型，集成输出 softmax 平均。
    - **系统框架**：DomainBed 框架 + PyTorch + LSQ 量化模块。训练超参与 ERM baseline 完全相同（Adam, lr=5e-5, batch=32），仅额外加入 QAT 逻辑。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 PyTorch CUDA kernel，量化后推理使用 INT8 内核加速（如 torch.fx 或 TensorRT 等），论文在 AMD EPYC 7302 CPU 上实测 INT8 推理延迟 21.02ms vs FP32 34.28ms。
    - **硬件架构**：单张 NVIDIA A100 GPU 训练，CPU 推理延迟测试使用 AMD EPYC 7302。无定制硬件。
  - **Baseline 缺陷 → 方法设计映射**：
    - (i) ERM 收敛到尖锐极小值，OOD 泛化差 → QAT 引入的量化噪声 Δ 通过 Hessian 交互惩罚尖锐区域（½Δ^T H Δ 项在 H 大时导致损失激增），迫使优化器自然收敛到平坦极小值。Figure 3 验证 QT-DoG 的训练/测试平坦度优于 ERM、SAM、SWA，与 SWAD 相当但模型缩小 75%。
    - (ii) ERM 训练过程中 OOD 性能高度不稳定（Figure 4），模型选择不可靠 → QAT 的正则化效应在 2000 步量化后显著稳定 OOD 准确率曲线，使源域验证集上的最优模型真正对应 OOD 最优模型。
    - (iii) Ensemble DG 方法（DiWA 需训练 60 个全精度模型，EoA 需 6 个全精度模型）内存和计算开销巨大 → EoQ 利用量化模型体积缩小 4.6x 的优势，集成 5 个模型总内存仅 1.1x 单全精度模型，训练计算量减少 12x（vs DiWA）。
    - (iv) PTQ（OBC）无法提升 DG 性能（Table 4: OBC 83.7% vs No quant 84.7%）→ 证明关键在于训练过程中的量化噪声引导优化（QAT），而非仅推理时压缩（PTQ）。只有 QAT 允许模型在噪声存在下重新优化到平坦极小值。

## QERA: an Analytical Framework for Quantization Error Reconstruction

- baseline方法是什么？
  - Baseline 方法：LoftQ（QPEFT 场景）和 ZeroQuant-V2（PTQ 场景）——两者都使用截断 SVD 对权重量化误差 (W - W̃) 进行低秩近似来重建量化误差：C_k = SVD_k(W - W̃)。这种方法最小化的是权重逼近误差 ||W - W̃ - C_k||_F，即 Frobenius 范数下的最优低秩逼近（Eckart-Young-Mirsky 定理）。LoftQ 在此基础上增加了迭代优化（Algorithm 1），交替更新量化权重和低秩项。LQ-LoRA 进一步引入启发式的行列缩放矩阵 D_row, D_col。
  - 全栈执行例子（Baseline: ZeroQuant-V2 / LoftQ on LLaMA-2-7B, 4-bit, rank=32）：
    - **算法pipeline**：对每个线性层权重 W ∈ R^{m×n}，先量化 W_q = q(W)（如 MXINT block size=32），反量化 W̃ = dq(W_q) → 计算权重量化误差 E = W - W̃ → 对 E 做截断 SVD：U, Σ, V^T = SVD(E)，取前 k 个奇异值/向量 → A_k = U_{:,:k}√Σ_{:k,:k}, B_k = √Σ_{:k,:k}V_{:k,:}^T → 前向传播 y = x(W̃ + A_k B_k) → 推理时合并 C_k = A_k B_k 不增推理开销。LoftQ 迭代 T=5 次，每轮用 W - A_k B_k 替代 W 重新量化。关键缺陷：最小化权重误差不等于最小化输出误差。实验证明（Figure 1），LoftQ 迭代数增加时所有层权重误差单调降，但模型输出误差不一定降甚至可能升（如 LoftQ 5-iter vs 3-iter 在 rank k=8 时输出误差更大）；rank 增加也不保证输出误差单调降（如 rank 16 输出误差 > rank 4）。
    - **系统框架**：PyTorch + HuggingFace Transformers + PEFT，GPU 上执行。LoftQ 迭代需反复量化-反量化-SVD，计算开销随迭代数线性增长。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。使用标准 PyTorch CUDA 矩阵乘法和 SVD 算子。
    - **硬件架构**：NVIDIA A100 80GB / A6000 48GB GPU。无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QERA 重新审视 QER 问题的优化目标，指出应最小化层输出误差（Problem 2: min E[||x(W̃ + C_k) - xW||₂²]）而非权重误差（Problem 1: min ||W - W̃ - C_k||_F）。通过严格的数学推导（Theorem 1 和 Theorem 2），给出两个闭式解：(1) QERA-exact：使用输入自相关矩阵 R_{XX} = E[x^T x] 的矩阵平方根对标度化后的权重量化误差做 SVD 再反标度化；(2) QERA-approx：在"不同嵌入维度不相关"假设下，将 R_{XX} 简化为对角矩阵 S²，大幅降低计算开销。两个解都对任意量化函数 q(·) 适用，且 QERA-approx 从理论上解释了 LQER 启发式标度的成功和失败原因。
  - 全栈执行例子（QERA on LLaMA-2-7B, 4-bit, rank=32）：
    - **算法pipeline**：
      1. 校准阶段：对校准集 X 中所有输入 x ∈ R^{1×m} 累积计算：(a) QERA-approx: s_sq[i] = E[x_i²]，构建对角标度矩阵 S = diag(√E[x₁²], ..., √E[x_m²])；(b) QERA-exact: R_{XX} = E[x^T x] ∈ R^{m×m}（FP64 精度累积外积），计算矩阵平方根 R_{XX}^{1/2}（blocked Schur algorithm, CPU, SciPy）。
      2. 量化权重：W_q = q(W), W̃ = dq(W_q)。
      3. 标度化误差：Q = S(W - W̃)（approx）或 Q = R_{XX}^{1/2}(W - W̃)（exact）。
      4. 截断 SVD：U, Σ, V^T = SVD(Q)，取前 k 个分量。
      5. 反标度化：A_k = S^{-1}U_{:,:k}（approx）或 A_k = (R_{XX}^{1/2})^{-1}U_{:,:k}（exact），B_k = Σ_{:k,:k}V_{:k,:}^T。
      6. 前向/推理：y = x(W̃ + A_k B_k)，低秩项可预合并。
      7. 结果：相比 ZeroQuant-V2，4-bit LLaMA-3.1-70B 上 6 个下游任务平均 Δacc = +2.97%，WikiText2 Δppl = -0.38；3-bit LLaMA-2-7B 上 Δppl = -2.65（ZeroQuant-V2 13.00 → QERA-exact 10.67）。QPEFT 中 2-bit RoBERTa-base @ GLUE Δacc = +6.05% vs LoftQ；微调收敛速度更快（Figure 2）。QERA 的模型输出误差随 rank 单调递减（而 LoftQ 不单调），且更多校准样本一致提升 QERA 性能（而 LQER 随机波动）。
    - **系统框架**：PyTorch + Transformers + PEFT + Accelerate（训练/微调），SciPy blocked Schur（矩阵平方根），lm-evaluation-harness（评测）。矩阵平方根计算在 CPU 上执行（FP64），GPU 加速矩阵平方根将是未来优化方向。QERA-approx 初始化时间约 21s-30min（模型规模相关），QERA-exact 约 1.6min-4.9h。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。推理时 QERA 的 A_k, B_k 预合并入 W̃，与 baseline 使用相同的矩阵乘法 kernel，无额外运行时开销。
    - **硬件架构**：NVIDIA A100 80GB GPU × 4（QPEFT）/ A6000 48GB GPU × 8（PTQ），AMD EPYC CPU。无定制硬件。
  - **Baseline 缺陷 → 方法设计映射**：
    - (i) Baseline 最小化权重误差 ||W - W̃ - C_k||_F（Problem 1），但不能保证降低模型输出误差 → QERA 重新定义优化目标为最小化层输出误差 E[||x(W̃ + C_k) - xW||²]（Problem 2），并严格推导出闭式解。
    - (ii) LoftQ 迭代增加和 rank 增加不保证输出误差单调降（Figure 1）→ QERA 的模型输出误差随 rank 单调递减，消除了迭代的不确定性，无需迭代算法。
    - (iii) LQER 启发式标度 S 导致校准样本数增加时模型性能随机波动（Figure 3 purple curve）→ QERA-approx 从理论上推导出正确的 S = diag(√E[x_i²])（而非 LQER 的 E[|x_i|]），使更多校准样本一致提升性能直至收敛（Figure 3 green curve）。
    - (iv) Baseline 在低比特（2/3-bit）下精度崩溃（如 QLoRA 2-bit CoLA Matt=0, LoftQ=3.43）→ QERA-exact 使用完整的 R_{XX} 信息，在 2-bit RoBERTa @ CoLA 上达到 Matt=26.43，3-bit LLaMA-2-7B QERA-exact PPL=10.67 vs ZeroQuant-V2 13.00。


## QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

- baseline方法是什么？
  - Baseline 方法：QLoRA（Dettmers et al. 2023a）——将预训练权重从 FP16 量化为 NF4 精度，在 NF4 精度上添加 LoRA 适配器 (A, B) 进行微调。微调后将 LoRA 权重 s·AB 加回量化权重 W̃，使最终模型恢复为 FP16 精度。若想获得量化推理模型，需对合并后的 FP16 模型做 PTQ（如 GPTQ），导致不可控的精度损失。QLoRA 仅解决了微调阶段的显存节省问题，推理阶段仍需 FP16 或承受 PTQ 精度损失。
  - 全栈执行例子（Baseline: QLoRA on LLaMA-7B, Alpaca, INT4 inference）：
    - **算法pipeline**：预训练权重 W_{FP16} 经由 bitsandbytes 量化为 NF4 格式 → 添加 LoRA A∈R^{D_in×r}, B∈R^{r×D_out} 随机初始化 → 加载 Alpaca 52K 数据，以 frozen NF4 W 和可训练 AB 做 LM 交叉熵损失微调 → 微调后合并 W' = W̃_{NF4} + s·AB → 反量化回 FP16 → 需要时再做 GPTQ 后训练量化（INT4），此步骤 PTQ 产生的量化误差无法通过微调补偿，在低比特（INT3/INT2）下尤其严重（如 LLaMA-7B INT2: MMLU 5-shot 仅 25.0-25.8%）。
    - **系统框架**：HuggingFace Transformers + PEFT + bitsandbytes，LoRA rank r（论文未明确说明具体值），Tesla V100 GPU。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：NF4 缺乏 CUDA 算子优化，INT4 有 CUDA 优化的矩阵乘法算子。NF4 微调速度慢于 INT4。
    - **硬件架构**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QA-LoRA 引入分组操作（group-wise operators），通过两个核心设计同时解决 QLoRA 的两个缺陷：
    1. **分组量化增加量化自由度**：将每列权重 W_{:,j} 划分为 L 组，每组 g = D_in/L 个元素独立计算 α_{l,j} 和 β_{l,j}，量化参数从 D_out 对增至 L×D_out 对，显著降低量化误差。
    2. **分组聚合减少适应自由度**：对输入 x 按组求和聚合 A(x)（维度从 D_in 降为 L），LoRA 矩阵 A 从 D_in×D_int 缩小为 L×D_int（L << D_in），行向量在组内共享。这使得 s·AB 的每列 c_{i,j} 在组内为常数，满足合并后仍可表示为 INT 量化格式的数学条件。
    3. **合并推理保持 INT**：微调后只需更新零点矩阵 β'_{l,j} = β_{l,j} - s·(Σ b_{mid,j}·a_{l,mid})/α_{l,j}，无需反量化到 FP16 也无需 PTQ，推理直接使用 INT 格式。
  - 全栈执行例子（QA-LoRA on LLaMA-7B, Alpaca, INT4）：
    - **算法pipeline**：预训练权重 W 通过 GPTQ 以组大小 g=32 进行 INT4 分组不对称量化（act-order=false, true-sequential=true）→ 每组 g 个元素有独立 (α_{l,j}, β_{l,j}) → 初始化 LoRA A∈R^{L×r}, B∈R^{r×D_out}，L = D_in/g = 4096/32 = 128（相比 baseline 的 D_in=4096 减少 32×）→ 前向传播 y = W̃x + s·(A(x)·g)·A^T B^T, 其中 A(x) 将每组 g 个元素求和降维至 L → 微调后合并，仅更新 β' = β - s·(BA)⊘α → 合并后权重 W' 保持 INT4 格式，推理执行 INT4 矩阵乘法（有 CUDA 算子加速），无需 PTQ。INT4 下 MMLU 5-shot 39.4%（与 QLoRA FP16 的 38.4% 相当甚至更优），INT2 下 27.5%（远超 QLoRA w/ GPTQ 的 25.0%）。
    - **系统框架**：HuggingFace Transformers + PEFT，Tensor V100 GPU（7B 单卡 / 65B 双卡），微调时间仅为 QLoRA 的约 35-55%（7B: 21.5h vs 40.0h；65B: 100.5h vs 284.5h）。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：利用 CUDA 优化的 INT4 矩阵乘法算子（vs QLoRA 的 NF4 无优化算子），推理阶段 INT4 GEMM 比 QLoRA FP16 推理快 50% 以上。
    - **硬件架构**：论文未明确说明。

- baseline方法是什么？
  - **Baseline 方法**：标准后训练量化（PTQ：AWQ、AQLM）和量化感知训练/微调（QAT：LLM-QAT、QLoRA），使用 INT4/INT8 位宽，搭配不同风险等级的校准数据集（Risk-I: UltraChat benign、Risk-II: AOA indirectly harmful、Risk-III: AdvBench directly harmful）进行量化。量化目标以效用（utility）为中心，不专门考虑安全。
  - **全栈执行例子（Baseline: QLoRA INT4 + Risk-I benign calibration on Llama-2-7B-Chat）**：
    - **算法pipeline**：QLoRA 将全精度 Llama-2-7B-Chat 权重 W∈R^{d_in×d_out} 量化到 NF4/FP4 精度得到 Q⁰，再通过 LoRA 低秩适配矩阵 (A∈R^{d_in×r}, B∈R^{r×d_out}) 在 benign 数据集（UltraChat）上微调，损失为因果语言模型标准交叉熵 L_LM = -E log p(y|x)，只优化 A、B 而冻结 Q⁰。量化过程中权重被整体修改以最小化效用损失，但安全相关的权重子空间未得到特殊保护。ASR 从 0.3%（FP16）飙升至 42.3%（INT4 Risk-I），MT-Bench 从 6.65 降至 6.40。
    - **系统框架**：HuggingFace Transformers + bitsandbytes 库实现 4-bit 量化，PyTorch 训练循环。4× A100 40GB GPU 上进行微调，batch inference 输出。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。使用标准 PyTorch CUDA kernel，未引入自定义量化 kernel。
    - **硬件架构**：NVIDIA A100 40GB GPU × 4。无定制硬件。
  - **Baseline 缺陷**：(i) 所有以效用为中心的量化方法都会损害安全能力——AWQ（PTQ w/o FT）使 ASR 从 0.3% 升至 42.4%（INT4），QLoRA（QAT w/ LoRA）在 benign 数据集上 ASR 即已达 42.3%，harmful 数据集下更飙升至 85.3%；(ii) 低 bit-width（INT4 vs INT8）导致更严重的安全退化——3-bit 和 2-bit 下 ASR 可达 67.3% 和 82.0%；(iii) 校准数据集若包含有害样本（Risk-II/III），安全风险急剧放大——AQLM 在 Risk-III 上 ASR=77.4%，远超 Risk-I 的 18.5%；(iv) 现有的安全对齐方法（SFT/DPO 全量微调）虽能恢复安全，但计算开销大——SFT 需 8.4 GPU-hours，DPO 需 9.6 GPU-hours。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：Q-resafe——量化感知的安全修补框架，通过三个核心设计有针对性地恢复量化 LLM 的安全能力，同时保持效用的最小损失：(1) 利用预量化全精度 LLM 的强安全能力作为教师，自动构建 DPO 偏好数据集（y_w 来自全精度模型，y_l 来自量化模型），实现安全知识的蒸馏迁移；(2) 使用 SNIP 分数周期性地识别仅 top-τ% 的安全关键权重（基于连接敏感性 |W_ij · ∇_{Q_ij} L|），而非更新全部权重；(3) 在 LoRA 低秩结构约束下，仅对安全关键权重进行选择性 DPO 更新，其余权重量化后保持不变，避免破坏量化模型的效用。
  - **全栈执行例子（Q-resafe on AWQ INT4 Llama-2-7B-Chat, τ=0.6, r=128, K=1000, benign UltraChat）**：
    - **算法pipeline**：
      1. 数据集构造：对 UltraChat 的每个 prompt x，分别用全精度 Llama-2-7B-Chat（ASR=0.3%）和 AWQ INT4 量化版（ASR=42.4%）生成响应 y_w 和 y_l，构建 200k DPO 三元组。
      2. 周期性安全权重识别（每 K=1000 步）：对当前权重 Q^t 的每层计算 SNIP score = E_x|W_ij · ∇_{Q_ij}(-log p(y|x))|，排序取前 60% 生成 M_Q。该步骤确保随训练进行，安全关键权重子集随模型更新而动态调整。
      3. 选择性 DPO 更新：L_DPO = -log σ(β[log(π_Q(y_w|x)/π_Q⁰(y_w|x)) - log(π_Q(y_l|x)/π_Q⁰(y_l|x))])，仅对 M_A、M_B 掩码位置的 LoRA 参数做 SGD 更新 A^{t+1} = M_A ⊙ (A^t - η∇_A L) + (1-M_A) ⊙ A^t，其余保持零初始化不变。
      4. LoRA 更新后重新量化为 INT4：Q^{t+1} = Q⁰ + Quant(A^{t+1}B^{t+1})，确保修补后模型仍为 INT4 精度，保持内存效率（model size 从 12.6GB FP16 降至 ~2.8-3.5GB）。
      5. 结果：ASR 从 42.4% 降至 1.8%（τ=0.6），MT-Bench 从 6.40 升至 7.14（甚至超过 FP baseline 6.65），仅需 1.2 GPU-hours（vs SFT 8.4h / DPO 9.6h）。在 Risk-III（直接有害数据集）上 ASR 仅 13.6% vs baseline QLoRA 的 85.3%。
    - **系统框架**：PyTorch + HuggingFace Transformers + bitsandbytes。训练使用 4× A100 40GB GPU。推理时量化模型直接加载使用，无需额外全精度模型。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。Q-resafe 在算法层面操作，不涉及自定义 kernel。
    - **硬件架构**：NVIDIA A100 40GB GPU × 4。无定制硬件。
  - **Baseline 缺陷 → 方法设计映射**：
    - (i) 效用为中心的量化损害安全 → 方法引入预量化模型的安全知识蒸馏（y_w vs y_l 偏好对），将安全明确作为优化目标而非副作用。
    - (ii) 低 bit-width 加剧安全退化 → 方法只更新极小部分权重（τ=0.6 时仅 60% 权重参与但通过 LoRA 低秩分解实际参数量极少），不扰动大量已校准的量化权重，在 INT4 下即可将 ASR 从 42.4% 恢复到 1.8%，在 2-bit 下仍能维持 ASR=12.4%（vs QLoRA 82.0%）。
    - (iii) 有害校准数据放大风险 → 方法使用 benign 校准集构建 D_patch（y_w 来自安全的全精度模型），即使校准数据包含有害样本也不直接用于权重优化。
    - (iv) 全量安全微调开销大 → 方法通过 SNIP 识别 + LoRA 低秩更新将 GPU-hours 从 8.4h(SFT)/9.6h(DPO) 压缩至 1.2h（~7-8× 加速），且模型大小保持在量化水平（2.8-3.5GB vs FP16 12.6GB）。


## RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound

- baseline方法是什么？
  - **Baseline 方法**：PQ (Product Quantization) 及其变体 OPQ (Optimized Product Quantization) 和 LSQ (Locally Searchable Quantization)。PQ 将 D 维向量拆分为 M 个子段，对每段做 KMeans 聚类（k=8/4），码本为各子段聚类质心的笛卡尔积。查询时预计算 M 个 LUT（每 LUT 含 2^k 个距离），通过查表和累加估计距离。PQx4fs（k=4）使用 AVX2 SIMD 加速：LUT 量化为 8-bit 整数装入 256-bit 寄存器（每寄存器 2 个 LUT），批量处理 32 个量化码，大幅提升吞吐。LSQ 使用多码本加法量化替代笛卡尔积，追求更高精度但编码为 NP-Hard。
  - **全栈执行例子（Baseline: PQx4fs + IVF @ M=D/2=64, k=4, 总码长 2D=256 bits, 8x 压缩）**：
    - **算法pipeline**：将 D=128 维向量分成 M=64 个子段（每段 2 维），每段 KMeans 聚类 2^4=16 个中心。量化码 = 64×4-bit = 256 bits。查询时：对每子段计算 16 个距离 → 64 个 LUT → 对每个候选量化码 64 次查表累加得估计距离 → 选取估计距离最小的 rerank 个候选 → 取原始向量精确计算距离。PQ/OPQ 在码本构造（启发式优化）和距离估计（直接用量化向量替代数据向量）两个环节均无理论误差界，估计器有偏。
    - **系统框架**：Faiss 1.7.4 开源库，IVF 索引（4,096 聚类）。Raw vectors → IVF 分区 → PQ 量化每条向量为 M×4-bit → RAM。查询：q → 找最近 nprobe 个聚类质心 → 对候选向量批量 FastScan SIMD 估计距离 → re-rank。需要全维度原始向量用于重排序，re-ranking 参数（500/1000/2500）需手工调参。
    - **编译框架**：g++ 9.4.0（或 GCC 11.4.0），-Ofast -march=core-avx2，AVX2 SIMD 指令集。
    - **kernel调度**：FastScan [4,5] 将 LUT 装入 SIMD 寄存器，通过 shuffle 指令并行查表。未修改底层 kernel，直接使用 Faiss 的实现。
    - **硬件架构**：AMD Threadripper PRO 3955WX CPU（Zen2 架构），64GB RAM。无 GPU/加速器。
  - **Baseline 缺陷**：(i) PQ 和其变体在两个环节均无理论误差界——码本构造基于启发式 KMeans（理论无法分析其最优性），距离估计直接用量化向量替代原始向量（有偏且无误差界），导致在未知数据集上可能灾难性失效（如 MSong 数据集上 recall<60%）；(ii) 距离估计器有偏——将量化向量当作数据向量直接计算距离，引入系统性偏差，缺乏理论保证；(iii) 单向量距离估计低效——原版 PQ 依赖在 RAM 中查表（多级指针间接访问），虽然 FastScan SIMD 解决了批量场景，但对单向量场景不可用（需将量化码打包并重排布局）；(iv) re-ranking 参数需要经验性调参——PQ 通过固定超参数（如 rerank=500/1000/2500）决定重新排序的候选数，该参数因数据集而异，无法预测最优值；(v) k=8→k=4 不总是可行——从 k=8 降到 k=4 以适配 SIMD 寄存器时，某些数据集（如 MSong）精度灾难性下降，说明基于启发式的方法无法可靠地压缩。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：RaBitQ——通过随机旋转双值向量构造码本，设计无偏距离估计器，并证明严格的 O(1/√D) 概率误差界（渐近最优）。具体包括：(1) 归一化数据向量到单位超球面；(2) 以超立方体顶点 ±1/√D 为确定码本 C，用随机正交矩阵 P 旋转得到 C_rand；(3) 对每个数据向量取 P^{-1}o 的符号位构成 D-bit 量化码；(4) 基于几何关系推导无偏估计器 ⟨o,q⟩ ≈ ⟨ō,q⟩/⟨ō,o⟩，证明其无偏且误差界 O(1/√D)；(5) bitwise 操作（单向量）或 FastScan SIMD（批量）实现高效计算。
  - **全栈执行例子（RaBitQ + IVF @ D=128, 量化码 ~128 bits, ~32x 压缩, ε₀=1.9, B_q=4）**：
    - **算法pipeline**：
      - **码本构造**：C = {±1/√D}^D（超立方体顶点，共 2^D 个向量），采样随机正交矩阵 P∈R^{D×D}（高斯矩阵+QR分解），C_rand = {Px | x∈C}。由于 P 是正交矩阵，C_rand 中的向量在单位超球面上均匀随机旋转分布，无偏好。
      - **量化编码**：对每个数据向量 o（已归一化），计算 o'=P^{-1}o，取符号位：x̄_b[i]=1 if o'[i]≥0 else 0。量化向量 ō = P·((2x̄_b-1_D)/√D)。复杂度 O(D²) 每向量（P^{-1} 矩阵乘法），整体索引时间与 PQ 相当（GIST 上 117s vs 105s）。
      - **几何关系分析**：Lemma 3.1：⟨ō,q⟩ = ⟨ō,o⟩·⟨o,q⟩ + ⟨ō,e₁⟩·√(1-⟨o,q⟩²)，其中 e₁⟂o。⟨ō,o⟩ 期望值约 0.8（浓度集中），⟨ō,e₁⟩ 期望值为 0 且浓度集中。
      - **无偏估计器**：Theorem 3.2：E[⟨ō,q⟩/⟨ō,o⟩] = ⟨o,q⟩（无偏），且 |⟨ō,q⟩/⟨ō,o⟩ - ⟨o,q⟩| = O(1/√D) w.h.p.（渐近最优误差界，对比理论下界[3]）。PQ 的无偏性和误差界均无保证。
      - **距离重构**：dist² = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·⟨ō,q⟩/⟨ō,o⟩。
    - **系统框架**：C++ 实现（独立于 Faiss，代码开源），IVF 索引（4,096 聚类）。Index：raw vectors → IVF 分区 → 每个聚类独立归一化 → RaBitQ 量化 → 存储 D-bit 量化码 + 辅助值。Query：q → 变换 q'（O(D²)，所有向量共享）→ 量化 q̄_u（O(D)）→ 对候选逐个 bitwise 计算距离估计 → 基于 error bound 的剪枝（若下界 > 当前最优精确距离则跳过，无需手工调 rerank 参数）→ 仅对通过的候选取原始向量精确计算 → 返回 NN。在 MSong 上 OPQ 灾难性失效（recall<60%）时，RaBitQ 保持高 recall。
    - **编译框架**：g++ 9.4.0，-Ofast -march=core-avx2，Ubuntu 20.04 LTS。支持 AVX2 SIMD（batch 模式使用 FastScan 相同技术）。
    - **kernel调度**：
      - 单向量：bitwise-and + popcount（对 D-bit 字符串执行 B_q=4 次），平均比 PQ 原版（RAM 查表）快 3× 达到相同精度。
      - 批量（FastScan SIMD）：D-bit 拆分为 D/4 个 4-bit 字段，预计算 D/4 个 LUT（每个含 2^4=16 个值），LUT 装入 AVX2 256-bit 寄存器，shuffle 指令并行查表累加。与 PQx4fs 相同技术栈，但量化码仅 D bits（vs PQ 的 2D bits），因此更高效。
    - **硬件架构**：AMD Threadripper PRO 3955WX CPU（Zen2, AVX2），64GB RAM。查询单线程，索引 32 线程。无 GPU/加速器依赖。
  - **对应关系的核心逻辑**：
    - Baseline 因"码本构造和距离估计均无理论保证"→ 论文用随机旋转双值向量码本 + 基于几何关系的显式分解（Lemma 3.1）设计无偏估计器，证明严格 O(1/√D) 误差界（Theorem 3.2），且该界被证明是渐近最优的（匹配理论下界 [3]）。
    - Baseline 因"有偏估计 + 无误差界导致 MSong 等数据集灾难性失效"→ 论文的估计器无偏且误差界不依赖数据分布（additive bound 对所有数据成立），在所有六个数据集上均稳定工作，最大相对误差 ≤40%（vs PQ/OPQ 在某些数据集上 >100%）。
    - Baseline 因"单向量距离估计需 RAM 查表低效"→ 论文利用量化码为 bit-string 的特性，将距离估计归约为 B_q 次 bitwise-and+popcount（通用硬指令），平均快 3× 达到相同精度。批量场景无缝复用 FastScan SIMD 技术栈。
    - Baseline 因"re-ranking 参数需手工调参"→ 论文的 error bound 提供了置信区间（16），可直接在查询时判定候选是否需要精确计算（下界比较），参数 ε₀=1.9 在全数据集通用固定（由理论分析给出 ε₀=Θ(√log(1/δ))），无需调参。
    - Baseline 因"k=8→k=4 可能精度崩溃（MSong）"→ 论文的误差界随维度 D 增加而减小（O(1/√D)），高维下即使 D-bit 短码也保证精度。默认码长 = 仅 ~D bits（vs PQ 的 2D bits），在更短码长下达到更好精度。

## Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

- baseline方法是什么？
  - **Baseline 方法**：原始 RaBitQ [27] 仅支持高压缩率（32x，每个向量用 D 位表示），以及其简单扩展 RaBitQ(pad)——通过将 D 维向量零填充至 B·D 维再应用 RaBitQ。同时包括传统量化方法：SQ（全局均匀标量量化）、LVQ（逐向量标量量化）和 PQ/OPQ（乘积量化，k=8）。
  - **全栈执行例子（Baseline: RaBitQ(pad) + IVF @ B=4, 8x 压缩）**：
    - **算法pipeline**：将 D 维向量零填充到 B·D 维空间，在扩展空间中构造超立方体顶点码本（±1/√(B·D)），随机旋转后量化。距离估计基于 Lemma 2.1 的无偏估计器，误差界 O(1/√(B·D))——误差随 B 线性根号衰减而非指数衰减，导致即使增加 bit 数也不能有效提高精度。理论要求 Θ(B·D) bits 才能达到误差界 ε，而理论上最优仅需 Θ(D log B) bits。
    - **系统框架**：C++ 实现，IVF 索引（4,096 聚类），Raw vectors → IVF 分区 → RaBitQ(pad) 量化每条向量为 B·D 位 → 存于 RAM。查询时：q → 找最近聚类质心 → 对候选向量批量 FastScan SIMD 估计距离 → 返回最小估计距离的向量。需要访问所有候选的全部 B·D 位才能估算距离，无剪枝。
    - **编译框架**：C++ 由 GCC 11.4.0 编译，-Ofast -march=native，使用 AVX512 SIMD 指令集。
    - **kernel调度**：FastScan [4] 通过 SIMD 批量计算 `<二进制码, query>`，但对 B>1 需额外将 B-bit 整数拆分为多位二进制码分步计算。论文未修改 kernel。
    - **硬件架构**：Intel Xeon Gold 6418H CPU（Sapphire Rapids, 48 cores），1TB RAM。无 GPU/加速器。
  - **Baseline 缺陷**：(i) RaBitQ(pad) 的填充策略将向量升维后再量化，误差界仅 O(1/√(B·D))，B 增加时误差衰减慢，与理论上最优的 O(log B / √D) 存在巨大差距；(ii) 原始 RaBitQ 和 RaBitQ(pad) 需要高压缩率（32x）配合 re-ranking 才能产生合理 recall，不重新排序时 recall 低（<90%），但 re-ranking 需要存储原始向量，违背了压缩省内存的初衷；(iii) SQ/LVQ 在 B<4 时误差比 RaBitQ 大数个数量级，PQ/OPQ 在 B≥4 时精度不如 SQ，且依赖 RAM 查表导致效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：Extended RaBitQ——通过在 D 维空间（而非 B·D 维）中构造包含 2^{B·D} 个向量的码本，将 B-bit 无符号整数网格向量归一化后随机旋转。结合高效量化编码算法（O(2^B·D log D)）和两阶段距离比较（先用 MSB 快速剪枝）。
  - **全栈执行例子（Extended RaBitQ + IVF @ B=5, 6.4x 压缩, >95% recall 无 re-ranking）**：
    - **算法pipeline**：
      - **码本构造**：G = {-(2^B-1)/2 + u | u=0,...,2^B-1}^D（D 维空间中的均匀网格），然后 G_r = {P·y/||y|| | y∈G}。码本向量是随机旋转的单位向量，继承了 RaBitQ 的无偏估计器。误差界理论：∀ε, 仅需 B = Θ(log(ε^{-2}·log(1/δ)/D)) bits，达到渐近最优（对比 RaBitQ(pad) 需要 Θ(ε^{-2}·log(1/δ)) bits）。
      - **量化编码 (Algorithm 1)**：利用 Lemma 3.1 的几何性质（∃ t>0 使得 argmax_y ⟨y/||y||, o'⟩ = argmin_y ||t·o' - y||），仅需枚举至多 D·2^{B-1} 个 critical values（每个维度 i 的临界值 = (x+0.5)/o'[i]），使用最小堆维护 O(log D) 插入/弹出，总复杂度 O(2^B·D log D)。B=7 时百万级 3072 维数据集仅需 ~98 秒。
      - **两阶段距离比较**：ȳ_u 的 MSB ȳ₀ 恰好等于原始 RaBitQ 的二进制码 x̄_b。第一阶段用 FastScan 批量计算基于 ȳ₀ 的粗略距离，误差界已知可判定大部分候选；第二阶段仅对未剪枝候选访问 ȳ_last 增量计算 ⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ₀,q'⟩ + ⟨ȳ_last,q'⟩。距离估计公式：⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u,q'⟩ - (2^B-1)/2 · Σq'[i])。
      - **误差控制**：经验公式 ε < 5.75·2^{-B}/√D（>99.9% 置信度），误差随 B 指数衰减（对比 RaBitQ(pad) 仅 O(1/√(B·D))）。
    - **系统框架**：C++ 实现，IVF 索引。Index: raw vectors → 中心化（每聚类局部质心）→ Algorithm 1 量化 → 分离存储 MSB 和剩余位 → RAM。Query: q → 变换 q' → 找 nprobe 个最近质心 → 第一阶段 FastScan(MSB) → 剪枝 → 第二阶段 剩余位增量计算 → 返回最小估计距离向量。B=5 时 >95% recall（6.4x 压缩），B=7 时 >99% recall（4.5x 压缩），均无需 re-ranking。
    - **编译框架**：GCC 11.4.0，-Ofast -march=native，Ubuntu 22.04 LTS。AVX512 SIMD 批量计算。
    - **kernel调度**：FastScan SIMD 批量计算 ⟨ȳ₀,q'⟩；B=4 或 8 时直接复用现有系统的 4-bit/8-bit 整数与浮点内积实现 [1,17]；其他 B 通过拆分（如 B=9 → 1-bit + 8-bit）实现。论文未修改底层 kernel。
    - **硬件架构**：Intel Xeon Gold 6418H CPU（Sapphire Rapids），1TB RAM。无 GPU/加速器。搜索单线程，索引多线程（96 threads）。
  - **对应关系的核心逻辑**：
    - Baseline 因"升维填充导致误差界 O(1/√(B·D)) 次优" → 论文在原始 D 维空间中构造 2^{B·D} 规模码本，误差界随 B 指数衰减，理论达到 Θ(D log B) 渐近最优。
    - Baseline 因"高压缩率必须 re-ranking 但违反省内存目标" → 论文支持 B=5~7 的中等压缩率（4.5x-6.4x），可独立产生 >95%~>99% recall，无需存储原始向量。
    - Baseline 因"全 bit 参与所有候选评估导致效率低" → 论文利用 MSB=RaBitQ 二进制码的特性，第一阶段用 SIMD 快速剪枝，仅对少数候选做完整距离计算。
    - Baseline 因"SQ/LVQ 精度不足" → 论文在同样 bit 数下 error 比 LVQ 小 1.3x-3.1x（B>6），B<6 时差距更大（可差数量级）。

## PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

- baseline方法是什么？
  - **Baseline 方法**：现有的多步扩散模型量化方法（MaxMin、LSQ、Q-Diffusion、EfficientDM 等）直接应用于 OSEDiff（one-step diffusion SR 模型）。
  - **全栈执行例子（Baseline: OSEDiff + Q-Diffusion PTQ @ W6A6）**：
    - **算法pipeline**：Q-Diffusion 为多步扩散模型设计，依赖多时间步的校准策略（如多步噪声水平采样和跨步一致标定）来量化 UNet，但 OSDSR 仅有一个去噪步（time-step 为常数），这些多步特定技术失效。VAE 保持 FP32 未量化，占据 80%+ 计算量。CLIPEncoder 和 DAPE 等分支模块增加了标定复杂度但 Q-Diffusion 未提供针对性处理。
    - **系统框架**：PyTorch 推理，`LR → VAE_encoder(FP32) → UNet(W6A6) + text_embedding_branch(FP32) → VAE_decoder(FP32) → HR`。VAE encoder/decoder 的 1,781G MACs 全部在 FP32 执行，UNet 的 339G MACs 在 INT6 执行，DAPE 等附加模块约 141G MACs 在 FP32 执行。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。量化推理通过 PyTorch fake-quant 模拟，非真实 INT8 kernel。
    - **硬件架构**：论文未明确说明。运行在单张 GPU（推测 A100 40GB），PyTorch 2.0.1 + CUDA 11.8。
  - **Baseline 缺陷**：(i) VAE 未量化导致整体压缩率低（仅 ~17% params/ops 减少）；(ii) 多步校准策略在单步模型上不适用，甚至产生比 MaxMin 更差的结果（W6A6 时 Q-Diffusion PSNR 仅 19.75 vs MaxMin 15.55 vs PassionSR 25.15 on RealSR）；(iii) 分支模块未经处理，增加标定复杂度和不稳定性。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：PassionSR 是首个面向 one-step diffusion SR 的 PTQ 方法，通过三个创新设计系统性解决上述缺陷。
  - **全栈执行例子（PassionSR W6A6 UNet-VAE 量化）**：
    - **算法pipeline**：
      - **模型简化**：将 OSEDiff 的 DAPE-CLIPEncoder 分支替换为基于空字符串 "" 预计算的 ClipEncoder 常数 embedding，模型简化为 PassionSR-FP = UNet + VAE（参数减少 27.13%, Ops 减少 6.25%, 性能持平）。这使得 VAE 和 UNet 可采用统一校准策略，且 VAE 显式纳入量化范围。
      - **LBQ（Learnable Boundary Quantizer）**：使用可训练上下界 B_l, B_u 替代固定 min/max 范围。量化过程 X_q = α·round(clamp(X,B_l,B_u)-B_l)/α) + B_l，其中 α=(B_u-B_l)/(2^N-1)。仅 B_l, B_u 可训练，通过 STE 反向传播，在极小参数开销下自适应找到最优量化区间。
      - **LET（Learnable Equivalent Transformation）**：对 Linear: W̃=s⊙W, X̃=(X-δ)⊘s, B̃=B+δW；对 Conv: 沿 channel 维相同变换；对 Attention: Q̃=Q⊘s, K̃=s⊙K。s 和 δ 为逐通道可学习参数，训练后 s 融入前层权重/归一化层，δ 融入权重/偏置，零额外推理开销。这解决了 Baseline 中激活离群值导致的量化困难——激活分布从分散/含大量离群值变为集中/友好（Fig. 7），W6A6 下 PSNR 从仅用 LBQ 的 23.15 提升至 25.40（+2.25 dB）。
      - **DQC（Distributed Quantization Calibration）**：Stage 1 冻结 LBQ、仅训练 LET；Stage 2 重新初始化 LBQ、联合训练。这解决了 Baseline PTQ 中量化参数训练不稳定的问题：DQC 使标定时间从 3.87h 降至 1.07h，GPU 显存从 40GB 降至 28GB。
      - **损失函数**：VAE encoder: L_VAE_e = ||V_qe(X_fp) - V_fpe(X_fp)||₂；VAE decoder: L_VAE_d = ||V_qd(X_q) - V_fpd(X_fp)||₂；UNet: L_UNet = ||I(Z_lq, ε_q) - I(Z_l, ε_fp)||₂，其中 I 是从 latent+噪声预测到输出 latent 的变换函数。模块级逐层标定避免跨模块误差累积。
    - **系统框架**：量化推理 `LR → VAE_encoder(INT8) → UNet(INT8) → VAE_decoder(INT8) → HR`，全链路 INT8。W8A8 PassionSR-UV 参数 238M（↓81.77%）、操作 1,060G（↓76.56%）；W6A6 参数 178M（↓86.32%）、操作 795G（↓82.42%）。对比 Baseline W6A6 UNet-only 量化：参数 246M（↓81.11%）、操作 3,689G（↓18.44%），PassionSR 将 VAE 量化纳入后操作压缩率从 18% 跃升至 82%。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。推理通过 `inference_single.py` 执行 PyTorch 模型推理，标定在 `ptq_quantize_single.py` 中实现 fake-quant 模拟。等价变换的硬件兼容性由 AWQ [22] 已确认。
    - **硬件架构**：论文未明确说明。标定 GPU（推测 A100 40GB），PyTorch 2.0.1 + CUDA 11.8。最终部署至移动设备或其他资源受限硬件为目标场景。
  - **对应关系的核心逻辑**：Baseline 因"VAE 不量化 → 压缩率低"，PassionSR 通过模型简化 + LET 统一 UNet/VAE 量化协议解决；Baseline 因"多步技术不适用单步模型 → 性能崩溃"，PassionSR 通过 LBQ 可学习边界 + OSD 专用损失函数解决；Baseline 因"激活离群值 → 量化误差大"，PassionSR 通过 LET 逐通道等效变换 + DQC 稳定两阶段标定解决。

## ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

- baseline方法是什么？
  - **AWQ (Lin et al., 2024b)**：广泛使用的 W4A16 权重量化方法，采用 channel-wise scaling（逐通道缩放因子）对权重进行预处理以抑制离群通道，缩放因子通过 grid search 优化，且可完全合并到前序算子中，推理零开销。缺陷：(1) 仅使用逐通道缩放，无法利用跨通道交互来进一步收窄组内动态范围——当离群值在通道内分散而非集中在特定通道时效果有限；(2) 在推理 LLM 的长链思维（CoT）生成中，每个解码步的量化误差会累积，导致在推理任务（如 MMLU-Pro）上精度显著下降（Qwen3-4B FP16 71.0 → AWQ 68.2）。
  - **QTIP (Tseng et al., 2024b)**：SOTA 向量量化方法，采用随机 Hadamard 变换 + trellis 量化算法。Hadamard 变换是全旋转矩阵（O(n log n) 复杂度），可跨通道交互消除离群值。缺陷：(1) Hadamard 变换固定或由随机向量生成，忽略各层权重分布的独特性；(2) Hadamard 变换仍有较大推理开销（比 AWQ 慢约 30%），因为变换在全局 channel 维度上有依赖关系，无法充分利用 GPU 并行性。
  - **SpinQuant (Liu et al., 2025b)**：将旋转矩阵合并到前序线性层权重中以避免推理开销，但仅适用于少数可合并层（如 output projection），decoder block 中多数线性层前有 element-wise 算子或残差连接，无法吸收矩阵乘法。
  - 全栈执行例子（以 AWQ W4A16 量化 Qwen3-4B 在 MMLU-Pro 上的推理为例）：
    - **算法层**：加载 FP16 权重 W → 校准集上逐 channel 计算激活幅值 s = mean(|X|)（activation-aware）→ grid search 优化逐通道缩放因子 α（缩放范围 [0.5, 1.5]）→ W' = diag(α)·W → INT4 均匀量化（group=128）：s_q = (max(W'_g)-min(W'_g))/15, W_q = round((W'_g-min)/s_q) → 推理时缩放因子合并到前序 LayerNorm/激活 X' = X·diag(1/α)。
    - **系统框架层**：PyTorch + Transformers + vLLM serving → 量化权重存储、GEMM 用 INT4 kernel。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：AWQ 提供高效 W4A16 GEMM kernel（Triton/CUDA），无需额外 transform kernel（channel-wise scaling 已合并）。
    - **硬件架构层**：NVIDIA GPU (RTX A6000/4090/H200)，论文未涉及硬件设计。
  - Baseline 核心缺陷：(1) **仅 scaling 不足以消除复杂离群值模式**——AWQ 的逐通道缩放只调整每个通道的整体幅值，无法处理通道内 token 级别的数值分散；(2) **全旋转过于昂贵**——QTIP/QuIP# 的 Hadamard 变换虽有跨通道交互能力但推理开销大（约 30% slowdown）；(3) **合并旋转的范围受限**——SpinQuant 的旋转合并策略只适用于少数线性层；(4) **推理 LLM 的长生成使误差累积放大**——现有方法在设计时未充分考虑 CoT 生成场景。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **ParoQuant** 通过三个核心设计系统性解决 baseline 的精度-效率权衡困境：
    1. **Scaled Pairwise Rotation（独立 Givens 旋转 + 逐通道缩放）**替代仅 Scaling 或全旋转：
       - 在逐通道缩放（拉平整体幅值）基础上，叠加 K=8 个 **independent rotations**（每个由 group_size/2=64 对互不重叠的 Givens 旋转组成），实现 **稀疏参数化的跨通道交互**——仅旋转幅值差异大的通道对（实验证明 top 10% 关键对的表达能力与全旋转几乎等价，Figure 2）。对应解决缺陷 (1)：既保留 scaling 拉平全局幅值的能力，又通过旋转收窄每对通道内 token 级别的数值分散（Figure 1 Right，数据点聚集到 x=y 线附近）。
    2. **Independent Rotation 约束（无依赖、全并行）**替代 Hadamard 全局依赖：
       - 强制每个 rotation 内的通道对互不重叠（每个通道最多参与一对），使所有 Givens 旋转完全并行化且无需同步。K 个 rotation 按顺序应用，但在一个 fused kernel 内完成（一次加载激活到 shared memory，8 次旋转均在 shared memory 上执行）。对应解决缺陷 (2)：推理开销仅约 10%（vs Hadamard 的 30%），且 channel 维度越大加速比越显著（Figure 4）。
    3. **两阶段逐层优化 + 混合校准集**替代 grid search/固定变换：
       - Stage 1：用 AdamW 优化旋转角度和缩放因子（而非 grid search），基于 2048 个多样化校准样本（WikiText2+C4+RedPajama 均匀混合），最小化量化层输出误差。
       - Stage 2：QAT-like 微调权重和量化参数，进一步消除 Stage 1 后残留的孤立离群值。
       - 逐层使用已量化前层的输出 X' 作为校准输入，使后续层能补偿前层累积的量化误差。对应解决缺陷 (3)(4)：每层独立学习最优变换参数，且考虑长生成中的误差传播。
  - 全栈执行例子（以 ParoQuant W4A16 量化 Qwen3-4B 在 MMLU-Pro 上推理为例，对比 AWQ）：
    - **算法层**：FP16 Qwen3-4B → 分组（group=128）→ 配对选择（Algorithm A1: shuffle 后贪婪选互不重叠 pair，跨 rotation 跳过已选 pair）→ Stage 1: AdamW 优化 θ∈R^{K×64} 和 α∈R^{128}，minimize ||l'(X')-l(X)|| → 量化：s=range/15, z=-round(min/s), W_q=clamp(round(T(W)/s)+z,0,15) → Stage 2: AdamW 微调 W, s, z → 推理：X → T^{-1}(X) = X·diag(1/α)·R_1^{-1}·...·R_K^{-1}（fused CUDA kernel, 3-level parallelism）→ INT4 GEMM (AWQ kernel) → Y。对比 AWQ：AWQ 仅 diag(α)·W → 量化 → 推理（X 直接做 INT4 GEMM，无 transform kernel，α 已合并到前序 op），无旋转、无 Stage 2 微调。
    - **系统框架层**：PyTorch 2.8.0 + Transformers 4.55.2（量化优化）→ PyTorch 2.6.0 + torch.compile max-autotune + CUDA Graphs（推理）→ vLLM 0.10.1 + Lighteval 0.8.1（推理任务评测）。对比 AWQ：统一在 Transformers 框架上仅替换量化层实现。
    - **编译框架层**：torch.compile max-autotune 用于推理图优化，论文未涉及自定义编译器。
    - **kernel调度层**：Fused CUDA kernel（token/group/pair 三级并行，shared memory 常驻）→ AWQ W4A16 GEMM kernel。对比 AWQ：AWQ 无额外 transform kernel（α 已合并），直接调用 GEMM kernel。ParoQuant 多一个 transform kernel（~10% 开销）但换取显著精度提升。
    - **硬件架构层**：NVIDIA H200 (训练)、RTX A6000/6000 Ada/4090 (推理)，论文未涉及硬件设计。
  - 关键结果：
    - 推理任务平均精度：ParoQuant 仅降 0.9%（FP16→W4），vs AWQ 降 3.3%、EfficientQAT 降 7.2%。
    - MMLU-Pro Qwen3-4B：ParoQuant 70.1 vs AWQ 68.2 vs QTIP 69.7。
    - 吞吐：ParoQuant 比 AWQ 慢约 10%，比 QTIP 快约 25%（Qwen3-4B: 160 vs 176 vs 117 tokens/s）。



## PT²-LLM Post-Training Ternarization for Large Language Models

- baseline方法是什么？
  - **GPTQ (Frantar et al., 2023)**：基于 Hessian 的逐块权重量化方法，使用逆 Hessian 信息进行逐列量化误差补偿，支持 2-4 bit 均匀量化。其可选列重排序基于 Hessian 重要性排序。在 PT²-LLM 中作为底层框架使用。缺陷：(1) 未针对三值量化设计——直接对 2-bit 均匀量化进行 Hessian 优化无法处理三值仅 3 个离散值的情况，量化误差大；(2) Hessian 重排序仅考虑重要性、未考虑列间结构相似性，块内权重分布散乱。
  - **TWN (Li et al., 2016)**：经典对称三值量化方法，阈值 Δ ≈ 0.75/n × Σ|W| 基于均匀/正态分布假设近似，缩放因子 α = Σ(T·W)/Σ|T| 逐行解析计算。缺陷：(1) 对称方案假设权重均值为零，LLM 实际权重分布常见非零均值；(2) 阈值和缩放因子均为一次性近似，无迭代优化；(3) 仅优化权重层面 L2 误差，完全忽略激活信息。
  - **PB-LLM (Shang et al., 2024)**：1.7-bit 混合精度 PTQ，部分显著权重量化为 8-bit、其余二值化为 1-bit。缺陷：(1) 非结构化掩码引入额外 1 bit/权重开销，等效位宽实际 >2-bit；(2) 二值化（±1）表达能力弱于三值化（±1,0），后者能更好地匹配 LLM 权重的单峰分布。
  - 全栈执行例子（以 GPTQ 2-bit 量化 + TWN 三值初始化 LLaMA-7B 为例）：
    - **算法层**：加载 FP16 权重 W∈R^{4096×4096} → Hessian 矩阵 H=2XX^T 计算 → 逐列量化：Δ_i = (W_max - W_min)/3（2-bit 均匀量化区间）→ W_q = round(clamp((W - Z)/Δ, 0, 3)) → 量化误差 δ=W-W_q → 剩余列权重补偿 W_rest -= H^{-1}δ → 或使用 TWN 初始化：Δ≈0.75·mean(|W|) → T = sign(W)·1_{|W|>Δ} → α = Σ(TW)/Σ|T|。最终存储：GPTQ 2-bit 需 2 bit/权重 + 量化参数；TWN 1.58 bit/权重仅需 T（2-bit 索引）+ α（fp16/行）。
    - **系统框架层**：PyTorch + HuggingFace Transformers 进行 fake-quantization 仿真。2-bit 以下无原生 GPU 推理支持，仅精度验证。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：llama.cpp 提供 2-bit 量化推理 kernel，三值/1.58-bit 需自定义 kernel（论文使用 llama.cpp，但未说明是否修改或使用何种 kernel 实现 1.58-bit 推理）。
    - **硬件架构层**：Nvidia A800-80GB GPU，论文未涉及硬件设计。
  - Baseline 核心缺陷：(1) **对称三值假设不成立**——LLM 权重普遍存在非零均值，对称 TWN 强制 μ=0 导致大量权重被错误映射；(2) **权重-激活脱节**——GPTQ 和 TWN 仅优化权重层面 ||W-Ŵ||²，与实际推理输出 ||WX-ŴX||² 存在偏差；(3) **块内分布散乱**——固定顺序或 Hessian 重排序无法保证块内列间结构相似性，三值化对散乱分布极为敏感，离群列严重扭曲块内量化范围。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **PT²-LLM** 通过三个互补设计系统性地解决 post-training 三值量化的核心挑战：
    1. **非对称三值量化 (ATQ)** 替代对称三值：引入逐行偏移 μ（初始化为行均值），量化网格变为 {−α+μ, μ, α+μ} 三个非对称值，自然适配非零均值权重分布。对应解决缺陷 (1)：不再强制对称假设。
    2. **ITF 迭代交替优化** 替代一次性阈值/缩放解析：将三值参数 (α, μ, T) 的优化建模为交替最小化问题——固定 T 时闭式求解最优 α*, μ*（Eq. 9，向量化并行逐行求解），固定 (α*, μ*) 时通过 Z_ij = (W_ij-μ_i)/α_i 弹性舍入到最近三值更新 T*（Eq. 10）。两步骤交替贪婪减小 E_w，约 10 轮收敛。相比 TWN 的一次性近似，ITF 将 E_w 从 22.88 降至 11.56 PPL。
    3. **AGA 激活感知对齐** 弥补权重-激活脱节：在 ITF 收敛后，切换优化目标从 E_w = ||W-Ŵ||² 到 E_x = ||WX-ŴX||²，利用校准数据 X 的协方差矩阵 C=ΣXX^T，以闭式解（Eq. 13）更新 (α, μ)，使量化输出更匹配全精度输出。冻结 T 防止对校准集过拟合。对应解决缺陷 (2)：将 W-激活交互纳入优化，Avg Acc 从 38.12 提升至 43.33（LLaMA-2-7B）。
    4. **SSR 结构相似性重排序** 解决块内散乱问题：替代 GPTQ 的固定/Hessian 重排序，在每次选块时计算残差矩阵中剩余列与均值参考向量的余弦相似度，选 top-k 最相似列组成量化块。结构相似的列数值接近、方向对齐，形成更紧凑的分布，块内方差显著降低，离群列因彼此相似而不再是离群值。对应解决缺陷 (3)：SSR vs 无重排降低 PPL 从 13.06→11.56；SSR vs Hessian 重排（12.35）更优。
  - 全栈执行例子（以 PT²-LLM 量化 LLaMA-7B 为例，对比 baseline）：
    - **算法层**：FP16 W(4096×4096) → SSR 逐块选列（余 m 列时计算 cos_sim(W_col, w_bar) → top-128）→ ATQ 初始化（μ=row_mean(W), W̃=W-μ, Δ≈0.75·mean(|W̃|), T=三值映射, α=最小二乘解）→ ITF 循环（while T≠T_prev: α*,μ*=闭式解(Eq.9); T*=弹性舍入(Eq.10)）→ AGA 最终对齐（用 X_calib 128×2048 的协方差矩阵 C 更新 α*,μ*=Eq.13）→ Ŵ=α*T+μ*（每行仅3个可选值）。Baseline GPTQ+TWN 则是一次性 T→α 后即停止。
    - **系统框架层**：PyTorch + HuggingFace，GPTQ 逐块框架 + 误差补偿（H^{-1}δ 传播到未量化列）。SSR 在此框架中替换了列选取策略。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：llama.cpp 推理，论文测试 prefill/decode/end-to-end 吞吐。1.58-bit 模型对比 2-bit 模型在 LLaMA-65B end-to-end 上实现 2.1× 加速。三值 W∈{−α+μ, μ, α+μ} 使得矩阵乘法退化为加减运算，消除浮点乘法。
    - **硬件架构层**：Nvidia A800-80GB GPU，论文未涉及硬件设计。 Push the Real Limit of Extremely Low-Bit Post-Training Quantization

- baseline方法是什么？
  - **PB-LLM (Shang et al. 2023)**：选择 10% 显著权重保持 8-bit，其余二值化为 1-bit，使用 1-bit 非结构化细粒度掩码区分显著权重（掩码形状与权重矩阵相同）。等效位宽 = 0.1×8 + 0.9×1 + 1(mask) = 2.7-bit。缺陷：(1) 非结构化掩码无法压缩，每权重额外 1-bit，使总位宽超过 2-bit；(2) 缩放因子 α^w = ||w||_1/n_w 逐行解析计算，假设行间独立，忽略隐式行间相关性和角度偏差。
  - **BiLLM (Huang et al. 2024)**：使用更细粒度的分组二值化，将所有权重分为 3 组并计算组级缩放因子。提出基于 Hessian 的结构化掩码（显著权重）和基于幅值的非结构化掩码（非显著权重）。等效位宽 = 1.0 + 0.1(additional) + 1.0(mask) = 2.1-bit。缺陷：(1) 仍使用非结构化掩码，额外 1-bit 开销无法消除；(2) 缩放因子仍为解析推导，无法捕获行间依赖和方向偏差。
  - **OmniQuant (Shao et al. 2023)**：当前 2-bit PTQ SOTA，通过可学习的平滑参数和量化参数进行反向传播优化。缺陷：在 sub 2-bit 场景下性能显著下降（如 LLaMA-2-7B WikiText2 PPL=37.37），无法做好极低位量化。
  - 全栈执行例子（以 PB-LLM 量化 LLaMA-7B 为例）：
    - **算法层**：加载 FP16 LLaMA-7B 权重 → 按幅值选择 top-10% 显著权重 → 显著权重 8-bit 均匀量化（W_salient_q = clamp(round(W/S_q)+Z_q, 0, 255)）→ 非显著权重二值化（W_nonsalient_q = α sign(W)）→ 存储 1-bit 非结构化掩码（4096×4096 bits per layer）→ 推理时：若 mask[i,j]=1 则用 INT8 权重解量化，若 mask[i,j]=0 则用二值权重 ±α。
    - **系统框架层**：PyTorch fake-quantization。实际 GPU 不支持 2-bit 以下推理，仅做精度仿真。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明（无专用 kernel，纯 Python/PyTorch 仿真）。
    - **硬件架构层**：Nvidia A800 GPU（PTQ 校准用），论文未明确涉及硬件设计。
  - Baseline 核心缺陷：(1) **非结构化掩码内存开销巨大**——PB-LLM 和 BiLLM 的掩码额外占用 ≥1 bit/权重，等效位宽实际 >2-bit，未能真正实现 sub 2-bit；(2) **解析缩放因子忽略行间依赖和方向偏差**——独立计算每行 α^w 未考虑行间相关性，MSE loss 忽略余弦相似度方向偏差；(3) **预训练模型权重分布散乱**——显著权重在矩阵中散乱分布，不匹配逐通道/逐行量化的 row-wise 模式，导致量化误差大。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - PTQ1.61 通过以下设计解决 baseline 缺陷：
    1. **一维结构化掩码替代非结构化掩码**：通过数学推导 E ≤ Σ_i (|x_i| * Σ_j |w_{i,j}^q - w_{i,j}|) 证明量化误差上界由输入激活通道幅值主导（激活幅值 ~1000× 权重幅值）。因此保留激活幅值最高的 20% 通道对应的权重行（而非散乱元素）为 4-bit，掩码仅为 1×4096 一维向量，额外开销从 ≥1-bit 降至 0.0002-bit。对应解决缺陷 (1)：真正实现 sub 2-bit（1.61-bit）。
    2. **分块可学习缩放因子替代解析推导**：将每行缩放因子 α 设为可学习参数，联合 MSE loss（幅值差距）+ NLC loss（-log(cos_sim)，方向偏差）进行分块反向传播优化。优化目标 min E(F(W_q'), X) + E(F(W_q'), X_q)，考虑 block 内部量化误差传播。对应解决缺陷 (2)：同时捕获行间隐式相关性和角度偏差。
    3. **量化预处理（Restorative LoRA）**：用 LoRA (rank=64) 在 RedPajama 上微调（20K steps），将预训练模型中散乱的显著权重分布转化为行集中模式（row-wise concentrated pattern），使模型更适合逐通道 PTQ。对应解决缺陷 (3)：降低逐通道量化时的行内方差。
  - **如何对应解决 Baseline 缺陷**：
    - 掩码内存：一维结构化掩码（4096×1 bit）vs 非结构化掩码（4096×4096 bit），开销降低 4096 倍，等效位宽从 2.1-2.7 bit 降至 1.61 bit。
    - 缩放因子质量：可学习 + 联合 MSE+NLC 优化的 α 比解析 α^w=||w||_1/n_w 更好捕获行间依赖（block-wise 优化在同一 block 内联合优化多行）和方向偏差（NLC loss）。
    - 权重分布适配：预处理的 LoRA 将散乱显著权重转化为行集中分布，使每行内所有元素具有相似显著性水平，逐行量化时同一量化参数可更好地表示整行，减少量化误差。
  - 论文方法全栈执行例子（以 PTQ1.61 量化 LLaMA-7B 为例）：
    - **算法层**：
      1. 量化预处理（可选）：在 RedPajama 上用 LoRA(rank=64) 微调 20K steps → 权重从散乱显著模式变为行集中显著模式。
      2. 结构化掩码：收集校准数据 X → 计算每通道激活幅值 ||x_i|| → rank top-20% → 显著行 4-bit 量化，非显著行二值化。掩码为 {0,1}^{4096} 向量。
      3. 分块优化：按 transformer block 为单位，用 AdamW 优化缩放因子 α_s, α_r，损失 = MSE(output_fp, output_q) + NLC(output_fp, output_q)，20 epochs。
      4. 输出量化模型：平均 1.61-bit/权重 = 1.6(weight) + 0.0002(mask) + 0.008(scaling+zero-point)。
    - **系统框架层**：PyTorch fake-quantization，lm-evaluation-harness 评估。PTQ 在 2×A800 GPU 上进行。注：当前商用 GPU 不支持 sub 4-bit 整数推理，因此论文仅做精度仿真，非真实部署加速测量。
    - **编译框架层**：论文未明确说明（无自定义编译优化）。
    - **kernel调度层**：论文未明确说明。注：论文指出因 NVIDIA GPU 尚不支持极低位推理，实际 kernel 设计需更大团队和资金支持，未提供真实推理速度。
    - **硬件架构层**：论文未明确说明。

## PARQ Piecewise-Affine Regularized Quantization

- baseline方法是什么？
  - **STE/BinaryConnect（Courbariaux et al. 2015）**：QAT 的标准 baseline，使用硬量化映射 Q(·)（图 1 的阶梯函数）在训练全程对权重进行硬量化。更新规则：u^{t+1}=u^t-η_t∇f(Q(u^t), z^t)，w^{t+1}=Q(u^{t+1})。其中 u^t 作为全精度隐变量累积在量化点 w^t=Q(u^t) 处的梯度。STE 在反向传播中将 dQ/du=0 替换为 dQ/du=1，本质是一个启发式近似，缺乏严格收敛理论支撑。缺陷：(1) 全程硬量化导致训练动态不稳定（如图 12 中的 sudden accuracy drops），在小模型/极低位宽（1-bit/ternary）下尤其明显；(2) 仅在特殊情形下有弱收敛结果（如期望收敛而非最后迭代收敛），理论保证不足；(3) 硬量化图（阶梯函数）对应非凸 indicator 函数 δ_Q 的 proximal map，无法享有凸优化的收敛性质。
  - **BinaryRelax（Yin et al. 2018）**：用 W 形非凸正则化的 proximal map（图 9b）替代硬量化映射。slanted segment 斜率逐步减小至 0，通过放松量化约束来稳定训练。缺陷：(1) 使用的正则化是非凸的（W 形），梯度方法容易在初始权重落入"错误山谷"时被困于局部最优；(2) 同样缺乏最后迭代收敛保证，仅提供平均迭代的收敛结果——而平均迭代通常不满足量化结构；(3) 非凸正则化无法享受凸优化的全局收敛性质。
  - 全栈执行例子（以 STE/BinaryConnect 训练 1-bit ResNet-20 为例）：
    - **算法层**：加载 FP32 ResNet-20 权重 → Q(u^t) 将每个权重二值化投影到 {±q}（q=|u|₁/d）→ 在二值化权重 w^t 处计算交叉熵损失梯度 → STE 直接将梯度传递给隐变量 u^{t+1}=u^t-η_t∇f(w^t)→ 下一次迭代再次硬量化。全程在二值权重处计算梯度，没有从软到硬的渐进过程。
    - **系统框架层**：PyTorch，标准 SGD optimizer，GPU 训练（200 epochs on CIFAR-10）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（baseline 为纯 PyTorch 训练）。
  - Baseline 核心缺陷：**硬量化全程使用 → 训练初期不稳定**；**缺乏凸性 → 无全局收敛保证**；**弱收敛理论（仅平均迭代收敛）→ 量化结构在理论分析中无法得到保证**；**非凸正则化的 W 形 valley → 初始点不良时可能陷入局部最优**。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - PARQ 通过以下设计解决 baseline 缺陷：
    1. **凸 PAR 替代非凸/无正则化**：构建凸分段仿射正则化函数 PAR(w)=max_k{a_k(|w|-q_k)+b_k}，其中斜率 0≤a_0<a_1<...<a_m=+∞ 严格递增。该函数是凸的（有限个线性函数的最大值），但非光滑点（±q_k）的自然聚类效应使其能有效诱导量化。相比之下：(a) STE 无显式正则化，对应非凸 indicator δ_Q；(b) BinaryRelax 使用的 W 形正则化是非凸的。凸性确保了全局收敛性质和对初始点不敏感。
    2. **AProx 聚集 proximal 算法替代 Prox-SGD**：AProx 的关键在于用累积步长 γ_t=Ση_s 缩放 proximal map（而非 Prox-SGD 的单步步长 η_t）。由于 γ_t→∞，prox 中的 flat segments（长度 γ_t λ(a_k-a_{k-1})）不断增大，sloped segments 相对缩小，proximal map 从软量化渐近到硬量化（图 7→图 8）。这解决了 Prox-SGD 因 η_t→0 导致正则化消失的问题。
    3. **最后迭代收敛理论保证**：证明了 AProx 的最后迭代（last-iterate）收敛率 O(ln(t)/√t)，与平均迭代收敛率匹配。这比 BinaryRelax/Dockhorn et al. 仅证明平均迭代收敛更强——因为平均迭代通常不被量化，而最后迭代可以在渐近阶段被保证量化。
    4. **LSBQ 在线估 Q + 独立斜率 schedule**：避免预设量化值和正则化强度的难题。LSBQ 从隐变量 u^t 中在线估计目标量化值 {q_k}；独立逆斜率 schedule ρ_t^{-1}（cosine decay）从 1→0，使 proximal map 从近似 identity（训练早期，平滑过渡）→ 硬量化（训练末期）。这使训练初期 PARQ 接近全精度训练（loss 曲线靠近 FP），后期自然过渡到量化状态。
  - **如何解决 Baseline 缺陷**：
    - **训练稳定性**：PARQ 的渐进软→硬量化（而非全程硬量化）使训练更稳定。图 12 显示 PARQ 无 STE 的 sudden accuracy drops，训练 loss 曲线更平滑。
    - **凸优化保证**：凸 PAR 确保全局收敛性质，不会像 BinaryRelax 的 W 形非凸正则化那样陷入局部最优。
    - **更强的理论保证**：最后迭代收敛结果保证最终模型权重被量化（而非平均迭代），理论更贴近实际需求。
    - **实用自适应性**：无需为不同模型/数据集调优 q_k、a_k、λ，LSBQ+schedule 自动适应。
  - 论文方法全栈执行例子（以 PARQ 训练 2-bit DeiT-Ti 为例）：
    - **算法层**：
      1. 初始化：u¹=w¹（随机初始化的 FP32 权重）。
      2. 每轮迭代：(a) 在 w^t 处计算 mini-batch 梯度 g^t；(b) u^{t+1}=u^t-η_t g^t（累积纯梯度）；(c) LSBQ(u^{t+1}, n=2) 估计 Q^{t+1}={±q₁,±q₂}（q₁≈v₁-v₂, q₂≈v₁+v₂），值从随机初始化时的小量快速膨胀→缓慢收缩（图 13）；(d) w^{t+1}=prox_PARQ(u^{t+1}, Q^{t+1}, ρ_t)，其中 ρ_t^{-1} 从 1→0：(早期) slope≈1，prox 接近 identity → 权重几乎未被量化，训练接近 FP；(中期) slope 增大，prox 呈现 slanted+flat 混合结构（图 11 中），soft quantization → 权重开始向 Q 中离散值聚类；(晚期) slope→∞，prox 收敛为 hard quantization（图 11 右），权重完全量化。
      3. 最后 20 epochs：lr 固定在 1e−8，所有方法均在硬量化模式下微调。
    - **系统框架层**：PyTorch 实现，SGD (ResNet) / AdamW (DeiT)，标准数据增强 pipeline。开源代码 `parq` 包可直接 pip install 使用。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明。PARQ 是纯训练算法，不涉及推理时的编译或 kernel 修改。量化后的模型可用标准 PyTorch runtime 推理，或进一步结合 TensorRT/其他量化推理框架加速。
  - 关键理论洞察：**凸性 + 非光滑性 → 量化诱导**。最优条件分析揭示：在 PAR 正则化的最优解处，与非量化区间 (q_{k-1}, q_k) 对应的梯度值只能是 2m 个离散值 {±λ a_k} 之一，而几乎所有其他梯度值可通过将权重置于 Q 的 2m+1 个离散值上来平衡。这意味着最优解处的权重"大概率"聚合在离散量化值上——这是 PARQ 的数学基础。此外，AProx 可被解读为 STE 的渐近形式：当 γ_t→∞ 时，prox_PAR 收敛到硬量化映射 Q(·)，此时 AProx 退化为 BinaryConnect/STE。

## Optimal and Approximate Adaptive Stochastic Quantization

- baseline方法是什么？
  - **ZipML**：通过动态规划（DP）求解 ASQ 问题的最优量化值集合 Q ⊆ X。DP 状态 MSE[i,j] 表示用 i 个量化值量化前缀向量 X_j 的最优 MSE。转移：MSE[i,j] = min_{k} MSE[i-1,k] + C[k,j]，其中 C[k,j] 为区间内所有条目在连续量化值 {x_k, x_j} 下的方差和。时间复杂度 O(s·d²)，空间复杂度 O(d²)。当 d > 10⁵ 时内存和时间均不可行，无法用于"on the fly"量化场景。
  - **ZipML-CP (Candidate Points)**：从 X 中选取 m 个候选点（均匀网格或分位数），在候选点子集上运行 ZipML 精确解。时间复杂度 O(d + m²·s)，但未提供最优候选点选择策略，近似质量不可控。
  - **ZipML 2-Apx**：保证 MSE ≤ 2·opt_{X,⌊s/2⌋}（使用两倍量化值保证误差不超过最优解的两倍），时间复杂度 O(d log d + s³)。
  - **ALQ**：假设输入服从截断正态分布，拟合分布参数后通过迭代积分求解近似最优量化值。单次量化需 ≈10s 次积分计算，实际速度慢，且假设分布不总是成立。
  - 全栈执行例子（以 ZipML baseline 量化一个 LogNormal 梯度向量为例）：
    - **算法层**：加载 d=10⁵ 维 FP32 向量 → 排序 O(d log d) → 计算 C[k,j] 矩阵（对所有 k≤j 逐一求和，O(d³) 或优化到 O(d²)）→ 逐行填充 MSE[i,j] 表（i=2..s, j=i..d，每步 O(d) 枚举 k）→ 回溯构建 Q。对 d=10⁵, s=16 无法在 commodity PC 上完成（内存 > 80 GB，时间 > 10³ 秒）。
    - **系统框架层**：无特定框架依赖，纯算法计算。对于分布式/联邦学习场景，量化器运行在 CPU 上（gradient compression sender/receiver 端）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（baseline 为纯 CPU 算法）。
  - Baseline 核心缺陷：**时间复杂度 O(s·d²) 和空间复杂度 O(d²) 使得 ASQ 在大向量上不可行**。即使 d=10⁵ 级别也因内存溢出无法运行，而实际 ML 场景中梯度向量维度常达 10⁶-10⁷。已有近似方法要么假设特定分布（ALQ）、要么近似保证弱（ZipML 2-Apx 使用 2× 量化值只保证 2× 误差）、要么无理论保证（ZipML-CP）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **QUIVER** 通过三个创新将 ASQ 从不可行变为实用：
    1. **O(1) C[k,j] 计算**：预处理累积和数组 β_j=Σx_i, γ_j=Σx_i²，使方差和 C[k,j] = -x_j·x_k·(j-k) + (x_j+x_k)·(β_j-β_k) - (γ_j-γ_k) 可常数时间求值，无需预计算/存储 O(d²) 的 C 矩阵。
    2. **Quadrangle Inequality → SMAWK**：证明 C 满足 quadrangle inequality（∀a≤b≤c≤d: C[a,c]+C[b,d] ≤ C[a,d]+C[b,c]），从而 DP 矩阵 A[k,j]=MSE[i-1,k]+C[k,j] 是 totally monotone。对 totally monotone matrix，SMAWK 算法可在 O(d) 时间内找到每列的行最小值索引，替代原 DP 中每步 O(d²) 的枚举。
    3. **s=3 闭式解 → Accelerated QUIVER**：推导三个量化值时的中间值闭式解 b*_{k,j} = ⌈(j·x_j - k·x_k - (β_j-β_k))/(x_j-x_k)⌉，使得 C²[k,j] 亦可 O(1) 计算。每次 SMAWK 调用跳过两个量化值，调用次数减半，速度提升最高 5.4×。
    4. **离散化 → Apx. QUIVER**：将候选量化值限制在 m 个均匀网格点上，使用直方图预处理实现 O(d+m·s) 时间，并给出严格近似保证 AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)。
  - **如何解决 Baseline 缺陷**：
    - **时间从 O(s·d²) 到 O(s·d)**：通过预处理+隐式矩阵+SMAWK 的组合，将每步 DP 转移的枚举优化从 O(d) 降为 amortized O(1)，最终使 d=10⁶, s=16 的精确解在 1 秒内完成（vs ZipML 无法运行）。
    - **空间从 O(d²) 到 O(s·d)**：不再需要存储 C 矩阵（仅存 β/γ 两数组和当前 MSE 行），使 d=2²⁴（16M）可在一台 commodity PC 上运行。
    - **近似方案的严格保证**：Apx. QUIVER 提供 additive error bound（而非 ZipML 2-Apx 的 multiplicative 2× bound），且通过参数 m 提供可控的 accuracy-speed tradeoff（m=400 时 vNMSE 接近最优，6ms 完成 1M 向量）。
    - **无需分布假设**：与 ALQ（需截断正态假设）不同，QUIVER 是精确算法（Q ⊆ X），不受 input distribution 限制。
  - 论文方法全栈执行例子（以 Accelerated QUIVER 量化 d=10⁶ LogNormal 向量为例）：
    - **算法层**：
      1. Preprocess: 排序（若非已排序则 O(d log d)），一趟扫描计算 β_j=Σx_i, γ_j=Σx_i²（O(d) 空间）。
      2. 初始化 MSE 表：若 s 为偶数则 MSE[2,j]=C[1,j]；若 s 为奇数则 MSE[3,j]=C²[1,j]（利用 b* 闭式解 O(1)）。
      3. SMAWK 迭代（⌊s/2⌋-1 轮）：每轮在隐式 totally monotone matrix B[k,j]=MSE[prev,k]+C²[k,j] 上运行 SMAWK，O(d) 时间求得当前行 MSE 和 argmin K。
      4. 回溯重建：从 x_d 开始，沿 K 逆向跳转，并在每个区间中用 b* 恢复跳过的中间量化值。
      5. 结果：Q={q₁,...,q₁₆} ⊆ X，信源量化：对每个 x∈X，找到包围它的两连续量化值 q_l≤x≤q_r，以概率 p_l=(q_r-x)/(q_r-q_l) 输出 q_l，否则输出 q_r（保证 E[x̂]=x）。
    - **系统框架层**：C++17 单线程 CPU 实现，compiled with -O3。用于分布式学习的 gradient compression 时，sender 端先排序→QUIVER→量化→发送 Q+量化比特；receiver 端根据 Q 和比特解码。实际场景中排序可由 GPU 完成（T4 GPU 上 1M 向量排序仅 4ms）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（纯 CPU 算法，不涉及 GPU kernel 或专用硬件）。论文明确指出现有 QUIVER "not GPU friendly"，将 GPU-friendly ASQ 设计列为 future work。
  - 关键数学洞察：**C 的 quadrangle inequality 证明**是算法的理论基石——它将看似 ad-hoc 的方差和函数与 totally monotone matrix 的经典理论（SMAWK, 1986）联系起来，实现了从 O(d²) 到 O(d) 的渐进改进。这一观察不仅适用于 ASQ，也可能适用于其他满足该性质的序列分割 DP 问题。

## Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

- baseline方法是什么？
  - **QuaRot (quant-only)**：单独使用量化，通过 Hadamard rotation 平滑权重 outliers 实现 W4A4KV4 量化。缺陷：sub-4bit 时性能急剧下降（如 Llama2-7B W3A4KV4 perplexity 达 132.97），且单独压缩方法逼近极限。
  - **QuaRot+WANDA (naive joint)**：直接将 QuaRot 量化（经过 Hadamard rotation）与 WANDA 剪枝组合，即对 rotated weights 直接施加剪枝 mask 后再量化。缺陷：Hadamard rotation 使权重分布平坦（quantization-friendly），但这与剪枝的需求（需要 weight magnitudes 差异大以呈现天然稀疏性）冲突，导致性能灾难性下降（Llama2-7B perplexity 达 5868.24）。
  - **SparseGPT+GPTQ (strong baseline)**：使用 SparseGPT 剪枝 + GPTQ 量化的组合。缺陷：虽然较 naive combination 有改善，但未专门调和量化（偏好窄范围）与剪枝（偏好高方差）对权重分布的冲突需求，在 W4A4KV4 + 50% sparsity 激进压缩下性能依然不足。
  - 全栈执行例子（以 QuaRot+WANDA baseline 为例）：
    - **算法层**：加载 FP16 Llama2-7B → Hadamard rotation R 作用于 W 和 X (将异常值扩散打平) → WANDA 用激活统计 (|W|·||X||₂) 做重要性得分 → 对 rotated W 施加 50% 非结构化剪枝 mask → RTN 量化 W 到 INT4 → 激活/X 也经 rotation 后量化到 INT4 → KV cache INT4 量化。输出 W4A4KV4 + 50% sparse 模型。
    - **系统框架层**：PyTorch + HuggingFace Transformers, block-wise GPU 加载。校准用 128 WikiText2 样本。
    - **kernel调度层** (OBR baseline 特有)：NVIDIA CUTLASS INT4 2:4 sparse GEMM kernel，利用 Ampere/Hopper 的 Sparse Tensor Cores 实现硬件加速。INT4 权重（packed）+ 2:4 结构化稀疏 mask，FP16 激活经反量化后与 sparse 权重通过 `mma.sp.sync` 指令在 Tensor Core 上执行 MMA。
    - **硬件架构层**：NVIDIA A100 GPU (Ampere 架构)，第三代 Tensor Cores 原生支持 2:4 结构化稀疏和 INT4/INT8 混合精度计算。
  - Baseline 核心缺陷：**量化与剪枝的权重分布需求冲突**——量化希望紧凑的数值范围以减少量化误差，剪枝希望大幅值差异以暴露可剪权重。Hadamard rotation 虽利于量化但破坏了剪枝所需的 distributional disparity。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **OBR (Optimal Brain Restoration)**：在剪枝后、量化前插入 Group Error Compensation，通过 Hessian 矩阵作为"桥梁"将压缩误差从敏感 group 转移到鲁棒 group，从而调和量化与剪枝的冲突。核心公式化：minΔw_R ½[Δw_R e_E] [H_{RR} H_{RE}; H_{ER} H_{EE}] [Δw_R^T e_E^T]，闭式解 Δw_R^* = -H_{RR}^{-1} H_{RE} e_E。
  - **如何解决 Baseline 缺陷**：
    - **调和权重分布冲突 (核心贡献)**：不改变 Hadamard rotation（维持其量化友好性），也不改变剪枝 mask，而是在二者之间通过 Error Compensation 重新分配信息。剪枝损失的权重信息（e_E^{prune} = w_E）通过 Hessian 子矩阵 H_{RE} 传播到 unpruned 权重中进行补偿，使 unpruned 权重在量化前就已经"吸收"了剪枝损失的知识。类似地，量化误差也通过第二轮 OBR 补偿到更多元素中。
    - **二阶 Hessian 目标**：用 min E[ΔL] ≈ ½ vec(ΔW) H_full vec(ΔW)^T 建模权重扰动对 downstream task 的影响。通过 Kronecker 分解 H_full ≈ G ⊗ H 和 row-wise decoupling (G≈I) 将 C_out×C_in 维问题分解为 C_out 个独立的 C_in 维子问题，使原本 O((C_out·C_in)²) 不可行的问题变为可解。
    - **无需额外训练 (training-free)**：OBR 是纯后训练方法，只需一次 calibration 数据前向传播收集 Hessian 统计信息，然后通过闭式解完成补偿。比 QAT (quantization-aware training) 方法更实用。
    - **兼容多种剪枝方法和量化器**：OBR 将剪枝 mask 和 quantizer 视为给定输入（"黑盒"），因此兼容 WANDA、SparseGPT、magnitude-based、甚至 Random 剪枝；也兼容 RTN 和 GPTQ 量化器。
  - 论文方法全栈执行例子（以 OBR via QuaRot + WANDA + RTN 为例）：
    - **算法层**：(0) 128 WikiText2 样本 × 2048 seq_len 前向传播，收集每层激活 X，计算 Hessian H = 2XX^T。(1) Hadamard rotation R 作用于 W 和 X 打平 outliers。(2) WANDA 根据 |W_rot|·||X_rot||₂ 计算重要性得分，生成 50% 非结构化剪枝 mask M。(3) **OBR for Pruning**：对每行，R₁=unpruned, E₁=pruned, 计算 Δw_{R₁}^{prune} = -H_{R₁R₁}^{-1}H_{R₁E₁}W_{c,E₁}, 补偿到 w_{R₁}。(4) **OBR for Quantization**：w̄ = w_{R₁}+Δw^{prune}, 计算量化误差 e^{quant}=w̄-⌊w̄⌉, 按 α=50% 划分 E₂ 和 R₂, 计算 Δw_{R₂}^{quant} = -H_{R₂R₂}^{-1}H_{R₂E₂}e_{E₂}^{quant}。(5) 合并补偿 ΔW^{OBR} = ΔW^{prune} + ΔW^{quant}, W^{quant} = W^{prune} + ΔW^{OBR}, 对 W^{quant} 做 RTN 量化到 INT4。(6) 对 activation X 和 KV cache 也量化到 INT4。输出 W4A4KV4 + 50% sparse Ŵ。
    - **系统框架层**：PyTorch + HuggingFace Transformers, block-wise loading, single A100 GPU。
    - **kernel调度层**：NVIDIA CUTLASS INT4 2:4 sparse GEMM kernel。权重以 packed INT4 + 2:4 metadata 存储（50% 稀疏将访存带宽减少 2×），激活 INT4 → FP16 反量化后进入 Sparse Tensor Core，通过 `mma.sp.sync` 指令利用硬件跳过零值。在 seq_len=4096 时达到 5.9× (vs FP16-dense) 和 1.4× (vs INT4-dense) 加速。
    - **硬件架构层**：NVIDIA A100 GPU (Ampere)，Tensor Cores 原生支持 2:4 sparse MMA 和 INT4 推理。

 One-Line Revolution for Generative AI Model Compression

- baseline方法是什么？
  - **Layer-wise PTQ (RTN / GPTQ)**：将每个 linear layer 独立量化为最小二乘问题。RTN 直接逐权重量化（Eq.1: Ŵ = argmin ||Ŵ - W||²），GPTQ 使用 activation-aware 目标（Eq.2: Ŵ = argmin ||X̂(Ŵ - W)||²）逐行贪心量化并用 Hessian 补偿残差。缺陷：局限于单个 linear layer，无法建模跨层/跨子模块的误差传播。
  - **QEP**：在 layer-wise PTQ 基础上引入误差传播修正（Eq.3: Ŵ_QEP = argmin ||X̂Ŵ - XW||²），等价于将量化目标从 W 改为修正后的 W* = (I + α Ĥ^{-1}C)W。缺陷：仍仅限于单个 linear layer，不处理 attention、MLP、残差连接等更复杂子模块。
  - **LoaQ**：将 QEP 扩展到残差路径（Eq.4: 最小化 ||(R̂ + X̂Ŵ) - (R + XW)||²），通过 W*(α,β) 同时修正线性层误差和残差路径误差。缺陷：限于特定的 attention+MLP+残差+RMSNorm 子模块组合，不提供任意子模块的统一处理，且为单步修正。
  - 全栈执行例子（以 GPTQ+QEP baseline 量化一层 Q 投影为例）：
    - **算法层**：加载 FP16 LLaMA → 校准数据前向收集每层输入 X → 计算 Hessian H = XᵀX 和误差传播矩阵 C = X̂ᵀ(X - X̂) → 计算修正目标 W* = (I + αĤ^{-1}C)W → 对 W* 逐行 GPTQ 贪心量化（取整 → 更新 Hessian 残差补偿到剩余列 → 直到所有列量化完毕）→ 输出 INT-k 权重。
    - **系统框架层**：PyTorch + HuggingFace Transformers, block-wise GPU 加载。推理时标准量化 matmul（反量化×激活）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（依赖 PyTorch 默认 CUDA kernel 和 H100 TensorCores）。
  - Baseline 核心缺陷：layer-wise 独立优化忽视子模块内部的非线性交互（如 softmax、残差连接、门控激活），导致 sub-4bit（尤其是 INT3/INT2）时量化误差急剧累积；QEP/LoaQ 通过单步修正改善误差传播，但修正能力受限于特定模块和固定公式。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LPCD (Layer-Projected Coordinate Descent)**：将 PTQ 重新定义为对任意子模块 blocks 的离散优化问题（Eq.5: min_{M̂_1,...,M̂_R} L(M̂_1,...,M̂_R)），通过交替坐标下降求解：
    1. **Relaxation Step**（Eq.6）：固定其他 blocks，对当前 block 的连续变量 U 求解无约束优化 M̄_r = argmin_U L_r(U)。当 L_r 为严格凸二次函数时直接求闭式解；否则用梯度下降近似。
    2. **Projection Step**：用标准 layer-wise PTQ 投影器 Π_Q（RTN 的 Π^(d) 或 GPTQ 的 Π^(a)）将 M̄_r 投影回量化域 M̂_r = Π_Q(M̄_r)。
    3. 交替更新所有 blocks，一轮完成后可多轮迭代。
  - **如何解决 Baseline 缺陷**：
    - **从单层到子模块**：LPCD 的 block 变量 M_r 可以是任意 Transformer 子模块的权重、激活或 KV cache。对 QK Module（grouped-query attention 的 Q/K 投影），最小化 masked attention score 的 Frobenius 误差 ||M ⊙ (Ŝ - S)||²；对 VO Module（value-output 聚合），最小化残差流输出误差 ||Ω̂ + R̂ - (Ω + R)||²；对 Up-Down Module（MLP 含 SiLU 门控），最小化 ||F̂ + R̂ - (F + R)||²。这直接建模了子模块内部的非线性交互。
    - **统一 QEP/LoaQ 为特例**：Proposition 4.1 证明 QEP 是两-block (Ŵ, X̂) LPCD 的单步更新；Remark 4.2 证明 LoaQ 是三-block (Ŵ, X̂, R̂) LPCD 的单步更新。LPCD 通过增加迭代次数和扩展 block 集，显著超越单步修正。
    - **避免 STE 不稳定性**：LPCD 的 Relaxation Step 求解严格连续优化（闭式解或梯度下降），Projection Step 复用成熟的 layer-wise PTQ，无需引入 pseudo-gradient（STE），避免了 QAT 中常见的不稳定问题。
    - **正交于基础 quantizer**：LPCD 的 Projection Step 可插拔 RTN 或 GPTQ，因此其收益独立于底层量器选择。实验表明 RTN+LPCD 在部分设置下已超越 GPTQ+QEP。
  - 论文方法全栈执行例子（以 VO Module 的 LPCD 为例）：
    - **算法层**：前向传播收集校准数据的全精度 S^(h), V^(g), Ω, R 和量化版的 Ŝ^(h), V̂^(g), R̂ → 固定 Ŵ_O，对每个 group g 的 Ŵ_V^(g) 做 Relaxation（求解最小化 ||Y - Ŷ_{¬g} - Ŷ_g||² 的线性最小二乘，设计矩阵过大时用 Adam 梯度下降近似）→ Projection（RTN/GPTQ 量化 W̄_V^(g)）→ 固定 Ŵ_V，对 Ŵ_O 做 Relaxation（Ŵ_O = (ĤᵀĤ)⁻¹Ĥᵀ(Y - R̂)，闭式解可行）→ Projection → 完成一轮 VO Module LPCD。同流程应用于 QK 和 Up-Down 模块。
    - **系统框架层**：PyTorch + HuggingFace Transformers，与 layer-wise PTQ 兼容的 block-wise 内存管理。量化流程先运行 LoaQ 作为初始化，再在 LoaQ 结果上运行 LPCD。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明（仅使用 PyTorch 默认 CUDA kernel + H100 GPU）。
  - LPCD 还将 QEP 风格修正扩展到 activation quantization（Eq. 16-17: X̄ = XWŴᵀ(ŴŴᵀ)⁻¹ → 投影）、KV-cache quantization（Eq. 18-22: Key cache 对齐 pre-softmax logits，Value cache 对齐 post-softmax outputs）、正交旋转矩阵（Eq. 23-27: 通过 LPCD 在固定 X̂, Ŵ 下优化旋转矩阵 R，闭环式解 + 正交 Procrustes 投影）、以及 LoRA 误差补偿（weighted low-rank projection onto E = BA）。

## Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- baseline方法是什么？
  - **Transformer (self-attention)**：隐藏状态为 KV cache（一个随 t 线性增长的列表），更新规则为 `K_t,V_t` 追加到列表，输出规则为 `z_t = V_t softmax(K_t^T q_t / √d)`。优势：显式存储所有历史上下文，长上下文表达能力强。缺陷：每个 token 的 cost 随 t 线性增长 O(t)，总复杂度 O(T²)，在长上下文时计算和内存开销巨大。
  - **Mamba (现代 RNN)**：隐藏状态为固定大小的 state space model 状态，更新规则为输入依赖的选择性 SSM 扫描。优势：线性复杂度 O(T)。缺陷：固定大小的隐藏状态表达能力有限，在超过 16k 上下文后无法有效利用额外 token 信息（perplexity 不再下降）。更新规则为手工设计的选择机制，缺乏灵活性。
  - Baseline 全栈执行（以 Transformer 推理一个 token 为例）：
    - **算法层**：x_t 经 θ_Q,θ_K,θ_V 投影 → q_t,k_t,v_t → 与历史所有 k_s 计算点积注意力分数 → softmax 归一化 → 加权求和 v_s → 输出 z_t。每个 token 需 O(t) 次内积。
    - **系统框架层**：vLLM 管理 KV cache（PagedAttention），调度 prefill 和 decode 阶段。
    - **编译框架层**：论文未明确说明（使用 JAX XLA 自动编译）。
    - **kernel调度层**：FlashAttention kernel 将 Q,K,V 矩阵分块加载到 SRAM，tiled matmul + online softmax 减少 HBM 访问。
    - **硬件架构层**：NVIDIA A100 TensorCores 执行 16×16 matmul，HBM 存储 KV cache。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **TTT 层**：将隐藏状态定义为一个机器学习模型 f 的权重 W_t，更新规则为对自监督 loss ℓ 的一步梯度下降 `W_t = W_{t-1} - η ∇ℓ(W_{t-1}; x_t)`，输出规则为 `z_t = f(θ_Q x_t; W_t)`。核心理念：**自监督学习能将大规模训练集压缩进模型权重**——这正是 LLM 的工作原理——因此也将这种"压缩启发式"用于 RNN 隐藏状态的更新。自监督任务（多视角重建）本身通过外循环学习，而非手工设计。
  - 两个实例化：
    - **TTT-Linear**：f(x) = Wx（线性模型），隐藏状态为 d×d 矩阵
    - **TTT-MLP**：f 为两层 MLP（hidden dim 4×，GELU，LN + 残差），表达能力更强
  - 解决 Baseline 缺陷的具体设计：
    1. **线性复杂度 + 强表达能力**：内循环梯度下降将任意长度的上下文压缩进固定大小的 W，复杂度 O(d²) 与 T 无关。相比于 Mamba 的固定状态更新规则，TTT 的梯度更新是数据自适应的——产生大梯度的 token 被"记住"更多。解决了 Mamba 在 16k 后无法利用长上下文的痛点。
    2. **可学习的自监督任务**：θ_K（training view）、θ_V（label view）、θ_Q（test view）通过外循环学习，使得内循环的 reconstruction 任务专门为最终的下一个 token 预测目标服务。解决了手工设计 reconstruction（如 denoising autoencoder）可能不是最优自监督任务的痛点。
    3. **mini-batch TTT (b=16)**：从 online GD (b=1, 序列化) 和 batch GD (b=T, 仅一步) 之间取折中，既保持了多步梯度下降的搜索空间（perplexity 接近 online GD），又利用 mini-batch 内的并行化（比 online GD 快得多）。解决了内循环梯度更新无法并行的痛点。
    4. **Dual form**：将 `W_b = W_0 - 2η(W_0X̂ - Y)X̂^T` 和 `Z = W_0X̄ - 2η(W_0X̂ - Y)mask(X̂^TX̄)` 全部表达为 matmul 操作，避免显式计算逐 token 梯度 G_t（外积），从而充分利用 GPU TensorCores 的 16×16 matmul 单元。TPU 上比 primal form 快 5× 以上。
    5. **与 linear attention 的理论等价与超越**：TTT-Linear + batch GD 等价于 linear attention（Theorem 1）。由此出发，mini-batch GD 贡献最大改进（PPL 12.35 vs 15.23），LN+residual in f 次之（PPL 14.05 vs 15.27）（Table 1）。这些设计在 attention 框架下难以自然产生。
  - TTT-Linear 全栈执行（以内循环一个 mini-batch 为例）：
    - **算法层**：x_1,...,x_b 经 θ_K,θ_V,θ_Q 投影 → 公式 (4) loss ℓ(W_0; x_t) = ||W_0 x̂_t - y_t||² → 外循环学习 θ_K,θ_V,θ_Q 使此自监督任务对最终语言建模最优 → 内循环梯度下降更新 W → 公式 (5) 输出 z_t = f(θ_Q x_t; W_t)。每个 token O(d²)，与 T 无关。
    - **系统框架层**：EasyLM (JAX) 训练框架，与 Transformer 相同的训练循环和 recipe（Chinchilla），TTT 层可即插即用替换 self-attention。Gradient checkpointing through time 节省内循环中间 W_t 的内存。
    - **编译框架层**：JAX XLA 自动编译。dual form 使所有关键操作为 matmul + element-wise，适配 XLA fusion。
    - **kernel调度层**：Forward (prefill) 使用 dual form kernel——所有操作为 matmul + mask，最大化 TensorCore 利用率；Decode 使用 primal form kernel——单 token 的梯度外积和权重更新。最终 TTT-Linear 1.3B 在 A100 上 prefill latency 略高于 Mamba 但远低于 Transformer（32k 时约 1/3）。
    - **硬件架构层**：与 Transformer 相同（A100 TensorCores + HBM），但 dual form 将全操作转为 matmul，卸载了 softmax 和 attention pattern 的非 matmul 开销。

## MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

- baseline方法是什么？
  - **直接 PTQ（RTN / OPTQ）**：对预训练权重直接做 uniform quantization，不进行预处理。量化步长 δ = (max(w) − min(w))/(2^b − 1) 由原始权重的最大/最小值决定。由于预训练权重中存在 outliers 和大范围分布，δ 偏大，导致量化网格稀疏，量化误差大，尤其在 sub-4bit 时 perplexity 急剧上升。OPTQ 虽用二阶 Hessian 信息逐列贪心补偿量化误差，但无力改变权重本身的分布范围。
  - **线性变换预处理方法（AWQ / OmniQuant / QuIP）**：对权重施加可逆线性变换 T（通道缩放、随机正交变换等），使 TW 比 W 更"量化友好"（幅度小、无 outliers），然后量化 TW。推理时需在特征上施加 T⁻¹（XT⁻¹），额外引入计算和存储开销（QuIP 比 OPTQ 慢约 1.5×）。
  - 全栈执行例子（以 OPTQ baseline 为例）：
    - 算法层：加载 FP16 预训练 LLaMA 模型 → 逐层用 calibration 数据计算 Hessian H = XᵀX → 对每列权重贪心取整并更新未量化权重的 Hessian 补偿 → 得到 INT-k 量化权重。
    - 系统框架层：基于 PyTorch + HuggingFace Transformers，block-wise 加载（每次 7 个 linear layer 到 GPU），推理时执行量化矩阵乘法（FP16 反量化 × FP16 激活）。
    - 编译框架/kernel调度/硬件架构层：论文未明确说明（依赖 PyTorch 默认的 CUDA kernel 和 NVIDIA A100 Tensor Core）。
  - Baseline 核心缺陷：权重幅度大 → 量化步长大 → 量化误差大；线性变换方法虽能降幅度，但引入推理开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MagR 方法**：在量化前，对每层权重做 channel-wise ℓ∞-regularized 最小二乘优化（式2），通过 Proximal Gradient Descent（式3）+ ℓ₁-ball 投影（Moreau 分解，式4）迭代求解新的权重 W'，使 ‖W'‖∞ 最小化且保持 ‖XW' − XW_hat‖ ≤ ε。预处理后的 W' 直接替代原始权重，后续量化（RTN / OPTQ / QuIP）无需对特征做任何逆变换。
  - **核心洞察**：特征矩阵 X 近似秩亏（表2，fraction rank 均值 70%–84%，最低仅 0.1%），意味着 X 的核空间非平凡，存在大量 w 满足 Xw = Xw_hat。MagR 在核空间中寻找 ‖w‖∞ 最小的解，从而缩小量化步长 δ，而不改变层输出。
  - **如何解决 Baseline 缺陷**：
    - 针对"权重幅度大"：ℓ∞-regularization 直接将每列权重的最大绝对值作为优化目标压到最小，Figure 1 显示列最大幅度通常可减半以上；Table 1 表明 MagR 预处理后的 FP16 模型 perplexity 几乎无损（LLaMA2-7B WikiText2: 5.47→5.52）。
    - 针对"推理开销"：MagR 是非线性变换，不产生 T⁻¹，不修改特征/激活路径，推理时零开销。
    - 针对"sub-4bit 精度差"：通过缩小 δ（含 β 缩放因子进一步聚拢量化网格），大幅降低量化误差（Figure 2 显示各层量化 RMSE 显著下降）；W2A16 下 MagR+OPTQ† 在 LLaMA2-70B 上达 WikiText2 PPL 5.95，优于 OmniQuant (7.81) 和 QuIP (6.33)。
  - 全栈执行例子：
    - 算法层：加载 FP16 预训练权重 W_hat → 逐层用 calibration 数据计算 Hessian H = XᵀX → 运行 proximal gradient descent（K=150 迭代，每次迭代：梯度步 V^k = W^k − η·H·(W^k − W_hat)，然后列级 ℓ₁-ball 投影 W^{k+1} = V^k − ηα·proj_{‖·‖₁≤1}(V^k/(ηα))）→ 得预处理权重 W' → 对 W' 做 OPTQ/RTN 量化（含 β-scaled δ）→ 输出 INT-k 模型。
    - 系统框架层：与 baseline 相同，PyTorch + HuggingFace，block-wise GPU 加载，推理时标准量化矩阵乘法（无额外变换）。
    - 编译框架/kernel调度/硬件架构层：论文未明确说明（与 baseline OPTQ 同，依赖 PyTorch 默认 CUDA kernel）。
  - **MagR 预处理开销**：LLaMA2-7B ~15 min，13B ~30 min，70B ~3.5 hr（单 A100），仅一次性预处理；总量化时间（MagR+OPTQ）约为 OmniQuant 的一半。

## LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

- baseline方法是什么？
  - baseline 是 GPTQ+LoRA（类 QLoRA 方案）：先对预训练 LLM 权重 W 进行 GPTQ 非对称分组量化（W_q = s·W_int + z），然后在冻结的量化权重上训练 16-bit LoRA 适配器 A、B（ΔW_16 = A B）。推理时通过 y = (W_q + ΔW_16)^T x 计算，但量化权重（如 4-bit）和 16-bit 适配器之间的数据类型不匹配导致计算效率损失。若将 16-bit 适配器合并入量化权重（W'_int = round((W_q + ΔW_16 - z)/s)），适配器精度会被截断/量化，重新引入量化误差，导致微调精度退化。QA-LoRA 实现了无损合并，但其适配器仅能调整分组量化的零点因子 z，无法直接修改量化权重 W_int。
  - baseline 全栈执行例子（GPTQ+LoRA）：
    - 算法pipeline：预训练 FP16 权重 → GPTQ 逐列量化（二阶 Hessian 信息补偿误差）→ 冻结 W_int，s，z → 初始化 FP16 LoRA 适配器 A∈R^{D_in×r}, B∈R^{r×D_out} → 前向 y = (s·W_int + z + (α/r)AB)^T x → 反向更新 A, B（FP16 精度）
    - 系统框架：论文未明确说明具体 Serving 框架修改
    - 编译框架：论文未明确说明
    - kernel调度：推理时 4-bit 权重需反量化到 FP16 与 LoRA 适配器相加，kernel 为 TritonV2QuantLinear/TorchQuantLinear
    - 硬件架构：在 NVIDIA A800 GPU 上运行，无专用硬件设计

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - LoTA-QAF 通过三个核心设计解决 baseline 缺陷：
    1. **三元适配器（TA）对齐量化网格**：将适配器约束为三值 {-1,0,1}，乘积 ΔW = A_T B_T 为整数矩阵，通过阈值 ω 生成三元调整矩阵 Ŵ ∈ {-1,0,1}。因为 Ŵ 与 W_int 同属整数域，可以直接相加调整量化权重（W'_int = W_int + Ŵ），无需反量化后再量化。这解决了 baseline 中 FP16 适配器无法直接融入量化权重的问题。
    2. **无损合并机制**：微调后通过 W'_int = W_int + Ŵ（带边界检查防溢出）和 z' = z + s·μ 将适配器完全吸收。合并前后前向计算等价（y = (s·W'_int + z')^T x），消除了 baseline 中合并时因适配器精度截断导致的精度损失。
    3. **t-SignSGD 优化器**：针对三值离散约束空间设计，通过 sign(g_t) 和动态百分位阈值 σ_t（top-5%→0.01%）选择性地翻转三值权重，不需学习率缩放，filter 噪声梯度，天然适配 {-1,0,1} 的离散更新空间。
  - 论文方法全栈执行例子：
    - 算法pipeline：GPTQ 量化预训练权重得到 W_int, s, z → 初始化三值适配器 A_T ∈ {-1,0,1}^{D_in×r}（Kaiming normal + 0.75·mean(|A|) 阈值三值化）和 B_T = 0 → 前向：ΔW = A_T B_T, Ŵ_ij = sign(ΔW_ij)·I_{|ΔW_ij|>ω}, W'_int = clamp(W_int + Ŵ, 0, 2^N-1), μ = mean(ΔW - ω·Ŵ), z' = z + s·μ, y = (s·W'_int + z')^T x → 反向（t-SignSGD）：g_t = ∇_{A_T} L, σ_t = top-k% threshold（线性衰减）, A_{T,t+1} = clip(A_{T,t} - sign(g_t)·I_{|g_t|>max(τ,σ_t)}, -1, 1) → 微调完成后合并 W_int ← W'_int, z ← z'，推理时无需适配器计算
    - 系统框架：论文未明确说明
    - 编译框架：使用 Triton 实现自定义 kernel（融合 Ŵ 生成和边界检查为单一 GPU kernel），三元数据类型用 bfloat16 模拟（因 PyTorch 不支持原生 int2/ternary dtype）
    - kernel调度：推理时使用与 baseline 相同的 TritonV2QuantLinear（4/2-bit）或 TorchQuantLinear（3-bit）kernel，但无需适配器开销（合并后），吞吐较 LoRA 提升 1.7x-2.0x
    - 硬件架构：NVIDIA A800 GPU，无专用硬件

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

## LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

- baseline方法是什么？
  传统对数 PTQ（Log2/Log√2/DLog）使用 RTN（rounding-to-nearest）舍入 + 对称量化网格，存在三个根本缺陷：
  1. **对称量化网格**：所有现有对数 PTQ 先取 weight 绝对值再做对称量化，无法匹配 LLM 中常见的非对称 weight 分布（正负比例不均衡）。线性 PTQ 可通过 zero-point 偏移解决，但对数域因零附近的非线性间距无法简单实现。
  2. **对 outlier 高度敏感**：使用 max(|W|) 决定量化范围，单个 outlier 会撑大量化台阶，导致大量正常值被压缩到粗粒度的码字中。
  3. **RTN 舍入次优**：RTN 仅按数值最近原则分配码字，完全不考虑最终任务损失（如激活重建误差）。线性 PTQ 已有 AdaRound/BRECQ 证明了可学习舍入的显著优势，但对数域因 (a) 对数映射非线性、(b) 舍入操作不可微、(c) 混合基离散性，直接迁移可学习舍入不可行。
  - **全栈执行例子（baseline SLogII/DLog + RTN）**：
    - 算法层：取 max(|W|) 确定 s，对 |W| 做 DLog 量化（base-2 或 base-√2），RTN 舍入，固定量化网格
    - 系统框架：论文未明确说明
    - 编译框架：论文未明确说明
    - kernel调度：RTN 舍入无需反向传播，算子直接将 FP16 weight 映射到对数域整数码字
    - 硬件架构：base-√2 乘法需 LUT/multiplier + shifter（AdaLog AE）或 shift-add（Log√2 AE）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LogART 通过 LLR + OHS + HAF 三层创新逐一解决 baseline 缺陷：
  1. **LLR 解决 RTN 次优**：首次将对数域舍入建模为可学习变量 R，用 sigmoid σ(R) 软化为 0~1 之间的选择（floor or ceil）。梯度链：∂L/∂R = 2s·ln2 · M_c ⊙ 2^{-Q_W} ⊙ sign(W) ⊙ [(WX - W̃X)X^T] ⊙ σ'(R) + λ·∂f_reg/∂R。与线性可学习舍入的关键区别是对数域梯度包含指数项 2^{-Q_W}，对小幅值 weight 梯度较小、大幅值 weight 梯度较大——这与对数分布的密度结构一致。
  2. **OHS 逐一解决对称性和 outlier 问题**：(a) ABS 通过自适应边界 l_a 为非对称 weight 分配不同数量的正/负码字——纯 tensor-wise 计算无需校准；(b) SFS 通过块级重建误差搜索缩放因子 s_of 替代 max(|W|)，实现 outlier 自适应裁剪；(c) DBS 自适应分配 base-2:base-√2 码字比例，在硬件效率（base-2 纯移位）和精度（base-√2 细粒度）间分布感知权衡。
  3. **HAF 解决硬件效率与精度矛盾**：用 K-term SDE 展开（如 √2 ≈ 2⁰+2⁻¹）将乘 √2 替换为 shift-add。关键是 HAF 嵌入 LLR 前向传播中，近似误差被梯度下降作为噪声吸收——而非后处理修正。
  - OHS 与 LLR 的协同效应（核心 insight）：论文用三角不等式在 Hessian 加权度量下分解量化误差：||ΔW·H^{1/2}||² ≤ (E₁(OHS) + E₂(LLR))²。E₁ 是量化网格的固有离散化误差（OHS 通过搜索最优 θ*={s_of, n₁, l_a} 最小化），E₂ 是理想投影与 LLR 学习结果的残差（LLR 在 OHS 建立的优质网格上收敛更快更优）。实验证实：OHS+LLR 联合 500 次迭代比纯 LLR 2000 次迭代得到更低的 PPL（31.15 vs 36.27）和更短的总耗时（1.25 min vs 4.00 min）。
  - **全栈执行例子（LogART）**：
    - 算法层：OHS 先搜索 {l_a, s_of, n₁:n₂} 建立最优对数量化网格 → LLR 用 Adam 优化 R 最小化 ||ΔW·X||² + λ·f_reg → HAF 在 forward pass 注入硬件近似噪声 → 收敛后 hard round σ(R)
    - 系统框架：论文未明确说明
    - 编译框架：论文未明确说明
    - kernel调度：在 NVIDIA RTX 5090D GPU 上完成 PTQ 量化（一次离线过程），量化后的 weight 为 INT N-bit 码字，推理时 kernel 使用 LogART AE 设计做 shift-add 而非乘加
    - 硬件架构：LogART AE（Figure 4(e)）——Decoder（组合逻辑）+ Approx 模块（SDE shift-add）+ Shift 模块 + Adder Tree，纯 shift-add 实现，无乘法器。28nm UMC 工艺下面积 53.2 µm²、功耗 3.45 µW（比 BRECQ AE 减少 ~44% 面积和 ~45% 功耗）

## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- baseline方法是什么？
  KV Cache 量化的 naive baseline 是对 key cache 和 value cache **统一使用 per-token 量化**（如 FlexGen 的 4bit group-wise per-token 量化）。方法是将 KV cache 沿 token 维度分组做 round-to-nearest quantization，新到达的 KV tensor 直接 append 到已有 quantized cache 沿 token 维度。这种流式兼容性良好，但存在根本问题：
  - **Key cache per-token 量化**：由于 key cache 中某些固定 channel 存在极大 magnitude outlier（如图2所示），per-token 量化时这些 outlier 的误差会污染同一 group 内的所有 channel，导致 attention score 相对误差高达 47%（vs per-channel 的 9.6%）。将精度降到 INT2 时，LM-Eval 准确率大幅下降（Llama-2-13B CoQA: 66.37→52.93）。
  - **Value cache per-channel 量化**：由于 attention output 是 value cache 的加权求和（attention score 极为稀疏），per-channel 量化导致 token 间量化误差互相混合，attention output 相对误差比 per-token 高约 15×。INT2 per-channel value 量化导致 CoQA 准确率塌陷至 2.88%。
  - **全栈执行例子（baseline FlexGen 4bit per-token）**：
    - 算法层：统一 per-token group-wise INT4 量化 key 和 value cache
    - 系统框架层：Hugging Face Transformers PyTorch 前端，KV cache 按 token 存储量化张量
    - 编译框架层：论文未明确说明
    - kernel调度层：标准 PyTorch 反量化后 matmul，无反量化融合，量化-反量化在 Python 层面完成
    - 硬件架构层：NVIDIA A100 GPU，标准 HBM→SRAM 加载全精度 KV cache

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **KIVI 提出非对称 2bit KV Cache 量化**：key cache 沿 channel 维度做 group-wise per-channel 量化，value cache 沿 token 维度做 group-wise per-token 量化。关键设计：
  1. **异维度量化**：利用 key/value cache 不同的元素分布特征选择不同的量化轴，同时在各自有利的维度上实现 INT2 精度。
  2. **Grouped + Residual 分割**：为解决 per-channel 量化不兼容流式 append 的问题，将 KV cache 分为 grouped 部分（量化存储）和 residual 部分（FP16 保留，滑动窗口大小 ≤ R=128）。grouped 部分每 G=32 个元素一组量化，residual 部分提供全精度局部上下文，对 GSM8K 等困难任务至关重要。
  3. **Tiled Matrix Multiplication**：将 grouped（量化）和 residual（FP16）两部分用分块矩阵乘法分别计算 attention 后拼接，配合 fused dequantization+MatMul CUDA kernel 和 Triton group-wise quantization kernel。
  - **全栈执行例子（KIVI 方法）**：
    - 算法层：非对称量化——key cache per-channel (沿特征维度分组)、value cache per-token (沿序列维度分组)。prefill 时全精度 key/value 传至下一层，仅保留量化版本在内存。decoding 时新 token 先加入 residual FP16 buffer，residual 满（R=128）后量化并移入 grouped 部分。
    - 系统框架层：基于 Hugging Face Transformers 修改 attention 层 KV cache 管理，使用 grouped+residual 分块数据结构，兼容 weight-only 量化（如 GPTQ/AWQ），可实现 2.6× 峰值内存缩减（Llama-2-7B）。
    - 编译框架层：论文未明确说明
    - kernel调度层：CUDA 实现 Q_MatMul（fused dequantization + tiling matmul，避免 FP16 中间结果写回 HBM），Triton 实现 group-wise quantization kernel（在线计算 min/max → scale/zero-point → round-to-nearest INT2）。
    - 硬件架构层：NVIDIA A100 GPU (80GB)，batch size 增大 4×（相同内存限制下），吞吐量提升 2.35× ∼ 3.47×（ShareGPT workload）。

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

## Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4): Analysis and Variations

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

## GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

- baseline方法是什么？
  - **Layer-wise output-based PTQ**（如 GPTQ、QTIP、GPTVQ 1D、SpinQuant 等）：逐层最小化 `||XW - XŴ||_F^2`，即量化前后 layer output 的 MSE。将所有 hidden features 平等对待，忽略不同 output feature 对 end loss 的差异化影响。
  - **SqueezeLLM**：使用 diagonal Fisher 近似（weighted k-means objective `(ŵ - w)^T diag(F)(ŵ - w)`），考虑了 end loss 梯度但完全忽略 off-diagonal 的 cross-weight interactions，导致 Hessian 近似严重不准确。
  - 全栈执行例子：以 Llama-2-7B 推理一次 token 为例——①算法pipeline：GPTQ 对每层 W 做 uniform scalar quantization，minimize ||XW - XŴ||²，所有 output channels 权重的 error 被同等对待；②系统框架：使用 PyTorch + torch.compile 优化推理管线，GPU kernel 使用 LUT-GEMM/Any-Precision-LLM 解码量化权重；③编译框架-④kernel调度：论文未明确修改编译框架或 kernel；⑤硬件架构：论文未涉及硬件架构设计。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **GuidedQuant**：基于 end loss 对 layer output 的一阶 Taylor 展开，构建加权 layer-wise objective：`||(∂ℓ/∂Z) ⊙ (XW - XŴ)||_F^2`。这等价于 block-diagonal Fisher 近似——保留每个 output channel 内 weights 的 d_in×d_in Fisher block `F_j = (1/n) Σ (∂ℓ_i/∂w_j)(∂ℓ_i/∂w_j)^T`，忽略跨 channel 和跨层交互。
  - **Averaging approximation**：将 d_out 个 d_in×d_in Hessian 矩阵按 g 组平均为 g 个 `H̄_k`，大幅降低存储（从 Θ(d_in² d_out) 到 Θ(d_in² g)）和时间复杂度，使方法可扩展至 70B 级 LLM。
  - **LNQ**：针对 non-uniform scalar quantization，codebook 用闭式最小二乘解代替 GPTVQ 1D 的梯度下降；assignment 用 cyclic CD 代替 GPTQ，保证目标函数单调递减且证明收敛。
  - 全栈执行例子：以 Llama-2-7B 一次推理为例——①算法pipeline：单次 backward pass 计算 ∂ℓ/∂Z，按 g=4 组聚合为 H̄_k；对每组权重独立调用 LNQ（Cholesky 分解 H̄_k → codebook 闭式解 → cyclic CD 优化 assignment → 迭代 T=2 轮，K=4 CD 循环），最终得到量化权重 Ŵ；②系统框架：量化后权重使用 Any-Precision-LLM kernel（non-uniform scalar）或 QTIP HYB kernel（vector）在 PyTorch + torch.compile 管线中推理，吞吐与 baseline 持平（如 2-bit Llama-2-7B: 244.4 tok/s vs SqueezeLLM 245.1 tok/s），perplexity 显著更优（Wiki2-4K: 8.83 vs 39.58）；③编译框架-④kernel调度：论文未明确修改编译框架或 kernel；⑤硬件架构：论文未涉及硬件架构设计。

- baseline方法是什么？
  Baseline 是 **Shift Quantization**（INQ [26]、ADMM [14]），将权重量化为 2 的幂次值 `{0, ±1, ±2, ±4, ...}`，使乘法变为 bit-shift 操作。该方法在密集 CNN 上表现良好，因为权重分布较均匀地覆盖了量化层级。

  **Shift Quantization 全栈执行例子**（ResNet-18, 5-bit shift quantization, ImageNet）：

  - **算法pipeline**：全精度权重 θ → 逐层 shift quantization `Q^{shift}_{n,b}[θ] = s * 2^{e-b}`（s ∈ {-1,0,1}, e ∈ [0,2^k-1]）→ 量化精度层级均匀分布在 0 两侧且越远离 0 越稀疏 → 推理时卷积乘法替换为 `x << (e-b)` → 输出结果。
  - **系统框架**：基于 PyTorch/Caffe 的量化脚本，无 Serving 框架修改。Mayo 框架（https://github.com/deep-fry/mayo）提供量化工具链。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：CPU/GPU 上 bit-shift 替代乘法（无专用 kernel）。FPGA 加速器由 Mayo 框架自动生成 [24]。
  - **硬件架构**：FPGA 加速器（ICFPT 2019），卷积层展开为 bit-shift + 整数加法数据路径，无需乘法器。

  **Baseline 的核心缺陷**：
  1. **剪枝后的层权重大量远离零值**（Figure 1c, 1d）：细粒度剪枝（Dynamic Network Surgery）鼓励权重在训练中移动远离零值以形成高稀疏度，导致剪枝后稀疏层的权重分布集中在远离零的区域（±0.05~±0.15 附近），形成一个"中空"分布。
  2. **量化层级严重低效利用**：Shift quantization 在零附近层级最密集（±1, ±2, ±4, ±8, ...），恰好是剪枝后权重最稀疏的区域。大量近零量化层级未使用，而远离零的权重被迫用粗粒度量级表示，造成量化误差大且层级浪费（Figure 1d）。
  3. **一刀切策略不考虑逐层差异**：所有层使用相同的 shift quantization，不考虑剪枝后各层稀疏度不同导致的权重分布差异。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Focused Quantization (FQ)**，将量化 effort（即量化层级资源）重新集中到剪枝后权重实际分布的高概率区域：

  **(1) Gaussian Mixture Model + EM for Recentralized Quantization**
  - 对每层非零权重拟合 2 分量 GMM，找到两个高概率密度聚类区域。
  - 对每个聚类分别做 recentralized quantization：`Q_c^{rec}[θ] = Q^{shift}[(θ-μ_c)/σ_c] * σ_c + μ_c`
  - **解决缺陷 1, 2**：不再是"在零附近密集量化"，而是"在权重实际所在区域密集量化"。通过先平移（减 μ_c）再缩放（除 σ_c），使每个聚类相对自己的均值零中心化，然后在这一小区域内使用 shift quantization 的全部精度。
  
  具体而言：
  ```
  # Baseline: 对全范围做 shift quantization
  # 量化层级: {0, ±0.001, ±0.002, ±0.004, ...}  (bias调整后)
  # 剪枝后的权重全在 ±0.05~0.15 → 只用了最外层的几个粗粒度层级

  # FQ: Recentralized
  # Cluster +: μ₊≈0.10, σ₊≈0.02
  #   Normalize: (θ-0.10)/0.02 → 范围[-2, +2]
  #   Shift quantize within [-2,+2]: 细粒度量级利用全部层级
  #   De-normalize: back to original scale
  # Cluster -: μ₋≈-0.10, σ₋≈0.02 (同理)
  ```

  **(2) 自适应 Wasserstein 判定切换机制**
  - 用 2-Wasserstein 距离 `W(c₁,c₂) = ((μ₊-μ₋)² + (σ₊-σ₋)²) / σ²_global` 判断两分量是否充分分离。
  - `W < w_sep`（默认 2.0）：分量高度重叠 → 退化为普通 shift quantization，且因无需 component selection bit，精度等效高 1 bit。
  - **解决缺陷 3**：自适应地为每层选择最适合的量化方式，不强制所有层使用同一策略。

  **(3) 硬件友好的设计选择**
  - μ₊, μ₋ 量化为最近的 2 的幂次值（保持 shift 性质）。
  - σ₊ = σ₋（约束相等），接入逐层 α 缩放因子中。
  - α 融入 BN 融合，推理时无乘法。
  - **结果**：5-bit FQ（内部 3-bit 无符号 shift + 1-bit sign + 1-bit component select）比 LQ-Net (2-bit) 和 ABC-Net (5-bit) 硬件效率更高——比 3-bit shift quantization 仅多 0.15% gates，但精度显著更高（Table 4）。

  **FQ 全栈执行例子**（ResNet-18, 5-bit FQ + 剪枝, ImageNet, 自定义加速器）：

  - **算法pipeline**：剪枝后权重 W（sparse, 非零权重"中空"分布）→ 逐层拟合 2-GMM（EM）→ Wasserstein 判定：足够分离 → recentralized quantization，否则 → shift quantization → INQ 增量量化 fine-tune（25%→50%→75%→87.5%→100%）→ Huffman 编码 → 输出压缩模型。
  - **系统框架**：Mayo 框架（https://github.com/deep-fry/mayo），包含 pruning（Dynamic Network Surgery）、FQ quantization、Huffman encoding 完整流水线。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：bit-shift + 整数加法替代浮点乘法（CPU/GPU）。FPGA 加速器由 Mayo 框架自动生成 [24]。
  - **硬件架构**：自定义加速器 dot-product 单元（Figure 4）：整数激活 × shift-quantized 权重 → bit-shift（根据权重指数 e）→ 整数累加 → 最终 α 缩放（可融入 BN）。275.6M 逻辑门（5-bit FQ），远低于 LQ-Net/ABC-Net。无乘法器阵列，无需 N 路并行二值卷积。

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


## First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

- baseline方法是什么？
  Baseline 是 **GPTQ**（Frantar et al. 2022），一种基于 OBS→OBC 理论演进的经典 PTQ 方法。GPTQ 全栈执行例子（Llama3-8B, W3A16, group_size=128, A800 GPU）：

  - **算法pipeline**：校准数据 X（128 samples, seq_len=2048）→ 对每层权重 W：计算 H = XX^T → Cholesky 分解 H^{-1}=LL^T，保留上三角 T=L^T → 按 block（B列）逐列量化：(a) Q_{:,j} ← quant(W_{:,j})（RTN 量化）；(b) 补偿误差 δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}（仅二阶项）；(c) lazy update 批量更新后续列。核心假设：全精度模型已收敛到局部最优，一阶梯度 ≈ 0，可省略。
  - **系统框架**：自实现 PTQ 脚本（PyTorch），无 Serving 框架修改。量化后导出为 GPTQ 格式，部署至 vLLM 推理。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel（torch.matmul、torch.nn.functional.linear），无自定义 kernel。vLLM 端使用其内置的 W4A16 GEMM kernel。
  - **硬件架构**：NVIDIA A800-80GB GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **一阶项被错误忽略**：GPTQ 沿袭 OBD/OBS 假设"模型已充分优化→一阶梯度为0"，但逐列量化过程中，先量化列的补偿项 δw 持续更新后续列，导致 latent weights W 与原始 full-precision weights 𝕎 产生累积偏差。此偏差在后续列的损失函数 Taylor 展开中引入不可忽略的一阶梯度 g = ∂E/∂w，GPTQ 的纯二阶近似式 δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:} 在存在非零梯度下不再是理论最优解。
  2. **GPTAQ 的高开销替代方案不理想**：GPTAQ 尝试通过非对称校准改善量化，但引入了显著额外计算（Llama3-8B 量化时间从 825.50s 增至 1112.20s，+34.7%），且精度提升不稳定。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **FOEM（First-Order Enhanced Method）**，在 GPTQ 补偿框架中显式引入一阶梯度项：

  **(1) 保留一阶项重新推导最优补偿**
  从完整 Taylor 展开 δE = gδw^T + ½δwHδw^T 出发，构建带约束 Lagrangian（约束条件：e_q δw^T + w_q − ŵ_q = 0），求导得理论最优：
  δw = −(w_q − ŵ_q − gH^{-1}e_q^T)/[H^{-1}]_{qq} · [H^{-1}]_{q,:} − gH^{-1}
  对比 GPTQ 的 δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}，多了梯度相关的分子修正项和整体梯度项。

  **(2) 梯度近似消除计算开销**
  直接反向传播求 g 开销巨大。FOEM 利用 Taylor 展开近似：
  g(W) ≈ g(𝕎) + (W − 𝕎)H ≈ (W − 𝕎)H（因 g(𝕎)≈0，全精度模型已训练到最优）
  引入稳定化因子 β=0.1：g ≈ β(W − 𝕎)H
  将近似代入理论解后，H 和 H^{-1} 在代数运算中**自动消去**，最终补偿项：
  δw = −((w_q − ŵ_q) − β(w_q − 𝕎e_q^T))/T_{qq} · T_{q,q:} − β(W − 𝕎)
  仅需 T（Cholesky 因子）和权重差分运算，无矩阵乘法，无 Hessian 显式求逆。

  **(3) 全栈执行对比（Llama3-8B, W3A16, A800）**
  - **算法pipeline**：流程同 GPTQ，但每列补偿时额外：(a) 计算权重偏差 W_{:,j} − 𝕎_{:,j}；(b) 分子中减去 β(W_{:,j} − 𝕎_{:,j})；(c) 补偿完当前列后，全局减去 β(W_{:,j} − 𝕎_{:,j})。这些差分运算 O(n) 量级，量化时间 828.90s vs GPTQ 825.50s（仅 +0.4%），远优于 GPTAQ 的 1112.20s（+34.7%）。
  - **系统框架**：同 GPTQ，量化为 GPTQ 格式后部署 vLLM。W4A16 推理：input tokens/s 从 FP16 的 184.11 → 250.26（+36%），output tokens/s 从 470.11 → 616.01（+31%）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel + vLLM 内置量化 kernel，无自定义 kernel。
  - **硬件架构**：NVIDIA A800-80GB GPU，无自定义硬件。

  **关键结果**：
  - W3A16 Llama3-8B：WikiText2 PPL 从 GPTQ 9.86 → FOEM 8.32（↓15.6%），MMLU 从 GPTAQ 53.8% → FOEM 56.1%
  - W4A4KV4 Llama3-8B：WikiText2 PPL 从 GPTQ 8.55 → FOEM 8.35（↓0.20）
  - 跨架构泛化：Mamba-1.4B（SSM）W3A16 PPL 从 GPTAQ 14.10 → FOEM 13.91


## FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

- baseline方法是什么？
  Baseline 是标准 **FedAvg** 及现有量化FL方法（FedPAQ、FedHQ+）。FedAvg的典型全栈执行例子（ResNet-18, 100 clients, CIFAR-100, RTX 4090）：

  - **算法pipeline**：Server每轮广播GMP W_g → 每个client用本地non-i.i.d.数据SGD训练K步得到LMP → client计算LMPU ΔW_i = W_i - W_g → client传输全精度（32-bit）ΔW_i至server → server加权聚合 Δ = Σ h_i ΔW_i → 更新GMP W_g ← W_g + Δ。若加量化（FedPAQ）：用absmax scaling将ΔW_i缩放到[-1,1] → uniform quantization到固定B-bit → 概率舍入避免QL集中 → 传输量化值+scale。FedHQ+在此基础上对每个client按量化误差加权。
  - **系统框架**：PyTorch，自实现FL simulator（100 clients × 5% participation），无Serving框架。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准PyTorch CUDA kernel（conv2d、BN/GN、linear），无自定义kernel。
  - **硬件架构**：NVIDIA RTX 4090 GPU，无自定义硬件。

  **Baseline 的核心缺陷：**
  1. **Client drift源于梯度偏差**：在non-i.i.d.数据下，local SGD梯度包含两个偏差分量：(a) 与当前LMP对齐的分量——local模型过拟合本地数据导致参数偏离GMP；(b) mini-batch梯度均值分量——biased toward local data distribution。这两个分量叠加导致各client的LMPU方向不一致，全局聚合后偏离最优参数。
  2. **absmax scaling对离群值敏感**：FedPAQ和FedHQ+使用absmax将张量缩放到[-1,1]，outlier会过度扩展动态范围，在低比特（1-bit/2-bit）下导致严重的underflow——大部分正常值被压缩到极窄区间内，量化后信息丢失严重。
  3. **Uniform quantization浪费容量**：LMPU实际近似正态分布（密集区域在均值附近），但UQ将范围均匀划分为2^B等间隔，在密集区域精度不足、稀疏区域容量浪费。
  4. **现有NUQ方法低比特乏力**：NF（NormalFloat）和FP（Floating Point）在1-bit/2-bit下性能急剧退化（如CIFAR-100 α=0.1时NF仅24.0%，FP仅7.0%），因为它们的QL设计未针对极端低比特优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **FedWSQ = WS梯度过滤 + DANUQ分布感知量化**：

  **(1) WS梯度过滤解决client drift**
  Baseline local training的梯度直接沿raw gradient方向更新。FedWSQ在每层前向传播前对权重向量w_{n,m}应用WS：w̃_{n,m} = (ρ/σ(w_{n,m}))(I - P_1) w_{n,m}（先减均值去除DC分量，再除以标准差归一化）。反向传播时梯度经历双重投影（见Eq.6）：∂L/∂w_{n,m} = (ρ/σ)(I - P_1)(I - P_{w̃_{n,m}}) ∂L/∂w̃_{n,m}：
  - 第一重投影 (I - P_{w̃_{n,m}})：移除与WSP向量对齐的梯度分量（即local overfitting偏好的方向）
  - 第二重投影 (I - P_1)：移除mini-batch梯度均值分量
  结果：梯度被投影到 span{w̃_{n,m}, 1}^⊥，仅保留对global convergence有益的方向。这等价于一种隐式正则化，无需修改loss函数或优化器结构。

  FedWSQ传输PSP而非WSP（区别于FedWon），通过梯度过滤隐式纠正偏差而不强制client间参数统计一致，保留了本地适应性信息。

  **(2) DANUQ以标准差做scale + 正态分布最优QLs解决量化瓶颈**
  - scaling：不用absmax（对outlier敏感），改用LMPU的标准差σ作为scale factor。因为σ更稳健且与N(0,1)假设一致。Global EMA scale vector s_g = (1-β)s_g + β·mean(s_i) 在各client间共享，保证量化一致性。
  - QLs预计算：假设归一化后LMPU ∼ N(0,1)，求解 min_{q_1,...,q_R} E[(x-q)^2] 得到最优QLs。因closed-form不可得，用暴力搜索在合理范围内穷举，结果为：1-bit[-0.798,0.798] / 2-bit[-1.224,0,0.765,1.724] / 4-bit[16个非均匀间隔QLs]。这些QLs密集分布于高概率密度区域（均值附近），稀疏分布于尾部，比UQ同位数下信息损失小。
  - 无额外通信开销：QLs固定预计算，无需每轮传输量化参数或学习步长/零点。

  FedWSQ全栈执行例子（对比baseline）：
  - **算法pipeline**：Server广播 (W_g, s_g) → Client local training（WS前向+双投影梯度过滤，K步SGD） → Client DANUQ量化（ΔW_{i,l}/s_{g,l} → 查表映射到预计算QL → 得B-bit整数index） → Client上传 (ΔW̄_i, s_i) （量化值+1个scale/Layer的float） → Server dequantize（查表+乘scale还原全精度） → 聚合+EMA更新scale。
  - **系统框架**：PyTorch自实现FL simulator（同baseline），DANUQ为纯Python/CUDA查表操作，不依赖额外框架。
  - **编译框架/kernel调度/硬件架构**：论文未明确说明，与baseline相同的PyTorch CUDA kernel执行，无自定义kernel或硬件。

  **关键设计选择 vs Baseline缺陷对应**：
  - Baseline缺陷1（梯度偏差）→ WS投影过滤 (I-P₁)(I-P_{w̃}) 双重投影
  - Baseline缺陷2（absmax对outlier敏感）→ 标准差scaling + global EMA，对outlier更稳健
  - Baseline缺陷3（UQ容量浪费）→ DANUQ基于N(0,1) PDF设计非均匀QLs，密集区域细粒度、稀疏区域粗粒度
  - Baseline缺陷4（现有NUQ低比特弱）→ DANUQ直接为1/2/4-bit暴力搜索最优QLs，CIFAR-100 α=0.1 1-bit从NF 24.0%提升至84.8%


## YOCO (You Only Cache Once): Decoder-Decoder Architectures for Language Models

- baseline方法是什么？
  Baseline 是标准 **decoder-only Transformer**（Llama 架构优化版），包含 RMSNorm、SwiGLU、RoPE、Grouped-Query Attention (GQA) 等现代改进。每层执行全局因果 self-attention：Q=XW_Q, K=XW_K, V=XW_V，计算 Attention(Q,K,V)=softmax(QK^T/√d_k+M)V，需存储全部 L 层的 per-token KV cache。

  Baseline 全栈执行例子（Transformer 7B, 512K context, 4×H100-80GB）：
  - 算法pipeline：序列 x → Embedding → L=32 层 decoder：每层 Masked MHA（QKV 投影 → QK^T/√d → causal mask → softmax → ×V → output proj，Flash-Decoding + kernel fusion 优化）→ SwiGLU FFN → 残差连接 → classifier。prefill 阶段：512K tokens 全部并行前向，需存储 32 层 × 512K × 2(KV) × d_head × h_kv × 2bytes。decode 阶段：每步从 HBM 读取全部 KV cache，HBM 带宽是瓶颈。
  - 系统框架：HuggingFace Transformers / 自定义推理框架，Flash-Decoding attention kernel，Triton fused kernel。
  - 编译框架：论文未明确说明。
  - kernel调度：Flash-Decoding（适用于长序列 attention）、kernel fusion（融合 LayerNorm/QKV 投影等操作）。H100 GPU 上执行。
  - 硬件架构：NVIDIA H100-80GB GPU（Ampere 下一代，Hopper 架构），无自定义硬件。

  **Baseline 的核心缺陷：**
  1. **KV Cache 内存随 L 线性增长**：每层都需存储 N 个 token 的 K,V，总 KV cache = O(L×N×D)。65B 模型 512K tokens 时 KV cache 占用约 86GB，超过单张 H100-80GB 容量。这是 LLM 长上下文推理的主要内存瓶颈。
  2. **Prefill 延迟 O(N²)**：softmax(QK^T) 的计算复杂度与序列长度平方成正比。7B 模型 1M tokens prefill 需约 380s（4×H100），严重影响用户体验。
  3. **长序列训练通信瓶颈**：分布式长序列训练时，每层的 all-gather 通信随序列长度和层数增加，吞吐受限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **YOCO (You Only Cache Once)** decoder-decoder 架构，通过三个核心设计解决 baseline 缺陷：

  **(1) Self-Decoder + Cross-Decoder 分离解决 KV Cache 内存瓶颈**
  Baseline 每层都自注意力和自产 KV cache。YOCO 将 L 层平分为 Self-Decoder（前 L/2 层）和 Cross-Decoder（后 L/2 层）。Self-Decoder 使用高效自注意力（gated retention 或 sliding-window attention），仅需 O(1) 常量内存（如 retention 的 recurrent state S ∈ R^{d×d}，或 sliding-window 的固定窗口 C）。Cross-Decoder 所有层共享同一组全局 KV cache（K̂, V̂），仅由 Self-Decoder 最终输出生成一次。总 KV cache = O(N + CL) ≈ O(N)（C 为常量）vs Baseline O(NL)，大约节省 L 倍内存。65B 模型 KV cache memory 降低约 80×。

  **(2) Prefill Early Exit 解决 Prefill 延迟瓶颈**
  Baseline prefill 必须执行全部 L 层的前向计算（含 O(N²) attention）。YOCO 的 Cross-Decoder 仅依赖 Self-Decoder 的输出 K̂, V̂，因此 prefill 阶段可在 Self-Decoder（L/2 层）完成后立即退出。又因 Self-Decoder 使用高效 attention（如 retention 的 chunkwise recurrent），prefill 复杂度从 O(LN²D) 降至 O(LND)（线性于 N）。512K context：prefill 从 180s 降至 <6s（带 Flash-Decoding 优化的 Transformer baseline vs YOCO），1M context 加速 71.8×。

  **(3) Chunk Parallelism 解决分布式训练通信瓶颈**
  Baseline 的数据并行/序列并行中，每层都需 all-gather 通信。YOCO 的 Cross-Decoder 解耦了层间注意力依赖：Self-Decoder 仅需相邻设备的边界通信（如 retention 的 recurrent state 传递，或 sliding-window 的窗口边界）；Cross-Decoder 的 K̂, V̂ 仅需一次 all-gather（而非每层一次），大幅减少通信频率和 GPU memory fragmentation。

  论文方法全栈执行例子（YOCO_gRet 3B, 512K context, H100-80GB）：
  - 算法pipeline：序列 x → Embedding X^0 → **Self-Decoder（L/2=13 层，gated retention）**：每层 recurrent/chunkwise 计算 → S_n = γ_n S_{n-1} + K_n^T V_n → O_n = Q_n S_n → GroupNorm + swish gate → SwiGLU FFN → 输出 M = X^{L/2} → **生成全局 KV cache**：K̂ = LN(M)W_K, V̂ = LN(M)W_V（单次，共享给所有 Cross-Decoder 层）→ **Cross-Decoder（13 层，cross-attention）**：Q̂^l = LN(X^l)W_Q^l → Attention(Q̂^l, K̂, V̂) → SwiGLU FFN → 输出 X^L → classifier。**Prefill**：仅执行 Self-Decoder + 生成 K̂,V̂（提前退出，略过 Cross-Decoder），13 层而非 26 层。**Decode**：Self-Decoder 用 recurrent（O(1) state），Cross-Decoder 用标准 attention 复用 K̂,V̂。结果：GPU memory 12.4GB（Transformer 9.4× more），prefill <6s（Transformer 180s），throughput 43.1 tok/s（Transformer 4.5 tok/s）。
  - 系统框架：内部 CUBE 分布式训练系统（SuperScaler-based, https://github.com/microsoft/nnetscaler），HuggingFace Transformers 兼容 API。H100 GPU。
  - 编译框架：论文未明确说明。
  - kernel调度：Triton kernel：gated retention 的 chunkwise recurrent（prefill, chunk=256）+ recurrent（decode），基于 FLA 库。Baseline 使用 Flash-Decoding + kernel fusion 优化。
  - 硬件架构：NVIDIA H100-80GB GPU。无自定义硬件。论文在 Conclusion 中展望 YOCO + BitNet + Groq 的组合可进一步将部署成本降低数个数量级。

  关键设计动机映射：
  - Transformer 每层 KV cache 内存 O(LND) → YOCO 的单层全局 KV cache O(ND) + Self-Decoder 常量 cache O(CL)
  - Transformer prefill 延迟 O(LN²D) → YOCO prefill early exit + 高效 attention O(LND)
  - Transformer 分布式训练每层 all-gather → YOCO Cross-Decoder 仅一次 all-gather（KV cache） + Self-Decoder 仅边界通信
  - 不同场景可选用不同 Self-Decoder 实现：gated retention（性能最优）或 sliding-window attention（实现简单）

## Hymba: A Hybrid-head Architecture for Small Language Models

- baseline方法是什么？
  Baseline 包含三类架构：(1) **纯 Transformer**（Llama 架构）：全部 L 层使用 global causal self-attention，KV cache = O(L×N×d)，内存随层数和序列长度线性增长；(2) **纯 SSM**（Mamba/Mamba2）：所有层使用 state space model，O(1) 常量 cache，但 recall 能力弱（Mamba 300M recall accuracy 仅 19.23% vs Transformer 39.98%）；(3) **Sequential Hybrid**（Samba/Jamba/Zamba）：交替堆叠 Mamba 层和 Attention 层（如 Mamba-FFN-Attn-FFN 重复），但两种层独立处理输入，缺乏协同，信息瓶颈时后续层难以补偿。

  **纯 Transformer (Llama) 全栈执行例子**（Llama3-1B, 8K context, A100）：
  - **算法pipeline**：序列 X → Embedding → L 层 decoder：每层 Masked MHA（QKV 投影 → QK^T/√d → causal mask → softmax → ×V → output proj）→ SwiGLU FFN → 残差连接 → classifier。KV cache = L × N × 2 × d_head × h_kv × 2bytes（FP16），8K 下 Llama3-1B cache ~262MB。Recall accuracy 75.95%（SWDE），commonsense avg 52.82%。Throughput 721.1 tok/s at 8K/bs128（300M scale）。
  - **系统框架**：HuggingFace Transformers + PyTorch。lm-evaluation-harness 评估。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：PyTorch CUDA kernel（标准 FlashAttention 优化），无自定义 kernel。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **Transformer KV cache 内存爆炸**：KV cache = O(L×N×d)，长序列推理时 HBM 容量成为瓶颈。Llama-1B 8K cache=262MB，而 Hymba 同等规模仅需 79MB。
  2. **SSM recall 能力严重不足**：Mamba 的常量大小 state 无法精确存储和检索历史信息。Mamba 300M recall acc 仅 19.23%（vs Transformer 39.98%），SQuAD-C 仅 36.43%（vs Transformer 75.95%）。Attention sink 问题严重：>50% attention 聚焦于 BOS token（Figure 7）。
  3. **Sequential hybrid 缺乏协同**：Samba 式交替堆叠导致 Mamba 层和 Attention 层独立处理输入，当某一层类型不适合当前 token 的处理需求时，信息瓶颈无法被后续层充分补偿。Samba 1B avg 52.83%，与纯 Llama3 的 52.82% 几乎持平，recall 甚至下降（SWDE 30.00% vs Llama3 75.95%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Hymba hybrid-head 并行架构**，通过四个核心设计逐一解决 baseline 缺陷：

  **(1) Hybrid-Head 并行融合解决 sequential hybrid 缺乏协同（解决缺陷 3）**
  Baseline Samba 将 Mamba 和 Attention 交替堆叠，每层只执行一种操作。Hymba 在同一层内并行放置 attention heads 和 SSM heads：
  `Y = W_out_proj( β₁·norm(M_attn·X̃) + β₂·norm(M_ssm·X̃) )`
  两者同时处理相同输入，SSM 提供全局上下文摘要（fading memory），Attention 提供高分辨率局部召回（snapshot memory），输出经可学习 per-channel 重缩放 β₁, β₂ 后融合。ERF 分析证明 parallel 结构的有效感受野比 sequential 大一个数量级（Fig. 11），cache size 相当。300M scale 下 hybrid-head avg accuracy 45.19%（+1.12% over sequential 44.07%），recall 49.90%（+4.74% over sequential 45.16%）。

  **(2) KV Cache 优化解决 Transformer 内存瓶颈（解决缺陷 1）**
  - **SSM 摘要全局上下文 → 仅 3 层 global attention**：SSM heads 已经 summarize 了全局 context，因此可以大胆地用 local SWA 替代绝大多数 global attention。仅保留首/中/末 3 层为 global attention 即可恢复 recall 能力。对比实验：全 SWA 时 recall 从 49.90% 骤降至 29.78%；恢复 3 层 global attention 后 recall 回升至 48.79%。
  - **Cross-layer KV sharing**：相邻两层共享同一 KV cache（每 2 层一组），节省参数和 cache。同时将节省的参数重新分配到其他组件，提升 commonsense accuracy +0.60%。
  - **结果**：8K cache size 从 414.7MB（纯 Transformer）降至 39.4MB（10.5× reduction），throughput 从 721.1 tok/s 提升至 2756.5 tok/s（3.8×）。

  **(3) Meta Tokens 解决 Attention Sink 和 Recall 不足（解决缺陷 2 的 recall 部分）**
  Baseline Transformer 中 >50% attention 聚焦于 BOS token（"forced-to-attend"），浪费 attention 预算。Hymba 引入 128 个 learnable meta tokens 前置到输入：
  `X̃ = [r₁, r₂, ..., r₁₂₈, x₁, x₂, ..., x_n]`
  Meta tokens 的作用：(a) 作为 attention sink 的"吸收器"，吸收原本会浪费在 BOS 上的 attention，使后续 token 能关注有意义的信息；(b) 作为 learned cache initialization（推理时离线预计算 K/V/SSM 状态）；(c) 封装压缩的世界知识（不同 domain 的 prompt 激活不同的 meta tokens，Fig. 5）。引入 meta tokens 后：300M recall 从 48.04% 提升至 51.79%（+3.75%），attention map entropy 整体下降（Fig. 15），说明 attention 更集中于信息量大的 token。

  **(4) Attention Map 解耦增强表达能力（辅助解决缺陷 2,3）**
  Hymba 的 attention map 由三部分贡献组成：meta tokens + sliding window attention + SSM（Fig. 6）。相比 Transformer 中 'BOS' 和 'Self' 占比过高的失衡分布，Hymba 的 'Cross' attention（token 间信息交互）比例更高，分布更均衡（Fig. 7）。这意味着 hybrid-head 设计有效解耦了不同类型的信息处理：SSM 关注当前 token（Self），Attention 关注跨 token 关系（Cross），Meta tokens 吸收 attention sink 释放 attention 预算。

  **论文方法全栈执行例子**（Hymba-1.5B, 8K context, A100）：
  - **算法pipeline**：输入 X → prepend 128 meta tokens → X̃ → 32 层 hybrid-head block：每层并行执行 {Sliding Window Attention（仅 3 层为 global）+ Mamba SSM} → β 归一化重缩放融合 → SwiGLU FFN → 残差连接。KV cache：仅 global attention 层存储 + 每 2 层共享（cross-layer sharing）。SSM state：recurrent h_i（常量大小）。Prefill 时 meta tokens 的 K/V/SSM 状态从预计算值加载。8K cache=79MB（vs Llama3-3B 918MB），throughput=664 tok/s（vs Llama3-3B 191 tok/s）。Avg accuracy 61.06%（vs Llama3-3B 59.74%）。
  - **系统框架**：PyTorch + HuggingFace Transformers。训练：128×A100，WSD scheduler。后训练：LMFlow toolkit（FFT → DPO）。HuggingFace 发布 Hymba-1.5B-Base/Instruct。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：PyTorch CUDA kernel（标准 Mamba selective scan kernel + FlashAttention），无自定义 kernel。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  关键设计动机映射：
  - Sequential hybrid 缺乏协同 → Hybrid-head 并行融合（同层 attention + SSM，统一对称公式 Eq.3）
  - Transformer KV cache 内存 O(LND) → SSM 摘要全局 + 仅 3 层 global attention + cross-layer KV sharing → O(N + CL)
  - SSM recall 能力弱 → Attention heads 提供高分辨率 snapshot memory 补充 SSM 的 fading memory
  - Attention sink 浪费 >50% attention → Meta tokens 吸收 sink + 作为 learned cache initialization
  - Attention 分布失衡（BOS/Self 主导） → Hybrid-head 解耦：Meta tokens 吸收 BOS，SSM 处理 Self，Attention 处理 Cross

## Bridging the Gap Between Promise and Performance for FP4 Quantization

- baseline方法是什么？
  Baseline 是标准 **RTN（Round-to-Nearest）量化**直接应用于 MXFP4 和 NVFP4 微缩放格式，配合 absmax scaling。具体流程：(1) 将权重/激活按 G=32（MXFP4）或 G=16（NVFP4）分组；(2) 每组用 absmax 计算 shared scale（MXFP4 scale 量化为 E8M0 即 power-of-two，NVFP4 scale 量化为 E4M3 即完整 FP8）；(3) 以 FP4 E2M1 格式对归一化后的元素执行 RTN 量化。也对比了 GPTQ（标准 INT GPTQ 直接套用到 FP4）、SmoothQuant（对角 rescaling 迁移激活异常值到权重）、QuaRot/SpinQuant（全局 Hadamard 旋转后 RTN）。

  Baseline 全栈执行例子（Llama-3.1-8B-Instruct MXFP4 RTN W4A4）：
  - 算法pipeline：加载 FP16 权重 → 逐层线性层：权重按 G=32 分组 → 每组 absmax scale s_G → s_G 量化为 E8M0（power-of-two）→ 权重归一化后 RTN 量化到 E2M1 FP4 网格 → 激活同理 → 推理：Q(WH_k)@Q(XH_k)^T（无旋转时 H_k=I）。MXFP4 RTN 平均 accuracy recovery 仅 87.83%（FP16=78.93, RTN=69.32）。NVFP4 RTN recovery=94.67%。
  - 系统框架：PyTorch 模拟量化（fake quantization），HuggingFace Transformers。校准集 FineWeb 1024 sequences。
  - 编译框架：论文未明确说明。
  - kernel调度：标准 PyTorch FP16 GEMM，模拟量化仅用于精度测量。
  - 硬件架构：论文未明确说明（实验在 GPU 上执行 PyTorch 模拟量化）。

  **Baseline 的核心缺陷（通过量化误差分析揭示）：**
  1. **MXFP4 的 E8M0 power-of-two scale 引发严重量化误差**：scale 量化为 power-of-two（E8M0）在保持硬件乘法简化的同时，引入了较大近似误差。MXFP4 RTN 下 MLL 平均下降 ~10%，显著劣于 NVFP4 和 INT4。MSE 分析显示 MXFP4 的 top-element 误差随 group size 增大而保持恒定（受限于 E2M1 而非 E8M0），而 per-element MSE 随 G 增大增长。
  2. **NVFP4 的小 group size（G=16）天生做异常值抑制**：传统异常值缓解技术（如 SmoothQuant 的 per-channel scaling）在 NVFP4 的 G=16 下被证明无效——小 group 已经通过细粒度 absmax scaling 隐式处理了异常值。因此 NVFP4 RTN 即使不加任何额外技术已表现良好。
  3. **Hadamard 旋转对 NVFP4 精度有负面影响**：分析证明（Lemma 1-2），对 Laplace 分布（原生权重/激活）应用 Hadamard 旋转转为 Normal 分布后，在小 G 下 MSE 增大（因为 top-element 误差被均匀扩散到整组）。NVFP4 的 G=16 恰好在此区间，所以 RTN+HT 比 RTN 精度更差。这解释了为何 QuaRot/SpinQuant 在 NVFP4 上无效甚至有害。
  4. **标准 GPTQ 未针对 FP4 格式优化**：直接套用 INT GPTQ 的 absmax scaling + uniform grid 到 FP4 非均匀网格，未利用 MSE-optimized grid、未处理旋转后的格式适配。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MR-GPTQ（Micro-Rotated-GPTQ）**，通过量化误差理论分析驱动三个核心创新逐一解决 baseline 缺陷，并配套 **QuTLASS** GPU kernel 实现零开销部署：

  **(1) 量化误差分析驱动方法选择（解决缺陷 1-3 的根源问题）**
  论文建立了 MXFP4/NVFP4 的理论 MSE 模型（Laplace→原生权重、Normal→旋转后权重），推导 per-element MSE 和 top-element MSE 的渐近收敛率：
  - Laplace（原生）: R_L(G) = Θ((log G)² G^(-δ))，小 G 下 MSE 低
  - Normal（旋转后）: R_N(G) = Θ(√(log G) G^(-δ²))，大 G 下 MSE 低
  由于 0 < δ² < δ < 1，存在 crossover 现象：小 G 时 Laplace MSE 更低（NVFP4 G=16 不应旋转），大 G 时 Normal MSE 更低（MXFP4 G=32 应旋转）。这直接指导了设计决策：MR-GPTQ-MXFP4 必须旋转，MR-GPTQ-NVFP4 可选旋转（若配合 scale 优化可补偿旋转引入的局部误差）。

  **(2) MSE-Optimized Grids + Static Act-Order + Block-wise Rotations（解决缺陷 4）**
  - **MSE-Optimized Grids**：替代标准 absmax scale + RTN grid。对每个 tensor 求解 min_{s_T, s_{G_1...G_k}} Σ_i ||X̂_i - X_i||²，通过交替优化 per-tensor scale s_T 和 per-group scales s_G 最小化量化 MSE。NVFP4 无旋转时此优化产生一致改善；MXFP4 旋转后使用统一静态值。
  - **Static Activation Reordering**：标准 GPTQ 的 dynamic act-order 在推理时需实时重排列，产生 10-20% 延迟开销。MR-GPTQ 改为：先确定 scales/grid → 再按 Hessian 重排列 → 应用 GPTQ 量化 → 恢复原始列序。与 dynamic 效果相同，零推理开销。
  - **Block-wise Hadamard Rotations**：对 MXFP4（G=32），旋转将 Laplace 分布转为 Normal，降低 per-element MSE（与大 G 一致）。旋转大小匹配 group size，形成 "micro-rotation" 设计（k=32 for MXFP4, k=16 for NVFP4），区别于 QuaRot 的全局旋转。

  **(3) QuTLASS Fused Kernel 实现零推理开销（将理论加速兑现为实际加速）**
  - 权重端旋转离线预融合：W_rot = W·H_k，量化存储为 Q(W_rot)，无运行时旋转开销
  - 激活端 fused online rotation：QuTLASS kernel 将 H_k 加载（k<256 时 memory-bound，任意矩阵同成本）+ 旋转 + 量化 + scale 计算融合为单 kernel，epilogue 直接输出 FP4 量化值
  - MXFP4 在 B200 上 matmul throughput **超过** NVFP4 ~15%（power-of-two scales 降低硬件乘法开销）

  论文方法全栈执行例子（Llama-3.1-8B MXFP4 MR-GPTQ W4A4）：
  - 算法pipeline：加载 FP16 权重 → 离线阶段：对每个线性层（Q/K/V/O/gate/up/down）→ ① block-wise Hadamard 旋转 W_rot = W·H_32（k=32 匹配 MXFP4 G=32）→ ② FineWeb 1024 校准集前向计算 Hessian H=2X^T X → ③ 按原始列序计算 MSE-optimized scales & grid（MXFP4 使用统一静态 s_T）→ ④ Static act-order：按 Hessian 对角线重排列 → GPTQ 逐列量化 + 误差补偿（OBS 框架，各列共享 H^{-1}）→ 恢复原始列序 → ⑤ 存储 MXFP4 packed 权重（4.25 bits/elem）。推理时：FP16 激活 X → QuTLASS fused kernel: X_rot=X·H_32 → MXFP4 quantize(X_rot) → scale rearrangement（Triton kernel for tcgen05.mma）→ FP4 matmul（Blackwell hardware）→ 输出。结果：Average Recovery 93.31%（RTN=87.83%, GPTQ=89.47%），接近 NVFP4 水平。
  - 系统框架：PyTorch 模拟量化（精度实验）/ vLLM + QuTLASS kernel（性能实验）。量化代码：FP-Quant（https://github.com/IST-DASLab/FP-Quant）。
  - 编译框架：论文未明确说明。
  - kernel调度：QuTLASS v1.0（https://github.com/IST-DASLab/qutlass）。B200 单层 speedup 3.6×（ideal 4×），端到端 2.2×（vLLM Llama-3.3-70B）。RTX 5090 单层 6×（ideal 8×），端到端 4×。
  - 硬件架构：NVIDIA B200（SM100）/ RTX 5090（SM120）Blackwell GPU。利用 tcgen05.mma 硬件 FP4 矩阵乘指令。

  关键设计动机映射：
  - MXFP4 E8M0 scale 误差大 → MSE-optimized grids 交替优化 s_T 和 s_G 最小化整体 MSE + scale fitting (×4/3 unbiased estimate)
  - MXFP4 大 G=32 下 Normal 分布 MSE 更低 → block-wise Hadamard 旋转（G=32 匹配旋转 block size 32）
  - NVFP4 小 G=16 下 Hadamard 旋转有害 → 分析指导 NVFP4 无旋转 + MSE grid 优化（利用 NVFP4 E4M3 scale 精度优势）
  - Standard GPTQ dynamic act-order 有推理开销 → Static act-order：先定 grid → 重排量化 → 恢复原序，零开销
  - 在线旋转可能抵消 FP4 硬件加速收益 → QuTLASS fused kernel：k<256 时旋转 memory-bound，任意矩阵同成本，epilogue 直接量化无中间写入

## BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

- baseline方法是什么？
  Baseline使用基础XNOR-Net风格权重二值化：w^bi = σ * sign(w)，其中σ = ||w||/n初始化为逐通道可学习浮点标量（Rastegari et al., 2016）。激活使用LSQ逐层量化器（Esser et al., 2019）。训练损失为标准简化变分下界 L_simple = E_t,x_0,ε[||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t) ε, t)||²]。全栈执行例子：全精度DDIM/LDM权重 → 逐通道sign(w)二值化为{-1,+1} → σ可学习缩放因子调整幅度 → LSQ逐层激活量化为低bit整数 → 前向卷积 o = σ * (a ⊗ sign(w))（⊗仅含整数加法，无乘法） → L_simple损失反向传播 → STE近似sign梯度 ∂L/∂w ≈ ∂L/∂w^bi * 1_{|w|≤1} → 迭代优化。
  Baseline存在两个核心缺陷：(1) **表征能力坍塌**：权重从2^32候选值骤降到2^1，信息熵急剧下降，对生成模型关键表征造成灾难性损害；(2) **优化方向模糊**：离散化sign函数引入显著前向参数误差和反向梯度近似误差，QAT中的精细特征学习受扰，收敛不稳定甚至不可达。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BinaryDM通过EBB（表征角度）和LRM（优化角度）两个技术对应解决baseline两大缺陷：

  **(1) EBB应对表征能力坍塌**：训练第一阶段使用双基二值化 w_EBB^bi = σ_I*sign(w) + σ_II*sign(w - σ_I*sign(w))，通过残差结构将权重候选值从2个扩展到更多组合，显著提升信息熵和表征空间。正则化损失 L_EBB = τ/N * Σ σ_II 驱动高阶基σ_II→0，平滑演化到第二阶段的单基全二值化 w^bi = σ_I*sign(w)。仅应用于首尾各6层（约15%参数），中间层保持vanilla binarizer，减少过渡不稳定性。这使二值化DM从更高信息容量的初始状态开始优化，避免了表征骤然坍塌。

  **(2) LRM应对优化方向模糊**：对全精度DM中间表征通过PCA计算协方差矩阵 C_i = (hw)⁻² * ε̂^FP * (ε̂^FP)^T，特征分解后取前⌈c/K⌉列特征向量E_i作为低秩投影矩阵（K默认为4），将全精度和二值化DM的中间表征同时投影到低秩空间：R_i^FP = ε̂^FP * E_i^(⌈c/K⌉)，R_i^bi = ε̂^bi * E_i^(⌈c/K⌉)。MSE损失 ||R_i^FP - R_i^bi|| 在低秩空间中驱动二值化DM沿主成分方向学习全精度表征，避免高维空间直接对齐导致的优化方向模糊。投影矩阵在首batch计算后固定不变，保证优化方向稳定性。

  全栈执行例子：预训练全精度DDIM/LDM → EBB双基二值化初始化(σ_I, σ_II) → 第一阶段多基卷积 o = σ_I*(a⊗sign(w)) + σ_II*(a⊗sign(w - σ_I*sign(w))) → LRM在每组timestep embedding模块后计算低秩投影对齐：PCA(ε̂_θi^FP) → R^FP, R^bi → MSE loss → 总损失 L_total = L_simple + τ*Σσ_II/N + λ*Σ||R^FP - R^bi||/M → 第二阶段σ_II→0后转换为单基 w^bi = σ_I*sign(w) → W1A4推理（4-bit激活分解为4个1-bit激活+偏置） → 15.2×OPs节省、29.2×存储节省 → Qualcomm Snapdragon 855 Plus实测4.62×加速。

## Basis Sharing Cross-Layer Parameter Sharing for Large Language Model Compression

- baseline方法是什么？
  Baseline 是 **SVD-LLM**（Wang et al., 2024b），SVD-based per-layer weight compression 的 SOTA 方法。SVD-LLM 流程：(1) 对每层的每个权重矩阵独立处理；(2) 引入 whitening matrix 捕获激活中的 outlier 信息来调整权重矩阵，即 S(S^T) = cholesky(X^T X)，用 S 缩放权重后做 SVD；(3) 截断小奇异值实现压缩；(4) 压缩比 k 由目标压缩率决定。

  Baseline 的核心缺陷：**仅对单层内权重矩阵做独立压缩，完全忽略了跨层权重之间的相似性**。LLaMA/LLaMA2 等 decoder-only transformer 的不同层中，同类型权重矩阵（如 W_K, W_Q, W_V）可能具有相似的参数分布，独立 SVD 无法利用这种跨层冗余实现进一步压缩。在相同压缩比下，跨层共享可降低总体的 Frobenius loss（实验证实在 W_K 上 9-10 层共享后 loss 从 66682.9 降至 61817.3）。

  Baseline（SVD-LLM）全栈执行例子（LLaMA-7B, 20% 压缩）：
  - 算法pipeline：加载 FP16 LLaMA-7B → 逐层逐矩阵：评估 S（256 WikiText-2 样本, FP64）→ SVD(S·W) → 截断 k 个奇异值 → Ŵ = S^{-1}U_kΣ_kV_k^T → 推理时 Ŵ @ X。w/o basis sharing: PPL=7.94（WikiText-2）。
  - 系统框架：HuggingFace Transformers + PyTorch，两块 A100 GPU。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（标准 PyTorch FP16/BF16 GEMM）。
  - 硬件架构：论文未明确说明（NVIDIA A100，无自定义硬件）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Basis Sharing**，核心设计：将跨层同类型权重矩阵水平拼接为一个合并矩阵，对合并矩阵做一次性 SVD，分解为**共享基向量 B''**（所有层共用）+ **每层独有系数矩阵 C^(i)**。通过三个关键设计解决基线缺陷：

  **(1) 跨层拼接+SVD共享基向量（解决跨层冗余未被利用的问题）**
  SVD-LLM 对单层 W 分解得到 W ≈ U_kΣ_kV_k^T，U_kΣ_k（基矩阵）和 V_k^T（系数矩阵）都只为单层服务。Basis Sharing 将 n 层拼接为 W_cat ∈ R^{d1 × n·d2}，SVD 后得到共享基矩阵 B'' = S^{-1}U_k'Σ_k' 和系数 C（前 d2 列属第 1 层，后 d2 列属第 2 层，...）。共享基向量意味着所有层共享相同的"参数原型"，不同层通过不同的系数组合来表达各自的权重功能，区别仅在于系数。这实现了比 independent SVD 更低的 Frobenius loss（共享后 loss 可能小于独立压缩之和），从而在相同压缩比下获得更好的 model quality。

  **(2) 矩阵类型筛选（避免在无关矩阵上做有害共享）**
  并非所有矩阵类型都适合跨层共享。论文通过 Frobenius loss 热力图分析发现：W_K, W_Q, W_V, W_Up, W_Gate 共享后 Frobenius loss ≤ 独立 SVD 之和（对角块外颜色 ≤ 对角块），适合共享；W_Down（高维→低维投影，拼接后 rank 增大导致截断损失更大）和 W_O 共享后 loss 反而增大，不适合共享。这个设计避免了在错误的矩阵类型上强制共享导致的性能退化。

  **(3) 相邻层分组策略（最小化 group 内 Frobenius loss）**
  层分组不是任意的：相邻层共享基矩阵产生的 Frobenius loss 最小，因为相邻层在 transformer 中通常处理相似特征层次的特征。默认按 2 层一组顺序分组（1-2, 3-4, ...）。消融实验验证：2 层分组在 ≥30% 压缩比下优化，4-5 层在 ≤30% 下较优；LoRA 微调后更多层（甚至 32 层全共享）也在可接受范围内。

  论文方法全栈执行例子（LLaMA-7B, Basis Sharing, 20% 压缩, 2 层分组）：
  - 算法pipeline：加载 FP16 LLaMA-7B → 逐类型矩阵（W_K/Q/V/Up/Gate 共享；W_Down/W_O 独立 SVD-LLM）：① 垂直拼接相邻 2 层输入 X → ② 计算 S = cholesky(X^T X)^{1/2} → ③ 水平拼接 2 层权重 W_cat → ④ SVD(S·W_cat) → ⑤ 截断 k = (d1·d2·2·0.8)/(d1+2·d2) → ⑥ B'' = S^{-1}U_kΣ_k（共享基）, C = V_k^T（每层各 d2 列系数）→ 推理：Y_i = X_i·B''·C^(i)。WikiText-2 PPL=7.74（SVD-LLM=7.94）。50% 压缩比下 PPL=19.99（SVD-LLM=23.97, ↓17%）。
  - 系统框架：HuggingFace Transformers + PyTorch，两块 A100 GPU。压缩时间：GPT2 仅需 26.47s（Dynamic Tying 需 13.75h 训练）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（压缩后仍为 FP16 矩阵乘法：先 X_i·B'' 再乘 C^(i)，两次小矩阵乘代替一次大矩阵乘）。
  - 硬件架构：A100 GPU。50% 压缩比下 throughput=1.57× dense 模型（batch=512, seq=32）。

  关键设计动机映射：
  - SVD-LLM 不利用跨层相似性 → Basis Sharing 拼接多层的同类型矩阵，SVD 分解后共享基向量
  - 盲目共享所有类型矩阵会导致退化 → Frobenius loss 热力图筛选适合共享的矩阵类型（W_K/Q/V/Up/Gate vs W_Down/W_O）
  - 任意层分组可能导致高 loss → 相邻层分组策略最小化 Frobenius loss
  - 高压缩比（≥40%）下后续层输入偏差累积 → 更新后续层输入以补偿偏差（与 SVD-LLM 相同的补偿策略）

## Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

- baseline方法是什么？
  Baseline 是 QLoRA (Dettmers et al., 2023)，即 LoRA-finetuning quantization 标准范式：(1) PTQ 阶段使用 NormalFloat (NF) quantization 将 LLM 权重量化到 k-bit；(2) 在量化后的 LLM 上额外附加 LoRA 低秩适配器（rank r=64）进行参数高效微调。量化过程使用对称量化，scale factor s = absmax(w)，无 calibration constant（零点为 0）。同时比较的 baseline 还包括 QA-LoRA（integer 量化 + 量化感知 LoRA）、QLoRA w/ GPTQ（GPTQ 量化）和 PEQA（无 LoRA 的量化感知微调）。

  Baseline 全栈执行例子（QLoRA, 4-bit LLaMA-7B, Alpaca 微调, MMLU 评估）：
  - 算法pipeline：加载 FP16 LLaMA-7B 预训练权重 → 按 block_size=64 分块 → NormalFloat 4-bit 量化 ŵ = NF4(w/absmax(w)) → double quant scale s₁^FP8, s₂^FP16 → 附加 LoRA（r=64, α=16）适配所有 linear 层 → 在 Alpaca 52K 数据上 AdamW 微调 10000 steps → MMLU 5-shot 评估。此 baseline 在信息层面存在两个缺陷：(a) NF 量化采用零点固定为零的对称量化，导致量化权重信息熵最大化受限、与原始权重互信息不足；(b) LoRA 的两个低秩矩阵 ℓ₁, ℓ₂ 仅做矩阵乘法变换，变换形式同质化，且 ℓ₂ 只能使用 ℓ₁ 的中间表示而无法直接利用原始输入 x。
  - 系统框架：基于 HuggingFace Transformers + PEFT 库（LoRA 实现）→ 使用 QLoRA 官方代码库 → PyTorch 训练。
  - 编译框架：论文未明确说明（标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明（使用标准 PyTorch FP16/BF16 矩阵乘法 kernel）。
  - 硬件架构：NVIDIA Tesla A100 GPU，无自定义硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 IR-QLoRA，从统一的信息视角出发解决两个信息丢失问题：

  **(1) ICQ (Information Calibration Quantization) 解决量化信息丢失问题**：
  Baseline QLoRA 中对称量化 ŵ=NFk(w/s) 的零点固定为零，量化权重熵 H(ŵ) 不能最大化，导致与原始权重的互信息不足。ICQ 引入 calibration constant τ，将量化变为 ŵ=NFk((w-τ)/s)，并通过最大化量化权重的信息熵 H(ŵ) = -ΣP(q_i)log₂P(q_i) 来搜索最优 τ*。搜索以 median(w) 为初始值（符合正态分布对称性假设），在 [τ₀-0.1σ, τ₀+0.1σ] 区间内均匀采样 200 个候选，选最大熵对应的 τ*。ICQ 将 4-bit LLaMA-7B 的权重熵从 3.67 提升到 3.74，无需 LoRA 微调即可使 MMLU 提升 0.5%。

  **(2) IEC (Information Elastic Connection) 解决 LoRA 表征能力不足问题**：
  Baseline LoRA 的 ℓ₂ 矩阵只能使用 ℓ₁ 的低秩变换结果，无法访问原始输入 x。IEC 通过两个 parameter-free 操作解决：(a) U₁ 中对输入 x 按 (r/h) 比例分组平均后加到 ℓ₁ 输出，使 ℓ₁ 能融合原始输入信息；(b) U₂ 中对中间表示 x' 重复拼接 (o/r) 次后加到 ℓ₂ 输出，使 ℓ₂ 能直接利用多样化表示。IEC 仅引入 2 个 per-layer learnable scalars (β₁, β₂)，且在推理时可通过矩阵数学合并消除额外开销。

  论文方法全栈执行例子（IR-QLoRA, 4-bit LLaMA-7B, Alpaca, MMLU）：
  - 算法pipeline：加载 FP16 LLaMA-7B → **ICQ**: 按 block_size=64 对每块权重 search τ* → ŵ=NF4((w-τ*)/absmax(w-τ*)) → double quant τ* 和 s → **IEC 微调**: 在 Alpaca 上训练 LoRA+β₁,β₂ 10000 steps → 推理时 IEC 合并入 LoRA → MMLU 5-shot 评估得 40.8%（vs QLoRA 38.4%，提升 1.4%）。
  - 系统框架：基于 QLoRA 官方代码修改 → HuggingFace Transformers + PEFT → PyTorch 训练。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。
  - 硬件架构：NVIDIA Tesla A100 GPU。ICQ 搜索仅增加 0.46%（7B）/ 0.31%（13B）训练时间，IEC 无额外训练时间。存储方面 ICQ 增加 2.04% 参数（7B：2.34GB→2.39GB），IEC 仅增加 2 个 per-layer 标量。

  关键设计动机映射：
  - Baseline 对称量化零点固定 → ICQ 引入可搜索 calibration constant τ，通过熵最大化释放量化器的信息保留灵活性。
  - Baseline LoRA ℓ₂ 无法访问原始输入 → IEC U₁ 的分组平均连接使 ℓ₁ 输出融合原始信息。
  - Baseline LoRA 变换形式同质 → IEC U₂ 的重复拼接引入参数无关的多样化变换。

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

## 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution

- baseline方法是什么？
  Baseline 是 DBDC+Pac（CVPR 2023），当时 SR 领域的 SOTA PTQ 方法。DBDC 阶段需手动指定 clipping ratio 来截断长尾分布数据，Pac 阶段以极低学习率对量化器参数进行微调（导致收敛缓慢）。该方法主要针对 CNN-based 模型（EDSR、SRResNet）设计，无法有效适配 Transformer-based SR 模型（如 SwinIR），因为 SwinIR 的权重和激活分布呈现"对称+非对称共存+长尾"的独特特征。使用对称量化会浪费至少一半候选值；长尾效应会导致绝大多数浮点数被压缩到 1-2 个候选值中。
  
  Baseline 全栈执行例子（SwinIR ×2 4-bit）：
  - 算法pipeline：DBDC 对每个张量用统一的手动指定 clipping ratio 截断 → 对权重用对称量化、激活用不对称量化（未区分分布类型）→ Pac 阶段以极低 lr 微调 → 量化后 Linear/BMM 转为 INT4 算术。
  - 系统框架：论文未明确说明。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 2DQuant，一个 coarse-to-fine 的两阶段 PTQ 方法，针对 Transformer-based SR 模型的"对称+非对称共存+长尾"分布特征进行专门优化：
  (1) **DOBI 替代 DBDC**：不再需要手动指定 clipping ratio，而是基于张量分布类型自动选择搜索策略——对称钟形分布用双界同时收缩搜索，非对称指数分布用固定下界+仅收缩上界搜索，以 MSE 最小化为目标自动找到最优 (l, u)。
  (2) **DQC 替代 Pac**：使用知识蒸馏方式（教师=FP 模型，学生=量化模型），联合优化输出 L1 loss 和中间层特征归一化 L2 loss，学习率从极低值提升至 1e-2，大幅加速收敛并避免局部最优。
  (3) **两阶段组合**：DOBI 的 search-based 方法天然避免局部最优，为 DQC 提供良好初始化；DQC 的 optimization-based 方法进一步向任务目标 fine-tune，形成互补。

  论文方法全栈执行例子（SwinIR ×2 4-bit）：
  - 算法pipeline：加载预训练 FP32 SwinIR-light 权重 → Stage1 DOBI：对每个 Linear 层权重（对称钟形分布，双界同时搜索 K=100 步）和每个激活张量（Attention Map/FC2 输入为非对称指数分布固定下界仅收上界；V/FC1 输入为对称钟形双界同搜）执行 MSE 最小化搜索，得到粗粒度 (l_i, u_i) → Stage2 DQC：用 DF2K 校准集，以输出 L1 loss + 特征归一化 L2 loss 联合蒸馏，Adam lr=1e-2，CosineAnnealing 3000 iter，batch=32，STE 近似梯度反向传播 fine-tune 每个量化器的 (l_i, u_i) → 量化后权重存为 INT4，推理时 Linear/BMM 转为 INT 算术，speedup 3.99×。
  - 系统框架：论文未明确说明。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。
  - 硬件架构：论文未明确说明。

## AFPQ Asymmetric Floating Point Quantization for LLMs

- baseline方法是什么？
  基线方法是 **对称 FP 量化**（FP-sym），对每个 weight group 使用单一 scale 进行缩放。具体而言：`scale = max(w_max, |w_min|) / (range/2)`，所有正值和负值共享同一个 scale，即 FP 候选值的覆盖范围关于零对称。此方法无法适配 LLM 权重 tensor 中普遍存在的非对称分布——论文在 LLaMA2-7B 上随机抽样 group-size=128 的 weight groups，发现超过 50% 的组呈现最大最小值不关于零对称的特征。这导致：scale 由绝对值较大的一侧决定，另一侧的表达范围被浪费，部分 FP 候选值落在原始权重范围之外，量化精度下降。此问题在 group size 较小和 sub-4-bit 时尤为严重。另外，也尝试了仿照 INT-asym 的"一个 scale + 一个 zero_point"方法直接套用到 FP 量化，但这会使 FP 的密集表示区域从零偏移，丧失 FP 格式的核心优势。

  Baseline 全栈执行例子（LLaMA2-7B FP4-sym RTN group-size=128）：
  - 算法pipeline：weight group 内计算 `scale = max(w_max, |w_min|) / 7` → `w_4bit = round(w / scale)` → `w_deq = scale * w_4bit` → FP16 激活 × 反量化 FP16 权重 → Layer output。高级方法 GPTQ/AWQ 中使用 INT 量化（INT-asym），即 `scale = (w_max - w_min) / 15`、`zero_point = -w_min / scale`、`w_4bit = round(w / scale) + zero_point`。
  - 系统框架：基于 AutoGPTQ（https://github.com/PanQiWei/AutoGPTQ）执行量化。GPTQ 使用二阶 Hessian 信息逐列补偿量化误差，AWQ 在量化前对 salient channels 的权重乘以 per-channel scaling factor。
  - 编译框架：论文未明确说明。
  - kernel调度：FasterTransformer 中的 INT4/FP16 混合精度 GEMM kernel，INT4 权重通过 scale+zero_point 反量化到 FP16 后与 FP16 激活做矩阵乘。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AFPQ（Asymmetric Floating Point Quantization）**，核心设计：为 weight group 内的正值和负值分别设置独立 scale——`scale_pos = w_max / (range/2)` 用于正值，`scale_neg = -w_min / (range/2)` 用于负值。这解决了三个关键问题：
  (1) **适配非对称分布**：正负两侧各自缩放，FP 值的覆盖范围与原始权重范围精确匹配，不浪费表达空间。
  (2) **保留 FP 优势**：与"scale + zero_point"方法不同，AFPQ 不移动 zero point，FP 格式在零附近密集分布的优势得以完整保留，因为在 LLM 权重中大部分值集中在零附近。
  (3) **无额外存储开销**：每组存储 scale_pos 和 scale_neg 两个参数，与 INT-asym 存储 scale 和 zero_point 两个参数的开销完全相同。
  AFPQ 还作为即插即用的底层量化格式，无缝替换 GPTQ 和 AWQ 中的 INT 量化步骤，保持高层算法的二阶补偿/显著性缩放逻辑不变。

  论文方法全栈执行例子（LLaMA2-70B NF3-asym GPTQ group-size=128）：
  - 算法pipeline：加载 FP16 LLaMA2-70B 权重 → GPTQ 框架以 group_size=128 分组 → 对每组计算 `scale_pos = w_max / 3.5`、`scale_neg = -w_min / 3.5`（NF3 range/2 = 3.5）→ `w_3bit_pos = round(w_pos / scale_pos)`、`w_3bit_neg = round(w_neg / scale_neg)` → GPTQ 的二阶 Hessian 补偿：OBS 式逐列更新未量化权重以补偿当前列量化误差 → 最终每组存储 scale_pos、scale_neg 和 packed 3-bit NF3 权重 → 推理时 GPU kernel：LUT 将 NF3 索引映射为 FP16 → `w_deq = scale_pos * w_nf3_pos + scale_neg * w_nf3_neg` → FP16 GEMM。结果：WikiText-2 ppl 从 GPTQ-INT3 的 3.77 降至 3.66，MMLU 从 67.25% 升至 68.05%。
  - 系统框架：基于 AutoGPTQ 进行量化，在 FasterTransformer 中部署 NF3-asym dequantization kernel 用于推理。量化时使用 AutoGPTQ 的 GPTQ/AWQ 实现，仅修改底层 Quant/Dequant 函数。
  - 编译框架：论文未明确说明。
  - kernel调度：在 FasterTransformer 中自定义 NF-asym dequantization kernel——packed byte 解包 → LUT NF→FP16 映射 → 按正负通道分别乘 scale_pos/scale_neg → FP16 GEMM。在 A6000 GPU 上，NF4-asym LLaMA2-13B 推理延迟 485.42ms（FP16 baseline 788.01ms，1.62x speedup）。
  - 硬件架构：论文未明确说明。

## AWQ Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

- baseline方法是什么？
  Baseline 是 **GPTQ**（Frantar et al., 2022），当时 LLM weight-only 后训练量化的 SOTA 方法。GPTQ 流程：(1) 将权重量化问题建模为逐列（column-by-column）的二阶误差补偿；(2) 使用 Hessian 矩阵的逆来更新未量化权重，补偿已量化列引入的误差；(3) 通过对校准集做 block-wise reconstruction 降低量化误差。GPTQ 的核心缺陷：(i) **校准集过拟合**：reconstruction 过程对校准集分布敏感，当校准集（如 PubMed）与评估集（如 Enron）分布不同时，perplexity 恶化 2.3-4.9；(ii) **需要大量校准数据**：需要 192+ 条序列才能达到好的量化效果；(iii) **需要 trick**：对 LLaMA-7B 和 OPT-66B 需要 reordering trick 才能正常工作；(iv) calibration set 过拟合会扭曲预训练学到的一般性特征，影响 LLM 在 OOD 领域和多模态任务上的泛化能力。

  Baseline（GPTQ）全栈执行例子（LLaMA-7B INT3-g128）：
  - 算法pipeline：加载 FP16 权重 → 逐层对权重矩阵做 block-wise reconstruction：校准集前向传播缓存 layer input → 求 Hessian H=2XX^T → 计算 H^{-1} → 逐列量化：量化第 i 列 → 用 H^{-1} 更新剩余列以补偿第 i 列误差（OBS 算法）→ 重复至所有列量化完毕 → 输出 INT3 量化权重。对部分模型需 reordering（按 Hessian 对角线降序排列列，量化后恢复原序）。
  - 系统框架：AutoGPTQ（https://github.com/PanQiWei/AutoGPTQ）/ GPTQ-for-LLaMA，基于 PyTorch + HuggingFace Transformers。校准集 128-192 条 sequences from C4/WikiText。
  - 编译框架：使用 Triton 编写 INT4 reordered 量化 kernel（GPTQ-for-LLaMA）。
  - kernel调度：Triton kernel：INT4 反量化（通过 scale+zero_point 或对称量化 scale）→ 与 FP16 激活执行 GEMM/GEMV。对于 reordered 量化需额外的索引重排。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AWQ（Activation-aware Weight Quantization）**，通过以下核心设计解决 GPTQ 的缺陷：

  **(1) 激活感知的显著权重识别替代 Hessian 二阶误差补偿**：GPTQ 使用 Hessian 指导误差补偿，依赖校准集分布导致过拟合。AWQ 发现只需识别 0.1%-1% 的显著（salient）权重通道即可大幅降低量化误差，且识别方式非常简洁——看**激活分布**（per-channel 平均激活幅度）而非权重分布。这避免了 GPTQ 的逐列 reconstruction 过程，不需要反向传播或回归，从根本上消除了过拟合问题。

  **(2) Per-channel scaling 替代混合精度**：直接保留显著权重为 FP16 虽然有效但会引入混合精度，硬件实现低效。AWQ 通过数学推导（Eq. 2-3）证明：对显著权重通道乘以 s > 1，并将对应的激活除以 s（等效变换），可以降低显著权重的相对量化误差（误差比例 `Δ'/Δ · 1/s < 1`）。scale s 通过单超参 α 的网格搜索（`s = s_X^α`）自动确定，仅需 20 步搜索即可找到平衡显著与非显著通道误差的最优 α。

  **(3) 极简校准集需求**：AWQ 仅需从校准集计算 per-channel 平均激活幅度 `mean(|X[c]|)`，而非做复杂的 reconstruction。因此 AWQ 仅需 16 条序列（GPTQ 需要 192 条，节省 10×），且对校准集分布不敏感——当校准集和评估集分布不同时 perplexity 仅恶化 0.5-0.6（GPTQ 恶化 2.3-4.9）。

  **(4) 首次实现多模态 LLM 低比特量化**：由于 AWQ 不过拟合校准集，可直接应用于 OpenFlamingo-9B、LLaVA-13B、VILA-7B/13B 等视觉语言模型（仅量化语言部分），为领域首次。INT4-g128 下 COCO Captioning 32-shot CIDEr 仅下降 1.17（RTN 下降 4.57，GPTQ 下降 6.72）。

  **(5) TinyChat 推理系统将理论压缩转化为实际加速**：针对 W4A16 量化中存储精度（INT4）与计算精度（FP16）不一致的挑战，TinyChat 设计 on-the-fly dequantization kernel（反量化与 GEMM/GEMV 融合在寄存器完成）、SIMD-aware weight packing（ARM NEON 上 32 个 4-bit 权重仅需 3 条 SIMD 指令解包）、kernel fusion（LayerNorm/Attention/QKV 投影融合），在 4090/Orin/4070 上实现 3.2-3.3× 加速比。

  论文方法全栈执行例子（LLaMA-7B INT4-g128）：
  - 算法pipeline：加载 FP16 权重 → 16 条 Pile 校准集前向传播收集 per-channel 激活幅度 `s_X = mean(|X|)` → 网格搜索 α ∈ [0,1]（20 步），每步：`s = s_X^α → W_scaled = W·diag(s) → INT4-g128 group-wise 量化 W_scaled → 用 `diag(s)^{-1}·X` 评估输出 MSE → 选最优 α → 最终量化 W 并融合 `diag(s)^{-1}` 入前一层。无需 Hessian、无需 reconstruction、无需反向传播。量化后 PPL 5.60（FP16 5.47，GPTQ 5.69）。
  - 系统框架（TinyChat）：PyTorch 前端 + CUDA/PTX 后端 → 加载 AWQ INT4-g128 量化权重 → 推理时：LayerNorm（fused kernel）→ QKV projection（fused, on-the-fly dequantization + RoPE on-the-fly）→ Attention（fused, KV cache 更新在 kernel 内完成）→ Output projection（dequantization GEMV）→ MLP（gate/up/down，fused dequantization）→ 残差连接 → 生成 token。RTX 4090 上从 HF FP16 52 tokens/s 加速至 TinyChat W4A16 ~194 tokens/s（3.7×）。
  - 编译框架：论文未明确说明（使用 PyTorch eager mode + 自定义 CUDA/PTX kernel，未修改 compiler framework）。
  - kernel调度：TinyChat CUDA kernel——INT4 packed 权重（每 2 个 4-bit 占 1 byte）从 DRAM 读取 → 寄存器内 shift + AND 解包 → 乘以 group-wise Δ（FP16）→ 乘以 per-channel s → 与 FP16 激活做 FMA → 输出存回寄存器供下一操作使用。SIMD-aware packing 在 ARM NEON 上提供额外 1.2× 加速。Kernel fusion 将每次推理的 kernel launch 从数十次减少到数次（每个 Transformer Block 约 3-4 次）。
  - 硬件架构：TinyChat 在 NVIDIA Jetson Orin（15W，8GB，移动 GPU）上部署 Llama-2-70B（awq量化后），并在 Raspberry Pi 4B 上部署 Llama-7B（0.7 tokens/s）。

## ARB-LLM Alternating Refined Binarizations for Large Language Models

- baseline方法是什么？
  Baseline 是 **BiLLM**（ICML 2024），SOTA 二进制 PTQ 方法。BiLLM 的流程：(1) 用 Hessian 敏感度选出 salient columns，剩余的为 non-salient columns；(2) 对 salient weights 使用二阶 binarization（Ŵ = α₁B₁ + α₂B₂ + μ），non-salient weights 按 magnitude 分为 sparse/concentrated 两组，分别用一阶 binarization（Ŵ = αB + μ）；(3) 所有 binarization 参数（μ, α, B）通过标准的闭式解一次性确定（μ = mean(W), α = mean(|W-μ|), B = sign(W-μ)），不做迭代精炼。BiLLM 的缺陷：(i) 二值化参数一次性计算后不做精炼，导致二值化权重与全精度权重存在分布偏移（均值不对齐，见图 2）；(ii) calibration data 仅用于 Hessian 敏感度评估和 block-wise error compensation，未参与 binarization 参数的更新；(iii) 仅使用 row-wise scaling（α, μ），无法处理 LLM 权重中显著的列间偏差（某些列值远大于其他列，见图 3）；(iv) 仅对 non-salient weights 做 sparse/concentrated 分组，salient columns 的 group bitmap 区域未被利用（见图 5 左侧）。

  Baseline 全栈执行例子（LLaMA-7B, BiLLM, ~1.09-bit）：
  - 算法pipeline：FP16 权重 W → 逐层计算 Hessian 选出 salient columns → salient weights: 二阶 binarization Ŵ=α₁B₁+α₂B₂+μ，α₁,α₂,μ,B₁,B₂ 一次性闭式解 → non-salient weights: 分 sparse/concentrated 两组，分别一阶 binarization Ŵ=αB+μ，α,μ,B 一次性闭式解 → block-wise OBC 补偿 → 压缩权重存储（bitmap 记录分区，B 存 ±1，α,μ 存 FP16）。WikiText2 ppl: 49.79。
  - 系统框架：PyTorch + HuggingFace，单卡 A800-80GB，耗时 45 min。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（使用标准 PyTorch 推理，W 存储为 packed 1-bit + FP16 scaling factors，推理时解包并反量化后 FP16 GEMM）。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ARB-LLM 通过四个递进式创新解决 BiLLM 的缺陷：

  **(1) ARB（Alternating Refined Binarization）**：引入迭代精炼机制，每轮交替更新 μ → α → B 以逐步缩小量化误差。解决了 BiLLM 一次性计算导致的分布偏移问题。理论保证（Theorem 1）：每轮迭代后 L₁^τ ≤ L₁⁰，量化误差单调不增。仅 ARB 基础版本（无 CGB）就将 WikiText2 ppl 从 49.79 降至 22.67。

  **(2) ARB-X**：将 calibration data X 引入 binarization 参数更新，用 L₂ = ||WX - ŴX||² 替代 L₁ = ||W - Ŵ||² 作为优化目标。通过预计算 S = Σ X_b^T X_b 将高维 calibration tensor 压缩为 S ∈ R^{m×m}，理论加速 389×（Theorem 2）。解决 BiLLM 中 calibration data 仅用于 Hessian 评估而未参与参数优化的问题。WikiText2 ppl: 21.81。

  **(3) ARB-RC**：引入 column-wise scaling factor α^c，与 row-wise α^r 形成双轴缩放（Ŵ = α^r·α^c·B），同时移除 μ 以节省存储。解决 BiLLM 仅用 row-wise scaling 无法保留列间偏差的问题（图 3 右验证 ARB-RC 有效保留列偏差）。ARB-RC 在性能和压缩上双赢：WikiText2 ppl 14.03（ARB-X 的 21.81），同时存储从 2.93GB 降至 2.63GB（LLaMA-7B raw bitmap）。

  **(4) CGB（Column-Group Bitmap）**：将 salient columns 也按 magnitude 分为 sparse/concentrated groups，使 group bitmap 的 salient 区域不再浪费（G_s = 1_n C_s^T ⊙ G, G_ns = 1_n C_ns^T ⊙ G）。解决 BiLLM 中 group bitmap 在 salient columns 区域未被利用的空间浪费。CGB 进一步提升性能（ARB-RC + CGB: 14.03 vs ARB-RC w/o CGB: 15.85）。

  论文方法全栈执行例子（LLaMA-7B, ARB-LLM_RC = ARB-RC + CGB, ~1.09-bit, #Iter=15）：
  - 算法pipeline：FP16 权重 W → 逐层：① Hessian 评估选出 salient columns C_s，C_ns = ¬C_s → ② CGB 分区（salient-sparse / salient-concentrated / non-salient-sparse / non-salient-concentrated）→ ③ salient zones: 二阶 ARB-RC，每轮交替更新 α^r→α^c→B₁,B₂（15 轮，式 13 更新 α^r,α^c，式 8 binary search 更新 B₁,B₂），无 μ → ④ non-salient zones: 一阶 ARB-RC，每轮交替更新 α^r→α^c→B（15 轮）→ ⑤ block-wise OBC 补偿 → 输出：Ŵ = α^r·α^c·B（±α^r_i·α^c_j 加权二值矩阵）+ bitmap。WikiText2 ppl: 14.03，LLaMA-7B QA 平均准确率首次在 binary PTQ 中超越同尺寸 FP16 模型。
  - 系统框架：PyTorch + HuggingFace → 单卡 A800-80GB → 128 calib samples from C4 → binarization 耗时 76 min（#Iter=15, CGB）。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。推理时权重以 1-bit packed format 存储，bitmap + α^r/α^c scaling factors 用于反量化。ARB-RC 比 BiLLM 少存储 μ，存储效率更优（LLaMA-7B: 2.83GB raw → 2.09GB CSR vs BiLLM: 2.93GB raw → 2.19GB CSR）。
  - 硬件架构：论文未明确说明。

## AffineQuant Affine Transformation Quantization for Large Language Models

- baseline方法是什么？
  Baseline 是 **OmniQuant**（Shao et al., 2023），当时 LLM PTQ 中等价变换的 SOTA 方法。OmniQuant 的核心流程：(1) 逐 transformer block 优化，引入 learnable scale（对角矩阵）和 learnable shift（平移向量）两种等价变换；(2) 优化目标为 block 输出在量化前后的 MSE；(3) scale 和 shift 通过梯度下降联合优化，使用 Hessian-guided 学习率；(4) 变换合并入相邻层以保证推理无额外开销。OmniQuant 还比较了 AWQ（仅缩放变换，per-channel scale 由激活统计量确定）、SmoothQuant（手动设计的 scale，将激活量化难度迁移到权重）、RPTQ（per-cluster 重排等价于置换矩阵变换）。

  OmniQuant 等方法的**核心缺陷**是：优化空间仅限于对角线缩放（scale）和平移（shift），即权重矩阵 W 的每个 output channel 只能被统一缩放和平移，不能改变 channel 内部各维度的相对关系。这导致在低比特或小模型场景下，量化的固定点（2ⁿ-1 个量化级别）与权重分布不匹配，大量信息因无法重分布而丢失，量化误差显著增大。直观上如图 1 所示：scaling 仅能做统一的线性拉伸/压缩，translation 仅能做整体平移，两者都无法将二维向量的各维度分别对齐到量化固定点；而 affine 变换则可以实现任意维度的重新分布。

  Baseline（OmniQuant）全栈执行例子（LLaMA2-7B w4a4 量化）：
  - 算法pipeline：加载 FP16 LLaMA2-7B → 逐 transformer block：初始化 scale s=1（对角矩阵）、shift δ=0 → 校准集前向传播缓存输入 → block 输出 MSE loss 计算 → 梯度下降更新 s 和 δ → 量化权重 Q(sW) → 合并 scale 入 LayerNorm → 逐 block 重复至全模型量化完毕。C4 PPL=18.02，WikiText2 PPL=14.26。
  - 系统框架：PyTorch + HuggingFace Transformers，单卡 Nvidia A100 GPU。校准集：WikiText2 训练集 128 segments × 2048 tokens。
  - 编译框架：论文未明确说明（使用标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明（量化后权重以 INT4 格式存储，推理时通过标准 dequantize→FP16 GEMM 执行，使用 PyTorch 原生算子）。
  - 硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AffineQuant**，用一个完整的**仿射变换矩阵 A**（不再是 restricted 的对角矩阵）替代原有等价变换方法中受限的 scale 向量，极大扩展优化空间。具体设计包括三个方面：

  **(1) 仿射变换矩阵替代对角缩放：扩大优化空间。** 
  OmniQuant 的优化空间是 d 维对角 scale + d 维 shift（共 2d 自由参数），而 AffineQuant 的优化空间是 d×d 维矩阵 A + d 维 shift（共 d²+d 自由参数）。这意味着权重 W 的每一行（output channel）可以在行空间内实现任意线性重组——本质上是对每个 output channel 执行旋转+缩放，使权重向量更好地对齐到量化的 2ⁿ-1 个固定点上。图 1 直观展示了差异：scaling 仅能统一缩放（不能改变方向），translation 仅能平移（不能改变各维度的相对位置），而 affine 变换可以任意旋转和缩放各通道以贴合固定点网格。这直接解决了 baseline 在低比特下优化空间不足的根本问题。

  **(2) Gradual Mask 保证仿射矩阵可逆：解决高维矩阵不稳定问题。**
  仿射变换要求计算 A⁻¹，但 d×d 矩阵（d 可达 4096 以上）在有限校准数据下的自由优化极易退化为奇异矩阵（不可逆）。论文基于 Levy-Desplanques 定理（严格对角占优矩阵必可逆）提出 Gradual Mask（GM）方法：将 A 初始化为对角矩阵（严格对角占优平凡满足），在训练早期冻结所有非对角线元素为零，随着 epoch 推进逐步释放靠近对角线的元素参与优化。具体地，在第 e 个 epoch，只允许 |i-j| ≤ (e/t)·hidden_size 的非对角线元素更新，且更新幅度由稳定性因子 α（<1）抑制。GM 在前向通过 Hadamard 积缩小非对角线元素幅度保证 A* 可逆，在反向作为学习率调节器抑制非对角线参数更新速率。OTA-125M w3a16 无 GM 时 WikiText2 PPL 达 53.52（vs 有 GM 30.17），LLaMA-7B w2a16 无 GM 直接 NaN（训练崩溃）。

  **(3) 仿射变换与平移正交互补。**
  平移变换 b 是全局的：v → v + b。仿射变换 A 是旋转+缩放：v → Av。两者数学正交，可以同时施加而不互相干扰：v → Av + b。AffineQuant 同时学习 A 和 δ（shift），在 transformer block 级别优化 argmin_{A,δ} ||f_i(X,W) - f_i((X-δ)A⁻¹, Q(AW), b+δW)||²。

  论文方法全栈执行例子（LLaMA2-7B w4a4 量化）：
  - 算法pipeline：加载 FP16 LLaMA2-7B → 逐 transformer block：① 对每个线性层初始化 A 为对角矩阵（对角线=SmoothQuant scale）、δ=0；② 每 epoch：计算 GM（从中心对角线逐步向外释放），A* = A∘GM，A_inv = inv(A*)，X_t = (X-δ)A⁻¹，W_t = Q(A*W)，block 前向计算 MSE loss，梯度下降更新 A 和 δ（GM 抑制非对角线更新）；③ 多 epoch 后合并：W_final = Q(AW)，bias_final = b+δW，对角 A（LayerNorm 后）合并入 LN。C4 PPL=15.76（OmniQuant=18.02，↓2.26），WikiText2 PPL=12.69（OmniQuant=14.26，↓1.57）。LLaMA-30B w4a4 6-task zero-shot avg=58.61%（OmniQuant=56.63%，↑1.98%）。
  - 系统框架：基于 OmniQuant 代码修改 → PyTorch + HuggingFace Transformers → 单卡 Nvidia A100 GPU → 矩阵求逆使用 PyTorch linalg.inv（float/double 精度）。校准集：WikiText2 128 segments × 2048 tokens。优化参数（lr、epoch、clipping）对齐 OmniQuant。
  - 编译框架：论文未明确说明（使用标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明。推理时将 A 和 δ 合并入权重和 LayerNorm 参数，最终仅使用标准 INT4 group-wise 量化权重和 FP16 激活进行 GEMM 推理，无额外 kernel 需求。
  - 硬件架构：论文未明确说明。

  关键设计动机映射：
  - OmniQuant 仅对角 scale 优化空间有限（d 自由参数）→ AffineQuant 用完整仿射矩阵 A（d² 自由参数）极大扩展优化空间
  - 高维矩阵优化不稳定（易奇异）→ Gradual Mask + Levy-Desplanques 定理确保矩阵始终保持严格对角占优
  - Low-bit quantization 下固定点与权重分布不匹配 → 仿射变换旋转权重通道以重新分布，使所有维度都贴合量化固定点

## GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration

- baseline方法是什么？
  Baseline 是 **GPTQ**（Frantar et al., 2022），基于 Optimal Brain Compression (OBC) 框架的经典 PTQ 方法。GPTQ 每层独立执行**对称校准（Symmetric Calibration）**：最小化 `||(w+Δw)X − wX||²`，其中 X 来自前一层的量化输出。由于前层量化误差已改变激活分布，X 与全精度模型的输入 X̃ 存在偏差（即 ΔX = X̃ − X ≠ 0），且该偏差沿网络深度累积（Fig. 2a 显示激活 MAE loss 逐 block 持续增长）。GPTQ 忽略了这一偏差，仅优化当前层的局部 MSE。

  GPTQ 全栈执行例子（LLaMA2-7B, W4A4, 单卡 A100）：
  - **算法pipeline**：校准数据 X（128 sequences × 2048 tokens from WikiText2）→ 逐层：计算 Hessian H = XX^T → Cholesky 分解 H^{-1}=LL^T → 按 block size B 逐列量化：Q_{:,j} ← quant(W_{:,j}) → 误差补偿 δW = −(W_{:,j}−Q_{:,j})/L_{jj} · L_{j,j:}^T（仅二阶项）→ lazy-batch 更新后续列。激活量化在权重量化后应用（QuaRot 风格），目标为 `||ŵX − wX||²`，不对齐全精度输出。
  - **系统框架**：HuggingFace Transformers + PyTorch，自实现 GPTQ 脚本，无 Serving 框架修改。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel（matmul、cholesky），无自定义 kernel。推理使用标准 INT4 dequantize + FP16 GEMM。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **对称校准忽略前层误差累积**：GPTQ 每层假设输入激活 X 已经正确（即等价于全精度前向的 X̃），但前层量化后的实际 X 与 X̃ 存在偏差 ΔX，该偏差沿网络深度累积，导致深层量化严重偏离全精度模型行为。
  2. **校准目标是"局部最优"而非"全局最优"**：最小化 `||ŵX − wX||²` 确保当前层输出与传统前向一致，但传统前向的输入 X 已经是"错的"（受前层量化误差影响），导致最终模型输出与全精度模型偏差大。
  3. **在低比特场景下偏差急剧放大**：W2A4 时 GPTQ 对 LLaMA 模型仍退化严重，RTN 直接崩溃（PPL > 1000）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **GPTAQ**，将校准目标从对称的 `||ŵX − wX||²` 改为非对称的 `||ŵX − wX̃||²`（X̃ 为全精度模型的输入激活），并推导出高效的闭式解实现：

  **(1) 非对称校准框架（解决缺陷 1, 2）**
  对称校准时优化目标中的 target 是 wX，即"量化层在已有误差输入下的输出"。非对称校准时 target 改为 wX̃，即"全精度层在全精度输入下的输出"。引入残差 r = wX̃ − wX = wΔX（输入激活偏差在输出空间的投影），通过 Lagrangian 求导得最优权重更新：
  ```
  Δw = −(ŵ_q − w_q)/H_{qq}^{-1} · H_{q,:}^{-1} + r X^T H_{-q}^{-1}
  ```
  第一项是 GPTQ 的量化误差补偿，第二项是 GPTAQ 新增的**残差误差补偿项**，显式将前层累积误差通过 Hessian 逆回传到当前层权重。消融实验（Table 5）验证：仅用第一项 = GPTQ（WikiText2 7.80），仅用第二项名曰 GPTAQ'（WikiText2 7.97），两项联合 = GPTAQ（WikiText2 7.36，最优）。

  **(2) 残差分解避免重复计算（解决效率瓶颈）**
  直接计算残差 r 需每次迭代重新评估 R = W X̃ − W X，复杂度 O(mnk) 极高（k ≈ 128×2048 >> n）。GPTAQ 利用 R = Σ_{q=1}^n W_{:,q} ΔX_{q,:} 将残差分解为 n 个独立神经元分量。预计算一次 ΔX 后，第 q 次迭代仅关注第 q 个神经元残差分量 `W_{:,q} ΔX_{q,:} X_{:,q:}^T H_{-q}^{-1}`，复杂度从 O(mnk) 降至 O(mn)，无需重复计算 R。

  **(3) Cholesky 重构化 + 矩阵融合实现 GPU 并行（解决数值稳定性和效率）**
  通过 **Lemma 4.1**：Cholesky 因子 L 的子矩阵 L_{q+1:,q+1:} 即为消去前 q 行的逆 Hessian H_{-q}^{-1}，替代数值不稳定的逐次 Gaussian Elimination。通过 **Theorem 4.2** 将 P 矩阵（存储每行残差补偿项）计算融合为一行 GPU 友好代码：
  ```
  P = ((ΔX X^T L) ⊙ M_U) L^T
  ```
  其中 M_U 是严格上三角掩码矩阵。利用 CUDA 高度并行，计算 P 仅需 <1ms（vs 非并行实现 >10⁴× slower，Fig. 4a）。

  **(4) Lazy-Batch 更新（解决 GPU 利用率）**
  与 GPTQ 的 lazy-batch 策略一致：block 内逐列更新，block 后批量更新 block 外列。GPTAQ 的 block 外更新为：
  ```
  ΔW_{:,Q:} = E · L_{Q,Q:}^T + W_{:,Q} · P_{Q,Q:}
  ```
  两项融合在同一 kernel 中。GPTAQ 整体额外延迟：n<4096 时 <10%，n>4096 时 30-40%（Fig. 4b）。

  论文方法全栈执行例子（LLaMA2-7B, W4A4, 单卡 A100）：
  - **算法pipeline**：校准数据 128 sequences → 先全精度前向收集每层 X̃（FP 输入），同时记录量化后输入 X → 计算 ΔX X^T 和 H → 启用激活量化（A→W 顺序，Table 6 证明此顺序对 GPTAQ 更优）→ 逐 block：对每层 ① Cholesky 分解 L ← H + λI → ② P ← ((ΔX X^T L) ⊙ M_U) L^T → ③ 按 block 逐列量化 + GPTQ 项补偿 + GPTAQ 残留项补偿。整模型流程（Algorithm 2）：每次仅一个 transformer block 在 GPU，X̃ 临时存储可释放，内存瓶颈为 P 矩阵（n×n FP16），LLaMA2-7B 每层 P 约占 0.16-0.70GB。
  - **系统框架**：HuggingFace Transformers + PyTorch，基于 GPTQ 代码修改（仅多约 20 行）。单卡 A100 量化 LLaMA3.1-405B（126 blocks, intermediate=8096）。W4A4 LLaMA3-8B 耗时 0.3 GPU-hours（GPTQ 0.2h），LLaMA3-70B 耗时 2.7 GPU-hours（GPTQ 1.8h）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel（Cholesky、matmul、element-wise mask）。无自定义 Triton/CUDA kernel。推理使用标准 INT4 dequantize + FP16 GEMM。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  关键设计动机映射：
  - GPTQ 对称校准假设 X = X̃，忽略前层误差累积 → GPTAQ 非对称校准目标 `min ||ŵX − wX̃||²`，显式补偿 r X^T H_{-q}^{-1}
  - 直接非对称优化需每次迭代重算 R（O(mnk)）→ 残差分解 R = Σ W_{:,q} ΔX_{q,:}，复杂度降至 O(mn)
  - Gaussian Elimination 数值不稳定 → Cholesky 重构化（L_{q+1:,q+1:} 等价于 H_{-q}^{-1}）+ Theorem 4.2 矩阵融合
  - W2A4 下 GPTQ/RTN 严重退化 → GPTAQ 在 W2A4 LLaMA 上将 GPTQ PPL 降低 20%-90%
  - 实现仅比 GPTQ 多 ~20 行代码，易部署和复现


## AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

- baseline方法是什么？
  Baseline 是 **Any-Precision LLM**（Park et al., 2024），当时多精度 LLM 量化的 SOTA 方法。其核心设计：(1) 采用聚类-based 非均匀量化，将权重矩阵按 K-means 聚类为若干 centroid，每个权重存储 centroid index；(2) 通过 Incremental Upscaling 实现多精度：从低精度开始，逐步分裂聚类中心（如 4 个 centroid → 8 个），使单个模型覆盖多种精度；(3) 推理时将权重以比特平面形式存储，按需加载 p 个比特平面，经比特转置（bit-transpose）重组为索引，再通过 centroid table lookup 获取反量化值后执行 GEMM。

  Baseline 全栈执行例子（Any-Precision LLM, 3-bit Llama-3.1-8B 推理）：
  - 算法pipeline：FP16 权重 → K-means 聚类为 2^3=8 个 centroid → 存储 centroid table + 3 个比特平面 → 3-bit 推理时加载 3 个比特平面 → bit-transpose 重组为 8 值索引 → table lookup 获取 FP16 centroid → GEMM。与 FP16 baseline 相比，任何精度下推理均可通过加载更少比特实现内存带宽节省。
  - 系统框架：PyTorch + 自定义 CUDA kernel，GPU 推理。
  - 编译框架：论文未明确说明。
  - kernel调度：自定义 CUDA kernel 实现 bit-transpose + centroid table lookup + GEMM。主要开销：bit-transpose（占 kernel 延迟 35-58%）和 centroid table lookup（占 9-17%）。
  - 硬件架构：论文未明确说明（标准 GPU 执行）。

  **Any-Precision LLM 的核心缺陷：**
  (a) **硬件不友好**：依赖 centroid table lookup，无法直接对二进制比特平面操作（bitwise operations 不适用于非均匀量化的 index 语义）。bit-transpose 和 table lookup 引入额外开销，限制了实际加速比。
  (b) **极低比特退化严重**：2-bit 时准确率急剧下降（MMLU=24.66 vs FP16=65.02，Wiki PPL=1680.77 vs FP16=6.24），有效可用范围仅限于 3-4 bit。
  (c) **非均匀量化的表达能力在低比特下无法发挥**：2-bit 仅 4 个 centroid，K-means 聚类无法充分捕获权重分布。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AnyBCQ**，基于 Binary-Coded Quantization (BCQ) 的多精度量化框架，通过三个层次的设计解决 baseline 缺陷：

  **(1) BCQ 替代非均匀量化：实现硬件友好的比特平面直接操作（解决缺陷 a）**
  BCQ 将权重表示为 Ŵ = Σ α_i B_i（B_i ∈ {-1,+1}），而非 centroid index。每个比特平面 B_i 天然是二值操作数，可以直接与激活值进行加减运算（+α_i 或 -α_i），无需 centroid table lookup 和 bit-transpose。推理时仅加载所需 p 个比特平面，每个平面独立计算后按 α_i 缩放累加。这使 kernel 可以直接执行：`output = Σ_{i=1}^p α_i · (B_i ⊗ activation)`，其中 ⊗ 表示基于 LUT 的二值-浮点 GEMM。

  **(2) 渐进式精度扩展（Progressive Precision Expansion）：实现单调精度改善（解决缺陷 b）**
  从基础精度 p_L（如 2-bit）开始，逐比特扩展至 p_H（如 4-bit）。每次扩展时：(i) 冻结之前所有比特平面 B_1...B_{p-1}（共享二值表示），(ii) 从残差 R = W - Ŵ^{(p-1)} 中提取新比特平面 B_p = sign(R)，(iii) 通过最小二乘重新优化所有 α_i（而非重新优化 B_i）。这保证了 p-bit 模型的精度 ≥ (p-1)-bit 模型的精度（单调改善），因为新增比特平面总是捕获残差信息。配合 block-wise MRE（最小化重建误差）校准，2-bit 时 MMLU=35.32（baseline=24.66），Wiki PPL=19.01（baseline=1680.77）。

  **(3) 共享二值表示 + 独立 scale：内存高效的多精度存储（解决缺陷 c 的工程影响）**
  所有精度共享同一组比特平面 B_1...B_{p_H}，仅维护精度特定的缩放因子 {α_i^{(p)}}。因为比特平面占总存储的绝大部分（如 4-bit BCQ: 3.89GB binary vs 0.49GB scale），共享二值表示使多精度模型仅需 4.99GB（vs Multi-model 独立存储三个模型的 9.85GB，↓49%）。

  论文方法全栈执行例子（AnyBCQ, 3-bit Llama-3.1-8B 推理）：
  - 算法pipeline：FP16 权重 W → 基础精度 p_L=2：GREEDY(W) → T=20 交替优化（LS + BS）→ 精度扩展 p=3：冻结 B_1,B_2，初始化 B_3=0, α_3=0 → 从残差 R=W-(α_1B_1+α_2B_2) 提取 B_3=sign(R) → T=20 LS 优化 α → Block-wise MRE 校准 10 epochs → 最终模型：1 组比特平面 {B_1,B_2,B_3} + 3 套 scale {α_i^{(2)}}, {α_i^{(3)}}（2-bit 和 3-bit 各一套）。3-bit 推理：Ŵ = α_1^{(3)}B_1 + α_2^{(3)}B_2 + α_3^{(3)}B_3。
  - 系统框架：PyTorch + HuggingFace Transformers → lm-eval-harness v0.4.5 评估。校准集：C4 512 sequences。
  - 编译框架：论文未明确说明。
  - kernel调度：自定义 CUDA kernel：① 加载 p 个比特平面（packed binary）→ ② LUT-based GEMM（每个比特平面独立计算加减结果）→ ③ 乘 α_i 累加 → ④ 输出。消除 bit-transpose 和 centroid lookup 开销。GEMV 延迟：M=4096, K=14336, 2-bit: 315µs（×2.78 vs cuBLAS, Any-Precision LLM=356µs）。端到端吞吐：2-bit=245 tok/s vs Any-Precision LLM=228 tok/s vs FP16=105 tok/s。
  - 硬件架构：CUDA 12.6 + NVIDIA A100 80GB。论文讨论中提到 AnyBCQ 可部署到 BCQ 原生加速器（iFPU、FIGLUT）获得更大加速。

  关键设计动机映射：
  - Any-Precision LLM 非均匀量化无法直接操作比特平面（需 centroid lookup + bit-transpose）→ AnyBCQ 用 BCQ 二值表示，使计算简化为 {-1,+1} 的加减操作
  - 2-bit 精度退化严重（K-means 仅 4 centroid 不足以表达权重分布）→ MRE-based BCQ 在 2-bit 时已有 2 个比特平面（4 种组合值），且逐比特贪心优化更有效利用有限表达空间
  - 多精度模型存储冗余（多套独立模型）→ 共享比特平面 + 独立 scale 减少 49% 内存
  - Bit-transpose 和 LUT lookup 占 kernel 延迟 44-75% → AnyBCQ kernel 直接比特平面操作消除两项开销，换算为 1.07-1.17× 端到端吞吐提升

## BiLLM Pushing the Limit of Post-Training Quantization for LLMs

- baseline方法是什么？
  Baseline 是现有的 LLM PTQ 方法，包括 **GPTQ**（基于 Hessian 的二阶误差补偿 block-wise 量化，在 4-bit 表现良好但在 ≤2-bit 崩溃）、**PB-LLM**（部分二值化，保留 10% 权重为 INT8 其余二值化，平均 1.7-bit）、**RTN**（直接 round-to-nearest 量化，≤2-bit 完全崩溃）。

  Baseline 全栈执行例子（LLaMA-7B, GPTQ-2bit, block size=128）：
  - 算法pipeline：对每个 Linear 层的权重矩阵 W ∈ R^{n×m}，按 block size=128 逐列进行 2-bit 均匀量化（4 个量化级别），通过 Hessian 矩阵 H=2XX^T 进行 block-wise 误差补偿，即量化当前 block 后将误差 E = (W_q - W)/H^c 乘以 H^c 的对应子矩阵补偿到后续 block。PB-LLM 则先按 Hessian 选择 top-10% salient 元素保留 INT8，其余二值化，最终平均 1.7-bit。
  - 系统框架：PyTorch + HuggingFace Transformers 加载预训练模型，在单卡 NVIDIA A100 80GB 上完成 PTQ 过程。Calibration data 为 C4 的 128×2048 tokens。
  - 编译框架：论文未明确说明。模型量化后以 PyTorch 自定义 Linear 层加载，推理时在 Python 层进行反量化。
  - kernel调度：论文未明确说明。量化权重存储为 packed integer 格式，推理时反量化为 FP16 计算。
  - 硬件架构：NVIDIA A100 80GB GPU（Ampere 架构），无自定义硬件加速器。

  **Baseline 的核心缺陷：**
  1. GPTQ 在 ≤2-bit 时，均匀量化只有 4 个离散值（2-bit），无法表达 LLM 权重的钟形分布和少数 salient 权重的极端值，导致性能崩溃（LLaMA-7B 2-bit PPL=152.31 vs FP16=5.68）。
  2. PB-LLM 虽然保留 10% salient 权重为 INT8，但（a）非结构化选择需要 1-bit bitmap index 导致额外存储开销；（b）简单二值化 salients 的量化误差仍然很大；（c）未处理非 salient 权重的非均匀钟形分布，二值化误差随分布非均匀性增加。
  3. Vanilla RTN 在 ≤2-bit 完全崩溃（如 OPT-6.7B 2-bit PPL=28363.14），因无任何误差补偿机制。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BiLLM 通过三项创新设计解决上述缺陷：

  **(1) 结构化 Salient 列选择 + 二进制残差逼近**（对应解决 PB-LLM 缺陷 a, b）：
  观察到 LLM 的 Hessian salient 权重在特定列中聚集（尤其在 Q/K/V 的 attention 投影层和 Out Projection 层），因此采用**按列结构化选择**而非逐元素非结构化选择。通过搜索最优 salient 列数 n* 最小化整体二值化重构误差。对选中的 salient 列，不保留 INT8（浪费位宽），也不简单二值化（误差大），而是使用**二进制残差逼近**：先二值化得到 B_o，再对残差 (W - α_o·B_o) 进行第二次二值化得到 B_r，最终用 α_o·B_o + α_r·B_r 两个二值矩阵之和逼近原始 salient 权重。这相当于用 2-bit 表达 salient 权重（vs PB-LLM 的 8-bit），且可证明残差逼近的量化误差 ε_rb ≤ 直接二值化的 ε_direct。

  **(2) 钟形分布最优分裂二值化**（对应解决 baseline 缺陷 c）：
  观察到非 salient 权重呈钟形分布（类似高斯/拉普拉斯），二值化作为极端均匀量化在此分布上误差极大。BiLLM 搜索一个最优分裂点 p*，将分布分为稀疏区（|w|>p，远离 0 的值）和集中区（|w|≤p，聚集在 0 附近），分别以独立的 scaling factor α_s 和 α_c 进行二值化。这相当于在钟形分布上用两个分段常数逼近，显著降低二值化 MSQE。搜索策略使用百分位搜索（步长 0.1），目标 min_p θ²_q,p = ||W_s - α_s·B_s||² + ||W_c - α_c·B_c||²。尽管实际分布偏离理想高斯，搜索曲线仍呈凸性，保证可找到最优 p*。

  **(3) Block-wise OBC 误差补偿**（继承自 GPTQ，Block size=128）：
  移除 column-wise 补偿以提升 PTQ 效率，仅保留 block-wise 补偿，确保分布探索不受干扰。

  **BiLLM 全栈执行例子（LLaMA-7B, ~1.09-bit）：**
  - 算法pipeline：加载 LLaMA-7B FP16 权重 → 提取 C4 calibration data（128×2048 tokens）→ 计算每层 Hessian H=2XX^T → Cholesky 稳定求逆 → 对每个 Linear 层逐 block（128 列）处理：① 计算 S = W²/H² 逐元素显著矩阵 → 按列聚合显著性 → 搜索最优 salient 列数（3-30 列）→ ② salient 列：residual binarization (B_o + B_r, 2-bit) → ③ 非 salient 列：搜索最优 p* (百分位 0.1-0.9) → split binarization (1-bit + 1-bit flag) → ④ 合并 + OBC block-wise 补偿 → 存储为 packed binary + scaling factors + bitmap。推理时：加载 packed binary → 按列解包 scaling factors → 需要时反量化 → FP16 GEMM，但论文主要关注 memory footprint 而非推理加速。
  - 系统框架：PyTorch + HuggingFace Transformers。自定义 quantized Linear 层替代原始 FP16 Linear。量化后模型以自定义 format 存储（packed binary + scaling factors + group/sparse-concentrated bitmap）。推理时可通过 custom kernel 实现 memory-efficient 推理，论文未提供完整推理 engine。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。量化权重以 1/2-bit packed binary 格式存储，推理时需反量化回 FP16 进行 GEMM。论文指出 "binarized GEMM is hard to implement directly due to fine-grained grouping"，主要收益在于 GPU memory footprint 降低（而非 GEMM 加速）。
  - 硬件架构：NVIDIA A100 80GB GPU。无自定义硬件加速器。

  关键设计动机映射：
  - GPTQ/PB-LLM 的非结构化 salient 选择浪费 bitmap 存储 + INT8 保留位宽过高 → BiLLM 的结构化列选择（利用 attention 层中 sensitivity 的列聚集特性）+ 残差二值化（2-bit 替代 8-bit 表达 salient 权重）
  - 钟形分布下直接二值化 MSQE 极大（因权重非均匀分布）→ optimal splitting 将分布分为 concentrate/sparse 两区独立二值化，用分段常数逼近降低误差
  - 简单二值化误差过高导致 ≤2-bit 崩溃 → 残差逼近可证明降低误差（ε_rb ≤ ε_direct）+ splitting 搜索凸性保证最优解
  - PB-LLM 平均 1.7-bit 仍过高（30%+ INT8 权重）→ BiLLM 将平均 bit-width 推至 1.07-1.13 bit（接近理论下限），且 PPL 更低

## Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

- baseline方法是什么？
  Baseline 是 **vanilla transformer**（Pythia 架构），即标准自回归 Transformer decoder，每层为全局因果 self-attention + FFN。其推理时存在两个核心瓶颈：
  
  (1) **Prefill 瓶颈**：生成第一个 token 前，必须先前向传播所有 prompt token，计算并缓存其 key-value（KV）状态。prompt 长时（如 2048 tokens），预填充延迟显著。
  
  (2) **Decode 瓶颈**：自回归生成阶段每步仅计算一个 token，但必须从 HBM 中检索所有先前 token 的 KV cache。KV cache 大小与序列长度 L 和 batch size B 线性增长（L×B），KV cache 内存访问总量随 L 二次增长（O(L²)）。在 batch decoding 场景下（实际部署常见），KV cache IO 成为主要吞吐瓶颈，远超参数 IO 开销。
  
  Baseline 全栈执行例子（vanilla Pythia 302M, L=2048, B=16, prefill-heavy scenario）：
  - 算法pipeline：prompt token ids → Embedding → 24 层 transformer decoder：每层 Masked MHA（QKV 投影 → QK^T/√d → causal mask → softmax → ×V → output proj）→ FFN（gate+up → SiLU → ×down）→ 残差连接。prefill 阶段：所有 2048 tokens 并行前向，缓存 24 层 × 2048 tokens × 2(KV) × 1024 dim × 16 heads × 2 bytes = 3.2GB KV cache。decode 阶段：每步生成 1 token，从 HBM 读取全部 KV cache 和模型参数（302M × 2 bytes = 604MB），受 HBM 带宽限制，batch size 受限，MFU 典型仅 ~1%。
  - 系统框架：HuggingFace Transformers + GPT-NeoX 库，PyTorch eager mode。
  - 编译框架：论文未明确说明（使用标准 PyTorch）。
  - kernel调度：论文未明确说明（标准 cuBLAS GEMM/GEMV kernel）。
  - 硬件架构：8× A100 40GB 训练，1× H100 推理。

  Baseline 的核心缺陷：**KV cache 内存和 IO 是推理吞吐的主要瓶颈**。随着模型规模增大和 context length 增长（百万 token 级别趋势），KV cache 开销愈发严重，超越了参数 IO，成为 batch inference 吞吐量的硬上限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**Block Transformer**，通过分层全局到局部（global-to-local）建模，将标准自注意力的两个功能（全局上下文理解 vs 局部细节建模）分解到两个独立 decoder 中：
  
  **(1) Block Decoder 解决预填充和全局 KV cache 开销问题**：
  Baseline 的每一层都对全部 L=2048 tokens 做全局 attention，KV cache 存储、预填充计算和 decode IO 均随 L 线性或二次增长。Block Decoder 在块级别（L/LB=512 blocks）而非 token 级别做全局自注意力，将上下文长度降至 1/LB。这使得：(i) 预填充计算量降低 LB 倍；(ii) KV cache 存储量降低 LB 倍；(iii) KV cache IO 降低 LB² 倍（因 IO ∝ 上下文长度²）；(iv) 每 LB 个 token 仅执行一次前向（而非每 token 一次），参数 IO 降低 LB 倍。
  
  **(2) Token Decoder 解决 decode 阶段 KV cache IO 瓶颈**：
  Baseline 中上层 decoder 的 KV cache 存储和 IO 随整个序列长度 L 二次增长。Token Decoder 将全局上下文压缩为单个 context embedding（由 Block Decoder 输出），仅对当前块内 LB 个 token 做局部注意力（跨所有 token decoder 层）。这使得：(i) KV cache 存储从 O(L) 降至 O(LB)，对 L=2048, LB=4 降低 R=L/LB=256 倍；(ii) 预填充可完全跳过（除最后一个块外），大幅降低首 token 延迟；(iii) KV cache IO 从 O(L²) 降至 O(L·LB)，即线性复杂度，从根本上解决了长上下文场景的 KV cache IO 瓶颈。prefix token 机制允许通过增加 prefix 长度扩展 token decoder 的"计算宽度"，在几乎不影响推理吞吐的前提下提升性能（因推理受 memory-bound 而非 compute-bound 限制）。
  
  **(3) Embedder 简化全局到局部信息传递**：
  Baseline 无此组件。Embedder 通过 lookup table 将每 LB 个 subword token 拼接为一个 block embedding，为 Block Decoder 提供粗粒度输入。简单 lookup 策略（vs 小型 RoBERTa encoder）既高效又不损失性能。
  
  **(4) 1:1 参数分配比 + prefix token 设计逆转了先前工作（MEGABYTE）的结论**：
  MEGABYTE 认为全局模块（block decoder）应占 6 倍于局部模块的参数，且局部模块可很小。Block Transformer 通过参数分配消融证明：(i) 1:1 的 block:token decoder 参数比在固定总参数约束下达到最优 perplexity（U-shaped trade-off）；(ii) 更大的 token decoder 可在稍有性能妥协下显著提升吞吐量（因局部 KV cache 极小）；(iii) prefix token 机制赋予 token decoder 进一步处理上下文的能力（类似 pause tokens），这在先前工作中被完全忽视。

  论文方法全栈执行例子（Block Transformer 302M, LB=4, prefix=2, L=2048, prefill-heavy scenario）：
  - 算法pipeline：prompt token ids → **Embedder**：lookup table 将每 4 token 的 D/4-dim embedding 拼接为 D-dim input block embedding（共 512 blocks）→ **Block Decoder（12层）**：对 512 blocks 做全局 causal self-attention → 输出 context embedding [B, D] → 投影为 2 个 prefix tokens [B, 2, D] → **Token Decoder（12层）**：每步将 prefix + 当前块 4 token embedding [B, 6, D] 输入 → 局部 causal attention（仅 6 token）→ FFN → classifier 输出下一个块 4 token 的 logits → 重复 L/LB=512 次生成完整序列。prefill 阶段：Block Decoder 预填充 512 个 block（vs baseline 2048 tokens，4× 降低），Token Decoder 仅预填充最后一个块。decode 阶段：Block Decoder 每 LB=4 token 仅执行一次前向（参数 IO 降低 4×），Token Decoder 虽每 token 执行一次前向但 KV cache 仅 6 token（vs baseline 2048），KV cache IO 降低 256×。总吞吐量：prefill-heavy (2048/128) 吞吐量 21.0K tok/s vs vanilla 0.8K tok/s（~26×），decode-heavy (128/2048) 吞吐量 44.1K tok/s vs vanilla 2.1K tok/s（~21×）。PPL: LAMBADA=29.5 vs vanilla=10.0, WikiText=27.7 vs vanilla=20.1, HellaSwag=31.13 vs vanilla=35.05。batch size 约 6× vanilla（KV cache 节省 105.0MB/sample vs 1140.0MB/sample）。
  - 系统框架：HuggingFace Transformers + GPT-NeoX 库 + DeepSpeed ZeRO，PyTorch mixed precision training。
  - 编译框架：论文未明确说明。
  - kernel调度：论文验证 FlashAttention 应用于 Block Decoder 的全局 attention 可进一步提升吞吐（最高 31%），但整体趋势不变（附录 I）。其他 kernel 论文未明确说明。
  - 硬件架构：8× A100 40GB 训练，1× H100 推理。Uptraining 策略：从预训练 vanilla transformer 分割层初始化 block/token decoder，仅需 10% 训练数据即可接近全量训练性能，提供了从现有模型迁移的低成本路径。

## Binarized Diffusion Model for Image Super-Resolution

- baseline方法是什么？
  Baseline 是扩散模型图像超分辨率 (SR) 的二值化，直接套用现有二值化方法（BNN、DoReFa、XNOR、IRNet、ReActNet、BBCU）到 SR3 的 UNet 架构上。BBCU 是 SOTA 二值化 SR 方法，但仅针对端到端 CNN 设计，不涉及扩散模型的多步迭代。BBCU 的核心结构是 basic binary conv unit (BBCU)，移除了 BN 层以适配低层视觉任务。这些 baseline 方法在扩散模型上直接应用时面临三个核心缺陷：

  **(1) 维度不匹配（Dimension Mismatch）**：UNet 中 encoder 逐层下采样（H×W 减半、C 翻倍）、decoder 逐层上采样，特征维度不断变化。现有二值化方法依赖 identity shortcut 传递 FP 信息以补偿二值化信息损失（因 1-bit 表达严重受限），但维度变化使 shortcut 无法使用，切断了 FP 信息在全网络的传播路径。

  **(2) 融合困难（Fusion Difficulty）**：UNet 的核心结构 skip connection 需将 encoder 特征与 decoder 特征融合。常用方法 concatenation 导致输出维度翻倍（与 ResBlock 输入维度不匹配），addition 则因两种特征值域差异巨大（Fig. 3d 显示 encoder 和 decoder 特征的激活范围可达数倍差距）导致小值域特征被"遮盖"而无法有效参与融合。

  **(3) 多步激活分布变化（Multi-step Activation Distribution Shift）**：扩散模型需多步迭代去噪（T=2000），不同 timestep 下激活分布差异显著（Fig. 4 可视化：相邻 timestep 分布相似，但间隔大时分布剧烈变化）。二值化模块（Sign 函数和 1-bit 权重）对激活分布极度敏感——分布变化加剧了二值化后的信息损失（Sign 函数放大零附近值的差异 [38]），静态的 bias 和 RPReLU 无法适配多步的极端分布变化。

  Baseline 全栈执行例子（BBCU + UNet, ×2 SR, 4.82M Params）：
  - 算法pipeline：加载 SR3 的卷积 UNet → 所有卷积替换为 BBCU（二值化权重+激活，无 BN）→ 下采样/上采样用 stride 卷积 / PixelShuffle（维度变化，无 identity shortcut）→ skip connection 用 concatenation + 1×1 二值卷积调整维度 → 单组 bias+RPReLU 处理所有 timestep → 50 步 DDIM 推理。Manga109 PSNR=31.99dB, LPIPS=0.0326（全精度 SR3 PSNR=35.11dB, LPIPS=0.0161）。Baseline（无任何改进）PSNR 仅 27.66dB, LPIPS=0.0780。
  - 系统框架：PyTorch，A100-80G GPU 训练，无自定义 Serving/调度框架。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（标准 PyTorch 推理，未实现 XNOR/bit-count 定制 kernel）。
  - 硬件架构：论文未明确说明（NVIDIA A100 GPU，无自定义硬件）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 BI-DiffSR，首个专门为扩散模型 SR 设计的二值化方法，从结构（structure）和激活（activation）两个维度分别解决 baseline 的三个核心缺陷：

  **(1) CP-Down/CP-Up 解决维度不匹配（对应缺陷 1）**：
  Baseline 使用 stride 卷积或 PixelShuffle 改变维度，导致 ResBlock 输入输出维度不一致，identity shortcut 无法使用。BI-DiffSR 将所有维度变化操作统一封装到 CP-Down/CP-Up 中：CP-Down 先用双分支卷积（维度不变，可加 shortcut）处理输入，再用 Pixel-UnShuffle 降低分辨率+增加通道数；CP-Up 先用双分支卷积处理，再用 Pixel-Shuffle+Concat 提升分辨率。这样 ResBlock 内部始终保持维度一致，identity shortcut 全程可用，FP 信息流不被切断。消融验证：+Identity 使 PSNR 从 27.66→29.29 dB (+1.63)，+CP-Down&Up 进一步提升至 31.08 dB (+1.79)。

  **(2) CS-Fusion 解决融合困难（对应缺陷 2）**：
  Baseline 用 concatenation（维度翻倍，不匹配）或 addition（值域差异大，小值被覆盖）。CS-Fusion 利用 channel shuffle 思想：将两个输入特征 x1, x2 按奇偶通道索引交叉重组为 x1^sh（x1 奇数通道 + x2 偶数通道）和 x2^sh（x1 偶数通道 + x2 奇数通道）。shuffle 后两个特征的值域接近（Fig. 6 可视化验证），再分别经过二值卷积后相加融合。此设计维度不变（每个 shuffle 输出仍为 H×W×C），可加 identity shortcut，且 channel shuffle 为零开销索引操作。消融验证：CS-Fusion PSNR=31.99 dB vs Concat=31.08 dB (+0.91), Split=29.67 dB (+2.32), Add=18.89 dB (完全失效)。

  **(3) TaR/TaA 解决多步激活分布变化（对应缺陷 3）**：
  Baseline 使用单组静态 bias + RPReLU 处理所有 timestep。TaR/TaA 受 MoE 启发，设置 K=5 对 (bias, RPReLU)，将总 timestep T=2000 均分为 5 组，每组 timestep 激活专属的一对参数。这等价于将多步过程分割为 5 个子区间，每个区间内激活分布变化较小，bias/RPReLU 只需适配区间内分布，降低了学习难度。相邻 timestep 分布相似性使得固定分组策略有效。每个 timestep 仅激活 1 对参数，推理无额外计算开销。消融验证：+TaR&TaA PSNR 从 31.99→32.66 dB (+0.67)，LPIPS 从 0.0261→0.0200；需要同时使用 TaR 和 TaA（单独使用反而降低性能：仅 TaR=29.27dB, 仅 TaA=29.13dB vs 无=31.99dB），说明输入/输出端激活调整需协同工作。

  论文方法全栈执行例子（BI-DiffSR, ×2 SR, 4.58M Params）：
  - 算法pipeline：加载卷积 UNet（4 层 E-B-D, C=64）→ **结构层面**：所有 ResBlock 用 BI-Conv block（TaR→Sign→XNOR-Conv→TaA→+shortcut, 维度不变）→ CP-Down（双分支卷积+PixUnShuffle）→ CP-Up（双分支卷积+PixShuffle+Concat）→ skip connection 用 CS-Fusion（channel shuffle + 双分支二值卷积 + addition）→ **激活层面**：每个 BI-Conv block 的 TaR/TaA 有 K=5 对 (bias, RPReLU)，推理时按 floor(5*t/2000) 选对应组 → 50 步 DDIM 推理。Manga109 PSNR=33.99dB（BBCU=31.99dB, SR3 FP=35.11dB），LPIPS=0.0172（BBCU=0.0326, SR3=0.0161），达到全精度模型 93.6% 的感知质量。
  - 系统框架：PyTorch + 自定义 BI-Conv/CP-Down/CP-Up/CS-Fusion 模块 → 2× A100-80G 训练 → 无修改 Serving 框架。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。理论加速基于 XNOR+bit-count 替代浮点 MAC（32× 内存节省，64× 计算节省），但未实现定制 CUDA kernel。
  - 硬件架构：论文未明确说明（NVIDIA A100 GPU 训练+推理）。

  关键设计动机映射：
  - UNet 维度变化切断 FP shortcut → CP-Down/CP-Up 将维度变化隔离到独立模块，确保 ResBlock 维度一致、shortcut 全程可用
  - Skip connection 中 encoder/decoder 特征值域差异大无法融合 → CS-Fusion 通过 channel shuffle 平衡值域，使二值卷积能有效融合
  - 扩散模型多步迭代中激活分布随 timestep 剧烈变化 → TaR/TaA 用分组 (bias, RPReLU) 将 timestep 分段，降低单组参数适配范围
  - 分开使用 TaR 或 TaA 反降低性能 → 输入端（TaR）和输出端（TaA）激活调整需协同作用，因二值化前后均需适配当前 timestep 分布

## DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

- baseline方法是什么？
  Baseline 是 TFMQ-DM (CVPR 2024)，结合基本 PTQ 量化（MinMax quantizer + BRECQ/Adaround）和跨时间步校准策略。具体流程：(1) 使用 Q-Diffusion 的均匀时间步采样策略生成校准集（N=5120，每步 n=256，20 步 DDIM）；(2) 逐层使用 MinMax 量化器确定 activation per-tensor scale 和 weight per-channel scale；(3) 使用 BRECQ 进行 block-wise 权重量化重建（Adaround 自适应舍入）；(4) TFMQ-DM 额外引入时间步特定的量化参数来适配跨时间步的激活分布变化。也对比了 SmoothQuant（将 LLM 的等效缩放直接迁移到扩散模型），效果极差（W4A8 FFHQ: FID=454.16 vs Baseline=36.08）。
  
  Baseline 全栈执行例子（LDM-4 FFHQ 256×256 W4A8, 20 步 DDIM 采样）：
  - 算法pipeline：加载 FP32 预训练 LDM-4 U-Net → 校准数据集采样（20 步 DDIM, 256 样本/步 = 5120 校准点）→ per-tensor 激活 scale s^X = (max|X|)/(2^{b-1}-1) → per-channel 权重 scale s^W → BRECQ block-wise 量化重建 → 量化推理：x_t 输入 → 所有 Linear/Conv 层权重为 INT4、激活为 INT8 → QKV/FFN 计算 → 输出 ε̂_θ(x_t,t) → DDIM 更新 x_{t-1} → 重复 T 步。SmoothQuant 直接套用时：τ = (max(|X_c|)^β/max(|W_c|)^{1-β})^(1/2)，因扩散模型中激活 >> 权重，τ 极大 → 权重量化范围被显著扩展 → 权重量化误差骤增（Weight Quant. Error: 0.0694 vs Baseline 0.0060）。
  - 系统框架：PyTorch + LDM（latent-diffusion, https://github.com/CompVis/stable-diffusion）。评估使用 guided-diffusion 的 ADM TensorFlow 评估器。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明（标准 PyTorch 量化推理，无自定义 kernel）。
  - 硬件架构：论文未明确说明（GPU 上执行 PyTorch 推理）。

  **Baseline 的核心缺陷：**
  1. **通道级异常值未处理**：扩散模型的 activation 存在显著的通道间方差，尤其 skip connection 层（ResBlock skip connection 的 activation 通道间方差远大于 Transformer 层，Fig. 5）。标准 per-tensor 量化中，一个 outlier 通道的极端值拉伸了整层量化范围，使非 outlier 通道的量化精度严重下降。
  2. **SmoothQuant 迁移失败**：SmoothQuant 的缩放因子 τ = (max|X_c|/max|W_c|)^{1/2} 基于最大幅值比，但扩散模型中激活 >> 权重，导致 τ 极大 → 权重量化范围扩展 → 权重量化误差放大（权重在每步都使用，误差累积严重）。Baseline + SmoothQuant 的 Weight Quant. Error 从 0.0060 飙升至 0.0694。
  3. **等效缩放仅重分布不消除异常值**：即使正确应用等效缩放，它只是将激活的量化难度转移到权重（或反之），无法根本消除极端层中的异常值（如 skip connection 中某些通道的激活值远大于其他通道）。
  4. **时间步均匀加权次优**：早期去噪步（大 t）的量化误差虽小，但因在迭代过程前期引入，其影响会在后续步中累积放大；后期步的量化误差大但对最终质量的影响并非线性对应。均匀加权忽略了这一不对称性。
  5. **小校准集下 PTS 因子选择不可靠**：直接最小化校准集上的量化 MSE 选择 δ 因子会过拟合，在未见数据上性能退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 DMQ，通过两个核心设计解决缺陷：

  **(1) Learned Equivalent Scaling (LES) 解决异常值重分布问题（对应缺陷 1-2）**
  SmoothQuant 手动计算 τ 失败的本质原因是忽略了扩散模型的特殊性（激活 >> 权重）。DMQ 改为**学习** τ：以 layer-wise block reconstruction MSE 为目标（L_i = ||X_i W - Q(X̂_i) Q(Ŵ)||²），通过梯度下降直接优化 τ。不再依赖最大幅值比的启发式，而是利用校准数据找到最小化整体量化误差的 τ。
  - 引入 **Adaptive Timestep Weighting**（对应缺陷 4）：损失权重 λ_t = (1-Λ_t/ΣΛ_{t'})^α，Λ_t 为时间步 t 的累积损失（指数移动平均，ξ=0.95）。低误差的早期步得到更高权重，高误差的后期步仍有足够优化信号。避免了均匀权重偏向后期高误差步的问题，也避免了固定线性/二次权重忽略各层差异的问题（Fig. 4 右：各层误差趋势随 t 变化不同）。
  - **零推理开销融合**：τ 融合入权重和激活 scale（τ^T ⊙ W 预计算，τ ⊙ s^X 预计算），推理时无需额外操作。

  **(2) Power-of-Two Scaling (PTS) 解决极端异常值根本消除问题（对应缺陷 3、5）**
  等效缩放本质是双向转移量化难度，不能消除异常值。PTS 直接对 activation 施加通道级 2^δ 缩放：
  - 数学形式：X̃ = clamp(⌊X / (2^δ ⊙ τ ⊙ s^X)⌉, l, u)，输出时 Y ≈ s^X s^W · Σ X̃ · (W̃ ≪ δ)
  - **Bit-shift 高效实现**（对应缺陷 3）：2 的幂次缩放等价于整数 bit-shift，在 kernel 加载权重后立即执行 Ŵ^{shifted} = Ŵ ≪ δ，不需要乘法。仅应用于 skip connection 层（高通道间方差的层），总体开销极小。
  - **Voting Algorithm**（对应缺陷 5）：对每个校准样本和通道评估候选 δ ∈ {0,...,D}，选择最优 δ*_{i,k}；对每个通道计算众数 δ_k^{mode} 和一致性 r_k；仅当 r_k > κ(=0.85) 时采用 δ_k^{mode}，否则 δ_k=0（不缩放）。这种保守策略避免了小校准集下的过拟合，仅对确有统计共识的通道应用 PTS。
  - **Selective application**（对应缺陷 5 延伸）：消融实验（Tab. 7）证实仅对 skip connection 层应用 PTS 优于全层应用（FID: 30.37 vs 31.91），因为只有 skip connection 层存在严重通道间方差。

  论文方法全栈执行例子（LDM-4 FFHQ 256×256 W4A8, 20 步 DDIM）：
  - 算法pipeline：FP32 LDM-4 U-Net → **离线阶段**：① 校准数据收集（20 步 DDIM, 256/步 = 5120 校准点）→ ② LES 逐层学习 τ（4000-6000 iter, B=32, α=20, L = Σ λ_{t_i}||X_iW - Q(X_i/τ)Q(τ^TW)||²）→ ③ BRECQ 权重量化重建 → ④ PTS 因子投票（仅 skip connection 层，D=3, κ=0.85）→ ⑤ 融合：τ^TW 预计算存储，τ⊙s^X 预计算 → **推理**：⑥ 量化前向 X̃ = MinMaxQ_8bit(X/(2^δ⊙τ⊙s^X)) → ⑦ CUDA kernel：加载 W̃（INT4）→ Ŵ^{shifted} = W̃ ≪ δ → INT8@INT32 GEMM → 反量化 Y = s^X·s^W·C → 输出 → ⑧ DDIM 更新 → 重复 T 步。结果：W4A8 FID=30.37（Baseline=36.08, ↓15.8%），W4A6 FID=26.38（Baseline TFMQ-DM=29.76, ↓11.4%）。W4A6 下 Stable Diffusion 的 LPIPS=0.537（TFMQ-DM=0.691, ↓22.3%），CLIP=30.67（TFMQ-DM=25.32, ↑21.1%）。
  - 系统框架：PyTorch + LDM（https://github.com/CompVis/stable-diffusion）+ 自定义 CUDA kernel（W4A8 GEMM with bit-shift）。评估使用 guided-diffusion 的 ADM TensorFlow 评估器（50K 样本）。
  - 编译框架：论文未明确说明。
  - kernel调度（Section E）：自定义 CUDA kernel 将量化 + bit-shift + GEMM + 反量化融合为单 kernel，在 M=3072 时 vs FP32 GEMM 达到 5.17× 加速。bit-shift 在权重加载时执行（不影响 multiply-accumulate 路径），PTS 仅影响 skip connection 层（网络子集），整体延迟增长极小。
  - 硬件架构：论文未明确说明（GPU 上执行 PyTorch + CUDA kernel）。

  关键设计动机映射：
  - SmoothQuant 手动 τ 导致权重量化误差暴增 → LES 通过梯度下降学习最小化输出 MSE 的 τ，避免手动启发式
  - 通道间异常值仅靠等效缩放无法根除 → PTS 用 2 的幂次缩放直接压缩超大激活值，完全不同的机制
  - 等效缩放转移负担不消除 → PTS + bit-shift 在硬件层面以极低成本消除异常值影响
  - 均匀/固定时间步加权忽略早期步积累效应 → Adaptive Timestep Weighting 动态优先低误差高影响步
  - 小校准集下直接 MSE 选择 δ 过拟合 → Voting Algorithm 基于统计共识选择 δ，仅 r_k>0.85 的通道生效
  - Skip connection 层的 extreme outlier 是主要瓶颈 → PTS 仅针对性应用于 skip connection 层（消融验证优于全层应用）

## D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

- baseline方法是什么？
  Baseline 是标准 **PTQ（后训练量化）+ 已有量化误差修正方法**，具体参考以下三种方法：
  1. **PTQ4DM**：基于正态分布的校准采样策略，使用 MSE 最小化确定量化参数（scale s, zero-point z, bit-width b），但不专门建模或修正量化噪声对采样过程的影响。
  2. **Q-diffusion**：提出时间步感知的校准策略和 U-Net 的 shortcut-splitting 量化方法，在量化阶段优化校准数据分布，但不修正推理时的量化噪声偏差。
  3. **PTQD**：假设量化噪声与量化输出线性相关，基于此假设进行 PTQ 误差修正。但该线性假设在不同时间步并不总是成立，导致修正不精确。

  Baseline 全栈执行例子（LDM-4 W4A8 PTQ 条件生成 ImageNet 256×256，scale=3.0, η=0.0, steps=20）：
  - 算法pipeline：FP32 预训练 LDM-4 → BRECQ/Adaround PTQ 量化（W4A8，uniform asymmetric quantization，首尾层固定 8-bit）→ 逐时间步逆扩散采样：x_t 输入量化后的噪声估计网络 model_q → 输出 ε̂_θ(x_t, t)（含量化噪声 Δε）→ 直接代入 DDIM/DDPM 采样方程计算 x_{t-1}（不对量化噪声做任何矫正）。量化噪声通过 drift term 改变采样方向（均值偏差 μ_Δ），通过 diffusion term 放大采样波动（方差偏差 σ²_Δ），导致生成质量下降。PTQD 的 FID：W8A8=8.46, W4A8=10.41。
  - 系统框架：PyTorch + LDM（latent-diffusion）。BRECQ 块重建量化，Adaround 自适应舍入权重量化。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。性能评估使用 BOPs 衡量理论加速，无定制 CUDA kernel 或实测延迟。
  - 硬件架构：论文未明确说明（NVIDIA GPU 执行 PyTorch 推理）。

  **Baseline 的核心缺陷（通过 SDE 分析揭示）：**
  1. **均值偏差改变采样轨迹方向**：量化噪声均值 μ_Δ(t) 被叠加到反向 SDE 的 drift term `f(x,t) + g(t)²·(ε_θ + μ_Δ)/σ_t` 中，导致每个时间步的采样方向偏离最优路径，使生成图像偏离目标分布。
  2. **方差偏差放大采样波动**：量化噪声方差 σ²_Δ(t) 增加了反向 SDE 的 diffusion coefficient，使 `g(t) → g(t) + g(t)²·σ_Δ(t)·√(dt)/σ_t`，导致采样轨迹波动增大、收敛性下降。
  3. **PTQD 的线性假设不成立**：PTQD 假设 Δε 与 ε̂ 线性相关，但实验显示它们的联合分布更接近二元高斯分布（图 2c），线性建模无法精确刻画条件化量化噪声。
  4. **时间步差异性被忽略**：不同时间步的量化噪声分布不同（均值和方差随时间变化），baseline 方法未针对每个时间步 t 建立独立的噪声模型。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **D²-DPM（Dual Denoising for Quantized Diffusion Probabilistic Models）**，通过量化噪声的统计建模和双重去噪机制解决 baseline 缺陷：

  **(1) 时间步感知的量化噪声建模（TSQNM）（解决缺陷 3-4）**
  - 发现量化噪声在每个时间步近似服从高斯分布（Observation #1），且量化输出也近似服从高斯分布（Observation #2）
  - 建立量化输出 ε̂ 和量化噪声 Δε 的**联合高斯分布**：`[ε̂; Δε] ~ N([μ̂_ε; μ_Δ], [[σ²_ε̂, σ_ε̂Δ]; [σ_ε̂Δ, σ²_Δ]])`（假设元素不相关+各向同性简化）
  - 在推理时，通过 BRECQ 校准收集的样本估计每个时间步 t 的联合分布参数
  - 推理时，根据量化输出 ε̂_θ(x_t, t)，通过条件高斯公式精确预测量化噪声的**条件均值**和**条件方差**：
    - `μ_{Δε|ε̂} = (σ_ε̂Δ/σ²_ε̂)·(ε̂ - μ̂_ε) + μ_Δ`
    - `σ²_{Δε|ε̂} = σ²_Δ - σ²_ε̂Δ/σ²_ε̂`
  - 这比 PTQD 的线性假设更精确，且天然捕捉了时间步差异性

  **(2) 随机双重去噪 S-D²（解决缺陷 1-2）**
  - 从量化输出中减去完整的估计量化噪声：ε' = ε̂ - Δε'（其中 Δε' 从 N(μ_{Δε|ε̂}, σ²_{Δε|ε̂}) 中采样）
  - 恢复后的扩散噪声 ε' 与原始扩散噪声 ε 同分布：E[ε'] = E[ε]，Var[ε'] = Var[ε]
  - 然后代入标准 SDE 采样方程求解，均值和方差均被精确修正
  - 适用于 η=1.0（DDPM，有充足随机性容量）的场景

  **(3) 确定性双重去噪 D-D²（解决缺陷 1-2，利用缺陷 2 的"有利"成分）**
  - 仅减去量化噪声的条件均值：ε' = ε̂ - μ_{Δε|ε̂}（确定性修正 drift term 方向）
  - 额外方差 σ²_Δ 被巧妙地**吸收到扩散项**中：`g'(t) = √(g²(t) - g⁴(t)·σ²_Δ(t)/σ²_t)`
  - 当 g(t) 较大时（η=1.0），额外标准差被随机项有效容纳，不造成有害的方差溢出
  - 当 g(t) 较小时（η=0.0, ODE），额外方差反而**补偿了随机项**，将原始 ODE 隐式转换为更强的 Langevin SDE：数据对齐 ODE 的边际分布 p_t(x)，而额外的 Langevin 扩散项提供了更好的误差缓冲，平滑了每步量化带来的尖锐噪声
  - 这解释了实验中 D-D² 在低随机性容量（η=0.0）时反而部分优于 S-D² 的现象

  D²-DPM 全栈执行例子（LDM-4 W4A8 ImageNet 256×256，scale=3.0, η=0.0, steps=20）：
  - 算法pipeline：FP32 LDM-4 → BRECQ+Adaround W4A8 PTQ → 校准阶段：收集 (ε̂_i, Δε_i)_t 对 → 逐时间步估计高斯联合分布（对角协方差，各向同性简化）→ **推理采样**（双重去噪循环 T...1）：
    1. `ε̂_θ(t) = model_q(x_t)` — 量化模型前向
    2. TSQNM 条件化预测：`μ_cond = (σ_ε̂Δ/σ²_ε̂)·(ε̂_θ - μ_ε̂) + μ_Δ`，`σ²_cond = σ²_Δ - σ²_ε̂Δ/σ²_ε̂`
    3. D-D²：`ε' = ε̂_θ - μ_cond`，调整扩散项 `g' = √(g² - g⁴·σ²_cond/σ²_t)`
    4. DDIM 更新：`x_{t-1} = √α_{t-1}·(x_t - √(1-α_t)·ε')/√α_t + √(1-α_{t-1} - |Σ_t|^{1/d})·ε' + Σ_t^{1/2}·ε_t`
    - 最终 D-D² W4A8 FID=9.71（vs FP FID=11.13, PTQD FID=10.41），即量化模型 FID 比全精度模型还低 1.42
  - 系统框架：PyTorch + LDM（latent-diffusion）+ BRECQ PTQ 框架。校准数据集为 ImageNet 训练集子集。
  - 编译框架：论文未明确说明。
  - kernel调度：论文未明确说明。BOPs 理论加速 11.67× (W8A8) / 23.33× (W4A8)，体积压缩 3.99× / 7.95×。
  - 硬件架构：论文未明确说明（NVIDIA GPU 执行 PyTorch 推理）。

  关键设计动机映射：
  - PTQD 线性假设不准确 → TSQNM 用高斯联合分布→条件分布精确建模
  - 均值偏差改变轨迹方向 → 条件均值 μ_{Δε|ε̂} 修正 drift term
  - 方差偏差放大采样波动 → S-D² 完整减去估计噪声恢复分布；D-D² 将额外方差吸收到扩散项中
  - D-D² 在 ODE 下仍有优势 → 额外方差补偿随机项，将 ODE 隐式提升为更强的 Langevin SDE

## DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

- baseline方法是什么？
  Baseline 是端到端微调旋转矩阵的 LLM PTQ 方法，具体包括 **SpinQuant** 和 **OSTQuant**。核心流程：(1) 在 Transformer block 中插入 4 个旋转矩阵 R1-R4（基于 Computational Invariance）；(2) 将 R1, R2 视为可学习网络参数，插入 pseudo-quantizers；(3) 使用 Cayley SGD（黎曼优化器，在 Stiefel 流形上优化保证正交性）在 calibration set 上端到端微调；(4) 优化目标为 task-specific loss（如 KL divergence）。校准完成后将 R1, R2 融合入相邻权重矩阵实现零推理开销。

  Baseline 全栈执行例子（SpinQuant, LLaMA-2 70B, w4a4kv16）：
  - 算法pipeline：加载 FP16 LLaMA-2 70B → 在 Transformer block 中插入 pseudo-quantizers（W4A4）→ 初始化 R1,R2 为随机 Hadamard 矩阵 → 用 WikiText2 128 samples 作为校准集 → Cayley SGD 端到端微调 R1,R2（每步计算梯度 → Cayley 变换→ 投影回 Stiefel 流形保证正交 → 更新） → 量化 GPTQ 重建权重 → 融合 R1/R2/R1^T/R2^T 入相邻权重。校准需 42.9 GPU-hours、238.89 GiB（A800）。
  - 系统框架：PyTorch + HuggingFace Transformers。使用 Cayley SGD 优化器（`cayley_optimizer`）。
  - 编译框架：论文未明确说明。
  - kernel调度：推理时使用快速 Hadamard kernel 处理 R3, R4（在线旋转）。R1, R2 已离线融合，无额外推理开销。
  - 硬件架构：NVIDIA A800 GPU 服务器。

  **Baseline 的核心缺陷（通过图 3 和表 1 揭示）：**
  1. **资源开销巨大**：70B 模型旋转优化需 42.9 GPU-hours + 238.89 GiB 显存（SpinQuant），OSTQuant 更高达 44 GPU-hours + 583.86 GiB。与 PTQ "快速部署" 目标矛盾。
  2. **端到端微调引入过拟合**：小校准集上的 task-specific loss 微调导致过拟合——表 1 显示 SpinQuant 在 PTB 校准集上 PPL 提升明显（37.91→38.24），但在其他数据集退化严重（WikiText2: 5.47→6.02, C4: 7.26→8.13）。零样本任务上 SpinQuant 和 OSTQuant 反而不如 QuaRot（随机 Hadamard）。
  3. **无法显著降低 outliers 和量化误差**：图 3 显示端到端微调后的旋转矩阵在减少 outliers 数量和降低量化误差方面改进有限——变换后的激活与随机 Hadamard 差异不大。
  4. **Cayley SGD 正交优化器计算昂贵**：需在 Stiefel 流形上做复杂投影计算（约 6n³ 额外计算量），优化时间为标准 SGD 的约 2 倍。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DartQuant**，通过三个创新设计从根本解决 baseline 的问题：

  **(1) Rotational Distribution Calibration 替代端到端微调（解决缺陷 1-3）**
  将旋转矩阵优化从 "网络参数端到端微调" 重新定义为 "将激活分布变换为最适合量化的分布"。不再使用 task-specific loss，而是直接约束旋转后激活的分布特性——最小化 outliers 数量。这消除了对 task-specific loss 的依赖，从根本上避免过拟合。通过仅 128 条校准样本收集激活值（无需标签），校准过程不涉及反向传播至模型输出，大幅降低资源消耗。

  **(2) Whip Loss 驱动激活趋向均匀分布（解决缺陷 3）**
  Baseline 的量化 loss、方差、峰度等目标均无法有效优化激活分布。Whip Loss 的数学设计：`Whip = Σ exp(-|x_i|)`。受 Laplace→Uniform 的 CDF 变换 `U_X(x) = τ[exp(x/b)-1]` 启发，Whip 在零附近有较大梯度（将小值推开），在 norm-invariance 约束（||Rx|| = ||x||）下产生 "峰平滑" 效应：小值增大 → outliers 被迫减小以保持 L2 范数不变 → 整体分布趋向均匀。图 6 直方图验证：Whip 优化后的激活分布最接近均匀分布，outliers 被有效消除。图 7a 验证：Whip 的量化误差下降曲线远优于量化 loss、方差、峰度。

  **(3) QR-Orth 替代 Cayley SGD（解决缺陷 4）**
  不再直接在 Stiefel 流形上优化 R，而是引入隐参数 Z（任意矩阵），通过 QR 分解获得正交矩阵 `R = QR(Z)` 作为实际旋转矩阵。优化 Z 可以用任何标准优化器（SGD/Adam）在欧几里得空间进行，无需投影操作。校准完成后丢弃 Z，仅保留 R 融合入模型。QR-Orth 的额外计算量约 4/3 n³（vs Cayley 6n³），实测 100 步 SGD 耗时 5.7h vs Cayley 8.2h（1.44×）。由于收敛更快，QR-Orth SGD 仅 6 步即达到 Cayley SGD 100 步效果（41× effective 加速）。

  论文方法全栈执行例子（DartQuant, LLaMA-2 70B, w4a4kv16）：
  - 算法pipeline：加载 FP16 LLaMA-2 70B → 前向传播 128 WikiText2 samples 收集各层激活 X → Token sampling 10% → 初始化 Z_0 为随机 Hadamard 矩阵 → **每层独立校准（Algorithm 1）**：for k=0..T (T=10 epochs, lr=1e-3, SGD, batch=64) → R = QR_decomposition(Z) → O = X @ R → L = Whip(O) → Z = Z - η ∂L/∂Z → 最终 R = QR(Z) → **融合**: R1 融入 W_q/W_k/W_v/W_up/W_gate/W_o/W_down/W_embedding/W_lm_head，R2 融入 W_v/W_o → GPTQ 量化权重为 INT4 → 推理时激活 INT4 量化 → INT4×INT4 TensorCore GEMM → INT32 结果转换为 FP16。校准耗时 0.91 GPU-hours (A800) / 2.90 GPU-hours (3090)，内存 23.47 GiB。Llama-2 70B w4a4kv16 零样本 avg=69.02 (FP16=69.53, loss only 0.5%)。
  - 系统框架：PyTorch + HuggingFace Transformers。基于 SpinQuant/QuaRot 代码修改。标准 SGD + QR-Orth 替代 Cayley SGD。
  - 编译框架：论文未明确说明。
  - kernel调度：推理使用快速 Hadamard kernel 处理在线 R3, R4（与 SpinQuant 相同）。R1, R2 预融合无推理开销。
  - 硬件架构：NVIDIA A800 GPU 服务器 / RTX 3090 单卡。首次在 3090 上完成 70B 旋转校准。

  关键设计动机映射：
  - 端到端微调过拟合 + 资源高 → Rotational Distribution Calibration：用激活分布约束替代 task-specific loss，仅需前向传播收集激活
  - 量化 loss/方差/峰度无法有效减少 outliers → Whip Loss：数学上驱动 Laplace 分布趋向 Uniform，在 norm-invariance 约束下产生 "峰平滑" 效应
  - Cayley SGD 投影计算昂贵（6n³ 额外）→ QR-Orth：隐参数 Z → QR 分解得 R，欧几里得空间优化，仅 4/3 n³ 额外
  - 端到端微调不能显著降低量化误差（图 3）→ Whip 直接降低量化误差（图 7a 验证快速收敛），图 6 直方图验证分布均匀化效果

## Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight SLMs on the Edge

- baseline方法是什么？
  Baseline是现有QAT方法的典型方案——具体对比了**NIPQ**（噪声注入伪量化）、**PACT**（参数化裁剪激活量化）和**LLM-QAT**（数据无关QAT+蒸馏），三者都采用粗粒度逐层量化（layer-wise quantization，即每层一个scale factor）。

  Baseline的核心缺陷有两个层面：

  **(1) 算法层面——量化自注意力模块信息失真**：
  现有粗粒度QAT（包括LLM-QAT等）直接将逐层量化应用于自注意力模块的query和key投影，导致：(i) query和key量化后分布方差显著偏离FP16（Figure 2），信息熵降低，等效于MOE/MAE准则下引入较大量化误差；(ii) 注意力图中初始token列的特有分布模式（distinct column pattern）消失（Figure 3），自注意力模块的表征能力退化。论文实验（Figure 1）表明：仅量化query和key导致的性能下降几乎等同于量化整个self-attention模块。

  **(2) 硬件层面——细粒度量化与SIMD硬件不兼容**：
  SOTA QAT方法（LLM-QAT、EfficientQAT、TSLD等）采用channel-wise或token-wise细粒度量化（同一矩阵内多个scaling factor），在GPU上可有效恢复精度。但移动端SIMD（ARM NEON）的GeMM kernel无法处理同一矩阵内有多个scaling factor的整数MAC操作，细粒度量化无法在移动设备硬件上高效部署。标准SIMD INT8 multiplier也不支持sub-8-bit混合精度MAC，4-bit数据被零扩展（zero-extend）到byte边界当8-bit处理，浪费计算能力。

  Baseline全栈执行例子（LLM-QAT, LLaMA-58M W4A4）：
  - 算法pipeline：加载FP16预训练LLaMA-58M → 插入逐层伪量化器（layer-wise symmetric quantization, per-matrix single scale）→ 用FP16教师模型蒸馏：L_distill = (1-γ)·L_CE + γ·τ²·L_KL → STE近似梯度反向传播 → 逐层更新权重 → 量化权重以INT4格式存储。没有熵/分布感知的针对性优化，也没有token粒度的自适应量化。W4A4 BLiMP All Avg=66.9%（FP16=69.7%），NIPQ=48.9%（4-bit权重崩塌），PACT=64.9%。
  - 系统框架：PyTorch + HuggingFace Transformers。GPU训练。移动端推理使用标准SIMD INT8 GEMM kernel（如gemmlowp/QNNPACK），4-bit权重/激活需零扩展至8-bit处理。
  - 编译框架：论文未明确说明（标准PyTorch eager mode，无编译器修改）。
  - kernel调度：移动端使用标准INT8 SIMD kernel（gemmlowp或QNNPACK），仅支持逐层单scale的INT8×INT8 GEMM，不支持sub-8-bit混合精度和token级自适应量化。
  - 硬件架构：商用ARM Cortex CPU（Snapdragon/BCM2712），SIMD最高支持8-bit粒度，无自定义硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**Squat (EdgeQAT)**框架，通过三个递进的创新设计分别解决baseline各层面缺陷：

  **(1) Entropy-Guided & Distribution-Aligned Optimization 解决量化自注意力信息失真（对应缺陷1）**：
  - **熵损失 L_E**：利用query/key近似高斯的特性（q~N, k~N），推导熵H(q)∝σ_q²和H(k)∝σ_k²。最大化熵等价于MOE准则下最小化量化误差（Messerschmitt, 1971）。L_E = -log(Σ_l Σ_h log(1+σ_q²·σ_k²))，对数缩放防止梯度爆炸。
  - **分布损失 L_D**：对每层每头计算量化注意力图attn_q与FP16注意力图attn_f的余弦相似度，恢复初始token列特征。L_D = log(Σ_l Σ_h cos_sim(attn_q, attn_f))。
  - 消融验证（Figure 7）：L_D比L_E更有效（单独使用提升更多），两者组合最优。

  **(2) Token Adaptive Quantization 解决token级冗余未利用问题（对应缺陷1延伸）**：
  基于注意力图中初始token列（attn[:,0]）评估每个token的重要性，TopK选择ρ比例重要token分配8-bit、其余4-bit。Token Control Logic Module (TCLM)用Heapsort高效执行分组+拼接+分别量化。混合策略（half 4-bit + half 8-bit）优于等价位宽的均匀量化（如6-bit uniform），因重要token获得更高精度、非重要token节省计算——在Raspberry Pi上混合W4A8额外加速超40%（vs pure W8A8）。

  **(3) SIMD-based MKMP Multiplier 解决移动端硬件不兼容（对应缺陷2）**：
  - **INT4 Concatenation**：将相邻两行4-bit权重拼接入16-bit寄存器，用ARM `mla`指令（32-bit目标寄存器）同时做乘加，4-bit GEMM的计算操作数减半（vs 传统零扩展到8-bit）。
  - **INT4 Multiplier**：基于现有INT8 multiplier构建，利用bit-shift + row-wise summation累加，节省50% INT8 multiplier资源。
  - **TCLM集成**：8-bit token组走INT8 multiplier、4-bit token组走INT4 multiplier，在GeMM kernel内无缝衔接。
  - **Compiler优化**：分配计算线程重叠内存读取，缓解LLM推理的memory-bound瓶颈。

  论文方法全栈执行例子（Squat, LLaMA-58M W4A8(1:1)混合精度, OnePlus 11推理）：
  - 算法pipeline：加载FP16 LLaMA-58M → 插入逐层对称量化器（W=INT4 per-matrix, A=mixed 4/8 per-token）→ **训练**：FP16教师蒸馏 + L_E(×0.5)最大化query/key熵 + L_D(×1.0)对齐注意力图 → 每步前向TCLM根据最新注意力图动态分配token位宽 → STE反向传播 → 收敛后输出量化权重+scale。W4A8(1:1) BLiMP All Avg=69.4%（仅↓0.3% vs FP16），优于W8A8 uniform（69.3%）。
  - 系统框架：PyTorch训练（GPU）。移动端推理：自定义MKMP multiplier（ARM NEON SIMD kernel）+ gemmlowp/QNNPACK INT8 kernel复用。
  - 编译框架：论文未明确说明（从编译器层面优化内存读取时间线程分配，但未修改编译框架本身）。
  - kernel调度：MKMP Multiplier → TCLM（Heapsort分组）→ INT8 Multiplier（8-bit token组，`vmlaq_s8()`）→ INT4 Multiplier（4-bit token组，concatenation + `mla` + 内部拆分 → bit-shift累加）→ 合并结果。OnePlus 11: W4A8(1:1)=2.23 ms/tok（vs FP16=4.54, 2.04×），GPT2-97M Raspberry Pi 5: W4A4=9.74 ms/tok（vs FP16=23.04, 2.37×）。
  - 硬件架构：商用ARM Cortex CPU（Snapdragon 8 Gen 2 / BCM2712），无自定义硬件。SIMD指令粒度≥8-bit，通过INT4 concatenation突破sub-8-bit效率瓶颈。

  关键设计动机映射：
  - Baseline粗粒度QAT无query/key针对性优化 → Squat引入熵损失（最大化信息熵=最小化量化误差）+ 分布损失（恢复注意力图结构）
  - Baseline均匀量化浪费token级冗余 → Token自适应量化：按attention score分配位宽，混合精度优于等价均匀精度
  - 细粒度QAT（channel/token-wise multi-scale）移动端无法部署 → Squat坚持逐层粗粒度量化（per-matrix single scale），per-token分组仅改变位宽不改变scale
  - 标准SIMD INT8 multiplier不支持sub-8-bit混合精度 → MKMP multiplier用INT4 concatenation实现50%资源节省，TCLM无缝衔接INT8/INT4两种multiplier
  - INT4精度下降大（W4A4 BLiMP ↓1.9%）→ W4A8混合策略用部分8-bit重要token弥补精度，同时获得4-bit加速（Raspberry上额外40%）

## EfficientQAT Efficient Quantization-Aware Training for Large Language Models

- **baseline方法是什么？**
  Baseline有两类：(1) PTQ方法（GPTQ、AWQ、OmniQuant、AutoRound）采用block-wise reconstruction，但仅训练少量量化参数（rounding参数、clipping阈值或步长），限制优化空间且忽略跨block交互，低比特下精度损失严重；(2) Q-PEFT方法（PEQA、QA-LoRA、QLoRA）冻结量化权重，仅训练少量连续浮点参数或LoRA adapter，低比特下无法充分恢复量化信息损失。

  Baseline全栈执行例子（GPTQ, Llama-2-7B W2G128）：
  - 算法pipeline：加载FP16预训练Llama-2-7B → 逐block执行贪心OBQ量化：对每行权重，逐列贪心选择量化误差最小的列进行量化+补偿剩余权重 → Hessian矩阵用于计算补偿量 → 仅优化rounding决策（量化后向上/向下取整），所有权重值不参与梯度优化。无端到端训练、无跨block交互。W2G128 C4 PPL=不可用（退化严重），5-task Avg Acc≈41.56%。
  - 系统框架：PyTorch + HuggingFace Transformers。GPU推理。量化模型通过标准INT2 packing存储，推理时解包为FP16执行矩阵乘法。
  - 编译框架：论文未明确说明（标准PyTorch eager mode推理，无自定义编译器）。
  - kernel调度：依赖标准cuBLAS FP16 GEMM kernel，量化权重在kernel调用前解量化为FP16。无定制化INT2/INT3 kernel优化。
  - 硬件架构：NVIDIA A100-80GB GPU，Tensor Core仅加速FP16/INT8运算，INT2/INT3无硬件原生支持。

- **论文方法是什么？如何对应解决Baseline的缺陷？**
  论文提出**EfficientQAT**两阶段QAT框架，通过Block-AP和E2E-QP分别解决PTQ和Q-PEFT各自的缺陷：

  **(1) Block-AP解决PTQ优化空间受限（对应PTQ缺陷）**：
  - PTQ（GPTQ/OmniQuant/AutoRound）仅训练rounding参数或clipping阈值，每次权重更新被限制在(-1,+1)区间内作为正则化防止过拟合，但大幅缩小解空间 → Block-AP是首个在block-wise reconstruction中直接训练所有权重和量化参数的方案（W, s, z全训练），无需额外引入rounding参数设计。
  - 实验证明（Table 5）：Block-AP (s,z,W全训练) PPL=8.53 vs 仅训练rounding PPL=15.50 vs 仅训练clipping PPL=11.28。且全训练内存(8.5GB)低于rounding训练(8.6GB)，因后者需额外保存rounding参数副本。
  - 训练数据和epoch数：Block-AP仅需4096样本、2 epoch即可收敛，验证损失与训练损失差距从1.07缩至0.06（Figure 3）。

  **(2) E2E-QP解决跨block交互缺失（对应PTQ缺陷）**：
  - Block-AP虽恢复了block内精度，但各block独立训练忽略跨block交互 → E2E-QP冻结Block-AP产出的量化权重W_q，仅端到端训练步长s（每个group的scale factor）。步长s参数占比约1.6%（g=64），使端到端训练内存极低（Llama-2-70B W2G64仅需34.2GB）。
  - E2E-QP中无需量化操作（Eq.1前向不执行），仅执行反量化W_hat = (W_q - z) * s → 梯度仅需计算∂W_hat/∂s = W_q - z，计算图简单高效。

  **(3) Block-AP+E2E-QP组合解决Q-PEFT精度不足（对应Q-PEFT缺陷）**：
  - Q-PEFT（PEQA等）使用RTN初始化量化权重后仅训练步长，无法恢复低比特(2/3-bit)的严重信息损失 → EfficientQAT先用Block-AP提供高质量初始化，再用E2E-QP跨block微调。
  - Table 4消融：仅RTN（即无Block-AP无E2E-QP）Avg PPL=453.49；Block-AP单独降至8.53；E2E-QP单独降至9.33；组合(BP+E2E-QP)降至7.68，Avg Acc从40.69→58.99→55.71→60.14。

  论文方法全栈执行例子（EfficientQAT, Llama-2-7B W2G64）：
  - 算法pipeline：加载FP16预训练Llama-2-7B → **Block-AP**：逐block将线性层权重W量化为W_int=clamp(round(W/s)+z,0,3)，反量化W_hat=(W_int-z)*s → 前向用W_hat计算block输出 → MSE损失对齐FP16 block输出 → STE反向传播同步更新W(梯度截断)、s(Eq.3梯度)、z(Eq.4梯度)，epoch=2，lr_W=2e-5，lr_s=1e-4 → 输出W_q(N-bit)、s(FP16)、z(N-bit) → **E2E-QP**：冻结W_q和z，仅训练s → 全模型前向（反量化W_hat=(W_q-z)*s，无量化前向）→ 语言模型cross-entropy损失 → 仅s反向更新，lr=2e-5，epoch=1，ctx=4096 → 最终模型W_q(N-bit)+s(FP16)+z(N-bit)存储，推理时反量化为FP16执行。W2G64 C4 PPL=8.50，5-task Avg Acc=60.14%（vs FP16=64.85%）。
  - 系统框架：PyTorch + HuggingFace Transformers。单A100-80GB GPU训练（70B 2-bit仅需34.2GB E2E-QP内存）。量化模型兼容MLC-LLM、AWQ、BitBLAS、Marlin、T-MAC等推理框架。
  - 编译框架：论文未明确说明（标准PyTorch eager mode，无自定义编译框架）。
  - kernel调度：使用BitBLAS在A100-80GB上评估INT2矩阵向量乘法加速（2.9x-4.4x vs FP16）。量化kernel原理：INT2权重packing存储 → kernel内SIMD解包 → 低精度整数MAC → 乘步长s反量化 → FP16累加输出。
  - 硬件架构：NVIDIA A100-80GB GPU。标准CUDA Core执行低精度整数运算，Tensor Core可通过BitBLAS等工具映射到INT8硬件通路。


## GPTVQ: The Blessing of Dimensionality for LLM Quantization

- baseline方法是什么？
  Baseline 是 **均匀 INT4 量化 + GPTQ 后训练量化**（例如 GPTQ W3/W4 g128）。标准 LLM 推理全栈中，权重以 INT4 存储，每个 group（128 个权重）共享一个 FP16 scale，推理时通过 scale 反量化到 FP16 后执行矩阵乘法。

  Baseline 全栈执行例子（Llama-2-7B INT4 g128, 移动 CPU/GPU）：
  - **算法pipeline**：FP16 权重 → GPTQ 逐列量化 + Hessian 误差补偿 → 每 group 128 个权重共享一个 scale → 推理时读取 INT4 packed 权重 → scale 反量化到 FP16 → FP16 GEMM。均匀量化的 grid 是等间隔的，优化空间仅为 2^4=16 个等间隔值。
  - **系统框架**：llama.cpp（开源）或自研推理引擎。Llama.cpp 使用 Q4_0 INT4 量化（block size 32）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：CPU 端使用 SIMD 指令（ARM NEON / x86 AVX）加速 INT4 反量化 + GEMM。GPU 端使用 CUDA kernel 或 Triton kernel。
  - **硬件架构**：移动端 Snapdragon X Elite CPU / NVIDIA GPU（RTX 4090/A100/H100），无自定义硬件。

  **Baseline 的核心缺陷：**
  1. **均匀量化 grid 表达能力受限**：均匀量化将每个 group 映射到等间隔的 2^b 个值上，无法自适应权重分布。当权重分布非均匀（如长尾分布、多峰分布）时，等间隔 grid 浪费大量量化级别在低密度区域，高密度区域精度不足。
  2. **DRAM 带宽瓶颈限制 token rate**：LLM 自回归推理中每生成一个 token 需从 DRAM 读取所有权重一次，DRAM 带宽是主要瓶颈。INT4 虽已将 FP16 压缩 4×，但 8B 模型仍需约 4.3 GB footprint，DRAM 带宽余额有限。
  3. **现有 VQ 方法在移动端低效**：AQLM 等方法使用 8D VQ + 16-bit 索引 + 大 codebook（2^16 个 8D entries），无法利用移动 CPU 的 TBL 指令（仅支持 5-6 bit index → 8-bit value）。解码 latency 过高，抵消了 footprint 减小的带宽收益。
  4. **AQLM 压缩时间长**：Llama-v2-7B 需约 35 小时 on H100（含 block FT），GPU 资源消耗巨大。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **GPTVQ**，通过 VQ 表示 + 移动 CPU 实现 + 快速 PTQ 算法的协同设计，系统性解决 baseline 各项缺陷：

  **(1) VQ 替代均匀量化，提升 representational accuracy（解决缺陷 1）**
  将权重按 d 维向量分组量化到非均匀 codebook。对于给定 bit budget（如 3.125 bpv），2D VQ 的 64 个质心可自由分布在二维空间中，形成非均匀 grid，比均匀 INT4 的 16 个等间隔点表达能力更强。论文用 SQNR metric 验证：d 越高（1D→2D→4D），SQNR 越高，且实验证实 2D 和 4D VQ 在几乎所有模型和 bitwidth 下比 1D VQ 和均匀量化有更低 perplexity 和更高零样本准确率。

  **(2) 硬件友好的 VQ 参数选择 + LUT 解码 kernel（解决缺陷 3）**
  与移动 CPU 硬件协同设计 VQ 参数：固定 2D VQ + 6-bit index（3 bits/dimension），codebook 最多 64 entries。这直接匹配移动 CPU 的 TBL 指令规范（6-bit index → 8-bit value），每个维度仅需 1 条 TBL 指令解码。对比 AQLM 的 16-bit index 需要 SVE gather 指令（性能更差），GPTVQ 的解码延迟极低，使 VQ 的 footprint 减小（19%）能转化为实际的 token rate 提升（10%）。Table 6 验证：CPU 端 VQ 2D 2.25 bpv 延迟 0.87× vs INT4，吞吐反超。

  **(3) GPTVQ 算法 = GPTQ 扩展 + 加权 EM 初始化 + Codebook Update（解决缺陷 4）**
  - 将 GPTQ 的逐列量化扩展为逐 d 维向量量化，误差沿 d 维累积后一次性补偿
  - EM 初始化：用 Hessian 加权的马氏距离（公式 4-6），E-step 分配质心，M-step 伪逆闭式解更新质心，比标准 k-means 更好地利用校准数据信息
  - Codebook update（附录 A）：GPTVQ 完成后通过梯度下降（PyTorch）层内微调 codebook 值，以极小开销（~30% 额外时间）提升精度
  - 结果：Llama-v2-7B 2D VQ 压缩时间仅 2.5h（H100），vs AQLM no BFT 18.3h（7.3× 加速），且精度 competitive（WikiText2 PPL 7.11 vs AQLM 7.49）

  **(4) 正交组合 LoRA adapters 恢复精度（额外贡献）**
  GPTVQ 量化后的 base model 可与 LoRA adapters 结合：frozen adapter（FP16 模型训练的 LoRA 直接挂载）或 trained adapter（在量化模型上训练 LoRA）。GPTVQ 4D 2.125 bpv + LoRA trained 在 GSM8k 上达 32.5-35.0%（L2-7B），显著超越 LoftQ（20.9%）。

  论文方法全栈执行例子（Llama-v3-8B 2D VQ 3.125 bpv, Snapdragon X Elite）：
  - **算法pipeline**：FP16 权重 → 校准集（WikiText2 128×2048）前向收集 Hessian → 逐 column block：每 256 columns 进入新 group → EM 初始化该 group 的 8-bit codebook（64 entries, 2D）→ GPTVQ 逐 d=2 列量化 + Hessian 补偿 → Codebook update（梯度下降 fine-tune）→ Codebook 量化到 INT8（或 INT4）→ 输出：packed 6-bit indices + per-block 64-entry INT8 LUT + per-block FP16 scale → **推理**：DRAM → SoC cache → TBL 指令解码 6-bit index → 2D 值合并 → scale × decoded int = FP16 → SIMD GEMM。Footprint 3.52GB (-19% vs INT4 4.33GB)，Throughput 26.15 tok/s (+10% vs Ours INT4 23.81 tok/s, +45.7% vs llama.cpp INT4 17.95 tok/s)。
  - **系统框架**：Qualcomm 自研 C 语言推理引擎（vector intrinsics + SIMD + polyhedral compiler）。移动端 Snapdragon X Elite + Clang 18.1 + Polly。
  - **编译框架**：Polyhedral compiler（Polly）用于细粒度向量化编排。
  - **kernel调度**：CPU TBL kernel（解码）+ SIMD GEMM kernel。GPU CUDA kernel（char4/uchar4/char128 vector types）。VQ 2D CPU 数据加载延迟测试：3.125 bpv = 0.96× 延迟 vs INT4（同 footprint 仅 0.78×），2.25 bpv = 0.87× 延迟 vs INT4（footprint 仅 0.56×）。
  - **硬件架构**：Snapdragon X Elite CPU（ARM TBL 指令 6-bit→8-bit）+ NVIDIA GPU（RTX 3080 + H100）。无自定义硬件，仅利用现有 CPU ISA 扩展。

  关键设计动机映射：
  - Uniform quantization 表达力低（16 等间隔点）→ VQ codebook 64 个任意分布质心，提高 SQNR + 降低 PPL
  - DRAM 带宽瓶颈 → 更小的 bpv（3.125 vs 4.125）减少 footprint 19%，直接转化为 10% token rate 增益
  - 移动 CPU TBL 指令特性（6-bit→8-bit）→ 2D VQ + 6-bit index 配置匹配硬件，解码快于 DRAM 带宽
  - AQLM 压缩时间长 → GPTVQ 单次从左到右扫描（复用 GPTQ lazy update）+ 闭式解 EM，2.5h vs 18.3h
  - VQ 固有精度损失 → Codebook update 梯度下降 + LoRA adapter 正交补偿

## Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

- baseline方法是什么？
  Baseline 是 **NF4（Normal Float 4-bit）** [Dettmers et al. 2023] 和 **INT4** 量化。NF4 假设权重服从正态分布 N(0,σ²)，使用 Gaussian 分位数函数等概率划分 16 个量化层级。INT4 为均匀量化，在 [-8, 7] 范围内等间距分布。

  **Baseline 全栈执行例子**（LLaMA2-7B, weight-only PTQ with NF4, block size 128）：
  - **算法pipeline**：加载预训练 FP32/FP16 权重 → 按 block size 128 分块 → 每块用 absmax（w_max = max|w_i|）归一化到 [-1,1] → NF4 码本查表量化（16 个固定值基于 Gaussian 分位数）→ 存储 4-bit index + per-block w_max（FP16）。推理时：Ŵ = w_max × NF4_table[index]。
  - **系统框架**：bitsandbytes 库（HuggingFace PEFT/QLoRA 集成），基于 PyTorch + HuggingFace Transformers。推理时使用 CUDA fused dequantization kernel。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：bitsandbytes 4-bit CUDA kernel（dequantize → FP16 GEMM），无专用硬件 kernel。
  - **硬件架构**：NVIDIA GPU（无自定义硬件）。但论文同时评估了各数据类型的 MAC 单元面积和功耗（SystemVerilog + Design Compiler + TSMC 28nm），INT4 MAC = 160.7 µm²、48.5 µW。

  **Baseline 的核心缺陷**：
  1. **正态分布假设错误**：论文对 30+ DNN 的 weight/activation 做大规模 profiling，用 Kolmogorov-Smirnov 检验证明大多数 DNN 分布由 Student's t-distribution（ν≈5）最优近似，而非正态分布。正态分布无法同时拟合分布的尖峰（peak）和厚尾（tail）（Figure 2：Mistral-7B 的 Q-Q plot 中 t-distribution 呈直线，normal 显著偏离）。NF4 基于错误分布假设，其量化层级在概率空间的分布与实际 weight 分布不匹配。
  2. **INT4 均匀量化忽略分布结构**：INT4 在 [-8,7] 区间等间距分布，绝大多数量化层级落在权重稀少的边缘区域，而对权重密集的中心区域仅分配少量层级，导致对典型值的量化精度不足。
  3. **E2M1 FP4 浪费位数空间**：因存在正负零的浮点表示冗余，E2M1 仅使用 15/16 = 93.75% 的位数空间。在 4-bit 仅有 16 个可能值的极端受限条件下，6.25% 的浪费显著。
  4. **缺少质量-效率联合优化视角**：数据类型选择通常在精度和硬件效率之间各说各话，没有系统的 Pareto 权衡分析。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过四个层次的设计逐一解决 baseline 缺陷：

  **(1) SF4（Student Float）基于 t-distribution 导出最优查找表（解决缺陷 1）**
  NF4 用 Gaussian 分位数划分，SF4 改用 Student's t-distribution (ν=5) 分位数。Algorithm 1 将概率质量均匀分 16 份，经 t-distribution 分位数函数 Q_S(p; ν=5) 映射到量化值空间。这确保了每个量化层级的使用频率大致相等（等概率原则），量化直方图近似平坦。实验证实 SF4 比 NF4 在各 LLM 上持续提升精度（如 LLaMA2-7B LAMBADA: NF4=71.98%, SF4=72.54%），且 ν 的选择基于 profiling 结果（最频 ν≈5），非任意参数。

  **(2) Supernormal Support 回收 E2M1 的浪费位数（解决缺陷 3）**
  将 E2M1 的负零位重映射为额外超常值，提出两种变体：
  - **Super-range (SR)**：将负零 → 8.0，扩展动态范围。精度提升有限，因额外点位于分布边缘很稀疏的区域。
  - **Super-precision (SP)**：将负零 → 5.0，在分布内部增加一个层级。精度提升更显著（如 Phi-2 W4A4: E2M1 平均准确率降 -8.41%, E2M1+SP 降至 -7.25%），因额外层级位于高概率密度区域。SP 的硬件开销（MAC 面积 +27.9%，系统 +3.6%）高于 SR（MAC +12.3%，系统 +1.9%），但在精度-面积 Pareto 上提供更高精度选项。

  **(3) t-distribution 洞察解释 E2M1 为何优于 INT4（解决缺陷 2 的根源）**
  论文发现 E2M1 的形状分段逼近 SF4：E2M1 对分布中心的小值区域分配更密集的层级（0, 0.5, 1, 1.5, 2），而对边缘的极大值分配稀疏层级（3, 4, 6）。这恰好匹配 t-distribution 的尖峰厚尾特征——中心概率密度高需要更细粒度量化，尾部概率密度低可以用粗粒度。这就从理论上解释了为何 FP4 优于 INT4：不是因为浮点格式本身，而是因为 E2M1 的形状隐含地匹配了 t-distribution 的形状。

  **(4) 质量-效率 Pareto 曲线系统化设计指导（解决缺陷 4）**
  论文首次将 11 种 4-bit 数据类型在模型精度（LAMBADA+HellaSwag+Winogrande+PIQA+BoolQ+ARC-c 平均准确率损失）和芯片面积（MAC 单元面积 + 系统级开销估算）两个维度上绘制 Pareto 曲线，揭示：
  - Pareto frontier: INT4（最低面积/精度）→ E2M1（0.6% 系统开销，精度损失降低 7.34%）→ E2M1+SP（3.6% 系统开销，最高精度）
  - E2M1-I 和 E2M1-B 属于严格劣化点（面积更大且精度更低），应从实际部署中排除
  - APoT4 精度接近 E2M1 但需要额外格式转换逻辑，实用性有限

  **论文方法全栈执行例子**（以 LLaMA2-7B weight-only PTQ with E2M1+SP 为例）：
  - **算法pipeline**：
    1. 离线 t-distribution profiling（可选，ν=5 已固定）→ 确定 SF4/E2M1+SP 量化层级
    2. 加载预训练权重 → 按 block size 128 分块 → 可选 MSE clipping calibration 优化裁剪阈值
    3. 对每 block：归一化 → E2M1+SP 码本查表量化（16 个层级，含 SP 值 5.0）→ 存储 4-bit index + per-block FP16 scale
    4. 推理：dequantize → FP16 GEMM（标准流程，与 NF4/INT4 完全相同）
  - **系统框架**：修改版 Intel Neural Compressor 库（添加 lookup-based quantization for SF4/NF4/E2M1 variants + supernormal support）。基于 PyTorch + HuggingFace Transformers。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：与 NF4 相同的查表解码 + FP16 GEMM（bitsandbytes 风格）。论文未涉及自定义 kernel。
  - **硬件架构**：
    - E2M1 MAC 单元：17-bit accumulator, 总面积 170.4 µm², 功耗 49.6 µW
    - E2M1+SP MAC 单元：19-bit accumulator（需更大累加器容纳 SP 引入的额外值 5.0），总面积 218.0 µm², 功耗 54.6 µW
    - 系统开销：E2M1 0.6%，E2M1+SP 3.6%（假设 MAC 占芯片 10%、存储 60%）
    - 设计决策：SP 的 MAC 面积比 SR 大（27.9% vs 12.3%），但精度提升也更显著，在精度敏感场景值得额外面积投入

## LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

- baseline方法是什么？
  Baseline 是 **QLoRA** [Dettmers et al., 2023]：先对预训练权重 W 直接做 N-bit 量化得到 Q = q_N(W)，再按标准 LoRA 方式初始化低秩适配器 A ∼ N(0,σ²), B = 0。由于量化引入了不可忽略的误差，初始权重 Q + AB^T = Q ≠ W，导致 LoRA fine-tuning 的起点偏离原始预训练权重。这在低比特（如 2-bit）场景尤其严重——QLoRA 在 2-bit 下直接不收敛（perplexity 爆炸），在 3-bit/4-bit 下也存在与 full fine-tuning 之间的持续性能差距。

  **Baseline 全栈执行例子**（DeBERTaV3-base, NF2 2-bit QLoRA on MNLI）：
  - **算法pipeline**：加载 FP16 预训练权重 W ∈ R^{d1×d2} → 用 NF2 量化函数 q_2(·) 直接量化 W → Q = q_2(W)（存储为 2-bit index + lookup table + absmax）→ 初始化 A ∈ R^{d1×r} ∼ N(0,σ²), B = 0 → fine-tuning 时 freeze Q，AdamW 优化 A,B → 推理：Y = X × dequant(Q) + X × A B^T。由于 Q 与 W 的偏差 ‖W − Q‖_F 大，且 A,B 初始化为零无法补偿，fine-tune 从错误的起点开始。
  - **系统框架**：HuggingFace Transformers + bitsandbytes/PyTorch 量化后端。QLoRA 使用 NF4 双量化（double quantization）和分页优化器（paged optimizer）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：bitsandbytes 4-bit CUDA kernel（dequantize → FP16 GEMM）。2-bit 下需自定义 kernel 或使用 simulated quantization（论文使用后者）。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **量化与 LoRA 初始化解耦**：QLoRA 先独立量化 W 得到 Q，再用零初始化 LoRA，这两个步骤完全独立。目标函数 ‖W − Q‖_F 与后续 fine-tuning 的目标无关，没有考虑 LoRA adapter 可以补偿量化误差的潜力。
  2. **初始权重偏离**：量化误差 ‖W − Q‖_F 随比特数降低指数增长。2-bit 时误差巨大，Q 与 W 的谱范数和 Frobenius 范数差异显著（Figure 2 验证），fine-tuning 无法从此起点恢复。
  3. **LoRA 零初始化的浪费**：标准 LoRA 零初始化 (B=0) 保证训练起点等于预训练权重，但量化后这一保证失效（Q ≠ W），零初始化白白浪费了 LoRA adapter "提前补偿" 量化误差的能力。
  4. **对低比特缺乏弹性**：2-bit QLoRA 在所有模型（DeBERTaV3/BART/LLAMA-2）和所有任务上均不收敛（表 1/2/4/5 均为 N.A.），说明方法缺乏对极端量化的容错。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LoftQ 通过交替优化将量化与低秩近似耦合，解决 baseline 每个缺陷：

  **(1) 联合目标函数（解决缺陷 1 的解耦问题）**
  不再独立做量化和 LoRA 初始化，而是联合优化 min_{Q,A,B} ‖W − Q − AB^T‖_F。该目标函数显式承认 LoRA adapter AB^T 可以在 fine-tuning 前就补偿部分量化误差，使 Q + AB^T 比单独的 Q 更接近 W。这是 LoftQ 的核心洞察——将 LoRA fine-tuning 的初始化视为优化问题的一部分。

  **(2) 交替优化缩小初始化差距（解决缺陷 2 的偏离问题）**
  每一 alternation step：(a) 量化 W 去除当前低秩近似后的残差 Q_t = q_N(W − A_{t-1}B_{t-1}^T)；(b) SVD 分解量化残差 W − Q_t，取 top-r 分量作为新的低秩近似 A_t B_t^T。这确保每一步中，量化聚焦于"低秩分量尚未覆盖的部分"，而 SVD 则补偿"量化无法表达的部分"。效果：2-bit 时 LoftQ 的初始化 ‖W − (Q_T + A_T B_T^T)‖_F 远小于 QLoRA 的 ‖W − Q‖_F（Figure 2 验证谱范数和 Frobenius 范数均大幅降低）。

  **(3) 非零 LoRA 初始化（解决缺陷 3 的浪费问题）**
  输出不再要求 B=0，而是直接使用 SVD 得到的 A_T, B_T。这些 adapter 包含量化残差中的低秩结构信息（最大的 r 个奇异值/向量），在 fine-tuning 前已部分恢复量化损失的精度。这不同于先在量化模型上训练 LoRA 然后用作初始化——LoftQ 在训练前通过纯代数方法（SVD）找出"最佳补偿"。

  **(4) T=1 已是有效方案，T>1 进一步可选（解决缺陷 4 的弹性问题）**
  T=1 时 Q_1 恰好等于 QLoRA 的量化权重，而 A_1 B_1^T 是 W−Q_1 的 top-r SVD——即仅添加一步 SVD 后处理。这已经让 DeBERTaV3-base 2-bit MNLI-m 从 QLoRA 的 79.9% 提升到 84.7%（+4.8%, rank 16）。T=5 进一步达到 88.0%（+8.1%）。LLAMA-2-7b 2-bit 上 LoftQ 收敛到 WikiText-2 PPL 7.85，QLoRA 直接不收敛。

  **论文方法全栈执行例子**（DeBERTaV3-base, Uniform 2-bit, rank=32, T=5, on MNLI）：
  - **算法pipeline**：
    1. 离线 LoftQ（逐矩阵并行，CPU 执行，单矩阵 <1s）：
       W_in ← 加载预训练权重矩阵（如 q_proj W_q ∈ R^{768×768}）
       A_0, B_0 = 0
       for t=1..5:
         Q_t = UniformQuant_2bit(W_in − A_{t-1} B_{t-1}^T)  // 2-bit 均匀量化当前残差
         R_t = W_in − Q_t                                   // 量化误差
         U, Σ, V^T = SVD(R_t)                              // 全 SVD
         A_t = [√σ₁ u₁, ..., √σ₃₂ u₃₂]                     // top-32 左奇异向量
         B_t = [√σ₁ v₁, ..., √σ₃₂ v₃₂]                     // top-32 右奇异向量
       存储：Q_5 → M[768][768]（2-bit int）+ scale[768]（FP16 per-column）+ lookup table（4 entries）
       LoRA init: A_5, B_5

    2. LoRA Fine-tuning（NVIDIA A100, AdamW）：
       for batch in MNLI_train:
         Q_sim = dequant(M, table, scale)               // [768, 768] FP16
         h = X @ Q_sim^T + X @ A_5 @ B_5^T             // adapter 非零初始化
         ... 其余 Transformer 层同理（所有 MHA + FFN 权重均量化+adapter）
         loss = CrossEntropy(logits, labels)
         loss.backward()                                 // 仅 A,B 有梯度, M 冻结
         AdamW.step(A, B)

    3. 推理：
       Q_sim + A_5 @ B_5^T 作为融合后的权重直接计算
  - **系统框架**：HuggingFace Transformers（基于 PyTorch），LoftQ 作为预处理步骤在 fine-tuning 前离线执行。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：Simulated quantization（存储为整数 + 查表解量化出 FP16 再进入标准 FP16 GEMM），无自定义 kernel。与 bitsandbytes 4-bit kernel 思路一致但扩展支持 2-bit。
  - **硬件架构**：NVIDIA A100 GPU。无自定义硬件。

  关键设计动机映射：
  - QLoRA 量化误差 → 联合目标 min‖W−Q−AB^T‖_F 显式建模 adapter 补偿
  - 2-bit 精度崩溃 → 交替 SVD 提取量化残差的低秩结构，缩小初始化差距
  - 零初始化不补偿量化 → 非零 A_T,B_T 初始化，预补偿量化误差
  - 对量化函数无依赖 → q_N(·) 可替换为任意量化方法（Uniform/NF4/NF2 均验证）
  - 预处理成本可控 → 逐矩阵独立执行+可并行，无需训练或梯度计算

## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- baseline方法是什么？
  - **Transformer（GPT3/LLaMa 风格）**：基于 multi-head self-attention (MHA) 的序列模型。attention 机制通过 QK^T 计算所有 token-pair 的相似度得分，经 softmax 归一化后对 V 加权求和。核心优势是内容感知（content-aware）推理能力强——每个 token 可以"关注"上下文中任意位置的 token。但存在两个根本性缺陷：i) **二次复杂度**：训练 FLOPs = O(BL²D)，推理时需存储 KV cache（每 token 约 2·n_layers·D 个浮点数），自回归生成每步需重读整个 cache，导致 O(L) 时间/步；ii) **有限上下文窗口**：无法建模窗口外的信息，长序列性能受限。
  - **LTI（Linear Time-Invariant）SSM（S4, H3, Hyena 等）**：基于结构化状态空间模型的序列模型。参数 (Δ, A, B, C) 在时间上固定不变，可通过卷积模式（FFT，O(L log L)）做并行训练，或循环模式（O(1)/步）做自回归推理。优点是线性/近线性缩放于序列长度。核心缺陷是**无法进行内容感知推理**（lack of content-based reasoning）：模型动态（Ā, B̄）对所有 token 相同，无法根据当前 token 内容"选择"传播或遗忘哪些信息。这在 Selective Copying（需根据内容决定记住哪些 token）和 Induction Heads（需根据上下文检索相关信息）等任务上暴露为致命弱点。
  - 全栈执行例子（以 Transformer baseline 为例）：
    - **算法层**：输入 token ID → embedding → 逐层 multi-head attention（Q=XW_Q, K=XW_K, V=XW_V → A=softmax(QK^T/√d_k) → O=AV → OW_O）→ FFN/SwiGLU MLP → residual + LayerNorm → LM head → softmax → 采样/argmax 输出下一个 token
    - **系统框架层**：基于 PyTorch + HuggingFace Transformers（GPT3/LLaMa 实现），推理时维护 KV cache 结构（每层存 K, V ∈ R^{B×n_heads×L×d_head}），自回归生成时每步追加新 token 的 K,V 到 cache
    - **编译框架层**：使用 FlashAttention-2 CUDA kernel（tiling + recomputation 将完整 QK^T 矩阵限制在 SRAM 内计算），torch.compile 做图优化
    - **kernel调度层**：FlashAttention-2 将 Q,K,V 分 tile 加载到 SRAM，在线 softmax rescaling 避免将中间 attention matrix 写回 HBM
    - **硬件架构层**：NVIDIA A100 GPU (80GB HBM, 108 SM, 40MB L2 cache, SRAM per SM 192KB)
  - 全栈执行例子（以 LTI SSM — S4/H3 baseline 为例）：
    - **算法层**：输入 x ∈ R^{B×L×D} → 逐通道应用 SSM: 预计算卷积核 K̄ = (CB̄, CĀB̄, ..., CĀ^{L-1}B̄) → y = x ∗ K̄（FFT 加速卷积）或 h_t = Āh_{t-1} + B̄x_t, y_t = Ch_t（循环模式）。参数 (Ā,B̄) 对所有 t 相同
    - **系统框架层**：H3 架构（SSM sandwich：gate → shift-SSM(局部卷积) → SSM → gate），模块间 interleave MLP blocks。PyTorch + FFT convolutions
    - **编译框架层**：论文未明确说明（依赖 PyTorch 默认 FFT/conv 实现）
    - **kernel调度层**：标准卷积 kernel（FFT-based: FFT(x) × FFT(K̄) → IFFT），O(L log L) FLOPs，但无法解决 Selective Copying 任务
    - **硬件架构层**：同 Transformer — NVIDIA A100 GPU
  - Baseline 核心缺陷总结：
    - Transformer: 二次复杂度 → 长序列训练/推理成本高；KV cache → 推理内存线性增长、batch size 受限
    - LTI SSM: 缺乏选择性 → 无法在序列维度上做内容感知的"聚焦/忽略"决策 → 在离散信息密集型数据（文本、DNA）上效果差

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Mamba 方法**通过三项核心创新逐一解决：
    1. **选择机制（Selection Mechanism, S6）**：将 SSM 参数 (Δ, B, C) 从静态改为输入 x 的函数，使模型可以"理解输入内容"后决定传播还是遗忘信息。具体而言：Δ_t = softplus(Parameter + Linear_1(x_t)) 控制"关注当前 vs 保持历史"（大 Δ ≈ 关注当前输入并 reset 状态 → 选择机制；小 Δ ≈ 忽略当前并保持 → 过滤无关信息）。B_t = Linear_N(x_t) 和 C_t = Linear_N(x_t) 提供输入到隐藏状态和隐藏状态到输出的细粒度内容调制。Theorem 1 证明选择机制是经典 RNN gating 的泛化
    2. **硬件感知并行扫描算法**：选择性 SSM 不再是 time-invariant，丢失了卷积形式（FFT）的可用性。若用朴素循环，需物化大小为 (B,L,D,N) 的中间状态 h（比输入大 N=16 倍）。Mamba 通过 kernel fusion（将离散化+扫描+输出计算融合在 SRAM）+ parallel scan（Blelloch 算法，O(L) work O(log L) depth）+ recomputation（反向时重计算 h 而非从 HBM 读取）解决了这一问题，IO 减少 O(N) 倍，实测比 naive scan 快 20–40×
    3. **简化的 Mamba 架构**：将 H3 的 SSM 块和标准 MLP 块合并为同质化单一模块（gate → Conv1d → SiLU → Selective SSM → × gated SiLU → output projection），无需 attention 甚至无需 MLP 块。每个块有 3ED² 参数（E=2 固定），两个 Mamba 块 ≈ 一个 Transformer 块（12D²）
  - 论文方法全栈执行例子：
    - **算法层**：
      输入 token ID → embedding → 逐层 Mamba block →
        x → RMSNorm → Linear (投影到 gate + main 两分支, 2ED 维) →
        gate分支: SiLU → 作为 multiplicative gate
        main分支: Conv1d (kernel=4) → SiLU → 选择性 SSM (S6):
          Δ = softplus(Linear_R(x) + bias) [R=64, 输入投影→D维broadcast]
          B = Linear_N(x), C = Linear_N(x)  [N=16]
          Ā = exp(Δ ⊙ A), B̄ = Δ ⊙ B  [ZOH discretization, fused in SRAM]
          h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t  [parallel scan, fused in SRAM]
          y_ssm_t = C_t ⊙ h_t
        → y_ssm × gate → Linear (投影回 D 维) → residual → RMSNorm →
      → 最后层输出 → LM head → softmax → 采样
    - **系统框架层**：PyTorch + 自定义 CUDA kernel (fused selective scan) + 标准 HuggingFace-style 训练 pipeline（AdamW, BF16, gradient clip）。自回归推理无需 KV cache——每步仅将新的 (h_t, x_t) 送入循环更新，O(1) 时间和 O(DN) 内存/步
    - **编译框架层**：论文未明确说明（使用 CUDA 直接实现 scan kernel，未修改编译器框架）
    - **kernel调度层**：
      Fused Selective Scan kernel 执行流程（per chunk in SRAM）：
        Load: Δ(BLD), A(DN), B(BLN), C(BLN) from HBM [共 O(BLD)]
        SRAM: discretize(Δ, A, B) → Ā, B̄ (BLDN) → parallel scan → h (BLDN) → y = C⊙h (BLD)
        Write: y (BLD) to HBM
      反向: 重新加载输入 O(BLD) → 重计算 h → 计算梯度 → 写回 O(BLD)
      总 HBM IO ≈ 2BLD（vs naive 的 3BLDN，N=16 时节省 16×）
    - **硬件架构层**：NVIDIA A100 GPU。利用 GPU 内存层级（HBM → L2 cache → SM shared memory/SRAM → register），将扫描完全限定在 SRAM 执行避免 HBM 往返。当 L 超过 SRAM 容量时分 chunk 处理（chunk 间通过 HBM 传递 scan state）
  - 关键设计动机映射：
    - Transformer O(L²) 复杂度 → 选择性 SSM 的 O(L) 训练 + O(1) 推理（无需 KV cache）
    - LTI SSM 缺乏内容感知 → 选择机制（Δ, B, C 输入依赖）实现上下文相关的信息过滤/记忆 → 解决 Selective Copying 和 Induction Heads
    - 选择机制破坏卷积可用性 → 硬件感知 fused parallel scan（kernel fusion + recomputation）克服效率瓶颈
    - H3/Transformer 异构架构复杂 → Mamba 同质化简化（H3 + MLP 合一），无 attention、无 MLP 块的极简设计
    - LTI SSM 长上下文不改善（甚至恶化）→ 选择机制天然支持过滤无关上下文 → DNA/音频 1M 长度下性能单调提升

## MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

- baseline方法是什么？
  - 现有 INT4 weight-activation 量化方法（如 Atom、QuaRot、QUIK、FlatQuant）采用固定数量的高精度通道（如 Atom 固定 128 个 INT8 通道），或使用旋转/平滑变换抑制 activation outlier。这些方法的 INT kernel 需要在 CUDA Core 上进行反量化（因为 INT8 Tensor Core 仅输出 INT32 部分和），无法利用 Blackwell 的 FP4 Tensor Core。
  - 全栈执行例子（Atom baseline）：
    - **算法层**：固定 keeper_size=128 个 INT8 通道，其余为 per-group INT4 (group_size=128)，activation sort metric="hessian"，无自适应层间精度分配
    - **系统框架层**：PyTorch + 自定义 CUDA kernel，INT8 Tensor Core 执行 GEMM，CUDA Core 执行 dequant + partial sum（INT32→FP16 转换），仅支持 Llama2-7B
    - **编译框架层**：论文未明确说明
    - **kernel调度层**：INT4×INT4 MMA 在 INT8 Tensor Core 上计算 → INT32 部分和 → CUDA Core 上 dequant（乘以 scale）→ FP16 累加。INT8 Tensor Core 限制：FP4 吞吐为 FP16 的 4×，而 INT8 Tensor Core 仅 2× FP16 吞吐；且 dequant 在慢速 CUDA Core 上执行
    - **硬件架构层**：NVIDIA RTX 4090 (Ada Lovelace) 等非 Blackwell 架构，无 FP4 Tensor Core 支持

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MicroMix 通过三个关键设计解决 baseline 缺陷：(1) **自适应混合精度分配**——每层基于激活分布动态计算 MXFP4/MXFP6/MXFP8 通道比例，替代固定 128 通道；(2) **量化阈值 T(n)**——从 INT8 量化误差上界推导 MXFP4/MXFP6 的允许值域，超阈值元素升级精度，确保 MXFP 误差不超 INT8；(3) **Blackwell 原生 FP4 kernel**——利用 Blackwell MMA 指令直接执行 FP4/FP6/FP8 GEMM，反量化融合在 Tensor Core 内完成，无需 CUDA Core dequant。
  - 全栈执行例子（MicroMix）：
    - **算法层**：每层离线计算 p4^k/p6^k/p8^k 和排列 σ^k（基于校准数据的通道均值排序 + 阈值分组），激活 online fused reorder-and-quantize，权重 offline 预量化。量化 block_size=32（比 Atom 的 group_size=128 更细粒度），使用 E8M0 scale 实现纯移位反量化
    - **系统框架层**：PyTorch + 自定义 CUTLASS GEMM kernel，支持 Llama/Qwen/Mixtral 多模型系列，模型无关的通用混合精度框架
    - **编译框架层**：基于 CUTLASS 模板实例化各精度 GEMM kernel，未修改编译器框架本身
    - **kernel调度层**：Blackwell Tensor Core 上执行：
      1. Fused reorder-and-quantize kernel（共享内存内重排 + 32 元素 block-wise MX 量化，E8M0 scale shift-only dequant）
      2. 三路 MXFP GEMM（MXFP4/MXFP6/MXFP8），MMA 指令融合 scale dequant → FP32 累加 → BF16 输出
      3. 通道恢复排列 σ^{-1}
      FP4 Tensor Core 提供 4× FP16 吞吐，dequant 零额外开销（MMA 内置），CUTLASS GEMM 高度解耦支持任意精度比例
    - **硬件架构层**：NVIDIA RTX 5070Ti/5090/PRO 6000 (Blackwell)。利用 FP4 Tensor Core (4× FP16 吞吐)，MMA 指令原生支持 MX 格式的 block-scaled 数据类型。shared memory / Tensor Memory 缓存 input tile + scales
  - 缺陷→设计映射：
    - 固定通道数忽略层间分布差异 → 自适应 p4/p6/p8（每层独立校准），p4 始终 >50% 保证效率
    - INT kernel 需要 CUDA Core dequant → Blackwell MXFP MMA 原生融合 scale dequant，反量化零额外延迟
    - 无 MXFP 异常值阈值定义 → 首次给出 MXFP4/MXFP6 的显式量化阈值 T(n)，确保误差不超过 INT8 上界
    - Atom kernel 仅支持 Llama2-7B → MicroMix 的 CUTLASS 解耦设计支持任意模型和多精度组合
    - 粗粒度 group quantization (group_size=128) → MX 标准 block_size=32 的细粒度量化，E8M0 移位反量化消除乘法开销

## MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

- baseline方法是什么？
  - Baseline 是 **Wanda (2:4 结构化稀疏)** 和 **传统 post-training weight pruning 方法**（如 SparseGPT）。这些方法对 LLM 的线性层权重矩阵做非结构化或半结构化稀疏，虽然能减少总参数数量，但依赖专用硬件（FPGA 等）才能实现高效部署。此外还有 **Random Expert Pruning**（随机丢弃专家）和 **Frequency-based Expert Pruning**（按校准数据上的激活频率丢弃专家）作为 MoE 专家级 baseline。
  - 全栈执行例子（以 Wanda 2:4 baseline 在 Mixtral 8x7B 上一个 token 的推理为例）：
    - **算法层**：Wanda 逐层对每个线性层的权重计算 importance score = |W_{ij}|·‖X_j‖_2，在 2:4 模式下每 4 个连续权重保留 2 个，其余置零 → 得到稀疏权重矩阵。Mixtral 8x7B 上 Wanda 2:4 实现约 50% 参数减少，但推理速度反而低于 dense 模型（0.91-0.92× speedup），因为 2:4 结构化稀疏需要特定硬件加速（NVIDIA Ampere Sparse Tensor Core 或 FPGA），通用 GPU 上无加速优势。此外权重稀疏对 MoE 架构无针对性优化——专家参数占总参数 ~96%，但 Wanda 对所有权重同等对待，不区分专家重要性差异。
    - **系统框架层**：基于 PyTorch semi_structured_sparse（https://pytorch.org/tutorials/prototype/semi_structured_sparse.html）实现。推理时使用 HuggingFace Transformers 标准加载管线，稀疏权重以 dense 格式存储（mask + values），无专门 MoE 优化。原始 Mixtral 8x7B (bf16) 需 2 张 A100-80G GPU 部署，Wanda 2:4 后仍需 2 张 GPU（51GB 内存）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel，无自定义 sparse kernel。2:4 结构化稀疏在通用 GEMM kernel 上无加速（甚至因 mask 检查额外开销而减速）。
    - **硬件架构层**：NVIDIA A100-80G GPU，未使用专用稀疏硬件加速器。
  - **Baseline 的核心缺陷**：
    1. **权重级稀疏与 MoE 结构不匹配**：Wanda/SparseGPT 对所有 FFN/Attention 权重均匀裁剪，忽略 MoE 架构中专家才是主要参数载体（8 个专家占 Mixtral 8x7B 总参数 96%）这一结构特征。专家作为独立的 FFN 子网络可整体移除而无需改变其余模型结构，这是权重级稀疏无法利用的。
    2. **需要专用硬件才能实现推理加速**：2:4 结构化稀疏的加速依赖 FPGA 或支持 sparse MMA 的 Tensor Core，通用 GPU 上实际减速。部署不具 plug-and-play 特性。
    3. **Random/Frequency baseline 对专家重要性估计不准**：Random 丢弃不考虑专家贡献；Frequency-based 仅按校准数据上的激活频率排序，忽略不同 token 对专家使用的差异——论文实验显示 activation frequency baseline 甚至比 random 更差，因为 MoE 模型可能对特定专家有路由偏好但偏好不完全等于重要性。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法：Expert Pruning + Dynamic Expert Skipping**
  - **(1) Expert Pruning 解决缺陷 1 和 3**：
    - 将稀疏化粒度从"权重元素"提升到"专家"级别，利用 MoE 结构特征：每个 MoE 层包含 n 个独立专家 FFN，专家是天然的可移除单元。通过逐层枚举保留 r 个专家的组合，以最小化 Frobenius 范数量化重构损失 ‖F'(x,C) − F(x)‖_F 为目标，在 token 级别评估专家子集的重要性。
    - 重构损失在 token 输出层面度量（而非权重层面），直接衡量专家组合对模型最终输出的影响，比 activation frequency 更准确地反映专家贡献。在 Mixtral 8x7B 上，论文方法 r=6 时平均性能仅下降 2.9 点（vs Random 4.5 点, Frequency 6.8 点）。
    - 领域特定剪枝：将校准数据集从 C4 切换到 MATH 训练集，使剪枝过程聚焦领域知识保留。在 GSM8K 5-shot 上 C4 剪枝 r=6 仅 41.02 vs MATH 剪枝 51.25——校准数据选择对领域性能至关重要。
  - **(2) Dynamic Expert Skipping 解决加速问题**：
    - 推理时根据路由权重比值 w_{e1}/w_{e0} 与逐层阈值 β（校准集上该比值的中位数）动态决定是否跳过次优专家。不依赖硬件稀疏支持，是纯软件层面的推理加速——通过减少每个 token 实际执行的专家 FFN 数量来减少 FLOPs。
    - 跳过机制基于权重比而非固定阈值，自适应不同 token 的路由分布。β 取中位数使跳过概率约 50%，在精度与加速间取得平衡。
  - **(3) 组合使用实现全局优化**：
    - 剪枝（减少静态参数 → 内存节省）+ 动态跳过（减少运行时计算 → FLOPs 节省）正交互补。r=6 剪枝 + 动态跳过的组合（62.91 avg accuracy）比 r=4 纯剪枝（59.57）精度更高，但推理速度相当（1.23× vs 1.27×）——以更少专家数获得更高精度，证明动态跳过的效率。
  - 论文方法全栈执行例子（以 Mixtral 8x7B 一个 token 推理为例）：
    - **算法层**：
      1. **离线剪枝阶段**：加载 C4/MATH 校准集（128 条 × 2048 tokens）→ 逐层前向传播缓存输入-输出对 → 逐层枚举 expert combinations（C(n,r)），对每个组合 C 计算 F'(x,C) = Σ w̃_{e_j}·E_{e_j}(x)，取 min ‖F'(x,C)−F(x)‖_F 的组合 → 修改模型 config 仅保留 r 个专家 → 保存 pruned checkpoint
      2. **离线 β 校准阶段**：pruned checkpoint 前向校准集 → 逐层收集 w_{e1}/w_{e0} 比值 → β[l] = median(ratios)
      3. **推理阶段**：token x → 每 MoE 层：路由计算 top-2 (e0, e1) → 若 w_{e1} < β[l]·w_{e0} 则 y = E_{e0}(x)，否则 y = w̃_{e0}·E_{e0}(x) + w̃_{e1}·E_{e1}(x)
    - **系统框架层**：HuggingFace Transformers，仅修改模型 config 中的 expert 数量即可加载剪枝模型——不需要修改模型代码或引入新的 layer type。动态跳过的路由逻辑通过自定义 MoE layer forward 实现，核心改动不到 20 行代码。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel。减少 GPU 间通信（从 2 张 GPU 降为 1 张 GPU 后无需跨 GPU 通信）是推理加速的主要来源。动态跳过减少每个 token 的 expert FFN 计算量（平均少执行 0.5 个 expert/token）。
    - **硬件架构层**：NVIDIA A100-80G GPU。剪枝后单卡部署消除了跨 GPU NCCL 通信开销。论文无需专用硬件——plug-and-play 部署。
  - 关键设计动机映射：
    - 权重级稀疏需要专用硬件 → 专家级稀疏（整体移除/跳过专家）在标准 GPU 上即插即用
    - Activation frequency 不反映真实重要性 → 基于 token 重构损失的枚举搜索准确评估专家贡献
    - 静态剪枝无法减少 FLOPs → 动态跳过在线减少激活专家数，真正减少计算量
    - 通用校准集不适合领域任务 → 切换校准数据集到领域数据实现 task-specific pruning
    - 剪枝后模型仍有性能下降 → 通过微调（MetaMathQA 900 步）几乎完全恢复性能，r=7 剪枝模型在 GSM8K 上超越 8-expert 原始模型

## MobiLlama Small Language Model tailored for edge devices

- baseline方法是什么？
  - **baseline1（22 层 / hidden 1024）**：通过减少 hidden dimension size 来缩小模型。22 个 Transformer 块，每层 hidden dim = 1024，每层有独立的 MHA + MLP（含 3 个 FFN）。总参数 0.54B。训练时间 7.5 天（A100）。
    - 缺陷：hidden dim 从 2048 缩减到 1024，模型表征能力受限（bottleneck effect），难以捕捉复杂数据模式。
  - **baseline2（8 层 / hidden 2048）**：通过减少层数来缩小模型。8 个 Transformer 块，每层 hidden dim = 2048，每层有独立的 MHA + MLP。总参数 0.52B。训练时间 7 天。
    - 缺陷：深度从 22 层缩减到 8 层，丧失层次化语言表征学习能力，深层语义理解能力下降。
  - **large-base（22 层 / hidden 2048 / 1.2B）**：直接将 baseline1 的宽度和 baseline2 的深度结合，得到 22 层 + hidden 2048 的模型。每层独立 FFN（占 65% 参数量），总参数 1.2B，训练时间 12 天，GPU 内存 6 GB。虽然精度高（avg 49.06），但参数量和训练成本显著增大，不适合边缘部署。
  - Baseline 全栈执行例子（以 baseline1 为例，0.54B model 推理一个 token）：
    - 算法层：token embedding → 22 层 decoder：每层 RMNorm → MHA（32 heads, Q/K/V/O projection）→ 残差连接 → RMSNorm → SwiGLU FFN（W_gate, W_up, W_down，164M FFN 参数/层）→ 残差连接 → 最终 LM head。总计 22 套独立 FFN 参数。
    - 系统框架层：HuggingFace Transformers PyTorch 推理；边缘部署使用 GGUF 格式 4-bit 量化。
    - 编译框架层：论文未明确说明。
    - kernel调度层：标准 PyTorch CUDA kernel + Flash-Attention（预训练时），无自定义 kernel。
    - 硬件架构层：NVIDIA A100 训练，RTX2080Ti/i7 CPU/Snapdragon-685 部署推理。
  - Baseline 核心缺陷：**宽度（表征能力）和深度（层次化学习）不可兼得**——保持两者都意味着 1.2B 的参数爆炸（large-base），而满足 0.5B 参数约束就必须牺牲其一（baseline1 牺牲宽度，baseline2 牺牲深度）。根源在于：每层独立 FFN 导致 FFN 参数占 65%，构成参数冗余的核心来源。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MobiLlama 共享 FFN 设计**：从 large-base 的架构（22 layer / 2048 hidden）出发，通过**跨层共享 FFN 参数**大幅削减参数量——所有 22 个 Transformer block 复用同一套 FFN（W_gate, W_up, W_down），省去 21 份冗余 FFN 副本。共享 FFN 将总参数从 1.2B 降至 0.5B（减少约 60%），同时保持 22 层的深度和 2048 的宽度。
  - **核心洞察**：FFN 参数占 65% 但并非都需要独立——Transformer 不同层对 FFN 的使用有冗余。通过共享可以保留模型的高容量（深度+宽度），并将节省的参数预算用于保持架构完整性。
  - **如何解决 Baseline 缺陷**：
    - 针对"baseline1 宽度受限"：MobiLlama 保持 hidden dim=2048（vs baseline1 的 1024），不产生 bottleneck effect，模型可充分捕捉复杂模式。
    - 针对"baseline2 深度受限"：MobiLlama 保持 22 层（vs baseline2 的 8 层），保留层次化语言表征学习，深入理解上下文。
    - 针对"large-base 参数爆炸"：共享 FFN 从 1.2B→0.5B（60% 参数减少），训练 GPU 小时从 46.1K→26.6K（42% 减少），GPU 内存从 6GB→3GB（50% 减少）。
  - 论文方法全栈执行例子（以 MobiLlama 0.5B 推理一个 token 为例）：
    - 算法层：token embedding → 22 层 decoder：每层 RMSNorm → MHA（32 heads, 每层独立 Q/K/V/O proj）→ 残差连接 → RMSNorm → **shared SwiGLU FFN**（22 层共用同一 W_gate, W_up, W_down，仅 56M 参数）→ 残差连接 → LM head。注意：共享的仅是 FFN，attention 层的 Q/K/V/O projection 每层独立。
    - 系统框架层：同 baseline，HuggingFace Transformers PyTorch 推理。边缘部署使用 GGUF 4-bit 量化。MobiLlama 0.5B 在 RTX2080Ti bf16：63.38 tok/s，内存 3046 MB，电池 8.19 mAH/1k tokens。
    - 编译框架层：论文未明确说明。
    - kernel调度层：标准 PyTorch CUDA kernel + Flash-Attention，无自定义 kernel。共享 FFN 意味着更少的参数需要从 HBM 加载，减少内存带宽压力。
    - 硬件架构层：预训练用 160×A100(80GB)，部署在 RTX2080Ti/i7 CPU/Snapdragon-685 手机上。无专用硬件设计。
  - 关键设计动机映射：
    - baseline1 宽度受限（hidden 1024 → bottleneck） → MobiLlama 保持 hidden 2048，不产生信息瓶颈
    - baseline2 深度受限（8 layers → 无法深度理解） → MobiLlama 保持 22 layers，保留层次化表征
    - large-base FFN 冗余（65% 参数 × 22 份副本） → 共享 FFN 仅保留 1 份，削减 60% 参数
    - "从大开始再缩小"的设计哲学：先设计高容量架构（large-base），再用参数共享机制降低到目标参数量。这种思路保证了架构设计的最优性，而非在设计之初就在容量上妥协。
  - 开源与透明度：完整训练数据pipeline、训练代码、模型权重、300+ 中间 checkpoints、评估代码全开源（https://github.com/mbzuai-oryx/MobiLlama）。训练数据使用 Amber dataset 1.2T tokens 全透明。

## Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

- baseline方法是什么？
  Baseline 是**标准 PTQ 方法（Q-Diffusion / LCQ）直接量化扩散模型的原始激活值**。各时间步 t 独立进行：加载激活 a_t → 计算量化参数 s, z（min-max 动态计算或通过校准集 MSE 优化）→ 量化 a_t 到低 bit integer → 反量化后送入线性层 A 计算 → 输出 o_t = A(Q(a_t))。每步独立、无跨时间步信息共享。
  
  Baseline 全栈执行例子（以 DDIM + LCQ, CIFAR-10, 100步去噪为例）：
  - 算法层：对去噪 U-Net 中每个线性层/卷积层，逐时间步独立：a_t → 逐通道 min-max scaling → clamp+round → b-bit int → dequantize → A(a_hat_t) → 传递到下一层。各时间步量化参数 s_t, z_t 独立于其他步，无跨步信息或误差补偿。
  - 系统框架层：基于 PyTorch + Q-Diffusion 代码库或 BRECQ 框架，校准数据逐时间步采样。推理时 fake-quantization 模拟量化推理。
  - 编译框架/kernel调度/硬件架构：论文未明确说明（标准 PyTorch CUDA kernel 推理，无硬件加速实现）。
  
  Baseline 核心缺陷（由论文 preliminary study 揭示）：
  1. **激活范围跨时间步波动大**：不同时间步的激活值范围差异显著（图1b 蓝色 violin plot），导致单一量化参数难以覆盖所有时间步，造成严重 clipping/rounding 误差。
  2. **激活分布含大量 outlier**：每个时间步内激活呈长尾分布，大量 outlier 撑大量化步长 s，使得正常值被粗粒度量级覆盖，量化误差大。
  3. **低比特（<6-bit）下质量崩溃**：8-bit activation 是现有 PTQ 方法的安全下限，降至 6-bit 时 FID/sFID 显著退化（如 Q-Diff 8/4 bit CIFAR-10 sFID 从 4.49 → 100.37），4-bit 以下基本不可用。
  4. **缓存方法（如 DeepCache）存在误差累积**：重用历史计算结果跳过某些时间步，但 reuse 误差随步数累积（图1a），最终步误差可达 40%，需 heuristic 手动调 reuse schedule。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoDiff（Modulated Diffusion）**，通过 modulated quantization + error compensation 两大核心机制解决 baseline 缺陷：
  
  **(1) 调制量化（Modulated Quantization）解决缺陷 1-3**
  不直接量化原始激活 a_t，而是量化相邻时间步的差值 a_t − a_{t+1}。利用线性算子 A 的线性性，将计算等价重写为：
  o_t = A(a_t) = A(a_t − a_{t+1}) + o_{t+1}
  然后对差值进行量化：Q(a_t − a_{t+1}) → A(Q(a_t − a_{t+1})) + o_{t+1}
  
  关键洞察（图1b 橙色 vs 蓝色对比）：
  - 原始激活范围大、波动大、含 outlier → 量化误差大
  - 差值范围约 10× 更小、跨步一致性好、分布集中 → 同等 bit-width 量化误差大幅降低
  - Theorem 4.3 量化误差 bound：||x − Q(x)||² ≤ (max(x)−min(x))²d/(2^b−1)²，差值范围缩小直接降低误差 bound
  - 效果：使 PTQ 激活位宽从 8-bit 推至 3-bit 仍无损（CIFAR-10 LCQ+MoDiff W8A3 FID=4.14 vs FP=4.24）

  **(2) 误差补偿调制（Error-Compensated Modulation）解决缺陷 1, 4**
  标准调制直接用原始 a_{t+1}：o_t = A(Q(a_t − a_{t+1})) + o_{t+1}
  问题：量化误差 (a_{t+1} − Q(a_{t+1})) 在每步累积且被缓存传递。
  
  MoDiff 误差补偿：用 â_{t+1} = Q(a_{t+1} − â_{t+2}) + â_{t+2} 替代 a_{t+1}，使上一步量化误差被显式纳入下一步的差值计算：
  â_t = Q(a_t − â_{t+1}) + â_{t+1}
  ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}
  
  重写后等价于：ô_t = A(Q(a_t − a_{t+1} + e_{t+1})) + o_{t+1} − A(e_{t+1})
  即上步误差 e_{t+1} 在下一步被减去 A(e_{t+1})、同时注入 Q 的输入中，实现自动抵消。
  
  Theorem 4.4 理论保证：标准调制误差 O(2^{T−k}) 指数增长，误差补偿调制误差 O((2c)^{T−k}) (c<1/2) 指数衰减。

  全栈执行例子对比（DDIM + LCQ+MoDiff, CIFAR-10, 100步）：
  - 算法层：不再每步独立量化，而是跨步耦合：
    (1) t=T (warm-up)：â_T = Q(a_T), ô_T = A(â_T)
    (2) t=T−1→1：â_t = Q(a_t − â_{t+1}) + â_{t+1}, ô_t = A(Q(a_t − â_{t+1})) + ô_{t+1}
    每层独立执行此流程。解耦后仅需缓存 â_t 和 ô_t（额外内存约 3-4 MB per layer）。
  - 系统框架层：基于 Q-Diffusion + BRECQ 代码库，MoDiff 作为 plugin 无侵入集成。校准数据集重构为基于 MoDiff pipeline 的输入输出对，逐层独立校准以保持稳定。
  - 编译框架/kernel调度/硬件架构：论文未明确说明。硬件实现标为 future work。
  
  关键结果：
  - CIFAR-10 W8A3：LCQ+MoDiff FID=4.14（vs FP=4.24，vs LCQ alone=143.39），IS=9.02（vs FP=9.00），10× 运算节省（154 vs 1636 GBops）
  - LSUN-Churches W8A3：LCQ+MoDiff FID=12.05（vs LCQ=341.62）
  - Stable Diffusion W8A6：LTQ+MoDiff FID=13.21（vs LTQ=71.38）
  - DiT-XL/2 W8A6：PTQ4DiT+MoDiff FID=54.74（vs PTQ4DiT=200.26）
  - 兼容所有 sampler（DDIM/DDPM/DPM/PLMS），兼容 QAT 方法 MixDQ

## MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

- baseline方法是什么？
  - **GPTQ（weight-only uniform quantization）**：对所有 linear block 使用统一 bitwidth 的 weight-only 量化（per-group, group size 128, asymmetric min-max），配合 random Hadamard 变换预处理（incoherence processing）。不考虑 MoE block 内不同 linear block 的量化敏感度差异，也不利用 expert 激活频率差异优化计算效率。
  - **QuaRot（weight-activation uniform quantization）**：统一 4-bit weight + 4-bit activation 量化，使用 Hadamard 旋转消除 outlier。在 W4A4 下精度严重退化（DeepSeek-V2-Lite WikiText2 PPL 8.44 vs FP16 5.92，Qwen2-MoE PPL 110.66 vs FP16 5.84）。
  - **全栈执行例子（GPTQ baseline, DeepSeek-V2-Lite, RTX 4090）**：
    - 算法层：加载 FP16 MoE 模型 → 逐 MoE block 校准：128 seqs WikiText2 → 对每个 expert 的所有 linear block 统一使用 GPTQ 3.25-bit per-channel asym quantization → 得到量化权重。所有 linear block 位宽相同。
    - 系统框架层：基于 PyTorch + HuggingFace Transformers，CUDA kernel 执行 GEMM。调用 VLLM-Marlin-MoE 或 HQQ kernel 处理 low-precision GEMM。
    - 编译框架层：论文未明确说明。
    - kernel调度层：VLLM-Marlin-MoE kernel 顺序调用 Marlin kernel 处理每个 expert 的 GEMM（每次一个 expert），kernel launch overhead 和 GPU under-utilization 严重。HQQ kernel 不做 dequantization fusion，性能更差。两者均不支持混合精度——所有 expert 使用相同精度 kernel。
    - 硬件架构层：NVIDIA RTX 4090 GPU，无自定义硬件。
  - **Baseline 核心缺陷**：
    1. **统一位宽忽视 MoE 内 linear block 的异构量化敏感度**：同一 expert 内 gate_proj 和 down_proj 量化敏感度差异显著（Fig. 1a），统一位宽要么对不敏感 block 浪费 bit 预算，要么对敏感 block 精度不足。在 2.25-bit 下 GPTQ 的 Qwen1.5-MoE WikiText2 PPL 达 11.19（vs FP16 6.79），Mixtral PPL 达 5.69（vs FP16 3.88）。
    2. **不利用 expert 激活频率差异优化计算效率**：expert 激活频率差异超过 10×（Fig. 1b），部分 expert 形成的 GEMM 是 memory-bound（低激活频率，tokens 少），部分是 compute-bound（高激活频率）。统一位宽无法选择性对 memory-bound expert 用 W4A16、对 compute-bound expert 用 W8A8。
    3. **现有 low-precision kernel 缺乏混合精度支持**：VLLM-Marlin-MoE 和 HQQ 均不支持同一 MoE block 内混合精度 GEMM 并行执行，顺序处理导致 GPU under-utilization（Fig. 2）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MxMoE = 硬件感知 linear-block 级 bitwidth 分配 + 自动混合精度 Group-GEMM kernel 生成**：
    - **解决缺陷 1（linear-block 粒度混合精度）**：
      - 对每个 expert 的每个 linear block 独立评估量化敏感度 Δ_{i,j,k}（校准集上的输出 Euclidean distance），通过 ILP 联合优化 L（输出扰动求和）和 T（tile 级执行时间求和），在内存预算 M 约束下求解最优 {x_{i,j,k}}（为每个 linear block 分配一个量化方案 k ∈ S）。
      - Linear-block 粒度 vs expert 粒度：同一 expert 内 gate_proj/down_proj 敏感度不同，expert 级分配只能折中，linear-block 级分配可针对性优化。实验验证 linear-block 分配 consistent 优于 expert 级（Table 3，DeepSeek-V2-Lite PPL 6.11 vs 6.32）。
    - **解决缺陷 2（硬件感知优化）**：
      - 目标函数 T 基于 tile 级 profiling + roofline model。expert 激活频率 f_i 影响其 GEMM shape（token 数决定 m 维度），进而影响 arithmetic intensity（A≈m），决定该 GEMM 是 memory-bound 还是 compute-bound。
      - 硬件感知分配自动对 memory-bound GEMM 分配 W4A16（减少 memory traffic）、对 compute-bound GEMM 分配 W8A8（利用 Tensor Core 高吞吐），同时保证精度约束。W4.25A15.5 在 memory-bound 下比 uniform W4A16 快 up to 25%。
    - **解决缺陷 3（自动混合精度 Group-GEMM kernel）**：
      - Micro-kernel specialization 为每种精度实现专用 CUDA device function（如 W2A16 fused dequant+bit-manip, W4A4-g128 multistage pipeline），避免 universal kernel 的性能损失（unified kernel 比 specialized 慢 13-38%，Table 6）。
      - Resource configuration 强制统一 warp count + shared memory 按最大需求分配 + k-dimension tiling (slice-K)，使不同精度 micro-kernel 可在同一 kernel launch 中水平融合。
      - Tile scheduler 使用 greedy LPT 启发式调度，消除顺序 expert 处理的 kernel launch overhead。
  - **全栈执行例子（MxMoE W5A5, Qwen1.5-MoE, RTX 4090）**：
    - 算法层：离线校准（128 seqs）→ 计算每 linear block 的 {Δ_{i,j,k}} → 统计 expert 激活频率 → ILP 求解最优 {x_{i,j,k}}（r=0.75）→ 各 linear block 按分配方案量化（randomized Hadamard transform + GPTQ）→ 激活运行时动态量化。结果：Qwen1.5-MoE W5A5 WikiText2 PPL 7.01（vs QuaRot W4A4 18.44，+11.43 PPL 提升），Avg Acc 66.72（vs QuaRot 43.47，+23.25%）。
    - 系统框架层：基于 PyTorch + CUDA/CUTLASS。MxMoE kernel generator 自动编译融合 kernel，替代 VLLM-Marlin-MoE/HQQ。
    - 编译框架层：论文未明确说明。
    - kernel调度层：自动生成混合精度 Group-GEMM kernel。一个 kernel launch 内并行处理 MoE block 的所有 expert GEMM（不同精度），tile scheduler 按 greedy LPT 分配 tile 到 SM。消除 VLLM-Marlin-MoE 的 per-expert kernel launch overhead。W5A5 比 FP16 快 3-3.4×（compute-bound），比 uniform W8A8 快 29.4%。
    - 硬件架构层：NVIDIA RTX 4090 GPU。无自定义硬件。
  - **关键设计选择映射**：
    - Baseline 缺陷1（统一位宽忽视敏感度）→ Δ_{i,j,k} 量化扰动建模 + ILP 逐 linear block 优化
    - Baseline 缺陷2（不利用激活频率差异）→ tile 级执行时间建模 T = Σ c·y·x + roofline-guided hardware-aware allocation
    - Baseline 缺陷3（kernel 缺混合精度支持）→ micro-kernel specialization + resource config + tile scheduler 自动生成

## Scaling Law for Quantization-Aware Training

- baseline方法是什么？
  Baseline 是现有 QAT 缩放定律 [Frantar et al. 2025, Kumar et al. 2024]，其核心建模方式为在 Chinchilla 缩放定律 L(N,D) = A/N^α + B/D^β + E 中引入 Effective Parameter Multiplier（EPM）eff(C) 乘以参数项 N：L(N,D) = A/(N·eff(C))^α + B/D^β + E。由此可推导出量化误差 δ_p(N) = A/(N·eff(C))^α − A/N^α，仅依赖模型参数量 N。

  Baseline 全栈执行例子（Kumar 2024 scaling law, W4A4 QAT per-tensor granularity, Llama-style model）：
  - 算法pipeline：用 per-tensor 量化粒度（activation 全层一个 scale）训练 W4A4 QAT 模型 → 统计不同 N 的最终训练 loss → 拟合 eff(C) 作为 N 无关常数 → 预测其他 N 的量化误差。eff(C) 仅依赖模型架构和压缩类型，不随 N、D、G 变化。然而实际实验中，当 D 从 10B→100B tokens 增长时，W4A4 量化误差平均增加 22%（论文 Figure 4b），baseline 无法捕捉这一趋势。
  - 系统框架：基于 PyTorch + HuggingFace Transformers 的标准 QAT 训练。使用 STE 模拟量化前向，FP32 权重 + fake-quantize。
  - 编译框架：论文未明确说明（标准 PyTorch eager mode）。
  - kernel调度：论文未明确说明（标准 CUDA kernel 模拟量化推理）。
  - 硬件架构：NVIDIA A100 GPU。量化格式为 INT4/FP4。

  **Baseline 的核心缺陷：**
  1. **忽略训练数据量 D 对量化误差的影响**：现有缩放定律假定 δ_p 与 D 无关（式 3 中 D 被消除）。但论文实验（Figure 4b）证明 δ_{W4A4} 随 D 增加显著上升（10B→100B 平均 +22%），原因是 QAT 训练中模型参数会"适应"量化误差——更多训练数据意味着更充分的全精度训练收敛，从而放大量化带来的差距。
  2. **忽略量化粒度 G 的影响**：Baseline 未能建模 group-wise 量化粒度对误差的影响，通常使用 per-tensor 或单一固定 G。论文实验显示 finest→coarsest G 的 δ 差异达 0.037（占粗粒度误差的近半数）。不同 G 需分别拟合独立曲线（baseline 需 5 条曲线覆盖 5 种 G），无法统一建模。
  3. **未区分权重/激活误差来源**：Baseline 将量化视为单一压缩比参数，未揭示 W4A4 中权重与激活量化误差的不同行为——激活误差主导（ratio R>1）、但权重误差对 D 更敏感（γ_D: 0.1610 vs 0.0331）。
  4. **未识别激活瓶颈层**：Baseline 对激活量化误差的来源缺乏逐层分析，无法定位 FC2 Proj 输入（kurtosis=89）为根本瓶颈——这是 SwiGLU 输出的系统性 outlier 导致的特异性问题。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**统一的 QAT 缩放定律框架**，通过三个层次的递进分析解决 baseline 缺陷：

  **(1) 三维统一建模替代单参数建模（解决缺陷 1 和 2）**
  不再修改 N 的有效参数值，而是直接建模量化误差项 δ_p(N, D, G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}，作为 Chinchilla loss 的独立加项。log₂(G) 满足 G=1 时 δ_p=0（无量化）的边界条件。拟合 80 次 W4A4 QAT 实验数据，单条曲线即可覆盖所有 G（vs baseline 需 5 条曲线），且 W4A16/W4A4 预测相对误差分别从 19.3%/8.5% 降至 5.2%/4.7%。

  **(2) 误差解耦：权重 vs 激活独立分析（解决缺陷 3）**
  通过训练 W4A16（仅权重量化）和 W16A4（仅激活量化）两种额外配置解耦误差源，发现 δ_{W4A4} ≈ 0.906 · (δ_{W4A16} + δ_{W16A4})（强相关性）。通过分别拟合 δ_{W4A16} 和 δ_{W16A4} 的缩放定律参数，揭示：
  - 权重量化误差对 D 更敏感（γ_D=0.1610），更多训练数据时需重点优化权重
  - 激活量化误差对 G 更敏感（γ_G=0.9812），粗粒度下需重点优化激活
  - 激活量化误差始终大于权重（ratio R>1），但随 D/N 增大差距缩小

  **(3) FC2 瓶颈识别与混合精度方案（解决缺陷 4）**
  逐层分析 kurtosis（峰度）揭示 FC2 Proj 输入层的 kurtosis=89，远高于 QKV Proj、O Proj、FC1 Proj 等其他层（均 <10）。根源：FC2 输入来自 SwiGLU 非线性输出（gating + SiLU + element-wise multiply），复合非线性运算产生系统性 outlier，即使 QAT 正则化也无法完全消除。方案：FC2 Proj 输入保持 8-bit 量化，其余保持 4-bit。效果：量化误差降 20.5%（G=32）至 42.9%（G=256）；激活误差对 G 的敏感度 γ_G 从 0.9812 降至 0.4471；δ_{W16A4} 与 δ_{W4A16} 的 ratio R 降至 0.85-1.10，两者贡献趋于均衡。

  论文方法全栈执行例子（W4A4 QAT, 595M Llama3-style model, 100B tokens, G=128）：
  - 算法pipeline：
    1. BF16 基线训练：用 OLMo2-Mix-1124 全精度训练 → 记录 L_bf16
    2. W4A4 QAT 训练：插入 AbsMax per-group 量化器（weight: AbsMax, activation: AbsMax for G<256 or LAC for G≥256）→ STE 前向 + 反向 → 记录 L_W4A4
    3. 误差分解训练：W4A16 (weight-only) 和 W16A4 (activation-only) 分别训练
    4. 缩放定律拟合：用 80 次实验数据通过 Huber loss + L-BFGS 拟合式 5 的参数 k, γ_N, γ_D, γ_G
    5. FC2 Proj 分析：统计每层 kurtosis → 识别 FC2 Proj input → 实施 8-bit FC2 Proj + 4-bit others 混合精度 QAT → 重新拟合缩放定律
    6. 外推验证：973M 模型 100B/200B tokens 预测 vs 实际误差
  - 系统框架：PyTorch + HuggingFace Transformers，基于 OLMo2 训练 pipeline。LR 实验证明 4-bit QAT 无需高于 FP 训练的 LR（量化误差在 [0.60, 0.65] 内几乎恒定），可直接复用全精度训练超参数。
  - 编译框架：论文未明确说明（标准 PyTorch eager mode，fake-quantization 推理模拟）。
  - kernel调度：论文未明确说明（无自定义 kernel，使用标准 CUDA fake-quantize 模拟 INT4 GEMM）。
  - 硬件架构：NVIDIA A100 GPU。总计 268 次实验消耗 276K GPU-hours。

  全栈执行例子对比基线改进：
  - 量化误差预测：从 5 条独立曲线（每种 G 一条）→ 1 条统一曲线涵盖所有 G
  - 误差分解：从 unknowing（无法区分权重 vs 激活贡献）→ 明确 δ_{W16A4} > δ_{W4A16}（R>1），且提供 D/N-G 二维 heatmap 指导优化方向
  - FC2 瓶颈：从无法定位 → 明确 FC2 Proj input（SwiGLU 输出）为根本瓶颈，8-bit 处理可降误差 20-43%
  - EPM 量化：从常数值 → EPM(N, D, G) 动态值（式 13），W4A4 EPM 始终 >0.5（4-bit QAT trade-off 优于 8-bit QAT），FC2 8-bit 后提升 0.06-0.14

  关键设计动机映射：
  - 现有 QAT 缩放定律仅依赖 N → 直接建模 δ_p(N,D,G) 独立相加项，纳入 D 和 G
  - 均匀粒度建模（per-tensor）→ Group-wise 量化引入 log₂(G) 参数，统一拟合
  - 量化误差来源未知 → W4A16/W16A4 解耦训练 + 独立缩放定律拟合 + ratio R 分析
  - 无法定位激活瓶颈 → 逐层 kurtosis 分析 → FC2 Proj input（SwiGLU 输出 outlier）→ 混合精度
  - EPM 与 D/G 无关（baseline）→ 式 13 量化 EPM 随 N, D, G 演化，指导实际部署决策

## OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models

- baseline方法是什么？
  **Baseline 为使用手工设计的量化参数的后训练量化（PTQ）方法**：

  - **Weight-only quantization baseline（GPTQ/AWQ）**：GPTQ 使用逐层 block-wise 重建优化权重 round 方案（无需训练额外参数），AWQ 利用 grid-search 寻找最优 per-channel scaling 参数使重要权重获得更精确的量化。两者在低比特（W2A16）下急剧退化——GPTQ 在 LLaMA-13B W2A16 上 perplexity 从 5.09 飙升至 5500+，AWQ 在 group-wise 量化失效时退化为 e5 量级。
  
  - **Weight-activation quantization baseline（SmoothQuant/OS+）**：SmoothQuant 使用预定义的 migration strength（α）将激活量化难度迁移到权重上，OS+ 在此基础上加入 grid-search 的通道级 scaling 和预定义的 shifting。两者在 W4A4 下精度崩溃——LLaMA-7B 平均零样本准确率仅 38.41%（SmoothQuant）和 48.43%（OS+），远低于 FP16 的 64.09%。

  **Baseline 全栈执行例子（LLaMA-7B W4A4）**：
  - 算法pipeline：SmoothQuant 用固定 α 对 weights/activations 做 per-channel scaling → 进入 MinMax 量化器（per-channel weight, per-token activation）→ 手工参数（α=0.5 或 grid-search 最优值）无梯度反馈。
  - 系统框架：PyTorch + HuggingFace Transformers → HuggingFace `model.generate()` → 单卡 A100 GPU。量化在模型加载时一次性完成，无训练循环。
  - 编译框架：论文未明确说明（PyTorch eager mode，fake-quantization 推理模拟）。
  - kernel调度：论文未明确说明（无自定义 kernel，使用标准 CUDA fake-quantize 模拟低比特 GEMM）。
  - 硬件架构：NVIDIA A100 GPU → CUDA core 执行 fp16 matmul + scale/dequant → 显存约 13GB（7B FP16），量化后 ~3.8GB（W4A16g128）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：OmniQuant 将量化参数（clip threshold、equivalent transform scale/shift）作为可学习变量，在冻结 FP16 权重的条件下，通过 block-wise 量化误差最小化（Eq.1：`argmin_Θ1,Θ2 ||F(W,X) - F(Q_w(W;Θ1,Θ2), Q_a(X,Θ2))||`）用 SGD 端到端优化，实现 PTQ 效率 + QAT 级别的性能。

  **具体设计如何解决 Baseline 缺陷**：
  1. **LWC 解决手工 weight clipping 次优问题**：Baseline 的 MinMax（γ=β=1）截断所有 outlier，网格搜索（AWQ）在连续空间中代价高且粗粒度。LWC 以 SGD 连续优化相对截断强度 γ,β ∈ [0,1]，能自适应学习最优截断阈值——Table A13 显示 LWC 将 W3A16 的 ||W-Wq|| 从 0.0062 降至 0.0044，||X-Xq|| 从 2.80 降至 1.05。与 PACT/LSQ 的关键区别：LWC 使用相对缩放（γ·max, β·min）而非绝对阈值，当 LET 每轮改变权重分布时仍稳定收敛（Figure A5 证明 PACT/LSQ 在分布变化时发散）。
  
  2. **LET 解决手工等效变换次优问题**：SmoothQuant 预定义 α（无法适配不同层/模型），OS+ 的 grid-search 在高维 joint space 中粗糙。LET 通过梯度下降在连续空间中 joint optimize 所有层的 scale/shift，同时扩展到 attention 的 Q/K 矩阵乘法（Eq.5）使 KV cache 也可量化。Figure A2 显示：原始激活 outlier 幅值约 70，SmoothQuant 后降至 2（仍有明显 gap），LET 后 outlier 与 regular channel 幅值几乎一致，说明 LET 比 SmoothQuant 更彻底地均衡了激活分布。

  3. **Block-wise 量化误差最小化框架**：Baseline PTQ（AdaRound/BRECQ）需要优化所有权重，在 LLM 上不可行。OmniQuant 仅优化少量可学习参数（每通道 2-3 个），使得 7B-70B 模型都可在单卡 A100-40G 上完成量化（7B W4A4 1.6h, 70B W4A4 ~16h），时间约为 GPTQ 的 5×，但远低于 QAT 的数百 GPU-hours。

  4. **LWC + LET 协同效应**：LET 将激活 outlier 迁移到权重上加重了 weight quantization 难度 → LWC 恰好专门处理 weight quantization → 两者形成"LET 迁移难度 → LWC 消解难度"的递进关系。Table A2 消融：LET alone (16.97 PPL) < LET + grid-searched WC (15.82) < SmoothQuant + LWC (15.80) < LET + LWC 联合训练 (12.87)，证明了 differentiable joint optimization 的关键性。

  **论文方法全栈执行例子（LLaMA-7B W4A4）**：
  - 算法pipeline：加载 FP16 权重 → for each block: 初始化 LET (s=SmoothQuant, δ=OS+) 和 LWC (γ=β=1) → 20 epochs AdamW 优化 → Eq.(3) 计算 tilde_X = (X-δ)⊘s, tilde_W = s⊙W → Eq.(2) 以学习的 γ,β 计算 h 并量化 tilde_W → 计算 MSE loss → backward 更新 Θ1,Θ2 → 收敛后 fuse s 入权重（tilde_W → W_int）、δ 入 bias → 最终 INT4 权重矩阵 + INT4 激活 → 推理时无额外参数/计算。
  - 系统框架：PyTorch + HuggingFace Transformers → 单卡 A100-40G（量化校准）→ MLC-LLM（部署推理，A100-80G）→ 量化后 W4A16g128 LLaMA-7B weight memory 3.8GB，token/s=134.2。
  - 编译框架：论文未明确说明（校准阶段 PyTorch eager mode fake-quantization → 部署阶段 MLC-LLM 的 INT4 CUDA kernel）。
  - kernel调度：论文未明确说明（MLC-LLM 提供 INT4 GEMM kernel，OmniQuant 的均匀 INT 量化可直接对接无需自定义 kernel）。
  - 硬件架构：NVIDIA A100 GPU → CUDA Tensor Core → 量化后 LLaMA-7B running memory 5.7GB（vs FP16 约 13GB），W2A16g128 下 token/s=83.9（因 MLC-LLM 对 INT2/INT3 支持欠优化，实际潜力更高）。

  **核心优势**：可微分量化的灵活性（逼近 QAT 性能）+ PTQ 的效率（128 样本、单 GPU、1-16 小时）+ 均匀 INT 量化的硬件友好性（可直接部署，无需混合精度或非均匀量化）。

## PB-LLM Partially Binarized Large Language Models

- baseline方法是什么？
  Baseline 是将已有的网络二值化方法（BNN, XNOR, Bi-Real, ReCU, FDA）直接应用于 LLM 量化的方案，以及传统的 uniform quantization 方法（RTN）。全栈执行例子：
  - **算法pipeline**：已有的 binarization 方法（如 XNOR-Net）使用 sign 函数将所有权重二值化为 ±1，乘以 channel-wise scaling factor（L1 norm 平均）；或 RTN 直接 round-to-nearest 量化到目标 bit-width。但这些方法在 LLM 上完全崩溃——BNN/XOR/Bi-Real/ReCU/FDA 二值化后的 OPT-1.3B 在 7 个零样本常识推理任务上的平均准确率（0.30-0.32）低于随机猜测（0.36）。原因是 LLM 中存在少量对模型容量至关重要的 salient weights（显著权重），全部二值化会导致这些关键权重的信息完全丢失。已有的 LLM 量化方法（如 GPTQ）在 4-bit 以下也出现显著的性能退化。
  - **系统框架**：论文未明确说明。baseline 使用标准 PyTorch 训练流程，无特殊的分布式或 serving 框架修改。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：论文未明确说明。理论上一值化权重可将 FP 乘法替换为 bitwise XNOR+Bitcount 操作，但论文主要关注 memory 压缩（memory-bound LLM inference）而非 compute kernel 加速。
  - **硬件架构**：论文未明确说明。使用标准 GPU 训练和推理。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  PB-LLM 提出**部分二值化**（Partially-Binarized LLM）策略，核心思想是识别并保留少量 salient weights 在高位宽，其余权重二值化。这源自一个关键发现：LLM 中存在少量显著权重对容量至关重要，全部二值化会完全丢失这些信息。PB-LLM 在 PTQ 和 QAT 两种框架下分别实现了这一思路。

  **具体设计如何解决 Baseline 缺陷**：

  1. **Salient Weight Detection + 保留（解决全部二值化崩溃问题）**：Baseline 的所有权重一视同仁地二值化，导致 LLM 完全崩溃（< random guess）。PB-LLM 通过 magnitude（QAT）或 Hessian metric v_i = w_i^2/[H^{-1}]_{ii}^2（PTQ）检测权重矩阵中的 salient weights，保留 5%-30% 为高比特（如 INT8），剩余才二值化为 ±1。图 6 显示即使 50% salient（等效 ~5-bit）且无训练的 OPT-1.3B 仍有 PPL ~20（非崩溃），证明了 salient weights 对 LLM 容量的关键性。

  2. **PB-GPTQ 解决 PTQ 中的二值化误差传播问题**：Baseline RTN 直接逐列二值化/量化，量化误差在列间累积导致最终输出严重偏离。PB-GPTQ 将 GPTQ 的 Hessian 引导误差补偿扩展到部分二值化：每量化一列后，将该列的量化误差通过 Hessian 矩阵加权补偿到剩余未量化列，使得后续列的量化可以在"误差已校正"的权重基础上进行。Table 1 显示 PB-GPTQ 相比 RTN 在 10% salient 时将 PPL 从 4889 降至 895（Magnitude），从 7508 降至 165（Hessian）。

  3. **Salient Weights Frozen 解决训练困难问题**：LLM QAT（如 LLM-QAT）即使只做 4-bit 量化也需要 100K iterations。PB-LLM 的 QAT 通过冻结 salient weights（不参与梯度更新），仅优化 binary weights 的 FP latent，将训练迭代数从 100K 降至 1-10K（图 7 上半部分：30% salient，10K iters 即可恢复性能）。图 5 训练曲线显示冻结 2% salient 权重就能显著加速收敛。

  4. **Optimal Scaling Factor 闭式解解决手工/搜索 scaling factor 次优问题**：Baseline XNOR-Net 的 L1 norm scaling 和 AWQ 的 grid search 分别有近似误差和搜索成本问题。PB-LLM 从 L2 误差最小化出发解析推导 α* = ||w_F||_1/n（当 w̄_B = sign(w_F) 时），无需任何搜索，且在 column-wise 粒度上做到最优。反直觉的是，仅凭 Salient Frozen + Optimal Scaling 两个机制直接应用于未训练的 LLM 就能维持一定语言能力（图 6）。

  **论文方法全栈执行例子（LLaMA-7B QAT，10% salient，等效 ~1.7 bit）**：
  - **算法pipeline**：加载 LLaMA-7B FP16 checkpoint → 对每个 Linear 层按 |W| 排序选 top-10% salient weights → freeze salient weights（INT8 MinMax quantize）→ 剩余 90% 权重：正向 sign(W_F^{unsal}) 二值化 + α* = mean(|W_F^{unsal}|) column-wise scaling → STE 反向传播更新 FP latent W_F^{unsal} → AdamW, lr=2e-5, cosine decay, 10K iters, 每个 GPU batch=1 → 训练数据 RedPajama-simple-1B → 最终得到 partially-binarized LLaMA-7B → 推理时存储 W^{sal}(INT8) + sign(W_F^{unsal})(binary) + α* scaling factors + bitmap index → 总存储 ≤ 1 * 0.9 + 8 * 0.1 + 1 ≈ 2.7 bit/weight。
  - **系统框架**：PyTorch + HuggingFace Transformers → 标准 GPU 训练（论文未明确 GPU 型号）→ `model.generate()` 推理。与标准 LLM 推理流程一致，仅权重矩阵从 FP16 替换为 mixed-precision（INT8 salient + binary unsalient + scaling factors）。
  - **编译框架**：论文未明确说明。部分二值化矩阵可受益于 bitwise XNOR+Bitcount kernel 加速（理论 64x vs FP multiply），但论文未实现此优化。
  - **kernel调度**：论文未明确说明。论文主要聚焦 memory 压缩（binary weights 在显存中占用极少），不涉及自定义 kernel 实现。
  - **硬件架构**：论文未明确说明具体 GPU 型号。推理时 GPU 需将 binary weights 和 INT8 salient weights 反量化为 FP16 进行矩阵乘法（或未来利用 bitwise 操作加速）。

  **核心优势**：将 LLM 量化推至接近 1-bit（部分二值化）+ PTQ/QAT 双框架灵活选择 + Salient Freeze + Optimal Scaling 闭式解双机制加速训练 + 解析最优而非搜索 scaling factor + 训练效率远超已有 LLM QAT 方法。

## PM-KVQ Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

- baseline方法是什么？
  - **KIVI**（代表主流 post-training KV Cache 量化方法）：对 Key Cache 做 per-channel 量化、Value Cache 做 per-token 量化，使用 group-wise 非对称量化，保留首 token 和最近窗口内 token 的 FP16 全精度。在 long-CoT 场景下的两个核心缺陷：
    - **(1) 大累积误差**：KIVI 在每次解码步直接把新生成的 KV Cache 量化为目标位宽（如 2-bit），且采用 uniform bit-width 分配。如图 1(a) 左所示，在生成初期显存大量闲置（因为 token 数远未达到最大上下文长度），但这些本可用来以高精度存储早期 token 的显存被浪费了。在 32K 上下文 long-CoT 推理过程中，每个解码步的量化误差累积，导致随 token 增多推理质量急剧下降。
    - **(2) 短上下文校准无法反映长上下文数据分布**：RoPE 将位置信息通过不同频率的正弦/余弦注入 Key Cache 各通道，低频通道（如 DeepSeek-R1-Distill-Qwen-7B 的最低频通道周期达 54410 tokens）在短校准数据（512 tokens）下只能观察一小段正弦曲线，无法获得准确的 channel-wise reparameterization factor λ_i = (max_m K_{m,i})^α，导致 outlier channel 被错误平滑。
  - 全栈执行例子（以 KIVI 执行 2-bit DeepSeek-R1-Distill-Qwen-7B long-CoT 推理为例）：
    - **算法层**：prefill 阶段计算 K = X·W_K, V = X·W_V → Key Cache 做 per-channel group-wise 量化（G=128）→ Value Cache 做 per-token group-wise 量化 → 首 token INT16 保留 → 每个 decoding step：新 token 直接量化为 2-bit → 最近 128 token 保留 INT16 → 注意力计算时反量化到 FP16 做 softmax attention → 输出。全程 uniform 2-bit，显存利用率低。
    - **系统框架层**：论文未明确说明 serving 框架。评测使用 HuggingFace Transformers + fake quantization（不实际节省显存，仅模拟量化误差），在 8×A100-80G 服务器上运行。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - PM-KVQ 通过三项创新解决 baseline 累积误差和校准偏差问题：
    - **(1) 渐进量化（Progressive Quantization）→ 解决累积误差**：不再在每次解码步直接量化到目标 Fbit，而是先从 16-bit 开始存储 KV Cache，当显存预算被占满后，通过"等价右移"逐步将已存储的 KV Cache 位宽从 16→8→4→2 缩减（`X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b`），为新 token 腾出空间。在 32K 上下文的例子中（Fbit=2），第 1-2K token 以 16-bit 存储（零量化误差），第 2K-4K token 以 8-bit 存储（极低误差），以此类推。只有最后一部分 token 以 2-bit 存储。这使得 long-CoT 推理过程的前期（最多 token）享受低误差，后期才承受高误差——远优于 KIVI 的全程 2-bit。
    - **(2) 块级内存分配（Block-wise Memory Allocation）→ 解决 uniform bit-width 浪费**：用一阶泰勒近似估计每个 transformer block 的 KV Cache 敏感度 `s_{i,b}`，将位宽分配形式化为 Integer Programming 问题：`min Σ_i Σ_b x_{i,b}·s_{i,b}` s.t. `Σ_i Σ_b x_{i,b}·(Mem(Q_b(K_i)) + Mem(Q_b(V_i))) ≤ M`。CVXPY 在数秒内求解，为深层 block（更敏感）分配更高 Fbit。当 batch size 从 40 减少到 32（单样本显存更多但仍不足以统一升到 4-bit）时，PM-KVQ 将多余显存分配给敏感 block，额外提升 0.84%。
    - **(3) 位置插值校准（Calibration with Positional Interpolation）→ 解决短校准数据偏差**：在 RoPE 旋转矩阵中引入位置缩放因子 s：`cos(s·mθ_i)`，使 2048 token 的校准数据模拟长上下文（s=4 → 有效 8192 token）的位置分布。如图 1(c) 底所示，低频通道的完整正弦周期得以在短校准数据中展现，λ_i 校准更准确。消融实验：2048 token + s=4 的 pass@1（48.33%）与直接使用 8192 token 校准（48.33%）持平，远超无插值 baseline（46.67%）。
  - 论文方法全栈执行例子（以 PM-KVQ 执行 2-bit DeepSeek-R1-Distill-Qwen-7B long-CoT 推理，batch=40，Fbit=2 为例）：
    - **算法层**：
      - 离线阶段：加载校准数据（512 seqs × 2048 tokens，arXiv RedPajama）→ 逐 block 计算 s_{i,2} 和 s_{i,4} → CVXPY 求解 ILP 得到每个 block 的 Fbit（如 block 1 最敏感→4-bit，block 28 最敏感→4-bit，其他→2-bit）→ 位置插值校准 position scaling s=4（有效 8192）→ 计算 channel-wise reparameterization λ_i → 应用式(9)将 Key Cache outlier 迁移到 Query。
      - 推理阶段：prefill 计算 K, V → 首 token INT16 → 渐进量化循环：t=1..2048 以 16-bit 存储（显存未满）→ t=2049 触发 16→8 bit 缩减（等价右移，更新 S 和 Z）→ 腾出空间继续 8-bit 存储 → 类似地 8→4→2 逐步缩减直到 32K 上下文 → 注意力计算时混合精度：首 token INT16 + 滑动窗口 128 token INT16 + 渐进量化部分按各自当前位宽反量化到 FP16 做 attention。
    - **系统框架层**：8×A100-80G GPU 服务器，HuggingFace Transformers，fake quantization 评测。论文明确声明未与系统级优化和推理引擎结合（Limitations 章节）。
    - **编译框架/kernel调度/硬件架构层**：论文未明确说明。PM-KVQ 是纯算法方法，位宽缩减通过整数移位操作实现，不涉及自定义 CUDA kernel 或硬件修改。

  **所有三个技术的协同关系**：渐进量化解决单体请求内的累积误差（时间维度）；块级内存分配解决跨层的显存分配优化（空间维度）；位置插值校准解决校准阶段的值域估计准确性（数据维度）。三者正交且互补，共同将 long-CoT LLM 的 2-bit KV Cache 推理从接近随机（RotateKV/MiKV pass@1≈0%）提升到接近 FP16 水平（差距 < 5%）。

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

## PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models

- baseline方法是什么？
  Baseline 是已有的通用 PTQ 量化方法，包括：(1) 训练无关 scaling 方法——**SmoothQuant**（per-channel 平均对齐激活和权重 range）、**OS+**（对齐所有激活通道到共同中心 + 缩放）、**RepQ***（重参数化统一激活 range）；(2) 训练依赖方法——**OmniQuant**（反向传播优化 scaling factor 和 weight clipping，需数小时训练）；(3) 旋转变换方法——**QuaRot**（随机 Hadamard 旋转抑制 outlier，在 ARVG 中因 AdaLN 不保持旋转不变性需在线计算，引入严重开销）；(4) 低秩分解方法——**SVDQuant**（低秩分解隔离 outlier，需自定义 CUDA kernel，在 ARVG 中效果不佳）。

  这些 baseline 在 ARVG 模型上存在三个核心缺陷：
  (a) **无法处理 channel-wise outlier**：ARVG 中 AdaLN 模块调整后的激活存在严重的 channel-wise outlier（激活 range 跨通道差异极大），SmoothQuant/OS+ 等经验设计的 scaling 方法缺乏理论保证，次优且无法保证有效性；OmniQuant 需昂贵训练且不稳定。
  (b) **无法高效处理 token-wise 动态激活**：ARVG 中 AdaLN 输入沿 token 维度高度动态（含位置嵌入信息），线性层存在 sink token（首 token 含条件信息，分布显著不同于其他 token）。LLM 的动态 per-token 量化（如 LLM.int8）引入在线 min-max 校准开销（0.5× speedup loss）且精度下降（VAR 上 FID 降 15.3）。
  (c) **样本间分布不匹配导致校准偏差**：ARVG 中网络激活跨样本高度相似（尤其无条件样本），样本级冗余导致量化参数校准不匹配。现有校准策略（如 EDA-DM 的时序校准）针对扩散模型的时间步维度，无法处理 ARVG 的样本级冗余。

  Baseline 全栈执行例子（以 SmoothQuant 量化 RAR-B 到 W6A6 为例）：
  - **算法层**：加载 RAR-B 预训练权重 → 对 qkv 和 fc1 层计算 per-channel 激活和权重 range → 使用默认平滑因子 α=0.5 做等效缩放：激活除以 s_i = max(|X_i|)^α / max(|W_i|)^{1-α}，权重乘以 s_i → 缩放因子融合到 AdaLN 权重中（离线）→ 校准 128 张随机采样 ImageNet 图像，layer-wise min-max 确定激活量化 range，channel-wise min-max 确定权重量化 range → uniform quantizer：W_int = round(W/δ_W) + z_W, X_int = round(X/δ_X) + z_X → 推理时 INT 矩阵乘法 + 反量化。
  - **系统框架层**：PyTorch fake quantization，GPU 上完成校准（论文未明确指定 GPU 型号，baseline 中 OmniQuant 用 A100-80G）。使用 ADM's TensorFlow evaluation suite 做 FID/IS 评估。生成 50K ImageNet 图像。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。使用标准 PyTorch fake quantization，无自定义 CUDA kernel。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **PTQ4ARVG**，一个 training-free 且 hardware-friendly 的 PTQ 框架，通过三个组件分别解决 baseline 的三个缺陷：

  **(1) GPS (Gain-Projected Scaling) → 解决 channel-wise outlier**：不同于 SmoothQuant 等经验设计的 scaling，GPS 首次基于数学优化推导 scaling factor。将量化损失做 Taylor 展开，定义 scaling gain g(s) = g_x − g_W（激活量化损失减少 − 权重量化损失增加），通过 ∂g/∂s = 0 求得闭式最优解 s_i = s_k · √(Σ|ΔW_{i,j}·x_i|) / √(Σ|W_{i,j}·Δx_i|)。无需训练（vs OmniQuant 需数小时训练），无在线计算开销（vs QuaRot 的在线 Hadamard 旋转），零推理开销（scaling factor 离线融合到权重中）。

  **(2) STWQ (Static Token-Wise Quantization) → 解决 token-wise 动态激活**：利用 ARVG 的两大独有特性——固定 token 序列长度和跨样本位置不变分布——将 per-token 量化参数离线静态设定。对 AdaLN 沿 token 序列分配独立量化参数；对线性层将 sink token 与 normal token 分开量化。使用 percentile 校准保证精度。相比 LLM 的动态 token-wise 量化（LLM.int8），STWQ 无在线校准开销（speedup 保持 2.92× vs DTWQ 的 2.46×），且精度更高（FID 10.41 vs DTWQ 30.14）。

  **(3) DGC (Distribution-Guided Calibration) → 解决样本间分布不匹配**：基于 Mahalanobis 距离 ρ(x) = √((x-u)^T S^{-1} (x-u)) 量化每样本对整体分布熵的贡献，选择 top 50% 高熵样本构成校准集。相比 random/uniform 采样，DGC 在所有指标上一致提升，且随校准集增大保持鲁棒。

  PTQ4ARVG 全栈执行例子（以 RAR-B W6A6 量化为例）：
  - **算法层**：
    - 离线校准阶段：加载 RAR-B 预训练权重 → DGC：从校准池计算 Mahalanobis 距离，选 top 50% 高熵样本（128 张 ImageNet）→ GPS：对每个 block 的 qkv 和 fc1 层，量化当前权重和激活计算 ΔW 和 ΔX → 找到 activation range 最大的通道 k → s_k = √(R_x^k/R_W^k) → 对每个通道 i≠k，闭式求解 s_i = s_k·√(Σ|ΔW_{i,j}·x_i|)/√(Σ|W_{i,j}·Δx_i|) → 应用等效缩放 X'=X⊘s, W'=s⊙W，离线融合到 AdaLN 权重 → STWQ：对 AdaLN 输入逐 token 做 percentile 校准设定 δ[t]；对线性层输入分离首 token（sink）和其余 token，分别 percentile 校准 → 存储所有静态量化参数（δ, z, bit-width）。
    - 推理阶段：输入条件信息（类别标签 + 位置编码）→ AdaLN 生成 shift/scale 参数 → 每步 token 生成：使用预设的静态 per-token 量化参数对激活做 INT 量化 → INT 矩阵乘法（权重已事先量化并融合 scaling）→ 反量化 → 输出 token → 最终生成 50K ImageNet 图像评估 FID/IS/Precision。
  - **系统框架层**：PyTorch fake quantization 用于校准和精度评估；RTX 3090 GPU 实际部署 8-bit 量化模型测试延迟/内存。标准 CUDA kernel，无自定义 kernel（与 SVDQuant 不同）。使用 ADM's TensorFlow guided-diffusion 评估套件。128 张校准图像（DGC 选择），50K 生成图像评估。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。PTQ4ARVG 使用标准 CUDA kernel 部署，无自定义 kernel 或硬件修改。PTQ4ARVG 的 STWQ 兼容标准 CUDA kernel（论文明确论证了这一点）。

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

## QTIP: Quantization with Trellises and Incoherence Processing

- baseline方法是什么？
  - Baseline 是当前 SOTA 的 VQ-based LLM PTQ 方法 **QuIP#** 和 **AQLM**。它们使用 Vector Quantization (VQ) 将 d 维向量量化到 2^{kd} 大小的 codebook。由于 codebook 大小随维度指数增长，这些方法被硬件限制在 d≤8 维度：AQLM 使用 8D codebook (1MiB, 无法放入 L1 cache)，QuIP# 使用 8D E8 格点 codebook（高对称性可压缩 256×, 勉强放入 L1 cache）。低维度限制了 VQ 的 shaping/packing 优势，导致量化失真较高。
  - Baseline 全栈执行例子（QuIP# 2-bit 量化 Llama 2 7B 推理一个 token）：
    - **算法层**：RHT 使权重近似 i.i.d. 高斯 → BlockLDLQ 逐块量化（group size g, 8D VQ 每 8 个权重选 E8 格点最近邻） → 每 8 维存储 kd=2×8=16 bits → 反量化时查 8D codebook 恢复 FP16 权重。
    - **系统框架层**：PyTorch + 自定义 CUDA kernel。QuIP# 的 E8 codebook 压缩 256× 后可放入 L1 cache (约 8Kb)，通过查表实现快速反量化。
    - **编译框架/kernel调度层**：自定义 CUDA kernel 实现 on-the-fly E8 格点反量化 + FP16 GEMV。由于 d=8 维度过小，VQ 的 shaping 优势未充分发挥。
    - **硬件架构层**：NVIDIA GPU（RTX 6000 Ada 等），E8 codebook 查找在 GPU L1 cache 中完成。
  - Baseline 核心缺陷：VQ 维度被硬件 codebook 缓存大小限制在 ≤8，而信息论表明更高维度可显著降低量化失真（256D TCQ MSE 0.069 vs 8D VQ 0.089 vs D_R=0.063）。codebook 大小 O(2^{kd} d) 的指数增长在硬件上不可行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - QTIP 用 **Trellis Coded Quantization (TCQ)** 替代 VQ，利用 TCQ 的 **线性复杂度（O(2^L T)）** 实现超高维量化（有效维度 256），同时设计硬件高效的 bitshift trellis + compute-based codes 消除 TCQ 的推理开销。三个对应设计：
    1. **Bitshift Trellis → 解决 TCQ 的顺序解码和 trellis 存储问题**：bitshift trellis 中节点 i→j 有边当且仅当 j=(i·2^{kV} mod 2^L)+c，第 t 组权重仅依赖连续 L-bit 窗口，解码时仅需 kV-bit 位移（硬件原生支持）且完全并行化，无需存储 trellis 图结构（对比 naive TCQ 需存储 2^L×2^{kV} 边信息）。
    2. **Compute-based Codes → 解决 codebook 存储问题**：1MAD/3INST/HYB 码均为 lookup-free 或小 LUT 设计，在 GPU 上仅需 ≤4 指令/权重即时生成伪随机高斯值。HYB codebook 仅 2KiB（2^9×2 FP16），比 AQLM 的 1MiB 小 512 倍，可完全放入 L1 cache。这消除了 TCQ 需要存储 2^L×V 大小 codebook 的瓶颈。
    3. **RHT Incoherence Processing → 使权重适合 TCQ**：RHT 将 LLM 权重转化为近似 i.i.d. 高斯分布，而 TCQ 对 i.i.d. 高斯源天然高效（256D TCQ MSE 0.069 接近 D_R=0.063）。
  - QTIP 全栈执行例子（Llama 2 7B 2-bit HYB 码推理一个 token，L=16, V=2, Tx=Ty=16, Q=9）：
    - **算法层**：离线——RHT 变换 W̃ ← V_m S_m W S_n V_n^T → BlockLDLQ 逐块量化：每 Tx×Ty=16×16=256 维序列用 Viterbi 算法在 (L=16, k=2, V=2) bitshift trellis 上最小化 MSE 失真（O(2^16 × 256) ≈ O(1.7M) 操作/序列） → 输出每 256 维权重的 kT=2×16×16=512 bits 编码。在线——从 packed bitstream 读取 512-bit 块 → 通过 bitshift 操作逐 2D 向量提取 16-bit 状态 → HYB code 即时解码为 FP16 权重对（hash → LUT lookup → sign-flip, 摊销 2 指令/权重）。
    - **系统框架层**：PyTorch + QuIP# BlockLDLQ 框架 + 自研 CUDA kernel。QTIP 作为 BlockLDLQ 中 VQ 的 drop-in 替换（Algorithm 5），g=Ty 但有效维度 TxTy=256 >> g。HYB codebook 常驻 GPU L1 cache (2KiB, 可 32× 复制消除 bank conflicts)。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：自定义 CUDA kernel：bitshift trellis 解码（每步 kV=4 bit 位移）→ 1MAD/3INST/HYB code 即时生成高斯权重 → 16×16 MMA tile 矩阵-向量乘（Tensor Core）。支持 tail-biting（Algorithm 4）对齐 32-bit word。解码达 >80% 峰值显存带宽。1MAD: 2 GPU instr (MAD+vabsdiff4)、3INST: 3 instr (MAD+lop3+FADD)、HYB: 摊销 2 instr。
    - **硬件架构层**：NVIDIA GPU (RTX 6000 Ada, 3090, A6000 Ampere)。利用 16×16 MMA tile 和 L1 cache。ARMv8 CPU 也可用 NEON vqtbl4q_u8 查表实现 6-bit 1D HYB code（Q=6, V=1），达到与 3INST 相当的质量。
  - 效果：QTIP 256D TCQ 2-bit 量化 i.i.d. 高斯源 MSE 0.069（vs QuIP# 8D VQ 0.089, 改善 22%）；Llama 2 70B 2-bit perplexity gap 约减半；端到端推理速度与 QuIP# 持平（188 vs 186 tok/s, ≤4 instr/weight），同时量化质量更高。代码开源：https://github.com/Cornell-RelaxML/qtip

## QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

- baseline方法是什么？
  - Baseline 方法分为两类：(1) LoRA-based QA-PEFT（以 CLoQ 为代表）：使用 SVD 分解量化误差 ΔW_Q，用低秩矩阵 BA 近似补偿，初始化时最小化层输出误差。适配器参数限制在 rank-r 子空间内，表示能力有限（normalized rank < 6.3%）。(2) FT-based 适配器（LoCA/DCA, SSH/DHA）：将权重更新表示为 ΔW = H'^{-1} F H^{-1}（双变换），F 为稀疏系数矩阵。初始化时参数位置 E 随机选择或部分随机+部分幅值选择（SSH），系数 c 初始化为零，不做量化误差的显式补偿。在 QA-PEFT 场景下，这类方法缺乏量化感知初始化，往往表现不如 LoRA-based 方法。
  - 全栈执行例子（Baseline: CLoQ on LLaMA-3.2-3B, 4-bit, P(r=64)）：
    - **算法pipeline**：GPTQ + MagR 4-bit 量化 W_0 → 收集 WikiText-2 校准集激活 X → 计算 R = (XX^T)^{1/2} → 最小化 ||ΔW_Q R - BA R||_F^2 初始化 A, B（低秩近似）→ Alpaca 数据集 fine-tuning 3 epoch → 推理时 y = (W_Q + BA)x。缺陷：(i) LoRA 的 rank ≤ r，对复杂量化误差模式（尤其是异常值）重建能力有限；(ii) 低秩结构限制 fine-tuning 的表示能力，增加参数 budget P(r) 也无法缩小与 WHA 的 gap（Figure 6 显示 QWHA P(r>32) 已超越 CLoQ 最大评分）。
    - **系统框架**：PyTorch + HuggingFace Transformers + PEFT，AdamW optimizer，cosine LR scheduler。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 PyTorch CUDA kernel（矩阵乘法 forward 和 LoRA 低秩分解）。推理吞吐 188.1 tok/s。
    - **硬件架构**：NVIDIA A100 80GB GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QWHA 提出三大创新组件解决 baseline 缺陷：
    (1) **WHA (WHT-based Adapter)**：将权重更新定义为 ΔW = F H^{-1}（仅单变换），H 为 WHT 矩阵（±1 元素）。WHT 基函数为方形波（sharp transitions），相比 DCT/DHT 的正弦基（smooth transitions），天然更适合捕获量化误差的异常值结构。理论分析：WHT 系数的 Pareto hill index η 最小（能量分布最陡），即 WHT 将最大比例的误差能量集中在最少系数中，使得稀疏适配器可以用少量参数高效补偿量化误差。实验验证 WHA 捕获最多异常值系数（avg 18.12% vs DCA 7.23%/DHA 17.06%）。
    (2) **AdaAlloc**：通道级自适应参数分配 p_i ∝ ||(ΔW_Q X)_{i,:}||_F^t，高误差输出通道获得更多参数，同时保证每个通道 ≥2 参数以维持 full rank（满足 Coja-Oghlan et al. 的稀疏随机矩阵 full-rank 条件）。在每个通道内，选取 |(ΔW_Q H)_{i,j}| 最大的 p_i 个系数位置。对比：纯幅值选择（Magnitude）过度集中参数于少数通道，导致 low-rank F 和 fine-tuning 能力下降（rank 接近 0）；随机选择（Random/LoCA/SSH）虽保持 high rank，但初始化误差大（Table 2: avg error 5.96/4.57 vs AdaAlloc 3.86）。AdaAlloc 是唯一同时实现 high rank 和 low init error 的策略（Figure 4 + Table 2）。
    (3) **Refinement**：对已选参数位置通过 v B'^T (B' B'^T)^{-1} 重新投影，使选中 basis vectors 的线性组合能补偿未选中向量。无 Refinement 时系数直接取自稠密解，忽略列间相关性，层输出误差仅略微降低（avg 7.21→7.06）；加入 Refinement 后层输出误差大幅下降（avg 7.21→3.86，约 46.5% 降幅）。
  - 全栈执行例子（QWHA on LLaMA-3.2-3B, 4-bit, P(r=64)）：
    - **算法pipeline**：
      1. GPTQ + MagR 量化 W_0 → W_Q（4-bit, group size 64）
      2. WikiText-2 128 条序列前向收集激活 X → 计算 R = UΣ^{1/2}, B = H^{-1}R
      3. AdaAlloc：p_i = floor(p × ||(ΔW_Q X)_{i,:}||_F / Σ||(ΔW_Q X)_{j,:}||_F)，每通道 ≥2
      4. Per-channel：v = (ΔW_Q)_{i,:} R → 选最大 |(ΔW_Q H)_{i,j}| 的 p_i 个位置 → E
      5. Refinement：B' = B[E], c = v B'^T (B' B'^T)^{-1}
      6. Alpaca fine-tuning：Y = (W_Q + α ΔW) X，ΔW = Scatter(c, E) H^{-1}
      7. 推理：WHT 通过 fast Hadamard kernel（仅加减法，O(n log n)），184.6 tok/s
    - **系统框架**：PyTorch + fast-hadamard-transform (Dao-AILab) + GPTQ。AdamW, lr=3e-5 (4-bit LLaMA-3.2-3B Alpaca)。训练时间 6.0h (batch=4)，远快于 LoCA (30.1h, DCA 双变换) 和 SSH (26.1h, DHT 双变换)，与 CLoQ (5.0h) 相当。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：fast Hadamard kernel（Dao-AILab）：无需显式矩阵构造，通过递归 fused kernel 实现 H^{-1} X 仅用加法和减法。WHT 比 DCT/DHT 更快因其无需复数运算。1D WHT vs 2D WHT 训练时间：batch=1 时 18.2h vs 25.3h，batch=4 时 6.0h vs 8.0h。
    - **硬件架构**：NVIDIA A100 80GB GPU。推理显存 QWHA 52.68GB vs CLoQ 59.53GB（减少 13.0%），因稀疏适配器 scatter ops 无额外内存开销，fast Hadamard kernel 无矩阵乘法。
  - **Baseline 缺陷 → 方法设计映射**：
    - (i) LoRA 低秩限制（rank ≤ r << d_min）→ WHA 的 full-rank 适配器（rank ≈ d_min），稀疏矩阵 F 的每行/每列 ≥2 非零元即保证 full rank，P(r≥4) 下 100% 满足条件。Figure 6 验证增加 LoRA rank 无法追上 WHA。
    - (ii) LoRA 低秩结构对量化误差异常值重建不足 → WHT 的方形波基函数在频域中天然适合表示突变/尖峰（异常值），Pareto hill index η 最小使 WHT 系数能量最集中，稀疏适配器用等量参数捕获更多误差能量（Figure 2(b), Figure 3）。
    - (iii) FT-based adapters 无量化感知初始化（参数随机/零初始化），导致 QA-PEFT 表现差甚至不如 LoRA → AdaAlloc + Refinement 实现误差驱动的参数位置选择和值优化，直接最小化层输出误差 ||ΔW_Q R - F H^{-1} R||_F^2。
    - (iv) 传统幅值选择（Magnitude）导致 low-rank F，随机选择初始化误差大 → AdaAlloc 通道级分配保证 full rank + 通道内幅值选择最小化误差，同时兼顾 fine-tuning 能力和初始化质量。
    - (v) 双变换（H'^{-1} F H^{-1}）计算开销大（DCT/DHT 需 63.3h/45.8h batch=1）→ WHA 单变换 + fast Hadamard kernel（仅加减法，无矩阵乘法），训练时间分别降至 18.2h/9.7h（batch=1/2），接近 CLoQ 的 12.5h/7.1h。

## QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

- baseline方法是什么？
  - Baseline 方法：PTQ (Post-Training Quantization) 方法——以 PTQ4DM、Q-Diffusion、PTQ-D 为代表的典型扩散模型后训练量化方法。这些方法通过小规模校准集计算量化参数（scaling factor s、zero-point Z），尝试平衡裁剪误差（clipping error）和舍入误差（rounding error），实现 W8A8 高位宽下的有效量化。PTQ 在低比特下失效的核心原因：(1) **激活分布不均衡**：扩散模型激活值大多数集中在零附近，但存在稀疏的大值（例如范围 [-10, 34] 但多数值在 [-0.6, 1.7]），这些大值对生成质量很重要（消融实验表明破坏最大值 token 会导致图像严重退化）；(2) **低比特下舍入误差主导**：在 4-bit 下，舍入误差远大于裁剪误差，导致 PTQ 方法优化过程中过度裁剪（over-clipped），产生损坏图像；(3) **理论失效**：Proposition 3.1 指出基于重建的 PTQ 方法在低比特下失去理论保证——激活扰动 Δ 太大导致 Taylor 展开不准确（ḡ ≠ 0，不能简化为二次型）；(4) **无法按时间步动态调整**：部分方法虽支持分时间步量化参数，但需要存储多组参数抵消效率收益。
  - 全栈执行例子（Baseline: Q-Diffusion PTQ on LDM-4 LSUN-Bedrooms W4A4）：
    - **算法pipeline**：全精度 LDM-4 模型加载 → 构建校准集（真实图像经 encoder 得 latent，5120 样本） → 校准各层量化参数 s_w, s_a：W̃_ij = clamp(round(W_ij/s_w)+Z_w; 0, 15), x̃_k = clamp(round(x_k/s_a)+Z_a; 0, 15) → AdaRound 优化舍入策略 → 分块/分层重建：min_s Σ MSE(FP_block_output, Q_block_output) → W4A4 量化模型 → DDIM 200 步生成 → FID 评估。核心缺陷：由于 4-bit 仅 16 个量化等级，大值被过裁剪或舍入，W4A4 下 Q-Diffusion 生成失败（表 3、4 中 FID=N/A）。
    - **系统框架**：论文未明确说明特定 Serving 框架，PTQ 为离线量化流程，量化后的模型可用 PyTorch 直接推理。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 INT4 × INT4 矩阵乘法 kernel（cuBLAS 或类似），量化-反量化在计算前后插入。
    - **硬件架构**：NVIDIA A6000 GPU，标准的 Tensor Cores/INT4 推理路径。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuEST = 选择性渐进权重微调 + 数据无关训练 + TLA + CMA + 全局损失。对应解决 Baseline 缺陷：(1) **解决激活分布不均衡**：通过微调权重间接调整激活分布（图 2 显示范围从 [-10,34] 缩小到 [-4,14]），消除稀疏大值，使分布更紧凑——既保护了重要的大值不被过度裁剪，又使小值获得更细的量化粒度；(2) **解决低比特下舍入误差/裁剪误差难以平衡**：不直接优化 s 的 trade-off，而是修改模型本身使其在量化约束下仍然能保持性能——Theorem 3.2 将大扰动 Δ 分解为 K 小扰动 ε，证明微调权重 w_n 可使模型对量化扰动鲁棒；(3) **解决 PTQ 理论失效**：微调使模型"学习"量化后的输入分布，相当于在量化扰动样本上重新收敛到局部最小值，恢复 Taylor 展开的准确性；(4) **解决时间嵌入量化退化**：Property ❶——量化时间嵌入导致 FID 上升 15%（W4A8 下从 6.77→7.58），通过 TLA (L_TLA = Σ_{l∈C_TE} E_t[||O(t;w_l) - Õ(t;w_l,s_l)||²]) 微调时间嵌入层权值和量化参数，甚至超越全精度 baseline（FID 5.61 vs FP 6.77）；(5) **解决敏感层退化**：Property ❷——FeedForward 层在 6-bit 即失败（而其他线性层在 4-bit 才失败），通过 CMA (L_CMA = Σ_{l∈C_A} E_t[||O(z;w_l) - Õ(z̃;w_l,ŝ)||²]) 专门微调注意力相关层；(6) **全局损失补充局部对齐**：仅 TLA+CMA 只能对齐局部信息，L_G = E_t[||O(x_t;w) - Õ(x_t;w,s)||²] 提供网络级全局监督，且仅含 L_G 时性能反而退化（FID 退化 7.13），说明局部+全局结合的必要性；(7) **数据无关 + 高效**：校准集完全由高斯噪声通过全精度模型采样构造，无需真实数据；仅微调 <7% 参数（约 1% 时间嵌入 + ~5% 注意力层），时间和显存优于 EfficientDM 和 Full-finetune（0.45h vs 2.60h vs 0.85h）。
  - 全栈执行例子（QuEST on LDM-4 LSUN-Bedrooms W4A4, A6000）：
    - **算法pipeline**：(1) 全精度 LDM-4 加载 → 输入随机高斯噪声 x_T∼N(0,I)，在不同时间步 t 采样获得 256 样本/时间步的校准中间激活；(2) 量化初始化：W̃ = clamp(round(W/s_w^init)+Z_w; 0,15), s_a 初始化为 MinMax → 权重量化参数冻结；(3) 阶段一 TLA：仅微调时间嵌入层 w_TE + 对应 s_a_TE，优化 min ΣE_t[||FP_TE(t;w_TE) - Q_TE(t;w̃_TE,s_TE)||²]，约 0.5% 参数，Adam lr=1e-5；(4) 阶段二 CMA：冻结 w_TE/s_TE，微调注意力相关层 w_A + 所有剩余 s_a，优化 min ΣE_t[||FP_attn(z;w_A) - Q_attn(z̃;w̃_A,ŝ)||²]，约 5% 参数；(5) 阶段三：叠加 L_G = E_t[||FP_final(x_t;w) - Q_final(x_t;w̃,s)||²]，联合优化 min(L_TLA + L_CMA + 2L_G)，全流程 <7% 参数更新；(6) 推理：DDIM 200 步，每步 t → TimeEmbed(t) 经微调层，UNet 经量化层前向，端到端 FID 5.64 (W4A4)。
    - **系统框架**：论文未明确说明特定 Serving 框架，量化模型使用 PyTorch 直接推理。校准和微调流程为离线优化，可独立复现。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 INT4 量化 kernel，矩阵乘法使用量化整数运算，量化/反量化和 clamp 操作在层间插入。无定制 kernel。
    - **硬件架构**：NVIDIA A6000 48GB GPU，Tensor Cores 支持 INT8/FP16 推理路径。

## QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

- baseline方法是什么？
  - Baseline 方法：(1) **QuIP** (Chee et al., 2023)：使用 Kronecker 积构造的 2-factor 正交矩阵（U = U_1 ⊗ U_2, V = V_1 ⊗ V_2）做非相干处理，复杂度 O(n√n)；标量 LDLQ 按列自适应舍入（一次一列）；舍入目标为半整数格（1D）；无微调。(2) **OmniQuant** (Shao et al., 2024)：通过学习可微的模型保持变换（model-preserving transformations）按 Transformer Block 减少离群值，启发式方法在低比特下失效。(3) **AWQ** (Lin et al., 2023)：在量化前按激活幅度缩放权重，2.15-bit 即崩溃。(4) **AQLM** (Egiazarian et al., 2024)：使用可学习非结构化 8D 向量量化码书（每层一个 2^16×8 码书占 1MiB），码书太大无法放入 L1 cache 导致推理慢于 FP16。
  - 全栈执行例子（Baseline: QuIP 2-bit on Llama 2 7B）：
    - **算法pipeline**：Llama 2 7B FP16 权重加载 → 计算代理 Hessian H（RedPajama 6144 seqs × 4096 ctx）→ Kronecker 非相干处理：随机生成正交矩阵 U_1,U_2（≈√n 维）和 V_1,V_2（≈√m 维），构造 U=U_1⊗U_2, V=V_1⊗V_2 → Ŵ ← UWV^T, Ĥ ← VHV^T → LDLQ 按列标量舍入：对 H 做 LDL 分解 H=L^TDL，设置 U=L^T-I，逐列 Ŵ_k = round(Ŵ_k + (Ŵ_{:(k-1)} - Ŵ̂_{:(k-1)})a_k) → 推理：激活 x → Vx → 量化权重矩阵乘法 → U^T(quantized(Ŵ)(Vx))。主要缺陷：(a) Kronecker μ_W 依赖 log²(mn/δ)，不如 RHT 的 log 依赖；(b) 标量舍入产生的可表示权重向量形成超立方体，与 RHT 变换后球状高斯分布不匹配；(c) 无微调导致量化误差仅局部最小化。
    - **系统框架**：无特定 Serving 框架修改，量化后模型以 PyTorch 推理。离线 PTQ 流程。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 FP16 矩阵乘法 + Hadamard 变换乘法。无定制压缩解码 kernel。QuIP 标量量化权重可直接用标准 INT 运算。
    - **硬件架构**：GPU（A100 量化 / A6000 推理），标准 CUDA 路径。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuIP# = **RHT 非相干处理** + **BlockLDLQ + E8P 格基码书向量量化** + **层间微调**。三大组件分别解决 Baseline 核心缺陷：
    (1) **RHT 取代 Kronecker**（解决非相干效率与理论界问题）：用 Randomized Hadamard Transform（Had(S·x)）替代 QuIP 的 2-factor Kronecker 积。优势：(a) μ_W = 2log(4mn/δ) vs Kronecker 的 A²log²(4Cmn/δ)²——对数依赖替代对数平方依赖；(b) 时间复杂度 O(n log n) vs O(n√n)；(c) Hadamard 矩阵元素为 {±1}，乘法无需浮点运算，常数因子更低；(d) 消融实验（Table 4 "no E8" 行 vs QuIP 行）验证 RHT 独立于其他组件即带来显著困惑度改善（2-70B 2-bit: 4.58 vs 5.90 Wikitext2）。
    (2) **BlockLDLQ + E8P 格基码书取代标量舍入**（解决分布匹配问题）：(a) BlockLDLQ 将 QuIP 的标量 LDLQ 推广到 g 列块级向量量化——基于 g-block LDL 分解 H=L^TDL，设置 U=L^T-I，按 8 列块迭代 Ŵ_k = Q(Ŵ_k + (Ŵ_{:(k-1)} - Ŵ̂_{:(k-1)})A_k)——Theorem 4.1 给出误差界 ∝ gmμ²σ² tr(H^{1/2})²/n；(b) E8P codebook 基于 E8 格（8 维最优球填充密度，kissing number 最优），通过符号翻转对称性将 2^16 条目压缩为 2^8 源码书（1KiB = L1 cache fit），球状码书形状匹配 RHT 变换后的亚高斯权重分布；(c) RVQ 扩展高比特：4-bit = 2×2-bit E8P 残差量化；(d) E8P 在相同的 2-bit 下显著优于标量半整数舍入（2-70B: 4.16 vs 4.58 Wikitext2）。
    (3) **层间微调取代纯 PTQ**（解决层间交互缺失问题）：(a) Transformer Block 内微调：冻结已量化层权重，Adam 优化未量化层和 sign vectors（FP16）以最小化 Block 输出 MSE——减少激活误差累积；(b) 端到端微调：所有层量化完毕后，优化 layernorms、S_U、S_V、LM head 以最小化 CrossEntropy——捕获全局层间交互。2-bit 模型受益最大（2-7B: 8.22 → 6.19 Wikitext2 含 FT）。约 50 GPU-hours 量化 70B 模型，显著少于 QAT（LLM-QAT 需 960 GPU-hours 仅生成训练数据）。
  - 全栈执行例子（QuIP# 2-bit on Llama 2 70B, A100 + RTX 4090）：
    - **算法pipeline**：FP16 权重加载 → RedPajama 生成 Hessian H（6144 seqs × 4096 ctx）→ IP-RHT（Algorithm 3）：采样 S_U∼{±1}^m, S_V∼{±1}^n → Ŵ←Had(S_U·Had(S_V·W^T)^T), Ĥ←Had(S_V·Had(S_V·H)^T) → BlockLDLQ（g=8）：Ĥ=L^TDL → U=L^T-I → 逐 8 列块 Ŵ̂_k = Q_E8P(Ŵ_k+(Ŵ_{:k-1}-Ŵ̂_{:k-1})A_k) → RVQ 残差：2× E8P 2-bit → 层间微调（Algorithm 5）：per-block Adam MSE → 端到端 Adam CrossEntropy → 推理：x → Had(S_V⊙x) (FWHT O(n log n)) → E8P_decode_matvec kernel (MMA Tensor Core) → Had(S_U⊙y) → 下一层。Wikitext2 PPL 3.91（2-bit 70B）。
    - **系统框架**：论文未修改特定 Serving 框架（如 vLLM），但提供可直接加载的 PyTorch 量化模型（HuggingFace relaxml），推理过程集成 FWHT + E8P GEMV CUDA kernel。未来可集成到 vLLM 等框架的量化后端。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：自研 CUDA kernel `decode_matvec_e8p_kernel`：(a) 从压缩 uint2 码字中解码 4 个 E8P 码字 → 查 256 条目 codebook_abs → XOR 符号翻转 + ±1/4 偏移 → 生成 FP16 权重；(b) `mma.sync.aligned.m16n8k16` Tensor Core MMA 指令累加；(c) 1KiB codebook 放 L1 cache → 无 DRAM 往返 → 2-70B 达 56.84% peak mem BW（32.74 tok/s）。AQLM 1MiB codebook 导致 cache thrashing（8.27 tok/s, < FP16）。
    - **硬件架构**：NVIDIA A100（量化计算）、RTX 4090（推理性能测试），标准 Tensor Core + CUDA 路径。无定制硬件。

## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

- baseline方法是什么？
  - Baseline: 传统 LLM 激活量化方法（SmoothQuant, OmniQuant, QUIK, Atom）采用"识别并特殊处理离群值"的策略——用校准集识别离群值特征通道（outlier features），将这些通道保持在高精度（FP16 或 INT8），其余通道量化到低比特。SmoothQuant 通过 per-channel scaling 将量化难度从激活值迁移到权重，解决了 8-bit 量化问题，但在 4-bit 失效。QUIK 和 Atom 在 4-bit 下仍需保留部分高精度通道或使用复杂混合精度矩阵乘法 kernel。这些方法的根本缺陷：(1) **治标不治本**：依赖校准集识别离群值，未从根本上消除离群值产生的原因；(2) **混合精度 kernel 开销大**：需要特殊内存布局分离离群值/正常通道，增加 kernel 复杂度和延迟；(3) **无法全 4-bit**：始终有部分计算或参数保持更高精度，限制了内存节省和加速的理论上限；(4) **KV cache 离群值问题未系统解决**：KV cache 量化（如 KVQuant, KIVI）需要 feature-wise 量化、非均匀表示、保留高精度离群值等复杂机制。
  - 全栈执行例子（Baseline: 4-bit Atom 量化 LLAMA2-7B, NVIDIA RTX 3090）：
    - **算法pipeline**：FP16 LLAMA2-7B → 校准集推理识别激活值离群通道 → 离群通道保留 FP16，非离群通道 per-token 4-bit 量化 → 离线权重量化（GPTQ-128G, group=128）→ 推理时混合精度 MatMul：离群通道 FP16×FP16 + 非离群通道 INT4×INT4 → 需 special reordering kernel 分离两类通道 → KV cache: 论文未明确说明 4-bit KV cache 量化方案 → WikiText-2 PPL: 6.03 (7B), 5.26 (13B)。
    - **系统框架**：Hugging Face Transformers + PyTorch。论文未明确说明 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：Atom 混合精度 MatMul kernel：按 outlier channel mask 将输入 X 分拆为 X_outlier (FP16) 和 X_normal (INT4)，分别执行 FP16 GEMM 和 INT4 GEMM → 合并结果。需要特殊的 weight reordering 预处理。离群值 mask 需校准集确定。
    - **硬件架构**：NVIDIA RTX 3090 GPU，Tensor Cores 加速统一精度 GEMM。混合精度路径需额外的 memory reordering 和 kernel launch overhead。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuaRot 从根源上消除离群值——通过随机 Hadamard 旋转将权重和激活值"失相关"（incoherence processing），利用计算不变性定理将旋转矩阵融入网络权重中，使激活值分布均匀化，无需识别和特殊处理任何离群通道。具体设计映射：(1) **旋转消除离群值（对应"治标不治本"）**：随机 Hadamard 矩阵 Q 具有"扩散"效应——Q^T 将单通道的大值均匀分布到所有通道（每通道的 Hadamard 变换是 ±1 加权求和），图 1 验证变换后激活值从长尾分布变为类高斯分布，无任何离群值；(2) **计算不变性保证等价性（对应"混合精度 kernel"）**：利用 RMSNorm 旋转等变性 RMSNorm(X) = RMSNorm(XQ^T)Q，将 Q 融入相邻权重矩阵 W ← Q^T W，前向网络数学上完全等价，因此无需混合精度——所有 MatMul 均为统一 INT4×INT4；(3) **每层仅 1.5 次在线 Hadamard（对应"复杂 kernel"）**：对比 QuIP# 每权重矩阵需 2 次 Hadamard 变换，QuaRot 将大部分 Hadamard 融入权重，仅保留 down-projection 和 out-projection 前的在线变换。Walsh-Hadamard 变换 O(d log d) 极快，在线开销仅 ~7%（Table 14 验证）；(4) **KV cache 全量化（对应"KV cache 未解决"）**：head-wise Hadamard 旋转消除 Key 和 Value 中的离群值，Post-RoPE 在线旋转使量化 KV cache 在 4-bit 下困惑度几乎无损（+0.04 on 7B, +0.03 on 13B, +0.01 on 70B, Table 6），keys 比 values 更敏感（K3V4 困惑度 5.65 vs K4V3 的 5.54）；(5) **无需校准集的无损 8-bit（附加优势）**：RTN 8-bit 量化完全无需校准数据，困惑度 5.50 vs FP16 5.47（Table 3），同时 Hadamard 旋转提升 weight-only 量化质量：4-bit GPTQ 从 8.25 → 5.60 (7B)，2-bit 从 NaN → 22.07（Table 7）。
  - 全栈执行例子（QuaRot: 4-bit LLAMA2-7B, NVIDIA RTX 3090）：
    - **算法pipeline**：FP16 LLAMA2-7B → 离线：生成 Q = H_4096 diag(s), s_i∈{±1}（利用 Walsh-Hadamard O(d log d) 结构和已知 Hadamard 矩阵库 Sloane 2024 处理非 2^n 维度）→ 吸收 RMSNorm α 到相邻权重 → 所有 "输入侧" 权重左乘 Q^T：W_gate/up/k/q/v ← Q^T diag(α) W → 所有 "输出侧" 权重右乘 Q: W_down ← H W_down Q, W_out ← H(I⊗H_{128})W_out Q → W_v 右乘 (I⊗H_{128}), W_out 左乘 (I⊗H_{128})（利用 identity H = (I⊗H_{d_h})(H_{n_h}⊗I)）→ GPTQ 权重量化（128 calib samples, per-column symmetric INT4, MSE-optimal clipping）→ 推理：X 经 RMSNorm → per-token 量化（s_x = max(|X|)×0.9/7, X_q=round(clip(X/s_x,-7,7))) → CUTLASS INT4 GEMM (W_gate/up) → dequant FP16 → SiLU gate → 在线 Hadamard → per-token 量化 → CUTLASS INT4 GEMM (W_down) → dequant → 输出 YQ → 无任何离群值通道，无混合精度。
    - **系统框架**：Hugging Face Transformers + PyTorch。论文未明确说明 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：CUTLASS INT4 TensorCore GEMM（sub-byte packed activations+weights, INT32 accumulator, per-token+per-column scale dequant output）→ 在线 Walsh-Hadamard fast kernel（O(d log d), FP16/FP32, ~7% overhead）→ FlashInfer 量化 KV cache kernel（Append: quantize→pack→store; Decode: load→dequant→FP16 dot product with online softmax）。LLAMA2-7B prefill 加速 2.16×（batch=64, seq=2048），LLAMA2-70B 达 3.33×。解码内存节省 3.63×−3.89×。
    - **硬件架构**：NVIDIA RTX 3090 GPU（Ampere Tensor Cores），FP16/INT4/INT32 精度层次。B200 GPU 的 FP4 硬件支持与本方法的 INT4 路径可类比（论文 conclusion 提出）。

## QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

- baseline方法是什么？
  - Baseline 方法：(a) **Open-Sora 1.2**（FP16, 无优化）：DiT-based video generation，100 timesteps denoising，生成 512×512×64 frames 视频耗时 130s on A800-80GB；(b) **现有量化方法**（ViDiT-Q, SmoothQuant, Q-DiT, PTQ4DiT, Quarot, Q-diffusion）：均为 static uniform quantization——固定 bit-width（W8A8 或 W4A8）应用于所有 layers 和所有 timesteps，不区分 layer importance 和 timestep redundancy；(c) **现有缓存方法**（AdaCache, Δ-DiT, T-Gate, PAB, DeepCache）：使用预定义的 static cache schedule（固定缓存间隔），不根据 content-dependent feature divergence 动态调整；(d) **现有剪枝方法**：static layer pruning（预定义固定子集剪枝），不根据运行时 feature similarity 动态调整。Baseline 的核心缺陷：(1) **静态量化**：uniform bit-width 对所有 layer 和 timestep 一视同仁，导致关键层/关键 timestep 精度不足、冗余层/冗余 timestep 计算资源浪费；(2) **静态缓存**：固定间隔缓存不考虑帧间内容变化速度——静态背景也应频繁刷新，剧烈运动场景也按固定间隔缓存导致质量下降；(3) **单技术孤立**：量化、缓存、剪枝各自独立应用，未联合优化，总加速比受限于单一维度（如 ViDiT-Q 仅 1.71×，AdaCache-fast 仅 2.24×）；(4) **DiT 架构适配不足**：DiT 缺乏 U-Net 的 skip connections，传统 feature map caching（如 DeepCache）效果差。
  - Baseline 在模型推理全栈的执行例子（以 Open-Sora FP16 + ViDiT-Q W8A8 为例）：
    - **算法pipeline**：Open-Sora 1.2 FP16 权重 → ViDiT-Q per-tensor/per-channel W8A8 uniform quantization → 所有 100 timesteps 用固定 8-bit 精度 → 所有 DiT blocks（STA + CA + FFN）每步完整执行 → VAE decoder 生成视频。
    - **系统框架**：论文未明确说明 Serving 框架。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：标准 INT8 GEMM CUDA kernel，无 kernel fusion，无 caching。量化、GEMM、dequant 各为独立 kernel launch。ViDiT-Q 1.71× speedup on A800-80GB。
    - **硬件架构**：NVIDIA A800-80GB GPU (Ampere)。生成 512×512×64f 视频 130s（Open-Sora FP16 baseline）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QuantCache = HLC + AIGQ + SRAP 三层联合优化。解决 Baseline 缺陷的方式：(1) **HLC 动态缓存替代静态缓存**：通过 inter-step feature divergence D_t^(l) = ||p_t^(l)-p_{t-k}^(l)||_1/k · ||∇_t m_t^(l)|| 实时评估内容变化速度，分三档自适应刷新（τ_max/τ_mid/τ_min）。变化小的帧（如静态背景）→ 长间隔缓存（τ_max）；变化大的帧（如场景切换）→ 频繁刷新（τ_min）。解决 AdaCache 等静态调度的内容无关性问题，单独 HLC 即实现 4.12× speedup；(2) **AIGQ 动态量化替代 uniform quantization**：权重层面，先离线评估每层 sensitivity（numerical error + perceptual distortion + temporal dynamics），在总预算 B_total 约束下迭代分配 bit-width（Σ B(l) ≤ B_total），关键层（精细纹理/运动连续性相关）多分配精度，冗余层少分配。激活层面，基于 timestep 冗余度 D_t 动态选择 bit-width(t) ∈ {Bit_max, Bit_mid, Bit_min}——高冗余步（连续帧变化小）用 Bit_min 激进量化，低冗余步（场景转换/细节涌现）用 Bit_max 保留精度。额外引入 channel balancing（scaling + rotation）消除量化 outlier。解决 ViDiT-Q 等 uniform quantization 的"一刀切"问题，AIGQ+HLC 联合达 6.33× speedup；(3) **SRAP 动态剪枝替代静态剪枝**：在线计算相邻层 cosine similarity S_t^(l,l+1)，S > τ_high 时完全跳过 layer l+1；同时追踪累积 feature variation V_t 动态调整全局剪枝概率——V_t 低（精细 refine）→ 激进剪枝，V_t 高（剧烈变换）→ 保守剪枝。解决静态剪枝的刚性，总 speedup 达 6.72×；(4) **联合优化替代孤立应用**：HLC（跨 timestep 冗余消除）、AIGQ（精度自适应）、SRAP（同 timestep 层间冗余消除）三个维度互补——HLC 跳过冗余 timestep 时 AIGQ 降低精度进一步加速；AIGQ 激进量化后的 low-precision features 使 SRAP 的相似度计算和剪枝判断更高效；小 skip（低 τ_t^(l)）用更小 bit-width 利用高冗余，大 skip（高 τ_t^(l)）后增加精度补偿缓存带来的 drift。
  - 全栈执行例子（QuantCache W4A6 + HLC + SRAP on Open-Sora 1.2, A800-80GB）：
    - **算法pipeline**：Open-Sora 1.2 预训练权重 (FP16) → Offline calibration → AIGQ 混合精度分配（sensitive layers: W8A8, medium: W6A6, redundant: W4A4）→ Channel balancing factors offline 吸收到前层权重。Inference: timestep t → (a) HLC: D_t^(l) = ||p_t^(l)-p_{t-k}^(l)||_1/k · ||∇_t m_t^(l)|| → 三档刷新决策 → 如命中缓存则跳过当前层；(b) AIGQ: bit-width(t) = f(D_t, θ_1, θ_2) → 量化权重 W̄ 和激活 X̄ → 执行 low-precision fused GEMM；(c) SRAP: S_t^(l,l+1) = cosine(p_t^(l), p_t^(l+1)) → 如 S > τ_high → 跳过 layer l+1；(d) V_t 全局剪枝率调整。最终 speedup 6.72×（130s → ~19.3s）。
    - **系统框架**：论文未明确说明特定 Serving 框架，基于 Open-Sora 推理代码直接修改。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：Fused CUDA kernel（quantization + rotation + GEMM 单次 launch）+ HLC cache buffer（GPU global memory）+ SRAP kernel-skip 逻辑（kernel 调用侧判断）。CUDA 12.1, A800-80GB。Kernel fusion 将 3 次 kernel launch（quantize/rotate/GEMM）合并为 1 次。
    - **硬件架构**：NVIDIA A800-80GB GPU (Ampere, 80GB HBM2e)。6.72× speedup 在单 GPU 上实现，论文未涉及多 GPU 或分布式推理。

## SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

- baseline方法是什么？
  - Baseline 方法：(a) **传统 PTQ 方法**（PTQ4VM, Quamba, MambaQuant）：均为 quantizer-only 方法——仅优化量化器裁剪界 (a, b) 以最小化 ||x - x_q||，不修改预训练模型权重。量化过程为：固定权重 W → clip(W, a, b) → round → dequant → W_q，模型自身不具备主动适应量化的能力。(b) **CNN/Transformer PTQ 方法**（DBDC+Pac, 2DQuant）：为 CNN 或 Transformer 设计的 PTQ，移植到 Mamba 架构时无法处理 Mamba 的 recurrent state 和 dynamic gating 带来的 error accumulation 和 numerical sensitivity 问题，导致在 SR 任务上细节模糊和纹理丢失。(c) Baseline 的核心缺陷：(1) **仅优化量化器参数，不修改模型权重**——权重本身不包含针对量化的补偿信息，在极端低比特（2-bit）下信息损失严重；(2) **补偿信息单一**——单组补偿参数无法覆盖不同层的异构量化误差模式，产生 homogenization 问题；(3) **Mamba 架构适配缺失**——现有 Mamba 量化方法（Quamba, MambaQuant）主要验证在 classification/language modeling，移植到 SR 时因 pixel-level precision 要求无法满足 fidelity 需求。
  - Baseline 在模型推理全栈的执行例子（以 MambaIRv2-light, PTQ4VM W4A4, ×2 SR 为例）：
    - **算法 pipeline**：预训练 MambaIRv2-light FP32 权重 (3.01MB) → PTQ4VM 仅校准量化器 (a, b)（无权重修改）→ Q_{a,b}(W) = clip(W, a, b) → round → dequant → 得到 W4A4 量化权重 → 前向推理 X @ W_q → 输出 SR 图像。Set5 PSNR=37.17 (vs FP32 38.26, 下降 1.09dB)。
    - **系统框架**：论文未明确说明。
    - **编译框架**：论文未明确说明。
    - **kernel 调度**：标准低精度 GEMM（PaddlePaddle 框架内置），无额外 kernel fusion 或算子优化，量化前后无缓存或调度逻辑。RTX 4090, 无自定义 CUDA kernel。
    - **硬件架构**：NVIDIA RTX 4090 GPU。推理时 MambaIRv2-light 以 75.6G FLOPs 执行，PTQ4VM 仅降低精度不改变计算图结构，FLOPs 与量化后模型参数位宽成比例（4-bit: 22.0G，2-bit: 18.2G）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：SPR²Q = PQFR (Pre-Quantization Fine-tuning with Fused Rectifier) + SPR² (Static Priority-Based Rectifier Routing)。核心思想是"在量化前主动向模型注入补偿信息"，使模型自适应量化过程。(1) **PQFR 解决"仅优化量化器"缺陷**：引入可训练低秩 rectifier ΔW = BA (A∈ℝ^{r×d_in}, B∈ℝ^{d_out×r})，在量化前将 rectifier 增量融合到冻结权重 W' = W + ΔW，再对 W' 进行量化。同时联合优化 rectifier 参数 (A,B) 和量化器参数 (a,b)，使用混合损失（L_pixel 像素级 + L_feature 逐块特征对齐），通过 STE 反向传播。这使权重本身包含了补偿量化误差的先验信息，P只用 PQFR 即从 37.20 提升至 37.44 dB PSNR。(2) **SPR² 解决"补偿信息单一"缺陷**：将单 rectifier 扩展为 N=4 个 rectifier 组成的 rectifier group。训练阶段使用轻量门控网络 g_i 为每个输入动态加权聚合不同 rectifier 的增量（ΔW_fused = Σ g_i · ΔW_i），鼓励各 rectifier 专业化处理不同类型量化误差。离线校准阶段通过梯度下降学习每个模块的最优静态门控权重 ĝ（Eq. 12），构建 SPR²Q Table 记录每个模块的最优增量。推理时每个模块直接从 Table 检索最优 ΔW，离线融合并量化，推理时无任何额外计算。(3) **跨架构泛化**：在 SwinIR-light (Transformer) 上 2-bit ×2 SR 超越 2DQuant (+1.14 dB)、FIMA-Q (+1.22 dB)、APHQ-ViT (+1.14 dB)，验证方法不依赖特定架构。
  - 全栈执行例子（SPR²Q W4A4, MambaIRv2-light, ×2 SR, RTX 4090）：
    - **算法 pipeline**：
      **训练阶段**：(a) Rectifier Group Training: 初始化 N=4 个 rectifier {(A_i, B_i)}, r=8 → 对每层 Mamba 模块，门控网络 G 输出 g_i → ΔW_fused = Σ g_i·(B_i@A_i) → W' = W + ΔW_fused → 伪量化 W_q' = Q_{a,b}(W') → 前向 Y = X @ W_q' → Loss = L_pixel + λ·L_feature → STE 梯度近似 → Adam(lr=1e-2) 更新所有参数，12k iters, batch=8；(b) Offline Static Routing Calibration: 冻结所有 rectifier 参数 → 对每层优化 ĝ = argmin_g L (500 iters, batch=8) → 计算每个模块最优增量 ΔW_opt[l] = Σ ĝ_i·(B_i@A_i) → 构建 SPR²Q Table。
      **推理阶段**：对每个 Mamba 模块 l：检索 ΔW_opt[l] → W_final = W + ΔW_opt[l] → 量化 W_q_final = Q_{a,b}(W_final) → 前向 Y = X @ W_q_final（计算图与原始 MambaIRv2-light 完全一致，零额外开销）。输出 Set5 PSNR=37.72 (vs FP32 38.26, 仅降 0.54dB；vs PTQ4VM +0.55dB)。
    - **系统框架**：论文未明确说明。
    - **编译框架**：论文未明确说明。
    - **kernel 调度**：使用 PaddlePaddle 框架内置量化算子，无自定义 CUDA kernel。推理时所有 rectifier 参数已离线融合，不引入额外 kernel launch 或内存访问。MambaIRv2-light 4-bit: 1.20MB (2.51× 压缩), 22.0G FLOPs (3.44× 加速)；2-bit: 1.07MB (2.81×), 18.2G (4.15×)。
    - **硬件架构**：NVIDIA RTX 4090 GPU。推理时模型结构与原始 MambaIRv2-light 完全相同，仅权重值为量化后的整数值。压缩和加速完全来自 bit-width 降低，无额外硬件修改或异构计算。

## SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

- baseline方法是什么？
  - **Baseline: FlashAttention2 (FP16) + xformers (FP16)**。全精度Attention计算流程：Q, K, V在FP16精度下，使用FlashAttention-2的tiling策略（tiling Q/K/V into blocks b_q, b_kv）和online softmax，通过FP16 Tensor Core mma(f16.f16.f32)指令完成$QK^\top$和$PV$两次Matmul。S和P矩阵（N×N）无需显式写入HBM，通过online softmax逐步累加O_i。
  - **全栈执行例子（FlashAttention2, Llama2-7B, RTX4090, headdim=128, seq_len=1536）**：
    - **算法pipeline**：Q ∈ R^{1536×128}, K ∈ R^{1536×128}, V ∈ R^{1536×128} FP16 → $S=QK^\top/\sqrt{d}$（FP16 Matmul, FP32 accum）→ $P=\sigma(S)$（FP16 online softmax）→ $O=PV$（FP16 Matmul, FP32 accum）→ 输出O ∈ FP16。全程无量化，精度无损但吞吐仅165 TOPS。
    - **系统框架**：PyTorch + HuggingFace Transformers，调用`flash_attn_func()`或xformers的`memory_efficient_attention()`。
    - **编译框架**：FlashAttention2使用CUDA C++直接编写，经NVCC编译为PTX/SASS，不经过高层编译框架。
    - **kernel调度**：CUDA kernel on RTX4090: 使用FP16 mma指令（理论330 TFLOPS FP16），实际165 TOPS（50%利用率）。Kernel block sizes b_q=128, b_kv=64, Num Warps=4/8，数据从HBM→SRAM→Tensor Core→SRAM→HBM。
    - **硬件架构**：NVIDIA RTX4090 Ada Lovelace架构，Tensor Core FP16 throughput 330 TFLOPS（non-sparse），128KB SRAM/SM，HBM bandwidth 1008 GB/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：SageAttention，INT8量化Attention，在保证端到端精度无损的前提下实现2.1×加速（vs FlashAttention2）。三大设计对应baseline缺陷：(1) **Smooth K解决K的channel-wise outlier导致INT8量化精度崩溃**：FlashAttention2无法直接降精度——K矩阵存在channel-wise large bias outlier（所有token共享的大偏置），直接INT8量化产生完全模糊图像（Unidiffuser FID从163→267）。SageAttention发现$\sigma(q(K-\text{mean}(K))^\top)=\sigma(qK^\top)$，通过减去token均值消除outlier后量化，INT8 per-token量化精度从CosSim 62%提升到99.5%，overhead <0.2%。(2) **FP16 Accumulator for PV解决PV INT8量化最差层精度不达标**：FlashAttention2的PV用FP16 precision FP32 accum，无量化。直接INT8量化PV在部分层cosine similarity仅56%，引入FP16 accumulator方案——P,V保持FP16但accumulator用FP16而非FP32，RTX4090上FP16 accum比FP32 accum快2×，且与FP32 accum精度完全一致（CosSim差值=0.00%, Relative L1差值=0.0000）。(3) **Adaptive Quantization解决speed-accuracy tradeoff的单kernel选择困境**：在所有层用保守的SAGEAttn-B（QK INT8 + PV FP16）可获得2×加速但非最优；或激进使用SAGEAttn-vB（全INT8）更快4%但部分层精度不足。Adaptive方案对每层离线测试vB cosine similarity，若>99.8%（B的最差cosine sim）则用vB，否则用B，实现+11.7% OPS提升且零精度损失。
  - **全栈执行例子（SageAttention, Llama2-7B, RTX4090, headdim=128, seq_len=1536）**：
    - **算法pipeline**：
      1. Smoothing: K_smooth = K - mean(K)  # [1536×128] - [1×128], 消除channel bias outlier
      2. Fused ROPE + Quant: Q̂_INT8 = ψ_Q(Q/√d)  # per-block INT8, scale δ_Q[b_q]
      3. Fused ROPE + Quant: K̂_INT8 = ψ_K(K_smooth)  # per-block INT8, scale δ_K[b_kv]
      4. FlashAttention-style tiled loop（Triton kernel）:
         - S_i^j = Matmul(Q̂_i, K̂_j^T) × δ_Q[i] × δ_K[j]  # INT8 Tensor Core mma(u8.u8.s32), dequant via scale mul
         - Online softmax: P̃_i^j = exp(S_i^j - m_i^j)  # FP16
         - O_i^j = diag(exp())O_i^{j-1} + Matmul(P̃_i^j, V_j, accum=FP16)  # FP16+FP16 accum mma(f16.f16.f16), 2× faster than FP32 accum
      5. Final: O_i = diag(l_i)^{-1}O_i^{T_n}  # FP16 output, 231.74 TOPS (vs FlashAttn2 130.99 TOPS)
      End-to-end: WikiText perplexity 5.824 vs FP16 5.823 (Δ=+0.001), LAMBADA 0.887 vs 0.886, MMLU 0.46 vs 0.46.
    - **系统框架**：即插即用替换——`import sageattention; replace_attention(model)` → 自动将PyTorch模型中的`scaled_dot_product_attention`或`flash_attn_func`替换为SageAttention Triton kernel。与AWQ（W4A16线性层量化）正交组合，在AWQ+Llama2上attention加速2×而perplexity仅从5.4729→5.5998。
    - **编译框架**：Triton（OpenAI）编译链：Python DSL → Triton IR → Triton MLIR → LLVM IR → PTX → SASS。通过`tl.dot()`自动映射到Tensor Core mma指令，Triton compiler自动处理shared memory allocation、register allocation、instruction scheduling。无额外编译框架修改。
    - **kernel调度**：Triton kernel on RTX 4090 Tensor Core:
      - INT8 mma: Q̂_i[128×128] @ K̂_j^T[128×64] = S[128×64], u8.u8.s32, 660 INT8 TOPS峰值 → 实测340 TOPS (52% util)
      - FP16+FP16 accum mma: P̃[128×64] @ V[64×128] = ΔO[128×128], f16.f16.f16, 330 FP16 TOPS峰值
      - 混合精度kernel: 同一kernel内交替使用INT8和FP16 Tensor Core指令
      - Block sizes: Q tile 128, KV tile 64; Num Warps=8 (headdim=128), Num Stages=5 (causal)
      - 231.74 TOPS实测（Llama2 1536×128）, 1.77× speedup vs FlashAttention2
    - **硬件架构**：NVIDIA RTX 4090 Ada Lovelace。SageAttention无硬件修改，完全利用现有Tensor Core指令集。INT8 mma throughput 660 TOPS（理论）和FP16 mma 330 TFLOPS（理论，FP16 accum为2× FP32 accum的512 FLOPS/cycle/SM vs 256）。INT8计算使HBM→SRAM数据传输量减半，缓解memory bandwidth瓶颈。

## Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification

- baseline方法是什么？
  Baseline 方法为标准的 1.58-bit/1.67-bit 三值量化（TWN、BitNet、TernaryLLM、LLM-QAT、ParetoQ、Spectra、Tequila），将权重限制为 {+1, 0, -1} 三值集合。推理时通过查表引擎（BitNet.cpp, T-MAC）将浮点乘法转换为硬件高效的整数加法。Baseline 在存储和 SIMD 对齐方面有两种策略：(1) **2-bit 打包**（如 BitNet I2_S）：每权重用 2 bits 存储（4 值需要 2 bits），简单但对齐付出了 0.42 bit/weight 的浪费（实际信息只有 1.58 bits）；(2) **1.67-bit 打包**（如 Tequila TL2）：3 个三值权重打包为 5 bits（3³=27 < 2⁵=32），虽节省存储但引入 3-way 不规则 pattern，与 SIMD 的 4/8/16-way 向量通道不对齐，需昂贵的 bit shuffle 操作。训练时标准三值 QAT 存在 weight trapping 问题：处于 deadzone [-Δ, Δ] 内的权重因 STE 接收无信息梯度而停滞，梯度同质化导致表示坍缩。

  **Baseline 全栈执行例子（以 BitNet 1.58-bit 在 LLaMA-3.2 1B 上的推理为例）：**
  - **算法 Pipeline**：权重矩阵 W 经三值量化 Q(W) = α·sign(W)·I[|W| ≥ Δ] → 每权重存储为 2 bits（浪费 0.42 bits）或以 1.67-bit 打包（3-权重 5-bit → SIMD 不友好）→ 推理时查表执行 X·Q(W) 替代浮点 MUL → BF16 激活保持全精度。
  - **Serving 框架**：论文未明确说明（推理使用 llama.cpp 或其衍生引擎，加载 GGUF 格式，无特定 serving 调度优化）。
  - **编译框架**：论文未明确说明。
  - **Kernel 调度**：BitNet.cpp 的 SIMD kernel 使用 2-bit 打包 → 4 权重/8 bits → 4-way 对齐 SIMD（128-bit: 4×FP16 = 64 bits 激活，256-bit: 4×FP32 = 128 bits 激活），但存储效率只有 1.58/2 = 79%。若使用 1.67-bit 不规则打包（如 Tequila），3-way pattern 导致 SIMD vector lane 不对齐，需 bit-level shuffle → 额外的位操作开销和缓存线碎片化 → 推理速度变慢。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  Baseline 的核心缺陷：(a) **存储-SIMD 对齐矛盾**：2-bit 打包浪费存储（~21% 开销），1.67-bit 打包破坏 SIMD 对齐（3-way 不规则 pattern 导致 kernel 需 bit-level 操作，抵消低位宽的推理加速收益）；(b) **训练时 weight trapping**：标准三值 QAT 中，梯度通过 STE 量化函数时，deadzone 内权重梯度为零信号 → 权重停滞无法逃离 deadzone → 同质化梯度导致表示坍缩（低秩，模型精度显著下降）；(c) 现有三值方法难以在 1.25-bit 极低位宽下同时保持精度和硬件效率。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 Sherry，通过两个核心创新解决 baseline 缺陷：

  **创新一：3:4 细粒度结构化稀疏**
  打破 "2-bit 浪费 vs 1.67-bit 不对齐" 的矛盾——强制每 4 个权重中恰好 3 个非零和 1 个为零，4 个权重恰好有 C(4,3)×2³ = 32 种状态，完美饱和 5-bit 索引 → 等效 1.25 bit/weight（比 1.67-bit 节省 25% 存储），同时 4-way pattern 天然对齐 128/256/512-bit SIMD 向量通道 → 零 bit-level shuffle 开销。

  **创新二：Arenas（Annealing Residual Synapse）**
  解决 weight trapping 问题——训练时注入全秩残差 Y = X·Q(W) + λ_t·X·W（λ_t 从初值退火至零），为 deadzone 内权重提供连续梯度信号，防止 ∂L/∂X 坍缩为低秩。λ_t → 0 后 Arenas 路径融合入静态参数，推理零耗。

  **论文方法全栈执行例子（Sherry 1.25-bit 在 LLaMA-3.2 1B 上的推理）：**
  - **算法 Pipeline**（训练时）：
    1. 每个连续 4 权重打包为一组，argmin|w_i| 权重置零，其余 ±1 量化 → 32 种排列 → 5-bit 索引
    2. Arenas 路径 Y = X·Q(W) + λ_t·X·W 并行注入异构梯度，λ_t 退火到 0
    3. 梯度 ∂L/∂X = ... 含全秩分量（来自 Arenas），避免低秩坍缩
    4. 训练完成：Arenas 融合入 bias，模型仅有 1.25-bit 权重 + BF16 激活
  - **Kernel 调度**：5-bit 打包 → 4-way SIMD 对齐 → 128-bit SIMD 处理 4 个 FP16 激活 × 4 个三值权重（1 组完美对应），256-bit SIMD 处理 4 个 FP32（1 组完美对应），512-bit SIMD 处理 8 个 FP16（2 组整除）→ 零 bit shuffle → 查表引擎（BitNet.cpp/T-MAC）将 MUL→ADD
  - **编译框架**：论文未明确说明。
  - **Serving 框架**：论文未明确说明。
  - **硬件架构**：论文未明确说明。
  - **芯片设计**：论文未明确说明。

  方法如何解决 Baseline 缺陷：
  - 缺陷 (a) 存储-SIMD 矛盾：3:4 稀疏的 5-bit 4-way 打包同时实现 1.25 bit/weight 存储效率（优于 2-bit 的 1.58-bit 实际和 1.67-bit 的存储）和 SIMD 友好对齐（4-way pattern 无 bit shuffle），在 Intel i7-14700HX 上实现 148.27 t/s（0.7B）和 45.55 t/s（3B），相比 BitNet I2_S 分别快 12% 和 9%，模型大小减少 ~20%。
  - 缺陷 (b) weight trapping：Arenas 的残差路径 λ_t·X·W 为 deadzone 内权重提供 heterogeneous 梯度信号，防止同质化和低秩坍缩，使 Sherry 在 1.25-bit（比 BitNet 少 25% bits）下平均基准精度反而更高（1B 模型 0.519 vs BitNet 0.483; 3B 模型 0.567 vs BitNet 0.527）。
  - 缺陷 (c) 极低位宽精度保持：Sherry 在 1.25-bit 下仅比 BF16 基线低 3.9%（1B）和 6.9%（3B），与 1.67-bit Tequila 持平甚至略高的同时使用更少 bits。这证明 3:4 稀疏 + Arenas 的组合在 ~1.25 bit 处找到了硬件效率和模型精度的"甜点"。

## ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

- baseline方法是什么？
  Baseline 方法为 SpinQuant、QuaRot、QUIK 等 4-bit 后训练量化方案。以 SpinQuant 为例的全栈执行：
  - **算法层**：使用可学习的旋转矩阵（通过 Cayley 优化）对权重和激活进行旋转，使旋转后的张量对均匀 4-bit 量化更友好，所有通道统一 4-bit。
  - **系统框架层**：基于 HuggingFace Transformers，旋转矩阵可融入相邻权重，避免运行时显式投影开销。使用 GPTQ 做权重优化。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：调用标准 INT4 GEMM kernel（如 CUTLASS）。因所有操作均为统一精度，kernel 选择简单。
  - **硬件架构层**：运行在 NVIDIA A100/RTX 3090 GPU 上，使用 TensorCore。
  Baseline 的核心缺陷：统一 4-bit 量化的误差上界由激活的最大 outliers 决定。旋转虽能部分抑制 outliers，但无法从根本上消除极端值的影响。SpinQuant 在 Meta-Llama-3-8B 上相比 16-bit baseline 仍有约 20% 困惑度退化。QUIK 的 mixed-precision 方案（按 l_∞-norm 选高精度通道）缺乏理论最优性保证。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ResQ 通过 **PCA 驱动的低秩残差混合精度量化** 解决 baseline 缺陷：
  - **算法层**（核心创新）：不依赖启发式 outlier 检测，而是通过 PCA 从理论上找到最小化量化误差的投影基。具体地：对校准激活做 PCA，将特征向量按特征值排列，后 r（=d/8）列对应方差最大的低秩子空间保留 8-bit，前 d-r 列对应低方差的互补子空间量化到 4-bit。Theorem 4.2 从理论上证明 PCA 基选择是最优的——最小化量化误差上界。同时在每个子空间内应用随机旋转（Lemma 4.1 保证旋转后分布近似高斯，进一步降低量化误差）。
  - **系统框架层**：四种投影矩阵（U_A/U_B/U_C/U_D）分别处理不同位置的激活投影：U_A 融入跨 block 边界的权重（后乘 o_proj/down_proj 权重，无运行时开销）；U_B 处理 value 投影并融入 o_proj；U_C 为 key/query 投影，因 RoPE 存在需运行时计算但量化为 8-bit；U_D 用于 FFN 内部激活投影，用 Hadamard 矩阵实现快速变换。权重量化可结合 GPTQ 进一步优化。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：实现 CUTLASS INT4 + INT8 混合精度 GEMM kernel，分别计算低精度和高精度分量的矩阵乘法，结果在 INT32 累加器求和。相比纯 INT4 kernel 仅增加约 14% 延迟。运行时 U_C 投影和 U_D Hadamard 变换均有高效 CUDA kernel 实现。
  - **硬件架构层**：运行在 NVIDIA A100/RTX 3090 GPU 上，利用 TensorCore 加速 INT4/INT8 GEMM。

  对比 baseline 的全栈改进：从一个请求（token 序列）出发，激活 X 进入 decoder block 后，首先经 U_A（已融合到前一层权重中）自动完成投影 → 在注意力块内 query/key 经 U_C 运行时 8-bit 量化投影计算 attention → value 经 U_B 投影后写入 KV cache（4/8-bit 混合精度）→ FFN 内激活经 U_D Hadamard 变换 → 各线性层执行 INT4+INT8 混合精度 GEMM。整个过程将量化误差理论上界最小化，实际达到相比 SpinQuant 4-33% 的 Wikitext 困惑度降低，无需梯度优化（SpinQuant 需 Cayley 优化训练旋转矩阵），且提供通过调节 rank r 实现的 Pareto 最优精度-效率权衡。

## SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

- baseline方法是什么？
  Baseline是GPTQ（3-bit/4-bit group-wise PTQ）和RTN（round-to-nearest uniform quantization）。以LLaMA-7B GPTQ 4-bit在A100上推理为例：
  
  - **算法Pipeline**：输入tokens(2048) → embedding → L层Transformer Block。每Block内：RMSNorm(FP16) → MHA(Q/K/V/O投影+RoPE+Softmax+Attention) → 残差 → RMSNorm(FP16) → FFN(Gate/Up/Down投影+SiLU) → 残差。GPTQ对所有Linear层权重W做4-bit group-wise量化（group_size=128），基于Hessian矩阵 H=2XXᵀ 做逐列OBQ误差补偿：每量化一列，误差通过Cholesky分解的逆Hessian传播到右侧未量化列进行补偿。量化尺度和零点以16-bit存储。
  
  - **系统框架**：GPTQ量化后的PyTorch推理。每个128权重group共享一组scale/zero，dequantize ŵ = scale × (Q - zero)，与FP16 activation执行FP16 matmul。
  
  - **编译框架**：论文未明确说明。
  
  - **Kernel调度**：PyTorch默认矩阵乘法（cuBLAS），无自定义kernel。权重以INT4 packing存储，推理时解包为FP16后计算。Token-by-token生成是memory-bound：batch_size=1时，算术强度极低，瓶颈在DRAM带宽。
  
  - **硬件架构**：NVIDIA A100 GPU（Ampere架构），论文未涉及RTL或模拟器修改。
  
  Baseline的两大缺陷：
  1. **统一精度忽略权重敏感性差异**：GPTQ对所有权重同等处理（同一bit-width, 同一group内部统一scale/zero），但论文分析表明约1%的敏感权重贡献了75%以上的总量化误差。这些高敏感度权重呈现特定的结构模式（行异常值、列异常值、敏感attention heads、rotary embedding pattern、非结构化异常值），但GPTQ的group-wise量化只能以粗粒度group（128权重）补偿误差，无法精确处理离散的敏感权重。
  2. **量化统计量存储开销限制group size下限**：直觉上更小的group size可以提高精度（每个group的scale/zero更适配局部数据分布），但量化统计量（scale+zero）存储开销随group size减小而增大。例如对4-bit权重，group_size=16时统计量开销为2×16/16=2 bits/param，抵消了量化的内存收益，因此传统方法采用较大的group size（128）导致精度不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出SpQR（Sparse-Quantized Representation），包含三个核心机制对应解决baseline缺陷：

  **双层量化（Bilevel Quantization）** 解决缺陷2（统计量存储开销限制group size）：
  - 关键思想：将极小group（β₁=8~16）的scale/zero这些"量化统计量"本身再量化为3-bit，然后对量化统计量的统计量做第二层量化（β₂=16），用16-bit存储最终统计量
  - 平均bits数：b̄ = b_w + (b_s+b_z)/β₁ + 64/(β₁β₂) + 32·r_o。例如b_w=3, b_s=b_z=3, β₁=16, β₂=32 → 统计量开销仅6/16+64/512=0.5 bits/param，使小group size的实际内存开销可控
  - MMQ量化器的min-max要求放宽：去除"max>0, min<0"约束，允许全正/全负group的非整数零点，进一步提升小group下的精度

  **非结构化异常值检测与高精度保留** 解决缺陷1（敏感权重识别并隔离）：
  - 基于Optimal Brain Surgeon框架推导敏感度：s_ij = (w_ij - quant(w_ij))² / (2[H⁻¹]_jj)。该公式捕捉了权重间的相关性——某个权重的高rounding error可被其他权重补偿（通过连续值优化补偿）
  - 异常值检测发生在GPTQ量化过程中（而非预处理）：这样检测的不仅是"初始敏感"权重，还包括量化过程中因误差累积而变得敏感的权重（即能补偿其他权重量化误差的权重）
  - 检测到的高敏感度权重（约1%，τ阈值由binary search确定）保留为16-bit，使得min-max scale计算排除outlier后大幅减小，进一步提升剩余权重的量化精度

  **CSR格式稀疏矩阵乘法GPU Kernel** 解决缺陷1引发的推理效率问题：
  - 虽然1%异常值以非结构化CSR存储，但设计专门的GPU kernel通过tile-based load balancing和row-wise越权内存访问，结合dense-quantized matmul实现了比FP16基线更快的推理速度（20-30%加速）

  全栈执行例子（LLaMA-65B 3-bit SpQR对比GPTQ 4-bit）：
  
  - **算法Pipeline**：校准数据X通过模型前向传播收集每层输入 → 计算Hessian H=2XXᵀ和Cholesky分解Hⁱᶜ。对每层W ∈ R^{d_out×d_in}，逐β₁列组处理：
    1. 在当前列组内检测outliers（leave-one-out error比对）→ 标记O
    2. 排除O对剩余权重拟合3-bit group-wise quantizer（双层：先fit 3-bit scales/zeros，再fit scales-of-scales/zeros-of-zeros）
    3. 量化非outlier权重为3-bit codes
    4. OBQ误差传播（GPTQ风格）到右侧未量化权重
    5. 收集outliers为CSR格式（row-first排序）
    → 输出：Q (3-bit packed), S_q/Z_q (3-bit first-level stats), S_s/Z_s/S_z/Z_z (16-bit second-level stats), W_sparse (CSR outliers)
  
  - **系统框架**：SpQR PyTorch推理代码 + 自研CUDA kernel。权重以custom SpQR格式存储（每256权重block内：256×3-bit codes + 16×3-bit scales/zeros + 4×16-bit statistics），CSR存储1% outliers
  
  - **编译框架**：论文未明确说明
  
  - **Kernel调度**：自研GPU kernel执行SpQR格式推理：
    - Dense部分：Thread block加载block statistics到SRAM→双层反量化→packed weights反量化→与activation做点积
    - Sparse部分：Tile-based load balancing→从CSR加载outlier slice到SRAM→逐row检测→加载列值→sparse dot product
    - Merge两种结果。在A100上batch_size=1时比FP16快20-30%（因为压缩率>3.4x，memory-bound场景下DRAM读取量大幅减少）
  
  - **硬件架构**：NVIDIA A100 GPU，论文未涉及RTL或模拟器修改
  
  效果：SpQR 3.94 avg bits LLaMA-65B WikiText2 PPL=3.68（vs FP16=3.53），GPTQ 4-bit=3.83。SpQR 4-bit将误差较GPTQ减半。在24GB GPU上可运行33B参数模型。

## SqueezeLLM Dense-and-Sparse Quantization

- baseline方法是什么？
  **Baseline是uniform weight-only post-training quantization（以GPTQ为代表）**，该方法对LLM权重采用逐通道或逐组的均匀量化。

  全栈执行例子（GPTQ 3-bit uniform quantization with activation ordering, LLaMA-7B, 单batch推理）：

  - **算法Pipeline**: 校准数据（128个C4样本）逐层前传→收集每层输入activation statistics→逐列OPTQ贪心量化：对每列权重，用Hessian信息（基于输入activation的outer product, H=2XX^T）做quantize-and-compensate——量化当前列→计算量化误差→把误差按H的逆传播到右侧未量化列→重复直到全矩阵量化完成。Uniform格式：每128个连续元素共享一个scale+zero（group-wise uniform scaling factor, g128）。这实际上是最小化层间输出activation扰动（layer-wise L2: ||WX - W_QX||^2），而非最终模型loss。

  - **系统框架（Serving）**: 论文未明确说明修改Serving框架。推理使用PyTorch + 自研/社区CUDA kernel加载quantized权重，以FP16执行dequant + matvec。

  - **编译框架**: 论文未明确说明。

  - **Kernel调度**: Uniform quant kernel：加载packed int3 weights→按group读取scale/zero→dequantize为FP16→与FP16 activation向量做matvec。**问题**：GPTQ with activation ordering引入permutation，使得同一channel的权重分布在不同的group中，需要不同scaling factor（通过group index间接访问）。在GPU上这种scattered memory access破坏内存合并（coalesced access）→导致latency从1.4s暴涨到13.7s（LLaMA-7B, A6000, 128 tokens）。

  - **硬件架构**: NVIDIA A6000/A100 GPU。GPU memory bandwidth是瓶颈（A6000: 768 GB/s内存带宽 vs 222 TFLOPS计算吞吐，带宽仅为算力的~0.3%）。LLM单batch推理是memory-bound——每个权重加载后仅参与一次乘加（arithmetic intensity极低）。

  Baseline的根本性缺陷：
  1. **Uniform quantization在非均匀权重分布下浪费量化分辨率**：LLM权重分布高度非均匀（99.9%的值集中在~10%的范围内），uniform bin allocation将大量bin浪费在稀疏分布区域，对密集区域的敏感权重分辨不足。
  2. **Layer-wise优化目标与end-to-end loss不一致**：GPTQ最小化||WX-W_QX||（层间输出扰动），而SqueezeLLM证明直接最小化final loss扰动（Hessian-weighted objective）显著更优（D.4消融实验：LLaMA-7B 3-bit PPL gap ~0.3）。
  3. **Outliers膨胀量化范围**：极少数outlier values使整个量化range扩大10x，严重降低quantization resolution。
  4. **Grouping不是outlier问题的直接解决方案**：GPTQ/AWQ用grouping (g128)间接隔离outliers→增加storage overhead（per-group scale+zero）→且在非均匀量化下overhead更严重（需per-group LUT）。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **SqueezeLLM = Sensitivity-based non-uniform quantization + Dense-and-Sparse decomposition**。两个技术逐一解决baseline缺陷：

  全栈执行例子（SqueezeLLM 3-bit + 0.45% sparsity, LLaMA-7B, 单batch推理）：

  - **算法Pipeline**:
    **缺陷①→方案**：将均匀量化替换为sensitivity-weighted k-means非均匀量化。优化目标从min||W-W_Q||^2改为min Σ F_ii(w_i-Q(w_i))^2，其中F_ii是Fisher信息矩阵对角线（≈Hessian对角线），通过calibration数据集（仅需10-100样本）的一次梯度前反向计算获得（7B: 0.3min on A100）。Weighted k-means自动将centroid向高敏感度权重聚拢（Fig. 3直观展示：均匀量化的8个bin均匀分布，而sensitivity-based的8个bin在敏感值区域更密集）→3-bit LLaMA-7B PPL从uniform的28.26降至7.75。
    **缺陷②→方案**：直接minimize final loss perturbation（Eq. 4-6），通过Taylor展开和Fisher近似将二阶Hessian信息融入k-means权重。相比layer-wise objective（如AWQ使用activation magnitude作为importance），final-loss-based方法在所有sparsity level下PPL优约0.3（D.4）。
    **缺陷③→方案**：Dense-and-Sparse decomposition——提取0.4% outlier (百分位阈值) + 0.05%最敏感值(按Fisher排名)作为稀疏矩阵S（CSR格式, FP16），剩下99.55%的dense矩阵D值域压缩约10x→非均匀量化的分辨率大幅提升→3-bit PPL从7.75再降至7.56。
    **缺陷④→方案**：直接用sparse component隔离outliers+sensitive values，而非用grouping间接处理。D.3消融实验证明：pure Dense-and-Sparse decomposition在所有model size下PPL优于grouping (g512/g1024)或grouping+sparsity hybrid方案。在非均匀量化下grouping需存per-group LUT（overhead巨大），而sparsity方案overhead可控（仅0.24 bit for 0.45% sparsity）。

  - **系统框架（Serving）**: 论文未明确说明修改Serving框架。推理时每个Linear层执行两个融合的kernel调用：LUT dequant matvec + balanced CSR SpMV。Dense和Sparse kernel在单次launch中融合，无额外result sum kernel开销。

  - **编译框架**: 论文未明确说明。

  - **Kernel调度**: 
    - **Dense kernel**: LUT-based非均匀dequant+matvec。压缩格式存3-bit indices→从per-channel LUT (8个FP16 centroid)查表获得真实FP16 weight→与activation做FP16内积。LUT overhead极小（延迟仅比uniform高~7%，即1.4→1.5s），但换来PPL 9.55→7.75的巨大提升。
    - **Sparse kernel**: Balanced CSR SpMV（10 nz/thread）。对比标准CSR kernel (thread-per-row)在处理skewed sparsity distribution（Fig. C.1: 少数channel含大量nonzeros）时的严重负载不均衡（3.9s vs 1.7s for 7B），balanced kernel通过固定per-thread nonzero数实现workload均衡（使用atomicAdd合并同一行的多线程结果）。
    - **整体**: 0.45% sparsity时延迟1.7s vs FP16的3.2s（1.9x加速），PTQ 9.55→7.56。相比之下GPTQ g128因permutation引入的scattered memory access降速到13.7s→实际上不可用。

  - **硬件架构**: NVIDIA A6000 GPU。Roofline model验证：LLM单batch生成推理是memory-bound问题（arithmetic intensity极低），因此LUT查表的overhead（少量额外计算）完全被memory bandwidth瓶颈掩盖→理论加速≈压缩比。实际speedup: 1.9x (3-bit) / 1.7x (4-bit) for 0.45% sparsity。

  效果总结（LLaMA-7B 3-bit, C4 perplexity）：
  | 方法 | PPL | Speedup |
  |------|-----|---------|
  | FP16 Baseline | 7.08 | 1.0x |
  | GPTQ uniform (no group) | 9.55 | 2.3x |
  | GPTQ uniform (g128, reorder) | 7.89 | 0.2x (unusable) |
  | AWQ (g128) | 7.90 | 2.0x |
  | **SqueezeLLM dense-only** | **7.75** | **2.1x** |
  | **SqueezeLLM 0.45% sparse** | **7.56** | **1.9x** |

  核心创新映射：
  - Sensitivity-based non-uniform quantization → 解决了uniform在非均匀分布+memory-bound场景下的次优性
  - Fisher-weighted k-means → 将final loss sensitivity融入量化，优于layer-wise perturbation minimization
  - Dense-and-Sparse decomposition → 直接解决outlier问题，比grouping更高效且与non-uniform天然兼容
  - Balanced sparse kernel → 使sparsity的latency overhead可控（<15%），实现practical speedup


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

## DiJiang: Efficient Large Language Models through Compact Kernelization

- baseline方法是什么？
  Vanilla Transformer的自注意力机制：Attention(Q,K,V) = softmax(QK^T)V，计算复杂度O(n²d)，其中n为token数量、d为head维度。对于长序列，二次复杂度导致训练和推理成本急剧增长。以LLaMA2-7B为例，完整训练需约82,432 GPU-hours和~36 MWh电力。虽然Performer等方法通过Positive Random Features (PRF)可实现线性注意力，但其Monte Carlo采样近似效率仅为O(1/m^{-0.5})，需要m >> d才能保持性能，极大削弱了线性注意力带来的加速收益。其他线性Transformer方法（Linformer, Cosformer, RetNet）在fine-tuning场景下精度损失严重（Pythia-410M fine-tuning中最高仅Performer的0.4183 vs 原始0.454）。

  全栈执行例子（一条token序列通过vanilla Transformer推理）：
  - 算法Pipeline：对于每个token i，计算 o_i = Σ_j exp(q_i·k_j^T)/Z * v_j，其中对每对(i,j)做d维内积 → Softmax归一化 → 加权求和，O(n²d)复杂度。
  - Serving调度：论文未明确说明（依赖标准推理框架如HuggingFace Transformers/PyTorch）。
  - 编译框架：论文未明确说明（标准PyTorch eager/graph模式）。
  - Kernel调度：标准cuBLAS/cuDNN GEMM和Softmax kernel，未做针对性优化。
  - 硬件架构：NVIDIA A800 GPU，标准CUDA core和Tensor Core执行矩阵乘法。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  DiJiang使用Frequency Domain Kernelization (FKA)将vanilla Transformer转换为线性复杂度模型，通过三个递进创新解决baseline痛点：

  **(1) Quasi-Monte Carlo (QMC)替代Monte Carlo**：传统PRF用Monte Carlo采样近似Gaussian核 e^{qk^T}，收敛速度仅O(1/m^{-0.5})。DiJiang基于Bochner定理将核函数转为球面积分，用渐近均匀点集（QMC）替代随机采样，收敛速度提升至O(1/m)，使m=d即可保持近似精度。
  
  **(2) 加权QMC (WPFF)**：进一步引入可学习权重D对标采样点进行加权，通过求解凸优化问题（最小化Paley-Wiener空间的discrepancy度量）获得最优权重配置，理论上证明WPFF的积分估计误差上界不大于PFF。
  
  **(3) DCT频域映射 (WDCF)**：将随机投影替换为确定性DCT系数矩阵C进行频域变换，利用DCT的快速算法（O(log m) vs O(m)）和能量集中特性（稀疏表示），在实数域操作无需处理复数，比FFT更高效且硬件友好。

  全栈执行例子（一条token序列通过DiJiang FKA推理）：
  - 算法Pipeline：给定Q,K,V ∈ R^{n×d}，先用DCT系数矩阵C计算频域映射 φ_WDCF(x) = D⊙exp(T·C·x^T)，再按 φ(Q)·φ(K)^T·V = φ(Q)×(φ(K)^T×V) 计算，先乘K^T×V得O(nmd)再乘φ(Q)。当m=d时总复杂度O(nd²)，序列长度n与计算量成线性关系（而非二次）。
  - Serving调度：论文未明确说明（继承预训练模型的推理管道）。
  - 编译框架：论文未明确说明。
  - Kernel调度：借鉴RetNet的高效推理实现，利用线性注意力可合并K和V计算的特点，实现O(1)每token推理开销；但论文未描述具体kernel实现细节。
  - 硬件架构：NVIDIA A800 GPU，利用DCT的快速算法（可通过FFT-like蝶形结构实现）在GPU上高效执行频域变换。

  **核心对比**：
  - 训练成本：DiJiang-7B仅需40B tokens（LLaMA2-7B用2000B tokens，约1/50）；DiJiang-410M训练6.6天 vs Pythia-410M 105.8天（约1/16）。
  - 推理速度：DiJiang-2.8B推理284 tokens/s vs Pythia-2.8B 34 tokens/s（约8.4×），且随token长度增加显存和延迟不增长（线性复杂度优势）。
  - 精度保持：DiJiang-410M平均0.4567 vs Pythia-410M 0.454（几乎无损）；DiJiang-7B平均0.557 vs LLaMA2-7B 0.565。

## PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

- baseline方法是什么？
  Baseline为PanGu-π-1B原始模型（Wang et al., 2023），以及当时SOTA的开源小模型系列（TinyLLaMA-1.1B, MobileLLaMA-1.4B/2.7B, Sheared-LLaMA-1.3B, Phi-1.3B/Phi2-2.7B, Open-LLaMA-3B, Qwen-1.8B等）。Baseline模型在训练策略上直接沿用了大语言模型的开发方法：使用大模型的tokenizer（100k+词表）、LLaMA标准架构（12层/2048宽）、随机初始化或简单裁剪、单轮训练（所有数据仅使用一次）。

  全栈执行例子（以PanGu-π-1B baseline为例）：
  - 算法Pipeline：100k BPE tokenizer → 12层/2048宽/expansion_rate=2.77 LLaMA-like Transformer → 随机初始化参数N(0, σ²) → AdamW + Cosine LR训练1.6T tokens一轮 → 在OpenCompass十个benchmark上评估
  - 系统框架：PyTorch，Huawei Ascend 910集群训练
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明

  Baseline核心缺陷：
  1. **Tokenizer参数冗余**：大模型tokenizer（100k词汇）在小模型中embedding+head层占比高达38.19%，大量参数浪费在低频词汇的表征上（bottom 50k+ vocab仅覆盖<3%语料），挤压了模型本体的表达能力。
  2. **架构配置不匹配**：小模型直接沿用大模型的宽-浅架构（12层/2048宽），未针对1B参数的约束探索depth-width-expansion的最佳配比，导致性能欠优。
  3. **初始化信息缺失**：随机初始化使小模型从零开始学习表征，无法利用大模型已学到的强表征能力，收敛慢且最终性能受限。
  4. **严重灾难性遗忘**：小模型容量有限，在1.6T tokens的单轮训练中，后期数据会严重覆盖早期学到的知识，表现为"之前seen的数据loss大幅上升"的forgetting现象。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过四项核心设计系统性地解决了小模型训练的四个痛点：

  - **Idea 1（Compact Tokenizer）** 解决缺陷1：统计分析发现top-48k词汇覆盖97.86%训练语料（长尾效应），将词表从100k压缩至48k，使embedding+head层参数占比从38.19%降至18.07%，释放~20%参数给Transformer本体。本质是识别并移除对语义覆盖贡献极低的词汇冗余，在词表覆盖率与参数效率间找到最优平衡点。

  - **Idea 2（Architecture Tweak: Depth > Width）** 解决缺陷2：通过网格搜索和Spearman相关性分析发现depth是小模型性能的首要决定因素（Spearmanr=0.528），expansion rate几乎无影响。在1B参数量约束下，将架构从12层/2048宽改为21层/1792宽，性能从Avg=42.41提升至46.53（+4.12，含tokenizer改进）。深层窄架构以推理速度为代价换取更强的序列建模能力——符合Transformer深度增加有利于学习层次化特征的直觉。

  - **Idea 3（Parameter Inheritance with Learnable Masks）** 解决缺陷3：从大模型（PanGu-π-7B）通过数据驱动的learnable binary masks继承关键参数。Layer Selection阶段发现首尾层关键、中间层冗余的普适规律（在LLaMA2/InternLM/PanGu-π四个大模型上验证），据此移除中间冗余层。Intra-layer阶段用Gumbel-Sigmoid可微二值mask自动学习重要神经元，相比L1/L2/Taylor启发式标准显著提升（Learnable: 48.08 vs Taylor: 47.90 vs L2: 47.00）。核心思想是将大模型的表征能力"蒸馏"到小模型初始化中，使小模型从一开始就站在大模型的肩膀上。

  - **Idea 4（Multi-round Training with Loss-guided Data Sampling）** 解决缺陷4：发现小模型在单轮训练末尾，early data的loss已从训练时的低值大幅反弹（forgetting证据）。提出第二轮训练：按p_i ∝ exp(loss_i)的概率采样数据（困难样本高频出现），50%采样率可达到接近全量二轮训练的效果。两轮训练Avg从51.61提升至54.46（+2.85），三轮开始饱和，在性能与训练成本间取得平衡。

  全栈执行例子（PanGu-π-1B Pro）：
  - 算法Pipeline：
    Step 1: 频率分析 → BPE训练48k紧凑tokenizer
    Step 2: Depth=21/Width=1792/Expansion=2.77 架构
    Step 3: PanGu-π-7B → Layer Selection（移除中间冗余层）→ Learnable Mask训练（Gumbel-Sigmoid binarization）→ 提取子矩阵作为初始化
    Step 4: Round 1全量训练1.6T tokens（AdamW, LR=2e-4, bs=2M, cosine decay）→ 记录per-batch loss → Round 2按p_i=exp(l_i)/Σexp采样50%数据继续训练 → PanGu-π-1B Pro
    Step 5: OpenCompass十项benchmark评估 → Avg=51.28（vs baseline 42.41, +8.87）
  - 系统框架：PyTorch LLaMA-like架构，Huawei Ascend 910集群训练
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明

  各组件增量贡献（Figure 1，Avg性能提升）：
  PanGu-π-1B (42.41) → +Compact Tokenizer (44.11) → +Architecture Tweak (46.53) → +Parameter Inheritance (49.79) → +Multi-round Training (51.28)
  其中Parameter Inheritance贡献最大（+3.26），Multi-round Training次之（+1.49），两者合计贡献整个流程一半以上的提升。


## UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

- baseline方法是什么？
  **Baseline 1（结构化剪枝）**：MoDeGPT（Lin et al., 2025）和SVD-LLM（Wang et al., 2025b）。
  - MoDeGPT：使用伪逆（Moore-Penrose inverse）在FP64精度下对MLP中间激活的通道相关性矩阵求解，排序权重通道以最小化剪枝误差。每个剪枝率需要重新计算伪逆（因为(W')^† ≠ (W^†)'），复杂度O(n³)，在Llama-3.1-8B的D_int=14336矩阵上伪逆耗时20.58分钟。对Qwen-2.5-7B（D_int=18944）因病态条件矩阵导致严重精度下降。
  - SVD-LLM：对权重矩阵做SVD分解后截断特征值，每次仅支持单一剪枝率，需为不同剪枝率独立运行。截断后需FT恢复精度。对于多个压缩率需多次独立运行（O(n)复杂度）。

  **Baseline 2（PTQ）**：TRT-AWQ（TensorRT-Model-Optimizer中的AWQ实现）和TAO-HQQ（TorchAO中的HQQ实现）。均为W4A16 PTQ框架，但仅支持固定4-bit量化，不支持结构化剪枝，且embedding/output层保持FP16（占用更大内存）。
  
  **全栈执行例子（Baseline MoDeGPT + AWQ在Llama-3.1-8B边缘部署）**：
  - 算法层：云上对MLP用伪逆排序（FP64, 20min/层），固定25%剪枝率 → 权重截断后精度下降（64.9% avg acc）→ GPTQ量化到W4A16。
  - 系统框架层：每个剪枝率需单独跑完整流程，生成多个模型副本存储（FP16 25%剪枝: 13.9GB per variant）。
  - 编译框架层：论文未明确说明。
  - Kernel调度层：使用标准INT4 GEMM kernel（Marlin），无融合RoPE优化。
  - 硬件架构层：部署到Nano 8G时仅使用固定尺寸的量化模型，无法根据设备当前内存使用动态调整。若设备负载高导致可用内存不足，需要换用更小模型或重新压缩。
  - 痛点总结：①伪逆计算慢且需FP64；②一个流程只产出一个压缩率；③无设备端自适应能力；④PTQ框架不支持SSM/Hybrid模型。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **UniQL方案**：用ridge leverage scores替代伪逆（复杂度降至O(n²)），用量化感知SVD融合Σ到U来降低量化误差，用masked LoRA一次训练多剪枝率，实现云侧一次压缩+设备侧自适应剪枝。

  **缺陷→方案映射**：
  1. 伪逆慢且不稳定 → 伪逆无关的ridge leverage score排序（Algorithm 1），在BF16下完成，Llama-3.1-8B MLP排序从7h03m降至19min（22×加速），且对Qwen-2.5-7B的大D_int/小D_h比不再失效。
  2. 每个压缩率需独立运行 → 一次masked LoRA微调（随机采样P_t），产出单个模型副本支持0%-35%所有剪枝率。压缩时间O(1) vs baseline O(n)。
  3. SVD分解在4-bit下引入量化误差 → 量化感知SVD（QSVD）：将W=UΣV分解后把Σ融合到U（W=(UΣ)V），使σ_i成为每列的量化scaling factor，避免长尾特征值被量化截断，4-bit 25%剪枝下精度提升7.5%。
  4. 不支持SSM/Hybrid模型 → 状态感知SSM权重排序：B-C排序考虑输入依赖的离散化广播外积，z-x-o排序从SSM状态H收集相关性。
  5. 设备端无自适应能力 → 部署INT4全量化模型（head-to-toe 4-bit，含embedding/output层），设备端在线解包→剪枝通道→重打包，支持按当前系统负载动态选择0-35%剪枝率。

  **全栈执行例子（UniQL端到端流程）**：
  - 算法层：校准集（Alpaca 128 samples, seq_len=2048）→ MLP ridge leverage scores排序 + MHSA量化感知SVD + Mamba状态感知排序 → 所有模块已排序 → Masked LoRA（r=8, Alpaca, 5 epochs, 每步随机采样P_t）→ GPTQ W4A16全局量化（含embedding/output层）→ 单个4-bit模型文件产出。
  - 系统框架层：单次云端压缩（A6000, 7h43m总耗时含FT+PTQ for Llama-3.1-8B），产出模型4.1GB → 推送到边缘设备。设备端无额外训练/压缩开销，仅做轻量级通道裁剪。
  - 编译框架层：论文未明确说明。
  - Kernel调度层：融合RoPE kernel（对称索引gather+slicing+旋转在单kernel完成，减少10%延迟）→ 设备端在线INT4解包→裁剪→重打包→送入Marlin 4-bit GEMM。Nano 8G上Qwen-2.5-7B的TPOT从TAO-HQQ的133.6ms降至77.2ms（1.7×）。
  - 硬件架构层：Jetson Orin Nano 8G统一内存。当OS报告高内存压力时，应用层触发p=25%或35%剪枝，模型从4.1GB降至3.2GB或2.8GB；当资源充足时p=0%，享受最高精度但最小延迟（2.7×-3.4× throughput vs FP16）。
  - 关键量化效果：Llama-3.1-8B在15%剪枝+4-bit下维持71.4% avg acc（仅比FP16全精度低2.6%），同时模型尺寸4.7×压缩（16GB→3.4GB），生成吞吐量3.4×提升。

