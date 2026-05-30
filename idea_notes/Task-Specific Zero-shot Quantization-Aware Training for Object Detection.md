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
