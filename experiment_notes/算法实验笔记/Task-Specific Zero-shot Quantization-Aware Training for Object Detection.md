## Task-Specific Zero-shot Quantization-Aware Training for Object Detection

- 属于算法pipeline的实现是什么？实验比较什么？
  提出首个面向目标检测的task-specific Zero-shot Quantization（ZSQ）框架，包含两个阶段：(1) **Task-Specific Calibration Set Synthesis**：使用Adaptive Label Sampling从预训练检测网络中以零样本方式重建目标类别、位置和尺寸分布，结合task-specific检测损失L_detect（含L_category、L_box、L_conf）与task-agnostic先验损失L_prior（BNS对齐或Patch Similarity Entropy）合成带标注的校准集；(2) **Task-Specific QAT with Distillation**：联合KL散度知识蒸馏（L_KD）、特征级MSE蒸馏（L_feat）和task-specific检测训练损失（L_detect）微调量化网络。实验在W8A8/W6A6/W4A8/W5A5/W4A4多种位宽下比较：YOLOv5-s/m/l、YOLO11-s/m/l、CNN-backbone Mask R-CNN、Swin-T/S Transformer-backbone Mask R-CNN。对比方法包括LSQ、LSQ+（real-data QAT）以及Genie、ZeroQ（task-agnostic ZSQ），均在MS-COCO 2017和Pascal VOC验证集上用mAP/mAP50评估。

- 硬件平台是什么，配置是什么。
  YOLOv5/YOLO11实验：2× NVIDIA GeForce RTX 4090 GPU；Mask R-CNN实验：4 GPU；ViT实验：8 GPU。实现基于PyTorch框架。

- 模型是什么。数据集和bench分别是什么。
  模型：YOLOv5-s/m/l、YOLO11-s/m/l（单阶段检测器）；Mask R-CNN + ResNet backbone（CNN两阶段检测器）；Mask R-CNN + Swin-T/S backbone（Transformer两阶段检测器）。数据集与Bench：MS-COCO 2017验证集（mAP/mAP50）、Pascal VOC验证集（mAP）。所有模型使用预训练FP32权重作为teacher初始化。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/DFQ-Dojo/dfq-toolkit
  
  算法Pipeline（以YOLOv5-s W6A6 MS-COCO为例）：
  
  **Stage I — Task-Specific Calibration Set Synthesis：**
  1. 初始化输入x ∈ R^{N×3×160×160}，每个像素从高斯噪声N(0,1)采样。随机生成单目标标签y（category ∼ U(0,C)，bbox中心∼U(W/2,1-W/2)，bbox宽高∼U(0.2,0.8)）。
  2. Adaptive Label Sampling循环（Algorithm 1）：每固定间隔用预训练teacher ϕ(θ)对当前x做前向推理 → 取conf > conf_thresh的高置信度预测作为new_tgts → 计算IOU(new_tgts, 当前tgts) → 添加不与现有标签重叠的新标签 → 移除未被teacher检测到的旧标签 → 确保每张图至少保留一个标签。
  3. 固定采样得到的标签y，重新初始化高斯噪声x ∈ R^{N×3×640×640}，用task-specific损失优化：
     L_total = α_prior · L_prior(x) + α_detect · L_detect(ϕ(x), y) + L_reg(x)
     其中L_prior为BNS alignment loss（CNN模型）或Patch Similarity Entropy loss（Transformer模型），L_detect = L_category + L_box + L_conf，L_reg = α_TV·L_TV + α_l2·||x||₂²。
  4. 优化2500次迭代（YOLOv5），Adam优化器，lr=1e-2，余弦退火，使用Cutout数据增强。生成2k张合成校准样本。

  **Stage II — QAT with Task-Specific Distillation：**
  1. 对全精度网络ϕ(θ)的所有内部层（除首尾层外）附加LSQ量化器，使用per-tensor symmetric quantization，量化公式：w_int = clip(⌊w_fp/s⌉, -2^{b-1}, 2^{b-1}-1)，ŵ_fp = w_int × s。
  2. 对每个合成样本(ẍ_i, ŷ_i)计算三项损失：
     - L_KD = (τ²/N)·Σ KL(z^F(ẍ_i;θ), z^Q(ẍ_i;θ'))：预测匹配KL散度蒸馏，τ为温度
     - L_feat = (1/(NL))·Σ||f_l^F(ẍ_i;θ) - f_l^Q(ẍ_i;θ')||₂²：特征级MSE蒸馏，L为蒸馏层数
     - L_detect = L_category + L_box + L_conf：task-specific检测损失
  3. 总损失：L^Q = β_KL·L_KD + β_feat·L_feat + β_detect·L_detect
  4. Adam优化器训练QAT，YOLOv5 lr=1e-4，超参{β_detect, β_KL, β_feat} = {0.04, 0.1, 1.0}。量化scale因子s通过反向传播联合学习。
  
  关键数值结果：YOLOv5-l W6A6 mAP=45.1%（超越full-data LSQ 43.3%达+1.8pp），使用仅1/60训练数据（2k vs 120k）；收敛速度可达LSQ的16×。
