## Learned Step Size Quantization (LSQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learned Step Size Quantization（LSQ，学习步长量化）是 Esser et al. (ICLR 2020) 提出的 QAT 方法。核心创新：将量化步长 s 作为可学习参数，通过梯度下降与权重一同优化。公式：v̄ = ⌊clip(v/s, -Q_N, Q_P)⌉, v_q = v̄ × s。与传统固定步长方法不同，LSQ 让每层自动学习最优量化粒度。梯度缩放策略：grad_scale ∝ 1/√(Q_P × n_features)，平衡不同大小层之间的步长更新速率。支持 per-tensor 和 per-channel 两种粒度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QT-DoG 中 LSQ per-channel 量化的关键计算：
```
w_scaled = w / s              # s: [out_c, 1, 1, 1]
w_bar = round(clip(w_scaled, -Q_N, Q_P))  # Q_N=2^(b-1), STE through round
w_q = w_bar * s               # 量化权重
# 反向梯度：∂L/∂w = STE(w_bar in range) ? ∂L/∂w_q : 0
#           ∂L/∂s = ∂L/∂w_q * (w_bar - w_scaled)
```
QT-DoG 消融：channel-wise LSQ (87.8% PACS OOD) > layer-wise LSQ (86.3%) > no quant (84.7%)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LSQ 的 PyTorch 实现在 Brevitas、AIMET 等量化库中可用。超参：梯度缩放因子 g = 1/sqrt(Q_P x N)；步长初始化为 2*mean(|w|)/sqrt(Q_P)；支持 signed/unsigned 量化。LSQ+（Bhalgat et al., CVPR 2020）增加了可学习零点偏移。QT-DoG 使用 7-bit LSQ 做 per-channel 权重量化，在 2000 步后启动。

在 Task-Specific ZSQ for Object Detection 中，LSQ 被扩展用于目标检测网络的 QAT：per-tensor symmetric quantization 对权重和激活均使用 LSQ，量化器附加到除首尾层外的所有内部层。QAT 使用 Adam 优化器（YOLOv5 lr=1e-4，YOLO11 lr=1e-5，CNN Mask R-CNN lr=1e-4，Transformer Mask R-CNN lr=1e-6），量化 scale 因子通过反向传播联合学习。实验表明 LSQ 在极低校准集（2k vs 120k full）下效果急剧退化（YOLOv5-s W6A6: LSQ full 31.5% vs LSQ 2k 28.9%），但结合 task-specific 合成数据和蒸馏后大幅超越 full-data LSQ（YOLOv5-l W6A6: 45.1% vs LSQ full 43.3%）。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization
- Scheduling Weight Transitions for Quantization-Aware Training
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection
