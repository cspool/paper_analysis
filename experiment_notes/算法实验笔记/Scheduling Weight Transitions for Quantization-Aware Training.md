## Scheduling Weight Transitions for Quantization-Aware Training

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出了一种 Transition Rate（TR）调度技术，替代 QAT 中传统的学习率（LR）调度。核心实现是：在每个训练迭代中计算量化权重的 TR（发生离散级别变化的权重占比），用指数移动平均估计 running TR，然后通过 Transition-Adaptive Learning Rate（TALR）自适应地调整潜权重的更新步长，使得 running TR 匹配目标 TR。实验比较：（1）plain optimizer（SGD/Adam/AdamW，使用传统 LR 调度）vs 论文方法的 variants（SGDT/AdamT/AdamWT，使用 TR 调度），在 ImageNet、CIFAR-10/100、MS COCO 上的分类/检测精度对比；（2）不同类型调度器（step decay vs cosine annealing）下 plain vs TR 调度器的鲁棒性对比；（3）不同优化器（SGD, Adam, NAdam, Adamax, AdamW, RMSProp, Adagrad）下 TR 调度的泛化能力。

- 硬件平台是什么，配置是什么。
  4 × NVIDIA A5000 GPU（ImageNet 训练用时测量，Table S7）。CIFAR 实验使用论文未具体指明 GPU 型号的训练平台。

- 模型是什么。数据集和bench分别是什么。
  模型：MobileNetV2、ResNet-18/20/34/50、ReActNet-18（binary quantization specialized architecture）、DeiT-T/S（ViT-based）。数据集与 Benchmark：ImageNet（ILSVRC2012，top-1 validation accuracy）、CIFAR-10/100（top-1 test accuracy）、MS COCO 2017（RetinaNet 目标检测，AP/AP50/AP75/APS/APM/APL）。量化位宽覆盖：W1A1（binary）、W2A2、W3A3、W4A4。所有模型从 pretrained full-precision 权重初始化，第一层和最后一层不量化。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://cvlab.yonsei.ac.kr/projects/TRS/

  **算法 Pipeline：TR 调度 QAT 的每一步迭代（来自 Algorithm 1）：**

  Step 1 — 正向传播：潜权重 w 经 normalizer `w_n = clip(γ·w/s, α, β)` 归一化，再经 discretizer `w_d = round(w_n)` 转为离散整数值，最后经 fixed de-normalizer `w_q = w_d/γ` 输出量化权重（γ、α、β 为位宽常量，s 为可学习的 scale 参数，TR 调度时 weight quantizer 的 s 固定不变）。

  Step 2 — 计算当前 TR `k^t = Σᵢ I[w_d^t(i) ≠ w_d^{t-1}(i)] / N`，即发生离散级别变化的量化权重占总权重的比例。

  Step 3 — 估计 running TR `K^t = m·K^{t-1} + (1-m)·k^t`，使用 momentum m=0.99。

  Step 4 — 调整 TALR `U^t = max(0, U^{t-1} + η(R^t - K^t))`，其中 R^t 是目标 TR（由 scheduler 如 cosine decay 衰减），η = U^0（初始 TALR 值）。

  Step 5 — 逆/反向传播：gradient term g^t 用 STE 通过 discretizer 回传到潜权重，g^t 取决于优化器类型（SGD 用一阶矩，Adam 用动量归一化梯度）。

  Step 6 — 更新潜权重 `w^{t+1} = w^t - U^t·g^t`。注意这里用的是 TALR U^t 而非固定 LR。

  关键设计：初始 target TR = λ·√b_w，其中 λ 是 TR factor（超参，如 5e-3），b_w 是权重量化位宽；target TR 按 cosine scheduler 衰减到零；η 等于初始 TALR，使调整步长与初始值成比例。量化器基于修改版 LSQ（multi-bit）和 ReActNet（binary）。
