## Quantization-Aware Training (QAT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware Training（QAT，量化感知训练）是一种在训练过程中模拟量化效果的方法。与 PTQ 不同，QAT 在训练时就在前向传播中插入伪量化操作，保持权重为浮点同时模拟低位宽推理的精度损失；反向传播使用 STE 将梯度穿过伪量化操作，同时优化模型权重和量化器参数。QAT 通常能取得比 PTQ 更好的精度（尤其是在极低位宽如 2-bit/3-bit），因为权重可以通过训练自适应量化的影响。但其代价是需要完整的训练流程，计算成本可能超过 FP 模型的原始训练。2DQuant 论文选择 PTQ 路线就是因为 QAT 对 SR 任务来说"训练成本过高甚至超过 FP 模型训练"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QAT 的典型训练流程与标准训练相似，但插入了量化模拟：
```
for epoch in range(N_epochs):
    for x, y in train_loader:
        # 前向：伪量化权重和激活
        w_q = fake_quantize(w, l_w, u_w, bit)
        x_q = fake_quantize(x, l_x, u_x, bit)
        y_pred = model_forward_with_quantized_tensors(w_q, x_q)
        loss = criterion(y_pred, y)
        # 反向：STE 穿过量化操作
        loss.backward()  # ∂L/∂w 通过 STE 近似获得
        optimizer.step()  # 同时更新权重 w 和 clip bounds (l,u)
```
SR 领域的代表性 QAT 方法：PAMS（ECCV 2020，用可训练的截断参数动态确定量化范围上界）、DAQ（WACV 2022，channel-wise 分布感知量化）、CADyQ（ECCV 2022，内容感知的动态位宽分配）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中，QAT 通过 `torch.quantization.prepare_qat()` 将伪量化模块插入模型，训练后通过 `torch.quantization.convert()` 转换为 INT 推理模型。典型步骤：(1) 在模型定义中标记量化位置（`torch.quantization.QuantStub/DeQuantStub`）；(2) `prepare_qat(model)` 插入 FakeQuantize 模块；(3) 正常训练循环；(4) `convert(model)` 将伪量化替换为真正的 INT 算子。在 HuggingFace 生态中，`transformers` 库也支持通过 `BitsAndBytesConfig` 或 `Quanto` 进行 QAT。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

在 Task-Specific ZSQ for Object Detection 中，QAT 被用于目标检测网络的零样本量化微调。QAT 阶段的输入是 Stage I 生成的 2k 张 task-specific 合成校准集（仅真实训练数据的 1/60），使用 LSQ 将 per-tensor symmetric quantization 附加到除首尾层外的所有内部层。总损失 L^Q = beta_KL*L_KD + beta_feat*L_feat + beta_detect*L_detect，其中 L_KD 为 KL 散度预测蒸馏，L_feat 为 MSE 特征蒸馏，L_detect 为检测任务损失（L_category + L_box + L_conf）。Adam 优化器，YOLOv5 lr=1e-4。该方法在 W8A8 YOLOv5-l 上达到 47.3% mAP（超越 full-data LSQ 46.0%），收敛速度可达 full-data LSQ 的 16x。注意该方法属于 ZSQ 范畴——不使用真实图像，仅使用合成校准集和合成标签。

在 Squat 中，QAT 被用于SLM（LLaMA-58M、GPT2-97M）的粗粒度层级别（layer-wise）量化训练。与通常GPU上的channel-wise/token-wise细粒度QAT不同，Squat坚持每层单scale因子的粗粒度量化以兼容移动端SIMD硬件。QAT训练采用FP16教师蒸馏 + 熵损失L_E + 分布损失L_D，通过Token自适应量化动态分配位宽（4-bit或8-bit），并使用STE反向传播。结果：W4A8 BLiMP精度仅↓0.3% vs FP16，移动端加速2.37×。

在 EfficientQAT 中，QAT 通过两阶段策略实现高效量化感知训练：(1) Block-AP：逐block训练所有权重和量化参数(W, s, z)，仅需4096样本/2 epoch即可收敛；(2) E2E-QP：冻结量化权重，仅端到端训练步长s（占参数约1.6%），实现跨block交互。该方案使得Llama-2-70B的2-bit QAT可在单张A100-80GB GPU上41小时内完成（vs LLM-QAT需900h、DB-LLM需82h），训练内存仅34.2GB。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- QT-DoG Quantization-Aware Training for Domain Generalization

在 QT-DoG 中，QAT 被首次用作域泛化（Domain Generalization）的隐式正则化器。QT-DoG 的核心洞察：权重量化引入均匀分布的量化噪声 Δ ∈ [−s/2, +s/2]，该噪声通过二阶 Taylor 展开 L(w+Δ) ≈ L(w) + ∇L Δ + ½Δ^T H Δ 与 Hessian H 交互——在尖锐极小值区域（H 特征值大），Δ 导致损失急剧上升，迫使优化器"逃离"尖锐区域并向平坦极小值收敛。QT-DoG 使用 LSQ 作为量化方法，在训练进行到 2000 步（DomainNet 8000 步）时启动 7-bit 量化，除最后一层外所有层量化至低比特，每通道独立学习 scaling factor s。在 DomainBed 五大基准上，单模型 QT-DoG (7-bit, 0.22× 体积) 达到 66.2% 平均准确率（超过 ERM 的 63.8%），集成版 EoQ 达到 68.4%（超过 DiWA 的 68.0% 且训练开销减少 12×）。关键实验发现：(1) QAT (LSQ/INQ) 有效提升 DG，PTQ (OBC) 无效——因缺少训练阶段无法找到平坦极小值；(2) 7-bit 为最优比特精度；(3) QAT 作为正则化还提升了域内 (IID) 准确率。QAT 在 QT-DoG 中的独特角色不同于传统模型压缩目标——它将量化噪声作为优化正则项，在降低模型体积的同时提升泛化能力。

在 Scaling Law for QAT 中，QAT 被用于训练 74M–973M Llama3-style 模型的 W4A4/W4A16/W16A4 量化配置，共 268 次实验消耗 276K GPU-hours。关键实践：(1) 4-bit QAT 无需高于全精度训练的学习率（量化误差在 LR 5e-4 到 4e-3 范围内几乎恒定于 [0.60, 0.65]）；(2) 权重使用 AbsMax 量化器（因与 LWC/LSQ 精度差异 <0.003），激活使用 AbsMax (G<256) 或 LAC (G≥256)；(3) 使用 INT4 格式（优于 FP4 E2M1，尤其在 per-channel/token 粒度下差距 0.015 loss）；(4) QAT 本身作为正则化器抑制激活 outlier（FC2 kurtosis 从 123→89），但仍无法完全消除 SwiGLU 输出中的系统性 outlier。QAT 缩放定律核心发现：δ_{W4A4} 随 N 增大而减小（74M→594M 平均降 34%）、随 D 增大而增大（10B→100B 平均升 22%）、随 G 变粗而增大（finest→coarsest 差 0.037）。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- Scaling Law for Quantization-Aware Training
- PARQ Piecewise-Affine Regularized Quantization
- Scheduling Weight Transitions for Quantization-Aware Training

在 PARQ 中，QAT 被构建为一个凸正则化优化问题 minimize_w f(w) + λΨ(w)，其中 Ψ 为凸分段仿射正则化函数（PAR）。不同于标准 QAT 使用硬量化映射（STE）全程训练，PARQ 使用渐进软→硬量化：训练初期 proximal map 斜率接近 1（近 identity，几乎不施加量化约束）；训练末期斜率→∞（收敛到硬量化）。这使训练初期的 loss 曲线接近全精度训练，避免了 STE 全程硬量化造成的训练不稳定（如 sudden accuracy drops）。PARQ 在 ResNet (CIFAR-10/ImageNet) 和 DeiT (ImageNet) 的 1-4 bit + ternary 量化上取得与 STE/BinaryRelax 竞争的性能，尤其在小模型极低位宽（1-bit ResNet-20）上有显著优势。PARQ 使用 LSBQ 在线估计目标量化值，无需预先设定量化集合和正则化强度。
