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
