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
