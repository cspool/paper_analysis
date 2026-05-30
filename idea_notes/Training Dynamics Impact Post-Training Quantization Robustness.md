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
