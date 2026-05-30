## MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?

- baseline方法是什么？
  Baseline 为现有 MoE 模型的 expert 压缩策略，主要包括四类：
  1. **Expert Quantization**（如 MC-MoE、MoQE、QMoE、CMoE、MoE-MPTQS、HOBBIT、EdgeMoE）：将 expert 参数从浮点精度量化为低精度整数（1/2/3/4/8-bit），以约 4× 内存节省换取推理加速。然而低比特量化引入不可控和不可预测的误差，导致生成性能显著下降（如 QMoE 在 20× 压缩比下 accuracy drop 达 6.7%，CMoE 在 150× 压缩比下 accuracy drop 达 23.81%）。
  2. **Expert Distillation**（如 ExpertFlow）：将大型 MoE 模型的知识蒸馏到更小的模型/reduced expert set。
  3. **Expert Pruning**（如 Lu et al. 2024）：识别并移除贡献小的 expert。
  4. **Expert Decomposition**（如 MiLo）：使用低秩分解技术减少参数量。

  以 expert quantization（最常见的 offloading 场景压缩策略）为 baseline，全栈执行路径如下：
  - **算法层（Expert Quantization + Offloading）**：MoE 推理时，所有 expert 权重预先量化为低精度（如 2-bit 或 4-bit）并存储在 CPU 主内存中。Router 选择 top-K expert → 通过 PCIe 从主内存加载对应量化 expert 权重到 GPU 显存 → GPU 上反量化恢复浮点精度 → 执行 FFN 计算 → 输出加权聚合。**核心缺陷**：(1) 量化误差不可控——低比特量化 (1-2 bit) 的量化噪声分布无法保证 bounded error，且不同 expert 对量化误差的敏感度高度异质（shallow/middle/deep layer 的 sensitivity 差异巨大），uniform 位宽分配导致重要 expert 欠保护而冗余 expert 过保护；(2) 缺乏系统性的压缩误差敏感性分析——现有工作未回答"哪些 expert 对压缩误差更敏感"这一关键问题，导致 compression 策略盲目、无法针对性优化。
  - **系统框架层**：基于 HuggingFace Transformers 或类似推理框架，使用 GPU offloading 技术（如 MoE-Infinity、SwapMoE、Pre-gated MoE）。核心瓶颈为 PCIe 带宽（PCIe 4.0: 32 GB/s << GPU HBM: 300 GB/s），数据传输延迟无法被计算隐藏。量化压缩减少传输量但牺牲精度。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA）。
  - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM 或量化 kernel（如 HQQ）执行反量化+矩阵乘法。
  - **硬件架构层**：PCIe 4.0 GPU 服务器（GPU 内存有限，需 offloading 到 CPU 主内存），论文未指定具体 GPU 型号。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出用 **error-bounded lossy compression（SZ3/CuSZp）替代 quantization**，并首次系统性地从 7 个维度分析了压缩误差对不同 layer expert 的推理精度影响。核心贡献是**误差敏感性分析**而非完整的 compression system implementation（论文明确说明其聚焦于三步中的前两步：①高效压缩算法选择、②误差敏感性分析）。

  全栈执行路径（以 Moonlight 模型 + GSM8K dataset 为例）：

  - **算法层（Error-Bounded Compression + Layer-Aware Sensitivity Analysis）**：
    1. **Error-Bounded Lossy Compression 替代 Quantization**：不同于量化（误差不可控），SZ3（CPU）和 CuSZp（GPU）提供严格 error bound ê 保证。压缩后专家参数的最大误差不超过 ê，压缩误差分布近似 N~(0, ê)。这使误差可控——通过调整 ê 可精确 trade-off 压缩比与精度。论文通过模拟 normal distribution error N~(0, ê) 注入 expert 参数来研究误差影响，其中 ê 取 L1 范数平均值的 10%/30%/50%/80%。
    2. **层级化误差敏感性分析（7 个维度）**：
       - 维度 1（单 expert）：分析了 expert-0 in layer 1 在不同 ê 下的表现。结果：小误差不影响推理，但完全随机化参数→错误输出——证明即使是"不重要"的 expert 也 critical。
       - 维度 2（最高频 expert）：layer 1 的 expert-26（激活频率最高），ê=30%/50%/80%。结果：PIA 保持 0.95-0.96（仅降 0-0.01），ICA 从 0.86 降至 0.79——误差先影响指令遵循能力、后影响推理能力。
       - 维度 3（跨层最高频 expert）：layer 1/13/20/26 的各自最高频 expert，ê=80%。结果：ICA 呈非单调分布——layer 1: 0.79, layer 13: 0.75, layer 20: 0.89, layer 26: 0.96（**深层 ICA 反而超过 baseline 0.86**）。PIA 始终保持 ≥0.94。
       - 维度 4（Top-K expert）：layer 1 和 layer 26 的 top-6 highest-frequency experts，ê=80%。结果：layer 1 ICA 从 79%→74%（累积效应），layer 26 ICA=0.90 仍 > baseline 0.85。
       - 维度 5（全层 expert）：layer 1/13/20/26 全部 64 experts，ê=80%。结果：shallow layer ICA 骤降至 0.33，middle layer 13 ICA 最低 0.38（最敏感），deep layer 26 ICA=0.85（几乎不下降）。
       - 维度 6（跨层 group）：Group1 L1-L10 / Group2 L9-L18 / Group3 L17-L26，ê=30%/50%/80%。结果：ê=80% 时所有 group 模型完全失效（不输出），ê=50% 时 Group2 (middle) ICA 最低 0.69——中间层对误差最敏感。
       - 维度 7（跨数据集泛化）：在 MATH dataset 上重复维度 3 和 6。结果：更难的数据集上误差影响更显著（baseline PIA 本身仅 0.70，ê=80% on layer 13 时 PIA 降至 0.60），但深层误差仍可能带来增益（layer 26 ICA: 0.66 vs baseline 0.62）。
    3. **9 条关键结论（Takeaway）**指导实践：
       - 浅层专家（attention + token→vector 转换）：对 bounded error 最鲁棒，可激进压缩。
       - 中层专家（核心推理）：最敏感，需保守压缩/保护参数完整性。
       - 深层专家（指令遵循 + 输出整合）：可控误差可提升性能（隐式集成效应→多样化 ensemble），可作为优化策略。
       - 多 expert/多 layer 误差呈非线性级联放大效应，跨层 group 注入比单层影响大得多。

  - **系统框架层**：论文未实现完整的 offloading-compression 集成系统（明确说明此步骤留待 future work）。但给出了设计方向：在 MoE offloading 框架中，于 expert 从主内存传输前执行压缩（CPU 上 SZ3 或 GPU 上 CuSZp），传输后再解压。未来需设计 pipeline 算法 overlap compression/decompression 与 offloading 任务以隐藏延迟。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。使用了现有的 error-bounded lossy compressor（SZ3 for CPU, CuSZp for GPU），但未实现自定义 kernel 或将其集成到推理 pipeline 中。
  - **硬件架构层**：论文未明确说明具体 GPU。在 PCIe-limited 场景下（GPU HBM 有限 + 主内存 abundant），压缩-传输-解压流水线需要 CPU 或 GPU 上的 compressor 支持。

  **对比 baseline 的改进映射**：
  - **Quantization 的不可控误差 → Error-Bounded Compression 的可控误差**：量化（尤其是 1-2 bit）引入不可预测误差导致严重精度退化 → SZ3/CuSZp 提供严格 bounded error 保证（‖θ_compressed - θ_original‖_max ≤ ê），且误差分布可建模为 Normal 分布，使误差影响可预测、可控制。
  - **Uniform 位宽分配 → Layer-Aware 差异化压缩**：quantization 对所有 expert 使用相同位宽（或简单 heuristic）→ 论文的敏感性分析结果表明应以不同压缩力度处理不同层 expert：浅层可激进压缩、中层保守保护、深层可适当注入噪声。
  - **缺乏误差敏感性理解 → 7 维度系统分析 + 9 条实践指导**：这是本工作的首要贡献——不是提出新的 compression algorithm，而是回答"哪些 expert 的压缩误差对推理精度影响最大"这一基础性问题。结果直接指导 MoE compression 系统的设计（如 MC-MoE 的 expert 重要性驱动位宽分配可受益于本篇的 layer 级敏感性洞察）。
  - **深层误差增益的发现 → 隐式集成优化策略**：最反直觉的发现——深层 expert 注入可控噪声可提升 ICA（layer 26: 0.96 vs baseline 0.86, layer 20: 0.89 vs 0.86），揭示了一种无需训练的低成本模型鲁棒性增强方法：在推理时对深层 expert 参数添加微小随机扰动。
