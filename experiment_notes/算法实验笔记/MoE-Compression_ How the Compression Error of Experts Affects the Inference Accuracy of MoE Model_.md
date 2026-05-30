## MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：首次将 error-bounded lossy compression（SZ3 for CPU、CuSZp for GPU）应用于 MoE 推理中的非激活 expert 压缩，以减少 PCIe offloading 的数据传输开销。核心方法为模拟压缩误差（以 Normal 分布 N~(0, ê) 随机生成），添加到 expert 参数上，从 7 个维度系统分析压缩误差对不同层次 expert 的推理精度影响：
    1. 单一 expert（单层内，如 expert-0 in layer 1）
    2. 最高频激活 expert（单层内，如 layer 1 的 expert-26）
    3. 不同层的最高频激活 expert（layer 1/13/20/26）
    4. Top-K 最高频激活 expert（layer 1/layer 26 的 top-6 experts）
    5. 单层全部 expert（layer 1/13/20/26 的全部 64 experts）
    6. 跨层 group 的最高频 expert（Group1: L1-L10, Group2: L9-L18, Group3: L17-L26，每组选 10 个最高频 expert）
    7. 跨数据集泛化（GSM8K → MATH dataset）
  - 误差 bound 设置为：ê = (10%/30%/50%/80% * ||θ_{ℓ,expert}||_1 / n_{ℓ,expert})，即 expert 参数 L1 范数平均值的百分比。
  - 评估指标：Instruction Compliance Accuracy (ICA，输出格式+内容均正确)、Pure Inference Accuracy (PIA，仅内容正确性，忽略格式)。
  - 实验比较：baseline（无误差注入的原始模型）vs 不同误差 bound 下的模型性能。论文也总结了量化方法的比较（Table 1: MC-MoE、MoE-CSP、MoQE、QMoE、CMoE、MoE-MPTQS、HOBBIT、EdgeMoE），但主要贡献是误差敏感性分析而非实现完整压缩系统。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号。论文讨论的 motivation 场景为：GPU 内存有限 + PCIe offloading（如 PCIe 4.0 32 GB/s vs GPU 内部 300 GB/s on-chip bandwidth），推理阶段涉及 expert 参数在 GPU 内存和主内存之间的传输。
  - 推理实验的具体 GPU 配置论文未给出。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Moonlight（MoE 架构，26 个 expert layers，每层 64 expert submodules，inference 时每层 top-6 routing 激活 6 个 expert）。
  - 数据集：
    - GSM8K（数学推理 benchmark，作为主要分析数据集）
    - MATH dataset（Hendrycks et al. 2021，更难的数学数据集，用于泛化评估）
  - 评估指标：ICA (Instruction Compliance Accuracy)、PIA (Pure Inference Accuracy)。此外还自定义了 Imbalance Score、Expert Utilization、Entropy (Normalized)、Gini Coefficient 等指标来量化 expert 激活分布。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供开源代码链接。该工作发表于 SC'25 workshop。
  - 算法 Pipeline 伪代码（误差敏感性分析流程）：

```
# === 准备阶段：分析 Expert 激活分布 ===
# 在 GSM8K/MATH 数据集上对 Moonlight 模型做完整推理
for each question q in dataset:
    model.forward(q)
    for each MoE layer l in {1..26}:
        for each selected expert (top-6) in layer l:
            record expert_index, routing_weight
# 计算每个 expert 的总激活次数 ϕ_{l,e}
# 输出激活热力图（Fig. 3）和各层利用统计（Fig. 4）

# === 误差注入：模拟 error-bounded lossy compression 的误差 ===
# 对于给定 expert (layer l, expert e)，误差 bound ê：
θ = model.layers[l].experts[e].parameters
n = numel(θ)  # expert 参数总数
L1_avg = ||θ||_1 / n  # L1 范数平均值

# 生成遵循 Normal 分布的随机误差（模拟 SZ3/CuSZp 压缩误差分布）
ê = error_bound_pct * L1_avg  # 如 ê = 80% * L1_avg
errors = normal(mean=0, std=ê, shape=θ.shape)  # N(0, ê)

# 注入误差到 expert 参数
θ_perturbed = θ + errors

# === 评估：推理 + 指标计算 ===
output = model.forward_with_perturbed_experts(dataset)
# 计算 ICA: 检查输出格式（如 \boxed{}）和答案正确性
# 计算 PIA: 仅检查答案正确性，忽略格式要求

# === 跨实验维度 ===
# 1. 单一 expert 注入 (Section 3.2.1): expert-0 in layer 1
# 2. 最高频 expert (Section 3.2.2-3.2.3): layer 1/13/20/26 中 ϕ 最大的 expert
# 3. Top-K expert (Section 3.3): layer 1/26 中 ϕ 最大的 6 个 expert
# 4. 全层 expert (Section 3.4): layer 1/13/20/26 的全部 64 experts
# 5. 跨层 group expert (Section 3.5): Group1 L1-10/Group2 L9-18/Group3 L17-26
# 6. 跨数据集 (Section 3.6): 在 MATH dataset 上重复实验
```

- 关键实验发现（9 条 Takeaway）：
  1. 单一 expert 参数误差对推理影响小，但完全随机化参数导致严重退化——即使"不重要"的 expert 也 critical。
  2. 高频 expert 即使误差大（ê=80%），模型仍保持较高 PIA（如 layer 1 expert-26: PIA=0.95），误差首先影响 ICA 再影响 PIA。路由机制可自适应保护核心推理能力。
  3. 不同层的 expert 误差对性能影响呈非单调分布——shallow 层 ICA 降 10-20%，deep 层 ICA 反升 7-10%（layer 26 expert-40: ICA=0.96 vs baseline 0.86）。
  4. 浅层 expert 负责 attention + token→vector 转换（误差影响小），中层负责核心推理（误差影响最大），深层负责指令遵循+输出整合（可控误差可能带来增益）。
  5. 中层（layer 13）全层注入误差时 ICA 降至 0.38（vs baseline 0.86），说明中层对模型推理最关键。
  6. Deep layer 注入可控误差可提升性能——一种隐式集成效应（implicit integration effect），自动生成多样化 ensemble 提升鲁棒性。
  7. 多 expert 同时注入误差产生累积效应（layer 1 top-6: ICA 79%→74%），但 layer 26 top-6 的 ICA 仍高于 baseline（0.90 vs 0.85）。
  8. 误差传播呈非线性级联放大效应——cross-layer 注入（多组 expert 同时扰动）影响远超 single-layer。
  9. 当 ê=80% 时跨层 group 注入导致模型完全失效（所有 group 均无法输出有效结果），仅 ê≤50% 时才产生有效输出。
