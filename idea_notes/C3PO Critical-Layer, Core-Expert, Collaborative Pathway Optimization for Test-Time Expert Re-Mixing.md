## C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

- baseline方法是什么？
  **Baseline 为 MoE LLM 的默认 Router（端到端训练的 top-k gating）**：pretraining 阶段与模型参数联合训练的 router/gate 为每个 token 选择 top-k experts，形成固定的 expert pathway。推理时，router 权重冻结，所有样本共享同一套路由逻辑。

  **Baseline 全栈执行例子（以 OLMoE-1B-7B, 16 layers, 64 experts/layer, top-8 激活为例）**：

  - **算法层**: 输入 token x 经过第 l 层 → Gate 计算 routing logits = x · W_gate (shape: [64]) → Softmax → Top-8 选择 → 仅被选中的 8 个 expert FFN 处理 → h_l = x + Σ_{j∈top8} w_j · Expert_j(x)。所有 16 层使用相同的预训练 router，路径固定不可调。
  - **系统框架层**: 论文未明确说明推理框架。C3PO 通过替换 HuggingFace transformers 中的 `olmoe_modeling.py`，在 forward 时注入修改后的 routing weights。无 Serving 框架层面的调度修改。
  - **编译框架层**: 论文未明确说明（使用 PyTorch + HuggingFace transformers 标准推理）。
  - **Kernel/运行时调度层**: 论文未明确说明具体 kernel 实现。MoE 层使用标准 sparse MoE kernel（top-k gating + 分组 GEMM）。C3PO 在算法层面修改 routing weights，不涉及 kernel 修改。
  - **硬件架构层**: 论文未明确说明具体 GPU 硬件。

  **Baseline 的核心缺陷**：
  1. **Router 次优性 (Sub-optimality)**：预训练的 end-to-end router 对困难样本或分布外样本产生次优的 expert pathway，导致显著的 accuracy gap。实验表明 base model 与 Oracle（知道 ground truth 的最优 routing）之间存在 10-20% 的 accuracy gap（Table 1: OLMoE base 69.9% vs Oracle 85.2%，gap=15.3%；DeepSeekMoE base 66.4% vs Oracle 80.8%，gap=14.4%）。
  2. **静态路由缺乏样本级自适应性**：所有测试样本使用同一套预训练 router，无法根据具体样本特征动态调整 expert 选择。
  3. **Expert 利用不充分**：大多数 expert 被欠利用（仅 12-20 个 expert 被频繁激活），路由缺乏 specialization 导致计算资源浪费。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: C3PO = Critical-Layer (关键层选择) + Core-Expert (核心专家选择) + Collaborative Pathway Optimization (协同路径优化)。核心思路是 test-time adaptation——对每个测试样本，利用参考集中相似样本的 "成功 pathway" 来优化当前样本的 routing weights，无需模型参数更新。

  **Defect → Design 映射**：

  | Baseline 缺陷 | C3PO 设计选择 | 解决机制 |
  |---|---|---|
  | Router 次优性导致 10-20% gap | NGD: 用邻居 loss 作 surrogate objective 做梯度下降优化 ω | 将 routing weights 从固定值变为可优化变量，test-time gradient-based search 逼近 Oracle 性能（NGD 可达 Oracle 的 85-95%） |
  | 静态路由缺乏样本适应性 | Collaborative Pathway Optimization: 基于 embedding 相似度检索 k=3 邻居，用邻居的 successful pathway 指导优化 | 每个样本获得定制化的 routing weights，动态适应样本特征 |
  | Expert 利用率低、缺乏 specialization | Core-Expert: 只优化 top-20 experts 的 routing weights（覆盖最终 top-8 的 99.8%）；Critical-Layer: 只优化最后 5 层 | 激活更集中（Figure 7），强化高频 expert 的 specialization，减少冗余 |

  **论文方法全栈执行例子（OLMoE-C3PO, NGD 变体, k=3, Gaussian kernel, 10 steps, last token）**：

  - **算法层**: 
    1. 测试样本 x 用 NV-Embed-V2 获取 embedding E(x)
    2. 在参考集中 kNN 检索 k=3 个相似样本 {(x_i, y_i, ω_i)}
    3. 提取 x 的初始 ω_0（仅最后 5 层, top-20 experts）
    4. NGD 10 步迭代: L = Σ K_gaussian(x_i, x)·ℓ(f(x_i, ω), y_i) / Σ K_gaussian(x_i, x)，cosine LR 1e-2→1e-5，更新 ω
    5. 用优化后的 ω* 推理 f(x, ω*)
  - **系统框架层**: 替换 HuggingFace transformers 的 `olmoe_modeling.py`，注入优化后的 routing weights。通过 `olmoe_optimizer.py` 执行优化流程。未修改 Serving 框架。
  - **编译框架层**: 论文未明确说明。
  - **Kernel/运行时调度层**: 论文未明确说明。路由权重在算法层面被修改，不改变底层 MoE kernel 的执行方式。
  - **硬件架构层**: 论文未明确说明具体 GPU 硬件。
