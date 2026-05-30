## Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models

- **baseline方法是什么？**
  Baseline 是未经任何 expert 剪枝的原始 Mixtral 8x7B/Instruct MoE LLM（每层 8 个 expert，top-2 routing）。部署时整模型 bf16 加载需 2 块 A100-80G GPU（总参数 47B，expert 占 96%/45B），每 token 固定激活 2 个 expert 计算 SwiGLU FFN。现有 weight pruning 方法（Wanda/SparseGPT 的 2:4 结构化稀疏）虽能减少参数量，但需专用硬件（FPGA/N:M sparse tensor core）支持 plug-and-play 部署。

  全栈执行例子：
  - 算法 Pipeline：输入 token x → Router(logits l ∈ R^8) → Softmax → top-2 选择 e0, e1 → SwiGLU FFN：E_i(x) = W_down·(SiLU(W_gate·x) ⊙ W_up·x) → 输出 z = w̃_{e0}·E_{e0}(x) + w̃_{e1}·E_{e1}(x)。8 个 expert 各含 3 个权重大矩阵，每 token 仅激活 2 个。
  - 系统框架：HuggingFace Transformers 加载完整 47B 模型 → 2×A100-80G GPU 推理 → GPU 间通信（expert 分布在跨 GPU）。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明（标准 PyTorch linear/SwiGLU kernel）。
  - 硬件架构：NVIDIA A100-80G GPU。

- **论文方法是什么？如何对应解决Baseline的缺陷？**

  **(1) Expert Pruning** 解决 Baseline 的 **内存占用过大** 缺陷：Baseline 8 个 expert 需 2 GPU 加载，实际每 token 仅用 2 个 expert，大量参数闲置。论文方法逐层枚举 expert 组合，以最小化校准集上的 token 重建损失（||F'(x,C) − F(x)||_F）为标准，永久丢弃 n−r 个不重要 expert。Prune 2 个 expert (r=6) 后只需 1 块 GPU 加载，内存从 89,926MB 降至 68,383MB（↓24%）；prune 4 个 (r=4) 降至 46,879MB（↓48%），减少 GPU 间通信开销从而实现 1.27× 加速。

  **(2) Dynamic Expert Skipping** 解决 Baseline 的 **运行时 FLOPs 不减** 缺陷：Pruning 后每 token 仍激活 k=2 个 expert，FLOPs 不变。论文方法利用 routing weight 比值 w_{e1}/w_{e0} < β 在推理时动态决定跳过权重较小的 expert，减少实际激活 expert 数。β 取校准集每层 w_{e1}/w_{e0} 的中位数，使跳过概率 ~50%。此方法与 expert pruning 正交组合：r=6 + skipping 可达 1.23× 加速，同时精度（62.91）显著高于 r=4 纯 pruning（59.57）。

  **(3) Task-Specific 校准** 解决 Baseline/通用 Pruning 的 **domain 迁移差** 缺陷：C4 通用校准集 prune 的模型在数学任务上表现差（GSM8K 从 58.61 降至 41.02）。论文将校准集从 C4 切换到 MATH training set，使 prune 后 GSM8K 5-shot 从 41.02 升至 51.25（r=6），经 MetaMathQA fine-tune 后可达 79.53，接近原 8-expert 模型的 81.35。Fig.4 显示 C4 与 MATH 校准后的 expert 选择仅在 4/32 层相同，说明 domain 对 expert 重要性分布有显著影响。

  论文方法全栈执行例子：
  - 算法 Pipeline：**Pruning 阶段**：校准集 token x → 原始 MoE 层 F(x) → 缓存 (x, Y) → 枚举 C⊆{expert_0,...,expert_7}, |C|=r → 计算 F'(x,C) → 选 min||F'(x,C)−Y||_F 的组合 → 逐层拼接得到 r-expert 模型。**Inference 阶段**：token x → Router(仅 r 个 expert 的 weight) → top-2 → 若 w_{e1} < β·w_{e0} 则仅 E_{e0}(x) 否则 E_{e0}+E_{e1}。
  - 系统框架：HuggingFace Transformers → 修改 model config（expert 数 8→r）→ 加载 pruned checkpoint → 1×A100-80G 推理（r≤6），无需跨 GPU 通信。
  - 编译框架：论文未明确说明。
  - Kernel 调度：论文未明确说明。
  - 硬件架构：NVIDIA A100-80G GPU。论文方法不依赖专用硬件，是 plug-and-play 的算法层面稀疏化技术。
