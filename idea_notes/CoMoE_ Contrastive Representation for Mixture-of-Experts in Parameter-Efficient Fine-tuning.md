## CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

- baseline方法是什么？
  **Baseline 为标准的 LoRA-based MoE PEFT 方法（MixLoRA、MOELoRA、MiLoRA、OMoE 等），以及非 MoE 的 PEFT 方法（LoRA、DoRA）。**

  **Baseline MoE-PEFT 的核心机制**：
  - 将 LoRA 的低秩矩阵 A, B 替换为 n 个并行 expert {E_i = B_i A_i}，通过 Router g(x; G) 进行 top-k 稀疏激活
  - Router 计算每个 expert 的 importance score，选择 top-k 激活，其余 expert 输出贡献为 0
  - 输出: y' = W₀·x + Σ_{i∈T} ĝ_i · E_i(x)

  **这些 baseline 方法的共性问题**：
  1. **Expert Knowledge Redundancy（专家知识冗余）**：缺乏足够的专业化约束导致不同 expert 学习到重叠/相似的功能，浪费 MoE 的容量。
  2. **Expert Load Imbalance（专家负载不均）**：训练中仅部分 expert 被频繁激活，其他 expert 利用不足，违背 MoE 设计初衷。
  3. **Capacity Underutilization（容量利用不足）**：简单堆叠更多 expert 不会线性提升性能（Qian et al., 2024），反而遇到性能瓶颈。

  现有方法（如 load balance loss, localized balancing constraint）尝试缓解上述问题但远不足够。

  **Baseline 全栈执行例子（以 MixLoRA on LLaMA-2 7B 为例）**：

  - **算法层**: 输入 token x → Router 计算 n=8 个 expert 的 gating score → top-2 激活 → 仅 top-2 expert 的 LoRA 输出参与残差计算 → 交叉熵损失反向传播。Router 无专业化约束，expert 参数更新仅由下游任务 loss 驱动，导致 expert 功能趋同。训练后约 2.9% 参数可训练。
  - **系统框架层**: 论文未明确说明推理框架。使用 HuggingFace PEFT + transformers 标准训练/推理流程。MoE-LoRA 的 expert 计算本质是独立的低秩矩阵乘法并行执行后加权求和，不涉及推理框架修改。
  - **编译框架层**: 论文未明确说明（使用 PyTorch eager mode + HuggingFace transformers）。
  - **Kernel/运行时调度层**: 论文未明确说明。expert 的 LoRA 前向为 standard GEMM（B·A·x），多个 expert 在 PyTorch 层面并行计算后 weighted sum。top-k routing 通过 argmax + mask 实现。
  - **硬件架构层**: NVIDIA A6000 48GB GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **无专业化信号**：Router 仅基于 task loss 隐式学习路由，缺乏显式的 "expert 应差异化" 信号。导致 expert 学习到相似的参数分布（OMoE 论文已通过实验证明 vanilla MoE 的 expert 分布坍缩）。
  2. **非激活 expert 被浪费**：训练中 top-k 只激活少数 expert，非激活 expert 的前向输出被丢弃（乘以 0），反向传播中这些 expert 的参数更新梯度仅间接来自未来可能被选中的概率，缺乏直接利用。
  3. **Balance loss 治标不治本**：Load balance loss 强制 expert 负载均匀但不保证 expert 功能差异化——可能导致不同 expert 在相同功能上轮流激活而非真正专业化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **CoMoE (Contrastive Representation for MoE)** 在 MoE-based PEFT 训练中引入基于 InfoNCE 的对比学习辅助目标，从信息论角度促进 expert 专业化。

  **关键创新——将非激活 expert 从 "浪费" 变为 "负样本"**：
  CoMoE 将 top-k routing 下的非激活 expert 重新利用为对比学习的负样本（negative keys），同时将激活 expert 作为正样本（positive keys）。这种设计使得每个 expert 在训练中同时受到两个方向的梯度信号：
  - **正信号**（来自激活 expert 的 L_CE + 正对比样本）：学习匹配当前输入
  - **负信号**（来自非激活 expert 的负对比样本）：被推离当前输入的表示空间

  **对比 Baseline，CoMoE 全栈执行例子（LLaMA-2 7B, n=4, k=2）**：

  - **算法层**: 
    1. 输入 token x → Router 计算 4 个 expert 的 gating score → top-2 激活（如 expert 1 和 3）
    2. 标准前向：y' = W₀·x + ĝ₁·E₁(x) + ĝ₃·E₃(x) → 计算 L_CE
    3. 对比前向：收集所有 4 个 expert 的输出 E₁(x)...E₄(x) 作为表示向量（不经 weighted sum）
    4. 随机选一个激活 expert（如 expert 1）为 anchor q = Normalize(E₁(x))
    5. 正样本 P = {Normalize(E₃(x))}（另一个激活 expert）
    6. 负样本 N = {Normalize(E₂(x)), Normalize(E₄(x))}（非激活 expert）
    7. 计算 cosine similarity: s_pos = q·E₃(x)/τ, s_neg = q·E₂(x)/τ + q·E₄(x)/τ
    8. InfoNCE: L_con = -log( exp(s_pos) / (exp(s_pos) + Σexp(s_neg)) )
    9. 总损失: L_total = L_CE + 0.01·L_con → 反向传播
    10. **效果**：激活 expert 的表示被拉近（专业化协作），非激活 expert 的表示被推远（避免冗余）。多个 task 自然分配给不同 expert 组合（Figure 4 可视化验证）。

  - **系统框架层**: 论文未明确说明推理框架。训练基于 HuggingFace PEFT + transformers。对比损失计算仅依赖 expert 输出表示（前向传播中已计算的中间结果），不增加额外前向开销。推理时无需对比损失——仅标准 top-k routing 前向，因此推理延迟与 MixLoRA 同级甚至更优（3,789ms vs 4,217ms）。
  
  - **编译框架层**: 论文未明确说明（使用 PyTorch eager mode）。

  - **Kernel/运行时调度层**: 论文未明确说明。对比损失在 Python 层面计算（normalize + dot product + softmax + log），张量操作量 O(n·D) vs expert FFN 计算 O(d_model²)，可忽略不计。

  - **硬件架构层**: NVIDIA A6000 48GB，无自定义硬件。单卡可完成全量实验。训练 3.5h（multi-task, n=4）。

  **CoMoE 解决 Baseline 缺陷的映射关系**：

  | Baseline 缺陷 | CoMoE 解决方案 | 实现机制 |
  |--------------|---------------|---------|
  | Expert 功能冗余（无专业化信号） | 对比损失显式惩罚 expert 表示相似性 | 正样本拉近同类 expert，负样本推远异类 expert → 表示空间分散化（Figure 5 可视化验证） |
  | 非激活 expert 信息浪费 | 将非激活 expert 用作负样本，赋予其训练信号 | s_neg 梯度更新非激活 expert 参数，推动其学习不同于当前输入的功能 |
  | Balance loss 治标不治本 | 对比损失自然产生专业化分工，无需显式 balance loss | Figure 4: 加入 contrastive loss 后不同 task 自然分配到不同 expert 组合（如 ARC-c→expert{1,3}, BoolQ→expert{1,4}） |
  | 堆叠 expert 不线性提升性能 | 通过提升每个 expert 的利用率而非数量来提升容量 | n=4 在 multi-task 上 avg +1.3 超过 n=8 的 MixLoRA（参数效率 2×） |
  | 多任务性能退化（ST→MT 下降 7-12%） | 专业化 expert 更好地处理异质数据分布 | ST→MT 下降仅 0.1-1.8%（vs baseline 4.4-8%） |

  **理论保证**：
  Theorem 1 证明 InfoNCE loss 是对 MI Gap ΔI = I(x; M⁺) - I(x; M⁻) 的紧下界：ΔI ≥ log(N) - L_NCE。最大化该下界 = 最大化输入与激活专家的互信息同时最小化输入与非激活专家的互信息，从信息论角度保证 expert 专业化。

  **核心结果**：
  - Multi-task 平均 accuracy +1.3 (LLaMA-2 7B, vs 最强 baseline)
  - 参数效率 2×：1.45% tunable params vs MixLoRA 2.9%，性能更优
  - 推理延迟降低 10% vs MixLoRA (3,789ms vs 4,217ms)
  - GPU 内存节省 465 MiB vs MixLoRA
  - 固定负样本采样策略将训练复杂度从 O(n) 降至 O(1)，性能无损
