## Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

- baseline方法是什么？
  - **Sparsely Activated MoE with Discrete Token-Level Routing**：传统稀疏激活 MoE 模型（Switch Transformer, GShard, Expert Choice, ST-MoE 等）使用 top-k 离散路由网络将每个 token 分配到 k 个专家。路由决策是非可微的离散选择（argmax 或 top-k），使训练变得困难：(1) 需要精心设计的负载均衡辅助损失（auxiliary loss）来防止专家坍缩和负载不均；(2) 离散路由可能导致训练不稳定和专家欠专业化（Zoph et al., 2022）；(3) 路由网络梯度信号稀疏（仅选中的 k 个专家接收梯度），路由器学习效率受限；(4) 推理时需要维护所有专家的稀疏激活路径，增加系统实现复杂度。Token-level routing 学到的专家专业化是浅层的（标点、冠词等词级特征），缺乏深层语义/领域级别的专业化。
  - 全栈执行例子（Baseline Expert Choice MoE, 0.3B/8E, 8x A100, token-level routing）：
    - **训练算法层**：Router linear W_r·h_x → softmax → 每个 expert 选 top-k 输入（根据路由分数）→ capacity factor C=1 限制每 expert 处理 token 数 → token dispatch via all-to-all → 每个 expert 独立 FFN 计算 → token combine via all-to-all → 加权聚合输出。L_aux = α·N·Σ f_i·P_i 负载均衡 loss + L_lm 交叉熵。
    - **系统框架层**：PyTorch + Megatron-LM 或 DeepSpeed。Expert parallelism + all-to-all 通信。实现 all-to-all dispatch/combine + barriers between layers。论文未明确说明具体 Serving 框架（预训练场景）。
    - **编译框架层**：论文未明确说明。PyTorch eager mode / torch.compile。
    - **kernel 调度层**：NCCL All-to-All for token dispatch/combine + cuBLAS GEMM for expert FFN。Dispatch/combine 通信量 ∝ K·L·d（K experts per token，L tokens）。Token-level routing 产生不规则形状 all-to-all（各 expert 分配的 token 数不同），需 padding 或 drop。
    - **硬件架构层**：A100 GPU。Inter-node all-to-all 通信瓶颈（随 expert 数增加而加剧）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Lory 方法**：通过三个核心设计使 MoE 训练完全可微并实现有效的段级路由：
    1. **Fully Differentiable MoE via Expert Merging**：替代离散 top-k 路由为参数空间的软合并。路由权重 e_i = Softmax(R(h̄)) 直接作为专家参数 θ_i 的加权系数：θ̄ = Σ_i e_i · θ_i。合并后的 FFN 处理输入 o_x = FFN(h_x; θ̄)。整个过程端到端可微，梯度通过合并操作和路由网络全程流动，无需辅助负载均衡损失。解决了"离散路由不可微、梯度信号稀疏"的缺陷。
    2. **Causal Segment Routing**：将 token-level routing 改为 segment-level routing（T=256 tokens/段）。使用前一段的隐藏表示计算当前段的合并专家，保持自回归因果性。关键效率：合并操作从每 token 一次降为每段一次（L/T 次，对 L=4096/T=256 为 16 次），额外 FLOPs 上限仅 E/T（E=32 时约 12.5% MoE 层开销，总模型开销 15-28%）。推理时仅用 prompt 路由一次，后续生成与 Dense 模型效率完全相同。解决了"token-level 合并计算代价过高"的缺陷。
    3. **Similarity-based Data Batching**：用 Contriever 计算文档语义相似度，将相似文档拼接为训练实例，使相邻段来自相关领域。这鼓励段级路由学习领域级别的专家专业化（如 Python code 专家、学术论文专家），而非传统 token-level routing 学习的浅层语法特征（标点、冠词）。解决了"段级路由容易导致专家欠专业化"的缺陷。
  - 对应解决 Baseline 缺陷：
    - **离散路由不可微 → 需要负载均衡 loss、训练不稳定** → Expert merging 实现完全可微路由，全程梯度回传，无需离散决策和辅助 loss
    - **Token-level 合并计算代价 O(L·E·d·d') → 成本过高** → Causal segment routing 降为 O(L/T·E·d·d')，每段仅合并一次，合并开销与段数成正比
    - **Token-level 路由学到的是浅层词法特征 → 专家缺乏深层语义** → Similarity-based batching + segment routing 使专家学习领域级专业化（如 Python、arXiv、Books），不同层的专家在不同深度展现领域偏好
    - **Training recipe 复杂 → 负载均衡 + 离散决策 + 负载损失调参** → Lory 仅用单一的 cross-entropy loss 做端到端训练（无需 auxiliary load balancing loss，仅 warmup 阶段用 dense 初始化）
    - **推理时需维护专家稀疏激活路径** → Prompt-only routing 使推理退化为合并后的单一 Dense 模型，零额外内存或计算开销
  - 全栈执行例子（Lory 0.3B/32E, 64x A100, 150B tokens 预训练）：
    - **训练算法层**：Dense warmup（前 5% 步训练 dense 模型）→ 复制 FFN 初始化 MoE → Similarity-batched 训练实例（L=4096, 16 段 × 256 tokens）→ 每层的 Causal Segment Routing：S_0 的 h̄_0 经 stop_grad → softmax → e_0 → merge 32 FFN → 处理 S_0；S_1 用 h̄_0（无 stop_grad）→ e_1 → merge → 处理 S_1；依此类推 → Cross-entropy loss 回传（无额外 auxiliary loss）→ 梯度通过合并操作更新所有专家和路由网络
    - **系统框架层**：PyTorch + ZeRO 数据并行。无需 all-to-all 通信（参数合并替代了 token dispatch/combine）。论文 Section 6 讨论了 expert-wise model parallelism（按 hidden dim 切分所有专家到不同设备）用于扩展至 100B+ 参数。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。合并操作实现为逐专家参数的加权求和（纯 PyTorch tensor ops）。
    - **kernel 调度层**：合并后的 FFN 计算等价于单个 Dense FFN 的 GEMM 操作（cuBLAS），因为合并后的权重是单个矩阵。无需 GroupedGEMM 或 expert dispatch kernel。每段仅执行一次合并操作（16 次/层 vs 4096 次 token-level），合并 overhead 小。
    - **硬件架构层**：A100 GPU。合并操作在参数空间进行（通信专家参数而非 token 激活），适合 expert-wise model parallelism（图 7）。Data parallelism 用于非 MoE 部分（attention），model parallelism 用于 MoE 层。merge 操作无跨设备通信需求（每个设备持有完整的 expert 权重副本或按 hidden dim 分片）。

- 关键洞察：
  - **段级路由学到的专家专业化与 token 级路由完全不同**：Token-level MoE 学到的是浅层词法特征（标点专家、冠词专家），而 Lory 的段级路由学到的是领域级特征（中高层专家按 Books/arXiv/Python/Wikipedia 领域分化，图 6）。这种互补性暗示未来可结合两种路由策略。
  - **Warmup 训练至关重要**：无 warmup 时大量专家未被利用（1.5B/32E 图 10），warmup 确保专家从良好的 dense 初始化出发。更多专家数（32E）时专家利用率的持续提升可持续到训练结束（图 9）。
  - **Prompt-only routing 在推理中足够**：推理时仅用 prompt 做一次 routing（vs 逐段 reroute），下游任务性能差异不显著（Table 9），使 Lory 推理简化为 Dense 模型。
