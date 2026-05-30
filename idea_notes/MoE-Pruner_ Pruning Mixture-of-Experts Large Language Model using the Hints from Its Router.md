## MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

- baseline方法是什么？
  Baseline 是现有 LLM 后训练剪枝方法 SparseGPT 和 Wanda。SparseGPT 的剪枝度量 S = [|W|^2 / diag(H^{-1})]，需要估计逆 Hessian 并更新剩余权重；Wanda 简化度量 S = |W_ij| * ||X_j||，仅需校准数据计算输入激活列范数，不更新权重。两种方法都**未考虑 MoE router 的路由信息**，对所有 FFN 层（包括 MoE expert 层）使用统一的剪枝度量。
  全栈执行例子：一个 token 进入 MoE layer → Router 计算 top-2 gating 并选择 Expert_i 和 Expert_j → 每个 expert 执行 SwiGLU FFN：x → W_gate·x 与 W_up·x → SiLU(W_gate·x) ⊙ (W_up·x) → W_down 输出 → 加权求和。Wanda 剪枝时：对 expert 内的 W_gate/W_up/W_down，用校准数据前向得到 X → 计算 S = |W| * ||X||（所有 expert 共用同一度量，router 选择的差异化信息被丢弃） → 每个输出神经元保留 top-(1-p%) 重要性权值。问题：Router 权重 Gate 本身反映了"这个 expert 对这个 token 的重要性"——若 Gate_i ≈ 0（该 expert 对此类 token 几乎不被激活），其权值即使 magnitude 大也应被优先剪除，而 Wanda 无法捕获此信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 MoE-Pruner，核心是将 router 权重显式纳入剪枝度量：**S = |W_ij| * ||X_j * Gate_j||**。Gate_j 是 router softmax 输出中对当前 expert 的归一化权重（广播到所有输入维度后与 X_j 逐元素乘），使重要性计算包含"这个 expert 对这一批输入 token 有多重要"的信息。此外提出 expert-wise knowledge distillation 做剪枝后性能恢复：以未剪枝 pretrained model 为 teacher，逐 expert 计算 MSE loss 叠加 CE loss 蒸馏 student。
  全栈执行例子（对比 baseline）：同一 token → Router 计算 top-2 gating 同时输出所有 expert 的 Gate 权重向量 [g_0, g_1, ..., g_7] → Expert_i 执行前向得到 X' 和 Gate[:,i] → **MoE-Pruner 度量计算**：X_gated = X ⊙ broadcast(Gate[:,i])，S = |W| * ||X_gated||（此时被 Gate 放大的激活维度对应权值重要性更高，被 Gate 压制的维度权值重要性更低） → 剪枝后保留的权值集中在"高 router 权重 token 的活跃激活路径"上。关键区别：（1）Wanda 对所有 expert 平等对待，MoE-Pruner 利用 router 告诉它"哪些 expert 对这个 token 更重要"，从而更精确地保留关键权值、剪除非关键权值；（2）剪枝后通过 expert-wise KD 蒸馏：对每个 MoE layer 的每个 expert，L_expert = MSE(E_teacher, E_student)，保证每个 expert 输出分布逼近 teacher，而非仅靠全局 CE loss。编译框架/Kernel调度/硬件架构：论文未明确说明。

<｜｜DSML｜｜parameter name="replace_all" string="false">false

- baseline方法是什么？
  Baseline 是标准的 MoE 架构（如 Mixtral-8x7B、DeepSeek-V2-Lite），其中每个 MoE layer 包含 N 个 monolithic expert（每个 expert 是一个完整 FFN），router 通过 top-k 选择机制激活固定数量的 expert（如 k=2 或 k=6），最终输出为激活 expert 输出的加权求和。
  全栈执行例子：一个请求 token 进入→Router（线性层）计算所有 N 个 expert 的 logits→top-k 选择 Expert 3 和 Expert 7→GPU 加载 Expert 3 和 Expert 7 的完整权重矩阵（W_gate, W_up, W_down）→依次计算两个 expert 的 SwiGLU FFN→加权求和得到输出。若降低 k 从 6 到 5，模型质量会因"Quality Cliff"而出现不成比例的大幅下降（因为 monolithic expert 内部冗余性未被利用，丢弃任何一个完整 expert 都会丢失其中被训练为协作的关键 neuron）。云端调度：FIFO 或 FullBatch 调度器使用批次内最高 k_min 作为全局 k_active，粗粒度配置导致小 batch 时资源浪费或大 batch 时无法灵活降级。Offloading 场景：即使只需 expert 中 25% 的 neuron 计算，仍需从 CPU 加载整个 multi-GB monolithic expert 到 GPU VRAM，I/O 浪费严重。编译框架/Kernel调度/硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法分为 Offline Refactoring Engine（离线模型重构）和 Online Scheduling Engine（在线调度引擎）两阶段。
  全栈执行例子：
  - **算法Pipeline层**：Neuron Activation Profiler 在 Wikitext-2-raw-v1 上运行原模型，从每个 expert 的 SwiGLU FFN 中收集激活矩阵 M(B×C numpy)→Partitioning Optimization Solver 用 Simulated Annealing（T₀=100, α=0.995, 100K 迭代）将每个 expert 的 C 个 neuron 划分为 N=4 个子 expert 的分区 P*，优化目标为最小化所有 batch 上被 deactivated sub-experts 的 L1 norm 之和→Gating Mechanism Reconstructor 构建共激活矩阵 C_co=B^T·B（B 为 top-k_a 激活二值化矩阵），从每个子 expert 选择 centrality 最高的 r=4 个 gate neurons 作为代理评分器。可选：在 SlimPajama 上仅微调 linear router（<0.1% 参数），curriculum training 逐步增加 k。
  - **Serving调度层**：部署前 benchmark 构建 C(k_active) 性能模型（sub-expert 数量→延迟/内存的 lookup table）。云端场景：请求到达→按 k_min 加入所有符合条件的虚拟队列→对 M 个虚拟队列并行计算效用 U_m=Σtokens/C(|Q_m|,m)→发射最高效用批次（或触发 Batch Full/Timeout 硬触发器）→修改版 vLLM 0.9.1 执行 fine-grained sub-expert 推理，仅激活必要的子 expert。Offloading 场景：VRAM Cache Manager 用 LRU 管理 sub-expert 缓存→解码循环中 router 通过 gate neurons 估算每个子 expert 的 L1 norm→对 miss 子 expert 异步 CPU→GPU 传输→GPU 计算→LRU 更新。
  - **编译框架/Kernel调度/硬件架构/芯片设计**：论文未明确说明。
  核心设计如何解决 Baseline 痛点：
  1. **Quality Cliff → Smooth Trade-off Curve**：通过将 1 个 monolithic expert 分解为 4 个子 expert，一个 k=6 的模型变为 k_active=24 的细粒度配置空间，提供 4 倍以上的可区分稳定操作点。例如原模型只能选择 k=2 或 3，MoE-Prism 可选择 k=9（相当于原模型 2.25 个 expert），精确匹配 SLO 需求。
  2. **粗粒度 Offloading I/O → 精准按需传输**：传统方式必须加载整个 expert（即使只需其中部分 neuron），MoE-Prism 只传输 S_req(t) 中命中的子 expert，16GB 配置下 cache hit ratio 从 0.4375 提升到 0.4453，且可通过加载 17 个子 expert（等效 4.25 experts）满足"需要 4.2 experts"的 SLO，避免被迫加载 5 个完整 expert 的浪费。
  3. **固定 k 调度僵局 → 效用驱动的动态多队列调度**：打破"批次组成依赖 k_active，k_active 选择又依赖批次组成"的循环依赖，通过维护 M 个虚拟队列并行评估所有可能的 (k_active, batch) 组合，选择最高瞬时吞吐效用的配置，实现吞吐提升 19.9%（Deepseek）和 14.9%（OLMoE）。
