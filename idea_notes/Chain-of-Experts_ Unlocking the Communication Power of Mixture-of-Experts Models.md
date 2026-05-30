## Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

- baseline方法是什么？
  **Baseline 为传统并行 MoE（Standard MoE）**：每层 Transformer 的 MoE 模块中，所有 expert 独立并行工作——Router 一次性为每个 token 计算 gating scores（$s_i = \text{Softmax}(e_i^\top x)$），从 N 个 expert 中 TopK 选择 K 个，然后并行执行：$y = \sum_{i=1}^{K} g_i \cdot E_i(x)$。Expert 之间无任何交互或信息传递，所有 expert 在单步 forward pass 中独立完成计算。

  **Baseline 全栈执行例子（以 MoE K=8, C=1, 544M 模型推理一个 token 为例）**：

  - **算法层**：token embedding x → Router: $s_i = \text{Softmax}(e_i^\top x)$ → TopK(s, 8) → 并行激活 8/63 routed experts + 1 shared expert → $y = \sum_{i\in\text{top8}} g_i \cdot E_i(x)$。所有 8 个 expert 看到的是同一个原始输入 x，各自独立计算，无顺序依赖，最后简单加权求和。
  - **系统框架层**：PyTorch + veRL FSDP Trainer（https://github.com/volcengine/verl），标准 MoE forward。Expert 计算可并行化——8 个 expert 的 FFN 可同时执行（batch = 8 expert × tokens_per_expert），最大化 GPU 矩阵乘法并行度。
  - **编译框架层**：论文未明确说明（PyTorch eager execution）。
  - **Kernel/运行时调度层**：论文未明确说明。标准 MoE kernel：Gate kernel → TopK selection → grouped GEMM（8 expert FFN 合并计算）→ weighted sum combine。单步完成，无迭代 loop。
  - **硬件架构层**：NVIDIA H100 GPU，单设备。所有 64 experts（63 routed + 1 shared）参数常驻 GPU 显存。

  **Baseline 的核心缺陷**：
  1. **Expert 独立并行无交互**：expert 之间完全独立，无法进行互补推理——每个 expert 只能从原始输入 x 中提取信息，而非基于其他 expert 已精炼的中间表示。限制了 expert 组合的多样性（最多 C(N,2K) 种组合）。
  2. **静态路由不可迭代调整**：每个 token 在单步中被静态分配到固定的 K 个 expert，无法根据中间计算结果重新评估和调整路由决策。
  3. **"深度"仅能通过增加 Transformer 层数实现**：增加表示深度的唯一方式是加层（L↑），导致参数量和内存线性增长。无 within-layer 的深度扩展机制。
  4. **Scaling 效率低**：扩展模型 capacity 只能通过 width scaling（增加 expert 数 N 或每 token 选择数 K）或 depth scaling（增加层数 L），均带来显著的内存和计算开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：Chain-of-Experts (CoE) = 将传统 MoE 层的单步并行 expert 计算改为 C 步迭代顺序计算，每步使用独立 Router 基于前一步的中间表示重新选择 expert，并加入 inner residual connection 稳定训练。

  **Defect→Design 映射**：

  | Baseline 缺陷 | CoE 设计选择 | 解决机制 |
  |---|---|---|
  | Expert 独立并行无交互 | C 步迭代处理：$x^{(t)} = \sum g_{t,i} \cdot E_i(x^{(t-1)}) + x^{(t-1)}$ | 每步 expert 的输入是前一步所有 expert 处理后 + residual 的中间表示，形成 "relay race" 式的顺序精炼 |
  | 静态路由不可迭代调整 | Iteration-based Independent Routing：每步独立 Router 参数 $e_{t,i}$ | 第 t 步 Router 基于 $x^{(t-1)}$ 动态重新评估，可自适应选择不同的 expert 集合 |
  | 深度扩展需加层导致内存线性增长 | C 步迭代作为新的 scaling axis（depth through iteration） | C=2, L=4 匹配 MoE L=12 性能但减少 42% memory；C 增加不增加参数或层数 |
  | 组合多样性受限（C(N,2K)种） | 两次 TopK 独立选择：C(N,K)² 种组合 | N=64, K=4 → 823× 更多 expert 组合，显著提升 representational capacity |
  | 缺乏 inner residual 导致训练不稳定 | 每步 inner residual：$x^{(t-1)}$ 直接加到 $x^{(t)}$ | 消融实验：inner residual loss=1.12 vs outer=1.21 vs init=1.18，inner residual 显著优于其他设计 |

  **论文方法全栈执行例子（以 CoE K=4, C=2, 544M 模型推理一个 token 为例）**：

  - **算法层**：
    1. 初始化：$x^{(0)} = x$
    2. 第一步（t=1）：$s_{1,i} = \text{Softmax}(e_{1,i}^\top \cdot x^{(0)})$ → TopK(s₁, 4) → 选择 expert 集合 A（如 experts 3, 15, 28, 42）→ $h_1 = \sum_{i\in A} g_{1,i} \cdot E_i(x^{(0)})$ → $x^{(1)} = h_1 + x^{(0)}$（inner residual）
    3. 第二步（t=2）：$s_{2,i} = \text{Softmax}(e_{2,i}^\top \cdot x^{(1)})$ → TopK(s₂, 4) → 基于 $x^{(1)}$ 重新选择 expert 集合 B（如 experts 5, 18, 33, 60，与 A 可能完全不同）→ $h_2 = \sum_{i\in B} g_{2,i} \cdot E_i(x^{(1)})$ → $x^{(2)} = h_2 + x^{(1)}$
    4. 输出：$y = x^{(2)}$
    总 expert 计算量 = 4+4 = 8，与 baseline MoE (K=8) 完全相同，但多了 inner residual 的 element-wise add（可忽略）。

  - **系统框架层**：PyTorch + 修改的 veRL FSDP Trainer → 扩展支持 multi-round expert execution。与 baseline 相比，CoE 的 forward 增加了 C-1 次额外的 Router+TopK（低成本）和 inner residual add（element-wise，忽略不计）。但由于每步只选 K/C=4 个 expert（vs baseline K=8），单步 grouped GEMM 的并行度减半，这是论文提到的 "time overhead" 来源——H100 上大 batch grouped GEMM 对小 expert 数的利用率下降。

  - **编译框架层**：论文未明确说明（PyTorch eager execution）。

  - **Kernel/运行时调度层**：论文未明确说明。CoE forward kernel 执行序列：Gate_1 → TopK_1 → GroupedGEMM_4experts → ResidualAdd_1 → Gate_2 → TopK_2 → GroupedGEMM_4experts → ResidualAdd_2。vs Baseline：Gate → TopK → GroupedGEMM_8experts → Combine。CoE 多了 kernel launch 次数（2× Gate + 2× TopK + 2× FFN vs 1×）但总 GEMM FLOPs 相同。

  - **硬件架构层**：NVIDIA H100 GPU，单设备。CoE 的主要 hardware-level tradeoff：sequential processing 减少了单步并行度（4 experts/step vs 8），但通过 "depth through iteration" 在总计算量不变的情况下提升模型表达力。在 H100 上的 time overhead（Limitations 中提及）来自小 batch grouped GEMM 的 GPU 利用率下降。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | C>1 的 iteration depth | MoE 单步并行无交互 → 提供新的 scaling axis | C=2, L=4 匹配 L=12 MoE 性能(-42% memory)；C=2, N=48 匹配 N=64 MoE (-17.6% memory) |
  | Iteration-independent Router | 静态路由无法根据中间状态调整 | 消融：共享 router 导致 loss plateau 在 1.5（远差于独立 router 1.12 + MoE baseline 1.20） |
  | Inner residual every iteration | 多步训练不稳定 | Inner 1.12 vs Outer 1.21 vs Init 1.18 |
  | 保持 sparsity（每步 K/C experts） | 总计算量不变但增加 expert 组合多样性 | Validation loss 1.20→1.12（相同 FLOPs）；823× expert 组合数提升 |
  | 理论分析：C(N,k)² > C(N,2k) | Demonstrates why iterative routing increases representational capacity | 组合空间指数级扩展 |

  **创新总结**：CoE 的核心洞察是将 MoE 的 "shallowly parallel expert processing" 重新定义为 "sequential expert reasoning process"。这不需要修改 expert 架构、不需要增加参数或 FLOPs，仅通过改变 "router 何时调用、基于什么中间状态调用" 来解锁 expert 之间的通信能力。其设计本质是一种 within-layer 的 "recurrence"——类似 Universal Transformer 的跨层参数复用，但 CoE 的 expert 复用发生在同一层内，且每步 Router 独立重新决策。这种设计的代价是减少了单步的矩阵乘法并行度（K/C vs K），在论文的小模型规模下 H100 上用 1 GPU hour 可完成验证；局限性在于尚未在大规模模型（>1B）和多节点训练上验证，且 C>2 时观察到 diminishing returns 甚至不稳定。

**创新总结**：Capacity-Aware Inference 的核心洞察是将 MoE 推理中的不可控 routing skew 通过容量约束转化为可控的延迟上限，然后用扩展候选集在容量约束内"回收"丢弃 token 的表示能力。其本质是在 pre-communication 阶段做一个轻量的 token-to-expert 重调度——不是改变 dispatch 通信模式，而是改变 dispatch 的输入（哪些 token 去哪些 expert）。这使其能无缝集成到任何 Expert Parallelism 框架中（Megatron-LM、DeepSpeed 等），无需修改底层通信或 kernel。额外开销仅来自 topk/capacity/scatter 等逻辑操作（vs expert FFN 和 All-to-All 通信可忽略），收益来自 straggler expert 负载的降低和 GPU 利用率均衡。
