## Parameters vs FLOPs Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models

- baseline方法是什么？
  - Baseline 是传统的 Dense Transformer Scaling Laws（Kaplan et al. 2020, Hoffmann et al. 2022），以及将 MoE 中除稀疏度外的其他变量（如 expert 数量、granularity）纳入分析的 MoE Scaling Laws（Clark et al. 2022, Ludziejewski et al. 2024）。这些 baseline 的核心假设是：(a) 模型容量（capacity）主要由总参数量 N 定义；(b) 在 Dense 模型中 N 和 FLOPs per example 线性耦合（FLOPs ≈ 6N per token），因此 N 可作为计算成本的代理变量；(c) MoE Scaling Laws 通常固定稀疏度配置（如固定 K 个 active experts）而仅变化其他变量。
  - 全栈执行例子（Baseline: Hoffmann et al. 2022 Chinchilla Scaling Law + fixed-sparsity MoE）：
    - 算法层：Hoffmann 的 L(N,D) = a/N^α + b/D^β + e 仅含 N 和 D 两个变量，按 C = 6ND 约束求解最优 N*(C) ∝ C^α。在 MoE 中扩展时（Clark et al. 2022），将 N 替换为总参数量并在固定 expert 配置（如 fixed granularity G，fixed E_active/E_total 比）下拟合。稀疏度 S 作为隐含固定变量存在，不被显式建模为可控维度——无法回答"给定 C 和 N，最优 S 是多少"。论文未明确说明系统框架、编译框架、kernel调度、硬件架构层——该工作纯算法层 Scaling Law 分析。
    - 关键缺陷：baseline 无法量化 FLOPs per example 与总参数量之间的最优权衡。在 MoE 中，S 控制活跃参数量 N_a = N·(1-S)，进而控制 FLOPs per example。若 S 被固定，则无法知道在给定训练计算预算下，应该通过增加总参数（提高 S）还是增加活跃参数（降低 S）来提升性能。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文将稀疏度 S 作为独立的第三维度纳入 Scaling Law 分析框架，系统研究 N（总参数）与 FLOPs per example（通过 S 控制）之间的最优权衡。
  - 全栈执行例子（论文方法: S-aware IsoFLOP surfaces + parametric scaling law）：
    - 算法层：论文将问题拆解为两个子问题并统一求解。(1) 固定 S 求最优 N：N* = argmin_N L(N; C, S)，通过 IsoFLOP 曲面沿 N 轴切片得到——发现固定 C 下 N* 随 S 增加而增加，而 N*_a 随 S 增加而减少，即稀疏模型更"大"但推理更"便宜"。(2) 固定 N 求最优 S：S* = argmin_S L(S; C, N)，通过 IsoFLOP 曲面沿 S 轴切片得到——发现 L(S; N, C) 呈抛物线形状，存在最优 S*，且 S* 随 N 增大而增大直至趋近 1，随 C 增大而减小。最终提出包含 S 的参量式 L(N,D,S) = a/N^α + b/D^β + c/(1-S)^λ + d/((1-S)^δ N^γ) + e，其中 (1-S) 近似活跃参数占比，乘法交互项 d/((1-S)^δ N^γ) 捕捉 N 和 S 的耦合效应。拟合结果显示 λ = -0.1666（负值）且 δ ≈ γ ≈ 0.16，验证稀疏度提高确实降低 loss。(2) 下游分析：发现多数任务上 pretraining loss 是 downstream performance 的良好预测器（与 S 无关），但在阅读理解类任务上，相同 perplexity 的稀疏模型比稠密模型表现更差——揭示了 FLOPs per example 在推理阶段的重要性。进一步通过 length-controlled CoT prompting 实验证明 MoE 比同等活跃参数的 Dense 模型从额外推理计算中获益更多。论文未明确说明系统框架、编译框架、kernel调度、硬件架构层——该工作纯算法层 Scaling Law 分析。
    - 解决 baseline 缺陷的方式：(a) baseline 将 N 作为唯一容量维度 → 论文将容量分解为 N（总参数/知识存储）和 FLOPs per example（活跃参数/计算深度）两个独立维度，通过 S 作为控制旋钮；(b) baseline 无法预测最优 S → 论文通过二次 IsoFLOP 曲面拟合发现 L(S; N, C) 的抛物线性，给出给定 N 和 C 下的 S* 解析趋势；(c) baseline 忽略下游任务中推理计算的角色 → 论文区分了"pretraining-efficient sparsity"（S→1 最优）与"inference-beneficial compute"（某些任务需更低 S/更高活跃参数），为推理时动态分配计算提供理论依据。

- baseline方法是什么？
  - Baseline 是每个 client 各自部署独立的 dedicated MoE 实例（dedicated MoE instances），无专家共享。
  - 全栈执行例子（Baseline: dedicated MoE instances, 2 clients × Mixtral-4x7B, TopK=2, 8×A100 40GB）：
    - **算法层**：每个 client 的 MoE 模型独立运行。每层 MoE layer 中，gating network 对输入 hidden states 执行 Softmax(LinearGate(X)) → TopK(k=2) 选择 2 个 experts。client 1 的请求由 client 1 的 experts A,B,C,D 服务，client 2 的请求由 client 2 的 experts A,B,C,E 服务。即使 experts A,B,C 在两个 client 中完全相同（相同权重），也独立加载两份到 GPU 显存。
    - **系统框架层**：vLLM 为每个 model instance 预分配全部 expert 参数的 GPU 显存。两个 model instance 各占用一份完整模型内存，相同 experts 不共享。每个 client 的请求独立进入 gating network，路由到各自的 expert 计算。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：vLLM 使用 Triton kernel 执行 expert FFN 计算。每个 expert 的 nn.Parameter 独立，即使权重相同也各自独立执行 kernel。batch 中 client 1 的 token 和 client 2 的 token 被不同 expert 对象分别处理，即使它们路由到本质上相同的专家。由于单个 client 的请求量有限，每个 expert 的 batch dimension 较小，GPU 计算资源利用率低（高 memory-to-compute ratio 下 SM 占用率不足）。
    - **硬件架构层**：8×NVIDIA A100 40GB。每个 Mixtral-4x7B 的 expert 约 7GB，2 个 model instance 需 ~112GB 纯参数显存。若 2 个 client 还使用了其他不同专家变体，总内存需求更高。由于 MoE 的稀疏执行特性——每次仅激活 TopK 个 experts——GPU 显存在模型参数上饱和，但其 SM 计算能力未充分利用（GPU underutilization）。
  - Baseline 缺陷根因（两个核心问题）：(1) **专家重复导致的显存浪费**：不同 client 部署的 MoE 变体中常包含完全相同的 experts——例如从 MergeKit 等工具组合 off-the-shelf experts 而来——但每个 client 需要独立 instance 加载全部参数，相同 experts 在不同显存空间各自占据一份（Mixtral-8x7B 每 expert 14GB）。在 multi-tenant 环境中，显存很快成为模型数量的瓶颈。(2) **稀疏执行导致的 GPU 计算利用率低**：MoE 每次请求仅激活 TopK 个 experts，单个 client 的请求量不足以在 per-expert 粒度形成大批次。高 memory-to-compute ratio 导致 GPU 在显存用满前就因请求不足而计算资源闲置。多 client 环境下问题加剧——每个 model instance 各自请求量少，无法形成有效的大批量计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：MoEsaic 通过专家去重（deduplication）、合并表示（merged representation）和 fused gate，在多 client 间共享相同专家，减少显存占用并提升批处理效率。
  - 全栈执行例子（MoEsaic, 2 clients, Mixtral-4x7B, TopK=2, 2 shared experts, 8×A100 40GB）：
    - **算法层**：
      - Expert Deduplication：模型加载时对每个 expert 的每个 tensor 计算 128-bit hash digest（如 SHA-512/128 或类似），存入 in-memory dictionary。后续加载的新 expert 计算 hash 后查 dictionary：命中 → 引用已有 tensor（共享显存）；未命中 → 分配新 GPU 显存。不修改任何 expert 权重，不改变模型精度或行为。
      - Merged Expert Representation：初始化后，将去重共享相同 tensor 的 expert 合并为单一 nn.Parameter。每个 client 的 gate 维护一个 expert ID → merged expert ID 的映射表，路由请求时自动指向合并后的表示。
      - Fused Gate：将多个 model instance 的 gating network 合并为单一 fused gate kernel。单次 CUDA kernel 调用完成所有 client 的 Softmax(LinearGate(X)) → TopK 路由，替代 separate gate 逐 model 串行调用。fused gate 维护 per-model gate mapping，输出被正确解析为各 client 对应的 merged expert ID。
      - Lazy Memory Allocation：初始化时用 tiny pseudo experts 占位（几乎零显存），加载参数时才逐步扩容并填充。去重后的 expert 仅保留一份，峰值内存 = 去重后模型大小 + 当前正在加载（尚未去重）的一个 expert。
    - **系统框架层**（vLLM 修改）：
      - Expert 加载：vLLM 原本在 model init 时预分配所有 expert 内存 → MoEsaic 改为 lazy allocation（tiny pseudo experts 初始化，resize 在加载时）。
      - Expert 表示：vLLM 原本 per-layer 所有 expert co-located 在单个 tensor → MoEsaic 拆分为独立 nn.Parameter per expert，支持张量级别共享。
      - Tensor-Parallel Support：vLLM 原本不支持向已部署模型动态添加 TP expert → MoEsaic 新增 Ray workers，每个 worker 加载指定 GPU 的 expert shard，新 expert 继承初始模型的 sharding 方式。
      - Non-disruptive Add/Remove：MoEsaic 支持在无活跃推理时动态添加/移除 model instance，通过独立 expert 表示 + hash dictionary 实现增量去重，无需系统重启。
      - LoRA-like Interface：client 通过类似 LoRA adapter 的接口向其 base MoE 添加新 experts 和 gates。
    - **编译框架层**：论文未明确说明。vLLM 的 Triton kernel 编译不受 MoEsaic 修改影响。
    - **kernel调度层**：Triton kernel 的 expert FFN 计算逻辑不变。关键差异在数据流：(a) 去重后相同 expert 使用单一 nn.Parameter，来自不同 client 的 token 被 Triton kernel 在同一 batch 中处理——client 1 的 8 tokens + client 2 的 6 tokens = 14 tokens batch，而非 baseline 的 8 tokens 和 6 tokens 两个独立 batch。较大的 batch 更充分利用 GPU 并行能力。(b) Fused gate 替代 separate gate：4 model instances 下 separate gate 需 4 次 CUDA kernel 调用，fused gate 仅需 1 次。对小型模型（如 Mixtral-4x1B，expert 计算时间短）节省尤为显著——separate gate 路由延迟每模型增加 8%，fused gate 降至 4%。
    - **硬件架构层**：同一 8×NVIDIA A100 40GB。核心变化：Baseline 中 2 个 Mixtral-4x7B model instances 占用 ~112GB 参数显存（各 ~56GB）；MoEsaic 以 2 shared experts 去重后节省 ~14GB×2=28GB，仅占 ~84GB。扩展到 Mixtral-8x7B (每 expert 14GB)，7 shared experts + 1 unique → 14 model instances 仅需 ~294GB（baseline 需 ~224GB 仅支持 2 instances），可服务 7× 更多变体。Batching 效果：4 instances Mixtral-3x1B 全共享时 per-expert batch size 从 ~10 增至 ~42（4×），NVIDIA Nsight 测量 SM 占用率随共享比例提升而下降（更高效地利用计算资源）。
  - 解决 Baseline 缺陷的方式：
    1. **针对"专家重复导致显存浪费"**：MoEsaic 用 hash-based tensor-level deduplication 检测并共享跨 model instance 的相同专家。Lazy memory allocation 确保仅去重后保留一份显存副本，峰值内存不超过去重后模型 + 当前加载 expert。Mixtral-8x7B 可服务 7× 更多变体，将 multi-tenant MoE 部署从"显存约束"的问题转变为"GPU 数量能力内可扩展"。
    2. **针对"稀疏执行导致计算利用率低"**：MoEsaic 用 merged expert representation 将来自不同 client 的请求在共享专家上自动批处理——多个 client 的少量请求汇聚成有效的计算批量。同时 fused gate 将 multi-model gating 合并为单次 kernel 调用，避免逐 model 串行 CUDA kernel 调用累积的路由延迟。路由开销对大模型（Mixtral-4x7B, expert 计算占比高）几乎可忽略，对小模型（Mixtral-4x1B）通过 fused gate 控制增长。
    3. **正交性与兼容性**：MoEsaic 不修改 expert 权重、MoE 架构或 gating 逻辑。与 quantization、pruning 等内存优化技术正交——去重后的专家可进一步量化。通过 LoRA-like interface 提供与 vLLM 生态的兼容性。
