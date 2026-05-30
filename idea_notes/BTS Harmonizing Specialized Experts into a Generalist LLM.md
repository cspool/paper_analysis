## BTS Harmonizing Specialized Experts into a Generalist LLM

- baseline方法是什么？
  Baseline 方法分为两大类：
  1. **Expert Merging（无学习连接的合并）**：
     - BTM (Branch-Train-Merge)：对 Seed 和各 Expert 的输出 logits 做 Bayes 规则加权 ensemble，不做任何训练。执行流程：输入 → 每个模型独立 forward → 输出 logits → 加权平均 → 输出 token。
     - Model Soup：直接对 Seed 和各 Expert 的权重做均匀平均。执行流程：参数空间线性插值 → 合并后的单模型 forward。
     - Expert Routing：训练一个线性路由器 ∈ R^{dim×n}，基于 prompt 平均 embedding 选择 Seed 或某个 Expert 处理整个序列。执行流程：输入 embedding → 路由器分类 → 选择单模型 → 该模型处理全部 token。
  2. **Expert Upcycling（破坏模块性的 MoE 转换）**：
     - BTX：将 Seed 和 Expert 的 FFN 拷贝为 MoE Expert，训练全部参数。执行流程：输入 → Attention → Router 选 Expert(s) → 加权 FFN 输出。
     - BAM：将 Attention 和 FFN 都改为 MoE/MoA 结构（所有参数参与训练）。

  全栈执行例子（以 BTM 为例）：
  - 算法层：输入 prompt → Seed + Code Expert + Math Expert + Multilingual Expert 各自 forward（每个 20 层 Transformer）→ 各模型分别计算 4 个 logit 向量 → Bayes 加权 ensemble → argmax 输出下一个 token。
  - 系统框架层：论文未明确说明（推理使用标准 PyTorch forward，无特殊 Serving 框架）。
  - 编译/kernel/硬件层：论文未明确说明。

  Baseline **核心缺陷**：BTM/Model Soup 在 Expert 之间**缺乏可学习的中间表示连接**，合并是仅在输出层/参数空间的静态合并，表达能力受限，尤其在跨领域任务（cross-capability）上表现差。BTX/BAM 虽然通过 MoE 训练获得学习连接，但**破坏了模块性**（所有参数参与训练，Expert 不再保持完整独立），无法灵活增删 Expert。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BTS 方法在保持 Expert 参数完全冻结（模块性）的前提下，通过在 Seed（Hub）和 Expert（Spoke）模型层之间插入并训练轻量 **Stitch Layer**，提供**可学习的中间表示连接**，实现 token 级粒度的 Expert 表示融合。

  **设计对缺陷的映射**：
  | Baseline 缺陷 | BTS 设计选择 |
  |---|---|
  | BTM 无学习连接 → 表达能力弱 | 插入可训练的 stitch 层 w_gate + w_proj（264M 参数） |
  | BTX/BAM 破坏模块性 → Expert 不可增删 | 仅训练 stitch 层，Expert 完全冻结 |
  | Expert Routing 整序列级选单模型 → 不能 context-switch | Stitch 层在每个 token 重新计算 gate → token 级 Expert 动态选择 |
  | 无 Cross-capability（交叉领域） | 交替 Experts-into-Hub / Hub-into-Experts 架构使 Expert 之间双向信息流动 |

  **全栈执行例子（BTS 推理流程）**：
  - 算法层：输入 prompt → Layer 1-4：Seed/Experts 各自 forward → Stitch Layer 1（Hub-into-Experts）：对每个 Expert，sigmoid gate 控制 Hub 信息注入比例 → Layer 5-9 forward → Stitch Layer 2（Experts-into-Hub）：softmax gate 控制各 Expert 投影到 Hub 空间并加权合并 → Layer 10-14 forward → Stitch Layer 3（Hub-into-Experts）→ Layer 15-19 forward → Stitch Layer 4（Experts-into-Hub）→ Hub 最后层输出 → LM head → token 预测。Gate 值可视化验证：Math 任务时 Math Expert gate → 1，Translation 任务时 Multilingual Expert + Seed gate 交替激活，Context-switching 场景下 gate 动态切换。
  - 系统框架层：论文未明确说明（使用标准 PyTorch forward，无需修改 Serving 框架）。
  - 编译/kernel/硬件层：论文未明确说明。

- baseline方法是什么？
  **Baseline**: DeepSpeed-MoE 的 **expert parallelism (EP)** : 每 GPU 持有若干完整 expert（全部 W_i, W_o 矩阵），self-attention 和 router 层复制。Forward pass: (1) 每 GPU 独立执行 router，分配 token→expert; (2) all-to-all scatter 将 token 发送到持有对应 expert 的 GPU; (3) 各 GPU 本地执行 expert FFN (x · W_i · W_o); (4) all-to-all gather 结果回源 GPU。使用 capacity factor (CF) 限制每 expert 最大 token 数，超限 token 被丢弃。

  **Baseline 缺陷**:
  1. **Load Imbalance**: 真实推理中 expert popularity 高度倾斜——以 Switch 128-expert 模型为例，最后一层 14 个 expert 收不到任何 token，最繁忙 expert 收到 3105 tokens。导致部分 GPU 过载、部分 GPU 空闲，端到端延迟由最繁忙 GPU 决定。
  2. **Token Dropping**: 使用 CF 缓解不平衡会丢弃超限 token，直接损害模型精度。
  3. **专家复制方案的开销**: 其他方案（Lazarus, Prophet）通过复制热门 expert 到多 GPU 平衡负载，但需要 profiling、重调度和额外 GPU 内存。
  4. **Batch 越大越严重**: 即使 router skew 参数固定，batch size 增大时 token 分配的绝对差异也增大，imbalance 方案（如 DeepSpeed）中 GPU idle time 绝对值增加。

  **Baseline 全栈执行例子（以 4 GPU, 128 expert Switch-Base encoder 推理一个 MoE block 为例）**:
  - **算法层**: Switch Transformer top-1 gating, 128 FFN experts → token→expert 路由
  - **系统框架层**: DeepSpeed-MoE expert parallelism → 4 GPU 各持有 32 个完整 expert → all-to-all scatter/gather 通信原语 → CF=min(128, 50)=50 限制 token 数
  - **编译框架层**: 论文未明确说明（PyTorch eager execution + NCCL all-to-all）
  - **Kernel/运行时调度层**: 每 GPU 对持有的 32 个 expert 执行完整矩阵乘法 x·W_i·W_o（单个或多个 kernel launch），GPU 间负载不均导致部分 GPU kernel 提前完成等待 all-to-all gather barrier
  - **硬件架构层**: 4× A100 80GB NVLink 互联 → 最忙 GPU 处理最多 token（SM 全占用），最闲 GPU 提前 idle（SM 空闲等待 all-to-all barrier）

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: MoEShard = **Expert Tensor Sharding (TS)** 替代 Expert Parallelism: 将每个 expert 的 W_i 列切分、W_o 行切分到所有 GPU → 每 GPU 持有所有 expert 的 partial shard → 所有 GPU 处理所有 token 的 partial computation → pointwise sum 恢复完整输出。配合两个 kernel 优化: (a) per-expert token concatenation 减少 kernel launch 数; (b) MegaBlocks block-sparse MM 将全部 expert shard 计算融合为单次操作。

  **Defect→Design 映射**:

  | Baseline 缺陷 | MoEShard 设计选择 | 解决机制 |
  |---|---|---|
  | 路由倾斜导致 GPU 负载不均 | W_i 列切分 + W_o 行切分的 expert tensor sharding | 所有 GPU 处理完全相同数量的计算（全部 token × 全部 expert shard），天然 perfect load balancing |
  | CF 丢 token 损害精度 | 所有 token 全程保留 | 每 token 在所有 GPU 上参与 partial computation 并最终求和，零 token dropping |
  | 专家复制需要 profiling 和额外内存 | 无专家复制，无 profiling | 每 GPU 只需每个 expert 的 1/|G| 列/行 shard，总参数量与 EP 相同 |
  | batch 增大加剧 idle time | 计算量与 batch size 线性 scaling | 所有 GPU 的计算量始终相等，无论 batch 多大 |

  **MoEShard 全栈执行例子（以 4 GPU, 128 expert Switch-Base encoder, batch=250, seq=120, h=768 推理一个 MoE block 为例）**:

  - **算法层**: 同 Baseline —— Switch Transformer top-1 gating, 128 FFN experts。区别：sharding 而非 placement 策略改变，路由机制不变。
  - **系统框架层**: MoEShard 自定义 PyTorch forward pass (Algorithm 1) → 每 GPU 复制 router + self-attention → Step 2 metadata exchange (all-to-all broadcast per-expert token counts) → Step 3 token scatter (all GPU send all tokens, NVLink ~0.15ms) → Step 4 sharded expert computation (每 GPU 对每 expert 执行 x · W_i^g · W_o^g) → Step 5 gather + pointwise sum partial outputs → **无 all-to-all scatter/gather 的 barrier 等待**（全部 GPU 计算量相同，同时完成）
  - **编译框架层**: 论文未明确说明（PyTorch eager execution）
  - **Kernel/运行时调度层**: Fusion opt 1: per-expert token concatenation → 128 kernel launches (|E|) vs 512 (|E|×|G|)。Fusion opt 2: MegaBlocks block-sparse MM → 1 kernel launch 处理全部 expert shard。每 kernel 内 SM 计算均匀。
  - **硬件架构层**: 4× A100 80GB NVLink → 4 GPU SM 均计算 x·W_i^g·W_o^g (同数据量，同计算量) → 同时完成 → 直接进入 gather → 无 SM idle。NVLink 仅在 Step 3 scatter 和 Step 5 gather 使用（带宽充足）。

  关键洞察：MoEShard 通过将"按 expert 分配 GPU"改为"按 tensor 维度分配 GPU"，将不可控的路由倾斜问题转化为可控的均匀张量计算问题。代价是 token 全复制（NVLink 吸收）和 partial output 求和（pointwise addition, negligible）。
