## MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts

- baseline方法是什么？
  Vanilla MoE（Top-2 Routing）：每个 MoE 层包含 N 个结构相同的 FFN 专家和一个 Router G=Wx。每个 token 固定选择 Top-2 个 FFN 专家，加权聚合输出：y = Σ g_i * FFN_i(x)。所有 token（无论难易）都激活相同数量的 FFN 专家，导致简单 token（标点、词片段）浪费计算资源。
  全栈执行例子：输入 token x 经过 Router 计算 logits → Top-2 选择 2 个 FFN 专家 → 每个 FFN: x → Linear(D→4D) → GELU → Linear(4D→D) → 加权求和。训练时使用均匀负载均衡损失（所有专家相同 token 分配），推理时 FFN 专家分布在多 GPU 上通过 All-to-All 通信。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MoE++ 引入三种零计算专家（zero/computation expert）与 FFN 专家混合：
  1. **Zero Expert**（输出 0）：使 Top-2 退化为 Top-1，减少简单 token 的 FFN 计算。
  2. **Copy Expert**（输出 x）：允许 token 跳过当前 MoE 层（shortcut），对齐残差网络思想。
  3. **Constant Expert**（输出 α1*x + α2*v，可训练向量 v 和权重 W_c）：用少量参数调整输出。
  同时引入 Gating Residuals（W_g 将前一层路由分数融入当前层）和异构负载均衡（τ 参数控制 ZC/FFN token 分配比例）。

  全栈执行对比（以 τ=0.75, MoE++ 1B/(16+4)E 为例）：

  **算法层（对比核心）**：
  - Baseline：每个 token 固定激活 2 个 FFN。简单 token（如标点","）仍然消耗 2 个 FFN 的计算。
  - MoE++：Router 计算 logits = Wx + W_g * G_prev（加入前层路径信息）→ Top-2 选择。简单 token 可能被路由到 zero + copy expert（跳过该层，0 计算），或 zero + FFN（退化为 Top-1 FFN），或 constant + FFN（用可训练向量微调 FFN 输出）。挑战性 token（如动词"touch"）仍可用满 2 个 FFN。

  **系统框架层**：
  - Baseline：FFN 专家分布在多 GPU，All-to-All 通信同步 token，负载不均时某些 GPU 空闲。
  - MoE++：零计算专家参数极少（constant expert 仅 W_c∈R^{2×D} 和向量 v），可全部部署在每个 GPU 上，无需跨 GPU 通信。token 被路由到 ZC expert 时直接本地计算，消除对应 All-to-All 开销。Expert forward throughput: 1B 模型从 610.9ms → 500.3ms（提升 22.1%）。

  **编译框架层**：论文未明确说明（使用 Megatron 训练框架，未修改编译层）。

  **Kernel 调度层**：
  - Baseline：每个 token 的 2 个 FFN GEMM kernel 必须全部执行。
  - MoE++：ZC expert 无 GEMM（zero/copy 是 O(1) 操作，constant 仅 O(D) 标量操作），等效减少了 GEMM kernel 调用次数。计算复杂度从 O(T) 降至 O(τ*N_FFN*T / (τ*N_FFN + N_ZC))，τ=0.75 时约为 baseline 的 85.7%。

  **硬件架构层**：论文未明确说明（纯算法/软件层面改进）。

  **关键设计思路映射**：
  - **痛点：固定 Top-K 对简单 token 计算浪费** → **零计算专家**：提供 skip/discard/replace 三种低成本路径，token 按难度动态选择 FFN 数量。
  - **痛点：Router 独立决策无层间一致性** → **Gating Residuals**：W_g 矩阵连接前后层路由，稳定异构专家选择，减少 routing score 方差（Fig.6）。
  - **痛点：均匀负载均衡不适用于异构专家** → **异构 Load Balance Loss**：τ 参数控制 FFN vs ZC 的 token 分配比例，τ 越小 ZC 分配越多（throughput 越高但可能性能略降），τ=0.75 为默认平衡点。
