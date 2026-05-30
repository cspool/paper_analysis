## FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

- baseline方法是什么？
  Baseline 为 **Megatron-LM 原生的 Expert Parallelism (EP)**，以及基于 expert scheduling 的 SmartMoE 和 FlexMoE。以 Megatron-LM 的 vanilla EP 为例说明全栈执行路径（GPT 32×1.3B, DP=8, EP=4）：
  - **算法层**：每层 MoE 包含 self-attention（DP）→ gate network（top-2 routing）→ token dispatch（EP group 内 all-to-all）→ expert FFN（SwiGLU）→ token combine（all-to-all）→ residual。Token 的 GPU 分配完全由 gate network 决策决定——每个 token 被路由到 top-2 expert 所在的 GPU，无法调整。
  - **系统框架层**：Megatron-LM 3D 并行（DP + TP + PP），EP 将 experts 分布在 EP group 内（每 group 含每个 expert 恰好 1 个 replica）。All-to-all 通信限定在 EP group 内（size=4），通信时间与 expert 计算时间串行执行（通信期间 GPU 空闲）。
  - **编译框架层**：论文未明确说明（标准 PyTorch CUDA kernel + NCCL）。
  - **kernel 调度层**：NCCL all-to-all collective + PyTorch CUDA kernel（batched GEMM for expert FFN）。无自定义通信-计算重叠 kernel。
  - **硬件架构层**：NVIDIA H100 80GB SXM GPU, 900 GB/s NVLink intra-node, 400 Gbps InfiniBand inter-node。
  - **核心缺陷**：
    1. **Token-to-GPU 固定映射无调度空间**：EP group 内每个 expert 仅 1 个 replica，token 必须计算在 gate 选中的 expert 所在 GPU 上。GPU load 由 expert load 固定，无法通过调度调整。
    2. **Straggler 效应**：最重负载的 GPU 成为瓶颈（straggler），所有 GPU 等待其完成 all-to-all 同步。Expert load 分布动态变化且高度偏斜（training 初期尤其严重），每个 micro-batch 都产生 GPU 空闲。
    3. **Expert scheduling 粗粒度**：SmartMoE 和 FlexMoE 通过调整 expert-to-GPU placement 实现 load balancing，但（a）以 expert 为调度单元导致离散有限调度空间，无法实现最优均衡；（b）placement 调整需迁移大量 expert 参数，无法 per-micro-batch 适应动态 load 变化；（c）SmartMoE 基于长期 load 分布优化，面对 micro-batch 间波动时反而可能劣于 Megatron-LM。
    4. **DeepSpeed 的 padding 浪费**：DeepSpeed 将每个 expert 的 load padding 到最大 expert load，在 load 高度不均衡时浪费大量计算和内存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FineMoE 通过 **token scheduling（替代 expert scheduling）+ graph-theoretic expert placement** 双层设计解决上述缺陷。全栈执行路径（GPT 32×1.3B, DP=8, EP=4, d=2）：
  - **算法层 — Token Scheduling（FineEP, §5）**：
    1. **扩展调度空间**：合并 d=2 个 EP group 为 1 个 FineEP group（8 GPU），利用 Expert Data Parallelism——每个 expert 在多个 GPU 上有 replica（同一 EDP group），token 可选择任一 replica 计算。
    2. **Shuffle Expert Placement**：打乱 EP group 间的 expert placement（如图 3c），使不同 experts 的 EDP groups 交叉，扩大调度空间。例如 expert 0 在 GPU {0,2}，expert 1 在 GPU {0,1}（交叉），而非 expert 0 和 expert 1 都在 GPU {0,2}（无交叉）。
    3. **LPP 建模（§5.1）**：每 micro-batch 将 load balancing 建模为线性规划问题——变量 `x_e^g`（expert e 在 GPU g 的 replica load），约束 `Σ_g x_e^g = load_e`，目标 `min max_g Σ_e x_e^g`。使用 HiGHs 求解器在 CPU 上 warm-start 求解（变量数 O(|E|d)，~100 μs 到 <1 ms）。
    4. **Locality-Aware Routing（§5.2）**：优先将 token 路由到本地 GPU 上的 replica（减少 all-to-all 通信），再路由到 remote replica。
    5. **Distributed + Overlapped Scheduling（§5.3-5.4）**：所有 GPU all-gather 收集 load 信息 → 各 GPU 独立运行确定性调度 → CPU 调度与 GPU token permutation 重叠。
  - **算法层 — Graph-Theoretic Expert Placement（§6）**：
    1. **Symmetric Placement（§6.2）**：无先验 load 知识时，用 Cayley graphs 构造对称 placement（如 8 GPU + 8 experts → cycle graph），保证 max induced subgraph density 最小化。
    2. **Asymmetric Placement（§6.3）**：已知 load 分布时，greedy 确定各 expert 的 replica counts + Monte Carlo sampling 选择最优 placement graph（Equation 3: m = max_{G_max} (1/|G_max| · Σ load_e)）。
    3. **Adaptive Replacement（§6.4）**：后台监控 load → 时间序列预测 → Equation 3 评估 → 触发 placement 更新，token scheduling 负责 fine-grained 均衡，adaptive replacement 处理 coarse-grained 偏差。
  - **系统框架层**：
    1. 基于 Megatron-LM 实现：修改 MoELayer forward（插入 Token Dispatcher）→ 扩展 all-to-all 通信组大小 → 新增 Placement Manager（Python + C++ token scheduling）。
    2. 实现了 Distributed Scheduling 和 Overlapping（利用 Megatron-LM 的 token permutation 阶段）。
    3. 额外实现 SmartMoE 和 FlexMoE 在 Megatron-LM 上用于公平对比。
    4. 集成 DeepEP（high-performance all-to-all backend）和 Pipelining FineEP。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：HiGHs 求解器在 CPU 上执行（单 thread），LPP 求解和 token routing 结果与 GPU CUDA kernel 通过 overlapping 和 warm-start 降低开销。NCCL all-to-all 或 DeepEP 用于通信。无自定义 GPU kernel。
  - **硬件架构层**：4 节点，32×H100 80GB GPU，900 GB/s NVLink + 400 Gbps InfiniBand。

  - 对比 baseline 的改进映射：
    - **Token-to-GPU 固定映射 → Token Scheduling 提供细粒度调度空间**：Vanilla EP 的 token-GPU 映射固定 → FineEP 通过合并 EP groups + shuffle expert placement 创建交叉 EDP groups，使每个 expert 的 token 可在多个 GPU 间选择。调度空间从无到有（O(|E|d) 个变量），可实现 per-micro-batch 的细粒度负载均衡。
    - **Straggler 效应（GPU idle）→ LPP 最小化 max load**：LPP 1 的优化目标直接最小化最大 GPU load（straggler），在 load skewness s<1 时 FineMoE (w/o AR) 即能完美均衡所有 GPU（max_load/avg_load = 1.0）。端到端加速最多 47.6% vs Megatron-LM。
    - **Expert scheduling 粗粒度&动态性不足 → Token scheduling 细粒度&per-micro-batch**：Expert scheduling（SmartMoE/FlexMoE）以 expert replica 为调度单元（离散有限空间，placement 调整缓慢）→ token scheduling 以单个 token 为调度单元（连续优化空间，per-micro-batch 调整）。FlexMoE 的 replica count 调整需迁移参数 → FineMoE 的 token scheduling 仅需 all-gather load 信息（~数 us）+ LPP 求解（~100 μs）。结果：FineMoE 在所有条件下优于 FlexMoE 和 SmartMoE。
    - **DeepSpeed padding 浪费 → 根本无需 padding**：DeepSpeed 通过 padding 使 expert loads 相等 → FineMoE 通过 token scheduling 使 GPU loads 相等，不 padding，不浪费计算。
    - **Long-term 部署 → Graph theory 指导 placement**：Symmetric placement（Cayley graphs）为 unknown load 提供数学最优保证——max induced subgraph density 最小化，保证最坏情况下的均衡能力。Asymmetric placement + Adaptive Replacement 为 known/evolving loads 提供持续优化——greedy replica count 基于 load-per-replica + Monte Carlo 图密度最小化。
