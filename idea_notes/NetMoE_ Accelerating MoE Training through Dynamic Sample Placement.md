## NetMoE: Accelerating MoE Training through Dynamic Sample Placement

- baseline方法是什么？
  Baseline 是现有 MoE 分布式训练框架（FastMoE、FasterMoE、SmartMoE），它们采用静态的样本放置策略：每个 GPU 上的训练样本在迭代内固定不变，All-to-All 通信时 tokens 按 routing 结果发送到对应 expert 所在 GPU，计算结果再按原路径返回。FasterMoE 通过动态 expert placement 减少通信量（将热门 expert 复制到多个 device），但调整 expert 位置需要传输大量 expert 参数（expert size 远大于 sample size），因此无法每迭代调整，导致次优。SmartMoE 在 FasterMoE 基础上增加负载均衡，但主要针对计算负载而非通信效率。

  全栈执行例子（FastMoE baseline, MoE-GPT-S, 2 nodes × 8 A800 GPUs）：
  - 算法 Pipeline：GPT-2 backbone MoE 模型，每层 MoE layer 有 gating network（softmax top-K routing, K=2）+ E=16 experts（FFN）。token → embedding → attention → RMSNorm → gating network 产生 `route ∈ N^{I×L×K}` → tokens 按 routing 分发到各 expert 所在 GPU → experts FFN 计算 → 结果按原 token 位置 gather 回来 → + residual → 下一层。
  - 系统框架（训练框架）：FastMoE/FasterMoE/SmartMoE on PyTorch。样本在迭代开始时固定分配到各 GPU（data parallelism 方式），每个 GPU 持有 I/J 个样本。All-to-All scatter + gather 操作通过 NCCL 通信原语在 NVLink（intra-node 400 GB/s）和 InfiniBand（inter-node 100 GB/s）上执行。
  - 编译框架：论文未明确说明（使用 PyTorch 默认 JIT，无自定义编译器）。
  - Kernel 调度：论文未明确说明（使用标准 NCCL All-to-All kernel + cuBLAS GEMM for expert FFN）。
  - 硬件架构：NVIDIA A800-SXM4-40GB GPU，NVLink 400 GB/s intra-node，InfiniBand 100 GB/s inter-node。baseline 下 inter-node 通信量约占总通信量的 50-75%（取决于 expert 分布），而 inter-node 带宽仅为 intra-node 的 1/4，导致 inter-node 通信成为主要瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  NetMoE 提出**动态样本放置（Dynamic Sample Placement）**：在每个 MoE layer 的 All-to-All gather 阶段，不将 tokens 返回原 GPU，而是按优化后的目标位置发送——将原本需要通过 inter-node 传输的 tokens 改为在 intra-node 甚至 intra-device 传输。

  **对应缺陷 1（样本固定 → 跨节点通信浪费）→ Stage 1 跨节点二分图匹配**
  - Baseline 中样本位置固定不变，tokens 的 routing 目标 expert 的分布是动态且有偏的（同一 sample 的 tokens 倾向于路由到相同 expert），导致大量 tokens 需要跨节点传输。
  - NetMoE Stage 1 将"每个 sample 分配到哪个 node"建模为加权二分图最小权完美匹配问题：左侧 P 为 I 个 samples，右侧 Q 为 N 个 nodes（每个 node 容纳 I/N samples），边权重 `W_{i,n} = c_{i,n}^{(l,gather)} + c_{i,n}^{(l+1,scatter)}` 表示 sample i 放在 node n 时产生的跨节点通信量（tokens 数），使用 Kuhn-Munkres (KM) 算法 O(I^3) 求解，使跨节点通信总量最小化。

  **对应缺陷 2（仅靠节点级优化不够 → Stage 2 节点内二分图匹配）**
  - Stage 1 只确定 sample 到 node 的映射，node 内多 GPU 间的各 sample 放置仍有优化空间。
  - NetMoE Stage 2 在每个 node 内独立求解第二个二分图匹配问题：左侧为分配到该 node 的 I/N 个 samples，右侧为该 node 上的 J/N 个 GPUs（每 GPU 复制 I/J 次），KM 算法求解最小化 intra-node 通信量的样本-GPU 分配。N 个 node 的 Stage 2 并行求解。

  **对应缺陷 3（求解效率不足 → 求解全过程被通信+计算隐藏）**
  - KM 算法在 CPU 上后台线程执行，求解时间与 All-to-All scatter + expert computation 重叠（Table 4 显示 KM 时间 0.48ms < scatter+computation 7.13ms），实现零额外开销。
  - Expert Residual Inlining：将 `output = input + expert_output` 从 gather 之后移到 scatter 之后、gather 之前执行，保证样本位置变化后计算逻辑等价。
  - 下一层 routing 预测：将当前层 input 传入下一层 router 提前获取路由结果，为 Stage 1/2 求解提供 `c^{(l+1,scatter)}` 信息。

  全栈执行例子（NetMoE, MoE-GPT-S, 2 nodes × 8 A800 GPUs）：
