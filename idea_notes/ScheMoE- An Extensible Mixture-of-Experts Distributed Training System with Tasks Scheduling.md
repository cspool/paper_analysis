## ScheMoE- An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling

- baseline方法是什么？
  Baseline 为现有 MoE 分布式训练系统（Tutel [16] 和 Faster-MoE [14]），它们在 MoE 训练中采用 expert parallelism + data parallelism，通过 all-to-all（A2A）collective 通信完成 token 的 dispatch 和 combine。以 Tutel 为例的全栈执行例子：
  - **算法pipeline**：MoE layer 替换 Transformer fflayer，gating function（softmax + top-k routing）动态选择 expert → 每个 GPU 持有 E/P 个 expert，capacity factor f 控制每个 expert 最大 token 数 C = f × k × B × L / E
  - **系统框架**：Tutel 基于 PyTorch，MoE layer 输入 token tensor I ∈ R^{(E, C, M)} → GPU i 将本地 token dispatch 到对应 expert 所在 GPU j（通过 NCCL-A2A 或 2DH-A2A）→ expert 计算 fflayer（linear1 → GELU → linear2）→ 结果 combine 回原 GPU
  - **编译框架**：论文未明确说明，Tutel/Faster-MoE 均为 PyTorch 原生扩展，不涉及独立编译框架
  - **kernel调度**：Tutel 和 Faster-MoE 将输入 token tensor 按 capacity 划分为多个 chunk 进行通信-计算流水线化（pipelining degree 由用户手动设定或有限搜索空间内的启发式搜索），但 A2A 通信和 expert 计算之间的重叠是 sub-optimal 的——schedule 模式固定、未证明最优性，且当硬件配置或模型配置变化时容易失效
  - **硬件架构**：PCIe 3.0 ×16 intra-node + 100Gb/s InfiniBand inter-node，NCCL-A2A 或 2DH-A2A 顺序执行所有 Send/Recv 操作，intra-node 和 inter-node 带宽无法同时利用

  Baseline 的三个核心缺陷：
  1. **可扩展性差**：Tutel/Faster-MoE 的调度算法与其 A2A 实现紧耦合，新增 A2A 算法或压缩方法需重新设计调度，无法复用
  2. **A2A 带宽利用 sub-optimal**：2DH-A2A 虽利用层次化拓扑，但 intra-node 和 inter-node 通信仍顺序执行，无法同时占用两种带宽
  3. **调度 sub-optimal**：计算-通信流水线的任务执行顺序未经过最优性证明，不同硬件/模型配置下性能不保证

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ScheMoE 提出三个技术组件，对应解决上述三个缺陷：
  1. **模块化任务抽象**：将 MoE layer 的三个关键操作抽象为 AbsCompressor（compress/decompress）、AbsAlltoAll（all_to_all）、AbsExpert（fflayer compute），每个接口可独立替换实现。通过继承抽象基类即可集成新的压缩算法（ZFP/FP16/INT8）和 A2A 算法（NCCL-A2A/1DH-A2A/2DH-A2A/Pipe-A2A），无需修改调度器。
  2. **Pipe-A2A 算法**：引入两个异步 CUDA stream（Intra-Stream 和 Inter-Stream），将 A2A 中的 SR(i,j) 操作按 GPU 对是否同节点分配到不同 stream → intra-node 和 inter-node 通信并行执行，理论加速比 S_max = (M×t₁ + (P-M)×t₂) / max(M×t₁, (P-M)×t₂)。当消息 ≥ 200MB 时实测 1.4×-2× 优于 2DH-A2A。
  3. **OptSche 最优调度算法**：将 MoE layer 的 7 类任务（C1/A1/D1/E/C2/A2/D2）按 r 路输入分区后，在数据依赖约束（4)-(9）下数学证明最优任务执行顺序，确保通信任务被最大程度隐藏。r=2 时最优 CompTask 顺序为：(C₁¹C₁²)→(D₁¹E¹C₂¹)→(D₁²E²C₂²)→(D₂¹D₂²)，CommTask 在前置完成后即时启动。

  ScheMoE 全栈执行例子（CT-MoE, 32 GPU, r=2, ZFP+ Pipe-A2A + OptSche）：
  - **算法pipeline**：gating 输出 I ∈ R^{(32, C, M)} → ZFP_compress(I, rate=8) 将 32-bit 压缩为 8-bit（通信量 ↓4×）→ Pipe-A2A dispatch → ZFP_decompress → expert fflayer 计算 → ZFP_compress → Pipe-A2A combine → ZFP_decompress
  - **系统框架**：PyTorch MoE layer 替换为 ScheMoE.MOELayer，内部 task queue 将 7×2=14 个 sub-tasks 入队 → Profiler 在预热阶段测量各 task 耗时 → Scheduler 按 OptSche 最优顺序调度
  - **编译框架**：论文未明确说明，基于 PyTorch C++/CUDA extension（~1200 行），不涉及独立编译栈
  - **kernel调度**：CUDA stream 管理——compute tasks 在 default stream，Pipe-A2A intra-node SR 在 Intra-Stream、inter-node SR 在 Inter-Stream → 三个 stream 并发执行，当 t_intra < t_inter 时 intra-node 通信被完全隐藏
  - **硬件架构**：32-GPU（8×4 RTX2080Ti），PCIe 3.0 ×16 + 100Gb/s IB，Pipe-A2A 利用两个 stream 同时占用 PCIe 和 IB 带宽

  解决 Baseline 缺陷的方式总结：
  1. **针对"可扩展性差"**：AbsCompressor/AbsAlltoAll/AbsExpert 三层抽象接口让新算法通过继承和虚函数即可接入，Scheduler 通过 Profiler 自动获取新模块的性能模型，无需修改调度代码
  2. **针对"A2A 带宽利用 sub-optimal"**：Pipe-A2A 通过 Intra-Stream/Inter-Stream 双路异步执行消除 intra-node 和 inter-node 通信的串行化瓶颈。理论加速比由 t_intra 和 t_inter 的比例决定，当两者接近时加速最大（式 18）
  3. **针对"调度 sub-optimal"**：OptSche 在给定输入分区度 r 的条件下数学证明了最优任务执行顺序（定理 1），保证了任何满足约束 (4)-(9) 的调度方案无法超越。消融实验验证：Naive → +ZFP 已有 1.9× → +Pipe-A2A 达到 2.2× → +OptSche 最终 2.4× speedup
