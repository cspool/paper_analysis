## Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference

- baseline方法是什么？
  - Baseline 是现有 CPU offloading 两类方案：(a) MoE-OnDemand（fetch-on-demand）——所有 expert 参数 offload 到 CPU，gate 选择激活 experts 后按需从 CPU 迁移到 GPU，但 expert selection 与 expert execution 串行执行，直接暴露 PCIe 延迟；(b) MoE-Prefetch（prefetch-all）——在当前 block 执行期间迁移下一个 block 的全部 experts 到 GPU，但需传输全部 expert 参数（如 128/256 个），PCIe 带宽成为瓶颈且 GPU 内存需同时容纳两个 block 的全部 experts。
  - 全栈执行例子（Baseline: MoE-OnDemand, Switch-Base 128 experts, A100 80GB, PCIe Gen4 32GB/s）：
    - **算法层**：传统 MoE gate function 在第 N 个 block 内选择同一 block 的激活 experts。gate(W_gate @ x) → softmax → TopK(k=1) → 选择 1 个 expert。gate 输出与 expert execution 存在数据依赖——必须先知道哪个 expert 被选中，才能执行该 expert 的 FFN。
    - **系统框架层**：FasterTransformer 上的 MoE-OnDemand 实现。non-MoE 参数在 GPU，全部 expert 参数在 CPU。每个 MoE block 执行流程：gate → cudaMemcpy(选中的 expert, CPU→GPU) → expert FFN。gate 计算 (~0.05ms) → PCIe 传输 1 个 expert (~85MB/32GB/s ≈ 2.7ms) → expert FFN (~2ms)。总延迟 ≈ 4.75ms/block，其中 PCIe 传输占 57%。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer CUDA kernel 执行 gate (Linear+Softmax+TopK) 和 expert FFN (2×GEMM+GELU)。cudaMemcpy 在 default stream 上同步执行，阻塞后续 kernel launch。gate 和 FFN 之间因数据依赖无法重叠。
    - **硬件架构层**：单 A100 80GB。PCIe Gen4 32GB/s。CPU 1.8TB DDR4。缺陷：gate→expert 串行依赖导致 PCIe 传输直接暴露在关键路径上；多 GPU expert parallelism 方案下 expert 稀疏激活导致 GPU 利用率低（Switch-Base 128 experts Top-1 仅激活 0.8% experts）。
  - Baseline 核心缺陷根因：传统 MoE block 中 expert selection (gate) 与 expert execution 的**数据依赖在同一 block 内**——gate 必须执行完才知道激活哪些 experts，然后才能执行 expert FFN。这使得无论采用何种 CPU offloading 策略（按需取或全量预取），都无法避免 PCIe 延迟对关键路径的影响。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 Pre-gated MoE，通过 algorithm-system co-design 解决：**(Algorithm)** Pre-gate function 将 expert selection 从"为当前 block 选择"改为"为下一个 block 选择"，消除同一 block 内的 gate→execution 数据依赖；**(System)** Preemptive expert migration 利用 pre-gate 提前知道下一个 block 的 experts，在当前 block 执行期间异步迁移仅激活的 experts。
  - 全栈执行例子（Pre-gated MoE, Switch-Base 128 experts, A100 80GB, PCIe Gen4 32GB/s）：
    - **算法层（解决"gate→execution 同 block 数据依赖"缺陷）**：
      - Pre-gate function 设计：第 N 个 MoE block 的 pre-gate 函数 (轻量 Linear layer: W_pre_gate @ x → softmax → TopK) 输出第 (N+1) 个 block 的激活 expert mask。第一个 block 使用双 gate（传统 gate + pre-gate），最后一个 block 无 pre-gate。
      - 训练：复用 pretrained SwitchTransformer 权重，仅在 fine-tuning 阶段训练 pre-gate function（2,048 steps, lr=0.0001, 与常规 MoE 相同配置）。不对 resource-intensive pretraining 做任何修改。
      - 准确率：Pre-gated MoE 在 Xsum/CB Web QA/SQuAD 三个下游任务上准确率与原始 SwitchTransformer 相当（Rouge-1 差异 < 0.1, ExactMatch 差异 < 1.6），部分配置甚至略优。
      - **对比 baseline**：baseline 的 gate 为"当前 block"选择 experts，导致 gate 输出与 expert execution 串行依赖；Pre-gated MoE 的 pre-gate 为"下一个 block"选择 experts，使 expert execution 可以立即开始（experts 已由上一个 block 的 pre-gate 选定），消除了同一 block 内的数据依赖。
    - **系统框架层（解决"PCIe 延迟暴露在关键路径"和"全量预取浪费带宽"缺陷）**：
      - 分层存储：non-MoE 参数常驻 GPU HBM，全部 expert 参数 offload 到 CPU DRAM。
      - Preemptive expert migration：Block (N-1) 的 pre-gate 输出 A_N → 在 Block (N-1) 的 expert execution 期间 → 异步 cudaMemcpy(A_N 的 expert weights, CPU→GPU)。到 Block N 开始执行时，A_N 已在 GPU memory 就绪。
      - 通信-计算重叠：Expert execution (compute-bound, ~2ms) || Expert migration of A_{N+1} (communication-bound, ~2ms via PCIe 32GB/s)。Pre-gate 本身是轻量 Linear（~0.05ms），几乎不占时间。
      - GPU 峰值内存：Peak_GPU_mem = Non_MoE_M + Act_Exp_N + Act_Exp_{N+1}，仅需容纳非 MoE 参数 + 两个连续 block 的激活 expert 参数。实际峰值仅占 GPU-only 的 23%。
      - **对比 baseline**：(a) vs MoE-OnDemand——Pre-gated MoE 消除了 expert migration 的串行暴露，迁移与计算重叠使延迟降低 1.7×；(b) vs MoE-Prefetch——Pre-gated MoE 仅传输激活 experts（~1-2 个），而非全部 experts（~128 个），传输量减少 ~100×，使 PCIe 带宽不再成为瓶颈。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer CUDA kernel 不变。区别在数据流：baseline 的 cudaMemcpy 在关键路径上与 gate 串行；Pre-gated MoE 的 cudaMemcpy 在独立 CUDA stream 上与 expert FFN kernel 并行。第一个 MoE block 是唯一例外——因无 previous pre-gate 为其选择 experts，仍需串行 gate→cudaMemcpy→FFN。但由于 LLM 通常有数十个 block，大部分 block 可受益于重叠。
    - **硬件架构层**：同一 A100 + PCIe Gen4 硬件。核心变化：PCIe 链路在 baseline 中处于"等待 gate 完成 → 传输 → FFN 等待传输完成"的串行模式；Pre-gated MoE 下 PCIe 持续在 expert execution 期间并行传输下一个 block 的 experts，链路利用率提升。结果：Pre-gated MoE 达到 GPU-only（oracular 上界）的 81% 吞吐，峰值 GPU 内存仅为其 23%。Switch-Large (26.4B, 128 experts) 在 GPU-only OOM 的情况下，Pre-gated MoE 仍能以 42 tokens/sec 运行。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"gate→execution 同 block 数据依赖"**：Pre-gate function 将 expert selection 提前到上一个 block，消除同一 block 内的串行依赖。这是 algorithm 层的根本创新——仅增加一个轻量 Linear layer 就改变了整个 MoE block 的执行语义。
    2. **针对"PCIe 延迟暴露在关键路径"**：Preemptive expert migration 利用"提前知道下一个 block experts"的能力，将 expert 迁移与当前 block computation 重叠，使 PCIe 传输时间完全隐藏在计算时间之后（除第一个 block）。
    3. **针对"全量预取浪费带宽和 GPU 内存"**：仅迁移激活 experts（而非全部），使传输量减少 ~100×（128 experts Top-1 场景），GPU 峰值内存接近 memory-optimal 的 MoE-OnDemand（仅多 0.2%），同时性能接近 GPU-only（达到 81%）。
    4. **通用性**：Pre-gate function 的训练仅需 fine-tuning（复用 pretrained weights），不修改 resource-intensive pretraining 阶段，准确率无损，可直接适用于任何 MoE-based LLM。
