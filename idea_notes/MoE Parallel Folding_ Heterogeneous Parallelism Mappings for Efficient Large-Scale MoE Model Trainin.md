## MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

- baseline方法是什么？
  Baseline 为 Megatron-Core 的标准 5D 混合并行（TP+EP+CP+DP+PP），其中 Attention 层和 MoE 层使用**相同的并行映射**——EP group 被限制在 DP 的子组内，最大 EP 度受 DP 度约束。以 Mixtral-8x22B 在 128 H100 GPU（TP=2, EP=4, PP=8）上的训练为例，全栈执行路径：
  - **算法层**：MoE Transformer 每层执行 Self-Attention（全序列密集计算，TP/CP 切分 hidden dim/sequence）→ Router top-K gating → Expert FFN（稀疏计算，仅激活部分 expert）。Attention 和 MoE 层共享同一套 (TP, CP/EP, DP, PP) 并行映射，即 EP=CP 且两者绑定。Token dispatching 使用 full-sequence-based token dropping（需跨 rank 收集 logits 保证一致性，引入额外通信开销）。
  - **系统框架层**：NVIDIA Megatron-Core（https://github.com/NVIDIA/Megatron-LM）。PyTorch 2.5.0 + CUDA 12.6，使用 NCCL 集合通信（All-to-All、AllGather、ReduceScatter）。
  - **编译框架层**：Megatron-Core 框架作为训练系统的核心引擎，负责并行组初始化和通信调度。Baseline 中 Attention 和 MoE 的并行组生成逻辑耦合（EP 从属于 DP），ranks 布局固定。
  - **kernel 调度层**：NCCL collective communication library 处理 EP 的 All-to-All（token dispatch/combine）和 TP 的 AllGather/ReduceScatter。通信算子与 GEMM 计算串行执行，且 EP 的 All-to-All 可能跨越节点间低带宽 InfiniBand（400 Gbps），与 Attention 层密集通信叠加。
  - **硬件架构层**：NVIDIA Eos 集群：DGX H100 节点（8×H100 GPU，NVLink 4th Gen 450 GB/s intra-node，InfiniBand 400 Gbps inter-node），最多 1024 GPU。
  - **核心缺陷**：(1) 统一并行映射导致 sub-optimal——Attention 需要 TP/CP（序列级通信），MoE 适合 EP（token 级通信），但 baseline 强制两者相同；(2) EP 受 DP 约束，最大 EP 度有限，scalability 受限；(3) 通信域不可折叠——当 EP group 跨越节点时，All-to-All 走低带宽 inter-node 链路，通信开销占主导；(4) token-dropping 需 full-sequence 通信收集 logits，额外开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoE Parallel Folding** 策略：将 Attention 层和 MoE 层的并行映射**解耦**，允许各自独立配置最优并行度。Attention 层使用 TP×CP×DP×PP 四维组，MoE 层使用 TP(ETP)×EP×DP(EDP)×PP 四维组（仅 PP group 数量和成员保持一致）。同时设计统一的 flexible token dispatcher 支持 token-dropping（sub-sequence dropping 替代 full-sequence dropping）和 token-dropless 两种范式。

  MoE Parallel Folding 全栈执行路径（以 Mixtral 8x22B 在 128 H100 GPU 上的最优配置 TP=2, EP=8, ETP=1, PP=8 为例）：
  - **算法层（并行映射解耦 + Folding）**：
    - Attention 层：TP=2, CP=1, DP=8, PP=8 → 使用 TP 切分 hidden dim，DP 处理不同 micro-batch。
    - 转换（Attention → MoE）：仅需 reshape 操作（将 sequence/subsequence 展平为 batch of tokens），无显式通信开销。
    - MoE 层：ETP=1, EP=8, EDP=1, PP=8 → EP=8 将 8 个 expert 分布到 8 GPU，无需 ETP（即 MoE 层不做 tensor parallelism），最大化 GEMM 效率。
    - **Folding 效果**：Attention 的 TP(2)×DP(8)=16 个 GPU 被"折叠"到 MoE 的 EP=8 组。通过将 EP 组与 Attention 的 TP/DP 子组折叠，使 EP 的 All-to-All 通信尽可能在节点内 NVLink（450 GB/s）完成，避免跨节点 InfiniBand。
  - **系统框架层（Megatron-Core 修改）**：
    - 并行组生成：实现 generate_mappings() 函数，为 Attention 和 MoE 分别生成独立并行组。Attention ranks 布局 (attn_dp, pp, cp, tp)，MoE ranks 布局 (moe_dp, pp, ep, tp)。
    - Token Dispatcher：统一处理 ETP 和 EP 组合，forward 流程为 Router→Permutation→All-to-All-V(跨 EP)→AllGather-V(跨 ETP)→Expert GEMM→ReduceScatter-V(跨 ETP)→All-to-All-V→Unpermutation。backward 流程中 AG/RS 与 RS/AG 互换。
    - Sub-sequence dropping：基于本地 sub-sequence logits 做 token dropping 决策，无需跨 rank 通信收集 logits（经验验证不影响模型收敛）。
  - **编译框架层**：Megatron-Core 框架，代码开源在 https://github.com/NVIDIA/Megatron-LM。核心修改为并行组生成和 token dispatcher。
  - **kernel 调度层**：NCCL All-to-All-V、AllGather-V、ReduceScatter-V 作为通信原语。MoE Parallel Folding 将 EP 通信限制在更紧凑的组内（folding 使得 EP group ≤ NVLink domain），降低 All-to-All 的通信带宽需求。FP8 训练使用 Transformer Engine 的 delayed scaling。
  - **硬件架构层**：同 baseline（NVIDIA Eos 集群，H100 GPU）。MoE Parallel Folding 通过折叠并行组充分利用节点内 NVLink 高带宽（450 GB/s），减少跨节点 InfiniBand（400 Gbps）通信。FP8 实验在 H100 上达到 631.7 TFLOPS。

  **缺陷 → 方法设计直接映射**：
  - **统一并行映射 sub-optimal → MoE Parallel Folding 解耦 Attention/MoE 并行策略**：Attention 需要 TP/CP（序列级密集通信），MoE 需要 EP（token 级稀疏通信），分离后各自独立优化。Mixtral-8x22B: MCore baseline 46.3% MFU → MCore w/ Folding 49.3% MFU（+3.0pp）。Qwen2-57B-A14B: 35.3% → 39.0%（+3.7pp）。
  - **EP 受 DP 约束 → Folding 使 EP 可折叠到 Attention 任意子组**：baseline 中 EP group 被限制在 DP 子组内，最大 EP=DP。Folding 后 EP 可独立于 DP 扩展，例如 Mixtral-8x22B 使用 EP=8 同时 ETP=1（MoE 层不做 TP），而 Attention 层用 TP=2。
  - **跨节点通信开销大 → Folding 使通信域紧凑化**：当 EP×CP group 超过 8（跨越 NVLink domain），baseline 的 All-to-All 走低带宽 InfiniBand。Folding 将 CP 和 EP 折叠在一起，使 All-to-All 优先走 NVLink。Ablation（Figure 6）：无 Folding 时 CP×EP>8 导致延迟急剧上升，Folding 后保持稳定。
  - **Fine-grained MoE 通信瓶颈 → ETP 替换为 EP**：Fine-grained MoE（Mixtral-8x22B-G8T8）中 ETP 通信占比超 70%（因 expert hidden size 小导致 GEMM 效率低）。Folding 支持用 EP 替代 ETP，EP 的通信开销远低于 ETP（All-to-All vs AllGather+ReduceScatter）。G8T8: MCore baseline 17.1% MFU → MCore w/ Folding 28.8% MFU（+11.7pp，相对提升 68%）。
  - **Token dropping 通信开销 → Sub-sequence dropping**：baseline 的 full-sequence dropping 需跨 rank 收集 logits（额外 AllGather 通信）。Sub-sequence dropping 仅基于本地 logits 决策，零额外通信开销，且经验验证不影响模型收敛（附录 validation loss 曲线与 MCore v0.9 对齐）。
