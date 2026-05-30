## FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

- baseline方法是什么？
  - **MoE-only token-level overlapping（Tutel）**：仅在 MoE 层内部进行 token-level 的 A2A 通信与 expert 计算的流水线重叠。具体来说，将输入 MoE 层的 token 序列按 token 数量均匀切分为微批次（micro-batches），在 GPU 的两个 stream 上分别执行 A2A 通信和 expert 计算，使不同微批次的通信和计算重叠。
  - 全栈执行例子（以 GPT-MoE-L、32K seqlen、16 GPU、EP=16 为例）：
    - **模型推理算法层**：GPT-MoE decoder-only Transformer，每个 block 包含 masked self-attention + top-1 GShard MoE（每隔一层替换 FFN）。Training 时按 autoregressive 方式计算 causal attention，MoE 层使用 top-1 gate routing tokens 到对应 expert。
    - **系统框架层**：Megatron-LM 训练框架，使用 DP=2（跨节点）+ TP/SP=8（节点内，仅 attention 层）+ EP=16（MoE 层）。Tutel 作为 MoE 训练插件，对 MoE 层进行 token-uniform 微批次划分（overlap degree d=2/4/8/16），在 NCCL A2A 通信 stream 和 CUDA expert computation stream 之间做 token-level overlapping。**但 attention 层的计算和 A2A 通信完全串行**。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：NCCL 2.21.5 提供 A2A dispatch/combine 的集合通信 kernel；FlashAttention 提供 fused attention kernel。Tutel 基于 PyTorch 的 CUDA stream 机制实现通信与计算的 overlap。
    - **硬件架构层**：AWS g5.48xlarge 节点，每节点 8 × NVIDIA A10G-24G GPU，100 Gbps 跨节点网络。
  - **Baseline 痛点**：随着序列长度增长（4K→32K），A2A 通信时间增长斜率大于 expert 计算时间（见 Figure 1a）。在 32K 序列下，expert computation 仅占总执行时间的 21%，A2A 通信成为主导瓶颈。MoE-only overlapping 受限于 MoE 层计算量太小（expert 是轻量 FFN），无法充分掩盖 A2A 通信延迟。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FOLDMOE 方法**：将 token-level overlapping 从 MoE 层扩展到整个 Transformer block，利用 attention 层在处理长序列时的 O(n²) 计算量掩盖 A2A 通信延迟。三项核心设计：
    1. **1A1M 调度**：交错 attention 计算与 expert 计算，消除 aAaM 调度中因阶段不平衡导致的流水线尾部气泡。
    2. **Token Buffer + 时间均匀微批次**：在 attention/MoE 层之间插入 buffer 解耦二者的微批次划分——attention 按时间均匀切片（后序微批次 token 数更少以补偿 causal attention 的计算不平衡），MoE 仍保持 token 均匀切片。
    3. **Quick-start 启发式切片算法**：基于 attention FLOPs 模型（FLOPs(l,c) = (4H+3h)lc + 8H²l）快速确定最优切片方案，最小化启动延迟并最大化饱和阶段重叠。

  - 全栈执行例子（与 baseline 同配置，d=8）：
    - **模型推理算法层**：与 baseline 相同（GPT-MoE、causal attention、top-1 GShard MoE），不改变模型架构或收敛特性（Figure 12 验证 loss curve 一致）。
    - **系统框架层**：基于 Megatron-LM 修改，将每个 Transformer block 重构为四级流水线：attention → A2A dispatch → expert → A2A combine。在 CUDA stream 层面实现：attention 计算 stream 和 A2A+expert stream 并行执行。Token buffer 在 attention 输出端维护，实现 FIFO 解耦逻辑。**注意力计算现在与 A2A 通信重叠**，而非 idle 等待。对比 baseline 的 MoE-only overlapping，FOLDMOE 在 32K seqlen 下将 A2A 关键路径延迟减少约 2x（Figure 11），forward pass 加速 1.94x（d=8），overall 加速 1.49x（vs Tutel）和 2.72x（vs Megatron-MoE）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FlashAttention 与 FOLDMOE 的 micro-batch causal attention 兼容（二者保持相同的 causal mask pattern）。NCCL A2A 通信与 CUDA kernel 在分离的 stream 上 overlap。Overlap degree d 需要 trade-off：d 增大减少流水线气泡但增加 kernel launch overhead（Figure 9 展示 d 从 1 到 16 的吞吐量曲线）。
    - **硬件架构层**：与 baseline 相同（AWS g5.48xlarge + A10G）。
