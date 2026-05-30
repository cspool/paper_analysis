## APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

- baseline方法是什么？
  Baseline 方法是现有的长上下文推理加速方案，可分为两类：(1) 序列并行方法（RINGATTN、ULYSSES）——通过将长序列分布到多个 GPU 并行加速 attention 计算，但保持精确 attention（FULLATTN），计算量不变；(2) 近似注意力方法（MINFERENCE）——通过稀疏 attention pattern 减少单 GPU 计算量，但缺乏序列并行支持，长输入下扩展性差。STARATTN 首次合并两者，通过在每 host 上 prepend 一个与 local block 等大的 anchor block 并取消通信实现近似分布式注意力。Baseline 的核心缺陷：
  - Challenge 1: Localized Attention Pruning——现有近似注意力方法（H2O、SNAPKV）依赖全局序列的 attention score 来剪枝 KV cache，这与序列并行中各 host 仅持有部分上下文的架构冲突；
  - Challenge 2: Multi-host Scalability——序列并行受限于 attention head 数量（ULYSSES 的 head-splitting 方式），且 STARATTN 随 host 数量增加导致大量 middle context 不可见，性能持续退化；
  - STARATTN 的额外开销：anchor block = local block size（即 l_a = l_b），导致 FFN 中 anchor block 的计算开销过大，限制了加速收益。

  全栈执行例子（Baseline / STARATTN on 8 hosts, 128K input）：
  - 算法pipeline：每 host 持有 l_b=16K 的 local block + l_a=16K 的 anchor block（文档首 16K token 在所有 host 上复制）；仅计算 anchor↔local 之间的 attention，无跨 host 通信；passing block 不处理
  - 系统框架：基于 FLASHATTN kernel 的 HuggingFace Transformers 推理，分布式执行（8-GPU per node）
  - 编译框架：论文未明确说明
  - kernel调度：FLASHATTN kernel（标准 attention mask），无通信调度
  - 硬件架构：8× NVIDIA A800-80GB, NVLink 3.0 + HDR InfiniBand

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 APB，通过三项核心设计解决 baseline 缺陷：

  1. Localized KV Cache Compression（解决 Challenge 1）：使用 LOCRET 的 retaining heads R（小型训练 MLP），在每个 host 上独立对 local KV cache 打分，无需全局序列视图。取 Top-l_p 作为压缩后的 passing block。该设计与序列并行的分布式架构完全兼容。

  2. Compressed Passing Blocks（解决 Challenge 2 的 STARATTN 退化问题）：通过 AllGather 只共享压缩后的 Top-l_p KV pair（l_p << l_b），构造 passing block P_h 作为前序 host 的"关键信息摘要"。即使 host 数量增加，每个 host 仍能通过 passing blocks 获取前序上下文的精华，维持跨 host 的长距离依赖。消融实验（Table 3）证明：移除 passing block 导致 E.MC 从 72 降至 64（-8%），移除 anchor block 则导致任务完全失败（降至 28）。

  3. Smaller Anchor Blocks（解决 STARATTN 的 FFN 开销问题）：APB 使用 l_a = l_b/4 或 l_b/8 的小 anchor block（STARATTN 为 l_a = l_b），大幅减少 anchor block 在 FFN 中的重复计算开销。Wall-time 分解（Figure 5）显示 APB 的 FFN 时间（30.76 ms/block）显著低于 STARATTN（50.01 ms/block）。

  4. Query-Embedded Anchor Block：将 query q 嵌入 anchor block 头部，使 retaining heads 能够感知查询相关信息以更精准地选择相关 KV pair。消融实验（Table 3, No.1-3）表明 query embedding 需与 retaining heads 配合使用才有效果。

  全栈执行例子（APB on 8 hosts, 128K input, l_a=4K, l_p=2K）：
  - 算法pipeline：Context Splitting（l_b=16K, H=8）→ Block Compression（retaining heads 打分 → Top-2K KV pair）→ AllGather Communication（K^C, V^C）→ Modified Attention（[A, P_h, B_h] 三部分联合计算）→ FFN（仅 A+B_h，P_h 丢弃）；Decoding 用 STARATTN stage-2 accurate attention
  - 系统框架：基于 HuggingFace Transformers + 定制 FLASHATTN kernel（修改 attention mask M'）+ NCCL AllGather 通信调度
  - 编译框架：论文未明确说明
  - kernel调度：每层 Transformer：QKV Proj (4.01ms) → Retaining Head (1.72ms) → AllGather K^C+V^C (0.62ms) → Modified FLASHATTN Attention (34.07ms) → O Proj (2.67ms) → FFN (30.76ms)。通信占比仅 ~0.8%
  - 硬件架构：8× NVIDIA A800-80GB, NVLink 3.0 + HDR InfiniBand

  核心创新：APB 在序列并行框架中引入"压缩-传递"机制——每个 host 独立压缩自己的 KV cache，仅将最重要的 Top-l_p 个 KV pair 通过 AllGather 传递给后续 host 作为 passing block。这同时解决了三个问题：(a) KV cache 剪枝不需要全局 attention score（localized scoring by retaining heads）；(b) 多 host 扩展时 passing blocks 确保跨 host 上下文可见性不丢失；(c) 小的 anchor block 减少 FFN 重复计算开销。实验结果证明：passing block + retaining heads + anchor block + query embedding 四组件共同起作用（Table 3 No.0 vs No.6-8），缺一不可。
