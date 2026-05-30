## HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs

- baseline方法是什么？
  - **Expert Parallelism (EP) on heterogeneous GPUs（DeepSpeed MoE + Tutel/Lina optimizations）**：MoE 模型的 expert 按 expert parallelism 分布到所有 GPU（包括新旧 GPU），attention blocks 在每个 GPU 上复制。每 iteration：attention 计算（本地）→ all-to-all dispatch → expert FFN 计算 → all-to-all combine。由于 EP 不区分 GPU 型号，新旧 GPU 的 compute capability 差异导致更快 GPU 在 attention 完成后等待更慢 GPU 的 expert 计算，产生严重 idle。Tutel/Lina 的优化（grouped GEMM、通信重叠）虽然提升了单 GPU 效率，但没有解决异构场景下新旧 GPU 间的 compute imbalance。
  - **DistEP（naïve attention-expert disaggregation）**：将 attention 和 expert 模块分离——attention 仅在新 GPU 上执行，expert 仅在旧 GPU 上执行。但 attention GPU 和 expert GPU 之间存在严格的数据依赖（attention GPU 需等待 expert GPU 完成上一层 combine 后才开始下一层 attention），导致两侧 GPU 交替空闲，大部分时间在等待对方。DistEP 在 4K 序列长度下吞吐量仅为 HeterMoE 的 56%，甚至比 EP 还差 32%。
  - **Heterogeneity-aware Pipeline Parallelism（Whale, Metis, FlashFlex）**：将不同 pipeline stage 分配给不同 GPU 型号，每 stage 分配不同数量的 layer 来 balance compute time。但在 MoE 场景下有三重限制：(1) 不区分 attention 和 expert 模块——旧 GPU 仍被分配 attention 操作，效率低下；(2) balance 粒度为整 layer，无法像 HeterMoE 那样做 per-layer 的细粒度调整；(3) 内存限制——旧 GPU 可能无法容纳单个 MoE block（包含多个 expert 的权重 + 长序列激活），导致无法形成有效 pipeline。
  - 全栈执行例子（Baseline EP on O1 setup: 6×A40 + 6×V100, Mixtral-D1, 32K sequence）：
    - **模型训练算法层**：Mixtral-D1 (8 layers, hidden=1024, 24 experts, top-2 gating)。每 token → self-attention（各 GPU 复制执行）→ gate routing → all-to-all dispatch → expert FFN (gate_proj/up_proj/down_proj GEMMs) → all-to-all combine。训练 loss = LM loss + auxiliary load balancing loss。FP16 mixed precision + activation checkpointing。
    - **系统框架层**：DeepSpeed MoE (PyTorch v2.2 + DeepSpeed v0.14) with Tutel/Lina optimizations。EP group = 12 GPUs（6× A40 + 6× V100）, 24 experts 均匀分布在 12 GPU（每 GPU 2 experts）。Data parallelism 跨 ZP/EP groups。All-to-all 通过 NCCL collective 实现。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL backend。
    - **kernel 调度层**：NCCL all-to-all dispatch/combine + cuBLAS GEMM (expert FFN) + FlashAttention (仅 A40 支持，V100 不支持)。Attention 在 V100 上使用 xformers memory-efficient attention，受限于 memory bandwidth。A40 完成 attention 后 idle 等待 V100 expert 计算完成 + all-to-all combine。随 sequence length 增长（32K），A40 idle 严重（A40 attention 比 V100 快 3.7× for 64K）。
    - **硬件架构层**：6× A40 (48GB, GA102 Ampere) + 6× V100 (16GB, Volta)，100 Gbps RoCE。A40 支持 FlashAttention v2（利用 Ampere-specific TMA 和 async copy），V100 不支持 FlashAttention（无硬件 MHA 加速），Attention 在 V100 上是 memory-bandwidth bound。
  - Baseline 痛点：
    1. **忽视 MoE 组件异构性（核心痛点）**：新旧 GPU 在 attention 和 expert 上的相对效率差异显著——V100 在 expert 上达到 A40 的 80% 性能（因为 expert 主要是 GEMM，CUDA core 高度优化），但 attention 上 V100 性能远差于 A40（V100 不支持 FlashAttention，64K 序列下 A40 比 V100 快 3.7×）。EP 不区分这两种组件，将 attention 也分配给 V100，导致 V100 成为 attention bottleneck。
    2. **计算负载不均衡导致的 idle**：新 GPU 完成 attention 后必须等待旧 GPU 完成 expert 计算和 all-to-all combine 才能开始下一层 attention。即使旧 GPU 在 expert 上仍有 80% 新 GPU 的性能，attention 的时间差仍导致新 GPU 大量 idle。
    3. **PP 的不足**：pipeline parallelism 的 balance 粒度限制为整 layer，且旧 GPU 内存限制可能导致无法容纳单个 MoE block（包含多个 full expert 权重 + 长序列 activations）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HeterMoE 方法**：通过 attention-expert disaggregation + zebra parallelism + Asym-EA 三个机制，实现异构 GPU 上 MoE 训练的高效利用。
    1. **Attention-Expert Disaggregation（解决痛点 1）**：将每个 MoE transformer layer 的 attention blocks 和 expert blocks 分配到不同 GPU 型号——新 GPU 只执行 attention + gate，旧 GPU 只执行 expert FFN。由于 MoE 训练本就用 EP 的 all-to-all 做 expert 间的 token exchange，将 attention 和 expert 分离到不同 GPU 不引入额外通信（dispatch/combine 的数据总量不变，只是从 "attention GPU 到 attention GPU" 变为 "attention GPU 到 expert GPU"）。同时，expert 权重（占模型参数绝大部分）被 offload 到旧 GPU，减轻了新 GPU 的内存压力。
    2. **Zebra Parallelism（解决痛点 2）**：将 input batch 分为 R 个 microbatch，attention GPU 和 expert GPU 同时处理不同 microbatch。执行顺序：attention GPU 先完成 microbatch j 的 attention → dispatch token 到 expert GPU → expert GPU 计算 microbatch j 的 expert（同时 attention GPU 计算 microbatch j+1 的 attention）→ combine → 下一层。这形成了 "zigzag" 式的跨 GPU 流水线（因此得名 zebra）。Theorem 1 证明了最优 task ordering。同时，每 GPU 内 3 个 CUDA stream（2 通信 + 1 计算）重叠所有通信和计算——dispatch 和 combine 走相反方向，在独立 stream 上不发生竞争。
    3. **Asymmetric Expert Assignment（解决痛点 3）**：当 expert GPU 计算慢于 attention GPU（短序列常见），attention GPU 产生 bubbles。Asym-EA 将部分 expert 迁回 attention GPU 以 balance 计算时间。通过 "gather and squeeze"（Algorithm 1）决定在哪些层 offload 多少 expert：accumulate 跨多层的 bubble（T_E^Exp - T_A^Attn）直到足够 offload 至少一个 chunk（n_2 = n_1·M/N 个 experts per expert GPU），然后在 accumulation 最多的层 squeeze。考虑 attention GPU 内存上限 n_max 和 expert GPU 内存下限 n_min 约束。
  - 全栈执行例子（HeterMoE on O1: 6×A40 + 6×V100, Mixtral-D1, 32K sequence，与 baseline EP 同配置对比）：
    - **模型训练算法层**：与 baseline 相同 MoE 模型结构（Mixtral-D1: 8 layers, hidden=1024, 24 experts, top-2 gating）。差异在于执行方式：
      - ZP group: M=6 attention GPUs (A40) + N=6 expert GPUs (V100)，24 experts 分布在 6 个 V100（每 V100 4 experts），A40 默认不持有 expert。
      - Forward: 每 microbatch j → A40 attention (FlashAttention v2) → dispatch all-to-all (A40→V100) → V100 expert FFN → combine all-to-all (V100→A40) → next layer。下一 microbatch j+1 的 attention 与 microbatch j 的 expert 并行。
      - Backward: 对称执行，gate backward 在 A40 上分两路（confidence scores 分支 + expert outputs 分支），等 V100 发回 expert gradients 后 accumulated.
    - **系统框架层**：基于 PyTorch v2.2 + DeepSpeed v0.14 (3K 行 Python)。ZP engine 管理 ZP group 内的 module splitting 和 3-stream scheduling。使用分离的 NCCL dispatch/combine all-to-all group。NCCL all-to-all wrapper 传入不等 split size（因 Asym-EA 导致不同 GPU 处理不同数量 tokens）。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL backend。
    - **kernel 调度层**：3 个 CUDA stream per GPU：
      - Stream 0 (compute): attention/expert 计算
      - Stream 1 (comm D): dispatch all-to-all
      - Stream 2 (comm C): combine all-to-all
      - Sync via CUDA events
      - A40 上的执行顺序: Dispatch_j → (等 event) Attention_j → Combine_j → Dispatch_{j+1} → (等 event) Attention_{j+1} → ...
      - V100 上的执行顺序: (等 dispatch 到的 data) Expert_j → (等下一个 microbatch data) Expert_{j+1} → ...
      - 关键：Communicate 和 compute 在独立 stream 上重叠——dispatch 和 combine 同时在 V100 上执行，互不干扰
    - **硬件架构层**：与 baseline 相同（6×A40 + 6×V100, 100 Gbps RoCE）。
    - **关键性能数据**：
      | Sequence Length | HeterMoE vs EP | vs DistEP | vs EP (Ideal) | vs Homogeneous 4×A40 |
      |----------------|---------------|-----------|---------------|---------------------|
      | 4K | +22% | +79% | +18% | — |
      | 16K | +67% | +69% | — | — |
      | 32K | +89% (up to 2.29×) | +69% | — | — |
    - **核心设计洞察**：HeterMoE 的本质洞察是 MoE 架构本身包含两种计算特征截然不同的组件（attention 和 expert），且这两种组件在不同代 GPU 上的相对效率不同。旧 GPU 缺乏新 GPU 的 attention 硬件优化（FlashAttention 的 TMA/wgmma），但在 expert GEMM 上仍有不俗表现（V100 = 80% A40 on experts）。因此，与其让旧 GPU 勉强执行 attention（严重低效），不如让其专注 expert 计算，将 attention 全权交给新 GPU。这种 disaggregation 不引入额外通信（因为 EP 本就通过 all-to-all 在不同 GPU 间交换 token），且将 bulky expert 权重从稀缺的新 GPU 内存中卸载。Zebra parallelism 和 Asym-EA 的组合形成了一个 elegant 的两级优化：ZP 解决了 coarse-grained 的 compute-compute/communication 重叠（让新旧 GPU 同时忙碌），Asym-EA 解决了 fine-grained 的 bubble 消除（用 "gather and squeeze" 在气泡最大的层迁回部分 expert 计算）。两者的结合使 HeterMoE 能在仅一半新 GPU 的集群上达到 95% 全量新 GPU 的吞吐。
