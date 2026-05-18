## PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

- baseline方法是什么？
  Baseline分为两类：(1) **Query-centric kernels**（FlashAttention v2.5.9, FlashInfer v0.2.5）：每个query和对应KV cache独立映射到一个CTA（one-query-per-CTA），GPU上多个CTA并行执行decode attention。全栈执行例子：算法层标准multi-head attention→Serving框架层vLLM v0.9.0 continuous batching→kernel调度层FlashAttention将每个query分配到一个CTA，CTA从global memory加载完整KV cache到shared memory，以tiling pipeline（tile size固定m=64,n=128 for FlashAttention; m=16,n=128 for FlashInfer）执行QK^T和PV→硬件架构层A100 GPU的SMs执行CTA，KV cache从HBM→L2→shared memory→register逐级搬运。NCU profiling显示FlashAttention的KV cache traffic比理论最小值高4.3-8.7×，比PAT高4.1-9.5×。(2) **KV-centric kernels**（FastTree, RelayAttention, DeFT, Cascade Inference）：将共享prefix的多个query与其KV cache放入同一CTA以减少重复KV读。全栈执行例子：算法层标准attention→Serving框架层vLLM block table管理→kernel调度层FastTree用compute-oriented cost model打包shared KV queries到CTA，用固定两种tile configs (64,32)和(16,32)串行执行→硬件架构层仍受限于one-size-fits-all tile design导致shared memory padding浪费（query数<𝑚时padding填充）和tail execution bubble（KV长度差异大时最后完成的CTA拖慢整体）。RelayAttention仅支持单层first-level prefix；RelayAttention++扩展到multi-level但依赖L2 cache而非kernel级复用；DeFT用fixed (32,16) tile和load balancing；Cascade Inference用fixed settings打包。

  Baseline的核心缺陷：(a) Query-centric内核在batch内多query共享prefix时重复从global memory加载相同KV blocks，是memory-bound decode attention的主要瓶颈；(b) KV-centric内核的one-size-fits-all tile设计无法同时适配动态变化的CTA query数和KV长度，造成shared memory/register浪费（I_mem）和execution bubble（I_exe）；(c) 两者均未充分利用workload-level shared prefix结构来系统性地减少global memory bandwidth压力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出PAT，一个prefix-aware attention kernel实现，遵循pack-forward-merge执行范式。四个核心设计对应对冲baseline缺陷：

  **(1) Memory-centric pack scheduler（解决query-centric的redundant KV loads和KV-centric的compute-oriented packing）**：将vLLM block table转为prefix tree（internal node=shared prefix段含𝑙 tokens和𝑠个query），用memory-centric profit model比较split/merge两种scheme——intra-node profit=(s-1)*l*d vs overhead=8*s*d（intermediate写回/读回）；inter-node profit比较merge child到parent CTA的增量收益4*s_j*d - l_u*d，当child query数足够多且parent prefix短时merge更优。TreeHeuristic算法以𝑂(|𝑉|+|𝐸|)线性复杂度遍历prefix tree生成CTA partition。对比FastTree的compute-oriented cost model，PAT的memory-centric model使memory read/write分别降低10.9%和16.7%。

  **(2) Multi-tile kernel + runtime tile selector（解决KV-centric的one-size-fits-all资源浪费）**：Offline通过三约束求解可行(m,n) tile set——① register/shared memory上界约束（m*h*b + n*h*b + 中间结果 ≤ S_smem, per-thread register ≤ S_reg_thr, 总register ≤ S_register）；② bandwidth lower bound（n ≥ LB/(S*C*h*b)，保证in-flight data覆盖memory latency以饱和带宽）；③ CUTLASS constraint（m,n为2的幂且≥16）。A100上得11组可行配置，H100上得12组。Runtime tile selector constant-time决策：m用round-up规则（选≥当前CTA query数的最小可行m避免padding→消除I_mem）；n根据KV length profiling做piecewise决策（长KV偏大n降低CTA concurrency减少tail bubble→消除I_exe；短KV偏小n避免最后tile compute bubble）。相比于PAT-fixed（固定(64,128) tile），PAT的multi-tile降低attention latency 39%。

  **(3) Multi-stream forward + long-KV split（解决execution bubbles）**：为每种active tile配置创建独立CUDA stream并行执行，kernel launch overhead与前置kernel执行overlap。Long-KV split将KV length超过batch均值的CTA沿KV维拆分为多个子CTA，缩短最后完成CTA的时间。相比于PAT-serial（串行multi-kernel），multi-stream降低attention latency 4.8%。PTX profiling显示multi-stream显著减少execution bubble。

  **(4) Lightweight merge kernel with online softmax（解决multi-CTA splitting的merge overhead）**：每个CTA输出per-query/per-head的partial max score、log-sum-exp accumulator和partial value-weighted sum到global memory，merge kernel读取同一query所有partial intermediates以online softmax归并max/sum再归一化partial sum。Merge overhead已纳入pack scheduler的profit model。

  论文方法全栈执行例子（以Qwen3-8B + A100 + vLLM + Conversation trace为例）：
  - 算法层：标准GQA transformer attention (32 heads / 8 KV heads, head dim=128, FP16)，论文未修改算法
  - Serving框架层：vLLM v0.9.0 continuous batching + paged KV cache。每次decode step开始：vLLM维护batch中每request的block table → PAT pack scheduler读取logical block IDs → 构建prefix tree（Conversation trace有三层prefix: 45/351/2126 tokens for Qwen3 tokenizer） → TreeHeuristic遍历tree按profit model决策打包CTA → lazy update在block table未变时跨iteration复用
  - kernel调度层：packed CTA送入forward stage → tile selector为每个CTA选(m,n)（如q=20选m=32; KV=4096选n=128） → CTAs按tile config分组进入各自CUDA stream → cp_async+double buffering搬运K/V tile从global→shared memory → CTA内多query共享同一shared-memory KV tile → QK^T、online softmax stats、PV累加 → partial results写回global memory → merge kernel读回partial results以online softmax合并 → 输出final attention output
  - 硬件架构层：A100-80GB GPU (108 SM, 40MB L2, 1935 GB/s HBM bandwidth)。packing使共享KV block仅从HBM加载一次（而非每query一次）→ 直接减少memory-bound decode attention的主开销。Multi-tile配置在A100上维持83%-86% bandwidth utilization，multi-stream在多种tile config间并行减少execution bubble

  最终效果：synthetic batch下平均降低attention latency 53.5%；真实ToolAgent/Conversation trace下端到端TPOT降低17.0-93.1%（vs FlashAttention/FlashInfer/RelayAttention++）；分布式下Qwen2.5-72B-Instruct TPOT降低14.3-26.7%；MoE下Qwen3-30B-A3B TPOT降低5.53-16.9%。
