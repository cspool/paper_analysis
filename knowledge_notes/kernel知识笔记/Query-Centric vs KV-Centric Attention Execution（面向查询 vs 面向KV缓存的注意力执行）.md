## Query-Centric vs KV-Centric Attention Execution（面向查询 vs 面向KV缓存的注意力执行）

术语是什么？通过联网搜索让回答具体和精准。
这是两种不同的GPU attention kernel CTA组织策略，用于LLM推理的decode阶段。Query-Centric执行将每个decode query和其完整KV cache独立映射到一个CTA（one-query-per-CTA），各CTA并行执行。该方法调度简单（每个query独立），但在batch内多个query共享prefix时造成冗余的global memory访问：同一shared KV blocks被不同CTA重复从HBM加载到各自的shared memory。代表性实现包括FlashAttention (v2.5.9, tile m=64,n=128)、FlashInfer (v0.2.5, tile m=16,n=128, dynamic CTA partitioning改善load balance)。KV-Centric执行将共享同一prefix的多个queries和对应KV cache打包进一个CTA，共享KV仅从global memory加载一次在CTA内共享复用。该方法减少redundant memory access，但常采用one-size-fits-all tile设计（如FastTree固定两种tile configs、DeFT固定(32,16)、Cascade Inference固定打包参数），当query数小于tile size m时需padding浪费shared memory（I_mem），当KV长度参差不齐时造成execution bubble（I_exe）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
两种策略的对比（以batch含4个query，Q1/Q2共享KV-0/KV-1/KV-3前缀，Q3/Q4共享KV-0/KV-2前缀为例）：

**Query-Centric (FlashAttention)**:
```
// 每个query → 独立CTA
CTA_0: load KV-0, KV-1, KV-3, KV-6 for Q1 → compute → output
CTA_1: load KV-0, KV-1, KV-3, KV-7 for Q2 → compute → output
CTA_2: load KV-0, KV-2, KV-4 for Q3 → compute → output
CTA_3: load KV-0, KV-2, KV-5 for Q4 → compute → output
// 问题: KV-0被加载4次, KV-1被加载2次, KV-2被加载2次 → 冗余
// NCU profiling: FlashAttention KV cache traffic = 4.3-8.7× 理论最小值
```

**KV-Centric (FastTree)**:
```
// 共享KV的queries → 同一个CTA
CTA_0: Q1, Q2 share KV-0, KV-1, KV-3 → compute partial for shared part
CTA_1: Q1 finish KV-6 → compute remaining
CTA_2: Q2 finish KV-7 → compute remaining
CTA_3: Q3, Q4 share KV-0, KV-2 → compute partial for shared part
CTA_4: Q3 finish KV-4 → compute remaining
CTA_5: Q4 finish KV-5 → compute remaining
// Merge partial results across CTAs
// 问题: 使用固定tile (64,32), Q1+Q2只有2个query → 62行padding浪费shared memory
//       CTA_0 (KV-0,1,3) vs CTA_4 (KV-4) KV长度差异大 → tail execution bubble
```

**PAT (Memory-Centric Prefix-Aware)**:
```
// 动态CTA partitioning + multi-tile
CTA_0: Q1, Q2 share KV-0, KV-1, KV-3 → tile (32, 128) for q=2, kv_len~=long
CTA_1: Q3, Q4 share KV-0, KV-2 → tile (32, 64) for q=2, kv_len~=medium
CTA_2: Q1 finish KV-6, Q2 finish KV-7 → tile (32, 32) for q=2, kv_len=short
CTA_3: Q3 finish KV-4, Q4 finish KV-5 → tile (32, 32) for q=2, kv_len=short
// Multi-stream: CTA_0 on stream_A, CTA_1 on stream_B, CTA_2+3 on stream_C
// KV-0仅加载一次（被CTA_0和CTA_1 packing后shared，实际通过prefix tree split决策）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Query-centric实现相对简单——将batch中每个query和其block table entries直接映射到CTA，无额外packing逻辑。FlashAttention/FlashInfer是代表性开源实现。KV-centric实现需要prefix identification和query packing逻辑：FastTree使用compute-oriented cost model做packing并运行两个串行kernel；RelayAttention pack first-level shared prefix使用FlashAttention kernel；DeFT聚合shared KV queries并均衡KV length。PAT在KV-centric基础上引入memory-centric profit model、multi-tile kernel和multi-stream forward，在synthetic batch下相对query-centric FlashAttention/FlashInfer平均降低attention latency 67.8%/52.1%，相对KV-centric FastTree降低3.8-68.9%。选择query-centric还是KV-centric取决于workload特征：无共享prefix时query-centric更简单高效（PAT在无prefix配置下仅1.6% improvement），共享prefix比例高时KV-centric显著更优。

涉及论文标题：
- PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

