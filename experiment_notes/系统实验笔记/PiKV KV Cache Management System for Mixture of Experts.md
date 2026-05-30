## PiKV KV Cache Management System for Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：PiKV 是一个面向 MoE 架构的并行分布式 KV Cache 管理框架，包含四个协同模块：(1) Expert-Sharded Distributed KV Storage——通过 hash 函数 h(t,e) 将 KV cache 按 token 和 expert 维度分片分布到多 GPU，每个 GPU 仅存储 O(L/G + L/E) 个 token 的 KV；(2) PiKV Routing——支持 7 种路由策略（Base hash/TopK/Load-Balanced/Cache-Aware/Entropy-Penalized/RL-Adaptive/Hierarchical），将 KV 查询复杂度从 O(BLhE) 降至 O(BLhk)；(3) PiKV Compression——集成 LoRA、PyramidKV、ChunkKV、Truncated SVD、FastV、Distillation、Structured Pruning，压缩比 ρ=d/d' 线性降低 read+decode 时间；(4) PiKV Scheduling——实现 H2O/StreamingLLM/QUEST/FlexGen/LRU/LRU+/AdaKV/Duo Attention 等调度策略，基于 per-page utility score（注意力强度+访问热度+复用模式）自适应驱逐。四模块通过异步流水线编排。
  - 实验比较：论文正文未包含独立定量 Evaluation 章节。GitHub README 提供合成 benchmark：(a) Standard MoE vs PiKV 各配置内存占用（100%→30%-85%）和推理速度（1.0×→1.3×-2.5×）；(b) Cache Hit Rate 最高 95%、Throughput 最高 3×、SLO 合规率 99%+；(c) 压缩方案对比：LoRA/Pyramid 2.1×/1.8×/98%，SVD 3.2×/1.6×/96%，Quantization 4.0×/2.2×/94%。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明实验硬件平台。代码仓库支持 CUDA 11.8+、PyTorch 2.0+ 的 GPU 平台；分布式支持 RDMA；FPGA 子系统支持 AMD Alveo U55C（xcu55c-fsvh2892-2L-e）。README 推荐 8GB+ RAM（大模型 16GB+）。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：vLLM（`core/single/vllm_integration.py`），同时支持 DeepSpeed（`core/distributed/deepspeed_integration.py`, ZeRO-1/2/3）。
  - vLLM 修改：(1) 注入 PiKV Routing——在 MoE gating 前替换为标准路由器；(2) 注入 PiKV Compression——KV 写入前压缩 (K,V) 对；(3) 注入 PiKV Scheduling——替换默认 cache eviction；(4) PagedKVCache 多级存储（GPU/CPU/SSD）；(5) DistributedKVCachePool RDMA 跨节点缓存传输与负载均衡；(6) CacheAwarePrefillScheduler TTFT SLO 约束下优化缓存复用；(7) LoadBalanceDecodingScheduler TBT SLO 约束下最大化吞吐。
  - 开源代码：https://github.com/NoakLiu/PiKV（155 commits, Python 84%/CUDA 6.5%/Verilog 4.3%, 4 releases, 最新 v2.1, ICML 2025 ES-FoMo III）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：完全开源（Apache 2.0 风格 LICENSE）。arXiv:2508.06526。
  - 框架输入到硬件执行全过程（PiKV Enhanced, MoE LLM, multi-GPU）：

    **初始化**：加载 MoE 模型 → create_moe() 选路由策略 → PiKVvLLMEngine 配置 world_size/expert_count/top_k → create_compressor('pikv_unified') 选压缩方案 → 选调度策略（AdaKV） → 初始化 PagedKVCache（GPU/CPU/SSD 三级 pool） → 初始化 DistributedKVCachePool RDMA。

    **Prefill**：prompt tokens → embedding → PiKV Router gating（EPLB: load-balanced softmax TopK） → PiKV Compression 压缩 (K,V)（high-importance→LoRA rank-r matvec, medium→PyramidKV, low→FastV tail crop） → Expert-Sharded Storage: hash s(t,e) = (t mod N_tok) ⊕ (e mod N_exp) 分配 shard → circular buffer O(1) 插入。

    **Decoding**：新 query q_t → Router 选 top-k experts g_t → 仅从 g_t shards 检索 KV（page table Γ lookup） → PiKV Scheduling 计算 utility scores u_i（AdaKV: u_i = Σ_j α_j φ_j(i), θ ← θ + γ(η*-η)） → 低分 pages 驱逐 → 命中 pages 解压（LoRA decode: K̂ + W_d W_u K̂ + b） → FlashAttention f_attn(q_t, C[q_t]) → 新 (K,V) 压缩插入。

    **调度维护**：LoadBalanceDecodingScheduler 监控 per-GPU 负载 → 动态调整 token-to-expert → AdaKV 每 Δ tokens MMIO 更新 θ。

    PiKV 核心作用：将 MoE KV cache 从"全量存储+全量访问"转为"稀疏路由+分片存储+压缩+自适应调度"四层协同，内存 O(EL)→O(L/G+KS)，查询 O(BLhE)→O(BLhk)。
