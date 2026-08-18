## PagedAttention 与软件 Block Table（分页 KV 缓存与块表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PagedAttention 是 vLLM（SOSP'23）提出的 KV cache 分页管理机制：把每个请求的 KV cache 切成固定大小 block（常见 16–32 token/block；ConServe 用 16），block 从初始化时预分配的共享池按需取用、可位于不连续物理内存；每个序列维护一张软件 block table，把逻辑 KV 块号映射到 block 地址（类比 OS 页表把虚拟页映射到物理帧）。收益：消除按 max 长度预分配连续 KV 的碎片与显存浪费，提高可并发请求数；代价：引入两层翻译（软件 block table 查表 + 硬件页表 VA→PA 走查）并打破虚拟连续性——attention kernel 须散页 gather。vLLM 自报 PagedAttention kernel 比非分页 FasterTransformer kernel 慢 20–26%（主要来自 block table 查表）；ConServe 用 FlashInfer native vs FlashInfer-paged 复现：prefill kernel 慢 12–24%，多轮累积至 1.75×。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
vLLM 中一次 KV 访问的路径：kernel 得逻辑块号 → 查 block table 得块虚拟地址 → 算块内偏移 → 跨块边界再取下一块指针 → 硬件页表把虚拟页翻译到物理帧。block 池在服务初始化时由剩余显存预切；块生命周期由引用计数与前缀哈希（APC）管理。ConServe 的对比测量（A100，Llama-3-8B，8K prefill+1K decode，batch=8，Nsight Compute）：paged 布局长 scoreboard stall 84.64% vs 79.37%、eligible warps/cycle 0.718 vs 0.825、SM/L2/DRAM 吞吐 −22.4%/−16.7%/−21.1%——散页扩大 TLB 工作集、页走查缓存复用下降，翻译延迟成为关键路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现（vLLM）：decode kernel 每 warp 处理 1 个 query token × 1 个 block 的 K token 点积，grid=(num_heads, num_seqs, max_num_partitions)；FlashInfer 把 PagedAttention 实现为 block-sparse attention（paged_kv_t 的 indptr/indices 数组，page_size=1 即向量稀疏，用于 SGLang token 级 KV 裁剪）。使用：动态 batch 分配、前缀缓存（vLLM APC 哈希链按块复用）、KV 逐出（ref count=0 且 LRU 优先）。Web 证据：vLLM 官方 PagedAttention 设计文档（https://docs.vllm.ai/en/latest/design/paged_attention/）与 KV 管理综述（arXiv:2607.02574）确认 block 大小 16–32、block table 逻辑→物理映射机制。

Understanding Inference Scaling 补充视角（ISCA'26，PagedAttention 在容量饱和下的行为）：论文配置 block size B=16 平衡量化粒度与访存效率，并指出 KV 容量可从模型与批参数解析估算（analytical KV sizing），但动态长上下文 reasoning 负载下调度效应与 KV 碎片造成瞬时容量尖峰，解析估计不足以预防 preemption——KV-aware 并发帽须按"可用 HBM headroom + 活跃序列长度 + 预期 decode 增长"在线设定。PagedAttention 的分块管理本身不解决容量问题：它消除内部碎片，但推理负载的总 KV 需求仍超物理 HBM，最终仍触发抢占与重算。
涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
