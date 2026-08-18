## vAttention（基于 CUDA VMM 的动态 KV 内存管理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
vAttention（ASPLOS'25，微软，arXiv:2405.04437，github.com/microsoft/vattention）用 CUDA VMM API 解耦虚拟与物理内存：启动时为每个请求预留连续虚拟地址 slice（按 max context 大小），物理页随序列增长按需提交（demand paging），使 attention kernel 免于 block table、可直接用未修改的非分页 kernel（FlashAttention-2/3、FlashInfer）。技术点：(1) VMM 调用延迟隐藏——异步分配、与计算重叠、延迟回收；(2) 修改开源 CUDA UVM 驱动支持 64/128/256 KB 小页，缓解 2 MB 大页碎片。效果：decode 吞吐最高 1.99×（FA2 非分页）、端到端 1.18–1.23×（vs PagedAttention kernel）。局限（ConServe 视角）：面向单请求设计——multi-turn 下 turn-as-request（vAttention-Turn）使同一 conversation 的历史跨多个不相邻虚拟区；conversation-as-request（vAttention-Conv）须按 max-context 预留 slice（长上下文单会话 KV 可达数十 GB，48-bit VA 也限制并发会话数，且活跃 KV 区相距远 → TLB sub-entry 利用率低）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
vAttention 流程：cuMemAddressReserve 预留 → 序列增长时 cuMemCreate + cuMemMap + cuMemSetAccess 按需映射（与计算重叠）→ 完成时 cuMemUnmap + cuMemRelease。ConServe 把它的两种多轮适配作为 baseline：vAttention-Turn（每 turn 一个 request）与 vAttention-Conv（整会话一个 request）。结果：TTFT −31.4%~−43.4%（Turn）/ −8.3%~−19.1%（Conv）、离线吞吐 +8.6%~15.6% / +7.2%~12.1%——Turn 输在跨 turn 分散（attention 暴露 B×k 个区域），Conv 输在 max-context 预留破坏跨会话翻译局部性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源 repo 含 allocator、Sarathi-Serve 集成、benchmark 脚本与修改版 nvidia-vattn-uvm-driver（Web 证据：https://github.com/microsoft/vattention；vLLM 集成讨论见 issue #17612）。使用：作为 PagedAttention 的替代 KV 内存管理，与未修改的非分页 kernel 直接兼容；注意默认 2 MB 页下碎片会限制 batch（vAttention 报告 2 MB 粒度 batch 187/203/56 vs 64 KB 粒度 240/258/68）。

涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
