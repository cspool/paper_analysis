## Asynchronous Incremental KV Cache Checkpointing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asynchronous Incremental KV Cache Checkpointing（异步增量 KV Cache 检查点）是 Tarragon 为 AW 故障恢复设计的轻量级状态持久化机制。核心思想：(1) 增量：每 token 每层仅 checkpoint 新产生的 KV segment（而非整个 KV cache），segment 大小 C = 2 × H_kv × (N_hidden_size / H_attn) × S_elem，对于 GQA/MQA 模型显著小于 expert 通信量（Mixtral-8×7B 中仅 ~12.5% of expert traffic）；(2) 异步：使用 one-sided RDMA write 直接写入 checkpoint store，不涉及 receiver CPU；(3) 时机：利用 AW-EW 通信间隙（attention 计算期间 link idle）进行 opportunistic interleaving，不干扰正常推理流量。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Checkpointing 完整流程：
1. **初始化**：AW 启动时分配连续 GPU KV cache region → 通过 RDMA 注册该 region → 建立与 checkpoint store 的 RDMA 连接 → checkpoint store 分配对应 bucket → 交换 base address。
2. **增量更新**：每层 attention 完成后，新 KV segment 写入 GPU KV cache → compute engine 调用 `async_update(segment_addr, segment_size)` → REFE 等待 AW-EW link idle → issue one-sided RDMA write（携带单调递增的 work request ID 作为 sequence number）→ 将 segment 写入 checkpoint store 对应偏移 → 写入 commit record。
3. **顺序保证**：采用 "async log + commit record" 设计（类似 RDMA-backed write-ahead logging），利用 RDMA work request ID 的单调性保证 segment 顺序。
4. **开销**：利用 AW-EW 通信的 bursty 特性（AW-EW link 在 attention 计算期间大量空闲），异步 checkpointing 对推理吞吐的影响 < 0.1%（1148 vs 1147 tokens/s）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：checkpoint store 为独立 C++ 服务（使用 libibverbs），AW 侧 REFE 负责发起 RDMA write。两端均分配固定连续 buffer，通过 base address + offset 计算远程写入地址。
- 与传统 pause-checkpoint-resume 对比：传统方案每次 checkpoint 需暂停整个 pipeline，在 8-token 间隔下吞吐量下降 2.15×。Tarragon 的增量方案无需任何暂停。
- 适用范围：对 attention 采用 GQA/MQA 的模型特别友好（H_kv << H_attn 使 checkpoint traffic 极小）。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---
