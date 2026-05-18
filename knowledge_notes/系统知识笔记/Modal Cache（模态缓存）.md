## Modal Cache（模态缓存）

术语是什么？

Modal Cache 是 EEVEE 提出的面向多模态模型 serving 的模块级中间结果缓存机制，用于存储和复用 modality-specific modules 的推理输出以消除跨请求重复计算。与传统 LLM serving 中仅用于加速 autoregressive decoding 的 KV Cache 不同，Modal Cache 缓存的是跨请求可共享的模块输出——主要是 visual encoder 产生的 visual tokens 或对应的 KV pairs，使后续引用同一图像/视频/音频的请求可直接复用而无需重跑 encoder。Modal Cache 还引入 compression 机制（按 token-wise attention score 剪除低重要性 visual token）生成尺寸更小的 critical modal cache，优先放置于 GPU memory；完整 full modal cache 保留在 host memory 备用。

从系统架构角度拆解术语：

Modal Cache 在 EEVEE 中的运转流程：

1. **Cache 构建**：当用户提交包含图像（或视频/音频）的请求时，visual encoder 处理输入并生成 visual tokens（BLIP-2 产生 32 个、LLaVA 产生最多 576 个）。Controller 进程将这些 tokens 或对应的 KV cache entries 存储到 GPU shared memory，以 64-bit hash（基于输入内容而非请求 ID）作为索引。每模块维护独立 hashmap：〈hash, tensor, last-access-timestamp〉。Controller 同时维护 global LRU queue 追踪所有 cached entries。

2. **Cache 结构依模型架构不同**：
   - **Encoder-Decoder 模型**（如 BLIP + BERT decoder、BLIP-2 + FlanT5）：缓存 cross-attention 层消费的视觉 token KV pairs。视觉 tokens 通过 cross-attention 注入 text encoder/decoder 的每个 transformer block。
   - **Decoder-Only MLLM**（如 LLaVA + LLaMA、BLIP-2 + OPT）：缓存 decoder self-attention 中 visual prefix 部分的 KV pairs。视觉 tokens 作为 prefix 拼接到文本序列前，后续文本 token 的 attention 可访问这些 prefix。

3. **Cache 复用流程**：当前请求引用已缓存的图像时：(a) Controller 通过 hash 查找 local module hashmaps；(b) 命中时，controller 将 cached tensors 通过 CUDA IPC 传给下游 module process；(c) 下游模块跳过 visual encoder 计算，直接从 cache 加载 visual tokens/KV；(d) Cache loading 通过 CUDA streams 与当前正在执行的计算 kernel 重叠进行。

4. **Cache 压缩（Modal Cache Compression）**：对 visual tokens 计算 token-wise attention score，按 compression budget（如 30%）剪除最低分的 tokens。LLaVA 的 576 visual tokens 可压缩至约 170 个，显著减小 cache 尺寸和加载延迟。论文报告：compression 对模型 score 仅产生轻微下降，但大幅降低 memory 和 I/O 开销。压缩策略是 modular 的，可替换为其他 token pruning 方法（如 MADTP、FastV 等）。

5. **Cache 逐出与 Spill**：
   - GPU memory 使用超过 quota 时，global LRU 选择最久未访问的 entries
   - Stale tensors spill 到 host memory（CPU RAM），或由后台线程释放
   - 后台释放操作与 CUDA kernel launches overlap，避免阻塞 computation pipeline
   - 系统在 GPU memory 紧张或请求量高时优先使用 critical modal cache；资源充裕时加载 full cache 以最大化精度

6. **Pipeline Overlap**：如图 12(c)-(d) 所示，新请求的 encoder computation 可与之前请求的 modal cache loading 重叠执行。当 GPU memory 充足时，full modal cache 的 loading 时间可能超过 encoder inference 时间；使用 critical modal cache 后 loading 时间显著缩短，pipeline 效率进一步提升。

术语一般如何实现？如何使用？

EEVEE 的 Modal Cache 实现在每 GPU 的 controller 进程中。Controller 管理 shared GPU memory pool，module processes 和 scheduler 通过 CUDA IPC handler 访问。Cache 条目标识使用 64-bit 内容哈希以保证跨请求可复用（而非绑定特定请求 ID）。对于 weight-heavy modules（encoder/decoder 参数），controller 在 GPU shared memory 中维护 read-only 参数副本以避免多 process 重复加载——因为每个模块只有一个 vLLM process，但多个请求对该模块的访问通过 shared memory 共享参数。论文在 BLIP VQA/NVLR、BLIP-2、LLaVA、InternVL 等任务上验证，compression ratio 默认设为 30%。Ablation 显示 request-aware reuse 单独贡献 116% capacity 提升。Modal Cache 可推广至其他模态（audio、video、depth、IMU 等），只要对应 encoder 架构可产生可缓存的 token/KV 输出。与 LLM KV Cache 的关系：对 decoder-only MLLM，Modal Cache 实质上是 KV Cache 中 visual prefix 部分的跨请求持久化版本——传统 KV Cache 随请求生命周期创建和销毁，Modal Cache 则在请求之间持久化可共享部分。

涉及论文标题：
- Efficient Multimodal Serving via Module Multiplexing
