## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- baseline方法是什么？
  Baseline 方法包括：(a) **Full Attention**：完整 KV cache 保留在 GPU 显存，对每 token 做完整 attention。优点是精度无损，缺点是 KV cache 显存占用随序列长度线性增长（128K context × batch=8 时 KV cache ~800MB/layer），导致 batch size 受严重限制（60K→max batch 8，122K→max batch 4，244K→max batch 2），超出则 OOM。全栈执行示例：用户输入 prompt 128K tokens → prefill 阶段每层计算 QKV 投影 → FlashAttention 计算完整 attention → 所有层的完整 KV cache 驻留 GPU HBM → decoding 阶段每步从 HBM 读取完整 KV cache → Batch MatMul Q·K^T → Softmax → MatMul A·V → 输出投影 → next token。瓶颈：KV cache 占 GPU 显存 >80%，限制 batch size，吞吐受显存而非计算能力约束。(b) **Dynamic Sparse Attention（Quest/Loki/InfiniGen）**：保留完整 KV cache 但仅对选中的稀疏 KV 对做 attention。Quest 用 chunk-level min-max 近似选择 KV pages，Loki 用 PCA 在低维空间计算注意力分数，InfiniGen 用离线 SVD 投影做 KV 选择并做 CPU offload。全栈执行示例（以 Quest 为例）：128K token 输入 → prefill 阶段完整 FlashAttention 计算并保存完整 KV cache 在 GPU → decoding 阶段每个 query 与每个 chunk（size=16）的 min-key/max-key 做内积估算 attention 上界 → 选 top-k pages → 仅对选中的 KV 做 sparse attention。缺陷：(1) 未减少 GPU 显存占用，KV cache 仍全量存储在 GPU，batch size 无法扩容；(2) 若 offload 到 CPU（InfiniGen），需 fetch 完整 KV 对（key + value），PCIe 传输量大且 KV selection 不精确导致精度下降；(3) 未利用 key cache 的低秩特性压缩存储。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ShadowKV 通过三个核心观察驱动设计，分别解决 baseline 的显存、延迟和精度缺陷：
  
  **(1) 低秩 Key + CPU Offload 解决显存瓶颈（Observation 3.1）**：发现 pre-RoPE key cache 极低秩（rank 160 可达 6× 压缩无精度损失），且同序列内的低秩子空间高度共享但跨序列不同——因此对每序列在线做 prompt-dependent SVD 而非离线 data-independent 投影。Prefilling 阶段对 pre-RoPE K 做截断 SVD，仅保留 A∈R^{s×r} 和 B∈R^{h_kv×r×d} 在 GPU（大小从 s×h_kv×d 降至 sr + h_kv×r×d），value cache 不低秩因此直接 offload 到 CPU。与 data-independent 方法（如 Palu 训练投影矩阵）不同，在线 SVD 自适应每个 prompt 的低秩结构。
  
  **(2) Landmark 近似 + Outlier 静态缓存解决精度（Observation 3.2）**：发现 post-RoPE keys 具有空间局部性（chunk 内 cosine similarity 高），可用 chunk 均值作为 landmark 来准确近似注意力选择；同时仅 0.2-0.3% 的 chunk 是 outlier（cosine similarity 显著低），将这些 outlier 的完整 KV 对静态存储在 GPU 上。Decoding 时，用 Q 与 landmarks L 做 MatMul 近似 chunk 级 attention score → ArgTopK 选 k 个 chunk → 仅对选中的 (K+k) × C 个 token 做真实 attention。与 Quest 的 min-max 近似相比，landmark 近似更准确，允许更低的 sparse budget（1.56% vs 6.25%）同时保持精度。
  
  **(3) Multi-Stream Overlap + Cache Mechanism 解决延迟（Section 4.2）**：由于仅需从 CPU fetch value cache（key 可在 GPU 端从低秩投影重建），PCIe 传输量减半；同时 CUDA multi-stream 将 key 低秩重建（GPU compute）与 value CPU fetch（PCIe）并发执行，总延迟 ≈ max(t_recon, t_fetch) 而非两者之和。此外，KV cache temporal locality（相邻 step 命中率 ~60%）通过 cache mechanism 以 index scan 跳过已命中 chunk 的重复操作，减少 60% 重建和传输。
  
  全栈执行示例（128K context, batch=24, Llama-3.1-8B, A100）：
  - **Prefilling**：128K tokens → embedding + QKV 投影 → 对 K_pre-RoPE 做在线 SVD（rank=160，耗时仅 attention 的 3-5%）→ 存 A(128K×160)、B(8×160×128) 在 GPU → K_post-RoPE 分 chunk_size=8 → 16K chunks → 每 chunk 算均值作为 L → cosine similarity 检测出 48 个 outlier chunks → K_outlier、V_outlier 存 GPU → 其余 V offload 到 CPU → 完整 FlashAttention prefill (论文保留 exact prefilling)
  - **Decoding step**：Q(24×32×1×128) × L^T(8×16K×128) → softmax → 聚合 → TopK 选 256 chunks → cache mechanism 对比上一步 indices 发现 60% 命中 → 仅重建 miss 的 ~102 chunks → Stream1 GPU 端 A[I_miss](102×160) × B(8×160×128) → RoPE → K_sparse；Stream2 CPU→GPU cudaMemcpyAsync V[I_miss] → 总延迟 ~1.84ms (overlap后) vs 无 overlap 的 ~3ms → K=[K_outlier(48×8); K_sparse(102×8); K_new(1)] ≈ 1201 ~ 2432 tokens；FlashAttention → FFN → 输出 → next token 的 K 投影到同低秩空间保存
  
  **对比 baseline 的解决效果**：
  - 显存：KV cache GPU 占用降 6-7× → batch size 从 4 扩至 24 (6×) at 122K context
  - 吞吐：Llama-3.1-8B 从 80.78 tok/s (batch=4) 升至 245.90 tok/s (batch=24)，3.04× 加速，超过无限显存理论吞吐 134.30 tok/s
  - 精度：RULER 128K 平均 83.57 vs Full Attention 85.53（仅降 2%），远超 Quest 的 35.52、Loki 的 35.52、InfiniGen 的 59.27
  - PCIe 效率：仅 fetch value（vs InfiniGen fetch KV 对），理论等效带宽 7.2 TB/s = 3.6× A100 原生带宽
