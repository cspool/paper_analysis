## Fine-Grained Expert Offloading（细粒度专家卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-Grained Expert Offloading 是 FineMoE 提出的 MoE serving 优化范式，与 Coarse-Grained Expert Offloading 相对。核心区别在于 expert pattern 追踪和预测的粒度：Fine-grained 在 **iteration-level**（每个 auto-regressive step）追踪每层 gate network 对每个 expert 的完整概率分布 P_l^{(i)} ∈ R^J，而 Coarse-grained 仅在 **request-level** 聚合 expert activation count（binary hit count）。Shannon entropy 分析表明 fine-grained iteration-level pattern 的 entropy 显著低于 coarse-grained request-level 聚合（图 3b），意味着 fine-grained 保留了更多专家选择的可预测性信息。Entropy 随 iteration 累积逐渐升高并 plateau（约 10 iterations 后，图 3c），因为随着 decode 推进更多 expert 被访问，request-level 聚合使 pattern 逐渐模糊化。FineMoE 通过 Expert Map Store（CPU memory）、Expert Map Searcher（semantic + trajectory similarity search）、异步 Publisher-Subscriber 架构实现细粒度 offloading。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
Fine-Grained Expert Offloading 在 FineMoE 中运转流程（以 Mixtral-8×7B, LMSYS-Chat-1M 为例）：

Step 1 — Inference Context Collection
  每次 auto-regressive iteration 开始前：
  - 提取 semantic embedding: token_ids → Embedding Layer → sem_new ∈ R^{B×4096}
  - 收集前序 trajectory: [P_1, ..., P_{l-d}] from 前 (l-d) 层 gate network

Step 2 — Expert Map Search (异步，不阻塞 forward)
  ┌─ Semantic-based (前 d 层): cos_sim(sem_new, sem_old[1..C]) → 选最高分 iteration y
  └─ Trajectory-based (d 层之后): cos_sim(traj_new, map_old[1..C, :(l-d), :]) → 选最高分 iteration y
  → 从 Expert Map Store (CPU, ≤1K maps, <200MB) 检索最相似 historical expert map

Step 3 — Similarity-Aware Expert Selection
  δ_l = clip(1 - similarity_score, 0, 1)
  从 searched map 中按概率从高到低选取 experts，直到 Σp ≥ δ_l 且 count ≥ K (=2 for Mixtral)
  → high similarity → low δ → 选少量 expert (省 GPU memory)
  → low similarity → high δ → 选更多 expert (防 miss)

Step 4 — Asynchronous Expert Prefetching (异步)
  Prefetch priority = p_{l,j} / (l - l_now)
  GPU task pool 按 priority 调度：cudaMemcpyAsync(host→device) via PCIe 4.0 32GB/s
  → Expert Cache (GPU) hash map 更新: expert_id → GPU memory location

Step 5 — Expert Serving
  Gate network 选 top-K experts → 查 Expert Cache:
  - Hit: CUDA GEMM kernel 直接用 GPU 上 weights 计算
  - Miss: 暂停所有 prefetch → 立即 cudaMemcpy on-demand load → 恢复 prefetch

Step 6 — Expert Eviction
  Expert Cache 超 GPU memory budget 时:
  Eviction priority = 1 / (p_{l,j} × freq_{l,j})
  驱逐最低 eviction priority (最不重要) experts 到 CPU

Step 7 — Expert Map Update
  新 iteration 的 expert map 写入 Store → 超容量时计算 redundancy score:
  RDY = (d/L) × score_sem + ((L-d)/L) × score_traj
  剔除冗余 maps 维持多样性
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 基于 MoE-Infinity 代码库（https://github.com/TorchMoE/MoE-Infinity）在 HuggingFace Transformers 上构建。Expert Map Store 用 Python/PyTorch/NumPy ndarray 实现，Expert Cache 用 C++ CUDA Runtime API 管理 GPU memory。Multi-GPU deployment 使用 Expert Parallelism（EP）with round-robin expert distribution + hash map expert-to-GPU mapping。整体 latency overhead（context collection + non-async operations）< 50ms (< 1% iteration time)。实验用 6× RTX 3090 24GB + A100 80GB，三个 MoE 模型（Mixtral-8×7B, Qwen1.5-MoE, Phi-3.5-MoE），两个数据集（LMSYS-Chat-1M, ShareGPT）。FineMoE 相比 MoE-Infinity、ProMoE、Mixtral-Offloading、DeepSpeed-Inference 平均降低 TTFT 53-74%、TPOT 22-46%、提升 expert hit rate 14-68%。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
