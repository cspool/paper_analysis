## LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

- 属于算法pipeline的实现是什么？实验比较什么？
  LOCRET 是一种轻量级训练式 KV cache 淘汰框架。在每层 transformer 注入一个小型 retaining head（两层 MLP，中间维度 d_R=1024），训练 retaining head 预测每个 KV cache unit 的 Causal Importance Score (CIS)。CIS 定义为 answer token 对该 prefix token 的最大 attention score (softmax 前)。推理时在 chunked prefill 过程中每处理一个 chunk，使用 retaining head 对 KV cache 打分，evict 低 CIS 的 cache unit 以维持固定 budget b。同时保留最后 n_s 个 token 的 KV cache 作为 stabilizers 以缓解上下文不连续性。还提出 LOCRET-Q 变体：训练时将 query token 前置以感知 query，推理时 query 插入序列首部实现 query-aware eviction。实验比较 LOCRET vs FULLATTN、InfLLM（offloading）、HF-2BITS（KV cache 量化）、SIRLLM（eviction）、MINFERENCE（sparse attention）在 ∞Bench 和 L-Eval 上的 task accuracy + peak memory，以及 NVIDIA 4090 上的推理速度（tok/s）。LOCRET-Q 与 SNAPKV、H2O、SIRLLM 在 RULER benchmark 上比较 query-driven task 性能。

- 硬件平台是什么，配置是什么。
  训练：单张 NVIDIA A800 GPU（<1 GPU hour）。推理评估（非 4090 实验）：工作站 8×NVIDIA A800/H800 GPU + 104 Intel Xeon Platinum 8470 CPU + 1.0 TB CPU 内存，Red Hat 4.8.5；单 GPU 运行除 FULLATTN 外所有实验，FULLATTN 用 2 GPU（vLLM tensor parallelism）。消费级设备速度实验：单张 NVIDIA 4090 24GB + 512 AMD EPYC 9754 CPU + 1.0 TB CPU 内存，PCIe Gen 4 (16GT/s)，Ubuntu 9.4.0。

- 模型是什么。数据集和bench分别是什么。
  模型：Phi-3-mini-128K（MHA，3.8B 参数）和 Llama-3.1-8B-instruct（GQA，8B 参数）。训练数据：LongAlpaca（QA SFT 数据集），3000 steps，seq_len=10240，lr=5e-4，AdamW，warmup=2000 steps，α=0.0025。Benchmark：(1) ∞Bench——R.PassKey、R.Number、E.Sum、E.QA、E.MC、Z.QA、E.Dia、C.Debug、M.Find（平均 ~100K tokens，Z.QA ~2000K）；(2) L-Eval——CodeU、NQ、CUAD、NarrativeQA、QMSum、SPACE（>16K tokens）；(3) RULER——13 子任务 128K context（LOCRET-Q 评估）；(4) LongBench（附录）；(5) 自定义 10M-token R.PassKey（附录 J）；(6) Rock-Paper-Scissors 多轮对话 benchmark（附录 K）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/huangyuxiang03/Locret

  **训练阶段**：
  1. 在每层 transformer 注入 retaining head R：一个小型 FFN，包含两个线性变换 W1 ∈ R^{(d_m + 2d_kv) × d_R} 和 W2 ∈ R^{d_R × h/g}，激活函数 σ 对齐原模型的非线性函数。d_R=1024。
  2. CIS 预测：Ŝ = R([Q, K, V]) = σ([Q, K, V]·W1)·W2。Ŝ[k] ∈ R^{h/g}，第 j 个分量为 token k 在 head j 的预测 CIS。
  3. Ground truth CIS：对于训练实例 d，S[k]_j := max_p (Q_j K_j^T)_{p,k}，其中 p 遍历所有 answer token，k 遍历所有 prefix token。对 GQA 模型，取同一 group 内不同 query head 的最大 attention score。
  4. 训练 Loss：Smooth-L1(Ŝ, S) + α·L2(Ŝ[k], Ŝ[k+1])，后者为相邻 token 平滑项。
  5. 保留 LLM backbone 冻结，仅训练 retaining head 参数。训练开销 < 1 GPU 小时。

  **推理阶段（Algorithm 1）**：
  ```
  Input: Model M, Prompt tokens x, Local length n_loc, Stabilizer length n_s, Budget b, Chunk size B
  // 保留最后 n_loc 个 token 不被 evict
  chunk_positions = split_chunk(0, x.length - n_loc, B)
  K_cache, V_cache, score_cache = [], [], []
  for chunk ∈ chunk_positions:
      begin, end = chunk.begin_pos, chunk.end_pos
      K_chunk, V_chunk, score_chunk =
          M(x[begin:end], K_cache, V_cache)  // forward pass with retaining heads
      K_cache = Concat(K_cache, K_chunk)
      V_cache = Concat(V_cache, V_chunk)
      score_cache = Concat(score_cache, score_chunk)
      if chunk is not the last:
          score_cache[-n_s:] = +∞  // stabilizers: never evict last n_s tokens
      indices = top-b(score_cache).indices  // keep top-b highest CIS tokens
      K_cache, V_cache, score_cache = K_cache[indices], V_cache[indices], score_cache[indices]
  // 处理最后 n_loc 个 token
  K_cache, V_cache, score_cache = M(x[-n_loc:], K_cache, V_cache)
  x_gen = M.generate(K_cache, V_cache)  // decoding with compressed KV cache
  ```

  **LOCRET-Q 变体**：训练时将 query 最后 l_a 个 token 前置到序列首部，收集 CIS labels。推理时 query 插入序列首部确保所有 eviction 操作感知 query。

  **关键超参数**：
  - Phi-3-mini-128K: b=6000, B=3072, n_s=2500, n_loc=100
  - Llama-3.1-8B-instruct: b=16384, B=1024, n_s=2500, n_loc=100
  - Training: lr=5e-4, batch_size=1, max_seq_len=10240, 3000 steps, AdamW, linear scheduler with warmup=2000, α=0.0025

  **关键结果**：KV cache 压缩比 up to 20× (<10% perf loss)；128K+ 长上下文推理在单张 NVIDIA 4090 上可行；10M token 上下文评估（1747.6× 压缩比）100% 准确率；LOCRET-Q >2× prefill speedup on RULER。
