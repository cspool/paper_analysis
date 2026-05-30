## HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  HATA提出Hash-Aware Top-k Attention，将learning-to-hash系统性地集成到top-k attention中。与现有方法追求qk scores精确数值估计不同，HATA将query和key映射为二进制hash codes（rbit=128），通过Hamming距离获取相对qk score排序，以极低成本实现top-k key选择。核心实现：(1) Hash Modeling：定义query-key hashing优化问题 min Σ_i s_i||h(q)-h(k_i)||² + η||Σh(k_i)||² + λ||W_H^T W_H-I||，使用sigmoid松弛sign函数支持梯度训练，每head独立训练hash权重W_H∈R^{d×128}；(2) Training Data Construction：prefill阶段从Q/K pairs采样，top 10%标记为正样本(线性衰减标签[1,20])，其余为负样本(标签-1)；(3) HATA Prefill：额外计算K_H=HashEncode(K)缓存hash codes；(4) HATA Decode：HashEncode新Q和K→bitwise_xor+bitcount计算Hamming距离→TopK选N个最近keys→sparse attention。实验比较LongBench-e/RULER accuracy vs Dense/Loki/Quest/MagicPIG/StreamingLLM/H2O/SnapKV，end-to-end/prefill/decode效率，HATA-off vs MagicPIG KVCache offloading，scalability to 14B/32B models和256K context，hash bits/token budget/optimizations ablation。

- 硬件平台是什么，配置是什么。
  48GB HBM GPU (最高149.7 TFLOPS FP16)，96 cores。Ubuntu 24.04，CUDA 12.1，PyTorch 2.4，FlashInfer。效率评估使用batch_size=1~8，sequence length=8K~256K。Offloading实验使用PCIe 4.0 + 48 CPU threads。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B-32K-Instruct (MHA, 32 layers×32 heads, d=4096, ctx=32768)，Llama-3.1-8B-Instruct (GQA, 32 layers×32 heads/8 KV heads, d=4096, ctx=131072)，Qwen2.5-14B-Instruct-1M (GQA, 48 layers×40 heads/8 KV heads, d=5120, ctx=1M)，Qwen2.5-32B-Instruct (GQA, 64 layers×40 heads/8 KV heads, d=5120, ctx=131072)。
  Benchmark：LongBench-e (12 tasks)，RULER (11 tasks, 32K-256K)，InfiniteBench，LongBench-v2，Needle-in-a-Haystack。
  Hash训练数据：Qasper(5短序列)、LSHT和RepoBench-P(各2中序列)、LongBench-v2(2超长序列)，覆盖中英文QA和code understanding，最终150K-300K qk pairs。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/gpzlx1/HATA。代码量：1470行C++/CUDA + 940行Python。算法pipeline：

  **Phase 1: Hash Training**
  ```
  # 对每个attention head独立训练W_H ∈ R^{d×128}
  for each head:
      W_H = init(d, 128)
      for epoch in 1..15:
          for iter in 1..20:                         # 20 iterations/epoch
              # h(x) = 2 * Sigmoid(σ * x @ W_H) - 1, σ=0.1
              loss = ε*Σ_i s_i*||h(q)-h(k_i)||²     # similarity preservation
                   + η*||Σ_i h(k_i)||²               # bits balance
                   + λ*||W_H^T @ W_H - I||           # bits uncorrelation
              W_H = SGD(lr=0.1, momentum=0.9, wd=1e-6)(loss)
  ```
  超参数：σ=0.1, ε=0.01, λ=1.0, η=2.0, chunk_size=32K, 2-3 chunks/epoch

  **Phase 2: HATA Prefill**
  ```
  Q, K, V = Proj(X)                      # standard QKV projection
  K_H = HashEncode(K)                    # [s, 128/32] = [s, 4] INT32
  K_H_cache = K_H                        # cache hash codes
  K_cache, V_cache = K, V                # standard KVCache
  O = DenseAttention(Q, K, V)            # dense output for prefill
  
  # HashEncode(K):
  #   K_H = Sign(K @ W_H)                # [s, d] × [d, 128] → [s, 128] binary
  #   K_H = BitPack(K_H)                 # pack 128 bits → 4 INT32
  ```
  Prefill额外开销：O(s×d×rbit)，rbit=128 ≪ s，实际<1%

  **Phase 3: HATA Decode (核心加速)**
  ```
  Q, K, V = Proj(x)                      # projection for single new token
  K_cache = [K_cache; K]                 # update KVCache
  V_cache = [V_cache; V]
  Q_H = HashEncode(Q)                    # [1, 4] INT32
  K_H = HashEncode(K)                    # [1, 4] INT32
  K_H_cache = [K_H_cache; K_H]           # update hash cache
  
  # Hamming distance computation
  S = bitcount(bitwise_xor(Q_H, K_H_cache))  # [1, s], s=seq_len
  # GQA: aggregate S across shared KV head queries
  
  Idx = TopK(S, N)                       # N = top-k token budget
  K_sparse = Gather(K_cache, Idx)        # [N, d]
  V_sparse = Gather(V_cache, Idx)        # [N, d]
  O = FlashAttention(Q, K_sparse, V_sparse)  # [1, d]
  ```
  Decode复杂度：O(s×rbit/32 + s log N + N×d) vs Dense O(s×d)，N≪s

  **关键性能数据**：
  - Llama2 batch=8 seq=32K: 7.20× speedup over Dense, 1.99× over Loki
  - Llama2 batch=1 seq=256K: 6.51× over Dense, 2.21× over Loki, 1.19× over Quest
  - HATA-off vs MagicPIG on Llama2: 6.04× prefill + 2.54× decode speedup
  - Accuracy: LongBench-e avg 34.60 (Llama2) / 53.94 (Llama3.1) vs Dense 34.47/54.10
