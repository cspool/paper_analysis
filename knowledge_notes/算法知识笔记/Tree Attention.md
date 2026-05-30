## Tree Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Tree Attention 是一种拓扑感知的多 GPU 精确注意力并行解码算法，由 Zyphra 和 EleutherAI 提出。核心思想是将 self-attention 重新表述为能量函数（moment generating function）的梯度，利用 logsumexp 和 max 操作的结合律（associative property），将序列维度上的归约操作通过树形归约（tree reduction）并行化。该算法专门针对解码阶段（单个 query token），将 KV cache 在序列维度分片到 p 个 GPU，每 GPU 使用 Flash Attention 2 计算局部 attention，再通过 3 次 AllReduce（max + sum×2）合并全局结果。

理论创新链条：
(1) **Observation 1**：证明 self-attention = ∂F/∂ζ|_{ζ=0}，其中 energy function F(ζ) = log Σ_a exp(q·k_a^T + ζ·v_a^T)，ζ 是 "source" 辅助向量。这是首次推导出 self-attention block 的完整 scalar energy function。
(2) **Associative Reduction (Theorem 1)**：logsumexp 和 max 是 associative 操作，对大小为 N 的数组在 p 个并行处理器上的归约时间为 O(N/p + log p)，当 p=N 时降至 O(log N)。
(3) **Tree Decoding (Algorithm 3)**：解码时 K,V 分片，每 GPU 本地 Flash Attention 2 → AllReduce(max) 获取全局 max → 本地数值稳定化 → AllReduce(sum)×2 获取全局分子分母 → 归一化输出。

从算法pipeline角度拆解术语。
Tree Attention 的算法 pipeline（单 token 解码，p 个 GPU）：
```
输入: q ∈ R^{1×d_h}, K ∈ R^{N×d_h}, V ∈ R^{N×d_h}
分片: 每 GPU_i 持有 K_i, V_i ∈ R^{t×d_h}, t = N/p

Step 1: q 广播到所有 p 个 GPU
  scatter(q, all_gpus)

Step 2: 每 GPU 本地 Flash Attention 2
  o_i, lse_i = FlashAttention2(q, K_i, V_i)
  # o_i ∈ R^{1×d_h}, lse_i = log Σ_j exp(q·k_{ij}^T) ∈ R

Step 3: AllReduce(max) → 全局 max (tree reduction)
  m_global = max(lse_1, lse_2, ..., lse_p)
  # 通信步数 O(log p), 传输 1 个标量

Step 4: 本地数值稳定化
  n_i = o_i × exp(lse_i - m_global)  # 分子, [1, d_h]
  d_i = exp(lse_i - m_global)        # 分母, [1]

Step 5: AllReduce(sum) × 2 → 全局分子分母
  n_global = Σ_i n_i  # [1, d_h], tree reduction
  d_global = Σ_i d_i  # [1],      tree reduction

Step 6: 归一化
  z = n_global / d_global  # 精确 attention 输出
```

与 Ring Attention 的对比：
```
Ring Attention pipeline (p GPU, 单 token 解码):
  for step = 0..p-1:
    o_i, lse_i = FlashAttention2_and_accumulate(q, K_current, V_current)
    Send(K_current, V_current) → GPU_{(i+1)%p}  # P2P
    Recv(K_current, V_current) ← GPU_{(i-1)%p}  # P2P
  # 通信步数 O(p), 每次传输 K,V chunk (2bt×d_h elements)
```

关键性质：Tree Attention 是精确计算（exact attention），数值结果与标准 attention 前向传播完全一致，是 Ring Attention 的 drop-in replacement。

术语一般如何实现？如何使用？
实现：开源在 https://github.com/Zyphra/tree_attention，使用 JAX + Flash Attention 2 (JAX binding) + shard_map。通过 `lax.pmax` (max reduction) 和 `lax.psum` (sum reduction) 调用 NCCL AllReduce。NCCL 自动选择 intra-node ring reduce（NVLink 高带宽）和 inter-node tree reduce（InfiniBand 低带宽），实现拓扑感知通信。

使用场景：长上下文 LLM 解码（>32K tokens），跨多个 GPU 的注意力并行化。适用于 DGX H100 集群（NVLink 4.0）、AMD MI300X（Infinity Fabric）、RTX 4090（PCIe）等硬件。在 Llama 3.1-8B、128K context、8×H100 上，解码延迟比 Ring Attention 快 2-4×。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters
