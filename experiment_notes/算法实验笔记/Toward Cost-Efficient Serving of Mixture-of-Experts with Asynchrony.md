## Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**Asynchronous Expert Parallelism (AEP)** 算法——一种打破 MoE EP serving 中 barrier 同步的新执行范式。核心算法组件：
    - **µ-queuing（层粒度 token 队列）**：将每个 decoding block 的每个 expert 层视为独立调度单元。Token 到达后按 LayerID 分离入队，GPU 空闲时从任意 ready layer 拉取 token 自适应 re-batch 执行。Cold expert tokens 被允许在队列中积累，直到 batch size 足够大（实验中 batch≈128 时 GEMM 达到 near-linear throughput scaling）才被调度执行，避免 HBM-bound 小 batch 效率损失。
    - **Defragging Scheduler（Algorithm 1，伪代码见下文）**：对每个 (block b, expert e) 计算 Score[b][e] = LScore + Q[b][e]，其中 lookahead score LScore = sum_{k=1}^{K} (sum_{e'} Q[(b+k) mod N_B][e'] / N_e) × δ^k，衰减因子 δ ∈ (0,1)，K 为 lookahead 窗口。该算法同时鼓励 defragmentation（通过 lookahead 使 token wave 保持连续）和 queue occupancy 感知（通过 Q[b][e] 项避免过度忽略孤立的 token 碎片）。
    - **异步 Token Merge（Top-K > 1 支持）**：当 K > 1 时，每个 token 被复制 K 份发送到 K 个 expert。Receptor 维护一个 token pool，通过 <RequestID, LayerID> 元组在所有 K 路 expert 输出到达时 merge 为完整 token，然后才移入下一 attention layer 的 µ-queue。
    - **Token 依赖追踪**：每个 token 携带 metadata <RequestID, LayerID, Tensors[] 引用, prefill_length, topk_weights>，使异步乱序执行中仍能正确追踪请求归属和下一层路由目标。
    - **Hot/Cold Expert 自适应调度**：通过将一个 expert 在所有 block 中的层 colocate 到同一 GPU，scheduler 利用 layer 间 precede/ordering 关系将多数 tokens 收敛到 1-2 个连续 block 的 frontier，hot expert 的 tokens 快速积累优先调度，cold expert tokens 延迟积累到高效 batch size。
  - 实验比较：
    - (a) vs SGLang EP Top-1 routing: throughput 2.0-2.7×（取决于 workload 长度），ITL 低负载下略高但高负载下相当或更优。
    - (b) vs SGLang EP Top-2 routing: throughput 提升幅度减小（Top-2 降低 skew + token merge 同步效应）。
    - (c) Multi-node scalability: 16 experts/16 GPUs, throughput 3× vs SGLang, AMoE 从 8 GPU 扩展到 16 GPU 实现 1.92× throughput 提升（SGLang 无提升）。
    - (d) Scheduler ablation: defragging vs MTFS vs FLFS, 验证 defragging 算法在 batch fragmentation 和 forward progress 之间的 Pareto 优势。
- 硬件平台是什么，配置是什么。
  - Lambda：8× A100-SXM4-80GB，NVSwitch 600 GB/s per GPU，CUDA 12.8，NCCL 2.25.1。
  - AWS P4 (multi-node)：2× 8× A100-SXM4-40GB，NVSwitch 600 GB/s，4× 100 Gbps EFA，CUDA 12.4，NCCL 2.22.3。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral 8x7B 修改版：(a) GQA → MQA 减少 KV cache 竞争；(b) expert routing 替换为基于 Dolly 数据集 profiling 的指数分布随机路由（模拟真实 expert load skew）；扩展实验用 16 experts 版 Mixtral（模拟 LLaMA-V4 等更大 MoE）。
  - 数据集/Workload：Databricks-Dolly-15k 用于 profiling expert load distribution。三类 synthetic Poisson arrival decoding workload：Short (input [30,70], output [70,130]), Medium (input [50,150], output [50,250]), Reasonable (input [100,300], output [100,500])。
- 开源情况。论文声明将开源 AMoE，**但论文正文和 arXiv 页面均未给出具体 GitHub URL，当前无法确认开源仓库地址。**

  AEP 算法 pipeline 详解（基于论文 Algorithm 1 + §3.2 描述）：

  ```
  ═══════════════════════════════════════════════════════════
  Algorithm: Asynchronous Expert Parallelism (AEP) 核心
  ═══════════════════════════════════════════════════════════

  // 常量定义
  N_B:  decoding blocks 数量 (e.g., 32 for Mixtral 8x7B)
  N_E:  experts per GPU (depends on placement)
  K:    lookahead window (e.g., K=3)
  δ:    衰减因子 (e.g., δ=0.5)

  // 全局状态 (每个 GPU runtime)
  mu_queue[N_B][N_E]:  每层 token 队列
  token_pool:          等待 merge 的 Top-K 不完全 token

  // ── Step 1: Receptor ──
  // 接收 incoming token batch, 分离入 µ-queue
  function receptor(incoming_batches[]):
    for batch in incoming_batches:
      for token in batch:
        // Top-K merge: 检查是否需要多路输入
        if token.needs_k_merge:   // K > 1 时 expert 输出需 merge
          key = (token.RequestID, token.LayerID)
          merged = token_pool.add_and_check(key, token)
          if merged is not None:
            mu_queue[merged.LayerID.block][merged.LayerID.expert].enqueue(merged)
        else:
          mu_queue[token.LayerID.block][token.LayerID.expert].enqueue(token)

  // ── Step 2: Scheduler (Algorithm 1) ──
  // GPU idle → 选择最优 (block, expert) pair 执行
  function scheduler():
    Scores[N_B][N_E] = 0
    for b = 0 to N_B - 1:
      // 计算 lookahead score
      LScore = 0
      for k = 1 to K:
        b_next = (b + k) mod N_B
        // 前方 block 所有 expert 的总 token 数
        TotalTokens = sum_{e=0}^{N_E-1} mu_queue[b_next][e].size()
        LScore += (TotalTokens / N_E) * (δ ** k)
      
      for e = 0 to N_E - 1:
        if mu_queue[b][e] is not empty:
          Scores[b][e] = LScore + mu_queue[b][e].size()
    
    // 选最高分 layer
    (b_opt, e_opt) = argmax_{b,e} Scores[b][e]
    return (b_opt, e_opt)

  // ── Step 3: Executor ──
  function executor(b, e):
    batch = mu_queue[b][e].drain_all()
    
    // 自定义 CUDA kernel: fuse 多个独立到达的小 batch 为连续 batch
    contiguous_batch = fuse_fragmented_batches(batch)
    
    if is_attention_layer(b, e):
      // 分配 KV cache slot, 查找已有 KV pages
      page_table.allocate_and_lookup(contiguous_batch)
      // fused CPU→GPU transfer: prefill_length, KV page indices
      cuda_stream.transfer_metadata(contiguous_batch)
      // paged attention kernel (from vLLM)
      output = paged_attention_kernel(contiguous_batch)
      // GPU→CPU: expert routing 结果 (indices + weights)
      routing_info = cuda_stream.transfer_back(output)
    else:  // expert layer
      // GEMM fusion: W_expert × input (no metadata dependency)
      output = expert_gemm_kernel(contiguous_batch)
      routing_info = None  // expert 执行不需要回传 routing info
    
    return output, routing_info

  // ── Step 4: Dispatcher ──
  function dispatcher(output_tokens, routing_info):
    for token in output_tokens:
      if it was an attention output:  // 下一层是 expert
        token.assigned_expert = route(token.embedding)  // MoE gating
        token.LayerID = (current_block, token.assigned_expert)
      else:  // 下一层是 attention
        token.LayerID = (current_block + 1, token.attn_dp_rank)
    
    // Permute: 按 expert ID（去 expert）或 DP rank（去 attention）分组
    permuted = sort_by_next_target(output_tokens)
    batches = split_into_groups(permuted)  // 按目标 GPU 分组
    
    communicator.send_async(batches)  // Phase 1: ZeroMQ → Phase 2: NCCL P2P
  ```

  张量计算视角（以单个 token 通过 MoE block 为例）：

  ```
  Token embedding x ∈ R^d (d=4096 for Mixtral 8x7B)
  
  Block b 的 attention 层 (GPU A, attention DP rank):
    q, k, v = W_Q·x, W_K·x, W_V·x         // 投影
    attn_out = MQA(q, k, v, KV_cache)      // Multi-Query Attention
    gate_logits = W_gate · attn_out         // MoE gating
    expert_idx, weight = top_k(gate_logits)  // 选 K 个 expert
  
  Expert 层 (GPU E, 持有 expert e 的所有 block 的层):
    // token 被路由到此 GPU 的 expert e
    expert_out = FFN_e(attn_out)            // W_up·σ(W_gate·x) ⊙ W_down
    // 如果 K>1: 需等待 K 路 expert 输出都在 attention GPU 上 merge
    merged = sum_{k=1}^{K} weight_k · expert_out_k
  
  // 循环到 block b+1 的 attention 层...
  ```
