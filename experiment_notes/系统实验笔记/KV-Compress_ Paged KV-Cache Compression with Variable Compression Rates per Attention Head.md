## KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head

- 属于Serving调度的实现是什么？实验比较什么？
  将 KV-Compress 的 KV cache 压缩方法集成到 vLLM v0.6.0，修改 PagedAttention 的 block 管理机制支持 per-head per-layer 可变 KV cache 大小的 paged attention，并通过 GPU 端 block 管理器实现并行化的 block 分配与调度。核心改动：(1) PagedAttention Block Layout 扩展：将原 vLLM 中每 block 存储所有 layer×all heads 的 KVs 改为每 block 仅存储单个 KV head 的 KVs，block table 从 B×L_max/b 扩展到 B×l×H×L_max/b；(2) GPU 端 Block 管理器：将 block table 和 context lengths 移至 GPU device memory，避免 CPU 端调度在 block 数量变为 l×H 倍后的性能瓶颈，实现 block 计数、分配、preemption 的并行化；(3) Block-level Eviction 调度：压缩后释放被 evicted 的连续 blocks，block 管理器回收后可用于新序列的 prefill 或 decoding；(4) 压缩调度策略：prefill 后 + 当 preemption 即将发生时触发压缩，以最大化 batch 扩展与最小化 preemption。实验比较：throughput benchmark 上 KV-Compress 修改的 vLLM vs vanilla vLLM v0.6.0，在 Llama-3.1-8B on L4 和 Llama-3.1-70B-FP8 on H100 上测量不同压缩率（1x-64x）和不同输入长度（500-12000 tokens）下的总吞吐量（tokens/s）及最大 decoding batch size。

- 硬件平台是什么，配置是什么。
  Llama-3.1-8B-Instruct：单 NVIDIA L4 GPU（24GB），gpu_memory_utilization=0.9，max-model-length=19,000；Llama-3.1-70B-Instruct-FP8：单 NVIDIA H100 GPU（80GB），gpu_memory_utilization=0.96，max-model-length=33,000。Both 配置中 vRAM 受限于大模型参数（L4 上 8B ~16GB + KV cache，H100 上 70B-FP8 ~70GB + KV cache），是 throughput benchmark 的理想场景。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：vLLM v0.6.0（https://github.com/vllm-project/vllm）。KV-Compress 修改版开源在 https://github.com/IsaacRe/vllm-kvcompress/tree/main。核心修改：
  (1) **Block Table 扩展**：原 vLLM block table T ∈ R^{B×L_max/b} 共享索引跨所有 layers 和 heads。KV-Compress 扩展为 T ∈ R^{B×l×H×L_max/b}，每 (layer, head) 对有自己的 block table，block 中仅包含该 head 的 KVs ∈ R^{b×d}（原 block 为 l×H×b×d）。物理 cache 从 l 个 per-layer tensor K^{(m)} ∈ R^{N×H×b×d} 改为单一 unified cache K_u, V_u ∈ R^{N×b×d}。
  (2) **GPU 端 Block 管理器（On-device Allocation）**：原 vLLM 的 block 管理器在 CPU 端运行，scheduling runtime 随 block 数量线性增长。KV-Compress 中 block 数量为 l×H 倍（Llama-3.1-8B: 32×8=256 倍），CPU 端调度 loop 在某些情况下耗时超过 forward pass。因此将 block table、context lengths、free/allocated block tracking 全部移至 GPU，并行计算 block 分配与释放。Prefill 时从 token length 直接计算所需 blocks；decoding 时从 on-device context lengths tensor 并行计算额外 block 需求；preemption 时并行计算所有 layers 和 heads 的 freed blocks。
  (3) **压缩调度集成**：压缩步骤在每次 prefill 后和每次 preemption 即将发生时执行。使用 PyTorch sort API 进行 metric 排序和 block eviction 选择。GPU block 管理器回收 evicted blocks。
  (4) **Block Size**：b=16，所有实验使用 eager mode（无 CUDA graph）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：https://github.com/IsaacRe/vllm-kvcompress/tree/main

  **vLLM + KV-Compress 推理全流程（256 prompts, Llama-3.1-8B on L4, compression rate 32x）**：

  ```
  输入：256 input prompts，固定 output=500 tokens
  ↓
  [1] vLLM Scheduler 初始化
    - GPU block manager 分配 unified KV cache K_u, V_u ∈ R^{N×16×d}
    - Block tables 初始化 T ∈ R^{256×32×8×L_max/16}
    - Context lengths tensor C ∈ R^{256×32×8} on GPU
  ↓
  [2] Prefill Loop（逐 prompt chunk size 调度）
    For each schedulable prompt:
      a) Kernel: QKV projection (cuBLAS) → FlashAttention/PagedAttention kernel
         - 每层每 head 通过 T[seq, layer, head, :] 索引对应 blocks
         - 从 K_u, V_u 中按 block 加载 K,V 到 SRAM
         - Attention 计算（eager mode, no CUDA graph）
      b) Block Allocation: GPU block manager 计算并分配所需 blocks
         - Prefill: 每 head 初始分配相同数量 blocks = ceil(L_prompt/16)
      c) First token generation (TTFT)
      d) Metric Calculation（KVC-w, w=8, p=7）:
         - 对 observation window 内 queries 计算 Σ(A_hij)² 累积到 M ∈ R^{N×16}
         - GQA query-group aggregation
      e) KV-Compress compression iteration:
         - Sort M by (head, metric) → block-level eviction candidates
         - Sort blocks by max metric → select E_s blocks to evict
         - MoveCache: 重排物理 cache 使 evicted blocks 连续
         - Free E_s blocks → GPU block manager 回收
      f) Store logical indices P for this sequence's remaining KVs
  ↓
  [3] Decoding Loop（逐 token 生成）
    For each new token per sequence:
      a) QKV projection → PagedAttention（通过 block table 索引 compressed KV cache）
      b) New KV pair 写入 cache（分配新 block 或填入现有 block 空隙）
      c) Context lengths C updated on GPU
    ↓
    [3a] 压缩调度检查（每次 iteration 后）:
      if 有序列新完成 prefill:
          将该序列加入 compression batch
          执行步骤 [2e] 的压缩流程
      if preemption 即将发生（free blocks 不足）:
          选择 compression batch 中最早未压缩的序列
          执行压缩 → 释放 blocks
          若仍不足：preempt 最低 priority 的序列
    ↓
    [3b] Continual Compression:
      每 step 累积新 token 的 Σ(A_hij)² 到 M
      按需触发基于更新后 metric 的再次 eviction
  ↓
  输出：256×500 generated tokens → detokenize → 文本
  ```

  **GPU Block Manager 并行分配细节**：
  ```
  # Prefill 分配（token length → blocks）
  required_blocks = ceil(prompt_length / 16)  # 每 head 相同
  flat_free_tensor: 长度为 N 的 bool tensor（1=free, 0=allocated）
  allocated = cumsum_prefix_scan(flat_free_tensor)  # GPU parallel prefix scan
  for each sequence s, layer m, head h (parallel on GPU):
      T[s, m, h, 0:required_blocks] = allocated_indices[offset_s_m_h: ...]

  # Decoding 分配（对已有序列的 running heads）
  for each (s, m, h) in parallel:
      last_block_used = C[s, m, h] % 16
      if last_block_used == 0:  # 需要新 block
          allocate one block from free list
          T[s, m, h, C[s, m, h] // 16] = new_block_idx
  ```

  **关键性能数据（Llama-3.1-8B on L4, compression rate 32x）**：
  - L_c=500: 2.54x throughput over vanilla vLLM
  - L_c=2000: ~3x throughput
  - L_c=6000: 4.93x throughput
  - L_c=6000, compression rate 64x: 5.18x throughput
  - Max decoding batch size: 100+ (vs vanilla <20)，compression rate 16x+ 时 observed

  **Llama-3.1-70B-FP8 on H100**：
  - L_c=6000, compression rate 64x: 2.14x throughput
  - L_c=6000, compression rate 8x: 1.8x throughput

  **较大的 input context length 需较大 compression rate 才能观察到近似线性的 batch size 增长**：因为序列需在 prefill 后才能被压缩，即使 cache 空间足够装 10 个 compressed 序列，若无法装 1 个 uncompressed 序列也仍无法扩展 batch。
