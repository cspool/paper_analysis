## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- 属于Serving调度的实现是什么？实验比较什么？
  将 InfiniteHiP 的模块化层次化剪枝注意力机制集成到 SGLang LLM serving 框架中，实现单 GPU 上长上下文（最高 3M tokens）的高吞吐推理服务。核心 Serving 修改包括：(1) 在 SGLang 的 attention 计算路径中插入多阶段剪枝 kernel，以 block sparse attention 替代 full attention；(2) 基于 Nvidia UVM（Unified Virtual Memory）实现 KV cache offloading——维护 GPU key bank（cache）+ CPU 统一内存空间（完整 KV cache）+ page table（global-to-local index 映射），采用 LRU 驱逐策略替换 HiP Attention 的原始策略；(3) 稀疏注意力 mask 按 stage 独立缓存，通过可配置的 refresh interval（fast: 32/16/8, flash: 96/24/8）减少解码时 mask 重计算频率；(4) 将 KV cache offloaded attention 实现为 graph-capturable 操作，避免 CPU overhead。实验对比 SGLang Runtime（SRT with FlashInfer）的 end-to-end decoding throughput，在 RTX 4090 24GB 和 L40S 48GB 上测试单 batch 场景（预期单序列即超出显存，因此仅测 batch=1）。

- 硬件平台是什么，配置是什么。
  (1) NVIDIA RTX 4090 24GB（PCIe 4.0 x8）+ AMD Ryzen 7950X 16C/32T + 128GB DDR5 5600MHz + Ubuntu 22.04.4 LTS + GPU Driver 535.171.04；(2) NVIDIA L40S 48GB（AWS g6e.48xlarge 节点）。KV cache offloading 通过 PCIe 4.0 x8（31.5× 比 VRAM 访问更慢的延迟）访问 CPU 内存。

- 开源Serving框架是什么。修改了什么。
  开源 Serving 框架：SGLang（https://github.com/sgl-project/sglang），InfiniteHiP 的修改版开源在 https://github.com/DeepAuto-AI/sglang/。
  修改内容：
  (1) **Attention 层替换**：将 SGLang 的标准 FlashAttention2/FlashInfer attention kernel 替换为 InfiniteHiP 的多阶段剪枝 + Block Sparse Attention pipeline。
  (2) **KV Cache 管理层**：增加 UVM-based KV cache offloading 机制，包括 GPU key bank（两个独立的 bank：mask-selection 用和 BSA 用）、CPU unified memory space、GPU page table、LRU 驱逐策略。
  (3) **Mask 缓存与刷新调度**：每个 pruning stage 维护独立的稀疏 mask 缓存和 refresh counter，根据 configurable interval 周期性更新。
  (4) **Graph Capture 兼容**：将 offloaded attention 实现为 CUDA graph capturable 操作，消除 CPU kernel launch overhead。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源地址：核心库 https://github.com/DeepAuto-AI/hip-attention/，SGLang 集成版 https://github.com/DeepAuto-AI/sglang/。

  **SGLang + InfiniteHiP 推理全流程（单请求 3M token 上下文，L40S 48GB，AWQ Llama 3.1 8B + FP8 KV cache）**：

  ```
  输入：用户 prompt（3M tokens）
  ↓
  [1] Tokenization & Prefill 入口
    - SGLang tokenizer 将输入文本切分为 token 序列
    - 创建请求对象，分配序列 ID
  ↓
  [2] Prefill Stage（逐 chunk 处理，chunk_size=32K）
    For each chunk of 32K tokens:
      a) QKV Projection: GPU 上对当前 chunk 执行 linear projection
      b) InfiniteHiP Context Pruning:
         - Stage 0: 从全部 KV cache（含 host memory 中已有 token）选 32K key
         - Stage 1: 从 32K chunk 中通过 SelectRep + max chunk score 选 8K key
         - Stage 2: 从 8K 中选 ~2-4K key（基于 preset）
         BSA: 仅对这 ~2-4K key 执行完整 block sparse attention
      c) FFN: 标准 FFN 计算
    End For
    首 token 生成（TTFT）
    KV cache 写入 UVM：新 key/value 写入 CPU unified memory + GPU key bank（cache miss 时 fetch）
  ↓
  [3] Decoding Loop（逐 token 生成）
    For each new token:
      a) QKV Projection（仅新 token）
      b) KV Cache 管理:
         - 检查 GPU key bank 是否有命中（通过 page table 查 global→local 映射）
         - Cache miss: 从 CPU UVM 通过 PCIe 加载缺失 key/value 到 GPU bank
         - LRU 驱逐: 若 GPU bank 满，驱逐最久未使用的 cold token 回 CPU
      c) InfiniteHiP Context Pruning（mask refresh 检查）:
         - c^(i) mod n_refresh^(i) == 0? 
           Yes: 重新运行第 i stage pruning，更新 mask I^(l,i)
           No: 复用缓存的 mask（temporal locality）
         - 默认 refresh: stage1 每 16 步、stage2 每 8 步、stage3 每 4 步
         - Flash 配置: (96, 24, 8) → 解码速度大幅提升
      d) Block Sparse Attention: 使用 I^(l,N) mask 执行稀疏 attention
      e) FFN + Sampling: 生成下一个 token
    Until EOS or max_tokens
  ↓
  输出：生成的 token 序列 → SGLang detokenizer → 文本
  ```

  **关键性能数据（RTX 4090 24GB, 3K-Fast Offload）**：
  - 64K context: 64.5 tok/s
  - 128K context: 55.9 tok/s
  - 256K context: 46.6 tok/s
  - 512K context: 31.8 tok/s
  - 1024K context: 17.3 tok/s

  **关键性能数据（L40S 48GB, 3K-Flash Offload, 带 mask 缓存加速）**：
  - 64K context: 56.6 tok/s
  - 256K context: 49.4 tok/s
  - 512K context: 43.7 tok/s
  - 1024K context: 35.2 tok/s
  - 2048K context: 28.0 tok/s
  - 3072K context: 23.8 tok/s（3M tokens!）
  - vs SRT Estimated 3M: 7.25× speedup（23.8 vs 3.3 tok/s）

  **Mask 缓存效果（Table 4, 256K decoding latency per token）**：
  - No cache（所有 stage 重算）: 9,803 µs
  - Stage 1 cached: 2,579 µs（3.8× faster）
  - Stage 1&2 cached: 779 µs（12.6× faster）
  - All stages cached: 110 µs（89.1× faster）
  - Mask hit ratio: Stage 1: 71.67% → Stage 1&2: 98.75% → All: ~100%
