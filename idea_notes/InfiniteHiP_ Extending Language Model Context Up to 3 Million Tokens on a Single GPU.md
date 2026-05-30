## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- baseline方法是什么？
  Baseline 是三种现有方法的组合缺陷：(A) **Full Attention (FlashAttention2)** + (B) **HiP Attention** 的层次化剪枝 + (C) **SelfExtend/NTK** 的 RoPE 外推。

  (A) FA2 的缺陷：通过 tiling+recompute 优化 memory access 但不减少 FLOPs——1M context 解码时每次 attention 需 4,645 µs，KV cache 在 BF16 下 1M tokens 需约 64GB 远超单卡容量，不支持 OOL generalization（仅在训练长度内有效）。
  
  (B) HiP Attention 缺陷：(a) 层次化剪枝算法的迭代式 top-k 涉及大量 global thread synchronizations，阻碍 GPU 并行——SelectRep 的每次迭代需全局同步，无法利用 key sequence dimension parallelism（类似 FlashDecode 的 split-KV）；(b) 启发式剪枝精度不足——top-k 估计的 recall 较 InfiniteHiP 低 4.72%；(c) 无 per-stage mask 缓存——解码时每次都要运行完整的 costly initial pruning stage（O(T_kv)），导致解码延迟高；(d) KV cache offloading 的驱逐策略简单，未使用 LRU。
  
  (C) RoPE 外推缺陷：SelfExtend 使用统一的 group-size 缩放 RoPE，不区分不同层/head 的 attention pattern 差异——早期层倾向于 dynamic sliding window（关注相对位置），后期层依赖语义信息，统一处理导致 OOL 性能不佳。

  全栈执行例子（baseline: HiP Attention + SelfExtend, Llama 3.1 8B, 128K context, RTX 4090）：
  - **算法层**：输入 128K tokens → HiP 层次化剪枝（heap-based top-k selection across all heads, 需 global sync）→ 从 128K 缩减到 ~1K token → Block Sparse Attention on ~1K token → 输出 token。SelfExtend 对全部层使用相同 group_size 缩放 RoPE。
  - **系统框架层**：基于 PyTorch 自定义 attention 层，HiP 的 offloading 使用 UVM，无 SGLang 集成。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：HiP kernel：迭代式 SelectRep + 全局 top-k → 每个 stage 内 global thread synchronization → 解码 450 µs/token @ 1M context。FA2 kernel：全量 dense attention，无剪枝优化。
  - **硬件架构层**：NVIDIA RTX 4090 24GB。HiP+UVM offloading 在 256K+ 上下文时因 offload 开销大，解码延迟高。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  InfiniteHiP 通过三个核心设计解决 baseline 缺陷：

  **1. 高度并行的模块化层次化剪枝（Modular Hierarchical Pruning）→ 解决缺陷 (B-a, B-b)**
  将 HiP 的迭代式 top-k 替换为 per-chunk top-1（SelectRep）+ max chunk score + top-K chunk selection 的单 kernel 流程。关键创新：SelectRep 算法每次迭代仅访问 2 个 token（左右分支首 token）→ 无全局同步即可实现 → 可利用 key sequence dimension parallelism（类似 FlashDecode split-KV）→ 在 A100/4090 等现代 GPU 上实际运行速度更快。同时由于使用 per-query-block 的动态代表 token 选择（而非 InfLLM 的预选固定代表 token），top-k 估计 recall 比 HiP 高 4.72%，比 InfLLM 高 1.57%。

  **2. 分层动态 RoPE 调整（Layer-wise Dynamic RoPE）→ 解决缺陷 (C)**
  观察到 LLM 早期层（layer ≤ 5）呈现 dynamic sliding window-like attention pattern（关注相对位置），后期层依赖语义信息。因此：前 3 层使用 Chunk-indexed RoPE（每 chunk 赋相同 position ID，引导滑窗式 mask）、后续层使用 Relative-style RoPE（分左右分支赋不同 position offset，依赖内容信息）。Block Sparse Attention 阶段使用 StreamingLLM-style RoPE（保持 causality 和相对顺序）。这种异构 RoPE 策略使 En.MC OOL 评测从纯 Relative 的 68.55% 提升到混合 Chunk-indexed+Relative 的 74.23%（+5.68%）。

  **3. Per-stage Mask 缓存 + LRU KV Offloading → 解决缺陷 (B-c, B-d, A)**
  (a) 每个剪枝 stage 维护独立的稀疏注意力 mask 缓存，利用 temporal locality（query 的 mask 在连续 decoding step 中变化缓慢），以 configurable refresh interval 更新。效果：256K context 解码从 no cache 的 9,803 µs 降至 all cache 的 110 µs（89× speedup）。
  (b) 将 KV cache offloading 驱逐策略从 HiP 的简单策略改为 LRU，更精确地识别 cold token。同时实现为 graph-capturable CUDA 操作避免 CPU launch overhead。
  (c) 所有组件集成到 SGLang serving 框架中，使用 Triton 实现跨硬件可移植。

  全栈执行例子（InfiniteHiP, Llama 3.1 8B, 3M context, L40S 48GB, 3K-Flash preset）：
  - **算法层**：输入 3M tokens → 保留 n_sink=256 + n_stream=1024 → Stage 0: 全量 key 分 chunk(l_c=256) → SelectRep(top-1 per chunk，每次 2 token 点积) → max chunk score → top 32K key → Stage 1: 32K key 分 chunk(l_c=32) → SelectRep → top 8K key → Stage 2: 8K key 分 chunk(l_c=8) → SelectRep → top 4K key（前 3 层）/ 2K key（后 29 层）→ BSA on ~3K selected keys → Dynamic RoPE（前 3 层 Chunk-indexed / 后 29 层 Relative-style / BSA 阶段 StreamingLLM-style）→ 输出 token。复杂度从 O(T²) 降至 O(T*q)：3M 解码时仅对约 3K token 执行完整 attention。
  - **系统框架层**：基于 SGLang（https://github.com/DeepAuto-AI/sglang/），在 SGLang 的 attention 计算路径中替换为 InfiniteHiP pipeline。支持 mask cache 按 stage 独立管理、refresh interval configurable（flash preset: 96/24/8）、单 batch 长上下文 serving。End-to-end 解码吞吐：L40S 上 3M context 达 23.8 tok/s（Flash offload），比 SRT 估计值快 7.25×。
  - **编译框架层**：论文未明确说明。Triton kernel 通过 Triton compiler 编译为 GPU 代码。
  - **kernel调度层**：(a) Pruning Stage Triton Kernel：单一 kernel 实现完整 stage，parameterized by (l_c, b_q, k)，key sequence dim parallel（类似 FlashDecode split-KV），无全局同步。(b) BSA Triton Kernel：FlashAttention-style（prefill）+ FlashDecoding-style（decoding）+ PagedAttention（block KV memory）。(c) UVM Offloading：CUDA UVM dynamic page migration + LRU eviction on GPU key bank。Latency breakdown（1M context, 3K, 无 mask cache）：Stage 0 28.2% + Stage 1 4.0% + Stage 2 5.3% + BSA 2.2% + Extra(offload overhead) 60.3%。
  - **硬件架构层**：NVIDIA RTX 4090 24GB（消费级）和 L40S 48GB（云端性价比），单 GPU。PCIe 4.0 x8 连接 CPU RAM（访问延迟 31.5× VRAM）。3M context KV cache 约需 192GB → 远超显存 → 通过 UVM offloading 使用 CPU RAM。

  对比 baseline 的关键差异：
  - HiP 的迭代 top-k → InfiniteHiP 的 per-chunk top-1 + max score + top-K chunk（无全局同步，key dim 并行）
  - HiP 无 mask 缓存 → per-stage mask caching（89× 解码加速）
  - HiP 简单 offloading 驱逐 → LRU 驱逐 + graph-capturable CUDA operation
  - SelfExtend 统一 RoPE → 分层异构 RoPE（Chunk-indexed + Relative + StreamingLLM，En.MC +5.68%）
  - InfLLM 预选固定代表 token → 动态 per-query-block 代表 token 选择（recall +1.57%）
  - InfLLM 不在 attention kernel 内访问 CPU memory（牺牲精度）→ InfiniteHiP 在 kernel 内访问 CPU memory（保持精度，但 offload 开销 60.3%）
  - FA2 不支持 OOL generalization + KV 随 T 线性增长无法 fit GPU → InfiniteHiP 训练无关 OOL + UVM offloading 支持 3M tokens on 48GB GPU
