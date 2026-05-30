## Toward Efficient Inference for Mixture of Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文在 MoE gating 函数层面改变了 token dispatch 的 kernel 计算模式：将 Static Gating 的 **batch matrix multiplication (bmm, sparse mask × tokens)** 替换为 Dynamic Gating 的 **argsort + bin-count + indexing** 方案。这是 kernel 级别的优化——用 O(SD) indexing kernel 替代 O(S²EDC) batch matmul kernel，消除 dispatch mask 的内存分配和 placeholder computation。此外，Dynamic Gating 使用两轮 all-to-all（先通知 size 再传 tokens），将通信模式从固定大小 all-to-all 改为可变大小 all-to-all。
  实验比较：与 baseline Fairseq（static gating + bmm dispatch）、Tutel（hash table lookup + custom cumulative sum kernel）、FasterMoE（kernel launch overlap）、Megablock（block-sparse BCSR kernel）对比。分析 latency breakdown（gating、all-to-all、expert 执行的贡献比例）和 memory trace。

- 后端平台是什么，配置是什么。
  GPU：NVIDIA Tesla V100 (32GB, NVLink, Volta SM) 和 NVIDIA RTX A5000 (24GB, Ampere SM)。CPU：Intel Xeon E5-2698 v4 (Apple) & Intel Xeon Gold 5317 (Pear)。CPU-GPU：PCIe 3.0 16GB/s 和 PCIe 4.0 32GB/s。
  Megablock 的 custom kernel 需要 A5000 的 Ampere 架构特性，但不支持 bias term。

- 评估性能的软件/脚本是什么。修改了什么。
  Fairseq MoE 实现作为 baseline。评估使用 Python `time` 模块记录 latency + PyTorch Profiler 收集详细 CUDA kernel trace + memory trace。
  修改内容：
  1. **Gating kernel 替换**：将 `bmm(mask, tokens)` 替换为三步操作：
     - `torch.argsort(assignments)`: 按 expert ID 排序，GPU kernel，O(S log S)。
     - `torch.bincount(assignments)`: 统计每个 expert 的 token 数量，O(S)。
     - `tokens[sorted_idx]`: 高级索引（advanced indexing）kernel，O(SD)，直接内存重排。
  2. **Reordering kernel 替换**：dispatch 后的 token 重排序也从 bmm 替换为 indexing。
  3. **两轮 all-to-all 通信模式**：第一轮传 sizes（标量，极低延迟，avg 20µs）；第二轮传 tokens（可变大小，消除 zero-padding）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。

  评估原理与 kernel 数据流：

  ```
  [Kernel-level Performance Evaluation Pipeline]

  PyTorch Profiler → CUDA trace (kernel launch events, duration)
  Python time.time() → end-to-end batch latency
  Memory trace → torch.cuda.memory_allocated() / memory_reserved()

  === Static Gating Kernel Flow (Baseline) ===

  Input: tokens X ∈ R^{S×D}, gating assignments A ∈ Z^{S×k}
  
  [Kernel 1] Gate Linear:       W_gate @ X          → gate_logits (S, E)
  [Kernel 2] Top-K:              topk(gate_logits)    → assignments (S, k)
  [Kernel 3] Mask Construction:  create_dispatch_mask → mask (E, S, S×C)
            → 内存分配: E × S × S×C × 4 bytes (float32)
            → LM S=8, E=512, C=0.05: 512 × 8 × 25.6 × 4 ≈ 419KB
            → 但论文 Fig.10 显示 gating/reordering 内存尖峰可达 GB 级别
            → （因 batch matmul 内部需要大工作区）
  [Kernel 4] Batch MatMul:       bmm(mask, X)        → dispatched (E, S×C, D)
            → O(S² × E × D × C) ≈ 512 × 64 × 25.6 × 1024 ≈ 860M FLOPs
            → 其中大部分计算为 ×0 (mask 极稀疏) → 浪费
  
  [Kernel 5-7] All-to-All, Expert FFN forward, All-to-All collect
  [Kernel 8] Batch MatMul:       bmm(mask^T, out)    → 还原 token 顺序

  === Dynamic Gating Kernel Flow (Proposed) ===

  Input: tokens X ∈ R^{S×D}, gating assignments A ∈ Z^{S×k}

  [Kernel 1] Gate Linear:       W_gate @ X           → gate_logits (S, E)
  [Kernel 2] Top-K:             topk(gate_logits)     → assignments (S, k)
  
  [Kernel 3] Argsort:           torch.argsort(A[:,1]) → sorted_idx (S,)
            → GPU radix sort / merge sort kernel
            → O(S log S), S=8 时 trivial; S=512 时 ~tens of µs
  
  [Kernel 4] BinCount:          torch.bincount(A[:,1]) → sizes (E,)
            → GPU reduction kernel, O(S)
  
  [Kernel 5] Advanced Index:    X[sorted_idx]          → sorted_X (S, D)
            → GPU gather kernel, directly reorder via indices
            → O(SD) memory bandwidth bound, NOT compute bound
            → 无临时 mask tensor 分配!
  
  [Comm Round 1] All-to-All:    sizes (E integers)     → 20µs avg latency
            → 各 GPU 现在知道 incoming tensor shapes → pre-allocate
  
  [Kernel 6] Split:             torch.split(sorted_X, sizes)
            → 按 sizes 切分 sorted_X → variable-length groups
  
  [Comm Round 2] All-to-All:    variable-size tokens (zero padding = 0)
            → 仅传输实际 tokens，无 placeholder 浪费
  
  [Kernel 7-8] Expert FFN forward per GPU
  [Comm Round 3] All-to-All:    expert outputs back
  
  [Kernel 9] Advanced Index:    inverse_permutation  → 还原 token 顺序

  === Kernel 效率对比 ===

  Static Gating batch matmul (waste analysis for LM, E=512, C=0.05):
    - 每个 expert 配置处理 ECS = 512 × 0.05 × S = 25.6S tokens
    - 实际仅需 2S tokens (top-2 gating)
    - Waste factor: 25.6S / 2S = 12.8×
    - 即 92.2% 的 batch matmul FLOPs 浪费在零值上

  Dynamic Gating indexing:
    - 传输的 tokens 数 = 实际需要的 tokens 数 (zero waste)
    - Indexing 是纯内存操作 (O(SD) BW), 无浪费计算
    - 但增加 1 次 light all-to-all (~20µs) 开销

  为什么 Dynamic Gating 在大 batch size 下优于 Megablock:
    - Dynamic Gating: 多个 dense matmul (每 expert) → GPU efficient
    - Megablock: 单个 BCSR sparse matmul → 需要 metadata (col indices, row offsets)
      → indexing 开销随 batch size 增大 (matrix 增大)
    - Dynamic Gating: kernel launch 数 = expert 数量（固定）
      → overhead 不随 batch size 变化
    - 实验: batch=80 时 Dynamic Gating 比 Megablock 快 1.46×

  Expert Buffering 相关的 kernel 操作:
  [Kernel] cudaMemcpyAsync: CPU→GPU expert 参数 (PCIe stream)
           → 与 all-to-all NCCL stream 并发 → 零额外延迟
  ```
