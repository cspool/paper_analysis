## Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了一个 fused attention kernel，在同一 kernel launch 中同时处理 retrieval heads（FA mode）和 sparse heads（SA mode），替代传统的 serial dispatch 方案。核心设计：(1) **Unified Kernel Launch**：将 routing decisions r 直接作为轻量元数据 m 传入 kernel，kernel 内部通过 thread-block level branching 动态判断每个 head 的类型并执行对应的 attention logic（FA 或 SA）；(2) **Eliminate Tensor Splitting**：无需像 Serial Dispatch 那样在 kernel 外先 split Q/K/V 为 full 和 sparse 两组、再分别 launch 两个 kernel，避免了非连续 tensor fragment 的内存分配和拷贝开销；(3) **Grid Integrity**：保持 grid 维度不变（Batch × Heads × Sequence Blocks），允许 GPU hardware scheduler 最优地分布 sequence blocks 到各 streaming multiprocessor。基于 Block Sparse Attention (BSA) Kernel（Guo et al., 2024, https://github.com/mit-han-lab/Block-Sparse-Attention）实现。实验比较 fused kernel 与 Torch-based sequential、layer-wise hybrid attention 实现在不同 sequence length 下的 prefill-time 加速比（Figure 4）。

- 后端平台是什么，配置是什么。
  单 GPU 部署（无跨设备通信），具体 GPU 型号论文未明确说明（fused kernel 加速测试环境）。基于 Block Sparse Attention (BSA) Kernel，block_size=64, chunk_size=16384, sink_size=128。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + Block Sparse Attention (BSA) Kernel。核心修改：
  1. **Fused Hybrid Attention Kernel**：替代 PyTorch baseline 的三步 serial dispatch（split Q → FlashAttn + SlidingWin → merge O），实现 unified kernel:
  ```
  # PyTorch Baseline (Serial Dispatch)
  r = Router(x)
  I_full = {h | r[h]=0}, I_sp = {h | r[h]=1}
  Q_full = Q[:, I_full]
  O_full = FlashAttn(Q_full, K_full, V_full)      # kernel 1
  O_sp = SlidingWin(Q[:, I_sp], K_sp, V_sp)       # kernel 2
  O[:, I_full] = O_full; O[:, I_sp] = O_sp        # merge

  # Fused Kernel (Parallel via BSA)
  r = Router(x)
  m = Map(r)  # lightweight metadata
  O = BSA_Kernel(Q, K, V, m)  # single kernel
  # Inside kernel:
  # par for h do:
  #   if m[h]==SP: O[h] = Sparse(Q[h], K, V)
  #   else:        O[h] = Full(Q[h], K, V)
  ```
  2. **Thread-block Level Branching**：每个 thread block 从 metadata m 中动态获取所分配 head 的类型，根据类型执行对应的 attention 计算逻辑（FA 或 SA），避免 kernel 外部的 tensor rearrangement。
  3. **Sequence-level Parallelism**：当输入序列足够长时，parallelism 沿 sequence dimension 主导执行。GPU 在完成一个 head 的大部分 sequence blocks 后才切换到下一个 head。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/LCM-Lab/Elastic-Attention。BSA Kernel 来自 https://github.com/mit-han-lab/Block-Sparse-Attention。

  **评估原理与 Kernel 执行全流程（以单层 prefill 为例）**：

  ```
  输入：Q, K, V ∈ R^{s×H×d'}（H KV heads via GQA）, router decisions r ∈ {0,1}^H
  输出：O ∈ R^{s×H×d'}

  Step 1: Attention Router 计算 routing decisions
    x_K = K  # Key hidden states as input
    x_K' = BoundaryPooling(x_K)  # [H, d'], 聚合前100+后100 tokens
    z = MLP_router(MLP_task(x_K'))  # [H, 2]
    r_hard[h] = argmax(softmax(z[h]))  # 0=FA, 1=SA

  Step 2: Map routing decisions to metadata
    m = {h: "full" if r_hard[h]=0 else "sparse" for h in 1..H}

  Step 3: Unified BSA Kernel Launch
    grid = (Batch, Heads, ceil(s / T_s))  # T_s = tile size along sequence
    # Single kernel launch — no pre-splitting of tensors

    # Inside kernel, each thread block:
    block_idx = (b, h, seq_tile)
    if m[h] == "full":
        # Standard FlashAttention-like tiling for FA
        Q_tile = Q[b, h, seq_tile]       # load to SRAM
        K_tile = K[b, h, :]              # load to SRAM (full K)
        V_tile = V[b, h, :]
        S = Q_tile @ K_tile^T / sqrt(d')
        P = online_softmax(S)
        O_tile += P @ V_tile
    else:  # m[h] == "sparse"
        # Sparse attention: only attend to sink + recent + selected tokens
        K_sparse = K[b, h, sp_indices]   # sp_indices = {sink, recent, selected}
        V_sparse = V[b, h, sp_indices]
        S = Q_tile @ K_sparse^T / sqrt(d')
        P = online_softmax(S)
        O_tile += P @ V_sparse

  Step 4: Output concatenation
    O = concat all head outputs along head_dim
    # No post-kernel merge needed — output already in correct order
  ```

  **性能优势（Figure 4）**：
  - Fused kernel 相比 Torch-based sequential 实现在 prefill 阶段实现加速
  - 加速收益随序列长度增加而增大（较长的序列使 sequence-level parallelism 更充分地利用 GPU SMs）
  - 两种主要 overhead 被消除：(1) Memory overhead — 不再需要 allocate 和 copy 非连续 tensor fragment（split Q_full/Q_sp）；(2) Kernel Launch & Scheduling overhead — 不再需要多次 kernel launch，避免 work fragmentation 和 GPU SM 调度中断

  **Router Latency（Figure 10, 消融）**：
  - Attention Router 产生 negligible overhead：平均 0.196 ms/router call
  - 延迟不随序列长度增长（512 → 1M tokens 保持恒定），因为 router 的 pooling 仅处理 boundary tokens（首部+尾部各100）
