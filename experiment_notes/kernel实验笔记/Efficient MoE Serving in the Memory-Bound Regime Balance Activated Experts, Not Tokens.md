## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 METRO 的 CUDA kernel，将 Algorithm 1（greedy expert-to-GPU assignment）实现为运行在单个 Streaming Multiprocessor (SM) 上的 GPU kernel。设计要点：(1) **单 SM 绑定**：由于算法并行度受限于 expert 数量（Qwen: 128 experts, DeepSeek-V3: 256 experts），加上 locking 进一步降低并发度至 <64，单个 A100 SM 即可提供足够的并行处理能力；(2) **SM-local shared memory**：将 per-GPU activated expert 计数器 L[1..G] 和对应的锁 l[1..G] 放置在 SM 共享内存中，实现快速访问；(3) **test-and-set lock**：使用简单的 test-and-set 自旋锁进行 GPU 线程间同步；(4) **全序锁获取**：通过按 GPU ID 全局顺序获取锁来避免死锁；(5) **CUDA Graph 预编译**：将 kernel 集成进 vLLM 的 decode phase CUDA Graphs，为 power-of-two batch sizes（up to 32 tokens/GPU）预编译，非 power-of-two 通过 padding 复用。

  此外 METRO 在通信层面改变了 kernel 间的调度：原 all-to-all dispatch 的 kernel 序列被替换为 all-gather + top-k + METRO routing + FFN + all-to-all combine 的序列。

  实验比较：(a) METRO routing kernel 延迟 vs FFN 层延迟 —— kernel 最多 26us vs FFN 最多减少 81us；(b) all-gather communication time vs all-to-all communication time — 两者无统计显著差异（NVLink latency 主导，bandwidth 开销 3us << NCCL launch ~100us）；(c) METRO top-k 额外开销（redundant computation on full token set）vs 原 top-k — 最多增加 3us (<1% 层时间)；(d) CUDA-based optimal algorithm (GPU push-relabel max-flow) vs METRO greedy — optimal 开销 290us+ (86.4%-103.8% FFN time) vs METRO <26us；(e) CPU-based optimal algorithm (Dinic max-flow) vs METRO — 含 CPU-GPU 数据传输 116-128us + 26.5-29.2us transfer (31.4%-41.3% FFN time)。

- 后端平台是什么，配置是什么。
  NVIDIA A100 40GB GPU（108 SM, 每个 SM 64 FP32 CUDA cores + 4 Tensor Cores），600 GB/s NVLink（8 GPU 在同一 NVLink domain）。NVIDIA B200 192GB GPU（模拟器建模），900 GB/s NVLink。CUDA kernel 运行在单 SM 上，计数器 L 和锁 l 驻留在 SM shared memory。

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 vLLM 框架中集成的 METRO CUDA kernel 和 NCCL 通信原语。修改内容：
  (a) **METRO routing CUDA kernel**：实现 Algorithm 1，单 SM 执行，test-and-set lock 同步，shared memory 计数器；
  (b) **CUDA Graph integration**：在 vLLM compilation framework 中为 power-of-two batch sizes 预编译包含 METRO kernel 的 CUDA Graphs；
  (c) **通信原语替换**：将 MoE layer 的 dispatch 阶段从 NCCL all-to-all 替换为 NCCL all-gather；
  (d) **Top-k 范围扩展**：top-k 从 per-GPU local tokens 扩展到全局 all-gathered tokens；
  (e) **Latency breakdown 测量**：通过 profiling 各组件（top-k, routing kernel, all-gather, FFN）的延迟来验证 METRO 的 overhead 可被 FFN 减少所抵消。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供开源代码。基于 vLLM 开源框架实现，以下是评估原理和 kernel 级全流程：

  **METRO CUDA Kernel 执行原理**：
  ```
  Kernel Configuration:
    - Grid: 1 block
    - Block: min(N, 64) threads (N = 128 for Qwen, 256 for DeepSeek)
    - SM: 1 (bound to single SM)
    - Shared Memory: L[G] (int32 array, G=8-16), l[G] (int32 locks)

  Kernel Pseudocode (per-thread, expert i):
    if T[i] == 0: return                      // skip experts with no tokens
    
    // Step 1: Determine candidate GPUs from placement matrix
    candidates = []
    for g in 0..G-1:
        if A[i][g] == 1: candidates.append(g)
    
    // Step 2: Acquire locks in total order (GPU ID ascending)
    for g in sorted(candidates):
        while atomicCAS(&l[g], 0, 1) != 0:   // test-and-set spinlock
            // spin
    
    // Step 3: Find GPU with minimum activated experts
    best_g = candidates[0]
    for g in candidates:
        if L[g] < L[best_g]: best_g = g
    
    // Step 4: Assign expert to best_g
    y[i][best_g] = 1
    atomicAdd(&L[best_g], 1)
    
    // Step 5: Release locks in reverse order
    for g in reverse(sorted(candidates)):
        atomicExch(&l[g], 0)
  ```

  **Kernel 输入到性能输出全流程**：
  ```
  Input:
    - A[128][8]: expert-GPU placement matrix (host->device, read-only, global mem)
    - T[128]: token count per expert (device, populated by top-k on all-gathered tokens)
    - G = 8: number of GPUs
  
  Execution (on single SM):
    1. Load A and T from global memory
    2. Initialize L[0..7] = {0} in shared memory
    3. Initialize l[0..7] = {0} in shared memory (0 = unlocked)
    4. Launch min(N,64) threads:
       - Each thread processes one expert i
       - Concurrently execute lock acquire-assign-release cycle
       - Total global memory reads: |A| entries (128 * avg_replicas)
       - Shared memory access: O(G) per lock acquire/release
    5. Write y[128][8] to global memory (output)
    6. Kernel overhead measured: 17-26us (varies with replication ratio)
  
  Communication sequence:
    NCCL All-gather: 2MB/GPU -> ~3us bandwidth + ~100us launch latency
    Top-K on all tokens: 17us->20us (vs original 17us->19us, +3us max)
    METRO Kernel: 17us->26us
    FFN (activated experts only): 230us->311us (varies with replication, ~81us reduction vs EPLB)
    NCCL All-to-all Combine: same as EPLB baseline
  ```

  关键结论：METRO routing kernel 的计算 overhead (17-26us) + top-k overhead (<3us) + 通信 overhead (~3us bandwidth) 总计最多约 30us，远低于其带来的 FFN 时间减少 (up to 81us)，净收益 ~50us/layer。在 30-layer Qwen3-30B 模型上，每 decode step 累积收益显著，最终端到端 decode latency 降低 11%-22%。
