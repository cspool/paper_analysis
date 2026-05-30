## HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - HEXA-MoE 包含三个 kernel 调度/运行时计算层面的实现：
    1. **Re-Index Vector based Expert-Specific CUDA Kernels**：实现 ESMM、ESS、ESTMM 三个 expert-specific 算子的 GPU kernel。通过构建 re-index vector（按 routing choice 重排 token indices，同 expert 的 token 聚集为 sub-vector，padding -1 对齐 tiling size）作为 I/O 指导。ESMM kernel：thread-block 加载 sub-vector → 按 vector 值加载 tokens 和对应 expert 权重（同一 expert 的 tokens 只需加载一次权重）→ 沿 input feature 维度累加 dot product → 按 sub-vector 写回 HBM。ESS kernel：每个 thread-block 分配某 expert 的某些 channel → 加载该 expert 对应 sub-vector 中 tokens → 累加后写回 HBM。ESTMM kernel：输入以 re-indexed 格式提供，两输入共享 re-index vector → thread-block 加载同 expert 两输入特定 channel → 累积外积 → 写回 HBM。
    2. **Expert-Specific Fused Kernel (ESFK)**：将 backward pass 中的 ESS、ESTMM、ESMM 融合为单一 kernel。通过统一各算子的 thread-block shape 为 (WARP, TIMES)，并将 thread-grid 扩展为 3 维使各算子 grid 可对齐并聚合。单 MoE 层 backward 仅需 2 个 fused kernels + 1 个 element-wise dot product。
    3. **Pipeline-Shared Cache + Communication-Computation Overlap**：在 data-centric 配置下，每设备分配额外 HBM 区域作为 pipeline-shared cache，动态缓存 all gather 来的 MoE parameter shards。All gather 通信与 attention/router 计算重叠，backward pass 无需保存完整 MoE 参数（通过 cache 动态获取），解决 Janus 等方法的 backward 内存膨胀问题。
  - 实验比较：(1) Memory Analysis: HEXA-MoE vs Tutel vs MegaBlocks 的 GPU 内存占用，2 homogeneous GPUs, 8 global experts, top-1~top-8 routing；(2) Latency Analysis: 平均每步训练延迟，4 homogeneous GPUs, 4 experts, 不同 batch size；(3) Ablation: 各组件的 memory footprint breakdown 和 latency breakdown。

- 后端平台是什么，配置是什么。
  - 同构机器 M_homo：4× NVIDIA GeForce RTX 4090 (24 GB)，CPU 2× Intel Xeon Platinum 8352V 2.10GHz。
  - 异构机器 M_hete：1× NVIDIA TITAN RTX (24 GB) + 1× NVIDIA GeForce RTX 2080 Ti (11 GB)，CPU 2× Intel Xeon Gold 6130 2.10GHz。
  - CUDA kernel 参数：BLK (block size)，WARP (warp size=32)，TIMES (thread block 有 WARP×TIMES 个线程)。使用 nvcuda::wmma 接口调用 Tensor Core 做 16×16×16 矩阵乘法。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 PyTorch + CUDA C++ 实现 expert-specific kernels（ESMM、ESS、ESTMM、ESFK）。
  - 修改 NCCL all gather / all reduce 通信后端，适配 tensor parallelism 替代 expert parallelism。
  - 使用 PyTorch automatic mixed precision 训练。
  - 评估指标：NVIDIA SMI 监控 GPU 内存占用 (GB)，PyTorch CUDA event 计时测量每训练步延迟 (s)，2k steps 取平均。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源链接：https://github.com/UNITES-Lab/HEXA-MoE（支持 Triton 和 CUDA 两种 kernel 实现）
  - ESMM Kernel 执行原理全过程（基于 Algorithm 2）：

```
┌── Kernel Input ────────────────────────────────────────┐
│ R: routing choice [N], v: re-index vector [N']          │
│ x: input tokens [N, D1], w: weights [E, D1, D2]         │
│ b: bias [E, D2]                                         │
│ BLK: tiling size, WARP=32, TIMES: threads/BLK           │
└────────────────────────────────────────────────────────┘

┌── ESMM CUDA Kernel ──────────────────────────────────┐
│ Parallel for i in range(0, N', BLK):                   │
│   Parallel for j in range(0, D2, BLK):                 │
│     exp = R[v[i]]                    // expert index   │
│     c = b[exp, j:j+BLK].repeat(BLK, 1)  // init bias  │
│     for k in range(0, D1, BLK):                        │
│       // Load BLK tokens for this expert               │
│       Parallel for t = 0 to BLK-1:                     │
│         if v[i+t] != -1:                               │
│           xsub[t] = x[v[i+t], k:k+BLK]                 │
│         else: xsub[t] = 0       // skip padding        │
│       wsub = w[exp, k:k+BLK, j:j+BLK]                  │
│       c += xsub @ wsub           // Tensor Core MMA    │
│     // Write back                                      │
│     Parallel for t = 0 to BLK-1:                       │
│       if v[i+t] != -1:                                 │
│         y[v[i+t], j:j+BLK] = c[t]                      │
└────────────────────────────────────────────────────────┘
```

  - Re-Index Vector 构建原理（Algorithm 1）：
    1. 统计每个 expert 的 token 数量 ctr[e]（atomicAdd）
    2. 将 ctr[e] 向上取整到 BLK 的倍数
    3. 计算累积偏移 idx[e]（prefix sum）
    4. 按 routing choice 将 token index 写入 v[idx[R[i]]++] = i
    5. v 中未填满的 BLK 位置填充 -1

  - ESFK 融合原理（Table 6）：
    所有算子 thread-block shape 统一为 (WARP, TIMES)，thread-grid 扩展为 3 维。ESS grid: (E, D2/(TIMES·BLK), 1) → 扩展第三维为 1；ESMM grid: (N'/BLK, D1/(TIMES·BLK), 1) → 扩展第三维为 1；ESTMM grid: (E, D1/(TIMES·BLK), D2/(TIMES·BLK))。聚合后 ESFK grid 第三维 = N'/BLK + D2/BLK + D2/(TIMES·BLK)。

  - Pipeline-Shared Cache 原理：
    ```
    # Data-centric 配置下的内存管理
    # 每设备在 HBM 额外分配 cache 区域
    # Forward: all gather MoE shards → 写入 cache → ESMM 计算
    # Backward: 从 cache 读取所需的 gathered shards（无需永久保存）
    # All gather 与 attention/router 在分离 CUDA stream 上 overlap
    ```
