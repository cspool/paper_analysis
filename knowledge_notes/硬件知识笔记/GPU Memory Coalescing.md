## GPU Memory Coalescing

术语是什么？

GPU memory coalescing (内存合并) 是 GPU 内存系统的关键优化机制：当 warp 内 32 个线程访问 global memory 时，GPU 硬件将访问同一 128B cache-line 的线程请求合并为单次 memory transaction，从而高效利用 memory bandwidth。若 warp 内线程访问连续、对齐的地址（如 thread i 访问 base + i*4），所有 32 个线程的请求被合并为 1-2 个 128B transactions。若线程访问分散、未对齐的地址（如 scatter/gather pattern），每个线程的请求可能产生独立的 transaction，导致大量 bandwidth 浪费和 latency 暴露。

从硬件架构角度拆解术语：

Coalescing 在 GPU memory hierarchy 中的运转流程：

```
Warp 内 32 threads 发出 global memory load/store 请求
        ↓
LSU (Load-Store Unit) 检测地址分布：
  Case A: Coalesced → 1-2 transactions
    thread 0: addr 0x1000, thread 1: addr 0x1004, ..., thread 31: addr 0x107C
    → 所有地址在 [0x1000, 0x1080) 128B range 内
    → 合并为 1 个 128B transaction → 高效利用带宽
  Case B: Uncoalesced → up to 32 transactions
    thread addresses 分散在多个 128B cache line
    → 每个 cache line 产生独立 transaction
    → 可能 32× 更多 transactions，bandwidth 利用率极低
        ↓
L1 Cache (per SM, 128B cache line):
  - Coalesced: cache line 命中率高，减少 global memory 往返
  - Uncoalesced: 大量 cache line miss → 触发 global memory access
        ↓
L2 Cache (device-wide, 共享):
  - Coalesced: 连续地址的 prefetch 友好
  - Uncoalesced: 随机访问模式 → L2 miss → DRAM access
        ↓
DRAM (HBM2e on A100: 1555 GB/s peak):
  - Coalesced: 接近峰值带宽（如连续 copy）
  - Uncoalesced: 有效带宽远低于峰值（如 scatter atomic ~270 GB/s vs peak 1555 GB/s）
```

Coalescing factor γ 的量化：
```
γ = A / (M × 128)
```
其中 A = 实际 useful bytes, M = 所需的 128B transaction 数量。γ=1 表示完美 coalescing（每个 transaction 满载 useful data），γ<1 表示 bandwidth 浪费。若 hash aggregation 减少 total write-back data (A 减小) 同时改善 coalescing (M 显著减小)，γ 可超 1——VDHA 在 it-2004 上 γ 从 0.744 提升至 2.607 (density=100%)。

SpMSpV write-back 的 coalescing 问题：
- Atomic write-back: warp 内线程的 row_idx 分布随机 → 大量 uncoalesced transactions → γ ~0.5-0.7
- Hash-based flush: entries 按 bucket order 输出 → hash 保留 row index 低位 → 相近 hash 的 entries 有相近 row index → flush 时 consecutive threads 访问 consecutive addresses → γ >> 1

术语一般如何实现？如何使用？

程序员视角的 coalescing 优化：
- **数据布局**: 使用 SoA (Structure of Arrays) 而非 AoS 使连续线程访问连续地址
- **数据结构**: CSR/CSC 等压缩格式按 row/col 连续排列非零元，warp 遍历时自然 coalesced
- **哈希函数选择**: modulo hash 保留低位 → flush 时 entries 以 row order 大致排列 → 改善 coalescing
- **Vectorized access**: 使用 float4/uint4 等 128-bit 类型一次 transaction 加载更多数据
- **Shared memory staging**: 先在 shared memory 中重组数据 → coalesced global write
- Nsight Compute 的 memory workload analysis 可量化 uncoalesced transactions 数量

在 Swift SpMM 中的 coalescing 问题（双输入同时 coalesce）：
- **Sparse A 侧**：CSC 格式中 warp 内线程按列分配，同列非零元 row_idx 连续 → A 的 value/rowIdx 读取天然 coalesced（128B aligned with 32 threads × 4B）
- **Dense B 侧**：传统 CSR SpMM 中 warp 线程的 colIdx 分布随机 → B 访问地址跳跃 → uncoalesced。Swift 的改进：列排序+行重排后 warp 内相邻线程处理连续列 → B[colIdx+j] 地址连续 → coalesced
- **双输入 coalescing 效果**：数据加载开销从 >32% 显著降低（如 regular kernel 下 coalesced B access 带来 1.32×-1.38× speedup）

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

