## Counting Table Signaling

术语是什么？

Counting table signaling 是 FlashOverlap 中用于追踪 tile 完成状态的轻量级 GPU global memory 同步机制。Counting table 是一个大小为 P（wave group 数）的 int 数组，初始化为 0。每当一个 tile 在 GEMM epilogue 中完成时，通过 `atomicAdd(&counting_table[group_id], 1)` 递增计数。当 group G_j 计数达到 |G_j|（该 group 包含的 tile 总数）时，signaling kernel 检测到条件并触发 NCCL 通信。

从kernel调度角度拆解术语：

Counting table 并发交互 timeline：

```
时间 →
Stream A (GEMM):  |-- GEMM main W1 --|-- epilogue atomicAdd G1 tiles --|-- GEMM W2 --|
Stream B (Signal): | spin-wait c[1]<|G1| ... | detect c[1]==|G1| → ncclAllReduce(G1)   |
Stream B (Comm):   |                         |                        |-- NCCL AR G1 --|
                   |                         |<-- Overlap: W2 computed while G1 communicated -->|
```

**Annotations**: AtomicAdd 在 epilogue 中执行，开销 ~0.07% GEMM latency (A800)。Counting table 大小 = P × sizeof(int)，典型 P ≤ 10，总大小可忽略。Signaling kernel 通过 __ldg (read-only cache) 读取 counting table 减少 memory traffic。不需要 heavy-weight CUDA synchronization primitives。

术语一般如何实现？如何使用？

Counting table 通过 `cudaMalloc` 分配在 GPU global memory，通过 `cudaMemset` 初始化。Signaling kernel 在独立 CUDA stream 中 launch，与 GEMM stream 并发。对于多次执行的场景（如 training loop），counting table 可复用。开源实现见 github.com/infinigence/FlashOverlap。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
