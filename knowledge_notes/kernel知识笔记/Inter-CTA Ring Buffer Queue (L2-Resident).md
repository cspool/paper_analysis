## Inter-CTA Ring Buffer Queue (L2-Resident)

术语是什么？
Inter-CTA Ring Buffer Queue 是 Kitsune 的纯软件实现的 CTA 间通信队列。使用 L2 cache 作为存储介质 + global atomics 作为同步机制，在 producer CTA 和 consumer CTA 之间传递 tile 级（64-256KB）中间数据。Queue 为双 buffer 设计（两个 entry），使用 sequence number 实现无锁 producer-consumer 同步。通过 CUDA API 将 queue memory pin 在 L2 cache 中，避免数据溢写到 HBM。

从kernel调度角度拆解术语：
Queue 的同步协议：

```
struct QueueEntry {
    float data[ENTRY_SIZE];    // tile数据 payload (64-256KB)
    int seq         __attribute__((aligned(128)));  // producer递增
    int consumed    __attribute__((aligned(128)));  // consumer递增
    // 全部cache-line对齐避免false sharing
};

// acquire/release API (仅CTA内threadid==0执行)

int wr_acquire(Queue* q, int tile_id):
    while true:
        seq = atomicAdd(q->seq, 0)       // 原子读取seq number
        if seq == tile_id:               // entry空闲 (可写入)
            return seq % NUM_ENTRIES     // 返回double-buffer索引
        // spin wait

void wr_release(Queue* q):
    atomicAdd(q->seq, 1)                 // 递增seq通知consumer
    __syncthreads()                      // CTA barrier: 确保所有线程完成写入

int rd_acquire(Queue* q, int tile_id):
    while true:
        seq = atomicAdd(q->seq, 0)
        if seq == tile_id + 1:           // producer已释放此entry
            return (tile_id) % NUM_ENTRIES
        // spin wait

void rd_release(Queue* q):
    atomicAdd(q->consumed, 1)            // 递增consumed释放entry
    __syncthreads()
```

Queue 性能（A100 硅片实测）：
- 无争用：100 M atomics/sec/CTA（→ 385-1541 GB/s/queue 上限）
- 54 queues（对应 108 SM）：aggregate 2 TB/s（37 GB/s/queue）@ 128-256KB payload
- 同步 overhead：12× reduction @ 1KB payload，<63% @ ≥64KB payload
- Payload > 256KB 时性能下降：queue 总大小超过 L2 capacity 溢写到 HBM（降至 1.5 TB/s）

术语一般如何实现？如何使用？
纯软件 C++ library，提供 acquire/release API。每个 CTA 仅 thread 0 执行 queue 管理操作。支持三种拓扑：1-to-1（producer-consumer pair）、1-to-many（multicast）、many-to-1（parallel reduction tree）。限制：每个 CUDA kernel 需手动改写（约 8 人时，10-40 LOC）将 global memory 读写改为 queue 读写。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---
