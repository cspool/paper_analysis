## GPU-CPU FIFO Channel with TransferCmd（GPU-CPU FIFO 通道与传输命令）

术语是什么？
UCCL-EP 设计的 lock-free FIFO 通道，用于 GPU threads 向 CPU proxy 传递 128-bit 的 TransferCmd（传输命令描述符）。通道的 head 元数据在 CPU 内存中、tail 元数据在 GPU 内存中，双方各自访问本地内存侧的头/尾以减少 PCIe 穿越。GPU 侧缓存 tail index 避免跨 PCIe 读取。支持 4 种 TransferCmd：Write、Atomics、Drain、Barrier。

从kernel调度角度拆解术语：
```
// TransferCmd 结构 (128-bit = 16 bytes, 可单条 GPU 指令+MMIO doorbell 写入):
//   type: Write | Atomics | Drain | Barrier
//   dest_rank: 目标 GPU rank
//   src_offset: 源 buffer offset (symmetric memory)
//   dst_offset: 目标 buffer offset
//   length: 传输字节数
//   seq_num: 序列号 (用于 ordering)

// GPU 侧 API:
//   idx = Push(TransferCmd)    // GPU thread 入队命令，返回 index
//   CheckCompletion(idx)       // GPU thread 检查命令是否被 CPU 消费完成

// CPU 侧 API:
//   cmd = Poll()    // CPU proxy 读取但不移除队首命令
//   Pop()           // CPU proxy 移除队首命令 (表示已完成处理)

// 多 FIFO channels per GPU:
//   8 FIFO channels / GPU × 4 CPU threads
//   同 channel 内的命令保证 ordering，跨 channel 不保证
//   GPU kernel 将需要 ordering 的消息映射到同一个 channel

// CPU proxy 背压机制:
//   kMaxInflight: 每个 channel 最大 in-flight 命令数
//   当 channel 满时 GPU thread 阻塞在 Push() 上
//   延迟 Pop() = 延迟 GPU enqueue = rate-limiting GPU sender
```

术语一般如何实现？如何使用？
实现依赖 GPU memory（device buffer）+ CPU memory（host buffer）的共享 FIFO。GPU writes 需 bypass L2 cache（volatile + memory fence），CPU writes 需 flush 到 host memory。在 NVLink-C2C（GH200）等 cache-coherent CPU-GPU 互联上，一致性由硬件保证。FIFO 吞吐量达 8 Mops/s（单 channel），latency 比网络延迟低一个数量级。TransferCmd 使用 offset 而非全局地址（配合 symmetric memory），节省 bits 并使 128-bit 紧凑编码可行。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
