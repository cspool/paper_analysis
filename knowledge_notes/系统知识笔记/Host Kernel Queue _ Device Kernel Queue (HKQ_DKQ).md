## Host Kernel Queue / Device Kernel Queue (HKQ/DKQ)

术语是什么？通过联网搜索让回答具体和精准。
HKQ (Host Kernel Queue) 和 DKQ (Device Kernel Queue) 是Infera中用于低延迟kernel launch的多级队列架构。HKQ是host端的priority queue，存储fused kernel的binary code、arguments和launch configurations，按launch timestamp排序。DKQ是device端的kernel buffer，daemon kernel从中取kernel并发射。Kernel通过GDRCopy从HKQ以低延迟（<100ns小数据/<5μs典型kernel）拷贝到DKQ，避免传统cuLaunchKernel的host-device同步开销。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HKQ/DKQ在Infera kernel launch pipeline中的运转流程：
```
FuseKernels() → fused kernel {
    binary code,
    arguments,
    launch config (gridDim, blockDim, sharedMem, ...)
}
        ↓
    HKQ (Host Kernel Queue, priority queue sorted by launch timestamp)
        ↓ GDRCopy (gdr_copy_to_mapping)
    Device-side kernel slot (placeholder kernel overwrite)
        ↓
    DKQ (Device Kernel Queue, managed by daemon kernel)
        ↓ cudaLaunchDevice (fire-and-forget)
    GPU SM execution
```
1. **HKQ入队**：kernel fuser生成fused kernel→入HKQ（按launch timestamp排序的priority queue）→标记为"on-device"
2. **HKQ→DKQ传输**：host launcher通过GDRCopy将kernel code拷贝到device kernel slot（覆盖预留placeholder kernel）、arguments拷贝到global memory→将kernel pointer+arg pointer+launch config入DKQ
3. **DKQ消费**：daemon kernel从DKQ的shared-memory double-ended queue取kernel→cudaLaunchDevice fire-and-forget launch→GPU立即调度
4. **Completion**：daemon kernel被通知kernel完成→cudaGetLastError错误检查

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Infera中HKQ/DKQ的关键实现细节：(1) Driver-level placeholder kernel slots：预留在GPU memory中的特殊kernel slot，运行时通过driver-level修改覆盖其code section，避免cuModuleLoad的global host-device synchronization；(2) GDRCopy (gdr_copy_to_mapping) bypasses DMA engines，实现超低延迟host→device传输；(3) DKQ使用device-side shared memory实现double-ended queue，daemon kernel可直接访问无需PCIe round-trip；(4) Fire-and-forget launch无需等待前序grids完成，避免CUDA stream的Head-of-Line blocking。HKQ还维护host memory pool缓存已fused kernel文件以便复用。

涉及论文标题：
- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
