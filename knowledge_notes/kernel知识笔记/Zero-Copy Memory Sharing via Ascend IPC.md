## Zero-Copy Memory Sharing via Ascend IPC

术语是什么？

Zero-Copy Memory Sharing via Ascend IPC 是 ElasticMoE 在 Ascend NPU 上实现的跨进程零拷贝张量共享机制。它允许两个独立进程（HMM 守护进程和 IMM 推理进程）引用同一块 NPU HBM 物理内存，而无需实际拷贝数据。核心流程：(1) HMM 使用 `IpcSafeAllocator` 分配 IPC 兼容的 HBM 物理内存；(2) 通过 `rtIpcSetMemoryName()` 为内存块注册唯一名称；(3) 通过 `rtSetIpcMemPid()` 将目标进程 PID 加入访问白名单；(4) 通过 ZMQ/UNIX domain socket 将内存句柄名称传递给目标进程；(5) 目标进程通过 `rtIpcOpenMemory()` 导入物理内存指针；(6) 通过 `torch::from_blob()` 将裸指针封装为 PyTorch tensor。

从 kernel 调度角度拆解术语：

```
Zero-Copy 操作伪代码：

// 发送端 (HMM, Process A)
tensor = ipc_safe_allocator.allocate(shape, dtype)  // aclrtMalloc + IPC flag
handle_name = "model_layer_0_attention"
rtIpcSetMemoryName(tensor.data_ptr(), handle_name)
rtSetIpcMemPid(target_pid)  // 白名单 IMM 进程
send_over_socket(target_socket, handle_name, shape, dtype, stride)

// 接收端 (IMM, Process B)
handle_name, shape, dtype, stride = recv_from_socket(source_socket)
physical_ptr = rtIpcOpenMemory(handle_name)
tensor = torch::from_blob(physical_ptr, shape, dtype, stride)
// tensor 现在指向与 Process A 完全相同的物理内存
// 读/写操作直接在 HBM 上进行，无拷贝
```

与 P2P copy 的区别：zero-copy 在两个进程引用同一 NPU 时使用（共享 NPU 的场景），P2P copy 在数据需要跨越不同 NPU 时使用。Zero-copy 速度远快于 P2P copy（无实际数据传输）。

术语一般如何实现？如何使用？

基于华为 Ascend CANN IPC API 实现。在 CUDA 生态中等效为 `cudaIpcGetMemHandle` + `cudaIpcOpenMemHandle`。在 ElasticMoE 中用于共享 attention 权重、KV cache 和 expert 权重。Ablation 表明禁用 ZeroCopy 后 downtime 从 0 升至 67.40s（scale-up 期间无法共享权重和 KV cache）。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models
