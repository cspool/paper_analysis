## Asynchronous P2P Communication for MoE Serving（MoE推理中的异步点对点通信）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asynchronous P2P Communication 是 AMoE 系统中替代传统 EP 中 barrier all-to-all 的通信机制。不同于要求所有 GPU 同时参与的集体通信，AMoE 使用点对点（P2P）NCCL 传输，通过 ZeroMQ 消息队列协调发送方和接收方，实现完全异步的 token 传输。每个 GPU 可以向任意其他 GPU 独立发送/接收 token batch，无需任何全局同步。

从kernel调度角度拆解术语：
AMoE 两阶段异步 P2P 通信流程（Figure 8）：
```
// Phase 1: Metadata Exchange (CPU, ZeroMQ)
// Sender:
zeromq_send(receiver_rank, {tensor_size, tensor_dtype, src_gpu_rank})
// → Sender CPU 不等待，继续处理下一个任务

// Receiver:
metadata = zeromq_recv()  // 从消息队列消费
// 创建一个 size 匹配的 NCCL receive buffer

// Phase 2: Tensor Transfer (GPU, NCCL P2P)
// Sender GPU:
ncclSend(tensor_data, tensor_size, receiver_rank, comm, cuda_stream)
// → Sender CPU 回到 Phase 1 处理下一个传输

// Receiver GPU:
ncclRecv(recv_buffer, tensor_size, sender_rank, comm, cuda_stream)
// → Receiver CPU 回到 Phase 1 检查 ZeroMQ queue

// Before using received tensor (receiver only):
cudaStreamSynchronize(cuda_stream)  // 确保 NCCL 传输完成
```

关键设计：(1) 单线程 Communicator 可并发管理多个传输——每个传输的 CPU 侧启动 NCCL 后立即返回，GPU 侧异步执行；(2) Sender 不用同步——batch 发出后不再使用；(3) 接收方延迟同步——仅在 Scheduler 需要使用 tensor 前确保传输完成。

术语一般如何实现？如何使用？
AMoE 中 Communicator 在 C++ POSIX thread 中实现（避免 Python GIL），使用 NCCL 作为底层 GPU P2P 传输协议。与 NCCL 标准 P2P API 的挑战：(1) NCCL send/recv 需双方同时调用——通过 ZeroMQ 提前告知；(2) 动态 tensor size——ZeroMQ metadata 包含 size 信息。ZeroMQ 是开源通用消息库（zeromq.org），提供高性能异步消息队列。

涉及论文标题：
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
