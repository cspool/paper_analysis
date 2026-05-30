## MPS (NVIDIA Multi-Process Service)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MPS (Multi-Process Service) 是 NVIDIA 提供的 CUDA 软件组件，允许来自不同 OS 进程（不同 CUDA context）的 GPU kernel 在同一 GPU 上同时执行。通过启动 MPS control daemon 和 MPS server，多个 client 进程将 kernel dispatch 请求发送给 MPS server，由 server 统一调度到 GPU。与 Priority Streams（仅限同进程）和 Time-Slicing（时分复用，无真正并发）不同，MPS 支持真正的 cross-process spatial sharing——不同进程的 thread blocks 可以在同一 SM 上 colocate。MPS 可配置 per-client 的最大 thread 使用比例（如 `set_active_thread_percentage`），但**无优先级概念**（所有 client 被同等对待，FCFS + leftover policy 调度）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

MPS 在 DL 训练+推理并发场景中的执行流程：

```
Host:
  nvidia-cuda-mps-control -d          // 启动 MPS control daemon
  export CUDA_MPS_PIPE_DIRECTORY=...  // 配置通信管道
  // Client 1 (Training, best-effort):
  process_train:
      cudaSetDevice(0)
      for batch in training_data:
          train_kernel<<<grid, block>>>()  // 通过 MPS server 提交
  // Client 2 (Inference, latency-sensitive):
  process_inference:
      cudaSetDevice(0)
      for request in inference_queue:
          inf_kernel<<<grid, block>>>()    // 通过 MPS server 提交

GPU MPS Server:
  // 接收两个 client 的 kernel dispatch 请求
  // FCFS 处理: 先到达的 kernel 先被调度
  // Leftover policy: 队头 kernel 所有 blocks 调度完才处理下一个
  Client_Queue = [train_kernels..., inf_kernels...]
  
  for kernel in Client_Queue (FCFS order):
      dispatch kernel to GPU thread block scheduler
      // thread block scheduler uses leftover + most-room
      // blocks from different clients can colocate on same SM

SM-level:
  SM0: [TrainBlock(256t)][InfBlock(64t)][InfBlock(64t)]  // colocation
       ↑ 如果 client thread limit 设为 100%，两个 client
         的 blocks 可能在所有 SM 上自由混合
```

本文在 RTX 3090 上的核心发现：
- MPS utilization（training execution time）最好（通常仅比 baseline 多 20-30s，vs priority streams 30-40s, time-slicing 50s+）
- 但 inference turnarround time 可能因无优先级而显著退化（ResNet-152: +100%）
- RNNT 训练任务几乎无 large kernels → MPS 对 inference TT 的影响较小
- Leftover policy + FCFS 使得后到达的 inference kernel 可能被已有 training kernel 的 blocks 堵塞
- Compounded delay 在 100% thread limit 下也影响 MPS（与 priority streams 同理）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

配置：`nvidia-smi -c EXCLUSIVE_PROCESS` 设置 GPU 为独占模式，启动 MPS daemon 后 client 进程通过 `CUDA_MPS_PIPE_DIRECTORY` 连接。Thread limit: `export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`。推荐设置: `100 / N`（N=client 数），但本文用 100% 以最大化 utilization。MPS 允许 client 独立管理自己的 GPU memory（无 memory isolation），需谨慎 batch size 配置避免 OOM。局限性：(i) 无优先级 → latency-sensitive task 无法被优先服务；(ii) leftover policy 下大 kernel 会阻塞后续 kernel；(iii) 所有 client 的 kernel 共享 HBM bandwidth/L2 cache，存在跨进程 memory bandwidth 竞争。

涉及论文标题：
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
