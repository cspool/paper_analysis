## MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  MegaScale-Infer 实现两类 kernel 级优化：
  1. **High-Performance M2N Communication Library**：针对 MoE 推理中 attention 模块到 expert 模块的 M（发送者数量）×N（接收者数量）通信模式，自研 M2N 通信库替代 NCCL 的 peer-to-peer primitives。核心优化包括：
     - **消除 GPU-to-CPU 数据拷贝**：使用 GPUDirect 技术，数据直接从 GPU 内存通过 RDMA 发送，无需经过 CPU proxy。
     - **消除 Group Initialization Overhead**：NCCL 的 group operations 按 batch of 8 处理（随 N 增大性能恶化），M2N 使用独立的 point-to-point RDMA write with immediate，无 group batching 限制。
     - **消除 GPU Synchronization**：M2N Sender 使用 CUDA events 等待 kernel 完成 → cuStreamWaitValue32 阻塞 stream → Core Sender RDMA 传输 → poll completion queue → 通过共享内存 flag 唤醒 stream。避免了 NCCL 中复杂的 GPU 同步操作和 device memory accesses（这些是性能不稳定来源）。
     - **M2N Receiver**：使用 GDRCopy 进行 GPU 内存 flush 操作确保数据一致性，无需 GPU-to-GPU 拷贝。
     - **Traffic-oriented optimizations**：(a) 高优先级 ACK packets：ACK 与数据包隔离到高优先级队列，避免双向通信中的 ACK 排队延迟；(b) 拥塞控制微调：针对不均衡通信场景微调 congestion control 算法，减少 rate-limiting 效应。
     - 与 DeepEP 对比：M2N 使用 CPU 进行 inter-node 通信（单线程 CPU 足以在几百 KB 数据量下饱和带宽），DeepEP 使用 GPU-to-GPU 通信（消耗 GPU SM 资源但并行处理能力更强）。M2N 不需要 PTX 级别的 low-level 优化。
  2. **Fused Kernels**：
     - **TP Communication-Computation Fusion**：使用 Flux 将 tensor parallelism 的 all-gather/reduce-scatter 通信与相邻 GEMM 操作融合为单 kernel，消除 TP 通信开销。
     - **Sequential Memory-Intensive Operator Fusion**：将 gating network（router 计算）、top-k expert 选择、per-expert token count 计算、normalized token weights 计算、token scatter 等多个连续的小型 memory-intensive 操作融合为一个 kernel，减少 kernel launch 和 memory access。
  实验比较了 M2N 通信库 vs NCCL（median/P99 latency, throughput）在不同 data size（1KB–2048KB）和不同 M/N 配置下，以及与 perftest（CPU baseline）的对比。同时通过 ablation study 测量了 M2N 优化带来的端到端 throughput 增益。

- 后端平台是什么，配置是什么。
  同构集群：8 节点，每节点 8×NVIDIA 80GB Ampere GPU，128 CPUs，2 TB host memory，8×200 Gbps InfiniBand NICs（每 GPU 一张 NIC），节点内 400 GB/s NVLink。
  异构集群：NVIDIA H20（96 GB, 4096 GB/s bandwidth, 900 GB/s NVLink, 4×400 Gbps NICs）+ NVIDIA L40S（48 GB, 864 GB/s bandwidth, PCIe intra-node, 2×400 Gbps NICs）。
  bfloat16 所有计算。

- 评估性能的软件/脚本是什么。修改了什么。
  - M2N 通信库：实现为 PyTorch C++/CUDA extension，约 4,900 行 C/C++ + 5,000 行 Python。核心依赖：GPUDirect（GPU 内存直接 RDMA）、GDRCopy（GPU memory flush via CPU）、RDMA write with immediate、CUDA driver API (cuStreamWaitValue32)、CUDA runtime API (cudaEventQuery)、NVIDIA RDMA 完成队列。
  - 对比基线：NCCL peer-to-peer primitives（group operations + send/recv）和 perftest（CPU-side networking microbenchmark，作为延迟下界）。
  - 测试方法：micro-benchmark 测量不同 data size（1KB–2048KB）和不同 M×N 配置（8×8, 16×16, 32×32）下的 latency 分布（median, P99）和 throughput。
  - 与 NCCL 的关键区别：M2N 避免 NCCL 的 GPU-to-CPU 中间拷贝（即使 user buffer registration 也无法完全消除）、batch size=8 的 group operation 限制、general collective operation setup 开销、GPU 同步/memory access 导致的不稳定性。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未声明开源。M2N 通信库是 ByteDance 内部实现的生产级组件。

  **M2N 通信全流程（从 GPU kernel 输出到 Remote GPU 显存可用）**：
  ```
  // ===== M2N Sender 端（Attention GPU → Expert GPU） =====
  // 步骤 1: 等待前序 kernel 完成
  cudaEventRecord(event_start, stream);          // 记录 kernel 完成事件
  cudaEventSynchronize(event_start);             // 确保 tensor 数据就绪
  
  // 步骤 2: 阻塞 CUDA stream（防止后续 kernel 覆盖正在传输的数据）
  cuStreamWaitValue32(stream, &flag,             // 阻塞直到 flag 被远端更新
                      CU_STREAM_WAIT_VALUE_EQ, 0);
  
  // 步骤 3: RDMA 数据传输（CPU Core Sender 执行）
  // 使用 RDMA write with immediate：
  //   - 直接将 GPU 显存中的数据写入远端 Expert GPU 的注册内存
  //   - "immediate" 值携带 metadata（如 token count、expert ID）
  //   - 无需 CPU proxy buffer 拷贝（GPUDirect RDMA）
  ibv_post_send(qp, wr, &bad_wr);                // 提交 RDMA write + immediate
  // 每个 QP 以 doorbell ring 方式并行发送
  // 单线程 CPU 即可饱和带宽（每连接几百 KB 数据量下）
  
  // 步骤 4: 轮询完成队列确认传输
  while (ibv_poll_cq(cq, 1, &wc) == 0) { /* spin */ }
  // 确认 RDMA write 完成（数据已写入远端 GPU 显存）
  
  // 步骤 5: 唤醒 CUDA stream
  flag = 1;                                      // 共享内存 flag 更新
  // cuStreamWaitValue32 检测到 flag 变化 → 唤醒 stream
  // 后续 kernel 可以安全地复用 registered tensor 内存
  
  // ===== M2N Receiver 端（Expert GPU 侧） =====
  // 步骤 1: 等待接收 buffer 可用
  cudaEventRecord(event, stream);
  cudaEventSynchronize(event);
  
  // 步骤 2: 阻塞 CUDA stream
  cuStreamWaitValue32(stream, &recv_flag, CU_STREAM_WAIT_VALUE_EQ, 0);
  
  // 步骤 3: 轮询完成队列，确认数据已到达
  while (ibv_poll_cq(recv_cq, 1, &wc) == 0) { /* spin */ }
  
  // 步骤 4: GPU 缓存一致性 flush（关键！）
  // 因为数据是通过 RDMA 直接写入 GPU 显存的
  // GPU 的 L2 cache 可能持有该内存区域的 stale data
  // 使用 GDRCopy 执行 flush 操作确保后续 kernel 读到最新数据
  gdr_copy_to_mapping(...);                      // GDRCopy flush via CPU BAR mapping
  // 等效于 NCCL 的 GDR flush operation
  // 参考：https://github.com/NVIDIA/nccl/issues/683
  
  // 步骤 5: 唤醒 CUDA stream
  recv_flag = 1;
  // Expert FFN kernel 开始执行，读取接收到的 token embeddings

  // ===== 性能测量原理 =====
  // Latency: 从 Sender 端 event_start 到 Receiver 端 recv_flag 被设置的 wall-clock time
  // Throughput = total_bytes_transferred / T_c（含所有 QP 并行传输）
  // T_c = max(send_time, recv_time) per Equation 6
  // 
  // Key optimization comparison with NCCL:
  // NCCL 额外开销来源：
  //   1. GPU→CPU proxy buffer copy（即使 user buffer registration 也无法完全消除）
  //   2. Group operation 的 batch-of-8 处理
  //   3. Group init/launch/topology verification overhead
  //   4. GPU 同步操作（cudaDeviceSynchronize）引发的不稳定性
  // M2N 消除所有这些开销，仅保留：
  //   1. cudaEventSynchronize（kernel 完成等待）
  //   2. RDMA write + poll CQ
  //   3. GDRCopy flush
  ```

  关键性能结果：
  - 256KB data size（最常见 MoE serving 场景）：68.2% median latency 降低、92.9% P99 latency 降低、4.2× throughput 提升 vs NCCL。
  - P99 latency 稳定性：随 N 增大 NCCL 的 tail latency 显著上升（32 receivers 时 P99 > 1000μs），M2N 保持稳定（<100μs），因消除 GPU synchronization 消除了主要的不稳定性来源。
  - Throughput scalability：跨不同 M/N 配置（8×8 到 32×32）M2N 均保持 3.3×–5.8× throughput 提升。
  - Ablation study：disaggregated architecture alone 提升 4.66× vs colocated baseline；加上 M2N 优化额外提升 1.53×（因通信可被 pipeline 完全覆盖）。
