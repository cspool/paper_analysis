## Pipe-A2A (Pipelined All-to-All)

术语是什么？
Pipe-A2A 是 ScheMoE (EuroSys '24) 提出的一种新型 All-to-All 集合通信算法，专为异构 GPU 集群（intra-node 高带宽 + inter-node 较低带宽）上的 MoE 分布式训练设计。其核心思想是将 A2A 中的 Send/Recv (SR) 操作按 GPU 对是否位于同一节点分为 intra-node SR 和 inter-node SR，分配在两个独立的异步 CUDA stream（Intra-Stream 和 Inter-Stream）上并行执行，使得 intra-node 通信可以被 inter-node 通信的时间隐藏。

假设集群有 N 个节点、每节点 M 个 GPU（P = N×M），对所有 GPU i ∈ [0, P-1]，A2A 包含 P 个 SR(i,j) 操作。其中 j 表示目标 GPU。若 i 和 j 同节点，SR(i,j) 为 intra-node 操作（使用 Intra-Stream）；否则为 inter-node 操作（使用 Inter-Stream）。两 stream 并发执行，理论执行时间为：

$$t_{pipea2a} = \max\{M \times t_1, (P - M) \times t_2\}$$

而传统顺序执行的 NCCL-A2A 时间为：

$$t_{nccla2a} = M \times t_1 + (P - M) \times t_2$$

其中 t₁ 为单次 intra-node SR 耗时，t₂ 为单次 inter-node SR 耗时。理论最大加速比 S_max = (M×t₁ + (P-M)×t₂) / max(M×t₁, (P-M)×t₂)。当 t_intra ≈ t_inter 时加速最大（接近 2×）。

从kernel调度角度拆解术语：
以 8 GPU（2 node × 4 GPU）为例，GPU 0 的 Pipe-A2A 执行流程：

```
// GPU 0 的 A2A dispatch（输入 tensor I_0 按 expert 切分为 8 份）
// 同节点 GPU: GPU 0,1,2,3; 跨节点 GPU: GPU 4,5,6,7
// Intra-Stream 和 Inter-Stream 为两个独立的 cudaStream_t

// Intra-Stream (处理同节点 SR):
cudaStream_t intra_stream;
for target in [0, 1, 2, 3]:  // GPU 0 到同节点 GPU 的 SR
    if target != 0:
        cudaMemcpyAsync(send_buf[target], I_0[target], size, 
                        cudaMemcpyDeviceToHost, intra_stream)  // or GPUDirect RDMA
    // Recv 对端数据...
// 总共 M=4 个 intra-node SR 操作

// Inter-Stream (处理跨节点 SR, 并行执行):
cudaStream_t inter_stream;
for target in [4, 5, 6, 7]:  // GPU 0 到跨节点 GPU 的 SR
    // 通过 InfiniBand/NCCL 发送
    ncclSend(I_0[target], size, target, comm, inter_stream)
// 总共 P-M=4 个 inter-node SR 操作

// 两 stream 并发: cudaStreamSynchronize(intra_stream)
//               cudaStreamSynchronize(inter_stream)
// 总耗时 ≈ max(intra_time, inter_time) 而非 intra_time + inter_time
```

ScheMoE 实验表明：在 32 GPU (8×4 RTX2080Ti) 集群上，当消息大小 ≥ 200MB 时 Pipe-A2A 实现 1.4×-2× 加速优于 2DH-A2A 和 NCCL-A2A；小消息时约 3%-5% 提升。在 BERT-Large-MoE (~6.5B params) 上 Pipe-A2A 贡献有限（A2A 输入仅 524KB），说明该算法对大消息 MoE 配置（高 M × B × L）效果最显著。

术语一般如何实现？如何使用？
Pipe-A2A 在 ScheMoE 中通过 C++/CUDA 实现为 AbsAlltoAll 的子类。用户可通过继承 AbsAlltoAll 接口实现自定义 A2A 算法。实现依赖：两个 cudaStream_t（Intra-Stream 和 Inter-Stream），GPU Direct RDMA（同节点 GPU 间通过 PCIe/NVLink 直接访问内存），NCCL（跨节点 InfiniBand 通信）。在 ScheMoE Python 接口中通过 `all_to_all_impl = ScheMoE.PipeAlltoAll` 指定使用 Pipe-A2A。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling
