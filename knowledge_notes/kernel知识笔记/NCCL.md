## NCCL

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NCCL (NVIDIA Collective Communications Library) 是 NVIDIA 开发的高性能多 GPU 集合通信库，提供优化的 all-reduce、all-gather、reduce-scatter、broadcast、all-to-all 等集合通信原语。NCCL 为 NVIDIA GPU 拓扑（NVLink、NVSwitch、InfiniBand）做了专门优化，使用 ring、tree、collnet 等算法。在 MoE 训练的 EP 中，NCCL 提供 A2A 通信的实现。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FOLDMOE 训练中的 NCCL 通信流程：

```
# 训练配置: 2 nodes × 8 A10G GPUs, 100 Gbps network
# NCCL 通信分解:
# Intra-node (NVLink/NVSwitch): TP all-reduce, SP all-gather/reduce-scatter
# Inter-node (100 Gbps network): EP A2A dispatch/combine, DP all-reduce (gradients)

# MoE Layer A2A via NCCL:
for each Transformer block with MoE:
    # A2A Dispatch (inter-node bottleneck)
    ncclGroupStart()
    for peer in EP_group:
        ncclSend(tokens_for_peer, peer, stream=comm_stream)
        ncclRecv(tokens_from_peer, peer, stream=comm_stream)
    ncclGroupEnd()

    # Expert Compute (on compute stream, overlaps with above)
    ...

    # A2A Combine
    ncclGroupStart()
    for peer in EP_group:
        ncclSend(results_for_peer, peer, stream=comm_stream)
        ncclRecv(results_from_peer, peer, stream=comm_stream)
    ncclGroupEnd()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NCCL 是 PyTorch 分布式训练的默认通信后端（通过 `torch.distributed` 调用）
- FOLDMOE 使用 NCCL 2.21.5 + CUDA 12.4
- NCCL 通信可与 CUDA kernel 在分离的 stream 上重叠（FOLDMOE 核心依赖此特性）
- 跨节点 A2A 带宽（100 Gbps）是 FOLDMOE 评估中的主要瓶颈——这也是为什么 FOLDMOE 需要 attention-MoE pipelining 来隐藏此通信
- FSMoE 使用 NCCL 2.12 + CUDA 11.3 + PyTorch 1.12，支持 4 种 AlltoAll 算法（NCCL-A2A、1DH-A2A、2DH-A2A），通过 Dispatch/Combine 子模块的抽象实现即插即用切换。FSMoE 的在线 profiler 使用 nccl-tests 微基准测量各通信原语的 α/β 参数。
- FUSCO 构建在 NCCL 2.26.3 transport 层之上，复用 NCCL 的设备注册、连接管理和底层网络协议栈（TCP/IP、InfiniBand/RoCE），在其 network abstraction layer 之上实现 Fused Data+Communication。FUSCO 约 2000 行 C++/CUDA 实现 dComm runtime（包括 on-device descriptor interpretation、pipeline coordination 和 fused communication），作为独立 collective primitive（类似 send/recv/allgather）暴露。FUSCO 的关键洞察是 NCCL 的 all-to-all 原语对 MoE token 的 logical structure 和 routing 语义无感知——它将数据视为无结构的字节流，迫使上层框架在通信前后做显式 permute/repack。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing
