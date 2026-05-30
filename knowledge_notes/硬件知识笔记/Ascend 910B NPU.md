## Ascend 910B NPU

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ascend 910B 是华为推出的 AI 训练/推理 NPU（Neural Processing Unit），属于华为 Ascend 系列 AI 处理器的第二代产品。基于华为自研的 Da Vinci 架构，提供 FP16/BF16/INT8 等多种精度的 AI 计算能力，配备 64 GB HBM 显存，通过 HCCS 实现 intra-node 全互联（最高 480 Gbps per link），通过 RoCE 实现 inter-node 通信（最高 200 Gbps）。在 MixServe 中，Ascend 910B 集群是主要实验平台之一——4 台 Atlas 800T A2 server 组成 32 NPU 集群。

从硬件架构角度拆解术语：
Ascend 910B 的架构组成和 MixServe 中的执行流程：

```
Single NPU 在 MixServe 推理中的执行:
1. Weight Loading: 通过 CANN 加载 TP shard 到 64 GB HBM
2. Attention (Cube Unit): QKV projection → attention → output projection
3. Intra-node RS (HCCS): NPU 间 reduce-scatter 交换 partial hidden states
4. Inter-node A2A (RoCE): NPU 间 pairwise isend/irecv for expert routing
   (Step 3 和 4 通过 fused algorithm 异步重叠)
5. Expert FFN (Cube Unit): Gate+Up projection → SiLU → Down projection
6. Intra-node AG + Inter-node Dispatch (HCCS+RoCE 重叠)
7. KV Cache 管理: HBM 中 PagedAttention block 管理
```

计算单元：Da Vinci Cube Unit（矩阵乘加，主力 AI 计算）+ Vector Unit（向量运算）+ Scalar Unit（标量运算）。

术语一般如何实现？如何使用？
- 通过 CANN SDK + PyTorch Ascend 适配版使用，HCCL 自动处理 collective 通信。
- 与 H20 对比：H20 96 GB vs 910B 64 GB（显存差距）；NVLink 900 GB/s vs HCCS 480 Gbps（互联差距）；得益于较小的显存和互联带宽，hybrid TP-EP 在 Ascend 910B 上的加速比（TTFT 2.67×）大于 H20（1.23×），因为瓶颈更严重、优化空间更大。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm
