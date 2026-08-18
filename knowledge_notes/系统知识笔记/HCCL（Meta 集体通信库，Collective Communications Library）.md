## HCCL（Meta 集体通信库，Collective Communications Library）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HCCL 是 MTIA 300（ISCA'26，配套论文 arXiv:2608.00358）的集体通信库，功能上对标 NCCL（AlltoAll/AllReduce/ReduceScatter/AllGather + 点对点 send/recv），但执行模型不同：通信被编译成 work packets/subgraphs/WQEs（SEND/RECV/WRITE/WAIT/SET/REDUCE + 流控字段）卸载到 16 个 Message Engine 设备端执行，主机在通信到达设备后"uninvolved"。API 经 PyTorch Distributed 与 torchcomms 接口暴露（backend 把 Tensor/process group 语义翻译成 contiguous buffer + communicator）。控制路径用 RDMA verbs（ibv_create_qp/ibv_modify_qp 到 ready 态、express doorbell 映射、ibv_get_async_event 捕获非 WC 错误）；因无硬件 QP caching，HCCL 按需连 QP、设计算法最小化闲置 QP、communicator 内复用 QP（12 NIC × 1088 = 13056 QP 可切分/共享于 scale-up/scale-out 域）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HCCL 的一次集体通信流程（AllReduce ring，40 卡）：
```python
# 用户侧: PyTorch Distributed / torchcomms 接口
dist.all_reduce(grad_tensor, group=world)     # Tensor + process group
# HCCL 侧:
#   1. 后端翻译: Tensor → contiguous buffer, process group → communicator
#   2. 先验决策: 按 outstanding work/拓扑/消息大小/类型选算法与通道
#   3. 编译: 生成 work packet → subgraphs → WQEs（ring: ReduceScatter+AllGather 两阶段）
#   4. 提交: MTIA streaming interface 提交, CPU-C 派发到 16 ME
#   5. 执行: ME NIC interface 单 FIFO → 12 NIC express doorbell → RoCE 网络
#             NMC 就近 HBM 做 REDUCE; ME 完成后回报 CPU-C 解阻塞
#   6. 清理: HCCL 维护线程跟踪 outstanding work、回收资源、错误监控（TorchWork 完成信号）
```
并行度分层：同一 stream 的 work packet 保序、subgraph 逻辑并行（硬件可用性排队）、WQE 顺序发出仅按流控字段阻塞。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HCCL 为 Meta 专有（未开源）；接口层开源（PyTorch Distributed、torchcomms https://pytorch.org/blog/torchcomms/）；数据路径走 MTIA streaming interface、控制路径走 RDMA verbs。使用场景：DLRM 训练（AllReduce 1.6 GB/AllGather 2.1 GB/35 次 AllToAllv）与 LLM 推理（MoE AllToAll）；40 卡通信整体超 H100/NCCL 3.9×；小消息弱于 NCCL（未优化）。信息缺口：论文未给出 HCCL 的算法选择表与多 ME 分配的负载均衡策略。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
