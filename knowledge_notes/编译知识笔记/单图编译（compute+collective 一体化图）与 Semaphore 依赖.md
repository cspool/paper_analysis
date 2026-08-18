## 单图编译（compute+collective 一体化图）与 Semaphore 依赖

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
单图编译（monolithic graph）是 MTIA 300（ISCA'26）相对 GPU 的编译器特性：collectives 经 torch.export/torch.compile 与 compute 算子一起编译进同一个图，而不是像 GPU 那样 compute（PyTorch 图）与 communication（NCCL 主机驱动）分属两套执行路径。收益：降低子图启动开销、提升效率与确定性；compute 与 communication 依赖用 semaphore 管理（编译期生成的同步原语）。当前静态 shape collectives 完全支持，动态 shape 与 device-resident AllToAll（动态 send/recv counts）为进行中工作。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
单图编译的运转流程（一次 AllReduce 训练迭代）：
```python
# 编译期: torch.export/torch.compile 把 forward + collectives 合入一张图
graph = compile(model, comm_ops=[allreduce(grad), alltoall(sparse)])
#   → compute 节点与 collective 节点统一 DAG，边带 semaphore 依赖
# 运行期: 工作包（含 compute 与/或 communication 任务）到达 CPU-C
#   → CPU-C 检查依赖（semaphore）后派发:
#       compute 子图 → 72 PE（DPE/SFU/MLU/FI）
#       communication 子图 → 16 ME（HCCL 翻译为 WQEs, 每 ME 并发多 subgraph）
#   → ME 完成后向 CPU-C 报告, semaphore 释放 → 解阻塞后续 compute
```
对比 GPU：NCCL 通信由主机发 kernel 与流同步，collective 不在计算图内；MTIA 把两者统一后降低启动开销并让重叠确定化（重叠微基准双 ~100% 效率）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：MTIA 编译器（自定义 PyTorch backend）在 TorchDynamo/AOTAutograd/TorchInductor 之上把 HCCL 调用并入图；semaphore 在硬件侧由 CPU-C 依赖检查与 ME 完成回报实现（WQE 的 SET/WAIT 亦为内存级同步原语）。使用场景：DLRM 训练（AllReduce/AllToAllv/AllGather 与 GEMM 同图）+ LLM 推理。局限：动态 shape collective 未完全支持。信息缺口：论文未给出 semaphore 的硬件实现细节（计数器/邮箱）。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
