## ESP-AllGather / ESP-ReduceScatter

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

ESP-AllGather 和 ESP-ReduceScatter 是 ESP 引入的两个集合通信操作。ESP-AllGather 在 expert 计算前将 ESP group 内各 GPU 上的 token 分片收集到所有 GPU；ESP-ReduceScatter 在 expert 计算后将各 GPU 计算的输出分片聚合并按 token 分配切分。当 ESP group 对齐节点内 GPU 数时，此二操作为节点内通信（NVLink），与节点间 AlltoAll（InfiniBand）物理隔离，可实现无竞争重叠。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 ESP-AllGather/ESP-ReduceScatter 的流水线调度时间线：

```
# r=4 chunks, 节点内通信 (ESP-AG/RS) 与节点间通信 (A2A) 重叠
t:  0    1    2    3    4    5    6    7    8
    | C0:AG|    | C1:AG|    | C2:AG|    | C3:AG|
    | C0:A2A | C1:A2A | C2:A2A | C3:A2A | GAR  |
    |    | C0:RS | C1:RS | C2:RS | C3:RS |      |
    |    | C0:Exp| C1:Exp| C2:Exp| C3:Exp|      |
```

线性模型：t_{ag,r} = α_{ag} + n_{ag}/r · β_{ag}，在 Testbed-A 上 α_ag=3.37e-1, β_ag=2.32e-6（NVLink 高带宽），对比 AlltoAll 的 β_a2a=2.21e-7（InfiniBand 低带宽），节点内通信显著快于节点间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 通过 NCCL ncclAllGather/ncclReduceScatter 实现，在线 profiler 用 nccl-tests 微基准测量 α/β。FSMoE vs FSMoE-No-IIO 实验显示此重叠贡献约 5-6% 额外加速。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
