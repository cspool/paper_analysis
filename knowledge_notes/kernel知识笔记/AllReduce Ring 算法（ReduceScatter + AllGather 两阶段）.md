## AllReduce Ring 算法（ReduceScatter + AllGather 两阶段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AllReduce Ring 是带宽最优（bandwidth-optimal）的 AllReduce 实现（Patarasuk & Yuan, JPDC 2009）：N 个节点组成逻辑环，每节点把数据分成 N 个分片，分两阶段——**ReduceScatter**（N-1 步，每步节点把本地分片发给下一个节点、从上一个节点收分片并归约，最终每节点持有全局和的 1/N 分片）与 **AllGather**（N-1 步，每步转发已归约分片，最终每节点持有完整全局和）。总通信量 2(N-1)/N × 数据量，是渐近最优；代价是依赖链长（N-1 步串行），小消息/大集群下延迟高。MTIA 300（ISCA'26）把 ring 算法编码为 WQE 数组在 Message Engine 上执行（见 WQE 条目）：HCCL 按消息大小/拓扑在 ring/recursive doubling/ordered tree 间选择。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
4 节点 AllReduce ring 的具体过程（每节点数据 B 分 4 片）：
```python
# 阶段1: ReduceScatter（3 步，每步: RECV(prev) → REDUCE(A+recv) → SEND(next)）
#   步1: 节点0 把 c0 发节点1, 节点1 归约 c0 到本地...
#   步2: 继续转递已归约分片
#   步3: 完成后每节点持有一个全局归约分片 g_i
# 阶段2: AllGather（3 步, 每步: SEND(已归约分片) → RECV(上一节点分片) 拼接）
#   步1: 节点0 把 g0 发节点1, 节点1 拼接 g0
#   步2-3: 继续, 最终每节点持有 [g0,g1,g2,g3] = 完整 AllReduce 结果
```
MTIA 300 中阶段间/步间依赖用 WQE 流控字段表达（图 10：自下而上第一对 RECV/SEND 无依赖并行发布，ADD 依赖前 RECV、再解阻塞下一 RECV/SEND；AllGather 每步依赖前一步）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NCCL/HCCL 库自动选 ring；MTIA 300 上 ring 由 HCCL 翻译成 WQE（REDUCE 在 NMC 执行归约、SEND/RECV 走 RDMA NIC）并在 16 ME 上并行 subgraph。使用场景：DLRM 训练梯度 AllReduce（40 卡、1.6 GB 入站，MTIA 300 通信整体超 H100 3.9×）；大消息/多卡时 MTIA 300 靠 16 节点 scale-up 域与 2.2× 带宽占优，小消息（依赖链延迟主导）NCCL 更优。信息缺口：论文未披露每消息的 ring vs tree 选择阈值。

RoCC 补充视角（ISCA'26，ring AllReduce 的 ROP 硬件执行）：RoCC 论文采用 4-GPU/8-GPU 的 NCCL 式 ring 算法，把 ring AllReduce 分解为 7 阶段（4 GPU）primitive 序列：send → recvReduceSend×2 → recvReduceCopySend → recvCopySend×3 → recv，每 primitive 再译成 ROP μOp（如 recvReduceCopySend = ReadDoorbell→DepBarrier→ReadDoorbell→Add→Write→RingDoorbell），由 ROP 的 collective/primitive 双译码器查表执行、doorbell 门铃跨 GPU 接力（8 GPU 时最多 15 阶段）。与软件 ring 的区别：ReduceScatter/AllGather 的归约（Add）与转发（RingDoorbell）由近内存 ROP 的 4 路 ALU 完成，SM 全程只算 GEMM；结果平均 51% 加速 vs SM 顺序 baseline、23% vs oracle 软件重叠（20% SM 专做 CC）。
涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication

PipeComm 补充视角（ISCA'26，ring 作为 baseline 的对照）：PipeComm 把 Ring AllReduce 视为"通用但不拓扑感知"的 baseline——NCCL 用高带宽 ring 算法（低延迟则用 tree），但在物理拓扑与逻辑环不对齐时产生显著低效。仿真对照（8×8 2D Torus，α=150ns、1/β=16GB/s）：同质 Torus 上 ring 近似最优（常数轮通信即可饱和对分带宽），Pipe-Sol 需放宽 II 约束回到最优非流水最短路径才勉强追平 Themis；但在 2D Mesh/异构拓扑上 Pipe-Sol 相对 ring 系 baseline 大幅领先（vs MultiTree 2.23×、vs BlueConnect 1.98×）。这量化了 ring 的优势边界：高对称拓扑（环/Torus）上 ring 已最优、流水线化收益递减（细粒度 stage 的延迟开销反而抵消收益），非对称/异构拓扑上 ring 依赖链与链路错位使其带宽利用率不足 65%——正是拓扑感知合成（PipeComm）的用武之地。
