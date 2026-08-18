## NCCL 通信原语（Broadcast / Reduce / AllReduce / AllGather / ReduceScatter）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NCCL（NVIDIA Collective Communications Library）是多 GPU 集合通信库，提供 5 种原语：AllReduce（各 GPU 归约后结果广播给所有参与者）、Broadcast（一对 N：单 GPU 缓冲复制给 N 个接收者）、Reduce（N 对一归约）、AllGather（各 GPU 数据拼接后全员可见）、ReduceScatter（归约结果按块分散到各 GPU）。底层按拓扑（NVLink 环、NVSwitch 树等）选 ring/tree 算法并做 chunk 流水。论文用其原语特征化 NVLink：Broadcast 测一对 N 带宽与链路内争用、AllReduce 测 N 对 N 聚合带宽与 crossbar 争用。
- MSDP 语境（DisDP）：MSDP 每层前向/反向各 1×AllGather + 梯度 1×ReduceScatter（ZeRO-3 一层共 2×AG+1×RS），peer-based 实现下每 worker 收/发 (N-1)S/N 流量，每方向总流量 3(N-1)S/N；algorithm bandwidth（算法带宽）= 集合数据量 S / 执行时间 t（BWalg=S/t），是衡量集合性能的标准指标。SmartSwitch 无法加速单独 AG/RS：AG 只能把发送降到 S/N、接收仍 (N-1)S/N，RS 反之——集合时间由未减小的一侧主导。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文实测（DGX A100，NVLink 3.0）：单个 Broadcast 仅 ≥1GB 传输接近峰值（约 262 GB/s vs 300 GB/s 理论峰值）；7 个接收 GPU 并发时延迟仅比单接收者 +13.27%（链路内争用可忽略）；AllReduce 8 GPU 比 2 GPU 平均延迟 −9.47%、每 GPU 带宽 +12.41%（NVLink crossbar 调度增益），聚合带宽最高 1878 GB/s；而 4KB/64KB 小传输仅 1.12/17.12 GB/s——量化了"细粒度页传输严重浪费 NVLink 带宽"，是 CDFD 32MB 粗粒度复制的直接依据。NVLink 4.0（DGX H100，450 GB/s 理论峰值）呈同样结论，7 接收者延迟仅 +0.11%。伪代码（Broadcast 一对 N 概念）：
for chunk in split(buffer, chunk_size):       # 大缓冲分块流水
    dst = 0
    while dst < N:                             # 逐接收者复制（或树形转发）
        memcpy_peer(gpu[dst], src_chunk)       # NVLink 远端写
        dst += 1
    fence_across_gpus()                        # 跨 GPU 可见性

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
API：ncclBroadcast/ncclReduce/ncclAllReduce/ncclAllGather/ncclReduceScatter，各 rank 持 device buffer 调用，配合 ncclCommInitRank 建通信域；内部按节点内/节点间拓扑自动选 ring/tree 算法与分段。论文实测平台：DGX A100（8×A100 80GB SXM4，NVLink 3.0，驱动 570.148.08，CUDA 12.8）与 DGX H100（8×H100 80GB SXM5，NVLink 4.0，驱动 570.195.03，CUDA 12.8），1–8 GPU、4KB–32MB 传输多轮取平均。
- DisDP 实测与大规模算法对比：并发 GEMM 下 NCCL AllReduce 的 algorithm bandwidth 降 30%（SM + 内存带宽争用）；DisDP 的 SmartNIC push/pull 相比 NCCL 在 2/4/8 GPU 上带宽高 2%/35%/44%（并发 GEMM）、4/8 GPU 上高 8%/20%（无并发）——push/pull 相对 AllReduce 流量最多减半。大规模下 NCCL ring 集合依赖链长（DP>16 扩展差、易受干扰），PAT（层次化树形 RS/AG）更好；DisDP 在 DP=256 时吞吐 2.0×（vs ZeRO-Infinity+PAT）、15.1×（vs ring）。

Lit Silicon 补充视角（ISCA'26，FSDP 训练中 AG/RS 与 C3 重叠的行为）：FSDP 前向层与层之间用 AG 收集下一层权重分片，反向用 RS 归约上一层梯度，均与 GEMM 并发执行（C3）。但重叠并非免费——计算/通信 kernel 共享 GPU 计算与内存资源互相干扰，计算 kernel 运行时被拖慢最多 40%，且重叠率跨 GPU 不同：straggler GPU 的通信 kernel 起始更晚、重叠率恒定最低（29.6%），leader 重叠率动态增长（最高 52.7%），重叠率与 kernel 时长强相关——这是节点级性能波动的直接来源。AMD 平台（MI300X）上通信集合由 RCCL/amd-smi 生态提供（论文用 Chopper 工具解析 PyTorch trace 得到各 GPU 上 AG/RS 的起始时间与重叠率，用于 lead value 检测）。

MTIA 300 补充视角（ISCA'26，HCCL 与 NCCL 的对照）：MTIA 300 的集体通信由 HCCL 执行（非 NCCL）——API 类似（AlltoAll/AllReduce/ReduceScatter/AllGather + 点对点 send/recv，经 PyTorch Distributed/torchcomms 接口暴露），但执行路径不同：HCCL 把通信编成 work packets/subgraphs/WQEs（SEND/RECV/WRITE/WAIT/SET/REDUCE + 流控字段）卸载到 16 个 ME（RDMA verbs 控制路径），主机不参与数据面。性能对照（vs H100/NCCL）：AllGather/AllReduce/AllToAll 在 16+ 加速器或 >16 MB 消息时显著更优（16 节点 scale-up 域 + 2.2× 带宽），40 卡 DLRM 训练整体通信超 H100 3.9×；小消息 HCCL 弱于 NCCL（未优化、占比小）。

RoCC 补充视角（ISCA'26，ROP 上执行 NCCL 原语的对照）：RoCC 论文以 NCCL/RCCL 为 baseline 软件库（NCCL 把 CC 编译成独立 CC kernel 在 SM 上执行），但把 CC 的执行引擎从 SM 换成 GPU 的 ROP 硬件：新增每条 CC 一个 intrinsic（rocc_allreduce 等）+ 一条 ISA 指令（ROP_AR/ROP_AG/ROP_A2A），并遵循 NCCL 的 CC 算法设计把 collective 分解为 primitive（send/recv/recvReduceSend/recvReduceCopySend/recvCopySend）再译成 ROP μOp。对比结论：CC 在 SM 上执行占 tensor 并行执行时间 40%-70%（PyTorch distributed + NCCL 实测），因 SM 距内存远、CC 网络/内存 bound 浪费算力；RoCC 把 CC 卸载到近内存 ROP 后 CC-only 延迟大消息下 AllReduce 快 35%、AllGather 快 11%、AllToAll 快 25%。
- STAGE 补充视角（ISCA'26）：STAGE 生成的执行图在验证时与 NCCL 实际行为对齐——NCCL 实现 AllToAll 时将其分解为多次 Send/Recv，Kineto 只记录分解后的原语，因此 STAGE 也把 AllToAll 通信量按 Send/Recv 分解后再与真实 trace 对比（Table VII，总通信量误差 0.000%~2.980%）；通信算子估时交给 ASTRA-Sim 模拟，端到端 runtime 平均误差 3.53%。通信匹配器按 producer/consumer 张量分布匹配出 NCCL 集合原语（AllReduce/AllGather/ReduceScatter/AllToAll 及其组合）。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication

PipeComm 补充视角（ISCA'26，NCCL 作为 baseline 的对照）：PipeComm 在真实两节点 ×8 NVIDIA L20 GPU（节点内 PCIe switch、节点间 InfiniBand+RDMA 的分层异构带宽）上，把 PipeComm 合成的 AllReduce schedule 与 NCCL v2.20.3 对比（2MB–2GB 消息）：NCCL 按 GPU 拓扑在 ring/tree 算法间动态选择，但未显式建模底层链路的详细特征（分层异构下 PCIe 高带宽 + IB 低带宽的静态启发式无法充分榨取非对称带宽），Pipe-Sol 平均取得 1.24× speedup over NCCL、1.18× over partitioned TACOS、1.19× over 非分区 TACOS。这佐证了 NCCL 对均匀/对称拓扑（NVLink 环、NVSwitch 树）近似最优、但在异构/非对称带宽层次下留有提升空间，也说明拓扑感知合成（PipeComm/TACOS 类）可作为 NCCL 的补充后端。

Tetris 补充视角（ISCA'26，NCCL 作为 serving 中 KV cache 传输与并发 communicator 的使用）：Tetris 用 NCCL 实现两类跨实例通信——(1) CDSP cache balancing（chunk 间把前序 KV cache 均匀重分布到当前实例组，复用 ring 通信器与下一层 prefill 跨层重叠）；(2) prefill→decoding 的 KV cache 流式传输（handshake backend 分配后由 send/receive engine 执行）。关键依赖：NCCL 自 v2.26 支持并发 communicator 执行，使多组 cache transfer 可并行、与计算重叠；Tetris 预留专用 buffer 与 CUDA stream 提升带宽利用率。论文量化：CDSP balancing 额外开销 ≤1.8%，handshake 传输 0.6%-11.8%（平均 2.1%），backend 减半压力测试下 RPC 开销 1.5%-5.4%。与 ring attention 的 NVSHMEM one-sided 传输分工：ring 内 K/V 轮转用 NVSHMEM（kernel 内 fine-grained），跨实例/跨阶段的 cache 汇聚用 NCCL（collective/异步传输）。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
