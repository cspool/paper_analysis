## UCCL-EP Portable Expert-Parallel Communication

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：(a) **Lock-free GPU-CPU FIFO Channel**：GPU threads 通过 shared memory 的 FIFO 队列向 CPU proxy 发送 128-bit TransferCmd（Write/Atomics/Drain/Barrier），CPU 通过 Poll/Pop 消费命令并执行 GPUDirect RDMA。GPU 侧缓存 tail index 避免 PCIe read，CPU 和 GPU 分别将 head/tail 元数据置于各自内存侧以减少跨 PCIe 访问。(b) **LL（Low-Latency）Kernel**：immediate send token activation via CPU proxy，接收端 CPU 用 immediate data 中的 expert index 做 conditional check（partial completion fence）确保 atomic 仅在指定 expert 的 X 个 writes 完成后才 apply。(c) **HT（High-Throughput）Kernel**：多 ring buffer 通信通道 + token deduplication + intra-node forwarding + hierarchical reduce。per-channel local ordering 通过将同一通道消息映射到同一 FIFO queue 保证，接收端 CPU buffer out-of-order 的 atomic 消息，按 sequence number 有序 apply。(d) **CPU proxy 模拟 RDMA atomics**：在 AWS EFA 等不支持硬件 atomics 的 NIC 上，发送端将 atomic 值打包进 immediate data 的 RDMA write，接收端 CPU 更新 host memory（cudaMallocHost）上的 completion counter，GPU 直接读取 host memory 用于 control decisions。
  - 实验比较：(a) Microbenchmark：dispatch/combine latency 对比 UCCL-EP vs DeepEP vs PPLX vs NCCL/RCCL vs CPU-assisted IBGDA vs Theoretical Best（RDMA bandwidth 理论值），在 NV_EFA3/NV_EFA4/NV_IB/AMD_CX7/AMD_BRC 上 EP=2/8/16/24/32，LL 和 HT 两种模式，varying tokens（128~4096）；(b) FIFO 性能 stress test：单 FIFO 的 message throughput（ops/s）vs latency，NV_EFA3 和 AMD_BRC；(c) CPU threads 数量敏感性：1/2/4 threads per GPU；(d) sender-side vs receiver-side delivery semantics 强制方式对比。

- 后端平台是什么，配置是什么。
  - NVIDIA H200×8（141GB HBM, 132 SMs）+ AWS EFAv3 200G×16（NV_EFA3, AWS p5en）
  - NVIDIA B200×8（192GB HBM, 160 SMs）+ AWS EFAv4 400G×8（NV_EFA4）
  - NVIDIA H100×8（80GB HBM, 132 SMs）+ ConnectX-7 400G×8 IB（NV_IB, Nebius）
  - NVIDIA GH200×1（96GB HBM, 132 SMs）+ ConnectX-7 200G×1 IB（NV_C2C_IB, Lambda）
  - AMD MI300X×8（192GB HBM, 304 CUs）+ ConnectX-7 400G×8（AMD_CX7, OCI）
  - AMD MI300X×8 + Broadcom Thor-2 400G×8（AMD_BRC, Vultr）
  - 所有平台使用 4 CPU threads per GPU 进行 UCCL-EP 通信代理

- 评估性能的软件/脚本是什么。修改了什么。
  - UCCL-EP 自带的 microbenchmark 套件（dispatch/combine latency bench），使用与 DeepEP 相同的评估方法
  - 修改内容：UCCL-EP 扩展 DeepEP 的 GPU kernel 实现，核心修改包括：(a) **GPU-side kernel migration**：CUDA→ROCm（PTX intrinsics → ROCm alternatives，warp→wavefront，WARP_SIZE 32→64，TMA-based copy→CU-based copy，coordinator wavefronts merged into receiver wavefronts）；(b) **CPU proxy 代码**：multi-threaded lock-free FIFO channel + RDMA work request 构造 + immediate data ordering enforcement + control buffer management；(c) **NIC 适配层**：EFA SRD unordered delivery 的 ordering emulation，Broadcom Thor-2 适配，CX7 IB 栈适配。
  - 开源情况：https://github.com/uccl-project/uccl/tree/main/ep，20.8K 行 C++（含 2.4K 行 CUDA/ROCm C++）和 1K 行 Python。

- 基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - UCCL-EP Microbenchmark 评估原理（以 NV_EFA3 EP=32 HT mode dispatch 为例）：
    1. **benchmark 配置**：EP=32（32 GPUs across 4 nodes × 8 GPUs），token size=7KB（FP8, hidden=7168），HT mode（32 tokens/chunk）
    2. **输入**：GPU 生成 T 个 tokens（varying T=128~4096），每个 token 由 MoE gate 分配 destination experts ranks
    3. **GPU Kernel 执行**（计时起点）：
       - Token dedup：同一节点内多个 experts 的去重逻辑，合并 destinatino 相同的 token
       - Pack to ring buffer：按 (dest_rank, chunk_id) 将 token data 写入对应 ring buffer slot
       - 每个 chunk（32 tokens）填满后，GPU thread 构造 Write TransferCmd（含 dest_rank, ring buffer offset, 7KB×32 length, seq_num），写入 FIFO channel
       - 随后构造 Atomic TransferCmd（increment ring buffer tail），写入同一 FIFO channel（同通道保证 ordering）
    4. **CPU Proxy 执行**：
       - Poll FIFO head → 读取 TransferCmd → 构造 ibv_wr_send WR（含 remote base + offset, length, immediate data = seq_num | expert_idx）
       - 通过 ibv_post_send() 提交到对应 QP（round-robin across multiple QPs per NIC）
       - 多个 NICs per GPU 情况下（如 2×200G EFA），CPU proxy 将请求 distribute 到不同 NIC 的 QP 上
    5. **接收端**：
       - Remote CPU proxy polling CQ → 提取 immediate data → conditional check（所有 prior writes for this channel done?）→ 若通过，apply atomic 更新 ring buffer tail → remote GPU reads tail → 确认 token 到达
    6. **计时终点**：所有 T tokens 被目标 rank 确认接收完成（通过 barrier TransferCmd 同步）
    7. **输出**：dispatch latency = 计时终点 − 计时起点，转换为 dispatch throughput（GB/s）= T × 7KB × 2（send+recv）/ latency
