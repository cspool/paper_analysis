## UCCL-EP Portable Expert-Parallel Communication

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：UCCL-EP 是一个可移植的专家并行（EP）通信系统，作为 DeepEP 的直接替换（drop-in replacement）集成到 SGLang 推理框架和 AMD Primus/Megatron-LM 训练框架中。核心设计是将 GPU-initiated token-level RDMA 通信解耦为 GPU→CPU 控制通道 + CPU→NIC 数据通道：(a) GPU 通过 lock-free FIFO channel 向 CPU proxy 发送 128-bit TransferCmd（Write/Atomics/Drain/Barrier）；(b) 多线程 CPU proxy 解析命令后通过 libibverbs 发出 GPUDirect RDMA 操作；(c) CPU proxy 利用 RDMA immediate data 在接收端模拟各种 delivery semantics（如 write-then-atomic ordering），使 correctness 在不支持 ordering 的 NIC（如 AWS EFA）上也能保证。
  - 实验比较：(a) SGLang v0.5.3 推理吞吐对比：UCCL-EP vs NCCL（因 DeepEP 不能在 EFA 上运行），使用 DeepSeek-R1-0528 和 Qwen3-235B-A22B-FP8 模型，prefill-heavy workload（input length 4096, output length 5），EP=16/32；(b) AMD Primus/Megatron-LM 训练吞吐对比：UCCL-EP vs RCCL，使用 DeepSeek-V3（downscaled 到 32 layers, 379B params），16-node AMD MI300X + Broadcom NICs 平台。

- 硬件平台是什么，配置是什么。
  - NV_EFA3: 4×AWS p5en instances，每节点 NVIDIA H200×8（141GB HBM, 132 SMs, 900 GB/s NVLink），AWS EFAv3 200G×16 NICs，192 CPU cores
  - NV_EFA4: 4 nodes，NVIDIA B200×8（192GB HBM, 160 SMs, 1800 GB/s NVLink），AWS EFAv4 400G×8 NICs，192 CPU cores
  - NV_IB: 4 nodes，NVIDIA H100×8（80GB HBM, 132 SMs, 900 GB/s NVLink），NVIDIA ConnectX-7 400G×8 NICs，128 CPU cores，Nebius 云
  - NV_C2C_IB: 2 nodes，NVIDIA GH200×1（96GB HBM, 132 SMs），NVIDIA ConnectX-7 200G×1 NICs，72 CPU cores，Lambda 云
  - AMD_CX7: 4-16 nodes，AMD MI300X×8（192GB HBM, 304 CUs, 896 GB/s xGMI），NVIDIA ConnectX-7 400G×8 NICs，128 CPU cores，OCI 云
  - AMD_BRC: 4 nodes，AMD MI300X×8，Broadcom Thor-2 400G×8 NICs，128 CPU cores，Vultr 云

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang v0.5.3（推理），AMD Primus/Megatron-LM（训练）
  - 修改内容：UCCL-EP 以 DeepEP API 兼容的方式实现 drop-in replacement，无需修改上层框架代码。UCCL-EP 替换了 DeepEP 的底层通信实现：(a) 移除了 IBGDA（InfiniBand GPUDirect Async）依赖，改为 CPU-proxy-based RDMA；(b) 移除了 NVSHMEM 依赖，用自管理的 symmetric memory + CPU proxy 替代；(c) 添加了对 AWS EFA、Broadcom Thor-2 等异构 NIC 的支持（通过 libibverbs 可移植层）。
  - 开源情况：已开源，https://github.com/uccl-project/uccl/tree/main/ep。实现 20.8K 行 C++（含 2.4K 行 CUDA/ROCm C++）和 1K 行 Python。支持 NVIDIA 和 AMD GPU，以及 NVIDIA CX7、AWS EFA、Broadcom NIC。

- 基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 以 SGLang + UCCL-EP 在 NV_EFA3（AWS p5en, H200×8 + EFAv3）上推理 DeepSeek-R1 为例：
    1. **输入**：用户请求到达 SGLang frontend，tokenizer 将 prompt 转为 token IDs，SGLang scheduler 将请求组 batch。
    2. **MoE Layer 触发**：在 prefill 阶段，每个 MoE layer 的 gating network 在 GPU 上计算每个 token 到 top-K experts 的路由决策。
    3. **GPU 发起 TransferCmd**：UCCL-EP 的 GPU kernel（HT mode）执行 token deduplication（同一节点多专家去重），将需跨节点传输的 token 打包到 ring buffer，GPU threads 通过 PCIe 将 128-bit TransferCmd（含 dest rank、buffer offset、length、sequence number）写入共享 lock-free FIFO channel。
    4. **CPU Proxy 解析并执行 RDMA**：4 个 CPU proxy threads（每个 polling 多个 FIFO channels）从 FIFO head 读取 TransferCmd，通过 libibverbs 构造 RDMA write work request，指定 dest memory region（symmetric memory 的 offset）+ immediate data（embed 32-bit sequence number + expert index），直接写入远程 GPU memory（GPUDirect RDMA）。
    5. **接收端 ordering enforcement**：EFA SRD 协议不保证 delivery ordering。接收端 CPU proxy 从 completion queue 获取 immediate data，check sequence number——若 write 先于其对应的 atomic 到达，则将 atomic 暂存于 control buffer，待所有之前的 writes 被确认后按序 apply atomic（更新 ring buffer head/tail）。
    6. **Combine 阶段**：expert 计算完成后，GPU 再次发起 TransferCmd 将 expert output 送回原 GPU，CPU proxy 执行 hierarchical reduce（intra-node reduce → inter-node RDMA → final reduce）。
    7. **输出**：所有 MoE layers 计算完成后，SGLang 输出 generated tokens。
