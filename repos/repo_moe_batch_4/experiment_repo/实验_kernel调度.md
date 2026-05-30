## Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：(1) **基于 CUB radix sort 的 GPU Token Routing**——将 MoE token-to-expert 路由实现为 GPU-friendly 的 radix sort：对每个 token 的 (expert_scale, expert_idx, row_idx) 三元组按 expert_idx 排序，将同 expert 的 tokens 连续排列，用 CUTLASS Grouped GEMM 并行计算所有 expert 的矩阵乘法。(2) **Fused GEMM+Dequantize CUDA Kernel**——将 weight dequantization（INT4/INT8→FP16）融合进 CUTLASS Grouped GEMM，避免单独 dequantize kernel 的额外内存读写。核心优化：用 FP16 bit-trick 序列（mantissa 直接编码 + 0x6400 构造 + FP16 减法）替代原生 IntToFloat (I2F) 指令。(3) **INT8/INT4 优化的 GPU Dequantize**——INT8：一次加载 4 个 INT8 到 32-bit 寄存器，并行构造 2 个 FP16；INT4：weight layout 重排减少 bit 操作指令量。
  - 实验比较：(a) MoE GEMM 归一化吞吐量：FP16 vs INT8 native I2F vs INT8 optimized I2F vs INT4 optimized I2F，在不同 active experts（1~32）下共 40 tokens（Table 1）；(b) 各种 kernel 组合的端到端 throughput（Table 3）；(c) Batch pruning 的加速效果（1.14×）。

- 后端平台是什么，配置是什么。
  - 单卡 NVIDIA PCIE V100（Volta 架构），CUDA 11.6，nvcc + gcc/g++ 9.3编译
  - 生产部署：单卡 NVIDIA T4（Turing 架构，16GB，INT4 支持，无 NVLink）

- 评估性能的软件/脚本是什么。修改了什么。
  - NVIDIA FasterTransformer 推理框架：扩展以支持 MoE layers（encoder+decoder），使用 CUTLASS Grouped GEMM + CUB radix sort + fused dequantize
  - 修改内容：(a) 实现 MoE token routing kernel——基于 CUB DeviceRadixSort，对每个 token 的 (expert_scale, expert_idx, row_idx) 三元组排序，permute activation matrix 使同 expert tokens 连续；(b) 实现 CUTLASS Grouped GEMM 调用——为每个 expert 构造子矩阵指针（sub-matrix start pointer + weight pointer + bias pointer），单 kernel 并行执行所有 expert matmul；(c) 实现 Fused GEMM+Dequantize——在 CUTLASS GEMM kernel 的 weight load 阶段 fused dequantize，替换原生 I2F 为 FP16 bit-trick 序列；(d) 实现 batch pruning——在 decoder token routing 中跟踪 active_tokens 计数。
  - 开源情况：核心组件 FasterTransformer、CUTLASS、CUB 均为开源；论文自身未提供独立开源仓库。

- 基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 以单次 MoE Decoder Layer 的 expert computation 为例，评估原理和全过程：
    1. **输入**：decoder 生成的 hidden states H，形状 (B, S, 1024)，其中 B=batch_size=32, S=beam_size（beam search 当前步的有效 token 数），FP16。
    2. **Gating/Routing**：Router（FP16 GEMM + softmax）为每个 token 生成 (expert_scale, expert_idx) 元组（top-1 gating）。若某句子已完成（生成了 EOS），Batch Pruning 将 expert_idx 设为 INT_MAX。
    3. **Token Routing Kernel**（CUB radix sort）：
       - 对每个 token 构造三元组 (expert_scale, expert_idx, row_idx)
       - CUB DeviceRadixSort::SortPairs 按 expert_idx 排序 → sorted order + permutation indices
       - Gather kernel：按 permutation indices 从 H 中 gather rows → H_permuted，同 expert tokens 连续排列
       - Offsets kernel：扫描排序后的 expert_idx 数组，计算每个 expert 的 (start_offset, num_tokens) → expert_ptr
       - 仅前 active_tokens 行参与计算
    4. **Fused GEMM+Dequantize Kernel**（CUTLASS Grouped GEMM）：
       - 输入：H_permuted 的子矩阵（各 expert 的 token batch, FP16）+ 各 expert 的 INT4/INT8 weights (1024×4096) + FP16 scales + FP16 biases
       - Weight load 阶段 fused dequantize：4 个 INT8 → 1 个 32-bit reg → 构造 `0x6400 | val`（FP16 1024+val）→ FP16 减 1152 → 乘 scale → FP16 dequantized weight
       - 标准 FP16 GEMM 计算（使用 Tensor Cores 若可用）
       - 所有 experts 的 GEMM 在单个 kernel launch 中并行执行（CUTLASS Grouped GEMM）
    5. **Un-permute + Scale**：按逆排列将输出 rows 恢复原始顺序，每行乘以对应 expert_scale
    6. **输出**：MoE layer output H_out，形状 (B, S, 1024)，FP16
    7. **评估原理**：Throughput 测量翻译 1000 tokenized English sentences（约 40K tokens）的总时间，计算每秒处理 input tokens 数。GEMM 级别的性能评估通过固定 total tokens=40（解码阶段典型值），测量不同 active experts 数下的 GEMM 延迟，归一化为 FP16 baseline 的倍数（Table 1）。

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

## Tutel Adaptive Mixture-of-Experts at Scale

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：(1) **Fast Encode/Decode GPU Kernels（K0, K1, K2）**——将 MoE dispatch 的 encode（生成 All-to-All 输入）和 combine 的 decode（从 All-to-All 输出恢复 MoE 层输出）从稠密 einsum（O(T·E·C_g·D)）替换为稀疏 SIMT-efficient 实现（O(T·k·D)）。基于 CUDA warp-level programming：每个 warp 处理单个 token 沿 M 维度的计算，利用 warp shuffling、Blelloch scan、half2 向量化等优化。(2) **Flexible All-to-All**——将 All-to-All 输出 layout 从 (W, E_g, C_g, D) 优化为 (E_g, C, D)，消除 C_g 对 world size W 的依赖，保证任意规模下 expert matmul 的输入 shape 一致。(3) **2DH (Two-Dimensional Hierarchical) All-to-All**——利用 stride memory copy 对齐非连续 chunks，将小消息聚合为大消息后再进行节点间 All-to-All，避免小消息导致的 InfiniBand 带宽利用不足。支持 MSCCL DSL 编译优化和 LL128 协议。(4) **自适应多流流水线调度**——将 token 沿 capacity 维度分区，在独立 CUDA stream 上异步执行 dispatch All-to-All → Expert FFN → combine All-to-All，实现通信与计算重叠。
  - 实验比较：(a) Kernel computation breakdown: TUTEL (Fast Encode/Decode) vs Fairseq/DeepSpeed MoE 各阶段的 kernel 耗时对比（Figure 15）；(b) Flexible A2A vs A2A layout 的 expert computation throughput（Figure 11）；(c) 2DH vs Linear All-to-All latency 对比（1MiB~256MiB, 64~4096 GPUs），使用 nccl-tests alltoall_perf benchmark（Figures 18, 19）；(d) GPU 内存节省：TUTEL (Fast Encode/Decode) vs Fairseq MoE，20%~90% 节省（Table 5/9）；(e) 自适应流水线在不同 capacity factor f 下的提升（Figure 13）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 SXM 80GB GPUs（Azure ND96amsr_A100_v4），节点内 NVLink 3.0 + NVSwitch，节点间 HDR InfiniBand (200 Gbps × 8 = 1,600 Gbps non-blocking)。NCCL 2.10.3-1 + NCCL RDMA SHARP plugin。最大 2,048 GPUs。

- 评估性能的软件/脚本是什么。修改了什么。
  - NCCL 2.10.3-1 + nccl-tests（alltoall_perf benchmark）：用于评估 2DH vs Linear All-to-All 延迟。修改：在 nccl-tests 中实现了 2DH All-to-All 算法（Algorithm 2）替代原生 Linear All-to-All。
  - MSCCL (Microsoft Collective Communication Language)：用 DSL 描述 2DH 算法并通过编译器优化生成 NCCL kernel，支持 LL128 协议。
  - PyTorch 1.8.0 + Fairseq moe branch（baseline）+ TUTEL MoE 集成。
  - 修改内容：(1) Fast Encode/Decode：CUDA kernel 实现 K0/K1/K2（Figure 21），替代 Fairseq 中的 einsum dense 实现；(2) Flexible All-to-All：自定义 layout 变换替代标准 NCCL All-to-All 的 (W, E_g, C_g, D) → (E_g, C, D)；(3) 2DH All-to-All：4-phase stride-memcpy + 节点内/节点间 All-to-All；(4) Adaptive Pipelining：token capacity 维度拆分 + 多 CUDA stream 异步调度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/microsoft/tutel，集成于 Fairseq 和 DeepSpeed。
  - Fast Encode Kernel（K0/K1/K2）输入到输出全过程：
    1. **输入**：moe_input (T, M)，logits (T, E) — T 个 token 的隐藏特征和 gate logits
    2. **K0 (Gate Processing)**：Softmax(logits) → gate_probs；TopK(gate_probs) → idxs (T, k), scores (T, k)；compute_location(idxs) → locations (T, k)。每个 warp 处理一个 token，warp-level 并行。
    3. **K1 (Sparse Encode/Decode)**：对每个 token t，执行 `dispatch_input[idxs[t]][locations[t]] = bool(scores[t]) * moe_input[t]`。使用 warp shuffling + Blelloch scan 实现稀疏写入，复杂度从 O(T·E·C_g·D) 降至 O(T·k·D)。
    4. **K2 (All-to-All 输入/输出变换)**：Flexible All-to-All layout 变换 (E, C_g, D) → (E_g, C, D)，inline 完成无需额外 memory copy。
    5. **输出**：dispatch_input (E, C_g, M) → All-to-All dispatch → Expert FFN (matmul) → All-to-All combine → decode → moe_output (T, M)。
  - 2DH All-to-All 评估原理：
    1. **输入**：S bytes 数据，分布于 n GPUs（m GPUs/node, n/m nodes）
    2. **Phase 1 (Stride Memcpy)**：重排节点内 chunks 使同一本地目标 GPU 的数据连续
    3. **Phase 2 (Intra-node A2A)**：节点内 m GPUs 交换 S/m 大小的 chunks
    4. **Phase 3 (Stride Memcpy)**：重排使同一远程目标节点的数据连续
    5. **Phase 4 (Inter-node A2A)**：节点间 n/m nodes 交换合并后的大 chunks（S/n × m）
    6. **输出**：All-to-All 完成后的重排数据。通过 nccl-tests alltoall_perf 测量端到端延迟，对比 Linear（Algorithm 1）vs 2DH（Algorithm 2）。

## TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：开发 **Triton kernel** 高效实现 Expert Group Approximation 的计算逻辑。核心 kernel 为 "Router backward" kernel，用于计算传递给路由器的近似稠密梯度。此外还有用于构造专家分组近似和聚合的 Triton kernel。这些 kernel 负责在 GPU 上高效执行 $N^2$ 个专家组近似的构造、token 分组、以及梯度聚合操作。
  - 实验比较：(a) Throughput 对比——不同 hidden dim（1024/2048/4096）下 Expert Group Approx. vs TopK vs Sparsemixer 的 tokens/sec 和 overhead 百分比（Table 4）；(b) CUDA 时间开销 scaling——随 hidden size 增大，各 kernel 组件的 CUDA 时间占比变化（Figure 9），overhead 从 1024-dim 的 13.32% 降至 4096-dim 的 1.57%。

- 后端平台是什么，配置是什么。
  - NVIDIA GPU（具体型号论文未明确说明）。单 GPU 用于 throughput 测量和可复现性分析。

- 评估性能的软件/脚本是什么。修改了什么。
  - GPT-NeoX 训练框架 + Megablocks 稀疏训练库。修改内容：(1) 在 MoE 层的反向传播中插入 Expert Group Approximation 计算；(2) 用自定义 Triton kernel 替代原生 PyTorch 操作实现 token 分组、近似构造和梯度注入；(3) 添加跨数据并行 worker 的 all-reduce 通信以聚合近似梯度。
  - 评估指标：tokens/sec、TFLOPS、CUDA time breakdown。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源状态：论文声明开源但未提供具体链接（匿名投稿）。基于 GPT-NeoX + Megablocks 开源框架实现。
  - Kernel 输入到性能输出全过程：
    1. **输入**：每个 MoE 层的 token 嵌入 x（shape: [batch_size, seq_len, d_token]）、路由器权重 W、专家参数 E_1..E_N
    2. **Triton "Router backward" kernel**：对所有 $\binom{N}{K}$ 个路由决策分组，在每组内计算 $\hat{E}_i(x)$ 近似，聚合到稠密梯度向量 ∂y/∂π
    3. **All-reduce 通信**：跨数据并行 workers 聚合近似梯度，利用大批量样本估计稠密梯度
    4. **梯度注入**：通过 stop-gradient 机制将近似梯度注入计算图，更新路由器和专家权重
    5. **输出**：更新后的路由器权重 W 和专家参数 E_1..E_N，CUDA time profiling 记录各 kernel 耗时分布<br>
  - Overhead 分析原理：通过记录 MoE 层总的 CUDA 时间中 expert MLP matmul、Router backward kernel 等各自占比，展示随 hidden size 增大（1024→2048→4096），matmul 时间占比增大使得方法 overhead 占比下降。

## Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Sem-MoE 中为 Attention-TP 场景实现的两个融合通信原语（Triton kernel）：(1) **Shuffled-Reduce-Scatter (SRS) kernel**——将 speculative token shuffling 嵌入标准 reduce-scatter 操作，在 ring-based 通信过程中同时完成 token 的预测性重排。核心步骤：查询 token-to-device table T 和 inter-layer expert-sequence-to-device table A → 比较置信度选择预测 → argsort 计算 shuffle indices → 按 indices 重排 token 排列 → reduce-scatter 分发各 device 的 token 分片；(2) **Shuffled-AllGather (SAG) kernel**——MoE 计算完成后，allgather 收集各 device 的 token 分片 → 依据保存的 shuffle indices 进行反向 argsort 恢复原始 token 顺序；(3) **优化 argsort kernel**——自定义 Triton 实现的 argsort，比 PyTorch 原生实现快 25%，是 shuffle 操作的核心性能关键路径；(4) **DeepEP 集成**——在 Sem-MoE 中集成 DeepEP 作为高效 all-to-all 通信后端。shuffling 逻辑嵌入 ring-based communication 的额外开销约 1%。
  - 实验比较：(a) 单 MoE layer 延迟——不同 LAR（Local Activation Rate）下 expert layer latency，Sem-MoE 将 LAR 从 25% 提升至 62%/68%，对应 41.8%/46.6% latency reduction；(b) Attention-TP 端到端 TTFT——不同 input length（256/512/1024）下 vs SGLang/MoETuner；(c) SRS/SAG overhead 测量——shuffling 逻辑 overhead 约 1%，argsort kernel 比 PyTorch 原生快 25%。

- 后端平台是什么，配置是什么。
  - GPU：8-GPU server（96GB HBM/GPU，>400GB/s 专用互联）
  - CPU：2× 44-core Intel CPU，2TB DDR5
  - 软件栈：Triton（OpenAI）+ PyTorch + DeepEP + NCCL/HCCL
  - 评估模型：DeepSeek-V2-Lite（64 routed experts/layer）、Qwen3-30B-A3B（128 experts/layer）、Moonlight-16B

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 SGLang 的 MoE 推理流程。修改内容：(1) 新增 SRS kernel（Triton）——实现 token shuffling + reduce-scatter 融合通信；(2) 新增 SAG kernel（Triton）——实现 allgather + token 顺序恢复；(3) 自定义 argsort Triton kernel——比 PyTorch 原生快 25%；(4) 集成 DeepEP 通信库——替换标准 NCCL all-to-all；(5) mock routing 模块（用于 LAR vs latency 的假设性分析）——通过跳过通信延迟来模拟不同 LAR 下的理想性能上限。
  - 评估方法：通过 CUDA event timing 测量单 MoE layer 及 end-to-end 推理各阶段延迟；通过 mock routing 跳过通信来构建不同 LAR 下的性能参考线。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文基于 SGLang（开源 https://github.com/sgl-project/sglang）构建，Sem-MoE 的自研 kernel 代码（~5000 行 Python + Triton）未提供独立开源仓库链接。
  - Kernel 执行全过程（Attention-TP, SRS + MoE + SAG pipeline）：
    ```
    输入：Attention TP 输出 hidden states X ∈ R^{B×H}

    [SRS Kernel — 融合 Shuffle + Reduce-Scatter]
    Step 1: 查询 scheduling tables
      - 对 batch 中每个 token_id j，查询 T[j] → predicted device d_tok
      - 查询 A[(d_prev_layer1, d_prev_layer2)] → predicted device d_seq
      - 选 C_p 较高的预测源 → final device_ids list D ∈ R^B
    Step 2: Triton argsort kernel
      - input: D ∈ [0, E-1]^B  (per-token target device)
      - 按 device_id 排序 → shuffle_indices ∈ R^B
      - 比 PyTorch 原生快 25%（论文实测）
    Step 3: Token shuffling
      - X_shuffled = X[shuffle_indices]  (GPU tensor indexing)
      - 重排使同一目标 device 的 token 连续排列
    Step 4: Reduce-Scatter
      - ring-based: 每 GPU 获得自己负责的 token 分片 X_local
      - shuffling 逻辑 overhead ≈ 1%（嵌入在 ring communication schedule 中）

    [MoE Computing]
    Step 5: Gate function + Expert FFN
      - 各 GPU 对本地 token 分片 X_local 执行 gate + selected expert computation
      - 由于 SRS 已将 token 预 shuffle 到 expert 所在 device
      - 大部分 expert 计算在本地完成（高 LAR）

    [SAG Kernel — 融合 AllGather + 顺序恢复]
    Step 6: AllGather
      - 收集各 GPU 的 expert 输出，恢复完整 batch Y_shuffled
    Step 7: 反向 argsort
      - reverse_indices = argsort(shuffle_indices)  （恢复原始顺序）
      - Y = Y_shuffled[reverse_indices]
    Step 8: 输出恢复后的 hidden states 进入下一 layer

    性能输出：
    - Per-layer expert latency: Normal w/ all-to-all vs SRS pipeline
    - LAR = (#tokens computed locally) / (#total tokens)
    - Overhead: shuffling ≈ 1% of ring communication time
    - EP communication reduction: 41.8% (DeepSeek) / 46.6% (Qwen3) expert layer latency
    ```

## SonicMoE Accelerating MoE with IO and Tile-aware Optimizations

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SonicMoE 是基于 CuTe-DSL 编写的 GPU kernel 库，面向 NVIDIA Hopper (H100) 和 Blackwell (B300) GPU，为细粒度、高稀疏 MoE 训练提供 8 个高性能 kernel：(1) **Forward：Up-proj A kernel**——Gather + varlen-M Grouped GEMM + SwiGLU，将 token gather 操作与 GMEM-to-SMEM load 融合（cp.async），SwiGLU 融合到 GEMM epilogue；(2) **Forward：Down-proj Y kernel**——varlen-M Grouped GEMM + 异步 TMA store，使用 Ping-Pong scheduling（Hopper）或 TMEM 两阶段流水线（Blackwell）将 MMA 与 heavy epilogue IO 重叠；(3) **Forward：Expert aggregation O kernel**——每个 token gather-and-sum 所有激活 expert 的输出，基于 TMA gather 实现高带宽；(4) **Backward：dH kernel**——Gather dO + varlen-M Grouped GEMM + dSwiGLU + dS计算 + A'输出，将 Gather fusion、epilogue fusion（dH/dS/A'同一 kernel）和异步 TMA load of H 全部融合；(5) **Backward：dW2 kernel**——Gather dO + varlen-K Grouped GEMM；(6) **Backward：dX~ kernel**——varlen-M Grouped GEMM + 异步 TMA store；(7) **Backward：dW1 kernel**——Gather X + varlen-K Grouped GEMM；(8) **Backward：dX kernel**——每个 token gather-and-sum 所有 expert 的 dX~。关键设计：(a) **Gather Fusion**：在 forward 和 backward 的所有需要 gather 的地方均将 gather 与 GMEM-to-SMEM load 融合，消除 X_e 和 dO_e 的显存物化；(b) **Epilogue Fusion**：SwiGLU/dSwiGLU/dS 均融合在 GEMM epilogue 中，dS 使用 dS=⟨dA',A⟩ 路径（而非 ⟨dO,Y⟩），避免缓存 Y 和额外 HBM 访问；(c) **Ping-Pong Scheduling**（Hopper）：2 consumer warpgroups 交替执行 MMA 和 epilogue/IO，实现 MMA 与 IO 重叠；(d) **TMEM 两阶段流水线**（Blackwell）：利用 UMMA 的单线程异步特性和 TMEM 的 2-stage 结构，MMA warp 与 epilogue warps 并发操作不同 TMEM stage；(e) **Token Rounding Routing**：将 per-expert token 数舍入到 GEMM tile size (M_tile=128) 的倍数，消除 Grouped GEMM padding 带来的浪费 FLOPs。
  - 实验比较：(a) **Kernel 级 TFLOPS**：SonicMoE vs ScatterMoE/MoMoE/MegaBlocks/Megatron/DeepGEMM++/DeepGEMM-pt，在 H100 和 B300 上测 forward+backward 的 TFLOPS 和 memory bandwidth；(b) **Activation Memory**：各方法 per-layer peak activation memory（1.4B~120B MoE 配置），SonicMoE 减少 45%（7B, n=256 vs ScatterMoE）；(c) **端到端训练吞吐**：7B MoE (n=256) FSDP-2 训练，SonicMoE 64 H100s = 213B tokens/day vs ScatterMoE 96 H100s = 225B tokens/day；(d) **Grouped GEMM 基础性能**：contiguously-packed inputs 和 gathered inputs 的 varlen-M/K Grouped GEMM TFLOPS vs DeepGEMM/cuBLAS；(e) **Token Rounding 训练吞吐**：TR vs TC top-K 在稀疏 MoE (E=128/256) 下的 TFLOPS 对比；(f) **Top-K sorting kernel**：SonicMoE bitonic sort top-K vs PyTorch/Triton/Tilelang/RTop-K 的带宽对比；(g) **Expert aggregation kernel**：gather-and-sum vs scatter-and-sum 策略的带宽对比；(h) **Ablation**：Gather fusion 有无、Ping-Pong scheduling 有无、TMA store vs st.global scatter 对 TFLOPS 的影响。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA H100 (Hopper, SM90) 80GB 和 NVIDIA B300 (Blackwell, SM100)。H100 使用 CUDA toolkit v12.9；B300 使用 CUDA toolkit v13.0。
  - 多节点训练：64× H100（8 nodes × 8 GPUs）使用 FSDP-2 + ZeRO-3 单节点内 shard，节点间复制。
  - 软件框架：CuTe-DSL（CUTLASS 的 C++ template DSL）编写 kernel，PyTorch 接口。基于 lm-engine 代码库进行端到端训练。

- 评估性能的软件/脚本是什么。修改了什么。
  - SonicMoE 开源：https://github.com/Dao-AILab/sonic-moe（permissive license）
  - 基础框架：CUTLASS CuTe-DSL + PyTorch。对比 baseline：ScatterMoE (Triton)、MoMoE (Triton)、MegaBlocks (block-sparse)、Megatron-LM GroupedMLP (CUTLASS Grouped GEMM)、DeepGEMM (SM90/SM100 BF16 Grouped GEMM)
  - 修改/新增内容：(1) 实现 8 个独立 MoE kernel（forward A/Y/O + backward dH/dW2/dX~/dW1/dX），使用 CuTe-DSL 的 warp-specialized kernel 设计；(2) 实现 Gather fusion with cp.async 在 varlen-M 和 varlen-K Grouped GEMM 中；(3) 实现 Ping-Pong scheduling（Hopper）和 TMEM-based overlapping（Blackwell）；(4) 实现 top-K bitonic sorting kernel（支持 E≤4096, K≤16）；(5) 实现 Token Rounding routing 算法；(6) 实现高效 expert aggregation kernel（TMA gather-and-sum）
  - 快速使用：`pip install sonic-moe`，提供 PyTorch nn.Module 接口直接替换 MoE 层

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/Dao-AILab/sonic-moe，论文 ICLR'26（https://openreview.net/pdf?id=KzTJ1raEgB）
  - 评估原理：kernel 级 benchmark 使用 CUDA event 计时测量每个 kernel 的 wall-clock time，计算 TFLOPS = (6+12)dn(Σf_e) / time（model FLOPs 而非 hardware FLOPs）。Memory bandwidth = total IO bytes / time。端到端训练吞吐使用 lm-engine 代码库测量 tokens/day。
  - Kernel 执行全过程（以 dH kernel 为例，H100，7B MoE, n=256）：
    ```
    输入: dO ∈ R^{T×d}（上游梯度）, W_2 ∈ R^{E×n×d}（down-proj weights）, S（router scores）, π（routing mask）, H（cached pre-activation）
    
    [Prologue - TMA + cp.async]
    Step 1: Producer warpgroup 启动 TMA load W_2 到 SMEM pipeline stage
    Step 2: Consumer warpgroup 0 启动 WGMMA，accumulate 到 RF
    Step 3: Producer warpgroup 启动 cp.async + gather 加载 dO（按 π 索引 gather）
    
    [Mainloop - Ping-Pong]
    Step 4: Consumer warpgroup 0 继续 WGMMA over K dim
    Step 5: Consumer warpgroup 1 启动 WGMMA，利用另一组 accumulator registers
    Step 6: Producer warpgroup 交替为两个 consumer 提供下一 tile 数据
    // Consumer 0 和 1 交替执行 MMA 和 epilogue
    
    [Epilogue - Consumer warpgroup 0]
    Step 7: MMA 完成，dA'_e = dO_e W_{2,e}^T 结果在 RF 中
    Step 8: 异步 TMA load H_e 从 HBM 到 SMEM（dedicated pipeline）
    Step 9: 从 SMEM 读取 s_e = Gather(S, π_{:,e}) → dA_e = Broadcast(s_e) * dA'_e
    Step 10: dSwiGLU(dA_e, H_e) → dH_e, A_e（forward activation recompute）
    Step 11: A'_e = Broadcast(s_e) * A_e（write to HBM via TMA store, 用于 dW2）
    Step 12: dS_{e,t} = ⟨dA'_{e,t}, A_{e,t}⟩（reduce over n dim）
    Step 13: TMA store dH_e, dS, A'_e to HBM
    
    // Consumer warpgroup 1 同时执行下一个 tile 的 MMA
    
    输出: dH ∈ R^{TK×2n}（up-proj activation grad）, dS ∈ R^{T×E}（router score grad）, A' ∈ R^{TK×n}（输入给 dW2 kernel）
    ```
  - 8 个 kernel 的工作流：Forward: X → [A kernel: Gather+GEMM+SwiGLU] → H,A → [Y kernel: GEMM+TMA store] → Y → [O kernel: Gather-and-sum] → O。Backward: dO → [dH kernel: Gather+GEMM+dSwiGLU+dS+A'] → dH,dS,A' → [dW2 kernel: Gather+GEMM] → dW2；dH → [dX~ kernel: GEMM+TMA store] → dX~ → [dW1 kernel: Gather+GEMM] → dW1；dX~ → [dX kernel: Gather-and-sum] → dX。

## SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：SmartMoE 是面向 MoE 模型分布式训练的两阶段自动并行化系统，核心运行时调度包含：(1) **Offline Pool Construction（离线池构建）**：基于 workload-aware 性能模型，在训练前搜索最优混合并行策略组合（Data + Tensor + Pipeline + Expert Parallelism），以及同构策略组合中仅 expert placement 可变的执行计划池（pool）。Pool 中所有候选执行计划有相同 expert slot 配置，保证运行时切换开销只涉及参数交换（无内存分配/释放）；(2) **Online Adaptive Parallelization（在线自适应并行化）**：在训练运行时，根据 gating network 实际输出的 per-expert token 负载，周期性（默认每 10 iteration）执行三种轻量级 expert placement 搜索算法——Greedy（O(NE) 复杂度）、Dynamic Programming（O(N×4^E) 最优解）、Hybrid（Greedy+DP 组合，两阶段：先将 E 个 expert 用贪心分配到 M 个虚拟设备，再用 DP 将 M 个虚拟设备分配到 N 个物理设备，总复杂度 O(ME + N×4^M)，M 可调）；(3) **切换开销控制**：设置切换阈值过滤掉性能提升有限的 placement plan，利用 expert selection 的时间局部性（相邻 iteration 分布变化小）降低搜索频率，只在搜索前少数 iteration 收集历史统计避免无用开销。
  - 实验比较：(a) 端到端训练加速比——3 个 GPU 集群上 GPT-MoE（NLP）和 Swin-MoE（CV）模型，SmartMoE vs FasterMoE baseline 最高 1.88× speedup，平均 1.53×（inky A100 cluster）、1.17×（pinky V100 SXM cluster）、1.14×（blinky V100 PCIe cluster）；(b) 离线并行化消融——vs Alpa 推荐的 data-insensitive 并行方案，data-sensitive 方案达到 2.67× speedup（vs 2.36×）；(c) 在线并行化消融——16 层 MoE 在 64 V100 上 per-layer 平均 1.16× 加速，最高 1.43×；(d) 性能模型精度——R² > 0.5 for all configurations；(e) Overhead 分析——搜索 <1ms（1024 expert 时 <50ms），切换 ~20ms，而 Alpa 搜索需 825s。

- 后端平台是什么，配置是什么。
  - Cluster 1 (blinky)：8× NVIDIA V100 PCIe per node，max 32 GPUs，50Gb/s InfiniBand
  - Cluster 2 (pinky)：4× NVIDIA V100 SXM per node，max 64 GPUs，100Gb/s InfiniBand
  - Cluster 3 (inky)：8× NVIDIA A100 SXM per node，max 32 GPUs，200Gb/s InfiniBand
  - 软件：PyTorch（基于 FastMoE 框架），支持集成 Megatron-LM 和 DeepSpeed

- 评估性能的软件/脚本是什么。修改了什么。
  - 基础框架：FastMoE（Tsinghua 自研 MoE 训练框架，https://github.com/laekov/fastmoe）
  - SmartMoE 代码（https://github.com/zms1999/SmartMoE），Artifact Evaluation repo（https://github.com/MachineLearningSystem/23ATC-SmartMoE-AE）
  - 修改内容：(1) 新增 expert slot 抽象——支持任意组合的 DP/TP/PP/EP 混合并行策略表达；(2) 新增 workload-aware 性能模型——基于 gating network 语义（capacity factor / topology-aware gate）估算 per-expert 负载上界，无需实际训练数据；(3) 新增 offline pool search——对候选池穷举搜索，使用性能模型预测最优池；(4) 新增 online 轻量级搜索算法——Greedy（O(NE)）+ DP（最优）+ Hybrid（两阶段可调复杂度）的 expert placement 搜索；(5) 新增 runtime 策略切换机制——包括切换开销阈值控制和搜索频率自适应；(6) 新增 MoE gating history 采集与同步模块
  - 快速复现（无需 GPU）：`./RUNME-a.sh`（处理预录 trace 数据生成图表，约 2 分钟）；深度复现：需 16× V100，约 2 小时

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/zms1999/SmartMoE（主代码），Artifact https://github.com/MachineLearningSystem/23ATC-SmartMoE-AE（完整 AE）
  - 安装：基于 FastMoE，`cd src/fastmoe && USE_NCCL=1 python setup.py install --user`
  - 评估原理：测量端到端训练 latency（forward + backward + gradient sync + optimizer 全流程），使用真实数据集（OpenWebText for GPT-MoE，ImageNet for Swin-MoE）而非随机输入，确保 MoE 动态负载真实。Micro-benchmark 仅测量 MoE 层 forward/backward 时间。使用综合 expert selection 数据集（真实训练过程收集的不同模型结构和 gating 方法）
  - 运行时专家放置调度全过程（以 4 GPUs, 16 experts, GShard gate为例）：
    ```
    输入：token batch 分布在 4 GPUs（Expert Parallelism）
    
    [Offline - 训练前，一次性执行]
    Step 1 - 枚举混合并行候选池:
             遍历 DP/TP/PP/EP 组合 + expert slot 配置
             如：DP=2×TP=2, expert slots per GPU=4
    Step 2 - Workload-Aware 性能预测:
             基于 capacity factor=2.4 估算上界:
               max_tokens_per_expert = (capacity_factor × batch_tokens) / num_experts
             基于 topology-aware gate 估算通信量:
               intra-node expert 分配优先 → 估算 cross-node all-to-all 量
    Step 3 - 穷举搜索最优池:
             对候选池按预测性能排序 → 选最优池
             输出：固定 DP/TP/PP 策略，保留 expert placement 可变
    
    [Online - 每 iteration，周期性执行]
    Step 4 - Gate Network Forward (GPU):
             x → Linear(W_gate) → Top-K softmax → per-token expert indices
             
    Step 5 - Expert Selection History (CPU):
             收集 per-GPU 的 {expert_i: token_count_i}
             通过 All-Gather 聚合到 CPU scheduler
             
    Step 6 - Light-weight Expert Placement Search (<1ms):
             输入: C[16] = {E0:512t, E1:480t, E2:501t, ..., E15:490t}, N=4 GPUs
             
             方案A - Greedy (O(16×4)=64 ops):
               Sort experts by C_i descending: E5(520), E0(512), E3(508), ...
               For each expert in sorted order:
                 Pick GPU with min(samples[j]) AND experts[j] < E/N=4
                 Place expert on that GPU, update samples[j] and experts[j]
               结果: GPU_0={E5(520),E7(240),E12(245),E1(240)} → load=1245
                     GPU_1={E0(512),E9(250),E15(248),E2(242)} → load=1252
                     GPU_2={E3(508),E11(252),E6(249),E10(238)} → load=1247
                     GPU_3={E4(505),E13(248),E14(247),E8(245)} → load=1245
                     Imbalance = (1252-1245)/1245 = 0.56% (极低)
             
             方案B - Hybrid (Greedy + DP):
               先 Greedy: 16 experts → M=4 virtual devices（per node）
               再 DP: 4 virtual devices → N=4 physical devices（per node内最优）
               M=8（per GPU in node）时 → DP state: 2^8=256, 复杂度 O(4×4^8)
               
    Step 7 - 切换决策:
             计算 Δ = (当前plan延迟 - 新plan延迟) / 当前plan延迟
             若 Δ > threshold → 执行切换
             若 Δ ≤ threshold → 保持当前plan（避免微小改进引入通信开销）
             
    Step 8 - Expert Parameter Exchange (如触发切换):
             比较新旧 placement，确定需移动的 expert 参数
             All-to-All 交换参数（~20ms for 16 experts on 16×V100）
             无内存分配/释放（slot 配置不变）
             
    Step 9 - Expert FFN Computation:
             各 GPU 按新 placement 计算分配的 expert tokens
             与前 iteration 相比：优化后的 placement 消除负载热点
    
    性能输出：
    ├─ Per-iteration latency (ms): forward+backward total
    ├─ Speedup = T_baseline / T_SmartMoE
    ├─ 搜索开销 <1ms（搜索频率每10 iterations）
    ├─ 切换开销 ~20ms（被后续 iterations 的性能提升摊销）
    └─ End-to-end elapsed time with dynamic plan switching
    ```
  - 关键技术点：(1) Expert slot 是核心抽象——通过 capacity/#slots/#layers 三个属性统一表示所有并行策略，使得各种混合策略可互相比对和转换；(2) Workload-aware 性能模型的关键是利用 gating 超参数（capacity factor / topology-aware constraints）估算负载上界，而不是依赖实际运行时统计；(3) Pool 设计的核心洞察——固定 hybrid 策略、可变 expert placement 是在"优化空间"和"切换成本"之间的最佳平衡点。

## Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ES-MoE的运行时调度核心包含三个紧密协作的组件：(1) **Pipelined expert processing调度**：在token permutation（GPU间all-to-all通信）阶段异步启动第1个expert的CPU→GPU上传（通过PCIe 4.0），后续experts顺序处理时，前一个expert的GPU kernel（FFN forward/backward）与下一个expert的DMA传输重叠，形成compute ↔ upload流水线；(2) **Dynamic Expert Placement算法**：将n个expert分配到k个GPU的最小化makespan问题（strong NP-hard），采用greedy approximation（Graham 1969, 4/3-approximation）。算法按expert处理时间（max(upload_time, compute_time_token_count)）降序排序，逐个分配expert到累积处理时间最低的GPU，复杂度O(m*log n + m*log m)，实际运行时间<2.69us；(3) **Expert-wise CPU optimization调度**：每个expert完成backward后立即触发CPU Adam optimizer更新（与ZeRO-Offload的delayed update不同，无staleness），expert granularity细粒度使得靠近output的layer优化被靠近input的layer的GPU backward计算隐藏。
  - 实验比较：(a) 训练吞吐量(Tokens/s或words/s) vs Zero-Offload^E/FairSeq/Tutel；(b) GPU间token负载不平衡度：ES-MoE动态placement vs FairSeq静态placement下的token分配方差；(c) GPU utilization（effective %）vs Zero-Offload^E；(d) 最大可训练expert数量；(e) microbatch size对吞吐量影响

- 后端平台是什么，配置是什么。
  - GPU：4× NVIDIA A100 40GB
  - CPU：AMD EPYC 7543 32核
  - CPU内存：512 GiB DDR4
  - GPU-GPU通信：NVLink 600 GB/s（token all-to-all交换）
  - CPU-GPU通信：PCIe 4.0（expert参数上传/下载）
  - SSD：4TB，用于扩展offloading策略

- 评估性能的软件/脚本是什么。修改了什么。
  - 基础框架：Fairseq（Meta的序列建模训练框架）
  - CPU Optimizer：DeepSpeed的CPU Adam
  - 修改内容：共3.3k行Python + 3.0k行C++，核心修改包括：
    - 实现expert-wise offloading管理器（CPU/SSD expert参数管理、LRU eviction、prefetch）
    - 实现pipelined expert processing调度器（token permutation与expert upload的overlap控制、compute-communication流水线同步）
    - 实现dynamic expert placement算法（greedy scheduling，CPU侧运行）
    - 实现expert-wise CPU optimizer（与backward流式衔接）
  - 对比baseline：修改ZeRO-Offload支持expert-wise offloading得到Zero-Offload^E变体

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/kaist-ina/es-moe
  - 评估原理：测量训练过程中各框架的wall-clock throughput（words/s或tokens/s），在同一硬件配置下控制microbatch size为各框架的最佳值，使用gradient accumulation维持per-device batch size=32的统一逻辑batch size
  - 运行时调度全过程（以MoE-L 16 experts, 4 GPUs, 1个MoE layer为例）：
    ```
    输入：batch tokens (microbatch_size, seq_len, d_model=1536)
    
    GPU侧：
    ├─ [Kernel] Gating Network Forward: 
    │   tokens → Linear(d_model, 16) → softmax → expert_ids per token
    │   输出：{token_idx: expert_id} 映射
    │
    ├─ [CPU调度] Dynamic Expert Placement (CPU, <2.69us):
    │   统计 per-expert token count → 按降序排序 →
    │   Greedy assign → GPU_0: {E3(511 t), E7(502 t), E12(498 t), E1(495 t)}
    │                       GPU_1: {E5(509 t), E9(503 t), E15(497 t), E0(490 t)}
    │                       GPU_2: {E2(507 t), E11(504 t), E6(499 t), E10(493 t)}
    │                       GPU_3: {E8(510 t), E13(501 t), E14(496 t), E4(492 t)}
    │   结果：各GPU负载方差<15%（vs Fairseq的102%）
    │
    ├─ [调度+DMA] 各GPU异步上传第1个expert（CPU→GPU PCIe 4.0）+
    ├─ [Kernel] Token Permutation: all-to-all交换tokens到目标GPU（NVLink）
    │   重叠：permutation ≈ 上传E3耗时 → E3到达时perm接近完成
    │
    ├─ [PIPELINE] 每个GPU的4个experts顺序计算:
    │   GPU_0:
    │   ├─ [Kernel] Expert E3 Forward (FFN: gate_proj→SiLU→×up_proj→down_proj)
    │   │   同时：DMA上传E7参数到GPU
    │   ├─ [Kernel] Expert E7 Forward   同时：DMA上传E12
    │   ├─ [Kernel] Expert E12 Forward  同时：DMA上传E1
    │   └─ [Kernel] Expert E1 Forward
    │   GPU_1/2/3 同理并行
    │
    ├─ [Kernel] Token Un-permutation: inverse all-to-all + weighted sum
    │
    ├─ [GPU→CPU] Expert梯度下载
    │
    └─ [CPU调度] Expert-wise CPU Optimization（streaming）:
        专家完成backward即触发：
        ├─ Expert E3 gradient → CPU Adam Update(E3.params, E3.opt_states)
        │   同时：GPU继续处理其他layers的forward
        ├─ Expert E7 gradient → CPU Adam Update(...)
        └─ ... (叠加在前一layer的GPU backward上)
    
    性能输出：
    ├─ Throughput: words/s（挂钟时间÷处理词数）
    ├─ GPU utilization: GPU active time / total iteration time
    └─ Offloading overhead: (upload+download+optimizer time) / compute time
    ```

## Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：基于CUDA的**双端稀疏-稀疏矩阵乘法专用kernel**，利用NVIDIA Sparse Tensor Core（SpTC）的`mma.sp` PTX指令实现稀疏计算加速。核心kernel设计：(1) **双端稀疏数据格式**：权重端采用(N,M,V)格式——M×V块内保留N个Sub-Row，每个Sub-Row内2:4 element-wise稀疏；激活端采用vector-wise稀疏通过SEL选择数组记录routing结果。(2) **3-step hierarchical tiling**：step0为thread block tile（$m_b \times n_b$），step1为warp tile（$m_w \times n_w$），step2为SpTC指令tile（$m_i \times n_i$），K维度$K_b$受V约束需较小以避免精度损失。(3) **Data stationary优化**：引入中间寄存器$C_{IR}$，每$\frac{V}{k_b}$次迭代将C寄存器按indices矩阵shuffle，避免频繁global memory读写。(4) **Packing策略**：矩阵A按SpTC spec通过ldmatrix加载；矩阵B以转置形式packing，支持行内连续访问和跳过零值行；metadata矩阵采用自定义2-bit→32-bit映射packing，对齐32-bit memory transaction。(5) **Pipeline机制**：使用cp.async非阻塞拷贝实现fetch阶段和compute阶段重叠。kernel编译为动态库（NVCC），通过pybind11注册为Python模块。
  - 实验比较：(a) kernel级：238个合成尺寸（m,k,n ∈ [256,16384]）上对比cuBLAS、Sputnik、cuSPARSELt、VENOM的TFLOPs；(b) 真实模型benchmark：6种MoE模型配置的kernel吞吐量；(c) 不同dimension（m/k/n）独立scale时的吞吐量趋势；(d) break-down分析：逐步开启weight sparsity→input sparsity→layout optimization→data stationary的加速效果；(e) 与编译器方案PIT对比MoE层speedup。

- 后端平台是什么，配置是什么。
  - 主要GPU：NVIDIA GeForce RTX 4070 Super（Ada Lovelace，含SpTC及async copy/ldmatrix支持）
  - CPU：Intel i7-12700，16G×2 DDR5
  - OS：Ubuntu 22.04LTS，CUDA 12.1
  - 可移植性验证GPU：NVIDIA 3090、4090、A100 40G
  - 理论兼容：AMD MI300（CDNA3，有sparse ALU但缺async copy和collective load/store native支持）
  - kernel实现语言：CUDA + PTX inline assembly（mma.sp instruction）

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本（来自artifact appendix）：
    - `./artifacts/kernel/synthetic_scripts.sh`：运行238种合成尺寸的kernel性能测试（对应Figure 12/13）
    - `./artifacts/kernel/kernel_model_config_scripts.sh`：运行真实模型配置的kernel测试
    - `./artifacts/MoE/figure14_scripts.sh`：MoE层性能测试
    - `./artifacts/model/figure15_scripts.sh` 和 `figure16_scripts.sh`：端到端模型测试
    - `./artifacts/MoE/figure17_scripts.sh`：breakdown分析
  - 修改内容：Samoyeds kernel替代了标准MoE执行流程中的GEMM操作。在vLLM/Transformers框架中，MoE层的线性投影（gate_proj, up_proj, down_proj）由Samoyeds sparse-sparse kernel替代，同时集成input permutation消除和weighted accumulation fusion。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/guqiqi/Samoyeds.git，Docker: kevinwu2017/samoyeds:1.0.0
  - 评估原理：每个kernel调用测量GPU wall-clock时间（通过CUDA event timing），计算TFLOPs = 有效FLOP（仅非零元素计算量）/ 执行时间。合成benchmark覆盖m,k,n各维度从256到16384的238种组合。
  - Kernel输入→性能输出全过程：
    ```
    输入：编码后的权重（data + indices + metadata矩阵）+ 稀疏输入矩阵B + SEL选择数组
    
    Kernel执行流程（Algorithm 1）：
    1. Init阶段：分配shared memory（A_tile, Indices, B_tile, SEL）和register（metadata, C）
    2. 加载SEL：GMEM → SMEM
    3. Pipeline loop (compute=0 to k/k_b):
       a) 加载metadata：GMEM → Register（跳过innermost tiling，直接到寄存器）
       b) Fetch阶段（异步）：
          - cp.async: 加载Indices, A_tile, B_tile: GMEM → SMEM
          - 3-step tiling：thread block tile → warp tile → SpTC tile
          - commit group for pipeline
       c) Compute阶段：
          - wait group（同步）
          - ldmatrix: SMEM → Register（按SpTC spec排列）
          - 若compute % (V/k_h) == 0: shuffle C寄存器（data stationary）
          - mma.sp: 触发SpTC执行稀疏MMA（M=16,N=8,K=32或M=16,N=8,K=16）
       d) 两步overlap（pipeline机制）
    4. 输出transposition（layout优化）：Register → GMEM（压缩格式，仅输出非零行）
    
    性能输出：TFLOPs = (2 * m * k * n * sparsity_ratio) / elapsed_time
    ```
  - Docker使用：`docker pull kevinwu2017/samoyeds:1.0.0 && docker run -it --gpus all --name samoyeds-ae kevinwu2017/samoyeds:1.0.0`，进入容器后执行上述脚本，结果用配套Jupyter notebook绘图（figureXX_plot.ipynb）。

## SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现：基于SN40L streaming dataflow的自动算子融合与kernel调度。编译器自动将PyTorch级别算子图编译为空间融合dataflow kernel，将20+ operators融合到单个kernel launch（参见Figure 11，llama70B-4k-inf-prefill的fused/unfused kernel call比率约11×）。两种kernel调度模式对比：(1) Software Orchestrated (SO) — host CPU通过AGCUs发出Program Load→Argument Load→Kernel Execute命令序列调度kernel；(2) Hardware Orchestrated (HO) — AGCUs内置硬件kernel调度器，将静态kernel schedule offload到硬件，消除host往返延迟。
  实验比较：在8-socket SN40L Node上，对Table III所列benchmark（Llama2-7B/70B、sparseGPT-13B、Bloom-176B、Mistral-7B、Falcon-40B、LLaVA1.5-7B、FlashFFTConv）对比三种配置：(a) Unfused — 每个PyTorch operator作为独立kernel执行，中间结果materialize到HBM/DDR；(b) Fused+SO — 编译器自动融合+host软件调度；(c) Fused+HO — 编译器自动融合+AGCUs硬件调度。测量speedup（Figure 10）和kernel call数量比（Figure 11）。

- 后端平台是什么，配置是什么。
  SN40L RDU（8 socket Node for大多数benchmark，单socket for FlashFFTConv）：638 BF16 TFLOPS/socket，1040 PCU + 1040 PMU，SRAM 520 MiB，HBM 64 GiB/1.8 TB/s，DDR 1.5 TiB/200 GB/s。额外16 socket用于Llama 3.1推理benchmark。

- 评估性能的软件/脚本是什么。修改了什么。
  SambaNova自研编译器（非开源）— 接收PyTorch/Python级别模型描述，自动将计算图编译为PCU/PMU/AGCU/RDN上的空间融合dataflow kernel。编译器核心修改/功能包括：(1) 静态符号生命周期分析实现garbage collection — 将非重叠生命周期的逻辑符号映射到相同设备虚拟地址；(2) 符号temporal locality分析 + 带宽估计 — 决定哪些符号溢出到DDR（优先溢出总传输带宽最小的符号）；(3) 静态带宽建模 — 建模RDN/TLN上的并发数据流带宽需求，指导PCU/PMU资源分配；(4) Place-and-Route层 — 配置RDN routing table、flow ID、multicast路径。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未开源。Kernel调度评估原理和执行流程：以FlashFFTConv benchmark为例，原始PyTorch实现包含多个GEMM、element-wise Mul、Transpose等算子的Monarch FFT decomposition。Unfused配置下，每个算子编译为独立kernel — 数据流：HBM→AGCU→RDN→PCU执行GEMM→RDN→AGCU→materialize到HBM→AGCU→RDN→PCU执行Mul→RDN→AGCU→HBM...每步都产生HBM读写。Fused配置下，编译器将整段Monarch FFT编译为单个kernel：HBM通过AGCUs流式加载→PCU(systolic GEMM)→PMU(stage buffer I0)→PCU(SIMD Mul)→PMU(transpose via data alignment unit, T0*→T1*)→PCU(systolic GEMM)→AGCUs写回HBM。操作强度从39.5 Ops/Byte提升至410.4 Ops/Byte，FlashFFTConv实现13× speedup。性能测量：在硬件上运行warmup + timed iterations，使用SN40L switch和PMU内置performance counter监控RDN拥塞和bank冲突。

## PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PuzzleMoE 设计了一个自定义 CUDA GEMV kernel 用于 on-the-fly 解码 bit-packed 权重。核心逻辑：(1) Bit-packed 存储——利用 Bfloat16 中 3 个 underutilized exponent bits 嵌入 binary mask 和 sign bit，压缩后的 W_merged 直接以标准 Bfloat16 格式存储，无需额外 metadata 存储；(2) On-the-fly Decoding GEMV Kernel——每个 weight W[i,j] 在矩阵乘法使用前即时从 packed 格式解码，解码操作（bit shift + mask + exponent 恢复）在 kernel 的 data-loading path 上 piggyback，利用 warp-level scheduling 和 coalesced memory access 实现零额外延迟；(3) 消除 decoded matrix 的 materialization——不在内存中创建独立的解码后权重矩阵，避免额外内存分配和访存开销。
  - 实验比较：(1) 推理加速——Mixtral-8x7B 50% sparsity 下 1.28× speedup，Qwen3-MoE 50% sparsity 下 1.19× speedup（vs full model on same GPU count）；(2) 内存节省——Mixtral-8x7B 从需要 2×A100-80GB 降至 1×A100-80GB，Qwen3-MoE 从 2×A100-40GB 降至 1×A100-40GB；(3) 压缩时间——Mixtral-8x7B 仅 2 分钟，Deepseek-MoE（64 experts）仅 10 分钟；D2 需 55 分钟（因 SVD），NAEE 对 Deepseek-MoE 需 10^18 次 forward pass 不可行。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100-80GB (Mixtral-8x7B) 和 A100-40GB (Qwen3-MoE)。CUDA kernel 基于 Bfloat16 计算路径，kernel 融合 decoding + GEMV，prefill length=1024, decode length=512。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 CUDA GEMV kernel。修改内容：(1) 新增 bit-packed weight 的 on-the-fly decoding 逻辑（Algorithm 1: mask_bit 提取 → 零值判断 → sign_bit 提取 → exponent 恢复 → Bfloat16 重构）；(2) 将 decoding 逻辑嵌入 GEMV data-loading path，利用 warp 级并行和 coalesced memory access；(3) 标准 Bfloat16 格式兼容——packed 后的数据仍可被 PyTorch 作为 Bfloat16 tensor 加载，仅在内核执行时通过自定义 kernel 进行解码。
  - 推理评估：基于论文自研推理框架（含自定义 CUDA kernel），对比 full model 和压缩后模型的 latency 和 memory usage。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/Supercomputing-System-AI-Lab/PuzzleMoE
  - 评估原理：自定义 GEMV kernel 加载 bit-packed Bfloat16 tensor（输入 X ∈ R^{B×d}，packed W ∈ R^{d×h} 含嵌入式 mask 和 sign），在 CUDA thread block 中每个 warp 处理一行 d 维的输入向量。具体流程：
    1. **Kernel Launch**：每个 MoE expert FFN 层的 weight 矩阵以 packed Bfloat16 格式存储在 GPU global memory。输入 activation X 从上一层的 FP16/Bfloat16 tensor 传入。
    2. **Data Load**：每个 warp 从 global memory 加载 packed W[i,j]（16 bits）进入寄存器。W[i,j] 的 bit layout：[15: sign_packed][14:13: mask bits for expert 0/1][12:7: shifted exponent (5 bits)][6:0: mantissa (7 bits)]。
    3. **On-the-fly Decode**（见 Algorithm 1）：
       - mask_bit ← (W ≫ (13 - expert_pos)) & 1
       - 若 mask_bit = 0 → 该 weight 对当前 expert 无效，W_decoded = 0
       - 否则 sign_bit ← (W ≫ (15 - expert_pos)) & 1
       - exp ← (W & 0x0F80) + (112 ≪ 7) 恢复原始 exponent
       - W_decoded ← (sign_bit ≪ 15) | exp | (W & 0x007F) → Bfloat16
    4. **FMA Compute**：W_decoded 作为 Bfloat16 值直接参与 FMA (Fused Multiply-Add) 计算 Y[p] += X[p,k] × W_decoded[k,j]。
    5. **性能输出**：end-to-end inference latency（prefill + decode phases）、GPU memory usage（通过 nvidia-smi 或 PyTorch memory stats 测量）、speedup ratio（latency_compressed / latency_full）。
  - 关键技术点：Decoding 在 data-loading path 上与 warp-level memory access 高度融合——解码的 bit ops 远小于 global memory read 延迟，因此解码开销被访存延迟隐藏。Expert 选择通过 gate network 的标准 Top-K routing 完成，gate 计算不涉及 packed weight——gate weights 保持原始精度。

## QMoE Sub-1-Bit Compression of Trillion-Parameter Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QMoE 设计了一个自定义 CUDA kernel（Sub1MatVec）用于 fused decompression + matrix-vector product，将压缩存储的三元权重以字典解码方式 on-the-fly 转换为可计算值。核心设计要点：(1) **Warp-per-Row 并行**——每个 warp (32 threads) 处理权重矩阵的一行，每行独立编码，使用 28/32 threads 进行解码和乘加累加；(2) **Dictionary-Based Decoding**——2^16 个 UINT16 codewords 映射到最多 14 对三元权重（28 weights），字典 512KB 存储于 GPU L2 cache，高频 codeword 通过概率排序实现 L1 cache prefetch；(3) **Shared Memory Dequant Table**——三元值 {0, 1, 2} 通过复制 32× 的 shared memory lookup table deq[3][32×num_warps] 转换为 {0, w_min, w_max}，避免 bank conflict；(4) **Coalesced Memory Access**——每次取 32 个 UINT16 codewords 到 shared memory（单次 coalesced transaction），输入向量 x 预加载到 shared memory 实现快速连续读取；(5) **Ternary 解码优化**——每权重 2-bit 存储于 UINT32 中，通过 shift + mask 提取（无 modulo/division 等慢速操作），线程 0-13 处理前半权重、14-27 处理后半。
  - 实验比较：(1) Per-layer kernel 性能——Sub1MatVec vs PyTorch bfloat16 cuBLAS GEMV（各 MoE 层矩阵形状），A6000 和 RTX 3090 上 compressed kernel 在全部情况下比 uncompressed baseline 更快（最高 35% speedup）；(2) End-to-end 推理——压缩后的 c2048 在 4×A6000 和 8×3090 上的 HuggingFace 全流程 runtime，与理想化 uncompressed baseline（同一专家数据复用，估计值）对比，<5% 额外延迟开销。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A6000 (48GB) 和 NVIDIA RTX 3090 (24GB)。Per-layer kernel 评估：单 GPU 上各类 MoE 矩阵形状。Compressed 推理：4×A6000 或 8×3090（单服务器）。c2048 模型若不压缩需 >65 A6000 / >130 3090 GPU，因此 uncompressed baseline 用时通过"所有 expert 指向同一权重数据"的估计方式获得（下界估计，实际需更多 GPU 及通信开销）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 PyTorch + HuggingFace Transformers 的自研 CUDA kernel（Listing 1: Sub1MatVec）。修改内容：(1) 新增 CUDA kernel 实现 on-the-fly 字典解码 + dequant + matvec 融合计算；(2) One threadblock per SM，每 warp 处理一行，超过 32 行时 warp 串行处理多行；(3) HuggingFace 推理框架的 MoE 层调用被替换为压缩 kernel；(4) HuggingFace 中空 CUDA kernel launch 的 bugfix（跳过无 token 分配的 expert 调用，>10× 加速大模型推理）。评估指标：per-layer latency (ms)、end-to-end latency per token (ms)、speedup ratio。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/ISTDASLab/qmoe。CUDA kernel 源码（含完整边界条件处理）见官方仓库。
  - Kernel 评估原理与执行流程：
    1. **Kernel Launch Configuration**：每个 threadblock 处理一个 weight matrix block（含多行），每 warp 处理一行。`num_warps = min(rows_in_block, 32)`，若 rows > 32 则部分 warp 串行处理多行。1 threadblock per SM 避免 wave quantization 效应。
    2. **Input Preparation**：压缩权重 w_comp (UINT16 array, 每行独立编码的 codeword 序列)、row_off (每行 codeword 偏移索引)、dec (UINT32[2^16 * 2] 字典表，512KB)、ter_minmax (每行的 {w_min, w_max} dequant 参数)、x (bfloat16 input vector)。
    3. **Shared Memory 初始化** (lines 7-17)：
       - 所有 warp 协作：将 x 向量加载到 `x_shared[w_width]`（bf16→float 转换）。
       - 每 warp 独立：构建 dequant lookup table `deq[3][32*num_warps]`。`deq[0]=0, deq[1]=w_min, deq[2]=w_max`，复制 32× 在列方向避免 bank conflict。
    4. **Per-Row Decoding Loop** (lines 22-33)：
       - (a) Coalesced load: `w_comp_block[warp][lane] = w_comp[i + lane]` — 32 threads 联合加载 32 个 UINT16 codewords。
       - (b) 仅 lanes 0-27 (28 threads) 参与解码：遍历 32 个 codewords，每个 codeword 去字典查表 `dec[2*enc + (lane/14)]` → 线程 0-13 取第一个 UINT32、线程 14-27 取第二个。
       - (c) Ternary 提取：`ter = (wx14 >> (4 + 2*(lane%14))) & 0x3` — 每 weight 仅需 shift + mask（硬件友好），无 modulo/division。
       - (d) Dequant + FMA：`res += deq[ter][thread] * x_shared[idx + lane]` — 连续 shared memory 读（无 bank conflict）。
       - (e) 偏移更新：`idx += 2 * (wx14 & 0xf)` — pair_count 存于低 4 bits。
    5. **Warp Reduction** (lines 37-38)：对 28 threads 的部分积进行 warp shuffle 求和。
    6. **Output**：`y[row] = float2bfloat16(res)` 写入全局内存。
    7. **性能原理**：字典 512KB 适合 GPU L2 cache（A6000 L2=6MB, 3090 L2=6MB），高频 codeword 因概率排序被 L1 自动 prefetch。Compressed kernel 读取更少的 global memory（<1 bit/param vs 16 bits/param），虽增加 bit unpacking 计算，但 global memory latency（~200 cycles）远大于 bit ops（~1 cycle），净效果为加速。

## PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PROBE 的 kernel 调度涉及多类运行时计算 kernel：(1) 单SM CUDA kernel 实现的 Greedy Balance-Optimal Planning Solver——串行迭代更新，hard cap kmax=16 次迭代，运行在单个 SM 上与 MoE Compute 重叠；(2) 自定义 Triton kernel 实现的 remote P2P put 操作——受控 SM occupancy 下发 expert weights 到目标 rank 的 replicated-expert buffer；(3) NVSHMEM-based 全局 All-Gather——聚合 per-rank 预测结果；(4) Lookahead Predictor MLP inference kernel——轻量级前向推理；(5) Phase-Locked Co-Scheduling 的分裂相位传输——在 MoE Compute 期间启动 P2P 传输，All-to-All Combine 前暂停以释放带宽，Combine 后恢复直到下一层 Attention 完成。
  - 实验比较：与 SGLang（无冗余 expert）和 DeepSeek-EPLB（统计式 2 冗余 slot）对比。在 GPU timeline 微操作级别通过 Figure 10 展示 Predict/Plan/Prefetch/Update 各阶段如何被完全隐藏在 Dispatch/MoE Compute/Attention 等关键阶段之后。比较指标：IR 从 2.13→1.09，Max/Avg 计算延迟比从 2.27→1.18，Combine 阶段的同步等待消除。
- 后端平台是什么，配置是什么。
  - 8×NVIDIA H800-141GB，NVSwitch 900 GB/s。CUDA 12.9、PyTorch 2.9、NCCL 2.27.3、NVSHMEM 3.3.20。DeepEP (normal mode) 作为 All-to-All 通信后端。
- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 SGLang 框架的端到端推理评估。在 SGLang 基础上集成了 DeepEP 通信库、NVSHMEM symmetric memory 管理、自定义 CUDA/Triton kernel。评估了：(a) Prefill latency (TTFT) scaling——不同总输入 token 数（16K-64K GPT-OSS / 32K-128K Qwen3-MoE）；(b) Decoding throughput-latency Pareto——per-rank batch 512-1536，前 500 decoding steps；(c) Robustness stress test——Code→Chinese 突然语义切换；(d) Predictor 精度——per-layer Top-K Accuracy、Top-Half-K Hit-Rate、2×Top-K Recall；(e) Timeline breakdown——NVIDIA Nsight / 自定义 profiling 工具捕获各阶段的微操作延迟。
- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明开源。评估原理：在 SGLang 推理引擎中，每个 decoding step 的 MoE 层执行流程为——(1) Predict phase: Lookahead Predictor MLP 读取上一层的 hidden state tensor [B, H]，输出预测 logits [B, num_experts]，通过 All-Gather 聚合全局预测 n̂；(2) Plan phase: 单SM Solver 以预测 n̂ 和当前 placement P' 为输入，在 ≤16 次迭代内输出 Δ_r^{in}/Δ_r^{out}（需复制/驱逐的 expert 集合）和 token assignment A；(3) Prefetch phase: Triton P2P kernel 以 Δ_r^{in} expert indices 为输入，从源 rank 读取 |Δ| × W bytes 的权重 tensor，写入目标 rank 的 NVSHMEM buffer；(4) Execute phase: 下一层按更新后的 A 进行 token dispatch 和 Grouped GEMM 计算。性能输出：通过 GPU timeline 验证 Predict(MLP+All-Gather) ⊂ Dispatch latency、Plan(≤16 iter single-SM) ⊂ MoE Compute、Prefetch(split-phase) ⊂ MoE Compute + Attention 的全重叠，确保零关键路径开销。

## PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PopFetcher 的 kernel 调度/运行时计算涉及多层调度：(1) **Sliding-Window Popularity Prediction**——在每次 forward pass 中，routing information collector 以滑动窗口（s=10 次迭代）收集 gate network 的路由选择数据，通过 expert 层间相关性条件概率 Pr(E^{h,j+1}|E^{i,j}) 预测下一层 expert 流行度，该计算异步运行在 CPU 上；(2) **Hybrid Push-Pull Expert Prefetching**——基于 popularity 预测，asynchronous scheduling executor 在非 MoE 计算（Attention 层）期间，通过 C++/CUDA 实现的 expert 预取逻辑，利用 torch.cuda.Stream 管理的独立 prefetch stream，从 remote GPU 异步拉取热门 expert 参数到本地 GPU memory；(3) **Stream Pipelining in Backward Pass**——将 All-to-All 和 All-Reduce 通信分解为 micro-operations，以流水线方式交错执行，优先让 All-to-All 数据流抢占 GPU 资源，避免 backward computation blockage；(4) **Custom MoE Operator**——通过 torch.autograd.Function 自定义 MoE operator 的 forward 和 backward 行为，封装所有计算、通信和预取活动；(5) **Internal Expert Sharing via CPU Memory**——server 级 cache manager 利用 CPU memory 在节点内 GPU 间共享已预取的 remote expert 参数，避免冗余传输。
  - 实验比较：与 DeepSpeed、FasterMoE、Megablocks、Tutel、Janus 五个 baseline 系统对比。在 GPU 集群上进行端到端训练评估，指标为 token throughput 和 per-iteration time。Ablation study 中对比了 popularity-based vs random expert prefetching（1.30× MoE-GPT, 1.26× MoE-BERT）和 pipelined stream scheduling vs built-in stream strategy（减少 10.9% MoE-GPT, 10% MoE-BERT 迭代时间）。还评估了 hybrid push-pull vs pure expert-centric vs pure data-centric 范式。

- 后端平台是什么，配置是什么。
  - 双 GPU 集群：(1) Cluster A——2 台机器 × 4 NVIDIA RTX 4090 24GB，100Gbps Mellanox ConnectX-5 InfiniBand；(2) Cluster B——8 台机器 × 4 NVIDIA A10 24GB，共 32 GPU，节点间最大 32Gbps 带宽。软件栈：PyTorch 2.3、CUDA 12.4、NCCL 通信后端。支持集成到 Megatron-LM 框架。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 PyTorch 自研的 MoE 训练框架（8000+ 行 Python/C++/CUDA），以 PyTorch plugin 形式实现，可独立运行或集成到 Megatron-LM。修改内容：(a) 新增 routing information collector（Python，通过 All-Gather 聚合 per-token gate 选择数据）；(b) 新增 prefetching decision-maker（CPU 异步分析 expert popularity 并求解最优 prefetch 方案）；(c) 新增 asynchronous scheduling executor（C++/CUDA，管理 expert 预取的 prefetch stream 和 backward pass 的通信流优先级调度）；(d) 修改 MoE operator——用 torch.autograd.Function 自定义 forward/backward，封装所有预取和调度逻辑；(e) 修改 backward pass 的 CUDA stream 管理——将 All-to-All 和 All-Reduce 通信分解为 micro-operations 并交错执行。评估模型：MoE-GPT（decoder-only）、MoE-BERT（encoder-only），在 OpenWebText、PILE（~600GB）、OSCAR-2201 数据集上训练。配置组合包括 12/24 layers、16/32 batch size、16/32/64 experts、8/32 workers。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明开源链接，搜索未找到公开 GitHub repository。评估原理：基于 PyTorch 自建的 MoE 训练系统，完整流程为：(1) **每 iteration 的 forward pass**——Input tokens 进入 MoE layer → gate network（GShard 或 naive top-k）计算 routing → routing information collector 通过 All-Gather 聚合各 worker 的路由选择数据 → prefetching decision-maker 在 CPU 上异步执行 sliding-window popularity 预测（s=10 iterations），利用公式 p(E^{h,j+1}) = Σ Pr(E^{h,j+1}|E^{i,j}) p_seq^{i,j} 预测下一层热门 expert → 结合 GPU memory 约束（Eq. 10）和 transfer time 约束（Eq. 9），调用 pruning 策略筛选 candidate experts（至多 k×N 个）→ 求解 min max Lat_w^{prefetch} 优化问题，确定 δ_{n,w}^i（各 worker 应预取哪些 remote expert）；(2) **Prefetch execution**——asynchronous scheduling executor 在当前 MoE layer 的非 MoE 计算（Attention 层）期间，通过独立 prefetch stream 从 remote GPU 拉取 δ_{n,w}^i 标记的 expert 参数（优先通过 NVLink 节点内链路，次选 inter-node GDR NIC），若 expert 已在同节点 CPU memory 则直接本地共享；(3) **下一 MoE layer 计算**——token 对已预取到本地的 expert 直接本地计算（无需 All-to-All dispatch），未预取的 expert 仍需通过标准 All-to-All 发送 token；(4) **Backward pass**——prefetched expert 的 gradient 需 All-Reduce 回 primary expert → 将 All-to-All（token 回传）和 All-Reduce（gradient 聚合）分解为 micro-operations 交错流水线执行，All-to-All 优先级高于 All-Reduce。性能输出：per-iteration time（forward+backward 总时间）、token throughput（每秒处理 token 数）、All-to-All token transfer volume 减少比例（~14-15%）、GPU workload balance（token 分布差异减少 43-57%）。

## Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：All-to-All 集合通信在 N-D torus 网络上的算法与调度优化。在无故障场景下提出 **HalfRing 算法**（利用双向链路实现最短路径单跳 store-and-forward 数据传输）和 **DimRotation 调度**（将数据分块并轮转每块的维度通信顺序，实现无气泡的全带宽利用）。在故障场景下提出 **FoldedRing 算法**（在故障链路上构建折叠环以维持容错通信）和 **MATE/MATEe 调度**（利用其他维度的链路加速故障环上的数据传输）。
  - 实验比较：
    - 无故障：Ring+Pipeline（baseline）vs HalfRing+Pipeline、Ring+DimRotation、HalfRing+DimRotation
    - 有故障：Ring+Pipeline（fault-free baseline）vs FoldedRing+Pipeline、FoldedRing+DimRotation、MATE、MATEe
    - 与 Google TPUv4 的 DOR（Dimension-Order Routing）和 WFR（Wild-First Routing）对比
    - 指标：性能加速比、All-to-All 带宽、维度利用率、可扩展性、端到端训练/推理时间分解、非均匀 All-to-All 性能、多故障弹性

- 后端平台是什么，配置是什么。
  - 模拟平台：ASTRA-SIM 模拟器（analytical backend + GARNET cycle-accurate backend）
  - 拓扑：2D/3D/4D torus（合成实验），4×4×4 TPUv4 pod（单pod），8×4×4 TPUv4 pod（双pod），TPUv3 8×8，TPUv4 8×8×8（实际工作负载）
  - 链路带宽：32 GB/s（合成实验），56 GB/s（TPUv4），82 GB/s（TPUv3）
  - 网络延迟：100 ns
  - 真实机器：16×Ascend 910B4 NPU（2节点，每节点8设备），节点内高带宽链路，节点间 200Gb/NPU RoCE

- 评估性能的软件/脚本是什么。修改了什么。
  - 软件：ASTRA-SIM 模拟器（https://github.com/astra-sim/astra-sim），GARNET 网络模拟器
  - 修改内容：在 ASTRA-SIM 中实现了 HalfRing、FoldedRing 算法及 DimRotation、MATE/MATEe 调度的 collective communication 策略；在 GARNET backend 中实现了 DOR（Dimension-Order Routing）和 WFR（Wild-First Routing）作为对比 baseline，并加入 dateline 死锁避免机制
  - 真实机器：PyTorch Distributed 模块（torch.distributed），在 Ascend NPU 上模拟 4×4 torus 拓扑（禁用节点内互联，限制通信到特定设备对）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：是。Zenodo DOI: https://doi.org/10.5281/zenodo.16735313；GitHub: https://github.com/redbird-arch/micro2025-torus-ft-all2all-artifact
  - 使用方式：
    1. **分析后端实验**：`cd analytical_backend/ && conda env create -f astra-sim-analytical.yml && conda activate astra-sim-analytical && ./build/astra_analytical/build.sh -c`，然后运行 `cd examples/scripts/ && bash run-all.sh`
    2. **GARNET 后端实验**：`cd garnet_backend/ && conda env create -f astra-sim-garnet.yml && conda activate astra-sim-garnet && bash setup_protobuf.sh && ./build/astra_garnet/build.sh -c`，然后运行 `bash run-all.sh`
    3. **真实机器实验**：`cd real_machine/ && bash Run_All_to_All.sh`
  - 评估原理：ASTRA-SIM 接收计算工作负载描述（DLRM/MoE 模型的 layer 定义与并行策略）和网络拓扑配置，在 analytical backend 下使用线性成本模型（启动时间 α + 传输时间 S/B）直接计算通信时间；在 GARNET backend 下进行 cycle-accurate 网络模拟，逐 flit/packet 模拟路由、链路分配和拥塞。最终输出 All-to-All 完成时间、带宽、维度利用率等指标。输入为系统配置（拓扑、带宽、延迟）和通信数据量，输出为 PDF 图表（Fig 11-19）。

## Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ES-MoE 在 Fairseq 框架上实现 expert 参数 offload 到 CPU 内存/SSD，并通过三项核心运行时调度机制提升 MoE 训练效率：(1) **Pipelined Expert Processing**：expert 级别的流水线——在 token permutation 阶段上传第一个 expert（重叠 permutation 延迟），后续 experts 串行处理时并发上传与计算（expert I/O 与 expert FFN 重叠）；(2) **Dynamic Expert Placement**：基于 per-batch gating network 输出的 token 分布，使用贪心近似调度算法（Graham 1969, 4/3-approximation）动态将 n 个 experts 分配到 k 个 GPUs，使各 GPU 的聚合负载均衡，消除 zero-padding（复杂度 O(m*log n + m*log m)，CPU 执行 < 2.69μs）；(3) **Expert-wise CPU Optimization**：将 CPU Adam optimizer 从 layer-wise 改为 expert-wise 粒度——每个 expert 完成 backward pass 后立即启动 CPU 端参数更新，与后续 layers 的 GPU 计算重叠；(4) **Adaptive Offloading**：根据 expert 数量与 GPU 内存比值自动选择 GPU-only / CPU offload / CPU+SSD offload 模式，expert pinning 将 top 25% 热门 expert 固定在 GPU 上。
  - 实验比较：(a) 训练吞吐量对比（words/s）：ES-MoE vs Zero-Offload^E / FairSeq GShard / Tutel，覆盖 MoE-S/M/L 模型 × 8/16/32/64 experts（Table 1）；(b) 可扩展性：各框架最大支持 expert 数量（Figure 4）；(c) 微 batch size 对吞吐量影响（Figure 5）；(d) Component-wise 分析：pipelined expert processing 带来的 GPU 利用率提升（+61.1%）、dynamic expert placement 的 token 负载均衡效果（102% → 15% 差异）、adaptive offloading 模式切换（Figure 7）；(e) Ablation study：逐一去除 expert pinning / optimizer overlapping / larger batch size / zero-padding elimination 对吞吐量的影响（Table 3）；(f) Fine-tuning：Fairseq-MoE-15B 在 4 GPUs 上用 SST-2/MNLI/BoolQ 数据集微调 6.5 小时（Table 2）；(g) Pretraining 端到端对比（Table 5）；(h) GPU 利用率分析（Table 6）。

- 后端平台是什么，配置是什么。
  - GPU：4× NVIDIA A100 40GB（PCIe 4.0 for CPU-GPU, NVLink 600 GB/s for GPU-GPU）
  - CPU：AMD EPYC 7543 32-core
  - CPU Memory：512 GiB DDR4
  - Storage：SSD（实验中最高配 4 TB 用于 SSD offloading 模式）
  - Software：Fairseq framework（基于 PyTorch），DeepSpeed CPU Adam optimizer
  - 实现代码：3.3k lines Python + 3.0k lines C++

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 Fairseq 训练框架。修改内容：(1) 新增 expert offload 模块——将 expert 参数和 optimizer states 从 GPU 迁移到 CPU pinned memory 和 SSD；(2) 新增 pipelined expert scheduler——在 MoE block forward pass 中，token permutation 与首个 expert upload 重叠，后续 expert upload 与 FFN 计算重叠；(3) 新增 dynamic expert placement 模块——CPU 端贪心调度算法，per-batch 决定 expert→GPU 映射；(4) 新增 expert-wise CPU optimizer——override Fairseq 的 layer-wise optimizer，改为 per-expert 触发 CPU Adam step；(5) 新增 adaptive offloading 控制器——运行时决定使用 GPU-only / CPU-offload / CPU+SSD-offload 模式。
  - 评估脚本评估原理：每个 training iteration 测量 wall-clock time，计算 training throughput (words/s) = batch_size × sequence_length / iteration_time。GPU 利用率通过 PyTorch profiler 测量。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/kaist-ina/es-moe
  - 评估原理：在 Fairseq 框架中，每个 training iteration 的 MoE layer 执行流程如下：
    ```
    输入：Input tokens [B, S, H] 分布在 k 个 GPUs 上（Expert Parallelism）
    
    Forward Pass per MoE Block:
    1. Gating Network (GPU): x → Linear(W_g) → softmax → Top-1 expert index per token
    2. Dynamic Expert Placement (CPU, <2.69μs):
       - 收集 per-expert token counts from all GPUs
       - Greedy scheduling: sort experts by (upload_time + compute_time)
       - Assign each expert to GPU with minimum accumulated load
       - 输出：expert→GPU 映射表
    3. Token Permutation (GPU): All-to-All scatter tokens to target GPUs
       【同时：异步上传第一个 expert 权重 CPU→GPU via PCIe】
    4. Expert Processing Loop (per GPU):
       for expert in assigned_experts:
         a) 若 expert 不在 GPU: 异步上传 expert weights CPU→GPU
         b) Expert FFN: gate_proj(x) → SiLU ⊙ up_proj(x) → down_proj (与上传重叠)
         c) 输出 intermediate activations
    5. Token Un-permutation (GPU): All-to-All gather expert outputs back
    
    Backward Pass:
    6. Expert FFN backward (GPU): 计算 expert weight gradients
    7. Expert-wise CPU Optimizer:
       - 每个 expert backward 完成后立即触发:
         a) 下载 gradients GPU→CPU
         b) CPU Adam step: m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²
            w = w - lr * m̂ / (√v̂ + ε)
       - Expert N 的 CPU optimizer 与 Expert N+1 的 GPU backward 重叠
    8. Non-Expert backward (GPU): Attention 等 dense 参数梯度计算（GPU optimizer）
    
    性能输出：iteration_time → throughput = tokens / iteration_time
    ```
  - 关键技术点：(1) 不使用 batched matrix multiplication——sequential expert processing 避免了 dispatch mask（节省 >48 GiB for MoE-L batch 32），允许 8× 更大的 microbatch；(2) Expert 从 CPU memory 到 SSD 的 eviction 使用 LRU cache policy + prefetching（基于 forward/backward 可预测序列），使用 DMA-able pinned memory 避免 naïve VM page fault stall。

## ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：在DeepSpeed分布式训练框架上实现三项运行时通信与调度优化：(1) **Adaptive All-to-All Communication**：运行时监控每个GPU上per-expert的token选择计数，通过all-gather聚合各GPU的计数信息，计算精确的输入/输出slice大小，消除传统zero padding带来的不必要数据传输（训练早期zero ratio 88%，后期升至98%）；(2) **Dynamic Expert Clustering**：运行时profiling每个token在各MoE层的expert选择历史（使用<batchID, sequenceID, tokenIndex, tokenName>唯一标识，12B开销可忽略），基于K-means聚类（距离函数 = 序列长度 - 重叠expert选择数）将相似expert选择模式的token分组，复制热门expert到本地GPU HBM、将冷门expert offload到host pinned memory，重新映射expert到GPU位置以减少跨设备通信；(3) **Topology-aware Expert Remapping**：构建coverage matrix（C×C，cluster间expert覆盖度）和bandwidth matrix（GPU对间点对点网络带宽），使用遗传算法（fitness function = Σ((b·s - CM[SV[i]][SV[j]]·h) / BM[i][j])）搜索近最优cluster-to-GPU映射，在异构网络中最小化跨节点通信延迟。CPU侧聚类和remapping操作通过superbatch机制与GPU迭代overlap执行（overhead从12.48%降至0.001%）。
  - 实验比较：(a) 端到端迭代时间对比：Baseline(Tutel) vs +ADPT vs +ADPT+DEC vs ScaleMoE（全优化），在homogeneous和heterogeneous网络下评估MoE-BERT和MoE-GPT；(b) 性能随时间分析：epoch 1-21的speedup变化、all-to-all通信时间、通信量变化；(c) 灵敏度分析：MoE layer ratio (4/12, 6/12, 12/12)、k:Ne ratio (1:16, 1:32, 1:64)、superbatch size (1-400)、expert replica数量 (0-31)；(d) overhead breakdown：各操作延迟分解及overlap效果。

- 后端平台是什么，配置是什么。
  - 硬件：Amazon EC2 p4d.24xlarge实例 × 4节点，每节点8× NVIDIA A100 40GB GPU（共32 GPUs）
  - 节点内互联：NVLink 3.0 (600 GB/s)
  - 节点间互联：Ultra Ethernet (100 Gbps)；heterogeneous配置中限制一个节点至50 Gbps模拟云环境网络异构（带宽差2×）
  - 软件：PyTorch v2.0、DeepSpeed（基础分布式训练框架）、Tutel（baseline，2DH All-to-All配置）
  - 模型：BERT-MoE（encoder-only）、GPT-MoE（decoder-only），12层Transformer，hidden dim=768，sequence length=128，batch size=512
  - 变体参数：MoE layers={4, 6, 12}，experts Ne={32, 64, 128}，k:Ne ratio={1:16, 1:32, 1:64}

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：在DeepSpeed+Tutel框架上集成ScaleMoE的三个优化模块（Python package形式），测量平均迭代时间（average iteration time），计算speedup = baseline_iteration_time / ScaleMoE_iteration_time
  - 修改内容：在DeepSpeed的MoE层all-to-all通信路径中hook入adaptive all-to-all逻辑（替换原有zero-padded all-to-all dispatcher），聚合per-expert选择计数后使用精确slice size的NCCL all-to-all；在训练循环中加入dynamic expert clustering和topology-aware expert remapping模块（CPU执行），通过superbatch机制与GPU迭代overlap；修改expert-to-GPU内存布局以支持expert replication（热门expert）和offload（冷门expert到host pinned memory）
  - 基线：Tutel (built on DeepSpeed) with 2DH All-to-All configuration，不包含ScaleMoE的任何优化

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/SKKU-IDEAL/ScaleMoE
  - 评估原理：测量每个training iteration的wall-clock时间（从forward开始到backward结束），计算speedup vs Tutel baseline；单独profiling all-to-all通信时间、通信量（bytes）、expert selection分布
  - 运行时通信调度全过程（以4 GPUs, 4 experts, 10 tokens/GPU的MoE层forward pass为例）：
    ```
    输入：40 tokens (batch×seq) 分布在4 GPUs上，hidden dim=768

    [GPU侧 - 每个iteration]
    Step 1 - Router: GPU-i本地对每个token执行 gate(x·W_g) → top-k expert indices
               e.g., GPU-1 tokens选择: E1×4, E2×1, E3×3, E4×2
    Step 2 - 监控: GPU-i统计 per-expert dispatch counts
               GPU-1: {E1:4, E2:1, E3:3, E4:2}
    Step 3 - All-gather counts: 4 GPUs交换counts → 每个GPU计算全局input/output slice sizes
               dispatch: GPU-i 的第j列 = 发给GPU-j的token数
               combine: GPU-i 的第j行 = 从GPU-j接收的output数
    Step 4 - Adaptive All-to-All (dispatch): NCCL alltoallv仅发送有效token数据（无zero padding）
               token embedding (768 floats) → GPU(对应expert)
    Step 5 - Expert FFN: 每个GPU对local experts执行 FFN(assigned tokens)
    Step 6 - Adaptive All-to-All (combine): FFN outputs按slice size返回原始GPU
    Step 7 - Reorder: tokens按原始sequence顺序重排

    [CPU侧 - 每superbatch=100 iterations，与GPU overlap]
    Step A - Profiling: 收集per-token expert选择历史（前一个epoch）
    Step B - K-means Clustering: 按expert选择模式聚类tokens → C个cluster
    Step C - Topology-aware Remapping: 遗传算法搜索最优 cluster→GPU 映射
              Fitness = Σ((b·s - coverage[i][j]·h) / bandwidth[i][j])
    Step D - Expert Redistribution: 热门expert复制到本地HBM，冷门expert offload

    输出：FFN输出tokens（重排后）→ 下一Transformer层
    性能输出：iteration_time → speedup = baseline_time / ScaleMoE_time
              all-to-all通信量 (MB) → 减少比例
              GPU 负载均衡度
    ```

## ScheMoE- An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Pipe-A2A：一种新型 pipelined all-to-all 通信算法，将 intra-node Send/Recv 操作与 inter-node Send/Recv 操作通过两个异步 CUDA stream（Intra-Stream 和 Inter-Stream）并行执行，充分利用异构带宽资源。
  - OptSche 最优调度算法：在 MoE layer 的 forward/backward 中，将 compute tasks（compression、decompression、expert computation）与 communication tasks（A2A dispatch/combine）按最优执行顺序重叠，最大化隐藏通信开销。
  - 数据压缩集成：支持 ZFP、FP16、INT8 等压缩算法在 MoE A2A 通信前后的 compress/decompress 任务纳入调度。
  - 实验比较：ScheMoE vs Tutel vs Faster-MoE 的 step time；Pipe-A2A vs NCCL-A2A、1DH-A2A、2DH-A2A 的 A2A 通信时间；消融实验（Naive / +ZFP / +ZFP+Pipe-A2A / +全部）。

- 后端平台是什么，配置是什么。
  - 32-GPU 集群：8 nodes × 4 Nvidia RTX2080Ti（@1.35GHz, 11GB Memory）
  - CPU：Dual Intel Xeon Gold 6230 CPU@2.10GHz，Memory: 512GB DDR4
  - Intra-node：PCIe 3.0 ×16
  - Inter-node：Mellanox MT27800 (ConnectX-5) 100Gb/s InfiniBand
  - 软件栈：PyTorch-1.10, Ubuntu-18.04, CUDA-10.2, cuDNN-7.6, OpenMPI-4.1.4, NCCL-2.13

- 评估性能的软件/脚本是什么。修改了什么。
  - ScheMoE 系统本身（开源于 https://github.com/Fragile-azalea/ScheMoE），基于 PyTorch 的 C/C++ 和 CUDA 扩展实现（~1200行 C/C++）。
  - 修改 PyTorch MoE layer：将原有 MoE layer 替换为 ScheMoE 的抽象模块（AbsCompressor、AbsAlltoAll、AbsExpert），支持任务队列化、Profiler 性能建模、Scheduler 调度。
  - 修改 A2A 实现：新增 Pipe-A2A 算法，使用两个异步 stream 分别处理 intra-node 和 inter-node SR（Send/Recv）操作。
  - 开源情况：代码开源在 GitHub（https://github.com/Fragile-azalea/ScheMoE），使用 ZFP、NCCL、Hetu、Tutel 等第三方库，采用 MIT 或类似许可证。
  - 评估原理与全过程（以 CT-MoE 模型为例）：
    ```
    输入：MoE layer config (B, L, M, H, E=32, k)
    ↓
    Step 1 - Profiler 预热：对 AbsCompressor (compress/decompress)、AbsAlltoAll (A2A)、AbsExpert (fflayer)
              分别 profile 时间，构建 t(C1), t(C2), t(A1), t(A2), t(D1), t(D2), t(E) 性能模型
    ↓
    Step 2 - 输入分区：将 gating 输出 tensor I ∈ R^{(E, C, M)} 按容量 C 均匀划分为 r=2 份
              I_1, I_2 → 各自进入独立的任务管道
    ↓
    Step 3 - OptSche 调度（r=2 时最优顺序）：
              CompTask顺序: (C_1^1 → C_1^2) → (D_1^1 → E^1 → C_2^1) → (D_1^2 → E^2 → C_2^2) → (D_2^1 → D_2^2)
              CommTask(A1,A2): 在前置 CompTask 完成后立即启动，由 CUDA stream 异步执行
    ↓
    Step 4 - Pipe-A2A（以 8-GPU, 2-node 为例）：
              Intra-Stream: SR(i, intra-node-j) 依次执行
              Inter-Stream: SR(i, inter-node-k) 依次执行
              → 两 stream 并行，intra 通信被 inter 通信隐藏
    ↓
    Step 5 - 数据压缩（可选）：
              Compress: I → ZFP_compress(I) 或 FP32→FP16 量化（减少 4× volume）
              → A2A dispatch (压缩后数据)
              → Decompress: 恢复原精度 → expert computation
    ↓
    Step 6 - 计时：cudaEvent 记录 MoE layer 起始到结束的 wall-clock time
    输出：step_time (ms) → speedup = t_baseline / t_ScheMoE
    ```

## Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ScMoE的**自适应算子调度（Adaptive Operator Scheduling）**实现专家并行中的通信-计算重叠，核心调度机制：(1) **双流并行架构**：ScMoE将MoE操作从backbone网络完全解耦后，执行两条独立CUDA stream——Shared Expert stream（处理当前层表示的计算路径）和MoE stream（处理前一层表示的通信+计算路径）；(2) **自适应专家计算定位**：MoE stream中gate routing和encode算子调度到最早可行位置，decode算子延迟到最后位置以最大化重叠窗口，核心挑战是将expert computation插入Shared Expert stream的4个候选位置（①②③④）之一（见图6），目标函数为 min_K(|ΣCOMP_pre - T_disp| + |ΣCOMP_post - T_comb|)，其中T_disp和T_comb分别是All-to-All Dispatch/Combine的通信时间，根据实际模型和硬件配置的性能数据自适应选择最优位置；(3) **异步All-to-All通信**：使用异步All-to-All通信算子，在CUDA stream间实现通信与计算的并行；(4) **与pipeline的兼容组合**：当通信时间超出重叠窗口时，ScMoE策略可与传统pipeline策略叠加——先利用ScMoE的扩展窗口隐藏部分通信，剩余部分通过pipeline以fine-grained chunk隐藏。与传统pipeline对比：pipeline策略将tokens均匀切分为chunks并行处理但无法重叠首尾chunk的通信（受限于prologue/epilogue bubbles），ScMoE直接消除这些限制实现100%通信隐藏。
  - 实验比较：(a) Overhead breakdown：各MoE架构（Standard top-2/top-1 + pipeline、Shared-Expert、ScMoE）在三种硬件配置下的通信/计算时间分解，ScMoE在8×A30-PCIe重叠70%通信、8×A800-NVLink完全重叠、16×A800-NVLink（cross-node）完全重叠；(b) 加速对比：ScMoE在8×A30-PCIe vs pipelined standard top-2 MoE 提升42%、vs pipelined top-1 MoE 提升15%、vs Shared-Expert MoE 提升27%；(c) 端到端训练/推理speedup：各模型+各硬件配置下的wall-clock加速比。

- 后端平台是什么，配置是什么。
  - GPU：8×NVIDIA A30-PCIe（PCIe互联，高通信开销，All-to-All占总时间60%）；8×NVIDIA A800-NVLink（NVLink互联，低通信开销，All-to-All占15%）；16×NVIDIA A800-NVLink across 2 nodes（节点间Ethernet互联）。单卡A30-PCIe用于memory-limited inference实验。
  - CPU-GPU通信：PCIe 4.0（expert offloading场景）。
  - 软件：PyTorch + CUDA streams，Tutel MoE + Fairseq框架。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于Tutel MoE框架和Fairseq训练框架。修改内容：(1) 实现MoE模块的双CUDA stream架构——将shared expert计算放在主stream，gate-routed expert的通信+计算放在独立stream；(2) 实现自适应算子调度器——基于profiled T_disp/T_comb和T_comp数据，计算min_K目标函数选择最优expert computation位置，运行时动态配置；(3) 实现异步All-to-All通信接口（NCCL异步模式）；(4) 实现与Tutel pipeline策略的组合模式开关。评估方法：测量每个Block-MLP+Block-MoE对的实际wall-clock时间（通过CUDA event timing），计算speedup ratio和通信重叠比例。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未提供独立的ScMoE调度器开源代码仓库。基于Tutel (https://github.com/microsoft/Tutel) 实现。
  - 评估原理与调度全过程（ScMoE Pos-2, 8×A30-PCIe, training one iteration）：
    ```
    输入：microbatch tokens [B, S, d] 分布在8 GPUs (DP+EP混合)
    
    # === Block-MLP (前一层，GPU主Stream) ===
    ├─ [Kernel] MultiHead_MLP Forward: QKV projections + attention + output proj
    │   输入: H_{l-1} [B, S, d] → 输出: H_l^{MH} [B, S, d]
    └─ [Kernel] MLP Forward: gate_proj → activation → up_proj → down_proj
        输入: H_l^{MH} → 输出: H_l^{MLP} [B, S, d]
    
    # === 调度决策（CPU侧，基于profiled数据） ===
    读取已profile的: T_Atten, T_SE, T_MLP, T_disp, T_comb
    计算overlap_window = T_Atten + T_SE + T_MLP (Pos-2)
    计算4个候选位置K∈{1,2,3,4}的cost:
      cost(K) = |Σ_{i=1}^{K-1} COMP_i - T_disp| + |Σ_{i=K+1}^{4} COMP_i - T_comb|
    选择 argmin_K cost(K) → 设expert computation位置
    
    # === Block-MoE (当前层，双CUDA stream) ===
    
    # MoE Stream (独立stream，与主stream并行):
    ├─ [Kernel] Gate Routing: H_l^{MH} @ W_gate + noise → TopK softmax
    │   → top-1 expert index per token (可在Block-MLP执行时提前调度)
    ├─ [Kernel] Input Encode: 聚合token data到连续layout
    ├─ [Communication] Async All-to-All Dispatch: 将tokens发送到目标expert所在GPU
    │   (与主Stream的Attention + Shared Expert重叠)
    ├─ [Kernel] Expert Computation (在调度器选择的位置插入):
    │   expert FFN: gate_proj(H_l^{MH}) → SiLU ⊙ up_proj(H_l^{MH}) → down_proj
    ├─ [Communication] Async All-to-All Combine: 将expert输出发回原始GPU
    │   (与主Stream的后续计算重叠)
    └─ [Kernel] Output Decode: 恢复token原始顺序
    
    # Shared Expert Stream (主GPU stream):
    ├─ [Kernel] MultiHead_MoE Forward (与MoE stream的gate+encode+dispatch并行)
    │   输入: H_l^{MLP} → 输出: H_{l+1}^{MH}
    ├─ [Kernel] Shared Expert Forward (与MoE stream的expert comp+combine并行)
    │   SE^{(l+1)}(H_{l+1}^{MH}): gate_proj → SiLU → up_proj → down_proj
    └─ [Kernel] Merge: coef * se_out + gate_weight * expert_out + residual
    
    # 通信重叠效果（8×A30-PCIe, T_disp+T_comb ≈ 60% of MoE time）:
    # overlap_window ≈ T_Atten + T_SE + T_MLP ≈ 70% of total → 70%通信被隐藏
    # 剩余30%通信无法隐藏 → 可叠加pipeline策略进一步隐藏
    
    性能输出：
    ├─ per Block-MLP+Block-MoE pair wall-clock时间 (ms)
    ├─ 通信重叠比例 = (T_disp+T_comb - 实际暴露的通信时间) / (T_disp+T_comb)
    ├─ 端到端speedup = T_baseline / T_ScMoE
    └─ 各component时间分解 (CUDA profiler timeline)
    ```
  - 关键技术点：(a) 自适应调度关键：expert computation位置的选择直接影响重叠效果——选在T_comp_pre最接近T_disp的位置使dispatch通信几乎完全隐藏，选在T_comp_post最接近T_comb的位置使combine通信同理；(b) ScMoE在通信时间≤约50%总MoE时间时可实现完全重叠（公式上下界保证）；(c) 与pipeline对比的核心优势：pipeline受限于prologue/epilogue不能被重叠的首尾传输，ScMoE无此限制。

## Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Skywork-MoE 的训练基础设施基于内部开发的 Skywork-Megatron 框架（基于 Megatron-LM 23.06 分支），核心 kernel/运行时调度创新包括：
    (1) **Expert Data Parallelism (EDP)**：自定义并行策略，定义为 Size_EP = Size_TP。在注意力层以 Tensor Parallelism 方式运行，在 FFN/MoE 层以 Expert Parallelism 方式运行，同一数据同时穿越 TP Group 和 EP Group。相比 Megatron-LM Core 0.6.0 的 EP（Size_EP = Size_DP * Size_TP，受 expert 数量上限限制）和 ETP（Size_EP = Size_DP，AllToAll 通信开销随 TP 增大），EDP 对中等 expert 数量（≤64）的模型优化了门控层 token 路由的 AllToAll 通信。
    (2) **Unbalanced Pipeline Parallelism**：打破均匀层分割（如 [6,6,6,6]），采用非均匀分割（如 [5,5,5,5,4]）减少 pipeline bubble time 达 10%。梯度重计算（checkpointing）也按 stage 差异化配置，平衡各 stage 的内存使用和计算开销。
    (3) **通信优化**：实现了 expert parallelism 相关通信缩减、kernel fusion、通信与计算重叠等优化，最终达到 38% MFU 和 690 tokens/GPU/sec。
  - 实验比较：(a) Uniform vs Non-uniform PP bubble time 对比（图 6），24 层 Transformer 模型 pipeline bubble time 减少约 10%；(b) 训练吞吐量：38% MFU on 1536 A800 GPUs, 690 tokens/GPU/sec；(c) EDP 与 EP/ETP 的理论对比分析（通信开销、扩展性约束）。

- 后端平台是什么，配置是什么。
  - GPU：192 节点 × 8 × NVIDIA A800-80G SXM = 1536 GPUs
  - 节点内互联：400 GB/s NVLink
  - 节点间互联：800 Gb/s RoCE 网络
  - 并行配置：12-way pipeline parallelism + 4-way tensor-expert parallelism (EDP) + 32-way data parallelism + ZeRO-1
  - 设备 mesh：Attention weights 为 [Size_PP, Size_DP, Size_TP]，Expert weights 为 [Size_PP, Size_DP, Size_EP]

- 评估性能的软件/脚本是什么。修改了什么。
  - 基础框架：Megatron-LM 23.06 分支
  - 内部框架名：Skywork-Megatron
  - 修改内容：(a) 实现自定义 MoE 架构（门控层、expert 层、tailored distributed parallel strategy）；(b) 实现 EDP 并行策略（设备 mesh 动态切换 Attention TP ↔ Expert EP）；(c) 实现 Unbalanced PP（非均匀层分割 + 差异化的梯度重计算配置）；(d) expert parallelism 通信缩减、kernel fusion、通信-计算 overlap 优化。
  - 训练配置：学习率采用多阶段调度（unique learning rate schedule per stage），cosine decay。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/SkyworkAI/Skywork-MoE，训练代码基于 Megatron-LM，包含自定义 MoE 架构实现。重点关注 `megatron/model/moe/` 目录下的 expert 并行与门控实现。
  - 训练框架全流程（Skywork-Megatron, 1536 A800 GPUs, Skywork-MoE 146B/16 experts）：
    ```
    输入：tokenized text batch (micro_batch_size, seq_len=8192)
    
    1. Data Loading & DP Sharding
       - 32-way data parallelism，每个 DP group 处理不同的 micro-batch
       - ZeRO-1 分布 optimizer state
    
    2. Pipeline Stage Partitioning (12 PP stages, Unbalanced)
       - 52 层非均匀分割到 12 个 PP stage
       - 最后 stage 少 1 层以补偿 loss calculation 的计算开销
       - 梯度重计算按 stage 差异化（buffer 大的 stage 少存 activations）
    
    3. Attention Layer (TP Group)
       - Device Mesh: [Size_PP, Size_DP, Size_TP=4]
       - Self-Attention: QKV projection → RoPE → Flash Attention → Output projection
       - TP 切分 head 维度，每 GPU 处理 36/4=9 heads
    
    4. MoE Layer (EP Group via EDP)
       - Device Mesh 切换: [Size_PP, Size_DP, Size_EP=4]
       - Gating: 4 GPUs × 4 experts each = 16 experts total
         - Gate forward + Logit Normalization (z_tilde = λ*(z-μ)/σ)
         - Softmax + Top-2 selection
       - Token Dispatch: AllToAll 通信在 4 EP GPUs 间路由 tokens 到目标 expert
       - Expert FFN (SwiGLU): 每 expert 独立计算 FFN(W_gate, W_up, W_down)
       - Token Combine: AllToAll 通信将 expert 输出送回原 GPU
       - Weighted sum: y_i = (g1*E1(x_i) + g2*E2(x_i)) / (g1+g2)
    
    5. Communication Optimizations
       - Expert parallelism AllToAll 通信缩减
       - Kernel fusion (如 gate + dispatch 融合)
       - 通信与 expert FFN 计算 overlap
    
    6. Loss & Backward
       - Cross-entropy loss + Σ α^(l) * L_aux^(l) (52 adaptive coefficients)
       - Backward through MoE (gradient through gating + expert FFN)
       - ZeRO-1 all-reduce gradients across 32 DP ranks
    
    输出：更新后的模型参数，throughput = 690 tokens/GPU/sec, MFU = 38%
    ```

## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Stratum 的 kernel 调度/运行时计算涵盖 Expert Processing 和 Attention Processing 两大类的算子映射和执行调度。(1) **Expert Processing 执行流程**（§4.1）：MoE layer 的 3 阶段 GeMM 操作（projection-up GeMM1 + GeMM2 → SiLU/Hadamard → projection-down GeMM3），采用 tensor parallelism 策略——沿不同维度分区矩阵（GeMM1/2 垂直分片，GeMM3 水平分片），避免 expert weight 复制，通过 all-gather 复制输入 token 再每个 PU 独立计算。Executed experts 顺序处理（非并行），所有 PUs 协作处理一个 expert。(2) **优化执行 Pipeline**（Figure 9）：输入 token 分片发送到各 DRAM channel → sub-ring all-gather 重建完整矩阵 → GeMM2 与 activation function evaluation 重叠 → GeMM3 的 reduce-scatter 与下一 expert GeMM1 并行 → weighted-sum 由 special function engines 在 expert 输出就绪后立即执行。(3) **Attention Processing**（§4.2）：利用 head-level parallelism + PU groups 分区执行，每个 PU group 处理多个 attention heads，interleaved Softmax 与其他算子执行。Query×Key + Softmax 与 Attn×Value 在多个 heads 间交错。Key/Value 矩阵沿 sequence length 维度分片，每 PU 独立计算 local max/sum 后仅交换标量进行全局 Softmax 归一化。(4) **On-chip Ring Network 通信调度**：16 PUs 通过双向 ring 互联，支持 all-gather、reduce-scatter、scalar exchange 等 collective 通信原语，ring router 内含 aggregator 实现 in-situ data reduction。
  - 实验比较：(a) Per-layer MoE latency vs hot expert hit rate（Figure 17a）；(b) Overall system throughput vs hot expert hit rate（Figure 17b）；(c) Decoding throughput scaling vs batch size (1-32)，different sequence lengths (256-4096)（Figure 18a）；(d) Throughput per area vs Mono3D DRAM layers (64/256/1024)（Figure 18b）；(e) Expert swap time/energy overhead per benchmark（Table 4）；(f) 与 Duplex 对比——2.2-3.0× throughput, 1.9-2.9× energy improvement。

- 后端平台是什么，配置是什么。
  - Stratum NMP Logic Die Processor：7nm process, 0.7V supply, 121 mm² die area, FP16 arithmetic. 16 PUs, 每 PU: 16 PEs（16×16 MAC tensor core each），64k MAC units total, 1 GHz operating frequency, 128 TFLOPS peak performance. 36 MB on-chip SRAM, aggregated ring bandwidth 2.048 TB/s. Aggregated Mono3D DRAM bandwidth 19.01-34.34 TB/s（8 tiers）. Peak power 43W（logic die only）.
  - xPU：NVIDIA H100 SXM5 HBM3（Stratum-L）、RTX A6000（Stratum-S）。
  - SystemVerilog 实现，Cadence Genus 综合（ASAP7 7nm PDK），post-synthesis 网表仿真。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 in-house cycle-level simulator——接受 tensor size、parameter tier assignments、attention head mappings、routed expert IDs 以及各组件 delay/energy 参数作为输入，输出总体执行时间和 component-level 能耗分解。
  - System-level simulator——包含 Request Generator（Poisson arrival）、SLO-Aware Scheduler、Memory/Computation Mapper、Stratum NMP interface。
  - 组件 delay/energy 参数来源：(a) Cadence Genus synthesis reports for area/timing/power；(b) post-synthesis netlist simulation with annotated switching activity for energy；(c) FinCACTI for SRAM modeling (shared memory, psum memory)。
  - GPU baseline 评估使用 vLLM 0.8.1 benchmark throughput mode, GPU energy from NVIDIA-SMI。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未开源 Stratum 仿真器代码。
  - NMP 执行全过程（single MoE layer on Stratum-L, Mixtral 8×7B）：
    ```
    输入：batch tokens X_t [M×K] 从 xPU 发送到 Mono3D DRAM
    xPU 完成 Gating → expert routing IDs → switch Mono3D DRAM to NMP mode
    
    [Step 1] xPU → DRAM data transfer:
      - X_t partitioned into slices → each sent to distinct DRAM channel
      - Sub-ring all-gather: each PU reconstructs full X_t
    
    [Step 2-7] Sequential Expert Processing (tensor-parallel across all 16 PUs):
    For each activated expert e:
      [Step 2] GeMM1: Z_1 = X_t @ W_1[i] (projection-up, matrix partitioned vertically)
               → PE Tensor Cores: 16×16 MAC array, k-tap dot-product engines
               → Data loaded from Mono3D DRAM via hybrid bonding (19-34 TB/s)
      [Step 3] GeMM2: Z_2 = X_t @ W_2[i] (projection-up, parallel with Step 2)
               → Overlapped with Step 4 activation
      [Step 4] Activation: SiLU(Z_1) via Special Function Engine (256-way SIMD)
      [Step 5] Hadamard: SiLU(Z_1) ⊙ Z_2 = X_2  (no inter-PU communication needed)
      [Step 6] GeMM3: Z_3 = X_2 @ W_3[i] (projection-down, matrix partitioned horizontally)
               → Partial sums accumulated in intra-channel reducer tree
      [Step 7] Reduce-Scatter: aggregate Z_3 partial sums across PUs via ring network
               → Overlapped with next expert's GeMM1 (pipeline optimization)
    
    [Step 8-9] Post-Processing:
      [Step 8] Go to next expert (repeat Steps 2-7)
      [Step 9] Weighted Sum: Σ gate_score_e * expert_output_e via Special Function Engine
    
    [Step 10] Write back results to DRAM designated address → exit NMP mode → xPU reads
    
    Performance output:
    - MoE layer latency = Σ expert compute times + max(communication, compute) overlaps
    - Energy = Σ (PE energy + ring network energy + DRAM access energy)
    - Tiering impact: fast tier tRCD=2.29ns vs slow tier tRCD=22.88ns
    ```
  - Attention 执行全过程：
    ```
    输入：xPU 写入 new KV pairs 到 DRAM, queries Q via DRAM channels
    
    [PU Group Assignment]:
      - 8 attention heads → 4 PU groups × 2 heads/group
      - Each PU group: neighboring PUs on ring topology
    
    [Per-Head Execution (within PU group)]:
      Step A: Sub-ring all-gather Q (replicate to all PUs in group)
      Step B: S = Q @ K^T (partitioned along sequence length dim)
              → Each PU computes local S slice → scalar exchange local max/sum
              → Global softmax normalization
      Step C: O = softmax(S) @ V (partitioned along sequence length dim)
      Step D: Reduce-scatter aggregated O across PUs in group
    
    [Interleaved Pipeline (2 heads per group)]:
      Head1: Q@K → Softmax(steps 1,2,3 with inter-PU scalar comm) → Attn@V
           while Head1 Softmax: Head2 Q@K overlaps
           while Head2 Attn@V: Head1 next decode step
    
    Key optimization: Softmax decomposed into 3 steps with 2 rounds inter-PU comm,
    interleaved with other head's MatMul for latency hiding.
    ```

## Toward Efficient Inference for Mixture of Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  论文在 MoE gating 函数层面改变了 token dispatch 的 kernel 计算模式：将 Static Gating 的 **batch matrix multiplication (bmm, sparse mask × tokens)** 替换为 Dynamic Gating 的 **argsort + bin-count + indexing** 方案。这是 kernel 级别的优化——用 O(SD) indexing kernel 替代 O(S²EDC) batch matmul kernel，消除 dispatch mask 的内存分配和 placeholder computation。此外，Dynamic Gating 使用两轮 all-to-all（先通知 size 再传 tokens），将通信模式从固定大小 all-to-all 改为可变大小 all-to-all。
  实验比较：与 baseline Fairseq（static gating + bmm dispatch）、Tutel（hash table lookup + custom cumulative sum kernel）、FasterMoE（kernel launch overlap）、Megablock（block-sparse BCSR kernel）对比。分析 latency breakdown（gating、all-to-all、expert 执行的贡献比例）和 memory trace。

- 后端平台是什么，配置是什么。
  GPU：NVIDIA Tesla V100 (32GB, NVLink, Volta SM) 和 NVIDIA RTX A5000 (24GB, Ampere SM)。CPU：Intel Xeon E5-2698 v4 (Apple) & Intel Xeon Gold 5317 (Pear)。CPU-GPU：PCIe 3.0 16GB/s 和 PCIe 4.0 32GB/s。
  Megablock 的 custom kernel 需要 A5000 的 Ampere 架构特性，但不支持 bias term。

- 评估性能的软件/脚本是什么。修改了什么。
  Fairseq MoE 实现作为 baseline。评估使用 Python `time` 模块记录 latency + PyTorch Profiler 收集详细 CUDA kernel trace + memory trace。
  修改内容：
  1. **Gating kernel 替换**：将 `bmm(mask, tokens)` 替换为三步操作：
     - `torch.argsort(assignments)`: 按 expert ID 排序，GPU kernel，O(S log S)。
     - `torch.bincount(assignments)`: 统计每个 expert 的 token 数量，O(S)。
     - `tokens[sorted_idx]`: 高级索引（advanced indexing）kernel，O(SD)，直接内存重排。
  2. **Reordering kernel 替换**：dispatch 后的 token 重排序也从 bmm 替换为 indexing。
  3. **两轮 all-to-all 通信模式**：第一轮传 sizes（标量，极低延迟，avg 20µs）；第二轮传 tokens（可变大小，消除 zero-padding）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。

  评估原理与 kernel 数据流：

  ```
  [Kernel-level Performance Evaluation Pipeline]

  PyTorch Profiler → CUDA trace (kernel launch events, duration)
  Python time.time() → end-to-end batch latency
  Memory trace → torch.cuda.memory_allocated() / memory_reserved()

  === Static Gating Kernel Flow (Baseline) ===

  Input: tokens X ∈ R^{S×D}, gating assignments A ∈ Z^{S×k}
  
  [Kernel 1] Gate Linear:       W_gate @ X          → gate_logits (S, E)
  [Kernel 2] Top-K:              topk(gate_logits)    → assignments (S, k)
  [Kernel 3] Mask Construction:  create_dispatch_mask → mask (E, S, S×C)
            → 内存分配: E × S × S×C × 4 bytes (float32)
            → LM S=8, E=512, C=0.05: 512 × 8 × 25.6 × 4 ≈ 419KB
            → 但论文 Fig.10 显示 gating/reordering 内存尖峰可达 GB 级别
            → （因 batch matmul 内部需要大工作区）
  [Kernel 4] Batch MatMul:       bmm(mask, X)        → dispatched (E, S×C, D)
            → O(S² × E × D × C) ≈ 512 × 64 × 25.6 × 1024 ≈ 860M FLOPs
            → 其中大部分计算为 ×0 (mask 极稀疏) → 浪费
  
  [Kernel 5-7] All-to-All, Expert FFN forward, All-to-All collect
  [Kernel 8] Batch MatMul:       bmm(mask^T, out)    → 还原 token 顺序

  === Dynamic Gating Kernel Flow (Proposed) ===

  Input: tokens X ∈ R^{S×D}, gating assignments A ∈ Z^{S×k}

  [Kernel 1] Gate Linear:       W_gate @ X           → gate_logits (S, E)
  [Kernel 2] Top-K:             topk(gate_logits)     → assignments (S, k)
  
  [Kernel 3] Argsort:           torch.argsort(A[:,1]) → sorted_idx (S,)
            → GPU radix sort / merge sort kernel
            → O(S log S), S=8 时 trivial; S=512 时 ~tens of µs
  
  [Kernel 4] BinCount:          torch.bincount(A[:,1]) → sizes (E,)
            → GPU reduction kernel, O(S)
  
  [Kernel 5] Advanced Index:    X[sorted_idx]          → sorted_X (S, D)
            → GPU gather kernel, directly reorder via indices
            → O(SD) memory bandwidth bound, NOT compute bound
            → 无临时 mask tensor 分配!
  
  [Comm Round 1] All-to-All:    sizes (E integers)     → 20µs avg latency
            → 各 GPU 现在知道 incoming tensor shapes → pre-allocate
  
  [Kernel 6] Split:             torch.split(sorted_X, sizes)
            → 按 sizes 切分 sorted_X → variable-length groups
  
  [Comm Round 2] All-to-All:    variable-size tokens (zero padding = 0)
            → 仅传输实际 tokens，无 placeholder 浪费
  
  [Kernel 7-8] Expert FFN forward per GPU
  [Comm Round 3] All-to-All:    expert outputs back
  
  [Kernel 9] Advanced Index:    inverse_permutation  → 还原 token 顺序

  === Kernel 效率对比 ===

  Static Gating batch matmul (waste analysis for LM, E=512, C=0.05):
    - 每个 expert 配置处理 ECS = 512 × 0.05 × S = 25.6S tokens
    - 实际仅需 2S tokens (top-2 gating)
    - Waste factor: 25.6S / 2S = 12.8×
    - 即 92.2% 的 batch matmul FLOPs 浪费在零值上

  Dynamic Gating indexing:
    - 传输的 tokens 数 = 实际需要的 tokens 数 (zero waste)
    - Indexing 是纯内存操作 (O(SD) BW), 无浪费计算
    - 但增加 1 次 light all-to-all (~20µs) 开销

  为什么 Dynamic Gating 在大 batch size 下优于 Megablock:
    - Dynamic Gating: 多个 dense matmul (每 expert) → GPU efficient
    - Megablock: 单个 BCSR sparse matmul → 需要 metadata (col indices, row offsets)
      → indexing 开销随 batch size 增大 (matrix 增大)
    - Dynamic Gating: kernel launch 数 = expert 数量（固定）
      → overhead 不随 batch size 变化
    - 实验: batch=80 时 Dynamic Gating 比 Megablock 快 1.46×

  Expert Buffering 相关的 kernel 操作:
  [Kernel] cudaMemcpyAsync: CPU→GPU expert 参数 (PCIe stream)
           → 与 all-to-all NCCL stream 并发 → 零额外延迟
  ```
## X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：X-MoE 实现了一套基于 Triton 的跨平台 sparse/irregular kernel，用于支持 PFT padding-free MoE pipeline 在 AMD MI250X GPU 上的高效执行：
    (1) **Triton Gather Kernel**：将 gating 输出 gate_out [S, H] 按 token_ids [B] 索引 gather 到 dispatch_in [B, H]。执行 `dispatch_in[i,:] = gate_out[token_ids[i],:]`。每个 token 分配一个 thread-block（256 threads），block bi 负责复制 gate_out[token_ids[bi],:] → dispatch_in[bi,:]，沿 hidden dimension 循环 H/256 次。通过连续线程处理 hidden dimension 的连续内存位置，确保即使有 `gate_out[token_ids[i],:]` 的不规则索引，内存请求仍是 coalesced。
    (2) **Triton Scatter Kernel**：将 MLP 输出 mlp_out [B, H] 按 token_ids 反向 scatter 回原始序列位置，同时乘以 combine_weights。执行 `combine_in[token_ids[i],:] = mlp_out[i,:] * combine_weights[i]`。不规则写访问通过连续线程沿 hidden dimension 写入保证 coalescing。
    (3) **Sequential GeMM**（非 Triton，Python for-loop 驱动）：在 dispatch_out [Bexp, H] 上，按 tokens_per_expert 数组切片，依次为每个 expert 启动一个标准 GeMM。第 i 个 expert 处理 tokens `dispatch_out[sum(tpi[:i]):sum(tpi[:i+1])]`，共 Elocal 次 GeMM launch。替代传统 padded batched matmul，避免 zero-padding 带来的无效计算。
  - 实验比较：(a) MoE layer forward time breakdown：X-MoE vs DeepSpeed-MoE，在 Small 模型 (EP=8) 和 Large 模型 (EP=64) 上，gating/buffer dispatch/dispatch alltoall/expert compute/combine alltoall/buffer combine 各阶段延迟对比；(b) 激活内存消耗：X-MoE vs DeepSpeed-MoE vs Tutel 在 Large 模型 256 GPU 下的 per-MoE-layer 内存 (GB)；(c) RBD 的 dispatch time breakdown：with/without RBD 的 inter-node alltoall + intra-node alltoall + data transform 时间；(d) Cross-platform：8×NVIDIA A100 上 X-MoE vs DeepSpeed-MoE vs Tutel 的 TFLOPs 和 OOM 情况。

- 后端平台是什么，配置是什么。
  - 主平台：AMD MI250X GPU（Frontier 超级计算机），每 GPU 2 GCD（视为独立 GPU），峰值 191.5 TFLOPs/effective-GPU，Infinity Fabric intra-node（50-200 GB/s），Slingshot 25 GB/s inter-node。
  - 软件栈：ROCm 5.7.1，PyTorch 2.2.0，DeepSpeed 0.15.5，Triton（版本论文未明确说明），AWS-OFI-RCCL plugin + libfabric 1.20.1。
  - 跨平台验证：NVIDIA A100 40GB，CUDA 平台。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 DeepSpeed 0.15.5 + DeepSpeed-Megatron 实现。核心修改：
    (a) **PFT 构造与 ERI-arrays 管理**：在 MoE gating 后实现 PFT construction 例程（flatten + sort top_experts + one-hot + cumsum + token dropping + histogram），替换传统 dispatch_mask 生成。
    (b) **Triton Gather/Scatter Kernel**：替换 einsum-based dispatch，实现 coalesced memory access 的 Triton kernel（gather 读 coalesced，scatter 写 coalesced，均沿 H 维度连续线程分配）。
    (c) **Sequential GeMM**：替换 batched matmul，按 tokens_per_expert 切片为每个 expert 单独 launch GeMM。
    (d) **RBD dispatch/combine 流程**：实现 pilot selection + s1_mapping_indices 构建 + 两级 alltoall（跨节点 uneven + 节点内 uneven）+ local replica 重建 + merge/reorder。
    (e) **SSMB 序列切分**：在 TP→EP 转换处 drop partial tokens，MoE block 结束后 all-gather 恢复。
  - 开源：https://github.com/Supercomputing-System-AI-Lab/X-MoE，集成于 DeepSpeed。

- 基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 以单次 MoE layer forward pass（在 AMD MI250X，Large 模型，EP=64，256 GPU）为例，X-MoE kernel 执行全过程：
    1. **输入**：gating 输出 gate_out [S, H]（S=sequence_length×batch_size, H=7168），top_experts [S, K]（K=8），combine_weights [S, K]，均为 FP16/BF16。
    2. **PFT Construction**（CPU/GPU 混合）：flatten top_experts [S*K] + argsort by combine_weights + one_hot → cumsum（优化为 [E, S*K] layout 使 cumsum 在 outer dimension coalesced，加速 10×）+ weight_mask 过滤 dropped tokens → 生成 ERI-arrays（token_ids [B], expert_ids [B], tokens_per_expert [E], combine_weights [B]），B ≤ max_token_count × E。
    3. **Gather Kernel**（Triton，GPU）：launch B thread-blocks × 256 threads，block bi 执行 `dispatch_in[bi, :H] = gate_out[token_ids[bi], :H]`，H=7168，每个 thread-block 循环 7168/256=28 次。内存访问 coalesced（连续线程→连续 hidden dim 元素）。输出 dispatch_in [B, H]。
    4. **Dispatch Alltoall**（RCCL + libfabric）：先 alltoall tokens_per_expert [E]（metadata，轻量），后 alltoallv dispatch_in [B, H] → dispatch_out [Bexp, H]。RBD 模式下仅 pilot tokens（去重后）走跨节点 alltoall，local replica 走节点内 alltoall。
    5. **Sequential GeMM**（rocBLAS）：for i in 0..Elocal-1: slice = dispatch_out[offset:offset+tpi[i]]; inter = slice @ w1[i]; out = inter @ w2[i]。每 expert 独立 GeMM，无 zero-padding 计算。
    6. **Scatter Kernel**（Triton，GPU）：launch B thread-blocks，执行 `combine_in[token_ids[i], :H] = mlp_out[i, :H] * combine_weights[i]`。写 access coalesced 沿 hidden dimension。输出 final_output [S, H]。
    7. **评估原理**：Throughput (TFLOPs) = 模型单步总 FLOPs / iteration_time。通过 PyTorch profiler 记录各阶段耗时（gating/dispatch/alltoall/expert/combine），内存通过 `torch.cuda.max_memory_allocated()` 峰值。Scalability 通过 weak scaling（固定 per-GPU batch，增加 GPU 数）和 strong scaling（固定 global batch，增加 GPU 数）评估。
