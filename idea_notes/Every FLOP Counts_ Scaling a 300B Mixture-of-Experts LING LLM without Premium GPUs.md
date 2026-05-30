## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- baseline方法是什么？
  Baseline 是传统的大规模 MoE 模型训练范式：依赖高端 GPU（H100/H800）集群进行同步分布式训练（All-Reduce），所有计算在同类高端加速器上执行，存储和 I/O 使用传统并行文件系统（如 GPFS）。

  **Baseline 全栈执行例子（以 Ling-Plus 290B MoE 单步训练为例）**：
  - **算法层**：标准 MoE 架构，top-k token-to-expert routing，所有 expert 等容量设计。训练初期 router 随机初始化导致某些 expert 严重过载或闲置——早期 expert 负载崩溃是最常见的训练失败模式。Loss spike 发生时无自动处理机制，可能导致 wide spike 持续多个 step 使 benchmark 降到随机水平。
  - **系统框架层**：Megatron-LM / DeepSpeed 同步 All-Reduce 分布式训练。所有 worker 完成整个 step 后 barrier 同步——慢节点（straggler）瓶颈所有节点。checkpoint 写操作：Megatron 默认所有 DP group 的 rank_0 负责 checkpoint 数据聚合和写入，导致这些 rank_0 集中到少数物理节点，CPU 和网络带宽竞争激烈。
  - **编译框架层**：论文未明确说明（使用标准 PyTorch + CUDA 编译路径，依赖框架搭建）。
  - **Kernel 调度层**：同步 All-Reduce 每 step 做全局通信 barrier，无法重叠通信与计算。算子和通信 kernel（group_gemm、permute/unpermute、all2all、expert parallelism）在不同加速器平台实现不一致。无有效的 kernel 级性能诊断工具——需要全量监控（NVTX profiler）消耗大量内存而难以长期在生产环境使用。
  - **硬件架构层**：高端 H100/H800 GPU，NVLINK/NVSwitch 互联。设备类型单一。训练成本极高：Ling-Plus 在 Device D（989 TFLOPS）上训练 1T tokens 需约 635 万 RMB。

  **Baseline 核心痛点**：
  1. 高端 GPU 供需严重失衡——商业部署高峰期高端 GPU 被抢占用于在线推理，研发团队面临长期短缺
  2. 异构设备间计算/通信算子实现不一致（group_gemm, all2all, permute/unpermute），跨平台迁移导致精度累积偏差
  3. StoCworker 问题导致大规模同步训练效率剧烈下降（1000+ 节点时 baseline 速度降至 5.49e-2 step/s）
  4. MoE 训练早期极其不稳定——expert 负载崩溃、loss spike/divergence
  5. 跨集群存储同步慢（PB 级数据 OSS List 需 >6h）、checkpoint I/O 瓶颈（rank_0 集中写入）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出系统性方案在**低规格异构硬件**上完成 300B MoE 模型训练，核心是 5 个系统层级优化：

  1. **模型架构层面的 MoE 稳定性**：Fine-Grained Experts + Shared Expert 设计提升专家专业化的同时保持通用能力；Stochastic Routing Warmup 在训练早期引入受控随机路由噪声防止 expert 崩溃；NormHead 通过 L2 归一化 LM-Head 权重抑制 loss divergence；Skip Loss Spikes + Sample Retry 自动检测/跳过/重试 spike 步。
  2. **EDiT 异步分布式训练**：layer-wise sync + pseudo gradient penalty + time-based sync 替代同步 All-Reduce，解决 straggler 问题，最大加速 66.1%。
  3. **XPUTimer 轻量诊断**：selective tracing + async CUDA event + 数据压缩，90% 内存节省，O(1) 快速定位。
  4. **PCache + Babel 存储优化**：FUSE+shm 消除用户/内核态切换；AI co-design 分散 DP rank_0 写入（checkpoint 写延迟 -50%，峰值内存 -60%）；并行 metadata prefetch 加速跨集群同步 36×。
  5. **Flood 纯 PP 离线推理**：放弃 TP 避免低互联带宽下的通信开销，Segment Cache 替代 PageAttention。

  **论文方法全栈执行例子（以 Ling-Plus 290B MoE 单步训练为例）**：
  - **算法层**：Fine-grained experts (N 个低维 expert) + Shared expert（所有 token 不经路由全量通过）。Stochastic Routing Warmup：step i 时 ŝ_t = (i/W)·s_t + (1-i/W)·(μ_s+σ_s·ε)，ε~N(0,I)。NormHead：h_o = W_lm_head/||W_lm_head||₂ · h。Load balance loss (α=0.015) + router z-loss (α=1e-4)。训练阶段：初始预训练 9T tokens (4K ctx) → 长上下文 150B tokens (16K ctx, RoPE θ 10K→600K) → 退火（inverse sqrt decay lr 1.2e-4→1.2e-8）。Loss spike 发生 → 跳过当前 update → 保存数据 → 随机重注入后续 batch → 持续 spike 则 lr *= decay_factor。
  - **系统框架层**：DLRover 统一管理 DeepSpeed/Megatron-LM/Megatron vendor version 跨平台部署。EDiT 实现：每 worker 逐层 forward→backward→逐层 sync（非全局 barrier）。Pseudo gradient penalty 三步：(i) EMA 追踪 pseudo_grad 检测异常 worker→排除；(ii) 剩余 worker 按 pseudo_grad norm 加权平均；(iii) 统一梯度裁剪。Time-based sync：到达时间阈值而非固定步数触发同步。checkpoint 写入：PCache 将 DP group 写入任务 round-robin 分配到不同物理节点避免集中竞争。
  - **编译框架层**：论文未明确说明。使用 FlagScale 等开源分布式训练框架，针对不同加速器平台做底层算子一致性验证（matmul、linear、Attention、MLP、Router）。
  - **Kernel 调度层**：XPUTimer 运行时轻量追踪——Python 层通过 TRACED_PYTHON_API 环境变量动态拦截 API；C++/CUDA 层框架无关 kernel 监控（cuBLAS, Flash Attention, NCCL, 自定义算子）；CUDA event pool 复用 + 异步后台线程日志 + 数据压缩（仅记录时间戳和 kernel 输入 layout，~1.5MB/加速器/step）。EDiT 中通信与计算重叠：layer-wise sync 的 prefetch 机制在 backward 时同步下一层权重。PCache 使用 FUSE + shm 实现用户-内核态零拷贝写入 NVMe SSD。
  - **硬件架构层**：五种异构加速器混合训练（Device A 370 TFLOPS 64GB 无 FP8 → Device D 989 TFLOPS 80GB FP8 → Device E 147 TFLOPS 96GB FP8）。通过跨平台操作一致性验证（matmul, linear, attention, MLP, router forward → backward alignment）确保不同硬件上训练精度收敛一致——即便单个操作精度差异微小，累积后也会导致 loss 收敛巨大偏差，因此必须逐 operator 和逐 framework module 进行 forward+backward 完整对齐。节省约 20% 计算成本。

- baseline方法是什么？
  Baseline 是 vLLM 中的 TP+TP（Attention TP + MoE TP）并行策略，使用 cutlass GroupGemm 作为 MoE FFN 的默认 GEMM 实现，ncclAllReduce 作为通信原语。

  **Baseline 全栈执行例子（以 Mixtral 8x7B 单个 token 推理为例）**：
  - **算法层**：MoE gating 选择 top-k experts，FFN 包含 Gate/Up/Down 三个 GEMM
  - **系统框架层**：vLLM TP+TP 模式——Attention 和 MoE 权重均在 D 个设备上 TP 切分，所有 token 在所有设备上存在
  - **编译框架层**：论文未明确说明（使用标准 PyTorch + CUDA 编译路径）
  - **Kernel 调度层**：cutlass GroupGemm 单次 kernel launch 处理所有 expert 的 GEMM（所有 expert 共享一个 kernel grid）。ncclAllReduce 进行 TP 通信，每层 2×(D-1)×P/D 通信量。GEMM 和通信串行执行，无重叠。GEMM kernel 独占所有 132 SM。
  - **硬件架构层**：NVIDIA H800 SXM GPU，NVLink 互联。GEMM 计算在 Tensor Cores，all2all 通信经 NVLink。SM 全部分配给 GroupGemm，通信在 GEMM 完成后才启动，导致 SM 资源在通信期间空闲。

  **Baseline 核心痛点**：
  1. TP+TP 通信量最大（V_{TP+TP} > V_{TP+EP} > V_{DP+EP}），且 ncclAllReduce 不拆解，无法与 GEMM 并行
  2. GroupGemm 在大输入规模（prefill 阶段 m≥4096）效率低于 DenseGemm，但 baseline 固定使用 GroupGemm
  3. GEMM 和 all2all 串行执行，GPU SM 资源在通信阶段闲置，资源利用率低

- 论文方法是什么？如何对应解决Baseline的缺陷？
  EPS-MoE 提出三模块组合方案：
  1. **并行策略重设计**：MoE 块从 TP 切换到 EP，Attention 块保持 TP（MHA/GQA）或 DP（MLA）。将 ncclAllReduce 拆解为 ReduceScatter+all2all（dispatch）和 all2all+AllGather（combine），通信量从 V_{TP+TP} 降至 V_{TP+EP}。
  2. **Expert Pipeline Scheduler**：水平切分输入（按行），权重按专家切分。每次只传输当前专家组所需的 token，将 GroupGemm 的 group 数从 E 降至 E/N。当 N=E 时 GroupGemm 退化为 DenseGemm。根据负载（m 大小）动态选择 GEMM 类型。
  3. **SM 控制的计算-通信重叠**：限制 GEMM 的 SM 数（如 116 SM），留出 SM 给通信 kernel（16 SM），使 GEMM 计算与 all2all 通信在不同 SM 上并行执行。

  **论文方法全栈执行例子（以 Mixtral 8x7B 单个 token 推理为例）**：
  - **算法层**：同 baseline，MoE gating 不变
  - **系统框架层**：vLLM TP+EP 模式——Attention 权重 TP 切分，MoE 专家按 EP 分布。每个 token 经 router 确定 top-k experts 后，只将 token 发送到对应专家所在的设备（all2all），而非广播到所有设备
  - **编译框架层**：论文未明确说明（cutlass/cublas 库调用）
  - **Kernel 调度层**：
    - 输入按行水平切分 N 组（N=PN），权重按专家切分
    - Dispatch 阶段：ncclReduceScatter + all2all，token 只传输到目标专家设备
    - 第 1 组 token all2all 完成后启动第 1 组专家的 GEMM（根据 m/N 大小选 GroupGemm 或 DenseGemm），同时第 2 组 token 开始 all2all
    - GEMM 限制 116 SM，通信占用 16 SM，两者在不同 SM 上并行
    - Combine 阶段：all2all + ncclAllGather 聚合结果
    - ⚡ 关键改进：GEMM 类型自适应切换（小 m → GroupGemm，大 m → DenseGemm），计算与通信在 SM 级并行
  - **硬件架构层**：同 NVIDIA H800 SXM GPU，SM 资源被划分为计算区（116 SM）和通信区（16 SM），两者同时工作，消除通信阶段的 SM 空闲

  **痛点映射**：
  | Baseline 痛点 | EPS-MoE 解决方案 |
  |---|---|
  | TP+TP 通信量最大 | TP+EP / DP+EP 减少通信量，all2all 替代 AllReduce |
  | GroupGemm 在大 load 下效率低 | 水平切分 + 按专家切权重，m 大时自动切换 DenseGemm（cublas） |
  | GEMM 和通信串行，SM 闲置 | SM 控制 + pipeline 重叠，GEMM 和 all2all 在不同 SM 上并行 |
  | 固定调度策略忽略负载特性 | Load-aware 自适应调度，根据 m 动态选择 GEMM 类型和 pipeline 数 PN |

  实验效果：DeepSeekV2 prefill throughput 从 100K 提升至 121.8K tokens/s (+21.8%)；Mixtral 8x7B TTFT 最多降低 24.3%；DBRX TTFT 最多降低 30.5%。
