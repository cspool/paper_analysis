## HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference

- baseline方法是什么？
  - **TP-based static parallel inference**：主流 MoE 推理系统（vLLM、DeepSpeed-FastGen、SGLang、TensorRT-LLM）采用静态 TP（Tensor Parallelism）作为默认并行策略。TP 沿 hidden dimension 切分模型权重到多 GPU，通过 AllReduce 聚合部分计算结果。全栈执行例子（以 Mixtral-8x7B 在 4×A6000 PCIe、4096-token context + 64-token generation 为例）：
    - **模型推理算法层**：Mixtral-8x7B, 32 layers, hidden=4096, 8 experts/layer, top-2 routing。Token → Embedding → 32× MoE Transformer Layer（Attention + MoE gate + top-2 expert FFN）→ LM head → next token。
    - **系统框架层**：DeepSpeed-FastGen，TP=4。每层执行：Attention（TP=4，Q/K/V/O 各投影沿 hidden dim 切分，局部计算 + AllReduce 聚合）→ MoE gate（各 GPU 复制执行）→ Expert FFN（TP=4，gate/up/down 投影沿中间维度切分，各 GPU 计算 1/4 输出 → AllReduce 聚合）。固定并行策略，prefill 和 decode 使用相同配置。
    - **编译框架层**：论文未明确说明。PyTorch + DeepSpeed-Inference，标准 CUDA kernel。
    - **kernel 调度层**：NCCL AllReduce collective kernel + cuBLAS GEMM。prefill 期间 AllReduce 通信数据量 ≈ 2× hidden × batch × seqlen，在大 batch/长序列下成为瓶颈。decode 期间通信量小（单 token），但 EP 替代 TP 时负载不均衡问题突出。
    - **硬件架构层**：4× A6000，PCIe Gen4 互联（≤32 GB/s per direction），节点内低带宽。
  - Baseline 痛点：
    1. **固定张量分区无法适应场景变化（核心痛点 1）**：TP 对所有算子使用统一的张量切分策略，但不同算子（Attention vs Expert FFN）和不同推理阶段（prefill vs decode）对计算/通信的需求不同。长上下文 prefill 场景下 TP 的 AllReduce 通信成为瓶颈（通信数据量 ∝ batch×seqlen），而短序列 decode 场景下 EP 的负载不均衡导致计算资源浪费。
    2. **带宽利用不匹配（核心痛点 2）**：不同并行策略有不同的通信模式（TP 用 AllReduce，EP 用 All-to-All），静态策略无法根据实际硬件带宽（NVLink vs PCIe vs InfiniBand）自适应选择通信模式。在 PCIe 低带宽环境下，TP 的 AllReduce 通信开销严重，而在 NVLink 高带宽环境下 TP 的通信开销可接受。
    3. **多样化 MoE 架构适配能力差（核心痛点 3）**：Mixtral 系列（少 expert、大 expert）与 Qwen 系列（多 expert、小 expert、含共享 expert）的计算/通信特征差异大，静态策略无法自动适配不同模型配置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HAP 方法**：通过 Module Decomposition + ILP-based 搜索 + 动态策略切换三阶段设计，将 MoE 推理的并行策略从"人工静态配置"转变为"自动动态最优搜索"。
    1. **Module Decomposition + 仿真模型**（解决痛点 2 的基础）：将 MoE 模型分解为 Attention 模块和 Expert 模块独立建模。计算仿真模型基于 FLOPs（T_cal = F_module / Max_FLOPs × η，η 由随机森林回归拟合），通信仿真模型基于数据量和带宽（T_comm = V_data / Bandwidth × ρ，ρ 由随机森林回归拟合）。这使得不同硬件平台（A100 NVLink vs A6000 PCIe vs V100 PCIe）的延迟可被精确预测（计算误差 <10%，通信误差 <5%）。
    2. **ILP-based Hybrid Parallel Strategy Search**（解决痛点 1 和 3）：构建包含所有可行并行策略组合的搜索空间——Attention 模块（DP/TP/DP+TP）、Expert 模块（EP/TP/EP+TP，排除 DP 以节省内存）。将最小化端到端延迟问题形式化为 ILP（包含 prefill 延迟、decode 延迟、策略切换开销三项），约束条件包括显存限制、整除约束、并行度约束。求解器（Python PuLP）在典型 8-GPU 配置下 <1 秒完成。**关键设计**：ILP 允许 Attention 和 Expert 模块使用不同策略，且 Expert 模块在 prefill 和 decode 可分别使用不同策略。
    3. **Dynamic Parallelism Transition Strategy**（解决 prefill→decode 策略切换开销）：Expert 层权重约占 90% 总参数，naive 权重重分布（AllGather/AllToAll）通信开销大。HAP 维护 INT4 per-group 量化的权重备份于 CPU memory，通过多 stream 异步上传 + GPU 端反量化恢复 BF16。过渡方案在仿真中选择 min(T_reshard, max(0, T_upload + T_dequant - (T_attn + T_experts + T_comm)))——当上传+反量化时间小于当前层计算时间时，过渡开销可被完全隐藏。
  - 全栈执行例子（HAP, Mixtral-8x7B, 4×A6000, 4096in+64out，与 baseline 同配置对比）：
    - **模型推理算法层**：与 baseline 相同（Mixtral-8x7B, top-2 routing），不修改模型架构、gate 逻辑或计算精度。
    - **系统框架层**：基于 DeepSpeed-FastGen 修改。初始化：microbenchmark → 仿真模型训练 → ILP 搜索。执行：prefill 阶段 Attention DP=4（各 GPU 独立，无通信）+ Expert EP=4（All-to-All dispatch/combine，通信量 ∝ batch×hidden×seqlen）；prefill→decode 过渡：触发 INT4 权重上传 + GPU dequant（与当前层计算重叠）；decode 阶段：Attention DP=4 + Expert TP=4（各 GPU 持有完整 expert 的 1/4 中间维度，AllReduce 聚合，单 token 通信量小）。对比 baseline 的关键差异：(a) Attention 使用 DP 而非 TP，prefill 无 AllReduce 通信；(b) Expert 在 prefill 用 EP（减少通信量，无 AllReduce 的 2× AllGather 开销），在 decode 用 TP（消除 EP 的负载不均衡）。
    - **编译框架层**：论文未明确说明。PyTorch + DeepSpeed-Inference 标准栈。
    - **kernel 调度层**：prefill 期间 All-to-All dispatch/combine（NCCL）替代 baseline 的 AllReduce。decode 期间 AllReduce 通信量极小（单 token）。过渡期 CPU→GPU async copy（cudaMemcpyAsync）+ GPU 端 dequant kernel 与 attention 计算在独立 CUDA stream 上重叠。
    - **硬件架构层**：与 baseline 相同（4×A6000 PCIe）。结果：1.68× speedup。关键硬件利用策略：PCIe 低带宽（≤32GB/s）下避免 TP 的大通信量 AllReduce，改用 DP（无通信）+ EP（All-to-All 通信量更低）。NVLink 高带宽下 TP 通信开销可接受，HAP 可能仍选 TP。
  - **关键性能对比**：
    | Scenario | Hardware | Model | HAP vs TP Speedup | 策略选择 |
    |----------|----------|-------|-------------------|---------|
    | 4096in+64out | 4×A6000 | Mixtral | 1.68× | Attn:DP, Exp prefill:EP, Exp decode:TP |
    | 4096in+64out | 4×A100 | Qwen1.5-MoE | 1.77× | 低带宽硬件上 HAP 优势更大 |
    | 256in+64out | 4×A6000 | Mixtral | 1.13× | TP 在短上下文下已接近最优 |
    | 4096in+2048out | 4×A100 | Mixtral | 1.13× | decode 占主导时加速有限 |
    | 2048in+64out | 8×V100 | Mixtral | 1.57× | V100 PCIe 上 HAP 同样有效 |
    | 256in+2048out | 4×A6000 | Qwen2 | 1.01× | decode 主导场景，TP 已最优 |

  - **核心设计洞察**：HAP 的本质是将 MoE 推理的并行策略选择从"固定配置"问题重新定义为"带约束的最优化搜索"问题。其核心创新在于**层次化分解**——将 MoE 模型分解为 Attention/Expert 两个计算特征迥异的模块，允许它们使用不同的并行策略；将推理过程分解为 prefill/decode 两个通信-计算比例相反的阶段，允许 Expert 模块在阶段间切换策略。这种分解的粒度和可组合性使得策略的搜索空间从 TP/EP 的二选一扩展为 "Attention(prefill)×Expert(prefill)×Expert(decode)" 的组合空间，ILP 求解器在其中找到真正的最优解。对比 Fiddler（CPU-GPU 协同）、FloE（on-the-fly 内存管理）等 offloading 方法，HAP 面向的是同一类问题（资源受限下的 MoE 推理加速）但采用完全不同的路径——不改变计算的位置（GPU vs CPU），而是优化计算的并行方式（TP/DP/EP 组合）。二者是互补的：Fiddler/FloE 适合"设备显存放不下模型"的场景，HAP 适合"模型能放进多 GPU 但并行策略不是最优"的场景。
