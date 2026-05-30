## Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

- baseline方法是什么？
  Baseline 是**传统的蒸馏小模型部署在 CPU 上的多语言机器翻译系统**。具体流程：
  1. 对每个语言对（如 EN→DE, EN→FR），分别训练一个 teacher 模型（大而准确）
  2. 将每个 teacher 模型蒸馏为一个小 student 模型（约 40M 参数，精度较低）
  3. 将每个小 student 模型部署在 CPU 上（AVX512），独立服务每个语言对
  4. 对 100 种语言，需训练+蒸馏+部署至少 200 个独立模型

  全栈执行例子（以 EN→DE 翻译，CPU 部署 40M 小模型为例）：
  - **模型推理算法层**：小模型（0.04B 参数）经 teacher-student 蒸馏压缩，FP32 dense Transformer。知识迁移不充分（teacher→student 信息瓶颈），各语言对独立训练无法共享跨语言知识。
  - **系统框架层**：每个模型独立部署在 Azure F16s CPU 实例上（AVX512），batch_size=1，latency 75ms，throughput 351 words/sec，月成本 $0.209/token。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：CPU 上的标准 Transformer 推理（无 GPU kernel 优化），MoE 无法在 CPU 上高效运行（5.32B MoE 模型在 CPU 上 throughput 仅 26 words/sec，latency 1080ms，月成本 $22.6/token — 不可接受）。
  - **硬件架构层**：Intel Xeon CPU（AVX512），无 GPU 加速。限制：单 CPU 内存带宽不足以支持 5B+ MoE 模型的 expert 权重加载，导致 30×+ 的推理减速（vs dense counterpart）。

  Baseline 的核心缺陷：(a) **可扩展性差**——每增加一种语言需新增一个模型，训练-蒸馏-部署 pipeline 呈线性增长；(b) **精度受限于蒸馏**——teacher→student 的知识迁移有信息瓶颈；(c) **无法利用跨语言迁移学习**——每个语言对独立训练，无法共享表示；(d) **内存瓶颈**——大规模 MoE 模型无法在 CPU 上运行（内存带宽和 FLOPs 均不足）；(e) **GPU 上 MoE 推理也无优化**——naive PyTorch MoE 推理比同等 dense 模型慢 30×（token routing overhead + expert weight loading memory bandwidth bound）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**一套完整的 MoE 推理优化框架，包括 GPU kernel 优化、量化、batch pruning，并结合 FasterTransformer + Triton Inference Server 实现生产级部署**，具体设计：

  1. **单 GPU 部署大 MoE 模型替代数十个小模型**——将 100+ 语言对的小 CPU 模型替换为单个 5.32B MoE 模型运行在 1 张 T4 GPU 上，利用 MoE 的 sub-linear FLOPs 特性和多语言迁移学习同时提高翻译质量和部署效率。

  2. **GPU-optimized Token Routing（CUB radix sort + CUTLASS Grouped GEMM）**——解决 naive PyTorch MoE 的 token routing 开销（30× 慢于 dense）。将 MoE dispatch 从 Python 循环改为 CUB radix sort + permute + CUTLASS Grouped GEMM 单 kernel launch，所有 experts 并行计算。

  3. **4-bit/8-bit Weight-Only Quantization + Fused GEMM+Dequantize**——解决 MoE expert 权重过大（>90% 总参数）导致的内存容量和带宽瓶颈。仅量化 expert weights（activations 保持 FP16），per-channel 对称量化，无需 QAT 或 calibration。Fused dequantize 进 GEMM kernel 避免额外内存读写。用 FP16 bit-trick 替代原生 I2F 加速 dequantize。INT4 实现 8× 模型压缩（5B→~625MB expert weights），和最高 1.85× GEMM 加速（32 active experts）。

  4. **Batch Pruning**——解决 decoder beam search 中部分句子完成后仍加载所有 expert weights 的浪费。已完成句子在 gating 阶段被路由到末尾（expert_idx → INT_MAX），后续 expert GEMM 仅处理 active_tokens 行，避免加载无需计算的 expert 权重，1.14× 加速。

  5. **Triton Inference Server 集成**——解决生产环境的动态扩缩容问题（vs Baseline 每个语言对独立 CPU 实例无法弹性调度）。

  全栈执行例子（以 EN→DE 翻译，T4 GPU，INT4 MoE 模型为例）：
  - **模型推理算法层**：单个 5.32B 参数 MoE encoder-decoder（32 experts, top-1 gating, TUPE attention）替代 200+ 个 0.04B distilled 模型。MoE 通过 top-1 gating 为每个 token 选择 1/32 专家（约 166M 激活参数），sub-linear compute cost。Quantization 仅量化 expert FFN weights（INT4 per-channel 对称量化，FP16 scales），activations 保持 FP16，无需 QAT。BLEU 质量接近 FP16 baseline（10 语言对平均 ∆BLEU = -0.167 INT4 vs FP16）。
  - **系统框架层**：NVIDIA FasterTransformer（扩展支持 MoE）+ CUTLASS Grouped GEMM + CUB radix sort → 编译为 CUDA binary → Triton Inference Server 管理模型生命周期与 dynamic batching。Decoder beam search 中 batch pruning 动态移除已完成句子。月成本 $0.153/token（T4 batch_size=64），低于 Baseline $0.209/token（CPU batch_size=1）且 BLEU 更高（大模型质量优势）。
  - **编译框架层**：论文未明确说明。nvcc + gcc/g++ 9.3 编译 CUDA kernels。
  - **kernel调度层**：(a) Token Routing: CUB DeviceRadixSort 按 expert_idx 排序三元组 → gather permute activation rows → 计算各 expert 子矩阵 offsets；(b) Expert GEMM: CUTLASS Grouped GEMM 单 kernel launch 并行执行各专家 matmul；(c) Fused Dequantize: GEMM kernel weight load 阶段，4 个 INT8 → 1×32-bit reg → `0x6400 | val` → FP16 sub 1152 → × scale → FP16 weight → Tensor Core GEMM。INT4 layout 重排 `[e0,e1,e2,e3,e4,e5,e6,e7]→[e0,e2,e4,e6,e1,e3,e5,e7]` 减少 bit 操作。32 active experts 下 INT4 GEMM 达 1.85× FP16 GEMM 速度（Table 1）。对比 Baseline naive PyTorch：torch-FP16 batch_size=1 throughput=16 tokens/sec → FT-INT4 batch_size=1 throughput=400 tokens/sec（25×）。
  - **硬件架构层**：单卡 NVIDIA T4（16GB HBM, Turing Tensor Cores）。5.32B 模型 FP16 需约 10GB，INT4 expert weights 仅约 1.25GB（8× 压缩），轻松放入 T4 16GB 内。V100 用于 Kernel 开发和评估。单 GPU 避免 all-to-all 网络通信开销。

  **Baseline 缺陷 → 方法映射表**：
  | Baseline 缺陷 | 论文方法 |
  |--------------|---------|
  | 每语言需独立模型 | 单 MoE 模型服务全部 100+ 语言对 |
  | 蒸馏信息瓶颈 | 大模型直接推理，无需蒸馏 |
  | CPU 内存带宽不够跑 MoE | GPU (T4) + 量化 (INT4) 压缩 8× |
  | Naive PyTorch MoE 比 dense 慢 30× | CUB radix sort + CUTLASS Grouped GEMM 并行化 |
  | Expert 权重内存带宽瓶颈 | INT4/INT8 量化 + fused dequantize 减少内存传输 |
  | Decoder 浪费算力在已完成句子 | Batch Pruning 动态移除 |
  | 无法弹性扩缩容 | Triton Inference Server |

## Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

- baseline方法是什么？
  Baseline 是**传统 MoE LLM 的分析与剪枝方法，将专家视为独立实体**。具体包括：(1) Router 分析——仅研究单个 router 的 top-k 选择行为（如输出 norm 偏好、token ID 关联），不揭示专家间的协作关系；(2) 独立专家剪枝——SEER-MoE 基于路由分数高低剪枝（Muzio et al., 2024），GEM 基于 |x-f(x)| 差异识别输出影响最小的专家（Zhang et al., 2024），Random 随机删除专家。这些方法均将每个专家独立评估，忽略了跨层专家之间的协作与互补关系。

  全栈执行例子（以 DeepSeek-MoE-16B 上 GEM 剪枝 25% 专家为例）：
  - **模型推理算法层**：对每个 MoE 层的每个专家独立计算其对最终输出的影响度（|x - f_expert(x)|），按影响度排序后删除影响最小的 25% 专家，不考虑跨层专家共激活模式。剪枝后剩余专家保留原始路由权重，Router 仍按 top-k 选择（可能选到对特定任务已不完整的专家组合）。在 HellaSwag 上 accuracy 从约 0.69 降至 0.658。
  - **系统框架层**：基于 HuggingFace Transformers 加载 DeepSeek-MoE-16B，通过直接修改模型权重（删除 experts）实现剪枝。评估使用 EleutherAI LM Harness 框架，normalized zero-shot accuracy。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出**基于层级稀疏字典学习（HSDL）发现专家协作模式，并基于协作模式进行贡献感知专家剪枝（CAEP）**的方法，通过两个关键设计解决 Baseline 缺陷：

  1. **HSDL 揭示跨层专家协作模式**——不同于 Baseline 将专家视为独立实体，HSDL 从 expert activation matrix X 出发，通过多层稀疏字典学习递归分解，发现跨层专家之间的共激活协作模式（如 Layer 5 Expert 21 + Layer 6 Expert 3 频繁同时激活）。字典每个 atom 编码一组协作者，稀疏编码 R 控制各模式在不同样本上的参与度。实验验证：(a) 60% 的字典模式对应穷举搜索中 Top 10% 最高频组合；(b) 层级语义标注显示高层字典捕获粗粒度类别（"数学计算"），深层字典细化为子任务（"日期/符号识别"）；(c) 语义相近领域（数学/物理/计算机科学）的专家激活分布相似度高，语义不同的领域（数学/法律）分布差异大。

  2. **CAEP 基于协作模式贡献剪枝**——不同于 Baseline 按单个专家的路由分数或输出影响力独立排序，CAEP 结合字典矩阵 D 和稀疏编码 R 计算每个专家的综合贡献分数 e = Σ D_sum[:,i]，在迭代中优先移除最少被使用的协作模式（pattern），而非仅按个体分数截断。这确保了剪枝后保留的专家仍然形成完整的协作组合，维持任务处理能力。实验表明：CAEP 在剪枝 25% 专家后平均 accuracy 0.612，优于 SEER-MoE (0.5872) 和 GEM (0.5870)，如在 OBQA 上从 0.420 提升至 0.473，HellaSwag 上从 0.658 提升至 0.691。

  全栈执行例子（以 DeepSeek-MoE-16B 上 CAEP 剪枝 25% 专家为例）：
  - **模型推理算法层**：Step 1——在 MMLU 128 个样本上收集 expert activation matrix X（每个 token 的 router 分配 α 按句子求和 v_{i,j,k} = Σ α(i)_{t,j,k}）。Step 2——HSDL 多层分解 X 得到字典 D（编码跨层专家协作模式）和稀疏编码 R。Step 3——计算每个专家的贡献分数 e，取 k_1-分位数作为阈值生成初始 mask，迭代移除最少使用的 pattern 并更新 mask 直至达到 25% 剪枝比。与 GEM 不同，CAEP 保留了如 {L5-E21, L6-E3} 等共激活组合中的双方专家，避免因单独删除一个专家而破坏协作模式。剪枝后模型在 HellaSwag 上 accuracy 为 0.691（vs GEM 0.658），OBQA 上 0.473（vs SEER-MoE 0.420）。
  - **系统框架层**：基于 HuggingFace Transformers 加载 DeepSeek-MoE-16B，通过 mask vector m ∈ {0,1}^{N_e} 标记保留/删除的 normal experts（shared experts 保留），剪枝后参数量 = 16.4 - 14.7 × k% B（式(10)）。评估使用 EleutherAI LM Harness 框架。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

## Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

- baseline方法是什么？
  Baseline 是传统的 **稠密（Dense）统一多模态大语言模型**，如 Macaw-LLM、X-InstructBLIP，以及单专家 Dense 模型（Single-Modality-Expert）。这些模型在处理每种输入时激活全部参数（稠密计算），导致训练和推理的计算开销随模型规模和多模态数据种类增加而成比例增长。全栈执行例子（以 X-InstructBLIP 处理"语音提问+图像"三模态输入为例）：
  - **模型推理算法层**：语音编码器（Whisper）+ 图像编码器（CLIP）分别编码，通过 Connector 映射到 LLM 语言空间，所有 tokens 串联后送入 LLM 的每一层，每一层的 FFN 均为稠密计算——所有参数被激活处理每个 token。多模态数据混合训练时，Dense 模型一个专家/MLP 需要同时学习图像、语音、文本等所有模态的表示，容易产生模态间干扰，训练 loss 波动大，且对长语音等复杂模态的外推泛化能力差。
  - **系统框架层**：标准 PyTorch + HuggingFace Transformers 训练栈，数据并行（DP）训练。多模态多任务数据混合训练时所有样本经过相同模型参数，无专家级模型并行和模态级数据并行。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Uni-MoE**——基于稀疏 MoE 架构的统一多模态 LLM，通过三个关键设计解决 Baseline 缺陷：
  1. **稀疏 MoE 替代稠密 FFN**——将 LLM 中部分层的稠密 FFN 替换为含 4~8 个专家的稀疏 MoE 层，每个 token 仅激活 top-2 专家（激活参数远小于总参数），大幅降低推理计算开销。例如 Uni-MoE-7B×4-Top2 激活 8.9B 参数但总参数 13.2B，相比 Dense 7B 模型仅增加约 2.2B 激活参数即可处理 5 种模态。
  2. **模态特定专家预训练（阶段二）**——每个专家在不同模态数据上分别预训练（如 Expert 2 用文本-图像数据，Expert 3 用语音-图像数据，Expert 4 用纯音频数据），使各专家发展出模态偏好。Router 在学习过程中能自动将不同模态的 tokens 路由到对应专业专家（如音频 tokens → Expert 4，图像 tokens → Expert 2），解决 Dense 模型中单一 FFN 需同时学习所有模态导致的模态间干扰。
  3. **LoRA 微调 + 专家级模型并行**——在阶段三使用 LoRA（rank=8）微调预训练专家和自注意力层，冻结专家本体参数，仅更新低秩适配器和 Router。同时实现专家级模型并行（expert-level model parallelism）和模态级数据并行（modality-level data parallel），使训练可扩展到多节点多 GPU。
  
  全栈执行例子（以 Uni-MoE MoE-Task3 处理"视频+音频+文本"为例，含 4 个预训练专家）：
  - **模型推理算法层**：视频 8 帧通过 CLIP-V 编码后平均池化→视觉 tokens；音频通过 BEATs 编码→Audio-QFormer（4 层 cross-attention 蒸馏）→音频 tokens；文本通过 Word-Embedding→文本 tokens。三类 tokens 串联后输入 LLM。在 MoE 层中，Router 对每个 token 计算 softmax(W_router · x)，选择 top-2 专家。可视化分析（Figure 4-5）显示：音频 tokens 主要由 Expert 4（音频预训练）处理，图像 tokens 主要由 Expert 2（图像-文本预训练）处理，视频的多模态 tokens 在后期层由多专家协作处理。相比 Dense 模型，MoE 路由实现了模态感知的负载分配，使得长语音理解（RACE-Audio high）从 Dense 的 29.02% 提升至 49.37%，且在混合多模态数据上训练 loss 更稳定收敛（Figure 3 蓝色线）。
  - **系统框架层**：基于 PyTorch 分布式训练，实现数据并行（modality-level data parallelism——不同模态数据在不同 device 上处理）+ 专家并行（expert-level model parallelism——不同专家分布在不同 GPU 上）。阶段三支持多节点多 GPU（16×A800）训练 8 专家模型。论文公布了两种分布式并行训练方法。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

## UCCL-EP Portable Expert-Parallel Communication

- baseline方法是什么？
  Baseline 是 **DeepEP**（DeepSeek, 2025），基于 NVIDIA IBGDA（InfiniBand GPUDirect Async）的 GPU-initiated token-level EP 通信系统。全栈执行例子：
  - **模型推理算法层**：MoE layer 中 gating network 为每个 token 选择 top-K experts，需要执行 dispatch（token activations 发送到 expert GPUs）和 combine（expert outputs 收集回原 GPU）。
  - **系统框架层**：DeepEP 被集成到 SGLang、vLLM、Megatron-LM 等训练/推理框架中，通过 NVSHMEM 提供 symmetric memory 和 GPU-initiated one-sided RDMA 操作。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：GPU threads 直接通过 IBGDA 接口向 NVIDIA ConnectX NIC 的 MMIO doorbell/register 写入 RDMA work requests，实现 token-level fine-grained 通信。LL mode 为 per-token immediate send，HT mode 使用多 ring buffer 实现 token deduplication + intra-node forwarding + hierarchical reduce。GPU kernel 假设 NIC 提供严格的 ordering 保证（如 write-then-atomic 语义），所有 transfer 完全 bypass CPU。
  - **硬件架构层**：NVIDIA GPU（H100/H200/B200）+ NVIDIA ConnectX-7 InfiniBand NICs（400G）。IBGDA 要求 GPU 能直接写入 NIC driver 定义的 MMIO 接口，因此仅支持 NVIDIA GPU + NVIDIA/Mellanox NIC 组合。

  Baseline（DeepEP）的核心缺陷：
  (1) **GPU-NIC 紧耦合导致可移植性差**：IBGDA 要求 GPU 直接操作 NIC MMIO doorbell/register，这意味着需要分别为每一种 (GPU vendor, NIC vendor) 组合编写集成代码。假设 m 种 GPU, n 种 NIC，需 O(m×n) 开发工作量。实际结果是 DeepEP 官方仅支持 NVIDIA GPU + NVIDIA NICs。
  (2) **GPU 对 NIC delivery semantics 的刚性假设**：GPU kernel 假设下层的 NIC 提供 strict ordering（如 write-then-atomic），但许多 NIC（如 AWS EFA SRD 协议）不保证 ordering，导致 DeepEP 无法在此类 NIC 上正确运行。
  (3) **GPU 缺少灵活的网络管理能力**：GPU threads 难以实现 congestion control、flow control、failure recovery 等高级网络策略。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **UCCL-EP**，通过将"通信发起"与"通信执行"解耦来实现可移植的高性能 EP 通信。全栈执行例子：
  - **模型推理算法层**：保持 MoE dispatch/combine 计算逻辑不变（数学等价），GPU 仍负责 token-level routing decisions 和 fine-grained communication initiation。
  - **系统框架层**：UCCL-EP 作为 DeepEP 的 API-compatible drop-in replacement，无需修改上层框架代码。移除 NVSHMEM 依赖，通过 CPU proxy 管理 symmetric memory。支持 SGLang（推理）和 AMD Primus/Megatron-LM（训练）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：核心创新——GPU-CPU 解耦通信架构：
    - **(a) Lock-free GPU-CPU FIFO channel**：GPU threads 将 token routing 决策编码为 128-bit TransferCmd（Write/Atomics/Drain/Barrier），通过 shared memory FIFO 队列传递给 CPU proxy。GPU 侧缓存 tail index，CPU 和 GPU 分别将各自常用元数据置于本地内存侧，减少 PCIe 穿越。多 FIFO channels 映射减少 GPU SM 竞争。
    - **(b) Multi-threaded CPU proxy**：每个 GPU 分配一个 CPU proxy（4 threads），通过 libibverbs（Linux 可移植 RDMA 库）发出 GPUDirect RDMA 操作。CPU proxy 负责 QP load balancing、多 NIC bandwidth aggregation、connection management。
    - **(c) Ordering emulation via immediate data**：发送端在每个 RDMA write 的 immediate data（32-bit RoCEv2 标准包头字段）中嵌入 sequence number + expert index。接收端 CPU proxy 从 CQ 提取 immediate data，若消息 out-of-order 到达，将 atomic 暂存于 control buffer，待所有 prior writes 完成后再有序 apply。实现了 partial ordering（per-channel）而非全局 ordering，避免硬件成本。
    - **(d) Software atomics on EFA**：在 EFA 等不支持硬件 RDMA atomics 的 NIC 上，CPU proxy 通过 immediate data write + host memory counter update 模拟 atomics，GPU 直接读取 host-allocated memory（cudaMallocHost）用于 control decisions。
    - **(e) AMD GPU porting**：将 CUDA PTX intrinsics → ROCm alternatives；warp (WARP_SIZE=32) → wavefront (WAVEFRONT_SIZE=64)；TMA-based copy → CU-based copy；merge coordinator wavefronts into receiver wavefronts。
  - **硬件架构层**：支持 NVIDIA GPU（H100/H200/B200/GH200）+ AWS EFA NICs（SRD unordered transport）、NVIDIA GPU + ConnectX-7 IB、AMD MI300X + ConnectX-7 IB、AMD MI300X + Broadcom Thor-2 等多种异构组合。仅需 O(m) 移植工作（GPU kernel 变化），CPU-NIC 侧通过 libibverbs 可移植层自动适配。

  **设计思路核心映射**：
  - 缺陷(1) "GPU-NIC 紧耦合 O(m×n) 移植成本" → 方案：解耦 GPU 通信发起与 CPU 通信执行，CPU 通过 libibverbs 可移植层适配任意 NIC → O(m) 移植成本（从 AMD GPU 到 Broadcom NIC 仅需修改 GPU kernel，CPU 侧无需额外适配代码）
  - 缺陷(2) "GPU 对 NIC ordering 的刚性假设" → 方案：CPU proxy 使用 RDMA immediate data 嵌入 sequence number + control buffer 延迟 apply → 在 EFA（无序传输）上正确运行，无需 NIC 硬件支持 ordering
  - 缺陷(3) "GPU 缺少网络管理灵活性" → 方案：CPU proxy 可实现 congestion control（控制 kMaxInflight 限制 in-flight 消息数）、多 QP 负载均衡、failure recovery（elastic EP）等策略
  - 最终效果：UCCL-EP 在 EFA 上 dispatch/combine 吞吐量超过最佳现有方案（PPLX）最多 2.1×；在 NVIDIA-only 平台上性能与 DeepEP 原版可比（HT mode dispatch 延迟差异 <5%）；SGLang 推理吞吐提升最多 40%（NV_EFA3），Megatron-LM 训练吞吐提升最多 45%（AMD_BRC）。

## Tutel Adaptive Mixture-of-Experts at Scale

- baseline方法是什么？
  Baseline 是 **Fairseq MoE / DeepSpeed MoE 的静态执行框架**（Ott et al., 2019; Rajbhandari et al., 2022），遵循 GShard 计算逻辑（Lepikhin et al., 2021）。全栈执行例子：
  - **模型推理算法层**：GShard 风格的 Top-K 稀疏门控 MoE。Token t 经 gate = Softmax(W_g · x_t) 计算 E 个专家的路由概率，TopK 选出 K 个专家，dispatch → expert FFN → combine。Baseline 使用静态 capacity factor f = f_upper（固定上界），导致：(a) f 偏大时浪费计算；(b) f 偏小时丢弃 token。
  - **系统框架层**：Fairseq/DeepSpeed MoE 采用固定并行策略（如 EP+DP），运行时不可切换。切换到不同并行策略需要：不同的张量分片布局、参数迁移开销（Figure 4）、框架接口变更。并行策略在所有训练步中保持静态，不适应动态变化的 expert capacity。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 NCCL 的 Linear All-to-All（Algorithm 1），所有 GPU 直接 P2P 通信。Dispatch/Combine 的 encode/decode 使用稠密 einsum 实现（Figure 20a），时间复杂度 O(T·E·C_g·D)，内存消耗大（Table 5: 32,768 tokens/step 时 57.9 GiB）。All-to-All 与 Expert FFN 顺序执行，无通信计算重叠。
  - **硬件架构层**：NVIDIA A100 80GB GPU + HDR InfiniBand。Linear All-to-All 在 scale-out 时消息大小 S/n 变得过小，无法饱和 InfiniBand 链路带宽（Figure 16），All-to-All 通信开销占比从 16 GPUs 的 33.7% 增长到 256 GPUs 的 56.7%（Table 2）。

  Baseline 的核心缺陷：
  (1) **静态并行不适应动态负载**：MoE 的 expert capacity 随训练步动态变化（实测 4.38× 波动，Figure 1），而 Fairseq/DeepSpeed 固定使用一种并行策略，不同并行策略在不同 capacity 下有 7.39%~27.76% 的性能差距（Figure 3）。
  (2) **静态流水线度/All-to-All 算法低效**：不同 scale 和模型配置的最优流水线策略不同（Figure 5），静态策略导致最坏情况下 23%~599% 的性能损失（Table 6b）。
  (3) **密集 encode/decode 计算冗余**：稠密 einsum 包含大量零乘加，且消耗大量 GPU 内存。
  (4) **Linear All-to-All 不可扩展**：大规模下小消息无法饱和网络带宽。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **TUTEL 自适应 MoE 全栈系统**，通过统一张量布局实现零成本并行切换、自适应流水线、稀疏 GPU kernel 和层次化 All-to-All。全栈执行例子：
  - **模型推理算法层**：保持 GShard 计算逻辑不变（数学等价），支持动态 capacity factor（capacity_setting 参数控制：正值=固定值，0=自适应最小不丢 token，负值=带上限自适应，Figure 10），以及动态 Top-ANY 路由（每步可调整 k 值）。
  - **系统框架层**：自适应并行切换——基于 ZeRO-DP Stage-3 风格的统一张量分片布局，DP（r=0）和 EP+DP+MP（r∈[1,⌈W/E⌉]）共享相同的 weight slicing 和 data layout（Figure 6/7/8）。运行时通过 O(1) 字典查表选择最优 r，无需参数迁移或数据重整（零成本切换）。通过通信复杂度分析（Table 4）将 7 种并行策略化简为 2 种。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：(a) Fast Encode/Decode（K0/K1/K2 CUDA kernels）——稀疏 SIMT-efficient 实现，每个 warp 处理一个 token，利用 warp shuffling、Blelloch scan、half2 向量化，复杂度从 O(T·E·C_g·D) 降至 O(T·k·D)，GPU 内存节省 20%~90%（Table 9）；(b) Flexible All-to-All——输出 layout 从 (W, E_g, C_g, D) 变为 (E_g, C, D)，消除 scale 对 expert matmul 的影响（Figure 11）；(c) 2DH All-to-All——4-phase 算法（stride memcpy → intra-node A2A → stride memcpy → inter-node A2A, Figure 17），聚合小消息为大消息，在大规模下大幅降低延迟（Figure 18），支持 MSCCL 编译优化和 LL128 协议；(d) 自适应多流流水线——token capacity 维度分区 + 多 CUDA stream 异步执行，重叠 All-to-All 通信与 Expert FFN 计算，动态选择流水线度 d∈{1,2,4,8} 和算法 a∈{Linear,2DH}。
  - **硬件架构层**：NVIDIA A100 GPU + HDR InfiniBand，NCCL 2.10.3-1 + RDMA SHARP plugin。2DH All-to-All 通过聚合小消息提升 InfiniBand 带宽利用率。

  **设计思路核心映射**：
  - 缺陷(1) "静态并行" → 方案：统一张量布局 + 零成本自适应切换 → 1.35×~14.57× MoE 层加速
  - 缺陷(2) "静态流水线" → 方案：字典式最优策略查找 + 多流异步调度 → 平均 9%~101% 提升，最坏情况 23%~599% 提升
  - 缺陷(3) "密集 encode/decode" → 方案：SIMT-efficient 稀疏 CUDA kernel (K0/K1/K2) → kernel 加速 + 20%~90% 内存节省
  - 缺陷(4) "Linear A2A 不可扩展" → 方案：2DH 层次化 All-to-All + Flexible layout + MSCCL 编译优化 → 大规模下 4.25× 提升（2,048 GPUs）
  - 最终效果：2,048 GPUs 上单 MoE 层 5.75× speedup（vs Fairseq），SwinV2-MoE 端到端训练 1.55× 推理 2.11× 加速

## TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation

- baseline方法是什么？
  Baseline 是标准 **Top-K (K=2) 路由的稀疏 MoE 训练**（Fedus et al., 2022; Zoph et al., 2022）。在 Baseline 中：
  - **模型推理算法层**：输入 token x 通过路由器计算 Softmax(Wx) 得到 N 个专家的概率分布 π，TopK 选出 K 个最高概率的专家，MoE 输出为 $y = \sum_{i \in A} \pi_i E_i(x)$。反向传播时，由于 Top-K 是不可微操作，使用 Straight-Through Estimator 绕过，理论上需要所有 N 个专家的输出来计算稠密梯度 $\partial y/\partial \pi = [E_1(x), E_2(x), \dots, E_N(x)]$，但实际只有 K 个专家有输出，导致只有 K/N 比例的专家参数和路由嵌入被更新。
  - **系统框架层**：基于 GPT-NeoX 训练框架 + Megablocks 稀疏专家库，支持数据并行训练，通过 NCCL 进行通信。每个 token 只激活 K 个专家，其余 (N-K) 个专家的参数和路由嵌入对该 token 无梯度更新。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准的 MoE 前向/反向 kernel（Megablocks 提供），包含 expert matmul、all-to-all 通信、路由器 softmax + TopK 等操作。
  - **硬件架构层**：标准 NVIDIA GPU 集群，论文未明确说明具体型号。

  Baseline 的核心缺陷：(1) **路由器接收稀疏梯度**——只有被 Top-K 选中的 K/N 比例的专家嵌入行获得梯度，未被选中的专家对应的路由嵌入 $W_i$ 得不到更新，导致路由器无法学习到所有专家的路由分布；(2) **专家负载不均衡**——稀疏梯度导致部分专家（hot experts）被过度使用，部分专家（cold experts）几乎空闲，资源利用率低；(3) **专家参数更新稀疏**——每个 token 只更新 K 个专家的参数，(N-K) 个专家对该 token 无任何更新，参数利用效率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Expert Group Approximation（专家组近似）** 方法，核心思路是在反向传播中用已有专家输出来近似未激活专家的输出，从而为路由器和专家参数提供稠密梯度信号：

  - **模型推理算法层**：不改变前向传播（保持推断时的稀疏性），仅在反向传播中通过以下机制引入稠密梯度：
    (1) 将所有 token 按路由决策 $R(x)$ 分组（共 $\binom{N}{K}$ 组）；
    (2) 对于每组 $X_R$ 中的 token x 和每个未激活专家 i ∉ R，用同时被路由到 i 和 x 的某个激活专家 j ∈ R 的其他 token 的 $E_i$ 输出来近似 $E_i(x)$：
    $\hat{E}_i(x) = \frac{1}{K} \sum_{j \in R} \frac{1}{|X_{\{i,j,\cdot\}}|} \sum_{x' \in X_{\{i,j,\cdot\}}} E_i(x')$
    (3) 通过 stop-gradient 将近似注入计算图：$y := y + y' - \operatorname{sg}(y')$，前向不变，反向有完整梯度；
    (4) 同时更新路由器（所有 N 行嵌入接收梯度）和专家参数（所有专家对近似产生贡献的 token 接收额外梯度更新），两者通过 all-reduce 跨数据并行 worker 聚合。
  - **系统框架层**：基于 GPT-NeoX + Megablocks，在 MoE 层插入 Expert Group Approximation 计算和梯度注入逻辑。通过数据并行 all-reduce 聚合近似梯度，利用大全局 batch size 提升近似的统计质量。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 Triton 实现自定义 kernel（"Router backward" kernel 等）高效执行 token 分组、近似构造和梯度聚合。随 hidden size 增大（1024→4096），方法 overhead 从 13.32% 降至 1.57%（Table 4）。在 multi-node 训练中通信 overhead 主导时，方法 overhead 趋近于零。
  - **硬件架构层**：NVIDIA GPU（具体型号论文未明确说明）。

  **设计思路核心映射**：
  - 缺陷(1) "稀疏路由器梯度" → 方案：通过 Expert Group Approximation 用 N² 个组近似填充所有 N 个专家的梯度分量，使 $\partial y/\partial \pi$ 从稀疏 [K 个非零] 变为稠密 [N 个非零]
  - 缺陷(2) "负载不均衡" → 方案：路由器接收稠密梯度后能更好地学习 token-专家匹配分布，实验证明最大负载不均衡显著降低（Figure 7）
  - 缺陷(3) "专家参数更新稀疏" → 方案：近似梯度同时回传给参与近似的专家参数，使每个 token 对 (N-K)/N 比例的额外专家产生梯度贡献，专家参数利用率从 K/N 提升至接近 N/N

## Towards MoE Deployment Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

- baseline方法是什么？
  Baseline 是 fairseq [23] 中基于 **Static Gating（静态容量门控）** 的 MoE Transformer 推理部署方案。Baseline 的核心缺陷：(1) **静态容量造成巨大 waste factor**——对于 LM（E=512, C=0.05），每个 token 需要 top-2 gating（仅 2 个专家真正计算），但每个专家固定处理 ECS = 512×0.05×S = 25.6S 个 token，waste factor = 12.8×；对于 MT（E=128, C=1），waste factor = 64×；(2) **dispatch mask 矩阵乘法的内存开销**——每次 MoE 层都构造大小为 (E, S, S×C) 的稀疏 dispatch mask，通过 batch 矩阵乘法（BMM）将 tokens 分发到各专家，LM batchsize=8 时激活内存高达 6.29GB；(3) **专家参数完全驻留 GPU**——所有 512 个 expert FFN（LM 52B 模型）的参数全部占用 GPU HBM，静态内存达 18.88GB（单 GPU），限制了 batch size 扩展；(4) **专家负载极度不均衡**——hot experts 承载大量 tokens，cold experts 几乎空闲，且某些 experts 完全不被激活（MT Decoder 约 75% experts 不活跃），但不活跃的 experts 仍占据 GPU 内存并处理空 placeholder。

  Baseline 全栈执行例子（以 LM 推理，batch_size=8，单节点 8×V100 为例）：
  - **算法层**：Transformer decoder-only LM，24 layers，每 MF=2 层中 1 层为 MoE 层（含 512 个 expert FFNs）。Gating 为 top-2，static capacity C=0.05。token emb → MHA → Static Gating → Dispatch Mask BMM → All-to-All (fixed size) → Expert FFN → All-to-All → Combiner BMM。每 expert 固定处理 25.6S tokens，大部分为零填充（placeholder 向量），产生 token dropping 风险（超出容量的 token 被丢弃，仅靠残差连接保留信息）。
  - **框架层**：fairseq MoE Transformer + expert parallelism（每个 GPU 持有 64 个 experts for LM E=512/8 GPU）。PyTorch 实现，通信使用 NCCL all-to-all collective（单节点内 NVLink 300GB/s，多节点 InfiniBand）。
  - **编译框架层**：论文未明确说明（使用 PyTorch eager mode / fairseq 默认编译路径）。
  - **Kernel 层**：cuBLAS 矩阵乘法（expert FFN 的小 batch GEMM）、NCCL all-to-all（使用 NVLink/IB 的 RDMA 原语）、dispatch/combiner batch 矩阵乘法（cublasGemmBatchedEx 或 cublasGemmStridedBatchedEx）。大量时间消耗在 dispatch BMM 和 fixed-size all-to-all 上。
  - **硬件层**：NVIDIA V100 GPU（32GB HBM2），SM 执行 FFN 的 MAC 运算，HBM 存 expert 参数和中间激活，NVLink 互联用于 all-to-all 通信。CPU（Xeon E5）仅用于数据预处理，不参与推理计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文通过三项互补优化解决 baseline 缺陷：
  **(1) Dynamic Gating** 消除静态容量约束：用 argsort + 两阶段 all-to-all 替代 dispatch mask BMM。不再传输空 placeholder，只传输实际分配给 expert 的 tokens。直接解决了 waste factor（12.8× 或 64×）和 dispatch mask 的激活内存（LM batchsize=8 从 6.29GB 降至 1.28GB）。同时，token 不再被丢弃（因为不再有容量限制），提升了模型鲁棒性。
  **(2) Expert Buffering** 减少静态参数内存：利用 expert 激活的时序局部性（同一 expert 连续 batch 活跃），在 GPU 内存中只缓存 hot experts，其余放在 CPU 内存按需加载。通过 LIFO 淘汰策略 + Memcopy 与 all-to-all 通信 overlap 隐藏延迟。MT Decoder 静态内存减少 2.25GB（1.47× reduction）。
  **(3) Load Balancing** 改善设备间负载分布：基于历史激活数据，将 expert-to-GPU 映射形式化为 multi-way number partitioning（NP-hard），用 greedy 算法近似求解，将高负载和低负载 expert 混搭分配到各 GPU，减少瓶颈设备。LM 的 Max Load 从 0.6 降至 0.4 以下。

  论文方法全栈执行例子（以 LM 推理，batch_size=64，单节点 8×V100，所有优化启用为例）：
  - **算法层**：同样 Transformer decoder-only LM 结构。变化点：Gating → top-2 仍输出 expert assignments；**Dynamic Gating**：不再构造 dispatch mask，而是 argsort(flat_expert_indices) → 得到最优 token 排列 idx → bincount 统计每 expert token 数 → 两阶段 all-to-all（先传 sizes，再传 data）→ 各 GPU 的 expert FFN 只处理实际分配到的 tokens → index-based gather 恢复顺序。整个过程 O(S log S + SD)，远小于原方案 O(S²EDC)。**Expert Buffering**：接收到 all-to-all sizes 后，检查本 GPU 上的 active experts 是否在 GPU cache 中 → hit 直接使用，miss 触发 CPU→GPU Memcopy（与 all-to-all 并行）。**Load Balancing**：初始化时按 greedy 算法将 512 experts 分配到 8 GPU（每 GPU 64 experts），高负载+低负载混合。
  - **框架层**：仍是 fairseq，但 gating 模块改写：新增 DynamicGatingLayer（含 argsort、bincount、two-phase all-to-all wrapper）、ExpertCache（含 GPU buffer、CPU parameter store、LIFO eviction、Memcopy overlap 逻辑）、LoadBalancedExpertPlacement（含 greedy/anti-correlation assignment 模块）。Python+PyTorch+CuPy 实现。
  - **编译框架层**：论文未明确说明。
  - **Kernel 层**：核心变化——(a) dispatch/combiner 的 BMM 被 index-based gather/scatter 替代（大幅减少内存访问和计算量）；(b) all-to-all 通信量从固定 EC tokens 降至实际 token 分配量（waste factor 消除），通信量降至原方案的 1/12.8（LM）或 1/64（MT）；(c) 新增 CUDA Memcopy kernel 用于 CPU→GPU expert 参数传输，与 NCCL all-to-all 通过 CUDA stream 并行 overlap。
  - **硬件层**：同一 V100 平台。但 batch size 可从 8 扩展到 64（单节点 LM），得益于：(a) 动态内存从 6.29GB 降至 1.28GB；(b) 静态内存通过 Expert Buffering 降低 2.25GB（MT Decoder）。多节点时性能提升更显著（LM 多节点吞吐提升达 11.55×），因为减少的通信量在跨节点场景下影响更大。瓶颈从 GPU 内存容量和通信带宽转移到 CPU-GPU 带宽（Expert Buffering 的 Memcopy 路径），实测 CPU-GPU 带宽饱和在 12GB/s（PCIe 3.0）。

## Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

- baseline方法是什么？
  Baseline 是传统 transformer LLM（GPT-3、Llama4-Maverick）的 serving 部署，使用 MHA 或 GQA + dense FFN。Baseline 的核心缺陷：(1) **MHA/GQA 的核心注意力层 ArI 极低（≈1 Op/B）**——每个请求的 KV$ 和 attention score 值独占，无法跨批次请求共享，导致 decode 阶段始终 memory-bound，计算资源严重利用不足；(2) **大 KV$ 容量限制 batch size**——GPT-3 的 KV$ 单请求可达 9 GB（L=2048），限制了 $B_{\rm cap}$ 和 $B_{\rm SLO}$，使得 FC 层无法达到足够 batch size 来接近 ridge point，FC 层也处于 memory-bound；(3) **dense FFN 计算成本随参数量线性增长**——每个 token 需要计算所有参数，Scaling 受限。传统研究焦点：设计专用硬件（如 attention-specialized PIM）来缓解 attention 的内存带宽瓶颈。

  全栈执行例子（Baseline: GPT-3, 32 B200 GPU, L=4096, B=128）：
  - **算法Pipeline层**：MHA, 每 head 独立计算 Score = Q_i @ K_i^T → Softmax → Context = P_i @ V_i，ArI=1 Op/B（memory-bound）。Q/K/V 投影为 FC GEMM，无 KV 压缩。
  - **Serving/系统框架层**：Disaggregated prefill/decode, $deg_{\rm TP}=8$, $deg_{\rm DP}=4$。KV$ 每 token 4.5 MB(=12288×96×2×2B)，B=128 时 KV$ 总量 54 GB/GPU，接近 HBM 容量上限。无法扩大 batch size 来使 FC 层达到 ridge point。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：FlashAttention kernel 优化 attention 计算，但 ArI 仍 ≈1。FC 层使用 GEMM kernel，但因 batch size 受限，GEMV-like 特性导致 memory-bound。
  - **硬件架构层**：GPU HBM 提供 8000 GB/s 带宽，但因 ArI=1，实际吞吐远低于峰值。此前工作提倡 PIM 加速 attention（高 BW 匹配低 ArI）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法不是提出新系统，而是对 MLA + MoE 架构进行系统级 ArI 分析，揭示新的瓶颈并给出三条设计原则：(1) **MLA + layer reordering 将核心注意力层的 ArI 从 ≈1 提升到 ≈100-200 Op/B**——通过低秩 KV 压缩（$d_{\rm KVco}=512$ vs $d_{\rm dec}=16384$）+ 矩阵乘法结合律重排+ decoupled RoPE，消除 decode 阶段的 K 解压缩开销（减少 L 倍），Score 层读取压缩的 $\mathbf{C}_{\rm KV}$ 而非完整解压缩的 K，内存访问量急剧降低；FlashMLA 进一步复用 Score 层加载的 $\mathbf{C}_{\rm KV}$ 到 Context 层，ArI 逼近 ridge point。(2) **MLA 的极小 KV$ + MoE 的稀疏专家激活形成 Synergy**——MLA 将每 token KV$ 从 4.5 MB 降至 68.6 KB（67× 缩小），释放内存容量用于扩大 batch size（$B_{\rm cap}$ 增大 60×），使 MoE 各 expert 的 FC 层获得足够 token 达到 ridge point。(3) **瓶颈从 memory bandwidth 转移至互联带宽和 expert 负载均衡**——互联带宽（NVLink vs InfiniBand）直接决定 all-to-all dispatch/combine 通信延迟；expert 分布偏斜（Zipfian s=0.8）导致热 expert 饱和造成吞吐量下降和延迟上升，小粒度部署（32 GPU×8）比大粒度（256 GPU monolithic）更能缓解偏斜。

  全栈执行例子（论文方法: DeepSeek-R1, 32 B200 GPU, L=4096, B=128）：
  - **算法Pipeline层**：MLA 低秩压缩 $\mathbf{C}_{\rm KV} \in \mathbb{R}^{L\times 512}$ 替代完整 KV$。Layer reordering: Score = $(Q_i W_{\rm DK_i}^T) @ C_{\rm KV}^T$ 替换 $Q_i @ (C_{\rm KV} W_{\rm DK_i})^T$，Context = $(\text{Softmax} @ C_{\rm KV}) @ W_{\rm DV_i}$。ArI：Score/Context ≈100 Op/B（FlashMLA 加倍至 ≈200），不再 memory-bound。MoE：每 token 仅激活 8/256 experts，计算量/参数比降至 37B/671B。
  - **Serving/系统框架层**：$deg_{\rm TP}=1$（reordered MLA 中 TP 无益），$deg_{\rm DP}=32$, $deg_{\rm EP}=32$。KV$ 每 token 68.6 KB，B=128 时 KV$ 总量仅 0.54 GB/GPU。$B_{\rm cap}=7360$（vs GPT-3 的 124），可在 $B_{\rm RP} \approx 32 \times 281.25$ 下使 attention FC 层达到 ridge point。MoE $B_{\rm MoE} = 281.25 \times 256/8 = 9000$（32 GPU 中每 GPU 处理 8 experts）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：FlashMLA kernel 实现 reordered MLA decode，$\mathbf{C}_{\rm KV}$ 在 Score/Context 间复用。MoE 使用 fused expert kernel（gate/up/down projection fused）。all-to-all 通信使用 DeepEP 库。
  - **硬件架构层**：PIM 不再必要——MLA+MoE 使核心注意力层 ArI 接近 GPU ridge point，高 memory BW 的优势被高 compute 需求取代。仅低 batch 场景下 PIM 仍有优势。互联带宽是关键：高 BW NVLink（1.8 TB/s）vs 低 BW InfiniBand（100 GB/s）直接影响 MoE token dispatch/combine 延迟。小粒度部署（32 GPU×8）比大粒度（256 GPU monolithic）在 expert 偏斜时保持更高吞吐量（s=0.8 时 $\Gamma_{imb}^{acc}$ 低 6.13×）。

## Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 是 **(1) Dense Transformer 模型** 以及与先前 MoE scaling law 研究中的方法对比。Baseline 的核心缺陷：(a) **Dense 模型中参数量直接绑定计算量**——每 token 需要计算所有参数对应的 FLOPs，无法像 MoE 一样通过稀疏激活解耦参数量和计算量；(b) **MoE 缺乏统一的效率预测框架**——虽然 MoE 通过稀疏激活提高效率，但给定一个 MoE 架构配置（activation ratio, granularity, shared experts），无法预先知道其相对于等性能 dense 模型的计算效率；(c) **先前 MoE scaling law 研究的局限性**——Clark et al. (2022) 在固定数据集上评估导致 MoE 被 undertrained 的错误结论；Ludziejewski et al. (2024) 使用统一超参导致不公平比较，且他们的 granularity 定义 (G=4d_model/d_expert) 更粗粒度，观测到单调递增而非最优范围。
  
  全栈执行例子（Baseline: Dense Transformer, 训练大模型场景, C=1e22 FLOPs）：
  - **算法Pipeline层**：Dense decoder-only Transformer with GQA + RoPE。每层 FFN 为 dense，每 token 计算所有参数：C_dense_ffn = 6·B·s·d_model·d_ffn。模型参数量 = 计算量，无法通过 sparsity 提升效率。原问题：给定 C=1e22 FLOPs，应该训练多大的 dense 模型？过去凭经验猜测，无 EL 指导。
  - **Serving/系统框架层**：标准训练框架（AdamW optimizer, WSD LR schedule, data parallelism）。论文未明确说明训练系统细节（基于 Ling 系列内部框架）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 PyTorch/CUDA GEMM kernel 执行 attention 和 FFN。MoE 的 all-to-all communication 和 expert parallelism 论文未在实验中涉及。
  - **硬件架构层**：GPU（具体型号论文未明确说明，基于 Ant Group Ling Team 先前工作推测 A100/H800 级别 GPU）。Dense 模型训练仅需 data parallelism，无 expert 通信开销。
  Baseline 的核心缺陷：(a) **缺乏 EL 预测能力**——不知道 MoE 能带来多少效率增益，无法在训练前决定最优 MoE 配置；(b) **不公平的 MoE vs Dense 对比**——先前工作使用固定数据量评估 MoE（Clark et al. 2022）或统一超参（Ludziejewski et al. 2024），导致 MoE 被低估；(c) **Granularity 最优范围未发现**——Ludziejewski et al. 使用更粗粒度定义，观测到 monotonic trend，未发现 U 形最优范围 G=8~12。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Efficiency Leverage (EL)** 作为量化 MoE 计算效率的核心指标，通过大规模实证研究（300+ 模型，最大 28B 参数）建立 MoE 的统一 scaling law。具体设计解决 baseline 的三重缺陷：
  
  (1) **EL 指标解决"缺乏预测能力"缺陷**——EL = C_dense / C_moe 直接量化 MoE 的效率增益。基于 compute-optimal allocation 和 optimal hyperparameters 训练，保证公平比较。最终 joint scaling law：EL(A,G,C) = Â^{α + γ(log G)² + β log G}，给定 A, G, C 可直接预测 EL。
  
  (2) **三阶段实验方法解决"不公平对比"缺陷**——Stage 1: 推导 MoE 的最优超参 scaling law（η^opt ∝ C^{-0.1529}, B^opt ∝ C^{0.3644}），MoE 需要更大 batch size 和略低 LR（因 expert 梯度稀释）。Stage 2: 推导 MoE 的最优模型-数据分配（MoE 偏向更小 M、更多 D）。Stage 3: 在最优条件下消融各架构参数。这确保每个配置都在 near-optimal 条件下评估。
  
  (3) **更精细的 Granularity 定义解决"U 形最优未发现"缺陷**——使用 G=2d_model/d_expert（而非 4d_model/d_expert），探索更细粒度（G 最高 16），发现 U 形 loss-G 关系，最优 G≈12。此外发现 routing balance 影响：poor balance 使最优点下移至 coarser G。

  全栈执行例子（论文方法：Ling-mini-beta, C=1e22 FLOPs, A=3.4%, G=12, S=1/13≈7.7%）：
  - **算法Pipeline层**：基于 joint scaling law 预测 EL>7x → 选择 A=3.4%, G=12, E^s=1。Architecture: 20 layers, d_model=2048, d_ffn=5120, d_expert=384, 16 heads/4 kv_heads, E=384, E^a=12, E^s=1 (N=17.5B, N^a=0.85B)。每个 token 仅激活 3.4% 的总参数。Training: AdamW (β1=0.9, β2=0.95, wd=0.1), WSD LR schedule, η_max=3.78e-4, B=1792。Dense baseline: 28 layers, d_model=4096, d_ffn=14336, N=6.11B, η_max=2.93e-4, B=2048。
  - **Serving/系统框架层**：Ling 系列内部训练框架。论文未说明 distributed training 细节（DP/TP/EP 配置、all-to-all communication、expert parallelism 等）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准训练 pipeline。MoE routing: top-k softmax gate + load balancing loss (coeff=0.01) + router z-loss (coeff=0.001)。Training tokens=1T (vs dense 的 1T)，C_moe ≈ 1.43e21 FLOPs vs C_dense ≈ 1e22 FLOPs → EL ≈ 7x。
  - **硬件架构层**：GPU 训练（具体型号未明确说明）。MoE 的高 total params (17.5B) 需要更多 GPU memory 存放 expert 参数，但仅 0.85B 激活计算，FLOPs 远低于 6.1B dense。
  
  效果：Ling-mini-beta (0.85B active) 在 1T tokens 训练后，overall benchmark average 45.5 vs Dense-6.1B 44.0，验证 >7x EL。证明了 scaling law 的预测准确性。

## Symphony-MoE: Harmonizing Disparate Pre-trained Models into a Coherent Mixture-of-Experts

- baseline方法是什么？
  Baseline 方法包括三类 upcycling 方案：(1) **BTX (Branch-Train-Mix)**——将各 dense model 的 FFN 权重直接复用为 expert，共享 backbone 由所有模型权重的简单线性平均构成。(2) **BAM**——复用 FFN 权重和部分 attention 权重 (W^q, W^o) 作为 expert，其余权重线性平均构成 shared backbone。(3) **Drop-Upcycling**——复用 FFN 权重并对随机选择的参数施加 Gaussian perturbation 防止 expert 同质化，其余权重平均复用为 shared backbone。三者的共同缺陷：要么来自单一 dense checkpoint（expert 多样性受限），要么在融合多个不同训练历史的模型时使用粗糙的线性平均，无法解决 parameter space misalignment 问题——各 source model 的神经元在数值和语义参数空间中占据互不兼容的位置，直接合并导致 catastrophic interference。

  全栈执行例子（Baseline: BTX, Qwen 1.5B×4, 24× V100, upcycling + post-training on 5B tokens）：
  - **算法Pipeline层**：4 个 source dense models 的 FFN 层直接作为 4 个 experts（无 alignment）→ 共享 backbone = linear average(W_1, W_2, W_3, W_4) → 初始化随机 router → post-training 6 epochs with load balancing loss。问题：无 alignment 时，各 expert 的内部神经元排序不同（同一功能的神经元在不同 model 中的索引位置不同），linear averaging 产生"功能交叠"——每个 expert 的独特能力被互相稀释的共享 backbone 混淆，CKA 分数高达 0.65-0.75（专家功能崩溃为冗余子空间）。
  - **Serving/系统框架层**：LLaMA-Factory 框架训练 dense models，post-training 使用标准 transformer 前向传播。MoE forward 时 router 分配 token 到 4 个 experts。无 expert 间的功能区分，router 难以学习有意义的 dispatching。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 PyTorch GEMM/CUDA kernel 执行 expert FFN。无 expert-level 的 kernel 调度优化。
  - **硬件架构层**：NVIDIA V100（32 GB HBM2），24 卡。各 expert 的 FFN 参数驻留 GPU 内存。无跨卡 expert parallelism（单卡容纳全部 experts）。
  Baseline 的核心缺陷：(a) **Parameter space misalignment**——不同训练历史的模型占据互不兼容的参数空间，简单拼接/平均导致功能崩溃；(b) **Expert diversity loss**——naive merging 使 expert 的独特功能指纹被模糊化，测量为高 CKA 值；(c) **Router 失效**——在功能不一致的 experts 上，router 无法学习有意义的 token 到 expert 的映射。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Symphony-MoE**，核心是 **training-free functional alignment + post-training coordination** 两阶段框架。这直接解决 baseline 的三大缺陷：(1) **Parameter space misalignment**——通过 activation-based Hungarian permutation alignment（min_P ||A_1 - A_i P||²_F）将各 model 的 FFN 神经元在 training-free 下重排到 anchor 的功能空间，使得不同 model 中功能等价的神经元被映射到相同的空间位置，消除参数空间的"坐标系差异"。(2) **Expert diversity loss**——alignment 在消除 misalignment 的同时保留各 model 的独特参数值（permutation 仅改变神经元顺序，不改变数值），使 experts 功能兼容但不相同，CKA 分数恢复到接近原始 unmerged experts 的水平。(3) **Router 失效**——alignment 后的 experts 共享一致的坐标空间，router 可以基于 token-level 的语义内容学习有意义的 dispatching，实现真正的 expert 专化激活。

  全栈执行例子（论文方法：Symphony-MoE, Qwen 1.5B×4, 24× V100, Stage 1 alignment + Stage 2 post-training）：
  - **算法Pipeline层**：**Stage 1 (Training-free)**：
    (a) 共享 backbone 构建：对 self-attention 的 Q/K/V/O 矩阵用 SLERP 融合（preserving geometric integrity，sphere 上的最短测地线插值，公式为 SLERP(W1,W2,t) = (sin((1-t)Ω)/sin(Ω))·W1 + (sin(tΩ)/sin(Ω))·W2，其中 Ω = arccos(tr(W1·W2)/||W1||·||W2||)）；对 embedding 用 MergeKit selective linear（共享 vocabulary token → 线性平均，独有 token → 保留原 embedding）；对 LayerNorm 用简单平均。
    (b) Expert FFN alignment：以 M1 (General) 为 anchor → 从 General/Code/Math/Science 四个 domain 等量采样构建 D_cal (10.4M tokens) → 对每层每对 (M1, Mi) 提取 FFN 输出激活 → 用 Hungarian 算法求解 min_P ||A_1 - A_i P||²_F（O(d_ff³)) → 应用 P 重排 W_up, W_gate, W_down。结果：4 个 experts 功能对齐但参数各异。
    **Stage 2 (Post-training)**：随机初始化 router W_g ∈ R^{d_model × 4} → top-2 routing → 在扩展 D_cal (5B tokens) 上训练 6 epochs，AdamW lr=5e-5 → L_total = L_lm + 0.01·L_bal → 专家协作学习。最终：MMLU 58.91 vs BTX 45.12 (+13.8%)，HumanEval 42.39 vs BTX 29.08 (+13.3%)，MedCQA 35.26 vs BTX 26.92 (+8.3%)。

  - **Serving/系统框架层**：LLaMA-Factory 用于 dense model instruction tuning。MoE 架构为标准 decoder-only transformer，每 L 层替换一个 dense FFN 为 N 个 expert FFN + top-2 router。Forward 时为每个 token 激活 2/4 experts，FLOPs per token 为 dense 模型的 2×。无 serving framework 修改。

  - **编译框架层**：论文未明确说明。

  - **Kernel调度层**：标准 PyTorch 和 CUDA kernel 执行 expert FFN forward/backward。Align 阶段使用 Hungarian 算法（scipy.optimize.linear_sum_assignment）。Post-training 使用标准 transformer 训练 pipeline。无自定义 kernel。

  - **硬件架构层**：NVIDIA V100 (32 GB HBM2)，24 卡。Stage 1 alignment 是单卡计算（仅需 calibration data 前向 + Hungarian + weight remap）。Post-training 在 24 卡上进行数据并行训练。所有 4 个 experts 的完整参数驻留 GPU 内存（1.5B×4 架构，约 6B 总参数但每个 token 仅激活 2 experts ≈ 3B 参数量，与 dense 1.5B 相比 FLOPs per token 近似翻倍）。

  方法 vs Baseline 对比核心差异：(a) Training-free functional alignment vs naive averaging——用 permutation 消除 misalignment 而无需训练，保留专家多样性；(b) Layer-aware backbone vs uniform averaging——SLERP 保护 attention 的几何结构，MergeKit 处理 embedding 的 vocabulary mismatch，避免功能退化；(c) 两阶段协调 vs end-to-end 训练——先解决参数空间 misalignment（Stage 1），再学习 router routing（Stage 2），分离了结构融合和功能协调两个目标。

## Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- baseline方法是什么？
  Baseline 方法包括两类：(1) **Dense T5 Transformer**（T5-Base 223M, T5-Large 739M, T5-XXL 11B）——所有 token 共享同一套 FFN 参数，每次 forward 使用全部参数计算。随模型规模增长，FLOPs per token 同比例增加，计算开销大。(2) **Standard MoE Transformer (top-k routing, k≥2)**——使用 Shazeer et al. (2017) 的 Noise Top-k Gating，每个 token 路由到 k>1 个 expert，输出为各 expert 的 gate 值加权求和 y = Σ_{i∈T} p_i(x) E_i(x)。k>1 导致：(a) 路由计算量是 Switch 的 k 倍；(b) 每个 expert 需处理更多 token（expert capacity 增大），增加计算和通信成本；(c) 需要更复杂的 all-to-all 通信模式。
  全栈执行例子（Baseline: Dense T5-Base，224M params，TPUv3，pre-training on C4）：
  - **算法Pipeline层**：输入 token sequence X [B, 768] → Multi-Head Self-Attention (QKV 投影) → FFN: X × W_in [768, 2048] → ReLU → × W_out [2048, 768] → 输出。所有 B 个 token 共享 W_in/W_out，无需路由。单层 FFN 参数量 = 768×2048×2 ≈ 3.1M。12 层总参数 224M，FLOPs per token 固定为 124B。
  - **Serving/系统框架层**：Mesh TensorFlow 的 Data Parallelism（所有 cores 持有完整模型副本，仅 batch 拆分）或 Model + Data Parallelism（d_ff 维度拆分，增加 all-reduce 通信）。无 expert 分发逻辑。每个 TPU core 在 forward/backward 末端进行 all-reduce 梯度聚合。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 TPU XLA 编译的矩阵乘法和 All-Reduce 通信。每个 FFN 层为固定形状的 dense matmul [B,768]×[768,2048]，无动态路由 kernel。
  - **硬件架构层**：TPUv3，32 cores，每个 core 在 forward/backward 全程参与计算，无 idle。通信仅在梯度聚合时发生（all-reduce），中间激活无跨 core 通信。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Switch Transformer**，核心是 **k=1 routing**：将传统 MoE 的 top-k 路由简化为仅路由到单个得分最高的 expert。这直接解决 Baseline 的缺陷：(1) Baseline Dense 模型计算效率低——所有参数对每个 token 都执行计算，Switch Transformer 通过稀疏激活（每个 token 仅用 1 个 expert 的参数），在相同 FLOPs per token 下将参数量从 224M 扩大到 7B+，实现 7x 训练加速。(2) Baseline MoE (k>1) 通信和计算冗余——k=2 时每个 token 需两个 expert 计算，Switch 的 k=1 将 expert capacity 减半，通信量减少约 50%。(3) Baseline 训练不稳定——Switch Transformer 通过 selective precision（router 内部 float32）、reduced initialization scale（0.1x）、expert dropout 三种技术组合稳定训练 bfloat16 稀疏模型。(4) Baseline expert 负载不均——通过 auxiliary load balancing loss（α·N·Σ f_i·P_i）使 token 均匀分发到各 expert，保持 expert 利用率。
  全栈执行例子（Method: Switch-Base，7B params，128 experts，TPUv3 32 cores，训练 on C4）：
  - **算法Pipeline层**：输入 token sequence X [B, 768] → Self-Attention（未修改）→ Switch FFN: Router 计算 logits = X × W_r [768, 128] → softmax → argmax 选 top-1 expert index i → 每个 token 仅路由到 expert i → 对该 expert 执行 FFN_i(X) = ReLU(X × W_in_i)·W_out_i → 输出乘以 gate value p_i → residual add + layer norm。只有被路由到的 expert 参数参与该 token 的计算，其余 127 个 expert 保持 idle。参数量 = 128 × 3.1M（每 expert FFN）+ 共享参数 ≈ 7B，但 FLOPs per token 仍是 124B（同 T5-Base）。
  - **Serving/系统框架层**：Mesh TensorFlow Expert + Data Parallelism（Section 5.4）。32 cores 对应 32 条 data-parallel 路径，每个 core 持有 unique expert（或 experts 子集）。Router 在 local core 上计算各 token 的目标 expert index，产成 binary dispatch tensor [n, B/n, E, C] → einsum 将 tokens gather 到对应 expert → all-to-all 通信交换 tokens（shape [E, C, d_model]）→ 每个 core 上的 expert 执行 FFN → all-to-all 通信返回结果 → combine tensor 加权汇总。额外通信开销：forward 和 backward 各一次 all-to-all，传输量 = E×C×d_model × 2（来/回）× bfloat16。
  - **编译框架层**：Mesh TensorFlow (MTF) 的 SPMD 编程模型。将物理 TPU cores 映射为逻辑 mesh [n, m]，tensor 沿命名维度 shard。Switch layer 的 dispatch/combine 通过 mtf.einsum 和 mtf.reshape 实现，XLA 编译器处理底层通信生成。所有 tensor shape 在编译时静态确定（包括 expert capacity C）。
  - **Kernel调度层**：Router 使用 bfloat16→float32→bfloat16 selective precision，仅 local 计算在 float32。All-to-all 通信传输 bfloat16 精度的 expert 输入/输出 tensor。Expert FFN 为 TPU 上标准 dense matmul kernel，每个 expert 独立执行。论文未使用自定义 GPU/TPU kernel。
  - **硬件架构层**：TPUv3，32 cores 通过高速互联（ICI）连接。All-to-all 通信利用 TPU 的环形拓扑高效完成 tensor 交换。Expert Parallelism 下每个 core 持有 128/32 = 4 个 expert 的完整参数，其余 expert 的参数分布在其他 core 上，通过 all-to-all 按需获取。Extra communication 占总时间比例随 expert 数量增加而增加，但被样本效率提升所抵消。

## SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget

- baseline方法是什么？
  Baseline 方法包括两类：(1) **On-demand Loading（内存交换）**——按需通过 PCIe 从外部存储加载 MoE layer 的 expert 参数到 GPU memory，推理完成后释放。每次 MoE layer 参数传输引入显著延迟（6.2×-8.9×），参数传输占推理时间的绝大部分，即使异步加载也无法避免 I/O 阻塞计算（因为 loading 始终慢于 computation）；(2) **Expert Pruning（专家剪枝）**——基于 expert 权重的 magnitude（||E_i||）永久剪除不重要的 experts，缩小模型后直接推理。剪枝后不可恢复，导致显著的准确率损失（例如 SwitchT-32 减少 30% memory 时准确率下降 14%），且需要额外训练来恢复性能。
  全栈执行例子（Baseline: On-demand Loading，SwitchT-16，Jetson AGX ORIN，summarization task）：
  - **算法Pipeline层**：输入 token sequence X → Self-Attention → Router gating G(x) = softmax(W_r @ x) → 路由到 expert k = argmax G(x) → 此时 E_k 不在 GPU memory 中 → 触发 PCIe 加载 E_k 参数（W_in, W_out）→ 加载完成后执行 E_k(X) → 完成后释放 E_k → 下一个 MoE layer 重复此过程。每层需等待 expert 参数传输，expert 参数大小 = 2 × d × d_ff × 4 bytes（FP32），在 14 GiB 模型下单层传输可达数 GB。Baseline 准确率无损失（使用完整 expert set），但延迟极高。
  - **系统框架层**：HuggingFace Transformers 标准 MoE layer forward，每层执行前检查 expert 是否在 device，不在则触发 load_state_dict。无异步机制或 IO/计算 overlap 优化，expert loading 完全阻塞 computation pipeline。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 cuBLAS GEMM 执行 expert FFN 计算。无 expert 选择或跳过逻辑，无 IO 调度。
  - **硬件架构层**：Jetson AGX ORIN，GPU-CPU 通过 PCIe 连接，expert 参数存储在 CPU memory。PCIe 带宽限制（10-30 GiB/s）成为瓶颈。
  Baseline 的核心缺陷：(a) On-demand loading 用延迟换内存——延迟开销不可接受（6.2×-8.9×）；(b) Pruning 用准确率换内存——永久性准确率损失无法恢复；(c) 两者都无法实现可调的 memory-accuracy-latency tradeoff。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法通过三个层面解决 baseline 缺陷：(1) **Virtual Experts 动态子集维护**——不再按需加载单个 expert 或永久剪枝，而是维护一个动态更新的 expert 子集（Virtual Experts），基于数据分布 locality（连续 token 语义相关、同一用户 conversation 上下文连续）预测未来最可能需要的 experts，预加载到 GPU memory。这样，每次推理的计算量 = 小型 MoE 模型（仅 VE 参与），但大模型的能力保留（每个 expert 都有机会参与）；(2) **Importance-aware Expert Selection + Masked Gating**——设计 expert importance score = Σ ||x|| · |G(x)_i| · ||E_i||（综合 token norm、routing weight 和 expert weight magnitude），高效量化每个 expert 对当前数据分布的贡献。使用 Masked Gating 将推理请求重定向到 VE，避免运行时 routing 到不在内存中的 expert；(3) **Profiling-guided Memory Planning + Genetic Search**——离线 profile 每个 expert 的性能特征（memory/latency/loading time/IO bandwidth），训练小型 DNN 建模 config→accuracy 映射，用遗传算法在巨大搜索空间中搜索最优层间 expert 分配方案（而非枚举 12^16 种组合）。
  全栈执行例子（论文方法：SwapMoE，SwitchT-16，Jetson AGX ORIN，summarization task，memory budget 4.7 GiB）：
  - **算法Pipeline层**：输入 token sequence X → 对每个 MoE layer l：(a) Router 计算 gating_scores = softmax(W_r @ X)；(b) Masked Gating：mask[i] = 1 if i ∈ VE else 0，masked_scores = normalize(gating_scores ⊙ mask)；(c) Expert 计算：仅 i ∈ VE 执行 E_i(X)，output y = Σ masked_scores[i] · E_i(X)；(d) 收集 importance score：对每个 expert E_i 和其处理的 tokens X_i，importance = Σ ||x|| · |masked_scores[i]| · ||E_i||_F；(e) 每 frequency 个 sample 后：异步加载 top-k experts 到 GPU memory，释放 bottom experts。结果：memory 14.2 GiB → 4.7 GiB (67% reduction)，latency 降低 50%（因为仅计算 VE subset），ROUGE-2 仅下降 0.041。
  - **系统框架层**：HuggingFace Transformers 中修改 MoE layer forward：(a) 插入 Runtime Scheduler——在 router 和 expert FFN 之间插入 VE selection 和 Masked Gating；(b) Amortized Expert Loading——跨多个 sample 摊销 expert 加载开销（图6 ii），避免每 sample 后同步更新；(c) Asynchronous Expert Loading——使用 async copy engine 加载 expert 参数，与 computation pipeline overlap。IO overhead 极低（peak ~40 MiB/s vs PCIe 10-30 GiB/s）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：standard GEMM kernel 不变，但计算量大幅减少——仅计算 VE subset 的 experts。Masked Gating 避免了 runtime routing 到缺失 expert 的 penalty。IO 调度通过 async copy engine 实现。
  - **硬件架构层**：Jetson AGX ORIN 和 Jetson Nano。Expert 参数分层存储：VE 在 GPU memory（快速访问），其余 experts 在 CPU memory 或 SSD（通过 PCIe 或存储总线访问）。每个 expert loading 的时间被 precise profiling 并纳入 offline planning。
  方法 vs Baseline 对比核心差异：(a) VE 动态子集 vs Static pruning——保留所有 expert 参数完整性，按需 swap 而非永久丢弃；(b) Amortized + Async loading vs On-demand synchronous loading——将延迟峰值从 per-layer 平摊到 per-N-samples，通过 async I/O 与计算 overlap；(c) Genetic search configuration vs Manual fixed allocation——在 12^16 搜索空间中找到接近最优的层间 expert 分配，而非均匀分配或手工调优；(d) Memory-Accuracy-Latency tunable tradeoff vs 二元选择（全 accuracy 高延迟 vs 低 accuracy 低内存）。

## Sparse Upcycling Training Mixture-of-Experts from Dense Checkpoints

- baseline方法是什么？
  Baseline 是将预训练 dense Transformer checkpoint 直接继续训练（"dense continuation"），即对已有 dense checkpoint 不做架构修改，用相同的超参数（batch size、inverse square root LR schedule、Adafactor optimizer）再训练额外 steps。另一个 baseline 是从头训练 MoE（MoE from scratch），即从随机初始化权重训练同样结构的 MoE 模型。
  全栈执行例子（Baseline: T5 Base Dense Continuation，TPU v4，64 chips）：
  - **算法pipeline层**：从已训练 1M steps 的 T5 Base dense checkpoint 继续训练，保持相同模型结构和超参数。输入 token sequence X ∈ R^{T×d} → Self-Attention → Dense FFN (W_in → GEGLU → W_out) → LayerNorm → output。每步 FLOPs 固定，参数量固定（248M），无 capacity 扩展。
  - **系统框架层**：TPU v4 集群上使用 T5X 框架（https://github.com/google-research/t5x），数据并行 + 模型分片进行分布式训练。dense continuation 仅需持续执行 forward/backward/optimizer step，无额外的路由计算或 expert 通信开销。
  - **编译框架/Kernel调度层**：论文未明确说明。
  - **硬件架构层**：TPU v4，使用 Adafactor optimizer 进行 mixed precision 训练，无 MoE expert 分片和路由相关通信。
  Baseline 的核心缺陷：dense continuation 受限于原始 dense 模型的参数容量，增加 compute 的边际收益递减——训练曲线已经饱和，额外 compute 带来的性能提升有限。MoE from scratch 则需要完全重新训练，浪费了已投入的 dense checkpoint 训练成本。

## Sparsing Law Towards Large Language Models with Greater Activation Sparsity

- baseline方法是什么？
  Baseline 是主流 SiLU 激活的密集 LLM（如 LLaMA 结构 + gated FFN）：使用 SiLU 激活函数、无激活稀疏度感知的架构设计、依赖固定宽深比的 Transformer 结构。在稀疏度度量方面，baseline 方法包括：(1) Straightforward ReLU——用零阈值判断弱贡献神经元（仅适用于 ReLU，无法泛化到 SiLU）；(2) Top-k——强制每层保持固定 k 个激活神经元（MoE 常用，牺牲灵活性和性能）；(3) FAT-ε——全局统一阈值忽略绝对激活值低于 ε 的神经元（无法层间自适应）。
  全栈执行例子（Baseline: SiLU LLaMA-like 0.8B 模型，GPU 推理）：
  - **算法pipeline层**：输入 token x ∈ R^{d_h} → gated FFN：s = SiLU(W^{gate}x)，FFN(x) = W^{out}[s ⊙ (W^{in}x)]。SiLU 激活函数产生大量非零但幅度可忽略的负输出值，而这些值在零阈值下不被视为"弱贡献"，导致稀疏度被严重低估。训练过程中，SiLU 模型激活比满足 A_SiLU(D) = -c/D^α + A_0（递增幂律），意味着更多训练数据反而降低稀疏度（激活比收敛到 ~40%）。以 0.1B SiLU 模型为例，极限激活比 A_0=40.9%，即最多只有 ~59.1% 的稀疏度。
  - **系统框架层**：使用 PowerInfer 或 llama.cpp（https://github.com/ggerganov/llama.cpp）进行推理。llama.cpp 进行密集 FFN 计算，即对每一层所有 d_f 个神经元计算完整的 W^{in}x、SiLU、W^{out} 输出求和，无法跳过弱贡献神经元。每个 token 经过所有层所有神经元，每 token 的 FLOPS 为常量。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：密集 FFN 计算使用标准 GEMM kernel（如 cuBLAS），无稀疏度感知的 kernel 优化。PowerInfer 虽有离线 profiler 和在线预测器来跳过弱贡献神经元，但 baseline 的 SiLU 模型稀疏度低（~60%），加速效果有限。
  - **硬件架构层**：NVIDIA A800 GPU（80GB），密集计算模式，无稀疏加速硬件支持。
  Baseline 的核心缺陷：(a) SiLU 激活函数的负输出使简单的零阈值度量失效，缺乏通用且精确的激活稀疏度评估指标；(b) 主流 SiLU LLM 的激活稀疏度随训练数据增加而降低（递增幂律），与高效推理的目标背道而驰；(c) 缺乏宽深比、参数规模等架构因素如何影响稀疏度的定量理解，无法指导稀疏 LLM 的架构设计。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法通过三个层面解决 baseline 缺陷：(1) **CETT-PPL-1% 通用稀疏度度量**——使用 CETT（累积尾部截断误差）自适应搜索每层阈值，识别弱贡献神经元集合 D = {i | ||n_i||_2 < ε}，通过控制 L2 范数相对误差而非原始激活值来泛化到任意激活函数（包括 SiLU）；引入 PPL 增加容忍度 1%，二分搜索 CETT 超参数，确保稀疏化后性能退化可忽略（表1验证 CETT-PPL-1% 平均性能退化仅 -0.16% C.R.、-0.30% R.C.）；(2) **激活稀疏度定量标度律**——发现 ReLU 模型满足递减对数空间幂律 A_ReLU(D) = exp(-cD^α + b) + A_0，即更多数据可降低激活比（提高稀疏度）；而 SiLU 模型满足递增幂律，更多数据反而损害稀疏度；(3) **架构指导原则**——确定宽深比的瓶颈效应（0.1B 模型~114）：低于瓶颈时激活比与宽深比线性正相关；建议选择确保训练稳定性的最小宽深比。发现极限激活比与参数规模弱相关（规模不敏感性）。
  全栈执行例子（论文方法：ReLU 2.4B μP Transformer，NVIDIA A800 GPU，800B tokens 训练）：
  - **算法pipeline层**：输入 token x ∈ R^{d_h} → gated FFN：s = ReLU(W^{gate}x)，FFN(x) = W^{out}[s ⊙ (W^{in}x)]。ReLU 激活函数天然产生大量零值（s_i=0 的神经元输出为零向量，对最终输出无贡献）。训练中使用 CETT-PPL-1% 度量监控稀疏度演化：对最后 5 个 checkpoint，二分搜索 [0,1] 内的 CETT 超参数，计算平均 PPL 比率 = exp(loss_sparse - loss_dense) 是否达到 1.01，确定统一 CETT 阈值后应用于全过程。ReLU 模型服从递减对数空间幂律，800B tokens 训练后极限激活比 A_0=6.48%（稀疏度 93.52%）。宽深比选择接近 0.1B-1.2B 实验模型的值（~48-56），确保在训练稳定性区间内。
  - **系统框架层**：推理时使用 PowerInfer（https://github.com/SJTU-IPADS/PowerInfer）——其离线 profiler 统计每层每个神经元的历史激活频率，在线预测器根据当前输入预测哪些神经元可能被激活。由于 ReLU 模型的高稀疏度（93.52%），对任意给定输入，仅约 6.48% 的神经元被激活（s_i > 0），PowerInfer 的预测器能高置信度跳过其余 93.52% 弱贡献神经元的计算，相比之下 SiLU 模型只能跳过 ~60%。llama.cpp 作为密集 baseline 无法利用此稀疏度。
  - **kernel调度层**：PowerInfer 根据预测的激活模式选择性地执行 GEMM：仅加载和计算被预测为"活跃"的神经元对应的 W_{i,:}^{gate}、W_{i,:}^{in}、W_{:,i}^{out} 行/列，跳过弱贡献神经元。对于 ReLU 模型密集版本（所有 d_f 个神经元都参与），每层 FLOPS 固定；稀疏版本仅计算 ~6.48% 的神经元，FLOPS 按比例减少。实测 2.4B 模型解码速度 41.79 tok/s（PowerInfer）vs 10.23 tok/s（llama.cpp 密集），4.1× 加速。
  - **编译框架层**：论文未明确说明。
  - **硬件架构层**：NVIDIA A800 GPU（80GB），104 CPUs。PowerInfer 编译为 CUDA 版本。CETT 二分搜索中，dense 和 sparse forward pass 均在 GPU 上执行，loss_dense 和 loss_sparse 在验证集 VS 上计算（每个 checkpoint 的每个 batch 执行一次 dense forward 和一次 sparse forward）。
  方法 vs Baseline 对比核心差异：(a) ReLU 替代 SiLU → 天然更高的稀疏度且数据越多越稀疏（递减对数空间幂律 vs 递增幂律）；(b) CETT-PPL-1% 替代零阈值/全局阈值 → 通用、精确、可泛化的稀疏度度量；(c) 小宽深比 + 大规模数据 → 有理论指导的稀疏 LLM 训练策略；(d) 规模不敏感性 → 小模型上的发现可直接推广到大模型。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Sparse Upcycling 通过将 dense checkpoint 的 MLP 层扩展为 MoE 层（每个 expert 初始化为原 MLP 的拷贝，router 随机初始化），以微小的额外 compute budget（~50% 原训练成本）将 dense 模型升级为容量大得多（参数增加 8-10×）但推理 FLOPs 相近（因为稀疏激活）的 MoE 模型。
  全栈执行例子（Sparse Upcycling: T5 Base → MoE，TPU v4，64 chips）：
  - **算法pipeline层**：Input token X → Self-Attention（复制自 dense）→ 若该层为 MoE 层（每间隔一层）：Router 计算 softmax(X @ W_r)，Expert Choice routing 让每个 expert 独立选择 C·T/E 个 token → 32 个 expert 各自执行 FFN（参数复制自原 dense MLP）→ Combine（加权合并 expert 输出）→ LayerNorm。非 MoE 层保持原 dense FFN。训练继续使用原 inverse sqrt LR schedule。
  - **系统框架层**：TPU v4 上使用 T5X 框架（https://github.com/google-research/t5x/tree/main/t5x/contrib/moe），数据并行 + expert 并行（将 32 experts 分布到多个 chip）+ XL 模型额外使用 model 并行（4 partitions）。Router 计算通信量小，expert 之间的 all-to-all token dispatch 是主要通信开销。
  - **编译框架/Kernel调度层**：论文未明确说明。
  - **硬件架构层**：TPU v4，使用 Adafactor optimizer，mixed precision 训练不变。
  Baseline 缺陷的对应解决：(a) **容量瓶颈**——Dense continuation 参数容量固定，Sparse Upcycling 通过 MoE 将参数量从 248M 扩展到 2.00B（T5 Base），expert 专业化学习使模型容量大幅提升；(b) **训练成本浪费**——MoE from scratch 浪费 dense checkpoint 训练投入，Sparse Upcycling 复用所有已有参数和 optimizer state（vision），仅需 ~50% 原训练成本即可超越 dense continuation；(c) **初始性能下降**——通过 Expert Choice routing + router weight normalization（vision）减小 model surgery 带来的初始性能损失，确保被至少一个 expert 选中的 token 保持与原始 dense 相同的输出；(d) **路由效率**——Expert Choice routing (C=2) 相比 Top-K routing 更快且 per-compute-time 性能更好，避免 token dropping 和 expert 负载不均衡。

- baseline方法是什么？
  Baseline 方法包括现有开源 MoE kernel 实现：(1) **ScatterMoE**（Triton, Tan et al. 2024）：forward gather fusion 仅实现 varlen-M（非 varlen-K），backward 需单独 gather kernel，dS=⟨dO,Y⟩ 需缓存 Y，无 MMA/IO 重叠，无 TMA 支持；(2) **MoMoE**（Triton, Costin et al. 2025）：类似 ScatterMoE，dS 虽融合于 up-proj act grad 但仍用 ⟨dO,Y⟩ 路径，scatter 操作较慢；(3) **MegaBlocks**（Gale et al. 2023）：block-sparse GEMM 方式，需单独 gather+pad+scatter kernel，总 IO 达 8TKd bytes；(4) **Megatron-LM GroupedMLP**：使用 CUTLASS Grouped GEMM 但无 gather fusion，假设输入已 contiguous-packed；(5) **DeepGEMM**（Zhao et al. 2025b）：高度优化的 SM90/SM100 BF16 Grouped GEMM，但仅支持 contiguous-packed 输入，无 gather fusion、无 epilogue fusion、无 MMA/IO 重叠。所有 baseline 的共同痛点：(a) 在细粒度 MoE（高 G=d/n）下 activation memory 随 G 线性增长，(b) IO cost 随 G 线性增长导致 memory-bound，(c) 稀疏 MoE 下 Grouped GEMM tile padding 浪费大量 FLOPs。
  全栈执行例子（Baseline: ScatterMoE，H100，7B MoE, n=256, K=8, E=64）：
  - **算法Pipeline层**：Forward 执行 up-proj group GEMM (gather X, GEMM, SwiGLU) → down-proj group GEMM (GEMM, scatter Y)。Backward 需要缓存 X, X_e, H, Y, S, π。dS = ⟨dO, Y⟩ 需要加载 dO 和 Y (2TKd bytes HBM 访问)，dH 通过 dSwiGLU(dY W_2^T, H) 从 H 重算。总 activation memory = 2Td + 4TKn + 2TKd ≈ 2Td + 4TKn + 2T×8×(d/256)×d bytes，随 G 线性增长。
  - **系统框架层**：PyTorch autograd 引擎管理前向/反向，Triton kernel 编译为 CUDA。FSDP-2 + ZeRO-3 分布式训练，使用 lm-engine 代码库。
  - **编译框架层**：Triton → MLIR → PTX → SASS。Triton 无法直接控制 TMA 异步操作和 warp-specialized 调度，限制了 Ping-Pong scheduling 和异步 IO 重叠的能力。
  - **Kernel调度层**：ScatterMoE forward up-proj gather+GEMM (Triton kernel, ~600 TFLOPS on H100) → forward down-proj GEMM+scatter (Triton kernel, st.global store, ~550 TFLOPS) → backward dH (单独 Triton kernel, ~300 TFLOPS) → backward dS kernel (单独 launch, 读取 dO+Y) → backward dW2 (Triton kernel, 无 gather fusion 需单独 gather kernel) → backward dX~ (Triton kernel) → backward dW1 (Triton kernel, 无 gather fusion) → backward dX aggregation。Triton kernel 无法异步 overlap MMA 与 IO，kernel 间有 CUDA stream bubble。
  - **硬件架构层**：H100 GPU SM (132 SMs)，每个 SM 有 4 warp schedulers、Tensor Core (WGMMA)、TMA engine、256KB SMEM。Triton 使用 TMA（ScatterMoE 基于旧版 Triton 不支持），cp.async 仅用于 gather fusion in forward。st.global 同步 store 阻塞下一 tile MMA 执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **SonicMoE**，一种硬件-模型架构协同设计方案，通过三个核心创新解决 baseline 痛点：
  **(1) Memory-efficient backward 算法**：重新设计计算图——dS=⟨dA',A⟩ 替代 dS=⟨dO,Y⟩（节省 2TKd bytes activation 和 2TKd bytes HBM 访问），dH 从 dA 和 H 通过 dSwiGLU 重算。Gather fusion 消除 X_e 和 dO_e 的 HBM 物化。结果：activation memory 恒定于 2Td+4TKn bytes，不随 G 增长。对 7B MoE (n=256) 减少 45% activation memory vs ScatterMoE。
  **(2) IO-aware kernel 设计**：三个层面减少/隐藏 IO——(a) Gather fusion with cp.async 在 varlen-M 和 varlen-K Grouped GEMM 中均实现（baseline 仅 forward varlen-M），在 Blackwell 上通过 relay warp + mbarrier cluster-scope 解决 2-CTA cluster 的 gather 同步问题；(b) Heavy epilogue fusion：SwiGLU/dSwiGLU/dS/A' 全部融合在 GEMM epilogue 中，用一个 kernel 产出多个输出；(c) MMA/IO 重叠：Hopper 上 Ping-Pong scheduling（2 consumer warpgroups 交替 MMA 和 epilogue），Blackwell 上利用 UMMA 单线程异步 + TMEM 2-stage 实现 MMA warp 与 epilogue warps 并发。异步 TMA store（非 st.global scatter）避免阻塞 MMA。
  **(3) Token Rounding (TR) 路由**：消除 Grouped GEMM tile padding 浪费——将 per-expert token 数舍入到 M_tile=128 倍数，每个 expert 最大偏离 TC 结果 1 tile。TR 在极稀疏 MoE (K/E≤1/64) 下额外提升 16% kernel TFLOPS 且不损失下游任务质量。
  全栈执行例子（SonicMoE，H100，7B MoE, n=256, K=8, E=64）：
  - **算法Pipeline层**：Forward 与 baseline 语义等价。Backward 重设计：dH kernel 同时产出 dH（by dSwiGLU(dA, H)）、dS（by ⟨dA', A⟩ reduce over n）、A'（by Broadcast(s)·A）。无 Y 缓存，activation memory 恒定为 2Td+4TKn bytes。TR 路由：top-K TC → 计算 f_e → nearest-round to 128 → EC 排序选择 padded/discarded tokens → 修改 π 和 S → 输入 MoE compute kernel（与路由算法解耦）。
  - **系统框架层**：CuTe-DSL (C++) 编写 kernel，PyTorch nn.Module 封装为 drop-in MoE 层。lm-engine 代码库管理 FSDP-2 训练循环。64 H100s 实现 213B tokens/day（与 ScatterMoE 96 H100s 的 225B 可比）。
  - **编译框架层**：CuTe-DSL → C++ → NVCC → PTX → SASS。CuTe-DSL 允许直接控制 TMA、cp.async、WGMMA/UMMA、warpgroup scheduling 的底层指令，实现 TileLang/Triton 无法表达的异步 IO/MMA 重叠。
  - **Kernel调度层**：8 个 CuTe kernel 流水线执行：(A kernel: cp.async gather + WGMMA + SwiGLU epilogue, ~650 TFLOPS) → (Y kernel: WGMMA + TMA store + Ping-Pong epilogue, ~600 TFLOPS) → (O kernel: TMA gather-and-sum, ~2.5 TB/s) → backward: (dH kernel: cp.async gather + WGMMA + heavy epilogue 含 dSwiGLU/dS/A' + 异步 TMA load H, Ping-Pong scheduling, ~450 TFLOPS) → (dW2 kernel: cp.async gather + WGMMA, ~500 TFLOPS) → (dX~ kernel: WGMMA + TMA store, ~580 TFLOPS) → (dW1 kernel: cp.async gather + WGMMA, ~520 TFLOPS) → (dX kernel: TMA gather-and-sum, ~2.5 TB/s)。Ping-Pong 重叠 MMA 与 IO 使得 dH kernel 的 heavy epilogue 不显著拖慢吞吐。
  - **硬件架构层**：H100 的 TMA 异步 copy（GMEM↔SMEM）与 Tensor Core WGMMA 通过 CUDA pipeline 异步并发。Ping-Pong: consumer warpgroup 0 执行 WGMMA 时，consumer warpgroup 1 执行上一 tile 的 epilogue（SwiGLU/TMA store），下一 tile 角色互换。Blackwell B300 的 TMEM (256KB/SM, 2-stage=2×128cols) + UMMA（单线程异步，无 RF 压力）让 epilogue warps 直接从 TMEM stage 读取 MMA 结果并执行 epilogue，与 MMA warp 写入另一 TMEM stage 完全并发。所有 scatter 操作替换为 gather-and-sum aggregation，使用 TMA gather 保持高 bandwidth。

## Self-MoE Towards Compositional Large Language Models with Self-Specialized Experts

- baseline方法是什么？
  Baseline方法：(1) **单体LLM微调（Monolithic Fine-tuning）**：直接在目标领域数据上对整个模型进行全参数或参数高效微调（如LoRA），得到一个在特定领域增强的模型。然而这种方式存在灾难性遗忘问题——在目标领域上的提升往往以牺牲非目标领域性能为代价（如Knowledge Self-Spec在MMLU上从58.4→64.0，但在BBH上从56.1→41.7）。(2) **Instance Merging（多任务微调）**：将所有领域的合成数据合并后直接微调单一模型（LoRA），得到一个多任务模型，但缺乏动态适应能力，无法根据具体输入选择最合适的专家知识。(3) **Weight Merging（TIES/DARE）**：将多个独立训练的专家模型通过权重平均或参数融合合并为单一模型，但合并后的模型参数是静态的，不同专家之间可能产生参数干扰（interference），且丧失了语义可解释性。(4) **传统MoE预训练方法（Switch Transformer, Mixtral, BTX）**：使用FFN层作为expert，随expert数量增长参数总量线性增加（Mixtral 8x7B总参数47B），且expert在预训练中隐式学习，缺少显式的语义区分。需要大量计算资源（BTX需要900 GPU天）。
  全栈执行例子（Baseline: Instance Merging方案，基于Gemma-7B，单A100-80GB）：
  - **算法Pipeline层**：收集所有目标领域的合成数据 D_knowledge ∪ D_reasoning ∪ D_math ∪ D_coding（共20K样本），用LoRA(rank=8)直接微调Gemma-7B，得到单一适配器 ΔΘ_merged，推理时对所有输入统一使用 h = θ_0 x + Δθ_B Δθ_A x，无动态路由选择。
  - **系统框架层**：使用HuggingFace PEFT加载base模型+单一LoRA adapter，使用Alpaca prompt template进行推理，evaluation通过LM Evaluation Harness统一调用。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法Self-MoE构建了**MiXSE（MiXture of Self-specialized Experts）**——一个组合式模块化系统。通过两阶段设计解决baseline的缺陷：
  **(1) 自专业化（Self-Specialization）解决标注数据瓶颈**：不依赖昂贵的人工标注数据或更强的教师模型，而是利用base LLM自身的生成能力，通过Instruction Brainstorming→Response Generation的自我循环，从仅100条种子数据自动扩展出每领域5K条高质量合成训练数据。这解决了MOLE依赖人工标注、PHATGOOSE依赖预训练外部专家、GLAN依赖GPT-4蒸馏等baseline对数据的强假设。
  **(2) 模块化LoRA专家+动态路由解决遗忘与静态融合问题**：每个领域独立训练轻量LoRA专家模块（<0.3%参数），保持base LLM Θ_0 不变，在推理时通过训练好的路由层 θ_r 对每个token动态计算 top-k 软权重选择最相关专家：h = θ_0 x + Σ α_i Δθ_i x。路由器的自学习仅需各领域合成数据的聚合进行轻量训练（冻结所有专家）。这直接解决了：Instance Merging/TIES/DARE的静态参数无法动态适应不同任务的问题；单体微调的性能权衡（遗忘）问题；传统MoE（如Mixtral用FFN expert）参数膨胀问题（Self-MoE总参数仅增加~1% vs Mixtral 8x7B的47B）。
  **(3) 语义专家的显式区分增强可解释性**：每个expert对应明确的语义领域（知识/推理/数学/编程），路由权重可视化验证了路由器正确将任务分配到对应专家，且能跨领域协同（如推理专家参与数学和编程任务），解释了MiXSE为何超越所有单独专家。
  全栈执行例子（Self-MoE/MiXSE，基于Gemma-7B，单A100-80GB）：
  - **算法Pipeline层**：输入token序列，每层LoRA处：x → θ_r计算4个expert的路由logits → softmax → top-k mask → 加权组合Δθ_i的输出 → 与base θ_0结果相加。路由决策是token级别的（per-token routing），top-1配置下仅激活一个expert（活跃参数7B+0.3%），top-2激活两个。训练时仅优化θ_r（线性层，可忽略参数量），冻结所有ΔΘ_i保持语义区分。
  - **系统框架层**：使用HuggingFace PEFT管理多个LoRA adapter的加载与切换，XLoRA实现MoE-compatible的LoRA路由机制（在每个LoRA层前插入路由层），训练使用Alpaca prompt template，评估通过LM Evaluation Harness调用模型生成并在benchmark上计算准确率/pass@k。整个MiXSE训练仅需约1 GPU天（vs BTX的900 GPU天）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明。

## Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

- baseline方法是什么？
  Baseline方法：(1) **Expert Parallelism（Fairseq GShard, Tutel, DeepSpeed-MoE）**：将所有expert参数加载在GPU显存中，expert通过batched matrix multiplication同时计算，tokens通过all-to-all在GPU间交换到达目标expert所在GPU。需创建dispatch mask（大小为(num_padded_tokens, num_tokens)的稀疏张量）进行token重排，为对齐batch size进行zero-padding。(2) **ZeRO-Offload**：layer-wise将参数和optimizer state卸载到CPU，但MoE场景下单个layer内所有experts超过GPU内存时仍OOM，且CPU optimizer延迟通过delayed update隐藏但引入staleness影响精度。(3) **Token Dropping**：限制每个expert最大token数，超出丢弃，但可能导致>40% token丢失影响收敛。
  全栈执行例子（Baseline: Tutel训练MoE-L 32 experts, 4 GPUs）：
  ```
  训练一个MoE layer（Transformer decoder第L层）：
  ├─ [GPU Kernel] 所有expert参数驻留在4 GPU显存中（每个GPU 8个experts）
  │   问题：MoE-L 32 experts ≈ 14.8B参数，4×A100 40GB共160GB无法容纳
  │   → OOM（论文数据：16 experts时Tutel已OOM on MoE-L）
  │
  ├─ [GPU Kernel] Gating Network: tokens → gate_logits → expert routing
  │
  ├─ [GPU Kernel] Dispatch Mask Creation:
  │   创建(num_padded_tokens, num_tokens)大小的mask
  │   32 experts batch_size=32, 1024 tokens/batch → mask ≥48 GiB
  │
  ├─ [GPU Kernel] Zero-Padding: 负载最重的expert决定batch维度
  │   32 experts时39%的计算用于处理zero-padding → GPU资源浪费
  │
  ├─ [GPU Communication] All-to-All Token Exchange (NVLink)
  │
  ├─ [GPU Kernel] Batched Matrix Multiplication:
  │   - 所有experts同时执行FFN计算（gate_proj, up_proj, down_proj）
  │   - 需要所有expert参数在GPU显存中
  │   - GPU_0上8个expert可能负载严重不均（差距达102%）
  │
  ├─ [CPU/GPU] Layer-wise Optimizer (所有experts的backward完成后再optimizer):
  │   或delayed update：延迟到下一iteration → staleness
  │
  └─ Scalability bottleneck: 加入更多expert需要更多GPU（128 experts需52 GPUs）
  ```
  Baseline缺陷：(1) 所有expert参数必须驻留在GPU显存，scaling expert数量受限于GPU数量和显存容量；(2) batched matrix multiplication需要大的dispatch mask消耗大量显存；(3) zero-padding浪费GPU计算资源，expert越多浪费越严重（32 experts时39%浪费）；(4) 静态expert assignment导致GPU间负载严重不平衡（差距达102%）；(5) CPU optimizer的延迟隐藏引入staleness影响模型精度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：ES-MoE——将expert参数和optimizer state卸载到CPU内存/SSD，通过expert-level流水线处理、动态expert placement和自适应offloading实现大规模MoE训练的可扩展性。
  核心设计：
  (1) **Expert-wise offloading** → 解决缺陷(1)：仅non-expert参数和当前使用中的expert参数在GPU，其余expert在CPU/SSD，突破GPU内存上限，用4 GPU训练29.3B参数模型（67×更多expert）。
  (2) **Sequential expert computation（非batched MM）** → 解决缺陷(2)：消除dispatch mask（不用同时加载所有expert），节省的显存用于增大microbatch（8×更大microbatch → 3.1× throughput）。
  (3) **Dynamic expert placement（greedy scheduling）** → 解决缺陷(3)(4)：按token负载动态分配expert到GPU，消除zero-padding，GPU间负载差异从102%降至15%。
  (4) **Expert-wise CPU optimization** → 解决缺陷(5)：expert粒度而非layer粒度触发optimizer，与backward重叠且无delayed update staleness，GPU利用率提升61.1%。
  (5) **Adaptive offloading（GPU only / CPU offload + pinning / CPU+SSD）** → 小模型避免不必要的offloading overhead，大模型充分利用CPU+SSD。
  全栈执行例子（ES-MoE训练MoE-L 16 experts, 4 GPU, microbatch=32）：
  ```
  训练一个MoE layer（Transformer decoder第L层）：
  ├─ [CPU Memory] 16个expert参数 + optimizer states驻留在CPU（512 GiB DDR4）
  │   GPU显存仅保留：non-expert参数 + 当前active expert + activations
  │
  ├─ [GPU Kernel] Gating Network: tokens → gate_logits → expert routing
  │
  ├─ [CPU Scheduling] Dynamic Expert Placement (<2.69us):
  │   统计per-expert token count → 降序排序 →
  │   Greedy assign: 各GPU总token load均衡（方差<15%）
  │   无需zero-padding！每个expert处理其实际分配的tokens
  │
  ├─ [PIPELINE overlap] Permutation + 1st Expert Upload:
  │   ├─ [GPU Comm] All-to-All Token Exchange (NVLink, ~few ms)
  │   └─ [DMA] GPU_0异步上传E_first（CPU→GPU via PCIe 4.0）
  │       两者并行 → permutation完成时第1个expert已就绪
  │
  ├─ [PIPELINE] Sequential Expert Processing per GPU:
  │   GPU_0: E_a forward → [overlap upload E_b] → E_b forward → ... 
  │   每个GPU顺序处理其分配的experts，计算与前一个expert的通信重叠
  │
  ├─ [GPU Kernel] Token Un-permutation + Weighted Sum
  │
  ├─ [PIPELINE] Backward pass (对称结构)
  │
  └─ [CPU Streaming] Expert-wise Optimizer:
      接近output的expert最先完成backward → optimizer立即在CPU启动
      同时GPU正在计算后面layers → optimizer时间完全被隐藏
      无staleness（exact same iteration的梯度更新）
  
  结果：MoE-L 16 experts throughput = 20,247 tokens/s
  各component贡献：larger batch +301%, optimizer overlap +8.7%, 
                  expert pinning +3.8%, zero-padding elimination +27.4%
  ```

## Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

- baseline方法是什么？
  Baseline方法：(1) **标准Transformer MoE推理**（HuggingFace Transformers）：MoE层执行input permutation（tokens按routing结果重排到各expert tensor，产生额外内存分配memcpy）、各expert独立执行dense GEMM（cuBLAS）、output un-permutation（expert输出写回global memory后加权求和，产生额外I/O）。(2) **vLLM fused MoE kernel**：将多expert计算融合为单个kernel，消除部分permutation开销，但不利用权重稀疏性。(3) **VENOM（SOTA structured sparse）**：利用SpTC加速sparse-dense矩阵乘法，支持灵活V:N:M稀疏格式，但仅处理单端权重稀疏——当输入也稀疏时，跳过行导致I/O amplification和uncoalesced memory access（Figure 6中②③④格式），性能退化严重。
  全栈执行例子（Baseline: vLLM-DS + VENOM权重稀疏，单个token decode）：
  ```
  Prompt请求 → Transformer decoder layer
  ├─ Attention层：FlashAttention2（已优化）
  └─ MoE层（瓶颈，80%+时间）：
      ├─ Router: token → gating scores → select top-k experts
      ├─ Input Permutation: 创建per-expert tensor → memcpy tokens（GPU GMEM allocation+copy）
      ├─ Expert计算（VENOM kernel，单端稀疏）:
      │   - gate_proj: sparse(W_gate_VENOM) × dense(input) → dense output
      │   - up_proj: sparse(W_up_VENOM) × dense(input) → dense output  
      │   - SiLU(gate) * up（separate kernel launch）
      │   - down_proj: sparse(W_down_VENOM) × dense(hidden) → dense output
      │   ★ 问题：input tensors中大量零行（未路由token）仍参与计算
      │   ★ I/O amplification：SEL跳过列导致加载多余数据（②③），或非连续访问（④）
      ├─ Output Un-permutation: expert_outputs → GMEM → reload → weighted_sum（额外I/O roundtrip）
      └─ 结果：内存管理开销+冗余计算+非连续访存 → VENOM在高稀疏输入时性能退化
  ```
  Baseline缺陷：(1) 输入permutation/un-permutation产生额外GMEM分配和数据搬运；(2) 单端稀疏无法利用MoE路由产生的激活稀疏性；(3) VENOM的dual-side稀疏场景下存在I/O amplification和uncoalesced access；(4) 稀疏格式与SpTC硬件未充分对齐，导致硬件利用率不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：Samoyeds系统——提出双端结构化稀疏数据格式+定制sparse-sparse kernel+四项系统优化。
  核心设计：(1) **双端稀疏格式（N:M:V）**：权重端=(M×V block内N个Sub-Row) × (2:4 element-wise per Sub-Row)；激活端=vector-wise稀疏通过SEL数组记录routing结果。数学等价于原始MoE计算。(2) **定制sparse-sparse kernel**：基于PTX mma.sp指令直接调用SpTC执行双端稀疏MMA，pipeline机制overlap fetch和compute。(3) **四项优化**：3-step hierarchical tiling、data stationary ($C_{IR}$中间寄存器shuffle)、data packing（metadata 2-bit→32-bit映射）、optimized layout（offline weight transpose + in-kernel input transpose + compressed output）。
  全栈执行例子（Samoyeds，同场景单个token decode）：
  ```
  Prompt请求 → Transformer decoder layer（Samoyeds优化后）
  ├─ Attention层：FlashAttention2（不变）
  └─ MoE层（Samoyeds优化）：
      ├─ Router: token → gating scores → 生成SEL（仅记录indices，无内存拷贝）
      ├─ Expert计算（Samoyeds kernel，双端稀疏+fused）:
      │   - 权重预编码（offline）：原始W → (data, indices, metadata) Samoyeds格式
      │   - Kernel执行（单个kernel覆盖gate+up+down+act+acc）:
      │       1. 加载SEL→SMEM，识别有效token columns
      │       2. Pipeline: 异步加载A_tile(编码权重), B_tile(仅有效token列), indices→SMEM
      │       3. 3-step tiling: block_tile(m_b×n_b)→warp_tile(m_w×n_w)→SpTC_tile(m16×n8×k32)
      │       4. ldmatrix按SpTC spec排列数据到register
      │       5. mma.sp: SpTC执行 M_sparse × N_sparse → P（硬件2:4加速）
      │       6. 每V/k_h步shuffle C register（data stationary，避免写回GMEM）
      │       7. gate_proj→SiLU→×up_proj→down_proj 全在kernel内fused
      │       8. Weighted accumulation fused: output += router_score * expert_C
      │       9. 压缩output写入GMEM（仅非零行）
      └─ 结果：无input permutation开销 + 无output roundtrip + SpTC双端加速
  ```
  对比baseline的关键改进：
  - 输入permutation消除 → 直接通过SEL在kernel内索引，零内存拷贝（对比baseline的GMEM alloc+memcpy）
  - 双端稀疏 → weight稀疏（2:4 SpTC加速 × N:M:V灵活稀疏比） + input稀疏（仅计算路由到的tokens），解决VENOM在dual-side稀疏时I/O amplification问题
  - Operator fusion → activation+weighted accumulation+matmul融合，消除多次kernel launch和中间GMEM roundtrip
  - Data stationary → $C_{IR}$中间寄存器避免C频繁写回，保持C在register中跨越Sub-Row边界
  - Packing + Layout → metadata对齐32-bit transaction，B矩阵转置packing coalesced access，offline weight transpose消runtime开销
  - 量化收益：kernel级up to 1.99× vs VENOM，模型级up to 1.58× vs vLLM-DS，最大batch size 4.41× average boost

## SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts

- baseline方法是什么？
  Baseline方法：(1) **GPU上的传统AI加速器架构**（NVIDIA DGX A100 / DGX H100）：不具备三级存储体系（仅有HBM+host DRAM via PCIe），无法高效部署Composition of Experts系统。GPU operator fusion受限于rigid memory hierarchy（SM间仅通过shared cache/HBM交换数据）、on-chip SRAM容量不足、SIMT编程模型不支持跨算子的pipeline parallelism。全栈执行例子（CoE推理在DGX H100上）：prompt请求→ router在GPU HBM中执行→需切换expert时从host DRAM通过PCIe拷贝权重（DGX H100: 64 GB/s）→expert在GPU上以多个独立kernel执行（prefill/attention/MLP各为独立kernel launch，中间结果materialize到HBM）→自回归decoding每次迭代重复加载权重和KV cache→超过50个expert时HBM溢出到host DRAM导致切换延迟剧增→DGX在150 experts时OOM。
  全栈执行例子（Baseline: DGX H100 CoE推理，single token decode）：
  ```
  # 算法层：标准Llama2-7B decoder layer，attention + FFN
  # 系统框架层：PyTorch/TensorRT，每个operator→独立CUDA kernel
  # GPU执行：HBM→L2→SM→HBM，intermediate materialization
  # 编译框架层：TensorRT operator fusion（1-5 operators，access pattern受限）
  # Kernel调度层：CUDA kernel launch开销 + software orchestration
  # 硬件架构层：H100 HBM (3.35 TB/s) + host DRAM via PCIe (64 GB/s)
  # 缺陷：
  # 1. 无法融合含transpose的arbitrary access pattern → 操作强度低（39.5 Ops/Byte），memory-bound
  # 2. HBM容量有限（80 GB），150个7B experts无法全部驻留
  # 3. 模型切换走PCIe（64 GB/s），延迟高（Figure 1）
  # 4. Decode kernel短，CUDA kernel launch开销占比高
  # 5. 无pipeline parallelism → 小矩阵乘法（32×32×32）无法充分利用所有SM
  ```

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**SN40L Reconfigurable Dataflow Unit + 三级存储体系 + Streaming Dataflow + Samba-CoE**。核心设计：(1) Streaming dataflow — coarse-grained pipeline执行算子图，tensor被tiled并streaming通过PCU/PMU pipeline，transpose等复杂access pattern通过PMU data alignment unit在buffer内实现（in-place），无需HBM materialization；(2) 三级存储体系 — 520 MiB片上SRAM(PMU) + 64 GiB HBM + 1.5 TiB DDR，CoE专家参数存DDR、活跃专家拷贝到HBM（聚合带宽>1 TB/s vs GPU 32-64 GB/s）、Router+KV cache常驻HBM；(3) 自动空间融合 — 编译器将整个decoder layer融合为单个kernel调用，20+ operators在一个kernel内完成；(4) 硬件orchestrated kernel launch — AGCUs硬件调度kernel序列，消除host software调度开销，对短执行时间的decode kernel效果显著（1.4×-8× speedup）。
  全栈执行例子（对比baseline，SN40L CoE推理，single token decode）：
  ```
  # 算法层：相同Llama2-7B decoder layer（不改模型架构）
  # 系统框架层：Samba-CoE Runtime — Router常驻HBM，expert权重按需 DDR→HBM>1 TB/s
  # → CoE Runtime LRU管理HBM中活跃expert，读多写少weight跳过回写
  # 编译框架层：SambaNova编译器自动融合 — 将PyTorch算子图编译为空间融合dataflow kernel
  # → 静态符号生命周期分析实现garbage collection
  # → 静态带宽建模控制RDN/TLN并发流资源分配
  # → Place-and-Route配置RDN路由、flow ID、multicast路径
  # Kernel调度层（硬件orchestrated）：单个Kernel Execute命令发出后
  # → AGCUs硬件执行静态kernel schedule序列，无需host参与
  # → HBM→AGCU→RDN→PCU(systolic GEMM for QKV)→PMU(stage buffer)→PCU(SIMD activation)→PMU→...
  # → 整个decoder layer在一个kernel内完成，无中间结果materialize
  # 硬件架构层：PCU(systolic/SIMD双模)+PMU(scratchpad+地址生成+data alignment)+RDN(mesh+flow control)
  # → PMU data alignment unit: transpose通过diagonally striped write实现（无需数据搬移）
  # → RDN credit-based flow control + sequence ID重排序
  # → AGCU桥接片上RDN与片外HBM/DDR
  # → Die-to-die接口: 两个RDD间直接流式传输
  # → Peer-to-peer: 跨socket direct RDU通信（AllReduce等collective fused进kernel）
  # 
  # 解决Baseline缺陷：
  # 1. 操作强度：39.5→410.4 Ops/Byte（全面融合），FlashFFTConv 13× speedup
  # 2. 模型切换延迟：DDR→HBM >1 TB/s vs PCIe 32-64 GB/s → 15×-31× 切换加速
  # 3. HBM带宽利用率：~85%（dataflow重叠weight load与compute）vs GPU ~50%
  # 4. Kernel launch开销：硬件orchestrated消除host往返（decode阶段1.4×-8× speedup）
  # 5. System footprint：单Node支持850 experts (TP8) vs DGX需19节点 → 19× footprint缩减
  # 6. 150+ experts: DGX OOM, SN40L正常服务
  # 7. 小模型pipeline parallelism：PCU间chaining实现pipeline，利用多个小GEMM的并行性
  ```

## ST-MoE: Designing Stable and Transferable Sparse Expert Models

- baseline方法是什么？
  - Baseline 方法：(1) **Dense T5 模型**（T5-Large, T5-XXL）：标准 Transformer encoder-decoder，所有 token 经过相同的 FFN 层，无稀疏 MoE 路由。全栈执行例子：输入 token → embedding lookup → 27 层 encoder（每层 self-attention → RMSNorm → FFN_{GEGLU} → dropout → residual add），24 层 decoder（含 enc-dec attention），输出 logits → softmax 预测。硬件为 TPU，Mesh Tensorflow 分布式训练，bfloat16 mixed precision，Adafactor optimizer。(2) **Switch Transformer**（Fedus et al., 2021）：token-based top-1 routing，单 expert 路由，expert 容量限制，训练不稳定且微调质量低于密集模型。全栈执行例子：token → router 计算 softmax → 选择 top-1 expert → 若 expert 容量未满则 dispatch via all2all → GEGLU FFN 计算 → combine via all2all → residual add（若 token 被 drop 则跳过计算直接 residual）。核心缺陷：训练 1/3 运行不稳定、微调质量落后于密集模型（尤其在 SuperGLUE 等小任务上）、训练时无有效的稳定化机制。
  - 全栈执行例子（Baseline: Switch Transformer / Dense T5, TPU, single token）：
    ```
    # 算法层：标准 MoE routing，token 进入 → router softmax → top-1 dispatch → GEGLU FFN
    router_logits = W_r @ x  # [d_model, num_experts]
    gate = softmax(router_logits)  # float32 selective precision
    expert_id = argmax(gate)  # top-1 routing
    # 无稳定化 loss，logits 可能非常大 → bfloat16 roundoff error 累积
    
    # 系统框架层（Mesh Tensorflow）：
    # Data parallelism over rows, Model parallelism over columns
    # einsum one-hot tensor dispatch/combine, group-based load balancing
    group_size = batch / num_groups  # 通常 2-8 sequences/group
    token_capacity = CF * group_size / num_experts  # train CF=1.25
    
    # 编译框架层：Mesh Tensorflow 编译为 TPU XLA，einsum → all2all → matmul → all2all
    # Kernel 调度层：bfloat16 matmul on TPU MXU (systolic array)，float32 allreduce
    # 硬件架构层：TPU v3/v4 cores，HBM，interconnect for all2all
    
    # === 缺陷 ===
    # 1. 训练不稳定 → 大 logits 进入 router softmax → bfloat16 roundoff errors
    # 2. 微调过拟合 → 稀疏模型收敛快但在小任务(CB/WSC)上验证质量差
    # 3. 微调超参数 → 使用密集模型最优超参数(batch_size/learning_rate)会掩盖稀疏改进
    ```

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：**ST-MoE（Stable and Transferable Mixture-of-Experts）**通过三个维度解决 baseline 缺陷：
    - **(A) Router Z-Loss 稳定训练**：引入 $L_z(x) = \frac{1}{B} \sum_i (\log \sum_j e^{x_j^{(i)}})^2$ 作为辅助损失，直接约束路由器输出 logits 的绝对值大小。这解决了 routing 时 bfloat16 roundoff errors 导致的训练不稳定问题（大 logits 在 softmax 中的微小扰动会巨大改变输出）。Router z-loss 在不损害模型质量的前提下使训练 100% 稳定，甚至轻微提升质量（neg log perp 从 -1.755 → -1.741）。相比之下，其他稳定方法（移除 GEGLU、加 noise、tight update clipping）都严重损害质量。
    - **(B) 差异化微调协议**：发现稀疏模型需要与密集模型不同的微调超参数——更小的 batch size（65k vs 1M）和更高的 learning rate。提出仅更新非 MoE 参数子集的微调策略以对抗过拟合。这解决了稀疏模型微调质量差的问题，使得 ST-MoE-32B 在 SuperGLUE 上达到 91.2 的 SOTA（超过人类 baseline）。
    - **(C) 架构设计原则**：sparse-dense stacking（每个 MoE 层前后加 dense FFN 层，改善 -0.014 neg log perp）、multiplicative bias（expert FFN 中加逐元素乘性偏置，4% 收敛加速）、实际 capacity factor 考量（train CF=1.25 在 Pareto 效率上优于 CF=2.0，后者导致 +14% step time）、以及 BPR routing for low CF。
  - 全栈执行例子（ST-MoE-32B, TPU, single token）：
    ```
    # === 算法层 ===
    x = embedding(token_id)  # d_model=5120
    
    # Router Z-Loss (training only, not inference)
    router_logits = W_r @ x  # [5120, 64]
    z_loss = mean(logsumexp(router_logits)**2)  # 惩罚大logits
    total_loss = CE_loss + 0.01 * load_balance_loss + 0.001 * z_loss
    
    # Top-2 routing with capacity factor and relative threshold
    gate = softmax(router_logits)  # cast to float32
    top2_gates, top2_indices = top_k(gate, k=2)
    top2_gates = top2_gates / sum(top2_gates)  # renormalize
    # 2nd expert routed if gate2/gate1 > 0.2, else stochastic
    
    # Sparse-Dense Stacked FFN
    x = DenseFFN(x)                    # 先经过 dense FFN
    # Multiplicative Bias in Expert
    for expert_i in top2_indices:
      expert_out = [GELU(x @ W_11) * (x @ W_12) * B_scale] @ W_2
      # B_scale: learnable [1, d_ff] initialized to ones, 逐元素乘
    y = DenseFFN(sum(gate_i * expert_out_i))  # 后接 dense FFN
    
    # === 系统框架层（Mesh Tensorflow）===
    # 2D mesh: d x m (data x model parallelism)
    # 64 experts, expert layer frequency = 1/4
    # train CF=1.25 (Pareto optimal), eval CF=2.0
    # all2all dispatch/combine, allreduce for model parallelism
    # 3D mesh when num_experts < data_parallel_rows
    
    # === 微调策略 ===
    # 仅更新 Non-MoE 参数 (Attention + Non-MoE FFN params, ~20% of total)
    # batch_size=65k, learning_rate=1e-3 (sparse 偏好更小 batch, 更高 lr)
    # dropout=0.1 global, 额外 expert dropout
    # aux load balancing loss 在微调时可关闭（不影响质量）
    
    # === 编译框架层 ===
    # Mesh Tensorflow → XLA → TPU (bfloat16 matmul, float32 accumulate)
    # === Kernel 调度层 ===
    # TPU MXU: bfloat16 matrix multiply, GELU via lookup table
    # === 硬件架构层 ===
    # Google TPU v3/v4, HBM, inter-chip interconnect (ICI)
    ```

  - 解决效果对比：
    | 缺陷 | Baseline | ST-MoE 解决方案 | 效果 |
    |------|----------|-----------------|------|
    | 训练不稳定 | 1/3 runs diverge | router z-loss | 100% stable, 质量略微提升 |
    | 微调质量差 | 密集模型更好 (小任务) | 差异化微调协议 + 参数子集更新 | SOTA on SuperGLUE (91.2) |
    | 微调过拟合 | CB: 100% train acc, 低 validation | 更小 batch size, 更高 lr | ST-MoE-L 在所有 task 上超越 Dense-L |
    | CF 过大低效 | CF=2.0 慢 14% | train CF=1.25 Pareto 选择 | 几乎相同的质量，显著降低计算成本 |
    | 小规模结论不迁移 | top-1 vs top-2 结论反转 | 更大规模 (8x) 实验验证 | top-2 在更大规模下优于 top-1 |

## S'MoRE: Structural Mixture of Residual Experts for LLM Fine-tuning

- baseline方法是什么？
  - Baseline 方法有三类：(1) **LoRA**：在每个 transformer 层的权重矩阵旁插入低秩 adapter $x' = B \cdot A \cdot x$（$A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times d}$），参数效率高（仅 $2dr$ 可训练参数），但模型容量受限于扁平的单层低秩结构，无法根据 token 特性动态调整计算路径。(2) **MixLoRA (MoLRE)**：将多个 LoRA expert 组合为 flat MoE 层 $x' = \sum_{i=1}^s \text{ROUTE}(x)^i \cdot B^i \cdot A^i \cdot x$，通过 top-k 路由为不同 token 激活不同 expert 组合。虽然增加了容量，但 expert 之间缺乏结构关系，路由灵活性仅来自"选择哪些 expert"，无法利用 expert 之间的连接方式产生额外的表达能力。(3) **HydraLoRA**：将 LoRA 的 up-projection 矩阵 B 拆分为多个 head，通过 dense gate 加权组合多 head 输出。类似 MoE 变体，但仍是单层结构，且参数利用率低（增加参数不提升准确率）。
  - 全栈执行例子（Baseline: MixLoRA on LLaMA 3-8B, A100 GPU, single token）：
    - 算法层：token embedding x 输入 flat MoE → 路由器对 s=8 个 expert 打分 → 选择 top-2 → 计算 $x' = \alpha_1 \cdot B^1 A^1 x + \alpha_2 \cdot B^2 A^2 x$ → 加回 frozen pre-trained 输出。不同 token 可激活不同 expert 对，但所有 8 选 2 组合 = $\binom{8}{2}=28$ 种可能输出。
    - 系统框架层：基于 HuggingFace PEFT + LLaMA-Factory SFT pipeline 训练，PyTorch 原生实现。
    - 编译框架层：论文未明确说明。
    - kernel调度层：论文未明确说明。
    - 硬件架构层：4× NVIDIA A100 80GB GPU，标准 PyTorch CUDA kernel 执行矩阵乘法。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：**S'MoRE** 通过三个核心设计解决 baseline 缺陷：
    (1) **层次化结构式专家混合** 解决 "expert 缺乏结构关系导致路由灵活性受限" 的缺陷：将 expert 权重分解为多阶低秩残差 $W^i \approx \sum_{\ell=0}^{L-1} B^i_\ell A^i_\ell$，排列为 L 层结构。不同层的残差通过层间投影矩阵 $W_\ell$ 和 skip connection（原始 token x 直连每层残差）互联。同一组 activated expert 可以形成指数级多种非同构树结构，每种产生不同输出，从"选择哪些 expert"扩展为"expert 如何连接"。
    (2) **树形条件路由** 解决 "flat MoE 路由仅做 flat 选择" 的缺陷：路由器逐层自顶向下选择子节点，每层路由条件于祖先路径 $p(i_{\ell-1} \mid i_{L-1}, \dots, i_\ell, x)$，为每个 token 定制深度为 L 的专属激活树。路由概率通过 learnable key-query dot product + MLP 计算，支持 dense/sparse noisy top-k/switch 三种 gate 类型。
    (3) **非线性激活保证结构区分能力** 解决 "无结构关系时不同连接方式产生相同输出" 的缺陷：在每层聚合公式中引入非线性 $\sigma$（如 ReLU），使 S'MoRE 的 L 层传播模拟 Graph Isomorphism Network (GIN) 的 L 轮 Weisfeiler-Lehman (WL) test，理论上保证所有非同构树产生不同输出（Theorem 3.4）。无 $\sigma$ 时退化为 MoMOR，无法区分 Fig.3 中的非同构图。理论证明 S'MoRE 的 structural flexibility $\Gamma_{\text{S'MoRE}} = \prod_{\ell=0}^{L-1} \binom{s_\ell}{f_\ell}^{F_{\ell+1}}$ 比 MoMOR 上界 $\Gamma_{\text{MoMOR}}$ 呈指数级增长（Fig.2）。
  - 全栈执行例子（Method: 2-layer S'MoRE on LLaMA 3-8B, A100 GPU, single token）：
    - 算法层：token x 输入 → $x_{\text{down}} = W_{\text{down}} x$（降至 24 维）→ Layer 2 router: MLP₂ 生成 query，与 4 个 key vector 点积得 softmax score → 选 top-2 顶层 expert → Layer 1 router: 对每个父 expert，MLP₁(concat($x_{\text{down}}$, 父 key)) 生成 query → 选 top-2 子 expert → 形成激活树（2 个父节点各含 2 个子节点，共 4 条路径）→ 自底向上聚合：Layer 1 计算 $x_1^{p} = \sum_{n} \alpha_0^{p,n} \cdot \text{ReLU}(B_0^n A_0^n x)$ → Layer 2 计算 $x_2 = \sum_i \sum_n \alpha_1^{i,n} \cdot \text{ReLU}(B_1^n A_1^n x + W_1 x_1^{i \to n})$ → 最终投影 $x' = W_{\text{proj}} x_2$。不同 token 可获得 $\binom{4}{2} \times \binom{4}{2}^2 = 6 \times 6^2 = 216$ 种结构不同的激活树，远超 MixLoRA 的 $\binom{8}{2}=28$ 种。
    - 系统框架层：基于 HuggingFace PEFT 自定义 adapter 实现，LLaMA-Factory SFT pipeline 训练，OpenCompass 评估，PyTorch 原生实现。
    - 编译框架层：论文未明确说明。未来可集成到 vLLM 或 LMDeploy 等推理框架。
    - kernel调度层：论文未明确说明。论文提及可通过 CUDA kernel fusion 合并多层操作减少 kernel launch 开销，token-level parallelism（Triton kernel 或 torch.compile）交错不同层处理提升 GPU 利用率。
    - 硬件架构层：4× NVIDIA A100 80GB GPU。训练 wall-clock time 仅比 MixLoRA 增加约 24%（平均），router 计算代价相对 expert 传播最多 26%。

## Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing

- baseline方法是什么？
  - Baseline 有四类：(1) **CPU-only 部署**：整个 MoE 模型部署在 CPU 上，所有 expert 权重常驻 CPU 内存，无 expert offloading，但推理延迟极高；(2) **GPU-only 部署**：整个 MoE 模型部署在 GPU 上，所有 expert 权重常驻 GPU 显存，低频 expert 占用昂贵 GPU 内存造成成本浪费；(3) **Expert Offloading（Fetch/MIX）**：Fetch 假设理想 expert offloading（无 misprediction、无 offloading/reloading 时间），所有 expert 缓存于 CPU 内存并按需交换到 GPU；MIX 将 expert 模块部署在 CPU、非 expert 模块部署在 GPU，但 CPU/GPU 内存需足以缓存所有模块。两者均需持续分配大块 CPU 内存持有 inactive experts；(4) **Per-expert Serverless Function [14]**：将每个 expert 作为独立 serverless function，对 Deepseek-V3 (256 experts × 61 layers) 等现代 MoE 完全不可行。
  - 全栈执行例子（Baseline: MIX with GPU+CPU, Deepseek-v2-lite on A100, 500 token input）：
    - 算法层：无 expert 激活预测，所有 64 experts/层 权重常驻 CPU 内存
    - 系统框架层：Kubernetes 调度单 Pod（GPU + CPU），模型以单体 serverless function 部署
    - 编译框架层：论文未明确说明
    - kernel调度层：论文未明确说明
    - 硬件层：GPU 执行 Attention 和 Shared Experts（A100 Tensor Cores），token embedding 通过 PCIe 传输到 CPU，CPU 执行 top-6 experts 的 FFN → 结果回传 GPU → 重复 27 层 → 持续占用 CPU 内存持有所有 64×27=1728 个 expert 权重

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：**Remoe** 异构 serverless MoE 推理系统，通过五个关键设计解决 baseline 缺陷：
    (1) **SPS 算法**解决 "expert 激活无法预知导致无法预分配资源" 缺陷：利用 prompt 语义相似度（SCS）+ 多叉聚类树实现 prompt 级 expert 激活预测，将预测提前到请求到达时完成，避免 token-by-token 在线预测在 serverless 环境下导致的冷启动开销；(2) **异构架构**解决 "所有 expert 常驻内存造成浪费" 缺陷：非 expert 模块在 GPU，expert 模块在 CPU，仅高频 "local experts" 与主模型同容器，低频 "remote experts" 部署为独立 serverless function（按需冷启动，pay-per-use），消除低频 expert 的持续内存占用；(3) **MMP 算法**解决 "serverless 冷启动时无法保证 SLO" 缺陷：基于霍夫丁不等式证明最坏情况 expert 负载上界（Theorem 1: per-expert token ≤ √(3n)/2 + n/K_l），在最坏情况下仍保证 TTFT/TPOT；(4) **Lagrangian 对偶优化**解决 "remote expert 内存规格选择是 NP-hard" 的缺陷：将离散内存规格连续化，构造指数衰减拟合函数 T̃=θ1·exp(-θ2·ỹ)+θ3，分析凹凸性（Theorem 2），利用 Slater 条件和 KKT 条件（Lemma 1, Theorem 3）找到全局最优解；(5) **LPT 多 replica 划分**解决 "多 replica 下 expert 子集划分导致负载不均" 缺陷：建模为 Multiway Number Partitioning 问题，LPT 算法 O(n log n) 复杂度，近似比 (4/3 - 1/3z_l)。
  - 全栈执行例子（Method: Remoe, Deepseek-v2-lite on A100, 500 token input, 200 output tokens）：
    - 算法层：请求到达 → SPS 预测 expert 激活矩阵 S_pred [27, 64] → utility score u_{l,k} 排序 → b=15% 的 experts 为 remote（约 10/64 per layer）
    - 系统框架层：MMP 用 Theorem 1 上界计算主模型最小内存 w_v → Kubernetes 调度主模型 Pod（GPU+CPU）→ Lagrangian 优化确定 remote experts 每层内存 ỹ_l → LPT 划分 remote expert 集到 z_l 个 replica → Kubernetes 创建 remote expert Pods（CPU-only）→ 冷启动并行（主模型启动 ∥ remote experts 冷启动 ∥ 优化计算）
    - 编译框架层：论文未明确说明
    - kernel调度层：C++ LibTorch 实现 expert FFN 计算，gRPC 传输 token embedding（token size 10KB < 6MB payload limit，无需中间存储）
    - 硬件层：GPU (A100) 执行 Attention + Shared Experts → token embedding 通过 gRPC/PCEe 发送到 remote expert Pods → CPU 并行执行 local experts (high-frequency) + remote experts (low-frequency, separate Pods) → 合并输出 → 27 层迭代 → 仅主模型 + 高频 experts 常驻内存，低频 experts 按需付费

## Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design

- baseline方法是什么？
  - Baseline 有两类：(1) **Open-source dense models**（Pythia 2.8B/6.9B, Open-Llama-v2 3.4B/6.9B）和 **dense compression methods**（Sheared-Llama 2.7B, LLM-Pruner, SliceGPT, LaCo, Compresso），它们将大模型压缩为更小的 dense 模型，完全依赖 dense FFN 计算，无法利用 MoE 的稀疏激活优势；(2) **传统 layerwise MoE**（Mixtral-8×7B, OpenMoE），每层配备独立 router G^(l)，基于第 l-1 层 hidden states 动态决定第 l 层的 expert 选择，导致系统层两个根本问题：Memory Management——无法预知未来层所需 expert，只能依赖 naive prefetching（如前一层 hidden states 预测下一层 expert + LRU cache）或 on-demand loading（在关键推理路径上增加加载延迟）；Token Batching——每层每个 token 可能选择不同 expert，导致一个 batch 内需激活大量 unique expert（Mixtral-8×7B 在 batch_size=56.8 时平均激活 7.63/8 experts），batch 内 token 必须等待所有 expert 计算完成，batching 效率退化至接近无 batching。
  - 全栈执行例子（Baseline: Mixtral-8×7B layerwise MoE, 单卡 A100-80GB, Chatbot Arena workload）：
    - **算法层**：Mixtral 采用 standard layerwise top-K routing（N=8, K=2），每层 router 为 linear layer W_r ∈ R^{d_model×N}，激活函数 softmax。第 l 层输出：y = Σ_{i=1}^N I(top-K G^{(l)}(x)) · G^{(l)}(x)_i · F_i^{(l)}(x)。Router 参数 32 layers × (4096×8) ≈ 1M params。
    - **系统框架层**：HuggingFace Transformers / vLLM 等标准推理框架。Tokens 组成 batch，每层对所有 expert 的 FFN 按 selected experts scatter-gather：scatter tokens 到对应 expert → expert FFN 计算 → gather 结果按 gating weights 加权求和。Token 间存在 implicit barrier——所有 expert 计算完成后才能进入下一层。
    - **编译框架层**：论文未明确说明。使用 PyTorch JIT / torch.compile。
    - **kernel调度层**：标准 cuBLAS GEMM kernel。Expert FFN 计算为：gate_proj(x) → SiLU(gate) ⊙ up_proj(x) → down_proj(result)。每个 activated expert 执行一次 GEMM。Batch 内不同 token 分散到不同 expert 导致多个小 GEMM（非合并大 GEMM），kernel launch overhead 和低 GPU 占用率。Expert weights 保持在 GPU memory 或通过 LRU cache + speculative prefetching 部分加载。
    - **硬件架构层**：单卡 A100-80GB。Expert weights 以 FP16 存储于 GPU HBM（~94GB for 8 experts × 32 layers）。若 GPU memory 不足 → offload 未使用 expert 到 host memory（PCIe 4.0 ~25 GB/s），推理时按需加载 → 加载延迟在关键路径上。
  - Baseline 核心缺陷：(1) **逐层 Router 冗余**——相邻层 expert 选择高度相关（transition matrix 稀疏，MI 高），独立 router 浪费参数且阻止系统预调度；(2) **Expert 预取不可靠**——基于前一层 hidden states 预测下一层 expert，假设不成立时导致 cache miss penalty；(3) **Batching 低效**——batch 内 unique expert 过多（~7.63/8），token 间同步等待主导延迟；(4) **Cache 策略次优**——LRU 基于单请求 temporal locality，跨请求共享 cache 时失效。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Read-ME 通过 **Router-Decoupled MoE Architecture + System Co-Design** 两大创新系统性解决 baseline 缺陷：(1) **架构层**：将 dense LLM refactorize 为 MoE 并采用跨层共享的 pre-gating router，消除逐层 router 冗余，使 expert 选择可在推理前一次性确定；(2) **系统层**：利用 pre-gating 先验信息实现 expert-aware batching、fine-grained prefetching 和 Belady 最优缓存。
  - 全栈执行例子（Read-ME 4.7B-17B, 单卡 A100-80GB, Chatbot Arena workload）：
    - **算法层（解决"逐层 Router 冗余"和"Dense 模型效率低"的缺陷）**：
      - **Domain-Aware Expert Construction**：利用 activation sparsity 从 Llama2-7B-chat 构建 8 个 domain expert + 1 permanent expert。对每个子域数据 D_i，选择 top-d 激活通道（d≈D/2），通过结构化 pruning 初始化 expert F_i(x) = W_2 M_i^T σ(M_i W_1 x)，选择矩阵 M_i 最大化 E[||M_i W_1 x||_1]。
      - **Pre-Gating Router（核心架构创新）**：用单一 1-layer Transformer block（18M params）替代 32 个逐层 router。Router G 以 causal attention 处理 x_{≤t}，输出跨所有层统一的 expert 选择：y_t = Σ_{i=1}^N I(top-K G(x_{≤t})) · G(x_{≤t})_i · F_i^{(l)}(x_t)。Router 与 layer index l 无关——同一 token 在所有层选择相同的 expert。
      - **对比 baseline**：Mixtral 每层独立 router → 32 个无关 routing path；Read-ME 单一 router → 专家选择跨层一致，消除路由歧义。仅 18M router params vs Mixtral ~1M（但 Read-ME router 贡献 0.4% 延迟 vs Mixtral 3.95%）。
      - **Routing Distillation Loss**：L_RD = KL(softmax(G) || softmax([||M_0 M_1^T||_F^2, ...]))，利用 dense 模型激活稀疏性指导 router 学习，加速收敛。
      - **关键结果**：仅 1.04B training tokens 达到 MMLU 38.9%（Sheared-Llama 50B tokens 仅 26.4%），平均 accuracy 55.5% 超越同规模所有 baseline。MoE 结构（4.7B activated）比同等大小 dense 模型 MMLU +11.8%。
    - **系统框架层（解决"Expert 预取不可靠"和"Batching 低效"的缺陷）**：
      - **Expert-aware Batching**：修改 DeepSpeed inference engine。Pre-gating 后，Scheduler 按 ReqQueueByExpert 收集选择同一 expert 的 tokens → 组 batch 确保 batch 内所有 token 共享同一 expert。Algorithm 1 从请求最多的 expert 开始取 tokens。
      - **Fine-grained Prefetching**：利用 pre-gating 预知所有层 expert 需求 → compute stream（第 i 层 FFN）与 loading stream（第 i+1 层 expert 传输）流水线重叠，隐藏 PCIe 加载延迟。
      - **Belady-inspired Optimal Caching**：因 pre-gating 预知所有 future expert references → 可精确计算 F(e,t)（expert e 的下次访问时间）→ 实施 Belady 最优驱逐策略 (evict argmax F(e,t))。Cache 跨所有并发请求共享。
      - **对比 baseline**：Baseline 基于前一层 hidden states "猜测"下一层 expert + LRU cache → 跨请求时 LRU 对 temporal locality 假设失效（Table 4: LRU hit ratio 66.95% vs Belady 77.21% at capacity=4）。Baseline batching 平均 5.08-5.21 unique experts/batch → Read-ME 3.51 experts/batch。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 cuBLAS GEMM kernel。与 baseline 相同使用 SwiGLU FFN 计算（gate_proj + up_proj → SiLU(gate)⊙up → down_proj）。但因 expert-aware batching 减少 batch 内 unique experts → 增大有效 batch size → 更高 GPU 占用率。Prefetching 将 expert 加载与计算重叠 → 加载延迟不进入关键路径。
    - **硬件架构层**：单卡 A100-80GB。与 baseline 相同的物理硬件。Read-ME 通过 pre-gating + Belady cache + prefetch 使同硬件下的端到端平均延迟降低 5.0-6.1%，p95 延迟降低 9.5-10.0%。Prefetching 模式在 cache capacity 受限时比 On-demand Loading 延迟低最多 30%。Router 计算开销仅 0.4% vs 传统 MoE router 3.95%。
  - **方法核心 insight**：将路由决策从"每层交互式"变为"推理前一次性"，使得系统可以在推理开始前完全掌握 token-to-expert 的全路径映射，从而将 expert 调度从"反应式"（reactive）升级为"前瞻式"（lookahead），同时通过算法-系统 co-design 将 MoE 的稀疏性优势从算法层贯穿至系统层。

## PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed Inference

- baseline方法是什么？
  - Baseline 方法分为两类：(1) **Expert Dropping**（如 NAEE, STUN）——在 calibration dataset 上评估各 expert 重要性，直接移除被认为不重要的整组 expert 参数。但不同下游任务需不同 calibration data——NAEE 对 commonsense benchmarks 用 C4 校准，对 math tasks 需换 MATH 数据集，且校准数据选择严重影响模型精度。(2) **Expert Merging**（如 HC-SMoE, D2, Sub-MoE）——通过 hierarchical clustering 或多阶段合并（先 clustering 再低秩近似）合并相似 expert。但它们采用 coarse-grained 合并，将整组 expert 权重聚合，破坏了 expert 间的关键区分。D2 和 Sub-MoE 还需要 SVD 分解等重计算操作。
  - 全栈执行例子（Baseline: HC-SMoE, Mixtral-8x7B, 50% sparsity, 2×A100-80GB）：
    - **算法层**：HC-SMoE 基于 expert 输出相似度进行 hierarchical clustering：(1) 在 calibration data 上收集各 expert 的 output activation；(2) 计算 expert 间的 cosine similarity 构建距离矩阵；(3) 使用 agglomerative clustering 将 expert 按相似度层次合并；(4) 每个 cluster 内对 expert 权重做直接平均 W_merged = (W_i + W_j)/2 生成 merged expert。这种方式不区分 shared knowledge（共享权重）和 expert-specific knowledge（专有权重），coarse-grained 平均化导致 MMLU 从 67.9% 骤降至 49.0%（-18.9 points）。
    - **系统框架层**：论文未明确说明 HC-SMoE 使用的推理框架。压缩后加载 merged model checkpoint 进行标准 autoregressive decoding。无 specialized inference kernel——使用标准 PyTorch dense GEMM。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 cuBLAS GEMM kernel。Expert weights 以 Bfloat16 格式存储。50% sparsity（压缩后 expert 数量减半）的 merge 操作完成后，inference 使用标准 dense GEMM 计算，不涉及 sparse 计算或自定义 kernel。但 Baselines 的 mask/sign 存储引入额外 metadata 开销——Lasby et al. (2025) 指出 CSR 格式存 50% unstructured sparsity 无 memory savings。
    - **硬件架构层**：2×A100-80GB（tensor parallelism 分片 expert 权重到两卡）。压缩后仍需 2 GPUs（仅减少 expert 数量的一半 weight，而非减少 attention/embedding 部分）。每个 GPU 上的 GEMM 计算仍为 dense matmul。
  - Baseline 核心缺陷根因：(1) **粗粒度合并**——整组 expert 权重平均化，无法区分共享知识和专家特殊化参数，导致 -18.7% MMLU；(2) **任务依赖**——calibration data 改变显著影响精度（NAEE C4 校准在 GSM8K 仅 41.5%，换 MATH 校准升至 48.7%）；(3) **高压缩成本**——SVD 分解（D2 需 55min）、exhaustive search（NAEE 对 64-expert Deepseek-MoE 需 10^18 次 forward pass 不可行）；(4) **mask 存储开销**——binary mask 的 metadata 存储抵消压缩收益。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：PuzzleMoE 通过两个核心创新解决：(1) **Pairwise Dual-Mask Sparse Expert Merging**——元素级（entry-wise）而非整组 expert 的合并，利用 similarity mask 保留共享知识、saliency mask 保留专家特化参数；(2) **Bit-packed Encoding + Custom CUDA GEMV Kernel**——利用 Bfloat16 的 underutilized exponent bits 嵌入 mask/sign，消除 metadata 存储开销并实现高效推理。
  - 全栈执行例子（PuzzleMoE, Mixtral-8x7B, 50% sparsity, 1×A100-80GB）：
    - **算法层（解决"粗粒度合并"和"任务依赖"的缺陷）**：
      - **Fine-grained Entry-wise Merging**：对 W^i, W^j ∈ R^{d×h} 的每个元素位置 [p,q] 独立决定是平均化还是保留更重要的 expert 权重，而非对整个矩阵做统一操作。
      - **Similarity Mask M^sim**：Δ[p,q] = | |W_i[p,q]| - |W_j[p,q]| | / (|W_i[p,q]| + |W_j[p,q]|)，M^sim[p,q] = 1 若 Δ[p,q] ≤ 0.4。在数值上与 Wanda (Sun et al., 2024) 的 pruning 不同——M^sim 识别的是"两个 expert 都共识认为重要的位置"，而非"对单一 expert 重要的位置"。理论分析（Appendix B.2）证明元素级相似性源于 MoE weights 的 Gaussian 分布特性。
      - **Saliency Mask M^sal**：A_i = |W_i| ⊙ ||X_i||_2（activation-aware importance），M_i^sal[p,q] = 1 若 A_i[p,q] ≥ A_j[p,q]。仅需一次 forward pass 完成校准，且 C4 与 MATH 校准结果等价（GSM8K: 51.7 vs 51.7; Avg Acc: 72.6 vs 72.5），证明 task-agnostic。
      - **对比 baseline**：HC-SMoE 的 coarse-grained averaging 将 MMLU 从 67.9% 降至 49.0%（-18.9pts, 50% sparsity）；PuzzleMoE 降至 65.7%（-2.2pts）。在更难的 reasoning benchmarks（Qwen3-MoE），HC-SMoE 25% sparsity 时 Math-500 从 97.2 降至 24.6、AIME24 从 83.3 降至 0.0；PuzzleMoE 分别保持 96.2 和 71.1。
    - **系统框架层（解决"高压缩成本"的缺陷）**：
      - PuzzleMoE 的压缩流程为 linear time：前向 pass 计算 saliency（O(N_layers × d×h)）→ 元素级 mask 构造（O(d×h) per expert pair）→ merging（O(d×h) per pair）。无需 SVD、无需 exhaustive search。
      - Mixtral-8x7B 压缩仅 2 分钟 vs D2 的 55 分钟 vs NAEE 的不可行。Deepseek-MoE（64 experts）仅 10 分钟。
      - Pairwise grouping 采用随机策略（与 search-based 差异 <0.3% Avg Acc），进一步降低压缩复杂度。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层（解决"mask 存储开销"的缺陷）**：
      - **Bit-packed Encoding**：观察到 Bfloat16 exponent 集中在 [112, 128]（仅需 5 bits 编码 32 个值），通过减去 112 的 shift 操作释放出 3 bits。释放的 bits 用于存储 2 bits mask（M_i, M_j）+ 1 bit sign（S_i 或 S_j，取决于 expert position）。Packed 后 Perplexity 无变化（Mixtral-8x7B: before=4.37, after=4.37）。所有 data 仍在 Bfloat16 格式内，无需额外 metadata。
      - **Custom CUDA GEMV Kernel**：在 data-loading path 上融合 decoding——每个 weight 在从 global memory 加载后、FMA 计算前执行 Algorithm 1 的 bit-level decoding（3-4 条 bit ops）。Decoding 延迟远小于 global memory 读取延迟（~200 cycles vs ~1 cycle for bit ops），因此完全被访存隐藏。
      - **对比 baseline**：baseline 用 CSR 存 50% sparse matrix 无 net memory savings（Lasby et al., 2025）；PuzzleMoE 的 bit-packed 形式无额外 metadata——Mixtral-8x7B 从 2 GPUs → 1 GPU 部署。
    - **硬件架构层**：PuzzleMoE 的推理在 A100 GPU 上运行——压缩后的模型仅需单卡 A100-80GB（从原需 2 卡降至 1 卡），1.28× speedup。Qwen3-MoE 从 2×A100-40GB 降至 1×A100-40GB，1.19× speedup。与 quantization 结合（50% sparsity + 3-bit group quantization）：4.8× 总压缩比，Mixtral-8x7B 仅 -1.7% accuracy drop vs full model。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"粗粒度合并"**：元素级 dual-mask——M^sim 保留 expert 间共享的 consensus 参数，M^sal 保留各 expert 的独特重要参数。在 -50% experts 时 MMLU loss 仅 -2.2pts（HC-SMoE -18.9pts）。
    2. **针对"任务依赖"**：activation saliency 指标 A_i = |W_i| ⊙ ||X_i||_2 对 calibration data 不敏感——C4 和 MATH 校准结果几乎相同，简化部署无需领域特化。
    3. **针对"高压缩成本"**：single-pass forward（1 次）→ O(d×h) merging，无 SVD/exhaustive search。Mixtral-8x7B 压缩 2min vs D2 55min，Deepseek-MoE 10min。
    4. **针对"mask 存储开销"**：Bit-packed encoding 将 mask/sign 嵌入 Bfloat16 exponent bits——zero metadata overhead。Custom CUDA GEMV kernel 在 data-load path 上 decode，解码延迟被访存完全隐藏。

## Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

- baseline方法是什么？
  - Baseline 是 HuggingFace TGI（Text Generation Inference）——生产级 LLM 推理引擎，采用 iteration-level scheduling 和 continuous batching。Baseline 的核心问题：(a) priority-oblivious：无法区分 LS（latency-sensitive）和 BE（best-effort）请求，以 FCFS (first-come-first-served) 策略对待所有请求；(b) run-to-completion semantics：每个 decode batch 固定后必须跑完所有 N 层才返回 Scheduler，中途无法插入新请求；(c) Head-of-Line (HOL) Blocking：当 BE 请求先占据 batch 时，LS 请求必须等待当前 iteration 的完整 300-400ms decode iteration 完成后才能被调度；(d) 无 inner-layer state tracking：模型内部只看到 concatenated tensors，无法区分 individual sequence，导致 inner-layer preemption 需要昂贵的 tensor split-merge 操作且可能破坏数据流对齐。
  - 全栈执行例子（Baseline: HF TGI, Mixtral 8×7B, 4-bit 量化, batch_size=32, single A100 80GB）：
    - **算法层**：Mixtral 8×7B MoE 模型：每层 self-attention → router (gating network predicts top-k=2 experts) → selected expert FFN → combine。4-bit GPTQ quantization + FP16 compute precision。无算法层面的 priority 区分——所有 token 经过相同 pipeline。
    - **系统框架层**：HF TGI continuous batching：Job 到达 → Scheduler 接收 → 在 iteration boundary（所有 N 层执行完成后）决定 batch composition。如果 decode batch 有空位且新 job 到达，Scheduler 停止 decode、执行新 job 的 prefill、扩展 batch、然后继续 decode。Batch 内所有 job 的 tensors 被 pad+concatenate 为单一 tensor，model 内部无法区分 individual sequence。Scheduler 仅在 iteration boundary 获得控制权（即每 300-400ms 一次机会）。
    - **编译框架层**：论文未明确说明。HF TGI 使用 PyTorch 原生执行或 torch.compile，无专项编译修改。
    - **kernel调度层**：论文未明确说明。使用标准 PyTorch CUDA kernel（attention: scaled dot-product attention via cuBLAS; expert FFN: GEMM via cuBLAS）。
    - **硬件架构层**：单卡 Nvidia A100 80GB, PCIe 4.0, dual-socket Intel Xeon Gold 6336Y。GPU 执行全流程：prefill phase（compute-bound, 并行处理所有 input tokens）→ decode phase（memory-bound, 逐 token 迭代，每次 iteration 约 300-400ms for 32 layers × (attention + router + expert FFN)）。

  - Baseline 核心缺陷根因：iteration-level granularity (300-400ms per iteration) + FCFS policy 导致即使 LS 请求在 BE batch 执行的第 1 层就到达，也必须等待全部 32 层完成（300-400ms 级延迟）才能被调度。每个 iteration 内无抢占能力——Scheduler 在 iteration boundary 才取回控制权，且模型内部无 per-sequence state 支持任意点恢复。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QLLM——在 HF TGI 基础上实现 expert-level priority-aware preemptive scheduling。核心创新：(1) Per-Expert FIFO Queues 打破 layer-wise barrier，使每个 expert 可独立处理 token；(2) Scheduler 通过 closed-loop feedback 在任意 layer 的 router 后 preempt BE batch；(3) Sequence/Batch Facade Pattern 支持 zero-copy individual sequence 状态保存/恢复；(4) Unified Dynamic KV Cache 避免大 tensor split-merge。
  - 全栈执行例子（QLLM, Mixtral 8×7B, 4-bit 量化, batch_size=32, single A100 80GB）：
    - **算法层（解决"无法区分 priority"）**：在 MoE layer 内插入 per-expert FIFO queues——router 输出的 top-k=2 expert selection 将 sequence reference push 进对应 expert 队列。LS sequence 优先入队（通过 policy 在 preempt 时先处理 LS）。QLLM 区分 fully processed tokens（两个 expert 都输出完毕）和 partially processed tokens（只完成一个 expert），确保只有完整 token 的 hidden state 进入下一层。
    - **系统框架层（解决"FCFS + HOL blocking + run-to-completion"）**：① Scheduler 新增 Dispatcher（按 priority 分发到 4 个优先级队列：LS_PrefillQueue, LS_DecodeQueue, BE_PrefillQueue, BE_DecodeQueue）和 Batch Engine（Algorithm 1：LS_Decode > LS_Prefill + 填充 BE > BE_Decode > BE_Prefill）。② Closed-loop feedback controller：Inference Engine 在每个 attention 和 router stage 后回调 Scheduler。当 LS 到达时，Scheduler 立即发送 preempt 信号——无需等待 iteration boundary。③ Expert-level preemption：在 layer L 的 router 后，BE batch 暂停，其 partial state 通过 Sequence 对象的独立 tensor 原地保存。Engine 立即转向执行 LS prefill + decode，完成后动态将 LS 加入当前 batch，BE 从 preemption point 恢复（无 recomputation）。④ Facade Pattern Batch：对外呈现为单一 concat tensor，对内维护 per-sequence 独立 tensor——model 无感知 batch composition 变化，但系统可在任意时刻修改 individual sequence 状态。
    - **编译框架层**：论文未明确说明。QLLM 继承 HF TGI 的 PyTorch 执行路径，不涉及编译框架修改。
    - **kernel调度层**：论文未明确说明。preemption 的额外开销来自 per-sequence state tracking (routing_weights, hidden_states 的读写)，不涉及新 GPU kernel。
    - **硬件架构层**：同一 A100 80GB GPU。关键变化：per-expert queuing 将 "batch 必须同步跑完全部 layers" 解耦为 "各 expert 独立处理其队列中的 token"——在 A100 的 CUDA stream 层面，expert FFN kernel 仍是串行执行的（同一 GPU），但 LS token 优先被选中执行：当 Scheduler 在 layer 1 的 router 后 preempt BE、prefill LS、恢复 BE 时，GPU 上的实际计算序列变为 attention_l1 → router_l1 → (暂停 BE expert FFN) → LS prefill (all layers) → LS decode iteration → (恢复 BE expert FFN_l1) → attention_l2 → ...。preemption latency 远小于 300-400ms iteration 时间（因为只在单层内切换而非等待整轮 iteration）。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"priority-oblivious FCFS"**：将队列按 priority 分为 LS/BE × prefill/decode 四级，Batch Engine 严格优先 LS。LS 不因早到达的 BE 而被排队阻塞。
    2. **针对"iteration-level granularity (300-400ms)"**：Closed-loop feedback 在每层 attention/router 后给 Scheduler 控制权，preempt 可在任意 layer 触发——LS 到达后的等待时间从 300-400ms 降低到当前 layer 执行时间（~10ms for a single layer）。
    3. **针对"run-to-completion (无法中途插入)"**：Expert-level preemption + Sequence 对象独立状态保存——BE batch 在任意 layer 可暂停，LS 执行完毕后动态合并恢复，无需 recomputation。
    4. **针对"inner-layer state tracking 困难"**：Facade Pattern 的 Batch/Sequence 抽象——上层 model 看到的是 concat tensor（兼容现有代码），下层系统维护 per-sequence 独立 tensor，支持零拷贝 individual update。

## Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing

- baseline方法是什么？
  - Baseline 是现有的简单预测方法（如 moving average——直接使用历史数据均值作为预测结果），以及负载均衡的辅助 loss 方法（如 load balancing loss 加正则项、capacity factor 限制 experts token 处理上限、expert-based routing / hash-based routing）。Baseline 的核心问题：(a) 无法区分 MoE 训练过程中 expert 负载的 transient state（早期波动阶段）和 stable state（后期稳定阶段），导致在波动阶段预测无效，在稳定阶段预测精度不足；(b) 现有方法（FlexMoE/Prophet）使用启发式或 moving average 进行负载预测，在不平衡负载持续存在且状态转换时精度受限。
  - 全栈执行例子（Baseline: Moving Average 预测 + Load Balancing Loss, GPT-3 350M training, 4×A800）：
    - **算法层**：Gating network 对每个 token 执行 Softmax(W_gate·x) → Top-K routing 选择激活 experts。Model training 阶段加入 auxiliary load balancing loss（expert activation entropy 之和）迫使 gating 网络均衡分配 token。Moving average 预测直接取历史 t 轮的 expert 负载均值作为下一轮预测——在 transient state（前约 5,000 iterations）中因负载剧烈波动而预测不准；在 stable state 中因未建模时序相关性而精度受限。
    - **系统框架层**：论文未明确说明 training framework 细节。FlexMoE 使用启发式算法基于负载动态优化 expert placement；Prophet 利用 temporal locality 进行 layer-wise fine-grained 资源调度——但两者依赖的负载预测在 transient state 下不可靠。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明——预测算法在 CPU 上执行，不涉及 GPU kernel 修改。
    - **硬件架构层**：4×A800 GPU。在 MoE training 中 experts 可能因负载不均导致 GPU 资源浪费——hot expert 所在 GPU 计算拥塞而 cold expert 所在 GPU 资源闲置。Baseline 在全量 load balancing loss 下虽均衡了 expert 负载但可能影响模型精度（过度干预 gate network）。
  - Baseline 核心缺陷根因：现有方法无法准确判断 expert 负载何时从 transient 转换为 stable state，也无法在两种状态下提供高精度的负载预测，导致基于预测的 expert placement 和 resource allocation 决策不可靠。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：通过大量 MoE 模型训练实验系统性追踪和分析 expert 负载的时序变化，定义 transient state 和 stable state，并部署三种经典预测算法（LSTM、ARIMA、SW Avg）在不同状态下实现高精度 expert 负载预测。
  - 全栈执行例子（GPT-3 350M, 4×A800）：
    - **算法层（解决"无法区分负载状态"和"预测精度低"的缺陷）**：
      - 状态定义：通过计算各 expert 负载在不同滑动窗口（w=10,100,200）下的方差和极差，发现 expert 负载依次经历 transient state（早期波动、无规律变化）和 stable state（负载稳定、呈现 temporal locality——相邻迭代间负载变化微小）。以 GPT-3 125M 为例，约 5,000 iterations 后进入 stable state。
      - LSTM-based 预测：输入为所有 MoE 层所有 expert 的历史负载比例序列 [n_1^1,...,n_e^m]，输出为未来 k 步负载比例值；两次独立训练数据分别作为 train/test set。GPT-3 350M stable state 误差率约 <10%（每 1,000 iterations 预测粒度）。
      - ARIMA-based 预测：对每个 expert 负载时序做平稳性/季节性检验选取 ARIMA(p,d,q) 参数，实验中 ARIMA(5,1,5) 取得低于 LSTM 的误差率。GPT-3 350M stable state 误差率约 1.4%。
      - SW Avg-based 预测：算术平均历史多轮数据直接作为预测值，通过 k 轮滑动计算预测未来。该算法在三种方法中表现最佳：GPT-3 125M stable state 误差率约 0.25%，GPT-3 350M stable state 误差率约 1.3%（1,000-step）和 1.7%（2,000-step）。方法计算极其简单，硬件友好。
      - 对比 baseline：Baseline 的 moving average 不做状态区分且不建模时序依赖；论文方法通过 LSTM/ARIMA 捕捉时序模式提升精度，通过 SW Avg 在极低计算成本下获得最佳精度，并根据状态区分指导资源分配策略——stable state 下可基于预测精细分配，transient state 下需预留充足资源应对波动。
    - **系统框架层**：论文未明确说明具体框架修改。预测结果可对接 FlexMoE（启发式 expert placement 算法）或 Prophet（fine-grained layer-wise 资源调度），在 stable state 下提供准确的 per-expert 负载信息作为 placement/scheduling 决策依据。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明——预测算法为 CPU 端时序分析任务，不修改 GPU kernel。
    - **硬件架构层**：同一 A800/4090 GPU 平台。核心变化：通过高精度负载预测使基于负载的 expert placement（将 hot expert 部署到更多 GPU、cold expert 合并部署）成为可行——在 stable state 下约 1.3% 误差率意味着可精确预知 GPU 资源需求、提前规划 expert-to-GPU 映射，减少 resource underutilization 和 load imbalance 导致的训练效率损失。在 transient state 下由于波动不可预测，保留充足 GPU 资源以应对负载突发。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"无法区分负载状态"**：通过大规模实验统计分析定义 transient/stable state 的量化特征（方差和极差的时序趋势），并指出 shallow MoE layer 的波动幅度大于 deep layers（spatial dimension 观察）。为后续 resource allocation 提供分阶段策略基础。
    2. **针对"预测精度低"**：部署三种预测算法在两种状态下量化评估——ARIMA 优于 LSTM，SW Avg 在极简计算下反而最佳（因为 stable state 下负载具有强 temporal locality，简单平均即高精度），在 GPT-3 350M stable state 下仅 1.3% 误差率。
    3. **指导价值**：高精度负载预测结果为 downstream 的 expert placement 和 resource allocation 提供可靠输入——在 stable state 下可动态按 expert 负载分配 GPU 资源（hot expert 占更多资源、cold expert 占更少资源），最大化训练效率的同时最小化资源消耗。论文在此篇中仅提供预测能力，并声明后续 work 将设计具体的 expert placement 方案。

## Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference

- baseline方法是什么？
  - Baseline 是现有 CPU offloading 两类方案：(a) MoE-OnDemand（fetch-on-demand）——所有 expert 参数 offload 到 CPU，gate 选择激活 experts 后按需从 CPU 迁移到 GPU，但 expert selection 与 expert execution 串行执行，直接暴露 PCIe 延迟；(b) MoE-Prefetch（prefetch-all）——在当前 block 执行期间迁移下一个 block 的全部 experts 到 GPU，但需传输全部 expert 参数（如 128/256 个），PCIe 带宽成为瓶颈且 GPU 内存需同时容纳两个 block 的全部 experts。
  - 全栈执行例子（Baseline: MoE-OnDemand, Switch-Base 128 experts, A100 80GB, PCIe Gen4 32GB/s）：
    - **算法层**：传统 MoE gate function 在第 N 个 block 内选择同一 block 的激活 experts。gate(W_gate @ x) → softmax → TopK(k=1) → 选择 1 个 expert。gate 输出与 expert execution 存在数据依赖——必须先知道哪个 expert 被选中，才能执行该 expert 的 FFN。
    - **系统框架层**：FasterTransformer 上的 MoE-OnDemand 实现。non-MoE 参数在 GPU，全部 expert 参数在 CPU。每个 MoE block 执行流程：gate → cudaMemcpy(选中的 expert, CPU→GPU) → expert FFN。gate 计算 (~0.05ms) → PCIe 传输 1 个 expert (~85MB/32GB/s ≈ 2.7ms) → expert FFN (~2ms)。总延迟 ≈ 4.75ms/block，其中 PCIe 传输占 57%。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer CUDA kernel 执行 gate (Linear+Softmax+TopK) 和 expert FFN (2×GEMM+GELU)。cudaMemcpy 在 default stream 上同步执行，阻塞后续 kernel launch。gate 和 FFN 之间因数据依赖无法重叠。
    - **硬件架构层**：单 A100 80GB。PCIe Gen4 32GB/s。CPU 1.8TB DDR4。缺陷：gate→expert 串行依赖导致 PCIe 传输直接暴露在关键路径上；多 GPU expert parallelism 方案下 expert 稀疏激活导致 GPU 利用率低（Switch-Base 128 experts Top-1 仅激活 0.8% experts）。
  - Baseline 核心缺陷根因：传统 MoE block 中 expert selection (gate) 与 expert execution 的**数据依赖在同一 block 内**——gate 必须执行完才知道激活哪些 experts，然后才能执行 expert FFN。这使得无论采用何种 CPU offloading 策略（按需取或全量预取），都无法避免 PCIe 延迟对关键路径的影响。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 Pre-gated MoE，通过 algorithm-system co-design 解决：**(Algorithm)** Pre-gate function 将 expert selection 从"为当前 block 选择"改为"为下一个 block 选择"，消除同一 block 内的 gate→execution 数据依赖；**(System)** Preemptive expert migration 利用 pre-gate 提前知道下一个 block 的 experts，在当前 block 执行期间异步迁移仅激活的 experts。
  - 全栈执行例子（Pre-gated MoE, Switch-Base 128 experts, A100 80GB, PCIe Gen4 32GB/s）：
    - **算法层（解决"gate→execution 同 block 数据依赖"缺陷）**：
      - Pre-gate function 设计：第 N 个 MoE block 的 pre-gate 函数 (轻量 Linear layer: W_pre_gate @ x → softmax → TopK) 输出第 (N+1) 个 block 的激活 expert mask。第一个 block 使用双 gate（传统 gate + pre-gate），最后一个 block 无 pre-gate。
      - 训练：复用 pretrained SwitchTransformer 权重，仅在 fine-tuning 阶段训练 pre-gate function（2,048 steps, lr=0.0001, 与常规 MoE 相同配置）。不对 resource-intensive pretraining 做任何修改。
      - 准确率：Pre-gated MoE 在 Xsum/CB Web QA/SQuAD 三个下游任务上准确率与原始 SwitchTransformer 相当（Rouge-1 差异 < 0.1, ExactMatch 差异 < 1.6），部分配置甚至略优。
      - **对比 baseline**：baseline 的 gate 为"当前 block"选择 experts，导致 gate 输出与 expert execution 串行依赖；Pre-gated MoE 的 pre-gate 为"下一个 block"选择 experts，使 expert execution 可以立即开始（experts 已由上一个 block 的 pre-gate 选定），消除了同一 block 内的数据依赖。
    - **系统框架层（解决"PCIe 延迟暴露在关键路径"和"全量预取浪费带宽"缺陷）**：
      - 分层存储：non-MoE 参数常驻 GPU HBM，全部 expert 参数 offload 到 CPU DRAM。
      - Preemptive expert migration：Block (N-1) 的 pre-gate 输出 A_N → 在 Block (N-1) 的 expert execution 期间 → 异步 cudaMemcpy(A_N 的 expert weights, CPU→GPU)。到 Block N 开始执行时，A_N 已在 GPU memory 就绪。
      - 通信-计算重叠：Expert execution (compute-bound, ~2ms) || Expert migration of A_{N+1} (communication-bound, ~2ms via PCIe 32GB/s)。Pre-gate 本身是轻量 Linear（~0.05ms），几乎不占时间。
      - GPU 峰值内存：Peak_GPU_mem = Non_MoE_M + Act_Exp_N + Act_Exp_{N+1}，仅需容纳非 MoE 参数 + 两个连续 block 的激活 expert 参数。实际峰值仅占 GPU-only 的 23%。
      - **对比 baseline**：(a) vs MoE-OnDemand——Pre-gated MoE 消除了 expert migration 的串行暴露，迁移与计算重叠使延迟降低 1.7×；(b) vs MoE-Prefetch——Pre-gated MoE 仅传输激活 experts（~1-2 个），而非全部 experts（~128 个），传输量减少 ~100×，使 PCIe 带宽不再成为瓶颈。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer CUDA kernel 不变。区别在数据流：baseline 的 cudaMemcpy 在关键路径上与 gate 串行；Pre-gated MoE 的 cudaMemcpy 在独立 CUDA stream 上与 expert FFN kernel 并行。第一个 MoE block 是唯一例外——因无 previous pre-gate 为其选择 experts，仍需串行 gate→cudaMemcpy→FFN。但由于 LLM 通常有数十个 block，大部分 block 可受益于重叠。
    - **硬件架构层**：同一 A100 + PCIe Gen4 硬件。核心变化：PCIe 链路在 baseline 中处于"等待 gate 完成 → 传输 → FFN 等待传输完成"的串行模式；Pre-gated MoE 下 PCIe 持续在 expert execution 期间并行传输下一个 block 的 experts，链路利用率提升。结果：Pre-gated MoE 达到 GPU-only（oracular 上界）的 81% 吞吐，峰值 GPU 内存仅为其 23%。Switch-Large (26.4B, 128 experts) 在 GPU-only OOM 的情况下，Pre-gated MoE 仍能以 42 tokens/sec 运行。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"gate→execution 同 block 数据依赖"**：Pre-gate function 将 expert selection 提前到上一个 block，消除同一 block 内的串行依赖。这是 algorithm 层的根本创新——仅增加一个轻量 Linear layer 就改变了整个 MoE block 的执行语义。
    2. **针对"PCIe 延迟暴露在关键路径"**：Preemptive expert migration 利用"提前知道下一个 block experts"的能力，将 expert 迁移与当前 block computation 重叠，使 PCIe 传输时间完全隐藏在计算时间之后（除第一个 block）。
    3. **针对"全量预取浪费带宽和 GPU 内存"**：仅迁移激活 experts（而非全部），使传输量减少 ~100×（128 experts Top-1 场景），GPU 峰值内存接近 memory-optimal 的 MoE-OnDemand（仅多 0.2%），同时性能接近 GPU-only（达到 81%）。
    4. **通用性**：Pre-gate function 的训练仅需 fine-tuning（复用 pretrained weights），不修改 resource-intensive pretraining 阶段，准确率无损，可直接适用于任何 MoE-based LLM。

## PreMoE: Proactive Inference for Efficient Mixture-of-Experts

- baseline方法是什么？
  - Baseline 是现有 MoE 模型的静态全量部署方案和基于统计的 Expert 剪枝方法。两类典型 baseline：(1) 全量部署——所有 N_r 个 expert 常驻内存，仅随 token routing 激活 K 个（通常 K≪N_r，如 DeepSeek-R1 仅激活 8/256），导致大量 expert 参数占用内存却极少被使用；(2) 统计剪枝方法——基于 activation frequency（激活次数）、all-logits 平均值、或 activated-logits（仅对激活 expert 的 logit 求平均）来排名 expert 重要性并剪枝。SEER-MoE 使用 local/global 变体的统计指标；EASY-EP 使用 few-shot demonstrations 识别相关 expert。
  - 全栈执行例子（Baseline: Full DeepSeek-R1-671B, Ascend 910B2-64GB）：
    - **算法层**：MoE router 对输入 token x 计算 logits s(x) ∈ R^{256}，softmax 后 Top-8 选择激活 expert。全量 256 experts × 58 layers 的 FFN weights 必须全部常驻 NPU HBM。Frequency-based pruning: 在校准集上统计每个 expert 被 Top-K 选中的次数，按频次排序，保留 Top-M。缺陷——Frequency 将"频繁但弱"的 generalist expert（激活数千次但几乎从不成为 top-1）排入高位，同时丢弃"稀少但关键"的 specialist expert（激活少但每次激活都是决定性 top-1 选择）。
    - **系统框架层**：vLLM/SGLang 等标准 serving 框架加载完整模型 checkpoint (670.92B params, BF16 ~1.3TB)，所有 expert 常驻。每次 MoE layer forward: router → gather selected expert weights → all-to-all communication → expert FFN → reduce。无 expert 选择优化——所有 expert 均等占用内存和通信资源。
    - **编译框架层**：论文未明确说明。
    - **Kernel调度层**：MoE layer 的 all-to-all dispatch + expert GEMM kernel。每个 MoE layer 执行 256 个 expert 的 GEMM（仅 K=8 个被实际激活计算，其余 idle）。Ascend 910B2 NPU 上 latency 115.35ms/tok at 256 experts/layer。
    - **硬件架构层**：64×Ascend 910B2-64GB NPU。每个 NPU 64GB HBM 需容纳 expert shards。全量部署需 64 NPUs，参数 670.92B。缺陷：(1) 内存浪费——大量 expert 在特定领域极少被激活却占据 HBM；(2) 通信浪费——all-to-all 通信涉及所有 expert shard 所在 NPU，即使大部分不参与计算；(3) 静态部署无法利用领域特化——同一模型服务数学和代码两类负载时无法自适应调整 expert 集合。
  - Baseline 核心缺陷根因：传统 MoE 部署采用"反应式"范式——模型被完整加载后再由 router 动态选择 expert。这种范式无法利用 MoE 的核心优势（领域相关稀疏激活），因为 expert 效用评估缺乏高质量的预测信号。Frequency 等粗粒度统计混淆了"频繁但低效用"的 generalist 和"稀少但高价值"的 specialist，导致高稀疏度下性能崩溃。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 PreMoE，通过 Predicted Expert Utility (PEU) 实现"主动编译"范式——部署前基于校准集提取领域计算 pattern，提前确定最小高性能 expert 子集，仅编译加载所需 expert。核心设计：(1) TopK 过滤缩小候选池；(2) 自适应阈值过滤去除低置信度激活；(3) Logit 变换 f(s)=max(s, sigmoid(s)) 解决 0-vs-negative 问题。
  - 全栈执行例子（PreMoE, DeepSeek-R1-671B, Ascend 910B2-64GB）：
    - **算法层（解决"粗粒度统计无法区分 expert 角色"的缺陷）**：
      - PEU 计算 pipeline: 对校准集 X_T 中每个 token x，router 计算 s(x) ∈ R^{256} → TopK_8 过滤保留 top-8 logit expert → local softmax p_i(x) → 仅保留 p_i(x)≥r_l 的 expert → f(s) = max(s, sigmoid(s)) 变换 logit → 跨 token 平均得 PEU_i^T。
      - 自适应阈值 r_l = E_x[max_i p_i^l(x)] 逐层计算，自动适配每层 activation 分布——避免固定阈值在不同层/领域的非鲁棒性。
      - 结果：PEU 将"频繁 but 低效用"的 generalist (Freq rank top-63 but PEU rank 252-256) 排到尾部，将"稀少 but 关键"的 specialist (Freq rank tail but PEU rank 51-148) 提升到高位。
      - 对比 baseline：Frequency 在 DeepSeek-R1 50% sparsity 下造成 25.58 点平均 drop，PreMoE 反而提升 1.01 点。
    - **系统框架层（解决"静态全量部署浪费内存"的缺陷）**：
      - 主动编译——校准后仅将 PEU Top-M expert weights 加载到 NPU HBM。完整模型 (670.92B) 从未加载到内存。
      - 50% sparsity (128/256 experts/layer)：参数降至 343.96B，NPUs 从 64 降至 32（2× fewer），throughput 从 52.01 升至 64.02 tok/s（+23%）。
      - 75% sparsity (64/256 experts/layer)：参数降至 180.49B，NPUs 降至 16（4× fewer），throughput 升至 81.97 tok/s（+58%）。
    - **编译框架层**：论文未明确说明。
    - **Kernel调度层**：kernel 执行不变（同一 MoE GEMM + all-to-all）。差别：(a) 参与 all-to-all 的 expert 数量从 256 降至 128 或 64，all-to-all 通信量减半；(b) 仅需 dispatch 到的 NPU 数从 64 降至 32 或 16，减少通信跳数和拥塞；(c) 每 NPU 的 expert shard 大小减半，GEMM kernel 的 memory footprint 减半。
    - **硬件架构层**：同一 Ascend 910B2-64GB 集群。核心变化：baseline 下 64 NPU 承载 670.92B 参数，大量 expert 在其不活跃的领域中闲置；PreMoE 下按领域 pattern 仅部署最小 expert 集合，闲置 expert 不再占用 HBM。latency 从 115.35 降至 93.72 ms/tok (50% sparsity)。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"粗粒度统计混淆 expert 角色"**：PEU 通过高置信度过滤 + logit 变换从 router logits 提取决定性偏好信号，从而区分 generalist（频繁激活 but 很少 top-1）和 specialist（稀少激活 but 经常 top-1）。相比 Frequency 的 25.58 点平均 drop，PreMoE 在 50% sparsity 下反超全量模型。
    2. **针对"静态全量部署浪费资源"**：主动编译范式仅在部署时加载 PEU 选中的 expert，实现 50% sparsity 时 2× NPU 减少、23% throughput 提升、近无损准确率。
    3. **针对"缺乏领域自适应能力"**：PEU pattern 是领域特定的（Math/Science/Code pattern 在 Top-2 experts 仅 4-16% 重叠），可实现 single-domain specialist（极致 in-domain 效率）或 multi-domain generalist（平衡跨域能力）。

- baseline方法是什么？
  - Baseline 是 cross-layer expert prediction——使用前一层的 hidden state 或 gate 输出来预测当前层的 expert 选择。以 FATE (Fang et al., 2025) 为代表性 baseline：从前一层 activations 预测当前层 experts，prediction accuracy 78.79%（DeepSeek-V2-Lite）。其他 baseline 包括 DuoServe-MoE（layer-level predictor，54-67% top-2 accuracy）、SP-MoE（speculative decoding 场景，70% accuracy）、HOB-BIT（cache-based，55% hit rate）。Baseline 三个核心缺陷：(1) 跨层预测不准——前一层信息经过 attention 变换后与当前层 expert selection 的关联性减弱；(2) 第一层无法 prefetch——没有"前一层"，bootstrap problem 导致前几层 accuracy 显著降低（AdapMoE 专门为此设计 mitigation）；(3) 架构复杂——跨层 predictor 需要跨层通信、state buffering、inter-layer coordination，增加系统复杂度。
  - 全栈执行例子（Baseline: FATE cross-layer prediction, DeepSeek-V2-Lite on V100-32GB）：
    - **算法层**：FATE 从 layer l-1 的输出 activations 预测 layer l 的 expert selection——使用前一层 hidden state h_{l-1} 通过 predictor 网络输出 expert scores。预测器基于跨层信息外推（extrapolation），accuracy 贡献 78.8%（FATE 还使用 cache+confidence threshold 达到 97.2% combined hit rate）。
    - **系统框架层**：跨层 predictor 需要维护前一层输出的 buffer，在 layer l-1 完成后、layer l 开始前执行预测。predictor 与当前层 pipeline 无天然并行窗口——预测必须在拿到前一层结果后进行，无法与 attention 重叠。
    - **编译框架层**：论文未明确说明。
    - **Kernel调度层**：cross-layer prediction 时序: layer l-1 expert computation → collect activations → predictor inference → expert prefetch for layer l → layer l attention + expert routing。预测结果到实际使用之间存在 attention computation 的延迟窗口，但跨层信息本身陈旧（经过了一层 attention 变换）。
    - **硬件架构层**：V100-SXM2-32GB GPU。Expert loading from disk: 48.1ms (6 experts, 99MB)，from memory: 9.5ms。Baseline 缺陷：(a) 第一层无前一层信息，bootstrap 准确率极低；(b) 跨层信息衰减导致整体 accuracy 仅 78.8%；(c) cross-layer predictor 不能与 layer l 的 attention 重叠执行，增加 latency 或减少 prefetch 窗口。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 pre-attention same-layer expert prediction——在同一层内、attention 之前使用 pre-attention normalization 后的 hidden state 预测该层的 expert selection。核心 insight：(1) softmax 和 layer normalization 是 ranking-preserving 的，因此可以用简单的线性函数近似 expert selection 的 ranking；(2) pre-attention weights 比前一层输出包含更"新鲜"的信息——temporal proximity 更高，直接捕捉 token representation 到 expert routing 的映射。方法通过轻量 2-layer predictor + ranking-aware loss 实现。
  - 全栈执行例子（Pre-Attention Prediction, DeepSeek-V2-Lite on A100-80GB）：
    - **算法层（解决 cross-layer 不准的缺陷）**：
      - Pre-attention hidden state X 经过 layer l 的 RMSNorm 后 fork——一份送入 self-attention，一份送入 predictor。
      - Predictor: s = W_2 @ SiLU(W_1 @ X)，W_1 ∈ R^{2048×d}，W_2 ∈ R^{E×2048}（仅 2 层线性，~4M 参数/层，远小于 standalone network）。
      - Ranking-aware loss: L = L_WBCE + 0.3·L_ranking。L_WBCE 对 top-10 experts 赋权 3.0，top 11-30 赋权 1.5，其余 0.5——针对三阶段 affinity score 分布（top few high → middle flat → bottom sharp drop）设计。L_ranking 为 pairwise margin ranking loss 保证 top experts 相对顺序。
      - 结果：93.03% exact-match accuracy（DeepSeek-V2-Lite），比 FATE 提升 15 个百分点。
      - **对比 baseline**：same-layer 信息比 cross-layer 更新鲜；ranking-aware loss 比 FATE 的通用 predictor 更精确匹配 MoE routing 的 Top-K selection 本质。
    - **系统框架层（解决第一层 bootstrap 和跨层复杂度的缺陷）**：
      - 每层维护独立 predictor f_l——第一层也有自己的 predictor，使用 embedding 后的 pre-attention X 预测，无需前一层信息。
      - Predictor 与 self-attention 并行执行——pre-attention norm 后 CPU clone X → CPU predictor (0.15ms) 与 GPU self-attention (0.74-1.13ms) + post-attention norm (0.08-0.13ms) 并行。总可用 prefetch 窗口 0.82-1.26ms，足以从 memory prefetch 1-2 个 experts (0.7-1.6ms each)。
      - 无需跨层 state buffer、inter-layer communication——同层独立操作，系统复杂度显著降低。
      - 三种部署策略：cloud（over-provision, load 10 vs 6 experts, 98.65% hit rate）、standard（93.03% exact-match）、edge（top-1 only, 98.85% accuracy）。
      - **对比 baseline**：baseline 第一层无 predictor，FATE 第一二层 accuracy 显著低于其他层。论文方法所有层 accuracy 均匀（Table 4 直接展示第一层 accuracy，与其他层相似或更好）。
    - **编译框架层**：论文未明确说明。
    - **Kernel调度层（解决 predictor overhead 和 prefetch 时序的缺陷）**：
      - 预测 latency 0.15ms（CPU），<10% pre-MLP pipeline，可被 attention 完全覆盖。
      - 93.03% tokens 实现 zero expert loading latency——expert 在 attention 期间预取就绪。
      - Expected loading time: (1-0.9303)×9.5 = 0.66ms/token (V100)，vs FATE (1-0.7879)×9.5 = 2.01ms/token。1000-token 会话节省 569-1352ms。
      - Best-case pipeline (Fig.8b): attention 期间并行预取 experts → expert selection 直接命中 → 无额外 latency。
      - Worst-case pipeline (Fig.8c): 预测错误时，6.97% miss rate，紧急从 disk load (5.6-8.3ms/expert)，但与 expert computation (6.2-10.3ms) 部分重叠。
      - **对比 baseline**：baseline predictor latency 不可被隐藏（需等前一层完成后串行执行），论文方法 predictor 与 attention 天然并行，prefetch 窗口更大。
    - **硬件架构层**：
      - 适配 GPU 异构内存层次：expert 可在 GPU memory (4.0ms/6 experts on A100-80GB)、system memory (8.5-9.5ms)、或 disk (33.5-49.8ms) 之间分级存储。预测精度越高，越能用更快的 memory tier。
      - CPU predictor 推理：CPU 执行预测不占用 GPU SM，GPU 可专注 attention + expert computation。

- baseline方法是什么？
  - Baseline 是 Megatron-DeepSpeed 框架提供的全量 checkpointing 方法——每个 checkpoint 保存所有模型状态：包括全部 N 个 experts 的 weights 和 optimizer states（占总 checkpoint 体积约 86%）、非 expert 部分的 weights 和 optimizer states（约 13%），以及 epoch/iteration 数和 RNG state 等辅助状态（<1%）。对于 MoE 模型，expert 部分的 optimizer states 是最大的单一部分（以 GPT-350M-16E 为例，experts 占 checkpoint 总体积 74%）。在 ZeRO-2 DP + EP 的混合并行策略下，baseline 的 sharding 策略存在显著局限性：非 expert 状态仅由 EP-Group-0 的 Rank0 保存，expert 状态仅由 EP-Group-0 的 ranks 保存——未能利用全部并行 ranks 的带宽和存储能力。
  - 全栈执行例子（Baseline: Megatron-DeepSpeed full checkpointing, GPT-350M-16E, ZeRO-2 DP=16 + EP=16/8, A800×8/16）：
    - **算法层**：每次 checkpoint 触发时，全体模型状态的 tensor 从 GPU memory 读取——包括所有 MoE layer 的 expert FFN weight（P_e 个参数，每个 B_w bytes）和对应 optimizer states（每个 B_o bytes，通常 B_o = 2B_w 因 Adam 需 momentum + variance）。无任何选择性保存策略——所有 experts 同等对待。
    - **系统框架层**：Megatron-DeepSpeed 的 checkpointing 模块执行两阶段流程——Phase 1 (GPU-to-CPU snapshot)：所有 DP rank 同步将模型 state tensor 从 GPU memory 通过 PCIe 复制到 pinned CPU memory；Phase 2 (CPU-to-Storage persist)：序列化 CPU memory 中的 tensor 并通过网络写入分布式文件系统。Baseline 的同步 checkpointing 在这两阶段都会阻塞训练进程。异步 checkpointing 版本（Base-Async）可让 GPU-to-CPU snapshot 与下一迭代的 forward+backward 重叠，但 snapshot 必须在 weight update 前完成，否则产生 checkpoint stall。
    - **编译框架层**：论文未明确说明。
    - **Kernel调度层**：ZeRO-2 DP 已将 optimizer states 按 DP degree 分片分布在各 rank 上，但 baseline checkpointing 未充分利用此分片——仅 EP-Group-0 执行 checkpoint 写操作。bottleneck rank 的 GPU→CPU 复制时间决定整体 checkpoint 时长。在 Case1 (1 node/8 GPU) 中，baseline snapshot 时长（约 1.7s）超出 F&B 时间（约 1.3s），导致每次 checkpoint 都会触发 stall（blocking），直接延长训练时间。Case3 类似。
    - **硬件架构层**：A800 SXM4 80GB GPU，GPU-to-CPU PCIe 带宽约 1 GB/s。以 GPT-350M-16E 为例，单 rank 需传输约 350M params × (2 + 12) bytes ≈ 5GB 数据（含模型参数 + optimizer states），约需 5s。在 60-node×8-GPU 集群上，所有 ranks 从分布式文件系统写入 checkpoint，对存储造成显著 IO 压力。Baseline 核心缺陷：(a) checkpoint 数据量大——MoE 模型因数十个 expert FFN 导致 checkpoint 体积远超同计算量的 dense 模型；(b) sharding 效率低——未利用全部 EP groups 的带宽；(c) snapshot 与 F&B 重叠不足——MoE 的 F&B 时间不随 expert 数量成比例增加，但 checkpoint 数据量随 expert 数线性增长，snapshot 无法被 F&B 完全覆盖导致 stall。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 MoC-System（Mixture-of-Checkpoint System），通过 algorithm-system co-design 的 PEC（Partial Experts Checkpointing）机制在 checkpoint 时仅选择性保存 K_pec 个 expert，大幅降低 checkpoint 数据量；配合 Fully Sharded Checkpointing 充分利用全部 DP ranks 和 EP groups 的带宽；以及 Two-Level Checkpointing Management（snapshot-PEC + persist-PEC + triple buffering）实现两级异步管理。每个设计直接对应解决 baseline 的特定缺陷。
  - 全栈执行例子（MoC-System, GPT-350M-16E, ZeRO-2 DP=16 + EP=16, K_snapshot=4, K_persist=1, A800×16）：
    - **算法层（PEC 机制——解决 checkpoint 数据量过大的缺陷）**：
      - PEC 在每次 checkpoint 时仅保存 K_pec 个 expert 的 weights + optimizer states（而非全部 N 个），非 expert 部分完整保存。Checkpoint size 从 C_full ≈ (P_ne + P_e) · (B_w + B_o) 降至 C_pec ≈ (P_ne + K_pec/N · P_e) · (B_w + B_o)。以 GPT-350M-16E 为例，K_pec=1 时 checkpoint 总量降至 baseline 的 42.3%（即减少 57.7%），expert optimizer states 部分从 74% 降至约 5%（1/16）。
      - 量化精度的 PLT 指标：PLT = (1/N_moe) Σ_i Σ_j L_{i,j} / (T_i · TopK_i)。实验证明 PLT < 3.75% 时 validation loss 与非故障 case 可比（delta < 0.005）。
      - Sequential Selection：交错轮转选择策略——checkpoint_iteration c 时，MoE layer l 保存 expert[l+c : l+c+K_pec] mod N，保证各 EP rank 的 workload 均衡且 PLT 可控。
      - Dynamic-K：当累积故障导致 PLT 逼近 3.75% 阈值时，K_pec 自动翻倍（如 1→2→4→...→N），将额外故障的精度风险控制在阈值内。
      - **对比 baseline**：baseline 保存全部 N 个 experts（K_pec=N），PEC 仅保存 1~4 个——在 K_pec=1 时 checkpoint 体积缩减至 baseline 的 42%。
    - **系统框架层（Fully Sharded Checkpointing——解决 sharding 效率低的缺陷）**：
      - Expert Part Equal Sharding：将每个 expert 按参数切分，由不同 EP group 的对应 rank 分担保存（而非仅 EP-Group-0）。以 expert 为最小切分单位，EP-Group-0 Rank0 保存 Expert0 的前半，EP-Group-1 Rank2 保存 Expert0 的后半。
      - Non-Expert Part Equal Sharding：以 layer（如 Attention、FFN）为最小单位，在所有 DP ranks 之间均匀分配非 expert 部分的保存任务。
      - Adaptive Sharding for PEC：当 (K_pec · N_moe) mod D_ep ≠ 0 导致部分 rank 保存更多 experts 时，用贪心算法将非 expert shards 优先分配给负载最轻的 rank。例如 K_pec=1, N_moe=12, D_ep=8 时，"Rank0" 需保存 2 个 experts 而其他 rank 仅保存 1 个——adaptive sharding 将更小的非 expert layer 分给 Rank0。
      - 理想负载公式：C_rank ≈ (P_ne + P_e)·B_o/D_ep + P_ne·B_w/D_dp + P_e·B_w/D_ep。
      - **对比 baseline**：baseline 仅用 EP-Group-0 保存，bottleneck rank workload 高且不均衡；fully sharded 将 bottleneck workload 减少 12%-29%（full saving）和 22%-29%（PEC saving），adaptive sharding 额外减少 3.7%-6.1%。
    - **编译框架层**：论文未明确说明。
    - **Kernel调度层（Two-Level Checkpointing Management——解决 snapshot 无法被 F&B 完全重叠的缺陷）**：
      - Snapshot-PEC 配置 K_snapshot≥K_persist（如 K_snapshot=4, K_persist=1）——GPU→CPU 时传输更多 experts（利用 PCIe 高带宽），CPU→Storage 时仅持久化最少 experts（利用持久存储的高可靠性）。
      - Triple Buffering：三个异步 buffer（snapshot/persist/recovery）状态机——snapshot buffer 完成 GPU→CPU 后自动转 persist buffer → CPU→Storage → 转 recovery buffer。第三 buffer 保证 snapshot 和 persist 可并行，且总有恢复用 buffer 可用。
      - 异步线程：每个训练进程内的独立线程触发 snapshot，与下一迭代的 F&B 完全重叠。仅当 snapshot 未在 F&B 完成前结束才导致 stall。
      - Two-Level Recovery：故障后，未故障节点直接从 CPU memory snapshot 恢复 K_snapshot 个 experts（比持久存储中的 K_persist 个更近期），有效降低 PLT。
      - **对比 baseline**：baseline 的 O_save 在 Case1/Case3 中无法被 F&B 完全覆盖（snapshot 时长 > F&B 时长）→ checkpoint stall；MoC-Async 通过 K_snapshot=4 将 snapshot 数据量减至原来的 4/16 = 25%，snapshot 时长缩短使完全重叠成为可能。实际结果：O_save 减少 98.2%-98.9%（Case1-3），每迭代加速 3.25×-5.12×（vs baseline blocking），I_ckpt 减半（Case2 中从 2.3 降至 1.2）。
    - **硬件架构层**：利用 GPU→CPU PCIe 的高带宽（快于网络到存储）做 snapshot 级保留更多 experts，利用 CPU memory 作为 intermediate buffer 减少 PLT。Scaling 模拟（ASTRA-SIM）显示：在 ≤1024 GPU 时 MoC-Async 优于 Base-Async（因 snapshot 更长无法重叠）；H100 场景下即使 1024 GPU 也无法重叠 Base-Async 的 snapshot；序列长度和模型大小实验证明 MoC-Async 在所有场景下均保持效率优势。
    - **算法层额外贡献（Fine-tuning 和下游任务反直觉收益）**：PEC 恢复后的模型在下游 8 个任务上平均 accuracy 提升 0.62%-1.08%（vs baseline full checkpointing），其中 BoolQ 提升高达 6.97%。假设 state loss 可能作为 dropout 变体防止过拟合。Fine-tuning 实验中，冻结全部 expert 参数仍可达 full fine-tuning 的 98.8% accuracy（61.16%→63.32% vs 64.09%），验证了 expert 参数对更新次数不敏感的观察——这是 PEC 可行的算法基础。

## ProMoE: Fast MoE-based LLM Serving using Proactive Caching

- baseline方法是什么？
  Baseline 方法是 **reactive caching with LRU/static cache + expert offloading**，即通过 LRU 或 static policy 将频繁访问的 experts 缓存于 GPU memory，未命中时从 CPU memory 被动按需加载（reactive cache miss）。Baseline 的两种变体：(a) Transformers Offloading (TO)——仅 expert 参数 offload 到 CPU，inference 时按需 cudaMemcpy 加载；(b) Llama.cpp Offloading (LO)——同时 offload expert 参数和计算到 CPU。
  全栈执行例子（Baseline: LRU cache + transformers, DS-1 FP16, RTX 4090, 50% cache rate, single token decode）：
  ```
  # 算法层：传统 MoE decoder layer
  token → embedding → for layer 1..28:
      # 系统框架层：HuggingFace transformers
      self_attention(x) → RMSNorm → gate: softmax(W_gate@x) → TopK=6 experts
      # Kernel调度层：cuBLAS GEMM for each expert FFN
      for each selected expert:
          if expert NOT in GPU cache:        # reactive cache miss!
              cudaMemcpy(CPU→GPU, expert_weight)  # BLOCKS critical path
          FFN_expert(x)  # SwiGLU: gate_proj→SiLU→×up_proj→down_proj
      weighted_sum(outputs)
  # 硬件架构层：RTX 4090 + PCIe 4.0 32GB/s + Intel i9-14900K
  ```
  核心缺陷：(1) **Reactive cache miss on critical path**——cache miss 时 expert 加载与 GPU 计算串行，DS-1 50% cache 时 decode 阶段 60.4% 时间用于等待 expert 加载，prefill 阶段达 82.7%；llama.cpp 更严重（prefill 94.2%, decode 79.0%）因推理速度更快使等待时间占比更高；(2) **Modern decoder-only MoE 的 uniform access pattern**——现代 MoE（DS/QW/Mixt）通过 Device-Limited Routing 和 Expert-Level Balance Loss 训练避免 routing collapse，导致 expert 访问分布均匀（low skewness），LRU cache hit rate 受限（不同于早期 encoder-decoder MoE 如 Switch Transformer 的 power-law 分布）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 ProMoE，通过 **proactive caching** 将 expert 加载从被动反应式（reactive）变为主动预测式（proactive），核心设计：(1) Learned Predictor——二层 MLP 学习 hidden state → expert selection 映射，accuracy 84.7%（vs token-based 58.3%, skip-based 66.9%）；(2) Stride Prefetching——预测与 prefetch pipeline 并行，将预测延迟隐藏，最大化 CPU-GPU 带宽利用率；(3) Chunked Prefetching + Early Preemption + Reordered Inference——三级协调机制消除被动 cache miss，最大化 prefetch 与 inference 重叠。
  全栈执行例子（ProMoE + transformers, DS-1 FP16, RTX 4090, 50% cache rate, single token decode）：
  ```
  # 算法层：ProMoE 增强的 MoE decoder layer
  token → embedding → for layer 1..28:
      # === 以下 3 步与 GPU self_attention 并行 ===
      # 算法层：Learned Predictor (CPU, ~200μs)
      h_prev_cpu = clone_to_cpu(x)              # 前一层 hidden state
      pred_logits = MLP_predictor[l](h_prev_cpu) # 2-layer MLP, ~2M params
      pred_experts = TopK(softmax(pred_logits), k=6)
      
      # 系统框架层：Prefetcher PushPredictedExperts (LOW priority)
      for e in pred_experts:
          if e not in GPU cache:
              queue.push(Task(layer=l, expert=e, chunk=0..2, pri=LOW))
      
      # 系统框架层：GPU self_attention
      x = self_attention(x)
      
      # Kernel调度层：gate function → hook
      gate_logits = W_gate @ x
      precise_experts = TopK(softmax(gate_logits), k=6)
      
      # 系统框架层：Early Preemption + Reordered Inference
      queue.remove_low_pri_tasks(layer=l)       # clear LOW tasks for this layer
      reordered = sort_by_cache_status(precise_experts) # cached → prefetching → none
      for e in reordered:
          if e not fully prefetched:
              queue.push(Task(layer=l, expert=e, chunk=0..2, pri=HIGH))
      
      # 硬件架构层：GPU computation ←→ PCIe prefetching pipeline
      for e in reordered:
          wait_until_chunks_ready(e)            # wait for prefetch completion
          output += gate_weight[e] * FFN_e(x)   # SwiGLU FFN on GPU
      # Prefetcher worker thread (CPU) concurrently:
      #   while True: task = queue.pop() → cudaMemcpyAsync(CPU→GPU, chunk)
  ```
  解决 Baseline 缺陷的方式：
  1. **针对"reactive cache miss on critical path"**：Proactive caching 通过 predictor 提前预测 + prefetcher 异步传输，将 expert 加载从关键路径移除。Chunked prefetching（3 chunks per expert）使高优先级任务等待延迟≤1 chunk。Early preemption 将 cache miss 检测提前到 gate 完成时刻（而非 expert 访问时刻）。Reordered inference 让已缓存 experts 先执行，同时异步 prefetch 缺失 experts。最终将关键路径加载时间从 69.68% 降至 30.96%（QW-2, 50% cache）。
  2. **针对"uniform access pattern 限制 cache hit rate"**：ProMoE 不依赖 expert access skewness——predictor 直接从 hidden state 预测 expert 选择，而非从历史访问频率推测。learned predictor 在 uniform access pattern 下仍维持 84.7% accuracy。stride prefetching 确保即使预测有 15.3% 误差，FetchRate 仍高（因预测与传输 pipeline 并行）。
  3. **整体效果**：vs offloading baselines: prefill 平均 2.20× (up to 3.21×), decode 平均 2.07× (up to 5.02×)。vs hand-crafted caching (LRU/static): prefill 1.78×, decode 1.34×。

## PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

- baseline方法是什么？
  - Baseline 是标准 Expert Parallelism (EP) 下的 static sharded placement，即每个 expert 被唯一地分配到 EP group 中的一个 rank，token dispatch 严格按 router 输出将 token 发往对应 expert 所在 rank。
  - 全栈执行例子（Baseline: SGLang + static EP, GPT-OSS-120B, Top-4/128 experts, ep=8, H800×8）：(a) 模型推理算法层——MoE layer 执行 router forward → Top-K selection → token-to-expert dispatch → Grouped GEMM per expert → All-to-All combine。Router 按输入语义独立决策，无全局负载感知。(b) 系统框架层——SGLang continuous batching 管理请求队列，chunked prefill 处理 prompt tokens，CUDA Graph 加速 decoding。静态 EP placement 在模型加载时确定，运行时不变。(c) 编译框架层——论文未明确说明。(d) Kernel调度层——DeepEP All-to-All dispatch/combine 基于 NVLink/NVSwitch 通信，Grouped GEMM 按 per-expert token count 调用 cuBLAS。Hot expert rank 因 token 集中导致 GEMM 执行时间长（计算 skew），同时 All-to-All 收发量最大（网络 skew），形成"double penalty"——同一 rank 依次被网络 ingress、计算、网络 egress 拖慢。(e) 硬件架构层——8×H800 NVSwitch 全互联，计算瓶颈 rank 的 SM 利用率高但其他 rank 在 All-to-All barrier 处空闲等待，IR 可达 2.6（prefill）或 1.43-2.28（decoding），约 50% 全局算力在同步点闲置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 PROBE，将负载均衡范式从被动调整（reactive adjustment）转变为主动准备（proactive preparation），通过 Continuous Lookahead Pipelining 实现预测→规划→预取的零开销流水线。
  - 全栈执行例子（PROBE: SGLang + DeepEP + NVSHMEM, GPT-OSS-120B, Top-4/128 experts, ep=8, H800×8）：(a) 模型推理算法层——Gate-Initialized Lookahead Predictor 以冻结的目标层 router + 可训练残差 MLP 预测下一层 expert 激活分布 n̂（≈90% Top-K accuracy, Top-Half-K Hit-Rate 接近 100%），但不替代实际 router——实际 dispatch 仍用 ground-truth router，保证语义等价。对比 baseline 的 router 无预测能力、只能等当前层执行完才知道下一层分布。解决的是：baseline 无法预知 expert 热点，必须等 router 执行后才能反应。(b) 系统框架层——双轨架构：Deterministic Track 执行标准 MoE 算子序列，Auxiliary Track 异步执行 Predict/Plan/Prefetch。Phase-Locked Co-Scheduling 将 Predict（MLP+All-Gather）映射到 All-to-All Dispatch 阶段（两者均网络带宽密集但互补资源），Plan（单SM CUDA kernel ≤16 iter）映射到 MoE Compute 阶段，Prefetch 通过 split-phase transmission——MoE Compute 期间启动 P2P expert 传输→All-to-All Combine 前暂停释放带宽→Combine 后恢复直到下一层 Attention 完成。对比 baseline SGLang 的纯串行执行。解决的是：baseline 的负载均衡决策（如 EPLB）暴露在关键路径上导致延迟惩罚；PROBE 将所有控制开销完全隐藏在计算/通信窗口内。(c) 编译框架层——论文未明确说明。(d) Kernel调度层——单SM CUDA Planning Solver 在 1 个 SM 上串行执行贪心迭代（kmax=16），以 max_r(T_comp^r + T_comm^r) 为目标函数最小化 bottleneck rank latency，受 T_trans^r ≤ T_window^r 约束。自定义 Triton kernel 实现受控 SM occupancy 的 P2P expert weight put。NVSHMEM symmetric memory 作为 replicated-expert buffer（双缓冲 6 slots）。IR 从 2.13 降至 1.09，Max/Avg 计算延迟比从 2.27 降至 1.18。对比 baseline static EP 无 kernel 级负载均衡调度。解决的是：baseline 的计算 skew + 网络 skew 双重惩罚——通过动态 expert 复制将 hotspot 负载分散到低负载 rank，同时 split-phase 传输避免 P2P 与 All-to-All 竞争带宽。(e) 硬件架构层——8×H800 NVSwitch 900 GB/s，利用 NVSwitch 高带宽使 3-expert P2P 传输能在 MoE Compute + Attention 窗口内完成。prefill 加速最高 1.32×，decoding 吞吐提升最高 1.26×。对比 baseline 约 50% 算力在同步点闲置。解决的是：baseline 的硬件利用率低——通过消除 straggler 使各 rank 负载均衡（IR≈1.09），释放先前闲置的算力。

## Parameters vs FLOPs Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models

- baseline方法是什么？
  - Baseline 是传统的 Dense Transformer Scaling Laws（Kaplan et al. 2020, Hoffmann et al. 2022），以及将 MoE 中除稀疏度外的其他变量（如 expert 数量、granularity）纳入分析的 MoE Scaling Laws（Clark et al. 2022, Ludziejewski et al. 2024）。这些 baseline 的核心假设是：(a) 模型容量（capacity）主要由总参数量 N 定义；(b) 在 Dense 模型中 N 和 FLOPs per example 线性耦合（FLOPs ≈ 6N per token），因此 N 可作为计算成本的代理变量；(c) MoE Scaling Laws 通常固定稀疏度配置（如固定 K 个 active experts）而仅变化其他变量。
  - 全栈执行例子（Baseline: Hoffmann et al. 2022 Chinchilla Scaling Law + fixed-sparsity MoE）：
    - 算法层：Hoffmann 的 L(N,D) = a/N^α + b/D^β + e 仅含 N 和 D 两个变量，按 C = 6ND 约束求解最优 N*(C) ∝ C^α。在 MoE 中扩展时（Clark et al. 2022），将 N 替换为总参数量并在固定 expert 配置（如 fixed granularity G，fixed E_active/E_total 比）下拟合。稀疏度 S 作为隐含固定变量存在，不被显式建模为可控维度——无法回答"给定 C 和 N，最优 S 是多少"。论文未明确说明系统框架、编译框架、kernel调度、硬件架构层——该工作纯算法层 Scaling Law 分析。
    - 关键缺陷：baseline 无法量化 FLOPs per example 与总参数量之间的最优权衡。在 MoE 中，S 控制活跃参数量 N_a = N·(1-S)，进而控制 FLOPs per example。若 S 被固定，则无法知道在给定训练计算预算下，应该通过增加总参数（提高 S）还是增加活跃参数（降低 S）来提升性能。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文将稀疏度 S 作为独立的第三维度纳入 Scaling Law 分析框架，系统研究 N（总参数）与 FLOPs per example（通过 S 控制）之间的最优权衡。
  - 全栈执行例子（论文方法: S-aware IsoFLOP surfaces + parametric scaling law）：
    - 算法层：论文将问题拆解为两个子问题并统一求解。(1) 固定 S 求最优 N：N* = argmin_N L(N; C, S)，通过 IsoFLOP 曲面沿 N 轴切片得到——发现固定 C 下 N* 随 S 增加而增加，而 N*_a 随 S 增加而减少，即稀疏模型更"大"但推理更"便宜"。(2) 固定 N 求最优 S：S* = argmin_S L(S; C, N)，通过 IsoFLOP 曲面沿 S 轴切片得到——发现 L(S; N, C) 呈抛物线形状，存在最优 S*，且 S* 随 N 增大而增大直至趋近 1，随 C 增大而减小。最终提出包含 S 的参量式 L(N,D,S) = a/N^α + b/D^β + c/(1-S)^λ + d/((1-S)^δ N^γ) + e，其中 (1-S) 近似活跃参数占比，乘法交互项 d/((1-S)^δ N^γ) 捕捉 N 和 S 的耦合效应。拟合结果显示 λ = -0.1666（负值）且 δ ≈ γ ≈ 0.16，验证稀疏度提高确实降低 loss。(2) 下游分析：发现多数任务上 pretraining loss 是 downstream performance 的良好预测器（与 S 无关），但在阅读理解类任务上，相同 perplexity 的稀疏模型比稠密模型表现更差——揭示了 FLOPs per example 在推理阶段的重要性。进一步通过 length-controlled CoT prompting 实验证明 MoE 比同等活跃参数的 Dense 模型从额外推理计算中获益更多。论文未明确说明系统框架、编译框架、kernel调度、硬件架构层——该工作纯算法层 Scaling Law 分析。
    - 解决 baseline 缺陷的方式：(a) baseline 将 N 作为唯一容量维度 → 论文将容量分解为 N（总参数/知识存储）和 FLOPs per example（活跃参数/计算深度）两个独立维度，通过 S 作为控制旋钮；(b) baseline 无法预测最优 S → 论文通过二次 IsoFLOP 曲面拟合发现 L(S; N, C) 的抛物线性，给出给定 N 和 C 下的 S* 解析趋势；(c) baseline 忽略下游任务中推理计算的角色 → 论文区分了"pretraining-efficient sparsity"（S→1 最优）与"inference-beneficial compute"（某些任务需更低 S/更高活跃参数），为推理时动态分配计算提供理论依据。

- baseline方法是什么？
  - Baseline 是每个 client 各自部署独立的 dedicated MoE 实例（dedicated MoE instances），无专家共享。
  - 全栈执行例子（Baseline: dedicated MoE instances, 2 clients × Mixtral-4x7B, TopK=2, 8×A100 40GB）：
    - **算法层**：每个 client 的 MoE 模型独立运行。每层 MoE layer 中，gating network 对输入 hidden states 执行 Softmax(LinearGate(X)) → TopK(k=2) 选择 2 个 experts。client 1 的请求由 client 1 的 experts A,B,C,D 服务，client 2 的请求由 client 2 的 experts A,B,C,E 服务。即使 experts A,B,C 在两个 client 中完全相同（相同权重），也独立加载两份到 GPU 显存。
    - **系统框架层**：vLLM 为每个 model instance 预分配全部 expert 参数的 GPU 显存。两个 model instance 各占用一份完整模型内存，相同 experts 不共享。每个 client 的请求独立进入 gating network，路由到各自的 expert 计算。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：vLLM 使用 Triton kernel 执行 expert FFN 计算。每个 expert 的 nn.Parameter 独立，即使权重相同也各自独立执行 kernel。batch 中 client 1 的 token 和 client 2 的 token 被不同 expert 对象分别处理，即使它们路由到本质上相同的专家。由于单个 client 的请求量有限，每个 expert 的 batch dimension 较小，GPU 计算资源利用率低（高 memory-to-compute ratio 下 SM 占用率不足）。
    - **硬件架构层**：8×NVIDIA A100 40GB。每个 Mixtral-4x7B 的 expert 约 7GB，2 个 model instance 需 ~112GB 纯参数显存。若 2 个 client 还使用了其他不同专家变体，总内存需求更高。由于 MoE 的稀疏执行特性——每次仅激活 TopK 个 experts——GPU 显存在模型参数上饱和，但其 SM 计算能力未充分利用（GPU underutilization）。
  - Baseline 缺陷根因（两个核心问题）：(1) **专家重复导致的显存浪费**：不同 client 部署的 MoE 变体中常包含完全相同的 experts——例如从 MergeKit 等工具组合 off-the-shelf experts 而来——但每个 client 需要独立 instance 加载全部参数，相同 experts 在不同显存空间各自占据一份（Mixtral-8x7B 每 expert 14GB）。在 multi-tenant 环境中，显存很快成为模型数量的瓶颈。(2) **稀疏执行导致的 GPU 计算利用率低**：MoE 每次请求仅激活 TopK 个 experts，单个 client 的请求量不足以在 per-expert 粒度形成大批次。高 memory-to-compute ratio 导致 GPU 在显存用满前就因请求不足而计算资源闲置。多 client 环境下问题加剧——每个 model instance 各自请求量少，无法形成有效的大批量计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：MoEsaic 通过专家去重（deduplication）、合并表示（merged representation）和 fused gate，在多 client 间共享相同专家，减少显存占用并提升批处理效率。
  - 全栈执行例子（MoEsaic, 2 clients, Mixtral-4x7B, TopK=2, 2 shared experts, 8×A100 40GB）：
    - **算法层**：
      - Expert Deduplication：模型加载时对每个 expert 的每个 tensor 计算 128-bit hash digest（如 SHA-512/128 或类似），存入 in-memory dictionary。后续加载的新 expert 计算 hash 后查 dictionary：命中 → 引用已有 tensor（共享显存）；未命中 → 分配新 GPU 显存。不修改任何 expert 权重，不改变模型精度或行为。
      - Merged Expert Representation：初始化后，将去重共享相同 tensor 的 expert 合并为单一 nn.Parameter。每个 client 的 gate 维护一个 expert ID → merged expert ID 的映射表，路由请求时自动指向合并后的表示。
      - Fused Gate：将多个 model instance 的 gating network 合并为单一 fused gate kernel。单次 CUDA kernel 调用完成所有 client 的 Softmax(LinearGate(X)) → TopK 路由，替代 separate gate 逐 model 串行调用。fused gate 维护 per-model gate mapping，输出被正确解析为各 client 对应的 merged expert ID。
      - Lazy Memory Allocation：初始化时用 tiny pseudo experts 占位（几乎零显存），加载参数时才逐步扩容并填充。去重后的 expert 仅保留一份，峰值内存 = 去重后模型大小 + 当前正在加载（尚未去重）的一个 expert。
    - **系统框架层**（vLLM 修改）：
      - Expert 加载：vLLM 原本在 model init 时预分配所有 expert 内存 → MoEsaic 改为 lazy allocation（tiny pseudo experts 初始化，resize 在加载时）。
      - Expert 表示：vLLM 原本 per-layer 所有 expert co-located 在单个 tensor → MoEsaic 拆分为独立 nn.Parameter per expert，支持张量级别共享。
      - Tensor-Parallel Support：vLLM 原本不支持向已部署模型动态添加 TP expert → MoEsaic 新增 Ray workers，每个 worker 加载指定 GPU 的 expert shard，新 expert 继承初始模型的 sharding 方式。
      - Non-disruptive Add/Remove：MoEsaic 支持在无活跃推理时动态添加/移除 model instance，通过独立 expert 表示 + hash dictionary 实现增量去重，无需系统重启。
      - LoRA-like Interface：client 通过类似 LoRA adapter 的接口向其 base MoE 添加新 experts 和 gates。
    - **编译框架层**：论文未明确说明。vLLM 的 Triton kernel 编译不受 MoEsaic 修改影响。
    - **kernel调度层**：Triton kernel 的 expert FFN 计算逻辑不变。关键差异在数据流：(a) 去重后相同 expert 使用单一 nn.Parameter，来自不同 client 的 token 被 Triton kernel 在同一 batch 中处理——client 1 的 8 tokens + client 2 的 6 tokens = 14 tokens batch，而非 baseline 的 8 tokens 和 6 tokens 两个独立 batch。较大的 batch 更充分利用 GPU 并行能力。(b) Fused gate 替代 separate gate：4 model instances 下 separate gate 需 4 次 CUDA kernel 调用，fused gate 仅需 1 次。对小型模型（如 Mixtral-4x1B，expert 计算时间短）节省尤为显著——separate gate 路由延迟每模型增加 8%，fused gate 降至 4%。
    - **硬件架构层**：同一 8×NVIDIA A100 40GB。核心变化：Baseline 中 2 个 Mixtral-4x7B model instances 占用 ~112GB 参数显存（各 ~56GB）；MoEsaic 以 2 shared experts 去重后节省 ~14GB×2=28GB，仅占 ~84GB。扩展到 Mixtral-8x7B (每 expert 14GB)，7 shared experts + 1 unique → 14 model instances 仅需 ~294GB（baseline 需 ~224GB 仅支持 2 instances），可服务 7× 更多变体。Batching 效果：4 instances Mixtral-3x1B 全共享时 per-expert batch size 从 ~10 增至 ~42（4×），NVIDIA Nsight 测量 SM 占用率随共享比例提升而下降（更高效地利用计算资源）。
  - 解决 Baseline 缺陷的方式：
    1. **针对"专家重复导致显存浪费"**：MoEsaic 用 hash-based tensor-level deduplication 检测并共享跨 model instance 的相同专家。Lazy memory allocation 确保仅去重后保留一份显存副本，峰值内存不超过去重后模型 + 当前加载 expert。Mixtral-8x7B 可服务 7× 更多变体，将 multi-tenant MoE 部署从"显存约束"的问题转变为"GPU 数量能力内可扩展"。
    2. **针对"稀疏执行导致计算利用率低"**：MoEsaic 用 merged expert representation 将来自不同 client 的请求在共享专家上自动批处理——多个 client 的少量请求汇聚成有效的计算批量。同时 fused gate 将 multi-model gating 合并为单次 kernel 调用，避免逐 model 串行 CUDA kernel 调用累积的路由延迟。路由开销对大模型（Mixtral-4x7B, expert 计算占比高）几乎可忽略，对小模型（Mixtral-4x1B）通过 fused gate 控制增长。
    3. **正交性与兼容性**：MoEsaic 不修改 expert 权重、MoE 架构或 gating 逻辑。与 quantization、pruning 等内存优化技术正交——去重后的专家可进一步量化。通过 LoRA-like interface 提供与 vLLM 生态的兼容性。

## Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching

- baseline方法是什么？
  - Baseline 是 MoE 推理中现有的 offloading 方案，分为两类：(a) Prefetch-based（如 Pre-gated MoE）——提前预取下一 MoE 层的全部或预测 experts，重叠传输与计算；(b) Cache-based（如 MoE-Infinity、LRU-based caching）——在 GPU memory 中缓存频繁激活的 experts，基于 LRU 或估计重用概率进行驱逐。
  - 全栈执行例子（Baseline: MoE-Infinity，batch_size=64，Switch-Base，XSum，5% cache ratio）：
    - **算法层**：MoE gating network 对每个 token 计算 Softmax(LinearGate(X)) → TopK (默认 k=1)。batch 中约 34 个不同 experts 被激活。
    - **系统框架层**：MoE-Infinity 在 FasterTransformer 基础上实现全局共享缓存（所有 layers 共享一个 cache pool），按估计的 expert 重用概率管理驱逐。batch_size=64 时，每次迭代激活的 ~34 experts 几乎占满 5% 缓存容量（36 experts），导致缓存被整批刷新，下一迭代命中率骤降至 <0.1%。每个 miss 触发 host→GPU PCIe 传输（128 GB/s），通信时间主导延迟。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer 的 MoE kernel 在 GPU 上并行执行 FFN 计算。Expert 参数的 host→GPU 加载由 FasterTransformer 的 offload 模块按需触发 cudaMemcpy。计算时间（batch matmul）基本恒定（约 2-2.5 ms for batch=64），而通信时间随 batch size 线性增长至 ~15-20 ms，占 per-iteration 延迟的 97% 以上。
    - **硬件架构层**：H200 GPU (141 GB HBM) ↔ Host DRAM (1 TB) 通过 PCIe 5.0 (128 GB/s) 连接。每个 expert 参数约 85 MB（7B/128/2，考虑 MoE 占约一半层）。batch=64 时 34 experts 的总传输量约 2.9 GB，占 PCIe 带宽 ~23 ms 传输时间。Prefetch-based 方案最多只能隐藏 1-2 个 expert 的传输（对应 ~2.6 ms 计算窗口），其余无法重叠。
  - Baseline 缺陷根因：在 batched inference 下，随着 batch size 增大：(1) Prefetch-based 方法的通信时间增长远超计算时间增长（batch 1→16，通信 6.53× vs 计算 1.55×），可隐藏的传输比例急剧下降；(2) Cache-based 方法的缓存命中率随 batch 增大而崩塌（batch 1→16, 5% cache ratio 下 miss rate 从 6.91% → 68.84%），因为单次迭代激活的 experts 集合频繁超过缓存容量，导致大量替换和重复传输。两种方法的共同根因是**没有利用 expert 激活的全局和时间局部性（global & temporal locality）**来差异化缓存管理。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：Diff-MoE 通过观察发现 MoE 推理中 expert 激活具有两种局部性——(a) Global Locality：每层中少数 experts 在全部推理过程中频繁激活（top-6 experts 占 33%-83% 激活）；(b) Temporal Locality：某些 experts 在短解码窗口内（3-6 次迭代）反复激活（1.13×–2.40× reactivation）。基于此提出三层差分缓存层级 + GRU predictor。
  - 全栈执行例子（Diff-MoE, batch_size=64, Switch-Base, XSum, α=5%）：
    - **算法层**：
      - Priority Initialization：离线微调阶段统计每个 MoE layer 的 expert 激活频率，top-N（默认 N=2, α=5%）标记为 globally hot → p=MaxP=2, 永久锁定在 HPCi。
      - Dynamic Priority Update：在线推理每层 MoE 执行时，按公式 p_k^i = clip(p_k^i + Δ_inc if activated | p_k^i - Δ_dec_in if inactive+cached | p_k^i - Δ_dec_out if inactive+uncached) 更新 non-global experts 的优先级。
      - Locality-Preserving Replacement：当 activated expert 优先级 ≥ threshold_hot(=1) 且不在 MPCi 时，按优先级降序排列候选；MPCi 中优先级 < 1 的 resident 按优先级升序排列；最高候选替换最低 resident。
      - GRU Predictor：6 层 GRU，以当前层 expert IDs 为输入，输出下一层各 expert 的激活概率分布。batch 内多个样本聚合 → top-2 uncached experts 预取到 LPC。
    - **系统框架层**（FasterTransformer 修改）：
      - 在初始化阶段将每个 MoE layer 的 globally hot experts 永久加载到 HPCi（各层 2 个，共 12 experts，约 1 GB）。
      - 创建 per-layer MPCi（各层 4 experts，共 24 experts，约 2 GB）和跨层共享 LPC（临时缓冲当前层激活 experts + 预取 experts）。
      - 在 gating 后拦截 expert 加载：先查 HPCi → MPCi → LPC，仅对三级都 miss 的 expert 触发 host→GPU 传输。
      - 在当前层计算期间，异步预取预测的下一层 experts 到 LPC 预取缓冲区，与计算重叠。
      - 当前层计算完成后：LPC 中的 locally hot experts 晋升到 MPCi，其余驱逐。LPC 仅保留已预取的下一层 experts。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FasterTransformer MoE kernel 不变。差异在于：Baseline 的 per-iteration expert 传输量由 batch 激活的全部 experts 决定（batch=64 约 34 experts × 85 MB = 2.9 GB），而 Diff-MoE 通过 HPC/MPC 命中削减了大量传输——28.5% cache hit rate（vs MoE-Infinity <0.1%），实际传输量大幅减少。Predictor 的 top-1 accuracy 在 batch≥16 时稳定 >90%，top-2 accuracy 则更高。预取的 1-2 个 experts（约 170 MB）传输完全被 GPU 计算时间覆盖。
    - **硬件架构层**：同一 H200 + PCIe 5.0 硬件。区别在于：Baseline 的 memory hierarchy 是 Host DRAM → GPU LPC 的简单二级结构；Diff-MoE 引入 Host DRAM → GPU LPC → GPU MPC → GPU HPC 的四级结构（LPC 为临时层，MPC 和 HPC 为持久层），且 HPC/MPC 按 layer 隔离而非全局共享。这保证在 batched 场景下每层都有独立的缓存空间（不会因其他层的 experts 挤占而驱逐本层热点），避免全局缓存竞争导致的命中率崩塌。
  - 解决 Baseline 缺陷的方式：
    1. **针对 Prefetch-based 方案（通信增长远超计算增长）**：Diff-MoE 通过 HPC 和 MPC 的差分缓存消除冗余传输——globally hot experts 一次性加载后永驻 GPU，locally hot experts 在短窗口内复用不反复传输。只有 cold experts 才按需加载，大幅削减传输总量，使有限的 PCIe 带宽不再被重复流量占满。
    2. **针对 Cache-based 方案（命中率随 batch 增大崩塌）**：Diff-MoE 用 per-layer 独立缓存 + 优先级驱动替换取代全局共享缓存 + LRU。per-layer 设计保证每层有稳定缓存容量（不被其他层挤占）；优先级机制综合频率和时效信息（LRU 仅看 recency），保护了短窗口内即将复用的 locally hot experts，避免了 LRU 在单次迭代刷新全部缓存的问题。
    3. **通用性**：Diff-MoE 不修改 MoE 模型架构本身（不改变 gating、expert 权重、top-K），使其可直接适用于任何 MoE-based LLM，部署成本低。

## Batch Tiling on Attention: Efficient Mixture of Experts Training on Wafer-Scale Processors

- baseline方法是什么？
  - Baseline 是 Conventional Uniform Batching（G=1），即 MoE 训练中 attention 层和 expert MLP 层使用相同的全局 batch size，不进行 batch tiling。
  - 全栈执行例子（Baseline, G=1）：
    - **算法层**：输入 X ∈ ℝ^(B̃×S×H)，Attention 在完整 B̃ 上执行，产生 O(S²) 级别的 KV cache 和 softmax 中间激活，当增大 B̃ 来提升 expert 计算密度时，attention 层的激活内存超出 on-chip SRAM 限制（WSE-2 为 40 GB），导致 OOM 或强制降低 B̃。
    - **系统框架层**：论文未明确说明。Cerebras CS-2 使用其专有软件栈执行训练。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：Attention kernel 在 B̃ 上执行 matmul/softmax，expert MLP kernel 在 B̃ 上按 router 分配后的 token 子集执行。B̃ 增大使 attention 内存溢出，B̃ 减小使 expert 计算密度不足——无法同时满足两者。
    - **硬件架构层**：WSE-2 的 850,000 核心和 40 GB on-chip SRAM。Attention 的激活中间结果（KV/softmax）占用大量 SRAM，限制了 B̃ 的上限；而 expert MLP 需要大的有效 batch 来利用 20 PB/s 带宽和大量核心。G=1 时两阶段使用相同 B̃，形成"batch interface conflict"。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：BTA (Batch Tiling on Attention)，通过 attention 层的 batch 维度 tiling 解耦 attention 和 expert 的 batch size 需求。
  - 全栈执行例子（BTA, G > 1）：
    - **算法层**：输入 X ∈ ℝ^(G×B×S×H)。Attention 以 per-tile batch B 执行 G 次循环（B = B̃/G），每次 attention 的 KV/softmax 中间激活仅对应 B 个序列，降低到激活内存安全范围。G 次 attention 的输出拼接为 B̃ 张量，送入 expert MLP。Router 在 B̃ 张量上执行，expert 以 B̃ 为有效 batch size 执行大 matmul，填满计算核心。
    - **系统框架层**：论文未明确说明。BTA 修改训练循环中 attention 前置的 batch reshaping/tiling 逻辑。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：Attention kernel 在 B 大小的 tile 上执行，每次 tile 的激活中间结果可放入 SRAM；G 个 attention tile 的 kernel 可串行或流水线执行。Expert MLP kernel 在 B̃ = G·B 上执行，获得足够的计算密度。关键区别：attention 和 expert 使用不同的有效 batch size，解决了 G=1 时的 trade-off。
    - **硬件架构层**：同一 WSE-2 硬件。BTA 通过算法层面的 batch tiling 避免了 attention 层因 B̃ 过大导致的 SRAM 溢出，同时保证了 expert 层因 B̃ 足够大而获得的高核心利用率。对比 baseline：G=1 时 128 experts/top_k=1 的 throughput 为 7,091 tokens/s，而 BTA (G=64) 为 49,335 tokens/s，提升约 7×。
  - 核心洞察：BTA 不是通过通信优化（如 FlashAttention、expert parallelism）或 placement 调整来解决批处理冲突，而是通过改变每个阶段处理的 token 数量（attention 少、expert 多）来直接解决 attention 内存 vs expert 计算密度的矛盾。Baseline 缺陷的根因是"attention 的峰值激活内存限制了全局 batch size 的上限，从而限制了 expert 的计算密度"，BTA 通过解耦两阶段的 batch size 直接消除了这一约束。

## Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

- baseline方法是什么？
  - Baseline 是 Ring 算法 + Pipeline 调度（Ring+Pipeline）：在 N-D torus 网络上执行 All-to-All 集合通信，将数据沿顺时针/逆时针双向拆分传输，采用 store-and-forward 方式逐跳转发以避免网络拥塞；跨维度调度采用 Pipeline 方式将数据分块后在 X→Y→Z 固定顺序上流水线化执行。
  - 全栈执行例子（Baseline: Ring+Pipeline, 3D torus, 4×4×4, HalfRing+DimRotation 论文方法对比）：
    - **算法层**：每个节点将 All-to-All 数据按 N-1 个阶段拆分，每个阶段对应不同的跳距。以 4 节点环为例，共 3 个阶段（跳距 1/2/3）。每个阶段再拆分若干子阶段完成逐跳 store-and-forward 转发。Ring 算法同时利用顺时针和逆时针链路双向通信，但在大跳距阶段存在非最短路径问题——如节点 1 到节点 4 逆时针仅 1 跳，但顺时针需 3 跳，造成额外链路带宽消耗。每个子阶段执行单跳传输（如 Stage 3-1: 节点 1→节点 2 转发紫色数据块），共需 N(N-1)/2 次单跳传输。
    - **系统框架层**：论文未明确说明具体分布式训练框架。通信算法在 MPI 层或 PyTorch Distributed 的 collective communication 后端执行（论文 real machine 实验使用 PyTorch Distributed 模块、Ascend torch_npu）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**（通信调度层）：All-to-All 按维度分解为 N 个顺序 phase（3D torus 下为 X→Y→Z）。Pipeline 调度将数据分为多个 chunk（如 6 chunks），所有 chunk 使用相同的 X-Y-Z 维度顺序。每个 phase 内使用 Ring 算法在对应维度的环上执行通信。Pipeline 不可避免地引入气泡（bubbles）——当某些 chunk 完成某维度通信后等待其他 chunk 完成才能进入下一维度，链路利用率 < 100%。Chunk 大小选择困难：过大导致维度间重叠不足，过小导致调度开销增加。
    - **硬件架构层**：N-D torus 网络拓扑。以 TPUv4 为例，4×4×4 3D-torus 含 64 个 TPU，每个 TPU 有 6 条链路（±X/±Y/±Z），链路带宽 56 GB/s。数据通过 ICI (Inter-Chip Interconnect) 在 torus 网络上进行单跳 store-and-forward 传输。Ring 算法在单维环上约需 N(N-1)/2 次单跳传输，每次传输数据量 S/N（S 为每节点数据量）。
  - Baseline 缺陷根因（三个核心问题）：(1) **非最短路径导致额外带宽消耗**：Ring 算法固定使用双向传输，大跳距阶段存在绕远路——N 节点环中最大跳距 ⌊N/2⌋，但实际最短路径可能仅需 1 跳，多跳转发消耗额外链路带宽，降低整体吞吐；(2) **Pipeline 调度引入气泡**：多维 torus 上 Pipeline 调度因固定维度顺序产生气泡，各维度链路利用率无法达到 100%，且最优 chunk 数量难以确定；(3) **无容错机制**：Ring 算法要求相邻节点间直接链路存在，任意链路故障导致整个环的 All-to-All 通信中断——而大规模训练运行数周，链路故障概率不可忽略。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：针对 N-D torus 网络的 All-to-All 集合通信，从"单维算法"和"多维调度"两个正交维度进行优化。无故障场景下提出 HalfRing（利用双向链路的最短路径算法）+ DimRotation（轮转维度顺序的无气泡调度）；故障场景下提出 FoldedRing（折叠环容错算法）+ MATE（多维度加速调度，利用健康维度链路加速故障环通信）。
  - 全栈执行例子（Paper method: HalfRing+DimRotation 无故障 / FoldedRing+MATEe 有故障, 3D torus, TPUv4）：
    - **算法层**：
      - **HalfRing（取代 Ring）**：在每个阶段，HalfRing 根据收发节点间的实际距离选择传输方向——最短路径方向。由于所有节点均沿最短路径通信，每个阶段仅消耗一个方向的带宽（顺时针或逆时针），另一方向留给配对的另一阶段使用。N=2k+1 个节点有 2k 个阶段，可配对为 k 对同时执行；N=2k 个节点有 2k-1 个阶段，剩一个未配对阶段将数据对半拆分后双向发送以充分利用带宽。对比 Ring 算法，HalfRing 在 N 为偶数时传输时间为 N/8·S/B（Ring 为 (N-1)/2·S/2B），比值 1~2×；N 为奇数时比值为 1.5~2×。关键原理：HalfRing 通过逐跳 store-and-forward 显式编排，保证无死锁（无多跳传输）、无活锁（无绕路）、无网络争用（无链路共享）。
      - **FoldedRing（取代 Ring 处理故障）**：当环上某链路故障时，故障链路两端节点之间通过所有逆时针物理链路构建逻辑补偿连接——形成"折叠环"。故障链路的顺时针方向所有健康链路保持不变，逆时针方向的全部链路被"折叠"来替代故障链路。由此 Ring 算法的逻辑通信模式得以恢复。代价是传输时间翻倍（Table 1: FoldedRing 传输时间为 (N-1)/2·S/B，而 Ring 为 (N-1)/2·S/2B，即 0.5× 性能）。
      - **MATE/MATEe（加速故障环通信）**：MATE 利用 N-D torus 中同一维度其他健康环的链路，通过其他维度链路（如 Y-dim）构建故障环上相邻节点的双向连接，使故障环也能使用 HalfRing 执行剩余数据传输。在 2D torus 例中（Fig 9），MATE 通过 Y-dim 链路连接故障 X-dim 环上节点（如 (0,1)→(0,2)→(1,2)→(1,1) 三条红色链路构成一条逻辑 X-dim 连接）。MATE 将故障环通信拆分为正常 phase（仅 FoldedRing）+ 加速 phase（利用构建的逻辑连接执行 HalfRing），可额外利用 N-1 个平面的链路同时加速传输。MATEe 增强版在正常 phase 也传输部分数据（按 HalfRing/FoldedRing 性能比静态分配），减少加速 phase 数据量。
    - **系统框架层**（调度层）：All-to-All 按维度分解为 N 个 phase。DimRotation 调度（取代 Pipeline）：将数据分为恰好 N 个 chunk（N 为维度数），第 i 个 chunk 的维度执行顺序为维度 i → i+1 → ...（循环轮转）。3D torus 下：chunk 1: X→Y→Z，chunk 2: Y→Z→X，chunk 3: Z→X→Y。三个 chunk 在三个维度上形成完美的无冲突全覆盖，实现 100% 链路利用率，零气泡。DimRotation 的 chunk 数固定为 N（最小充分数量），调度开销远小于 Pipeline。对于异构带宽或 mixed-radix torus，总时间受限于性能最差的维度——DimRotation 确保总时间不超过最差维度上完整数据的通信时间。
      - MATE 调度（多层 phase 结构）：每个 chunk 的正常 phase 后插入加速 phase M/M_e。正常 phase 在故障维度上使用 FoldedRing（MATEe）或跳过（MATE），加速 phase 利用其他维度链路构建逻辑连接后使用 HalfRing 传输。对多故障场景（同环多故障、多环各一故障、异维各一故障），MATE 为每个故障分配独立加速 phase，或当故障不在相同维度链路冲突时允许并行加速 phase。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**（通信执行层）：HalfRing/FoldedRing 算法在大规模网络中显式编排每个子阶段的单跳传输对（发送方→接收方+转发方），形成确定的通信时间表（Schedule[chunk][phase][i][j]）。该时间表可离线预计算，在实际执行时直接下发到通信后端（MPI/nccl/torch.distributed），CPU 开销从 kernel launch 中大幅削减（real machine 实验显示 startup time 显著降低）。每个单跳传输的数据量固定（S/N 或减半），保证链路负载完全均衡。
    - **硬件架构层**：同一 N-D torus 硬件。对比 baseline：Ring+Pipeline 受限于绕远路传输 + Pipeline 气泡 → 链路利用率不足；HalfRing+DimRotation 通过最短路径 + 无气泡调度 → 每维度每时刻所有链路处于活跃传输状态（Fig 13 维度利用率显示 DimRotation 三轴 100% 利用率，Pipeline 存在周期性下降）。故障场景下：FoldedRing+Pipeline 受限于故障环传输速度减半 + 气泡 → 性能降至 fault-free 的 0.55×；MATE/MATEe 利用其他维度链路（额外平面数 = N-1）加速后 → 性能超过 fault-free baseline（1.36×/1.37×）。
  - 解决 Baseline 缺陷的方式：
    1. **针对"非最短路径导致额外带宽消耗"**：HalfRing 根据收发间实际距离选择最短路径方向，每个阶段仅使用单向带宽，配对阶段利用对向带宽，消除了 Ring 算法在大跳距阶段的绕远路浪费。最短路径 + 全带宽利用 = 单维环上带宽和延迟均为最优（optimal in both bandwidth and latency），理论加速比 1~2×（取决于 N 的奇偶性）。
    2. **针对"Pipeline 调度引入气泡"**：DimRotation 将固定维度顺序替换为轮转顺序——N 个 chunk 各以不同维度作为起始维度循环执行，恰好覆盖所有维度在任意时刻的并行传输需求。气泡被完全消除，链路利用率达到 100%。Chunk 数量固定为 N（最小充分数量），调度开销可控。等价于在 N 维 torus 上实现了最优的 collective 调度。
    3. **针对"无容错机制"**：整套方案正交地扩展了容错能力——FoldedRing 在单维环上通过"折叠"全部反向链路构建故障链路的逻辑替代路径，保持 Ring 通信模式完整；MATE 利用 torus 的多维正交特性——健康维度链路可在不冲突的前提下构建故障环相邻节点的逻辑连接——将故障环通信部分卸载到健康环，使性能甚至超过 fault-free baseline。理论依据：N-D torus 中每个平面包含故障环时可额外提供 1 组双向链路，共 N-1 个加速平面可用。MATE 同样适用于 OCS 故障、多故障（同环、多环、异维）等更复杂场景，且加速 phase 可并行化。


## EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance

- baseline方法是什么？
  - Baseline 是静态图模式下基于 expert parallelism 的标准 MoE 训练（如 Switch Transformers 的 MindSpore 静态图移植、Fastermoe 的 MindSpore 算子迁移）。静态图要求输入 shape 在编译前固定，所有 expert 共享同一 capacity，无法在运行时动态调整。
  - 全栈执行例子（Baseline: Switch Transformers 在 MindSpore 静态图, 32 Ascend 910, Expert Parallelism = 16, MoE-θ 21B）：
    - **算法层**：Gating network 对每个 token 执行 Softmax(W_gate · x) → TopK(k=1) 路由到最相关 expert。各 expert 的 FFN 独立处理分配给它的 token 子集，输出按 token 聚合。在大规模 MoE 中，token 分布极度不均匀——约 70% token 集中于 2 个 hot experts，其余冷 experts 仅处理少量 token。
    - **系统框架层**：MindSpore 2.0 静态图模式。使用 DP=16、MP=2、EP=16 的混合并行策略。32 个 expert 分布在 16 个 expert-parallel 组中的 accelerator 上。每个 iteration 中，All-to-All 通信在 dispatching 阶段将 token 从源 accelerator 发送到对应 expert 所在的 accelerator，在 combining 阶段将处理后的 token 回传。
    - **编译框架层**：MindSpore 静态图编译器在训练前编译整个计算图。所有 expert 使用相同的固定 capacity（由编译时预定义的最大 token 数决定），无法在运行中改变。hot expert 的输入超出 capacity 时剩余 token 被直接丢弃；cold expert 的输入远小于 capacity 时需填充 zero vectors 至固定 shape。
    - **kernel调度层**：Ascend 910 上执行 MindSpore 算子。All-to-All 通信占纯通信时间的 75%（论文 profiling 分析，32 加速器集群）。Expert FFN 计算时，cold expert 因大量 padding 浪费约 50% 的计算资源——zero vectors 参与的矩阵乘法结果被丢弃但计算已执行。
    - **硬件架构层**：4 节点 × 8 Ascend 910，节点间 100 GB/s RoCE。节点内 8 卡共享高速互联带宽。All-to-All 跨节点通信成为瓶颈——32 加速器集群中纯通信时间占 MoE 训练总时间的 75%，大量 AI accelerator 计算周期因等待通信而闲置。
  - Baseline 缺陷根因（两个核心问题）：(1) **负载不均 + All-to-All 通信开销**：token 分布的极端不均衡（70% token 集中在 2 个 expert）导致 hot expert 过载/cold expert 空转，同时 expert parallelism 要求将 token 跨节点发送到 expert 所在 accelerator 处理，All-to-All 通信占比高达 75%，训练效率受通信瓶颈限制；(2) **静态 capacity 下的 token 丢弃与 padding 浪费**：静态图要求编译前固定 expert capacity，hot expert capacity 不足导致 token 丢弃影响精度，cold expert 因 capacity 远大于实际负载而大量 zero-padding，浪费约 50% 计算资源。这两个问题在动态图框架（PyTorch）下有成熟的解决方案（如 Fastermoe 的动态 shadowing、Megablocks 的动态 capacity），但在高效的静态图框架（MindSpore）下缺乏对应优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：EfficientMoE 在 MindSpore 静态图框架内，通过三阶段优化解决上述问题——(1) Sampler + Load Prediction Model 实时监控并预测 expert 负载；(2) Dynamic Expert Scheduling 将 hot expert replica 部署到冷 expert 所在 accelerator，本地化 token 处理；(3) Expert Capacity Model 在编译前为 hot/cold expert 设置差异化 capacity。
  - 全栈执行例子（EfficientMoE, MoE-θ 21B, 32 Ascend 910）：
    - **算法层**：
      - Sampler 在每个 iteration 收集 token 分布、expert 参数、内存和通信四维信息（Table II）。
      - Load Prediction Model 在 m 次 iteration 的 cycle 中统计各 expert token 负载 L_i^k = mean(Top_p%(sorted(token_counts)))，并计算 accelerator 负载 D_j = Σ(Compute(T_i) + Memory(T_i))（公式 1-2）。
      - Expert 按负载因子与阈值 q=60% 比较，分为 hot（负载 > q·max_load）和 cold（负载 ≤ q）。
      - Dynamic Schedule（Algorithm 1）：为 hot expert 生成 replica，调度到有空闲资源的 cold expert 所在 accelerator。带来的关键变化：原本需跨节点 All-to-All 发往 hot expert 的 token，现在被本地 replica 直接处理，仅需同步 replica 参数更新（体积远小于 token 传输量）。
      - Expert Capacity Model（公式 3-6，Algorithm 2）：C_j^i = (1-r)·B + r·(1/m)·ΣF_t，其中 B 为统计 baseline，r 由峰值负载决定（公式 4），F = γ·(T_i-B)/Total_tokens（公式 5）。hot expert C_j^i > B（增大 capacity 减少 token 丢弃），cold expert C_j^i < B（减小 capacity 减少 padding）。总内存需满足 M_total ≤ accelerator_memory（公式 6）。
    - **系统框架层**（MindSpore 2.0 + Mindformers 1.0 修改）：
      - 在 MindSpore 静态图内核中插入 Sampler 模块，在每个 training iteration 的 MoE 层前拦截并收集 token 路由信息。
      - 在静态图编译前，Load Prediction Model 读取 sampler 历史数据，输出 hot/cold 分类和 per-expert capacity 设置。
      - Dynamic Scheduling 模块在 load prediction cycle 边界修改 MindSpore 的 expert placement（将 hot expert replica 部署到目标 accelerator），并调整 token routing 逻辑——gate 后的 token routing 表被修改为：若有本地 replica 则路由到本地，否则保留 All-to-All 到原 expert。
      - 对比 Baseline 的"固定 expert placement + 全量 All-to-All"，EfficientMoE 变为"动态 replica placement + 部分本地处理 + 剩余 All-to-All"。
    - **编译框架层**：MindSpore 静态图编译器。EfficientMoE 不修改编译器内部，而是在编译前注入差异化的 expert capacity（C_j^i）以替代原本统一的固定 capacity。编译后每个 expert 的输入 buffer shape 已按各自 capacity 分配，hot expert buffer 更大（接收更多 token），cold expert buffer 更小（减少 padding）。capacity 调整在 cycle 间触发 re-compilation（论文未详细说明 re-compilation 策略的具体开销）。
    - **kernel调度层**：Ascend 910 上算子执行逻辑不变。关键差异在数据流：(a) Baseline 中每个 iteration 有大量 token 通过 RoCE 跨节点 All-to-All 传输到 hot expert，通信占据了 75% 时间；(b) EfficientMoE 中，因 hot expert replica 本地化了大量 token——原本需跨节点发送的 token 数据流变为本地 NVLink/内存拷贝，仅剩少量 cold expert token 需 All-to-All。同时 cold expert 的 zero-padding 因 capacity 减小而大幅削减，减少了约 35% 的无效计算量。
    - **硬件架构层**：同一 4 节点 Ascend 910 集群。核心变化：All-to-All 通信从占迭代时间的 75% 显著降低（约 12% 通信时间减少），计算资源浪费从约 50% 降低（35% 计算资源节省），综合训练时间缩短 30%。RoCE (100 GB/s) 的带宽瓶颈因 All-to-All 流量减少而缓解，accelerator 计算利用率因减少 padding 而提升。
  - 解决 Baseline 缺陷的方式：
    1. **针对"负载不均 + All-to-All 通信瓶颈"**：EfficientMoE 用"expert replica 调度"替换"token 跨节点路由"。核心转换——将大体积高频的 token All-to-All 通信变为小体积低频的 expert 参数同步（replica 参数更新在 load prediction cycle 边界同步一次而非每 iteration）。这类似于计算与数据的 co-location 优化——不是"把数据送给计算"，而是"把计算搬到数据所在地"。在 4 节点 32 accelerator 集群上实现约 12% 通信时间降低、30% 端到端加速。
    2. **针对"静态 capacity 下的 token 丢弃与 padding 浪费"**：EfficientMoE 用 Expert Capacity Model 在静态图编译前的 cycle 边界为各 expert 设置差异化 capacity——hot expert replica 获得更大 capacity（减少 token 丢弃，保护精度），cold expert 获得更小 capacity（减少 zero-padding，节省 35% 计算资源）。capacity 的周期性重评估（每 m 次 iteration 基于 token 分布变化触发）解决了"静态 capacity 无法适应负载变化"的核心矛盾。
    3. **针对"动态图优化方法无法用于静态图"**：EfficientMoE 将所有优化保持在静态图范式内——load prediction cycle 的周期性评估 + 编译前 capacity 注入 + replica placement 修改——不依赖动态图运行时的 shape 变化能力。这使得 MindSpore/Ascend 生态的 MoE 训练能同时享受静态图的计算效率优势和 EfficientMoE 的负载均衡优化。

## Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

- baseline方法是什么？
  - Baseline 是 DPMoE（Data Parallel MoE），即传统的 bound data parallel + expert parallel 的 MoE 并行架构。每个 data parallel replica 持有完整 backbone + E/D 个 expert，experts 分布在所有 DP rank 上。每个 MoE layer 需要两次 all-to-all 通信（dispatch 和 gather），每次传输 b×s×h 大小的 hidden embeddings。
  - 全栈执行例子（Baseline: DPMoE, GPT-3 Medium → 6.7B MoE, 64 experts, top-1 gating, DP=4 + TP=8 + EP=64, 32 V100, Megatron-LM v2.5 + DeepSpeed v0.5.10）：
    - **算法层**：MoE layer 执行 gating(W_gate @ h) → softmax → TopK=1 routing → 1st all-to-all（dispatch tokens 到对应 expert 所在 rank）→ per-expert FFN(GeLU(XA)B) → 2nd all-to-all（gather processed tokens 回原 rank）→ Dropout + LayerNorm。两次 all-to-all 占 forward 总时间的 65.5%（实测），MoE 层总时间占 forward 的 82.6%。token→expert → all-to-all 需跨节点 InfiniBand 传输（BW=12.5 GB/s），expert FFN 计算被通信严重拖慢（t'_a2a/t'_FFN > (E-1)E/16 ≈ 252 for E=64）。
    - **系统框架层**：Megatron-LM v2.5 管理 TP + DP + EP 三路并行。DeepSpeed ZeRO optimizer 用于 DP ranks 间 optimizer state 分片以节省显存。bound DP+EP 要求 E 能被 D 整除（如 E=64, D=4 则每 rank 16 experts）。DP 和 EP 维度强耦合——改变 DP 规模会改变 expert 分布，无法灵活配置。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：PyTorch/Megatron 标准的 all-to-all 通信（NCCL backend）、GEMM FFN（cuBLAS）。ring-style all-to-all 的时间随 EP world size 线性增长，跨节点 InfiniBand 带宽远低于节点内 NVLink。
    - **硬件架构层**：华为云 V100 SXM2 服务器（8 GPU/节点），节点内 NVLink 互联，节点间 InfiniBand 100 Gb/s（BW≈12.5 GB/s）。all-to-all 跨节点通信是性能瓶颈——每个 expert 的 EP world size = 64 = 8 节点，ring all-to-all 延时 O(N·m/B)。
  - Baseline 双缺陷：(1) **all-to-all 通信瓶颈**：两次 all-to-all 占 forward 时间 65.5%，严重限制 training throughput（6.7B DPMoE 仅达到 backbone 的 66.2% 吞吐）；(2) **backbone 扩展受限**：DPMoE 的 DP+EP 绑定使得每个 DP rank 只能容纳 ≈single-expert 大小的 backbone，无法通过 tensor parallel 或 pipeline parallel 有效扩展 backbone，而最新研究表明 thick backbone + moderate experts 更有优势。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 PPMoE（Pipeline MoE），通过两大核心设计解决 DPMoE 缺陷：(1) 将 expert parallel 与 tensor parallel 绑定（代替与 data parallel 绑定），所有 experts 位于同一 TP group（同一节点），用 tensor index slicing 替代 all-to-all；(2) 无缝集成 pipeline parallel 扩展 backbone。
  - 全栈执行例子（PPMoE, GPT-3 Medium → 6.7B, 64 experts, top-1 gating, TP=8 + PP=4 + DP=1 + EP=64, 32 V100, Megatron-LM v2.6）：
    - **算法层（解决 all-to-all 通信瓶颈）**：
      - 传统 DPMoE 流程：hidden → gating → all-to-all dispatch → expert FFN → all-to-all gather → output。关键瓶颈是 all-to-all。
      - PPMoE 流程：hidden → TP all-reduce sync（与 TP 的 attention/FFN 相同）→ 各 TP rank 独立执行 gating（因输入相同，产出相同 dispatching order）→ tensor index slicing（index_select, 纯本地操作，零通信）→ 串行执行 N=E/T 个 local expert FFN → index assignment 重建 → inner-node all-reduce（与 TP FFN 的 all-reduce 完全相同，走 NVLink）。
      - 核心替换：**all-to-all (跨节点 InfiniBand) → index_select (本地) + all-reduce (节点内 NVLink)**。MoE all-reduce 时间 = FFN all-reduce 时间（差异仅 1.9% of total forward），通信从 forward 的 65.5% 降至 20.7%。
    - **系统框架层（解决 backbone 扩展受限）**：
      - 传统 DPMoE：DP+EP 绑定，每 DP rank backbone = single-expert 容量，扩展 backbone 需要重新分配 expert 布局。
      - PPMoE：expert parallel 与 tensor parallel 绑定在同一节点内。MoE 层的输入/输出格式和通信模式与非 MoE FFN 完全一致（均为 TP all-reduce）。因此 dense 模型的 TP+PP 框架可"即插即用"地替换部分 FFN 为 MoE 层 → 无需修改 pipeline stage 划分和通信拓扑。
      - 功能等价性：DPMoE 是"空间并行"（不同 DP rank 同时处理 micro-batches），PPMoE 是"时间并行"（同一 pipeline 串行处理 micro-batches，通过 gradient accumulation 等效全局 batch）。两者在数学上等价，但并行架构不同。
    - **编译框架层**：论文未明确说明。基于 Megatron-LM v2.6 实现，核心修改在 model definition 层（expert parallel 与 TP 绑定）和 forward function（index slicing 替代 all-to-all）。
    - **kernel调度层**：PyTorch 标准的 index_select、GEMM（cuBLAS）、all-reduce（NCCL over NVLink）。串行 N 个 expert FFN 的计算速度与处理单个大 batch 几乎相同（因为低层算子优化），无额外性能损失。all-reduce 走 NVLink（300 GB/s）而非 InfiniBand（12.5 GB/s），通信带宽比 DPMoE 高约 24×。
    - **硬件架构层**：同一 V100 集群。关键差异：DPMoE 的 all-to-all 跨 8 节点 InfiniBand（每 rank 传 bsh bytes，ring 延时 O(E)×），PPMoE 的 all-reduce 仅在单节点 8 GPU 间 NVLink（每 rank 传 bsh/T bytes，延时 O(log T)）。因此 PPMoE 的通信效率远高于 DPMoE。
  - 解决 Baseline 缺陷的方式：
    1. **针对 all-to-all 通信瓶颈**：PPMoE 通过将 experts 全部置于同一节点（TP group）内，从根本上消除了跨节点 all-to-all。dispatch 用本地 index slicing 替代（零通信），gather 用节点内 NVLink all-reduce 替代（与标准 TP 通信一致）。结果：MoE 通信从 forward 65.5% → 20.7%，total MoE forward 从 82.6% → 38.2%。
    2. **针对 backbone 扩展受限**：PPMoE 通过使 MoE 层与 TP+PP 框架兼容（因输入/输出格式和通信模式一致），实现了 backbone 在 depth 维度（PP）和 width 维度（TP）的自由扩展。6.7B backbone 可扩展为 143B PPMoE，且 even 达到 backbone（20× smaller）的 90.7% 吞吐。
    3. **功能等价性保证**：PPMoE 与 DPMoE 在数学上等价——相同的 global batch、相同的 gradient accumulation、相同的更新规则。Convergence 验证确认 training/validation loss 与 backbone 一致收敛。PPMoE 仅改变并行架构，不改变模型语义。

## ResMoE: Space-efficient Compression of Mixture of Experts LLMs via Residual Restoration

- baseline方法是什么？
  - Baseline 有三类 MoE 压缩方法：(1) **Expert Merging**：M-SMoE、MEO、OneS 等将多个 expert 合并为更少的 expert（如 8→2），直接减少 expert 数量。Git Re-Basin、OT Fusion 通过 permutation + optimal transport 对齐权重后合并。核心缺陷：直接减少 expert 数量导致各 expert 的专业知识大量丢失，且通过 OSE（Oblivious Subspace Embedding）框架理论分析证明，压缩维度 d < O(p log p/ε²) 时合并误差不可忽略；(2) **Expert Pruning**：基于重要性评分移除整组 expert 权重，依赖 calibration data 的 i.i.d. 假设；(3) **Direct Compression**：unstructured pruning、structured pruning、Wanda、truncated SVD 直接对每个 expert 独立压缩，未利用 expert 间的共同模式。
  - 全栈执行例子（Baseline: M-SMoE, Mixtral 8×7B, 压缩率 75%, 4×V100 32GB）：
    - 算法层：M-SMoE 利用 router gating score 分布将 8 个 expert 合并为 2 个——每个合并 expert 是原 expert 权重的加权平均。参数量降至原始的 25%，但丢失 6 个 expert 的专有知识。WikiText PPL 从 3.87 升至 10.45。
    - 系统框架层：HuggingFace transformers + PyTorch，合并后直接加载新 checkpoint 推理。
    - 编译框架层：论文未明确说明。
    - kernel调度层：标准 cuBLAS GEMM，合并后仅 2 个 expert 参与计算。
    - 硬件架构层：4× Tesla V100 32GB。内存从 87GB 降至约 22GB，但精度损失不可接受。核心矛盾：减少内存必须丢 expert，丢 expert 就丢精度——在 baseline 中互斥。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：ResMoE 核心创新是**保留所有 expert，但用 barycenter + compressed residual 存储**——不减少 expert 数量，而减少每个 expert 的表示开销。通过 Wasserstein barycenter 提取所有 expert 的共同模式（barycenter expert），仅压缩残差 Δ_k = T_k W_k - W_ω。
  - 全栈执行例子（ResMoE (UP), Mixtral 8×7B, 75% 压缩率, 4×V100 32GB）：
    - 算法层（解决"合并丢专业知识"和"直接压缩未利用共同模式"的缺陷）：
      - **Barycenter Expert Extraction**：利用 MLP = Σ bottleneck-1 sub-MLP 的视角（E_k(x) = Σ_i W_{k,·,i}^{(2)} · σ(⟨W_{k,i,·}^{(1)}, x⟩ + b_{k,i}^{(1)}) + b_k^{(2)}），将每个 MLP 的每一行作为一个"粒子"构造均匀分布 μ_k 在 W_k 的行上，求解 Wasserstein barycenter: μ_ω = argmin (1/N) Σ W_2^2(μ_k, μ_ω)。Proposition 4.1 证明 W_ω + T_k = p_I · OT(μ_k, μ_ω) 是优化问题 min (1/N) Σ [||T_k W_k - W_ω||_F^2] 的最优解。
      - **对比 baseline**：Baseline 合并 8→2 个 expert 直接丢专家；ResMoE 保留 8 个 expert，以 barycenter + residual 表示。barycenter 捕获共同模式，残差 Δ_k 独立编码各 expert 的差异化信息。残差矩阵 Δ_k 的权重幅值远小于原始 W_k（大部分值接近 0），同样 75% sparsity 下信息损失远小于直接压缩 W_k。
      - 结果：Mixtral PPL 3.87 → ResMoE UP 5.38（vs M-SMoE 10.45, vanilla UP 13.03）。LAMBADA ACC 74.05 → ResMoE UP 69.44（vs M-SMoE 58.57, vanilla UP 36.10）。
    - 系统框架层：PyTorch + HuggingFace transformers。压缩离线完成（one-shot, <1 day for Mixtral vs OT Fusion >4 days），推理时动态恢复 expert。标准推理框架无需修改。
    - 编译框架层：论文未明确说明。
    - kernel调度层：标准 cuBLAS GEMM。ResMoE 的 runtime（38.85s/39.44s Mixtral Winogrande）几乎与原始模型相同。ResMoE (SVD) 的 FLOPs 从 3.26 降至 2.73 TFLOPs。
    - 硬件架构层：同 baseline V100/A100。ResMoE (SVD) 将 Mixtral 单层 MoE 内存从 5,376MB 降至 2,016MB（Table 10），DeepSeekMoE 单层从 2,112MB 降至 561MB——overhead 随 expert 数增加而摊薄。
  - 解决 Baseline 缺陷的方式：
    1. **针对"合并丢专业知识"**：保留所有 expert，barycenter + residual 各自独立编码。WikiText PPL: 3.87 → ResMoE UP 5.38 vs M-SMoE 10.45（~2× lower PPL）。
    2. **针对"直接压缩未利用共同模式"**：barycenter 使残差 Δ_k 接近 0，压缩残差几乎不丢信息。vanilla UP PPL 13.03 vs ResMoE UP PPL 5.38（同 75% sparsity）。
    3. **针对"逐层对齐计算开销"**：ResMoE 基于 MLP = sub-MLP ensemble 的分布视角一次性提取 barycenter（同时对齐 W^{(1)} 和 W^{(2)}），避免 layer-by-layer 策略的多次 permutation。Mixtral: <1 天 vs OT Fusion >4 天。
    4. **通用性**：encoder-decoder (Switch)、decoder-only (Mixtral)、fine-grained MoE (DeepSeekMoE 64 experts) 均有效。One-shot、data-agnostic、无需 retraining。

## ReMoE Fully Differentiable Mixture-of-Experts with ReLU Routing

- baseline方法是什么？
  Baseline 是 vanilla TopK-routed MoE（Token-choice dropless TopK, dMoE）。其路由函数为 R(x) = TopK(Softmax(x · W_l), k)，即先对 router logits 做 Softmax 归一化为概率分布，再通过 TopK 保留最大的 k 个值，其余强行置零。核心缺陷：TopK 操作在第 k 大值 x_{[k]} 处引入跳变不连续性——当 Softmax 输出从 (0.51, 0.49) 变为 (0.49, 0.51) 时，TopK 输出从 (0.51, 0) 跳变为 (0, 0.51)，导致训练目标函数非连续、非可微，限制了 router 的优化效果和模型的可扩展性。此外，TopK 使每个 token 被固定路由到恰好 k 个 expert，无法根据 token 难度动态分配计算资源。
  全栈执行例子（Baseline: dMoE, N=182M/E=8/k=1, LLaMA architecture, 8×A100 GPU, Megatron-LM）：
  - **算法层**：MoE layer 中 router 执行 Softmax(x · W_l) → TopK(·, k=1) → 每个 token 选择 1 个 expert → expert FFN (SwiGLU: W_down @ (SiLU(W_gate @ x) * W_up @ x)) → 加权求和。TopK 在第 k 大值处的跳变使 loss 在此处不可微，梯度估计不准确。训练需额外的 auxiliary load balancing loss (weight=0.01) 防止 routing collapse。
  - **系统框架层**：Megatron-LM 支持 Data/Expert/Tensor/Pipeline Parallelism。MoE layer 执行 all-to-all dispatch → per-expert FFN → all-to-all combine。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 cuBLAS GEMM + NCCL all-to-all。无自定义 kernel。
  - **硬件架构层**：8×A100 GPU。TopK routing 的离散性不影响硬件执行效率，但限制了模型性能上限。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 ReMoE，用 ReLU 路由 R(x) = ReLU(x · W_l) 完全替代 TopK(Softmax(x · W_l), k) 路由，消除不连续性。配合自适应 L1 正则化精确控制稀疏度至目标水平 (1-k/E)，以及精炼负载均衡 L1 正则化天然防止 routing collapse。
  全栈执行例子（ReMoE, N=182M/E=8/k=1, LLaMA architecture, 8×A100 GPU, Megatron-LM）：
  - **算法层（解决 TopK 不连续性 + 固定激活数缺陷）**：
    - **ReLU 路由**：将 TopK 的断点 x_{[k]} 统一设为 0，即 ReLU(x)_e = x_e · 1{x_e ≥ 0}。当 x_e 在 0 附近平滑过渡时，输出连续变化（如 (0.01, 0) → (0, 0.01) 是连续的），消除了 TopK 的跳变不连续性（如 (0.51, 0) → (0, 0.51) 是不连续的）。训练 pipeline 因此完全可微，梯度流畅通。
    - **自适应 L1 正则化**：λ_{i+1} = λ_i · α^{sign((1-k/E)-S_i)}，当稀疏度 S_i < 目标时扩大 λ，反之缩小。正则项 L_reg = (1/LT) Σ||R(x)||_1 对所有非零 router output 加梯度偏置 λ_i/(LT)，驱动输出向零，使平均稀疏度稳定在 (1-k/E)，保证 FLOPs 与 TopK MoE 统计等价。
    - **负载均衡精炼**：f_{l,e} = (E/kT) Σ 1{R(x_t^l)_e > 0} 作为 per-expert 权重，使过载 expert 的 router output 受到 λ_i · f_{l,e}/(LT) 的更强梯度惩罚，自动均衡负载。与 TopK MoE 的 auxiliary load balancing loss 在数学形式上等价，但 ReLU 输出可任意小（无 Softmax 的和为 1 约束），因此需 λ_i 自适应更新以防止 routing collapse 至全零。
    - **动态 expert 分配**：每个 token 激活的 expert 数量可变——高频 token（如 "the", "\n"）激活较少 expert，低频 token（如特殊符号、罕见词）激活更多 expert，类似 Huffman 编码的自适应资源分配。Domain 级别也呈现差异化激活。
    - **自然三阶段训练**：Stage I (dense warm-up, ~100 steps)：λ 小，几乎所有 expert 被激活，从随机初始化中分化。Stage II (sparsifying)：L_reg 增强，expert 开始稀疏化。Stage III (stable sparse)：稀疏度稳定在目标值。
    - **对比 baseline**：baseline 的 TopK 在第 k 大值处不可微 → ReMoE 的 ReLU 在 0 处连续可微。Baseline 固定 k 个 expert/token → ReMoE 动态可变数量。Baseline auxiliary load balancing loss 需手动调权重 → ReMoE 通过自适应 λ_i 在单一 L1 正则化公式中统一稀疏度控制和负载均衡。
  - **系统框架层**：与 baseline 相同（Megatron-LM），ReLU routing 作为 drop-in replacement，仅需替换路由函数。支持 Data/Tensor/Pipeline/Expert Parallelism（全兼容）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：与 baseline 相同。ReMoE 仅改变路由逻辑（ReLU 替代 Softmax+TopK），不修改 FFN 计算或通信。Training throughput 差异在 -2.29% ~ +3.89% 之间（近似等效）。
  - **硬件架构层**：同一 8×A100 硬件。Stage I/II（~100 steps, ~0.17% 总步数）有额外计算开销（激活更多 expert），但总体上 negligible。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对 TopK 的不连续性**：ReLU 将断点统一设为零，从 (0.51,0)→(0,0.51) 的跳变变为 (0.01,0)→(0,0.01) 的连续过渡，训练目标完全可微，router 优化更稳定。实验证实：flip rate 在 E=16/32 时 ReMoE 比 MoE 低 2-3×，flip count 不随 E 增长（MoE 的 flip count 随 E 增大而增大）。
    2. **针对固定激活数**：ReLU routing 各 expert 独立决策，token 可激活 0~E 个 expert，实现动态计算分配——高频 token 激活少、低频 token 激活多。
    3. **针对负载均衡需额外 loss**：L1 正则化的精炼版 f_{l,e} 权重天然实现负载均衡，与稀疏度控制在单一公式中统一，无需额外 auxiliary loss。
    4. **Superior scalability**：ReMoE 随 E 增长的性能提升斜率比 MoE 更陡（从 E=4 到 E=128，ReMoE 的 loss 下降更大），且 fine-grained ReMoE (G=32/64) 达到理论上限 Dense×8 的性能而 FLOPs 显著更少。

## PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning

- baseline方法是什么？
  - Standard Prompt Tuning (PT): 将可训练的 soft prompt 向量 P∈R^{T×H}（T=prompt length, H=hidden dim）直接 prepend 到输入 embedding 序列前，冻结 base model 所有参数，仅优化 P。Soft prompt 从 task-relevant 文本的 word embeddings 初始化。训练 loss 为语言模型的标准 NLL loss，只计算非 prompt 位置的 token。
  - 全栈执行例子（Baseline: PT, LLaMA-3.2-1B-Instruct, 1 node 4×A100 80GB, MRQA QA 任务, 81K trainable params）：
    (a) 模型推理算法层——输入文本经 tokenizer → base model embedding 层 → soft prompt P (40×2048) prepend 到 embedding 前 → frozen LLaMA decoder 逐层 self-attention + FFN 前向 → 输出 token logits → 仅计算非 prompt 位置的 CE loss → 反向传播仅更新 P（81K/1.2B=0.007% 参数）。PT 无 MoE、无分解、无 routing，所有输入共享同一 soft prompt。
    (b) 系统框架层——HuggingFace Transformers Trainer + DeepSpeed ZeRO-3。AdamW optimizer, lr=2e-5, constant_with_warmup schedule, warmup 500 steps, per_device_batch_size=32, gradient_accumulation=2。Inference 用 greedy decoding (do_sample=False, num_beams=1, temperature=1.0)。
    (c) 编译框架层——论文未明确说明。
    (d) Kernel 调度层——论文未明确说明。标准 PyTorch embedding concat + transformer forward，无自定义 kernel。
    (e) 硬件架构层——4×A100 80GB，基于 PyTorch 2.3.1+cu118 的标准 GPU 计算，无自定义硬件修改。
  - Baseline 的核心缺陷：(1) PT 使用单一的共享 soft prompt，缺乏对不同输入语义的适应能力——同一 prompt 处理所有输入，无法根据输入内容动态调整；(2) 参数量固定，每个 soft prompt 占据 T×H 参数，多个 task 需要多个完整的 prompt，线性增长；(3) 在数学推理任务上表现弱（PT 46.16% accuracy vs LoRA 56.47%），在 QA 任务上表现较好但并非最优。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - PT-MoE 通过两大核心设计解决 PT 缺陷：
    1. **Matrix Decomposition（矩阵分解）解决参数效率与共享问题**：将每个 expert soft prompt P_i∈R^{T×H} 分解为 P_i = A_i B，其中 A_i∈R^{T×R} 为 prompt 专属矩阵，B∈R^{R×H} 为所有 experts 共享矩阵。总参数量从 O(NTH) 降至 O(NTR+RH)。这实现了：(a) 参数跨 expert 共享——B 矩阵捕获公共知识，各 expert 的 A_i 仅编码差异化知识；(b) 低秩约束作为正则化，防止过拟合，尤其在 out-of-domain 数据上提升泛化。
    2. **Dynamic MoE Router 解决输入自适应问题**：Router 根据输入 embedding 均值 μ∈R^H 动态计算 routing weights w = softmax((Wμ+b) ⊙ (1+ε))，使用 top-k 硬选择（straight-through estimator）和 probationary routing（输出乘 router confidence），实现输入条件化的 expert 选择与加权聚合 P = Σᵢ w_i A_i B。

  - 对比 baseline 的全栈执行例子（PT-MoE, LLaMA-3.2-1B-Instruct, 1 node 4×A100 80GB, MRQA QA 任务, 80K trainable params）：
    (a) 模型推理算法层——输入 text → embedding E (b×s×h) → mean pooling μ (b×h) → linear router Wμ+b (b×n=2) → noise injection + softmax + top-k hard selection → routing weights w (b×2) → 对 i=1,2: weighted A_i ∈ R^{k×d} (k=40, d=36) → summed P_raw = Σ w_i A_i (b×k×d) → project to model dim P = P_raw × B (B∈R^{36×2048}) → concat(P, E) → frozen LLaMA decode → NLL loss only on non-prompt positions → backprop to router + {A_i} + B（80K params）。初始化：task-relevant texts → word embeddings → SVD → UΣV^T → A_i init from U_{:R} Σ_R^{1/2}, B init from Σ_R^{1/2} V_R^T。
    (b) 系统框架层——与 PT 相同：HuggingFace Transformers + DeepSpeed ZeRO-3, AdamW lr=2e-5, warmup 500 steps, batch_size=32, gradient_accumulation=2。增加 router 的 noise 调度——训练时加乘性高斯噪声 σ² 鼓励探索，推理时去掉噪声保证确定性。
    (c) 编译框架层——论文未明确说明。
    (d) Kernel 调度层——论文未明确说明。router linear + weighted sum + matrix multiply (P_raw × B) 均为标准 PyTorch 操作，无自定义 kernel。
    (e) 硬件架构层——与 PT 相同：4×A100 80GB，无自定义硬件。

  - 效果对比：
    - QA (MRQA F1 avg): PT 56.77% → PT-MoE 58.26%（+1.49 pts），SMoP 56.25%（仅用 MoE 反而下降），DPT 55.77%（仅用分解也下降），证明分解与 MoE 的互补性——单独使用均不如 PT，组合后超越 PT。
    - Math (Accuracy avg): PT 46.16% → PT-MoE 56.91%（+10.75 pts），超越 LoRA 56.47%。在 Division 子集上 PT-MoE 79.16% > LoRA 52.08%，体现跨任务一致性。
    - 参数效率: PT-MoE 80K < LoRA 106K < HydraLoRA 278K，同时性能超越。PT-MoE 用 25% fewer params than LoRA，达到更优性能。
    - 核心设计洞察：矩阵分解提供参数共享与正则化（B 矩阵跨 expert 共享低秩基），MoE routing 提供输入条件化动态适应（不同输入激活不同 A_i 组合）。两者协同：没有分解，MoE 的参数随 expert 数线性增长失去效率优势；没有 MoE，分解的共享 B 缺乏对不同输入模式的差异化能力。

## PiKV KV Cache Management System for Mixture of Experts

- baseline方法是什么？
  - Baseline 是标准 MoE 推理中的 dense KV cache 管理方案：每个 GPU 保存全部 token 的完整 KV cache 副本（或通过模型并行复制），所有 GPU 间需全局同步 KV 状态。推理时每个 token 生成需要 attend 到所有 prior tokens 的完整 KV cache，导致 O(BLhE) 的注意力计算复杂度和 O(EL) 的 per-device 内存消耗。KV cache 以未压缩的全精度格式（FP16/BF16）存储，无自适应驱逐策略（采用简单 LRU 或 sliding window）。
  - 全栈执行例子（Baseline: Dense KV Cache + Static Eviction, MoE 7B, 16 experts, 128K context, multi-GPU）：
    - **算法层**：MoE gating（Softmax TopK）选 k 个 experts per token → 所有 prior tokens 的 (K,V) 不分 expert 归属全部参与注意力计算 → C_dense = BLhE FLOPs。Router 按输入语义独立决策，无 KV locality 感知。
    - **系统框架层**：vLLM/类似 serving 框架管理 continuous batching → PagedAttention 将 KV 按 block 管理 → 每个 decode step 需从 GPU VRAM 加载全部 pages → CPU/GPU 间通过 LRU 策略交换（无 query-aware 评分）。当序列长度增长到 128K 时，KV cache > 24GB，超出单 GPU HBM 容量。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FlashAttention kernel 执行 exact attention，需完整加载全量 KV → KV 加载成为 compute-bound 外的额外瓶颈。GPU SM 大量时间花在等待 HBM→SRAM 的 KV 数据传输上，Compute Utilization 不足。
    - **硬件架构层**：multi-GPU 节点（如 8×H100），NVLink/NVSwitch 互联。跨 GPU 的 KV cache 同步需通过 All-Gather 或 RingAttention 通信，O(BLhE) 的 KV 数据在 GPU 间交换。通信延迟在 autoregressive decoding 中累积。
    - **Baseline 核心缺陷**：(a) KV cache 全量存储——内存与 expert 数 E 和序列长度 L 线性相关，E=16, L=128K 时 >24GB；(b) 全量注意力——计算与 E 成正比，大部分 expert 的 KV 与当前 query 无关却被加载；(c) 静态驱逐——无法区分高价值 token（heavy hitters）与低价值 token；(d) 未压缩——全精度存储浪费带宽；(e) 同构架构——所有 KV 管理在 GPU 上，metadata 开销占用 GPU 算力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：PiKV 通过四个协同模块将 KV cache 从"全量静态存储"升级为"稀疏动态检索系统"：(1) Expert-Sharded Storage → 内存 O(EL) → O(L/G+KS)；(2) PiKV Routing → 注意力 O(BLhE) → O(BLhk)；(3) PiKV Compression → 带宽 ρ× 降低；(4) PiKV Scheduling → query-aware 自适应驱逐。可选 FPGA offload（§3.5）。
  - 全栈执行例子（PiKV Enhanced, MoE LLM, multi-GPU + optional FPGA）：
    - **算法层（解决"全量注意力"缺陷）**：
      - PiKV Routing 将 KV 查询空间从 E 个 experts 缩至 k 个：C_sparse = BLhk ≪ C_dense = BLhE，理论加速 E/k。
      - 支持 7 种路由策略：Base hash (O(1))、TopK softmax (O(E log k))、Load-Balanced TopK (O(E))、Cache-Aware (含 miss penalty, O(E))、Entropy-Penalized LB (O(E))、RL-Adaptive (learned, O(k²))、Hierarchical coarse→fine (O(E + k log k))。
      - Cache-Aware Router (R_P) 的 penalty term -λ log(1+miss_e) 使 routing 决策倾向于 KV cache 命中率高的 expert，减少不必要的 KV miss 和重加载。
      - **对比 baseline**：baseline 的 attention 需要加载所有 E 个 expert 的 KV；PiKV 仅加载 k 个。
    - **系统框架层（解决"全量存储+静态驱逐"缺陷）**：
      - Expert-Sharded Storage：hash s(t,e) = (t mod N_tok) ⊕ (e mod N_exp) 将 KV 按 expert 和 token 分片到不同 GPU。每个 GPU 仅存储 O(L/G + L/E) tokens，而非 O(EL)。
      - PagedKVCache 三级存储（GPU VRAM → CPU DRAM → SSD）：hot pages 留在 GPU，warm pages 在 CPU，cold pages 在 SSD。
      - DistributedKVCachePool：RDMA-based 跨节点 cache 池化，自动 load balancing。
      - CacheAwarePrefillScheduler：在 TTFT SLO 约束下优化 prefill 阶段的 cache 复用。
      - LoadBalanceDecodingScheduler：在 TBT SLO 约束下最大化 decoding 吞吐。
      - **对比 baseline**：baseline 的 vLLM PagedAttention 仅 CPU/GPU 二级 + LRU 驱逐；PiKV 三级 + expert sharding + SLO-aware scheduling。
    - **编译框架层**：论文未明确说明。vLLM 的 Triton kernel 编译不受 PiKV 修改影响。
    - **kernel调度层（解决"未压缩+metadata开销"缺陷）**：
      - PiKV Compression：支持 8 种压缩方案，压缩比 ρ=1.0-4.0×。在 KV 写入时执行 C(K,V)=(K̂,V̂)∈R^d'×2，读取时执行解压 D(K̂,V̂)。LoRA 方案：K̂ = W_d W_u K（rank-r matvec），K 重构 = K̂ + W_d W_u K̂ + b。Pipeline 中 T_step(ρ) = (dkB/ρ)(2/β + η/γ)，Speedup(ρ1→ρ2) = ρ2/ρ1。
      - Optional FPGA offload：metadata-heavy routing/compression/scheduling 在 FPGA 上执行，GPU 仅接收打包好的 {(K̂,V̂,idx)}_i∈P_t。
      - **对比 baseline**：baseline 无压缩，FP16 全精度存储和传输；PiKV 压缩减少 HBM 带宽和 PCIe 传输量。
    - **硬件架构层（解决"同构架构"缺陷）**：
      - GPU+FPGA 异构（§3.5）：FPGA SmartNIC (Alveo U55C) 通过 CXL Type-3 链接 disaggregated DDR 内存。GPU 仅执行 encoding + attention 的核心计算。
      - 32B MMIO command queue (AXI-Lite) → PiKV-CTRL → routing/compression/scheduling engines 并行执行。
      - On-chip 资源约 224 KB（BRAM_Γ + BRAM_meta + URAM_W），单 U55C SLR 内。
      - T_fpga = T_route + k(T_Γ + K(T_ddr + T_codec))，B_step ≈ (2kd'|P_t|/ρ_link) + k log E。
      - **对比 baseline**：baseline 纯 GPU 架构，KV 管理 metadata 开销与 attention 计算竞争 GPU SM；PiKV-FPGA 将 metadata 卸载，GPU 专注计算。
  - 解决 Baseline 缺陷的方式总结：
    1. **KV 内存过大**：Expert sharding (O(EL)→O(L/G+L/E)) + 压缩 (ρ× 降低) + 调度 (仅保留 K 个 pages)。
    2. **注意力计算浪费**：稀疏路由 (E→k experts per query)。
    3. **静态驱逐误删高价值 token**：Query-aware utility scoring (u_i = Σ α_j φ_j) + 自适应阈值 (θ ← θ + γ(η*-η))。
    4. **带宽瓶颈**：压缩 (d→d/ρ) + FPGA offload (metadata 与 payload 分离)。
    5. **同构架构**：GPU+FPGA 异构，KV metadata 在 FPGA，KV payload 在 CXL-attached DDR，GPU 仅接收精炼后的 KV 子集。

## PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch

- baseline方法是什么？
  - Baseline 是标准 Expert Parallelism (EP) 下的 MoE 训练系统，包括 DeepSpeed、FasterMoE、Megablocks、Tutel、Janus。其核心执行模式为：每个 MoE layer 中 token 通过 gate network 路由后，执行两次 All-to-All 通信（dispatch token 到 remote expert 所在 worker，combine 计算结果回原 worker），通信占单层总时间的 50%-60%。FasterMoE 通过 shadowing/replicating 热门 expert 缓解负载不均，但 expert 参数的 periodic broadcasting 可能抵消 token 传输减少的收益。Janus 尝试在训练前优化 fetch expert vs send token 的决策，但当 expert 参数或 token 数据一方持续占优时失效。这些方法的共同缺陷是：(a) coarse-grained expert scheduling 与 All-to-All 通信同阶段执行，无法消除 All-to-All 瓶颈；(b) 仅支持 push-only 或 pull-only 范式，无法根据 token 分布动态选择最优数据传输方式；(c) backward pass 中 All-Reduce 和 All-to-All 争抢网络带宽，导致 computation blockage。
  - 全栈执行例子（Baseline: FasterMoE + MoE-GPT, ep=8, 8×RTX 4090, OpenWebText, top-k gating）：
    - **算法层**：Gating network（GShard 或 naive top-k）对每个 token 执行 Softmax(LinearGate(X)) → TopK(k=1/2) 选择 expert → 第一次 All-to-All（所有 worker 间全交换 dispatch token 到目标 expert 所在 GPU） → per-expert FFN 计算（两个 linear layer，GeLU 激活，H×αH → αH×H） → 第二次 All-to-All（combine expert 输出回原 worker）。FasterMoE 额外在所有 worker 上 shadow/replicate 热门 expert，通过 periodic broadcast 同步 expert 参数。All-to-All 占总时间 56%（单层 16 expert, batch 16）。
    - **系统框架层**：DeepSpeed-MoE / Megatron-LM 管理 EP（expert parallelism）+ DP（data parallelism）。各 worker 持有部分 expert，非 MoE 层（Attention）在 DP 组内 replicated。Communication backend 为 NCCL。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：PyTorch standard communication primitives（torch.distributed.all_to_all for token dispatch/combine），cuBLAS for expert FFN GEMM。两次 All-to-All 同步阻塞：expert computation 必须等所有 token 到达后才开始，combine 后必须等所有结果收集完才继续。热门 expert 承载更多 token 导致 compute skew，同时其所在 worker 的 network ingress/egress 量最大形成 network skew。非 MoE 层的计算期间 network link 完全 idle。
    - **硬件架构层**：Cluster A——2 节点 × 4 RTX 4090 24GB，节点间 100Gbps InfiniBand，节点内 PCIe 互联。All-to-All 通信走 InfiniBand（远慢于节点内），expert FFN 计算走 GPU SM。baseline 中 idle 链路（非 MoE 计算期间 InfiniBand 空闲）未被利用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 PopFetcher，通过 popularity-based expert-wise prefetching + hybrid push-pull + backward stream scheduling 三个核心机制，将 MoE 训练从"reactive scheduling（等 token 到达再调度 expert）"转变为"proactive preparation（预测热门 expert 并提前预取）"。
  - 全栈执行例子（PopFetcher, MoE-GPT, ep=8, 8×RTX 4090, sliding window s=10, OpenWebText）：
    - **算法层（预测 + 预取 — 解决"无法提前获知 expert 热度"缺陷）**：
      - Sliding-window popularity prediction：在每次 forward pass 中，routing information collector 记录各 token 在各 MoE layer 的 expert 选择 → 滑动窗口 s=10 iterations 内统计各 expert 的 token 分配比例 p_seq → 利用 expert 层间相关性，计算条件概率 Pr(E^{h,j+1}|E^{i,j}) = (1/M) Σ Pr(E^{h,j+1}|E^{i,j}, T_m) → 预测下一层 expert 流行度 p(E^{h,j+1}) = Σ Pr(E^{h,j+1}|E^{i,j}) p_seq^{i,j}（Eq. 3）。该计算在 CPU 异步执行，不影响 GPU training。对比 baseline 的 gate network 只能"事后知道" expert 分布——等当前层 router 执行完才知道哪些 expert 被选中。
      - Hybrid push-pull paradigm：不固定使用 push token 或 pull expert，而是根据公式对比——当 token 传输量 > 2048 tokens 时 pull expert（约 16MB for H=1024），否则 push token。对比 baseline FasterMoE 的 expert-only shadowing 和 Janus 的 pull-only 范式。
      - Expert prefetching decision formulation：建立 end-to-end training latency 模型 Lat_w^{prefetch}（Eq. 6），包含 forward pass computation time（local + prefetched expert）、backward pass computation time（×2）、token transfer time（仅未预取 expert 的 token 需 All-to-All）、gradient reduction time（prefetched expert 需额外 All-Reduce）。目标为 min max_w Lat_w^{prefetch}（Eq. 7）。
    - **系统框架层（预取决策 — 解决"coarse-grained scheduling 无法最优选择预取 experts"缺陷）**：
      - Expert prefetch pruning：两重约束剪枝搜索空间——(a) GPU memory limitation: 2αH² Σδ_{n,w}^i ≤ Mem_w^{free}（Eq. 8）；(b) Transfer time constraint: 2αH² Σδ_{n,w}^i/W_{n,w} ≤ Time^{non-MoE}（Eq. 9）。预取仅在计算-带宽比 ε = P_w/W_{n,w} > 3αH 时有效（Eq. 12-13），如 B200 + NVLink 400Gb/s 场景。被预取 expert 需满足 B_{n,w}^i > εαH / 2(ε-3αH)。按 popularity 排序预取 expert 直到 GPU memory 满。
      - Internal expert sharing via CPU memory：节点内 server-level cache manager 用 CPU memory 缓存已预取的 remote expert 参数 → 同节点其他 GPU 可直接从 CPU memory 读取，避免重复从 remote 拉取。优先通过 NVLink（1800GB/s）节点内检索，再由 GDR NIC（400Gb/s）跨节点拉取。
      对比 baseline 的 expert shadowing（无 memory-aware 决策）和 Janus（OOM on limited GPU memory）。
    - **编译框架层**：论文未明确说明。PopFetcher 实现为 PyTorch plugin（torch.autograd.Function 自定义 MoE operator），可集成到 Megatron-LM。
    - **kernel调度层（异步预取 + 流调度 — 解决"All-to-All 占关键路径"和"backward stream 争抢"缺陷）**：
      - Asynchronous prefetch execution：asynchronous scheduling executor 在 Attention 层（非 MoE 计算）期间，通过独立 CUDA stream 从 remote GPU 拉取已决策的 expert 参数 → 预取与当前层计算完全重叠，zero additional overhead on critical path。已预取到本地的 expert 的 token 直接本地计算——消除这部分 token 的 All-to-All dispatch/combine。
      - Stream pipelining in backward pass：将 All-to-All（token 回传）和 All-Reduce（prefetched expert gradient 聚合）分解为 micro-operations 交错流水线执行 → All-to-All 优先级高于 All-Reduce → 避免 All-Reduce 阻塞 All-to-All 导致 backward computation 等待。对比 baseline 三种通信（EP All-to-All + non-MoE All-Reduce + prefetched expert All-Reduce）启动三个独立 CUDA stream 并发时无优先级控制，network contention 导致 All-to-All 被延迟。
    - **硬件架构层**：同一 Cluster A/B 硬件。核心变化：baseline 中非 MoE 计算期间 InfiniBand/NIC 完全 idle → PopFetcher 利用 idle link 预取 expert 参数；baseline 中 All-to-All 占总时间 56% → PopFetcher 通过 token transfer 减少 14.85%（MoE-GPT）和 13.46%（MoE-BERT），GPU workload balance 提升（轻/重 worker token 差异减少 43.1% MoE-GPT, 57.1% MoE-BERT）；baseline 中 Janus 因 pull all experts 导致 OOM → PopFetcher 通过 pruning 约束内存，可训练模型尺寸比 FasterMoE 大 12.3%-20.1%，比 Janus 大 49.0%-58.2%。
  - 解决 Baseline 缺陷的方式总结：
    1. **All-to-All 通信瓶颈（占单层 50-60% 时间）**：通过 popularity prediction + asynchronous prefetching，在非 MoE 计算期间提前将热门 expert 参数拉到本地，使原本需 All-to-All dispatch 的 token 变为本地计算，减少 token 传输量 13-15%。
    2. **Coarse-grained expert scheduling（push-only 或 pull-only)**: Hybrid push-pull 根据 token 体积 vs expert 参数体积动态选择最优传输方式，当 token > 2048（H=1024）时 pull expert 否则 push token。
    3. **Backward pass network contention**: Stream pipelining 将 All-to-All 优先级置于 All-Reduce 之上，交错执行 micro-operations，减少 backward computation blockage 10-11%。
    4. **GPU memory 不足以 pull all experts**: Pruning strategy 基于 GPU memory capacity + transfer time budget 约束，优先预取 popularity 最高的 expert，middle-to-late training 阶段可固定预取方案或降低 replanning 频率。

## Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models

- baseline方法是什么？
  - Baseline 是两种代表性 MoE 训练框架：(1) DeepSpeed-MoE (Microsoft)——纯 Expert Parallelism (EP) 训练框架，无 load balancing 优化；(2) FasterMoE——系统性 load balancing 方法（dynamic shadowing），通过动态调整 expert placement 来平衡负载，但引入三个核心缺陷：Search（在 runtime 搜索 load balancing 策略，占训练时间 3-7%）、Place（传输 expert 的 parameters 和 gradients 到所有 devices，占 12-16%）、Reduce（将梯度汇聚回原 device，占 12-18%），总 load balancing 开销最高达 37.1%。
  - 全栈执行例子（Baseline: FasterMoE, MoE-GPT-M, 16 GPU, HPWNV 集群）：
    - **算法层**：Gating network 对每个 token 执行 Softmax(LinearGate(X)) → TopK(k=1/2) 路由到 expert。FasterMoE 的 dynamic shadowing 方法：在 runtime 检测各 device 的负载 → search 最优 expert placement → 将 heavy-load expert 的 parameters 传输到 light-load device → 完成后执行 expert computation → 反向传播后将 gradients 聚合回原 device。Baseline 的核心缺陷在算法层：heavy-load expert 的 parameters/gradients 需要在所有 devices 之间全局传输（而非仅必要 device 子集），通信量巨大；且 search 过程本身耗时。
    - **系统框架层**：Expert Parallelism 将 experts 均匀分配到各 device，非 MoE 层（Attention）复制到所有 device。每次 MoE layer 执行：gate → All-to-All dispatch（将 token 按 routing 发送到对应 expert 所在 device） → expert FFN → All-to-All combine（将输出返回原 device）。FasterMoE 额外插入 search→place→reduce 流程。PyTorch Distributed + NCCL backend。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：All-to-All 通信使用 Tutel 的高效 P2P 实现。Expert FFN 计算使用标准 cuBLAS GEMM。关键问题：search/place/reduce 与 expert computation 之间存在数据依赖——必须在 gate 输出后才能 search placement，必须在 placement 确定后才能 place（传输参数），必须在 backward 完成后才能 reduce（聚合梯度）——导致这些操作串行执行无法被隐藏，产生大量 communication idle 和 computation idle。
    - **硬件架构层**：NVIDIA 3090 GPU (24GB) × 16，PCIe 3.0 连接，节点间 100Gb/s Infiniband。关键缺陷：place 阶段将 heavy-load expert 参数传输到所有 device 导致不必要的跨节点 Infiniband 通信；reduce 阶段类似；且这些通信在时间线上与计算串行无法重叠。以 MoE-GPT-M 为例，load balancing 开销占总训练时间 29.2%（含 search 3.2% + place 12.5% + reduce 12.5%）。
  - Baseline 核心缺陷根因（两个）：
    1. **Heavy communication of model states**：FasterMoE 的 expert placement 采用全局传输——heavy-load expert 的参数/gradients 在所有 devices 之间传输（而非仅传输到该 expert 有 input 的 device 子集），导致大量不必要的跨节点通信。
    2. **Poor communication-computation overlapping**：由于数据依赖（必须先 search 才能 place，必须先 backward 才能 reduce），place/search/reduce 操作串行化在关键路径上，无法与 computation 重叠，大量通信和计算时间 idle。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 Pro-Prophet，包含 Planner + Scheduler 两个核心组件。核心 insight 是发现 MoE 训练中相邻迭代的 input distribution 存在 locality（高度相似），使得可以预测未来的分布并据此提前做 scheduling。Planner 解决 heavy communication 问题（通过 lightweight expert placement 仅传输到必要 device 子集 + greedy search 找 communication-efficient placement），Scheduler 解决 poor overlapping 问题（通过 block-wise scheduling 将 load balancing 操作与 computation 重叠）。
  - 全栈执行例子（Pro-Prophet, MoE-GPT-M, 16 GPU, HPWNV 集群）：
    - **算法层（Planner — 解决 heavy communication 缺陷）**：
      - **Lightweight Expert Placement**：每个 expert 独立映射到必要的 device 子集（而非全部 devices）。例如 expert E_0 的 input 分布在 device 0 和 device 1（device 2 没有 E_0 的 input），则 E_0 的 parameters 仅从 device 0 传输到 device 1（不传输到 device 2），gradients 也仅在 device 1 传回 device 0。对比 baseline FasterMoE 将 expert 参数传输到所有 devices。
      - **Performance Model**：抽象公式化 MoE 层执行时间——T'(R,H,s,n) = 4·T_A2A(R) + 3·T_FEC(H) + T_Trans(s,n) + T_Agg(s,n)。mean estimation error <5%。
      - **Locality-based Greedy Algorithm**：利用 locality 减少 search 频率（用户可调节）。算法在 runtime 贪心迭代——每次选择当前负载最重的 device 上的 expert，将其 parameters 传输到持有该 expert 最多 input 的 devices 子集——每次迭代用 performance model 评估 placement，直到负载满足 max(H) - min(H) < α·I/E 的平衡条件。因搜索空间为 2^(N·E) 的 brute-force 不可行，greedy 策略使 search 可行性成立。
      - 对比 baseline：FasterMoE 的 search 是全局传输策略（heavy-load expert → 所有 devices），通信量 O(D·size(expert))；Pro-Prophet planner 的 lightweight placement 仅传输到必要 devices 子集，通信量 O((D-n)·size(expert))，其中 n 是不必要的 device 数——当 n 大时通信量显著降低。
    - **系统框架层（Scheduler — 解决 poor overlapping 缺陷）**：
      - **Scheduling Space 建立**：定义每个 MoE block 内的操作类型（comm vs comp）和数据依赖。利用 locality 提前预测 iteration j+1 的 input distribution → Plan_{i}^{j+1}（决定 j+1 迭代的 placement）可在 iteration j 的 A2A 通信中执行。Trans 原语（传输 expert 参数）可在同一 iteration 内与 forward computation 重叠。Agg 原语（聚合 gradients）可在同一 iteration 内与 backward computation 重叠。Scheduling 约束：Plan 必须在上一迭代执行（需要上一迭代的 distribution 数据）；Trans 和 Agg 各自限制在单个 iteration 内（因 layer-by-layer 和 concentrated updating 两种参数更新方式兼容性）。
      - **Block-wise Scheduling Strategy**：以 MoE block 为单位进行 sub-operator 级调度。将 Trans 原语拆分为 2 个 sub-operators，分别与同一 block 的 FEC（Forward Expert Computation）和 FNEC（Forward Non-Expert Computation）并行。类似地，Agg 原语与 BEC 和 BNEC 重叠。FNEC 和 BNEC 的时间是静态的（可在训练前估计），用于精确规划 split。Plan 操作的 sub-operators 被调度到前一迭代的 A2A 通信中执行。
      - 对比 baseline：FasterMoE 的 search/place/reduce 串行暴露在关键路径上。Pro-Prophet scheduler 将 Plan（search）隐藏在前一迭代的 A2A 通信中，将 Trans（place）隐藏在 forward computation 中，将 Agg（reduce）隐藏在 backward computation 中——所有 load balancing 开销被计算时间覆盖。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel 不变。关键变化在数据流和执行时序：baseline 中 search→place→reduce 与 compute 串行——gate → search → place (transmit params) → A2A → compute → backward → reduce (aggregate grads)；Pro-Prophet 中 Plan 提前到前迭代执行（与 A2A 并行），Trans 与 FEC+FNEC 并行，Agg 与 BEC+BNEC 并行——gate → A2A (含 Plan_{next}) → [compute FEC || Trans sub-op1] → [compute FNEC || Trans sub-op2] → ... → backward [compute BEC || Agg sub-op1] → [compute BNEC || Agg sub-op2]。
    - **硬件架构层**：同一 NVIDIA 3090 × 16 + Infiniband 硬件。核心变化：baseline 中 Infiniband 链路在 compute 期间 idle（search/place/reduce 独占通信链路但串行执行），Pro-Prophet 下 Infiniband 持续在 compute 期间并行传输 expert 参数和 gradients（Trans/Agg 与 FEC/BEC 重叠），链路利用率提升。另外 locality 减少了 search 频率，进一步降低通信开销。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对 heavy communication of model states**：Lightweight expert placement 将 expert 仅映射到有其 input 的 device 子集（而非全部 devices），Trans 和 Agg 仅在子集内执行。Performance model + greedy algorithm 在 runtime 高效搜索 communication-efficient placement。对比 FasterMoE 全局传输策略，communication volume 因 device 子集缩小而显著降低。
    2. **针对 poor communication-computation overlapping**：Block-wise scheduling 利用 locality 预测未来分布，将 Plan 提前到前迭代的 A2A 中执行，将 Trans 与 forward computation 重叠，将 Agg 与 backward computation 重叠——所有 load balancing 开销被计算隐藏。对比 FasterMoE 串行执行导致 29-37% 的 load balancing 开销，Pro-Prophet 通过重叠将这些开销几乎消除。
    3. **关键数据支撑**：Planner 单独贡献 1.12-1.26x 加速，Scheduler 单独贡献 1.01-1.14x 加速，Full 协作额外贡献 1.02-1.03x 加速。vs FasterMoE 的 load balancing 提升（RB ratio）最高达 11.01x。Performance model estimation error <5%，验证了建模精度。

## QMoE Sub-1-Bit Compression of Trillion-Parameter Models

- baseline方法是什么？
  - Baseline 是 Round-To-Nearest (RTN) 量化——直接将 bfloat16 权重按量化网格进行最近邻舍入，无数据依赖校准。RTN 在 2-bit 精度下尚可运行但 loss 显著增加（base128: 1.73→2.27, c2048: 1.18→1.33），三元精度下几乎崩溃（base128: 4.54）。另一个 baseline 是"不压缩"——c2048 的 bfloat16 推理需 3.2TB 存储，对应 >65 A6000 或 >130 3090 GPU，对于普通硬件完全不可行。
  - 全栈执行例子（Baseline: RTN 三元量化 + 朴素 sparse format, SwitchTransformer-c2048, bfloat16 inference, 需 >65 A6000）：
    - **算法层**：RTN 对每行权重独立量化——w_min = min(W_row), w_max = max(W_row)，三元网格 {w_min, 0, w_max}，每个权重映射到最近的值。无 calibration data 参与，无法补偿跨层误差累积，不做 Hessian 校正。量化后权重具有高稀疏度（~88.6% 零值），但直接使用 CSR/bitmask 等 sparse format 存储时：bitmask 占 1 bit/param（几乎抵消三元 2-bit 表示的压缩），column index 占 10-13 bit/param（反而膨胀）。
    - **系统框架层**：HuggingFace Transformers 加载完整的 bfloat16 checkpoint (3.2TB)，标准 autoregressive decoding 流程。每层 MoE: router → Top-1 expert selection → expert FFN (2×GEMM + GELU) → combine。若无压缩，需通过模型并行（tensor parallelism + expert parallelism）将 expert 分片到 ~65 张 A6000 或 ~130 张 3090 GPU。单 GPU 无法装入完整模型，涉及多机通信开销。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 cuBLAS bfloat16 GEMV kernel。在 uncompressed 场景下接近内存带宽理想利用率（因矩阵较小、访存模式规则）。压缩后的 RTN + sparse format 场景下，由于 metadata 开销（bitmask/column index），全局内存读取量并不比 uncompressed 减少多少，且 sparse kernel 的开销（gather/scatter、非连续访存）显著降低有效带宽。
    - **硬件架构层**：>65×A6000 GPU 组成多机集群，通过 NVLink（节点内）+ Infiniband/网络（节点间）互联。各 rank 持有 expert shards，all-to-all 通信将 token 路由到对应 expert 所在 GPU。多机部署的核心成本在内存（3.2TB 模型权重）和通信（跨机 token dispatch）。
  - Baseline 核心缺陷根因（三个）：
    1. **Post-training 量化精度不足**：RTN 无法突破 ~3 bit/param 的"可用精度墙"——对于 MoE 模型虽比 dense 模型更鲁棒（不崩溃），但三元精度下 loss 增加太大无法实用。根本原因是 RTN 只做逐元素的最近邻映射，不利用 layer-wise Hessian 信息来补偿量化误差的跨层累积。
    2. **Sparse format metadata 开销抵消压缩收益**：三元量化后虽 ~88.6% 权重为零，但存储这些零的位置信息（bitmask 或 column index）比压缩后的 2-bit 权重本身还大，直接使用 sparse representation 无法实现 sub-1-bit 压缩。
    3. **Scaling 瓶颈**：现有 data-dependent 量化方法（如 GPTQ）针对 dense 模型优化——每层 few large matrices 的 GPU 利用率良好，但 MoE 有 1000× 更多的小层，导致现有实现 GPU 利用率极差、内存需求巨大（需存储 >100× 的 calibration data）、可靠性问题（10000+ 层中大概率 hit 数值不稳定 edge case）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 QMoE，通过系统-算法-格式-内核的垂直协同设计，将万亿参数 MoE 压缩到 <1 bit/param 并实现高效推理。核心创新链：(1) Scalable GPTQ for MoEs → (2) 利用自然稀疏的低熵字典编码 → (3) Warp-per-row 字典解码 CUDA kernel。
  - 全栈执行例子（QMoE, SwitchTransformer-c2048, 4×A6000）：
    - **算法层（解决"Post-training 量化精度不足"和"Scaling 瓶颈"缺陷）**：
      - **Scalable GPTQ Adaptation for MoEs**：将 GPTQ 的 layer-wise Hessian-based 量化扩展到 MoE 场景——(a) Activation Offloading：calibration data 的中间激活存于 CPU RAM 的 list buffer 数据结构（大连续 buffer + delimiter indices），GPU 仅按需取小块数据，实现单卡 A6000 处理 160K calibration samples；(b) Expert Grouping：将 16 个 expert 组批处理，权重和 Hessian 堆叠为 3D tensor，batched GPTQ 算法同时压缩组内所有 expert，实现约 6× 加速（vs per-expert 串行）；(c) Lazy Weight Fetching：3.2TB 原始模型权重按需从磁盘加载到 GPU，压缩后写回并释放内存——不必同时加载全部权重到 RAM；(d) Robustness Mods：10× 提高 Hessian dampening (δ=0.1)、对不可逆 Hessian layer 退化为 RTN、token cap 为均值 4× 防 OOM、特殊 token premasking（MLM 的 mask token 从校准数据中排除，因模型对其过于鲁棒、误差补偿无益）。
      - 对比 baseline：RTN 三元精度 c2048 loss 从 1.18→2.15 (+82%)；QMoE (GPTQ) 三元精度 loss 仅 1.18→1.26 (+6.7%)。RTN 等价于 Hessian = Identity（无误差补偿），GPTQ 通过二阶 Hessian 信息逐列校正量化误差的跨层传播。
    - **系统框架层（解决"Scaling 瓶颈"缺陷）**：
      - 基于 PyTorch + HuggingFace Transformers 实现，所有修改通过运行时动态 patch（无需修改官方安装）。修复 HuggingFace 对大 MoE 的两个 bug：(a) 配置和模型设置 fix；(b) 跳过无 token 分配的 expert 的（空）CUDA kernel launch——>10× 加速大模型推理。
      - Compression pipeline 的 CPU-GPU 协同设计：list buffer in CPU RAM + lazy weight fetching from disk → 单 A6000 可在 ~16h 内压缩 c2048（1.6T params）。原始模型仅需加载到磁盘（3.2TB），不要求同时装入 RAM。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层（解决"Sparse format metadata 开销抵消压缩收益"缺陷）**：
      - **Dictionary-Based Sub-1-Bit Encoding**：利用三元量化后 88.6% 自然稀疏度带来的低熵——P(0)≈0.886, P(1)≈P(2)≈0.057——设计 fixed-length codeword → variable-length sequence 的字典编码。算法 1（max-priority queue）生成 2^16 个最高概率的三元对序列（每序列 ≤14 pairs = 28 weights），存入 UINT16→2×UINT32 的字典。编码时不直接存储零的位置信息（bitmask/column index），而是用概率最高的 codeword 更频繁地表示"多个零连续出现"的 pattern——信息论上接近熵编码但不需要变长码字的序列依赖。
      - c2048 实现 20.07× (MoE-only) / 19.81× (full model) → 0.807 bits/param。与理论极限 25.40× 仅差 ~20%，换取快速 GPU 解码。
      - 对比 baseline：Baseline 的 bitmask (1 bit/param) + 2-bit ternary = 3 bits/param，无 net compression；column index (10-13 bits) 更差。QMoE 的字典编码直接压缩到 0.8 bits/param，不存储任何 per-weight metadata。
      - **Custom CUDA Decoding Kernel (Sub1MatVec)**：warp-per-row 并行，每 warp 通过 coalesced load 取 32 个 UINT16 codewords → 查 GPU L2 cache 中的 512KB 字典 → 28/32 threads 并行提取 2-bit ternary 权重 → shared memory dequant lookup table（复制 32× 避免 bank conflict）→ FMA 累加 → warp shuffle reduction。字典按概率降序排列，高频 codeword 自动 L1 cache prefetch。Global memory 读取量仅约 1/20 of bfloat16（0.8 vs 16 bits/param），bit ops 开销被 global memory latency（~200 cycles vs ~1 cycle）完全隐藏。性能上，压缩 kernel 在所有矩阵形状上比 uncompressed cuBLAS bfloat16 GEMV 更快（最高 35% speedup）。
    - **硬件架构层**：4×A6000 (48GB) 或 8×3090 (24GB) 单服务器。核心变化：baseline 的 c2048 bfloat16 推理需 >65 A6000 / >130 3090 GPU（多机集群），QMoE 压缩后仅需 4 或 8 卡（单服务器）。每 GPU 约 40GB compressed model weights (160GB total / 4)。Uncompressed baseline 因 GPU 数量多、跨机通信开销大；QMoE 在单服务器内 GPU 间 NVLink 互联、无可感知通信瓶颈。端到端推理仅 <5% 开销 vs 理想化 uncompressed baseline（该 baseline 估计将所有 expert 指向同一权重数据来回避内存限制——实际部署需 20× 更多 GPU 及对应通信开销，因此 <5% 是下界估计）。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"RTN 精度不足"**：QMoE 将 GPTQ 的 data-dependent Hessian-based 量化适配到 MoE 场景（通过 expert grouping 批量化 + activation offloading + robustness mods），三元精度下 loss 增加仅 6.7%（vs RTN 的 82%）。关键 insight：GPTQ 用每层 Hessian 矩阵的二阶信息逐列校正量化残差，RTN 等价于 Hessian=Identity（无校正）。
    2. **针对"sparse format metadata 开销"**：不直接使用 sparse format，而是通过字典编码利用低熵（而非直接 sparsity）实现压缩——高概率 codeword 编码"频繁出现的零模式"，不存储 per-weight 位置信息。字典优化为 c2048 权重分布，c2048 压缩率 20.07×，与独立同分布模型仅差 ~5%。
    3. **针对"Scaling 瓶颈（GPU 利用率差、内存需求大、可靠性差）"**：Activation offloading（list buffer + CPU RAM）+ Lazy weight fetching（磁盘按需加载）+ Expert grouping（16 expert batch = 6× 加速）+ Robustness mods（dampening、fallback RTN、token cap、premasking）→ 实现单卡 A6000 在 <1 天内压缩 c2048 (1.6T params)。
    4. **针对"压缩后解码效率"**：Dictionary code 的 fixed-length codewords + warp-per-row 并行 + L2-cache resident 字典 + shared memory dequant + coalesced access → 解码开销被 global memory latency 完全隐藏，压缩 kernel 比 uncompressed cuBLAS 更快。

## ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts

- baseline方法是什么？
  - Baseline 是标准 TopK Routing MoE，采用 layer-local routing 机制——每层的 router 仅能从本层的 N 个 expert pool 中选择激活的 experts。核心计算公式：h' = Σ_{i=1}^{N} g_i · E_i(h)，其中 g_i 通过 Softmax(W_gate · h) 后 TopK 选择确定。该架构的根本限制是 expert 维度（每个 expert 的 hidden dimension）受 per-layer 参数预算约束——要在固定总参数量下平衡 expert 数量（粒度）和 expert 容量（hidden dim）。增加 expert 数量（fine-grained MoE）可丰富 expert 组合的灵活性，但减少每个 expert 的 hidden dim 会降低其表达容量；反之保持 expert dim 而增加 expert 数量则膨胀总参数量。这种 trade-off 是 MoE 架构设计的核心矛盾。以 DeepSeek-MoE、Qwen3、Kimi-K2 为代表的 fine-grained MoE 趋势（128-384 experts）选择了"更多更小的 expert"路径，但牺牲了单个 expert 的容量。
  - 全栈执行例子（Baseline: vanilla MoE-2.3BA0.3B，TopK routing，4 nodes × 32 Hopper GPUs）：
    - **算法层**：L 层 MoE Transformer，每层有 N=64 个 experts，每层独立 router W_gate^l ∈ R^{N×d} 执行 Softmax + TopK 选择。第 l 层 router 仅从 E^l = {E_1^l, ..., E_N^l} 中选择——无法访问其他层的 experts。每个 expert 为独立 FFN，参数固定于其所在层。当采用 fine-grained 设计（更多 experts/层）时，受总参数量约束，每个 expert 的 intermediate_size 必须减小，降低单个 expert 的表达能力。
    - **系统框架层**：Megatron-LM 分布式训练框架。Expert Parallelism (EP)=8 将 experts 分布到 8 个 GPU。每层 MoE forward 执行 All-to-All dispatch（token 按 router 结果发送到对应 expert 所在 GPU）→ local expert FFN computation → All-to-All combine（收集 expert 输出）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。Megatron-LM 的 MoE kernel 包括 gating（Linear + Softmax + TopK）、All-to-All 通信、per-expert FFN（SwiGLU: W_down @ SiLU(W_gate @ x) ⊙ W_up @ x）。
    - **硬件架构层**：4 nodes × 32 Hopper GPUs。每层 64 experts × ~32 层 = 2048 个 FFN 模块。Expert Reuse 为 0（每层 experts 独立不共享）。
  - Baseline 核心缺陷根因：**Layer-local routing 将 expert 组合的灵活性（expert 数量）与单个 expert 的表达容量（hidden dim）绑定在 per-layer 预算上**——无法在不牺牲 expert 容量或不增加总参数的前提下扩大路由空间。Fine-grained MoE 增加了 expert 数量但降低了每个 expert 的 capacity；DeepSeek-MoE 的 shared expert 缓解了部分问题但仍是 layer-local 方案。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 REXMOE，通过**允许 router 跨相邻层复用 experts**来解耦 expert 维度与 per-layer 参数预算。核心创新：(1) 跨层 Expert Reuse——将 r 个相邻层的 expert pool 合并为扩展候选池 U = ∪_{i∈G} E^i，使每层 router 可访问 r×N 个 experts 但无需增加任何 expert 参数；(2) Progressive Scaling Routing (PSR)——训练期间从 N 逐步扩展至 rN 候选 experts，遵循 curriculum learning 避免训练初期的负载崩溃。
  - 全栈执行例子（REXMOE-R4, MoE-2.3BA0.3B, 4 nodes × 32 Hopper GPUs）：
    - **算法层（解决"layer-local routing 绑定 expert 容量与粒度"的缺陷）**：
      - Cross-layer Expert Reuse：r=4 层连续层分为 group G，每层 router 可访问 4×64=256 个 candidates——但物理上仅存在 64 个 experts（因其来自相邻 4 层，每层 64 个共享）。不需要创建新 experts，不增加任何 expert FFN 参数。仅 router W_gate ∈ R^{rN×d}（256×512）相比 baseline（64×512）增加 4× router 参数，占模型总参数 <1%。
      - PSR 训练策略：t=0~10k steps 仅路由到 N=64 local experts（退化为 baseline）→ t=10k~30k 线性扩展至 256 candidates → t>30k 全量 256 candidates。
      - 结果：Avg Acc 从 49.15% 提升至 50.23%（+1.08 pts），WikiText PPL 从 21.19 降至 20.73。仅 Expert Reuse 贡献 +0.13% acc（边际），PSR 贡献 +1.05%（关键）。
      - 对比 baseline：baseline 中每层独立 64 experts 的路由多样性受限于 layer-local 64 选 TopK；REXMOE 通过跨层复用将每层的路由空间扩大为 256 选 TopK（4× 更多组合），不增加任何 expert FFN 参数。这打破了"更多组合必伴随更小 expert 或更多总参数"的 trade-off——expert 的 hidden dim 保持不变（intermediate=744），但路由组合多样性从 (64 choose TopK) 扩展到 (256 choose TopK)。
    - **系统框架层**：
      - 修改 Megatron-LM 的 MoE Block 和 TopK Router 实现：在每层 MoE forward 时，从相邻 r 层收集 expert 参数引用（不复制）组成扩展候选池。PSR 的 masking 在 gating score 计算后、TopK 选择前执行。
      - 论文未明确说明跨层 expert 参数的 EP 分布变化。推测相邻层的 experts 参数需要通过跨 GPU 通信获取引用，但论文声称 overhead 可忽略。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。PSR 的 expert masking 是简单的 gating score 置零 + TopK 选择，不涉及新 kernel。
    - **硬件架构层**：同一 4×32 Hopper GPU 集群。核心变化：baseline 每层 router 仅激活本地层的 N 个 experts → REXMOE 每层 router 可激活来自相邻 r 层的 experts，但物理参数不增加，因此 GPU memory 和 compute 中 expert FFN 部分不变。唯一 overhead 来自 router 的 r× 权重增加和跨层 expert 参数引用——论文声称可忽略。
    - **定性分析收益**：Layer-wise expert activation ratio 可视化（Figure 5）显示 REX-SE-R2 相比 Base-MoE-SE 展现出更强的 task-specific specialization——同一 expert 在不同任务（SciQ/LogiQA/WinoGrande）上呈现明显差异化的激活模式，暗示扩展候选池使模型的 expert ensemble 效应对不同任务自适应。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"layer-local routing 绑定 expert 容量与粒度"**：REXMOE 通过跨层 expert reuse 将路由空间从 N 扩展到 rN，不增加任何 expert FFN 参数——仅增加 <1% 的 router 参数。这直接打破了"更多路由组合 = 更小 expert 或更多参数"的 trade-off。
    2. **针对"训练稳定性（直接跨层 routing 导致负载崩溃）"**：PSR 策略通过 curriculum learning 从 local-only routing 逐步扩展到 full cross-layer routing，使模型在训练早期建立稳定 routing 后逐步适应更丰富的 expert 组合。消融实验（Table 4）证明 PSR 是 critical component：仅 Expert Reuse 仅 +0.13% avg acc，加入 PSR 后 +1.05%。
    3. **针对"大 r 值的负载不均衡崩溃"**：通过 LBV 和 under-utilized experts ratio 的 ablation（Figure 3），论文发现 r=2~4 是最优平衡点——更大的 r（16/32）导致 expert 负载严重不均衡和大量 expert 几乎不被激活的崩溃现象。这为实际部署提供了 r 值选择的指导（r 不宜过大）。
    4. **通用性和兼容性**：REXMOE 不改变 MoE 的核心计算语义（仍是 gating + TopK + FFN），可与 shared experts、fine-grained MoE 等其他设计正交组合。推理部署仅需 vLLM 等框架的 minimal adaptation。

## ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models

- baseline方法是什么？
  Baseline方法：Tutel（基于DeepSpeed的分布式MoE训练框架），采用标准expert parallelism + zero-padded all-to-all通信 + 静态expert-to-GPU映射。

  全栈执行例子（Baseline: Tutel on DeepSpeed, 32 GPUs, 32 experts, 1个MoE层forward pass）：
  ```
  Input tokens (batch=512, seq=128, hidden=768) 分布在32 GPUs上
  ├─ 算法层：
  │   └─ Router: G(x)=Softmax(TopK(x·W_g)) → 每个token选择top-k experts
  │   └─ Expert FFN: 对分配给本地GPU的tokens执行标准FFN计算（无压缩/无稀疏/无量化）
  │   └─ 论文未明确说明训练使用的优化器、学习率调度、混合精度等训练超参
  ├─ 系统框架层（DeepSpeed + Tutel）：
  │   ├─ Expert Parallelism: 32 experts 平均分配到 32 GPUs（每GPU 1 expert）
  │   ├─ All-to-All dispatch: 将tokens按expert选择路由到对应GPU
  │   │   ★ 问题1：每个GPU统计本地的per-expert token数 → 取全局max → zero pad到统一size
  │   │   ★ 问题2：expert selection高度不平衡，zero ratio从88%升至98%
  │   │   ★ 问题3：all-to-all通信占端到端延迟的58%-69%（随expert数增加而加剧）
  │   ├─ Expert FFN计算: 各GPU独立执行
  │   │   ★ 问题4：GPU负载不均——处理少量tokens的GPU必须等待最繁忙GPU完成（barrier同步）
  │   └─ All-to-All combine: 将FFN输出返回原始GPU（同样含大量zero padding）
  │   └─ 论文未明确说明数据并行、模型并行、pipeline并行与expert并行的具体组合方式
  ├─ 编译框架层：
  │   └─ 论文未明确说明
  ├─ Kernel调度层：
  │   ├─ NCCL all-to-all collective: GPU间通过NVLink (600 GB/s 节点内) 或 Ultra Ethernet (100 Gbps 节点间) 传输
  │   │   ★ 问题5：拓扑无感知——不区分快慢链路，不区分节点内/节点间带宽差异
  │   │   ★ 问题6：异构网络中（带宽差2×），慢链路拖累全局all-to-all barrier同步
  │   └─ 论文未明确说明是否使用NCCL的alltoallv变体
  ├─ 硬件架构层：
  │   └─ 论文未明确说明
  └─ 芯片设计层：
      └─ 论文未明确说明
  ```
  Baseline缺陷总结：(1) all-to-all通信中大量zero padding导致高通信量（zero ratio 88-98%）；(2) expert selection严重不均衡→GPU利用率低+通信延迟高；(3) 静态expert-to-GPU映射不考虑异构网络拓扑→慢链路成为瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：ScaleMoE——在DeepSpeed上实现三项运行时优化：(1) Adaptive All-to-All Communication消除zero padding；(2) Dynamic Expert Clustering通过K-means聚类和expert复制重新均衡负载；(3) Topology-aware Expert Remapping利用遗传算法在异构网络中优化expert放置。所有优化保持模型计算语义不变，不修改router或expert FFN。

  全栈执行例子（ScaleMoE, 同场景32 GPUs, 32 experts, 1个MoE层forward pass）：
  ```
  Input tokens (batch=512, seq=128, hidden=768) 分布在32 GPUs上
  ├─ 算法层（与Baseline完全相同，保持训练精度）：
  │   ├─ Router: G(x)=Softmax(TopK(x·W_g)) —— 不修改
  │   └─ Expert FFN: 标准计算 —— 不修改
  │   └─ Expert Replication: 热门expert在多个GPU上复制副本（最多31 replicas）
  │       解决→ 增加local GPU HBM access从3.28%到61.32%，减少远程通信
  │   └─ Unpopular Expert Offload: 冷门expert移到host pinned memory
  │       解决→ 释放GPU内存给热门expert replicas，miss rate仅~1%
  ├─ 系统框架层（DeepSpeed + ScaleMoE）：
  │   ├─ Expert Parallelism + Dynamic Expert Clustering:
  │   │   ├─ Profiling: 每个token记录<batchID,seqID,tokenIdx,tokenName> + expert选择历史
  │   │   ├─ K-means Clustering: token按expert选择模式聚类（距离=序列长-重叠expert数）
  │   │   ├─ Expert Redistribution: 按聚类结果更新expert-to-GPU映射
  │   │   └─ 解决→ 聚类后同cluster tokens共享expert偏好→减少跨GPU通信
  │   ├─ Adaptive All-to-All dispatch:
  │   │   ├─ 监控: 每个GPU统计per-expert选择计数
  │   │   ├─ All-gather: 32 GPUs交换计数（overhead 44.50ms，可忽略 vs GB级zero传输）
  │   │   ├─ Slice计算: 精确的input/output slice sizes（无需zero padding）
  │   │   └─ NCCL alltoallv: 仅传输有效数据
  │   │   解决→ zero padding 消除→通信量减少up to 81%
  │   ├─ Expert FFN计算: 各GPU对本地experts（含replicas）执行计算
  │   │   解决→ 通过clustering+replication减少负载不均衡
  │   └─ Adaptive All-to-All combine: 精确slice size返回output
  ├─ 编译框架层：
  │   └─ 论文未明确说明
  ├─ Kernel调度层（运行时优化核心）：
  │   ├─ Topology-aware Expert Remapping:
  │   │   ├─ Coverage Matrix (C×C): cluster i 对 cluster j 的expert覆盖度
  │   │   ├─ Bandwidth Matrix (GPU×GPU): 点对点网络带宽（含NVLink/Ultra Ethernet差异）
  │   │   ├─ Genetic Algorithm: 搜索最优 cluster→GPU 映射向量 SV
  │   │   │   Fitness = Σ_{i,j} ((b·s - CM[SV[i]][SV[j]]·h) / BM[i][j])
  │   │   │   每代: uniform order-based crossover + swap mutation
  │   │   └─ 解决→ 高覆盖cluster对放在高带宽GPU对上，低覆盖对放在低带宽对上
  │   ├─ CPU-GPU Overlapping: clustering+remapping（CPU）与 iteration（GPU）overlap
  │   │   解决→ clustering overhead从12.48%降至0.001%
  │   └─ 通信路径: NCCL + NVLink (节点内) + Ultra Ethernet (节点间)
  │       解决→ 异构网络中speedup高达3.31×（vs homogeneous 1.84×）
  ├─ 硬件架构层：
  │   └─ 论文未明确说明
  └─ 芯片设计层：
      └─ 论文未明确说明
  ```

  - 解决 Baseline 缺陷的方式总结：
    1. **针对"all-to-all zero padding通信膨胀"**：Adaptive All-to-All通过all-gather聚合per-expert选择计数→精确slice size→NCCL alltoallv仅传输有效数据，消除88-98%的zero传输。all-to-all通信开销减少up to 81%。
    2. **针对"expert selection负载不均衡"**：Dynamic Expert Clustering使用K-means聚类token（基于expert选择模式相似度）+ 热门expert复制（最多31 replicas，local access从3.28%→61.32%）+ 冷门expert offload。聚类结果驱动expert-to-GPU重映射，减少跨设备通信。
    3. **针对"异构网络拓扑无感知"**：Topology-aware Expert Remapping构建coverage matrix + bandwidth matrix，使用遗传算法搜索近最优cluster-to-GPU映射，在异构网络中实现高达3.31× speedup（vs homogeneous 1.84×）。
    4. **保持训练正确性**：ScaleMoE不修改router或expert计算语义，保持token的<sequenceID, tokenIndex, tokenName>信息以保证顺序；replicated expert在backward pass后正确更新梯度；可与其他MoE优化正交组合。

## Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training

- baseline方法是什么？
  Baseline 方法是现有的 MoE 训练框架：(1) **Fairseq GShard**：使用 expert parallelism 将所有 expert 参数加载到 GPU 显存，通过 batched matrix multiplication + dispatch mask 执行 MoE 计算，随 expert 数量增加需要更多 GPU；(2) **Tutel**：MoE 特化框架，同样将所有 expert 驻留在 GPU 显存，使用 batched GEMM 和 zero-padding 对齐批量大小；(3) **ZeRO-Offload^E**（论文自行实现的 expert-wise offloading 版本）：将 expert 参数 offload 到 CPU 内存，但采用 naive 的层级别 pipelining 和 model-level CPU optimizer，无动态负载均衡。Baseline 的核心问题：(a) GPU 内存受限——expert 增加需要更多 GPU（MoE-L 8 experts 需 4×A100，128 experts 需 52×A100）；(b) dispatch mask 内存爆炸——batched matrix multiplication 需要 (tokens after padding)×(tokens) 的巨大映射表（如 MoE-L batch 32 需 48 GiB）；(c) zero-padding 浪费——expert 越多 token 负载越不均衡（32 experts 时 39% 零填充），GPU 利用率低；(d) CPU optimizer 慢——layer-wise CPU Adam 比 GPU Adam 慢 31×，造成 GPU 空转。

  全栈执行例子（Baseline: Fairseq GShard, MoE-L 32 experts, 4×A100 40GB）：
  ```
  # 算法层：标准 MoE，GPT-based decoder layer
  # - Gate: softmax(W_gate @ x) → Top-1 expert selection
  # - Expert FFN: gate_proj → SiLU ⊙ up_proj → down_proj
  # - 使用 batched matrix multiplication（所有 expert 同时在 GPU 上）
  
  # 系统框架层：Fairseq，expert parallelism + batched GEMM
  # - 所有 32 experts 的 params + optimizer states 常驻 GPU HBM
  # - 创建 dispatch mask [N_tokens_padded, N_tokens]
  # - All-to-All scatter tokens → experts → All-to-All gather
  # - 限制：MoE-L 32 experts + 4 GPUs → OOM（batch=1 都不够）
  
  # 编译框架层：PyTorch JIT / cuBLAS backend（论文未明确说明）
  
  # kernel调度层：
  # - cuBLAS batched GEMM 执行多个 expert 的 FFN 计算
  # - GPU-side Adam optimizer
  # - zero-padding 导致 39% 的无效计算
  # - 无 CPU-GPU 通信重叠
  
  # 硬件架构层：4×A100 40GB, PCIe 4.0, NVLink 600GB/s
  # - GPU HBM 容量成瓶颈，扩容专家需加 GPU
  # - PCIe 带宽未利用于 offloading
  ```
  Baseline 缺陷根因：(1) 所有 expert 必须同时驻留在 GPU，将 MoE 的"计算-参数量解耦"特性与系统内存解耦割裂；(2) batched GEMM 强制使用 dispatch mask 和 zero-padding；(3) layer-wise optimization 串行化 GPU 和 CPU 任务；(4) 静态 expert placement 无法适应 per-batch token 分布变化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**ES-MoE**——通过 expert 级别 offload + pipelined processing + dynamic placement 三项核心设计，使 MoE 训练规模不再受 GPU 显存限制。

  全栈执行例子（ES-MoE, MoE-L 32 experts, 4×A100 40GB）：
  ```
  # 算法层：标准 MoE，数学等价于 baseline（不做模型修改）
  # - Gate: softmax(W_gate @ x) → Top-1 expert selection（不变）
  # - Expert FFN: sequential processing（替代 batched GEMM）
  
  # 系统框架层：ES-MoE on Fairseq
  # - Expert params + optimizer states offloaded to CPU RAM / SSD
  # - GPU 仅保留：non-expert params + 当前处理 expert + activations
  # - 不使用 dispatch mask（零内存开销）
  # - Sequential expert processing → 支持 8× 更大 microbatch
  
  # kernel调度层（核心创新）：
  # Forward per MoE Block:
  # 1. Gate Network (GPU) → per-token expert selection
  # 2. Dynamic Expert Placement (CPU, <2.69μs):
  #    - Greedy scheduling: sort experts by (upload_time + compute_time)
  #    - Assign each expert → GPU with min accumulated load
  #    - 消除 zero-padding：expert 按实际 token 数分配到 GPU
  # 3. Token Permutation (GPU)：
  #    【同时：pipeline 上传第 1 个 expert, CPU→GPU via PCIe】
  # 4. Expert Processing Loop:
  #    for expert in assigned_experts:
  #      [Expert N 计算] || [Expert N+1 上传, CPU→GPU]
  #    — 计算与通信完全重叠
  # Backward per MoE Block:
  # 5. Expert FFN backward (GPU) → per-expert gradients
  # 6. Expert-wise CPU Optimizer：
  #    - Expert 0 backward done → 立即启动 CPU Adam
  #    - Expert 0 CPU optimizer || Expert 1 GPU backward
  #    - Expert N CPU optimizer || Layer(N+1) GPU forward
  #    — CPU optimizer 延迟被隐藏于 GPU 计算之后
  
  # 硬件架构层：4×A100 40GB, PCIe 4.0 (confirmed bandwidth ~25 GB/s), NVLink 600GB/s
  # - PCIe 持续用于 expert 上传/下载，与 GPU 计算重叠
  # - SSD offloading 模式下：expert CPU↔SSD 使用 LRU cache + prefetching
  # - DMA-able pinned memory 避免 page fault stall
  # - 3 种自适应模式：
  #   - GPU only: ≤32 experts（全在 GPU 内，仅消除 zero-padding）
  #   - CPU offload: 32-104 experts（offload 到 CPU RAM）
  #   - CPU+SSD offload: >104 experts（LRU cache on CPU, evict to SSD）
  ```

  解决 Baseline 缺陷的方式总结：
  1. **针对"GPU 内存受限"**：Expert offloading 将 expert 参数和 optimizer states 迁移到 CPU/SSD——GPU 仅保留 non-expert + 当前活跃 expert。MoE-L 64 experts (29.3B params) 仅需 4 GPUs，而 baseline OOM。支持 up to 67× 更多 experts（with SSD）和 63× 更大参数量。
  2. **针对"dispatch mask 内存爆炸"**：Sequential expert processing 替代 batched GEMM——无需 dispatch mask，按 gating 结果直接逐 expert 分配 token。节省 >48 GiB（MoE-L batch 32），允许 8× 更大的 microbatch。
  3. **针对"zero-padding 浪费"**：Dynamic expert placement——greedy scheduling 按 per-batch token 分布分配 expert 到 GPU，GPU 负载差异从 102%（Fairseq）降至 15%。同时消除 zero-padding 无效计算（39% → 0%）。
  4. **针对"CPU optimizer 慢"**：Expert-wise CPU optimization——每个 expert backward 完成后立即启动 CPU Adam，与后续 layers 的 GPU forward/backward 重叠。GPU 利用率提升 61.1%（vs 无 pipelined optimizer），总吞吐量提升 up to 63.0%。
  5. **自适应 offloading**：3 种模式自动切换——GPU-only 模式下仍因 sequential processing + 去 zero-padding 而优于 Tutel（1.7×-3.16× speedup）。Expert pinning（固定 top 25% 热门 expert 在 GPU）进一步提升 22.8% 吞吐量。

## Scaling Laws for Fine-Grained Mixture of Experts

- baseline方法是什么？
  Baseline方法：**标准MoE（Vanilla MoE, G=1）**——每个expert的hidden dimension固定为d_ff（标准FFN层的4×d_model），expansion rate E控制总expert数量（N_expert = E），每个token经由router选择top-k个（通常k=1或k=2）expert处理。训练采用固定时长（如Clark et al. 2022使用130B tokens固定数据集），不调整训练token数与模型size的配比，也不调整expert粒度。Scaling law沿用Clark et al. (2022)的形式 L(N,E) = (10^{d/a}/N)^a × (1/E)^{b+c·log N}，仅适用于固定数据集大小，无法指导compute-optimal训练配置。
  全栈执行例子（Baseline: G=1 MoE, E=64, N_act=64×25M, D=130B tokens, A100 GPU）：
  ```
  训练一个Fine-Grained MoE layer（Transformer decoder第L层）：
  ├─ [算法Pipeline] MoE Layer定义：
  │   d_model=512, d_ff=2048, G=1, E=64
  │   d_expert = d_ff/G = 2048  ← 每个expert与标准FFN相同大小
  │   N_expert = G×E = 64个expert
  │   每token路由到k=1个expert
  │   问题：expert粒度粗，每个expert处理高度混合的token模式
  │
  ├─ [GPU Kernel] Router forward:
  │   W_router [512, 64] → router_logits [T, 64]
  │   softmax over expert dim, top-1 selection
  │   Router FLOPs ≈ d_model × E × 14 (c_r) per token per layer
  │
  ├─ [GPU Kernel] Expert Computation:
  │   每个expert: W1 [512, 2048], W2 [2048, 512]
  │   Per-token active FLOPs ≈ 8d_model^2 = 2.1M
  │   总active参数: 2 × d_model × d_ff = 2.1M per token
  │   问题：G=1时所有64个expert竞争token，expert specialization有限
  │
  ├─ [算法Pipeline] Training:
  │   D=130B tokens固定（Clark et al. 2022的设置）
  │   模型参数量N增加但D不相应增加 → undertraining
  │   问题：N增大时模型未能充分训练，dense逐渐追平MoE
  │
  └─ Scaling失效：
      Clark et al. (2022)结论：N>1T时dense超越MoE
      原因：固定D=130B使大模型undertrained
  ```
  Baseline缺陷：(1) **Expert粒度固定为G=1**：d_expert固定为d_ff（=4×d_model），expert层与标准FFN等价大，缺乏灵活度来更精细地匹配token-expert映射，限制了MoE的潜力；(2) **Scaling law未包含训练时长变量**：Clark et al. (2022)的公式仅适用于固定数据集大小，无法预测compute-optimal训练配置；(3) **固定训练时长导致错误结论**：训练时间不随模型size增长时，大模型undertrained，造成"dense在大模型时超越MoE"的假象；(4) **缺乏细粒度路由的建模**：没有理论工具来量化细粒度expert（d_expert < d_ff）的收益，实际中无法系统性选择最优G值。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**Fine-Grained MoE Scaling Laws**——引入granularity G作为新超参数（G = d_ff / d_expert），将expert大小从固定d_ff解放为可调整变量，并在Scaling Law公式中显式建模G的影响。核心机制：
  (1) **Granularity参数化与Power-Law假设** → 解决缺陷(1)：定义G使d_expert = d_ff / G，对任意G，每个token被路由到G个细粒度expert（而非1个粗粒度expert），保持激活参数量不变（d_model × 2d_ff = d_model × 2G·(d_ff/G)）。更细粒度的expert提供更灵活的token-to-expert映射，实验验证power-law关系 L(G) = g/G^γ + h。
  (2) **Joint Scaling Law L(N,D,G)** → 解决缺陷(2)(4)：推导包含G的联合Scaling Law——L(N,D,G) = c + (g/G^γ + a)/N^α + b/D^β，基于以下观察：(a) c不依赖于架构（数据固有熵），(b) D和G无交互（图3c：不同G下训练更多token的收益相同，故b/D^β项不含G），(c) a_G = g/G^γ + a（G对模型size scaling的修正项，a确保无限G不会超越密集模型）。用Huber loss + BFGS拟合100+实验数据，RMSE=0.015。
  (3) **Compute-Optimal配置求解** → 解决缺陷(3)：在含routing overhead的FLOPs约束下（F = (12d_model^2 c_f + d_model·E·G·c_r) × D × n_blocks），使用Brent's method求解min_{N,D,G} L(N,D,G)，找到给定计算预算F下的最优N、D、G组合。FLOPs建模包含routing overhead（c_f=6, c_r=14），使G的选择受实际计算成本约束。
  全栈执行例子（论文方法: G=8, E=64, N_act=64×25M, D=66B tokens, A100 GPU）：
  ```
  训练一个Fine-Grained MoE layer（Transformer decoder第L层）：
  ├─ [算法Pipeline] Fine-Grained MoE Layer定义：
  │   d_model=512, d_ff=2048, G=8, E=64
  │   d_expert = d_ff/G = 256  ← 每个expert缩小8倍
  │   N_expert = G×E = 512个expert
  │   每token路由到k=G=8个细粒度expert（而非1个）
  │   激活参数量不变: 8 × 2 × 512 × 256 = 2.1M = 2 × 512 × 2048 ✓
  │
  ├─ [GPU Kernel] Router forward:
  │   W_router [512, 512] → router_logits [T, 512]  ← router变大8×
  │   Expert Choice Routing: softmax over expert dim
  │   每group 256 tokens，每个expert选择top-k个token
  │   负载天然均衡（无需auxiliary loss），G越大expert越多routing越关键
  │   Router FLOPs ≈ d_model × E × G × c_r per token per layer
  │
  ├─ [GPU Kernel] Fine-Grained Expert Computation:
  │   每个细粒度expert: W1 [512, 256], W2 [256, 512]
  │   Per-token计算: 分配给G=8个expert
  │   每个expert计算量: 2 × d_model × d_expert = 2 × 512 × 256 = 262K FLOPs
  │   总每token: 8 × 262K = 2.1M FLOPs（与G=1时相同）
  │   G=8优势: 512个expert vs 64个expert，token-to-expert映射精细8倍
  │
  ├─ [GPU Kernel] Extra LayerNorm after MoE:
  │   critical for G>1: 稳定G个expert输出的加和
  │
  ├─ [算法Pipeline] Compute-Optimal Training:
  │   F = (12×512²×6 + 512×64×8×14) × D × n_blocks
  │     = (18.9M + 3.7M) × D × 8  ← routing占~16%
  │   D = 66B tokens（compute-optimal for N=64×25M, G=8）
  │   对比G=1: 同样D下loss从3.12降至2.95（图3a）
  │
  └─ Scaling效果：
      G=4时最优: N=64×25M, D=66B, loss更低
      G=8时进一步: wall-clock time最优（A100实测，图5b）
      G=16时: 更大routing开销开始主导（router参数512×1024）
      更大模型（N=64×7B）最优G=32→64，验证"compute budget越大→G越大"
  ```

  解决 Baseline 缺陷的方式总结：
  1. **针对"Expert粒度固定G=1"**：引入granularity G解耦expert大小与d_ff，更多细粒度expert（G=8时512个 vs G=1时64个）提供更精细的token-expert mapping，固定N、D下降低loss（L ∝ 1/G^γ with γ=0.58），在几乎所有FLOPs预算下G>1都优于G=1。
  2. **针对"Scaling law不含训练时长变量D"**：借鉴Chinchilla方法将D显式纳入公式 L(N,D,G) = c + (g/G^γ + a)/N^α + b/D^β，使Scaling Law能预测不同N、D、G组合下的loss，覆盖未训练过的region（validation RMSE=0.019）。
  3. **针对"固定训练时长导致错误结论"**：当N、D、G都选为compute-optimal时，MoE始终优于dense且差距随计算预算扩大（10^20 FLOPs时节省20×，10^25 FLOPs时节省>40×），推翻Clark et al. (2022)的"大模型时dense超越MoE"结论——其错误根源在于未调整训练token数D。
  4. **针对"缺乏细粒度路由的成本建模"**：在FLOPs约束中显式建模routing overhead（c_r=14），包含router计算的正反向传播、token dispatch/combine等7组operations。Brent's method优化时routing cost会限制G的选择，使G随模型增大而增长但不会无限增长（图5b：G=16时routing overhead超过granularity收益）。

## ScheMoE- An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling

- baseline方法是什么？
  Baseline 为现有 MoE 分布式训练系统（Tutel [16] 和 Faster-MoE [14]），它们在 MoE 训练中采用 expert parallelism + data parallelism，通过 all-to-all（A2A）collective 通信完成 token 的 dispatch 和 combine。以 Tutel 为例的全栈执行例子：
  - **算法pipeline**：MoE layer 替换 Transformer fflayer，gating function（softmax + top-k routing）动态选择 expert → 每个 GPU 持有 E/P 个 expert，capacity factor f 控制每个 expert 最大 token 数 C = f × k × B × L / E
  - **系统框架**：Tutel 基于 PyTorch，MoE layer 输入 token tensor I ∈ R^{(E, C, M)} → GPU i 将本地 token dispatch 到对应 expert 所在 GPU j（通过 NCCL-A2A 或 2DH-A2A）→ expert 计算 fflayer（linear1 → GELU → linear2）→ 结果 combine 回原 GPU
  - **编译框架**：论文未明确说明，Tutel/Faster-MoE 均为 PyTorch 原生扩展，不涉及独立编译框架
  - **kernel调度**：Tutel 和 Faster-MoE 将输入 token tensor 按 capacity 划分为多个 chunk 进行通信-计算流水线化（pipelining degree 由用户手动设定或有限搜索空间内的启发式搜索），但 A2A 通信和 expert 计算之间的重叠是 sub-optimal 的——schedule 模式固定、未证明最优性，且当硬件配置或模型配置变化时容易失效
  - **硬件架构**：PCIe 3.0 ×16 intra-node + 100Gb/s InfiniBand inter-node，NCCL-A2A 或 2DH-A2A 顺序执行所有 Send/Recv 操作，intra-node 和 inter-node 带宽无法同时利用

  Baseline 的三个核心缺陷：
  1. **可扩展性差**：Tutel/Faster-MoE 的调度算法与其 A2A 实现紧耦合，新增 A2A 算法或压缩方法需重新设计调度，无法复用
  2. **A2A 带宽利用 sub-optimal**：2DH-A2A 虽利用层次化拓扑，但 intra-node 和 inter-node 通信仍顺序执行，无法同时占用两种带宽
  3. **调度 sub-optimal**：计算-通信流水线的任务执行顺序未经过最优性证明，不同硬件/模型配置下性能不保证

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ScheMoE 提出三个技术组件，对应解决上述三个缺陷：
  1. **模块化任务抽象**：将 MoE layer 的三个关键操作抽象为 AbsCompressor（compress/decompress）、AbsAlltoAll（all_to_all）、AbsExpert（fflayer compute），每个接口可独立替换实现。通过继承抽象基类即可集成新的压缩算法（ZFP/FP16/INT8）和 A2A 算法（NCCL-A2A/1DH-A2A/2DH-A2A/Pipe-A2A），无需修改调度器。
  2. **Pipe-A2A 算法**：引入两个异步 CUDA stream（Intra-Stream 和 Inter-Stream），将 A2A 中的 SR(i,j) 操作按 GPU 对是否同节点分配到不同 stream → intra-node 和 inter-node 通信并行执行，理论加速比 S_max = (M×t₁ + (P-M)×t₂) / max(M×t₁, (P-M)×t₂)。当消息 ≥ 200MB 时实测 1.4×-2× 优于 2DH-A2A。
  3. **OptSche 最优调度算法**：将 MoE layer 的 7 类任务（C1/A1/D1/E/C2/A2/D2）按 r 路输入分区后，在数据依赖约束（4)-(9）下数学证明最优任务执行顺序，确保通信任务被最大程度隐藏。r=2 时最优 CompTask 顺序为：(C₁¹C₁²)→(D₁¹E¹C₂¹)→(D₁²E²C₂²)→(D₂¹D₂²)，CommTask 在前置完成后即时启动。

  ScheMoE 全栈执行例子（CT-MoE, 32 GPU, r=2, ZFP+ Pipe-A2A + OptSche）：
  - **算法pipeline**：gating 输出 I ∈ R^{(32, C, M)} → ZFP_compress(I, rate=8) 将 32-bit 压缩为 8-bit（通信量 ↓4×）→ Pipe-A2A dispatch → ZFP_decompress → expert fflayer 计算 → ZFP_compress → Pipe-A2A combine → ZFP_decompress
  - **系统框架**：PyTorch MoE layer 替换为 ScheMoE.MOELayer，内部 task queue 将 7×2=14 个 sub-tasks 入队 → Profiler 在预热阶段测量各 task 耗时 → Scheduler 按 OptSche 最优顺序调度
  - **编译框架**：论文未明确说明，基于 PyTorch C++/CUDA extension（~1200 行），不涉及独立编译栈
  - **kernel调度**：CUDA stream 管理——compute tasks 在 default stream，Pipe-A2A intra-node SR 在 Intra-Stream、inter-node SR 在 Inter-Stream → 三个 stream 并发执行，当 t_intra < t_inter 时 intra-node 通信被完全隐藏
  - **硬件架构**：32-GPU（8×4 RTX2080Ti），PCIe 3.0 ×16 + 100Gb/s IB，Pipe-A2A 利用两个 stream 同时占用 PCIe 和 IB 带宽

  解决 Baseline 缺陷的方式总结：
  1. **针对"可扩展性差"**：AbsCompressor/AbsAlltoAll/AbsExpert 三层抽象接口让新算法通过继承和虚函数即可接入，Scheduler 通过 Profiler 自动获取新模块的性能模型，无需修改调度代码
  2. **针对"A2A 带宽利用 sub-optimal"**：Pipe-A2A 通过 Intra-Stream/Inter-Stream 双路异步执行消除 intra-node 和 inter-node 通信的串行化瓶颈。理论加速比由 t_intra 和 t_inter 的比例决定，当两者接近时加速最大（式 18）
  3. **针对"调度 sub-optimal"**：OptSche 在给定输入分区度 r 的条件下数学证明了最优任务执行顺序（定理 1），保证了任何满足约束 (4)-(9) 的调度方案无法超越。消融实验验证：Naive → +ZFP 已有 1.9× → +Pipe-A2A 达到 2.2× → +OptSche 最终 2.4× speedup

## Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts

- baseline方法是什么？
  Baseline方法：标准top-2 MoE专家并行。在分布式MoE训练/推理中，每个Transformer block的MLP被替换为top-2 gating MoE模块（每第二个block，即"Block-MoE"与"Block-MLP"交替放置）。执行流程严格串行：gate routing → input encode → All-to-All dispatch → expert computation → All-to-All combine → output decode。All-to-All通信可占MoE层总时间的约50%（多节点场景下因低带宽inter-node Ethernet甚至接近50%），成为主要瓶颈。现有优化：(1) Hierarchical All-to-All：利用层次化拓扑减少通信量；(2) Pipeline策略（如Tutel）：将tokens切分为fine-grained chunks，不同chunks的通信和计算在不同CUDA streams上交错执行实现部分重叠。但pipeline受限于首尾chunks（prologue/epilogue）的通信无法被计算隐藏——初始chunks仅通信无计算、末尾chunks仅计算无通信产生bubble。核心根本限制：**通信与计算存在顺序依赖**——expert computation必须在当前层representations就绪后才能启动，All-to-All通信必须在gate routing之后且expert computation之前。
  全栈执行例子（Baseline: Standard top-2 MoE + Pipelin, 8×A30-PCIe, training one iteration）：
  - **算法Pipeline层**：token x进入Block-MoE → gate network计算G(x) = Softmax(TopK(H(x), 2)) → 选择top-2 experts → 按公式MoE(x) = ΣG(x)_i E_i(x)计算 → 输出。MoE模块仅处理当前层representations，无跨层信息复用。
  - **系统框架层**：Tutel MoE framework，expert parallelism（每GPU分配不同expert），All-to-All token dispatch/combine基于NCCL实现，pipeline将tokens等分chunks交错通信与计算。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：pipeline strategy——输入tokens等分为M个micro-batches，每个micro-batch的dispatch和computation在不同CUDA streams上流水线执行。但第1个chunk的dispatch无法被任何computation隐藏（无前置计算），第M个chunk的combine同理。bubble time = T_disp/M + T_comb/M。算子调度严格遵循序列顺序：gate → encode → dispatch → compute → combine → decode，不可重排。
  - **硬件架构层**：8×A30-PCIe（PCIe互联，通信带宽低，All-to-All占MoE总时间~60%），8×A800-NVLink（NVLink高带宽，通信占~15%但仍有不可隐藏的prologue/epilogue）。多节点场景下inter-node Ethernet引入额外通信瓶颈。
  Baseline核心缺陷根因：MoE模块的输入仅来自当前层的attention输出，通信-计算依赖链条完全串行——必须先完成gate routing（依赖当前层表示）、再dispatch tokens、再expert computation。这种**顺序依赖**从根本上限制了任何overlap优化的天花板。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：**ScMoE（Shortcut-connected MoE）**——通过引入shortcut连接将MoE模块的输入从仅当前层表示扩展为"前一层表示+当前层表示"，利用前一层表示进行gate-routed expert的选择和计算，从而将gating/通信/计算从当前层的串行链中解耦。结合自适应算子调度策略实现通信-计算最大重叠。
  核心设计对应：
  **(1) Shortcut-connected架构 → 解决"通信-计算顺序依赖"**：ScMoE使用top-1 MoE处理前一层（Block-MLP）的中间表示H_l^{MH}（通过shortcut），shared expert处理当前层（Block-MoE）的表示H_{l+1}^{MH}。由于前一层表示H_l^{MH}在Block-MLP的attention阶段就已确定，gating和token dispatch可以在Block-MLP的MLP计算和Block-MoE的attention计算期间提前执行——**gating不依赖当前层的任何计算**。这从根本上打破了baseline中"必须先等当前层attention完成才能开始gate routing"的顺序依赖。
  **(2) 自适应算子调度 → 解决"pipeline的prologue/epilogue bubble无法消除"**：在ScMoE解耦后，gate routing和encode可在MoE stream最早位置立即启动（不需要等待当前层计算），All-to-All dispatch可与shared expert stream的MultiHead attention + Shared Expert MLP重叠。Expert computation被插入shared expert stream的4个候选位置之一，通过最小化|T_comp_pre - T_disp| + |T_comp_post - T_comb|自适应选择。当通信时间≤重叠窗口（约50%总MoE时间）时实现100%通信隐藏——这是pipeline策略无法达到的，因为pipeline始终有prologue/epilogue bubble。
  **(3) 理论保证 → 解决"对模型质量的潜在担忧"**：Shortcut连接理论上保证梯度 ∂E/∂x_l = ∂E/∂x_L(1 + ∂/∂x_l ΣF_{W_i}(x_i))，加性分量确保信息直接反向传播至任意子层，避免梯度消失/爆炸。实验中表明相邻Transformer block的中间表示cosine相似度接近1.0（如Figure 10所示），因此用前一层表示替代当前层表示进行expert计算在模型质量上等价甚至更优。

  全栈执行例子（ScMoE Pos-2 + CG-1, 8×A30-PCIe, training one iteration）：
  - **算法Pipeline层**：同一对Block-MLP+Block-MoE中，Block-MLP先计算attention得到H_l^{MH} → 此时shortcut已将H_l^{MH}传给MoE stream → gate routing在Block-MLP MLP计算期间即可开始（因为H_l^{MH}已在Block-MLP attention后确定）→ 同时Block-MoE的attention以H_l^{MLP}为输入计算H_{l+1}^{MH} → shared expert直接处理H_{l+1}^{MH} → gate-routed expert处理H_l^{MH}（通过shortcut）→ 两路结果用CG-1系数组合：MoE(x) = Sigmoid(H_{l+1}^{MH}·W_CG) · SE(H_{l+1}^{MH}) + ΣG(H_l^{MH})_i E_i(H_l^{MH})。
  - **系统框架层**：基于Tutel+FaiRSeq实现。双CUDA stream架构——主stream执行Block-MoE attention + shared expert，MoE stream执行gating + All-to-All + expert computation。自适应调度器在CPU侧根据profiled T_disp/T_comb/T_comp选择最优expert computation位置K*。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：MoE stream中：gate + encode → Async All-to-All dispatch（与主stream的attention重叠） → expert computation（在自适应选择的K*位置，与主stream的shared expert重叠） → Async All-to-All combine（与主stream的后续计算重叠） → decode。8×A30-PCIe场景：重叠窗口 = T_Atten + T_SE + T_MLP ≈ 70% MoE时间，剩余30%可用pipeline augmentation进一步隐藏。8×A800-NVLink场景：通信仅占15%，完全被overlap_window覆盖 → 100%通信隐藏。
  - **硬件架构层**：同一A30-PCIe/A800-NVLink平台。关键变化：通信操作不再在硬件上串行暴露——因提前启动了gating和dispatch，通信时间被前置计算（Block-MLP MLP + Block-MoE attention）和后置计算（shared expert）双重吸收。对比baseline pipeline在A30-PCIe仍暴露显著通信bubble，ScMoE实现1.49×训练加速和1.82×推理加速。

  解决Baseline缺陷的方式总结：
  1. **针对"通信-计算顺序依赖"**：Shortcut连接将expert routing和dispatch的输入从"当前层表示"前移为"前一层表示"，打破串行依赖链——gating和通信可在当前层计算启动前就开始，从根本上扩大重叠窗口。
  2. **针对"pipeline的prologue/epilogue bubble"**：ScMoE的overlap window不依赖于数据切分（chunk），而是从时间线前端（Block-MLP计算期间）延伸到后端（Block-MoE shared expert期间），天然覆盖首尾，不存在pipeline的bubble限制。当通信≤overlap_window时实现100%隐藏。
  3. **理论+实验双重验证模型质量等价**：梯度传播理论保证训练稳定性；相邻层表示相似度分析（cos near 1.0）+ 多模型多任务实验证明ScMoE模型质量持平甚至超越baseline。为architecture-algorithm co-design提供可推广的范式。

## SiDA Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models

- baseline方法是什么？
  Baseline方法：(1) **Standard MoE 推理**：直接使用 HuggingFace Transformers 中的 Switch Transformer 实现，所有 expert 参数常驻 GPU 内存，每次前向通过 router 函数（线性层 + SoftMax + top-k）在线确定激活的 expert 并调用。存在三个核心问题：①低效 GPU 内存利用——大部分 expert 参数在推理中闲置（Switch-base-256 上 MoE 参数占模型 99.07%，但仅 <20% 的 expert 被激活，短句场景下浪费高达 50GB GPU 内存）；②高 MoE 开销——expert 选择、expert 调用和通信开销占据高达 72% 的总推理时间（Switch-base-256），且随模型规模放大；③router 在线选择的延迟惩罚——在小 batch 场景下，调用 expert 的开销超过计算本身。(2) **DeepSpeed-MoE** 和 **Tutel**：优化了设备间通信、自适应并行和流水线调度，但均未利用数据感知（data-awareness）来进一步提升内存效率，所有 expert 仍在 GPU 上，无效 GPU 内存利用问题未解决。
  全栈执行例子（Baseline: Standard Switch-base-256 推理，单 A100 80GB，SST2 数据集）：
  - **算法Pipeline层**：token embedding → self-attention → router（W_r^T x → SoftMax → top-1 expert选择）→ 调用选中的 expert MLP → α 加权输出。所有 256 个 expert 的 MLP 参数（54.114 GB）全部提前加载到 GPU 内存。
  - **系统框架层**：HuggingFace Transformers 默认实现——每个 MoE 层调用所有 expert（即使无 token 分配到该 expert），以对齐高效计算的硬件要求。无任何 CPU-GPU 之间的参数 offloading。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明——默认使用 PyTorch GEMM kernel，无 expert-specific kernel 优化。
  - **硬件架构层**：A100 80GB GPU，所有数据流经 GPU HBM ↔ SM 的片上路径，无 CPU-GPU 数据交换，但 GPU 内存利用率极低（<5% for short sentences）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **SiDA-MoE** 提出 **数据感知（data-aware）的稀疏驱动推理系统**，核心机制是通过离线训练的 hash 函数预先预测 expert 激活模式，使推理系统能提前知晓哪些 expert 将在当前 batch 中被激活，从而在推理开始前完成 expert 的动态加载/卸载。解决 baseline 缺陷的方式如下：
  
  **(1) Hash-building 线程解决 MoE 开销瓶颈**：用离线训练的 LSTM+sparse attention hash 函数替代在线 router 选择。Hash 函数在独立 CPU 线程中运行，预测每批输入在所有 MoE 层的 expert 激活模式。由此将 expert 选择从推理关键路径中移除——吞吐量提升达 3.93×，延迟降至 28%。这是针对 baseline 中"expert 选择占据 72% 推理时间"的直接解决方案。
  
  **(2) 动态 offloading 解决低效 GPU 内存利用**：Inference 线程根据 hash table 仅将当前批激活的 expert 加载到 GPU，将未激活的 expert 卸载到 CPU 主内存（FIFO 驱逐策略）。利用现代服务器 CPU 可达 TB 级内存容量的特性，使 GPU 只存放当前有效的参数——GPU 内存节省达 80%（SST2），大模型（Switch-base-256）在长句场景（MultiRC）仍节省 20%+。
  
  **(3) 双线程管道并行实现零开销调度**：Inference 线程和 Hash-building 线程并行运行——推理线程处理 batch X_i 时，hash-building 线程已在预测 batch X_{i+1} 的 expert 模式。由于推理耗时远大于 hash 预测，hash table 始终在推理需要时已就绪，两条线程无空闲等待。此设计使 expert 选择、动态 offloading 和模型计算三者完全重叠并行。
  
  **(4) Sparse Attention + Truncated KD 确保预测精度**：Hash 函数虽轻量但需要高精度——Sparse Attention（SparseMax 激活）使预测器自动关注少数关键 token，匹配实验中发现的稀疏跨 embedding 依赖（c^i ≈ 1-4 个关键 token 影响 expert 激活）；Truncated KD（T=30）配合交叉熵损失使预测器在容量受限条件下仍达到 Top-3 准确率 >99%（SST2）。
  全栈执行例子（SiDA-MoE，Switch-base-256，单 A100 80GB，SST2 数据集）：
  - **算法Pipeline层**：Hash-building 线程（CPU）：token embedding → 2层LSTM → Self-Attention(SparseMax, 仅关注 c^i≈1-4 个关键token) → FC压缩 → Residual → FC → top-1 expert预测 + scaling factor α。Inference 线程（GPU）：跳过 router，直接根据 hash table 的 (expert_id, α) 调用对应 expert MLP，α 加权输出。
  - **系统框架层**：HuggingFace Transformers + SiDA-MoE Manager——维护双线程协调（shared queue 传递 hash table），管理 expert 设备置放（GPU HBM ⇄ CPU DDR4 主内存），FIFO 驱逐策略。每层 MoE 完成后流水线触发下一层的 expert 加载/卸载。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：CPU-GPU 之间的 expert 参数传输（PCIe），GPU 内使用默认 PyTorch kernel。论文未引入新的 CUDA kernel。
  - **硬件架构层**：A100 80GB GPU + CPU 主内存（TB 级 DDR4）。数据流：CPU 主存 → PCIe → GPU HBM（加载激活 expert），GPU HBM → PCIe → CPU 主存（卸载未激活 expert）。80% 的 expert 参数（~43GB for Switch-base-256）存在 CPU 侧，仅在激活时才传输到 GPU。

## SimSMoE Toward Efficient Training Mixture of Experts via Solving Representational Collapse

- baseline方法是什么？
  Baseline方法：(1) **SMoE with Balancing Loss (Fedus et al., 2022)**：使用可训练的MLP路由器+辅助平衡损失迫使token在各expert间均匀分配，通过SWITCH Transformer的top-k gating选择expert。全栈执行例子（Baseline: Brainformer 135M, top-2 routing, enwik8, 单卡A100）：token x进入MoE layer → router计算logits = softmax(W_r @ x) → top-2选择expert → dispatch token到两个选中expert → 每个expert执行MLP(x) = W_down @ ReLU(W_up @ x) → 加权求和expert输出 → residual add。全局加L_balancing = N·Σ f_i·p_i鼓励expert均匀使用。③ 训练50k steps，Adam optimizer，linear LR schedule。**核心缺陷**：仅通过balancing loss控制token分配，不直接处理expert隐藏表征层面的collapse问题——随着训练进行，不同expert的隐藏表征逐渐趋同（CKA相似度升高），导致expert参数冗余、模型性能受限。XMoE和StableMoE通过改进routing策略间接缓解此问题（如XMoE在低维hypersphere上routing、StableMoE两阶段训练先冻结router再训练expert），但均无法保证解决representation collapse，且改进效果不一致或不显著（如论文Table 1中XMoE在Text8上反而不如vanilla SMoE）。
  全栈执行例子（Baseline: SMoE + Balancing Loss, Brainformer 135M, 单卡A100, single token）：
  - **算法Pipeline层**：token x进入layer l的MoE模块 → router: logits = W_r @ x ∈ R^N → softmax → top-2 gate = top2_indices → dispatch tokens to selected experts → Expert_i(x) = W_down_i @ ReLU(W_up_i @ x) → output = Σ gate_i * expert_i(x)。所有expert共享相同结构（两层MLP + ReLU），仅通过routing的稀疏激活造成差异——但随着训练进行，不同expert的隐藏表征h_i之间的CKA相似度逐渐趋近于1（论文Figure 4验证了此collapse现象），导致expert参数冗余。
  - **系统框架层**：基于CompeteSMoE公开实现（PyTorch），单卡A100 GPU训练，使用HuggingFace生态进行模型构建和评估。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。使用标准PyTorch CUDA kernel（cuBLAS GEMM）执行expert MLP计算。
  - **硬件架构层**：单卡NVIDIA A100 GPU。所有expert参数驻留GPU HBM，训练过程中无expert offloading。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：SimSMoE提出**直接解决expert表征层面的collapse**而非间接优化routing的策略。核心设计：
  (1) **用CKA量化collapse**：引入Centered Kernel Alignment度量两expert隐藏表征的相似度。CKA(K_i, L_j) = tr(K_i H L_j H) / sqrt(tr(K_i H K_i H) * tr(L_j H L_j H))，其中K_i为expert i隐藏表征的kernel矩阵，H为centering matrix。该度量对可逆线性变换不变，能可靠识别expert表征间的一致性。
  (2) **Similarity Learning Module直接最小化相似度**：在SMoE架构中插入MLP投影头（单隐层，输出维度≈N），将expert隐藏表征映射到投影空间，计算CKA相似度损失L_similarity，直接优化以降低collapsed expert对之间的相似度。总损失：L = L_task + α·L_balancing + β·L_similarity。
  (3) **高效的collapse识别策略（频率+阈值双控）**：为避免检查所有N(N-1)/2个expert对（违背conditional computation理念），引入f*（token共享频率阈值）和T*（collapse相似度阈值）。仅对共享token数≥f*的expert对检查CKA，仅当CKA≥T*时将L_similarity加入总loss。f*控制计算效率，T*控制collapse判定精度。
  (4) **与任何routing算法兼容**：SimSMoE的Similarity Learning Module作用于expert输出端而非router输入端，因此可直接叠加在SMoE/XMoE/StableMoE等任何routing机制上，增强已有routing方法的性能。
  全栈执行例子（SimSMoE, Brainformer 135M, 单卡A100, single token）：
  - **算法Pipeline层**：token x进入layer l的MoE模块 → router输出top-2 expert indices → Expert_i和Expert_j分别计算MLP隐藏表征h_i, h_j → **Similarity Learning Module切入**：h_i, h_j通过MLP投影头映射到投影空间（维度≈N）→ 计算kernel矩阵K_i=h_i_proj×h_i_proj^T, L_j=h_j_proj×h_j_proj^T → CKA(K_i, L_j) → 若f_ij ≥ f*且CKA ≥ T*，则L_similarity += CKA → 总loss = L_lm + α·L_balancing + β·L_similarity → 反向传播使CKA最小化，直接让expert i和j的隐藏表征变得不同 → expert输出 = gate_i * h_i + gate_j * h_j。与baseline的核心差异：baseline仅通过routing选择不同expert，但expert的表征本身可能趋同；SimSMoE通过CKA loss直接惩罚expert表征的相似性，使expert真正差异化。
  - **系统框架层**：基于CompeteSMoE实现。Similarity Learning Module作为轻量叠层，额外参数0.08M-0.16M可忽略。训练开销增加主要来自CKA计算（仅对满足f*的expert对），paper验证f*控制开销且对性能影响可控。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。使用标准PyTorch CUDA kernel。
  - **硬件架构层**：与baseline相同——单卡A100 GPU。Similarity Learning Module的参数量和计算开销极小，不增加显存瓶颈。
  解决Baseline缺陷的方式总结：
  1. **针对"routing-based方法间接、不可靠、不一致"**：SimSMoE直接操作expert隐藏表征层面，通过CKA loss最小化expert间相似度，而非依赖routing算法的间接效果。Paper验证SimSMoE叠加在SMoE/XMoE/StableMoE上均一致提升性能（Table 1-5）。
  2. **针对"无理论保证"**：论文通过CKA相似度指标为collapse提供了定量度量，并通过similarity loss提供了优化目标，使得"是否解决collapse"从隐含变为显式可控。f*和T*提供p≥p*的理论保证（若T≥T*则解决collapse，否则专注于task loss）。
  3. **针对"expert参数冗余"**：通过CKA loss使expert表征去相关化，论文通过heatmap和相似度-共享token相关性分析验证了SimSMoE确实减少了expert表征的重叠。
  4. **大规模有效性**：在1.031B参数Brainformer (64 experts)上，SimSMoE仍一致优于baseline，且性能差距随模型增大而扩大（Table 2）。

## Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 方法：(1) **Switch Transformer 标准 MoE 训练**：使用固定全局辅助损失系数 α（通常 1e-2 或 1e-3），原始门控层直接 softmax(Wx + b) 进行 top-k expert 选择，uniform pipeline parallelism + standard expert parallelism（EP 或 ETP）。门控输出可能退化为高熵分布（top-k 概率接近均匀），导致 expert 输出退化为简单平均而非加权平均，gating 失去区分能力。(2) **传统从 Dense 到 MoE 的 Upcycling**：直接复制 dense checkpoint 的 FFN 权重 n 次初始化 n 个 expert，所有 expert 完全相同（expert similarity = 1），依靠训练过程缓慢分化，初期 expert 多样性极差。(3) **传统 MoE 从头训练**：随机初始化所有 expert，不存在 expert 同质化问题，但需要较大训练预算才能达到与 upcycling 相当的性能。

  全栈执行例子（Baseline: Switch Transformer 标准 MoE，从 Skywork-13B dense upcycling，1536 A800 GPU）：
  - **算法Pipeline层**：Gate 直接 g = softmax(W_gate @ x + b_gate)，top-2 selection → expert FFN forward → weighted combine。辅助损失 L_total = L_ce + α · Σ L_aux^(l)，α 全局固定。门控概率可能退化为近似 1/16 均匀分布，Max1/Max2 ≈ 1, Max2/Max3 ≈ 1，expert 失去区分能力，输出 ≈ (E1(x) + E2(x))/2（简单平均）。
  - **系统框架层**：Megatron-LM 23.06 + Expert Parallelism (EP)，Size_EP = Size_DP * Size_TP。受 expert 数量限制（≤16），GPU 扩展性受约束。Pipeline parallelism 每 stage 均匀分配层数，最后 stage 因 loss calculation 成为 bottleneck。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：Uniform PP + 标准 EP/ETP，AllToAll 通信开销大（ETP 情况下随 TP 增大迅速增加），无通信-计算 overlap 优化。
  - **硬件架构层**：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **Skywork-MoE** 通过三项核心技术创新解决 baseline 缺陷：

  **(1) Gating Logit Normalization —— 解决门控退化问题**
  Baseline 的 g = softmax(Wx + b) 在训练中容易退化为高熵分布。Skywork-MoE 在 softmax 前插入归一化：z̃ = λ · (z − μ)/σ，确保 logit 向量具有零均值和可控标准差 λ。λ 控制 softmax 输出的锐度——λ 越大分布越集中。实验证实 λ=1 时 Max1/Max2 和 Max2/Max3 比率远高于无归一化（后者退化到 1），token drop rate 大幅降低，训练 loss 改善。

  **(2) Adaptive Auxiliary Loss Coefficients —— 解决辅助损失与主任务冲突**
  Baseline 的全局固定 α 要么过度正则化（牺牲交叉熵优化）要么欠正则化（负载不均）。Skywork-MoE 为每层引入独立的 α^(l)，并根据实时 token drop rate d_i^(l) 自适应调整：α̂ = min(ξ·d, α_max)，α 通过移动平均平滑更新（β=0.99）。这使得负载均衡正则化仅在需要时增强，在负载均匀时自动减弱，优先确保交叉熵损失优化。

  **(3) Upcycling 预算决策框架 —— 提供何时 upcycle 何时从头训练的量化指导**
  通过控制实验（0.3B dense × 100B/300B MoE tokens）提出量化规则：C_MoE ≪ C_dense 时 upcycling 占优，C_MoE ≥ 2·C_dense 时从头训练占优。并发现 upcycling 过程中 expert similarity 从 1 逐渐下降（diversification 过程），可作训练监控指标。
  
  **(4) EDP + Unbalanced PP —— 提升训练效率**
  EDP（Size_EP = Size_TP）优化中等 expert 数量场景的 AllToAll 通信。Unbalanced PP（如 [5,5,5,5,4]）减少 pipeline bubble 10%。最终达到 38% MFU。

  全栈执行例子（Skywork-MoE，1536 A800 GPU，146B/16 experts）：
  - **算法Pipeline层**：Gate forward: z = W_gate @ x + b_gate → z̃ = (z-μ)/σ → g = softmax(z̃) → top-2 selection → expert FFN (SwiGLU) → y = (g1·E1(x)+g2·E2(x))/(g1+g2)。辅助损失：每层独立 α^(l)，每 iteration 根据 token drop rate 更新 α^(l) = 0.99·α_prev^(l) + 0.01·min(0.2·d_i^(l), 0.01)。总 loss = L_ce + Σ_{l=1}^{52} α^(l)·L_aux^(l)。对比 baseline 的门控退化（均匀概率）和固定 α，Skywork-MoE 的门控分布更尖锐（Max1/Max2 > 1），token drop rate 更低，辅助损失系数随层和训练阶段动态变化。
  - **系统框架层**：Skywork-Megatron（基于 Megatron-LM 23.06），EDP 策略（Attention TP mesh [PP, DP, 4] → Expert EP mesh [PP, DP, 4] 灵活切换），12-way unbalanced PP + 32-way DP + ZeRO-1。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：EDP 优化 AllToAll 通信（受限于 ≤64 experts 场景），unbalanced PP 减少 bubble 10%，kernel fusion + 通信-计算 overlap，38% MFU / 690 tok/GPU/s。
  - **硬件架构层**：论文未明确说明。

## SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

- baseline方法是什么？
  Baseline方法：(1) **FasterMoE（纯Expert Parallelism + 运行时调度）**：仅支持 expert parallelism，通过 smart scheduling 和 expert shadowing 等运行时优化缓解负载不均衡。但不支持混合并行（DP/TP/PP 组合），通信开销大（全量 All-to-All），所有 expert 参数分布在所有 GPU 上。对高 capacity factor 场景（严重负载不均衡）调度效果有限。(2) **Alpa（纯离线自动并行化）**：基于 JAX 的通用自动并行化系统，在训练前用 ILP 一次性搜索最优混合并行执行计划（inter-op pipeline + intra-op data/tensor parallelism），不感知动态负载，生成静态执行计划后训练全程不变。搜索耗时 825s（对 16 expert 模型），远超单 iteration 时间。(3) **DeepSpeed-MoE/Tutel（混合并行但无自动搜索）**：支持 EP+DP+TP 混合并行，但需要专家手动调参，且不考虑 expert placement 策略影响。(4) **BaGuaLu**：在超算规模组合 EP+DP，但 expert placement 固定按串行顺序放置，不优化负载均衡。
  全栈执行例子（Baseline: FasterMoE 纯 EP，4 GPUs，16 experts，capacity factor=+∞）：
  - **算法Pipeline层**：Gate: x → W_gate → Top-K → expert indices；Expert FFN: SwiGLU (gate_proj → SiLU ⊙ up_proj → down_proj)
  - **系统框架层**：FastMoE 框架（PyTorch），纯 EP——16 experts 按索引顺序平均分配到 4 GPUs（GPU_0: E0-E3, GPU_1: E4-E7, GPU_2: E8-E11, GPU_3: E12-E15），无 expert placement 优化。All-to-All dispatch → Expert computation → All-to-All combine。FasterMoE 的 shadowing/scheduling 仅缓解计算端负载不均，不改变通信拓扑。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：All-to-All 通信在全部 4 GPU 间执行（全互联），跨节点带宽瓶颈时通信延迟高。GShard gate capacity factor=+∞ 时无负载上限，GPU_0 上的 {E0,E1,E2,E3} 可能因 token 路由不均而严重超载，其他 GPU 空闲。FasterMoE 通过 smart scheduling (token 级动态影子 expert 分配) 部分缓解但仅限计算端，通信量不变。
  - **硬件架构层**：GPU 集群（V100 PCIe, V100 SXM, A100 SXM），节点内 NVLink + 节点间 InfiniBand（50-200Gb/s）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **SmartMoE** 通过 "离线池构建 + 在线自适应并行化" 两阶段设计解决 baseline 缺陷：

  **(1) Enlarged Space of Hybrid Parallelism（扩大混合并行空间）→ 解决纯EP通信瓶颈**
  Baseline（FasterMoE）仅支持纯 EP，通信为全局 All-to-All。SmartMoE 通过 expert slot 抽象统一表达 DP/TP/PP/EP 的任意组合，支持混合并行（如节点内 TP+EP、节点间 DP）缩小通信范围至更少 GPU。同时引入 **Expert Placement** 新维度——expert 到 expert slot 的映射顺序从根本上影响负载均衡。如 4 GPU 16 expert 场景，按 token 负载交错映射 vs 按索引顺序映射，前者可使各 GPU 负载方差从 102% 降至 <1%。

  **(2) Workload-Aware Performance Modeling（负载感知性能建模）→ 解决无法离线预测MoE性能**
  Baseline（Alpa）假设 workload 均匀（data-insensitive），在 MoE 动态负载下预测不准。SmartMoE 利用 gating network 语义估算负载上界——对 GShard gate 用 capacity factor 计算 max_tokens_per_expert；对 topology-aware gate 按算法层次计算最大通信量。无需实际训练数据即可准确预测，R² > 0.5 for all configurations，支持离线阶段穷举搜索最优池。

  **(3) Two-Stage Auto-Parallelization（两阶段自动并行化）→ 解决离线搜索慢+在线需快速决策**
  Baseline（Alpa）的 ILP 搜索需 825s，无法在线执行。SmartMoE 分解为离线+在线两阶段：
  - **Offline**：构建"固定混合并行策略+可变 expert placement"的 pool，池内候选执行计划有相同 expert slot 配置（切换时无内存分配/释放，仅参数交换）
  - **Online**：三种轻量级算法——Greedy (O(NE), <1ms)、DP (O(N×4^E), 最优)、Hybrid (Greedy → virtual devices + DP → physical devices, 可调 M 权衡精度/速度)。利用 expert selection 的时间局部性每 10 iterations 搜索一次

  **(4) Adaptive Runtime Switching（自适应运行时切换）→ 解决固定执行计划效率退化**
  Baseline 使用训练前确定的静态执行计划，在动态负载下效率逐渐下降。SmartMoE 设置切换阈值过滤微小改进（避免通信开销超过计算收益），利用相邻 iteration 的 gating 分布局部性间歇性搜索。搜索在 CPU 侧执行（<1ms），与 GPU 计算并行不占关键路径。

  全栈执行例子（SmartMoE，4 GPUs×4 expert slots，GShard gate capacity=2.4）：
  - **算法Pipeline层**：同 baseline——Gate: x → W_gate → Top-K softmax → expert indices → Expert FFN。不同之处：expert 到 GPU 的 placement 由 SmartMoE 在训练过程中动态决定（ExpertPlacementHybrid(E=16, N=4, C[16])），不固定于模型初始化时。
  - **系统框架层**：SmartMoE 框架（基于 FastMoE+PyTorch），支持 expert slot 抽象下的 DP/TP/PP/EP 任意组合。Offline 阶段 pool search 遍历候选策略空间（如 DP=2×TP=2×(PP=1 inside node) × EP=16 across 4 GPUs），用 workload-aware 性能模型评估。Online 阶段每 10 iterations 触发 placement 搜索，搜索结果通过 NCCL All-to-All 交换 expert 参数（~20ms for 16 experts），然后各 GPU 按新 placement 执行 expert FFN。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：
    Offline: 穷举 + 性能模型 → 输出 pool（固定 DP/TP/PP，可变 expert placement）
    Online per 10 iters (CPU执行):
    输入: expert_token_counts C[16] from gate history
    Hybrid Placement: 
      Step1 Greedy to M virtual devices → 粗粒度均衡
      Step2 DP within each virtual device → 细粒度最优
    → 新 placement P': GPU_0={E5(520),E7(240),E12(245),E1(240)}
                       GPU_1={E0(512),E9(250),E15(248),E2(242)}
                       GPU_2={E3(508),E11(252),E6(249),E10(238)}
                       GPU_3={E4(505),E13(248),E14(247),E8(245)}
    Imbalance < 1%
    切换时 NCCL All-to-All 交换 expert 参数（仅被移动的 expert），Compute 在新 placement 下执行。
  - **硬件架构层**：3 种 GPU 集群——blinky (8×V100 PCIe/50Gb IB), pinky (4×V100 SXM/100Gb IB), inky (8×A100 SXM/200Gb IB)

  关键结果：vs FasterMoE baseline 最高 1.88× speedup（A100 cluster），平均 1.53×（A100）/1.17×（V100 SXM）/1.14×（V100 PCIe）

## Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

- baseline方法是什么？
  Baseline 方法：SGLang（vLLM）的 Expert Parallelism (EP) 推理方案。在 MoE 推理中，expert 按轮询或默认策略均匀分布到各 GPU 上，attention 层使用 DP 或 TP。每个 token 的 gate function 在运行时选择 top-k experts，通过 all-to-all 集合通信将 token dispatch 到远端 expert 所在 GPU 执行 FFN 计算，再 combine 回来。由于 expert 放置不考虑 token-expert affinity，导致大量 cross-device token 路由（~75% tokens 需远端通信），all-to-all 通信成为主要延迟瓶颈（占 MoE 层 59.2% forward latency，见图 1）。

  全栈执行例子（Baseline: SGLang Attention-DP + EP8, DeepSeek-V2-Lite, 请求 A/B/C/D 到达）：
  - **算法Pipeline层**：MoE Gate: Softmax(W_g · h) → Top-K → expert indices（如 token t1→E3,E15; t2→E7,E22; t3→E1,E8...）。Attention: Q/K/V projection → FlashAttention → output。每个 token 独立选择 expert，无先验知识引导。
  - **系统框架层**：SGLang continuous batching → 请求 A-D 被轮询分配到 4 个 DP rank → 各 rank 独立执行 attention → MoE layer: gate → all-to-all dispatch（每 rank 将 token 发送到其选中 expert 所在 rank）→ expert FFN → all-to-all combine（将输出发回原 rank）。expert 默认以轮询方式放置，与 token 语义无关。EP 通信由 NCCL all-to-all 实现，all-to-all 通信量 = αkBS/G。
  - **编译框架层**：SGLang 使用 Triton fused MoE kernel，将 gate + expert computation 融合以减少 kernel launch 开销。但 all-to-all 通信与计算串行执行，无法通过编译优化消除。
  - **Kernel调度层**：NCCL/HCCL all-to-all collective → GPU SM 执行 expert FFN（fused MoE kernel）→ NCCL all-to-all。通信与计算无重叠（EP 的 all-to-all 必须在 gate 之后、expert computation 之后分别执行，形成两道通信 barrier）。
  - **硬件架构层**：8-GPU server（96GB HBM/GPU，>400GB/s 互联），all-to-all 数据经 NVLink/NVSwitch 全交换。

  核心缺陷：(1) Expert placement 与 token 语义无关——轮询放置导致任何 token-expert 对的本地激活概率仅 ~1/E≈12.5%（EP8），~75% tokens 需要跨设备通信；(2) 请求调度不考虑 expert affinity——DP 场景请求被轮询分配，不同请求的 token 被随机散到各 device，加剧 all-to-all 通信；(3) TP 场景 token 在 reduce-scatter 后随机分布，到 MoE 层再通过 all-to-all 重新路由，重复通信。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 **Sem-MoE (Semantic Parallelism)** 通过 model-data collaborative scheduling 从根本上减少 EP 通信量：
  
  **(1) Offline Model Scheduling → 解决 expert placement 与 token 语义无关**
  通过对 ShareGPT/lmsys-chat-1m/MMLU 数据集的 offline profiling，发现 token-expert activation 存在强 context-independent correlation（median cumulative hotness of top-k experts: 0.833-0.976，Table 1）。基于此构建 token-to-expert confidence table C_p，将 expert 放置建模为 ILP co-clustering 问题——最小化 objective = θ·load_imbalance + (1-θ)·remote_activation，通过交替优化求解（Algorithm 1）。结果：语义相近的 expert 被 clustered 到同一 device，使得该 device 上的 token 的 top experts 大概率在本 device 上。

  **(2) Online Inter-Request Data Scheduling (Attention-DP) → 解决请求调度不考虑 expert affinity**
  将请求级调度纳入 affinity 框架：对每个请求 r，聚合其所有 token 的 device assignment → S_r = argmax Σ_i R_ij。配合 workload-aware round-robin（每 E 个请求一轮，每 rank 各一个），语义相似的请求被分配到同一 DP rank，大幅减少 all-to-all 通信。
  
  **(3) Online Intra-Request Data Scheduling (Attention-TP) → 解决 token 重复通信**
  在 post-attention reduce-scatter 中嵌入 speculative token shuffling（SRS kernel）：根据 token-to-device table T 和 inter-layer 2-gram confidence table A，预测每个 token 在下一 MoE 层的 target device → argsort 重排 token → reduce-scatter 分发。MoE 计算后通过 shuffled-allgather (SAG) 恢复顺序。将原本独立的 all-to-all dispatch 通信融入已有的 reduce-scatter 操作。

  **(4) Inter-Layer Expert-Expert Affinity Modeling → 增强 TP 场景预测精度**
  利用 2-gram Markov chain 建模跨层 device transition：Pr(D_k^(L)|D^(L-1), D^(L-2))。当 token-level confidence 低时（OOV tokens），切换到 inter-layer prediction。两表竞争选置信度高者，提升预测鲁棒性。

  **(5) 零额外修改 MoE 架构**
  Sem-MoE 不修改 MoE 模型架构（无 pre-gate module，无 router 修改），仅通过重排 gate matrix column + expert device placement + 通信原语修改实现，与现有 MoE 模型完全兼容。

  全栈执行例子（Sem-MoE Attention-DP, DeepSeek-V2-Lite, 4 DP ranks, 请求 A-D 到达）：
  - **算法Pipeline层**：同 baseline——Gate: Softmax(W_g · h) → Top-K experts。算法本身不变，但 gate matrix W_g 的 column 被 Sem-MoE 重排以匹配新的 expert placement（透明 shuffle）。
  - **系统框架层**：Sem-MoE on SGLang：
    Offline: profile → C_p → co-clustering solver → E (expert labels) + T (token labels) + A (inter-layer table)
    Online inter-request: 请求 A-D 到达 → Aggregator 查 T 表统计各 token 的 device assignment → S_r = argmax → 请求 A（代码类语义）→ DP_0（host experts for code tokens），请求 B（数学类语义）→ DP_1 → ...。Round-robin 保证每轮每 rank 各一个请求，防止解码阶段负载倾斜。
  - **编译框架层**：论文未明确说明，SGLang 原有 Triton fused MoE kernel 保持不变。
  - **Kernel调度层**：
    Attention (DP, 各 rank 独立) → MoE layer:
    1. Gate: G = Softmax(W_g · X)  （Gate column 已 shuffle，transparent to user）
    2. All-to-All Dispatch: 仅 cross-device token 参与通信（LAR 从 ~25% 提升至 ~62%）
    3. Expert FFN: 大部分 token 的 expert 在本地（LAR↑），远程通信量大减
    4. All-to-All Combine: 同样缩减
    
    (Attention-TP 场景):
    Post-attention: Shuffled-Reduce-Scatter (SRS):
      - 查 T 和 A 表 → 选置信度高者 → argsort → token shuffle → reduce-scatter
      - Token 被预送到 expert 所在 device，省去后续 all-to-all dispatch
    MoE: gate + local expert FFN (高 LAR)
    Post-MoE: Shuffled-AllGather (SAG): allgather + 反向 argsort 恢复 token 顺序
    
    调度表内存：~11.72 MB for DeepSeek-V2（int16），完全驻留 GPU memory。
  
  - **硬件架构层**：同 baseline——8-GPU server（96GB HBM/GPU，>400GB/s 互联），但 all-to-all 通信量减少，LAR 从 25% 升至 62%（DeepSeek）/68%（Qwen3），expert layer 延迟降低 41.8%/46.6%。

  关键结果：Attention-DP: Throughput ↑ 2.78× (E2E SLO vs MoETuner)、↑ 31% (TTFT SLO vs SGLang)；Attention-TP: TTFT ↓ 24.9% (Qwen3, input=512)。Cross-dataset 零样本迁移：ShareGPT 训练 → lmsys-chat-1m LAR 从 25% 提升至 41.25%（1.65× baseline），接近 in-domain 最优（47.19%）。

## Sparser Mixture-of-Adapters with Cross-Layer Generalization

- baseline方法是什么？
  - **Mixture of LoRA (MoL) / MultiLoRA**：传统 MoA 方法，每层维护独立的 N 个 LoRA adapter 专家池，由每层路由器（router）将输入 tokens 路由到该层的专家，专家的路由权重通过 token-expert 相似度或固定权重确定。每层专家之间不共享，所有 N×L 个专家始终处于激活状态（100% 利用率）。
  - 全栈执行例子（以 Phi-2 在 BoolQ 上推理为例）：
    - **算法 pipeline 层**：输入 token x 进入 layer l，该层的 router 计算 x 与该层 N=8 个 adapter expert embedding 的相似度，分配路由权重 u_n，输出为 y' = Vx + Σ_{n=1}^{N} u_n B_n^l A_n^l x。每个 adapter 只接收来自该层的 tokens 训练，不与其他层共享。
    - **系统框架层**：LoRA adapter 作为 HuggingFace PEFT 模块插入 target modules（如 q_proj, k_proj, v_proj）。每个 layer 的 adapter 参数独立存储和加载，总 trainable params = N×L×2dr（对 Phi-2: 8×32×2×2560×16 ≈ 33M，占 1.19%）。
    - **编译框架/Kernel调度/硬件架构层**：论文未明确说明，使用标准 PyTorch forward pass + NVIDIA A100 GPU 执行。
  - **核心缺陷**：Section 3 冗余分析揭示了四层冗余——(a) 同层专家之间冗余：mask 80% 同层专家性能几乎无下降；(b) backbone-expert 冗余：mask 全部同层专家（100%）仅降 ~0.14%；(c) 跨层冗余：同时 mask 多层专家性能下降极小（仅 mask all layers 才从 74.98% 降至 52.55%）；(d) 专家未充分利用：极端情况下单层专家超过全部专家。因此 baseline 的 adapter 缺乏专业化分工，未能充分利用 MoA 架构容量。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **SMOA (Sparser Mixture-of-Adapters)**：三个核心设计解决 baseline 缺陷：
    1. **跨层共享适配器池**：将 N 个 adapter 放入全局池，所有 L 层共享，每个 adapter 训练时接收来自不同层的 tokens。直接解决"跨层冗余"——通过强制共享消除每层独立训练导致的冗余。
    2. **全局路由器稀疏选择（Sparse Expert Selection）**：通过全局 router + 多数投票选出每层 top-n_l 个专家（而非全部 N 个），稀疏激活提升了专家利用率（Phi-2 利用率从 100% 降至 12.73%），解决"专家未充分利用"。
    3. **专家-冗余正则化 + Backbone Expert**：将 backbone 作为额外"专家"，通过正则项 R（式(8)）增大 backbone 路由权重 v_{l,i}，迫使 adapter 只学习 backbone 无法处理的残差知识。解决"backbone-expert 冗余"，同时促进 adapter 专业化（Figure 4 显示 SMOA 专家有明确任务偏好）。
    4. **课程学习（Specialization-to-Generalization）**：初始阶段 adapter 专注于特定层（深度专业化），逐渐允许跨层共享（泛化），平衡 specialization 与 generalization。
  - 全栈执行例子（以 SMOA + Phi-2 在 BoolQ 上推理为例，对比 baseline）：
    - **算法 pipeline 层**：
      - Baseline：每层独立计算 Δy = Σ_{n=1}^{8} u_n^l B_n^l A_n^l x，32 层共 256 个 adapter 参数独立。
      - SMOA：全局池 N=8 个 adapter，全局 router 计算所有 tokens 对 8 个 expert 的分数 w_{n,i} = softmax_n(<x_i, e_n>)，多数投票选出 top-n_l=8 个（即全选），重新归一化得 u_n，同时计算 backbone 相对适合度 v_{l,i}，输出 y' = Vx + (1 - v_l) · Σ_{n∈A_l} u_n · B_n A_n x。8 个 adapter 被 32 层共享复用，总 adapter 参数从 33M 降至仅需 8×2dr（但论文报告 trainable params 仍为 ~33.5M，与 MoL 相近，增量来自 embedding e_n 和 c_l）。
    - **系统框架层**：PEFT + PyTorch 实现，每个 adapter 的 LoRA 矩阵 B_n, A_n 全局存储一份，每层的前向计算通过全局 router 动态路由选择 adapter 子集并加权合并。训练时 curriculum learning 控制 adapter 可被哪些层选择。
    - **编译框架/Kernel调度/硬件架构层**：论文未明确说明。训练在 NVIDIA A100 上完成，wall-clock time per batch 38.54s（vs MoL 42.08s，MultiLoRA 31.85s），说明动态路由开销可控。
  - 关键结果：SMOA 在 4 个 base LLM 上全面超越 baseline：
    - Phi-2: 75.61% vs MoL 74.15%（+1.46%），adapter 利用率仅 12.73%
    - Phi-3: 82.23% vs LoRA 81.36%（+0.87%），利用率 58.75%
    - Gemma: 39.99% vs MultiLoRA 37.24%（+2.75%），利用率 60.39%
    - OLMo: 38.32% vs LoRA 36.82%（+1.50%），利用率 76.34%
    - OOD (Phi-2 MMLU): 56.19% vs MultiLoRA 55.19%（+1.00%）
    - 仅需 2 个激活专家即可达到接近 8 个专家的性能（Table 7）。

## Steering MoE LLMs via Expert (De)Activation

- baseline方法是什么？
  Baseline 是标准的 MoE LLM 推理方式：不对 router 做任何干预，模型的 top-K routing 完全由训练好的 router weights 决定。对于给定的输入 prompt，每层的 router 计算 z = W_r h，softmax 后取 top-K experts，加权求和输出。模型的行为（faithfulness、safety）完全由原始训练和 RLHF alignment 决定，推理时无法控制或调节。
  全栈执行例子（Baseline: GPT-OSS-120B 标准推理，GPU 推理）：
  - 算法pipeline层：输入 token h ∈ R^d → router 计算 z = W_r h → softmax 得概率 p → TopK(p, k=4) 选择 4/32 experts → Expert_i(h) 加权求和输出。所有 expert 的路由完全按训练好的 router weights 决定。在 RAG faithfulness 场景中，即使给定了 document context，模型可能仍然依赖 parametric knowledge（而非 document content），产生幻觉。在 safety 场景中，尽管 RLHF alignment 训练了 refusal 行为，但 alignment 集中在部分 experts 的稀疏子网络上——unsafe routing pathways 仍然存在，一旦被精心设计的 jailbreak 触发，模型仍会输出 unsafe 内容。
  - 系统框架层：标准的 HuggingFace transformers 推理，无 serving 框架修改。每个 token 前向通过所有 MoE layers，无额外控制。论文未修改任何 serving 框架。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明硬件平台。使用标准 NVIDIA GPU 集群加载 HuggingFace 模型权重进行推理。
  Baseline 的核心缺陷：(a) MoE router 被当作纯粹的计算分配机制，忽视了其作为行为控制接口的潜力——router 的 expert 选择实际编码了 behavior-specific signals；(b) RLHF alignment 训练的安全行为并非存在于所有 expert routing paths 中，而是集中在稀疏的"safe expert"子网络上——unsafe routing pathways 在 aligned 模型中依然存在，形成"alignment faking"状态（alignment concentrated in a subset of experts, neglecting alternate routing paths that can catastrophically bypass alignment when triggered）；(c) 无法在推理时动态控制模型行为——当需要对 faithfulness 或 safety 做精细调节时，baseline 无接口可操作。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法 SteerMoE 将 MoE router 重新解释为可控和可解释的行为调制机制（controllable and interpretable mechanism），而不仅仅是计算分配的工具。通过三个关键设计解决 baseline 缺陷：
  (1) Paired-Example Routing-Difference Detection：利用对比行为对（safe/unsafe, faithful/unfaithful）的 expert activation 差异，通过 Risk Difference (RD) 量化每个 expert 的行为关联强度。这直接解决了 baseline 中将 router 当作黑盒的问题——SteerMoE 揭示了哪些 experts 编码了特定的行为信号。
  (2) Inference-Time Expert (De)Activation：在推理时，通过 hook router 的 log-softmax scores，将行为关联 experts 的分数调整至 s_max + ε（激活）或 s_min - ε（去激活），然后重新 softmax 归一化执行 top-K routing。无需修改模型权重、无需额外训练。这解决了 baseline 中无法在推理时控制行为的问题。
  (3) Soft Steering Design：仅将目标 expert 的 logit 推至 max/min + ε，而非极端值（±∞），保留多 expert 加权平均结构，避免 MoE topology 坍缩。这保证了 steering 后的模型质量稳定（fluency 几乎不变）。

  全栈执行例子（论文方法：SteerMoE 在 GPT-OSS-120B 上 unsafe steering，GPU 推理）：
  - 算法pipeline层：
    (a) Detection Phase：使用 BeaverTails 构造对比对——x^(1) = "User: {Prompt} Assistant: I'm sorry, but I can't assist with that."（safe refusal），x^(2) = "User: {Prompt} Assistant: {Unsafe Response}"（unsafe compliance）。对每个 MoE layer ℓ 和 expert i，统计激活率 p_{ℓ,i}^{(1)} 和 p_{ℓ,i}^{(2)}，计算 Δ_{ℓ,i} = p_{ℓ,i}^{(1)} - p_{ℓ,i}^{(2)}。负 Δ 表示该 expert 与 unsafe 行为关联更强。top-K 负 Δ experts 作为 A^-（去激活集合，即"safe experts"被去激活），top-K 正 Δ experts 作为 A^+（激活集合，即"unsafe experts"被激活）。实际配置（Table A.2）：GPT-OSS-120B 的 Unsafe steering 去激活 100 个 experts、激活 0 个（仅去激活 safe experts 即可释放 unsafe 行为）。
    (b) Steering Phase：对每个 token h，router 输出 z ∈ R^32（32 experts per layer），计算 s = log softmax(z) ∈ R^32。对于去激活集合 A^- 中的 100 个 experts（跨层），在各自所在层执行 s_e ← s_min - 0.01，使这些 originally "safe" experts 的概率降至最低；安全相关 experts 被从路由中排除后，其余（unsafe）experts 自然承接路由。然后 p = softmax(s)，TopK(p, 4) 选择 4 experts，output = Σ p̃_i · Expert_i(h)。
    (c) 效果：GPT-OSS-120B 在 AdvBench 上从 100% safe（baseline 完全拒绝所有 harmful prompts）降至 0% safe（与 AIM jailbreak 结合后完全被攻破）。所有 jailbreak 方法均被 bypass。
  - 系统框架层：基于 HuggingFace transformers 的 hook 机制修改 router logits。具体实现：在 MoE layer 的 router forward 中插入 hook——计算 log-softmax 后应用公式 7/8，然后继续标准 MoE forward。无需修改 serving 框架。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。

  对比 baseline 的关键改进：
  - Baseline 将 router 视为固定的计算分配 → SteerMoE 将 router 视为可控的行为接口，通过对比激活统计发现行为-specific routing pathways
  - Baseline 的 safety alignment 仅存在于部分 experts 的稀疏子网络 → SteerMoE 揭示并利用了这一"alignment faking"漏洞——仅去激活 100/4608 experts 即能使完全 aligned 的模型 100% 被 jailbreak
  - Baseline 无法动态控制 faithfulness/safety → SteerMoE 提供了推理时可调的双向控制（既可增强安全又可削弱安全）
  - SteerMoE 的跨语言泛化（英文检测对发现的 safety experts 在意大利语/泰语上同样有效）表明 behavior-linked experts 编码的是行为本身而非语言特征

## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- baseline方法是什么？
  MoE 模型（如 Mixtral 8×7B、OLMoE、Llama-4-Scout）使用 HBM + GPU 的标准 serving pipeline：
  - **算法层**：Standard top-K gating (e.g., top-2)，所有 experts 在 GPU HBM 中均匀存储，无 topic-aware placement。Router 计算在 GPU 上轻量执行，expert FFN 在 GPU SM 上通过 Grouped GEMM 计算。
  - **系统框架层**：vLLM（或类似框架）做 continuous batching，请求按 FIFO 或 priority 调度，不考虑 query topic 与 expert affinity。所有 MoE expert weights 常驻 GPU HBM，decode 阶段注意力机制因 KV cache 访问成为 memory-bound bottleneck。
  - **编译框架层**：论文未明确说明，标准 PyTorch → CUDA kernel 编译路径。
  - **Kernel 调度层**：GPU SM 执行 Grouped GEMM（fused MoE kernel），每个 expert 的 tokens 通过 all-to-all dispatch/combine 在 GPU 内 HBM 上读写。Attention 的 KV cache 访问受限于 HBM bandwidth（~800 GB/s per stack），decode 阶段因批量小成为 memory-bound。
  - **硬件架构层**：NVIDIA H100/A100 GPU + HBM3 stack。HBM 通过 TSV（10μm pitch）连接 DRAM dies stack 和 base die，经由 1024-bit I/O 和 silicon interposer 与 GPU 通信。HBM 内部带宽受限于 TSV 数量，外部 bandwidth 受限于 interposer I/O。DRAM 工艺针对存储优化、不擅长逻辑计算，导致 NMP 在 DRAM die 内实现计算会面临 PPA 开销和散热挑战。
  - **芯片设计层**：HBM 通过 die stacking + TSV 互联，每 stack 6-12 层 DRAM dies，base die 通过 interposer 连接 GPU。制造 yield 低、成本高（TSV fabrication + bonding）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Stratum 通过 **系统-硬件协同设计（System-Hardware Co-Design）**，在三层（算法/系统、kernel/计算映射、硬件/芯片）同时对 baseline 进行改进：
  - **算法/系统层（Topic-Aware Expert Prediction + Placement）**：
    - Baseline 缺陷：所有 experts 在 HBM 中均匀存储，heat map 不考虑查询语义。热门 expert 和冷门 expert 访问延迟相同，内存带宽浪费在低效访问上。
    - Stratum 方法：(a) 用 DistillBERT-based topic classifier（67M params）预测查询主题 → SLO-aware scheduler 将同 topic 请求 batch 在一起 → 利用 offline profiled per-topic expert hit rate 表预测 batch 内 experts 的使用概率 → Algorithm 1 将 hot experts 放入 Mono3D DRAM 快 tier（tRCD=2.29ns）、cold experts 放入慢 tier（tRCD=22.88ns）；(b) 当切换 topic batch 时，通过 NMP 的 row-swap buffer 在 DRAM bank 内执行 expert swap，避免 traversing 高延迟的 interposer path（overhead <0.37% time, <0.03‰ energy）。
  - **Kernel/计算映射层（NMP Operator Mapping + Pipeline Optimization）**：
    - Baseline 缺陷：decode 阶段的 attention 和 MLP 因 HBM 带宽限制成为 memory-bound（GPU 计算资源利用率低）。Expert 之间无执行调度优化。
    - Stratum 方法：(a) Expert Processing——tensor parallelism 分区策略（GeMM1/2 垂直分片、GeMM3 水平分片），避免 expert weight 跨 PU 复制。所有 PUs 协作顺序执行 expert，输入 token 通过 sub-ring all-gather 复制。(b) Pipeline 优化——GeMM2 与 activation 重叠、GeMM3 reduce-scatter 与下一 expert GeMM1 重叠、weighted-sum 立即执行、输入 token 分片发送到各 DRAM channel 减少传输延迟。(c) Attention Processing——head-level parallelism 将 heads 分配到 PU groups，interleaved Softmax/MatMul pipeline 减少 latency。K/V 沿 sequence length 分片，利用 ring network 的标量交换进行全局 Softmax 归一化。
  - **硬件/芯片层（Mono3D DRAM + Hybrid Bonding + Logic Die NMP）**：
    - Baseline 缺陷：HBM 内部带宽受 TSV 数量限制（10μm pitch），外部 bandwidth 受 interposer I/O 限制。DRAM die 嵌入逻辑计算面临 PPA 开销大、散热差的问题。
    - Stratum 方法：(a) Mono3D DRAM 替代 HBM——1024 层垂直 stackable 1T1C DRAM，Cu-Cu hybrid bonding（1μm pitch, 5× denser than HBM TSV），内部带宽 19-34 TB/s（远超 HBM ~800 GB/s）。(b) Logic Die NMP——7nm 专用逻辑 die 通过 hybrid bonding 与 Mono3D DRAM die 直接互联，避免 DRAM process constraint 和 TSV 带宽瓶颈。128 TFLOPS peak compute, 16 PUs, on-chip ring network。(c) In-Memory Tiering——利用 Mono3D DRAM 的 WL staircase 延迟变化（tRCD 2.29-22.88ns），定义 8 个 memory tiers，通过 per-PE tiering table 动态控制 tRCD，快 tier 1.6× faster than slow tier。(d) CMOS-Under-Array (CUA)——高电压 DRAM 外围电路在 32nm CUA 层实现，低电压逻辑在 7nm logic die 实现，thermal modeling 确保 45W/chip 功率预算（vapor chamber + liquid cooling）。
  - **全栈执行例子（对比 Baseline）**：
    - Baseline（vLLM + H100, Mixtral 8×7B, decode one token）：embedding → attention（Q@K^T from HBM KV cache, ~800 GB/s bottleneck）→ gate routing → expert selection → load expert weights from HBM → GPU SM Grouped GEMM → weighted sum → output projection → next token。整个 decode loop 受 HBM bandwidth 约束（~800 GB/s per stack, ~3.35 TB/s 8 stacks aggregate）。
    - Stratum（Mono3D DRAM NMP, Mixtral 8×7B, decode batch）：xPU routing → prefill on xPU → send tokens + routing to Mono3D DRAM → Topic classifier predicts topic batch → Algorithm 1 places hot experts in fast tier → NMP 执行 expert computation with 19-34 TB/s internal bandwidth → ring network all-gather input tokens → sequential expert GEMM with pipeline overlap → attention on NMP with head-level parallelism → KV cache in intermediate tier → output tokens → xPU retrieve。整个流程的 memory 访问在 Mono3D DRAM 内部以 19-34 TB/s 带宽完成，仅 token I/O traverses interposer（1024-bit @ 6.4 Gbps = 819 GB/s），但大部分计算数据在 Mono3D DRAM-NMP 内部闭环，实现 8.29× throughput 和 7.66× energy efficiency 提升。
  - **关键设计决策映射**：
    | Baseline 缺陷 | Stratum 设计 | 层次 |
    |---|---|---|
    | Expert 访问延迟统一，无语义优化 | Topic classifier + hot/cold tier placement | 算法/系统 |
    | HBM 带宽不足（decode memory-bound） | Mono3D DRAM internal bandwidth 19-34 TB/s | 芯片设计 |
    | TSV 限制内部带宽 | Hybrid bonding 1μm pitch, 5× denser | 芯片设计 |
    | DRAM die 内计算 PPA 差 | 7nm logic die (128 TFLOPS) | 硬件架构 |
    | Expert swap 需 travers GPU-HBM | Near-memory row-swap buffer | Kernel调度 |
    | GPU SM 计算-通信串行 | Ring network + pipeline overlap | Kernel调度 |
    | 无 memory tier 概念 | 8-tier Mono3D DRAM (tRCD 2.29-22.88ns) | 硬件架构 |

## Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging

- baseline方法是什么？
  Baseline 方法包括两类：(1) **Expert Pruning**——Frequency-prune（按 router activation frequency 剪掉低频 expert）、Output-prune（剪掉输出范数最小的 expert）、NAEE（最小化 pruning error）、MoE-I²（遗传搜索）、SEER-MoE（regularization-based fine-tuning）；(2) **Expert Merging**——MC-SMoE（合并 routing policy 相似的 expert）、HC-SMoE（hierarchical clustering 合并）、EEP（进化搜索优化融合矩阵）。所有现有 merging 方法的共同缺陷：使用简单的加权平均（W_merged = Σ α_i W^(i)）直接合并 expert 权重，但由于 MoE 的 routing 机制故意训练出参数空间高度发散的 expert（inter-expert cosine similarity 仅 0.1~0.3），直接合并会引发 catastrophic parameter conflicts，导致性能严重退化。MC-SMoE 和 HC-SMoE 在 50% expert reduction 时性能退化严重（如 Mixtral-8→4 时 HC-SMoE 平均准确率仅 0.51 vs 原始 0.67），MC-SMoE 甚至完全崩溃（PPL > 854）。
  全栈执行例子（Baseline: HC-SMoE on Mixtral-8×7B → 4×7B，8× H800）：
  - **算法pipeline层**：对 8 个 expert 做 hierarchical clustering（基于权重的 cosine distance），将 8 个 expert 分成 4 组 → 每组内 expert 权重做 simple averaging：W_merged = (1/|Q|) Σ W_i → 4 个合并后的 expert 替换原 8 个。输入 token x → Router top-2 选择 2 个 merged expert → FFN 计算。由于原 expert 的权重空间高度不同（W_i 中同一位置的参数可能代表完全不同的特征方向），simple averaging 相当于对不同坐标系的向量直接相加，导致合并后的 W_merged 失去所有 expert 的 specialized knowledge。合并后 WikiText-2 PPL 从 3.98 升至 9.88，MMLU 从 0.67 降至 0.39。
  - **系统框架层**：Transformers / vLLM 推理框架，加载原始 checkpoint 或压缩后 checkpoint，forward 过程中 router 做 top-k expert selection，选中的 expert FFN 逐层执行 up-proj → SiLU → gate-proj → down-proj。压缩后模型参数量从 46.7B 降至 24.2B，GFLOPs 从 2989 降至 1546。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。NVIDIA H800 GPU（80GB × 8），标准 GEMM kernel 执行 expert 计算。
  Baseline 的核心缺陷：参数冲突问题——MoE 中不同 expert 被 router 训练出高度专业化的参数空间（inter-expert similarity 0.1~0.3），而现有 merging 方法（加权平均、clustering averaging）对不同坐标系的参数直接求和，破坏了每个 expert 的专业化知识。此外，post-merging fine-tuning（如 D²-MoE 的 delta compensation）虽然能部分恢复性能，但计算开销大，削弱了 merging 的效率优势。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法（Sub-MoE）通过**子空间对齐**解决参数冲突：(1) **Adaptive Expert Clustering**——基于 expert 输出的 cosine similarity（而非参数相似度）用 K-means 对 expert 分组，确保同组 expert 功能一致；(2) **Subspace Expert Merging**——对同组 expert 权重拼接后做联合 SVD，提取共享 U-matrix（正交基）作为共同坐标系，仅在 V 矩阵上做 frequency-weighted merging，利用 Σ 保留重要维度。核心思想：将不同坐标系中的 expert 投影到共享子空间（U），在同一坐标系下合并 V，避免参数冲突。
  全栈执行例子（Sub-MoE on Mixtral-8×7B → 6×7B，8× H800）：
  - **算法pipeline层**：(a) Calibration: 从 WikiText-2 采样 128 条 × 2048 token → forward pass 收集每个 expert 的输出 Y_i（而非权重），计算 pairwise cosine similarity Sim(E_i, E_j) = mean(cos(Y_i(x), Y_j(x)))。(b) Adaptive Clustering: 2-layer K-means（16 experts per group）聚类 → 将功能相似的 expert 分入同组。Multi-layer adaptive allocation 在目标压缩比下自动确定每层 k 值。(c) Union SVD: 对同组 expert 的权重垂直拼接 → W_cat = [W^(1); W^(2); ...] ∈ R^{O×nI} → SVD(W_cat) = U Σ [V^(1); ...; V^(n)]^T → U ∈ R^{O×r} 为共享正交基（语义："共同坐标系"），Σ 为重要性权重，V^(i) 为每个 expert 在该坐标系中的坐标。(d) Frequency-based V-Merging: 统计每个 expert 的 router activation frequency f(V_i) = Σ 1[i ∈ top-k(G(x))] / |X| → V_merged = Σ f(V_i)·V_i / Σ f(V_i)（高频 expert 贡献更大）。(e) Reconstruction: W_merged = U Σ V_merged^T ∈ R^{O×I} → 替换组内所有 expert。若需 intra-expert compression（Sub-MoE†），则用输入激活的 whitening matrix S_i 对权重重加权（W'_i = W_i S_i），再做 truncated SVD 截断 Σ 中最小的奇异值，控制 U 和 V 的秩。
  - **系统框架层**：PyTorch + Transformers 加载模型 checkpoint → CPU/GPU 上执行聚类和合并（纯后处理，无需 GPU training）→ 输出压缩后的 checkpoint。合并操作的计算瓶颈在 SVD（O(min(O, nI)² · max(O, nI))），但仅执行一次，与模型大小相比开销可忽略。压缩后模型在 Transformers/vLLM 上推理：router top-k expert selection 不变，选中的 merged expert 的 FFN 正常执行。若合并为 6 个 expert，模型从 46.7B → ~31.4B params，GFLOPs 从 2989 → ~2319。
  - **编译框架/kernel调度/硬件架构层**：论文未明确说明。NVIDIA H800 GPU（80GB × 8）。推理时使用标准 PyTorch GEMM kernel，无定制 kernel 优化。论文提供运行时吞吐数据：Sub-MoE† 在 30% intra-expert 压缩下可达 1.38× 吞吐加速（87.7→120.9 tok/s）。
  方法 vs Baseline 对比核心差异：
  - **子空间对齐 vs 直接平均**：Union SVD 将 expert 从各自参数空间投影到共享坐标系（U），消除坐标系不一致导致的参数冲突 → Mixtral-8→6：WikiText-2 PPL 5.16 vs HC-SMoE 5.92（↓12.8%），Mixtral-8→4：average accuracy 0.58 vs HC-SMoE 0.51（↑13.7%）
  - **功能聚类 vs 参数聚类**：基于 expert 输出的 cosine similarity 聚类（而非权重 cosine distance），捕捉功能行为而非参数结构的相似性 → 相比 random clustering 提升 accuracy 0.60→0.64
  - **Frequency-weighted merging vs uniform averaging**：高激活频率的 expert 处理更多常见模式，在合并中赋予更大权重保留其功能 → 相比 average merging 提升 accuracy 0.62→0.64
  - **零额外训练 vs post-merging fine-tuning**：Sub-MoE 完全不需要 retraining、fine-tuning 或 searching → 对比 D²-MoE（需要 delta compensation fine-tuning），Sub-MoE†+FT 在 ARC-e 上超出 6%、WinoGrande 超 5%、ARC-c 超 6%
  - **方法局限性**：依赖 calibration 数据集进行 clustering 和 merging（论文声明 future work 探索 data-free 方式）；仅针对 expert merging 而非 weight compression；SVD 在大规模 MoE（如 Qwen3-128 experts）上的计算开销需进一步评估


## S'MoRE Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning

- baseline方法是什么？
  **Baseline: MoLRE (Mixture of Low-Rank Experts, 即 MixLoRA) 和 MoMOR (Mixture of Multi-Order Residues)**
  
  LoRA 将预训练权重的更新限制在低秩空间 ΔW = B·A（B∈R^{d×r}, A∈R^{r×d}），参数高效但模型容量受限于单个低秩矩阵。MixLoRA（MoLRE）将其扩展为 x' = Σ_{i=1}^s ROUTE(x)^i · B^i · A^i · x，即多个低秩专家的加权组合。然而这种"扁平"结构存在两个缺陷：(1) 路由灵活性有限——每个 token 仅从 s 个专家中选 k 个，总路由组合数为 C(s,k)，增加专家数虽然能提高灵活性，但会导致专家利用不均和路由开销增大；(2) 结构无关性——同一组被激活的专家无论按何种结构连接，都输出相同的结果（因为等价于简单的加权求和），这意味着模型的表达能力完全取决于"激活哪些专家"，而无法从"如何连接专家"中获益。
  
  全栈执行例子（以 MixLoRA 推理一个 token 为例）：
  - 算法层：输入 token embedding x(4096d) → Router 计算 ROUTE(x)^i = softmax(W_gate·x) → top-k 选择 2 个专家 → 并行执行 B^i·A^i·x（8个秩16矩阵）→ 加权求和输出 x'(4096d) → 加到 pre-trained FFN 输出。整个过程是"选择+线性组合"，无结构化信息。
  - 系统框架：论文未明确说明（使用标准 PyTorch 训练框架 LLaMA-Factory，无自定义 serving 修改）
  - 编译框架：论文未明确说明
  - kernel调度：论文未明确说明（标准 PyTorch CUDA kernel 执行矩阵乘法）
  - 硬件架构：论文未明确说明

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **S'MoRE: Structural Mixture of Residual Experts**
  
  S'MoRE 的核心创新是将"扁平专家选择"升级为"分层结构化专家组合"。它没有增加专家数量，而是通过将专家组织成多层结构并利用非线性层间传播，使得同一组参数可以形成指数级的不同前向路径（非树形结构），从而极大提升模型容量。具体设计对应 baseline 的两个缺陷：
  
  **缺陷1"路由灵活性有限"→ 解决：树形分层路由**
  - 将 s 个专家按 L 层排列，路由器自顶向下逐层条件化选择：P(child | parent_ancestors, x) = softmax(⟨k_child, MLP(concat(x_down, ancestor_keys))⟩)
  - 每层 fanout f_ℓ，总激活节点 F_ℓ = Π_{i=ℓ}^{L-1} f_i
  - 1层 MixLoRA 路由组合数 ≤ C(s, k)；2层 S'MoRE 在相同参数下结构灵活性 Γ = Π_ℓ C(s_ℓ, f_ℓ)^{F_{ℓ+1}}（组合数作为指数的指数级增长）
  
  **缺陷2"结构无关性"→ 解决：非线性层间聚合 + GIN 启发的传播**
  - 每层聚合 x_{ℓ+1}^i = Σ α · σ(B·A·x + W·x_prev)，其中 σ 是非线性激活（ReLU）
  - 理论证明：无 σ 时，多层的线性组合可坍缩为等价单层（MoMOR）；加入 σ 后，L 层传播模拟 L 轮 WL 同构测试，不同树结构的输出不同
  - 图3示例：同一组激活专家 {"0,1","0,2","0,3","0,4","1,1","1,2"} 按三种不同树结构连接（非树形）→ MixLoRA/MoMOR 输出相同 → S'MoRE 输出三种不同结果
  
  全栈执行例子（以 S'MoRE L=2 推理一个 token 为例）：
  - 算法层：输入 x(4096d) → [路由阶段 自顶向下] 层2 Router MLP_1(concat(x_down)) → softmax(⟨k_1^i, q_1⟩) → top-2 选择专家 "0,1"和"0,2" → 对于每个选中父节点，层1 Router MLP_0(concat(x_down, k_parent)) → 分别选子节点（"0,1"选{"1,1","1,2"}，"0,2"选{"1,3","1,4"}）→ 构建两棵残差树 → [聚合阶段 自底向上] 层1：对每组子节点计算 σ(B_0^n·A_0^n·x)（带 skip connection 和 ReLU），求和生成 x_1^{parent} → 层2：对两个 x_1 输出计算 W_1·x_1 + B_1·A_1·x，经 ReLU 后求和 → 最终 W_proj 映射回 4096d。路由选择的树结构不同，即使专家集合相同，非线性传播路径不同，输出也不同。
  - 系统框架：LLaMA-Factory 训练框架，OpenCompass 评测框架。adapter 插入 FFN 和 attention 模块，与 LoRA 使用方式一致（论文未修改 serving 框架）
  - 编译框架：论文未明确说明
  - kernel调度：标准 PyTorch 矩阵乘法，论文未涉及自定义 kernel
  - 硬件架构：论文未明确说明（NVIDIA GPU 训练，具体型号未披露）
  
  关键理论保证：Theorem 3.4 证明 S'MoRE 的结构灵活性 Γ_{S'MoRE} = Π C(s_ℓ, f_ℓ)^{F_{ℓ+1}}，在 s_ℓ=4, f_ℓ=2, L=2 时远超 MoMOR 的上界 Γ_{MoMOR} ≤ C(s, f)。本质原因是：MoMOR 中激活专家组合数仅与选择相关（线性累加 C(s,i)），而 S'MoRE 中每个节点独立选择子节点，且非线性传播使不同树结构输出可区分，因此灵活性按"每个节点的组合选择"的乘积指数增长。

## The Omni-Expert: A Computationally Efficient Approach to Achieve a Mixture of Experts in a Single Expert Model

- baseline方法是什么？
  Baseline 方法为 Phoneme-based Mixture-of-Experts (MoE) 模型用于 CI 语音去混响。核心架构：40 个独立专家网络（每个对应一个音素类）+ 一个音素分类器作为门控网络。每个专家网络仅在对应音素组的数据上训练，门控网络输出 40 维概率向量，40 个专家的输出按概率加权求和得到最终 T-F mask。核心缺陷：(1) **计算成本随专家数线性增长**——在 CI 实时处理场景中，40 个专家意味着 40 倍以上的参数量（4.33M vs 108K）和 MACs（4377.6M vs 109.44M），远超资源受限的边缘设备（人工耳蜗声音处理器）承受能力；(2) **每个专家训练数据量小**——数据按音素划分后每个专家只看到约 1/40 的训练数据，导致训练收敛慢（MoE 训练 5h22m vs PI 模型 2h58m）；(3) **必须训练和存储所有专家**——即使稀疏 MoE 技术激活部分专家，完整的专家集合仍需训练和存储。传统 MoE 方法（稀疏 MoE、专家合并）未从根本上消除多专家架构的冗余。

  全栈执行例子（Baseline: Phoneme-based MoE, Titan V GPU）：
  - **算法Pipeline层**：输入 65 维 log-compressed 频谱 x → 音素分类器（LSTM/GRU+A → FC_40 sigmoid）输出 40 维概率 p → 并行运行 40 个专家网络（各自为 LSTM→FC→65 sigmoid）→ 输出加权：M_hat = Σ p_n * y_n → 增强语音 S_hat = M_hat ⊙ X。40 个专家互不共享参数，推理时需要执行 40 次完整前向传播。
  - **系统框架层**：PyTorch 实现。无 serving 框架修改。模型以 Python .pt 文件部署，需要加载 40 个完整专家网络到 GPU 显存。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 PyTorch GEMM 和 LSTM kernel，无自定义 kernel。推理时每个音素帧执行 40 次 LSTM cell forward + 40 次 FC forward。
  - **硬件架构层**：NVIDIA Titan V GPU（12 GB HBM2）。CI sound processor 端目标硬件论文未明确说明（论文指出"CI processor chip technology is expected to improve over time"）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 Omni-Expert (OE)，核心思路：**将"多专家"替换为"单专家 + 音素特定的输入特征变换"**，在特征空间中隐式编码子任务选择。具体设计：(1) 使用单一共享专家网络处理所有音素，而非 40 个独立网络，彻底消除专家数量带来的参数/计算膨胀；(2) 对输入特征施加音素特定的仿射变换 z_n = a_n ⊙ x + b_n（对角线尺度 + 偏移），在特征空间中将不同音素映射到不同区域实现专业化（t-SNE 可视化验证了变换增强音素簇内聚性和簇间分离性，Figure 5）；(3) 变换参数 a_n 和 b_n 由两个小 MLP 从 one-hot 音素编码预测，训练后预计算存入查找表，推理时无额外网络开销；(4) 单专家网络在全部训练数据上训练，充分利用数据量优势（每专家看 ~28h 全量数据 vs MoE 每专家看 ~0.7h）。这直接解决了 baseline 的三大缺陷：参数量从 40N 降至 N（消除计算膨胀）、训练数据量从 1/40 升至全量（加速收敛）、只需存储一个专家网络（消除存储冗余）。

  全栈执行例子（OE, Titan V GPU）：
  - **算法Pipeline层**：输入 65 维 log-compressed 频谱 x → 音素分类器输出 40 维概率 p → 对 n=0..39：查表得 a_n, b_n（65 维，预计算）→ z_n = a_n ⊙ x + b_n → 共享单专家网络 forward(z_n) → y_n → 最终 M_hat = Σ p_n * y_n → S_hat = M_hat ⊙ X。虽然仍需 40 次 expert forward（与 MoE 相同的 forward 次数），但每次 forward 用的是同一组参数——本质差异是：MoE 需要存储和加载 40 套参数，而 OE 只存储 1 套参数 + 40×2×65 个标量查找表值。OE 额外计算仅为 40 次逐元素乘法/加法（a_n ⊙ x + b_n），相比一次 LSTM/GRU 前向可忽略。
  - **系统框架层**：PyTorch 实现。无 serving 框架修改。推理部署只需加载单个 0.45MB 模型（vs MoE 16.51MB）。训练代码仅需一个专家网络的数据加载器，大幅简化工程复杂度。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 PyTorch kernel。推理时与 MoE 相比：省去 39 次独立专家网络的 kernel launch 和参数加载（共用同一组参数），但增加 40 次逐元素乘加。总计算量从 4377.6M MACs 降至 109.45M MACs（LSTM 变体，1/40）。
  - **硬件架构层**：NVIDIA Titan V GPU。论文指出 OE 的方法更实用（practical for CI deployment），为在 CI 片上处理器部署创造了可能性（芯片技术持续进步）。

  消融关键发现：(a) **尺度+偏移 > 单一变换**——仅尺度或仅偏移均优于无变换，但两者组合效果最好（SRMR-CI: Scale+Shift=1.794 vs Scale Only=1.706 vs Shift Only=1.711 vs None=1.683），因为尺度增强簇间分离性而偏移调整中心对齐；(b) **变换位置：输入层优于隐藏层**——在输入层施加变换（I: SRMR-CI=2.014）远优于仅在隐藏层（H: SRMR-CI=1.367），因为输入层变换能影响后续所有层的计算，而隐藏层变换受限于已编码的表征；(c) **OE 在理想音素知识下上界更高**——OEk SRMR-CI=2.113 远超 MoEk=1.945，说明特征变换编码的子任务专业化比分区训练的独立专家网络更有效；(d) **噪声鲁棒性**——在未见噪声条件下（训练仅含混响），OEk 仍优于 MoEk，证明学习到的子任务特征变换具有更好的泛化性。

## Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony

- baseline方法是什么？
  Baseline 是标准 Expert Parallelism (EP) MoE serving，以 SGLang [49] 为代表性系统。Baseline 的执行模式：
  - **全栈执行例子**：请求到达 → tokenizer → 整 batch 同步执行 decoding block 0 attention → **barrier all-to-all** → expert GPU 并行执行 expert layers → **barrier all-to-all** → block 1 attention → ... → block N → sampler → detokenizer。所有 GPU 在每个 barrier 等待最慢的 expert 完成（straggler effect）。
  - **Baseline 的缺陷**：
    1. **GPU stall on straggler experts**：hot expert 接收最多 tokens，计算时间显著长于 cold expert，其他 GPU 被迫空等。实验中 GPU stall 可达总时间的 70%（Figure 4）。
    2. **Cold expert 小 batch 低效执行**：cold expert 在每次 barrier 前只有少量 tokens，batch size 小 → HBM weight loading 时间主导 → GPU 计算单元利用率低。Figure 3 显示 batch < 128 时 throughput 远低于线性。
    3. **Barrier all-to-all 通信开销固定**：无论 load 如何偏斜，all-to-all 必须在全部分参与 GPU 间同步，无法通过增加 GPU 缓解。
    4. **无法利用动态负载变化**：expert load skew 随时间变化 [11,21,23,31]，固定 batch 执行无法自适应。
  - 缺失层次：编译框架（论文未涉及编译框架层）；kernel 调度（使用 NCCL + vLLM 现有 kernel，未提出新 kernel）；硬件架构/芯片设计（使用商用 A100 GPU）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Asynchronous Expert Parallelism (AEP)** 并通过 **AMoE** 系统实现，核心设计解决 baseline 各缺陷：
  
  - **全栈执行例子（AMoE）**：
    1. 请求到达 → tokenizer (API Server/CPU) → Load Balancer 选 attention DP rank
    2. Token 携带 metadata 发送到 attention GPU → Receptor 按 LayerID 入对应 block 的 µ-queue
    3. Defragging Scheduler 计算所有 layer 的 Score → 选最高分 layer 执行（无 barrier）
    4. Executor 对选中 layer：page table (attention) / GEMM (expert) → Dispatcher 按目标 GPU permute tokens
    5. Communicator: ZeroMQ (CPU) metadata 交换 → NCCL P2P (GPU) 异步传输（CPU 不等待 NCCL 完成）
    6. 循环 step 3-5，各 GPU 独立异步执行 → sampler → detokenizer
  
  - **解决 baseline 缺陷的对应设计**：
    1. **消除 straggler stall（对应缺陷 1）**：AEP 完全去除 all-to-all barrier。GPU 完成当前层后立即从 µ-queue 拉下一批 token 执行任意 ready layer，不存在"等待最慢 expert"。Hot expert GPU 持续处理积累的 tokens，cold expert GPU 可转而执行 token 充足的其他 layer。
    2. **自适应 batch size（对应缺陷 2）**：µ-queuing 允许 cold expert 的 tokens 在队列中积累，直到 batch size 足够大（接近高效的 batch=128 区域），由 defragging scheduler 的 Score 机制自动延迟低 token 计数的 layer 的调度。
    3. **异步 P2P 通信替代 all-to-all（对应缺陷 3）**：用 ZeroMQ + NCCL P2P 的异步点对点传输替代全局 barrier all-to-all，发送方 CPU 启动 NCCL 后立即继续处理下一个任务，接收方异步同步。通信自然与计算重叠。
    4. **动态自适应负载（对应缺陷 4）**：Defragging Scheduler 每步基于当前 µ-queue 状态动态决策最优 layer，lookahead 机制鼓励 token wave 向前传播并保持 compact，无需 profiling hot/cold experts 或预分配 expert replica。
  
  - **额外的系统级创新**：
    - **Attention-Expert 解耦**：attention 和 expert 层可独立扩展 GPU 数量（e.g., 4 attention GPUs + 4 expert GPUs），解决 KV cache 容量瓶颈限制并发请求数的问题。
    - **FLFS/MTFS 平衡调度**：在完全 defrag (FLFS) 和纯 throughput (MTFS) 之间折中——通过 lookahead decay δ 控制，避免 FLFS 的新请求 live-lock 和 MTFS 的 batch fragmentation。
    - **C++ 关键路径优化**：Receptor, Scheduler, Communicator, Dispatcher 用 C++ 实现（pybind11），避开 Python GIL，scheduling overhead 仅占执行时间的小部分（Figure 13）。
  - 缺失层次：编译框架（论文未涉及编译框架修改）；kernel 调度（使用 vLLM/NCCL 现有 kernel，未提出新 kernel 算子）；硬件架构/芯片设计（使用商用 A100 GPU）。

## Toward Efficient Inference for Mixture of Experts

- baseline方法是什么？
  Baseline 是 Fairseq 实现的 MoE Transformer 推理，采用 **Static Gating + Expert Parallelism**（基于 GShard [2] 和 ELSLM [8]）。核心设计：(1) static gating 函数为每个 expert 预分配固定容量 C（capacity factor），通过 batch matrix multiplication 构建 dispatch mask 进行 token 分配；(2) 使用 NCCL all-to-all 进行固定大小消息的 token 分发和收集（每个 GPU 预知消息大小）。
  
  Baseline 的核心缺陷：
  1. **Computation waste from placeholders**：Static gating 为每个 expert 预设 capacity C，当实际 token 数少于 C 时需填充 zero placeholder。LM（E=512, C=0.05, top-2 gating）的 waste factor 为 12.8×（实际只需 2S tokens，但要计算 25.6S tokens）。MT（E=128, C=1, top-4 gating）的 waste factor 为 64×（实际只需 4S tokens，但要计算 128S tokens）。
  2. **Large memory from dispatch mask**：Batch matmul 构建的 dispatch mask 维度为 (E, S, S×C)，需要大量 GPU 临时内存。论文 memory trace 显示 gating 和 reordering 阶段有瞬时内存尖峰。
  3. **Token dropping risk**：当负载不均衡时，超出 capacity 的 tokens 会被丢弃（只保留 residual connection），损失模型质量。
  4. **Full expert parameter loading**：所有 experts 参数必须常驻 GPU 显存，即使推理中大部分 expert 很少被激活。LM 单 GPU 需 18.9GB，dense 只需 2.2GB。
  5. **Load imbalance**：MoE 在训练时的 token 分布与推理时不同，导致某些 GPU 负载过高（oversubscribed, OOM 风险），某些 GPU 空闲。

  全栈执行例子（Baseline, Fairseq static gating, 单 node 8×V100, LM task）：
  - **算法Pipeline层**：输入 tokens X ∈ R^{8×1024} → gate_linear → top-2 gating 选择 2/512 experts → 为每个 expert 构建 dispatch mask M_e ∈ R^{8×8×25.6=204.8}（多为此维度，大量 zeros）→ batch matmul M @ X（92% FLOPs 为 ×0）→ dispatched tokens per expert → 每个 expert FFN forward → batch matmul reorder → next layer。
  - **系统框架层**：Fairseq (PyTorch) MoE Transformer，NCCL all-to-all 通信（固定消息大小），expert parallelism 跨 8×V100 GPU 分配 512 experts（每 GPU 64 experts）。Expert 参数常驻 GPU 显存。batch size 固定为 8（LM）或 48（MT），受限于显存。
  - **编译框架层**：论文未明确说明（使用 PyTorch JIT 或 eager 模式，未修改编译框架）。
  - **Kernel调度层**：NCCL all-to-all（cudaMemcpy），PyTorch batch matmul（cuBLAS/cuDNN），标准 PyTorch MLP forward（cuBLAS GEMM）。Batch matmul 中 92.2% 计算为 ×0。
  - **硬件架构层**：NVIDIA Tesla V100 (32GB HBM2, NVLink)。CPU: Intel Xeon E5-2698 v4。CPU-GPU: PCIe 3.0 16GB/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出三个正交的 MoE 推理优化技术，从算法、内存管理、负载均衡三个维度解决 baseline 缺陷：

  **1. Dynamic Gating（对应缺陷 1, 2, 3）**
  - 将 static gating 的 batch matmul dispatch 替换为 argsort + bin-count + indexing
  - 复杂度从 O(S²EDC) 降至 O(SD + S log S)
  - 每个 expert 的容量动态设为实际收到的 token 数（不丢 token）
  - 消除 dispatch mask（省内存）和 placeholder 计算（省 FLOPs）
  - 两轮 all-to-all：第一轮通知 sizes（20µs avg），第二轮传可变大小 tokens

  **2. Expert Buffering（对应缺陷 4）**
  - 利用 expert 激活的 temporal locality：仅将热 expert 留在 GPU cache，其余缓存在 CPU
  - LIFO eviction policy（适配 MoE 按 expert ID 顺序执行的特性）
  - 异步 CPU→GPU 参数拷贝，与 token all-to-all 传输重叠
  - Cache miss rate 接近理论最优 Belady's MIN
  - 减少 static GPU memory 达 1.47×

  **3. Load Balancing（对应缺陷 5）**
  - Greedy Balancing：按 expert 历史平均负载排序，贪心分配到最少负载 GPU
  - Anti-Correlation Balancing：针对 decoder 场景 expert 激活相关的情况，在负载估计中加 Pearson 相关系数惩罚
  - 约束每个 GPU 等量 experts，平衡内存和通信
  - 减少 Max load（OOM 风险）和 Avg-Max load（性能退化）

  全栈执行例子（Ours, Dynamic Gating + Expert Buffering + Load Balancing, 单 node 8×V100, LM task）：
  - **算法Pipeline层**：输入 tokens X ∈ R^{8×1024} → gate_linear → top-2 assignments → **argsort** by expert ID (O(S log S)) → **advanced indexing** X[sorted_idx] 重排 token (O(SD)) → **bincount** 计算每个 expert 的实际 token 数 → all-to-all round 1（通知 sizes, ~20µs）→ **split** 按 sizes 切分 → all-to-all round 2（仅传实际 tokens，无 placeholder）→ 各 GPU expert FFN forward（仅计算实际收到的 tokens）→ all-to-all collect → indexing 还原顺序。差异：无 dispatch mask 分配，零 placeholder FLOPs，work factor = 1×（实际需要 = 实际计算）。
  - **系统框架层**：基于 Fairseq (PyTorch) + 开源代码 https://github.com/hyhuang00/moe_inference。Expert Buffering 在 Fairseq MoE forward 前插入 cache check/cudaMemcpyAsync。Load Balancing 在推理前运行 profiling pass 收集 activation 数据，调用 Greedy/Anti-Correlation 算法优化 expert placement。支持可变 batch size（从原始 8 扩展到 64-96）。
  - **编译框架层**：论文未明确说明（基于 PyTorch eager execution，未修改编译层）。
  - **Kernel调度层**：argsort（GPU radix sort kernel）、bincount（GPU reduction kernel）、advanced indexing（GPU gather kernel，O(SD) memory BW bound）、NCCL all-to-all（可变大小）、cudaMemcpyAsync（PCIe stream 与 NCCL stream 并发）。与 baseline 的关键差异：用 indexing 替代 batch matmul，消除 92.2% 浪费计算。
  - **硬件架构层**：NVIDIA Tesla V100 (32GB, NVLink)，NVIDIA RTX A5000 (24GB, Ampere)。CPU-GPU PCIe 带宽是 Expert Buffering 的瓶颈（12GB/s peak），论文指出新技术（如 Grace Hopper）可缓解。

  性能结果（摘要）：
  - Dynamic Gating vs Fairseq static: LM throughput +6.21× (single-node), +11.55× (multi-node)
  - vs Megablock: batch=80 时 1.46× faster（因 dense matmul 优于 BCSR sparse matmul）
  - MT-decoder throughput: +5.75× encoder, +2.58× decoder
  - Dynamic memory (activations): LM -79.6% (6.29→1.28GB), MT -44.2% (1.89→1.05GB)
  - Expert Buffering: static memory -1.47× (~2.25GB)
  - Load Balancing: throughput +1.19× (Greedy, LM multi-node)
  - 允许更大 batch size: LM 8→64, MT 48→96

## Toward Inference-optimal Mixture-of-Expert Large Language Models

- baseline方法是什么？
  传统的 dense Transformer scaling law（Kaplan et al., 2020; Hoffmann et al., 2022）仅考虑 validation loss L(N, D) 与训练成本 C(N, D) 的关系，求解 $argmin L(N,D) \text{ s.t. } FLOPs(N,D) = C$ 得到 loss-optimal 配置。当扩展到 MoE 时，由于增加 expert 数量几乎不增加训练 FLOPs，从 loss-optimal 视角应无限扩展 expert 数量（直到饱和 E_max），但这在推理阶段会严重增加显存占用（更多 expert 参数挤占 KV-cache 可用显存），导致 batch size 下降、throughput 降低、cost per query 上升。现有 MoE scaling law（Clark et al., 2022）没有纳入训练数据量 D 的影响，无法给出具体的预算分配建议；Sardana & Frankle (2023) 虽然考虑了推理成本，但用恒定 MFU 估算，与实际 profiling 差异可达 10×。
  
  Baseline 全栈执行例子（loss-optimal 32-expert MoE，传统 scaling law 指导）：
  - **算法Pipeline层**：遵循 Hoffmann et al. 的 dense scaling law → 确定 loss-optimal (N_opt, D_opt) → 固定到所有 MoE 变体，不考虑 E 的影响 → 如果从训练成本看 E 越大越好，选择 E=32 → 推理时模型总参数 N_MoE = (1 + (32-1)*1/3) ≈ 11.33× N_dense，所有 32 个 expert 必须加载到 GPU 显存。
  - **系统框架层**：vLLM 部署 E=32 MoE → 每个 token 经 Top-2 gating 路由到 2 个 expert → 8×A100(40GB) 共 320GB 显存，但 32 个 expert 参数挤占后仅剩余小部分给 KV-cache → 最大 batch size b 极小 → throughput 低 → cost per token 高。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（使用 vLLM 默认 GEMM kernel）。
  - **硬件架构层**：8×40GB A100 GPU + NVLink。推理瓶颈：expert 参数占显存 → KV-cache 可用空间被压缩 → decode 阶段 batch size 受限（decode 是 memory-bound，batch size 直接影响 GPU 利用率）→ 更多 GPU 需求（单卡装不下）→ 更多通信开销（EP all-to-all）。

  Baseline 的核心痛点：**loss-optimal MoE 最大化 expert 数量导致推理成本不可控**，且传统 scaling law 无法提供同时考虑训练和推理的 budget allocation 建议。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法分为三步，层层推进：
  
  **Step 1: 建立包含 E 的 MoE Scaling Law（Section 3）**
  在 Kaplan/Hoffmann 的 dense scaling law 基础上，将 expert 数量 E 作为第三个独立变量纳入公式（公式 4），并引入 $E_{start}$ 和 $E_{max}$ 参数建模 expert 增长的饱和效应（与 Clark et al. 不同，本公式同时包含 N、D、E 三个变量且有 N-E 交互项）。通过在 100M-730M dense 模型上训练 4/8/16/32 expert 版本（SlimPajama 2.5B-20B tokens）拟合参数，RMSLE = 3.908e-3，Huber loss = 1.033e-3。
  
  **Step 2: 引入推理成本约束（Section 4）**
  在 8×40GB A100 + NVLink 上用 vLLM profiling 建立 MoE 模型的推理成本模型：$C_{Model,G} = GC_0 / T_{Model}(G)$。关键推导：MoE 总参数 $N_{MoE} = (1 + (E-1)c)N$（其中 c = MLP 占比 ≈ 1/3，因为每两层替换一层 MoE）。通过 profiling 发现推理成本与模型大小近似线性关系（Figure 2），从而将推理成本量化为可优化的指标。
  
  **Step 3: 提出 Over-training 策略（Section 5）**
  核心洞察：在图 3（middle）中，给定训练预算下，模型性能对模型大小的变化在 loss-optimal 附近相当"平坦"（loss 对 N 不敏感），但推理成本随 N 线性增长。因此文章提出**刻意训练比 loss-optimal 小很多（70-85% reduction）的模型，将节省的预算投入更多训练 token**。这种 "over-trained" 配置以微小的质量损失换取显著的推理成本降低。
  
  论文方法全栈执行例子（over-trained 16-expert MoE，以 loss-optimal 4-expert MoE 的 quality 为 target）：
  - **算法Pipeline层**：
    1. 用 scaling law（公式 4）计算 loss-optimal 4-expert 的 (N_4, D_4) 和质量 L_4_opt
    2. Algorithm 1: dichotomy search 找满足 L_16(N, B) = L_4_opt 的最小 N_16 → 仅为 N_4 的 ~15-30%（Figure 5 right）
    3. 对应推理成本 I_16_min = min_g Get_cost(N_16, E=16, g) → 仅为 I_4 的 48%-53%
    4. 节省的训练预算全投入 tokens: D_16 = B / (6 * N_16) >> D_4
  - **系统框架层**：vLLM 部署 over-trained 16-expert MoE → 模型小很多 → 更多显存留给 KV-cache → 更大 batch size → 更高 throughput → 更低 cost per token。虽然 expert 数量更多（16 vs 4），但模型本身大幅缩小更主导推理成本（因成本近似线性于 N，而非 E）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（vLLM 默认 kernel path）。
  - **硬件架构层**：8×40GB A100 + NVLink。差异：base model 因模型大可能需更多 GPU 或更大显存压力，over-trained 模型缩小后可在相同 GPU 数量下服务更多并发请求，或减少所需 GPU 数量。

  关键定量对比（方法 vs Baseline）：
  | 对比维度 | Baseline (loss-optimal) | 论文方法 (over-trained) |
  |---------|------------------------|------------------------|
  | 训练预算 | 固定 B | 相同 B |
  | 模型大小 | N_opt (loss-optimal) | 15%-30% of N_opt |
  | 训练数据量 | D_opt = B/(6N_opt) | >> D_opt (预算重分配) |
  | 推理成本 | 基准 I_base | 48%-53% of I_base (16-expert vs 4-expert) |
  | 模型质量 | L_opt | L_opt (锚定相同) 或略低 |
  | 训练成本 (同类质量) | 100% | 仅需 23.7%-42.8% FLOPs (16-expert vs 4-expert) |

  论文本质发现：**MoE 的"免费午餐"仅在训练侧成立**。要同时优化训练和推理，应在 loss-optimal 配置基础上"有意训练差一点（模型更小但数据更多）"，以推理效率换取可忽略的质量下降。这一思路颠覆了传统 scaling law 仅追求 loss-optimal 的单目标优化范式。

## SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?

- baseline方法是什么？
  Baseline 是直接在 MoE LLM 上应用现有 unlearning 方法（GA、GDIFF、NPO、RMU），对所有参数（或 experts+router 全量参数）进行梯度更新以最小化 forget loss。全栈执行例子：
  - **模型推理算法层**：给定 MoE LLM（如 DeepSeek-V2-Lite），forget set D_f 包含待遗忘知识（如危险化学知识），retain set D_r 包含需保留的通用知识。现有 unlearning 方法对所有 expert FFN 权重 + router 权重进行梯度更新，目标为 min_θ l_f(θ; D_f) + λ l_r(θ; D_r)。例如 GA (Gradient Ascent) 直接对 forget set 做梯度上升使模型遗忘，RMU (Representation Misdirection) 对特定层 MLP 的 hidden representation 施加 steering vector 扰动。每次迭代后，router 为每个 token 计算 gating score g^{(l)} = Softmax(Router(u_t^{(l)}))，选 Top-K expert 做 FFN 计算 h_t' = u_t + Σ_i g_i * FFN_i(u_t)。
  - **系统框架层**：论文未明确说明（不涉及框架修改，直接对 Hugging Face 加载的模型做梯度更新）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（标准 PyTorch training loop，无特殊 kernel）。
  - **硬件架构层**：NVIDIA A100 GPU（论文 Sec. 5 "∼1 GPU hour on an A100 per soft prompt"）。

  Baseline 的核心缺陷：
  (1) **Expert selection shift ("short-cut")**：在 unlearning 过程中，Router 会逐渐将 token 从原本最相关的 target expert 切换到非目标 expert（Fig. 3a 显示 expert selection overlap ratio 持续下降）。原因：Router 发现切换激活的 expert 比真正抹除 target expert 中的知识更容易降低 forget loss——非目标 expert 原本不包含 target knowledge，对其做 unlearning 的 forget loss 更低，但这实际上是"假遗忘"。
  (2) **过度遗忘导致 utility 崩溃**：由于 expert selection shift，非目标 expert 被频繁激活参与 unlearning，但其原本包含的是与 forget set 无关的知识。强制对这些 expert 做 unlearning（即破坏其正常知识表示）导致模型 utility 严重下降——Table 1 显示 Qwen 在 GA unlearning 后 UT 从 0.5979 降到 0.3393（44% 下降），DeepSeek 从 0.5500 降到 0.3145（43% 下降）。
  (3) **Router 固定也无法解决**：即使固定 router 参数不动，unlearning 仍可间接影响 router 选择——因为第 l 层的 router 决策依赖前一层的 expert output，而前一层的 expert 已被 unlearning 修改，导致 cascading shift。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **SEUF (Selected Experts Unlearning Framework)**，通过"识别→锚定→聚焦遗忘"三步来解决 baseline 的 expert selection shift 和过度遗忘问题。全栈执行例子：
  - **模型推理算法层**：核心三步——(a) **Expert Attribution**：从 forget set D_f 随机采样 ~100K tokens 的子集 D_s，对每个 token t 在第 l 层的 Router 输出 g_{i,t}^{(l)} = Softmax(Router(u_t))[i]，按 s_i^{(l)} = (1/Z) * Σ_j (1/L_j) * Σ_t g_{i,t}^{(l)} 计算每个 expert 的全局 affinity score。在 DeepSeek-V2-Lite 的 64 experts 中，通常仅 ~6-9 个 expert 被频繁激活（长尾分布，Insight 1）。跨所有 layer 排序选 top-1 expert（M=1 时性能最优，Insight 4）；(b) **Router Anchor Loss**：L_anchor^{(l)} = ||g^{(l)} - a^{(l)}||_2^2，其中 a_i = 1 当且仅当 expert i 为选中的 target expert。这个 MSE loss 强制 router 在 unlearning 全过程中持续输出接近 [0,...,1,...,0] 的 gating 分布，防止 router 切换激活其他 expert；(c) **Focused Unlearning**：仅对 target expert 的 FFN 权重和对应 router 做梯度更新，冻结其他所有参数（仅更新 0.06% 参数）。损失函数：min_θ l_f(θ; D_f) + λ l_r(θ; D_r) + α * L_anchor（α=1 最优）。
  - **系统框架层**：论文未明确说明（不修改框架，可直接在 PyTorch 训练循环中实现）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。
  - **硬件架构层**：NVIDIA A100 GPU。

  **设计思路核心映射**：

  - 缺陷(1) "Expert selection shift (Router 作弊切换)" → 方案：**Router Anchor Loss**。通过 MSE loss L_anchor = ||g - a||_2^2 强制 router 输出固定在 target expert 上，消除 router 的"自由度"。这确保了整个 unlearning 过程中，forget set 的 token 始终被路由到 target expert，无法通过切换到其他 expert 来欺骗地降低 forget loss。Fig. 3b 验证了 forget loss 的降低是真正的知识抹除而非路由作弊。

  - 缺陷(2) "非目标 expert 被强迫参与 unlearning 导致 utility 崩溃" → 方案：**Focused Unlearning on top-1 expert only**。仅对 target expert 进行梯度更新，非目标 expert 完全冻结，确保它们存储的通用知识不受任何影响。Table 3 定量验证：GA+SEUF 将 Qwen 的 UT 从 0.3393（baseline GA）恢复到 0.5012（接近 pretrained 0.5979），同时 FE 保持 0.2987（vs baseline 0.2953，几乎不变）。在 RMU+SEUF 上效果更显著：UT 从 0.3560 恢复到 0.5351。

  - 缺陷(3) "Router 固定也无法阻止 cascading shift（因前层 expert 输出变化间接影响后续 router）" → 方案：**三层联合设计**。Expert Attribution 精确定位最相关的单个 expert + Anchor Loss 强制该 expert 持续激活 + 仅对该 expert 做 unlearning。由于所有非目标 expert 被冻结，前层非目标 expert 的输出不变，因此即使 router 未固定（实际上 router 也被冻结仅 target expert 的 router 可训练），非目标层的 cascading shift 也被消除。

  关键定量对比（SEUF vs Baseline）：
  | 对比维度 | Baseline（直接 unlearn） | SEUF |
  |---------|----------------------|------|
  | 更新参数比例 | 100% (或 experts+router 全部) | 0.06% (仅 top-1 expert + 对应 router) |
  | Expert selection 稳定性 | 持续 shift（overlap ratio 下降） | 稳定（anchor loss 强制保持） |
  | GA on Qwen/WMDP UT | 0.3393 | 0.5012 (+47.8% 相对改善) |
  | RMU on Qwen/WMDP UT | 0.3560 | 0.5351 (+50.3% 相对改善) |
  | GDIFF on DeepSeek/WMDP UT | 0.3929 | 0.4895 (+24.6% 相对改善) |
  | GCG jailbreak 后 FE | 未测试 | 保持 0.01（知识不可恢复） |

  论文核心洞察：MoE unlearning 的秘密在于"少即是多"——unlearning 一个 expert 就足够（Unlearning One Expert Is Enough），关键在于选对 expert 并锁定 router。这一发现颠覆了传统 unlearning"更新越多参数遗忘越彻底"的直觉，揭示 MoE 架构中知识高度集中在少数 expert 的特性。

## Upcycling Large Language Models into Mixture of Experts

- baseline方法是什么？
  Baseline 有两种：(1) **续训稠密模型（Continued Dense Training）**——在预训练后的稠密 checkpoint 上继续用新数据训练，保持稠密 MLP 架构不变；(2) **从头训练 MoE（Training from Scratch）**——随机初始化 MoE 架构并从头训练，不使用预训练权重。
  
  全栈执行例子（Baseline 续训稠密模型）：
  - **算法**：稠密 Transformer decoder，每层一个 FFN (MLP)。NVIDIA Nemotron-4 15B，SwiGLU 激活 + RoPE，预训练 8T tokens 后继续训练 1T tokens。
  - **系统框架**：Megatron-LM 分布式训练，data parallelism + tensor parallelism。
  - **编译框架**：论文未明确说明（Megatron-LM 内部使用 NCCL + cuBLAS/cuDNN）。
  - **kernel 调度**：论文未明确说明（标准的 GEMM kernel，无 expert 路由开销）。
  - **硬件架构**：NVIDIA GPU（论文未明确型号），标准 GPU 集群。
  - 执行流程：Token → Attention (QKV GEMM + MHA) → MLP FFN (W1 GEMM → SwiGLU → W2 GEMM) → 下一层。每 token 经过相同的单个 FFN，总激活参数 = 总参数 = 15B。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Upcycling + Virtual Group Init + Weight Scaling + Softmax-then-TopK Routing** 方法，将预训练稠密模型转换为 MoE 模型，在相同或略高计算量下获得更好的模型质量。

  全栈执行例子（论文方法 E8G1T2 Upcycling）：
  - **算法**：从 Nemotron-4 15B 稠密 checkpoint 出发，将每层 MLP 权重复制 8 份初始化 8 个 expert，使用 Virtual Group Init（if fine-grained）+ Weight Scaling。Router 随机初始化，使用 softmax-then-topK。每 token 路由到 top-2 experts，以 Router 概率加权求和。训练时使用 Load Balancing Aux Loss (coeff=1e-2)。
  - **系统框架**：Megatron-LM (expert parallelism) + NeMo。Upcycling 代码在 Megatron-LM moe/upcycling 模块中。
  - **编译框架**：论文未明确说明（Megatron-LM 内部 all-to-all 通信 + expert GEMM）。
  - **kernel 调度**：论文未明确说明（all-to-all token dispatch/combine 通信 + expert FFN GEMM）。
  - **硬件架构**：NVIDIA GPU（论文未明确型号），支持 DP + TP + EP。
  - 执行流程：Token → Attention → Router (softmax W_r @ x → top-2) → token dispatch (all-to-all by expert ID) → Expert_1 FFN (W1 GEMM → SwiGLU → W2 GEMM) / Expert_2 FFN → weighted sum (p1*o1 + p2*o2) → token combine (all-to-all back) → 下一层。总参数 ≈ 15B * (1 + (E-1)*2/3) ≈ 85B，激活参数 ≈ 15B (top-2 下约 30B FLOPs 等价)，总训练 FLOPs 约高于 dense 续训。

  **设计思路核心映射**（Baseline 缺陷 → 论文方法设计选择）：

  - 缺陷(1) **续训稠密模型 plateau 早，无法利用更多参数提升容量** → 方案：**Upcycling into MoE**。通过将单 MLP 扩展为多 expert MoE，增加模型容量（参数量扩大 ~5.7× 但不按比例增加 FLOPs）。Figure 4a 定量验证：Nemotron 2B 续训迅速 plateau，而 upcycling 持续改善，最终 loss 低 1.1%。Table 1 大规模验证：Nemotron-4 15B 续训 1T tokens 后 MMLU 65.3 vs upcycling E8G1T2 MMLU 67.6。

  - 缺陷(2) **Fine-grained MoE upcycling 初始 forward pass 与 dense 模型不等价，loss 爆炸无法收敛** → 方案：**Virtual Group Initialization + Weight Scaling**。原因分析：(a) fine-grained expert 被切分为更小 shard，每个 expert 只贡献 dense 输出的 1/G，router 需要恰好从每个 shard 选一个副本才能重建完整输出；(b) 随机 router 的 softmax 概率 ≈ 1/N (N=E×G)，导致每个 expert 输出被缩放 1/N。Virtual Group Init 通过将 router 权重在组内复制，保证 TopK 恰好覆盖所有 G 个 shard。Weight Scaling 通过 ³√(E×G²/T) 缩放因子补偿输出缩放。Figure 9 定量验证：w/ weight scaling loss 低 1.5%。

  - 缺陷(3) **TopK-then-softmax Router 丢失绝对值信息，导致 sub-optimal 训练** → 方案：**Softmax-then-TopK Router**。TopK-then-softmax (Mixtral 方案) 对 top-K logits 重新做 softmax，丢失非 top-K logits 的绝对值信息（softmax of single element = constant 1, no gradient）。Softmax-then-topK 保留完整的 softmax 分布，Router 能感知所有 expert 的相对重要性。Section 3.4 实验验证 softmax-then-topK 一致优于 topK-then-softmax。

  - 缺陷(4) **Upcycling 使用 fine-tuning 式的小学习率使模型停留在稠密模型的局部最小值，expert 无法分化** → 方案：**学习率重置策略**。将学习率从 pretraining 最低 (2e-5) 重新 warmup 到峰值 (2e-4)，再 cosine decay。这会降低 upcycled MoE 与 base dense model 的权重 cosine similarity（从 ≈1 降至 0.6-0.7, Figure 6），帮助模型逃离 dense 局部最小值，促进 expert 分化。Figure 5 定量验证：constant LR 快速 plateau，重置 LR 最终 loss 更低。

  - 缺陷(5) **小 batch size 下 MoE 每个 expert 只收到 1/E 的 tokens，梯度噪声大，load balancing loss 不稳定** → 方案：**大批量训练（4M tokens batch）**。通过增大 global batch size，每个 expert 收到的有效 token 数增加，梯度更稳定，load balancing loss 更准确。Figure 7 定量验证：batch size 1024 (4M tokens) 收敛快于 512 (2M tokens)，且 GPU FLOP utilization 更高。

  关键定量对比（Baseline vs 论文方法）：
  | 对比维度 | 续训稠密 15B (1T tokens) | Upcycling E8G8T8 (1T tokens) | Upcycling E8G1T2 (1T tokens) |
  |---------|------------------------|-------------------------------|-------------------------------|
  | val loss | 1.377 | 1.320 (-4.1%) | 1.306 (-5.2%) |
  | MMLU (5-shot) | 65.3 | 66.2 (+0.9) | 67.6 (+2.3) |
  | 总参数 | 15B | ~85B (64 experts, 1/8 hidden) | ~85B (8 experts) |
  | 训练 FLOPs | baseline | 等 FLOP (iso-FLOP) | 更高（top-2 增加计算量） |

  论文核心洞察：Upcycling 不仅利用预训练权重的知识加速训练，更重要的是 MoE 架构本身提供了比 dense 模型更高的容量上限。在足够长的 token horizon (1T+) 下，MoE 相对于 dense 的优势持续扩大。同时揭示了 upcycling 与 fine-tuning 本质不同——它需要更大的学习率来摆脱局部最优、需要大批量来稳定 expert 训练。

## X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

- baseline方法是什么？
  Baseline 是**现有 MoE 训练系统（DeepSpeed-MoE、DeepSpeed-TED、Tutel）在非 NVIDIA HPC 平台上训练 expert-specialized MoE（DeepSeek 风格）**。具体痛点：
  1. **CUDA 强依赖**：DeepSpeed-MoE 和 Tutel 的 MoE kernel 实现深度绑定 CUDA，无法高效移植到 AMD ROCm 平台。在 AMD MI250X 上性能 <10 TFLOPs（<10% 峰值），Megablocks 高度集成 Megatron-LM 无法在 AMD 上运行。
  2. **Zero-padding 内存和通信膨胀**：现有 MoE 框架（GShard、DeepSpeed-MoE、Fairseq）使用固定 expert capacity C 的 batched matmul pipeline。dispatch mask [S, E, C] + expert buffers [E, C, H] 大量 zero-padding，在 expert-specialized MoE（数百 fine-grained experts + large top-k）中 dispatch/combine 激活消耗 >70% 总激活内存，且 zero-padding 随 alltoall 传输浪费通信带宽。
  3. **跨节点通信冗余**：Large top-k 路由（如 k=8）使同一 token 被发送到多个跨节点 expert，在 Dragonfly 等层次化网络拓扑上产生大量重复跨节点传输（冗余率可达 75.1%），现有系统不感知网络拓扑。
  4. **激活内存瓶颈转移**：Expert-specialized MoE 中 dispatch/combine 激活（Adispatch, Acombine）随 fine-grained factor m 线性增长，而中间 FFN 激活保持不变。现有 TP+EP 混合并行在进入 EP 时仍复制全序列激活，无法缓解新瓶颈。

  全栈执行例子（以 DeepSpeed-MoE 在 Frontier AMD MI250X 上训练 DeepSeek 风格 201B MoE 为例，256 GPU，EP=64）：
  - **模型训练算法层**：DeepSeek 风格 expert-specialized MoE（256 experts, top-k=8, H=7168, HFFN=2048），GShard 式 gating + expert capacity C=1.25 × avg_tokens_per_expert，token dropping 策略。训练使用 ZeRO-1 DP + EP。
  - **系统框架层**：DeepSpeed-MoE v0.15.5，dispatch mask [S, 256, C] + expert buffers [256, C, 7168] with zero-padding，einsum + batched matmul pipeline，even alltoall（含 padding token 通信）。结果：OOM，无法训练。
  - **编译框架层**：论文未明确说明。CUDA/ROCm 编译，无跨平台编译优化。
  - **kernel调度层**：PyTorch einsum dispatch + batched matmul（含大量 zero-padding 计算），CUDA kernel（无法在 AMD 上高效运行，fallback 到慢速 PyTorch 实现）。Redundancy rate 54.8%（EP=32 时），跨节点重复传输。
  - **硬件架构层**：Frontier 超级计算机，AMD MI250X GPU（Infinity Fabric intra-node 200 GB/s, Slingshot inter-node 25 GB/s, Dragonfly 拓扑）。Alltoall 跨节点延迟 >10× intra-node，且随 scale 增加出现 outlier（>500ms per collective）。

  Baseline 的核心缺陷：(a) **跨平台可移植性差**——CUDA 绑定无法在 AMD/ROCm 上高效运行，需 costly 的 ROCm kernel 重写；(b) **Zero-padding 导致内存 OOM**——dispatch/combine 阶段 padded buffer 消耗 >70% 激活内存，限制可训练的模型规模；(c) **通信效率低**——even alltoall 传输 padded data + large top-k 导致跨节点 token 重复传输，在层次化网络上带宽利用极差；(d) **现有并行策略不适用**——TP+EP 不减少 Adispatch/Acombine 激活内存，ZeRO-DP 也不减少激活。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **X-MoE，一个跨平台 MoE 训练系统，通过 padding-free sparse pipeline、redundancy-bypassing dispatch 和 sequence-sharded hybrid parallelism 三大技术系统性解决 expert-specialized MoE 在非 NVIDIA HPC 平台上的训练瓶颈**。

  1. **PFT + Padding-Free MoE Pipeline（解决 Zero-padding 内存/通信膨胀 + 跨平台可移植性）**：
     - 设计稀疏 PFT 数据结构（token_buffer x + ERI-arrays），仅存储有效路由 token，消除 dispatch/MLP/combine 全流程 zero-padding
     - 用 uneven alltoall 替代 even alltoall，通信量随实际 token 数线性增长
     - Triton 实现 gather/scatter/sequential GeMM kernel，硬件无关（AMD ROCm + NVIDIA CUDA 均支持），无需 per-platform kernel 重写
     - 激活内存从 GShard 的 O(ckbsh)+O(ckb²s²) 降至 O(kbsh)

  2. **RBD（解决跨节点通信冗余）**：
     - 分层两级 dispatch：Pilot tokens（去重后最小跨节点 token 集）+ Local replica（节点内重复 token）
     - 仅 Pilot tokens 走跨节点 alltoall（低带宽 Slingshot 25 GB/s），Local replica 在目标节点从 Pilot 重建后走节点内 alltoall（高带宽 Infinity Fabric 200 GB/s）
     - 跨节点通信减少 52.5%（实测），总体 dispatch 加速 1.55×

  3. **SSMB（解决激活内存瓶颈转移）**：
     - 在 TP→EP 转换时，将输入序列切分到 EP ranks（drop partial tokens），每个 EP rank 仅保留 1/G 序列片段（G=TP group size）
     - Adispatch 和 Acombine 内存减少 G×
     - MoE block 结束后 all-gather 恢复完整序列，保持与下游 TP block 兼容
     - 相比 activation checkpointing：无重计算开销 + 无额外 alltoall（checkpointing 需 6 次 alltoall/layer，SSMB 仅 4 次）

  全栈执行例子（以 X-MoE 在 Frontier AMD MI250X 上训练 DeepSeek 风格 545B Super MoE 为例，1024 GPU，EP=256，TP=1-2）：
  - **模型训练算法层**：DeepSeek 风格 expert-specialized MoE（256 experts, top-k=8, H=7168, HFFN=2560, 61 layers, 545.4B params）。PFT padding-free pipeline 消除 zero-padding 开销。RBD 减少跨节点通信冗余。SSMB 切分 MoE block 序列减少激活内存。
  - **系统框架层**：X-MoE 集成于 DeepSpeed 0.15.5。PFT construction → uneven alltoall dispatch（仅有效 token）→ sequential GeMM（per-expert 无 padding）→ uneven alltoall combine。RBD: Stage 0 pilot selection → S1 inter-node uneven alltoall (pilot only) → S1 local replica reconstruction → S2 intra-node uneven alltoall (replica only) → merge。SSMB: TP block → drop partial tokens → EP MoE block (PFT+RBD) → all-gather → next TP block。
  - **编译框架层**：论文未明确说明。Triton 作为跨平台 kernel 编译器（Triton IR → AMD ROCm / NVIDIA CUDA PTX）。
  - **kernel调度层**：Triton gather kernel（B thread-blocks, 256 threads/block, coalesced read along H dim）→ uneven alltoallv (RCCL + libfabric, 仅实际 token) → sequential GeMM (rocBLAS, 每 expert 独立 launch, 无 padding 计算) → Triton scatter kernel（coalesced write along H dim）→ uneven alltoallv combine。RBD 模式：Pilot token gather kernel → inter-node alltoallv (Slingshot 25GB/s) → s1_mapping_indices-based local replica reconstruction → intra-node alltoallv (Infinity Fabric 200GB/s)。
  - **硬件架构层**：Frontier 超级计算机，1024 AMD MI250X GCD（128 nodes）。Dragonfly 拓扑：同一 rack ≤256 GPU（低延迟），256+ GPU 跨 rack 通信 alltoall 延迟 >10× 升高 + 频率高发 outlier (>500ms)。X-MoE 通过 RBD 最大化 intra-node 通信利用 + 限制 EP=256 避免跨 rack 延迟剧增。总计 10.44 PetaFLOPs 聚合吞吐量。

  **Baseline 缺陷 → 方法映射表**：
  | Baseline 缺陷 | 论文方法 |
  |---|---|
  | CUDA 绑定，无法在 AMD 上高效运行 | Triton 跨平台 kernel（gather/scatter/sequential GeMM），ROCm/CUDA 均支持 |
  | Zero-padding 导致 >70% 激活内存浪费 + 通信膨胀 | PFT 稀疏数据格式 + 全 padding-free pipeline（uneven alltoall + sequential GeMM） |
  | Large top-k 产生大量跨节点重复 token 传输 | RBD：Pilot token + Local replica 两级 dispatch，跨节点通信减少 52.5% |
  | TP+EP 并行无法减少 Adispatch/Acombine 激活内存 | SSMB：MoE block 内序列切分，激活内存减少 G× |
  | Even alltoall 随 scale 增大通信开销剧增 | PFT uneven alltoall + RBD hierarchical dispatch |
  | Activation checkpointing 需额外 alltoall + 重计算 | SSMB 无额外通信 + 无重计算，吞吐量更高 |
