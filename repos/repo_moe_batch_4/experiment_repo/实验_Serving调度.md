## Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：(1) **扩展 NVIDIA FasterTransformer 推理框架以支持 DeepSpeed MoE 模型**——在 FasterTransformer 中添加 MoE encoder/decoder layer 支持、TUPE attention、gate routing 和 MoE 层的高效计算。(2) **MoE Batch Pruning**——在 decoder 自回归生成过程中动态移除已完成翻译的句子，修改 gating 函数为已完成的句子分配大的 expert_idx（路由到末尾），仅处理 active_tokens 行，避免加载已完成句子的 expert 权重矩阵。(3) **与 Triton Inference Server 集成**——利用 Triton 的模型管理、动态 batching 和云规模弹性扩缩容实现生产部署。
  - 实验比较：(a) Batch pruning 的 throughput 对比：有/无 batch pruning 优化（1.14× speedup）；(b) 端到端推理吞吐对比：Torch-FP16 vs FT-FP16 vs FT-INT8 vs FT-INT4 在不同 batch size 和 beam 下的每秒处理 tokens 数（Table 3）；(c) 部署成本对比：优化后 NVIDIA T4 GPU 上的 5.32B MoE 模型 vs CPU 上的 0.04B 小模型 vs CPU 上的 5.32B 大模型，比较每月每 token 成本（Table 4）。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA PCIE V100（开发和评估），Docker Ubuntu 20.04 + CUDA 11.6
  - 生产部署：单卡 NVIDIA T4（16GB VRAM），Azure NC4as T4 v3 实例，价格为 $390.55/月
  - CPU baseline：Azure F16s 实例（AVX512），$587.65/月

- 开源Serving框架是什么。修改了什么。
  - 开源框架：NVIDIA FasterTransformer（https://github.com/NVIDIA/FasterTransformer），集成 Triton Inference Server（https://github.com/triton-inference-server/server）
  - 修改内容：(1) 在 FasterTransformer 中新增 MoE layer 支持——实现了 MoE encoder layer 和 decoder layer，包含 token routing（基于 CUB radix sort）、expert computation（基于 CUTLASS Grouped GEMM）、以及 TUPE attention；(2) 新增 batch pruning 机制——在 decoder beam search 的每次迭代中，gating 函数检测已完成句子并将其 expert_idx 设为极大值，token routing 将已完成句子排列到激活矩阵末尾，仅对 active_tokens 行执行 expert GEMM；(3) 新增 4-bit/8-bit 量化 expert weights 的 fused GEMM+Dequantize kernel；(4) 支持 Triton Inference Server 的 dynamic batching 集成用于云规模部署。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文基于开源 FasterTransformer 框架扩展，FasterTransformer、CUTLASS、CUB、Triton Inference Server、SentencePiece 均为开源。论文未提供独立开源仓库。
  - 框架输入到硬件执行全过程（以 EN-DE 翻译，batch_size=32, beam=2 为例）：
    1. **输入**：32 句英文句子通过 Triton Inference Server HTTP/gRPC 接口到达，Triton 将请求组 batch 发送给 FasterTransformer backend。
    2. **Tokenizer**：FasterTransformer 调用 SentencePiece tokenizer 将文本转为 token IDs（vocab 128K），形成 input_ids tensor。
    3. **Encoder 执行**：24 层 MoE encoder layers。每层的 self-attention（TUPE）使用标准 FP16 GEMM。每两层的 MoE FFN layer（共 12 个 MoE layers）：(a) Router（top-1 gating）为每个 token 计算 softmax weight 并选择 expert_idx；(b) Token Routing——CUB radix sort 按 expert_idx 排序 tokens，permute activation 使同 expert 的 tokens 连续排列；(c) Expert Computation——CUTLASS Grouped GEMM 并行执行所有 experts 的矩阵乘法，若使用量化则 fused dequantize；(d) Un-permute——恢复原始 token 顺序并乘以 expert scale。输出 encoder hidden states。
    4. **Decoder 执行**（自回归 + beam search）：12 层 decoder layers。每步生成一个 token。在 beam search 中，当某句子的 EOS token 生成后，Batch Pruning 机制将其 expert_idx 设为极大值（如 INT_MAX），token routing 将该句子的 tokens 排列到激活矩阵末尾，subsequent expert GEMM 仅处理前 active_tokens 行——避免为已完成句子加载 expert 权重矩阵。1.14× 加速。
    5. **输出**：生成的 target tokens 通过 Triton Inference Server 流式返回给客户端，de-tokenize 为德文文本。
    6. **部署**：Triton 管理模型实例的生命周期，根据请求流量动态扩缩容（scale up/down），所有实例加载同一 5.32B MoE INT4 量化模型到 T4 GPU（约 1.25GB）。
    7. **成本**：T4 上 5.32B MoE INT4 模型每月每 token 成本 $0.153，低于 CPU 上 0.04B 小模型的 $0.209（且 BLEU 质量更高）。

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

## Towards MoE Deployment Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：论文在开源 fairseq [23] MoE Transformer 推理框架上实现了三项 Serving 调度优化：(1) **Dynamic Gating（动态门控）**：将原有的静态容量（static capacity）all-to-all token 分发机制改为动态可变长度的 argsort + 两阶段 all-to-all（先交换各专家的 token 数量，再传输实际 token），消除了 dispatch mask 的大矩阵乘法和空 token placeholder 传输；(2) **Expert Buffering（专家缓冲）**：在 GPU 内存中维护一个可配置大小的 expert cache，仅缓存活跃（hot）专家参数，其余专家参数存放于 CPU 内存，按需通过 Memcopy 加载，结合 LIFO 淘汰策略利用时序局部性；(3) **Load Balancing（负载均衡）**：基于历史 expert 激活数据的 greedy 多路分区算法（NP-hard 近似），将高负载和低负载专家混合分配到各 GPU，以及针对 MT Decoder 的 Anti-correlation Balancing 变体。
  - 实验比较：(a) Dynamic gating vs Static gating (baseline [2]) vs Tutel gating [16] 在不同 batch size 下的吞吐量和内存消耗（Figures 9, 10）；(b) single-node vs multi-node (2/4 nodes) 部署下动态门控的吞吐提升（LM: 6.21× single-node, MT Encoder: 10.98× multi-node, MT Decoder: 5.71× multi-node）；(c) Expert Buffering 在不同 GPU cache size（每 GPU 1-16 experts）下的 cache miss rate vs Belady's MIN 理论最优（Figure 12）；(d) Expert Buffering 的延迟-内存帕累托前沿（Figure 13）；(e) Load Balancing 的负载均衡效果（Max Load 和 Avg Max Load，Figure 14）；(f) Dynamic Gating + Expert Buffering + Load Balancing 三者组合的端到端吞吐对比（Figure 9）。

- 硬件平台是什么，配置是什么。
  - CPU: 2×Intel Xeon E5-2698 v4 @ 2.2GHz，700GB 内存
  - CPU-GPU 互联: PCIe 3.0，带宽 16GB/s
  - GPU: 8×NVIDIA Tesla V100，每卡 5120 CUDA cores，32GB HBM2 @ 900GB/s，NVLink 互联 @ 300GB/s
  - 单节点 8 GPU 到多节点（2/4 节点，即 16/32 GPU）的扩展实验

- 开源Serving框架是什么。修改了什么。
  - 开源框架: fairseq [23]（https://github.com/facebookresearch/fairseq），Meta 开源的可扩展序列建模工具包，内置 MoE Transformer 实现
  - 修改内容: (1) **Gating 模块**：将 fairseq 中基于 static capacity + dispatch mask（E, S, S×C 大小的稀疏矩阵乘）的 token 分发改为 argsort 排序 + 两阶段 all-to-all（先 size all-to-all，再 data all-to-all）+ index-based 重排，取消了 token dropping；(2) **Expert 内存管理**：新增 Expert Buffering 缓存管理器，包含 GPU expert cache（可配置大小）、CPU 内存专家参数存放、LIFO 淘汰策略、Memcopy 与 all-to-all 通信的 overlap 机制；(3) **Expert 放置策略**：新增基于历史激活数据的 greedy load balancing 算法，在模型加载时重排 expert-to-GPU 的分配映射。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未明确提供独立开源仓库。修改基于 fairseq（开源 MoE Transformer 工具包），本论文工作实现了系统原型。fairseq 本身在 GitHub 开源。论文明确声明"implement it on an open-source, state-of-the-art MoE-based Transformer [23]"，但未提供论文修改的具体 fork/branch 链接。建议在 fairseq 仓库中搜索动态 gating 相关的 commit 或分支。
  - 框架输入到硬件执行全过程（以 Language Modeling，Dynamic Gating + Expert Buffering 为例）：
    1. **输入**：batch_size=8 的 token 序列（PILE 数据集），经过 tokenizer 嵌入后形成大小为 (B, S, TD) 的张量
    2. **Dense Transformer Layer**：输入经过 MHA（Multi-Head Attention）+ residual 计算，输出 (B, S, TD) 张量
    3. **MoE Layer - Gating**：Gating 函数（轻量级线性层）计算每个 token 到每个专家的 affinity score，top-2 gating 为每个 token 选择 2 个专家，输出 assignments (B, S, 2)
    4. **Dynamic Token Dispatch**：
       - argsort assignments 得到最优 token 排列索引（O(S log S)）
       - bin-count 统计每个 GPU 上各专家的 token 数量
       - 第一阶段 size all-to-all：各 GPU 交换 token 分配量信息（极小消息）
       - 第二阶段 data all-to-all：按 idx 重排 tokens 后分片发送
    5. **Expert Buffering Check**：
       - 检查 kernel 分配到的 experts 是否在 GPU expert cache 中
       - Cache hit → 直接使用 GPU 缓存中的 expert FFN 参数
       - Cache miss → 从 CPU 内存 Memcopy 加载 expert 参数到 GPU（与 all-to-all 通信并行 overlap）；若 cache 满则按"非当前 batch 活跃 + LIFO"策略淘汰
    6. **Expert FFN 执行**：各 GPU 上按 expert id 顺序串行执行 expert FFN（两个线性层 + 激活函数），输入 (tokens_per_expert, TD) → FFN1 → GELU → FFN2 → (tokens_per_expert, TD)
    7. **Token Collection**：处理完成后，通过另一个 all-to-all 将 tokens 发回原始 GPU，用 index-based gather 恢复原始顺序
    8. **后续层处理**：经残差连接合并，继续经过后续 dense/MoE 层（LM: 共 24 层，MoE 层频率 MF=2，即每 2 层中 1 层为 MoE）
    9. **输出**：LM head 产生 vocab 概率分布，输出下一个 token 预测

## Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：本论文提出一套 Serving 系统方法论，通过 ArI（算术强度）roofline 分析指导 MLA + MoE 模型的多加速器部署决策，而非实现新的调度器。核心分析包括：(1) **Batch Size 约束推导**：推导出 memory capacity 约束的 $B_{\rm cap}$ 公式（Eq. 11）和 SLO 约束的 $B_{\rm SLO}$ 上界（Eq. 12），分析 MLA（减小 $M_{\rm KV}$ 和 $M_{\rm attn}$）和 MoE（增大 $M_{\rm MoE}$）对可行 batch size 的互补效应；(2) **ArI 驱动的 batch size 目标**：推导 $B_{\rm RP} = \max(B_{\rm attn}, B_{\rm MoE}) = \max(RP_{\rm acc} \cdot deg_{\rm DP}, RP_{\rm acc} \cdot n_e/n_k)$，指导系统配置以满足计算利用率最大化；(3) **通信成本模型**：建立 all-to-all MoE 通信延迟模型 $Comm_{MoE}(B) = 2 \cdot \max_a(\Gamma_{imb}^{acc}(a)) \cdot M_{token} \cdot n_k \cdot B / (BW_{Int} \cdot n_{acc}) + \alpha$；(4) **分解式 serving 架构**：假设 prefill 和 decode 分离在不同机器上执行（disaggregated system），专注于 decode 阶段优化；(5) **部署粒度决策**：对比 256 GPU monolithic vs 32 GPU×8 多实例部署在不同 sequence length 和 skewness 下的吞吐量和负载均衡。
  - 实验比较：(a) **End-to-end throughput-latency tradeoff (Figure 9)**：GPT-3 vs Llama4-Maverick vs DeepSeek-R1 在 32 B200 GPU 下的 decode 吞吐量和 TPOT，分析各模型 $B_{\rm cap}$（DeepSeek-R1: 7360, GPT-3: 124, Llama4: 3328 at L=8192）；(b) **Interconnect 带宽影响 (Figure 10)**：NVLink (1.8 TB/s) vs InfiniBand XDR (100 GB/s) 下 DeepSeek-R1 的系统吞吐量和各阶段执行时间占比（FC/MoE/Attn/Comm），不同 L 和 B 下 all-to-all 通信延迟（151.8 µs vs 17.65 µs at B=128）；(c) **部署粒度对比 (Figure 11)**：256 GPU vs 32 GPU×8，不同 L (2048, 16384) 和不同互联带宽 (900/300/100 GB/s) 下的吞吐量；(d) **Expert 分布偏斜影响 (Figure 12/13)**：Zipfian skewness s 从 0.0 到 0.8，分析系统吞吐量、TPOT、load imbalance ratio ($\Gamma_{imb}^{acc}$) 的变化，以及 256 GPU vs 32 GPU×8 在不同 skewness 下的吞吐量对比；(e) **$B_{\rm RP}$ 与 $B_{\rm cap}$ 分析**：验证 MLA+MoE 模型是否能通过 batch size 使 FC 层达到 ridge point；(f) **FP8 精度对 $B_{\rm cap}$ 的影响**：低精度权重使 $B_{\rm cap}$ 增加，可以匹配 $B_{\rm RP}$ 实现最大吞吐量。

- 硬件平台是什么，配置是什么。
  - 主要评估平台：32 B200 GPU 系统，NVLink 5th Gen 全互联（1.8 TB/s 双向），遵循 NVL72 拓扑。
  - 部分配置使用 InfiniBand XDR（100 GB/s）连接 GPU 组间。
  - 真实硬件验证：DGX H100。
  - $deg_{\rm TP}=8$（GPT-3, Llama4），$deg_{\rm TP}=1$（DeepSeek-R1，因 reordered MLA 中 TP 无益）。
  - $deg_{\rm DP}=4$（GPT-3, Llama4），$deg_{\rm DP}=32$（DeepSeek-R1）。
  - $deg_{\rm EP}=32$（Llama4, DeepSeek-R1），GPT-3 无 EP。

- 开源Serving框架是什么。修改了什么。
  - 论文未使用或修改开源 Serving 框架（如 vLLM、SGLang 等）。实验基于自研 in-house simulator（基于 LLMSimulator https://github.com/scale-snu/LLMSimulator 构建）。模拟器中建模了现代 kernel 级和系统级优化（FlashAttention、FlashMLA、fused kernels、optimized communication），并在 DGX H100 上验证了单节点计算特性，使用 DeepEP 验证了多节点通信时间。
  - 并行策略：Attention block 使用 DP（数据并行），MoE block 使用 EP（专家并行）+ DP。DeepSeek-R1 的 attention block 不使用 TP，因 reordered MLA 中所有 head 共享 $\mathbf{C}_{\rm KV}$ 导致 TP 无延迟收益。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文的分析基于公开可复现的方法论，核心工具链包括：(1) LLMSimulator — 开源 LLM 推理模拟器；(2) FlashMLA — DeepSeek 开源的 MLA decode kernel；(3) DeepEP — DeepSeek 开源的 expert-parallel 通信库。
  - Serving 系统执行全过程（以 DeepSeek-R1 on 32 B200 GPU, decode 阶段为例）：
    ```
    # === 系统配置 ===
    - 32 GPUs, deg_DP=32, deg_TP=1, deg_EP=32
    - Attention Block: 每 GPU 独立处理 B/32 个请求（DP）
    - MoE Block: 256 experts 分布在 32 GPUs（每 GPU 8 experts）（EP）
    - 互联: NVLink 5th Gen, 1.8 TB/s 双向带宽

    # === 单个 Decode Step 的全栈执行流程 ===

    ## Phase 1: Attention Block (DP)
    For each GPU g in [0..31]:
      # 输入: B/32 个请求的 hidden state H (B/32, d_emb=7168)
      # 1. Q 压缩: C_Q = H @ W_CQ[g]  → (B/32, 1536)
      # 2. KV 压缩: C_KV_new = H @ W_CKV[g] → (B/32, 512)
      # 3. 更新 KV cache: C_KV[g] ← append(C_KV_new)
      #    每 token KV$ 仅 68.6KB (576×61×2B)
      # 4. Q 解压缩 (reordered): Q_i @ W_DK_i^T → (B/32, 512)
      # 5. Score (reordered): S_noPE = QW_i @ C_KV^T → (B/32, L)
      #    内存访问: 读 C_KV (d_KVco=512) 而非 K (d_dec=16384)
      #    ArI ≈ 100-200 Op/B, 接近 B200 的 RP_acc=281.25
      # 6. Context (reordered): O_i = softmax(S_i)@C_KV @ W_DV_i
      # 7. Output: U = concat(O) @ W_attn_out → (B/32, 7168)

    ## Phase 2: MoE Block (EP, all-to-all communication)
      # 8. Gating (本地计算):
      #    gate_score = U @ W_route → (B/32, 256)
      #    top_k_experts = topk(gate_score, k=8)  # 选 8/256 experts
      # 9. Token Dispatch (all-to-all通信):
      #    将 B/32 个 token 按 routing 结果发送到对应 expert 所在的 GPU
      #    Comm_dispatch = max_a(Γ_imb^acc(a)) × M_token × n_k × B / (BW_Int × 32)
      #    NVLink: ~17.65 µs at B=128; InfiniBand: ~151.8 µs at B=128
      # 10. Expert Computation (每 GPU):
      #     处理分配给该 GPU 上 8 个 experts 的 token
      #     每 expert: expert_out += gate(W_up × token) × W_down
      #     共享 expert: shared_out = shared_expert(U)  # 每 token 都执行
      #     平均每 expert 处理 B × n_k/n_e = B/32 个 token (均匀分布时)
      #     实际含偏斜: Γ_imb × B/32
      #     MoE FC 层 ArI 随 B 和 n_k/n_e 变化
      # 11. Token Combine (all-to-all通信):
      #     将 expert 输出从各 GPU 传回原 GPU
      #     Comm_combine ≈ Comm_dispatch

    ## Phase 3: 下一个 Decoder Block
      # 12. 重复 Phase 1-2 共 61 次（61 decoder blocks）
    ```
    - 系统瓶颈分析结果：当 B 足够大时，attention block 的延迟占比从 59%（K decompress）+ 40%（core-attention）降至 negligible，MoE 通信时间（dispatch/combine）和执行时间成为主要瓶颈。互联带宽和 expert 负载偏斜是决定端到端性能的主导因素。

## SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：SwapMoE 在 Serving 调度层面实现了完整的内存受限 MoE 推理框架，核心调度组件包括：(1) **运行时 Virtual Experts 管理**——维护动态更新的 Virtual Experts 子集，运行时根据 importance score 选择最重要的 experts 驻留在主存，不重要的 experts 按需从外部存储（CPU memory 或 SSD）加载/卸载。(2) **Amortized Expert Loading**——不在每个 sample 后同步更新所有 experts，而是跨多个样本摊销 expert 加载开销，异步进行 expert 置换，避免 IO 阻塞计算。(3) **Profiling-guided Memory Planning**——离线阶段对每个 expert 进行细粒度 profiling（memory footprint、inference latency、loading time、IO bandwidth），建立 config→performance 映射模型（E_accuracy, E_memory, E_latency），使用 Genetic Algorithm 搜索最优配置（包括 expert update frequency 和每层 Virtual Experts 数量 #experts_l）。(4) **Expert I/O Frequency 优化**——从低频率向高频率递增测试，找到 expert 更新不影响推理延迟的拐点频率。(5) **Layer Space Allocation**——利用遗传算法将有限 memory budget 分配到不同 MoE 层，更重要的层（如中间层）获得更多 Virtual Experts。
  - 实验比较：(a) **Overall Runtime Performance**：SwapMoE vs Pruning vs On-demand loading vs Original MoE，在不同 memory budget 下的 end-to-end latency 和 accuracy（ROUGE-2 / Perplexity）；(b) **Offline Planning Performance**：遗传算法找到的配置 vs 实际运行时 memory/latency/accuracy vs 给定 constraints；(c) **Robustness Analysis**：不同 expert 数量（16/32/64）的 Switch Transformer 的资源-性能 tradeoff；(d) **Ablation Study**：Simple scheduling (token counting) vs Simple planning (均匀分配) vs Full SwapMoE；(e) **Overhead Analysis**：峰值/平均 IO overhead、external memory consumption。

- 硬件平台是什么，配置是什么。
  - 设备：Jetson Nano（最大 GPU 内存 4GB）和 Jetson AGX ORIN
  - batch size = 1，模拟边缘设备连续 serving 场景（如个人 assistant 逐 token 生成）
  - external memory hierarchy：GPU main memory → CPU memory → SSD（PCIe 和 CPU-SSD 两种 IO 路径）
  - IO bandwidth reference：GPU-CPU PCIe 10-30 GiB/s，CPU-SSD 300-600 MiB/s

- 开源Serving框架是什么。修改了什么。
  - 框架：HuggingFace Transformers (Wolf et al., 2019)
  - 修改内容：(1) 在 MoE layer 的 forward 中插入 Masked Gating 逻辑——原始 router 输出后乘以 Virtual Expert mask，renormalize，将推理重定向到 VE subset；(2) 加入 Runtime Scheduler——管理 expert importance score 计算、Virtual Expert 更新（amortized + async loading/unloading）、IO 与计算协调；(3) 加入 Offline Memory Planner——基于 genetic algorithm 搜索最优 layer-wise expert 分配方案；(4) 加入 Fine-grained Profiler——profile 每个 expert 的 memory/latency/loading time，训练 E_accuracy 小 DNN 模型。
  - 关键修改点：SwapMoE 的核心调度逻辑可概括为——在 HuggingFace MoE layer 的 router 和 expert FFN 之间插入 Virtual Expert selection 和 update 逻辑，将原本的 full expert set 替换为动态维护的 subset，通过 coordinated I/O 和 computation scheduling 最小化 memory 和 latency。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未提供开源链接，实现基于 HuggingFace Transformers
  - 框架输入→硬件执行全过程（边缘设备连续 serving，batch_size=1）：
    ```
    输入：token sequence X = [x_1, ..., x_T]，memory budget LIMIT_memory
    
    离线阶段（一次性）：
    1. 对目标设备进行 Fine-grained Profiling：
       - 对每个 MoE layer 的一个 expert 进行 inference profiling
         (memory footprint, computation latency, parameter loading time via I/O)
       - profile I/O bandwidth (GPU↔CPU PCIe, CPU↔SSD)
    2. 收集 profiling dataset（少量 labeled samples from deployment scenario）
    3. 训练 E_accuracy(config) = small_DNN(config)，minimize Σ(actual - predicted)²
    4. 运行 Genetic Algorithm 搜索最优 config：
       - Random init population of configurations
       - For each iteration:
           mutate: randomly change one parameter in each config
           crossover: exchange or average existing configs
           evaluate: E_accuracy, E_memory, E_latency
           selection: remove configs violating constraints or suboptimal
       - Output: config* = {frequency*, #experts_1*, ..., #experts_L*}
    
    在线阶段（每个 sample）：
    1. Token 输入：X 进入 Transformer decoder layer
    2. Self-Attention：Q/K/V projection → FlashAttention → output（正常执行）
    3. MoE Layer（SwapMoE 调度路径）：
       a) Router：gating_scores = softmax(router(X))
       b) Masked Gating：
          mask = [1 if i∈VE else 0 for i in 1..num_experts]
          masked_scores = gating_scores * mask
          masked_scores = masked_scores / sum(masked_scores)
       c) Expert 计算：
          仅对 i ∈ VE 执行 E_i(X)，跳过非 VE experts
          y = Σ_{i∈VE} masked_scores[i] * E_i(X)
       d) Importance Score 收集：
          对每个 expert 计算 importance(E_i, X) = Σ_{x∈X_i} ||x|| * |G(x)_i| * ||E_i||
       e) Virtual Expert Update（每 frequency 个 samples 触发）：
          排序 experts by importance → 选择 top-k 为 VE_new
          VE_to_load = VE_new - VE_old → async I/O load from external memory
          VE_to_evict = VE_old - VE_new → release from main memory
    4. LM Head：hidden state → token logits → sample/argmax
    5. 输出 token y_t，作为下一轮 autoregressive 输入
    
    硬件执行路径（Jetson AGX ORIN）：
    - 当前 VE 的 expert 参数已在 GPU main memory → 直接参与 GEMM 计算
    - 需要加载的 expert 参数通过 async copy engine 从 CPU memory/SSD → GPU memory
    - 不重要的 expert 参数从 GPU memory 释放
    - IO overhead: peak ~40 MiB/s, mean ~20 MiB/s（远低于 PCIe/SSD bandwidth）
    ```

## Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：在SOTA serving框架（vLLM 0.4.0.post1 + Transformers v4.40.0）中集成Samoyeds sparse-sparse kernel，替代MoE层的标准GEMM执行流程。核心修改：(1) 消除input permutation开销——标准MoE流程需将tokens permute到各expert的tensor（产生额外内存分配和数据搬运），Samoyeds通过SEL选择数组在kernel内部直接索引有效tokens，跳过显式permutation；(2) 消除output un-permutation开销——expert输出从register直接写入压缩layout，避免先写global memory再读回的roundtrip；(3) operator fusion——将activation function与前驱operator融合，将weighted accumulation（scalar广播+点乘）与矩阵乘法融合，减少kernel launch和中间结果materialize；(4) 压缩output layout——MoE中间结果（expert输出）在accumulation前是row-wise稀疏的（稀疏比=expert数量），Samoyeds仅输出非零行，避免传输零值。batch size支持能力显著提升（平均4.41×）。
  - 实验比较：(a) MoE层级别：对比Transformers（v4.40.0）、MegaBlocks、vLLM-DS（含fused MoE kernel，2024年3月合并版本，~2.8× speedup over non-fusion），评估两类MoE（带/不带shared experts）的speedup；(b) 端到端模型级别：6种MoE模型（Qwen2-MoE/DeepSeek-MoE/MiniCPM-MoE/OpenMoE-34B/Mixtral-8×7B/Mixtral-8×22B）的latency speedup（seq_len=4096，batch=1或16）；(c) 不同batch size下吞吐量对比；(d) 最大batch size支持对比。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA GeForce RTX 4070 Super
  - CPU：Intel i7-12700，16G×2 DDR5，Ubuntu 22.04LTS
  - 软件栈：CUDA 12.1, cuSPARSELt 0.4.0, PyTorch 2.1.0, Transformers v4.40.0, vLLM 0.4.0.post1
  - 功耗控制：所有实验禁用CPU frequency scaling

- 开源Serving框架是什么。修改了什么。
  - 框架：HuggingFace Transformers v4.40.0 + vLLM 0.4.0.post1（含fused MoE kernel PR #2453）
  - 修改内容：将MoE decoder layer中的expert计算（gate_proj, up_proj, down_proj三个线性层）替换为Samoyeds sparse-sparse kernel调用。具体地：(a) 在vLLM的fused MoE kernel基础上，替换内部GEMM为Samoyeds kernel（处理双端稀疏）；(b) 消除expert输入/输出的permute/un-permute操作（原本由vLLM fused kernel管理）；(c) 集成weighted accumulation fusion和activation fusion。这些修改通过pybind11暴露的Python模块与框架对接。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：GitHub https://github.com/guqiqi/Samoyeds.git，Docker镜像可直接运行end-to-end实验
  - 框架输入→硬件执行全过程（单个decoder layer推理）：
    ```
    输入：prompt tokens [batch_size × seq_len]，路由权重
    
    1. Attention层（FlashAttention2）：
       输入tokens → Q/K/V projection → FlashAttention → output
       
    2. MoE层（Samoyeds优化路径）：
       a) Router: tokens → routing scores → top-k expert assignment
          - Qwen2-MoE/DeepSeek-MoE: 含shared experts（所有token都经过）
          - Mixtral-8×7B/22B: 仅routed experts
       b) Expert计算（Samoyeds kernel）：
          对每个expert E_i:
            - SEL[i] = {j | token[j] routed to E_i}  # 隐式permutation（无内存拷贝）
            - 加载编码权重: data[i], indices[i], metadata[i]（已在offline阶段压缩为Samoyeds格式）
            - gate_proj: C_gate = Samoyeds_spmm(W_gate_encoded, input[SEL])
            - up_proj: C_up = Samoyeds_spmm(W_up_encoded, input[SEL])
            - Activation fusion: C_act = SiLU(C_gate) * C_up  # fused in-kernel，无中间materialize
            - down_proj: C_out = Samoyeds_spmm(W_down_encoded, C_act)
            - 输出以压缩layout写入GMEM（仅非零行）
       c) Weighted accumulation（fused）:
          对每个token t:
            output[t] = Σ_i router_score[t][i] * expert_output[i][t]  # fused in-kernel
        
       Pipeline视图：
       GMEM → [cp.async] → SMEM (A_tile, B_tile) → [ldmatrix] → Register → [mma.sp] → SpTC计算 → Register C → [shuffle/stationary] → [store] → GMEM (压缩output)
    ```
  - 关键性能增益：消除input permutation overhead（大expert数量的模型收益更大，如Qwen2-MoE 60 experts和DeepSeek-MoE 64 experts），消除output zero-value传输（高sparsity模型加速达2.66×），fused activation+accumulation减少kernel launch和内存roundtrip。

## SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  实现：Samba-CoE运行时系统（CoE Runtime），用于在SN40L上部署和管理含150个experts、超1T参数的Composition of Experts系统。核心机制：(1) 三级存储管理 — Router始终驻留HBM，所有expert权重存储于高容量DDR，按需将当前活跃expert从DDR拷贝到HBM执行（DDR→HBM聚合带宽>1 TB/s）；(2) 动态内存管理器 — 类似传统动态链接器/加载器，每个expert模型独立编译，二进制文件中预先声明HBM和DDR空间需求，运行时由CoE Runtime在DDR中动态分配、按需激活（拷贝HBM段）、执行后回收；(3) LRU淘汰策略 — 尽量保持HBM中同时驻留尽可能多的活跃expert，超过HBM容量时淘汰最久未使用的expert（跳过read-only weight的回写）；(4) 硬件orchestrated kernel launch — AGCUs实现硬件级别的kernel调度，对decode阶段的短kernel消除host软件调度开销；(5) 静态垃圾回收 — 利用SN40L无动态内存分配/无指针别名的特性，编译器进行符号生命周期分析，将非重叠生命周期的符号分配到相同设备虚拟地址。
  实验比较：SN40L Node（8 socket）vs DGX A100（8×A100 80GB）vs DGX H100（8×H100 80GB），在Samba-CoE推理场景下，测量BS=1/BS=8、20/200 output tokens场景的端到端延迟和模型切换时间；以及随expert数量增加（1到150+）系统占用（machine footprint）的变化。DGX上模型切换需经过host DRAM（A100: 32 GB/s, H100: 64 GB/s），SN40L直接DDR→HBM（聚合>1 TB/s）。

- 硬件平台是什么，配置是什么。
  SN40L Node：8个SN40L RDU socket + 1个host x86 CPU。每socket：638 BF16 TFLOPS，64 GiB HBM（1.8 TB/s），最高1.5 TiB DDR（200 GB/s，8 socket聚合>1 TB/s）。模型以tensor-parallel (TP8) 方式映射到8个socket。DGX A100：8×A100 80GB PCIe（HBM ~2 TB/s aggregate），32 GB/s host-to-GPU PCIe带宽。DGX H100：8×H100 80GB（HBM ~3.35 TB/s aggregate），64 GB/s host-to-GPU带宽。DGX上假设全部HBM和host memory可用于权重和KV cache存储。

- 开源Serving框架是什么。修改了什么。
  论文未基于开源Serving框架。SambaNova自研CoE Runtime构建在低层设备驱动之上，是专有软件栈。论文未开源。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未开源。Serving执行全过程：(1) 应用层发出batch请求（如8个prompt）→ CoE Runtime接收；(2) Router在HBM中执行（BS=8），为每个prompt选择对应expert；(3) CoE Runtime检查被选中的experts是否已在HBM中（LRU cache hit）— 若未命中，从DDR按需拷贝expert的HBM段到HBM（跳过已在HBM中的部分）；若HBM空间不足，LRU淘汰旧expert；(4) 每个(prompt, expert)对依次在HBM中的expert上执行自回归解码（硬件orchestrated kernel launch模式，AGCUs编排kernel序列）；(5) 对多token生成（如200 tokens），expert权重在decode循环中被重复读取，充分利用HBM的时域局部性；(6) 完成后返回控制权给CoE Runtime，等待下一请求。

## Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：基于 **Kubernetes** 构建的异构 MoE 推理系统，面向 serverless 环境。核心调度创新有三项：(1) **异构架构**：非 expert 模块（Attention、Gate）部署在 GPU，expert 模块部署在 CPU；高频 expert 作为 "local experts" 与主模型同容器，低频 expert 作为 "remote experts" 部署为独立 serverless function，减少主模型内存占用并支持 local/remote experts 并行推理。(2) **SPS 算法（Similar Prompts Searching）**：基于输入 prompt 语义相似度预测 expert 激活模式。离线阶段用 Soft Cosine Similarity (SCS) 度量 prompt 级语义相似度，构建多叉聚类树；在线阶段遍历树找到 top-α 语义最相似的历史 prompt，用 softmax 加权的历史 expert 激活分布作为预测结果。复杂度 O(log n)，比暴力搜索快 10 倍以上。(3) **MMP 算法（Main Model Pre-allocation）**：基于 Hoeffding 不等式证明的最坏情况 expert 负载上界 (Theorem 1, Corollary 1)，预分配主模型内存规格 w_v 以满足 TTFT/TPOT SLO。(4) **Joint Memory and Replica Optimization**：用 Lagrangian 对偶方法优化 remote experts 的内存规格 y_{l,v}，用 LPT 算法划分 expert 子集到多 replica，并用 LPT 近似比 (4/3 - 1/3z_l) 保证最坏情况完成时间。全流程：请求到达 → SPS 预测 expert 激活 → MMP 预分配资源并启动主模型冷启动 → 选择低 utility expert 为 remote → Lagrangian 优化 remote expert 内存 → LPT 划分 replica。
  - 实验比较：(1) **Overall Performance**：Remoe vs CPU / GPU / Fetch（理想 expert offloading）/ MIX（CPU+GPU 异构但无 remote expert 分区），在 GPT2-moe (124M) 和 Deepseek-v2-lite (16B) 上对比推理成本，50 requests 随机采样，500 字符输入 + 200 输出 token；(2) **Cost under Different Prefilling/Decoding Ratios**：对比不同 prefill/decode token 比例下各方法的推理成本趋势；(3) **Cold Start and Algorithm Overhead**：对比各方法的冷启动时间，包括 REMOTE（remote expert functions 冷启动）、CALCULATE（优化逻辑开销）、主模型冷启动的分解；(4) **Prediction Accuracy**：SPS vs VarPAM / VarED / DOP / Fate / EF / BF，在 LMSYS-Chat-1M、WikiText-2、C4、SlimPajama 四个数据集上对比 JS Divergence 预测误差。
- 硬件平台是什么，配置是什么。
  - 服务器：双路 Intel Xeon Gold 6348 CPU（56核，112线程），2× NVIDIA A100 80GB GPU。
- 开源Serving框架是什么。修改了什么。
  - 开源框架：Kubernetes（容器编排）。修改内容：(1) 修改所有使用的 MoE 模型（GPT2-moe, Deepseek-v2-lite）支持 local/remote experts 并行推理；(2) 使用 C++ LibTorch 库和 gRPC 提供高效 serverless 推理服务，最小化数据传输开销和响应时间；(3) Pod 调度器为 NUMA-aware。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未提供开源链接。论文声明 "technical report will be made publicly available upon acceptance"。
  - 框架使用全流程：用户请求到达 → Pre-processing Layer 完成 tokenization → SPS 算法查询聚类树预测 expert 激活矩阵 → MMP 算法用 Theorem 1 的 worst-case bound 计算最小主模型内存 w_v，Kubernetes 调度主模型 Pod 到具有 GPU 的节点（A100）→ Remote experts 选择：计算每个 expert 的 utility score u_{l,k}，选 lowest-utility 的 experts 为 remote → Lagrangian 优化确定每层 remote expert function 的内存规格 y_{l,v} → LPT 算法划分 remote expert 集合并确定 replica 数 z_l，Kubernetes 创建 remote expert Pods（CPU-only）→ 主模型 Pod：GPU 执行 Attention/Gate（非 expert 模块），CPU 执行 local experts → 主模型通过 gRPC 将 N^{topk} 个 token embedding 发送到 remote expert Pods → Remote expert Pods 在 CPU 上并行执行 expert FFN 计算并返回结果 → 合并 local/remote expert 输出 → 逐层执行直至生成完整输出序列 → Post-processing 解码。Kubernetes 负责 Pod 生命周期管理（冷启动、健康检查、自动扩缩）。gRPC 负责 GPU↔CPU 间 token 数据传输（token size 7-14KB，远低于 6MB payload 限制）。

## Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Sem-MoE，基于 SGLang 构建的 semantic-aware model-data collaborative scheduling 系统（~5000 行 Python + 自定义 Triton kernel）。核心包含三部分：(1) **Offline Model Scheduling**——基于离线 profiling 构建 token-to-expert confidence table，将 ILP-based co-clustering 问题通过交替优化算法求解，生成 expert-to-device 调度表 E 和 token-to-device 调度表 T，在部署前调整每层 expert 的 device placement，同时重排 gate matrix 列实现透明 expert shuffle；(2) **Online Inter-Request Data Scheduling（Attention-DP）**——扩展 SGLang request scheduler，根据 token-expert affinity 将语义相似的请求动态路由到同一 DP rank，最大化 request-expert-group affinity。采用 workload-aware round-robin：每轮每个 device 各分配一个请求，一轮完成后 reset device mask，防止解码阶段负载倾斜；(3) **Online Intra-Request Data Scheduling（Attention-TP）**——在 attention 后的 reduce-scatter 中融合 speculative token shuffling（shuffled-reduce-scatter, SRS），将 token 提前 shuffle 到预测的 expert 所在 device，MoE 计算后再用 shuffled-allgather (SAG) 收集并恢复 token 顺序。同时利用 inter-layer expert-expert affinity（2-gram 马尔可夫链）增强预测精度。
  - 实验比较：(a) Attention-DP 场景——Sem-MoE vs SGLang（含 DeepEP）vs MoETuner，在三种数据集（MMLU/lmsys-chat-1m/ShareGPT）下绘制 latency-throughput 曲线，评估 TTFT SLO 和 E2E Latency SLO 下的最高吞吐（req/s）；(b) Attention-TP 场景——Sem-MoE vs SGLang vs MoETuner，请求率 1 req/s 不变，变化输入长度（256/512/1024），评估 TTFT 和 median E2E 延迟；(c) EP 通信缩减细节——单 MoE layer 的 LAR（Local Activation Rate）vs SGLang vanilla placement，LAR 从 25% 提升至 62%（DeepSeek）和 68%（Qwen3）；(d) 算法评估——Sem-MoE co-clustering vs SGLang vanilla vs MoETuner 的 LAR 和 load imbalance rate；(e) Cross-dataset 零样本迁移——预测器在 ShareGPT/lmsys-chat-1m/MMLU 间跨域评估 LAR；(f) Moonlight-16B 额外验证。

- 硬件平台是什么，配置是什么。
  - GPU：8-GPU server，每 GPU 96GB HBM，fast homogeneous interconnect（>400GB/s 专用互联带宽）
  - CPU：2× 44-core Intel CPU，2TB DDR5 memory
  - 软件栈：SGLang + PyTorch + Triton + DeepEP + NCCL/HCCL

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang（https://github.com/sgl-project/sglang）
  - 修改内容：(1) **Request Scheduler 扩展**——新增 token-expert affinity 感知的调度逻辑，在 Attention-DP 场景将语义相似的请求 batch 到同一 DP rank；(2) **Expert Placement 重配置**——利用离线求解的 expert placement table E 重排各层 expert 的设备分布，同时 shuffle gate matrix 列实现透明重分布；(3) **Shuffled-Reduce-Scatter (SRS) 和 Shuffled-AllGather (SAG) 融合通信原语**——在 Attention-TP 的 post-attention reduce-scatter 中嵌入 speculative token shuffling，用优化 argsort kernel（比 PyTorch 原生快 25%）计算 shuffle indices；(4) **DeepEP 集成**——使用 DeepEP 作为 EP 通信后端（normal mode），提升 all-to-all 效率；(5) **共享内存优化**——将 token-to-device table 等调度表常驻 GPU memory（<12 MB for DeepSeek-V2）。论文未提供独立开源仓库链接，框架修改集成在 SGLang 之上。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文基于 SGLang（开源），但 Sem-MoE 的自研修改代码未在论文中提供独立 GitHub 链接。可基于论文算法描述复现。
  - 框架输入→硬件执行全过程（Attention-DP 场景，DeepSeek-V2-Lite, 8 GPUs, EP8 + DP8）：
    ```
    输入：多 client 请求到达 SGLang HTTP server
    
    阶段 0 — 离线准备（一次性）：
    1. 用 20% ShareGPT/lmsys-chat-1m/MMLU 数据 profile DeepSeek-V2-Lite 各 MoE layer
       → 记录每 token 的 expert activation 频率
    2. 构建 token-to-expert confidence table C_p ∈ R^{t×N}（t=词汇量, N=experts/layer）
    3. 运行交替优化 co-clustering 算法（Algorithm 1）：
       - alternating between expert_place(Cp, req) and request_schedule(Cp, ep)
       - 输出：E（expert labels）、T（token labels）、Tp（confidence table）
    4. 按 E 重排各层 expert 的 device placement，shuffle gate matrix column

    阶段 1 — 在线请求调度（Attention-DP）：
    5. 多个 requests 到达 continuous batching scheduler
    6. Sem-MoE inter-request scheduler:
       - 对每个 request r，提取其 token IDs
       - 查表 T: 聚合各 token 的 device assignment
       - S_r = argmax_{j∈[E]} Σ_{token_i∈r} R_{ij}（最匹配 device）
       - dev_mask round-robin: 每 E 个请求一轮，每轮各 device 分配一个请求
    7. 请求被分配到各自最优 DP rank

    阶段 2 — MoE Layer 前向（每层）：
    8. Attention 计算在各 DP rank 独立完成（数据并行）
    9. Gate function: Softmax(W_g · x) → TopK experts per token
    10. Token→Expert Dispatch（all-to-all）：
        - 由于 expert placement 和请求调度已对齐 token↔expert affinity
        - 大部分 token 的 target expert 在本地 device
        - 仅少量 cross-device token 参与 all-to-all
    11. Expert FFN 计算（各 device 本地 + 来自远程的 token）
    12. Expert→Token Combine（all-to-all 反向）
    13. 输出 hidden states 进入下一 layer

    阶段 3 — Attention-TP 场景（Token-Level 调度）：
    14. Attention TP 计算后，不执行标准 allreduce
    15. 代之以 Shuffled-Reduce-Scatter (SRS):
        - 查表 T 和 inter-layer table A（2-gram Markov model）
        - 选置信度更高的表预测 token 应 shuffle 到的 device
        - argsort 产生 shuffle_indices
        - 按 shuffle_indices 重排 token → reduce-scatter（各 device 获得其管辖的 token 分片）
    16. MoE 层计算（gate + expert FFN）
    17. Shuffled-AllGather (SAG): 收集 token 分片 → 按 reverse_shuffle_indices 恢复顺序

    性能结果：
    - Attention-DP: Throughput ↑ 2.78×（DeepSeek-V2-Lite E2E SLO vs MoETuner）
    - Attention-TP: TTFT ↓ 24.9%（Qwen3-30B-A3B, input=512）
    - LAR 提升: 从 ~25% 到 ~62%（DeepSeek）/ ~68%（Qwen3），对应 41.8%/46.6% expert layer latency reduction
    ```

## Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：基于 **DeepSpeed inference engine** 构建的 expert-aware 推理系统。核心调度创新有三项：(1) **Expert-aware Batching（Algorithm 1）**：利用 pre-gating router 在推理前已知每个 token 的 expert 选择，将选择同一 expert 的 tokens 组 batch，最小化每 batch 所需激活的 unique expert 数量。伪代码：维护 ReqQueueByExpert（每个 expert 的请求队列），每次从请求最多的 expert 队列取 tokens 填充 batch（最多 MaxTokenLen），确保 batch 内 token 共享同一 expert。(2) **Fine-grained Prefetching**：利用 pre-gating 已知未来层所需 expert，在计算第 i 层时并行预取第 i+1 层的 expert 权重（compute stream 与 loading stream 流水线重叠），隐藏 expert 加载延迟。(3) **Belady-inspired Optimal Caching**：Belady 算法为理论最优离线缓存替换策略（替换未来最远访问的对象），传统上因无法预知未来访问而不可行。Read-ME 通过 pre-gating 预知所有 token 在所有层的 expert 需求，实现近似最优缓存替换：C(t-1)={e_1,...,e_k}，evict e* = argmax_{e∈C} F(e,t)（F(e,t) 为 expert e 下一次被请求的时间）。
  - 实验比较：(1) **Batching 策略**：Read-ME expert-aware batching vs Decoding-prioritized batching [38] vs Prefill-prioritized batching [39,47]，在 Chatbot Arena Dataset 重放负载下评估端到端延迟分布和 p95 延迟；(2) **Prefetching**：Read-ME Prefetching vs On-demand Loading [25]，在不同 expert cache capacity 下对比端到端延迟；(3) **Cache 策略**：Random vs LRU vs Belady 在不同 cache capacity (2/3/4/5 experts) 下对比 cache hit ratio；(4) **单请求延迟分解**：OpenMoE (layerwise router) vs Read-ME (pre-gating router) vs Dense Llama2，分解为 Router/Attention/Expert(MLP) 各部分延迟。

- 硬件平台是什么，配置是什么。
  - 推理系统评估：单卡 NVIDIA A100 80GB GPU。Host CPU memory 用于 expert offloading（论文未明确说明具体 CPU 配置）。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：DeepSpeed inference engine [38]（https://github.com/microsoft/DeepSpeed）。
  - 修改内容：
    1. **Expert-aware Batch Scheduler**：新增 ReqQueueByExpert 数据结构（每个 expert 一个 FIFO queue），Scheduler 收集 pre-gating router 输出的 expert assignment，将 tokens 按 expert 分入对应 queue。Algorithm 1 从 queue 中构建 batch：优先从请求最多的 expert 取 tokens，最大化 batch 内 expert 共享。
    2. **Prefetch Pipeline**：在 DeepSpeed 的 layer-wise 推理循环中插入异步 expert 加载流。Compute stream 执行第 i 层 expert FFN，同时 loading stream 从 host memory 向 GPU memory 传输第 i+1 层所需 expert 权重（cudaMemcpyAsync / PCIe transfer）。
    3. **Belady Cache Manager**：新增 expert cache 模块，维护 k 个 expert slots。Router 预计算所有 pending requests 在未来时间步所需的 expert 序列，构造 F(e,t) 映射（每个 expert 的下次访问时间）。Cache eviction 时选择 max F(e,t) 的 expert 驱逐。Cache 跨所有并发请求共享。
    4. **Pre-gating Router Integration**：将 Read-ME 的 pre-gating router 作为推理 pipeline 的第一步执行。Router 输出每个 token 在所有层的 expert assignment，传递给 Scheduler 进行 batch 规划和 cache 预热。
  - 开源：论文代码开源 https://github.com/VITA-Group/READ-ME。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/VITA-Group/READ-ME
  - 框架输入到硬件执行全过程（Read-ME 4.7B-17B, single A100 80GB, Chatbot Arena workload）：
    
    **阶段 0 — 系统初始化**：
    1. 加载 Read-ME model checkpoint（experts + pre-gating router + permanent expert）到 host memory。
    2. 初始化 DeepSpeed inference engine，替换原生 MoE inference pipeline 为 Read-ME pipeline。
    3. 初始化 Expert Cache：在 GPU memory 分配 k 个 expert slots（默认 k=5），初始为空。
    4. 初始化 ReqQueueByExpert[N]：N=8 个 expert queue（+ 1 permanent expert queue）。
    
    **阶段 1 — Pre-gating（推理前一次性路由）**：
    5. 多个 conversation requests 到达，Scheduler 收集所有 request tokens。
    6. Pre-gating Router G 执行：对每个 token sequence x_{≤t}，Router（1-layer Transformer, causal attention）计算 gating weights G(x_{≤t}) ∈ R^N，选 top-K=2 experts。因 router 与 layer index 无关，所有层 expert 选择一致。
    7. Router 输出：每个 token → {expert_i, expert_j}（跨所有 32 层相同）。
    
    **阶段 2 — Expert-aware Batching**：
    8. Scheduler 按 Algorithm 1 构建 batch：扫描 N 个 ReqQueueByExpert，从请求最多的 expert 开始，取 tokens 直到 batch size = MaxTokenLen。
    9. 结果：batch 内所有 tokens 共享同一 expert → 单 batch 仅需加载 1 个 expert（vs layerwise MoE 的 ~7.63/8 experts per batch）。
    10. Scheduler 将 batch 提交给 Inference Engine。
    
    **阶段 3 — 流水线推理执行**：
    11. Layer 1 开始：检查 Expert Cache——若 required expert e_1 在 cache → 直接使用；否则触发 cache miss，从 host memory 加载 e_1 到 GPU（PCIe 4.0, ~25 GB/s）。
    12. 同时启动 Prefetch Stream：异步加载 Layer 2 所需 expert e_2。
    13. Compute Stream：执行 Layer 1 attention（causal self-attention）→ Expert e_1 FFN（SwiGLU, d=5504）→ permanent expert FFN → 输出 hidden states。
    14. Cache Manager：Router 已预知 e_1 的下次使用时间 F(e_1, t)，若 cache 满，evict max F(e,t) 的 expert（Belady 策略）。
    15. 重复 Layer 2...32：每层计算与下层 expert 加载流水线重叠。
    
    **阶段 4 — Token Generation 循环**：
    16. 新生成的 token 通过 Router G（仅需对新 token 执行，causal attention 利用 KV cache）。
    17. 新 token 的 expert assignment 加入对应 ReqQueueByExpert。
    18. Scheduler 动态重组 batch，重复阶段 2-3。
    
    **关键结果**：
    - Expert-aware batching 将平均 unique experts/batch 从 5.08（decode-prioritized）/ 5.21（prefill-prioritized）降至 3.51。
    - 端到端平均延迟降低 5.0-6.1%，p95 延迟降低 9.5-10.0%。
    - Prefetching vs On-demand Loading：最高 30% 延迟改善。
    - Belady caching 在 cache capacity=4 时 hit ratio 77.21% vs LRU 66.95%（+10.26%）。

## Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：QLLM——基于 HuggingFace TGI 构建的 priority-aware 抢占式调度推理系统。核心设计：(1) Per-Expert FIFO Queues：每个 expert 拥有独立队列，token 按需入队，打破传统 layer-wise 同步 barrier；(2) Priority-Aware Scheduler：Dispatcher 将请求按优先级（LS/BE）分别入队（LS_PrefillQueue, LS_DecodeQueue, BE_PrefillQueue, BE_DecodeQueue），Batch Engine 按 Algorithm 1 优先级排序组 batch；(3) Expert-Level Preemption：在任意 layer 内收到 LS 请求时，Scheduler 立即通知 Inference Engine 停止 BE batch 执行，执行 LS prefill+decode 后动态合并 LS/BE 到同一 batch；(4) Unified Sequence/Batch Abstraction：用 Facade Pattern 封装每个 Sequence 的独立 tensor（KV cache, hidden states, routing_weights），对外呈现为统一 batch tensor 接口，支持零拷贝的 individual sequence 状态更新而无需 split-merge；(5) Unified Dynamic KV Cache：解耦 sequence-level 和 batch-level cache 操作，避免大 KV tensor 的 split-merge 开销；(6) Closed-Loop Feedback Controller：Inference Engine 在每个 attention 和 router stage 后回调 Scheduler，Scheduler 根据 user-defined policy 动态调整执行流。
  - 实验比较：(1) TTFT——QLLM vs HF TGI baseline，在 ShareGPT 数据集、20% LS/80% BE 混合负载下，Poisson 到达率从低到高；(2) Throughput——job completion rate vs request arrival rate；(3) Turnaround time——LS 和 BE 两类请求的 turnaround time 对比。QLLM 降低 LS TTFT 平均 65.2×（最高 101.6×），SLO 设置为 3s（10× 单次 decode iteration），QLLM 在 7 req/s 以内满足 SLO 而 baseline 任何负载下均不满足。LS turnaround time 降低最高 12.8×，BE turnaround time 增加 1.38×（最高 2.04×）。Throughput 持平或略优。
- 硬件平台是什么，配置是什么。
  - GPU: 单卡 NVIDIA A100 80GB HBM。CPU: 双路 Intel Xeon Gold 6336Y。DRAM: 256 GB。互联: PCIe 4.0。系统环境: bare-metal。
- 开源Serving框架是什么。修改了什么。
  - 开源框架：HuggingFace TGI (Text Generation Inference)，https://github.com/huggingface/text-generation-inference。
  - 修改内容：
    1. **MoE Layer 重设计**：在原有 MoE block 中插入 per-expert FIFO queues。每个 expert 队列独立存储待处理 token 的 Sequence 引用，eliminate layer-wise synchronization barrier。Router 输出的 top-k expert 选择结果将 sequence 引用 push 进对应 expert 队列，expert 从其队列中 pop 处理。
    2. **Scheduler 模块**：新增 Dispatcher（按优先级分派 jobs 到四个队列）和 Batch Engine（Algorithm 1：优先 LS_Decode → LS_Prefill → 填充 BE → BE_Decode → BE_Prefill）。Scheduler 在 LS 到达时通过 closed-loop feedback 通知 Engine 在任意 layer 处 preempt BE batch。
    3. **Sequence/Batch 抽象层**：新增 Sequence 对象封装 per-token 全部状态（KV cache tensors, hidden states, attention mask, residuals, routing metadata）。Batch 对象用 Facade Pattern 对外呈现为单一拼接 tensor，对内维护 per-sequence 独立 tensor，支持 zero-copy individual update。
    4. **Unified Dynamic KV Cache**：新增 KV cache 管理模块，解耦 sequence-level 和 batch-level cache ops，避免传统系统中 preempt 时的大 tensor split-merge。
    5. **Inference Engine** 改为 closed-loop feedback controller：每层 attention/router 阶段后回调 Scheduler，支持 user-defined preemption policy（<50 行 Python 实现）。
  - 论文声明 QLLM 是建立在 HF TGI 之上的原型系统，计划未来开源（"Our ultimate plan is to release QLLM as an open-source project in future versions"）。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：论文尚未开源（计划未来开源）。基于论文描述复原的全过程如下。
  - 框架输入到硬件执行全过程（Mixtral 8×7B, 4-bit 量化, FP16, batch_size=32, single A100 80GB）：
    
    **阶段 0 — 系统初始化与模型加载**：
    1. 基于 HF TGI 源码，继承其模型加载逻辑，加载 Mixtral 8×7B 权重（4-bit 量化，FP16 compute，约 22.93 GB GPU memory）。
    2. 替换原生 MoE block 为 QLLM MoE block：在每个 expert 前插入 FIFO queue 数据结构。初始化 Scheduler（Dispatcher + Batch Engine）和四个优先级队列。
    3. 初始化 Unified Dynamic KV Cache 管理器：预先分配 GPU memory pool 用于 dynamic cache 增长。
    
    **阶段 1 — 请求到达与调度（BE 先到，LS 随后）**：
    4. 4 个 BE decode requests（ShareGPT prompts）到达。Dispatcher 将它们 enqueue 到 BE_DecodeQueue。
    5. Scheduler Batch Engine 调用 GetNextBatch()：LS_DecodeQueue 和 LS_PrefillQueue 均为空 → 检查 BE_DecodeQueue 有 4 个 jobs → 返回 batch=[BE1, BE2, BE3, BE4]。
    6. Scheduler 将 batch 提交给 Inference Engine，进入 MoE layer 1 的 attention 计算。
    
    **阶段 2 — Expert-Level Preemption（LS 到达触发）**：
    7. 在 MoE layer 1 的 attention 阶段完成后，一个 LS job 到达。Dispatcher 将其 enqueue 到 LS_PrefillQueue。
    8. Closed-loop feedback：Inference Engine 在 attention 完成后回调 Scheduler。Scheduler 检测 LS_PrefillQueue 非空 → 发送 preempt 信号给 Engine。
    9. Engine 在 layer 1 的 router 阶段后暂停 BE batch：当前 BE batch 的 partial computation 状态（hidden states, routing_weights, expert assignments）通过 Sequence 对象的独立 tensor 原地保存，不需要 split BE batch tensor。
    10. Engine 切换到 LS job：执行 prefill——在 GPU SM 上并行处理 LS prompt 的全部 input tokens，生成 KV cache entries 并产出第一个 output token。
    11. LS job 转入 decode phase，Engine 执行 LS decode iteration 生成后续 token。
    
    **阶段 3 — 动态合并与恢复（LS 与 BE 同 batch 执行）**：
    12. LS decode 完成后，Scheduler 通过 Batch Engine 将 LS job 加入当前运行 batch。由于 Batch 的 Facade Pattern 对外表现为单一 tensor，model 无感知 batch composition 变化。
    13. Engine 在后续 layers 中同时处理 LS decode + BE decode：LS job 的 token 走 router → top-k expert selection → push 进入对应 expert queue → expert 从 queue pop 并执行 feed-forward。
    14. BE jobs 从 preemption point 恢复执行：Sequence 对象中保存的 routing_weights 和 hidden_states 被重新加载，Unified Dynamic KV Cache 恢复 BE 的 cache 行，无需 recomputation。
    
    **阶段 4 — 后续 Layers 的 Per-Expert Queue 执行**：
    15. 每个 MoE layer：Router 为每个 token（LS/BE 混合）计算 gating logits → softmax → TopK=2 → 将 sequence 引用 push 进选中的 2 个 expert 队列。
    16. 各 expert 独立从其 FIFO queue 中 pop token 执行 FFN 计算。Per-expert queue 的 FIFO 顺序确保 LS token 优先（因为 LS sequence 引用先于 BE 入队前的 preemption 点入队，或者通过 policy 显式优先 enqueue LS）。
    17. Expert 输出写入 Sequence 对象的 hidden_states tensor。Batch Facade 收集所有 Sequence 的 output 拼接为下一 layer 的 input tensor。
    18. 重复 layer 1..32（Mixtral 8×7B 的 32 层），每层 attention → router → per-expert queue → expert FFN → combine。
    
    **阶段 5 — 输出与完成**：
    19. 最终 layer 输出经 LM head 投影到 vocabulary，生成 logits → softmax → sample → output token。
    20. LS job 完成所有 decode iterations → Dispatcher 将其移出队列，output tokens 返回客户端。BE jobs 继续执行直至各自完成。

## Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Pre-gated MoE 的 system 部分——基于 CPU offloading 的 MoE 推理系统，通过 pre-gate function 实现 preemptive expert migration。核心设计：(1) 分层存储策略：dense non-MoE 参数（attention weights, embeddings, layernorm）常驻 GPU memory；sparse MoE 参数（全部 expert weights）完全 offload 到 CPU DRAM；(2) Preemptive Expert Migration：利用第 N 个 block 的 pre-gate function 提前知道 (N+1) 个 block 需要哪些 experts，在 GPU 执行第 N 个 block 的 expert computation 期间，从 CPU 经 PCIe 异步迁移仅激活的 experts 到 GPU；(3) 通信-计算重叠：pre-gate function 是轻量 MLP（计算量极小），所以 expert migration 阶段（蓝色，PCIe communication-bound）可与 expert execution 阶段（绿色，compute-bound）完全并行；(4) GPU 峰值内存公式：Peak_GPU_mem = max(Non_MoE_M + Σ_{L=N}^{N+1} Act_Exp_L)，即非 MoE 参数 + 连续两个 block 的激活 expert 参数之和。
  - 实验比较：(1) 单 MoE block 延迟——Pre-gated MoE vs MoE-OnDemand (按需加载) vs MoE-Prefetch (全量预取) vs GPU-only (oracular 上界，全部参数在 GPU)；(2) 端到端推理吞吐 (tokens/sec)；(3) 峰值 GPU 内存使用——Pre-gated MoE 仅占 GPU-only 的 23%，与 memory-optimal 的 MoE-OnDemand 几乎相同；(4) 模型准确率 vs 原始 SwitchTransformer；(5) Sensitivity studies——pre-gate activation level (N=0/1/2/3)、激活 expert 数量 (1~64)、叠加 expert caching (LIFO/LFU/LRU)、SSD offloading 场景。
- 硬件平台是什么，配置是什么。
  - CPU: AMD EPYC 7V12 64-Core, 1.8TB DDR4 memory。GPU: 单卡 NVIDIA A100 80GB HBM。互联: PCIe Gen4, 32 GB/s 单向数据带宽。系统配置：CPU-GPU (MoE 参数在 CPU，non-MoE 参数在 GPU) vs GPU-only (全部参数在 GPU，oracular 上界)。
- 开源Serving框架是什么。修改了什么。
  - 基于 NVIDIA FasterTransformer（https://github.com/NVIDIA/FasterTransformer），state-of-the-art CUDA 推理库。修改包括：(1) 实现分层参数存储——non-MoE 参数常驻 GPU，expert 参数 offload 到 CPU；(2) 实现 preemptive expert migration pipeline——在 MoE block N 的 expert execution 期间，异步启动 CPU→GPU cudaMemcpy 传输 (N+1) block 的激活 experts；(3) 利用 CUDA stream 实现通信与计算的重叠——expert migration 在一个 stream，expert computation 在另一个 stream；(4) 实现 pre-gate function 的 forward 逻辑——在 FasterTransformer 的 MoE block 中插入 pre-gate linear layer，输出传递给下一个 block；(5) 修改第一个 MoE block 使用双 gate（传统 gate + pre-gate），最后一个 block 无 pre-gate。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：GitHub https://github.com/ranggihwang/Pregated_MoE, Zenodo DOI: 10.5281/zenodo.10976343。Docker 镜像: nvcr.io/nvidia/pytorch:22.09-py3。编译：cmake -DSM=80 -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON。
  - 框架输入到硬件执行全过程（Switch-Base 128 experts, batch=1, single A100）：
    
    **阶段 0 — 模型加载与初始化**：
    1. 从 HuggingFace 下载 SwitchTransformer pretrained weights，fine-tune 为 Pre-gated MoE（2,048 steps）。
    2. 加载模型：non-MoE 参数（attention, embedding, layernorm）→ GPU HBM；所有 expert 参数 → CPU DRAM（1.8TB 充足）。
    3. 第一个 MoE block 加载两个 gate function（传统 gate + pre-gate），其余 block 各一个 pre-gate，最后一个 block 无 pre-gate。
    
    **阶段 1 — 第一个 MoE Block 执行（例外，无法重叠）**：
    4. Input hidden states x_0 ∈ R^{B×H} 进入 MoE block 0。
    5. 传统 gate: logits = W_gate @ x_0 → softmax → TopK → 选择激活 experts 集合 A_0。
    6. On-demand migration: cudaMemcpy(A_0 的 expert weights, CPU→GPU) —— 串行暴露 PCIe 延迟。
    7. Expert execution: GPU 计算 Σ w_i · Expert_i(x_0) for i ∈ A_0。
    8. 同时 pre-gate: logits' = W_pre_gate @ x_0 → softmax → TopK → 确定 block 1 的激活 experts A_1。
    
    **阶段 2 — 后续 MoE Block 执行 (N ≥ 1，核心优化)**：
    9. Input x_N 进入 MoE block N。A_N 已由 block (N-1) 的 pre-gate 提前确定。
    10. Expert execution 立即开始: GPU SM 执行 Σ w_i · Expert_i(x_N) for i ∈ A_N（compute-bound，约 2ms）。
    11. 同时 pre-gate: logits' = W_pre_gate @ x_N → softmax → TopK → A_{N+1}。
    12. 同时 preemptive migration: cudaMemcpy(A_{N+1} 的 expert weights, CPU→GPU) 在独立 CUDA stream 上异步执行（communication-bound，约 1-2ms 取决于 expert 大小）。
    13. Step 10 与 Step 11-12 完全重叠——expert execution 的 compute 时间 ≥ expert migration 的 PCIe 时间。
    14. 循环回到 Step 9 处理下一个 block。
    
    **阶段 3 — 最后一个 MoE Block**：
    15. 最后一个 block 无 pre-gate function（无需为不存在下一个 block 选择 experts）。
    16. Expert execution 完成后，输出经 layernorm → LM head → next token prediction。
    
    **阶段 4 — 测量**：
    17. block_lats.csv: 每个 MoE block 的平均延迟。
    18. throughputs.csv: end-to-end tokens/sec。
    19. peak_mems.csv: 峰值 GPU 内存使用。
    
    Pre-gated MoE 的核心作用：通过 pre-gate function 解耦 expert selection 与 expert execution 的数据依赖，使 CPU→GPU expert migration 与 GPU expert computation 完全重叠。对比 MoE-OnDemand（串行暴露 PCIe 延迟）和 MoE-Prefetch（传输全部 experts 浪费带宽和 GPU 内存），Pre-gated MoE 同时实现了接近 GPU-only 的性能和接近 MoE-OnDemand 的内存效率。

## ProMoE: Fast MoE-based LLM Serving using Proactive Caching

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：ProMoE 在 transformers 和 llama.cpp 两个主流 LLM 框架上实现 proactive caching 系统，通过 learned predictor + prefetcher 协调机制将 expert offloading 的数据传输移出推理关键路径。核心修改：(1) **Predictor 模块**——在 CPU 上运行二层的 MLP predictor（~2M 参数/层），基于前一层 hidden state 预测当前层将激活的 experts，以 layer-wise（或 stride）模式发出 prefetch 任务；(2) **Prefetcher 模块**——worker thread + 双优先级任务队列（LOW=speculative prefetch, HIGH=precise prefetch），通过 cudaMemcpyAsync 从 CPU memory 向 GPU memory 传输 expert chunk；(3) **Chunked Prefetching**——将每个 expert 参数按三个 linear layer 天然拆分为 3 个 chunk，使 worker thread 能以更细粒度调度，减少高优先级任务的等待延迟；(4) **Early Preemption**——在 gate function 完成后插入 hook 获取精确 expert 列表，清除同层 LOW 任务，将缺失 experts 作为 HIGH 任务入队；(5) **Reordered Inference**——在 gate 完成后根据 cache/prefetch 状态重排 expert 计算顺序：已缓存优先 → 正在 prefetch → 完全未开始，建立计算与 prefetch 的 pipeline；(6) **Cache 管理**——per-layer LRU cache，预分配连续 GPU memory 减少碎片。代码量 6,600 行 C++。
  - 实验比较：(1) Overall Performance（transformers 和 llama.cpp 两套 codebase）——ProMoE vs static cache vs LRU cache vs TO/UM/LO baselines，评估 5 个 MoE 模型（DS-1/DS-2/QW-1/QW-2/Mixt）的 TTFT 和 TPS/TPOT；(2) Ablation Study——逐步启用 prefetch/chunked-prefetch/early-preemption/reordered-inference 的加速贡献；(3) Impact of Cache Rate——cache rate 变化（10%-90%）对 TTFT 和 TPOT 的影响，含关键路径加载时间 breakdown；(4) Impact of Batch Size——batch size 1-4 对 prefill/decode throughput 的影响；(5) Impact of Model Size——BPW 从 4 到 16 变化对性能的影响。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX 4090（24 GB GDDR6X）
  - CPU：Intel i9-14900K，128 GB host DRAM
  - 互联：PCIe 4.0（单向 32 GB/s，实测可达带宽 23.9 GB/s host-to-GPU）
  - 量化：FP16（DS-1/DS-2/QW-1）和 INT4（QW-2/Mixt），INT4 使用 GPTQ 量化

- 开源Serving框架是什么。修改了什么。
  - 框架：HuggingFace transformers（https://github.com/huggingface/transformers）和 llama.cpp（https://github.com/ggerganov/llama.cpp）
  - 修改内容（集成到两个框架）：
    1. **MoE 层 hooks**——在每层 gate function 结束后插入 hook 捕获 gating 输出，获取精确 expert 列表（用于 early preemption 和 reordered inference）
    2. **Expert 计算顺序重排**——修改 MoE block 中 expert FFN 的执行顺序，按 cache/prefetch 状态重排（cached-first → prefetching → not-started），建立计算-prefetch pipeline
    3. **Predictor 集成**——在每层前向开始时，将前一层的 hidden state clone 到 CPU，CPU 上执行 MLP predictor 预测当前层 experts，通过 PushPredictedExperts API 入队 LOW 优先级 prefetch 任务
    4. **Prefetcher worker thread**——在推理进程中启动独立线程，轮询双优先级任务队列，通过 cudaMemcpyAsync 执行 expert chunk 的 CPU→GPU 传输
    5. **Memory manager 接管**——ProMoE 接管 expert 参数的内存管理，预分配 per-layer LRU cache 连续 GPU memory 区域
    6. **Dependency mechanism**——实现依赖机制确保 prefetch 与 computation 的正确同步（expert.ready_chunk 计数器跟踪每个 expert 的已加载 chunk 数）
  - 未集成到 vLLM/TGI 的原因：当时支持 MoE 量化不完善，且这些框架 fused expert execution 假设所有 experts 计算前就绪（不适合 memory-constrained GPU 上的 prefetch 场景）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：GitHub https://github.com/promoe-opensource/promoe
  - 框架输入到硬件执行全过程（ProMoE with transformers, DS-1 FP16, RTX 4090, single token decode）：
    ```
    初始化阶段：
    1. 非 expert 参数（attention, embedding, layernorm）常驻 GPU memory
    2. 全部 expert 参数在 CPU host memory，per-layer LRU cache 预分配 GPU memory
    3. 加载 offline 训练的 per-layer MLP predictor（~2M params/layer, 28 layers × 2M ≈ 56M params total）
    4. 启动 Prefetcher worker thread（CPU 线程，轮询双优先级任务队列）

    推理循环（第 l 层 MoE block）：
    5. Pre-attention norm 后 hidden state X ∈ R^{1×H} 被 clone 到 CPU
    6. CPU Predictor: 第 l-1 层 hidden state → MLP predictor_l → 预测第 l 层应 prefetch 的 experts
       → PushPredictedExperts(layer=l, experts=[e_pred_1,...,e_pred_k]) → LOW 优先级任务入队
    7. Prefetcher worker thread 开始异步传输预测 experts 的 chunk（CPU→GPU cudaMemcpyAsync）
    8. GPU 执行 self-attention（FlashAttention 或标准 attention）与 CPU prediction+prefetch 并行
    9. Gate function 完成 → hook 触发 → 获取精确 expert 列表 [e_1,...,e_k]
    10. Early Preemption: 清除第 l 层所有 LOW 任务 → Reordered: cached experts 排前 → PushPreciseExperts(layer=l, experts=reordered) → HIGH 任务入队
    11. Prefetcher worker thread 优先处理 HIGH 任务，以 chunk 粒度从 CPU 传输缺失 experts 到 GPU
    12. GPU 按重排顺序执行 expert FFN：cached experts 立即执行 → prefetching experts pipeline（计算与传输重叠） → 最后执行完全未开始 experts
    13. 每个 expert 计算时检查 ready_chunk 计数器，等待对应 chunk 就绪
    14. Expert 输出加权求和 → 进入第 l+1 层
    ```
  - 关键性能收益：ProMoE 将 key critical path 上的 expert 加载时间从 LRU 的 60.4% decode / 82.7% prefill 降至显著更低比例（QW-2 cache rate 增长时从 69.68% 降至 30.96%），通过 proactive prefetch 将大部分数据传输移出关键路径。

## PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：PROBE 在 SGLang 框架上实现 Continuous Lookahead Pipelining 机制，包含三个核心模块：(1) Gate-Initialized Lookahead Predictor——克隆目标层 router 参数并冻结作为先验，附加可训练残差 MLP，利用前一层 hidden state 预测下一层 expert 激活分布；(2) Hardware-Aware Balance Planning——单SM CUDA kernel 实现的贪心求解器，以 bottleneck rank 的 latency 为目标函数，在 hiding window 约束下动态复制 expert 到低负载 rank；(3) Phase-Locked Co-Scheduling——双轨架构，Predict/Plan/Update 阶段与 All-to-All Dispatch 重叠，Prefetch 通过 split-phase transmission 与 MoE Compute 和下一层 Attention 重叠。使用 NVSHMEM 管理 replicated-expert buffer，支持每 rank 最多 3 个冗余 expert，双缓冲异步写入。
  - 实验比较：PROBE vs SGLang（标准 EP + sharded placement）vs DeepSeek-EPLB（统计式负载均衡，2 冗余 slots）。比较指标：(a) Prefill Latency (TTFT)——chunked prefill 8K/16K tokens per rank；(b) Decoding Throughput-Latency Pareto frontier——per-rank batch size 512-1536；(c) Robustness to Semantic Shifts——Code→Chinese 突然切换下的吞吐稳定性；(d) Predictor Fidelity——Top-K Accuracy、Top-Half-K Hit-Rate、2×Top-K Recall；(e) Per-Layer Latency Breakdown。
- 硬件平台是什么，配置是什么。
  - 8×NVIDIA Hopper-141GB (H800) 节点，900 GB/s NVSwitch 互联。软件栈：PyTorch 2.9、CUDA 12.9、NCCL 2.27.3、NVSHMEM 3.3.20。
- 开源Serving框架是什么。修改了什么。
  - 基于 SGLang 框架。修改包括：(1) 集成 DeepEP (normal mode) 作为 All-to-All 通信后端；(2) 使用 NVSHMEM symmetric memory 管理 replicated-expert buffer region；(3) 实现轻量级全局 All-Gather（NVSHMEM primitives）用于同步 per-rank 预测结果；(4) 实现 Phase-Locked Co-Scheduling 双轨执行调度；(5) 实现 split-phase transmission 避免 P2P prefetch 与 All-to-All Combine 竞争带宽；(6) 使用双缓冲（double buffering）管理最多 6 个 expert slots（3 incoming + 3 outgoing）的异步写入。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未明确说明是否开源。框架使用流程：请求到达→SGLang continuous batching 组装全局 batch→进入 MoE 层时 PROBE Lookahead Predictor 读取前一层的 hidden states、预测下一层各 expert 的 token 分布→全局 All-Gather 聚合预测结果→单SM CUDA Planning Solver 在 All-to-All Dispatch 期间执行贪心复制决策→若决策复制 expert，通过自定义 Triton kernel 发起 P2P put 操作将 expert weights 从源 rank 传输到目标 rank→Split-phase：传输在 MoE Compute 期间进行，All-to-All Combine 前暂停，Combine 完成后恢复→下一层执行时复制已完成，按照更新后的 routing assignment A 进行 token dispatch→实现 straggler neutralization，将 IR 从 2.13 降至 1.09，prefill 加速最高 1.32×，decoding 吞吐提升最高 1.26×。

## Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Pre-attention same-layer expert prediction 与 prefetching pipeline。核心设计：(1) 在 pre-attention norm 后 clone hidden state 到 CPU，CPU 执行预测器推理（0.15ms），与 GPU 上的 self-attention（0.74-1.13ms）和 post-attention norm（0.08-0.13ms）并行；(2) 预测正确的 expert 在 attention 执行期间从 memory 预取到 GPU（每 expert 0.7-1.6ms），pipeline 到达 expert selection 时直接可用；(3) 预测错误时触发 emergency loading（5.6-8.3ms per expert from disk），但可被 expert computation（6.2-10.3ms）部分隐藏。支持三种部署策略：standard（精确 Top-K）、over-provisioning（多 load experts 换更高 hit rate）、top-1（边缘设备仅并行 load 1 个 expert）。
  - 实验比较：对比 FATE (cross-layer prediction, 78.79% accuracy)、DuoServe-MoE (54-67% top-2 accuracy)、HOB-BIT (55% cache hit rate)。比较指标：Exact-match accuracy、Over-provisioning accuracy、Top-1 accuracy。Expert loading latency 对比：disk→GPU vs memory→GPU，predictor overhead vs attention timing window。
- 硬件平台是什么，配置是什么。
  - 训练：NVIDIA TITAN RTX 24GB GPU。Profiling：Tesla V100-SXM2-32GB、NVIDIA A100-PCIE-40GB、NVIDIA A100 80GB PCIe。预测推理可在 CPU-only 上运行（无 GPU 要求）。
- 开源Serving框架是什么。修改了什么。
  - 论文未在现有开源 serving 框架（如 vLLM、SGLang）上修改。其 prefetching pipeline 设计是框架无关的：pre-attention norm → [fork: GPU self-attention || CPU predictor → expert prefetch] → post-attention norm → expert selection（hit: use prefetched; miss: emergency load）→ expert computation。论文给出了 parallel execution pipeline（Fig.8a-c）的时序设计，但未说明集成到具体 serving 框架的实现细节。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 未开源。Serving pipeline 全流程（DeepSeek-V2-Lite on A100-80GB）：
    - 输入：token 序列经过 embedding → 进入 MoE layer l。
    - Pre-Attention Norm (0.075ms)：RMSNorm 产生 hidden state X。
    - Fork：(a) GPU path——self-attention (0.739ms) → post-attention norm (0.080ms)；(b) CPU path——clone X → CPU predictor f_l(X) (0.15ms) → TopK(s, k) → 发起 expert prefetch 请求。
    - Expert Selection (0.102ms)：Ground-truth router 计算 g = Softmax(W_g · X')，TopK(g, k)。与预测 Ŷ 比对。
    - Hit case：prefetched experts 已在 GPU memory 就绪 → 直接进入 expert computation。
    - Miss case：紧急从 disk/CPU memory load miss 的 experts (single expert: 5.6ms from disk / 0.7ms from memory)，可与已就绪 experts 的 computation 重叠。
    - Expert Computation (6.8ms)：对已加载 experts 执行 FFN 计算。
    - 总 latency savings：1000-token inference session，93.03% accuracy → 569-1352ms total saved（vs FATE 78.79% accuracy）。

## MoEsaic: Shared Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：MoEsaic 在 vLLM 上实现多租户 MoE 模型的专家共享。核心实现包括四部分：(1) Expert Deduplication——在模型加载时计算每个 expert 张量的 128-bit hash digest，通过 in-memory dictionary 检测并去重跨模型实例的相同专家，使多个 client 共享同一份 GPU 显存；(2) Lazy Memory Allocation——使用 tiny pseudo experts 初始化模型，加载时才扩容并填充参数，避免预分配导致的内存不足；(3) Independent Expert Representation——将 vLLM 中 per-layer 的 co-located expert tensor 拆分为单个 expert 独立表示，支持张量级别的单独共享；(4) Fused Gate——将多个 model instance 的 gate 合并为单一 fused gate，批量处理路由请求，避免逐模型串行调用 CUDA kernel；(5) Merged Expert Representation——将去重后的相同专家合并为单一 nn.Parameter，使来自不同 client 的请求在专家计算时自动批处理。
  - 实验比较：MoEsaic 与 dedicated MoE instances (baseline) 对比。比较指标：(a) GPU Memory (GB)——MoEsaic 减少内存占用；(b) Inter-token Latency——fused gate vs separate gate 的路由延迟差异；(c) Throughput (tokens/s)——token 生成速率；(d) GPU Utilization——NVIDIA Nsight 测量的 SM 占用率；(e) Model Loading Time (s)——初始化耗时。MoEsaic 可服务 7× 更多 Mixtral-8x7B 变体且对推理性能影响不大。

- 硬件平台是什么，配置是什么。
  - GPU：8 × NVIDIA A100（40GB）。
  - CPU：64 × AMD EPYC 7742。
  - 推理流量：自定义 chat message 数据集，生成 512 token 序列。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：vLLM（https://github.com/vllm-project/vllm）。
  - 论文未提供独立开源仓库链接（SoCC '24 论文，发表于 2024 年 11 月，IBM Research），代码开源情况：论文未明确说明独立代码仓库链接。
  - vLLM 修改内容：
    1. **Lazy Allocation of Memory**：vLLM 原在模型初始化时预分配所有 expert GPU 显存。MoEsaic 改为用 tiny pseudo experts 初始化，加载时再扩容并填充参数。去重后最大仅占用去重后专家的显存量，避免"全部模型加载完才能去重"导致的内存峰值。
    2. **Independent Representation of Experts**：vLLM 中每层所有 expert 以单个 tensor co-located。MoEsaic 将每个 expert 独立表示为单独的 nn.Parameter 对象，使其内存可独立管理——相同专家共享底层 tensor 但保持独立参数对象。
    3. **Expert Population Tracking**：vLLM 中 in-memory 表示与 in-file 表示不同（多个 in-file tensor 对应一个 in-memory tensor）。MoEsaic 跟踪每个 expert 的张量分配状态，expert 完全填充后标记为"可去重候选"。
    4. **Tensor-Parallel Expert Loading**：vLLM 原生支持 TP 加载但不支持向已部署模型添加新 expert 的 TP。MoEsaic 新增 Ray workers——每个 worker 负责加载指定 GPU 上的 expert shard，新 expert 继承初始模型的 sharding 方式（如 4-way TP），去重在 shard 级别进行。
    5. **Fused Gate**：在每层 MoE layer 中合并多个 model instance 的 gating network 为单一 fused gate，一次性批量完成路由，对比 separate gate 减少了逐模型调用 CUDA kernel 的延迟开销。
    6. **Merged Expert Representation**：模型初始化后，将去重后共享相同底层 tensor 的 expert 合并为单一 nn.Parameter。每个 MoE 的 gate 将 expert ID 映射到合并后的表示，使 Triton kernel 中来自不同 client 的请求在专家计算时被自动批处理。
    7. **Non-disruptive Add/Remove**：支持在运行中动态添加/移除 model instance，无需系统重启（但不可在活跃推理期间执行）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文基于 vLLM（开源），但 MoEsaic 自身的修改代码未提供独立仓库链接。论文未明确说明是否开源。
  - 框架输入到硬件执行全过程（以 Mixtral-4x7B, 4 model instances, 2 shared experts 为例）：

    **阶段 0 — 模型加载与初始化**：
    1. 第一个 client 提交 Mixtral-4x7B 模型。vLLM 启动时，MoEsaic 用 tiny pseudo experts 初始化所有 expert 结构（不占用实际参数显存）。
    2. 从模型文件逐 tensor 加载参数。每加载完一个 expert（即所有 in-file tensor segment 聚合完整），计算该 expert 的 128-bit hash digest，存入 in-memory dictionary。
    3. 后续模型加载时，对每个新 expert 计算 hash，查 dictionary：若命中→新 expert 引用已有 tensor（共享显存）；若未命中→分配新 GPU 显存。
    4. 若 TP=4，生成 4 个 Ray workers，每个 worker 负责加载对应 GPU 上的 expert shard。新 expert 继承初始模型的 4-way sharding 策略。
    5. 所有 expert 加载去重后，将共享底层 tensor 的相同 expert 合并为单一 nn.Parameter 表示，供后续批处理使用。

    **阶段 1 — Gating 与路由**：
    6. 推理请求到达时，每个请求携带其所属 client 的 model_id。
    7. 在每层 MoE layer，MoEsaic 的 fused gate 接收所有请求的 hidden states X ∈ ℝ^(B×H) 和对应的 model_id 列表。
    8. Fused gate 在单次 CUDA kernel 调用中完成所有 model instance 的 gating 计算：对每个 model instance i，执行 Softmax(W_gate^i · X[model_i]) → TopK 选择专家。gate mapping 表将每个 model instance 的原始 expert ID 映射到合并后的 merged expert ID。
    9. 路由结果：每个 token 被分配到一个 merged expert ID，即使来自不同 client，只要路由到相同专家即进入同一个计算批次。

    **阶段 2 — Expert 计算**：
    10. Token-to-Expert Dispatch：按 merged expert ID 将 batch 中所有 token 分配至对应 expert。
    11. Triton kernel 执行各 expert 的 FFN 计算。由于共享专家使用单一 nn.Parameter，来自多个 client 的请求被自然批处理——例如 client 1 的 16 个 token + client 2 的 12 个 token 共 28 个 token 在同一 batch 中由 expert A 处理。
    12. 计算结果按 token 聚合为输出 Y，返回给各 client。

    **阶段 3 — 测量**：
    13. Inter-token latency：从发送请求到生成 token 的平均时间（包含 gating + expert 计算 + attention + KV cache 等全部开销）。
    14. Throughput：tokens/second。
    15. GPU Utilization：NVIDIA Nsight 测量的 SM 平均占用率。

    MoEsaic 的核心作用：在 multi-tenant MoE serving 场景下，通过专家去重减少显存占用（约 1.6× 到 7× 更多模型实例），并通过合并专家表示将多 client 请求自动批处理以提升 GPU 计算效率。

## Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Diff-MoE 是一个面向 MoE 稀疏 LLM 的高吞吐 batched inference 框架，基于 NVIDIA FasterTransformer 库构建。核心实现包括三部分：(1) 优先级管理器（Priority Manager），在离线微调阶段识别 globally hot experts（top-N per layer，分配 MaxP 优先级并永久固定），在线推理阶段动态调整 non-global experts 的优先级分数（激活 +Δinc，不活跃缓存 -Δdec_in，不活跃未缓存 -Δdec_out）；(2) 差分缓存层级（Differential Cache Hierarchy），将 GPU 显存组织为 per-layer high-priority cache (HPCi, 永久存储 globally hot experts)、per-layer medium-priority cache (MPCi, 动态管理 locally hot experts，容量为 HPCi 的 2 倍)、以及跨层共享的 low-priority cache (LPC, 临时存储 cold experts 和预取 experts，用后即清)；(3) 轻量级 GRU 预测器（6 层 GRU，在微调阶段收集的 expert 激活序列上训练），预测下一 MoE 层可能激活的 experts 并预取到 LPC 中。
  - 实验比较：Diff-MoE 与 3 种 SOTA offloading 方案对比：(a) DeepSpeed-Offload — 按需加载激活 experts，计算后立即驱逐；(b) Pre-gated MoE — 预取类方法，修改 gating 机制提前预取下一层所有 activated experts；(c) MoE-Infinity — 缓存类方法，全局共享缓存 + 基于估计重用概率的驱逐策略。比较指标：Cache Hit Rate、End-to-End Throughput (tokens/s)、Peak GPU Memory (GB)、Memory Efficiency (tokens/(GB·s))。Diff-MoE 平均吞吐提升 2.74× (vs DeepSpeed)、2.22× (vs Pre-gated MoE)、1.55× (vs MoE-Infinity)。

- 硬件平台是什么，配置是什么。
  - GPU：单卡 NVIDIA H200（141 GB HBM）。
  - CPU：2 × Intel Xeon Gold 6430。
  - Host DRAM：1 TB。
  - 互联：PCIe 5.0，双向带宽 128 GB/s。
  - 操作系统：Ubuntu 22.04。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：NVIDIA FasterTransformer v5.2（https://github.com/NVIDIA/FasterTransformer）。
  - 论文开源代码：https://github.com/ceciliawinter/Diff-MoE.git（DOI: 10.5281/zenodo.15879848）。
  - FasterTransformer 修改内容：
    1. **Expert 参数粒度拆分**：将原始 HuggingFace 格式模型的 bin 文件拆分为细粒度 expert 参数文件，使每个 expert 可独立从 host 加载到 GPU。
    2. **差分缓存层级注入**：在 FasterTransformer 的 MoE 层执行路径中注入 HPC/MPC/LPC 三级缓存管理逻辑。HPC 在推理启动前预先加载 globally hot experts，推理过程中不驱逐；MPC 按优先级驱动替换策略动态管理；LPC 作为临时缓冲，当前层计算完成后清空。
    3. **Gating 后拦截**：在 gating network 输出 top-K experts 后，拦截 expert 加载流程——先检查 HPCi ∪ MPCi ∪ LPC 是否已缓存目标 expert，仅对缺失的 expert 触发 host→GPU 数据传输。
    4. **优先级更新钩子**：在每层 MoE 计算结束后，按公式 (1) 更新所有 non-global experts 的优先级分数，触发 locality-preserving cache replacement。
    5. **预取流水线**：在当前层 expert 加载完成后，利用 GRU predictor 预测下一层所需 experts，异步预取未缓存的 top-1/2 experts 到 LPC，与当前层计算并行。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：代码已开源在 https://github.com/ceciliawinter/Diff-MoE.git，同时提供完整的 README 复现指南。
  - 框架输入到硬件执行全过程（以 Switch-Base, XSum, batch_size=64 为例）：

    **阶段 0 — 离线准备**：
    1. 下载 HuggingFace 格式的 pretrained Switch-Base 模型（7B, 12 layers, 6 MoE layers × 128 experts）。
    2. 使用提供的脚本将模型 bin 文件拆分为 per-expert 细粒度参数文件（T1）。
    3. 在各下游任务数据集（XSum/SQuAD/CoQA）上微调模型，同时记录每个 MoE layer 的 expert 激活频率，按激活频率降序选出 top-2 globally hot experts per layer（默认 α=5% 缓存比，HPC=2, MPC=4）（T2）。
    4. 用微调阶段收集的 expert 激活序列训练 6 层 GRU predictor（训练/验证 8:2 划分）（T3）。

    **阶段 1 — 推理初始化**：
    5. 将非 MoE 参数（attention weights, embeddings, layernorm 等）常驻 GPU memory。
    6. 将每个 MoE layer 的 globally hot experts（HPCi，各层 2 个，共 6×2=12 experts）从 host 加载到 GPU HPC，永久锁定。
    7. 将所有其他 expert 参数保留在 host DRAM 中，初始化优先级分数为 0。

    **阶段 2 — 在线推理循环（以第 i 个 MoE layer 为例）**：
    8. **Token Embedding 输入**：batch_size=64 的 token embeddings X ∈ ℝ^(64×S×H) 经过前置 attention layer 后进入 MoE layer i。
    9. **Gating**：FasterTransformer 执行 `Softmax(LinearGate(X))` → 得到 gating 权重 G，`TopK(G, k=1)` → 每 token 选出 1 个 activated expert，集合为 A。batch=64 时大约 30-34 个不同 experts 被激活。
    10. **Expert 查找**：遍历 A 中每个 expert E_k^i，检查是否在 HPCi ∪ MPCi ∪ LPC 中。若缺失（cache miss），触发 cudaMemcpy 从 host DRAM 经 PCIe 5.0 加载 expert 参数到 GPU LPC 的激活缓冲区。
    11. **并行计算**：Token-to-Expert Dispatch——将 batch 中每个 token 分配到其 gating 选择的 expert。各 expert 在 GPU 上并行执行 FFN 计算。结果按 token 聚合为输出 Y。
    12. **优先级更新**：按公式 (1) 更新所有 non-global experts 的优先级：p_k^i = clip(p_k^i + 1) for E_k^i ∈ A；p_k^i = clip(p_k^i - 0.4) for inactive cached；p_k^i = clip(p_k^i - 0.2) for inactive uncached。阈值 threshold_hot = 1。
    13. **LPC → MPC 晋升**：locality-preserving replacement —— 若当前 activated 但未在 MPCi 的 expert 优先级 ≥ threshold_hot（即刚激活一次），候选按优先级降序排列；MPCi 中优先级 < threshold_hot 的 resident 按优先级升序排列；用最高优先级候选替换最低优先级 resident，直到无符合条件者。所有冷 experts 从 LPC 驱逐。
    14. **下一层预测与预取**：GRU predictor 接收当前层 A 中的 expert IDs，通过隐藏状态建模历史激活模式，输出下一层 i' 的各 expert 概率分布。batch_size>4 时聚合各样本分布取 top-2 未缓存 expert，异步 cudaMemcpy 预取到 LPC 预取缓冲区。

    **阶段 3 — 测量**：
    15. 所有性能指标（throughput, memory, cache hit rate）从 FasterTransformer 内嵌的日志钩子输出。每个配置重复 3 次取平均。

    Diff-MoE 的核心作用：通过分级缓存 + 优先级策略减少 expert migration 次数（提升 cache hit rate），通过 GRU predictor 将 host→GPU 传输与 GPU 计算重叠（隐藏通信延迟），在 batched MoE 推理中克服 PCIe 带宽瓶颈。

## PiKV KV Cache Management System for Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：PiKV 是一个面向 MoE 架构的并行分布式 KV Cache 管理框架，包含四个协同模块：(1) Expert-Sharded Distributed KV Storage——通过 hash 函数 h(t,e) 将 KV cache 按 token 和 expert 维度分片分布到多 GPU，每个 GPU 仅存储 O(L/G + L/E) 个 token 的 KV；(2) PiKV Routing——支持 7 种路由策略（Base hash/TopK/Load-Balanced/Cache-Aware/Entropy-Penalized/RL-Adaptive/Hierarchical），将 KV 查询复杂度从 O(BLhE) 降至 O(BLhk)；(3) PiKV Compression——集成 LoRA、PyramidKV、ChunkKV、Truncated SVD、FastV、Distillation、Structured Pruning，压缩比 ρ=d/d' 线性降低 read+decode 时间；(4) PiKV Scheduling——实现 H2O/StreamingLLM/QUEST/FlexGen/LRU/LRU+/AdaKV/Duo Attention 等调度策略，基于 per-page utility score（注意力强度+访问热度+复用模式）自适应驱逐。四模块通过异步流水线编排。
  - 实验比较：论文正文未包含独立定量 Evaluation 章节。GitHub README 提供合成 benchmark：(a) Standard MoE vs PiKV 各配置内存占用（100%→30%-85%）和推理速度（1.0×→1.3×-2.5×）；(b) Cache Hit Rate 最高 95%、Throughput 最高 3×、SLO 合规率 99%+；(c) 压缩方案对比：LoRA/Pyramid 2.1×/1.8×/98%，SVD 3.2×/1.6×/96%，Quantization 4.0×/2.2×/94%。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明实验硬件平台。代码仓库支持 CUDA 11.8+、PyTorch 2.0+ 的 GPU 平台；分布式支持 RDMA；FPGA 子系统支持 AMD Alveo U55C（xcu55c-fsvh2892-2L-e）。README 推荐 8GB+ RAM（大模型 16GB+）。

- 开源Serving框架是什么。修改了什么。
  - 开源 Serving 框架：vLLM（`core/single/vllm_integration.py`），同时支持 DeepSpeed（`core/distributed/deepspeed_integration.py`, ZeRO-1/2/3）。
  - vLLM 修改：(1) 注入 PiKV Routing——在 MoE gating 前替换为标准路由器；(2) 注入 PiKV Compression——KV 写入前压缩 (K,V) 对；(3) 注入 PiKV Scheduling——替换默认 cache eviction；(4) PagedKVCache 多级存储（GPU/CPU/SSD）；(5) DistributedKVCachePool RDMA 跨节点缓存传输与负载均衡；(6) CacheAwarePrefillScheduler TTFT SLO 约束下优化缓存复用；(7) LoadBalanceDecodingScheduler TBT SLO 约束下最大化吞吐。
  - 开源代码：https://github.com/NoakLiu/PiKV（155 commits, Python 84%/CUDA 6.5%/Verilog 4.3%, 4 releases, 最新 v2.1, ICML 2025 ES-FoMo III）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：完全开源（Apache 2.0 风格 LICENSE）。arXiv:2508.06526。
  - 框架输入到硬件执行全过程（PiKV Enhanced, MoE LLM, multi-GPU）：

    **初始化**：加载 MoE 模型 → create_moe() 选路由策略 → PiKVvLLMEngine 配置 world_size/expert_count/top_k → create_compressor('pikv_unified') 选压缩方案 → 选调度策略（AdaKV） → 初始化 PagedKVCache（GPU/CPU/SSD 三级 pool） → 初始化 DistributedKVCachePool RDMA。

    **Prefill**：prompt tokens → embedding → PiKV Router gating（EPLB: load-balanced softmax TopK） → PiKV Compression 压缩 (K,V)（high-importance→LoRA rank-r matvec, medium→PyramidKV, low→FastV tail crop） → Expert-Sharded Storage: hash s(t,e) = (t mod N_tok) ⊕ (e mod N_exp) 分配 shard → circular buffer O(1) 插入。

    **Decoding**：新 query q_t → Router 选 top-k experts g_t → 仅从 g_t shards 检索 KV（page table Γ lookup） → PiKV Scheduling 计算 utility scores u_i（AdaKV: u_i = Σ_j α_j φ_j(i), θ ← θ + γ(η*-η)） → 低分 pages 驱逐 → 命中 pages 解压（LoRA decode: K̂ + W_d W_u K̂ + b） → FlashAttention f_attn(q_t, C[q_t]) → 新 (K,V) 压缩插入。

    **调度维护**：LoadBalanceDecodingScheduler 监控 per-GPU 负载 → 动态调整 token-to-expert → AdaKV 每 Δ tokens MMIO 更新 θ。

    PiKV 核心作用：将 MoE KV cache 从"全量存储+全量访问"转为"稀疏路由+分片存储+压缩+自适应调度"四层协同，内存 O(EL)→O(L/G+KS)，查询 O(BLhE)→O(BLhk)。

## SiDA Sparsity-Inspired Data-Aware Serving for Efficient and Scalable Large Mixture-of-Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：SiDA-MoE 是一个数据感知的 MoE 推理服务系统，在 HuggingFace Transformers 的 Switch Transformer 实现之上构建。核心架构包含两个并行线程：(1) **Hash-building 线程**——用离线训练的 LSTM + Sparse Attention hash 函数预先预测每批输入在各 MoE 层的 expert 激活模式，写入 hash table 队列；(2) **Inference 线程**——根据 hash table 将激活的 expert 动态加载到 GPU，将未激活的 expert 卸载到 CPU 主内存（FIFO 策略），并用 SiDA-MoE 特化层执行前向推理。所有 router 函数被卸载到主内存，不参与前向过程。两条线程通过管道并行机制运行，使得 expert 选择、动态 offloading 和推理完全并行化。
  - 实验比较：(a) GPU 内存节省：SiDA-MoE vs 原始 Switch Transformer 在不同数据集（SST2/MRPC/MultiRC）和模型规模（8/64/128/256 experts）下的GPU内存减少比例；(b) 吞吐量与延迟：SiDA-MoE vs Standard vs DeepSpeed vs Tutel 在四条模型上的吞吐量和延迟对比；(c) 有限 GPU 内存预算下的效率：不同 GPU 内存预算、不同 offloading 方案下的吞吐量对比；(d) 保真度分析：SiDA-MoE 相对于 fine-tuned Switch Transformer 的性能保持率（SST2 accuracy / MRPC F1 / MultiRC F1）；(e) Hash 命中率：Top-3 expert 预测准确率；(f) 困惑度：SiDA-MoE 替代 router 后预训练模型的 perplexity 退化。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB（单卡）
  - CPU：64 Intel(R) Xeon(R) Platinum 8358 @ 2.60GHz
  - 软件栈：HuggingFace Transformers, PyTorch, CUDA

- 开源Serving框架是什么。修改了什么。
  - 基础框架：HuggingFace Transformers（基于其 Switch Transformer 实现）
  - 修改内容：(1) 新增 Hash-building 线程——实现 LSTM+sparse attention hash 函数的离线训练和前向预测，构建 expert 激活 hash table；(2) 新增 Inference 线程——替换原始 MoE 前向流程，插入 expert 动态加载/卸载逻辑（GPU ⇄ CPU 主内存），基于 FIFO 的 expert 驱逐策略；(3) SiDA-MoE Manager——协调主推理线程和预测线程间的调度，通过共享队列同步 hash table，管理 expert 设备置放和 GPU-CPU 数据传输；(4) 双线程管道并行——推理线程处理当前 batch 时，hash-building 线程并发预测下一 batch 的 expert 激活模式。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：https://github.com/timlee0212/SiDA-MoE
  - 框架输入→硬件执行全过程（单个推理 batch 的处理流程）：
    ```
    输入：token序列 batch X_i = [seq_len×1]（batch_size=1）
    
    1. Hash-building 线程（与推理并行运行，处理下一批 X_{i+1}）：
       X_{i+1} → token embedding → LSTM层（2层）→ Sparse Attention层（SparseMax激活）
       → FC层（维度压缩）→ Residual连接 → 最终FC层 → top-k expert选择
       → hash table H_{i+1}[layer][token] = {activated_expert_ids, scaling_factors α}
       → 推入 Shared Queue
    
    2. Inference 线程（处理当前批 X_i）：
       a) 从 Shared Queue 取出 H_i（等待 hash-building 线程完成）
       b) 对每个 MoE layer l：
          - 扫描 H_i[l][:] 获取本层激活的 expert id 集合
          - 对本批激活的 expert：
            if expert not on GPU: CPU→GPU 加载 expert 参数（θ_i）
          - 对本批未激活的 expert：
            if expert on GPU and GPU memory budget exceeded: GPU→CPU 卸载（FIFO策略）
       c) 前向传播（SiDA-MoE 特化层）：
          - Self-Attention: Q/K/V projection → Attention → output（不变）
          - MoE层: 跳过 router（已卸载到CPU），直接根据 H_i[l][token] 的
            (expert_id, α) 调用对应 expert MLP → α 加权求和 → 输出
          - 每层完成后立即触发下一层的 expert 加载/卸载（管道并行）
       d) 输出 logits
       
    3. 硬件执行映射：
       - Hash-building 在 CPU 上运行（使用 PyTorch CPU 推理）
       - 主要的 Transformer 推理在 A100 GPU 上执行
       - Expert 参数传输：CPU 主内存 (DDR4) → PCIe → GPU HBM（A100 80GB）
       - 卸载方向：GPU HBM → PCIe → CPU 主内存
       - 未激活 expert 存放在 CPU 主内存中（可达 TB 级）
    ```


## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Stratum 的 Serving 调度系统包含四个核心模块：(1) **Topic-Aware Request Scheduler**——接收包含 topic tag 的 inference requests，SLO-aware 地将同一 topic 的请求批量分组，优先 dispatch 同 topic 请求以最大化 hot expert hits。SLO 定义为 TTFT（Time to First Token），确保请求不等待过久。(2) **Request Generator（Poisson Process）**——模拟不同 topic 的请求以定义速率到达，生成 realistic serving workload。(3) **Memory and Computation Mapper**——Memory Mapper 按 Algorithm 1 聚合 batch 内所有 topic 的 expert usage 表 → 计算最大化 hot expert hit 的 expert placement。Computation Mapper 将 prefill phase 分配给 xPU、decode phase 分配给 Stratum NMP（类似 AttAcc 策略）。Memory reconfiguration（expert swap）在两次 dispatch 之间执行。(4) **Expert Swap 机制**——当 scheduler 切换到新 topic batch 时，通过 NMP 的 row-swap buffer 在 Mono3D DRAM bank 内执行 tier-to-tier expert 迁移，避免 traversing DRAM-xPU interposer 接口。
  - 实验比较：(a) System-level decoding throughput：Stratum tiering vs GPU baseline (vLLM 0.8.1) vs Stratum no-tiering vs Duplex，四种模型（OLMoE/Mixtral/Qwen2.5/Llama-4），不同 input/output lengths；(b) Energy efficiency：同样配置下的能效对比；(c) Batch size scaling (1-32)：Stratum-XL on Llama-4-Scout，4.7-9.8× throughput vs GPU baseline；(d) Expert swap overhead：<0.37% time, <0.03‰ energy；(e) SLO-aware scheduling 效果：same-topic batching 最大化 hot expert hit rates。

- 硬件平台是什么，配置是什么。
  - Stratum-S：NVIDIA RTX A6000 + 1 Mono3D DRAM chip (32GB), 16 channels, 16 banks/channel
  - Stratum-L：NVIDIA H100 SXM5 HBM3 + 6 Mono3D DRAM chips (192GB total), 1024-bit xPU-DRAM I/F @ 6.4 Gbps/pin
  - Stratum-XL：2× Stratum-L modules, 384GB total, NVLink cross-chip interconnect
  - GPU baseline：vLLM 0.8.1 on RTX A6000 / H100 SXM5 HBM3 GPUs
  - GPU energy：measured via NVIDIA-SMI tool
  - System-level simulator：自研 in-house simulator，包含 Request Generator（Poisson process）、SLO-Aware Scheduler、Memory/Computation Mapper、Stratum NMP interface

- 开源Serving框架是什么。修改了什么。
  - 论文未基于开源 Serving 框架。Stratum 使用自研 system-level simulator（非开源）和自研 NMP simulator 进行端到端 serving 评估。GPU baseline 使用 vLLM 0.8.1 但不修改其代码——仅作为对比基准。
  - System-level simulator 架构：Request Generator（Poisson 到达，topic-tagged）→ SLO-Aware Scheduler（动态 batching，优先同 topic dispatch）→ Memory Mapper（Algorithm 1 expert placement）→ Computation Mapper（prefill→xPU, decode→NMP）→ Stratum NMP Simulator（cycle-level execution + energy accumulation）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文未开源 Stratum 系统代码。GPU baseline 使用开源 vLLM 0.8.1。
  - Serving 执行全过程（Stratum-L, Mixtral 8×7B, batch=1-32）：
    ```
    输入：多个 client 发送不同 topic 的 inference requests
    
    阶段 0 — 预处理：
    1. Topic Classifier (CPU, <10ms): 每个 query → DistillBERT inference → topic label (6 classes)
    2. Scheduler 按 topic 将 requests enqueue 到对应 queue，维护 per-request SLO (TTFT)
    
    阶段 1 — Batch 构建与调度：
    3. Scheduler 按 SLO 约束 periodically 检查 queue：
       - Within SLO slack: 优先等待同 topic 请求到达 → 最大化同 topic batch
       - Near SLO deadline: 立即 dispatch 现有 batch（混合 topic）
    4. Scheduler 将 batch dispatch 到 Stratum 处理系统，携带 topic tag 列表
    
    阶段 2 — Memory Mapping (Expert Placement):
    5. Memory Mapper 读取 batch topic tags → 查询 per-topic expert usage table → 聚合
       → 按 Algorithm 1 计算目标 expert placement（hot→fast tier, cold→slow tier）
    6. 若当前 placement ≠ target: 触发 expert swap（near-memory 操作，row-swap buffer）
    
    阶段 3 — 计算分配与执行：
    7. Computation Mapper: Prefill tokens → xPU (H100 GPU), Decode tokens → Stratum NMP
    8. xPU: Gating network forward (lightweight linear layer: 4096→8) → routing decisions
    9. xPU: 发送 input tokens + expert IDs + scaling weights → Mono3D DRAM → switch to NMP mode
    10. Stratum NMP: 顺序执行 activated experts (sequential, tensor-parallel across all PUs)
        - GeMM1 + GeMM2 (projection-up) → Activation (SiLU) + Hadamard → GeMM3 (projection-down)
        - Reduce-scatter 与下一 expert GeMM1 并行（通信-计算 overlap）
    11. NMP: Weighted sum of expert outputs → write back to DRAM → exit NMP mode
    12. xPU: 读取 output tokens from designated DRAM address space
    
    阶段 4 — KV Cache Management（Attention 处理）：
    13. xPU: 写入新生成 KV pairs 到对应 DRAM channels
    14. Stratum NMP: 使用 head-level parallelism 执行 attention（PU groups 分区）
        - 每 PU group 处理多个 heads，interleaved Softmax/MatMul pipeline
        - Query 通过 sub-ring all-gather 分发，减少跨 bank 访问
    
    阶段 5 — 循环与完成：
    15. 重复解码循环直至所有 requests 完成
    16. 输出 tokens → 返回客户端
    
    性能指标：
    - Decoding Throughput (tokens/s): GPU vs Stratum no-tiering vs Stratum tiering
    - Energy Efficiency (tokens/J): 同样对比
    - TTFT SLO compliance: scheduler 保证请求在约束时间内开始处理
    ```


## Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：提出 AMoE——一个兼容 vLLM 的异步 Expert Parallelism (AEP) 原型系统，从零实现 6K 行 Python + 4.8K 行 C++. 核心实现包括：
    - **µ-queuing（微队列）**：将 token 按 layer 粒度而非全局 batch 排队，每个 GPU 为每个 expert layer 维护独立的 µ-queue，使 GPU 可以自由选择任意 ready layer 执行。
    - **Defragging Scheduler（Algorithm 1）**：为每个 (block, expert) pair 计算 Score = LScore + Q[b][e]，其中 LScore = sum_{k=1}^{K} (TotalTokens_{b+k} / N_e) × δ^k，lookahead K 个 block 并以衰减因子 δ 加权，优先调度下游 token 密集的 layer 以合并碎片化 mini-batch。
    - **Attention-Expert 解耦架构**：attention 层使用 Data Parallelism (DP) 部署在部分 GPU，expert 层使用 Expert Parallelism (EP) 部署在其余 GPU，两类 GPU 异步通信不阻塞。
    - **两阶段异步通信（Figure 8）**：Phase 1 通过 ZeroMQ CPU 消息队列传递 metadata (tensor size, rank)；Phase 2 通过 NCCL P2P 直接 GPU-to-GPU 传输 tensor，CPU 不等待 NCCL 完成即处理下一个传输任务。
    - **Token 级依赖追踪**：每个 token 携带 metadata <RequestID, LayerID, Tensors[], prefill_length, topk_weights>，使异步重排序执行中仍可正确追踪请求归属和下一层目标。
    - **Coordinator-Runtime 架构**：Coordinator (CPU) 包含 API Server (tokenizer/detokenizer, 请求状态管理)、Load Balancer (按 GPU memory 分配 DP rank)、Cluster Manager (GPU 内存追踪，通信通道建立)；每个 GPU 一个 Runtime 实例负责该 GPU 上所有层的执行。
    - **CUDA Graphs 逐层优化**：为每个 layer 独立记录 G 个 CUDA Graph（不同 batch size），共计 L×G 个 graph，通过共享 input buffer 减轻 GPU memory 压力，但 expert 层不使用 graph（因 GEMM 主导时 kernel launch latency 可被第一个 GEMM 掩盖）。
    - **Execution 四阶段流水线**：Receptor (按 LayerID 分流入 µ-queue) → Scheduler (选最优 layer) → Executor (page table 管理 + kernel launch) → Dispatcher (按 expert/DP rank permute tokens 后发送)。
  - 实验比较：
    - (a) **Top-1 routing throughput-latency (Figure 9a-c)**：AMoE vs SGLang (EP)，Mixtral 8x7B，8× A100 80GB。在 Short/Medium/Reasonable 三种 workload 下，AMoE throughput 分别提升 2.7×/2.3×/2.0×；低负载下 AMoE ITL 略高（layer-wise scheduling overhead + attention disaggregation 延迟）。
    - (b) **Top-2 routing (Figure 9d-f)**：AMoE throughput 优势减小，因 Top-2 (12.5%→25% expert activation) 降低 load skew，且 token merge 引入部分同步点。
    - (c) **多节点可扩展性 (Figure 10)**：16 experts + 16 GPUs (2×AWS P4)，medium workload + Top-1。AMoE throughput 3× vs SGLang，从 8→16 GPU 实现 1.92× 线性扩展，SGLang 无扩展。
    - (d) **Scheduler 消融 (Figure 11/12)**：defragging vs MTFS (most-token-first) vs FLFS (first-layer-first) 在 80% 最大 throughput 下的 ITL 和 throughput；FLFS 存在新请求打断高层 block 导致 输出率低于输入率 的 live-lock 问题。
    - (e) **Execution breakdown (Figure 13)**：attention step 2.7ms (page table overhead 显著)，expert step 0.8ms (GEMM 计算主导)，scheduling stage 仅占总时间小部分（C++/CUDA 优化效果）。
- 硬件平台是什么，配置是什么。
  - 单节点 (Lambda)：8× NVIDIA A100-SXM4-80GB，NVSwitch 600 GB/s per GPU，CUDA 12.8，NCCL 2.25.1，2× AMD EPYC 64 cores，1800 GB RAM，Ubuntu 22.04。
  - 多节点 (AWS P4)：2× p4dn.24xlarge，每节点 8× A100-SXM4-40GB，NVSwitch 600 GB/s，4× 100 Gbps EFA，CUDA 12.4，cuDNN v9.1.0，NCCL 2.22.3，CPU 2× AMD EPYC 64 cores @ 1.5 GHz，988 GB RAM。
- 开源Serving框架是什么。修改了什么。
  - 框架：AMoE 从零构建，但 runtime 中 model executor 复用了 vLLM 的 paged attention 和 CUDA graph 等优化基础设施。Communicator, Receptor, Scheduler, Dispatcher 用 C++ 实现 + pybind11 暴露 Python 接口，以规避 Python GIL 并保证各组件并发运行。Scheduler 和 Executor 在主 Python 线程运行，Receptor 和 Dispatcher 在独立 POSIX 后端线程运行。
  - 关键修改（相对标准 EP serving）：
    1. **Scheduling**：从全模型同步 batch 调度 → 逐层异步 µ-queuing + defragging scheduler（Algorithm 1）
    2. **Communication**：从 barrier all-to-all → ZeroMQ (CPU metadata) + NCCL P2P (GPU tensor) 两阶段异步通信
    3. **Execution model**：从固定 batch 遍历所有 layer → GPU 按 Score 自主选 layer 执行，cold expert tokens 积累到足够 batch size 才执行
    4. **Architecture**：Attention-Expert disaggregation → 不同类型层部署到不同 GPU 组，独立扩展
    5. **Token tracking**：新增 metadata-based token dependency tracking 支持异步乱序执行
- 开源情况。论文声明将开源 AMoE（"We open-source our serving system, AMoE, for public use"），**但论文全文及 arXiv 页面均未给出具体 GitHub URL，当前无法确认开源仓库地址。**
  
  基于论文描述的 AMoE Serving 全流程（Figure 5-6 对应）：

  ```
  [请求到达]
      │
      ▼
  API Server (Coordinator/CPU)
    ├─ tokenizer: 将 request text → token embeddings
    ├─ Load Balancer: 按 GPU memory 选最空闲 attention DP rank
    └─ 为 token 附加 metadata → 发送至对应 GPU Runtime
      │
      ▼
  [GPU Runtime - Attention Worker]
      │
      ├─ Communicator (Phase 1: ZeroMQ CPU metadata exchange)
      │     └─ sender 告知 receiver tensor size + GPU rank
      ├─ Communicator (Phase 2: NCCL P2P GPU direct transfer)
      │     └─ ncclSend/ncclRecv on CUDA stream, CPU 不等待完成
      ▼
  Receptor (C++ POSIX thread)
    ├─ 按 token.LayerID 将 token 分入对应 (block#, expert#|attnDPrank) µ-queue
    └─ Top-K: token pool 等待 K 路输入全部到达 → merge → 入队
      │
      ▼
  Scheduler (Defragging Algorithm 1, main Python thread)
    ├─ 遍历所有 (block, expert) pairs
    ├─ 计算 Scores[b][e] = LScore(lookahead) + Q[b][e]
    └─ 选 argmax → drain 该 µ-queue 全部 tokens → 合成 batch
      │
      ▼
  Executor
    ├─ [Attention layer] Page Table: 为 new tokens 分配 KV cache slot
    ├─ Pre-processing: fused batched CPU→GPU metadata transfer (dedicated stream)
    ├─ Kernel: paged attention or expert GEMM (mixture of kernels)
    ├─ Post-processing: GPU→CPU routing info (expert indices + weights) 
    └─ CUDA Graph: attention 层用预录制 graph 加速小 batch；expert 层不用 (GEMM 掩盖 kernel launch)
      │
      ▼
  Dispatcher (C++ POSIX thread)
    ├─ Attention output → permute by expert ID → 分组发送到各 expert GPU
    ├─ Expert output → permute by attention DP rank → 发送回 attention GPU
    └─ 递增 LayerID → 循环到下一 block
      │
      ▼
  [最后一层 attention output → Sampler (在首层 attention GPU)]
      └─ sample next token → detokenizer (API Server) → 返回用户
  ```

## Toward Efficient Inference for Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出两个针对 MoE 推理 serving 的调度优化：
  (1) **Expert Buffering**：利用 expert 激活的时序局部性（temporal locality），在 GPU 显存中仅保留热 expert 参数，其余 expert 参数缓存在 CPU 内存中。当冷 expert 被激活时，通过 PCIe 从 CPU 向 GPU 传输参数（与 token 传输重叠）。采用 LIFO cache eviction 策略，适配 MoE 中 experts 按 ID 顺序执行的特性——evict 最近使用的 expert 以保留复用距离最短的 expert。
  (2) **Load Balancing**：基于运行时 expert 激活数据优化 expert 到 GPU 的放置。(a) Greedy Balancing：按 expert 历史平均负载排序，贪心分配到负载最小的 GPU，约束每个 GPU 等量 experts；(b) Anti-Correlation Balancing：针对 MT-Decoder 中 expert 激活相关的场景，在负载估计中引入 Pearson 相关系数惩罚，避免相关 experts 放置到同一 GPU。
  实验比较：与原始 Fairseq（无 expert buffering、无 load balancing）对比，评估 throughput、memory usage、cache miss rate（vs Belady's MIN）、load distribution（Max load / Avg-Max load）。

- 硬件平台是什么，配置是什么。
  - *Apple* 集群：8×NVIDIA Tesla V100 (32GB)，NVLink 互联，2×Intel Xeon E5-2698 v4，700GB CPU DRAM，16GB/s PCIe 3.0。支持 1/2/4 node。
  - *Pear* 集群：4×NVIDIA RTX A5000 (24GB)，2×Intel Xeon Gold 5317，64GB CPU DRAM，32GB/s PCIe 4.0。仅单节点。

- 开源Serving框架是什么。修改了什么。
  开源框架：**Fairseq**（Meta 开源的序列建模工具包，基于 PyTorch），作为 baseline MoE 实现；代码改进开源在 https://github.com/hyhuang00/moe_inference。
  修改内容：
  1. 在 Fairseq MoE Transformer 的 Expert Parallelism 层中新增 Expert Buffering 模块：每个 GPU 上维护一个 expert cache（大小可配置），在 MoE forward 前检查当前 batch 需要的 experts 是否在 GPU cache 中。若缺失，通过 `torch.cuda.stream` 异步从 CPU 向 GPU 拷贝 expert 参数，与 token 的 all-to-all 传输重叠。
  2. 新增 Load Balancing 模块：在推理前运行 profiling pass（收集 expert activation 数据），然后运行 Greedy 或 Anti-Correlation 算法重新分配 experts 到不同 GPU，生成新的 expert placement 方案。
  3. Dynamic Gating 替换了 Fairseq 中的 static gating 函数实现——用 argsort + bin-count + indexing 替代 batch matmul dispatch mask（详见算法pipeline条目）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  MoE 推理全流程（Fairseq + Dynamic Gating + Expert Buffering + Load Balancing，单 node 8×V100）：

  ```
  [预部署阶段 - Load Balancing]
    │
    ├─ Step A: Profiling pass
    │    用少量 batch 执行推理，收集 expert activation 数据 A_mb
    │    (expert m 在 batch b 中处理的 token 比例)
    │
    ├─ Step B: Expert Placement Optimization
    │    ┌─ Greedy: sort experts by avg_load desc →
    │    │         循环分配 each expert to GPU with min load
    │    └─ Anti-Correlation (for MT-Decoder):
    │              在 estimated_load 中加 Pearson corr 惩罚项
    │              load[n] += 0.5 × Σ S_am (correlated experts)
    │    → 输出: expert_to_device 映射，启动时分配
    │
    └─ Step C: GPU Cache 初始化
        每个 GPU 分配固定大小的 expert cache (e.g., 10 experts/GPU)
        初始状态: cache 为空，全部 expert 参数在 CPU memory

  [推理阶段 - Per Batch Forward]
    │
    ├─ Input: token batch S=8 (LM) / S=48 (MT), seq tokens X ∈ R^{S×D}
    ├─ Multi-Head Attention (标准 Transformer, 非 MoE 层)
    │     QKV projection → attention → output X_attn
    │
    ├─ MoE Gating Layer (Dynamic Gating)
    │   │
    │   ├─ gate_logits = W_gate @ X_attn            // O(SD×E)
    │   ├─ assignments = top_k(gate_logits)          // top-2 or top-4
    │   ├─ sorted_idx = argsort(assignments.专家ID)   // O(S log S)
    │   ├─ sorted_X = X_attn[sorted_idx]             // O(SD) indexing
    │   ├─ sizes = bincount(sorted assignments)      // O(S)
    │   │
    │   └─ [All-to-All Round 1]: 通知各 GPU 即将接收的 token 数量
    │       各 GPU 传送 size 整数（~20µs average latency）
    │
    ├─ [Expert Buffering - Fetch & Execute]
    │   For each GPU (hosting subset of experts per Load Balance placement):
    │   │
    │   ├─ Step 1: Check Expert Cache
    │   │    for each expert e needed by this GPU's tokens:
    │   │      if e not in GPU cache:
    │   │        launch async CPU→GPU copy for e's parameters (cudaMemcpyAsync)
    │   │        // 与以下 all-to-all token 传输重叠
    │   │
    │   ├─ [All-to-All Round 2]: Transfer actual tokens
    │   │    tokens per device = split(sorted_X, sizes)
    │   │    → NCCL all-to-all → 各 GPU 收到 assigned tokens
    │   │
    │   ├─ Step 2: Expert Execution (sequential by expert ID)
    │   │    for expert e on this GPU:
    │   │      await expert e params ready (CPU→GPU copy 完成)
    │   │      tokens_e = received_tokens[expert_e_indices]
    │   │      W_up, W_gate_act, W_down = expert_e_parameters
    │   │      out_e = W_down @ (σ(W_gate_act @ tokens_e) ⊙ (W_up @ tokens_e))
    │   │      // 若 e 不在 cache: cache[e] = params (LIFO evict)
    │   │
    │   ├─ Step 3: Cache Eviction (LIFO policy)
    │   │    if cache full:
    │   │      evict most recently accessed inactive expert
    │   │      // 理由: MoE 按 ID 顺序执行 → LIFO 保留复用距离最短的
    │   │
    │   └─ [All-to-All Round 3]: Collect expert outputs back
    │        expert_outputs → NCCL all-to-all → 返回原始 GPU
    │
    ├─ Output Reordering
    │    restore original token order via inverse permutation
    │
    └─ → Next Transformer layer (attention → gating → experts)
  ```

  Expert Buffering 关键数据流：
  ```
  GPU Memory Layout (per GPU, e.g., 10/32 expert slots):
  ┌──────────────────────────────────────┐
  │  Expert Cache (GPU HBM)              │
  │  slot 0: expert_42  [W_up, W_gate, W_down]  │
  │  slot 1: expert_7   [W_up, W_gate, W_down]  │
  │  ...                                  │
  │  slot 9: empty                        │
  ├──────────────────────────────────────┤
  │  Non-Expert Params ( Attention W_QKV,│
  │    Gate Linear, Token Buffers, etc.)  │
  └──────────────────────────────────────┘
  
  CPU Memory (Host DRAM): 全部 128/512 experts 的完整参数
  
  Cache Miss Flow:
    expert_99 needed → not in cache →
      cudaMemcpyAsync(CPU→GPU, expert_99_params, PCIe stream)
      // 与 all-to-all token transfer 并发
      → LIFO evict slot (e.g., evict expert_7, 最近使用但当前 inactive)
      → cache[evicted_slot] = expert_99
  ```

  性能影响（论文数据）：
  - Expert Buffering 减少 static GPU memory 达 1.47×（~2.25GB on V100）。
  - 对 MT-Decoder，cache size=10 experts/GPU（80 across 8 GPUs）时吞吐仍优于 baseline；再小则 cache miss 主导延迟。
  - LIFO cache miss rate 接近理论最优 Belady's MIN。
  - Greedy Load Balancing 额外提升 throughput up to 1.19×（vs dynamic gating alone）；Anti-Correlation balancing 对相关 expert 场景提供 1.02× 增益。
  - Multi-node: Dynamic Gating + Expert Buffering + Load Balancing 总计提升 throughput 2.21×–4.30× (vs Fairseq baseline)。

  关键区别：标准 EP 在每个 expert layer 前后有 all-to-all barrier——所有 GPU 必须等待最慢的 expert 完成。AMoE 每个 GPU 独立决策执行哪个 layer，不等待其他 GPU。当某 expert tokens 不足时不执行（积累中），GPU 转去执行 token 充足的另一个 layer。
