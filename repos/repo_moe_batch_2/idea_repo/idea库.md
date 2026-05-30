## LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

- baseline方法是什么？
  - **固定 top-k MoE 推理 + Expert Pruning**：现有 MoE 模型在推理时所有层使用统一的 top-k（如 Mixtral 全部层 top-k=2，Qwen 全部层 top-k=4），无论每层的实际计算需求如何。Post-training 优化方法主要通过 expert pruning 减少模型参数：Inter-Expert Pruning (NAEE) 删除整层中不重要的 expert；Intra-Expert Pruning (MoE-I²) 缩减每个 expert 内部 FFN 的 intermediate 维度。然而，pruning 虽然减少显存占用，在 vLLM 等优化推理框架上的实际推理吞吐量提升有限甚至退化——原因是 token-to-expert 路由不变（仍需路由到固定 top-k 个 expert），剩余 expert 需处理更多 token，导致负载不均衡和 latency 增加。此外，pruning 方法依赖 calibration 数据集进行 expert 重要性评估，使得剪枝模型可能过拟合到 calibration 分布。
  - 全栈执行例子（Baseline Mixtral-8x7B, 4×H100, vLLM + FusedMoE）：
    - **推理算法层**：对所有 32 层，top-k=2 的固定路由。每个 token 经 router 选出 top-2 experts，Expert FFN (W_gate, W_up, W_down, dim=14336) 计算 → 加权求和。Inter-pruning (50%) 删除每层 4 个 expert 后，剩余 4 个 expert 需处理原 8 个 expert 的全部 token。
    - **系统框架层**：vLLM + FusedMoE。PagedAttention 管理 KV-cache。Tensor Parallelism 跨 4×H100。FusedMoE 将 expert 计算和路由融合以减少 kernel launch overhead。固定的 top-k 意味着固定的 all-reduce/broadcast 通信模式。
    - **编译框架层**：论文未明确说明（vLLM 使用 PyTorch eager mode + custom CUDA kernels）。
    - **kernel 调度层**：FusedMoE kernel 在 H100 Tensor Cores 上批量执行 expert GEMM。Token dispatch 按 expert 分组。Pruning 后 expert 负载不均衡：某些 expert 收到远超平均的 token 数 → 长尾 latency。
    - **硬件架构层**：4×H100 80GB，NVLink 互联，Tensor Cores。Expert 参数从 HBM 加载。Pruning 虽减少 HBM 占用但 bandwidth 节省有限（仍需加载所有未剪枝 expert）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LExI 方法**：核心洞察是"不同层对 expert 数量的敏感度不同"——有些层减少 active expert 几乎不影响输出（低敏感层），而有些层则需要更多 expert（高敏感层）。LExI 利用这一特性实现 layer-adaptive top-k 分配：
    1. **Data-Free Sensitivity Profiling**：仅用模型权重 + 随机 Gaussian 输入，通过 Monte Carlo 采样计算每层在不同 top-k 下的 Frobenius 范数输出偏差。解决了 pruning 方法对 calibration 数据的依赖问题。
    2. **Evolutionary Search**：以 sensitivity proxy 为引导，在总 active expert budget B 约束下搜索全局最优的逐层 top-k 分配。将"top-k 从{1,2,...,k_base}^L 的组合优化问题"转化为进化搜索问题，避免了梯度方法的巨大计算开销。
    3. **Static Per-Layer Top-k Assignment**：不同层使用不同数量的 active expert——低敏感层用更少的 expert 省计算，高敏感层保留更多 expert 保精度。
  - 对应解决 Baseline 缺陷：
    - **Pruning 不提升推理吞吐量（甚至退化）** → LExI 不删除任何 expert，而是通过减少低敏感层的 active expert 数量直接减少 FFN 计算量。每个 expert 仍保留完整参数，不存在负载不均衡问题（所有 expert 均可被路由）。
    - **固定 top-k 导致冗余计算** → LExI 的 layer-adaptive top-k 在低敏感层激活更少 expert，在高敏感层保持充分 expert 容量，实现"按需分配"的精细化计算。
    - **Pruning 依赖 calibration 数据** → LExI 的 sensitivity profiling 仅使用随机 Gaussian 输入和模型权重，完全 data-free。
    - **Pruning 不可逆/不可调节** → LExI 的 budget B 是可控参数：B 越小，计算越少（吞吐越高）；B 越大，越接近 baseline 精度。无需重新训练或重新 profiling 即可在不同 B 之间切换。
  - 全栈执行例子（LExI on Mixtral-8x7B, 4×H100, vLLM + FusedMoE）：
    - **推理算法层**：LExI 离线计算得到 32 层的 top-k 分配，如 [1, 2, 1, 2, 1, 1, 2, ..., 2]。总 budget B = Σ k_j = 50（vs baseline B = 32×2 = 64）。低敏感层用 k_j=1（仅激活 1 个 expert 而非 2 个）直接省去一个 expert 的 FFN 计算。
    - **系统框架层**：vLLM + FusedMoE 不变。仅修改 MoE 路由参数（set_topk），无需改变调度、内存管理或 kernel。推理时每层自动按各自的 k_j 激活对应数量的 expert。
    - **编译框架层**：论文未明确说明。与 baseline 相同。
    - **kernel 调度层**：FusedMoE kernel 不变，但减少了 total expert computation——每 token 每层减少 k_base - k_j 次 FFN forward。H100 Tensor Cores 计算负载降低。All-reduce/broadcast 通信量随 active expert 减少而降低。No load imbalance（所有 expert 保留完整权重）。
    - **硬件架构层**：同 baseline。减少的 computation 直接转化为更低的 latency 和更高的 throughput。

## Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

- baseline方法是什么？
  - **DeepSpeed/Tutel 的 All-to-All + Expert 重叠方案**：现有 MoE 训练优化（Tutel, FasterMoE）的 focus region 仅限于 all-to-all 通信和 expert 计算之间。通过沿 capacity 维度分区 all-to-all 和 experts 并组成 computation-communication pipeline，使 expert 计算与 all-to-all 通信重叠执行。但 all-to-all 通信时间通常远超 expert 计算时间（可达 3.36x），因此重叠仅能隐藏 expert 计算时间，all-to-all 通信本身仍是瓶颈。其他计算（self-attention、前后 Transformer layer 的 FFN、backward 的 dW 计算）不参与重叠，处于整体执行时间的 critical path 上。非 MoE 模型的 communication scheduling（如 P3, ByteScheduler）依赖 all-reduce 同步参数，不适用于 MoE 中 all-to-all 与其他计算之间有直接数据依赖的情形。
  - 全栈执行例子（Baseline Tutel，GPT2-S-MoE 在 8×A100 上前向传播）：
    - **训练算法层**：Top-k routing + expert capacity C 限制 + 超量 token drop。Switch gate 或 Batch Prioritized gate。
    - **系统框架层**：PyTorch + DeepSpeed/Tutel。Expert parallelism with all-to-all。Tutel 沿 capacity 维度将 all-to-all 分为 m 个 micro-batch，与对应的 expert 计算重叠。非 MoE 计算（self-attention, FFN before/after MoE layer）串行等待 all-to-all 完成。
    - **编译框架层**：PyTorch eager mode。Tutel 使用自定义 CUDA kernels 实现 MoE dispatch/gather。无编译器级别优化。
    - **kernel 调度层**：NCCL all-to-all (uniform-shaped C×E) + Tutel CUDA kernels for expert dispatch/combine + cuBLAS GEMM for expert FFN。Partitioned all-to-all 的每个 micro-batch 通信量与专家计算量按比例缩放。
    - **硬件架构层**：8× A100 80GB per node，NVLink intra-node + 100Gbps NIC inter-node。All-to-all 跨节点通信成为 40% 训练时间的瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Lancet 方法**：将 focus region 从仅 all-to-all + expert 扩展到整个训练图的两种新重叠机会：
    1. **前向传播 — Non-MoE 计算分区与 Pipelining**：将 self-attention 和前后 Transformer layer 的 non-MoE 计算沿 batch 维度分区，与 all-to-all + expert 组成更大的 computation-communication pipeline。解决"all-to-all 是瓶颈而 expert 计算不足够填满 overlap"的问题——引入更多 computation（self-attention、FFN）参与重叠，使 all-to-all 有足够的计算量可以隐藏。
    2. **反向传播 — Weight Gradient Computation (dW) 调度**：dW 计算依赖于 activation gradient (dX) 但不依赖于传输 activation gradient 的 all-to-all，因此无依赖路径。将 dW 指令重排序到 all-to-all 之后，使 dW 的 GEMM 计算与 all-to-all 通信并发执行。
    3. **解决 Partition 的数学等价性**：沿 batch 维度分区而非 capacity 维度分区（避免 token drop 差异），但 batch 分区导致 expert capacity 也要缩小从而可能引入额外 token drop。Lancet 实现特殊 gating operator 在各 partition 间传递 capacity 信息（第一个 partition 使用多少 C，后续 partition 动态调整 remaining C），保证 "所有 partition token-to-expert mapping 和 token dropping 与不分区的原版完全一致"。由此引入不规则形状 all-to-all（每个 partition 向每个 expert 发送的 token 数从 0 到 C 不等），通过双趟 NCCL Send/Recv 实现。
    4. **DP 搜索最优 Partition Range**：并非所有 non-MoE 算子都值得分区（GPU kernel launch overhead + SM under-utilization）。DP 算法在 O(N'GK) 复杂度下探索 partition range 和 partition count，Pipeline Scheduler 模拟每个候选方案的时间线并提供反馈。
  - 对应解决 Baseline 缺陷：
    - **All-to-all 是瓶颈（通信时间远长于 expert 计算）** → 引入 non-MoE 计算（self-attention, FFN）和 dW 计算参与重叠，增加可与 all-to-all 重叠的计算量，使 all-to-all 通信被更大范围的计算覆盖。
    - **Focus region 局限于 all-to-all+expert 导致 sub-optimal** → 扩展到 whole training graph，在 forward 中 pipelining non-MoE ops，在 backward 中 scheduling dW。
    - **Partition overhead（kernel launch + SM underutil）** → DP 自动搜索最优 partition range 和 count，避免 over-partitioning。
  - 全栈执行例子（Lancet，GPT2-S-MoE 在 8×A100 上前向传播，Switch gate，3 partitions）：
    - **训练算法层**：Switch gate routing。Special gating operator 在 partition 间传递容量信息，保证 token assignment 数学等价。Batch 维度分区（而非 capacity 维度）。
    - **系统框架层**：PyTorch 模型 → RAF compiler IR。Data parallel + Expert parallel。Lancet 优化通过 RAF pass manager 自动注入，无需修改 Python 代码。
    - **编译框架层**：RAF compiler IR 级别变换。Weight Gradient Computation Schedule Pass（依赖图 BFS+ 贪心分配）→ Operator Partition Pass（DP+CSP+PipelineScheduler）。Pass 输出是重排的 dW 指令和分区的 forward 算子 IR → RAF 编译为可执行代码。
    - **kernel 调度层**：Irregular All-to-All kernel（双趟 NCCL Send/Recv group）+ Tutel MoE dispatch kernel + cuBLAS GEMM for expert FFN。Pipeline scheduler 将 partitioned computation kernel 和 communication 按 stage 交错 launch：Partition 0 的 Non-MoE compute 先 launch，然后 Partition 0 的 All-to-All 与 Partition 1 的 Non-MoE compute 重叠 launch，依次类推。dW GEMM load 与 backward All-to-All 重叠。
    - **硬件架构层**：A100/V100 GPU。Irregular all-to-all 不传输 padding tokens → 总通信量低于 uniform all-to-all。Pipeline 中不同 partition 的 computation/communication 交错执行，提高 GPU SM 和 NIC 利用率。实现 non-overlapped communication time 减少 77%（V100 vs Tutel），吞吐量提升至 1.3x。

## LongCat-Flash Technical Report

- baseline方法是什么？
  - **固定数量 Expert 激活的 MoE + 传统 EP 通信范式（以 DeepSeek-V3 为代表）**：
    - 算法层：所有 token 激活固定数量的 FFN experts（top-K），无论 token 难易程度都消耗相同计算量。MLA 低秩分解路径无方差对齐，训练缩放时注意力分数不稳定。Fine-grained expert segmentation 导致初始化方差缩小，需额外调参。
    - 系统框架层：DeepSeek-V3 的 TBO (Two Batch Overlap) 策略需要两个不同的 batch 来构造 computation-communication overlap——一个 batch 的 attention 计算与另一个 batch 的 MoE 通信重叠。这导致：(a) 需要维护双 batch pipeline state，(b) communication overlap 窗口受限于单个 expert 的计算时间，(c) 单个用户请求无法享受 overlap 加速。
    - 编译框架层：论文未明确说明。
    - kernel 调度层：默认 FlashAttention gradient kernel 使用 atomicAdd 归约，非确定性执行顺序导致不同 run 的 loss 无法精确复现。Grouped GEMM compute density 低，无法有效与 all-to-all 通信重叠。
    - 硬件架构层：H800 GPU + NVLink + RDMA，all-to-all 通信成为 MoE 推理的主要瓶颈（non-overlapping communication 占 25.3%）。TP 与 EP 的通信资源竞争：intra-node NVLink（TP 的 all-gather/reduce-scatter）与 inter-node RDMA（EP 的 all-to-all）无法并发利用。
  - 全栈执行例子（Baseline DeepSeek-V3 TBO 单 token 推理）：
    - **训练/推理算法层**：固定 top-K=8 routing → 每 token 激活 8 个 FFN experts。MLA 无 scale-correction → 注意力分数方差随 d_q/d_kv/d_model 比例漂移。每个 expert 输出无 variance compensation → 细化 expert 后需重新调参。MTP 使用 MoE layer head。
    - **系统框架层**：TBO 需要两个 batch：Batch A 的 Attention + Batch B 的 All-to-All Dispatch → Batch B 的 MoE GEMM + Batch A 的 All-to-All Combine。需要双 batch 的 context 和 KV cache 管理。Attention/FFN 计算与 expert 计算串行。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：NCCL all-to-all（dispatch/combine）→ 不可重叠部分占 25.3% 时间。默认 Grouped GEMM 无 double-buffer/diagonal tiling 优化 → compute density 低。默认 ScatterAdd 串行 → 50x 减速。
    - **硬件架构层**：H800 GPU × 128。All-to-All 跨节点 RDMA（dispatch 275us + combine 551us）与 Attention computation（471us）几乎无法重叠 → TPOT ≈ 30ms。28 层的每层时间累积 → 总延迟高。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LongCat-Flash 通过四项核心设计解决 Baseline 各层缺陷**：
    1. **Zero-Computation Experts + PID Budget Control**：对应解决"所有 token 消耗相同计算量"的缺陷。引入 Z 个 identity experts，Router 自适应选择性跳过 FFN 计算。PID controller + expert bias 确保平均激活计算量收敛到目标值（K_e），同时允许 per-token 变异（标准差~3）。简单 token（标点、虚词）利用更多 zero-comp experts 节省计算，困难 token（语义关键词）激活更多 FFN experts。
    2. **ScMoE (Shortcut-Connected MoE) + SBO Scheduling**：对应解决 TBO"需要双 batch"和"overlap 窗口小"的缺陷。跨层 shortcut 将前一层 Dense FFN 连接到当前 MoE block 之前，使 Dense FFN 计算可与 MoE dispatch/combine 通信重叠。SBO 四阶段 pipeline 在单 batch 内实现 module-level overlap：Dense FFN 和 Attention 的 computation time (~264us) 覆盖 all-to-all dispatch (~236us) 和 combine (~472us) 的大部分通信时间。Token 维度 chunking（分两个 chunk）实现 chunk 间互相重叠。
    3. **Variance Alignment (MLA Scale-Correction + Expert Variance Compensation)**：对应解决"MLA 缩放时注意力不稳定"和"expert segmentation 方差缩小"的缺陷。α_q/α_kv 将低秩路径方差对齐到 d_model 参考尺度 → 任意 d_q/d_kv 比例下注意力分数稳定。γ=m 补偿 expert 分割导致的方差衰减 → 不需要重新调参。
    4. **MTP with Dense Head + TVD Fusion**：对应解决 MTP 推理开销问题。单一 dense layer（1.41% params, 92.1% accept rate）vs MoE layer（4.17% params, 92.9% accept rate）——以略低接受率换取远低 draft cost。TVD CUDA graph fusion 消除分离调度的 kernel launch overhead。
  - 全栈执行例子（LongCat-Flash SBO 单 token 推理）：
    - **训练/推理算法层**：Zero-comp experts 使简单 token 激活少 FFN experts → 节省计算。α_q = √(d_model/d_q), α_kv = √(d_model/d_kv) 修正 MLA 方差 → 注意力分数稳定。γ=m 补偿 fine-grained expert 方差 → 训练稳定。单 dense MTP head 提效。
    - **系统框架层**：SBO 四阶段 pipeline 在单 batch 内重叠：Stage 2 的 Dense FFN 计算与 all-to-all dispatch 并行；Stage 4 的 Attention Core + Dense FFN 与 all-to-all combine 并行。ScMoE shortcut 是关键 enabler——没有 shortcut 时 Dense FFN 在 MoE 之后无法用于 overlap。Multi-step scheduler 预启动 4 步 forward pass 消除 CPU bottleneck。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：Deterministic FAG（1.6x vs naive deterministic, 0.95x vs non-deterministic）→ training loss 可 bitwise 复现。Grouped GEMM 三重优化（double-buffer + diagonal tiling + HBM bandwidth control）→ 5-45% 加速。SwapAB MoE GEMM → 利用 N 维度 8 元素对齐消除 padding。NVLink Sharp PTX kernels → 仅需 4 thread blocks，全 message size 范围优于 NCCL/MSCCL++。
    - **硬件架构层**：H800 GPU × 128。SBO overlap 使 non-overlapping communication 从 25.3% 降至 8.4%。ScMoE 下 intra-node NVLink (Dense FFN TP) 与 inter-node RDMA (MoE all-to-all) 并发 → 网络总利用率最大化。TPOT 理论值 16ms（vs DeepSeek-V3 TBO 30ms），实测 26ms。成本 \$0.70/1M output tokens，\$0.09/1M 理论极限。
    1. **内存膨胀**：共享参数每 GPU 复制一份，多卡时大量内存浪费，限制了 batch size 和 context length 的扩展。例如 Qwen2-57B-A14B（64 experts, 4 machines），每台 machine 额外消耗大量内存仅用于冗余的 attention/norm 参数。
    2. **GroupedGEMM 延迟高**：随着 batch size 增大，被激活的 expert 数量几乎线性增长（直到全部 expert 被激活），GroupedGEMM 作为 memory-bound 操作成为延迟瓶颈，且 MoE 的动态控制流使 CUDA Graph / Torch Compile 优化失效。
  - 全栈执行例子（Baseline Full Model，Qwen2-57B-A14B 在 4× A6000 上单请求 decode）：
    - **模型推理/训练算法层**：Router 选 top-Ek=6 experts → GroupedGEMM 并行计算 6 个 expert 的 FC1+FC2 → 加权 combine → Shared Expert → 输出 logits。每个 decode step 固定激活 6 experts，计算量大。
    - **系统框架层**：纯 EP 并行，PyTorch 推理。每次 decode 需要 All-to-All dispatch+combine。论文未明确说明 Serving 框架。
    - **编译框架层**：论文未明确说明。PyTorch eager mode，无法使用 CUDA Graph 加速（因 MoE 动态路由）。
    - **kernel 调度层**：Cutlass GroupedGEMM kernel 执行 expert 计算。Batch size 增大时，激活 expert 数线性增长 → GroupedGEMM memory footprint 线性增长 → memory-bound 延迟上升。论文未明确说明其他 kernel 优化。
    - **硬件架构层**：4× NVIDIA A6000 GPUs，节点内通信。Attention 参数每卡完整复制，浪费显存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **IFMoE 方法**：从并行策略和计算调度两个层面同时优化：
    1. **EP+TP Hybrid Parallelism**：Shared 参数（Attention、Norm、Shared Expert）用 Tensor Parallelism 切分到多卡，消除每卡全量复制。Expert 参数保持 EP。通信从 All-to-All 改为 double All-Gather（因为节点内通信带宽充足）。内存节省直接转化为更大的 KV-cache 容量，支撑更大 batch size 和更长 context。
    2. **Self-Draft Speculative Decoding with KV-cache Revision**：不需要额外 draft model。用 MoE 自身激活 Dk=2 experts（< Ek=6）作为 draft，每 α=10 步 draft 后用全量 Ek=6 experts 重算 KV-cache。因为 fine-grained MoE 的 expert 粒度细、单个 expert 计算量小，减少激活 expert 数直接降低 GroupedGEMM 延迟。
  - 对应解决 Baseline 缺陷：
    - Baseline 共享参数内存浪费 → EP+TP Hybrid 节省大量内存（Deepseek-Lite: 4.6GB, Qwen2: 10GB, Deepseek-v2: 23GB per machine）
    - Baseline GroupedGEMM 延迟高 → Draft 阶段仅激活 Dk=2 experts（减少 67%），延迟显著降低；KV-cache revision 每 α=10 步才执行一次 Ek=6 全量计算，amortized overhead 小
    - Baseline 无法使用编译优化 → 减少 decode step 的 expert 激活数简化控制流，降低对 CUDA Graph 的依赖
  - 全栈执行例子（IFMoE，Qwen2-57B-A14B 在 4× A6000 上单请求 decode）：
    - **模型推理/训练算法层**：Draft decode（10 steps）：每 step 仅 Router 选 Dk=2 experts → GroupedGEMM 计算 2 expert → combine → 输出 token。KV-cache revision（每 10 steps）：全量 Ek=6 experts encode → 覆盖 KV-cache。Draft 阶段 expert 激活从 6 降至 2，computation 减少 ~67%。
    - **系统框架层**：EP+TP 混合并行。Shared 参数 TP 切分到 4 卡，每卡仅存储 1/4 的 Attention/Norm 参数。Expert 参数 EP 分布。Double All-Gather 替代 All-to-All。内存节省 ~10GB（Qwen2）→ 最大 batch size 达 256。
    - **编译框架层**：论文未明确说明。PyTorch 推理，draft 阶段的简化控制流理论上更易被 Torch Compile 优化，但论文未实现此特性（列入 Future Work）。
    - **kernel 调度层**：Cutlass GroupedGEMM。Draft 阶段每次仅处理 Dk=2 个 expert 的 GEMM，memory footprint 大幅降低（从 6 expert → 2 expert），memory-bound 延迟显著减少。Revision 阶段仍用 Ek=6，但频率仅 1/α（~10%）。
    - **硬件架构层**：4× NVIDIA A6000 GPUs，节点内通信。EP+TP hybrid 使每卡可用显存增加约 10GB（Qwen2 case），可用于 KV-cache 扩展。

## I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts

- baseline方法是什么？
  - **Vanilla Multimodal Fusion**（如 Early Fusion, Late Fusion, MulT, LRMF）：使用统一的融合参数处理所有模态交互。流程为：各模态编码器 E_i 分别编码输入 → 融合方法 F 将所有隐嵌入融合为单一向量 x = F(e1, e2, ..., en) → 预测头 H(x) = ŷ。核心缺陷是使用相同参数建模所有交互类型（唯一性/协同/冗余），无法区分"图像独有的视觉线索"、"文本独有的语义信息"、"两者协同产生的新信息"、"两者共享的冗余信息"。在 Figure 1 的 IMDB 电影分类例子中，Horror 依赖图像唯一性、Romance 依赖语言唯一性、Fantasy 依赖冗余信息、Drama 依赖协同信息——vanilla fusion 对这些不同交互一视同仁。
  - **SwitchGate & MoE++**：在 MulT 中将 MLP 层替换为稀疏 MoE 层，但 MoE routing 仅做 conditional computation（负载均衡），不鼓励专家按交互类型分化。本质上仍是 implicit interaction modeling。
  - **MMoE (Yu et al. 2024)**：唯一显式建模交互类型的 MoE 方法，但将交互建模作为预处理步骤（非端到端），限制灵活性和可解释性。
  - 全栈执行例子（Baseline MulT 在 IMDB 上做电影分类）：
    - **模型推理/训练算法层**：图像用 VGG16 提取特征，文本用 Google Word2vec 提取特征 → VGG11 编码器处理 → MulT 多模态 Transformer（cross-modal attention）融合 → 线性分类头输出 23 类 logits → CrossEntropy 损失。所有模态交互通过同一套 attention 参数处理。
    - **系统框架层**：PyTorch + torchvision，标准训练脚本，单卡 A100。论文未明确说明框架级定制。
    - **编译框架层**：论文未明确说明。PyTorch eager mode，cuDNN/cuBLAS 后端。
    - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM（attention 矩阵乘法）+ PyTorch autograd。
    - **硬件架构层**：单卡 NVIDIA A100 GPU。论文未明确说明 GPU 架构级优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **I²MoE 方法**：基于 PID 框架设计端到端 MoE，包含三种核心机制：
    1. **Interaction Experts 分化**：用 n+2 个独立融合模型（含各自参数和预测头）替代单一融合模块。每个专家通过不同的弱监督交互损失（perturbation-based）被迫专精——唯一性专家通过 Triplet Margin Loss 学习"去除了某模态就无法正确预测"的信息；协同专家通过最小化与遮蔽输入的 Cosine Similarity 学习"必须联合两模态才有的"信息；冗余专家通过最大化与遮蔽输入的 Cosine Similarity 学习"任一模态都能做到的"信息。
    2. **Adaptive Reweighting**：MLP 重加权模型根据输入样本的模态特征动态分配 wi，替代均匀融合。这使得模型对不同样本可自适应选择交互策略（某样本偏重图像唯一性、另一样本偏重文本唯一性或协同）。
    3. **端到端可解释性**：重加权模型输出 w_i 天然提供样本级局部解释（每个交互的贡献度）+ 测试集 w_i 统计提供数据集级全局解释（整体交互趋势）。
  - 对应解决 Baseline 缺陷：
    - Baseline 用同一参数建模所有交互 → I²MoE 用 4 个独立专家（各自参数+各自损失）显式分化。
    - Baseline 无法区分交互类型 → I²MoE 通过 perturbation-based 弱监督为每种交互提供明确训练信号（TripletLoss/CosSim/MSE）。
    - Baseline 缺乏可解释性 → I²MoE 的 w_i 天然提供 local/global 两层可解释性，人类评估 70.4% 正面。
  - 全栈执行例子（I²MoE-MulT 在 IMDB 上做电影分类）：
    - **模型推理/训练算法层**：图像用 VGG16 提取特征，文本用 Google Word2vec 提取特征 → VGG11 编码器处理 → 4 个 MulT 交互专家（F_uni_img, F_uni_lang, F_syn, F_red）各做 cross-modal attention 融合 + 线性预测头输出各 23 类 logits → 训练时每个专家额外做 2 次 masked 前向（遮蔽图像/遮蔽文本，用随机向量 r 替换嵌入）→ 计算交互损失 → MLP 重加权模型输出 w_i = softmax(MLP(e_img, e_lang) / temperature) → ŷ = Σ w_i · ŷ_i。推断时仅 1 次完整模态前向 + 1 次重加权。任务损失 + λ_int · 交互损失联合优化。
    - **系统框架层**：PyTorch + torchvision。训练脚本在 https://github.com/Raina-Xin/I2MoE/tree/main/scripts/train_scripts。论文未明确说明框架级定制。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：论文未明确说明。标准 PyTorch 算子，每次训练需 (n_experts) × (n_modalities+1) 次 forward pass，额外计算开销约为 MulT 的 (n_modalities+2) 倍。训练时间开销：IMDB 上 MulT 3.62s/epoch → I²MoE-MulT 44.20s/epoch；推断开销：MulT 0.53s → I²MoE-MulT 3.23s。
    - **硬件架构层**：单卡 NVIDIA A100 GPU。论文未明确说明 GPU 架构级优化。

## FlashMoE: Fast Distributed MoE in a Single Kernel

- baseline方法是什么？
  - **现有分布式 MoE 系统**（DeepSpeed-MoE, Megatron-LM, FasterMoE, COMET, DeepEP）使用 **CPU 管理的同步通信 + 多 kernel 分片执行** 的范式：
    - **同步 AlltoAll Collective**：所有 GPU 通过 NCCL AlltoAll 或 AllGather 同步交换 token。每个 GPU 必须等待所有 peer GPU 完成通信才能继续，straggler GPU 拖慢全组（commercial VM 上 P95 idle delay 达 11.4X）。AlltoAll 通信可占 MoE layer 总运行时间的 68%。
    - **多 Kernel Launch 开销**：MoE layer 的 Gate → Dispatch → Expert → Combine 各阶段被分割为 33-550 个独立 GPU kernel（Table 1），由 CPU 串行调度。每次 kernel launch 产生：非确定性 kernel start time（加剧 straggler）、不必要的同步点（GPU 等待 CPU/peer）、kernel 边界的重复 global memory round trip。
    - **Payload 低效**：为满足 collective 通信的对称性约束，不对称路由（某 expert 收到少于 capacity 的 tokens）时，buffer 中零填充 token 仍被传输，浪费通信带宽和后续计算资源。
  - 全栈执行例子（Baseline DeepSpeed-MoE / Megatron-LM 执行一次 MoE forward pass on 8 A100 GPUs）：
    - **模型推理/训练算法层**：Standard MoE layer: Gate(top-2 routing) → Token Dispatch → Expert FFN(2×GEMM with GELU/SiLU) → Expert Combine(weighted sum)。使用 PyTorch eager mode，autograd 追踪。
    - **系统框架层**：PyTorch + DeepSpeed-MoE/Megatron-LM。CPU 调度 85-550 个 CUDA kernel launch（每 MoE layer）。NCCL AlltoAll collective 用于 token dispatch/combine。
    - **编译框架层**：论文未明确说明。PyTorch eager mode → CUDA compiler → cuBLAS/NCCL backend。
    - **kernel 调度层**：CPU 串行发射 kernel（Gate kernel → Router kernel → Dispatch AlltoAll kernel → Expert GEMM kernels × E × 2 → Combine AlltoAll kernel → Scale kernel）。kernel 间存在 CPU-GPU 同步间隙，GPU SM utilization 低至 14%（图 4a）。AlltoAll kernel 内所有 GPU 同步等待，straggler 延迟传播。
    - **硬件架构层**：8× NVIDIA H100 80GB GPU，NVLink 互联，单节点。GPU 间通过 NVLink + NCCL collective 通信。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FlashMoE 方法**：将整个分布式 MoE operator（Gate、Dispatch、Expert FFN、Combine、跨 GPU 通信）融合为**单一持久 GPU kernel（megakernel）**，基于 actor model 实现 warp 级并发和 GPU-initiated one-sided 通信。
  - 四大设计对应解决 baseline 缺陷：
    1. **单一持久 Kernel 消除 Launch 开销**：
       - Baseline 缺陷：33-550 次 kernel launch，每次产生 CPU-GPU 同步间隙、非确定性 start time、global memory round trip。
       - 方法：仅 1 次 kernel launch。Kernel 持续活跃到 MoE operator 完成。消除所有 kernel launch 间隙和 CPU 干预。
    2. **Actor-based In-Kernel Scheduling 消除 SM 空闲**：
       - Baseline 缺陷：kernel 间等待导致 GPU SM idle（DeepEP 14% utilization）。
       - 方法：OS block 内 Scheduler warp 持续 sweep doorbells，基于 task readiness 动态分配 tile 级 task 给 Processor block。Scheduler 是 multithreaded 和 work-conserving，确保无 SM 空闲。FlashMoE 达 93.17% SM utilization。
    3. **Device-Initiated One-Sided (R)DMA 替代同步 AlltoAll**：
       - Baseline 缺陷：同步 AlltoAll straggler 效应（P95 11.4X idle delay）。
       - 方法：NVSHMEM PGAS 模型建立全局地址空间，GPU 直接发起 one-sided put/get。Symmetric Tensor Layout L ∈ R^{P×R×B×E×C×H} 通过 temporal buffering (2 rounds × 2 buffers = 4× overprovision) 保证所有 one-sided 写入无冲突（Theorem 3.1 证明）。写入完成后 signal 通知远程 Subscriber，实现完全非阻塞通信。
    4. **In-Place Padding 实现 Payload Efficiency**：
       - Baseline 缺陷：零填充 token 在网络上传输和 GPU 上无效计算。
       - 方法：在本地 symmetric tensor buffer 中做 in-place padding（对齐 tile block size bM=128），仅传输有效 tokens 到有激活 expert 的 GPU。
  - 全栈执行例子（FlashMoE 执行一次 MoE forward pass on 8 H100 GPUs）：
    - **模型推理/训练算法层**：Standard MoE（top-2 routing, capacity factor=1.0, FFN with GELU/SiLU, expert-combine），但计算图被完全 fuse 到单一 kernel。
    - **系统框架层**：PyTorch + FlashMoE custom C++/CUDA extension (6820 LOC)。通过 `torch.autograd.Function` 注册。CPU 仅 launch 1 次 kernel，kernel 内 GPU 自主完成所有任务调度和执行。
    - **编译框架层**：CUDA 12.8 compiler。CUTLASS 提供 in-kernel BLAS (GEMM)。NVSHMEM v3.2.5 提供 GPU-initiated (R)DMA。
    - **kernel 调度层**：
      ```
      GPU i 上单 kernel 内:
      Time ──────────────────────────────────────────────────▶
      
      [FusedGate: 所有 block 计算 T_φ, G_φ]
      └─▶ [OS Block (4 warps)]
      │    ├─ Scheduler(warp0): sweep doorbells → 分发 task
      │    └─ Subscriber(warps1-3): decode remote packets
      │
      ├─▶ [Processor Blocks × (N-1)]
      │    ├─ GEMM0(tile): fused GEMM + epilogue + async staging
      │    ├─ Notify Scheduler
      │    ├─ GEMM1(tile): fused GEMM + NVSHMEM put → remote GPU
      │    └─ Combine(tile): weighted expert output aggregation
      │
      └─▶ All done: Interrupt subscriber + processors
      ```
    - **硬件架构层**：8× NVIDIA H100 80GB GPU，NVLink 互联。NVSHMEM one-sided (R)DMA 通过 NVLink 直接访问 remote GPU memory (UVA)。

  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | FlashMoE 设计 | 效果 |
    |-------------|-------------|------|
    | 33-550 kernel launches | 单持久 kernel (1 launch) | 消除 launch 间隙，93.17% SM util |
    | CPU 控制 kernel launch 时序 | GPU actor-based in-kernel scheduling | 确定性的 GPU-native 调度 |
    | 同步 AlltoAll (straggler P95 11.4×) | NVSHMEM one-sided (R)DMA | 非阻塞点对点传输 |
    | 零填充 token 网络浪费 | In-place padding + payload-efficient | 仅传有效 token |
    | Global memory round trips at boundaries | Kernel 内 shared/global memory actor 通信 | 减少 HBM 访问 |

## Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

- baseline方法是什么？
  - **Expert Parallelism (EP)**：MoE layer 的 experts 均匀分布到多个 device，每个 device 持有若干完整 expert 的参数和 optimizer states。Token 通过 All-to-All 通信 dispatch 到持有对应 expert 的 device，计算完成后 All-to-All 收集回原 device。但由于 MoE gate 训练的随机性，expert loads 频繁波动和不平衡（图 3），导致最重载 device 成为通信和计算的 straggler，拖慢整个 MoE layer 的执行时间。评估显示 EP 在最坏 load imbalance 下性能下降达 5.18×。
  - **Expert Rearrangement Systems（FasterMoE, SmartMoE, FlexMoE）**：在 EP 基础上，通过动态调整 expert placement（重排/复制 expert）来减轻 straggler。但面临两个核心挑战：(C1) **Memory challenge**：更 balance 的 placement 需要更多内存来容纳 replica experts 及其 optimizer states，预留内存不足会限制 placement 优化空间（FlexMoE 实验中 4× 内存仅换 2.65× speedup）；(C2) **Timeliness challenge**：rearrangement 频率的 trade-off——频率高则 placement 更 timely 但通信开销大，频率低则 placement 过时；某一场景的最优频率无法泛化到其他场景（SmartMoE 实验中每 10 steps 相比每 25 steps，non-rearrangement iteration 快 2.9% 但整体慢 10.2%）。
  - 全栈执行例子（Baseline FlexMoE 训练 GPT-MoE-S on Cluster B, 32 A100 GPUs）：
    - **模型推理/训练算法层**：标准 MoE training loop。MoE gate 做 Top-2 token-to-expert assignment → expert FFN 计算（W_gate, W_up, W_down GEMMs）→ gate loss backward + expert backward。FlexMoE 按 token-to-expert 分配统计，启发式决策 expert replica 创建/删除 → AllReduce 同步 replica gradients。
    - **系统框架层**：PyTorch + Megatron-LM 训练框架。Expert parallelism 通过 All-to-All collective 实现 token dispatching。FlexMoE 的 rearrangement manager 在 iteration 间迁移 expert 参数+优化器状态（参数 6× 大小的量级，Adam optimizer mixed precision 下 optimizer states 至少 6× parameters），通过 P2P 通信进行 expert relocation/replication。
    - **编译框架层**：论文未明确说明。PyTorch eager mode，NCCL collective communication backend，cuBLAS GEMM。
    - **kernel 调度层**：NCCL All-to-All 通信 kernel（token dispatching）+ cuBLAS GEMM kernel（expert FFN 计算）。FlexMoE 的 rearrangement 引入额外 NCCL P2P Send/Recv（expert 参数+优化器状态传输，在 critical path 上）。AllReduce（gradient sync of replicated experts）在 backward 结束时执行。通信量：对于 placement P'，每个 DP group D_i 做 AllReduce for expert e_i，总通信量 O(2λS)。
    - **硬件架构层**：A100-40G GPU × 32（4 nodes × 8 GPUs），NVSwitch 600 GB/s intra-node，400 Gbps NIC inter-node。V100-32G × 32（4 nodes × 8 GPUs），NVLink 300 GB/s intra-node，100 Gbps inter-node。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Hecate 方法**：提出 Fully Sharded Sparse Data Parallelism (FSSDP)，从全新角度解决 MoE 训练的 straggler 问题。核心思想受到 FSDP 的启发：**将 MoE layer 的参数和 optimizer states 完全分片到所有 device，每次 iteration 从 shards 中用稀疏通信原语 (SparseAllGather + SparseReduceScatter) 从零构建一个临时的 expert placement，而不需要在 iteration 间迁移 expert 状态**。
  - 三大设计对应解决 baseline 缺陷：
    1. **FSSDP 消除 rearrangement memory overhead（解决 C1）**：
       - Baseline 缺陷：rearrangement 系统需要预留内存来接收迁移的 expert 参数+优化器状态（6× 参数量），越 balance 的 placement 越消耗内存。
       - Hecate 方法：FSSDP 全局只保留一份 optimizer states（不再每个 replica 一份），sharding phase 将其均匀分布在所有 device 上。Materialization phase 仅物化 expert 参数（非 optimizer states），用完即释放。Heterogeneous sharding 跨层统一调度，进一步优化 underloaded expert 的 placement 减少 All-to-All congestion，同时保证内存均衡。Re-materialization 将物化参数的额外内存开销降低 90.2%，总内存仅比 EP 增加 11.6%。
    2. **Sparse collectives 消除 rearrangement timeliness trade-off（解决 C2）**：
       - Baseline 缺陷：rearrangement 在 critical path 上，频率越高 placement 越 timely 但通信开销越大，最优频率不可泛化。
       - Hecate 方法：FSSDP 将 "rearrangement" 的概念从 "在 iteration 间迁移 expert 状态" 变为 "在 iteration 内从 shards 物化临时 placement"。SparseAllGather 和 SparseReduceScatter 的通信量与同 placement 下 rearrangement 的 AllReduce 通信量等价（O(2λS)），但消除了额外的迁移通信。两个稀疏 collective 的通信与 Attention computation 重叠调度（forward 重叠 spAG，backward 重叠 spRS + spAG），使 sparse materialization 脱离 critical path。每 iteration 都能工作在当前最优 placement 下，不存在 timeliness trade-off。
    3. **拓扑感知的 materialization 和 dispatching（超越 baseline 的优化）**：
       - Hecate 的 sparse materialization (Algorithm 1) 在搜索 placement 时考虑 interconnect topology：overlap degree t 的计算使用 inter-node bandwidth（异构网络）或 uniform bandwidth（同构网络），优先 intra-node 通信。Token dispatching 同样优先 intra-node，减少 inter-node All-to-All congestion。这些拓扑感知设计使 Hecate 的 All-to-All 通信时间比 EP 减少 12.3×。
  - 全栈执行例子（Hecate 训练 GPT-MoE-S on Cluster B, 32 A100 GPUs）：
    - **模型推理/训练算法层**：FSSDP 替代 EP。Sharding phase：Heterogeneous sharding (Algorithm 2) 将 64 experts 跨 32 devices 分片，underloaded experts 优先按 node/device 负载均衡放置，overloaded experts 填充剩余 slots。Materialization phase (Algorithm 1)：基于滑动窗口 (w=5) 估计下轮 expert load → 在 overlap degree t 和 memory capacity m 约束下搜索 placement P' → Calibration stage（gate 输出后）用实际 token assignment 决定是否追加物化。Token dispatching 优先 intra-node。Backward 梯度通过 spRS reduce 到 MoE shard 所在 device → optimizer step 更新本地 shard。
    - **系统框架层**：PyTorch + Megatron-LM 框架。Hecate 的 Executor 驱动 FSSDP workflow（sharding → materialization → dispatching → compute → gradient reduce → optimizer）。Communicator 管理 NCCL sparse collectives 和 All-to-All 通信队列。Scheduler 生成 placement plan。Dispatcher 做拓扑感知 token 路由。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：SparseAllGather = NCCL group calls (一组 Broadcast)，SparseReduceScatter = NCCL group calls (一组 Reduce)。Forward 中 spAG 与 Attention forward 重叠（约束：spAG 延迟 ≤ Attention fwd 延迟）。Backward 中 spRS (layer l) + spAG (layer l+1 re-materialize) 与 Attention backward 重叠（backward 耗时 ~2× forward，容纳两个 collective）。若启用 Hecate-RM：expert 参数 forward 后立即释放，backward 时重新 spAG 物化，增加 3.6× sparse collective 通信开销但仍优于 baseline 1.4×。
    - **硬件架构层**：A100-40G GPU × 32。Sparse collectives 利用 NVSwitch 600 GB/s intra-node 做高效 Broadcast/Reduce。Topology-aware scheduling 优先 intra-node 通信路径，减少 400 Gbps NIC inter-node 链路的拥塞。Hecate 在 Cluster A (V100, 100 Gbps inter-node) 上加速比更高（geo-mean 2.05× vs Cluster B 的 1.26× vs EP），因为低带宽环境的 All-to-All straggler 效应更强，Hecate 的拓扑感知优化收益更大。

- baseline方法是什么？
  - **Expert-Offloading 技术**：将 non-expert 权重 + 部分 "hot expert" 缓存于 GPU memory（expert cache），其余 expert offload 到 CPU memory 或 SSD，按需加载。当 cache miss 时，通过 PCIe/SSD 加载缺失 expert 并 evict 现有 expert。
  - Baseline 系统包括：
    1. **EdgeMoE**：对不同 expert 使用静态量化级别（基于特定数据集 profiling 确定 bit-width），缺乏跨环境灵活性。
    2. **AdapMoE**：激进跳过某些 expert 以减少加载开销，导致显著精度下降（特别是 small top-k 时，如 Mixtral-8x7B 的 k=2）。
    3. **MoE-Infinity**：按 expert activation ratio 优先级做 prefetching，但 prefetching 收益有限（expert 加载延迟 >> GPU 计算延迟）。
    4. **MoE-Offloading**：用当前层 gate input 预测下一层 expert（LRU 缓存策略），但预测带来的 overlap 收益同样受限于加载/计算比。
    5. 通用缓存策略（LFU、LRU）无法利用 mixed precision expert cache 中不同精度加载代价的差异。
  - 全栈执行例子（Baseline MoE-Infinity on RTX 4090，Mixtral-8x7B FP16）：
    - **模型推理算法层**：标准 MoE layer，router 计算 top-K=2 experts → 若 cache miss，从 CPU memory 加载完整 FP16 expert 权重（~10.5MB/expert for 7B hidden × 4096 FFN × 3 矩阵），传输时间 ~0.33ms (PCIe 4.0 32GB/s)。Expert FFN 计算：W_g/W_p/W_o GEMM，GPU 计算 ~3ms/layer。加载占 85.5% 总时间。
    - **系统框架层**：MoE-Infinity 基于 PyTorch + CUDA，expert 权重存储在 CPU memory (mmap)，LRU/LFU cache 管理 GPU expert cache。Prefetching 基于 expert activation 历史统计。
    - **编译框架层**：论文未明确说明。PyTorch eager mode，cuBLAS GEMM kernel。
    - **kernel 调度层**：标准 cuBLAS FP16 GEMM，expert loading 是单一大块连续内存拷贝 (cudaMemcpy)。所有 cache-miss expert 按相同 FP16 精度加载。
    - **硬件架构层**：RTX 4090 24GB + CPU 256GB，PCIe 4.0 ×16 (32GB/s)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HOBBIT 方法**：混合精度 Expert Offloading 系统，通过三个层次的创新利用 MoE 计算的自然层次结构，核心洞察是 **"动态将不太关键的 cache-miss expert 替换为低精度版本可以显著减少 expert 加载延迟同时保持模型精度"**。
  - 三大设计对应解决 baseline 缺陷：
    1. **Token-level Dynamic Expert Loading（解决静态/激进量化/跳过问题）**：
       - Baseline 缺陷：EdgeMoE 的静态量化依赖特定数据集 profiling，不灵活；AdapMoE 的激进跳过导致精度显著下降。
       - HOBBIT 方法：用 ||G(x)|| 作为 expert 重要性的动态代理（与 ||G(x)E(x)|| Pearson r=0.99），计算 unimportance degree score，双阈值灵活决策精度。避免了静态 profiling，运行时自适应输入；用低精度替换替代跳过，精度下降 <1%。
    2. **Layer-level Adaptive Expert Prefetching（解决预取收益低问题）**：
       - Baseline 缺陷：prefetching 因 expert 加载时间 >> GPU 计算时间而收益有限，错误预测惩罚严重（无法中断 cudaMemcpy）。
       - HOBBIT 方法：利用层间 gate input 高余弦相似度（相邻层 top-1 准确率平均 96%），Stacking Computer 一次性批量计算所有后续层 gating。关键创新：用低精度预取替代高精度预取，即使预测错误，低精度 expert 的错误加载惩罚仅为高精度的 1/4，使预取在任何精度下都产生正向收益。
    3. **Sequence-level Multidimensional Expert Caching（解决低效缓存管理问题）**：
       - Baseline 缺陷：LRU/LFU 等通用策略不考虑 mixed precision 特性（高精度 miss 代价 4× 低精度 miss）。
       - HOBBIT 方法：提出 LHU (Least High Precision Frequently Used) 策略追踪高精度使用频次 H_t，与 LRU + LFU + FLD 四策略加权组合。高/低精度 cache 分离管理，按加权优先级公式 evict。序列级 LFU（同一 sequence 内统计）相比模型级 LFU 提升 4.5% hit ratio。
  - 全栈执行例子（HOBBIT on RTX 4090，Mixtral-8x7B FP16+INT4）：
    - **模型推理算法层**：每 token 进入 MoE layer → Router 计算 top-2 gate weights → ||G(x)|| 归一化后计算 s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}|| → e_0 得分 0（高精度）→ e_1 得分 = ||G(x)_{e_0}||，若 ≤T1 为高精度加载，≤T2 为 INT4 加载（4× 更小），>T2 跳过。INT4 expert FFN 计算使用量化 GEMM。精度下降 GSM8K 准确率最大仅从 0.52→0.51（FP16→FP16+INT4）。
    - **系统框架层**：基于 Llama.cpp（8,000 行 C++/C 修改）。权重分布：所有 non-expert + 多精度 expert cache 驻留 GPU memory。主线程 GPU 计算 + scheduler 线程异步加载。Dynamic Expert Loader 通过 read() 系统调用从 CPU memory 加载对应精度 expert。Adaptive Expert Predictor Stacking Computer 一次性矩阵乘预测后续层 expert。Multidimensional Cache Manager Policy Performer 维护 LRU/LFU/LHU/FLD 记录，加权公式决定 eviction。
    - **编译框架层**：论文未明确说明。Llama.cpp 原生 CPU/GPU 混合编译，使用 CUDA/OpenCL 后端。
    - **kernel 调度层**：除标准高精度 GEMM 外，低精度 (INT4/INT2) expert 使用对应的量化矩阵乘 kernel。expert loading 从单一 FP16 大块拷贝变为多精度分块异步加载（FP16 expert ~10.5MB vs INT4 expert ~2.6MB）。由于 67%/30%/3% 的精度分布，平均加载量大幅减少。
    - **硬件架构层**：与 baseline 相同（RTX 4090 24GB + CPU 256GB，PCIe 4.0 32GB/s）。关键差异在于 PCIe 传输量：baseline 加载 2 个 FP16 expert（~21MB），HOBBIT 平均加载 ~12.4MB（1.0×FP16 + 0.3×INT4 + 0.03×skip），传输时间减少 ~41%。
    - **关键结果对比**：
      - vs MoE-Infinity on RTX 4090：decoding speedup 2.30× (Mixtral) / 3.92× (Phi-MoE)，prefill latency 降低 14%/29%。
      - vs MoE-Offloading on RTX 4090：decoding speedup 3.21× (Mixtral) / 3.29× (Phi-MoE)，prefill latency 降低 51%/54%。
      - vs Llama.cpp on Jetson Orin：decoding speedup 13.0× (Mixtral) / 18.9× (Phi-MoE)。
      - 精度：GSM8K 和 TruthfulQA 上所有配置下 accuracy 下降 ≤1%。

## HMoE: Heterogeneous Mixture of Experts for Language Modeling

- baseline方法是什么？
  - **Homogeneous MoE（同构专家混合）**：传统 MoE 模型（GShard, Switch Transformer, Mixtral, DeepSeekMoE）中所有 expert 具有相同的结构和参数量（相同的 FFN hidden dimension）。每个 expert 为 h_input × h_ffn 的 FFN，通过 Top-K (k=2) 或 Top-P routing 选择 1 到多个 expert。使用 load balancing loss L_lb = N · Σ T_i · P̂_i 鼓励均匀的 expert 负载分布。全栈执行例子（以 Homogeneous MoE-3B, Top-P routing, 8×A800, RedPajama 预训练为例）：
    - **模型训练算法层**：LLaMA-based decoder-only, 12 layers, 8 homogeneous experts/layer, 每 expert FFN hidden=4096 (总和32768)。Token 进入每层 → Router softmax(W_r·x) → Top-P (p=0.6) 动态选择 expert → selected experts 各自计算相同大小的 FFN (W_g [4096,4096], W_p [4096,4096], W_o [4096,4096])，gate 加权求和输出。训练 loss = L_lm + λ·L_lb（λ=1e-2）。
    - **系统框架层**：PyTorch + DeepSpeed Zero2 + gradient checkpointing。Expert 分布在 8 GPU 上（expert parallelism 或 DP），all-to-all dispatch/combine 通信。每个 GPU 持有部分 expert 的完整参数。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 后端。
    - **kernel 调度层**：标准 cuBLAS GEMM kernel + NCCL all-to-all collective。所有 expert 使用相同大小的 GEMM——token 需 padding 对齐 batch size 或使用 Megablocks block-sparse kernel。同构设计简化了批量计算（统一 GEMM shape）。
    - **硬件架构层**：NVIDIA A800 (80GB) 或 H800 (80GB)，同构 GPU 集群。
  - Baseline 痛点：
    1. **缺乏专家专业化（核心痛点 1）**：同构 expert 具有相同的建模能力，训练中 router 随机分配 token，导致 expert 学习到相似的表示（representation convergence, Zhou et al. 2022），expert 间缺乏显著的知识差异和专业分化。
    2. **低效的参数分配（核心痛点 2）**：所有 token——无论简单或复杂——都被分配给相同大小的 expert，简单 token（如常见冠词/介词）和复杂 token（如需要深度推理的词汇）消耗相同计算资源，造成参数浪费和计算效率低下。Top-P routing 尝试通过动态激活不同数量 expert 来应对，但依赖固定阈值和粗略的难度建模，无法自适应多样输入。
    3. **表示坍塌和负载不均衡（核心痛点 3）**：同构 MoE 倾向于 representation collapse——多数 token 被分配给少数 expert，导致负载不均衡（load imbalance）。虽然 load balancing loss 鼓励均匀分配，但它强制所有 expert 被均等使用，无论 token 复杂度，本质上与"让不同 expert 处理不同复杂度 token"的目标矛盾。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HMoE 方法**：通过异构 expert 设计 + P-Penalty 训练目标两个核心机制解决 baseline 的全部痛点：
    1. **异构 Expert 大小（解决痛点 1）**：为不同 expert 分配不同的 FFN hidden dimension（如 {2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888} for 3B model），使 expert 天然具有不同的表示容量。大 expert（5888 dim）有更强的建模能力，适合处理复杂语义 token；小 expert（2304 dim）容量有限但计算经济，适合处理简单 token。这种容量差异强迫 router 根据 token 复杂度做差异化分配，打破同构 expert 的表示趋同问题。实验验证：HMoE 中专家相似度更低（Wasserstein distance 聚类显示不同大小 expert 形成明显差异化分组），而异构 MoE 中 expert 趋向于两个相似聚类。
    2. **P-Penalty Loss（解决痛点 2）**：提出 Parameter Penalty loss L_P-Penalty = N · Σ M_i · P̂_i，其中 M_i = (1/T) Σ 1{e_i ∈ E^t} × h_ffn,i，将 expert 大小直接纳入损失。激活大 expert 时 penalty 更大（因为 h_ffn,i 大），鼓励模型优先激活小 expert 处理简单 token，仅在必要时激活大 expert 处理复杂 token。对比传统 load balancing loss（仅追求均匀激活，不考虑 expert 大小差异），P-Penalty 实现了"按需使用"的参数经济性。实验验证：训练过程中小 expert 激活率持续上升，大 expert 激活率下降，同时总激活参数量呈下降趋势，实现更低 loss 同时更少激活参数。
    3. **Router Entropy Loss + Top-P 协同（解决痛点 3 的辅助机制）**：对 Top-P routing 额外加入 router entropy loss L_entropy = N · Σ P_i · log(P_i)，防止训练中激活 expert 数量无限制增长。异构设计 + P-Penalty + Top-P routing 三者协同：异构提供容量差异，P-Penalty 引导偏好小 expert，Top-P 动态选择适配每个 token 的真实需求，同时 entropy loss 防止过度稀疏化。
  - 全栈执行例子（HMoE-3B, Top-P routing, 8×A800, RedPajama 预训练，与 baseline 同配置对比）：
    - **模型训练算法层**：LLaMA-based decoder-only, 12 layers, 8 heterogeneous experts/layer。Token 进入每层 → Router softmax(W_r·x) → Top-P (p=0.6) 动态选择 expert → selected experts 各自计算不同大小的 FFN：
      - Small expert e_0 (h_ffn=2304): e_0(x) = W_o,0 · (SiLU(W_g,0·x) ⊙ (W_p,0·x)), W_g: [4096,2304]
      - Large expert e_7 (h_ffn=5888): e_7(x) = W_o,7 · (SiLU(W_g,7·x) ⊙ (W_p,7·x)), W_g: [4096,5888]
      - Gate 加权组合输出
      - 训练 loss = L_lm + α·L_P-Penalty + β·L_entropy (α=0.1, β=3e-2)
    - **系统框架层**：PyTorch + DeepSpeed Zero2 + gradient checkpointing（与 baseline 相同）。差异：异构 expert 的不规则形状需要 Megablocks block-sparse kernel 或 ES-MoE expert-wise offloading 来高效执行批量计算（而非传统 unified GEMM）。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 后端。
    - **kernel 调度层**：不同于 baseline 的统一 GEMM shape（所有 expert 都是 [4096,4096]），HMoE 中不同 expert 的 GEMM 形状各异（从 [4096,2304] 到 [4096,5888]）。使用 Megablocks block-sparse 矩阵乘法 kernel 处理不规则 expert 计算，或使用 ES-MoE 方式将 expert 参数 offload 到 CPU 后按需加载。P-Penalty loss 引导下，token 流向偏向小 expert（更小 GEMM），实际总计算量低于 baseline。
    - **硬件架构层**：与 baseline 相同（NVIDIA A800/H800 80GB）。
    - **关键结果对比**：
      - 3B scale: HMoE (Top-P) avg=46.53 vs MoE (Top-P) avg=45.62，激活参数 0.68B vs 1.23B（减少 45% 激活参数的同时提升 0.91 avg score）
      - 0.4B scale: HMoE (Top-P) avg=44.51 vs MoE (Top-K) avg=43.45，激活参数 173M vs 163M（HMoE 用更少参数获得更好性能 vs MoE-TopK）
      - isoFLOP 曲线：HMoE 从 ~2×10^19 FLOPs 起稳定优于 Homogeneous MoE，且随训练规模增大优势扩大
  - **核心设计洞察**：HMoE 的本质洞察是 MoE 中"同构"不是必然的——expert 间的参数同构假设（同一大小）导致了专家专业化的坍塌和参数效率的低下。通过引入容量异构（不同大小的 expert）和配套的 P-Penalty 激励信号，HMoE 将 MoE 训练从"均匀分配 token + 均匀使用 expert"的均值模式转变为"按 token 复杂度差异化分配 + 经济性激励"的市场模式。P-Penalty 的精妙之处在于它将 expert 大小作为显式信号融入 loss，使得"少用大 expert"成为可优化的目标而不是外部约束，从而在不牺牲模型表达能力的前提下实现参数经济性。实证中的关键发现：适度异构（arithmetic, ratio≈2.5x）优于极度异构（geometric, ratio=128x）和完全同构（ratio=1x），说明最佳异构度需要在"容量差异足以驱动专业化"和"小 expert 仍有足够能力参与训练"之间取得平衡。

## HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

- baseline方法是什么？
  - **Tutel (Static MoE Library)**：使用 expert parallelism + 所有 token 的 all-to-all dispatch/combine 通信，配合 expert capacity 超参数校准各 expert workload。通过 GeMM 接口执行 expert FFN 计算。由于各 expert 的 workload 动态变化，需要 token padding（填充到 capacity）或 discarding（丢弃超出 capacity 的 token），产生冗余 FLOPs 和冗余内存分配/访问。全栈执行例子（以 Swin-MoE-Base, top-2 routing, 4×RTX 4090, Tutel baseline 训练单步为例）：
    - **模型训练算法层**：Swin-MoE-Base, MoE 层替换标准 FFN，每层 E 个 expert (E=4/8)，top-k routing。Forward: Gate softmax(W_g·x) → top-k selection → dispatch tokens 到各 expert → 各 expert 用 GeMM 计算 FFN (需 token padding 对齐 capacity) → combine 聚合输出。Backward: 按 auto-diff 计算各 expert 权重梯度和输入梯度。Auxiliary load balancing loss 鼓励 expert 均匀使用。
    - **系统框架层**：Tutel 基于 PyTorch，expert parallelism 将 expert 分布到多 GPU。执行流程：① attention 计算（数据并行）→ ② MoE gate routing → ③ all-to-all dispatch（跨设备同步发送 token）→ ④ 各设备本地 GeMM（grouped GeMM 或 padding 后标准 GeMM）→ ⑤ all-to-all combine（跨设备同步聚合输出）。All-to-all 通信可占 40%+ runtime。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 通信后端。
    - **kernel 调度层**：NCCL all-to-all collective kernel + cuBLAS GEMM kernel。执行顺序：all-to-all（通信）→ GeMM（计算）→ all-to-all（通信），通信和计算严格串行。token padding 引入冗余内存分配和冗余 FLOPs（padding token 的计算结果被丢弃）。
    - **硬件架构层**：4× NVIDIA RTX 4090 (24 GB)，同构 GPU 集群，节点内 PCIe/NVLink 互联。
  - **MegaBlocks (Dynamic MoE Library)**：使用 block-sparse 操作和对应 GPU kernel 替代 GeMM，消除 token padding。但运行时行为动态不确定，可能触发 OOM；且 grouped GeMM 的动态性导致 memory 管理复杂。
  - Baseline 痛点：
    1. **冗余 FLOPs 和内存（核心痛点 1）**：Expert parallelism + GeMM 接口强制 token padding/discarding。Tutel 的 expert capacity 超参数需要 hand-tune——capacity 太小则丢弃 token 损失精度，capacity 太大则大量 padding token 产生冗余计算和冗余内存。MegaBlocks 虽消除 padding 但引入 runtime OOM 风险。
    2. **All-to-All 通信瓶颈（核心痛点 2）**：Expert parallelism 依赖同步 all-to-all dispatch/combine 通信，可占 40%+ runtime。随模型规模和设备数增加，通信开销线性增长。
    3. **同构设备限制（核心痛点 3）**：Expert parallelism 主要部署在同构设备上。异构设备（新旧 GPU 混用）更便宜易获取，但 expert parallelism 的负载均衡依赖同构硬件假设，无法直接利用异构设备的差异化计算能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HEXA-MoE 方法**：通过三个系统性设计解决 baseline 的全部痛点：
    1. **Expert-Specific Operators (ESMM, ESS, ESTMM) 替代 GeMM**（解决痛点 1）：将 MoE 前向和反向传播重新公式化为三个 expert-specific 算子——ESMM（expert-wise 矩阵乘法）、ESS（expert-wise 求和累加）、ESTMM（expert-wise 转置矩阵乘法）。每个 token 仅与其路由 expert 的权重做矩阵乘法，无需 token padding/discarding，实现 **in-place 计算**，几乎零冗余 FLOPs。公式化对比：传统方法对每 expert 构造 N_e×D 的 padded batch（含 padding token），HEXA-MoE 直接对原始 N×D batch 做 expert-specific 计算——ESMM 通过 re-index vector 指导 I/O，thread-block 加载同 expert 的 tokens 和对应权重，跳过 padding 位置（填 -1 跳过），结果原位写回。对比 Tutel 的 "capacity padding + GeMM" pipeline，HEXA-MoE 的 "re-index + ESMM" pipeline 消除了 dispatch/combine 中的 token 重排和 padding 计算。
    2. **Tensor Parallelism 替代 Expert Parallelism**（解决痛点 2）：用 tensor parallelism（沿 FFN intermediate size 切分）替代 expert parallelism。Data-centric 下各设备 all gather 完整 MoE 参数后本地计算（无 all-to-all）；Model-centric 下各设备 all gather 数据批次后本地计算参数 chunk（all reduce 替代 all-to-all）。Tensor parallelism 的 all gather / all reduce 通信 pattern 比 expert parallelism 的 all-to-all 更规整，且配合 pipeline-shared cache 可实现通信-计算重叠。
    3. **Heterogeneous-Aware Expert Allocation**（解决痛点 3）：基于各设备 benchmark 延迟按反比分配 workload——Data-centric 下调整 local batch size B_i，Model-centric 下调整 FFN intermediate sub-dimension h_i。Tensor parallelism 的 workload 可精确预测（由 batch size 或 sub-dimension 决定），使异构调度成为确定性问题，无需处理 expert parallelism 的动态 workload 不确定性。
  - 全栈执行例子（HEXA-MoE, Swin-MoE-Base, data-centric, 4×RTX 4090，与 baseline 同配置对比）：
    - **模型训练算法层**：与 baseline 相同的 Swin-MoE-Base 模型结构，差异在 MoE 层的计算方式：
      - Forward: y1 = ESMM(x, W1, b1, R(x)) → y2 = F(y1) → y = ESMM(y2, W2, b2, R(x))
      - Backward: ∂ℓ/∂b2 = ESS(∂ℓ/∂y, R(x)) → ∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x)) → ∂ℓ/∂y2 = ESMM(∂ℓ/∂y, W2^T, null, R(x)) → ∂ℓ/∂y1 = ∂ℓ/∂y2 ⊙ F'(y1) → ∂ℓ/∂b1 = ESS(∂ℓ/∂y1, R(x)) → ∂ℓ/∂W1 = ESTMM(x, ∂ℓ/∂y1, R(x)) → ∂ℓ/∂x = ESMM(∂ℓ/∂y1, W1^T, null, R(x))
      - Backward 中 3 个算子（ESS, ESMM, ESTMM）融合为 ESFK（单 kernel），单 MoE 层 backward 仅需 2 fused kernels + 1 element-wise dot product。
    - **系统框架层**：基于 PyTorch + CUDA C++ 自定义 kernel。Tensor parallelism 替代 expert parallelism——各设备沿 FFN intermediate size 切分所有 expert 的权重。Data-centric 执行流程：① attention + router 计算 → ② (并行) all gather MoE 参数到 pipeline-shared cache → ③ ESMM 用 cache 中的完整 MoE 参数本地计算 → ④ 下一层。All gather 与 attention 在分离 CUDA stream 上重叠。对比 baseline Tutel 的 "all-to-all dispatch → GeMM → all-to-all combine"，HEXA-MoE 的 "all gather → ESMM" 流程无 token 重排、无 padding、通信量更可控。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 通信后端 + 自定义 CUDA kernel。
    - **kernel 调度层**：ESMM kernel 使用 re-index vector 指导 I/O——thread-block 加载 sub-vector 定位同 expert tokens → 加载对应 expert 权重（仅一次）→ Tensor Core MMA (nvcuda::wmma) → 按 sub-vector 写回。ESS kernel 按 expert+channel 分配 thread-block → 累加同 expert tokens。ESTMM kernel 对 re-indexed 输入做 expert-wise 外积。ESFK 融合 kernel 统一 thread-block shape 为 (WARP, TIMES)，grid 扩展为 3 维聚合 ESS+ESMM+ESTMM。Pipeline-shared cache 在 HBM 额外区域动态缓存 gathered shards。对比 baseline Tutel 的 "NCCL all-to-all + cuBLAS GEMM"，HEXA-MoE 的 "NCCL all gather + ESMM/ESFK kernel" pipeline 消除了 token padding 的冗余 kernel launch 和冗余 HBM 访问。
    - **硬件架构层**：与 baseline 相同（4×RTX 4090）。结果：10%-48% 内存节省，0.5-4.3× 加速 vs Tutel/MegaBlocks。Heterogeneous 实验（TITAN RTX + RTX 2080 Ti）：data-centric 下 optimal allocation 可降低延迟 13.2%-25.3%，model-centric 下降低 6.3%-11.9%。
  - **核心设计洞察**：HEXA-MoE 的本质洞察是 MoE 计算中存在"冗余的源头"——token padding/discarding 不是 MoE 的本质需求，而是 GeMM 接口强加的人为约束。通过将 MoE 从 "GeMM 视角"（先重排为规则 batch 再调 GeMM）重新定义为 "Expert-Specific 视角"（不重排 token，直接做 expert-wise 计算），HEXA-MoE 从根本上消除了冗余 FLOPs 的源头。这一视角转换带来两个连锁效应：(1) Tensor parallelism 变得自然可行——expert-specific 算子天然适配沿 intermediate size 的切分（ESMM 中每个 expert 的权重可独立切分），使 expert parallelism 的 all-to-all 被替换为 tensor parallelism 的 all gather/reduce；(2) 异构调度变得确定——tensor parallelism 下各设备的 workload 由 batch size 或 sub-dimension 精确决定，消除了 expert parallelism 中动态 workload 的不确定性。EXA-MoE 的 elegant 之处在于它不需要在现有系统上"修补"冗余，而是通过重新定义计算范式使冗余在数学层面消失。

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

## GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

- baseline方法是什么？
  - **ERM (Empirical Risk Minimization) 训练的标准 GNN**：在源分布 D_s 上通过最小化经验风险训练 GNN（GCN/GIN/GAT），直接用于目标分布 D_t 的推理。全栈执行例子（以 WebKB node classification 为例）：输入网页节点特征 x_i + 邻接矩阵 A → 3层 GCN（GraphConv → ReLU → GraphConv → ReLU → GraphConv）→ 节点 embedding → MLP classifier → 输出 5-class 概率。训练使用 Adam optimizer + cross-entropy loss。
  - **Invariant Learning 方法（IRM [1], VREx [28], EERM [67]）**：假设存在对环境变量不变的表示或预测器，通过环境划分（environment partition）学习 invariant representations。全栈执行例子（EERM）：将源域数据构造成多个环境 → 对每个环境训练 GNN encoder → 通过正则化项强制 encoder 在不同环境间产生相似的表示 → classifier 在 invariant representation 上训练。IRM 额外约束最优分类器在环境间一致。
  - **Data Augmentation 方法（G-Mixup [22], SRGNN [85], OOD-G-Mixup）**：通过对训练数据进行特定类型 shift 的增广（如 graph size variation、local structure perturbation）来提升分布外泛化能力。全栈执行例子（G-Mixup）：在训练图的图元空间进行 mixup → 生成虚拟 OOD 样本 → 在增强数据集上训练 GNN。
  - **Graph-Specific OOD 方法（DIR [69], GSAT [45], CIGA [6]）**：通过 causal intervention 或 attention stochasticity 学习因果子图/不变子结构。全栈执行例子（DIR）：构建 intervention 分布 → 通过 causal 干预蒸馏 causal subgraph pattern → 在 causal subgraph 上做分类。GSAT：在 attention weights 中注入 stochasticity → 通过 information bottleneck 原则阻断 label-irrelevant 信息。
  - Baseline 痛点：
    1. **单一 shift 假设与现实脱节（核心痛点）**：现有 data augmentation 方法假设目标分布遵循某种特定的 shift 类型（如 graph size [49, 14]、feature noise [26, 8]、degree shift [65, 39]），但真实世界的分布偏移往往由多个 shift 维度融合而成（如 WebKB 的大学域偏移、Twitch 的语言域偏移），且每个维度的统计特性不同。单一维度的合成增广无法覆盖这种复合 shift。
    2. **环境划分的组合爆炸**：Invariant learning 方法依赖将数据划分为不同环境来学习不变性。但在复杂分布偏移下，环境空间 E 是巨大的——环境由不同子集节点和不同 shift 维度的组合构成，环境数量呈组合爆炸（product of subsets × shifts），使标准 invariant learning 不可行。
    3. **忽略 instance-wise heterogeneity（关键痛点）**：同一目标分布中，不同 instance（node/graph）可能经历不同类型和不同程度的分布偏移（如图 1 中 u^1 和 u^2 在 target domain 的内容特征变化程度不同）。现有 invariant learning 关注 group-level patterns，缺乏对 instance 级异质性的建模能力。
    4. **环境变量依赖**：EERM 等方法需要 domain/environment 信息来指导训练，但许多场景下 domain 标签不可得。GraphMETRO 不需要 domain 信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **GraphMETRO 方法**：基于 Mixture-of-Experts 架构，通过"分解-对齐-聚合"三步策略处理复杂图分布偏移。
    1. **Shift Decomposition via Mixture Modeling**（解决痛点 1 和 2）：代替直接学习不变预测器，将未知分布偏移分解为 K 个 shift components，每个 component 由一种 stochastic transform function τ_i 定义（如 subgraph sampling、feature noise、edge removal）。关键假设（Assumption 1）：任意分布偏移都可以建模为最多 k 个 transform classes 的混合（k ≤ K）。这避免了环境空间的组合爆炸——用 K 个 basis transforms 替代 combinatorial environments，通过连续权重 w ∈ R^{K+1} 实现无限环境表达。
    2. **Instance-Adaptive Gating via MoE Architecture**（解决痛点 3）：Gating model ϕ 对每个 graph/node instance 输出个性化的权重向量 w ∈ R^{K+1}，权重编码了该 instance 的分布偏移中各个 shift component 的贡献度。这使得模型能捕捉 instance-wise heterogeneity——不同 instance 的 w 不同，因而被不同类型和程度的 shift 影响时产生不同的 expert 组合。
    3. **Referential Invariant Representation**（核心创新）：不同于传统 invariant learning 直接优化"表示在环境间不变"，GraphMETRO 设计每个 expert ξ_i 对其对应的 τ_i 产生 referentially invariant 表示：ξ_0(G) ≈ ξ_i(τ_i(G))。ξ_0 作为 reference model 为所有 expert 提供统一的表示空间 anchor。这使得不同 expert 的输出处于相同表示空间，可以在聚合时避免信息丢失。
    4. **Alignment via Frobenius Distance**（解决 aggregation 兼容性）：在 L2 中加入 Frobenius norm distance penalty d = (1/n)·||h(τ^{(k)}(G)) - ξ_0(G)||_F，强制聚合后的表示与 reference model 对齐。若缺少此对齐项（λ=0），WebKB 准确率从 41.11 暴跌至 18.79，验证了 expert 输出空间对齐的必要性。
    5. **τ^{(k)}-invariance 训练目标**：通过联合采样 τ^{(k)}（k 个 transform 的组合）和在 L2 中同时优化分类+对齐，模型学会对组合 shift 产生不变性（Theorem 2: composition of shifts）。
  - 全栈执行例子（GraphMETRO 在 WebKB node classification，与 ERM baseline 对比）：
    - **算法层**：
      - Baseline (ERM GCN)：x_i → 3× GCNConv → node embedding h_i → MLP → prediction。对 domain shift 无任何处理。
      - GraphMETRO：
        1. 对输入 subgraph 应用 5 种 τ_i（noisy_node_feat, add_edge, drop_edge, drop_node, random_subgraph）
        2. Gating GNN ϕ 输出 w ∈ R^6（K=5 + reference expert），表征该节点的分布偏移成分
        3. K+1=6 个 expert GNNs 分别编码 → reference expert ξ_0 在原始图上编码，其他 expert 在 transformed 图上编码
        4. Softmax(w) 加权聚合 → h → MLP classifier → prediction
        5. 训练时 L1（BCE: gating 预测正确 τ_i 组合）+ L2（CE + Frobenius alignment）
    - **系统框架层**：基于 PyG (PyTorch Geometric) 实现，标准 GNN 训练框架。GraphMETRO 训练时每 batch 对每个 graph 采样 τ^{(k)} 并生成变换图。独立 GNN encoder design（每个 expert 一个完整 GNN）更占内存但 expressiveness 更强；共享 encoder design 更省内存但性能降低（WebKB: 31.14 vs 41.11）。
    - **编译框架层**：论文未明确说明。使用 PyTorch 标准编译栈。
    - **Kernel 调度层**：论文未明确说明。标准 PyTorch Scatter/Gather + cuBLAS GEMM，无自定义 CUDA kernel。
    - **硬件架构/芯片设计层**：论文未明确说明。使用 NVIDIA GPU。
  - 关键实验数据：
    - Real-world datasets (Table 1): WebKB 41.11% (vs EERM 24.61%, +67.0%), Twitch 53.50% (vs EERM 51.34%, +4.2%), Twitter 57.24% (vs GSAT 56.40%), SST2 81.87% (vs DIR 81.55%)
    - Synthetic datasets: Average +4.6% over ERM across all shift environments
    - Synthetic DBLP average: GraphMETRO 81.08 vs ERM 77.88 vs ERM-Aug 78.63
    - Ablation: w/o L1 → WebKB 41.11→23.22 (gating 失效导致 expert 选择不准), λ=0 → WebKB 41.11→18.79 (alignment 是核心设计)
    - Invariance Matrix (Fig 4a): 对角线值最小，验证每个 expert 专精于其对应的 shift component
    - Distribution Discovery (Fig 4b): WebKB 主导 shift=add_edge, Twitch 主导 shift=noisy_node_feat+drop_node
    - Gating accuracy: WebKB 92.4%, Twitch 93.8%（多标签二分类）

  - **核心设计洞察**：GraphMETRO 的核心贡献是将"复杂分布偏移的泛化"问题从 invariant learning 的"寻找在所有环境中都不变的表示"重新定义为"将偏移分解为基础成分，智能地组合 expert 的表示来适应每个 instance"。关键在于 MoE 架构天然适合这个范式——gating 负责识别偏移成分（decomposition），expert 负责消除各自对应的偏移（mitigation），weighted aggregation 负责组合输出适应 instance-specific 的偏移（adaptation）。Referential invariant representation 是一个优雅的设计：通过 reference model ξ_0 作为 anchor，解决了不同 expert 独立训练时表示空间不兼容的问题，将 K+1 个表示空间的"翻译"简化为每个空间与同一 reference 空间的"对齐"。这使得聚合操作（weighted sum）在数学上有意义，而非简单地将异质向量拼在一起。

- baseline方法是什么？
  - **GPT-3 (175B Dense Decoder-only)**：标准 dense Transformer，175B 参数全部在每 token 推理时激活 (nact-params=175B)。全栈执行例子（推理 1 token）：Token → Embedding [1, 12288] → 96 层 dense Transformer（每层包含 Multi-head Self-Attention + Standard FFN, ReLU activation, 绝对位置编码）→ 所有 175B 参数参与逐层 matrix multiply → LM head → logits。FLOPs/Token=350G。训练使用 V100 GPU 集群, 训练能耗 1287 MWh, 300B tokens。
  - **GLaM Dense 基线**：同架构但无 MoE 层的 dense decoder-only 模型（0.1B, 1.7B, 8B, 137B），使用与 GLaM MoE 相同的数据集、tokenizer、优化器和超参数训练。
  - Baseline 痛点：
    1. **计算效率**：Dense 模型每 token 激活全部参数，FLOPs 随参数线性增长。训练能耗极高（GPT-3 1287 MWh）。
    2. **参数容量受限于计算预算**：在固定 FLOPs 预算下 dense 模型的参数量即表达能力上限，无法通过增加参数不增加计算的方式扩展容量。
    3. **知识存储效率低**：Dense 模型的知识密集型任务（如 TriviaQA）性能受限于 nact-params，而扩展 nact-params 同时线性增加推理成本。
    4. **数据效率低**：Dense 模型需要更多 training tokens 才能达到与 MoE 模型相同的 downstream 性能。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **GLaM 方法**：基于 sparse MoE 的 decoder-only Transformer。每隔一层 Transformer FFN 替换为 64-expert MoE 层，top-2 softmax gating 动态路由每 token 到 2 个 expert。1.2T 总参数，每 token 仅激活 96.6B（8%）。非 MoE 层使用 GLU+GeGLU 提升表达能力。采用 GSPMD 2D sharding 实现 expert 维度 + hidden 维度并行分布。
  - 解决 baseline 缺陷的对应机制：
    1. **计算效率飞跃**：sparse activation 使推理 FLOPs/Token 降至 180G（≈GPT-3 的 51.4%），训练能耗降至 456 MWh（≈GPT-3 的 35.4%）。原因：每 token 仅激活 2/64 experts，计算量主要由 nact-params (96.6B) 决定而非 nparams (1.2T)。
    2. **解耦参数容量与计算量**：通过增加 expert 数量 (1→256) 在固定 FLOPs 预算下指数级增加模型容量 (nparams 从 1.7B→105B, nact-params 仅从 1.700B→1.886B)，实现 "more capacity, same compute"。O(E²) 种 expert 组合为每 token 提供灵活的 sub-network 选择。
    3. **知识存储密度提升**：TriviaQA one-shot 达 75.8%（GPT-3 one-shot 68.0%），超越 fine-tuned SOTA KG-FiD (69.8%)。因 1.2T 参数提供了 7× GPT-3 的知识存储空间，而推理计算量仅为一半。
    4. **数据效率提升**：相同 training token 量下 MoE 模型性能远超 dense（Fig 4），GLaM (64B/64E) 用 280B tokens 训练即匹配/超越 GPT-3 用 300B tokens 的性能。因 expert 专业化使有限数据学习更有效的表示。
  - 全栈执行例子（GLaM 64B/64E 推理单 token）：
    - **算法层**：Token → SentencePiece tokenizer (vocab 256K) → Embedding [1, M=8192] → 64 层 decoder-only Transformer：
      - Layer 0 (Dense Attention): Q/K/V 投影 [M=8192, dhead=128 × nheads=128] → Relative Positional Bias → Softmax(QK^T/√d + rel_bias) → Attn output
      - Layer 1 (MoE FFN): Gating softmax(W_gate·x) 输出 [1, E=64] → top-2 选 expert i,j → dispatch x 到 expert_i, expert_j → 各 expert FFN: Linear[8192→32768] → GeGLU → Linear[32768→8192] → gate 加权 sum → residual add
      - Layer 2-63 交替 Attention/MoE 层... 
      - Non-MoE FFN 层: GLU → gate=GeGLU(x·W_g), value=x·W_v → (gate * value)·W_o → residual add
      → Final LM head [8192→256K] → softmax → next token logits。
    - **系统框架层**：模型权重和计算通过 GSPMD 2D sharding 分布到 1,024 TPU-v4 芯片。Expert tensor [E=64, M=8192, H=32768] 沿 E 和 H 划分。Input activation [B, S=1024, M=8192] 沿 B 和 M 划分。同一个 index 的 expert 跨层驻留同一 device（减少 expert-to-device 映射开销）。MoE 层用 while_loop 包装重复模块以降低 XLA 编译时间。All-to-all 通信用于 token dispatch/combine（expert 并行模式）。
    - **编译框架层**：使用 GSPMD 编译器自动推导非显式标注张量的 sharding 属性。XLA 编译器进行 TPU 后端代码生成和 while_loop 控制流优化。论文在 Section C 中描述了将 expert 按 index 对齐跨层放置以生成 "identical computation graph"，从而复用 while_loop 编译结果。
    - **Kernel 调度层**：论文未明确说明。TPU-v4 使用 TPU-specific matrix multiply unit (MXU) 执行 expert FFN 的 dense matmul。Sparse gating 的 token dispatch/gather 通过 all-to-all collective 通信实现（Lepikhin et al. 2021 GShard protocol）。
    - **硬件架构/芯片设计层**：使用 Google TPU-v4（326W/chip, PUE 1.11），未涉及自定义 RTL 或芯片架构修改。训练总能耗 456 MWh（600B tokens）或 213 MWh（280B tokens）。TPU-v4 的 bfloat16 支持用于激活值，float32 用于权重。
  - 关键实验数据：GLaM (64B/64E) vs GPT-3 (175B)：Zero-shot Avg NLG 54.6 vs 47.6, NLU 66.2 vs 60.8；One-shot NLG 58.4 vs 52.9, NLU 68.6 vs 65.4；Few-shot NLG 61.6 vs 58.8, NLU 71.4 vs 68.4。FLOPs/token: 180G vs 350G (-48.6%)。训练能耗: 456 MWh vs 1287 MWh (-64.6%)。CO₂排放: 40.2 tCO₂e vs 552 tCO₂e。

## FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

- baseline方法是什么？
  - **LoRA(r=8) / LoRA(r=32)**：标准低秩适配，将参数更新 ΔW 分解为 ΔW = B_{m×r}·A_{r×n}，其中 A 和 B 均可训练。全栈执行例子：输入 token x → 预训练权重 W0·x + (α/r)·B·(A·x)。存在 **intra-task interference**：同一 LoRA 组件内不同 rank 间梯度相关（full covariance），rank 间参数耦合导致 suboptimal 性能，且高 rank 需要更多训练参数。存在 **inter-task interference**：多任务模型合并时，task_i 的 B_i·A_i 与 task_j 的 B_j·A_j 在参数空间中重叠（因可训练 A 之间无正交性保证），合并出现性能崩塌（如 ScienceQA 上 LoRA(r=8) 合并后 Δ=-60.34%）。
  - **Split-LoRA(4×8)**（MoE-based LoRA 代表）：将 rank r=32 分解为 4 个 expert × 8 rank，router G(x)=top-k(sigmoid(W_g·x)) 选择激活 expert。引入显式 router 参数 W_g∈R^{N×n}。全栈执行例子：输入 x → router 计算 softmax(W_g·x) 选 top-k expert → Σ_i G(x)_i·B_i·(A_i·x)。痛点：(1) router 参数开销随 expert 数 N 线性增长（N=32 experts 时 router 成为瓶颈）；(2) 单 expert 内仍存在 rank 间 interference；(3) 多任务合并无本质改善（ScienceQA Δ=-54.74%），因 A 仍可训练、Expert 更新不天然正交。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FlyLoRA 方法**：基于果蝇嗅觉回路的三层设计。(1) 矩阵 A 替换为**冻结的稀疏随机投影**（每行 p 个非零 ~N(0,1/r²)），既是下投影又是隐式 router；(2) 在 B 矩阵中执行 **rank-wise top-k 专家激活**——r=32 个 rank-1 专家，每 token 仅激活 top-k=8 个；(3) 通过 **loss-free 负载均衡偏置 d** 辅助专家选择。
  - 解决 baseline 缺陷的对应机制：
    1. **消除显式 router 参数**：A 的稀疏随机投影天然保留了 pairwise 距离（Theorem 3.1, Johnson-Lindenstrauss 延伸），语义相似 token 被路由到相似 expert，实现与 hash router 等效的轻量级隐式路由。不再需要 W_g，彻底消除 router 参数开销。激活训练参数仅 d·k（vs LoRA 的 2·d·k 和 Split-LoRA 的 2·d·k+d·N）。
    2. **Intra-task 去耦合**：top-k 稀疏激活使不同 B 列的梯度仅在同时被选中时才产生协方差。Theorem 3.3 证明 off-diagonal 梯度协方差按 k²/r² 因子缩减。当 k=8, r=32 时，off-diagonal 协方差约为 dense 训练（k=r）的 1/16。图 3(b-c) 的梯度相关热力图验证了 FlyLoRA 显著稀疏的正交模式。
    3. **Inter-task 去耦合**：独立初始化的冻结随机投影 A_i 和 A_j 天然近似正交。Theorem 3.4/Collorary 3.5 证明 <B_i·A_i, B_j·A_j>_F ≈ 0，不同任务的参数更新位于近似正交子空间，合并时无破坏性 interference。模型合并后 MMLU 仅降 2.02%（vs LoRA 降 4.91-6.48%，Split-LoRA 降 4.86%），ScienceQA 降 43.05%（vs LoRA 降 60.34%）。
  - 全栈执行例子（FlyLoRA 单 task 训练+合并推理）：
    - **算法层**：Token x → 冻结 A 做 sparse projection y=A·x（每行 p 稀疏非零，O(r·p)）→ 加负载偏置 d → top-k argmax 选 I_topk → 仅激活 B 的 k 列 → ΔW·x = (α/r)·Σ_{i∈I_topk} b_i·(a_i·x)。Backward 仅更新 B：grad_B 中非 top-k 列为 0。
    - **系统框架层**：论文使用标准 PyTorch + Transformers 训练框架（AdamW optimizer），未修改 serving 框架。PEFT 注入方式：对 LLM 所有 linear 层（q,k,v,o,gate,down,up}_proj）替换为 FlyLoRA 模块。训练内存：FlyLoRA 在 Llama-3.1-8B 上最低 10.6GB（vs LoRA(r=8) 12.5GB），因 A 冻结减少了激活值存储。
    - **编译框架层**：论文未明确说明（使用标准 PyTorch 编译栈，无自定义编译 pass 或 kernel）。
    - **Kernel 调度层**：论文未明确说明（top-k 操作、稀疏 mask 乘法均使用 PyTorch 原生算子，未自定义 CUDA kernel）。
    - **硬件架构/芯片设计层**：论文未明确说明（使用 consumer GPU RTX 3090 / A100，无自定义 RTL 或硬件修改）。
  - 关键数据（Llama-3.1-8B）：FlyLoRA(k=8) 在 MMLU 上 40.88% vs LoRA(r=32) 38.93% vs Split-LoRA(4×8) 38.44%，且激活训练参数仅 0.13%（vs LoRA(r=32) 1.03%, Split-LoRA 0.33%）。训练时间：FlyLoRA MMLU 上 4.73h（8×RTX3090）vs LoRA(r=32) 5.09h。多任务合并后 MMLU 仅降 2.02% vs LoRA(r=8) 降 6.48%。

## FloE: On-the-Fly MoE Inference on Memory-constrained GPU

- baseline方法是什么？
  - **Naive MoE Offloading（DeepSpeed-MII ZeRO-Infinity）**：非 expert 权重常驻 VRAM，expert 权重全部 offload 到 DRAM。每 token 推理时，router 确定激活 expert → PCIe 传输对应 expert 权重（~300MB/expert，各投影矩阵 FP16 全量）→ GPU 执行完整 dense GEMV。瓶颈：MoE 推理从 memory-bound 转变为 I/O-bound，PCIe 4.0 单向带宽仅 ~32GB/s，而 VRAM 带宽 ~300GB/s+，单 expert 传输耗时 ~15ms vs 计算 ~5ms，3:1 传输-计算比。
  - **Advanced MoE Offloading（Mixtral-Offloading, Fiddler）**：增加 expert 预测器 + LRU cache + uniform INT3 量化。Mixtral-Offloading 用 intermediate results 预测下一层 expert 并预取+缓存，配合 uniform 量化压缩传输。Fiddler 将部分计算卸载到 CPU 以减少传输需求。痛点：(1) uniform quantization 对所有投影矩阵同等对待，gate/down 对量化敏感，INT2 时 perplexity 暴涨；(2) 预测-传输-计算串行执行，无法 pipeline；(3) 学习型稀疏预测器额外消耗 2.19~9GB VRAM。
  - 全栈执行例子（Mixtral-Offloading baseline）：token 嵌入 → GPU 计算 Attention → Router 选择 expert E_i → CPU 从 DRAM 加载 INT3 E_i 权重（gate/up/down 统一量化）→ PCIe 传输 → GPU dense GEMV（gate→SiLU, up→×, ×→down）→ 输出。传输与计算串行，吞吐受 I/O 限制。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - FloE 通过三个核心设计解决 baseline 缺陷：
    1. **Hybrid Compression（混合压缩）**：不再 uniform 量化，而是利用 expert 内部参数的差异化敏感性——gate/down 用 contextual sparsification（按 up projection 输出幅值剪枝对应通道，理论证明 L_down ≤ L_up < L_gate），仅保留 ~10% 通道；up projection 用 INT2 HQQ（因 up 对量化天然不敏感）。实现 9.3× 压缩比而无 uniform INT3/INT2 的精度崩溃。
    2. **Dual Sparsity Predictors（双稀疏预测器）**：利用相邻层 hidden state 相似度 >0.95 的观察，inter-expert predictor（MLP）预测下一层激活 expert，intra-expert predictor（复用 W_up 做矩阵乘）预计算稀疏掩码。两个预测器均用当前层 hidden state 预测下一层，打破串行依赖，实现传输-计算 pipeline。intra predictor 零额外参数，相比学习型 baseline（PowerInfer 9GB / SparseInfer 2.19GB）极大降低内存开销。
    3. **System Co-optimization（系统协同优化）**：Sparse GEMV kernel（Triton, 列主序转置+选择性列加载+SiLU 融合）消除稀疏带来的计算开销；Compact async transfer（AVX-512 SIMD+多线程+pinned memory+多 stream）将传输利用率从 ~7%（PyTorch 原生）提升至 88% PCIe 峰值。
  - 全栈执行例子（FloE 第 i 层推理）：
    - **算法层**：hidden state x_i → inter-expert predictor MLP(x_i) → 预测层 i+1 的激活 expert 索引 + intra-expert predictor(x_i, W_up_{i+1}) → 预计算稀疏掩码。
    - **系统框架层**：FloE scheduler 根据预测结果触发 compact async transfer，从 DRAM 预取层 i+1 的压缩 expert（gate/down 仅 ~10% 列/行转置 + up INT2）。
    - **编译框架层**：论文未明确说明（使用 PyTorch/Triton 标准编译栈，无自定义编译 pass）。
    - **Kernel 调度层**：Sparse GEMV kernel 在 GPU 执行——加载 x_i → W_up 全精度 GEMV → 阈值掩码 → 选择性加载 W_gate[mask] 列和 W_down^T[mask] 列 → 融合 SiLU+Hadamard+sparse GEMV → 输出。同时 AVX-512 多线程在 CPU 端打包下一层 expert 权重到 pinned memory，CUDA stream 异步传输。
    - **硬件架构/芯片设计层**：论文未明确说明（使用 consumer GPU + CPU + PCIe 标准硬件）。
  - 关键对比：baseline 中单 expert 传输 15ms + 计算 5ms（串行 20ms），FloE 中压缩后传输 ~1.6ms + 计算 ~3ms（pipeline 后有效延迟 ~3ms），实现 48.7× vs DeepSpeed-MII 的端到端加速。

## Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

- baseline方法是什么？
  - **Baseline 1: Offloading-based 方法（DeepSpeed-MII ZeRO-Infinity, Mixtral-Offloading）**：将全部或大部分 expert 权重存储在 CPU memory 中，推理时按需通过 PCIe 将所需 expert 权重从 CPU memory 拷贝到 GPU memory 后执行 GPU 计算。DeepSpeed-MII 使用 ZeRO-Infinity 将模型参数 offload 到 CPU 并动态加载；Mixtral-Offloading 使用 per-layer 的 offload_per_layer 参数控制每层多少 expert 放在 CPU。
  - **Baseline 2: CPU-based 方法（llama.cpp）**：将部分层放在 GPU 执行，其余放在 CPU 执行（通过 ngl 参数控制 GPU 层数）。CPU 部分直接在 CPU memory 中计算，避免 CPU↔GPU 数据传输开销，但不考虑 MoE 的 expert 级别稀疏性和 CPU/GPU 的不同 batching 效应。
  - **全栈执行例子（以 Mixtral-8x7B 16-bit, Environment 1 Quadro RTX 6000 24GB, single-batch decode 为例）**：
    - **模型推理算法层**：Mixtral-8x7B, 32 layers, 每层 8 experts (top-2 routing), 16-bit precision, >90GB 参数总量
    - **系统框架层**：
      - Offloading-based（DeepSpeed-MII）：每 token 每层 → gate 选 top-2 expert → 检查 expert 权重是否在 GPU → 若不在，CPU→GPU PCIe 传输 ~300MB/expert → GPU GEMM 计算 → 下一层。**瓶颈**：每次 expert cache miss 需要 PCIe 传 300MB 权重（2-5× GPU 计算时间），MoE 的 2/8 expert 稀疏性未被充分利用来减少传输。
      - CPU-based（llama.cpp ngl=8）：前 8 层 GPU 执行，后 24 层 CPU 执行。GPU 层直接在 GPU 端计算；CPU 层无需 PCIe 传输（权重已在 CPU RAM）。**瓶颈**：长 prefill (1024+ tokens) 时 CPU 计算延迟随 token 数线性增长，成为严重瓶颈；beam search 时多 beam 并行放大 CPU 计算量，性能崩溃。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel调度层**：GPU 端使用 PyTorch CUDA GEMM kernel；CPU 端使用 PyTorch 默认 CPU GEMM（无 AVX512_BF16 优化）。Offloading-based 方法的执行顺序：PCIe copy → GPU GEMM（串行，无法重叠）。llama.cpp 的执行顺序：CPU GEMM（串行，按层执行）。
    - **硬件架构层**：NVIDIA Quadro RTX 6000 24GB + Intel Xeon Gold 6126 48-core, PCIe Gen3 x16 32GB/s。
  - **Baseline 痛点**：
    1. **Offloading 的 PCIe 传输开销**（核心痛点）：每次 expert cache miss 需通过 PCIe 传输 ~300MB 的 expert 权重（3 个 4096×14336 矩阵），传输延迟是 GPU 计算的 2-5×。在 single-batch decode 场景（s=1），传输开销占主导，offloading 方法延迟显著高于纯 CPU 方法。
    2. **CPU-based 方法忽略 batching 效应**：llama.cpp 不区分 CPU 和 GPU 的不同 batching 行为——GPU 延迟近乎恒定（受限于内存带宽），CPU 延迟随输入量线性增长（受限于计算能力）。在长 prefill（s>512）和 beam search（多 beam 并行）场景，CPU 计算成为不可接受的瓶颈。
    3. **静态执行策略无适应性**：Offloading 方法总是 GPU 执行（传输+计算），CPU 方法总是按层静态分配 CPU/GPU。两者都不根据实际输入量 s 动态选择最优策略——对小 s 应避免 PCIe 传输用 CPU 计算，对大 s 应忍受传输换 GPU 加速。
    4. **Expert 放置不考虑访问频率**：llama.cpp 按层连续分配 GPU/CPU（前 ngl 层放 GPU），而非按 expert 热门度选择性地将热门 expert 放 GPU。这导致 GPU memory 中放置了冷门 expert，而热门 expert 反而在 CPU memory。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Fiddler 方法**：通过三类技术协同解决 baseline 的痛点：
    1. **动态 Per-Expert 执行策略（Algorithm 1）**（解决痛点 1、3）：每个 expert 在运行时根据输入 token 数 s 独立决定 CPU 还是 GPU 执行。决策基于 latency model——`cpu_lat(s) ∝ s`（CPU 线性增长），`gpu_lat(s) ≈ constant`（GPU 恒定），`trans_lat()`（PCIe 传输恒定）。当 `cpu_lat(s) < gpu_lat(s) + trans_lat()` 时选 CPU 执行（小 s，避免 PCIe 传输）；否则选 GPU 执行（大 s，忍受传输换加速）。关键洞察：对 small s（decode 阶段 s=1），CPU execution strategy (c) 的 activation 拷贝量（s×4096 floats）远小于 weight 拷贝量（3×4096×14336 floats），因此 PCIe 开销可以忽略。
    2. **Expert 热门度导向的 GPU 放置**（解决痛点 4）：离线用 calibration data（ShareGPT）profiling 各 expert 激活频率，initialization 时按热门度降序放置 expert 到 GPU（尽可能多，不超过显存）。相比随机放置提升 hit rate 约 3-5 个百分点（Env1: 25.2% vs 21.9%, Env2: 53.0% vs 48.8%）。
    3. **CPU AVX512_BF16 Expert Kernel**（强化策略 c）：利用 Intel AVX512_BF16 的 VDPBF16PS 指令（每周期 32 BF16 MAC）加速 CPU 端 expert FFN 计算，弥补 PyTorch 默认 CPU GEMM 无法利用 BF16 硬件加速的不足。
  - **全栈执行例子（Fiddler, 与 baseline 同配置）**：
    - **模型推理算法层**：与 baseline 相同（Mixtral-8x7B, top-2 routing, 16-bit），不修改模型架构或 router 逻辑。
    - **系统框架层**：基于 PyTorch 自建调度系统。初始化阶段：non-expert 层常驻 GPU → 按热门度排序填充 GPU expert → 其余 expert 放 CPU pinned memory → 校准 latency model 参数。执行阶段：每层 gate 输出后统计各 expert 输入量 s → Algorithm 1 决策 → 对每个 expert 独立执行 Strategy (a)/(b)/(c) → 聚合加权输出。对比 baseline 的关键差异——Offloading 方法缺少 strategy (c)，llama.cpp 缺少 strategy (b) 和动态决策。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + 自定义 C++ CPU kernel。
    - **kernel调度层**：
      - GPU 端：与 baseline 相同（PyTorch CUDA GEMM），但通过 latency model 避免不必要的小 batch GPU 执行+传输。
      - CPU 端：自定义 AVX512_BF16 kernel 替代 PyTorch 默认 CPU GEMM，tile 分块最小化 cache miss。
      - 数据传输：Strategy (b) 使用 cudaMemcpyAsync 异步 CPU→GPU 传权重；Strategy (c) 使用 cudaMemcpyAsync GPU→CPU 传 activation（可忽略 <1%）。
      - 调度时序（以 layer l 含热门/冷门 expert 混合为例）：
        ```
        Time →
        Expert 0 (GPU hit):     |== GPU GEMM ==|
        Expert 3 (GPU miss, s大): |== PCIe W copy ==|== GPU GEMM ==|
        Expert 5 (GPU miss, s小): |== PCIe A copy ==|== CPU AVX512 GEMM ==|== PCIe out copy ==|
        ```
    - **硬件架构层**：与 baseline 相同（Quadro RTX 6000 24GB + Xeon Gold 6126 / RTX 6000 Ada 48GB + Xeon Platinum 8480+）。
    
    **关键性能对比**：
    | Scenario | Fiddler vs Best Baseline | 核心获益来源 |
    |----------|--------------------------|-------------|
    | Single batch (avg) | 1.26× vs llama.cpp | 小 s 时 strategy (c) 避免 offloading 的 PCIe 传输 |
    | Long prefill TTFT (avg) | 1.30× vs DeepSpeed-MII | 大 s 时 strategy (b) 利用 GPU 并行能力 |
    | Beam search (avg) | 11.57× vs llama.cpp | 多 beam 放大 s，strategy (b) 避免 CPU 线性增长灾难 |
    | Phi-3.5-MoE | 6.5× vs DeepSpeed-MII | 验证方法跨模型通用性 |

    **核心设计洞察**：Fiddler 的本质是将 MoE 推理的"what to offload/where to compute"问题建模为一个极小开销的 per-expert latency-driven 决策问题。其 design philosophy 是"不预设 CPU 或 GPU 哪个更好，而是让运行时数据（输入量 s）驱动决策"。这使 Fiddler 能同时覆盖两种 baseline 各自擅长的场景（offloading 擅长大 batch prefill，CPU 擅长小 batch decode），实现 Pareto 改进而非 trade-off。MoE 的 expert 稀疏性（每 token 仅 2/8 expert 需要计算）是使此策略可行的关键前提——每个 expert 的输入量 s 通常在 0 到数百之间，跨度极大，正是这种方差使得动态决策有意义。

## FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

- baseline方法是什么？
  - **Baseline 方法**：
    1. **ZeRO Optimizer（数据并行 baseline）**：使用 DeepSpeed 的 ZeRO stage 3，将 optimizer states、gradients、parameters 按 tensor 维度切分到所有 worker。MoE 模型被复制到所有 worker 上。每 iteration：forward（本地 GPU 全模型计算）→ backward → all-reduce 同步梯度 → 参数更新。ZeRO stage 3 虽然能容纳大模型，但引入了大量通信开销（梯度同步和参数 gather/scatter），导致 DDL-Roofline 中 R_CC 极低，训练效率差。
    2. **FastMoE（expert parallelism baseline）**：使用 expert parallelism，expert 分布在不同 worker 上。每 iteration：非 MoE 层 = 数据并行（本地计算），MoE 层 = all-to-all-v 发送 tokens 到目标 expert worker → 各 worker 计算本地 expert → all-to-all-v 返回输出 → 重组序列。所有通信和计算操作按同步模式执行（先通信完再计算，或先计算完再通信），导致通信和计算硬件交替闲置。
    3. **GShard / BASE Layer（修改 expert selection 的 baseline）**：GShard 使用辅助 loss 做 load balancing + top-2 gate；BASE Layer 使用 matching 算法分配 tokens 到 experts。两者都修改了 expert 选择以平衡计算负载，但未考虑网络拓扑对通信性能的影响。
  - **全栈执行例子（以 MoE-GPT 3.42B, 16 experts, 16×V100, johnny 集群，FastMoE baseline 为例）**：
    - **训练算法层**：MoE-GPT 3.42B, H=2048, α=2, 12 层，每层 16 experts，每个 expert 为 2 层 FC（GeMM: (B_w, H)×(H, αH) 和 (B_w, αH)×(αH, H)）。Gate 使用 softmax top-2 选择 expert。每 token 激活 2 个 expert。
    - **系统框架层**：FastMoE（基于 PyTorch），每 worker 持有 1 个 expert（64 GPUs / 16 experts 时每个 expert 有多个副本或部分 workers idle）。非 MoE 层使用数据并行，MoE 层使用 expert parallelism。All-to-all 通过 NCCL/MPI 同步原语实现。
    - **编译框架层**：PyTorch eager mode，无编译框架修改。
    - **kernel调度层**：NVIDIA cuBLAS 执行 GeMM。All-to-all 由 NCCL group 调用完成。调度顺序：all-to-all（发送 tokens）→ 同步 barrier → GeMM FC1 → GeMM FC2 → all-to-all（返回输出）→ 同步 barrier。通信和计算严格串行。
    - **硬件架构层**：16× V100-PCIE，2 节点，节点内 PCIe switch 互连，节点间 Infiniband 50Gb/s。
  - **Baseline 痛点**：
    1. **动态负载不均衡（skewed expert selection）**：训练数据自然服从偏斜分布，热门 expert 接收远超平均的 tokens（观察到 4/16 experts 处理约 20% tokens，3.2× 平均值），导致其所在 worker 重载而其他 worker 空闲，且此模式随 training iteration 动态变化。
    2. **同步执行模式低效**：All-to-all 通信和 GeMM 计算严格串行执行，当通信进行时 GPU 计算单元闲置，反之亦然。在非均匀 token 分布下，通信和计算的不均衡进一步放大资源浪费。
    3. **网络拓扑与 expert 选择不匹配**：All-to-all 通信在树形拓扑的上层链路（跨节点）产生严重拥塞——跨节点流量 T_n = M(N-1)/N · BH ≈ M 倍于节点内流量 T_w = (MN-1)/MN · BH。现有方法（GShard, BASE Layer）仅均衡计算负载，未考虑网络拓扑对通信的影响。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法（FasterMoE）**：通过性能模型引导的三个系统性优化，分别解决上述三个痛点：
    1. **Dynamic Shadowing（解决痛点 1）**：在每 iteration 运行时，通过轻量级 Algorithm 1 判断是否将热门 expert 参数广播到所有 worker（影子化），使热门 expert 的 tokens 在各 worker 本地计算，消除热门 worker 的计算瓶颈。核心决策基于性能模型——当 token 传输开销大于模型传输开销（B_max > rαH），或减少的计算延迟大于增加的通信开销时，启用影子化。
    2. **Fine-grained Smart Scheduling（解决痛点 2）**：将 all-to-all 通信拆分为 n 个 group 的 pairwise exchange 操作序列 S/C/R，分别在独立的 communication stream 和 computation stream 上异步执行，打破同步 barrier。通过将最快的操作（同 group 的 S_{i,0} 和环通信的 R_{i,n-1}）放在首尾，最小化首尾通信开销对整体延迟的影响。使程序从 DDL-Roofline 的半理想曲线跃升至理想曲线附近。
    3. **Topology-aware Gate（解决痛点 3）**：修改 expert 选择策略，限制跨节点 token 数为 L = W_net/(M·W_local) · B，将超出限制的 token 重新分配给本地节点内的 expert。同时保留 best-fit 的 token-expert 对，减少对模型质量的影响。
  - **全栈执行例子（FasterMoE w/ all optimizations, MoE-GPT 3.42B, 16 experts, 16×V100, johnny 集群）**：
    - **训练算法层**：与 baseline 相同的 MoE-GPT 模型结构。差异：(a) 每 iteration 开始前执行 SelectShadowExperts 算法，在每 worker 上基于 token-to-expert 分配矩阵 T 判断影子化哪些 expert；(b) Gate 使用拓扑感知逻辑，限制跨节点 tokens 上限 L；(c) 被影子化的 expert 参数在 forward 开始时 broadcast 到所有 worker，backward 结束时 reduce 梯度并 update 在原 worker。
    - **系统框架层**：FasterMoE（基于 FastMoE 扩展），在 FastMoE 的 transformer.py 中实现动态影子化决策逻辑（fastermoe/fmoe/transformer.py:34），通过环境变量 FMOE_FUSE_GRAN 控制分组粒度。Communication stream 和 computation stream 分别为独立 CUDA stream。使用 grouped pairwise exchange 算法替代 coarse-grained all-to-all。
    - **编译框架层**：论文未明确说明（沿用 PyTorch eager mode）。
    - **kernel调度层**：不同：S/C/R 操作序列在 comm stream 和 comp stream 上交错执行（图 8b/c）。S_{i,0} 接收本地 group tokens 最快（无跨节点连接），排在首位；R_{i,n-1} 为环通信（充分利用带宽），排在末位。comp stream 从 C_{i,0} 到 C_{i,n-1} 连续执行。两个 stream 间的依赖：C_{i,j} 等 S_{i,j} 完成，R_{i,j} 等 C_{i,j} 完成。
    - **硬件架构层**：与 baseline 相同硬件。执行特征：(a) 跨节点通信量减少至 W_net/W_local · BH（拓扑感知门控）；(b) GPU SMs 在通信期间执行 GeMM 计算（stream overlap）；(c) 热门 expert 所在 worker 的计算负载转移到全部 worker 分担（影子化）。
    - **关键性能对比**：
      - johnny 集群：vs ZeRO stage 3 加速 6.63×，vs FastMoE 加速 2.20×（影子化 1.95× + 调度 1.40× 联合）。
      - trevor 集群（64 GPU）：vs ZeRO stage 3 加速 17.87×，vs FastMoE 加速 5.72×（影子化 4.74× + 调度 1.40× 联合）。
      - 收敛加速：vs GShard 1.37× 更快收敛，vs BASE Layer 2.19× 更快收敛（FasterMoE w/ topo. gate, MoE-GPT）。

## Faster MoE LLM Inference for Extremely Large Models

- baseline方法是什么？
  - **Baseline: 标准 fine-grained MoE 推理（DeepSeek-V2-Lite na=6, DeepSeek-V3 na=8）**：使用 sglang 等 serving 框架对 fine-grained MoE 模型进行标准推理，保持训练时的 expert 配置不变——所有 ne 个 expert 全部加载，每 token 激活全部 na 个 expert。MoE 层的执行与 FFN 类似，但因稀疏激活导致额外的 expert 参数加载开销，使得批次效应（batch effect）比同参数量 FFN 更弱。
  - 全栈执行例子（以 DeepSeek-V2-Lite na=6, ne=64, concurrency=512, 2×A800 为例）：
    - **模型推理算法层**：DeepSeek-V2-Lite，16B 参数，每层 64 个 routed expert（de=1408）+ 1 个 shared expert（ds=10944）。Router 使用 softmax Top-k（k=6）选择 6 个 expert，输出为 6 个 expert 的加权和 + shared expert。na=6 时激活中间维度 da=8448，共享 expert 占比 54.4%（ds/(ds+da)）。
    - **系统框架层**：sglang v0.4.4 post 1，EP=2（2×A800），continuous batching。每层执行：attention（MLA）→ MoE gate（softmax top-6）→ 6 个 expert FFN + 1 个 shared expert → all-reduce 聚合。低并发时 memory I/O bound（需加载 6 个 expert 参数），高并发时 compute bound。固定 1024 input + 1024 output tokens。
    - **编译框架层**：论文未明确说明。PyTorch + torch.compile (Section 4)，sglang 内置 CUDA kernel。
    - **kernel调度层**：论文未明确说明。sglang 使用 RadixAttention + fused MoE kernel，各 expert 作为独立 GEMM 执行。
    - **硬件架构层**：2× NVIDIA A800 80G PCI-e，EP=2，NVLink 160GB/s（单机多卡）。
  - **Baseline 痛点**：
    1. **MoE 弱化批次效应（核心痛点）**：MoE 虽降低了 FLOPS（仅激活 na/ne 的 expert），但多 token 间很少复用同一 expert，导致增加 token 数时需额外加载更多 expert 参数到显存。这使 MoE 的 arithmetic intensity 上升速度远慢于 FFN，在低/中并发时 memory I/O 瓶颈严重，peak efficiency 更难达到。
    2. **固定 na 浪费容量**：fine-grained MoE 的 na=6-8 远大于 coarse-grained MoE 的 na=2，但并非所有层/所有 token 都需要如此多的 expert。在低并发（memory I/O bound）和高并发（compute bound）场景下，固定 na 的最大值分别导致不必要的 expert 参数加载和不必要的计算开销。
    3. **全量 ne 资源浪费**：fine-grained MoE 经过 load-balancing 训练后 expert 重要性仍有巨大差异（Section 6.2 的奇偶索引实验表明偶数索引 expert 和后半部分 expert 更重要），大量低重要性 expert 占用了显存和参数加载带宽，但对模型质量贡献甚微。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：从效率分析和算法优化两个角度探索 fine-grained MoE 的推理优化空间，提出两种互为补充的优化策略：
    1. **Inference Time Expert Skipping（解决痛点 2）**：按层级别动态减少激活 expert 数 na，通过四元组 (b,h,e,p) 定义跨层的 expert 分配模式。核心洞察来自 Roofline 分析——低并发时减少 na 直接降低参数加载量（memory I/O 改善），高并发时减少 na 降低计算量（FLOPS 减少），中并发时因总量 token 足够多、减少 na 不显著减少总 expert 池而加速有限。策略层面：softmax 路由（V2）使用 descending 策略最优（早期层更多 expert 提供足够上下文，后期层特征已足够丰富可减少 expert）；sigmoid 路由（V3）使用 ascending 策略最优（sigmoid 的极化特性意味着早期需要更多 expert 探索，后期可集中到少数高权重 expert）。
    2. **Pre-Inference Expert Pruning（解决痛点 3）**：在推理前通过 calibration 数据统计各 expert 的激活频率（soft count），仅保留 top-ne' 个最活跃 expert。核心发现：fine-grained MoE 的全随机初始化使得 expert 间无结构相似性（不像 Mixtral 的 expert 从 dense checkpoint 初始化），因此 random 和 naive structured selection 完全失效。但 activation count 方法可有效识别关键 expert——仅去掉 25% expert（ne 64→48）即可获得显存节省，性能退化可控（Avg −2.7%）。
    3. **Expert Parallelism 通信优化分析（解决痛点 1 的结构性缺陷）**：尽管 MoE 单机效率不如 FFN，但在分布式部署中，EP 的通信量仅为 TP 的 na/(nd-1)（典型值 28%）。结合 fine-grained MoE 的 group-constrained routing（限制 token 仅从少数 EP group 选 expert），EP 可在跨节点 InfiniBand (50GB/s) 上实现与节点内 TP NVLink (160GB/s) 相当的延迟，从而通过跨节点 EP 补偿单节点 MoE 效率不足，允许每个节点承载更高 batch、达到更高 arithmetic intensity。

  - 全栈执行例子（与 baseline 同配置，expert skipping na=2 descending 策略 + V2-Lite）：
    - **模型推理算法层**：DeepSeek-V2-Lite，使用 descending expert skipping——首层 na=6 → 逐层递减 → 末层 na=2，平均 na≈3.3。Router 仍使用 softmax top-k，仅 k 值按层变化。相比 baseline na=6（da=8448），平均 da 降至 ≈4646（3.3×1408），shared expert 占比从 54.4% 升至 70.2%（ds/(ds+da)），意味着 baseline 中 45.6% 的 routed expert 计算被部分削减。
    - **系统框架层**：sglang 修改——在 MoE layer 初始化时按 (b,h,e,p) 四元组计算每层 na(l)，forward 时 router 动态使用对应的 top-k 值。执行流程与 baseline 一致，但每层的 expert 加载数量和 FFN 计算量减少。低并发时 benefit 来自更少的 expert 参数 I/O（na=2 时仅加载 2 个 expert + 1 shared vs baseline 的 6+1），高并发时 benefit 来自更少的 GEMM FLOPs。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明（沿用 sglang 内置 MoE kernel）。关键差异在于每个 MoE layer forward 的 expert GEMM 调用次数从 6 降到 2（含 shared 仍为 3 次），reduce 从 6-way 降为 2-way。
    - **硬件架构层**：与 baseline 相同。结果：concurrency=2 时 speedup 1.32×（memory I/O bound 场景，na 减少显著降低参数加载），concurrency=128 时 speedup 仅 1.10×（中并发，总 expert 池已由足够多 token 覆盖），concurrency=768 时 speedup 1.16×（compute bound 场景，na 减少降低 FLOPS）。

    关键性能对比（V2-Lite expert skipping）：
    - na 从 6→2：Avg benchmark 下降 7.5%（best strategy 仅 6%），但 throughput 提升 10-50%（取决于并发度）
    - na 平均 3.3：Avg benchmark 下降 <1%，零性能退化下的最佳吞吐量提升
    - V3 best strategy：throughput +10% 且零性能退化
    - Expert pruning ne 64→48：throughput up to 2.3× at low concurrency，但性能退化不可忽略

    **核心设计洞察**：这篇论文是一种"效率-性能 Pareto 探索"，而非提出单一优化方法。核心贡献在于：(1) 从 Roofline 模型出发系统性分析了 fine-grained MoE 在不同 batch size / concurrency 下的效率特征（记忆 I/O bound → compute bound 的过渡），(2) 量化了 expert skipping 和 pruning 在不同并发度下的效率收益与性能代价的 trade-off，揭示了两种方法在不同场景（低并发 vs 高并发）下的互补性，(3) 发现 sigmoid vs softmax routing 导致截然不同的 skipping 最优策略（ascending vs descending），表明 fine-grained MoE 的推理优化高度 model-dependent，不存在 universal approach。

## Fast Inference of Mixture-of-Experts Language Models with Offloading

- baseline方法是什么？
  - **Naive offloading（HuggingFace accelerate device_map="auto"）**：标准的 per-layer offloading 方案。每个 Transformer 层（含所有 expert）作为一个整体，在需要时从 host RAM 完整加载到 GPU，用完后卸载。对于 MoE 模型，这意味着每次计算一个 MoE 层时，需要将全部 8 个 expert（每个约 5.6B 参数在 FP16 下）加载到 GPU，但实际只使用 top-2。
  - 全栈执行例子（以 Mixtral-8x7B-Instruct 在 T4 16GB 上 batch=1 为例）：
    - **模型推理算法层**：Mixtral-8x7B，32 层，每层 attention + MoE（8 experts, top-2 routing）。Per-token autoregressive decode，batch=1。
    - **系统框架层**：HuggingFace accelerate 的默认 offloading（`device_map="auto"`）。执行顺序：① 加载 layer 0 全部参数到 GPU（含 attention weights + 8 expert weights）→ ② 运行 attention → ③ 运行 MoE gate → ④ 运行 top-2 expert FFN → ⑤ 卸载 layer 0 全部参数 → ⑥ 加载 layer 1 全部参数...。每层加载全部 8 个 expert，但仅使用其中 2 个，浪费 75% 的 PCIe 带宽。
    - **编译框架层**：论文未明确说明。PyTorch + HuggingFace accelerate，无编译框架修改。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行，per-layer load → compute → unload 循环。
    - **硬件架构层**：NVIDIA T4 (16GB VRAM)，host RAM ~13-16GB，PCIe Gen.3 8-16GB/s。
  - **Baseline 痛点**：
    1. **无效参数传输**（核心痛点）：Naive offloading 以层为单位加载，但 MoE 每层只需 2/8 expert。每次加载浪费 75% 的 host-to-device 带宽传输不需要的 expert。
    2. **无缓存复用**：相邻 token 常复用相同 expert（图 1 显示 expert 局部性），但 naive offloading 每层都从 host RAM 重新加载，无状态记忆。
    3. **无法预取专家**：MoE gate 在当前 layer 输出后才选择 expert，无法像 dense 模型那样预先加载下一层参数。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：三项针对 MoE 特性的 offloading 优化：
    1. **Expert LRU Cache**（解决痛点 1 和 2）：以 expert 为单位（而非层）进行 offloading，保持每层 k 个最近使用的 expert 在 GPU 上。仅加载 cache miss 的 expert，大幅减少 host-to-device 传输量。通过观测到 MoE 模型在相邻 token 间存在 expert 复用局部性（图 1 蓝色方块），LRU 策略简单但有效。
    2. **Speculative Expert Loading**（解决痛点 3）：利用 Transformer 残差连接的归纳偏置——当前层 hidden states 可作为下一层 hidden states 的合理近似。在对当前 token 计算的同时，将下一层 MoE gate 应用于当前 hidden states 预测下一层最可能使用的 1-2 个 expert，在独立的 CUDA stream 上后台异步预取。预测正确时可消除下一层的加载延迟；错误时仅浪费带宽但正确性不受影响。
    3. **混合量化（Mixed MoE Quantization）**：attention 层保持 4-bit（高质量），expert 层压缩到 2-3 bit。利用 expert 占总参数 96.6% 这一特性，显著缩小 offloading 传输量，同时保持 perplexity 可接受（2-bit experts 下 WikiText2 从 3.59→4.52）。

  - 全栈执行例子（与 baseline 同配置，Full algorithm + 2-bit experts）：
    - **模型推理算法层**：与 baseline 相同（Mixtral-8x7B-Instruct，top-2 routing），不改变模型架构、gate 逻辑或生成质量。
    - **系统框架层**：自建 offloading 系统替代 accelerate。执行顺序：① 加载 layer l 的 attention（常驻 GPU）→ ② MoE gate → ③ **检查 expert cache**，仅加载 cache miss 的 expert → ④ **异步启动投机预取**（对 layer l+1 预测所需 expert，后台 host-to-device copy）→ ⑤ 运行 top-2 expert FFN → ⑥ 进入 layer l+1 时投机预取结果可能已就绪。与 baseline 的关键区别：仅加载 0-2 个 expert/token/layer（而非 8 个），且下一层加载与当前层计算重叠。
    - **编译框架层**：论文未明确说明。纯 PyTorch 实现，使用 pin_memory + CUDA stream 异步拷贝。
    - **kernel调度层**：Contiguous pinned memory buffer 实现单次 host-to-device copy。b=4 个共享 device buffer 实现异步 expert 交换。CUDA stream 层面：计算 stream 执行 attention + expert FFN，拷贝 stream 执行投机预取。使用 `tensor.pin_memory()` + `cudaMemcpyAsync` 模式。
    - **硬件架构层**：与 baseline 相同。关键约束仍为 PCIe 带宽（8-16GB/s），瓶颈从"加载 8 个 expert"变为"加载 0-2 个 expert + 后台预取"。结果：T4 上从 0.66 tok/s（naive）提升到 2.09 tok/s（full algo），3.2× 加速。

    核心设计洞察：该工作是一种"特性感知 offloading"——不改变模型，而是利用 MoE 的两大固有特性（expert 局部性和残差连接的 gate 预测能力）来优化 offloading 调度。相比通用的 per-layer offloading，这种 MoE-aware 调度在 consumer hardware 上实现了可交互的推理速度（2-3 tok/s），使 47B MoE 模型能在免费 Colab 上运行。

## FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

- baseline方法是什么？
  - **Baseline 1: NCCL（通用集合通信库）**：PyTorch 等框架默认使用的多 GPU 通信库。在 MoE 数据 shuffling 中，NCCL 要求数据以 device-major layout 排列，而 MoE 模型要求 expert-major layout。这导致每次 all-to-all 通信前后都需要显式的数据重排（permute/repack），使用 `torch.index_select` 等算子（或等价 CUDA kernel）扫描并修改整个 token buffer。
  - **Baseline 2: DeepEP（SOTA MoE 通信库）**：基于 NVSHMEM 的 MoE 专用通信库，使用 warp specialization 和 IBGDA 实现高效的跨节点通信。但其 token deduplication 是局部和静态的，优化与特定硬件（InfiniBand、NVLink、IBGDA）紧密耦合。
  - 全栈执行例子（以 DeepSeek-V3 MoE、EP=64、H100、8-node cluster、seqlen=16k 为例）：
    - **模型推理算法层**：DeepSeek-V3 MoE，top-k=8 routing，256 experts，hidden_dim=7168。Router 为每个 token 选择 top-8 experts，产生 T×8 的 token-expert 分配矩阵。
    - **系统框架层**：Megatron-LM（训练）/ SGLang（推理）使用 NCCL 或 DeepEP 作为通信后端。MoE 层的典型执行流程为：① `index_select(tokens, rank_indices)` 按 destination rank 重排 → ② `all_to_all(permuted_tokens)` 跨设备交换 → ③ `index_select(received, expert_indices)` 按 expert layout 再次重排 → ④ expert FFN 计算 → ⑤-⑥ 对称的反向 permute + all-to-all。步骤①和③是 memory-bound 的 permutation 操作，每次扫描整个 token buffer。
    - **编译框架层**：论文未明确说明。NCCL 和 DeepEP 均直接编译为 CUDA kernel，无中间编译框架。
    - **kernel调度层**：NCCL 使用高度优化的 all-to-all collective kernel。DeepEP 使用 NVSHMEM 的 one-sided put/get 操作 + warp-specialized kernel 实现低延迟通信。但二者均**不感知**数据的 logical segment 结构和 routing 语义——NCCL 的 all-to-all 将数据视为无结构的字节流；DeepEP 的 deduplication 限于特定硬件特性。数据 layout transformation 作为独立的 kernel launch 在通信前后执行。
    - **硬件架构层**：8 节点 × 8×NVIDIA H100 80GB，节点内 NVLink（480 GB/s per GPU），节点间 10×400Gbps RoCE（约 50 GB/s）。
  - **Baseline 痛点**：
    1. **冗余数据复制与重排**（核心痛点）：通信前后各需一次 memory-bound 的 permute/repack。Profiling 显示 rearrangement 占 intra-node 总延迟的 68.8%，占 inter-node 总延迟的 25%。这是因为 all-to-all 要求 device-major layout，而模型需要 expert-major layout，每次通信都产生一对对称的逆排列。
    2. **冗余数据通信**：当同一 token 被路由到同一节点上的多个 expert 时，NCCL 会通过跨节点网络多次发送完全相同的 token payload。DeepEP 有一定 deduplication 但限于局部和静态优化。
    3. **通信负载不均衡**：token routing 的偏斜分布导致各 GPU 跨节点流量不均，产生网络热点和带宽利用不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FUSCO 方法**：基于 "fusing data transformation and communication" 原则，将数据重排嵌入通信操作内部，消除显式的通信前后 permute 步骤。三项核心设计：
    1. **Data-Fused Communication Engine (dComm)**（解决痛点 1）：引入 Segment Descriptor 抽象（{addr, size} 对数组），将 token 建模为 logical segments。发送端 GPU kernel 在 gather 数据到 NIC ring buffer 的过程中 inline 完成 expert-major→device-major 的 layout transformation；接收端 kernel 在 scatter 数据时直接写入 expert activation tensor 的最终位置。使用 pipelined 设计：GPU 准备 slice 与 NIC RDMA 传输完全重叠。
    2. **Communication Planner + Hierarchical Routing**（解决痛点 2）：构建两级 descriptor——Node-Level Forwarding（每个目的节点仅发送一份 token 拷贝给 forwarder GPU）和 Expert-Level Distribution（forwarder 经 intra-node NVLink 分发给各 expert GPU）。这利用节点内高带宽（480 GB/s）替代跨节点重传，显著减少跨节点流量。
    3. **Online Load Balancer**（解决痛点 3）：贪心算法——各节点内按跨节点负载降序排列 GPU，circular shift by node index，构成 communication groups（每组含每节点一个 GPU，组内互为 forwarding endpoints），使高负载 GPU 分散到不同组，利用独立物理通道（多 NIC）并行执行各组通信。

  - 全栈执行例子（与 baseline 同配置，EP=64，H100，8-node，seqlen=16k）：
    - **模型推理算法层**：与 baseline 相同（DeepSeek-V3 MoE，top-k=8，256 experts），不改变模型架构、router 逻辑或收敛性。
    - **系统框架层**：Megatron-LM（训练）和 SGLang（推理）通过约 500 行 Python 适配层替换原有 all-to-all 为 FUSCO。Communication Planner（约 1000 行 Python，使用 PyTorch GPU operators）基于 MoE router 输出构建两级 descriptor plan；dComm（约 2000 行 C++/CUDA）作为独立 collective primitive 执行 fused 通信。**流程从 5 步（permute→all-to-all→permute→compute→reverse）简化为 3 步（FUSCO dispatch→compute→FUSCO combine）**，消除所有通信前后的显式重排。
    - **编译框架层**：论文未明确说明。FUSCO 基于 NCCL transport 层，复用其设备注册和连接管理，无编译框架依赖。
    - **kernel调度层**：dComm 的 GPU kernel 采用 producer-consumer 模式。GPU producer kernel 根据 descriptor 从非连续内存 gather segments 到 contiguous ring buffer（inline layout transformation）。NIC consumer 通过 RDMA 从 ring buffer 发送 slice。由于 slice 的 RDMA 传输时间 > GPU gather 时间，GPU 操作完全被 NIC 掩盖。Intra-node 使用 GPUDirect P2P + inline descriptor 解释。对比 baseline 的 5 次 memory pass（index_select 读+写 ×2 + NCCL 内部拷贝），FUSCO 仅需 1 次 GPU memory pass（descriptor-driven gather to ring buffer）+ 1 次 NIC 传输（pipelined）。
    - **硬件架构层**：与 baseline 相同（8×H100 per node，NVLink + RoCE）。FUSCO 的关键硬件利用策略：① 利用节点内 NVLink 高带宽（480 GB/s vs 跨节点 50 GB/s）做 hierarchical routing 的 expert-level 分发；② 利用多 NIC（10×400Gbps per node）配合 Online Balancer 的 communication groups 实现并行跨节点传输。
    
    关键性能对比（16k seqlen，real-world traffic）：
    - FUSCO vs NCCL: 1.66× communication speedup，训练 1.17-1.39× speedup，推理 TTFT 1.09-1.25× speedup
    - FUSCO vs DeepEP: 1.38× communication speedup，训练 1.10-1.19× speedup，推理 TTFT 1.06-1.16× speedup
    - 消融：dComm 贡献约 27-33%，Planner（含 deduplication）贡献约 27-67%（single-node routed 下最高），Balancer 贡献约 3-17%

## FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

- baseline方法是什么？
  - **MoE-only token-level overlapping（Tutel）**：仅在 MoE 层内部进行 token-level 的 A2A 通信与 expert 计算的流水线重叠。具体来说，将输入 MoE 层的 token 序列按 token 数量均匀切分为微批次（micro-batches），在 GPU 的两个 stream 上分别执行 A2A 通信和 expert 计算，使不同微批次的通信和计算重叠。
  - 全栈执行例子（以 GPT-MoE-L、32K seqlen、16 GPU、EP=16 为例）：
    - **模型推理算法层**：GPT-MoE decoder-only Transformer，每个 block 包含 masked self-attention + top-1 GShard MoE（每隔一层替换 FFN）。Training 时按 autoregressive 方式计算 causal attention，MoE 层使用 top-1 gate routing tokens 到对应 expert。
    - **系统框架层**：Megatron-LM 训练框架，使用 DP=2（跨节点）+ TP/SP=8（节点内，仅 attention 层）+ EP=16（MoE 层）。Tutel 作为 MoE 训练插件，对 MoE 层进行 token-uniform 微批次划分（overlap degree d=2/4/8/16），在 NCCL A2A 通信 stream 和 CUDA expert computation stream 之间做 token-level overlapping。**但 attention 层的计算和 A2A 通信完全串行**。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：NCCL 2.21.5 提供 A2A dispatch/combine 的集合通信 kernel；FlashAttention 提供 fused attention kernel。Tutel 基于 PyTorch 的 CUDA stream 机制实现通信与计算的 overlap。
    - **硬件架构层**：AWS g5.48xlarge 节点，每节点 8 × NVIDIA A10G-24G GPU，100 Gbps 跨节点网络。
  - **Baseline 痛点**：随着序列长度增长（4K→32K），A2A 通信时间增长斜率大于 expert 计算时间（见 Figure 1a）。在 32K 序列下，expert computation 仅占总执行时间的 21%，A2A 通信成为主导瓶颈。MoE-only overlapping 受限于 MoE 层计算量太小（expert 是轻量 FFN），无法充分掩盖 A2A 通信延迟。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FOLDMOE 方法**：将 token-level overlapping 从 MoE 层扩展到整个 Transformer block，利用 attention 层在处理长序列时的 O(n²) 计算量掩盖 A2A 通信延迟。三项核心设计：
    1. **1A1M 调度**：交错 attention 计算与 expert 计算，消除 aAaM 调度中因阶段不平衡导致的流水线尾部气泡。
    2. **Token Buffer + 时间均匀微批次**：在 attention/MoE 层之间插入 buffer 解耦二者的微批次划分——attention 按时间均匀切片（后序微批次 token 数更少以补偿 causal attention 的计算不平衡），MoE 仍保持 token 均匀切片。
    3. **Quick-start 启发式切片算法**：基于 attention FLOPs 模型（FLOPs(l,c) = (4H+3h)lc + 8H²l）快速确定最优切片方案，最小化启动延迟并最大化饱和阶段重叠。

  - 全栈执行例子（与 baseline 同配置，d=8）：
    - **模型推理算法层**：与 baseline 相同（GPT-MoE、causal attention、top-1 GShard MoE），不改变模型架构或收敛特性（Figure 12 验证 loss curve 一致）。
    - **系统框架层**：基于 Megatron-LM 修改，将每个 Transformer block 重构为四级流水线：attention → A2A dispatch → expert → A2A combine。在 CUDA stream 层面实现：attention 计算 stream 和 A2A+expert stream 并行执行。Token buffer 在 attention 输出端维护，实现 FIFO 解耦逻辑。**注意力计算现在与 A2A 通信重叠**，而非 idle 等待。对比 baseline 的 MoE-only overlapping，FOLDMOE 在 32K seqlen 下将 A2A 关键路径延迟减少约 2x（Figure 11），forward pass 加速 1.94x（d=8），overall 加速 1.49x（vs Tutel）和 2.72x（vs Megatron-MoE）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FlashAttention 与 FOLDMOE 的 micro-batch causal attention 兼容（二者保持相同的 causal mask pattern）。NCCL A2A 通信与 CUDA kernel 在分离的 stream 上 overlap。Overlap degree d 需要 trade-off：d 增大减少流水线气泡但增加 kernel launch overhead（Figure 9 展示 d 从 1 到 16 的吞吐量曲线）。
    - **硬件架构层**：与 baseline 相同（AWS g5.48xlarge + A10G）。

## Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

- baseline方法是什么？
  - **CLIP (Vanilla)**：标准的 Vision-Language Model，使用 ViT 作为图像 encoder、Transformer 作为文本 encoder，通过对比学习对齐图像和文本 embedding。CLIP 的 encoder 对所有输入不加区分地通过 MLP 层处理，可能无意识地学习偏置信息。
  - **FairCLIP (SOTA fair VLM)**：基于 CLIP 架构，通过最小化不同受保护属性组分布之间的 Sinkhorn distance 来增强公平性。但 FairCLIP 保留了 CLIP 的原始架构，未针对公平性进行特定架构适配。
  - 全栈执行例子（以 FairCLIP/b16 在 Harvard-FairVLMed 青光眼诊断任务为例）：
    - **模型推理算法层**：CLIP 对比学习框架。图像经过 ViT-B/16 encoder（12 层 Transformer blocks），每层包含 multi-head self-attention + MLP。文本经过 Transformer encoder 对称处理。最终通过 cosine similarity 匹配图像-文本对。FairCLIP 在原 CLIP loss 基础上加入 Sinkhorn distance loss，最小化不同属性组（race/gender/ethnicity/language）embeddings 分布之间的距离。
    - **系统框架层**：PyTorch + HuggingFace Transformers，标准训练循环。论文未明确说明具体的训练框架细节。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行。
    - **硬件架构层**：NVIDIA GeForce RTX 3090 (24GB)。
  - **Baseline 痛点**：
    1. **CLIP 架构无偏置过滤能力**（核心痛点）：CLIP 的 encoder 不加区分地处理所有 patch embedding，模型可能从偏置的 patch 中学习到与受保护属性（如种族、性别）相关的 spurious correlation，而非真正的疾病特征。在 Harvard-FairVLMed 上，CLIP/b16 在 Race 属性上的 DPD=14.57、EOD=18.47（数值越大越不公平）。
    2. **FairCLIP 仅通过 loss 约束公平性**：FairCLIP 仅在损失函数层面通过 Sinkhorn distance 约束来缩小不同组分布之间的距离，但架构层面没有任何机制来过滤或抑制偏置特征的提取。这导致 FairCLIP/l14 在 Race 上 DPD=16.01（甚至比 CLIP 更差），说明单纯的距离最小化不足以保证公平性。
    3. **loss 设计仅关注分布距离**：现有 fairness loss（包括 FairCLIP 的 Sinkhorn distance）仅最小化不同属性组分布之间的距离，忽略了分布离散度（dispersion/variance）的作用。方差既是 MoE load balancing 的关键（影响训练稳定性和模型容量利用），也是 fairness 的重要度量（不同组的方差差异过大意味着某组内个体差异被系统性放大或压缩）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Fair-MoE 方法**：通过两个关键组件——FO-MoE（架构层面）和 FOL（损失函数层面）——同时解决上述三个痛点：
    1. **FO-MoE（Fairness-Oriented Mixture of Experts）**（解决痛点 1 和 2）：在图像和文本 encoder 中引入两级 MoE——
       - **Embedding-based MoE**：替换最后一个 attention block 的 MLP 层，使用 sparse gating + expert capacity 机制：`Ŵ^1 = Top_c(Top_r(W^1, k^1), α)`。`Top_c` 通过 capacity C 限制每个 expert 处理的 patch 数量，只有权重最高的 α = C(N+1)k/M 个 patch 被各 expert 处理，其余被清零。这使模型能够**主动过滤偏置 patch embedding**——偏置 patch（如包含肤色、性别特征信息的图像区域）分配到不相关的 expert 或权重过低而被过滤，仅保留与疾病诊断任务相关的公平特征。
       - **Feature-based MoE**：放置在 encoder 之后，对 [CLS] feature 做进一步 sparse gating，消除编码后的偏置特征，提取最终公平的特征向量供对比学习使用。
    2. **FOL（Fairness-Oriented Loss）**（解决痛点 3）：在 Sinkhorn distance（L_distance）基础上新增四个方差优化项——
       - `F_EI = Σ_{p∈P} Σ_{j=0}^{M^1-1} (Var(O_{N_j}) - Var(O_{N|p_j}))^2`（图像 embedding-based MoE）
       - 类似地定义 F_ET、F_FI、F_FT
       - 核心思想：对整个数据集的 gate weight 方差 Var(O_N) 和各属性组内 gate weight 方差 Var(O_{N|p}) 之间的差异进行惩罚，使不同组的 gate weight 分布不仅**位置接近**（距离最小化）而且**形状一致**（方差对齐）。这同时服务于 load balancing（MoE 训练稳定性）和 fairness（各组专家使用模式一致）。

  - 全栈执行例子（与 baseline 同配置，FairMoE/l14）：
    - **模型推理算法层**：基于 CLIP/ViT-L/14，在图像和文本 encoder 的最后一个 attention block 中用 embedding-based MoE（M^1 experts）替换 MLP，在 encoder 输出后增加 feature-based MoE（M^2 experts）。Gate 为可学习 MLP，sparse top-k 路由 + capacity filtering。对比学习 loss 加上 FOL = F_EI + F_ET + F_FI + F_FT + L_distance。**架构改变使模型能主动过滤偏置信息而非被动地仅通过 loss 约束**。
    - **系统框架层**：与 baseline 相同（PyTorch + HuggingFace Transformers），标准训练循环。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行。MoE 的 sparse gating 通过 mask 操作实现，无额外通信开销（单 GPU 训练）。
    - **硬件架构层**：与 baseline 相同（NVIDIA RTX 3090）。
    
    关键性能对比（FairMoE/l14 vs baselines，Harvard-FairVLMed）：
    - **Race**: ES-AUC 72.53（+5.00% vs FairCLIP/l14），AUC 73.93（+2.36%），DPD 2.63（↓83.6% vs FairCLIP/l14），EOD 4.25（↓75.1% vs FairCLIP/l14）
    - **Gender**: ES-AUC 69.97（+2.60% vs FairCLIP/l14），AUC 74.97（+4.17% vs FairCLIP/l14）
    - **Ethnicity**: ES-AUC 67.10（+2.87% vs FairCLIP/l14），DPD 8.79（↓42.8% vs FairCLIP/l14）
    - **Language**: ES-AUC 63.80（+0.23% vs FairCLIP/l14），AUC 71.37
    
    消融关键发现：
    - 移除 FOL → Race AUC 下降 2.56%，Gender ES-AUC 下降 2.34%，验证方差优化的必要性
    - 移除 embedding-based MoE → Race ES-AUC 从 70.9 降至 66.2，验证 patch 级偏置过滤的有效性
    - 移除 Image MoE → Language ES-AUC 从 66.1 降至 54.8（最大降幅），验证图像侧公平性更关键

## FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models

- baseline方法是什么？
  - **Baseline 1: Tutel (w/ PipeMoE)**：Tutel 是专用的 MoE 训练系统，PipeMoE 为其优化版本，通过自适应调度确定流水线度（pipeline degree），将 AlltoAll 通信与专家计算重叠。但存在以下局限：
    1. 仅支持有限的 routing function，对新路由机制不灵活。
    2. 前向和反向使用相同的流水线度，未考虑二者计算量差异。
    3. 仅重叠 AlltoAll 与 expert computation，不探索 ESP-AllGather/ESP-ReduceScatter（节点内通信）与 AlltoAll（节点间通信）之间的重叠。
    4. Gradient-AllReduce 仅与 non-MoE 部分重叠，未与 MoE 层协同设计。
  - **Baseline 2: DeepSpeed-MoE**：专用 MoE 训练系统，支持 EP 和 ESP，但调度能力更弱（手动配置或不进行自适应调度）。
  - 全栈执行例子（以 Mixtral-7B、Testbed-A、48 GPU、EP=6, ESP=8, MP=8 为例）：
    - **模型推理算法层**：Mixtral-7B decoder-only Transformer，MoE 层使用 top-2 GShard gate routing，8 experts，每个 expert 为 Mixtral FFN（SwiGLU）。前向时 tokens 通过 gate 分派到 top-2 experts，反向时计算 expert 权重梯度和输入梯度。
    - **系统框架层**：Tutel/DeepSpeed-MoE 基于 PyTorch 实现 DP+MP+EP+ESP 混合并行。Tutel 使用统一流水线度 r 切分输入 token，在 CUDA stream 上重叠 AlltoAll Dispatch/Combine 与 expert GEMM 计算。Gradient-AllReduce 在 MoE 层完成后执行，仅与 attention 等非 MoE 部分重叠。节点内 ESP-AllGather/ESP-ReduceScatter 与节点间 AlltoAll 串行执行，无重叠。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：NCCL 2.12 提供 AlltoAll/AllGather/ReduceScatter/AllReduce 集合通信 kernel。PyTorch torch.matmul 提供 GEMM kernel。Tutel/PipeMoE 通过 PyTorch CUDA stream 机制调度通信 kernel 与计算 kernel 的 overlap。
    - **硬件架构层**：Testbed-A: 6 节点 × 8×NVIDIA RTX A6000 (48GB)，节点内 NVLink 112.5GB/s，节点间 200Gb/s InfiniBand。N_MP=N_ESP=8（对齐节点内 GPU 数），N_EP=6（等于节点数）。
  - **Baseline 痛点**：
    1. **路由函数不灵活**：Tutel 和 DeepSpeed-MoE 仅支持有限的 routing function，新增路由机制需要大量侵入式修改。
    2. **节点内/节点间通信无重叠**：节点内 ESP-AllGather/ESP-ReduceScatter（NVLink）与节点间 AlltoAll（InfiniBand）完全串行，浪费了 NVLink 的高带宽（900GB/s vs 100GB/s InfiniBand on DGX H100）。
    3. **前向/反向统一流水线度不最优**：912/1458 配置下前反向最优度不同（反向计算量约为前向 2 倍），统一度导致性能次优。
    4. **Gradient-AllReduce 未与 MoE 层协同设计**：Gradient-AllReduce 和 AlltoAll 均为节点间通信，不重叠时 Gradient-AllReduce 成为额外延迟开销。现有方案（PipeMoE、Lina）要么仅与非 MoE 部分重叠，要么使用固定 chunk size 无法适应多变配置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FSMoE 方法**：通过三大技术解决 baseline 的四个痛点：
    1. **MoE 模块化与统一抽象**（解决痛点 1）：将 MoE 层分解为 Gate、Order、I-Order、Dispatch、Combine、Expert 六个独立子模块，每个模块可被替换/扩展。预实现 4 种路由函数（GShard、Sigmoid、X-MoE、Expert Choice）和 4 种 AlltoAll 算法（NCCL-A2A、1DH-A2A、2DH-A2A）。通过 Hook 机制（BeforeMoeStartHook, BeforeDispatchHook 等）实现非侵入式扩展。
    2. **节点内/节点间通信协同调度 + 前向/反向分别优化**（解决痛点 2 和 3）：在 MP/ESP 对齐节点内 GPU 数时，节点内通信（ESP-AllGather/ESP-ReduceScatter）和节点间通信（AlltoAll）可通过流水线重叠。将调度场景分为 4 种 Case（Case1: 节点间通信主导；Case2: 计算主导；Case3: AlltoAll 主导；Case4: 节点内通信主导），通过线性性能模型建模各操作耗时，SLSQP 求解器分别优化前向和反向的最优流水线度 r_fwd 和 r_bwd。
    3. **自适应梯度分区**（解决痛点 4）：两阶段算法将 Gradient-AllReduce 的梯度分配到各 MoE 层的 overlappable parts 中。Step 1 贪心将梯度分配到各层的空闲时间段；Step 2 用差分进化算法优化剩余梯度的跨层分配，实现 Gradient-AllReduce 与 MoE 层通信/计算的最大重叠。

  - 全栈执行例子（与 baseline 同配置，FSMoE pipeline degree r=4）：
    - **模型推理算法层**：与 baseline 相同（Mixtral-7B, top-2 gate），不改变模型结构或收敛性。额外支持 4 种路由函数的即插即用切换。
    - **系统框架层**：FSMoE 基于 PyTorch + C/C++/CUDA 扩展实现，替代 Tutel/DeepSpeed-MoE 的 MoE 层实现。在 DP+MP+EP+ESP 混合并行下，输入 token 按 r=4 切分，每个 chunk 依次经过：ESP-AllGather (intra-node) → AlltoAll Dispatch (inter-node) → ESP-ReduceScatter (intra-node) → Expert Compute → ESP-AllGather (intra-node) → AlltoAll Combine (inter-node) → ESP-ReduceScatter (intra-node)。**节点内通信与节点间通信在不同 chunk 上并行执行**（chunk i 的 ESP-AllGather 与 chunk i-1 的 AlltoAll 重叠）。反向额外将 Gradient-AllReduce 的梯度按 overlappable parts 自适应分配到各层，与最后一个 chunk 的 ESP-AllGather/ReduceScatter 和 expert 计算重叠。前向 r_fwd 和反向 r_bwd 独立优化。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：NCCL 2.12 集合通信 kernel（AlltoAll/AllGather/ReduceScatter/AllReduce）与 PyTorch GEMM kernel 在分离 CUDA stream 上调度。FSMoE 的在线 profiler 使用 nccl-tests 和 torch.matmul 微基准测量 α/β 参数（拟合 R² > 0.99），一次拟合 <10ms，SLSQP 求解平均 193ms。调度算法 O(1) 复杂度，训练前执行一次。
    - **硬件架构层**：与 baseline 相同（Testbed-A: 48×A6000）。对比 Tutel：1.18×–1.22× 加速（1458 配置层），对比 DeepSpeed-MoE：1.28×–3.01× 加速（真实模型）。在 Mixtral-7B 上，节点内/节点间通信重叠贡献约 5–6% 额外加速（FSMoE vs FSMoE-No-IIO），梯度分区贡献约 5–7%（FSMoE vs Tutel-Improved）。

## FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

- baseline方法是什么？
  - **Baseline**: 标准 MoE Transformer 架构，使用标准的残差连接（residual connections），即每个子块的输出立即通过残差加到主路径上，下一子块的计算必须等待当前子块的完整输出。在分布式执行中，这种"严格顺序依赖"导致以下阻塞通信模式：
    1. **训练中的阻塞 all-to-all**：在 EP 下，MoE 层的 Dispatch（token 发送到对应 expert rank）和 Combine（专家输出聚合回原 rank）均为 all-to-all 集合通信。Dispatch 必须在 attention 完成后、routed expert 计算前执行；Combine 必须在 routed expert 计算后、下一层计算前执行。两者均无法与计算重叠。
    2. **推理中的阻塞 all-reduce**：在 vLLM/SGLang 的 EP 推理实现中，各 rank 计算本地 expert 后通过 all-reduce 聚合结果。Attention 层的 TP 输出投影（RowParallelLinear）也包含 all-reduce。这些 all-reduce 均为同步阻塞调用。
  - 全栈执行例子（以 DeepSeek-V2-Lite 16B 在 Megatron-LM EP=8 训练，单节点 MI325X 8GPU 为例）：
    - **模型推理算法层**：标准 MoE Transformer 层，每层包含 Attention（MLA）+ MoE（64 experts, top-k routing, shared experts）。残差连接：`o_k = o_{k-1} + f_k(o_{k-1})`，即 `f_k` 必须等待 `o_{k-1}` 完全就绪。
    - **系统框架层**：Megatron-LM 训练框架。每层执行顺序：① Attention 子块（含 layer-norm）→ ② 可能的 TP 后 attention all-reduce → ③ layer-norm + gating + router → ④ **Dispatch all-to-all（阻塞）** → ⑤ routed expert 计算 + shared expert 计算 → ⑥ **Combine all-to-all（阻塞）** → 输出到下一层。Dispatch 和 Combine 期间 GPU 计算单元空闲（通信气泡）。
    - **编译框架层**：论文未明确说明。PyTorch + NCCL 通信后端。
    - **kernel调度层**：NCCL all-to-all collective kernel 在执行期间占用 GPU SM（通信计算单元），计算 kernel（expert GEMM）必须等待通信完成。无计算-通信重叠。
    - **硬件架构层**：AMD MI325X 8GPU 单节点，节点内高带宽互联。通信时间占 layer 总时间的显著部分。
  - **Baseline 痛点**：
    1. **阻塞通信导致 GPU 计算资源空闲**（核心痛点）：Dispatch 和 Combine all-to-all 期间，GPU 的 CUDA cores/Tensor cores 无法执行有用的计算，造成"通信气泡"（communication bubble）。随着硬件计算速度提升和 MoE 规模增大（更多 experts、更大 EP 度），通信时间在端到端延迟中的占比不断增大。
    2. **直接加载 FarSkip 架构权重导致性能崩溃**（Fig. 3）：如果不经训练直接将原始 checkpoint 加载到修改后的 FarSkip 连接架构中，模型性能随修改层数增加急剧下降——全部层修改后 MMLU 达到随机基线、HumanEval+ 为 0%。这是因为模型接收到的输入激活值与训练时的分布完全不同（OOD）。
    3. **SFT 微调不足以恢复性能**（Tab. 1-2）：仅用 SFT 数据微调 FarSkip 修改后的模型，在下游任务上显著劣于原始模型（DeepSeek-V2-Lite SFT 平均 55.0 vs 原始 64.5），尤其是在生成任务上（HumanEval+ 仅 11.0 vs 40.2）。SFT 缺乏足够的粒度信号来恢复原始模型的内部表征。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FarSkip-Collective 方法**：通过修改模型架构的连接性（residual connectivity）来消除子块之间的严格顺序依赖，使计算能够在通信进行期间继续执行。两项核心设计：
    1. **FarSkip-Collective 架构修改**（解决痛点 1）：将下一子块的输入从完整的最新激活值改为可用的"过时"或"部分"激活值——
       - 对于 Attention 子块输入（partial）：`attn-in_k = o_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1}`，省略 `routed-exp-out_{k-1}`。这使得 Combine all-to-all 可与 Attention 计算重叠（因为 Attention 输入不需要 routed expert 输出）。
       - 对于 MLP 子块输入（outdated）：`mlp-in_k = o_{k-1}`。这使得 Dispatch all-to-all 可与 Attention 计算重叠（因为 MLP 输入不需要最新 attention 输出）。
       - 数学上，`o_k = o_0 + f_1(o_0) + f_2(o_1^*) + ... + f_k(o_{k-1}^*)`，每个 `f_i` 仍贡献到残差路径，仅输入激活值不同。所有未来层 `f_j (j ≥ k+2)` 都能访问完整的 `f_k` 输出，保证信息最终不丢失。
    2. **FCSD（FarSkip-Collective Self-Distill）**（解决痛点 2 和 3）：以原始模型为 teacher，FarSkip 修改后的模型为 student，使用 KL 散度知识蒸馏进行训练。关键配方发现：
       - KL 散度 loss（而非 SFT cross-entropy）：teacher 的概率分布提供更细粒度的训练信号，帮助恢复原始模型的内部表征
       - 大 batch-size（2^17-2^18 tokens）和大 learning rate（2e-5−8e-5）：通过 short sweep 确定
       - 全参数训练（不冻结 embedding/LM-head）
       - MBPP+ early stopping：防止训练后期出现 mode collapse
       - 成本：约 100-1000× 低于从头预训练（< 10B tokens vs 数万亿 tokens）

  - 全栈执行例子（与 baseline 同配置，DeepSeek-V2-Lite FarSkip 在 Megatron-LM EP=8 训练）：
    - **模型推理算法层**：FarSkip-Collective 修改后的 MoE 架构，通过 FCSD 蒸馏恢复准确率。以 DeepSeek-V2-Lite 为例，FCSD 模型在 11 个 benchmark 上平均 62.0 vs 原始 64.5（−2.5），SFT baseline 仅 55.0（−9.5）。Llama-4-Scout 109B：FCSD 平均 75.1 vs 原始 76.0（−0.9）。
    - **系统框架层**：修改 Megatron-LM 执行顺序——
      - 前向：① MLA q/k/v 准备 → ② 同步上层的 Combine → ③ gating → ④ **异步 Dispatch**（async_op=True）→ ⑤ core attention + output projection（**与 Dispatch 重叠**）→ ⑥ 同步 Dispatch → ⑦ routed experts → ⑧ **异步 Combine**（async_op=True）→ ⑨ shared experts（**与 Combine 重叠**）
      - 反向：使用两项新技术——① **Stateful Async All-to-All Autograd Function**：在 stateful dictionary 中存储前向和反向通信 handles，通过 backward hook 在输入被访问前同步通信；② **Sequence Number Hijacking**：利用 PyTorch autograd 的 Sequence Number 内部机制，重新排序反向节点优先级——提高子块计算节点优先级，降低通向通信输入的节点优先级，使计算在通信等待期间优先执行。
    - **编译框架层**：论文未明确说明。所有修改在 PyTorch API 层面完成，不涉及编译器修改。
    - **kernel调度层**：基于 PyTorch 的 CUDA Stream 机制和 torch.dist async_op 实现通信-计算重叠。训练中使用两个 CUDA queue：计算 queue（expert GEMM, attention kernel）和通信 queue（NCCL all-to-all）。通过 async_op=True 启动通信后立即返回 handle，计算 queue 继续执行，仅在需要通信结果时调用 handle.wait() 同步。反向通过 backward hook 在 autograd 图中注入同步点。**设计原则：避免 low-level kernel/Triton 修改，保持在 PyTorch API 层面，确保硬件无关性和广泛适用性**。
    - **硬件架构层**：与 baseline 相同（MI325X 8GPU）。单节点训练结果（Tab. 3）：
      - DeepSeek-V2 Lite: 前向重叠率 87.6%, 反向重叠率 89.0%, 总重叠率 88.4%, 端到端加速 1.11×
      - DeepSeek-V3 (L=6): 前向重叠率 92.9%, 反向重叠率 84.1%, 总重叠率 88.9%, 端到端加速 1.04×
      - 多节点强扩展（Fig. 5）：EP=32 时端到端训练加速达 1.22×
    - **推理侧**：
      - Llama-4-Scout (109B) vLLM: all-reduce 重叠率 95.3%, TTFT 加速 12.2%-18.5%
      - DeepSeek-V2 (235B) vLLM: all-reduce 重叠率 97.6%, TTFT 加速 8.2%-16.8%
      - DeepSeek-V3 (671B) SGLang: TTFT 加速 up to 1.34×（TP=8, EP=8）
      - 多节点 decode (TP=16, EP=16, BS=1024): 稳定且一致的 TBT 加速（Fig. 7）

    **核心设计洞察**：FarSkip-Collective 的独特之处在于它是一种"算法-系统协同设计"——不是简单地优化系统实现来隐藏通信（bit-exact 方法），而是在模型架构层面主动消除导致阻塞通信的依赖关系，然后用轻量级知识蒸馏恢复准确率，最后在框架层面实现显式的通信-计算重叠。这种方法比纯系统优化（如 operator decomposition）能覆盖更多的重叠窗口（仅 routed experts + gating 不可重叠），且不依赖特定的硬件特性或 low-level kernel 修改。

## FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

- baseline方法是什么？
  - **Baseline 1: FedProx**：联邦优化算法，在本地更新时引入正则化项 $\frac{\mu}{2} \|w - w^t\|^2$ 来约束本地模型不偏离全局模型太远，缓解数据异构带来的 client-drift。每轮所有客户端收到相同的全局 dense 模型（Switch Transformers, 8 experts/layer），本地微调后上传，服务器 FedAvg 聚合。缺点是所有客户端共享相同参数，无法针对不同任务/数据分布定制模型；正则化参数 μ 敏感且难以推广到跨任务复杂场景。
  - **Baseline 2: SCAFFOLD**：使用控制变量（control variates）$c$ 和 $c_k$ 来修正本地更新方向 $w_k \leftarrow w_k - \eta(g_k - c_k + c)$，克服异构数据导致的 client-drift。每轮除模型参数外还需传输控制变量，额外通信和内存开销随训练累积。同样使用 dense 模型所有客户端共享参数，缺乏个性化能力。
  - **Baseline 3: randomMoE**：从全局 MoE（32 experts/layer）中为每个客户端随机选择 expert 子集构建个性化边缘模型，保证一定程度的信息隔离。但由于 expert 选择是随机的，可能选到对客户端任务无关或次优的 expert，无法利用 MoE 的稀疏激活特性来针对性适配数据分布。
  - **全栈执行例子（以 FedProx 在 Standard-Hetero-T 设置、Switch Transformers 8 experts、30 客户端为例）**：
    - **模型推理算法层**：Switch Transformers dense 模型，每层 8 experts，top-1 routing。所有 30 个客户端共享相同的模型架构和参数。每轮随机选 5 个客户端，各客户端在本地 AG News/SQuAD/XSum 数据上执行: forward（gate 选 top-1 expert）→ compute loss（cross-entropy + load balance loss + proximal term $\frac{\mu}{2}\|w-w^t\|^2$）→ backward → 上传模型 → 服务器 FedAvg 聚合。所有客户端用统一模型处理分类、阅读理解、摘要三种不同任务，不同任务的梯度更新方向可能冲突。
    - **系统框架层**：基于 PyTorch + HuggingFace Transformers 实现 FL 模拟。服务器-客户端通信模式：broadcast 全局模型 → 客户端本地训练 → 上传更新 → FedAvg 聚合。每轮传输完整模型参数（约 24.7GB 内存 + 2.30GB 通信量）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel，每个 MoE 层 gate softmax top-1 selection + 选中 expert GEMM 计算。
    - **硬件架构层**：模拟边缘设备 18-24GB 内存（高端智能手机/边缘计算平台级别），云服务器执行聚合。18-24GB 内存限制下全局模型只能用 8 experts/layer，无法充分发挥 MoE 大容量优势。
  - **Baseline 痛点**：
    1. **统一模型无法适应异构任务**（核心痛点）：FedProx/SCAFFOLD 等传统 FL 方法让所有客户端共享相同模型参数，但不同客户端有不同的数据分布和任务类型（分类/阅读理解/摘要），统一模型要么牺牲个性化性能，要么不同任务的梯度方向相互冲突导致收敛缓慢或不稳定。
    2. **资源受限与模型容量的矛盾**：边缘设备内存仅 18-24GB，限制全局模型只能使用 8 experts/layer（FedProx/SCAFFOLD），而 MoE 的优势在于大量 expert 提供丰富的知识库。设备能力限制了模型容量上限。同时 FedProx 的 proximal term 和 SCAFFOLD 的控制变量引入额外内存和通信开销（FedProx: 24.71GB 内存/2.30GB 通信，SCAFFOLD: 17.29GB 内存/4.61GB 通信）。
    3. **randomMoE 的盲目性**：随机选择 expert 构建子模型虽然保证了个性化（不同客户端不同 expert），但无法利用数据特性选择最优 expert，导致性能次优（Standard-Hetero-T 设置下 randomMoE TC 仅 91.63, TS 仅 14.51）。
    4. **缺乏结构-性能协同优化**：现有 PFL 方法要么固定模型结构只用 loss 约束（FedProx/SCAFFOLD），要么静态剪枝/蒸馏后不再调整（knowledge distillation 类方法），无法在训练过程中根据实际反馈动态调整模型结构。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FedMoE 方法**：以 MoE 架构天然的解耦 expert 参数空间为基础，构建两阶段个性化 FL 框架：
    1. **Stage One: Coarse-grained Submodel Initialization**（解决痛点 3——randomMoE 盲目性）：通过短轮次 PEFT + LoRA 收集各客户端 expert 激活模式，以激活概率 $p_{i,j}=n_{i,j}/N$ 衡量 expert 对特定数据的重要性。基于此执行启发式二分搜索——在每层保留 expert 组合概率 ≥ θ 的约束下，寻找满足内存限制的最大 θ，构建"高性价比"初始子模型。这替代了 randomMoE 的随机选择，确保每个客户端获得与其数据分布最相关的 expert 子集。
    2. **Stage Two: Modular Aggregation**（解决痛点 1——统一模型无法个性化 + 痛点 4——缺乏结构-性能协同）：突破 FedAvg 的"一刀切"聚合——dense 层保持 FedAvg；sparse 层按 expert 粒度的共享情况分别处理（未激活不变、单客户端直接更新、多客户端加权聚合）。这使得相关客户端在共享 expert 上协作学习，不相关客户端互不干扰，实现"知识共享 + 负迁移隔离"。
    3. **Expert Recommendation**（解决痛点 4——缺乏结构动态调整）：利用其他客户端作为"全局视角"的参考——基于 expert 激活概率的 cosine similarity 找到 top-K 最相似客户端，通过加权平均估算所有 expert（包括子模型外的）的预期激活概率。若参考组平均 expert 数更多则推荐引入高效 expert；否则推荐裁剪低效 expert。调整具有探索性（性能不改善则回退），在训练过程中持续优化个性化结构。
    4. **资源效率**（解决痛点 2）：通过子模型 sub-sampling 使每个客户端仅持有最优 expert 子集（平均 65-78 experts/layer 从 32 中选出），大幅降低内存（13.44GB vs FedProx 24.71GB，−45.6%）和通信量（1.76GB vs FedProx 2.30GB，−23.5%）。Stage One 的一次性开销约 7.46GB 通信和 13.06GB 内存且仅持续数轮。

  - **全栈执行例子（FedMoE 在 Standard-Hetero-T 设置、Switch Transformers 32 experts global/子模型平均 65 experts、30 客户端）**：
    - **模型推理算法层**：全局模型 Switch Transformers 32 experts/layer，top-1 routing。两阶段流程——Stage 1: 5 轮 PEFT+LoRA 微调，收集各客户端 expert 激活概率 → 二分搜索构建初始子模型（每层子集 experts 满足概率阈值且不超内存）；Stage 2: 联邦训练，每轮 subsample 子模型 → 本地训练（cross-entropy + load balance loss，无 proximal term）→ 上传（仅上传子模型所含参数而非全量）→ Modular Aggregation（expert 粒度差异化更新）→ Expert Recommendation（相似客户端参考调整结构）。关键差异 vs baseline：不同客户端持有不同 expert 子集（个性化参数空间），expert 选择由激活数据驱动而非随机，聚合策略按 expert 粒度差异化。
    - **系统框架层**：基于 PyTorch + HuggingFace Transformers 的自建 FL 模拟框架。与 baseline 的关键差异：服务器维护 client-expert map（每客户端每层保留哪些 expert），subsample 逻辑在服务器端执行（从 32 experts 中按 map 提取参数），上传/下发仅涉及子模型参数。Stage 1 增加一轮激活概率收集通信（one-time 约 7.46GB）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行。关键差异在每个 MoE 层 gate 的 top-1 选择范围从全量 32 experts 缩小为子模型内的 experts 数量，单个 token 的 expert GEMM 计算不变但总参数量因加载更少 expert 而降低。
    - **硬件架构层**：与 baseline 相同（边缘 18-24GB，云端聚合服务器）。结果：Standard-Hetero-T 下 TC 94.76（+2.0% vs best baseline），TS 16.92（+16.7% vs best baseline 的 14.51），通信 1.76GB（−23.5%），内存 13.44GB（−45.6%），收敛加速 1.35×–2.92×。

    **关键性能对比**：
    - Standard-Hetero-T: TC 94.76 (FedProx 92.92), RC 86.64 (FedProx 87.99), TS 16.92 (randomMoE 14.51)
    - Standard-Hetero-TD: TC 88.44 (FedProx 85.09), TS 16.63 (randomMoE 13.51)
    - Enforced-Hetero-T: TC 94.85 (FedProx 92.51), TS high (baselines significantly lower)
    - 消融: w/o stage1 → TS 降至 14.50 (vs 16.92), expert 数不减反增 (96→104); w/o stage2 → expert 数不变 (78), 丧失动态优化能力

    **核心设计洞察**：FedMoE 的核心创新在于利用 MoE 架构的"expert 并行 + 稀疏激活"天然属性来实现个性化 FL——不再强迫所有客户端共享相同参数，而是让每个客户端从全局 expert 池中"挑选"最相关的 expert 构建个性化子模型。两阶段设计（先粗后细）巧妙平衡了搜索效率（Stage 1 快速收敛到近优解）和优化精度（Stage 2 动态调整），Modular Aggregation 在 expert 粒度实现"合作但不干扰"，Expert Recommendation 利用群体智慧指导个体结构调整。这是一种"数据驱动的结构个性化"思路，区别于传统 loss 约束或静态剪枝方法。

## Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

- baseline方法是什么？
  - **单模态方法**：仅使用某一 modality 做分类/诊断，如 3D CNN [17] 处理 MRI、VGG19 [43] + transfer learning 处理 2D MRI slices、ResNet-18 [45] 处理 fMRI、DLG (ResNet-34) [36] 处理 Genetic SNP、deep learning-assisted spectroscopy [29] 处理 Biospecimen。这些方法完全忽略多模态互补信息。
  - **多模态方法（仅处理全模态交集）**：如 Tensor Fusion Network (TF) [74] 使用 tensor fusion layer 融合多模态 embedding；MulT [57] 使用 cross-modal attention 捕获跨模态交互；MAG [52] 将多模态特征映射为 adaptation vector；LIMoE [44] 通过 contrastive learning+entropy regularization 处理多模态 MoE；FuseMoE [19] 直接通过 MoE 融合多模态数据。这些方法假设所有 modality 都可用（只取交集训练），对 missing modality 场景缺乏设计——FuseMoE 在少模态组合下甚至比全模态更低（FuseMoE 3-modality ACC=59.52 反而高于 full-modality 但依然不如 Flex-MoE）。
  - **多模态 Missing Modality 方法**：ShaSpec [60] 使用 spectral attention 增强跨模态特征；mmFormer [76] 使用 transformer-based attention fusion。这些方法虽声称处理 missing modalities，但未考虑 observed modality combination 与 missing modality 之间的关系。
  - **全栈执行例子（以 ADNI full modality "IGCB" 预测为例的 baseline FuseMoE）**：
    - **模型推理算法层**：FuseMoE 接收 4 个 modality 完整样本 → modality-specific encoders 编码 → MoE layer 做 sparse routing（所有 expert 对任意输入 token 均可被激活，无 modality combination-specific specialization）→ MLP head 输出 AD 分类。对缺失模态采用 zero-padding → encoder 训练被合成零值干扰，批次内不同 modality combination 导致 encoder 接收低质量输入。无 missing modality bank，无课程学习排序，无 expert 的 generalization→specialization 两步训练。
    - **系统框架层**：PyTorch 原生实现，batch-wise training，无特殊 serving/scheduling 优化。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel，MoE gate routing + expert FFN 为标准 GEMM。
    - **硬件架构层**：NVIDIA A100 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Flex-MoE 方法**：核心是让 SMoE 架构显式建模 modality combination，分三阶段解决 baseline 缺陷：
    1. **Missing Modality Bank (解决 zero-padding/imputation 缺陷)**：构建 learnable embedding bank `B ∈ R^(2^|M|-1 × |M| × d)`，每个条目对应一种观测模态组合下的缺失模态 embedding。编码器仅用 observed 样本训练（不含合成数据），缺失部分按当前样本的观测模态组合索引查找 bank → encoder 训练质量不受 zero-padding 影响，bank 自动学习"缺少某模态时应该补充什么信息"。实验中 embedding bank 从 cosine similarity 验证了"共享更多观测模态的组合有更相似缺失 embedding"（full "ICBG" 与 "ICB" 相似度 0.56，与 "IC" 仅 0.46）。
    2. **Expert Generalization → Specialization (解决无 modality combination awareness 缺陷)**：训练分两阶段——(a) Warm-up 阶段：样本按可用模态数降序排列（课程学习），先用全模态样本 + G-Router + load/importance balancing loss 训练所有 expert 的通用知识；(b) Specialization 阶段：S-Router 通过 cross-entropy loss `L_ce = -Σ MC(x_j) log(max(S_Router(x_j)))` 将 top-1 gate 强制绑定到当前样本的 modality combination expert index，其余 top-(k-1) expert 做 load/importance balancing → 每个 expert 既保有全模态的通用知识，又获得自身模态组合的专有知识。实验验证：expert BCG 激活最多的两个输入 tokens 是 BCGI (通用) 和 BCG (专有)，expert BCI 激活最多的是 BCGI 和 BCI。
    3. **课程学习排序 (解决训练不稳定性)**：按 modality 数量降序排列样本先易后难 → 先 generalize 后 specialize，优于随机排列和升序排列（消融实验验证：降序 ACC=66.11, 随机=62.65, 升序=63.87）。

  - **全栈执行例子（Flex-MoE 在 ADNI 的 "IGC" (3-modality, missing B) 预测）**：
    - **模型推理算法层**：样本 i 有 {I, G, C} 三个模态输入 → (1) 各 encoder 仅用对应 modality 的 observed 数据训练，I 经 3D-CNN 得 e_i^I，G 经 ResNet-34 得 e_i^G，C 经 MLP 得 e_i^C；(2) 缺失 B 从 missing modality bank B[mc_idx=1 "IGC"][B] 取 embedding；(3) concat 得 h_i ∈ R^(4×128) → Transformer + SMoE layer。S-Router 计算 gate_logits，top-1 通过 L_ce 绑定到 expert_1 (MC index "IGC")，剩余 top-3 按 load balancing 选择 → y_i = gate_1·f_1(h_i) + gate_a·f_a(h_i) + gate_b·f_b(h_i) + gate_c·f_c(h_i)；(4) MLP head → 输出 Dementia/CN/MCI 分类概率。
    - **系统框架层**：PyTorch 实现，使用 batch_size=8 训练，50 epochs (5 warm-up + 45 specialization)，ADNI 16 experts/4 attention heads/128 hidden dim/MIMIC-IV 32 experts/3 attention heads/128 hidden dim。与 baseline 关键差异：encoder 训练与 modality completion 解耦（不需要 imputation），SMoE layer 的 routing 受 modality combination 监督信号约束。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel。与 baseline 的关键差异：GFLOPs 约 59.07（vs FuseMoE 59.76-59.74），因 SMoE 稀疏激活仅计算 top-k expert 的前向路径，参数量 36.9M（vs FuseMoE 340.9M）约减少 89%。
    - **硬件架构层**：NVIDIA A100 GPU。

  - **关键性能对比**（ADNI full modality "IGCB"）：
    - ACC: Flex-MoE 66.11 ±1.14 vs best baseline MAG 61.44 ±1.16 (+7.6%), vs FuseMoE 59.52 ±1.00 (+11.1%)
    - Macro-F1: Flex-MoE 64.73 ±2.01 vs best baseline MAG 61.38 ±1.32
    - AUC: Flex-MoE 81.67 ±0.54 vs best baseline mmFormer 73.93 ±5.97
    - Mean time/iter (IGCB): Flex-MoE 16.00s vs FuseMoE 20.71s (−22.7%)
    - # Params (IGCB): Flex-MoE 36.9M vs FuseMoE 340.9M (−89.2%)
    - 消融实验 (ACC): Flex-MoE 66.11, w/o Expert Specialization 62.75, w/o ES+EG 62.49, w/o embedding bank 63.87, w/o sorting (random) 62.65, w/o sorting (ascending) 63.87

## FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training

- baseline方法是什么？
  - **Baseline: Tutel / ScheMoE（仅 MoE 层内流水线）**：现有 MoE 分布式训练框架（Tutel、ScheMoE、FasterMoE、FSMoE）仅对 MoE 层内部的 expert 计算和 all-to-all 通信做 token-level 流水线重叠。具体来说，将输入 MoE 层的 token 序列按 token 数量均匀切分为微批次，在分离的 CUDA stream 上分别执行 A2A 通信和 expert GEMM 计算，使不同微批次的通信和计算重叠。
  - 全栈执行例子（以 Tutel baseline、LLaMA2-MoE、16 × RTX 3090、EP=16 为例）：
    - **训练算法层**：LLaMA2-MoE decoder-only Transformer，每 block 包含 MHA + gating + top-k MoE（k=1，expert=GPUs），标准 cross-entropy loss + load balancing loss。前向：MHA → gate → A2A dispatch → expert FFN → A2A combine → 下一层。反向：流程逆向。
    - **系统框架层**：PyTorch + Tutel（MoE 加速库，集成 A2A 异步通信）。Tutel 调度方案——输入 tensor 按 R=2 切分 → chunk_0 的 A2A dispatch 与 chunk_1 的 A2A dispatch 在不同 stream 上，chunk_0 的 expert 计算与 chunk_1 的 A2A dispatch 重叠。**但 MHA、gating、all-reduce 全部串行执行**，不在流水线内。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 通信后端。
    - **kernel调度层**：NCCL all-to-all collective kernel + PyTorch CUDA GEMM kernel。执行顺序——Forward: [MHA 计算 (串行)] → [Gate (串行)] → [A2A dispatch 与 expert GEMM 重叠 (R=2)] → [A2A combine] → [下一层]。Backward: [expert grad GEMM 与 A2A reverse 重叠] → [MHA grad (串行)] → [All-reduce grad (串行, 跨所有层)]。MHA 和 All-reduce 为独立串行阶段。
    - **硬件架构层**：2 节点 × 8 × RTX 3090 (24GB)，100Gb/s 跨节点网络。
  - **Baseline 痛点**：
    1. **MHA 和 Gating 被忽略（核心痛点 1）**：现有方法仅在 MoE 层内做流水线，MHA 计算和 gating 完全串行。论文 profiling 显示 MHA+gating 占单次迭代时间的 **29.8%-36.1%**（GPT2-Tiny-MoE: 29.8%, BERT-Large-MoE: 35.7%, LLaMA2-MoE: 34.2%, DeepSeek-V2-S: 36.1%），这意味着约 1/3 的迭代时间里 GPU 计算单元仅在执行 MHA 和 gating，A2A 通信链路闲置。
    2. **All-Reduce 通信串行（核心痛点 2）**：反向传播结束后，梯度 all-reduce 在所有层的 backward 完成后集中执行，与任何计算均不重叠。在较大模型上，all-reduce 通信时间占比不可忽略，却完全没有被流水线覆盖。
    3. **异构通信任务未协同调度（核心痛点 3）**：A2A 通信（all-to-all）和 All-Reduce 通信（all-reduce）是两种不同类型的集合通信，数据量和通信模式不同。现有方法将它们视为独立的串行阶段，未探索两者间的协同——all-reduce 可以利用 A2A 通信的间隙执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FlowMoE 方法**：通过三个递进式设计，将流水线调度从 MoE 层内扩展到整个 Transformer block，覆盖所有计算和通信任务：
    1. **Unified Pipeline Scheduling（解决痛点 1）**：将整个 Transformer block 的输入 tensor 等分为 R 份，**全部任务**（MHA、gating、expert 计算、A2A dispatch/combine）按层流动的方式统一编排。前向顺序：AT_1→AT_2→...→AT_R→E_1→...→E_R（计算任务）与 D_1→...→D_R→C_1→...→C_R（A2A 任务）交错执行。这使得 MHA 计算也与 A2A 通信并行，将先前 ~30-40% 的串行开销转化为计算-通信重叠。Pipe-AT 消融实验贡献 **10.3% 加速**（vs Tutel）。
    2. **Tensor Chunk-Based Priority Scheduling（解决痛点 2 和 3）**：在反向传播中，将每层 all-reduce 的梯度张量切成大小为 S_p 的 chunk，放入通信任务池。**A2A 任务优先级高于 all-reduce chunk**——当 A2A 任务 pending 时优先执行 A2A，仅当 A2A 闲置时 AR chunk 立即填充间隙。Theorem 1 证明此策略可减少反向传播时间。更小的 S_p 提供更细粒度的 gap filling（Theorem 2），但需平衡系统开销。Pipe-AR 消融实验贡献 **24.6% 加速**。
    3. **Bayesian Optimization Auto-Tuning（完善痛点 2/3 的实用化）**：AR chunk 大小 S_p 对性能影响显著（过大则无法充分填充间隙，过小则系统开销增大），且最优值依赖硬件环境（GPU 型号、网络带宽、模型配置）。使用轻量级 BO 自动搜索 S_p——约 8 次采样收敛，开销 < 1% 迭代时间。BO 单独贡献 **8.3% 加速**（vs 固定 S_p=1MB）。

  - 全栈执行例子（FlowMoE、LLaMA2-MoE、16 × RTX 3090、EP=16，R=2）：
    - **训练算法层**：与 baseline 相同的 LLaMA2-MoE 模型结构和训练算法（cross-entropy + load balancing loss），不改变模型架构、gate 逻辑或收敛性。
    - **系统框架层**：基于 PyTorch + Tutel，新增三个队列（DataQueue、A2AQueue、ARQueue）和一个后台通信池管理器。执行流程——**前向**：AT_1 → AT_2（MHA+gating 流水线与下一层交叠）→ E_1 → E_2（expert 计算）→ D_1 → D_2 → C_1 → C_2（A2A 通信）；**反向**：E_2' → E_1' → AT_2' → AT_1'（计算反向）→ C_2' → [AR_chunk if idle] → C_1' → [AR_chunk if idle] → D_2' → [AR_chunk if idle] → D_1'（A2A+AR 混合调度）。与 baseline 的关键差异——MHA 计算不再串行于 A2A 之前，all-reduce 不再集中执行而是切碎填充通信间隙。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 通信后端。
    - **kernel调度层**：
      - 前向：两个 CUDA stream——计算 stream（MHA GEMM + expert GEMM）和通信 stream（A2A dispatch/combine），MHA chunk_i 的计算与 A2A chunk_{i-1} 的通信重叠。
      - 反向：通信池管理器在主线程外运行，维护优先级队列。当 A2AQueue 非空 → 执行 A2A 通信；当 A2AQueue 为空且 ARQueue 非空 → 执行一个 AR chunk。计算 stream 执行 MHA grad 和 expert grad 的 GEMM。
      - 执行时序图：
        ```
        Time →
        Comp Stream: |== MHA_grad chunk_1 ==|== MHA_grad chunk_2 ==|== Expert_grad ==|
        Comm Stream: |== Combine_A2A ==|== AR_c1 ==|== Dispatch_A2A ==|== AR_c2 ==|
                     // AR chunks fill gaps between A2A tasks
        ```
    - **硬件架构层**：与 baseline 相同（2 节点 × 8 × RTX 3090, 100Gb/s 网络）。结果：LLaMA2-MoE 上 FlowMoE 1124.0ms vs ScheMoE 1374.3ms（1.22× 加速），vs Tutel 1534.1ms（1.36× 加速），vs vanillaEP 1987.7ms（1.77× 加速）。DeepSeek-V2-S 上 FlowMoE 3205.3ms vs ScheMoE 4093.7ms（1.28× 加速），vs FasterMoE 4562.5ms（1.42× 加速），vs vanillaEP 5843.3ms（1.82× 加速）。

  - **关键消融贡献分解**（M=8192, H=8192, 16 GPU）：
    | 优化组件 | 累积速度提升 | 边际贡献 | 解决痛点 |
    |---------|------------|---------|---------|
    | Tutel (Pipe-MoE) | 1.46× | — | — (baseline) |
    | + Pipe-AT (MHA+gating 纳入流水线) | 1.61× | +10.3% | 痛点 1 |
    | + Pipe-AR w/o BO (all-reduce chunk 填充) | 1.68× (w/o AT) | +15.1% | 痛点 2/3 |
    | + BO (自动调优 S_p) | 1.82× (w/o AT) | +8.3% | 痛点 2/3 |
    | Full FlowMoE (AT+AR+BO) | **2.05×** | +12.8% (over AR) | 全部 |

  - **核心设计洞察**：FlowMoE 的本质洞察是——MoE 训练中的"被忽略任务"（MHA、gating、all-reduce）虽单个占比不高，但合计占 30-40% 迭代时间，且执行模式有天然的流水线友好性（MHA 在前向初期、all-reduce 在反向末期）。通过将流水线边界从 MoE 层扩展到整个 Transformer block，并将 all-reduce 切碎后以低优先级填充 A2A 通信间隙，FlowMoE 实现了"无死角"的计算-通信重叠。BO 的引入解决了"最优 S_p 依赖硬件环境"的实用化难题，使得方法在无需手动调参的情况下即可部署到不同集群。

## GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

- baseline方法是什么？
  - **Baseline 1: Flat Global All-to-All + Uniform Expert Grouping（Occult, C2R, Tutel）**：现有分布式 SMoE 推理系统使用 flat global All-to-All 通信模式，所有 ranks 在同一通信组内严格同步。Expert 分组采用 uniform grouping（每组 expert 数相等）以简化负载均衡，如 Occult 的 No-Prune 变体。路由使用标准 top-k softmax gating，不做 topology-aware 优化。
  - 全栈执行例子（以 OLMoE 6.92B, 2 nodes×2 GPUs, Occult baseline 为例）：
    - **模型推理算法层**：OLMoE, 64 experts/layer, top-8 routing。每 token 由 gate 选择 8 个 expert。Occult 使用 uniform expert grouping——64 experts 均匀分配到 4 GPUs（每 GPU 16 experts），通过 collaboration-constrained routing 减少跨设备通信。
    - **系统框架层**：Megablocks 实现，flat global All-to-All 通信。执行顺序：① Gate 计算 → ② All-to-All Dispatch（global sync, 4 ranks 全参与）→ ③ Expert FFN (block-sparse matmul) → ④ All-to-All Combine（global sync）→ ⑤ 残差加和。跨节点通信走 25 Gbps Ethernet，节点内 NVLink 50 GB/s。All-to-All 占 MoE 层执行时间的 70%+，端到端延迟的 ~40%。
    - **编译框架层**：PyTorch 2.5 + Triton 3.1，无自定义编译 pass。
    - **kernel调度层**：NCCL All-to-All collective kernel + Megablocks block-sparse matmul。由于 flat global All-to-All 需要 strict synchronization，受 heterogeneous 链路中最慢的一侧限制（cross-node 25 Gbps vs intra-node NVLink），straggler effect 放大同步开销。
    - **硬件架构层**：2 nodes × 2 GPUs A100-SXM4 (80GB)，节点内 NVLink 12 links/GPU @ 50 GB/s，节点间 25 Gbps Ethernet。
  - **Baseline 痛点**：
    1. **通信-负载均衡 trade-off（核心痛点）**：现有方法将通信优化和负载均衡作为独立问题处理。Uniform grouping (Occult) 强制等大分组，破坏了 expert 间的自然共激活模式（affinity），限制了通信压缩的上限。Non-uniform grouping 虽能更好利用 affinity 减少通信，但 concentrate co-activated experts → 加重负载倾斜 → GPU 空闲和 straggler。这一 trade-off 在多节点场景下被跨节点低带宽（25 Gbps）放大，成为多节点 SMoE 推理的核心瓶颈。
    2. **Flat All-to-All Synchronization Overhead**：Flat global All-to-All 要求所有 ranks 严格同步。在 heterogeneous 链路共存的环境（NVLink + Ethernet）中，快链路被迫等待慢链路（straggler effect），global synchronization 将同步开销放大为可扩展性瓶颈。
    3. **跨节点重复传输**：当同一 token 被路由到同一 node 上的多个 expert 时，flat All-to-All 会通过跨节点链路发送多次相同的 token payload，浪费宝贵的跨节点带宽。
    4. **Replication 缺乏系统化设计**：现有 expert replication 方法（如 training 中的 FlexMoE, Lazarus）要么未针对 inference，要么使用 fixed replication 无自适应能力——复制太少不足以缓解倾斜，复制太多退化为 data parallelism 且破坏 affinity grouping 的通信收益。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **GRACE-MoE 方法**：一套 lossless 的 offline-online 协同优化框架，通过四个紧密耦合的设计解决上述全部痛点：
    1. **Non-Uniform Hierarchical Expert Grouping（解决痛点 1 的通信侧）**：基于 spectral clustering 对 expert affinity matrix 做分层分组——跨节点层面 fully non-uniform grouping 最大化 intra-node affinity（cross-node 带宽最贵）；节点内 controlled non-uniform grouping（r 控制 group size deviation）保留 affinity 同时限制 group size 差异。将原本 trade-off 中的"通信 vs 负载均衡"转化为"通信最大化优化 + replication 补偿负载"的协同路径。
    2. **Dynamic Expert Replication based on Load Skew（解决痛点 1 的负载侧 + 痛点 4）**：不固定复制数，定义 load skew factor ρ = Wmax/W，n_replica = min(max(1, floor(ρ)), n_gpu-1) 逐层自适应决定。仅复制 heaviest group 中的 hottest experts，避免 full group replication 的冗余和 affinity 破坏。Replicas 作为 secondary copies 放置到最空闲 GPU，保持 primary 在原有 grouping 中——通信收益 intact，负载重分布。
    3. **Topology-Aware Routing with Locality Preference（解决痛点 1 的在线协同）**：三级 locality-first 策略——优先同 GPU、其次同节点、最后跨节点。每级内 WRR with load prediction 平衡负载。牺牲部分负载均衡换取大幅通信节省，在通信主导瓶颈的大规模推理中实现更优 trade-off。
    4. **Hierarchical Sparse Communication (HSC)（解决痛点 2+3）**：physical global but logical sparse 两阶段通信——Stage 1 跨节点 token forwarding（同一 dest node 多 token 聚合单次发送，零填充 + global group implicit barrier 做 soft sync）；Stage 2 节点内 NVLink redistribution。Cross-node 通信与 intra-node routing computation fine-grained pipelining。消除 flat All-to-All 的 explicit global barrier 和 duplicate cross-node transmission。

  - 全栈执行例子（GRACE-MoE full, OLMoE, 2 nodes×2 GPUs）：
    - **模型推理算法层**：与 baseline 相同的 OLMoE 模型结构（MoE 层不变），差异在 expert placement 和 routing：
      - Offline: profiling → affinity matrix → hierarchical grouping（跨节点 fully non-uniform: 64→2 groups; 节点内 controlled non-uniform r=0.15: 每组→2 GPU groups）→ dynamic replication（每层计算 ρ, hot expert replicas distributed to underutilized GPUs）。
      - Online: Gate 输出 top-8 expert indices → topology-aware routing（locality-first replica selection + WRR load prediction）→ HSC 两阶段通信。
    - **系统框架层**：Megablocks + GRACE-MoE 修改。执行流程：
      ① Gate → ② Topology-Aware Routing（选择 replica）→ ③ HSC Stage 1: 跨节点 token forwarding（global group, sparse P2P, token dedup）→ ④ HSC Stage 2: 节点内 NVLink redistribution（overlapped with ③ 的 routing computation）→ ⑤ Expert FFN → ⑥ HSC Combine（反向对称）。与 baseline 关键差异——(a) routing 不是直接 dispatch 到 primary expert 而是 locality-aware replica selection；(b) 通信从 flat global All-to-All 变为 hierarchical sparse communication；(c) offline grouping + replication 确保通信最小化 + 负载均衡。
    - **编译框架层**：论文未明确说明。PyTorch 2.5 + Triton 3.1。
    - **kernel调度层**：HSC 使用 global collective group + zero-padded sparse transfer 替代 NCCL All-to-All。Cross-node 通信与 intra-node routing decision computation 在不同 CUDA stream 上 pipelined。Expert computation 使用 Megablocks block-sparse matmul kernel。
    - **硬件架构层**：与 baseline 相同。关键硬件利用——借助节点内 NVLink 高带宽（50GB/s×12 vs 跨节点 25Gbps）做 hierarchical routing 的 intra-node redistribution；HSC 的 implicit barrier 避免 explicit synchronization 的 straggler effect。

  - 关键性能对比：
    - End-to-end latency reduction: up to 78.55% (OLMoE), 73.17% (DeepSeek), 77.64% (Qwen3)
    - Speedup: up to 4.66× (OLMoE), 3.73× (DeepSeek), 4.47× (Qwen3) vs existing systems
    - MoE layer time reduction: up to 80.11%, 75.45%, 78.59%
    - Component contributions (vs Occult): HSC −35.19% All-to-All time; HG+HSC −48.33% All-to-All, −50.67% cross-node traffic; +DR+TAR +50.57% All-to-All, −52.11% cross-node traffic, −25.66% GPU idle
    - Cross-dataset transfer: worst-case +4.52% latency vs in-domain, still ≥12.06% lower than Occult

  - **核心设计洞察**：GRACE-MoE 的核心贡献在于识别并系统性地解决了 SMoE 分布式推理中"通信 vs 负载均衡"的根本性 trade-off——这一 trade-off 在前人工作中被各自独立处理（C2R/Occult 优化通信但加重负载倾斜，expert replication 方法平衡负载但增加通信）。GRACE-MoE 的解决方案不是 trade-off 中的某一个折中点，而是将 trade-off 分解为"grouping 做通信最大化优化 → replication 做负载补偿 → routing 做在线协同"的三段式协同优化，辅以 HSC 消除多节点同步瓶颈。这种"offline 做结构优化 + online 做路由决策"的分离架构具有强泛化性——offline placement 可跨 dataset 复用（最差 ≤4.52% 退化），使方法具备实际部署价值。

## GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

- baseline方法是什么？
  - **Standard MoE with Auxiliary Balance Loss（Switch Transformer / GShard 类）**：稀疏 MoE 架构使用 top-k gating 进行 token 到 expert 的路由，配合 auxiliary load balancing loss（LBL）和/或 z-loss 来鼓励 token 在所有 expert 上的均匀分配。全栈执行例子（推理单 token）：Token → Embedding → Transformer Layer（Attention + MoE）→ Gate softmax(W_g·x) 输出 [N] logits → top-k selection → k 个 expert FFN 计算 → weighted sum → residual add。训练时额外计算 L_aux = α·Σ f_i·p_i（f_i 为 token 分配比例，p_i 为平均 gating 概率）。Gate 参数 W_g∈R^{N×d} 通过反向传播和 LBL 共同优化。Baseline 只关注 token 分配的统计均衡，不区分 expert 的功能相似性——即使两个 expert 的 gating weight 高度相似（S_{ij}≈1），只要 token 数量均衡，它们仍可被同时激活，产生冗余计算。
  - Baseline 痛点：
    1. **Expert selection diversity 被忽视（核心痛点）**：辅助平衡损失仅保证 token 在各 expert 间的均匀分布，但无法阻止功能相似的 expert 被同时选中。Gating weight vectors w_{g,i} 和 w_{g,j} 高度相似的 expert 对学习到相似的激活模式，它们的 co-activation 产生功能冗余——两个 expert 做相似的计算却在 loss 中被视为等价的资源利用。这降低了模型的有效容量（effective capacity），特别是在深层（deep layers）中 expert specialization 至关重要。
    2. **早期训练的 expert 激活延迟**：在 pretrain 早期（前 100-1000 steps），gating 机制倾向于将 tokens 集中于少数几个 dominant expert，造成大量 expert 长时间处于零激活状态（zero token count）。这导致这些 expert 在关键的基础学习阶段（foundational learning）严重欠训练，限制了模型从训练初期就充分利用全部容量的能力。论文观察到 Layer 14 的零激活 expert 从 128 降至 20 需要 3000+ steps（baseline），而 GatePro 仅需 1500 steps。
    3. **深层 expert specialization 更加困难**：深层 MoE 层需要学习更复杂和抽象的表示，expert 之间的功能边界难以建立。深层 expert 的零激活收敛时间远长于浅层——baseline 下深层 expert 需要 4000+ steps 才能达到 near-zero unused，GatePro 可将此缩短至 2000 steps——这表明 baseline 的 expert specialization 在深层面临更大的瓶颈。
    4. **负载均衡与多样性非协同优化**：LBL 关注的是 token 数量均衡，GatePro 关注的是 expert 功能去冗余。二者是正交但互补的目标——LBL 保证资源利用效率，GatePro 保证资源利用质量。实验证明 GatePro 与 LBL 结合后收敛最快（Layer 7 仅 1000 steps 降至 20 unused vs GatePro alone 1500 steps, baseline alone 2500 steps），验证二者互补而非冗余。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **GatePro 方法**：一种无参数、可 hot-swappable 的 MoE gating 优化方法，通过局部竞争机制直接提升 expert 选择的多样性。核心包含两个组件：(1) **Gate Similarity Computation**：周期性计算 gating weight matrix W_g 的 cosine similarity matrix S∈R^{N×N}，识别功能冗余的 expert 对；(2) **Localized Competition Mechanism**：对每个 expert i，找到最相似的 expert j*(i)，在 token 级根据 logit 大小决定 competition winner，对 loser 施加固定惩罚 λ=10^{-4}，防止相似 expert 被同时激活。
  - 解决 baseline 缺陷的对应机制：
    1. **Localized competition 解决 expert diversity 问题（痛点 1）**：GatePro 不引入全局约束（如 LBL 的 token 分布均匀），而是实行 targeted local competition——仅对最高相似度的 expert 对施加竞争。这确保功能冗余的 expert 被差异化选择，每个 token 获得更多样化的 expert 组合。验证指标：GatePro 的 average cosine similarity 持续低于 baseline（Layer 8/16 均显著降低），average angle 更高（expert 之间更多正交性），spectral entropy 更高（激活分布更均匀，无少数 expert 主导）。
    2. **Competitive propagation 加速早期 expert 激活（痛点 2）**：通过在竞争关系中 loser expert 被抑制，不同 token 自然将 logit "重定向"到不同的 expert 组，加速了 expert 的初始激活。论文 expert utilization analysis (Figure 4) 显示 GatePro 在所有层都表现出更陡的零激活下降曲线——Layer 7 从 128→20 unused experts 仅需 1500 steps (vs baseline 2500 steps), 配合 LBL 后仅需 1000 steps。
    3. **深度感知的多样性增强（痛点 3）**：GatePro 的 cosine similarity 计算对所有层独立执行，深层中 expert 的 gating weight 差异更大（S_{ij} 值更低），竞争机制自然地更活跃——深层 expert 获得更强的 differentiation 信号。这解释了 GatePro 在深层的加速优势更显著（Layer 14 从 128→20 仅 1500 steps vs baseline 3000 steps）。
    4. **与 LBL 的互补性（痛点 4）**：GatePro 的竞争惩罚不改变 token 的 total count 分布（logit 抑制不影响跨 token 的 load balance），因此与 LBL 正交。实验证实 GatePro w/o LBL 已优于 baseline w/ LBL（Layer 7: 1500 steps → 20 unused vs 2000 steps），GatePro + LBL 效果最佳（1000 steps），验证"diversity + balance > balance alone"。
  - 全栈执行例子（GatePro MoE, Seed-MoE-0.7B/7B, 128 experts, top-k=6, 推理单 token）：
    - **训练/推理算法层**：Token x → gating projection W_g·x → logits (128) → GatePro competition: 对每个 expert i，比较 logits[i] 与 logits[j*(i)]，loser 减 λ=10^{-4} → suppressed logits (128) → top-6 selection → softmax renormalize → 6 expert FFN 前向（各自 Linear[d→αd]→GeLU→Linear[αd→d]）→ weighted sum → residual add → output。对比 baseline：GatePro 仅在 softmax 前增加了 O(N) 的 per-expert 比较和条件惩罚，无额外参数。Cosine similarity 矩阵 S 可每隔若干 steps 更新一次（无需每 token 计算），计算开销 O(N²d) 相对于 expert FFN 的 O(k·d·αd) 可忽略。
    - **系统框架层**：PyTorch + FSDP (Zhao et al. 2023) 分布式训练，8 节点 64 GPUs。Flash Attention (Dao et al. 2022) 优化 attention 计算。GatePro 以 hook/插件形式注入 MoE 层的 gating 计算中——在 top-k 选择前加入 competition penalty 逻辑。支持 hot-swappable 模式：通过 training flag 控制 penalty_mask 是否生效，切换无需模型参数修改或 re-compilation。
    - **编译框架层**：论文未明确说明（标准 PyTorch eager 或 torch.compile，无自定义编译 pass）。
    - **Kernel 调度层**：论文未明确说明。标准 PyTorch CUDA GEMM kernel。GatePro 的额外计算（条件惩罚 + 日志比较 128 次）为 O(N) 标量操作，在 GPU 上 kernel launch overhead 可忽略。
    - **硬件架构/芯片设计层**：论文未明确说明。使用 64 GPUs（推断为 NVIDIA H800/A100 级别），无自定义 RTL 或硬件修改。
  - 关键实验数据：
    - Seed-MoE-0.7B/7B, 500B tokens: MMLU-Pro 21.8% (vs baseline 20.5%), GSM8K 45.0% (vs 43.0%), BBH +0.8%
    - Seed-MoE-1.3B/13B, 1.2T tokens: MMLU-Pro 31.6% (vs 30.6%), BBH 50.7% (vs 49.8%), GSM8K 65.5% (vs 64.7%)
    - CT stage (0.7B/7B): Overall 52.55% (vs 51.92%), GSM8K +1.9pp, MBPP +0.8pp
    - CT stage (1.3B/13B): Overall 64.88% (vs 63.95%), GSM8K +2.0pp, MBPP +1.9pp
    - OLMoE-1B/7B, 400B tokens: Overall 62.5% (vs 61.8%), ARC-Challenge +1.1pp
    - Hot-swappable: 400B GatePro → 100B MoE: MMLU-Pro 30.0% (vs Full 500B GatePro 30.1%), BBH 44.5% (vs Full 44.2%), 验证 training legacy effect
    - 256 experts: GatePro 在深层 (Layer 21/28) 的加速优势更加显著，验证有效 scaling 到更大 expert pool
  - **核心设计洞察**：GatePro 的核心洞察是将 MoE 的 expert selection 问题从"负载均衡"（load balancing）重新定义为"功能多样性"（functional diversity）问题。现有的 auxiliary balance loss 方法本质是一种统计干预——它们改变 token 分配的概率分布但不关心 expert 是否在功能上相似。GatePro 通过"竞争传播"（competitive propagation）机制引入了一种结构干预——gating weight 的 cosine similarity 直接反映了 expert 在功能空间的相对位置，localized competition 则确保相邻（相似）的 expert 不会同时被激活，从而将 expert 的选择空间从"均衡的冗余组合"推向"多样的互补组合"。这种方法的优雅之处在于：(1) 不需要额外参数（parameter-free），因为 gating weight 本身就已经编码了 expert 的功能信息；(2) hot-swappable，因为 competition 是在 logit 空间操作而非权重空间，对训练动力学的影响是平滑的；(3) 与 LBL 互补而非替代，因为 diversity 和 balance 分别建模 expert 选择的"质量"和"数量"维度。

## HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

- baseline方法是什么？
  - **Tensor Parallelism (TP)、Expert Parallelism (EP)、Hybrid TP-EP with Compute-Balanced**。这三种策略是现有 GPU 集群和分布式系统上部署 MoE 模型的标准并行策略，但在 3D NMP 架构上各有根本缺陷。
  - 全栈执行例子（以 Mixtral-8x7B-Instruct 在 3D NMP 4×4 mesh 上推理为例）：
    - **模型推理算法层**：Mixtral-8x7B, 32 MoE layers, 8 experts/layer, top-2 gating。每个 token 经 gate 网络选择 2 个 expert，执行 gate_proj + up_proj + SiLU + down_proj（FFN 计算量 = 2h * IS = 2*4096*14336 FLOPs/token/expert）。
    - **系统框架层**：
      - **TP baseline**：每个 expert 的权重沿 intermediate dimension 切分到所有 16 个节点。每层执行：各节点计算 expert FFN 的 1/16 输出 → ring all-reduce 聚合结果（通信量 = 4Bh/BW per layer，其中 B 是 batch size，h 是 hidden dim）。计算负载完全均衡，但 all-reduce 通信随 batch size 线性增长，在 3D NMP 的有限 NoC 带宽下成为瓶颈。
      - **EP baseline**：每个 expert 完整分配给一个节点（8 experts 分布在 16 个节点，部分节点空闲）。token dispatch 经 all-to-all 通信送到对应 expert 所在节点 → 节点本地计算完整 expert FFN → all-to-all combine 聚合结果。通信量小（仅 token hidden states 传输），但 expert 激活频率不均衡导致计算负载倾斜（hot expert 所在节点成为瓶颈）。
      - **Hybrid TP-EP with Compute-Balanced baseline**：2D mesh 划分为子区域（mixtral 2 子区），区域内 TP + 区域间 EP。根据 expert 平均激活频率静态分配 expert 到子区域以平衡计算量。但忽略物理拓扑对通信的影响，在带宽受限下性能退化。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：EP 的 all-to-all dispatch/combine（NCCL 等价原语）在 2D mesh 上产生不规则通信模式；TP 的 ring all-reduce 是结构化通信但数据量大。静态调度无运行时适应能力。
    - **硬件架构层**：3D NMP 加速器，每个节点有独立 local memory bank + compute die，通过 2D mesh NoC 互联（无 shared memory / shared L2 cache）。节点间通信通过 NoC 链路（configurable bandwidth 25-75 GB/s per link）。
  - Baseline 痛点：
    1. **TP 通信瓶颈（3D NMP 特有痛点 1）**：3D NMP 架构无共享内存，TP 的 all-reduce 通信必须经过 2D mesh NoC 的有限带宽链路。通信量 = 4Bh/BW 随 batch size 线性增长，在大 batch 或低带宽配置（10 TFLOPS, 25 GB/s）下成为严重瓶颈。
    2. **EP 负载不均衡（3D NMP 特有痛点 2）**：MoE 的 expert 激活天然不均衡（如 Qwen2 中近半 token 汇聚到单一 expert），EP 将完整 expert 分配给单一节点，导致 hot expert 所在节点成为计算瓶颈，其他节点空闲。3D NMP 的分布式内存无法像 GPU 集群那样通过复制 hot expert 来缓解（内存容量受限）。
    3. **Hybrid TP-EP 忽视物理拓扑（3D NMP 特有痛点 3）**：现有 hybrid 策略只考虑逻辑层面的 compute balance，忽略物理 2D mesh 拓扑中链路级拥塞。计算均衡的 placement 可能产生集中通信路径，导致链路热点和通信 tail latency。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HD-MoE 方法**：通过 Offline Automatic Hybrid Parallel Mapping + Online Dynamic Scheduling 两阶段设计，在 3D NMP 架构上协同优化 MoE 推理的计算均衡和通信效率。
    1. **Performance Analytical Model**（解决痛点 1、2、3 的基础）：构建统一性能分析框架，computation 模型基于 per-node token load 取 max，communication 模型通过 discrete-event simulation（XY routing + priority queue scheduling）精确建模 2D mesh 上不规则 all-to-all 延迟。推导线性近似 t̂_comm = (4/BW) * max_c{ Σ_g (Π_{i∈g} ⌈P_ic⌉) * f_g * B * h }，经验验证 R² > 0.9，为 LP 优化提供可微目标函数。
    2. **Node-Link Balance Co-optimization**（解决痛点 1、2、3）：
       - **Stage 1 - Node Balance (LP)**：将 expert placement 形式化为线性规划问题。连续变量 P_ic ∈ [0,1] 表示 expert i 分配到节点 c 的比例（允许部分分配，即 TP 模式）。目标 min(t_comp + 2γ*t̂_comm)，约束包括：(a) 每个 expert 分配完整（Σ_c P_ic = 1），(b) 计算负载有界（基于 compute/communication ratio R_CC），(c) 通信量均衡。**关键设计**：P_ic 是连续值而非 binary，允许 hot expert 部分切分（TP 模式）以平衡计算，cold expert 完整分配（EP 模式）以减少通信。这实现了 TP-EP 的自动混合。
       - **Stage 2 - Link Balance (Bayesian Optimization)**：将 LP 得到的逻辑集群映射到 2D mesh 物理节点。目标是最小化链路拥塞和通信 tail latency。Bayesian Optimization 适合此问题因为：(a) 每次评估需运行 discrete-event simulation（expensive），(b) 目标函数相对平滑（相邻节点交换只引起微小通信成本变化）。
    3. **Dynamic Placement Strategy**（解决 EP 的运行时负载不均）：
       - Priority Detection：利用相邻层 expert activation 的时间局部性预测下一层热点，计算 per-expert 优先级 = 2*P_ic*f̂_i*IS/comp。
       - Optimal Pre-broadcast：对预测热点 expert，在上层推理进行时利用空闲 NoC 带宽将其预广播到所有节点。α-β 模型推导最优 chunk size c = sqrt(α*h*IS/(2*β*k*sqrt(D)))。
       - Communication-Efficient Dispatch：预广播后 token 路由到持有其激活 expert 的节点中负载最低者，避免额外通信。
  - 全栈执行例子（HD-MoE, Mixtral-8x7B, 3D NMP 4×4 mesh，与 baseline 同配置对比）：
    - **模型推理算法层**：与 baseline 相同（Mixtral-8x7B, top-2 routing），不修改模型结构或 gate 逻辑。
    - **系统框架层**：Offline 阶段运行 LP + BO 搜索最优 P_ic 和物理映射。对 Mixtral 的 8 experts：hot expert（如 expert 0, 3）部分切分到多个节点（TP mode, P_ic < 1），cold expert 完整分配到单一节点（EP mode, P_ic = 1）。Online 阶段：每层推理前运行 Priority Detection，预测下一层热点 → 在上一层计算进行时预广播热点 expert（利用空闲 NoC 带宽）→ 每个 token dispatch 到负载最低的候选节点。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：对比 baseline 的静态 all-to-all（EP）或 all-reduce（TP），HD-MoE 的通信模式是 hybrid——hot expert 的分片节点间进行 TP 式 all-reduce（小范围），cold expert 保持 EP 式 all-to-all（稀疏）。Dynamic 阶段在上层计算时异步预广播热点 expert 权重（利用 α-β 最优 chunk size），dispatch 时基于节点实时负载做贪心路由。
    - **硬件架构层**：与 baseline 相同的 3D NMP 4×4 mesh。结果：speedup 1.1-1.8× vs TP，1.1-1.5× vs EP，1.0-1.4× vs Hybrid TP-EP。关键硬件利用策略：(a) 计算受限（2.5 TFLOPS, 75 GB/s）时，HD-MoE 自动偏向 EP 为主的混合策略，减少 TP all-reduce 开销；(b) 通信受限（10 TFLOPS, 25 GB/s）时，自动偏向 TP 为主的混合策略，避免 EP 的 all-to-all 不规则通信造成链路拥塞。

## HarMoEny: Efficient Multi-GPU Inference of MoE Models

- baseline方法是什么？
  - **现有 MoE 多 GPU 推理系统**：使用 expert parallelism (EP) 将 expert 分布到多 GPU，通过 all-to-all 通信完成 token dispatch/combine。负载均衡策略分为两类：
    1. **Static placement（DeepSpeed, FastMoE, FasterMoE）**：使用 round-robin 将 expert 分配到 GPU，不含 token 级负载均衡。FasterMoE 增加 dynamic shadowing（将热门 expert 参数广播到所有 worker），但 shadowing 受 GPU memory 限制（大 expert 模型如 Qwen 33MB/expert 时效果受限）。
    2. **Profiling-based placement（ExFlow）**：离线 profiling 后使用 integer programming 计算最优 expert placement，利用 inter-layer expert affinity 减少 all-to-all 通信。但 profiling 开销极大（Switch128 需 8.5 分钟，Qwen 需 45 分钟），无法适应 batch 间动态变化的 expert 流行度。
  - Baseline 痛点：
    1. **动态 Expert 流行度偏斜导致 GPU 严重欠利用**（核心痛点）：实际 workload 中 expert 流行度随输入 domain 变化（如 medical vs programming prompts），且 batch 间剧烈波动。图 1 显示 Qwen model 的层 0 仅 3/128 expert 接收平均 19% token，最后层 3 expert 接收 60%。这导致 GPU idle time 高达 82-86%。
    2. **Static/profiling 方案无法适应动态偏斜**：静态方案（round-robin placement）完全无法处理偏斜；profiling 方案（ExFlow integer programming）在 batch 间偏斜波动时来不及重新计算（profiling 时间 >> batch 处理时间）。
  - 全栈执行例子（Baseline DeepSpeed/FastMoE on 8×V100, Switch128, 90% skew workload）：
    - **模型推理算法层**：Switch128 MoE, 128 experts, top-1 routing。Token → self-attention（各 GPU 复制执行）→ Router（各 GPU 复制执行）→ all-to-all dispatch token 到 expert 所在 GPU → 各 GPU 本地 expert FFN (GeMM) → all-to-all combine 返回输出。
    - **系统框架层**：DeepSpeed Tutel/FastMoE，EP=8。Expert 按 round-robin 分配：GPU0 持有 expert 0,8,16,..., GPU1 持有 expert 1,9,17,...。All-to-all 通信引入两个同步 barrier（dispatch + combine）。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 后端。
    - **kernel 调度层**：NCCL all-to-all collective kernel + cuBLAS GEMM。GPU0（持有热门 expert 0-9 中的 2 个）计算时间远长于其他 GPU → GPU1-7 在 all-to-all barrier 处等待 GPU0。图 5(a) 显示 GPU1-7 idle >82% 时间，mean batch latency 289ms。
    - **硬件架构层**：8× V100 (32GB), NVLink, 500GB system memory。PCIe 带宽用于可能的 expert 交换。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HarMoEny 方法**：通过两个互补的轻量级技术在不做 profiling 的前提下实现 near-perfect 负载均衡：
    1. **Dynamic Token Rebalancing（Algorithm 2）**（解决痛点 1）：每个 batch 中 GPU 先交换轻量 metadata（约 4kB），构建全局 token-to-expert-to-GPU 分布。贪心算法迭代识别最过载 GPU g_max → 贡献最大 token 的源 GPU g_from → 发送最多 token 的 expert e_max → 将 e_max 的 token 从 g_max 重路由到最欠载 GPU g_min。重复直至所有 GPU 负载平衡或无可转移 token。token threshold q 控制最小 transfer 粒度（由硬件规格决定的静态参数：q > φ·d_type/(2β)）。
    2. **Asynchronous Expert Prefetching（Section 4.3）**（使 rebalancing 可行）：rebalancing 可能将 token 分配到未持有对应 expert 的 GPU。HarMoEny 通过独立 CUDA stream 从 system memory 异步 prefetch 所需 expert 权重，直接覆写已完成 expert 的内存（无需写回 system memory），overwrite 加速 5.5×。prefetch 与当前 expert 计算重叠，隐藏传输延迟。
  - 全栈执行例子（HarMoEny on 8×V100, Switch128, 90% skew，对比 baseline）：
    - **模型推理算法层**：与 baseline 相同的 Switch128 MoE 模型。差异在于 MoE forward 流程被重写为 Algorithm 1 的 6 steps——在 all-to-all dispatch 之前插入了 Step 2 (metadata exchange) + Step 3 (token rebalancing)。
    - **系统框架层**：HarMoEny 用 1115 行 PyTorch 代码实现自定义 MoE Layer（nn.Module），替换标准 MoE 层。Step 2 metadata exchange 在所有 GPU 间广播 token-expert assignment（4kB），Step 3 各 GPU 独立并行运行相同 rebalancing algorithm（因 metadata 一致，结果 deterministic）。Step 4 all-to-all 基于 rebalanced schedule S（而非原始 round-robin assignment）。Step 5 中 async expert loading 在独立 CUDA stream 执行，overlap 当前 expert computation。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL + CUDA streams。
    - **kernel 调度层**：与 baseline 差异：(a) all-to-all 通信的 token 分布从 skewed（GPU0 接收 9× tokens）变为 balanced（各 GPU 处理 ≈t_avg tokens）；(b) 异步 CUDA stream 执行 system memory → GPU memory 的 expert weight copy（expert 18MB → ~2ms transfer on V100, vs sync 11ms），与 compute stream 的 expert FFN GeMM 重叠。图 11 显示 rebalancing 后 GPU idle 从 82.6% → 2.6%，async prefetch 进一步降低 latency 8.6%。
    - **硬件架构层**：与 baseline 相同（8×V100 DGX1）。关键差异：GPU idle time 从 82.6% → 2.6%（rebalancing alone）→ 进一步减少（async prefetch）。Scheduler 开销：30.8% (Switch128) / 20.3% (Qwen) of mean layer latency，但 total latency 仍显著降低（Switch128: 289ms → 136.6ms, -52.7%；Qwen: -63.7%）。
  - **关键性能对比**：
    - 90% skew, Switch128: throughput 186 tok/s vs ExFlow 106 tok/s (+75%), TTFT 5.36ms vs 9.38ms (-43%)
    - 50% skew, Switch128: throughput 201 tok/s vs FasterMoE 155 tok/s (+30%)
    - Qwen (larger experts 33MB): throughput 36 tok/s, consistently 15-28% faster than FastMoE/FasterMoE across real-world datasets
    - Real-world datasets: HarMoEny maintains steady 201 tok/s (Switch128) and 36 tok/s (Qwen), while baselines fluctuate with expert popularity changes
    - Fluctuating skew (0-95% per batch): HarMoEny throughput variance 152 tok²/s² vs FasterMoE 447 tok²/s² (+2.9× more stable)
    - GPU idle reduction: 84.7% vs baseline policies
  - **核心设计洞察**：HarMoEny 的本质是将 MoE 推理的负载均衡问题从一个"offline placement 问题"重新定义为"online scheduling 问题"。其核心创新在于利用了 MoE 推理的一个被忽视的性质——all-to-all 同步 barrier 之前存在天然的决策窗口。通过在这个窗口中插入轻量 metadata exchange（4kB），所有 GPU 获得全局视图，可以 deterministic 地计算相同的 rebalanced schedule，无需额外同步。这使得负载均衡的开销从 profiling 的分钟级降到 metadata exchange 的微秒级，从而能够适配 batch-by-batch 的 expert 流行度波动。Async expert prefetching 是一个精巧的补充设计——它将"rebalancing 需要 expert 移动"这个看似限制转化为优势：overwrite-based loading 比传统的 write-back-then-load 快 5.5×，因为省去了 system memory 回写步骤。论文通过 Equation (4) 将 token threshold q 形式化为仅依赖硬件规格的静态参数，使系统设计者无需 per-model/per-workload 调参。最终效果是近乎完美的 GPU 负载均衡（图 2 ECDF），将 GPU idle time 降至几乎为零。

## HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs

- baseline方法是什么？
  - **Expert Parallelism (EP) on heterogeneous GPUs（DeepSpeed MoE + Tutel/Lina optimizations）**：MoE 模型的 expert 按 expert parallelism 分布到所有 GPU（包括新旧 GPU），attention blocks 在每个 GPU 上复制。每 iteration：attention 计算（本地）→ all-to-all dispatch → expert FFN 计算 → all-to-all combine。由于 EP 不区分 GPU 型号，新旧 GPU 的 compute capability 差异导致更快 GPU 在 attention 完成后等待更慢 GPU 的 expert 计算，产生严重 idle。Tutel/Lina 的优化（grouped GEMM、通信重叠）虽然提升了单 GPU 效率，但没有解决异构场景下新旧 GPU 间的 compute imbalance。
  - **DistEP（naïve attention-expert disaggregation）**：将 attention 和 expert 模块分离——attention 仅在新 GPU 上执行，expert 仅在旧 GPU 上执行。但 attention GPU 和 expert GPU 之间存在严格的数据依赖（attention GPU 需等待 expert GPU 完成上一层 combine 后才开始下一层 attention），导致两侧 GPU 交替空闲，大部分时间在等待对方。DistEP 在 4K 序列长度下吞吐量仅为 HeterMoE 的 56%，甚至比 EP 还差 32%。
  - **Heterogeneity-aware Pipeline Parallelism（Whale, Metis, FlashFlex）**：将不同 pipeline stage 分配给不同 GPU 型号，每 stage 分配不同数量的 layer 来 balance compute time。但在 MoE 场景下有三重限制：(1) 不区分 attention 和 expert 模块——旧 GPU 仍被分配 attention 操作，效率低下；(2) balance 粒度为整 layer，无法像 HeterMoE 那样做 per-layer 的细粒度调整；(3) 内存限制——旧 GPU 可能无法容纳单个 MoE block（包含多个 expert 的权重 + 长序列激活），导致无法形成有效 pipeline。
  - 全栈执行例子（Baseline EP on O1 setup: 6×A40 + 6×V100, Mixtral-D1, 32K sequence）：
    - **模型训练算法层**：Mixtral-D1 (8 layers, hidden=1024, 24 experts, top-2 gating)。每 token → self-attention（各 GPU 复制执行）→ gate routing → all-to-all dispatch → expert FFN (gate_proj/up_proj/down_proj GEMMs) → all-to-all combine。训练 loss = LM loss + auxiliary load balancing loss。FP16 mixed precision + activation checkpointing。
    - **系统框架层**：DeepSpeed MoE (PyTorch v2.2 + DeepSpeed v0.14) with Tutel/Lina optimizations。EP group = 12 GPUs（6× A40 + 6× V100）, 24 experts 均匀分布在 12 GPU（每 GPU 2 experts）。Data parallelism 跨 ZP/EP groups。All-to-all 通过 NCCL collective 实现。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL backend。
    - **kernel 调度层**：NCCL all-to-all dispatch/combine + cuBLAS GEMM (expert FFN) + FlashAttention (仅 A40 支持，V100 不支持)。Attention 在 V100 上使用 xformers memory-efficient attention，受限于 memory bandwidth。A40 完成 attention 后 idle 等待 V100 expert 计算完成 + all-to-all combine。随 sequence length 增长（32K），A40 idle 严重（A40 attention 比 V100 快 3.7× for 64K）。
    - **硬件架构层**：6× A40 (48GB, GA102 Ampere) + 6× V100 (16GB, Volta)，100 Gbps RoCE。A40 支持 FlashAttention v2（利用 Ampere-specific TMA 和 async copy），V100 不支持 FlashAttention（无硬件 MHA 加速），Attention 在 V100 上是 memory-bandwidth bound。
  - Baseline 痛点：
    1. **忽视 MoE 组件异构性（核心痛点）**：新旧 GPU 在 attention 和 expert 上的相对效率差异显著——V100 在 expert 上达到 A40 的 80% 性能（因为 expert 主要是 GEMM，CUDA core 高度优化），但 attention 上 V100 性能远差于 A40（V100 不支持 FlashAttention，64K 序列下 A40 比 V100 快 3.7×）。EP 不区分这两种组件，将 attention 也分配给 V100，导致 V100 成为 attention bottleneck。
    2. **计算负载不均衡导致的 idle**：新 GPU 完成 attention 后必须等待旧 GPU 完成 expert 计算和 all-to-all combine 才能开始下一层 attention。即使旧 GPU 在 expert 上仍有 80% 新 GPU 的性能，attention 的时间差仍导致新 GPU 大量 idle。
    3. **PP 的不足**：pipeline parallelism 的 balance 粒度限制为整 layer，且旧 GPU 内存限制可能导致无法容纳单个 MoE block（包含多个 full expert 权重 + 长序列 activations）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HeterMoE 方法**：通过 attention-expert disaggregation + zebra parallelism + Asym-EA 三个机制，实现异构 GPU 上 MoE 训练的高效利用。
    1. **Attention-Expert Disaggregation（解决痛点 1）**：将每个 MoE transformer layer 的 attention blocks 和 expert blocks 分配到不同 GPU 型号——新 GPU 只执行 attention + gate，旧 GPU 只执行 expert FFN。由于 MoE 训练本就用 EP 的 all-to-all 做 expert 间的 token exchange，将 attention 和 expert 分离到不同 GPU 不引入额外通信（dispatch/combine 的数据总量不变，只是从 "attention GPU 到 attention GPU" 变为 "attention GPU 到 expert GPU"）。同时，expert 权重（占模型参数绝大部分）被 offload 到旧 GPU，减轻了新 GPU 的内存压力。
    2. **Zebra Parallelism（解决痛点 2）**：将 input batch 分为 R 个 microbatch，attention GPU 和 expert GPU 同时处理不同 microbatch。执行顺序：attention GPU 先完成 microbatch j 的 attention → dispatch token 到 expert GPU → expert GPU 计算 microbatch j 的 expert（同时 attention GPU 计算 microbatch j+1 的 attention）→ combine → 下一层。这形成了 "zigzag" 式的跨 GPU 流水线（因此得名 zebra）。Theorem 1 证明了最优 task ordering。同时，每 GPU 内 3 个 CUDA stream（2 通信 + 1 计算）重叠所有通信和计算——dispatch 和 combine 走相反方向，在独立 stream 上不发生竞争。
    3. **Asymmetric Expert Assignment（解决痛点 3）**：当 expert GPU 计算慢于 attention GPU（短序列常见），attention GPU 产生 bubbles。Asym-EA 将部分 expert 迁回 attention GPU 以 balance 计算时间。通过 "gather and squeeze"（Algorithm 1）决定在哪些层 offload 多少 expert：accumulate 跨多层的 bubble（T_E^Exp - T_A^Attn）直到足够 offload 至少一个 chunk（n_2 = n_1·M/N 个 experts per expert GPU），然后在 accumulation 最多的层 squeeze。考虑 attention GPU 内存上限 n_max 和 expert GPU 内存下限 n_min 约束。
  - 全栈执行例子（HeterMoE on O1: 6×A40 + 6×V100, Mixtral-D1, 32K sequence，与 baseline EP 同配置对比）：
    - **模型训练算法层**：与 baseline 相同 MoE 模型结构（Mixtral-D1: 8 layers, hidden=1024, 24 experts, top-2 gating）。差异在于执行方式：
      - ZP group: M=6 attention GPUs (A40) + N=6 expert GPUs (V100)，24 experts 分布在 6 个 V100（每 V100 4 experts），A40 默认不持有 expert。
      - Forward: 每 microbatch j → A40 attention (FlashAttention v2) → dispatch all-to-all (A40→V100) → V100 expert FFN → combine all-to-all (V100→A40) → next layer。下一 microbatch j+1 的 attention 与 microbatch j 的 expert 并行。
      - Backward: 对称执行，gate backward 在 A40 上分两路（confidence scores 分支 + expert outputs 分支），等 V100 发回 expert gradients 后 accumulated.
    - **系统框架层**：基于 PyTorch v2.2 + DeepSpeed v0.14 (3K 行 Python)。ZP engine 管理 ZP group 内的 module splitting 和 3-stream scheduling。使用分离的 NCCL dispatch/combine all-to-all group。NCCL all-to-all wrapper 传入不等 split size（因 Asym-EA 导致不同 GPU 处理不同数量 tokens）。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL backend。
    - **kernel 调度层**：3 个 CUDA stream per GPU：
      - Stream 0 (compute): attention/expert 计算
      - Stream 1 (comm D): dispatch all-to-all
      - Stream 2 (comm C): combine all-to-all
      - Sync via CUDA events
      - A40 上的执行顺序: Dispatch_j → (等 event) Attention_j → Combine_j → Dispatch_{j+1} → (等 event) Attention_{j+1} → ...
      - V100 上的执行顺序: (等 dispatch 到的 data) Expert_j → (等下一个 microbatch data) Expert_{j+1} → ...
      - 关键：Communicate 和 compute 在独立 stream 上重叠——dispatch 和 combine 同时在 V100 上执行，互不干扰
    - **硬件架构层**：与 baseline 相同（6×A40 + 6×V100, 100 Gbps RoCE）。
    - **关键性能数据**：
      | Sequence Length | HeterMoE vs EP | vs DistEP | vs EP (Ideal) | vs Homogeneous 4×A40 |
      |----------------|---------------|-----------|---------------|---------------------|
      | 4K | +22% | +79% | +18% | — |
      | 16K | +67% | +69% | — | — |
      | 32K | +89% (up to 2.29×) | +69% | — | — |
    - **核心设计洞察**：HeterMoE 的本质洞察是 MoE 架构本身包含两种计算特征截然不同的组件（attention 和 expert），且这两种组件在不同代 GPU 上的相对效率不同。旧 GPU 缺乏新 GPU 的 attention 硬件优化（FlashAttention 的 TMA/wgmma），但在 expert GEMM 上仍有不俗表现（V100 = 80% A40 on experts）。因此，与其让旧 GPU 勉强执行 attention（严重低效），不如让其专注 expert 计算，将 attention 全权交给新 GPU。这种 disaggregation 不引入额外通信（因为 EP 本就通过 all-to-all 在不同 GPU 间交换 token），且将 bulky expert 权重从稀缺的新 GPU 内存中卸载。Zebra parallelism 和 Asym-EA 的组合形成了一个 elegant 的两级优化：ZP 解决了 coarse-grained 的 compute-compute/communication 重叠（让新旧 GPU 同时忙碌），Asym-EA 解决了 fine-grained 的 bubble 消除（用 "gather and squeeze" 在气泡最大的层迁回部分 expert 计算）。两者的结合使 HeterMoE 能在仅一半新 GPU 的集群上达到 95% 全量新 GPU 的吞吐。

## HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

- baseline方法是什么？
  - **Megatron-LM 标准 AlltoAll（HD1-AlltoAll）**：MoE 训练中，expert parallelism 要求通过 AlltoAll collective 将 token dispatch 到对应 expert 所在的 GPU，计算完成后再 AlltoAll combine 回来。标准 AlltoAll 不利用 GPU 集群的分层拓扑结构（Node/IB → QPI → NVLink → Intra-GPU），所有 GPU 之间平等通信，导致低带宽链路（如 InfiniBand inter-node、QPI）成为瓶颈。在 MoE 场景下，由于 E/G > 1（每 GPU 持有多个 expert），同一 GPU 上的多个 expert 可能被同一 token 选中（top-K），导致 token 在 AlltoAll 中被重复传输（重复率可达 55%，见表 II），进一步放大通信开销。通信占训练总时间的 30-60%。
  - **Tutel-2DH（二维分层 AlltoAll）**：将 AlltoAll 分解为 Inter-Node + Intra-Node 两层，利用 intra-node 高带宽。但仅支持二维分层，无法适应更复杂的拓扑（如四层：Node/QPI/NVLink/Intra-NVLink），且不进行 token 去重。
  - **SmartMoE（expert placement 优化）**：通过动态调整 expert 在各 GPU 间的分布来平衡负载，但不考虑 token 去重，也不适配分层拓扑的带宽差异。在分层去重 AlltoAll 场景下，其 expert swap 策略反而可能增加通信量。
  - 全栈执行例子（Baseline Megatron-LM + 标准 AlltoAll, 4 nodes × 8 A6000 GPUs, DeepSeek-V3, K=8, E=256）：
    - **算法层**：MoE gate 做 Top-8 token-to-expert routing → mask I_route。标准 AlltoAll：每个 GPU 向所有 31 个其他 GPU 发送分配给其 experts 的 tokens。由于每 GPU 持有 E/G=8 个 experts，同一 token 可能选中同一 GPU 上的多个 experts → token 被重复发送 8 次至该 GPU。以 K=8, R=4（按 nodes 分组）为例，每 group 去重前最多 8 条相同 token → 重复率 55%（表 II）。
    - **系统框架层**：Megatron-LM → NCCL AlltoAll collective。NCCL 内部使用 Ring 或 Tree 算法，不感知分层拓扑——inter-node IB 链路 (200Gb/s) 和 intra-node NVLink (112.5GB/s) 被平等对待，低带宽 IB 链路的传输量决定整体通信延迟。
    - **编译框架层**：论文未明确说明。PyTorch eager mode → NCCL → CUDA。
    - **kernel 调度层**：NCCL AlltoAll kernel（GPU SM 上执行 send/recv）。每个 GPU 发送 max(p) ≈ T/G 的 token embeddings（M 维 × FP16=2 字节），总通信量 ≈ G · max(p) · M · 2 字节。由于 token 重复，55% 的流量是冗余的。
    - **硬件架构层**：4 nodes × 8 A6000-48G。NVLink 112.5GB/s，PCIe 4.0 x16，IB 200Gb/s。Inter-node AlltoAll 通过 IB NIC → PCIe → GPU memory，带宽受限于 IB ~25GB/s（远低于 NVLink）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HierMoE 方法**：通过两个拓扑感知的算法联合优化 MoE 训练的 AlltoAll 通信。核心洞察是：利用 GPU 集群的分层拓扑，在高层（低带宽链路）消除 token 重复以减少通信量，通过 expert 交换平衡各层级负载分布。
  - 两大设计对应解决 baseline 缺陷：
    1. **HierD-AlltoAll 解决 token 重复传输和拓扑不适配问题**：
       - Baseline 缺陷：标准 AlltoAll 无视分层拓扑，低带宽链路拖累全局；token 重复传输（55% 重复率）浪费带宽。Tutel-2DH 仅支持二维。
       - HierMoE 方法：将 AlltoAll 分解为 D 维，每层按 expert group 进行 token 去重。通过线性性能模型自动选择最优 d*。关键权衡：高层（小 group 数）去重收益大但传输量大；低层（大 group 数）去重收益小但带宽高。所有 7 种 AlltoAll 线性模型 r² > 0.997。
    2. **HierD-ES 解决去重 AlltoAll 下的负载不均衡问题**：
       - Baseline 缺陷：SmartMoE 的 expert swap 不考虑 token 去重和分层拓扑，在去重 AlltoAll 上反而降低性能（HD2-MoE-Smart < HD2-MoE）。
       - HierMoE 方法：为 HierD-AlltoAll 设计的分层 expert swap，统计去重后 token 分布变化（四种 case），增量更新通信时间估计矩阵 Q_d*（O(D·T·K·E)），选择最小化通信时间的 expert pair。smooth-max (γ=10) 平滑优化。Expert 交换 ~1% 时间开销。
  - 关键实验结果：
    - AlltoAll 通信加速比：HierMoE vs Megatron-LM 1.99×-2.72×, vs Tutel-2DH 2.34×-3.32×
    - 端到端训练加速比：HierMoE vs Megatron-LM 1.18×-1.27×
    - HierD-AlltoAll (HD-MoE) vs HD2-MoE: 1.37×-1.45× 加速
    - HierD-ES vs HD-MoE: 额外 1.13×-1.17× 加速

## Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

- baseline方法是什么？
  - **标准 MoE 架构基线**：传统 Transformer-based MoE 模型（如 Mixtral-8x22B, DeepSeek-V2）使用 top-k 路由策略，将 token 分配到 top-k 评分的 experts。存在以下缺陷：
    1. **Token Dropping 问题**：当某些 expert 过载（capacity overflow）时，top-k 路由会直接丢弃超额的 token，导致关键信息丢失，影响训练稳定性和最终性能。
    2. **KV Cache 内存开销**：标准 MHA 的 KV cache 为 4 × n_h × d_h × l bytes，在大规模推理中消耗大量显存，限制 batch size 和长上下文能力。
    3. **统一学习率导致 expert 训练不均衡**：MoE 中不同 expert（shared vs specialized）处理的 token 数量差异巨大（shared expert 处理全部, specialized 仅 1/16），使用统一学习率会使 specialized experts 训练效率低下。
    4. **缺乏 MoE 专用 Scaling Laws**：传统 scaling laws（Kaplan et al., Hoffmann et al.）针对 dense 模型设计，C = 6ND 不适用于 MoE 的稀疏激活和长序列场景。
  - 全栈执行例子（Baseline Mixtral-8x22B 训练过程）：
    - **模型推理/训练算法层**：token → MHA (80 heads, full KV cache, 4×80×d_h×64) → Router (softmax top-2 scoring) → top-2 expert FFN (SwiGLU: W_gate, W_up, W_down) → 2 expert outputs weighted sum → next layer。Token 若落入 overloaded expert 则直接丢弃，梯度不传递。
    - **系统框架层**：论文未明确说明 baseline 的训练框架。推测使用 PyTorch + Megatron-LM 或类似框架，expert parallelism 通过 All-to-All 通信。
    - **编译框架层**：论文未明确说明。标准 PyTorch eager mode + NCCL collectives + cuBLAS GEMM。
    - **kernel 调度层**：论文未明确说明。NCCL All-to-All (token dispatching), cuBLAS batched GEMM (expert FFN), FlashAttention (MHA forward/backward)。
    - **硬件架构层**：论文未明确说明。推测使用 NVIDIA H100/A100 GPU 集群。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Hunyuan-Large 方法**：从数据、模型架构、训练策略三个维度全面改进 MoE 训练：
    1. **Recycle Routing（解决 Token Dropping）**：对 top-1 路由中被丢弃的超额 token，随机重新分配到未满 capacity 的其他 specialized experts。相比直接丢弃，recycle routing 保留关键信息，提升训练稳定性，确保每个 token 都对梯度更新有贡献。
    2. **GQA + CLA 联合 KV Cache 压缩（解决 KV Cache 内存开销）**：GQA 将 80 个 attention heads 分为 8 组共享 KV，压缩 head 维度；CLA 每 2 层共享 KV cache，压缩 layer 维度。联合使用将 KV cache 从 4nhdhl 降至 2ngdhl，减少 ~95% KV cache 内存。
    3. **Expert-Specific LR Scaling（解决 expert 训练不均衡）**：基于 AdamW 最优学习率公式 ε_opt(B)，为 shared expert 和 specialized experts 分配不同的学习率（比例约 0.31），让每个 expert 在各自的 effective batch size 下达到最优训练效率。
    4. **MoE Scaling Laws（提供 architecture 决策依据）**：修正计算预算公式为 C ≈ 9.59ND + 2.3×10^8 D（针对 MoE 的稀疏激活和 attention 复杂度），训练 10M-1B 参数的 MoE 模型系列，通过 isoFLOPs 曲线拟合得到最优激活参数量 N_opt 和最优训练数据量 D_opt 与计算预算的关系。
    5. **四步合成数据 pipeline（补充数据质量和多样性）**：Instruction Generation → Instruction Evolution → Response Generation → Response Filtering，生成 1.5T tokens 高质量合成数据，针对性增强数学、代码、低资源语言和高教育价值领域。
  - 全栈执行例子（Hunyuan-Large 训练 + 推理过程）：
    - **模型推理/训练算法层**：token → RoPE encoding (256K context, base freq 1B) → GQA (80 heads, 8 KV groups, 2 layers share KV, KV cache = 2×8×d_h×64) → Router (softmax scoring) → shared expert FFN (all tokens) + top-1 specialized expert FFN (recycle routing: 若 top-1 expert 未超 capacity 直接 dispatch；若超 capacity 随机 reassign 到有空余的 expert) → combine: shared_out + specialized_out → next layer。Expert-specific LR: shared expert LR = ε_opt(B), specialized expert LR = ε_opt(B/16) ≈ 0.31×ε_opt(B)。Annealing phase（最后 5% tokens, LR 降至 peak 的 1/10, 使用最高质量数据）。Long-context pre-training: 32K→256K, RoPE base frequency scale to 1B, 每阶段 ~10B tokens (25% long corpus + 75% normal-length)。
    - **系统框架层**：论文未明确说明训练框架。开源仓库基于 PyTorch + HuggingFace Transformers。推理阶段利用 GQA+CLA 压缩 KV cache 减少显存占用，支持 256K tokens 长上下文高效推理。
    - **编译框架层**：论文未明确说明。标准 PyTorch eager mode。
    - **kernel 调度层**：论文未明确说明。cuBLAS GEMM (expert FFN), FlashAttention/GQA kernel (compressed attention), NCCL All-to-All (token dispatching in expert parallelism)。
    - **硬件架构层**：论文未明确说明训练硬件配置。开源模型支持标准 GPU 推理。
  - 关键实验结果：
    - Pre-training: MMLU 88.4 (vs LLama3.1-405B 85.2, 仅 52B activated vs 405B), MATH 69.8 (vs LLama3.1-405B 53.8), HumanEval 71.4 (vs LLama3.1-405B 61.0)
    - Post-training: MMLU 89.9 (vs LLama3.1-405B-Inst 87.3), MATH 77.4 (vs 73.8), Arena-Hard 81.8 (vs 69.3)
    - KV cache: GQA+CLA 减少 ~95% KV cache 内存 vs MHA
    - Long-context: RULER 64K-128K: 89.53 vs LLama3.1-70B-Inst 86.48; PenguinScrolls Overall 85.23 vs 69.37

## HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

- baseline方法是什么？
  - **kTransformers**：SOTA CPU-GPU hybrid MoE 推理框架。使用**静态映射策略**——基于历史 expert 激活频率，将高频激活 expert（如 shared expert）固定映射到 GPU，低频 expert 在 cache miss 时由 CPU 执行。缓存策略使用 LFU（Least Frequently Used）。全栈执行例子（以 Mixtral-8x7B on RTX A6000 + 10-core Xeon, single token decode 为例）：
    - **模型推理算法层**：标准 MoE decoder，token → Attention (GPU) → MoE Gate (GPU, top-K routing) → expert FFN。kTransformers 将高频 expert 固定映射到 GPU（hot expert cache），低频 expert 由 CPU 计算（cache miss 时）。无动态 workload 适应能力。
    - **系统框架层**：kTransformers (kernel injection + CPU-GPU parallel execution)。Warmup 阶段 profiling 确定高频 expert（基于 calibration data 上的历史激活频率）。Runtime 阶段：gate 输出后 → 检查每个 activated expert 是否在 GPU cache → hit: GPU 执行 Marlin 4-bit quantized GEMM → miss: CPU 执行 llama.cpp C++ GEMM → CPU→GPU copy output → 聚合。
    - **编译框架层**：论文未明确说明。PyTorch/C++，llama.cpp backend。
    - **kernel 调度层**：GPU 端使用 Marlin 4-bit quantized GEMM kernel，CPU 端使用 llama.cpp C++ GEMM kernel。CPU/GPU 执行在独立 stream/thread 上并行，但任务分配完全由预定义的 static mapping 决定——低频 expert 总是 CPU 执行，即使 GPU idle。LFU cache eviction 不考虑 expert routing score 的预测信号。
    - **硬件架构层**：NVIDIA RTX A6000 + Intel Xeon Gold 5220R (10 cores)，PCIe。CPU 首 expert 计算较慢（cold cache），但后续 expert 因 CPU cache 命中而加速。
  - Baseline 痛点：
    1. **静态映射不适应动态激活（核心痛点 1）**：MoE expert 激活模式高度不稳定——相比 neuron-level sparsity，expert 激活频率分布更均匀（图 3a），预测困难。kTransformers 的固定映射（基于历史频率）忽略 runtime 的动态 workload 变化，导致 suboptimal CPU/GPU 资源利用和 load imbalance（图 1b vs 1c）。
    2. **缺乏 MoE-specific 缓存策略（核心痛点 2）**：LFU/LRU 忽略 MoE expert routing score 的预测信号。图 3b 表明高 score expert 在下一 iteration 中更可能被重用，但 LFU 仅基于历史使用频率，无法利用 score 信息做前瞻性缓存决策。
    3. **预取缺乏优先级决策（核心痛点 3）**：现有预取方法未探讨当多个后续层 expert 可被预取时如何做优先级决策。在 MoE 中，由于残差连接导致的层间 hidden state 相似性，gating 信息可复用做预测，但预取哪个层的 expert 对整体调度效率影响最大未被讨论。
    4. **调度问题 NP-Hard 但存在可利用规律（核心痛点 4）**：CPU-GPU workload 分配是 NP-hard 问题，但在 MoE 特定上下文中，expert transfer time 恒定、GPU 计算时间与 expert 数线性、CPU 首 expert 慢后续快（cache 利用）等规律可被利用来设计高效调度规则，而现有方法未利用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HybriMoE 方法**：通过三个互补的优化设计，将 MoE CPU-GPU 推理的"静态映射"转变为"动态自适应调度"。
    1. **Dynamic Intra-Layer Hybrid Scheduling（解决痛点 1 和 4）**：
       - Baseline 缺陷：静态映射无法适应动态 workload 变化，NP-hard 调度问题未利用 MoE-specific 规律。
       - HybriMoE 方法：三条优先级规则（GPU 优先高负载缓存 expert、CPU 优先低负载未缓存 expert、PCIe 优先高负载未缓存 expert）将调度从排序问题简化为分配问题 `argmin max(CPU_TIME, GPU_TIME)`。执行前仿真（贪心 fill CPU/GPU/PCIe timelines）评估不同分配策略，runtime 选择最优。利用的关键规律：expert transfer time 恒定、GPU 时间 ∝ expert 数（nearly constant per expert）、CPU 首 expert 慢后续快（因 CPU cache 重用）。
    2. **Score-Aware Caching / MRS Policy（解决痛点 2）**：
       - Baseline 缺陷：LFU/LRU 忽略 expert routing score 的预测信号。
       - HybriMoE 方法：Minus Recent Score (MRS) 策略，S = α × TopP(s) + (1-α) × S。仅累积 top-p 个 expert 的 score（因低 score expert 的 reuse probability 差异不显著）。利用图 3b 的观察——高 score expert 重用概率显著更高。25% cache capacity 下 MRS 比 LRU 高 6-8% hit rate。
    3. **Impact-Driven Prefetching（解决痛点 3）**：
       - Baseline 缺陷：预取缺乏多候选下的优先级决策机制。
       - HybriMoE 方法：复用后续 3 层的 gating 信息预测 expert activation → 对每个候选 expert 仿真其预取对整体调度效率的影响（复用 IV-B 的仿真逻辑）→ 贪心选择 impact 最高的 expert 预取。低开销（仿真 <μs 级），适合 real-time 推理。
  - 全栈执行例子（HybriMoE on RTX A6000 + 10-core Xeon, Mixtral-8x7B single token decode, 25% GPU cache ratio）：
    - **模型推理算法层**：与 baseline 相同的 MoE 模型结构（Mixtral-8x7B, top-2 routing）。差异在于运行时行为：
      - Warmup: profiling CPU/GPU expert latency + PCIe transfer latency → 初始化 MRS cache (random k experts)。
      - Per-layer: Gate → top-K expert selection → Simulation Scheduler (优先级队列 + 贪心 fill timelines) → 输出 expert-to-device assignment → Multi-stream parallel execution (GPU Marlin 4-bit GEMM + PCIe cudaMemcpyAsync + CPU llama.cpp C++ GEMM) → Impact-driven prefetching (预测 next-3-layers expert activation → 仿真评估预取收益 → 贪心预取) → MRS cache update (S = α·TopP(s) + (1-α)·S)。
    - **系统框架层**：基于 kTransformers + llama.cpp kernels。核心修改：Hybrid Scheduler (优先级规则 + simulation) 替换 static mapping；MRS Cache Manager 替换 LFU；Prefetching Module (impact-driven) 新增。Parallel execution engine: fine-grained CUDA stream (GPU compute + PCIe transfer) + CPU thread pool，CUDA event sync。
    - **编译框架层**：论文未明确说明。PyTorch/C++ (llama.cpp backend)。
    - **kernel 调度层**：
      - GPU Stream 0: Marlin 4-bit quantized GEMM kernel (SOTA 4-bit GPU GEMM) for cached experts。
      - GPU Stream 1: cudaMemcpyAsync for PCIe expert weight transfer (CPU→GPU or GPU→CPU)。
      - CPU Thread Pool: llama.cpp C++ GEMM kernel for uncached experts (CPU 端，首 expert cold cache → 后续 expert cache 命中加速)。
      - Simulation Scheduler: CPU 侧轻量级仿真 (<μs 级 latency)，不占 critical path。
      - 关键时序（以 4 experts: E₁(cached,high), E₂(uncached,high), E₃(cached,low), E₄(uncached,low) 为例）：
        ```
        Time →
        GPU:     |=== E₁ (Marlin) ===|=== E₂ (transferred+execute) ===|
        PCIe:    |== E₂ transfer ==|                                   |
        CPU:     |== E₄ (low-load) ==|                                 |
        ```
        vs Baseline (static mapping, E₂ 固定 CPU):
        ```
        GPU:     |=== E₁ ===|=== E₃ ===|  // idle after
        CPU:     |==== E₂ (high-load) ==============|== E₄ ==|
                 ↑ CPU 成为瓶颈
        ```
    - **硬件架构层**：与 baseline 相同（RTX A6000 + Xeon Gold 5220R 10-core）。但硬件利用率显著提升：GPU 避免了 baseline 中因等待 CPU 完成 heavy expert 导致的 idle，CPU 专注于其擅长的 low-load expert（延迟与 load 线性相关，low-load 时 CPU 优势最大）。
    - **关键性能数据**：
      | Model | Stage | Cache | HybriMoE vs kTransformers |
      |-------|-------|-------|---------------------------|
      | All (avg) | Prefill | 25-75% | 1.33× |
      | All (avg) | Decode | 25-75% | 1.70× |
    - **核心设计洞察**：HybriMoE 的本质洞察是将 MoE CPU-GPU 的 "task-to-hardware mapping" 问题解耦为三个子问题——(i) 当前层的 expert 如何分配到 CPU/GPU（intra-layer scheduling），(ii) 下一层哪些 expert 值得预取（inter-layer prefetching），(iii) 跨 iteration 哪些 expert 值得缓存（inter-iteration caching）。这三个子问题的时间维度不同（intra-layer 是 ms 级优化，inter-layer 是 μs 级预测，inter-iteration 是 s 级状态管理），HybriMoE 对每个时间维度使用了匹配的优化机制。优先级规则的设计体现了对 MoE CPU-GPU 异构计算特性的深刻理解——GPU 延迟近乎恒定（memory-bound），CPU 延迟线性增长（compute-bound），PCIe 传输恒定——这三者的"恒定 vs 线性"差异正是动态调度的 physics 基础。MRS 的 elegant 之处在于它不需要额外的预测模型（如 MLP predictor），而是直接利用 gating network 天然输出的 routing score——因为这个 score 本身就编码了"模型认为哪些 expert 对当前 token 重要"的信息，而"重要的 expert 更可能被未来 token 重用"是 MoE expert specialization 的自然推论。

## Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

- baseline方法是什么？
  - **One-shot Expert Pruning**（如 Lu et al., 2024; Muzio et al., 2024; He et al., 2024）：对 MoE 模型一次性估算所有 expert 的重要性，在每层丢弃特定比例的专家。流程为：使用单一准则（如 expert usage frequency 或 token reconstruction loss）一次估算所有 expert 重要性 → 按分数排序 → 每层丢弃最不重要的 r 个 expert → 从 router gating 矩阵 W_G^{d×n} 中删除对应列 → W_G^{d×(n-r)}。核心缺陷：(1) 一次性估算无法反映丢弃某些 expert 后其余 expert 重要性的变化，选出的子网络质量差；(2) 丢弃后 router 矩阵直接移除了对应 expert 入口，导致负载分布严重偏斜（某些 retained expert 被过度路由），子网络处于 sub-optimal 状态；(3) 随 sparsity 增加，性能急剧下降（≥25% sparsity 时 zero-shot MMLU 已降至 random guess 水平）。
  - **LLM Weight Pruning**（Wanda, Magnitude, Random）：对 FFN 权重矩阵做 2:4 structured sparsity（NVIDIA Ampere 硬件支持），移除不重要的权重连接，但不改变模型架构。缺陷是无法利用 MoE 架构特有的 expert-level redundancy 和 conditional computation 特性。
  - **全栈执行例子（One-shot Expert Pruning on Mixtral-8×7B Base）**：
    - **模型推理/训练算法层**：加载 HuggingFace Mixtral-8×7B → C4 calibration set 256 samples → forward pass 收集 expert usage frequency → 按层排序选取每层 top-r 最不常用 expert → 修改 model.config (num_experts 从 8→(8-r)) → 删除对应 router 列和 expert weights → 直接 zero-shot 评估 MMLU/ARC/WinoGrande。任务性能在 25% sparsity 时已崩溃。
    - **系统框架层**：HuggingFace Transformers + PyTorch，标准训练/推理脚本。论文未明确说明 Serving 框架修改。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM、PyTorch autograd。
    - **硬件架构层**：8×NVIDIA A100 GPU。论文未明确说明 GPU 架构级优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MC-Suite + MoE Lottery Subnetworks 方法**：
    1. **MC-Suite（16 种多维准则）**：从四个维度（Weight/Inference/Activation/Gradient）系统化评估 expert 重要性，涵盖 expert & router weight dynamics、expert inference behavior dynamics、intermediate activation properties、expert gradient properties。从 16 种准则中实验验证最优者：Min-EAN（最小激活范数）和 Min-EGE（最小梯度熵），因为它们同时考虑了 input tokens 和 weight parameters，比单纯基于 expert usage 的准则更精确。
    2. **Iterative Pruning（替代 One-shot）**：将 s% 的 expert 丢弃分成 k 轮，每轮丢 s/k%（如 50%=4×12.5%）。每轮丢弃后 re-estimate 剩余 expert 的 MC-Suite 准则值。这解决了 one-shot 无法反映 expert 间依赖关系变化的缺陷——第一轮丢弃后，剩余 expert 的重要性发生变化，re-estimation 捕捉到了这一点。
    3. **Task-Agnostic Budget Finetuning（MoE Lottery 核心）**：每轮 pruning 后使用 ~0.2M training tokens 做 next-token prediction finetuning（C4 calibration data），progressive schedule 逐轮翻倍（0.2M→0.4M→0.8M→1.6M），总 budget ~1M tokens 即饱和。Finetuning 的作用：(i) 重新调整 router weights 使负载分布重新均衡（Figure 6: 红色虚线→绿色实线大幅改善）；(ii) 恢复因 expert 丢弃造成的 abrupt performance drop。
    4. **Instruction-Following Recovery**：实验验证 expert dropping 主要损害的是 instruction-following 能力（非 pretraining knowledge/reasoning）。通过 k-shot examples 或 SFT（supervised fine-tuning with instruction dataset）可显著恢复下游性能。
  - **对应解决 Baseline 缺陷**：
    - One-shot 一次性估算 → Iterative 多轮 re-estimation 捕捉 expert 间依赖关系变化（Figures 5a/5b 直观展示了 one-shot 与 iterative 选出的 expert 高度不一致）。
    - One-shot 丢弃后 sub-optimal 状态 → Task-agnostic finetuning 重调 router weights + 负载均衡（Table 2: 75% sparsity 下 MoE Lottery pp=13.05 vs one-shot=30.59）。
    - 性能随 sparsity 急剧下降 → MoE Lottery @ 50% sparsity 仍保持 robust（MMLU 40.79 vs one-shot 18.91）；k-shot/SFT 可进一步恢复至接近 full-MoE 水平。
  - **全栈执行例子（MoE Lottery Subnetwork on Mixtral-8×7B Base, 50% sparsity, k=4 rounds, Min-EAN criterion）**：
    - **模型推理/训练算法层**：Round 1: C4 256 samples forward pass + forward hooks 收集每层 8 个 expert 的 output activations → 计算 Min-EAN = argmin ||A_Ep||₂ → 每层丢弃 1 个 expert → W_G^{4096×8}→W_G^{4096×7} → 0.2M tokens C4 next-token-prediction finetuning (AdamW, lr=1e-6, batch=8, cosine schedule) → Round 2: 重新 forward calib data 收集 7 个 expert 的 activations → 再丢 1 个 → 0.4M tokens finetuning → Round 3→Round 4 → 最终每层剩 4/8 experts, 50% expert sparsity → zero-shot 评估 5 个下游 benchmark → 对某些任务可补充 k-shot examples 或额外 SFT。关键张量流: router_score = softmax(H @ W_G^{4096×n}) → top-2 → expert_i(H) = SiLU(H @ W_gate) * (H @ W_up) @ W_down → ∑ G_i * expert_i。
    - **系统框架层**：HuggingFace Transformers (MixtralForCausalLM) + PyTorch DistributedDataParallel on 8×A100。论文未修改 Serving 框架，但 load balancing 的改善间接利于 GPU memory utilization。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：论文未明确说明。标准 cuBLAS GEMM（FFN linear layers）+ PyTorch autograd（梯度计算）。
    - **硬件架构层**：8×NVIDIA A100 GPU。Memory 从 180GB→~99GB（50% sparsity），speedup 1.27×。论文未明确说明 GPU 架构级优化，speedup 来自 expert weight 减少后 kernel launch 次数和参数加载量减少。

## HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

- baseline方法是什么？
  - **标准 Expert Parallelism (EP) + MoE 训练系统**（Tutel, FasterMoE, SmartMoE）：在单 DC 高性能环境下，通过 All-to-All (A2A) 通信在 GPU 间交换 token data 和 expert 输出。核心流程：Gate network 计算 routing → A2A dispatch（将 token 发送到对应 expert 所在 GPU）→ Expert FFN 本地计算 → A2A combine（将 expert 输出合并回原 GPU）。现有优化方法集中在对计算和通信重叠（FasterMoE 的 overlap scheduling、Tutel 的 2D 分层 A2A、SmartMoE 的 pipeline 调度），本质是在高带宽 DC 内部通过隐藏通信延迟来提升吞吐。但在跨 DC 场景下，低带宽（如 10Gbps Ethernet vs PCIe 128Gbps）导致通信时间远远超过计算时间，重叠策略失效——因为无论怎么重叠，通信时间本身无法被消除或缩减。此时 EP 占训练总时间的 50%-90%（Figure 2b），成为核心瓶颈。
  - **全栈执行例子（Baseline Tutel on Cluster-L: 4 DCs × 8 GPUs, MoE with E=32 experts, K=2 activated experts）**：
    - **模型推理/训练算法层**：Standard MoE layer: token embedding → Attention → Gate(top-K routing) → A2A dispatch(token data 跨 DC 传输) → Expert FFN(2×GeMM with SiLU on GPU Tensor Cores) → A2A combine(output 跨 DC 传输) → next layer。每层 2 次全局 A2A，每次传输 data size = D·(G-1)/G per GPU。当 G=32 时，跨 DC 低带宽链路的 A2A 传输时间主导了整个 iteration。
    - **系统框架层**：PyTorch v1.12.1 + Tutel。Tutel 使用 NCCL All-to-All collective 原语执行 token dispatch/combine。训练使用 Adam optimizer + DDP (All-Reduce 同步梯度)。跨 DC 通信经过 10Gbps Ethernet，与 intra-DC 的 128Gbps PCIe 形成巨大带宽差距。
    - **编译框架层**：论文未明确说明。PyTorch eager mode → CUDA compiler → cuBLAS/NCCL backend。
    - **kernel 调度层**：标准 cuBLAS GEMM (expert FFN) + NCCL All-to-All collective。无自定义 kernel。通信与计算通过 Tutel 的 pipeline 调度尝试重叠，但在 10Gbps 下 A2A 通信时间（数十 ms）远超 expert 计算时间（<1ms），重叠收益极低。
    - **硬件架构层**：NVIDIA A800 GPU (PCIe 3.0 x16 128Gbps 节点内) + Ethernet 10Gbps 跨节点。EP 的 A2A 通信经过低带宽跨 DC 链路，成为系统瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HybridEP 方法**：通过三部分设计将 EP 从纯 A2A 通信改造为数据-专家混合传输，结构化消除跨 DC 通信瓶颈：
    1. **Stream-Based Modeling（流建模，§III）**：将 MoE 训练解耦为计算流和通信流，建模 A2A（传输 data）和 AG（传输 expert）的延迟特性——A2A 延迟 O(1)（与 GPU 数无关），AG 延迟 O(n)（随 GPU 数线性增长）。通过推导 A2A→AG 转换关系：减少 $\frac{D}{G}$ 的 A2A 流量换取 $P_E$ 的 AG 流量，求解最优混合比例 p。关键洞察：当 $2D \geq G \cdot P_E$ 时，仅使用 AG（p=0）即可消除所有跨 DC 的 token data 传输，用更适合压缩和异步的 expert 传输替代。
    2. **Domain-Based Partition（域分区，§IV-A）**：将 modeling 输出的 p 值映射到 GPU 级通信拓扑。定义 Expert Domain（域内 AG，域间 A2A），通过 Multilevel Description → Location Renumbering → Topology Construction 三步在处理复杂层级硬件架构（DC→Node→GPU）的同时保持通信模式清晰。
    3. **Parameter-Efficient Migration（参数高效迁移，§IV-B）**：利用 expert 的两大优势——(i) 可压缩性（expert weight 分布紧凑，残差稀疏，可通过 shared+residual Top-k 实现 50× 压缩），(ii) 异步潜力（expert 仅间歇参与计算，可提前传输）。通过 SR-Based Expert Compression 减少传输流量，Asynchronous Communicator 实现 expert 通信与 pre-expert computation 的完全重叠。
  - **对应解决 Baseline 缺陷**：
    - Baseline 的 A2A 通信时间在低带宽下无法消除 → HybridEP 通过 expert 迁移将 A2A 转换为 AG，用更适合压缩和异步传输的 expert 替代 data 传输，从根本上减少跨 DC 链路上的通信量。
    - Baseline 的重叠策略在低带宽下失效 → HybridEP 的 AG 通信可以与 pre-expert computation 重叠（expert 不需要 token data 作为输入，可独立传输），且 AG 的异步特性和高可压缩性使其 overhead 远小于 A2A。
    - Baseline 通信流量随 token 数线性增长 → HybridEP 的 AG 流量与 token 数无关（仅与 expert 大小相关），提供固定上界的通信流量，使系统性能更可预测（Figure 16）。
    - Baseline 的 A2A 频率为 O(G²) → HybridEP 通过扩展 expert domain 将 A2A 频率降低并转换为 AG（Table VII），减少跨域通信次数。
  - **全栈执行例子（HybridEP on Cluster-L: 4 DCs × 8 GPUs, p=0, AG-only case, Mistral-Small, E=32, K=2）**：
    - **模型推理/训练算法层**：
      - Token batch → Embedding → Attention → Gate(top-K routing)
      - **关键差异**：不再执行跨 DC 的 A2A dispatch/combine
      - 每个 Expert Domain 内（假设 S_ED=8，每 DC 的 8 GPU 为一个 domain），所有 expert 提前通过 AG 收集到域内每 GPU
      - Expert FFN 全部在本地 GPU 执行（因为所有 expert 参数已通过 AG 在域内可用）
      - 无需跨域数据传输，仅需域内 AG 同步 expert 参数
      - SR-Based Expert Compression 将 expert 传输量压缩 50×：原始 P_E=4.7MB → 压缩后 ~0.094MB per expert
    - **系统框架层**：
      - 修改后的 Tutel + PyTorch v1.12.1
      - 训练前：Stream-Based Modeling 根据 G=32, B_inter=10Gbps, P_E=4.7MB, D=3MB → 判断 2×3MB - 32×4.7MB < 0 → Case 2.1 或转换判断 → 若转为 Case 2.2 则 p=0 (仅 AG)
      - Domain-Based Partition：构建 2 层 topology (S_ED^0=4, S_ED^1=8)
      - SREncode 与 optimizer step 融合：将每 expert 压缩为 value-index 稀疏格式存入 Send Queue
      - Asyn-comm：AG 通信在 Attention 计算时并发执行，NCCL All-Gather 从 Send Queue 收集域内其他 GPU 的压缩 expert
      - SRDecode 与 expert FFN 融合：解码恢复完整 expert 参数并立即执行 FFN
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：NCCL All-Gather (expert 参数域内收集) 替代了 NCCL All-to-All (token data 跨域传输)。Top-k 压缩/解压通过 custom CUDA kernel 实现。AG 通信通过 CUDA stream 与 Attention 的 GEMM kernel 并行。
    - **硬件架构层**：4 × 8 × NVIDIA A800。intra-DC PCIe 3.0 128Gbps, inter-DC Ethernet 10Gbps。当 p=0 时，跨 DC 的 A2A token 传输完全消除，仅保留域内的 AG expert 传输，而 AG 的流量被 SR 压缩减少 50×，并通过异步通信隐藏延迟。端到端 speedup：up to 5.6× (192MB data, Cluster-L) vs Tutel/FasterMoE/SmartMoE。

## Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- baseline方法是什么？
  - **Monolithic MoE Inference（SGLang, vLLM, LINA）**：Attention 和 MoE 层共址于同一 GPU，使用共享并行度配置（如相同的 TP/EP degree），通过复制完整模型实例进行 scale。全栈执行例子（DeepSeek-V2 decode, 8×H100 TP=8）：
    - **模型推理算法层**：Token → Embedding → 每层: Attention (MLA, 512 context) → MoE Gate (Top-K routing) → Expert FFN (每 token 激活少量 expert, 其余 idle) → Shared Expert → LM head。所有 layer 使用相同并行度。
    - **系统框架层**：SGLang/vLLM monolithic deployment。模型实例包含完整 attention + MoE 权重。Scaling 单元 = 完整模型副本（如 DeepSeek-V3 最小 16 H100）。Elasticity 粗粒度：加载全部参数 + 重建并行组。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL。
    - **kernel 调度层**：cuBLAS GEMM + NCCL AllReduce/All-to-All。MoE expert 参数占据 >90% 显存，仅少量 expert 被每 token 激活，大量显存被 idle expert 占用。KV cache 与 expert 参数共享同一 GPU memory budget。
    - **硬件架构层**：8× H100 80GB NVLink。Expert 参数完全 resident GPU HBM。
  - **Disaggregated MoE Inference（MegaScale-Infer, xDeepServe, EaaS）**：Attention 和 MoE 层部署于不同节点，实现独立配置。但仍有三个缺陷：① Expert scheduling 聚焦 token 负载均衡（如 EPLB 均匀分配 tokens），不直接 minimize a_max（各 GPU 的最大 distinct activated expert 数），token-balanced 配置下仍存在 straggler；② Resource scaling coarse-grained（MegaScale-Infer 限制为 balance attention/MoE 执行时间的配置，EaaS 仅提供弹性通信通道，xDeepServe 无 scaling policy）；③ Communication 使用 attention 侧 gating + pairwise all-to-all（每 attention instance 与所有 MoE instance 通信），产生 O(m×n) 次小消息传输。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **JANUS 方法**：Disaggregate attention/MoE + Adaptive two-phase communication + AEBS scheduling + SLO-aware fine-grained scaling。三大核心设计：
    1. **Adaptive Two-Phase Communication（解决 cross-sub-cluster 通信瓶颈）**：不同于 baseline 的 O(m×n) pairwise 传输或 attention-side gating，JANUS 将 gating 移至 MoE 侧（发送完整 activation 而非 per-expert packed tensor），通过 Phase-1 intra-node aggregation + Phase-2 inter-node bulk transfer 减少跨节点传输次数。案例选择（Case-1 直接 vs Case-2 中继）自适应流量负载。NVSHMEM one-sided put 消除同步开销，元数据打包进 signal 避免额外传输。
    2. **AEBS - Activated-Expert-Balanced Scheduling（直接解决 a_max straggler）**：发现 MoE 层 latency ∝ a_max（各实例最大 distinct activated expert 数），而非 token count。AEBS 直接 minimize a_max 而非 token imbalance。单副本 expert 固定分配，多副本 expert 贪心选负载最低实例。Synchronization-free：每个 MoE instance 独立运行相同 deterministic kernel，消除 CPU-GPU 同步和跨 GPU 协调，开销 <90μs。
    3. **Fine-Grained SLO-Aware Resource Scaling（解决 coarse-grained elasticity）**：构建 Roofline-based TPOT model + Monte Carlo â_max estimator + Little's Law 稳态求解。枚举完整 (n_a, n_e) 二维搜索空间（不限于 attention/MoE balanced 配置），选择满足 SLO 的最小 GPU 配置。联合优化 expert placement（Algorithm 3: activation-aware co-activation minimization）。15 min 间隔增量调整。

  - 全栈执行例子（JANUS, DeepSeek-V2, 1A6E 配置 vs Monolithic SGLang 8GPU）：
    - **模型推理算法层**：与 baseline 相同 MoE 模型。JANUS 通过 AEBS 改变 expert activation 的调度方式——从 "random/EPLB token balancing" 变为 "expert-count balancing"。Roofline 分析指导 MoE 层 latency 建模 T_moe = β·a_max + c_e（线性依赖 a_max）。
    - **系统框架层**：JANUS modifies SGLang。Attention instances (n_a=1) 与 MoE instances (n_e=6) 独立部署。对比 baseline monolithic SGLang (8 GPUs 统一 TP=8)，JANUS 使用 7 GPUs (1A+6E) 达到更高 throughput。Scaling 控制器每 15min 根据 workload 变化调整配置 (e.g. 1A6E → 2A6E → 4A6E → 5A10E)。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：AEBS GPU kernel 替代 baseline 的 EPLB/random scheduling。NVSHMEM one-sided put 替代 NCCL collective (baseline 的 pairwise/all-to-all)。MoE 侧 gating (EGate) + 2PC 替代 baseline attention-side gating + 1PC。关键性能：2PC+EGate 比 2PC+AGate throughput 提升 4-34%（因避免 per-link routing metadata 传输）。
    - **硬件架构层**：H100 80GB × 8 per node, NVLink + IB。JANUS 将 attention 和 MoE 分离到不同 GPU，expert 参数仅驻留 MoE GPU。与 baseline 所有 GPU 同时持有 attention + expert 参数不同，JANUS 各 GPU 内存仅用于单类型参数，减少 memory pressure。

  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | JANUS 设计 | 效果 |
    |-------------|-----------|------|
    | 统一并行度导致 attention/MoE 资源错配 | 解耦 attention/MoE 子集群 | 1A6E 配置在低负载下比 8GPU monolithic 高 4.7× TPG |
    | Token-balanced 调度未 minimize a_max | AEBS: expert-count balanced scheduling | a_max 降低 2-5 experts, MoE latency ↓ |
    | Coarse-grained scaling (full replica) | Fine-grained (n_a, n_e) 二维搜索 | 24h trace GPU-h 节省 39% vs SGLang |
    | O(m×n) pairwise 通信 | Two-phase aggregation + MoE-side gating | 2PC+EGate throughput +4-34% vs 2PC+AGate |
    | 无 expert placement 优化 | Activation-aware co-activation minimization | 减少高频共激活 expert 在同一 GPU 的冲突 |

## Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

- baseline方法是什么？
  - **Chinchilla Scaling Laws for Dense Models (Hoffmann et al. 2022)**：L(N, D) = m·N^μ + n·D^ν + c，仅考虑模型参数量和训练 token 数两个变量，未涉及 MoE expert 数量维度。应用于 MoE 时，通常固定 expert 数量和模型规模，按 dense 方式做 compute-optimal 配置。Clark et al. (2022) 的 MoE scaling law L(N_act, Ê) = a·Ê^δ · N_act^(α+γ·ln(Ê)) 仅建模 N_act 和 E 的关系，忽略 dataset size D 的影响。
  - 全栈执行例子（Baseline: compute-optimal dense model, E=1, F=10^20 FLOPs）：
    - **模型推理/训练算法层**：Decoder-only Transformer，标准 FFN（SwiGLU, hidden=3×d_model），无 MoE routing。训练流程：固定 N_act=1.7B, D=9.7B tokens。Optimizer: 固定 LR（无 E-dependent scaling）。
    - **系统框架层**：标准 PyTorch 训练，数据并行/模型并行视 GPU 配置而定。论文未明确说明具体框架。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：标准 dense GEMM kernel。无 MoE-specific sparse computation。
    - **硬件架构层**：PLGrid HPC / Writer.com 计算资源。Memory budget 示例中引用 H100 (80GB)、RTX 4090 (24GB)。论文未明确说明具体 GPU 配置。
  - Baseline 缺陷：
    1. **缺少 E 维度的联合优化**：传统 scaling law 或仅针对 dense (E=1)，或将 E 视为固定超参而非优化变量。无法在给定 compute+memory budget 下系统性地选择最优 E。
    2. **E 与 D 的关系不明确**：Clark et al. (2022) 的形式未含 D 项，无法回答"固定 memory budget 下，增加 E 后应分配多少 token"的问题。
    3. **无 memory optimality 指导**：传统 wisdom 认为 MoE 因 total params 远大于 active params 而 memory-inefficient，实践中倾向于选择 dense 模型以满足 memory 约束。
    4. **超参数 (LR) 无 E-dependent scaling**：不同 E 的 MoE 训练使用相同 LR 策略，导致 suboptimal tuning。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Joint MoE Scaling Laws 方法**：提出联合 scaling law L(N_act, D, Ê) = aÊ^δ · N_act^(α+γ·ln(Ê)) + bÊ^ω · D^(β+ζ·ln(Ê)) + c，将 D 项引入 MoE scaling，并通过 exponent 中的 ln(Ê) 交互项捕捉 E 与 N_act 和 D 的交叉效应。
    1. **D 项的引入**：在 Clark et al. (2022) 的基础上增加 bÊ^ω · D^(β+ζ·ln(Ê)) 项，使 scaling law 能同时描述不同 E 下的 token scaling behavior。ν(E)=β+ζ·ln(Ê) 中 ζ<0，表明 expert 越多 dataset exponent 越负（更多 expert → 需要更多 data）。
    2. **Memory optimality 的数学化**：将 memory constraint 形式化为 N_total ≤ M（或加上 KV-cache），在 {N_act, D, E} 三维空间求解约束优化的 argmin L。推导出 E≤8 的 rule of thumb：固定 total params 的 MoE 用 E× tokens 训练可超越 compute-optimal dense。
    3. **Inference optimality**：将 inference cost (2·N_act·D_inf) 纳入 joint FLOPs budget，揭示 MoE 在 inference 阶段的额外优势（每 token 仅激活 N_act，而非 N_total）。
    4. **LR scaling law for MoE**：实证推导 LR(N_act\e, E) = exp(8.39-0.81·ln(N_act\e)-0.25·ln(E))，E 的负系数表明更多 expert 需要更低 LR，确保不同 E 配置的 fair comparison。
  - 全栈执行例子（对比 Baseline，论文方法在 E=4, F=10^20 FLOPs 下）：
    - **模型推理/训练算法层**：与 baseline 同架构的 Switch MoE（E=4 experts/layer）。根据 joint scaling law，compute-optimal 配置为 N_act=1.2B, D=13.9B tokens（vs dense N_act=1.7B, D=9.7B）。训练用 LR 由 LR scaling law 按 (N_act, E) 计算。关键张量流变化：每 token 经 Router 从 4 个 expert 中选择 top-1，仅激活对应 expert FFN，FLOPs/token = baseline 的 1.2/1.7 ≈ 70%。
    - **系统框架层**：论文未明确说明。推测标准 PyTorch + expert parallelism（因 E=4 较小，routing cost negligible）。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：与 baseline 相同的 dense GEMM kernel，但 expert FFN 的矩阵维度更小（因 N_act 更小），计算量降低。因 E=4 采用 expert parallelism，无复杂 all-to-all 通信。
    - **硬件架构层**：同 baseline 硬件平台。Memory 对比：dense N_total=1.7B 占用约 3.4GB (BF16)；E=4 MoE N_total≈3.0B（根据 Eq.9 估算，专家参数约 4× baseline FFN），但 N_act=1.2B 比 dense 小 → inference 时仅需 2.4GB 活跃参数，KV-cache 也更小（d_model 更小）。
  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | Joint Scaling Law 设计 | 效果 |
    |-------------|----------------------|------|
    | E 非优化变量 | 将 E 作为连续优化维度 | 可在 3D 空间系统性选择最优 (N_act,D,E) |
    | 无 D-E 交互 | 增加 D^(β+ζ·ln(Ê)) 项 | ζ<0 揭示 expert 越多需越多 data |
    | MoE 被认为 memory-inefficient | Memory optimality 分析 | 证明 E=4 MoE 在 1.1B total params 下 loss 低于 dense |
    | Inference cost 被忽略 | Joint train+inference budget | MoE 用 36-61% 更少 FLOPs/token |
    | 无 E-dependent LR | LR scaling law with ln(E) term | 不同 E 的 fair comparison 成为可能 |

## Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

> ⚠️ 目录名为 "KTransformers..." 但 PDF 实际为轮式装载机自主导航论文（ICRA 2025, arXiv:2409.15717），属于机器人控制/AI 决策领域。按 "提出新的算法模型" 归入此库，为弱匹配。

- baseline方法是什么？
  - **传统高层轨迹规划器 + MPC 跟踪控制**：Baseline 采用两步法：
    1. 高层轨迹规划器生成参考轨迹：RRT* based planner [22] 或 optimization-based planner [23] 离线/非实时生成全局路径 → LPV-MPC 或 adaptive MPC 作为参考轨迹跟踪控制器。
    2. 直接轨迹优化（论文 baseline）：使用 CasADi + IPOPT 求解非线性轨迹优化问题（Eq. 27-28），T=25s horizon, direct collocation 离散化（采样 200ms），目标函数为 p-norm 形式的误差积分。求解时间 >5s（AMD Ryzen 3900x desktop CPU），无法实时运行。
  - **Baseline 缺陷**：
    1. **非实时 planners 导致次优性**：离线规划器（如 optimization-based planner in [23]）生成轨迹后，因建模误差实际执行会出现偏离，而 planner 无法及时更新 → 沿过时轨迹跟踪导致次优。实时 planners（如 RRT* in [22]）通过采样/离散化保证实时性，但轨迹高度次优。
    2. **短 horizon MPC 缺乏规划能力**：MPC 若单独使用（无高层 planner），受限于预测 horizon（如 N=10, 2s），无法求解需要远见（>10s ahead）的复杂规划任务。baseline 轨迹优化用 T=25s 但无法实时（>5s 求解时间）。
    3. **RL actor 无法直接部署**：纯 RL policy 在仿真中可完成任务，但无法考虑执行器限幅和其他约束，直接部署不安全。
  - 全栈执行例子（Baseline 轨迹优化，Eq. 27-28）：
    - **模型推理/训练算法层**：构建 NLP 目标函数 ∫[0→T] ∜e(t) + β² + β̈² + a_f² dt，使用 direct collocation + IPOPT 求解 → 输出全程轨迹。论文未明确说明。
    - **系统框架层**：CasADi（符号优化框架）→ IPOPT（内点法 NLP solver）。论文未明确说明其他系统级组件。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明。CasADi/IPOPT 内部数值计算（矩阵分解等）为标准 CPU 计算。
    - **硬件架构层**：AMD Ryzen 3900x desktop CPU。离线求解，无实时性要求。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Goal-Conditioned Actor-Critic MPC**：将 RL 训练出的 critic 作为 MPC 的 cost function，使短 horizon MPC 继承 RL agent 的长期规划知识。
  - **核心设计选择与 Baseline 缺陷的直接映射**：
    1. **Critic as Terminal + Stage Cost**：RL critic L_ψ(s,a,g) 经离线训练后编码了"从当前状态到 goal 的 long-term optimal value"（Eq. 15: 折扣无限 horizon 期望 cost 和）。作为 terminal cost 时，即使 MPC horizon N=10 (2s)，也能通过 terminal cost 评估后续无限 horizon 的代价 → 等价于"MPC 有隐式长 horizon 规划能力"。Stage cost 通过二阶 Taylor 近似（Eq. 25）提供逐步引导，解决"非终端阶段犹豫行为"。
    2. **Lyapunov 稳定性保证**：使用 ALAC 算法训练 critic 满足 sampling-based Lyapunov 条件（Eq. 3a-3c）→ critic 值沿系统轨迹递减，保证 mean cost stability → MPC 继承稳定性。
    3. **Gradient Penalty for MPC 优化**：在 critic loss 中加入 ρ·E[(1-||∇L_ψ||₂)²]（Eq. 16），鼓励 critic 为 1-Lipschitz → 为下游 SQP-RTI solver 提供更平滑的优化景观，缓解前人工作中 "因 NN critic 高度非线性导致优化困难" 的问题 [17]。
    4. **输入延迟补偿 + 约束强制执行**：MPC 将初始状态向前传播 200ms 补偿执行器延迟。MPC 强制状态/控制约束（Eq. 6, 21），克服纯 RL actor 无法处理约束的问题。
  - 对应解决 Baseline 缺陷：
    - Baseline 非实时 planner → method 将 planner 知识编码到 critic 中，MPC 每步 <100ms（Jetson），实时。
    - Baseline 短 horizon MPC 缺乏规划能力 → critic 提供 implicit 长 horizon 规划（terminal cost 等价于无限 horizon value）。
    - Baseline RL 不安全 → MPC 强制执行约束（状态/控制/障碍物）。
    - Baseline 轨迹优化 >5s → MPC solver <100ms per iteration（10-20× faster on weaker hardware）。
  - 全栈执行例子（Actor-Critic MPC，Avant 635 真机）：
    - **模型推理/训练算法层**：离线 RL 训练（PyTorch + stable-baselines3）→ 在线 MPC（CasADi + Acados + L4CasADi）。每 200ms 迭代：编码状态 → Critic forward（terminal cost）+ Taylor 近似（stage cost）→ SQP-RTI 求解 NLP → 输出 β̇, v_f 给低层控制器。
    - **系统框架层**：CasADi（NLP 建模）+ Acados（SQP-RTI solver）+ HPIPM（QP solver）+ L4CasADi（NN 集成到 CasADi 符号表达式）。低层控制器跟踪 MPC 输出的速度指令。论文未明确说明操作系统级框架。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明。NN 推理（PyTorch on Jetson GPU）和 SQP solver（CPU）并行在 Jetson AGX Orin 上运行。
    - **硬件架构层**：NVIDIA Jetson AGX Orin（嵌入式平台，ARM CPU + NVIDIA GPU on SoC）。真机液压/柴油执行器延迟约 200ms。Baseline 用 desktop AMD Ryzen 3900x。
  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | Actor-Critic MPC 设计 | 效果 |
    |-------------|----------------------|------|
    | 非实时 planner (>5s) | Critic 编码规划知识 → MPC 每步 <100ms | 实时控制 |
    | 短 horizon 缺乏规划 | Terminal cost = critic (无限 horizon value) | N=10 等价长 horizon 规划 |
    | RL actor 不安全 | MPC 强制约束 (Eq. 6, 21, 22) | 安全真机部署 |
    | NN critic 难以优化 (前人工作) | Gradient penalty (1-Lipschitz) | 平滑 MPC 优化景观 |
    | 执行器延迟 200ms | 前向传播初始状态 200ms | 延迟补偿 |
    | 无逐步指导（前人工作仅 terminal cost） | 二阶 Taylor stage cost (Eq. 25) | 消除犹豫行为 |

## LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

- baseline方法是什么？
  - **Standard MoE**（以 Qwen3-235B-A22B / DeepSeek-v2-lite 为代表的 fine-grained MoE）：每个 MoE layer 包含 N=128 routed experts + S=2 shared experts，每个 expert 的 FFN 在原始 hidden dimension d=4096 中操作，中间维度 m=2688。Router 计算 softmax(W_r·x) 后 top-K=6 选择 expert，加权组合输出。所有 expert 共享相同的输入空间维度 d，expert 参数规模 3×d×m per expert。Standard MoE 的设计逻辑是基于高层次的稀疏性论证（仅激活少数 expert 减少 FLOPs），未充分考虑硬件实际瓶颈。
  - Baseline 核心缺陷：
    1. **Memory Bandwidth 瓶颈（低延迟场景）**：小 batch 推理中 MoE expert 计算是 memory-bound。GB200 系统 arithmetic intensity ≥1250 FLOPs/byte 才 compute-bound，Qwen3-235B 在 latency-critical 部署中 t_exp<1418，operating point 在 memory-bound 区域。
    2. **All-to-All 通信瓶颈（高吞吐场景）**：大 batch 下 MoE layer 通信/计算时间比 ~9:1（GB200 NVL72 + Qwen3-235B），all-to-all 是主要瓶颈。
    3. **专家组合空间有限**：N=128, K=6 仅 C(128,6) 种组合，限制 token 级组合稀疏性带来的表达能力。
    4. **d 过度预留给 routed experts**：虽然任务特征秩 r_eff << d，但所有 routed expert 的 input dim 保持为 d。
  - 全栈执行例子（Baseline Standard MoE, 95BT-8BA, EP on multi-GPU）：
    - **模型推理算法层**：Attention(DP/TP) → Router top-K=6 → 6 experts 在 d=4096 做 FFN(d×m, d×m, m×d) → gate-weighted combine → shared experts。Router/shared/attention 均在 d。
    - **系统框架层**：Expert Parallelism (EP), All-to-All dispatch/combine, vLLM serving。Data parallelism for attention。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL backend + vLLM CUDA kernels。
    - **kernel 调度层**：NCCL All-to-All(FP4+BF16) + CUTLASS GroupedGEMM + cuBLAS GEMM。通信 ∝ K·d=6×4096。Expert 权重加载 ∝ d·m。
    - **硬件架构层**：H100 GPUs / GB200 NVL72。All-to-All 占 ~90% 总时间(throughput)。Memory BW bound(latency)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LatentMoE 方法**：将 routed experts 输入从 d 压缩到 ℓ（latent dim），节省的 memory BW/communication 预算重新投资到 expert 数量和 top-K，实现 accuracy per FLOP/parameter Pareto 改进。核心创新：将"浪费"在过大 hidden dim 上的资源转为 expert diversity 提升。
    1. **Latent Space Projection (W_↓, W_↑)**：所有 routed experts 共享 W_↓∈R^{ℓ×d} 和 W_↑∈R^{d×ℓ}。Expert 权重从 3×d×m 降至 3×ℓ×m (α=4× reduction)。All-to-All 在 latent space ℓ 中进行。
    2. **Expert Count Scaling (N→N'=αN)**：用节省的参数预算将 experts 从 128 扩至 512。C(512,K) >> C(128,K)。
    3. **Top-K Scaling (K→K'=αK, ℓ-MoE_acc)**：top-K 从 6 扩至 24。通信量 K'·ℓ = 24×1024 = baseline K·d = 6×4096 (iso-communication)。C(αN,αK) ≥ C(N,K)^α，指数级扩大组合空间。
    4. **压缩比下界验证**：α sweep 实验验证 r_eff≈d/4，α=4 safe, α=8 精度塌缩。
    5. **Shared Experts 保持原始空间**：非瓶颈，保持 d 操作。
  - 对应解决 Baseline 缺陷：
    - Memory BW bottleneck → Expert 权重 ℓ×m << d×m (per expert ↓α×)，总 loading 在 ℓ-MoE_acc 不变、ℓ-MoE_eff ↓。
    - All-to-All bottleneck → 通信在 latent space (msg size ↓α× per token)，K'=αK 使 ℓ-MoE_acc 总通信不变、ℓ-MoE_eff ↓α×。
    - 组合空间有限 → N:128→512, K:6→24, C(512,24) >> C(128,6)。
    - d 过度预留 → 压缩到 ℓ=d/4，验证 r_eff 下界。
  - 全栈执行例子（LatentMoE ℓ-MoE_acc, 95BT-8BA, d=4096, ℓ=1024, α=4）：
    - **模型推理算法层**：Attention(d=4096) → Router(原始空间) → top-K'=24 → Shared W_↓[1024,4096] down-project → 24 experts 在 latent space ℓ=1024 做 FFN(W_g/W_up[2688,1024], W_FC2[1024,2688]) → combine → Shared W_↑[4096,1024] up-project → Shared Experts(d) → output。
    - **系统框架层**：EP, All-to-All in latent space ℓ。vLLM FP8。All-to-All msg size = ℓ (vs d baseline), token count = K' (vs K baseline), 总通信量相同(ℓ-MoE_acc)。
    - **编译框架层**：TensorRT-LLM v1.2.0+ 支持。提议优化: (1) 分离 CUDA streams for routed/shared experts, (2) CUTLASS small-inner-dim GEMM kernels for latent-space experts。
    - **kernel 调度层**：W_↓/W_↑ GEMM (modest, ~9% overhead at trillion scale) + Routed Expert GEMM(ℓ×m, 4× smaller) + Shared Expert GEMM(d×m) + All-to-All(FP4+BF16 in ℓ)。
    - **硬件架构层**：H100 GPUs (实测), GB200 NVL72 (投影)。High concurrency 下 throughput ↓≤6%。Trillion-scale: Kimi-K2-1T-LatentMoE 比 iso-accuracy Kimi-K2-1.35T 快 1.24×-3.46×。
  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | LatentMoE 设计 | 效果 |
    |-------------|---------------|------|
    | Memory BW bottleneck | ℓ·m << d·m per expert | Weight loading ↓α× per expert |
    | All-to-All bottleneck | Communication in latent space ℓ | Msg size per token ↓α× |
    | d 过度预留 | 压缩到 ℓ=d/α, α=4 | r_eff≈d/4, info loss negligible |
    | C(128,6) 有限 | N→512, K→24, C(512,24) | 指数级增长组合多样性 |
    | 仅优化 offline throughput | Accuracy per FLOP + per parameter | 两种 deployment 均受益 |
    | 压缩导致训练不稳定 | Expert scaling N→αN 补偿 | 恢复 baseline 稳定性 |
  - **核心设计洞察**：Standard MoE 的 hidden dim d 在 routed expert 层面过度配给——任务特征秩 r_eff<<d。N 和 K 受制于 d·m (memory BW) 和 K·d (communication) 无法扩展。LatentMoE 通过解耦 input space (d→ℓ) 获得三个自由度: (1) 降低 input dim → memory BW per expert ↓; (2) 增加 N → 更细粒度专业化; (3) 增加 K → 更大非线性预算 U_eff=K·m。在 iso-inference-cost 下将"浪费"在过大 d 上的资源转为 expert diversity 和 nonlinearity budget，系统性提升 accuracy-efficiency Pareto frontier。五项 Design Principles 构成 hardware-software co-design 框架使这一重新分配有据可依、有界可循。

## Layerwise Recurrent Router for Mixture-of-Experts

- baseline方法是什么？
  - **标准 SMoE (Switch Transformer)**：每层路由由独立的线性层 + softmax + top-k 构成。每层 router 参数 G_i ∈ R^(h, N)，对输入 token hidden state x_i 计算 gating score 后选择 top-k experts 进行 FFN 计算。不同层的 router 独立决策，不共享跨层路由信息。论文指出 (1) token 的 hidden state 虽然通过残差连接可以隐式传递跨层信息，但路由相关信息可能被 LM loss 的优化"淹没"；(2) 单个线性层 router 表达能力有限，token embedding 容易 collapse 到 expert embedding 附近（representation collapse）；(3) 早期 router 梯度主要来自 load balancing loss（而非 LM loss），导致早熟收敛到次优路由策略。
  - 全栈执行例子（Baseline SMoE，decoder-only transformer）：
    - **算法 Pipeline**：token x_i → Linear(G_i) → softmax → top-k → sparse FFN(selected experts) → output y_i → 残差 + LayerNorm → x_{i+1}。每层独立路由，无跨层信息。
    - **系统框架**：论文未明确说明，小实验用 PyTorch 原生实现，大实验用 Megablocks 框架。
    - **编译框架**：论文未明确说明。
    - **Kernel 调度**：论文未明确说明。Megablocks 本身提供 block-sparse kernel 加速 MoE 计算。
    - **硬件架构**：NVIDIA A100 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **RMoE**：在每层 router 前插入跨层共享的 GRU 单元，将路由决策从独立逐层计算改为跨层循环依赖。核心公式：x_i' = Proj_i(x_i), h_i = GRU(x_i', h_{i-1}), y_i = sum_n g_n(h_i; G_i, k) * E_n(x_i)。同时额外提供 Recurrent Gradient 路径优化 router 训练。
  - 对应解决 Baseline 缺陷：
    1. **跨层信息共享**：GRU 显式传递历史路由决策（h_{i-1} → h_i），使当前层 router 知道 token 在之前层被分配到哪些 experts，支持跨层协作。实验证明 RMoE 的跨层 mutual information 显著高于 SMoE/XMoE/HyperMoE。
    2. **Representation collapse 缓解**：Proj_i 投影 + GRU 将 hidden state 从 expert embedding 空间分离，类似于 XMoE 的低维投影策略，减少 token embedding collapse。
    3. **Router 梯度优化**：GRU 提供额外的 Recurrent Gradient 路径（跨层反向传播），使 router 优化不再被 LB loss 主导。实验显示 SMoE 的 linear router 梯度早期被 LB loss 主导，而 RMoE 的 GRU router 梯度持续由 LM loss 主导，达到更好的 LM/LB 权衡。
    4. **Moderate flat gating scores**：跨层信息共享使 gate score 分布呈现适度平坦（高熵但非随机），在 exploration vs exploitation 之间取得更好平衡，避免早熟收敛。RMoE 的 Top-1/Top-2 比率和 Outer Balance 均显著低于 SMoE。
    5. **正交兼容**：GRU 路由作为一个新的计算阶段，可与 XMoE、DeepSeekMoE 等现有方法组合，实验验证 XMoE+GRU router 在 3 种配置下均优于纯 XMoE。
  - 全栈执行例子（RMoE，decoder-only transformer）：
    - **算法 Pipeline**：token x_i → Proj_i(x_i) 降维 → x_i' → GRU(x_i', h_{i-1}) 结合历史 → h_i → Linear(G_i) + softmax + top-k → sparse FFN → y_i → 残差 + LayerNorm → x_{i+1}。跨层 GRU 提供前向路由信息和反向 Recurrent Gradient。
    - **系统框架**：论文未明确说明，小实验用 PyTorch 原生实现，大实验用 Megablocks 框架。
    - **编译框架**：论文未明确说明。
    - **Kernel 调度**：论文未明确说明。
    - **硬件架构**：NVIDIA A100 GPU。RMoE 仅增加 ~3.5M 参数（相对于 0.91B 模型），训练速度仅从 48.87 s/step 增加到 49.07 s/step（+0.4%），GPU 内存从 48.00GB 增加到 48.69GB（+1.4%），开销可忽略。

## Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement

- baseline方法是什么？
  - **DeepSpeed MoE (DS) with checkpoint-based fault tolerance**：传统 Expert Parallelism (EP) 将 experts 等分到 EP size 个 GPU 上，每个 expert 只有一个 replica。故障发生时，所有 GPU 必须等待故障节点被替换（可能需要数小时到数天），然后从 checkpoint 重新开始训练。每 50 steps 进行一次 checkpoint 保存到 NFS server。EP 要求 GPU 数是 EP size 的整数倍，故障后可能有多余 GPU 空闲。All-to-all 通信在 EP group 内进行，使用 padded all-to-all（padding 到最大的 expert token 数）。
  - 全栈执行例子（DS baseline，GPT-L 16 experts，EP size=4，10 GPU 集群）：
    - **算法 Pipeline**：token → Gate(top-1) → route to expert → Expert FFN → combine。无 adaptive allocation，每个 expert 1 replica。
    - **系统框架**：DeepSpeed MoE (PyTorch + DeepSpeed v0.13)。EP size=4 → 8 GPU 使用（4 GPU/EP group × 2 groups），10 GPU 中 2 GPU 闲置。Data parallelism across EP groups。
    - **编译框架**：论文未明确说明。PyTorch eager mode。
    - **Kernel 调度**：NCCL all-to-all collective（padded），在 EP group 内固定大小传输。Training step: all-to-all dispatch (padded) → expert computation → all-to-all combine (padded) → checkpoint save every 50 steps (blocking I/O to NFS server, ~10s overhead for GPT-L)。
    - **硬件架构**：5 nodes × 2× RTX 3090 GPU/node，100 Gbps Mellanox ConnectX-5 NIC。故障恢复：失败后 NCCL timeout → 等待 replacement node（hours）→ 从 NFS 加载 checkpoint → 重启 NCCL groups → 继续训练。丢失当前 step 到上次 checkpoint 间的所有训练进度。

  - **DS(FT) baseline**：与 DS 相同的 EP 分配，但使用 Lazarus 的 reconfiguration runtime 进行快速故障恢复。如果完整 replica of all experts 仍然存在，则重新配置 EP groups 并从其他节点获取 expert states。否则必须从 checkpoint 重启。GPT-L 上 EP size=4，只有 <8 GPU 时无法利用超过 4 GPU，且丢失超过一个 EP group 时需 checkpoint 重启。

  - Baseline 的核心缺陷：
    1. **无弹性 (Inelastic)**：必须等待 replacement nodes 才能继续训练，无法利用剩余 GPU 继续推进。
    2. **GPU 浪费**：需要 GPU 数为 EP size 的整数倍，如 GPT-L 的 EP size=4 且 10 个 GPU 时只能使用 8 个。
    3. **Expert load 不均衡未处理**：传统 EP 等分 experts，不根据 expert 负载分配更多 replicas 给 popular experts，导致 GPU 间计算不均衡（up to 87% tokens routed to 2 experts）。
    4. **Checkpoint 开销巨大**：随着模型增大，checkpoint 和 restart 开销越来越显著，频繁故障下可能占据 >50% 训练时间。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Lazarus**：通过自适应 expert replica 分配（根据 expert 负载动态调整 replica 数）+ 可证明最优的 MRO expert placement 算法（最大化故障恢复概率）+ Flexible Token Dispatcher CUDA kernel（处理非对称 placement 下的高效 token dispatch）+ 高效 reconfiguration runtime（最小化迁移开销），实现 MoE 训练的高吞吐、高弹性、强容错。
  - 对应解决 Baseline 缺陷：
    1. **解决无弹性**：自适应 replica 分配使得每个 expert 有 ≥f 个 replicas（f 为容错阈值），只要每个 expert 至少有一个 replica 存活即可恢复训练。所有剩余 GPU 全部利用，无需等待 replacement nodes。
    2. **解决 GPU 浪费**：不要求 GPU 数为 EP size 整数倍，N 个 GPU 全部使用。任何 GPU 数下均能完全利用。
    3. **解决 Expert load 不均衡**：Eq. 1 的 r_e 分配公式使 popular experts 获得更多 replicas（r_e ∝ t_e），更多 computation resource 给热门 experts，加速训练。无故障时 Lazarus 的 GPT-M 吞吐 45 samples/s vs DS 的 34 samples/s。
    4. **解决 Checkpoint 开销**：故障恢复无需从 checkpoint 重启，expert states 通过 NCCL send/recv 从 other nodes 并行获取（如 GPT-L 160 expert states 仅需 7.6s 传输），总 reconfiguration 时间 20~40s，远小于 checkpoint restart。

  - 全栈执行例子（Lazarus，GPT-L 16 experts，10 nodes，f=2）：
    - **算法 Pipeline**：token → Gate(top-1) → route to expert e → **Flexible Token Dispatcher CUDA kernel**（Algorithm 1：计算每 rank 处理容量 → 优先本地处理 → 按剩余容量比例分发 overflow tokens → reshuffle activations）→ **flexible all-to-all (no padding)** → Expert FFN（popular experts 有多个 replicas 并行处理）→ flexible combine → output。Expert loads 周期性收集（每 200 steps rebalance），动态调整 r_e。
    - **系统框架**：Controller（CPU node，Python async）管理集群，Agent（per GPU node）relay。PyTorch + DeepSpeed components + Lazarus runtime。NCCL groups: expert gradients all-reduce + non-expert gradients all-reduce + all-to-all (flexible, non-padded)。Controller 每 200 steps 或故障时重新计算分配和 placement（<100ms CPU 计算）。
    - **编译框架**：论文未明确说明。
    - **Kernel 调度**：Flexible Token Dispatcher CUDA kernel（Algorithm 1，对所有 E 个 experts 和 N 个 ranks 并行执行）→ flexible all-to-all collective（各 rank 发送/接收不同数量的 tokens，无 padding）→ expert computation → all-gather T_{e,j} (E integers per rank, negligible overhead) → periodic rebalance。流水线：all-to-all 与 computation 可在不同 streams 上执行。故障时：NCCL timeout (10~20s) → reconfig NCCL groups (5~15s) → NCCL batched send/recv 并行状态迁移 → 恢复训练。
    - **硬件架构**：5 nodes × 2× RTX 3090 GPU (10 emulated nodes)，100 Gbps NIC。Lazarus 将 expert replica slots per GPU 设为 6（GPU memory limit）。MRO placement: 将 16 experts 按 c=6 分为 ⌈16/6⌉=3 组，每组内最大化 expert overlap。故障时重新路由到剩余 alive nodes 上的 expert replicas。如 4 node failures：Lazarus 41% recovery prob vs spread placement 12%。

## Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

- baseline方法是什么？
  - **标准 Expert Parallelism (EP)**：将 MoE 模型的 N 个 expert 等分到 P 个 GPU 上（每 GPU 持有 M=N/P 个 expert 权重）。推理/训练时，每 GPU 的 tokens 通过 router 计算 top-K expert 选择，经由 All-to-All 通信将 token 分发到对应 expert 所在的 GPU（dispatch），各 GPU 用本地 expert 权重执行 FFN 计算，再通过 All-to-All 将结果合并回原始设备（combine）。EP 假设所有 GPU 的计算负载均衡，但在实际 imbalanced routing 下，少数 expert 可能接收大多数 token，导致持有这些 expert 的 GPU 过载（高延迟 + OOM 风险），而其他 GPU 空闲。
  - 全栈执行例子（标准 EP，gpt-oss-120b MoE layer, 128 experts / K=4 激活, EP=8, 8×H200）：
    - **算法 Pipeline**：token batch → Router (u^T W_r) → top-4 gating → softmax → per-expert token assignment。MoE 输出: h = Σ g_i · SwiGLU_i(u)。标准 EP 不做任何负载均衡干预，tokens 严格按 routing indices 分发。
    - **系统框架**：PyTorch distributed + NCCL。EP=8, 每 GPU 16 experts。Alg. 1 dispatch_combine: sort routing indices → index_select to reorder tokens → All-to-All dispatch (NCCL) → 本地 Grouped-GEMM (cuBLAS, 16 experts per GPU) → All-to-All combine (NCCL reverse) → reverse_sort → reshape → sum over K。不使用任何 serving 框架（vLLM/SGLang），基于 PyTorch 原生分布式训练范式。
    - **编译框架**：论文未明确说明。PyTorch eager mode, cuBLAS GEMM。
    - **Kernel 调度**：NCCL All-to-All collective（padded 或 unpadded）+ cuBLAS GEMM kernel per expert。GEMM 效率随 B_i (per-expert token 数) 增大而提高：少量大 GEMM > 大量小 GEMM。论文 Fig. 8 显示 cuBLAS 独立 GEMM 优于 Triton fused grouped-GEMM。
    - **硬件架构**：8× NVIDIA H200 GPU，单节点 NVLink/NVSwitch 互联。B_p=32K tokens per GPU。95% token 集中在 1 个 expert 时：所有 token 汇聚到 1 个 GPU → GPU 处理时间 4.6× 慢 → peak memory 4× 增长 → 可能 OOM crash。

  - Baseline 的核心缺陷：
    1. **无负载均衡**：标准 EP 严格按 routing indices 分发 token，对 expert 负载不均衡无任何干预。极端情况下 95% token 汇聚到 1 个 GPU（持有热门 expert）。
    2. **内存不可控**：过载 GPU 的 token buffer 随不均衡度线性增长（Eq. 4），peak memory 可达平衡时的 4×，直接导致 OOM crash。
    3. **GPU 计算资源浪费**：过载 GPU 成为 straggler，决定整个 EP group 的 collective latency（max_i time-of-GPU_i），其余 GPU 计算完成后空闲等待。
    4. **Naive 缓解方案均有问题**：减小 batch size 降低吞吐量；chained gradient checkpointing 仍有硬内存上限；EPLB (Liu et al., 2024) 复制热门 expert 增加内存、仅用于推理且极端情况仍 OOM；预留额外内存 (Huang et al., 2024) 增加 CPU/GPU 内存开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Least-Loaded Expert Parallelism (LLEP)**：在标准 EP 的 dispatch 阶段之前，通过 LLA 贪心算法预先计算全局 optimal token-to-GPU 分配方案，将超载 GPU 的多余 token 和对应的 expert 权重（Wi）传输到负载最轻的 GPU 上执行，使所有 GPU 几乎同时完成计算。
  - 对应解决 Baseline 缺陷：
    1. **解决无负载均衡**：LLA 算法（Alg. 2+3）按 expert 负载降序分配，优先让原生 GPU 计算本地 expert 的 token（减少权重传输），超出容量 m_α 的部分溢出到 g_a+g_p 最小的 GPU。α 因子控制每 GPU 容量上限，m 控制最小 GEMM 效率阈值。最终所有 GPU 负载接近均值，消除 straggler。
    2. **解决内存不可控**：通过 m_α = α · Σl_i / P 硬限制每 GPU token 数，每个 expert 的 token 被拆分到多个 GPU 并行处理。peak memory 保持稳定（图 4 bottom row），不随不均衡度增长，最多节省 5× 内存。
    3. **解决 GPU 资源浪费**：LLEP 将计算负载均匀分布到所有 P 个 GPU，最小化 max_i time-of-GPU_i，实现 collective latency 最小化。在 batch size 越大时加速越明显（图 6a），因为大 batch 饱和各 GPU 容量后 LLA 开销被摊薄。
    4. **优于 naive 缓解方案**：不降低 batch size（反而可利用均衡内存提高 batch size）；不做 expert 复制（无额外内存）；支持训练和推理；保持 exact computation（不改变模型输出）；支持 backward pass（梯度回流）。

  - 全栈执行例子（LLEP，gpt-oss-120b MoE layer, 128 experts / K=4, EP=8, 8×H200，假设 80% token 集中到 4 个 expert）：
    - **算法 Pipeline**：MoE 数学计算与标准 EP 完全相同（exact computation）。Router → top-K gate → 但 token-to-GPU 分配不再严格遵循 routing index，而是按 LLA 计划重新分配。对每个被路由到 expert e 的 token，由 LLA 决定由哪个 GPU 执行该 expert 对该 token 的 FFN 计算。所有 expert 的输出被正确聚合且 gate weight 不变。
    - **系统框架**：PyTorch distributed + NCCL。Alg. 4 LLEP dispatch_combine: 收集全局 expert 负载 l → max(l)/mean(l) ≥ λ=1.3 → 触发 LLA → Python CPU 侧执行 Alg. 2 (LLA) + Alg. 3 (LLAS) 计算分配计划 A + 权重传输计划 W → 按 A 构建 per-GPU token chunks（含 foreign expert tokens）→ All-to-All dispatch (NCCL) → P2P 权重传输 W_i: overloaded GPU → underloaded GPU (NCCL P2P Send/Recv) → Grouped-GEMM (cuBLAS, native+foreign experts) → All-to-All combine (NCCL reverse) → reverse_sort → reshape → sum。Backward: foreign expert 梯度通过 P2P 返回原生 GPU 累加。
    - **编译框架**：论文未明确说明。PyTorch eager mode, cuBLAS GEMM。
    - **Kernel 调度**：NCCL All-to-All (tokens) + NCCL P2P (expert weights) + cuBLAS GEMM (per expert)。P2P 权重传输开销取决于 D×H 大小——hidden size 越大，每个 expert 权重传输成本越高，但同时 GEMM 效率也越高（Fig. 7b 显示 LLEP 在更大 hidden size 下加速比更高）。LLA 算法时间为 Python CPU 计算，token 量级大时开销可忽略。
    - **硬件架构**：8× NVIDIA H200 GPU，单节点 NVLink/NVSwitch。参数设定：λ=1.3, α=1, m=1024。80% token 集中在 4/128 experts → 持有这些 expert 的 GPU(s) 超载 → LLA 将多余 token + 对应 W_i 溢出到负载最轻的 GPU → 最终 8 个 GPU 负载接近 = 总 token 数/8。加速效果：MoE layer 3-5× speedup（vs 标准 EP），全模型 gpt-oss-120b 1.88× speedup。

## Llama 3 Meets MoE: Efficient Upcycling

- baseline方法是什么？
  - **Baseline**：Llama 3-8B dense 模型继续进行 Continued Training (CT)，即不进行 upcycling，直接使用 dense 模型在同量数据上继续训练。全栈执行例子：
    - **算法 Pipeline**：标准 Transformer decoder-only，每层包含 Multi-Head Self-Attention + SiLU-gated FFN，8B 参数全部激活，单 token 前向 FLOPs 约 4.7e14
    - **系统框架**：Megatron-Core + NeMo 分布式训练框架，使用标准 TP+PP+DP 并行策略
    - **编译框架**：论文未明确说明
    - **Kernel 调度**：标准 cuBLAS/cuDNN kernel，无 MoE-specific kernel
    - **硬件架构**：512× H100 GPUs，bfloat16 训练
  - Baseline 缺陷：(1) 扩展模型容量需等比增加计算量（参数翻倍 ≈ FLOPs 翻倍）；(2) 已投入的预训练 GPU 小时无法复用，每次扩展需从头训练；(3) Dense 模型在给定 compute budget 下存在性能天花板。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：Sparse Upcycling——从 Llama 3-8B dense checkpoint 出发，将部分 FFN 层复制 N=8 次初始化 expert（总计 34.4B 参数，仅 11.8B 激活），添加随机初始化 router，仅用 100B tokens (<1% 预训练 compute) 完成 MoE 训练。全栈执行例子：
    - **算法 Pipeline**：
      1. Upcycling：复制 FFN 权重 N 次为 experts，随机初始化 router（Mixtral-type: KeepTopK→Softmax，确保初始输出与 dense 一致）
      2. Expert Capacity Factor (CF=4)：控制每个 expert 最大处理 token 数，隐式引入正则化，防止 expert 过拟合
      3. 仅 100B tokens 训练（vs dense pre-training 数 T tokens），11K GPU hours（vs 估计 1.6M GPU hours from scratch）
    - **系统框架**：
      1. Online Upcycling in NeMo：按并行配置分片 dense checkpoint，各设备独立 upcycle，无需跨设备权重复制
      2. MoE Parallel Folding：解耦 Attention (TP×CP×DP×PP) 和 MoE (Expert-TP×EP×Expert-DP×PP) 并行映射，将通信密集操作折叠到 NVLink 高带宽域
      3. 5-D Hybrid Parallelism (TP+EP+PP+CP+DP ZeRO-1)
    - **编译框架**：论文未明确说明
    - **Kernel 调度**：Megatron-Core 提供的 AllToAll-based token dispatcher（TopK=1-4 时优于 AllGather-based）；对早期训练阶段 MoE 层启用 recomputation
    - **硬件架构**：512× H100 GPUs，bfloat16 训练，NVLink intra-node 高带宽通信 + InfiniBand inter-node
  - **解决 Baseline 缺陷的映射**：
    - 缺陷1（容量扩展与计算量等比增加）→ Upcycling 后 34.4B 总参数仅需 1.6× FLOPs（11.8B 激活参数），实现参数-计算解耦
    - 缺陷2（预训练投入不可复用）→ 直接复用 Llama 3-8B 预训练权重，100B tokens (<1% compute) 完成训练，11K vs 1.6M GPU hours
    - 缺陷3（dense 性能天花板）→ E8T2 MoE 在 MMLU 0-shot 提升 2%（65.20→64.00 in 5-shot, 62.10→64.10 in 0-shot），整体平均提升 ~1.2%
  - **关键设计选择**：
    - CF=4 为 accuracy-MFU 最佳平衡点（MMLU 0-shot 64.0 vs CF=1 的 63.7，MFU 39.4% vs 46.8%）
    - Mixtral-type router 比 ST-type 收敛更快（初始 loss 更低，因 upcycling 后初始输出与 dense 一致）
    - MoE Parallel Folding 下 TP1CP2EP8 配置达 46.8% MFU（128 H100）

## Load Balancing Mixture of Experts with Similarity Preserving Routers

- baseline方法是什么？
  - **Load Balancing Loss (LBL) [Fedus et al. 2022]**：MoE 训练中广泛使用的辅助负载均衡损失 L_LBL = α · E · Σ f_i · P_i，其中 f_i 为 expert i 被路由的 token 比例、P_i 为平均路由概率。LBL 通过鼓励接近均匀的 expert 分布来防止路由 collapse（所有 token 被路由到少数几个 expert）。该方法是当前 SOTA MoE 模型的标准组件（OLMoE, DeepSeek-V3, DBRX 等）。LBL 的缺陷：
    1. **知识冗余**：强制 uniform distribution 导致不同 expert 接触到相似的 token 集合，模型容量被浪费于在多个 expert 中学到冗余知识
    2. **路由不稳定**：训练早期 embedding 快速变化 + near-uniform 分配 → 微小输入扰动可导致 token 被重新分配给不同 expert → 进一步加剧 expert 间知识冗余
    3. **路由不一致**：相似 token 可能被路由到完全不同的 expert，使模型无法利用 token 间语义结构
    4. **需要超参调优**：损失系数 α 需要在主任务损失和负载均衡之间平衡，且对 batch size 敏感
  - 全栈执行例子（Baseline LBL，MoE-L 在 8× AMD MI300X 上训练一个 Transformer decoder layer 的前向传播）：
    - **算法 Pipeline 层**：Token 输入 x ∈ R^{B×S×1536} → Router R ∈ R^{1536×32} → softmax → top-4 选择 → SwiGLU expert FFN（D_E=1536）→ expert 输出加权和。LBL 作为辅助损失加入 L_total = L_lm + 0.01 · L_LBL，鼓励 32 个 expert 的 f_i · P_i 接近均匀。训练早期 LBL 可能主导梯度，导致路由决策不稳定。
    - **系统框架层**：PyTorch + OLMo 训练框架 + DDP（Distributed Data Parallelism）。数据来自 DCLM-pool-400m-1x，cl100k_base tokenizer。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **Kernel 调度层**：PyTorch bfloat16 GEMM 用于 Router (1536×32)、Expert FFN (1536×1536)。无需自定义 CUDA kernel。
    - **硬件架构层**：8× AMD MI300X 192GB per node 或 8× NVIDIA A100 40GB。训练 FLOPs ≈ 2.84×10^20 (MoE-L)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **SIMBAL 方法**：
    1. **正交 Router Loss**：L_orth = ||R^T R - I_E||_1，鼓励 Router 权重矩阵 R ∈ R^{D_M×E} 逼近正交矩阵（正交矩阵保留点积即角度，因此保留 token 间成对相似性）。相似 token 得到相似的 expert 分布 → 一致的 routing 行为 → 减少冗余。
    2. **Loss-based 而非显式参数化**：相比于通过 QR 分解强制正交（计算昂贵、频繁重正交化、数值不稳定），SIMBAL 使用辅助损失实现软约束，直接在 bfloat16 中训练，无需 float32 转换。
    3. **数据无关**：L_orth 仅依赖 Router 权重，不依赖数据分布或 batch size，消除 LBL 对 batch size 的敏感性。
    4. **正交初始化**：使用 Saxe et al. 2014 初始化使 Router 接近正交，加速收敛（或简单执行少量 router-only 训练步也可）。
    5. **PES 指标**：提出 Pairwise Expert Similarity (PES) = mean over token batches of pairwise expert output cosine similarity，作为轻量级 expert 冗余度度量。
  - 对应解决 Baseline 缺陷：
    - 缺陷1（知识冗余）→ 正交 Router 保留 token 相似性 → 相似 token 获得一致 routing → 各 expert 专精于处理特定类型的 token（PES 从 0.0241 降到 0.0028）
    - 缺陷2（路由不稳定）→ 正交 Router 对 input perturbation 更加鲁棒（角度保持 property）→ 训练早期不出现频繁 routing shift → 冗余增长率显著低于 LBL
    - 缺陷3（路由不一致）→ 相似 token 获得相似的 expert 分布 → Router 输出间保持 pairwise angle → 结构化 routing
    - 缺陷4（超参调优）→ SIMBAL 系数不敏感（0.01/0.1/1.0 下 perplexity 13.687/13.685/13.716），无需分布式同步
  - 全栈执行例子（SIMBAL，MoE-L 在 8× AMD MI300X 上训练）：
    - **算法 Pipeline 层**：与 baseline 相同的 MoE forward path，但训练 loss 替换为 L_total = L_lm + 0.1 · ||R^T R - I||_1。Router 从正交初始化开始，每步 optimizer step 后权重被 L_orth 的梯度拉向正交方向。Gram matrix R^T R 的 L2 distance 从 ~0.03 (LBL) 降至 ~2×10^-8 (SIMBAL)——即 Router 高度正交。Token 相似性通过 Router 保留：cos(x1,x2) ≈ cos(x1·R, x2·R)。
    - **系统框架层**：PyTorch + OLMo + DDP，与 baseline 相同。SIMBAL loss 计算仅涉及 Router 权重矩阵（1536×32），计算代价可忽略。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：与 baseline 相同——bfloat16 GEMM。SIMBAL 不引入额外 kernel，仅增加一个 O(E^2·D_M) 的 Gram matrix + L1 norm 计算。
    - **硬件架构层**：8× AMD MI300X 192GB 或 8× NVIDIA A100 40GB。训练吞吐量与 LBL 相当或更好（因 SIMBAL 更快收敛，36% 更少 token）。
  - **关键设计选择**：
    - 选择 orthogonal router 而非 orthogonal experts（OMoE, MOORE）——Router 参数极少（0.018% total params）但编排 billions of parameters，施加正交约束于此更有 leverage
    - 选择 L1 norm（而非 L2/Frobenius）作为 Gram matrix deviation measure——L1 在数值上更稳定
    - 推理时与 expert pruning 的良好协同：SIMBAL 产生 less uniform routing → 低 weight expert 更可安全丢弃 → 7.4% throughput improvement

## LocMoE: A Low-overhead MoE for Large Language Model Training

- baseline方法是什么？
  - **HashMoE / SwitchMoE 的经典 MoE 路由策略**：
    - HashMoE：采用平衡哈希函数将 token 均匀分配到各 expert，不使用可学习参数。优点是绝对负载均衡，缺点是缺乏语义区分能力（token 与 expert 匹配无学习过程），导致收敛速度虽快但推理精度可能不足。
    - SwitchMoE：使用 Dense 层门控网络计算 gating scores，选 Top-1 expert 并通过 softmax 加权。辅助 loss (L_aux) 鼓励负载均衡。但路由策略无局部性感知，token 可能被路由到远程节点的 expert，导致频繁的跨节点 All-to-All 通信。且存在 "winner-take-all" 问题——少数 expert 接收大部分 token，约 40% expert 几乎不被使用。
    - 两者均未优化 expert capacity：使用经验性的 capacity factor c_f 设定 expert capacity = ceil(b_s·c_f/(ep·n))，未从理论上分析容量下界，导致冗余计算。
  - 全栈执行例子（Baseline SwitchMoE 在 PanGu-Σ 128 Ascend 910A NPU 上的一个 training step）：
    - **训练算法层**：Dense 层计算门控值 G(x) = ReLU(ω·x + ε) → Softmax → Top-1 argmax 选 expert → Token dispatch via All-to-All → Expert FFN (GeLU 激活) → Token combine via All-to-All。Load balance 仅依赖 L_aux = α·n·Σ f_i·P_i。
    - **系统框架层**：MindSpore 2.0.0 框架，PanGu-Σ 模型。RRE 两级路由：第一级按领域分组，第二级随机哈希（无学习参数）。16 experts 通过 expert parallelism=16 分布在 Ascend NPU 上。All-to-All 通信由 HCCL 执行。
    - **编译框架层**：论文未明确说明（MindSpore 提供图编译优化，但非本文修改目标）。
    - **kernel 调度层**：HCCL All-to-All 原语（算法带宽见图 2，随节点数增加而递减）。通信与计算串行执行——All-to-All 完成前无法开始 FFN 计算。All-to-All 占总训练时间 18.10%（128N）~28.74%（256N）。
    - **硬件架构层**：Ascend 910A NPU × 128，32 AI Cores/芯片，HCCS intra-node 高带宽互联，Fat-tree + RoCE inter-node 网络。跨节点 All-to-All 因小数据量频繁传输导致带宽利用率低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LocMoE 方法**：通过三个核心设计解决 baseline 缺陷：
    1. **GrAP 正交门控（解决语义区分 + 降低开销）**：用固定正交权重矩阵替代可学习 Dense 层计算门控值。正交性使不同领域 token 被路由到不同 expert（降低 cosine similarity），增强语义区分能力，同时避免 Dense 层参数量和计算开销。满足 Lemma 2 的均匀分配概率假设，为理论分析奠定基础。
    2. **局部性正则化（Locality Loss）（解决跨节点通信开销）**：在辅助 loss 基础上增加 KL 散度项 L_loc = μ·KL(D_c||D_l)，促使 token 优先路由到同节点本地 expert。将部分跨节点 All-to-All 转为节点内高带宽通信（HCCS），直接降低 All-to-All 时间 5.13%。同时 locality 的软约束避免 "winner-take-all"——更多 expert 参与早期训练（图 12 显示 LocMoE 的 expert 分配比 SwitchMoE 更均衡）。
    3. **理论下界指导 expert capacity 缩减（解决冗余计算）**：基于高维球面几何 + 正交门控假设，推导 NLP 领域 expert capacity 临界值 ec_min。证明当 token 与 gating weight 夹角余弦 δ 较大时，class-discriminative token 概率 p_δ 极小（≈0.3 at δ=Θ(1/√d)），说明仅少量 token 需要特定 expert 处理。据此可安全降低 expert capacity 不损失精度（pMoE 在 CV 领域得到类似结论，LocMoE 首次推广到 NLP 并结合网络结构分析）。
  - 对应解决 Baseline 缺陷：
    - **Dense 门控开销大 + 缺乏语义正交性** → GrAP 固定正交权重，O(1) 计算（仅 mean pooling），无参数量，自然正交
    - **跨节点 All-to-All 占比高（18%~29%）** → Locality loss 鼓励本地路由，减少跨节点通信量，配合 MindSpore Group-wise All-to-All + FFN 重叠
    - **Expert capacity 经验设定冗余** → 理论推导下界，提供安全的 capacity 缩减指导
    - **SwitchMoE 的 winner-take-all（~40% expert 闲置）** → Locality 软约束 + auxiliary loss 双重正则化，使 expert 分配更均衡
  - 全栈执行例子（LocMoE 在 PanGu-Σ 128 Ascend 910A NPU 上的一个 training step）：
    - **训练算法层**：GrAP 分组平均池化计算门控值 → Softmax → Top-1 argmax → KL(L_aux + L_loc) 双约束路由决策 → 按理论下界设定 expert capacity → Token dispatch via Group-wise All-to-All（TP/EP 域拆分）→ Expert FFN 与 All-to-All 切片重叠执行 → Token combine → L_task = L_aux + L_loc + L_cross 反向传播
    - **系统框架层**：MindSpore 2.0.0，PanGu-Σ 的 RRE 第二级路由被 LocMoE 改写（从随机哈希变为可学习的 locality-aware routing）。Group-wise All-to-All 利用 TP 域高速带宽分担 EP 域通信压力。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：HCCL Group-wise All-to-All + All-Gather（TP 域同步）+ FFN kernel 与 All-to-All 切片重叠。通信时间下降 5.13%，64N 下每 epoch 总时间减少 12.68%~22.24%。
    - **硬件架构层**：Ascend 910A NPU × 128，HCCS intra-node + Fat-tree inter-node。Locality loss 使更多通信利用 HCCS 高带宽（256GB/s），减少 RoCE inter-node 通信。但 256N 下因部分节点无本地 expert 导致 locality 策略失效，性能不如 HashMoE。

## Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

- baseline方法是什么？
  - **MoE-LLaVA / Molmo / GMoE 的 Standard Load Balancing TER**：现有 LVLM MoE 架构对所有 tokens（vision + language）统一施加 load balancing 约束 `L_balancing = K * Σ F_i * G_i`，鼓励 tokens 在 K 个 experts 间均匀分布。Router 是一个 trainable linear layer W ∈ R^{D×K}，通过 softmax 产生 routing probabilities，选择 Top-k experts 进行加权求和输出。Baseline 未区分 vision/language tokens 的分布特性差异。
  - 全栈执行例子（MoE-LLaVA-4Top2, StableLM-1.6B 在 A800-80G 上的一个 forward pass）：
    - **训练/推理算法层**：CLIP encoder 提取 vision tokens（~576 per image）→ Visual projector (MLP) 映射到 LLM hidden dim D → Vision + Language tokens concatenate → MoE layer: linear router W·x → softmax → Top-2 experts → load balancing loss 同时施加于 vision 和 language tokens → Expert FFN (GeLU + linear) → Output via weighted sum
    - **系统框架层**：HuggingFace Transformers + PyTorch。MoE-LLaVA 基于 LLaVA 框架，将指定层的 FFN 替换为 MoE layer（MoE-LLaVA 每 2 个 Transformer block 中替换 1 个 FFN 为 MoE）
    - **编译框架层**：论文未明确说明（PyTorch eager mode / torch.compile 均可）
    - **kernel 调度层**：cuBLAS GEMM for expert FFN + standard token dispatch/gather。All-to-all 通信的瓶颈是最慢 expert 负载
    - **硬件架构层**：A800-80G GPU。Memory ≈ 9.44G，GPU Utilization ≈ 60%，avg inference time ≈ 917s（MoE-LLaVA with StableLM-1.6B）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LTDR 方法**：
    1. **MsDaR**：发现 vision tokens 服从 long-tailed distribution（少量高信息 foreground + 大量低信息 background），language tokens 服从 uniform distribution。传统 load balancing 迫使 vision tail tokens 均匀分散到各 expert，阻碍 expert 专业化。LTDR 将 L_balancing 公式改为 `L_balancing = Σ F_i(T) · G_i(T)`（仅 language tokens 参与），让 vision tokens 自由路由到最匹配的 expert。通过 RPV (Routing Probability Variance) 分析验证：移除 load balancing 后 vision tokens 的 RPV 提升，表明 tail tokens 获得了更集中的 expert 分配。
    2. **VsDEA**：将 RPV > Mean(RPV) 的 vision tokens 定义为 tail tokens（约 13%），激活更多 experts（a > k）处理这些高信息量 tokens。本质上是一种 data-augmentation 策略——通过让 tail tokens 接受多个 expert 的联合处理来提升容错性和学习充分性。
  - 对应解决 Baseline 缺陷：
    - **Load balancing 与 vision long-tailed 分布冲突 → tail tokens 被打散导致学习不足** → MsDaR 移除 vision TER 的 load balancing，提高 vision tail tokens 的 RPV，使它们能选择专业化 expert 集中学习
    - **Vision tail tokens 信息密度高但数量少（~13%）→ 易被忽略或路由错误** → VsDEA 为 vision tail tokens 激活更多 experts（Top-a），通过 renormalized softmax 加权求和实现 data-augmentation 效果，降低 expert 错误路由的影响
    - **Conventional modality-aware MoE 将 experts 硬性划分给不同模态 → 损失模型容量和灵活性** → LTDR 不修改 expert 组织结构，仅通过分布感知的 routing 策略实现模态差异适配，保持 full expert pool 共享
  - 全栈执行例子（LTDR + MoE-LLaVA-4Top2, StableLM-1.6B 在 A800-80G 上的一个 forward pass）：
    - **训练/推理算法层**：CLIP encoder → Visual projector → Vision (M tokens) + Language (N tokens) concatenate → **MsDaR**: linear router W·x → softmax → RPV 计算 (per vision token variance) → language token 的 L_balancing 计算（vision 不参与）→ **VsDEA**: 基于 Mean(RPV) 动态分类 vision head/tail → tail tokens 激活 Top-a=4 experts (renormalized softmax weights) → head tokens + all language tokens 激活 Top-k=2 experts → Expert FFN → Weighted sum output
    - **系统框架层**：HuggingFace Transformers + PyTorch。与 baseline 完全兼容，仅修改 router 的 loss 计算逻辑和 expert activation 数量。Training config: batch size per GPU=16, precision=FP16, 1 epoch, cosine LR 2e-5
    - **编译框架层**：论文未明确说明
    - **kernel 调度层**：与 baseline 相同的 cuBLAS GEMM + token dispatch/gather。Inference time 略微更快（A800 avg 846s vs 917s baseline），因为 all-to-all 速度仍由最慢 expert 决定，VsDEA 不显著增加最慢 expert 负载
    - **硬件架构层**：A800-80G GPU。Memory ≈ 9.44G（几乎无增加），GPU Utilization ≈ 59.29%（vs 59.57% baseline），无额外计算开销

## Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

- baseline方法是什么？
  - **Sparsely Activated MoE with Discrete Token-Level Routing**：传统稀疏激活 MoE 模型（Switch Transformer, GShard, Expert Choice, ST-MoE 等）使用 top-k 离散路由网络将每个 token 分配到 k 个专家。路由决策是非可微的离散选择（argmax 或 top-k），使训练变得困难：(1) 需要精心设计的负载均衡辅助损失（auxiliary loss）来防止专家坍缩和负载不均；(2) 离散路由可能导致训练不稳定和专家欠专业化（Zoph et al., 2022）；(3) 路由网络梯度信号稀疏（仅选中的 k 个专家接收梯度），路由器学习效率受限；(4) 推理时需要维护所有专家的稀疏激活路径，增加系统实现复杂度。Token-level routing 学到的专家专业化是浅层的（标点、冠词等词级特征），缺乏深层语义/领域级别的专业化。
  - 全栈执行例子（Baseline Expert Choice MoE, 0.3B/8E, 8x A100, token-level routing）：
    - **训练算法层**：Router linear W_r·h_x → softmax → 每个 expert 选 top-k 输入（根据路由分数）→ capacity factor C=1 限制每 expert 处理 token 数 → token dispatch via all-to-all → 每个 expert 独立 FFN 计算 → token combine via all-to-all → 加权聚合输出。L_aux = α·N·Σ f_i·P_i 负载均衡 loss + L_lm 交叉熵。
    - **系统框架层**：PyTorch + Megatron-LM 或 DeepSpeed。Expert parallelism + all-to-all 通信。实现 all-to-all dispatch/combine + barriers between layers。论文未明确说明具体 Serving 框架（预训练场景）。
    - **编译框架层**：论文未明确说明。PyTorch eager mode / torch.compile。
    - **kernel 调度层**：NCCL All-to-All for token dispatch/combine + cuBLAS GEMM for expert FFN。Dispatch/combine 通信量 ∝ K·L·d（K experts per token，L tokens）。Token-level routing 产生不规则形状 all-to-all（各 expert 分配的 token 数不同），需 padding 或 drop。
    - **硬件架构层**：A100 GPU。Inter-node all-to-all 通信瓶颈（随 expert 数增加而加剧）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Lory 方法**：通过三个核心设计使 MoE 训练完全可微并实现有效的段级路由：
    1. **Fully Differentiable MoE via Expert Merging**：替代离散 top-k 路由为参数空间的软合并。路由权重 e_i = Softmax(R(h̄)) 直接作为专家参数 θ_i 的加权系数：θ̄ = Σ_i e_i · θ_i。合并后的 FFN 处理输入 o_x = FFN(h_x; θ̄)。整个过程端到端可微，梯度通过合并操作和路由网络全程流动，无需辅助负载均衡损失。解决了"离散路由不可微、梯度信号稀疏"的缺陷。
    2. **Causal Segment Routing**：将 token-level routing 改为 segment-level routing（T=256 tokens/段）。使用前一段的隐藏表示计算当前段的合并专家，保持自回归因果性。关键效率：合并操作从每 token 一次降为每段一次（L/T 次，对 L=4096/T=256 为 16 次），额外 FLOPs 上限仅 E/T（E=32 时约 12.5% MoE 层开销，总模型开销 15-28%）。推理时仅用 prompt 路由一次，后续生成与 Dense 模型效率完全相同。解决了"token-level 合并计算代价过高"的缺陷。
    3. **Similarity-based Data Batching**：用 Contriever 计算文档语义相似度，将相似文档拼接为训练实例，使相邻段来自相关领域。这鼓励段级路由学习领域级别的专家专业化（如 Python code 专家、学术论文专家），而非传统 token-level routing 学习的浅层语法特征（标点、冠词）。解决了"段级路由容易导致专家欠专业化"的缺陷。
  - 对应解决 Baseline 缺陷：
    - **离散路由不可微 → 需要负载均衡 loss、训练不稳定** → Expert merging 实现完全可微路由，全程梯度回传，无需离散决策和辅助 loss
    - **Token-level 合并计算代价 O(L·E·d·d') → 成本过高** → Causal segment routing 降为 O(L/T·E·d·d')，每段仅合并一次，合并开销与段数成正比
    - **Token-level 路由学到的是浅层词法特征 → 专家缺乏深层语义** → Similarity-based batching + segment routing 使专家学习领域级专业化（如 Python、arXiv、Books），不同层的专家在不同深度展现领域偏好
    - **Training recipe 复杂 → 负载均衡 + 离散决策 + 负载损失调参** → Lory 仅用单一的 cross-entropy loss 做端到端训练（无需 auxiliary load balancing loss，仅 warmup 阶段用 dense 初始化）
    - **推理时需维护专家稀疏激活路径** → Prompt-only routing 使推理退化为合并后的单一 Dense 模型，零额外内存或计算开销
  - 全栈执行例子（Lory 0.3B/32E, 64x A100, 150B tokens 预训练）：
    - **训练算法层**：Dense warmup（前 5% 步训练 dense 模型）→ 复制 FFN 初始化 MoE → Similarity-batched 训练实例（L=4096, 16 段 × 256 tokens）→ 每层的 Causal Segment Routing：S_0 的 h̄_0 经 stop_grad → softmax → e_0 → merge 32 FFN → 处理 S_0；S_1 用 h̄_0（无 stop_grad）→ e_1 → merge → 处理 S_1；依此类推 → Cross-entropy loss 回传（无额外 auxiliary loss）→ 梯度通过合并操作更新所有专家和路由网络
    - **系统框架层**：PyTorch + ZeRO 数据并行。无需 all-to-all 通信（参数合并替代了 token dispatch/combine）。论文 Section 6 讨论了 expert-wise model parallelism（按 hidden dim 切分所有专家到不同设备）用于扩展至 100B+ 参数。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。合并操作实现为逐专家参数的加权求和（纯 PyTorch tensor ops）。
    - **kernel 调度层**：合并后的 FFN 计算等价于单个 Dense FFN 的 GEMM 操作（cuBLAS），因为合并后的权重是单个矩阵。无需 GroupedGEMM 或 expert dispatch kernel。每段仅执行一次合并操作（16 次/层 vs 4096 次 token-level），合并 overhead 小。
    - **硬件架构层**：A100 GPU。合并操作在参数空间进行（通信专家参数而非 token 激活），适合 expert-wise model parallelism（图 7）。Data parallelism 用于非 MoE 部分（attention），model parallelism 用于 MoE 层。merge 操作无跨设备通信需求（每个设备持有完整的 expert 权重副本或按 hidden dim 分片）。

- 关键洞察：
  - **段级路由学到的专家专业化与 token 级路由完全不同**：Token-level MoE 学到的是浅层词法特征（标点专家、冠词专家），而 Lory 的段级路由学到的是领域级特征（中高层专家按 Books/arXiv/Python/Wikipedia 领域分化，图 6）。这种互补性暗示未来可结合两种路由策略。
  - **Warmup 训练至关重要**：无 warmup 时大量专家未被利用（1.5B/32E 图 10），warmup 确保专家从良好的 dense 初始化出发。更多专家数（32E）时专家利用率的持续提升可持续到训练结束（图 9）。
  - **Prompt-only routing 在推理中足够**：推理时仅用 prompt 做一次 routing（vs 逐段 reroute），下游任务性能差异不显著（Table 9），使 Lory 推理简化为 Dense 模型。

## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- baseline方法是什么？
  - **vLLM 默认 MoE 推理**：标准的 top-k expert routing，每个 token 独立选择 k 个 expert。在 decode 阶段，随着 batch size 增大，batch 中所有 token 选择的 expert 并集快速增长，最终激活几乎全部 expert。此时 MoE 丧失了稀疏性优势：激活的参数数量接近甚至超过同等容量的 Dense 模型，同时还要额外承担动态 expert dispatch 的开销。现有优化技术分为两类：(1) 静态方法——pruning、quantization、expert clustering 等，依赖离线校准数据集识别 workload 级 expert 冗余，永久修改模型。(2) 动态方法——per-token level 减少激活 expert 数（如 dynamic k），但未解决 batch 级 expert utilization 问题。
  - **全栈执行例子（Baseline vLLM on H200, Qwen2-57B, decode iteration, B=16）**：
    - **算法 Pipeline 层**：Router 计算 B×N=16×64 logits → softmax → top-8 per token → 每个 token 独立选 8 个 expert。由于 load-balancing loss 在训练中强制 uniform expert 利用，16 个 token 的 expert 选择并集覆盖几乎所有 64 个 expert（~55-60 个）。
    - **系统框架层**：vLLM v0.10.1, v1 scheduler, continuous batching。每层 MoE 的 expert 权重需从 HBM 全部加载（因几乎所有 expert 都被激活），decode iteration 中 42% 时间花在 expert weight 的 HBM 数据搬运上，成为 memory-bandwidth-bound。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：vLLM 默认 fused expert kernel（grouped GEMM），以全部 active expert（~55-60 个）为 dispatch 参数启动。每个 decode iteration 的专家计算 kernel 需加载 ~55-60 个 expert 的完整权重矩阵。
    - **硬件架构层**：H200 GPU (141 GB HBM, SXM NVLink)。Memory bandwidth 是瓶颈——decode 的 arithmetic intensity = (B × k / N) = 16 × 8 / 64 = 2，远低于 GPU compute 能力，decode latency 与 active expert 数量线性相关。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LYNX 方法**：通过 **AffinityBinning** 技术在 batch 级别动态重映射低置信度 token-to-expert assignment，将 batch 的 expert 激活从"并集"压缩为"最小关键集"。核心创新：
    1. **Insight 1 — Phase Sensitivity**：prefill 和 decode 对 expert fidelity 的敏感性截然不同。Prefill 需要严格 expert fidelity（建立 context），decode 因 attention/残差连接/累积 context 的补偿机制而对 expert 选择高度容错。因此 LYNX 仅在 memory-bound decode 阶段启用 remapping。
    2. **Insight 2 — Router Confidence as Reliability Signal**：Router 输出的 logits 差异（log-ratio to top-1）可靠地标识 token 对 expert assignment 的"在意程度"。高置信度 token（router 强烈偏好某个 expert）的 assignment 必须保留；低置信度 token（router 的各 expert 分数接近）可以安全地重映射到其他 expert 而不影响输出质量。这种区分无需校准数据，是 MoE router 的固有特性。
    3. **Insight 3 — Expert Rank Hierarchy**：top-1 expert 主导输出质量（deny top-1 会 catastrophic accuracy drop），lower-ranked experts (rank 2-8) 高度冗余。LYNX 利用此等级制：保留所有 token 的 top-1 expert 作为"锚点"，仅重映射 lower-ranked experts。
    4. **AffinityBinning**：将 per-token router confidence 按 sparsity ratio (k/N) 决定的 bin width 和 count 离散化。batch-size 为底数的指数加权确保：被多个高置信度 token 偏好的 expert 获得指数级高分，仅被低置信度 token 偏好的 expert 被大幅降权。这实现了 batch 级别专家重要性的自动校准——bin 参数仅由模型架构决定，无需任何 workload-specific tuning。
    5. **Expert Remapping**：在 batch 级别决策最小关键专家集后，将低置信度 token 的 lower-ranked expert assignment 重映射到该集合内。每个 token 仍激活 k 个 expert（保持 top-k 语义），只是 expert 选择在 batch 内被 consolidated 到更小的并集上。
  - **对应解决 Baseline 缺陷**：
    - **Baseline: batch 级 expert 并集随 batch size 线性增长 → memory bandwidth bound** → LYNX 在 batch 级减少 active expert 总数（~25→15），直接降低 HBM 数据搬运量，使 decode latency 与 reduced expert count 相关而非 full expert count。
    - **静态方法：依赖离线校准，永久修改模型，不灵活** → LYNX 完全 runtime，不从模型中永久移除任何 expert，不修改权重，适应 workload 变化。
    - **动态 per-token 方法：减少 per-token k，但 batch 级 expert 利用仍高** → LYNX 保持 per-token k 不变，通过 batch 内 remapping 减少并集大小。
    - **需要 calibration data** → LYNX 仅依赖 router 输出 logits 作为信号，self-calibrating（参数仅由 sparsity ratio 决定）。
  - **全栈执行例子（LYNX on vLLM, Qwen2-57B, decode iteration, B=16）**：
    - **算法 Pipeline 层**：Router logits → AffinityBinning（每 token 的 log-ratio 离散化到 6 个 bin）→ Adaptive Scoring（16^bin 指数加权，高置信度 token 的偏好主导分数）→ 动态确定 active expert set（如从 ~55 降至 ~15）→ Low-confidence token assignment 重映射到 active set → 每 token 仍激活 8 个 expert。
    - **系统框架层**：vLLM batch scheduler 中 Phase-aware Optimizer 识别 decode-only iteration → 启用 LYNX pipeline。每层 router 后插入 4 个 fused kernel → 最终 dispatch 到更小的 expert set。
    - **编译框架层**：论文未明确说明。Triton kernel 编译为 CUDA，CUDA Graph 捕获静态执行图。
    - **Kernel 调度层**：4 个 fused Triton kernel（替代 700+ PyTorch ops, <4% overhead）→ Expert GEMM kernel 仅加载 ~15 个 expert 的权重（vs baseline 的 ~55 个），HBM 数据搬运量减少 ~73%。
    - **硬件架构层**：H200 GPU。LYNX kernel overhead (<4%) 远小于内存带宽节省（expert 加载量减少 ~73%），net latency 降低 1.09-1.30x。准确率偏差 <1%，平均情况甚至提升（因移除了 training load-balancing 强制的低质量 expert assignment）。

## FineMoE: Fine-Grained Expert Offloading for Large Mixture-of-Experts Serving

- baseline方法是什么？
  - **Coarse-grained Expert Offloading**（以 MoE-Infinity、ProMoE、Mixtral-Offloading、DeepSpeed-Inference 为代表）：现有 expert offloading 方法在 **request-level（粗粒度）** 粒度上追踪和预测 expert activation pattern：
    - **MoE-Infinity**：track request-level expert hit counts (Expert Activation Matrix)，synchronous expert prediction and prefetching。prefetch distance > 1 层时无法观测到足够的 expert trajectory history，只能对所有层使用 request-level 聚合统计（最流行 experts）进行 prefetching。
    - **ProMoE**：stride-based speculative prefetching，需要 per-layer NN predictor 训练（millions of params per layer），训练和 retraining 开销大。
    - **Mixtral-Offloading**：layer-wise speculative prefetching + LRU cache。synchronous prefetching with prefetch distance = 1（无法隐藏 prefetch latency）。
    - **DeepSpeed-Inference**：layer-wise parameter offloading without expert awareness，pure on-demand loading（无 prefetch），latency 最高。
  - **Coarse-grained 的三个核心缺陷**：
    1. **Insufficient latency-memory trade-off**：要么低延迟大内存（MoE-Infinity），要么高延迟小内存（DeepSpeed-Inference），无法同时优化。
    2. **Low expert hit rate**：request-level 聚合抹去了 iteration-level 的细粒度 expert selection pattern（图 3a 的 heatmap 对比：iteration-level 有明显 pattern，request-level 熵高、可预测性低）。Shannon entropy 分析表明 coarse-grained pattern 的 entropy 显著高于 fine-grained（图 3b），且 entropy 随 iteration 累积逐渐升高并 plateau（图 3c），说明 request-level 聚合使 expert pattern 越来越不可预测。
    3. **Ignorance of MoE models' and prompts' heterogeneity**：不同 MoE 模型（Mixtral-8×7B 8 experts/layer vs Qwen1.5-MoE 60 experts/layer）和不同 prompts（semantic diversity）在 one-fits-all 方式下被同等对待，失去了按模型和 prompt 特征自适应优化的机会。
  - **全栈执行例子**（Baseline MoE-Infinity, Mixtral-8×7B, LMSYS-Chat-1M）：
    - **算法 Pipeline 层**：Gate network 输出 top-K expert selection → request-level expert activation count 在所有 iteration 上聚合 → 以 historical request 的 aggregated activation counts 作为 prediction signal。因为 decoder-only MoE + load-balancing loss 导致 balanced routing（expert activation 接近均匀分布），aggregated counts 的预测能力弱。
    - **系统框架层**：MoE-Infinity on HuggingFace Transformers。Expert Activation Matrix stored in CPU memory → synchronous prediction before each MoE layer → expert 从 CPU to GPU via PCIe → forward computation。Synchronous design 导致 prefetching latency 无法与 computation overlap。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：CUDA Runtime API for expert memory management。Expert Cache 使用 LFU 策略。Multi-GPU EP with round-robin expert distribution。Synchronous prefetching 阻塞 forward：每层必须先完成 expert prediction + prefetching 才能执行 computation。
    - **硬件架构层**：6× RTX 3090 24GB + PCIe 4.0 32GB/s。Expert miss → on-demand loading latency = T_e per miss 直接增加到 iteration latency。Request-level 粗粒度预测导致 expert hit rate 低 → 大量 on-demand loading → 高 inference latency。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FineMoE 方法**：从 coarse-grained (request-level) 切换到 **fine-grained (iteration-level)** expert offloading，通过三个核心创新：
    1. **Expert Map（取代 Expert Activation Matrix）**：记录每个 iteration 中每层 gate network 输出的完整概率分布 P_l^{(i)} ∈ R^J（而非 binary hit count）。保留 gate network 对每个 expert 的 confidence/preference 信息，支持退化恢复 coarse-grained 信息（top-K + 聚合）。粒度从 request-level 变为 iteration-level，解决"aggregation destroys predictability"的问题。
    2. **Dual-Similarity Expert Map Search**：利用两种 fine-grained 指标搜索最准确的 historical expert map：(a) **Semantic similarity**——semantic embeddings（embedding layer output）的 cosine similarity。基于"语义相似 prompts 有相似 expert 选择"的假设，用于 prefetch distance d 以内的初始层（尚无 trajectory history）。Pearson correlation 验证 semantic similarity score 与 expert hit rate 正相关；(b) **Trajectory similarity**——已观察到的前 (l-d) 层 expert probability distributions 的 cosine similarity。用于第 l ∈ [d+1, L] 层。随 iteration 推进，越来越多的 trajectory history 可被利用，提高预测准确度。
    3. **Similarity-aware Adaptive Expert Selection**：并非固定 top-K 选择，而是根据 search confidence 动态调整。δ_l = Clip(1 - similarity_score, 0, 1)——高 similarity 时 δ 低，只需选 highest-probability 的少数 experts；低 similarity 时 δ 高，选更多 experts 防止 miss。类似 confidence-based 的 adaptive exploration：系统自信则 lean（省内存），不自信则 wide（保准确）。
    4. **Asynchronous Publisher-Subscriber Architecture**：将 map searching + expert prefetching 与 inference forward pass 解耦（弥补 MoE-Infinity synchronous design 的缺陷）。Inference process 作为 Publisher 持续写入 context → Expert Map Searcher 作为 Subscriber 异步消费 context 并 prefetch。
  - **对应解决 Baseline 缺陷**：
    - **Coarse-grained → Fine-grained** → iteration-level pattern tracking（expert map）降低 Shannon entropy（图 3b），提高 pattern predictability → expert hit rate 提升 39%（vs SOTA）。
    - **No semantic awareness → Semantic-based search** → 利用 input prompt 的 semantic embedding 为缺少 trajectory history 的初始层提供有效的 expert prediction → 解决 prefetch distance 内无法观测 trajectory 的问题。
    - **Fixed prefetching → Adaptive δ prefetching** → 根据 search confidence 动态调整 prefetch 量，high confidence 时节省 GPU memory，low confidence 时增加 coverage → 在 latency-memory trade-off 上找到更优均衡点（6GB cache limit 时 TPOT 降低 16-36% vs baselines）。
    - **Synchronous → Asynchronous** → map searching 和 prefetching 的 overhead 不进入 critical path → iteration overhead < 1%（< 50ms）。
    - **One-fits-all → Model/prompt heterogeneity aware** → 不同 MoE 模型（Mixtral/Qwen/Phi）independent profiling 确定最优 prefetch distance (3/6/4) → 适应不同 expert 数量和 layer 深度的模型。
  - **全栈执行例子**（FineMoE, Mixtral-8×7B, LMSYS-Chat-1M）：
    - **算法 Pipeline 层**：Embedding layer → semantic embedding extraction → cosine similarity search 在 Expert Map Store（1K maps）→ 前 d=3 层用 semantic match 的 expert map 指导 prefetch。Iteration i, Layer 4：收集 P_1, P_2, P_3 的 probability trajectory → cosine similarity with historical maps → 选 best match 的 P_4 → δ_4 = Clip(1-score_traj, 0, 1) → 选 experts 直至 Σp ≥ δ_4 且 count ≥ K=2 → prefetch E_prefetch。每层 Gate network 仍按原始 top-K 做最终 expert 选择（lossless），prefetching 只是预测性加载。
    - **系统框架层**：HuggingFace Transformers + FineMoE Expert Map Store (Python/PyTorch/NumPy) + Expert Cache (C++, CUDA)。Publisher-Subscriber 异步通信。Prefetch distance d=3 使 map searching + prefetching latency 被 overlap 到 attention + gate + expert 计算中。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：GPU task pool 异步调度 expert prefetch 任务 → PCIe 4.0 32GB/s cudaMemcpyAsync(host→device) → Expert Cache hash map 更新。LFU + probability-based eviction (PRI^{evict} = 1/(p*freq))。Expert miss 时暂停所有 prefetching → 立即 on-demand cudaMemcpy → 恢复 prefetching。Prefetch priority = p/(l-l_now) 确保近层的 high-probability experts 优先。
    - **硬件架构层**：6× RTX 3090 24GB + NVLink + PCIe 4.0 32GB/s。Expert Map Store < 200MB CPU memory（1K maps）。Round-robin EP 将 experts 分布到 6 个 GPU → per-GPU cache 独立管理。FineMoE 在 6GB GPU cache limit 时 TPOT 降低 16-36% vs baselines，在 48GB+ 时所有方法趋近（因足够 cache 几乎所有 experts）。A100 (80GB, single GPU, no EP) 上提升减小（fast inference + single GPU 减少了 offloading 收益），但仍一致优于 baselines。

## LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

- baseline方法是什么？
  - Baseline是标准的expert parallelism MoE训练流程。在全栈的执行例子：
    - 算法层：每个MoE层，gate网络对每个token计算top-K expert选择（如top-2 gating），选中的token通过all-to-all通信发送到对应expert GPU，expert计算FFN输出，再通过all-to-all通信将结果传回原GPU。
    - 系统框架层：使用PyTorch + NCCL实现all-to-all通信，DeepSpeed-MoE/Tutel等框架管理expert分布和通信调度。
    - 编译框架层：论文未明确说明。
    - kernel调度层：论文未明确说明。使用标准NCCL all-to-all collective communication。
    - 硬件架构层：在A100/V100 GPU集群上运行，跨机通过RDMA NIC互联。All-to-all通信占训练总时间平均45%（GPT-MoE约30%，RoBERTa-MoE约40%，Swin-MoE约70%），且无法与计算重叠，成为训练瓶颈。
  - Baseline的缺陷：all-to-all通信量过大，因为每个token的完整hidden state（h维向量）都需要传输。随着模型规模（expert数、层数）和GPU数量增加，通信/计算比基本不变，通信瓶颈持续存在。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：LSH-MoE在all-to-all通信前插入LSH聚类压缩步骤，仅传输聚类中心而非全部token，在接收端通过残差补偿还原近似结果。
  - 全栈执行例子（对比baseline）：
    - 算法层：**核心创新**。在gate网络完成token-to-expert映射后、all-to-all通信前，对每个expert的token集合执行cross-polytope LSH聚类。具体地：将每个token x通过随机旋转矩阵R映射到cross-polytope顶点（`argmax |Rx|_i`），相同bucket的token归为一类，计算聚类中心`cluster_mean`作为传输单元。传输量从n×h降至m×h（m为cluster数，m<<n）。接收端expert对中心计算后，通过残差补偿`E(cluster) + Δx`还原每个token的近似输出。这直接减少了all-to-all通信量，压缩率可达11.7%-20%。
    - 系统框架层：方法框架无关，可插入PyTorch+NCCL的标准MoE训练pipeline。论文在PyTorch 1.11上实现LSH聚类模块和残差管理，通信层仍使用NCCL all-to-all，但传输数据量大幅减少。
    - 编译框架层：论文未明确说明。
    - kernel调度层：论文未明确说明。LSH聚类本身是GPU上的矩阵运算（旋转+argmax+mean），使用PyTorch原生算子。
    - 硬件架构层：与baseline相同硬件平台，但通信量减少直接转化为1.28×-2.2×端到端加速。Scalability分析表明加速比在更大模型和更多GPU下依然保持，因为通信/计算比保持恒定。
  - 设计思路总结：观察到MoE训练中all-to-all通信的token存在高相似度（PCA可视化呈现聚类现象，源于Zipf分布数据和Transformer attention的同质化效应），利用LSH将相似token在通信前聚类、仅传中心、传后残差补偿，以可控的精度损失换取大幅通信量减少。

