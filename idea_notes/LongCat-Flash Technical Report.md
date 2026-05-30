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
