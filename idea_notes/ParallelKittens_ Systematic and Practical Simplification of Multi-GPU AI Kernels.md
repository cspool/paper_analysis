## ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

- baseline方法是什么？
  现有多GPU AI kernel开发存在三类baseline方法，各有缺陷：
  
  (1) **算子特定手写kernel**（Flux, CUTLASS distributed GEMM, Comet, FlashDMoE, Ring Attention等）：针对特定operator（GEMM、attention、MoE）手工实现compute-communication overlap，使用host-triggered copy engine配合device kernel stream-level overlap，或高度优化的on-device scheduler。缺陷：实现复杂、缺乏可复用抽象（如FlashDMoE仅支持TF32精度，BF16/FP16支持在发布5个月后仍在开发中）。
  
  (2) **编译器方法**（Triton Distributed, TileLink）：扩展Triton DSL支持OpenSHMEM风格的单边操作，编译器自动生成多GPU kernel。缺陷：缺乏显式的负载分布控制（warp/SM specialization），无法实现最优overlap；跨硬件平台泛化差（Triton Distributed原为H800调优，在H100上有时慢于非overlap基线）。
  
  (3) **通信库方法**（NCCL+NCCLX, NVSHMEM）：NCCL使用host-initiated copy engine在独立CUDA stream上与compute kernel做stream-level overlap。缺陷：
  - NCCL强制双向同步（sender和receiver必须相互确认才能传输），即使点对点通信也如此，细粒度通信开销显著；
  - NCCL使用小型预分配中间缓冲区（communication channels），引入额外数据搬运；
  - NVSHMEM每次remote peer access执行__ldg获取peer地址并强制__syncthreads，导致高达4.5×的element-wise NVLink访问延迟；
  - 不支持TMA和in-network acceleration，仅使用copy engine或寄存器操作。
  
  全栈执行例子（以8×H100上cuBLAS GEMM + NCCL all-gather的数据并行前向为例）：
  - 算法层：数据并行下的AG+GEMM+Activation+GEMM+RS流程。输入按行分片，权重按列分片。AG收集完整输入 → GEMM1 → 激活 → GEMM2 → RS分散结果。
  - 系统框架层：PyTorch Distributed + Megatron-LM风格的parallelization。使用torchrun多进程管理（每GPU一进程）。
  - 编译框架层：Triton分布式或直接CUDA kernel。baseline使用独立kernel launch：NCCL all-gather → cuBLAS GEMM → NCCL reduce-scatter，每个操作是独立的kernel或stream。
  - kernel调度层：NCCL使用host-initiated copy engine在单独CUDA stream上执行all-gather/reduce-scatter，与cuBLAS GEMM在不同stream上做stream-level overlap。Host端管理同步（cudaStreamSynchronize或cudaEvent），无device-side tile级overlap。copy engine需要至少256MB消息才能饱和带宽；对小矩阵（N=2048），Triton Distributed和Flux可能慢于非overlap基线。
  - 硬件架构层：H100 GPU的copy engine独立于SM运作，但host发起、仅支持连续内存传输。NVSwitch fabric仅作为passive switch转发数据，未利用in-network加速能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ParallelKittens通过三大设计原则和8个核心原语+LCSC统一编程模板，系统化解决上述baseline的缺陷：
  
  **(A) 传输机制原则**：PK分析三种传输机制（Copy Engine、TMA、寄存器指令）的带宽利用率vs消息粒度的trade-off曲线，仅暴露每种功能最有效的机制：
  - TMA用于点对点通信：74%峰值利用率仅需2KB消息（vs copy engine需256MB），支持单线程异步启动，不增加寄存器压力，使intra-SM overlap成为可能。
  - 寄存器操作用于in-network reduction：虽然效率较低（70%利用率）需要更多SM占用，但提供了copy engine和TMA均不具备的NVSwitch in-network reduction功能（multimem.ld_reduce, multimem.red）。
  - 完全放弃host-initiated copy engine用于device-side通信叠加。
  
  **(B) 调度策略原则**：PK统一支持Intra-SM和Inter-SM两种overlap调度，通过LCSC模板自动选择最优策略：
  - Intra-SM overlapping：所有SM同时执行compute和communication，通过单线程TMA异步调用实现通信与tensor core计算并发。对GEMM+RS场景：T_comp_tile = 2mnK/R, T_comm_tile = s*m*n/B，当K ≥ sR/(2B)≈2197时通信完全隐藏。实测验证K=4096时通信占比<1%。适用于通信模式与计算模式对齐的场景。
  - Inter-SM overlapping：将部分SM专用于communication，其余SM全用于compute。关键优势：(i) 利用in-network reduction大幅减少通信量（GEMM+AR中T_comm降低约N倍）; (ii) 对于Ring Attention，通信SM将下一block的KV tensor批量传输到local HBM，避免remote L2 cache miss导致的重复传输。通过num_comm_sms参数运行时自动搜索最优SM分配。
  - 对比：intra-SM在GEMM+RS中比inter-SM快1.2×（更高compute利用、更低sync开销），inter-SM在GEMM+AR中通过in-network reduction实现3.62×提升。
  
  **(C) 设计开销消除**：
  - 使用预分配目标缓冲区实现单向TMA P2P传输，消除NCCL的双向同步和中间缓冲区overhead。纯all-reduce通信kernel由此获得1.79×加速。
  - 将peer地址保持在寄存器中并移除不必要的__syncthreads，使element-wise NVLink访问延迟降低4.5×，带宽利用率提升约20 GB/s。
  - PK的IPC utility通过VMM+POSIX fd机制透明处理多进程地址空间映射和multicast object创建，使kernel代码无需感知底层IPC复杂度。
  
  PK方法全栈执行例子（以8×H100上PK实现的fused GEMM+AR kernel为例）：
  - 算法层：同baseline的AG+GEMM+AR流程，但通过LCSC模板将GEMM和all-reduce融合为单个kernel。
  - 系统框架层：PK提供IPC和PyTorch utilities集成，使用VMM手动分配的multicast memory（通过cuMemCreate→cuMemExportToShareableHandle→Unix domain socket传输fd→cuMemImportFromShareableHandle→cuMulticastCreate流程），使各进程拥有local address和multicast address两个虚拟地址映射。
  - 编译框架层：PK本身是C++ embedded DSL编译为CUDA kernel，无额外编译层。LCSC模板编译时通过config struct确定SM数量、thread分配和warpgroup布局。
  - kernel调度层：Inter-SM overlapping模式。Compute SMs运行loader/consumer/storer流水线：(loader) TMA从local HBM异步加载A_tile和B_tile到SMEM → (consumer) warpgroup执行mma_AB累积C_accum → (storer) warpgroup::store写入local output，arrive信号量通知下一stage，signal原语原子加barrier通知communication SM。Communication SMs运行communicator：(wait) 等待所有compute SM的barrier达到NUM_DEVICES → __syncthreads → (all_reduce) 整个warp使用multimem.ld_reduce PTX指令从multicast memory读取各GPU的partial结果，通过NVSwitch in-network reduction直接归约，再写入multicast memory。每kernel的通信相关代码仅10行。
  - 硬件架构层：利用NVSwitch SHARP in-network reduction（multimem.red/multimem.ld_reduce PTX指令）将all-reduce的通信量从O(N) peer写入降为O(1)对multicast object的读取+归约。TMA单线程异步操作不占用tensor core，使所有SM的计算单元保持繁忙。

- baseline方法是什么？
  Baseline是已有的expert skipping方法（NAEE、MC-MoE、DiEP），这些方法最初为text-only LLMs设计（top-2 routing），论文中将其适配到MLLMs的top-k（k>2）setting。Baseline方法的核心决策方式：仅依赖per-layer内的局部routing probability π_i^{(l)}来判定expert去留。NAEE在top-2场景跳过top-2 expert若π_top-2 < β · π_top-1；MC-MoE在NAEE基础上增加attention-aware protection；DiEP联合考虑routing probability和expert similarity做可微分剪枝。所有baseline方法共享两个根本缺陷：(1) 忽略expert贡献在不同层之间的不对等性——浅层expert对最终输出的影响远大于深层expert，但跳过策略对所有层一视同仁；(2) 将text和vision token等同对待——未考虑不同模态token在FFN层中的行为差异（vision token在FFN前后的余弦相似度更高，说明FFN对vision token的更新幅度更小，vision expert冗余度更高）。

  全栈执行例子（以Kimi-VL-A3B-Instruct在8×H200上处理多模态推理请求为例，baseline NAEE方法，跳过67% experts）：
  - 算法层：标准MoE MLLM架构——Visual Encoder提取vision token → Projector对齐到text embedding空间 → LLM backbone（26层transformer，每层含self-attention + MoE FFN，64 experts/layer，k=6）。NAEE在每层FFN中：计算routing probability → 若sum_{u=i}^k π_top-u < β^{(l)} · sum_{v=1}^k π_top-v，跳过top-i到top-k experts。β^{(l)}通过genetic search在GQA上确定。所有层使用相同规则，不区分text/vision token。
  - 系统框架层：基于transformers库加载模型，PyTorch forward pass。router kernel执行routing logits → top-k selection → NAEE规则判断 → expert dispatch/gather。Baseline未对MoE层做特殊kernel优化（论文未明确说明baseline的具体kernel实现）。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Baseline MoE dispatch/gather按NAEE规则跳过部分expert后，剩余active experts通过标准Group GEMM执行。NAEE的跳过决策逻辑在router kernel的top-k之后执行，仅增加少量比较操作。但由于baseline仅看局部π_i^{(l)}，在较高跳过率（>67%）时跳过过多浅层关键expert，造成严重准确率下降（83%跳过时avg accuracy从100%降至82.81%-88.32%）。
  - 硬件架构层：8×H200 GPU。Baseline与MoDES在相同跳过率下kernel wall-clock speedup几乎相同（<1%差异，因为skip操作本身的开销相似），但由于baseline的跳过决策质量差，无法在保持准确率的同时达到MLLMs所需的高跳过率。

  Baseline两大核心缺陷：
  1. **Layer-agnostic skipping**：不考虑层间expert贡献差异。Shallow layer experts对final output的影响大（error会在后续层被放大），deep layer experts影响小。Baseline对所有层使用同一类阈值规则，导致shallower layers中critical experts被不当跳过（error explosion），而deeper layers中redundant experts未充分跳过（opportunity waste）。
  2. **Modality-agnostic thresholding**：未区分text/vision token。经验证：vision token在FFN前后的余弦相似度更高（更接近1），即FFN对vision token的更新幅度小于text token。而且vision token的expert冗余度更高（降低k值对vision的性能影响小于text）。Baseline单一阈值要么对text过于激进（导致关键信息丢失），要么对vision过于保守（浪费计算资源）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出MoDES——第一个面向MoE MLLMs的训练无关expert skipping框架，通过三个协同组件解决baseline缺陷：

  **(1) GMLG（Globally-Modulated Local Gating）**——解决"Layer-agnostic skipping"缺陷：
  引入全局逐层重要性因子α^{(l)}，通过离线KL divergence校准量化每层expert对final output的贡献。Importance score s_i^{(l)} = α^{(l)} · π_i^{(l)}同时编码全局层重要性和局部token- expert匹配度。α^{(l)}大的浅层：s_i^{(l)}整体偏高 → 即使π_i^{(l)}较低也可能被保留；α^{(l)}小的深层：s_i^{(l)}整体偏低 → π_i^{(l)}中等也可能被跳过。Inference时无额外计算开销（α^{(l)}预计算）。

  **(2) DMT（Dual-Modality Thresholding）**——解决"Modality-agnostic thresholding"缺陷：
  为text token和vision token分别设置阈值τ_t和τ_v。基于验证：vision expert冗余度高 → τ_v < τ_t → vision token跳过更多expert。Modality-specific decision使text token保留关键专家保证质量，vision token更激进地跳过冗余专家节省计算。

  **(3) Frontier Search**——解决"Threshold optimization效率"问题：
  利用f（KL divergence，性能损失）和g（跳过比例，效率）关于(τ_t, τ_v)的单调性，用O(ND)的frontier search替代O(ND²)的exhaustive search。搜索时间~45×降低（从>2天降至<2小时），性能差异<0.01%（96.24% vs 96.25%）。

  全栈执行对比baseline（以Kimi-VL-A3B-Instruct在8×H200上处理同一多模态推理请求，MoDES方法，跳过83% experts）：
  - 算法层：相同MoE MLLM架构。每层MoE FFN中：router → softmax → 对每个top-k expert i计算s_i^{(l)} = α̃^{(l)} · π_i^{(l)} → 根据token类型选择阈值τ_t或τ_v → 跳过s_i^{(l)} < τ的expert → 仅active experts参与计算和aggregation。Vision token的τ_v < τ_t → vision token上的跳过率显著高于text token。Shallow层α̃^{(l)}大 → s_i整体高 → 跳过少；Deep层α̃^{(l)}小 → s_i整体低 → 跳过更多。MoDES在跳过83% experts时仍保持96.25%的平均准确率（vs baseline 82.81%-88.32%），在跳过88%时对Qwen3-VL-MoE-30B保持97.33%（vs baseline 86.66%）。
  - 系统框架层：基于transformers库加载模型 + 自定义CUDA extension。PyTorch forward pass中每个MoE层调用MoDES的custom kernel。Calibration和search离线完成于8×H200（20分钟至<4小时，depending on model）。
  - 编译框架层：论文未明确说明编译框架。
  - kernel调度层：custom CUDA kernels实现：(a) Router kernel内嵌DMT thresholding——branch-free masked comparison，跳过expert路由设为sentinel ID M+1；(b) Sentinel-aware dispatch/gather——自动过滤sentinel entries；(c) Group GEMM with offline-profiled tile sizes——根据动态expert activation pattern选择最优kernel配置。MoDES的额外开销：仅一次α̃^{(l)}乘法和一个masked comparison per expert（与baseline NAEE的β比较开销相当）。跳过率相同时speedup与baseline几乎相同（<1%差异），但MoDES可在高跳过率下保持准确率。
  - 硬件架构层：8×H200 GPU。Calibration每条数据需2次forward pass（original + layer-skipped），1024 samples over all layers。Frontier search对每对(τ_t, τ_v)需1次forward pass，O(ND)复杂度。Prefill speedup ~2.16×（batch=8, Kimi-VL），Decode speedup ~1.26×。Decode speedup较低原因：(i) decode为memory-bound，(ii) decode阶段仅处理text token，总体跳过率低于prefill。

  设计思路核心：MoDES的本质是将expert skipping从"只看局部（单层routing概率）"提升为"全局感知+模态感知"的决策。GMLG通过预校准的α^{(l)}将层重要性信息编码到重要性分数中，DMT通过双阈值区分text/vision token的不同特征。两个insight（层间贡献不均+模态行为差异）各对应一个设计组件，二者叠加产生非线性增益——在极高跳过率（>80%）下差距尤为显著（差距从~6%扩大到~10%）。同时Frontier Search保证了从"insight→设计→优化"全pipeline的效率可行性。
