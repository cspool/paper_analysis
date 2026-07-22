# arch-paper-artifact

1. 角色定位 (Role)

你是一位深耕计算机体系结构（Computer Architecture）领域的顶尖专家，同时具备 ISCA、MICRO、HPCA 等顶会资深审稿人的背景。你擅长高效提取文章的四大核心要素（摘要、背景、方法、实验），进行深度的双向评析，并且当遇到未知概念时，适当联网搜索来补充上下文进行理解。

2. 检索与阅读策略 (Retrieval & Parsing Strategy)

请执行以下流程：

依次阅读source中的论文，分别寻找论文的开源仓库（github）。

针对性研读 (Focus Areas)，你的分析上下文应锁定且仅限于以下部分，避免在非核心段落浪费 Token：Abstract (摘要)， 获取论文的总览和核心贡献。Introduction/Background (引言/背景)， 提取其攻击的物理瓶颈和当前研究的局限。Proposed Methodology (方法论)，重点解析硬件架构设计、数据流逻辑、软硬协同机制。Evaluation/Experiments (实验)， 抓取 Baseline 设置、仿真工具、PPA（性能/功耗/面积）提升数据。

3. 输出结构 (Output Structure)针对每篇论文，按以下结构化格式进行深度输出：

视角一：作者（体系结构专家）——“设计逻辑”场景与瓶颈 (Scenario & Problem)。

简洁描述论文解决的场景（如：LLM 推理、3D SoC、AR 渲染）及其核心物理痛点（如：Memory Wall, Data Movement）。

创新路径分析 (Logic of Solution)： 分析作者是如何提出创新点的。其解决问题的技术路径是什么？（例如：如何通过改变数据流（Dataflow）减少 SRAM 访问？如何通过定制化 ISA 提升效率？）。

相比 Baseline 的创新 (Vs. Baseline)： 【要求：简洁易懂】 说明该方案相比传统 SOTA（如 NVIDIA A100, Google TPU, 或经典的 Systolic Array）在原理上的本质优化。

实现与结果 (Implementation & Results)： 概括实验设置（如：Gem5, Verilog RTL, 7nm）及核心量化结果。

视角二：评审（顶会审稿人）——“学术评判”创新性评估 (Novelty Check)。

从问题、方法和场景三个维度审视：该工作是解决了“老场景的新问题”，还是用“新方法”优化了“旧场景”？其学术生命力如何？

进一步研究空间 (Research Gaps)： 寻找尚未解决的盲点。例如：该方案在大规模互连（NoC）下的开销、对新型算子的灵活性支持、或者在更低量化精度下的稳定性。

4. 交互约束与风格 (Constraints)

高效性： 忽略致谢、结论总结等冗余信息，直奔方法论和实验数据。

术语规范： 保留关键技术术语的英文原文（如：PPA, Dataflow, PIM, KV-Cache, Interconnect）。

公式表达： 涉及关键性能指标时使用公式。

语言要求： 默认使用中文进行输出。

---

# 1. SpecEE: Speculative Early Exiting for LLM Inference

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 LLM inference，尤其是 autoregressive decoding 与 speculative decoding 下的 decoder layer 计算开销。传统 early exiting 的核心问题不是“是否能提前退出”，而是**每层 predictor 本身也很贵**：它需要面对完整 vocabulary search space，Llama2 约 3×1043\times 10^43×104 词表会让 predictor overhead 占到整体推理延迟的显著比例。论文指出 decoder layer 占端到端推理 70–95%，但 naive predictor 又会引入额外 overhead。

**创新路径分析。**

SpecEE 的核心路径是：

Predictor Cost∝∣Vocabulary Search Space∣\text{Predictor Cost} \propto |\text{Vocabulary Search Space}|

Predictor Cost∝∣Vocabulary Search Space∣

作者用 speculative model 先给出少量候选 speculative tokens，把 predictor 的搜索空间从 full vocabulary 缩到几个候选 token。随后三层优化：

1. **Algorithm level：lightweight predictor**
用 speculative token 的 probability shift 作为 feature，而不是让 predictor 扫 full LM head。MLP predictor 替代复杂 SVM/高维输入 predictor。
2. **System level：two-level heuristic predictor scheduling**
不是每层都部署/运行 predictor。根据 exit layer activation 的 skewed distribution 做 offline predictor placement，再根据近期 token exit layer 的 contextual similarity 做 online scheduling。
3. **Mapping level：context-aware merged mapping**
针对 speculative decoding 的 token tree，避免把每个 tree node 当成独立 search space，改为 context-aware merged mapping，降低指数级 mapping complexity。

**相比 Baseline 的创新。**

相对 AdaInfer 这类 early-exit 方案，SpecEE 的本质优化是：**把 early-exit predictor 从“全词表分类/搜索问题”转化为“小候选集合上的验证问题”**。相对 EAGLE/vLLM 等 speculative decoding，SpecEE 是正交优化：不是只减少 LLM forward 次数，而是进一步减少每次 forward 需要走的 decoder layers。

**实现与结果。**

论文给出的核心结果是：Llama2-7B 上，cloud 场景 speedup 2.25×，PC 场景 speedup 2.43×；lightweight predictor 约 100× parameters/FLOPs reduction；two-level scheduling 约 68% predictor reduction。仓库显示其实现覆盖 HuggingFace cloud 与 llama.cpp edge/PC 场景。

Speedup=TbaselineTSpecEE\text{Speedup}=\frac{T_{\text{baseline}}}{T_{\text{SpecEE}}}

Speedup=TSpecEETbaseline

## 视角二：评审——学术评判

**创新性评估。**

这是“旧场景的新问题 + 新方法”：LLM early exiting 不是全新问题，但作者抓住了 predictor search space 这个常被忽略的 secondary bottleneck。学术价值在于把 speculative decoding 和 early exiting 两条线结合起来，提出了一个较清晰的系统化优化框架。

**进一步研究空间。**

主要风险在于 predictor 的鲁棒性：不同模型、不同 domain、不同 decoding temperature 下 speculative token 和正确 token 的 probability correlation 是否稳定？另一个问题是与量化、KV-cache compression、paged attention、continuous batching 组合时，调度策略是否仍然有效。

---

# 4. SpecInfer: Tree-based Speculative Inference and Verification

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 distributed/offloading LLM serving。自回归 decoding 每步只生成一个 token，GPU parallelism 低；同时 KV-cache 占用大，限制 batch size。Sequence-based speculative decoding 只预测一条路径，SSM 和 LLM capacity gap 会限制 acceptance rate。

**创新路径分析。**

SpecInfer 用 small speculative models, SSMs 生成 **token tree**，同时覆盖多个候选序列；LLM 不再只做 incremental decoder，而是作为 tree verifier，一次并行验证多个候选 token path。核心包括：

1. expansion-based token tree：挖掘单个 SSM 内部多样性；
2. merge-based token tree：融合多个 SSM 的候选；
3. tree-based parallel decoding：一次 LLM step 验证整棵树；
4. multi-step speculative sampling：保证 stochastic decoding 分布等价。

**相比 Baseline 的创新。**

相对 incremental decoding，SpecInfer 把每步生成 1 token 变成“每步验证多个未来 token”。相对 sequence speculative decoding，它不是押注单一路径，而是用 token tree 提高命中概率。

**实现与结果。**

论文报告 distributed LLM inference speedup 1.5–2.8×，offloading-based inference speedup 2.6–3.5×，并保持 generative performance。AE artifact 为 `goliaro/specinfer-ae`，SpecInfer 也被描述为 FlexFlow 生态中的 open-source distributed multi-GPU speculative inference system。

## 视角二：评审——学术评判

**创新性评估。**

这是“旧解码瓶颈的新 speculative structure”。树结构的引入是核心 novelty，它用更多候选换更高 acceptance probability，并通过 parallel verification 抵消额外候选成本。

**进一步研究空间。**

token tree 的收益强依赖 SSM/LLM alignment。对于 instruction-following、代码生成、多语言、high-temperature sampling，树的 branching factor 如何动态自适应仍有空间。另一个关键问题是 token tree verification 与 continuous batching、paged KV-cache 的调度耦合。

---

# 5. BBS / BitVert: Bi-directional Bit-level Sparsity

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 bit-serial DNN acceleration。传统 bit sparsity 只跳过 zero-bit，但 zero-bit 随机分布导致 load imbalance；同时仍需从 off-chip fetch 所有 bits，硬件 scheduler、MUX、shifter 开销高。量化后 INT8 value sparsity 很低，传统 value sparsity 与 quantization 存在 tension。

**创新路径分析。**

BBS 的 insight 是 bit-level sparsity 可以双向看：如果 0 少，则 1 多。于是可以 prune all-zero bit columns 或 all-one bit columns，形成 **bi-directional sparse bit columns**。这带来两个结果：

BBS Sparsity≥50%\text{BBS Sparsity}\geq 50\%

BBS Sparsity≥50%

并且比只制造 zero columns 更能保持原始 INT8 quantization levels。硬件侧 BitVert 采用 bit-serial PE，利用 balanced workload 减少 synchronization/scheduling overhead，并通过 channel reordering 支持 binary pruning。

**相比 Baseline 的创新。**

相对 BitWave/Pragmatic 等 zero-bit skipping，BBS 的本质优化是：**把“无效 bit”从单边 zero 扩展到双边 zero/one，从而同时改善压缩率、负载均衡和硬件简单性。**

**实现与结果。**

七个 DNN 模型上，平均 model size reduction 1.66×，accuracy loss <0.5%；最高 3.03× speedup 和 2.44× energy saving。公开仓库主要提供 BBS binary pruning 与 BitWave bit-flip 的 PyTorch 复现。

## 视角二：评审——学术评判

**创新性评估。**

这是“旧 bit-serial accelerator 问题的新 sparsity definition”。BBS 的概念非常干净，且与 PTQ、pruning 正交，体系结构和算法共同闭环。

**进一步研究空间。**

当前风险是模型范围：CNN/传统 DNN 上强，但 LLM 权重分布、activation outlier、group-wise quantization 下是否仍能稳定保持 <0.5% loss 需要更多验证。另一个问题是 BitVert 是否能在现代 tensor-core/NPU 数据通路中自然集成。

---

# 6. Pre-gated MoE: Fast and Scalable MoE Inference

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 MoE LLM inference，尤其是单 GPU + CPU memory offloading。MoE 能用 sparse expert activation 降低 FLOPs，但专家参数巨大；传统 MoE-offload 把 experts 放 CPU/SSD，运行时 gate 选中 expert 后才搬运，导致 expert selection 与 expert execution 串行，CPU→GPU migration latency 无法隐藏。

**创新路径分析。**

Pre-gated MoE 改变 gate 的语义：第 NNN 个 MoE block 的 pre-gate 不选择当前 block 的 experts，而是提前选择第 N+1N+1N+1 个 block 的 experts。这样 expert migration 可以与当前 block execution 重叠：

Tblock≈max⁡(Texpert_exec,Texpert_prefetch)T_{\text{block}}\approx \max(T_{\text{expert\_exec}},T_{\text{expert\_prefetch}})

Tblock≈max(Texpert_exec,Texpert_prefetch)

而不是

Tblock=Tgate+Tmigration+Texpert_execT_{\text{block}}=T_{\text{gate}}+T_{\text{migration}}+T_{\text{expert\_exec}}

Tblock=Tgate+Tmigration+Texpert_exec

**相比 Baseline 的创新。**

相对 MoE-offload，Pre-gated MoE 的本质优化是：**把 runtime-dependent expert selection 从同层依赖改成跨层预取依赖，从而隐藏专家搬运延迟。**

**实现与结果。**

论文报告其相对 oracular GPU-only solution 只增加约 23% overhead，同时比 GPU-only 降低 4.2× peak GPU memory consumption，并保持模型质量。AE artifact 已公开。

## 视角二：评审——学术评判

**创新性评估。**

这是“旧 MoE-offload 场景的新 dependency transformation”。创新点不是新的 offload pipeline，而是对模型结构本身做 algorithm-system co-design，把系统预取需求反馈到 gate function 设计。

**进一步研究空间。**

核心问题是模型质量与可训练性：pre-gate 预测下一层 expert 是否会牺牲原始 MoE 的 routing expressiveness？对于更深/更稀疏/更动态的 MoE，如 Mixtral、DeepSeekMoE 风格，跨层 routing correlation 是否足够强，还需要进一步验证。

---

# 8. Duplex: xPU + Logic-PIM for MoE/GQA/Continuous Batching

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是带 MoE、GQA 和 continuous batching 的 LLM inference。MoE 和 attention 在 continuous batching 下频繁访问 DRAM，呈现低 arithmetic intensity, Op/B；GPU 对低 Op/B 层利用率低。但传统 PIM 又只适合 Op/B < 1 的极低算术强度，无法处理 MoE 层 Op/B 波动和 GQA attention 的 4–8 Op/B。

**创新路径分析。**

Duplex 是单 device 内的 heterogeneous architecture：xPU 处理 high-Op/B，Logic-PIM 处理 low-to-medium Op/B。Logic-PIM 利用更多 TSV，把较强 processing units 放在 HBM logic die，而不是 DRAM die 内的弱 PIM ALU。

系统层有两个 co-processing：

1. **Expert co-processing**：token 多的 experts 给 xPU，token 少的 experts 给 Logic-PIM；
2. **Attention co-processing**：prefill/input request 的 high-Op/B attention 给 xPU，ongoing decode 的 low-Op/B attention 给 Logic-PIM。

**相比 Baseline 的创新。**

相对 GPU-only，Duplex 降低 DRAM bandwidth bottleneck。相对 PIM-only，Duplex 不把所有 low-ish Op/B 都交给弱 PIM，而是按 Op/B 动态选择 xPU/Logic-PIM。其本质是：**用 arithmetic intensity 作为 layer/device binding 的一等调度信号。**

**实现与结果。**

论文报告相比 GPU baseline，最高 2.67× throughput、2.57× lower E2E latency、42.03% less energy。公开 simulator 基于 Duplex MICRO’24。

## 视角二：评审——学术评判

**创新性评估。**

这是“新 LLM 结构组合带来的新硬件 mapping 问题”。它比单纯 attention PIM 或 MoE accelerator 更完整，因为考虑了 continuous batching 下 Op/B 动态变化。

**进一步研究空间。**

盲点在于物理可实现性：Logic-PIM 的 TSV 数量、HBM logic die area/power、热密度、memory vendor integration 都是高风险。另外 MoE expert load imbalance 在真实在线流量中可能更复杂。

---

# 9. llm.npu: Fast On-device LLM Inference with NPUs

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 mobile/on-device LLM inference，尤其是 UI automation、email reply 等长 prompt 短输出任务。瓶颈是 prefill stage：移动端 CPU/GPU parallel compute 弱，而 NPU 适合 INT vector ops，但 COTS mobile NPU 通常只支持 static shape，且对 FP ops、per-group quantization 支持差。

**创新路径分析。**

llm.npu 从三个层级重构：

1. **Prompt level：chunk-sharing graph**
把 variable-length prompt 拆成 fixed-size chunks，用预构建 NPU graph，避免每个 prompt 重新编译 graph。
2. **Tensor level：shadow outlier execution**
activation outlier channel 很少，把 outlier 抽出来在 CPU/GPU 旁路执行，NPU 保持 per-tensor MatMul 高效。
3. **Block level：out-of-order subgraph execution**
不严格按 chunk 顺序执行，而是调度 CPU/GPU FP ops 与 NPU INT ops，减少 NPU stalls。

**相比 Baseline 的创新。**

相对 llama.cpp、TFLite、MNN、MLC-LLM、PowerInfer-v2，llm.npu 的本质优化是：**不是让 NPU 完整跑 LLM，而是把 NPU 的 static-shape/int-only 优势和 CPU/GPU 的 FP/outlier 处理能力进行细粒度协同。**

**实现与结果。**

实现基于 MLLM/QNN，约 10K 行 C/C++/assembly。论文报告平均 22.4× faster prefill、30.7× energy savings，真实应用最高 32.8× E2E speedup，并首次在 COTS mobile device 上实现 billion-sized model >1000 tokens/s prefilling。代码在 `UbiquitousLearning/mllm`。

## 视角二：评审——学术评判

**创新性评估。**

这是“新部署场景下的系统 co-design”。移动端 NPU 运行 LLM 是强现实问题，论文贡献在于绕开 COTS NPU 限制，而不是假设理想 NPU。

**进一步研究空间。**

decode stage 仍可能是瓶颈，尤其长输出任务。另一个风险是厂商 NPU SDK 差异大，QNN 上的优化是否可迁移到 Apple Neural Engine、MediaTek APU、Google Edge TPU，需要更多实证。

---

# 12. Tandem Processor: Emerging Non-GEMM Operators in Neural Networks

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是端到端 neural network acceleration。传统 NPU 过度聚焦 GEMM/Conv，但现代模型，尤其 BERT/GPT，引入大量 non-GEMM ops：Softmax、GeLU、ReduceMean、Transpose、Reshape、Cast 等。随着 GEMM unit 越来越强，Amdahl’s bottleneck 转移到 non-GEMM 和数据搬运/编排。

**创新路径分析。**

Tandem Processor 是 GEMM unit 的 on-chip companion processor。它不是固定 function block，也不是普通 RISC-V core，而是 specialized yet programmable：

1. memory access logic 专门化，降低 register file 与 load/store overhead；
2. ISA/microarchitecture 支持 strided iterator 和 nested-loop style data access；
3. arithmetic 仍用 primitive vector ops 保持 mathematical programmability；
4. 与 GEMM unit 紧耦合，负责 non-GEMM execution 和 operand orchestration。

**相比 Baseline 的创新。**

相对 dedicated blocks：更可编程，能覆盖增长的 non-GEMM diversity。

相对 on-chip general-purpose core：数据访问更专用，能跟上 GEMM throughput。

本质优化是：**把 non-GEMM 从“边角料处理”提升为 NPU end-to-end dataflow 的一等公民。**

**实现与结果。**

提供 synthesizable RTL、compiler、FPGA/ASIC flow，并给出 floorplan/post-layout analysis。结果包括：相对 dedicated-block/host 方案 2.7× speedup、20.6× energy reduction；相对 on-chip multi-core RISC-V 5.9×；相对 TPU-like vector unit 2.6× speedup、1.4× energy reduction。GeneSys 仓库公开。

## 视角二：评审——学术评判

**创新性评估。**

这是“旧 NPU 架构假设的新反驳”。学术生命力很强，因为模型算子生态不断变化，non-GEMM 的比例和结构复杂度只会增加。

**进一步研究空间。**

挑战在 compiler：如何从 PyTorch/XLA/ONNX 自动识别 non-GEMM pattern 并高效 lower 到 Tandem ISA。另一个问题是与 transformer-specific fused kernels，比如 FlashAttention、RMSNorm+MatMul fusion 的边界如何划分。

---

# 8-bit Transformer Inference and Fine-tuning for Edge Accelerators

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 edge accelerator 上的 Transformer inference 与 fine-tuning。int8 inference 虽成熟，但 training/fine-tuning 需要更大 dynamic range；FP8 已用于 training，但 prior work 多只量化 GEMM inputs，非 GEMM ops 仍保留高精度，导致 memory 和 datapath 复杂度没有完全下降。

**创新路径分析。**

作者比较 FP8 与 Posit8。Posit8 采用 tapered precision，在接近 1 的数值附近精度更高，适合 Transformer weights/activations。核心技术：

1. **all-operation 8-bit quantization**
不只 GEMM，forward/backward 中更多操作也量化到 FP8 / Posit8。
2. **operation fusion**
降低量化误差传播，使 inference accuracy loss <1%。
3. **8-bit LoRA fine-tuning**
将 LoRA 适配到 FP8/Posit8，使 fine-tuning GEMM 也能用单一 8-bit datatype。
4. **Posit8 softmax hardware**
用 bitwise operations 近似 exponential 和 reciprocal，降低 vector unit area/power。

**相比 Baseline 的创新。**

相比 int8，FP8/Posit8 避免大量 scaling factors / outlier handling。相比只量化 GEMM 的 FP8 training，本文的本质优化是：**把 Transformer 的非 GEMM 数据通路也拉入 8-bit 设计空间**。

**实现与结果。**

论文显示 FP8 与 Posit8 可达到接近 BFloat16 的 inference / fine-tuning accuracy；Posit8 vector unit 比 FP8 accelerator 的 vector unit 平均小 33%、功耗低 35%；整体 accelerator area 分别降低 30%/34%，power 降低 26%/32%。

## 视角二：评审——学术评判

**创新性评估。**

这是“旧量化问题的新 datatype + 全算子覆盖”。贡献不只是提出 Posit8，而是把 Transformer fine-tuning 的 backward / LoRA / softmax 都纳入 edge hardware co-design。

**进一步研究空间。**

主要盲点是大模型规模和现代 LLM 算子：RMSNorm、RoPE、GQA/MQA、SwiGLU、KV-cache quantization 下 Posit8 是否继续稳健。另一个问题是 Posit8 工具链成熟度与商用硬件支持。

# H2-LLM: Hybrid-Bonding Heterogeneous Accelerator

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 edge-side low-batch LLM inference。prefill compute-intensive，decode memory-intensive；传统 in-die NMP 放在 DRAM die 内，受限于 DRAM process，compute capacity 不足。低 batch 场景下，centralized processor 与 NMP 都可能 sub-optimal。

**创新路径分析。**

H2-LLM 利用 hybrid bonding，把 memory bandwidth 与 logic die compute capacity 结合。它同时探索：

1. **architecture design space**：compute capacity 与 bandwidth 的 trade-off；
2. **data-centric dataflow abstraction**：不固定 operator-to-device mapping，而是围绕数据移动与 stage 特性做 dataflow exploration；
3. **DSE framework**：自动搜索 architecture + dataflow 最优组合。

**相比 Baseline 的创新。**

相比 in-die NMP，H2-LLM 的本质优化是：**把近存计算从 DRAM die 中解放到 hybrid-bonded logic die，从而同时提升 bandwidth-proximity 和 compute density**。相比 GPU-only，它专门针对 low-batch decode 的 memory-bound 约束。

**实现与结果。**

论文报告相比 existing in-die NMP-based heterogeneous accelerators，H2-LLM 达到 2.72× geomean speedup 和 1.48× geomean better energy efficiency。论文声明 data-centric dataflow exploration framework 已开源。

## 视角二：评审——学术评判

**创新性评估。**

这是“新封装技术带来的新 architecture/dataflow co-exploration”。学术生命力强，因为 hybrid bonding 正在成为 memory-compute integration 的重要路线。

**进一步研究空间。**

关键风险是技术假设：hybrid bonding 的成本、良率、散热、controller 面积占比、edge device power envelope 都可能限制落地。未来需要更接近 silicon 的 floorplan、thermal、yield-aware DSE。

# MAGIS: Graph Transformation + Scheduling for Memory Optimization

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 DNN training/inference 的 peak memory optimization。传统 graph scheduling 通过 rematerialization、swapping、reordering 调整 tensor lifetime，但会引入 recomputation / transfer overhead，且无法改变 tensor shape。MAGIS 观察到 graph transformation 可以改变 tensor shape 和 graph structure，提供新的 memory-performance tradeoff。

**创新路径分析。**

MAGIS 引入 **Fission Transformation, F-Trans**：把大 operator / subgraph 拆成更小块，降低 tensor shape 和 peak memory，但可能降低 hardware utilization。

为控制搜索复杂度：

1. 用 **Fission Hierarchy Tree, F-Tree** 表达 F-Trans 后结构；
2. 用 graph analysis 构造 lightweight search space；
3. 把 scheduling 分解为 graph transformation + re-ordering；
4. 使用 incremental scheduling 降低每步 transformation 后的调度开销。

**相比 Baseline 的创新。**

相比 Checkmate、Capuchin、POET 等 scheduling-only 方法，MAGIS 的本质优化是：**不仅移动 tensor 生命周期，还改变 tensor shape 本身**。

Peak Memory=max⁡t∑v∈Live(t)size(v)\text{Peak Memory}
=
\max_t \sum_{v \in \text{Live}(t)} \text{size}(v)

Peak Memory=tmaxv∈Live(t)∑size(v)

MAGIS 同时作用于 Live(t)\text{Live}(t)Live(t) 和 size(v)\text{size}(v)size(v)。

**实现与结果。**

相比 SOTA，MAGIS 在相同 latency constraint 下 peak memory 仅为 baseline 的 15%–85%；在不超过 10% latency overhead 时，可把原始 peak memory 优化到 15%–50%；同 memory constraint 下最高 1.25× speedup。

## 视角二：评审——学术评判

**创新性评估。**

这是“旧 memory scheduling 问题的新搜索维度”。将 graph transformation 与 graph scheduling 合并，是比较扎实的 compiler/system contribution。

**进一步研究空间。**

主要问题是对 dynamic shape、LLM variable sequence length、operator fusion backend 的兼容。F-Trans 可能破坏已有高性能 fused kernels，因此需要与 TensorRT/XLA/TVM 的 lowering pipeline 更紧密结合。

# FlashGen: Multi-turn Dialogue KV Management

[https://github.com/Sys-KU/LMServe](https://github.com/Sys-KU/LMServe)

## 视角一：作者——设计逻辑

**场景与瓶颈。**

目标是 multi-turn dialogue LLM serving。多轮对话导致 prompt amplification：每轮都把历史拼接进 prompt，vLLM / TensorRT-LLM 常通过 recomputation 处理 previous turns，造成大量 KV recomputation。另一个问题是 FCFS scheduler 遇到长 prompt 时产生 head-of-line blocking，GPU memory 闲置。

**创新路径分析。**

FlashGen 两个核心模块：

1. **FlashGen-Cache**
多级 KV cache：GPU memory + CPU DRAM + SSD。保留 previous turns 的 attention KVs，用低成本 restoration 替代 recomputation。
2. **FlashGen-Sched**
request reordering。当队首 request 因内存不足不可运行时，允许较新的 runnable request 先执行；同时用 preemption 防止 starvation。

**相比 Baseline 的创新。**

相比 vLLM 的 PagedAttention，FlashGen 的本质优化是：**把多轮对话历史当成可跨 memory hierarchy 管理的 persistent KV state，而不是每轮重新构造 prompt 计算**。

**实现与结果。**

论文基于 vLLM 实现，在 Azure Standard_NC48ads_A100_v4，2×A100 80GB、440GB host memory、2×960GB NVMe SSD 环境评估。ShareGPT 上，OPT-30B 和 Llama-2-70B 分别达到 1.63× 和 2.85× throughput improvement，latency 保持在类似边界。

## 视角二：评审——学术评判

**创新性评估。**

这是“旧 KV-cache 问题在 multi-turn dialogue 场景下的新资源管理”。贡献不在单个 kernel，而在把 GPU/CPU/SSD memory hierarchy 与 scheduler 联动。

**进一步研究空间。**

SSD restoration 在更高并发下可能受 PCIe / NVMe bandwidth 限制。另一个问题是 multi-tenant serving 下，KV cache 的隐私隔离、安全擦除、session eviction policy 还需要系统化设计。

# Exploring the Performance Improvement of Tensor Processing Engines through Transformation in the Bit-weight Dimension of MACs

**开源仓库。**

论文和 GitHub 均给出官方仓库：`wqzustc/High-Performance-Tensor-Processing-Engines`。仓库包含 Verilog/SystemVerilog、Tcl、Makefile、仿真与综合脚本，目录按 OPT1、OPT2、OPT3/OPT4C 组织；README 说明使用 Synopsys Design Compiler、VCS、PrimeTime 等工具，并提供 PE/array 级功能仿真与综合流程。

---

## 视角一：作者（体系结构专家）——“设计逻辑”

### 1. 场景与瓶颈：Tensor Processing Engine 的 MAC 内部瓶颈

这篇文章面向 **Tensor Processing Engines, TPEs**，也就是 GPU Tensor Core、NPU、DSA 中用于 GEMM/MM 的核心矩阵乘单元。传统体系结构优化通常关注 **Dataflow**、operand reuse、systolic array mapping，例如 OS/WS systolic array、3D-Cube、multiplier-adder tree、2D-Matrix 等；而 arithmetic 研究则单独优化 multiplier / adder / compressor tree。作者认为这两条线长期是“割裂”的：架构层看 PE array，电路层看单个 MAC，但缺少把 **GEMM triple loop** 与 **MAC 内部 bit-weight dimension** 统一起来的视角。

核心物理痛点是：传统 INT MAC 内部的 **high-width full adder / accumulator** 是 timing、area、power bottleneck。论文 Figure 1 明确标注 full adder 和 high-width accumulator 是 MAC 中 latency 与 area 的关键瓶颈。

可以把传统 MAC 抽象成：

C=A×B=∑bw=0BW−1SubAbw×BC = A \times B = \sum_{bw=0}^{BW-1} SubA_{bw} \times B

C=A×B=bw=0∑BW−1SubAbw×B

其中 bwbwbw 是 bit-weight dimension；传统 TPE 设计通常把这个维度隐藏在 MAC 内部，而本文把它显式暴露出来，作为新的优化维度。论文把 8-bit MBE 和 complement bit-serial 都统一表达为 bit-weight decomposition。

---

### 2. 创新路径分析：从 PE array 维度深入到 bit-weight 维度

作者的核心创新不是提出一个全新的 array topology，而是提出一种 **finer-grained TPE notation**，把 GEMM 的 triple loops 拆到 PE microarchitecture 层面，进一步把 MAC 内部的 partial products, PPs、bit-weight、shift、compress、accumulate 等操作显式化。论文提出四类优化技术，分别对应 GitHub 中的 OPT1、OPT2、OPT3/OPT4C 等实现。

### OPT1：Compressed Accumulative PE

传统 MAC 的瓶颈在于每个 PE 内做 full-width accumulation。OPT1 的思路是减少或延后高位宽累加，把部分累加压缩到 sum/carry 或 compressor-friendly representation 中，降低 full adder / accumulator 的 critical path 和面积压力。

直观理解：

Traditional MAC: PP→Compressor→FullAdder→Accumulator\text{Traditional MAC: } PP \rightarrow Compressor \rightarrow FullAdder \rightarrow Accumulator

Traditional MAC: PP→Compressor→FullAdder→Accumulator

OPT1: PP→Compressed Accumulation→Delayed/Reduced FullAdder\text{OPT1: } PP \rightarrow Compressed\ Accumulation \rightarrow Delayed/Reduced\ FullAdder

OPT1: PP→Compressed Accumulation→Delayed/Reduced FullAdder

这类优化特别适合 systolic array / 3D-Cube 中大量 PE 重复出现的场景，因为单 PE 的 full adder 改善会乘以整个 array 的规模。

### OPT2：Same Bit-weight Compressor Array for GEMM

OPT2 利用 bit-weight dimension 重新组织 GEMM 的 reduction。传统做法是每个 PE 独立完成某个 A×BA \times BA×B 的 PP reduction；OPT2 则尝试把**相同 bit-weight 的 partial products** 组织到同一 compressor structure 中，减少重复 shift / accumulation。

本质上，它把：

∑kAikBkj=∑k∑bwSubAik,bwBkj\sum_k A_{ik}B_{kj}
=
\sum_k \sum_{bw} SubA_{ik,bw}B_{kj}

k∑AikBkj=k∑bw∑SubAik,bwBkj

重排为：

∑bw∑kSubAik,bwBkj\sum_{bw} \sum_k SubA_{ik,bw}B_{kj}

bw∑k∑SubAik,bwBkj

这样可以在 GEMM reduction 维度和 bit-weight 维度之间做 loop transformation，形成更硬件友好的 compressor array。论文把这称为从 PE microarchitecture perspective 重新看 GEMM loop。

### OPT3 / OPT4：Sparse / Encoder-based Bit-slice PE Array

这部分针对 bit-slice / bit-serial architecture。传统 Radix-2 bit-serial 依赖跳过 0 bit，但要处理 bit index、shift unit、accumulator，且速度依赖 non-zero bit-slices 数量。本文用 encoder-based Radix-4 / sparse encoding，把 bit-slices 变成更高 radix 的 partial product selection，减少 PPs 数量，并改善 bit-slice 架构中的 shift 和 accumulator 开销。论文 Figure 2 对比了 traditional MAC、Radix-2 bit-serial、Radix-2 bit-interleaved、proposed Radix-4 bit-serial / bit-interleaved 等结构。

这里的核心不是简单“跳零”，而是：

bit-slice extraction→encoding→fewer / denser PPs\text{bit-slice extraction} \rightarrow \text{encoding} \rightarrow \text{fewer / denser PPs}

bit-slice extraction→encoding→fewer / denser PPs

所以它比 Laconic 这类 bit-serial sparse accelerator 更进一步：不仅利用 sparsity，还通过 encoder transformation 改变 PP 生成与 reduction 的形态。

# SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs

**开源仓库。**

论文中明确给出官方仓库：`microsoft/SeerAttention`。仓库 README 描述其为 SeerAttention / SeerAttention-R 的官方实现，包含 trainable sparse attention、block-level sparsity、self-distillation、block-sparse FlashAttention kernel 等组件；

### 2. 创新路径分析：用 learnable AttnGate 学习 block-level attention sparsity

SeerAttention 的核心思想是：

**不要人工规定 sparse pattern，而是让模型自己学习哪些 attention blocks 重要。**

它借鉴 MoE 的 gating 思想，在标准 attention 旁边加入一个轻量的 **AttnGate**，用于预测 block-level attention mask。整个过程可以理解为：

(Q,K)→AttnGate→Block Mask→Block-sparse FlashAttention(Q,K) \rightarrow \text{AttnGate} \rightarrow \text{Block Mask} \rightarrow \text{Block-sparse FlashAttention}

(Q,K)→AttnGate→Block Mask→Block-sparse FlashAttention

### Step 1：Sequence pooling 降低 gate 成本

如果直接在 token-level 预测 attention mask，成本仍接近 O(n2)O(n^2)O(n2)，失去稀疏化意义。因此 SeerAttention 先沿 sequence dimension 对 Q,KQ,KQ,K 做 block pooling，把 token-level 表示压缩到 block-level 表示。

### Step 2：Learnable linear layers 建模 block relevance

pooled Q,KQ,KQ,K 经过 learnable linear layers 得到 block-level representation，然后矩阵相乘得到 gating scores。论文摘要中明确描述：gate 先对 Q,KQ,KQ,K 沿 sequence dimension pooling，再经过 learnable linear layers，最后相乘生成 gating scores，用于预测 block-level attention sparsity。

### Step 3：TopK / Threshold 生成 binary block mask

SeerAttention 支持两种将 floating-point gating scores 转换为 binary mask 的方式：

训练好 AttnGate 后，用户可以在 test time 调整 TopK ratio 或 threshold，在 accuracy 和 latency 之间切换。

### Step 4：Block-sparse FlashAttention kernel 跳过无效 block

SeerAttention 将 AttnGate 的 block size 与 FlashAttention tiling size 对齐，通常是 64 或 128。这样 binary block mask 可以直接指导 customized block-sparse FlashAttention kernel 跳过未激活 block 的 I/O 和 compute。论文当前实验中 block size 固定为 64，并且 AttnGate 只应用于 prefill stage。

## 相比 Baseline 的创新（Vs. Baseline）

### 相比 FlashAttention-2

FlashAttention-2 仍然计算 dense attention，只是通过 tiling 降低 HBM traffic 并提高 GPU kernel efficiency。SeerAttention 的本质优化是：

**不是更高效地算完整 attention，而是学习哪些 attention block 可以不算。**

FlashAttention-2:efficient dense attention\text{FlashAttention-2}: \text{efficient dense attention}

FlashAttention-2:efficient dense attention

SeerAttention:learned block-sparse attention+sparse FlashAttention kernel\text{SeerAttention}: \text{learned block-sparse attention} + \text{sparse FlashAttention kernel}

SeerAttention:learned block-sparse attention+sparse FlashAttention kernel

### 相比 MInference

MInference 使用 predefined sparse patterns，例如 Vertical-Slash，并在 runtime 用 heuristic 生成 sparse indices。SeerAttention 不依赖固定 pattern，而是通过 AttnGate 对不同 context/head 自适应预测 block mask。论文实验中 MInference 在 Llama-3.1-8B-Instruct 上使用 Vertical-Slash pattern；SeerAttention 的 kernel-level speedup 更能随 sparsity 线性增长。

### 相比 MoA

MoA 使用 offline search，为不同 heads 校准 A-shape sparse pattern。SeerAttention 不需要人工或离线搜索静态 pattern；训练后的 gate 可以在 test time 通过 TopK/threshold 调整 sparsity ratio。

### 相比 DuoAttention

DuoAttention 将部分 heads 作为 streaming heads，其他 heads 保持 dense。它是 head-level 选择，而 SeerAttention 是 **block-level sparsity**，粒度更细，能在同一个 head 内动态选择不同 attention blocks。

一句话总结：

> SeerAttention 的核心创新是把 sparse attention 从“人工 pattern / head-level heuristic”变成“learned block-level routing problem”，并用 block-sparse FlashAttention kernel 把理论 sparsity 转化为实际 GPU speedup。
> 

**Mind the Gap (Orojenesis): Attainable Data Movement and Operational Intensity Bounds for Tensor Algorithms** (ISCA'24 Best Paper Nominee, NVIDIA + MIT)