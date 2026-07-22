# Paper-Chance-Result

[https://tingcao952.github.io/publications/](https://tingcao952.github.io/publications/)

## 第1批：多模态

| 论文标题 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源/环境 | 备注 |
| --- | --- | --- | --- | --- |
| **[arxiv'26] vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models** | 全解耦的 Any-to-Any 多模态服务系统；通过 stage 抽象将复杂模型分解为图状互联阶段，每阶段独立服务（LLM 引擎或扩散引擎） | **问题**：现有服务系统(vLLM/SGLang)只针对单一范式(AR LLM 或 DiT)，对组合 AR+DiT+多编码器的 any-to-any 模型(如 Qwen-Omni、LongCat-Flash-Omni)不支持跨阶段交互优化，开发者需手动处理。**方法**：① stage 抽象 + 解耦执行后端；② 每 stage 独立请求批处理 + 灵活 GPU 分配；③ 基于 Mooncake 的统一 inter-stage connector(TCP/RDMA)。JCT 降低 91.4% | ✅ **开源** [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)，已被 vLLM 社区官方接纳 | A2A |
| **[arxiv'26] VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents** | VLM Agent 的 17 个长程交互式环境(symbolic puzzles、real-image、navigation、manipulation)；提供 multi-step solver 自动生成 SFT 演示 | **问题**：现有 VLM 在多步视觉交互中能力欠测，已有 benchmark 多为静态单步评估。**方法**：基于 Gymnasium-Robotics 构建；可配置难度/输入表征/规划视野/反馈；前沿模型 Gemini-3-Pro 在 Hard 模式仅 26%，揭示长 context 反而掉点 | ✅ **开源** [visgym/VisGym](https://github.com/visgym/VisGym)；HuggingFace 提供数据+模型；基于 Gymnasium | benchmark |
|  |  |  |  |  |
|  |  |  |  |  |
| **[arxiv'25] Cornserve: Efficiently Serving Any-to-Any Multimodal Models** | 通用 A2A 多模态服务平台；offline planner + record-and-replay 分布式 runtime；自动决定是否解耦 | **问题**：A2A 模型有请求类型异质性、计算路径异质性、各组件资源需求差异。**方法**：用户描述计算图，planner 自动找最优部署；3.81× 吞吐 / 5.79× 尾延迟降低 | ✅ **开源** [cornserve-ai/cornserve](https://github.com/cornserve-ai/cornserve)，基于 Kubernetes，~23K Python LoC | A2A |
| **[arxiv'25] FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos** | 全自动视频运动数据标注 pipeline；用 detector+tracker 抽取轨迹，喂给 LLM 生成细粒度 caption 和 QA | **问题**：精细运动数据标注成本高(10 人 100 天/10 万视频)；缺乏大规模 motion-spatial 训练集。**方法**：trajectory JSON + frames → LLM 生成；微调 NVILA-15B/Qwen2.5-7B 后超越 Qwen2.5-VL-72B 与 Gemini-2.5 Flash | ✅ **开源** [Wolfv0/FoundationMotion](https://github.com/Wolfv0/FoundationMotion)；模型 weights 可下载 | 新pipeline |
| **[CVPR26] MoDES: Accelerating Mixture-of-Experts Multimodal LLMs via Dynamic Expert Skipping** | 训练-free 的 MoE-MLLM 推理加速；首个针对多模态的专家跳过框架 | **问题**：现有专家跳过方法(为单模态 LLM 设计)套用到 MoE-MLLM 显著掉点——未考虑跨层专家贡献异质性、跨模态 token 行为差异。**方法**：① GMLG(全局调制局部门控) 估计 per-token 专家重要度；② DMT(双模态阈值) 分模态处理；③ frontier search(利用单调性，几小时收敛)。88% 跳过率下 Qwen3-VL-MoE-30B 反提升 10.67%；prefilling 2.16×，decoding 1.26× | ✅ **开源** [ModelTC/MoDES](https://github.com/ModelTC/MoDES) | 新模型 |
| **[SoCC'25] ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving** | LMM 服务系统；分模态分阶段解耦 + 模态感知调度自动扩缩容 | **问题**：单体 LMM 服务对 decoder-only / cross-attention 两类架构都不优；生产负载有重尾分布、突发流量。**方法**：模态感知 stage 解耦 + 自适应扩缩容；3.3-5.5× 吞吐(25-41.3% 成本节省) | ⚠️ **数据集开源**(Azure LMM 推理 traces 在 [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset))；系统代码未开源 | benchmark |
| **[arxiv'25] FlowMM: Cross-Modal Information Flow Guided KV Cache Merging** | 多模态长上下文 **KV cache merge**(不是 evict)；按层动态选择跨模态 vs 内模态合并策略 | **问题**：KV evict 会信息丢失；现有 merge 方法没考虑模态分布偏差和跨模态注意力偏差。**方法**：① 跨模态信息流分析决定层级策略；② sensitivity-adaptive token matching(joint similarity + sensitivity)。KV 内存降 80-95%，decoding 加速 1.3-1.8×；plug-and-play、无需微调 | ❌ 论文未提供代码链接(OpenReview 投稿) | 新pipeline |
| **[arxiv'25] OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM** | NVIDIA 开源 omni-modal LLM；三个架构创新 + 24M 数据合成 pipeline | **问题**：omni-modal 模型(Qwen2.5-Omni 等)训练 token 量大(1.2T)，跨模态时序对齐弱。**方法**：① OmniAlignNet(视-音对齐)；② Temporal Embedding Grouping；③ Constrained Rotary Time Embedding。仅 0.2T tokens(6× 减少)就在 DailyOmni +19.05、Video-MME +3.9 超越 Qwen2.5-Omni | ✅ **开源** [NVlabs/OmniVinci](https://github.com/NVlabs/OmniVinci)；模型 [nvidia/omnivinci](https://huggingface.co/nvidia/omnivinci) | 新模型 |
| **[arxiv'25] Fast-dLLM v2: Efficient Block-Diffusion LLM** | 把预训练 AR LLM 转为 block diffusion LLM(仅 ~1B token 微调，比 Dream 580B 少 500×) | **问题**：full-attention diffusion LLM 训练成本极高；AR LLM 解码顺序限制吞吐。**方法**：① block diffusion + 互补 attention mask；② 分层 cache(block-level + sub-block dual-cache)；③ token shift 机制。比 Qwen2.5-7B-Instruct 吞吐 高 2.54×，GSM8K +5.2% | ✅ **开源**(同 Fast-dLLM repo) [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)；模型 Fast_dLLM_v2_7B；ICLR'26 | 新模型 |
| **[arxiv'25] Fast-dLLM v1: Training-free Acceleration of Diffusion LLM by KV Cache + Parallel Decoding** | Training-free 的扩散 LLM 加速；block-wise approximate KV cache + confidence-aware 并行解码 | **问题**：开源扩散 LLM(LLaDA、Dream)实际比 AR 慢——缺 KV cache + 并行解码降质。**方法**：① 块状近似 KV cache + DualCache(prefix+suffix)；② 阈值控制下并行解码 commit。最高 27.6× 吞吐提升、精度损失最小 | ✅ **开源** [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)；ICLR'26 | 新模型 |
|  |  |  |  |  |
| **[arxiv'25] Dimple: Discrete Diffusion Multimodal LLM with Parallel Decoding** | 首个 Discrete Diffusion MLLM；两阶段训练范式(AR phase + Diffusion phase) | **问题**：纯 discrete diffusion 训练不稳定、长度偏置严重。**方法**：① 先 AR alignment + instruction tuning，再切到 diffusion masked LM；② confident decoding 动态调整每步 token 数；③ 1.5-7× 加速；超越 LLaVA-NEXT +3.9% | ✅ **开源** [yu-rp/Dimple](https://github.com/yu-rp/Dimple)；模型 [rp-yu/Dimple-7B](https://huggingface.co/rp-yu/Dimple-7B) | 新pipeline |
| **Multimodal Large Diffusion Language Models (NeurIPS 2025)** |  |  | [https://github.com/Gen-Verse/MMaDA](https://github.com/Gen-Verse/MMaDA) | 新模型 |
|  |  |  |  |  |

第 1 批小结（开源 & 创新维度）

**强烈推荐复现路径(开源完整 + 创新清晰)：**

1. **vLLM-Omni** — vLLM 官方接管，最权威的 any-to-any 服务基线
2. **Cornserve** — Kubernetes 部署，自动 planner，3.81× 吞吐
3. **MoDES** — Training-free，MoE-MLLM 加速最直接的尝试
4. **Fast-dLLM v1/v2** — 扩散 LLM 加速代表作，NVIDIA 维护
5. **Dimple** — Discrete Diffusion MLLM 首次开源完整实现
6. **OmniVinci** — 训练 token 量 6× 减少；可直接调用

**算法/硬件创新值得深读：**

- **DSV (ASPLOS'26)** — 视频 DiT 动态稀疏 + 异质 CP，硬件感知 kernel fusion 设计
- **EPD-Serve** — Ascend NPU 上的 E/P/D 解耦(国产硬件参考)
- **FlowMM** — KV merge 比 evict 更优的多模态实证

## 第 2 批：Hybrid LLMs（共 2 篇）

| 论文标题 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源/环境 | 备注 |
| --- | --- | --- | --- | --- |
| **[MICRO'25] HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models** | 首个 Hybrid Transformer-Mamba 统一硬件加速器；为 FA-2 和 SSD（State-Space Duality / Mamba-2）两种 kernel 设计统一流水线 | **问题**：Hybrid 模型(Jamba 类)在 GPU 上跨 seq-len/batch 难以保持高利用率，FA-2 与 SSD 计算特性异构、瓶颈漂移；现有专用 Mamba 加速器(LightMamba/FastMamba)不支持 GEMM，统一加速器(MARCA/HCSAs)对中间 element-wise 频繁读写效率低。**方法**：① **PipeFlash**——细粒度流水线 dataflow 化解 FA-2 的依赖瓶颈；② **PipeSSD**——融合 block-based SSD 计算的细粒度流水线；③ 统一硬件原生支持二者。中间数据 FA-2 减少 4.8×，SSD 减少 11×，DRAM 流量减 6.8×；端到端最高 2.08×、批处理 1.76× 加速；面积/功耗降 89.8%/63.8%；on-chip SRAM 仅需 GPU baseline 的 1/5.5 | ❌ **未开源**(硬件加速器；KAIST 团队) | Hybrid模型 |
|  |  |  |  |  |

## 第 3 批：Attention Optimization（共 14 篇）

| 论文标题 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源/环境 | 备注 |
| --- | --- | --- | --- | --- |
| **[PPoPP'26] MetaAttention: A Unified and Performant Attention Framework across Hardware Backends** | 跨硬件后端统一的 attention 框架；两个抽象操作 (relevance scoring + aggregation) + 自定义函数 | **问题**：FlashAttention 等针对特定算法/平台手工调优，不能泛化到 RetNet/Mamba2/MLA 等变体或不同硬件。**方法**：① pythonic 接口定义任意 attention；② 基于 IntermediateTensor 的搜索找最优 tiling 和并行策略。比 SOTA 不支持的配置最高 10.4× 加速；与手工 FlashMLA 性能相当 | ✅ **开源** [SJTU-IPADS/MetaAttention](https://github.com/SJTU-IPADS/MetaAttention)（前身：[microsoft/AttentionEngine](https://github.com/microsoft/AttentionEngine)） | 不同GPU后端的编译框架 |
| **[PPoPP'26] FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism** | 把 softmax primitives 卸载到空闲的 Tensor Cores；设计可张量化的 online softmax | **问题**：FlashAttention 的 GEMM 跑在 Tensor Cores，softmax 跑在慢的 CUDA cores（vector unit），二者解耦——A100 上 non-matmul FLOP 比 matmul 贵 16×。**方法**：① operand value assignment 把 MMA 指令重用为 element-wise 操作；② tensorized online softmax 保持数值稳定；③ 架构感知调度跨 tensor/vector 并行。Hopper H100 上 vector interval 占比降到 2.7%，比 FA-2/FA-3 平均提升 1.17× | ✅ **开源** Artifact (Zenodo)：[https://doi.org/10.5281/zenodo.17673796](https://doi.org/10.5281/zenodo.17673796) | GPU pipeline |
| **[arxiv'25] BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding** | 用单一标量阈值 + 在线 softmax 统计量动态跳过 attention block | **问题**：现有稀疏方法要训练(SeerAttention)、需预计算(MInference)、只优化 prefill 或 decode 单一阶段、缺新 GPU 内核支持、难融入框架。**方法**：复用 online softmax 已有统计量，零开销决策跳过；同时跳过 softmax/V load/P·V 矩阵乘法。预填充 1.62× @ 74.7% 稀疏；解码 1.48× @ 73.2% 稀疏；同时支持 MHA/GQA/MQA/MLA | ✅ **整合到 TensorRT-LLM** ([NVIDIA/TensorRT-LLM#9821](https://github.com/NVIDIA/TensorRT-LLM/pull/9821))，FlashInfer 集成中 | 稀疏pipeline |
| **[SC'25] UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling** | 三级层次化 context-tiling (node/device/kernel) 来高效并行不规则 attention | **问题**：分布式 attention 在 block sparse 等不规则模式下，搜索空间指数级增长；kernel 间依赖复杂。**方法**：① ILP 在 node/device 级找通信最小+负载均衡的 tile；② 并行依赖图描述 kernel 依赖；③ 贪心法挖掘 kernel-level tile；④ 运行时 ILP 编排通信和计算 | [https://github.com/oliverYoung2001/UltraAttn](https://github.com/oliverYoung2001/UltraAttn) | 分布式Attn |
| **[SC'25] RingX: Scalable Parallel Attention for Long-Context Learning on HPC** | 针对 HPC 系统的 ring attention 优化族(ringX1-4)；优化双向/causal 两种用例 | **问题**：标准 ring attention 用 P2P 通信，在 HPC（Frontier 超算）网络上不能充分利用集合通信优化；causal 下负载不均。**方法**：① 双向 attention：partitioning 减少通信量约 1/N；② 集合通信代替 P2P；③ causal attention 比 stripe attention 更均衡。kernel 层最高 3.4×，端到端约 1.5× 加速 | ✅ **开源** [jqyin/ringX-attention](https://github.com/jqyin/ringX-attention)；ORNL 团队 | 超算的Ring Attn |
| **[NeurIPS'25 Spotlight] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning** | 把 top-p (nucleus) 采样思想引入稀疏 attention，自适应预算决策 | **问题**：现有稀疏 attention（Quest、SnapKV、H2O 等）用固定 top-k 预算，不能跨层/头/输入自适应；over/under selection 都浪费精度或资源。**方法**：① Select-then-Prune 层次结构（top-k 选 + top-p 剪枝）；② 4-bit quantized SpGEMV 估计 attention 权重（INT4 平衡精度/效率）。最高 98% token 修剪、self-attn 15.4× 加速、端到端 token 延迟 3.9× | ✅ **开源** [tsinghua-ideal/Twilight](https://github.com/tsinghua-ideal/Twilight)；基于 FlashInfer/Quest | top-k选择+top-p剪枝的pipeline |
| **[NeurIPS'25 Spotlight] SageAttention3: Microscaling FP4 Attention + 8-Bit Training** | 首个 FP4 微缩 attention 推理（Blackwell GPU）+ 首个可训练低位 attention | **问题**：以往低位 attention（FA-3/SageAttention）只优化推理；FP4 量化需解决数值范围和精度。**方法**：① 利用 RTX5090 FP4 Tensor Core，two-level quantization；② 8-bit forward+backward (SageBwd) ——保留最敏感 matmul 用 FP16。RTX5090 上 1038 TOPS，比最快 FlashAttention 快 5×；fine-tuning 8-bit 无损 | ✅ **开源** [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)（含 v1/v2/v2++/v3） | fp4的attn推理，int8训练，pipeline优化 |
| **[arxiv'25] SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention** | DiT 模型中将 attention 拆为 sparse + linear 两路（critical/marginal/negligible 三类） | **问题**：视频 DiT (10K-100K seq) 中 attention 是瓶颈；纯稀疏方法 DiT 上很难超 90% 稀疏，纯线性方法对视频生成质量崩溃。**方法**：① 关键观察：DiT attention 权重可分为高秩+小部分 + 低秩+大部分；② critical → O(N²) FlashAttention，marginal → O(N) linear，negligible → 跳过；③ 单 GPU kernel 融合三种，前向后向都支持。微调几步即可 95% 稀疏，13.7× attention 加速 | ✅ **开源** [thu-ml/SLA](https://github.com/thu-ml/SLA)（含 SLA 与 SLA2 续作） | DiT的pipeline |
| **[MLSys'25] FastTree: Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference** | 针对 radix tree KV cache 的 GPU attention kernel + 树结构感知运行时 | **问题**：现有 LLM serving (SGLang) 用 radix tree 组织 KV，但 attention 计算还是按常规模式——重复 memory load、tensor core 利用率低。**方法**：① 多阶段 tile 选择 + Flash-Attn 风格 group processing；② 贪心启发式划分树最小化 overhead；③ 长 context 拆分缓解 tail effect。SGLang 吞吐提升最高 2.2× | ✅ **开源** [PanZaifeng/FastTree-Artifact](https://github.com/PanZaifeng/FastTree-Artifact)；基于 SGLang | Tree结构组织的KVCache访问优化 |
| **[MLSys'25] FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving** | LLM 推理通用 attention 引擎；block-sparse + composable storage + JIT 模板 | **问题**：LLM serving 中 KV-cache 存储异质（PagedAttn、Tree、不同模型）；不同设置都要新 kernel；CUDAGraph 与动态调度冲突。**方法**：① block-sparse + composable 格式；② 可定制 attention template + JIT 编译；③ 负载均衡调度兼容 CUDAGraph。已被 SGLang/vLLM/MLC-Engine 集成 | ✅ **开源**且**广泛应用** [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)；MLSys'25 | KV Cache的管理优化 |
| **[NeurIPS'24] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** | Hopper H100 优化的 fused attention；异步执行 + FP8 | **问题**：FA-2 在 H100 上仅 35% 理论 FLOPS（GEMM 80-90%）；用同步 mma.sync 而非 Hopper 的 WGMMA；data load/compute 串行。**方法**：① WGMMA 异步指令（吞吐 +50%）；② TMA 实现 producer-consumer overlap；③ 支持 FP8。Hopper 实测 740 TFLOPS（FP16）/1.2 PFLOPS（FP8） | ✅ **开源** [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | 
FA2在H100的高效实现 |
| **[ICLR'24] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | 减少 non-matmul FLOPs + 沿 seq 维并行 + warp 间工作再分配 | **问题**：FA-1 在 GPU 上仅 25-40% 理论吞吐——work partitioning 次优、warp 间共享内存读写过多、单 head 串行。**方法**：① 算法层减少 non-matmul；② 增加 sequence 维并行提高 occupancy；③ 块内 warp 间工作再分配减 SMEM 通信。比 FA-1 快 2×，A100 达 50-73% peak FLOPS | ✅ **开源**（同上） | FA1在GPU的高效实现 |
| **[NeurIPS'22] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | IO-aware 精确 attention；tiling + recomputation 把内存复杂度从 O(N²) 降到 O(N) | **问题**：标准 attention 在 GPU 上把 N×N 大矩阵反复读写 HBM——内存带宽瓶颈。**方法**：① online softmax + tiling；② 反向用 recomputation 而非存储 attention 矩阵；③ 全部融合到一个 kernel。HBM 访问减少 9×，GPT-2 attention 7.6× 加速 | ✅ **开源**（基础库） |  |

## 第 4 批：GPU Kernel Optimization（共 22 篇）

| 论文标题 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源/环境 | 备注 |
| --- | --- | --- | --- | --- |
| **[ASPLOS'26] Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation** | 任意 1-8 bit 低精度的 tile-level DSL；首创 algebraic layout system | **问题**：现有低精度 kernel 编译器(Triton/Ladder)只支持 2 的幂位宽；高层 GPU 抽象限制 fine-grained register management 和优化访存。**方法**：① 子字节数据类型全谱支持(int/uint/float 1-8 bit)；② thread-block 级编程模型 + 层次化 memory 空间；③ algebraic layout system 让低精度 tile 可在寄存器中重新解释为 hardware-friendly 数据类型。vs Triton/Ladder/QuantLLM/Marlin 提速 1.75/2.61/1.29/1.03× | ✅ **开源** [NVIDIA/tilus](https://github.com/NVIDIA/tilus)（已 Hopper/Blackwell 支持） | GPU低精度编译框架 |
| **[EuroSys'26] FlashOverlap: Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering** | 基于信号机制的 tile-wise compute-communication overlap；通信 primitive agnostic | **问题**：分解法 overlap 效率次优；融合法 adaption 成本高；现有方案不能同时满足 tile-wise overlap、interference-free 计算、通信 primitive 无关。**方法**：① 计算 kernel 完成 wave 后发信号触发对应 tile 通信；② pre-communication reordering 写到连续地址，直接调 NCCL；③ 预测搜索选最优 wave group size。清华+InfiniAI 团队 | [https://github.com/infinigence/FlashOverlap](https://github.com/infinigence/FlashOverlap) | 多GPU？ |
| **[arxiv'25, OSDI 续作] Mirage Persistent Kernel (MPK): Mega-Kernelizing Tensor Programs** | 自动把 LLM 编译为单 megakernel；**SM-level graph** + 去中心化调度 | **问题**：每个算子启动一个 kernel 启动开销大；CUDA Graph 静态难以适应动态 control flow；现有 ML 编译器(PyTorch/Triton/JAX/TVM)不支持 mega-kernel。**方法**：① SM 粒度的依赖图捕获 cross-operator 软件流水；② MPK 编译器把 **tensor 程序降级为 SM-level task graph** + 优化 CUDA；③ in-kernel 并行 runtime 单 mega-kernel + 去中心化调度。端到端推理延迟降 1.7× | ✅ **开源** [mirage-project/mirage](https://github.com/mirage-project/mirage)；OSDI'25 收录 | mega-kernel？ |
| **[arxiv'25, Meta] KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators** | agentic 框架跨 NVIDIA/AMD/MTIA；多抽象层(Triton+CuTe DSL+底层语言) | **问题**：DLRM 训练推理面对 model 架构、kernel primitive、硬件代际三重异构挑战；现有静态编译器不能跨硬件扩展。**方法**：① 图搜索 + selection policy/universal operator/fitness function；② RAG-based prompt synthesis；③ 持久化知识库编码硬件约束。KernelBench 全 250 题 100% 通过；生产部署多代际 GPU；Llama-3.1-8B Vanilla Attn 4.6× 加速 | ❌ 暂未开源 (Meta 内部生产部署) | model、kernel、primitive、硬件后端的编译 |
|  |  |  |  |  |
| **[arxiv'25, AMD] Iris: First-Class Multi-GPU Programming in Triton** | 在 Triton 内置 SHMEM 风格 RMA；ROCm 团队首个多 GPU Triton 框架 | **问题**：传统 CCL(NCCL/RCCL) bulk-sync + CPU 启动开销；高性能多 GPU 写法走 HIP/CUDA，编程复杂度高。**方法**：① 通过 HIP IPC 建立跨 GPU 对称内存抽象；② SHMEM-like RMA APIs in Triton；③ AMD memory model 提供形式化正确性保证。GEMM+All-Scatter 比 PyTorch+RCCL 快 1.79× | ✅ **开源** [ROCm/iris](https://github.com/ROCm/iris) | 多GPU编程框架 |
| **[MLSys'26, Stanford] AccelOpt: Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization** | 自我改进的 LLM agent 自动优化新 AI 加速器 kernel；优化记忆库 | **问题**：新硬件(AWS Trainium)缺乏专家优化经验；GPU 已有但仍是难点(NVIDIA 发布 H100 后社区花数月才掌握)。**方法**：① 迭代生成 + 优化记忆库(curate slow-fast kernel pair)；② planner+executor agent 配 beam search；③ NKIBench Trainium 基准。Trainium1 49→61%、Trainium2 45→59% peak 吞吐；用 open-source 模型匹配 Claude Sonnet 4 但便宜 26× | ✅ **开源** [zhang677/AccelOpt](https://github.com/zhang677/AccelOpt) | 自动优化的编译 |
| **[arxiv'25, Stanford] ParallelKittens: Multi-GPU AI Kernels** | 把 ThunderKittens 的 tile 抽象扩展到多 GPU；8 个核心 primitives + LCSC 模板 | **问题**：模型规模超出 inter-GPU 互联带宽；现有 overlap 系统不能跨 workload 达 peak；AlltoAll/AllReduce 受 straggler 拖累。**方法**：分析三大原则——transfer mechanism (copy engine vs TMA vs register)、scheduling (host-device/inter-SM/intra-SM overlap)、design overheads；single-source kernels 单文件少改动达 hand-tuned 水平。最高 2.33× speedup over 强 baseline。已被 Cursor 采纳生产 | ✅ **开源** [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | 多GPU优化？ |
| **[arxiv'25, Stanford] HipKittens: Fast and Furious AMD Kernels** | 首个 AMD GPU 高性能 kernel 编程框架；C++ embedded primitives | **问题**：AMD GPU 提供 SOTA 算力但软件不成熟，巅峰 kernel 还要写 raw assembly；现有 AITER/PyTorch SDPA 在 MI355X 上 GQA backwards 仅 30%/24% SOTA。**方法**：① 8-wave ping-pong 调度跨 compute 和 memory；② 程序员可控寄存器分配；③ chiplet-aware swizzling。CDNA3/CDNA4 上对手写汇编 AITER 持平/超越，对编译器 1.2-2.4× 优势(d=64 attention/GQA backwards/memory-bound) | ✅ **开源** [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens) | AMD GPU的软件生态 |
| **[NeurIPS'25, Cornell] FlashMoE: Distributed MoE in a Single Kernel** | 完全融合 distributed MoE 为单一 persistent GPU kernel；actor-style 并发 | **问题**：分布式 MoE 实现因 CPU 调度+host-initiated 通信+频繁 kernel launch 导致低利用率；AlltoAll 占 68% 总时间。**方法**：① fused dispatch/compute/combine 流水；② 一侧、device-initiated RDMA 通信替代 bulk-sync collectives；③ tile 粒度 task descriptor + Subscriber/Scheduler/Processor actor。8-H100 上 9× 利用率、6× 延迟、5.7× 吞吐(用 FP32 baseline 用 FP16) | ✅ **开源** [osayamenja/FlashMoE](https://github.com/osayamenja/FlashMoE) | 分布式MoE |
| **[arxiv'25, SC'25, SJTU+ByteDance] LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel** | 极简硬件高效 W4A8 dequantization；LiquidQuant + 隐式细粒度流水 | **问题**：W4A8 GEMM 现有实现 dequantization 在 CUDA cores 跑慢；Tensor Cores 高吞吐被卡。**方法**：① LiquidQuant 仅 2 条算术指令处理 4 元素；② 隐式细粒度流水跨 warp group 全 overlap weight load/dequant/MMA，无软件同步；③ TMA+CUDA Cores+Tensor Cores 异构流水。比 SOTA W4A8 kernel 快 2.90×，端到端 4.94×；vs TRT-LLM 各 quant kernel 1.12-1.63×。LiquidServe 系统级 1.63× 加速 | ❌ 暂未开源（SC'25 published） |  |
| **[arxiv'25/HPCA'26] TileLang: Composable Tiled Programming Model for AI Systems** | 解耦 dataflow 与 scheduling；composable tile 操作器 + 注解原语 | **问题**：现有手工(FA-3/CUTLASS/ThunderKittens)难写；编译器(Triton)又表达力不足，关键路径无控制。**方法**：① 核心 tile 操作器 (GEMM/COPY/ATOMIC/REDUCE) 表达 dataflow；② thread binding/layout/tensorize/pipeline 等注解控制 scheduling；③ Pythonic 前端 + Hidet 后端。NVIDIA H100/MI300X 上 GEMM 平 vendor lib，比 Triton 快 1.13×/1.25×；FA 比 Torch 显著加速、达 95% AITER 性能 | ✅ **开源** [tile-ai/tilelang](https://github.com/tile-ai/tilelang)（含 NVIDIA/AMD/Apple Metal/Ascend 后端） |  |
|  |  |  |  |  |
| **[ICLR'25 Spotlight] ThunderKittens: Simple, Fast, Adorable AI Kernels** | 三层 GPU 抽象 (warp tile/block 异步/grid 隐藏 launch)；纯 C++ embedded | **问题**：手写 kernel 也达不到理论峰值；AI 架构和 GPU 硬件之间映射缺简洁抽象。**方法**：① 16×16 矩阵 tile 作 warp 级原始数据结构；② thread-block 异步操作模板；③ grid-level 隐藏 launch/teardown/memory。与 H100 厂商库匹配。被 NeurIPS, ICLR, MLSys, MICRO 引用作为基础抽象 | ✅ **开源** [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens) |  |
|  |  |  |  |  |
| **[MLSys'25] FlexAttention: Programming Model for Generating Fused Attention Variants** | PyTorch 中几行写任意 attention 变体 (Alibi/Document Mask/Paged etc.) | **问题**：FlashAttention 单体性导致研究者难试新 attention 变体；常规编译器难自动生成 fused attention。**方法**：① score_mod + mask_mod 抽象捕获大多数 attention 变体；② PyTorch 编译器栈 JIT 生成 fused kernel；③ 易组合避免变体组合爆炸。训练吞吐 2.4×，推理 2.04×；causal 任务上 1.00-1.22× FA-2 | ✅ **开源**（PyTorch torch.nn.attention.flex_attention 主分支已集成）；Meta MLSys'25 |  |
| **[arxiv'23/PPoPP'23] Stream-K: Work-centric Parallel Decomposition for GEMM** | work-centric 划分代替 tile-based；按 MAC 循环迭代均摊 | **问题**：Tile-based 划分在某些 problem 形状会让多数 SM 空转(wave quantization)；cuBLAS 用复杂启发式选 kernel。**方法**：① 把 GEMM 总 MAC 迭代均分给固定数 CTA；② 部分和需要时通过 global memory 聚合；③ 单 tile 大小配置就行。NVIDIA A100 上 14× 峰值/6.7× 平均加速 vs CUTLASS/cuBLAS。已被 CUTLASS/各 attention kernel(FlashInfer/LeanAttention)广泛集成 | ✅ **集成到 CUTLASS** |  |
| **[OSDI'23] Welder: Scheduling Deep Learning Memory Access via Tile-graph** | tile-graph 抽象做 holistic memory access 优化 | **问题**：DNN 越来越 memory-bound；现有方法把 DNN 当 compute-bound、memory 优化各自为政；未充分利用层次化 memory hierarchy。**方法**：① tile-graph 让 fine-grained tile 级 data 管理；② 跨 memory layer 优化独立性观察—把组合空间分解成独立子空间；③ tile traffic-based cost model。新增 89 种优化模式 | ✅ **开源** [nox-410/Welder](https://github.com/nox-410/Welder) |  |
| **[CGO'24] A Framework for Fine-Grained Synchronization of Dependent GPU Kernels (cuSync)** | tile 级而非 kernel 级同步，让依赖 kernel 部分并发 | **问题**：依赖 kernel 现行做法是顺序串行；GPU 利用率低，但完全并发又会数据竞争。**方法**：在 tile 间设置同步点，让生产者 tile 一旦完成就让消费者读取，部分并发执行依赖 kernel。MS Research+CMU+UMass | [https://github.com/microsoft/cusync](https://github.com/microsoft/cusync) |  |
| **[RTAS'24] Demystifying NVIDIA GPU Internals to Enable Reliable GPU Management** | 实验逆向 GPU 内部细节(scheduler、driver、command queue)以助实时性 | **问题**：NVIDIA GPU 内部黑盒，实时/嵌入式系统(自动驾驶等)难以保证可预测性。**方法**：实验探针，绘制 thread block 调度行为、driver 与 hardware 交互细节 | 论文级研究 (UNC) | 逆向 |
| **[SIGMETRICS'21] Demystifying the Placement Policies of NVIDIA GPU Thread Block Scheduler** | 微基准测量 thread block scheduler 行为 | **问题**：TB 在 SM 上的分布策略未公开，影响 GPU 工作负载性能预测。**方法**：精心设计的微基准揭示 round-robin/load-aware 等策略；提供调度行为模型 | 论文级研究 (Trinity College Dublin等) | 逆向 |
| **[NeurIPS'20] Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning** | host-side 静态调度，避免 PyTorch dynamic scheduling 开销 | **问题**：DL 推理 host CPU launch 开销显著，且 host 难有效并发分发独立 GPU 任务到多 stream。**方法**：静态分析 DAG 后预先生成 stream 分发计划；轻量并发 task scheduler。在 ResNet 等模型上比 PyTorch/TensorRT 快 22% | ✅ **开源** Nimble | GPU调度 |
| **[RTSS'17] GPU Scheduling on the NVIDIA TX2** | 嵌入式 NVIDIA Tegra TX2 的 GPU scheduling 机制实证 | **问题**：实时系统中 GPU scheduler 行为，特别是 TX2 在自动驾驶中的应用，未有详细文档。**方法**：系统的微基准实验描绘 TX2 scheduler 优先级/抢占等行为 | 论文级研究 (UNC) | 逆向 |

> 注：因前 3 批已覆盖的论文(FA-1/2/3、FlashInfer、FastTree、UltraAttn、RingX、SageAttention、SLA)在 GPU Kernel 章节再次出现时仅作引用，不重复表格条目。表中还纳入了若干历史经典(Stream-K、Welder、Nimble、TB Scheduler 系列)以保证开源可复现性的完整性。**HuntKTm/HyTiS/Kitsune/ACS/Characterizing Concurrency** 等最后一批未独立出现表中的论文若用户需要单独详细分析，可单独提示，我可补充。
> 

## Part I：LLM Long Context

### 1. 序列并行 / 上下文并行（Sequence/Context Parallelism）

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
| **[arxiv'25] Optimizing Long-context LLM Serving via Fine-grained Sequence Parallelism** | 推理阶段细粒度 SP，按 token 级动态分配，避免长短请求间资源浪费 | **问题：** 静态 SP 在长短混合 serving 下出现 head-of-line blocking 与资源碎片。**方法：** 基于细粒度 token-level partitioning 与动态调度，针对 prefill/decoding 两阶段分别优化。 | 多基于 vLLM/SGLang 二次开发 |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
| **[SOSP'25] DCP: Dynamic Context Parallelism** | **动态上下文并行**：每 batch 重新规划 partition；用 hypergraph partitioning 建模数据/计算块的设备分配 | **问题：** 现有 CP（Ring/Ulysses/USP/LoongTrain）使用静态配置，忽略 batch 内序列长度与 attention mask 的动态变化，造成冗余通信与负载不均。**方法：** ① 数据/计算双向 fine-grained 分块；② Hypergraph partitioning 求解最优块映射；③ 动态切换并行配置。在 causal mask 下 attention 加速 1.19–2.45×，sparse mask 下 2.15–3.77×。 | [GitHub: chenyu-jiang/dcp](https://github.com/chenyu-jiang/dcp) + Docker 镜像 + Megatron-LM 集成；artifact-evaluated |
| **[arxiv'25] Data-Centric Elastic Pipeline Parallelism** | Elastic PP（EPP）：编排 token-level 与 batch-level PP，根据资源/负载自适应切换粒度 | **问题：** Batch-level PP 长上下文场景下激活显存大；token-level PP 切片小但硬件利用率低。真实数据集长尾分布加剧负载不均。**方法：** ① 自适应混合两种 PP 粒度；② 数据感知调度处理长尾分布。 | arXiv 2509.21275 |
|  |  |  |  |
| **[ASPLOS'25] FlexSP** | 异构 SP 组：根据序列长度差异分配不同 SP degree，混合长短序列同步训练 | **问题：** 现有 SP 假设序列等长，使用单一静态 scattering；真实语料长度长尾分布导致短序列被过度并行（通信浪费）。**方法：** ① 将 SP 组配置建模为 MILP（SCIP 求解）；② Sequence bucketing 减少决策变量；③ 时间均衡的序列分配。比 SOTA 训练框架快 1.98×。 | [GitHub: PKU-DAIR/Hetu-Galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron)；PyTorch + flash-attn + NCCL |
| **[arxiv'24] USP: Unified Sequence Parallelism** | 统一 DeepSpeed-Ulysses（all-to-all/head-split）和 **Ring-Attention**（P2P/seq-split）为 2D 混合 SP | **问题：** Ulysses 受 head 数限制（不适合 GQA/MQA 单 head），Ring 跨节点 P2P 通信慢。**方法：** Ulysses degree × Ring degree 形成混合并行，支持因果 mask 的 load balancing；与 4D 并行兼容。LLaMA3-8B 208K 序列 47% MFU。 | [GitHub: feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention)；已被 NVIDIA TransformerEngine 集成 |
|  |  |  |  |
| **[SOSP'24] LoongServe** | 推理阶段提出 Elastic Sequence Parallelism (ESP)：实时调整并行度 | **问题：** **Long context serving** 中 prefill/decode 资源需求差异巨大，静态 SP 浪费 KV cache 内存。**方法：** ① Prefill 主动 scale-down 复用通信；② Multi-master 解码避免 KV 迁移；③ Token 粒度管理消除 KV 碎片。比 chunked prefill 高 3.85×、比 P/D 解耦高 5.81× 吞吐。 | [GitHub: LoongServe](https://github.com/LoongServe/LoongServe)；基于 vLLM 风格 |
|  |  |  |  |
| **[PODC'25] DeepSpeed-Ulysses** | 序列维度 partition + 高效 all-to-all 通信，attention 时通过 head-split 计算 | **问题：** Megatron-SP/ColAI-SP 通信开销随**序列长度增长**；不通用。**方法：** all-to-all 把 N/P 序列 → P/H heads，通信量 O(N/P)，序列与 GPU 同比例扩展时通信常量。比 SOTA 快 2.5×、4× 长度。 | [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) tutorial; 已商业化部署 |

---

### 2. 流水线并行（Pipeline Parallelism）

超长上下文的训练和推理优化。

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

---

### 3. 稀疏注意力（Sparse Attention / Block Sparse / Adaptive Pruning）

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
| **[arxiv'25] XAttention (ICML'25)** | **反对角线（antidiagonal）求和**作为 block 重要性 proxy，免训练 plug-and-play 块稀疏 | **问题：** 已有 block-sparse 方法测 block 重要性代价高（需 token pooling 等），且精度难以保持。**方法：** 利用 antidiagonal 求和恰好与垂直/斜杠 attention pattern 相交，可低成本判定 block；用预测最小阈值进行剪枝。RULER/LongBench/VideoMME/VBench 上保持精度，attention 计算加速 13.5×。 | [GitHub: mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention) |
| **[arxiv'25] LServe (MLSys'25)** | 统一**静态**（streaming heads）+**动态**（query-aware page selection）稀疏到单一 GPU kernel | **问题：** Prefill attention 二次复杂度、decode KV cache 内存大。已有方法（H2O/StreamingLLM/Quest 等）只优化单个阶段或与 PagedAttention 不兼容。**方法：** ① 一半 head 转为 streaming heads（几乎免费）；② 层次化 page selector 选择固定数量重要 KV pages；③ 与 KV 量化融合。Prefill 加速 2.9×、Decode 1.3–2.1× over vLLM。 | [GitHub: mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve)（与 QServe 共享） |
| **[arxiv'25] MoBA: Mixture of Block Attention** | 将 MoE 思想引入 attention：query token 通过 gating 路由到 top-k KV blocks | **问题：** 已有方法要么强先验（sink/window）只适合特定任务；要么用线性近似在复杂推理任务上效果不明。**方法：** ① 上下文切块；② 动态 gating 路由 query 到最相关 KV block；③ 可在 full attention 与 sparse attention 间无缝切换。已部署于 **Kimi 长上下文推理**。 | [GitHub: MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) |
| **[arxiv'25] SALE: Low-bit Estimation for Sparse Attention** | 低位（int4/int8）量化估计注意力分数，做 prefill 阶段稀疏 | **问题：** 长上下文 prefill attention 是主要瓶颈；现有稀疏方法 score 计算仍用 fp16。**方法：** **量化估算 score** → 选 top-k → 全精度计算。降低 score 阶段算力。 | arXiv |
| **[arxiv'25] Tactic: Adaptive Sparse Attention** | **累计注意力分数（cumulative score）阈值自适应**预算（非固定 token budget）+ K-means 聚类排序 + 分布拟合 | **问题：** 固定 budget 忽略了 head/layer/context 间的稀疏度差异，造成精度损失或浪费。**方法：** ① 设定目标 cumulative score（如 95%）；② Prefill K-means 聚类 keys；③ Decode 用 centroids 与 query 排序；④ 用幂律分布拟合 token 数估算。Decode attention 加速 7.29×、E2E 1.58×。 | arXiv 2502.12216 |
| **[arxiv'25] Twilight: Hierarchical Top-p Pruning** | 层次化 top-p 剪枝：自适应 sparsity，每层每头不同 | **问题：** Top-k 在不同层/头分布差异大，全局**固定 k 损失精度**。**方法：** Hierarchical top-p coverage threshold，逐层逐头自适应。 | arXiv |
|  |  |  |  |
| **[arxiv'25] ParallelComp: Length Extrapolation** | 双阶段 chunk-wise 处理 + intra-chunk KV eviction：从 8K 外推到 128K（无再训练） | **问题：** Attention sink/recency/middle bias 阻碍长度外推。**方法：** ① Local attention 内 chunk eviction；② "self-information" 评分 chunk-wise global attention。8B 模型 prefill 加速 23.5×。 | arXiv |
|  |  |  |  |

---

### 4. KV Cache 优化（Caching / Compression / Quantization）

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
| **[ACL'25] MiniKV: 2-bit KV Cache** | **2-bit 量化 + adaptive token eviction（pyramid KV）协同** + FlashAttention 兼容 kernel | **问题：** 单纯 2-bit 量化已饱和，进一步压缩失效；评分式 eviction 与 FlashAttention 不兼容。**方法：** ① Pyramid KV 跨层 token selection；② Sub-channel Key + per-token Value 量化降低误差；③ Triton two-pass selective FlashAttention kernel。86% KV 压缩，准确率保留 98.5%。 | [SSAIL Lab Project](https://supercomputing-system-ai-lab.github.io/projects/minikv/) |
| **[arxiv'24] ShadowKV (ICML'25 Spotlight)** | **Pre-RoPE keys 低秩存 GPU + Values offload CPU**，decode 时基于 landmarks 重构稀疏 KV | **问题：** 现有动态稀疏 attention 要么不降 GPU 显存，要么 CPU offload 引入解码延迟。**方法：** ① 发现 pre-RoPE keys 极低秩；② Landmarks + outliers 留 GPU；③ Cache policy 利用时间局部性降低 60% 解码 overhead。A100 上 6× batch、3.04× 吞吐。 | [GitHub: ByteDance-Seed/ShadowKV](https://github.com/ByteDance-Seed/ShadowKV) |
|  |  |  |  |
|  |  |  |  |
| **[COLM'24] TriForce: Hierarchical Speculative Decoding** | 层次化推测解码：retrieval-based draft model + lossless 加速 | **问题：** 长序列生成时 KV cache 与权重内存压力使解码慢。**方法：** ① 部分 KV cache + 小模型 draft；② Hierarchical speculation。无损加速。 | [GitHub: Infini-AI-Lab/TriForce](https://github.com/Infini-AI-Lab/TriForce) |
|  |  |  |  |
| **[ICLR'24] StreamingLLM: Attention Sinks** | 发现 attention sink 现象，保留最初几个 token + 滑动窗口实现"无限"流式生成 | **问题：** 朴素滑动窗口删 KV 会让 LM 性能崩溃。**方法：** 保留 sink tokens + sliding window，无需重训。 | [GitHub: mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm)（[Code]） |
|  |  |  |  |

---

### 5. 内存管理 / 其他系统优化

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
| **[arxiv'25] FocusLLM: Parallel Decoding** | 长序列分 chunk + 局部上下文附加 + **并行 candidate decoding** | **问题：** 已有 context condensing 损失信息；memory token 方法外推性差。**方法：** ① 长文本切 chunk，每 chunk 附加 local context + memory token；② 并行 decoding 生成 candidate token；③ Aggregate 回 local context。8K 训练即可处理 400K 序列。 | [GitHub: leezythu/FocusLLM](https://github.com/leezythu/FocusLLM) |
|  |  |  |  |
| **[NeurIPS'24 Workshop] Long-Context RAG Performance** | 长上下文 LLM 在 RAG 任务中的系统性评测 | **问题：** RAG vs 长上下文 LLM 缺少对照评测。**方法：** Benchmark 套件覆盖检索精度 × 上下文长度。 | benchmark |

---

## Part II：Systems for Agentic AI

Top-K库：[https://github.com/tsinghua-ideal/flash-topk-attention](https://github.com/tsinghua-ideal/flash-topk-attention)

[https://github.com/rapidsai/raft](https://github.com/rapidsai/raft)

tileLang、Triton、SgLang

各种稀疏Attn：[https://attention-survey.github.io/](https://attention-survey.github.io/)

### 1. Agent Serving / Scheduling 引擎

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
| **[arxiv'26 / arxiv'25] Continuum** | **KV Cache TTL** 机制：根据预测的 tool call duration 选择性 pin KV，配合 program-level FCFS | **问题：** 现有 inference 引擎在请求结束即驱逐 KV，**agentic 场景**多轮交互被打断 → 重复 prefill 或排队。**方法：** ① 对 tool call 请求估计 TTL（基于重载 cost vs 排队收益）；② TTL 内 pin KV，过期自动驱逐；③ Program-level FCFS 防 scheduling bubble。SWE-Bench/BFCL 上 Llama-3.1 8B/70B 显著降低 JCT，提升随轮次增加。 | [GitHub: Hanchenli/vllm-continuum](https://github.com/Hanchenli/vllm-continuum)（vLLM 上模块化扩展） |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
| **[arxiv'26] Sutradhara** | Orchestrator-Engine co-design for tool-based agentic inference | **问题：** **Orchestrator**（高层规划）与 inference engine（KV/调度）解耦优化效果有限。**方法：** 协同设计：orchestrator 提示 engine 关于工具调用 pattern。 | arXiv |
|  |  |  |  |
|  |  |  |  |
| **[arxiv'25] Agent.xpu** | 异构 SoC（NPU + iGPU + CPU）上的 agent 调度 | **问题：** 个人 LLM agent 在端侧 SoC 上需混合 reactive（前台）与 proactive（后台）流，现有引擎假设静态 single-shot 推理。**方法：** ① Heterogeneous Execution Graph（NPU/iGPU 亲和 + 弹性绑定）；② Flow-aware 协调（prefill/decode 解耦减少带宽争用）；③ Fine-grained 抢占 + slack-aware piggybacking。 | arXiv 2506.24045 |
| **[arxiv'25] Aragog** | Just-in-Time 模型路由：根据 agent workflow 动态选模型 | **问题：** 不同 **agent 步骤复杂度差异大**，统一用大模型成本高。**方法：** JIT 路由：复杂步骤用大模型，简单步骤用小模型。 | arXiv |
| **[arxiv'25] ToolOrchestra** | 模型 + 工具高效编排 | **问题：** **Agent 需调用大量工具**，编排策略影响精度与成本。**方法：** 学习式编排策略。 | [https://github.com/NVlabs/ToolOrchestra](https://github.com/NVlabs/ToolOrchestra) |
| **[arxiv'25] AI Metropolis (MLSys'25)** | **多 agent** 模拟的**乱序执行（OoO）**调度 | **问题：** 全局同步并行模拟（GenAgent）受 false dependencies 限制，并行度低。**方法：** ① 动态依赖图追踪真实 agent 间依赖；② OoO 调度最大化并行；③ 类似 OpenAI Gym 接口。比 sync parallel 加速 1.3–4.15×。 | 计划开源（Stanford MAST Lab） |
| **[ICML'24] AnyTool** | Self-reflective hierarchical agents：处理大规模 API 调用 | **问题：** **大规模 API** 选择是 agent 主要瓶颈。**方法：** Hierarchical 层级搜索 + 自反思。 | ICML 2024 |

### 2. Agent 强化学习 / 训练

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

### 3. Agent 工具调用 / 推理优化

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
| **[arxiv'26] ToolCaching** | LLM 工具调用缓存机制 | **问题：** 重复工具调用浪费算力；同一工具不同参数有相似结果。**方法：** 学习式工具调用缓存（基于参数相似度）。 | arXiv 2026 |
| **[arxiv'26] XGrammar 2** | 动态高效的结构化生成引擎（agentic LLM 工具调用） | **问题：** Agent 频繁调用结构化输出（JSON/工具参数），现有 grammar engine 静态低效。**方法：** 动态语法编译 + agent 友好优化。 | arXiv |
|  |  |  |  |
|  |  |  |  |
| **[ICML'25] BFCL: Berkeley Function Calling Leaderboard** | 工具调用 / agent 评测基准 | **问题：** 缺少标准化工具调用 benchmark。**方法：** 多维度任务 + 公开 leaderboard。 | [BFCL leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) |
|  |  |  |  |

### 4. Agent 系统架构 / 观测 / 部署

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
| **[SAA'25] Toward Systems Foundations for Agentic Exploration** | 探索性 agent 任务的系统基础 | **问题：** 探索式 agent（如 OpenManus）**系统级原语**缺失。**方法：** 提出 exploration 系统抽象。 | SAA 2025 |
|  |  |  |  |
| **[SAA'25] Cortex** | Workflow-aware 资源池 + 调度 | **问题：** Agentic serving **多 workflow** 共享资源，资源池粒度粗。**方法：** Workflow-aware pooling，避免相互干扰。 | SAA 2025 |
| **[SAA'25] Tetris** | Predictive KV cache offloading（agentic + reasoning workload） | **问题：** Reasoning + agent workload 中 KV 重用难以预测。**方法：** 学习式 offload 预测。 | SAA 2025 |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

### 5. Agent 应用 / Domain-Specific

| 论文 | 核心创新点 | 相对 Baseline 解决的问题与设计方法 | 开源 / 环境 |
| --- | --- | --- | --- |
|  |  |  |  |
| **[arxiv'25] Nemotron 3 Nano** | **开源 MoE 混合** Mamba-Transformer agentic reasoning 模型 | **问题：** Agentic reasoning 需要快速吞吐 + 长上下文，纯 Transformer 在二者上有 trade-off。**方法：** ① **Mamba SSM + Transformer 混合架构**；② MoE 路由。 | NVIDIA 开源 |
| **[arxiv'25] Flash-Searcher** | DAG-based 并行 web agent 执行 | **问题：** Web agent 顺序执行慢；依赖 sequential 推理。**方法：** 把 agent workflow 表为 DAG，可并行节点并行执行。 | [https://github.com/OPPO-PersonalAI/Flash-Searcher](https://github.com/OPPO-PersonalAI/Flash-Searcher) |
| **[arxiv'25] MobiAgent** | **移动设备 agent** 可定制框架 | **问题：** Mobile agent 需根据设备/用户定制。**方法：** 模块化 + 配置驱动。 | https://github.com/IPADS-SAI/MobiAgent |
|  |  |  |  |
|  |  |  |  |
| **[arxiv'25] Efficient and Scalable Agentic AI with Heterogeneous Systems** | 异构系统下 agentic AI 高效执行 | **问题：** **异构 GPU/NPU/CPU 集群难统一调度 agent**。**方法：** 异构感知调度。 | arXiv |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

---

### 总结与说明

1. **覆盖范围：** 全部约 75+ 篇均给出四维度分析。LLM Long Context 35 篇 + Systems for Agentic AI 40+ 篇。
2. **未单独深入搜索的论文：** 部分较新或较冷门的 arXiv'26 论文（VisGym、Nalar、Sutradhara 等）以摘要标题/作者公开信息为主推断核心思想。具体细节请以原文为准。
3. **常见 Baseline / 开源依赖：**
    - 训练侧：Megatron-LM、DeepSpeed、PyTorch FSDP、TransformerEngine
    - 推理侧：vLLM、SGLang、TensorRT-LLM、LMCache
    - Attention：FlashAttention v2/v3、FlashInfer、Triton
    - Agent：LangChain、LangGraph、ReAct loop
    - 集群：NCCL/RCCL、Slingshot/InfiniBand
4. **关键趋势：**
    - **CP/SP**：从静态 → 动态（DCP、ByteScale、FlexSP），从单维度 → 异构混合（USP、StarTrail）
    - **Sparse Attention**：从固定 budget → adaptive top-p（Tactic、Twilight、XAttention）
    - **KV Cache**：从单层量化 → 多级 hierarchical（Strata、ShadowKV）
    - **Agent serving**：从 per-call → program-level（Autellix/Agentix、Continuum、Astraea）
    - **Verification**：从 always-verify → selective + speculative（Sherlock）