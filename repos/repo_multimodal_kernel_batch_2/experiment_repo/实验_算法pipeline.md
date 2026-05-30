# 实验_算法pipeline

## VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是：(1) 一个包含17个视觉交互环境的gymnasium框架，用于评估和训练VLM在多步视觉决策中的表现；(2) 多步求解器（multi-step solvers），通过BFS/DFS/图搜索/状态机等策略为每个任务生成结构化demonstration轨迹，用于监督微调（SFT）；(3) 训练方法：基于solver生成的demonstration进行全参数SFT，使用LlamaFactory框架。

  实验比较：(1) 12个前沿VLM模型在17个环境的easy/hard设置下的零样本成功率对比（Sec 3）；(2) 诊断实验——历史长度（full/4/2/1 turns）、文本vs图像观察表示、有无文本反馈、有无目标观察的对照实验（Sec 4）；(3) 训练实验——单任务SFT vs 多任务SFT的效果、easy→hard泛化能力对比（Qwen2.5-VL-7B vs Qwen3-VL-8B）、vision encoder vs LLM的模块贡献分析（Sec 5.3）、信息揭示型demonstration vs 标准demonstration的数据筛选策略对比（Sec 5.4）。

- 硬件平台是什么，配置是什么。
  论文未明确说明评估和训练的硬件平台。评估中proprietary模型通过OpenRouter API调用（云端推理），open-weight模型和训练使用的GPU未显式列出。训练配置：Qwen2.5-VL-7B-Instruct全参数微调，global batch size=64，learning rate=1×10^{-5}，bf16精度，单任务训练1500步，多任务训练5000步，使用LlamaFactory进行数据处理和训练编排。

- 模型是什么。数据集和bench分别是什么。
  评估模型（12个）：proprietary — Gemini 3 Pro、Gemini 2.5 Pro、GPT-5、Claude Sonnet 4、Grok 4 Fast、Qwen-VL-Max；open-weight — Qwen3-VL-235B-Instruct、GLM-4.5V、Llama-4-Maverick、Qwen-2.5-VL-72B-Instruct、Gemma 3-27B-Instruct；专用模型 — UI-Tars-1.5-7B。训练基座模型：Qwen2.5-VL-7B-Instruct。
  
  17个环境覆盖：Colorization（LLaVA数据集）、Counting（LVIS数据集）、Jigsaw（LLaVA）、Matchstick Equation（合成）、Matchstick Rotation（合成）、Maze 2D（合成/Maze-World）、Maze 3D（合成/Maze-World）、Mental Rotation 2D（LLaVA）、Mental Rotation 3D Cube（合成）、Mental Rotation 3D Objaverse（Objaverse数据集）、MuJoCo Fetch Pick-and-Place（MuJoCo）、MuJoCo Fetch Reach（MuJoCo）、Patch Reassembly（合成）、Referring Dot-Pointing（RefCOCO数据集）、Sliding Block（合成）、Video Unshuffle（SS2 Something-Something数据集）、Zoom-In Puzzle（LLaVA数据集）。每个环境70 episodes/task/setting，easy设定最大20步，hard设定最大30步。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：GitHub [visgym/VisGym](https://github.com/visgym/visgym)，Apache-2.0 license；HuggingFace模型 [VisGym/visgym_model](https://huggingface.co/VisGym/visgym_model)；项目页面 [visgym.github.io](https://visgym.github.io/)。

  算法pipeline核心流程：
  1. **环境交互循环**：每步t，模型接收完整历史 H_t = (I, {(o_τ, a_τ, f_τ)}_{τ<t})，其中I为任务指令，o_τ为观察图像（H×W×C），a_τ为已执行动作，f_τ为环境文本反馈。模型输出动作a_t。
  2. **Function-Conditioned Action Space**：动作表示为函数调用格式，如 ('swap', ((0,0),(0,1))), ('rotate', (30.5, 20.4, 15.1))，而非离散/连续动作向量。
  3. **Step函数**（伪代码）：
     ```
     function Step(a):
         ρ ← 0; (τ, υ) ← (false, false)
         Parse a → (α, π)
         if invalid format: return (obs(), 0, τ, υ, "invalid format")
         if α ∈ A and π ∈ A[α]: (φ, τ, υ) ← Apply(α, π)
         else: return (obs(), 0, τ, υ, "invalid action")
         if τ = true: ρ ← ComputeReward()
         return (obs(), ρ, τ, υ, φ)
     ```
  4. **Solver生成Demonstration**：每个任务配备启发式solver（Maze用图搜索找最优路径，Sliding Block用BFS，Matchstick Equation用BFS/DFS，Jigsaw用贪心swap或直接reorder排列，Fetch用状态机oracle）。Solver支持多策略和可选随机性，生成多样化轨迹。轨迹预处理过滤掉失败轨迹和与测试集重叠的初始状态。
  5. **SFT训练**：Qwen2.5-VL-7B-Instruct全参数微调，LlamaFactory编排，solver demonstrations仅来自easy难度——hard表现衡量难度泛化。

  **近似层次匹配说明**：本论文是VLM多步视觉决策评估与训练框架，不完全匹配传统算法pipeline（稀疏/量化/蒸馏加速推理），但因其包含训练方法（SFT with solver demonstrations），按最接近层次分类到算法pipeline。

## Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Tilus生成的任意位宽（1-8 bit）低精度矩阵乘法kernel，支持三种数据类型族：有符号整数（int2-int8）、无符号整数（uint1-uint8）、浮点数（float3-float8，含任意exponent/mantissa分布如e4m3、e3m3、e3m2、e2m2、e2m1、e1m1）。低精度计算流程：权重在kernel启动前做global memory layout变换（如i6[K,N] → u8[K/BK, N/BN, BK*BN*6/8]）实现连续内存访问；kernel内通过LoadGlobal加载为标准u8类型，View指令零开销将register tensor reinterpret为低精度类型和layout，Cast指令使用PRMT/LOP3/bitwise指令做vectorized casting到float16；最后Dot指令调用Tensor Core mma.m16n8k16完成矩阵乘累加。

  实验比较：vs cuBLAS FP16 kernel（标准精度baseline），覆盖uint1-uint8、int2-int8、float3-float8共21种低精度数据类型，矩阵乘法维度BS=16, K=8192, N=57344（来自Llama-3.3-70B的典型matmul）。同时与Triton、Ladder、QuantLLM、Marlin在uint8、f6e3m2、int4、uint4、uint2、uint1上比较speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA L40S GPU (48 GiB)，GPU driver 565.57.01，CUDA 12.6.3。跨硬件验证：NVIDIA A100 (Ampere)、H100 (Hopper)。

- 模型是什么。数据集和bench分别是什么。
  三个LLM：Gemma-2-9B、QWen2.5-32B、Llama-3.3-70B-Instruct。使用dummy inputs和weights（系统性能不依赖具体输入内容权重内容），模型metadata从Hugging Face Hub自动获取。Benchmark使用CUDA Events测量latency，每次kernel执行50次取median，L2 cache每次执行前清除。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/NVIDIA/tilus，artifact: https://github.com/yaoyaoding/tilus-artifacts

  低精度FP16×INT6矩阵乘法pipeline（图2）：
  ```
  # 预处理（kernel启动前执行transform_b kernel）
  # 将权重tensor从 i6[K, N] 变换为 u8[K/BK, N/BN, ceil(BK*BN*6/8)]
  # 每个tile [BK, BN] 内的 16*8*6=768 bits 以连续 u8 字节存储 (96 bytes)

  # Kernel内部 (Tilus VM program)
  C_accum = AllocateRegister(f32, [BM, BN])        # 累加器
  for k in range(0, K, BK):
      # Step 1: 加载activation tile (标准f16)
      a_tile = LoadGlobal(A_view, layout=m16n8k16_A, offset=[bi*BM:, k:])
      # Step 2: 加载weight tile (低精度→标准类型通过layout变换)
      b_tile = LoadGlobal(B_transformed_view, dtype=u8,
                          layout=local(3).spatial(32),
                          offset=[k/BK:, bj*BN:, 0:])  # u8加载，连续内存访问
      # Step 3: 零开销reinterpretation
      # 32 threads × 24 bits/thread → 原始为 4×i6, reinterpret为 3×u8
      b_tile = View(b_tile, dtype=i6,
                    layout=local(2,1).column_spatial(4,8).local(2,1))
      # Step 4: 向量化casting (i6 → f16)
      b_tile = Cast(b_tile, f16)  # PRMT + LOP3 + bitwise, 全在registers内完成
      # Step 5: Tensor Core矩阵乘累加
      C_accum = Dot(a_tile, b_tile, C_accum)  # mma.m16n8k16

  # 结果写回
  C_accum = Cast(C_accum, f16)
  StoreGlobal(C_accum, C_view, offset=[bi*BM:, bj*BN:])
  ```

  关键优化：weight loading pipeline（图1c）避免了Triton的shared memory layout conversion和Ladder的缺少pipelining问题。所有低精度类型（1-8bit）可在同一个参数化程序模板中通过改变tile size超参数支持，约200个配置per operator，编译时间~1分钟。

## SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - **SageAttention3 (FP4 inference)**：利用 Blackwell GPU 的 NVFP4 Tensor Core（E2M1 数据类型，1×16 量化块，E4M3 scale factor），对 attention 中的两个矩阵乘法 QK^T 和 PV 应用 FP4 microscaling quantization。同时提出 two-level quantization 解决 attention map P 的量化精度问题：先 per-token 归一化到 [0, 448×6]，再做 FP4 microscaling 量化。继承 SageAttention2 的 Smoothing Q 和 Smoothing K 技术。
  - **SageBwd (INT8 training)**：前向对 QK^T 和 PV 做 per-block INT8 量化，对 P 做 per-token INT8 量化（复用 online softmax 的 max 值避免额外 max 操作）。反向中保持 dOV^T 在 FP16（因其误差主导 dQ/dK 的梯度精度），其余四个 MatMul 做 INT8 per-block 量化。
  - 实验比较：与 FlashAttention2、xformers、SageAttention、SageAttention2 对比 kernel speed (TOPS) 和端到端推理/训练指标。

- 硬件平台是什么，配置是什么。
  - SageAttention3 推理：NVIDIA RTX5090 (Blackwell 架构，支持 FP4 Tensor Core)
  - SageBwd 训练：NVIDIA RTX4090
  - 理论对比包含 B300、B200、H100 的 TOPS

- 模型是什么。数据集和bench分别是什么。
  - **文生文模型**：Qwen2.5 (1.5B, 3B)、Llama3.2 (1B, 3B)；数据集：GSM8K、DROP、MMLU、HELLASWAG
  - **文生视频模型**：CogVideoX (2B)、HunyuanVideo、Mochi；数据集：Open-Sora prompt sets；metrics：CLIPSIM、CLIP-T、VQA-a、VQA-t、FScore
  - **文生图模型**：Flux、Stable-Diffusion3.5；数据集：COCO annotations；metrics：FID、sFID、CLIP、ImageReward
  - **预训练**：Llama 400M (hidden=1024, 20 layers, intermediate=3072, 16 heads)；数据集：FineWebEdu

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/thu-ml/SageAttention
  - FP4 算法流程（对应论文 Algorithm 1）：输入 FP16 Q, K, V ∈ R^{N×d}，分块 B_q, B_{kv}。对 Q_i 做 Smoothing（减去均值），然后 φ(Q_i - q̄_i) → (s_Q, Q̂_i) 量化到 NVFP4（1×16 块，E2M1+E4M3）。对 K_j^T 做 φ(K_j^T) → (s_K, K̂_j)，对 V_j 做 φ(V_j) → (s_V, V̂_j)。S_{ij} = FP4MM(Q̂_i, s_Q, K̂_j, s_K) + GEMV(q̄_i, K_j^T)。online softmax 计算 P̃_{ij} = exp(S_{ij} - m_{ij})。Two-level quantization: s_{P1} = rowmax(P̃_{ij})/(448×6)，P̃_{ij} = P̃_{ij}/s_{P1}，然后 s_{P2}, P̂_{ij} = φ(P̃_{ij})。O_{ij} = diag(e^{m_{i,j-1}-m_{ij}})O_{i,j-1} + FP4MM(P̂_{ij}, s_{P2}, V̂_j, s_V) × s_{P1}。最终 O_i = diag(l_{i,T_n})^{-1} O_{i,T_n}。

## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是MoDES，一个training-free的MoE MLLM动态expert skipping框架。包含三个核心算法组件：(1) GMLG（Globally-Modulated Local Gating）——将离线校准的全局逐层重要性因子α^{(l)}（通过KL divergence量化整层跳过对最终输出的影响）与局部routing概率π_i^{(l)}相乘得到expert重要性分数s_i^{(l)} = α^{(l)} · π_i^{(l)}；(2) DMT（Dual-Modality Thresholding）——为text token和vision token分别设置跳过阈值τ_t和τ_v，根据s_i^{(l)} < τ_t·I_t + τ_v·I_v判定该expert是否跳过；(3) Frontier Search——利用f(τ_t,τ_v)和g(τ_t,τ_v)的单调性在O(ND)时间内找到最优阈值对(τ_t, τ_v)，替代O(ND²)的naive exhaustive search，搜索时间从数天降至数小时。

  实验比较的算法baseline包括：NAEE（routing probability-based skipping，单层内阈值判定）、MC-MoE（attention-aware expert protection + skipping）、DiEP（differentiable expert pruning + adaptive skipping）、直接降低top-k的k值。所有baseline从LLMs场景适配到MLLMs的top-k（k>2）setting。

- 硬件平台是什么，配置是什么。
  8×H200 GPU用于calibration、search和accuracy evaluation；单张H200 GPU用于inference speed测量。Software: PyTorch transformers库，flash-attention2，lmm-eval评估框架。为inference speedup编写了自定义CUDA kernel实现MoE层内的双模态阈值判定和Group GEMM。

- 模型是什么。数据集和bench分别是什么。
  模型：3个MLLM系列——Kimi-VL-A3B-Instruct（64 experts/layer, k=6, 26 MoE layers）、Qwen3-VL-MoE-30B-A3B-Instruct（128 experts/layer, k=8）、InternVL-3.5-30B-A3B-HF（128 experts/layer, k=8）、InternVL-3.5-GPT-OSS-20B-A4B-Preview-HF（32 experts/layer, k=4）。数据集：GQA（1024 samples用于calibration和search）；8个image understanding benchmarks（TextVQA, ChartQA, MMStar, MMBench, MMVet, MME, RealWorldQA, COCO2017-Cap）+ 5个video understanding benchmarks（MVBench, EgoSchema, VideoMME, LongVideoBench, VideoM-MMU）。评估框架：lmm-eval，MMBench和MMVet使用DeepSeek-V3.1进行生成文本评分。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/ModelTC/MoDES
  
  MoDES算法pipeline（以Qwen3-VL-MoE-30B-A3B-Instruct的l-th MoE层，top-k=8个expert为例）：

  **离线阶段（Calibration + Search）：**
  ```
  # Step 1: Calibrate global importance α^{(l)} for each MoE layer
  C = randomly sample 1024 examples from GQA dataset  # calibration set
  for l in 1..L:  # for each MoE layer
      for each example c_j in C:
          prob_j = original_model(c_j)          # full model output distribution
          prob_j^{(l)} = model_with_layer_l_skipped(c_j)  # skip all experts in layer l
      α^{(l)} = (1/N) * sum_j D_KL(prob_j || prob_j^{(l)})
      # α^{(l)}大 → 浅层贡献大 → 跳过影响大 → 应少跳过
      # α^{(l)}小 → 深层贡献小 → 跳过影响小 → 可多跳过
  normalize: α̃^{(l)} = α^{(l)} / sum_{l'=1}^L α^{(l')}

  # Step 2: Frontier Search for optimal (τ_t, τ_v)
  B = {τ^{(1)}, τ^{(2)}, ..., τ^{(D)}}  # D=100 grid points in (0,1)
  target_skip_ratio = ρ  # e.g., 0.85 for 88% skipping
  for each (τ_t=q, τ_v=p) pair on frontier (Algorithm 1):
      evaluate f(τ_t, τ_v) = KL divergence between original and skipped model
      evaluate g(τ_t, τ_v) = fraction of experts skipped
      # Monotonicity: larger thresholds → more skipping → higher KL divergence
  (τ_t*, τ_v*) = argmin f(τ_t, τ_v) s.t. g(τ_t, τ_v) ≥ ρ
  # Time complexity: O(ND) vs naive O(ND²), ~45x speedup
  ```

  **在线推理阶段（per-token, per-MoE-layer）：**
  ```
  # Input: token x^{(l)} ∈ R^d at layer l, with modality indicator
  # x^{(l)} can be text token (I_t=1, I_v=0) or vision token (I_t=0, I_v=1)

  # Step 1: Standard MoE routing
  r^{(l)} = Router_l(x^{(l)})                    # routing logits: [M]
  π^{(l)} = softmax(r^{(l)})                      # routing probabilities: [M]
  S^{(l)} = topk_indices(π^{(l)}, k)              # top-k expert indices

  # Step 2: GMLG - compute importance scores with global modulation
  for i in S^{(l)}:
      s_i^{(l)} = α̃^{(l)} * π_i^{(l)}              # Eq.(3): global × local importance
      # α̃^{(l)} pre-computed offline, π_i^{(l)} from router

  # Step 3: DMT - modality-specific expert skipping
  τ = τ_t * I_t + τ_v * I_v                        # select threshold by modality
  active_experts = {i ∈ S^{(l)} : s_i^{(l)} ≥ τ}   # Eq.(5): keep only important experts

  # Step 4: Compute output with only active experts
  y^{(l+1)} = sum_{m ∈ active_experts} π_m^{(l)} · Expert_m^{(l)}(x^{(l)})
  # In practice: skipped experts → sentinel expert ID → filtered out during dispatch/gather
  ```

  关键设计要点：
  - GMLG在inference时无额外开销——α^{(l)}预计算，s_i^{(l)}仅需一次乘法
  - DMT对vision token的τ_v < τ_t（vision token expert冗余度更高），跳过更多vision experts
  - Frontier search exploit单调性：更大的τ → 更多跳过 → g递增、f也递增，只需O(ND)搜索
  - 校准数据鲁棒：GQA/COCO/VMMMU上α^{(l)}趋势一致，性能差异<1%

## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是LiquidQuant (LQQ)，一种硬件高效的两级W4A8量化算法。LQQ通过rotation-based transformation将INT8值映射到UINT8域再量化为UINT4，配合two's complement特性设计dequantization，仅需两条32-bit硬件指令（IMAD + XOR）处理四个元素，避免溢出。离线量化流程：FP16 → 第一级per-channel量化到INT8（限制在[-119, 119]范围）→ 第二级shift到UINT8域 → per-group量化到UINT4（group size=64）。激活量化动态使用SmoothQuant per-token量化。实验比较的算法baseline包括：QServe（W4A8，group size=128）、TRT-W4A16、TRT-W8A8、TRT-FP8、TRT-FP16。准确率评估使用WikiText2 perplexity、PIQA/ARC/HellaSwag/WinoGrande zero-shot accuracy。性能评估通过系统级吞吐量和kernel级延迟。

- 硬件平台是什么，配置是什么。
  NVIDIA H800 GPU（80GB HBM），Intel Xeon Platinum 8457C CPU，2.9TB RAM。软件：PyTorch 2.4.0，CUDA 12.4。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA（30B, 7B/2 13B/2 70B），LLaMA3-8B，Mistral-7B，Mixtral-8×7B，Yi-34B。数据集：WikiText2（perplexity），PIQA/ARC/HellaSwag/WinoGrande（zero-shot accuracy）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：LiquidGEMM未提供开源代码。论文说明LiquidGEMM已部署为ByteDance Seed生产LLM serving基础设施的主要GEMM kernel。baseline系统QServe开源（https://github.com/mit-han-lab/omniserve），TensorRT-LLM开源（https://github.com/NVIDIA/TensorRT-LLM）。

  算法pipeline（两级量化+dequantization）：

  离线量化阶段：
  ```
  // Step 1: FP16 → INT8 (per-channel, 第一级量化)
  W_fp16: shape [K, N]
  for c in range(N):  // per-channel
    s_i8[c] = (max(W_fp16[:,c]) - min(W_fp16[:,c])) / (119 - (-119))
    Q_i8[:,c] = clamp(round(W_fp16[:,c] / s_i8[c]), -119, 119)

  // Step 2: INT8 → UINT4 (per-group, 第二级量化, LiquidQuant)
  group_size = 64
  for c in range(N):
    for g in range(0, K, group_size):
      group = Q_i8[g:g+group_size, c]
      min_val = min(group)  // 负数
      Q_u8 = group - min_val  // shift到UINT8域 [0, max-min]
      s_u8 = round(max(Q_u8) / 15)  // scale factor, ≤16
      Q_u4 = round(Q_u8 / s_u8)  // UINT4 [0,15]

  // Precompute dequantization constant
  a = 128 + min(Q_i8)  // precomputed per-group, ∈ [0,255]
  ```

  在线dequantization（kernel main-loop内）：
  ```
  // Dequantize four UINT4 elements using two hardware instructions:
  // Input: packed UINT4 in 32-bit register reg_in
  //         s_u8: per-group scale factor (broadcast to 32-bit)
  //         a: precomputed offset (128+min, broadcast to 32-bit)

  // Unpack: expand 8 × 4-bit → 2 × 32-bit registers (QServe method)
  reg_lo = unpack_low_4bits(reg_in)
  reg_hi = unpack_high_4bits(reg_in)

  // Dequantization with two instructions per 4 elements (Equation 12):
  // Q_i8 = (Q_u4 * s_u8 + a) XOR 0x80
  result_lo = IMAD(reg_lo, s_u8, a)   // multiply-add, 1 instruction
  result_lo = XOR(result_lo, 0x80)    // flip MSB, 1 instruction

  result_hi = IMAD(reg_hi, s_u8, a)
  result_hi = XOR(result_hi, 0x80)

  // First-level dequantization in epilogue:
  // W_fp16 ≈ Q_i8 * s_i8 (back to FP16)
  ```

  关键特性：LQQ利用two's complement同余性质（i ≡ j mod 2^8 → 相同二进制表示）消除溢出。dequantization全部在UINT8域内计算，确保中间结果∈[0,255]。XOR 0x80等价于条件性地加/减128，将UINT8结果映射回INT8的二进制表示，可直接用于Tensor Core MMA。

## Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是以Block Low-Rank (BLR) 压缩技术——Monarch和BLAST——替换transformer模型中的dense线性层（Q/K/V/Oproj、gate/upproj、downproj等），将权重矩阵分解为多个小块的低秩因子表示。Monarch将dense权重划分为b₁×b₂块，每块独立做低秩分解W_{l,k}=V_{l,k}U_{l,k}，参数从i×o降至b₁b₂r'(p+q)，FLOP降至nb₁b₂r'(p+q)。BLAST进一步共享V_l和U_k并引入per-block对角矩阵S_{l,k}（W_{l,k}=V_l S_{l,k} U_k），参数为r(p+q+b²)，FLOP为nr(p+q+b²)，渐进复杂度与低秩相同但具有更高的表达能力和准确率。
  
  实验比较的算法baseline包括：Dense、Low-Rank (LR，标准SVD分解)、Monarch（Dao et al. 2022）、BLAST（Lee et al. 2024），全部在相同压缩比（CF=1.85×至3×）下对比。准确率评估：语言模型用WikiText-103/2 perplexity和zero-shot commonsense reasoning（PIQA, HellaSwag, Winogrande, BoolQ, ARC, OpenBookQA）accuracy；视觉模型用ImageNet classification accuracy；扩散模型用DDPM sampler生成图像后计算FID/sFID/IS vs 50,000 ImageNet validation images。

- 硬件平台是什么，配置是什么。
  NVIDIA A40（40GB显存，6MB共享L2 cache，数据中心GPU）和NVIDIA Jetson Orin Nano（8GB显存，4-6MB L2 cache，DDR DRAM，边缘GPU）。A40用于中大规模模型（Llama-7B、DiT-XL/2、Llama-3.2-1B），Jetson用于中小规模模型（Llama-3.2-1B、DiT-XL/2、GPT2-S、ViT-B）。软件：A40用Python 3.12.8、PyTorch 2.8.0、Triton 3.4.0、CUDA 12.6.3；Jetson用JetPack 6.2、L4T 36.4.3、CUDA 12.6.11、PyTorch 2.6.0、Triton 3.2.0。所有benchmarking用Triton do_bench()和PyTorch benchmarking utilities，torch.compile() + CUDA graph capture消除CPU dispatch overhead。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-7B（32层，i=o=4096，CF=2×）、Llama-3.2-1B（32层，i=o=2048，CF=2×）、GPT2-S（12层，i=o=768，CF=1.85×）、DiT-XL/2（28层，i=1152, o=3456/4608/6912，CF=2×）、ViT-B（12层，i=o=768，CF=3×）。数据集：WikiText-103/2（perplexity）、PIQA/HellaSwag/Winogrande/BoolQ/ARC/OpenBookQA（zero-shot accuracy）、ImageNet（分类accuracy）、SlimPajama-6B subset（re-training, 4000 steps）。所有替换层的具体配置（rank, blocks, i/o dimensions）记录在Table 3。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/pabillam/mem-efficient-blr

  BLAST算法pipeline（以Llama-7B QKVproj层为例，i=o=4096, r=1024, b=16, b₁=b₂=16, n=1024 tokens）：

  ```
  # BLAST权重参数化
  V ∈ R^{b₁ × p × r} = R^{16 × 256 × 1024}     # p = i/b₁ = 256
  S ∈ R^{b₁ × b₂ × r} = R^{16 × 16 × 1024}
  U ∈ R^{b₂ × r × q}   = R^{16 × 1024 × 256}   # q = o/b₂ = 256

  # 离线压缩（训练后/微调前）：
  # 1. 对dense权重 W[256:256, 256:256] 的每个 block (l,k) 执行preconditioned gradient descent
  #    300步求解: argmin ||W_{l,k} - V_l S_{l,k} U_k||_F
  #    SVD初始化 V_l, U_k, S_{l,k}=I
  # 2. 替代方案: SVD用于Low-Rank；block-wise SVD用于Monarch

  # 在线推理（BLAST线性层前向）：
  def blast_forward(X: [n, i]) -> Y: [n, o]:
      # Step 1: 对所有b₁个输入块并行计算 X_l @ V_l
      X_blocks = X.reshape(n, b₁, p)                    # [n, 16, 256]
      Z_l = batched_bmm(X_blocks, V)                    # [b₁, n, r] = [16, 1024, 1024]
      
      # Step 2: 对每个输出块k，计算加权求和
      for k in range(b₂):  # k = 0..15
          Y_k = zeros(n, q)                              # [1024, 256]
          for l in range(b₁):  # l = 0..15
              # Hadamard product with diagonal S_{l,k}
              Z_lk = Z_l[l] * S[l, k]                    # [n, r] ⊙ [r] → [n, r]
              Y_k += Z_lk @ U[k]                         # [n, r] @ [r, q] → [n, q]
      Y = concat([Y_0, ..., Y_{15}], dim=-1)            # [n, o]
  ```

  Monarch算法pipeline（同层配置，r'=r/b=64）：
  ```
  # Monarch权重参数化
  V ∈ R^{b₁ × (r'b₂) × p} = R^{16 × 1024 × 256}  # r'·b₂ = 64·16 = 1024
  U ∈ R^{b₂ × q × (b₁r')} = R^{16 × 256 × 1024}   # b₁·r' = 64·16 = 1024

  def monarch_forward(X: [n, i]) -> Y: [n, o]:
      X_blocks = X.reshape(n, b₁, p)                    # [n, 16, 256]
      # Step 1: 第一批bmm
      Z = batched_bmm(X_blocks, V.transpose(-1, -2))    # [b₁, n, r'b₂]
      # Step 2: 两次permutation (r'↔b₂, 然后 b₂↔b₁)
      Z = Z.reshape(b₁, n, b₂, r').transpose(0, 2, 1, 3)  # [b₂, n, b₁r']
      # Step 3: 第二批bmm
      Y_k = Z[k] @ U[k] for k in range(b₂)             # [n, q] each
      # Step 4: 最终permutation (b₂, n, q) → (n, q, b₂)
      Y = permute(Y, ...)
  ```

- 关键实验结果：
  - 准确率（Table 1）：BLAST在多数模型上取得最优accuracy-efficiency tradeoff。Llama-7B CF=2×: BLAST WikiText-2 PPL=14.21 vs Monarch 19.54 vs LR 26.33（Dense=9.37）；ViT-B CF=3×: BLAST ImageNet=79.3% vs Dense 78.7%（BLAST甚至略高于Dense）。
  - Roofline分析：多token推理(n=1024)下，Monarch和BLAST从compute-bound落入memory-bound，因为block结构产生大量中间数据移动。Monarch 1.14-1.68× slower than dense，BLAST 2.63-4.31× slower（在未优化的PyTorch实现下）。

## Mordal: Automated Pretrained Model Selection for Vision Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Mordal，一个自动化VLM预训练模型选择框架，包含三个核心算法组件：(1) Candidate Clustering——使用CKA (Centered Kernel Alignment) 计算vision encoder和language model的表示相似度，通过两步聚类（先聚类VE、再基于每个VE cluster的medoid聚类LLM）构建VLM候选聚类，每个cluster的候选具有相似性能；(2) Efficient Evaluation (Early Stopping)——采用Successive Halving Algorithm (SHA)在inter-cluster evaluation阶段早期淘汰表现差的cluster，每轮保留top 1/η候选，逐步增加评估budget；(3) Scaling Prediction——利用observational scaling law发现VLM alignment性能与训练数据量存在log-linear关系，通过对数线性回归从部分数据训练预测完整数据训练的最终性能，减少每个候选的评估时间。

  实验比较的baseline是grid search（穷举搜索），即对49个VLM候选（7个VE × 7个LLM）全部用full alignment data训练feature projector并评估。衡量指标：总搜索时间（GPU hours）、Top-1 model quality（accuracy）、Kendall's τ（Top-10候选排序一致性）。也对比了LLaVA-1.5-7B equivalent structure (CLIP-Vicuna) 的准确率。

- 硬件平台是什么，配置是什么。
  16× NVIDIA A40 GPU（每GPU 48 GB GDDR6），部署在cluster的一组VM上。软件：PyTorch（bfloat16精度）、HuggingFace Transformers、PEFT（LoRA）、Flash Attention-2。所有模型训练使用Adam optimizer（minibatch=4, initial lr=1e-4, linear schedule）。

- 模型是什么。数据集和bench分别是什么。
  模型：7个vision encoders（CLIP-ViT-L/14@336, SigLIP-so400m-patch14@384, DFN-CLIP-ViT-H/14@378, InternViT-300M/14@448, DINOv2-ViT-L/14@518, EVA-CLIP-02-ViT-L/14@336, ConvNeXt-L/14@256） + 7个LLMs（Vicuna-1.5-7B, Llama-2-7B, Llama-3-8B, Mistral-v0.2-7B, Qwen2-7B, Phi-3-Small-7B, Gemma-1.1-7B），共49个VLM候选。Feature projector: MLP (两层linear + GELU)。Alignment数据集：LLaVA-1.5-Instruction mixture。Benchmark数据集（3个domain 6个任务）：Visual QA — GQA, VizWiz；Doc QA — ChartQA, DocVQA；Knowledge — ScienceQA, AI2D。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/SymbioticLab/Mordal

  Mordal算法pipeline（以7个VE × 7个LLM = 49个candidate搜索GQA任务为例）：

  **阶段1: Candidate Clustering（候选聚类，Section 3.1 + Algorithm 2）：**
  ```
  # Step 1: Vision Encoder Clustering
  for each pair (VE_A, VE_B) in 7 vision encoders:
      # Pass calibration images through each VE, get output embeddings
      act_A = VE_A(images_from_target_task)   # shape: [N, D_ve_A]
      act_B = VE_B(images_from_target_task)   # shape: [N, D_ve_B]
      # Compute CKA similarity via MinibatchCKA
      cka_score = CKA(act_A, act_B)           # Equation (1)(2)
      dist = 1 - cka_score
      Dist_ve[VE_A][VE_B] = dist
  C_ve = HierarchicalClustering(Dist_ve, t_ve=0.7)
  # e.g., C_ve = [{CLIP, SigLIP, DFN-CLIP}, {InternViT, DINOv2}, {EVA-CLIP, ConvNeXt}]

  # Step 2: LLM Clustering (per VE cluster)
  for each VE_cluster in C_ve:
      medoid_ve = PickMedoidModel(VE_cluster)  # most central VE
      # Warm up a feature projector to match VE output to LLM input shape
      wp_projector = WarmupProjector(medoid_ve, alignment_data, rounds=10)
      fixed_ve_output = wp_projector(medoid_ve(images))

      for each pair (LLM_A, LLM_B) in 7 LLMs:
          # Feed fixed VE output to each LLM, get last hidden state
          rep_A = LLM_A.last_hidden_state(fixed_ve_output)
          rep_B = LLM_B.last_hidden_state(fixed_ve_output)
          cka_score = CKA(rep_A, rep_B)
          dist = 1 - cka_score
          Dist_llm[LLM_A][LLM_B] = dist

      C_llm = HierarchicalClustering(Dist_llm, t_llm=0.8)
      # e.g., for VE cluster 1: C_llm = [{Vicuna, Llama-2}, {Llama-3, Mistral, Qwen2}, {Phi-3, Gemma}]

      # Step 3: Cartesian product → VLM candidate clusters
      C_vlm.append(CartesianProduct(VE_cluster, C_llm))
      # e.g., for VE cluster 1: 2 × 3 = 6 candidate clusters
  # Total candidate clusters ≈ 10-15 (vs 49 individual candidates)
  ```

  **阶段2: Inter-cluster Evaluation with Early Stopping（Section 3.2 + Figure 5a）：**
  ```
  # Pick medoid candidate from each cluster as representative
  representatives = [PickMedoidCandidate(cluster) for cluster in C_vlm]
  # e.g., 10 representatives for 10 clusters

  # Successive Halving Algorithm (SHA)
  R = 0.125    # max data sample ratio (1/8 of alignment data)
  b = 0.03     # initial budget per candidate
  eta = 2      # reduction factor
  budget = b   # current budget per candidate

  while len(representatives) > top_k_inter (e.g., 3):
      for rep in representatives:
          # Train candidate with budget portion of alignment data
          train(rep, data_ratio=budget)
          # Evaluate on target task
          score[rep] = evaluate(rep, target_task)
      # Keep top 1/eta candidates
      keep = top_k(score, k=len(representatives) // eta)
      representatives = keep
      budget *= eta  # increase budget for next rung

      if convergence(representatives) or len(representatives) <= top_k_inter:
          break
  # After SHA: 3 representative candidates from 3 best clusters remain
  ```

  **阶段3: Intra-cluster Evaluation with Scaling Prediction（Section 3.2 + Algorithm 1）：**
  ```
  # Gather all individual candidates from remaining Top-K clusters
  C_remain = flatten(remaining_clusters)  # e.g., 3 clusters → 12-15 candidates

  # Scaling Prediction for each candidate
  for c in C_remain:
      P = []  # list of (log(r), log(Err)) pairs
      r = 0.125  # start from 1/8 data

      # Iteratively reduce data to find log-linear region
      while True:
          # Train from existing intermediate checkpoint if available
          train_from_checkpoint(c, data_ratio=r)
          Err = evaluate(c, target_task)
          P.append((log(r), log(Err)))

          if len(P) > p (e.g., 3):
              # Fit linear regression: log(Err) = α * log(r) + β
              f_c = LinearRegression(P)
              if fitting_loss(f_c) < delta (e.g., 5e-5):
                  break  # log-linear relationship confirmed

          r = r / u  # reduce data ratio (u = 2)

      # Predict performance at full data (r = 1)
      predicted_err = exp(f_c(log(1)))
      L.append((c, predicted_err))

  # Select candidate with best predicted performance
  best_candidate = argmin(L, key=lambda x: x[1])
  return best_candidate
  ```

  关键设计要点：
  - CKA可比较不同shape的表示（传统cosine similarity不可），且对MLP projection变换鲁棒
  - 两步聚类（先VE后LLM）避免对所有VE×LLM pair计算CKA，减少pair-wise计算量
  - SHA提供rough filtering，Scaling Prediction提供fine-grained ranking——二者正交
  - Scaling prediction利用observational scaling law：VLM alignment性能与训练数据量存在log-linear关系，但仅在一定数据量后出现
  - 聚类阈值t_ve=0.7和t_llm=0.8平衡聚类粒度和搜索效率
  - 除feature projector外，可使用LoRA fine-tune pretrained LLM
  - Flash Attention-2用于高效attention计算
  - Mordal自动将空闲GPU资源分配给未收敛candidate

- 关键实验结果：
  - Mordal 8.9×–11.6× faster than grid search（5439 GPU hours → 469-607 GPU hours）
  - 6个task中5个成功选出Top-1 candidate（1个选出Top-2，因最优candidate属于表现差的cluster被过早淘汰）
  - 所有找到的最优VLM性能均超过LLaVA-1.5-7B equivalent (CLIP-Vicuna)
  - Kendall's τ: 0.76–0.96（Top-10候选排序一致性）
  - Top-5候选中识别出4/5（GQA和AI2D）
  - Ablation: candidate clustering + early stopping + scaling prediction三者协同效果最优
  - 敏感性分析：Mordal对大多数超参数鲁棒，clustering threshold t_ve影响最大

## Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Twilight，一个hierarchical KV cache pruning框架，通过引入top-p (nucleus) sampling到sparse attention中，为任何现有sparse attention算法赋予自适应budget选择能力。核心算法包含三层：(1) **Token Selector**——将现有top-k sparse attention算法（如Quest、DS）作为黑盒，使用保守的大budget（如1/4 sparsity）预选token子集；(2) **Twilight Pruner**——基于top-p thresholding，将选出的token子集进一步剪枝：计算normalized attention weights，通过binary search找到最小预算B使得累积概率≥阈值p（通常p=0.85-0.95），仅保留top-p token；(3) **Sparse Attention Kernel**——仅对top-p token执行最终精确attention计算。

  实验比较：(a) Accuracy —— Longbench 12 tasks + RULER (long-context) + GSM8K/COQA/PG-19 (medium-context)，比较Quest/DS baselines在不同budget(256-8192)下 vs Quest-Twi/DS-Twi的自适应budget；(b) Efficiency —— self-attention operator speedup (FlashInfer-Twi vs FlashInfer, Quest-Twi vs Quest, vs FlashAttention2) + end-to-end decoding TPOT (batch=32-256)；(c) Ablation —— p threshold sensitivity (perplexity vs speed on PG-19/TrivialQA), time breakdown (TokenSel+Pruner+SparseAttn)。

- 硬件平台是什么，配置是什么。
  单张NVIDIA A100 GPU（40GB/80GB HBM）。Software: PyTorch, CUDA, OpenAI Triton, FlashInfer。Batch inference实验。

- 模型是什么。数据集和bench分别是什么。
  模型：Longchat-7B-v1.5-32k (MHA, 32k context)、LLaMA2-7B-Chat (MHA)、LLaMA-3.1-8B-Instruct (GQA, 128k context)。数据集：Longbench (12 tasks: Qasper, MF-en, HotpotQA, 2WikiMQA, Musique, GovReport, QMSum, MultiNews, TriviaQA, PR-en, LCC, Repobench-P), RULER (16k-96k), GSM8K (8-shot CoT), COQA, PG-19 (perplexity)。Efficiency datasets: Qasper, GovReport, LCC from Longbench (10k-30k prompts)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/tsinghua-ideal/Twilight

  Twilight hierarchical pruning pipeline（以Quest为base algorithm, p=0.85为例）：
  ```
  # 输入: query q ∈ R^{1×d}, FP16 K,V ∈ R^{N×d}, INT4 K_quantized ∈ R^{N×d/2} (额外存储)
  
  # Step 1: Token Selector (base algorithm, e.g., Quest with conservative budget B0=N/4)
  # Quest 使用per-page max-pooling估计token重要性
  K_pooled = max_pool(K, page_size=16)     # R^{N/16 × d}
  scores_approx = q @ K_pooled^T            # R^{1 × N/16}
  top_pages = topk(scores_approx, k=B0/16)  # 选择top pages
  I0 = expand_pages_to_tokens(top_pages)    # |I0| = B0 tokens
  
  # Step 2: Twilight Pruner (top-p sparsity)
  # 2a: SpGEMV with INT4 K cache —— 估计attention weights
  # K_quantized: per-head asymmetric INT4 quantization, paged layout
  q_fp16 @ K_int4^T → W_approx ∈ R^{1×N}   # 使用FlashInfer的SpGEMV kernel
  # Dequantization: K_fp16 ≈ (K_int4 - zero) * scale, per-head动态量化
  # INT4在shared memory中解包，cp.async异步加载，2-stage pipeline隐藏延迟
  
  # 2b: Normalize to get attention weights
  W_norm = softmax(W_approx[I0])            # 仅对I0中的token做softmax
  
  # 2c: Top-p via Binary Search (Algorithm 1, GPU并行)
  l=0, r=max(W_norm), m=(l+r)/2
  repeat:
    W0 = where(W_norm < m, 0.0, W_norm)     # mask below threshold
    if sum(W0) >= p: l = m                  # 累积概率足够，提高阈值
    else: r = m                              # 累积概率不够，降低阈值
  until max(W_norm[W_norm > r]) - min(W_norm[W_norm >= l]) < ε
  # 所有element-wise操作(max/where/sum)融合为单次GPU循环，tensorized执行
  I1 = indices where W_norm >= l            # |I1| = B1 << B0, 自适应budget
  M = mask[I1] = 1                          # 稀疏mask
  
  # Step 3: Sparse Attention Kernel (FlashInfer-based, 仅计算I1中的token)
  # 使用head-wise varlen attention for MHA, group-wise varlen for GQA
  # GQA处理: 同一query group内取各head选择token的union
  S = q @ K[I1]^T / sqrt(d)                # 精确FP16 attention scores
  P = softmax(S)                             # online softmax
  O = P @ V[I1]                              # 精确FP16 attention output
  ```

  关键设计要点：
  - Token Selector使用保守budget B0≈N/4保持高recall，Pruner用top-p做精确筛选
  - INT4 K cache降低SpGEMV的memory access至1/4（2-bit精度不足，8-bit浪费带宽）
  - Top-p binary search将排序O(N log N)降为O(log(range/ε))次并行reduction
  - Head-wise dynamic budget → 使用FlashInfer的load balancing (flatten head dim)处理不平衡
  - Extra memory: INT4 K cache = 1/8 × FP16 KV cache（可复用base algorithm已有的INT4 K cache）
  - p的选择比k更鲁棒：p代表累积概率，对不同分布head/layer/query的敏感度远低于k

## LMFusion: Adapting Pretrained Language Models for Multimodal Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是LMFusion框架，核心创新为：(1) 模态特异性Transformer模块——将预训练text-only LLM (Llama-3 8B) 的权重复用于文本处理，同时引入并行的图像专用Transformer模块用于图像扩散（diffusion）处理；(2) 模态特异性设计——FFN层、QKV投影、LayerNorm均为模态独立，各模态数据路由到各自的专用模块处理；(3) 共享self-attention层——跨模态交互通过共享的自注意力实现，text和image的Q/K/V在attention层内concat后进行统一注意力计算；(4) 文本模块冻结、仅训练图像模块——通过设text学习率η_text=0保持Llama-3的语言能力；(5) 学习率解耦——text和image参数组使用独立学习率。

  实验比较：(1) **主实验** vs Transfusion 7B（从头训练的多模态生成模型）——0.5× FLOPs配置（仅用image data，匹配Transfusion的图像数据量）和1× FLOPs配置（匹配总FLOPs）；(2) **Ablation实验**——No separation（dense Llama-3直接finetune）、Shallow separation（仅FFN模态特异性）、Deep separation（FFN+Attention模态特异性，即LMFusion）三种架构对比，以及不同学习率比η_text/η_image ∈ {0, 0.1, 1}的影响；(3) **LLaVAFusion扩展**——从LLaVA-NeXT 8B出发延续LMFusion范式，与EMU-3、Show-O、Janus、Chameleon、MetaMorph、Transfusion对比image understanding（MMMU/ChartQA/RealWorldQA/MME-P）和generation（FID）。

- 硬件平台是什么，配置是什么。
  论文未明确说明训练和评估的具体GPU型号及数量。论文提及使用的是Llama-3 8B模型初始化和Transfusion训练recipe，推测类似Meta FAIR的基础设施（通常为NVIDIA H100集群）。训练配置：最大上下文长度4096 tokens，batch size 250K tokens/image tokens。论文未说明推理延迟或GPU利用率等硬件性能指标。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3 8B（text backbone），额外引入U-Net downsampler/upsampler（0.27B参数，从头训练）。VAE encoder使用stabilityai/sd-vae-ft-mse（将256×256图像压缩为32×32×8 tensor，经2-block U-Net downsampler后得256 patches）。LLaVAFusion扩展基于LLaVA-NeXT 8B。
  
  数据集：训练数据——380M Shutterstock image-caption pairs（与Transfusion一致），80% caption-before-image顺序，20% image-before-caption。图像编辑finetuning——8K MagicBrush image editing examples。
  
  Benchmarks：
  - 语言能力：HellaSwag、PIQA、SIQA、WinoGrande（accuracy）
  - 图像理解：MS-COCO Captioning test split（CIDEr scores）
  - 图像生成：MS-COCO 30K validation prompts（FID、CLIP scores），含无classifier-free guidance (CFG=1.0) 和CFG=1.55两版本
  - LLaVAFusion额外benchmark：MMMU、ChartQA、RealWorldQA、MME-Perception、FID

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文发表于NeurIPS 2025 (https://papers.nips.cc/paper_files/paper/2025/hash/0d33b1148c6ee6bb3ea9f056ae3301e6-Abstract-Conference.html)，arXiv: 2412.15188。截至搜索时，论文未明确提供官方代码仓库链接，但有HuggingFace论文页面 (https://huggingface.co/papers/2412.15188)。基础模型Llama-3 8B为Meta开源权重，VAE encoder (sd-vae-ft-mse) 为Stability AI开源。

  LMFusion算法pipeline核心流程（以1个Transformer layer，text token数为M，image patch数为N为例）：

  **初始化阶段：**
  ```
  # 文本模块初始化（从Llama-3加载，冻结）
  Proj_text = Llama3.embed_tokens          # 线性embedding
  QKV_text = Llama3.self_attn.qkv_proj     # QKV投影 [d, 3d]
  O_text = Llama3.self_attn.o_proj         # O投影 [d, d]
  FFN_text = Llama3.mlp                    # FFN (gate+up+down)
  LM_Head_text = Llama3.lm_head            # 输出投影 head

  # 图像模块初始化（从Llama-3加载，训练）
  QKV_img = copy(Llama3.self_attn.qkv_proj)  # 并行QKV投影
  O_img = copy(Llama3.self_attn.o_proj)      # 并行O投影
  FFN_img = copy(Llama3.mlp)                # 并行FFN
  UNet_Down_img = random_init()             # 从头训练 (0.27B)
  UNet_Up_img = random_init()
  ```

  **前向传播（per layer）：**
  ```
  # Step 1: Input Projection
  h_txt = Proj_text(x_txt)                 # [M, d]  文本embedding
  h_img = UNet_Down_img(x_img_t, t)        # [N, d]  图像下采样，t为扩散时间步

  # Step 2: Modality-specific QKV projection
  Q_txt, K_txt, V_txt = QKV_text(h_txt)   # 各 [M, d]
  Q_img, K_img, V_img = QKV_img(h_img)    # 各 [N, d]

  # Step 3: Cross-modal self-attention
  # 文本token的attention（Eq.9）:
  K_all = concat(K_img, K_txt)            # [M+N, d]
  V_all = concat(V_img, V_txt)            # [M+N, d]
  A_txt = softmax(Q_txt @ K_all^T / sqrt(d) + M)  # [M, M+N]，M为混合mask
  h_O_txt = O_text(A_txt @ V_all)        # [M, d]

  # 图像token的attention（Eq.10）:
  K_all' = concat(K_txt, K_img)           # [M+N, d]
  V_all' = concat(V_txt, V_img)
  A_img = softmax(Q_img @ K_all'^T / sqrt(d) + M)  # [N, M+N]
  h_O_img = O_img(A_img @ V_all')        # [N, d]
  # M: causal mask for text tokens (i<=j), bidirectional for image tokens

  # Step 4: Modality-specific FFN
  h_FFN_txt = FFN_text(h_O_txt)           # [M, d]  冻结参数，无梯度
  h_FFN_img = FFN_img(h_O_img)            # [N, d]  可训练参数

  # Step 5: Output projection
  p_logits = LM_Head_text(h_FFN_txt)      # [M, vocab_size]  文本logits
  ε_pred = UNet_Up_img(h_FFN_img, t, h_img)  # [N, 32*32*8]  预测噪声
  ```

  **训练目标（Eq.4）：**
  ```
  # LM loss on text tokens
  L_LM = CrossEntropy(p_logits, x_txt_labels)

  # DDPM loss on image tokens  
  L_DDPM = MSE(ε_pred, ε)   # ε ~ N(0,I) 为真实噪声

  # Total loss
  L = L_LM + λ * L_DDPM

  # 参数更新（仅图像模块）：
  θ_img = {UNet_Down_img, QKV_img, O_img, FFN_img, UNet_Up_img}
  θ_img = θ_img - η_img * ∇L(θ_img)
  # θ_text = {Proj_text, QKV_text, O_text, FFN_text, LM_Head_text} 冻结
  ```

  **推理阶段图像生成：**
  ```
  # 文本条件编码（单次前向）
  h_txt_all = text_forward(prompt)        # 文本token通过冻结的文本模块

  # 扩散去噪循环（T步）
  x_T ~ N(0, I)  # 初始纯噪声
  for t = T, T-1, ..., 1:
      ε_pred_t = image_forward(x_t, t, h_txt_all)  # 图像模块 + cross-attn to text
      x_{t-1} = denoise_step(x_t, ε_pred_t, t)     # DDPM/DDIM sampler
  generated_image = VAE_decoder(x_0)
  ```

  关键设计要点：
  - 文本和图像模块均从Llama-3初始化，使图像模块获得文本预训练的knowledge transfer
  - 虽然参数量是Transfusion的2倍，但每个token仅激活对应模态的模块（一半参数），FLOPs与Transfusion相同
  - Cross-modal attention是双向的——文本可attend到图像、图像可attend到文本，实现模态间信息融合
  - 混合attention mask：文本使用因果mask（autoregressive），图像使用双向mask（diffusion的去噪特性）
  - 训练时80%数据为caption→image顺序（训练图像生成）使模型学习文本条件下的图像生成

## Marconi: Prefix Caching for the Era of Hybrid LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Marconi prefix caching系统中两个核心算法：(1) Taxonomy-based Judicious Admission算法——通过radix tree bookkeeping将前缀复用模式分类为Purely Input（系统提示词等被多请求共享）和Input+Output（对话历史续写），对每种模式采用不同缓存策略，每序列至多2个SSM state checkpoint；(2) FLOP-Aware Eviction算法——定义FLOP Efficiency = Total FLOPs saved / Memory consumption of cached states，Utility Score = recency + α × flop_efficiency，替代传统LRU-only eviction。α参数由config_tuner根据workload自动调优。

  实验比较算法baseline：(1) fine-grained checkpointing（naive admission，每x token存checkpoint，使用LRU eviction）；(2) SGLang+ LRU eviction（recency-only eviction）。评估指标：token hit rate (%)。从algorithms角度，比较的是admission policy的精准度（judicious vs naive）和eviction policy的计算感知能力（FLOP-aware vs recency-only）。

- 硬件平台是什么，配置是什么。
  论文实验为离线trace-based模拟评估，运行于Cloudlab节点（Ubuntu 22.04, 32-core CPU）。算法本身与具体GPU硬件解耦——admission/eviction策略通过radix tree操作实现，不依赖特定GPU kernel。

- 模型是什么。数据集和bench分别是什么。
  模型：NVIDIA Mamba2-Hybrid-7B，层结构为4 Attention + 24 SSM + 28 MLP layers。Tokenizer: meta-llama/Llama-2-7b-hf。实验也对不同SSM-to-Attention比例进行了sweep（如Jamba等架构）。数据集/workloads：LMSys-Chat-1M（conversational）、ShareGPT_Vicuna_unfiltered（conversational）、SWEBench（agentic，长上下文代码agent轨迹）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/ruipeterpan/marconi。

  Algorithm 1: Judicious Admission（基于radix tree的前缀复用模式分类）
  ```
  // Radix tree node structure
  class RadixNode:
    token_sequence: List[int]        // 从根到该节点的token路径
    kv_cache: List[Tensor]           // Attention层KV cache (per-token)
    ssm_states: Optional[List[Tensor]] // SSM层recurrent states (仅特定节点)
    reuse_type: Enum["purely_input", "input_output", "none"]
    children: Dict[int, RadixNode]

  // Admission on new request arrival
  def admit(request_tokens: List[int], radix_tree: RadixNode):
      node = radix_tree.root
      match_len = 0
      // Step 1: Traverse radix tree to find longest prefix match
      for i, token in enumerate(request_tokens):
          if token in node.children:
              node = node.children[token]
              match_len += 1
          else:
              break

      // Step 2: Speculative insertion - check if new branching point
      remaining = request_tokens[match_len:]
      curr = node
      for token in remaining:
          if token not in curr.children:
              curr.children[token] = RadixNode(token_sequence=...)
          curr = curr.children[token]

      // Step 3: Classify reuse pattern
      // Purely input: intermediate node shared by multiple requests
      if curr has multiple descendant leaf paths:
          curr.reuse_type = "purely_input"
          cache_all_states(curr)  // admit KV + SSM states

      // Input+output: leaf node (end of sequence)
      if curr is leaf:
          curr.reuse_type = "input_output"
          cache_final_ssm_state_only(curr)  // only last token's SSM state
  ```

  Algorithm 2: FLOP-Aware Eviction
  ```
  // Compute FLOP efficiency for a cached entry
  def flop_efficiency(entry: RadixNode, model_config: ModelConfig):
      // Total FLOPs saved = sum of FLOPs for all layers covering this prefix
      total_flops = 0
      for layer in model_config.layers:
          if layer.type == "attention":
              // Attention FLOPs: O(L^2 * d) for prefill
              total_flops += layer.attention_flops(entry.prefix_len)
          elif layer.type == "ssm":
              // SSM FLOPs: O(L * d_state * d_model)
              total_flops += layer.ssm_flops(entry.prefix_len)
          elif layer.type == "mlp":
              total_flops += layer.mlp_flops(entry.prefix_len)

      // Memory consumed by this cache entry
      memory_bytes = 0
      for layer in model_config.layers:
          if layer.type == "attention":
              // KV cache: 2 * L * d_head * num_heads * sizeof(fp16)
              memory_bytes += 2 * entry.prefix_len * layer.d_head * layer.num_heads * 2
          elif layer.type == "ssm":
              // SSM state: fixed size, d_state * d_model * sizeof(fp32)
              memory_bytes += layer.d_state * layer.d_model * 4

      return total_flops / memory_bytes  // FLOPs per byte

  // Eviction decision
  def evict(cache: Dict, α: float):
      scores = []
      for entry_id, entry in cache.items():
          recency = current_time - entry.last_access_time
          flop_eff = flop_efficiency(entry, model_config)
          // Utility = recency + α × flop_efficiency
          utility = recency + α * flop_eff
          scores.append((entry_id, utility))

      // Evict entry with lowest utility
      victim = argmin(scores, key=lambda x: x[1])
      cache.remove(victim)
  ```

  关键设计决策：
  - SSM state大小固定（与序列长度无关），而KV cache大小随序列长度线性增长——因此长前缀的FLOP efficiency更高，Marconi优先保留
  - α参数自动调优：config_tuner.py根据workload命中率反馈动态调整
  - 每个序列最多2个SSM state checkpoint：避免naive checkpointing的稀疏命中问题
  - 统一radix tree管理KV+SSM状态：因为所有层的state必须代表同一前缀才能复用

## Modulated Diffusion (MoDiff): Accelerating Generative Modeling with Modulated Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Modulated Diffusion (MoDiff)，一个加速扩散模型采样过程的框架，包含两个核心算法：(1) Modulated Quantization——利用扩散过程相邻时间步之间激活的相似性，将每层线性算子的计算从直接量化激活重构为量化时序差分：o_t = A(Q(a_t - a_{t+1})) + o_{t+1}。时序差分 a_t - a_{t+1} 的分布范围比原始激活小10×以上，更集中且异常值更少，因此量化误差显著降低；(2) Error-Compensated Modulation——通过中间变量 â_t = Q(a_t - â_{t+1}) + â_{t+1} 跟踪量化误差 e_t = a_t - â_t，在下一时间步将误差反馈到输入中（â_t 替代 a_{t+1} 作为差分基准），实现误差补偿而非累积，理论证明误差以指数速率(2c)^{T-k-1}递减（相比标准调制的2^{T-k-1}c线性以上累积）。

  实验比较的算法baseline：(1) Q-Diffusion (Q-Diff)——基于MSE reconstruction loss的PTQ方法，使用time-step-aware校准数据采样，8/8 bit为强baseline；(2) Dynamic Channel-wise Quantization (LCQ)——基于BRECQ框架，per-channel min-max动态量化；(3) Dynamic Tensor-wise Quantization (LTQ)——per-tensor min-max动态量化，更硬件友好；(4) Full Precision Activation (32-bit) 作为上限。评估指标：IS (Inception Score)、FID (Fréchet Inception Distance)、sFID (Sliced FID)、Precision/Recall、GBops（通过DeepSpeed计算理论binary operations）。核心实验证实在W8A4及以下位宽时MoDiff的优势急剧扩大——baseline方法在W8A4时质量塌陷（FID>300），MoDiff保持FID≈4。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体GPU型号用于实验。效率评估使用DeepSpeed计算GBops（binary operations per denoising step per image），而非实际wall-clock hardware speedup。论文明确说明"Implementing acceleration on specialized hardware is beyond the scope of this work"。Weight quantization使用MSE reconstruction method（Q-Diffusion checkpoint），activation量化使用动态per-channel/tensor min-max scaling。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) DDIM on CIFAR-10（32×32, 100 denoising steps）；(2) Latent Diffusion Models (LDM-4) on LSUN-Bedrooms（256×256, 500 sampling steps）；(3) LDM-8 on LSUN-Churches（256×256, 200 steps）；(4) Stable Diffusion v1.4 with DPM-Solver on MS-COCO 2014（50 steps, 30K images generated）；(5) DiT-XL/2 on ImageNet 256×256（50 steps, 10K images）；(6) SDXL-Turbo (few-step) on MS-COCO。数据集：CIFAR-10（50K generated images for evaluation）、LSUN-Bedrooms、LSUN-Church-Outdoor、MS-COCO 2014、ImageNet 256×256。评估指标：IS/FID/sFID（基于50K generated images）、Precision/Recall。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/WeizhiGao/MoDiff

  MoDiff算法pipeline（以DDIM on CIFAR-10，T=100 denoising steps，单层linear operator A^{(l)}为例）：

  **离线阶段（Calibration for Q-Diff + MoDiff）：**
  ```
  # 重构校准数据集以捕获时序差分信息
  for each calibration sample:
      # 完整运行扩散采样过程
      for t = T, T-1, ..., 1:
          # 标准扩散步骤获取每层的激活
          a_t^{(l)} = layer_l_input(x_t, t)    # 第l层在时间步t的输入激活

  # 对每层使用MoDiff重跑校准以学习量化参数
  for layer l in 1..L:
      # 在Q-Diff框架中，用MSE reconstruction学习scaling factor s
      # 输入变为时序差分而非原始激活
      loss = MSE(A^{(l)}(a_t^{(l)}), A^{(l)}(Q(a_t^{(l)} - â_{t+1}^{(l)})) + ô_{t+1}^{(l)})
      s^{(l)} = argmin loss  # 学习per-channel/per-tensor scale
  ```

  **在线推理阶段（逐denoising step, 逐layer）：**
  ```
  # Step 1: 第一个时间步T（warm-up，使用全精度激活避免初始量化误差）
  â_T = Q(a_T)                                       # Eq.(8): 量化激活
  ô_T = A^{(l)}(â_T)                                  # Eq.(9): 全精度计算
  # 可选：重复warm-up 4-5步收敛到全精度激活

  # Step 2: 后续时间步 t = T-1, T-2, ..., 1
  for t = T-1, T-2, ..., 1:
      # Error-compensated modulated quantization (Eq.13):
      â_t = Q(a_t - â_{t+1}) + â_{t+1}                # 量化时序差分 + 补偿前一量化激活
      # 等价于: â_t = (a_t - â_{t+1} - e_t') + â_{t+1}
      #          = a_t - e_t'
      # 其中 e_t' 是当前时间步的量化误差

      # Modulated computation (Eq.14):
      ô_t = A^{(l)}(Q(a_t - â_{t+1})) + ô_{t+1}       # 计算差分输出 + 累加上一时间步的输出
      # 等效于: ô_t = A^{(l)}(â_t)   (但实际通过差分计算)

      # 误差追踪分析 (Eq.18):
      e_t = (a_t - â_{t+1}) - Q(a_t - â_{t+1})       # 当前步量化误差
          = (a_t - â_{t+1}) - (â_t - â_{t+1})
          = a_t - â_t                                  # 该误差将在t-1步被补偿
  ```

  关键设计要点：
  - Bias Removal: 应用MoDiff的层必须去除bias项，因为Eq.(13)需要对算子做线性分解（A(a+b)=A(a)+A(b)要求A为纯线性算子，无bias）
  - Warm-up: 第一步使用全精度激活，避免初始量化误差被累积。经4-5次warmup后量化误差收敛到可忽略水平
  - Calibration Dataset Reconstruction: Q-Diff+MoDiff重新构造校准数据以捕捉时序差分而非原始激活
  - Layer-wise Reconstruction: 逐层独立重建（非整块重建），性能更稳定
  - 0-bit skipping (Remark 4.1): 当时序差分幅度低于可容忍阈值时，MoDiff允许分配0-bit表示跳过计算——此时等效于caching方法的特例

  定理保证：
  - Theorem 4.3: 量化误差 ∥x-Q(x)∥²₂ ≤ (max(x)-min(x))²d/(2^b-1)²，即误差正比于输入范围的平方。时序差分的范围比原始激活小10×+，因此相同位宽下量化误差降低100×+，或可用低3-4位达到相同误差界
  - Theorem 4.4: 标准调制误差以2^{T-k-1}c速率累积（指数增长），而error-compensated调制以(2c)^{T-k-1}速率递减（c<1/2时指数衰减），而非累积

## OmniVinci Enhancing Architecture and Data for Omni-Modal Understanding LLM

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出三个架构创新：(i) **OmniAlignNet**：通过CLIP-style对比学习在共享omni-modal潜在空间中强化视觉和音频嵌入的对齐；(ii) **Temporal Embedding Grouping (TEG)**：基于时间戳将视觉和音频嵌入按时间组重新排列，编码相对时序信息；(iii) **Constrained Rotary Time Embedding (CRTE)**：通过频率几何级数（$\omega_i = 2\pi / (T_{\max} \theta^{i/C})$）和元素级旋转变换编码绝对时间戳。此外提出**Omni-Modal Data Engine**：先用视觉/音频captioning模型独立生成标注，再用LLM进行跨模态纠错和总结生成omni-modal captions，最后用reasoning LLM合成QA对。训练策略包括Implicit Learning（利用视频自带的audio track进行隐式omni-modal监督）和Explicit Learning（通过data engine合成显式omni-modal标注数据）。最后应用GRPO post-training增强omni-modal reasoning。
  - 实验比较：(1) Ablation: Token Concatenation Baseline vs +TEG vs +Learned Time Embedding vs +RoTE vs +CRTE vs +OmniAlignNet，评估Worldsense/Dailyomni/Omnibench；(2) Implicit vs Explicit Learning消融，评估Video-MME;(3) 最终模型与Qwen2.5-Omni、Gemini、GPT-4o等对比omni/audio/video/image benchmarks；(4) GRPO消融；(5) downstream tasks（机器人导航、体育解说、语音翻译、医疗AI、半导体制造）。

- 硬件平台是什么，配置是什么。
  - 训练：NVIDIA DGX H100基础设施（论文致谢部分提及）。
  - 推理部署：NVIDIA A100（体育解说实验，AWQ量化后1.85s/clip）、NVIDIA L40s GPU、GeForce RTX 4090（24GB，评估latency：1.7x faster TTFT, 2.72x faster decoding vs Qwen2.5-Omni）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：OmniVinci-9B，基于Qwen2.5-7B-Instruct LLM backbone，SigLip vision encoder (paligemma-siglip-so400m-patch14-448) + Dynamic S2 (2x2 spatial scale-then-compress)，AF-Whisper audio encoder (Audio Flamingo 3 backbone)，2-layer MLP projectors。
  - 训练数据：24M多模态对话样本（image 36%, non-speech sound 21%, speech 17%, omni 15%, video 11%），150+子数据集。Omni-modal conversations: 3.6M；Image-text: 8M；Video-text: 2.7M；Speech-text: 5.3M（ASR）；Sound-text: 4.3M（audio QA/captioning）。
  - Benchmarks：Omni: Worldsense, Dailyomni, Omnibench；Audio QA: MMAR, MMAU；Speech Recognition (WER): LibriSpeech clean/other, AMI, Tedlium, VoxPopuli；Video: LongVideoBench, MVBench, Video-MME；Image: AI2D, ChartQA, DocVQA, InfoVQA, MathVista, MMMU, RealWorldQA, SEED, TextVQA, VQAv2；Downstream: R2R-CE (robot nav), SPORTU-video, CoVoST2 (speech translation), WM-811K (wafer defect), UCR time-series。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub (NVlabs/OmniVinci)，HuggingFace (nvidia/omnivinci)，arXiv: 2510.15870。License: NVIDIA OneWay Noncommercial License。
  - 算法Pipeline（OmniAlignNet核心张量计算）：
    ```
    # 输入: 视频的视觉嵌入 E_v ∈ R^{Nv×C} 和音频嵌入 E_a ∈ R^{Na×C}
    # 可学习query: Q_v ∈ R^{1×C}, Q_a ∈ R^{1×C}
    
    # Step 1: Query-based projection (cross-attention)
    V_proj = CrossAttn(Q_v, E_v, E_v)  # → R^{1×C}
    A_proj = CrossAttn(Q_a, E_a, E_a)  # → R^{1×C}
    
    # Step 2: 3-layer self-attention + L2 normalize
    for batch with K videos:
        V = L2Norm(SelfAttn3(V_proj_batch))  # → R^{K×C}
        A = L2Norm(SelfAttn3(A_proj_batch))  # → R^{K×C}
    
    # Step 3: CLIP-style contrastive loss
    s_ij = dot(V_i, A_j)  # 相似度矩阵
    L_v→a = -1/K * Σ_i log(exp(s_ii) / Σ_j exp(s_ij))
    L_a→v = -1/K * Σ_i log(exp(s_ii) / Σ_j exp(s_ji))
    L_o-align = (L_v→a + L_a→v) / 2
    ```
  - CRTE核心计算：
    ```
    # 基础频率 (geometric progression)
    ω_i = 2π / (T_max * θ^{i/C}), for i = 0,...,C-1
    
    # 频率调制
    Ω_{i,j} = ω_i * t_j  # 维度i，时间戳t_j
    
    # Rotary Embedding (类似RoPE)
    CRTE(x, Ω) = x ⊙ cos(Ω) + RotateHalf(x) ⊙ sin(Ω)
    # RotateHalf(x) = [-x_2, x_1, -x_4, x_3, ..., -x_C, x_{C-1}]
    ```
  - 训练Pipeline（7阶段）：
    1. Vision Projector Alignment → 2. Vision Encoder Alignment → 3. Vision Pre-Training → 4. Image Instruction Tuning → 5. Video Instruction Tuning → 6. Audio Projector & Encoder Alignment + Audio Instruction Tuning → 7. Omni-Modal Joint Training (200B tokens, cosine LR schedule, base LR=2e-5, vision/audio encoders frozen)。GRPO post-training: 18K omni-modal MCQ, 64 frames, max prompt 1024 tokens, max response 2048 tokens, rollout=8, temperature=1.0, top-p=0.99。

## SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是SLA（Sparse-Linear Attention），一个可训练的混合稀疏+线性注意力方法，用于加速Diffusion Transformer（DiT）模型，特别是视频生成场景。核心算法包括三个关键设计：(1) 三级注意力权重分类——通过压缩注意力权重矩阵P_c = Softmax(pool(Q)pool(K)^T/√d)将注意力权重块动态分为critical（top k_h%）、marginal（中间k_h%~k_l%）和negligible（bottom k_l%）三类，对应应用O(N²)稀疏FlashAttention、O(N)线性注意力和跳过三种策略；(2) 统一融合GPU kernel——将稀疏注意力和线性注意力的前向和反向计算融合到单个GPU kernel中，预计算h_j = φ(K_j)^T V_j和z_j = rowsum(φ(K_j)^T)使marginal块仅需单次矩阵加法；(3) 可学习投影层Proj(O^l)——对线性注意力输出O^l应用可学习线性变换R^d→R^d，减少softmax注意力和线性注意力之间的分布不匹配。仅需少量fine-tuning步骤（2000步，<0.1% pretraining cost），SLA即可将注意力计算减少95%而不损失生成质量。

  实验比较的baseline包括：(1) VSA——训练式稀疏注意力（89% sparsity）；(2) VMoBa——训练式MoE block注意力（85% sparsity）；(3) SpargeAttn-F——训练无关稀疏注意力（85% sparsity）；(4) SpargeAttn-T——训练式稀疏注意力（84% sparsity）；(5) Linear Only——仅线性注意力；(6) Sparse Only——仅SLA的稀疏组件；(7) L+S——稀疏和线性注意力的简单输出相加（无Proj层）。视频质量用VBench的IQ/OC/AQ/SC + Vision Reward + Aesthetic/Technical Video Quality评估，效率用FLOPs和FLOPS评估。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 5090 GPU用于kernel速度和端到端延迟评估。FlashAttention2作为参考attention实现。训练使用batch size=64，fine-tune 2000 steps。软件：PyTorch，自定义CUDA kernel实现SLA前向和反向pass。

- 模型是什么。数据集和bench分别是什么。
  模型：Wan2.1-1.3B（视频生成，30K sequence length用于视频生成）为主要实验模型；LightningDiT-1p0B/1（1.03B参数，图像生成）用于补充实验。数据集：私有数据集（来自Pexels和Common Crawl，20,000个5秒480p视频）用于视频fine-tuning；ImageNet 512×512用于图像实验。Benchmarks：VBench（IQ/OC/AQ/SC四个维度）、Vision Reward（人类偏好）、Aesthetic Video Quality (VA)、Technical Video Quality (VT)；图像用FID。FLOPs和FLOPS用于效率评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/thu-ml/SLA（论文声明代码即将公开）。

  SLA算法pipeline（以Wan2.1-1.3B注意力层，block size b_q=b_{kv}=64, k_h=5%, k_l=10%, φ=softmax为例）：

  **压缩mask预测（离线/在线都执行）：**
  ```
  # Step 1: 预测压缩注意力权重矩阵 (Eq.2)
  Q_pool = mean_pool(Q, block_size=b_q)      # R^{N/b_q × d}
  K_pool = mean_pool(K, block_size=b_{kv})    # R^{N/b_{kv} × d}
  P_c = Softmax(Q_pool @ K_pool^T / sqrt(d))  # R^{N/b_q × N/b_{kv}}

  # Step 2: 三级分类生成压缩mask M_c (Eq.3)
  for each row i in P_c:
      M_c[i, :] = -1  # 初始化为negligible
      top_kh_indices = TopK(P_c[i,:], k_h=5%)   # critical: top 5%
      M_c[i, top_kh_indices] = 1
      bottom_kl_indices = BottomK(P_c[i,:], k_l=10%)  # negligible: bottom 10%
      # 剩余15%保持marginal (M_c=0)
  ```

  **SLA前向pass（Algorithm 1，单kernel执行）：**
  ```
  # Precompute for linear attention (line 4)
  for j in range(T_n):
      K_phi_j = softmax(K_j)  # activation function φ
      h_j = K_phi_j^T @ V_j           # R^{d × d}
      z_j = rowsum(K_phi_j^T)          # R^{d × 1}

  # Main loop over query blocks (line 7-17)
  for i in range(T_m):  # T_m = N/b_q
      O_i_s = 0  # sparse output accumulator
      H_i = 0; Z_i = 0  # linear attention accumulators
      m_prev = -inf; l_prev = 0  # OnlineSoftmax state

      for j in range(T_n):  # T_n = N/b_{kv}
          if M_c[i,j] == 1:  # CRITICAL → O(N²) sparse FlashAttention
              S_ij = Q_i @ K_j^T / sqrt(d)
              m_curr = max(m_prev, rowmax(S_ij))
              P_ij = exp(S_ij - m_curr)
              l_curr = exp(m_prev - m_curr) * l_prev + rowsum(P_ij)
              O_i_s = diag(exp(m_prev - m_curr)) @ O_i_s + P_ij @ V_j
              m_prev = m_curr; l_prev = l_curr

          elif M_c[i,j] == 0:  # MARGINAL → O(N) linear attention
              H_i += h_j    # 仅矩阵加法（已预计算）
              Z_i += z_j    # 仅向量加法（已预计算）

          # else M_c[i,j] == -1: NEGLIGIBLE → skip

      # Finalize outputs (line 16)
      O_i_s = diag(l_prev)^{-1} @ O_i_s        # sparse output normalization
      Q_phi_i = softmax(Q_i)
      O_i_l = (Q_phi_i @ H_i) / (Q_phi_i @ Z_i)  # linear attention output
      L_i = m_prev + log(l_prev)                # log-sum-exp for backward

  # Final output (Eq.6)
  O = O_s + Proj(O_l)  # Proj: learnable ℝ^d → ℝ^d linear
  ```

  **反向pass关键设计（Algorithm 2）：**
  - 稀疏attention梯度：复用FlashAttention的backward公式，dO^s → dS_ij → dQ_i/dK_j/dV_j，使用D_i^s = rowsum(dO_i^s ⊙ O_i^s)进行softmax梯度计算
  - 线性attention梯度：dO^l → dH_i/dZ_i → dQ_i^φ/dK_j^φ/dV_j，dH_i和dZ_i预计算后对每个marginal块仅需矩阵加法
  - 稀疏和线性组件的梯度融合在同一kernel内执行

  关键设计要点：
  - 压缩mask P_c的分辨率为N/b_q × N/b_{kv}（而非N×N），预测开销可忽略
  - 线性注意力在Wan2.1中仅占full attention的<0.5%，因此marginal块的线性注意力替代是"几乎免费"的
  - 三级分类中仅critical块（5%）执行完整O(N²)计算，marginal块（~85%）用O(N)线性注意力，negligible块（10%）跳过
  - Proj层解决softmax和线性注意力的分布不匹配，使线性注意力作为"learnable compensation"而非直接近似
  - 额外效率优化（Appendix A.3）：Lookup table（sparsity>90%时预处理非零位置）、Pre-aggregation（用减法替代90%加法）、Method of Four Russians（group预计算2^g子集和）

## TileLang: A Composable Tiled Programming Model for AI Systems

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是TileLang支持的低精度/混合精度矩阵乘法（Dequantized Matmul），覆盖多种量化方案：(1) Weight-Only Quantization Matmul: W_INT2 × A_INT8, W_INT4 × A_FP16, W_NF4 × A_FP16；(2) 混合精度FlashAttention: FP16 compute with FP32 accumulation, online softmax rescaling；(3) Multi-Head Latent Attention (MLA): KV cache压缩下的高效attention计算；(4) Linear Attention: Mamba-2 chunk-scan/chunk-state函数的高效实现。

  实验比较：(1) Dequantized Matmul (A100): W_INT2A_INT8 7.65× over cuBLAS-W_FP16A_FP16; W_INT4A_FP16 平均1.04× over Marlin; W_NF4A_FP16 平均1.62× over BitsandBytes。(2) FlashAttention (H100): 1.36× over FA3, 1.41× over Triton, 1.70× over PyTorch。(3) Linear Attention (H100): 平均1.77×和2.10× over Triton (chunk-scan和chunk-state)。(4) MLA (H100): 1075.9× over Torch, 98% of FlashMLA。(5) GEMM (RTX 4090/A100/H100/MI300X): FP16/F32 precision, 0.97-1.10× vs vendor libraries。

- 硬件平台是什么，配置是什么。
  Dequantized Matmul: NVIDIA A100 (80 GB, Ampere, CUDA 12.4)。FlashAttention/Linear Attention/MLA: NVIDIA H100 (80 GB, Hopper, CUDA 12.4)。MLA: AMD Instinct MI300X (192 GB, ROCm 6.1.0)。GEMM: RTX 4090, A100, H100, MI300X。所有平台Ubuntu 20.04。

- 模型是什么。数据集和bench分别是什么。
  算子级benchmark（非end-to-end模型推理），覆盖的算子来自大模型典型workload：
  - GEMM: Table 2的16种矩阵shape（M ∈ [1, 8192], N ∈ [1024, 57344], K ∈ [8192, 57344]），覆盖不同矩阵乘法问题的尺寸
  - FlashAttention: Table 3的5种配置（batch=1, nheads=32, seq_len=512/1024/4096, head_dim=128, causal/non-causal）
  - Linear Attention: Table 4的12种配置（chunk-scan CC0-CC5和chunk-state CT0-CT5, batch=1/64, nheads=64, seq_len=1024/2048/8192, head_dim=64, d_state=128）
  - MLA: 论文未列出具体MLA benchmark形状的详细表格（主要展示性能speedup和代码行数对比）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/tile-ai/tilelang

  算法pipeline —— Dequantized Matmul (FP4_E2M1 × FP16, 对应图17):
  ```
  输入: A[K,M] f16 (activation), B[K,N] u8 packed (4-bit weight, 2 elems per byte)
  输出: C[N,M] f16

  伪代码（Python TileLang程序）:
  @tilelang.jit
  def matmul_fp16_fp4(A: T.Tensor, B: T.Tensor, Ct: T.Tensor):
    with T.Kernel(N//block_N, M//block_M, threads=threads) as (bx, by):
      # 1) 内存分配
      A_shared = T.alloc_shared([block_M, block_K], f16)         # shared mem: activation tile
      B_shared = T.alloc_shared([block_K, block_N//2], u8)       # shared mem: packed weight tile
      B_local  = T.alloc_fragment([block_N, block_K], u8)        # registers: weight
      B_deq    = T.alloc_fragment([block_N, block_K], f16)       # registers: dequantized weight
      Ct_local = T.alloc_fragment([block_N, block_M], f32)       # registers: accumulator

      T.clear(Ct_local)
      # 2) Pipelined主循环
      for k in T.Pipelined(K // block_K, num_stages=2):
        # Step a: 异步加载activation和packed weight到shared memory
        T.copy(A[by*block_M, k*block_K], A_shared)
        T.copy(B[bx*block_N, k*block_K // 2], B_shared)         # B每byte含2个FP4

        # Step b: 从shared memory到register
        T.copy(B_shared, B_local)

        # Step c: 寄存器内解量化 (FP4 → FP16)
        for i, j in T.Parallel(block_N, block_K):
          B_deq[i,j] = _tir_packed_to_unsigned_convert("int", 8)(
            num_bits=4, B_local[i, j//2], j%2, dtype=f16)
          # 从u8字节中提取高/低4-bit (j%2选择) → 转为unsigned int8 → cast to float16

        # Step d: Tensor Core矩阵乘法
        T.gemm(B_deq, A_shared, Ct_local, transpose_B=True)
        # B_deq[block_N, block_K]^T × A_shared[block_M, block_K]^T 累加到 Ct_local[block_N, block_M]

      # 3) 写出结果
      T.copy(Ct_local, Ct[bx*block_N, by*block_M])
  ```

  张量计算流程（单tile）:
  - A activation tile: f16 [block_M, block_K]  global→shared (cp.async)
  - B weight tile:    u8  [block_K, block_N/2]  global→shared (cp.async, pipelined)
  - Dequant: u8 [block_N, block_K] → f16 [block_N, block_K] (in-register, per element)
  - MMA: B_deq^T [block_K, block_N] × A_shared^T [block_K, block_M] → Ct_local [block_N, block_M] (Tensor Core f32 accumulate)
  - Ct_local → Ct[bx*block_N : (bx+1)*block_N, by*block_M : (by+1)*block_M] (register→global, f16 store)

  关键优化点（与Triton的区别）：
  - 权重以packed u8形式直接在shared memory存储，无需shared memory上的layout conversion（Triton需要将解包后的register tensor layout通过shared memory转换到Tensor Core兼容格式）
  - 解量化（dequantize）在寄存器内完成，配合View零开销类型reinterpret
  - Pipeline自动overlap weight/activation loading与computation
  - 对于INT2/INT4/NF4格式，TileLang可由同一程序模板参数化生成，仅需改变num_bits和dtype

  **近似层次匹配说明**：TileLang本身是编译框架/kernel调度工具，但其支持的Dequantized Matmul、FlashAttention、Linear Attention、MLA属于算法pipeline层面的低精度/高效attention算法实现。按最接近层次分类到算法pipeline。

