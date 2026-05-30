# 实验_kernel调度

## SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  基于AutoGPTQ扩展CUDA kernel支持group-wise混合精度推理。核心实现：(1) 按group组织的混合精度weight memory layout：每个group内的元素按相同精度packed为整数，无需额外padding（因为group_size=128是整数类型的倍数，即使3-bit也可充分利用整数字节空间）；(2) 引入extra bit-widths array记录每个group的精度（每个group用2-bit编码，聚合为整数）；(3) CUDA kernel按group逐一dequantize并计算向量点积：一个thread处理一列连续内存的ŵ_int，与block内共享的input activation做点积，累加到结果矩阵；一个warp内32 threads的data access模式保持相似，确保对齐。实验比较GPTQ和SliM-LLM在A800上的推理速度（Token/s）、Weight Memory (WM)和Runtime Memory (RM)，分别在LLaMA-7B/13B/2-7B和LLaMA-2-70B的2/3-bit配置下（Table 5, Table 14）。

- 后端平台是什么，配置是什么。
  单张NVIDIA A800-80GB GPU。CUDA kernel基于AutoGPTQ框架开发，利用CUDA Warp的32-thread单元，当group size=128时确保warp内threads的code structure和data access logic相似。

- 评估性能的软件/脚本是什么。修改了什么。
  评估框架是修改版AutoGPTQ (https://github.com/AutoGPTQ/AutoGPTQ)。修改内容：(1) 量化后输出每个group的scales、zeros和bit-widths信息；(2) AutoGPTQ根据各group精度将weights和zeros pack为整数压缩表示（ŵ_int, ẑ_int），zeros沿channel方向统一精度；(3) 添加extra array存储每个group的2-bit精度标记；(4) 在GPU端实现按group mixed-precision dequantization kernel：逐group处理→thread负责一列连续数据的dequantization→与block共享input activation做vector dot product→通过所有logical blocks迭代完成全linear layer计算。评估原理：测试FP16 baseline和不同bit-width配置下的weight memory (WM)、runtime memory (RM)、perplexity (PPL)和token/s推理速度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/Aaronhuang-778/SliM-LLM，基于AutoGPTQ扩展。
  
  混合精度GPU推理全流程（以LLaMA-7B 2-bit SliM-LLM为例）：
  
  1. **量化阶段输出**：SliM-LLM量化完成后，对每个Linear层输出：
     - scales: FP16, shape (n, m/group_size)
     - zeros: 对应各group的zero point（packable格式）
     - 每个group的bit-width: b_i ∈ {1, 2, 3}
  
  2. **Weight Packing**：
     ```
     for each group g_i (128 elements):
         将128个权重值用b_i-bit量化为整数
         将这些整数沿channel方向pack进32-bit整数（无需padding，因为128可被任意2的幂整除）
     packing后: ŵ_int ∈ R^{m* × n}  (m*是压缩后维度)
     bit-widths array: 每group用2-bit存{1,2,3}标示，聚合为整数
     ```
  
  3. **GPU推理Kernel**：
     ```
     // 对每个Logical Block（覆盖一段连续channel区域）
     for each block:
         加载共享的input activation片段到shared memory
         for each group in this block:
             读取bit-widths[g]确定精度b
             从ŵ_int中按累积偏移读取该group的packed整数
             for each thread (处理一列连续数据):
                 解包(dequantize): w_fp = (ŵ_int_val - z) * scale
                 向量点积: acc += w_fp · activation[shared]
         累加结果写入output matrix对应位置
     ```
  
  4. **性能权衡**：2-bit LLaMA-7B: WM 2.3G（vs GPTQ 2.2G），PPL 14.58（vs GPTQ 152.31），Token/s 61.2（vs GPTQ 83.9）。混合精度因额外bit-widths array和1-bit group的额外计算开销略有降速，但换取了大幅质量提升。

## SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：(1) **Hadamard + Quantization/Dequantization CUDA Kernel Fusion**：将 Hadamard transform 与对称线性 (de)quantization 操作融合为单个 CUDA kernel，利用 shared memory 局部性使融合 kernel 运行速度与单独的量化操作几乎相同；(2) **Buffer Reuse**：利用 Megatron-LM 维持完整 model weights 的特性，复用 model weights buffer 用于权值差值计算，消除额外内存分配；(3) **Operation Pruning**：利用 Hadamard 正交性（H·H=I）和分配律（Σ Hg = HΣg）裁剪冗余的 Hadamard transform，将每轮迭代中的 transform 次数从 6 次（naive）减少到 2 次；(4) **梯度 AlltoAll Pipeline**：将梯度分 chunk 在 intra-node 和 inter-node all-to-all 之间流水线化，利用二者使用不同网络带宽（NVLink vs InfiniBand）的特性实现通信重叠。
  - 实验比较：w/ vs w/o fused Hadamard kernel 的 (de)quantization throughput（Table 5，GB/s）；w/ vs w/o fused Hadamard kernel 的 E2E gradient communication time 和 TFLOPs（Table 4）；TLq-HS vs ULq 的 grad communication time（Table 4）；不同输入大小下的 (de)quantization throughput（Table 5：8MB→2048MB）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100-SXM4-40GB（per-node 4×, NVLink intra-node + 100Gbps Slingshot10 inter-node）和 NVIDIA H800-SXM5-80GB（per-node 8×, NVLink + NVSwitch intra-node + 3.2Tbps InfiniBand inter-node）。
  - GPU 编程模型：CUDA kernel（自定义 Hadamard + quant kernel fusion），NCCL 用于集体通信（all-gather, all-to-all）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估框架：基于 Megatron-LM 的 `pretrain_gpt.py` 训练入口，通过在 Megatron-LM 的 distributed optimizer 和 model forward/backward 中插入量化/反量化操作来评估。
  - 修改内容：
    - `megatron/core/tensor_parallel/random.py` 等文件中增加 Hadamard transform CUDA kernel + (de)quantization CUDA kernel 的融合实现
    - 在 distributed optimizer 的 `all_gather` 和 `reduce_scatter` / `all_to_all` 调用前插入量化步骤
    - 添加权值差值计算逻辑：在 optimizer step 后计算 `w_main - w_model` 差值
    - 梯度 all-to-all pipeline 实现：将梯度分 chunk 在 intra/inter all-to-all 之间流水线化
  - 评估原理：warm-up 20 iterations 后进行 10 iterations 的 E2E throughput 测量（TFLOPs），单独记录 gradient communication time（ms）。Hadamard kernel 融合效果通过 (de)quantization throughput（GB/s）评估：对 8MB→2048MB 的数据进行 quantize/dequantize，测量带/不带 Hadamard 的 throughpout。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源仓库：https://github.com/hanlin-lu/SDP4Bit（Apache-2.0），核心 CUDA kernel 实现在 `megatron/core/tensor_parallel/` 目录下。
  - 评估流程详解：
    **Hadamard Kernel Fusion 评估**（Table 5）：
    ```
    # 输入：FP32/BF16 梯度张量 grad (size: 8MB ~ 2048MB)
    # 输出：INT4 量化数据 + scale factors
    
    # 无 Hadamard fusion (naive):
    # Step 1: Quantize kernel reads grad from global memory → compute scale per group → write INT4 to output
    # Step 2: (separate kernel) Hadamard kernel reads INT4 data → does H·x·H^T → writes back

    # 有 Hadamard fusion (SDP4Bit):
    # Single fused kernel:
    for each thread block (handling one group of 32×32 elements):
        # Load from global memory into shared memory (1 read)
        data_smem = load_global(grad[block_offsets])
        # Hadamard transform in shared memory (memory-bound at 32×32)
        data_smem = H_32 @ data_smem @ H_32.T
        # Quantize in shared memory (no extra global memory traffic)
        s = max(abs(data_smem))
        qdata = round(clip(data_smem, -s, s) / s * 7)
        # Write INT4 output to global memory (1 write)
        store_global(output[block_offsets], qdata, s)
    ```
    关键设计：必须保证 `group_size` 能被 Hadamard matrix size 整除，使得内存在 kernel block 内部保持局部性。论文选择 H=32×32，因为此时 transform 在 GPU 上是 memory-bound，几乎无计算开销，且 32×32 足以平滑梯度 outlier。Table 5 结果显示 w/ vs w/o Hadamard 的 throughput 差异 < 0.3%，证明融合理想。
    
    **AlltoAll Pipeline 评估**（Table 4）：
    梯度通信评估：记录从 backward pass 完成到梯度同步完成的时间。具体流程：
    ```
    # 原始梯度: Float32, collective via reduce-scatter (baseline)
    # SDP4Bit 量化梯度通信:
    t0 = timer()
    # 1. Hadamard + INT8 quantize (fused kernel)
    # 2. Intra-node all-to-all via NVLink (INT8 data)
    # 3. Local reduce
    # 4. Hadamard + INT4 quantize (fused kernel)
    # 5. Inter-node all-to-all via InfiniBand/Slingshot (INT4 data) — 与 step 2 流水线重叠
    # 6. Final reduce + Hadamard inverse
    t1 = timer()
    grad_comm_time = t1 - t0
    ```
    Table 4 结果：TLq-HS grad comm time 45.9ms vs Baseline 379.3ms（8.26× reduction）。ULq 45.0ms vs TLq-HS 45.9ms 几乎相同（因 intra-node 通信带宽高且与 inter-node 重叠）。Fused Hadamard kernel 相比于未融合版本（64.6ms），grad comm time 降低 29%。

## Quamba2 A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Quamba2 实现了完整的 4-bit/8-bit CUDA kernel 栈，覆盖 Mamba1 和 Mamba2 block 的所有关键算子：(1) **4-bit/8-bit matmul kernels**：基于 CUTLASS 实现 W4A8、W4A16、W8A8 三种投影层 kernel，权重按 per-group scaling factors 重排以最大化 Tensor Core 加载吞吐，output scaling factor 融合到 input scaling: s_fused = s_X/s_Y，使得 Ȳ = s_W × s_fused × W̄ × X̄；(2) **W4A8/W4A16 fused matmul-transpose kernels**：专为 Mamba1 block 实现 fused matmul+transpose；(3) **W8A8 causal convolution kernel**：将 causal depthwise conv1d 适配为 8-bit 激活和 8-bit 权重；(4) **8-bit selective scan 和 SSD kernels**：修改 Mamba 原版 selective scan/SSD kernel，接受量化权重、8-bit 激活和对应 scaling factors，加载 8-bit cached states 以减少显存压力（latency 提升约 1.18× at L=1024）；(5) **量化 Fast Hadamard Transform kernel**：在 fast Hadamard transform CUDA kernel 中集成 scaling factor s_y，执行 ȳ^H = (1/s_y) × H_n × y，避免额外量化计算开销；(6) **4-bit/8-bit embedding kernel** 和 **4-bit/8-bit lm_head kernel**：支持 head-to-toe 量化。所有 kernel 针对 auto-regressive 推理场景优化：生成阶段（memory-bound）使用 4-bit weight kernel 减少显存带宽压力，prefill 阶段（compute-bound）使用 8-bit activation kernel 利用 Tensor Core INT8 算力。
  - 实验比较：(a) SSD kernel latency: FP16 vs INT8 activations at L=256/512/1024/2048（Table 3）；(b) W8A8/W4A8/W4A16 end-to-end TPOT/TTFT on A5000 and Orin Nano 8G（Table 5）；(c) batch size scaling TPOT b=1/32/64/128/256 on A5000（Figure 11）；(d) roofline model 分析各 bit-width 的 compute/memory bound 特性（Figure 10）；(e) TTLT vs batch size Pareto 分析（Figure 12）。

- 后端平台是什么，配置是什么。
  - NVIDIA A5000 GPU 24GB（cloud），NVIDIA Orin Nano 8G（edge）。CUDA kernel 基于 CUTLASS（Thakkar et al. 2023）实现。4-bit/8-bit matmul kernel 适配自 Xiao et al. 2023、Frantar et al. 2024（Marlin）、Zhang et al. 2024、LY 2024b/a（CUDA HGEMM/HGEMV）。Fast Hadamard Transform CUDA kernel 集成自 Dao 2024b。Causal Conv1d CUDA kernel 集成自 Dao 2024a。Selective Scan/SSD kernel 修改自 Gu and Dao 2024 / Dao and Gu 2024 官方实现。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估框架：基于 vLLM（Kwon et al. 2023）进行 serving latency 测量，集成 Quamba2 自定义量化 kernel。修改内容：(a) 将 vLLM 的投影层替换为 Quamba2 的 W4A8/W4A16/W8A8 matmul kernel；(b) SSD/selective scan kernel 修改为接受 8-bit activations + scaling factors；(c) causal conv1d kernel 修改为 W8A8；(d) embedding/lm_head 替换为 4-bit/8-bit kernel；(e) 集成量化 fast Hadamard transform。Latency profiling：warm-up iterations + 100 iterations 平均，逐 operator 记录 latency。Model size profiling：统计所有量化参数和 buffers 的显存占用。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/enyac-group/Quamba（论文声明 will be released）
  - Kernel 输入到性能输出全过程（以 W4A8 Mamba2 SSD block 为例）：
    1. **Input projection (W4A8)**：输入 u_t ∈ R^D (FP16) → per-group 量化到 8-bit → ū_t。4-bit weights W̄_in 和 per-group scales s_W 预加载到 shared memory → Tensor Core 执行 INT8 matmul: (x̄_t, B̄_t, C̄_t, Δ̄_t) = dequant(W̄_in, s_W) @ ū_t。Output scaling 融合: s_fused = s_X/s_Y，kernel 输出 Ȳ（INT8）+ s_fused（FP16）。
    2. **Online Hadamard + Sort-and-cluster**：x̄_t^H = FWHT_kernel(x̄_t)（in-place Hadamard transform, O(d log d)）。按 pre-computed cluster indices 重排 channel → 分组 → 各组 quantize: x̄_t^s[c] = clamp(round(x_t^H[c]/s_{m,n}), -127, 127)。
    3. **Causal Conv1d (W8A8)**：x̄_t^s 与 8-bit conv weight → conv1d_kernel 执行 INT8 depthwise conv: y_conv[t,c] = Σ_{k=0}^{K-1} W̄_conv[c,k] × x̄^s[t-k,c]。
    4. **SSD scan (8-bit states)**：从 HBM 加载 8-bit h_{t-1}（cached state）→ 加载 Ā_t, B̄_t^g, C̄_t^g（8-bit）→ SSD_kernel 执行: h_t = Ā_t ⊙ h_{t-1} + B̄_t^g ⊗ x̄_t^s, y_ssd = C̄_t^g ⊙ h_t → 写回 8-bit h_t 到 HBM 作为下步 cached state。8-bit memory traffic 降低约 2× vs FP16。
    5. **Output projection (W4A8)**：quantize y_ssd ⊙ SiLU(z_t) → ȳ → FWHT → ȳ^H = (1/s_y) × H_n × ȳ（Hadamard kernel 内联 scaling）。加载 4-bit W̄_out → dequant + matmul → ȳ_out。
    6. **性能输出**：kernel profiler 记录每个 operator 的 GPU time (ms) → 累加得 TTFT（prefill 1024 tokens）和 TPOT（generation per token）。Memory profiler 记录 HBM 占用：weights（4-bit）+ cached states（8-bit）+ activations + scaling factors + buffers。

## QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QeRL 论文的核心 kernel 贡献是使用 **Marlin kernel**（Frantar et al., 2024）加速 NVFP4 量化权重的推理。Marlin 是一个面向 auto-regressive 大语言模型推理的 mixed-precision kernel，原生支持 FP16×INT4 操作。QeRL 将 Marlin kernel 适配到 NVFP4×BF16 操作：在 rollout 阶段，policy model 的量化权重以 NVFP4 格式存储（仅 5.9GB for 7B model vs 15.2GB BF16），通过 Marlin kernel 执行快速的 NVFP4 dequant + BF16 matrix multiplication，大幅加速 rollout 生成。AQN 噪声注入被设计为融入 RMSNorm scale 参数，避免破坏 NVFP4×BF16 乘法 kernel 的兼容性，无需额外矩阵乘法开销。kernel 的加速效果随模型规模增大而增强：32B 模型 rollout 达到 2.0× speedup。
  - 实验比较：(a) rollout 吞吐量对比：NVFP4(Marlin) vs BF16 baseline，batch=2/4/8；(b) end-to-end GRPO 训练速度对比：QeRL vs LoRA vs QLoRA；(c) 不同 LoRA rank 下的推理吞吐量；(d) 启用/不启用 gradient checkpointing 的端到端延迟；(e) 模型大小对比（GPU 显存占用：7B 5.9GB vs 15.2GB BF16）。

- 后端平台是什么，配置是什么。
  - NVIDIA H100 80GB GPU。使用 vLLM 引擎进行 rollout。环境：CUDA≥12.4.1，需要支持 NVFP4 的 GPU（如 H100, B100, RTX 5090）。Marlin kernel 依赖 CUDA 的 mixed-precision 矩阵乘法原语。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：vLLM 引擎（用于 rollout 推理吞吐量测量），GRPO/DAPO 训练脚本（端到端速度测量）。论文修改：将 Marlin kernel 适配到 NVFP4 格式（原版 Marlin 针对 INT4/FP16 设计），并在 kernel 前向过程中保持 NVFP4×BF16 操作路径不被 AQN 噪声注入破坏（通过将噪声融入 RMSNorm 而非直接在量化权重上做加法实现）。speedup 测试基于训练前 30 step 的平均速度，rollout throughput 在 batch=1 设置下测试。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/NVlabs/QeRL (Apache 2.0)。Marlin kernel 原版开源：https://github.com/IST-DASLab/marlin。
  - kernel 输入到输出全流程：
    1. **输入**：NVFP4 量化权重 \tilde{W}（4-bit）+ FP32 全局 scale S_FP32 + FP8(E4M3) block-wise scale S_E4M3（block size=16）+ BF16 输入激活 X
    2. **kernel 内**：(a) 从 GPU global memory 加载 4-bit packed weights 到 shared memory；(b) 对每个 block 内的 16 个元素执行 dequant：ŵ = s_FP32 × s_E4M3[block_idx] × dequant_4bit_to_fp16(w̃_block)；(c) 执行 BF16 matrix multiplication: output = X · \hat{W}^T；(d) 加 LoRA adapter 输出: output += X · (BA)^T（可选的 fused LoRA compute）
    3. **输出**：BF16 激活张量
    4. **评估原理**：vLLM 引擎在 rollout 阶段调用 Marlin-accelerated 的 NVFP4 线性层，测量每个 forward pass 的 wall clock time 和 tokens/s。通过固定输入长度=256 tokens、最大生成长度=2048 tokens、控制 vLLM GPU memory utilization 来公平对比 BF16 vs NVFP4 的吞吐量差异。

## ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 scaled pairwise rotation transform 编写了**单一 fused CUDA kernel**，在推理时对激活张量应用逆变换 T^{-1}。kernel 在三个层次上完全并行化：(1) **token 维度**——沿 batch×seq_len 维度并行；(2) **channel group**——不同 CUDA block 处理不同的 channel group（group size=128）；(3) **rotation pair**——同一 group 内不同 CUDA thread 处理不同 Givens 旋转对。所有 Givens 旋转彼此独立（无数据依赖），因此同一 rotation 内的所有 pair 可**无同步**（synchronization-free）并发执行。由于 group size 小（128），激活张量可放入 on-chip shared memory，旋转参数（pair indices 和 angles）可放入寄存器，多个独立旋转（K=8）在一次内存加载后即可融合执行，无需多次 global memory 访问。
  - 实验比较：ParoQuant fused kernel vs **Fast Hadamard Transform** (Dao, 2024) 在不同 channel 维度上的 speedup (RTX A6000)。与 AWQ（无额外 transform kernel）、QTIP（Hadamard transform kernel）对比端到端 decode 吞吐（tokens/s）。在 RTX A6000、RTX 6000 Ada、RTX 4090 上的完整吞吐表（Table A4）。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX A6000 (48GB, Ampere)、NVIDIA RTX 6000 Ada (48GB, Ada Lovelace)、NVIDIA RTX 4090 (24GB, Ada Lovelace)。推理测速使用 PyTorch 2.6.0 + torch.compile (max-autotune) + CUDA Graphs。W4A16 GEMM kernel 使用 AWQ 仓库的实现。

- 评估性能的软件/脚本是什么。修改了什么。
  - **Transformers library** (HuggingFace)：仅替换原始 FP16 线性层为量化线性层实现。AWQ、QTIP、ParoQuant 分别使用各自的官方开源仓库提供的量化层实现。
  - **Fast Hadamard Transform** (https://github.com/Dao-AILab/fast-hadamard-transform)：作为 Hadamard-based 方法的 transform kernel baseline。
  - ParoQuant 修改/新增：实现了 fused CUDA kernel 将 channel-wise scaling 逆变换和 K 个 independent rotations 融合为单次 kernel 调用。Kernel 采用三步并行策略：(1) token-level grid stride loop、(2) group-level CUDA block 分配、(3) pair-level CUDA thread 处理。旋转参数（2K 个 float per group × group 内 pair 数）预加载到寄存器，激活 tile 加载到 shared memory，8 个旋转在一个 kernel 内融合完成。
  - 评估方式：decode throughput (tokens/s)，batch size=1，测量端到端 token 生成速度。各方法使用统一的 Transformers 推理框架，仅修改量化层的权重变换和反量化代码。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/z-lab/paroquant（MIT License），PyPI `pip install paroquant[vllm]`。W4A16 GEMM kernel 复用 AWQ 开源仓库。
  - Kernel 输入到性能输出的全过程：
    1. **输入**：INT4 量化权重 W_q ∈ ℤ^{C_in/2 × C_out}（packed，每 2 个 4-bit 值打包为 1 byte）+ 量化参数 (s ∈ R^{C_out}, z ∈ ℤ^{C_out})；FP16 激活 X ∈ R^{T×C_in}（T=tokens）；旋转参数：每组 128 channels 存储 K=8 个 independent rotation 的 pair indices (2×64 个 uint8 per rotation) 和 angles (64 个 float per rotation)。
    2. **Kernel Launch**：对每个 linear 层，先调用 ParoQuant transform kernel 对激活 X 应用逆变换 T^{-1}，再调用 AWQ W4A16 GEMM kernel 执行 INT4 矩阵乘法。
    3. **Transform Kernel (Global→Shared Memory)**：按 token 和 group 分配 grid/blocks。每个 thread block 负责一个 (token, group) pair，将对应激活片 X[token, group_start:group_start+128]（128 × FP16 = 256 bytes）从 global memory load 到 shared memory。旋转参数（pair indices + angles）预取到寄存器。
    4. **Transform Kernel (Givens 旋转计算)**：按 pair 分配 threads。对每个 rotation r=1..K：
       - 读取 pair (i, j)、angle θ
       - `x_i' = cosθ * x_i - sinθ * x_j`
       - `x_j' = sinθ * x_i + cosθ * x_j`
       - 所有 pairs 并发执行（无依赖，无同步）。先做 8 次旋转，最后乘以 1/α 完成逆缩放。
    5. **Transform Kernel (Shared→Global)**：变换后的激活 tile 写回 global memory，作为后续 GEMM kernel 的输入。
    6. **GEMM Kernel**：AWQ 的 W4A16 GEMM kernel 读取 packed INT4 权重 + FP16 激活 → dequantize → FP16 matmul → FP16 output，写回 global memory。
    7. **性能测量**：CUDA Events 计时，包含 transform kernel + GEMM kernel 的端到端时间。计算 throughput = num_tokens / total_time (tokens/s)。
  - 关键优势：(1) Group 内 shared memory 常驻——128 个 FP16 仅 256 bytes，远小于 typical shared memory (48-100KB)，8 次旋转全部在 shared memory 上完成；(2) 无同步开销——同一 rotation 内 pairs 互不重叠，threads 间无数据竞争；(3) 随 channel 维度扩大，相比 Hadamard transform（有全局依赖）加速比递增（Figure 4）。
  - 结果：ParoQuant decode 吞吐仅比 AWQ 慢约 10%（如 Qwen3-4B: 160 vs 176 tokens/s），比 QTIP 快 15%-30%（如 Qwen3-4B: 160 vs 117 tokens/s）。



## Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：基于 **CUTLASS** 库实现 **INT4 2:4 semi-structured sparse GEMM kernel**，利用 NVIDIA Ampere/Hopper 架构的硬件原生支持（如 Sparse Tensor Cores）。对 joint OBR 压缩后的 LLM（W4A4KV4 + 50% sparsity），权重 W 使用 INT4 量化 + 2:4 结构化稀疏，激活 X 使用 INT4 量化，执行 batched INT4 sparse GEMM。
  - 实验比较：INT4 2:4 Sparse GEMM vs. INT4 Dense GEMM vs. FP16 Dense GEMM。在单张 NVIDIA A100-SXM4-80GB 上评测不同序列长度（128/512/1024/2048/4096）下的 **latency (ms)**、**theoretical FLOPs (GFLOPs)** 和 **TOPS**。权重矩阵尺寸为 W ∈ ℝ^{4096×4096}，激活为 X ∈ ℝ^{32×seq_len×4096}，模拟典型 LLM 推理场景。

- 后端平台是什么，配置是什么。
  - NVIDIA A100-SXM4-80GB GPU（Ampere 架构），80GB HBM2e，支持第三代 Tensor Cores 和 2:4 structured sparsity。

- 评估性能的软件/脚本是什么。修改了什么。
  - **CUTLASS** (https://github.com/NVIDIA/cutlass)：NVIDIA 开源 CUDA C++ 模板库，用于实现高性能 GEMM kernel。论文使用 CUTLASS 实现 INT4 sparse GEMM，将 2:4 结构化稀疏与 INT4 量化结合到单个 kernel 中。修改/新增了支持 INT4 数据类型 + 2:4 sparse 模式的 GEMM kernel。
  - 评估方式：测量给定 batch=32 下的 wall-clock latency；计算理论 FLOPs（2×M×N×K - 2×M×N×K×sparsity）；计算 TOPS (Tera Operations Per Second) = FLOPs / latency。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：OBR 算法代码 https://github.com/csguoh/OBR；CUTLASS kernel 库 https://github.com/NVIDIA/cutlass。
  - Kernel 输入到性能输出的全过程：
    1. **输入**：经 OBR 压缩后的 INT4 量化权重 Ŵ ∈ ℤ^{M×K}（存储为 packed INT4，每 2 个 4-bit 值打包为 1 byte），2:4 稀疏 mask M（每 4 个连续元素中恰好 2 个非零）；INT4 量化激活 X̂ ∈ ℤ^{K×N}；对应的量化 scale/zero-point。
    2. **Kernel 编译**：使用 CUTLASS 模板定义 `Gemm<Int4Sparse>`，指定 warp tile size、threadblock shape、MMA (Matrix Multiply-Accumulate) 指令（利用 Sparse Tensor Cores 的 `mma.sp.sync` 指令，每次取 4 个值中自动跳过 2 个零）。
    3. **Global → Shared Memory**：将 packed INT4 Ŵ 和 sparse metadata 从 global memory load 到 shared memory；将 FP16/INT4 X̂ load 到 shared memory。利用 2:4 结构化稀疏将权重访存量减半。
    4. **MMA 计算 (Tensor Core)**：warp-level Tensor Core 指令执行 `D = A * B + C`，其中 A (激活, INT4 → FP16 反量化), B (权重, INT4 2:4 sparse, 利用硬件跳过零值), C/D (累加器, FP32)。2:4 稀疏使有效计算减少 2×，内存带宽需求降低 2×。
    5. **输出**：FP32 累加器结果 → 写回 global memory 作为输出矩阵 Y ∈ ℝ^{M×N}。
    6. **性能测量**：用 CUDA Events (cudaEventRecord) 计时，排除 kernel launch overhead。测量纯 GEMM 计算时间，计算 TOPS = (2 × M × N × K × (1 - sparsity)) / latency。
  - 结果：在 seq_len=4096 时，INT4 2:4 Sparse GEMM 比 FP16 Dense GEMM 快 **5.9×**，比 INT4 Dense GEMM 快 **1.4×**；理论 FLOPs 比 INT4 Dense 减半（因 50% 稀疏）；GPU 资源饱和（seq_len > 2048）时 TOPS 更高。

 (Learn at Test Time): RNNs with Expressive Hidden States

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 TTT-Linear 和 TTT-MLP 编写**自定义 GPU 推理 kernel**，分别针对 forward（prefill，可并行化）和 generate（decode，序列化）两种模式。训练 kernel 未实现（因 TPU 上使用 JAX，不兼容 GPU kernel）。
    - **Forward (prefill) kernel**：使用 dual form，将内循环梯度计算转化为 matmul 操作以最大化 TensorCore 利用率。dual form 等价于 primal form 的输出，但通过 `W_b = W_0 - 2η(W_0X̂ - Y)X̂^T` 和 `Z = W_0X̄ - 2η(W_0X̂ - Y)mask(X̂^TX̄)` 将所有操作转为 matmul，训练速度比 primal form 快 **5× 以上**（TPU 上）。
    - **Generate (decode) kernel**：使用 primal form，因为逐个 token 生成本质上是序列化的，每次只需对单个 token 计算梯度 `G_t = ∇ℓ(W_{t-1}; x_t)` 并更新 `W_t = W_{t-1} - ηG_t`，无需 dual form 的批处理优势。
  - 实验比较：在 NVIDIA A100 80G PCIe 上评测 1.3B 模型的 forward latency（prompt processing）和 generate latency（token 解码），对比 Transformer（vLLM serving）、Mamba、TTT-Linear、TTT-MLP。随上下文长度从 1k 到 32k 增加，Transformer latency 线性增长，TTT 和 Mamba 基本恒定。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80G HBM，PCIe 连接
  - 训练平台：TPU v5e-256 pod（JAX 实现，与 GPU kernel 路径分开）

- 评估性能的软件/脚本是什么。修改了什么。
  - Transformer baseline：使用 **vLLM**（state-of-the-art LLM serving 系统），而非 HuggingFace Transformer，确保 baseline 具有竞争力。
  - Mamba baseline：使用作者公开的 PyTorch+Triton+CUDA 代码。
  - TTT 方法：自写 GPU inference kernel，**未实现训练 kernel**（训练在 TPU 上用 JAX 完成）。修改/新增了 forward (prefill) 的 dual form kernel 和 generate (decode) 的 primal form kernel。
  - 评估方式：测量单次 forward（给定 prompt 长度）和单 token generate 的 wall-clock latency。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：JAX 代码 https://github.com/test-time-training/ttt-lm-jax，PyTorch 代码 https://github.com/test-time-training/ttt-lm-pytorch。GPU inference kernel 位于 PyTorch 仓库中。
  - 评估原理与 kernel 执行流程：
    1. **输入**：输入序列 X ∈ R^{d×T}（T 为 prompt 长度，forward 模式）或单个 token x_t ∈ R^d（generate 模式）
    2. **Forward (prefill) kernel 流程**：
       - 按 mini-batch size b=16 将 T 个 token 分块
       - 对每个 mini-batch，执行 dual form：
         a. 投影：X̂ = θ_K @ X_block, Y = θ_V @ X_block, X̄ = θ_Q @ X_block（matmul, 利用 TensorCore）
         b. 计算 W_end：W_b = W_0 - 2η(W_0 @ X̂ - Y) @ X̂^T（matmul）
         c. 计算输出：Z = W_0 @ X̄ - 2η * (W_0 @ X̂ - Y) * mask(X̂^T @ X̄)（matmul + 上三角 mask）
       - 将 W_b 传入下一个 mini-batch 作为新的 W_0
    3. **Generate (decode) kernel 流程**：
       - 对单个新 token x_t，使用 primal form：
         a. 投影：x̂ = θ_K x_t, y = θ_V x_t, x̄ = θ_Q x_t（向量操作）
         b. 梯度：G = 2(W_current x̂ - y) x̂^T（外积）
         c. 更新：W_new = W_current - η(x_t) · G
         d. 输出：z = W_new @ x̄（矩阵-向量乘）
    4. **输出**：output tokens Z ∈ R^{d×T}（forward）或 z_t ∈ R^d（generate），性能指标为 wall-clock latency（ms）

## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - KIVI 的 **System Support** 部分提供了 GPU 上的硬件友好实现，包含两个关键 kernel：
    1. **Q_MatMul（Fused Dequantization + Matrix Multiplication）**：使用 CUDA 实现，将 KV cache 的反量化过程与矩阵乘法在 tiling 级别融合，避免将反量化后的全精度数据写回全局内存。在计算 attention score 和 attention output 时使用，减少内存访问开销。
    2. **Group-wise Quantization Kernel**：使用 Triton 实现，执行 group-wise round-to-nearest 量化（per-channel 用于 key，per-token 用于 value）。支持 streaming 场景下将新到达的 KV tensor 动态量化并追加到已有 quantized cache。
    3. **Tiled Matrix Multiplication**：将 grouped quantized 部分和 residual FP16 部分的矩阵乘法分块执行后 Concat。
  - 实验比较：
    - KIVI（residual length 32/128）vs FP16 baseline 在 Llama-2-7B 上的峰值内存和吞吐量（ShareGPT workload）
    - KIVI 可使 batch size 增大 4×，吞吐量提升 2.35× ∼ 3.47×

- 后端平台是什么，配置是什么。
  - GPU：单张 NVIDIA A100 GPU（80GB）
  - 计算后端：CUDA（用于 fused dequantization + MatMul kernel）、Triton（用于 group-wise quantization kernel）

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **Hugging Face Transformers** 代码库实现 KIVI 算法
  - 使用 **LM-Eval** 框架评估准确率（CoQA, TruthfulQA, GSM8K）
  - 使用 **LongBench** 评估长上下文性能
  - 使用 **ShareGPT** 真实对话数据合成 workload，参考 **vLLM** 的方式评估吞吐量和内存
  - 修改内容：
    - 在 attention 层中插入 KIVI 的量化/反量化逻辑
    - 实现了 CUDA fused dequantization+MatMul kernel（Q_MatMul）
    - 实现了 Triton group-wise quantization kernel
    - 修改了 KV cache 的数据结构为 grouped quantized + residual FP16

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/jy-yuan/KIVI
  - **Kernel 执行流程详解**：

  **Q_MatMul（Fused Dequantization + MatMul）**：
  ```
  输入: t_Q ∈ R^{1×d} (query), Q(X_K_g) ∈ int2 (quantized grouped key)
  过程:
    1. 将 t_Q 按 tile 加载到 GPU SRAM
    2. 将对应的 Q(X_K_g) tile 加载到 SRAM
    3. 在 SRAM 中即时反量化 int2 → FP16：
       X_K_g' = Q(X_K_g) × s_K + z_K  // s_K是scaling factor, z_K是zero-point
    4. 直接在 SRAM 中计算 tile 的矩阵乘法: A_tile = t_Q_tile × X_K_g'_tile^T
    5. 输出 A_tile 写回全局内存
  输出: A_g (attention logits for grouped part)
  ```
  避免将 FP16 大小的 X_K_g' 整体写回全局内存，节省 HBM 带宽。

  **Group-wise Quantization Kernel**：
  ```
  输入: X ∈ R^{l×d} (float16), G=32 (group size), dim (quantization axis)
  过程 (dim=channel 时):
    1. 将 X 沿 channel 维度分成 d/G 个 group
    2. 每个 group 包含 G 个连续的 channel
    3. 对每个 group 计算: min, max → s_X = (max-min)/3, z_X = min
    4. 执行量化: Q(X) = round((X - z_X) / s_X)，clamp 到 [0, 3] (2bit)
    5. 将 Q(X), s_X, z_X 存储
  输出: Q(X) ∈ int2, 每 G 个元素共享一组 (s_X, z_X)
  ```

  **Tiled Attention 全流程**：
  ```
  KV cache = {Q(X_K_g): int2, X_K_r: FP16, Q(X_V_g): int2, X_V_r: FP16}
  
  1. Q_MatMul(t_Q, Q(X_K_g)) → A_g    // fused dequant+matmul, grouped部分
  2. t_Q × X_K_r^T → A_r               // 标准 matmul, residual FP16部分
  3. A = Concat([A_g, A_r])            // 拼接 attention logits
  4. A_g_sm = Softmax(A)[:-R], A_r_sm = Softmax(A)[-R:]
  5. Q_MatMul(A_g_sm, Q(X_V_g)) → t_O_g  // fused dequant+matmul
  6. A_r_sm × X_V_r → t_O_r              // 标准 matmul
  7. t_O = t_O_g + t_O_r
  ```

## FlatQuant: Flatness Matters for LLM Quantization

- **属于kernel调度/运行时计算的实现是什么？实验比较什么？**
  设计了融合 kernel，将仿射变换 Q(P₁^T ×₁ X̃ ×₂ P₂) 在单个 OpenAI Triton kernel 中完成，避免中间结果写回全局内存。kernel 设计动机：仿射变换使用 Kronecker 乘积的两个轻量矩阵（如 64×64），计算强度低，属于 memory-bound 操作；量化也是 memory-bound。融合后在单个 thread block 内完成：加载 P₁、P₂ 到 SRAM → slicing tile block X̄ ∈ R^{n₁×n₂} → 计算 P₁ X̄ P₂ → 即时量化 → 写回全局内存。实验对比了融合前后的 prefill/decoding latency 加速比（不同 hidden dimension 4096/5120/8192/11008/13824/14336，batch size 1-64）。同时集成了 CUTLASS INT4 matmul kernel 和 FlashInfer KV cache quantization。

  三种 SRAM 容量场景的 kernel 设计：
  - **默认设计**（图 8a）：共享内存足够容纳 P₁、P₂、X̄ 及其中间结果，单 kernel 完成所有操作
  - **Corner Case 1**（图 8b）：n 和 n₁ 过大，对 P₁ 非规约维度 tiling，分两个 kernel（先 P₁X̄P₂，再独立量化 kernel）
  - **Corner Case 2**（图 8c）：n 和 n₂ 极大，分三步（先 P₁^T X̄ 写回全局内存释放 SRAM，再乘以 P₂ 并即时量化）

- **后端平台是什么，配置是什么。**
  NVIDIA RTX 3090 GPU。SRAM 大小决定 kernel 设计策略，hidden_dim ≤ 14336 且 n₁,n₂ ≤ 128 时使用默认设计。评测覆盖 hidden dimensions：4096（LLaMA-2-7B）、5120、8192（LLaMA-3-8B）、11008（LLaMA-2-7B FFN intermediate）、13824、14336。Corner case 测试使用 hidden_dim=28762。

- **评估性能的软件/脚本是什么。修改了什么。**
  评估软件：OpenAI Triton（编写融合 kernel）、CUTLASS（INT4 矩阵乘法）、FlashInfer（KV cache 量化）、PyTorch（baseline 对比）。修改内容：
  1. **融合 kernel 实现**：将原本分离的 3 个操作（加载 → 仿射变换 → 写回 → 加载 → 量化 → 写回）融合为单 kernel 内流水线
  2. **SRAM tiling 策略**：根据 P₁(n₁×n₁)、X̄(n₁×n₂)、P₂(n₂×n₂) 的总 FP16 字节数（×2）判断是否超过 shared memory per block（m），自动选择默认/Corner Case 1/Corner Case 2 路径

- **开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。**
  开源代码：https://github.com/ruikangliu/FlatQuant

  **Kernel 执行全过程（以默认设计，hidden_dim=4096, n₁=n₂=64, RTX 3090, batch=1, prefill）**：

  **输入**：激活 X ∈ R^{k×4096}（k=tokens），P₁ ∈ R^{64×64}，P₂ ∈ R^{64×64}（FP16），量化参数（scale, zero-point）

  **Kernel Launch 配置**：Grid = (k, 1, 1)，每个 thread block 处理一个 token（X 的一个 64×64 tile）

  **Thread Block 内执行流程**：
  ```
  // Step 1: 加载到 SRAM
  P₁_sram ← load(P₁)                    // 64×64×2B = 8KB
  P₂_sram ← load(P₂)                    // 64×64×2B = 8KB
  X̄_sram ← load(X[token_i])             // 64×64×2B = 8KB (tile from X̃)

  // Step 2: 仿射变换 (在 SRAM 内)
  X̄' = P₁_sram^T @ X̄_sram @ P₂_sram     // 64×64 matmul × 2

  // Step 3: 即时量化 (在 SRAM 内)
  scale = max(|X̄'|) / (2^{b-1} - 1)     // per-token symmetric
  X̄'_q = round(X̄' / scale)              // quantize to INT4
  X̄'_q = clamp(X̄'_q, -2^{b-1}+1, 2^{b-1}-1)

  // Step 4: 写回全局内存
  store(X̄'_q, scale) → global memory    // INT4 + FP16 scale
  ```

  **Profiling 结果**（Table 6, hidden_dim=4096, batch=1, seq_len=2048 prefill / decode_1token）：
  - 无融合：prefill 0.1956ms, decode 0.0184ms
  - 有融合：prefill 0.0625ms, decode 0.0082ms
  - 加速比：prefill 3.13×, decode 2.25×

  **端到端加速**（LLaMA-2-7B, batch=64, prefill 2048 tokens + decode 256 tokens）：
  - vs FP16：prefill 2.30×, decode 1.76×
  - 与纯 INT4 量化（无变换）相比，仅损失 0.07× 加速比


## YOCO (You Only Cache Once): Gated Retention Triton Kernel

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了 gated retention 的 Triton kernel，支持三种计算范式：(1) **Parallel** 模式用于训练（充分利用 GPU 并行）；(2) **Recurrent** 模式用于自回归生成（O(1) 常量 KV 内存）；(3) **Chunkwise Recurrent** 模式用于 prefill（结合并行和 recurrent 优势，chunk 内并行 + chunk 间 recurrent，降低 FLOPs 和迭代次数）。prefill 阶段使用 chunkwise（chunk_size=256），生成阶段切换到 recurrent。基于 FLA (Flash-Linear-Attention) 库实现。同时 baseline Transformer 使用了 Flash-Decoding 和 kernel fusion 进行公平比较。实验比较了 YOCO_gRet 与优化 Transformer（GQA + Flash-Decoding + kernel fusion）在 H100-80GB 上的推理性能（GPU memory, prefill latency, throughput），序列长度从 32K 到 1M。

- 后端平台是什么，配置是什么。
  NVIDIA H100-80GB GPU。Triton kernel 基于 FLA 库（https://github.com/sustcsonglin/flash-linear-attention）实现。Baseline Transformer 使用 Flash-Decoding 做优化 attention kernel。

- 评估性能的软件/脚本是什么。修改了什么。
  评估方式：profiling 测量 GPU memory breakdown（model weights + intermediate activation + KV cache）、prefill latency（编码输入 prompt 的时间）、throughput（tokens/s，包含 prefill + generation）。
  修改内容（Gated Retention Triton kernel 设计）：
  1. **Chunkwise Recurrent Kernel（prefill 用）**：
     - 将序列分为 chunk_size=256 的块
     - Inner-Chunk 部分用 parallel 计算（chunk 内 QK^T ⊙ D + V，利用 Tensor Core）
     - Cross-Chunk 部分用 recurrent state R_i 传递跨 chunk 信息
     - 输出 = (Q_{[i]} K_{[i]}^T ⊙ D_{[i]}) V_{[i]} + (Q_{[i]} R_{i-1}) ⊙ β_{[i]}
     - 比 fully parallel 节省 FLOPs，比 fully recurrent 减少迭代轮次
  2. **Recurrent Kernel（decode 用）**：
     - S_n = γ_n · S_{n-1} + K_n^T V_n（state 更新，O(d²) per step）
     - O_n = Q_n · S_n（vector-matrix multiply）
     - 仅维护 single state matrix S ∈ R^{d×d}，不存储 per-token KV cache
  3. **数据依赖门控优化**：γ 使用 head-wise decay（而非 element-wise），使计算可充分利用 NVIDIA Tensor Core

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://aka.ms/YOCO。Triton kernel 基于 FLA 库 (https://github.com/sustcsonglin/flash-linear-attention)。

  **Kernel 执行全过程（以 YOCO_gRet 3B, H100-80GB, 512K 输入 + 1024 token 生成 为例）**：

  **阶段 1: Prefill（Chunkwise Recurrent Kernel）**
  - 输入：embedding X ∈ R^{512K×3072}，权重 W_Q, W_K, W_V, W_γ ∈ R^{3072×3072}
  - Step 1a: 投影 Q = XW_Q ⊙ Θ, K = XW_K ⊙ Θ̄, V = XW_V（利用 Tensor Core GEMM）
  - Step 1b: 计算 gate γ = sigmoid(XW_γ)^{1/τ}（head-wise, 24 heads）
  - Step 1c: 分 chunk（256 tokens/chunk, 2000 chunks），对每个 chunk：
    - Inner-Chunk: (Q_c K_c^T ⊙ D_c) V_c（parallel, Tensor Core matmul）
    - Cross-Chunk: (Q_c R_{c-1}) ⊙ β_c（state R 传递，O(d²) 计算）
    - 更新 state: R_c = K_c^T (V_c ⊙ value_decay) + chunk_decay · R_{c-1}
  - Step 1d: GroupNorm + swish gate 输出
  - 输出：Self-Decoder 最终 activation M = X^{L/2} ∈ R^{512K×3072}
  - 关键：此时 prefill 可提前退出，无需进入 Cross-Decoder

  **阶段 2: KV Cache 生成（单次）**
  - 输入：M（Self-Decoder 最终输出）
  - K̂ = LN(M) W_K, V̂ = LN(M) W_V
  - 存储：K̂, V̂ ∈ R^{512K×3072}（单层全局 KV cache，约 512K×3072×2×2bytes = 6.3GB, 3B 模型）

  **阶段 3: Decode（Recurrent Kernel for Self-Decoder, Standard Attention for Cross-Decoder）**
  - 每个新 token：
    - Self-Decoder(recurrent): S_n = γ_n · S_{n-1} + K_n^T V_n → O_n = Q_n · S_n（极小内存，仅维护 S）
    - Cross-Decoder: Q̂ = XW_Q → Attention(Q̂, K̂, V̂)（标准 Flash-Decoding kernel, 复用全局 cache）

  **性能输出（H100-80GB, 3B model）：**
  - GPU Memory: 1M context 仅 12.4GB（Transformer 需 9.4× more）；32K context 节省 ~2×
  - KV Cache per token: 128K tokens 仅需 1GB KV cache（65B model 时 Transformer 仅支持 1.6K tokens）
  - Prefill Latency: 512K 从 180s（Transformer）降至 <6s（71.8× 加速 for 1M, 2.87× for 32K）
  - Throughput: 512K queries 43.1 token/s vs Transformer 4.5 token/s（9.6× 加速）
  - 加速来源：(a) prefill 仅需 L/2 层 + 高效 attention；(b) KV cache 内存节省允许更大 batch size

## DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  定制 CUDA kernel 实现了 W4A8 GEMM，集成量化（quantization）、权重 bit-shifting、GEMM 和反量化（dequantization）的融合操作。Section E 中展示了该自定义 kernel 与 PyTorch FP32 GEMM 的延迟对比（Figure 8），在 M=3072 时达到 5.17× 加速。bit-shifting 引入的开销极小（仅在权重加载后执行 Ŵ^{shifted}_{kj} = Ŵ_{kj} ≪ δ_k），且 PTS 仅应用于 skip connection 层（网络总层数的小子集），对整体延迟影响微乎其微。

- 后端平台是什么，配置是什么。
  论文未明确说明 GPU 型号。实验基于 PyTorch + 自定义 CUDA kernel。PTS 的 bit-shift 操作在 GPU kernel 执行时于权重加载后立即完成，验证了 2 的幂次缩放在硬件上的高效性。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：自定义 CUDA kernel 与 PyTorch FP32 GEMM 直接对比延迟。修改内容：kernel 将量化（MinMax Q）→ bit-shift on weight → INT GEMM → dequantization 融合为单次 kernel launch，消除中间结果的 DRAM 写入。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未说明 kernel 代码是否开源（https://github.com/LeeDongYeun/dmq 主要包含量化算法代码）。kernel 执行流程：

  **输入**：INT4 packed 权重 W̃ ∈ Z^{Cin×Cout}，INT8 激活 X̃ ∈ Z^{B×Cin}，scale s^X（标量）、s^W ∈ R^{Cout}，PTS 因子 δ ∈ N^{Cin}
  
  **Kernel 执行全过程**（W4A8 GEMM with PTS）：
  1. 从 global memory 加载 W̃ 行块到 shared memory
  2. **Bit-shift（PTS）**：对每个通道 k，W̃^{shifted}_{kj} = W̃_{kj} ≪ δ_k（左移操作，等价于乘 2^{δ_k}），在寄存器中完成
  3. 从 global memory 加载 X̃ 块到 shared memory
  4. INT8 × INT32 矩阵乘累加：C_ij = Σ_k X̃_ik · W̃^{shifted}_{kj}
  5. **Dequantization**：Y_ij = s^X · s^W_j · C_ij（转 FP32）
  6. 输出 Y 写回 global memory

  **评估原理**：固定矩阵维度（K=N=4096），变化 M 测量延迟。对比 PyTorch FP32 GEMM baseline，W4A8 量化 kernel 因数据位宽减小（4-bit 权重 + 8-bit 激活 vs 32-bit）实现吞吐提升，bit-shift 开销因仅在权重加载阶段执行而极低。

## Bridging the Gap Between Promise and Performance for FP4 Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现 QuTLASS v1.0，一套面向 NVIDIA Blackwell GPU 的高性能低精度量化 kernel 库，基于 NVIDIA CUTLASS 构建。包含两类 kernel：
    1. **Quantization-related kernels**：轻量级 fused kernel，实现在线 block-wise Hadamard 旋转 + 量化 + scale 计算的融合。支持 k∈{16,32,64,128} 的 block diagonal 矩阵。对 k<256，dense 变换仍为 memory-bound，任意旋转矩阵（非仅 Hadamard）可在运行时加载实现同成本运行。量化方法支持 MSE 和 Abs-Max，模板设计便于扩展。
    2. **Matmul-related narrow precision kernels**：处理 FP4 量化与矩阵乘法间的硬件强制 scale 重排（tcgen05.mma 要求），通过 Triton kernel 实现。Matmul 支持多后端（CUTLASS, FlashInfer），灵活插拔。
  - 实验比较：在 B200 和 RTX 5090 上测量单层 throughput（TFLOPS），对比 "ideal"（纯 FP4 matmul 上限）和 "actual"（含 Hadamard/量化/scale 计算开销）。端到端速度在 vLLM 中测量，MXFP4 vs BF16 baseline，不同 batch size（1-256）。

- 后端平台是什么，配置是什么。
  - NVIDIA B200 GPU（Blackwell SM100 架构）。
  - NVIDIA RTX 5090 GPU（Blackwell SM120 架构）。
  - CUDA/CUTLASS 框架 + FlashInfer 后端 + Triton kernel（scale 重排）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：vLLM 框架集成 QuTLASS kernel，测量 Llama-3.3-70B 端到端推理吞吐。
  - 修改内容（关键 kernel 设计）：
    1. **Fused Online Rotation Kernel**：将 MR-GPTQ 的激活端 Hadamard 旋转与量化+scale 计算融合为单个 kernel。Hadamard 对 k<256 的 block 为 memory-bound，因此旋转矩阵可运行时加载。Epilogue 直接完成量化，避免中间 DRAM 写入。
    2. **Scale Rearrangement Kernel**：Blackwell tcgen05.mma 要求特定的 scale factor layout，QuTLASS 用 Triton kernel 在 FP4 量化后、矩阵乘前完成硬件强制的 scale 重排。
    3. **Multi-backend Matmul**：支持 CUTLASS 和 FlashInfer 后端，根据 workload 和硬件灵活选择。B200 上 MXFP4 的 matmul throughput *超过* NVFP4（~15%），得益于 power-of-two scales 和更大 group size 降低 overhead。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/IST-DASLab/qutlass
  - Kernel 执行全过程（以 Llama-3.3-70B MXFP4 单层推理在 B200 为例）：

  **阶段 1: 预处理（离线）**
  - 权重 W 经 MR-GPTQ 量化完成：W_q = MXFP4_quant(W · H_k)，其中 H_k 为 block_size=k 的 Hadamard 旋转（k=32 for MXFP4）。旋转已融合入量化权重，权重以 MXFP4 packed 格式存储。

  **阶段 2: 推理时 Fused Rotation + Quantization（在线）**
  - 输入：FP16 激活 X ∈ R^{M×K}，block Hadamard 矩阵 H_k
  - 步骤 2a: 加载 H_k 到寄存器（k<256，Dense 矩阵，memory-bound，可在运行时任意加载）
  - 步骤 2b: X_rot = X @ H_k（block-wise 旋转，每 k×k block 独立）
  - 步骤 2c: 计算 per-group scale s_G = absmax(X_rot per G=32 elements)（fused epilogue）
  - 步骤 2d: X_q = FP4_quantize(X_rot / s_G)（fused epilogue，直接输出 E2M1 4-bit 值 + E8M0 scale）
  - 输出：MXFP4 量化激活 X_q + per-group scales

  **阶段 3: Scale Rearrangement（硬件强制）**
  - 输入：per-group scales（原始 group layout）
  - 步骤：Triton kernel 将 scales 重排为 tcgen05.mma 要求的 layout（block scaling factors layout，参照 cuBLAS 文档）
  - 输出：重排后的 scales

  **阶段 4: FP4 Matrix Multiplication（硬件加速）**
  - 输入：MXFP4 packed 权重 W_q、MXFP4 量化激活 X_q、重排后的 scales
  - 步骤：Blackwell tcgen05.mma 指令执行 FP4 矩阵乘法
  - 输出：FP16/BF16 格式的输出 activation

  **阶段 5: 性能输出**
  - "Ideal" 曲线：仅测量 tcgen05.mma matmul throughput（不含步骤 2-3 开销）
  - "Actual" 曲线：包含步骤 2-4 全部开销
  - B200 单层 speedup（vs FP16）：MXFP4 ≈ 3.6×（ideal 4×），NVFP4 ≈ 3.0×
  - RTX 5090 单层 speedup：MXFP4 ≈ 6×（ideal 8×）
  - MXFP4 B200 上比 NVFP4 高 ~15% throughput（power-of-two scales + 更大 group size 降低 overhead）
  - B200 端到端 vLLM Llama-3.3-70B speedup：MXFP4 vs BF16 = up to 2.2×（batch size=1-256）
  - RTX 5090 端到端 speedup：nearly 4×

## Squat (EdgeQAT): SIMD-based Multi-Kernel Mixed-Precision Multiplier for Mobile LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现SIMD-based Multi-Kernel Mixed-Precision (MKMP) multiplier，支持sub-8-bit混合精度MAC操作。核心创新：(1) **INT4 concatenation**：将两个4-bit权重拼接存入一个16-bit寄存器，利用ARM `mla`指令（32-bit目标寄存器INT32）同时完成乘加，理论上将4-bit GEMM的计算量减半；(2) **Token Control Logic Module (TCLM)**：在推理时根据注意力分数动态分组token为8-bit和4-bit，分别用INT8 multiplier和INT4 multiplier执行；(3) **Compiler-level memory优化**：优化计算线程分配，重叠内存读取时间。
  实验比较：在OnePlus 11和Raspberry Pi 5上测量W4A4、W8A8以及多种W4A8混合比例（4:8=1:3/1:1/3:1）的端到端推理延迟（ms/Token），对比FP16 baseline。

- 后端平台是什么，配置是什么。
  - OnePlus 11：Snapdragon 8 Gen 2处理器，全部核心多线程计算。
  - Raspberry Pi 5：BCM2712四核Arm Cortex A76处理器，四核全用。
  - 指令集：ARMv8 SIMD（NEON），利用`vmlaq_s8()`等8-bit SIMD乘加指令。`mla`指令使用32-bit目标寄存器（INT32 datatype）在单指令内完成乘法和累加。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：在目标设备上部署量化模型，测量1000次迭代的延迟取平均（ms/Token），输入序列长度128。
  - 修改内容（MKMP multiplier核心设计）：
    1. **INT4 Concatenation Kernel**（Figure 6）：将两个4-bit操作数拼接进一个16-bit寄存器。低比特优先策略（low-bit priority strategy）均匀利用位宽，最小化冗余零。16-bit宽乘法操作后内部拆分结果，维护数学精度。数学上将4-bit GEMM的乘加操作数减半（vs 传统扩展到8-bit再计算）。
    2. **INT4 Multiplier**：基于现有INT8 multiplier构建。将相邻行权重拼接，与共享激活值在SIMD kernel中相乘。利用SIMD mem机制，通过bit-shift和逐行求和累加中间值。INT4 multiplier节省50% INT8 multiplier硬件资源。
    3. **TCLM (Token Control Logic Module)**：Heapsort实现TopK重要token选择（marginal overhead）→ 分别拼接8-bit和4-bit token组 → 调用对应multiplier执行混合精度MAC。
    4. **Compiler-level优化**：针对LLM巨大内存读出的特点，优化并分配不同操作的计算线程，从编译器层面重叠内存读取时间。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/shawnricecake/squant

  **MKMP Multiplier执行全过程（以LLaMA-58M W4A8(1:1)推理一个token为例）：**

  **阶段 1: Token Adaptive Quantization（TCLM）**
  - 输入：上层的output hidden state h ∈ R^4096 × N_tokens，最新的attention map attn
  - Step 1a: scores = attn[:, 0]（每个token对初始token的平均注意力）
  - Step 1b: threshold = Heapsort_TopK(scores, k=ρ*N)，ρ=0.5即为一半token
  - Step 1c: 按threshold分组 → x_8bit = concat(important tokens)，x_4bit = concat(less important tokens)

  **阶段 2: INT8 Multiplier（处理8-bit token组）**
  - 输入：W_q INT4 packed weights（离线量化），x_8bit INT8量化激活
  - Step 2a: SIMD加载packed INT4权重 & INT8激活
  - Step 2b: ARM `vmlaq_s8()` SIMD指令执行INT8×INT8乘加 → INT32累加寄存器
  - Step 2c: Dequantization：o_8bit = α_x·α_w·C_int32
  - 输出：FP16格式的部分结果

  **阶段 3: INT4 Multiplier（处理4-bit token组）**
  - 输入：W_q INT4 packed weights，x_4bit INT4量化激活
  - Step 3a: INT4 Concatenation：将相邻行权重各4-bit拼接为16-bit寄存器（low-bit priority策略）
  - Step 3b: 16-bit宽乘加指令执行（利用`mla`，32-bit目标寄存器），内部拆分保持数学精度
  - Step 3c: Bit-shift + row-by-row summation累加中间值
  - Step 3d: Dequantization：o_4bit = α_x·α_w·C_int32
  - 输出：FP16格式的部分结果
  - INT4 multiplier节省50% INT8 multiplier的硬件资源，理论计算量减半

  **阶段 4: 结果合并与输出**
  - o = concat_and_reorder(o_8bit, o_4bit)，按原始token顺序恢复
  - 输入到下一层的LayerNorm → QKV projection → ...

  **性能输出：**
  - LLaMA-58M OnePlus 11：FP16=4.54 ms/tok → W8A8=3.22 (1.41×) → W4A4=2.02 (2.24×)
  - LLaMA-58M Raspberry Pi 5：FP16=15.63 → W8A8=9.40 (1.66×) → W4A4=6.78 (2.31×)
  - GPT2-97M OnePlus 11：FP16=6.22 → W8A8=4.35 (1.43×) → W4A4=2.75 (2.26×)
  - GPT2-97M Raspberry Pi 5：FP16=23.04 → W8A8=13.75 (1.68×) → W4A4=9.74 (2.37×)
  - 混合精度W4A8(1:1)：Raspberry Pi上额外加速超40%（vs uniform 8-bit），同时保持W4A8精度优势
  - 模型越大加速越显著（GPT2-97M > LLaMA-58M），因内存访问减少带来的效率提升更大

## AWQ Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现 TinyChat 推理系统，将 AWQ 的 4-bit weight-only 理论内存节省转化为实际端到端加速。核心包括：(1) **On-the-fly weight dequantization**：将 4-bit→FP16 反量化与矩阵乘法 kernel 融合，避免将反量化后的 FP16 权重写回 DRAM；(2) **SIMD-aware weight packing**：在 ARM NEON（128-bit SIMD）和 GPU（CUDA cores）上重新排列 4-bit 权重以高效利用 SIMD 指令解包；(3) **Kernel fusion**：融合 LayerNorm、QKV 投影、positional embedding 计算和 KV cache 更新，减少 GPU kernel launch 开销。
  - 实验比较：在 RTX 4090（桌面）、RTX 4070（笔记本）、Jetson Orin（移动 GPU）、A100（数据中心）上与 HuggingFace FP16 实现对比 token/s 加速比。在 Jetson Orin 上与 llama.cpp、exllama、AutoGPTQ 等第三方推理系统对比 Llama 模型的 token/s。在 Raspberry Pi 4B 上展示 7B 模型部署能力。

- 后端平台是什么，配置是什么。
  - GPU 后端：CUDA/PTX（NVIDIA RTX 4090, RTX 4070, Jetson Orin, A100）。实现时注重跨 GPU 架构的通用性，使用原生 PyTorch API 编写 forward pass。
  - CPU 后端：ARM NEON（128-bit SIMD，Raspberry Pi 4B）、AVX（x86 CPU）。CPU 端将整个计算图下沉到 C++ 以最小化框架开销。
  - 推理框架：TinyChat（自研），PyTorch 前端 + 设备特定指令集后端（CUDA/PTX、NEON、AVX）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方法：batch_size=1，固定 prompt 长度 4 tokens，生成 200 tokens，取中位延迟（tokens/s）。协议参照 exllama。
  - TinyChat 系统的 kernel 修改：
    1. **On-the-fly dequantization kernel**：线性层中，INT4 权重从 DRAM 读取后立即在寄存器中反量化为 FP16，融合进 GEMM/GEMV 主循环。避免写回 FP16 权重到 DRAM。对于矩阵-矩阵乘（prefill）和矩阵-向量乘（decode）分别实现融合 kernel。
    2. **SIMD-aware weight packing**（ARM NEON 128-bit 为例，Figure 4）：
       - 传统排布：`w_0, w_1, w_2, ..., w_31` 顺序存储，每 4-bit 权重解包需 3 条标量指令（shift + AND + FMA scaling），32 个权重共 96 条指令。
       - AWQ 排布：`w_0, w_16, w_1, w_17, ..., w_15, w_31`。一个 128-bit 寄存器可同时解包 32 个 4-bit 权重，仅需 3 条 SIMD 指令（AND、shift、mask）。
       - 通用规则：对于 2^n-bit SIMD 寄存器，相邻权重的索引差为 `1/8 × 2^n`。
       - GPU 端排布：每 8 个权重打包为 `w_{0,2,4,6,1,3,5,7}` 顺序（参照 Kim et al., 2022）。
    3. **Kernel fusion**：
       - LayerNorm kernel：融合乘法、除法、平方根等所有操作。
       - Attention kernel：QKV 投影融合为单个 kernel + on-the-fly positional embedding 计算 + KV cache 预分配及更新。
       - 对比 Falcon-7B 官方实现中 KV cache 处理有 bug，TinyChat 的 FP16 kernel fusion 即使不量化也带来 1.6× 加速。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/mit-han-lab/llm-awq（含 TinyChat 完整推理代码）
  - TinyChat 推理全过程（以 Llama-2-7B W4A16 在 RTX 4090 上生成一个 token 为例）：

    **阶段 1: 预处理与权重加载**
    - 输入：INT4 packed 权重（每 2 个 4-bit 权重占 1 byte）+ per-group 量化参数（Δ，FP16，每组 128 个权重存 1 个 Δ）+ per-channel scale s（FP16，C_in 维，用于 AWQ 的等效变换）。
    - 输入 prompt tokens 已处理完毕，KV cache 已填充。

    **阶段 2: 自回归生成循环（decode 阶段，memory-bound）**
    - Step 2a: **LayerNorm kernel**（fused）。输入：上一层的 output hidden state h ∈ R^4096。执行：μ = mean(h), σ = sqrt(var(h)+ε)，output = γ·(h-μ)/σ + β。所有运算在单个 CUDA kernel 中完成，无中间 DRAM 写入。
    - Step 2b: **QKV projection kernel**（fused）。输入：LayerNorm 输出 h。执行：W_Q、W_K、W_V 三个 INT4 packed 权重依次读取，on-the-fly dequantization 后执行 GEMV（h 为向量，W 为矩阵，batch_size=1 时 decode 为 matrix-vector 乘积），输出 Q、K、V 三个向量。三个投影在单个 kernel 内完成。
    - Step 2c: **Attention kernel**（fused）。输入：Q, K, V。执行：Q·K_cache^T 计算 attention scores → softmax → score·V_cache → 更新 K_cache 和 V_cache（直接写入预分配 cache 地址，无额外 kernel launch）。Positional embedding（RoPE）在计算 Q·K^T 前 on-the-fly 应用。
    - Step 2d: **Output projection kernel**。输入：attention output。执行：W_O（INT4 packed）dequantize → GEMV → 输出 h_attn。
    - Step 2e: **MLP kernel**。输入：h + h_attn（残差连接，在 LayerNorm 之后）。执行：W_gate（INT4）dequantize → GEMV → SiLU activation；W_up（INT4）dequantize → GEMV → element-wise multiply → W_down（INT4）dequantize → GEMV → 输出 h_mlp。
    - Step 2f: **残差相加** → 完成一层 Transformer Block → 输入到下一层。

    **阶段 3: 性能输出**
    - 每个 kernel 的耗时在 0.01ms 量级（RTX 4090），kernel launch overhead 与计算时间可比，因此 kernel fusion 效果显著。
    - FP16 baseline：52 tokens/s（HuggingFace）。TinyChat FP16（含 kernel fusion）：62 tokens/s。TinyChat W4A16（fusion + dequantization）：~194 tokens/s（≈3.1× over FP16 fusion baseline, 3.7× over HF FP16）。
    - On-the-fly dequantization 避免了将 4× 的 FP16 反量化权重写回 DRAM，使 decode 阶段的 arithmetic intensity 从 ≈1 提升到 ≈4 FLOPs/Byte，将 4090 上的峰值性能上限从 ~1 TFLOPS 提升至 ~4 TFLOPS。

    **阶段 4: 跨平台部署**
    - TinyChat 使用同一套 PyTorch forward pass 代码，通过不同的设备后端（CUDA/PTX for NVIDIA GPU, NEON for ARM CPU, AVX for x86 CPU）部署。VILA-7B 在 Jetson Orin 上从 FP16 的 11.5 tokens/s 加速至 W4A16 35.6 tokens/s（3.1×）。

## GPTVQ: The Blessing of Dimensionality for LLM Quantization

- **属于kernel调度/运行时计算的实现是什么？实验比较什么？**
  实现了 VQ 解码 kernel（CPU 端使用 TBL 指令，GPU 端使用 CUDA vector types），将 VQ 编码的索引解压缩为 native data type。CPU kernel：利用移动 CPU 的 hardware lookup table instruction (TBL)，将 6-bit index 映射到 8-bit signed integer（2D VQ 需 2 条 TBL 指令链接），解码后的整数用于矩阵-向量乘（SIMD 加速）。GPU kernel：使用 CUDA vector types（char4/uchar4 乃至 char128）快速加载/写回数据。

  实验比较（Table 6, Appendix B）：
  - CPU 端：Data Transfer 实验：对比 Uniform INT4、Uniform INT8、VQ 2D（3/2.75/2.25 bpv）的相对延迟和相对 footprint。Token Generation 实验：VQ 1D 3.125 bpv vs Uniform。
  - GPU 端（RTX 3080）：Data Transfer 实验：对比 Uniform INT4、Uniform INT8、FP16、VQ 2D（2.125/3.125 bpv）、VQ 4D（2.125 bpv）的相对延迟和 footprint。
  - 端到端推理（Table 1, Section 5.1）：Llama-v3-8B 在 Snapdragon X Elite 上，对比 llama.cpp INT4、自有引擎 INT4(g128)、自有引擎 VQ 2D(3.125 bpv)，测量 Model Footprint (GB) 和 Throughput (tok/s)。

- **后端平台是什么，配置是什么。**
  - CPU 端：Snapdragon X Elite 平台（mobile CPU），Windows OS，Clang 18.1 with Polly。利用 ARM TBL (Table Lookup) 指令支持 6-bit→8-bit 映射。
  - GPU 端：NVIDIA GeForce RTX 3080 GPU，CUDA。
  - 推理引擎：自研 C 语言实现（含 vector intrinsics、SIMD 扩展、polyhedral compiler capabilities），支持高度参数化 transformer 架构。

- **评估性能的软件/脚本是什么。修改了什么。**
  - 评估方式：自定义 VQ decoding kernel 集成到自研 LLM 推理引擎，测量 data transfer/decoding 延迟和端到端 token generation rate。
  - 修改内容（VQ 解码 kernel 设计）：
    1. **CPU VQ Decode Kernel**：
       - 6-bit indices 紧凑打包（packed tightly）存储，与 LUT（lookup tables）和量化 scale 按 block 组织以实现高效向量化
       - 每个 block 加载流程（Section 3.2）：DRAM → SoC cache → VQ decode kernel 使用 TBL 指令解码 → 输出 signed 8-bit int → 矩阵-向量乘
    2. **GPU VQ Decode Kernel**：
       - 使用 CUDA native vector types（char4/uchar4 + 自定义 char128 agglomerations）并行加载/写回
       - 支集 2D VQ 和 4D VQ 的解码
    3. **Packing 格式**：6-bit indices 紧凑打包（每个 weight 占 6 bits），LUT（64 entries × 8-bit），scale（FP16）；block size=8192（移动端实测配置）

- **开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。**
  论文声明 GPU kernel 代码 "will be made available in the future"（尚未开源）。推理引擎为 Qualcomm 内部实现。

  **CPU VQ Decode Kernel 执行全过程（Llama-v3-8B 2D VQ 3.125 bpv, Snapdragon X Elite）**：

  1. **输入**：packed 6-bit indices（每个 weight 占 6 bits），per-block 64-entry LUT（8-bit signed int per entry），per-block FP16 scale
  2. **数据加载**：DRAM 中读取 block tuple (indices + LUT + scale) → SoC CPU cache
  3. **Decode 阶段**：
     - 从 packed indices 解包 6-bit index（维度1和维度2各一个）
     - TBL 指令：dimension 1 index → LUT lookup → 8-bit signed int (v1)
     - TBL 指令：dimension 2 index → LUT lookup → 8-bit signed int (v2)
     - 合并：v_decoded = v1 + v2（2D VQ 合并两维结果）
     - 反量化：w_fp = scale × v_decoded
  4. **矩阵-向量乘**：SIMD 加速的 INT8/INT32 乘法累加
  5. **输出**：下一层 activation

  **性能结果**（Table 1, Llama-v3-8B, Snapdragon X Elite）：
  - llama.cpp INT4: Footprint 4.64GB, Throughput 17.95 tok/s
  - Ours INT4 g128: Footprint 4.33GB, Throughput 23.81 tok/s
  - Ours VQ 2D 3.125 bpv: Footprint 3.52GB (-19%), Throughput 26.15 tok/s (+10% vs Ours INT4)

  **CPU Data Transfer 结果**（Table 6, gate_proj 层 11008×4096）：
  - Uniform INT4: Rel. FP 1.00×, Rel. Lat 1.00×
  - VQ 2D 2.25 bpv: Rel. FP 0.56×, Rel. Lat 0.87×（延迟更低 + footprint 更小）

  **GPU Data Transfer 结果**（Table 6, RTX 3080）：
  - VQ 2D 2.125 bpv: Rel. FP 0.53×, Rel. Lat 1.03×（footprint 减半，延迟近似持平 FP16）
  - VQ 4D 2.125 bpv: Rel. FP 0.53×, Rel. Lat 0.71×（footprint 减半 + 延迟降低 29%）


## AFPQ Asymmetric Floating Point Quantization for LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：在 FasterTransformer 框架中实现了低比特 NF4-asym 反量化（dequantization）kernel。W4A16 模式下，低比特权重需要在推理时从 4-bit 反量化到 FP16，再与 FP16 激活值进行计算。具体实现：将两个 4-bit 量化权重打包存储在 1 个 byte 中；反量化时，先通过 LUT（查找表）将 NF4 值转为 FP16 值，再用 scale_pos/scale_neg 进行非对称反量化得到最终 FP16 权重。
  - 实验比较：在 A6000 GPU 上测量 LLaMA2-7B 和 LLaMA2-13B 端到端推理延迟，batch_size=1，输入序列长度 128，输出 20 token，对比 FP16、INT4、NF4-sym、NF4-asym 四种推理系统。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A6000（单卡）。
  - 推理框架：FasterTransformer（https://github.com/NVIDIA/FasterTransformer）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：FasterTransformer，测量端到端延迟（ms）。
  - 修改内容：在 FasterTransformer 中新增 NF4-asym dequantization kernel。该 kernel 实现：
    1. 从 packed byte 中解包两个 4-bit NF4 权重索引。
    2. 通过查找表（LUT）将 NF4 索引映射到对应的 FP16 值。
    3. 使用两组 scale（scale_pos 用于正值、scale_neg 用于负值）对 FP16 值进行非对称反量化。
    4. 将反量化后的 FP16 权重与 FP16 激活执行标准矩阵乘法。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/zhangsichengsjtu/AFPQ
  - kernel 执行全过程（以 LLaMA2-7B NF4-asym 推理为例）：
    1. **输入**：packed 4-bit NF4 权重矩阵（每 2 个 4-bit 值占 1 byte），每组 128 个权重对应的 scale_pos 和 scale_neg（FP16 格式）。
    2. **Dequantization 过程**：
       - Step 1 解包：从 byte 中提取高 4-bit 和低 4-bit，分别得到两个 NF4 索引（0-15）。
       - Step 2 LUT 映射：通过预置的 16 项 FP16 LUT `[0, 0.0796, 0.1609, 0.2461, ..., 1]` 的正负版本，将索引转为对应的 FP16 值。正负号由解包时的额外 bit 或独立通道判断（论文未详细说明符号位的具体编码方式）。
       - Step 3 非对称反量化：`w_fp16 = (w_nf4 > 0 ? scale_pos : scale_neg) * |w_nf4|`
    3. **矩阵乘法**：反量化后的 FP16 权重与 FP16 激活值执行标准 GEMM。
    4. **输出**：FP16 格式的输出 activation，传入下一层。
    5. **性能结果**：NF4-asym 在 LLaMA2-7B 上延迟 265.54ms（vs FP16 415.06ms，speedup 1.56x），在 LLaMA2-13B 上延迟 485.42ms（vs FP16 788.01ms，speedup 1.62x）。论文指出 NF4-asym 的 kernel 相比 INT4/NF4-sym 有额外开销，有待后续优化。

## AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 AnyBCQ 多精度 LLM 设计的 CUDA kernel，直接对二值比特平面（binary bit-planes）操作。与 Any-Precision LLM 的 kernel（需比特转置 + centroid table lookup）不同，AnyBCQ kernel 直接按需加载 p 个比特平面，利用 BCQ 的 {-1,+1} 二值特性将运算简化为激活元素的加减，并通过 LUT-based GEMM 方案缓存高频重复的部分和结果以减少算术开销。每个比特平面的计算结果乘以对应缩放因子 α_i 后累加为部分和，p 个比特平面完成后输出最终结果。
  - 实验比较：GEMV kernel 延迟（µs），对比 cuBLAS FP16、Any-Precision LLM（2/3/4-bit）在三种模型层形状（Llama-3.1-8B、Phi-4-14B、Llama-3.1-70B 的线性层维度）下的延迟。端到端吞吐量（tokens/sec）对比 Any-Precision LLM。附录还包括 kernel 延迟分解（bit-transpose vs LUT lookup vs GEMM）和 A100/H100 跨 GPU 平台吞吐量对比。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB HBM（主平台），NVIDIA H100（附录 A.3 跨平台验证）。
  - 运行环境：CUDA 12.6。
  - 测量工具：nvidia-smi（功耗/利用率）、CUDA clock64()（kernel 内周期级延迟分解）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：自定义 CUDA kernel 基准测试脚本，测量不同矩阵形状 (M×K) 的 GEMV 延迟。端到端评估使用 HuggingFace 模型加载 + 自定义 kernel 的推理流程。
  - Kernel 设计修改（对比 Any-Precision LLM kernel，Figure 3）：

    **Any-Precision LLM kernel（baseline kernel，Figure 3a）：**
    1. 加载 M×K×p 比特平面张量
    2. 比特转置（bit-transpose）：将 p 个比特平面重新排列为 M×K p-bit 索引矩阵
    3. 通过 centroid table lookup 获取每个权重的反量化值
    4. 执行 GEMM

    **AnyBCQ kernel（论文方法，Figure 3b）：**
    1. 按需加载前 p 个 M×K 比特平面（仅加载需要精度的比特，不做冗余加载）
    2. 每个比特平面直接操作：因 B_i ∈ {-1,+1}，激活元素仅需加法/减法
    3. LUT-based GEMM 优化：预计算并缓存频繁重复的部分和组合
    4. 比特平面输出 × 对应 α_i → 部分和累加
    5. p 个比特平面完成后输出最终结果
    - 消除了 Any-Precision LLM kernel 的两个主要开销：bit-transpose（占延迟 35-58%）和 centroid table lookup（占延迟 9-17%）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/naver-aics/anybcq（含 CUDA kernel 和基准测试脚本）

  **Kernel 执行全过程（以 Llama-3.1-8B 某线性层 M=4096, K=4096, p=2-bit 推理为例）：**

  | 阶段 | 描述 | 输入/操作 | 输出 |
  |------|------|-----------|------|
  | 1. 比特平面加载 | 从 HBM 加载前 p 个比特平面（每个 B_i ∈ {-1,+1}^{4096×4096}, 1 bit/元素）| packed binary tensor（M×K×p） | 寄存器中的 B_1, B_2 |
  | 2. 缩放因子加载 | 加载当前 p-bit 精度对应的缩放因子 | α_1, α_2 ∈ R（per-group, g=128） | 寄存器中的 α |
  | 3. LUT 构建 | 对激活向量每 w 个元素预计算 2^w 种可能的加减组合 | 激活向量片段 | LUT entries（FP16） |
  | 4. 比特平面计算 | B_1 比特平面：每个 M×K 元素用 sign 选择加/减激活值；LUT 查表代替逐一计算 | B_1, activation, LUT | partial_sum_1 |
  | 5. 缩放累加 | partial_sum_1 × α_1；同理对 B_2 执行步骤 4-5 | partial_sum_{1,2}, α_{1,2} | accumulated = Σ α_i · (B_i ⊙ activation) |
  | 6. 输出 | p 个比特平面累加完成，写回 HBM | accumulated ∈ R^{M} | 输出向量 |

  **性能结果（GEMV latency, µs, A100）：**
  - M=4096, K=4096, 2-bit: AnyBCQ=223 (×1.33 vs cuBLAS), Any-Precision LLM=230 (×1.29)
  - M=14336, K=4096, 2-bit: AnyBCQ=319 (×2.67 vs cuBLAS), Any-Precision LLM=353 (×2.41)
  - M=4096, K=14336, 2-bit: AnyBCQ=315 (×2.78 vs cuBLAS), Any-Precision LLM=356 (×2.47)
  - M=8192, K=28672, 2-bit: AnyBCQ=742 (×4.00 vs cuBLAS), Any-Precision LLM=971 (×3.06)

  **端到端吞吐量（Llama-3.1-8B, tokens/s, A100）：**
  - 2-bit: AnyBCQ=245 vs Any-Precision LLM=228 vs FP16=105
  - 3-bit: AnyBCQ=212 vs Any-Precision LLM=196 vs FP16=105
  - 4-bit: AnyBCQ=186 vs Any-Precision LLM=169 vs FP16=105

  **Kernel 延迟分解（Any-Precision LLM kernel，Table 7）：**
  - Bit-transpose 占 35-58% 延迟（最大开销）
  - LUT lookup 占 9-17%
  - GEMM + memory 等其余操作占 31-50%
  - AnyBCQ 通过消除 bit-transpose 和 centroid lookup 这两项开销获得加速。

## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Mamba 设计了**硬件感知的选择性扫描（Hardware-Aware Selective Scan）**内核，用 kernel fusion + 并行扫描（parallel scan）+ 重计算（recomputation）三种经典技术使选择性 SSM 在现代 GPU 上高效运行。核心问题是：选择性 SSM 不再是 time-invariant，无法使用卷积模式（FFT）高效计算，而朴素循环模式下需要物化 (materialize) 大小为 (B, L, D, N) 的中间状态 h（N≈16 时比输入 x 大 N 倍），导致大量 HBM 读写。Mamba 的解法是：
    i) **Kernel Fusion**：将离散化（discretization）、并行扫描、与 C 的乘加融合进单个 CUDA kernel。直接从慢速 HBM 加载 O(BLD) 的 (Δ, A, B, C) 参数到快速 SRAM，在 SRAM 内完成离散化得到 (Ā, B̄) → 并行扫描计算 h → 乘以 C 得到 y，仅将最终 O(BLD) 的输出写回 HBM。减少 IO 量约 O(N) 倍
    ii) **Parallel Scan**：使用 Blelloch 工作高效的并行关联扫描算法（work-efficient parallel associative scan），将序列循环转化为 O(log L) 次并行操作
    iii) **Recomputation**：前向时不保存中间状态 h (B, L, D, N)，反向传播时重新加载输入从 HBM 到 SRAM 并重计算 h。由于输入/输出/梯度之和 O(BLD) 远小于 h 的 O(BLDN)，重计算比存储并读取 h 更快
  - 实验比较：
    - 扫描速度：Mamba fused scan vs 标准 PyTorch scan（完整物化 Ā, B̄, C 于 HBM）vs FlashAttention-2 vs 卷积（FFT）。测量序列长度 512–500K，batch size=1, D=1024, N=16, BF16
    - 端到端推理吞吐：Mamba-1.4B/6.9B vs Transformer-1.3B/6.7B（HuggingFace 实现），prompt 长度 2048，生成长度 128，batch size 1–128
    - 训练内存：Mamba-125M vs Transformer-125M（w/ FlashAttention-2 + torch.compile），batch size 1–32，序列长度 2048

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB PCIe GPU
  - 计算后端：CUDA（自定义 fused scan kernel），PyTorch（标准 scan baseline + Transformer 端到端推理）
  - 精度：BF16（训练/推理主流配置）

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：自定义 CUDA scan kernel + PyTorch benchmark 脚本
  - 修改内容：在选择性 SSM 层中，将原本需先物化 (Ā, B̄, C) ∈ R^{B,L,D,N} 到 HBM → 调用 PyTorch scan → 乘 C → 写回 HBM 的标准流程，替换为单一 fused kernel
  - 对比的 baseline：
    - 标准 PyTorch scan：使用 `torch.cumsum` 或手工并行 scan 实现，完整物化中间状态
    - 卷积（FFT）：PyTorch 的 FFT-based convolution，O(L log L) FLOPs 但 LTI 限制
    - FlashAttention-2 (Dao 2024)：带 causal mask，当前最快的 attention kernel

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/state-spaces/mamba
  - **Fused Selective Scan Kernel 执行流程**：

  ```
  输入（从 HBM 加载到 SRAM）:
    Δ ∈ R^{B,L,D}  (step size)
    A ∈ R^{D,N}    (diagonal state matrix, 可复用)
    B ∈ R^{B,L,N}  (input projection)
    C ∈ R^{B,L,N}  (output projection)
    x ∈ R^{B,L,D}  (optional, 若未在上游 projection 融合)

  SRAM 内计算:
    1. 离散化 (Discretization) — 对每个 timestep t:
       Ā_t = exp(Δ_t ⊙ A)       → R^{D,N}, element-wise exp
       B̄_t = Δ_t ⊙ B_t          → R^{D,N}, 一阶 Taylor 近似
       (注意: ⊙ 表示沿 N 维度的 broadcast element-wise 乘法)

    2. 并行关联扫描 (Parallel Associative Scan) — Blelloch 算法:
       // 输入: 序列 {(Ā_t, B̄_tᐧx_t)} for t=1..L
       // 关联操作: (a, b) ∘ (a', b') = (a'⊙a, a'⊙b + b')
       // h_t 为扫描输出: h_t = Ā_t⊙h_{t-1} + B̄_t⊙x_t

       Up-sweep (reduce phase): O(log L) parallel steps
         for d = 0 .. log₂(L)-1:
           for k = 0 .. L/2^{d+1}-1 in parallel:
             combine elements at indices 2^d·(2k+1)-1 and 2^d·(2k+2)-1

       Down-sweep (distribution phase): O(log L) parallel steps
         将中间结果传播, 输出完整 h_{0..L-1}

    3. 输出乘加:
       y_t = C_t ⊙ h_t  → R^{D}, for t=1..L

  输出（写回 HBM）:
    y ∈ R^{B,L,D}  (最终输出，与输入同形状)

  内存 IO 对比:
    标准方法: Read O(BLDN) + Write O(BLDN) + Read O(BLDN) = O(3BLDN) HBM I/O
    Fused方法: Read O(BLD) + Write O(BLD) = O(2BLD) HBM I/O
    加速比 ≈ N (当 N=16 时约 16×)，实测 20–40× 比 PyTorch naive scan
  ```

  - **长序列分块处理**：当序列长度 L 超过 SRAM 容量时，将序列分成 chunks。每个 chunk 内的 fused scan 在 SRAM 执行，通过保存 chunk 间的中间扫描状态（scan state）在 HBM 中连接相邻 chunks

  - **反向传播重计算**：
    ```
    前向: 不保存 h ∈ R^{B,L,D,N}（太大）
    反向:
      1. 从 HBM 加载 Δ, A, B, C, x (O(BLD))
      2. 在 SRAM 中重计算 h（与正向相同计算）
      3. 用 h 和 upstream gradient (从 HBM 加载, O(BLD)) 计算 Δ, A, B, C, x 的梯度
      4. 写回梯度 (O(BLD)) 到 HBM
    总 HBM I/O = O(BLD), 相比于保存/加载 h 的 O(BLDN) 更少
    ```

  - **内存消耗对比**（Table 15）：Mamba-125M ≈ 4.8GB (batch=1) ~ 38.2GB (batch=32) vs Transformer w/ FlashAttention-2 ≈ 4.6GB ~ 34.5GB，处于同一量级。每个 selective SSM 层约 16 bytes/token 激活内存，两层 ≈ 32 bytes/token（等价于 attention+MLP）

  - **关键结果**：
    - Fused scan 在序列长度 >2K 后超越 FlashAttention-2，在 32K 时快约 7×
    - 比 naive PyTorch scan 快 20–40×（所有序列长度）
    - Mamba-6.9B 推理吞吐 > Transformer-1.3B 的 5×（因无 KV cache 可用更大 batch size）
    - 扫描比 FFT 卷积 O(L log L) 在长序列上常数因子优势增大

## MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - MicroMix 在 Blackwell GPU 上实现了基于 CUTLASS 的混合精度 GEMM kernel 和 fused reorder-and-quantize kernel。GEMM kernel 支持单一 kernel 内任意混合 MXFP4/MXFP6/MXFP8 通道比例，利用 Blackwell Tensor Core 的 MMA 指令（原生支持 FP4/FP6/FP8 + 融合 scale 反量化），输出 BFloat16。Fused reorder-and-quantize 将通道重排和 block-wise MX 量化合并为一个 kernel，避免 irregular memory access 的开销。实验比较了：(1) 单 kernel 延迟 vs TensorRT FP16/FP8/W4A16；(2) 自定义 GEMM kernel vs CUTLASS 的 TFLOPS 和加速比；(3) prefill 延迟和 decode 吞吐 vs Atom/QuaRot/FP16/INT8；(4) 峰值内存占用。
- 后端平台是什么，配置是什么。
  - NVIDIA RTX 5070Ti Laptop GPU（Blackwell）、RTX 5090（Blackwell）、RTX PRO 6000（Blackwell）。Blackwell Tensor Core 支持 FP4 (E2M1) MMA，FP4 吞吐为 FP16 的 4×、FP8/INT8 的 2×，MMA 指令原生融合 scale factor 反量化。
- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 CUTLASS 实现自定义 MXFP GEMM kernel，支持 MXFP4/MXFP6/MXFP8 的混合精度矩阵乘法。baseline 使用 TensorRT FP16、TensorRT FP8 (per-tensor)、TensorRT W4A16 (per-token)、HuggingFace FP16、Bitsandbytes INT8、Atom/QuaRot 的 INT4 kernel。修改：(1) GEMM kernel 按精度分组分别调用对应的 CUTLASS MXFP GEMM 实例；(2) 实现 fused reorder-and-quantize kernel（将通道重排 + block-wise 量化 + scale 计算融合）；(3) 集成 FlashInfer 进行 KV cache INT4 量化以进一步减少内存占用。
- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/lwy2020/MicroMix
  - Kernel 评估原理与流程：
    ```
    1. 输入：FP16 激活 X ∈ R^{M×K}，已量化权重 W ∈ MXFP_format（包含 scale）
    2. 配置加载：读取该层的 p4/p6/p8 比例和排列 σ
    3. Fused Reorder-and-Quantize Kernel（GPU kernel）：
       - 从 global memory 读取 FP16 X
       - 按 σ 对通道索引重排（在 shared memory / register 中完成）
       - 分组：前 p4*K 通道 → G4, 中 p6*K 通道 → G6, 后 p8*K 通道 → G8
       - 对每组内每 32 个元素 (block_size=32)：
         s = 2^{floor(log2(max(|block|))) - b}  (E8M0)
         Q(x) = round(clip(x/s, -q_max, q_max))
       - 输出三组 MX format 张量（含 element + shared scale）
    4. GEMM Kernel（CUTLASS-based MMA）：
       - 对每组精度分别：加载 A_tile (MXFP) + B_tile (MXFP) + scales → Tensor Core
       - MMA.884 或类似指令：每次操作 = A·B + scale_dequant → FP32 accum
       - 累加到 BFloat16 输出 tile，写回 global memory
    5. 输出：三组结果按 σ^{-1} 恢复原通道序，得到 BF16 Y ∈ R^{M×N}
    ```
  - 性能评估：对 M={1,2,4,8,16,32,64,128}, N=K=4096 测量 TFLOPS 并与 CUTLASS 对比。对 sequence length {128,256,512,1024,2048,4096} 测量单 kernel 延迟并与 TensorRT 对比。
  - 关键结果：RTX 5070Ti laptop 上 MicroMix kernel 2.45-2.93× vs TensorRT-FP16, up to 1.45× vs TensorRT-FP8。RTX 5090 上 2.29-3.38× vs TensorRT-FP16, up to 1.74× vs TensorRT-FP8。自定义 GEMM vs CUTLASS：W6A6 在 M=32 时最大 5.0× 加速。Fused reorder-and-quantize 仅占总 kernel 时间的 <20%。RTX PRO 6000 上 MicroMix prefill 延迟约 Atom/QuaRot 的 15%，decode 吞吐约 Atom 的 1.82-3.02×。memory 减少：vs FP16 减少 2.29-2.84×，vs INT8 减少 1.60-2.01×。

## MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MxMoE 自动生成混合精度 Group-GEMM GPU kernel，包含三个核心组件：
    1. **Micro-Kernel Specialization**：为每种量化精度（W2A16, W4A16, W4A4-g128, W8A8 等）实现可配置的 CTA 级 CUDA device function，利用 CTA index independence 实现水平融合。例如 W2A16 micro-kernel 使用 fused dequantization + bit manipulation 优化 int-to-float 转换；W4A4-g128 使用 multistage software pipelining 严格遵循 128 量化 group 约束。Memory access 模式针对每种量化方案手工调优 compute-to-memory access pipeline。
    2. **Resource Configuration**：为水平融合的混合精度 Group-GEMM kernel 配置计算资源。强制所有 micro-kernel tiles 使用相同 warp count（满足 CUDA 编程模型的 uniform resource 要求），shared memory 按融合操作中最大需求分配。为减少因 tile size 差异导致的 shared memory 浪费，引入 k-dimension tiling（slice-K）对较小 tile 增加 k 维并行度，同时减少 warp under-utilization。
    3. **Tile Scheduling**：因不同精度和 tile shape 组合的执行时间差异显著，tile 调度顺序直接影响总完成时间。MxMoE 使用 greedy 启发式优先调度计算密集 tile，在 MoE block tiles 数远大于 SM 数时实现近最优性能（符合 Graham 1966 的 bound）。
  - 实验比较：因缺乏已建立的 low-precision Group-GEMM baseline，比较 MxMoE 生成的 uniform-bitwidth 和 mixed-precision kernel vs CUTLASS 16-bit Group-GEMM。评估 memory-bound（512 tokens）和 compute-bound（8192 tokens）两种 workload 下的 MoE block 计算吞吐量。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX 4090 GPU（Ada Lovelace 架构）
  - CUDA/CUTLASS 框架

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估方式：从 WikiText-2 随机采样 512/8192 token 序列，测量 MoE block 计算吞吐量（仅计 expert GEMM 计算，gate/topk/sort 等开销可忽略）。
  - 修改内容（混合精度 Group-GEMM kernel 设计）：
    1. Micro-kernel：每种精度独立实现为 CUDA device function，模板参数指定资源配置
    2. Kernel generator：根据 ILP 分配的方案，自动组合多个 micro-kernel 为统一 kernel，生成 precision-aware routing logic
    3. Tile scheduler：greedy LPT（Longest Processing Time first）启发式
    4. K-dimension tiling (slice-K)：W4A16 tile 比 W8A8 tile 显著更小，slice-K 将 W4A16 的 k 维切分为多个 sub-tile，增加 SM 利用率
  - 对比 kernel：
    - HQQ kernel（不融合 dequantization，性能差）
    - VLLM-Marlin-MoE kernel（顺序调用 Marlin W4A16 kernel，suboptimal GPU utilization）
    - CUTLASS 16-bit Group-GEMM（full-precision baseline）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源地址：https://github.com/cat538/MxMoE
  - Kernel 执行全过程（以 DeepSeek-V2-Lite MoE block W4.25A15.5 在 RTX 4090 上为例）：

  **阶段 1: Kernel 编译时生成**
  - 输入：混合精度方案 {x_{i,j,k}}（如某些 expert 的 down_proj 用 W4A16，gate_proj 用 W8A8）
  - Step 1a: 对方案中出现的每种精度，选择对应的 micro-kernel（如 W4A16 → Marlin-style fused dequant kernel, W8A8 → standard INT8 GEMM kernel）
  - Step 1b: Resource Configuration — 所有 micro-kernel 统一 warp count（如 4 warps/CTA），shared memory 取所有方案中的最大值
  - Step 1c: K-dimension tiling — 对 tile size 较小的方案增加 k-dim split（如 W4A16 的 k=256 分为 2×128）
  - 输出：编译后的 fused mixed-precision Group-GEMM kernel

  **阶段 2: 运行时执行**
  - 输入：MoE block 输入 X ∈ R^{T×d}（T tokens），各 expert 的 INT4/INT8 packed 权重 + scale + zero-point
  - Step 2a: Gating → 每个 token 分配到 top-k expert → 按 expert 分组 token，得到 per-expert X_e
  - Step 2b: Tile Scheduler 构建 tile list：
    对每个启用的 expert e，对每个 linear block j（gate/up/down），根据其分配精度 k 和 tile config t，将 GEMM (X_e, W_{e,j}) 分解为 tiles {(c, n_t)}，所有 tiles 汇总为全局调度队列
  - Step 2c: Greedy LPT 调度 — 按 tile 执行时间 c 降序排列，依次分配到有空闲 SM 的 tile slot
  - Step 2d: SM 执行 micro-kernel：
    - W4A16 tile：从 global memory 加载 INT4 packed W + FP16 scale → shared memory → fused dequantization（INT4→FP16）→ Tensor Core MMA → FP16 accumulator
    - W8A8 tile：加载 INT8 W + INT8 activation → Tensor Core IMMA → INT32 accumulator → dequant to FP16
  - Step 2e: 所有 tiles 完成后，reduction 得到 MoE block output

  **阶段 3: 性能输出**
  - Memory-bound（512 tokens）：W4.25A15.5 比 FP16 快 1.6-2.7×，比 uniform W4A16 快 up to 25%（Qwen1.5-MoE）
  - Compute-bound（8192 tokens）：W5A5 比 FP16 快 3-3.4×，比 uniform W8A8 快 up to 29.4%
  - 混合精度优势来源：hardware-aware bitwidth allocation 将低精度 activation 分配给高频激活 expert（compute-bound），保持高频 expert 高精度

## QTIP: Quantization with Trellises and Incoherence Processing

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QTIP 设计了硬件高效的 **bitshift trellis 解码 kernel**，在 GPU 上将 TCQ 编码的权重实时解码用于矩阵乘法。核心设计：(1) **Bitshift Trellis 解码**——每个 group of V 权重仅依赖连续 L-bit 窗口，解码时仅需 kV-bit 位移（bitshift by kV）获取下一组权重，硬件原生支持且完全并行化；(2) **Compute-based 解码 kernel**——1MAD 码仅需 2 GPU 指令（MAD + vabsdiff4）、3INST 码仅需 3 GPU 指令（MAD + lop3 + add）、HYB 码摊销 2 指令（MAD hash + lop3 sign-flip + LUT lookup），均 ≤4 指令/权重；(3) **MMA Tile 映射**——T_x=T_y=16 使每个 trellis 序列精确对应一个 16×16 MMA tile（NVIDIA Tensor Core 基础 tile 尺寸），矩阵乘法直接利用硬件加速单元；(4) **Tail-biting** 使编码比特总数能被 32-bit 字长整除，避免读取浪费比特；(5) **Codebook Cache 优化**——HYB 码 codebook 仅 2KiB (2^9×2 FP16)，可放入 L1 cache 甚至多次复制（32×）以消除 bank conflicts。
  - 实验比较：(1) 端到端推理吞吐（Table 4）：RTX 6000 Ada 上 batch_size=1 decode，Llama 2 7B/70B 的 QTIP vs QuIP# vs AQLM vs FP16，QTIP 2-bit 达 188/23.5 tok/s vs QuIP# 186/22.2、AQLM 81.5/8.78；(2) 跨 GPU 解码速度（Table 17）：RTX 3090、RTX A6000 Ampere、RTX 6000 Ada 上的 2/3/4-bit tok/s；(3) 与峰值带宽对比：QTIP 解码达到 >80% 峰值显存带宽。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX 6000 Ada (960GB/s 显存带宽, Ada Lovelace)、NVIDIA RTX 3090 (Ampere)、NVIDIA RTX A6000 Ampere。CUDA/PTX 实现。利用 16×16 MMA tile（Tensor Core）进行矩阵-向量乘法，decode 阶段 memory-bound。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 CUDA kernel 实现 bitshift trellis 解码 + dequantization + GEMV 融合。修改内容：
    1. **Bitshift 解码逻辑**：从 packed bitstream 读取当前 L-bit 窗口 → 通过 compute-based code (1MAD/3INST/HYB) 即时生成伪随机高斯权重 → 执行 FP16 GEMV。下一组权重通过 kV-bit 位移获取，无需存储完整 trellis 结构。
    2. **1MAD Kernel**：x = (a*x + b) mod 2^32 → 求和四个 8-bit unsigned ints → scale/shift → 输出近似高斯。2 指令: MAD (mul-add) + vabsdiff4（求和 4 个 8-bit 整数）。
    3. **3INST Kernel**：x = (a*x + b) mod 2^32 → 取 bottom/top 16 bits 分别 XOR magic FP16 m 的尾数/指数/符号位 → m1 + m2 → 输出近似高斯。3 指令: MAD + lop3 (logic op 3-input) + FADD。
    4. **HYB Kernel**：x = x²+x mod 2^32 → 取 bits (14-Q+1):14 作为 LUT index → 查表得 2D 向量 → XOR bit 15 翻转第二分量符号。摊销 2 指令。LUT 2KiB 常驻 L1 cache。
    5. **Tail-biting 对齐**：通过 Algorithm 4 近似 tail-biting，使 kT 能被 32 整除，无浪费比特读取。
  - 评估方式：测量 batch_size=1 decode 的端到端吞吐量 (tokens/s)，比较各量化方法的推理速度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/Cornell-RelaxML/qtip
  - Kernel 输入到性能输出全过程（以 Llama 2 7B 2-bit QTIP HYB decode 为例，RTX 6000 Ada）：
    1. **输入**：packed bitstream（每 L=16 bits 编码一组权重，tail-biting 对齐 32-bit word），HYB codebook C ∈ R^{2^9×2} (2KiB, 常驻 L1 cache), LCG 参数 a, b。
    2. **Bitstream 加载**：从 DRAM 读取 32-bit word 到寄存器，通过 tail-biting 结构每 kT = 2×16×16 = 512 bits 对齐一个 16×16 tile。
    3. **Bitshift Trellis 解码**：对 Tx×Ty=16×16 tile 内每个权重位置，通过 bitshift 操作从 bitstream 窗口提取 L=16 bits 状态字，并行处理——每个权重仅依赖 16-bit 连续窗口。
    4. **HYB Code 解码**（per weight, 摊销 2 instrs）：
       - x ← x²+x mod 2^32 (MAD: 1 inst hash)
       - idx ← (x >> 6) & 511 (bitmask, fused in lop3)
       - v ← C[idx] (L1 cache lookup, 2×FP16)
       - sign-flip v[1] via XOR bit 15 (lop3: 1 inst)
       - 输出 2 个 FP16 权重值
    5. **MMA 计算**：16×16 tile 的 FP16 权重 × FP16 激活向量 → Tensor Core MMA (matrix-vector multiply accumulate) → FP32 accumulator → FP16 output。
    6. **输出**：当前 token 的 hidden state，传入下一 Transformer 层。
    7. **性能结果**：Llama 2 7B 2-bit 188 tok/s (>3× FP16 55.9 tok/s)，70B 2-bit 23.5 tok/s (FP16 OOM)。QTIP 与 QuIP# 吞吐相当，但有效维度为 256（QuIP# 仅 8），量化质量更高而无额外推理开销。
  - 关键优化：compute-based codes 消除了 VQ 方法需要的大 LUT 存储（AQLM 1MiB codebook 无法放入 L1 cache），HYB codebook 仅 2KiB → 32× 复制消除 bank conflicts。Bitshift trellis 的并行解码消除了 naive TCQ 的顺序依赖。

## QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QuIP# 设计了针对 E8P 编解码的高效 CUDA kernel，用于量化线性层的矩阵-向量乘法（decode_matvec_e8p_kernel），结合 Fast Walsh-Hadamard Transform (FWHT) 实现完整推理管线。Kernel 核心设计：
    - **E8P 解码 + MMA 融合**：在一个 warp 级 kernel 中完成 E8P 码字解码和 Tensor Core MMA 累加。(1) 从压缩权重 `weights_compressed` 读取 uint2 码字，每 uint2 包含 4 个 16-bit E8P 码字；(2) 用 `codebook_abs[256]` 查找源码书绝对值条目（4-bit 整数存储，128B L1 cache）；(3) 通过 XOR 操作和移位实现符号翻转解码（7+1 位 sign bits）；(4) 生成 FP16 权重后直接送入 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` PTX 指令做矩阵乘法累加。Kernel 使用 warp 级数据共享（`__shfl_sync`）和 bank conflict 避免（32× 复制 codebook 到 L1）。
    - **RHT（Fast Walsh-Hadamard Transform）**：O(n log n) 时间复杂度，仅涉及 ±1 加减法运算，无浮点乘法。在推理时对激活向量做两次 Hadamard 变换（前/后），用于撤销量化时的非相干处理。
    - **压缩格式**：权重以 2-bit E8P 码字打包存储——每个 16-bit 码字编码 8 个权重共 16 bits（2 bits/weight × 8），E8P 源码书仅 256×8 条目 = 1KiB，可完全放入 L1 cache（即使在 32× bank conflict 复制后）。
  - 实验比较：(a) NVIDIA RTX 4090 上 Llama 2 7B/13B/70B 和 Llama 1 30B 的 2-bit/4-bit 生成吞吐（tok/s）及峰值显存带宽利用率（2-bit 下 70B 达 56.84%）; (b) RTX 4090 上 QuIP# vs AQLM vs FP16 的 Llama 2 7B/70B 吞吐对比——QuIP# 2-bit 显著快于 FP16 和 AQLM；(c) A6000 上 QuIP# vs QuIP 吞吐对比——同 bitrate 下 QuIP# 约为 QuIP 的 2 倍。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA RTX 4090（Ada Lovelace 架构，1TB/s 峰值显存带宽，L1 cache 128KB/SM）；NVIDIA A6000（Ampere 架构）。
  - CUDA PTX：使用 mma.sync.aligned.m16n8k16 指令，针对 Ampere 及更新架构 Tensor Core（FP16 输入→FP32 累加）。
  - 软件栈：FlashAttention 库的 Llama 实现和 HuggingFace Llama 实现分别用于吞吐测试。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本：基于 FlashAttention 库（Dao et al., 2022; 2023）的 Llama 模型实现进行生成吞吐测试；HuggingFace Llama 实现用于与 AQLM 的公平对比。
  - 修改内容：(1) 将标准 FP16 线性层替换为 QuIP# 量化线性层（包含两次 FWHT + E8P GEMV kernel）；(2) 自研 E8P decode_matvec CUDA kernel，编译为 quiptools_e8p_gemv.cu；(3) 权重预量化为压缩码字格式后以 uint2 数组加载。
  - 优化程度：论文描述为 "proof of concept" 实现，minimal kernel fusion in the RHT，仍有进一步融合优化空间（Hadamard transform 与后续操作融合）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/Cornell-RelaxML/quip-sharp 包含 `quiptools/quiptools_e8p_gemv.cu`；预量化模型：https://huggingface.co/relaxml
  - Kernel 输入到性能输出全过程（以 Llama 2 70B 2-bit QuIP# decode 为例，RTX 4090）：
    1. **输入**：压缩权重 `weights_compressed`（uint2 数组，每个 uint2 含 4 个 16-bit E8P 码字，平均 2 bits/weight）、`codebook_abs`（256 条 uint32_t 绝对值条目，常驻 L1 cache）、`input`（当前 token 的 hidden state，FP16 向量）。
    2. **线程组织**：Grid 按输出维度 N 分块（每 16 行一个 block），Warp 内 32 个线程处理 K 维度上连续的权重块。warpId = threadIdx.y 确定处理 K 维度哪个 chunk，laneId = threadIdx.x 确定 Warp 内线程 ID。
    3. **E8P 码字解码**（per weight group of 8 weights）：
       - 从 `weights_compressed` 读取 uint2 w_compr（a = w_compr.x, b = w_compr.y）
       - Parity 计算：s = b XOR (b>>4) XOR (b>>8) XOR (b>>16)，sb = s & 15 为每 8 个权重的奇偶性位
       - 对 4 个 E8P 码字（a 的 4 个字节）分别解码：
         a. `x = codebook_abs[(a >> shift) & 255]` — 查表得 8 个权重的绝对值（4-bit pack）
         b. `x = x XOR ((s & mask) * factor)` — 解码 7 位符号翻转
         c. `o = BASE_OFFSET | ((sb & mask) << shift2)` — ±1/4 偏移
         d. `w00..w03 = add_as_half2(mask_lop3(...))` — 生成 4 组 FP16 权重对
       - 使用 `lop3` 指令融合多个位操作，< 5 指令/权重
    4. **MMA 累加**：`__shfl_sync` 广播激活片段 → `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` Tensor Core MMA 指令，输入 FP16 权重（w00-w13）和 FP16 激活片段（x_in0, x_in1），FP32 累加到 z0-z3。
    5. **输出**：`atomicAdd` 将各 Warp 的 FP32 累加器写回 output 数组 → 经第二个 FWHT (S_U) → 最终 hidden state。
    6. **性能结果**：Llama 2 70B 2-bit 达 32.74 tok/s（56.84% peak mem BW），7B 2-bit 达 170.50 tok/s（29.60%），均远超 AQLM（2-7B 20.6 tok/s, 2-70B 8.27 tok/s）和 FP16（2-7B 33.1 tok/s）。
    7. **关键设计优势**：E8P codebook 仅 1KiB → 可放 L1 cache → 避免 AQLM 1MiB codebook 导致的 L1 cache miss；E8 格高 packing density → 量化误差低于 D4 格和 K-Means。

## QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QuaRot 实现了三类核心 CUDA kernel：(1) **4-bit 线性层 kernel**：输入 FP16 激活 → 可选在线 Hadamard 变换 → 量化 kernel 将激活转为 sub-byte INT4 → CUTLASS 4-bit GEMM kernel（TensorCore, INT32 accumulator）执行 INT4×INT4 GEMM → dequant 输出 FP16；(2) **量化 KV cache attention kernel（基于 FlashInfer）**：三阶段——Init（prefill 时初始化量化 KV cache，直接用 Flash Attention 计算 attention output）、Append（解码时对当前 K/V 做 asymmetric group-wise 量化 → pack sub-byte → 追加到 cache）、Decode（从 HBM 加载量化 KV cache → 反量化 → 与 FP16 query 做 online softmax attention → FP16 output）；(3) **在线 Walsh-Hadamard 变换 kernel**：对激活值执行 O(d log d) 快速 Walsh-Hadamard 变换，支持 FP16 和 FP32，在 down-projection 和 out-projection 前调用。
  - 实验比较：(a) 4-bit linear layer vs FP16 linear layer 延迟对比（不同矩阵规模 4096×4096 ~ 28672×8192，batch=1-32，Table 14）；(b) 有无在线 Hadamard 变换的开销（INT4 vs INT4+FP32 Had vs INT4+FP16 Had）；(c) 单 transformer block 的 prefill 加速比（TTFT, batch=1/4/16/32/64, seq=2048，Figure 4 左，Table 16）；(d) 解码阶段峰值内存节省（batch=1/16, seq=256-4096，Figure 4 右，Table 17）；(e) KV cache decode kernel 延迟 vs FP16（不同 head_num×head_dim, batch=1-32, Table 15）；(f) 2D linear layer 速度随 batch scaling 行为（Figure 7）。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX 3090 GPU（Ampere 架构）。CUDA 12.1。PyTorch + Hugging Face Transformers。CUTLASS 库（github.com/NVIDIA/cutlass）提供 INT4 TensorCore GEMM template。FlashInfer 库（github.com/flashinfer-ai/flashinfer）提供量化 KV cache attention 的 append/decode 路径。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 benchmark 脚本：对单 transformer block 中每个线性层和 attention 层分别测量延迟（100 次运行取平均, CUDA events 计时）。修改 CUTLASS：适配 QuaRot 的 sub-byte packed INT4 数据布局（激活值 per-token 量化为 INT4 → 2×INT4 pack 为 1 byte，权重 per-column INT4 → pack 格式）。修改 FlashInfer：在 attention decode 路径中加入量化 KV cache 的加载和反量化逻辑，支持 asymmetric group-wise dequant（K_fp16 = (K_q - z_k) × s_k, group=128）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/spcl/QuaRot
  - Kernel 输入到性能输出全过程（以 4-bit 线性层 W_down 11008×4096 为例）：
    1. **输入**：FP16 激活 X ∈ R^{B×T×4096}（prefill 阶段）+ INT4 packed 权重 W_q（per-column symmetric quant, weight scales s_w ∈ R^{11008}）+ per-column weight scales
    2. **在线 Hadamard**：X_h = FastWalshHadamard(X, dim=-1)（O(B×T×4096×log₂(4096)) = O(B×T×4096×12) 操作, ~7% 额外开销, FP16 和 FP32 精度几乎等效）
    3. **激活量化**（per-token symmetric INT4）：s_x[t] = max(|X_h[t,:]|) × 0.9 / 7.0（clipping ratio=0.9）→ X_q[t,i] = round(clip(X_h[t,i] / s_x[t], -7, 7)) → 2×INT4 pack 为 1 byte
    4. **CUTLASS INT4 GEMM kernel launch**：
       - Grid/Block: 按 M (B×T) 和 N (11008) 维度 tile 分配
       - Shared memory: 加载 packed X_q tile + packed W_q tile
       - Dequant on-the-fly: ŵ = unpack_4bit(w̃) × s_w[col]（per-column scale）
       - TensorCore: INT4×INT4 → INT32 accumulate（m16n8k32 tile）
       - 输出: Y_int32 ∈ R^{B×T×11008}
    5. **Dequant 输出**：Y_fp16 = (float(Y_int32) ⊙ s_x[:, None] ⊙ s_w[None, :]) → cast to FP16
    6. **评估原理**：CUDA events (cudaEventRecord) 记录 kernel launch 到 completion 的 wall clock time。预热 10 次后测量 100 次取平均。对比 FP16 cuBLAS GEMM baseline 的 wall clock time。加速比 = FP16_time / INT4_time。LLAMA2-7B 4-bit linear layer 达到 3.2× speedup (W_down, batch=1)，LLAMA2-70B 达到 4.3×。
    7. **Attention Decode Kernel 流程**（32 heads × 128 dim, KV cache 2047 tokens）：
       - 输入: FP16 query q (B×32×128) + INT4 packed KV cache
       - 加载: 按 group=128 从 HBM 加载 INT4 cache → dequant K/V → FP16
       - Online softmax: qK^T/√128 → softmax → ×V（逐 tile 累加, 避免完整 attention matrix 物化）
       - 输出: attention output (B×32×128)
       - 性能: batch≥16 时 4-bit 比 FP16 快 1.72×；小 batch (≤8) 时 4-bit 因量化/反量化开销略慢于 FP16

## QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 QuantCache 框架开发了 optimized GEMM CUDA kernels，通过 kernel fusion 技术将三个运行时计算组件融合为高效的单次 kernel 调用：(1) **Quantization + Rotation kernel fusion**：将 activation quantization（uniform min-max quantize，online 计算 s_X = (max(X)-min(X))/(2^b-1)）与 channel-balancing rotation 变换（R @ X_balanced）融合为单一 CUDA kernel，避免中间结果写回 global memory；(2) **Intermediate feature caching kernel**：在 GPU shared memory / L2 cache 中缓存 HLC 判定可复用的 intermediate features，跨 timestep 直接复用，减少 HBM 访问；(3) **Scaling factor absorption**：受 QServe、SmoothQuant、ViDiT-Q 启发，将 channel-balancing 的 scaling factors offline 吸收到前层权重中（W'_prev = S ⊙ W_prev），消除推理时的额外 scaling 开销。Kernel 融合后，QuantCache 的量化过程不再引入额外的 kernel launch 和 global memory round-trip，使整个 DiT 推理的 CUDA kernel launch 次数显著减少。
  - 实验比较：(a) Speedup 对比：QuantCache 6.72× vs Open-Sora baseline 1.00× on A800-80GB；(b) 对比 T-Gate (1.10×), PAB (1.34×), ViDiT-Q (1.71×), AdaCache-slow (1.46×), AdaCache-fast (2.24×) 的端到端加速比。论文未提供逐 kernel 的 micro-benchmark。

- 后端平台是什么，配置是什么。
  - NVIDIA A800-80GB GPU（Ampere 架构，80GB HBM2e），CUDA 12.1。CUDA kernel 实现受 QServe (Lin et al., MLSys 2025)、SmoothQuant (Xiao et al., ICML 2023)、ViDiT-Q (Zhao et al., ICLR 2025) 启发，吸收了 scaling factor absorption 和 kernel fusion 技术。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：Open-Sora 1.2 推理代码（基于 PyTorch + custom CUDA kernels）。论文未明确说明评估框架名称（论文未使用标准 serving framework）。
  - 修改内容：(1) 将 Open-Sora DiT blocks 中的标准 FP16 GEMM 层替换为 QuantCache 的量化 GEMM kernel（支持 W8A8 和 W4A6 低精度 GEMM）；(2) 在 kernel 内集成 online activation quantization（动态计算 min/max → 计算 scale → quantize → GEMM）；(3) 融合 rotation transformation 到量化 kernel（避免额外 kernel launch）；(4) 实现 HLC 缓存逻辑：在 kernel 输入侧检查 D_t^(l) 是否低于阈值，若是则从 cached buffer 直接读取 feature（存储在 GPU global memory 的 dedicated cache buffer）；(5) 实现 SRAP 剪枝逻辑：计算 S_t^(l,l+1) → 若超过阈值则跳过当前 kernel launch（kernel 调用侧逻辑）；(6) Speedup 在单张 A800-80GB 上测量 end-to-end latency（包含 VAE encode/decode + DiT denoising + 所有 quantization/caching/pruning 开销），100 timesteps。
  - 评估方式：测量 end-to-end video generation wall-clock time（从输入 prompt 到输出 512×512×64 frames 视频），speedup = baseline Open-Sora latency / QuantCache latency。论文未提供 per-kernel 级别的 profiling 数据或 roofline model 分析。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/JunyiWuCode/QuantCache（论文声明 code and models will be available）
  - Kernel 输入到性能输出全过程（以 Open-Sora DiT block Single Kernel Call with W4A6 为例）：
    1. **Kernel Launch 准备**：CPU 侧根据 timestep t 和当前 layer l → 读取 D_t^(l)（HLC 决策）→ 如果命中缓存 → 跳过 kernel launch，直接从 cache buffer 读取 output（HLC 缓存命中 path）。否则继续执行量化 GEMM kernel。计算 S_t^(l,l+1)（SRAP 决策）→ 如果 S > τ_high → 跳过当前 kernel launch（SRAP 剪枝 path）。
    2. **Kernel 输入**：FP16 输入激活 X ∈ R^{seq_len × d_model}（global memory） + 4-bit packed weights W̄ ∈ R^{d_model × d_ff}（global memory） + per-channel weight scales s_W（global memory） + offline fused scaling factors S_absorbed（已吸收到 W̄ 中） + rotation matrix R。
    3. **Kernel 内执行**（单次 kernel launch，fused）：(a) 从 global memory 加载 X tile 到 shared memory；(b) Online activation quantization: 在 shared memory 中计算 min(X_tile) / max(X_tile) → 计算激活 scale s_X = (max-min)/(2^6 - 1)（W4A6 配置）→ quantize: X̄ = clamp(round(X/s_X) + z_X, 0, 63) → 6-bit INT8-compatible representation；(c) 加载 4-bit packed W̄ tile → dequant to INT8: W_deq = W̄ × s_W_tile；(d) Rotation transform fused：X_rot = R @ X̄（轻量 rotation，O(d²)，在 shared memory 中完成）；(e) INT8 Tensor Core GEMM: Y = W_deq @ X_rot（利用 A800 Tensor Core INT8 算力）；(f) Dequant output: Y_FP16 = Y × (s_X × s_W)（fused output scaling）。
    4. **Kernel 输出**：FP16 输出激活 Y_FP16 ∈ R^{seq_len × d_ff} → 写回 global memory → 同时写入 HLC cache buffer（如果 D_t^(l) < δ_1，标记该 feature 为可缓存）。
    5. **性能输出**：end-to-end video generation latency = Σ(kernel launch overhead + kernel compute time + cache hit skip time)。总 speedup = 6.72×（包含所有 HLC cache hit、AIGQ low-bit compute、SRAP skip 的累积收益）。CUDA kernel fusion 使单项 kernel launch overhead 从 3 次（quantize + rotate + GEMM）降为 1 次。

## SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：基于OpenAI Triton实现的自定义attention kernel，融合INT8量化与FlashAttention-2风格tiling。核心技术：(1) 利用NVIDIA Tensor Core的INT8 mma(u8.u8.s32)指令加速$QK^\top$ Matmul（相比FP16有2-4×吞吐提升）；(2) 利用FP16-with-FP16-accumulator mma(f16.f16.f16)指令加速$PV$ Matmul（RTX4090/3090上相比FP32 accumulator有2×加速）；(3) Kernel Fusion — 将ROPE（Rotary Position Embedding）与Q/K量化融合，在ROPE结果写入global memory前完成量化，消除量化的IO开销；(4) 将系数$1/\sqrt{d}$在芯片上乘入Q后再量化（on-chip fuse），避免在attention kernel内额外操作；(5) 四个kernel变体（SAGEAttn-T/B/vT/vB）实现不同speed-accuracy tradeoff：T=per-token INT8 QK + FP16 PV, B=per-block INT8 QK + FP16 PV, vT=per-token INT8 QK + INT8 PV, vB=per-block INT8 QK + INT8 PV。
  - 实验比较：kernel speed vs FlashAttention2、xformers、Torch Attention（TOPS和GFLOPS，head_dim=64/128，sequence length 512~32768，w/wo causal mask）；real model speedup（Llama2, CogvideoX, UltraPixel, Unidiffuser, TIMM on RTX4090/3090）；quantization overhead（smooth K <0.2%）；adaptive quantization benefit（+11.7% OPS）。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA RTX 4090（Tensor Core INT8 throughput 660 TOPS理论值）；NVIDIA RTX 3090。RTX 4090服务器：PCIE 5.0, 16-core Xeon 6430 CPU, 120GB DDR4 RAM。RTX 3090服务器：16-core Xeon 8358P CPU, 80GB DDR4 RAM。
  - 软件栈: OpenAI Triton（triton-nightly 20240816版）→ PTX → NVIDIA Tensor Core指令。torch 2.4.0+cu121, python 3.11, gcc/g++ 9, Ubuntu 22.04。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估框架：自建Triton kernel benchmark + end-to-end模型推理脚本（基于HuggingFace Diffusers/Transformers加载模型，替换attention实现）。
  - 修改内容：
    - 实现Triton kernel：`sageattention.py`中实现forward kernel，调用`tl.dot()`利用Tensor Core mma指令
    - Fused ROPE + quantization：在ROPE kernel的epilogue中插入`ψ_Q`和`ψ_K`量化操作（`(δ, x̂) = quantize_int8(x)`）
    - kernel配置：block sizes b_q=128, b_kv=64；Num Warps=4/8 (headdim=64/128)；Num Stages=3/4/5
    - Adaptive quantization selector：对每层计算SAGEAttn-vB的cosine similarity，若>99.8%则选vB
  - 评估原理：(1) Kernel micro-benchmark: 对不同sequence length和headdim测量单个attention kernel的TOPS（Tera Operations Per Second），warmup后取多次平均；(2) 真实模型speedup：将模型中所有attention调用替换为SageAttention，测量attention部分latency和end-to-end latency speedup。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/thu-ml/SageAttention
  - Kernel输入到性能输出全过程（以SAGEAttn-B, headdim=64, RTX4090为例）：
    1. **Kernel Launch准备**：CPU侧将FP16 Q, K, V tensor（each ∈ R^{N×64}）从PyTorch传入Triton kernel。Kernel配置：grid=(T_m=N/128,), block size由Num Warps=4和Num Stages=3决定。Q block size b_q=128, KV block size b_kv=64。
    2. **Kernel输入**：FP16 Q ∈ R^{N×64}, K ∈ R^{N×64}, V ∈ R^{N×64}（HBM）。ROPE已在上一kernel完成并fused量化写入HBM：Q̂_INT8, K̂_INT8连同per-block scales δ_Q, δ_K 已就绪。
    3. **Kernel内执行**（单次Triton kernel launch，outer loop parallel on SMs）：
       (a) 从HBM加载Q̂_i[128×64] INT8 tile + δ_Q[i]到SRAM；
       (b) Inner loop j=1..T_n: 加载K̂_j[64×64] INT8 + V_j[64×64] FP16 + δ_K[j]到SRAM；
       (c) INT8 Tensor Core MMA: S_temp = tl.dot(Q̂_i, K̂_j^T) → INT32 accumulator → S_i^j = S_temp.to(FP16) × δ_Q[i] × δ_K[j]（dequant via scale broadcast）；
       (d) Online Softmax (FP16): m_i^j = max(m_i^{j-1}, rowmax(S_i^j)), P̃_i^j = exp(S_i^j - m_i^j)；
       (e) FP16 Tensor Core MMA with FP16 accumulator: ΔO = tl.dot(P̃_i^j.to(FP16), V_j.to(FP16), accum=FP16) → O_i^j = diag(e^{m_i^{j-1}-m_i^j})O_i^{j-1} + ΔO；
       (f) 循环结束：O_i = diag(l_i^{T_n})^{-1}O_i^{T_n}；
    4. **Kernel输出**：FP16 O_i[128×64] → 写回HBM。
    5. **性能输出**：TOPS = (2×N×d + 2×N²×d) / latency_μs × 10^{-6}。实测340 TOPS (headdim=64, N=8192, non-causal)，达到RTX4090 INT8理论峰值660 TOPS的52%。FlashAttention2对比：165 TOPS（FP16理论峰值330 TOPS的50%）。SageAttention 2.1× faster than FlashAttention2。

## Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization via Fine-grained Sparsification

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现：(1) **3:4 稀疏 5-bit 打包方案的 SIMD 高效推理 kernel**：将每 4 个三值权重（含恰好 3 个非零、1 个为零）打包为 5-bit 索引，利用 4-way 对齐模式对现代 CPU SIMD 向量通道友好，通过查表引擎（BitNet.cpp, T-MAC）替代浮点乘法为整数加法；(2) **与 BitNet I2_S（2-bit 打包）和 Tequila TL2（1.67-bit 打包）的运行时性能对比**。实验比较：(1) CPU 推理吞吐量（tokens/s, Intel i7-14700HX）；(2) 模型大小（MB, GGUF 格式）。

- 后端平台是什么，配置是什么。
  Intel i7-14700HX CPU，固定线程配置（AngelSlim 框架层面使用 2 threads）。查表引擎：BitNet.cpp（基于 ARM/x86 SIMD 的三值矩阵乘法 kernel）和 T-MAC（CPU 查表低比特部署 kernel）。推理使用 GGUF 格式。

- 评估性能的软件/脚本是什么。修改了什么。
  评估使用 AngelSlim 框架内置的性能测量脚本（论文未给出具体脚本名称），通过 llama.cpp 类推理引擎加载 GGUF 格式量化模型，测量 generation throughput（tokens/s）和模型大小。修改内容：Sherry 的 3:4 稀疏 5-bit 打包方案需要自定义 (de)packing kernel —— 在推理时将 5-bit 索引解码为三值权重 + 位置 mask，然后通过查表进行矩阵乘法。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/Tencent/AngelSlim（sherry 分支）

  **Sherry 推理 kernel 评估流程详解**：

  **Kernel 输入到性能输出全过程：**

  ```
  # ===== 离线阶段：模型打包 =====
  # 输入：训练好的三值权重矩阵 W ∈ {+1, 0, -1}^{N}，含 3:4 结构化稀疏
  # 处理：
  for each group of 4 consecutive weights (w_0, w_1, w_2, w_3):
      # 该组有 C(4,3) × 2³ = 32 种可能状态
      zero_pos = index of the single zero weight  # 4 种可能
      signs = sign of the 3 non-zero weights      # 2³ = 8 种可能
      packed_5bit = encode(zero_pos, signs)       # → [0, 31]  
      store packed_5bit in GGUF format

  # ===== 在线阶段：推理解包与计算 =====
  # 输入：packed 5-bit indices, scale factor α, activation vector X
  # 过程：
  for each group of 4 weights:
      # Step 1: Decode
      packed = load_5bit(idx)                    # 从 GGUF 内存读取 5-bit
      zero_pos, signs = decode(packed)            # 解出零值和符号
      # Step 2: Reconstruct ternary weights
      w_hat = [s_0, s_1, s_2, s_3] where s at zero_pos = 0  # 三值向量
      # Step 3: Multiply via LUT (Lookup Table)
      # 4-way 对齐天然适合 128-bit SIMD:
      # - 128-bit SIMD 处理 4 个 FP16 = 完美 1 组
      # - 256-bit SIMD 处理 4 个 FP32 = 完美 1 组
      y_g = α · (X[g*4]·w_hat[0] + X[g*4+1]·w_hat[1] + 
                 X[g*4+2]·w_hat[2] + X[g*4+3]·w_hat[3])
      # 或等效于查表: y_g = α · LUT[packed](X[g*4:g*4+4])

  # ===== 性能测量 =====
  # 输入：输入 tokens（seq_len=256~1024 for TTFT, output_len=p for generation）
  # 测量：
  # - Prefill latency (TTFT, ms): 从接收 prompt tokens 到生成第一个 token 的延迟
  # - Generation throughput (tokens/s): 每秒生成的输出 token 数
  # - Model size (MB): GGUF 文件在磁盘上的大小
  ```

  **SIMD 对齐优势（对比 BitNet 2-bit 和 Tequila 1.67-bit）：**
  - BitNet 2-bit: 4 权重 → 8 bits → 浪费 37.5% 存储，但 4-way 对齐 SIMD ✓
  - Tequila 1.67-bit: 3 权重 → 5 bits → 3-way pattern，512-bit SIMD = 10 group 余 2 权重 → 不完美对齐 ✗
  - Sherry 1.25-bit: 4 权重 → 5 bits → 4-way pattern，512-bit SIMD = 128 group 整除 → 完美对齐 ✓

  **评估结果（Table 3，Intel i7-14700HX）：**
  | Scale | Method  | Bits | Speed (t/s) | Size (MB) |
  |-------|---------|------|-------------|-----------|
  | 0.7B  | BF16    | 16   | 34.01       | 1360.0    |
  | 0.7B  | BitNet  | 2.0  | 132.13      | 256.56    |
  | 0.7B  | Tequila | 1.67 | 116.83      | 233.44    |
  | 0.7B  | Sherry  | 1.25 | 148.27      | 205.50    |
  | 3B    | BF16    | 16   | 7.55        | 6190.0    |
  | 3B    | BitNet  | 2.0  | 41.87       | 873.65    |
  | 3B    | Tequila | 1.67 | 38.80       | 846.01    |
  | 3B    | Sherry  | 1.25 | 45.55       | 712.40    |

## ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ResQ 使用 CUDA 11.8 + CUTLASS 实现混合精度推理 kernel，利用 TensorCore 执行 INT4 和 INT8 GEMM 操作。kernel 实现的分块矩阵乘法中，低精度（4-bit）和高精度（8-bit）操作数分别走各自精度的 GEMM kernel，通过 ResQ 的投影融合（U_A 融入权重）和运行时投影（U_C 8-bit 量化，U_D Hadamard 快速变换）保证正确性。实验比较 ResQ kernel 与 16-bit baseline 以及纯 INT4 kernel 在 NVIDIA RTX 3090 上单 decoder block 的性能加速比。

- 后端平台：NVIDIA RTX 3090 GPU（resource-constrained 场景代表）。CUDA 11.8 + PyTorch。

- 评估性能的软件/脚本：使用 CUTLASS (https://github.com/NVIDIA/cutlass) 实现 INT4/INT8 GEMM 在 TensorCore 上的运算。测试单 decoder block 在不同模型和序列长度下的延迟，对比 16-bit FP baseline 与纯 INT4 kernel。论文未提供具体 benchmarking 脚本。

- 开源情况：代码开源 https://github.com/utkarsh-dmx/project-resq。CUDA kernel 实现在开源仓库中，包含 CUTLASS 集成的 INT4/INT8 GEMM 调用。

- 评估原理和 kernel 输入到性能输出的全过程：
  1. **输入**：量化后的模型权重（W_q = Q_L(U_l^T·W) + Q_H(U_h^T·W)，已离线完成）和运行时激活 X。
  2. **前处理**：激活 X 通过已融入前一层权重的 U_A 自动投影；若为注意力块内，key/query 先经 U_C 显式投影（8-bit 量化）；若为 FFN 块内，经 U_D Hadamard 变换。
  3. **GEMM 执行**：调用 CUTLASS INT4 GEMM kernel 计算 Q_L(XU_l)·Q_L(U_l^T·W)；调用 CUTLASS INT8 GEMM kernel 计算 Q_H(XU_h)·Q_H(U_h^T·W)。两路结果在 INT32 累加器中求和得到输出。
  4. **性能测量**：在 RTX 3090 上运行，batch size=1，测量 decoder block 的总延迟（含 GEMM 和投影开销）。加速比：1.61× 到 3.03× 相比 16-bit baseline（含 Hadamard 变换开销）。相比纯 INT4 kernel 仅慢 14%。
  5. **结果**：更大模型和更短序列获得更高加速比。混合精度计算和运行时投影的额外开销较小。

## SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  设计了专门的GPU稀疏矩阵乘法kernel来处理SpQR格式中的unstructured outlier weights。核心实现：(1) 基于CSR（Compressed Sparse Row）格式存储outlier weights：将outlier按row-first, column-second排序，每个outlier存储16-bit value + 16-bit column index，每行一个32-bit cumulative row pointer；(2) 在推理时，warp-level load balancing：将权重矩阵划分为等大小blocks，每个thread block加载outlier slice到shared memory (SRAM)，每个GPU core判断outlier是否在其segment内，从DRAM加载对应权重值，执行有效行上的稀疏矩阵乘法；(3) 与dense-quantized matmul相结合：先用bilevel quantized weights做dense dequantize+matmul，再用CSR sparse kernel处理1% outliers的contribution。实验比较SpQR optimized kernel vs PyTorch sparse（cuSPARSE）vs FP16 baseline的token generation latency（tokens/s），在LLaMA-7B/13B/30B/65B上，batch size=1，分别测from scratch（100 tokens）和prefix 1024（扩展到1124 tokens）两种场景。

- 后端平台是什么，配置是什么。
  单张NVIDIA A100 GPU（80GB）。CUDA kernel为自研实现。PyTorch版本≥2.0.0 with CUDA support。

- 评估性能的软件/脚本是什么。修改了什么。
  评估脚本：`inference_demo.py`（SpQR源码仓库中的端到端推理脚本）。自研的SpQR CUDA kernel替换PyTorch默认的cuSPARSE实现。Kernel原理及修改：
  
  1. **Weight Layout变换**：quantized weights和量化统计量按block（β₁×β₂ = 256 weights）连续存储于DRAM，每个block包含256个packed 3-bit codes + 16 packed 3-bit scales/zeros + 4个FP16 second-level statistics。
  
  2. **Dense DequantMatmul Kernel**：Thread block加载当前block的统计量到SRAM → 第二层反量化(3-bit→FP16)→第一层反量化(3-bit→FP16) → 加载block内的packed 3-bit weights到SRAM → 逐weight dequantize到FP16 → 与SRAM中的activation vector执行点积 → 累加到output。
  
  3. **CSR Sparse Kernel（outlier处理）**：
     ```
     步骤1: 将矩阵划分为等大小blocks (tile)
     步骤2: 每个thread block加载一段outlier slice到shared memory (SRAM)
     步骤3: 每个GPU core遍历其tile内的rows:
             if tile包含该行的outlier:
                 从row pointer确定该行outlier range
                 加载列索引和对应FP16值
     步骤4: 执行sparse dot product: output[row] += Σ col_value[outlier] × activation[col_idx]
     ```
     通过步骤1-3实现load balancing，步骤4因outlier的row-wise pattern获得连续内存访问。

  4. **最终merge**：dense_matmul_result + sparse_outlier_result = final output。

  评估原理：在单张A100上，batch_size=1逐token生成，测量Scratch（从零生成100 token）和Prefix 1024（在1024-token prompt后追加100 token）两种场景下的tokens per second。结果显示SpQR optimized kernel相比FP16 baseline获得20-30%加速，比PyTorch稀疏+量化组合快约2倍。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/Vahe1994/SpQR

  推理全流程（以LLaMA-65B 4-bit SpQR在A100上逐token生成）：
  
  1. **模型加载**：SpQR量化后的模型包含：
     - Dense部分：packed quantized weights (3-bit/4-bit) + bilevel quantized scales/zeros
     - Sparse部分：CSR格式outliers (row_pointers[N+1], col_indices[num_outliers], values[num_outliers])
  
  2. **逐层推理**（每层Linear层）：
     ```
     Thread Block分配: 每block负责一段连续的output rows (tile)
     
     // Dense MatMul部分
     for each weight block in tile:
         加载 quantization statistics 到 SRAM
         反量化统计量 (second→first level)
         for each group in block:
             加载packed weights到SRAM
             反量化到FP16
             dot_product(weights, activation_segment) → partial_dense[tile]
     
     // Sparse MatMul部分
     加载outlier slice (row_pointers tile范围) 到SRAM
     for each row in tile:
         if row有outliers:
             遍历该行outliers:
                 partial_sparse[row] += value[k] × activation[col_idx[k]]
     
     output[tile] = partial_dense[tile] + partial_sparse[tile]
     ```

  3. **性能关键**：SpQR的token generation是memory-bound操作，高压缩率（3.4x+ memory reduction）降低了DRAM读取量，即使增加sparse compute开销，整体wall-clock time仍比16-bit推理少20-30%。

## SqueezeLLM Dense-and-Sparse Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  自研CUDA kernel实现两类操作：(1) **LUT-based非均匀量化矩阵-向量乘法**：对3/4-bit量化的dense权重矩阵，加载压缩bit indices → LUT查表获得FP16 centroid值 → FP16向量内积。每个output channel有独立LUT（如8个FP16 centroid对应3-bit），kernel按piece-by-piece方式dequantize以最小化内存带宽占用。(2) **Balanced CSR稀疏矩阵-向量乘法**：处理Dense-and-Sparse decomposition中的稀疏分量（CSR格式存储outliers+sensitive values）。由于sparsity pattern在各输出channel间高度不均衡（部分channel含大量nonzero），标准thread-per-row策略效率低。采用balanced hybrid kernel：按固定nonzeros/thread (10 nz/thread)分配工作，线程间额外同步但负载均衡。Dense和sparse kernel在单次launch中融合执行，避免中间结果叠加开销。

  实验比较：在A6000 GPU上对比FP16 baseline、GPTQ（non-grouped和grouped g128 with activation ordering）的延迟(s)和峰值内存(GB)，生成128和1024 tokens。A100上额外对比kernel-only matrix-vector runtime。关键对比项：dense-only (0% sparsity) vs 0.45% sparsity balanced kernel vs standard CSR kernel vs 0.45% sparsity。

- 后端平台是什么，配置是什么。
  NVIDIA A6000 GPU (48GB, primary latency benchmark)、NVIDIA A100 GPU (80GB, kernel-only matrix-vector runtime benchmark)。CUDA kernel实现，LUT-based dequant + balanced CSR SpMV。

- 评估性能的软件/脚本是什么。修改了什么。
  使用Torch CUDA profiler测量延迟和峰值内存。自研kernel代码开源在 https://github.com/SqueezeAILab/SqueezeLLM。修改/新增内容：
  - 新增3/4-bit LUT-based非均匀dequantization+矩阵向量乘CUDA kernel
  - 新增balanced CSR稀疏矩阵-向量乘kernel (10 nz/thread)
  - Dense+Sparse kernel融合launch

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/SqueezeAILab/SqueezeLLM (MIT license)。

  **Kernel推理全流程（以LLaMA-7B 3-bit + 0.45% sparsity, A6000 GPU, batch_size=1为例）**：

  **1. 权重数据加载到GPU内存**：
  ```
  对每个Linear层:
    Dense分量:
      - indices_3bit: uint8 packed array [in_features // packed_elements]
        将连续的3-bit indices pack进更大的整数类型以高效memory access
      - LUTs: FP16 array [out_features × 8]
        每个output channel的8个centroid值 (对应3-bit的8个量化级别)
    Sparse分量 (CSR格式):
      - row_ptrs: int32 [out_features + 1]    行边界偏移量
      - col_indices: int16 [nnz]               列索引
      - values: FP16 [nnz]                     稀疏权重值 (≈0.45% × 总参数)
  ```

  **2. LUT-based Dense Matrix-Vector Kernel**：
  ```
  // grid: (out_features / BLOCK_SIZE) blocks
  // block: BLOCK_SIZE threads
  
  __global__ void lut_dequant_matvec_kernel(
      const uint32_t* packed_indices,  // 3-bit indices packed
      const half* LUTs,                 // [out_c × 8] FP16 centroids
      const half* activation,           // [in_features] FP16
      half* output                      // [out_features]
  ) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row >= out_features) return;
  
      half* lut_row = LUTs + row * 8;  // 当前channel的8-entry LUT
      half acc = 0.0;
  
      // 逐块加载packed indices, LUT查表, 乘accumulate
      for (int chunk = 0; chunk < num_chunks; chunk++) {
          uint32_t packed = packed_indices[row * num_chunks + chunk];
          for (int j = 0; j < indices_per_chunk; j++) {
              uint8_t idx = extract_bits(packed, j * 3, 3);  // 提取3-bit index
              half w_deq = lut_row[idx];                     // LUT查表→FP16
              acc += w_deq * activation[global_col];         // FP16乘累加
          }
      }
      output[row] = acc;
  }
  ```
  关键设计：weight按块(而非一次性)dequantize以减少寄存器压力和内存带宽；所有算术在FP16完成。

  **3. Balanced CSR Sparse Matrix-Vector Kernel**：
  ```
  // 问题：标准CSR kernel (每线程处理一行)在行间nonzeros严重不均衡时效率低下
  // 解决：Balanced kernel (每线程固定10个nonzeros, 一行可由多线程合作处理)
  
  __global__ void balanced_csr_matvec_kernel(
      const int32_t* row_ptrs,
      const int16_t* col_indices,
      const half* values,
      const half* activation,
      half* output
  ) {
      // 按nonzeros总数分配线程: num_threads = nnz / 10
      int nz_start = threadIdx.x + blockIdx.x * blockDim.x * 10;
      int nz_end = nz_start + 10;
  
      half local_acc = 0.0;
      for (int nz = nz_start; nz < min(nz_end, total_nnz); nz++) {
          int col = col_indices[nz];
          half val = values[nz];
          local_acc += val * activation[col];
      }
  
      // 确定该nonzeros范围所属的行
      int row = binary_search_row(row_ptrs, nz_start);
  
      // Atomic add到output (同一行可能被多个线程更新)
      atomicAdd(&output[row], local_acc);
  }
  ```
  性能对比：Standard CSR kernel 0.45% sparsity → 3.9s (7B); Balanced kernel 0.45% sparsity → 1.7s (7B) (>2x faster)。

  **4. Fused Kernel Launch (Single Call)**：
  ```
  // Dense和Sparse kernel在单个CUDA stream中顺序launch
  // 但output buffer复用, 无需额外中间结果sum kernel
  cudaMemset(output, 0, ...);
  lut_dequant_matvec_kernel<<<grid, block>>>(...); // Y = D @ X (写入output)
  balanced_csr_matvec_kernel<<<grid, block>>>(...); // Y += S @ X (累加到output)
  ```

  **5. 评估原理和端到端性能 (A6000, 128 tokens, LLaMA-7B 3-bit)**：
  | Kernel配置 | Latency (s) | Mem (GB) | PPL (C4) |
  |-----------|-------------|----------|----------|
  | FP16 Baseline | 3.2 | 12.7 | 7.08 |
  | GPTQ 3-bit (no group) | 1.4 | 2.9 | 9.55 |
  | SqueezeLLM 0% sparse | 1.5 | 2.9 | 7.75 |
  | GPTQ 3-bit g128 (w/ reorder) | 13.7 | 3.0 | 7.89 |
  | SqueezeLLM 0.45% (standard CSR) | 3.9 | 3.2 | 7.56 |
  | **SqueezeLLM 0.45% (balanced)** | **1.7** | **3.1** | **7.56** |

  Key takeaway：
  - LUT-based dequantization overhead vs uniform quant: ~7% latency increase (1.4→1.5s)换来perplexity从9.55→7.75
  - Balanced sparse kernel将CSR overhead从>2x (3.9s)降至~13% (1.7s vs 1.5s)
  - GPTQ grouped kernel因activation ordering引发的scattered memory access导致严重降速(13.7s)
  - A100上的kernel-only benchmark: SqueezeLLM 3-bit达到1.5-2.5x speedup vs FP16 matvec kernel

## UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  UniQL实现了一个**融合RoPE kernel**（fused rotary positional embedding kernel），以及**INT4在线解包-剪枝-重打包**的运行时kernel。

  融合RoPE kernel：结构化排序破坏了RoPE的原始位置嵌入索引（因Q/K权重列的对称排序使通道顺序改变）。为最小化内存访问，UniQL将gathering和slicing rotary positional embeddings的索引收集操作融合到单个kernel中。排序时采用对称排序策略：将norm score向量s ∈ R^{D_hd}对半分为[s1, s2]，对s1+s2排序得到对称索引向量idx_sym = [argsort(s1+s2), D_hd/2 + argsort(s1+s2)]，只需存储一半索引（硬件高效）。RoPE kernel直接使用该对称索引从旋转位置嵌入表中gather对应位置，避免多次内存往返。

  INT4在线处理kernel：设备端部署4-bit量化模型后，运行时的INT4权重需要：在线解包（unpack from INT4 to computation format）→ 按指定剪枝率去除末尾通道 → 重新打包为INT32向量 → 送入矩阵乘法kernel。

  实验比较：有/无融合RoPE kernel的延迟对比（Table 9），以及UniQL vs TRT-AWQ和TAO-HQQ在A6000和Jetson Orin Nano 8G上的TPOT和TTLT延迟。

- 后端平台是什么，配置是什么。
  NVIDIA A6000 GPU（48GB显存）：云端推理，测量TPOT（time-per-output-token）和TTLT（time-to-last-token），配置1024 prefill + 1024 generation tokens。
  NVIDIA Jetson Orin Nano 8GB：边缘设备推理，统一内存架构，测量TPOT和TTLT，配置512 prefill + 512 generation tokens。
  在Nano 8G上FP16模型OOM无法运行，必须使用量化模型。

- 评估性能的软件/脚本是什么。修改了什么。
  Kernel基础实现改编自：Marlin 4-bit kernels（Frantar et al., 2024）和Liger-Kernel RoPE kernels（Hsu et al., 2025）。
  
  修改内容：
  1. **融合RoPE kernel**：在原有RoPE实现中加入索引gather逻辑。传统做法是先将排序后的索引向量传入，再从sin/cos表中分别取出对应位置再应用旋转——这需要多次global memory访问。融合kernel在单个CUDA kernel中完成gather + slice + RoPE旋转计算，减少10%延迟（1.1× speedup for 4-bit Llama-3.1-8B at 0%和25%剪枝，Table 9）。
  2. **INT4运行时处理**：在Marlin 4-bit kernel中增加在线通道剪枝功能——解包INT4权重后，按剪枝率去除末尾通道，重打包为INT32供后续矩阵乘法使用。这允许同一量化模型在不同设备负载下支持0%-35%可变剪枝率。

  延迟profiling：每配置运行20次测量（5次warmup后），报告平均TPOT和TTLT。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/enyac-group/UniQL
  
  Kernel执行流程（以Llama-3.1-8B的Q/K投影后RoPE为例）：
  ```
  # 输入：已排序的WQ' = WQ @ S_qk, WK' = WK @ S_qk
  # 对称排序索引：idx_sym ∈ R^{D_hd}, D_hd=128 for Llama-3.1-8B

  # 传统两阶段（无融合）：
  X_q = X_h @ WQ'                              # [T, D_hd]
  cos_gathered = cos_table[idx_sym]             # 从cos表中gather
  sin_gathered = sin_table[idx_sym]             # 从sin表中gather
  X_q_rope = cos_gathered ⊙ X_q + sin_gathered ⊙ rotate_half(X_q)
  # 以上需要3次global memory往返

  # UniQL融合kernel（单kernel完成）：
  # 在同一个thread block中：
  For each position t:
      For each half-dimension pair (2d, 2d+1):
          i = idx_sym[d]                       # 从寄存器中的对称索引
          cos_val = cos_table[t, i]            # fused gather
          sin_val = sin_table[t, i]
          x0 = X_q[t, 2d]; x1 = X_q[t, 2d+1]
          X_q_rope[t, 2d]   = cos_val * x0 - sin_val * x1
          X_q_rope[t, 2d+1] = cos_val * x1 + sin_val * x0
  ```

  评估原理（延迟profiling）：
  - **TPOT（Time Per Output Token）**：测量decode阶段每个生成token的平均耗时。用CUDA events记录每个decode iteration的开始和结束，取20次运行平均。
  - **TTLT（Time To Last Token）**：从prefill开始到最后一个token生成的端到端总耗时 = prefill时间 + TPOT × 生成token数。
  - **Orin Nano 8G上的完整推理流程**：加载4-bit UniQL模型 → 设备端按当前负载配置剪枝率（0%-35%）→ 在线解包INT4权重 → 去除末尾通道 → 重打包 → Run inference。W4A16模型在Nano上TPOT从TAO-HQQ的133.6ms降至77.2ms（Qwen-2.5-7B），35%剪枝进一步降至57.7ms（2.3× vs TAO-HQQ）。

