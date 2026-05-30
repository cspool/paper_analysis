## 82-COMET- Towards Practical W4A4KV4 LLMs Serving.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：FMPQ (Fine-Grained Mixed-Precision Quantization) 算法——首个实用的 W4A4KV4 后训练量化算法。核心设计：(1) **Block-wise mixed-precision quantization**：将激活张量沿 channel 维度划分为大小为 k=128 的多个 block（对应 A100 FP16 tensor core 的最小计算粒度 64×64×32 的整数倍），对每个 block 独立判断是否包含 outlier，正常 block 量化为 INT4，含 outlier 的 block 量化为 INT8，实现超过 84% 的 GEMM 以 W4A4 执行；(2) **Channel permutation**：通过校准数据集采样激活，定位 outlier channels，利用 channel permutation 将 outlier channels 聚集到少数 block 中，使需要 8-bit 量化的 block 占比降至 16% 以下；(3) **KV cache 量化**：利用 RoPE 和 softmax 的正则化特性，对 K cache 采用 channel-wise INT4 量化；V cache 本身 outlier 少，同样直接用 channel-wise INT4 量化；(4) **权重量化**：采用 OmniQuant 算法实现 4-bit 权重量化。FMPQ 整体配置为 W4AxKV4，无需重训练。
  - 实验比较：(1) Perplexity (WikiText2)：FMPQ vs SmoothQuant (W8A8) / GPTQ (W4A16) / AWQ (W4A16) / Omniquant (W4A16+W4A4) / QoQ (W4A8 KV4)，覆盖 LLaMA-1 (13B/30B/65B)、LLaMA-2 (7B/13B/70B)、LLaMA-3 (8B/70B)、Mistral-7B、OPT-13B、Qwen2-72B；(2) Zero-shot accuracy (PIQA/ARC-e/ARC-c/HellaSwag/WinoGrande)：FP16 vs SmoothQuant vs Omniquant vs QoQ vs FMPQ，LLaMA-3-8B 和 70B。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100-80GB-SXM4 (80GB HBM, 2.0TB/s bandwidth, 312 TFLOPS FP16 / 624 TOPS INT8 / 1248 TOPS INT4 tensor core)
  - CUDA 12.1
  - 单 GPU 环境

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA-1 (13B/30B/65B), LLaMA-2 (7B/13B/70B), LLaMA-3 (8B/70B), Mistral-7B, OPT-13B (及175B kernel only), Qwen2-72B
  - 数据集：WikiText2（perplexity），校准数据集来自 [45]（C4子集，数千token）
  - Benchmarks：PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande（zero-shot，使用 lm_eval [13]）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/rhmaaa/COMET-LLM（W4Ax kernel、Python接口和C++ API）
  - FMPQ 算法伪代码：
    ```
    # 输入：预训练LLM模型M，校准数据集D_calib，block大小k=128
    # Step 1: 权重量化（使用OmniQuant）
    for each linear layer in M:
        W_quant = OmniQuant(W, target_bits=4)
    
    # Step 2: 激活统计与Channel Permutation
    activations = sample(M, D_calib)  # 采样各层激活张量
    for each layer in M:
        outlier_channels = find_channels_with_outliers(activations[layer])
        # Channel permutation: 将outlier channels聚集到少数block
        perm = compute_permutation_to_cluster(outlier_channels)
        apply_permutation(W_quant[layer], perm)  # 权重对应重排
        apply_permutation_to_activation_path(layer, perm)
    
    # Step 3: Block-wise 混合精度激活量化
    for each activation tensor A with shape [B, L, H]:
        # 沿H维度按k=128分block
        for b in range(0, H, k):
            A_block = A[:, :, b:b+k]
            if contains_outlier(A_block):
                A_quant[b:b+k] = quantize_INT8(A_block)
            else:
                A_quant[b:b+k] = quantize_INT4(A_block)
    
    # Step 4: KV cache量化
    K_cache_quant = channel_wise_quantize_INT4(K_cache)
    V_cache_quant = channel_wise_quantize_INT4(V_cache)
    ```
  - 张量计算示例（LLaMA-1-30B, 某线性层 GEMM [B, L, H_in] × W^T [H_out, H_in]，激活channel数 H_in=6656）：
    1. 激活沿 H_in 维度划分为 6656/128 = 52 个 block
    2. Channel permutation 后，outlier 集中在约 8 个 block（~16%），这些 block 以 INT8 存储；其余 44 个 block 以 INT4 存储
    3. W4A4 GEMM：44 个 INT4 block 与对应 INT4 权重 block 直接使用 INT4 tensor core 计算（1248 TOPS）
    4. W4A8 GEMM：8 个 INT8 block 先经 CUDA core 快速 INT4→INT8 转换后，与对应权重以 INT8 tensor core 计算（624 TOPS）
    5. 两类计算结果通过 reduction 累加得到最终输出

## 81-Klotski- Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline .pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Expert-aware multi-batch pipeline paradigm——一种针对 MoE 推理的多 batch pipeline 调度算法。核心算法（Algorithm 1）：对 n 个 batch 组成的 batch group，在非 expert 层（attention/normalization）按 batch 顺序竖向计算并 prefetch 下一层；在 expert 层则按 expert 维度（而非 batch 维度）组织所有 token 的计算，即同一 expert 处理来自所有 batch 的所有 token。关键设计：(1) 仅 prefetch gate 和 K 个 hot experts 而非整个 MoE layer（减少 inter-layer bubbles）；(2) hot experts 优先计算，利用其高计算需求覆盖 cold experts 的 I/O 传输时间（减少 intra-layer bubbles）；(3) 每个 expert 完成所有 token 计算后立即 offload，不等整个 layer 完成（降低 GPU peak memory）；(4) Constraint-sensitive I/O-compute planner 通过不等式组(4)-(7)求解最优 n，考虑 attention/gate/expert 的计算时间和 I/O 时间的约束；(5) Correlation-aware expert prefetcher 使用预跑生成的 expert 激活路径频率表，基于前 l 层的 expert selection 预测当前层的 hot experts；(6) 可选 HQQ 量化（4-bit，group size 64）和 StreamingLLM sparse attention 集成。
  - 实验比较：(1) End-to-end throughput vs Accelerate / FastGen / FlexGen / MoE-Infinity / Fiddler，Mixtral-8×7B 和 Mixtral-8×22B 在不同环境和 batch size 下；(2) Throughput-latency trade-off 曲线；(3) GPU memory usage 在 prefill 阶段的变化（红线=最小所需内存，橙线=GPU memory limit，蓝线=全部 offload 后的内存使用，绿线=进一步利用剩余 VRAM 的内存使用）；(4) Ablation：逐步添加 multi-batch、hot expert only prefetch、expert computation reordering 和 quantization 的 throughput 变化；(5) Prefetch accuracy per layer：prefetched hot experts 实际参与计算的比例 vs 确实是该层 hot experts 的预测准确率；(6) n 和 batch size 对 throughput 的影响网格分析；(7) 实际 pipeline timeline 对比（profiler 数据分析）：simple overlap 单 batch vs Klotski 多 batch pipeline。

- 硬件平台是什么，配置是什么。
  - Environment 1：NVIDIA RTX 3090 (24GB VRAM) + Intel Xeon Gold 5318Y (256GB DRAM) + SSD (2TB, 1GB/s read, PCIe 4.0 x16)
  - Environment 2：NVIDIA H800 (80GB VRAM) + Intel Xeon Platinum 8470 (800GB DRAM) + SSD (1TB, PCIe 5.0 x16)

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8×7B (46.7B params, bfloat16) 和 Mixtral-8×22B (141B params, bfloat16)，均为 open-source MoE 模型，top-2 gating。
  - 数据集：输入从 wikitext-103 随机采样，expert correlation table 构建使用 wikitext-2。
  - Benchmark 指标：throughput (tokens/s)，latency，GPU memory usage。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未明确说明 Klotski 代码仓库链接。基础依赖 PyTorch、Hugging Face Transformers、FlexGen 均为开源。
  - 算法pipeline（Algorithm 1 核心逻辑）：
    ```
    for i in 0..l-1:                           # 对每个生成的 token
      for j in 0..n_layer-1:                   # 对每个 layer
        if layers[j] is not Gate:
          load(layers[j+1])                    # prefetch 下一层（非gate层触发）
        if layers[j] is Expert_Layer:
          load(c[i][j+1][0])                   # 加载下一个 batch 的 KV cache
          compute(layers[j])                   # experts处理所有batch的所有token
          store(h[i][j])                       # 存储 hidden state
          load(h[i][j+1][0])                   # 加载下一个 batch 的 hidden state
        else:                                  # 非expert层（attention/normalization）
          for k in 0..n_batch-1:               # 逐 batch 处理
            sync(load_cache_stream)
            load(h[i][j][k+1], c[i][j][k+1])   # 逐 batch 加载 hidden state 和 KV cache
            compute(layers[j][k])              # 逐 batch 计算
            sync(store_cache_stream)
            store(h[i][j][k], c[i][j][k])      # 存储 hidden state 和 KV cache
      sync(load_weight_stream)
    ```
  - 张量计算具体例子（Mixtral-8×7B, batch size=16, n=10, RTX 3090）：
    1. Attention layer 计算 10 个 batch 的 attention（约 2.6ms × 10），同时 I/O stream prefetch gate 权重和 2 个 hot experts 权重（约 21ms × 3 = 63ms）。因为 26ms < 63ms 但仅 prefetch gate+2 experts 而非全层 8 experts 使得不等式 `n*tc_A >= tIO_G + K*tIO_E` 成立。
    2. Gate 计算完成后，检查每个 token 的 top-2 expert selection。如果选中的 expert 不在已 prefetch 的 hot experts 中，立即在 expert transfer stream 上发起该 expert 的数据传输。
    3. Expert layer 按 expert 维度组织计算：所有 batch 中选中 expert 2 的 token 先一起计算（约 data_hot / 2 tokens × <1ms，因 hot experts 承担约 53.7% tokens），此时 expert 5、3、1 等在并行传输中。
    4. Hot experts 计算完成后，按传输完成顺序依次计算 cold experts。
    5. 下一个 MoE block 的 attention layer 多 batch 计算同理。

## 79-Anda_Unlocking_Efficient_LLM_Inference_with_a_Variable-Length_Grouped_Activation_Data_Format.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Anda数据格式——一种变长分组的Block Floating Point (BFP)激活数据格式，用于替换weight-only量化(W4A16)LLM中的FP16激活。Anda格式包含1-bit符号、5-bit共享指数（组内64个元素共享）、可变长尾数（1-16 bit，连续可调，针对不同LLM模块动态分配不同长度）。基于BOPs（bit operations）度量的自适应精度组合搜索算法：以LLM模型L、校准数据集D、精度损失容限δ、最大迭代次数N为输入，使用优先队列从激进组合[4,4,4,4]到保守组合[13,13,13,13]初始化搜索，每次迭代取出最小BOPs的组合评估精度，若满足精度约束则更新最佳组合，并生成邻近候选（将最佳组合中各维度分别减1，如[6,7,5,5]→[5,7,5,5], [6,6,5,5], [6,7,4,5], [6,7,5,4]），仅优化4个关键激活张量类型：A_qkv（Q/K/V投影）、A_o（输出投影）、A_u（上投影）、A_d（下投影）。无需重训练，复用weight-only PTQ的校准数据（数千token），迭代限制32次。
  - 实验比较：(1) 精度对比：FP16（全精度）、Omniquant W4A16g128（weight-only PTQ baseline）、FIGNA（无损BFP，14-bit尾数）、VS-Quant（激进4-bit BFP，无重训练直接应用）、Anda（0.1%/1%精度损失容限）在WikiText-2/PTB/C4三个数据集上的Perplexity和BOPs reduction。Anda在1%损失下BOPs reduction达2.44-3.31×（vs Omniquant 1.00×, FIGNA 1.23×, VS-Quant 4.00×但精度崩溃）；(2) 精度组合可视化：不同模型/数据集/精度容限下的最优4元组[M_qkv, M_o, M_u, M_d]；(3) 精度-性能trade-off：LLaMA-13B在0.1%-5%损失容限范围内的加速比和能效变化。

- 硬件平台是什么，配置是什么。
  - 模型精度评估：PyTorch + Hugging Face库，GPU未明确指定型号。
  - 硬件评估：SystemVerilog RTL + Cadence Genus综合@16nm工艺，285MHz，0.8V。
  - 内存：HBM2，3.9 pJ/bit，256 GB/s。

- 模型是什么。数据集和bench分别是什么。
  - 模型：OPT系列(1.3B/2.7B/6.7B/13B/30B)、LLaMA系列(7B/13B)、LLaMA2系列(7B/13B)。
  - 数据集/bench：WikiText-2、Penn Treebank (PTB)、C4，perplexity评估使用2048序列长度。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供代码开源链接。arXiv: https://arxiv.org/abs/2411.15982，MICAS项目页: https://micas.esat.kuleuven.be/research/topics/anda-unlocking-efficient-llm-inference-with-a-variable-length-grouped-activation-data-format。未找到公开GitHub仓库。
  - 算法pipeline例子（以OPT-6.7B, WikiText-2, 1%损失为例）：
    ```
    # Calibration: 从WikiText-2训练集采样128个随机序列(长度2048)，复用weight-only PTQ校准数据
    # Anda精度搜索伪代码:
    Q = PriorityQueue([4,4,4,4], [5,5,5,5], ..., [13,13,13,13])  # 初始等精度组合
    best_comb = null, best_bops = INF, visited = {}
    fp_acc = EvaluateAccuracy(L, D)  # FP16基线精度
    
    while iterations < 32:
        curr_comb = Q.pop_min_BOPs()  # [M_qkv, M_o, M_u, M_d]
        visited.add(curr_comb)
        anda_acc = EvaluateAccuracy(L, D, curr_comb)
        if EvalBOPs(curr_comb) < best_bops AND anda_acc >= (1-δ) * fp_acc:
            best_comb = curr_comb
            # 生成候选: 每维度减1
            neighbors = [decrease_dim(curr_comb, i) for i in 0..3]
            for n in neighbors:
                if n not in visited: Q.push(n)
        iterations++
    return best_comb  # 如 [7, 5, 5, 4]
    
    # 推理时BFP转换 (以A_qkv张量, M=7为例):
    # FP16 activation → 按64元素分组 → 取max exponent为共享 exponent
    # 各元素尾数右移 (max_exp - own_exp) → 截断到M=7 bit
    # 若全尾数为0 → 表示数值0
    # FP-INT GeMM: 64个Anda格式激活 (尾数7bit, 共享exp) × INT4 weight
    #   → INT bit-serial乘法 (mant×weight) → 移位 (共享exp)
    #   → 组内INT32累加 → 跨组FP32累加 → FP32转Anda写回
    ```

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MX8量化格式结合随机舍入（stochastic rounding）对SU-LLM的state矩阵进行压缩。MX8属于Microsoft MX block floating point格式的变体：每16个值共享一个8-bit共享指数（group exponent），组内每对值共享1-bit微指数（microexponent），每个值的符号+尾数共7-bit。State update操作中的decay（element-wise乘法）和update（element-wise加法）直接使用MX格式算术。MX乘法器：加group exponent、加microexponent（若溢出则右移尾数）、整数乘法单元处理sign+mantissa。MX加法器：比较exponent取max作为结果exponent、较小exponent的尾数右移对齐差值和microexponent、整数加法单元处理。量化在存储state到DRAM时施加，SPU读取MX8格式state直接执行MX算术，无需dequantization避免int8格式所需的scale乘除开销。随机舍入通过LFSR生成随机数加到mantissa实现，面积开销极小。
  - 实验比较：(1) 量化格式准确率对比：fp16、int8、int8SR、e4m3、e4m3SR、e5m2、e5m2SR、mx8、mx8SR在WikiText-2上的perplexity，涵盖LLaMA、OPT（transformer）和RetNet、GLA、Mamba-2（SU-LLM）；(2) 准确率-面积tradeoff：各格式在Mamba-2上的perplexity vs 面积开销散点图，MX8SR为Pareto最优；(3) 端到端准确率：Pimba（MX8SR）vs GPU（fp16）在WikiText-2/PIQA/Lambda/HellaSwag/ARC-E/ARC-C/WinoGrande上，geomean差距≤0.3%。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A100 80GB (small scale)，8×A100 via NVLink3 600GB/s (large scale)；也评估H100配置。PIM: 40 HBM2E-based PIM modules, 1,512MHz memory bus, SPU频率378MHz (4× tCCD_L)。

- 模型是什么。数据集和bench分别是什么。
  - 模型: RetNet 2.7B, GLA 2.7B, HGRN2 2.7B, Mamba-2 2.7B, Zamba2 7B, OPT 7B；scaled to 70B。
  - 数据集/bench: WikiText-2 (perplexity), PIQA, Lambda, HellaSwag, ARC-Easy, ARC-Challenge, Winogrande (accuracy)。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源: https://github.com/casys-kaist/pimba (full-system simulator + accuracy evaluation code); DOI: 10.5281/zenodo.16946084。
  - 算法pipeline例子: 对每个SU-LLM head的state更新，伪代码:
    ```
    # 原始fp16 state: S_{t-1} ∈ R^{dim_head × dim_state}
    # 量化: 按16值分组，计算group max exponent E_g = max_i(⌊log2(|x_i|)⌋)
    # mantissa = x_i / 2^{E_g - 6}  (保留7-bit)，pairs share 1-bit μexp
    # 存储为MX8: (group_exponent, [(μexp_01, mant0, mant1), ...])
    
    # State update (t-th token, single head, MX8 arithmetic in SPE):
    S_decay = MX_multiply(d_t, S_{t-1})          # d_t broadcast × state, element-wise
    outer = MX_outer_product(k_t, v_t)            # k_t v_t^T, using MX multiplier
    S_t = MX_add(S_decay, outer)                  # state update, MX adder
    y_t = GEMV(S_t^T, q_t)                       # dot product unit in pipeline stage 4
    ```
    MX乘法: result.exp = A.exp + B.exp; result.μexp = A.μexp + B.μexp (若=2则设为1并右移尾数1位); result.mant[i] = int_mul(A.mant[i], B.mant[i]) >> shift。
    MX加法: max_exp = max(A.exp, B.exp); align mantissas by right-shifting diff = max_exp - own_exp - own_μexp; result.mant[i] = int_add(aligned_A.mant[i], aligned_B.mant[i]); result.μexp = 0。

## 76-InstAttention_In-Storage_Attention_Offloading_for_Cost-Effective_Long-Context_LLM_Inference.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 SparF Attention，一种基于 SparQ Attention [54] 的 flash 感知稀疏 q-attention 算法，专门适配 NAND flash 存储特性。核心步骤：(1) 从当前 token 的 q 向量（1×dh）中选取 top-r 绝对值最大的通道索引 i（argtopk(|q|, r)），标识稀疏通道模式；(2) 按通道索引从 flash 加载 K[:,i]（仅加载含稠密通道的 flash page，通过双步加载机制：第一步 page 粒度过滤——若某 page 内所有 token 的对应通道均为稀疏则跳过整页；第二步 token 粒度过滤——NFC 从含混合 token 的 page 中只保留强弱 token）；(3) 用稀疏 K[:,i] 和 q[i] 计算近似注意力分数 ŝ（softmax(q[i]·K[:,i]^T) × ||q[i]||_1/||q||_1 归一化）；(4) 从 ŝ 中选 top-k 最大 token 索引 j；(5) 双步加载对应 K[j,:], V[j,:]（同样 page 级+token 级过滤）；(6) 计算精确注意力分数 s = softmax(q·K[j,:]^T/√dh)，输出 out = α·s·V[j,:] + (1-α)·v̄，其中 α=sum(ŝ[j]) 为近似分数之和，v̄ 为 V cache 的加权均值补偿被忽略 token。SparF 与 SparQ 的核心差异在于双步加载：步骤 4-5 和 14-15 将 KV cache 加载组织为 group（与 flash page 大小对齐，4KB=2048 FP16），先 page 粒度后 token 粒度过滤，避免为少量有效 token 读取整页，大幅减少 flash 随机访问和 write amplification。
  - 实验比较：(1) 稀疏准确率对比：SparF vs SparQ vs H2O vs Local Attention vs StreamingLLM 在 OPT-13B 和 Llama-2-7B 上，不同压缩比（1, 0.8, 0.6, 0.4, 0.2, 0.125, 0.1），数据集 SQuAD 和 TriviaQA；(2) 压缩比 sensitivity：1/8 压缩比下 SparF 几乎无损；(3) 端到端 throughput 中间接体现算法效果。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A6000 48GB VRAM, PCIe Gen4x16
  - CPU: Intel Xeon 5320 2.2GHz, 96GB DDR4
  - SSD: Samsung 980 Pro 2TB

- 模型是什么。数据集和bench分别是什么。
  - 模型: OPT-13B, OPT-30B, Llama-2-13B（论文也评估 Llama-2-7B 用于准确性实验）
  - 数据集/bench: ShareGPT, Wiki-Text-2, SQuAD, TriviaQA。输入/输出序列长度：OPT 1024/1024 tokens，Llama-2 2048/2048 tokens

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源: https://github.com/ChaseLab-PKU/InstAttention
  - SparF Attention 伪代码（dh=hidden dim, S=sequence length, r=top-r channels, k=top-k tokens）:
    ```
    # Input: q ∈ R^{1×dh}, K ∈ R^{S×dh}, V ∈ R^{S×dh}, v̄ ∈ R^{dh} (mean V)
    # Step 1: 选 top-r q 通道
    i = argtopk(|q|, r)  # i: r 个稀疏通道索引, shape (r,)
    
    # Step 2-7: 双步加载 K[:,i]
    for each group_i in channel_groups(i):  # group 对齐 flash page size
      if all entries in group_i are zero:    # SparF Filter-1: page级过滤
        skip this group
      else:
        load K[:, group_i] from flash        # 加载整页 (含稀疏token)
        filter out sparse entries in K[:, group_i]  # SparF Filter-2: token级过滤
    
    # Step 8: 计算近似注意力分数
    ŝ = softmax(q[i] · K[:,i]^T / √dh) * (||q[i]||_1 / ||q||_1)
    
    # Step 9-10: 选 top-k token
    mask = [1 if idx > S else 0]  # padding mask
    j = argtopk(ŝ + mask, k)     # j: k 个重要 token 索引, shape (k,)
    α = sum(ŝ[j])                 # 近似权重归一化因子
    
    # Step 12-17: 双步加载 K[j,:], V[j,:]
    for each group_j in token_groups(j):  # group 对齐 flash page size
      if all entries in group_j are zero:  # SparF Filter-3: page级过滤
        skip this group
      else:
        load K[group_j,:], V[group_j,:] from flash
        filter out sparse tokens            # SparF Filter-4: token级过滤
    
    # Step 18-19: 精确注意力计算 + 输出
    s = softmax(q · K[j,:]^T / √dh)   # shape (1, k)
    out = α * (s · V[j,:]) + (1-α) * v̄  # 加权补偿被忽略 token
    ```
    SparF 的核心是双步加载 (Filter-1/2, Filter-3/4)：Flash page 为 4KB = 2048 FP16 数，等于 16 个 token × 128 hidden channels。从 q 的 top-r 通道确定 K cache 中哪些列需读取（channel-indexed），从 ŝ 的 top-k token 确定哪些行需读取（token-indexed）。每步先判断 group 是否全稀疏（跳过整页节约带宽），若非全稀疏则加载整页后用 NFC 过滤掉稀疏条目。相比原始 SparQ（需 1.5× KV cache 因双方向索引），SparF 在 flash 上复制了 K cache 两份（token-indexed + channel-indexed 各一份），利用 flash 低成本容量换取访问效率，同时将压缩比设为 1/8 维持精度。

## 73-AdapTiV_Sign-Similarity_Based_Image-Adaptive_Token_Merging_for_Vision_Transformer_Acceleration.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出三种算法优化用于Vision Transformer (ViT)的image-adaptive token merging (TM)：(1) **Local Matching (LMatch)**：将TM搜索空间从随机两两比较O(N²)限制为仅与左侧和上方相邻token比较O(N)，利用图像空间局部性保证相邻token更高相似性；(2) **Sign Similarity**：替代cosine similarity，仅比较两个d维向量的符号位（sign bits）是否一致，使用d-bit XNOR操作替代d个n-bit乘法器。设向量a,b ∈ R^d，Sign similarity = Σ_{i=1}^d (1 if sign(a_i)=sign(b_i) else 0)。在ImageNet-1K上，Sign similarity与cosine similarity的correlation和mutual information均为0.95；(3) **Dynamic Merge Rate (MR)**：按左→右、顶→底顺序迭代扫描所有effective token（代表性token），对每个effective token执行LMatch+Sign similarity检查→相似则立即聚类合并。merge rate完全由图像内容决定（图像自适应），不预设每层合并比例，累计扩展合并cluster。所有优化不需额外训练或微调。
  - 实验比较：(1) 准确性：AdapTiV vs baseline (vanilla ViT) 在ViT-tiny/small/base/large、PiT-tiny/small/base、Swin-tiny/small/base上的Top-1 accuracy对比，ImageNet-1K，准确率损失<1%；(2) Merge rate动态范围box plot：0%–96.5% per-image merge rate variation；(3) Ablation study：逐方案拆解speedup来源（specialized datapath 2.9× → +Dynamic MR 5× → 总计~14.6× over edge GPU），TM energy consumption拆解（Brute-force+Cosine+No Scheduling baseline → +LMatch 3.7× energy saving → +Sign Similarity 2.7× energy saving → +Sign-Driven Scheduling 1327× energy saving）。

- 硬件平台是什么，配置是什么。
  - Edge CPU: 6-core ARM Cortex A78AE
  - Edge GPU: Nvidia Jetson Orin Nano (DDR4 2400MHz, 76.8GB/s)
  - Server CPU: Intel Xeon Platinum 8452Y
  - Server GPU: Nvidia RTX 6000 ADA (91 TFLOPS peak, 960GB/s DRAM BW)
  - AdapTiV ASIC: 16 lanes of 64-element MAC lines (PE Array) + VPU (VFU+SFU) + AdapTME, 16-bit fixed-point, 385KB on-chip SRAM (W/K/V buffer 128KB, Activation/Query buffer 128KB, Output buffer 128KB, SP 1KB), DDR4 2400MHz 76.8GB/s, synthesized @1GHz Samsung 28nm, 2.49mm², 11.06W. Server comparison scaled to 45 AdapTiV accelerators (peak 91 TFLOPS matched to RTX 6000 ADA).

- 模型是什么。数据集和bench分别是什么。
  - 模型: ViT-tiny (patch16-224/384), ViT-small (patch16-224/384), ViT-base (patch16-224/384), ViT-large (patch16-224/384) [Dosovitskiy et al.]; PiT-tiny/small/base (224) [Heo et al.]; Swin-tiny/small/base (patch4-224) [Liu et al.]。所有模型使用timm框架从HuggingFace获取预训练权重，off-the-shelf评估（无额外训练）。
  - 数据集/bench: ImageNet-1K (image classification)。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源: 论文未明确说明开源代码或RTL。
  - 算法pipeline例子（ViT-base-patch16-224, N=196 tokens, layer l）：
    ```
    # 输入: tokens X ∈ R^{N×d}, d=768 (embedding dim), image width W=14
    # TIM (Token Integration Map): 每个token entry含O_SC[1:0] + origin_addr
    
    # === Step 1: LayerNorm (VPU执行) ===
    for each token x_i ∈ X (按top-left→bottom-right顺序):
        μ_i = mean(x_i)           # VPU-VFU: element-wise sum ÷ d
        x_i' = x_i - μ_i          # VPU-VFU: element-wise subtract
        sign_i = sign(x_i')       # 提取sign bits (d=768 bits → SP存储)
        σ_i² = mean(x_i'^2)       # VPU-VFU: element-wise square + sum
        # LN继续与TM并行...
    
    # === Step 2: Token Merging (AdapTME执行, 与LN并行) ===
    for each effective_token t_k (按top-left→bottom-right顺序):
        col_k = k % W                         # 当前token列索引
        # SPMU查找left和above token的sign bits在SP中的物理地址
        left_sign = SP[SPMU.lookup("Left")]   # 列索引col_k-1的token
        abv_sign = SP[SPMU.lookup("Abv#%d" % col_k)]  # 同列最近above token
        
        # SSCU计算Sign similarity (与LN的x_i'减法完成后的sign bits stream同步)
        sim_left = Σ_{j=1}^d XNOR(curr_sign[j], left_sign[j])  # d个XNOR + PopCount
        sim_abv  = Σ_{j=1}^d XNOR(curr_sign[j], abv_sign[j])
        
        # 阈值比较 (≥threshold则判定相似)
        if sim_left ≥ thresh:   # O_SC[0] = 1 → 合并到left cluster
            # Early stop LN当前token, 更新TIM/SMPU/SP
            TIM[k].O_SC = 2'b10  # 标记为similar-to-left
            TIM[k].origin_addr = TIM[left_idx].origin_addr  # 共享origin
            # SP无需写入当前token (已merged)
            # 清理不再needed的SP block
        elif sim_abv ≥ thresh:  # O_SC[1] = 1 → 合并到above cluster
            TIM[k].O_SC = 2'b01
            TIM[k].origin_addr = TIM[abv_idx].origin_addr
        else:  # 不相似, unique token
            TIM[k].O_SC = 2'b00  # 保持独立
            TIM[k].origin_addr = k
            SP[col_k] = sign_i   # 写入SP作为新的comparison token
            # LN正常完成
    
    # === Step 3: Cluster aggregation ===
    # Prune merge: 每cluster仅保留origin_addr token, 其余prune
    # 输入从N tokens减少到N' effective tokens (N' = unique clusters数)
    
    # === Step 4: ViT backbone (PE Array) ===
    QKV = GEMV(X_reduced, W_qkv)    # PE Array MAC lines, 输入维度=N'×d
    Attn = Softmax(Q·K^T/√d_h)·V    # Self-attention on reduced tokens
    # TIM记录的cluster population用于Softmax rescaling保持accuracy
    FFN_out = GEMV(Attn, W_ffn1)    # FFN on reduced tokens
    X_out = GEMV(FFN_out, W_ffn2)
    ```
    关键的算法创新：LMatch将TMatch从O(N²)降至O(N)；Sign similarity用d个1-bit XNOR替代d个n-bit乘法器；Dynamic MR使merge rate完全由图像内容决定（同一模型同一层不同图像merge rate可0%–96.5%）；cluster population记录在TIM中用于后续Softmax的attention rescaling（参考ToMe[12]的proportional attention方法）以维持精度。

## 2-VQ-LLM_High-performance_Code_Generation_for_Vector_Quantization_Augmented_LLM_Inference.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：多种向量量化（VQ）算法用于LLM权重和KV cache压缩，包括QuiP#-4（weight，4-bit等价，vector_size=8, #Entry=65536*, Residual=2，lattice-based codebook）、AQLM-3（weight，3-bit等价，vector_size=8, #Entry=4096, Residual=2，additive quantization）、GPTVQ-2（weight，2-bit等价，vector_size=4, #Entry=256, Residual=1，product quantization）、CQ-4（KV cache，4-bit等价，vector_size=2, #Entry=256, Residual=1，coupled quantization）、CQ-2（KV cache，2-bit等价，vector_size=4, #Entry=256, Residual=1）。VQ算法的核心流程：原始d维向量→按vector_size分割为子向量→对子向量集合做k-means聚类（#Entry个簇），用簇心索引替代原子向量→计算残差（原始-最近簇心）→可重复residual quantization轮次（Residual参数控制）→聚类中心组织为codebook。Dequantization时按索引查codebook→残差结果按element-wise accumulation→concatenate所有sub-space结果。
  - 实验比较：(1) 各VQ算法配置与element-wise quantization方法对比：AWQ-4bit（权重）和QoQ-4bit（KV cache）在相同等价bit-width下的latency；(2) 开源VQ kernel实现（GC版本：codebook存global memory；SC版本：codebook存shared memory）的latency对比；(3) 消融实验：O1(hierarchical cache)→O2(register cache)→O3(codebook-centric dataflow)→O4(hierarchical fusion)各优化逐级叠加的效果。

- 硬件平台是什么，配置是什么。
  - NVIDIA RTX 4090 24GB（kernel level + end-to-end，主要实验平台）
  - Tesla A40 GPU（end-to-end evaluation，lower bandwidth场景，提供RTX 4090 67%的memory bandwidth）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama-7B, Llama-65B
  - Kernel shapes按Llama-7B和Llama-65B设定。GeMM用于prefill阶段、GeMV用于decode阶段权重、Attention (FlashDecoding)用于decode阶段KV cache。
  - Accuracy benchmark：arc-challenge task，通过LMEval framework评估

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - VQ-LLM框架本身：论文未明确说明独立开源仓库，但基于论文描述的实现框架。依赖的VQ算法各有开源：AQLM (https://github.com/vahe1994/AQLM), QuiP#, GPTVQ, CQ。
  - 以CQ-2配置（VQ<4,8,1>）下Attention (Decode)的VQ算法pipeline为例（张量计算流程）：
    1. **离线量化（Offline）**：对Llama-7B每层的KV cache（shape: [num_tokens, num_heads, head_dim=128]），按vector_size=4分割为32组4维子向量→对每组子向量做k-means聚类（#Entry=8）→每个子向量替换为3-bit索引（log2(8)=3）→量化后KV cache shape: [num_tokens, num_heads, 32]，每个元素为3-bit codebook index。
    2. **Codebook存储**：32组codebooks，每组含8个4维FP16 centroid向量。总codebook大小 = 32×8×4×2 bytes = 2KB per head。
    3. **在线Dequantization（Runtime per token）**：`for sub_space in 0..31: index = quantized_kv[token][head][sub_space]; centroid = codebooks[sub_space][index]; dequantized[sub_space*4 : (sub_space+1)*4] = centroid`。输出为FP16精度的full KV vector。
    4. **后续Attention计算**：`Q @ K^T / sqrt(d_k)` 对dequantized K进行，`softmax(...) @ V` 对dequantized V进行。

## 72-ALISA_Accelerating_Large_Language_Model_Inference_via_Sparsity-Aware_KV_Caching.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Sparse Window Attention (SWA) 算法——在每步LLM解码时，对每个attention head选择最重要的token作为稀疏KV tensors进行缓存和计算。具体：将缓存比r的token均分为k=⌊nr/2⌋个局部静态token和k个全局动态token。局部静态token = 最近k个token（保持语言顺序语义）；全局动态token = 对最近k步的attention权重沿head维度求和得到局部注意力总分S（每个被关注token的分数为其在最近k步中作为"/被关注token"的累计attention weight），取S中top-k个token。最终稀疏KV tensors从完整KV tensors中按indices I=[I_l, I_g] gather出来（K_s=K[I,:], V_s=V[I,:]），后续执行dense矩阵操作（QK_s^T/√d, softmax, AW·V_s）。与dense attention的区别：(1) caching ratio r控制每步保留的token比例实现KV sparsity；(2) gather操作将稀疏KV tensor打包为dense tensor，保持计算和访存规整。
  - 实验比较：(1) 准确率：SWA(80% KV sparsity) + KV Compression(INT8) vs dense attention / local attention (Longformer) / strided attention (SparseTransformer)，在Wiki-Text-2/Penn Treebank/Alpaca（语言建模perplexity）和PIQA/COPA/OpenBookQA/Winogrande（4-shot QA accuracy），涵盖OPT-6.7B/13B/30B、LLaMA-7B/13B/33B、Pythia-6.7B/12B；(2) Ablation：SWA w/o compression vs ALISA(SWA+Compression)准确率对比；(3) 注意力权重sparsity分析：不同KV sparsity下attention weight的实际sparsity变化。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA Tesla V100 (16/32 GB HBM) 和 NVIDIA H100 (80 GB HBM)。30B级别模型仅使用H100。
  - CPU: 2.60 GHz Intel Xeon, 128 GB DRAM。CPU-GPU带宽20 GB/s (PCIe)。

- 模型是什么。数据集和bench分别是什么。
  - 模型: OPT (6.7B, 13B, 30B), LLaMA (7B, 13B, 33B), Pythia (6.7B, 12B)。
  - 算法评估数据集: Wiki-Text-2, Penn Treebank, Alpaca (语言建模perplexity); PIQA, COPA, OpenBookQA, Winogrande (4-shot QA accuracy)。输入长度2048匹配最大context length。
  - 系统评估数据集: Alpaca (输入128, 输出512), batch size 4-64。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源: 论文未提供独立开源链接。实现基于FlexGen (https://github.com/FMInference/FlexGen) 和 HuggingFace Transformers。
  - SWA算法pipeline例子（以OPT-6.7B, r=0.4, d=4096, n为当前序列长度, h为hidden size）：
    ```
    # 输入: Q ∈ R^{1×d}, 完整 K ∈ R^{(s+n)×d}, V ∈ R^{(s+n)×d}
    # 前步attention weights AW_pre ∈ R^{n×n} (仅最近k步)
    
    k = floor(n * r / 2)  # r=0.4 → 保留40% tokens
    # Step 1: 计算局部注意力总分 (沿head维度reduce)
    # S[j] = sum_{i=n-k}^{n-1} AW_pre[i, j]  for j=0..n-1, 其中AW_pre[i,j]是第i步对token j的注意力权重
    S = AW_pre[n-k:n, :].sum(dim=(0,head))  # 形状: [n]
    
    # Step 2: 选择局部静态token（最近k个）
    I_l = [n-k, n-k+1, ..., n-1]  # 2k个中的k个
    
    # Step 3: 选择全局动态token（top-k S值）
    I_g = argmax_k(S[:n-k])  # 从非局部token中选
    
    # Step 4: 合并indices并gather稀疏KV
    I = concat([I_l, I_g])  # 总共2k个token
    K_s = K[I, :]  # 形状: [2k, d]
    V_s = V[I, :]  # 形状: [2k, d]
    
    # Step 5: 标准dense attention
    AW = softmax(Q @ K_s^T / sqrt(d))  # 形状: [1, 2k]
    Attn = AW @ V_s  # 形状: [1, d]
    ```

## 86-Oaken- Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Oaken的online-offline混合KV cache量化算法。核心包含三个算法组件：
    1. **Threshold-based Hybrid Grouping**（基于阈值的混合分组）：通过offline profiling（每模型一次性，~100次inference，~10分钟），对每个decoder layer统计KV cache值的分布，设定两个阈值T_lo和T_hi，将每个token的KV值分为三组——Outer group（top 4%，异常值/outliers，FP16存储）、Middle group（90%，中间值，4-bit INT量化）、Inner group（6%，近零值，sparse encoding填充为0）。
    2. **Group Shift Quantization**（组移位量化）：对Outer group的outliers应用shift操作缩小其动态范围，使用INT5量化（需5-bit）替代原始FP16（23 bit/entry），将average bitwidth从5.9降至4.8。
    3. **Fused Dense-and-Sparse Encoding**（融合稠密-稀疏编码）：将Middle group的4-bit量化值与Inner group的稀疏编码融合为8-bit对齐的稀疏矩阵（硬件高效且内存对齐），将average bitwidth从4.8进一步降至4.4（10% sparsity下）。
  - Insight驱动设计：(i) KV分布跨模型和decoder layer变化→per-model per-layer确定量化尺度；(ii) KV分布跨输入数据集一致→可使用共享量化尺度（offline profiling结果可复用）；(iii) KV分布存在channel-wise pattern的例外→需多组量化按magnitude分段。
  - 实验比较：FP16 baseline（vLLM）、KIVI [ICML'24]、KVQuant [NeurIPS'24]、QServe [MLSys'25]、Atom [MLSys'24]、Tender [ISCA'24]。评估指标：throughput (token/sec)、perplexity（WikiText2）、zero-shot accuracy（PIQA/WinoGrande/HellaSwag）。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100（312 FP16 TFLOPS, HBM 80/160GB, 2.0 TB/s bandwidth）
  - Oaken-HBM加速器（270 TFLOPS, HBM 80GB, 2.0 TB/s, TSMC 28nm synthesis）
  - Oaken-LPDDR加速器（270 TFLOPS, LPDDR 256GB, 1.1 TB/s, TSMC 28nm synthesis）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama2 (7B, 13B, 70B), OPT (6.7B, 13B, 30B), Mistral (7B), Mixtral (8x7B)。70B和30B模型使用2 GPU pipeline parallelism。
  - 数据集/benchmark：WikiText2（perplexity），PIQA、WinoGrande、HellaSwag（zero-shot accuracy）。Offline profiling使用WikiText2数据集。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/casys-kaist/oaken（KAIST CASYS lab），包含accuracy evaluation代码（Oaken、KVQuant、QServe、Tender、KIVI baselines的GPU量化模拟）。
  - 算法pipeline执行流程（以Llama2-7B单token的KV cache量化为张量计算为例）：
    1. **Offline Profiling**：`python3 oaken_preprocess_activation.py -m llama2 -s 7b -t wikitext -f 0.04 0.9 0.06 -o quantizer/oaken/llama2-7b.json`。对每层decoder layer l：用WikiText2跑~100次inference，收集所有token的K/V值→统计分布直方图→取top 4%（f_out=0.04）magnitude值作为Outer group阈值T_hi_l→取bottom 6%（f_in=0.06）作为Inner group阈值T_lo_l→剩余90%为Middle group→输出quantizer JSON：{layer_l: {T_lo_l, T_hi_l, scale_mid_l, scale_out_l}}。
    2. **Online Quantization per token**（张量计算）：
       - 输入：new token的K/V向量 v ∈ R^{d_head}
       - Step 1 - Group assignment：v_outer = v[v > T_hi_l]（FP16保留），v_inner = v[v < T_lo_l]（→0, sparse），v_mid = v[T_lo_l ≤ v ≤ T_hi_l]
       - Step 2 - Group Shift for outer：v'_outer = v_outer - shift_l（shift_l = mean of outer group），scale_out = max(|v'_outer|) / (2^4-1)，v_outer_quant = round(v'_outer / scale_out)，以INT5存储（5-bit index + 6-bit scale + 1-bit sign = 12 bit/entry）
       - Step 3 - Middle quantization：scale_mid = (T_hi_l - T_lo_l) / (2^4-1)，v_mid_quant = round((v_mid - T_lo_l) / scale_mid)，以INT4存储（4-bit index + shared 6-bit group scale = effective 4 bit/entry）
       - Step 4 - Fused encoding：将inner group稀疏mask与middle/outer group量化值融合为8-bit对齐格式：[6-bit idx | 1-bit group flag | 1-bit sign or 5-bit val] → 8 bit/entry total
       - Step 5 - 写入KV cache：encoded_kv = fuse_dense_sparse(v_mid_quant_4bit, v_outer_quant_5bit, sparsity_mask)，通过DMA写入Device Memory
    3. **Dequantization for Attention**：attention计算时，Dequant Engine从KV cache读取8-bit encoded数据→按group flag拆分→分别dequantize到FP16→送入MPU计算attention score。

## 20-PIM-DL- Expanding the Applicability of Commodity DRAM-PIMs for Deep Learning via Algorithm-System Co-Optimization.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PIM-DL框架中的eLUT-NN（enhanced LUT-NN）算法，将DNN中线性层（linear layers）的GEMM（矩阵乘法）替换为基于查找表（LUT）的计算。核心包含：
    1. **LUT-NN Conversion**（模型转换）：将预训练DNN的H×F weight矩阵转换为多个codebooks（centroids聚类）和pre-computed look-up tables (LUTs)。将激活矩阵沿H dim分割为1×V sub-vectors，在calibration数据集上对每列sub-vectors执行K-means聚类（每列含CT个1×V centroids），生成VH个codebooks。然后将weight matrix也按1×V子向量拆分，与codebooks做inner-product得到CT个F×VH大小的LUTs。
    2. **eLUT-NN Calibration**（增强LUT-NN校准）：在基础LUT-NN之上引入两个关键技术：(a) Reconstruction Loss——将各层computation approximation error的L2距离加入总体loss（L = Model Loss + β Σ||Â_l W - A_l W||²），实现direct gradient propagation至centroids，克服gradient vanishing；(b) Straight Through Estimator (STE)——用STE替代Gumbel-Softmax进行梯度传播，加速收敛。仅需<1%的calibration data（vs baseline需100%训练集）即可将全部linear layers替换为LUTs。
    3. **LUT-NN Inference**：推理时将输入activation matrix (N×H)分割为1×V tiles→每个tile与对应列的codebook做inner-product计算最小L2-distance→得到closest centroid index (argmin)→根据index从对应LUT中fetch F×1 vector→accumulate所有tile的结果得到F×N output matrix。FLOPs从GEMM的2×N×H×F降至约3×N×H×CT + N×F×VH（CT≪F），计算量减少3.66×~18.29×，multiplication仅占总操作量的2.9%~14.3%。
  - 实验比较：baseline LUT-NN [84]（accuracy对比）；GEMM-based inference on DRAM-PIMs（throughput/energy对比，22.57×/37.06×/27.25× speedup on PIM-DIMM/HBM-PIM/AiM）；CPU-based inference（FP32/INT8，up to 3.54× speedup）；GPU-based inference（V100，up to 1.20× speedup on AiM）。

- 硬件平台是什么，配置是什么。
  - **DDR4-PIM Platform**（真实硬件）：dual-socket Intel Xeon 4210 CPUs，128GB DDR4 Memory，8× UPMEM PIM-DIMMs（每DIMM 2 ranks×64 PEs = 1024 PEs total，64GB DDR4 PIM memory，43.8 GOP/s per DIMM）。Host operators用C++/OpenMP + GGML tensor library（AVX intrinsics），PIM operators用UPMEM SDK 2021.3.0。
  - **HBM-PIM Platform**（模拟）：NVIDIA A2 GPU host + 4× Samsung HBM-PIM Cubes（512 PEs，8GB HBM2 memory，1.2 TFLOPS per cube）。使用Samsung PIMSimulator评估。
  - **AiM Platform**（模拟）：NVIDIA A2 GPU host + 16× SK-Hynix AiM Chips（512 PEs，16GB GDDR6 memory，1 TFLOPS per chip）。使用扩展的PIMSimulator评估。

- 模型是什么。数据集和bench分别是什么。
  - 模型：BERT-base (hidden dim 768)、BERT-large (1024)、ViT-base (768)、ViT-huge (1280)。full-layer replacement（所有linear layers替换为LUTs）。
  - 数据集：GLUE benchmark（NLP，含MNLI/QQP/QNLI/SST-2/CoLA/STS-B/MRPC/RTE 8 tasks）、CIFAR-10 / CIFAR-100（CV）。
  - 超参数：V=2 or 4, CT=16, 𝛽=1e-3 (BERT) / 1e-4 (ViT), LR=1e-5 (BERT-large) / 5e-5 (others)。LUTs量化为INT8（≤0.1% accuracy drop）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/leesou/PIM-DL-ASPLOS (MIT License)，DOI: https://doi.org/10.5281/zenodo.10531532。
  - 算法pipeline详解（以BERT-base的FFN1层为例，输入shape N=64×512, H=768, F=3072）：
    1. **Calibration阶段**（Python3/PyTorch）：
       - 随机采样<1% training tokens → forward得到每层activation matrix A (M×H)
       - 每层沿H dim分V=4 columns → 每列K-means聚类CT=16个centroids（1×4向量）→ 生成VH=768/4=192个codebooks
       - Inner-product: Codebook[:,i] (16×4) × Weight_subvec[4] → LUT结果(16×1)，遍历VH=192次 → 最终CT=16个F×VH=3072×192的LUT矩阵
       - 校准：forward → H(A)用nearest centroid替换激活子向量（STE梯度估计） → backward计算Reconstruction Loss梯度 → 更新centroids
    2. **Inference阶段**（C++/PIM kernel）：
       - CCS operator（Host侧）：Activation (N×H) → 分割为N×(H/V)个1×V tiles → 每个tile与对应codebook(CT×V)做inner-product (GEMM: N×VH × VH×CT) → argmin得到index matrix (N×VH)，shape=N×192
       - LUT operator（PIM侧）：for each activation row i: for each sub-vector column j: idx = Index[i,j]; result[:,i] += LUTs[idx][:,j]（fetch F elements per LUT column, accumulate N×VH次） → output (F×N)
       - 其他算子（Add, Norm, GeLU, Attention）：可选的Host/PIM offloading，根据target PIM功能支持决定

## 29-ASADI_Accelerating_Sparse_Attention_Using_Diagonal-based_In-Situ_Computing.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：ASADI的DIA-based稀疏矩阵计算范式（DIA-based SpMM和SDDMM），包含两个核心软件组件：
    1. **DIA压缩方法**：(a) Classic Bubble-free DIA——针对静态稀疏注意力（如Longformer sliding window），对角线无零值（bubble），直接按对角线方向存储每列（column-of-diagonal），对角线窗口大小为ω；(b) Bubble-containing DIA——针对动态稀疏注意力（如Sanger量化剪枝），中心ω对角线区域用bubble-free压缩，ω区域外的非零元素移动到最近的bubble中对角线，并记录原始行坐标(Rd, Ro)。ω默认选n/8以平衡bubble数量和非零元素容纳。解压时：中心ω对角线按bubble-free方式解压，灰色元素按(Rd, Ro)列表恢复原始位置。
    2. **DIA-based计算范式**：(a) SpMM (S×V)——DIA格式的S矩阵按列（对角线方向）存储，每列元素天然对齐列坐标，直接与V矩阵做vector-vector multiplication，无需CSR格式所需的row-wise remapping。假设对角线平均5个元素/列 vs 行平均2个元素/行，DIA比CSR减少2.5×迭代次数，真实Longformer场景达7.5×；(b) SDDMM (Q×K^T)——DIA格式的mask矩阵M的每列DI（对角线索引）控制Q矩阵shift up/down（DI>0 down shift，DI<0 up shift），结合(Rd, Ro)列表处理灰色元素，然后做vector-vector multiplication。DIA仅需ω次迭代（每次含5个有效计算），而CSR需5次迭代（每次仅2个有效计算），真实场景节省7.5× latency。
  - 实验比较：
    - Baseline: PIM baseline (Samsung FIMDRAM HBM2, CSR格式存储S矩阵, Ramulator-PIM模拟)
    - 消融实验：(a) ASADI vs DIA-PIM（仅软件优化，DIA计算范式运行在PIM上），DIA-PIM平均1.3× speedup但受限于cross-bank transfer；(b) ASADI vs CSR-ASADI（仅硬件优化，CSR计算范式运行在ASADI ReRAM上），短序列时CSR-ASADI性能低于baseline（因CSR中大量bubble降低in-situ并行度）
    - 与其他加速器对比：GPU (NVIDIA RTX A6000), SPRINT (ASIC+in-memory pruning), CPSAA (crossbar-based PIM)
    - 对角线局部性影响：人工构造60%-10%对角线局部性的mask矩阵，评估BERT+GLUE/SQuAD性能退化
    - 稀疏度影响：6种剪枝阈值(1.5τ到4τ)在GLUE/SQuAD上的性能

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA RTX A6000, 46GB, 300W TDP, CUDA v11.6, PyTorch v2.0.0 (用于pre-processing和fine-tuning)
  - Baseline PIM平台: Samsung FIMDRAM (HBM2-based), 10GB HBM2 memory, 500MHz on-chip logic units/bank, CSR格式, Ramulator-PIM v2.0模拟器
  - ASADI: 全流式in-situ ReRAM加速器, 9.7GB ReRAM, 12 En-PE + 12 De-PE, MSL 8192, ω=MSL/8, OCI 1000GB/s (inner-Encoder), PCIe-6.0 128GB/s (cross-Encoder), 1GHz 1T1M ReRAM arrays, Float32精度, 1-bit per cell

- 模型是什么。数据集和bench分别是什么。
  - 模型: BERT-Base (BERT), BART, GPT-2-Small (GPT2) for NLP, ViL-Medium-Wide (ViL) for CV
  - 数据集: GLUE (cola, mnli, mrpc, qnli, qqp, rte, sst-2, stsb, wnli, MSL<384), SQuAD v1.1 (MSL 512), WikiText-2 (MSL 1K), IMDB (MSL 2K), ImageNet-1K (MSL 1K for ViL), Syn-4K (MSL 4K), Syn-8K (MSL 8K, 仅测latency/energy)
  - 动态稀疏采用Sanger [22]的quantize-and-pruning方法

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明ASADI代码是否开源。Pre-processing代码修改自Sanger的GitHub项目 (https://github.com/sanger-project)；模型和数据集从Hugging Face获取。
  - 算法pipeline执行流程（以BERT单层attention的SpMM计算S×V为例，n=6, d=2, ω=2, DIA格式已生成）：
    1. **Pre-processing (GPU端)**：(a) fine-tune模型得到weight matrices W_Q, W_K, W_V；(b) 运行quantize-and-pruning（Sanger方法）生成稀疏mask matrix M和稀疏score matrix S；(c) 将S压缩为DIA格式：选择中心ω对角线→bubble-free压缩→ω区域外非零元素移动到最近bubble→记录(Rd, Ro)列表→输出DIA格式S矩阵(n×ω)。
    2. **In-situ SpMM (S×V)**（ReRAM端）：
       - Step 1 Mapping: V矩阵每维d_i存入独立ReRAM array Arr_i ∈ R^n。DIA格式S矩阵均分到d个arrays，每array存ωd个对角线向量。
       - Step 2 vector-vector multi: For all arrays in parallel: I = S_col × V_row（S的对角线向量与V的对应行做element-wise乘）→写结果到I矩阵的新列。
       - Step 3 Transfer & repeat: S的对角线向量轮转到下一个array（DI_{-1}→Arr_1, DI_0→Arr_0, etc.），重复vector-vector multi，累加到I矩阵。
       - Step 4 Decompress: 按图7(d)(e)解压I矩阵——中心ω列按bubble-free解压，灰色元素按(Rd, Ro)列表定位→移回原始行。
       - Step 5 Accumulation: 对各array的I矩阵做逐元素加法 Z = Σ I，得到输出矩阵Z ∈ R^{n×d}。
    3. **In-situ SDDMM (Q×K^T)**（ReRAM端）：M的DIA格式(M ∈ R^{n×ω})逐对角线处理——对每个DI：memory controller shift Q up/down（DI=DI_j则Q shift down j×, DI=DI_{-j}则Q shift up j×）→执行(Rd, Ro) memory copy处理灰色元素→vector-vector multi SlicesS_i = Q × K→恢复被修改的Q行→下一DI迭代。最后transfer各SlicesS_i到同一array并累加得DIA format S矩阵。

## 38-SpecInfer- Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SpecInfer提出tree-based speculative inference and verification算法，核心包含三个算法组件：(1) **Expansion-based token tree construction**——从单个SSM的每步top-k tokens展开构造token tree。Observation: 当SSM与LLM的top-1 token不匹配时，LLM的token通常在SSM的top-5范围内（greedy decoding成功率从70%提升至89%，stochastic从57%提升至97%）。采用静态expansion configuration ⟨k₁,k₂,...,kₘ⟩，第i步对每个token展开kᵢ个子token。(2) **Merge-based token tree construction**——使用adaptive boosting无监督地collective boost-tune多个SSM。具体流程：用通用文本语料（OpenWebText Corpus）提取prompt，LLM生成token序列；逐SSM fine-tune，每次用上一个SSM失败（输出与LLM不一致）的prompt samples fine-tune下一个SSM；所有SSM以data parallelism分布在不同GPU，推理时merge所有SSM的输出为统一token tree。(3) **Multi-step Speculative Sampling (MSS)**——用于stochastic decoding的token tree verification算法（Algorithm 2 VerifyStochastic）。核心逻辑：对每个tree node u，遍历其子节点集合ℋ，对每个SSMᵢ预测的token x_s，以概率min(1, P_LLM(x_s|u)/P_SSM_s(x_s|u))接受；若接受则前进至x_s；若拒绝则normalize residual分布（norm(max(0, P_LLM - P_SSM_s))）并尝试下一个SSM。Theorem 4.2证明MSS生成的输出分布与LLM原始incremental decoding等价（P_SpecInfer(u_i|U) = P(u_i|U;Θ_LLM)）。Theorem 4.3证明MSS的rejection概率uniformly lower than naive sampling。
  - 实验比较：(a) 不同token tree width（1/2/3/4/5）对每step平均verified tokens数的影响（CDF图，Alpaca数据集，LLaMA-7B + LLaMA-68M）；(b) 不同tree width下的end-to-end per-token latency（LLaMA-7B，BS=1/2/4/8/16）；(c) MSS vs Naive Sampling的verified tokens数对比（5个数据集，tree width=5, depth=8）；(d) 与vLLM、HuggingFace TGI、FasterTransformer的分布式inference latency对比；(e) 与FlexGen的offloading-based inference对比；(f) SpecInfer自身的incremental/sequence-based/tree-based三种模式消融。

- 硬件平台是什么，配置是什么。
  - 2× AWS g5.12xlarge instances，每台4× NVIDIA A10 24GB GPU, 48 CPU cores, 192 GB DRAM。节点间100 Gbps以太网。
  - 单GPU场景（LLaMA-7B）：1× A10。多GPU场景：4× A10用于OPT-30B（1节点），8× A10用于LLaMA-65B（2节点）。
  - Offloading场景：单A10 GPU + CPU DRAM offloading。

- 模型是什么。数据集和bench分别是什么。
  - LLMs: LLaMA-7B, OPT-13B, OPT-30B, LLaMA-65B
  - SSMs: LLaMA-68M, OPT-125M（均为同模型家族的预训练小模型，可直接用作SSM）
  - 模型来源：HuggingFace repositories（huggyllama/llama-7b, huggyllama/llama-65b, facebook/opt-13b, facebook/opt-30b, JackFram/llama-68m, facebook/opt-125m）
  - Datasets: Chatbot Instruction Prompts (CIP), ChatGPT Prompts (CP), WebQA, Alpaca, PIQA。仅使用prompts/questions部分，每prompt生成最多128 tokens。
  - SSM boost-tuning corpus: OpenWebText Corpus

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/goliaro/specinfer-ae（Apache License v2.0），含完整speculation和verification算法实现。
  - Tree-based speculative inference算法pipeline（以expansion-based, greedy decoding为例，LLaMA-7B + LLaMA-68M, expansion config ⟨1,1,3,1,1,1,1,1⟩）：
    1. **Input**: prompt tokens ℐ = [t₀, t₁, ..., tₙ]
    2. **SSM Expansion**（每step的张量计算）：
       Step 0: input ℐ → SSM forward → logits ∈ R^|V| → top-1 token t̂₁。Tree root: t̂₁。
       Step 1: input [ℐ; t̂₁] → SSM forward → top-1 token t̂₂,₀。Tree node: t̂₂,₀ as child of t̂₁。
       Step 2: input [ℐ; t̂₁; t̂₂,₀] → SSM forward → top-3 tokens {t̂₃,₀, t̂₃,₁, t̂₃,₂}。3个child nodes of t̂₂,₀。
       Step 3-7: 每个分支各取top-1继续展开。
    3. **Token Tree Structure**: 最终token tree 𝒩 的nodes = {t̂₁, t̂₂,₀, t̂₃,₀, t̂₃,₁, t̂₃,₂, ...}，每个node通过parent指针表示其token序列。
    4. **Tree-based Parallel Decoding**（LLM forward pass）：
       将𝒩所有节点tokens + prompt tokens batch为一个输入 → LLM GPU执行单次forward pass（使用topology-aware causal mask）。
       Q = X @ W_Q, K = X @ W_K, V = X @ W_V
       A = softmax(mask_tree(Q @ K^T / sqrt(d)))
       其中 mask_tree(j,k) = 0 if k是j在token tree中的祖先, else -∞
       输出 𝒪 = {𝒪(u) | u ∈ 𝒩}，每个𝒪(u)是LLM对序列S_u的next-token概率分布。
    5. **Greedy Verification**（VerifyGreedy, Algorithm 2）：
       u ← root; 𝒱 ← []
       while ∃v s.t. p_v = u AND token(t_v) = argmax(𝒪(u)):
         𝒱.append(token(t_v)); u = v
       𝒱.append(argmax(𝒪(u)))  // 最后一步不匹配，附加LLM输出
       return 𝒱
    6. **Output**: verified tokens序列追加到原prompt后，若遇到<EOS>则终止。
  - MSS的采样等价性（Theorem 4.2核心）：通过rejection sampling保证 P_SpecInfer(uᵢ|U) = P(uᵢ|U; Θ_LLM)，即通过调整接受概率和residual distribution normalization，MSS的多步多分支验证输出分布与LLM incremental decoding完全一致。

## 39-8-bit Transformer Inference and Fine-tuning for Edge Accelerators.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：(1) **全操作8-bit量化推理**（Post-Training Quantization, PTQ）：对Transformer中GEMM之外的所有操作（residual addition, layer normalization, softmax/GeLU activation, attention scaling）进行FP8和Posit8量化，并通过**operation fusion**将element-wise操作融合进前序GEMM以减少量化误差，无需scale factor。(2) **8-bit LoRA微调**：将LoRA的低秩矩阵(B, A)在乘法前量化为8-bit（Posit8或FP8），并将LoRA weights(α·B·A)与8-bit pretrained weights融合并重新量化到8-bit，使所有GEMM操作均用单一8-bit数据类型执行：(formula: h = quant(W0_8 + α·quant(B16)quant(A16))x)。forward pass用E4M3、backward pass用E5M2。(3) **Posit近似Softmax**：利用posit的tapered precision特性，用bitwise operations近似sigmoid函数（最高位取反+右移2位），并结合posit reciprocal（bitwise XOR with negated signmask）构造指数函数：e^x = 1/S(-x) - 1。通过threshold截断（x < θ= -3时输出0）和shift优化（subtract ε = -1.188），将近似误差降至 <1%。(4) **Per-tensor scaling**：训练时对activation gradients做per-tensor scaling，amax scaled to 64（Posit8）来覆盖gradient范围。
  - 实验比较：(a) 不同操作融合级别（No Fusion → +Attn Scaling → +Activation → +LayerNorm → +Residual）下Posit8 vs E4M3 vs BFloat16的accuracy，MobileBERT/BERT/DistilBERT在SQuAD v1.1上的F1 score；(b) Whisper/GPT-2/LLaMA 2在大模型上的WER/Perplexity，包含Posit(8,2)变体；(c) 8-bit LoRA (Posit8/FP8/BFloat16)在GLUE(MNLI/QNLI/MRPC/SST-2)和SQuAD上的accuracy对比；(d) 近似softmax中threshold θ和epsilon ε的消融实验；(e) 8-bit fine-tuning memory reduction（~3×）。

- 硬件平台是什么，配置是什么。
  - 软件实验：CUDA-enabled NVIDIA GPU（RTX 4090 24GB用于主实验，部分需32GB VRAM用于LLMs如LLaMA 2 7B/13B）
  - 硬件合成：Design Compiler 40nm工艺，0.9V电压，评估频率范围：Relaxed (5 MHz) / Typical (200 MHz) / Aggressive (400 MHz)

- 模型是什么。数据集和bench分别是什么。
  - 模型：MobileBERTtiny (15M)、MobileBERT (25M)、DistilBERTbase (66M)、BERTbase (109M)、BERTlarge (334M)、Whispertiny (39M)、Whispersmall (244M)、Whisperlarge (1.5B)、GPT-2 Large (762M)、GPT-2 XL (1.5B)、LLaMA 2 (7B, 13B)、RoBERTabase (125M)、RoBERTalarge (355M)
  - 数据集和benchmark：SQuAD v1.1 (question answering, F1 score)、GLUE (MNLI, QNLI, MRPC, SST-2, accuracy)、LibriSpeech (speech recognition, WER)、WikiText-103 (language modeling, perplexity)

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/jeffreyyu0602/quantized-training (MIT License)
  - 以MobileBERT在SQuAD上的8-bit LoRA Posit8微调算法pipeline为例（张量计算流程）：
    1. **模型加载**：从HuggingFace加载pretrained MobileBERT权重W0，各层均转为Posit8格式存储(8-bit per weight, shape与原始一致)。LoRA矩阵A和B初始化为BF16（rank=8, A: [hidden_dim, rank], B: [rank, hidden_dim]），仅训练LoRA参数(0.3M vs 25.3M total)。
    2. **Forward Pass (E4M3量化)**：`x_quant = quantize_e4m3(x)`对每层输入量化。`h_ge_lo = B16 @ A16 → quantize_posit8(result) → merge with W0_posit8`。`h = fused_ge_linear(h_ge_lo, x_quant)`：先执行Posit8 GEMM (W_merged_8 @ x_8)，然后在32-bit accumulator中融合residual add + layer normalization + activation + attention scaling（operation fusion），最后将结果量化回Posit8。`softmax_output = posit_approx_softmax(attention_scores)`使用近似指数和reciprocal: `s = approx_sigmoid(x) via bitwise NOT+shift; exp = 1/s - 1; scaled = exp - max(exp); output = scaled / sum(scaled) via approx_reciprocal`。Scale factor=1（inference不需要scaling）。
    3. **Backward Pass (E5M2量化)**：`grad_output`先经per-tensor scaling（amax→64 for Posit8; amax→57344 for E5M2）。Approx softmax backward使用revised gradient formula：`∂σ(z)_j/∂z_i = σ(z)_j + exp(z_j)·f'·exp(z_i)` for i=j；`exp(z_j)·f'·exp(z_i)` for i≠j，其中f' = -2^(-floor(log2(Σexp(zk)))·2-1)。Gradient计算用E5M2格式GEMM。Optimizer (AdamW或SGD)更新仅应用于LoRA参数A和B（BF16精度）。
    4. **内存节省**：权重: 8-bit (1/2 of BF16)。Activation: 8-bit (1/2 of BF16)。Optimizer states: 仅对0.3M LoRA参数(约1.2% of total)。总训练内存从BF16的~500MB降至~165MB（约3× reduction）。

## 40-Fractal- Joint Multi-Level Sparse Pattern Tuning of Accuracy and Performance for DNN Pruning.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  Fractal是一个面向DNN剪枝的多级稀疏模式自动调优系统。算法核心：(1) 将稀疏模式抽象为循环打孔（loop perforation），提出PatternIR表示多级稀疏tiling——穿孔外层循环跳过整列（coarse-grain），穿孔内层循环跳过单个元素（fine-grain），通过组合多层穿孔形成Hybrid模式；(2) 基于重要性分数（默认magnitude）在操作符级别精度感知搜索，以unstructured pruning的magnitude loss作为各算子的精度上界阈值；(3) 多级贪婪剪枝从coarse-grain到fine-grain迭代执行，每级按选择区域（Selection Region）独立排序；(4) ML cost model（双向LSTM）预测候选模式延迟以加速搜索。实验比较Fractal与cuBLAS（dense）、cuSPARSE BlockELL（BW）、TileWise（TW）、cuSPARSELt（VW）、SparTA（EW）、Sputnik（EW）、Triton-BW在50%/75%/93.75%稀疏率下的单算子加速比和BERT-base/large模型级F1-加速比Pareto前沿。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 (80G) GPU 和 NVIDIA RTX-1080Ti GPU，Intel Xeon E5-2620 v3 @ 2.40GHz CPU。软件栈：CUDA-11.7、TVM-0.12.0、SparseTIR。

- 模型是什么。数据集和bench分别是什么。
  模型：BERT-base、BERT-large（operator benchmark和模型精度）、VGG、ResNet（仅operator benchmark）。数据集：MRPC语义分类（模型精度评估）。Benchmark算子：12个代表性GEMM/SpMM形状，覆盖Transformer attention（768/768/512、128/576/784等）、ResNet/VGG卷积im2col转GEMM后的形状（64/576/3136、256/1152/3136等）、大GEMM（4096/4096/4096等）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  Fractal自身未明确开源。底层依赖SparseTIR（开源 https://github.com/microsoft/SparseTIR）和TVM（开源）。算法pipeline伪代码（Alg.1）：
  ```
  Input: Dense Operator OP, Tuner Config Config
  Output: Sparse Pattern Sch
  1. Schs = GenTilingSpace(OP)  // 从dense PatternIR生成tiling组合
  2. CachedScores = Config.pruner(sparsePatterns) // 缓存重要性分数
  3. Schs = GenPerforationSpace(Schs) // 对所有tiling可能做Perforate(loop, nnz)
  4. Scores = Config.pruner(Schs, CachedScores) // 评估各模式精度影响
  5. Schs = FilterByScore(Schs, Scores, Config.max_score) // 阈值过滤
  6. Latencies = CostModel(Schs) // BiLSTM预测延迟
  7. Sort by predicted latency
  8. for j in Config.search_num:
  9.     SparseOP = CodeGen(Schs[j]) // PatternIR→SparseTIR→TVM IR
  10.    SparseOP = OperatorTuner(SparseOP) // MetaSchedule调优
  11.    Latency = Exec(SparseOP)
  12.    if Latency > Config.latency_limit: continue // 早停
  13.    for k in Config.tune_iteration: SparseOP = OperatorTuner(SparseOP)
  14.    Latency = Exec(SparseOP)
  15.    if Latency < BestLatency: Sch = Schs[j]
  ```
  多级剪枝过程（Sec 4.2）：给定PatternIR含n个I维稀疏循环和m个J维稀疏循环，模式尺寸W=第t个循环后继所有循环长度之积，选择区域Q=第t个及后继循环长度之积。剪枝从外到内（coarse→fine）贪婪执行，每级在各选择区域内按magnitude独立排序保持均匀稀疏率。例如BERT-base某层的PatternIR `K08I03K1712K278I1256`表示4级稀疏模式：最外层在K0上8选3、次层在I0上3选?...、第三层K17上12选?、第四层K278上?，最终体现为多级Hybrid稀疏。

## 42-Cambricon-D_Full-Network_Differential_Acceleration_for_Diffusion_Models.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  实现基于时序差分计算（temporal differential computing）的扩散模型inference加速方法。核心算法：(1) 将每时间步的激活Xt表示为上一时间步Xt-1与delta值ΔXt之和（Xt = Xt-1 + ΔXt），对delta而非raw值执行卷积（Conv(ΔXt) 而非 Conv(Xt)），利用线性卷积的性质（Conv(Xt) = Conv(Xt-1) + Conv(ΔXt)）恢复最终结果；(2) delta值数值范围远小于raw值（FP16→INT3，entropy降低2.11×），无需增加量化误差即可用更低bitwidth表示；(3) 用ReLU近似SiLU（实验中<0.5%精度损失），并利用sign-mask近似（sgn(Yt) ≈ sgn(Yt-1)，在Stable-Diffusion中99.59%情况下成立）实现全网络差分——差分ReLU: ΔYt' = ΔYt · sgn(Yt-1)（AND mask操作）；(4) Group Normalization的差分近似：每时间步独立计算μG和σG²经验值，使用相邻两时间步平均值；(5) Outlier-aware结构化量化：激活tensor沿inner product维度等分group，每组最多m个FP16 outlier，其余INT3 inlier，超出m的outlier clip到INT MAX/MIN作inlier处理（概率<1%）。实验比较：(a) 量化精度：Cambricon-D vs FP16 vs INT8 vs Diffy-like量化（窄PE+丢弃leading zeros）——Diffy方案精度严重退化（GUID512: 87%→43%），Cambricon-D仅0~4%精度下降；(b) Speedup：各种设计组合（DiffyDF/DiffyPE/AsyncPE/DiffyAll/Cambricon-D）vs systolic baseline/物理A100。

- 硬件平台是什么，配置是什么。
  自研Cambricon-D加速器（SystemVerilog RTL, Synopsys工具链, TSMC 45nm→DeepScaleTool缩至7nm），PE array 128×128, 1GHz，等效A100的3×10^14 FLOPS纯fp16吞吐，1.5TB/s HBM bandwidth。量化方案另在PyTorch实现用于快速精度评估。比较平台：NVIDIA A100 GPU（物理硬件），TPU-like systolic array cycle-accurate模拟器（128×128, 1GHz, 3×10^14 FLOPS, 1.5TB/s）。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Guided-Diffusion [5]（DDPM with gradient guidance），0.4B~0.5B参数，三种分辨率变体：GUID256（LSUN bedroom/cat, 256×256）、GUID128（ImageNet, 128×128）、GUID512（ImageNet, 512×512）；(2) Stable Diffusion v1.4 [18]（Latent Diffusion Model），扩散部分0.86B参数，STBL512（Conceptual Captions, 512×512）。精度度量：Inception V3网络提取特征→manifold estimation估算生成图像落入数据集分布的频率（precision指标，继承自Guided-Diffusion）。SSIM=0.9650（>0.95即人眼不可察觉）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  论文未明确说明开源。算法pipeline（temporal differential computing for diffusion models）执行流程如下：

  ```
  # 每个timestep t，对U-Net每层的计算
  # 输入：delta激活 ΔX_t（上层的输出, INT3）
  # 权重：W（预加载FP16）
  # 步骤1：差分卷积（PE Array执行）
  ΔY_t = Conv2D(W, ΔX_t)  # INT3×FP16乘法 → INT3输出

  # 步骤2：加载上步sign bits（DRAM→SFU, 1-bit/tensor element）
  Sgn_{t-1} = LoadSignBits(DRAM, layer_id)

  # 步骤3：差分ReLU（SFU执行，AND mask）
  ΔY'_t = ΔY_t ⊙ Sgn_{t-1}  # sign=1保留delta, sign=0置零

  # 步骤4：Group Normalization差分计算
  μ_G = (μ_G(t) + μ_G(t-1)) / 2  # 相邻timestep平均值
  σ²_G = (σ²_G(t) + σ²_G(t-1)) / 2
  ΔY''_t = (ΔY'_t - Δμ_G) / sqrt(Δσ²_G + ε)

  # 步骤5：delta输出压缩后写回DRAM更新raw值
  CompressAndWrite(DRAM, ΔY''_t)
  # NDP engine在DRAM侧: Y_t = Y_{t-1} + ΔY''_t
  # NDP engine更新sign bits: Sgn_t = sign(Y_t)

  # 步骤6（仅Attention层，占0.9%总时间）：回退到raw值计算
  Q,K,V = FetchRaw(DRAM)  # 直接加载FP16 raw值
  Attn(Q,K,V) = softmax(QK^T/√dk) V
  ```

  张量维度示例（GUID256某层）：输入ΔX_t ∈ R^{1×256×256×128} (NHWC, INT3)，权重W ∈ R^{3×3×128×128} (K_h×K_w×C_in×C_out, FP16)，输出ΔY_t ∈ R^{1×256×256×128} (INT3)，sign tensor Sgn_{t-1} ∈ {0,1}^{256×256×128} (1-bit)。Outlier处理：沿C_in维度每n=64元素一组，每组最多m=60个FP16 outlier（走fp×fp乘法器），其余INT3 inlier（走int×fp乘法器）。

## 44-MECLA_Memory-Compute-Efficient_LLM_Accelerator_with_Scaling_Sub-matrix_Partition.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Scaling Sub-matrix Partition (SSMP) —— 一种参数高效的矩阵分解方法，将大规模LLM线性层权重矩阵分解为小尺度的source sub-matrices (SS) 和 derived sub-matrices (DS)。每个DS由对应SS乘以一个scaling scalar得到。配置四元组 (x, y, nx, ny)：x/y为SS的纵向/横向维度，nx/ny为DS在纵向/横向的扩展数量。例如配置(8,8,4,4)时，一个权重矩阵被划分为n=DxDy/(x·nx·y·ny)个region，每region含1个SS和nx·ny-1个DS。存储需求：n个SS（尺寸[x,y]）+ n·(nx·ny-1)个scaling scalar，DS本身不存储——推理时从SS和scaling scalar实时生成。同时提出SSMP-oriented fine-tuning方法（Algorithm 2）：冻结预训练权重，创建可训练的WSS和S参数，引入forget factor σ（初始=1，正则化推向0），当σ<10^-4时移除预训练权重仅保留SSMP权重。该fine-tuning类似LoRA但目的不同——为推理效率而非训练效率。
  - 实验比较：(1) SSMP不同配置（standard/aggressive）的accuracy评估——20个model-task组合（RoBERTa base/large × 7 GLUE tasks × 2 settings + Bloom 1B7/7B × 3 GLUE tasks + LLaMA-2 7B × Dolly + Bloom 1B7/7B × WikiText2），比较MECLA standard（GLUE <2%/WikiText <5%精度损失的最小模型）和MECLA aggressive（GLUE <5%/WikiText <10%精度损失）；(2) SSMP不同模型规模下的压缩性能——RoBERTa base/large和Bloom 1B7/7B在wikitext-2 perplexity vs 压缩率曲线；(3) SSMP vs KD和MiniLLM的LLaMA压缩对比（Rouge-L score）；(4) V100 GPU上SSMP优化前后的推理加速（standard 2.32×/aggressive 2.88×）。

- 硬件平台是什么，配置是什么。
  - GPU baseline：NVIDIA V100 32GB (INT8 peak 125 TOPS)
  - MECLA硬件：28nm CMOS，22.02 mm²，1GHz，8 PE clusters，1.25MB on-chip SRAM，8GB external DDR4 per processor。32 MECLA processors集群（131.2 TOPS@INT8），900GB/s inter-processor bandwidth（模拟NVLink）
  - Fine-tuning：论文未明确说明GPU型号，使用PyTorch + HuggingFace library

- 模型是什么。数据集和bench分别是什么。
  - 模型：RoBERTa base, RoBERTa large, Bloom 1B7, Bloom 7B, LLaMA-2 7B（fine-tuning teacher: RoBERTa large, Bloom 7B, LLaMA 13B for knowledge distillation）
  - 数据集/benchmark：GLUE（8 tasks: CoLA/MCC, SST-2/Acc, MRPC/F1, QQP/F1, STS-B/Pearson, MNLI/Acc, QNLI/Acc, RTE/Acc），Databricks-dolly-15k（Rouge-L），WikiText-2（Perplexity）。共20个benchmark配置
  - Fine-tuning参数：learning rate {5e-4, 1e-4, 5e-5}，batch size {16, 32}，20 epochs；SSMP超参数搜索空间 {2,4,8,16}，grid search + successive halving

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明开源（ISCA 2024，Tsinghua University，未找到GitHub链接）
  - SSMP算法pipeline（张量计算流程，以Bloom-7B一个FFN linear层为例，配置(8,8,4,4)）：
    1. **离线SSMP Fine-tuning**（Algorithm 2）：
       ```
       # 输入: 预训练权重 W ∈ R^{4096×11008}（Bloom-7B FFN）
       # 初始化可训练参数
       WSS = randn(size=[Dx/(x*nx), Dy/(y*ny), x, y])  # [128, 344, 8, 8]
       S = randn(size=[Dx/(x*nx), Dy/(y*ny), nx, ny])  # [128, 344, 4, 4]
       σ = 1.0  # forget factor
       # 训练循环
       for batch in trainLoader:
           WDS = WSS * S  # broadcast: [128,344,8,8] × [128,344,4,4] → [128,344,8,8,4,4]
           Wnew = reshape(WDS) + concat(WSS, WDS)  # SSMP full weight reconstruction
           Wcombined = σ * W + (1-σ) * Wnew  # 混合预训练+SSMP weight
           output = Forward(batch, Wcombined)
           loss = LLM_loss(output) + λ * |σ|  # 正则化σ→0
           update(WSS, S, σ)
       # 当 σ < 1e-4: 删除W, 仅保留{WSS, S, σ≈0}
       ```
    2. **在线推理**（per token, per linear layer）：
       ```
       # 输入: activation x ∈ R^{1×4096}（单token embedding）
       # 存储的SSMP参数: WSS [128,344,8,8], S [128,344,4,4]
       # 恢复权重（on-the-fly on MECLA chip）:
       for r in 0..127:  # regions
           for i in 0..3:  # nx
               for j in 0..3:  # ny
                   DS_block[r,i,j] = WSS[r] * S[r,i,j]  # [8,8] × scalar → [8,8]
       # 等效完整权重: W_full ∈ R^{(128*4*8)×(344*4*8)} = R^{4096×11008}
       # 但实际不存储完整W，PE array直接使用SS+scalar计算
       # 输出: y = x @ W_full^T（通过PSum reuse完成，见硬件/kernel层）
       ```
    3. **存储节省**：原始W需4096×11008×2B = 90.2MB；SSMP后WSS需128×344×8×8×2B = 5.5MB，S需128×344×4×4×2B = 1.4MB，合计6.9MB（节省92.3%）。

## 46-Fast On-device LLM Inference with NPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  - **实现**：llm.npu 提出增强的 per-tensor W8A8 量化算法 + shadow outlier execution。将激活值中超出 INT8 量化范围（-127~128）的 outlier channels（约占 0.1%–0.3% 总 channels）提取为紧凑张量，在 CPU/GPU 上用 FP16 执行 MatMul；其余正常值在 NPU 上执行 per-tensor INT8 MatMul，最后将两部分结果相加。量化公式：`MatMul(x, w) = MatMul(clip(x/s, -127, 128), w) [on NPU] + MatMul(extract(x/s, ⌊x/s⌋×128), w) [on CPU]`。离线阶段通过 profiling 识别 outlier 重要性并剪枝不重要层（默认 85% 剪枝率），以减少 CPU-NPU 同步开销。基于观察“outliers 集中在少数 hot channels（<3% channels 贡献 >80% outliers）”，仅缓存 hot channels 的权重到 CPU 内存，cold channels 按需从磁盘加载。
  - **实验比较**：
    - 量化准确率：在 LAMBADA、HellaSwag、WinoGrande、OpenBookQA、MMLU 五个 benchmark 上对比 FP16、SmoothQuant（per-tensor）、K-Quant（per-group，用于 llama.cpp）、LLM.Int8()（float outlier handling）。llm.npu 平均准确率损失 <1%（vs FP16），优于 SmoothQuant 最高 32.9%，优于 K-Quant 最高 70.9%。
    - Prefill 速度/能耗：对比 llama.cpp-CPU、MNN-CPU、MLC-GPU、TFLite-GPU、PowerInfer-V2-NPU。1024-token prompt 下 prefill 加速 1.27–43.6×，能耗降低 1.85–59.52×。
    - Ablation：naive NPU → +chunk-sharing graph（1.46–5.09×） → +shadow outlier（3.91–8.68×） → +out-of-order execution（18–44% 延迟降低）。
  - 硬件平台是什么，配置是什么。
    - 两台 Android 手机：Redmi K70 Pro（Snapdragon 8gen3，Hexagon NPU 73 TOPS INT8，24GB 内存）、Redmi K60 Pro（Snapdragon 8gen2，16GB 内存）。NPU 时钟 500–750MHz。Qualcomm QNN SDK。内存共享架构（CPU/NPU 统一物理内存）。
  - 模型是什么。数据集和bench分别是什么。
    - 模型：Qwen1.5-1.8B、Gemma-2B、Phi2-2.7B、LLaMA2-Chat-7B、Mistral-7B。
    - 精度 benchmark：LAMBADA、HellaSwag、WinoGrande、OpenBookQA、MMLU。
    - 速度 benchmark：LongBench（2wikimqa、TriviaQA）、DroidTask（UI 自动化）、Persona-Chat（对话摘要）。
  - 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
    - 开源：https://github.com/UbiquitousLearning/mllm（MIT license），同时有 Zenodo artifact DOI: https://doi.org/10.5281/zenodo.14392760。
    - 算法 pipeline（量化 + shadow outlier 执行全过程）：
      ```
      # 离线准备阶段
      for each linear_layer in model:
          W_int8 = quantize_weights(W_fp16, method="max-min_symmetric")
          # 使用大规模语料库（wikitext）profiling 激活 outlier 重要性
          for each prompt_batch in calibration_corpus:
              activations = forward_pass(layer, prompt_batch)
              outliers = find_channels_where(abs(activations/s) > 128)
              importance[channel] = max(outlier_val / scale_s)
          # 按 importance 排序层内 channels，剪枝 85% 最不重要 layers 的 outliers
          prune_outliers_below_threshold(importance, prune_rate=0.85)
          # 识别 hot channels（<3% channels 贡献 >80% outliers）
          hot_channel_weights = W_fp16[hot_channel_indices, :]  # 仅缓存到 CPU 内存
          cold_channel_weights = W_fp16[cold_channel_indices, :]  # 保留在磁盘

      # 运行时推理（per-tensor W8A8 + shadow execution）
      def linear_layer_with_shadow_outlier(x_fp16, W_int8, s):
          # Part 1: NPU 上 per-tensor INT8 MatMul
          x_int8 = clip(round(x_fp16 / s), -127, 128)          # 量化激活
          y_npu = INT8_MatMul_NPU(x_int8, W_int8)               # NPU 执行 W8A8 MatMul

          # Part 2: CPU 上 shadow outlier 执行（处理超出量化范围的值）
          outlier_i8 = extract_outliers(x_fp16 / s, threshold=128)
          # outlier_i8 shape: (seq_len, num_outlier_channels), very sparse
          if len(outlier_channels) > 0:
              outlier_vals = (outlier_i8 - 128) * s              # 反量化到 FP16
              W_cpu = get_outlier_weights(outlier_channels)       # 从 hot 缓存或磁盘
              y_cpu = FP16_MatMul_CPU(outlier_vals, W_cpu)        # CPU FP MatMul
          else:
              y_cpu = 0

          # 合并结果
          y_final = y_npu * s + y_cpu                            # 反量化 + 合并
          return y_final
      ```
    - 关键设计点：(1) 利用加法分配律将 W8A8 量化 MatMul 分解为"NPU 整数 MatMul + CPU 小数补偿"两部分；(2) outlier 稀疏性（0.1–0.3%）使得 CPU 侧计算量极低，可被 NPU 执行完全隐藏；(3) hot/cold channel 机制减少 CPU 内存占用（降低 34.3%）；(4) 层级 outlier 剪枝消除 CPU-NPU 同步开销。

## 47-SpecEE- Accelerating Large Language Model Inference with Speculative Early Exiting

- 属于算法pipeline的实现是什么？实验比较什么？
  - SpecEE 提出 speculation-based lightweight predictor（基于推测的轻量预测器）来加速 LLM 推理中的 early exiting。核心算法：利用 speculative model（EAGLE DLM）生成的 speculative tokens 将预测器的在线搜索空间从完整词汇表（~3×10⁴ tokens）缩减到推测 token 范围（~3-4 tokens），实现 ~100× 搜索空间缩减。利用概率偏移（probability shift）现象，从 LLM 中间层 hidden states 中提取 3 类特征（speculative token logits、local probabilities、probability variation），输入到 2 层 MLP（hidden dimension 512，ReLU + Sigmoid 二分类）进行 exit/continue 决策。通过 verification algorithm（检查 global top-1 token 是否在 speculative set 中）保证准确率。实验比较 Dense baseline（HuggingFace）、AdaInfer（SVM based early exiting）、SpecEE+AWQ 量化，在 MMLU/CommonsenseQA/SST2/GSM8K 上评估准确率，在 MT-Bench/SUM/QA/Alpaca 等 8 个数据集上评估 speedup。
  - 特征维度：每层 4 个 speculative tokens × 3 类特征 = 12 维输入。MLP weights：12×512 + 512×1，~6.7K params/FLOPS（vs baseline early exiting predictor ~6.7M params/FLOPS，~100× 缩减）。

- 硬件平台是什么，配置是什么。
  - Cloud：NVIDIA Tesla A100-80GB GPU + Intel Xeon Platinum 8358 2.60GHz，NVIDIA RTX 4090 24GB + AMD EPYC 7542 2.90GHz。CUDA 12.1
  - PC：Lenovo Legion Y7000，NVIDIA RTX 4060 Laptop 8GB + Intel i7-13650HX 2.6GHz。CUDA 12.6

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama2-7B-chat（32 layers, h=4096）、Llama2-13B-chat（40 layers, h=5120）、Llama2-70B-chat（80 layers, h=8192）
  - Speedup/Throughput 数据集：MT-Bench、SUM、QA、Alpaca、GSM8K、HumanEval、MMLU、CommonsenseQA
  - Accuracy 数据集：MMLU (Acc)、CommonsenseQA (Acc)、SST2 (Acc)、GSM8K (Acc)
  - PPL 数据集：SUM、MT-Bench、Alpaca

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/infinigence/SpecEE，Zenodo DOI: https://doi.org/10.5281/zenodo.15102802，MIT License
  - 伪代码：
    ```python
    # speculative model 生成 4 个 speculative tokens，提取 speculative_lm_head ∈ R^(hidden_dim×4)
    spec_tokens = speculate(prompt)  # [T₁,T₂,T₃,T₄]
    spec_lm_head = lm_head[:, spec_tokens]  # hidden_dim × 4

    for layer_idx in range(num_layers):
        X = hidden_states[layer_idx]  # 1 × hidden_dim

        # 特征提取
        L = X @ spec_lm_head            # speculative token logits [l₁,l₂,l₃,l₄]
        P = softmax(L)                  # local probabilities [p₁,p₂,p₃,p₄]
        Δ = P - P_last                  # probability variation [Δ₁,Δ₂,Δ₃,Δ₄]
        feat = concat([L, P, Δ])        # 12 dim
        P_last = P

        # MLP 前向
        h = ReLU(W₁ @ feat + b₁)        # W₁∈R^(12×512)
        pred = Sigmoid(W₂ @ h + b₂)     # W₂∈R^(512×1)

        if pred > 0.5:
            # 验证
            global_logits = X @ lm_head  # 1 × vocab_size
            if argmax(global_logits) in spec_tokens:
                return argmax(global_logits)  # early exit
            # else: continue to next layer
    ```
  - MLP 训练：每 predictor 用 MT-Bench 生成 ~16K 训练数据（label=True 若该层 early exit token 匹配最终 token），仅需 ~2% 训练数据即可收敛，32 个 predictor 总训练约 5 分钟/A100。

## 4-BBS: Bi-Directional Bit-Level Sparsity for Deep Learning Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：BBS（Bi-directional Bit-level Sparsity）——一种新型post-training双向位级稀疏压缩算法。核心包含两种bit-level binary pruning策略无需retraining或calibration dataset：(1) **Rounded Averaging**：对weight group内所有weight的低significant bits计算rounded average常数，用一个常数替换所有weight的低位来生成bi-directional sparse columns；(2) **Zero-point Shifting**（Algo. 1）：遍历所有可能的BBS constant（p-bit，p=6），找到使MSE最小的optimal zero-point，将weight整体shift后生成zero sparse columns。在此基础上提出**Gloal Binary Pruning**（Algo. 2）：基于per-channel scaling factors识别top β% sensitive channels（保持8-bit），其余normal channels按conservative（rounded averaging, prune 2 bit columns）或moderate（zero-point shifting, prune 4 bit columns）压缩。sensitive channel数强制对齐到硬件的CH参数（e.g., 32）。BBS compression encoding：每个weight group存储8-bit metadata（2-bit #redundant columns + 6-bit BBS constant），BBS constant bit为0表示全零bit column（bit-serial dot product=0），为1表示全一bit column（bit-serial dot product=ΣA）。
  - 实验比较：(1) 准确率：BBS vs PTQ、BitWave bit-flip（相同sensitive ratio，conservative/moderate pruning）；BBS vs ANT 6-bit；BBS vs Microscaling/NoisyQuant（ViT）；BBS vs Olive（Llama-3-8B WikiText/C4 perplexity）。(2) 加速器性能：BitVert vs Stripes/Pragmatic/Bitlet/BitWave（bit-serial accelerators）和SparTen/ANT（value-based accelerators），包括speedup/energy/EDP。(3) Load imbalance分析：BitVert intra-PE/inter-PE stall vs 其他bit-serial加速器随PE column数增加的变化。(4) PE design space exploration：sub-group size (4/8/16) + area/power tradeoff。(5) Accuracy-efficiency Pareto前沿：EDP vs accuracy loss across pruning ratios。

- 硬件平台是什么，配置是什么。
  - 算法端（Binary Pruning执行）：单张NVIDIA RTX 3090 GPU，PyTorch实现，压缩整个ResNet-50约需15秒。
  - 硬件加速器评估（BitVert）：RTL（SystemVerilog），Synopsys Design Compiler综合 + TSMC 28nm工艺，目标频率800 MHz。Synopsys VCS生成data-driven activity factor用于功耗估计。CACTI建模on-chip SRAM buffer。DRAMSim3 DDR3模型估算DRAM功耗。自研cycle-accurate simulator建模端到端执行时间。
  - 公平对比：所有加速器含相同数量乘法器（1个8-bit=8个bit-serial乘法器）。SRAM: ANT及所有bit-serial加速器=256KB activation buffer + 256KB weight buffer；SparTen因PE内含local buffer相应减少。

- 模型是什么。数据集和bench分别是什么。
  - CNN: VGG-16、ResNet-34、ResNet-50 (ImageNet-1K)
  - Transformer: ViT-Small、ViT-Base (ImageNet-1K)、BERT (MRPC/SST2 from GLUE)
  - LLM: Llama-3-8B (WikiText/C4 perplexity)
  - 预训练模型来源：PyTorch Library and HuggingFace
  - Baseline 8-bit模型：post-training per-channel quantization（准确率损失可忽略）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/yc2367/BBS-MICRO.git
  - BBS算法pipeline（以8-bit weight, group size N=32为例）：
    1. **Per-channel 8-bit PTQ baseline**：对FP32 weight tensor做per-channel quantization→INT8 weight W + per-channel scaling factors S。
    2. **Global Channel Sorting**（Algo. 2）：对所有layer所有weight channel按scaling factor S排序→选top β%为sensitive channels（保持8-bit）→其余为normal channels。每层sensitive channel数强制对齐到CH（=32）倍数。
    3. **Group-level Binary Pruning**（对normal channels）：
       - Conservative (prune 2 columns): Step 1 识别并移除redundant columns（最高significant bit后重复的bit columns）。Step 2 对2个lower significant bit计算rounded average→用round(int)常数替换所有weight的低2位。输出：6 columns + 8-bit metadata。
       - Moderate (prune 4 columns): Zero-point shifting（Algo. 1）遍历constant ∈ [-32, 31]：
         ```
         for constant in range(-2^5, 2^5):
           W_tmp = Clip(W + constant, -128, 127)
           numRedunCol = GetNumRedunCol(W_tmp)
           W_tmp = RemoveRedunCol(W_tmp, numRedunCol)
           numSparseCol = N_target - numRedunCol
           W_tmp = GenSparseCol(W_tmp, numSparseCol)
           newMSE = MSE(W_tmp, W)
           if newMSE < bestMSE: keep (W_tmp, metadata={numRedunCol, constant})
         ```
         输出：4 columns + 8-bit metadata。
    4. **BBS Encoding**：压缩weight group = {2-bit #RedunCol, 6-bit BBS constant} + 剩余bit columns。BBS constant每一位指示对应bit column是all-zero（0）还是all-one（1）。
    5. **推理时BBS计算**（Eq. 1-3）：对各bit significance b，若weight bit-column中zero-bits > 50%→skip zero-bits（sum A where W_ib=1, Eq. 2）；若zero-bits < 50%→invert bit-vector skip one-bits（ΣA - sum A where W_ib=0, Eq. 3）。最大处理ceil(N/2)个effective bits per column。
    - 结果：average 1.29× (cons) / 1.66× (mod) 压缩，准确率损失 0.25% / 0.45%。BitVert up to 3.03× speedup, 2.44× energy saving vs prior accelerators。

## 54-HLX- A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models..pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：(1) **PipeFlash**: 对FA-2的fine-grained pipelined数据流算法。传统FA-2按block级别同步执行QKT→local softmax→PV→update O，softmax和update O的非MatMul延迟无法隐藏，compute utilization在A100饱和于~61%。PipeFlash将执行粒度细化为每次处理Q block中的2行，形成4-stage pipeline：DPE#0(QKT, 2 rows)→RVPE(softmax, 1 row)→DPE#1(PV, 2 rows)→UpE(update O, 1 row)。关键设计：通过行级流水打破op间dependency实现softmax/update O与QKT/PV重叠；K和V block被所有Q行复用；中间数据(score/prob矩阵)从128KB降至1KB (4.8× reduction)。(2) **PipeSSD**: 首个fused+pipelined SSD算法。传统SSD分5个独立kernel (chunk cumsum→chunk state→state passing→BMM chunk→chunk scan)，中间数据量大（642KB/block）且无复用，memory-bound（compute utilization ~27% on A100）。PipeSSD先做block-level fusion（类似FA-2 implicit attention map，single for-loop with O(N) complexity），再分解为3-stage fine-grained pipeline：(1st) dA预处理(RVPE: sdt=softplus(dt+dtbias), dACS=cumsum(dA), decay_states=exp, d2t)、(2nd) YDiag(DPE#0: CBT=C×B^T→RVPE: CBTLdt=CBT×L×sdt→DPE#1: YDiag=CBTLdt×x, 2 rows)、(3rd) YOff和statesN并行(RVPE→DPE#0: dCOff=exp(dACS)×C→YOff=dCOff×states(j-1), 8 rows || RVPE→DPE#1: dBdt=d2t×B→statesN=dBdt^T×x, 4 rows)→UpE: YFinal+update states。处理粒度由pipeline cycle平衡决定。中间数据642KB→58.5KB (11× reduction)，DRAM流量减少6.8×。Compute utilization达~78.4%。
  - 实验比较：(a) FA-2 compute utilization和speedup (1K-128K)对比A100/H100; (b) SSD compute utilization和speedup对比A100/H100/TPUv3; (c) FA-3 on H100 vs PipeFlash on HLX60 (varying seq len和batch size); (d) batch大小扫掠(1-128 @1K)的utilization和speedup; (e) end-to-end Hybrid-2.7B模型延迟对比; (f) 面积功耗对比GPU/TPU及SOTA加速器(VGA/MARCA/SOFA); (g) 统一架构的HW overhead分析。
  - 精度验证: FP16，PipeFlash和PipeSSD在8个benchmark (wikitext-2, Winogrande, ARC-challenge, ARC-easy, LAMBADA-openai, PIQA, OpenBookQA, HellaSwag)上与conventional FA-2/FA-3/SSD无精度损失。

- 硬件平台是什么，配置是什么。
  - HLX硬件: HLX60 (60 URSCs, 614.4 TFLOPS FP16, 2000 GB/s DRAM, 30.4MB SRAM, 14nm→7nm 169mm²/201.8W)对标H100; HLX30 (30 URSCs, 307.2 TFLOPS, 1935 GB/s, 15.2MB SRAM, 83.9mm²/108.47W)对标A100; HLX6 (6 URSCs, 61.44 TFLOPS, 450 GB/s, 3.04MB SRAM, 47.16mm²/35.06W)对标TPUv3 half-chip。
  - GPU baseline: A100 80GB (312 TFLOPS, 1935 GB/s HBM2E, 84.3MB on-chip, 826mm²/300W); H100 80GB (756 TFLOPS, 2000 GB/s HBM2E, 103.9MB on-chip, 814mm²/350W)。
  - TPUv3 half-chip: 61.5 TFLOPS, 450 GB/s HBM2, 16MB on-chip, 324mm²/225W, 2×128×128 MXU + vector unit。

- 模型是什么。数据集和bench分别是什么。
  - 模型: Hybrid-2.7B (Mamba2attn-2.7B) from [11]。架构: 64层 (6 attention + 58 Mamba-2 layers)。Attention: multi-head attention, 30 heads × dhead 128。SSD: 80 heads × dhead 64, dstate 128。Blocksize=256。
  - 准确率benchmarks: wikitext-2, Winogrande, ARC-challenge, ARC-easy, LAMBADA-openai, PIQA, OpenBookQA, HellaSwag (8个)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - HLX simulator/RTL未明确开源。GPU baseline开源: FA-2/FA-3 from https://github.com/Dao-AILab/flash-attention [47]; Mamba-2/SSD from https://github.com/state-spaces/mamba [11]。
  - **PipeFlash算法伪代码** (per attention head, per Q block of 2 rows, SeqLen dim parallel):
    ```
    # Q [Br, dhead], K/V [seqlen, dhead] partitioned into Tc blocks [Bc, dhead]
    # Pipeline: DPE#0→RVPE→DPE#1→UpE 各stage同时处理不同Q行
    for row in (0, 1):  # 2 Q rows per block
      for j in 0..Tc-1:
        # Stage1 DPE#0: QKT [1, Bc]
        S(row) = Q[row] @ K[j]^T
        # Stage2 RVPE: Local softmax (与下一次QKT重叠)
        m_new = max(m_old, rowmax(S(row)))
        P = exp(S(row) - m_new)         # [1, Bc]
        l_new = exp(m_old - m_new) * l_old + rowsum(P)
        # Stage3 DPE#1: PV (与softmax/下次QKT重叠)
        O_new = diag(exp(m_old - m_new))^{-1} * O_old + P @ V[j]
        # Stage4 UpE: Update m_old, l_old, O_old (与PV/softmax重叠)
    # Final: O = diag(l_Tc)^{-1} @ O_Tc, write to DRAM
    ```
  - **PipeSSD算法伪代码** (per Mamba-2 layer, per chunk block j, seqlen分解为c chunks):
    ```
    # Input: dt[j], dtbias, A, x[j], B[j], C[j]; states(j-1) from previous chunk
    # 1st Stage (RVPE): dA preprocessing
    sdt = softplus(dt[j] + dtbias)              # [cl]
    dA = sdt * A                                  # [cl, dstate]
    dACS = cumsum(dA)                             # [cl, dstate]
    decay_states = exp(dACS[-1:] - dACS)          # [cl, dstate]
    d2t = decay_states * sdt                      # [cl, dstate]
    # 2nd Stage: YDiag (DPE#0→RVPE→DPE#1)
    CBT = matmul(C[j], B[j]^T)                    # [cl, cl]
    CBTLdt = CBT * L * sdt                        # [cl, cl] element-wise
    YDiag = matmul(CBTLdt, x[j])                  # [cl, dhead]
    # 3rd Stage parallel:
    dCOff = exp(dACS) * C[j]                      # [cl, dstate]
    YOff = matmul(dCOff, states(j-1))             # [cl, dhead]
    dBdt = d2t * B[j]                             # [cl, dstate]
    statesN = matmul(dBdt^T, x[j])                # [dstate, dhead]
    # UpE:
    YFinal = YDiag + YOff                         # [cl, dhead]
    states(j) = exp(dACS[-1]) * states(j-1) + statesN  # [dstate, dhead]
    ```
  - PipeFlash与FA-2的关键差异：FA-2的step 7(QKT)→8(m/l update)→9(P计算)→10(O update) 顺序执行无法重叠；PipeFlash以2-row粒度使4个stage流水，non-MatMul与MatMul重叠。PipeSSD与SSD的关键差异：SSD 5 kernel有启动overhead且中间数据经DRAM；PipeSSD 3-stage融合流水将中间数据保留on-chip并在DPE/RVPE/UpE间直接forwarding。

## 56-AxCore- A Quantization-Aware Approximate GEMM Unit for LLM Inference.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：AxCore提出了一套完整的量化+近似计算算法pipeline用于LLM推理加速：
    **(1) Mixed-Precision FPMA (mpFPMA)**：将Mitchell对数近似 log2(1+M)≈M 推广到混合精度场景。对FP16 activation A和FP4量化权重Wq，先将Wq mantissa对齐至FP16域：`Align(Wq)=Wq << (NM_FP16 - NM_FP4)`，再执行 `R = A + Align(Wq) - B1 + C1`，其中B1=Ba+Bwq-Br为格式感知bias修正项（典型配置activation和result均为FP16时简化为B1=Bwq），C1为平均误差补偿常数。所有运算为整数加法，彻底消除浮点乘法器。
    **(2) Subnormal Number Conversion (SNC)**：低比特FP格式（FP4 E1M2/E2M1/E3M0）中subnormal值比例极高，FPMA因无hidden leading 1而对subnormal数学失效。SNC通过硬件查表将subnormal编码映射到数值最近邻的normalized编码（如E1M2 subnormal "011"=0.75 映射到 normal "010"=0.75）。无法精确映射的边界值采用随机rounding（up/down交替）消除系统偏置。运行时输出统一为S1E3M2内部格式。
    **(3) Mean-based Constant Error Compensation**：分析mpFPMA在mantissa空间(m_a, m_w)的误差分布，发现平均误差在各mantissa对上呈现可预测模式。定义per-format-pair补偿常数 C1 = (1/(2^NMa · 2^NMw)) · Σ_{ma,mw} ε(ma,mw)，即所有有效mantissa组合的平均error。该值在量化前precompute一次，对FP16×FP4、BF16×FP4、FP16×FP8等格式对均适用，运行时零开销。
    **(4) Adaptive Format-Aware Quantization**：支持block-wise自适应选择最优FP4子格式。对每组g×n weight block（g=group_size分组大小, n=output channels, 均需为array size倍数），在{E3M0, E2M1, E1M2}中通过min ||A·W^d - A·W||² 选择最优格式d。E3M0适合稀疏/power-of-two分布，E2M1均衡，E1M2适合均匀分布。量化/去量化重新定义为FPMA-friendly形式：wq = clamp(round(w - S + B - C)), wr = wq + S - B + C2，利用FPMA additive性质使量化噪声仅依赖加法/减法（无除法/乘法rounding偏置）。
    **(5) FPMA-based Dequantization**：后处理阶段O = Oq + S - B + C2，用两个整数加法替代缩放因子乘法。
    整体pipeline执行流程：离线阶段 — 权重分组→block-wise format selection→FPMA-aware quantization→wq存储。在线推理 — Activation A流入→PreAdd预计算 T=A-B1+C1→PE内 SNC(Wq)→Align→R=T+Align(Wq) →列累加→Norm→AxScale→Accumulator。
  - 实验比较：(1) PPL对比：FP16 baseline vs INT4 vs FP4 vs 基础FPMA vs mpFPMA vs mpFPMA+S(+SNC) vs mpFPMA+S+C(+Compensation) vs AxCore(full) vs FIGNA vs FIGLUT 在OPT(2.7B/6.7B/13B/30B)和LLaMA2(7B/70B)上的wikiText-2 PPL；(2) Zero-shot (ARC-e/HellaSwag/PiQA/Winogrande) average accuracy on OPT-30B/LLaMA2-70B；(3) KV cache quantization (AxCore-KV) accuracy loss；(4) SNR数值精度消融 (matrix size 128-32768)；(5) vs Tender W8A8KV4/W4A4KV4 compute density + accuracy；(6) 6种datatype配置下的PE面积和Compute Density对比。

- 硬件平台是什么，配置是什么。
  - GEMM加速器硬件：64×64 systolic array, 4× tiling, 1GHz, 28nm TSMC工艺。SpinalHDL RTL实现→Synopsys Design Compiler合成。
  - 准确性评估GPU：x86服务器 + 4× NVIDIA RTX 6000 Ada GPU (48GB), Ubuntu 22.04, CUDA 12.4, PyTorch 2.5.1。
  - 能耗模拟：DNNWeaver v2.0扩展 simulator + CACTI 7.0 SRAM功耗。

- 模型是什么。数据集和bench分别是什么。
  - 模型：OPT-2.7B, OPT-6.7B, OPT-13B, OPT-30B; LLaMA2-7B, LLaMA2-70B。
  - 数据集：WikiText-2（PPL评估, seq length=2048）；Pile（校准集=calibration防过拟合）；ARC-e, HellaSwag, PiQA, Winogrande（zero-shot评估, lm-eval-harness框架）。
  - 量化配置：weight-only quantization, group_size=128 (OPT) / 64 (LLaMA2)。block-wise adaptive format: block size=128×64 (OPT) / 64×64 (LLaMA2)。KV cache量化: 4bit, group_size=64。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/CLab-HKUST-GZ/micro58-axcore (GitHub) + https://doi.org/10.5281/zenodo.16895417 (Zenodo)
  - 算法伪代码（以W4A16 FP16 activation × FP4 weight为例）：
    ```
    # 离线阶段：Adaptive Format-Aware Quantization
    for weight_matrix W in model.layers:
        for block in partition(W, block_size=(g, n)):
            best_format = argmin_{d in {E3M0,E2M1,E1M2}} ||A_calib · W_d - A_calib · W||^2
            for w in block:
                wq = clamp(round(w - S + B_d - C_d))  # FPMA-aware quantize
            store(wq, best_format_flag, S)

    # 在线推理：mpFPMA-based GEMM
    # A: [M, K] FP16; Wq: [K, N] FP4 quantized; S: [N] FP16 scales
    for tile_a in A.split(tile_size=M_tile):     # weight-stationary: Wq 静止在PE列
        for row in tile_a:
            T = row - B1 + C1                     # PreAdd: 15-bit adder once per row
            for (wq, s) in zip(Wq_row, S_row):
                if is_subnormal(wq):
                    wq_norm = SNC_lookup(wq)      # subnormal→normal table+stochastic round
                else:
                    wq_norm = wq
                wq_aligned = wq_norm << (10 - NM_FP4)  # mantissa对齐至FP16域
                R = T + wq_aligned                # PE内: 7-bit integer adder (multiplier-free)
                Psum += R                         # 列向累加(partial FP add, un-normalized)
            Oq = Norm(Psum)                        # 共享归一化: LZD→shift→round
            O = Oq + s - B + C2                   # AxScale: FPMA-based dequantization (int add)
    return O  # FP16 result
    ```
  - SNC查表逻辑（E1M2为例）：subnormal(S-0-00)→返0、S-0-01→S-0-00(0.5↑)/return0(0↓)、S-0-10→S-00-0(0.5)、S-0-11→S-00-1(0.75)。随机bit取activation mantissa最高位。输出统一S1E3M2。

## 5-FuseMax_Leveraging_Extended_Einsums_to_Optimize_Attention_Accelerator_Design.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：FuseMax采用**1-pass attention cascade**（基于FlashAttention-2的算法形式化），通过Cascade of Einsums抽象对attention算法进行形式化、分类和优化。核心算法贡献包括：(1) **Cascade of Einsums形式化attention**：将attention的QK matmul、softmax、AV matmul表达为精确的Einsum cascade，显式定义iteration space、data dependencies和compute operations。例如QKm,p = (1/√E) × Qe,p × Ke,m（Einsum 22）、softmax分解为SNm,p = e^(QKm,p - GMp)、SDp = ΣSNm,p、Am,p = SNm,p/SDp（Einsums 29-30, 27-28）。(2) **Pass数分类学（Table I）**：3-pass cascades（PyTorch/TensorFlow/FLAT/E.T.）→需3次遍历M fiber（先compute global max+QK→再compute SN+SD→再compute A+AV）；2-pass cascades（TileFlow/Choi et al.）→先用local max partition→第二pass用global max correction；1-pass cascades（FlashAttention/FlashAttention-2/Rabe and Staats）→running max替代local max，迭代式adjust previous results。(3) **Division Reduction Optimization（Section IV-D）**：将Am,p = SNm,p/SDp（M×P次division）和AVf,p = Am,p × Vf,m 重构为：先SNVf,p = SNm,p × Vf,m（reduce across M），再AVf,p = SNVf,p/SDp（仅F×P次division）。由于M >> F（sequence length >> embedding dimension），减少division次数约M/F倍。论文指出此优化原用于FlashAttention-2，但可广泛应用于任意pass数的cascade。(4) **Numerically Stable Iterative Construction（Cascade 5）**：使用iterative ranks和running max/correction factors实现1-pass计算。Running max RM_{m1+1,p} = max(RM_{m1,p}, LM_{m1,p})；correction factor PRM_{m1,p} = e^{RM_{m1,p} - RM_{m1+1,p}}用于调整旧denominator到新max；running denominator RD_{m1+1,p} = SLD_{m1,p} + RD_{m1,p} × PRM_{m1,p}（先downscale旧值再add）。
  - 实验比较：(1) 1D PE array utilization（Unfused/FLAT/+Cascade/+Architecture/+Binding vs seq len）；(2) 2D PE array utilization；(3) Attention speedup vs unfused baseline and FLAT；(4) Attention energy vs baselines；(5) Full transformer inference speedup and energy；(6) 2D utilization breakdown by Einsum type（QK/AV compute vs softmax compute vs memory stall）。

- 硬件平台是什么，配置是什么。
  - 评估使用Timeloop + Accelergy在45nm技术节点建模TPUv2/TPUv3风格spatial array accelerator。940 MHz频率。2D PE array（dimension as FLAT cloud config），1D PE array (256 PEs)，global buffer与FLAT等面积。
  - Artifact运行于x86-64 machine + Docker。开源：https://github.com/FPSG-UIUC/micro24-fusemax-artifact

- 模型是什么。数据集和bench分别是什么。
  - 模型：BERT-Base [17]（E=64, 12 layers）、TrXL-wt103 [13]（E=64, 18 layers）、T5-small encoder [49]（E=64, 6 layers）、XLM [13]（E=128, 6 layers）。batch size B=64。所有评估为encoder-only inference（decoder因KV cache瓶颈受限于memory traffic，on-chip accelerator impact较小）。
  - Sequence lengths: 256、512、1K、2K、4K、8K、16K、32K、64K、128K、256K、512K、1M tokens。
  - 论文未使用传统accuracy benchmark（perplexity等）——算法保证numerically equivalent to standard attention（exact attention, no approximation），因此无accuracy degradation。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/FPSG-UIUC/micro24-fusemax-artifact
  - **1-Pass Attention Cascade算法pipeline（Cascade 5，即FlashAttention-2的Einsum形式化）**：

    **Initialization**（一次性）:
    ```
    BKe_{m1,m0} = Ke_{m1×M0+m0}       # Partition K along M into M1 chunks of M0
    BVf_{m1,m0} = Vf_{m1×M0+m0}       # Partition V similarly
    RM_{0,p} = -∞                      # Running max init
    RD_{0,p} = 0                        # Running denominator init
    RNV_{0,p} = 0                       # Running numerator-times-V init
    ```

    **Per iteration m1 (0 to M1-1), 对每个P tile并行**:
    ```
    # Step 1: Compute QK tile
    BQK_{m1,m0,p} = Σ_e Q_{e,p} × BK_{e,m1,m0}        # Einsum 44: Q × K^T

    # Step 2: Local max over M0 fiber
    LM_{m1,p} = max_{m0}(BQK_{m1,m0,p})                # Einsum 45

    # Step 3: Update running max
    RM_{m1+1,p} = max(RM_{m1,p}, LM_{m1,p})           # Einsum 46

    # Step 4: Local numerator (with running max for numerical stability)
    SLN_{m1,m0,p} = exp(BQK_{m1,m0,p} - RM_{m1+1,p}) # Einsum 47

    # Step 5: Local denominator
    SLD_{m1,p} = Σ_{m0} SLN_{m1,m0,p}                 # Einsum 48

    # Step 6: Local numerator-times-V (division deferred)
    SLNV_{f,m1,p} = Σ_{m0} SLN_{m1,m0,p} × BV_{f,m1,m0}  # Einsum 49

    # Step 7: Correction factor for old running denominator
    PRM_{m1,p} = exp(RM_{m1,p} - RM_{m1+1,p})          # Einsum 50

    # Step 8: Correct old denominator to new max, then add local
    SPD_{m1,p} = RD_{m1,p} × PRM_{m1,p}                # Einsum 51
    RD_{m1+1,p} = SLD_{m1,p} + SPD_{m1,p}              # Einsum 52

    # Step 9: Correct old numerator-times-V similarly
    SPNV_{f,m1,p} = RNV_{f,m1,p} × PRM_{m1,p}          # Einsum 53
    RNV_{f,m1+1,p} = SLNV_{f,m1,p} + SPNV_{f,m1,p}     # Einsum 54
    ```

    **Final output (after all M1 iterations)**:
    ```
    AV_{f,p} = RNV_{f,M1,p} / RD_{M1,p}                 # Einsum 55: Final division
    ```

    **关键算法特性**:
    - Running max RM替代global max：每步看到新M0 chunk后即刻更新，无需等全部M完成。
    - Correction factor PRM = e^{old_max - new_max}：用指数差值"rescale"旧accumulator到新max，保证numerical correctness。
    - Division reduction (Einsum 49 + 55)：在M维度上先做SN×V reduction再÷SD，division从M×P次降到F×P次。
    - 1-pass性质：只需要1次遍历M fiber（M1 iterations），每iteration同时计算BQK、local max、local numerator/denominator/numerator-times-V、running max/denominator/numerator-times-V update。无spill——所有intermediate live footprint为O(M0 × P0) = on-chip buffer size，independent of total sequence length M。
    - 对比3-pass：需pass1→QK+GM, pass2→SN+SD, pass3→A+AV，每pass间需完成整个M fiber → 必须buffer或spill整个M fiber（O(M) memory）。

## 68-SPARK_Scalable_and_Precision-Aware_Acceleration_of_Neural_Networks_via_Efficient_Encoding.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SPARK提出一种bit-level变长编码（variable-length encoding）方案来压缩已INT8量化的DNN参数。核心思想：INT8量化后的值在[0,7]区间占约80%（仅需4-bit表示），[8,255]区间占约20%（需8-bit表示）。SPARK利用最高位bit（c0）作为1-bit标识符区分高低精度：c0=0表示低精度4-bit编码（值域[0,7]），c0=1表示高精度8-bit编码（值域[8,255]）。编码规则（基于INT8 unsigned量化输入b0...b7）：Case 1 [0,7]：c0=0，输出{c4,c5,c6,c7}直接存低4-bit，无信息损失。Case 2 [8,127]：c0=1为标识符；若b0⊕b3=0则直接存{b1,b2,b3,b4,b5,b6,b7}到{c1...c7}；若b0⊕b3=1则向上舍入（设b3=0, b4..b7=1111），error≤16。Case 3 [128,255]：c0=1既是标识符也是有效位；若b0=1且b3=1则无损存储；若b0=1且b3=0则设b3=1、b4..b7=0000。**Accuracy Compensation Mechanism**：通过XOR check（b0⊕b3）决定舍入方向，最小化编码误差，无需finetuning即可保持准确率。实际lossless占比>95%。
  - 实验比较：(1) 准确率对比：SPARK（4-bit effective）vs FP32 baseline在VGG16/ResNet18/ResNet50（ImageNet）和ViT/BERT（SST-2/GLUE）上的accuracy，SPARK平均损失<0.1%，BERT反而提升0.6%；(2) 无finetuning准确率对比：SPARK（~5.33 bit avg）vs ANT（6-bit）vs BiScaled（6-bit）在VGG16/ResNet50/ResNet152上；(3) Attention模型量化准确率：SPARK（4.31 avg bit）vs Q8BERT/OS/Olive/ANT在BERT SST-2上；(4) 优化消融：Compensation Mechanism + w/-FT + w/o-FT对准确率的影响；(5) 性能对比：SPARK vs Eyeriss/BitFusion/OLAccel/BiScaled/AdaFloat/ANT/Olive在6个网络上的normalized latency（speedup）和normalized energy；(6) 面积对比：各架构PE/decoder/encoder的28nm面积breakdown；(7) Scalability：不同model size下的energy efficiency变化趋势；(8) Joint optimization：SPARK + DBB pruning (50% sparsity) 的性能提升。

- 硬件平台是什么，配置是什么。
  - 模拟器：自研cycle-accurate simulator模拟SPARK PE array（含decoder+encoder+accumulation）。RTL实现：Verilog，28nm TSMC工艺库，Synopsys Design Compiler综合。频率200MHz。Global buffer 5MB（CACTI估算）。Encoder/decoder带宽约50 GB/s（>PE pages峰值需求~25 GB/s，非阻塞）。CACTI [1] 评估on-chip memory。DeepScaleTool将所有设计统一缩放到28nm进行iso-area比较。

- 模型是什么。数据集和bench分别是什么。
  - 模型：CNN-based：VGG16、ResNet18、ResNet50（torchvision pretrained）；Attention-based：ViT（vision transformer）、BERT-Base（BERT）。
  - 数据集：ImageNet（CNN视觉任务），SST-2/GLUE（BERT NLP任务）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确说明代码开源。SPARK encoding框架在PyTorch中实现。
  - 算法pipeline例子（以8-bit unsigned INT8 → SPARK encoding为例）：
    ```
    # 输入：layer-wise INT8量化后的weight/activation张量 W ∈ [0,255]^{M×N}
    # Step 1: 逐元素SPARK编码
    for each element x in W (unsigned 8-bit):
        b = bin(x, 8)   # b0b1b2b3b4b5b6b7, MSB first
        if x in [0, 7]:                            # Case 1: short code
            identifier c0 = 0
            code = {c0, b5, b6, b7}               # 4-bit output
        elif x in [8, 127]:                        # Case 2: long code (7 valid bits)
            if b0 XOR b3 == 0:                     # no rounding needed
                code = {1, b1, b2, b3, b4, b5, b6, b7}  # 8-bit, lossless
            else:                                   # round up
                code = {1, b1, b2, 0, 1, 1, 1, 1}  # 8-bit, lossy (error ≤16)
        else:  # x in [128, 255]                   # Case 3: long code (8 valid bits)
            if b3 == 1:                             # lossless
                code = {b0, b1, b2, b3, b4, b5, b6, b7}
            else:                                   # round
                code = {b0, b1, b2, 1, 0, 0, 0, 0}  # 8-bit, lossy
    # Step 2: 存储——所有code以4-bit基本长度对齐存储
    # low-precision codes占4-bit slot，high-precision codes占连续2个4-bit slots
    # Step 3: Decoding (硬件端)
    # 读入4-bit + enable信号
    # 若c0=0: 直接输出低精度值
    # 若c0=1且EN=0: 根据c3决定输出3-bit(c1c2c3)或4-bit(c0c1c2c3)
    # 若c0=1且EN=1: 直接输出4-bit作为高精度后段
    ```

    SPARK编码与现有量化/压缩方法正交，可叠加pruning（如DBB sparsity 50%）实现联合优化。

## 6-MCBP- A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MCBP提出三种bit-grained算法优化——(1) **BRCR (BS-Repetitiveness-enabled Computation Reduction)**：将k-bit权重矩阵分解为k个bit-slice (BS)矩阵，再按group size m（本文选定m=4）分组为Group Matrix。对每个Group Matrix，①识别重复列向量（至多2^m种类型，因为有m行每行1bit），②将对应激活累加到Merged Activation Vector (MAV，长度2^m)，③用Enumeration Matrix（记录哪种列向量出现在每行的哪一列）乘以MAV重建GEMV结果。对于H维大矩阵(m=4)，BRCR总加法数为 k(H×(1-bs̃) + m×2^{m-1})，相比value sparsity (kHm×vs̃) 减少12.1×，相比naive bit-serial computing减少3.8×。(2) **BSTC (BS-Sparsity-enabled Two-State Coding)**：采用sign-magnitude格式，对高bit位(3rd-7th BS矩阵，BS sparsity>65%)的列向量进行two-state编码——全零列编码为1'b0，非零列编码为{1'b1, m-bit data}。group size m=4时对BS矩阵独立编码，与BRCR共享group粒度避免数据转换开销。平均CR>1（正收益），因为bit sparsity是value sparsity的10.1×。(3) **BGPP (Bit-Grained Progressive Prediction)**：对attention的top-k预测采用多轮bit-level渐进过滤——第r轮只加载Keys的当前bit（从MSB到LSB），计算部分QK估计后，用阈值θ_i^r = max(Â_i^r) - α_r×radius过滤出vital Keys的indices，仅这些Keys参与下一轮bit加载和计算。通过early termination避免低bit位的冗余计算和KV cache访问。α_r∈[0.5,0.6]平衡准确率和sparsity。
  - 实验比较：(1) INT8准确率：FP16 vs INT8 vs MCBP Standard (0% loss) vs MCBP Aggressive (1% loss) 在Llama7B/13B, OPT1B3, Bloom1B7, Qwen7B上，涵盖MMLU/Wikilingua/MBPP/WikiText-2/Winogrande/Cola/MNLI/SST2 9个task；(2) Computation Reduction对比：MCBP vs SOFA/Spatten/FACT/Bitwave/FuseKNA在prefill阶段的计算量归一化；(3) Memory Access Reduction对比：MCBP vs 上述baseline在decoding阶段的访存量归一化；(4) Ablation：BRCR单独→+BSTC→+BGPP的latency reduction，在不同prompt/decoding长度(Dolly & MBPP)下分解；(5) 算法在GPU上的增益分解：BRCR(1.2×)→+BSTC(1.44×)→+BGPP(1.23×)；(6) α_r sensitivity：α_r从0.3到0.8下MMLU和MBPP的准确率与attention sparsity变化；(7) QAT INT8 / PTQ INT8 / PTQ INT4下bit sparsity vs value sparsity对比；(8) BRCR+BSTC在PTQ INT8/QAT INT8/PTQ INT4下的computation/memory reduction。

- 硬件平台是什么，配置是什么。
  - ASIC评估：RTL设计，Synopsys DC综合，TSMC 28nm CMOS工艺，1GHz频率。MCBP accelerator含20 PE Clusters (160 PEs)，on-chip SRAM 1248KB (768KB Weight + 384KB Token + 96KB Temp)，HBM2 8×128-bit @2GHz 8GB。面积9.52mm²，功耗2.395W。
  - GPU对比：NVIDIA A100 80GB，TensorRT-LLM (INT8)，batch size 8/128。148个MCBP processors (总622TOPS@INT8)与A100 (624TOPS) peak算力对齐对比。
  - 模拟：Verilator RTL仿真提取cycle count，custom cycle-level simulator评估end-to-end性能，CACTI估计SRAM面积/功耗，Ramulator模拟HBM延迟，Cadence Virtuoso设计CAM cell。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama7B, Llama13B, Qwen7B, Bloom1B7, OPT1B3 (HuggingFace PyTorch预训练模型)。
  - 数据集/Benchmark：(1) MMLU (S=0.5k, reasoning); (2) Wikilingua (S=2k, summarization, ROUGE-1); (3) MBPP (S=1k, code generation); (4) WikiText-2 (S=2k, language modeling, perplexity); (5) Winogrande (S=0.25k); (6) Cola (S=0.25k); (7) MNLI (S=0.5k); (8) SST2 (S=0.25k); (9) Dolly (S=8k, long-context summarization)。均来自GLUE/HuggingFace/PyTorch生态。INT8 baseline使用post-training quantization (per-channel symmetric W, per-tensor asymmetric X)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明开源。算法pipeline伪代码（以Llama7B, GEMV, m=4, k=8为例）：
    ```
    # 输入: INT8 weight matrix W∈R^{H×H}, INT8 activation X∈R^{H}
    # 输出: INT8 result Y∈R^{H}
    
    # === Offline: BSTC Encode ===
    # Step 1: Decompose W into k=8 bit-slice matrices W_bs[8]∈R^{H×H} (each element is 0/1)
    #   W_bs[0] = W & 1          # LSB (1st bit)
    #   W_bs[1] = (W >> 1) & 1   # 2nd bit
    #   ...
    #   W_bs[7] = (W >> 7) & 1   # MSB (sign bit, SM format)
    
    # Step 2: For each BS matrix (3rd-7th), group rows by m=4 into column vectors
    #   For each 4-bit column vector v in group matrix G∈R^{4×H}:
    #     if v == 4'b0000: encode as 1'b0
    #     else: encode as {1'b1, v}
    #   BS matrices 1st, 2nd, 8th: no compression (sparsity too low)
    
    # === Online: BRCR GEMM (per BS matrix, per group of m=4 rows) ===
    # Step 3: Decompress BS matrix (BSTC decode)
    #   For each 1'b0 → expand to 4'b0000
    #   For each {1'b1, data} → extract 4-bit data
    
    # Step 4: Merge repetitive operations
    #   MAV = zeros(2^m)  # length 16
    #   For j in 0..H-1:
    #     col_idx = G[:,j] as integer  # 0..15
    #     if col_idx != 0:
    #       MAV[col_idx] += X[j]
    
    # Step 5: Computation reconstruction
    #   For i in 0..m-1:  # for each of the 4 rows
    #     For each col_idx in 0..15 where Enumeration[i][col_idx] == 1:
    #       Y[row_offset+i] += MAV[col_idx]
    #   # The Enumeration matrix has at most 2^{m-1}=8 ones per row
    
    # Step 6: Bit-level accumulation with shift
    #   Y = Σ_{b=0}^{7} (BRCR_result[b] << b)
    #   # MSBs weighted by 2^b per standard integer arithmetic
    
    # Step 7: Apply quantization scale and bias
    #   Y_q = Scale ⊙ (W_q X_q) + Bias
    #   # Scale = ΔWΔX/ΔY, Bias = ZY - ΔWΔX·W_q·Z_X/ΔY
    
    # === BGPP (for attention): ===
    # For r in 1..num_rounds:
    #   Load r-th bit of selected Keys from HBM (1 bit only)
    #   Compute partial QK: Â_r = Q(4bit) × K_r(1bit) << (r-1)
    #   threshold = max(Â_r) - α_r × radius
    #   selected_indices = where(Â_r > threshold)
    #   if selected_indices unchanged or early condition met: break
    # Final: compute full-precision attention only on selected KVs
    ```

## 71-Pre-gated_MoE_An_Algorithm-System_Co-Design_for_Fast_and_Scalable_Mixture-of-Expert_Inference.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Pre-gated MoE 引入 pre-gate function 替代传统 MoE 的 gate function——第 N 个 MoE block 的 pre-gate function 被训练为预先选择第 (N+1) 个 MoE block 需激活的 experts，而非选择当前 block 的 experts。这解耦了 expert selection 与 expert execution 之间的串行依赖。第一个 MoE block 使用两个 gate（第一个选当前 block 的 experts，第二个 pre-gate 选下一个 block 的 experts）；最后一个 MoE block 不使用 pre-gate。Pre-gate function 在 fine-tuning 阶段增量训练，利用已有的 pretrained MoE 权重，只修改 MoE block 的 gate 功能。每个 pre-gate 是一个紧凑的 MLP 层，计算量极低。
  - 实验比较：(1) MoE block 平均延迟（normalized to GPU-only），对比 Pre-gated MoE vs GPU-only / MoE-OnDemand / MoE-Prefetch；(2) 端到端推理吞吐（tokens/sec）；(3) 模型准确率对比（conventional MoE vs Pre-gated MoE），涵盖 Xsum（Rouge-1/Rouge-2）、CB Web QA（ExactMatch/F1）、SQuAD（ExactMatch/F1）；(4) pre-gate 激活层级消融（N=1/2/3 block ahead 对准确率影响）；(5) 激活 expert 数量 sensitivity。

- 硬件平台是什么，配置是什么。
  - GPU：单 NVIDIA A100 80GB HBM。
  - CPU：AMD EPYC 7V12 64-Core + 1.8TB DDR4 memory。
  - 互联：PCIe Gen4，32 GB/sec 单向带宽。
  - GPU-only baseline 使用单 GPU 存全部参数（oracular 性能上界）。CPU-GPU 系统将 MoE 参数全部 offload 到 CPU DDR4，非 MoE 参数常驻 GPU HBM。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Google SwitchTransformer，四个配置——Switch-Base (8/64/128 experts, 0.7–7.5B params)，Switch-Large (128 experts, 26.4B params, 105.6GB)。
  - 数据集/Bench：Xsum（摘要任务，Rouge-1/Rouge-2）；CB Web QA + SQuAD（闭卷问答任务，ExactMatch/F1）。
  - 训练：HuggingFace 预训练权重 → fine-tune 2,048 steps，batch 256 seq × 256 tokens，LR 0.0001 常数。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/ranggihwang/Pregated_MoE；Zenodo https://doi.org/10.5281/zenodo.10976343。
  - 算法 pipeline 例子（Switch-Base 128 experts, top-1 sparse activation, 单 batch inference）：
    ```
    # 传统 MoE (conventional): 第N个block的gate选第N个block的experts → serial依赖
    # for each MoE block N:
    #   gate_out = gate_N(hidden_N)           # (batch, seq, num_experts) softmax probs
    #   expert_indices = topk(gate_out, k=1)  # 选中expert #3
    #   output = expert_3(hidden_N)           # 执行选中expert的FFN，gate和execute必须串行

    # Pre-gated MoE: 第N个block的pre-gate选第(N+1)个block的experts → 解耦依赖
    # Block 0 (first MoE block): 两个gate函数
    gate0_out = first_gate(hidden_0)        # 选block 0的experts（等同传统gate）
    expert_idx_0 = topk(gate0_out, k=1)     # e.g., expert #2
    pregate0_out = pre_gate_0(hidden_0)     # pre-gate选block 1的experts
    next_expert_idx_1 = topk(pregate0_out, k=1)  # e.g., expert #5
    output_0 = expert_2(hidden_0)           # 执行block 0的expert #2
    # 同时，系统可异步prefetch expert #5 从CPU到GPU

    # Block 1:
    # 此时expert #5 已在GPU ready（prefetch完成）
    output_1 = expert_5(hidden_1)           # 直接执行，无CPU→GPU等待
    pregate1_out = pre_gate_1(hidden_1)     # 选block 2的experts
    next_expert_idx_2 = topk(pregate1_out, k=1)
    # ...继续重叠：block 1执行 + block 2的expert prefetch

    # Last MoE block: 无pre-gate，直接执行
    output_last = expert_k(hidden_last)
    ```
    数学上：pre-gate function 为 G_N^pre: h_N → p_{N+1} ∈ R^E，其中 h_N 为第 N 个 block 的输入 hidden states，p_{N+1} 为第 (N+1) 个 block 各 expert 的激活概率。训练目标不变，仅 gate 的输入-输出时序移位。Pre-gate 本身计算量 <1% of MoE block FLOPs（小型 MLP: h_dim → E logits → softmax）。

## 74-SOFA_A_Compute-Memory_Optimized_Sparsity_Accelerator_via_Cross-Stage_Coordinated_Tiling.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SOFA的算法优化包含三个关键设计：(1) **DLZS (Differential Leading Zero Summation)**：基于log域的无乘法计算范式预测attention稀疏性。将乘法转换为：`x·y ≈ XOR(Sx, Sy) × Mx · 2^(W−LOy)`，仅需shift和add操作。差分特性：仅将乘法的一个operand通过LZE (Leading Zero Encoder) 转换到对数域，另一个operand做移位。分两阶段——Key prediction phase（预将Wk转为LZ格式存储，8-bit token × 4-bit LZ weight → shift-sum得K̂）；Attention prediction phase（16-bit Q的LZ + K̂ → shift-sum得Â）。(2) **SADS (Sphere-Search-Aided Distributed Sorting)**：利用"Distributed Cluster Effect (DCE)"——长序列可划分为若干sub-segments，每段内的较大值能代表全局较大值。将一行Atten矩阵分为n=4个sub-segments，每段取top-(k/n)值；在每次迭代中以previous max为benchmark，引入search radius r生成feasible range (FR)，仅对FR内元素排序以减少比较器开销。Type-I（dominated by few tokens）和Type-II（均匀分布的多dominant tokens）共占>95%分布，SADS能有效捕获。(3) **SU-FA (Sorted-Updating FlashAttention)**：利用top-k阶段提供的排序信息消除FA-2的重复MAX比较开销。采用descending order更新——从MAX index开始descend更新到k-th value，使li更新仅需1 Exp + 1 Add（vs ascend的1 Exp + 1 Mul + 1 Add），平均降低25%计算复杂度。(4) **Bayesian Optimization DSE**：搜索每层的tiling size Bc（2-32, step=2）和top-k（5%-50%, step=5%），优化目标minimize L(R)=L_en + α×L_cmp + β×L_exp，其中L_cmp=∑(Bc_i·k)/(S·k)，L_exp=∑(S/Bc_i)。
  - 实验比较：(1) 复杂度消融：4bit+vanilla sorting+FA2 vs +DLZS vs +SADS vs +SU-FA，归一化计算复杂度对比，DLZS降低18%，SADS+SU-FA额外降低10%，总降低28%；(2) Computation reduction：SOFA的LP在0%/1%/2%准确率损失下降低Attention+QKV计算56.8%/62.6%/67.4%，单独Attention降低81.3%/87.7%/92.6%；(3) GPU软件加速：LP on A100 GPU实现1.08-1.78× speedup，LP+FA1实现~1.5× gain (总2.7×)，LP+FA2额外1.19×；(4) 端到端throughput：SOFA vs A100 GPU，在0%/1%/2% loss下分别为6.1×/7.2×/9.5× speedup；(5) 与8个SOTA accelerator对比：energy efficiency、area efficiency、latency。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 (software evaluation)，PyTorch + Huggingface Transformers，torch.cuda.synchronize计时，nvidia-smi测功耗。
  - TPU：Google Cloud TPU (对比evaluation)。
  - Accelerator：SOFA专用ASIC，RTL设计，Synopsys DC综合，TSMC 28nm CMOS @ 1GHz，5.69 mm²面积，core power 0.95W，总device power (core+I/O) 2.45W。HBM2 16 channels @ 2GHz作为off-chip DRAM。
  - 软件：Verilator RTL仿真提取cycle count，构建cycle-level simulator用于端到端性能评估。

- 模型是什么。数据集和bench分别是什么。
  - 模型：BERT-base、BERT-large（NLP Encoder）；GPT-2、Bloom-1.7B、Llama-7B/13B（LLM Decoder）；PVT（Vision Transformer）。
  - 数据集/benchmark：GLUE benchmark（MRPC/RTE/SST2/STSB/QNLI/CoLA）、SQuAD v1.1、WikiText-2、WikiLingua、Wiki-raw、WinoGrande、ImageNet-1k（fine-tune from ImageNet-21k checkpoint）。共20个benchmark。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确说明是否开源。RTL设计通过Verilator仿真，未提供源码链接。
  - 算法pipeline（以128 token parallel处理为例，int8 Q/K, int16 formal computing）：
    ```
    # === Phase 0: Pre-deployment Preparation (offline) ===
    # Convert Wk weights to LZ format and store
    # DSE: Bayesian optimization search for Bc_i and top-k per layer
    
    # === Phase 1: User Inference (online) ===
    # Step 1: DLZS Key Prediction
    # Input: x ∈ Z^{128×H} (tokens), Wk ∈ Z^{H×d} (pre-converted to LZ format: {sign, M, LZ})
    # For each pair (x_i, w_j):
    #   K̂_ij = XOR(sign(x_i), sign(w_j)) × M(w_j) × (x_i << LZ(w_j))
    # Output: K̂ ∈ Z^{128×d} (8-bit, cached in SRAM)
    
    # Step 2: DLZS Attention Prediction
    # Q 16-bit, LZE(Q) → 5-bit LZ; K̂ from Step 1
    # For each (Q_i, K̂_j):
    #   Â_ij = XOR(sign) × M(Q_i) × (K̂_j << LZ(Q_i))
    # Output: Â ∈ Z^{128×S} (Pre-Atten matrix in SRAM, no DRAM store)
    
    # Step 3: SADS Sorting (per row, n=4 sub-segments)
    # For each row r of Â (size S):
    #   Divide into 4 sub-segments of size S/4
    #   For each sub-segment s:
    #     Apply clipping with threshold = max(Max_prev - r, current_Min)
    #     Bitonic sort (16-to-4) on feasible range entries
    #     Select top-(k/4) from s → FC_s
    #   FC = ∪ FC_s (final vital KV indices)
    #   Also extract top-1 and top-2 for SU-FA
    
    # Step 4: On-demand QKV Generation + RASS scheduling
    # RASS: greedy search for shared K/V across queries
    # Only generate sparse K_s = {x_i · Wk for i in selected indices}
    # Only generate sparse V_s = {x_i · Wv for i in selected indices}
    
    # Step 5: SU-FA (Descending order, per query q_i)
    # For each tile j of K_s/V_s (Bc rows):
    #   For q = k down to 1 (descending by predicted value):
    #     s_{iq} = Q_i · K_{idx[q]}^T
    #     if first iteration in this tile:
    #       m_i = s_{iq}  (Max value, mode 1 in AP module)
    #     else:
    #       P_i = exp(s_{iq} - m_i)  (mode 0)
    #       l_i += rowsum(P_i)
    #       O_i += P_i × V_{idx[q]}
    #   # Max ensure: between tiles, compare s with cached Max, update if needed
    # O_i = O_i / l_i  (final normalization)
    # Output: O ∈ Z^{128×d}
    ```


## 77-Make_LLM_Inference_Affordable_to_Everyone_Augmenting_GPU_Memory_with_NDP-DIMM.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：基于LLM中激活稀疏性（activation sparsity）的hot/cold neuron分区策略。观察到ReLU等激活函数使约20%神经元（hot neurons）承担80%计算量，余下80%（cold neurons）仅承担20%计算量（hot neuron计算强度为cold的16×）。据此将hot neurons映射到consumer-grade GPU执行，cold neurons offload到NDP-DIMM执行。具体实现：（1）在LLM的self-attention block前插入ReLU函数（对原生非ReLU激活的LLaMA（SiLU）和Falcon（GELU）模型，使用SparseLLM社区修改版替换为ReLU），实现QKV generation的激活稀疏性；（2）设计lightweight online predictor——4-bit neuron state table（每个neuron一个4-bit state，范围0-15），采用token-wise prediction（借鉴分支预测的两级自适应策略：激活则state+=s（s=4），未激活则state-=1）和layer-wise prediction（offline采样前一层的top-2高相关neuron，建立correlation table），最终通过 s1 + λ·s2 > T 判定激活（λ=6, T=15, Th=10），预测准确率98%，内存占用<1MB（LLaMA-7B仅232KB）；（3）online hot/cold adjustment：当neuron state超过阈值Th=10时识别为hot neuron，在projection计算期间通过neuron mapper将hot neuron从DIMM拷贝到GPU memory，同时将GPU memory中state最低的cold neuron换出（仅覆盖地址，无需额外数据传输）。
  - 实验比较：（1）Hermes vs Hermes-base（无激活稀疏性的纯NDP-DIMM扩展系统）：在LLaMA2-13B/70B、Falcon-40B上平均5.17× speedup；（2）Ablation：Hermes-random（随机neuron放置）vs Hermes-partition（仅offline ILP优化）vs Hermes-adjustment（offline+online adjustment）vs Hermes（全优化含window-based remapping），验证每阶段贡献；（3）Predictor对比：token-wise only vs layer-wise only vs combined，证明98%准确率。

- 硬件平台是什么，配置是什么。
  - GPU: 单张NVIDIA RTX 4090 24GB GDDR6, 82.6 TFLOPS, 1321 Tensor TOPS (FP16), 936 GB/s带宽。敏感度分析也使用RTX 3090 (24GB GDDR6, 142 Tensor TOPS) 和Tesla T4 (16GB GDDR6, 65 Tensor TOPS)。
  - NDP-DIMM: 8× DDR4-3200 DIMM（32GB/DIMM, 4 rank/DIMM, 2 bank groups/rank, 4 bank/BG），每DIMM含1个NDP core（GEMV unit: 256 multipliers @1GHz, reduction tree accumulator, 256KB buffer; Activation unit: 256 FP16 exp+256 add+256 mul units + comparator tree + adder tree + divider; DIMM-link: 25Gb/s/Lane, 8 lanes, 25GB/s per link, 1.17 pJ/b）
  - 互联: PCIe 4.0 64GB/s (GPU↔Host/NDP-DIMM)
  - Host CPU: Intel i9-13900K (用于Hermes-host baseline和scheduler运行)

- 模型是什么。数据集和bench分别是什么。
  - 模型: OPT-13B, OPT-30B, OPT-66B (原生ReLU激活), LLaMA2-13B, LLaMA2-70B (SiLU→ReLU替换), Falcon-40B (GELU→ReLU替换)。修改版LLaMA2和Falcon来自 https://huggingface.co/SparseLLM
  - 数据集: ChatGPT prompts [39], Alpaca [47] (端到端评估), C4 [44], Pile [15] (offline profiling/ILP参数采集，128 samples), COPA [46], Wikitext2 [37], PIQA [7] (token-wise similarity和layer-wise correlation分析)
  - Metric: tokens/s (端到端生成速度), batch sizes 1-16, input/output序列长度固定128

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - Hermes系统未开源（论文未提供整体系统代码链接）。修改版LLaMA2/Falcon模型开源：https://huggingface.co/SparseLLM。ILP求解器使用开源PulP [55]（https://coin-or.github.io/pulp/）。
  - 算法pipeline核心流程（以LLaMA2-70B token generation阶段单层MLP为例）：
    1. **Prompting阶段**（全部在GPU执行）：记录每个neuron的激活频率，初始化4-bit neuron state table（16个bin：激活频率>90%→state=15，<2%→state=0）。
    2. **Token generation - QKV generation**: GPU和NDP-DIMM协同执行。GPU从GDDR6读取hot neurons的FC权重（Q、K、V各层），并行计算GEMM，输出部分Q/K/V。NDP-DIMM的GEMV unit从DRAM读取cold neurons权重，执行GEMV（256 multipliers bit-serial, FP16 ×8 per multiplier），结果通过merge kernel在NDP-DIMM侧汇总。
    3. **Attention**: 完全在NDP-DIMM上执行（利用DIMM高内部带宽）。Q·K^T→softmax→×V。KV cache存储在DIMM内存中（节省GPU显存）。
    4. **Projection**: 仅GPU执行，DIMM idle期间执行online adjustment：predictor用token-wise similarity（更新neuron state: 激活→state+=4, 未激活→state-=1）+ layer-wise correlation（查前层correlation table），对neuron state > Th(10) 的新hot neuron从DIMM拷贝到GPU memory，覆盖state最低的GPU旧cold neuron。
    5. **MLP (FC1→ReLU→FC2)**: 同QKV generation，GPU执行hot neuron部分（GEMM），NDP-DIMM执行cold neuron部分（GEMV），merge在DIMM侧。
    6. **Window-based remapping**: 每5个token一个window，统计各DIMM上激活neuron数量，最大激活数DIMM与最少激活数DIMM配对，将pair中最活跃neuron从高负载DIMM remap到低负载DIMM（通过DIMM-link @25GB/s），保证load balance。

## 78-BitMoD_Bit-serial_Mixture-of-Datatype_LLM_Acceleration.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：BitMoD提出**扩展非对称浮点数据类型**用于per-group LLM权重量化。核心设计：(1) 利用基本浮点格式（FP3/FP4）的符号-幅度表示中存在冗余零值（+0和-0），将冗余零替换为特殊值(special value)，构建扩展数据类型FP3-ER/FP3-EA（特殊值{±3, ±6}）和FP4-ER/FP4-EA（特殊值{±5, ±8}），ER增加额外分辨率(Extra Resolution)，EA增加额外非对称性(Extra Asymmetry)。(2) 细粒度数据类型自适应(Fine-grained Data Type Adaptation)：每个weight group（128个权重）从4个特殊值中选择使均方误差(MSE)最小的一个，配合基础FP值进行非线性量化（Algo.1: 遍历所有special value→非线性量化→计算MSE→选最低误差的特殊值）。(3) per-group INT8缩放因子（二级量化），实验证明INT8 per-group scaling factor无精度损失。BitMoD还支持INT8、INT6、FP4、FP3多种权重精度，与SmoothQuant/AWQ/OmniQuant等软件量化方法正交兼容——将原INT-Asym weight quantizer替换为BitMoD的扩展FP3/FP4数据类型即可。
  - 实验比较：(1) 4-bit和3-bit per-group权重量化下各数据类型的生成任务perplexity对比：BitMoD vs ANT vs OliVe vs MX-FP3/FP4 vs INT-Asym，6个LLM on Wikitext-2/C4；(2) 判别任务accuracy对比：BitMoD vs INT4-Asym vs INT3-Asym on HellaSwag/WinoGrande/Piqa；(3) 数据类型消融：FP4 vs FP4-ER vs FP4-EA vs BitMoD(ER+EA) on 3个Llama模型；(4) 特殊值消融：FP3的{±3,±6} vs {±5,±6} vs {±3,±5} on 4个模型；(5) BitMoD结合AWQ/OmniQuant vs QuaRot/GPTQ/AWQ/OmniQuant on 4/3-bit weight quantization；(6) BitMoD结合SmoothQuant(INT8 activation) vs INT-Asym+SmoothQuant。

- 硬件平台是什么，配置是什么。
  - GPU (量化实验): NVIDIA A6000 48GB。量化Llama-2-7B仅需～10秒。
  - 硬件评估: 自研加速器RTL (SystemVerilog) → Synopsys Design Compiler → TSMC 28nm工艺。Cycle-level simulator for end-to-end性能评估。DRAM power: DRAMSim3 DDR4模型。Buffer: CACTI建模(512KB activation buffer + 512KB weight buffer)。频率1 GHz。

- 模型是什么。数据集和bench分别是什么。
  - 模型: OPT-1.3B, Phi-2B, Yi-6B, Llama-2-7B, Llama-2-13B, Llama-3-8B。
  - 数据集/bench: Wikitext-2, C4 (生成任务perplexity); HellaSwag, WinoGrande, Piqa (判别任务accuracy, zero-shot via LM-Evaluation-Harness)。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  - 开源: https://github.com/yc2367/BitMoD-HPCA-25 (代码), DOI: 10.5281/zenodo.14252531 (artifacts)
  - 算法pipeline伪代码（per-group权重量化）:
    ```
    # Input: 原始FP16权重W ∈ R^{K×D}, 精度p ∈ {3, 4}, group size G=128
    # Output: 量化权重Wq, per-group特殊值选择SV, INT8 scaling factor Δ

    # Step 1: 按output channel分组，每G=128个权重为一组
    for each group Wg of size G in each channel:
      
      # Step 2: 获取基本量化值(含冗余-0)和特殊值候选
      if p == 3:
        basicValues = {0, ±1, ±2, ±4}         # FP3
        specialValues = {±3, ±6}               # FP3-ER + FP3-EA
      elif p == 4:
        basicValues = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}  # FP4
        specialValues = {±5, ±8}               # FP4-ER + FP4-EA

      # Step 3: 细粒度数据类型自适应 — 遍历特殊值选最优
      minError = +∞
      for sv in specialValues:
        quantValues = basicValues ∪ {sv}       # 替换冗余-0为特殊值sv
        Wq_tmp = NonLinearQuantize(Wg, quantValues)  # 非线性量化: 每个权重映射到最近quantValues值
        error = MSE(Wg, Wq_tmp)
        if error < minError:
          minError = error
          Wq_best = Wq_tmp
          bestSV = sv

      # Step 4: 计算INT8 per-group scaling factor
      Δ_max = max(|Wq_best|) / (2^{p-1} - 1)   # 对称量化scaling factor
      Δ = INT8_Quantize(Δ_max)                  # 二级量化至8-bit

      # Step 5: 存储 (Wq_best, bestSV_encoding(2-bit), Δ(8-bit))

    # 张量计算示例 — 推理时:
    # For each group g with G weights, activation A ∈ FP16^{G×1}:
    #   PartialSum = Σ_i (DEQUANTIZE(Wq[i], Δ) × A[i])  # FP16累加
    #   FinalOutput += PartialSum                        # per-channel累加
    ```
    BitMoD与AWQ/OmniQuant/SmoothQuant结合：将AWQ/OmniQuant的INT-Asym weight quantizer替换为BitMoD的扩展FP3/FP4数据类型，其余优化(weight clipping, scaling factor search, activation smoothing)保持不变。


## 84-MicroScopiQ: Accelerating Foundational Models through Outlier-Aware Microscaling Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  MicroScopiQ提出一种将剪枝与离群值感知量化相结合的PTQ框架。核心方法：(1) 使用3σ规则识别离群值(outliers)和内点(inliers)；(2) 离群值以2×精度使用MX-FP格式量化（如MX-FP-4_{8,8}），内点使用MX-INT格式量化（如MX-INT-2_{128}）；(3) 通过Hessian信息识别微块(μB=8)中最不重要的内点权重进行剪枝，将离群值的额外比特位重新分布到剪枝位置；(4) 使用Hessian-guided weight update补偿量化误差。实验比较离群值感知量化的两组baseline：Group A(GOBO, OWQ, SpQR, SDQ)保持离群值高精度但硬件效率低；Group B(OliVe, AWQ)使用统一精度但精度损失大。在W4A16、W2A16、W4A4、W2A8等量化配置下比较LLaMA2/3、OPT、Mixtral、Phi-3的WikiText2 PPL和6个LLM benchmark精度。EBW仅2.36bits(W2A16)。

- 硬件平台是什么，配置是什么。
  算法实现使用单张NVIDIA H100 GPU执行量化过程。量化时间根据模型大小从30分钟到9小时不等。

- 模型是什么。数据集和bench分别是什么。
  模型：LLMs—OPT(6.7B, 175B)、LLaMA-2(7B, 13B, 70B)、LLaMA-3(8B, 70B)、Mixtral-8x7B、Phi-3(3.8B, 14B)；VLMs—OpenFlamingo-9B、VILA-7B、LlaVa-1.5-7B；CNNs—ResNet50、VGG16；SSMs—VMamba-S、Vim-S。
  数据集：WikiText2(perplexity)；LLM benchmarks—BoolQ、HellaSwag、PIQA、ARC-c、MMLU、WinoGrande；VLM benchmarks—COCO captioning、VQAv2、VizWiz、TextVQA、GQA；CNN/SSM—ImageNet(Top-1 accuracy)。校准数据集为PILE数据集的256个随机样本。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码仓库：MicroScopiQ-LLM-Quantization.git（论文未提供完整URL）。算法pipeline伪代码：

  ```
  # Algorithm 1: MicroScopiQ Quantization Framework
  Input: W (d_row × d_col), calibration data X, H^{-1} = (2XX^T + λI)^{-1}
         row block (rB) = 128, macro-block (MaB) B_M = 128, micro-block (μB) B_μ = 8

  for each row block i: (0, rB, 2rB, ...):
    for j in row block:
      for each macro-block W_{j,MaB}:
        # Step 1: Separate inlier/outlier via 3σ rule
        W_in, W_out = sep_in_out(W_{j,MaB})

        # Step 1.2: Quantize inliers to MX-INT-b_{B_M}
        Q_in, I_sf = InlierQuant(W_in)  # MX-INT-2_{128} or MX-INT-4_{128}
        # Reduce outlier magnitude by 2^{I_sf} (pre-processing)
        W_out = W_out * 2^{I_sf}

        # Step 2: For each μB in MaB
        for W_{j,μB} in W_{j,MaB}:
          n = min(B_μ/2, NumOutliers(W_out_{μB}))
          M = {}  # pruned position list

          # Identify n least important inlier positions via Hessian
          for n iterations:
            p = argmin_{p in W_in_{μB}} (w_p^2 / [H^{-1}]_{pp})
            w_p = 0  # prune
            M = M ∪ {p}

          # Quantize outliers to MX-FP-b_{B_μ,B_μ}
          Q_out, O_sf = OutlierQuant(W_out_{μB}, I_sf)
          # O_sf = O_sf_l1 + μX - I_sf
          # Outliers: MX-FP-4_{8,8} (e1m2) or MX-FP-8_{8,8} (e3m4)

          # Step 3: Distribute outlier LSBs to pruned inlier positions
          perm = DistributeOutlierBits(Q_out, M)
          # Upper half: {sign, mantissa_1}; Lower half: {sign, mantissa_0}
          # Each outlier splits into two 2-bit halves placed at pruned
          # inlier positions + outlier original positions

        Q_{j,MaB} = Q_in + Q_out

    # Step 3.1-3.2: Compensate quantization error via Hessian
    E_{j-i,:} = (W_{j,:} - Q_{j,:}) / [H^{-1}]_{jj}
    W_{j:(i+rB),:} = W_{j:(i+rB),:} - E_{j-i,:} * H^{-1}_{j:(i+rB),j}

  # EBW calculation: if no outliers in μB→EBW=b_b; else→EBW=(perm_bits + 2*B_μ + O_sf_bits)/B_μ
  ```

  关键设计：(a) inlier scale factor (2^{I_sf}) 始终为负的2的幂→用于预缩放离群值以降低动态范围；(b) MX-FP的level-2 microExponent (μX)通过提取μB内所有离群值的公共指数获得；(c) 离群值拆分为Upper/Lower两半(sign+mantissa各2-bit)，分别放置在原离群值位置和剪枝位置→每个元素保持统一bit-budget (b_b)和数据格式(INT)，实现对齐内存访问；(d) 最终outlier scale factor: O_sf = O_sf_l1 + μX - I_sf。

## 88-Amove- Accelerating LLMs through Mitigating Outliers and Salient Points via Fine-Grained Grouped Vectorized Data Type.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**Amove量化框架**——一种数据类型与架构协同设计的LLM量化框架，通过Residual Approximation Mechanism和Fine-Grained Grouped Vectorized Data Type实现W4A4精度下保持近乎无损的模型准确率。核心设计：
    (1) **Residual Approximation Mechanism**：利用细粒度group-wise量化下scale factor分布的light-tailed特性（kurtosis < 3），以粗粒度group的shared base scale factor + shared residual + per-cluster compact encoding（2-bit）来近似每个fine-grained cluster的scale factor，避免存储per-cluster scale factor。公式：`S_ci = S_shared - R * E_ci`，其中S_ci是cluster i的scale factor，S_shared是base scale factor，R是shared residual，E_ci是per-cluster encoding。Residual的计算对权重使用search-based MSE最小化（搜索范围[-1,1]，粒度0.01），对激活使用平均偏差法（支持online quantization）。
    (2) **Fine-Grained Grouped Vectorized Data Type**：类似MX数据格式但引入shared residual和encodings。每个coarse-grained group包含：shared base scale (FP8 E4M3)、shared residual (FP8 E4M3)、K/C个per-cluster encodings (2-bit)、K个quantized elements (INT4/FP4等)。支持两种配置：Amove-Aggressive (linear: g=128, c=16; attention: g=32, c=4) scale factor overhead 0.25~1 bit/value；Amove-Conservative (g=32, c=4 uniform) scale factor overhead 1 bit/value。
    (3) **双模式支持**：weight-activation quantization (W4A4，quantize weights沿channel维、activations沿token维) 和 low-bit weight-only quantization (W3A16/W2A16，仅quantize weights)。
    (4) **Scale factor memory overhead**：公式为 `(R_bits + S_bits)/K + B_bits/C`，Amove-Aggressive在g=128,c=16下仅0.25 bits/value，相比group-wise quantization (16/K) 降低16×，相比MX format (8/K) 降低8×。
  - 实验比较：
    (1) Perplexity (Wikitext2, C4)：Amove-Aggressive/Amove-Conservative vs ANT / OliVe / Tender / INT-Sym / MX-FP4 (W4A4) 以及 vs ANT / INT-Sym / MX-FP3 (W3A16) 和 vs MX-FP2 (W2A16)，覆盖 Vicuna-7B-1.5 / Vicuna-13B-1.5 / OPT-6.7B / OPT-13B / Llama-7B / Llama2-7B；
    (2) Discriminative accuracy (WinoGrande, Piqa)：相同baselines，覆盖 OPT-13B / Vicuna-7B-1.5 / Llama2-7B；
    (3) Zero-shot MMLU (W4A4)：Vicuna-13B-1.5 和 Llama2-7B，VS FP16 / ANT / OliVe / INT-Sym / MX-FP4；
    (4) 集成实验：GPTQ/AWQ/OmniQuant + Amove (W3A16, g=64) 以及 M-ANT + Amove (W4A4, g=64)，Wikitext2/C4 perplexity对比同等bit-width下的原始方法。
  - 硬件平台是什么，配置是什么：
    GPU tensor core架构（基于NVIDIA Ampere A100架构为baseline，64 thread groups）；Systolic array accelerator（16×16 tile，每个tile含4个PE，output-stationary dataflow，32×32 PE配置，256KB activation buffer + 256KB weight buffer，DDR4 DRAM）。
  - 模型是什么。数据集和bench分别是什么。
    模型：Llama-7B/13B、Llama2-7B、Llama3-8B、OPT-1.3B/6.7B/13B、Vicuna-7B-1.5/13B-1.5、Bloom-3B。数据集/benchmark：Wikitext2、C4（perplexity）；WinoGrande、Piqa（discriminative accuracy）；MMLU（zero-shot accuracy）。
  - 开源情况：论文未提供开源代码。量化框架在PyTorch中实现。GPU模拟器基于修改的Timeloop（参考TB-STC和LUT-Tensor Core建模tensor core行为）；Accelerator模拟器基于修改的BitMoD simulator（扩展支持weight-activation quantization）；DRAM模拟使用Ramulator和DRAMPower/DRAMSim3；on-chip buffer面积使用CACTI。开源情况无法确认。

  量化算法伪代码（Residual Approximation Algorithm）：
  ```
  Input: Coarse-grained group matrix G; Encoding bit-width E;
         Quantization bit-width b; Cluster size C
  Output: Residual R

  // Step 1: Partition into Clusters
  Divide G into K clusters {C_1, C_2, ..., C_K}, K = |G|/C;

  // Step 2: Compute Cluster-wise Scales
  for i ← 1 to K do
      Δ_i = max(|X_f^(i)|) / (2^(b-1) - 1);  // scale factor for cluster i
  end

  // Step 3: Select Base Scale
  Set base scale Δ_base = max{Δ_1, ..., Δ_K};

  // Step 4: Compute Residual
  if G is activation data then
      // Online: average deviation (for calibration/distribution shift robustness)
      R = (1/(C*E)) * Σ_{i=1}^{K} |Δ_i - Δ_base|;
  else
      // Offline (weight): search-based MSE minimization
      Define search range R ∈ [M, N] with search step Q;
      Construct candidate set: R = {R | R = M + k*Q, ...};
      R = arg min_{R∈R} Σ_{i=1}^{K} (Δ_i - (Δ_base - e_i * R))^2;
      // e_i = floor((Δ_i - Δ_base) / R), e_i ∈ [-2^{E-1}-1, 0]
  end

  // Step 5: Quantization with cluster-wise scale factors
  for each cluster i:
      S_ci = S_shared - R * E_ci;  // recover cluster scale factor
      X_qci = round(X_ci / S_ci);   // quantize
      X_hat_ci = X_qci * S_ci;      // dequantize
  end
  ```

  张量计算示例（W4A4 GEMM with Amove data type）：
  ```
  // Weight: W in Amove format (group_size=32, cluster_size=4)
  // Activation: A in Amove format (group_size=32, cluster_size=4)
  // Output: O = D(A_q × W_q, S_a, S_w) + bias
  // where D(·) applies residual-based dequantization per cluster

  for each group g in output channels:
      // Load Amove metadata for this group
      S_base_w = W_metadata[g].base_scale    // FP8
      R_w = W_metadata[g].residual            // FP8
      E_w[] = W_metadata[g].encodings         // 2-bit per cluster

      for each input token group:
          S_base_a = A_metadata[t].base_scale
          R_a = A_metadata[t].residual
          E_a[] = A_metadata[t].encodings

          // Recover per-cluster scales
          for cluster i in 0..7:  // 32/4 = 8 clusters
              S_wi = S_base_w - R_w * E_w[i]   // dequant weight scale
              S_ai = S_base_a - R_a * E_a[i]   // dequant activation scale

          // INT4 MAC + dequantization
          partial_sum = INT4_MAC(A_q[t, g, cluster_i],
                                  W_q[g, cluster_i])
          O[t, g] += partial_sum * S_wi * S_ai
  ```


## 89-AQPIM_Breaking_the_PIM_Capacity_Wall_for_LLMs_with_in-Memory_Activation_Quantization.pdf

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**AQPIM**——基于Product Quantization (PQ) 的PIM感知激活量化框架，将KV cache压缩后直接在PIM内存中计算attention。核心算法设计：
    (1) **PQ-based KV cache量化**：将高维key/value向量分解为m=32个子向量(subvectors)，每个子向量空间用K=512个centroids进行k-means聚类量化。向量分解为子向量→各子向量独立聚类→生成codebook(centroids)和indices(centroid assignments)。
    (2) **Importance-weighted k-means聚类**：在标准k-means中加入注意力分数权重w（对最后t=32个token的attention scores求和），使高注意力token在聚类时获得更小的量化误差。加权目标函数：μ_k = Σ(w_n·x_n)/Σw_n (n∈C_k)。
    (3) **Channel sorting预处理**：通过cosine similarity对channel进行排序分组，将高相关性的channel聚集到同一subvector中，提高聚类内聚性。排序矩阵P_k/P_v离线生成（校准数据集：Wikitext-2-v1），吸收到projection矩阵中（W_q' = W_q·P_k, W_k' = W_k·P_k, W_v' = W_v·P_v, W_o = W_o·P_v^T）。
    (4) **PQ-based attention计算**（核心）：将GEMV q·K^T 转换为查表和求和操作：query按m个子向量切分①→各子query与对应key codebook子矩阵乘（生成m×K inner product matrix）②→用key indices查表取值③→沿子向量轴求和得近似qK^T④→softmax⑤→用value indices和value codebook重建value矩阵⑥→attention scores × value矩阵⑦。完全跳过显式矩阵乘法和解量化。
    (5) **Page-aware windowed clustering**：将序列划分为多个窗口，每个窗口内最多512个centroids，确保centroids数据fit在单个DRAM row(1KB=512×FP16)内，保证indirection查找全部为row-buffer hit。
  - 实验比较：(1) LongBench accuracy vs Memory Reduction Ratio（Mistral-7B-Instruct-v0.2 / Llama-3.2-3B-Instruct），AQPIM vs SnapKV / PQCache / SKVQ，覆盖NarrativeQA/HotpotQA/GovReport/TREC/PRetrieval/LCC六个任务；(2) Ablation：Standard PQ vs w/o weighting vs w/o pre-sort vs AQPIM（K=128 centroids高压缩场景）；(3) Subvector数m∈{2,4,8,16,32,64} grid search；(4) Centroids数K∈{64,128,256,512,1024} grid search。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA H100 GPU
  - PIM：4×16GB HBM-PIM modules（基于HBM3架构，带BankPE和BufferPE）
  - 额外：1×16GB conventional HBM（存储模型参数等）
  - CPU：Intel Xeon Platinum 8480+
  - 模拟器：Customized GPU-PIM simulator（基于AttAcc! simulator [55]，后者是Ramulator [20][36][48]的修改版）
  - PDK：ASAP7 [9]（7nm FinFET predictive process design kit）
  - 数值精度：模型用bfloat16，架构模拟用FP16（与AttAcc!可比）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mistral-7B-Instruct-v0.2 (32K context)，Llama-3.2-3B-Instruct (128K context)
  - Benchmark：LongBench（含NarrativeQA、HotpotQA、GovReport、TREC、PassageRetrieval-en、LCC六个代表性任务）
  - 校准数据集：Wikitext-2-v1（仅用于channel sorting矩阵离线生成）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：Simulator https://github.com/scalesnu/attacc_simulator；数据 https://zenodo.org/records/17378113（UAMP可视化数据）
  - AQPIM 算法pipeline伪代码：
    ```
    # === Prefilling Phase ===
    # GPU generates QKV matrices, offloads KV to HBM-PIM
    # PIM performs codebook generation in parallel with GPU attention/projection/FFN

    # Step 1: Channel Sorting (offline, absorbed into projection weights)
    W_q' = W_q @ P_k   # P_k: key channel sorting matrix
    W_k' = W_k @ P_k
    W_v' = W_v @ P_v   # P_v: value channel sorting matrix
    W_o = W_o @ P_v.T

    # Step 2: Vector Splitting (during prefilling)
    for each attention head:
        K_split = split(K, m=32 subvectors along d_head)  # (N, d/m) × m
        V_split = split(V, m=32 subvectors along d_head)

    # Step 3: Importance-Weighted k-means (PIM BankPE+BufferPE, 4 iterations)
    S = softmax(Q @ K.T)  # attention score matrix (N × N)
    w = sum(S[-32:, :], axis=0)  # weights from last 32 tokens

    for iter in range(4):  # 4 iterations sufficient
        # Distance Calculation (BankPE, all banks)
        for each subvector s in range(m):
            dist = ||K_split[:,s,:] - centroids[s]||²  # Euclidean

        # Cluster Assignment (BufferPE)
        assignments = argmin(dist)

        # Centroid Calculation: weighted update
        for k in range(K=512):
            centroids[s,k] = sum(w[n] * K_split[n,s,:] for n in C_k) / sum(w[n] for n in C_k)

    # Step 4: Store compressed KV
    key_codebook[s,k,:] = centroids[s,k]    # (m, K, d/m)
    key_indices[s,n] = assignments[s,n]     # (m, N)
    value_codebook[s,k,:] similarly
    value_indices[s,n] similarly

    # === Decoding Phase ===
    # GPU generates qkv vectors, offloads to PIM
    # PIM appends new indices, computes PQ-based attention

    # Step 5: PQ-based Attention (no dequantization)
    # ① Split query into m subvectors
    q_split = split(q, m)  # (m, d/m)

    # ② Inner product matrix: each subquery × key codebook submatrix
    inner_prod = zeros(m, K)
    for s in range(m):
        inner_prod[s,:] = q_split[s] @ key_codebook[s].T  # (1, d/m) × (K, d/m)^T

    # ③ Lookup with key indices (intra-row indirection, single row activation)
    qK_T_approx = zeros(1, N)
    for n in range(N):
        for s in range(m):
            k_idx = key_indices[s, n]
            qK_T_approx[0, n] += inner_prod[s, k_idx]

    # ④ Softmax
    attn_scores = softmax(qK_T_approx)

    # ⑤ Reconstruct value matrix (similar lookup)
    V_recon = zeros(1, d_head)
    for s in range(m):
        for n in range(N):
            v_idx = value_indices[s, n]
            V_recon[s*sd:(s+1)*sd] += attn_scores[0,n] * value_codebook[s, v_idx]

    # ⑥ Output
    attention_output = V_recon  # sent to GPU for projection/FFN
    ```

  - **压缩率计算**：原始KV：N×d_head×2(key+value)×2B(FP16) = 4Nd bytes；压缩后：codebook m×K×d/m×2B + indices m×N×log₂(K)/8 ≈ Kd×2 + mN bytes。当K=512, m=32, d=128时，达到约6.53×压缩。
