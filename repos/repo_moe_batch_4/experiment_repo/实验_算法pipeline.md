## Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**MoE 专家权重的 weight-only 量化（INT4/INT8）**——仅对 MoE 层的 expert weight matrices 进行对称、range-based per-channel 量化，不量化 activations 和 biases。所有非 expert 参数和中间激活保持 FP16。量化后使用 fused GEMM+Dequantize kernel（将 dequantize 融合进 CUTLASS Grouped GEMM），避免单独的 dequantize kernel 引入额外内存读写。INT4/INT8 均使用相同的量化方案：对形状为 (E, M, N) 的 expert weights 生成 scales (E, 1, N)，推理时将量化权重 dequantize 回 FP16 后进行浮点矩阵乘法。针对 INT 到 FP16 转换慢的问题，提出基于 FP16 位操作（mantissa 直接编码整数 + 0x6400 减法）的快速 I2F 转换替代原生的 IntToFloat 指令。
  - 实验比较：(a) INT8/INT4 fused GEMM+Dequantize vs FP16 GEMM 在不同 active experts 数量（1/4/8/16/24/32）下的归一化吞吐量（Table 1）；(b) INT8/INT4 量化 vs FP16 baseline 的 BLEU 分数差异（Table 2, EN-DE/DE-EN/10 语言对平均）；(c) 端到端推理吞吐对比：Torch-FP16 vs FT-FP16 vs FT-INT8 vs FT-INT4 在不同 batch size（1/8/20/32/64/96）和 beam（1/2）下的每秒处理输入 tokens 数（Table 3）。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA PCIE V100，Docker 容器运行 Ubuntu 20.04 + CUDA 11.6，代码由 nvcc + gcc/g++ 9.3 编译。

- 模型是什么。数据集和bench分别是什么。
  - 模型：encoder-decoder MoE Transformer（Deep encoder, shallow decoder），embedding dim 1024，FFN hidden dim 4096，24 encoder layers，12 decoder layers，32 experts，top-1 gating（Switch Transformer 风格），TUPE attention，总参数约 5B（FP16 下约 10GB）。
  - 数据集：WMT 公开数据集，10 语言对的 multilingual machine translation，vocabulary 128K（SentencePiece tokenizer），训练数据约 4B sentence pairs。
  - Benchmark：EN-DE 和 DE-EN 翻译，1000 tokenized English sentences（约 40K tokens），metric 为 BLEU 和 throughput（input tokens/sec）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文基于 NVIDIA FasterTransformer（https://github.com/NVIDIA/FasterTransformer）开源框架实现，CUTLASS（https://github.com/NVIDIA/cutlass）、CUB（https://github.com/NVIDIA/cub）和 Triton Inference Server（https://github.com/triton-inference-server/server）均是开源项目。论文自身未提供独立开源仓库。
  - 量化算法 pipeline（从训练后模型到 INT4/INT8 推理）：
    ```
    # === Step 1: 量化 Expert 权重（训练后，离线） ===
    # 输入：expert weights W_fp16，形状 (E, M, N)，E=32 experts
    # 对每个 expert e 的每个输出 channel n：
    for e in range(E):
        for n in range(N):
            # 对称 per-channel 量化
            max_abs = max(|W_fp16[e, :, n]|)
            scale[e, n] = max_abs / max_val    # max_val = 127 (INT8) or 7 (INT4)
            W_quant[e, :, n] = round(W_fp16[e, :, n] / scale[e, n])
            W_plus = W_quant + offset           # offset = 128 (INT8) or 8 (INT4), 转为无符号
    # 输出：W_plus (E, M, N) INT4/INT8, scales (E, 1, N) FP16

    # === Step 2: 推理时 Fused GEMM + Dequantize ===
    # 对于每个 MoE layer 的 Grouped GEMM 调用：
    # 输入 activation A 通过 CUB radix sort 路由到各 expert
    # 对每个 expert e（有该 expert 的 tokens）：
    for each expert e with active tokens:
        # 在 GEMM kernel 内部 fused 执行 dequantize：
        for each weight tile:
            # 加载 INT8 权重（4个 INT8 → 1个 32-bit reg）
            w_plus_int8 = load_4_int8(W_plus[e, tile_m:tile_m+4, tile_n])
            # 构造 FP16：fp16_repr = (0x6400 | w_plus_int8[i])
            # 即 (1024 + w_plus_int8[i]) 的 FP16 表示
            fp16_val = fp16_subtract(fp16_repr, 1152.0)  # 减去 1024+128
            # = w_original_float = int_to_float(w_quant)
            # 乘以 scale
            dequant_weight = fp16_val * scale[e, n]
            # 标准 FP16 GEMM
            C[e, :, :] += A_token @ dequant_weight
    # 输出：FP16 MoE layer output，与原 FP16 模型相同的激活精度
    ```
  - INT4 额外优化：权重 layout 重排 `[e0,e1,e2,e3,e4,e5,e6,e7] → [e0,e2,e4,e6,e1,e3,e5,e7]` 减少 bit 操作指令，减去的常数从 1152 变为 1032。

## Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：(1) **HSDL (Hierarchical Sparse Dictionary Learning)**——从 MoE LLM 的 expert activation matrix 出发，通过层级稀疏字典学习递归分解字典矩阵 $D_k \approx D_{k+1} \cdot R_{k+1}$，从粗到细地揭示专家之间跨层的协作模式（collaboration patterns）。引入三个约束：稀疏性约束 $L_{\text{sparse}} = ||R_{k,i,:}||_{\infty}$、层间一致性约束 $L_{\text{hier}}$、重构误差项 $L_{\text{rec}}$，总损失 $L_{\text{total}} = L_{\text{sparse}} + \lambda_1 L_{\text{hier}} + \lambda_2 L_{\text{rec}}$。(2) **CAEP (Contribution-Aware Expert Pruning)**——基于 HSDL 发现协作模式后，利用稀疏表示矩阵 R 和字典矩阵 D 计算每个专家的贡献分数 $\mathbf{e} = \sum_{i=1}^{N_p} \mathbf{D}_{\text{sum},i}$，通过初始阈值 mask + 迭代移除最少使用的 pattern 来逐步剪枝低贡献专家，直到达到目标剪枝比例。
  - 实验比较：(a) HSDL 发现的协作模式与穷举搜索（exhaustive search）的 pair/triplet 高频组合的覆盖度对比（Top-k% Coverage）；(b) 不同领域（数学、计算机科学、物理、法律、心理学）的专家激活频率分布和 cosine similarity 混淆矩阵；(c) CAEP vs Random/SEER-MoE/GEM 剪枝方法在 25% 专家删除后的 benchmark 性能（AVG/OBQA/ARC-C/HellaSwag/WinoGrande/RTE/PIQA）；(d) CAEP 在不同剪枝比例（25%/50%）下的性能退化曲线；(e) 按特定领域剪枝 50% 专家后在各领域的准确率退化热力图。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明硬件平台和 GPU 配置。

- 模型是什么。数据集和bench分别是什么。
  - 模型：phi-moe（协作模式挖掘实验，Section 4.3.1）；DeepSeek-MoE-16B（剪枝实验，Section 5 及 Appendix C，仅剪枝 normal experts，保留 shared experts）。
  - 数据集：(a) 协作模式挖掘——MMLU-pro 数据集 2,812 个样本，覆盖数学、计算机科学、物理、法律、心理学 5 个领域；(b) 剪枝实验——MMLU 数据集 128 样本，输入序列长度 2,048 tokens（遵循 He et al., 2024 的设置）。
  - Benchmark：使用 EleutherAI LM Harness 框架评估，包含 ARC-C、BoolQ、HellaSwag、MMLU、OBQA、PIQA、RTE、WinoGrande，报告 normalized zero-shot accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明代码开源链接。
  - 算法 pipeline（从 expert activation 提取到剪枝的全流程）：
    ```
    # === 阶段一：Expert Activation Data Collection ===
    # 输入：MoE LLM（m 层，n 专家），数据集 S（N_s 样本）
    # 对每个样本 i 的第 t 个 token，记录 router 分配 α(i)_{t,j,k}
    # 句子级激活值聚合：
    v_{i,j,k} = Σ_{t=1}^{T} α(i)_{t,j,k}      # 式(1)
    # 构造 expert activation matrix：
    X ∈ R^{N_e × N_s}                           # 式(2) N_e = m × n

    # === 阶段二：HSDL 层级稀疏字典学习 ===
    # Layer 1: 对 X 做稀疏字典学习
    X ≈ D_1 · R_1                               # D_1 ∈ R^{N_e × N_p}, R_1 ∈ R^{N_p × N_s}
    # Layer k+1: 递归分解上一层字典
    D_k ≈ D_{k+1} · R_{k+1}                     # 式(3)
    # 损失函数（式(4)-(7)）：
    L_total = L_sparse + λ_1 * L_hier + λ_2 * L_rec
    # 输出：多层字典 {D_k} 和稀疏编码 {R_k}
    # D_k 的每个 atom 代表一组专家协作模式

    # === 阶段三：CAEP 剪枝（Algorithm 1）===
    # 输入：字典矩阵 D，稀疏表示矩阵 R，阈值比 k_1，目标剪枝比 k_2
    R_sum = Σ_{j=1}^{N_s} R_{:,j}               # 对样本维度求和
    D_sum = D · R_sum^T                          # 专家-模式贡献矩阵
    e = Σ_{i=1}^{N_p} D_sum[:,i]                 # 每个专家的总贡献分数
    e_sorted = sort_descending(e)
    threshold = e_sorted[ceil(k_1 * N_e)]         # k_1-分位数阈值
    m = 1_{e ≥ threshold}                        # 初始二值 mask
    while ||m||_0 > (1 - k_2) * N_e:             # 未达到目标剪枝比
        i* = argmin_i R_sum[i]                   # 找到最少使用的 pattern
        remove column i* from D, row i* from R
        recompute R_sum, D_sum, e
        m = 1_{e > threshold}                    # 更新 mask
    return m                                     # 保留=1，丢弃=0
    ```
  - 核心思想：字典每个 atom 编码一组跨层专家（如图 2 中 Layer 5 Expert 21 和 Layer 6 Expert 3 的共激活模式），稀疏编码 R 控制各 pattern 在不同样本上的参与度。剪枝时优先移除贡献分数低于阈值的专家，同时在迭代中移除最少被使用的协作模式，确保保留高贡献的专家组合。

## Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：首次将 MoE 架构应用于统一多模态大语言模型（MLLM），构建 **Uni-MoE**，可处理图像、视频、音频、语音和文本五种模态。核心设计包括：(1) **模态特定编码器+连接器**——使用 CLIP-V（视觉）、Whisper-small（语音）、BEATs（音频）作为冻结编码器，通过 MLP 线性投影（视觉）和四层 Q-Former+线性投影（音频/语音）将各模态映射到 LLM 语言空间；(2) **稀疏 MoE LLM 架构**——基于 Vicuna-7B（LLaMA-7B），将部分 FFN 层替换为稀疏 MoE 层，每层包含 4 或 8 个专家，使用线性 Router 进行 token 级别的 Top-k（k=2）专家选择；(3) **三阶段渐进式训练**——阶段一：跨模态对齐（仅训练连接器），阶段二：训练模态特定专家（对各模态数据分别训练对应专家 FFN），阶段三：使用 LoRA 在混合多模态数据上联合微调整个 Uni-MoE；(4) **LoRA 高效微调**——阶段二 rank=64/alpha=16，阶段三 rank=8/alpha=16，仅微调 LoRA 参数和投影层而不更新全部专家参数。
  - 实验比较：(a) Uni-MoE 与 Dense 统一多模态模型（Macaw-LLM、X-InstructBLIP）在语音-图像理解（A-OKVQA、OK-VQA、VQAv2 语音版、MMBench-Audio、RACE-Audio、EHSL）、音频理解（ClothoAQA、ClothoV1/V2 CIDEr）、图像-文本理解（A-OKVQA、OK-VQA、VQAv2、MMBench）、视频 QA（ActivityNet-QA、MSVD-QA）上的性能；(b) Uni-MoE MoE 变体 vs 单专家（Dense）变体在不同模态任务上的对比；(c) Top-k 值（1 vs 2）、专家数量（1/2/4/8, mixture vs pure）、MoE 层插入方式（First-Half/Second-Half/Interval/All）的消融实验；(d) 纯 MoE（相同初始专家）vs 混合 MoE（预训练模态特定专家）的对比；(e) 辅助平衡损失（auxiliary balancing loss）在 pure MoE 和 mixture MoE 场景下的效果。

- 硬件平台是什么，配置是什么。
  - 阶段一（跨模态对齐）：2 块 NVIDIA A100 GPU，global batch size=32
  - 阶段二（专家训练）：2 块 NVIDIA A100 GPU，global batch size=16
  - 阶段三（MoE 训练）：1/8/16 块（2 节点）NVIDIA A800 GPU，支持数据并行+专家并行+多节点多 GPU 并行训练

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 Vicuna-7B（LLaMA-7B），Uni-MoE-7B×4-Top2（4 专家，激活 8.9B/总 13.2B）、Uni-MoE-7B×4-Top2†（32 MoE 层，激活 11.1B/总 19.7B）、Uni-MoE-7B×8-Top2（8 专家，激活 8.9B/总 21.9B）、Uni-MoE-7B×8-Top2†（32 MoE 层，激活 11.1B/总 37.0B）。编码器：CLIP-V（视觉）、Whisper-small（语音）、BEATs（音频）。
  - 训练数据集：Common Voice（短语音 1.7M）、LLaVA-Instruct-150K（含 TTS 转换的语音-图像变体 I-A 和原始文本-图像 T-I）、LibriSpeech（长语音拼接）、RACE（文本转长语音 T-A）、WavCaps/AudioCaps/ClothoV1/ClothoAQA/MELD（音频字幕和问答）、Video-Instruct-Dataset（视频指令 100K）、LLaVA-v1.5-665K（扩展图像-文本）。
  - 评估 Benchmark：A-OKVQA、OK-VQA、VQAv2（语音版，EM 指标）；MMBench-Audio（语音-文本-图像长语音推理）；RACE-Audio（长语音多选）；English High School Listening Test/EHSL（真实高考英语听力）；ClothoV1/V2（CIDEr 指标）、ClothoAQA（EM 指标）；ActivityNet-QA、MSVD-QA（准确率+Score，Video-ChatGPT 评估方法）；图像-文本 sanity check：MMBench、POPE、SEED-Bench、MM-Vet。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源地址：https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs
  - 算法 pipeline（以视频理解为例，跨模态 tokens → MoE LLM 推理）：
    ```
    # 第一步：模态编码和投影（不同模态并行编码）
    I = MLP(CLIP-V(image))              # 图像 token，式(2)
    V = Mean(CLIP-V([I_1,...,I_8]))     # 视频 8 帧平均池化，式(3)
    A = Audio-QFormer(BEATs(audio))     # 音频 Q-Former 蒸馏，式(4)
    S = Speech-QFormer(Whisper(speech)) # 语音 Q-Former 蒸馏，式(5)
    T = Word-Embedding(text)            # 文本嵌入，式(6)
    
    # 第二步：拼接多模态序列
    x_0 = [V_1,...,V_N; T_1,...,T_z; A_1,...,A_k]  # 式(12)
    
    # 第三步：MoE LLM 逐层前向（以第 l 层为例）
    X_l^s = MSA(LN(X_{l-1})) + X_{l-1}             # 共享自注意力，式(13)
    X_l^M = MoE(LN(X_l^s)) + X_l^s                  # 稀疏 MoE FFN，式(14)
    x_l = LN(X_l^M)                                  # 层归一化，式(15)
    
    # MoE 层内部（token 级路由）：
    P(X_l^s)_i = softmax(W_router · X_l^s)_i        # Router 计算专家概率，式(16)
    MoE(X_l^s) = sum_{i=1}^{k} P(X_l^s)_i · e_i(X_l^s)_i  # Top-k 专家加权，式(17)
    
    # LoRA 微调每个专家（不更新全量参数）：
    h_e1_LoRA = LoRA-e_1(X_E1) = (W_0 + B·A) · X_E1  # 式(19)-(20)
    h_e1 = e_1(X_E1) + h_e1_LoRA                      # 式(21)
    ```
  - 训练 pipeline 三阶段详见 Algorithm 1（论文原文伪代码）。阶段一仅训练 Connector 参数（Q-Former + 线性投影层），loss 为交叉熵生成损失；阶段二复制阶段一权重，训练各模态对应的单个专家 FFN（LoRA + 投影层）；阶段三加载阶段二预训练专家权重到 MoE 层，使用 LoRA 在混合多模态数据上联合微调，冻结 LLM 全部参数（包括专家），仅更新 LoRA 参数、Router 和投影层。

## Tutel Adaptive Mixture-of-Experts at Scale

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：(1) **自适应并行切换**——通过统一的张量分布布局（ZeRO-DP Stage-3 风格的分片），支持 DP（r=0）和 EP+DP+MP（r=1 到 r=⌈W/E⌉）之间的零成本运行时切换（无参数迁移、无张量重整开销）。通过通信复杂度分析（Table 4）将 7 种并行策略缩减为两种等价覆盖策略。(2) **自适应流水线**——根据 capacity factor f 动态选择最优的流水线度（d∈{1,2,4,8}）和 All-to-All 算法（Linear 或 2DH）。(3) **字典式最优策略查找**——预构建 hash map ⌊c/R⌋ → {r*, d*, a*}，运行时通过 O(1) 查表选择最优并行度和流水线策略。
  - 实验比较：(a) adaptive:r（DP ↔ EP+DP ↔ EP+DP+MP）在不同 capacity factor f（1.0~8.0）下的吞吐量对比（Figure 12），Base (H=2K) 和 Large (H=32K) 两种配置；(b) adaptive pipelining 在 243 种 MoE 模型配置（E_g∈{0.5,1,2}, D∈{1024,2048,4096}, H∈{1024,2048,4096}, tokens/step∈{4096,16384,65536}）下 16~256 GPU 的平均提升（Table 6a/6b）；(c) 单 MoE 层 scaling（16→2048 GPU）的优化逐项叠加分解（Figure 14），从 Fairseq baseline → +Kernel → +2DH A2A → +Flexible A2A → +Adaptive Pipelining → 最终 4.96×/5.75× speedup；(d) SwinV2-MoE 端到端训练/推理速度对比（Table 7, 8~128 GPUs），训练 1.14×~1.55× 加速，推理 1.95×~2.11× 加速；(e) SwinV2-MoE vs 稠密 SwinV2 准确率对比（Table 8）。

- 硬件平台是什么，配置是什么。
  - Azure Standard_ND96amsr_A100_v4 VMs：每 VM 配备 8× NVIDIA A100 SXM 80GB GPU，8× 200 Gbps HDR InfiniBand，96× 2nd-gen AMD Epyc CPU cores，1.9 TiB 内存。节点内 GPU 通过 3rd-gen NVLink + NVSwitch 互联，节点间通过 1,600 Gbps InfiniBand non-blocking 网络（adaptive routing）。实验规模最大 2,048 A100 GPUs (256 VMs)。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) SwinV2-MoE（Swin Transformer V2 的 MoE 版本，每两个 FFN 层替换为 MoE 层，前两个 stage 除外），SwinV2-S（~65.8M active params）和 SwinV2-B（~109.3M active params）两种 size，E=8~128 experts，top-k=1/2，capacity factor f=1.0/1.25；(2) 合成 MoE 配置用于 micro-benchmark：fflayer hidden size H∈{1K,2K,4K,16K,32K}，fflayer channel size D∈{1K,2K,4K}，E_g∈{0.5,1,2} local experts per GPU，tokens/step∈{4096,16384,65536}。
  - 数据集：ImageNet-22K（14.2M images, 22K classes）预训练；ImageNet-1K 微调/5-shot 线性评估；COCO object detection。
  - Benchmarks：ImageNet-22K acc@1、ImageNet-1K acc@1 (ft)、ImageNet-1K 5-shot acc@1、COCO box/mask AP。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/microsoft/tutel，已集成到 Fairseq 和 DeepSpeed。
  - 算法 Pipeline 核心——自适应并行切换的伪代码：

```
# === 自适应并行选择（运行时，O(1) 查表） ===
c = current_capacity_factor * k * T / E     # 当前 expert capacity
key = floor(c / R)                           # R=128 窗口大小
r_opt, d_opt, a_opt = dictionary[key]        # 查表得最优参数

# === Switchable DP (r=0): 等效 ZeRO-DP Stage-3 ===
# forward: all-gather weights across W GPUs
W_sliced = all_gather(W_local, group=range(W))
output = expert_ffn(W_sliced, local_tokens)
# backward: reduce-scatter gradients
grad_W = reduce_scatter(grad_W_local, group=range(W))

# === Switchable EP+DP+MP (r in [1, ceil(W/E)]): ===
# 将 W GPUs 分为 W/(ceil(W/E)/r) 组，组内 DP，组间 MP
group_size = ceil(W/E) / r
# Step 1: LOCAL_REPEAT — 生成 r 份 gating 结果副本
gating_replicated = repeat(gating_result, r)  # shape: (T*r, ...)
# Step 2: All-to-All dispatch (基于 replicated gating)
dispatched = all_to_all(dispatch_input)
# Step 3: Expert FFN 计算
expert_out = expert_ffn(dispatched)
# Step 4: All-to-All combine
combined = all_to_all(expert_out)
# Step 5: LOCAL_SUM — 对 r 份输出求和
output = reduce_sum(combined.reshape(r, T, ...), dim=0)
# Step 6: DP All-Gather（仅在 group_size > 1 时）
if group_size > 1:
    W_sliced = all_gather(W_local, group=groups)
```

  - 自适应流水线核心——Token 分区多流重叠（以 degree=2 为例）：

```
# 输入: (E, C_g, D) 沿 C 维度拆分为 C_0 和 C_1
C_0, C_1 = split(input, dim=C, partitions=2)
# Stream 0: C_0 → A2A_dispatch → Expert_FFN → A2A_combine
# Stream 1: C_1 → A2A_dispatch → Expert_FFN → A2A_combine
# 两流异步执行，A2A (通信流) 与 Expert FFN (计算流) 互相重叠
stream0: A2A_dispatch(C_0) → Expert_FFN(...) → A2A_combine(...)
stream1: A2A_dispatch(C_1) → Expert_FFN(...) → A2A_combine(...)
barrier()  # 等待所有流完成
output = concat([C_0_out, C_1_out], dim=C)
```

  - 字典构建（预搜索，执行一次）：

```
# 对每个 key = floor(c/R):
for r in TernarySearch([1, ceil(W/E)-1]) + [0, ceil(W/E)]:
    for d in {1, 2, 4, 8}:
        for a in {Linear, 2DH}:
            measure throughput(r, d, a, key)
dictionary[key] = argmax_r,d,a(throughput)
```

## TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **Expert Group Approximation（专家组近似）** 方法，在 MoE 训练中为路由器提供稠密梯度信号。核心思想：MoE 的 Top-K 路由只激活 K 个专家，未被路由到的专家不产生梯度，导致路由器学习信号稀疏。论文提出用已被路由到专家 i 的其他 token 的输出来近似 token x 对专家 i 的输出 $E_i(x)$。具体公式：$\forall x \in X_R : \hat{E}_i(x) = \frac{1}{K} \sum_{j \in R} \frac{1}{|X_{\{i,j,\cdot\}}|} \sum_{x' \in X_{\{i,j,\cdot\}}} E_i(x')$，产生 $N^2$ 个总近似。在前向传播中保持不变，在反向传播中通过 stop-gradient 操作将近似梯度注入：$y := y + y' - \operatorname{sg}(y')$。同时更新路由器参数和专家参数以保证一致性。
  - 实验比较：(a) Expert Group Approximation vs Top-K (K=2) baseline 在 FineWeb 200B tokens 上的 training loss 和 validation perplexity（Figure 5）；(b) 多 benchmark 评估：mathqa, logiqa2, mmlu, openbookqa, logiqa, arc challenge, arc easy, hellaswag, copa, piqa（Table 1，平均 +0.9%）；(c) 与 Sparsemixer (Liu et al., 2023) 对比（Section 4.3）；(d) Expert scaling（8 vs 32 experts）和 Batch Size scaling（$2^{19} / 2^{20} / 2^{21}$）ablation（Table 2），改善随稀疏度和 batch size 增大而提升（最高 1.5%）；(e) K=1 vs K=2 ablation（Table 3），方法在 K=1 时仍有效；(f) Accurate vs Viable 加权变体对比（Table 3）；(g) 与 K=3 baseline 对比（Table 5）：K=2 + Expert Group Approx. 达到与 K=3 相同的 perplexity 但不增加激活参数。

- 硬件平台是什么，配置是什么。
  - 单 GPU（用于 throughput 测量，reproducibility）；8 节点多节点训练集群（用于主实验）。具体 GPU 型号论文未明确说明。通过 NCCL 进行数据并行通信。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) Fine-grained MoE（DeepSeek 风格）：32 experts，hidden dim 1024，每个 expert 为 bottleneck MLP（intermediate size 704），2B total params，K=2 时 470M active params；(2) 标准 MoE：8 experts，hidden dim 1024，MLP intermediate size 2816，2B total params，780M active params。均使用 24 层 Transformer、16 attention heads（dim 64）、SwiGLU MLP、LayerNorm、RoPE。
  - 数据集：FineWeb（Penedo et al., 2024），200B tokens 训练，Llama3 tokenizer。
  - Benchmarks：mathqa, logiqa2, mmlu, openbookqa, logiqa, arc_challenge, arc_easy, hellaswag, copa, piqa。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：论文声明代码为开源（"Our code is currently open-source and will be linked here upon publication"），但为匿名投稿，链接未提供。使用开源框架 GPT-NeoX (Andonian et al., 2023) + Megablocks (Gale et al., 2022)。
  - 算法 Pipeline 的伪代码描述：

```
# === 前向传播（与标准 Top-K MoE 相同） ===
输入: token x, 路由器权重矩阵 W (shape: [N, d_token])
路由logits = W @ x                        # [N]
π = Softmax(路由logits)                     # [N], 专家权重
R(x) = TopK(π, K)                         # 选出的 K 个专家索引
y = Σ_{i∈R(x)} π_i * E_i(x)               # 标准 MoE 输出

# === 构造近似（Expert Group Approximation） ===
对于每个路由决策 R（共 C(N,K) 种可能）:
  对于每个未激活的专家 i ∉ R:
    近似 = 0
    对于每个激活专家 j ∈ R:
      相邻token集 = X_{i,j,·}  # 同时被路由到专家i和j的tokens
      近似 += mean({E_i(x') for x' ∈ X_{i,j,·}})  # 取平均
    近似 /= K  # 对 K 个组取平均
    ŷ_i(x) = 近似  # 对属于 X_R 的所有 x 使用同一近似

# === 构造稠密近似输出 ===
y' = Σ_{i∉R(x)} ŷ_i(x)                    # 所有未激活专家的近似输出之和

# === 注入近似梯度（前向不变，反向有梯度） ===
y = y + y' - stop_gradient(y')            # Eq.(6)
# stop_gradient 确保前向输出不变
# 反向时 ∂y/∂π = [E_1(x), ..., E_N(x)] 包含所有专家的梯度

# === 参数更新 ===
# 路由器 W: 接收来自所有 N 个专家的稠密梯度
# 专家 E_i: 除自身处理的 K/N 比例 token 外，还接收 (N-K)/N 比例的近似梯度
# 跨数据并行 workers 执行 all-reduce 聚合近似梯度
```

  - 张量计算关键步骤：路由器前向（matmul + softmax + TopK）不变；反向时用 stop-gradient 技巧将 N² 个专家组近似插入计算图，使路由器接收到稠密的 ∂y/∂π 梯度向量。近似在数据并行 workers 之间 all-reduce，增加样本量以估计稠密梯度。

## Towards MoE Deployment Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：论文提出三项 MoE 推理阶段的算法优化：(1) **Dynamic Gating（动态门控）**：提出基于 argsort 的可变大小 token 分发算法替代传统基于 static capacity + dispatch mask 的方案，核心计算为 argsort(O(S log S)) + bin-count(O(S)) + index(O(SD))，避免空 token placeholder 传输和 token dropping；(2) **Expert Buffering（专家缓冲）**：基于观察到的 MoE 推理中 expert 激活高度稀疏但具有强时序局部性（temporal locality），设计 GPU expert cache + CPU offload + LIFO 淘汰策略，利用两个关键观察——(a) expert 负载高度不平衡（存在高频 hot experts），(b) 同一 expert 在连续 batch 中持续活跃（时序局部性）；(3) **Load Balancing（负载均衡）**：将 expert-to-GPU 分配建模为 multi-way number partitioning 问题（NP-hard），提出 Greedy Balancing（基于独立激活的贪心分配）和 Anti-correlation Balancing（考虑 expert 间 Pearson 相关性，修改负载公式为 $\sum_{m} P_{mn} (\tilde{A}_{m} + 0.5 * S_{am})$）两种近似算法。
  - 实验比较：(a) Dynamic gating vs Static gating vs Tutel gating 在不同 batch size 和节点数下的吞吐量（LM: 6.21-11.23×、MT Encoder: 5.75-10.98×、MT Decoder: 2.58-5.71× 吞吐提升，Figure 9）；(b) 不同 gating 策略的内存消耗（动态内存+静态内存分解，Figure 10），dynamic gating 使 LM batchsize=8 的激活内存从 6.29GB 降至 1.28GB（79.6% 减少）；(c) Expert Buffering 下不同 cache 大小的 cache miss rate 与 Belady's MIN 对比（Figure 12）；(d) Load Balancing 对负载分布的影响（Max Load 和 Avg Max Load 指标，Figure 14）；(e) 三种优化组合的吞吐量和内存（Figures 9, 10）。

- 硬件平台是什么，配置是什么。
  - CPU: 2×Intel Xeon E5-2698 v4 @ 2.2GHz，700GB DDR4
  - CPU-GPU: PCIe 3.0 ×16，实测带宽饱和在约 12GB/s
  - GPU: 8×NVIDIA Tesla V100 (Volta)，32GB HBM2 @ 900GB/s，NVLink 300GB/s（单节点内）；多节点通过 InfiniBand 互联（带宽论文未明确说明）
  - 单节点(8 GPU)到四节点(32 GPU)的扩展实验

- 模型是什么。数据集和bench分别是什么。
  - 模型: (a) **Language Modeling MoE**：52B 参数，E=512 experts，24 layers，TD=1024，HD=4096，MF=2（每 2 层中 1 层为 MoE），C=0.05，top-2 gating，vocab=51200；Dense baseline: 355M 参数，相同层数和隐藏维度；(b) **Machine Translation MoE**：54.5B 参数，E=128 experts，48 layers（encoder+decoder），TD=2048，HD=8192，MF=4，C=1，top-2 gating，vocab=256206；Dense baseline: 3.3B 参数
  - 数据集: (a) LM: PILE [8] validation set，选取 Wikipedia、PubMed、Github 三个子域分析不同数据对 expert 激活模式的影响；(b) MT: NLLB-200 [22] validation set，English→French/Japanese/Asturian 三个目标语言
  - Benchmark 指标: Throughput (tokens/s)、Latency (ms per batch)、Memory Usage (peak GPU memory, static vs dynamic breakdown)、Cache Miss Rate、Max Load / Avg Max Load

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供独立开源仓库链接。论文明确说明实现基于 fairseq [23]（https://github.com/facebookresearch/fairseq），未提供论文修改版本的 fork/branch 链接。
  - 算法pipeline 伪代码级解释：
    **Dynamic Gating 核心算法**（在 MoE 层的 gating 函数之后执行）：
    ```
    输入: gate_scores = (B*S, E)  # gating 对每个 token 的打分
          capacity_factor = C     # 静态方案中的容量参数，动态方案不再需要
          n_experts = E
          top_k = 2               # top-2 gating

    # 1. 计算 routing decisions
    expert_weights, expert_indices = topk(gate_scores, top_k)  # (B*S, 2), (B*S, 2)
    # expert_indices[i] = [e1, e2], 第 i 个 token 被路由到专家 e1 和 e2

    # 2. 展平并排序（动态门控核心）
    flat_expert_indices = expert_indices.reshape(-1)  # (B*S*top_k,)
    sort_order = argsort(flat_expert_indices)          # O(n log n), n = B*S*top_k
    sorted_indices = flat_expert_indices[sort_order]

    # 3. 统计每个专家分配的 token 数量
    expert_counts = bincount(sorted_indices, minlength=E)  # O(n), shape (E,)

    # 4. 按 expert_id -> GPU_id 映射，聚合每 GPU 的接收量
    # expert_to_gpu[e] = device_id
    gpu_sizes = [sum(expert_counts[e] for e in experts_on_gpu[g]) for g in range(n_gpus)]

    # 5. 两阶段 all-to-all
    # Phase 1: 交换每个 GPU 的 token 接收量（极小消息）
    all_to_all_size(gpu_sizes)
    # Phase 2: 按 sort_order 重排 tokens 后分片发送（实际数据传输）
    reordered_tokens = tokens[sort_order]  # index-based, O(n*D)
    all_to_all_data(reordered_tokens, split_sizes=gpu_sizes)

    # 6. 各 GPU 按 expert 执行 FFN
    for expert_id in local_experts:
        if expert_counts[expert_id] > 0:
            expert_outputs = expert_ffn(expert_inputs[expert_id])
    ```
    与 Static Gating 对比（原方案）：
    ```
    # 原方案: 创建 dispatch mask, 大小 (E, S, S*C)
    dispatch_mask = zeros(E, S, S*C)           # 大量零值
    # 填 mask 过程中检查容量，超出的 token 被丢弃
    dispatched = einsum('ij,ijk->ik', tokens, dispatch_mask)  # 巨大稀疏矩阵乘
    # 即使用不到的空 capacity 也会传输 placeholder (零向量)
    ```
    **Expert Buffering 核心算法**：
    ```
    输入: active_experts = [e for e in local if expert_counts[e] > 0]
          gpu_cache = {expert_id: parameters}  # 大小可配置，如 10 experts/GPU
          cpu_memory = {expert_id: parameters} # 所有 experts 的完整参数

    for expert_id in sorted(active_experts):  # 按 expert_id 升序串行执行
        if expert_id in gpu_cache:
            params = gpu_cache[expert_id]      # Cache hit
        else:
            if gpu_cache.is_full():
                # 淘汰策略：优先淘汰非当前 batch 活跃的 expert
                inactive_in_cache = [e for e in gpu_cache if e not in active_experts]
                evict(inactive_in_cache[0])   # LIFO (最后加入的先淘汰)
            params = memcopy(cpu_memory[expert_id] -> gpu_cache)  # 与 all-to-all 并行
        output[expert_id] = expert_ffn(inputs[expert_id], params)
    ```
    **Load Balancing (Greedy Balancing) 核心算法**：
    ```
    输入: historical_activations A_mb, shape (E, B)  # expert m 在 batch b 的负载比例
    输出: expert_to_gpu[e] = device_id              # E/gpu 每设备

    avg_load = mean(A_mb, axis=1)                    # 每个 expert 的历史平均负载 (E,)
    sorted_experts = argsort_descending(avg_load)     # 按负载从高到低排序
    gpu_loads = zeros(n_gpus)                        # 各 GPU 累积负载
    gpu_capacity = E // n_gpus                       # 每 GPU 的 expert 数量上限

    assignment = [-1] * E
    for expert_id in sorted_experts:
        # 选当前负载最小的 GPU（已满的不可选）
        candidates = [g for g in range(n_gpus) if gpu_count[g] < gpu_capacity]
        best_gpu = argmin(gpu_loads[g] for g in candidates)
        assignment[expert_id] = best_gpu
        gpu_loads[best_gpu] += avg_load[expert_id]
        gpu_count[best_gpu] += 1
    ```

## Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：本论文并非提出新的算法模型，而是对 MLA（Multi-head Latent Attention）和 MoE 的算法特性进行系统级算术强度（ArI）分析。核心分析的算法优化是 **MLA 的 layer reordering（层重排）**：MLA 中 Q、K、V 经过低秩联合压缩（compressed latent space，$d_{\rm KVco}=512$），使用 decoupled RoPE 消除 Q 和 K 之间的非线性后，利用矩阵乘法结合律将 Score 层从 $\mathbf{S}_i = (\mathbf{C}_{\rm Q} \cdot \mathbf{W}_{\rm DQ_i}) \cdot (\mathbf{C}_{\rm KV} \cdot \mathbf{W}_{\rm DK_i})^T$ 重排为 $\mathbf{S}_i = (\mathbf{Q}_i \cdot \mathbf{W}_{\rm DK_i}^T) \cdot \mathbf{C}_{\rm KV}^T$，使得 decode 阶段 K 解压缩的代价降低 L 倍（只消与 $\mathbf{W}_{\rm DK}$ 相乘而非整个 $\mathbf{C}_{\rm KV}$ 解压缩），同时 Score 层读取的是压缩后的 $\mathbf{C}_{\rm KV}$ 而非完整解压缩的 KV$，将核心注意力层的 ArI 从 $\approx 1$ 提升到 $\approx 100$ Op/B（FlashMLA 优化后 $\approx 200$ Op/B），逼近现代加速器 ridge point。
  - 实验比较：(a) **MLA with/without reordering 的延迟与 ArI 对比**：prefill 和 decode 阶段的 K decompress 和 Score 层的 FLOPs、Memory Access、ArI 分析（Table III）；(b) **Attention block latency 对比**：reordering vs non-reordering，不同 batch size 和 sequence length 下各层执行时间占比（Figure 6）；(c) **核心注意力层 roofline 分析**：GPT-3 (MHA)、Llama4-Maverick (GQA)、DeepSeek-R1 (MLA) 各层在 H100 上的 ArI 与 ridge point 对比（Figure 3）；(d) **Layer reordering 对 prefill/decode 的不同影响**：decode 阶段延迟降低最多 103.12×，prefill 阶段延迟增加最多 2.21×；(e) **TP 对 reordered MLA 的影响**：因所有 head 共享 $\mathbf{C}_{\rm KV}$，TP 降低 ArI 为 $1/deg_{\rm TP}$，分析不同 $deg_{\rm TP}$ 和 batch size 下 attention block 的延迟（Figure 8）；(f) **MoE FC 层的 ArI 与 batch size 关系**：推导 $B_{\rm MoE} = RP_{\rm acc} \cdot n_e / n_k$ 公式。

- 硬件平台是什么，配置是什么。
  - 加速器：NVIDIA B200 GPU（BF16 2250 TFLOPS, 8000 GB/s 内存带宽, 192 GB HBM, ridge point 281.25 Op/B）作为主要评估平台。对比加速器包括 V100 SXM2 (125 TFLOPS, 900 GB/s, ridge point 138.89)、A100 SXM4 (312 TFLOPS, 2039 GB/s, ridge point 153.02)、H200 SXM5 (989.5 TFLOPS, 4800 GB/s, ridge point 206.15)、TPU v5P (459 TFLOPS, 2765 GB/s, ridge point 166)、TPU v7 (2307 TFLOPS, 7400 GB/s, ridge point 320.42)、MI325X (1307.4 TFLOPS, 6000 GB/s, ridge point 217.9)。
  - 真实硬件验证：DGX H100 系统。
  - 默认假设：32 B200 GPU 系统，NVLink 5th Gen 全互联（1.8 TB/s 双向带宽），遵循 NVL72 拓扑。

- 模型是什么。数据集和bench分别是什么。
  论文未使用任何模型或数据集。

- 开源情况。
  论文未使用任何模型或数据集。

## Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **Efficiency Leverage (EL)** 指标，定义 MoE 架构相对于等性能 dense 模型的计算效率比（EL = C_dense / C_moe）；通过分阶段实证研究（300+ 模型，最大 28B 参数）建立 MoE 架构的统一 scaling law。核心算法贡献：(1) 推导 MoE 最优超参 scaling law（η^opt = 1.1576·C^{-0.1529}, B^opt = 0.0694·C^{0.3644}）；(2) 推导 MoE vs dense 的最优模型-数据分配策略（MoE 偏向更小 M、更多 D）；(3) 对 activation ratio (A)、expert granularity (G)、shared expert ratio (S) 进行系统性消融，拟合联合 scaling law：EL(A,G,C) = Â^{α + γ(log G)² + β log G}，其中 α = a + d·log C。系数拟合值：a=1.23, d=-7.61e-2, γ=1.67e-2, β=-1.17e-1, A_start=1.63e-2, A_max=5.28e+16。
  - 实验比较：(a) **Activation ratio 消融**：固定 E^a=2, E^s=1，E 从 2 到 256 变化（A=0.8%~100%），在 C=1e18~6e20 FLOPs 范围观测 IsoFLOPs 曲线和 EL scaling；(b) **Expert granularity 消融**：G=2~16（E 从 64 到 512，proportionally 减小 d_expert），观测 U 形 loss-G 关系，最优 G≈12；(c) **Shared expert ratio 消融**：S=0%~83.3%，固定 E=256, E^a+E^s=12，观测最优 S 随 C 从 16.7% 降至 8.3%；(d) **Dense layer proportion 消融**：60层模型中前 0~3 层用 dense FFN 替代 MoE；(e) **Attention-FFN compute allocation 消融**：attn FLOPs ratio 20%~50%；(f) **Ling-mini-beta 验证**：17.5B 总参/0.85B 激活 vs Dense-6.1B，1T tokens 训练，验证 7× EL 预测。

- 硬件平台是什么，配置是什么。
  - GPU 训练。论文明确说明使用 "Ling series models" 训练框架（基于 Ling-Team et al. 2025），参考其 300B MoE 模型使用非高端 GPU 训练。具体 GPU 型号论文未明确说明，但基于 Ant Group Ling Team 的公开技术报告（Every FLOP Counts, 2025），可能使用 NVIDIA A100/H800 等 GPU。训练精度论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - **模型架构**：decoder-only Transformer with GQA (Grouped-Query Attention) + RoPE + BPE tokenizer (vocab=126,464)。MoE layer 使用 top-k routing（softmax gate + load balancing loss coeff=0.01 + router z-loss coeff=0.001）。消融实验模型规模从 8 layers/d_model=384 到 22 layers/d_model=1280，最大训练 FLOPs 6e20。Ling-mini-beta 验证模型：20 layers, d_model=2048, d_ffn=5120, d_expert=384, 16 heads/4 kv_heads, E=384, E^a=12, E^s=1, N=17.5B, N^a=0.85B。对比 dense 模型 Dense-6.1B：28 layers, d_model=4096, d_ffn=14336, 32 heads/8 kv_heads, N=6.11B。
  - **训练数据**：Ling Team 大规模多语言语料库（中英文为主），组成：Web 46%、Books 5%、Wiki 4%、Academic 6%、Code 25%、News 0.1%、Social 1.9%、Domain 1%、SFT 4%、Math 6%、Exam 1%。消融实验使用 2T token 子集，Ling-mini-beta 验证使用 1T token 子集。
  - **Benchmark**：(a) General Knowledge/Reasoning: ARC-challenge/easy, AGIEval, OpenBookQA, BBH, ProntoQA, Multi-LogiEval, HellaSwag, PIQA; (b) Language Understanding: RACE-middle/high; (c) Professional Knowledge: MMLU, MMLU-Pro, CMMLU, C-Eval, CommonsenseQA, GPQA; (d) Code: HumanEval, HumanEval-cn/Plus/FIM, MBPP, MBPP-Plus, LiveCodeBench, CruxEval; (e) Math: GSM8K, MATH, CMATH, MGSM-zh, CN-Middle School 24, Minerva-Math, MathBench, Gaokao2023-Math-En, GAOKAO-Math24。
  - **关键指标**：Training loss（cross-entropy），Efficiency Leverage (EL)，benchmark accuracy/F1。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **代码未开源**（Ant Group Ling Team 内部代码）。论文基于 Ling 系列 LLM 内部训练框架（Ling-Team et al., 2025 "Every FLOP Counts"）。

  **Efficiency Leverage (EL) 计算伪代码**：
  ```
  # === EL 定义 ===
  # 给定 MoE 架构 X_MoE 和 dense 架构 X_Dense
  # EL(X_MoE | X_Dense; C_target) = C_dense / C_moe
  # s.t. |L(C_moe; X_MoE) - L(C_dense; X_Dense)| ≤ ε

  # === Step 1: 拟合各架构的 loss scaling 曲线 ===
  def fit_loss_scaling(configs, flops_budgets):
      """
      对每个架构配置，在多个 FLOPs budget 下训练模型，拟合 L(C) = α · C^{-β}
      使用 compute-optimal allocation: M_opt, D_opt determined by scaling law
      """
      loss_curves = {}
      for arch in configs:  # 不同 A, G, S 的 MoE 配置 + dense baseline
          losses = []
          for C in flops_budgets:  # e.g., 3e18, 6e18, 2e19, 6e19, 2e20, 6e20
              M_opt = α_M · C^{β_M}   # optimal model scale (FLOPs/token)
              D_opt = α_D · C^{β_D}   # optimal data size (tokens)
              η = 1.1576 · C^{-0.1529}  # optimal learning rate
              B = 0.0694 · C^{0.3644}   # optimal batch size
              model = build_model(arch, M_opt)
              loss = train(model, D_opt, η, B)
              losses.append((C, loss))
          loss_curves[arch] = fit_power_law(losses)  # α · C^{-β}
      return loss_curves

  # === Step 2: 计算 EL ===
  def compute_el(dense_curve, moe_curve, C_moe):
      """
      EL = C_dense / C_moe
      其中 C_dense 满足 L_dense(C_dense) = L_moe(C_moe)
      """
      L_moe = moe_curve(C_moe)  # α_moe · C_moe^{-β_moe}
      # 解 L_dense(C_dense) = L_moe → α_dense · C_dense^{-β_dense} = L_moe
      C_dense = (L_moe / α_dense)^{-1/β_dense}
      return C_dense / C_moe

  # === Step 3: 拟合联合 EL scaling law ===
  # EL(A, G, C) = Â^{α + γ(log G)² + β log G}
  # 其中 Â 是 A 的饱和变换:
  #   1/Â = 1/(A + (1/A_start - 1/A_max)^{-1}) + 1/A_max
  # α = a + d · log C
  # 使用 Huber loss + BFGS 优化拟合参数 (a, d, γ, β, A_start, A_max)
  ```

  **MoE 前向传播张量计算（per token）**：
  ```
  # Input: h^t ∈ R^{d_model}, 第 t 个 token 的 hidden state
  # MoE layer with E experts (index i=1..E), E^a activated, E^s shared

  # Step 1: Router gating
  g^t = Softmax(W^g @ h^t)  # W^g ∈ R^{E × d_model}, g^t ∈ R^E
  selected = TopK(g^t, E^a)  # 选择 top-E^a 个 expert indices

  # Step 2: Expert computation (每个激活的 expert)
  # expert_i FFN: W_up_i ∈ R^{d_expert × d_model}, W_gate_i ∈ R^{d_expert × d_model}, W_down_i ∈ R^{d_model × d_expert}
  for i in selected:
      e_i = W_down_i @ (SwiGLU(W_gate_i @ h^t) ⊙ (W_up_i @ h^t))
  # SwiGLU: activation(W_gate @ h) ⊙ (W_up @ h), 其中 activation = SiLU

  # Step 3: Shared expert (if exists)
  if E^s > 0:
      e_shared = W_down^s @ (SwiGLU(W_gate^s @ h^t) ⊙ (W_up^s @ h^t))

  # Step 4: Weighted combination
  o^t = Σ_{i∈selected} g^t_i · e_i + e_shared  # g^t_i 是第 i 个 expert 的 gating score

  # FLOPs per token (forward):
  # M = L_attn · C_attn + L_dense · C_dense_ffn + L_moe · C_moe_ffn
  # C_moe_ffn ≈ 6 · d_model · (E^a · d_expert + d_shared)
  ```

  **Scaling Law 预测流程**：
  ```
  # 给定 A, G, C → 预测 EL
  def predict_el(activation_ratio, granularity, compute_budget):
      # 1. 计算饱和变换 Â
      inv_A_hat = 1/(activation_ratio + 1/(1/0.0163 - 1/5.28e16)) + 1/5.28e16
      A_hat = 1 / inv_A_hat
      # 2. 计算 compute-dependent exponent
      alpha = 1.23 - 0.0761 * log(compute_budget)
      # 3. 计算 granularity modulation
      log_G = log(granularity)
      gran_mod = 0.0167 * log_G^2 - 0.117 * log_G
      # 4. 联合 scaling law
      log_EL = (alpha + gran_mod) * log(A_hat)
      return exp(log_EL)
  ```

## The Omni-Expert: A Computationally Efficient Approach to Achieve a Mixture of Experts in a Single Expert Model

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Omni-Expert (OE)，一种在单个专家模型中通过子任务特定仿射特征变换（尺度+偏移）实现 MoE"分而治之"功能的高效替代方案。Baseline 包括：(1) Phoneme Independent (PI) 单模型，(2) Phoneme-based MoE（40 个独立专家网络 + 音素分类器作为门控网络）。实验比较：SRMR-CI 和 STOI 客观语音可懂度指标、信号损失频谱分布、计算复杂度（参数量/MACs/训练时间/模型大小）、特征变换消融（尺度vs偏移vs两者、输入层vs隐藏层vs两者位置）、噪声鲁棒性（DEMAND 多类噪声 + Cocktail Party 双人对话噪声，SNR -5~20 dB 在四种房间条件下）。

- 硬件平台是什么，配置是什么。
  NVIDIA Titan V GPU（12 GB HBM2，约 14 TFLOPS FP32）。所有模型训练在同一 GPU 上进行。CPU 等其他硬件论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  **模型架构**：两种 backbone：
  (i) 单层 123 单元单向 LSTM。
  (ii) 单层 117 单元单向 GRU + multi-head attention + layer norm + residual connection（GRU 输出经 layer norm → multi-head attention → residual + layer norm → 与 GRU 隐状态 element-wise 乘法融合）。
  OE 特有组件：(a) 音素分类器：40 单元 sigmoid 全连接输出层，作为门控网络产生加权概率 p_n；(b) 尺度 MLP：输入 40 维 one-hot 音素编码，输出 65 维尺度因子 a_n，ReLU 激活；(c) 偏移 MLP：同输入输出尺寸，输出 65 维偏移因子 b_n，LeakyReLU 激活。变换参数训练后预计算并存入查找表，推理时不使用 MLP。
  MoE Baseline：40 个独立专家网络（与 PI 模型结构相同），每个仅用对应音素组数据训练。
  **数据集**：
  训练：LibriSpeech 100h 语料库随机 8000 句（约 28 小时），RIR 来自 Brno University of Technology@FIT Reverberation Database。
  测试：HINT、CUNY-Male、CUNY-Female 数据集，RIR 来自 Aachen Impulse Response (AIR) database —— office（RT60=0.6s, DRR=0.4dB）、lecture（RT60=0.9s, DRR=-0.1dB）、stairway（RT60=0.9s, DRR=1.6dB）、church（RT60=6.5s, DRR=-0.6dB）四种房间。
  噪声鲁棒性测试：DEMAND 数据集（Domestic: kitchen/living room/washing machine; Public: cafeteria/restaurant/subway）+ Cocktail Party 双人对话噪声（TTB），SNR 级别 -5/0/5/10/15/20 dB。
  **Benchmark 指标**：(a) SRMR-CI —— 面向 CI 用户的 Speech-to-Reverberation Modulation Energy Ratio，使用 CI 滤波器组修改；(b) STOI —— Short-Time Objective Intelligibility；(c) T-F Signal Loss（公式5）；(d) 参数量/MACs（ptflops 包）/训练时间/模型大小（MB）。
  **特征提取流程**：ACE 策略（Nucleus CI 系统）→ 8ms 帧 + 2ms 重叠 → STFT (65 维频率特征) → log 压缩 → 全局均值/方差归一化。
  **音素标签**：Kaldi 强制对齐，39 个标准美式英语音素 + 1 个静音类 = 40 类。
  **CI vocoded speech**：Nucleus MATLAB Toolbox 生成 CI electrodograms → sine wave vocoder 重合成。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **代码未开源**（论文明确声明 proprietary）。

  **训练阶段伪代码**：
  ```
  # === 步骤1: 训练音素分类器（共享于 MoE 和 OE） ===
  for batch in DataLoader:
      h = LSTM_or_GRUplusA(features)        # features: (T, 65)
      p = sigmoid(FC_40(layer_norm(h)))      # p: (T, 40) 音素概率
      loss = CrossEntropy(p, phoneme_labels)  # phoneme_labels: (T, 40) one-hot
      # 优化器: SGD, lr=1e-5, momentum=0.9

  # === 步骤2: 预训练 Phoneme Independent (PI) 模型 ===
  pi_model = PhonemeIndependentModel()       # LSTM->FC->65 sigmoid output
  train(pi_model, all_data)                  # 初始化权重 U[-0.1,0.1]
  # 优化器: Adam, lr=1e-3, β=(0.9,0.999)

  # === 步骤3: 训练 OE 模型（用 PI 预训练权重初始化单专家网络） ===
  single_expert.load_state_dict(pi_model.state_dict())
  scale_mlp = MLP(40, 65, activation=ReLU)    # 输入one-hot → 尺度因子
  shift_mlp = MLP(40, 65, activation=LeakyReLU) # 输入one-hot → 偏移因子

  for batch in DataLoader:                     # batch_size=16, 按2s切分
      a_n = scale_mlp(phoneme_onehot)           # (40, 65) 所有音素的尺度因子
      b_n = shift_mlp(phoneme_onehot)           # (40, 65) 所有音素的偏移因子
      p = phoneme_classifier(features)           # (T, 40) 门控概率
      M_hat = 0
      for n in range(40):
          z_n = features * a_n[n] + b_n[n]      # 元素级仿射变换 (公式6)
          y_n = single_expert(z_n)               # single_expert(z): (T, 65)
          M_hat += p[:, n:n+1] * y_n             # 加权求和 (公式3)
      loss = mean((M_hat * X_mag - ideal_mask * X_mag)^2)  # 公式5
      optimizer.step()

  # 训练后将 a_n, b_n 预计算存入查找表:
  lookup_scale = {n: scale_mlp(one_hot(n)) for n in range(40)}
  lookup_shift = {n: shift_mlp(one_hot(n)) for n in range(40)}
  ```

  **推理阶段伪代码**：
  ```
  # 输入: features (T, 65) —— log-compressed频谱
  p = phoneme_classifier(features)               # (T, 40)
  M_hat = 0
  for n in range(40):
      a_n = lookup_scale[n]                      # (65,) 预计算
      b_n = lookup_shift[n]                      # (65,) 预计算
      z_n = features * a_n + b_n                  # 元素级仿射变换
      y_n = single_expert(z_n)                    # (T, 65)
      M_hat += p[:, n:n+1] * y_n
  enhanced_mag = M_hat * reverberant_mag          # 公式2
  # 注意: scale/shift MLP 不参与推理
  ```

  **关键张量计算流程**（以单帧为例）：
  1. x = log_compress(STFT(audio_frame)) → (1, 65)
  2. p = phoneme_classifier(x) → (1, 40)，所有 40 维
  3. 对 n=0..39: z_n = a_n ⊙ x + b_n → (1, 65)；y_n = expert(z_n) → (1, 65)
  4. M_hat = Σ_n p_n · y_n → (1, 65) 加权平均
  5. S_hat = M_hat ⊙ X → (1, 65) 增强频谱

  **计算复杂度对比**（LSTM backbone，来自原文 Table A4.1）：
  | 模型 | 参数量 | MACs | 大小 | 训练时间(Titan V) |
  |------|--------|------|------|-------------------|
  | PI | 108,225 | 109.44M | 0.43MB | 2h58m |
  | MoE | 40×108,225+PC | 4,377.6M+PC | 16.51MB+PC | 5h22m |
  | OE | 113,555+PC | 109.45M+PC | 0.45MB+PC | 1h57m |

  **GRU+A backbone**（来自原文 Table A4.2）：
  | 模型 | 参数量 | MACs | 大小 | 训练时间(Titan V) |
  |------|--------|------|------|-------------------|
  | PI | 127,946 | 127.76M | 0.51MB | 3h43m |
  | MoE | 40×127,946+PC | 5,110.58M+PC | 19.52MB+PC | 10h47m |
  | OE | 133,276+PC | 127.77M+PC | 0.53MB+PC | 1h21m |

  OE 以约 1/40 参数量和 MACs 达到或超越 MoE 性能。GRU+A 下 OEp SRMR-CI=2.014 vs MoEp=1.948；OEk STOI=0.850 vs MoEk=0.843。

## Toward Inference-optimal Mixture-of-Expert Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：本论文提出三部分算法贡献：
    1. **MoE Scaling Law（公式4）**：将 dense Transformer 的 scaling law 扩展至 MoE 架构，引入 expert 数量 E 作为第三个 scaling 维度。核心公式为 $\log L(N, D, E) \triangleq \log(A/N^{\alpha} + B/\hat{E}^{\beta} + C/D^{\gamma} + F) + d \log N \log \hat{E}$，其中 $\hat{E}$ 通过 $E_{start}$ 和 $E_{max}$ 两个参数建模 expert 增长的饱和效应。
    2. **Inference Cost Estimation（Section 4）**：提出 cost per token 指标 $C_{Model,G} = GC_0 / T_{Model}(G)$，在 vLLM 上 profiling 8×40GB A100 GPU（NVLink），建立模型大小与推理成本的线性关系，并推导 MoE 模型总参数量为 $N_{MoE} = (1 + (E-1) \cdot 1/3)N$（因为每两层 Transformer 中仅一层为 MoE，MLP 占总参数 2/3）。
    3. **Over-training 策略（Section 5.2）**：在固定训练预算 B 下，不使用 loss-optimal 配置，而是训练一个更小的模型（70-85% reduction），利用节省的预算训练更多 token，达到接近 loss-optimal 的质量但显著降低推理成本。
  - 实验比较：
    - (a) Scaling law 拟合：100M-730M 参数的 dense 模型，每个配置 4/8/16/32 experts，2.5B-20B tokens 训练，验证 loss predicted vs actual（Figure 1）。
    - (b) Inference cost profiling：不同模型大小和 expert 数量的推理成本曲线（Figure 2）。
    - (c) 三难权衡（Training-Inferece-Quality Trilemma）：训练预算、推理成本、模型质量三者 trade-off（Figure 3），loss-optimal 下 4/8 expert MoE 在相同推理成本下质量最优，但训练成本是 16/32 expert 的 2.5-4.3 倍。
    - (d) Over-training vs loss-optimal：固定训练预算下，over-trained 8/16-expert MoE 在相同质量下推理成本仅为 loss-optimal 4-expert 的 47%-52%；相同推理成本下可节省 68.4% 训练预算（Figure 4, 5, 6）。
    - (e) 对比 base model（dense/4-expert）：8-expert over-trained MoE 推理成本为 dense 的 31.6%-38.1%，仅需 dense 23.3%-28.2% 的 activated parameters（Figure 5, 6）。

- 硬件平台是什么，配置是什么。
  - 训练：A100 GPU（最多 32 卡），使用 data parallelism + tensor parallelism + model parallelism（Megatron-DeepSpeed）。
  - 推理 profiling：8×40GB A100 GPU，NVLink 互联，使用 vLLM serving 系统。
  - 推理 cost 计算：以单 GPU 每秒运行成本 $C_0$ 为常数单位。

- 模型是什么。数据集和bench分别是什么。
  - **模型架构**：Llama-style（gated-MLP，MLP intermediate hidden 为 2.67× hidden dimension），每两层 Transformer 中替换一层为 MoE（Top-2 gating）。具体配置（Table 1）：
    | Name | d_model | n_layers | n_heads | Actual Params (w/o embedding) |
    |------|---------|----------|---------|-------------------------------|
    | 100M | 768 | 12 | 8 | 81,395,712 |
    | 200M | 896 | 14 | 8 | 184,064,768 |
    | 320M | 1024 | 16 | 12 | 289,406,976 |
    | 730M | 1536 | 16 | 16 | 679,477,248 |
    每个 dense 模型分别训练 4/8/16/32 experts 版本，以及一个 dense baseline。
  - **数据集**：SlimPajama（open-source LLaMA pretraining data blend：82% 互联网内容、4.5% code、4.5% Wikipedia、4.5% books、2.5% Arxiv、2% StackExchange）。训练使用最多 20B tokens，validation 使用 0.58B tokens。
  - **Benchmark 指标**：Validation loss（perplexity）、inference cost per token（dollars/token）、training FLOPs。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **论文未明确说明代码/权重开源链接**。训练框架基于 Megatron-DeepSpeed（fork），推理 profiling 使用 vLLM（Kwon et al., 2023）。

  **Scaling Law 算法 pipeline（公式4 拆解）**：

  ```
  # === Step 1: 定义 effective expert count E_hat ===
  # 建模 expert 增长的饱和效应（E_start=专家线性增长起点, E_max=饱和上限）
  inv_E_start_max = 1/E_start - 1/E_max
  inv_E_hat = 1/(E - 1 + (inv_E_start_max)^(-1)) + 1/E_max
  # 当 E << E_start: E_hat ≈ E (linear growth)
  # 当 E >> E_max: E_hat ≈ E_max (saturated)

  # === Step 2: Scaling Law 预测 loss ===
  # log L = log(A/N^α + B/E_hat^β + C/D^γ + F) + d * log(N) * log(E_hat)
  N = model_dense_params    # e.g., 100M → 730M
  D = training_tokens       # e.g., 2.5B → 20B
  E = num_experts           # 4, 8, 16, 32

  term_1 = A / (N^alpha)    # 模型容量项
  term_2 = B / (E_hat^beta) # Expert 数量项 (饱和)
  term_3 = C / (D^gamma)    # 数据量项
  base_loss = term_1 + term_2 + term_3 + F
  interaction = d * log(N) * log(E_hat)  # N-E 交互项

  log_L = log(base_loss) + interaction
  # 最终 loss = exp(log_L)

  # === Step 3: 拟合参数 ===
  # 优化目标：min Σ Huberδ(log L_pred - log L_actual)
  # 使用 L-BFGS, δ=1e-3
  # 参数 α,β,γ ∈ [0,2]; A,B,C,D,F from grid search
  # 评估: RMSLE=3.908e-3, Huber=1.033e-3
  ```

  **Over-training 算法（Algorithm 1 & 2 张量计算路径）**：

  ```
  # === Algorithm 1: Optimal Inference Cost for Bounded Loss ===
  # 给定训练预算 B，base model (E experts)，target model (E' > E experts)
  # 求: 在保证 L_E' ≤ L_E_opt 的前提下，最小推理成本 I_E'_min

  (N_E, D_E) = loss_optimal_config(B, E)    # 求解损失最优配置
  L_E_opt = scaling_law(N_E, D_E, E)        # 基准 loss
  I_E = min_g Get_cost(N_E, E, g)          # 基准推理成本

  # Dichotomy search: 找满足 L_E'(N, B) = L_E_opt 的最小 N
  N_E' = dichotomy_search(E', L_E_opt, B)
  # 因为 L 在 loss-optimal 前单调递减，取刚好等于 L_E_opt 的 N 即可得到最小推理成本
  I_E'_min = min_g Get_cost(N_E', E', g)
  # N_E' 通常仅为 N_E 的 15%-30%

  # === Algorithm 2: Optimal Loss for Bounded Inference Cost ===
  # 给定训练预算 B，base model (E experts)，target model (E' > E experts)
  # 求: 在保证 I_E' ≤ I_E 的前提下，最低 validation loss L_E'_min

  (N_E, D_E) = loss_optimal_config(B, E)
  I_E = min_g Get_cost(N_E, E, g)          # 推理成本上限

  # Dichotomy search: 找满足 I_E'(N) = I_E 的 N
  N_E' = dichotomy_search_by_cost(E', I_E)
  D_E' = B / (6 * N_E')                    # 剩余预算全给数据
  L_E'_min = scaling_law(N_E', D_E', E')
  # N_E' 通常为 N_E 的 30%-85%
  ```

  **推理成本建模关键张量关系**：
  ```
  # 单 token KV-cache = 2hl (h=hidden_dim, l=n_layers)
  # 总 KV-cache 内存 = G*M_0 - N_m  (G GPU, M_0 per-GPU memory)
  # 最大并发请求数: b = (G*M_0 - N_m) / ((2p+n)hl)
  #   p = 平均 prompt length, n = 平均 output length
  # 
  # throughput: T_m = (G*M_0 - N_m) / (khl(L^P + L^D))
  #   k = 2p + n (常数, 由 traffic pattern 决定)
  #   L^P = prompt latency, L^D = decode latency
  #
  # cost per token: C = G*C_0 / T_m(G)
  # 最优 GPU 数: argmin_G C = G*C_0 / T_m(G)
  ```

  **关键发现汇总（定量）**：
  | 对比场景 | Base Model | Over-trained | 结果 |
  |---------|-----------|-------------|------|
  | Loss-optimal dense baseline | Dense Transformer | Over-trained 8-expert MoE | 推理成本 31.6%-38.1% of dense |
  | Loss-optimal 4-expert baseline | 4-expert MoE | Over-trained 8-expert MoE | 推理成本 47%-52% of 4-expert |
  | Loss-optimal 4-expert baseline | 4-expert MoE | Over-trained 16-expert MoE | 推理成本 48%-53% of 4-expert |
  | 相同推理成本 | Loss-optimal 4-expert | Over-trained 16-expert MoE | 训练 FLOPs 节省 68.4% |
  | Loss-optimal 4-expert baseline | 4-expert MoE | Loss-optimal 16-expert MoE | 训练 FLOPs 仅需 23.7%-42.8% |

## Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：**Asynchronous Expert Parallelism (AEP)** 算法——一种打破 MoE EP serving 中 barrier 同步的新执行范式。核心算法组件：
    - **µ-queuing（层粒度 token 队列）**：将每个 decoding block 的每个 expert 层视为独立调度单元。Token 到达后按 LayerID 分离入队，GPU 空闲时从任意 ready layer 拉取 token 自适应 re-batch 执行。Cold expert tokens 被允许在队列中积累，直到 batch size 足够大（实验中 batch≈128 时 GEMM 达到 near-linear throughput scaling）才被调度执行，避免 HBM-bound 小 batch 效率损失。
    - **Defragging Scheduler（Algorithm 1，伪代码见下文）**：对每个 (block b, expert e) 计算 Score[b][e] = LScore + Q[b][e]，其中 lookahead score LScore = sum_{k=1}^{K} (sum_{e'} Q[(b+k) mod N_B][e'] / N_e) × δ^k，衰减因子 δ ∈ (0,1)，K 为 lookahead 窗口。该算法同时鼓励 defragmentation（通过 lookahead 使 token wave 保持连续）和 queue occupancy 感知（通过 Q[b][e] 项避免过度忽略孤立的 token 碎片）。
    - **异步 Token Merge（Top-K > 1 支持）**：当 K > 1 时，每个 token 被复制 K 份发送到 K 个 expert。Receptor 维护一个 token pool，通过 <RequestID, LayerID> 元组在所有 K 路 expert 输出到达时 merge 为完整 token，然后才移入下一 attention layer 的 µ-queue。
    - **Token 依赖追踪**：每个 token 携带 metadata <RequestID, LayerID, Tensors[] 引用, prefill_length, topk_weights>，使异步乱序执行中仍能正确追踪请求归属和下一层路由目标。
    - **Hot/Cold Expert 自适应调度**：通过将一个 expert 在所有 block 中的层 colocate 到同一 GPU，scheduler 利用 layer 间 precede/ordering 关系将多数 tokens 收敛到 1-2 个连续 block 的 frontier，hot expert 的 tokens 快速积累优先调度，cold expert tokens 延迟积累到高效 batch size。
  - 实验比较：
    - (a) vs SGLang EP Top-1 routing: throughput 2.0-2.7×（取决于 workload 长度），ITL 低负载下略高但高负载下相当或更优。
    - (b) vs SGLang EP Top-2 routing: throughput 提升幅度减小（Top-2 降低 skew + token merge 同步效应）。
    - (c) Multi-node scalability: 16 experts/16 GPUs, throughput 3× vs SGLang, AMoE 从 8 GPU 扩展到 16 GPU 实现 1.92× throughput 提升（SGLang 无提升）。
    - (d) Scheduler ablation: defragging vs MTFS vs FLFS, 验证 defragging 算法在 batch fragmentation 和 forward progress 之间的 Pareto 优势。
- 硬件平台是什么，配置是什么。
  - Lambda：8× A100-SXM4-80GB，NVSwitch 600 GB/s per GPU，CUDA 12.8，NCCL 2.25.1。
  - AWS P4 (multi-node)：2× 8× A100-SXM4-40GB，NVSwitch 600 GB/s，4× 100 Gbps EFA，CUDA 12.4，NCCL 2.22.3。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral 8x7B 修改版：(a) GQA → MQA 减少 KV cache 竞争；(b) expert routing 替换为基于 Dolly 数据集 profiling 的指数分布随机路由（模拟真实 expert load skew）；扩展实验用 16 experts 版 Mixtral（模拟 LLaMA-V4 等更大 MoE）。
  - 数据集/Workload：Databricks-Dolly-15k 用于 profiling expert load distribution。三类 synthetic Poisson arrival decoding workload：Short (input [30,70], output [70,130]), Medium (input [50,150], output [50,250]), Reasonable (input [100,300], output [100,500])。
- 开源情况。论文声明将开源 AMoE，**但论文正文和 arXiv 页面均未给出具体 GitHub URL，当前无法确认开源仓库地址。**

  AEP 算法 pipeline 详解（基于论文 Algorithm 1 + §3.2 描述）：

  ```
  ═══════════════════════════════════════════════════════════
  Algorithm: Asynchronous Expert Parallelism (AEP) 核心
  ═══════════════════════════════════════════════════════════

  // 常量定义
  N_B:  decoding blocks 数量 (e.g., 32 for Mixtral 8x7B)
  N_E:  experts per GPU (depends on placement)
  K:    lookahead window (e.g., K=3)
  δ:    衰减因子 (e.g., δ=0.5)

  // 全局状态 (每个 GPU runtime)
  mu_queue[N_B][N_E]:  每层 token 队列
  token_pool:          等待 merge 的 Top-K 不完全 token

  // ── Step 1: Receptor ──
  // 接收 incoming token batch, 分离入 µ-queue
  function receptor(incoming_batches[]):
    for batch in incoming_batches:
      for token in batch:
        // Top-K merge: 检查是否需要多路输入
        if token.needs_k_merge:   // K > 1 时 expert 输出需 merge
          key = (token.RequestID, token.LayerID)
          merged = token_pool.add_and_check(key, token)
          if merged is not None:
            mu_queue[merged.LayerID.block][merged.LayerID.expert].enqueue(merged)
        else:
          mu_queue[token.LayerID.block][token.LayerID.expert].enqueue(token)

  // ── Step 2: Scheduler (Algorithm 1) ──
  // GPU idle → 选择最优 (block, expert) pair 执行
  function scheduler():
    Scores[N_B][N_E] = 0
    for b = 0 to N_B - 1:
      // 计算 lookahead score
      LScore = 0
      for k = 1 to K:
        b_next = (b + k) mod N_B
        // 前方 block 所有 expert 的总 token 数
        TotalTokens = sum_{e=0}^{N_E-1} mu_queue[b_next][e].size()
        LScore += (TotalTokens / N_E) * (δ ** k)
      
      for e = 0 to N_E - 1:
        if mu_queue[b][e] is not empty:
          Scores[b][e] = LScore + mu_queue[b][e].size()
    
    // 选最高分 layer
    (b_opt, e_opt) = argmax_{b,e} Scores[b][e]
    return (b_opt, e_opt)

  // ── Step 3: Executor ──
  function executor(b, e):
    batch = mu_queue[b][e].drain_all()
    
    // 自定义 CUDA kernel: fuse 多个独立到达的小 batch 为连续 batch
    contiguous_batch = fuse_fragmented_batches(batch)
    
    if is_attention_layer(b, e):
      // 分配 KV cache slot, 查找已有 KV pages
      page_table.allocate_and_lookup(contiguous_batch)
      // fused CPU→GPU transfer: prefill_length, KV page indices
      cuda_stream.transfer_metadata(contiguous_batch)
      // paged attention kernel (from vLLM)
      output = paged_attention_kernel(contiguous_batch)
      // GPU→CPU: expert routing 结果 (indices + weights)
      routing_info = cuda_stream.transfer_back(output)
    else:  // expert layer
      // GEMM fusion: W_expert × input (no metadata dependency)
      output = expert_gemm_kernel(contiguous_batch)
      routing_info = None  // expert 执行不需要回传 routing info
    
    return output, routing_info

  // ── Step 4: Dispatcher ──
  function dispatcher(output_tokens, routing_info):
    for token in output_tokens:
      if it was an attention output:  // 下一层是 expert
        token.assigned_expert = route(token.embedding)  // MoE gating
        token.LayerID = (current_block, token.assigned_expert)
      else:  // 下一层是 attention
        token.LayerID = (current_block + 1, token.attn_dp_rank)
    
    // Permute: 按 expert ID（去 expert）或 DP rank（去 attention）分组
    permuted = sort_by_next_target(output_tokens)
    batches = split_into_groups(permuted)  // 按目标 GPU 分组
    
    communicator.send_async(batches)  // Phase 1: ZeroMQ → Phase 2: NCCL P2P
  ```

  张量计算视角（以单个 token 通过 MoE block 为例）：

  ```
  Token embedding x ∈ R^d (d=4096 for Mixtral 8x7B)
  
  Block b 的 attention 层 (GPU A, attention DP rank):
    q, k, v = W_Q·x, W_K·x, W_V·x         // 投影
    attn_out = MQA(q, k, v, KV_cache)      // Multi-Query Attention
    gate_logits = W_gate · attn_out         // MoE gating
    expert_idx, weight = top_k(gate_logits)  // 选 K 个 expert
  
  Expert 层 (GPU E, 持有 expert e 的所有 block 的层):
    // token 被路由到此 GPU 的 expert e
    expert_out = FFN_e(attn_out)            // W_up·σ(W_gate·x) ⊙ W_down
    // 如果 K>1: 需等待 K 路 expert 输出都在 attention GPU 上 merge
    merged = sum_{k=1}^{K} weight_k · expert_out_k
  
  // 循环到 block b+1 的 attention 层...
  ```

## Toward Efficient Inference for Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出三个 MoE 推理优化技术：(1) **Dynamic Gating**——将静态 gating 的 batch matmul dispatch 替换为 argsort + bin-count + indexing 的动态路由，eliminates dispatch mask 和 placeholder computation；(2) **Expert Buffering**——GPU/CPU 两级缓存机制，仅将热 expert 留在 GPU 显存；(3) **Load Balancing**——Greedy 和 Anti-Correlation 两种算法优化 expert 到 GPU 的分配。
  实验比较：baseline 为 Fairseq static gating，对比方法包括 Tutel（custom kernel）、FasterMoE（communication-computation overlap）、Megablock（block-sparse kernel）。评估指标为 token throughput、memory usage、cache miss rate、load balance（Max load / Avg-Max load）。

- 硬件平台是什么，配置是什么。
  - *Apple* 集群：8×NVIDIA Tesla V100 (32GB HBM2) via NVLink，2×Intel Xeon E5-2698 v4，700GB CPU memory，16GB/s PCIe 3.0 CPU-GPU 带宽，支持单节点和多节点（2/4 nodes）。
  - *Pear* 集群：4×NVIDIA RTX A5000 (24GB)，2×Intel Xeon Gold 5317，64GB CPU memory，32GB/s PCIe 4.0，仅单节点。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - LM-Small: 125M dense / 15B MoE（E=512 experts, top-2 gating, C=0.05, 12 layers, d_model=768, d_ff=3072）
  - LM: 355M dense / 52B MoE（E=512 experts, top-2 gating, C=0.05, 24 layers, d_model=1024, d_ff=4096）
  - MT: 3.3B dense / 54.5B MoE（E=128 experts, top-4 gating, C=1.0, 48 layers, d_model=2048, d_ff=8192）
  数据集：LM 使用 PILE 数据集（Wikipedia, PubMed, Github 三个 domain）；MT 使用 NLLB-200 验证集（English→French, Japanese, Austrian）。
  开源情况。代码开源：https://github.com/hyhuang00/moe_inference，基于 Fairseq [14] 实现。

  算法pipeline 详解（伪代码）：

  ```
  # === Static Gating (Baseline, Fairseq) ===
  # O(S^2 * E * D * C) via batch matmul
  def static_gating(tokens, S, E, C, D):
      # Step 1: gating decisions → expert assignments (size S)
      gate_logits = gate_linear(tokens)           # (S, E)
      assignments = top_k(gate_logits, k)          # (S, k)
      
      # Step 2: create dispatch mask (E, S, S×C) — HIGHLY SPARSE
      mask = zeros(E, S, S*C)
      for i in range(S):
          for e in assignments[i]:
              if expert_capacity_remaining[e] > 0:
                  row = find_first_empty_row(mask[e])
                  mask[e, row, i] = 1
      
      # Step 3: Batch matmul to reorder tokens — O(S^2 E D C)
      dispatched = bmm(mask, tokens)              # (E, S×C, D)
      return dispatched
      
      # 问题: 大量 placeholder (zeros) 填充, mask 内存大

  # === Dynamic Gating (Proposed) ===
  # O(S*D + S*logS) via argsort + indexing
  def dynamic_gating(tokens, S, E, D):
      # Step 1: gating decisions
      gate_logits = gate_linear(tokens)           # (S, E)
      assignments = top_k(gate_logits, k)          # (S, k) → flatten
      
      # Step 2: argsort to group by expert — O(S log S)
      sorted_idx = argsort(assignments[:, 1])     # sort by expert ID
      sorted_tokens = tokens[sorted_idx]           # O(SD) indexing
      
      # Step 3: bin-count for sizes — O(S)
      expert_sizes = bincount(assignments[:, 1], minlength=E)
      
      # Step 4: all-to-all notify sizes (first round, 20µs avg)
      comm.all_to_all(expert_sizes)  # each GPU learns incoming sizes
      
      # Step 5: split + all-to-all transfer tokens (second round)
      token_groups = split(sorted_tokens, expert_sizes)  # by expert
      received = comm.all_to_all(token_groups)
      
      return received  # variable-length list per expert
      
      # 收益: 无 placeholder, 无 mask allocation, 根据实际负载动态容量
  ```

  张量计算对比：
  ```
  Static Gating:
    Tokens X ∈ R^{S×D}  →  Dispatch Mask M ∈ R^{E×S×S×C}
    → Dispatched = M × X  (batch matmul, O(S²EDC))
    → Experts see EXACTLY S×C tokens each, filled with zeros
  
  Dynamic Gating:
    Tokens X ∈ R^{S×D} → Gate → argsort → sorted_X[permutation]
    → bincount → sizes[e] = count of tokens for expert e
    → split sorted_X by sizes → variable-length dispatch
    → Experts see EXACTLY sizes[e] tokens, NO zeros
  ```

  关键：标准 EP 中，block b 的所有 token 必须先完成 attention（barrier），再 all-to-all 分发到 expert GPU，所有 expert GPU 完成计算后再 all-to-all 收集（barrier），才能进入 block b+1。AEP 将这两层 barrier 消除——每个 GPU 在任意时刻执行任意 block 的任意层，cold expert tokens 被自然延迟积累，GPU 永不等 barrier。

## SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **SEUF (Selected Experts Unlearning Framework)**——一种针对 MoE LLM 的参数高效 unlearning 框架。核心包含三步：(1) **Expert Attribution**：对 forget set 中每个 token，记录 Router 输出的 gating score g_{i,t}，按式 s_i = (1/Z) * Σ_j (1/L_j) * Σ_t g_{i,t} 计算每个 expert 的 affinity score，跨所有 layer 排序选出 top-M affinity 最高的 expert 作为"target expert"（默认 M=1）；(2) **Router Anchor Loss**：式 L_anchor = ||g - a||_2^2，其中 a_i = 1 当第 i 个 expert 为 target expert 否则 a_i = 0，强制 router 在 unlearning 过程中持续激活该 target expert，防止"expert selection shift"（router 切换激活非目标 expert 来作弊降低 forget loss）；(3) **Focused Unlearning**：仅对选中的 target expert 及其对应 router 进行梯度更新（冻结其他所有参数），更新参数仅约 0.06%。总损失为 min_θ l_f(θ; D_f) + λ l_r(θ; D_r) + α L_anchor。SEUF 为即插即用框架，可集成 GA、GDIFF、NPO、RMU 等任意现有 unlearning 方法。
  - 实验比较：(a) 四种 unlearning 方法（GA、GDIFF、NPO、RMU）在 w/ vs w/o SEUF 下的 FE（Forget Efficacy，越低越好）和 UT（Model Utility on MMLU，越高越好）对比（Table 3），涵盖 Qwen1.5-MoE-A2.7B-Chat 和 DeepSeek-V2-Lite 两个模型，WMDP 和 RWKU 两个 benchmark；(b) SEUF vs PEFT baseline（LoRA、ESFT）的参数效率和 unlearning 效果对比（Table 3, Table 4）；(c) Top-1 expert selection（affinity score-based）vs Random selection 的消融实验（Table 3 最后一行）；(d) 不同 affinity score 排名的 expert 对 utility 的影响（Table 5，排名 #1 → UT 0.5485，排名 #26 → UT 0.2355）；(e) Top-M 选择消融（M=1/3/6，same layer vs different layers, Table 2+Table 7）；(f) 对抗攻击鲁棒性：GCG jailbreak 攻击下 FE 保持 0.01 不变（Sec. 5），证明 unlearned knowledge 不可恢复；(g) Mixtral-8x7B 大模型扩展实验（Table 9）；(h) hyperparameter α 敏感性分析（Table 6，α=1 最优）。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 GPU（论文在 Sec. 5 中明确提到 "∼ 1 GPU hour on an A100 per soft prompt" 用于 GCG 攻击实验）。其他训练/推理所用 GPU 具体型号和数量论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) **Qwen1.5-MoE-A2.7B-Chat**（Qwen）：14.3B 总参数，2.7B 激活参数，upcycle-from-dense 训练方案；(2) **DeepSeek-V2-Lite**（DeepSeek）：16B 总参数，2.4B 激活参数，train-from-scratch 训练方案，包含 shared experts；(3) **mistralai/Mixtral-8x7B-Instruct-v0.1**（Mixtral）：45B 总参数，12.9B 激活参数，仅用于 SEUF vs PEFT 比较。
  - 数据集：(1) **WMDP** benchmark（Li et al., 2024）：评估移除 biosecurity、cybersecurity、chemical security 领域的 hazardous knowledge，使用 WMDP-Cyber 子集作为 forget set，MMLU 作为 utility evaluation；(2) **RWKU** benchmark（Jin et al., 2024）：评估消除 200 个真实世界名人信息，选取 100 人作为 unlearning target，使用 train_original_passage 中 Wikipedia 描述作为 forget set。
  - Benchmarks/metrics：**FE (Forget Efficacy)**——WMDP 上为 forget set 的四选一多选题准确率（理想值 0.25 即随机猜测），RWKU 上为 fill-in-the-blank 和 QA 任务的 Rouge-L recall（理想值 0.0）；**UT (Model Utility)**——MMLU zero-shot 准确率（越高越好）。使用 LM Evaluation Harness 进行评测。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供代码仓库链接。评估使用开源 LM Evaluation Harness（Gao et al., 2024）。
  - 算法 Pipeline 核心伪代码：

```
# === SEUF Unlearning (Algorithm 1) ===
# Input: pretrained_model θ_o, forget_set D_f, retain_set D_r
# Output: unlearned_model θ_u

# Step 1-3: Expert Attribution
D_s = sample_subset(D_f, n_tokens=100000)     # ~100K tokens for robust attribution
affinity_scores = {}                            # dict: expert_id → score
for each sample x_j in D_s:
    for each layer l in model:
        for each token t at position:
            g_{i,t}^{(l)} = Softmax(Router(u_t^{(l)}))[i]  # router gating score
            # accumulate: s_i = (1/Z) * Σ_j (1/L_j) * Σ_t g_{i,t}^{(l)}
            affinity_scores[(l,i)] += g_{i,t}^{(l)} / len(x_j)

# Rank experts globally by affinity score, select top-M (M=1)
all_experts = sorted(affinity_scores.items(), key=lambda x: x[1], reverse=True)
e_M = all_experts[:M]                           # selected target expert(s)

# Step 4: Activate only target expert & its router
for param_name, param in model.named_parameters():
    param.requires_grad = False                  # freeze all
for expert in e_M:
    expert.expert_weights.requires_grad = True   # unfreeze target expert
    expert.router.requires_grad = True           # unfreeze corresponding router

# Step 5: Unlearn with anchor loss
for batch in D_f ∪ D_r:
    # Forward pass
    for layer l:
        g^{(l)} = Router(u_t^{(l)})              # router output probabilities

        # Anchor loss: force router to keep target expert active
        a^{(l)} = [1 if i in e_M else 0 for i in range(num_experts)]
        L_anchor += ||g^{(l)} - a^{(l)}||_2^2

        # Expert computation (standard MoE)
        s^{(l)} = Softmax(g^{(l)})
        topk_indices = TopK(s^{(l)}, K)
        h' = u + Σ_{i∈topk} s_i * FFN_i(u)

    # Compute unlearning loss
    L = L_forget(D_f) + λ * L_retain(D_r) + α * L_anchor

    # Gradient update (only target expert + router params updated)
    θ ← θ - η * ∇_θ L
```

  - 核心张量计算——Router Anchor Loss：
    - Router 对第 l 层输出 gating score: g^{(l)} ∈ R^{E} （E 个 expert 的概率分布，经由 Softmax）
    - Anchor target: a^{(l)} ∈ {0,1}^{E}，target expert 对应位置为 1，其余为 0
    - Anchor Loss: L_anchor^{(l)} = Σ_i (g_i^{(l)} - a_i^{(l)})^2，对所有 MoE layer 求和
    - 效果：MSE loss 强制 router 在 unlearning 时保持 target expert 的 gating score 接近 1，防止 router 将 token 路由到非目标 expert 来"作弊"降低 forget loss
  - Expert Attribution 核心张量计算：
    - 对 calibration set Z 个样本，第 j 个样本序列长度 L_j
    - 第 l 层第 i 个 expert 的 affinity: s_i^{(l)} = (1/Z) Σ_{j=1}^{Z} (1/L_j) Σ_{t=1}^{L_j} g_{i,t}^{(l)}
    - g_{i,t}^{(l)} 是 Router Softmax 后第 i 维的标量值
    - 跨所有 layer 全局排序，选 top-1 expert（M=1 时性能最优，Insight 4）

## Upcycling Large Language Models into Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：将预训练稠密 LLM 转换为稀疏 MoE 模型的 **upcycling** 训练算法，包括：(1) **Virtual Group Initialization（虚拟组初始化）**——针对 fine-grained MoE（granular upcycling），将稠密 MLP 的 FFN 权重按 hidden dimension 切分为 G 个 shard，复制 E 次形成 N=E×G 个 expert，路由器权重也分组复制，保证初始状态下 Router TopK 恰好能选到每个 shard 的一份拷贝，使 MoE 输出与稠密模型功能等价；(2) **Weight Scaling（权重缩放）**——对 expert MLP 的 W1 和 W2 权重同时缩放，缩放因子 = ³√(E×G²/T)，对 squared-relu 推导，对 SwiGLU 也有效，同时适用于 coarse-grained 和 fine-grained MoE；(3) **Softmax-then-TopK Routing**——采用先 softmax 后 topK 的 Router 顺序（而非 Mixtral 的 topK-then-softmax），保留 Router 输出的绝对值信息；(4) **学习率重置策略**——upcycling 时从预训练最低学习率重新提高到峰值学习率（如 2e-4），配合 cosine decay，帮助模型逃离稠密模型的局部最小值、促进 expert 分化；(5) **大批量训练**——batch size 增大至 4M tokens 以降低 expert 梯度噪声和负载均衡损失噪声。
  - 实验比较：(a) Upcycling vs 续训稠密模型——Nemotron 2B 在 0.1T tokens 下，upcycling loss 比续训低 1.1%，且续训迅速 plateau（Figure 4a）；(b) Upcycling vs 从头训练——在固定 compute budget 下 upcycling 显著优于 from scratch（Figure 4b）；(c) 学习率调度消融——constant LR (2e-5) vs 重置 LR (warmup to 2e-4) vs 重置 LR (warmup to 1e-4)，验证重置 LR 优于 constant LR（Figure 5），且权重 cosine similarity 从 ≈1 降至 0.6-0.7（Figure 6）；(d) Batch size 消融——512 (2M tokens) vs 1024 (4M tokens) vs 8192 (32M tokens)，4M tokens batch 最优（Figure 7）；(e) Softmax-TopK 顺序——softmax-then-topK 优于 topK-then-softmax（Section 3.4）；(f) Weight Scaling 消融——w/ vs w/o weight scaling，weight scaling 带来 1.5% loss 改善（Nemotron-4 15B E8G1T1, Figure 9），多种替代方法（MoE output scaling、post expert layernorm）均不如 weight scaling；(g) Granularity 消融——8/64/128/256 experts iso-FLOP 对比，64 experts 最优，128/256 experts 有 diminishing returns（Figure 10）；(h) TopK 消融——Top-1 vs Top-2 (E8G1)，Top-2 优于 Top-1 但计算量加倍（Figure 11）；(i) Shared experts（Deepseek-MoE 风格）——8 shared + 64 experts top-8 vs iso-FLOP 64 experts top-16，性能持平（Figure 13）；(j) 大规模最终实验——Nemotron-4 15B upcycling on 1T tokens：E8G8T8 (64 experts top-8) val loss 1.320 / MMLU 66.2，E8G1T2 (8 experts top-2) val loss 1.306 / MMLU 67.6，均优于续训稠密模型的 val loss 1.377 / MMLU 65.3（Table 1）。

- 硬件平台是什么，配置是什么。
  - NVIDIA GPU（论文未明确说明具体 GPU 型号和数量），使用 Megatron-LM 进行分布式训练，支持 data parallelism + tensor parallelism + expert parallelism。Codebase：Megatron-LM（https://github.com/NVIDIA/Megatron-LM），训练框架还包括 NeMo。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) **Nemotron 2B**——decoder-only Transformer，SwiGLU 激活，RoPE，max seq len 4096，no dropout，no bias，untied embedding，vocab size 256K，预训练 1.1T tokens；(2) **Nemotron-4 15B**——15B 参数多语言 LLM，预训练 8T tokens，vocab size 256K。
  - MoE 变体（消融用）：E8G1T1/T2、E8G8T8/T16、E64G8T8、E128G8T8、E256G8T8 等多种配置（E=experts数、G=granularity、T=topK）。
  - 数据集：(a) Nemotron 2B 消融——使用预训练数据（已见过的数据），110B tokens（约 10% 预训练 token 数）；(b) Nemotron-4 15B 消融——续训数据 blend（与预训练数据分布不同），0.1T tokens；(c) Nemotron-4 15B 大规模实验——续训数据 blend，1T tokens。Validation loss 在 1% held-out 数据上测量。
  - Benchmarks：MMLU (5-shot)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码位于 Megatron-LM 仓库的 moe/upcycling 分支：https://github.com/NVIDIA/Megatron-LM/tree/0431153bf1b5c405057b158189c260107d8b7c3a/megatron/core/transformer/moe#upcycling
  - 算法 Pipeline（Upcycling 完整流程）：

```
# ===============================
# Step 1: Coarse-Grained Upcycling (E experts, Top-K routing)
# ===============================
# 稠密 FFN: y = W2(activation(W1(x))) = W2(sigma(W1(x)))

# 复制 MLP 权重 -> 每个 expert 初始化相同
experts = [copy.deepcopy(dense_ffn) for _ in range(E)]

# Router 随机初始化
W_r = random_init(shape=(d_model, E))  # 路由器权重矩阵

# Weight Scaling: 对每个 expert 的 W1 和 W2 缩放
# 缩放因子 (针对 Squared-ReLU 推导, SwiGLU 也适用):
#   scale = (E * G^2 / T)^{1/3}
# 对 E8G1T1: scale = (8 * 1 / 1)^{1/3} = 2.0
# 对 E8G8T8: scale = (8 * 64 / 8)^{1/3} = 4.0
for expert in experts:
    expert.W1 *= scale
    expert.W2 *= scale

# ===============================
# Step 2: Softmax-Then-TopK Routing (Forward Pass)
# ===============================
# 输入: x in R^{S x d_model} (S tokens)
r_logits = x @ W_r                          # (S, E), Router logits
r_probs = softmax(r_logits, dim=-1)         # (S, E), Router probabilities
topk_probs, topk_idx = topk(r_probs, T)     # (S, T), select top-T experts

# Expert 计算
MoE_output = zeros_like(x)
for t in range(T):
    expert_idx = topk_idx[:, t]             # (S,)
    expert_weight = topk_probs[:, t]        # (S,)
    # gather tokens to each expert, compute FFN
    MoE_output += expert_weight * gather_expert_ffn(x, expert_idx)

# ===============================
# Step 3: Fine-Grained Upcycling (Virtual Group Init)
# ===============================
# 例子: E8G8T8 = 64 experts, shard by G=8, route to T=8 experts
# 原始 FFN hidden dim = H -> expert hidden dim = H/G

# Shard dense FFN weights by intermediate dim
# W1 in R^{d_model x H} -> shard into G parts: {W1_0, ..., W1_{G-1}}
# W2 in R^{H x d_model} -> shard into G parts: {W2_0, ..., W2_{G-1}}
shards_W1 = W1.split(G, dim=1)   # shape per part: (d_model, H/G)
shards_W2 = W2.split(G, dim=0)   # shape per part: (H/G, d_model)

# 复制 shards 形成 E copies per shard, total N=64 experts
experts = []
for i in range(G):
    for _ in range(E):
        experts.append((shards_W1[i], shards_W2[i]))
# expert_0..expert_7 都是 shard_0, expert_8..expert_15 都是 shard_1, ...

# Virtual Group Router 初始化
# Router weights W_r in R^{d_model x N}, N=E*G=64
# 将 W_r 分为 G 个 group, 每个 group 内的 E 列相同
W_r = random_init((d_model, E))  # 只生成 E 组权重
W_r_full = zeros(d_model, N)
for g in range(G):
    W_r_full[:, g*E : (g+1)*E] = W_r  # copy same weights
# 初始化后 router 在每组内相同, 保证 top-T 均匀覆盖 G 个 group

# ===============================
# Step 4: Loss Computation & Training
# ===============================
# Load balancing aux loss (ST-MoE / Switch Transformer):
# f_e = 1/T * sum_{token} 1_{token routed to expert e}
# P_e = 1/T * sum_{token} softmax_prob[token, e]
# L_aux = E * sum_e f_e * P_e
# Total loss: L = L_LM + alpha_aux * L_aux  (alpha_aux = 1e-2)

# 优化器: Adam (Megatron-LM default)
# 学习率: warmup -> peak (2e-4 or 1e-4) -> cosine decay -> min (2e-5)
# Batch size: 1024 (4M tokens) for Nemotron 2B
```

  - 关键张量计算流（以 E8G1T2 单个 token 为例）:
    ```
    x in R^{d_model} (e.g., 4096)
    
    # Attention 输出 -> MoE Router
    r = W_r^T @ x          # (8,) Router logits
    s = softmax(r)          # (8,) Router probabilities
    [p1, p2], [e1, e2] = top2(s)  # 选 top-2 experts
    
    # Expert 1 计算
    h1 = W1_{e1} @ x       # (H,) 第一线性投影
    a1 = sigma(h1)         # SwiGLU activation
    o1 = W2_{e1} @ a1      # (d_model,) 第二线性投影
    
    # Expert 2 计算
    h2 = W1_{e2} @ x
    a2 = sigma(h2)
    o2 = W2_{e2} @ a2
    
    # 加权输出
    y = p1 * o1 + p2 * o2   # (d_model,)
    ```

  - 大规模实验最终参数（1T tokens upcycling of Nemotron-4 15B）:
    - E8G8T8: peak LR = 3e-4, cosine decay to 1/100 of pretraining min LR
    - E8G1T2: same LR schedule, different batch size (top-2 per-expert more tokens)

## X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：X-MoE 提出三项系统级算法优化以支持 expert-specialized MoE（DeepSeek 风格，fine-grained experts + large top-k routing）在非 NVIDIA HPC 平台上的大规模训练：
    (1) **PFT（Padding-Free Token buffers）+ 全 padding-free MoE pipeline**：设计稀疏数据结构 PFT（token_buffer x + ERI-arrays: token_ids, expert_ids, tokens_per_expert, combine_weights），消除传统 MoE dispatch/MLP/combine 各阶段的 zero-padding。PFT 仅存储有效路由 token，dispatch 使用 uneven alltoall 替代 even alltoall，通信量和激活内存随实际 token 数（非 capacity）线性增长。复杂度从 GShard 的 O(ckbsh)+O(ckb²s²) 降至 O(kbsh)。
    (2) **RBD（Redundancy-Bypassing Dispatch）**：分层两级 dispatch——Stage 0 选择 Pilot tokens（跨节点去重后的最小 token 集）和 Local replica（节点内重复的 token）；Stage 1 仅 Pilot tokens 通过跨节点 uneven alltoall + 节点内从 Pilot 重建 Local replica；Stage 2 Local replica 通过快速 intra-node alltoall 分发。消除 Dragonfly 拓扑下因 large top-k 导致的跨节点重复通信（实测冗余率可达 75.1%）。
    (3) **SSMB（Sequence-Sharded MoE Blocks）**：在 TP+EP 混合并行中，进入 MoE block 时将序列切分到 EP ranks（drop partial tokens），使 Adispatch 和 Acombine 激活内存按 TP group size 比例缩减。MoE block 结束后通过 all-gather 恢复完整序列。解决 expert-specialized MoE 中激活内存（尤其是 dispatch/combine 阶段）从模型参数转移的瓶颈。
  - 实验比较：(a) 可训练性与吞吐量：X-MoE vs DeepSpeed-MoE vs DeepSpeed-TED vs Tutel，在 Small（10.1B）/Medium（55.2B）/Large（201.4B）/Super（545.4B）四种 DeepSeek 风格模型配置上，256-1024 AMD MI250X GPU，对比训练吞吐量（TFLOPs）和 OOM 情况；(b) Weak scaling（16→256 GPU）和 Strong scaling（128→1024 GPU）；(c) MoE layer 时间分解：X-MoE vs DeepSpeed-MoE 的 gating/dispatch/alltoall/expert compute/combine 各阶段延迟；(d) 激活内存：X-MoE vs DeepSpeed-MoE vs Tutel 每 MoE layer 内存消耗；(e) SSMB vs activation checkpointing 的吞吐量对比；(f) Cross-platform 验证：8×NVIDIA A100 40GB 上 X-MoE vs DeepSpeed-MoE vs Tutel。

- 硬件平台是什么，配置是什么。
  - 主平台：Frontier 超级计算机（OLCF），每节点 4×AMD MI250X GPU（每 GPU 2 GCD，视为独立 GPU），GCD 间 Infinity Fabric 互联（50-200 GB/s 峰值），节点间 Slingshot 25 GB/s NIC（Dragonfly 拓扑），最多使用 128 节点（1024 MI250X GCD）。每 effective GPU 峰值 191.5 TFLOPs。
  - 跨平台验证：8×NVIDIA A100 40GB GPU。

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-MoE 风格的 expert-specialized MoE。四个配置：
    - Small: H=2048, HFFN=1408, 64 experts, top-k=6, 28 layers, 10.1B params (1.3B activated)
    - Medium: H=5120, HFFN=1536, 128 experts, top-k=6, 28 layers, 55.2B params (5.2B activated)
    - Large: H=7168, HFFN=2048, 256 experts, top-k=8, 28 layers, 201.4B params (11.5B activated)
    - Super: H=7168, HFFN=2560, 256 experts, top-k=8, 61 layers, 545.4B params (28.7B activated)
  - 数据集：论文未明确说明具体训练数据集名称，使用标准 LLM 预训练数据。Benchmark：训练吞吐量 (TFLOPs)、迭代时间、激活内存消耗、alltoall 延迟。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Supercomputing-System-AI-Lab/X-MoE，集成于 DeepSpeed 0.15.5。
  - PFT padding-free MoE pipeline 伪代码（对应 Listing 1）：
    ```
    # === Step 1: Gating ===
    # tokens: [S, H], S=sequence_length, H=model_dim
    logits = softmax(FFN(tokens), axis=-1)  # [S, E]
    combine_weights, top_experts = topk(logits, k)  # [S, K] each

    # === Step 2: PFT Construction ===
    # Input: top_experts [S, K], combine_weights [S, K], max_token_count
    flat_top_experts = flatten(top_experts)  # [S*K]
    flat_combine_weights = flatten(combine_weights)  # [S*K]
    sorted_indices = argsort(flat_combine_weights)
    sorted_top_experts = flat_top_experts[sorted_indices]
    # Token dropping: one_hot + cumsum + mask
    one_hot_enc = one_hot(sorted_top_experts, num_classes=E)  # [S*K, E]
    rank_in_expert = cumsum(one_hot_enc, axis=0)  # [S*K, E]
    weight_mask = rank_in_expert <= max_token_count
    # Filter retained tokens → ERI-arrays
    filtered_indices = sorted_indices[weight_mask]
    token_ids = token_ids[retained_token_ids]  # [B], B=retained tokens
    expert_ids = expert_ids[retained_token_ids]  # [B]
    combine_weights = combine_weights[retained_token_ids]  # [B]
    tokens_per_expert = histogram(expert_ids, bins=E)  # [E]

    # === Step 3: Padding-free Dispatch ===
    # Gather kernel: reorder tokens locally per expert routing
    dispatch_in = gather_kernel(gate_out, pft.token_ids, pft.expert_ids)  # [B, H]
    # Uneven alltoall: only valid tokens, no zero-padding
    pft.tokens_per_expert = alltoall(pft.tokens_per_expert)
    dispatch_out = alltoallv(dispatch_in, pft.tokens_per_expert)  # [Bexp, H]

    # === Step 4: Padding-free MLP (sequential GeMM) ===
    # For expert i (0..E_local-1), process tokens 
    #   dispatch_out[sum(tpi[:i]):sum(tpi[:i+1])] with expert_i weights
    inter_activ = sequential_gemm(pft.x, w1)  # [Bexp, HFFN]
    mlp_out = sequential_gemm(inter_activ, w2)  # [Bexp, H]

    # === Step 5: Padding-free Combine ===
    combine_in = alltoallv(pft.x, pft.tokens_per_expert)  # [B, H]
    combine_out = scatter_kernel(combine_in, pft.token_ids,
                                  pft.expert_ids, pft.combine_weights)  # [S, H]
    ```
  - RBD 分层 dispatch 流程：
    ```
    # Stage 0 (S0): Pilot Selection
    # 对每个 token 的 k 个 destination experts，按 destination node 分组
    # 每组随机选 1 个 pilot token，其余标记为 local replica
    # 构建 s1_mapping_indices: local replica → pilot token 的映射

    # Stage 1 (S1): Inter-Node Exchange (Pilot Only)
    pilot_tokens = gather_kernel(x, pilot_token_ids)  # 仅 pilot tokens
    pilot_tokens = alltoallv(pilot_tokens, ...)  # 跨节点 uneven alltoall
    # Local replica 在目标节点从对应 pilot token 重建:
    local_replica[i] = pilot_tokens[s1_mapping_indices[i]]

    # Stage 2 (S2): Intra-Node Exchange (Local Replica Only)
    local_replica = intra_node_alltoallv(local_replica, ...)  # 节点内快速 alltoall
    # Merge pilot + local replica + reorder by expert index
    ```
