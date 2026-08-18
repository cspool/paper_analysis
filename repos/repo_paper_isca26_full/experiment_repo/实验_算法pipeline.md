# 实验_算法pipeline

## A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 IcePack 的三级无损 syndrome 压缩算法 pipeline：(1) 空间聚类（spatial clustering）——把表面码格点上由单个 X/Z 数据错误触发的水平对、垂直对（2 个非零 syndrome → 1 个 index）与 Y 错误触发的 cross（4 个非零 syndrome → 1 个 index）编码为"首个非零 syndrome 的索引 + 2-bit opcode"（cross=3、vertical=2、horizontal=1、isolated=0，按 Table I 优先级覆盖）；(2) 时间聚类（temporal clustering）——对 opcode 0 孤立 syndrome 做静态预测（测量错误比错误链常见得多，下一轮同位置大概率复现），预测命中则丢弃该 index，预测失败则以 opcode 0 显式补发一个 index 实现无损纠错；(3) Rice-Golomb 编码（RGE，m=2^k）——对压缩后保留 index 之间的 gap（近似几何分布）做变长编码：unary quotient + truncated-binary remainder。
  - 实验比较对象：AFS [11] 的 sparse representation（只传非零 syndrome 索引，不做任何非零 syndrome 压缩，index reduction=0.0）、无 4 K 压缩的数字读出 baseline。指标：syndrome index 减少率、总传输 bit 数（data-volume reduction factor）、数据量→串行化延迟与热负载。
- 硬件平台是什么，配置是什么。
  - 软件评估无特殊硬件：标准 x86 Linux 机器 + Docker（ARM/Apple Silicon 亦可），论文建议 64+ 核 CPU，约 1 GB 磁盘（Docker 镜像），完整跑完约 3–9 小时。无需 GPU 或量子硬件。
- 模型是什么。数据集和bench分别是什么。
  - 无 ML 模型（纯 QEC 稳定子电路采样 + 无损压缩）。数据集由 Stim（量子稳定子电路模拟器）运行时生成：表面码 code distance d=11–31（主分析 d=11–21、1000 个逻辑 qubit，每个 d–p 组合 20000 次独立运行、跨多测量轮）；物理错误率 p ∈ {10^-4, 10^-3, 10^-2}（0.01%–1%）；噪声模型以 phenomenological 为主，扩展 circuit-level（5p 测量噪声+2p reset、2p/1p 两配置）、非 IID qubit（Google Willow 实测检测概率分布，10 组 × 10 万 ancilla）、错误率 10× 漂移、multi-bit burst errors（宇宙射线）、leakage。规模对标 RSA factoring 所需 1000 逻辑 qubit。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：是。artifact 公开于 Zenodo https://doi.org/10.5281/zenodo.19446086（CC BY 4.0），Docker 一键运行：`docker run --rm -v $(pwd)/results:/output icepack-artifact`，产出对应论文图 5/7/8/15 的 4 个 CSV + 4 个 PNG；冒烟测试加 `-e SMOKE_TEST=1`；参数（d、p、Monte Carlo 样本数、噪声模型）可在 artifact 内 icepack.py / artifact.ipynb 修改。依赖 stim、numpy、pandas、matplotlib、mpire（容器内置）。
  - 算法 pipeline 执行例子（单轮 syndrome 流）：Stim 采样 d=21 表面码 syndrome 位图，ancilla 按行主序编号 → 空间聚类：非零 syndrome 位于 index 42、45、51，42 与 51 构成垂直对 → 记 (index=42, opcode=2)，45 孤立 → (45, opcode=0)；cross 优先级最高，其次 vertical、horizontal（Table I）→ 时间聚类：round t 的 opcode 0 条目生成下一轮同位置预测；round t+1 命中则丢弃（V_o=0），未命中则补发 (index, opcode=0) 以标识"该位置无 syndrome"（300 K 端未收到 index 即默认该位置有 syndrome）→ RGE：对已发送 index 流计算相邻 gap n，q = ⌊n/2^k⌋ = n >> k（unary：q 个 1 + 1 个 0）、r = n mod 2^k = n[k:0]（k bit truncated binary）拼接；例：m=4、gap=11（ID1=632→ID2=643）：q=2→3'b110、r=3→2'b11，得 5-bit 码 5'b11011，比 10-bit 绝对索引省一半 → 300 K 端在线无损解压（unary 计数→gap 累加→opcode 常量偏移展开，解压延迟 2.5 ns，Synopsys DC + Nangate 45nm 综合）。
  - 效果：总 bit 数比 AFS 稀疏表示少 2.4–4×（d=21：p=10^-4/10^-3/10^-2 时 clustering 贡献 1.99×/1.94×/1.61×、RGE 贡献 1.40×/1.78×/2.50×，Table II）；比无压缩数字读出最高 300×；circuit-level 噪声下仍达 1.9–3.1×（vs AFS）；非 IID qubit 下压缩率与理想配置偏差 <1%。

## AQuant: Repurposing CODEC for VLM Acceleration via Adaptive Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 AQuant 动态视觉 token 量化算法（三阶段）：(1) Exponent-Similarity Detection——从 N 个视觉 token 中按间隔 F=⌊N/M⌋ 选每 F 个连续 token 的中间 token 为 M 个候选 base token，用指数近似 L1 距离 D̄(i,j)=Σ_k |sign(T_k^i)·2^Ē_k^i − sign(T_k^j)·2^Ē_k^j|（只取 sign+exponent 位，整数移位/减法）选出每 token 最近的 base；(2) Adaptive Quantization——delta δ^i=T^i−B^i 内按幅值 top p（默认 25%）元素量化到 INT4、其余 75% 量化到 INT2（bitmask 标记，构成互补稀疏矩阵），base token 保持 INT8；(3) Result Reconstruction——W_q·T = W_q·B + W_q·δ，M 个候选 base 的 W_q·B 预计算、运行时直接查表选择，W_q·δ 低精度 GEMM，二者相加重构 Q。解码阶段把 KV projection 后的 KV-cache 重新量化后存 off-chip、使用时在线重构以省内存带宽。
  - 实验比较：baseline 为 PyTorch 全精度 VLM（GPU-Full，按面积预算核数缩放的 Jetson AGX Xavier 模型）；对比 GPU-Full-unscale（Xavier 实测）、GPU-Mixed-precision（理想混合精度解析上界，speedup=8/(4×0.25+2×0.75)=3.2×）、GPU-AQuant（算法跑在 GPU 上）、GPU-VisPruner（VisPruner 剪 token）、OliVe、LLM.265、CMC、AQuant-Pruning（去 INT2 剪枝消融）。指标：准确率、加速比、能耗效率、prefilling/decoding 分阶段延迟。
- 硬件平台是什么，配置是什么。
  - 算法评估：Python 实现 + PyTorch 框架（开源 VLM 实现），无重训练，校准数据取训练集 10%；GPU 基线实测平台 NVIDIA Jetson AGX Xavier。
- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaVA、VideoLLaVA、Qwen2.5-VL 72B，共 3 个 VLM。数据集/bench 共 14 个：VQAv2、GQA、TextVQA、POPE、MM-Bench、MMVet、Wild、ScienceQA、VisWiZ、ActivityNet、MSVD、TGIF、MSRVTT、Video-MME。超参数：F=18（对应 7.4% INT8 base token）、p=25%。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：AQuant 算法与架构代码论文未明确说明是否开源；模型采用开源实现（LLaVA/VideoLLaVA/Qwen2.5-VL）。基线 CMC、LLM.265、OliVe 由作者复现算法并自建 cycle-accurate 模拟器（CMC: ASPLOS'24；LLM.265: MICRO'25；OliVe: ISCA'23）。
  - 算法 pipeline 执行例子（prefilling，Q=W_q·T 为例，T∈R^{N×K}）：① 设 N 个视觉 token、F=18 → M=N/18 个候选 base token，取每 18 个连续 token 的中间 token；② 指数相似检测：提取每 token 每元素的 sign+exponent（视觉 token 指数集中 [0,8]，2^exponent∈[1,256] 用 9 bit + 1 bit sign = INT10），对每个 token i 计算 D̄(i,j)=Σ_k |sgn(T_k^i)·2^Ē_k^i − sgn(T_k^j)·2^Ē_k^j|（j=1..M），取 argmin 得 B^i；③ 求 delta δ^i=T^i−B^i（窄分布）；④ 每行 δ^i 内 top 25% 大值 → INT4（bitmask 置 1）、其余 → INT2，base token B 保留 INT8；⑤ GEMM 分解：W_q·T ≈ W_q·B + W_q·δ，W_q·B 只对 M 个候选 base 预计算并查表（每 token 直接取对应候选的输出）；W_q·δ 以 INT2/INT4 混合精度乘 INT16 权重；⑥ 逐 token 相加重构 Q 矩阵，再送 Softmax/GELU/LayerNorm 等 FP 非线性算子。指数越界（[0,8] 之外）token 作为 outlier 跳过相似检测直接高精度处理。
  - 效果：平均准确率损失 0.7%（AQuant-Pruning 消融损失 23%）；理论计算相对全 INT8 减少 3.2×；fast-motion 视频仅 0.83% 损失；F 在 12–24 扫描后固定 18（F=24 精度骤降），p 在 4 个 benchmark 扫描后固定 25%（p=20% 时 MSVD 掉 2.1% 不可接受）。

## Approaching Shannon Bound with Lossless LLM Weight Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 tile-level 无损 ANS（rANS）LLM 权重压缩算法 pipeline：(1) 信息论分析——逐层把权重矩阵 W^(l) 的每个元素按数值格式视为离散符号，建经验直方图估计 Shannon 熵 H^(l) = −Σ_i p_i^(l)·log2 p_i^(l)，模型级为按参数量加权平均 H_model = Σ_l H^(l)|W^(l)| / Σ_l |W^(l)|，即任意无损编码下的 bits/weight 下界；(2) 离线压缩——对每个投影矩阵（W_Q/W_K/W_V 等）先 profiling 确定 tensor-core tiling 几何（如 128×32、256×64、128×128），聚合整层权重统计构造共享 ANS codebook（概率精度 b=12，2^12 项），把权重张量切成与 GEMM tile 对齐的 tile，每 tile 以独立初始状态 rANS 编码为自包含 substream（ANS 支持任意初始状态不损失压缩率，故 tile 可独立解码），tile 起始偏移记入 4B/条 的 offset table；(3) 在线解码——运行时按 GEMM tiling 顺序按需解码 tile 直接写 shared memory 供 tensor core 消费。严格无损：解码结果与原始权重 bit-exact，零精度损失，且与量化/剪枝/低秩等有损方法正交可叠加。
  - 实验比较对象：名义存储位宽（bf16/fp8/int8/fp4/int4/sq8/awq4）与 Shannon 熵界（Fig. 6）；SOTA 无损压缩系统 NeuZip（层级 decompress-store-compute）与 DFloat11（H200 微基准）；CUTLASS 原生 GEMM（kernel 级）；KTransformer（OOM 场景 CPU offload）。指标：bits/weight、与 Shannon 界的比特差、TFLOP/s、吞吐 tokens/s、最大可行 batch size、median TPOT。
- 硬件平台是什么，配置是什么。
  - 两台 GPU 服务器：① 8× NVIDIA A100（80 GB HBM2e，2 TB/s 峰值带宽）；② NVIDIA Hopper H200。软件栈 PyTorch 2.5.1 + CUDA 12.1，GEMM 基于 CUTLASS。
- 模型是什么。数据集和bench分别是什么。
  - 模型（1.5B–405B，稠密 + MoE）：Qwen2-1.5B、Mistral-7B-v0.3、Qwen-14B、DeepSeek-LLM-67B、Llama-3.1-405B、Mixtral-8x22B（176B 总参数）。数值格式：bfloat16（bf16）、FP8-E5M2、INT8、FP4-E2M1、INT4、SmoothQuant sq8、AWQ awq4。数据集/benchmark：无损压缩为纯后处理、无需校准数据集；benchmark 为单层投影 GEMM 微基准（4096 input tokens、batch 1–64）与 SGLang 端到端推理（seq len 1024/2048）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文自身代码未找到公开仓库（arXiv 2606.15789，论文未给代码链接，联网搜索未见官方 repo）。基座开源：rANS 内核基于 DietGPU（https://github.com/facebookresearch/dietgpu），GEMM 基于 CUTLASS（https://github.com/NVIDIA/cutlass），serving 集成于 SGLang（https://github.com/sgl-project/sglang）。
  - 算法 pipeline 执行例子（离线压缩 W∈R^{K×N} 的一个 128×128 tile）：① 按数值格式把 tile 内权重视为符号 s∈{0..2^b−1}（如 int8 取 8-bit 码），聚合整层计数频率归一化为 ANS 概率表（b=12，频率定标到 2^12，共享 codebook）；② rANS 编码：对符号逆序维护状态 x，编码 s：x ← ⌊x/fs⌋·R + (x mod fs) + cs（R=2^b、fs 归一化频率、cs 累积频率）；③ 每 tile 独立初始状态 x0，输出 tile substream，offset 表记录其起始位置；④ 运行时解码（Algorithm V.2）：每 warp lane 维护一个 rANS 状态 s.value，x ← s mod R → 查 shared memory 解码表 T̃[x]=(σ,f,c) → w ← DecodeSymbol(σ) 写入 shared memory tile A[r,c] → s.value ← f·⌊s.value/R⌋ + (x − c) → s 低于归一化阈值时以 coalesced 32-bit load 从压缩流补位。效果：ANS bitrate 与 Shannon 界差距仅 0.01–0.05 bits/weight（bf16 因 2^12 表有限精度定标约 0.1–0.2 bits 偏差）；实测熵：bf16 约 10–12 bits、int8 约 4–5 bits、int4 仅 0.6–1.0 bits（熵比达 6–10× 冗余）；sq8/awq4 仍有 1.1–1.3× 冗余；元数据开销仅 0.015%–0.108% 权重大小（A100 32×128 / H200 64×256 tile）。

## Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 近似匹配（论文本体为硬件架构，此处仅覆盖其"近似激活函数计算 + 精度保证"部分）：实现 = BF16 低精度 + Taylor 截断指数（n=4..7，e^x ≈ 1+x+…+x^n/n!）与 Newton 迭代开方替代精确 exp/sqrt 的近似激活计算 pipeline：Curry ALU 以 ArgReg=6 为迭代轮计数器，从最内层向外每轮执行 *=X、/=IterRound、+=1 直到 IterRound=0，每通道 16 bank × 2 路 = 32 路并发；sqrt 同法（Newton 迭代）。实验比较：Llama2-7B perplexity——FP32 基线 vs 原生 BF16 vs BF16 泰勒截断 n=4/5/6/7，三档长度（prefill 73/341/1139 + decode 15/65/270 tokens）：相对偏差 ≤0.3%（最显著 medium 档 n=5..7 相对 FP32 为 −0.251%），且误差不随上下文增长而累积。
- 硬件平台是什么，配置是什么。
  - 无 GPU 实测：perplexity 与全部性能评估均跑在 CompAir cycle-accurate 模拟器（ramulator2.0 + Booksim + CENT 模拟器 + SRAM-PIM 规格 [14]）上；近似 exp 的硬件开销由 Synopsys Design Compiler + UMC 28nm 综合（4×Curry ALU 的资源少于一个定制 16 输入 Softmax 单元，Vivado 对比）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Llama2-7B（perplexity 主评估），其余端到端性能评估覆盖 Llama2-7B/13B/70B、Llama3（GQA 场景）、Qwen-72B、GPT3-175B。数据集：论文未明确说明 perplexity 所用具体数据集名称（仅给出 short/medium/long 三档 prefill/decode 长度）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Man0xbfc00380/comp-air.git（已确认，含 NoC 计算模型与 translate/ 翻译器）。算法 pipeline 执行例子（16 bank 上的 Softmax 分子计算）：① 每 bank 对本地分片 x_j 发 exp packet，flit Data=16b BF16；② Curry ALU 配置 ArgReg=6（迭代轮）、IterArg=1、IterOp='-'：每轮 ArgReg 先 −=1 再执行 acc = acc×X/IterRound + 1（自最内层向外展开 Taylor 截断），IterTag 触发 ArgReg 动态更新；③ 计算结果就地替换 flit Data 继续路由；④ NoC_Reduce 以 4 层二叉树对 16 bank 的 exp 结果求 Softmax 分母，广播回各 bank。效果：perplexity 相对 FP32/原生 BF16 偏差 <0.3%、不随 1139-token 长上下文累积；硬件代价为零额外流水级（计算与 switch traversal 并行），支持 32 路并发指数计算。

## Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为训练无关（training-free）的 self-speculative decoding 格式变换 pipeline：把目标模型（BF16）的权重与 KV cache 拆成"speculation data + verification data"两份数据，草稿模型不独立存储。(1) 非结构化值剪枝：权重采用 Wanda 的 activation-aware 剪枝（用校准数据计算激活 L2 norm，与权重逐元素相乘得 importance score，按 top-k 保留）；KV cache 采用 per-token 幅度剪枝（Key cache 存在 channel-wise outlier，Value cache 的 per-token 剪枝在注意力机制内功能等价于 output-aware 剪枝）。(2) 尾数截断（mantissa truncation）：不用量化而是直接截断低位 mantissa 比特——不改变数值表示、重建开销低，草稿模型是目标模型的严格比特子集。(3) 指数压缩：BF16 的 8-bit 指数占位宽 50%，是压缩率瓶颈；提供两种配置——Cassandra-1 用 unary 编码（1/01/001/…，以 1 结尾的码字边界可直接并行解析）做无损熵压缩（权重/KV 指数 Shannon 熵约 2.6/2.7 bits，实际平均压到约 2.85 bits）；Cassandra-2 用 MX 格式共享指数（压缩率更高、轻微精度损失）。draft 推理只加载 speculation data，zero-padding 重建为标准浮点格式后在标准 FP 单元上执行（降带宽不减算术）；target 推理加载 speculation+verification 全部数据完全重建原始模型做并行验证，无独立草稿模型 → 零额外显存。draft 长度 γ 在 3–5 内取最优。实际算法顺序：指数压缩 → mantissa 截断（图 4）。
  - 实验比较：zero-shot 精度 vs BF16 baseline 与损失压缩方法（SmoothQuant W8A8（vLLM 官方 INT8）、QoQ、SqueezeLLM、DuQuant、Wanda）；性能 vs BF16、INT8/FP8 量化与 speculative decoding（Draft&Verify、MagicDec、Lookahead Decoding、EAGLE-3）；内存容量 vs Llama3-based speculative decoding 与 Eagle-3。
- 硬件平台是什么，配置是什么。
  - 软件模拟器：PyTorch + 自定义 CUDA kernels（仅功能建模，不带来 GPU 性能收益）；超参搜索在 NVIDIA A100（8B 模型约 5 分钟）。性能：Accel-Sim 模拟的 Nvidia RTX 4090（用 Ampere 架构 trace + 近似 RTX 4090 的配置，因 Accel-Sim 不支持 Ada-Lovelace）与 Nvidia Jetson AGX Orin；NPU 为扩展 Scale-Sim + LPU simulator 的 cycle-level 模拟器（64 TFLOPS MAC、273 GB/s 带宽、128GB LPDDR5X、9MB scratchpad）。面积功耗：Synopsys Design Compiler 28nm + Samsung 28nm SRAM Compiler。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Deepseek-R1-Distillated-Llama3-8B、Qwen3-8B-Thinking、Qwen3-4B-Thinking-2507。benchmark：GPQA-Diamond、Math-500、AIME 2025（zero-shot 精度；GPQA-Diamond 与 Math-500 随机抽 100 题，AIME2025 全题）；LiveCodeBench、GPQA-Diamond、Longbench(QMSum)、Math-500（性能与接受率场景，按 benchmark 取平均输入/输出长度）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - Cassandra 自身代码：论文未提供开源链接，未检索到公开仓库（无法确认）；arXiv:2605.26558 仅论文正文。baseline 均为官方开源实现（Draft&Verify、MagicDec、Lookahead Decoding、EAGLE-3 官方实现，SmoothQuant 用 vLLM 官方 INT8 量化实现）。pipeline 伪代码（以权重 W 为例）：
    1. importance = |W| ⊙ norm2(X_calib)（Wanda，激活 L2 norm 按输出通道广播）；mask = topk(importance, 1−w_p)；spec_W = W[mask]，ver_W = W[¬mask]（KV cache 按 per-token 幅度 top-k 同法）。
    2. 指数压缩：spec_W 的 BF16 8-bit 指数按频率建 unary codebook（高频→短码，码字以 '1' 结尾），拼接为 bitstream（Cassandra-1）；或同 block 元素共享指数（Cassandra-2，MX）。
    3. 尾数截断：spec_W mantissa 保留高 w_t 位（默认 4-bit），低位丢弃。
    4. draft 前向：spec_W 与 KV 的 speculation 部分 → decoder 重建（unary 解码 = parallel zero counter 数连续 0 + LUT；bitmap de-sparsification 补 0）→ 标准 FP GEMM；target 前向：加载 ver_W 补全后完整 BF16 GEMM，对 γ 个候选 token 并行验证（greedy 或 rejection sampling，式 (1)：n ← min({i−1 | r_i > p_i/q_i} ∪ {γ})）。
    5. 超参数目标函数 J = α / (S_w(1−w_p)(B−w_t) + S_kv(1−kv_p)(B−kv_t))：grid search 剪枝率 30–60%（步长 10%）、截断 0–5 bits（步长 1 bit），dev set 8 样本；默认配置 40% 剪枝 + 4-bit 截断（权重与 KV cache 相同），可直接迁移到其他模型。
  - 效果：性能 1.78×–2.41×（vs BF16；INT8 量化仅 1.3×）；Cassandra-1 精度与 BF16 完全一致（Deepseek-R1-Distillated-Llama3-8B：GPQA 49.0 / Math-500 87.0 / AIME 26.7），Cassandra-2 与 SmoothQuant 相当；接受率 Cassandra-1(γ=5) 0.74–0.88、Cassandra-2(γ=3) 0.74–0.91；固定内存预算下比 Llama3-based speculative decoding 多生成 11.59× token、比 Eagle-3 多 1.81×。

## Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference（PLENA）

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 PLENA 面向 MX（microscaling）数据格式的非对称量化算法 pipeline（Pathway 2）：(1) 可配置 MX 数据格式——单级缩放方案，参数化 (M,E,S,B) 的 MXFP（minifloat）与 (M,S,B) 的 MXINT，块内元素共享一个 E8M0 幂次缩放因子；(2) 非对称量化——W/A/KV 可分别设不同精度与数据类型（如 4/4/4、4/4/16），权重用 MXINT、激活可选 MXINT/MXFP、KV cache 低精度；(3) 块级裁剪（microscaling block-wise clipping）——引入裁剪参数 p∈[0.5,0.99] 把有效范围缩到 [p·min_w, p·max_w]，平衡裁剪溢出误差与 inlier 下溢误差；(4) 输出范数引导的块级裁剪搜索——把裁剪搜索并入 GPTQ 式迭代误差传播：外层用 Hessian 信息 H_F=2X_F·X_F^T 迭代校准权重（残差传播更新 W_b+=δ_F），内层对每行权重搜索裁剪百分位 P_b*=argmin‖X_b(W_b−Q(W_b;P_b,τ))^T‖₂²，最小化输出块重建误差而非权重块误差；(5) 选择性旋转（selective rotation）——QuaRot 式 Hadamard 旋转只施加到增益为正的层子集 S，权重不旋转（MX 小块共享指数下旋转反而增 perplexity），激活/KV 旋转在运行时在线执行（l_rot*(X)=Q(XH)·H^{-1}·Q(W)，PLENA 硬件原生支持运行时乘 H^{-1}）。
  - 实验比较：quantization 基线 GPTQ、AWQ、OmniQuant、QuaRot、SmoothQuant、Atom、MicroScopiQ、M-ANT（W/A/KV 三档 4/16/16、4/4/16、4/4/4），FP16 全精度 baseline；消融对比 MXINT vs MXFP、RTN、rotation、Errw（权重范数裁剪）vs Erry（输出范数裁剪）、full-system emulation。指标：WikiText-2 perplexity（GEMM-only emulation，非线性算子全精度）、下游 zero-shot（PIQA/WinoGrande/HellaSwag/Arc-Easy/Arc-Challenge/LAMBADA）、agentic 任务（HumanEval pass@1、GSM8K EM、BFCL-Web Search Base Acc）、内存足迹与峰值带宽（Table X）。
- 硬件平台是什么，配置是什么。
  - 量化/PTQ 流程在 NVIDIA H100 GPU 上运行，全程约 2–20 GPU hours；推理端系统级评估平台为 4×A100 SXM（80 GB HBM、1.99 TB/s）、4×H100 SXM（80 GB、3.35 TB/s）、16×TPU v6e（32 GB、1.56 TB/s）；GPU 环境 Ubuntu 22.04、CUDA 12.8、Python 3.11、PyTorch 2.8.0、vLLM 0.10 V1，TPU 环境为 v2-alpha-tpuv6e 软件。
- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA-2（7B/13B/70B）、LLaMA-3（8B/70B）、LLaMA-3.1-8B、LLaMA-3.3-70B、GPT-OSS 20B（MoE）、Qwen3（8B/32B）、LLAMA3.2-1B（DSE 快速迭代用）。数据集/bench：WikiText-2（perplexity 主指标）、LM Eval 六项 zero-shot（PIQA/WinoGrande/HellaSwag/Arc-Easy/Arc-Challenge/LAMBADA）、HumanEval、GSM8K、BFCL-Web Search Base、OSWorld-L（90k prefill/8k output，agentic 主 workload；另有 GSM8K 1.4k/0.2k、BFCL-W 114k/5k，Table XIII）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文称"will be fully open-sourced upon acceptance"，项目主页 https://plena-cam.github.io/ 状态为 "Code Coming Soon!"（arXiv:2509.09505），截至检索未见公开 GitHub 仓库，无法确认代码链接；quantization 框架与 compiler、simulator、ISA、RTL 计划一并开源。模型（LLaMA/Qwen/GPT-OSS）与数据集（WikiText-2 等）均开源。
  - 算法 pipeline 执行例子（LLaMA-3-8B 一个线性层 Y=XW^T，W∈R^{N×K}、X∈R^{M×K}，目标 W/A/KV=4/4/4、MXINT4 块 B=16 为例）：① 权重按 K 维切成块 W_b∈R^{N×B}（B=16），每块共享缩放因子 s=max|w|/max_τ（τ=(INT,4,16)，max_τ=7），对称量化 w_τ=clip(RTN(w/s),−7,7)，反量化 Q(w;s,τ)=s·w_τ；② 外层 GPTQ 式循环：对每列块 b 计算残差 δ_F=−(W_b−Q(W_b;P_b*,τ))·([H_F^{-1}]_{bb})^{-1}·(H_F^{-1})_{:,b}，H_F=2X_F·X_F^T 为校准集（WikiText-2 抽样）Hessian，残差传播到后续列块（W_b+=δ_F）；③ 内层对每行搜索裁剪百分位 P_b∈P^N（P⊂[0.5,0.99] 离散集）：P_b*=argmin‖X_b(W_b−Q(W_b;P_b,τ))^T‖₂²，以输入激活加权的输出范数而非权重范数作裁剪目标；④ 激活与 KV：对层子集 S 施加在线 Hadamard 旋转，激活 XH 后量化、运行时乘 H^{-1} 复原；权重不旋转（MXINT4 权重旋转后 6.83→6.98 反而变差）；⑤ 新产生的 K/V 向量 append 到 KV cache 前先 Hadamard 旋转抑制 outlier，再量化为 MX 存 HBM，读入 Matrix SRAM 后做逆变换进入 attention GEMM；⑥ 其余非线性/向量算子量化到 MiniFloat E6M5（相对 FP16 内存 −25% 且 perplexity 无损失）。
  - 效果：4/4/4 下 LLaMA-3-8B PPL 7.22（vs GPTQ 8.12、QuaRot-128G 7.36、MicroScopiQ 4/4/16 的 8.12）；4/16/16 下 6.45；full-system 中 Erry 裁剪 8.28→7.60、再加选择性旋转 7.22；W-only 4bit 下输出范数裁剪 6.45 优于权重范数裁剪 6.53；优化裁剪在 LLaMA-3-8B W4 设定下 perplexity 改善 5.5%；4/4/4 下 OSWorld-L（LLAMA-3.3-70B，BS=8）KV 足迹 239.26→59.81 GB、权重存储 129.46→32.36 GB、峰值带宽 8192→2048 GB/s（Table X）。MXFP 直接跑 PTQ 完全失效（4/4/4 下 LLaMA-3-8B PPL 256.22），证实 MX 格式需专门 PTQ 改造。

## Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：coset ensemble decoding（陪集集成解码）——在 UF 式聚类基础上引入"逻辑等价陪集"视角：K 次独立随机优先级采样生成 K 个优先级森林（Priority Forests），各森林经 Reverse-Order Elimination（ROE，逆序单趟 peeling）产出候选纠错与对应逻辑错误，再按逻辑结果多数投票（MAJORVOTE，限定在最小 |E_i| 候选子集上），以多项式时间近似陪集级最大似然 argmax_L p(L|s)。配套 lossless graph compression（聚类后仅保留根-根、根-边界边）降低 K 次探索的复杂度。属于"提出新算法模型 + 近似求解"类实现。
  - 实验比较：与 MWPM（PyMatching 软件实现，硬件对应 Micro-Blossom）、UF（自研 baseline UF 软件实现，硬件对应 Helios、QUEKUF）、BP+OSD（product-sum BP + OSD-CS order 15）比较：逻辑错误率 LER、解码延迟（均值/p95/p99 分布）、吞吐（decodes/s）、以及论文自定义的系统 infidelity 指标 Ĉ(R)（反馈解码场景下延迟对条件逻辑操作另一方保真度的影响）。
- 硬件平台是什么，配置是什么。
  - 算法级评估：Python-based hardware simulator（镜像最终微架构数据流、逐 cycle 计数、跟踪 multi-bank 布局下访存冲突、可逐项开关硬件优化做消融），与 RTL 交叉验证；硬件实现：Xilinx Virtex UltraScale+ VU19P FPGA，SystemVerilog HDL，Vivado 2024.2 综合，163 MHz，108k LUT / 43k FF / 252 BRAM（d=15）。
- 模型是什么。数据集和bench分别是什么。
  - 模型/码：surface code（rotated，periodic boundary conditions 周期边界，与 QUEKUF 同设定），码距 d∈{3,5,...,19}（精度）、d=3~11（延迟）、d=3~25（资源估算，d=3/9/15 为完整 Vivado 综合）；另有 repetition code（d∈{5,7}，p∈[0.04,0.08]）验证跨码族通用性。
  - 数据集/bench（噪声模型）：Stim 库生成 circuit-level depolarizing noise（Clifford 门后与轮间以 p 施加 depolarizing，测量错误为同概率 p 的经典比特翻转，reset 理想，q=p，T=d 轮）；biased/unbiased phenomenological noise（bias η=p_Z/p_X∈{0.5,1,10}）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文给出 https://github.com/IMSeonL/coset-ensemble-decoder（实测解析到 https://github.com/ihc-fan-lab/coset-ensemble-decoder，公开仓库：Python 解码器 uf_decoder.py、cycle-accurate 硬件模拟器、pipeline.py/find_subgraph.py 等与 4 档复现脚本；README 注明 Verilog RTL 稍后在 hardware_code/ 发布，当前硬件评估走软件模拟器）。依赖开源组件：Stim（噪声电路生成）、PyMatching（MWPM baseline）。
  - 算法 pipeline 执行例子（d 码距 surface code 一个 d 轮 syndrome 任务，syndrome parity s，候选数 K=24，见 Algorithm 1/2/3）：
    ① Phase I Clustering：Ĝ←CLUSTERING(G,s)，UF 式增长-合并把 syndrome graph 划分为若干非平凡子图（把 stabilizer 群划分为局部子空间 B_c={b∈F_2^m | b_g=0 for s_g∉G_c}，把全局 coset ML 松弛为聚类内局部优化，Lemma 2）；
    ② Phase II Ensemble Forest Exploration：for i=1..K：对每个 (v,e) 计算 keyed priority φ(v,e)=HashToUnit(seed,i,v,e)，按 φ 升序对顶点集与各邻接表排序；PRIORITYFORESTS（Algorithm 2）以 BFS 队列按优先级建森林，返回 parent[] 与发现序 σ；
    ③ ROE（Algorithm 3）：for t=|σ| down to 1：x=σ_t, r=parent[x]，若 p[x]=1 则 E_i←E_i∪{(x,r)} 并翻转 p[x],p[r]——单趟线性 peeling，复用森林遍历序，免全局叶子检测与度数重算；
    ④ L_i=DECODELOGICAL(E_i) 得到每个候选的逻辑错误；
    ⑤ MAJORVOTE(E,L)：对最小 |E_i| 子集按逻辑结果投票，采样频率 n_L/K 估计 p̃(L|s)，得最终纠错 Ê（Lemma 1 证明同 L 的候选互为退化错误、属同一逻辑等价陪集；K→∞ 时在聚类划分的候选空间内收敛）。
  - 效果：p=0.002 circuit-level noise、d∈{3,5,...,19}、K=24：LER 与 MWPM 之比从 d=3 的 1.0× 升至 d=19 的 ~2.1×（增 K 可继续缩小），显著优于 UF；repetition code 上 LER 距 MWPM 1.0–1.4×，与 BP+OSD（1.0–1.7×）相当，UF 落后 2.7–5.7×；X-biased 噪声下填补 UF 到 MWPM 差距的 ~94%。吞吐 1.88 M decodes/s（d=9）~29.8 M（d=3，p=0.001），为 Micro-Blossom 的 4–5×。可调性：LER(K)=LER_∞+A·K^{−α} 幂律收敛（α 从 d=3 的 1.98 降到 d=9 的 0.27），K*=2^{⌊(d+1)/2⌋} 捕获约 70% LER 收益。

## DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为三层协同的稀疏算法栈：(1) ESS（Epitopological Sparse Sampling）——受 epitopological learning 方法 [141] 启发，基于 Pearson 相关系数离线生成全局相关性注意力掩码，对像素阵列做 attention 引导的 50× 像素下采样（仅掩码有效坐标的像素参与 eventification），把"重要性高的像素保留、冗余像素抑制"的全局结构先验固化进像素阵列；(2) PAC（Patch Activation）——把稀疏事件按 16×16 分组为 patch，仅当 patch 内累计事件数超过阈值（实验取 2）才被激活读出，实现 in-sensor token pruning，最多减少 61% 的 host 端算法 MAC；(3) Robust ViT——conv stem（两层 depthwise-separable 卷积，输出 128 维）+ conv enhancement（两层 3×3 卷积替代标准 ViT 位置嵌入，引入跨 patch 局部交互）+ 3 个 transformer encoder（8 head、128 维多头自注意力）+ 平均池化 + 检测头（两层 FC + sigmoid），在 50× 压缩率（仅 2% 像素使能）下达到 0.5° 角误差。
  - 实验比较：算法精度对比两种 SOTA 事件相机 gaze 算法（一种在 EVBEYE [120] 上报告、一种用 OpenEDS 的 RitNet backbone [28] + 本文检测头）；稀疏采样方法对比三种：BlissCam 的随机稀疏采样 [47]、像素并行 DVS 稀疏方法 [67]、event transformer [98]，以及 event-density based denoising [45]；消融 PAC-only（无 ESS）。指标：角误差 AE（压缩率 1×–50× 扫描）、pixel error（ini-30 瞳孔追踪）。
- 硬件平台是什么，配置是什么。
  - 算法训练：NVIDIA A100 GPU（ESS 掩码离线生成约 2 分钟，用 27 个 subject 中随机 22 个生成、其余 5 个 unseen 验证）；ViT 训练 batch size 64、500 epochs。模型部署评估：STM32N6x7 处理器（16 nm，Arm Cortex-M55 + Neural-ART NPU，执行 MobileNet v2 基准 6 mJ/inference），LSQ INT8 量化、ONNX 导出、STM32Cube.AI 异构部署（卷积/线性层在 Neural-ART NPU、LayerNorm/Softmax 等非线性层在 Cortex-M55）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Robust ViT（conv stem + conv enhancement + 3 transformer encoder + detector head），无参数量/FLOPs 明确报告。数据集/bench：EVBEYE [17]（首个事件相机 gaze 数据集，27 subject、random saccades + smooth pursuits 两类眼动任务，40 英寸 1920×1080 屏、40 cm 阅读距离，剔除 stop/pause 标签与每次 saccade 前 15 个标签，训练/测试集不区分左右眼，AE 全数据集逐 inference 平均）；泛化验证 ini-30 [22]（DVXplorer 事件相机 640×480 镜架式采集、pupil 中心标注，5-fold 交叉验证，对比 SNN 方法 Retina）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确说明 DESSCam 代码开源；联网搜索未找到其公开仓库。基座组件开源：EVBEYE 数据集与代码（https://github.com/aangelopoulos/event_based_gaze_tracking）、LSQ、ONNX、STM32Cube.AI 工具链。
  - 算法 pipeline 执行例子（H×W 像素阵列、50× 压缩率的 gaze 估计）：
    ① 离线 ESS 掩码生成：把训练集 N 个 event frame 展平成 M×1 向量（M=H×W）并按列堆叠为 M×N Sample Matrix → 用 Pearson 公式 ρ_{X,Y}=cov(X,Y)/(σ_X·σ_Y) 对每对特征求相关，得 M×M Correlation Matrix → 逐行求和得 M×1 Feature Importance Matrix（每个像素与其他像素的相关性和，即该像素对 eye-tracking 任务的重要性）→ 以稀疏阈值 TH 二值化（>TH 置 1）得 binary Mask Matrix，写入像素阵列 SCtrl SRAM 使能对应像素的 eventification。TH 可调以控制下采样率（50× 压缩率 → 仅 2% 像素使能）。
    ② 事件累积：使能像素中 Vdiff 越过 VH/VL 阈值时生成 ON/OFF 事件，按 16×16 patch 聚合，事件数 > 2 的 patch 被 PAC 激活读出。
    ③ ViT 前向（每次 gaze 估计使用 12 个激活 patch）：conv stem 对稀疏事件帧做 depthwise-separable 卷积 → conv enhancement 两个 3×3 卷积替代位置嵌入、在 token 序列形成前做跨 patch 交互 → 输出特征图应用同一 ESS 掩码得到稀疏 token → 3 个 transformer encoder（8 head、128 维）捕捉长程依赖与全局相关性 → 平均池化 → 两层 FC + sigmoid 输出 (x_pred, y_pred)，AE = arccos(v_pred·v_gt / (|v_pred|·|v_gt|))，其中 v=(x, y, L0)、L0 为受试者到屏幕距离。
    ④ 部署：LSQ 量化为 INT8 → 导出 ONNX → STM32Cube.AI 按算子切分（卷积/线性 → Neural-ART NPU，非线性 → Cortex-M55）。
  - 效果：50× 压缩率下 AE 0.5°（同压缩率下 PAC-only 无 ESS 为 4.7°，AR/VR 不可接受）；压缩率 1×–50× 全程 AE 保持在 2° 以内；ini-30 上 50× 压缩率 pixel error 2.76±0.15，优于无稀疏的 Retina（3.24±0.79）；PAC 最多减少 61% host MAC。

## DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture（近似层次匹配：论文核心为异构 ASIC 硬件，本层取其算法级推理优化 Adaptive Expert Selection 与 INT8 量化策略）

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Adaptive Expert Selection（AES）动态在线专家选择算法。MoE 路由对 attention 输出做线性投影输出各专家分数，标准 top-k 选专家；AES 在 top-k 基础上做冲突感知替换：始终优先选分数最高的专家，其余专家按分数降序迭代加入，若与已选专家冲突（共享 in-NAND OU 输出端口，或缺少兼容 mask 模式）则在剩余未选专家中选分数最高且无冲突者替代；设阈值 T 约束路由分数偏差——仅当无冲突替代专家与冲突专家的分数差 < T 时才替换，否则保留原冲突专家（接受 FFN 额外 read cycle，保精度）。量化：INT8，权重按 2's complement 存 SLC cell，激活位并行按 bit 施加电压（8 die 位并行单 read cycle 完成 8-bit VMM）。实验比较：(1) 消融 Base（仅异构架构）/Mask（mask 设计无 AES）/AES（全系统）：Mask 至多 1.73×、AES 至多 1.52×、合计 1.95× 解码加速，AES 生效后 FFN 层恰好 3 cycles（Up/Gate/Down 各 1 cycle）且 L/M/H 三档速度趋于一致；(2) 冲突率：Mask-only 下 10.2%~93.5%，AES 降超一个数量级接近 0（DeepSeek/Qwen 效果最明显）；(3) 阈值 T 敏感性扫描：pairwise difference（专家对中至少一位与原始 top-k 不同的比例）、expert similarity = Σ_{i∈E_T∩E_k} w_i / Σ_{i∈E_T} w_i、四数据集端到端精度（inset 图）；(4) 端到端精度 vs 理想软件精度（注入 D2D 变差/ADC 噪声前后）。
- 硬件平台是什么，配置是什么。
  - 无真实硬件执行：精度用软件仿真（模拟 in-NAND 模拟域噪声注入后模型精度 vs 'Base' 理想软件精度）；解码速度用基于 SSDsim 的 cycle-accurate 模拟器（详见硬件架构层条目）。对比硬件基线：NVIDIA A100（312 TFLOPS FP16、80GB、1.94 TB/s HBM）、Jetson AGX Orin（64GB、30W 模式、TensorRT-LLM）、Cambricon-LLM、Lincoln、3D-AIMC。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B（INT8 47GB）、DeepSeekMoE、Qwen1.5-MoE、GRIN-MoE。数据集/bench：ARC-Challenge、PIQA、HellaSwag、WinoGrande（AES 精度评估）；MT-Bench（类 chatbot 场景解码速度）。专家选择场景：DIAMoND-L+Mixtral（单专家粒度无冲突）与 DIAMoND-H+DeepSeek/Qwen（全专家可容纳）为天然无冲突配置，AES 不必要。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：DIAMoND 论文未给出代码/模拟器链接，联网搜索未发现公开仓库（无法确认）。
  - AES 伪代码（单层 FFN，专家分数 s，阈值 T，目标选 k 个）：
    ① 路由：s ← router(h)，h 为 self-attention 输出；S ← {argmax s}（最高分专家必选）；
    ② for e in sort_desc(s)：若 e 与 S 中专家无冲突（mask 向量 AND 兼容，见硬件层 Mask Pattern RAM）且 |S|<k，则 S ← S ∪ {e}；
    ③ 若 e 冲突：在未选专家中找分数最高且无冲突者 e'；若 s_e − s_e' < T 则 S ← S ∪ {e'}，否则保留 e（该 token FFN 需额外 read cycle 处理冲突专家）；
    ④ 输出 S 作为该 token 实际激活专家集合。
    张量层面：每个专家 FFN = Up/Gate/Down 三个投影（Mixtral 隐维 4096、专家中间维 14336），每个投影按 OU（H=min{ρ_in,d_min}, W=min{ρ_out,d_min·QB}）切分为多个子矩阵，在 in-NAND 阵列上以 VMM 执行；AES 保证 k 个专家的三个矩阵可在同一 read cycle 并行执行（FFN 层固定 3 cycles）。冲突 = 两位专家占用同一 OU 输出端口，或同一 Expert Group 内可用 mask 模式无法区分二者（AND 门控后电流串扰产生错误输出）。
  - 效果：expert similarity > 0.9 时端到端精度仅微小波动；AES 使解码速度在 L/M/H 三档输入并行度（512/1024/2048）下趋于稳定。

## DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：DiTPA 软件框架（PyTorch 实现，基于 Dita 模型 [arXiv:2503.19757] 的 DiT 动作规划器），对轨迹/去噪迭代/模型三层冗余分别近似加速：S1 朝向条件动作预测（orientation-conditioned action prediction，相邻动作朝向差 ≤ 阈值时复用当前动作、跳过整次 DiT 推理，Skip_flag 交替强制全推理纠错）；S2 交替去噪特征复用（alternating denoising with feature reuse，按相邻去噪步 attention/FFN 特征相似度与跳过引入的 MSE loss 选择可跳过迭代，跳过步省略 attention/FFN 计算与权重重载，只做低代价残差噪声更新）；S3 校准多模态近似计算（calibrated multimodal approximate computing，按模态 lifespan 跳过不变的 vision/language token 计算，缓存冗余模态 K 特征校准 SoftMax 防 attention-shift、存 V 特征保持数据对齐，周期性重插完整去噪重置累积误差，并利用 action 模态 attention score 列稀疏跳过近零列对应计算）。
  - 实验比较：vs NVIDIA A40 GPU（实测推理时延、NVIDIA SMI 测功耗）与 EXION、Ditto 两款 SOTA DiT 加速器（自研 simulator 建模，见硬件架构层条目）；比较 action frequency、task execution time、energy efficiency、task success rate、轨迹质量（position/velocity instability、轨迹长度）；消融逐级叠加 S1→S2→S3；阈值 0°–4° 扫描成功率-加速权衡；泛化 LIBERO-Spatial/Object/Goal、CALVIN、SimplerEnv 与 π0.5、GR00T N1.5。
- 硬件平台是什么，配置是什么。
  - NVIDIA A40 GPU（37.4 TFLOPS 峰值，INT8 PTQ4DiT 量化）作为对比平台；DiTPA 目标硬件为 28nm/500MHz/0.88V ASIC（见硬件架构层条目）；机器人环境 LIBERO 仿真（默认 20Hz 控制、最多 520 环境步），真机 AMD VCU128 FPGA + LeRobot SO-101 机械臂。
- 模型是什么。数据集和bench分别是什么。
  - 模型：Dita（DiT 约 100M）为主，扩展 π0.5（DiT 约 300M，10 去噪步）、GR00T N1.5（约 500M，4 去噪步）；输出 7-DoF 动作 = 平移 (ΔX,ΔY,ΔZ) + 旋转 (ΔΦ,ΔΘ,ΔΨ) + 夹爪状态 g。
  - 数据集/bench：LIBERO-Long（10 长程任务 × 50 环境配置，客厅/厨房/书房，pick-and-place 与 turn-on-and-off）为主；LIBERO-Spatial/Object/Goal；CALVIN（5 连续任务）；SimplerEnv。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文给出 https://github.com/fengbintu/ISCA2026-DiTPA（本分析未实际克隆验证）。
  - S1 伪代码（论文 Algorithm 1）：若 Skip_flag=False → Act_nxt = 完整 DiT 推理；否则 Rad_rel = ROTATION_EXTRACT(Act_cur)，Rad_acc += Rad_rel，Deg_abs = 180/π·Rad_acc，Deg_sim = 旋转向量点积 / 模积，Deg_diff = 180/π·arccos(Deg_sim)；若 Deg_diff ≤ th（自动搜索得 2°）→ Act_nxt = Act_cur 复用，否则完整推理；最后 Skip_flag = !Skip_flag（交替预测与全推理，防止轨迹段边界误判累积）。
  - S2 计算流：先做一次完整去噪；对后续步按 attention/FFN 特征相似度阈值初筛候选跳过步、再评估跳过该步引入的 MSE loss（早期步不敏感）确定跳过集；被跳过步复用缓存的上一步 attention/FFN 特征，仅执行低代价残差噪声更新（噪声调度递推），省略 Q/K/V/FFN GEMM 与外部权重访问；后段低相似步保留全量计算保精度。
  - S3 计算流：按 lifespan 判定 language token（整任务不变）与 vision token（跨多去噪步不变）→ 跳过这些 token 的 Q/K/FFN 计算；SoftMax 用缓存的冗余模态 K 特征补全全局归一化分母（防注意力分布漂移），V 特征缓存对齐注意力聚合；每 20 个跳过迭代插入一次完整去噪重置误差；action 模态：逐列检测 attention score 近零列 → 跳过 SoftMax 输出零列与 V 对应行、进而跳过相应 V 投影计算。
  - 效果：动作推理次数 −42.28%（2° 阈值）、去噪迭代 −40%、冗余 token 计算 −91.74%；软件框架消去约 97% 冗余计算，但部署 GPU 仅获 2.3× 加速（理论 33×），需硬件协同；最终动作频率 217.65Hz、单任务 1.73s、成功率与 baseline 持平（≤2% 损失），推理时延较 GPU 降 96.32×。

## Distilling Magic States in the Bicycle Architecture

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：在 Bivariate Bicycle (BB) QLDPC 码上运行魔法态蒸馏（MSD）工厂的算法设计。(i) 用 triorthogonal matrix G∈{0,1}^{m×n} 刻画协议（前 k 个奇权重行对应 k 个输出逻辑 qubit，m−k 个偶权重行对应 ancilla/parity check；每列 c 定义作用于 S_c={r: G_rc=1} 的 m 体 Z 型 π/8 旋转 exp(iπ/8·Z^⊗S_c)）。执行形式为 Pauli-measurement-based Clifford 电路：m 个逻辑 |+⟩ 初始化 → 逐列消费 1 个噪声 |T⟩ 实现旋转 → X 基测量 m−k 个 parity check 并 postselect 全 |+⟩；相比 Bravyi-Haah 原构造省去 n-qubit 稳定子态制备与 T 门 unencoding，仅用 m 个逻辑 qubit。(ii) 协议级压缩（qubit recycling）：对 G 做列置换、块内行置换、F_2 行加法，最小化峰值并发活动 qubit 数 C(G)=max_j|W(j)|（行 i 在列 j "working" 当 j≥f_i 且（偶行 j≤ℓ_i 或 奇行永不释放）），49-to-1 从 13→7、51-to-3CS 从 18→9、64-to-2CCZ 从 17→10 qubit，使大协议塞进单 BB 块。
  - 实验比较：与 surface-code lattice-surgery 工厂（Litinski [46]，(15-to-1)SC(17,7,7) 与两级 (15-to-1)SC(11,5,5)+(15-to-1)SC(25,11,11) 等）及 Gidney 魔法态培养 MSC（Cultivation_SC→d, d=3/5）比较：物理 qubit 数、逻辑时间步 τ_i（含 discard 率）、时空体积（qubits×timesteps）、输出错误率 p_out（union bound 解析 + 密度矩阵仿真双口径）。
- 硬件平台是什么，配置是什么。
  - 纯经典仿真：MacBook Pro（10 核 CPU、32GB RAM），无量子硬件/GPU。仿真为逻辑层：每个逻辑 qubit 抽象为单 qubit 并施加给定逻辑错误率的 depolarizing 信道；物理级错误率直接借用自行车架构论文 [27] Table I 的既有仿真结果（gross/two-gross 的 automorphism、in-module、inter-module 测量错误率）。
- 模型是什么。数据集和bench分别是什么。
  - "模型" = 量子纠错码与蒸馏协议：gross [[144,12,12]] 与 two-gross [[288,12,18]] BB 码（LPU 分别 90/158 qubit）；协议 15-to-1、20-to-4、8-to-CCZ、49-to-1（输出 |T⟩）、51-to-3CS（|CS⟩）、64-to-2CCZ（|CCZ⟩）。bench = 噪声模型：物理错误率 p_phys∈{10⁻³,10⁻⁴}，输入魔法态误差 p_in（depolarizing 信道）、automorphism 误差 p_auto、inter-module 测量误差 p_inter（双 qubit depolarizing）、in-module 误差 p_intra（测量翻转 p_meas + 逻辑 depolarizing，λ=p_meas/p_intra∈{0.5,0.9}）。无传统数据集。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文 arXiv:2602.20546（ISCA 2026）；作者代码 https://github.com/kunliu7/bb-code-magic-state-distillation（描述 "Compilation of magic state distillation on BB codes"）：scripts/conj_path/conj_path_searcher_cnt.py（in-module 测量计数）、prj_msd/10mapping/（逻辑 qubit 映射）、prj_msd/20compressed/min_meas_cnt_comp_v3.py（协议压缩）、prj_msd/30min_aut/min_aut.py（最小化 native automorphism）。
  - 算法 pipeline 伪代码（triorthogonal 蒸馏，15-to-1 在 two-gross 块上，m=5）：
    ① init：全部物理 qubit 置 |+⟩，跑 1 轮 syndrome extraction → m 个逻辑 |+⟩；
    ② for c in 1..n（n=15 列）：取 1 个噪声 |T⟩（误差 p_in），经 injection gadget 在逻辑 qubit 上实现 exp(iπ/8·Z^⊗S_c)；测量结果为 1 时补 exp(iπ/4·Z^⊗S_c) 条件 Clifford 校正；
    ③ measure m−k=4 个偶行 qubit 于 X 基，postselect 全 |+⟩ 结果；
    ④ 成功 → 前 k=1 行输出蒸馏 |T⟩；p_out ≈ 35·p_in³ + O(p_L)。
    压缩伪代码：每行 i 求首/末 1 列 (f_i, ℓ_i)，工作集 W(j)={i: j≥f_i 且（偶行 j≤ℓ_i 或 奇行）}，目标 min C(G)=max_j|W(j)|；允许变换 = 列置换 + 块内行置换 + F_2 行加法（全局最优 NP-hard，化简到 cutwidth）；贪心聚类行起止 + 定向行加法求启发式解。
  - 效果（Table III，pivot 注入）：(15-to-1)Gross 378 qubit / τ=6122 / 体积 2.3×10⁶ / p_out≈1.3×10⁻⁶（union bound）vs (15-to-1)SC(17,7,7) 4620 qubit / τ=256 / 1.2×10⁶；(15-to-1)Two-gross 734 qubit / 11249 / 8.3×10⁶ / ≈1.0×10⁻⁸；(49-to-1)Two-gross 734 qubit / 70748 / 5.1×10⁷ / 2.0×10⁻¹¹（p_phys=10⁻³）与 ≤10⁻¹⁷（10⁻⁴）；两级 Cultivation_SC+(15-to-1)Two-gross 454+734 qubit 达 4.1×10⁻¹²（10⁻³）与 ≤10⁻¹⁷（10⁻⁴）。

## Don't Surrender to Low QPS/$: Fast and Cost-Efficient ANNS with TridentANN

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是面向 SSD 上十亿级向量 ANNS 的混合索引（hybrid noise-clusters index），三步构建（Algorithm 1-3）：(1) 对采样数据分层 KMeans 生成初始质心，每个向量分配给最近质心作 member、top-2~m 近质心作 candidate（解决边界向量漏检，避免 SPANN 式 RNG checking）；(2) 用 KMeans 递归拆分超过 list size r 的超大簇，子簇直接继承父簇 candidate（减少重复加载）；(3) 滤除 member 数低于阈值 n 的小簇（其向量归为 noise），用 candidate 补齐簇列表并重校准质心。最终索引 = 内存 HNSW 质心图（约 15M-60M 质心）+ 内存 SPTAG-BKT noise 索引（约 100M 向量）+ SSD 上的簇列表。实验比较 DiskANN、SPANN、PipeANN、FusionANNS（REF 原文 + REP 乐观复现），指标为 QPS、latency、Recall@10、P99.9、QPS/$、QPS/watt；8 SSD 配置下 1.8-3.4× QPS、延迟降为 21-70%。
- 硬件平台是什么，配置是什么。
  AMD EPYC 7453 28 核 CPU；8×32 GiB DDR4（实际内存占用 <32 GiB）；8× 1-TiB Samsung 980 Pro PCIe-Gen4 NVMe SSD；2× NVIDIA A2000-6GB GPU。Ubuntu 22.04 + Linux 6.8 + CUDA 12.1 + GCC 11.4。10B 规模建索引另用 256 GB DRAM + NVIDIA A6000-48GB。
- 模型是什么。数据集和bench分别是什么。
  无神经网络模型（纯向量检索系统）。数据集：SIFT1B（128-dim uint8 图像描述子）、SPACEV1B（100-dim int8 文档向量）、GloVe（1.2M 向量、100-dim）、NYTimes（0.3M、256-dim，后两者研究数据分布偏斜）；距离度量 L2。Benchmark：BigANN（Recall@10=90% 为合格线），另测 recall 90-98%、top-1 到 top-1000。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文正文未给出 TRIDENTANN 自身代码的开源链接（联网检索仅在作者主页看到 Code 入口，具体仓库地址未能确认）；实现基于 Faiss 1.12（hierarchical KMeans）。算法 pipeline 伪代码：
  ```
  # Algorithm 1: 建簇（member/candidate）
  C_init, I = H-K(sampling(X), |sampling(X)|/r)   # 分层 KMeans 生成初始质心
  for x in X:
      [c1..cm] = KNN(x, C_init, m)                # m=3 (SIFT1B)
      I[c1] += x                                  # member
      E[c2..cm] += x                              # candidate（边界冗余）
  # Algorithm 2: 平衡簇
  while C_init != ∅:
      若 |I[c]| > r: 拆为 ceil(|I[c]|/r) 个子簇(KMeans)，子簇继承 E[c]
  # Algorithm 3: 分离 noise 并成列表
  for c: 若 |I[c]| > n 且候选足够: 从 E[c] 选离质心最近向量补齐列表至 r，重校准质心
  N = X - 所有簇列表并集（noise 向量）
  建内存 HNSW(质心 C) + SPTAG-BKT(N)，簇列表 R 落 SSD
  ```
  查询：CPU 查质心 HNSW 定位最近簇 → 簇列表经 GPU-SSD P2P 直通加载 → GPU cuBLAS 算距离 → CPU partial_sort 排序并合并 noise 结果（详见 kernel调度条目）。建索引：CPU-only 版 2-2.5 天、GPU 加速 KMeans 版 20-22 小时（对比 SPANN 4-5 天、DiskANN/PipeANN 2.5-3 天）；10B 规模 1 周内。

## ECHO: Efficient Head-Orientation-Guided Real-Time Sound Spatialization for Virtual Reality

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现是 ECHO 的四项算法优化，目标是压缩 VR 中 motion-to-sound 空间声化（SS）流水线的端到端延迟：①低精度位姿估计（low-precision pose estimation）：在 ORB-SLAM3 的 LM Tracking 迭代优化中，把大批量矩阵-向量乘法 x_i^c = R·x_i + t 从 FP64 降到混合精度——旋转矩阵 R（元素 ∈[-1,1]）用缩放因子 8 量化成 4-bit 有符号整数 Q_INT4(r)=clamp[round(8r), -8, 7]，3D 地图点 x_i 用 FP8 E4M3 格式 Q_FP8（范围 -448~+448，覆盖典型室内场景 896m 动态范围），即 x_i^c = Q_INT4(R)·Q_FP8(x_i)/8 + t；缩放因子取 8 是因为除 8 可通过浮点指数位调整实现（硬件友好），而鱼眼投影 π(·)（Kannala-Brandt 模型）及其 Jacobian 保持 FP32。②量化感知点过滤（quantization-aware point filtering）：优化前先按 FP8 量化误差 E1=||x_i - Q_FP8(x_i)||>α 剔除低质量 3D 点，再做一次性稳定性检查 E2^q=||u_i - π(Q_INT4(R)·Q_FP8(x_i)/8 + t)||²>β 丢弃不稳定 2D-3D 对应，最后用选择性采样（selective sampling）：当被 E2^q 拒绝的比例低于阈值 r1 时推断位姿已准，随机丢弃 r2 比例的剩余对应再进入 Gauss-Newton 优化，平均丢弃约 75% 的点。③IMU 高频位姿估计：轻量 RNN 以 IMU 数据（1000Hz）+ 最新 SLAM 优化位姿/速度/传感器偏置为输入，输出 100Hz 的 7D 位姿（3D 平移 + 4D 四元数），权重 per-channel INT4、激活 FP8，量化感知训练（QAT）保持精度，用于压缩两次 SLAM 帧间的 T_IN。④鲁棒声学注视（robust acoustic foveation）：按 MAA（最小可听角度，正前方 ~3° 到侧向 ~40° 单调递增）与距离感知阈值（标准差超过源距离 20%），把 3D 房间沿高度分块、层内按方位角+径向把声源聚类为单一虚拟源（质心），并在位姿误差存在时用保守下界 θ_eff = θ - Δθ_r - Δθ_t 收紧聚类阈值，减少 T_R1（声传播+BRIR 生成）与 T_R2（auralization 卷积）的活跃声源数。实验比较的是九种配置（Table VII，'Jet'=Jetson、'Full'=完整渲染、'Foveated'=声学注视、'ECHO'=算法+加速器）的 motion-to-sound 延迟，以及 ECHO 与 ORB-SLAM3 / VINS-Fusion / HybVIO / OKVIS 的位姿精度（ATE/RRE），并与 eSLAM / HcveACC / FSLAM 对比每帧跟踪延迟（TUM VI：11.0ms vs 23.3/20.0/20.1ms）。
- 硬件平台是什么，配置是什么。
  - Nvidia Jetson Orin NX 16GB（CPU 跑 ORB-SLAM3 位姿估计与 Pyroomacoustics 全/注视音频渲染，GPU 跑 gpuRIR 注视音频渲染），为近期 VR 性能研究常用平台；ECHO 加速器为 Verilog 实现、Synopsys Design Compiler 45nm（Nangate）综合、1GHz、CACTI 建模片上 buffer、DeepScaleTool 缩放到 22nm；用户研究用 Meta Quest Pro HMD。
- 模型是什么。数据集和bench分别是什么。
  - 模型：位姿估计基线为 ORB-SLAM3（visual-inertial SLAM，tracking/local mapping/loop closure 三线程）、VINS-Fusion、HybVIO、OKVIS；ECHO 的 IO 模式用自训练轻量 RNN（每数据集分别 QAT 训练）；音频渲染用 ISM 图像源法实现 Pyroomacoustics（CPU）与 gpuRIR（GPU）。数据集：AEA（Project Aria Everyday Activities，480×640 单目灰度 10Hz + IMU 1000Hz，4 条测试序列 + 10 条随机序列训练/验证，监测 3 条序列共约 6000 帧做延迟剖析）与 TUM VI 全部 6 条室内序列（每条随机 20% 测试、80% 训练/验证）；声学场景 5×5×2.7 / 50×50×5 / 200×200×10 m（客厅/会议室/展厅），混响 30dB 衰减 0.4/1.5/1.7s，声源数 8-256，位置用均匀随机分布或 Poisson Cluster Process；用户研究语音来自 LibriSpeech 文本 + edge-tts 合成。bench/评估工具：evo toolkit（https://github.com/MichaelGrupp/evo），指标 ATE（m，绝对平移误差 RMSE）与 RRE（°，100ms 间隔相对旋转误差 RMSE）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供 ECHO 算法/加速器代码链接，联网搜索未找到官方公开仓库（ISCA 2026 论文，作者 NYU Haiyu Wang / Tianhua Xia / Sai Qian Zhang；搜索到的 github.com/yucongzh/ECHO 为无关的机器音频表示学习项目，非本论文）；依赖组件均开源：ORB-SLAM3（https://github.com/UZ-SLAMLab/ORB_SLAM3）、Pyroomacoustics（https://github.com/LCAV/pyroomacoustics）、gpuRIR（https://github.com/DavidDiazGuerra/gpuRIR）、evo（https://github.com/MichaelGrupp/evo）、AEA（facebookresearch projectaria_tools）、TUM VI、edge-tts（https://github.com/rany2/edge-tts）。
  - 算法pipeline 执行例子（一个 LM Tracking 帧 + 一次 IO 高频位姿 + 一次声学注视聚类）：输入当前帧 ORB 关键点 u_i 与 3D 地图点 x_i、当前位姿 R,t → ①点过滤：对每个 x_i 计算 E1=||x_i-Q_FP8(x_i)||，E1>α=0.1 的点直接剔除；对剩余点按 x_i^c=Q_INT4(R)·Q_FP8(x_i)/8+t 投影并算 E2^q=||u_i-π(x_i^c)||²，E2^q>β=120 的对应丢弃；若被拒比例<r1=5% 则随机丢弃 r2=40% 剩余对应 → ②迭代优化：保留对应进入 Gauss-Newton，每次迭代在大批量矩阵-向量乘法中复用 Q_INT4(R)（缩放因子 8 的整数）与 Q_FP8(x_i)（E4M3），乘法在 INT4×FP8 混合精度下完成、除以 8 靠指数位-3 实现，鱼眼投影 π 与 Jacobian 全程 FP32（SFU/PJ 硬件加速）→ ③输出优化后 6DoF 位姿。IO 模式：IMU 测量（1000Hz）→ RNN（INT4 权重×FP8 激活，QAT 训练）输出 100Hz 7D 位姿（平移+四元数）填补 MI 帧间隔。声学注视：位姿 + 声源集合 → 沿高度分块 → 层内算每个声源的方位角 θ 与距离 r → 方位差<MAA(θ_eff)（θ_eff=θ-Δθ_r-Δθ_t 保守下界）的源并入角向组 → 组内距离差<最远源距离 20% 的合并 → 每簇质心生成单一虚拟源 → 进入 BRIR 生成与 auralization。端到端效果：ECHO 相对 Jet ORB+Full GPU baseline 在 256 声源下 TUM VI 2.79×、AEA 2.91× 延迟降低；位姿 ATE/RRE 与 ORB-SLAM3 相当（Hybrid 平均 0.033m/1.014°）。

## ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 近似匹配（论文本体为硬件架构，此处仅覆盖其 SNN 推理算法侧——稀疏/弹性推理算法）：实现为三部分算法 pipeline：(1) ST-BIF（双极积分-激发-脉冲追踪）神经元——数学上与量化 ReLU 在特定条件下等价，三元脉冲 {-1,0,1}，三步：积分（膜电位加权重积）、激发（决策函数 Θ 含 spike tracer 记忆）、膜更新（soft reset + tracer 累积）；(2) 弹性推理（elastic inference）+ 置信度早停——输出逐步涌现，分类用最大类概率、检测用 objectness score 作置信度，超过阈值提前终止剩余时间步；(3) mini-batch spiking Gustavson-product 数据流——利用 spike 稀疏性，BAER 行对齐成 mini-batch，row-wise 累加把稀疏 spike 计算转成低访存行累加。实验比较：ANN vs QANN vs SNN vs SNN+早停 的准确率与延迟（ResNet18/34/50、ViT-S on ImageNet，表 VII），早停平均 21.9% 延迟缩减且 <0.2% 精度损失（激进阈值 30.6% / <3.3%）；COCO2017 YOLOv2 mismatch rate vs 置信度阈值（sweet point 0.2：match 率 94.9%、45.4% geomean 延迟缩减）；显著性分析（目标面积比 vs 延迟，图 19）；消融（图 22：Gustavson -A 节能 3.1×/1.6×，spine/token 流水 -B 提速 6.7×/15.2×，BAER -C 延迟增益小）。
- 硬件平台是什么，配置是什么。
  - 算法评估：8× NVIDIA 4090 GPU 服务器（Ubuntu 22.04.3 LTS、CUDA 12.2、PyTorch 2.4.1、Anaconda 24.5.0、GCC 11.4.0）；模拟器评估：AMD EPYC 9334 32 核服务器。磁盘约 20GB，环境准备约 30 分钟。
- 模型是什么。数据集和bench分别是什么。
  - 模型：VGG16、ResNet18/34/50/101、ViT Small（分类）、YOLOv2（ResNet34 backbone，检测）。全部 4-bit 量化权重、时间步 T.S.=32，SNN 按 SpikeZIP-TF 生成（ST-BIF 与 QANN 精度一致）。数据集/bench（表 II）：W1 VGG16-CIFAR10、W2 VGG16-CIFAR100、W3 VGG16-CIFAR10-DVS、W4 ResNet18-ImageNet、W5 ResNet34-ImageNet、W6 ResNet50-ImageNet、W7 ViT-S-ImageNet、W8 YOLOv2-COCO2017/VOC2007、W9 ResNet101-ImageNet。指标：top-1 准确率、SOP（synaptic operation）、FCR（首正确响应）延迟、早停缩减、mismatch rate。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：是。GitHub https://github.com/Intelligent-Computing-Research-Group/ELSA（ELSA_Algorithm 目录），Zenodo 归档 https://zenodo.org/records/19449728。
  - 算法 pipeline 执行例子（ResNet50 一次推理、单 spine，权重 W 4-bit）：① 逐时间步 t=1..32：输入 spike x∈{-1,0,1} 经 im2col 展开，MM-sc 用 mini-batch Gustavson——同一膜行的 spike 捆绑成 mini-batch → 读出对应权重行 → 加法树并行累加进膜电位 V_t = V_{t-1} + Σ_{i} x_{i,t}·w_i（负 spike 权重取二补码）；② 激发 y_t = Θ(V_t, V_thr, S_t)：V_t≥V_thr 且 S_t<S_max → +1；V_t<0 且 S_t>S_min → −1；否则 0；③ 更新 V_t = V_t − y_t·V_thr（soft reset）、S_t = S_{t-1} + y_t（spike tracer 累积）；④ 每时间步末分类头取最大类概率 p_max，若 p_max ≥ 阈值（如 0.55）提前终止全部剩余时间步；⑤ 检测（YOLOv2）以 objectness score 为置信度，目标面积比大的更早终止（VOC2007 延迟 2.38→1.88 ms，COCO2017 1.73→1.64 ms）。效果：SNN 精度 = QANN（ResNet50 ImageNet 75.60%、ViT-S 79.07%），早停平均延迟减 21.9%（<0.2% 精度损失），FCR 可比稳定态输出早 82%（COCO 检测）。


## ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现是 ENEC（Efficient NPU-Enhanced Compression），一个针对 Ascend NPU 架构定制的 AI 模型权重无损压缩算法（块式定长熵编码类）。算法核心：(1) 分量分离——把浮点权重（BF16/FP16/FP32）拆成{指数 E, 符号 S, 尾数 M}，只压缩指数（Observation 1：BF16 指数熵仅 2.58 bits 高度可压，符号/尾数近均匀分布 7.97 bits 不可压），符号尾数直接存储；(2) 频率统计与参数搜索——每 8192 元素块统计指数频率，用 V-E 节三阶段搜索确定最优参数 (b*,n*,m*,L*)（对多数张量 BF16 收敛到 (b≈122/123, n=6, m=3, L=16)，见 Table IV）；(3) 三个 NPU 特定优化：bit-width quantization with hierarchical halving bit-packing（位宽量化+分层对半打包，Algorithm 2，把 reduction max/乘除换成 bitwise OR/shift 的 lane folding）、vectorized branch-free integer transformation（向量化无分支整数变换 y=(2^n-x+b)%2^n，用线性映射替代 gather 查表，Observation 5：指数值-频率排名呈负线性关系）、intra-segment dependency decoupled scan (IDD-Scan)（段内依赖解耦前缀和，绕开 AscendC 32 字节段内禁止直接 SIMD 的约束）。实验比较：①压缩比——vs CPU 的 ZipNN、GPU 的 DietGPU（Diet_ANS/Diet_Float）与 nvCOMP（NV_Zstd/NV_Deflate/NV_GDeflate/NV_ANS/NV_Bitcomp）、NPU 的 HANS，10 个模型全部 bit-identical 无损重建（Table II）；②压缩/解压吞吐（Figure 9，对数坐标）；③端到端推理 TTFT/TPOT（集成进 HuggingFace Transformers，逐层解压并与上一层 forward 重叠，Figure 10）；④参数消融（块大小、L、n/m、b）与参数鲁棒性/迁移性（用 DeepSeek 参数迁移到其他模型，Table V）；⑤消融 V0→V1→V2→V3（Figure 13）。
- 硬件平台是什么，配置是什么。
  - NPU：华为 Ascend 910B2，24 AI Cube（AIC）+ 48 AI Vector（AIV）单元（vector-to-cube 2:1）；Host：HiSilicon Kunpeng-920 CPU，Ubuntu 22.04.5 LTS，NPU 驱动 25.0.rc1.1；CANN toolkit 8.2.RC1.alpha002 + operator kernel package 同版本；AscendC（C++17）编程框架。每 AI core 的 AIV 用 Unified Buffer（UB）约 192KB；数据块默认 16,384 元素（32,768 会超 UB 192KB），每线程专用 32KB 打包 buffer（8,192 个 32-bit lane）。
  - GPU/CPU 对照：NVIDIA A800（Intel Xeon 8358P CPU，CUDA 12.6，与 NPU 性能相近的平台）；ENEC-CPU 用 AVX2 + BMI2 PEXT，ENEC-GPU 用 CUB 库前缀和。
- 模型是什么。数据集和bench分别是什么。
  - 10 个 HuggingFace 开源模型权重：BF16——Falcon-7B(14.40GB)、Qwen3-8B(16.38GB)、deepseek-llm-7b-base(13.82GB)、Qwen3-32B(65.60GB)、Llama-3.1-8B-Instruct(16.06GB)；FP16——CapybaraHermes-2.5-Mistral-7B(14.50GB)、stable-video-diffusion-img2vid-fp16(4.27GB)；FP32——OLMo-1B-hf(5.10GB)、bert-base-uncased(0.40GB)、wav2vec2-large-xlsr-53-english(1.20GB)。端到端推理模型：Qwen3-32B 与 Falcon-40B（均 BF16）。参数迁移实验：DeepSeek-V3 搜索参数 → Falcon-7B/Qwen3-8B/Qwen3-32B/Llama-3.1-8B。
  - 指标/bench：压缩比 CR=压缩前总大小/压缩后总大小；压缩吞吐、解压吞吐（GB/s，device-side kernel 时间，不含 PCIe）；端到端 TTFT（Time-To-First-Token）与 TPOT（Time-Per-Output-Token），固定输入/输出长度，10 次 warm-up + 50 次 test 取平均，baseline 为无压缩+部分 CPU offload。
  - 关键结果：压缩吞吐 263–523 GB/s、解压吞吐 188–336 GB/s；BF16 压缩吞吐平均 372 GB/s = ZipNN 的 987×、nvCOMP:Bitcomp 的 1.39×、HANS 的 1.36×；FP32 达 523 GB/s（2.47× HANS）；解压最高 4.22× NV-Bitcomp、2.11× HANS；vs DietGPU 吞吐 3.43× 高、vs nvCOMP 压缩比 1.12× 好；端到端 TTFT 最高 6.3×（Falcon-40B）/ 4.1×（Qwen3-32B）、TPOT 4.9×/3.9×；参数迁移零损失（Llama 仅 5% 比降）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：是。主仓库 https://github.com/hpdps-group/ENEC（csrc/ NPU kernel + python/ 数据处理/参数搜索/测试/msprof profiling，脚本 build_csrc.sh/data_prepare.sh/compressor_test.sh）；ISCA 2026 AE 仓库 https://github.com/jinwuyang/ENEC_ISCA_AE（BSD-3 许可，Ubuntu 22.04 aarch64 + Python 3.9 + torch 2.5.1 + torch_npu 2.5.1.post3 + ATB 8.0.0）。论文扩展了 LC framework（https://github.com/burtscher/LC-framework）的 HCLOG（支持不同子块数量的变体）用于最优组件搜索。
  - 算法 pipeline 执行例子（BF16 权重一个 8,192 元素数据块，参数 (b=123,n=6,m=3,L=16)）：① Split：把 BF16 拆成 8-bit 指数 E、1-bit 符号 S、7-bit 尾数 M，S/M 直接写压缩流；② 频率统计+参数搜索：统计块内指数频率直方图，V-E 三阶段搜索选 (b*,n*,m*,L*)（式(1) n=max(⌊log2(b-l)⌋+1, ⌈log2(h-b)⌉)+1；式(2) y=(2^n-x+b)%2^n；式(3) D=Σp(x)·y 最小化；式(4) B_exp=1/L+n+(m-n)·p(m)^L 联合搜 m,L）；③ 分支无关变换：x=125 → y=(2^6-125+123)%64 = (126)%64=62，x=122 → y=(64-122+123)%64=65%64=1（即 b-x=123-125=-2 经模 2^(n+1)=64 环绕成 62，正数 1 不变），实现=向量加（减 b）+向量乘（×(-1)）+移位取模；④ 分组+阈值：每 L=16 元素一组，组内最大值位宽 ≤m=3 → 整组 3 位存；>m → 整组 n=6 位存，1-bit bit mask 记录组类型；⑤ 分层对半打包（Algorithm 2）：N 元素 a-bit 值（N=2^k），迭代 data[i] |= data[i+length]<<width、width×2（lane folding，OR+shift 合并两元素到同一 lane），位宽达 8 边界时拆出低 8 位成字节、溢出位收集成子块递归；最后 byte 归一化+偶数对齐（16-bit aligned）→ 输出压缩流。解压：bit mask→{0,1} 整数→IDD-Scan 前缀和（8×16 half 张量：转置成 16×8 把行内依赖转成列方向 → log2(16)=4 步向量加法列前缀和 → 转置回；再层级行扫描 log2(8)=3 步得行偏移 → exclusive 偏移广播加回）→ 按偏移逆 gather 取低位 → OR 上高 (n-m) 位还原 y → 逆变换 x=(y+b-2^n)%2^n 还原指数 → Combine(E,S,M) 位级重建原始 BF16 权重。效果：Qwen3-32B BF16 CR=1.35、压缩 366.3 GB/s、解压 217.1 GB/s；Falcon-40B TTFT 6.3× 提速。

## EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现是 EVA 的 VQ-GEMM（codebook-driven GEMM）重构算法，配合 AQLM 加法向量量化（Additive Vector Quantization）。核心：把 LLM decoding 的 GEMV 重构为 GEMM——不再用权重索引（WI）从权重码本（WC）现场重建量化权重（去量化），而是直接把输入向量与权重码本做点积得到中间"输出码本"（OC），再以 WI 从 OC 做冲突无关查找并累加得到最终输出。实验比较：①模型精度——EVA(AQLM, d=8, n=8, 2/3/4-bit) vs SA(QSERVE INT8)、ANT(8-bit)、FIGNA(AWQ INT4/INT2)、FIGLUT(BCQ 4-bit)、FP16 的 WikiText-2 perplexity（L-2 7B/13B：EVA-4bit 5.43/4.76 为最优；2-bit 时 weight-only 全崩，EVA-2bit 6.69 vs FIGNA-INT2 2.2e5）；②下游准确率——EVA vs LLM.265 FB/VB 在 PIQA/COPA/ARC-E/ARC-C/Winogrande（2-bit 平均高 19pp，Table VI）；③MoE 模型——Mixtral-8x7B、Qwen3-30B-A3B 上 AQLM/GPTVQ vs AWQ/GPTQ 在 ARC-C/ARC-E/PIQA/BoolQ/Winogrande（2-bit AQLM-2×8 只掉 5.3pp，GPTQ 掉 32.7pp，Table VII）；④不同 VQ 配置（d/n/C 与码本数）的延迟与精度权衡（Table III）。
- 硬件平台是什么，配置是什么。
  - 算法精度评估：NVIDIA GPU ≥24GB VRAM（A100-80GB 推荐）、CUDA 12.x、约 100GB 磁盘（量化 checkpoint），Linux（Ubuntu 20.04+）、Python 3.11、aqlm[gpu,cpu]>=1.1.6、torch>=2.3.0、accelerate>=0.29.3、lm-evaluation-harness；AQLM 可能 JIT 编译 CUDA/C++ 扩展（GCC/G++ 11.x 推荐，11.4.0 已测）。硬件仿真器：任意 x86-64 CPU、≥16GB RAM、约 10GB 磁盘，无需 GPU。
- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA 1/2/3（主要 LLaMA-2-7B/13B）、Mixtral-8x7B、Qwen3-30B-A3B（MoE）；算法评估用预训练 AQLM 量化 checkpoint（Hugging Face dbw6/eva collection）。数据集/bench：perplexity=WikiText-2；下游准确率=PIQA、COPA、ARC-Easy、ARC-Challenge、BoolQ、Winogrande；端到端=Dolly、Arxiv Summarization、GSM8K（输入/输出长度见 Table IX）；硬件仿真跑每个模型第一个 Transformer block。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：是。https://github.com/dbw6/Eva.git（MIT License），Zenodo 归档 https://doi.org/10.5281/zenodo.19433707，预训练权重与数据集 https://huggingface.co/collections/dbw6/eva。仓库分 simulator/（硬件仿真）与 algorithm/（eval_ppl.py、lmeval.py，评估脚本迁移自 AQLM 仓库）。安装：conda create -n eva python=3.11 -y && conda activate eva && pip install -e . && pip install "aqlm[gpu,cpu]>=1.1.6"。精度评估运行 scripts/run_algorithm_parallel.sh（10 个评估：4 perplexity + 6 下游准确率）。
  - 算法pipeline 执行例子（LLaMA-2-7B 一个 token 的 FC 层，AQLM-2×8，d=8, n=8, C=2, q=2bit）：权重 W∈R^{4096×4096} 离线被 AQLM 量化成 C 个码本 B_c∈R^{8×256} 与索引矩阵 I_c∈[0,256)^{512×4096}（每 8 个连续权重为一个向量，2^8=256 个 centroid，2 个码本相加得 2-bit 平均精度）。在线解码：输入激活 x∈R^{1×4096}（FP16）→ reshape 成 X∈R^{512×8}（每 8 个连续元素一组）→ 对每个码本做紧凑 GEMM：O_c = X·B_c ∈ R^{512×256}（输出码本 OC，即每个输入向量与每个 centroid 的点积）→ 用索引从 OC 查找：ŷ = Σ_{c=1..C} Lookup(O_c, I_c)，每个输出通道按 I_c 逐行取 OC 元素（OC 与 WI 共享高度 V=K/d，无 bank 冲突）→ add-only 累加得最终 y∈R^{1×4096}。相比常规 VQ（查 WC 重建权重再 GEMV），计算量从 1×N×K MAC 降到 K×2^n（N=4096、2^n=256 时约 16× 少），查找带宽从每次读 d 个 FP16 降到 1 个 FP16，且访存全部规则合并化。

## FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 FEnc²（FEnc2），一个统一的、自动化的、基于 fragment（块）的 CKKS 密文打包/编码算法框架，加速 FHE 私密 CNN 推理，含两个组件：(1) Conv-aware Encoding——把 4D 特征张量 X∈R^{BS×C×H×W}（batch×channel×H×W）按 S×S 块做参数化分解（Algorithm 1：Step1 零填充到方形 M×M；Step2 划分 m×m 个 S×S 块；Step3 对每个块内坐标 (u,v)，跨所有块收集该坐标元素并按槽映射展平打包；Step4 加密成完全装满的密文 ct_(u,v)），用块分解同时解耦相邻像素（spatial/intra-channel）与跨通道（channel）依赖；最优块大小 S* 由凸/解析旋转代价模型导出：S* = ceil((K²·N_in/(α·N_out))^(1/4))（Theorem 1，内旋转代价随 1/S² 递减、外旋转随 S²α 递增，两者相等时总旋转最小），内旋转复杂度从 O(K²) 降到 O(K)。(2) Arch-aware Ct Compression (AAC)——针对通道/特征缩减层（MobileNet/SqueezeNet/ResNet 的 1×1 卷积）造成的槽稀疏，用 mask-rotate-add 轻量合并把有效通道压实成满装密文（例：瓶颈 8→2→8 通道，2 通道密文压实后扩展层只需 1 个密文而非 4 个 25% 利用率的稀疏密文），且通过尺度共享技巧（Δ₁·Δ₂=Δ 的两个 PMult 后只做一次 rescale）不增加乘法深度。
  - 实验比较：baseline 为 CHET（Toeplitz 单通道卷积）、HELayers（block tiling + 多图打包）、Batchwise+、Hyena+（on-the-fly 重旋转 + 邻居保持打包）、Orion（多通道打包 + BSGS，SOTA）。指标：端到端延迟（sec/image，GPU/CPU）、内存占用、旋转数、key-switching 数、NTT/INTT 数、密文数、slot 利用率、kernel 调用数、GPU 内存传输大小/次数。
- 硬件平台是什么，配置是什么。
  - AMD Threadripper 3975WX @ 3.5GHz CPU，256GB 8-Channel RAM @ 3200MT/s；NVIDIA RTX A6000 GPU 48GB。软件栈：GPU 后端用 Liberate-FHE（开源 GPU 优化的 HE 框架，https://github.com/Desilo/liberate-fhe）；深层 CNN 的 bootstrapping 用 NEXUS 的 GPU 优化实现（每次 bootstrap 消耗 14 个密文 level）。ReLU 用多项式近似 ax²+bx+c 替换；FC 层用对角线矩阵乘法 + BSGS（同 Orion）。固定 scale Δ=2^40（40 bits），模数 Q 保证安全级别 λ≥128 bits。CPU 平台内存 256GB 支持更大 batch。
- 模型是什么。数据集和bench分别是什么。
  - 模型与数据集（Table IV）：LeNet（MNIST，N=2^15，98.95%）、VGG5（CIFAR-10，N=2^15，86.32%）、SqueezeNet（CIFAR-10，N=2^16，81.5%）、ResNet18（ImageNet，N=2^16，66.8%）、MobileNet（ImageNet，N=2^16，72.0%）。SqueezeNet/ResNet18/MobileNet 需 bootstrapping（ResNet18/MobileNet 各加 191s/105s），LeNet/VGG5 不需要。GPU 48GB 下最大 batch：LeNet/VGG5=512、SqueezeNet=256、ResNet18/MobileNet=16。Transformer 扩展实验：BERT-base 单 encoder block（E=768，n=128，γ≈6），A6000 GPU，N=2^16、log Q=1768，S=4 时相对 Orion 6.74× 加速。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：FEnc² 本身未找到公开代码仓库（arXiv 2606.16359，ISCA 2026 待发表，联网搜索未见官方 repo）；底层 HE 框架 Liberate-FHE 开源（https://github.com/Desilo/liberate-fhe）。FEnc² 与 Orion 等 baseline 均运行于其上的 GPU HE 执行层。
  - 算法 pipeline 执行例子（Conv-aware Encoding，BS=1，输入 1×16×4×4、卷积 (16,16,3,1)，N/2=16 slots 全满装）：① 正方形填充 M=max(pad(H),pad(W))=4；② 块划分：S=2 → m=2，每 feature map 切成 2×2 个 2×2 块；③ fragment 打包：对块内坐标 (u,v)∈{(0,0),(0,1),(1,0),(1,1)}，从 m²=4 个块收集同坐标元素组成 X_(u,v)∈R^{C×BS×m²}=R^{16×1×4}，按槽映射 X_ijk^(u,v)→slot l（i=⌈l/(BS·m²)⌉，j=l mod BS，k=⌈l/BS⌉ mod m²）展平为 16 槽向量；④ 加密得 4 个满装密文 ct_(u,v)。旋转代价：K=3>S=2（CASE 1，相邻块重叠），每密文内旋转 (⌈K/S⌉²−1)=(⌈3/2⌉²−1)=3 次；外旋转 N_out/α×(αS²/BS−1)。S=1 退化为 row-major/Orion（无外旋转、内旋转最大），S=M=4 退化为 pixel-wise/CryptoNets（无内旋转、外旋转最大），S=2 是平衡内/外旋转的最优点。AAC 例子（瓶颈 8→2→8 通道）：1×1 缩减层后密文只剩 2 个有效通道（25% 利用率）→ 施加 0/1 掩码的 rot-mask-add（Δ₁·Δ₂=Δ 共享尺度，只做一次 rescale、不耗额外 level）把 2 通道压实到 1 个满装密文 → 扩展层 1 个密文生成 8 输出通道（无 AAC 需 4 个稀疏密文、4× HE 计算）。效果：相对 HELayers 旋转减少 67%-94%、keyswitch 减少 80%-93%、NTT/iNTT 减少 89%-94%、密文数减少 78%-94%、slot 利用率提升到 100%；LeNet+SqueezeNet 上 kernel 调用减少 88%、GPU 内存传输量减少 87.8%、传输次数减少 87.1%；相对 Orion 的加速比：LeNet 最高 228.83×（GPU）/226.06×（CPU），MobileNet 4.55×（GPU）/9.43×（CPU），内存减少最高 98.49%（CPU）/75.6%（GPU）。

## Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现是两类把 NAND flash 当作"主动数据层（active tier）"的新算法设计（RQ4 两个案例研究，全部为建模+模拟评估，无实物硬件）：①SSD-resident 阻塞 Cuckoo 哈希 KV 存储（blocked-Cuckoo hash table on SSD，消除全部 DRAM 驻留索引/元数据，不丢条目，桶溢出用重定位而非丢弃），对比内存 KV（Redis/FASTER/MICA）与混合 DRAM/SSD 引擎（RocksDB/WiredTiger/Bw-tree，这些仍保留大量 DRAM 驻留索引/过滤器/块目录）；②SSD-resident 两阶段渐进式 ANN 搜索（reduced-dimension 向量首筛 + full-dimension 向量精排），对比 DiskANN/Filtered-DiskANN 等以牺牲搜索质量为代价迁就常规 SSD 低 IOPS 的 SSD 驻留系统。实验比较：①KV 可达吞吐（Mops/s）vs DRAM 容量、GET:PUT 比（100:0/90:10/70:30/50:50）、强弱局部性（lognormal σ=1.2/0.4），GPU+SN vs CPU+SN vs GPU+NR vs CPU+NR（SN=Storage-Next SSD、NR=Normal SSD），并对照内存 KV 水平（FASTER 100+ Mops/s）；②ANN 吞吐（KQPS）vs DRAM 容量、四种 reduced→full 配置（512B→2/4/6/8KB，full-vector promotion 率 5%/10%/15%/20%），对照 DiskANN 约 5 KQPS；③KV 负载因子临界值 α_critical（B≥4 时通常 >0.95）与插入位移链期望长度 E[L]=α^(2B)/(1−α^B)；④ANN recall（MRL 生成的 MS MARCO、20 Newsgroups、DBpedia 语料上 recall>98%）。
- 硬件平台是什么，配置是什么。
  - 无实物实现，全部为解析建模 + MQSim-Next 模拟（模型驱动）。平台参数（Table III）：CPU+DDR（DDR5-5600 12 通道共 540GB/s DRAM 带宽、CPU IOPS 上限 100M、~1M IOPS/core、归一化核成本 4）与 GPU+GDDR（GDDR6-20 8 通道共 640GB/s、GPU IOPS 上限 400M、~4M IOPS/SM、归一化 SM 成本 3，遵循 NVIDIA SCADA 平台/Hopper 代 GPU）。每平台挂 4 块 SSD（Storage-Next SLC 或 Normal SLC），SSD 带宽利用率上限 70%（降低尾延迟）；99 分位读延迟约束 13μs(512B)/17μs(1KB)/26μs(2KB)/44μs(4KB)（ρ_max=90%）。DRAM 容量为可调变量（工作集 512GB~4TB 级，远大于 DRAM 容量）。
- 模型是什么。数据集和bench分别是什么。
  - KV 案例：5TB KV 存储、800 亿个 64B 条目（80 billion 64B items）、负载因子 0.7、桶大小匹配设备类别（Storage-Next 512B、Normal 4KB，桶=1 个 SSD 块，每桶 B=块大小/KV 大小）、20% PUT 为 insert 其余为 update、访问间隔 lognormal 强局部性（σ=1.2）/弱局部性（σ=0.4）。ANN 案例：80 亿 embedding 语料、reduced 固定 512B、full 2/4/6/8KB、HNSW 图（节点图链接元数据与节点同驻 SSD，DRAM 只缓存高层热节点）、MRL 生成的三个语料（MS MARCO、20 Newsgroups、DBpedia）评估 recall。benchmark 对照：内存 KV=Redis/FASTER/MICA，混合= RocksDB/WiredTiger/Bw-tree，SSD ANN=DiskANN/Filtered-DiskANN。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文（arXiv:2511.03944，ISCA 2026）与 MQSim-Next 均未见公开仓库（联网搜索无法确认）；基线 MQSim 开源（https://github.com/CMU-SAFARI/MQSim）。两个案例算法为论文提出、以分析框架+MQSim-Next 模拟评估，无开源实现。
  - 算法 pipeline 执行例子（KV 案例，一次 GET 的数据路径，按论文描述重建，非开源代码）：① 请求 key 到主机（CPU 或 GPU）→ 计算两个 Cuckoo 候选桶哈希（桶=1 个 SSD 块，l_blk=512B on SN / 4KB on NR）→ 无 DRAM 缓存命中则发 1~2 次 SSD 随机块读（平均 1.5 次）→ ② SSD 内部：FTL 逻辑→物理地址翻译（8B/条、控制器内部 DRAM 40GB/s、xlat 上限 ~5G IOPS 非瓶颈）→ NAND 多平面并行 sensing（SLC τ_sense=5μs、N_Plane=6、N_CH=20、B_CH=3.6GB/s、SCA τ_CMD=150ns）→ 512B 扇区 BCH 内码解码（LDPC 外码跨 8 扇区；小读只解 BCH 跳过 LDPC，无读放大）→ PCIe Gen7 ×8 返回主机 → ③ DRAM 缓存按 lognormal 访问间隔分布判定命中/未命中：未命中流量 2·Ψ_d(T)（SSD→DRAM DMA + 一次 DRAM 读）、命中流量 Ψ_c(T)（Eq.4 的 DRAM 带宽需求 B_DRAM^use(T)=Ψ_c(T)+2Ψ_d(T)）。插入路径：更新先写 SSD 驻留 WAL 合并同桶更新，WAL 超阈值后提交进 blocked-Cuckoo 块并回收日志空间；负载因子保持 0.7<α_critical>0.95，位移链 E[L]=α^(2B)/(1−α^B)≪1，插入延迟近常数。效果：GPU+SN 读重负载下 100+ Mops/s（FASTER 内存级）；换 CPU 主机则瓶颈转移至 host IOPS。
  - ANN 案例执行例子：查询向量 → ① 先取 512B reduced 向量（小块随机读、IOPS-bound，Storage-Next 高小块 IOPS 直接受益）→ 粗筛淘汰 >90% 候选（Gao et al. 报告 >90% 距离比较只是确认拒绝）→ ② 仅对 promoted 子集（512B→2/4/6/8KB 对应 5%/10%/15%/20%）取 full 向量（带宽-bound，被大拒绝率摊薄）重排 → HNSW 高层节点（访问间隔短）驻留 DRAM、底层节点图链接与节点同驻 SSD。效果：GPU+SN 相对 Normal SSD 一致 2-3× KQPS 优势，从 7-11 KQPS（小 DRAM）升到 13-17 KQPS（512GB），大 promotion 时 300-400GB 后 GDDR 带宽封顶（8.3 KQPS 平台），远超 DiskANN ~5 KQPS。

## From Memorization to Generalization: A Practical Neural Network Prefetching Framework

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现 = Moirai，一个面向 L1D 数据缓存的神经网络预取框架，核心是 CaPNet——一个高度紧凑的二值化时序卷积网络（Binarized TCN）。算法层面：输入为全局地址流的 delta 序列（10 个历史 delta），经 Window-based Extended Delta 预处理后送入 3 层 TCN（通道结构 [8,4,2]），输出下一个 delta 的预测，再结合辅助 stride 预取器与 Adaptive Control Unit 生成预取地址。训练采用在线 on-chip 方式：混合精度潜在权重 Wraw（第一层 7-bit，其余 4-bit）+ Straight-Through Estimator (STE) 反向传播 + 梯度共享（同一卷积核内多个权重共享梯度计算与存储单元）。前向推理把 MAC 变成 1-bit 的 XNOR+popcount。
  - 实验比较：baseline 预取器 = IPCP（DPC3 冠军，指令指针分类器）、Berti（本地 delta 预取器，2.55KB）、Pythia+Hermes（SOTA 智能预取器组合，L2C 强化学习 + L1D 感知机）、SPP-PPF（混合预取器 + 感知机过滤）。指标 = IPC speedup（相对无预取 baseline 的几何平均）、Accuracy/Coverage/Timeliness/Memory Traffic（算术平均）、多核（4核混合）speedup、存储开销、带宽敏感性、消融实验。
  - 结果：单核平均 11.48% speedup（SPP+PPF 12.87%、IPCP 12.12%、Berti 10.48%），多核 4 核混合 7.8%（Pythia+Hermes 8.3%、Berti 7.3%）；覆盖率 18.18%、准确率 43.63%、及时性 92.37%；存储仅 780 Bytes（Berti 2.55KB、IPCP 16.7KB、Pythia+Hermes 29.5KB、SPP-PPF 39.34KB）；面积 1178 μm²、功耗 8.5mW（ASAP7 7nm @ 4GHz）。
- 硬件平台是什么，配置是什么。
  - 模拟平台 = ChampSim trace-based 模拟器，4 GHz 乱序核心、6-issue/5-retire、352-entry ROB、TAGE-SCL 分支预测器；L1D 48KB 12-way 5-cycle、L2 512KB 8-way 10-cycle、LLC 2MB/core 16-way 20-cycle；TLB：L1 i/dTLB 16 sets 4-way 1 cycle、STLB 128 sets 16-way 8 cycles；主存单通道 3200 MTPS（tRP=tRCD=tCAS=12.5ns）。硬件开销评估用 Verilog RTL + ASAP7 7nm 预测性 FinFET PDK 综合 @ 4.0GHz（1-cycle-per-layer 流水线，前向 3 周期出预取地址；也可单周期 2.5GHz 设计）。ChampSim 实现严格建模 CaPNet 的 3-cycle 微架构延迟（baseline 常假设 0-cycle）。
- 模型是什么。数据集和bench分别是什么。
  - 模型 = CaPNet（3 层二值化 TCN，通道 [8,4,2]，输入窗口 10 个 delta，≈380 参数预算；权重 1-bit 二值化、潜在权重 Wraw 首层 7-bit 其余 4-bit）。设计空间探索对比了 RNN、LSTM、TCN 同参数预算下的预测准确率，TCN 最高。数据集/bench = ①DPC3 套件 43 个内存密集型 workload（MPKI>3，来自 SPEC 2006 与 SPEC 2017，用官方 Simpoint 权重加权平均）；②GAP Benchmark Suite（图处理，指针追逐不规则访问，如 bfs 70.65% speedup）；③多核 = 20 个随机生成的 4 核混合；④合成 workload（Braun 和 Litz 的复杂度递增流：predictable 短周期流、random 长周期伪随机流）用于展示泛化能力（Figure 1：Berti 随复杂度下降、Moirai 保持高准确率）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：是。Artifact 归档于 Zenodo 两个仓库：Workload Traces https://doi.org/10.5281/zenodo.19447159、Source Code https://doi.org/10.5281/zenodo.19450687（MIT License）。仓库内容：prefetcher/ 与 inc/（moirai.cc/moirai.h 及 baseline 实现）、inc/uarch/isca_config.h（处理器配置）、libbf/（Hermes 的 Bloom Filter 库）、ISCA26_AE.exp 与 ISCA26_AE.tlist（实验与 trace 定义）、build_champsim.sh（编译脚本）、scripts/（Python 作业生成）。基座 ChampSim 开源：https://github.com/ChampSim/ChampSim（wiki: https://champsim.github.io/ChampSim/develop/）。复现流程：source setvars.sh → ./build_champsim.sh（0-latency 或 3-latency 配置）→ 下载 trace → scripts/ 生成作业脚本 → 运行，结果在 output/ 文本文件中（IPC 等指标）。
  - 算法 pipeline 执行例子（一次预取推理，沿一个 delta 数据路径）：① LSU 生成 speculative VA → Window-based Extended Delta 预处理：以当前地址为窗口起点 aw[0]，在 5 个地址的小窗口内找下一个与 aw[0] 同页或 8KB 空间约束内的地址 aw[i]（判断 (aw[0]>>13) ⊙ (aw[i]>>13)==1），返回 Δ=aw[i]-aw[0]；若窗口内无命中则返回 abnormal（丢弃）；该机制同时滤掉交替页访问的大振荡 delta 与 delta 稀疏长尾 → 输出干净 delta 特征流。② 滑动窗口收集 10 个连续 delta 组成输入序列 A_{i-1}∈R^{10} → CaPNet 前向：第 i 层 FCC 做 A_{i-1} ⊙ W_bin^k 的 XNOR 与 bitcount：Ac_i^k = bitcount(A_{i-1} ⊙ W_bin^k)（式 1，1-bit 权重/激活，[8,4,2] 通道逐层下采样，整个输入窗口广播到各通道）→ 输出预测 delta D_pred。③ Prefetch Request Generation：以基地址+预测 delta 生成预取 VA，配合硬连线 block+1 偏移，按控制器策略（置信度三档：高=1 个、中=5 个 D_pred±2、低=9 个 D_pred±4）放入 PRQ → 通过标准流水线发到 L1D。④ 在线训练（异步，用 ROB 的 retired PA 流）：loss 反向经 STE 绕过二值化器更新 Wraw（第一层 7-bit、其余 4-bit），BCC 用 G_{i+1} 与 Ac_i^k 算 ΔW_raw^k = G_{i+1} * Ac_i^k（式 2），同一卷积核内共享梯度计算/存储单元。

## GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design

- 属于算法pipeline的实现是什么？实验比较什么？
  - GRAINS 在算法层面提出面向大规模基因组 de Bruijn 图（DBG）查询的存储友好执行流程，核心是两项新优化：(1) Cross-Read K-Mer Batching——不再逐条 read 查询，而是把同一读集中不同 read 的 k-mer 合并批量查询，用轻量数据结构维护 k-mer→所属 read 的映射，图查询返回匹配 k-mer 的颜色后按 read 汇总，从而成量减少对图的随机访问次数；(2) Genome-Graph-Aware Query Reordering——利用最小完美哈希 k-mer 字典（SSHash）的固有性质：定位小范围查找所需的 Sizes 数组远小于 Offsets 和 Strings（Sizes 仅占大图的 <4% 空间）。因此先在主机侧用 Sizes 完成 k-mer 查找、拿到 Offsets 索引，据此把 k-mer 排序并切成等长 disjoint 批次（3）再发给 SSD，使 SSD 侧对 Offsets 的访问变成顺序流；排序时利用"按 Sizes 排序后连续 k-mer 共享同一 minimizer"的性质，只存一次 minimizer、只保留差分，将传输数据量平均压缩 2.3×（4）。二者与数据传输/查询构成流水线：一批的排序与上一批的传输和 Offsets 查询重叠。
  - 实验比较：k-mer set lookup、alignment-free read mapping、alignment-based read mapping 三种任务的吞吐/速度与能耗，比较对象为 Fulgor（节点中心软件基线，SSHash 字典）、MetaGraph（边中心软件基线）、IdealAccMem（零主存开销的理想硬件加速基线，代表理想 PIM）、GRN-Ext（GRAINS 算法优化但 ISP/IFP 逻辑在 SSD 外的 PCIe 加速卡），以及消融配置 GRN-B（仅 batching）/ GRN-B-S（batching+scheduling）。
- 硬件平台是什么，配置是什么。
  - 主机：AMD EPYC 7742 CPU（128 物理核），1.5 TB DDR4 DRAM（除成本实验外）。两块 SSD：SSD-G4（PCIe Gen4，顺序读 7 GB/s，channel I/O 1.2 GB/s）与 SSD-G5（PCIe Gen5，顺序读 14.8 GB/s，channel I/O 2.4 GB/s）；均为 TLC NAND、4 TB 容量、4 GB 内部 DRAM、16 channel × 8 die/channel × 4 plane/die、4-KiB 页。成本优化系统（$）= 64 GB 主机 DRAM + SSD-G4；性能优化系统（$$$）= 1.5 TB DRAM + SSD-G5。
- 模型是什么。数据集和bench分别是什么。
  - 处理对象不是 NN 模型，而是基因组图分析任务：查询读集（reads）对大规模 DBG 做 k-mer 集合查找与 read mapping。图用两种工具构建（均为无损编码，各工具精度一致）：Fulgor 节点中心 DBG（SSHash 字典）与 MetaGraph 边中心 DBG。数据集：MetaSUB Consortium 全球样本子集构建的 G_MetaSUB 图（含颜色，Fulgor 格式 659 GB / MetaGraph 格式 822 GB）；SRA 公共部分代表性子集构建的 G_SRArep 图（161 GB / 231 GB）。查询集：QL=10M reads（3.5 GB）、QM=1M、QS=100K（35 MB）。bench = k-mer set lookup、alignment-free read mapping、alignment-based read mapping（集成 SeGraM 序列到图对齐加速器）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - GRAINS 本身：论文（arXiv:2606.26468）未提供代码仓库链接，arXiv 页面亦无代码可用性说明，开源状态无法确认。其依赖与基线工具均开源：Fulgor（github.com/jermp/fulgor）、MetaGraph（github.com/metagraph-rs/metagraph）、SSHash（github.com/jermp/sshash）、SeGraM（github.com/CMU-SAFARI/SeGraM）。
  - 算法 pipeline 执行例子（k-mer set lookup，单批）：① 对 read set 中每个 read 提取全部 k-mer；② 对每个 k-mer 取 minimizer（哈希值最小的 m-mer），用 minimizer 的最小完美哈希值 h 索引 Sizes（主机侧）；③ 由 Sizes[h+1]−Sizes[h] 得该 minimizer 对应的 Offsets 区间，据此把 k-mer 按 Offsets 索引排序、切成等长 disjoint 批次，批次内连续 k-mer 共享 minimizer 时只传一次 minimizer+差分（约 2.3× 压缩）；④ 批次经标准 NVMe 数据路径送入 SSD 内部 DRAM（不写 flash），SSD 按序访问 Offsets（顺序流）取出指向 Strings 的页内偏移；⑤ 调度器按 GST 表 round-robin 把请求分发到各 die/plane，读取 Strings 页并在 die 内做 k-mer 与窗口的位级比对（IFP comparison）；⑥ 命中 unitig 后流式扫描 Color Bitmap 得到颜色索引，返回主机；⑦ 主机按 k-mer→read 映射把颜色汇总到每个 read，得到物种/样本分类结果。


## HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现 = HERO，一个 hoisting 增强的 DFG 优化框架（算法级程序变换），在 CKKS 程序数据流图上重组 keyswitch 并行块（PKB）以最大化 hoisting 的通信/计算削减潜力：(1) PKB identifying——从 DFG 输入遍历，按路径顺序给 keyswitch 分层，同层分组为 PKB；(2) Degree-minimized PKB expanding——用交换律算子（EWO、Autom 与 ModUp/ModDown 可交换，Sec. II-B2）贪心扩展每个 PKB 把入/出度压到最小，使 hoisting 把冗余 ModUp/ModDown 提到 PKB 输入/输出端，通信频率随 ModUp/ModDown 削减而下降（代价是中间 MemOps 模域从 Q 升到 PQ 或 PQ·dnum，计算略增）；(3) PKB fusing（首次提出）——利用旋转可加性 Rot(Rot(ct,s),t)=Rot(ct,s+t) 与 EWO 后移（Rot(PMul(ct,pt))=PMul(Rot(ct),Autom(pt))），把两个串行 PKB（n1 与 n2 条旋转路径）融合成 O(n1·n2) 条并行旋转的大 PKB（逆 BSGS 变换），消去 PKB1 输出与 PKB2 输入的额外 ModDown/ModUp；代价是 evk 数量（按非重复子集计）、IP 数与中间 MemOp 计算量上升；(4) Fusion evaluator——FuseScore(i,j) 在融合后 evk 数超存储容量时判无效，否则表示 n_i·n_j 乘积不变约束下计算+通信的最大联合节省，按 DP 递推式 DP[i][j]=max_{1≤j'≤j−1} DP[i][j']+DP[j'+1][j]+FuseScore(j',j'+1) 求全局最优融合方案；(5) BSGS 配置探索——内存足够时禁用 BSGS（C2S/S2C），内存受限时偏好 bs 与 gs 差距更大的配置以暴露更多 hoisting 并行（BERT 首 FFT 阶段因高层级仍保留 bs=2/gs=32）。
  - 实验比较：在 Bootstrapping、HELR、ResNet-20/56、BERT 上比较 (a) Min-KS（SHARP [25] baseline 算法，均匀步长串行旋转）；(b) 直接对原始程序应用 hoisting（Anaheim [27]/FAST [16] 做法，受限于<10 的低 keyswitch 并行）；(c) hoisting 且禁用 BSGS；(d) HERO。结果：HERO 使 hoisting 相比 baseline 平均削减 1.64× 计算量与 3.27× 通信量（abstract）；相比"原始程序直接 hoisting"再多削减 2.25× 计算量与 2.42× 通信量（Sec. IV-B/Fig. 15）。融合 PKB 高并行（>30）后 ModUp/ModDown 可提到 8 条并行路径首尾，无需在 PKB2/PKB3 之间做额外 ModUp/ModDown（ConvBN 用例：3 个 9/8/8 并行 PKB，hoisting 只优化 PKB1，融合后 25→ 更少 ModUp/ModDown）。DFG 优化 + IRF 数据流使 8 GB HBM 即可容纳最优融合方案的 evk 工作集。
- 硬件平台是什么，配置是什么。
  - 无真实硬件，全部在自研 cycle-accurate 模拟器 + RTL 综合上评估（Sec. VI/VII）。FHE 参数 N=2^16, L=35, L_eff=8, k=12, α=12, dnum=3, λ=128-bit；xPU 1 GHz（HE²-SM 44 MB / HE²-LM 84 MB scratchpad），xMU 450 MHz；2×HBM2 栈、8 GB 容量、1 TB/s 带宽；xPU 7nm / xMU PE 12nm 工艺。算法优化本身与硬件解耦（DFG 输入由 FHE 编译器生成），但收益在 IRF 异构数据流下才充分体现（EVF 下 hoisting 反而因 evk 复用下降变慢）。
- 模型是什么。数据集和bench分别是什么。
  - 负载/模型：CKKS 密文上的明文矩阵×密文向量乘法（ML 推理核心）+ bootstrapping pipeline（C2S/S2C）。bench = ①Bootstrapping：FFT-like [6] 三阶段、8 个有效层、fully-packed 实现 [25][26]；②HELR [21]：logistic regression 二分类 ML 负载，batch 1024，32 次迭代平均延迟；③ResNet-20/ResNet-56 [30]：multiplexed packing CNN，加密 32×32×3 图像、batch 1（ResNet-56 的对比方案结果按计算量从 ResNet-20 缩放）；④12 层 BERT [53]：128×768 输入序列单次推理。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文全文 arXiv 2605.31004（CC BY-NC-SA 4.0 许可），全文无代码/artifact 链接，联网搜索未找到公开 GitHub 仓库，无法确认是否开源。输入 DFG 由开源 FHE 编译器（EVA [10]、CHET [11]、ResiBM [35]）生成。
  - 算法pipeline 执行例子（两串行 PKB 融合 + hoisting，张量级，对应 ConvBN 简化 DFG）：设 PKB1 有 n1=2 条并行旋转路径 {Rot(ct,s1), Rot(ct,s2)}，两路径输出经 EWO 求和后进入 PKB2 的 n2=2 条旋转 {Rot(·,s1'), Rot(·,s2')}。①融合：把 PKB1 到 PKB2 路径上的 EWO 沿每条路径后移（PMul 与旋转交换），使 PKB2 每条路径紧邻 PKB1 一条路径；②用旋转可加性合并两级旋转：Rot(Rot(ct,s_j),s_i')=Rot(ct, s_j+s_i')，得 2×2=4 条并行旋转的融合 PKB（步长 {s1+s1', s1+s2', s2+s1', s2+s2'}，若出现重复步长则去重减少 evk）；③hoisting：4 条并行的 ModUp（同一输入 ct）合并为 1 次共享 ModUp，输出端线性组合后只做 1 次 ModDown；未融合时 PKB1 需 1+2、PKB2 需 2+1 次 ModUp/ModDown。④权衡：融合后 IP 数从 n1+n2=4 增至最多 4、evk 数从非重复子集增长（本例 3~4 个），Fusion evaluator 用 FuseScore 量化"省下的 ModUp/ModDown 通信 vs 增加的 evk 存储与 MemOp 计算"，在 8 GB 存储约束下由 DP 选出全局最优融合方案（式 5）。

## HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  算法级设计（服务于 GPU kernel 实现，论文给出 Alg. 1/2/3）：(1) 分层 NTT 分解（Alg. 1，基于 Bailey 4-step FFT）：把 N=2^13~2^16 的多项式分解为 N = N1·N2、N1 = 8·8·N13、N2 = 8·8·N23，radix-64 作为 Inner-NTT 基例（匹配 FP64 TCU 8×4×8 MMA），递归应用 4-step 分解保持 O(N log N)（Inner-NTT 开销 C_NTT = Nk(t+1)、Hadamard C_HP = t·O(N)，N = k^{t+1}）；分解粒度按 (N13, N23) 取 (2,1)/(4,1)/(2,4)/(4,4) 对应 N=2^13/2^14/2^15/2^16，且两阶段 N1/N2 尽量接近以平衡负载。(2) FP64 多精度算术（MPA）方案：32-bit 乘法把一个乘数拆成两个 16-bit 部分分别乘另一个 32-bit 乘数、最后 INT64 位合并，单个 32-bit 乘法仅 2 次 FP64 乘法（INT8 方案 TensorFHE/WarpDrive 需 16 次），FP64 53-bit 精度下累加 8 个 MMA 乘积不溢出。(3) CRPMAC 密文复用 PMAC：BSGS PCMM 中把全部 GS-Rots 推迟到末尾、按 GS 方向批处理，单个 baby-step 密文乘以一批明文（跨 batch 复用密文读），内存足迹小于 [22] 的 BS 方向批处理。(4) Bootstrapping 操作重排：GS 相位 ModDown 后紧跟 ModUp 时把 EWSub 提前，使 ModDown INTT 与 ModUp INTT 可批处理合并、并改善 (NTT2-IP) 负载均衡。
  实验比较：消融实验逐项量化每项算法贡献（Table V 配置）：NTT 吞吐/延迟逐项叠加 TLMOP/TransOP/TFOP/RowMaj（图 15：延迟 -61.1%、吞吐 3.0× 到 932.6 KOPS）；KeySwitch/Bootstrap 分解对比 NTT+/COOP/CRPMAC 贡献（图 19/21：COOP 使 KeySwitch 快 1.24×、(BConv2-NTT1) 1.36×、(NTT2-IP) 1.32×、CRPMAC 使 CtS/StC EWArith 快 1.34×，HyperDrive-CORE bootstrap 1.85× vs BASE、3.10× vs BASE-MDRS、1.64× vs BASE-EXT）；MPA 开销对比（图 16：相对 INT8-TCU WarpDrive 降 84-91%，仅占 NTT 总延迟 5.0-7.3%）。
- 硬件平台是什么，配置是什么。
  NVIDIA A100-PCIE-80G GPU（1.41 GHz）+ INTEL XEON PLATINUM 8558P（2.7 GHz），CUDA 12.8，GCC 11.4；扩展实验 H100。NVIDIA Nsight Compute 做微架构分析（warp stall、occupancy、寄存器占用）。
- 模型是什么。数据集和bench分别是什么。
  评估对象为 CKKS 计算 workload，非传统 ML 模型：(1) CKKS Bootstrapping（slim + double-hoisting 算法，double scaling 下 CtS/StC 各 6 level、EvalMod 16 level）；(2) HELR——CKKS 下逻辑回归训练，32 轮迭代、每轮 batch 1024，报告每轮平均延迟；(3) ResNet20/32/56/110——CIFAR-10 加密 CNN 推理，65 度近似 ReLU 适配 Set-E/F。参数集 Set-A~Set-F（N=2^16，L=34+1~46+2，WordSize 32/36，dnum 3~35，BatchSize 1/128）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  开源情况：论文未给出代码链接，联网未找到 HyperDrive 公开仓库（无法确认）；基线 FIDESlib（https://github.com/CAPS-UMU/FIDESlib）与 Cheddar（https://github.com/scale-snu/cheddar-fhe）开源。
  算法 pipeline 张量计算例子（radix-64 Inner-NTT，Alg. 1 + 图 9）：把多项式系数按预转置布局组织为数据矩阵，每 warp 处理一个 64 点 Inner-NTT（FP64 fragment 分布：每线程 Fragment A/B 各 1 元素、Fragment C 2 元素、结果 Fragment D 均分）。一次 64 点 NTT = 4 次 MMA（8×8）：MMA1/2 计算 D = A·B，B 为 twiddle factor matrix TFM 的高/低 16-bit 分解（完成 8 个 radix-8 NTT），中间做 Bit-Merge（两路 16-bit 乘积合并回 32-bit）+ ModRed + EWMult；MMA3/4 把数据当 Fragment A、twiddle 当 Fragment B 完成第二级 radix-8，并利用 Fragment A 奇/偶列交替选取隐式完成转置；最后 Bit-Merge + ModRed 输出。MPA 例子：c = a·b mod q（a,b 32-bit），拆 a = a_hi·2^16 + a_lo，两次 FP64 MMA 分别算 a_hi·b 与 a_lo·b（乘积 ≤48-bit），累加 8 个乘积仍 ≤53-bit，INT64 位合并后模约减。CRPMAC 例子（BSGS PCMM，图 7/14）：BS 相位只做一次预旋转（ModUp）得到 bs 个 baby-step 密文 ct_i；原方案每轮 GS 生成都重读同一 ct_i（冗余 GMEM 读）；HyperDrive 把全部 gs 个 GS-Rot 推迟，按 GS 方向把 ct_i 与整批明文 pt_{i,j}（j=1..gs）批量乘加（一次读 ct_i 复用 gs 次），批量输出 ct'_j 后统一做 GS-Rots。

## Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为新的哈密顿量模拟算法范式 Partial Trotterization 及其配套算法优化：传统 Trotterization 对每个哈密顿量项分别指数化（e^{iΣH_i t} → Π e^{iH_i t}），误差与逐项 commutator 有关；Partial Trotterization 把非对易项分组成 partition（每个 partition ≤3 个 qubit，如 {H_i, H_j}），直接编译成组的指数 e^{it(H_i+H_j)}。理论分析（BCH/commutator scaling，Eq.7-9）表明分组消除了组内 commutator 贡献，误差从全 Trotter 的 sum_{i<j}[H_i,H_j] 降为仅跨组 commutator，一阶/高阶 Trotter 下电路深度随 group 大小呈二次缩减，因此达到相同精度所需 Trotter 步数大幅下降（实测一阶 Trotter 下 10 steps 误差随 group 增大急剧下降，图 8）；再叠加两项算法优化：跨 Trotter step 的可交换 group 合并（e^{iH_it}e^{iH_it}→e^{i2H_it}）与组内随机 shuffle 把 coherent 误差转 stochastic。
  - 实验比较什么：与状态最优哈密顿量编译方法对比——Qiskit Rustiq（Pauli network synthesis，arXiv:2404.03280）与 Qiskit PauliEvolutionGate（Paulihedral，arXiv:2109.03371 的重排/同时对角化思路），以及 Qiskit/BQSkit 通用 unitary synthesis（消融 MCTS 重写）。指标：达到 <1% 近似误差（L2 范数）时的 depth/CNOT/U3 门数与所需 Trotter 步数（图 6），及不含误差缩减的绝对门数对比（图 5）。结果：depth 与 CNOT 最多减 86%（平均 40%）、单比特门最多减 85%（平均 11%），一阶最优场景最高 10× 深度缩减。
- 硬件平台是什么，配置是什么。
  - 数值仿真/编译在 AMD EPYC 9654P 96 核 CPU + NVIDIA A100 GPU（80GB，运行 MCTS 搜索）；软件 PyTorch 2.5.1 + CUDA 12.1、Qiskit 1.3.2、BQSkit 1.2.0。精度验证用 8-10 qubit 哈密顿量的 L2 范数（目标 >99.5% 精度）；可扩展性评估用 28-220 qubit 哈密顿量的门数与编译时间。
- 模型是什么。数据集和bench分别是什么。
  - 无传统 ML 模型；"模型"即被模拟的物理哈密顿量集合（bench，Table 2）：Fermi-Hubbard（FH，三角/方格/1D 拓扑，8-128 qubits，qubit#=2×site）、Heisenberg（HB，10-144 qubits）、Ising（IS，10-144 qubits）、LiH 分子（10 qubits）、HF 分子（10 qubits）、PD-1 蛋白片段（28-222 qubits）。费米子模型用 Bravyi-Kitaev 映射到 qubit。拓扑覆盖 1D/2D（方格、矩形、三角）与分子/电子结构（非局域长程关联），term 权重多样。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - Kernpiler 未发现公开仓库（arXiv:2504.07214 无代码链接）；baseline Qiskit（Rustiq/Paulihedral）与 BQSkit 开源。
  - 算法 pipeline 伪代码：Algorithm 1（partitioning）：sort terms by highest qubit index then weight；对每个 term 贪心加入第一个"并集 qubit 数 ≤3"的 partition，否则新建 partition。Algorithm 2（reorder/randomize）：BuildConflictGraph（节点=unitary，[t_i,t_j]≠0 加边）→ GreedyCommutingGroups（贪心最大独立集迭代）→ 每 step 组内随机化顺序 → 相邻 step 重排使相同 group 相邻 → 跨 step 合并对易项。张量计算例子：H = H_i+H_j+H_k+H_l（4 个互不对易项，H_i=X_1Y_2Z_3 等，3-qubit），全 Trotter 误差 ∝ 6 个 commutator [H_i,H_j]+...；Partial Trotterization 把 (H_i,H_j) 与 (H_k,H_l) 分别合为 8×8 unitary e^{it(H_i+H_j)}、e^{it(H_k+H_l)}，误差只剩 4 个跨组 commutator [H_i,H_k]+[H_i,H_l]+[H_j,H_k]+[H_j,H_l]，消除 2 个组内 commutator（Eq.3 vs 4）。误差缩放：Error_partitioned = Σ_{A≤B}|[H_A,H_B]|Δt²/2，组大小 n_A 时组内 commutator 以 ~n_A² 组合增长地被消除，使误差随 partition 增大显著下降；非对易对占比不随 lattice 增大（图 9，n=3/5），保证可扩展性。

## Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration（近似层次匹配：论文本体为 FPGA 硬件架构，本层取其两个算法级贡献——外部内存 columnsort 播种算法与 tiled bit-parallel traceback 算法）

- 属于算法pipeline的实现是什么？实验比较什么？
  - 两个保持 Minimap2 算法语义（无精度/近似折衷）的算法级贡献：
    1. **外部内存 columnsort 播种算法**（替代 Minimap2 的 minimizer 哈希表随机查找）：把 reference/query 的 minimizer 元组流（16 B = minimizer + index）经 PCIe 溢出到 NVMe，按 minimizer 字典序做外部内存 columnsort 全局排序（每个 256 MB 列由 16 个 16-to-1 单发射 merge-sort kernel 排序，4 轮列排序 + 3 次转置回传）；排序后两有序流做流式 zip 匹配得 anchor（〈idxQ,idxR〉），再按 idxR 二次 columnsort 排序作为 seed 输出。算法目标是消除"内存容量随基因组增长"的瓶颈（内存恒定 ~8 GB、7× 降低），而非提升 seed 性能。刻意不做 Minimap2 的启发式 anchor 过滤（避免随机访存），代价是下游工作量放大（人类基因组 7.06× 更多 chains）。
    2. **tiled bit-parallel traceback 算法**（SWG 扩展阶段）：把 banded Smith-Waterman-Gotoh（affine gap）的逐单元串行 traceback 改为 8×8 tile 粒度并行回溯——每个 tile 边缘单元完整编码跨 tile 路径（2-bit x/y 偏移，最长 16 步、15 个边缘单元 → 480 bit/tile，对齐 512-bit HBM 接口），traceback 每 cycle 前进一整 tile（对编码 popcount x/y 位定位下一 tile），把"每 cycle 1 单元"变成"每 cycle 1 tile"。
  - 实验比较：seed 性能 vs mm64（快 70%）与 G³SA（慢 15%），内存峰值 vs Minimap2（7× 降低，恒定 ~8 GB，且避免 Minimap2 memory chunking 的跨 chunk 质量损失）；extend 性能 48 GCUPS/FPGA（96 GCUPS 双 FPGA，mm64 为 27.21，≈4×），traceback 延迟 vs Cheng24/Li21/Liao18/Turakhia18/Teng23（W=1024 固定带宽，W=512–25K 扫描；W=2048 时 traceback 开销比次优设计低 1.77×）。
- 硬件平台是什么，配置是什么。
  - 算法实现在 FPGA 上运行（非软件算法）：Lembas 原型 = 桌面级 i7-8700（16 线程/3.2 GHz/2017）+ 32 GB DDR4 + 4×1 TB M.2 NVMe + 2× Xilinx Alveo U50 FPGA（8 GB HBM2、32 pseudo-channel、PCIe Gen3x16 ~8 GB/s 双工、~250 MHz）。seeding 用 16 个 columnsort kernel（饱和全部 32 个 HBM pseudo-channel 与 ~8 GB/s 双工 PCIe，有效端到端排序吞吐 ~2 GB/s）；extend 用 16 kernel × 16 PE 脉动阵列。
- 模型是什么。数据集和bench分别是什么。
  - 无 ML 模型。"模型"即被比对的基因组/read 数据集：PBSIM3（https://github.com/yukiteruono/pbsim3）从真实参考基因组生成的 PacBio CLR 风格合成长读。三个基因组：Arabidopsis thaliana（TAIR10，119–135 Mbp，30×/50×/100×，6.4/12/24 GB）、Homo sapiens（HG16/HG002，3.1 Gbp，30×/50×/100×，176/357/719 GB）、Allium cepa（DHCU066619，14.9–16 Gbp，15×/30×，419/846 GB）。banded SWG 带宽配置：默认 20 kbp band（Minimap2 默认），扩展对比用 W=512/1024/2048/13K/25K。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文声明"正在将 Lembas 所有方面开源"，但未给仓库链接；截至 2026-08 联网搜索（含 SATA Lab satalab.github.io 的 ISCA 接收公告）未找到公开 GitHub 仓库，无法确认。依赖的开源组件：Minimap2（https://github.com/lh3/minimap2，baseline 与算法语义定义）、PBSIM3（https://github.com/yukiteruono/pbsim3）、NextDenovo（https://github.com/Nextomics/NextDenovo，all-to-all 工作流参数）。
  - 算法 pipeline 执行例子（一次 reference-based 人类基因组比对，覆盖 seed 算法与 extend 算法）：
    ① **seed（外部 columnsort）**：reference/query 经 minimizer parse kernel 产出流式 〈minimizer, index〉 16 B 元组（滑动窗口内字典序最小 k-mer）→ 溢出存 NVMe（数据量超过 FPGA HBM）→ columnsort：把数据组织成 r×c 网格（约束 r≥2c²，256 MB HBM bank 上限下可排序 ≤512 GB），4 轮"每列 16-to-1 单发射 merge-sort 排序 + 网格转置/移位"（每轮 NVMe→HBM→排序→转置→写回 NVMe；每个 250 MHz kernel 处理 4 GB/s，6 次 sweep 排序 256 MB，PCIe 8 GB/s → 有效 ~2 GB/s）→ 排序后 reference/query 两个有序 minimizer 流经 anchor matching kernel 流式 zip 匹配 → 〈idxQ,idxR〉 anchor 流存 NVMe → 按 idxR 再 columnsort 一次（此排序为按参考位置排序，供后续 chaining 用）→ 输出 anchors。
    ② **extend（tiled traceback）**：对每条 chain，16×16 PE 1D 脉动阵列按行交错计算 banded SWG 分数矩阵（affine gap：每单元三值 S/E/F，用 S[i−1][j−1]、E[i−1][j]、F[i][j−1]、b^j 四个输入；F 缓存在 PE 内、E/b 走 E,b 寄存器链、S 走 2 元素 FIFO）→ 前向路径每 8×8 tile 边界把"从相邻 tile 外起始到本 tile 各边缘单元的最优路径"以 2-bit 编码写入历史寄存器（480 bit/tile，覆盖 15 个边缘单元 × 最长 16 步）→ 反向 traceback 时读边缘单元编码，popcount x/y 位直接算出下一 tile 的入口单元，每 cycle 前进一整 tile（每次预取 4 个 tile 缓解 HBM 延迟）→ 输出 SAM 格式对齐结果。
    效果：seed 内存恒定 ~8 GB（Minimap2 需 384 GB 级且 chunking 有质量损失）；extend 48 GCUPS/FPGA；全系统成本效率 3× vs G³SA、5× vs Parabricks、2× vs Cheng24。

## Leveraging Phase Polynomials for Quantum Circuit Optimization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 新算法模型：把量子线路优化建模为 GF(2) 上 parity 矩阵的 CNOT 最小化合成问题（CNOT network synthesis 已知 NP-hard [10,34]，故采用启发式 A* 搜索）。两个算法级贡献：(1) 统一耦合 parity 矩阵 [phase-parity | output-parity] 联合优化 phase 与 output parity 网络——先前工作分开处理（phase-parity 用贪心/phase-only 优化，output-parity 事后高斯消元），错过二者相关性（同一 phase 实现可诱导不同 output 代价，Fig.3）；(2) cross-block IR + SSA 命名实现跨 H barrier 的长期奇偶性复用——把被 basis-changing gate 分割的多个 phase polynomial block 合并，用 rank-based 线性相关性检验（rank(M∪{v})=rank(M)）保证可消除性，暴露局部重写无法覆盖的跨 block 结构。算法保持精确等价（phase polynomial 精确建模 {CNOT,Rz}，sum-over-paths 形式 U|x⟩=e^{ip(x)}|g(x)⟩），无近似/误差折衷。
  - 实验比较什么：与 phase polynomial baselines（Rotation Merging、Single-block Greedy——平均总门减 26.93%/两比特门减 8.14%、Gray-Synth——平均 CNOT 减 17.62%）及通用 subcircuit rewriting（Quartz 22.17%/16.88%、QUESO 27.83%/20.70%）比门数与 CNOT 数；PhasePoly 平均总门减 34.70%（最高 50.00%）、CNOT 减 26.83%（最高 48.57%）。深度：小/中电路逻辑深度平均减 22.47%（vs Quartz 13.26%、QUESO 19.64%），大电路族 40.91%（vs 4.03%/14.45%）；映射后物理深度平均减 28.35%、大电路族 40.84%；FT wall-clock 运行时减 44.62%（vs 11.99%/31.80%）。
- 硬件平台是什么，配置是什么。
  - 算法（A* 搜索、GF(2) 矩阵操作）运行于 2.8 GHz AMD EPYC 7313 CPU，Python 实现，经典侧编译/仿真，无真实量子硬件。编译预算：PhasePoly 至多 3,600 s/电路，最大实例 hwb8_113（104,068 门）用时 <5,500 s；Quartz/QUESO 固定 7,200 s/电路（Fig.5 中 PhasePoly 与重写框架均在其预算内对比）。参数：priority queue 上限 Q、解池大小 P、cross-block group 大小 G，默认 Q=P=1000。
- 模型是什么。数据集和bench分别是什么。
  - 无 ML 模型，是组合优化（A*）搜索。Benchmark 电路集（沿用先前优化研究常用基准 [9-11,17-19,39,41,42] + 近短期/FT 应用）：量子算术 adder（23–383 qubits，637–12,637 门；含 mod_adder_1024）、MCX 多控 NOT（19–499 qubits，480–14,880 门）、Hamming coding 函数 HWB（固定 16 qubits，345–104,068 门）、Hamiltonian simulation HAM（如 ham15_med 1,272 门/534 CX、ham15_high）、QAOA（如 qaoa_n10_p4）、Grover 与 Shor 算法、GF(2) 乘法（gf2^4_mult、gf2^5_mult）、barenco_tof_10 等。变分电路（GridSynth 组合实验）：QAOA Max-Cut on 3-regular 图（4–24 qubits，2,150–12,900 门）与 VQE（UCCSD ansatz，Jordan-Wigner/Bravyi-Kitaev/parity 编码 + Hamming-weight-preserving HWPA ansatz，4–12 qubits，2,641–231,780 门）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：PhasePoly（https://github.com/ruadapt/PhasePoly），其仓库内置 Qiskit+MQT-QCEC 等价性检查器，合成方法 row_heap（默认 A*）等 6 种。baseline 相关：Gray-Synth/T-par [9,10]、Quartz（https://github.com/quantum-compiler/quartz）、QUESO（https://github.com/qqq-wisc/queso，并入 qqq-wisc/guoq）。
  - 算法 pipeline 伪代码/张量计算：输入线路 → phase polynomial 表示 U|x⟩=e^{ip(x)}|g(x)⟩，p(x)=Σθ_i(x·y_i)（parity 加权和，y_i∈{0,1}^n 为 parity 向量），g(x) 为 GF(2) 线性可逆映射 → 建耦合矩阵 M=[P|O]：P 每列=phase parity 向量（如 (110)ᵀ=q0⊕q1），O=g 转置后每列=output parity 向量；CNOT(i,j) 执行 row_i←row_i⊕row_j 同时作用于 P 与 O。A* 搜索：状态=当前矩阵；目标=P 空且 O=I；扩展=对 active row pair（能降低 active column set 中至少一列 Hamming weight 的行对）施加 CNOT；代价 f(n)=g(n)+h1(n)+h2(n)（h1=P 总 Hamming weight，h2=O 高斯消元到 I 的估计 CNOT 数）；tie-break 按 [f,h1,h2,−g] 字典序；space-bounded priority queue（满时丢低优先级节点）+ 多解池 k。张量计算例子（3 qubit，phase 项 θ1:q0⊕q1、θ2:q1⊕q2，output g(q)=(q0, q0⊕q2, q0⊕q1⊕q2)）：P=[[1,0],[1,1],[0,1]]（行=qubit，列=parity 项）、O=[[1,1,1],[0,0,1],[0,1,1]]（列式）；CNOT(q1,q0) 令 row0←row0⊕row1，使列 (110)ᵀ→(100)ᵀ 达 Hamming weight 1 → 发射对应 Rz 并删列，重复至 P 空，剩余 O 高斯消元合成（Fig.7 演示同型矩阵演化）。Cross-block：H 后新 SSA 行 inactive；消除 pre-H 行需 (1) 该行 phase 全零 + (2)/(3) 列/行隔离（单位向量），可满足性等价于目标单位向量 v∈span(候选行)，用 rank(M∪{v})=rank(M) 检验；求解 Mα=v（GF(2)）得 witness set S={i|α_i=1}，若待消行 t∉S 先 CNOT(j,t) 将 t 纳入 S'（不改变 span），再对每个 i∈S'∖{t} 做 CNOT(t,i) 实现行隔离，对每个 k≠t 且第 t 列有 1 的行做 CNOT(k,t) 清列，最后删行 t、插 H、激活 post-H 行（Fig.10 矩阵演化）。Incremental Block Merging：从单 block 优化起步，渐进扩大合并组（Group k=1..7），仅保留改善步骤，避免过合并回归（Table I：ham15_med Group 7 退化而 Incremental 最优 656 门/325 CX）。

## LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：LoKA 是一个让 FP8 低精度训练/推理对大型推荐模型（LRM）可用的系统-模型协同设计框架，核心是三个算法/模型级组件（全部以 PyTorch + 三个低精度库 TorchAO/DeepGEMM/FBGEMM 实现）：
    1. **LoKA Probe（分布感知的统计建模）**：在线学习每层激活与权重的统计分布（激活建模为多元高斯，利用 batch 维独立把存储从 O(M²N²) 降到 O(N²)，用批量 Welford tracker 流式更新均值/协方差；权重建模为矩阵正态 MN(M,U,V)，用 Kronecker-factor flip-flop EMA 更新，存储 O(M²+N²)，尺度用 trace 重归一化），再离线从学到的分布采样合成输入/权重做误差与吞吐评估，量化每层 MERE（Mean Element-wise Relative Error，相对 TF32 参考）。每 100 迭代激活、每 10000 迭代异步保存统计参数，开销 ≤1%。
    2. **LoKA Mods（模型组件重设计）**：No Bias（除最终预测层外移除所有 bias，借鉴 DeepSeek/PaLM/Falcon，同时减少 FSDP per-parameter padding 通信开销）；BlockNorm（把归一化改为沿特征维固定块（如 256 元素）的 RMS 归一化 `RMSNorm((Wx+b).view(-1,BlockN)).view(B,N)`，等价于无参数 Grouped RMSNorm，可融合进 GEMM epilogue、避免全局同步与 mean-cancellation 误差）；Hard Swish（`h-swish(x)=x·ReLU6(x+3)/6`，替换 sigmoid 型 Swish，消除指数运算、天然适合低精度且可与 BlockNorm 融合进同一 kernel）。
    3. **LoKA Dispatch**（见 kernel 调度层条目）。
  - 实验比较什么：(1) 无质量损失的 FP8 全轨迹训练——Wukong/Interformer/ELFM 相对原始高精度 baseline 的相对 log loss 全程持平（对比直接 TorchAO FP8 训练的 1.3× 变慢 + 2.5% relative log loss 退化）；(2) 分布感知误差——用学习分布 vs 标准正态输入的 MERE 几何均值增高达 15%，并用 LoKA Probe 发现 FBGEMM 生产 benchmark 的 faulty test code（正常/错误实现的 MERE 相差 47×，随机输入下几乎相同）；(3) LoKA Mods 消融——No Bias 单独贡献最大延迟降低，BlockNorm 提供数值调理与融合可能，Hard Swish 较小，三者合并延迟降低 >2× 且实现全轨迹 loss 中性；(4) 端到端加速——训练最高 1.19×、推理 1.4×（相对已用 LoKA Mods 重写的强 baseline），生产部署 5–20% 训练吞吐 / 10–17% 推理加速。
- 硬件平台是什么，配置是什么。
  - NVIDIA H100、B200、GB200（NVL72）与 AMD MI300X、MI350X 集群，16–256 张 GPU（Wukong H100 6K batch&32 GPU、B200/MI300X 12K&16、GB200/MI350X 20K&16；Interformer 4K&64/8K&64/20K&32；ELFM 2K&256/6K&128/20K&32）。基准分析实验用 64× H100。LoKA 在 H100/B200/MI300X 上开发验证，后在开发期间不可得的 GB200 NVL72 与 MI350X 上无修改直接评估，得到相当加速比。
- 模型是什么。数据集和bench分别是什么。
  - 模型：三个 SOTA LRM 家族——Wukong（257B 参数、24 GFLOPs/sample，含 FMB/LCB 开源组件 [5]）、Interformer（566B、28，Transformer 交互组件）、External Large FM / ELFM（1343B、40，整合 DHEN/DLRM/SUM/DCN 的复合架构）。数据集：生产级行业规模数据集（数百亿样本、数千特征），非公开；bench 即三家公司的生产广告排序/推荐 ranking 任务与 27 个生产 GEMM shape。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：LoKA 本身未开源（截至 2026-08 联网搜索未找到公开仓库，arXiv:2605.10886 未附代码链接）。依赖均开源：PyTorch（pytorch.org）、TorchRec（github.com/pytorch/torchrec，混合并行/嵌入分片）、TorchAO（github.com/pytorch/ao）、DeepGEMM（github.com/deepseek-ai/DeepGEMM）、FBGEMM（github.com/pytorch/FBGEMM）、torch.compile；模型参考 Wukong（arXiv:2403.02545）、Interformer（arXiv:2411.09852）、ELFM（arXiv:2502.17494）。
  - 算法 pipeline 例子（以 Wukong 一个线性层 FP8 训练为例）：
    ① **Probe 在线分布学习**：该层激活 X∈R^{B×K}（batch 独立，只建模特征维）：流式合并统计量 n_new=n_old+B、δ=μ_b−μ_old、μ_new=μ_old+(B/n_new)δ、Σ_new=Σ_old+S_b+(n_old·B/n_new)δδᵀ（S_b 为 batch scatter），样本协方差 Σ=Σ_new/(n_new−1)，FP32 累积；权重 W∈R^{M×N} 建模矩阵正态：每 minibatch 解 L_VL_Vᵀ=V+εI 得 U'=(1/N)(W_c L_V⁻ᵀ)(W_c L_V⁻ᵀ)ᵀ，解 L_UL_Uᵀ=U+εI 得 V'=(1/M)(L_U⁻¹W_c)ᵀ(L_U⁻¹W_c)，EMA 平滑 U''=mU+(1−m)U' 并对称化+正则化，trace 重归一化 s=trace(U)/M、U←U/s、V←sV。
    ② **离线采样与误差量化**：激活 T'=1_Bμᵀ+ZL_Σᵀ（Z~N(0,I_K)，L_Σ 为 Σ+εI 的 Cholesky 因子）；权重 W'=M+L_UZL_Vᵀ。对每层采样 100 对输入-权重，跑 FP8 kernel vs TF32 参考，按 MERE=Σ_mΣ_n|out−ref|/|ref| 量化误差，MERE 高或加速比低的层标记为低精度不安全。
    ③ **LoKA Mods 重写模型**：移除 bias → `out = RMSNorm(xW.view(B, BlockN)).view(B, N)`（BlockN=256，按块独立算 RMS、块内融合激活与(反)量化，训练/推理同块大小保证一致性）→ `out = out * ReLU6(out+3)/6`（Hard Swish）。
    ④ 张量计算实例：输入 x∈R^{2048×256}、权重 W∈R^{256×768} 的 GEMM → FP8 量化（按选定 recipe：tensorwise/rowwise/blockwise scaling，scales 取行/块内最大绝对值）→ FP8 张量核乘加（快速累加 FP32）→ dequantize + BlockNorm(256) + Hard Swish 融合在 epilogue 完成 → 输出，与 TF32 基线比较 MERE 判定是否安全低精度。


## LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Chebyshev 分段逼近算法（Python，PiecewiseChebFitter），为任意用户给定输入范围 [a,b] 的非线性函数生成 {breakpoints, 每段多项式系数, 次数} 配置，供 XCore 硬件执行。算法流程（Fig.3）：①采样（curvature-based：先均匀采样，数值微分估曲率，高曲率区插入更多样本点）；②无分段整体逼近，考虑最大项数（决定允许的多项式次数）、函数奇偶性（代数性质：偶/奇函数只用偶/奇阶项，6 项下非对称函数支持 5 次 x^0~5、奇函数支持 9 次 x^1,3,5,7,9，同项数更高次数→更高精度与更好数值稳定性）、定点硬件约束 Q_(m,n)^max（不等式 |p_i||x^i|_max<|Q_(m,n)^max| 并入最小二乘求解防溢出）；③从 2 段起逐段递增到最大支持段数 N，每段：区间变换 x'=(2x-(b+a))/(b-a)→[-1,1]（Chebyshev 自然定义域）→ 构造 Chebyshev 矩阵 V（第 j 列=T_j(x_i')）→ 最小二乘 min Σ_i(f(x_i')-Σ_j c_j T_j(x_i'))² 解系数 → 用递推 T_0=1,T_1=x',T_n=2x'T_{n-1}-T_{n-2} 把 Σc_iT_i(x') 展开回标准多项式 Σp_i'x'^i 再逆变换得 Σp_ix^i；④用遗传算法（个体=各段次数分配 k_seg1..k_#seg，设 #gen=10、#pop=16）优化次数分配防高次过拟合，breakpoint 由三种分段策略确定（见下）；选平均 MSE 最小的个体/段数作为最优解，输出 breakpoints+系数+次数存入 XCore LUT（Fig.3 红色参数）。算法也可探索 >6 项多项式（由多个 XCore 计算）。
  - 三种分段策略（图 3/图 4）：(1) uniform——等宽分段；(2) curvature-based——按累计曲率 W_k=Σ_{i=1..k} κ(x_i)Δx 均分，找 breakpoints 使每段累计曲率 = W_m/N，高曲率区段更密；(3) equal-error——从 curvature 分段出发迭代优化，新 breakpoint x_e 在 [x_(s-1), x_(s+1)] 间用 Brent 法 [9] 解 MAE(x_(s-1),x)-MAE(x,x_(s+1))<ξ（ξ=容差，默认 1.5e-5）使左右段 MAE 接近，直到各段 MAE 方差低于阈值，使误差分布更均匀、接近最优精度。
  - 实验比较什么：(1) 算法级——遗传算法设置（#gen、#pop、ξ 容差）对逼近 runtime 与最终精度的影响：ξ 越小误差分布越均匀、越接近近优但 runtime 越大；迭代越多越好但有收益递减；穷举 6 段×6 项=6^6 次分配一周内不可行 → 遗传算法更合适。(2) 单元级精度——XCore-A/B/C 与 LoRA-SW（软件逼近结果）在 Sigmoid/Tanh/GELU/Swish/Softplus/arcsinh/Sin/sqrt（±8、±4、[0,63.75]、[±π/2] 等范围）上与先前工作 [4][10][25][32][76][79][80] 比 AAE/sq-AAE/MSE：XCore 逼近误差与软件 LoRA-SW 同数量级（sin 因 log 转换器精度 ~1e-5 上限例外）；奇偶性使 cos 的 MAE 降 148×；与 PACE 比（同为 Chebyshev 分段逼近）：XCore 充分利用代数性质+equal-error 更稳定，3-term XCore 在 EfficientNet/MobileNetV3 上误差 0.002%/0.006% < PACE 最小 0.01%。(3) 端到端精度——DCT（cos，量化步长 0.1/0.75/1.5，比 MSE/PSNR/压缩率，baseline=AMD Ryzen 9 7945HX，LoRA 与 PICACHU-4th 持平或更好，故后续 Taylor 阶数取 ≥4）、DNN（SE-ResNet/EfficientNet/MobileNetV3 在 ImageNet，baseline=RTX 4090 FP32 输出，LoRA 精度差 ≤±0.008%，部分反超 baseline）、LLM（GPT2-XL/Mistral-7B/Mistral-7B-v0.3/DeepSeek-7B 在 ARC-e/HellaSwag/CommonsenseQA/COPA，baseline=A100 FP32 输出，Q16,16 定点，LoRA 精度差约 ±1% 内且部分任务反超 baseline；HardSwish 无指数但含多项式也由 XCore 直接算）。
- 硬件平台是什么，配置是什么。
  - 算法级/单元级/端到端精度评估为软件：DNN 用 NVIDIA GeForce RTX 4090 GPU，LLM 用 NVIDIA A100 GPU，DCT baseline 用 AMD Ryzen 9 7945HX CPU（FP32）。硬件目标平台是 LoRA CGRA SoC（Chisel 建模、TSMC 40nm 综合、~475MHz）；XCore 为 40nm、~510/485MHz、6 段（XCore-C）/7 段（XCore-A/B）配置，FP32+可编程定点格式。
- 模型是什么。数据集和bench分别是什么。
  - 模型：DNN——SE-ResNet、EfficientNet、MobileNetV3（ImageNet 数据集，FP32 与 Q16,16 定点）；LLM——GPT2-XL、Mistral-7B、Mistral-7B-v0.3、DeepSeek-7B（激活函数 GELU/Softmax/Swish）。数据集：ImageNet（DNN）；ARC-Easy、HellaSwag、CommonsenseQA、COPA（LLM 的 4 个 NLP 任务）。bench：8 个非线性函数（Sigmoid/Tanh/GELU/Swish/Softplus/arcsinh/Sin/sqrt）用于单元级；8 个系统级 benchmark（Mish、Logsigmoid、Softmax、Softplus、Swiglu、Svm、DCT、KNN，共 11 个 loop kernel），激活函数 benchmark 源自 CGRA-Nonlinear-Benchmark [55] 与 PICACHU [56]。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：算法部分开源（COFFA 仓库 LoRA-ISCA-AE 分支 https://github.com/Dai-dirk/COFFA/tree/LoRA-ISCA-AE，Docker docker.cnb.cool/fudaneda/docker/chipyard 内 /root/PiecewiseChebFitter，Zenodo DOI https://doi.org/10.5281/zenodo.19447155，BSD 3-Clause）。使用例子（算法级复现）：`cd /root/PiecewiseChebFitter && python3 run_func.py <gelu|sigmoid|softplus|swish|tanh>`，每函数配置在 config 文件夹，结果输出到 fig/ 与 result/ 文件夹。
  - 算法 pipeline 伪代码：
    ```
    # 输入: 目标函数 f(x), 区间 [a,b], 最大项数(默认6), 最大段数 N, 定点上限 Q_max
    # 1. 采样（curvature-based）
    {x_i} = uniform_sample(a, b, m)
    κ(x_i) = 数值微分估计曲率; 在 κ 大的区域插入额外样本 → {x_i, f(x_i)}
    # 2. 无分段整体逼近（决定每段允许的最大次数）
    x' = (2x - (b+a)) / (b-a)                    # 区间变换 → [-1,1]
    若 f 为奇函数: V 的偶数列置 0 (如 [0,T1,0,T3,0,T5])   # 奇偶性 → 同项数更高次数
    # 3. 分段搜索
    for seg in 2..N:
      breakpoints = 分段策略(seg)   # uniform | curvature (累计曲率 W_m/N 均分) | equal-error (Brent 迭代)
      best_deg = 遗传算法(个体=各段次数分配 k_seg1..k_segN, #gen=10, #pop=16)
      对每段: min_c Σ_i (f(x_i) - Σ_j c_j T_j(x_i'))²  s.t. |p_j||x^j|_max < Q_max   # 最小二乘+约束
      记录平均 MSE
    选平均 MSE 最小的 (段数, breakpoints, 系数, 次数)
    # 4. 输出: breakpoints, 每段多项式系数 c_i/log2(c_i) 与次数 k_i → 存入 XCore LUT
    ```
  - 张量计算例子（单段 6 项逼近）：采样 m 点后构造 6 列 Chebyshev 矩阵 V（第 j 列 = T_j(x_i')，按递推 T_0=1、T_1=x'、T_n=2x'T_{n-1}-T_{n-2}），解最小二乘 V·[c_0..c_5]^T ≈ [f(x_i')]^T 得系数 c；再把 Σc_iT_i(x') 展开为标准多项式 Σp_i'x'^i，经逆区间变换 x=(x'+1)(b-a)/2+a 得 Σp_ix^i。XCore 执行时每项 c_i·x^(k_i) 在 LNS 中算为 2^(log2(c_i) + k_i·log2(x))（log 转换 → 30b×6b 乘法（5 项并行，乘法器分解 30b 为 5×6b）→ 反log），Output 级加法树（CPA 可配定点/浮点加）求和，7 cycle 输出。示例：tanh(x)+1 或 sin(x)+cos(x) 等复合函数作为单一 XCore 节点直接逼近，无需分解。

## MC-ORAM: A Mask-Assisted and Counter-Based Non-Deterministic ORAM inside VM-Based TEEs

- 属于算法pipeline的实现是什么？实验比较什么？
  - （近似分层：本论文是 TEE 内 ORAM 的非确定性安全机制，不属于模型推理加速类算法；此处从"提出新的算法模型"角度记录）实现为 MC-ORAM 算法——面向 VM-based TEE（Intel TDX/AMD SEV-SNP）的掩码辅助+计数器非确定性方案。核心：把 TME 的确定性 AES-XTS 加密（同一物理地址同一明文产生同一密文，产生密文侧信道）转化为非确定性。每个 128 位 AES 块重排为 112 位 masked data + 16 位 counter；同一 ORAM 树节点或暂存（stash）内的所有 112 位块共享同一个 112 位随机掩码（不同物理位置由 AES-XTS 的地址 tweak 天然区分）。每次访问对 112 位数据做掩码写（data⊕mask）、16 位计数器 +1；计数器到达 2^16−1 即将溢出时执行 Refresh 算法（对整个节点/暂存重新生成随机掩码、所有计数器清零）。算法共 6 个：Algorithm 1 初始化（每节点/暂存生成 node.mask/stash.mask、计数器清零、D[i]⊕mask 写入）、Algorithm 2/3 读路径+TreeToStash（wrMask 条件写+全暂存计数器递增）、Algorithm 4/5 驱逐+StashToTree（反向，同时更新树节点与暂存计数器）、Algorithm 6 Refresh。刷新频率与访问模式无关：树节点刷新期望仅 3.05×10^−5 次/访问，暂存每 2^16/(2ZL) 次访问刷新一次（N=2^14 时每 585 次、N=2^20 时每 409 次、N=2^23 时每 356 次），摊销开销 <1% 运行时间；密文非确定性概率 1−2^−112。带宽开销仅 baseline 的 1.125×（对比 64 位交错计数器方案的 2×），存储减少 43.75%。
  - 实验比较什么：MC-ORAM vs 采用 64 位交错计数器（Obelix 风格，每 64 位数据配 64 位计数器）的 PathORAM/PathORAM+/RingORAM/RingORAM+ baseline（+ 表示采用 Oblix 的暂存优化：路径读只把目标块放入暂存，每 3 次访问额外驱逐一次防溢出），指标为平均访问延迟（1 百万次访问算术平均，含递归位置图查询+readPath+驱逐）与带宽/加速比。结果：MC-ORAM 访问延迟 0.87–40.08ms（PathORAM baseline 1.48–72.87ms），最高加速 1.82×；MC-ORAM+ 0.11–5.00ms vs 0.19–8.66ms，最高 1.77×；RingORAM 侧 MC-ORAM 0.42–19.05ms vs 0.78–33.08ms 最高 1.85×，MC-ORAM+ 0.10–3.44ms vs 0.16–5.83ms 最高 1.60×。另做：① 计数器位宽消融（4/8/16/32/64 位，16 位最优，8 位因刷新过频略慢）；② 访问模式不变性（LS 线性扫描/均匀随机/高斯/重复访问 RA 四种模式延迟几乎一致，N=2^14、B=256B）；③ 纯 masking baseline 对照（N=2^14、B=256B 时 38.4ms/访问，比 64 位计数器慢 13.5×）；④ N=2^14/2^23 × B=cacheline(512b)/256B/2048B(embedding) 全组合；⑤ SPEC CPU2017 九个 benchmark 映射（表 VI）；⑥ DLRM/Qwen-8B 安全嵌入端到端（表 VII）。
- 硬件平台是什么，配置是什么。
  - 单服务器：双路 Intel Xeon 6548Y+ CPU、512 GB DDR5 DRAM；host 与 guest 均 Ubuntu 22.04.5；guest 运行在隔离的 VM-based TEE（Intel TDX）内，TME 硬件 AES-XTS 作为唯一加密机制（ORAM 树/暂存直接放 TEE 加密内存，省去客户端额外重加密）。无 GPU/专用加速器参与。
- 模型是什么。数据集和bench分别是什么。
  - 模型/数据集：DLRM（深度学习推荐模型训练，N=2^23、B=2048B）与 Qwen-8B（LLM 推理、time-to-token，N=2^18、B=16384B）的嵌入表安全访问评估，方法学沿用 LAORAM（TEE 内 ORAM 作安全嵌入表管理器），报告"无 ORAM 执行基线延迟+ORAM 查找延迟"之和：Qwen-8B 13→25.8ms/token（1.41×）、DLRM 0.17→3.61ms/input（1.66×）。bench：SPEC CPU2017 的 9 个 benchmark（表 VI 列 povray/mcf/leela/blender/omnetpp/parest/x264/sjeng），用 Intel PIN 采集每个 benchmark 5 百万个连续数据地址轨迹，统计最大唯一缓存行数（工作集），映射到最小可容纳的 ORAM 高度 N=2^16~2^24；ORAM 自身配置：PathORAM/RingORAM Z=4、RingORAM S=3、A=4，暂存 naive 90 / 优化 10；位置图用递归 ORAM（N=2^14 单级 B=32、N=2^23 六级 B=16 每块 4 条目，根位置图 2^11 条存 TEE 内存并配 64 位计数器）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：MC-ORAM 本身未开源（截至 2026-08 联网搜索未找到公开仓库）；实现以两个开源 ORAM 参考为基础：PathORAMSimulator（https://github.com/renling/PathORAMSimulator，论文[33]）与 oram_simulator（https://github.com/wangxiao1254/oram_simulator，论文[41]）；TCB 每实现 <1000 行代码、其中 <200 行为 MC-ORAM 特有（mask/counter/refresh 元数据管理）。
  - 算法 pipeline 伪代码（以 PathORAM+MC-ORAM 一次逻辑访问 d，B=256B=每 ORAM 块 16 个 128 位 AES 块、Z=4、L=log2N 为例）：
    ```
    # 初始化 (Algorithm 1): 每节点 node: node.mask=Rand(); 计数器清零; 112 位数据初始化为 node.mask
    #   stash.mask=Rand(); 位置图 PosM[d]=Rand(); 数据 D[i] 以 D[i]⊕node.mask 写入随机块
    # 读路径 (Algorithm 2/3): 
    P = PosM[d]                              # 1) 查位置图得叶子路径
    for node in 路径P:                        # 2) 逐节点读整条路径到暂存
        for i in 1..Z:
            wrMask = [False]*|stash|
            for j in 1..|stash|: wrMask[j] = !found ∧ stash[j].empty; found ∨= wrMask[j]
            TreeToStash(stash, node[i], wrMask)   # 3) 条件写+掩码
    # TreeToStash 内每个 AES 块:
    #   if stash.ctr==2^16-1: Refresh(stash)
    #   dst = wrMask[j]·(node[i][j]⊕node.mask⊕stash.mask) + !wrMask[j]·stash[j][k].data
    #   stash[j][k]_bits = dst ∥ (ctr+1)          # 即使不写也 +1 保证密文变化
    # 4) 客户端处理数据 d; PosM[d]=Rand()          # 5) 驱逐写回 (Algorithm 4/5, StashToTree 反向, 双更新树与暂存)
    # 驱逐/读路径中若任一 16 位计数器=2^16-1:
    #   Refresh(node/stash) (Algorithm 6):
    #     new_mask=Rand(); 对每个 AES 块: dst = node[i].data ⊕ node.mask ⊕ new_mask; node[i]_bits = dst ∥ 0
    #     node.mask = new_mask
    ```
  - 张量/字节级例子（一次驱逐写回单 AES 块）：树节点旧密文前的明文表示 = (data_112 ⊕ node.mask_112) ∥ ctr_16；从暂存驱逐块到节点时 dst = (data ⊕ node.mask ⊕ stash.mask)，即把"暂存掩码域"转成"节点掩码域"，再拼 ctr+1 后由 AES-XTS(addr, ·) 加密写 DRAM；两次访问同一 (addr, data) 因 ctr 必不同（同掩码周期概率 1、跨周期掩码独立以 1−2^−112 概率不同）→ 加密前 128 位值不同 → 密文不同，消除确定性密文侧信道。


## MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现：**document attention decomposition（文档注意力分解）**——面向 KV 预计算（KV-precomputed）RAG 推理的注意力分解算法。把标准 softmax 注意力模块拆成两个独立分支：DocumentAttention 分支（在持有文档 KV 分片的 PIM 设备上本地执行，对本地 K/V shard 计算注意力）与 QueryResponseAttention 分支（处理用户 query 与已生成 token 的 KV），各自只产出紧凑的局部摘要（未归一化输出 o、局部最大值 m、归一化因子 l），再通过数值稳定的 softmax 全局聚合（基于共享基线 m=max(m_d,m_c)，公式 l=e^{m_d-m}l_d+e^{m_c-m}l_c，o=(e^{m_d-m}o_d+e^{m_c-m}o_c)/l，即 FlashAttention 的 online-softmax 思想 [10]）合并。算法 1（In-Layer Document Attention Decomposition）给出逐 token 流程：QKVProjection → DocumentAttention(q,state_doc) → QueryResponseAttention(q,k,v,state_ctx) → Fusion → LN1 → FFN → LN2。后续所有 transformer 层（LN、FFN、残差）保持不变，因此数值语义与标准 softmax attention 完全一致，无需重训即可保持精度。
  - 通信量对比（本算法的核心收益）：集中式需传输整份文档 KV：V_ce = #Document tokens × 2 × d_model × 2 (bytes)（FP16）；去中心化只需传 query 向量并回收局部摘要：V_de ≈ (#Query tokens + #Response tokens) × 2 × d_model × 2 (bytes)。由于检索文档平均比 query+response 长 ~380×，通信量降低两个数量级以上；文档 K/V 按 attention head 维度分片到 N 个设备后每设备只传 d_model/N 大小的输出切片，跨设备总流量基本恒定。
  - 实验比较什么：端到端吞吐（tokens/s，batch size 2/4/8/16，图 9）、每请求延迟（图 10）、通信/prefill/decode 三阶段延迟分解（图 11）、能量效率（图 12）、准确率（图 13，与 CPU-GPU baseline 差 <0.4pp）、与 HeterRAG 的端到端吞吐/延迟（图 14/15）、组件消融（M-pim/M-non/M-ad/M-ad+ire/M，图 16）、规模扩展（OPT-66B，4→32 设备，图 17）、长 query 通用性（4×/16×/64× 原始 query 长度，图 18）。平均吞吐提升 5.36×/6.64×/3.98×/3.32×/3.91×、延迟降低 4.30×/5.34×/3.31×/2.73×/2.79×（对 TurboRAG/BlockAttention/CENT/PAPI/HeterRAG）。
- 硬件平台是什么，配置是什么。
  - MERIDIAN 为仿真评估：32 个 PIM 设备（默认 16 个 Document Attention Cluster DAC + 16 个 Context Execution Cluster CEC），每个设备 512 GB 容量、32 TFLOPS 峰值；CXL 3.0 over PCIe Gen5 ×16（每链路 128 GB/s），端到端 CXL 访问延迟 165ns（25ns 端口 + 10ns retimer + 70ns switch + 60ns 内存控制器/DRAM）。内存：LPDDR5X，64 GB/package，8.5 Gb/s/pin，×128 channels，t_RC=60/t_RAS=40/t_CL=23/t_RP=20/t_RCDRD=17/t_RCDWR=8；外部带宽 1.1 TB/s、内部带宽 16 TB/s。PIM Unit：16 FP16 比较器 + 16 FP16 乘法器 + 16 FP16 加法器 + 4 KB 双缓冲 buffer；Controller-Side Unit：1 个加法单元（16 FP16 加法器）+ 1 个软max单元 + 8 个 BOOMv2 RISC-V 核。每 DRAM bank 配 16-lane PU 运行于 1 GHz，32 设备共提供 16 TB 容量。
  - CPU-GPU baseline 平台：Intel Xeon Gold 6454S + 1 TB DDR5 + 4× NVIDIA H100（每卡 80 GB HBM2e），GPU 能耗用 nvprof 测量。
- 模型是什么。数据集和bench分别是什么。
  - 模型（两个微调 RAG 模型）：① Qwen-TB，基于 Qwen2-7B [53] 用 TurboRAG [44] 微调，28 层、28 heads、hidden 3584；② Tulu3-Block-FT，基于 Llama-3.1-Tulu-3-8B-SFT [3] 用 BlockAttention [46] 微调，32 层、32 heads、hidden 4096。可扩展性实验另用 OPT-66B（64 层、72 heads、hidden 9216）。默认 batch size 8。
  - 数据集：2WikiMultiHopQA (2Wiki) [18]、HotpotQA (HQA) [63]、Natural Questions (NQ) [30]、TriviaQA (TQA) [26]。平均 token 长度：2Wiki Doc 856.76/Q 17.60/Resp 3.03；HQA 1341.04/20.41/3.97；NQ 14630.04/10.28/4.36；TQA 14748.69/18.57/4.59。与 HeterRAG 对比时采用 HeterRAG 的检索管线：AccelDIMM 加速器 + HNSW 索引 + 完整 Wikipedia 语料，模型用 Tulu3-Block-FT。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：MERIDIAN 未开源（截至 2026-08 联网搜索未找到公开仓库，ISCA 2026 论文，后续可能随 artifact 发布）。论文采用 PIM-SYCL 式异构编程模型，与 CUDA 类似，高层面 API 暴露 GEMV、GeLU 等操作与设备初始化、并行策略选择；编译为底层指令（PIM 计算命令 PIM_MAC/PIM_CMP/PIM_EW_MULT/PIM_EW_ADD 与数据移动命令 PIM_ACT/PIM_WR_PB/PIM_RD_PB，标准 load/store 经 CXL.mem）派发到设备控制器，控制器广播 PIM 指令到相关 channel 和 PU。
  - 算法 pipeline 伪代码（单设备持有文档 KV (K_d,V_d)、另一设备持有上下文 KV (K_c,V_c)，给定 query 向量 q）：
    ```
    # 1) 两个分支独立计算注意力 logits（GEMV）：
    s_d = q @ K_d^T ;   s_c = q @ K_c^T
    # 2) 各分支本地归一化基线（max）：
    m_d = max(s_d) ;    m_c = max(s_c)
    # 3) 各分支形成未归一化输出与归一化因子（累加，GEMV-like）：
    o_d = Σ_j exp(s_d[j]-m_d) * V_d[j] ;  l_d = Σ_j exp(s_d[j]-m_d)
    o_c = Σ_j exp(s_c[j]-m_c) * V_c[j] ;  l_c = Σ_j exp(s_c[j]-m_c)
    # 4) 全局融合（共享基线 m=max(m_d,m_c)，数值稳定）：
    l = exp(m_d-m)*l_d + exp(m_c-m)*l_c
    o = ( exp(m_d-m)*o_d + exp(m_c-m)*o_c ) / l
    # 5) 下游：x ← LN1(x + o)；f ← FFN(x)；y ← LN2(x + f)
    ```
  - 张量计算例子（一次解码 step，d_model=3584、Qwen-TB、单头场景）：只传输当前 query 向量 q（d_model×FP16 ≈ 7KB）进内存，文档侧 K_d（如 1.4 万 token × 3584，FP16 ≈ 96MB）保持静止在 PIM 设备内；设备本地算 s_d = q·K_d^T（1×14630 GEMM 退化为 GEMV），在线 softmax 输出 o_d、m_d、l_d 三个紧凑量（o_d 为 d_model 向量 + 2 个标量），仅此经 CXL 返回 host/聚合层。对比集中式需把 96MB 文档 KV 从 host 搬到设备再算注意力。批量/多头时 K/V 按 head 分片到 N 个设备，各设备只返回 d_model/N 的输出切片，跨设备聚合流量与设备数无关。

## MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现：**语义感知傅里叶压缩（Semantic-Aware Fourier Compression, FFT-CMP）+ 分层蝴蝶分解（Hierarchical Butterfly Decomposition, hierarchical BSMM）** 构成混合化 Transformer block（hybridized transformer block）。(1) FFT-CMP：观察到 LLM 层沿序列维呈现语义频率局部性（浅层偏高频细粒度 token 细节，深层偏低频上下文抽象，经 Llama2-7B Q/K/V 频谱验证），对投影后的 Q,K,V ∈ R^(N×D) 按每层 chunk 长度 L（L=N/f_H 取 power-of-two，f_H 为能量超过相对阈值的高频谱峰）分块，每块做 L 点 FFT→截断保留前 sL 个低频系数→sL 点 iFFT 生成缩短 token 表示（丢弃低能量高频分量，s 为可调压缩率）。prefill 代价从 O(N²D) 降到 O(s²N²D)，附加 chunked-FFT 开销仅 O(ND log L)。decode 侧用固定 L 的 append-only chunk-granular KV cache：新 token 累积到 L 才触发一次 FFT 压缩并追加新压缩块，已完成的 chunk 复用缓存压缩块，兼容 KV-cache 解码。(2) hierarchical BSMM：把权重矩阵 W 分成 (D/B)×(D/B) 个 B×B tile，仅在每个 tile 内应用蝴蝶因子，参数/计算复杂度从全局分解 O(D log D) 降到 O((D²/B) log B)；B 是第二个可调精度-效率旋钮（论文评估 B∈{16,32,64}，B=32 最佳）。两者结合在序列维 N 与隐藏维 D 两个正交维度暴露并行性。
  - 实验比较什么：(a) 算法精度-计算折中：FFT-CMP（s=0.5）在 ViT 上达到 65% FLOP 削减仅 1.6% 精度下降，优于 FNet 式 2D-FFT（同 FLOP 削减但 2-3% 精度损失）；BERT 上 k 层渐进应用（s=0.5）替换全部 12 层达 69% FLOP 削减、仅 1.75% EM 与 1.3% F1 损失；Llama2-7B/InternLM2-7B 用 LoRA 微调压缩层，>60% 层应用后 s=0.75 削减 57-64%、s=0.5 削减 67-72% 的 QKV+Attention 计算，总体精度下降 <1.45%（Winogrande-xl/Wikitext-2/103/Ada-LEval）。(b) H100 上对比原模型 eager attention 与 FlashAttention2（保守设置 s=0.5,B=32）：prefill 长序列最多 2.72×（vs eager）/1.64×（vs FA），短序列收益小（因 FFT-CMP 在 PyTorch 层实现、未与 FA 融合，TensorCore 对蝴蝶稀疏支持有限、回退到 CUDA core）；decode 结合块 BSMM 达 1.4-1.9× 端到端加速。
- 硬件平台是什么，配置是什么。
  - 算法验证运行平台：NVIDIA H100 GPU（prefill N=512/8K，cuFFT 优化的 roofline 分析）；NVIDIA AGX Orin（batch 64，N=512/8K，FFT-based transformer block 相对 dense baseline 仅 3.77×/2.56× 加速）；Jetson Xavier（12nm、1.7 TFLOP/s CUDA 峰值、6 TFLOP/s TCU 峰值、15W，16 GB 内存）；RTX-3090（structured-workload sweep）。FFT 用优化 cuFFT；kernel 以 PyTorch 层实现。
- 模型是什么。数据集和bench分别是什么。
  - 模型：ViT（196/1K patch，从头训练验证蝴蝶稀疏理论）、FABNet（128/768）、BERT（8K/1K，EM/F1 指标）、BERT-SQuAD（B0，512/1K）、InternLM2-7B（GQA，1K-4K/4K）、Llama2-7B（128-2K/4K）。数据集与 bench：Winogrande-xl（N=512）、Wikitext-2/103（1K/2K）、Ada-LEval（1K/2K/4K）；LLM 精度评估用 LoRA 微调（QA-LoRA 式）压缩层。对比基线：FNet 式 2D-FFT token mixing、dense Transformer、全局蝴蝶分解（Monarch/butterfly factorization）等。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文（ISCA 2026，ICT CAS，Best Paper Candidate）未提供任何开源代码或仓库（联网搜索未找到）；评估以 PyTorch 层实现（FFT 用 torch.fft，BSMM 可用 torch 矩阵分块实现）。算法 pipeline 伪代码（一次 prefill 的注意力阶段，chunk 长度 L、压缩率 s）：
    ```
    # 1) 逐层确定语义 chunk 长度：f_H = 最高能量超过阈值的高频谱峰；L = Pow2Round(N/f_H)
    # 2) FFT 压缩（对每个 chunk，沿序列维）：
    for each chunk c in range(N//L):                 # N/L 个 chunk
        F_c = FFT_L(Q[c*L:(c+1)*L, :])               # L 点 FFT，每特征维
        F_c_trunc = F_c[:s*L, :]                     # 保留前 sL 个低频系数
        Qs[c] = IFFT_{sL}(F_c_trunc)                 # sL 点 iFFT，得缩短 token 表示
    # K、V 同样处理；压缩后序列长 sN，注意力矩阵降为 sN×sN
    # 3) 注意力在缩短序列上执行：Attn = softmax(Qs·Ks^T/√d)·Vs
    # 4) QKV/FFN 投影用 hierarchical BSMM（B×B tile 内蝴蝶分解）：
    #    W → 分 (D/B)×(D/B) 个 tile；每 tile W_b = ∏_{k=1}^{log2 B} B_B^(k)（B×B 蝴蝶因子）
    #    输出 Y = X @ W 等价于逐 tile 做 B×B 蝴蝶稀疏矩阵乘，复杂度 O(D²/B·log B)
    # 5) decode：新 token 累积到 L 才 FFT 压缩一次并 append 新块到压缩 KV cache
    ```
    张量计算例子（Llama2-7B、N=2048、D=4096、L=256、s=0.5）：Q∈R^(2048×4096) 重塑为 8 个 256×4096 chunk，每 chunk 做 256 点 FFT、保留 128 个低频系数、128 点 iFFT，得 8×128×4096=1024×4096 的缩短 Q，注意力矩阵从 2048×2048 缩到 1024×1024（4× 缩减）；FFN/QKV 投影的 W 按 B=32 分 tile 做蝴蝶分解，复杂度 O(D²/B·log B)=O(4096²/32·5)。

## MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 算法级设计（服务于 GPU kernel 实现，详见实验_kernel调度条目）：(1) **BSK 分块跨密文密钥复用**——利用 External Product 中 BSK 与傅里叶系数是逐元素 Hadamard 积这一性质，把"一个线程块需持有整个 BSK"的朴素缓存策略（A100 上 BSK 可能超共享内存上限，且过度分配共享内存会蚕食与 L1 共享的物理容量）改为分块分解：单个线程块只处理一块 TBSK 对一块 TGLWE，使同一 BSK 分块被一批中多个 PBS 实例复用（从 L2 级复用提升为 SM 级复用），并定义分块几何为 8 个连续复数 FP64 元素（128B）保证全局内存合并访问、不改变 FFT 输出数据布局（避免显式数据重排的额外开销）。(2) **跨迭代 FFT/IFFT kernel 融合**——识别 IFFT 与 FFT 使用同一 twiddle factors / precomputation factors 系数集的共轭版本，构造跨越盲旋转迭代边界的融合 kernel，把两套系数在片上跨迭代复用，消除主循环内冗余全局载入；收益随 ℓ 增大（分解层数越多，融合窗口越宽）。(3) **Tensor Core FFT 映射**——8 点 FFT 基例（匹配 WMMA 8×8×4）、四步 FFT radix-8/64 分层分解、Fourier 矩阵运行时片上生成（{0,±1,±sin(π/4)} 寄存器内算术生成，替代共享内存载入）、宽数据类型 swizzle、64 点 fragment 布局免转置。(4) **精度分析**——噪声模型 n·2^ω·ℓ·2^(2β)·N²·(k+1)（ω≈2·(64−53)−2.6，假设 64-bit 密文空间）下，整数部分按溢出概率 2^−64 统计定宽、小数部分按输出近似噪声对照 FPT 理论界（图 10）：4-bit 明文、DeepCNN 设定、64-bit 密文空间下正确性要求至少 30 小数位（常需 >35），FP32（24 尾数位）与 FP16（11 尾数位）不足 → 确定 FP64 为 TFHE FFT 的精度底线，并讨论 ℓ↑/β↓ 与低位宽分解作为未来降低精度需求的方向。
  - 实验比较：消融逐项量化算法贡献（图 11：+MAC 单独 1.10×~1.77×，k 大时收益最显著，因 BSK 内存足迹随 k 占主导；+FFT 到完整版本）；参数敏感性（图 16：k 增大→MAC 贡献增大；ℓ 增大→MAC 与 FFT 同时受益；N 增大→FFT 占比增大）；精度分析对照 FPT 理论噪声界验证 FP64 必要性（图 10，Para-C/D/F）。
- 硬件平台是什么，配置是什么。
  - NVIDIA A100 80GB PCIe（108 SM、40MB L2、每 SM 192KB 合并 L1/SPM、FP64 Tensor Core WMMA 8×8×4）与 NVIDIA H100 80GB PCIe；双路 Intel Xeon Platinum 8558P（96 核）。FP64 比例 1/2（数据中心 GPU）vs 消费级 1/64，FP64 Tensor Core 自 A100 起为旗舰标配。
- 模型是什么。数据集和bench分别是什么。
  - 加密推理模型（Concrete-ML 构建，均为 quantized 神经网络，4-bit plaintext，线性层位宽超限时由 Concrete 编译器自动插入 PBS 取整恢复）：DeepCNN-X（[28] 风格，输入 8×8×1，3×3 卷积 2 filters + 3×3 卷积 92 filters stride 2 + X 个 1×1 卷积层各 92 filters（X∈{20,50,100}）+ 2×2 卷积 16 filters + FC 10 输出）、VGG-9 于 CIFAR-10（32×32×3，六个 3×3 卷积 64/64/128/128/256/256，第二/四卷积后 2×2 average pooling，三个 FC 512/512/10）、AES-128（128-bit 明文块，192 个独立输入，bit-level 工作负载）；另含纯 PBS 吞吐 benchmark（batch 4096，Para-A~F 及 Para-I/II）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：MNEMOS 未开源（论文无代码链接，联网搜索 2026-08 未找到公开仓库）；baseline 与执行栈开源：TFHE-rs（https://github.com/zama-ai/tfhe-rs）、Concrete（https://github.com/zama-ai/concrete）、Concrete-ML（https://github.com/zama-ai/concrete-ml）、tfhe-aes-128（https://github.com/sharkbot1/tfhe-aes-128）。
  - 算法 pipeline 张量计算例子（Para-A：N=512、k=4、ℓ=1、β=23；一批 4096 个密文）：(a) Tangent FFT（negacyclic 卷积）：输入系数向量 a（长度 512），构造 b_j=(a_j−i·a_{j+256})·ω^j（ω=e^{−iπ/512}），做 256 点复数 FFT → 正向 TFFT；逆变换 b=(IFFT[c])*、a_j=Re(b_j·ω^j)、a_{j+256}=Im(b_j·ω^j)。(b) MAC 分块复用：BSK 形状 (k+1)ℓ×(k+1)=(5×1)×5，傅里叶域外积 ACC_fourier ⊙ BSK 为逐元素 Hadamard 乘——把一个 BSK 分块（8 个连续复数 FP64=128B）读入线程块，与同一分块位置上的一批 GLWE（TGLWE）复用相乘，多线程块并行覆盖不同分块，跨密文共享同一份 BSK 读（BSK 读取体积从 (k+1)×GLWE 降为可忽略）。(c) Tensor Core FFT：复数矩阵 A=Ar+iAi 与 B=Br+iBi 的乘分解为 AB=(ArBr−AiBi)+i(ArBi+AiBr)，每个实数矩阵乘是 2 次 FP64 WMMA（8×8×4）；8 点 FFT 用片上生成的 Fourier 矩阵（元素 ∈{0,±1,±sin(π/4)}）直接矩阵乘实现，N=256 由四步 FFT 递归 radix-8/64 分解到基例。(d) 精度分析：固定点替换 FP64 后，Para-C/D/F 在 64-bit 密文空间测得的输出近似噪声 vs 小数位曲线对照 FPT 界，30 位以下超出理论界即解密失败，证明 FP32/FP16 Tensor Core 直接映射不可行。

## MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 MXFFP，一种可微缩的灵活浮点（数值量化）格式，扩展 OCP MXFP/MX 格式：每个 block 增加 1-bit configuration 字段，在 E1Mx（分辨率优先）与 E2Mx（范围优先）两种 exponent-mantissa 配置间选择（inter-block 优化）；大 block 拆分为 32 元素 sub-block，各自独立选配置（intra-block 优化）；配套低开销运行时转换算法（Algorithm 1，基于相对指数统计近似 oracle 配置选择）。实验比较：数值精度（WikiText-2 perplexity + ARC-easy/ARC-challenge/Lambada 三个 zero-shot 任务）在 8/6/4-bit 下对比 BF16 baseline 与 MXFP（OCP 规范固定 E4M3/E5M2/E2M1），覆盖 7 个 LLM 及 ViT-base/ViT-large；block size 扩展性（32~256）；以及 vs M-ANT、BitMoD、Microscopiq、MX+ 的 perplexity 与 memory 权衡。
  - 硬件平台：数值精度评估在软件侧做 post-training quantization（论文未明确说明 GPU 型号）；硬件性能用 Accel-Sim（RTX 5090 派生配置）+ CUTLASS GEMM kernel trace；硬件成本用 RTL + Synopsys Design Compiler + FreePDK 45nm 综合；能量用 AccelWattch。
  - 模型：Llama3-8B、Llama2-7B、Mistral-7B-v0.3、Deepseek-llm-7b-chat、OPT-6.7B、Qwen2.5-14B、Vicuna-13B，以及 ViT-base/ViT-large。数据集和 bench：WikiText-2（perplexity）、ARC-easy、ARC-challenge、Lambada（zero-shot accuracy）、ImageNet（ViT Top-1）。
  - 开源情况：MXFFP 论文未提供代码链接，联网搜索（2026-08）未发现同名公开仓库，开源状态无法确认。基础规范开源：OCP Microscaling Formats (MX) Specification v1.0（https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf）；相关开源实现：OpenXLA/StableHLO 的 MX 类型（f4E2M1FN、f6E2M3FN、f8E8M0FNU，https://github.com/openxla/stablehlo）、ggml/llama.cpp 的 MXFP（https://github.com/ggml-org/llama.cpp PR #20609）、NVIDIA CUTLASS（https://github.com/NVIDIA/cutlass）。
  - 算法 pipeline 张量计算例子（MXFFP4 运行时激活转换，Algorithm 1）：输入一个 block（或 sub-block）x={x_1..x_N}（N=32）：(1) 取每元素指数 E_i=exponent(x_i)；(2) E_b^MAX=max_i E_i；(3) 相对指数 E_i^r=E_i−E_b^MAX（恒 ≤0）；(4) 计数 count_E1=|{E_i^r=0}|（E1M2 分辨率更细）、count_E2=|{E_i^r∈{−2,−3}}|（E2M1 额外指数位更表意）；(5) 若 count_E1²>count_E2 选 cfg=E1M2、E_element^MAX=1，否则 cfg=E2M1、E_element^MAX=2；(6) 统一共享指数 E_b^shared=E_b^MAX−E_element^MAX；(7) 逐元素量化 x̂_i=quant(x_i/2^{E_b^shared})；(8) 输出(E_b^shared,cfg,x̂)。大 block（256）拆 8 个 32 元素 sub-block：各 sub-block 独立选配置，但共享一个按 E2Mx 假设计算的指数，E1Mx sub-block 使用时减 1（避免存多个指数）。权重为已知静态数据，离线在 E1M2/E2M1 两种配置下都量化、选 MSE 更小者（=oracle）；激活为运行时数据，用上述计数规则近似 oracle 配置选择，开销可忽略（Fig. 20 显示激活配置选择与 oracle 高度吻合、最终输出 MSE 几乎与 oracle 相同）。

## NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 **NS-FPS 算法**（硬件-软件协同设计的 FPS 加速；本条目覆盖纯 CPU 算法版本，ASIC 部分见实验_硬件架构条目）：利用 Voronoi 图把 FPS 从"全局暴力距离更新"重述为"迭代邻居搜索 + 部分更新"。核心四点：(1) **VD 部分更新**——证明距离缓存 T 的更新只发生在以最新采样点 s_k 为球心、半径 d_k=min_{s_i∈S_{k-1}}||s_k−s_i||² 的球 B(s_k,d_k) 内（该球严格包含真实 Voronoi cell），把每轮更新点数从 O(N) 降到 O(N/k)，总复杂度从 O(N²) 降到 O(N log M)；(2) **Morton 码空间划分**——坐标量化成 15/15/11-bit 整数、取 7/7/3 MSB 交织成 17-bit Morton 码，用桶排序（线性时间，避免 k-d 树的比较排序/建树）重排点云并建索引表；(3) **层次最大缓存**——距离缓存按 16:1 压缩成多级 max 候选（16 点/块），只刷新被更新的块，全局最大值自上而下遍历层次得到，接近对数复杂度；(4) 算法保持与传统 FPS **完全相同的采样结果**（lossless）。实验比较什么：CPU 侧 NS-FPS-CPU vs vanilla FPS（C++17）与 QuickFPS-CPU——16k–120k 点云上相对 vanilla FPS 加速 100.1×/130.3×/106.2×/191.7×，相对 QuickFPS-CPU 在 64k/120k 上加速 1.22×/1.80×，120k 点不同采样率（10%~70%）相对 QuickFPS-CPU 保持 1.23–2.59× 延迟降低；GPU 侧 NS-FPS-GPU vs OpenPCDet CUDA FPS 与 QuickFPS-GPU（小规模下因 Morton 遍历开销劣于 QuickFPS-GPU，大规模下内存缩减主导而反超，说明需 ASIC 才完全释放潜力）；4 个数据集上消融（T1 Morton 邻居搜索单独 12.4–23.4×，T1+T2 完整 31.5–81.7× over GPU）。
- 硬件平台是什么，配置是什么。
  - CPU 平台：AMD Ryzen AI 9 365 CPU（运行 vanilla FPS 与 NS-FPS-CPU 的 C++17 实现，-O3 编译）；GPU 平台：NVIDIA GeForce RTX 3090（运行 OpenPCDet 的 CUDA FPS、QuickFPS-GPU 与 NS-FPS-GPU）；GPU profiling 用 NVIDIA Nsight Systems 与 Nsight Compute。快速背景数据：120k 点 25% 采样在 RTX 3090 上 >900ms、95% 时间耗在内存事务，FPS 在 16k 点输入的点云神经网络中占 30–70% 总运行时间。
- 模型是什么。数据集和bench分别是什么。
  - 模型：FPS 是点云神经网络的前处理算子（PointNet++、PointRCNN、PointConv、3DSSD、IA-SSD 等均依赖迭代 FPS 降采样）。端到端集成实验把 FPS 卸载到 NS-FPS-ASIC、其余卷积/变换层跑在 RTX 3090，评估 PointRCNN、3DSSD、IA-SSD 三个 3D 检测网络：端到端加速 1.3×/1.7×/2.7×（IA-SSD 最依赖 FPS 故提升最大），用 19.2 GB/s PCIe 带宽建模的加速器-主机数据搬运开销 <2%。数据集和 bench：SemanticKITTI（主要基准，120k 点/帧，及 16k/32k/64k 子集）、KITTI（115k）、Waymo Open Dataset（117k）、nuScenes（34k），固定 25% 采样率。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：CPU 版开源，https://github.com/satreeby/ns-fps/（ISCA 2026 companion code，C++17 核心 src/yuezu_fps.h + src/yuezu_fps.cpp + pybind11 封装 yuezu_fps/yuezu_fps_pybind.cpp，另有 eval/、hardware/、example.py、semantickitti_example.py、simple_semantickitti_data/、test.py）。构建：`git clone https://github.com/satreeby/ns-fps && cd ns-fps && pip install -e .`，验证 `import yuezu_fps.yuezu_fps_module as yf`；quickstart 用 `yf.make_range(...)` 建 SpaceRange（每轴 min/max + 2 的幂块数，如 X:32/Y:16/Z:8），`yf.fps(points, n_samples=..., range=space_range)` 返回采样索引；编译宏 MORTON_BLOCK_SIZE/CACHE_BLOCK_SIZE/DEFAULT_X_Y_Z_BLOCKS/BOUNDARY_EPS/INF_DISTANCE 可调；README 声称"up to 191× faster than naive FPS on CPU"、1.72× over QuickFPS-CPU、4.2× over naive GPU FPS。
  - 算法 pipeline 伪代码（对应论文 Algorithm 1）：输入点云 P、采样数 M。(1) 初始化/预处理：按 Morton 码桶排序 P 并建索引表，T←∞，随机选 s_0；(2) for k=1..M−1：(a) d_k=T[s_{k−1}]（当前最远点缓存距离作为搜索半径）；(b) 枚举与球 B(s_{k−1},d_k) 相交的 Morton cube，对其中所有点 p 做 T[p]←min(T[p],||p−s_{k−1}||²)（只有被改进的距离才保留）；(c) 更新 max 缓存层次：对 Buffer_updated 中每块 B 求局部最大 τ_{l+1}^B=max_{p∈B}τ_l^B，并把父块标记进下一层 Buffer_updated，自底向上传播；(d) s_k←argmax_p T[p]，S_k←S_{k−1}∪{s_k}。张量/数据结构：距离缓存 T∈R^N 按 Morton 序存储，每 16 个连续项一组块、以 16:1 逐级压缩成多级 max 候选。搜索半径自适应收缩是剪枝关键：120k 点帧前 100 轮半径覆盖很多 cube、占 27.3% 总迭代时间，随后半径快速降到 1m 以下，每轮 cube/比较/访存数随之骤降。

## NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现是 VD-Zip，两段式 VecDB 压缩：(1) FEE-sPCA（feature-level early exiting with statistics-based PCA，统计 PCA 引导的特征级早退）——离线对向量库做 PCA 变换，使前 k 维集中最富信息的分量，用 α_k 放大、β_k 校正的部分距离 d_est^k = α_k·d_part^k/β_k 估计全维距离 d_all，与阈值（候选优先队列当前最远距离）比较，一旦超过即提前终止距离计算（比纯部分距离早退更早触发，收敛更快）；(2) Dfloat（NDP 感知的动态浮点表示）——把向量沿特征维分成 N_seg 段，每段用不同位宽 1+n_exp+n_man（∈[12,32]），让每个 DRAM burst 装入更多特征。实验比较：同 recall 下 QPS/延迟/能量效率/内存流量，对比 CPU baseline（HNSW、SCANN）、96 核 CPU-HP、GPU CAGRA（A100）、ASIC ANNA、FPGA DF-GAS、UPMEM PIMANN、NDP baseline、SOTA NDP ANSMET；另对比 PQ、RabitQ 的压缩内存流量；并做端到端 RAG（GPT-4o + text-embedding-ada-002）TTFT 与 RAGAS 质量评估。
- 硬件平台是什么，配置是什么。
  - 评估平台：host CPU 为 AMD EPYC 9334（32 核 2.7-3.9GHz，64KB L1/核、1MB L2/核、128MB 共享 L3）；NDP 目标为 DDR5-4800 DIMM，2 或 6 通道、每通道 2 DIMM、每 DIMM 2 rank、每 rank 2 个 sub-channel（各 4 个 8-bit DRAM device），每 sub-channel 一个 VPE 和一个 LNC（256KB LNC-D + 8KB LNC-T），1.2GHz；6 通道配置 48 个 sub-channel，聚合带宽 921.6 GB/s（每 sub-channel 19.2 GB/s）。算法与系统性能用 UniNDP 周期精确模拟器评估，RTL 用 Synopsys Design Compiler 28nm 综合 + Cadence Innovus P&R，FPGA 验证功能，3D-ICE 评估热效应。
- 模型是什么。数据集和bench分别是什么。
  - 面向 RAG/向量数据库的 ANNS 检索（无训练模型；嵌入模型用于构造数据集）。数据集与 bench：SIFT（L2，128 维，1M，10K query）、GIST（L2，960 维，1M，1K query）、BigANN（L2，128 维，1B，10K query）、GloVe（IP，100 维，1.2M，1K query）、Wiki（L2，768 维，1M，10K query，Sentence-BERT 嵌入）、MS_MARCO（L2，384 维，8M，1K query，BGE 嵌入）。HNSW 索引用 NVIDIA cuVS 构建；端到端 RAG 用 GPT-4o + text-embedding-ada-002（1536 维），语料 2WikiMultihopQA、HotpotQA、MultiFieldQA-en、QASPER、MS_MARCO，指标 recall@k、QPS、TTFT、RAGAS。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：代码公开，GitHub https://github.com/Intelligent-Computing-Research-Group/NasZip（Apache-2.0 许可，README 实际标注 MulanPSL-2.0），Zenodo 存档 DOI 10.5281/zenodo.19453078。仓库含 simulate/、sim_tools/（NDP 模拟）、preprocess_idx/、idx_tools/（索引预处理，含 FEE-sPCA 的 PCA 与 Dfloat 配置搜索）、result/Data（CSV，baseline 预填）+ sync_csv_results.py、Plot/（plot_fig_8/15/16/18/19/22.py）、UniNDP/（修改版模拟器）、下载脚本 download_full_dataset.py / download_query_groundtruth.py / download_anns_idx_cache.py、实验脚本 fee_dim_freq.sh / get_var.sh / overall_basic.sh / overall_hp.sh / qps_vs_recall_*.sh / prefetch_hit_rate.sh / cache_hit_rate.sh / build_index.sh。环境：Ubuntu 22.04、Conda、Python 3.12、PyTorch 2.7.1+cpu（预建索引路径）；从零建索引需 CUDA 12.x、PyTorch 2.5.1+cu124、CuPy 12.3.0、cuVS 25.06.00，GPU ≥24GB（BigANN ≥70GB VRAM、320GB DRAM）。复现流程：`conda create -n naszip python=3.12 && bash init_env.sh && pip install torch huggingface_hub matplotlib pandas seaborn numba`，下载数据集/query/预建索引后跑对应 .sh 脚本 → `cd result && python sync_csv_results.py` → 跑 Plot 脚本出图；预建索引约 1 小时下载（85GB）、总磁盘 150GB、模拟约 7 小时（16 核并行）。
  - 算法 pipeline 伪代码/张量计算（FEE-sPCA 在线搜索，对应论文 Fig.6 与 Eq.(2)-(6)）：设向量库 P∈R^(n×D)，PCA 离线得特征值 {λ_i} 与变换后库 VD。离线预处理：(a) 对 P 做 PCA，得到 VD 与 λ_i；(b) 由 Eq.(2) E(‖v_1:d‖²/‖v‖²)=Σ_{i≤d}λ_i/Σ_{i≤D}λ_i 得 α_k=Σ_{i=1..D}λ_i/Σ_{i=1..k}λ_i；(c) 用 Chebyshev 不等式 Eq.(5)(6)：P(α_k·d_part^k/β_k < d_all) ≥ 1−Var_k/(2ε_k²)，取该概率 ≥90% 解出 ε_k、β_k=1+ε_k（Var_k 在索引构建时统计）。在线搜索（每个候选向量 x，query q）：k 从 1 开始按 DRAM burst 步进——每步读入 2 个（或 burst 内）特征，累加 d_part^k=Σ_{i≤k}(x_i−q_i)²（L2，IP 同理），估 d_est^k=α_k·d_part^k/β_k，与 threshold（候选队列最远距离）比较：若 d_est^k ≥ threshold 则触发早退丢弃 x；否则继续下一 burst，直到 k=D 得到精确 d_all，若 d_all<threshold 则插入优先队列。Dfloat（Algorithm 1）：对 N_burst∈[N_burst^min=d/(B_burst/32), N_burst^max=d/(B_burst/12)] 二分搜索 + 枚举 cfg-validate（规则：同一 burst 特征同格式；burst 内特征数固定时尽量加宽位宽；位宽随特征索引递增而递减；N_burst 必须是每 sub-channel device 数的倍数；B_burst 对 DDR5=128bit、DDR4=64bit），用位掩码在 host CPU 上模拟各配置精度损失（无需重建索引），选出满足 recall@k≥R_target 且 burst 数最少的 C_opt={n_exp,i, n_man,i}_{i=1..N_seg}。Dfloat 值进 FPU 前零填充到 FP32，因此不改计算单元；与标准 DDR5 burst 格式和 on-die/side-band ECC 兼容。例如 SIFT 128 维：段 1~42/43~74/75~128 分别用 18/14/16 bit，每个 device 每 burst 128bit，三段分别需 6/4/6 个 burst，四 device 交叉并行。

## OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 OASIS 的非均匀权重-激活量化（NU-WAQ）算法：K-Means 学习码本量化（权重 4-bit，整矩阵共享质心+每输出通道独立缩放因子；激活 3/4-bit 按 token 量化，每 token 独立质心与缩放因子，激活质心用 C4 数据集 16 个校准样本经 Fisher 信息矩阵加权的 K-Means 离线学习）；outlier 处理（动态保留每 token top 0.5% 最大 + bottom 0.5% 最小激活为 FP16，OASIS-S 变体复用离线校准阈值）；WAQ LUT-GEMM（离线预计算权重×激活质心的 Cartesian Product LUT，在线用索引拼接+分布计数+加权和把 K 次 FP16 加法缩为最多 2^(nW+nA) 次查表归约）；look-ahead 计算与误差补偿（主分支对全激活量化做 LUT-GEMM，outlier 分支并行对 FP16 保留的 outlier 计算误差补偿，数学结果与动态检测等价）。实验比较：WikiText-2 PPL 与 6 个 zero-shot 任务准确率（W4A4/W4A3）对比 FP16 与 INT-WAQ baseline RTN、SmoothQuant、QuaRot、Atom（Atom 为 group-128 分组量化）。结果：W4A4 下 OASIS 平均相对 FP16 仅降 2.05% PPL 精度、W4A3 降 5.90%；平均 accuracy drop 1.94%，比 Atom 低 6.34%，比 QuaRot 高 6.92%（W4A4）。
  - 硬件平台：所有算法精度实验在 NVIDIA A100-80GB GPU 上运行。
  - 模型：OPT-6.7B/13B/30B、LLaMA-7B/13B/30B、LLaMA-2-7B/13B/70B、LLaMA-3-8B、Mistral-7B，基于 Transformers + PyTorch。数据集和 bench：WikiText-2（PPL，seq len 2048）；zero-shot 用 Language Model Evaluation Harness 评测 PIQA、ARC-easy、ARC-challenge、BoolQ、HellaSwag、WinoGrande 六个数据集；激活质心校准用 C4 数据集 16 个样本（论文验证 C4/PTB 下 PPL 稳定、16 样本后收敛，量化时间 16→32 样本从 42.47 增至 100.52 分钟）。
  - 开源情况：论文未提供代码链接，arXiv 摘要页（2507.23035）无 Code 链接，联网搜索（2026-08）未发现公开仓库，开源状态无法确认。算法核心定义（论文公式 1，K-Means 码本量化）：x̃_i = C_{idx_i}，idx_i = argmin_k ‖x_i − C_k‖²，即 n-bit 量化用 n-bit 索引矩阵 + 2^n 个 FP 质心码本表示数据。
  - 算法 pipeline 例子（WAQ LUT-GEMM 张量计算，M=1, K=6, N=4, nW=nA=1 示例）：(1) 离线：由权重质心 C_W 与激活质心 C_A 预计算 Cartesian Product LUT，LUT[j] = C_W[⌊j/2^nA⌋]·C_A[j mod 2^nA]，共 2^(nW+nA) 项（W4A4 下 256 项，相对 4096×4096 层的 inner-product LUT 小 64×）；(2) 在线：激活向量 x∈R^K 聚类为索引向量 idx_A∈[0,2^nA)^K，权重矩阵为索引矩阵 idx_W∈[0,2^nW)^(K×N)；(3) 对每个 (k,n) 拼接 concat_idx = idx_A[k] ∥ idx_W[k,n]；(4) 统计每个输出通道 n 上各拼接索引的出现次数 count[j] = Σ_k 1{concat_idx(k,n)=j}；(5) Y[n] = Σ_j count[j]·LUT[j] 完成沿 K 的归约（FP16 加法次数从 K 降到 2^(nW+nA)）；outlier 通道 x_i 的量化残差 (x_i − C_A[idx_i]) 与该通道反量化权重相乘后累加补偿到 Y。归约长度可等于整层输入通道数 K，不受 group 限制。

## Omni-LUT Energy-Efficient LUT-based Accelerator with Hardware-Aware KV Cache Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 Omni-LUT 的 hardware-aware KV cache 量化算法（ISCA 2026，NYCU）：(1) 硬件友好的 binary-coding 量化框架——把量化结果表达为 binary-coding bit-planes B∈{-1,+1}^{q×d} 与 power-of-2 缩放因子 {α_i}（x ≈ zp + Σ_{i=1}^q α_i ⊙ B_i），直接供 LUT-based GEMM 加速器的 PE 消费，支持两种离线校准方法：BC-UQ（Binary-Coding Uniform Quantization，per-channel 均匀量化，缩放因子固定为 power-of-2 basis {α_u·2^{-1}, α_u·2^0, ..., α_u·2^{b-2}} 整体乘以均匀 scaler，z_bcq = z_u - (2^b-1)/2）与 BCQ（Binary-Coding Quantization，用 Algorithm 1 交替优化学习最优缩放因子：GREEDY_INIT → 每轮 LEAST_SQUARES(固定B求α) + BST(固定α求B)，共 R 轮，非均匀、低位宽更准但校准更重）；(2) Key cache 用 offline-calibrated per-channel BCQ + AS-Bit（Attention-aware Sensitivity-based Bit Allocation）：按边际增益 ΔJ_d = E[Q²]_d·(MSE_{bℓ}[d]−MSE_{bh}[d]) 度量每通道敏感度（E[Q²]_d 为校准 Query 通道能量、MSE_b[d] 为 Key 在 bit width b 下的量化误差），把 top k%（论文用 25%）高敏感通道分配高位宽 b_h、其余用 b_ℓ；(3) Value cache 用 online per-token BC-UQ（Token-Scale Estimator 在线算 min/max → zp_v=(x_max+x_min)/2、δ_v=(x_max−x_min)/(2^b−1)，α_i=δ_v×power-of-2 basis），简单且足够准。实验比较：WikiText-2 PPL 与 PIQA/Winogrande/HellaSwag zero-shot 准确率；KV-only 量化对比 KIVI、KVQuant、Oaken、QServe、NVFP4；端到端量化对比 Atom（W4KVA4）、Tender（W8KVA8/W4KVA4）；还对比 Uniform-KV4/KV3（LLaMA3-8B、Qwen3-8B/14B/30B-A3B）与不同校准数据集（WikiText-2/C4/The Pile，Table V 显示对 zero-shot 影响可忽略）。结果：KV4-BCQ 平均 PPL 仅增 0.17、KV3 增 0.75；AS-Bit 下 Key 有效位宽仅 4.25 bit（25% 高位通道），远低于 KIVI/KVQuant/Oaken 依赖 sparsity-based outlier 处理的 4.8–5.0 bit。
- 硬件平台是什么，配置是什么。
  - 精度实验：NVIDIA H200 GPU 上做离线校准（LLaMA2-13B 校准 <10 分钟），实现基于 PyTorch + Hugging Face transformers，评估时在每个 attention 层内模拟 KV cache 量化。硬件评估（量化后硬件协同验证）：Synopsys Design Compiler（TSMC 7nm @ 500 MHz 综合）+ Synopsys PrimeTime PX（功耗）+ Synopsys VCS（门级仿真标注 cycle 数与 switching activity）+ DRAMPower（LPDDR5 51.2 GB/s peak）+ roofline-style 性能模型（compute time + DRAM access time），batch size=1 的 edge-serving 设定；FlashAttention 算法以 tile 256 流式计算 attention。
- 模型是什么。数据集和bench分别是什么。
  - 模型：OPT-1.3B/6.7B/13B、LLaMA2-7B/13B、Mistral-7B、Mixtral-8x7B、LLaMA3-8B、Qwen3-8B/14B/30B-A3B，覆盖 MHA、GQA、MoE 架构（edge-friendly 到较大规模）。数据集与 bench：WikiText-2（training split 用于校准、test split 报 PPL）；zero-shot 用 PIQA、Winogrande、HellaSwag；校准数据集对比 WikiText-2、C4 (En)、The Pile (Val)。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供代码链接；联网搜索（2026-08）未发现 Omni-LUT 的 arXiv 版本或 GitHub 仓库（仅有 NYCU CS 新闻页与 IEEE ISCA 2026 pp.322-337 记录），开源状态无法确认。baseline（KIVI、KVQuant、Atom、Tender、QServe、NVFP4）均有公开实现。
  - 算法 pipeline 张量计算例子（BEA 贪心残差编码 + AS-Bit + LUT 查表）：设 Key 向量 x∈R^d，离线校准得 zero-point zp 与缩放因子 {α_1,...,α_q}（BCQ 交替优化或 BC-UQ power-of-2）。在线 BEA 编码：r^(0)=x−zp；对 i=1..q：B_i=sign(r^(i-1))，r^(i)=r^(i-1)−B_i·α_i；输出 bit-plane 矩阵 B∈{-1,+1}^{q×d}（Key path 用离线 per-channel zp_{k,c}/α_{i,c}，Value path 用 TSE 的 token-wise zp_v/α_i）。AS-Bit 离线：对每通道 d 计算 E[Q²]_d=(1/T_cal)Σ_{t=1}^{T_cal}(Q_{t,d})² 与双路径量化误差 MSE_b[d]=(1/T_cal)Σ_t(K_{t,d}−K_{q,b}[t,d])²，得边际增益 ΔJ_d 后取 top 25% 通道用 b_h。LUT 侧：4 个激活一组，LGU 把组内每个激活乘其 row-wise scale 后枚举 8 个基础组合条目（half-LUT，PE 用符号对称补全 16 项），首 bit-plane 内嵌 zero-point 补偿；每 cycle 该组按量化矩阵一列的 binary 值查表，32 个 binary weight 并行 Read-and-Accumulate，lookup 次数 ∝ 位宽（2-bit 权重需 2 次查表），partial sum 累加得最终点积。

## Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现有两项：(1) Axis-shared rasterization（轴共享光栅化）——把每个 tile（16×16 像素）内 α-computation 的指数项分解为 X 轴二次项、Y 轴二次项、交叉项三部分，沿 X/Y 轴预计算共享项并广播给各 PE 复用，消除同行/同列像素间的冗余计算，使 α-computation 的 MAC 从每像素 8 MUL+4 ADD 降至摊销后 2.31 MUL+2.13 ADD/PE，完整 rasterization（含 α-blending）MAC 减少约 38%，且保持像素级全并行（不引入 GBU/FastSplat 的行序差分依赖）。(2) MLP-based order-independent transmittance（MLP 顺序无关透射率，OIT）——用轻量 2 层 10 参数 MLP 直接预测每个 Gaussian 的衰减因子 F(d_i) 替代显式深度排序：输入为 Gaussian 深度 d_i 与归一化视角方向 (x,y,z)（推理时视角恒定、其贡献折入 MLP bias），第一层 Leaky ReLU(系数 1/8)，输出层指数函数（复用光栅化 EXP 单元），推理仅 6 MAC；渲染公式由按深度序的 C=ΣT_iα_ic_i 改为顺序无关的 C=ΣF(d_i)α_ic_i / ΣF(d_i)α_i。
  - 实验比较：算法精度对比 (i) 原始排序 3DGS baseline，(ii) sort-free weight-sum 渲染 [18]（最优变体 LC-WSR），及 OIT 仅深度输入（OIT+d）与含视角输入（OIT+d+view）的消融，指标 PSNR/SSIM/LPIPS；动态场景（Neu3D + 4DGS）比较 baseline vs OIT 的 PSNR/SSIM。硬件消融 BS/BS+AR/BS+AR+OIT/BS+AR+OIT+IP 四变体比较吞吐（1×/1.37×/2.16×/2.27×，几何均值）。

- 硬件平台是什么，配置是什么。
  - 算法训练平台：NVIDIA RTX 3090 GPU。先按 [22] 训练 7000 epochs 得到 checkpoint（原始排序算法预训练模型），再以其初始化额外训练 10000 epochs：MLP 学习率 0.005，Gaussian 学习率按 0.01 缩放；关闭 Gaussian cloning/splitting 保证训练稳定；每场景约 30 分钟。动态场景每 30 帧更新一次 10 参数 MLP（300 帧序列共 10 组参数）。GPU 推理评估平台：Jetson Orin Nano edge GPU（8nm、200mm²、~15W、68.2 GB/s、1024 CUDA cores）与 RTX 3090（8nm、628mm²、350W、936 GB/s、10496 CUDA cores）。加速器硬件：TSMC 28nm、1 GHz、3.85mm²、1.64W、96KB on-chip SRAM、DDR5-4800 38.4 GB/s、256 PEs（16×16 可重构阵列）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：3D Gaussian Splatting 场景表示（每个 Gaussian 59 参数：3 均值 + 3 scale + 4 rotation + 1 opacity + 48 SH 颜色系数），外加 2 层 10 参数 MLP（Leaky ReLU(1/8) + 指数输出）预测透射率。动态场景用 4DGS [50] 建模（per-frame 渲染与静态 3DGS 相同）。
  - 数据集与 bench：静态用 MipNeRF-360 [1] 的 7 个真实场景 garden、bicycle、stump、bonsai、counter、kitchen、room（validation 集），指标 PSNR/SSIM/LPIPS；动态用 Neu3D 数据集 [27]（2704×2028 高分辨率、300 帧/10 秒，Cook Spinach、Cut Beef、Flame Steak），指标 PSNR/SSIM；GPU profiling 用 MipNeRF-360（7k 训练 checkpoint）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/WangZhican/ISCA26_3DGS_Acc（含 MLP-based_OIT 目录，CUDA 实现，9 commits，1 contributor）；算法与 GPU 实现基于开源库 gsplat [51]（https://github.com/nerfstudio-project/gsplat）。sort-free weight-sum 基线见 arXiv 2410.18931。
  - 算法 pipeline 解释（以 MipNeRF-360 一个场景为例）：(1) 预训练：用原始 3DGS（含排序）训练 7000 epochs 得 Gaussian checkpoint；(2) OIT 训练：加载 checkpoint，对每个相机位姿投影 Gaussian 得 2D 均值 μ、2D 协方差 Σ 与深度 d_i，构造样本（d_i, 视角(x,y,z)）→ 前向：MLP 输出 F(d_i)，按 C=ΣF(d_i)α_ic_i/ΣF(d_i)α_i 渲染，与 GT 图像计算 loss（原 3DGS 设置）→ 反向：MLP 大步长（lr=0.005）快速收敛，Gaussian 小步长（lr×0.01）缓慢精修；(3) OIT 推理：给定相机位姿，视角 (x,y,z) 对所有 Gaussian 相同，预计算并融合进 bias（b_i=b'_i+c_i·view 贡献），对每 tile 内 Gaussian 深度 d_i 做 6 MAC 前向得 F(d_i)，再按上述公式 α-blending；(4) 全程无显式排序。张量计算例子：单 Gaussian α 计算 α=o·exp(-½(p-μ)ᵀΣ⁻¹(p-μ))，指数分解为 -(½a(x-μx)² + ½b(y-μy)² + c(x-μx)(y-μy))；axis-shared 版本把 x 项、x² 项（X-PE 线）与 y 项、y² 项（Y-PE 线）各算一次广播到 16×16 阵列，每像素 PE 仅需 1 乘法 (x·y) + 2 加法合成指数，再乘 opacity o 过 EXP。

## P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 P3-LLM 的 operand-dependent 混合精度量化方案 W4A8KV4P8（权重4-bit、激活8-bit、KV-cache 4-bit、注意力分数8-bit），为每个 LLM 操作数分配专属数值格式以最小化量化误差：(1) KV-cache 用 4-bit INT4-Asym（非对称整数）按 head 量化（每 head 维度 Dh 个元素共享 16-bit 缩放因子与 4-bit zero-point，有效精度 4.16 bit），并配合无校准数据集的 dynamic input-aware smoothing 抑制 key cache 的 outlier 通道——平滑因子在 prefilling 阶段按每通道绝对最大值计算（K_S[:,c]=K[:,c]/Max(|K[:,c]|)），保存后在 decoding 阶段复用来缩放新生成的 key 向量；(2) 权重用 4-bit BitMoD 格式（在 FP4 基本值 {±0,±0.5,±1,±1.5,±2,±3,±4,±6} 基础上把冗余的负零编码重映射为 4 个特殊值 {±5,±8} 之一，按权重分组搜索最优特殊值，group size 128）；(3) 激活用 FP8-E4M3 按 token 量化（无需 Hadamard 变换或 SmoothQuant 式平滑，依靠宽数值范围容纳 outlier）；(4) 注意力分数用无符号 8-bit 浮点 FP8-S0E4M4（4-bit mantissa + 4-bit exponent，bias −15，覆盖 softmax 后 [0,1] 且无需符号位，直接截取 FP16 的高 4 位 mantissa）。同时实现 operator fusion 最小化运行时 dequantization 开销：线性层 dequant 缩放放在矩阵乘法之后；Q·K^T 把 post-RoPE key cache 的 per-channel smoothing factor (SSF) 融合进 query 再做 FP8 量化；P·V 把 per-value-head 缩放因子 S^V 融合进 attention-score（除以 S^V_max 做二级缩放使 S^V∈[0,1] 防止越界，S^V_max 随后乘回 P·V 结果）。实验比较：(a) 精度对比 KV-cache-only 量化 Oaken 与 weight-activation 量化 QuaRot、QoQ，指标为 Wikitext-2/C4 perplexity（2K context 对齐 baseline）与 MMLU/ARC-Challenge/GSM8K 推理准确率；(b) 加速器性能对比 NPU-FP16、HBM-PIM-FP16、Ecco、Pimba、SmoothQuant/AWQ 软件量化；(c) 算法消融（Table VI：逐项叠加 INT4 KV → 动态平滑 → BitMoD 权重 → FP8-S0E4M4 注意力 → FP8-E4M3 激活）。结果：W4A8KV4P8 平均 perplexity loss 仅 0.25（Wikitext-2）/0.31（C4），优于 Oaken（0.08/0.09，KV4 下 P3-LLM 更低精度 4.16bit vs 4.8bit 却更优）、QuaRot（0.30/0.48）与 QoQ（0.30/0.38）；reasoning 平均准确率比 QuaRot/QoQ 高 2.57%/3.05%。
  - 硬件平台：算法精度实验在 PyTorch 实现（HuggingFace 预训练模型），动态平滑开销 profiling 在 NVIDIA A6000 GPU 上运行（Llama-3.1-8B 全层 <5ms，即使 32K context，相对 250ms TTFT SLO 仍 <2%）。
  - 模型：Llama-1-(7B,13B)、Llama-2-(7B,13B)、Llama-3.1-8B、Llama-3.2-3B、Mistral-7B-v0.1/v0.3（8 个模型，来自 HuggingFace）。Llama-1/2 因序列短采用 pre-RoPE key-cache 量化，Llama-3/Mistral 采用 post-RoPE。数据集与 bench：Wikitext-2、C4（perplexity）；MMLU、ARC-Challenge、GSM8K（推理准确率，用 instruction-tuned 的 Llama-3.1-8B/Llama-3.2-3B，按 LM-Eval 框架建议）。
  - 开源情况：开源，https://github.com/yc2367/P3-LLM（MIT license）。仓库含 wkvaq_quant/（量化代码库）、3rdparty/ 子模块（AWQ、LM-Evaluation-Harness）、kv_profile/（KV cache profiling 支持）。README 流程：`git clone --recurse-submodules` 后配置 AWQ，跑 `wkvaq_quant/scripts/awq/run_awq.sh` 做 4-bit 权重量化（`wq_dtype` 可选 "int" 或 "bitmod"，默认 4-bit group size 128）；跑 `run_awq_save_4b_model.sh` 评估并保存 fake-quantized 4-bit 模型；用 `wkvaq_quant/scripts/test_ppl_template.sh` 测 Wikitext-2/C4 perplexity（仅支持 Llama 和 Mistral，关键 flag：`--kv_quant_method` 默认 "KTVT" 或 "KCVT"、`--kv_residual_len`、`--apply_k_scale`、`--k_quant_post_rope`、`--p_bits` 控制 attention-score 精度）。硬件模拟器与 RTL 未开源。
  - 算法 pipeline 例子（W4A8KV4P8 的量化-融合张量计算，以一次 decode 迭代为例）：(1) 预填充阶段：对输入 context 的 key cache K∈R^(NT×H)，按通道计算平滑因子 s_c=Max(|K[:,c]|) 存下（内存开销 <1%，与 context 长度成反比），K_S[:,c]=K[:,c]/s_c；解码阶段新 key 向量除以同一 s_c 后按 head 做 INT4-Asym：Δ=Max(|X|)/7，X_Q=Round(X/Δ,{-8..7})，量化后 (X_Q,z_KV,Δ) 存入 DRAM（有效 4.16 bit/元素）；(2) 权重离线 BitMoD 量化：对每组 128 个权重搜索最优特殊值 v∈{±5,±8} 替换负零编码，得到 4-bit 索引 + 组级缩放；(3) 解码某 token：query 线性层输出后做 RoPE（pre-RoPE 方案则需在线对 key 做 RoPE，因为量化 key 缺位置信息），Q·K^T 时把 key 的平滑因子 SSF 元素乘进 query（fuse 进 FP8-E4M3 per-token 量化前的缩放），P·V 时把 S^V/S^V_max 乘进 softmax 后注意力分数再转 FP8-S0E4M4（P 分数直接取 FP16 高 4 位 mantissa）；(4) 低精度 MAC：权重/KV 4-bit 值与激活 8-bit 尾数（5-bit 含隐藏位 + 符号位=6-bit 定点）相乘，4-bit 指数右移乘积，4:2 压缩树归约后 32-bit 定点累加，最后乘 dequant 缩放因子（线性层在 GEMM 后、QK^T/PV 已融合）输出到下一层。

## PRowhammer Propagating Bit-flips from CPU to GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现是 PRowhammer 的自动"可利用位翻转"定位算法（black-box 单 bit-flip 精度降级攻击的离线分析管线），核心是位翻转模拟 + 递归剪枝策略：(1) 可行性验证（CustomLib，自编译、含 vanilla 矩阵乘 kernel 的共享库，nv_fatbin 21KB）：随机单 bit-flip 后执行 kernel，每 trial 100ms、共 10000 次，跨三款 GPU 架构得到 8.13–11.16% 崩溃率、0.21–0.25% 可利用翻转率（输出与预期不同且不崩溃）。(2) 剪枝策略（应对大库：cuBLASLt 压缩 nv_fatbin 255MB、GGML 14MB，逐 bit 检查需约 11805 天）：把 nv_fatbin 均分为 n=2 段，对每段翻转全部 bit 并执行目标库 kernel，输出正确则丢弃该段，崩溃或输出改变则标记为有用段并递归二分为止直到阈值 T=1KB；再从有用段随机抽 10000 bit 逐个 flip+执行，cuBLASLt 得 3–83 个、GGML 得 41–99 个可利用 bit-flip，最大库运行不超 90 分钟，全程无需源码访问、不逆向压缩算法（NVCC --compress-mode 的闭源压缩）。(3) 黑盒 profiling 模型：cuBLASLt 含 sm_86 的 3508 个 kernel 而每个模型只调用 1–2 个，随机破坏成功概率仅 1/3508；利用"分类模型末层是线性层、线性层调 cuBLASLt kernel"的结构性质，用单线性层（随机权重、输出维度=目标类别数）构造 profiling 模型定位 kernel（输入维度 2–10000 扫描），在 profiling 模型上造成最大精度降级的 bit 在目标模型上同样最致命。
  - 实验比较：图像分类在 16 个 test case（ResNet-18/34/50、VGG-16 × MNIST/FMNIST/CIFAR-10/ImageNet，各模型每数据集分别训练）上比较攻击前后分类准确率（pristine vs corrupted）与随机猜测基线 ACC_random=1/Class(D)，指标为相对预测损失 RPL=(ACC_Pristine−ACC_Corrupted)/ACC_Pristine；单个 bit-flip（cuBLASLt 共享库）即把准确率压到接近随机猜测（10 类数据 8.10–13.70% vs 10.00%，ImageNet 最坏 0.00%，RPL 84.95–100%）；50000 次随机 bit-flip 找得 MNIST/FMNIST/CIFAR-10 共 218 个、ImageNet 93 个可利用翻转，同一翻转位跨模型/数据集转移有效（RPL>80% 的转移位 92–169 个，Table IV/V）。LLM 上比较 Llama-2-7B/Mistral-7B/Falcon-7B（4-bit 量化）在 Google Natural Questions 100 问上的平均 BERTScore F1（pristine 0.58–0.62 → corrupted 0.25–0.30），单 bit-flip（GGML 库的 ggml_mul_mat kernel）使模型输出 # 串或跨语言乱码，也有语法连贯但事实错误的输出。
- 硬件平台是什么，配置是什么。
  - 平台 A：Intel Core i7-4790 (Haswell)、8GB Kingston DDR3 (1600 MT/s)、kernel 5.15.0-131-generic；平台 B：Intel Core i7-8700 (Coffee Lake)、8GB Corsair DDR4 (2400 MT/s)、kernel 6.2.0-060200-generic。GPU：NVIDIA RTX 4090、RTX A6000、RTX 5060（RTX 5060 因当时 PyTorch 支持不稳定未用于图像分类，仅用于 LLM 实验）。OS Ubuntu 20.04.6，CUDA Toolkit 12.8。
- 模型是什么。数据集和bench分别是什么。
  - 模型：图像分类 ResNet-18/ResNet-34/ResNet-50、VGG-16；LLM 为 Llama-2-7B、Mistral-7B、Falcon-7B 的 4-bit 量化版（经 llama.cpp/GGML）。数据集与 bench：MNIST、FMNIST、CIFAR-10（10 类）、ImageNet（1000 类）；LLM 用 Google Natural Questions（100 问，手工标注参考答案），指标 BERTScore（F1）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文声明将发布代码，artifact 托管于 Zenodo https://doi.org/10.5281/zenodo.19326669（prowhammer-artifact.tar.gz，需 ~70GB 磁盘、准备 2–3 小时、实验 6–9 小时）；软件依赖 Ubuntu 24.04 LTS、CUDA Toolkit 12.8、Python 3.12 + Anaconda（PyTorch/NumPy/pandas/matplotlib）、GNU Make 4.3、CMake 3.28.3、md5sum；硬件需 RTX A6000/4090/5060 之一、≥8 核 CPU、≥16GB DRAM。安装：`tar -xvzf prowhammer-artifact.tar.gz && cd prowhammer-artifact/image_classification_and_llm_attack && bash setup_env.sh && conda activate prowhammer`。预训练模型与数据集已含在仓库，可从零训练（`cd model_training && python download_dataset.py && python run.py --model-dir trained_models/ | tee out/golden_out_1000.txt`）。
  - 算法 pipeline 例子（伪代码，对应 Sec. IV-A 剪枝 + artifact 五阶段 profiling）：输入为压缩共享库 nv_fatbin 字节串 B（如 cuBLASLt 255MB）。(1) 剪枝 `find_useful(seg, T=1KB)`：若 |seg|≤T 返回 [seg]；否则对该 seg 内全部 bit 逐一翻转后执行目标 kernel 并与 golden 输出比对——输出正确 → 丢弃 seg；崩溃或输出改变 → 把 seg 二等分后递归 `find_useful`。最终得到 1KB 有用段集合，再从中随机选 10000 个 bit，每个 flip+执行（500–700ms/次）统计出可利用集合。(2) artifact 的五阶段脚本管线：`kernel_locater_<lib>.sh`（定位应用调用的 kernel → regions 文件）→ `choose_target_region.sh`（选大而连续、可 Rowhammer 的候选区域）→ `run_flipper_watchdog_<lib>.sh`（候选区域精确 bit-flip 实验，记录原始结果）→ `segregate.sh`（整理）→ `extract_useful_flips.sh`（过滤汇总成 bitflip_data.csv）；用 cuobjdump + diff 验证翻转后 SASS 合法性。预计算关键位：mnist/fmnist/cifar10 用 cuBLASLt 偏移 0x95c787a 的 bit 4；imagenet 用偏移 0xc56745c 的 bit 8。图像分类复现：`python download_dataset.py && bash scripts/flip_analysis.sh most_critical_bit_flip.csv && bash get_output.sh`（imagenet 用 flip_analysis_imagenet.sh + get_output_imagenet.sh）；LLM 攻击先跑 `run_profile_ggml.sh` 得 bitflip_data_ggml.csv（ggml_mul_mat 的 profiling，RTX A6000/5060/4090 分别得 33/55/64 个可利用翻转）；绘图脚本在 plotting_tabulation/scripts/（plot_figure7_table_4_5.py、plot_figure8_table_4_5.py、plot_figure4.py）。

## ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 ParetoES 的"选择性计算"算法套件，用于加速 Top-K 稀疏矩阵向量乘（SpMV）形式的稀疏 embedding 相似度检索。三层正交离线优化叠加在基于聚类的倒排索引之上：(1) Spherical K-means++ Refine（球面 K-means++ + 动态分裂合并精化）——用余弦相似度替代 L2 距离做方向敏感聚类以适配稀疏高维向量几何；K-means++ 式初始化（采样概率 p(x_i)=(1−max_{c_j}x_iᵀc_j)/Σ(1−max x_kᵀc_j)，从 m=min(0.01n,10000) 候选采样，复杂度从 O(nd) 降到 O(mKd)）；动态精化：质心间 cos>θ_merge=0.9 时合并为最接近均值向量、簇内平均 cohesion<θ_split=0.6 时用 2-means 分裂直至两子簇都满足凝聚度（max_refine_iter 上限），减少候选簇扫描数。(2) 对称 INT6 均匀整数量化（动态缩放 α=max(|x|)，线性映射 x∈R→[-31,31]，不裁剪以保留角度缩放）——与 Ultra-CSR 结合使有效带宽较 FP32 提升 6×；混合精度：聚类用全精度浮点，向量/质心/查询聚类后一次性量化到 6-bit。(3) 增强版 ReSparse 非结构化剪枝——把剪枝阈值均值改为仅对非零元素计算（原始 ReSparse 用全局均值被零值拉低，极端稀疏下几乎不剪），使平均剪枝率从原始 23.93%（Sp.10M 上仅 2.34%）提升到峰值 61.25%、平均 37.41%（Sp.10M 上 11.48%），并集成到选择性计算 Top-K SpMV 范式（首次将非结构化剪枝用于选择性计算）。
  - 实验比较：精度消融对比 K-means（Faiss baseline）、Spectral Clustering、Hierarchical Clustering 的 Recall@100 vs 扫描向量比例曲线（Fig.9，nlist=√(m/2)，1000 迭代上限、10⁻⁴ 收敛阈值）；剪枝消融对比原始 ReSparse [60] 的非零元素保留量（Fig.10，AccelES 相比额外减非零平均 18.09%、最高 39.14%）；32×Top-16 分解 vs 精确全局 Top-K 排序的 Recall@K（K=10..512，K≤200 恒为 100%，Table III）；整体检索性能对比 CPU（Faiss-CPU、sparse_dot_topn 精确全计算基线）、GPU（Faiss-A100、Faiss-V100）、FPGA（AccelES、FPGA32）的 QPS-Recall@100/Recall@10 曲线（Fig.11）与 Sp.GloVe 延迟分布（Fig.12）；多优化消融（Fig.13，Sp.CC_zh：Kmeans 硬件 3307 QPS → +Spherical Refine 3622 (+11.3%) → +H2Balance 4065 (+15.9%) → +ReSparse 5869 (+64.9%)，全量 2.11× vs AccelES）。
- 硬件平台是什么，配置是什么。
  - 离线预处理与聚类在 NVIDIA A100 GPU（6912 CUDA cores、40GB HBM2）上执行；CPU baseline 为 Inspur NF5468M5 服务器（Intel Xeon Gold 5117，56 vCPU @2.00GHz、256GB DDR4）；GPU baseline 为 NVIDIA A100（40GB HBM2）与 V100-SXM2（5120 CUDA cores、32GB HBM2），CUDA 12.1；FPGA 为 Xilinx Alveo U280（双 HBM2 栈、32 pseudo-channel、512-bit @225MHz、理论峰值 460GB/s），Vitis HLS 2023.2 综合。
- 模型是什么。数据集和bench分别是什么。
  - 模型：无传统 DNN 模型；检索对象是稀疏 embedding 向量集（矩阵每行一个数据库候选向量），查询为稀疏 query 向量，相似度=内积（Top-K SpMV 形式）。稀疏 embedding 按 [18]（Sparse Overcomplete Word Vector Representations）流程生成。
  - 数据集与 bench：Sp.Baidu（6.4×10⁵ 行、900 维、14.02% 密度、8.03×10⁷ 非零，索引构建 71s）、Sp.CC_zh（2×10⁶ 行、900 维、2.12%）、Sp.GloVe（4×10⁵ 行、500 维、13.83%）、Sp.Wiki News（1×10⁶ 行、900 维、3.04%）、Sp.10M（合成，1×10⁷ 行、524 维、0.72% 密度，索引构建 363s、总预处理 680s）。指标：Recall@100/Recall@10、QPS、P99 延迟、预处理时间（A100 上索引构建 37–363s、总预处理 ≤680s）、TCO 每查询成本。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供开源链接，联网检索（2026-08）未发现公开仓库；同组先前工作 AccelES（HPCA 2025）亦无公开代码。Baseline 依赖开源库：Faiss v1.7.2（IndexIVFFlat + IndexFlatIP quantizer，faiss.METRIC_INNER_PRODUCT）、ing-bank/sparse_dot_topn（https://github.com/ing-bank/sparse_dot_topn）、cuSPARSE、Thrust。
  - 算法 pipeline 例子（伪代码/张量计算，一次查询）：(1) 离线聚类：对稀疏向量集 {x_i}（d 维）归一化 x̂=x/||x||₂；K-means++ 按 p(x_i) 采样 K=nlist=⌊√(m/2)⌋ 初始质心；交替 assign（j=argmax_k x̂_i·μ_k）与 update（μ_i=argmax_{x∈c_i} x·normalize(Σx)）至收敛；动态精化 merge（max_{i≠j}cos(μ_i,μ_j)>0.9 的簇对，μ_new=argmax_{x∈c_i∪c_j} x·normalize(Σx)）与 split（cohesion(c_k)=(1/|c_k|)Σxᵀμ_k<0.6 时 2-means 分裂，重分配至两子簇 cohesion≥0.6）。(2) 量化：每向量按 α=max|x| 动态缩放，Q_x=round(31·x/α)∈[-31,31]（INT6，无裁剪）。(3) ReSparse 剪枝：按仅非零元素的均值确定阈值，剪掉小幅度非零。(4) 编码：按簇 ID 重排矩阵为子矩阵，编码为 Ultra-CSR（每 512-bit packet 含 30 个非零）。(5) 在线检索：查询 v 量化 → 与质心内积选 Top-nprobe 簇（nprobe 在 Recall@100≥0.8 约束下确定，sub_nprobe=⌈nprobe/32⌉/核）→ 仅对选中簇子矩阵计算 y_i=⟨A_i,v⟩ → 32 核各局部 Top-16 → 聚合 Top-512 → Recall@K=|TopK_approx∩TopK_exact|/K。对应 GEMV 计算：y=Σ_{j∈nnz(A_i)} A_i[j]·v[j]，只在选中簇上执行。

## Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion

- 属于算法pipeline的实现是什么？实验比较什么？
  - Tree-encoded fusion（树编码融合）是论文提出的新型逻辑量子比特编码/融合算法方案（属于"新算法模型"层）：把参与 Type-II fusion 的逻辑量子比特编码为树结构——根量子比特 q_root 连接 b 个分支，每分支是 3 个量子比特的线性图 {q_i^a, q_i^b, q_i^c}，叶子 q_i^c 用于融合测量，q_i^a/q_i^b 是用于间接测量容错的辅助量子比特。按融合结果执行不同测量模式：(1) 融合成功——对 q_i^a、q_i^b 做一对 X 测量，把成功的融合纠缠直接连到 q_root；(2) 融合失败——q_i^c 被测量掉，q_i^a/q_i^b 留在树中，对 q_i^b 做 Z 测量移除它，留 q_i^a 作备份；(3) 融合擦除（光子丢失）——对 q_i^b 做 X 测量、q_i^a 做 Z 测量，基于 stabilizer X_i∏_{j∈E(i)}Z_j 实现对 q_i^c 的间接 Z 测量，无损消除被擦除 qubit 对图态的影响；(4) 全部分支失败/擦除的极端情况——用 (2) 留下的备份 q_i^a 再试一次融合。相比 redundantly-encoded fusion（m 次冗余融合尝试，但每个逻辑 qubit 的 m 个物理 qubit 独立暴露于擦除，逻辑擦除率 P_eras=1-(1-p_eras)^(2m) 随 m 指数恶化）和 repeat-until-success fusion（P_eras=Σ_{i=0}^{m-1}p_fail^i·2p_eras），树编码用间接 Z 测量把每个分支的擦除影响限制在单分支内，逻辑成功率 S_tree=1-(1-(1-p_eras)^2+p_fail)^b（b=分支数）。
  - 实验比较：同一 MemTree 编译框架下对比三种融合方案（固定融合失败率 p_fail=0.25，擦除率 p_eras 0%~10%，程序 2~20 qubit，执行时间上限 6×10^5 ns）：执行时间平均减少 1.9×10^-3×（vs redundantly-encoded）和 1.7×10^-2×（vs RUS）；光子源消耗多 2.55×（vs redundantly）和 1.63×（vs RUS）——用空间换时间，且光子源劣势随 #qubit 增大而缩小；Fig.4(f) 以 10^3 次融合试验统计成功率，显示 p_eras 升高时树编码优势显著。参数研究（Fig.5）：30-qubit caterpillar 限制下 b_prep>6 时光子源急剧增长，执行时间随 b 增大指数下降至 b=4 后收敛 → 选 b=4、b_prep=6（真实硬件 83.3% 单 timestep 制备成功率、97.1% 两 timestep 内）。Ablation study：MemTree+RUS 在 p_eras=0.5%、36-qubit 下除 Grover 外全面劣于 OneAdapt-ET，证明性能提升主要来自 tree-encoded fusion 而非架构差异。
- 硬件平台是什么，配置是什么。
  - 仿真平台：论文自研 realistic error-aware PQC 模拟器（spin memory 架构），配置来自实验工作 [25][29][30][43][52]：InGaAs 半导体量子点（QD）发射器（LA 纵向声学激发脉冲发射光子 + OSRP 光学自旋旋转脉冲定义 caterpillar 结构），caterpillar 初始化 12 ns + 每 qubit 发射 0.6 ns 时间周期，最大 caterpillar 30 qubit；融合成功概率 1-p_fail=0.75（无擦除时，需额外干涉测量装置），HOM 可见度 V_HOM=99.5% → 融合保真度 σ_fus=99.75%，OSRP fidelity 99%，退相干 T2=2.34 μs（由 4-qubit GHZ 95% fidelity 反推），t_cycle=30 ns；模拟 2×10^4 个 caterpillar 发射周期统计成功 shots。
  - 真实硬件：Quandela 云端 PQC 平台（24-photon modes），光学电路用 Perceval PQC 工具包构建（双轨 dual-rail 编码，融合电路 = 光子模式置换 + 相移 + 两个分束器；SNSPD 探测器延迟 <50 ps；经典 feed-forward 总延迟 <5 ns，用 Perceval FFCircuitProvider 实现）；实测硬件特征：HOM 不可区分度 92.0%、透射率 5.16%、g^(2)=2.0%。对比对象：IBM Torino 超导量子计算机（Qiskit 转译）。
- 模型是什么。数据集和bench分别是什么。
  - Benchmark：6 类典型量子算法——Bernstein-Vazirani (BV)、Quantum Approximate Optimization Algorithm (QAOA)、Grover's Algorithm、Quantum Fourier Transform (QFT)、quantum Hamiltonian simulation (QSIM)、Ripple Carry Adder (RCA)、Variational Quantum Eigensolver (VQE)。程序规模：融合方案对比 2~20 qubit（baseline 融合方案执行时间过长不可及）；编译器对比 36/64/100 qubit；真实硬件实验 6~12 qubit QAOA（EfficientSU2 SU2 默认 ansatz + RealAmplitudes RA 变体），指标 PST（Probability of Successful Trial）与 IST（Inference Strength）。模拟器基于 Perceval（[24]）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供代码仓库链接（arXiv:2604.21475），联网搜索未找到公开仓库，无法确认开源；对比融合方案（redundantly-encoded [25]、RUS [40]）按其论文最新协议实现（[12]），代码尺寸 m_Redun=5、m_RUS=6；Perceval（https://github.com/Quandela/Perceval，开源）用于真实硬件实验电路构建与 feed-forward 控制。
  - 使用例子（tree-encoded fusion 测量模式伪代码，b=4 分支）：逻辑量子比特 A、B 融合，对每个分支 i，Type-II 融合作用于叶子 q_i^c(A) 与 q_i^c(B)（HWP+PBS 线性光学干涉，双光子到异侧探测器=成功、同侧=失败、缺失=擦除）：
    ```
    for branch i in 1..b:
        outcome = type2_fusion(q_i^c(A), q_i^c(B))   # 成功 / 失败 / 擦除
        if outcome == SUCCESS:
            X_measure(q_i^a); X_measure(q_i^b)       # 融合纠缠直接连到 q_root
        elif outcome == FAILURE:
            Z_measure(q_i^b)                          # 移除 q_i^b，留 q_i^a 作备份
        elif outcome == ERASURE:
            X_measure(q_i^b); Z_measure(q_i^a)        # 间接 Z 测量 q_i^c（stabilizer X_i∏Z_j）
    if all branches failed/erased and backup exists:
        fusion_retry_with(q_i^a)                      # 备份尝试
    ```
    树结构从 caterpillar 态组装（Fig.4(e)）：主路径上的 q_root + b 个叶 qubit 由 caterpillar 提供；b 个 4-qubit 线性图从长线性图经 Z 测量分离，再融合到叶子上；制备参数 b_prep=6（>b），同 timestep 并行尝试 b_prep 次分支制备（失败自动测量掉、擦除用间接测量恢复），成功分支 <b 则下一 timestep 重试。


## Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure

- 属于算法pipeline的实现是什么？实验比较什么？
  - Prometheus 的核心算法实现是"多阶段 ML 集成回归"用于预报湿球温度（WBT）——CMIP6 物理气候模拟不直接提供 WBT，但蒸发冷却塔的冷却能力以 WBT 为极限，因此必须用 ML 从模拟数据推断。pipeline：输入 0.25° 网格上 CMIP6 投影的日最低/平均/最高干球温度（DBT）与相对湿度（RH）共 6 个特征；第一阶段两个基回归器各自独立预测每日最大 WBT——随机森林（RF，100 棵树、max_depth=5、每分裂节点最少 2 样本、输出为树平均）与支持向量机（SVM，RBF 核把输入映射到高维空间）；第二阶段神经网络（NN，两层隐层 16/8 神经元、ReLU、L2 正则 0.5、输出层单神经元无激活）以两个基模型预测 + 原始特征为输入，输出单一稳健 WBT 预报。其后做站点特异性偏差校正（用该站点 5 年历史数据算均值差并从输出中减去），最后用 Gumbel 分布（式 2，位置 µ/尺度 β）拟合"25 年历史观测 + 20 年 CMIP6 多模型投影"混合数据，由式 3 由均值 T_max-mean=µ+βγ 与标准差 T_max-std=πβ/√6 计算 N 年一遇回返温度 T_N（对应 1-1/N 百分位）。
  - 实验比较：①ML 精度（Table I）：比较 baseline1（解析公式直接从 DBT/RH 统计量算 WBT）、单独 SVM、单独 RF、Prometheus 集成——RMSE 1.71/0.70/0.71/0.67°C（集成降 43%）；99.5 百分位正误差 9.7/5.0/7.0/3.5（降 60%+，极端值预报最关键的场景），99.5 百分位净误差 -6.4/-2.7/-2.8/-2.4；②与 ASHRAE 历史对比（§VI-B 回测）：对历史年份用 25 年历史观测 + 20 年 CMIP6 投影"回测"1-in-50 年温度并对比官方 ASHRAE 值——DBT 差 -2.8~19.7°C（均值 4.4°C）、WBT 差 -4.0~12.1°C（均值 1.4°C），单个站点 WBT 差可超 3°C；③固定裕量对比（Fig.9）：3°C WBT 裕量中位低估 0.69°C、25% 站点低估超 1.4°C；6°C DBT 裕量对一半机群欠配、另一半过配；④未来排放情景（§VI-C）：SSP2-4.5/SSP5-8.5 下 2044 年 50 年回返温度 WBT 高 2.7~6.8°C、DBT 高 2.0~10.7°C；⑤生产部署（§VII）：30 个数据中心，冷却容量需平均增 11%、最挑战站点增 39%（DBT）/48%（WBT），12% 站点当前即有 >2% 年概率超设计温度。
- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/推理硬件（Google 内部生产环境）。计算需求（§V）：使用公开 CMIP6 数据，避免物理气候模拟的巨大算力；ML 集成训练每站点仅需数小时，推理成本可忽略且不在关键路径上（决策在年度/十年尺度）。数据经 Google Data Commons（https://datacommons.org）访问。
- 模型是什么。数据集和bench分别是什么。
  - 模型：两级集成——SVM（RBF 核）+ RF（100 棵树、max_depth=5、min_samples_split=2）→ NN（2 隐层 16/8、ReLU、L2=0.5、输出单神经元无激活）。输入 6 特征（DBT/RH 的日 min/mean/max），输出日最大 WBT。基线模型：ASHRAE 后向统计、GSTR 缩放法（式 1）、解析 WBT 公式（baseline1-3：max DBT + 日 min/mean/max RH）。
  - 数据集：CMIP6 六模型集成（覆盖低/中/高平衡气候敏感度 ECS：<2.87K / 2.87–4K / >4K，如 NEX-GDDP-CMIP6 降尺度），0.25° 网格、每日 DBT/RH 最小/最大值到 2100 年；每站点 25 年历史观测（1965-1990 或 1999-2024 等）+ 20 年 CMIP6 前向投影。Bench：30 个生产数据中心（北美/欧洲/中东/南美/亚太），公开 5 个站点：Dublin、London、Phoenix（DBT 敏感）、Council Bluffs、Dalles（WBT 敏感）；回测用 London St. James's Park（WMO:037720, 1994-2019）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供代码仓库链接（Google 内部框架），联网搜索未发现公开仓库，无法确认开源；依赖公开数据源 Google Data Commons（CMIP6 数据）与 NOAA GFS（两周运营预报）。
  - 算法 pipeline 例子（伪代码，站点 s 的年最大 WBT 预报 → 50 年回返温度）：
    ```
    # 1. 输入：0.25° 网格上 CMIP6 投影的日最低/平均/最高 DBT 与 RH（X ∈ R^(6)）
    # 2. 第一阶段基回归器（各自独立预测日最大 WBT）
    y_rf  = RF(n_estimators=100, max_depth=5, min_samples_split=2).predict(X)  # 100 棵树叶子均值平均
    y_svm = SVR(kernel='rbf', gamma=γ).predict(X)                             # K(x,xi)=exp(-γ||x-xi||²)
    # 3. 第二阶段元回归器
    y_nn = NN(hidden=[16,8], ReLU, L2=0.5).predict(concat(X, y_rf, y_svm))     # 输出层单神经元无激活
    # 4. 站点偏差校正（5 年历史）
    bias = mean(y_nn_hist - y_obs);  y_corr = y_nn - bias
    # 5. Gumbel 拟合（式 2/3）：β = T_max-std·√6/π，µ = T_max-mean - β·γ（γ=Euler 常数 0.5772）
    # 6. N 年回返温度：T_N = T_max-mean - (√6/π)[0.5772 + ln(ln(N/(N-1)))]·T_max-std
    #    → 伦敦 2044: T_50 = 41.2°C (SSP5-8.5) vs ASHRAE 2021: 37.7°C
    ```
    张量计算细节：RF 每棵树沿特征阈值分裂、输出 = 落入叶子的训练样本均值（平均 100 棵树）；SVM RBF 核 K(x,x_i)=exp(-γ||x-x_i||²) 把 6 维输入映射到高维使非线性可分；NN 前向 h1=ReLU(W1x+b1)（16 维）、h2=ReLU(W2h1+b2)（8 维）、ŷ=w3·h2+b3（标量 WBT），L2 正则 0.5 防过拟合；式 1 缩放法 T50_3C = [ln(50)-ln(9.4)]/[ln(9.4)-ln(3.7)]·(T50_1C-T10_1C) + T50_1C 作为对比基线。

## R-Max: Extending Bélády's MIN with Prefetching to Bound Realistic Cache Performance

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 R-Max——一种"全知未来访存序列（oracle/omniscient）、但受现实硬件约束（带宽、有限 MSHR、容量、延迟、组相联结构）"的近似理想 prefetching + replacement 上界评估算法（非可实现的预取器，而是研究工具）。算法分三阶段：(1) 记录（recording）：首轮在无预取、LRU 替换的仿真中按时间戳记录对目标缓存层（L1D/L2/LLC）的访存流；(2) 处理（processing）：把访存流按缓存 set 索引分组为 per-set 列表，Alg. 1 对每个 set 用 Bélády's MIN 离线处理——prefill 空 way（按访问顺序预填）、把"已预取但尚未被 demand 访问"的访问标 prefetch、已 demand 访问的标 hold、并比较"set 内各块下次访问时间"与"不在 set 内但未来将被访问的块的时间"，若后者更早则预取后者替换最远未来使用的块；Alg. 2 据此生成 dead block counter（块在被逐出前剩余的 demand 命中次数）；(3) 重放（replay）：仿真中每个 demand 命中递减对应块计数器，计数器归零即块死亡，R-Max 按序从 delayed prefetch list 发预取替换死块。由于预取本身会改变 OoO 核访存重排与 L1 过滤效果，R-Max 迭代 record/replay 直到收敛（论文规定最多 12 次迭代）。
  - 实验比较：R-Max 分别放 L1D、L2、LLC、L1D+L2 同时，对比 (a) 无预取 LRU baseline、(b) 无预取但 MIN（Bélády）替换（L1D/L2/L3，仅 0.4%–5.5% geomean 增益，说明无预取时 MIN 在 L2 提升很小）、(c) 现有硬件预取器 SPP+PPF、Berti、Berti+SPP+PPF、AMPM、IPCP、IP-Stride、(d) SPP-Max/Berti-Max（用 R-Max 的 oracle 时序与替换决策重放 SPP/Berti 自己预测出的地址，隔离"预测能力"的影响）、(e) 不可实现的 Always Hit L1D/L2/L3（过松上界）、(f) ×100 MSHR 与 ×100 容量消融。指标：IPC speedup（相对无预取 LRU baseline）、prefetch coverage、prefetch accuracy、DRAM utilization、prefetch timeliness（cycle 差）、收敛性。主要结果：R-Max(L2) 平均 72.6%、最高 299.6%，超出 SOTA 预取器 60.8%（SPP 11.3%、Berti(L2) 11.8%、Always Hit L2 121.1%）；R-Max L2 coverage 93.2%–97.7%（SPP 仅 13.7%–41.9%）；DRAM utilization 相对 baseline 降 47.93%；剩余 miss 来自带宽/延迟/容量导致的 late prefetch 与访存重排导致的 dropped prefetch；Always Hit 上界过松（如 619.lbm：Always Hit L2 660.5% vs R-Max 10.3%）。
- 硬件平台是什么，配置是什么。
  - 无真实硬件，trace-driven 仿真平台（修改版 ChampSim），模拟 Intel Golden Cove 风格单核：4.0 GHz、6 发射乱序、512 ROB、Load/Store queue 192/114、Hashed Perceptron 分支预测、BTB 1024 sets×8 ways、5 级页表系统、非包含（non-inclusive）LLC。缓存层次（表 V）：L1I 32KB/8-way/8 MSHR、L1D 48KB/12-way/16 MSHR、L2 1.28MB/10-way/32 MSHR、LLC 3.072MB/24-way/64 MSHR；ITLB 256/8-way、DTLB 96/6-way、STLB 2048/16-way；DDR4 物理内存 1 channel/1 rank/8 banks/65536 rows×128 columns/3200 MHz/8-byte-wide 通道；块 64B、页 4KB。每级缓存 MSHR 数量与每周期 tag 检查次数均受限（带宽约束）。
- 模型是什么。数据集和bench分别是什么。
  - 非神经网络模型：R-Max 是缓存算法（MIN 扩展到预取 + dead block counter + 迭代 record/replay），不训练模型，也没有模型推理成分。benchmark 四组：SPEC CPU2017（含 619.lbm、644.nab 等）、GAP 图负载（pr.kron、cc.kron、sssp.web 等）、CVP-1 公开 trace（srv / compute int / compute fp 子集）、XSBench（蒙特卡洛反应堆模拟）。trace 来源：CVP-1 用 IPV-based LLC replacement 论文公开 trace（10.5281/zenodo.15298021，只用 public set 未用 secret set）；SPEC CPU2017 用 https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/；GAP/XSBench 用 Jamet et al. 捕获的 ChampSim trace（10.5281/zenodo.20043527）。SimPoint 采样：SPEC/GAP/XSBench 50M 指令 warmup + 250M 仿真；CVP-1 用前 20% 指令 warmup、其余仿真。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：修改版 ChampSim 仓库 https://github.com/wilsonwang881/53rd ISCA 2026 R-Max Artifact（论文附录给出），Source Code DOI 10.5281/zenodo.19688265，Apache-2.0 许可证，测试 GCC/11.3.0、11.4.0 + vcpkg 依赖；基于原版 ChampSim（https://github.com/ChampSim/ChampSim）。trace 均公开（见上）。
  - 算法 pipeline 例子（伪代码级，一次 L2 迭代）：
    - 记录：首轮无预取 LRU 仿真把 L2 物理地址访存流写成 cache phy acc.txt（时间戳、地址、访问类型）。
    - 处理（Alg. 1，per set）：对每个 set s：Acc=[]；for i in 0..x-1: 若 mem[i] 属于 set s 则 Append；set=容纳 y 对 (address,timestamp) 的容器（y=way 数）；先 prefill（未满则按序预取填满）；再逐条 i：若 Acc[i] 地址在 set 中且未被 demand 访问过 → 标 prefetch，否则标 hold；更新该块下次访问时间为当前时间（未来不再出现则 ∞）；扫描 j=i+1..m-1：若 Acc[j] 不在 set 中，找 set 中时间戳最大（最远未来使用）的块 l，若 t_l > t_{Acc[j]} 则预取 Acc[j] 替换 l 并 break。
    - 计数（Alg. 2）：对每个标 prefetch 的访问 i：c=1；向后 j=i+1.. 直到同地址且标 hold 的访问 c++（否则 break）；把 (Acc[i], c) 加入 delayed prefetch list。
    - 重放（仿真中，Alg. 3/4）：demand 命中 → 计数器 -1；归零 → 该块死亡，把下一预取移入 Pending Prefetch Queue，MSHR 可用时直接开 MSHR 发预取；无空位时预取留在 Delayed Prefetch List；demand 与 Cache Status Map 不匹配（访存重排）→ 走 λ Queue / Do Not Fill Queue（跳过填充或 LRU 替换兜底，Alg. 4）。
    - 迭代：本轮 cache phy acc.txt 作为下一轮输入，直到 IPC 收敛（≤12 轮；少数 CVP-1 的 L1D R-Max 会在两个 IPC 值间振荡，取高者）。
  - 结果例子：无预取 LRU baseline IPC 归一为 1，SPP geomean 1.113、Berti(L2) 1.118、R-Max(L2) 1.726，R-Max L2 prefetch coverage 94.1%（SPEC CPU2017 组）。

## Random-Access Hardware Sequence Compression (RST)

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 RST（Randomly-decompressible Sequence Compression with Top-utility Selection）——一种面向硬件内存压缩的新序列压缩算法，让压缩页支持"随机解压"（random access）单个 64B 块，同时保持与 state-of-the-art 页级序列压缩（LZ 家族）相当的压缩率。核心是 top-utility 序列选择：对页内所有可能的 2~5 符号序列（4096B 页最多 8×10^6 种唯一序列，见 Fig.7：4095 条 2-symbol、4094 条 3-symbol…）计算 utility（utility = 该序列若入选字典可带来的总空间节省 / 该序列消耗的字典空间），迭代式地每轮选出 utility 最高的序列填入显式序列字典（每轮选中后先对所有出现做替换 substitution，再对与选中序列重叠的序列重算 utility 的 utility update，直到字典满 128B 或无正 utility 序列，Algorithm 1）。字典总开销压到每页 128B，由三部分组成：A) 显式序列字典（本身再做序列压缩，Fig.5 布局：序列数+长度数组 L1..Ln+序列背靠背存储）；B) 位置元数据 48B→8B（每 DRAM 块内嵌 6-bit 偏移定位块内首个逻辑块 + 页级 64-bit 逐块 0/1 向量）；C) 符号字典→0B（静态符号字典：零最短码、reuse codes 编码"重复 i 位置前的符号"，Table 1 码长，最坏 12.5% 字面量开销）。输出 (D, page') = 显式序列字典 + 静态符号编码后的压缩页。
  - 实验比较：压缩率上对比 ASIC Deflate（TMCC [24] 的页级 Deflate = LZ77+Huffman）在 4KB 页粒度、同一 88 个 benchmark 与同一方法论（忽略内存 dump 中全零页）下：RST 几何平均 3.4× vs ASIC Deflate 3.3×（软件 zlib Deflate 3.84× 作参考；~10–12% 硬件/软件差距与先验观察一致，Fig.18）。Fig.19 消融显示 top-utility 选择贡献最大；128B 字典为性价比点（256B 仅小幅提升，但每次访问需多取字典，per-access 流量 192B→320B +67%）。解压延迟：18ns vs TMCC 140ns（半页），还对比 OCP Zipline ~2µs、IBM ~1µs、CDPU ~1µs（Table 3）。

- 硬件平台是什么，配置是什么。
  - 无真实芯片流片。RTL 用 Synopsys Design Compiler 在 ASAP 7nm PDK（[8]）下综合到 2.5GHz。综合的压缩器配置：32 SRAM banks（4 个子表 = 2/3/4/5 符号长度各一）、16 sets/bank、4-way 组相联；序列长度上限 5 符号（L=2 位长度字段）。面积/功耗/吞吐：压缩器 0.0923mm²、349mW 峰值、4.13GB/s；解压器 0.03mm²、91mW、13.3GB/s（trace-driven 活动向量测功耗）。系统级用 gem5+Ramulator 全系统仿真（Table 4：4 核 2.8GHz 4-wide OoO、ROB 224、1024 TLB 项、L1D/L1I 32KB、L2 256KB、L3 2MB/核共 8MB、1 通道 25.6GB/s、8 ranks、FR-FCFS、tCL/tRCD/tRP=13.75ns；atomic warmup 5s + detailed warmup 1ms + 详细仿真 4ms）。

- 模型是什么。数据集和bench分别是什么。
  - 非神经网络模型：RST 是压缩算法+专用硬件，无模型推理成分。数据集：从 88 个 benchmark 采样的内存 dump（忽略全零页），覆盖 7 类：数据库（Redis OSS v7.2、TPC、SPECjbb 2015）、GraphBig、PARSEC-3.0、SPEC CPU 2017、Spark Bench、DaCapo、Renaissance（§A.3.4，内存 dump 打包在 Zenodo artifact 中）。压缩率在 88 benchmark 上测（每类算几何平均再对 7 类取几何平均）；系统级仿真用 DyLeCT 的 12 个 application workload + 新增 1KB-strided 微基准 readStride-1K（低局部性压力测试，5.1× 加速但不计入平均）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/HEAP-Lab-VT/rst（BSD 3-Clause Clear；含 README、visualizations/、LICENSE，完整 artifact 指向 Zenodo）；Zenodo artifact https://doi.org/10.5281/zenodo.19449274（rst-isca2026-artifact.tar.gz 4.3GB，含 C++ 参考实现、SystemVerilog RTL、QEMU VM 镜像、内存 dump、预生成综合报告与 synthesis/results/ 和 evidence/signoff_summary.md）。复现：E1 压缩率 `bash regenerate_figures.sh`（~50 分钟，生成 Fig.18/19 对应 PDF；--quick 冒烟测试 ~2 分钟）；E2 硬件验证 `python3 tools/run_rst_verify.py`（见硬件架构层）。
  - 算法 pipeline 伪代码（Algorithm 1，每页）：
    ```
    Input: page（4096B）
    U = COUNTSEQS2TO5(page)            // 统计所有 2~5 符号唯一序列出现次数进 utility 表
    D = {}                              // 空序列字典
    page' = page
    while HASSPACE(D) and HASPOSITIVEUTILITYSEQUENCE(U):
        s* = FINDTOPUTILITYSEQUENCE(U)          // 每长度子表取最高 count 算 utility，跨长度取最大（Fig.8b）
        dict_idx = AddToDictionary(D, s*)        // 显式存入字典
        substitution_sites[] = SUBSTITUTION(page', s*, dict_idx)   // CAM 匹配+替换+compaction（overlap filter 去重叠）
        UTILITYUPDATE(U, page', substitution_sites)   // splice-and-cancel 防双计数，更新受影响序列 count
    return (D, page')                   // 字典 + 压缩数据
    ```
    例子（页内 4 次 "XY"、多次 "YZ" 等，Fig.1/8/9/10）：COUNTSEQS2TO5 把 2-symbol（4095 条）、3-symbol（4094 条）…唯一序列按值分组记 count；第一轮 "XY"（count=4）的 utility = S/D = [4·2·9−(4·9+D)]/D，其中 D=2·9+L（2 个 9-bit 符号 + L=2 位长度字段），S=4 次出现共省 4·2·9 bits 减去 4 个 9-bit 索引与 D；选最高者入字典，页内全部 "XY" 替换为 9-bit 字典索引 1*；选中后与 "XY" 出现位置重叠的序列（如 "YZ"）count 需重算，重复直至字典满（最多 64 个序列）或无正 utility。序列选择上限 5 符号使跨迭代仍能捕获长重复：25 字节 run 两轮即可压成单个索引（"VWXYZ"×5 → 1*×5 → 2*）。每页 >3×10^5 次操作（经局部更新+5 符号上限优化降 ~1000×），靠三模块并行硬件单周期吞吐。解压反过程：8B 位置元数据前缀和(~1ns)定位 DRAM 块 → 静态符号字典解码（数据 ~8 cycle、序列字典 ~16 cycle）→ 展开表（≤2880 bits 寄存器）→ 多 LIFO 并行展开嵌套索引（如 [2*,1*]→2*=Z1*Y→展开为 ZXYYXY），最坏 <128 cycle，平均 18ns。

## RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 RangeGuard 元数据中心的"有界近似纠错"保护算法：为每个数值生成紧凑 Range Identifier（RID，4/8-bit），把 RID 符号用 RS 码编码进标准 GPU 内存的 16-bit parity（每 256-bit block，6.25%）；写路径丢弃显式 RID、只存原数据+parity；读路径从取出数据重新生成候选 RID，与存储 parity 一起做 RS 解码，纠正"跨范围"（inter-range）错误并把损坏值替换为该范围的代表值（误差有界于范围宽度），而"范围内"（intra-range）错误不改 RID、不消耗纠错预算直接放行。核心是保护"数值落在哪个区间"这个语义元数据，而不是原始比特。
  - 实验比较：① 错误覆盖（Table III）——SE/DAE/16E/32E/FC 单故障与双故障（SE+SE、SE+DAE、SE+16E、SE+32E）下各方案 CE/DUE/SDC/BE 概率，每场景随机选故障位置、区域内每 bit 以 50% 概率翻转，重复 10^9 次；对比无保护 baseline（HBM3 风格：16-bit RS(19,17) O-ECC + 16-bit CRC16 S-ECC）、Weight Nulling [33]、VAPI [34]、RangeGuard 8b SSC、RangeGuard 4b DSC。② DNN 端到端准确率——BER 扫描下的准确率退化曲线，对比 no-protection、Weight Nulling、VAPI、RG 8b SSC、RG 4b DSC；故障按 DDR5 现场分类比例注入 SE/DAE/16E/32E（Table IV：32E 贡献 0.793×BER、16E 贡献 0.175×BER、DAE 0.023×BER、SE 0.009×BER，组合贡献匹配目标 BER）。
- 硬件平台是什么，配置是什么。
  - 算法/准确率评估为 PyTorch 推理（模型自 Hugging Face [26] 实例化，BF16 权重与激活），LLM 用 lm-evaluation-harness [47] 评测；论文未明确说明具体 GPU 型号与显存配置。硬件开销与系统性能另见硬件架构层（SystemVerilog RTL + Synopsys Design Compiler + Accel-Sim V100）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：ResNet-50（CNN）、Llama-3.1-8B、Llama-3.2-1B（LLM）；Llama-3.2-1B 约 1B 参数、156B activations。数据集/bench：ResNet-50 用 ImageNet-1k top-1；LLM 用 ARC-Easy（四选一，随机猜测 0.25）。误差注入方法：PyTorch 推理期间向权重与中间激活注入随机 bit flips（单比特位实验按位定位注入；主实验按 fault-mode 加权注入），对每个 BER×方案跑 100 次 Monte Carlo trials（图 4 单比特实验用 box-and-whisker 汇总；单比特定位实验每 trial 注入约 1.5 flipped bits @BER=10^-11）。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供官方开源仓库链接（SKKU IRIS Lab 页面 http://iris.skku.edu/publication/c97_isca_2026/ 仅公布论文；联网检索未见 GitHub 仓库/artifact，无法确认）。模型与评测工具开源：Hugging Face [26]（https://huggingface.co）、lm-evaluation-harness [47]（https://zenodo.org/records/12608602）。注入方法学基于 PyTorchFI [43]、MRFI [44]、ReaLM [45]、FIdelity [46] 等 fault-injection 框架。
  - 算法 pipeline 执行例子（写路径→存储→读路径，8 个 FP32 值为例，图 5）：① 写：每个 FP32 值按 RangeMap 查表映射到 16 个预定义范围之一 → 4-bit RID；8 个 RID 组成 RS(12,8) 数据符号，在 GF(16) 上生成 4 个 4-bit parity 符号（共 16 bit）；丢弃显式 RID，只把 256-bit 原始数据+16-bit parity 写入内存。② 存储错误：低阶 mantissa 翻转使值仍在原范围 → intra-range，RID 不变（不耗纠错预算）；sign/exponent 翻转使值跳到另一范围 → inter-range，RID 变（有害）。③ 读：从取回数据重新生成 8 个候选 RID（可能含错）→ 与存储 parity 送入 RS 解码器，纠正最多 t=2 个错误 RID 符号（RG 4b DSC）或 t=1（RG 8b SSC）→ 被纠正位置用该范围代表值替换（每值位置一个 mux：解码器未标错或 intra-range 错误则直通原始值，inter-range 错误则用代表值）→ 误差限制在范围宽度内；超过 t 个 inter-range 错误报 DUE（罕见 SDC）。RangeMap 构造（简单映射，图 6）：对零均值高斯（标准差 σ）求 exponent 概率质量 P(e=k)=2[Φ(2^(k+1-127)/σ)−Φ(2^(k-127)/σ)]，把 exponent 域 E={0..255} 划分为 K 个连续区间、每区间赋代表 exponent ê_k，最小化 L=Σ_k Σ_{e=l_k}^{r_k} P(e)|f(e)−f(ê_k)|，f(e)=2^(e-127)；例（σ=4，4-entry 表，Table II）：[0,127]→0.5、128→2、129→4、[130,255]→8。理想映射用 L1 最优标量量化（Lloyd-Max 类）：σ=1 归一化时阈值 (−0.8217, 0, 0.8217)、代表值 (−1.2657, −0.3778, 0.3778, 1.2657)，可缩放复用于任意格式（FP32/BF16/INT8）。
  - 效果：ResNet/Llama-3.1 在 BER≤10^-7、Llama-3.2 在 BER≤10^-6 保持近 baseline 准确率（无保护分别在 ~10^-8/10^-10/10^-10 崩溃）；16-bit parity 预算下 RG 4b DSC 每 block 容忍两个 inter-range 错误、覆盖 64+ 翻转数据 bit（bit 级方案同预算仅 8 bit）。8b SSC 在 ResNet/Llama-3.2 更优（16-entry RangeMap 重建更精细），4b DSC 在 Llama-3.1 更优（抗重叠故障更强）。

## SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 SLICE——一种纯客户端（client-only）视频超分（SR）加速框架：利用标准 H.264 码流中解码器本就要解析的码流元数据（运动矢量 MV、像素域残差、频域残差）做 patch 级"选择性推理"调度，在满足 30FPS 延迟预算（33ms/帧）的同时把 SR 计算量降到最低。SLICE 把每个 inter 编码帧划分为等尺寸 16×16 非重叠 patch，对每个 patch 从三类策略中选一：① Reuse（MV 均值=0 且像素域残差均值=0 的 patch，直接从上一帧 HR cache 拷贝，不做任何推理）；② SR inference（按 codec 引导 score 取 TopK 的 patch，推理面积预算为帧面积的 35%）；③ Interpolation（其余 patch 用 GPU 像素插值）。Intra 编码帧（每个 GOP 开头，占比小）做全帧 SR 推理。核心动机来自两条相关性发现：(a) 频域残差中高频能量占比与 SR 相对插值的 PSNR 增益强正相关（Fig.7b），(b) 低运动（小 MV）patch 的 SR 增益小（Fig.7c）、且零 MV+零残差区域可直接复用上一帧超分结果（Fig.8 复用 ΔMSE<0.2%）。
  - 实验比较：三种主方案——(1) Per-frame inference（每帧全帧 EDSR）；(2) Server-orchestrated SR（无 per-video 训练：server 离线决定哪些帧做 SR，其余帧用类解码器重构，质量约束调到与 SLICE 相同 PSNR 以公平比较）；(3) SLICE（提出）。另含 SLICE-noreuse（每帧固定 patch 数做 SR、无复用，用于单独验证复用收益）。还比较了相关 baseline（Table I）：Random（随机选 35% patch）、Frame-skip（35% 帧做全帧 SR）、MV-only、Residual-only（各用单一码流信号选 patch）、Hybrid（帧跳过+patch 选择混合，SR 预算 70%）。指标：FPS（是否满足 30FPS 实时预算）、相对像素插值的 PSNR 增益、能量消耗（Tegrastats 实测）。主要结果：SLICE 相对 per-frame inference 帧率提升 2.72×、能量降低 62.57%、PSNR 仅降 0.35dB；SLICE-noreuse 降 0.78dB（量化了复用的质量贡献）；I2 视频复用率 79.43% 时达 52.19 FPS；Xiph 数据集上 2.82× 提升、0.35dB 损失。设计空间探索：推理面积比 35%（40% 仅多 0.08dB）、patch 大小 16×16（32×32 接受野更大但复用率下降）、GOP 120、score 权重 α=0.9/β=0.1（残差比 MV 更可靠）。
- 硬件平台是什么，配置是什么。
  - NVIDIA Jetson AGX Orin（真实设备端到端测量，非仿真）：PyTorch 在 GPU 上以 FP16 跑 EDSR 推理；patch 分析（AvgPool2D 池化）、TopK 选择、SR 推理、插值与合并全部在 GPU 上执行，避免 CPU 往返。能量用 Jetson 自带 Tegrastats 工具测整机功耗并对评估区间取平均。码流元数据提取在主机端用扩展版 Compressed Video Reader（基于 FFmpeg 补丁构建）模拟硬件解码器未暴露的码流侧信号（解码本身仍走标准 bitstream）。另外论文用桌面平台（Intel Core i7-14700K CPU + NVIDIA GeForce RTX 3080 Ti GPU）实测 server-orchestrated 方案的离线帧选择成本为每视频平均 24.95 分钟（用于动机论证）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：EDSR（Enhanced Deep Residual Networks），4× 上采样因子，全部 SR 推理 FP16；质量参照上采样方法为 bicubic 插值。数据集/bench：评估用 NEMO 论文的真实 YouTube 视频（10 个类别各选 2 个视频），每视频编码为 30 FPS 的 270p（650 kbps）与 1080p（4400 kbps）两档，LR 上采样后以原始 1080p 帧为质量参考；GOP=120，报告每视频前 5 分钟结果；泛化性验证用 Xiph 测试媒体数据集；复用机会的统计分析用 Vimeo90K（1000 视频）与 Kinetics（K400/K600/K700）。质量指标为 PSNR。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：SLICE 本身论文未提供官方仓库链接，联网检索亦未能确认（IEEE Xplore 11617444，2026-08 上线，代码可能尚未发布）。用到的开源组件：Compressed Video Reader（https://github.com/Yaojie-Shen/Compressed-Video-Reader，MIT 许可，基于 FFmpeg 补丁构建，Python API `cv_reader.read_video(video_path=..., with_residual=True)` 或 CLI `cv_reader <video> <output>`，专门提取 H.264 的 MV 与残差）、FFmpeg、EDSR（公开实现）、Tegrastats（Jetson 自带）、NEMO（论文 [5] 项目）。
  - 算法 pipeline 执行例子（一个 inter 编码帧，270p→1080p，Algorithm 1/2 伪代码级）：
    ```
    # 1) 解码+元数据提取：SoC 硬件解码器重建帧；扩展 CVR 解析 bitstream 得到
    #    G^mv（4×4 块粒度 MV 网格）、G^pix（像素域残差网格）、G^hf/G^t（频域残差高频/总能量网格）
    # 2) Patch Analysis（Algorithm 2，全 GPU，P=16 时各核一次 AvgPool2D 完成）：
    #    mv_mean   = AvgPool2D(G^mv,  kernel=P/4)          # MV 网格 4×4 块粒度 → 核/步长 4
    #    res_mean  = AvgPool2D(|G^pix|, kernel=P)          # 像素粒度 → 核/步长 16
    #    hf_ratio  = AvgPool2D(G^hf, P/4) / AvgPool2D(G^t, P/4)   # 频域网格 4×4 块粒度
    #    M^reuse   = (mv_mean==0) AND (res_mean==0)        # 复用 mask（用像素域残差防 intra 块被复用）
    #    score     = 0.9*hf_ratio + 0.1*(1 - clip(mv_mean/10, 0, 1))
    #    M^SR      = TopK(score, k=35%)                    # GPU TopK kernel 选前 35% 面积 patch
    # 3) Patchwise Upscale：unfold 把 patch 网格转成紧凑 GPU tensor，按 M^SR gather 出需
    #    推理的 patch 组成 batch → EDSR(FP16) 一次/少数几次 forward；reuse patch 从 GPU 上的
    #    HR cache 按坐标直拷；其余 patch GPU 插值
    # 4) Merging：全部在 GPU 上按行分带（row-wise banded，水平相邻 patch 合成连续带）
    #    拷贝合并 → 写 HR framebuffer；SR 结果留在 GPU cache 供下帧复用
    # 张量化示例：270p（480×270）→ 30×17 个 16×16 patch → TopK 选约 35%≈178 个 patch
    #    → 组成 (178, 3, 16, 16) batch 单次 EDSR forward（4× 输出 64×64）→
    #    reuse patch 直拷、其余 bicubic → 合并为 1080p 帧
    ```

## SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 expert substitution（专家替换）算法：针对 fine-grained MoE（DeepSeekMoE、Qwen2-57B-A14B、XVERSE-MoE），利用 router gate score 将激活专家分为 top-score 专家（score > (1+α)S_{k+1}）与 low-score 专家（S_{k+1} ≤ score < (1+α)S_{k+1}），把 low-score 专家替换为 GPU 显存中已缓存、gate score 落在 [(1−α)S_{k+1}, S_{k+1}] 的未激活专家，在几乎不损失精度的前提下减少 CPU→GPU 的 PCIe 专家加载与 CPU 专家计算。算法组件：①expert-cache router（Algorithm 1）：按 α 阈值分类并选替换候选，使 |E_l \ G| 与 |E_s| 匹配（替换不足的 low-score 专家回退为 PCIe 加载或 CPU 计算）；②score-aware cache eviction：按过去 n 次迭代的平均 activation score 淘汰最低分专家（Eq.3，替代 LRU），并对当层被选中专家加 protection shield 防止用前被淘汰；③top-score prefetching：用 GPU 常驻共享专家+缓存专家计算 hidden state 预测下一层 top-score 专家并预取（命中率约 82%，95% 概率为 active）；④CPU-assisted task load scheduling（Algorithm 2）：双指针按 C_load/C_CPU 成本最小化 max(T_load, T_CPU)。优化目标为逐层最大化 max |G∩E_a| + min(|E_l\G|, |E_s|)（Eq.1/2）；超参 α 通过 min_α A(α) s.t. T(α) ≤ R（TPOT 预算）多项式拟合+一维搜索选取。
  - 实验比较：baseline 为 4 个支持显存不足时本地运行 MoE 的系统——MoE-infinity（activation-aware 预取，依赖历史 router 数据）、llama.cpp（纯 CPU 推理）、DeepSpeed（layer-wise 加载、静态缓存）、HybriMoE（CPU-GPU 调度+缓存管理，基于 ktransformer 架构自带量化，论文移除了量化效果做公平对比）；另对比 expert-skipping（直接丢弃将被替换的专家，激活数动态变化）与 GPTQ INT8 量化（配置为与 SMoE 相同 GPU cache hit rate）。指标：TPOT（解码，batch=1/3）、TTFT（prefill，batch=1）、GPU expert cache ratio、Accuracy/GPT-4 Score/pass@1。主要结果：相对最优 baseline，TPOT 平均降 24%（batch=1）/35%（batch=3），S3 下 48%（batch=3）/34%（batch=1）；GPU cache ratio 相对提升 ≥65%、命中率 >60%（Qwen2-57B 达 71%）；TTFT 平均降 11%；α<0.35 时精度几乎无损（部分数据集反而提升，归因于抑制 low-score 专家的 noisy activation）。Ablation（batch=1，逐步增量）：baseline=LRU offloading（无 router/prefetch/cache 策略）；+CE（cache eviction）TPOT −8%、cache ratio +11%；+CR（expert-cache router）再 −20%、+60%；+Pre（prefetching）再 −14%、+12%（PCIe 时间略增但被重叠掩盖）；+BA（CPU-assisted scheduling）再 −34%、−3%。
  - 硬件平台是什么，配置是什么。
  - 单卡边缘/低端 GPU（无量化、lossless 推理）：S1 用 NVIDIA RTX 3080 Ti 12GB 跑 deepseek-moe-16b（31GB）；S2 用 NVIDIA RTX 4060 Ti 16GB 跑 XVERSE-MoE-A4.2B（49GB）；S3 用 NVIDIA A6000 48GB 跑 Qwen2-57B-A14B（107GB）。Edge/Legacy 平台为 PCIe 3.0 + Intel E5-2683 v3；High-end 平台为 PCIe 4.0 + Intel Xeon Gold 6444Y；S3 复现需约 150GB CPU 内存承载模型权重加载/卸载过程。
  - 模型是什么。数据集和bench分别是什么。
  - 模型：deepseek-moe-16b-base（S1）、Qwen2-57B-A14B-Instruct（S3）、XVERSE-MoE-A4.2B-Chat（S2），均为 fine-grained MoE（含 shared experts）。数据集/bench：Gaokao（Math_I/Math_II/History/Biology）、MMLU（College_computer_science/Management/International_law/Logical_fallacies）、TriviaQA、RACE-mid、WiC、GSM8K、MT-Bench（GPT-4 Score 1–10）、HumanEval（pass@1）；每数据集采样约 1000 条、prompt 取自 OpenCompass；精度评估用 OpenCompass 框架（Gaokao/triviaQA/WiC/Race-mid/gsm8k/MMLU）。
  - 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：已开源 https://github.com/goingshr/SMoE（figshare 备份 https://doi.org/10.6084/m9.figshare.31982136），artifact 聚焦复现 S3（Qwen2-57B-A14B）的 TPOT 与 GPU cache hit ratio（对应 Fig.12/13）。环境：Python 3.13 free-threading（no-GIL）conda 环境、Ubuntu 专属 dependency.sh（apt/snap/sudo、Rust 工具链、tokenizers 源码编译）；运行脚本 run.sh 以大写环境变量传参（MODEL_NAME/MODEL_PATH/CONFIG_PATH/DATASET_PATH/INPUT_NUM/BATCH_SIZE/OUTPUT_LEN/GPU_MEM/CPU_CORES/LOG_DIR 等），直接入口 main.py；--alpha 为论文文档中的替换阈值参数（S3 默认 0.25；仓库实际 config 用 replaceScoreRatio 等字段：replaceScoreRatio/window_size（null=LRU）/if_prefetch/if_usecpu/if_replace）。安装与运行例子：
    ```
    conda create -n SMoE python=3.13 python-freethreading -c conda-forge
    conda activate SMoE
    bash dependency.sh   # sudo；装 Rust 工具链并源码编译 tokenizers（no-GIL 无预编译 wheel）
    pip install -r requirements.txt
    MODEL_NAME=qwenmoe MODEL_PATH=parameters/qwenmoe GPU_MEM=43 \
      CONFIG_PATH=configs/qwen2moe_config.json bash run.sh
    # 或直接：
    python main.py --model_name qwenmoe --model_path parameters/qwenmoe \
      --config_path configs/qwen2moe_config.json --dataset_path gaokao_math_ii \
      --input_num 20 --output_len 60 --cpu_cores 4 --GPU_mem 43
    ```
    算法 pipeline 执行例子（Expert-Cache Router，Algorithm 1 伪代码级，单 token t）：
    ```
    # 1) gate 输出该层全部 expert 的 score 向量 S_t，降序排序；S_{k+1} = (k+1)-th score
    # 2) 分档：T=(1+α)S_{k+1}（top-score 线）、L=S_{k+1}、R=(1−α)S_{k+1}
    #    - score > T            → top-score 专家，直接保留进输出 O[t]（C 集）
    #    - L ≤ score < T        → low-score 专家（B_t 集 = E_l）
    #    - R ≤ score < L 且已在 GPU 或属于 C 集 → 可替换候选（A_t 集 = E_s）
    # 3) 替换决策：若 |A_t| ≥ |B_t|，取 A_t 中最高分的 |B_t| 个替换 low-score 进 O[t]；
    #    否则全部 A_t 替换，剩余 |B_t|−|A_t| 个 low-score 专家留给 PCIe 加载/CPU 计算
    # 4) score-aware eviction（Eq.3）：对 expert i 维护近 n 次迭代平均 activation score，
    #    淘汰 argmin_i (Σ_{k=max(1,j-n)}^{j} S_{i,k})/(窗口长度)；当层被选中的 expert
    #    加 protection shield（层计算完成后自动解除）避免用前被淘汰
    # 5) 输出：GPU 直接计算替换后 expert 集合；top-score 专家由 prefetch 提前加载
    ```
    （论文未涉及编译框架、硬件架构、芯片设计层次。）

## STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是面向内存受限单卡 MoE 推理的混合静态-动态优化框架 STEP，核心为三项算法：(1) 离线 spatial-aware expert allocation（空间感知专家分配）：用校准数据集逐层收集 top-k routing score，按归一化权重阈值 θ 识别低贡献专家并削减该层 routed 专家数 k_l（例如 top-4 权重 0.62/0.21/0.13/0.04 → 分配 3 个专家；0.72/0.18/0.08/0.02 → 分配 2 个），剩余专家权重重新归一化保持输出一致；默认 θ：Mixtral 0.25、Qwen 0.13、DeepSeek 0.07；(2) 在线 temporal-aware prefetching / cached temporary shared experts（缓存临时共享专家）：把输出序列切成 token 窗口，窗口内每步跟踪 top-2k 专家（而非 top-k）并投票，窗口结束时按票数选出 top-c 专家作为下个窗口的"临时共享专家"，使每层有效结构从 j shared + k routed 变为 j+c shared + k-c routed，动态加载从每步 k 个降到 k−c 个；临时共享专家常驻 GPU 但仅被 gating 选中时才计算，且在固定 cache 预算下替换低使用率专家、不增加显存；(3) token-aware adaptive candidate window selection（Token 感知自适应候选窗口）：为每层维护奖励分 r_i 与窗口大小 d_i，按 prefetch 准确率自适应调窗——准确率 > th_s(75%) 时 r_i+1、累计达 τ(3 或 4) 则窗口翻倍；准确率 < th_f(40%) 时窗口减半；介于两者之间窗口不变；窗口缩到 1 时禁用实际 prefetch 但继续统计投票。总目标最小化专家加载时间 T_load = S·Σ_l(k_l − p_l·R_l)·t_expert（Eq.1）。
  - 实验比较：① 准确率/生成质量与 prefetch hit rate——以平均每层激活专家数（Avg. #Experts，由 θ 控制）与窗口长度（Window Size）为自变量，对比 Origin（未优化）与不同 (Avg. #Experts, Window Size) 组合（Mixtral 2→1.75/1.5；Qwen 4→3/2.5/2；DeepSeek 6→5/4/3），指标为 MMLU/Arc-e/PIQA/WinoGrande Accuracy、CNN/DM 与 LongBench Rouge-L + Prefetch Hit Rate（Table II–IV）；② 端到端 prefill（TTFT）与 decode（TPOT/tok/s）速度对比 llama.cpp、AdapMoE、HybriMoE、DAOP、APTMoE、MoE-Lightning 六个 baseline，prefill 平均几何平均加速 3.12×/1.97×/1.52×/1.07×/1.07×/1.03×，decode 为 1.54×/2.22×/1.39×/1.15×/1.10×/1.25×（Fig.10/11）；③ 消融：spatial allocation 单独 1.46×、+prefetch 1.52×、+adaptive window 2.22×、全量 3.12×（Fig.13）；④ 与 MoE-I2 压缩、APTMoE offloading 的正交性（Table V/VI：MoE-I2+STEP decode 24.1 tok/s、TTFT 470.3ms；APTMoE+STEP 21.3 tok/s、TTFT 531.2ms）；⑤ 批量 1–8、硬件 V100/A100/H20 敏感性分析（Fig.18/19，STEP 始终 ≥1.3×）。
- 硬件平台是什么，配置是什么。
  - 4× NVIDIA A100 80GB GPU，经 PCIe 4.0、64 GB/s switch 互联；AMD EPYC 7542 32-core CPU，512GB 主存；GPU-GPU 与 GPU-CPU 通信均走 PCIe（实验刻意不用 NVLink peer-GPU 共享以保证公平）。硬件敏感性实验另覆盖 NVIDIA V100 与 H20。
- 模型是什么。数据集和bench分别是什么。
  - 模型（3 个代表性 MoE，见表 I）：Mixtral-8x7B-Instruct（32 层、0 shared、8 routed、top-2、激活 13B/总量 46.7B、无 shared expert）；DeepSeek-V2-Lite-Chat（26 层、2 shared、64 routed、top-6、激活 2.7B/总量 14.3B，routed expert (2048,1408)）；Qwen1.5-MoE-A2.7B（24 层、4 shared、60 routed、top-4、激活 2.4B/总量 16B）。这些模型专家参数总量超过单 GPU 显存，必须 offloading。
  - 数据集/bench：常识推理 ARC、PIQA、WinoGrande（Accuracy），MMLU（Accuracy），摘要生成 CNN/DM 与 Longbench(Summarization)（Rouge-L + Prefetch Hit Rate）；延迟评估从多个数据集采样定长 trace 保证跨模型/策略可比。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供官方开源链接，联网检索（2026-08）未能确认 STEP 的公开代码仓库；实现基于 Hugging Face Transformers 库，batch size=1 模拟实时推理。算法 pipeline 执行例子（伪代码级，以 DeepSeek-V2-Lite top-6、校准后分配 4 个 routed + 2 个临时 shared 为例）：
    ```
    # == 离线阶段（spatial-aware allocation，per layer l）==
    # 1) 校准数据集前向，收集每层 top-k routing score 分布
    # 2) 对第 l 层按归一化权重阈值 θ 剪枝：score/Σscore < θ 的低贡献专家被移除，
    #    有效 routed 数 k_l 下降（如 6→4），剩余权重在 softmax 后重新归一化
    #    （y_routed = Σ w_i^r · E_i(x)，权重重归一保证输出一致性）
    # == 在线阶段（decode，per window of size d_i）==
    # 3) 每 decode step：gating 对全部专家算 score（含已当选的临时 shared），
    #    记录每步 top-2k 专家并投票（出现即得 1 票，反映频率与选择强度）
    # 4) 窗口结束：按票数选 top-c（如 c=2）专家为下一窗口"临时共享专家"，
    #    结构变为 (j+c) shared + (k−c) routed；shared 全部在计算开始前预取到 GPU，
    #    每步动态加载从 k 降到 k−c
    # 5) 窗口末评估 prefetch 准确率 → 更新 r_i/d_i（>75% 累计 τ 次翻倍窗口、
    #    <40% 减半、否则不变）；d_i=1 时暂停 prefetch 仅统计
    # 6) 计算 y = y_shared(含临时 shared 的平均/加权和) + y_routed(top-(k−c))
    # 张量化示例：一个 decode 步 → gating 输出 (64,) score 向量 → top-2k=top-12 投票
    #   → 2 个临时 shared 权重 (2048,1408) 已常驻 GPU，其余被选 routed 专家经 PCIe 加载
    #   → 计算 4 个 routed 专家 GEMM（每专家 (2048,1408)×(1408,) 激活向量）+
    #   2 shared GEMM，全部输出按 softmax 权重聚合
    ```
    （论文未涉及编译框架、硬件架构 RTL、芯片设计层次。）

## Shining Light on Silicon Photonic DNN Accelerators

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现：在 DNN 推理精度评估中，通过 PyTorch hook 工具在模型前向中注入 SiPh 加速器的模拟非理想因素（调制器 E/O 非线性、ISI 导致的时序噪声、光路损耗导致的 AWGN 噪声），量化 3/4-bit 量化模型在 SiPh 加速器上的精度/困惑度损失。实验比较：(1) 非线性（MRM/MZM 调制器传递函数，不同偏置 ER 20/15/14 dB）对 ResNet50/MobileNetV2 在 ImageNet 上 3/4-bit 精度的影响；(2) ISI（不同 TX 驱动 -3dB 带宽 5/10 GHz、不同阵列尺寸 32×32~256×256）对精度的影响；(3) 噪声（不同 SNR/bit-precision、不同阵列尺寸）对 3/4/8-bit 模型精度的影响；(4) Qwen2.5-7B-instruct-AWQ 在不同激活量化精度（int8~int4）与粒度（per-tensor/per-feature/per-block）下的 Wikitext-2 困惑度。
- 硬件平台是什么，配置是什么。
  - 本层为精度评估（不跑真实硬件）：模型精度由 PyTorch 在通用服务器上评估。SiPh 加速器参数作为输入约束：时钟频率 5/10 GHz、MAC 精度 3/4-bit、波长数 8-128、阵列尺寸 32×32~256×256、TX 驱动 -3dB 带宽 5/10/15 GHz、调制器 ER 14/15/20 dB、噪声方差由光路损耗预算推导（Pmin = 2×SNR×In,rms/R，线性 TIA 输入噪声 1.3 μA）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：ResNet50（26M 参数）与 MobileNetV2（3.4M 参数），采用量化感知训练（QAT [81]）到 3/4-bit（另有 8-bit 对照）；Qwen2.5-7B-instruct 语言模型（fp16 基线 → AWQ int4 权重 [82] → 激活量化到 int4-int8）。数据集：ImageNet（图像分类 top-1 精度）、Wikitext-2（困惑度）。bench：ImageNet top-1 分类精度、Wikitext-2 perplexity（baseline 6.79）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确说明是否开源其评估代码。所用组件开源：PyTorch（https://github.com/pytorch/pytorch，hook 工具 [77]）、AWQ（https://github.com/mit-han-lab/llm-awq，[82]）、Qwen2.5-7B-Instruct（https://huggingface.co/Qwen/Qwen2.5-7B-Instruct，[80]）、ImageNet/Wikitext-2 为标准公开数据集。
  - 算法pipeline 伪代码（每 DNN 层，噪声注入方式见图 8）：
    ```
    # 输入：量化模型（图像模型 QAT 到 3/4-bit；LLM 为 AWQ int4 权重 + 激活量化）
    for layer in model:
        # 1) activation/weight/output 量化器保证低比特（LLM 激活按 per-tensor /
        #    per-feature(每 hidden dim) / per-block(14 batch × 74 hidden dim) 的
        #    affine scale+zero-point 量化 [85]；QAT 同时学习最优动态范围）
        x_q = Q_act(x); w_q = Q_wt(w)
        # 2) forward pre-hook：注入 E/O 调制器静态非线性（取决于调制器与偏置）
        #    MRM: P_out = P_in(1 - K_ring/(1 + βV²))；MZM: P_out = P_in/2(1 + cos(πV/Vπ))
        y = MAC(x_q, w_q)              # 光域加权和（点积）
        # 3) forward post-hook：注入 ISI 推导的 post-MAC 条件输出分布（高斯拟合，
        #    来自 Cadence Spectre 瞬态仿真眼图，随 TX 驱动带宽/阵列尺寸变化）
        y += N(0, σ²_ISI)
        # 4) forward post-hook：注入 AWGN（σ² 由光损耗预算推导，Pmin=2·SNR·In,rms/R；
        #    按点积长度/通道数添加多个噪声样本 [86]）
        y += Σ_{i=1..n_samples} N(0, σ²_opt)
        # 5) 输出量化器
        y_out = Q_out(y)
    # 评估：ImageNet top-1 精度 / Wikitext-2 perplexity
    ```
  - 张量计算示例（一个卷积/线性层，dot-product 长度 d、输出通道 C）：激活张量 (N,C,H,W) 经 Q_act 量化到 4-bit（scale=range/15 均匀量化，QAT 学习的最优动态范围）→ 与 int4 权重矩阵做整数 MAC → 输出累加值按 SiPh 阵列参数叠加 n_samples≈C 个独立 AWGN 样本（模拟每通道经不同 λ/PD 的噪声累加）→ 输出量化器 Q_out 回到 4-bit → 统计 top-1 精度。关键结论：任何单一非理想因素未补偿都会使 ResNet50 4-bit ImageNet 精度相对理想 4-bit 下降 >10%；3-bit 下 MZM 余弦非线性使 ResNet50 掉 >25%、MobileNetV2 崩溃到近零；Qwen2.5-7B 激活量化到 int5/int4 时困惑度从 6.79（fp16）剧增（per-block 粒度最优仍达 182/2129，int4 时 per-tensor 达 120 万），指示 LLM 部署 SiPh 需进一步算法/器件级改进。

## SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 属于算法pipeline的实现：两个协同的 LLM 压缩算法。(1) SingularBit-W：离线权重压缩，对每个线性层权重做 SVD 分解 W^T=Σσ_i u_i v_i^T，依据奇异值指数衰减把 U/V^T 列/行分成 K=4 个精度区域（4/3/2/1-bit），把 GPTQ 错误反馈（Hessian 引导的逐块量化+误差补偿）施加到 SVD 分量上，并用 ARB-LLM 分层二进制量化表示每个精度区域；(2) SingularBit-KV：在线 KV cache 压缩，在 token 维按注意力重要性分配 5 级 bitwidth（b~b+4），在 rank 维按 K/V 投影权重的奇异值边界做逐 token 内混合精度，缓存中间表示 K'=xU_{W_K}、V'=xU_{W_V}，attention 时用 V^T 重构。实验比较：权重压缩在 Wikitext-2/C4 困惑度与 6 个常识推理 benchmark 平均 zero-shot 精度上对比 RTN、GPTQ、AWQ、OmniQuant、MagR+OPTQ、SpQR、SqueezeLLM、OliVe、GuidedQuant（2/3/4-bit，含 ultra-low-bit 2-bit 极端区）；KV 压缩在 CoQA、TruthfulQA 生成质量与 GSM8K 数学推理的 Accuracy+压缩率（CR%）上对比 KIVI、GEAR、KVQuant、PALU、ReCalKV、ZipCache（KV2/KV3 标称精度）；并在 LongBench（prefill-heavy 长上下文）与 reasoning（decode-heavy）场景验证 W+K 联合（SingularBit-WKV）优于单独优化与 prior 组合（GuidedQuant+ReCalKV）。
- 硬件平台是什么，配置是什么。
  - 算法评估在通用 GPU 上执行（论文以 NVIDIA A100 80GB、2TB/s 内存带宽作为硬件对照基线；所有加速器配置在 iso-resource 下归一化到 A100 的峰值吞吐与内存带宽以做公平比较）。算法层本身不依赖专用硬件即可获得压缩收益（论文单独给出 GPU 上仅算法的能耗降低 25.5–37.3%）。
- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA（7B/13B/30B）、Llama 2（7B/13B）、Llama 3（8B-Instruct）、DeepSeek-R1-Distill-Qwen-1.5B、DeepSeek-R1-Distill-LLaMA-8B。数据集/bench：Wikitext-2 与 C4（perplexity，上下文窗口 2048 token）；PIQA、ARC-Easy、ARC-Challenge、BoolQ、HellaSwag、WinoGrande（zero-shot 常识推理平均精度）；CoQA（对话阅读理解）、TruthfulQA（事实性生成质量）；GSM8K（多步数学推理）；LongBench（8 个 prefill-heavy 长上下文任务：Qasper、QMSum、MultiNews、TREC、TriviaQA、SAMSum、LCC、RepoBench-P）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未明确说明开源情况；联网搜索未找到 SingularBit 的公开仓库或 arXiv 预印本（可能为近期未公开论文），无法确认代码链接。所用 baseline 均为开源工作（GPTQ、AWQ、KIVI、GEAR 等），模型（LLaMA/Llama 2/Llama 3/DeepSeek-R1-Distill）与数据集（Wikitext-2、C4、GSM8K、LongBench 等）均为公开资源。
  - SingularBit-W 算法伪代码（对应论文 Algorithm 1，calibration 输入 x ∈ R^{N×ich}）：
    ```
    # 输入：权重 W ∈ R^{och×ich}，calibration 激活 x ∈ R^{N×ich}，目标平均精度 B_avg
    U, S, V^T = SVD(W^T)                       # W^T = U·S·V^T, σ_i = diag(S)
    # 区域边界：累计尾部和 C_i = Σ_{k=i}^{r-1}σ_k / Σ_{k=0}^{r-1}σ_k
    # K=4 区域，C_{r1}=(1-p)^3, C_{r2}=(1-p)^2, C_{r3}=(1-p)，
    # 由 B_avg = (1/R)·Σ_{k=1..K} b_k·(r_k-r_{k-1}) 解析解出 p（无需启发式搜索）
    {r1,r2,r3} = BOUNDARY(S, B_avg)
    H_U = x^T·x                                # U 的 Hessian
    for idx in range(ich, 0, -blocksize):      # 逆序逐块量化 U（GPTQ 错误反馈）
        α_r, α_c, B = QUANTIZE(U, {r1,r2,r3}, idx)   # 分层二进制量化：每 bit 一层 {-1,+1} 基
        Û = Σ_i α_r,i·α_c,i·B_i
        E_u = U - Û
        U[剩余] -= E_u · H_U^{-1}              # 误差反馈到未量化参数
    H_VT = S·Û^T·H_U·Û·S                        # 考虑已量化 Û 的有效 Hessian（z=xÛS ⇒ H=z^T z）
    for idx in range(0, r, blocksize):          # 正序逐块量化 V^T
        α_r, α_c, B = QUANTIZE(V^T, {r1,r2,r3}, idx)
        V̂^T = Σ_i α_r,i·α_c,i·B_i
        E_v = V^T - V̂^T
        V^T[剩余] -= E_v · H_VT^{-1}
    return Û, V̂^T                                # 存 Σ_i α_r,i·α_c,i·B_i 层级表示
    ```
    - 张量计算示例（一个线性层，R 个 rank、K=4 区域）：W^T≈Û·S·V̂^T，其中 Û 的前 r1 列用 4-bit 分层二进制基表示（Û_{4bit}=Σ_{i=1..4}α_{r,i}α_{c,i}B_i，B_i∈{-1,+1}^{och×ich}），r1~r2 用 3-bit、r2~r3 用 2-bit、尾部 1-bit。推理时 Wx 以 bit-serial 方式按位执行：先算 1-bit 基 B_i 与激活的部分和，再按位偏移（2^i 缩放）累加，bitwidth 直接换算成硬件延迟与能耗。论文关键结论：2-bit 平均精度下 RTN/GPTQ/AWQ 困惑度崩溃（如 LLaMA-7B Wiki 1.9e3/44.01/2.6e5 vs SingularBit-W 7.56），SingularBit-W 在全部模型/bitwidth 上构成 Pareto 前沿。
  - SingularBit-KV 算法伪代码（每 decode 步）：
    ```
    # 每步 t：维护 recent-k 注意力窗口 M_t ∈ R^{k×N_t}（k=128）
    ã_t = normalize(headwise_max_pool(A_t))     # A_t ∈ R^{H×N_t}，逐 head 取 max 后归一化
    M_t.append(ã_t); M_t.evict_oldest()
    I_i = max_{j∈[t-k+1,t]} ã_j[i]              # 每 token 重要性 = 近 k 步 query 方向最大值（保守）
    # token 精度策略：5 级 bitwidth b..b+4，容量线性递增 l_i·2^{b+i}=m·i+c，Σl_i=1 解出边界 s_i
    b_i = map(I_i, thresholds{s_0..s_5})        # 得到该 token 最大 bitwidth
    # rank 精度策略：token 内按 SingularBit-W 边界 {r1,r2,r3} 从 b_i 起逐级降精度
    K' = x·U_{W_K}; V' = x·U_{W_V}              # 只乘 U，缓存中间表示（无在线 SVD）
    qKV = quantize(K', V', per_token_b_i, rank_boundaries)   # FP16→INTx 逐级降精度
    packed = bitpack(qKV, per_token_b_i)        # 无 padding 紧凑打包，路由到物理地址
    # attention 时：K = K'·V_{W_K}^T, V = V'·V_{W_V}^T 在 tensor core 上重构（延迟 +5%@ctx64, +2%@ctx2048）
    ```
    - 与 baseline 的关键差异：KIVI/KVQuant 只做单一维度量化，ZipCache 只按 token 显著性分 bitwidth，PALU/ReCalKV 只做低秩投影；SingularBit-KV 在 token 维（注意力重要性，5 级）与 rank 维（奇异值重要性）二维混合精度，且 KV2 下仍保持 CoQA 61.5%（FP16 63.5%，仅 -2.0%）与 GSM8K 0.81~0.85（baselines 多掉到 0–30%）。

## TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现为 Topology Dependency Subgraph (TDS) 拓扑感知稀疏化与合并方法：把图 Transformer 的全局 O(N²) 注意力近似为在稀疏子图 TDS 上的注意力，把每个顶点 attend 的边数从 O(N) 降到平均 O(m·log_m N)（m=2 时即 O(log N)），总边数 O(m·N·log_m N)=O(N log N)。TDS 由三类边构成：(1) original edges——保留输入图的原始局部邻域结构；(2) fusion edges——自底向上的分层聚合边：沿原生 1D 输入顺序每次递归合并 m 个内存连续顶点形成 fusion 顶点（约 log_m N 层直到单根），fusion 顶点持有其全部子顶点的聚合特征，作为"高阶代理顶点"保留远程/全局上下文；(3) association edges——指向目标顶点的逐层递归边：每层从目标顶点左右两侧各取 m 个关联顶点（起始下标 p_{l+1}=parent(p_l+m)，若 p_l+m-1 为奇数则再纳入下一个顶点且 p_{l+1}=parent(p_l+m+1) 保证集合互斥），覆盖多粒度远程上下文。TDS 总顶点数约 N + N/(m-1)。构造保证任意两个原始顶点通过 fusion+association 边最多 2-hop 可达，使每个目标顶点的 1-hop 注意力邻域同时含局部邻居、多粒度上下文与全局根顶点，一次稀疏注意力即达到多跳 message passing 的全局效果。
  - 实验比较：(1) 软件层——TAGT-S（用 TDS 方法修改 DGL v2.4.0 的软件实现，跑在 A100 GPU）对比 DGL-CPU（保留 O(N²) 全局注意力的 CPU baseline）与 TorchGT（SOTA GT 训练框架）：TAGT-S 比 TorchGT 快 1.8×–2.5×（TorchGT 依赖 Hamiltonian path 前提，现实图不满足时回退 O(N²) 全局注意力），不同序列长度下带宽利用率 >60% vs TorchGT；(2) 准确率（Table VI，序列长度固定 16K、100 次独立运行平均）——TAGT vs DGL-CPU 全注意力参考：全部数据集/模型上准确率下降 <1pp（GT 0.11–0.91pp、Graphormer 0.03–0.55pp、UGformer 0.22–0.84pp、EGformer 0.08–0.88pp），且 TAGT 准确率高于 TorchGT；(3) 训练效率——TAGT 相对全局注意力 baseline 有 >3× 训练与收敛加速（Fig.4）。
- 硬件平台是什么，配置是什么。
  - TAGT-S 软件实现跑在 NVIDIA Tesla A100 GPU（6,912 cores、80GB HBM）；DGL-CPU baseline 跑在 32 核 Intel Xeon Platinum 8357B @2.6GHz、503GB DDR4 RAM、16 内存通道。TAGT 硬件实现：Xilinx Alveo U280 FPGA（见 硬件架构 层条目）。
- 模型是什么。数据集和bench分别是什么。
  - 模型（Table III，按原论文超参）：Graph Transformer (GT) [Dwivedi & Bresson, 2020]（4 层、hidden 128、12 head）、Graphormer [Ying et al., 2021]（4 层、hidden 768、8 head）、UGformer [Nguyen et al., 2022]（4 层、hidden 384、4 head）、Edge Transformer / EGformer [Bergen et al., 2021]（8 层、hidden 200、4 head）。数据集（Table II）：Yelp (YP)（716,847 顶点、13,954,819 边、300 维特征、100 类分类）、Reddit (RD)（232,965 顶点、114,615,892 边、602 维、41 类分类）、Ogbn-Arxiv (OA)（169,343 顶点、1,166,243 边、128 维、40 类分类）、Ogbn-Products (OP)（2,449,029 顶点、61,859,140 边、100 维、47 类分类）、Ogbn-Papers100M (PM)（111,059,956 顶点、1,615,685,872 边、128 维、172 类分类）。任务均为节点分类。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未给出 TAGT/TAGT-S 开源链接；联网搜索（2026-08）未找到官方仓库，无法确认是否开源。baseline 开源：DGL（https://github.com/dmlc/dgl，v2.4.0）、TorchGT（https://github.com/hengruizhang98/torchgt）、Graphormer（https://github.com/microsoft/Graphormer）。
  - TDS 构造伪代码（m=2，沿 1D 输入顺序）：
    ```
    # 输入：顶点特征 x[0..N-1]（1D 内存顺序），合并基数 m，原始边集 E_orig
    # 阶段1：自底向上分层聚合——生成 fusion 顶点及其特征
    cur = list(range(N))                      # 基层：原始顶点为 leaves
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), m):       # 每 m 个内存连续顶点合并一次
            fus = 新建 fusion 顶点
            feature(fus) = aggregate(feature(c) for c in cur[i:i+m])   # 如 mean/sum 聚合
            for c in cur[i:i+m]: 添加 fusion 边 c -> fus               # 底向上有向边
            nxt.append(fus)
        cur = nxt                             # 上一层（约 log_m N 层到根）
    # 阶段2：为目标顶点构造 association 边（逐层左右各取 m 个关联顶点）
    for 目标顶点 v_k (k 为 1D 下标):
        for 右侧（左侧对称，用 k-1 递减）:
            p = k + 1
            for l in range(log_m N):
                在层 l 取下标 p..p+m-1 的 m 个顶点，添加 association 边 v_k -> 它们
                if (p + m - 1) 为奇数:  p = parent(p + m + 1)   # 互斥机制
                else:                   p = parent(p + m)
    # 阶段3：目标顶点注意力——只在 TDS 1-hop 邻域（original + fusion + association 边）上做
    ```
  - 张量计算示例（一个目标顶点 v 的一层更新，式1）：H^v = concat({h_u^l | u ∈ N_TDS(v)})，其中 N_TDS(v) 为 TDS 上 v 的 1-hop 邻域（K = O(m·log_m N) 个顶点，远小于 N）；注意力输出 h̄_v^{l+1} = softmax( h_v^l·W_Q · (H^v·W_K)^T / √d_K ) · (H^v·W_V)；随后 h_v^{l+1} = FFN(h̄_v^{l+1}) + h̄_v^{l+1}（残差）。Graphormer 类结构编码可融入初始嵌入（h_v^{(0)} = x_v + z^-_{deg^-(v)} + z^+_{deg^+(v)}）与注意力偏置（+ bias_{φ(v,u)}，φ 为最短路径距离）。总注意力边数 O(m·N·log_m N)，m=2 时即 O(N log N)。理论误差界（式2-4）：‖Δh_i‖₂ ≤ L‖V‖₂·Σ_{j∉T_i(m)} α_ij + ε_fus(m)，在注意力量重尾衰减 α_{i,(k)} ≤ c·k^{-β}（β>1）假设下 ≤ O((m·log_m N)^{1-β}) + ε_fus(m)——误差由结构截断与 fusion 粗粒度双因素决定，m 控制保真度-效率权衡，m=N 时退化为精确 O(N²) 全局注意力，m=2 为准确率最优（Fig.15d）。

## TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

- 属于算法pipeline的实现是什么？实验比较什么？
  - TUSQ 是为 noisy quantum circuit simulation (QCS) 提出的新算法框架，针对"时间密集+内存密集"（time-critical and memory-critical）的多状态向量模拟（SVS ensemble）场景（DMS 内存 O(2^2n) 不可扩展；naive SVS 每个 shot 都要重做完整矩阵向量乘）。实现含两大模块：(1) Error Characterization Module (ECM)——三步消除冗余/不显著电路实例：ER Tallying（对噪声通道预采样得到 Error Realization (ER)，统计唯一 ER 及频率，同一 ER 电路只算一次 statevector、按其频率重复采样输出）、ER Commutation（用 per-qubit 栈 + 6 条 Pauli 门穿通规则把噪声门尽量右移，识别输出等价的不同 ER 并合并 shot 计数）、Pruning（按频率 p_max 与阈值 α=0.01 划分 significant/insignificant 电路，对 insignificant 集合随机采样 β=100 个代表电路并加权采样以保持集体贡献）；(2) Depth First Tree Traversal (DFTT)——把待模拟电路集合按共享前缀组织成树，用 compute（正向乘 U）与 uncompute（反向乘 U†）做深度优先遍历复用重叠计算，操作数从 O(|E|log_b|E|) 降到 O(|E|)；对 non-invertible channel（mid-circuit measurement、erasure 等）用 DFTT+Caching（LIFO 缓存 pre-MCM 状态，容量 K 受内存约束，K=3 即可恢复 60%-100% 性能）。实验比较：TUSQ vs Qiskit 2.1.0、CUDA-Q 0.11.0、TQSim（同为 noisy SVS 加速器，BFS+memoization 方案），及 TNS 场景 vs CUDA-Q tensornet-mps；指标为相对加速比 γ 与 relative fidelity difference δ。198 个 benchmark × 1M shots：平均/最大加速 59.06×/7878.03×（vs Qiskit）、13.38×/439.38×（vs CUDA-Q）；同 fidelity 误差下 vs TQSim 平均/最大 39.32×/3134.31×；TNS+TUSQ vs 未优化 TNS 平均 248.39×；CPU 预处理平均/最大 3.97/18.52 秒。
- 硬件平台是什么，配置是什么。
  - NERSC Perlmutter 超算节点：AMD EPYC 7763 CPU（64 核/128 线程）+ NVIDIA A100 (40 GB) GPU；默认单 GPU（CUDA_VISIBLE_DEVICES=0），多 GPU 实验到 33 qubit，TNS 实验到 40 qubit。DFTT+Caching 性能恢复分析用 Stim 内置 rotated surface code memory 电路（26/64/118 物理比特，d=3/5/7，p=10^-2/10^-3/10^-4，d 轮测量，1M shots/电路，仅统计操作数不实际遍历）。MSC 验证用 18-qubit d=3 电路。代表性结果：30-qubit Adder × 10^6 shots 在单 A100 上约 820 秒（CUDA-Q/Qiskit 同硬件 >10 小时）。
- 模型是什么。数据集和bench分别是什么。
  - 无模型训练；benchmarks 用 Supermarq 套件：QAOA（13-25 qubit，depth 82-770，130-1250 门）、Adder（4-28）、Bitcode（5-25）、Phasecode（5-25）、GHZ（14-28）、QFT（14-24）、BV（4-24），覆盖线性（GHZ/Bitcode/Phasecode）/并行（QAOA）结构、单峰（Adder/Bitcode/Phasecode）/双峰（GHZ）/尖峰（QAOA）/均匀（QFT）输出分布。shots 取 32k/100k/1M/10M（Wang et al. TQSim 用 32k，Patti et al. 用 10^9）。噪声模型：depolarizing、measurement error、amplitude/phase damping（Pauli twirled 化），p=1% 默认（部分实验 p=0.1%）。VQE 正确性验证：10/15-qubit Ising 与 Heisenberg Hamiltonian。TNS 实验：40-qubit QFT/Adder/QAOA(p=2)，bond dimension=16，100k shots，α=0.01、β=100。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文正文声明 "An open-source implementation of TUSQ can be found here"，指向 GitHub 仓库 https://github.com/tinaoberoi/TUSQ（描述为 "C++ state vector and tensor network simulator for quantum circuits"，MIT license）。截至 2026-08 检索，该仓库仅含占位 README（无源码/文档/运行示例），实际代码未公开，无法按开源文档给出运行命令，复现需按论文方法自实现。baseline 与依赖开源：Qiskit v2.1.0（github.com/Qiskit/qiskit-aer）、CUDA-Q v0.11.0（github.com/NVIDIA/cuda-quantum）、NVIDIA cuStateVec v1.12.0 / cuTensorNet v2.9.1（cuQuantum SDK，github.com/NVIDIA/cuQuantum）、Supermarq（github.com/PrincetonQuantum/Supermarq）、Stim（github.com/quantumlib/Stim，DFTT+Caching surface code 电路来源）、TQSim（论文[41]，作者分享 GPU 兼容代码）。
  - 背景张量计算：noiseless SVS 只需一次矩阵向量乘 |ψ'⟩ = U|ψ⟩（U 为 2^k×2^k 幺正门矩阵，|ψ⟩ 为 2^n 维复向量，内存 O(2^n)）；noisy QCS 把 depolarizing channel 展开为加权经典混合 ρ'=(1-p)ρ+(p/3)XρX+(p/3)YρY+(p/3)ZρZ，等效于对 S 个固定噪声门的电路实例各做一次 SVS 再平均，故需 S 次完整矩阵向量乘（S-fold 开销）。
  - ER Tallying 伪代码：
    ```
    # 输入：带噪声通道的电路 C，总 shots S
    tally = {}                                        # ER n 元组 -> 频次
    for shot in 1..S:
        er = tuple(sample(ch) for ch in C.channels)   # DEP 采 I(1-p)/X/Y/Z(p/3)，measurement 采 I/X
        tally[er] += 1
    for (er_i, s_i) in tally.items():
        c_i = 把 er_i 的固定噪声门并入无噪声电路
        |ψ_i⟩ = SVS(c_i)                              # 只算一次 statevector
        输出分布 += 从 |ψ_i⟩ 采样 s_i 次                # 采样比矩阵向量乘便宜得多
    ```
  - ER Commutation（per-qubit 栈，保持"噪声门尽量靠右"不变量；复杂度从 O(g1·g2) 降到近最优）：
    ```
    # 规则：1) 相邻噪声 Pauli 相乘合并；2) Pauli 穿过无噪声 Pauli；3) X/Y/Z 穿过 R_X/R_Y/R_Z；
    #       4) CNOT：X(control)→X(control)X(target)，X(target)→X(target)
    #       5) CNOT：Z(target)→Z(control)Z(target)，Z(control)→Z(control)
    #       6) CNOT：Y(control)→Y(control)X(target)，Y(target)→Z(control)Y(target)
    stacks = [栈() for q in qubits]
    for gate in transpiled_circuit:                   # 基为单比特门 + CNOT
        if gate 无噪声:
            检查 gate 作用 qubit 的栈顶：若栈顶噪声门按规则可穿通 → pop 并按规则 push 新噪声门；否则直接 push gate
        else:  # 噪声 Pauli 门
            与栈顶已有噪声门按规则1合并，否则入栈
    结束后：ER 相同（含穿通后等价）的 (c_i, s_i) 合并 shot 计数
    ```
  - Pruning：p_max = 最高频电路频率；电路 c_i 显著 iff p_i ≥ α·p_max（α=0.01，用户可调）；insignificant 集合 C_I 中随机采 β=100 个代表 {t_1..t_β}，每个代表按 (p_insig/Σp_t)·p_t 次采样；S_final = Σ_i 1[p_i ≥ α·p_max] + min(β, |C_I|)。示例：10-qubit QAOA 1M shots 中 significant 占 58%、insignificant 合计 42%，不能直接丢弃。
  - DFTT：树节点 = 中间 statevector，边 = 门；DFS 正向边乘 U、反向边乘 U†（uncompute 回滚），共享前缀边只算一次（图5：算完 S1 后回滚到公共节点 d 再走 S6 分支）；T_dftt = 2|E|，T_naive = N_l·h = O(|E|log_b|E|)（b=4 DEP 或 b=2 measurement，h=log_b((b-1)|E|+b)-1，N_l=b^h）；非幺正边（MCM/erasure）用 DFTT+Caching：pre-MCM 节点入 LIFO 缓存，回滚跨非幺正边时取缓存状态而非求逆，K 不足时对"离叶子最近的 K 个 pre-MCM 节点"的子树分别 DFTT+Caching。单/双比特门矩阵向量乘分别计 1/4 个操作。

## Taking Analytic Databases to the Bank

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文实现了一套面向 BLIMP（Bank-Level In-Memory Processor，DDR DRAM bank 内嵌通用 RISC-V 处理器的 PIM 架构）的 OLAP 数据库算法与数据结构协同设计，核心三项：(1) PIMDT（PIM Data Type）列式存储格式——把指定数据库列以"PIM 友好"布局常驻存储（整字落在单个 bank 内），避免查询时昂贵的 relayout，仅支持定长类型，可查询时按元素边界 chunk 化以适配 32MB bank 内存，用 SQL 列约束（如 `Bar bigint NOT NULL PIMDT(BLIMP)`）声明；(2) BLIMP 优化的 row-buffer 对齐桶式哈希表——桶大小适配 1KB row buffer、初始桶数为 2 的幂（便于 hash 索引快速换算 row buffer 地址）、冲突在桶内 slot 链 + 桶链保持空间局部性，使用轻量乘法哈希 `hindex = (3634946921*value + 2096170329) & (initial_buckets-1)`；(3) 三个 BLIMP 核心算子：select（Algorithm 1 FILTERTOBITVECTOR，位图输出）、hash semijoin/join（host 建哈希表→relayout 复制到各 bank→BLIMP 探测，Algorithm 2 SEMIJOINPROBE）、aggregate（低基数分组用数组直接索引、高基数用哈希表，聚合器为 slot payload，平均等需 host reduce 收尾）。实验比较：BLIMP-S（标量 RISC-V）与 BLIMP-V（向量 RISC-V-V）相对 host CPU 基线（多线程 AVX512 手调 C++ kernel）以及 DuckDB 的算子级/端到端性能。
  - 关键数字：select（10 亿值、8/16/32/64-bit、< 谓词、1% 选择性）位图输出 BLIMP-S/BLIMP-V 平均 2.0×/12.9× 加速（选择性对位图输出影响 <0.5%）；值输出 2.0×/4.2×（低列宽时 BLIMP-S 反降速，弱核算力占比高；值输出随选择性增加而退化）。semijoin（探测侧 10 亿 32-bit、建侧 100 万 32-bit）BLIMP-S/BLIMP-V 1.4×/2.1×；join（4-byte payload）2.1×/3.0×（低选择性 host 哈希表可入 cache 占优，高选择性 BLIMP 的 row buffer 切换惩罚小于 host cache miss）。SUM 聚合（10 亿 32-bit）非分组 2.1×/33.7×，分组（基数 10000）1.9×/2.1×。端到端 SSB SF100：BLIMP-S/BLIMP-V 相对手调 C++ 基线 1.4×/2.3×，相对 DuckDB 3.1×/5.8×；PIM 感知规划比隔离算子外推快 3.2×，比 CPU 启发式规划快平均 28%（最大 40% Q3.3）。
- 硬件平台是什么，配置是什么。
  - Host：2× Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz（20 物理核/40 线程）、L1 1280KiB、L2 20MiB、LLC 27.5MiB，SK Hynix 384GB（64GB×6）2933 DDR4。BLIMP DRAM：PIM 使能 16GB（8GB×2）2133 DDR4，2 channel、2 rank、8 chips/rank、16 banks/chip、共 512 BLIMP core、32MB/bank、1KB row buffer、tRP=tRCD=21ns、tRFC=640ns、tREFI=7.8us。BLIMP-S core：RISC-V RV64GC SSIO @ 200MHz，1KB 指令缓冲、1KB scratchpad、5×1KB R/W buffer。BLIMP-V core：RISC-V "V0.9" RV64GCV SSIO @ 200MHz，1KB 指令缓冲、1KB scratchpad、5×1KB R/W vector register（v1-5）、32×64b vALU。
- 模型是什么。数据集和bench分别是什么。
  - 无模型训练。benchmark 为 Star Schema Benchmark（SSB，O'Neil et al. 2009，基于 TPC-H 修改），SF=100（60GB 数据），含事实表 LINEORDER + 维度表 CUSTOMER/DATE/PART/SUPPLIER，4 个 query flight QF1-QF4（复杂度递增、内部分 3-4 条 query 且选择性递增）。PIMDT 指定的 LINEORDER 列：全部外键列（lo_orderdate、lo_partkey、lo_suppkey、lo_custkey，用于 PIM 内 join）+ 常用过滤列（lo_quantity、lo_discount，用于 PIM 内过滤）。算子级实验数据：select/aggregate 用 10 亿值均匀随机 [0,100) 列（8/16/32/64-bit）；join 用探测侧 10 亿 32-bit 值 + 建侧 100 万 32-bit 值（4-byte payload，选择性决定 hash 表大小与存活记录数，如 1% 选择性 = 约 1 万 build 记录 + 约 1000 万 probe 存活）。示例查询：SSB Q1.1 SF100 的 "Fizz<25 AND Buzz BETWEEN 1 AND 3" 双选择、SSB Q3.1 SF100 的三 join（C/S/D 选择性 20%/20%/86%，hash 表 6MB/400KB/11KB）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未声明本文 DBMS 实现（PIMDT/hash 表/算子/查询规划器）的开源发布，论文未明确说明。同组 BLIMP 框架开源：dovedevic/blimp（https://github.com/dovedevic/blimp，"A PIM instrumentation, compilation, execution, simulation, and evaluation repository for BLIMP-style architectures"，随 ISCA'22 "To PIM or not..." 发布，含 /benchmarks、/chronometry、/compilation、/relayout、/simulation，检索于 2026-08）。仿真依赖：riscvovpsim（Imperas RISC-V ISS，https://github.com/riscv-ovpsim/imperas-riscv-tests，论文 ref [2]）、DRAMSim2（cycle-accurate 内存模拟器，https://github.com/umd-memsys/DRAMSim2，论文 ref [66]）。SSB 生成器开源（如 https://github.com/electrum/ssb-dbgen）；DuckDB 开源（https://github.com/duckdb/duckdb）。
  - 算法/张量级执行例子——select 内核（Algorithm 1 FILTERTOBITVECTOR）：BLIMP 核以 1KB row buffer 粒度 FetchMem(r) 读入整行 1024 个 8-bit 元素（32-bit 为 256 个），对每个元素 v1 ← apply(predicate, v1)（BLIMP-S 串行、BLIMP-V 用 32×64b vALU 向量化），再 coalesce 按元素大小 w 与已处理元素数 eproc 把布尔结果打包进位图行 v2（每 8192 元素对应一个 bitvector 行 HitmapRow），v2 ← v1 ∨ v2 累积，eproc 达 8192 倍数时 FetchMem+∧+StoreMem 写回位图行并清零 v2；行末不足 8192 时用 vs ∧ v2 回写并 ZeroMaskRemainder 清零尾部。哈希表索引（张量式）：对值 value 计算 hindex = hash(value) & (initial_buckets-1)，因桶大小固定且 2 的幂，hindex 直接换算目标 row buffer 内存地址 BucketRow(hindex)，实现 hash→内存地址的 O(1) 映射；探测时遍历该桶 slot 链（BLIMP-S 串行、BLIMP-V 先向量化 hash 再串行探测），冲突则沿 BucketNext 取下一桶（新 row buffer 读）。聚合：非分组 SUM 每行读入后向量加进单累加器，输出极小；分组聚合低基数（1024 范围）用 group 值直接索引聚合器数组，高基数用上述哈希表（payload=聚合器）。这些算法以"数据并行/可向量化/无跨数据依赖"为 PIM 适性原则（Map-Reduce 类），relayout 开销（host 侧 29GBps 平均吞吐 vs 90GBps 峰值带宽）在端到端评估中显式计入。

## TimeGaps Channels: Exploiting CPU Halted Time for Fun and Profit

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是"TimeGaps"侧信道检测/测量算法 + 基于 DNN 的网站指纹分类 pipeline。TimeGaps 指程序执行期间 CPU 被挂起、不做任何计算的时段（时间戳计数器仍在推进），其两大根因：CPU P-state 频率切换（Intel 已文档化）与 iGPU 频率切换（本文首次发现）。算法核心：(1) TimeGaps 检测——用 rdtscp 循环反复读取时间戳，记录超过阈值（5000 cycles，用于忽略 cache/TLB miss）的相邻读数差即 timestamp jump，再用 PMC 计数器（CPU_CLK_UNHALTED.THREAD / CPU_CLK_UNHALTED.REF_TSC 是否推进）区分"halted"与"unhalted"（中断引起）jump；(2) 非特权检测——native 环境用 SegScope 技巧（进入中断返回用户态时段寄存器 GS 被清零，故检测到 jump 时若 GS 仍为预设非零值则判定为 TimeGap 而非中断）；浏览器环境用毫秒级 JS 空循环计数（loop counter）探测 TimeGap 造成的吞吐下降；(3) 网站指纹分类——按 500μs 间隔记录 TimeGap 总时长/loop counter/CPU 频率作为特征，用 32 单元 LSTM（与 prior work [8,44,57] 相同）做 top-1 分类。实验比较：TimeGaps vs CPU 频率数据（DF-SCA [10]，读 scaling_cur_freq）vs loop counter 数据，在固定频率与默认 DVFS 两种场景下的指纹准确率；与 Hertzbleed 对比数据相关泄露（HW/HD 区分能力）；像素窃取对比 Hot Pixels [47] 与 Wang et al. [50]；击键检测对比 Rauscher et al. [39]。
  - 关键数字：网站指纹 native TimeGaps 固定频率 Chrome 92.2±0.7%/Tor 87.4±0.9%，默认 DVFS Chrome 98.0±0.9%/Tor 85.2±1.0%（vs 频率数据默认 DVFS 93.3%/63.0%，固定频率仅 ~1%）；0.1 秒窗口 TimeGaps 已可达 Top-5 57.1±3.5%（vs counter 31.8%、频率 18.2%）；SIKE-751 密钥提取只需判断首位为 0/1 把搜索空间降到 2；像素窃取 4.38 s/pixel、错误率 1.78%（Hot Pixels 8.1–22.6 s/pixel，Wang et al. 0.86–3.07 s/pixel）；击键检测 precision 84.6%/recall 99.6%/F1 91.5%（vs [39] 97.6/98.3/98.2）。
- 硬件平台是什么，配置是什么。
  - 4 台 Intel 真机（无模拟器）：Core i5-8259U（Coffee Lake，Iris Plus 655 iGPU，移动）、Core i7-9750H（Coffee Lake Refresh，UHD 630 iGPU，移动）、Core i3-10100（Comet Lake，UHD 630 iGPU + NVIDIA GTX 1080 Ti dGPU，桌面）、Core i9-10940X（Cascade Lake-X，AMD Radeon HD 7450，桌面）；网站指纹在 Core i7-7700（Kaby Lake）上。系统配置三种：default（默认 DVFS）、freq-control（cpufreq-set 固定频率）、isol-cpu（isolcpus 隔离核 + tickless 消除定时中断）。Linux（Ubuntu 20.04/22.04，kernel ≥5.4），root 权限用于 MSR/cpufreq/内核模块，攻击本身无特权假设。
- 模型是什么。数据集和bench分别是什么。
  - 分类模型：LSTM（32 units，与 [8,44,57] 相同超参），简化复现 E3 用 Random Forest。数据集：Alexa Top 150 中前 100 个活跃非成人网站（自带 sites/closed_world.csv），Chrome 每站 100 条 trace × 15 秒、Tor 每站 30 秒；10 折交叉验证（81% 训练/9% 验证/10% 测试）。密码学目标：Cloudflare CIRCL 库的 SIKE-751（378-bit 密钥 m，Montgomery ladder 相邻位不同产生零值 stall，属 post-quantum KEM 侧信道 benchmark）。无外部预训练数据集，时序数据实时采集。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：是，Zenodo 归档 https://doi.org/10.5281/zenodo.19450827，MIT license；artifact 含 C 采集器（gaps_collector_pmc/gaps_collector_5ms）、Python 自动化 attacker.py、OpenCL workload 生成器（改编自 bigger-fish [Cook et al., ISCA 2022]）；依赖 build-essential/linux-headers/msr-tools/cpufrequtils/stress-ng/opencl-headers/ocl-icd-opencl-dev/intel-opencl-icd + python3(numpy/pandas/matplotlib/scikit-learn/selenium/tqdm/pyopencl) + Chrome/ChromeDriver。
  - TimeGaps 采集伪代码（Listing 1）：
    ```
    index = 0; prev = rdtscp();
    loop {
      current = rdtscp();
      if (current - prev > threshold) {   # threshold = 5000 cycles
        jumps[index] = current - prev;
        index++;
      }
      prev = current;
    }
    ```
    特权判定：同迭代读 CPU_CLK_UNHALTED.THREAD/REF_TSC，若延迟期间 unhalted 计数不推进 → halted jump（TimeGap）；非特权判定：rdtscp 前置 GS=1，jump 后 GS 仍为 1 → TimeGap。
  - 指纹分类 pipeline 张量流：trace（500μs 采样，T 个时间步的 TimeGap 时长序列）→ LSTM(32 units) 逐时间步隐状态更新 h_t = LSTM(x_t, h_{t-1}) → 末步隐状态 → softmax 全连接（100 类）→ 10 折 CV 平均 top-1 准确率；TimeGap 时长本身与频率切换幅度相关（如 4.0→4.1GHz 约 35k cycles，4.4→4.3GHz 约 52k cycles），故时长序列携带频率切换信息。

## UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 UNICORE 的分布自适应量化算法框架（与 S-FPMA 硬件协同设计）：① DynFP——逐 group 可配置的低比特（4-bit/3-bit）浮点格式 DynFP(WE, WM, Z, I)，其中 WE/WM 为自适应指数-尾数分配（E3M0/E2M1/E1M2 等，优先动态范围或精度），Z 把冗余的负零码重映射为 E3M2 正常域内的有用值（更细分辨率中间值或更大 outlier 扩展动态范围，超出 group 格式最大值时符号吸收进 scale），I-flag 在指数码中间插入空 bit（gap-insertion，Φ(I,ℓ,E)=E_hi·2^(ℓ+1)+E_lo）以匹配非均匀分布；② 离线权重量化——贪心搜索框架：为每个张量枚举 96 个候选格式（4 个 E/M 布局 × 24 个 Z 赋值，Z≥0.5 避免 reintroduce subnormal），用迭代贪心（每轮选使全局 MSE 边际下降最大的格式）构造 k=16 的 palette，最后每 group 从 palette 选局部误差最小的格式，索引（4-bit）+ 8-bit scale 作元数据存储；无需激活校准、一次性离线执行（Llama-2-7B 单张 RTX 6000 Ada 约 2 分钟）；③ K/V 在线量化——用 crest factor κ（单趟 max-abs 归约 + RMS 归约，每元素仅 4 次标量运算）作轻量代理，经预计算阈值把 κ 映射到最优 E/M 布局（QSNR 行为）；④ 配合硬件侧双路径误差补偿（FG 细粒度 LUT 拼接 + CG 粗粒度进位）使 FP4 下 FPMA 结果与全精度乘法一致。量化方式为 PTQ direct-cast、无校准；8-bit 用 per-channel 权重 + per-token 激活量化，4/3-bit 用 group size 32 的 group 量化；激活用标准浮点格式 + INT8 scale；K/V 量化时 Q 与 softmax 输出 P 用与激活相同的 group size/位宽，仅 K、V 做在线格式选择。
  - 实验比较：WikiText-2 perplexity（Table I）与 zero-shot 准确率（Table II，ARC-e/HellaSwag/PIQA/Winogrande）上对比 INT、MXFP4、Tender、OliVe、M-ANT、BitMoD、AxCore（其原生 4/16/16 定宽）与 UNICORE/UNICORE-Q（UNICORE-Q 启用 DynFP 权重离线搜索 + K/V 在线格式选择），配置覆盖 4/4/16、4/8/16、3/8/16、4/4/4、8/8/16、16/16/16。结果：UNICORE-Q 在 4/4/16 各模型上 consistently 最低 PPL（如 OPT-6.7B 10.93 vs INT 11.18/BitMoD 11.08/M-ANT 11.14/MXFP4 11.26），接近 FP16（10.86）；4/8/16 与 3/8/16 稳定最优或近最优（3/8/16 下 INT 2158.19/BitMoD 190.12 崩坏而 UNICORE 14.60、UNICORE-Q 11.77）；4/4/4 下 UNICORE-Q 11.10 vs INT 11.33；zero-shot 8/8/16 下 UNICORE 平均准确率最高（Llama-3-8B 77.59 vs FP16 77.74），4/4/16 下 UNICORE 家族多数任务超越 INT/M-ANT/BitMoD，DynFP 进一步增益。消融：Table III 显示 FP4 下无 FG 补偿时 FPMA PPL 崩坏到 1.1E+4–4.9E+6，FG+CG 后与原始 FP4 完全一致（11.15）；Table IV 显示不同 group size（128/64/32）下 UNICORE PPL 均最优。
- 硬件平台是什么，配置是什么。
  - 精度评估在服务器上运行：4× NVIDIA RTX 6000 Ada（48GB）；软件栈 Ubuntu 22.04.5 LTS、Python 3.10.18、PyTorch 2.6.0、Conda 25.1.1、GCC 11.4.0、CUDA 12.4。加速器侧综合/模拟平台：SpinalHDL RTL → Synopsys Design Compiler TSMC 28nm @1GHz；扩展 BitMoD cycle-accurate 模拟器注入 post-synthesis timing/energy 模型；512KB activation + 512KB weight buffer（CACTI）；DRAMsim3 DDR4 25.6GB/s（prefill）/HBM2 256GB/s（decode）。在线激活量化开销测量（图 22）：Llama-2-7B 上激活量化占 prefill 时延 7.1%–20.7%、decode 仅 0.3%–1.6%（序列 512–8192），且与 GEMM 大部分重叠；UNICORE 的 crest factor 计算把量化 kernel 的 arithmetic intensity 从 0.63 提到 0.87（额外 reduction/sqrt/div），仍 memory-bound 无可见开销；对 L≥2K 序列，crest factor 计算占 QKᵀ FLOPs 不足 0.2%。
- 模型是什么。数据集和bench分别是什么。
  - 模型（Hugging Face 的 BF16 checkpoint 直接量化）：PPL 评估用 OPT-6.7B、Llama-2-7B、Llama-2-70B、Llama-3-8B、Qwen3-8B、Qwen3-14B；zero-shot 评估用 Llama-3-8B、Qwen3-8B。
  - 数据集/bench：WikiText-2（perplexity，ref [29]）；zero-shot 套件 ARC-e（AI2 Reasoning Challenge，ref [8]）、HellaSwag（ref [47]）、PIQA（ref [4]）、WinoGrande（ref [37]）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：是。GitHub https://github.com/CLab-HKUST-GZ/isca53-unicore（GPL-3.0），Zenodo 归档 https://doi.org/10.5281/zenodo.19449314。artifact 的 Software/Accuracy/ 含 ae_scripts（Table I-IV 复现脚本）、quant_utils、unicore_kernel；Conda 环境按说明创建后运行各表对应 shell 脚本，自动从 Hugging Face Hub 下载模型/数据集并执行 UNICORE 评估。模型与数据集（OPT、Llama-2/3、Qwen3、WikiText-2、ARC/HellaSwag/PIQA/WinoGrande）均来自 Hugging Face。
  - 算法/张量级执行例子——DynFP 量化与 FPMA 前向：① 权重 group（如 32 元素向量 w∈R³²）经贪心搜索选出格式 f=(E1M2, Z, I)，得到 E1M2 编码 + 8-bit group scale s + 4-bit 格式索引 idx；元数据随权重存主存，有效位宽 4.375 bits（4-bit 权重 + 1/8-bit 每 group 索引，仅比 MXFP4 高 2.9%）；② 前向时 Unified Format Converter 用 idx 查 LUT 把每个 DynFP4 编码解码/归一化为内部 E3M2 正常数：普通值 1→(−1)^S·2^(Φ−B)·(1+M)；subnormal（E=0,M≠0）→(−1)^S·2^(1−B)·M 经尾数左移归一化；负零（E=0,M=0,S=1）→ 由 mux 选择预定义 Z 值；I-flag 时指数按 Φ(I,ℓ,E)=E_hi·2^(ℓ+1)+E_lo 插入空 bit 扩展范围；③ FPMA 乘法——对归一化的 W、A 取 X=EW+MW、Y=EA+MA，乘积 R=X+Y−B（整数加法）；④ 补偿——FG 查表 C_fg(MA,MW) 拼接到 R 的 LSB 侧恢复低位，CG 1-bit 进位注入修正高位（FP4 下二者结合使结果与全精度乘法逐位一致）；⑤ 累加后经 group scale s 反量化（Rescale），K/V 场景在量化 kernel 内先算 κ=max_abs 与 RMS（各一次归约）再按阈值映射选 E/M 布局，对 L≥2K 序列开销 <0.2% QKᵀ FLOPs 且与 GEMM 重叠。核心张量运算即 FPMA 化 GEMM：C≈Σ(X_W+X_A−B) 的整数加法累加 + scale 反量化，把低比特 LLM GEMM 变成加法主导的线性可扩展计算。
