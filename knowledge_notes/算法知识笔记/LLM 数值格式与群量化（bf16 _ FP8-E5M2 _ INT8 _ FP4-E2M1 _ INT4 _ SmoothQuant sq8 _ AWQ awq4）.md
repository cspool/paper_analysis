## LLM 数值格式与群量化（bf16 / FP8-E5M2 / INT8 / FP4-E2M1 / INT4 / SmoothQuant sq8 / AWQ awq4）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 部署的权重数值格式谱系：bfloat16（8-bit 指数 + 7-bit 尾数，动态范围同 FP32）、FP8-E5M2（5 指数 2 尾数，Hopper 原生支持）、INT8、FP4-E2M1 与 INT4（4-bit 极低位宽）。群量化（group-wise quantization）：按组（如 g=128）共享 scale（与 zero-point）的权重量化；代表：AWQ（activation-aware weight quantization，per-channel 缩放保护 salient 权重、W4A16、MLSys'24 Best Paper）与 SmoothQuant（per-channel 平滑因子把激活离群难度转移到权重、W8A8；SmoothQuant+ 扩展 W4A16）。sq8/awq4 即 SmoothQuant 8-bit 与 AWQ 4-bit 格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文把这些格式作为无损压缩的信源符号表：符号即数值格式的离散码（FP8 码、INT4 值、per-channel 量化索引等）。符号表大小决定 ANS 行为：int8/sq8/fp4/int4 小符号表 + 尖锐分布 → ANS 与熵界差 0.01–0.05 bits；bf16 2¹⁶ 符号表 → 2¹² 精度 ANS 表定标偏差 0.1–0.2 bits（仍是现有无损方案里最接近熵界者）。群量化的 per-group scale 是结构化元数据，本身保留 1.1–1.3× 冗余，说明"量化之后仍可无损压缩"。正交性：先量化（有损）→ 再 rANS（无损），解压后与量化模型 bit-exact，不改变任何权重值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AWQ 实现：per-channel scale 网格搜索（α ∈ [0,1]）→ 权重×s / 激活÷s → INT4-g128 量化；SmoothQuant：s = max(|x|)^α / max(|w|)^(1−α)；部署格式 INT4 走 Marlin kernel（Ampere+）、FP8 W8A8 走 Hopper WGMMA。使用：模型压缩部署（TensorRT-LLM/vLLM/SGLang 均集成 AWQ/GPTQ）；本论文的用法是把这些格式作为"熵仍有冗余"的证据与压缩对象，证明无损压缩与量化正交、可叠加（int4 还能再压 4–6×）。

FlexQ-NDP 补充视角（ISCA'26）：该论文把群量化推进到"分组低比特浮点"形态——QGroup 粒度远细于 AWQ/SmoothQuant 的 g=128，如 (G_N,G_K)∈{(1,16),(1,32),(16,16),(32,32),(64,64),(128,128)}，每 16 个 FP4 元素配一个高精度 scale（FP8）。格式族为 microscaling：MXFP4/6/8（OCP 标准，32 个低比特元素共享一个 E8M0 2 的幂 scale，真实值 v_i = X·P_i）与 NVFP4（NVIDIA Blackwell 专有，16 元素块 + FP8 E4M3 scale + 每张量 FP32 二级 scale），以及 DeepSeek-FP8（FP8 块量化，块 scale 128×128）。反量化公式：partial^g = Σ_i (x_i^g · s_A^g · s_W^g)——每个 QGroup 的低比特乘加部分和须乘激活 scale 与权重 scale 恢复到高精度再累加；NVFP4 矩阵乘每处理 16 个 FP4 元素就要做一次双 scale 乘法，dequant 约占 NDP 上总执行延迟 35%。weight-only 量化（W4A16S8）不量化激活、计算前把权重反量化到激活精度，反量化开销与权重数据量相关；weight-activation 量化（W4A4S8）权重激活都量化、运行时需动态量化激活，反量化开销只与计算 bundle（点积次数）相关。该论文把量化配置抽象为 QConfig = {组尺寸, value 精度, scale 精度, W-only/W-A}，作为 NDP 编译策略选择（算子划分/缓冲分配/循环序）的输入变量——不同 QConfig 偏好不同编译策略，性能差距可达 70%。

Cassandra 补充视角（ISCA'26）：BF16 的 8-bit 指数占位宽 50%——在低 batch decode（memory-bound）下指数是剪枝/截断之后剩余的主要压缩率与带宽瓶颈，故 Cassandra 对指数单独压缩：Cassandra-1 用 unary 编码无损压缩（BF16 训练权重/KV 指数 Shannon 熵约 2.6/2.7 bits，实测平均约 2.85 bits）；Cassandra-2 用 MX 格式共享指数（每 32 元素块共享一个 E8M0 2 的幂 scale，OCP MX v1.0，v_i = X·P_i；草稿阶段即用 MXINT）。SmoothQuant W8A8 在该文语境：vLLM 官方 INT8 实现作 GPU baseline——低 batch decode 仅约 1.3×（在线激活量化 + scale 乘加开销在 GEMM 非瓶颈时无法隐藏，与文献报道的 INT8/FP8 decode 1.25–1.42× 一致）；且在推理 benchmark 上精度下降（Deepseek-R1-Distillated-Llama3-8B 上 SmoothQuant GPQA 46.0 vs BF16 49.0、AIME 23.3 vs 26.7）。该文结论：量化（有损）与 Cassandra（无损投机）正交且可组合——MXINT8 可直接融入 Cassandra 的草稿/验证格式（MXINT8 开销与其他 8-bit 量化相当且精度优于 MXFP8）。

PLENA 补充视角（ISCA'26）：把 MX 从"固定标准格式"扩展为可配置单级缩放格式——参数化 (M,E,S,B)（minifloat 元素）与 (M,S,B)（integer 元素），块内元素共享一个 E8M0 幂次 scale（8-bit 纯指数无符号位，范围 2⁻¹²⁷–2¹²⁷），PLENA 硬件原生支持 MXFP/MXINT 与可调块大小（BLEN∈[2,64] 纳入 DSE），权重默认 MXINT4、激活可选 MXINT/MXFP、KV cache 低精度 MX。关键经验：(1) MXFP 与 GPTQ/QuaRot 类 PTQ 不兼容——MXFP4 直接套用 4/4/4 量化在 LLaMA-3-8B 上 PPL 256.22（vs MXINT4+PLENA 方法 7.22），最小浮点格式表示区间窄，误差传播按整数格式设计时失效；(2) MXINT 是权重量化 de-facto 选择，但需搭配块级裁剪——裁剪参数 p∈[0.5,0.99] 收缩有效范围 [p·min_w, p·max_w]，配输出范数引导的逐行搜索 P_b*=argmin‖X_b(W_b−Q(W_b;P_b,τ))^T‖²（内层）与 GPTQ 式 Hessian 外循环（H_F=2X_FX_F^T）；(3) 硬件侧：PE 原生消费 MX 输入（元素与 E8M0 scale 分流）、INT 累加、写回前转激活精度；HBM 中数据块与 scale 分离存储以对齐内存边界（块+scale 拼接很少对齐 2 的幂边界）。与 NVFP4（16 元素块+FP8 E4M3 scale+张量级二级 scale）和 MicroScopiQ（两级缩放）相比，PLENA 选单级缩放以平衡硬件复杂度与精度；向量/非线性算子量化到 MiniFloat E6M5 相对 FP16 省 25% 内存且 perplexity 无损。

Shining Light 补充视角（ISCA'26，SiPh 加速器）：该论文把 AWQ 用作 LLM 部署到硅光子加速器时的权重侧量化——Qwen2.5-7B-instruct 先做 fp16→int4 的激活感知权重量化得到 Qwen2.5-7B-instruct-AWQ（Wikitext-2 困惑度 6.79，[82]），再把激活从 fp16 进一步量化为 int4-int8（per-tensor/per-feature/per-block affine 量化）以适配光域 MAC 的多电平编码。关键发现：AWQ 只解决权重侧动态范围，激活侧的 outlier（尤其 per-tensor 粒度）在 int5 以下使困惑度从 6.79 剧增到 182~120 万；且 SiPh 加速器无法像数字加速器那样用 FP8/24-bit 高精度累加（[94][96]）弥补激活量化丢失的动态范围，因为 ADC 量化位不可复用——说明"低比特激活 + 光域模拟计算"的组合对 LLM 尚不可行，需算法/器件级进一步改进。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- Shining Light on Silicon Photonic DNN Accelerators

XtraMAC 补充视角（ISCA'26，混合精度 MAC 硬件需求侧）：该论文从硬件视角梳理 LLM 量化方案产生的 MAC 数据类型组合（Table I）——权重仅量化（AWQ/GPTQ/SpQR）：投影层/FFN 为 INT×FP+FP→FP、注意力层 FP×FP+FP→FP；权重-激活量化（SmoothQuant/Atom）：投影层 INT×INT+INT→INT、注意力层 FP×FP+FP→FP；原生 LLM（GPT-oss-20b/120b）：MoE 块 MXFP4、其余 BF16（MXFP4/BF16×FP+FP→FP）。据此定义两类硬件需求：混合精度 MAC（乘数 A/B 异构格式，如 INT4×BF16）与运行时数据类型切换（同一硬件随模型组件交替数据类型，如 Qwen-3-8B-AWQ decode 期 68% MAC 为投影层 INT4×BF16、注意力层保留 BF16×BF16）。部署 profile（Table VI，HuggingFace 2025-10 下载量）：Qwen-3-8B-AWQ（222,126）、Llama-3.1-8B-W8A8（27,536）、Qwen-3-8B-FP8（429,968）、Llama-3.1-8B-FP8（168,122）、GPT-oss-20B（4,633,438）——实际部署的量化格式以 INT4/FP4/FP8 为主，要求 GEMV 引擎原生支持混合精度与运行时切换。FP4=E2M1、FP8=E4M3（E5M2 亦支持）；GPT-oss-20b 的 UE8M0×BF16 乘通过偏移 BF16 指数实现、不单独算 MAC。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- Shining Light on Silicon Photonic DNN Accelerators
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA
