## Group-wise Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Group-wise Quantization（分组量化）是一种细粒度的量化参数共享策略。将权重矩阵沿输入维度（或输出维度）划分为多个大小为 group_size 的组，每组独立计算自己的量化参数（scale、zero_point 或 AFPQ 中的 scale_pos/scale_neg）。相比 per-tensor 量化（整个 tensor 共享一组参数），group-wise 量化能更好地适应权重在 tensor 内的局部分布变化；相比 per-channel 量化（每行/列一组参数），group-wise 量化在精度和存储开销之间提供了可调节的折中。group_size 越小，量化越精细但参数存储开销越大（参数数 = tensor_size / group_size）。AFPQ 论文的关键发现：当 group_size 较小时，权重组的非对称分布现象更加显著（超过 50% 的组不对称），这正是对称 FP 量化在小 group_size 下表现差的原因，也使得 AFPQ 的非对称双 scale 设计在小 group_size 下收益最大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA2-7B 的 4096×4096 Linear 层 weight 矩阵、group_size=128 为例，group-wise 量化流程：
```
W = [4096, 4096]  # FP16 权重矩阵
group_size = 128   # 每组 128 个元素
num_groups = 4096 * 4096 / 128 = 131072  # 总组数

# 对称 FP 量化
for g in range(num_groups):
    w_g = W[g*128 : (g+1)*128]  # 取出第 g 组
    w_max = max(w_g)
    scale = max(w_max, abs(min(w_g))) / (range/2)  # 每组一个 scale
    W_q[g*128 : (g+1)*128] = round(w_g / scale)
    scales.append(scale)  # 共 131072 个 FP16 scale

# 存储开销
# 权重数据: 4096*4096 * 4bit = 8.39 MB (packed 4-bit)
# scale 参数: 131072 * 16bit = 262 KB
# 开销比: 262KB / 8.39MB ≈ 3.1%
```
AFPQ 的非对称版本中每组存储两个 scale（scale_pos, scale_neg），但总参数存储量不变：131072 × 2 × 16bit = 524KB，与 INT-asym 的 scale+zero_point 完全相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Group-wise quantization 广泛用于现代 LLM 量化框架：(1) GPTQ/AutoGPTQ 中 group_size 默认 128；(2) AWQ 中 group_size 默认 128；(3) bitsandbytes 中 block_size 等效于 group_size。在 PyTorch 中实现 group-wise 量化通常使用 `tensor.reshape(-1, group_size)` 然后在 dim=1 上计算统计量。常见 group_size 选择：128（最常用，精度/开销平衡）、64（更高精度）、256（更低开销但精度可能下降）、-1 表示 per-tensor。AFPQ 论文评估了 group_size = -1 (per-tensor)、256、128、64 四种设置。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AffineQuant Affine Transformation Quantization for Large Language Models
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache （KV Cache 量化：对 key 沿 channel 维度分组 G=32，value 沿 token 维度分组 G=32）
- Scaling Law for Quantization-Aware Training
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training（权值差值量化 group_size=2048，梯度量化 group_size=128/512）
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

QA-LoRA 将 group-wise quantization 与 LoRA 低秩适配深度结合：对每列权重 W_{:,j} 划分为 L = D_in/g 组，每组 g 个元素独立量化（α_{l,j}, β_{l,j}），增加了量化自由度。同时 LoRA 适配器 A 的行维度从 D_in 缩减为 L（因输入 x 通过组内求和聚合降维），使 A 矩阵的行向量在量化组内共享，保证了合并后模型仍可表示为 INT 量化格式。group_size g 越小（L 越大），量化越精细但存储开销越大，QA-LoRA 中默认 g=32（常用 GPTQ 设置），在 LLM 上取得精度-开销平衡。

ParoQuant 采用 group_size=128 的 block-wise INT4 线性量化，每个 128-channel group 独立应用 scaled pairwise rotation（channel-wise scaling + K=8 independent Givens rotations），使组内动态范围被收窄且离群值被跨通道交互压制。group 级独立旋转自然兼容 block-wise 量化——每个 group 的变换参数（θ, α）和量化参数（s, z）独立优化和存储。

SliM-LLM 利用group-wise量化的结构化特性实现硬件友好的混合精度：将权重矩阵沿列方向按group_size=128分组，每个group分配独立的bit-width（1/2/3-bit），group内元素共享相同精度。因为精度在group边界对齐（而非element-wise），packed integer存储时无需额外padding——即使3-bit，128个元素也恰好占满整数类型的字节空间。这种结构化混合精度只需额外存储每个group的2-bit精度标记（aggregated into integers），避免了SpQR、PB-LLM等element-wise混合精度方法的bitmap开销。SBA算法通过排序group平均salience（s_i = mean(W_g²/[diag(H^in)]_g²)）来确定哪些group提升/降低精度。

SpQR 采用极端小 group size（β₁=8~16，远小于常规的128）以提升量化精度，并通过双层量化（Bilevel Quantization）克服小 group 带来的统计量存储开销问题：第一层 scale/zero 以 3-bit 量化、第二层 scales-of-scales 以 16-bit 存储。平均统计量开销仅 (b_s+b_z)/β₁ + 64/(β₁β₂) ≈ 0.5 bits/param。传统方法因存储开销限制使用较大 group size（128），SpQR 证明：在极小 group 下双层量化能获得优于大 group 16-bit 统计量的精度。

在 Scaling Law for QAT 中，量化粒度 G（group_size）是缩放定律的核心变量之一：δ_p ∝ (log₂(G))^{γ_G}，其中 γ_G 衡量量化误差对粒度的敏感度。实验覆盖 G ∈ {32, 64, 128, 256, per-token/channel}，排除 per-tensor（因 4-bit 下退化严重）。激活量化误差对 G 的敏感度（γ_G=0.9812）远大于权重

在 Scaling Law for QAT 中，量化粒度 G（group_size）是缩放定律的核心变量之一：δ_p ∝ (log₂(G))^{γ_G}，其中 γ_G 衡量量化误差对粒度的敏感度。实验覆盖 G ∈ {32, 64, 128, 256, per-token/channel}，排除 per-tensor（因 4-bit 下退化严重）。激活量化误差对 G 的敏感度（γ_G=0.9812）远大于权重量化（γ_G=0.3533），因为激活中的 outlier 在粗粒度下被强制共享 scale 导致严重 clipping。使用对数项 log₂(G) 确保 G=1（无量化）时 δ_p=0。

---
