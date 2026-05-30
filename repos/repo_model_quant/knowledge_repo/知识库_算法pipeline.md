# 知识库_算法pipeline

## Symmetric Linear (Integer) Quantization（对称线性整数量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Symmetric Linear (Integer) Quantization 是一种最基础的均匀量化方案。将浮点张量 $x$ 映射到 k-bit 整数 $x_{int} \in [-2^{k-1}+1, 2^{k-1}-1]$，公式为 $x_{int} = \text{round}(x/s \cdot (2^{k-1}-1))$，其中 scale factor $s = \max(|x|)$（即取张量元素绝对值的最大值）或 per-group 的局部最大值。反量化公式：$x_{deq} = x_{int} \cdot s / (2^{k-1}-1)$。该方案"对称"的含义是量化格点关于零对称（无 zero-point 偏移），INT4 映射到 $\{-7, -6, ..., +6, +7\}$。计算开销极低——仅需一次 max reduction 和一次 element-wise scale+round，延迟在 GPU 上通常为 memory-bound。但对称量化对非对称分布的数据（如 ReLU 激活后的正偏分布或带有 outlier 的梯度）精度较差，因为零值的对称中心约束导致有效量化范围利用不足。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 中使用对称线性量化对权值差值和梯度进行压缩：
```
# INT4 对称量化（group-wise，per-group scale s = max(|x|)）
def symmetric_quantize_int4(x, group_size):
    # x: FP32/BF16 tensor, shape [N]
    for g in range(0, len(x), group_size):
        group = x[g:g+group_size]
        s = max(abs(group))                  # per-group scale
        x_q = round(clip(group, -s, s) / s * 7)  # map to [-7, +7]
        # 通信时发送: packed 4-bit x_q + FP32 scale s
    return packed_x_q, scales

# 反量化
def symmetric_dequantize_int4(packed_x_q, scales, group_size):
    for g, s in enumerate(scales):
        x_deq = packed_x_q[g] * s / 7       # recover FP32
    return x_deq
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对称线性量化广泛用于通信压缩场景（如 QSGD、SDP4Bit），因为在分布式通信中 scale factor 的计算和传输开销小（per-group 仅额外传输 1 个 FP32/FP16 scale）。在 PyTorch 中可用 `torch.quantize_per_tensor`（per-tensor 对称）或手动 group-wise 实现。与反量化结合使用时常见优化：(a) 将 scale 融合到后续的 dequantize-后操作中（如 gradient reduce）；(b) 对于非常小的 group_size（如 32-128），scale factor 存储/传输开销可能不可忽略，需权衡精度和带宽。SDP4Bit 中对权值差值使用 group_size=2048（开销 ≈ 0.1%），对梯度使用 group_size=128（intra）和 512（inter）以更好处理梯度中的局部变动。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
- QSDP: Quantized Distributed Training of Large Models with Convergence Guarantees

---

## Quantization on Weight Differences (qWD)（权值差量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization on Weight Differences (qWD) 是 SDP4Bit 提出的权值通信压缩策略。核心思想：不直接量化当前权值 $w_t$，而是量化两次迭代间的权值差值 $\delta w_t = w_t - w_{t-1}$（即 optimizer update 导致的变化量）。在 ShardedDP 的每轮迭代中，optimizer 更新 main weights 后计算 $\delta w = w_{main} - w_{model}$，对该差值做 INT4 group-wise 对称量化后通过 AllGather 分发，接收方反量化后加到本地 model weights $w_{model}$ 上。其有效性基于：(1) 经验上，差值分布比权值本身更均匀且数值范围更小（Fig. 4 直方图），INT4 量化误差更小；(2) 理论上，差值量化的相对误差 $\frac{\|q(\delta w_t) - \delta w_t\|}{\|w_t\|}$ 小于直接量化权值的误差 $\frac{\|q(w_t) - w_t\|}{\|w_t\|}$；(3) 收敛分析上，差值量化兼容 biased compressor（如 top-k sparsifier）而直接权值量化与 biased compressor 组合会收敛失败（Counterexample 4.1 证明 ternary quantizer 直接量化权重使 SGD 卡在初始值）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 中 qWD 的完整计算流程（Algorithm 2, 结合 Megatron-LM）：
```
# 每轮 iteration:
# 前置状态: w_main[p] (FP32 sharded), w_model (BF16 full, Megatron-LM 维护)

# Step 1: Optimizer 更新 main weights
w_main[p] = AdamW(g_main[p], w_main[p])

# Step 2: 计算权值差值
d[p] = w_main[p] - w_model[p]     # BF16 精度

# Step 3: INT4 group-wise 对称量化差值
# group_size = 2048
for g in range(0, len(d[p]), 2048):
    group = d[p][g:g+2048]
    s_g = max(abs(group))
    d_q[g:g+2048] = round(clip(group, -s_g, s_g) / s_g * 7)

# Step 4: AllGather 量化差值（带宽仅为 BF16 的 1/4）
d_q_global = AllGather({d_q, scales})

# Step 5: 反量化并更新 model weights
for each received shard:
    d_deq[p'] = dequantize(d_q[p'], scales[p'])
    w_model[p'_offset : p'_offset + size] += d_deq[p']

# Step 6: Forward pass 使用更新后的 w_model
output = ForwardPass(w_model, input)
```
与直接权值量化 qW 对比：qW 中 Step 2-5 被替换为直接量化 $w_{main}$ → AllGather → 反量化 → 赋值 $w_{model}$ = deq($w_{main}$)。qWD 的关键区别在于利用 $w_{model}$ 的历史值作为"锚点"，仅传输变化部分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
qWD 的实现依赖训练框架维持完整 model weights 副本——Megatron-LM 的 Distributed Optimizer 模式下天然支持（因不释放 weights），而 ZeRO-3/FSDP 等释放 weights 的框架需要额外适配（需在 backward 前重新 all-gather weights，对 qWD 的实现更复杂）。实现要点：(a) 差值计算需在 optimizer step 后、all-gather 前完成；(b) BF16 精度下的差值计算需注意数值稳定性——如果 $w_{main}$ 是 FP32 需先 cast 到 BF16；(c) group_size=2048 是 SDP4Bit 的默认配置，经 ablation 验证可达全精度训练准确率。SDP4Bit 开源代码在 Megatron-LM 中通过 `--quantized-weights --weight-quantization-bits 4 --wq-group-size 2048` 启用。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

---

## Two-Level Gradient Quantization (TLq / TLq-HS)（两级梯度量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Level Gradient Quantization (TLq) 是 SDP4Bit 提出的梯度通信压缩策略，针对 ShardedDP 中梯度同步的两阶段通信模式设计。将梯度通信分为两级：(1) **Intra-node（8-bit）**：节点内 all-to-all 通信走 NVLink/NVSwitch 高带宽链路，使用 INT8 量化以保持高精度；(2) **Inter-node（4-bit）**：跨节点 all-to-all 通信走 InfiniBand/Slingshot 相对低带宽链路，使用 INT4 量化大幅压缩通信量。两级间的衔接：intra-node all-to-all 后的数据先做 local reduce（将来自同节点内其他 GPU 的数据归约），再量化到 INT4 进行 inter-node all-to-all。其增强版 TLq-HS 额外在量化前施加 Hadamard Transform（32×32 Walsh-Hadamard 矩阵），将梯度中的 outlier 信息分散到邻近元素，产生更平滑的分布，从而显著降低量化误差。相比于 ZeRO++ 的 Uniform Level quantization (ULq) — 两级均用 4-bit — TLq-HS 用较小的通信开销增加（intra-node 8-bit vs 4-bit）换取了大幅的精度提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 中 TLq-HS 的完整流程（Algorithm 3）：
```
# 输入: grad (FP32, shape [N])
# 输出: g_final (FP32, 归约后的梯度)

# Step 1: Hadamard Transform 平滑
g_hat = H_32 @ grad @ H_32.T     # H_32 ∈ {+1,-1}^{32×32}

# Step 2: INT8 量化 + Intra-node AlltoAll
qg8 = round(clip(g_hat, -s8, s8) / s8 * 127)  # INT8, group=512
list_qg8 = IntraAlltoAll(qg8)    # NVLink/NVSwitch

# Step 3: 反量化 + Local Reduce（省略 Hadamard 逆向 ∵ H·H=I）
g_local = sum([dequantize(x) * s8/127 for x in list_qg8])

# Step 4: Hadamard + INT4 量化 + Inter-node AlltoAll
g_hat2 = H_32 @ g_local @ H_32.T
qg4 = round(clip(g_hat2, -s4, s4) / s4 * 7)  # INT4, group=128
list_qg4 = InterAlltoAll(qg4)    # InfiniBand/Slingshot

# Step 5: 反量化 + Final Reduce + Inverse Hadamard
g_reduced = sum([dequantize(x) * s4/7 for x in list_qg4])
g_final = H_32 @ g_reduced @ H_32.T   # 最终逆变换
```

优化技巧（Section 3.3）：
1. 利用 $H \cdot H = I$ 省略 Step 3 中的 Hadamard 逆向
2. 利用 $\sum_i H g_i = H \sum_i g_i$ 将 Step 5 的 Hadamard 从 dequant 之后移到最终 reduction 之后，使每轮 transform 次数从 6 降至 2

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TLq-HS 的实现基于 Megatron-LM + NCCL 的 all-to-all 集体通信原语：(1) Intra-node all-to-all 通过 `ncclGroupStart/End` 组织 per-rank P2P send/recv；(2) Inter-node all-to-all 同理跨节点 P2P。Hadamard transform 被融合到量化/反量化 CUDA kernel 中（Fused Hadamard Kernel），要求 group_size 能被 H 矩阵大小整除（SDP4Bit 设 H=32×32, group_size=512 intra / 128 inter）。启用参数：`--quantized-gradients --gradient-quantization-bits-intra 8 --gq-group-size-intra 512 --gradient-quantization-bits-inter 4 --gq-group-size-inter 128 --hadamard-transform --gradient-alltoall-pipeline 4`。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
- ZeRO++: Extremely Efficient Collective Communication for Giant Model Training

---

## δ-approximate Compressor（δ-近似压缩器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
δ-approximate compressor 是分布式优化理论中一类压缩算子的形式化定义：一个算子 $\mathcal{C}: \mathbb{R}^d \to \mathbb{R}^d$ 是 δ-近似压缩器（$\delta \in [0,1]$），如果对任意 $v \in \mathbb{R}^d$ 满足 $\mathbb{E}\|\mathcal{C}(v)-v\|^2 \leq (1-\delta)\|v\|^2$。直观上，δ 越大压缩越精确（δ=1 为无损）。该定义比 unbiased κ-approximate compressor 更广泛（含 biased compressor），可涵盖 top-k sparsifier、top-k low-rank compressor、随机量化（含随机取整 biased variant）等。Remark 4.1 指出任意 κ-approximate unbiased compressor $\mathcal{U}$ 可转换为 $\frac{1}{1+\kappa}$-approximate biased compressor：$\mathcal{C}(v) = \frac{1}{1+\kappa}\mathcal{U}(v)$。SDP4Bit 的 Theorem 4.1 使用 δ-approximate compressor 分析 weight difference compression 的收敛性，放宽了 QSDP 对特定 quantizer 和 Polyak-Łojasiewicz condition 的限制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 使用 δ-approximate compressor 分析 weight diff compression 的收敛（Algorithm 4 中的 $\mathcal{C}_w$）：
```
# SDP4Bit SGD with gradient + weight diff compression
for t in range(T):
    # Gradient: unbiased compressor (κ-approximate)
    g_tilde = U_g(g_t)           # E[U_g(v)] = v
    # Weight difference: arbitrary δ-approximate compressor
    w_t = w_{t-1} - η * g_tilde
    delta_tilde = C_w(w_t - w_tilde_{t-1})  # C_w ∈ class of δ-approx compressors
    w_tilde_t = w_tilde_{t-1} + delta_tilde
```
关键：$\mathcal{C}_w$ 可以是 biased 的（δ < 1），因为 weight diff compression 的误差传播受权重差值 $\|\delta w\|$ 而非权值本身 $\|w\|$ 控制，使得误差累积的范围更小（proof via $e_t = w_t - \tilde{w}_t$ 的递推收缩界）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
δ-approximate compressor 概念主要用于理论收敛分析，实践中的实现通常是 symmetric INT quantizer（如 SDP4Bit 的 INT4 group-wise 量化）、top-k sparsifier、或 low-rank compressor。在分析时，需推导具体量化方案对应的 δ 值（如 INT4 对称量化下 $\delta \approx 1 - \frac{1}{12 \cdot 2^{2k}}$），然后将 δ 代入 Theorem 4.1 得到收敛界。SDP4Bit 的实证验证表明 qWD 配合 INT4 group-wise 量化（group_size=2048）实际表现为 δ 充分接近 1 使得训练准确率与全精度几乎无异。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

## Adaptive Stochastic Quantization (ASQ)（自适应随机量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Stochastic Quantization (ASQ) 是一种根据输入向量 X 的具体分布自适应选择量化值集合 Q（而非使用固定量化值）来最小化随机量化均方误差（MSE）的技术。形式化地，给定排序向量 X ∈ R^d 和量化值个数 s，ASQ 寻找 Q ⊆ X（即最优量化值必定是输入的某个子集 [Zhang et al., ZipML 2017]），|Q| = s，使得 MSE = Σ_{x∈X} (b_x - x)(x - a_x) 最小化，其中 a_x, b_x 是包围 x 的连续两个量化值。与分布无关（distribution-agnostic）方法不同（如 QSGD 仅使用向量范数确定量化值，NUQSGD 使用全局 min/max），ASQ 针对每个特定输入向量优化 Q，可显著降低量化误差。然而，ASQ 问题是 non-convex 的（即使 s=4 即 2-bit 量化也不凸），排除了梯度下降等常规方法。ZipML 首次提出用动态规划（DP）在多项式时间内求精确解，但时间 O(s·d²) 和空间 O(d²) 使其在大向量上（d > 10⁵）不可行。QUIVER 通过预处理 + Quadrangle Inequality + SMAWK 将复杂度降至 O(s·d) 时间和 O(s·d) 空间，使 ASQ 在大规模 ML 场景中实用化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ASQ 在分布式学习的 gradient compression pipeline 中的执行流程：

```
输入: 梯度向量 X ∈ R^d, 量化值个数 s

// === Step 1: 自适应选择量化值（QUIVER 精确解）===
1. 排序 X（若非已排序，O(d log d)；GPU 上可并行排序）
2. 预处理 (O(d)):
   β_j = Σ_{i=1}^j x_i        // 累积和
   γ_j = Σ_{i=1}^j x_i²       // 累积平方和
   // 使 C[k,j] = -x_j·x_k·(j-k) + (x_j+x_k)·(β_j-β_k) - (γ_j-γ_k) 可 O(1) 求值
3. DP + SMAWK (O(s·d)):
   MSE[2, j] = C[1, j]  ∀j  // 初始化
   for i = 3 to s:
       K[i,·] = SMAWK(隐式矩阵 A where A[k,j] = MSE[i-1,k] + C[k,j])
       MSE[i, j] = MSE[i-1, K[i,j]] + C[K[i,j], j]
4. 回溯 (O(s)):
   Q = {x_1, x_d}, j = d
   for i = s to 3:
       j = K[i, j]; Q = Q ∪ {x_j}
   // 输出最优量化值集合 Q ⊆ X, |Q|=s

// === Step 2: 随机量化（Stochastic Quantization）===
for each x ∈ X:
    找到 a_x = max{q∈Q | q≤x}, b_x = min{q∈Q | q≥x}
    以概率 p_a = (b_x-x)/(b_x-a_x) 输出 x̂ = a_x
    以概率 p_b = (x-a_x)/(b_x-a_x) 输出 x̂ = b_x
    // 性质: E[x̂] = x (无偏), Var[x̂] = (b_x-x)(x-a_x)
```

核心数学洞察：C[k,j] 满足 quadrangle inequality，使得 DP 矩阵 A[k,j] = MSE[i-1,k] + C[k,j] 成为 totally monotone matrix，从而可用 SMAWK 算法在 O(d) 时间内找到每列的行最小值索引，替代原生 O(d²) 的逐列枚举。近似变体 Apx. QUIVER 将候选量化值离散化为均匀网格上的 m 个点，使用直方图预处理实现 O(d + m·s) 复杂度，并提供严格近似保证：AQ_{X,2s-2} ≤ opt_{X,s} + d·(x_d-x_1)²/(4m²)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：QUIVER 的 C++ 参考实现开源在 https://github.com/ranbenbasat/QUIVER。核心依赖 SMAWK 算法（1986 年提出，已有多种开源实现如 Python recipe by David Eppstein）。预处理部分需要 O(d) 额外空间存储 β, γ 数组。API 用法：输入已排序向量和 s → 输出最优 Q → 对每个 x 按概率做随机量化。

典型使用场景：（1）分布式/联邦学习中梯度压缩——sender 端先排序 → QUIVER 求 Q → 随机量化每个梯度分量 → 发送 Q + 每分量的比特编码 → receiver 端解码并反量化。1M 维梯度向量可在 ~1 秒内完成最优 4-bit 量化。（2）数据集量化——QUIVER 可扩展为 weighted variant，通过 weight 参数支持经验分布的 ASQ 求解，仅比非加权版慢 10-20%。（3）模型后训练量化——ASQ 可用于 weight/activation/KV cache 的无偏量化，避免偏置方法在分布式场景中误差不随 n 衰减的问题。

局限性：QUIVER 非 GPU-friendly（依赖 SMAWK 的 sequential recursion），Apx. QUIVER 不需要排序（O(d+m·s) 对未排序输入同样有效）。精确解要求输入已排序，否则需额外 O(d log d) 排序开销。

涉及论文标题：
- Optimal and Approximate Adaptive Stochastic Quantization

## Stochastic Quantization (SQ)（随机量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stochastic Quantization (SQ) 是一种无偏随机舍入方法。给定实数值 x ∈ [a, b] 和两个量化值 a, b，SQ 以概率 p_a = (b-x)/(b-a) 输出 a，以概率 p_b = (x-a)/(b-a) 输出 b，使得 E[x̂] = x（无偏性）。方差 Var[x̂] = (b-x)(x-a)，在区间中点处最大（= (b-a)²/4），在端点处为零。与 Round-to-Nearest (RTN) 不同（确定性、有偏），SQ 的随机性保证了无偏性，这对分布式场景（多客户端量化后聚合）至关重要：当 n 个独立无偏估计量平均时，MSE 随 1/n 衰减；而有偏量化中误差可能不随 n 衰减（因偏差相关）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SQ 在单个值的随机量化流程：

```
输入: x ∈ R, 量化值 a ≤ x ≤ b

// 计算概率
p_a = (b - x) / (b - a)
p_b = (x - a) / (b - a)

// 随机舍入
sample u ~ Uniform(0, 1)
if u < p_a:
    x̂ = a
else:
    x̂ = b

// 性质验证
E[x̂] = a·p_a + b·p_b
      = a·(b-x)/(b-a) + b·(x-a)/(b-a)
      = (ab - ax + bx - ab) / (b-a)
      = x·(b-a)/(b-a) = x  ✓ 无偏

Var[x̂] = (x-a)²·p_a + (x-b)²·p_b
        = (x-a)²·(b-x)/(b-a) + (b-x)²·(x-a)/(b-a)
        = (b-x)(x-a)  ✓
```

向量级量化：对 X = (x_1,...,x_d)，使用量化值集合 Q = {q_1,...,q_s}，每个 x_i 被包围它的连续量化值对 [q_j, q_{j+1}] 独立随机量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SQ 是众多分布式学习通信压缩方案的基础构建块。QSGD [Alistarh et al., NeurIPS 2017] 使用全局范数确定量化值（非自适应），NUQSGD [Ramezani-Kebrya et al., JMLR 2021] 使用全局 min/max。ASQ 与这些方法的区别在于：ASQ 使用优化的 Q 集合对 SQ 做"自适应"增强——SQ 本身是"如何量化单个值"的机制，ASQ 是"如何选择最优 Q 集合"的问题。SQ 实现简单（仅需一随机数生成器），可与 sparsification、top-k、Huffman coding 等正交压缩技术叠加。

涉及论文标题：
- Optimal and Approximate Adaptive Stochastic Quantization

## vNMSE (Vector Normalized Mean Squared Error)（向量归一化均方误差）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
vNMSE 是 ASQ 研究中的标准评估指标，定义为 vNMSE = E[‖X - X̂‖₂²] / ‖X‖₂²，即量化后向量与原始向量之间的期望 MSE 除以原始向量的 ℓ₂ 范数平方。通过归一化，vNMSE 消除了向量维度和幅值/方差的影响，使不同 d、不同分布下的量化质量可比。vNMSE 越低表示量化精度越高。在 QUIVER 论文中，vNMSE 用于衡量随 s（量化值个数）、d（维度）和分布类型变化时的量化精度趋势。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
vNMSE 计算流程：

```
输入: 原始向量 X, 量化值集合 Q = {q₁,...,q_s}

// 对每个 x ∈ X 做随机量化得到 x̂
// 实际实验中用多次独立随机量化求平均（QUIVER 论文用 5 seeds）
E_sq_error = 0
for seed in 1..5:
    对每个 x ∈ X, 以概率 (q_{j+1}-x)/(q_{j+1}-q_j) 取 q_j
    计算 sq_error_seed = Σ (x̂_i - x_i)²
    E_sq_error += sq_error_seed / 5

vNMSE = E_sq_error / ‖X‖₂²
```

vNMSE 与 ASQ 的理论关联：
- 精确 ASQ 最小化 Σ (b_x-x)(x-a_x)，这正是 MSE 的期望值
- vNMSE 将该 MSE 归一化，便于不同尺度的向量比较
- Apx. QUIVER 的近似保证：vNMSE_{2s-2} ≤ vNMSE_opt_s + d/(2m²)
  （使用 2s-2 个量化值的近似解的 vNMSE 不超过 s 个量化值最优解的 vNMSE + d/(2m²) 的附加项）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
vNMSE 在梯度压缩和量化方法评估中广泛使用 [Vargaftik et al., NeurIPS 2021]。实现简单：对量化输出向量 X̂ 和原始 X 计算 ℓ₂ 距离平方后除以 ‖X‖₂²。典型 vNMSE 取值范围：优秀 1-bit 量化（s=2, d=10⁶, LogNormal 分布）在 0.1-0.3 量级；4-bit（s=16）可降至 10⁻³ 以下。该指标与下游任务性能（如分布式学习收敛速度）有强相关性。

涉及论文标题：
- Optimal and Approximate Adaptive Stochastic Quantization

## TTT (Test-Time Training) Layers（测试时训练层）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TTT（Test-Time Training）层是一种将 RNN 的隐藏状态定义为一个机器学习模型 f 的权重 W，将更新规则定义为对自监督损失 ℓ 的一步梯度下降的序列建模层。核心理念：自监督学习能将大规模训练集压缩进模型权重（如 LLM 将互联网知识压缩进参数），因此将同样的"压缩启发式"用于 RNN 隐藏状态的更新。具体地，对于输入序列 x_1,...,x_T，隐藏状态为 W_t（模型 f 的权重），更新规则为 W_t = W_{t-1} - η ∇ℓ(W_{t-1}; x_t)，输出规则为 z_t = f(θ_Q x_t; W_t)。由于即使在测试序列上也会执行此训练过程，因此称为"测试时训练"（Test-Time Training）层。TTT 层具有线性复杂度 O(T × d²)，与序列长度 T 无关的每 token 开销。与 self-attention（KV cache 线性增长，O(T²) 总复杂度）相比，在长上下文下具有渐近优势；与传统 RNN（Mamba）相比，通过梯度更新的自适应性避免了固定大小隐藏状态的表达瓶颈。TTT 层可即插即用替换 Transformer 中的 self-attention，或集成到 Mamba backbone 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TTT 层在语言模型中的 pipeline 流程（以 TTT-Linear 为例）：

```
# ===== 外循环（常规训练）=====
# 优化目标：next-token prediction loss
# 可训练参数：θ_rest（网络其余参数）, θ_K, θ_V, θ_Q, θ_init, θ_lr

# ===== 内循环（TTT，每个序列执行）=====
# 输入：序列 x_1,...,x_T ∈ R^d
# 超参：mini-batch size b, base learning rate η_base

W = θ_init  # 初始权重，形状 d×d（TTT-Linear）或 MLP 参数（TTT-MLP）

for each mini-batch of b tokens:
    X_block = [x_{t},...,x_{t+b-1}]  # ∈ R^{d×b}

    # Step 1: 多视角投影（learned views）
    X̂ = θ_K @ X_block   # training view ∈ R^{d'×b}
    Y  = θ_V @ X_block   # label view ∈ R^{d'×b}
    X̄ = θ_Q @ X_block   # test view ∈ R^{d'×b}

    # Step 2: 自监督损失（multi-view reconstruction）
    # ℓ(W; x_i) = ||f(x̂_i; W) - y_i||²

    # Step 3: 梯度下降更新（mini-batch GD）
    # W_new = W - η Σ_i ∇ℓ(W; x_i)
    # 通过 dual form 高效计算（避免显式外积）

    # Step 4: 输出 token
    # z_i = f(x̄_i; W_i)，其中 W_i 是处理 x_i 时的权重
    Z_block = compute_outputs(X̄, X̂, Y, W, η)

    W = W_new  # 隐藏状态传递到下一个 mini-batch

# 输出：z_1,...,z_T
```

关键设计特点：
1. **隐藏状态本身是模型**：W 的维度为 d×d（TTT-Linear），参数量远大于传统 RNN 的隐藏向量（d 维），因此具有更强的表达能力。
2. **更新规则是梯度下降**：与手工设计的门控机制（LSTM）或选择机制（Mamba）不同，梯度下降使更新天然具有数据自适应性——产生大梯度的 token 被"记住"更多。
3. **自监督任务可学习**：θ_K, θ_V, θ_Q 通过外循环学习，使内循环的 reconstruction 任务专门为外循环的 next-token prediction 目标优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TTT 层的两个主要实例化：
- **TTT-Linear**：f(x) = Wx，W ∈ R^{d×d}（方形矩阵）。与 linear attention 有理论等价性（Theorem 1：batch GD + W_0=0 + η=1/2 时等价）。实现最简单，计算效率最高。
- **TTT-MLP**：f(x) = x + LN(MLP(x))，MLP 为两层（hidden dim 4×, GELU 激活, LN + 残差连接）。表达能力更强，在长上下文下优势更大，但 wall-clock 开销更高。

实际使用中：
- 训练：基于 EasyLM (JAX) 框架，遵循 Chinchilla recipe（与 Transformer 相同的训练配置）。外循环使用 AdamW 优化器，内循环使用 SGD（mini-batch GD）。
- 推理：forward (prefill) 使用 dual form（matmul 并行），decode 使用 primal form（逐 token 序列化）。
- 代码开源：JAX 版本 https://github.com/test-time-training/ttt-lm-jax，PyTorch 版本 https://github.com/test-time-training/ttt-lm-pytorch。

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Mini-batch TTT（小批量测试时训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mini-batch TTT 是 TTT 层中用于并行化内循环梯度计算的技术。标准的 online gradient descent（每个 token 后立即更新 W）无法并行化，因为 W_t 依赖于 W_{t-1}。Mini-batch TTT 将输入序列分成大小为 b 的 mini-batch，在 mini-batch 内部共享起始权重 W_{t'}（其中 t' = t - mod(t, b) 为上一个 mini-batch 的结束时间步），从而可以在 mini-batch 内并行计算 b 个梯度。更新公式：G_t = ∇ℓ(W_{t'}; x_t)，W_t = W_{t-1} - η G_t。b=1 等价于 online GD，b=T 等价于 batch GD。论文选择 b=16 作为 quality-speed 的最优折中。从消融实验（Table 1）看，从 batch GD (b=T=2048) 切换到 mini-batch GD (b=16) 是将 linear attention 转变为 TTT-Linear 的最大单一改进（PPL 从 14.05 降至 12.35，-1.70）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mini-batch TTT 的数学描述：

```
设序列长度 T，mini-batch size b（假设 T 整除 b）

W_0 = θ_init

for m = 0, 1, ..., T/b - 1:       # m 为 mini-batch 索引
    t_start = m * b
    W_ref = W_{t_start}             # mini-batch 内共享的参考权重

    # parallel: 计算 b 个梯度（均对 W_ref 求导）
    for i = 1, ..., b:
        t = t_start + i
        G_t = ∇ℓ(W_ref; x_t)       # 对 W_ref 求导，而非 W_{t-1}

    # sequential: 逐 token 累积更新
    for i = 1, ..., b:
        t = t_start + i
        W_t = W_{t-1} - η · G_t    # 从 W_{t-1} 开始梯度步

# 信息传播的两个通道：
# 1. cumsum 通道（始终活跃）：W_t = W_0 - η Σ_{s=1}^t G_s
# 2. 梯度通道（仅在 mini-batch 边界活跃）：G_t 对 W_{t'} 求导
```

直观理解：b 控制搜索空间大小 vs. 并行度的权衡。较小的 b 意味着更多的梯度步（更大的有效搜索空间），但更少的并行机会；较大的 b 意味着更高的并行度但更少的梯度步。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实践中，mini-batch TTT 与 dual form 结合使用。对于每个 mini-batch：
1. 使用 dual form 一次性计算出 W_b（mini-batch 结束时的权重）和 z_1,...,z_b（mini-batch 内所有输出 token）
2. W_b 作为下一个 mini-batch 的参考权重 W_ref
3. 论文设定 b=16（所有实验），η_base=1.0 (TTT-Linear) 或 0.1 (TTT-MLP)

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Multi-view Reconstruction Loss for TTT（TTT 多视角重建损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-view Reconstruction Loss 是 TTT 层内循环使用的自监督损失函数。基本形式为 ℓ(W; x_t) = ||f(θ_K x_t; W) - θ_V x_t||²，其中 θ_K x_t 是 training view（被破坏/降维的输入），θ_V x_t 是 label view（重建目标），f 是隐藏状态模型。与传统的 denoising autoencoder 不同，TTT 中的 training view 和 label view 投影矩阵 θ_K 和 θ_V 不是手工设计的，而是通过外循环学习得到的可训练参数。这种设计使自监督任务能端到端地优化为最终的下一个 token 预测目标服务。类似于对比学习中的多视角（multi-view）框架：training view 提供部分信息，模型需要发现维度间的相关性来重建 label view。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multi-view reconstruction 的计算流程：

```
# 输入 token x_t ∈ R^d

# Step 1: 生成三个视角（learned projections）
train_view = θ_K @ x_t    # training view，通常低秩投影（降维）
label_view = θ_V @ x_t    # label view，重建目标
test_view  = θ_Q @ x_t    # test view，用于生成输出

# Step 2: 计算自监督损失
# 模型 f 从 training view 重建 label view
pred = f(train_view; W)   # f 的当前参数为 W
loss = ||pred - label_view||²  # MSE reconstruction loss

# Step 3: 梯度更新 W
# W_new = W - η · ∇_W loss

# Step 4: 输出 token（使用 test view）
output = f(test_view; W_new)
```

维度关系：training_view 和 label_view 的维度 d' 小于 x_t 的维度 d（低秩投影），test_view 通常保持 d' 维度。θ_K, θ_V, θ_Q ∈ R^{d'×d} 是外循环参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实际 TTT 层实现中：
- θ_K, θ_V, θ_Q 类似于 self-attention 中的 Key, Value, Query 投影矩阵，但语义不同
- 这些投影通过外循环的 next-token prediction loss 反向传播学习
- TTT-Linear 中 d' = d（全维度投影），TTT-MLP 中类似
- self-supervised loss 的选择（MSE reconstruction）是初步的；论文指出未来工作可能探索更复杂的 self-supervised 任务族

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Inner Loop / Outer Loop Bi-level Optimization in TTT（TTT 双层优化内外循环）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TTT 将序列建模重构为一个双层优化（bi-level optimization）问题。**内循环（inner loop）**对每个序列执行：以 token 为数据点，最小化 self-supervised reconstruction loss ℓ，通过梯度下降更新隐藏状态 W。**外循环（outer loop）**在数据集层面执行：以序列为数据点，最小化 next-token prediction loss，优化网络参数 θ_rest 和内循环的超参数 θ_K, θ_V, θ_Q, θ_init, θ_lr。关键区别：常规的 learning to learn / meta-learning 中外循环是"高一层"的训练（需要多个数据集/task），而 TTT 中内循环是"低一层"的训练（每个序列是一个"数据集"），外循环与常规监督学习处于同一层级。这种设计使得外循环可以使用标准的大规模训练 recipe（如 Chinchilla），无需额外的 meta-learning 数据组织。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
双层优化的层次结构（Table 2）：

```
层级          | 内循环            | 外循环
-------------|-------------------|------------------
数据单元      | token x_t         | 序列 x_1,...,x_T
训练集        | 序列 x_1,...,x_T  | 数据集（如 Pile, Books3）
优化目标      | reconstruction ℓ  | next-token prediction
优化参数      | W（f 的权重）     | θ_rest, θ_K, θ_V, θ_Q, θ_init, θ_lr
优化器        | SGD (mini-batch)  | AdamW
```

计算图流程：
1. 外循环 forward：执行内循环（对序列 TTT），得到输出序列 z_1,...,z_T
2. 外循环 backward：计算 ∂L/∂θ_rest（通过内循环的梯度），更新外循环参数
3. 内循环涉及 "gradient of gradient"（二阶微分），因为外循环梯度需要穿过内循环的梯度步

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实践中：
- JAX/PyTorch 的自动微分原生支持 gradient-of-gradient，无需特殊处理
- 内循环使用 SGD（学习率 η 可学习），外循环使用 AdamW
- 梯度 checkpointing through time 用于节省内循环中间状态 W_1,...,W_T 的内存
- 内循环和外循环共享相同的 backbone 结构（Mamba backbone 或 Transformer backbone）

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Linear Attention（线性注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear attention 是 self-attention 的一种线性复杂度变体，由 Katharopoulos et al. (2020) 提出。核心思想是去除 self-attention 中的 softmax 非线性：z_t = Σ_{s=1}^t v_s k_s^T q_t = (Σ_{s=1}^t v_s k_s^T) q_t。由于去掉 softmax 后，计算可以重排为矩阵乘法的结合律形式，隐藏状态 M_t = Σ_{s=1}^t v_s k_s^T 可以通过 cumsum 递归更新，每个 token 的复杂度为 O(d²)（与 t 无关），总复杂度 O(T × d²)。与标准 self-attention 的 O(T² × d) 相比，在长序列下有渐近优势。TTT 论文证明（Theorem 1）：TTT-Linear + batch GD + W_0=0 + η=1/2 等价于 linear attention，即 TTT 框架可以严格推广 linear attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Linear attention 的递归形式和并行形式：

```
# 递归形式（类似 RNN，训练/推理均可用）
S_0 = 0  # d×d 矩阵
for t = 1,...,T:
    k_t = θ_K @ x_t   # key projection
    v_t = θ_V @ x_t   # value projection
    q_t = θ_Q @ x_t   # query projection
    S_t = S_{t-1} + v_t @ k_t^T    # 更新隐藏状态（外积）
    z_t = S_t @ q_t                 # 输出

# 并行形式（训练用，但也是 O(T²) 内存）
K = θ_K @ X, V = θ_V @ X, Q = θ_Q @ X
Z = V @ (K^T @ Q) * mask    # 等价于 attention without softmax
```

与 TTT 的关系：linear attention 的更新规则 `S_t = S_{t-1} + v_t k_t^T` 等价于 batch GD 的 TTT-Linear（G_t = -2 v_t k_t^T）。TTT 通过 mini-batch GD 打破了这一等价性，获得了更好的表达能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实践中：
- linear attention 因其简单的实现被广泛使用于需要线性复杂度的场景
- 其表达能力被证明弱于 softmax attention，因为缺少非线性归一化
- TTT 框架将 linear attention 作为特例包含，并通过 mini-batch TTT 和可学习组件显著提升了性能
- Mamba-2（Dao & Gu, 2024）也基于类似 linear attention 的矩阵状态设计

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Fast Weights / Fast Weight Programmers（快速权重/快速权重编程器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fast weights 是一种神经网络概念，最早由 Hinton & Plaut (1987) 和 Schmidhuber (1992) 提出。核心思想：神经网络的权重分为两类——"slow weights"（在全部数据上缓慢更新，即常规训练的参数）和"fast weights"（仅在最近/最相关数据上快速更新）。Fast Weight Programmers (FWPs) 是 Schmidhuber 提出的延伸概念：使用一个学习到的"slow"网络来生成"fast"网络的权重更新规则。TTT 层可被看作 FWP 的特例：内循环权重 W 是"fast weights"（在单个序列上更新），外循环参数 θ 是"slow weights"（在数据集上更新），更新规则是显式的梯度下降。TTT 框架的独特之处在于将 fast weights 的更新形式化为一个显式的最优化问题（自监督学习），而非使用手工设计的更新规则。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fast weights 在 TTT 中的具体体现：

```
# Slow weights (outer loop parameters, updated across datasets)
θ_K, θ_V, θ_Q    # reconstruction views
θ_init = W_0      # fast weight initialization
θ_lr              # learning rate for fast weight updates
θ_rest            # rest of the network

# Fast weights (inner loop state, updated per sequence)
W_0 = θ_init      # initialized from slow weight
for each token x_t in sequence:
    W_t = W_{t-1} - η(x_t) · ∇ℓ(W_{t-1}; x_t)   # fast update
    z_t = f(θ_Q x_t; W_t)                         # use fast weight
# After sequence ends, W_T is discarded; only θ_init is kept
```

与 DeltaNet 的关系：DeltaNet (Schlag et al., 2021) 等价于 TTT-Linear with b=1（online GD），但没有 LN 和残差连接。Gated DeltaNet (Yang et al., 2024) 添加了门控机制和数据依赖的衰减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Fast weights/FWP 概念已催生了多条现代 RNN 研究线：
- **Linear attention** (Katharopoulos et al., 2020)：最简单的 FWP 形式
- **DeltaNet** (Schlag et al., 2021)：delta rule 更新
- **Mamba-2** (Dao & Gu, 2024)：结构化状态空间对偶，使用矩阵隐藏状态
- **Gated DeltaNet** (Yang et al., 2024)：添加门控的 delta rule
- **TTT layers** (Sun et al., 2024)：通用框架，支持任意神经网络作为隐藏状态

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Ternary Adaptation (TA, 三元适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ternary Adaptation（TA，三元适配）是 LoTA-QAF 提出的一种面向量化 LLM 微调的适配器结构。与标准 LoRA 使用 FP16 可训练适配器不同，TA 将适配器权重约束为三值离散空间 {-1, 0, 1}：两个低秩矩阵 A_T ∈ {-1,0,1}^{D_in×r} 和 B_T ∈ {-1,0,1}^{r×D_out}（r ≪ min(D_in, D_out)）。A_T 通过 Kaiming normal 初始化后经阈值 0.75·mean(|A_T|) 三值化；B_T 初始化为零。乘积 ΔW = A_T B_T 形成辅助矩阵，元素为整数 ∈ [-r, r]。通过阈值 ω ∈ (0, r)，将 ΔW 映射为三元调整矩阵 Ŵ ∈ {-1,0,1}^{D_in×D_out}：Ŵ_ij = sign(ΔW_ij) · I_{|ΔW_ij|>ω}。由于 Ŵ 与量化权重 W_int 同属整数域，可直接相加 W'_int = W_int + Ŵ 实现量化网格内（in-grid）权重的直接调整，无需反量化-再量化。同时计算偏移因子 μ = mean(ΔW - ω·Ŵ)，用于更新零点因子 z' = z + s·μ。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LoTA-QAF 中 TA 的完整前向流程（以 4-bit 量化 linear 层为例）：
```
# 输入: x (激活), W_int (量化权重, shape D_in×D_out, 值域[0,15])
#      s, z (量化参数), A_T (D_in×r, 三值), B_T (r×D_out, 三值)
#      ω (阈值, 如 0.75r)

# 1. 计算辅助矩阵
ΔW = A_T @ B_T          # D_in×D_out, 元素 ∈ [-r, r]

# 2. 生成三元调整矩阵
Ŵ = zeros_like(ΔW)
Ŵ[ΔW > ω] = 1
Ŵ[ΔW < -ω] = -1

# 3. 应用边界检查后调整量化权重
W_int_new = W_int + Ŵ
W_int_new = clamp(W_int_new, 0, 2^N - 1)

# 4. 计算偏移因子
W̃ = ΔW - ω * Ŵ           # 残差
μ = mean(W̃)               # per-tensor / per-group / per-channel

# 5. 前向输出
z_new = z + s * μ
y = (s * W_int_new + z_new)^T @ x
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
由于 PyTorch 不支持原生 ternary/int2 数据类型，TA 使用 bfloat16 模拟三值 {-1,0,1}。辅助矩阵 ΔW 的形成和 Ŵ 映射通过 Triton 自定义 kernel 实现（kernel fusion 融合 Ŵ 生成和边界检查为单一 GPU kernel）。在 LoTA-QAF 开源代码（github.com/KingdalfGoodman/LoTA-QAF）中，TA 实现于 LoTA/layer.py 的 CustomLoraLinear 类。推理时，微调完成的适配器通过 lota_merge.py 合并到量化权重中，推理仅需标准量化 kernel（TritonV2QuantLinear/TorchQuantLinear），无额外适配器开销。超参数：rank r 通常 64（8B/14B）或 32（32B/70B），ω = 0.75r（通用）或 0.875r（ViGGO 等特殊任务）。

涉及论文标题：
- LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

## t-SignSGD (Ternary Signed Gradient Descent, 三元符号梯度下降)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
t-SignSGD 是 LoTA-QAF 为优化三值适配器 {-1,0,1} 而专门设计的优化器。受 SignSGD 在离散/约束域参数更新中表现优异的启发（Balles et al. 2020; AutoRound Cheng et al. 2023），t-SignSGD 使用符号梯度 + 动态百分位阈值选择性地更新三元适配器权重。核心更新规则：A_{T,t+1} = clip(A_{T,t} - sign(g_t) · I_{|g_t|>max(τ,σ_t)}, -1, 1)，其中 g_t = ∇_{A_T} L 为当前梯度，τ 为固定最小梯度阈值（如 1e-9），σ_t 为动态百分位阈值（基于梯度幅值分布，初始 top-5%，线性衰减至 0.01%）。关键设计：(1) 无学习率——选中更新的权重直接翻转（+1→0→-1 或反向），翻转方向由 sign(g_t) 决定；(2) 百分位阈值起自适应选择性作用——仅梯度幅值最大的 top-k% 权重被更新，小梯度被视作噪声而过滤；(3) σ_t 的线性衰减实现粗到细的搜索策略（早期高阈值聚焦关键参数做"大调整"，后期低阈值允许"精细调整"）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# t-SignSGD 单步更新（以 A_T 为例）
# 输入: A_T (当前三元适配器, 值 ∈ {-1,0,1})
#       g (梯度, same shape), τ (最小阈值, 1e-9)
#       σ (当前动态百分位阈值)
#       step, total_steps

# 1. 线性衰减 σ_t (前 80% 训练衰减)
if step < 0.8 * total_steps:
    σ_t = σ_init * (1 - step / (0.8 * total_steps))  # 5% → 0.1%
else:
    σ_t = 0.0001  # 后 20% 固定 0.01%

# 2. 计算百分位阈值（基于梯度幅值分布）
|g|_flat = abs(g).flatten()
σ_t_value = percentile(|g|_flat, (1 - σ_t) * 100)

# 3. 选择更新位置并更新
threshold = max(τ, σ_t_value)
update = (abs(g) > threshold)       # bool mask
A_T[update] -= sign(g[update])      # +/-1 翻转
A_T = clip(A_T, -1, 1)              # 保持三值约束
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
t-SignSGD 在 LoTA-QAF 中通过 PyTorch 自定义优化器实现（t_signSGD.py），继承 torch.optim.Optimizer。百分位阈值计算使用 torch.quantile。由于无动量机制和自适应学习率，t-SignSGD 的收敛性依赖 σ_t 的衰减调度设计：噪声过滤（低幅值梯度被阈值过滤，防止离散空间的震荡）+ 退火式搜索（粗→细的探索-利用平衡）。当前实现局限：(1) 无一二阶动量；(2) 仅线性衰减调度；(3) 未探索余弦退火或循环调度。LoTA-QAF 实验显示，在 4-bit/3-bit 量化下 t-SignSGD 收敛损失与 LoRA 差距 < 0.01；2-bit 下差距较大（0.132 vs 0.375），因三值调整在仅 4 个可能值的量化空间中更不稳定。

涉及论文标题：
- LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

## Quantization-Aware Fine-Tuning (QAF, 量化感知微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware Fine-Tuning（QAF，量化感知微调）是介于 PTQ（后训练量化）和 QAT（量化感知训练）之间的范式：模型首先通过 PTQ 方法（如 GPTQ）被量化，然后在冻结的量化权重上训练轻量级适配器进行微调。与 QAT 在训练中模拟量化的全参数训练不同，QAF 保持量化权重冻结（或通过适配器间接调整），以较低的微调成本实现量化模型的性能恢复和任务适配。代表性工作包括 QLoRA（16-bit LoRA 适配器）、QA-LoRA（适配器调整零点因子实现无损合并）、LoTA-QAF（三值适配器直接调整量化权重实现无损合并）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QAF 的通用流程（以 LoTA-QAF 为例）：
```
# Stage 1: PTQ 量化（离线，一次性）
W_int, s, z = gptq_quantize(W_fp16, calibration_data)
# 得到量化权重和量化参数

# Stage 2: QAF 微调（在线，trainable）
freeze(W_int, s)  # 量化参数冻结（或通过适配器间接更新）
init_adapters()   # 初始化适配器（LoRA/TA/QA-LoRA）
for batch in fine_tuning_data:
    # 前向：量化权重 + 适配器调整
    if method == "LoRA":  # QLoRA 方式
        y = (dequant(W_int, s, z) + α/r * A@B)^T @ x
    elif method == "QA-LoRA":
        z' = z + adapter_output  # 仅调整零点
        y = (s * W_int + z')^T @ x
    elif method == "LoTA-QAF":
        W'_int = clamp(W_int + ternary_adjustment, 0, 2^N-1)
        z' = z + s * μ_offset
        y = (s * W'_int + z')^T @ x
    loss = criterion(y, label)
    loss.backward()
    adapter_optimizer.step()  # 仅更新适配器

# Stage 3: 合并 & 推理（适配器合并入量化权重）
merge_adapters_into_quantized_weights()
inference_with_pure_low_bit_weights()  # 无适配器开销
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QAF 的三种主要模式：(1) QLoRA 模式——16-bit 适配器在量化权重上训练，推理时适配器与量化权重分开计算，存在混合精度开销；(2) QA-LoRA 模式——适配器结构对齐分组量化的零点因子，训练后无损合并进零点，但仅能间接调整量化权重；(3) LoTA-QAF 模式——三值适配器直接在量化网格内调整 W_int 和 z，训练后无损合并，推理零开销。QAF 的核心评估场景有两种：性能恢复（performance-recovery，用通用数据微调恢复量化造成的性能损失）和任务特化（task-specific，在特定下游任务上微调）。在 LoTA-QAF 实验中，QAF 展示了在低比特（尤其是 2-bit/3-bit）场景下显著恢复 PTQ 方法未充分利用的性能潜力（如 Qwen 2.5 32B 2-bit 微调后提升 16.97%）。

涉及论文标题：
- LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning

## KV Cache (Key-Value Cache, KV缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache（Key-Value Cache）是 Transformer 自回归推理中用于存储每层注意力机制中所有已生成 token 的 Key 和 Value 张量的高速缓存。在 prefill 阶段，输入 prompt 经各层权重 $W_K, W_V$ 投影得到 $X_K, X_V \in \mathbb{R}^{b \times l_{prompt} \times d}$，存入 KV Cache。在 decoding 阶段，每生成一个新 token $t$，计算 $t_K = tW_K, t_V = tW_V$ 后 Concat 到 KV Cache：$X_K \leftarrow \operatorname{Concat}(X_K, t_K)$。然后通过 $A = \operatorname{Softmax}(t_Q X_K^\top)$ 和 $t_O = A X_V$ 完成注意力计算。KV Cache 避免了每一 decoding step 都重新计算所有历史 token 的 Key/Value，将计算复杂度从 $O(l^2 d)$ 降至 $O(ld)$。但代价是内存占用巨大：KV Cache 形状为 $b \times (l_{prompt} + l_{gen}) \times d$，在 OPT-175B、b=512、l=544 时可达 1.2TB（3.8× 模型权重）。KIVI 论文针对 KV Cache 内存瓶颈提出了 2bit 量化方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KV Cache 在 LLM 推理 pipeline 中的生命周期：

```
# === Prefill Phase ===
输入: X ∈ R^{b × l_prompt × d}
for layer in layers:
    X_K = X @ W_K    # [b, l_prompt, d]
    X_V = X @ W_V    # [b, l_prompt, d]
    KV_cache[layer] = (X_K, X_V)  # 存储全精度 KV
    # ... attention computation ...
    X = output  # 传给下一层

# === Decoding Phase (per token) ===
输入: t ∈ R^{b × 1 × d}
for layer in layers:
    t_K = t @ W_K,  t_V = t @ W_V
    X_K, X_V = KV_cache[layer]           # 从内存加载
    X_K = Concat([X_K, t_K], dim=token)  # 追加新 token
    X_V = Concat([X_V, t_V], dim=token)
    KV_cache[layer] = (X_K, X_V)          # 更新缓存
    
    t_Q = t @ W_Q
    A = Softmax(t_Q @ X_K^T / sqrt(d))   # attention scores
    t_O = A @ X_V                         # attention output
    # ... FFN ...
    t = output
```

内存分析：KV Cache 总大小为 $2 \times n_{layers} \times b \times (l_{prompt} + l_{gen}) \times d_{head} \times n_{heads}$ bytes（FP16 时乘 2）。KIVI 将 Key/Value 压缩到 2bit 后减少了约 8× 的 KV Cache 内存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 中 `model.generate(use_cache=True)`（默认）自动管理 KV Cache。vLLM 中使用 PagedAttention 将 KV Cache 分页管理以消除碎片化。FlexGen 通过 offloading 将 KV Cache 转移到 CPU/磁盘。KIVI 通过量化将 KV Cache 压缩到 2bit 以减少 GPU 内存占用，兼容 weight-only 量化和 PagedAttention。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---

## Per-Channel Quantization (逐通道量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Channel Quantization 是将张量沿 channel（特征/隐藏维度）方向分组量化的策略。对于 KV Cache $X \in \mathbb{R}^{l \times d}$，per-channel 量化意味着沿 d 维度分组，每若干 channel 共享一组量化参数（scale, zero-point）。每个 channel 的量化参数独立计算，使得误差被限制在各自的 channel 内。

KIVI 论文的核心发现：key cache 中少量固定 channel 存在极大的 magnitude outlier，使用 per-token 量化时这些 outlier 会污染同一 group 内的所有正常 channel，导致 attention score 相对误差高达 47%。而 per-channel 量化将 outlier channel 的量化误差隔离在自身 channel 内，不干扰其他 channel，attention score 相对误差降至 9.6%（约 5× 更低）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 中 per-channel key cache 量化（KeyQuant 函数）：

```
procedure KeyQuant(X_K ∈ R^{l × d}):
    r = l % R                    # 不能被R整除的余数
    X_K_g = X_K[:l - r]          # grouped部分
    X_K_r = X_K[l - r:]          # residual部分 (FP16)
    Q(X_K_g) = GroupQuant(X_K_g, dim=channel, numGroup=l // G)
    # dim=channel: 沿channel维度分组，每G个连续token为一组
    # 每组内共享 scale 和 zero-point
    return Q(X_K_g), X_K_r
```

与 per-token 对比：
- per-token: X[l × d] → 沿 token 维分 d/G 组，每 G 个 channel 共享 scale
- per-channel: X[l × d] → 沿 channel 维分 l/G 组，每 G 个 token 共享 scale

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 使用 Triton 实现 per-channel group-wise quantization kernel。实现时每组 G=32 个 token，计算组内 min/max → scale=(max-min)/(2^B-1), zero-point=min → round((x-z)/s) → clamp。由于 per-channel 量化跨 token 维度，新到达 token 无法直接 append，KIVI 通过 grouped+residual split 解决流式兼容问题。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---

## Per-Token Quantization (逐Token量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Token Quantization 是将张量沿 token（序列）维度分组量化的策略。对于 KV Cache $X \in \mathbb{R}^{l \times d}$，per-token 量化意味着沿 l 维度分组，每若干 token 共享一组量化参数。这种量化方式与自回归生成的流式特性天然兼容——新生成 token 的量化张量可直接沿 token 维度 append 到已有量化缓存。

KIVI 论文的核心发现：value cache 必须使用 per-token 量化。value cache 无 channel-wise outlier 模式，但由于 attention output $t_O = AX_V = \sum_j A_{ij}[X_V]_{j*}$ 是 value cache 行向量的加权求和（权重为稀疏 attention score），per-token 量化将误差限制在每个 token 内部，使重要 token 不受其他 token 量化影响。per-channel 量化 value cache 会导致 attention output 相对误差比 per-token 高约 15×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 中 per-token value cache 量化：

```
# Prefill Phase: 初始量化
X_V_g = X_V[:l_prompt - R]     # grouped part
X_V_r = X_V[l_prompt - R:]     # residual (FP16)
Q(X_V_g) = GroupQuant(X_V_g, dim=token, numGroup=d // G)
# dim=token: 沿token维度分组, 每G=32个channel共享scale

# Decoding Phase: 流式追加
X_V_r = Concat([X_V_r, t_V], dim=token)   # 新token进入residual
if len(X_V_r) > R:
    outdated = X_V_r[:-R]                  # 超出窗口的旧token
    Q(outdated) = GroupQuant(outdated, dim=token, G=32)  # per-token quant
    Q(X_V_g) = Concat([Q(X_V_g), Q(outdated)], dim=token)
    X_V_r = X_V_r[-R:]                     # 保留最近R个token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlexGen 最先使用 4bit per-token group-wise 量化 key 和 value cache，但未区分 key/value 的不同分布。KIVI 在 FlexGen 基础上提出非对称策略：key per-channel、value per-token。标准化实现步骤：(1) 沿 token dim 分 group；(2) 每 group 计算 min/max → scale/zero-point；(3) round-to-nearest 量化。per-token 量化与流式推理天然兼容，是 KV Cache 量化中最常见的量化维度选择。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---

## KV Cache Quantization (KV 缓存量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache Quantization 是一类将 LLM 推理中的 KV Cache 从高精度（FP16/BF16）压缩到低比特整数（INT4/INT2）以减少 GPU 内存占用的技术。与权重量化不同：(1) KV Cache 是流式数据结构——新 token 的 K/V 实时到达并追加，无法使用需要离线校准的优化方法（如 GPTQ 的 Hessian 补偿）；(2) KV Cache 的数值分布随序列长度动态变化，预计算的量化参数可能失效；(3) 量化误差不仅影响当前层还通过 residual 累积传播。

KIVI 提出了一种免调优的非对称 2bit KV Cache 量化：基于对 key/value cache 元素分布的深入分析，发现 key cache 应 per-channel 量化（隔离 outlier channel），value cache 应 per-token 量化（保护重要 token 不受干扰）。同时通过 grouped+residual split 实现流式兼容和局部全精度滑动窗口。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 的 KV Cache 量化与 attention 计算融合流程：

```
# KV Cache 结构: {Q(X_K_g): int2, X_K_r: FP16, Q(X_V_g): int2, X_V_r: FP16}
# Q(X_K_g): per-channel group-wise 2bit quantized
# Q(X_V_g): per-token group-wise 2bit quantized
# X_K_r, X_V_r: 最近R个token保持FP16 (full precision sliding window)

# 每 decoding step:
t_Q = t @ W_Q, t_K = t @ W_K, t_V = t @ W_V

# --- 更新 KV Cache ---
X_K_r = Concat([X_K_r, t_K])
X_V_r = Concat([X_V_r, t_V])
if len(X_K_r) == R:
    Q(X_K_r_new) = KeyQuant(X_K_r)         # per-channel quant
    Q(X_K_g) = Concat([Q(X_K_g), Q(X_K_r_new)])
    X_K_r = empty

# --- 混合精度 Attention ---
A_g = t_Q @ Dequant(Q(X_K_g))^T  # grouped部分 (fused dequant+matmul)
A_r = t_Q @ X_K_r^T              # residual部分 (FP16)
A = Concat([A_g, A_r])
A_g_sm, A_r_sm = Softmax(A)[:-R], Softmax(A)[-R:]

t_O = A_g_sm @ Dequant(Q(X_V_g)) + A_r_sm @ X_V_r  # 混合精度加权求和
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
早期 FlexGen 使用 4bit per-token group-wise RTN 量化 KV Cache，但未区分 key/value。SmoothQuant 可通过 equivalent transformation 迁移量化难度使 KV Cache 可 8bit 量化，但 4bit 精度大幅下降。KIVI 是第一篇系统研究 KV Cache 元素分布并设计非对称量化策略的工作（与 KVQuant 同期独立发现）。KIVI 代码开源：https://github.com/jy-yuan/KIVI，基于 HuggingFace Transformers，使用 CUDA (fused dequant+matmul) + Triton (group-wise quantization) 实现。QuaRot 从不同角度解决 KV Cache 量化问题：通过 head-wise Hadamard 旋转消除 Key 和 Value 中的离群值（与消除激活值离群值相同的原理），使简单的 asymmetric group-wise INT4 量化（group=128）即可在 4-bit KV cache 下实现近乎无损的困惑度（+0.04 on 7B）。QuaRot 使用 Post-RoPE Caching（在 RoPE 后做在线 head-wise Hadamard 旋转再量化缓存），避免了解码时需对大量缓存 key 做逆旋转的开销。

ResQ 将混合精度量化扩展到 KV Cache：通过 U_B 和 U_C 两个投影矩阵分别处理 value 和 key 的 KV cache 量化。(1) U_B 后乘 value 投影层 W_v，将 value 向量投影到 PCA+PCA 基上，离线融合 U_B^T 到 o_proj 权重；(2) U_C 利用 computational invariance 同时投影 query 和 key（对称投影），因 RoPE 无法融合而在运行时显式计算，但量化为 8-bit 以降低开销。最终 KV cache 以 4/8-bit 混合精度存储（1/8 通道 8-bit）。key/value cache 均采用 per-head asymmetric 量化。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

---

## Attention Sparsity (注意力稀疏性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Sparsity 指 Transformer 的自注意力分数矩阵 $A \in \mathbb{R}^{l \times l}$ 中只有极少数 token 对具有显著权重（非零），大部分 token 对的 attention score 接近于零的现象。这是 softmax 指数归一化后的固有特性：softmax 将指数归一化后的分数集中到极少数"热点"。KIVI 论文中 Llama-2-13B 的 attention sparsity 高达 84.3%，即超过 84% 的 attention 权重接近零。

Attention sparsity 是 KIVI per-token value cache 量化有效性的理论基础：由于 $t_O = \sum_j A_{ij} [X_V]_{j*}$，attention output 只是少数重要 token 的 value cache 加权组合。Per-token 量化将误差限制在每个 token 内，quantizing 不重要 token 不影响重要 token 的精度，因此 per-token value 量化误差远小于 per-channel（约 15×）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Attention sparsity 与 value cache 量化的关系：

```
# 设 attention score 稀疏（仅少数 token 显著）
A = [[0.01, 0.02, 0.85, 0.01, 0.02, ...]]  # token 2 是heavy-hitter

# Per-token value quantization:
# value cache token j → 被量化为 Q([X_V]_j)
# 误差: ε_j = [X_V]_j - Q([X_V]_j)
# end-to-end error: Δ = Σ_j A_j * ε_j
# 由于A_j≈0 for j≠heavy-hitter, Δ ≈ A_heavy * ε_heavy
# → 仅heavy-hitter token的量化误差被放大

# Per-channel value quantization:
# 跨token共享量化参数 → token间误差混合
# heavy-hitter value 受其他token量化参数影响 → 精度崩塌
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Attention sparsity 被用于多种优化：H2O 基于 sparsity 只保留 heavy-hitter token 的 KV Cache；Scissorhands 利用 sparsity 的持久性 evict 不重要 token；KIVI 利用 sparsity 解释 per-token value 量化有效性的理论基础。sparsity 通常通过 `torch.topk(A, k)` 或阈值过滤 `A > threshold` 检测 heavy-hitter。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---

## Full Precision Sliding Window (全精度滑动窗口 / Residual KV Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Full Precision Sliding Window（Residual KV Cache）是 KIVI 中保留最近 R 个 token 的 Key/Value 在 FP16 精度、不进行量化的设计。KIVI 将 KV cache 分为 grouped 部分（量化）和 residual 部分（FP16），residual 部分大小上限为 R（通常 R=128），形成大小为 R/2（key）和 R（value）的全精度滑动窗口。

该设计的关键作用：对于 GSM8K 等数学推理等硬任务，fake 2bit 全量化准确率显著下降（Llama-2-7B GSM8K: 16bit=13.50, fake 2bit=5.76），但 KIVI-2（含 residual window）可达 12.74（仅约 1% 下降）。全精度滑动窗口保留局部重要 token 的精确 attention 计算能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Residual KV Cache 的管理逻辑：

```
初始化: R = 128 (residual length), G = 32 (group size)
new_tokens = 0

# 每次 decoding step:
X_K_r = Concat([X_K_r, t_K])   # new token added in FP16
X_V_r = Concat([X_V_r, t_V])   # newly arrived, keep FP16

if len(X_K_r) == R:
    # Residual满了, 将整组R个token量化并移入grouped
    Q(X_K_r) = KeyQuant(X_K_r)              # per-channel quant
    Q(X_K_g) = Concat([Q(X_K_g), Q(X_K_r)]) # merge into grouped
    X_K_r = empty                            # reset residual

if len(X_V_r) > R:
    # Value: 保留最近R个, 超出的量化移入grouped
    Q_outdated = GroupQuant(X_V_r[:-R], dim=token)
    Q(X_V_g) = Concat([Q(X_V_g), Q_outdated])
    X_V_r = X_V_r[-R:]   # keep latest R tokens FP16

# Attention: combined from both parts
A = Concat([t_Q @ Dequant(Q(X_K_g))^T,  t_Q @ X_K_r^T])
# Softmax split → weighted sum from quantized V_g + FP16 V_r
```

记忆开销分析：当 R ≤ 128 且序列长度 l ≫ R 时，residual FP16 部分的额外内存占总 KV Cache 比例 ≈ 128/l。例如 l=8192, R=128 时额外开销仅 1.6%，远小于量化带来的 8× 内存节省。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 中 R 默认为 128。ablation 实验显示 R∈{32, 96, 128} 效果相近但都远好于无 residual（fake quantization），说明一定量的全精度窗口对于保持精度至关重要。实现时 residual 和 grouped 通过 tiled matrix multiplication 组合计算 attention。类似设计也被 StreamingLLM 等使用（保留最近的 token 维持精度），但目的不同（StreamingLLM 是 evict 而非 quantize）。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---

## Asymmetric KV Cache Quantization (非对称KV Cache量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric KV Cache Quantization 是 KIVI 提出的核心设计理念：对 key cache 和 value cache 使用不同维度的量化策略——key per-channel（沿特征维度）、value per-token（沿序列维度）。这里的 "asymmetric"（非对称）不是指通常量化中 scale_pos≠scale_neg 的 asymmetric quantization，而是指 key 和 value 两个 caches 使用非对称的量化维度选择。

设计理由：(1) Key cache 少数固定 channel 存在极大 outlier（与 SmoothQuant/AWQ 观察一致），per-channel 量化将 error 隔离在 outlier channel 内；(2) Value cache 虽无 outlier 模式，但因 attention output = sparse-weighted sum of value tokens，per-token 量化保护重要 token 不受干扰。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI asymmetric quantization 与 baseline uniform quantization 的对比：

```
# Baseline (FlexGen): 统一 per-token 量化
Q(X_K) = GroupQuant(X_K, dim=token)  # key per-token
Q(X_V) = GroupQuant(X_V, dim=token)  # value per-token
# 问题: INT4 OK, INT2精度崩塌 (Llama-2-13B CoQA: 66.37→52.93)

# KIVI: 非对称量化
Q(X_K) = GroupQuant(X_K, dim=channel)  # key per-channel
Q(X_V) = GroupQuant(X_V, dim=token)    # value per-token
# 2bit效果: Llama-2-13B CoQA: 66.37→66.23 (几乎无损)
```

KIVI 实验证实的其他配置均不可行：
- 2bit (K per-token, V per-channel): CoQA 2.80（全坏）
- 2bit (K per-channel, V per-channel): CoQA 2.88（全坏）
- 2bit (K per-token, V per-token): CoQA 52.93（差但能用）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现关键：(1) 对 key 使用 `GroupQuant(X_K, dim=channel, numGroup=l//G)`；(2) 对 value 使用 `GroupQuant(X_V, dim=token, numGroup=d//G)`。KIVI 依赖 CUDA/Triton kernel 实现这两种量化方向的 fused dequant+matmul。代码开源：https://github.com/jy-yuan/KIVI。该设计被后续 KV Cache 量化工作（KVQuant、GEAR、PM-KVQ）广泛引用和扩展。PM-KVQ 在 KIVI 的非对称 per-channel Key + per-token Value 量化基础上，进一步叠加渐进量化和块级内存分配，并将首 token INT16 + 128 token 滑动窗口作为默认保留策略。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---

## Shift Quantization (幂次量化 / Power-of-Two Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shift Quantization（也称 Power-of-Two Quantization）是一种将神经网络权重约束为 2 的幂次值或零的量化方案。一个 (k+2)-bit 的 shift quantization 表示值为 `v = s * 2^{e-b}`，其中 `s ∈ {-1, 0, 1}` 表示符号（或零），`e ∈ [0, 2^k-1]` 为 k-bit 无符号指数，`b` 为逐层 bias 常数，用于缩放量化值的大小范围。例如 5-bit shift quantization：1-bit 符号 + 3-bit 指数 (k=3) + 1-bit 指示 pruning mask。

核心优势：量化后的权重值均为 2 的幂次，因此卷积中的乘法 `a * v = a * s * 2^{e-b}` 可替换为 bit-shift 操作 `s * (a << (e-b))`（或 `>>` 当 e < b），在硬件中消除乘法器阵列，大幅降低逻辑门数和功耗。

主要局限：量化层级在零附近最密集（±1, ±2, ±4, ±8, ...），距离零越远层级越稀疏。当 CNN 经过细粒度剪枝后，权重分布往往呈现"中空"现象——大量非零权重集中远离零的区间，近零的量化层级利用率极低，造成精度浪费。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Shift Quantization 在 CNN 推理 pipeline 中的执行流程：

```
# Input: weight tensor W ∈ R^{Cout, Cin, Kh, Kw}
#        bias b (layer-wise constant, power-of-two)
#        bit-width n (= k + 2)

# Step 1: Clamp weights to representable range
max_val = 2^{(2^k - 1) - b}
min_val = 2^{0 - b}
for each w in W:
    w_clamped = clamp(|w|, min_val, max_val)

# Step 2: Quantize to nearest power-of-two
for each w in W:
    # Find nearest representable power-of-two
    # Representable values: {0, ±2^{-b}, ±2^{1-b}, ..., ±2^{(2^k-1)-b}}
    v = round(log2(|w|) + b)  # quantized exponent
    v = clamp(v, 0, 2^k - 1)
    w_hat = sign(w) * 2^{v - b}

# Step 3: Convolution with bit-shift instead of multiply
# For activation a and quantized weight w_hat = s * 2^{e-b}:
output += s * (a << (e - b))  # when e >= b
output += s * (a >> (b - e))  # when e < b
```

**Annotations**: `k` 为指数位数（5-bit shift quant 中 k=3）；`b` 为 layer-wise bias，控制整层的 magnitude scale；sign 通过 MSB 编码；量化层级间距随 |v| 增大呈指数增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Shift quantization 的实现通常在训练后或训练中应用。INQ（Incremental Network Quantization）[26] 是代表性的 shift quantization 训练方法：将权重分组为已量化和未量化两组，逐步将未量化权重量化到最近的 2 的幂次值并冻结，仅微调未量化组。更通用地在 training-aware quantization 流程中，forward pass 使用 `Q^{shift}_{n,b}(w)` 计算量化权重，backward pass 将量化函数视为恒等映射（Straight-Through Estimator, STE）。

Mayo 框架（https://github.com/deep-fry/mayo）提供了完整的 shift quantization 工具链。推理时，bit-shift 替代乘法在 CPU（LEA/SAL 指令）、GPU（整数移位指令）和 FPGA（移位寄存器）上均可高效实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Focused Quantization (FQ / 聚焦量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Focused Quantization (FQ) 是专为稀疏 CNN 设计的混合量化策略，由 Cambridge/SIAT/UMacau 团队于 2019 年提出。FQ 的核心思想是：**将量化 effort（即量化层级资源）从权重分布稀疏的零附近重新集中到剪枝后权重实际分布的高概率区域**。FQ 包含两个子策略：

1. **Recentralized Quantization**：对逐层非零权重拟合 2-分量高斯混合模型（GMM，使用 EM 算法求 MLE），找到两个高概率密度聚类（Cluster+ 和 Cluster-），分别对每个聚类以各自的均值和标准差做零中心化（减均值、除标准差），再在归一化后的小范围内做 shift quantization，最后反变换回去。公式为 `Q_c^{rec}[θ] = Q^{shift}_{n,b}[(θ-μ_c)/σ_c] * σ_c + μ_c`。
2. **Shift Quantization（退化模式）**：当两个高斯分量高度重叠时（2-Wasserstein 距离 < 阈值 w_sep=2.0），Recentralized Quantization 退化为普通 shift quantization，且因为不需要 component selection bit，精度等效提升 1 bit。

FQ 的 5-bit 实现实际使用：1-bit sign + 1-bit component selection + 3-bit unsigned shift value = 5 bits total。硬件实现中，μ_c 额外量化为 2 的幂次值，σ_+ 和 σ_- 约束相等并融合到 α 缩放因子中，α 再融入 BN，消除推理时的所有乘法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FQ 完整 pipeline 伪代码（逐层处理）：

```
# ===== Preprocessing: EM for GMM fitting =====
θ_nonzero = {w ∈ W | w ≠ 0}

# Initialize
μ_- = mean({θ < 0}), σ_- = std({θ < 0})
μ_+ = mean({θ > 0}), σ_+ = std({θ > 0})
λ_- = λ_+ = 0.5

# EM Algorithm (repeat until convergence)
repeat:
    # E-step: Compute responsibilities
    for each θ:
        γ_c(θ) = λ_c * N(θ|μ_c, σ_c) / Σ_j λ_j * N(θ|μ_j, σ_j)

    # M-step: Update parameters
    for each component c:
        N_c = Σ_θ γ_c(θ)
        μ_c = Σ_θ γ_c(θ) * θ / N_c
        σ_c^2 = Σ_θ γ_c(θ) * (θ - μ_c)^2 / N_c
        λ_c = N_c / |θ|

# ===== Wasserstein Separation Check =====
σ²_global = Var(θ_nonzero)
W = ((μ_+ - μ_-)² + (σ_+ - σ_-)²) / σ²_global

if W < w_sep (2.0):
    # Use plain Shift Quantization (higher precision by 1 bit)
    for each θ:
        θ_hat = Q^{shift}_{n,b}(θ)  # n-bit, no component bit needed
else:
    # Use Recentralized Quantization
    # Quantize μ_c to nearest power-of-two
    for each θ:
        m_θ = argmax_c λ_c * N(θ|μ_c, σ_c)  # component assignment
        θ_norm = (θ - μ_{m_θ}) / σ_{m_θ}    # recentralize
        θ_hat_norm = Q^{shift}_{n-1,b}(θ_norm)  # (n-1)-bit shift quant
        θ_hat = α * (θ_hat_norm * σ_{m_θ} + μ_{m_θ})

# ===== Fine-tuning with INQ schedule =====
# Gradually increase quantized proportion: 25% → 50% → 75% → 87.5% → 100%
# Fine-tune at each step, update GMM hyperparams every k epochs
```

**Annotations**: GMM fitting 依赖剪枝后权重分布；EM 收敛到局部最优；Wasserstein 阈值 w_sep=2.0 是通过 CIFAR-10 上 9 层 CNN 的 grid search（1.0~3.5, step=0.1）确定的；INQ schedule 中每步 3 epochs（最后一步 10 epochs）, LR=0.001, 每 3 epochs 衰减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FQ 作为压缩 pipeline（称为 Focused Compression, FC）的一部分使用：Dynamic Network Surgery 细粒度剪枝 → FQ 量化 → INQ 增量 fine-tune → Huffman 编码。FQ 的逐层自适应特性使其特别适用于剪枝后稀疏度不均匀的 CNN（ResNet、MobileNet 等）。开源实现见 Mayo 框架 (https://github.com/deep-fry/mayo)。FQ 最适合的场景是：同时追求高压缩率（18× CR on ResNet-50）和极低硬件开销（5-bit FQ 仅需 275.6M 逻辑门，与 3-bit shift quant 相当），且对精度损失容忍度低（Top-5 损失 ≤0.24%）。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Recentralized Quantization (再中心化量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recentralized Quantization 是 Focused Quantization 的核心子机制。传统 shift quantization 假定权重以零为中心分布（量化层级最密集在零附近），但剪枝后 CNN 的权重分布"中空"——非零权重集中在远离零的特定值区域。Recentralized Quantization 解决这一问题：先对每层非零权重拟合 GMM 找到高概率聚类的均值 μ_c 和标准差 σ_c，然后对每个聚类独立做"再中心化"——即对归属于聚类 c 的权重 θ，先做 (θ - μ_c) / σ_c 将其映射为零均值、单位方差的分布，在此归一化空间内做 shift quantization（此时量化层级密集在零附近且恰好对应权重高概率区域），最后再反归一化回到原始尺度。

数学表达：`Q_c^{rec}[θ] = Q^{shift}_{n,b}[(θ - μ_c)/σ_c] * σ_c + μ_c`。其中 `Q^{shift}_{n,b}` 是 n-bit shift quantization，`μ_c` 经额外汇总量化为 2 的幂次值以保持硬件效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 ResNet-50 block3f/conv1 层为例（Figure 2 in paper）：

```
# 该层权重分布呈现双峰: 正权重簇在 ~0.10, 负权重簇在 ~-0.10

# Step 1: GMM fitting
# EM 算法收敛后:
μ_- ≈ -0.10, σ_- ≈ 0.02  (负值聚类)
μ_+ ≈ +0.10, σ_+ ≈ 0.02  (正值聚类)

# Step 2: 分量分配 (per weight)
# Example: θ = 0.095
p(m_θ = +) = λ_+ * N(0.095|0.10, 0.02) / q_mix(0.095) ≈ 0.92
p(m_θ = -) = λ_- * N(0.095|-0.10, 0.02) / q_mix(0.095) ≈ 0.08
→ m_θ = + (正分量)

# Step 3: Recentralization (将 θ 映射到零附近)
θ_norm = (θ - μ_+) / σ_+ = (0.095 - 0.10) / 0.02 = -0.25
# θ_norm ≈ 0, 位于 shift quantization 层级密集区

# Step 4: Shift quantization (利用零附近的高精度层级)
# 3-bit shift quant: v = s * 2^{e-b}
# θ_norm ≈ -0.25 → nearest power-of-two: -0.25 = -2^{-2}
θ_hat_norm = -2^{-2} = -0.25  (exact match, no error)

# Step 5: De-normalization
θ_hat = α * (θ_hat_norm * σ_+ + μ_+)
      = α * (-0.25 * 0.02 + 0.10) = α * 0.095
# 完美恢复原始值 (量化前后无损)
```

**Annotations**: 若不用 recentralization 而直接用 shift quantization，θ=0.095 的最近 2 的幂次值为 0.0625 (2^{-4}) 或 0.125 (2^{-3})，量化误差大得多。Recentralization 通过将每个聚类独立零中心化，使每个聚类内的量化精度最大化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Recentralized Quantization 不单独使用，而是作为 FQ 框架的内部组件。在实现中，GMM 的 EM 拟合为逐层离线计算（fine-tuning 期间每 k 个 epoch 更新一次超参数）；硬件实现中 μ_c 额外量化为 2 的幂次值，σ_+ = σ_- 约束为相等以融入逐层 α。Mayo 框架 (https://github.com/deep-fry/mayo) 包含完整实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Wasserstein Separation for Quantization Mode Selection (Wasserstein 分离度量)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wasserstein Separation 是 FQ 中用于自适应选择量化模式的决策度量。当剪枝后某层的权重分布没有明显的多峰结构（即 GMM 的两个分量高度重叠），recentralized quantization 与普通 shift quantization 效果几乎等价——因为两个分量差异过小，具体选择哪个分量来量化某个权重对结果影响极小。此时浪费 1 bit 来编码 component selection 反而不如将该 bit 用于增加 shift quantization 的精度。

FQ 使用 2-Wasserstein 距离（归一化后）来衡量两个高斯分量的分离程度：
`W(c₁,c₂) = ((μ_c₁ - μ_c₂)² + (σ_c₁ - σ_c₂)²) / σ²_global`
其中 σ²_global 是整层权重的方差。当 `W < w_sep`（默认 2.0），说明分量重叠严重，退化为 n-bit shift quantization（等效精度比 (n-1)-bit recentralized 高 1 bit）；否则使用 recentralized quantization。

选用 2-Wasserstein 而非 KL 散度的原因：KL 散度非对称，不同方向距离不一致，不适合作为阈值决策度量；Wasserstein 距离对称且同时考虑了均值和方差的差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Wasserstein 决策在整个 FQ pipeline 中位于 GMM 拟合之后、量化执行之前：

```
# After EM fitting
μ_-, σ_-, μ_+, σ_+, σ²_global ← from GMM fitting

# Compute Wasserstein Separation
W = ((μ_+ - μ_-)² + (σ_+ - σ_-)²) / σ²_global

# Decision
if W < 2.0:
    # 分量高度重叠 → shift quantization (n-bit, full precision)
    mode = "shift"
    # 可利用全部 n bits 做 shift quant (无 component bit)
else:
    # 分量充分分离 → recentralized quantization ((n-1)-bit internal)
    mode = "recentralized"
    # 用 1 bit 区分 component, 剩余 (n-1) bits 做 shift quant

# Example from paper:
# ResNet-18 block22/conv1 (Figure 3): W ≈ 1.2 < 2.0 → shift mode
# ResNet-50 block3f/conv1 (Figure 2): W ≈ 5.3 ≥ 2.0 → recentralized mode
```

**Annotations**: 阈值 w_sep=2.0 是经验值，通过在 CIFAR-10 9 层 CNN 上 grid search（1.0~3.5, step=0.1, 每个值训练 100 次取平均）确定。论文实验显示，w_sep 设置恰使 1 个重叠层使用 shift、其余 8 层使用 recentralized 时平均准确率最高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Wasserstein 分离度量作为逐层自适应判定，在 FQ 实现中完全自动化——给定量化 bit-width n 和阈值 w_sep，框架自动为每层计算 W 并选择最优模式。用户不需要手动干预。实现代码在 Mayo 框架中。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Incremental Network Quantization (INQ / 增量网络量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Incremental Network Quantization (INQ) 是由 Zhou et al. (ICLR 2017) 提出的 training-aware 量化方法，将全精度预训练 CNN 逐步转换为低精度（幂次或整数）模型。核心思想：不一次性量化所有权重，而是将权重分成两组——已量化组（冻结）和未量化组（可微调）。在每一步中，逐步扩大已量化比例（如 25% → 50% → 75% → 87.5% → 100%），未量化权重通过 fine-tuning 补偿前面步骤的量化误差。

INQ 的三种权重划分策略：(1) 随机划分；(2) 按 |weight| 最大权重优先量化（剪枝思路，大权重对输出影响最大，先量化再微调小权重补偿）；(3) 按量化误差最小优先。

在 FQ 论文中，INQ 被用作 FQ 量化的 fine-tuning schedule（25%→50%→75%→87.5%→100%，每步 fine-tune 3 epochs，最后一步 10 epochs，LR=0.001 每 3 epochs 衰减），使模型逐步适应 FQ 的量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
INQ 在 FQ pipeline 中的执行流程：

```
# Given: pre-trained & pruned model with FQ hyperparameters
# Schedule: partitions = [0.25, 0.50, 0.75, 0.875, 1.0]

for partition in partitions:
    # Step 1: Select weights to quantize (largest |weight| first)
    weights_to_quantize = top_k(|W|, partition * |W|)  # 按权重绝对值排序

    # Step 2: Quantize selected weights (frozen)
    for each w in weights_to_quantize:
        w_hat[w] = FQ_quantize(w)  # Q[θ] or Q^{shift}(θ)
        w.freeze()                  # 不参与后续梯度更新

    # Step 3: Fine-tune remaining (unquantized) weights
    for epoch in range(epochs_per_step):
        # Forward: use w_hat for quantized, w for unquantized
        # Backward: update only unquantized weights (STE through quantized)
        train_one_epoch()

    # Step 4: Update FQ hyperparameters (GMM fit) every k epochs
    if step % k == 0:
        update_GMM_hyperparams()

# Final step (partition=1.0): all weights quantized, final fine-tune
```

**Annotations**: `top_k` 选择权重绝对值最大的比例为 `partition`；freeze 通过 mask 实现，反向传播时梯度不更新已量化权重；GMM 更新间隔 k 指数增长以减少采样方差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
INQ 不需要特殊的框架支持，可在标准 PyTorch/TensorFlow 训练循环中实现。核心是维护一个 binary mask 区分已量化（冻结）和未量化（可训练）的权重。INQ 同样适用于 shift quantization 以外的量化方案（整数、三元、二值等），只要量化和微调可以交替进行。FQ 论文中使用 INQ 的随机划分策略（按权重绝对值最大优先量化），Mayo 框架 (https://github.com/deep-fry/mayo) 包含 INQ 实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Dynamic Network Surgery (动态网络手术 / 细粒度剪枝恢复)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Network Surgery (DNS) 是由 Guo et al. (NIPS 2016) 提出的细粒度剪枝方法。与传统的"剪完即弃"（prune and discard）不同，DNS 在剪枝后持续训练中允许被剪掉的连接**恢复**（surgery 操作）。具体地：DNS 维护一个 binary mask T_k 指示连接是否被剪枝。在训练迭代中：(1) 若 `|w_k * T_k|` 低于阈值 `a_k`，则 mask 被设为 0（剪枝）；(2) 若 `|w_k * (1 - T_k)|` 高于阈值 `b_k`，则 mask 被恢复为 1（surgery，恢复连接）。阈值 `a_k` 和 `b_k` 通过每层的均值和标准差自适应调整。

这种"可逆剪枝"设计使 DNS 能探索 sparser 的结构而不永久丢失有用的连接，最终达到比静态剪枝更高的压缩率和更好的精度。FQ 论文使用 DNS 作为其压缩 pipeline 的剪枝阶段，使 ResNet-50 达到 82.70% 稀疏度（仅 17.3% 连接保留）且精度损失仅 0.48% (Top-1)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DNS 剪枝训练循环伪代码：

```
# Initialize: W (weights), T (mask, all 1s)

for each training iteration:
    # Forward/backward with masked weights
    W_masked = W ⊙ T  # element-wise multiply with mask
    loss = forward(W_masked, x)
    backward(loss)

    # Update weights (all weights updated, even pruned ones)
    W = W - lr * ∇W

    # Dynamic pruning and surgery (per layer)
    for each layer l:
        # Compute layer-wise thresholds
        a_k = μ_l + c_a * σ_l  # pruning threshold
        b_k = μ_l + c_b * σ_l  # surgery (recovery) threshold
        # Note: c_b < c_a, so b_k < a_k

        for each weight w at index i:
            if T[i] == 1 and |W[i]| < a_k:
                T[i] = 0  # prune
            elif T[i] == 0 and |W[i]| > b_k:
                T[i] = 1  # surgery: restore connection

    # Periodically increase sparsity target
    if should_increase_sparsity():
        increase_threshold(c_a, c_b)  # tighten thresholds
```

**Annotations**: `c_a > c_b` 保证剪枝门槛高于恢复门槛，形成 hysteresis 防止连接反复切换；sparsity 通过逐步收紧阈值增量式提升；剪掉的权重仍然接收梯度更新（通过 1-T 通道），使其有机会在后续训练中恢复到有意义的值从而被"手术恢复"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DNS 的实现需要修改标准训练循环，在每个 optimizer step 后插入 pruning/surgery 逻辑。实现要点：(1) 维护 per-layer mask；(2) 每层的 μ_l 和 σ_l 从当前权重统计中计算（通常对非零权重）；(3) sparsity 通过 schedule 渐进增加（如 cosine annealing of sparsity target）。DNS 特别适合与后续量化和编码步骤组合成完整的压缩 pipeline（如 FQ 的 FC pipeline）。Mayo 框架 (https://github.com/deep-fry/mayo) 包含 DNS 实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Huffman Encoding for Model Compression (霍夫曼编码模型压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Huffman Encoding 是一种无损数据压缩算法，Deep Compression (Han et al., ICLR 2016) 首次将其引入神经网络模型压缩。在 CNN 压缩 pipeline 中，Huffman 编码作为最后一步，对量化后的权重进行无损编码——利用量化后权重值分布不均匀的特性（某些值出现频率远高于其他值），为高频值分配短码字、低频值分配长码字，进一步减少存储空间。FQ 论文的 FC pipeline 在 FQ 量化后应用 Huffman 编码，例如 ResNet-50 (5-bit FQ) 从 5.19 MB 变为最终压缩后大小（CR=18.08×）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Huffman 编码在 FC pipeline 中的位置和流程：

```
# After FQ quantization: weights are in low-precision format
# Example: quantized weight values and their frequencies
values = [0, +4, -2, +8, -8, 0, 0, +4, 0, -2, ...]
freq = {0: 82700000, +4: 5200000, -2: 3100000, +8: 1800000, -8: 1700000, ...}

# Step 1: Build Huffman tree
build_huffman_tree(freq)  # 贪心合并最低频率节点

# Step 2: Assign codes (shorter for higher freq)
codes = {0: "0", +4: "10", -2: "110", +8: "1110", -8: "1111"}

# Step 3: Encode quantized weights
for each w_hat in quantized_weights:
    bitstream += codes[w_hat]

# Step 4: Store codebook + bitstream
# Decompression: read codebook → decode bitstream → reconstruct weights

# Effective bits per weight:
# Original FQ: 5 bits/weight
# After Huffman: Σ freq_i * len(code_i) / total_weights
# Example: (82.7M*1 + 5.2M*2 + ...) / 100M ≈ 3.2 bits/weight avg
```

**Annotations**: Huffman 编码的压缩率取决于量化后权重值的分布熵——分布越集中（如大部分值为 0），压缩率越高。FQ 论文中压缩率从 5-bit fixed 进一步压缩到有效 ~3-4 bits/weight。解码需要存储码本（codebook），对模型尺寸略有增加但通常可忽略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Huffman 编码作为 lossless 后处理步骤，在模型部署前离线执行。推理前需先解码权重。实现可使用 Python `heapq` 构建 Huffman tree，或用 zlib/gzip 等通用压缩库。在 FQ 论文的硬件评估中，Huffman 编码对逻辑门数影响极小（275.6M → 276.4M gates, +0.3%），因为解压逻辑简单。Huffman 编码特别适合量化后分布高度偏斜的场景。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Minimum Description Length for Model Compression (MDL / 最小描述长度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Minimum Description Length (MDL) 是信息论中的模型选择原则，由 Hinton & van Camp (1993) 和 Graves (2011) 引入神经网络压缩。MDL 在模型压缩中将优化问题表述为在给定数据集 D 下，找到权重 θ 和超参数 φ 的最优编码，使得描述模型的总代价最小：`L(θ, φ) = L_E + L_C`，其中：

- `L_E = E_{θ~q_φ(θ)}[-log p(y|x, θ)]` 是**误差代价**（Error Cost），即量化模型在数据集上的交叉熵损失。
- `L_C = KL(q_φ(θ) || p(θ|D))` 是**复杂度代价**（Complexity Cost），即量化权重分布 q_φ(θ) 与原始后验分布 p(θ|D) 之间的 KL 散度。

直观上：`L_E` 惩罚精度损失，`L_C` 惩罚模型复杂度。FQ 将这一框架应用于量化：`L_E` 通过 SGD fine-tuning 优化，`L_C` 通过最小化 GMM 分布与原始权重分布的 KL 散度来近似——而拟合 GMM 的 MLE 恰好等价于最小化该 KL 散度。因此 FQ 的 EM+GMM 步骤和 fine-tuning 步骤交替进行，对应 MDL 目标中 `L_C` 和 `L_E` 的交替优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MDL 双目标优化在 FQ pipeline 中的对应关系：

```
# MDL Objective:
#   min L(θ, α, φ) = L_E + L_C
#
#   L_E = E_{θ~q}[ -log p(y|x, α, θ_hat) ]  ← Cross-entropy loss
#   L_C = KL( q_φ(θ) || p(θ|D) )            ← Distribution matching

# Optimization Strategy (alternating):

# Phase 1: Optimize L_C (Complexity Cost)
# → GMM fitting via EM (MLE):
#   这一步找到 q_φ^mix(θ) ≈ p(θ|D)
#   等价于 min KL( q_φ^mix(θ) || p(θ|D) )
for each layer:
    μ_c, σ_c, λ_c = EM_fit(θ_nonzero)  # Find best GMM

# Phase 2: Optimize L_E (Error Cost)
# → Fine-tuning with SGD:
#   固定量化超参数 φ，优化权重 θ 和缩放 α
#   Forward: θ_hat = Q[θ; φ]  (quantized)
#   Loss: CE(θ_hat(x), y)
#   Backward: STE through Q[·]
for epoch in range(epochs):
    train_one_epoch()  # Standard SGD

# Repeat: alternate between Phase 1 and Phase 2
# (FQ paper: update φ every k epochs, k increasing exponentially)
```

**Annotations**: MDL 框架解释了为什么 FQ 需要交替进行 GMM fitting 和 fine-tuning——二者分别优化 L_C 和 L_E；单纯的 post-training quantization 只优化 L_C 而忽略 L_E，会导致较大精度损失；而仅 fine-tuning 不调整量化参数则 L_C 无法改善。MDL 为 FQ 的设计提供了理论支撑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MDL 在模型压缩中主要作为理论框架（而非具体实现代码）使用。实际实现中不显式计算 KL 散度，而是通过 MLE（EM 算法）隐式最小化 L_C，通过 SGD 最小化 L_E。Hinton & van Camp (1993) 的原始工作使用 Gaussian 近似，Graves (2011) 将其推广到更一般的变分推断。FQ 使用 GMM 作为 q_φ 的参数化形式，比单一高斯更具表达力，能捕获剪枝后权重的多峰分布。

涉及论文标题：
- Focused Quantization for Sparse CNNs

## Weight Standardization (WS / 权重标准化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Standardization (WS) 是一种对神经网络中卷积层或全连接层权重向量进行标准化的训练技术，由 Qiao et al. (2019) 提出。对于第 l 层的权重向量 w_{n,m}（其中 m 索引输出通道），WS 将其标准化为 w̃_{n,m} = (ρ/σ(w_{n,m})) · (I - P_1) w_{n,m}。具体流程：(1) 先减均值（通过投影矩阵 P_1 = 1·1^T/d 去除 DC 分量），(2) 除以标准偏差 σ（归一化到单位方差），(3) 乘以缩放系数 ρ（可调超参数）。标准化后的 WSP (Weight-Standardized Parameter) 才被用于前向计算（卷积或矩阵乘法）。WS 通常与 Group Normalization (GN) 搭配使用，放在 GN 层之前。在 FL 场景中，WS 的核心价值在于其对梯度的隐式过滤作用：反向传播时梯度经历双重投影 ∂L/∂w = (ρ/σ)(I - P_1)(I - P_{w̃}) ∂L/∂w̃，依次移除与 WSP 对齐的分量（local overfitting 方向）和 mini-batch 均值分量（local data bias 方向），仅保留对全局收敛有益的方向。FedWSQ 传输的是 PSP (Pre-Standardized Parameter) 而非 WSP，区别于 FedWon（传输 WSP 强制统计一致但丢失本地适应性信息）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WS 在 FedWSQ local training 中的伪代码（每层每个输出通道 m）：

```python
# WS Forward (applied before each layer's computation)
w = local_model.layer[l].weight[:, m]  # shape: (I_l,), PSP vector
w_mean = w.mean()                        # scalar
w_centered = w - w_mean                  # (I - P_1)w, projection onto span{1}^⊥
w_std = w_centered.std()                 # σ(w)
w_tilde = (rho / w_std) * w_centered     # WSP vector, Eq.(5)

# Use w_tilde for forward computation:
y_l[m] = w_tilde^T @ x_l                 # Eq.(3)

# WS Backward (gradient filtering, automatic via autograd):
# ∂L/∂w = (rho/σ) * (I - P_1) * (I - P_{w̃}) * ∂L/∂w̃    # Eq.(6)
# Step 1: (I - P_{w̃}) removes component aligned with w̃
# Step 2: (I - P_1) removes mean component
# Result: gradient projected onto span{w̃, 1}^⊥
# Only directions orthogonal to both w̃ and 1 survive
```

**Annotations**: ρ 为超参数（FedWSQ 默认 ρ=0.001），控制标准化后的参数 scale。w̃ 满足 zero-mean 和 ρ-scaled unit variance。两个投影矩阵 P_1 和 P_{w̃} 的连续作用等价于将梯度投影到 span{w̃, 1}^⊥。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WS 作为 plug-and-play 技术，可直接插入任何 CNN/MLP 架构的卷积层或全连接层前。在 PyTorch 中可通过自定义 `weight_standardization` 函数实现，在 `forward()` 中对每层权重调用标准化后再执行 F.conv2d 或 F.linear。WS 通常与 GroupNorm (GN) 搭配（替换 BatchNorm），因为 BN 的 batch 统计在 FL 中不可靠（各 client 数据量不同）。推荐在 GN 之前应用 WS。超参数 ρ 不敏感（FedWSQ 实验表明 1e-4 到 1e-1 范围内准确率变化 <3%），因为 inference 时 normalization 层会消除常数 scale 的影响。在 FL 中，FedWSQ 传输 PSP 而非 WSP——这一设计选择使 client 隐式通过梯度过滤受益于 WS 的稳定性，同时保留 client-specific 的本地参数信息，避免 FedWon (传输 WSP) 导致的信息丢失。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

## Distribution-Aware Non-Uniform Quantization (DANUQ / 分布感知非均匀量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DANUQ (Distribution-Aware Non-Uniform Quantization) 是 FedWSQ 提出的新型非均匀量化方法，专为联邦学习中 LMPU (Local Model Parameter Update) 的通信压缩设计。与需要学习量化参数（scale, zero-point）的传统方法不同，DANUQ 使用**固定的预计算量化级别（QLs）**，无需每轮传输额外的量化参数。核心设计分两层：(1) **Scaling**：用 LMPU 的标准差 σ 而非 absmax 作为 scale factor，因为 σ 对 outlier 更稳健且与 N(0,1) 假设一致。Global scaling vector s_g 通过 EMA 在 server 端跨 client 聚合并广播，保证量化一致性。(2) **QLs 设计**：假设归一化后 LMPU ∼ N(0,1)，最小化期望量化误差 E[(x - q)^2] = Σ_{r=0}^R ∫_{u_r}^{u_{r+1}} (x - q_r)^2 φ(x) dx，通过暴力搜索在离散化搜索空间中枚举所有可能 QLs 组合，找到使该目标最小的 {q_1, ..., q_R}。预计算的最优 QLs 为：1-bit: [-0.798, 0.798]；2-bit: [-1.224, 0, 0.765, 1.724]；4-bit: 16 个非均匀间隔值。DANUQ 不需要 backpropagation 学习量化参数，因此在 FL 场景中不引入额外的通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DANUQ 量化 LMPU 的具体流程（Client-side）：

```python
# Input: ΔW_i ∈ R^P (full-precision LMPU for layer l)
#        s_{g,l} (global scale for layer l, received from server)
#        QLs = [q_0, q_1, ..., q_R] (pre-computed optimal quantization levels for B-bit)

# Step 1: Normalize by global scale
ΔW_norm = ΔW_i / s_{g,l}    # assume ~ N(0,1) after normalization

# Step 2: Build quantization boundaries
u_0 = 0                      # fixed q_0 = 0 for B >= 2
for r = 1 to R:
    u_r = (q_{r-1} + q_r) / 2   # boundary between q_{r-1} and q_r
u_{R+1} = +inf

# Step 3: Non-uniform quantization by nearest-neighbor mapping
# For the symmetric half [0, +inf):
for each element x in abs(ΔW_norm):
    if x < u_1:
        idx = 0
    elif u_1 <= x < u_2:
        idx = 1
    ...
    elif u_R <= x:
        idx = R
    q_x = sign(x) * QLs[idx]   # restore sign

# For 1-bit (special case: omit q_0=0 constraint):
# QLs = [-0.798, 0.798], boundary at 0
# idx = 0 if x < 0 else 1, q_x = QLs[idx]

# Step 4: Build output
ΔW̄_i = q_x_values                # B-bit indices into QL lookup table
s_i = std(ΔW_i)                  # local scale (transmitted alongside)
return (ΔW̄_i, s_i)
```

**Annotations**: QLs 在 [0, +∞) 非均匀分布：密集区域（均值附近 ∼0）分配更细粒度 QLs，稀疏区域（尾部 ∼2.5+）分配更粗粒度。1-bit 特例省略 q_0=0 约束以允许两个 QLs 对称放置（≈ ±0.798）。Dequantization 在 server 端通过查表 + 乘 s_{g,l} 恢复 full-precision。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DANUQ 的实现分两阶段：(1) **离线预计算 QLs**：给定 B-bit，在经验合理的搜索范围内（如 [-3, 3] for N(0,1)），将 QL 空间离散化为候选网格，枚举所有满足排序约束 q_0 < q_1 < ... < q_R 的组合，评估 Eq.(10) 的目标函数，选最小化的配置。使用并行处理加速。QLs 计算一次后便固定，所有 client 和所有通信轮次复用。(2) **在线量化/反量化**：Client 端将归一化的 LMPU 值通过二分查找或阈值比较映射到最近 QL，传输 QL 索引（B-bit 整数）和 per-layer scale s_i（float 向量）。Server 端通过 QL 查找表 + 乘 scale 恢复全精度值。与 FedPAQ (uniform quantization + absmax + 概率舍入) 相比，DANUQ 的主要优势是：(a) "标准差 scale"代替 "absmax"避免了 outlier 导致的大部分值 underflow；(b) "非均匀 QLs"在概率密度高的区域提供更高精度。在使用上，DANUQ 与 WS 协同工作：WS 稳定训练过程使 LMPU 统计更接近正态分布，DANUQ 利用这一分布特性进行高效压缩。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

## Client Drift (客户端漂移) in Federated Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Client Drift（客户端漂移）是联邦学习中的核心挑战之一。当各 client 的数据分布为非 i.i.d.（non-identically and independently distributed）时，每个 client 在 local training 中优化的本地损失函数 F_i(w) 与全局损失函数 F(w) = Σ h_i F_i(w) 存在系统性偏差。经过 K 步 local SGD 后，各 client 的 Local Model Parameter (LMP) w_i 向各自局部最优方向偏离，而非向全局最优收敛。Server 聚合这些"漂移"的 LMPU Δw_i 后，得到的 Global Model Parameter (GMP) w_g 远离真正全局最优。形式化：E[Δw_i] 的方差在 non-i.i.d. 设置下显著增大，导致聚合后梯度的有效信噪比降低。FedWSQ 从梯度过滤视角分析 client drift：local gradient ∂L/∂w 包含 (a) 与当前 LMP w̃ 对齐的分量——模型过拟合本地数据的方向，(b) mini-batch 梯度均值分量——biased toward local data distribution。WS 通过双重投影 (I-P_1)(I-P_{w̃}) 过滤掉这两个分量，使梯度仅保留对全局收敛有益的方向。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Client drift 在 FedAvg 中的具体表现（无缓解措施的 baseline）：

```python
# Client i local training at round t
w_i = w_g^{t-1}                    # initialize from GMP
for k = 1 to K:                    # K local SGD steps
    batch = sample(D_i)            # sample from local (non-i.i.d.) data
    g_i = ∇f_i(w_i; batch)         # local gradient
    w_i = w_i - η * g_i            # SGD update
Δw_i = w_i - w_g^{t-1}             # LMPU

# Drift analysis:
# After K steps: w_i ≈ w_g - η Σ_k g_i^{(k)}
# Each g_i^{(k)} is biased: E_Di[g_i] ≠ E_D[g] (global gradient)
# Accumulation: ||w_i - w*|| >> ||w_g - w*|| (deviation from true optimum)
# Aggregation: w_g = Σ h_i w_i → still deviates from w*
```

**Annotations**: Drift 程度 ∝ K（local steps 越多，漂移越严重）× α（Dirichlet 参数越小/数据异质性越高，漂移越严重）。缓解方法：FedProx（近端项约束）、SCAFFOLD（control variate 修正）、FedDyn（动态正则化）、FedWSQ（WS 梯度过滤）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Client drift 是 FL 几乎所有改进方法的 motivation。检测 drift 的方法：监控各 client LMPU 之间的 cosine similarity、梯度方差、或 Hessian top eigenvalue（FedWSQ 使用 loss landscape 分析，较低的 Hessian eigenvalue 表示更平滑的收敛）。缓解 drift 的主流方法分四类：(1) **近端约束**（FedProx）：在 local loss 中加 μ/2·||w - w_g||² 项限制偏离幅度；(2) **Control variate**（SCAFFOLD）：在各 client 维护 control variate c_i 修正梯度方向；(3) **动态正则化**（FedDyn/ACG）：每轮调整正则化强度以对齐 local 和 global 目标；(4) **梯度过滤**（FedWSQ）：通过 WS 的前向/反向投影，在每步 local SGD 中隐式过滤导致 drift 的梯度分量，无需修改 loss 函数或维护额外状态。FedWSQ 的独特之处在于将 drift 缓解与量化通信压缩结合在一个统一框架中。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

## Local Model Parameter Updates (LMPUs / 本地模型参数更新量)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LMPUs (Local Model Parameter Updates) 是联邦学习中的核心通信单元。在每轮通信 t 中，client i 经过 K 步 local training 后，计算 LMPU 为 Δw_i^t = w_i^t - w_g^{t-1}，即本地训练后的 LMP (Local Model Parameter) 与上一轮 GMP (Global Model Parameter) 的差值（向量）。LMPU 包含两个角色：(1) 携带 client i 从本地数据 D_i 中学到的梯度信息——保持 data locality；(2) 作为 uplink 通信的主要 payload——每个 client 将 LMPU 传输到 server，server 聚合所有 LMPU 更新 GMP。在量化 FL 中，LMPU 是量化压缩的目标张量。FedWSQ 利用 LMPU 的一个关键统计性质：由于 LMPU 是模型参数的差值，在大规模网络中其逐元素值近似服从正态分布 ∼ N(0, σ²)，其中 σ 随层次和训练阶段变化。DANUQ 正是基于这一正态性假设来设计 QLs。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LMPU 在 FedAvg/FedWSQ pipeline 中的角色：

```python
# Round t, Client i:
# 1. Receive GMP and (optionally) global scale
w_g = receive_from_server()       # GMP: shape matches model
s_g = receive_from_server()       # global scale vector (FedWSQ only)

# 2. Local training (K steps with WS if FedWSQ)
w_i = w_g.clone()
for k in range(K):
    w_i = local_sgd_step(w_i, D_i)    # WS applied in forward if FedWSQ

# 3. Compute LMPU (the communication payload)
Δw_i = w_i - w_g                 # LMPU: same shape as model

# 4. Quantize LMPU (FedWSQ/DANUQ)
Δw̄_i = danuq_quantize(Δw_i / s_g)   # B-bit per element
s_i = compute_std_per_layer(Δw_i)   # local scale vector

# 5. Upload (Δw̄_i, s_i) to server
upload_to_server(Δw̄_i, s_i)

# Server-side:
# 6. Aggregate LMPUs
Δ = Σ h_i * dequantize(Δw̄_i, s_i)   # weighted sum
w_g = w_g + Δ                         # GMP update
```

**Annotations**: LMPU 的大小 = 模型参数量（如 ResNet-18 约 11M）。不压缩时 uplink 通信 = 4 bytes × 11M = 44MB/client/round。4-bit DANUQ 压缩后 = 0.5 bytes × 11M + scale overhead = ∼5.5MB。LMPU 的正态性假设是 DANUQ 设计的理论基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMPU 是 FL 算法的标准抽象——所有 FL 方法都需要传输某种形式的模型更新。实现上，LMPU 是 client 在 local training 结束后用当前 LMP 减初始 GMP 得到的一个与模型同形的张量集合（每层一个 tensor）。在 PyTorch FL 实现中，通常用 `[p.clone() for p in model.parameters()]` 保存初始 GMP，训练后用 `[p - gmp for p, gmp in zip(model.parameters(), gmp)]` 计算 LMPU。量化 LMPU 时，FedPAQ 使用 absmax + uniform quantizer + 概率舍入，FedWSQ 使用 σ-scaling + DANUQ non-uniform quantizer。关键区别：FedWSQ 仅传输量化后的 LMPU (B-bit integers) 和 per-layer scale (1 float/layer)，无需传输额外的量化参数（zero point, 学习到的 scale 等）。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

## Global Scaling Vector (全局缩放向量) for Federated Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Global Scaling Vector (s_g) 是 FedWSQ 用于在 federated 设置中协调各 client 量化一致性的机制。s_g = [s_{g,1}, ..., s_{g,L}]^T 是一个 L 维向量，其中每个元素 s_{g,l} 是第 l 层 LMPU 的全局 scale factor。更新方式：每轮通信后，server 收集各 client 的 local scale vectors s_i（每层 LMPU 的标准差），通过 EMA 更新 s_g ← (1-β)s_g + β·(1/|S|)·Σ_{i∈S} s_i，其中 β 为 momentum 参数（默认 0.1）。Client 在量化 LMPU 前从 server 获取 s_g，将 LMPU 各层张量除以对应的 s_{g,l} 实现归一化。与 FedPAQ 的 per-tensor absmax scaling 不同，global scaling vector 提供了三层优势：(1) per-layer scaling 更精细地匹配各层 LMPU 量级差异；(2) EMA 聚合使 scale 平滑更新，抵抗 client 间波动；(3) 各 client 使用统一的 global scale，确保量化边界一致，server 端 dequantization 信息不丢失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Global scaling vector 的更新和使用流程：

```python
# Server-side: s_g update (after LMPU aggregation)
# Input: s_i for i ∈ S_t (local scale vectors, each shape [L])
s_mean = (1/|S_t|) * sum(s_i for i in S_t)  # element-wise mean
s_g = (1 - beta) * s_g + beta * s_mean       # EMA update, beta=0.1

# Client-side: using s_g for DANUQ quantization
# Input: ΔW_i (LMPU, L tensors), s_g (global scale vector)
for l in range(L):
    ΔW_norm_l = ΔW_i[l] / s_g[l]          # normalize layer l
    ΔW̄_i[l] = DANUQ_quantize(ΔW_norm_l)    # quantize to B-bit
    s_i[l] = std(ΔW_i[l])                  # compute local scale
return (ΔW̄_i, s_i)

# Server-side: dequantization using transmitted scales
for l in range(L):
    Δ_i[l] = DANUQ_dequantize(ΔW̄_i[l]) * s_i[l]  # restore full-precision
```

**Annotations**: s_g 的维度 L = 模型层数（如 ResNet-18 约 20 层卷积+FC，每层一个 scale）。s_i 以 float32 传输，overhead = L × 4 bytes ≈ 80 bytes（可忽略不计）。β=0.1 使 s_g 在约 10 轮内适应分布变化。初始化 s_g^0 通常设为 1.0 或从第一轮 local training 计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Global scaling vector 是 FedWSQ 设计的协作量化机制。实现上，s_g 在 server 端维护为 PyTorch buffer（形状 [L]），每轮在聚合 LMPU 后更新。s_g 对训练稳定性至关重要：如果各 client 使用独立的 local scale（如 FedPAQ），相同数值在不同 client 的量化中可能映射到不同 QL，导致 server 聚合时信息破坏。s_g 通过 EMA 平滑更新既保证一致性又允许随时间适应分布变化。β 的选择需平衡响应速度与稳定性——过大的 β 会过度受单轮 client 子集影响，过小则无法跟踪分布漂移。FedWSQ 实验显示 β=0.1 在多数场景下表现良好。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

## Quantization Levels for Normal Distribution (正态分布最优量化级别)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Levels (QLs) for Normal Distribution 是 DANUQ 通过数值优化预计算的一组最优离散值，用于最小化标准正态分布下的期望量化误差。给定 B-bit 表示和 QLs 集合 Q = {q_0, q_1, ..., q_R}（其中 R = 2^{B-1} - 1，q_0 = 0），量化边界为 u_r = (q_{r-1} + q_r)/2（1 ≤ r ≤ R），量化规则 x ∈ [u_r, u_{r+1}) → q_r。最优 QLs 通过最小化 E[(x-q)^2] = Σ_{r=0}^R ∫_{u_r}^{u_{r+1}} (x - q_r)^2 φ(x) dx 求解（φ 为标准正态 PDF）。因 closed-form 解不可得（含 erf 和高斯积分交互项），DANUQ 采用暴力搜索在离散化搜索空间中枚举所有满足排序约束的 QL 组合。预计算结果：1-bit[-0.798, 0.798]；2-bit[-1.224, 0, 0.765, 1.724]；4-bit[16个非均匀值]。QLs 非均匀分布反映了正态分布的高密度区域（均值附近）需要更细粒度量化的原则。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QLs 的数值优化过程：

```python
# Offline QL optimization (run once before FL training)
def optimize_QLs(B, search_range=(-3, 3), grid_points=1000):
    R = 2**(B-1) - 1
    candidates = linspace(search_range[0], search_range[1], grid_points)
    
    best_QLs = None
    best_error = inf
    
    # Exhaustive search over discretized QL space
    # Constraint: 0 = q_0 < q_1 < q_2 < ... < q_R
    for q_combo in combinations_with_order(candidates, R):
        QLs = [0.0] + list(q_combo)   # q_0 = 0 fixed (except 1-bit)
        # Build boundaries
        u = [0.0]  # u_0 = q_0
        for r in range(1, R+1):
            u.append((QLs[r-1] + QLs[r]) / 2)
        u.append(float('inf'))  # u_{R+1}
        
        # Evaluate expected error via Eq.(10)
        error = compute_expected_error(QLs, u)
        if error < best_error:
            best_error = error
            best_QLs = QLs
    
    return best_QLs
# For 1-bit special case: omit q_0=0, search symmetric pair directly
```

**Annotations**: search_range 限制在 [-3, 3] 因为 N(0,1) 的 99.7% 概率质量在 [-3σ, 3σ] 内。grid_points=1000 足够细粒度。计算量在 B≤4 时可控（4-bit: C(1000,15) 太大，实际用启发式 + 并行加速）。1-bit 特例：只有 2 个 QLs，无 q_0=0 约束，直接搜索对称对。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QLs 预计算后以 lookup table 形式存入代码常量。Client 端量化：对归一化后的值 x，查找最近 QL：|x - q_r| 最小 → 返回 B-bit 索引 r。Server 端反量化：通过 索引 r → QLs[r] → 乘 scale 恢复。DANUQ 的 QLs 设计的优势：(1) 固定 QLs 无需学习或传输，零额外通信；(2) 基于正态分布先验，在 LMPU 真实分布匹配假设时达到近似最优；(3) 与 UQ 的等距间隔相比，DANUQ 的 1-bit QLs (±0.798) 在实际正态分布下可压缩 MSE 约 18%-25%（FedWSQ Table 2 数据估计）。对比现有 NUQ 方法：NF (NormalFloat) 对各 bit-width 使用通用公式，DANUQ 为每个 B 独立优化；FP (Floating Point) 的 exponent/mantissa 分配不适合 1/2-bit 极端场景。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization

## Gated Retention (gRet / gRetNet / RetNet-3)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gated Retention（gRet，也称 gRetNet 或 RetNet-3）是 Retentive Network (RetNet) 的增强版本，由 YOCO 论文提出作为默认的高效自注意力模块。它在 Retention 的基础上引入**数据依赖的门控机制 (data-dependent gating)**，使 decay rate 不再是固定值而是由输入数据动态决定：γ_n = sigmoid(X_n W_γ)^{1/τ}，其中 τ 是温度参数鼓励 γ 趋向 1 以获得更好的记忆能力。门控是 head-wise 的（而非 element-wise），使计算可以充分利用 NVIDIA Tensor Core。核心创新在于统一了三种等价的计算范式：(a) Parallel 模式用于训练（充分利用 GPU 并行）；(b) Recurrent 模式用于自回归推理（O(1) 常量 KV 内存）；(c) Chunkwise Recurrent 模式用于 prefill（chunk 内并行 + chunk 间 recurrent，节省 FLOPs 并减少迭代次数）。这三种范式在数学上等价，输出结果相同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 YOCO Self-Decoder 中使用 gated retention 的自回归推理（recurrent mode）为例：

```python
# Recurrent Gated Retention (decode mode, single head)
# Input: x_n ∈ R^d (current token), S_{n-1} ∈ R^{d×d} (previous state)
# Weights: W_Q, W_K, W_V ∈ R^{d×d}, W_γ ∈ R^{d×1}
# Hyperparams: τ (temperature), θ (RoPE base)

Q_n = (x_n @ W_Q) * Θ_n          # Θ_n = e^{inθ}, RoPE applied
K_n = (x_n @ W_K) * Θ̄_n          # Θ̄_n = e^{-inθ}, conjugate RoPE
V_n = x_n @ W_V
γ_n = sigmoid(x_n @ W_γ)^{1/τ}   # data-dependent gate, head-wise

# State update: O(d²)
S_n = γ_n * S_{n-1} + K_n^T @ V_n   # outer product K_n^T · V_n ∈ R^{d×d}

# Output: O(d²)  
O_n = Q_n @ S_n                   # vector-matrix product
O_n = GroupNorm_h(O_n)            # per-head GroupNorm
O_n = swish(x_n @ W_G) * O_n      # swish gate for non-linearity
O_n = O_n @ W_O                   # output projection
```

**Annotations**: d = head_dim (e.g., 128 for YOCO-3B)。S_n 是唯一的中间状态（O(d²) = 128² = 16K floats），不存储 per-token KV cache。γ_n ∈ (0,1) 控制信息保留程度——趋近 1 时接近等权保留（更好记忆），趋近 0 时快速遗忘。温度 τ 默认较大（如 8），将 sigmoid 输出推向 1。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Gated Retention 的实现分为三种模式在不同阶段使用：(1) **训练阶段**使用 Parallel 模式，QK^T⊙D（causal decay mask）再乘 V，与标准 self-attention 接口兼容，可直接替换；(2) **Prefill 阶段**使用 Chunkwise Recurrent（chunk_size=256），chunk 内用 parallel 计算利用 Tensor Core，chunk 间通过 recurrent state R 传递信息；(3) **Decode 阶段**使用 Recurrent 模式，仅维护 S ∈ R^{d×d} 状态矩阵，每 token 更新为 O(d²) 计算量。实现基于 Triton kernel（FLA 库：https://github.com/sustcsonglin/flash-linear-attention）。数据依赖门控 γ 为 head-wise 而非 element-wise，使 decay mask D 形成低秩结构，可利用 Tensor Core 高效计算。与标准 self-attention 的 O(N²d) 内存不同，gated retention 推理时仅需 O(d²) 常量内存。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)

## Decoder-Decoder Architecture (YOCO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Decoder-Decoder Architecture（YOCO，You Only Cache Once）是微软提出的用于 LLM 的新型解码器架构，替代传统 decoder-only Transformer。核心思想是将 L 层均分为两部分：前 L/2 层为 **Self-Decoder**（使用高效自注意力如 gated retention 或 sliding-window attention，仅需 O(1) 常量 KV cache），后 L/2 层为 **Cross-Decoder**（通过 cross-attention 复用 Self-Decoder 最终输出生成的**单一全局 KV cache** K̂, V̂）。该架构的核心优势：(1) KV cache 总量从 O(L×N×D) 降至 O(N×D)，约节省 L 倍 GPU 内存；(2) Prefill 阶段可在 Self-Decoder 完成后提前退出（early exit），因 Cross-Decoder 仅依赖 K̂, V̂，prefill 延迟降至一半以下；(3) Cross-Decoder 的 KV cache 仅需一次 all-gather（分布式训练），而非每层一次。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
YOCO 推理 pipeline（L=26 层，L/2=13，3B model, 512K context）：

```
Input: x[1..N] → Embedding X^0 ∈ R^{N×d}

# === Self-Decoder (layers 1..13) ===
for l in 1..13:
    # Efficient Self-Attention (gated retention, recurrent mode for decode)
    Y^l = ESA(LN(X^{l-1})) + X^{l-1}   # ESA: GatedRetention or SlidingWindowAttn
    X^l = SwiGLU(LN(Y^l)) + Y^l

# === Generate Global KV Cache (once!) ===
M = X^13                              # Self-Decoder's final output
K̂ = LN(M) @ W_K                      # single global key cache
V̂ = LN(M) @ W_V                      # single global value cache

# === Prefill Early Exit: STOP HERE during prefill! ===

# === Cross-Decoder (layers 14..26) ===
for l in 14..26:
    Q̂^l = LN(X^{l-1}) @ W_Q^l        # per-layer query projection
    Y^l = CrossAttention(Q̂^l, K̂, V̂) + X^{l-1}  # all layers share K̂, V̂
    X^l = SwiGLU(LN(Y^l)) + Y^l

# Output
logits = softmax(X^26 @ W_lm_head)
```

**Annotations**: Prefill 时仅执行 Self-Decoder 的前向（13 层而非 26 层），然后生成 K̂,V̂ 并退出。Decode 时 Self-Decoder 用 recurrent 模式（O(1) state），Cross-Decoder 标准 attention 复用 K̂,V̂。KV cache 存储：仅 K̂,V̂（单层 N×d×2）+ Self-Decoder 的常量状态（如 gated retention 的 S ∈ R^{d×d}）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
YOCO 适用于以下场景：(1) 长上下文 LLM 推理——KV cache 内存从 O(LND) 降至 O(ND)，使 1M token 上下文可在 consumer GPU 上部署（3B: 12.4GB）；(2) 低延迟 prefill——early exit 机制将 512K prefill 从 180s 降至 <6s；(3) 分布式长序列训练——Chunk Parallelism 减少通信频率。实现代码开源：https://aka.ms/YOCO。Self-decoder 可选用不同的高效 attention 模块（gated retention 或 sliding-window attention），Cross-decoder 兼容 GQA 进一步节省 KV cache。架构与标准 Transformer 训练流程兼容（使用相同的 AdamW 优化器、SwiGLU FFN、RMSNorm 等），可从标准 Transformer 的预训练设置迁移。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)

## Sliding-Window Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sliding-Window Attention（滑动窗口注意力）是 Child et al. (2019) 在 Sparse Transformers 中提出的稀疏注意力机制。与标准 self-attention 中每个 query 关注所有历史 token 不同，sliding-window attention 限制每个 query 仅关注前 C 个 token（窗口大小 C 为常量），使用 window causal mask B：`B_{ij}=0 if i-C<j≤i else -∞`。这使得 KV cache 内存复杂度从 O(N) 降至 O(C)，即内存使用量是常量，不随序列长度增长。YOCO 将其作为 Self-Decoder 的备选高效自注意力模块（与 gated retention 并列），利用其 O(1) 推理内存的特性来降低整体 KV cache 开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sliding-window attention 在 YOCO Self-Decoder 中的计算过程：

```python
# Sliding-Window Self-Attention (YOCO Self-Decoder)
# Input: X ∈ R^{N×d}, window_size C (e.g., 1024)
# Weights: W_Q, W_K, W_V, W_O ∈ R^{d×d}

Q, K, V = X@W_Q, X@W_K, X@W_V  # shape: [N, d]

# Window causal mask: each query i attends to keys in [i-C+1, i]
B = zeros(N, N)
for i in range(N):
    for j in range(N):
        if i - C < j <= i:       # within window
            B[i,j] = 0
        else:                     # outside window
            B[i,j] = -inf

# Multi-head computation
for h in range(num_heads):
    scores = Q_h @ K_h^T / sqrt(d_head)   # [N, N]
    scores = softmax(scores + B)           # masked softmax
    head_h = scores @ V_h                  # [N, d_head]

# Output projection
Y = concat(head_1, ..., head_H) @ W_O
```

**Annotations**: 推理时仅缓存每个 head 的最近 C 个 token 的 K, V（而非全部 N），cache size = C × d_head × H_kv × layers。对于 YOCO Self-Decoder，C=1024。Prefill 阶段 window attention 的计算复杂度为 O(N×C×d)，而非 O(N²×d)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sliding-window attention 常用于：(1) 长文档建模——通过限制 attention range 获得线性级别的计算复杂度；(2) 作为高效 attention 模块嵌入混合架构——如 YOCO 的 Self-Decoder 或 Jamba 的 hybrid Mamba-Transformer；(3) 推理优化——固定窗口使得 KV cache 大小与序列长度解耦，适合流式/实时应用。实现时可使用 FlashAttention 的 windowed attention kernel 或自定义 Triton kernel。主要限制：窗口外信息完全丢失（缺乏全局 attention），在 YOCO 中由 Cross-Decoder 的全局 cross-attention 弥补；对于需要全局上下文的任务单独使用时可能精度不足。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
- Hymba: A Hybrid-head Architecture for Small Language Models

## Shared Global KV Cache (Single-Layer Cross-Attention KV Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Global KV Cache 是 YOCO 的核心创新组件。在传统 decoder-only Transformer 中，每层 decoder 独立计算并存储自己的 KV cache。YOCO 将全局 KV cache 的生成和使用分离：Self-Decoder 的最终输出 M=X^{L/2} 经过一次线性变换生成全局的 K̂=LN(M)W_K 和 V̂=LN(M)W_V，然后所有 Cross-Decoder 层（L/2 层）的 cross-attention 共享这组 KV cache。这意味着 KV cache 总量从 O(L×N×d) 降至 O(N×d+L×C×d) ≈ O(N×d)（C 为 Self-Decoder 的常量 memory），约节省 L 倍。全局 KV cache 与 GQA 兼容（K̂,V̂ 使用较少的 KV heads），可进一步压缩。该设计将"缓存一次"（cache once）的概念实体化为架构组件。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Shared Global KV Cache 的生成和使用流程：

```python
# === Generation (once per sequence) ===
M = SelfDecoder.forward(X_0)              # output of last self-decoder layer
M_norm = RMSNorm(M)                       # pre-normalization
K_hat = M_norm @ W_K                      # global key cache, [N, d_k]
V_hat = M_norm @ W_V                      # global value cache, [N, d_v]

# === Cross-Decoder Usage (all L/2 layers reuse) ===
for l in range(L//2):
    X_l_norm = RMSNorm(X_l)
    Q_l = X_l_norm @ W_Q_l                # per-layer fresh query
    # Standard cross-attention with SHARED K_hat, V_hat
    A = softmax(Q_l @ K_hat.T / sqrt(d_k) + causal_mask)  
    O = A @ V_hat
    X_{l+1} = SwiGLU(RMSNorm(O + X_l)) + (O + X_l)
```

**Annotations**: K̂,V̂ 的维度：d_k = d_v = d_head × h_kv（GQA 时可减少 h_kv）。causal_mask 确保 cross-attention 也是因果的（t 位置只能关注 ≤t 的全局 context）。K̂,V̂ 的生成仅需一次 O(N×d²) 的矩阵乘法，相比 L 层的 KV cache 生成节省 L×计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Shared Global KV Cache 的实现和使用场景：(1) 长上下文推理——单层 cache 使 65B 模型 128K token 仅需 1GB KV cache（vs Transformer 需数十 GB）；(2) Pre-caching for RAG——可以预先计算并缓存文档的 K̂,V̂，查询时直接复用；(3) 与 GQA 结合——减少 K̂,V̂ 的 head 数进一步节省（YOCO-3B 使用 h_kv=8 vs h_q=24）；(4) 分布式推理——K̂,V̂ 可以存储在一台 GPU 上，Cross-Decoder 分布到多 GPU 各自计算 Q_l。限制：K̂,V̂ 本质上是 Self-Decoder 的输出投影，其表达能力受限于 Self-Decoder 的质量和层数。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)

## Prefill Early Exit

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prefill Early Exit 是 YOCO 利用其 decoder-decoder 架构的计算依赖特性实现的 prefill 加速策略。在标准 Transformer 中，prefill 必须执行全部 L 层的前向计算。YOCO 的关键洞察是：Cross-Decoder 的 cross-attention 仅依赖 Self-Decoder 的输出 K̂,V̂，而 K̂,V̂ 在 Self-Decoder 完成后即可计算。因此 prefill 阶段只需执行 Self-Decoder（L/2 层）+ 生成 K̂,V̂，然后**提前退出**，无需执行 Cross-Decoder。生成的第一个 token 仍然是正确的，因为 Cross-Decoder 在 decode 阶段会逐步执行。这一特性质来自架构设计而非工程优化。结合 Self-Decoder 的高效 attention（线性复杂度），prefill 延迟从 O(LN²d) 降至 O(LNd/2)，在 512K context 上从 180s 降至 <6s。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Prefill Early Exit 的执行流程对比：

```
=== Standard Transformer Prefill === 
Input: x[1..N] tokens
for l in 1..L:
    X^l = DecoderLayer_l(X^{l-1})     # ALL L layers
    cache K^l, V^l for decode
Return: X^L (used to predict 1st token)
Latency: O(L * N² * d)  ← quadratic in N

=== YOCO Prefill with Early Exit ===
Input: x[1..N] tokens
for l in 1..L/2:
    X^l = SelfDecoderLayer_l(X^{l-1})  # ONLY L/2 layers
K̂, V̂ = proj_KV(X^{L/2})                # generate global KV cache
# === EXIT HERE ===
# Cross-Decoder NOT executed during prefill
# First token prediction uses Cross-Decoder in decode phase
Latency: O(L/2 * N * d) for gated retention  ← linear in N
         or O(L/2 * N * C * d) for sliding-window
```

**Annotations**: Prefill Early Exit 是安全的（sound），因为 Cross-Decoder 第 1 个 token 的 cross-attention 仅依赖位置 1 的全局 K̂,V̂（causal mask），而 K̂,V̂ 的位置 1 已在 Self-Decoder prefill 中完全计算。因此 prefill 退出时生成的第一个 token 与执行完整 prefill 的结果完全相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Prefill Early Exit 适用于：(1) 长上下文首次 token 延迟（TTFT）优化——1M context 从 380s 降至约 5s；(2) 流式场景——快速响应用户的第一个 token；(3) 批处理 prefill——减少 per-request 的 prefill 计算量使 batch size 增大。实现上只需在推理引擎的 prefill 阶段跳过 Cross-Decoder layers。限制：仅适用于 YOCO 类架构（Self-Decoder 输出可直接生成全局 KV cache）；对短上下文加速比相对较小（32K 时约 2.87×）；decode 阶段仍需执行 Cross-Decoder（但仅对单 token 做 cross-attention，开销较小）。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)

## Post-Training Quantization (PTQ)

## Post-Training Quantization (PTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-Training Quantization（PTQ，后训练量化）是一种在模型完成常规浮点训练后将预训练模型的高精度参数（FP32/FP16）转换为低比特整数表示（INT8/INT4/INT2等）的模型压缩技术。其核心流程为：(1) 加载已训练好的 FP32 模型权重；(2) 使用少量校准数据（通常无需标签）统计各层权重和激活的数值分布；(3) 为每层确定量化参数（scale factor s, zero point z, clipping bounds [l, u]）；(4) 执行量化：将浮点值 v 映射为离散整数值 v_int = round((clip(v,l,u) - l) / s)，其中 s = (u-l)/(2^N-1)；(5) 推理时使用整数算术替代浮点运算，实现存储压缩和计算加速。与 QAT 的核心区别在于 PTQ 不修改模型权重本身（仅确定量化器的 clip bounds），无需重新训练，只需分钟级别校准时间，适合训练资源受限或快速部署场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 2DQuant 论文的 PTQ pipeline 为例，量化一个 Transformer-based SR 模型的流程如下：
```
# Stage 1: 离线量化参数搜索
for each layer in model:
    w = layer.weight  # FP32 权重
    a = calibrate_activations(layer, calibration_images)
    if is_symmetric(w):  # 对称钟形分布
        l_best, u_best = symmetric_mse_search(w, K=100)
    else:  # 非对称指数分布
        l_best, u_best = asymmetric_mse_search(a, K=100)
    quantizer[layer] = (l_best, u_best)

# Stage 2: 蒸馏微调
for iter in range(3000):
    x = next_batch(calibration_set)
    o_fp = fp_model(x)           # 教师输出 (FP32)
    o_q = quantized_model(x)     # 学生输出 (INT4 算术)
    loss = L1(o_fp, o_q) + λ * feature_L2(f_fp, f_q)
    loss.backward()  # STE 通过量化操作回传梯度
    update_clip_bounds()  # 仅更新 l, u，不更新权重

# 部署推理
convert_all_linear_to_INT_arithmetic(model)
```
量化后的 Linear 层计算：`y = INT_MATMUL(W_int, x_int) * s_w * s_x`，其中 `s_w`, `s_x` 为 scale factors。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PTQ 的通用实现方式：(1) 使用 PyTorch 或 TensorFlow 的量化 API（如 torch.quantization、torch.fake_quantize）；(2) 通过校准数据统计 min/max 或使用 MSE/熵最小化搜索 clip bounds；(3) 可选使用 AdaRound、GPTQ、AWQ 等高级方法优化 rounding 策略；(4) 将模型导出为量化格式（如 ONNX QInt8、TensorRT INT8 engine）。在 PyTorch 中，基本用法为：`torch.quantization.prepare(model, inplace=True)` → 校准 → `torch.quantization.convert(model, inplace=True)`。现代 PTQ 方法（如 GPTQ、AWQ）可直接通过 pip 包使用，支持 HuggingFace 模型的 one-shot 量化。AffineQuant 进一步将 PTQ 的等价变换从缩放/平移扩展到完整的仿射变换（d² 维优化空间），通过 Gradual Mask 保持矩阵可逆性。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- AFPQ Asymmetric Floating Point Quantization for LLMs
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- AffineQuant Affine Transformation Quantization for Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs
- Bridging the Gap Between Promise and Performance for FP4 Quantization
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization
- DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization
- GPTVQ: The Blessing of Dimensionality for LLM Quantization
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement
- PT²-LLM Post-Training Ternarization for Large Language Models
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
- QT-DoG Quantization-Aware Training for Domain Generalization
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning
- Training Dynamics Impact Post-Training Quantization Robustness

S²Q-VDiT 将 PTQ 范式扩展到视频扩散模型（V-DMs）领域。V-DMs 的 token 序列极长（n=s×t vs I-DMs 的 n=s），导致两个关键挑战：(1) 校准预算受限（仅几十样本 vs I-DMs 的数千样本）下随机采样方差极大；(2) 长序列中所有 token 均等处理的量化损失优化效率低。S²Q-VDiT 提出 Hessian-aware Salient Data Selection (SDS)——联合扩散信息量 C_diff = ||x_t-x_{t-1}||²/||x_t||² 和量化敏感度 C_quant = ||x_t^T x_t||_2（Levenberg-Marquardt Hessian 近似）的乘积得分筛选校准数据——和 Attention-guided Sparse Token Distillation (STD)——通过多头注意力图 A 计算 token-wise 重要性权重 λ_j，重加权量化损失 L_quant = (1/n)Σλ_j||θ^f(x_j)-θ^q(x_j)||²。在 CogVideoX-2B/5B 和 HunyuanVideo-13B 上 W4A6 几乎无损、W4A4 维持 95% 性能，首次探索 V-DMs 的 4-bit 激活量化。部署基于 ViDiT-Q 和 FlatQuant 的 CUDA kernel，CogVideoX-5B 实现 3.94× 模型压缩、1.56× 推理显存节省、1.28× 推理加速。代码: https://github.com/wlfeng0509/s2q-vdit。

SynQ (ICLR 2025) 在 PTQ 场景下与 Genie (Jeon et al., 2023b) 集成：SynQ 的低通滤波、CAM 对齐和困难样本软标签策略应用于 Genie 生成的合成数据集，在 ResNet-18 ImageNet W2A2/W2A4/W3A3/W4A4 上平均提升 0.66%p 准确率，证明其 synthesis-aware fine-tuning 兼容 PTQ 范式下的 clipping/rounding 优化。

Q-resafe 对 PTQ 的安全评估：Q-resafe (Chen et al., ICML 2025) 首次系统评估了主流 PTQ 方法（AWQ、AQLM）在多种校准数据集和不同位宽下的安全影响。关键发现：(1) 所有 PTQ 方法都会损害安全——INT4 下 ASR 从 FP16 的 0.3% 升至 18.5-42.4%；(2) 有微调的 PTQ（AQLM）在 benign 数据集上更安全（ASR=18.5%），但在有害数据集上风险急剧放大（ASR=77.4%）；(3) 无微调的 PTQ（AWQ）安全退化中等（ASR=42.4%），但无法通过数据集选择控制风险；(4) 低 bit-width 显著加剧安全退化——2-bit 下 QLoRA ASR=82.0%。Q-resafe 提出的安全修补方法可在不损效用前提下将 PTQ 后 ASR 恢复至接近 FP16 水平。

PassionSR 将 PTQ 范式首次扩展到 one-step diffusion (OSD) 图像超分模型。其 PTQ 方法的核心差异在于：(1) 同时量化 UNet 和 VAE（传统多步扩散量化仅量化 UNet），因为 OSD 模型中去噪步数减至 1 后 VAE 占据 80%+ 计算量；(2) 通过可学习边界量化器（LBQ）以梯度下降替代传统 fixed-range 量化搜索；(3) 通过等效变换（LET）抑制激活离群值后量化；(4) 采用两阶段分布式标定（DQC）稳定训练，以 PTQ 级效率（1.07h 标定时间）实现接近 QAT 的量化精度。W8A8 下参数压缩 81.77%、操作减少 76.56%，W6A6 下参数压缩 86.32%、操作减少 82.42%。

ParoQuant 在 W4A16 PTQ 中引入了 Scaled Pairwise Rotation——结合 channel-wise scaling 和独立 Givens 旋转——通过 AdamW 梯度下降优化旋转角度和缩放因子（而非 AWQ 的 grid search），并在推理时通过 fused CUDA kernel 应用逆变换（~10% 开销）。两阶段逐层优化可学习变换参数，专门针对推理 LLM 长链思维生成中的误差累积问题。

PT²-LLM 将 PTQ 范式扩展到三值量化（Ternarization）领域。传统 PTQ 方法处理 2-8 bit 均匀量化，而 PT²-LLM 实现了仅 3 个量化级别（{−1,0,+1}，等效 1.58-bit）的 post-training 场景。其 ATQ 通过两阶段无训练优化（ITF 交替优化 + AGA 激活感知对齐）将三值参数在无梯度反传的情况下精细优化，是 PTQ 范式在极端低比特三值场景的首次系统探索。

QT-DoG 从域泛化（DG）角度揭示了 PTQ 的关键局限：PTQ (OBC, Frantar et al. 2022) 在 PACS 上无法提升 DG 性能（OBC 83.7% vs 无量化 ERM 84.7%），因为 PTQ 仅做推理时压缩而无训练过程，无法通过量化噪声引导模型搜索平坦极小值。这验证了 QT-DoG 的核心主张——只有 QAT（训练时注入量化噪声）才能通过 Hessian 交互机制找到平坦极小值，PTQ 的单纯推理时量化不具备此正则化效果。

Q-VDiT 将 PTQ 范式首次扩展到视频 Diffusion Transformer (V-DiT) 领域。视频 DiT 相比图像 DiT 有更高的 token 信息密度（n=s×t vs n=s），直接应用图像量化方法的 PTQ 导致剧烈性能退化。Q-VDiT 的 PTQ 创新包括：(1) TQE (Token-aware Quantization Estimator)——利用 H(Δ)≤H(W) 的信息论性质，在 token 和 feature 维度使用 rank=1 低秩参数估计和补偿量化误差，参数开销仅 d_in+d_out+t；(2) TMD (Temporal Maintenance Distillation)——在传统 MSE 重建损失上增加帧间时序分布 KL 散度对齐项，使每帧优化受整体视频分布共同引导。W3A6 下 Scene Consistency 翻倍（12.04→23.40），W4A6 几乎无损。校准用时 12.5-12.9 小时（W8A8），推理时通过 LoRunner Kernel 融合 TQE 低秩分支，额外延迟<5%。

## Round-to-Nearest (RTN) Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Round-to-Nearest（RTN）量化是最朴素的权重量化方法，也称为 Naive Rounding 或 Nearest Rounding。其过程为：对每个权重值 w，计算 w/scale 得到浮点索引，然后用 round() 函数将其映射到最近的整数索引（即最近的量化级别），反量化时乘以 scale 恢复。RTN 无需校准数据，不进行任何优化，计算开销极小。在 GPTQ 和 AWQ 等高级 PTQ 方法出现之前，RTN 是 LLM 量化的默认 baseline。AFPQ 论文在 RTN 基础上验证了非对称 FP 量化的有效性——表 1、表 2 的 RTN 实验对比了 INT4/NF4-sym/NF4-asym 等格式在不同 group-size 下的 WikiText-2 ppl 和 MMLU 精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AFPQ 论文中 RTN 量化（对称 FP 版本）的计算过程（Algorithm 1: FPSYMQuant/FPSYMDequant）：
```
def FPSYMQuant(weight_tensor, group_size, range):
    for each group of group_size weights:
        w_max = max(group)
        w_min = min(group)
        scale = max(w_max, abs(w_min)) / (range / 2)
        w_4bit = round(weight / scale)
    return w_4bit, scale

def FPSYMDequant(w_4bit, scale):
    w_deq = scale * w_4bit
    return w_deq
```
AFPQ 的非对称版本（FPASYMQuant/FPASYMDequant）在此基础上为正值和负值使用不同的 scale：
```
scale_pos = w_max / (range / 2)
scale_neg = -w_min / (range / 2)
w_deq = scale_pos * w_4bit_pos + scale_neg * w_4bit_neg
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RTN 量化通常在 PyTorch 中自定义实现。简单流程：(1) 分组 weight tensor（按 group_size 或 per-tensor）；(2) 计算每组的 scale 参数（可选 zero_point）；(3) `torch.round(weight / scale)` 得到量化索引；(4) 存储量化索引和 scale。在 bitsandbytes 库中，RTN 量化被用作 baseline，NF4 格式的量化即默认使用 RTN。在 HuggingFace transformers 中可通过 `load_in_4bit=True` 结合 `BitsAndBytesConfig` 使用。KIVI 中使用 group-wise RTN 量化 KV cache，对 key cache 沿 channel 维度、value cache 沿 token 维度分别分组计算 scale/zero-point 后 round-to-nearest。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## Logarithmic Quantization (对数量化, Logarithmic PTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Logarithmic Quantization（对数量化）是一种非线性后训练量化方法，其量化级别呈指数间隔排列（在对数域中对应均匀步长）。与线性量化（量化级别均匀分布）不同，对数量化在数值较小时提供密集的量化级别，数值较大时提供稀疏的量化级别，天然匹配深度神经网络中常见的钟形（bell-shaped）和长尾权重分布。LOGART 论文系统比较了三种对数量化变体：
- **Log2**：量化级别为 2 的幂次值 {..., 2^{-3}, 2^{-2}, 2^{-1}, 2^0, 2^1, ...}。硬件优势是可完全用移位器替代乘法器（乘以 2^Q 等价于左移 Q 位），但量化台阶粗糙，大值附近精度不足。
- **Log√2**：量化级别为 √2 的幂次值，粒度更细。代价是乘法涉及 √2 不能直接用移位器实现，需 LUT 或 shift-add 近似。
- **DLog (Dynamic Log)**：混合基方案，大值用 base-√2（细粒度），小值用 base-2（硬件友好），通过对数域阈值 t 分割两个区域。

LOGART 的对数量化核心公式（基-2）：
- Quant: Q_W = clamp(⌊-log_2(|W|/s)⌋, 0, 2^{N-1}-1)，其中 s = 2^{⌊log_2(max(|W|))⌉}
- Dequant: Ŵ = s · sign(W) ⊙ 2^{-Q_W}

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LOGART 论文中对数量化在 PTQ pipeline 中的位置：
```
# 对数量化推理流程 (Log2, weight-only)
输入: W ∈ R^{d_out × d_in}  (FP16 weight), X ∈ R^{d_in × seq}  (激活)
# Step 1: 计算 scale
s = 2^{round(log_2(max(|W|)))}  # per-channel 或 per-tensor
# Step 2: 量化 weight
Q_W = clamp(floor(-log_2(|W| / s)), 0, 2^{N-1} - 1)  # N-bit 整数码字
# Step 3: 推理 (使用移位器替代乘法器)
# 因为 Ŵ_{ij} = s · sign(W_{ij}) · 2^{-Q_W_{ij}}
# 所以 ŴX 的第 i 行 = Σ_j s · sign(W_{ij}) · X_j >> Q_W_{ij}  (右移)
output = shift_accumulate(X, Q_W, s)  # 无乘法器
```
对数量化在超低比特（3-4bit）下通常优于线性量化的原因是其非均匀量化级别能更好地保留大幅值权重的精度，而这些权重对模型输出影响更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对数量化可直接在 PyTorch 中实现：用 `torch.log2()` 转换到对数域 → `torch.floor()` 或 `torch.round()` 量化 → `torch.pow(2, ...)` 反量化。硬件实现上，Log2 量化推理时乘法器被移位器替代，Log√2 需额外的 shift-add 近似或 LUT 处理。LOGART 开源代码（https://github.com/logart-lab/logart）提供了完整的对数量化实现，支持 Calibration-based PTQ，可与 HuggingFace Transformers 模型集成。现有对数量化方法包括 LogNet (Lee et al. 2017), FQ-ViT (Lin et al. 2022), SLogII/Xu et al. (2023), 以及 LOGART。

I&S-ViT 在此基础上提出 SULQ (Shift-Uniform-Log2 Quantizer)：在 log2 变换前引入 shift bias η（X_q = UQ(-log₂(X+η), b)），后接均匀量化器。这解决了标准 log2 量化器在 post-Softmax 激活上的"量化低效"问题——大量值被 clamp 到远端量化级别。SULQ 仅增加一次 round 和两次加法，推理时仍通过 bit-shift 执行。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---

## GPTQ (GPT Post-Training Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPTQ 是 Frantar et al. (2022) 提出的基于 Optimal Brain Surgeon (OBS) 框架的 LLM 后训练量化方法。核心思想：将权重量化问题转化为逐列压缩问题——对权重矩阵的每一列执行量化后，使用二阶 Hessian 信息（从校准数据中估计）计算未量化列的误差补偿，将已量化列引入的误差"吸收"到尚未量化的权重中。这样量化一列时产生的信息损失，通过调整剩余列来补偿。GPTQ 的计算复杂度为 O(d_row * d_col^2)，通过 lazy batch 更新和 Cholesky reformulation 实现 GPU 加速。在 AFPQ 论文中，GPTQ 被用作对比 baseline（GPTQ-INT3），同时 AFPQ 也将 NF3-asym 格式集成到 GPTQ 中以替换 INT 量化步骤（GPTQ-NF3-asym），在 LLaMA2-70B 上 WikiText-2 ppl 从 3.77 降至 3.66。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GPTQ 逐列量化流程：
```
# 输入: 权重矩阵 W (d_row × d_col)
H = X^T X  # 从校准数据的输入 X 估计 Hessian，size: d_col × d_col
H_inv = Cholesky(H^-1)  # Cholesky 分解逆 Hessian
for col in range(d_col):
    w_col = W[:, col]  # 当前列
    w_q = quantize(w_col)  # 量化当前列 (RTN)
    error = (w_col - w_q) / H_inv[col, col]  # 量化误差
    # 补偿剩余列
    for j in range(col+1, d_col):
        W[:, j] += error * H_inv[col, j]
```
AFPQ 集成 GPTQ 时，将 `quantize(w_col)` 从 INT-asym 替换为 NF3-asym 量化，其余 Hessian 补偿逻辑保持不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTQ 官方实现：https://github.com/IST-DASLab/gptq。AutoGPTQ（https://github.com/PanQiWei/AutoGPTQ）是最广泛使用的社区版本，提供 Python API 和命令行工具，支持 HuggingFace transformers 模型。基本使用：`from auto_gptq import AutoGPTQForCausalLM; model = AutoGPTQForCausalLM.from_quantized(model_name, use_safetensors=True)`。GPTQ 支持 2/3/4/8 bit 位宽，可结合 group_size 参数调节精度-速度权衡。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

aespa 论文使用 OPTQ（即 GPTQ 的别名）在 weight-rounding 优化前初始化整数权重 W_int，以经验性减少 AdaRound 优化所需的迭代轮数。OPTQ 的 Hessian-based 逐列补偿机制与 aespa 的预计算 Hessian 策略共享相似的理论基础（均源自 OBS/OBC 框架），但 aespa 关键差别在于 Hessian 不再固定为 H=2E[XX^T] 而是按 attention 层定制化（H_V=2E[XA^TAX^T]）。
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- Bridging the Gap Between Promise and Performance for FP4 Quantization
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

Quamba2 在 W4A8 和 W4A16 配置中对所有投影层应用 GPTQ（Frantar et al. 2023）优化 4-bit 权重的量化精度。GPTQ 在 Quamba2 中与 per-group weight quantization 和 Hadamard transform 结合：W4A16 消融显示 PerG + GPTQ + Had. 达到 71.2% LAMBADA（与 FP16 持平，70.9%），而单独 PerG 仅 64.7%。
- First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
- GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration
- Training Dynamics Impact Post-Training Quantization Robustness
- GPTVQ: The Blessing of Dimensionality for LLM Quantization
- LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
- PB-LLM Partially Binarized Large Language Models
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
GPTVQ 将 GPTQ 从均匀/标量非均匀量化扩展到向量量化（VQ）：核心修改包括 (1) 逐 d 维向量量化替代逐标量量化——每次量化 d 列，误差沿 d 维累积后一次性补偿；(2) EM 初始化 codebook——用 Hessian 加权的马氏距离（E-step）+ 伪逆闭式解（M-step）；(3) Codebook update——量化完成后用梯度下降 fine-tune codebook 值。GPTVQ 使 2D VQ 在移动 CPU 上实现比 INT4 更小的 footprint（-19%）和更高的 token rate（+10%）。

GPTAQ 揭示 GPTQ 的**对称校准（Symmetric Calibration）**缺陷：GPTQ 每层独立最小化 `||(w+Δw)X − wX||²`，其中 X 来自前一层的量化输出。但前层量化误差使得 X ≠ X̃（全精度模型的输入激活），该偏差沿网络深度累积。GPTAQ 将此问题建模为**非对称校准（Asymmetric Calibration）**：最小化 `||(w+Δw)X − wX̃||²`，其中 X̃ 是前一层全精度模型的输出。这引入了一个额外的残差补偿项 `W_{:,q} ΔX_{q,:} X_{:,q:}^T H_{-q}^{-1}`，在 GPTQ 的量化误差补偿之上显式补偿前层累积的激活偏差。GPTAQ 通过残差分解（R = Σ W_{:,q} ΔX_{q,:}）避免重复计算 R，并利用 Cholesky 重构化（Theorem 4.2：P = ((ΔX X^T L) ⊙ M_U) L^T）实现 GPU 并行计算 P 矩阵。实现仅比 GPTQ 多约 20 行代码，额外延迟大维度时 30-40%、小维度时 <10%。

MR-GPTQ 将 GPTQ 适配到 FP4 微缩放格式：核心修改包括 (1) MSE-optimized grids——交替优化 per-tensor scale 和 per-group scales 最小化整体 MSE；(2) Static activation reordering——先确定 grid/scales 再重排列列、量化后恢复原序，避免 dynamic act-order 的 10-20% 推理开销；(3) Block-wise Hadamard rotations——对 MXFP4 (G=32) 旋转降低 per-element MSE，旋转融合入权重零推理开销。

FOEM 揭示 GPTQ 省略一阶项的假设在逐列量化中不成立：先量化列的补偿项累积使 latent weights 偏离 FP 权重，产生非零一阶梯度。FOEM 通过 g≈β(W−𝕎)H 近似一阶梯度，代入 Lagrangian 求解后 Hessian 自动消去，仅增加 O(n) 权重差分，量化时间几乎不变（+0.4%）而精度显著提升。

PB-LLM 将 GPTQ 扩展到**部分二值化**场景（PB-GPTQ）：在每列量化时，unsalient 权重二值化（sign + α scaling），salient 权重 INT8 量化（MinMax），然后将该列的二值化+量化误差通过 Hessian 补偿到剩余列。PB-GPTQ 显著优于 RTN（Round-to-Nearest）：10% salient 时 PB-GPTQ Hessian 将 C4 PPL 从 7508（RTN Hessian）降至 165，证明 Hessian 引导补偿在二值化场景同样关键。PB-GPTQ 也验证了 group-wise（g=128）变体，发现 group-wise 缩放因子的差异不是二值化性能的主要决定因素。

PT²-LLM 将 GPTQ 的逐块量化框架扩展为**后训练三值量化**引擎。PT²-LLM 保留 GPTQ 的 block-wise 误差补偿（Hessian 逆引导的逐列补偿），但将列选取策略从 Hessian 重要性排序替换为 SSR（Structural Similarity-based Reordering）——每次基于残差矩阵列间余弦相似度选取 top-k 最相似列组成量化块。量化步骤中，GPTQ 的 RTN/uniform quantization 被替换为 ATQ（Asymmetric Ternary Quantizer）的两阶段三值优化：ITF（Iterative Ternary Fitting，交替闭式求解 α*,μ* 和弹性舍入更新 T）→ AGA（Activation-aware Grid Alignment，利用校准数据协方差矩阵 C=ΣXXᵀ 闭式更新 α*,μ*）。误差补偿公式 E = (W - (αT+μ)) / H_chol_inv 保持 GPTQ 原有形式，仅去量化值从 uniform 变为 ternary。

QA-LoRA (Xu et al., 2023) 使用 GPTQ 作为量化 backbone，配置为 group-wise asymmetric INT 量化（group_size=32, act-order=false, true-sequential=true），量化后权重交给 QA-LoRA 的 group-wise LoRA 适配器微调，微调后仅更新零点矩阵保持 INT 格式。QA-LoRA 提出方法框架与 GPTQ 解耦，也兼容 AWQ、SPQR 等其他 PTQ 方法。

SliM-LLM 以 GPTQ 作为 backbone 框架，在其逐列量化+OBQ误差补偿流程中嵌入两个新模块：(1) SBA 在GPTQ量化前通过双指针搜索为每个group分配最优bit-width；(2) SQC 在GPTQ的fakequant步骤中增强对salient权重的感知，使用经τ优化的scale/zero替换默认量化器参数。SliM-LLM保留了GPTQ的block-wise Hessian逆和逐列补偿逻辑，仅扩展了量化器的精度多样性。SliM-LLM⁺变体则以OmniQuant为backbone，在OmniQuant的learnable weight clipping之上叠加SBA。

SpQR 将 GPTQ 扩展为支持**非结构化异常值检测与双层量化**的混合表示：关键修改包括 (1) 在GPTQ逐列量化过程中嵌入outlier检测——每处理β₁列前，通过leave-one-out敏感度准则判断每个权重是否为outlier（E_base − E_ol > τ），outlier权重跳过量化保留为16-bit，且outlier排除后的min-max scale显著减小；(2) 使用极小group size（β₁=8-16），但将group-wise统计量（scales/zeros）本身做第二层GPTQ量化（3-bit），使得统计量存储开销可控；(3) outlier检测发生在量化过程中而非预处理阶段，捕捉的不仅是"初始敏感"权重，还包括量化过程中因误差累积而变得敏感的权重。SpQR将GPTQ从单纯量化扩展为sparse-quantized混合格式。

S²Q-VDiT 以 GPTQ weight quantizer 作为量化 backbone，在其 block-wise 逐列量化流程中嵌入两个新机制：(1) SDS 在 GPTQ 校准前通过 Hessian 感知的 salient 得分筛选校准数据（替代随机采样）；(2) STD 在每 block 的 MSE 损失中引入 token-wise attention 重加权 λ_j，使高影响力 token 获得更大学优化权重。GPTQ 的 Hessian 逆和逐列误差补偿逻辑保持不变。校准在单卡 A800 上完成，W4A4 CogVideoX-2B 耗时 2.88h、显存 35.68GB。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models
- AFPQ Asymmetric Floating Point Quantization for LLMs
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---

## QA-LoRA (Quantization-Aware Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QA-LoRA（Quantization-Aware Low-Rank Adaptation）由 Xu et al. (2023, Huawei) 提出，是一种将量化感知引入 LoRA 低秩适配的联合微调-部署方法。核心创新在于引入分组操作（group-wise operators）来平衡量化与适应的自由度（degrees of freedom）：一方面增加量化的自由度（每列分为 L 组，每组独立 α_{l,j}、β_{l,j}，替代 per-column 量化），另一方面降低适应的自由度（输入 x 经组内求和聚合，LoRA 矩阵 A 维度从 D_in×D_int 缩减为 L×D_int，L << D_in）。QA-LoRA 解决了 QLoRA 的核心痛点：(1) QLoRA 微调后合并权重恢复为 FP16，推理时若需 INT4 则需额外 PTQ 导致精度损失；(2) QA-LoRA 通过数学变换，将 LoRA 适配器权重仅合并到零点矩阵 β' = β - s·(BA)⊘α，保持 Ŵ 和 α 不变，使合并后模型仍为 INT 格式，无需 PTQ 直接 INT 推理。因此 QA-LoRA 同时获得微调效率（INT 格式训练）和推理效率（INT 格式部署），在 INT2 极端低位宽下优势尤为显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# QA-LoRA 核心 Pipeline (基于 LLaMA-7B, INT4, group_size=32)

# Step 1: GPTQ 分组量化（离线，一次性）
W = [4096, 4096]  # FP16 weight, D_in=D_out=4096
g = 32            # group size
L = D_in // g = 128  # number of groups
# 对每列 j 的每 l 组:
for j in range(D_out):
    for l in range(L):
        w_group = W[l*g:(l+1)*g, j]
        α[l, j] = (max(w_group) - min(w_group)) / (2^N - 1)
        β[l, j] = min(w_group)
        W_hat[l*g:(l+1)*g, j] = round((w_group - β[l,j]) / α[l,j])
# 量化权重: W_tilde = α * (W_hat - β)  (group-wise 反量化形式)

# Step 2: QA-LoRA 初始化
# A = Parameter(L, D_int)  — 相比 QLoRA 的 (D_in, D_int) 减少 D_in/L = 32 倍
# B = Parameter(D_int, D_out)
QA = AvgPool1d(g)  # 组内求和聚合: 4096 → 128

# Step 3: 微调前向
def forward(x, W_tilde, A, B):
    # x: [batch, D_in]
    y_base = x @ W_tilde.T                     # INT4 矩阵乘
    x_agg = QA(x) * g                          # [batch, D_in] → [batch, L] 组内求和
    y_lora = (x_agg @ A.T) @ B.T * s           # s 为 adapt 系数
    return y_base + y_lora

# Step 4: 合并推理（无损 INT 格式）
# 仅更新零点矩阵 β:
β_new[l, j] = β[l, j] - s * (B @ A)[j, l] / α[l, j]
# W_hat 和 α 不变，合并后模型仍为 INT 格式，直接 INT 推理
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/yuhuixu1993/qa-lora。基于 HuggingFace Transformers + PEFT + GPTQ。核心修改：(1) 用 AvgPool1d（group_size=D_in/L）替换原始 linear 的输入，将 D_in 降维到 L；(2) 用分组量化（per-group α、β）替代 per-column 或 per-tensor 量化；(3) merge 时仅更新 β 矩阵。关键超参：L（组数）= D_in // group_size，常用 group_size=32（即 L=128 for 4096-d）；D_int（LoRA 中间维）。训练效率：INT4 算子由 CUDA 优化（vs QLoRA 的 NF4 无算子优化），训练时间比 QLoRA 减少 35-65%（LLaMA-7B: 21.5h vs 40.0h on V100）。推理效率：比 QLoRA FP16 推理快 >50%。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models

---

## NF4 (4-bit NormalFloat)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NF4（4-bit NormalFloat）是一种基于信息论最优化的 4-bit 量化数据类型，由 Dettmers et al. (2023) 在 QLoRA 中提出。其设计原理：假设预训练神经网络的权重服从零均值正态分布 N(0, σ²)，则最优的 4-bit 量化方案是将该分布的累积分布函数（CDF）的 2^N 个等概率分位点映射到对应的 4-bit 索引。具体地，NF4 的 16 个量化级别为 Q^NF4_map(q_i) = Φ⁻¹(i/(2^N+1)) = Φ⁻¹(i/17)，其中 Φ⁻¹ 为标准正态分布的分位函数。这 16 个值分别对应 -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0。实际使用时，权重按 block_size=64 分组，每组归一化到 [-1,1] 后映射到最近 NF4 级别。QA-LoRA 论文指出 NF4 缺乏 CUDA 算子级别优化，导致 QLoRA 的 NF4 微调速度慢于 QA-LoRA 的 INT4 微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# NF4 量化流程
NF4_LEVELS = [Φ⁻¹(i/17) for i in range(1, 17)]  # 16 values
# 对每个 block_size=64 的 block:
for block in W.reshape(-1, 64):
    s = absmax(block)           # block-wise scale
    W_norm = block / s          # 归一化到 [-1, 1]
    for each w in W_norm:
        idx = argmin |w - NF4_LEVELS[i]|  # 最近邻查找, 得 0-15 的 4-bit index
        W_q = idx                  # 存储为 4-bit index
    scales.append(s)
# 反量化: w_deq = s * NF4_LEVELS[idx]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NF4 通过 bitsandbytes 库（`bnb_4bit_quant_type="nf4"`）实现，配合 HuggingFace Transformers 使用。双重量化（Double Quantization）将 block-wise scale s 进一步以 FP8 量化（s_FP8）并保留 FP32 残差（s_FP32 = s - s_FP8），将 scale 存储从 0.5 bit/参数降至 0.127 bit/参数。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

QeRL (Huang et al., NVIDIA, 2025) 发现 NF4 在 RL 训练中存在严重性能瓶颈：NF4 反量化需通过 lookup table 将 4-bit index 映射回浮点值才能做矩阵乘法，使 QLoRA 的 rollout 速度比 BF16 LoRA 还慢 0.7-0.8×。此外 QeRL 实验显示 NF4 量化后 7B 模型 GSM8K 原始准确率下降 5.8 点（70.5% vs 76.3% BF16），RL 训练后恢复至 85.0%，仍低于 NVFP4+AQN 的 90.8%。

---

## Min-Max Quantization (Uniform Integer Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Min-Max Quantization 是最基础的均匀量化方法，将浮点权重 W 线性映射到 N-bit 整数 {0, 1, ..., 2^N-1}。公式：W̃ = α·⌊(W-β)/α⌉ + β，其中 α = (max(W) - min(W)) / (2^N-1) 为缩放因子，β = min(W) 为零点因子，⌊·⌉ 为整数舍入。这是 QA-LoRA 论文中用于 INT4/INT3/INT2 量化的基本公式。QA-LoRA 将其从 column-wise（α_j, β_j 对每列计算）扩展到 group-wise（α_{l,j}, β_{l,j} 对每组计算），通过增加量化参数数量降低量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Min-Max N-bit 量化
α = (max(W) - min(W)) / (2^N - 1)
β = min(W)
W_hat = round((W - β) / α)        # 量化: FP32 → INT (0 to 2^N-1)
W_tilde = α * W_hat + β           # 反量化: INT → FP32 (approximate W)
# 推理计算: y ≈ x @ W_tilde.T = α·(x @ W_hat.T) + β·sum(x)
# INT 矩阵乘 x @ W_hat.T 比 FP16 GEMM 更快且省显存
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Min-Max 量化是最广泛使用的量化 baseline，在 PyTorch 中通过 `torch.quantize_per_tensor()` 实现。GPTQ、bitsandbytes 等框架将其扩展为 group-wise 版本。QA-LoRA 使用 GPTQ 的 group-wise min-max 量化作为基础量化方法，INT4 支持由 CUDA 优化的 INT4 GEMM 算子加速。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---

## AWQ (Activation-Aware Weight Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AWQ（Activation-aware Weight Quantization）是 Lin et al. (MLSys 2024 Best Paper, MIT HAN Lab) 提出的硬件友好的 LLM 低比特 weight-only 后训练量化方法。核心发现：LLM 中权重并非同等重要——仅保护 0.1%-1% 的显著（salient）权重通道即可大幅降低量化误差（OPT-6.7B INT3-g128 PPL 从 43.2 降至 13.0）。关键洞察：要识别显著权重通道，应参考**激活分布**而非权重分布——激活幅度更大的通道对应的权重更重要（这些通道处理更重要的特征）。为避免硬件低效的混合精度实现（部分通道 FP16 + 部分 INT），AWQ 通过数学推导证明：对显著通道的权重乘以 s > 1，并对激活除以 s（等效变换），可以降低显著权重的相对量化误差（误差比例 `Δ'/Δ · 1/s < 1`，因为 `Δ' ≈ Δ` 且 `s > 1`）。Per-channel scale s 通过简洁的参数化搜索空间确定：`s = s_X^α`（s_X 为 per-channel 平均激活幅度，α ∈ [0,1] 通过 20 步网格搜索找到最优值），目标是最小化量化后输出与原始输出的 MSE。整个过程不依赖反向传播或 block-wise reconstruction，因此：(1) 仅需极少校准数据（16 条序列 vs GPTQ 的 192 条），(2) 不对校准集过拟合，可泛化到多模态 VLM、代码生成、数学推理等不同领域。可与 GPTQ 结合：AWQ per-channel scaling 作为前置步骤 → GPTQ 二阶误差补偿，在 INT2-g64 极端低比特下进一步提升性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B INT4-g128 量化为完整例子：

**数学原理（Eq. 1-3）：**
量化函数：`Q(w) = Δ · Round(w/Δ)`，其中 `Δ = max(|w|) / (2^{N-1} - 1)`。

对显著权重 w 乘以 s > 1，反向缩放激活 x：
```
Q(w·s) · (x/s) = Δ' · Round(w·s/Δ') · x · (1/s)
```
误差比例 = `Δ'/Δ · 1/s`。由于：(1) RoundErr ≈ 0.25（均匀分布），(2) 单元素缩放通常不改变组的 max → `Δ' ≈ Δ`，(3) s > 1 → 相对误差降低。但当 s 过大时，Δ' 会因 max 值改变而增大，导致 non-salient 通道误差放大。因此需通过网格搜索找到平衡点。

**完整算法 Pipeline：**
```python
# Step 1: 收集激活统计量
# 用 16 条 Pile 校准序列前向传播
for layer in model.layers:
    X = cached_input_activation[layer]     # [C_in, L]
    s_X = X.abs().mean(dim=1)              # [C_in], per-channel 平均激活幅度

# Step 2: 网格搜索最优 α
    best_alpha, best_loss = None, float('inf')
    for alpha in linspace(0, 1, 20):      # grid_size=20
        s = s_X ** alpha                   # per-channel scale

        W_scaled = W * s.unsqueeze(0)      # [C_out, C_in] * [1, C_in]
        W_q = groupwise_quantize(W_scaled, bits=4, group_size=128)
        # groupwise_quantize: 每 128 个元素一组，组内计算 Δ = max(|w_group|)/7
        #                    W_int = Round(W_scaled / Δ)

        X_scaled_inv = X / s.unsqueeze(1)  # [C_in, L]
        Y_q = dequantize(W_q) @ X_scaled_inv
        Y_fp = W @ X
        loss = MSE(Y_q, Y_fp)

        if loss < best_loss:
            best_alpha, best_loss = alpha, loss

# Step 3: 应用最优 scale 并量化
    s_final = s_X ** best_alpha
    W_final_q = groupwise_quantize(W * s_final.unsqueeze(0), bits=4, group_size=128)
    # diag(s)^{-1} 融合进前一层（LayerNorm weight 或前一层 Linear weight）

# Step 4 (可选): 与 GPTQ 结合
# 做完 AWQ per-channel scaling → 再做 GPTQ 逐列 Hessian 误差补偿
# INT2-g64 下 AWQ+GPTQ PPL: 15.71 vs GPTQ alone: 16.65 (OPT-6.7B)
```

**Table 1 关键数据（OPT-6.7B, INT3-g128, WikiText-2 PPL↓）：**
- FP16: 10.86
- RTN (全量 INT3): 23.54
- RTN + 1% FP16 (基于激活分布选): 11.39 ← 与 AWQ 全 INT3 精度相当
- RTN + 1% FP16 (基于权重分布选): 22.37 ← 无效
- AWQ (全 INT3, 用 per-channel scaling 保护显著通道): 11.92

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AWQ 官方开源：https://github.com/mit-han-lab/llm-awq (MIT License)。Python 使用：
```python
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model.quantize(tokenizer, quant_config={
    "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"
})
```
已被广泛集成到生产系统：
- **vLLM**: `vllm/model_executor/layers/quantization/awq.py`
- **HuggingFace Transformers**: `from transformers import AwqConfig`
- **NVIDIA TensorRT-LLM**: 原生支持 AWQ INT4 weight-only
- **LMDeploy**: TurboMind 引擎支持 AWQ 推理
- **Intel Neural Compressor**: 支持 AWQ 量化
- **llama.cpp**: 通过 GGUF 格式支持 AWQ

AWQ 覆盖的模型范围：LLaMA/Llama-2 (7B-70B)、OPT (1.3B-30B)、Mistral-7B、Mixtral-8x7B (MoE)、Falcon、MPT、StarCoder、Vicuna (指令微调)、OpenFlamingo-9B (VLM)、LLaVA-13B (VLM)、VILA-7B/13B (VLM)、CodeLlama-7B。首次实现多模态 LLM 的低比特量化。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- AFPQ Asymmetric Floating Point Quantization for LLMs
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

AWQ 的安全影响：Q-resafe 的安全评估发现，AWQ（PTQ w/o FT）INT4 量化后 Llama-2-7B-Chat 的 ASR 从 0.3%（FP16）飙升至 42.4%（decoding attack 下），Gemma-7B-Instruct 从 9.2% 升至 17.9%。由于 AWQ 不使用校准数据集微调，无法通过数据集选择控制安全风险，必须在量化后使用 Q-resafe 进行安全修补。Q-resafe 对 AWQ 采用不同的修补策略：在全精度模型上识别安全关键权重保留为 FP16，其余权重 AWQ INT4 量化——不执行 DPO 训练。修补后 ASR 降至 baseline+0.8%（Llama）和 baseline+0.4%（Gemma）。

---

## NormalFloat (NF4/NF3)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NormalFloat（NF）是 Dettmers et al. (2021) 设计的专为神经网络权重量化优化的数据类型。其核心思想：假设预训练神经网络权重近似服从正态分布 N(0, σ)，将量化级别按照正态分布的分位数等距划分，使得每个量化 bin 内的概率质量（权重数量）近似相等（信息论最优）。NF 格式的值在零附近密集（对应大部分权重集中区域），远离零则稀疏（对应长尾大权重）。NF4 有 16 个候选值，NF3 有 8 个候选值。具体值（来自 bitsandbytes）：NF4 = [-1, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1]；NF3 = [-1, -0.535, -0.247, 0, 0.183, 0.382, 0.623, 1]。AFPQ 论文中的 NF4-asym 和 NF3-asym 方法在标准 NF 格式基础上增加了非对称 scaling，显著提升了量化精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NF 格式的设计源于信息论：最优量化器应当使每个量化 bin 覆盖相等的概率质量（Max-Lloyd 量化器）。对于标准正态分布 N(0,1)，其 CDF 的反函数（分位数函数）直接给出最优量化级别的值：
```
def create_nf_levels(num_levels):
    # 对标准正态分布按等概率间隔取分位数
    probs = np.linspace(0, 1, num_levels + 1)
    # 取每个 bin 的中点（概率平均值处）的分位数
    mid_probs = (probs[:-1] + probs[1:]) / 2
    levels = norm.ppf(mid_probs)  # 正态分布分位数函数
    levels = levels / max(abs(levels))  # 归一化到 [-1, 1]
    return levels
```
AFPQ 的非对称版本在此之上，将正负值分别乘以不同 scale。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NF4 格式由 bitsandbytes 库（https://github.com/TimDettmers/bitsandbytes）首先实现并推广。使用方式：`from transformers import BitsAndBytesConfig; config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`。在 HuggingFace 生态中，NF4 是 `load_in_4bit=True` 的默认量化格式。在 AFPQ 论文中，NF3 也被用来进行 3-bit 量化实验，在 GPTQ/AWQ 中均使用 NF3-asym 替代 INT3。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---

## Fake Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fake Quantization（伪量化/模拟量化）是在训练或校准时模拟低比特精度量化效果的技术。其核心思想是：前向传播时对权重/激活执行完整的量化-反量化操作（quantization-dequantization），使数值被约束在离散的量化级别上；但张量本身仍保持浮点表示，而非真正存储为整数，因此被称为"伪"量化。数学公式为：`v_q = s * round(clip(v, l, u) / s)`，其中 s 为 scale factor。伪量化允许在 FP32 环境中模拟 INT 推理的精度损失，同时保持梯度可以通过 STE 回传。本论文（2DQuant）使用伪量化来模拟 Linear 层和 Batch MatMul 的 INT4/INT3/INT2 算术精度损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 2DQuant 的 DQC 阶段，对 SwinIR 中每个 Linear 层的权重和激活执行伪量化：
```
def fake_quantize(v, l, u, N):
    # N = bit数, 2^N-1 = 量化级别数
    v_c = clamp(v, min=l, max=u)
    scale = (u - l) / (2**N - 1)
    v_r = round((v_c - l) / scale)      # 离散整数值
    v_q = v_r * scale + l                # 反量化回浮点
    return v_q  # 输出仍是 FP32，但值被约束在离散级别

# 前向传播
w_q = fake_quantize(weight, l_w, u_w, bits=4)  # 4-bit 伪量化权重
x_q = fake_quantize(input, l_x, u_x, bits=4)   # 4-bit 伪量化激活
y = linear(w_q, x_q)  # 用伪量化后的值计算，模拟 INT4 精度
```
伪量化的关键效果：原本连续的 FP32 值被强制离散化到 2^N 个候选值上，导致信息损失。DOBI 阶段通过 MSE 搜索找到使 `||v - v_q||_2` 最小的 (l, u)，从而最小化这种信息损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中，伪量化可通过 `torch.quantization.FakeQuantize` 或自定义实现。`torch.fake_quantize_per_tensor_affine` 和 `torch.fake_quantize_per_channel_affine` 提供内置支持。典型用法：`torch.quantization.FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8)`。在 QAT/PTQ 训练中，伪量化模块被插入模型的计算图关键位置（如 Linear 之前和之后），前向时模拟量化，反向时通过 STE 让梯度绕过不可微的 round 操作。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

在 PMQ-VE 中，Fake Quantization 用于模拟多帧视频增强模型（RSTT、MIA、EMA-VFI）的 INT4/INT2 推理精度损失。量化公式：`x_clip = clamp(x, lb, ub)`, `Δ = (ub-lb)/(2^N-1)`, `x_int = round((x_clip-lb)/Δ)`, `x̂ = x_int·Δ + lb`。与 2DQuant 的单一 per-tensor 量化不同，PMQ-VE 为每帧独立使用不同的 (lb_i, ub_i)，量化后的 x̂ 仍保持 FP32 表示但值域被约束在 2^N 个离散级别上。

---

## Straight-Through Estimator (STE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Straight-Through Estimator（STE，直通估计器）是一种在神经网络训练中处理不可微操作（如量化中的 round、二值化中的 sign）的梯度估计技术。其核心思想是：在前向传播时正常执行不可微操作，但在反向传播时将该操作的梯度近似为单位矩阵（即"直通"）。具体而言：前向 `y = round(x)`，反向 `∂L/∂x ≈ ∂L/∂y * 1 = ∂L/∂y`。STE 最初由 Hinton 在 2012 年提出，后由 Courbariaux 等人（2016）在 Binary Neural Networks 中推广使用。在量化领域，STE 使得量化器的 clip bounds (l, u) 可以通过梯度下降进行优化，因为 round 操作的导数几乎处处为零，无法直接回传梯度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 2DQuant 的 DQC 阶段，STE 使梯度可以穿过伪量化操作回传到 clip bounds (l, u)：
```
# 前向（正常伪量化）
v_q = fake_quantize(v, l, u, N)  # 包含 round → 梯度在此断裂

# 反向（STE 近似）
# Eq.2 给出 ∂v_q/∂u 和 ∂v_q/∂l 的具体形式：
∂v_q/∂u = ∂v_c/∂u + (1/(2^n-1))*v_r - (v_c - l)/(u - l)
∂v_q/∂l = ∂v_c/∂l - (1/(2^n-1))*v_r + (v_c - l)/(u - l)
# 其中 ∂v_c/∂u = H(u-v), ∂v_c/∂l = H(l-v), H 为 Heaviside 阶跃函数
# round 项的导数被近似为 1（STE 核心假设）
```
实际代码实现中，PyTorch 通过 `detach()` 技巧或自定义 autograd Function 实现 STE：
```python
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, l, u):
        x_clipped = torch.clamp(x, l, u)
        x_int = torch.round((x_clipped - l) / scale)
        x_q = x_int * scale + l
        return x_q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None  # STE: 梯度直通
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
STE 在 PyTorch 中的典型实现方式：(1) 使用 `torch.autograd.Function` 自定义前向/反向逻辑；(2) 在反向函数中返回 `grad_output` 作为输入的梯度；(3) 对 clip 边界的梯度使用 Eq.2 的计算公式（如 2DQuant 的做法）。更简单的实现可以直接 `(x.round() - x).detach() + x`，利用 PyTorch 的计算图分离特性实现 STE。常见变体包括 Clipped STE（仅对量化范围内的值回传梯度）、ReLU STE（用 ReLU 导数替代 identity）。STE 的局限性在于它是有偏估计，但在实践中对深度网络效果良好。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- Binarized Diffusion Model for Image Super-Resolution
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- PB-LLM Partially Binarized Large Language Models
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models
- Scheduling Weight Transitions for Quantization-Aware Training

PB-LLM 使用 STE 处理二值化中 sign 函数的梯度断裂问题。在 BNN 训练中，前向 sign(W_F) 二值化，反向 Clipped STE：`∂L/∂x = ∂L/∂sign(x)` if |x| ≤ 1 else 0。PB-LLM 的 QAT 仅对 unsalient 权重的 FP latent 使用 STE，salient weights 冻结不参与梯度更新。

在 Squat 中，STE 用于 QAT 的反向传播，梯度穿过伪量化操作（round/clip）直通回传给权重和激活。Squat的对称逐层量化使用标准STE（梯度直通），同时熵损失L_E和分布损失L_D通过正常反向传播梯度。

在 BinaryDM 中，STE 应用于 EBB 的二值化操作：sign(w) 和 sign(w - σ_I*sign(w)) 在前向执行离散化，反向通过 STE 将 sign 导数近似为 1，使得 σ_I 和 σ_II 的梯度可以正常回传（式 7-8）。此外 LSQ 激活量化器同样通过 STE 回传 clip bound 梯度。

在 BI-DiffSR 中，STE 用于训练二值化扩散模型：Sign(·) 函数不可微，反向传播时 STE 将 Sign 的导数近似为 1（直通）。训练过程为常规 PyTorch 训练（非 PTQ），使用 L1 loss + Adam 优化器在 DIV2K+Flickr2K 上训练 1M iterations，STE 使得梯度可穿过所有 BI-Conv block 的 Sign 二值化操作。

在 QT-DoG 中，STE 用于 QAT 训练中梯度穿透过量化操作 round(clip(W/s, -Q_N, Q_P))。前向使用量化权重 W_q = round(clip(W/s, -Q_N, Q_P)) × s，反向通过 STE 将 round 的导数近似为 1，使梯度可回传至全精度权重 W 和可学习的 per-channel scaling factor s。QT-DoG 使用 LSQ 的 STE 梯度计算方式：步长 s 在量化范围内时 ∂Ŵ/∂s = ⌊W/s⌉ - W/s（STE 直通 round），权重 W 在量化范围内梯度为 1（STE 直通），超出范围时为 0（被 clamp 截断）。

在 SPR²Q 中，STE 用于 PQFR（Pre-Quantization Fine-tuning with Fused Rectifier）阶段的反向传播，使梯度穿过伪量化操作 Q_{a,b}(·) 回传到 rectifier 参数 (A_i, B_i) 和量化器裁剪界 (a, b)。具体来说：对于 low-rank rectifier 矩阵，梯度为 ∂L/∂A = B^T ∂L/∂W' 和 ∂L/∂B = ∂L/∂W' A^T（STE 近似 round 导数为 1）；对于可学习裁剪界 v∈{a,b}，梯度为 ∂L/∂v = ∂L/∂W_q' · ∂W_q'/∂v，其中 ∂W_q'/∂v 的 round 项通过 STE 直通。与 2DQuant 类似，SPR²Q 也联合优化 clip bounds 和额外参数（rectifier 低秩矩阵），但 SPR²Q 的区别在于优化的是 LoRA 风格的权重增量而非仅量化器参数。STE 在 Rectifier Group Training (RGT, 12K iterations) 和 Offline Static Routing Calibration (OSRC, 500 iterations) 两个阶段均被使用。

在 EfficientQAT 的 Block-AP 中，STE 采用 LSQ+ 的梯度计算方式，对量化公式 W_int = clamp(round(W/s) + z, 0, 2^N-1) 的三种参数分别计算梯度：(1) 步长s的梯度 ∂ŵ/∂s：当W_int在[0, 2^N-1]内时 ∂ŵ/∂s = ⌊W/s⌉ - W/s（round项STE为1），超出边界时退化为 -z 或 2^N-1-z（仅clamp贡献梯度）；(2) 零点z的梯度：在量化范围内为0（z在反量化中被抵消），超出范围时为 -1；(3) 权重W的梯度：在量化范围内为1（STE直通），超出范围时为0（被clamp截断）。

在 RoSTE 中，STE 用于 QA-SFT（量化感知监督微调）框架：前向时 `output = σ(Q_x(X R_i) · Q_w(R_i^T W_i))`（带旋转的伪量化 forward），反向时 STE 将 Q_w 的 Jacobian 近似为 `∂Q_w(R_i^T W_i)/∂W_i ≈ R_i`（即 rotation-aware STE），使得量化梯度直接通过旋转矩阵修正后回传。RoSTE 的 STE 更新规则：`w^{t+1} = w^t - η (⟨Q_x(Rx_t) | Q_w(Rw^t)⟩ - y_t) R^T Q_x(Rx_t)`，与标准 STE 的关键区别在于梯度中引入旋转矩阵 R，使得梯度方向与量化误差方向对齐。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- Binarized Diffusion Model for Image Super-Resolution
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- QT-DoG Quantization-Aware Training for Domain Generalization
- Scaling Law for Quantization-Aware Training
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 Scaling Law for QAT 中，STE 用于 268 次大规模 QAT 实验的反向传播：前向时 weight/activation 经 AbsMax/LAC 伪量化后再参与计算，反向时 STE 使梯度穿过 round/clamp 操作直通回传。论文中学习率实验证实 4-bit QAT 不敏感于 LR（[5e-4, 4e-3] 范围内 δ 近乎恒定），表明 STE 梯度估计在该场景足够稳定。

在 Scheduling Weight Transitions for QAT 中，STE 被用于标准 QAT 流程：前向 round/sign 离散化，反向 STE 将梯度近似为 1 直通回潜在权重。论文的贡献不在 STE 本身，而在于发现 STE 回传的梯度与用户设定 LR 结合时无法有效控制量化权重的 effective step size（因其由 transition 而非 LR 主导），进而提出 TR 调度 + TALR 来替代 LR 调度，但 STE 作为梯度估计器保持不变。论文使用修改版 LSQ quantizer（固定 post-scaling，不训练 weight scale parameter），STE 仍然用于通过 round 函数传播梯度。

在 PARQ 中，STE 被赋予了基于凸优化理论的严格解释：STE/BinaryConnect 的更新规则 `u^{t+1} = u^t - η_t ∇f(Q(u^t), z^t)`, `w^{t+1} = Q(u^{t+1})` 可被理解为 AProx (Aggregate Proximal) 算法的特例——当正则化函数 Ψ 取为量化集 Q 上的 indicator 函数 δ_Q 时，其 proximal map 就是硬量化映射 Q(·)，且该 proximal map 在任意缩放下不变。更重要的是，PARQ 证明 AProx 的 proximal map（prox_{γ_t λ Ψ}）在 γ_t → ∞ 时渐近收敛到硬量化，即 STE 可被视作 PARQ/AProx 的渐近形式。这为 STE 提供了严格的理论基础：它不仅是启发式近似，而是凸优化算法的渐近极限。

在 PMQ-VE 中，STE 被用于训练期间通过伪量化操作回传梯度，使得 BMFQ 搜索到的 per-frame clipping bounds (lb_i, ub_i) 在 PMTD 蒸馏阶段可通过梯度下降进一步微调。梯度链：∂L/∂x̂ → ∂x̂/∂lb_i（通过 STE 近似 round 导数为 1）→ 更新 lb_i。由于 PMQ-VE 的量化边界是 per-frame 的，每个帧独立接收 STE 梯度更新，而非共享梯度。

---

## Clipping Bound（量化裁剪边界）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Clipping Bound（裁剪边界）是量化器中定义数值截断范围的参数，通常包含下界 l 和上界 u。在量化公式 `v_q = s * round(clip(v, l, u) / s)` 中，所有小于 l 的值被映射为 l，所有大于 u 的值被映射为 u，只有 [l, u] 区间内的值保持正常的量化分辨率。裁剪边界的设置直接影响量化的信息保真度：边界太宽（如 [min, max]）会导致长尾分布中的离群值占据过多的量化区间，使得密集区域的表示精度不足（有效位利用率低）；边界太窄会截断过多信息，导致显著的信息丢失。2DQuant 论文发现 SwinIR 的权重和激活呈现"对称+非对称共存+长尾"的分布特征，需要针对每种分布类型采用不同的边界搜索策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 2DQuant 的 DOBI 阶段，clip bounds 的搜索过程：
```
Data: v (张量), K=100 (搜索点数), N (bit数)
Result: l_best, u_best
l ← min(v), u ← max(v)
min_mse ← +∞
if v 对称:    # 钟形分布 → 双界同时收缩
    Δl ← (max(v) - min(v)) / (2K)
else:         # 指数分布 → 固定下界不动
    Δl ← 0
Δu ← (max(v) - min(v)) / (2K)
for i in 0..K:
    l_i ← l + i*Δl, u_i ← u - i*Δu
    v_q ← fake_quantize(v, l_i, u_i, N)
    mse ← ||v - v_q||_2
    if mse < min_mse:
        min_mse ← mse; l_best ← l_i; u_best ← u_i
```
DQC 阶段进一步用梯度下降微调每个量化器的 (l, u)，优化目标从 MSE（值层面）转向任务目标（输出+特征蒸馏 loss）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Clipping bound 的确定方法主要有三类：(1) 统计方法——MinMax（取 min/max）、Percentile（取 p-百分位数）、MSE 最小化（在 min 到 max 之间搜索/优化）；(2) 训练方法——通过 STE 梯度回传直接学习最优边界值（如 PACT、LSQ、2DQuant 的 DQC）；(3) 分析/启发式方法——如基于分布的峰度/偏度自动调整。在 PyTorch 量化 API 中，Observer 模块（如 `MinMaxObserver`、`MovingAverageMinMaxObserver`、`HistogramObserver`）负责在校准阶段统计并确定 clip bounds。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

在 PMQ-VE 中，Clipping bound 被扩展为 per-frame 概念：对多帧激活张量 X∈R^{N×C×H×W}，每帧 X_i 拥有独立的 (lb_i, ub_i)，通过 BTBI（回溯搜索）在百分位约束的搜索空间 [p_{0.1}, p_{10}] × [p_{90}, p_{99.9}] 中找到使 ||X_i - X̂_i||_2 最小的边界。这与传统 per-tensor 量化中所有帧共享统一边界的做法不同——PMQ-VE 的逐帧边界可适配各帧特有的激活分布，解决了视频增强中帧间激活分布差异导致的量化次优问题。

---

## Knowledge Distillation for Quantization（蒸馏量化校准）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
蒸馏量化校准（Distillation Quantization Calibration, DQC）是 2DQuant 提出的将知识蒸馏应用于 PTQ 第二阶段的技术。其核心思想是：将全精度（FP）模型作为教师网络，将量化后的模型作为学生网络（两者结构完全相同），通过最小化教师和学生之间在输出层面和中间特征层面上的差异，来微调量化器的 clip bounds。这种做法将量化参数优化从"最小化数值偏移（MSE）"提升到"面向任务目标的优化"，能更有效地保持量化模型的感知质量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
2DQuant 的 DQC 阶段具体的损失函数与训练流程：
```
# 输出层蒸馏损失 (Eq.4)
L_O = (1/(C_O*H_O*W_O)) * ||O_fp - O_q||_1
# L1 Loss，促使量化模型的最终 SR 输出逼近 FP 模型

# 中间特征蒸馏损失 (Eq.5)
L_F = Σ_i (1/(C_i*H_i*W_i)) * ||F_i/||F_i||_2 - F_qi/||F_qi||_2||_2
# 对每层特征做 L2 归一化后计算 L2 距离，消除尺度差异影响

# 总损失 (Eq.6)
L = L_O + λ * L_F

# 训练配置：
optimizer = Adam(lr=1e-2, betas=(0.9, 0.999), weight_decay=0)
scheduler = CosineAnnealing
iterations = 3000
batch_size = 32
calibration_data = 32 random crops (3×64×64) from DF2K
```
关键设计：(1) 学生和教师共享相同网络结构，无需额外适配层；(2) L1 输出损失替代 L2（在 SR 任务中 L1 收敛性更好）；(3) 特征蒸馏使用 L2 归一化后的特征，消除量化引起的尺度变化干扰；(4) 仅更新量化器的 clip bounds（l, u），不修改模型权重本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
知识蒸馏用于量化的通用方式：(1) 输出蒸馏——最小化量化模型与 FP 模型最终输出的差异（L1/L2/KL 散度）；(2) 特征蒸馏——在中间层对齐特征表示，可使用 L2 距离、attention transfer 或 Gram 矩阵匹配；(3) Logit 蒸馏——对分类任务使用 soft targets（temperature-scaled softmax）。在 PyTorch 中实现类似 DQC 的蒸馏训练时，核心代码模式为：`loss = criterion(student_out, teacher_out.detach()) + lambda * feature_loss`，仅优化器和学生的量化参数被设为 requires_grad=True。

在 Task-Specific ZSQ for Object Detection 中，知识蒸馏被用于 QAT 阶段的三项联合蒸馏：(1) 预测匹配蒸馏 L_KD = (tau^2/N)*Sigma KL(z^F(x_i;theta), z^Q(x_i;theta'))，使用 KL 散度对齐 teacher 和 student 的输出预测分布；(2) 特征级蒸馏 L_feat = (1/(NL))*Sigma||f_l^F(x_i;theta) - f_l^Q(x_i;theta')||_2^2，MSE 对齐中间层特征图以稳定低比特训练、防止误差累积；(3) Task-specific 检测损失 L_detect（L_category + L_box + L_conf），直接利用合成标签训练 student 的检测能力。总损失 L^Q = beta_KL*L_KD + beta_feat*L_feat + beta_detect*L_detect。YOLOv5 超参 {beta_detect, beta_KL, beta_feat} = {0.04, 0.1, 1.0}。消融证明三项互补：同时移除 L_feat 和 L_KD 导致 mAP 下降 7.2pp（YOLOv5-s W4A4: 19.0% vs 11.8%）。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

在 PMQ-VE 中，知识蒸馏扩展为多教师层次化蒸馏（PMTD）：训练低比特量化模型时，同时使用 FP 全精度教师和中间比特（INT8）教师进行监督，通过 α(t) 线性增长权重使监督信号从 INT8 逐步过渡到 FP。每个教师包含输出 L2 重建损失和中间特征 MSE 损失（λ=5），相比 2DQuant 的单教师 L1+归一化 L2 损失，PMTD 的多教师策略通过弥合容量差距更有效地提升低比特（4-bit/2-bit）模型的性能。

在 Squat 中，知识蒸馏被用于QAT训练：FP16教师模型通过软蒸馏（soft distillation）指导量化学生模型。蒸馏损失 L_distill = (1-γ)·L_CE + γ·τ²·L_KL。此外，Squat创新地在蒸馏中加入熵损失L_E（最大化量化query/key熵）和分布损失L_D（对齐量化与FP16注意力图余弦相似度），形成 L_total = L_distill + 0.5·L_E + 1.0·L_D 的复合蒸馏目标。

---

## SwinIR

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SwinIR（Shifted Windows Image Restoration）是由 ETH Zurich 提出的基于 Swin Transformer 架构的图像复原模型。其核心架构包含三个模块：(1) Shallow Feature Extraction——一个 3×3 卷积层将输入 RGB 图像映射到高维特征空间；(2) Deep Feature Extraction——K 个 Residual Swin Transformer Block (RSTB) 堆叠加末尾 3×3 卷积提取深层特征；(3) High-Quality Image Reconstruction——融合浅层和深层特征后，通过 sub-pixel convolution（pixel shuffle）上采样重建超分辨率图像。SwinIR 使用残差学习（预测 LQ→HQ 的残差而非 HQ 本身）和长跳跃连接。与传统 CNN-based SR 模型相比，SwinIR 在更少参数下获得更好性能，得益于 Swin Transformer 的自注意力机制和 shifted window 机制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SwinIR-light（2DQuant 使用的版本，4 RSTB, 6 STL/RSTB, 6 heads, embed_dim=60, window=8, MLP ratio=2）的前向计算：
```
# Input: I_LQ (H×W×3)
F_0 = Conv3x3(I_LQ)                  # 浅层特征 (H×W×C)
F = F_0
for k in 1..4:                        # 4 个 RSTB
    F_in = F
    for l in 1..6:                    # 6 个 STL
        # Window-based MSA
        X = WindowPartition(LayerNorm(F))   # (HW/M^2, M^2, C)
        Q,K,V = X@W_q, X@W_k, X@W_v         # Linear 投影 → INT 量化对象
        Attn = SoftMax(Q@K^T/√d + B)        # Batch MatMul → INT 量化对象
        X = Attn@V                            # Batch MatMul
        F = F + WindowReverse(X)
        # MLP
        F = F + FC2(GELU(FC1(LayerNorm(F))))  # FC1, FC2 → INT 量化对象
    F = Conv3x3(F) + F_in              # RSTB 残差连接
F_DF = Conv3x3(F)                      # 深层特征
I_RHQ = SubPixelConv(F_0 + F_DF)      # 重建 (rH×rW×3)
return I_RHQ + Upsample(I_LQ)          # 残差学习
```
2DQuant 对其中所有 Linear（FC1, FC2, Q/K/V 投影）和 Batch MatMul（Q@K^T, Attn@V）执行伪量化和 INT 算术转换，这覆盖了总 FLOPs 的 85.66%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SwinIR 的官方实现基于 PyTorch，开源仓库为 https://github.com/JingyunLiang/SwinIR。模型有多个尺度变体：light（4 RSTB）、classical（6 RSTB）、large（更多通道）。训练使用 DF2K（DIV2K+Flickr2K）数据集，测试在 Set5/Set14/B100/Urban100/Manga109 上评估 PSNR/SSIM。预训练模型可通过 Google Drive 下载。在 2DQuant 的使用场景中，SwinIR-light 被作为 baseline 模型进行 PTQ 量化，其紧凑的参数规模（3.42MB）和高效结构使其适合边缘部署。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution

---

## Quantization-Aware Training (QAT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware Training（QAT，量化感知训练）是一种在训练过程中模拟量化效果的方法。与 PTQ 不同，QAT 在训练时就在前向传播中插入伪量化操作，保持权重为浮点同时模拟低位宽推理的精度损失；反向传播使用 STE 将梯度穿过伪量化操作，同时优化模型权重和量化器参数。QAT 通常能取得比 PTQ 更好的精度（尤其是在极低位宽如 2-bit/3-bit），因为权重可以通过训练自适应量化的影响。但其代价是需要完整的训练流程，计算成本可能超过 FP 模型的原始训练。2DQuant 论文选择 PTQ 路线就是因为 QAT 对 SR 任务来说"训练成本过高甚至超过 FP 模型训练"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QAT 的典型训练流程与标准训练相似，但插入了量化模拟：
```
for epoch in range(N_epochs):
    for x, y in train_loader:
        # 前向：伪量化权重和激活
        w_q = fake_quantize(w, l_w, u_w, bit)
        x_q = fake_quantize(x, l_x, u_x, bit)
        y_pred = model_forward_with_quantized_tensors(w_q, x_q)
        loss = criterion(y_pred, y)
        # 反向：STE 穿过量化操作
        loss.backward()  # ∂L/∂w 通过 STE 近似获得
        optimizer.step()  # 同时更新权重 w 和 clip bounds (l,u)
```
SR 领域的代表性 QAT 方法：PAMS（ECCV 2020，用可训练的截断参数动态确定量化范围上界）、DAQ（WACV 2022，channel-wise 分布感知量化）、CADyQ（ECCV 2022，内容感知的动态位宽分配）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中，QAT 通过 `torch.quantization.prepare_qat()` 将伪量化模块插入模型，训练后通过 `torch.quantization.convert()` 转换为 INT 推理模型。典型步骤：(1) 在模型定义中标记量化位置（`torch.quantization.QuantStub/DeQuantStub`）；(2) `prepare_qat(model)` 插入 FakeQuantize 模块；(3) 正常训练循环；(4) `convert(model)` 将伪量化替换为真正的 INT 算子。在 HuggingFace 生态中，`transformers` 库也支持通过 `BitsAndBytesConfig` 或 `Quanto` 进行 QAT。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

在 Task-Specific ZSQ for Object Detection 中，QAT 被用于目标检测网络的零样本量化微调。QAT 阶段的输入是 Stage I 生成的 2k 张 task-specific 合成校准集（仅真实训练数据的 1/60），使用 LSQ 将 per-tensor symmetric quantization 附加到除首尾层外的所有内部层。总损失 L^Q = beta_KL*L_KD + beta_feat*L_feat + beta_detect*L_detect，其中 L_KD 为 KL 散度预测蒸馏，L_feat 为 MSE 特征蒸馏，L_detect 为检测任务损失（L_category + L_box + L_conf）。Adam 优化器，YOLOv5 lr=1e-4。该方法在 W8A8 YOLOv5-l 上达到 47.3% mAP（超越 full-data LSQ 46.0%），收敛速度可达 full-data LSQ 的 16x。注意该方法属于 ZSQ 范畴——不使用真实图像，仅使用合成校准集和合成标签。

在 Squat 中，QAT 被用于SLM（LLaMA-58M、GPT2-97M）的粗粒度层级别（layer-wise）量化训练。与通常GPU上的channel-wise/token-wise细粒度QAT不同，Squat坚持每层单scale因子的粗粒度量化以兼容移动端SIMD硬件。QAT训练采用FP16教师蒸馏 + 熵损失L_E + 分布损失L_D，通过Token自适应量化动态分配位宽（4-bit或8-bit），并使用STE反向传播。结果：W4A8 BLiMP精度仅↓0.3% vs FP16，移动端加速2.37×。

在 EfficientQAT 中，QAT 通过两阶段策略实现高效量化感知训练：(1) Block-AP：逐block训练所有权重和量化参数(W, s, z)，仅需4096样本/2 epoch即可收敛；(2) E2E-QP：冻结量化权重，仅端到端训练步长s（占参数约1.6%），实现跨block交互。该方案使得Llama-2-70B的2-bit QAT可在单张A100-80GB GPU上41小时内完成（vs LLM-QAT需900h、DB-LLM需82h），训练内存仅34.2GB。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- QT-DoG Quantization-Aware Training for Domain Generalization

在 QT-DoG 中，QAT 被首次用作域泛化（Domain Generalization）的隐式正则化器。QT-DoG 的核心洞察：权重量化引入均匀分布的量化噪声 Δ ∈ [−s/2, +s/2]，该噪声通过二阶 Taylor 展开 L(w+Δ) ≈ L(w) + ∇L Δ + ½Δ^T H Δ 与 Hessian H 交互——在尖锐极小值区域（H 特征值大），Δ 导致损失急剧上升，迫使优化器"逃离"尖锐区域并向平坦极小值收敛。QT-DoG 使用 LSQ 作为量化方法，在训练进行到 2000 步（DomainNet 8000 步）时启动 7-bit 量化，除最后一层外所有层量化至低比特，每通道独立学习 scaling factor s。在 DomainBed 五大基准上，单模型 QT-DoG (7-bit, 0.22× 体积) 达到 66.2% 平均准确率（超过 ERM 的 63.8%），集成版 EoQ 达到 68.4%（超过 DiWA 的 68.0% 且训练开销减少 12×）。关键实验发现：(1) QAT (LSQ/INQ) 有效提升 DG，PTQ (OBC) 无效——因缺少训练阶段无法找到平坦极小值；(2) 7-bit 为最优比特精度；(3) QAT 作为正则化还提升了域内 (IID) 准确率。QAT 在 QT-DoG 中的独特角色不同于传统模型压缩目标——它将量化噪声作为优化正则项，在降低模型体积的同时提升泛化能力。

在 Scaling Law for QAT 中，QAT 被用于训练 74M–973M Llama3-style 模型的 W4A4/W4A16/W16A4 量化配置，共 268 次实验消耗 276K GPU-hours。关键实践：(1) 4-bit QAT 无需高于全精度训练的学习率（量化误差在 LR 5e-4 到 4e-3 范围内几乎恒定于 [0.60, 0.65]）；(2) 权重使用 AbsMax 量化器（因与 LWC/LSQ 精度差异 <0.003），激活使用 AbsMax (G<256) 或 LAC (G≥256)；(3) 使用 INT4 格式（优于 FP4 E2M1，尤其在 per-channel/token 粒度下差距 0.015 loss）；(4) QAT 本身作为正则化器抑制激活 outlier（FC2 kurtosis 从 123→89），但仍无法完全消除 SwiGLU 输出中的系统性 outlier。QAT 缩放定律核心发现：δ_{W4A4} 随 N 增大而减小（74M→594M 平均降 34%）、随 D 增大而增大（10B→100B 平均升 22%）、随 G 变粗而增大（finest→coarsest 差 0.037）。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- Scaling Law for Quantization-Aware Training
- PARQ Piecewise-Affine Regularized Quantization
- Scheduling Weight Transitions for Quantization-Aware Training

在 PARQ 中，QAT 被构建为一个凸正则化优化问题 minimize_w f(w) + λΨ(w)，其中 Ψ 为凸分段仿射正则化函数（PAR）。不同于标准 QAT 使用硬量化映射（STE）全程训练，PARQ 使用渐进软→硬量化：训练初期 proximal map 斜率接近 1（近 identity，几乎不施加量化约束）；训练末期斜率→∞（收敛到硬量化）。这使训练初期的 loss 曲线接近全精度训练，避免了 STE 全程硬量化造成的训练不稳定（如 sudden accuracy drops）。PARQ 在 ResNet (CIFAR-10/ImageNet) 和 DeiT (ImageNet) 的 1-4 bit + ternary 量化上取得与 STE/BinaryRelax 竞争的性能，尤其在小模型极低位宽（1-bit ResNet-20）上有显著优势。PARQ 使用 LSBQ 在线估计目标量化值，无需预先设定量化集合和正则化强度。

## Asymmetric Floating Point Quantization (AFPQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AFPQ（Asymmetric Floating Point Quantization，非对称浮点量化）是 Zhang et al. 提出的针对 LLM 权重非对称分布的浮点量化方法。核心创新：为每组 weight group 内的正值和负值分别设置独立的 scale 参数（scale_pos 和 scale_neg），替代传统对称 FP 量化中所有值共享一个 scale 的做法。数学上，传统对称 FP 量化使用公式 `scale = max(w_max, |w_min|) / (range/2)` 和 `w_q = round(w / scale)`，当 weight group 的分布不对称时，绝对值小的一侧会有大量 FP 候选值落在权重范围之外，造成表达能力浪费。AFPQ 使用 `scale_pos = w_max / (range/2)` 和 `scale_neg = -w_min / (range/2)`，分别量化正负值，使得 FP 候选值精确覆盖权重的实际范围。每组存储两个 scale（与 INT-asym 存储 scale+zero_point 的两个参数存储量相同），无额外存储开销。AFPQ 还保留了 FP 格式在零附近密集分布的优势——相比 INT-asym 的"scale+zero_point"方法（会移动 zero point），AFPQ 的双 scale 方案不改变零的位置。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AFPQ 的量化与反量化伪代码（来自论文 Algorithm 1: FPASYMQuant/FPASYMDequant）：
```
def FPASYMQuant(weight_tensor, group_size, range):
    for each group of group_size weights:
        w_max = max(group)
        w_min = min(group)
        scale_pos = w_max / (range / 2)
        scale_neg = -w_min / (range / 2)
        # 正值和负值分别量化
        w_4bit_pos = round(weight[weight > 0] / scale_pos)
        w_4bit_neg = round(weight[weight < 0] / scale_neg)
    return w_4bit, scale_pos, scale_neg

def FPASYMDequant(w_4bit, scale_pos, scale_neg):
    w_deq = scale_pos * w_4bit_pos + scale_neg * w_4bit_neg
    return w_deq
```
与 INT-asym 的对比（INT 使用 scale + zero_point）：
```
# INT-asym Quant
scale = (w_max - w_min) / (2^bit - 1)
zero_point = round(-w_min / scale)
w_int = round(weight / scale) + zero_point

# INT-asym Dequant
w_deq = scale * (w_int - zero_point)
```
AFPQ 的 scale_pos/scale_neg 与 INT-asym 的 scale/zero_point 都是每 group 两个参数，存储量相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AFPQ 作为即插即用的量化格式，可集成到现有的 W-only PTQ 方法中：(1) RTN 量化——直接使用 FPASYMQuant/FPASYMDequant 替代对称量化；(2) GPTQ 集成——在 OBS 框架的逐列量化步骤中，将 INT quant 替换为 NF-asym quant；(3) AWQ 集成——在 saliency-based scaling 后，将 INT quant 替换为 NF-asym quant。AFPQ 代码开源：https://github.com/zhangsichengsjtu/AFPQ。实验中使用的格式：FP4 E2M1、FP3 E2M0、NF4、NF3。AFPQ 的推理系统基于 FasterTransformer，自定义 NF-asym dequantization kernel（packed byte 解包 → LUT NF→FP16 映射 → 分正负 scale 反量化 → FP16 GEMM）。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

---

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

## Weight-only Quantization (W4A16)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-only Quantization（仅权重量化）是一种只对模型权重进行低比特量化而保持激活值为高精度（通常 FP16/BF16）的量化策略。典型配置为 W4A16（4-bit 权重 + 16-bit 激活）。与 W8A8（权重和激活均为 8-bit）相比，W4A16 的权重存储更紧凑且不需要对激活进行量化-反量化操作，简化了推理系统的 kernel 设计。W4A16 模式下推理流程为：(1) 加载 packed 4-bit 权重；(2) dequantize 权重到 FP16；(3) 执行 FP16 × FP16 GEMM。这种方式在 memory-bound 的 LLM 解码阶段（batch=1, decode token）中效果尤为显著，因为瓶颈在于从显存中读取权重。AFPQ 论文采用 W4A16/W3A16 的 weight-only 量化策略，专注于优化权重量化格式（FP-asym、NF-asym）的精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
W4A16 推理的计算流程（以 AFPQ 的 NF4-asym 为例）：
```
# 存储格式
# 权重: packed byte array, 每 2 个 4-bit NF4 索引占 1 byte
# 参数: 每 group (128 个权重) 存储 scale_pos 和 scale_neg (各 FP16)

# 推理时逐层计算
for each Linear layer:
    for each group of 128 weights:
        # Step 1: 解包
        for i in range(0, 128, 2):
            byte = packed_weights[byte_idx]
            nf4_idx_0 = byte & 0x0F  # 低 4-bit
            nf4_idx_1 = (byte >> 4) & 0x0F  # 高 4-bit
        
        # Step 2: LUT 映射 (NF4 index → FP16 value)
        for each index:
            fp16_val = NF4_LUT[index]  # 16-entry lookup table
        
        # Step 3: 非对称反量化
        for each fp16_val:
            if fp16_val > 0:
                w_deq = scale_pos * fp16_val
            elif fp16_val < 0:
                w_deq = scale_neg * fp16_val
            else:
                w_deq = 0
    
    # Step 4: FP16 GEMM
    output = FP16_GEMM(w_deq, activation_fp16)
```
与 W8A8 的区别：后者需在计算前同时 dequantize 权重和激活，且激活量化的信息损失更大（激活分布更难预测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
W4A16 在 HuggingFace 中通过 `BitsAndBytesConfig(load_in_4bit=True)` 使用。GPTQ 和 AWQ 默认也采用 W4A16 策略。在自定义推理系统中（如 AFPQ 基于的 FasterTransformer），W4A16 需要实现低比特 dequantization kernel。主流推理框架对 W4A16 的支持：vLLM 支持 GPTQ/AWQ 的 W4A16 推理；TensorRT-LLM 通过 Weight-Only Quantization plugin 支持 INT4/FP4 W4A16；llama.cpp 通过 GGUF 格式支持各种 W4A16 格式。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- AffineQuant Affine Transformation Quantization for Large Language Models
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

aespa 采用 W4A16/W3A16/W2A16 的 weight-only 量化策略，仅对权重进行低比特量化而保持激活为 FP16。论文论证理由：(1) LLM 推理中激活不是显著瓶颈；(2) 通过权重量化减少内存移动即可充分加速 LLM 推理；(3) 可兼容仅支持整数运算的硬件（如 NPU）作为未来扩展方向。，通过 scaled pairwise rotation 变换在量化前抑制权重离群值。与仅 dequantize + FP16 GEMM 的标准 W4A16 不同，ParoQuant 在 GEMM 前插入 fused CUDA kernel 对激活 X 应用逆旋转变换 T^{-1}(X)，使变换在推理时在线完成（~10% 开销）而非离线合并（受限且无法覆盖所有线性层）。


## Per-channel Scaling for Weight Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-channel Scaling 是 AWQ 提出的保护显著权重的等效变换技术。核心思想：在权重量化前，对显著通道的权重乘以一个大于 1 的缩放因子 s，同时对对应激活通道除以 s（等效变换保持 MatMul 输出不变）。数学上：`Q(w·s) · (x/s)`，量化误差比例从 1 变为 `Δ'/Δ · 1/s`。由于：(1) 单元素缩放通常不改变 group 的 max → `Δ' ≈ Δ`；(2) `s > 1` → 显著权重的相对误差降低。Scale s 通过参数化搜索确定：`s = s_X^α`，其中 s_X 为 per-channel 平均激活幅度，α ∈ [0,1] 单一超参。最优 α 通过 20 步网格搜索最小化 `||Q(W·diag(s))(diag(s)^{-1}·X) - WX||` 确定。最终 diag(s)^{-1} 融合进前一层（如 LayerNorm weight 或前一层 Linear weight），不增加推理开销。与 KV Cache 量化中的 per-channel scaling（在线实时计算，BitDecoding）关键区别：AWQ 的 weight per-channel scaling 是离线预计算的，不引入运行时延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 OPT-6.7B 某 Linear 层 INT3-g128 量化为例：
```python
# 输入: W [C_out, C_in] FP16, X [C_in, L] FP16 (校准集 cached activations)

# Step 1: 计算 per-channel 激活幅度
s_X = X.abs().mean(dim=1)  # [C_in], 每通道平均激活幅度

# Step 2: 网格搜索 α
best_alpha, best_loss = 0.0, float('inf')
for alpha in linspace(0, 1, 20):  # grid_size=20
    s = s_X ** alpha  # [C_in]
    
    # 等效变换的前向模拟
    W_scaled = W * s.unsqueeze(0)          # [C_out, C_in]
    W_q = groupwise_int_quantize(W_scaled, bits=3, group_size=128)
    # 反量化: W_deq = dequantize(W_q)  [C_out, C_in]
    # 激活反向缩放
    X_inv_scaled = X / s.unsqueeze(1)      # [C_in, L]
    Y_q = W_deq @ X_inv_scaled             # [C_out, L]
    Y_fp = W @ X
    
    loss = (Y_q - Y_fp).pow(2).mean()
    if loss < best_loss:
        best_alpha = alpha

# Step 3: 最终量化
s_final = s_X ** best_alpha
W_final_q = groupwise_int_quantize(W * s_final.unsqueeze(0), bits=3, group_size=128)
# diag(s)^{-1} 融合进前一层
```

Table 2 数据（OPT-6.7B, s_X 的 top 1% channels 缩放 s=2）：
- 未缩放 (RTN): Wiki-2 PPL = 23.54
- s=1.25: PPL = 12.87
- s=1.5: PPL = 12.48
- s=2 (最优): PPL = 11.92
- s=4 (过保护): PPL = 12.36 (non-salient 误差增大)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Per-channel scaling 在 AWQ 中被离线计算并存储在模型中。推理时的实现：(1) 激活侧——前一层输出的 hidden state 逐元素除以 per-channel scale（可融合进 LayerNorm 的 γ 和 β 参数，或融合进前一层 Linear 的 weight/bias）；(2) 权重侧——量化权重存储时已包含 scaling 信息（W_scaled = W · s 后再量化），因此无需额外推理逻辑。在 AutoAWQ 代码库中，per-channel scale 在量化脚本中计算并与模型权重一起保存为 safetensors。vLLM 和 TensorRT-LLM 推理时自动识别 AWQ 格式的模型，加载 per-channel scale 并融合进前一层。这种离线计算 + 推理融合的方式使 AWQ 的 per-channel scaling 在精度提升的同时零运行时开销。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

**FlatQuant 中的 Per-channel Scaling**：FlatQuant 在预量化仿射变换前显式引入可学习的逐通道缩放向量 diag(c) ∈ R^n，作为三个核心组件之一（LT + PS + LCT）。在 LLaMA Transformer 架构中，diag(c) 作用于激活 X 进入线性层之前：Y = (X diag(c)^{-1}) · (diag(c) W^T)。缩放因子 c_j 与仿射变换矩阵 P₁、P₂ 和裁剪阈值 α 联合通过梯度下降优化（Eq.4, MSE loss）。diag(c) 可以逐对融合到前层 LayerNorm 或线性层中消除推理开销。在 down-projection 层中，diag(c) 作用在仿射变换后的激活 X̃ 上，然后融合到 up-projection 的权重 W_u 中。消融实验（Table 16）表明 PS 叠加在 LT 上可将 WikiText-2 PPL 从 8.50 进一步降至 7.95。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- FlatQuant: Flatness Matters for LLM Quantization
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

**ParoQuant 中的 Channel-wise Scaling**：ParoQuant 将 channel-wise scaling diag(α) 与 independent Givens rotations 联合使用构成 scaled pairwise rotation：T(W) = (∏R_t)·diag(α)·W。与 AWQ（grid search 搜索 α）不同，ParoQuant 的 α 通过 AdamW 梯度下降优化（lr=0.05，初始化为 1），且与旋转角度 θ 联合优化——scaling 负责全局幅值均衡，rotations 负责 token 级跨通道值对齐。同时 ParoQuant 的 scaling 逆变换 diag(1/α) 无法简单合并到前序算子（因为后接 rotations），而是在 fused CUDA kernel 中与 rotations 一起在线计算，总开销约 10%。

---

## Post-Training Quantization (PTQ / 后训练量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-Training Quantization (PTQ) 是一种在模型训练完成后应用的量化方法，不需要重新训练或微调模型权重。PTQ 通常使用小规模校准数据集（如 128 条文本片段，每条 2048 tokens）来收集激活统计信息或优化量化参数（如步长、裁剪阈值、变换矩阵）。PTQ 的核心优势在于低成本：相比 Quantization-Aware Training (QAT) 需要完整训练流程，PTQ 仅需数小时在单 GPU 上完成。PTQ 的工作流程一般为：(1) 在校准数据上前向传播收集各层激活分布；(2) 基于分布确定量化参数（scale、zero-point、clipping range）；(3) 对权重逐层或逐 block 进行量化并补偿误差。典型的 LLM PTQ 方法包括 GPTQ（逐列量化+二阶误差补偿）、AWQ（per-channel scaling 保护显著权重）、SmoothQuant（per-channel scaling 平衡权重/激活离群值）、QuaRot（Hadamard 变换消除离群值）和 FlatQuant（可学习仿射变换平坦化分布）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FlatQuant 中的 PTQ 流程（逐 Transformer block 优化）：

```
输入: 预训练 LLM, 校准集 D_calib (128 segments × 2048 tokens)
对于 l = 1 到 L 个 Transformer block:
  1. 收集当前 block 在校准集上的输入 X_l
  2. 初始化可学习参数 Θ = {P₁, P₂, c, α_a, α_w}（仿射变换、缩放、裁剪阈值）
  3. 替换 block 内线性层为量化版本（Eq.3, Kronecker 仿射变换+量化）
  4. 优化: min_Θ || F_l(X_l) - F̂_l(X_l; Θ) ||²  (AdamW, 15 epochs, LR=5e-3)
  5. 保存优化后的 Θ_l，继续下一 block
输出: 量化后的模型权重 + 各层变换参数
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PTQ 通常以 Python/PyTorch 脚本实现，集成在 HuggingFace Transformers 之上。校准数据一般为 WikiText-2 或 C4 的随机子集（128-256 segments）。优化采用逐层（layer-wise）或逐 block（block-wise）策略而非端到端训练，以降低内存需求（逐层仅需加载单层而非全模型）。FlatQuant 对 LLaMA-3-8B 的 PTQ 校准仅需 26GB GPU 内存和 0.9 小时。完成后导出为量化格式（如 GPTQ 格式或自定义格式）供推理框架（vLLM、CUTLASS）加载。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

Quamba2 是 SSM 模型 PTQ 的代表性工作，支持 Mamba1/Mamba2 的 W8A8/W4A8/W4A16 和 W4AX-mixed。其校级核心发现：SSM 的 channel persistence 和 state persistence 使 offline 校准的量化参数在 online 推理中保持有效，从而提出 sort-and-cluster 和 per-state-group quantization。校准集：Pile 随机 512 句（fixed seed），A5000 单 GPU 即可完成全流程。

---

## Pre-quantization Transformation（预量化变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pre-quantization Transformation（预量化变换）是在量化之前对权重和/或激活应用的数学变换，目的是消除离群值（outliers）、平滑分布、降低量化误差。变换必须满足**等价性**：变换后的矩阵乘法输出与原输出相同，即 Y = XW^T = (XT)(T^{-1}W^T)，其中 T 为可逆变换矩阵。变换矩阵 T 作用于激活侧的 TX 可在线计算，T^{-1}W^T 作用于权重侧可离线预计算并融合到量化权重中。常见的预量化变换包括：(1) Per-channel Scaling（对角变换，T=diag(c)）；(2) Hadamard Transformation（T=H，H∈{+1,-1}^{n×n}，正交矩阵）；(3) 可学习正交旋转（SpinQuant）；(4) Kronecker 仿射变换（FlatQuant，T=P₁⊗P₂）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 FlatQuant 的预量化变换为例，在线性层 Y=XW^T 中：

```
// 权重侧（离线预计算，融合到量化权重中）
W̃ = reshape(W, [m, n₁, n₂])              // n = n₁ × n₂
W' = P₁^{-1} ×₁ W̃ ×₂ (P₂^{-1})^T         // 逆变换，维度 [m, n₁, n₂]
W_q = Q(W')                               // 量化到 INT4

// 激活侧（在线推理）
X̃ = reshape(X, [k, n₁, n₂])
X' = P₁^T ×₁ X̃ ×₂ P₂                      // 仿射变换，平坦化分布
X_q = Q(X')                               // 量化到 INT4

// 矩阵乘法（等价性保证）
Y = X_q W_q^T ≈ XW^T                      // 量化近似
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
预量化变换的核心在于选择变换矩阵 T。简单方法使用固定的 Hadamard 矩阵（QuaRot），高级方法通过梯度下降学习最优 T（FlatQuant、SpinQuant）。T 的构造需要在"表达力"和"推理开销"之间权衡：全尺寸矩阵 T∈R^{n×n} 提升表达力但带来 O(n²) 在线计算开销；Kronecker 分解 T=P₁⊗P₂ 将开销降至 O(n√n)。变换矩阵的逆 T^{-1} 通过 SVD 分解（P^{-1}=VΣ^{-1}U^T）稳定计算，并与权重离线融合，不增加推理时的额外存储。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---

## Kronecker Product Affine Transformation（Kronecker 仿射变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Kronecker Product Affine Transformation 是 FlatQuant 提出的核心技术，使用 Kronecker 乘积将大矩阵分解为两个小矩阵的乘积来构造高效的仿射变换。给定需要变换的维度 n，将其分解为 n = n₁ × n₂（取 n₁+n₂ 最小化以最小化参数量），构造两个轻量矩阵 P₁∈R^{n₁×n₁}、P₂∈R^{n₂×n₂}，总变换为 P = P₁ ⊗ P₂。利用 Kronecker 乘积的向量化性质：vec(V)(P₁⊗P₂) = vec(P₁^T V P₂)，将原本需要 O(n²) 的大矩阵乘法转化为两次小矩阵乘法。参数从 n² 降至 n₁²+n₂²（≤ n/2 倍节省），计算量节省 √n/2 倍（取 n₁=n₂=√n 时最优）。例如 LLaMA-2-7B 的 hidden_dim=4096 分解为 n₁=n₂=64，intermediate_dim=11008 分解为 n₁=64, n₂=172。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-2-7B hidden_dim=4096、n₁=n₂=64、单个 token 为例：

```
输入: X ∈ R^{1×4096}, P₁ ∈ R^{64×64}, P₂ ∈ R^{64×64}
W ∈ R^{4096×4096}

// 变换矩阵构造
P = P₁ ⊗ P₂  // 显式 Kronecker 乘积（仅在概念层面）

// 激活侧在线仿射变换
X̃ = X.reshape(1, 64, 64)               // [k, n₁, n₂]
X' = P₁^T @ X̃[0] @ P₂                   // [64×64 matmul] × 2 = 2×64³ ops
// 对比：使用完整 P 的 X' = X @ P 需要 4096² = 16.8M ops

// 权重侧逆变换（离线融合到量化权重）
P₁_inv = V₁ @ Σ₁^{-1} @ U₁^T            // SVD 分解求逆
P₂_inv = V₂ @ Σ₂^{-1} @ U₂^T
W̃ = W.reshape(4096, 64, 64)
W' = P₁_inv^T @ W̃[i] @ P₂_inv            // [per output channel]
W_q = per_channel_quantize(W')          // INT4
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlatQuant 中 Kronecker 仿射变换在 PyTorch 中实现：(1) P₁、P₂ 通过 Cayley 参数化（torch.nn.utils.parametrizations.orthogonal）保证正交性和可逆性；(2) 使用 SVD 分解在 FP16+AMP 下稳定计算逆矩阵（P^{-1}=VΣ^{-1}U^T，非对角元素误差 1×10^{-6} vs 直接求逆的 1×10^{-3}）；(3) 最优分解取 n₁,n₂ = arg min(n₁+n₂) s.t. n₁n₂=n, n₁≤n₂；(4) P₁、P₂ 通过 AdamW 优化器在逐 block MSE 损失上学习，随机初始化。LLaMA-2-7B 所有在线变换的总 FLOPs 仅占 FP16 模型的 2.61%，总内存开销仅 3.41MB。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization

---

## Flatness in LLM Quantization（LLM 量化中的平坦度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flatness（平坦度）在 LLM 量化中指权重和激活张量在各通道上的分布均匀程度。理想情况下，所有通道具有相近的量值（magnitude），在等距量化点下每个通道的量化误差相近且总体最小。度量方法：将 per-channel Frobenius norm 排序为一维向量 d，定义平坦度为 ||d - d'||₂（越小越平坦），其中 d' = (||d||₂/√N)·1_N 是完全均匀分布的理想参考向量。平坦度直接影响量化误差：尖峰分布（steep distribution）导致离群通道被过度压缩（clipping error）或小值通道被舍入噪声淹没（rounding error）。FlatQuant 证明通过可学习仿射变换可显式提升平坦度 —— 训练过程中 MSE loss 下降时，||d - d'||₂ 同步下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-3-8B 某 Transformer block 的平坦度衡计算为例：

```
// 输入: block 内所有权重和激活的 channel-wise magnitude
for each 线性层 l in block:
  for W in {W_q, W_k, W_v, W_o, W_gate, W_up, W_down}:
    d_W = sort(||W[0,:]||, ||W[1,:]||, ..., ||W[m-1,:]||)   // Frobenius norm per output channel
    N_W = len(d_W)
    d'_W = (||d_W||₂/√N_W) · 1_N_W                           // ideal flat reference
    flatness_W = ||d_W - d'_W||₂                              // lower = flatter
  
  for X in layer_inputs:
    d_X = sort(||X[:,0]||, ||X[:,1]||, ..., ||X[:,n-1]||)
    flatness_X = ||d_X - d'_X||₂

total_flatness = Σ(flatness_W) + Σ(flatness_X)
```

**Annotations**: 排序是为可视化（如图 1 的 envelope plot）。d' 的构造确保与 d 具有相同 ℓ₂ norm，可公平比较不同层的平坦度。训练过程中 total_flatness 持续下降（图 7），验证了 loss 下降 ↔ 平坦度提升的因果链。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
平坦度作为量化质量的代理指标，指导预量化变换的设计。FlatQuant 通过逐层学习最优仿射变换直接优化平坦度。在工程中，平坦度的计算不需要额外推理开销——它仅在校准训练阶段作为诊断工具。推理时，已学到的变换矩阵（P₁、P₂、diag(c)）被固化到模型权重中，自动保证量化后的低位计算具有最小误差。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization

---

## Learnable Clipping Thresholds（可学习裁剪阈值）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Clipping Thresholds (LCT) 是 FlatQuant 中用于进一步降低量化误差的组件。在仿射变换平坦化分布后，仍有少量极端值可能导致量化范围过大（浪费量化区间）。裁剪阈值 α ∈ (0,1) 经 sigmoid 函数约束后，定义量化范围为 [−α·max(|X|), α·max(|X|)]（或类似比例），在覆盖大部分信号和排除极端离群值之间寻求最优平衡。与传统网格搜索（如 QuaRot 使用固定 α_a=0.9、α_kv=0.95）不同，FlatQuant 将 α_w（权重裁剪）和 α_a（激活裁剪）作为每层可学习参数，与仿射变换矩阵 P 和缩放向量 c 联合通过梯度下降优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FlatQuant 中 LCT 的量化流程（激活侧）：

```
// 传统 RTN 量化（无裁剪）
X_q = round(clamp(X / s, -2^{b-1}+1, 2^{b-1}-1))

// FlatQuant 带 LCT 的量化
α_a = sigmoid(α_a_raw)                    // α_a_raw 为可学习参数
X_max = max(|X|)
clip_bound = α_a * X_max                   // 缩小量化范围
s = clip_bound / (2^{b-1} - 1)
X_q = round(clamp(X / s, -2^{b-1}+1, 2^{b-1}-1))
// 极端值被裁剪到 clip_bound
```

**关键设计**：LCT 必须放在仿射变换之后而非之前。消融实验（Table 18）表明 LCT before transformation 仅带来 marginal gain，而 LCT after transformation 提升显著（PPL 从 7.95→6.98）。原因是仿射变换先将离群值在通道间重新分配，然后裁剪可有效移除更大比例的极端值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LCT 通过 PyTorch 的 nn.Parameter 实现，α_raw 初始化为使 sigmoid 后约为 0.99（接近不裁剪）。优化时使用独立的更高学习率（5e-2 vs 变换矩阵的 5e-3），因为裁剪阈值需要快速响应分布变化。训练完成后 α 值被固化到模型中，推理时以固定阈值进行裁剪，无额外开销。FlatQuant 中 LCT 对权重和激活分别设置，且对 KV cache 也使用独立的裁剪阈值。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models

**OmniQuant 的 Learnable Weight Clipping (LWC)** 是该论文的核心贡献之一，与 FlatQuant 的 LCT 不同：

(1) **参数化方式**：LWC 学习**相对截断强度** γ ∈ [0,1], β ∈ [0,1]（分别控制上界和下界），而非绝对阈值 α。量化公式为 h = (γ·max(W) - β·min(W)) / (2^N-1), W_q = clamp(round(W/h)+z, 0, 2^N-1)。当 γ=β=1 时退化为标准 MinMax 量化。这种相对缩放设计使 LWC 在 LET（等效变换）每轮改变权重分布时仍能稳定收敛——而 PACT 直接学习绝对截断阈值和 LSQ 直接学习绝对 scale/zero-point 在权重分布动态变化时发散（OmniQuant Table A14, Figure A5 实验证明）。

(2) **与 LET 的协同**：LWC 通过 block-wise 量化误差最小化框架与 LET 联合训练。LET 将激活量化难度迁移到权重（加重了 weight quantization 负担），LWC 恰好专门降低 weight quantization 难度，形成递进优化关系。消融实验（Table A2）表明 LET+LWC 联合训练 PPL=12.87，优于 LET alone (16.97) 和 SmoothQuant+LWC (15.80)，证明了 differentiable joint optimization 的关键性。

(3) **训练效率**：LWC 仅引入 2 个可学习参数（γ,β），默认 20 epochs SGD（W2A16 需 40 epochs），LLaMA-7B 在单卡 A100-40G 上约 1.1h（weight-only）。

---

## Hadamard Transformation for LLM Quantization（LLM 量化中的 Hadamard 变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hadamard Transformation for LLM Quantization 是 QuaRot (Ashkboos et al. 2024) 引入的预量化变换技术。Hadamard 矩阵 H ∈ {+1,−1}^{n×n} 是一个正交矩阵（H^T H = I），其元素仅含 +1 和 −1，无需浮点乘法即可实现（仅需加减）。在 LLM 量化中，对激活和权重分别应用 H 变换：Y = XW^T = (XH)(H^T W^T)，利用矩阵乘法的正交等价性。Hadamard 变换将单个通道的离群值通过旋转分散到所有通道，消除极端离群值峰值。优点：(1) 快速——H 的元素为 ±1，可用快速 Walsh-Hadamard Transform (FWHT) 在 O(n log n) 时间内计算；(2) 全局通用——所有层共享同一 Hadamard 矩阵；(3) 无需学习。局限：(1) 不考虑逐层特性差异，某些层的分布仍呈现 steep envelopes；(2) 对 pivot tokens 的大量离群值效果有限；(3) 修改 LayerNorm 为 RMSNorm 导致全局变换共享受限。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuaRot 中 Hadamard 变换的量化流程（LLaMA-2-7B, W4A4）：

```
// 离线阶段：权重变换
H_n = hadamard_matrix(n)                   // n=4096, H∈{+1,-1}^{4096×4096}
W'_qkv = H_n @ W_qkv @ H_n^T               // QKV 投影权重融合
W'_o = H_n @ W_o @ H_n^T                   // 输出投影
// ... 所有线性层同理
W_q = RTN_quantize(W')                      // 量化

// 在线推理阶段：激活变换
X' = fast_walsh_hadamard_transform(X)       // O(n log n), 仅加减运算
X_q = per_token_quantize(X')               // INT4
Y_q = INT4_matmul(X_q, W_q)                // CUTLASS kernel
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuaRot 中的 Hadamard 变换使用在线 FWHT 实现，带来约 0.26× 端到端减速（3 次在线变换）。SpinQuant 用学习到的正交旋转矩阵替代固定 Hadamard 矩阵以提升表达力。FlatQuant 则完全放弃 Hadamard 变换，转而使用可学习的 Kronecker 仿射变换，在逐层定制化和推理开销间取得更好平衡。QTIP 使用 Random Hadamard Transform (RHT) 作为 incoherence processing 的核心：W̃ ← V_m S_m W S_n V_n^T，其中 V_k 为 Hadamard 矩阵、S_k 为随机符号向量。RHT 以概率 ≥1-δ 使 Ŵ 的 incoherence μ_Ŵ = 2log(4mn/δ)，意味权重近似 i.i.d. 高斯分布——恰好是 TCQ 对 i.i.d. 高斯源高效量化的前提。QTIP 中的 RHT 是离线预处理（无需在线推理开销），Hadamard 矩阵来自 Neil Sloane 网站 (http://neilsloane.com/hadamard/)。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- QTIP: Quantization with Trellises and Incoherence Processing
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

Quamba2 中 Hadamard 变换的使用：(1) offline fusion——将 H 矩阵 offline 融合到 input/output projection 权重（$W_{out}^H = H W_{out} H^T$, $W_{in}^H = W_{in} H^T$），配合 online FWHT 实现 compute-invariance，避免半精度激活中的 outlier 放大 4-bit 权重的量化误差；(2) 在 output proj input 上应用（与 Quamba/MambaQuant 一致），消除 outlier 以提升 4-bit weight 的量化效果——W4A16 ablation 中 Hadamard + PerG 从 64.7% 提升到 69.6% LAMBADA accuracy。
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

SDP4Bit 中 Hadamard 变换的新用途——梯度平滑通信压缩：(1) 在线 32×32 Walsh-Hadamard 变换应用于梯度张量（而非权重/激活），在 INT4 量化前平滑梯度 outlier，使量化误差大幅减少；(2) 两步 Hadamard：intra-node all-to-all 量化前一次 + inter-node all-to-all 量化前一次；(3) 利用 H·H=I 和 ΣHg=HΣg 的数学性质裁剪冗余 transform（6次→2次）；(4) Hadamard 与 (de)quantization 融合为单个 CUDA kernel，要求 group_size 能被 H size 整除（32），确保 fused kernel 内存局部性。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- QTIP: Quantization with Trellises and Incoherence Processing
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals


ResQ 中 Hadamard 的独特用法——U_D FFN 内投影：ResQ 在 FFN block 内部使用 U_D 投影 down_proj 的激活。因 SiLU/GELU 激活函数隔断导致 U_D 无法融入前一层权重，而 d_FFN（通常为 d_hidden 的 3-4 倍）上的直接矩阵乘法开销很大，ResQ 将 U_D 选择为 Hadamard 矩阵，利用快速 Hadamard 变换实现 O(d log d) 的运行时计算。Hadamard 矩阵不存在于特定维度时回退为随机正交旋转。

## Salient Weight Channels

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Salient Weight Channels（显著权重通道）是 AWQ 论文发现并命名的概念：LLM 权重中约 0.1%-1% 的通道对模型性能至关重要。关键发现：显著性的判断依据是**激活分布**而非权重分布——通过 activation magnitude（per-channel 平均激活幅度）筛选出的 top 0.1%-1% 通道，保持其为 FP16 可将 INT3-g128 量化的 OPT-6.7B WikiText-2 PPL 从 43.2 降至 13.0（接近 FP16 baseline 10.86）。而基于权重 magnitude 或随机选择的同等比例通道则几乎无效（PPL 仍 > 23）。直观解释：激活幅度大的通道处理更重要的特征（对应输出维度贡献大），因此这些通道对应的权重对模型精度至关重要。这一发现为 AWQ 的 per-channel scaling 策略提供了理论依据——不需要真正的混合精度（部分 FP16 + 部分 INT），而是通过等效缩放降低显著通道的量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
显著权重的识别与保护流程（以 OPT-6.7B INT3-g128 为例）：
```python
# Step 1: 用校准集收集激活统计
X = collect_activations(model, calibration_data)  # [C_in, total_tokens]
s_X = X.abs().mean(dim=1)  # per-channel 平均激活幅度

# Step 2: 识别显著通道 (按激活幅度排序，取 top k%)
k = 0.01  # 1% = top 1% channels
threshold = torch.quantile(s_X, 1 - k)
salient_mask = s_X >= threshold  # [C_in], True for salient channels

# Step 3 (直接方法 - 混合精度): 
# 保持显著通道为 FP16，其余为 INT3
W_salient = W[:, salient_mask]   # FP16
W_non_salient = W[:, ~salient_mask]  # INT3 量化

# Step 4 (AWQ 方法 - per-channel scaling):
# 避免混合精度，用等效缩放保护显著通道
s = torch.ones(C_in)
s[salient_mask] = s_X[salient_mask] ** alpha  # 放大显著通道
W_q = groupwise_quantize(W * s.unsqueeze(0))
# diag(s)^{-1} 融合进前一层
```

Table 1 关键数据（OPT-6.7B, INT3-g128, WikiText-2 PPL↓）：
| 保护方法 | 0.1% FP16 | 1% FP16 | 3% FP16 |
|----------|-----------|---------|---------|
| 基于激活分布 | 11.58 | 11.39 | 11.36 |
| 基于权重分布 | 23.41 | 22.37 | 22.45 |
| 随机选择 | 23.54 | 24.23 | 24.22 |
| 全量 RTN (无保护) | 23.54 |
| FP16 (上界) | 10.86 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
显著通道的识别仅需一次校准集前向传播 + per-channel 平均幅度统计，无需反向传播或重建，因此极快且数据高效。在 AWQ 实现中，每个需要量化的 Linear 层（Q/K/V/O/FC1/FC2 等）独立计算显著通道并搜索最优 scale。实际部署中，显著通道的 scale 和量化权重一起序列化保存，推理框架（vLLM/TensorRT-LLM）加载后自动执行等效变换。显著性概念启发了后续工作：SmoothQuant 使用 per-channel scaling 平滑激活 outlier；GPTQ + AWQ 组合中，先做 per-channel scaling 保护显著通道再做二阶误差补偿，取得更好 INT2 结果。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- PB-LLM Partially Binarized Large Language Models
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

PB-LLM 从二值化角度补充了显著权重检测的另一个维度：**检测粒度和准则的选择**。PB-LLM 比较了 (1) Magnitude vs Hessian 两种检测准则，发现在 PTQ 场景下 Hessian 准则（v_i = w_i²/[H⁻¹]_{ii}²，源自 SparseGPT）略优于 Magnitude，但差异在 QAT 中不明显；(2) Element-wise vs Column-wise 两种粒度，发现 LLM 中显著权重呈均匀随机散射分布（无列聚集模式，见图 3），因此 column-wise 选择会损害二值化性能，必须使用 element-wise 检测。这与 AWQ 的 per-channel activation-based 检测形成互补——AWQ 关注激活 outlier 通道（per-channel），PB-LLM 关注权重大幅值单点（element-wise）。

SLiM 将显著权重概念进一步推广到联合权重-激活显著性：saliency = |diag(x_mean) × W|，其中 x_mean 为校准集逐输入通道的平均激活幅度。与 AWQ 仅用激活幅度不同，SLiM 的显著性同时考虑权重和激活的幅值，作为量化误差补偿（SLiM-LoRA）和激活感知量化（SLiM-Quant^O）的统一重要性度量。在 SLiM-LoRA 中，该显著性函数满足可逆性和可加性，使低秩适配器能通过 SVD 直接从误差显著性矩阵中闭式推导。

---

## Average Perturbation Hessian (APH) Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Average Perturbation Hessian (APH) 是一种用于量化重建的损失函数，直接通过有限差分法计算输出 Hessian 矩阵的对角元，用以衡量每个输出元素对任务损失的重要性。推导流程：(1) 将量化视为对输出 O 的小扰动 ε，对蒸馏损失 L 进行 Taylor 展开：L(O+ε) - L(O) ≈ (1/2)·ε^T·H·ε，其中一阶项 J(O) 在 O 处为 0（蒸馏损失在 O=Ô 处取极小值）；(2) 对 block 输出施加微小扰动 ΔO=10^-6 得到 O⁺=O+ΔO 和 O⁻=O-ΔO；(3) 前向传播通过剩余 blocks 计算蒸馏损失；(4) 反向传播得到 Jacobian J⁺ 和 J⁻；(5) 用均值定理 H_i = (J⁺_i - J⁻_i) / (2·ΔO) 直接计算 Hessian 对角元；(6) 对所有校准样本取平均 H̄ = (1/N)·ΣH^(n)。最终 APH loss 为 L_APH = Σ_i (Ô_i - O_i)² · H̄_i。与 BRECQ 的 Hessian loss（Fisher Information Matrix + 梯度平方近似）相比，APH 直接从定义推导，避免 FIM 近似误差，理论上可泛化到检测、分割等多任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# APH 计算流程（APHQ-ViT Algorithm 1）
O = B.forward(X)                              # [N, L, D]
O_plus, O_minus = O + 1e-6, O - 1e-6
logit       = M.rest(O)                        # 通过剩余 blocks
logit_plus  = M.rest(O_plus)
logit_minus = M.rest(O_minus)
L_plus  = DistillLoss(logit, logit_plus)       # 分类: KL div
L_minus = DistillLoss(logit, logit_minus)      # 检测: KL + smooth L1
J_plus  = grad(L_plus,  O_plus)                # ∂L/∂O⁺
J_minus = grad(L_minus, O_minus)               # ∂L/∂O⁻
H_sample = (J_plus - J_minus) / (2e-6)         # 有限差分 Hessian 对角
H_bar = mean(H_sample, dim=0)                  # 跨样本平均

# APH loss 在量化重建中的使用
O_hat = B_quantized.forward(X)
L_APH = sum((O_hat - O)^2 * H_bar)             # APH 加权 MSE
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
APH loss 在 APHQ-ViT 开源代码中通过 PyTorch 自定义实现：(1) 在 block 重建前预先计算 H_bar（一次额外 forward+backward）；(2) H_bar 存储为张量在迭代中复用；(3) 蒸馏损失：分类用 nn.KLDivLoss，检测用 KLDivLoss + nn.SmoothL1Loss；(4) 定理 3.2 证明 APH 梯度方差更低（Var[∂L_APH/∂θ] ≤ Var[∂L_PH/∂θ]），训练更稳定。仅增加一次额外 forward/backward pass，训练复杂度不变。校准集：ImageNet 1024 张无标签图。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers

## MLP Reconstruction (MR) — GELU-to-ReLU Replacement for ViT PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLP Reconstruction (MR) 是 APHQ-ViT 提出的针对 ViT 中 post-GELU 激活量化困难的后训练重建方法。核心操作：(1) 将 ViT MLP 中的 GELU 激活函数替换为 ReLU；(2) 用 APH 加权的特征蒸馏损失重建 MLP，使 ReLU 版本逼近原始 GELU 版本输出。MR 同时解决 post-GELU 激活两个难题：负值集中在 [-0.17, 0] 造成的分布不平衡（GELU 密集负值被 ReLU 的精确零替代），以及层间激活范围变化大（通过 clamp loss 将正激活限制在 99% 分位数内）。因 MR 逐层单独重建（浅层网络），避免了深层 ReLU 的 dying ReLU 问题。额外收益：ReLU 可折叠进前层 FC，在 W8A8 CPU 推理中实现 1.49×-1.75× 加速。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MR 伪代码
O_GELU = FC2(GELU(FC1(X)))                    # 原始 MLP 输出
MLP.activation = ReLU()                        # 替换 GELU → ReLU

for iter in range(max_iter):
    A_fc2 = ReLU(FC1(X))
    O_direct = FC2(A_fc2)                      # 无 clamp 输出
    thresh = quantile(A_fc2, p=0.99)           # 99% 分位数
    O_clamp = FC2(clamp(A_fc2, max=thresh))    # clamp 输出
    
    L_direct = mean((O_GELU - O_direct)^2 * H_bar)  # APH 加权
    L_clamp  = mean((O_GELU - O_clamp)^2  * H_bar)
    L_distill = L_direct + 2.0 * L_clamp       # α=2
    L_distill.backward(); optimizer.step()
```
注：L_Direct 不可省略——L_Clamp 对硬截断区域的梯度为 0，单独使用会导致梯度消失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MR 在 APHQ-ViT 中作为 block 重建的第一阶段（在量化重建前）。参数：p=0.99, α=2, max_iter=20000, batch_size=32, lr=4e-5。单独使用 MR（不量化）精度损失 <0.5%，ViT-B 上甚至超全精度 baseline。局限性：仅在 ViT MLP 上验证，未在 LLM 等模型测试。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers

## Block-wise Reconstruction（逐块重建）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Reconstruction 是 BRECQ (ICLR 2021) 提出的后训练量化框架，将模型按 block 单元逐个进行量化重建。流程：(1) 将模型拆为若干 block；(2) 用校准集获取 block 原始 FP 输出 O；(3) 量化 block 内权重/激活得到 Ô；(4) 最小化 Ô 与 O 的重建损失（MSE 或 Hessian guided loss）；(5) 使用 AdaRound 学习舍入策略。Block-wise 相比 layer-wise 考虑了跨层依赖，相比全局重建计算开销可控。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for each block B in M.blocks:
    O_fp = B.forward(D)
    H_bar = compute_APH(O_fp, M.rest, D)       # (APHQ-ViT)
    if B has MLP:
        replace GELU → ReLU; reconstruct MLP   # MR (APHQ-ViT)
    for iter in range(max_iter):
        O_hat = B_quantized.forward(D)          # QDrop + AdaRound
        L = sum((O_hat - O_fp)^2 * H_bar)       # APH loss
        L.backward(); update(AdaRound_weights)
    M.replace_block(B, B_quantized)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过 hook 机制截取 block 输入/输出。典型超参：batch_size=32, max_iter=20000, lr_weight=1e-3, lr_act=4e-5。校准集：ImageNet 1024 张无标签图。BRECQ 用 MSE 或 FIM 近似 Hessian loss；QDrop 添加随机 activation dropout；APHQ-ViT 用 APH loss + MR。

I&S-ViT 使用标准 block-wise reconstruction L_l = ||X_l - X̄_l||_2 作为学习目标，只向后更新第 l 个 transformer block 的权重。在 SOS 三阶段策略中：Stage 1 用全精度权重 + channel-wise 激活量化优化 block-wise loss；Stage 2 通过 scale reparameterization 无损转换；Stage 3 在量化权重和 layer-wise 量化下再优化 block-wise loss。Adam 优化器，lr=4e-5，cosine 衰减，ImageNet batch_size=64，6-bit 200 iterations，其他 1000 iterations。

在 EfficientQAT 的 Block-AP 中，block-wise reconstruction 首次被扩展为真正的 QAT——直接训练所有权重(W)和量化参数(s, z)，而非仅训练辅助参数（rounding/clipping/step sizes）。Block-AP 实验证明（Table 5）：全训练(s,z,W) PPL=8.53 vs 仅训练rounding PPL=15.50 vs 仅训练clipping PPL=11.28，且全训练显存(8.5GB)反而低于rounding训练(8.6GB，因需额外存储rounding参数副本)。这表明无需复杂的可训练参数设计即可实现最优block-wise重建。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- AffineQuant Affine Transformation Quantization for Large Language Models
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
- PT²-LLM Post-Training Ternarization for Large Language Models
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

aespa 论文提出了一种介于 layer-wise 和 block-wise 之间的折中方案：逐层单独量化以保持效率（layer-wise granularity），但以 attention 输出为重构目标（block-wise-like reconstruction target）。与 BRECQ 的全 block 联合量化不同，aespa 将 W_Q、W_K、W_V 逐个量化，但每层的损失函数都指向 attention 输出误差最小化。该策略的复杂度为 O(d_h d^2)，远低于传统 block-wise 的 O(B d_h L·max{d,L})。实验表明该折中方案在 INT3/4 下性能几乎不损失，INT2 下仅有轻微退化（vs BRECQ），但量化速度提升 10× 以上。，在其基础上额外收集量化输出-噪声对 (ε̂, Δε)_t 用于估计时间步感知的联合高斯分布参数。

BiLLM 使用 GPTQ/OBC 的 block-wise 二阶误差补偿（block_size=128），但移除了 column-wise 补偿以提升 PTQ 效率。补偿过程：对每个 128 列 block 完成量化后，计算误差 E = (W - B) / H_chol_inv，将 E 乘以 Hessian 逆对应行补偿到后续未量化列中（W_{:,b+β:} -= E · H_chol_inv_{b:b+β, b+β:}），此方法在二值化场景同样有效。

PT²-LLM 在 block-wise ternarization（block_size=128）中使用 SSR（Structural Similarity-based Reordering）替代固定顺序或 Hessian 重排序。SSR 的核心动机：三值化对块内权重分布极为敏感——离群列和散乱分布会扭曲三值网格 {−α+μ, μ, α+μ}，导致大量权重被错误映射。SSR 在每次选块时，计算残差矩阵列间余弦相似度，选 top-k 最相似列组成量化块，使块内分布更紧凑、块内方差更小、离群列因彼此相似而不再是离群值。

## Hessian Guided Quantization Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hessian Guided Quantization Loss 是 BRECQ 提出的量化质量评估指标，通过二阶 Taylor 展开估计量化引入的任务损失。E[L(Ŵ)] - E[L(W)] ≈ (1/2)·ε^T·H(W)·ε。Hessian 近似为 Fisher Information Matrix (FIM)，FIM 对角元近似为输出梯度平方：H_i ≈ (∂L/∂O_i)²。最终 L_Hessian = Σ_i (Ô_i - O_i)² · (∂L/∂O_i)²。该近似的成立条件：(1) 模型完美拟合真实分布（FIM ≈ Hessian）；(2) 任务损失为交叉熵（Bartlett 第二恒等式）。当模型拟合不佳或用于检测/分割任务时近似误差显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
L_ce = CrossEntropy(O_hat, labels)
grad = dL_ce/dO_fp                              # 输出梯度
H_brecq = grad^2                                # FIM 对角近似
L = sum((O_hat - O_fp)^2 * H_brecq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BRECQ Hessian loss 在 APHQ-ViT 论文中被系统分析其局限性：(1) 蒸馏损失（KL 散度）不满足 Bartlett 恒等式；(2) Table 4 显示 MSE 在某些 ViT 架构上优于 BH（ViT-B: MSE=73.79 vs BH=66.62）；(3) 无法泛化到检测/分割。APHQ-ViT 的 APH loss 通过有限差分直接计算 Hessian 对角元规避这些近似误差。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs
- First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

aespa 提出 attention-aware Hessian 替代传统的 layer-wise Hessian H=2E[XX^T]。对 W_V 的 Hessian 定义为 H_V=2E[XA^TAX^T]（Equation 18），通过 attention map A 将 Q 和 K 投影的信息耦合进 Hessian。对 W_Q/W_K 的损失函数分别通过 E[K^TK] 和 E[Q^TQ] 注入跨投影信息，形式上仍使用 H=2E[XX^T] 作为基础 Hessian 但损失计算为 tr(E[K^TK]·ΔW·E[XX^T]·ΔW^T)。这种 attention-aware Hessian 的核心动机是：标准 H=2E[XX^T] 仅考虑输入 X 的统计特性，忽略了 attention 模块内 Q/K/V 之间的交互，而 H_V 通过 A=softmax(QK^T/√d) 天然编码了这种交互。

BiLLM 将 Hessian 矩阵用于两个关键环节：(1) Salient 权重识别：s_i = w_i²/[H^{-1}]_{ii}²，通过 Hessian 逆的对角元加权限定每个权重元素对层输出的二阶影响，比仅用 |w| 判断显著性更准确（捕获了权重值小但对输出敏感的权重）；(2) Block-wise OBC 误差补偿：H 的 Cholesky 分解 H_chol = Cholesky((H+λI)^{-1}) 用于计算补偿公式，其中 Hessian 由校准数据 X 计算：H = 2XX^T（L2 误差 Hessian）。λ 为正则化项防止 Hessian 矩阵奇异。

FOEM 将 Taylor 展开 δE = g δw^T + ½ δw H δw^T 的一阶项 g 从假设为零改为显式保留：通过 g ≈ β(W−𝕎)H（β=0.1）近似梯度，代入 Lagrangian 约束优化后 H 和 H^{-1} 在代数中自动消去。这揭示了二阶-only 建模（如 BRECQ Hessian loss 和 GPTQ）在逐列累积补偿场景下的系统性误差。

GuidedQuant (ICML 2025) 进一步将该思路推广到现代 LLM（70B）：(1) 使用 block-diagonal Fisher 近似替代 BRECQ 的对角近似，保留每个 output channel 内 d_in×d_in 的 Fisher block；(2) 引入 averaging approximation，将 d_out 个 Fisher block 按 g 组（g=2~4）平均，使存储从 Θ(d_in² d_out) 降至 Θ(d_in² g)；(3) 将 guided Hessian H̄_k 直接集成到现有 layer-wise output-based PTQ 方法（LNQ、QTIP、SpinQuant）中作为直接 plugin。

## AdaRound (Adaptive Rounding)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaRound (ICML 2020) 是后训练量化权重舍入优化方法。将舍入建模为二值优化：每个权重决定"向上舍入"还是"向下舍入"，通过连续软舍入变量 V（sigmoid→[0,1]）最小化重建损失，用 STE 梯度回传优化。V 初始化为 0（对应 RTN），经微调获得更好舍入决策。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
W_int = floor((W - z) / s)
h = sigmoid(V)                                  # [0,1] 软舍入
W_hat = s * (W_int + h) + z                     # 软量化
L = APH_loss(layer(W_hat, X), O_fp)
dL/dV = dL/dW_hat * s * sigmoid'(V)             # STE 梯度
V = V - lr * dL/dV
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaRound 在 BRECQ/QDrop/APHQ-ViT 中作为 block 重建核心优化变量。典型配置：lr_weight=1e-3，Adam 优化器。不改变量化参数（scale/zp），仅优化舍入方向，不增加推理开销。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

D2-DPM 使用 AdaRound 作为扩散模型 PTQ 的权重量化器，固定首尾层为 8-bit，其余层用 AdaRound 量化至目标位宽。

aespa 将 AdaRound 的 layer-wise 重构目标替换为 attention-wise 重构目标：对 W_Q 使用 `loss = tr(E[K^TK]·ΔW_Q·E[XX^T]·ΔW_Q^T)`、W_K 使用 `loss = tr(E[Q^TQ]·ΔW_K·E[XX^T]·ΔW_K^T)`、W_V 使用 `loss = tr(ΔW_V·E[XA^TAX^T]·ΔW_V^T)`，替代原始的 `loss = ||WX - W̃X||^2`。参数设置：2000 iterations, lr=0.015, rounding loss weight λ=1.5。

## QDrop (Random Activation Dropout for PTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QDrop (ICLR 2022) 在 block-wise 重建中随机丢弃部分激活量化器以提升泛化性。每次前向以概率 p 随机将某些层激活"掉回"为 FP，迫使模型学习在混合精度条件下重建。随机性打破量化误差的确定性模式，防止过拟合特定量化配置，原理类似 Dropout。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for iter in range(max_iter):
    for layer in B.layers:
        w_hat = quant_dequant(layer.weight)     # 始终量化权重
        x_q = quant(layer.input) if rand() > p else layer.input
        output = w_hat @ x_q
    O_hat = B.forward_quantized(X)
    L = APH_loss(O_hat, O_fp)
    L.backward(); update(AdaRound_weights)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QDrop 在 BRECQ 基础上添加随机激活 dropout（p=0.5），不改变优化目标。APHQ-ViT 直接使用 QDrop 作为量化重建框架，将 BRECQ Hessian loss 替换为 APH loss。额外开销极小（仅前向随机判断），显著提升 W3/A3 精度。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers

## Fisher Information Matrix (FIM) in PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FIM 在 PTQ 中作为 Hessian 矩阵近似用于量化质量评估。F = E[(∂log p/∂W)(∂log p/∂W)^T]，在模型完美拟合真实分布且使用负对数似然损失时 FIM = E[H]（Bartlett 第二恒等式）。BRECQ 将 Hessian 对角元近似为 H_i ≈ (∂L/∂O_i)²。APHQ-ViT 指出：实际模型拟合不完美时 FIM ≠ Hessian；蒸馏损失（KL 散度）不满足 Bartlett 恒等式条件；无法泛化到非分类任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BRECQ: FIM 近似 Hessian
H ≈ F,  F_i ≈ (∂L/∂O_i)²                     # 两点近似
L_brecq = Σ (Ô_i - O_i)² * (∂L/∂O_i)²

# APHQ-ViT: 直接计算 Hessian 对角
H_i = (J⁺_i - J⁻_i) / (2 * 1e-6), J = ∂L/∂O  # 有限差分
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FIM 近似实现简单（只需一次 backward），被 BRECQ/PTQ4ViT 广泛采用。但 APHQ-ViT 实验显示在 ViT 量化中不稳定（Table 4: BH 在 ViT-B 上 66.62% vs MSE 73.79%）。APH 以一次额外 forward/backward 代价换取更准确估计。

GuidedQuant (ICML 2025) 将 FIM 近似从 diagonal 扩展到 block-diagonal：保留每个 output channel 对应的 d_in×d_in Fisher block `F_j = (1/n) Σ (∂ℓ_i/∂w_j)(∂ℓ_i/∂w_j)ᵀ`，捕获 channel 内跨权重交互；并通过 averaging approximation 将 d_out 个 block 按 g 组（2~4）平均，使 70B 级 LLM 的存储和计算可行。FIM block 的构建等价于计算 `H̄_k = XᵀDiag(avg_squared_grad)X`，核心信息来自单次 backward pass 的梯度。

SqueezeLLM (ICML 2024) 使用 Fisher 对角作为 sensitivity-based non-uniform quantization 的权重。优化目标从 `argmin ||W - W_Q||²` 变为 `argmin Σ F_ii (w_i - Q(w_i))²`，其中 `F_ii = (1/|D|) Σ_d g_{d,i}²` 来自 calibration 数据集（仅需 10-100 样本）的一次 backward pass。该 weighted k-means 目标源自 Optimal Brain Damage (OBD) 框架：假设模型已收敛（梯度 g≈0），对 loss 做 Taylor 展开得到 `L(W_Q) ≈ L(W) + ½(W-W_Q)ᵀH(W-W_Q)`，用 Fisher 对角近似 Hessian 对角（H ≈ diag(F)）。与 GPTQ 的 layer-wise objective（最小化 ||WX - W_QX||²，即每层输出 activation 扰动）相比，SqueezeLLM 的 final-loss-based objective 在所有 sparsity level 下 PPL 低约 0.3（D.4 消融实验，LLaMA-7B 3-bit C4）。Fisher 计算开销小（65B 模型仅需 2.5 分钟 on A100），但需要一次完整 forward+backward（内存峰值 7B=33GB, 65B=292GB）。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- SqueezeLLM Dense-and-Sparse Quantization

## Post-GELU Activation Quantization in ViTs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-GELU Activation Quantization 指 ViT MLP 中经过 GELU(x) = x·Φ(x) 激活后的激活值量化挑战。GELU 导致：(1) 分布极不平衡——负值集中在窄区间 [-0.17, 0]，正值稀疏且范围可达 40；(2) 层间激活范围差异巨大。均匀量化器低效——密集负值的细微变化被粗糙量化，稀疏正值的宽范围浪费格点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GELU: 密集负值 + 稀疏宽范围正值 → 量化低效
GELU(-3)≈ -0.004, GELU(-1)≈ -0.159, GELU(0)=0
密集区间 [-0.17, 0], 正值可达 40

# ReLU: 精确零 + 线性正值 → 量化高效
ReLU(x<0) = 0 (精确零), ReLU(x>0) = x (可 clamp)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
已有方法：PTQ4ViT 的 Twin-Uniform Quantizer、AdaLog 的对数量化器、DopQ-ViT 的 Tangent 量化器——均需专用硬件。APHQ-ViT 的 MR 方法通过 GELU→ReLU 替换从根本上消除该问题，使用标准均匀量化器达到优于专用量化器精度。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers

## Channel-wise vs Layer-wise Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Channel-wise 为权重张量每个输出通道独立分配量化参数（s_c, z_c）；Layer-wise 为整个层使用单一参数组。Channel-wise 适应不同通道分布差异精度高但存储稍增；Layer-wise 参数少实现简单但精度低。APHQ-ViT 中权重使用 channel-wise 量化，激活使用 layer-wise 量化——精度与效率的标准折中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Channel-wise (权重): W[C_out, C_in]
for c in range(C_out):
    s_c = (max(W[c]) - min(W[c])) / (2^N - 1)
    W_int[c] = round(clip(W[c]) / s_c)

# Layer-wise (激活): X[B, C, H, W]
s = (max(X) - min(X)) / (2^N - 1)
X_int = round(clip(X) / s)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch: `PerChannelMinMaxObserver` (channel-wise), `MinMaxObserver` (layer-wise)。APHQ-ViT 在 block 重建中同时使用两者，量化参数校准阶段通过 min-max 确定后固定，重建阶段通过 AdaRound 优化舍入方向。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

I&S-ViT 首次系统分析了不同量化粒度组合对 loss landscape 的影响：channel-wise 权重量化 + layer-wise 激活量化 → rugged、高 loss landscape（不利于优化）；全精度权重 + channel-wise 激活量化 → 平滑、低 loss landscape（利于优化）。基于此分析提出 SOS 三阶段策略——先在 channel-wise 激活量化下优化（平滑 landscape），再通过 scale reparameterization 无损转 layer-wise（保持推理效率），最后量化权重微调。

在 Squat 中，粗粒度逐层量化（layer-wise quantization）是核心设计约束。Squat明确批评之前QAT工作（LLM-QAT、EfficientQAT、TSLD）采用channel-wise/token-wise细粒度量化——这些方法虽然在精度上有优势，但每个矩阵内有多个scaling factor，无法在移动端SIMD的GeMM kernel上高效执行。标准SIMD-based GeMM kernel（如gemmlowp/QNNPACK）仅支持每矩阵单scale的整数MAC操作。因此Squat坚持逐层粗粒度量化以保证移动端部署兼容性，通过熵损失+分布损失+token自适应量化来弥补粗粒度量化的精度劣势。

## Weight Binarization（权重二值化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Binarization 将浮点权重压缩为仅 1 bit（±1 值），是最极端的量化形式。标准一阶二值化：对于权重 W ∈ R^{n×m}，先计算行均值 μ = mean(W)，做中心化 W̃ = W - μ 使分布对齐零均值，然后取 α = mean(|W̃|, axis=1) 作为行缩放因子，B = sign(W̃) 为二值矩阵，重构 Ŵ = αB + μ。量化误差 L₁ = ||W - Ŵ||²。二阶二值化（Ŵ = α₁B₁ + α₂B₂ + μ）用两个二值矩阵和两个缩放因子更好地逼近原始权重，通过枚举 {±α₁±α₂} 四种组合选最近邻确定 B₁,B₂ 元素。二值化不改变推理时的计算范式（仍以 FP16 GEMM 执行），但因权重以 1-bit 存储，压缩比可达 ~16×（vs FP16）。二值化 PTQ 的关键挑战是分布偏移（二值化后均值不对齐）和列间偏差（某些列值远大于其他列）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B 某 Linear 层权重 W ∈ R^{4096×4096} 的一阶二值化为例：
```
μ = mean(W, dim=1)             # 行均值, shape (n,)
W_centered = W - μ              # 零均值化
α = mean(|W_centered|, dim=1)   # 行缩放因子, shape (n,)
B = sign(W_centered)            # 二值矩阵 ±1, shape (n,m)
Ŵ = α.unsqueeze(1) * B + μ.unsqueeze(1)   # 重构 W
```
二阶二值化：候选向量 V = {-α₁-α₂, -α₁+α₂, +α₁-α₂, +α₁+α₂}，对每个元素选最接近 (W-μ)[i,j] 的候选值，确定 B₁,B₂ 对应元素。推理时：`W_deq[i,j] = α₁[i]·B₁[i,j] + α₂[i]·B₂[i,j] + μ[i]`，标准 FP16 GEMM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch PTQ 实现：基于 Hessian 选出 salient columns → 二阶二值化 salient → 一阶二值化 non-salient → block-wise OBC 补偿。BiLLM（ICML 2024）是首个在 LLM 上实现 ~1.1-bit 的 PTQ 方法，ARB-LLM（ICLR 2025）进一步引入交替精炼（T=15 迭代）。推理时权重以 packed 1-bit 格式存储，通过 dequant 后 FP16 GEMM 执行。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- Binarized Diffusion Model for Image Super-Resolution
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models
- PB-LLM Partially Binarized Large Language Models

PB-LLM 提出**部分二值化**范式：不是全部二值化，而是检测保留 5%-30% salient weights 为 INT8，仅二值化剩余权重。关键发现：已有全部二值化方法（BNN, XNOR, Bi-Real, ReCU, FDA）直接应用于 LLM 导致完全崩溃（< random guess）。PB-LLM 使用 column-wise α* = ||w_F||₁/n 作为最优缩放因子（与 XNOR-Net 一致），结合 salient weights frozen 进行 QAT。

在 BinaryDM 中，权重二值化采用 Evolvable-Basis Binarizer (EBB)：通过双基残差结构 w_EBB^bi = σ_I*sign(w) + σ_II*sign(w - σ_I*sign(w)) 在训练初期增强表征，再通过正则化驱动 σ_II→0 平滑过渡到标准单基二值化 w^bi = σ_I*sign(w)。BinaryDM 是首个将 DM 权重推至真正 1-bit 的方法，W1A4 下实现 15.2× OPs 和 29.2× 存储节省。

扩散模型二值化的独特挑战：BI-DiffSR 首次将二值化应用于扩散模型 SR。与 LLM 二值化不同，扩散模型面临 (1) UNet 结构维度变化导致 identity shortcut 无法传递 FP 信息，(2) skip connection 中 encoder/decoder 特征值域差异巨大导致融合困难，(3) 多步迭代中激活分布随 timestep 剧烈变化。BI-DiffSR 通过 CP-Down/CP-Up（保持 ResBlock 维度一致以允许 shortcut）、CS-Fusion（channel shuffle 平衡值域后融合）、TaR/TaA（MoE 风格分组 bias+RPReLU 适配 timestep 变化）解决上述问题。二值化计算通过 XNOR + bit-count 替代浮点 MAC，理论节省 32x 内存和 64x 计算。

---

## Alternating Refined Binarization (ARB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ARB（Alternating Refined Binarization）是 ARB-LLM（ICLR 2025）提出的迭代二值化精炼算法。传统二值化一次性计算 μ, α, B 后不再调整，导致二值化权重与全精度权重存在分布偏移（残差 R = W - Ŵ 的均值非零）。ARB 通过交替迭代更新三个参数来逐步对齐分布：每轮先更新 μ_refine = μ + mean(R)（修正均值偏移），再解析更新 α = diag(B^T(W-μ))/(Σ(B⊙M)²)（∂L₁/∂α=0 解），最后更新 B = sign(W-μ)。理论保证（Theorem 1）：每轮 L₁^τ = L₁⁰ - m((α^τ)²-(α⁰)²-(μ^τ-μ⁰)²) ≤ L₁⁰。ARB 有两个扩展：ARB-X（引入 calibration data 优化 L₂）和 ARB-RC（双轴缩放消除 μ）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一阶 ARB 伪代码：
```
ARB(W, M, T):
    μ = mean(W⊙M, dim=1); α = mean(|W-μ|⊙M, dim=1); B = sign(W-μ)
    for iter in 1..T:
        R = W - (α·B + μ); δμ = mean(R⊙M, dim=1); μ += δμ
        α = Σ_j (B·,j⊙M·,j)·(W·,j-μ) / Σ_j (B·,j⊙M·,j)²    # ∂L/∂α=0
        B = sign(W-μ)
    return α·B + μ
```
二阶 ARB（salient weights）：Ŵ = α₁B₁+α₂B₂+μ，更新 α₁,α₂ 后对 {±α₁±α₂} 四候选 binary search 最近邻确定 B₁,B₂。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/ZHITENGLI/ARB-LLM。T=15 参数充分收敛，block_size=128。嵌入 BiLLM 框架：salient 用二阶 ARB，non-salient 用一阶 ARB。LLaMA-7B 上 ARB-RC 1 轮迭代即 ppl=15.23（BiLLM: 49.79），15 轮 ppl=14.03。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models

---

## Row-Column Binarization (ARB-RC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ARB-RC 是 ARB-LLM 中针对 LLM 权重列间偏差设计的双轴缩放二值化方法。传统二值化仅用 row-wise 参数（α^r 和 μ），无法处理 LLM 中某些列值显著大于其他列的现象。ARB-RC 引入 column-wise scaling factor α^c，模型为 Ŵ = α^r·α^c·B（无 μ）。移除 μ 节省存储同时 α^c 保留列间偏差。初始值：α^r_i = mean(|W_i|), α^c_j = mean(|W_{·,j}/α^r|)。交替更新：α^r = diag(W(α^cB)^T)/diag((α^cB)(α^cB)^T)，α^c = diag(W^T(α^rB))/diag((α^rB)^T(α^rB))。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# init: α^r = mean(|W|, dim=1); α^c = mean(|W|/α^r, dim=0); B = sign(W)
for iter in 1..T:
    # update α^r: A = α^c.unsqueeze(0) * B  → α^r = diag(W @ A^T) / diag(A @ A^T)
    # update α^c: A = α^r.unsqueeze(1) * B  → α^c = diag(W^T @ A) / sum(A², dim=0)
    B = sign(W)   # ARB-RC 无 μ
# Ŵ[i,j] = α^r[i] · α^c[j] · B[i,j]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ARB-RC 优于 ARB-X：LLaMA-7B ppl 14.03 vs 21.81，且存储更少（2.63GB vs 2.93GB raw bitmap）。ARB-RC 不引入 calibration data（L₂ 优化导致参数耦合），仅用 L₁ 优化。+ CGB = ARB-LLM_RC 最终模型。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models

---

## Column-Group Bitmap (CGB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CGB 是 ARB-LLM 提出的权重分区精炼策略。BiLLM 使用 column bitmap C_s 标记 salient 列和 group bitmap G 标记 non-salient 权重的 sparse/concentrated 分组，但 salient 列的 G 区域闲置。CGB 将 salient 列也纳入 group bitmap：G_s = 1_n C_s^T ⊙ G，G_ns = 1_n C_ns^T ⊙ G，产生四个 zone：salient-sparse、salient-concentrated、non-salient-sparse、non-salient-concentrated，每组独立二值化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Hessian 敏感度 → C_s (salient column bitmap) → C_ns = ¬C_s
# G = (|W| > threshold)  (magnitude-based group bitmap)
# 四个 zone masks:
G_s  = repeat(C_s, n, 1) ⊙ G          # salient-sparse & salient-concentrated
G_ns = repeat(C_ns, n, 1) ⊙ G         # non-salient-sparse & non-salient-concentrated
# 对每个 zone 独立执行 ARB: zone_mask → ARB(W, zone_mask, T)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
扩展 group 从 2→4 进一步改善性能（ARB-LLM_X ppl 6.55），但额外存储 ~0.8GB/7B 模型。CGB 是 ARB-LLM_X 和 ARB-LLM_RC 的共同组件，相比 BiLLM bitmap 有 consistent 提升。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models

---

## IR-QLoRA (Information Retention QLoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IR-QLoRA 是一种面向 LoRA-finetuning 量化的精确 LLM 压缩框架，从统一的信息保留视角提出两个核心技术：(1) **ICQ (Information Calibration Quantization)**：通过最大化量化权重信息熵的统计校准量化，使量化后的 LLM 权重尽可能保留原始参数信息；(2) **IEC (Information Elastic Connection)**：通过参数自由的弹性连接增强 LoRA 的表征多样性，使 LoRA 能够直接利用原始输入信息。IR-QLoRA 在 LLaMA/LLaMA2 系列的 2-4 bit 量化下均实现显著精度提升，且仅增加 ~0.31-0.46% 的训练时间开销。IEC 的额外参数可完全合并入 LoRA 矩阵，无推理开销。框架兼容 NormalFloat 和 Integer 两种量化格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IR-QLoRA 的完整 forward pass（以 NF4 量化的 linear 层为例）：
```
# === 离线 PTQ 阶段（执行一次，结果可缓存）===
for each block in weights with blocksize B=64:
    τ₀ = median(w_block)                               # 初始化 calibration constant
    H* = 0
    for τ in Linspace(τ₀ - 0.1, τ₀ + 0.1, step=0.001): # 200 候选
        w' = w_block - τ
        ŵ = NF4(w' / absmax(w'))                       # NormalFloat 4-bit 量化
        H = -Σ P(q_i) · log₂(P(q_i))                   # 信息熵
        if H > H*: τ* = τ; H* = H                       # 选最大熵
    # Double-quantize τ* 和 scale factor s
    τ₁^FP8 = FP8(τ* / absmax(τ*)), τ₂^FP16 = absmax(τ*)
    s₁^FP8  = FP8(s / absmax(s)),  s₂^FP16  = absmax(s)

# === 推理/微调阶段 ===
# ICQ 量化前向
ŵ^FP16 = NF4((w - τ*) / absmax(w - τ*)) · s₁ · s₂ + τ₁ · τ₂
y'_ICQ = x · ŵ^FP16

# IEC 增强的 LoRA 前向
# U₁: 输入 x (h维) → 低秩中间表示 (r维)
x₁ = x · ℓ₁                                              # 标准 LoRA 左矩阵
x₂ = β₁ · (r/h) · mean_pool(x, segments=r)               # IEC: 输入分组平均
x' = x₁ + x₂                                              # 弹性连接融合
# U₂: 低秩中间表示 (r维) → 输出 (o维)
y₁ = x' · ℓ₂                                             # 标准 LoRA 右矩阵
y₂ = β₂ · repeat(x', o/r times)                           # IEC: 重复拼接
U_IEC(x) = y₁ + y₂

# 最终输出
y = y'_ICQ + α · U_IEC(x)
```
其中 IEC 参数 β₁, β₂ 可在推理时通过矩阵数学合并入 ℓ₁, ℓ₂ 以消除全部推理开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/htqin/ir-qlora。基于 QLoRA 官方代码（HuggingFace Transformers + PEFT）修改。ICQ 在模型的 `prepare_model_for_kbit_training` 阶段执行搜索，IEC 通过修改 PEFT 中 LoRA 层的 `forward` 函数实现。使用时在 QLoRA 或 QA-LoRA 的标准微调流程前插入 ICQ 搜索步骤即可。ICQ 搜索 [τ₀-0.1, τ₀+0.1] 区间 200 个候选值，选择信息熵最大的 τ*。β₁, β₂ 作为可训练参数与 LoRA 权重一起被 AdamW 优化。默认超参数：λ=0.1（搜索范围系数）, n=100（搜索密度）, LoRA r=64, α=16, dropout=0.1, lr=2e-4, batch_size=16。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---

## Information Calibration Quantization (ICQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ICQ 是 IR-QLoRA 提出的基于信息熵最大化的量化校准技术。传统的对称量化（如 QLoRA 的 NormalFloat）零点固定为零，量化权重熵未经优化，导致与原始权重的互信息（mutual information）不足。ICQ 引入可学习的 calibration constant τ，将量化公式从 ŵ = NFk(w/s) 改为 ŵ = NFk((w-τ)/s)，并通过搜索最大化量化权重信息熵 H(ŵ) = -Σ P(q_i) log₂ P(q_i) 来确定最优 τ*。由于 PTQ 中原始权重 w 固定，最大互信息等价于最大熵。ICQ 能有效提升量化权重的信息表示能力，4-bit LLaMA-7B 的权重熵从 3.67 提升到 3.74，无需微调即可在 MMLU 上带来 0.5% 的精度提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ICQ 对每个权重块的搜索过程：
```
Input: Block weight w ∈ ℝ^64, λ=0.1, n=100, σ=1
Output: τ₁^FP8, τ₂^FP16

1. τ₀ = quantile_0.5(w)                     # 用中位数初始化（正态分布对称轴）
2. H* = 0
3. for τ in Linspace(τ₀ - λσ, τ₀ + λσ, 2n+1):  # 201 个候选值均匀采样
4.     w̃ = w - τ
5.     s = absmax(w̃)                           # scale factor
6.     ŵ = NF4(w̃ / s)                          # NormalFloat 4-bit 量化
7.     Calculate P(q_i) = count(ŵ == q_i) / len(w̃) for i=0..15
8.     H = -Σ_{i=0}^{15} P(q_i) · log₂ P(q_i)
9.     if H > H*:
10.        τ* = τ, H* = H
11. τ₁^FP8 = FP8(τ* / absmax(τ*))              # double-quantize τ*
12. τ₂^FP16 = absmax(τ*)
```
τ₀ 使用中位数而非均值初始化：正态分布在对称轴附近密度最高，中位数使更多数据落入量化 bin 的中心区域。搜索范围 [τ₀-0.1σ, τ₀+0.1σ] 覆盖 ~95% 情况下的最优解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ICQ 在 PTQ 阶段作为一次性预处理步骤执行，结果可缓存用于后续多次微调。对每个量化 block（默认 block_size=64）独立执行搜索，τ* 与 scale factor s 一起执行 double quantization 以控制存储开销（仅增加约 2% 参数）。ICQ 兼容 NormalFloat 和 Integer 量化：在 Integer 量化中，τ 相当于 zero point，可与原有 zero point 合并实现零开销。在 IR-QLoRA 代码中，ICQ 集成在模型加载后的 `prepare_model_for_kbit_training` 阶段。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---

## Information Elastic Connection (IEC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IEC 是 IR-QLoRA 提出的增强 LoRA 信息表征能力的参数自由连接技术。标准 LoRA 中 ℓ₂ 矩阵仅能使用 ℓ₁ 的低秩变换结果，无法直接访问原始输入 x 的信息，且变换形式局限于矩阵乘法（同质化）。IEC 通过两个参数自由操作解决：(1) **U₁ 弹性下采样**：将原始输入 x 按 (r/h) 比例分组平均后加到 ℓ₁ 输出，使低秩变换能融合原始输入信息；(2) **U₂ 弹性上采样**：将中间表示 x' 重复拼接 (o/r) 次后加到 ℓ₂ 输出，引入参数无关的多样化变换。IEC 每层仅引入 2 个可学习标量 β₁, β₂，且可通过矩阵数学合并入 ℓ₁, ℓ₂ 消除推理开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 h=4096, r=64, o=4096 为例（LLaMA-7B attention 投影层）：
```
# U₁: 输入 x ∈ ℝ^{b×4096} → 中间表示 x' ∈ ℝ^{b×64}
# 步骤 1: 标准 LoRA 左矩阵变换
x₁ = x · ℓ₁                          # [b, 4096] × [4096, 64] = [b, 64]

# 步骤 2: IEC 分组平均（parameter-free）
# 将 x 分成 r=64 组，每组 4096/64 = 64 维，按组求均值
x_split = reshape(x, [b, 64, 64])    # 64 groups of 64 dims
x_pool = mean(x_split, dim=-1)       # [b, 64], group-wise average
x₂ = β₁ · (64/4096) · x_pool         # [b, 64], scaled by learnable β₁

# 步骤 3: 弹性融合
x' = x₁ + x₂                         # [b, 64], IEC-enhanced intermediate

# U₂: 中间表示 x' ∈ ℝ^{b×64} → 输出 ∈ ℝ^{b×4096}
# 步骤 4: 标准 LoRA 右矩阵变换
y₁ = x' · ℓ₂                         # [b, 64] × [64, 4096] = [b, 4096]

# 步骤 5: IEC 重复拼接（parameter-free）
y₂ = β₂ · repeat(x', times=64)       # [b, 64] × 64 concat = [b, 4096]

# 步骤 6: 弹性融合
U_IEC(x) = y₁ + y₂

# 推理时合并（消除开销）：
# ℓ̃₁_{i,j} = ℓ₁_{i,j} + β₁·gcd(h,r)/h  if floor(i/(h/gcd)) == floor(j/(r/gcd))
# ℓ̃₂_{i,j} = ℓ₂_{i,j} + β₂·gcd(o,r)/r  if floor(i/(r/gcd)) == floor(j/(o/gcd))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
IEC 通过修改 PEFT 库中 LoRA 层的 `forward` 函数实现，对每个使用 LoRA 的线性层自动附加 IEC 连接。β₁, β₂ 初始化为小值（约 0.01），在微调过程中与 LoRA 权重一同训练。推理时执行矩阵合并（Eq. 16-17），使得 IEC 完全零开销。IEC 独立于量化方法，可与 QLoRA、QA-LoRA 等任意 LoRA-finetuning 量化框架直接结合。消融实验表明 IEC 单独带来 1.8% MMLU 提升（4-bit LLaMA-7B），其中 U₁ 贡献 1.0%，U₂ 贡献 1.3%。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---

## LoRA-finetuning Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA-finetuning quantization 是将 LLM 量化与低秩适配（LoRA）参数高效微调相结合的两阶段范式：(1) **PTQ 阶段**：对预训练 LLM 执行后训练量化（如 NormalFloat、GPTQ），将 FP16 权重压缩至 2-8 bit，获得量化器参数（scale factors, zero points）；(2) **LoRA 微调阶段**：在量化后 LLM 的线性层上附加低秩适配器（LoRA），冻结量化权重不动，仅训练 LoRA 的 ℓ₁, ℓ₂ 矩阵和少量参数。相比全模型微调，该范式大幅降低训练显存和时间；相比纯 PTQ，LoRA 微调可恢复量化造成的精度损失。代表方法：QLoRA（Dettmers 2023）、QA-LoRA（Xu 2023）、LoftQ（Li 2023）、IR-QLoRA（本文）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Phase 1: PTQ（离线，执行一次）
for each linear_layer in LLM:
    w_block = split_by_block(layer.weight, block_size=64)
    for block in w_block:
        s = absmax(block)                               # scale factor
        w_quant[block] = NF4(block / s)                 # NormalFloat 4-bit 量化
        s_FP8, s_FP16 = double_quantize(s)             # 双重量化 scale

# Phase 2: LoRA 附加
for each quantized_linear_layer:
    layer.lora_A = nn.Linear(h, r, bias=False)          # ℓ₁: down-project
    layer.lora_B = nn.Linear(r, o, bias=False)          # ℓ₂: up-project
    init: lora_B.weight = 0                              # 零初始化保证训练起始不变

# Phase 3: 微调（仅更新 LoRA）
for batch in dataset:
    y = quantized_linear(x) + α · lora_B(lora_A(x))     # FP16 计算
    loss = cross_entropy(y, target)
    loss.backward()  # 梯度仅流经 LoRA，量化权重保持冻结
    update(lora_A, lora_B)

# Phase 4: 推理部署（可选合并）
W_merged = dequant(w_quant) + α · ℓ₂^T · ℓ₁^T         # 合并 LoRA 到量化权重
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QLoRA (Dettmers 2023) 为该范式的奠基工作，开源代码在 https://github.com/artidoro/qlora。基于 HuggingFace Transformers + PEFT + bitsandbytes 实现。使用方式：通过 `BitsAndBytesConfig` 设置 NF4 量化 → `prepare_model_for_kbit_training()` → PEFT `LoraConfig` 添加 LoRA → 标准 Trainer 微调。关键配置：block_size=64（量化块大小）, LoRA rank r=64（适配低秩维度）, α=16（LoRA 缩放系数）。QLoRA 可在单张 48GB GPU 上微调 65B 模型（4-bit 量化下仅需 ~18GB 显存）。后续改进包括 QA-LoRA（integer 量化+量化感知 LoRA）、LoftQ（交替量化-LoRA 初始化）、IR-QLoRA（信息保留）。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models

QA-LoRA 对 LoRA 进行了重要修改以适配量化场景：标准 LoRA 中 A ∈ R^{D_in × D_int}，B ∈ R^{D_int × D_out} 均无约束；QA-LoRA 通过组内求和聚合操作 QA(x) 将输入维度从 D_in 降至 L（L = D_in/g，g 为量化组大小），使 A 矩阵尺寸缩减为 L × D_int。这一约束使 A 的行向量在量化组内共享，从而满足合并后权重仍可表示为 group-wise INT 量化格式的数学条件。

---

## Double Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Double Quantization 是 QLoRA (Dettmers 2023) 引入的量化常量压缩技术。在 block-wise 量化中，每个 block 需要存储一个 scale factor（FP32），以 block_size=64 为例，scale 相当于每参数 32/64 = 0.5 bit 的额外开销。Double Quantization 对第一层量化产生的 scale factors 执行第二次量化：将 scale factors 按 block_size=256 分组，每组量化为 8-bit FP8，再引入第二层 scale factor c₂^FP16。这使 scale 的每参数开销从 0.5 bit 降至 8/64 + 32/(64×256) ≈ 0.127 bit，减少约 75%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 第一层量化：权重 → 4-bit NF4
for each block w_i of size 64:
    s_i = absmax(w_i)                           # FP32 scale factor (first level)
    w_quant_i = NF4(w_i / s_i)                  # 4-bit quantized weights

# 第二层量化（Double Quant）：scale factors → 8-bit FP8
# 每 256 个 s_i 组成一组
for each group of 256 scale factors {s_1,...,s_256}:
    c₂ = absmax({s_1,...,s_256})               # FP16 second-level scale
    s_FP8_i = FP8(s_i / c₂)                     # 8-bit quantized scales

# 存储内容（块 i）:
# - w_quant_i: 64 × 4 bit = 256 bit
# - s_FP8_i: 8 bit
# - 对应的 c₂：每 256 块共享一个 FP16 = 16 bit
# 每块总开销 = 256 + 8 + 16/256 ≈ 264 bit
# 相比 FP16 (64×16=1024 bit) 压缩近 4x
```
在 IR-QLoRA 中，Double Quantization 同样应用于 ICQ 引入的 τ*，与 scale factor 执行相同的 FP8 量化以减少存储开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
bitsandbytes 库实现 Double Quantization：`bnb_4bit_use_double_quant=True`。在 HuggingFace 中使用：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)`。内存节省：对于 65B 模型，Double Quantization 单独节省约 3GB 显存。在 llama.cpp 社区中，类似的 super-block 方法独立探索了相同思想。在 BOF4 论文中，作者未使用 Double Quantization（即未进一步量化 quantization constants），因为 signed normalization（BOF4-S）会额外需要一个 sign bit 来编码归一化常数的符号，可能与 Double Quantization 不兼容。

SpQR 推广 Double Quantization 为**双层量化（Bilevel Quantization）**：核心差异在于 (1) 第一层：权重按极小 group（β₁=8~16，远小于 QLoRA 的 64）分组，每组独立计算 3-bit scale s_q 和 zero-point z_q；(2) 第二层：将 s_q 和 z_q 分别按 β₂=16 分组，以 3-bit 格式量化，并引入第三层 16-bit statistics s_s（scale of scales）、z_s（zero of scales）、s_z（scale of zeros）、z_z（zero of zeros）；(3) 总统计量开销 = (b_s+b_z)/β₁ + 64/(β₁β₂)，例如 β₁=16, β₂=32, b_s=b_z=3 时仅 0.5 bits/param；(4) 优化：去除"max>0, min<0"约束，允许全正/全负 group 使用非整数零点。与 QLoRA 的 Double Quantization（仅对 scale 做一次 FP8 量化）不同，SpQR 的双层量化在更小 group 上同时对 scale 和 zero-point 做二次量化，使极小 group 的存储开销可控。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

---

## Quantile Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantile Quantization（分位数量化）是一种信息论最优的量化策略，确保每个量化 bin 包含相等数量的来自输入张量的值，即量化后的值均匀分布。其核心思想：如果权重近似服从正态分布 N(0,σ)，则将正态分布的 CDF 反函数（分位数函数 Q_X）作用于等间距的概率值，得到的量化级别 q_i = (Q_X(i/(2^k+1)) + Q_X((i+1)/(2^k+1))) / 2 能使每个 bin 具有相等的概率质量。相比均匀量化（bin 等距），分位数量化在零附近密度更高（捕获大部分权重），在尾部间距更大，信息利用率更高。QLoRA 的 NormalFloat (NF) 数据类型就是分位数量化的具体实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# k-bit Quantile Quantization 的级别生成
from scipy.stats import norm

def quantile_quantization_levels(k):
    num_levels = 2^k
    # 等概率间隔（0到1之间num_levels+1个边界）
    probs = np.linspace(0, 1, num_levels + 1)
    # 每个 bin 的中点分位数
    mid_probs = (probs[:-1] + probs[1:]) / 2
    # 正态分布分位数函数（CDF反函数）
    levels = norm.ppf(mid_probs)
    # 归一化到 [-1, 1]
    levels = levels / max(abs(levels))
    return levels

# 4-bit: q = [-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091,
#              0.0, 0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
# 特点：零点附近最密集（8 values in [-0.1, 0.1]），远离零点逐渐稀疏
```
在 IR-QLoRA 论文中，NF2/NF3/NF4 均使用分位数量化生成量化级别，NF2 使用对称设置（避免信息过度偏离）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
bitsandbytes 库实现了基于分位数量化的 NF4 格式。使用 HuggingFace 的 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` 即可加载分位数量化模型。在 PyTorch 中，可自定义分位数量化器：对每块权重，使用 absmax scaling 将值归一化到 [-1, 1] → 匹配到最近的分位数级别 → 存储级别索引。分位数量化的理论最优性仅对正态分布严格成立；当实际权重分布偏态时，需考虑非对称扩展（如 AFPQ 的非对称 NF）。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---

## LoRA (Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA (Low-Rank Adaptation) 由 Hu 等人 (2021) 提出，是一种参数高效微调（PEFT）方法。核心思想：预训练模型的权重更新 ΔW 在实际自适应中具有内在低秩属性，因此可以在冻结的预训练权重 W₀ 旁添加低秩分解矩阵 BA（B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)），仅训练 A 和 B。前向过程：h = W₀x + α·BAx，其中 A 使用 Kaiming 初始化，B 使用零初始化（训练开始时 ΔW=0，不改变原始输出）。训练后将 ΔW 与 W₀ 合并（W_merged = W₀ + α·BA），推理无额外开销。LoRA 使全参数微调所需显存降低至千分之一级别（如 7B 模型仅需几 MB 额外存储），且多任务间可热切换 adapter。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LoRA 初始化
W = load_pretrained_weight()           # [d, k] 冻结
A = nn.Parameter(torch.randn(r, k))    # [r, k] Kaiming uniform
B = nn.Parameter(torch.zeros(d, r))    # [d, r] zero init
α = 16  # scaling factor

# 前向
def forward(x):
    h_base = x @ W.T                    # 冻结权重前向
    h_lora = (x @ A.T) @ B.T           # 低秩路径，r << min(d,k)
    return h_base + (α / r) * h_lora   # α/r 缩放

# 微调时仅更新 A, B
optimizer = AdamW([A, B], lr=2e-4)

# 推理时合并（可选）
W_merged = W + (α / r) * (B @ A)       # [d, k], 合并后无额外推理开销
```
对于 LLaMA-7B，典型配置：r=64, α=16，所有 linear 层均添加 LoRA，可训练参数 ~0.1% 的全模型参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace PEFT 库提供标准 LoRA 实现：`LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj", "v_proj", ...])` → `get_peft_model(model, config)`。开源实现：https://github.com/huggingface/peft。在 LLM 领域通过 transformers `load_adapter()` 加载，切换开销 < 1ms（Llama3-1B）。LoRA 变体包括 DoRA（动态 rank 分配）、AdaLoRA（自适应 layer-wise rank）、QLoRA（4-bit 量化+LoRA）、HydraLoRA（非对称架构）。在扩散模型领域，LoRA 权重通常以 .safetensors 格式分发。IR-QLoRA 论文在 QLoRA 的基础上通过 IEC 技术进一步增强了 LoRA 的信息利用能力。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

QeRL (Huang et al., NVIDIA, 2025) 将 LoRA 用于 RL 训练 (GRPO/DAPO)：在 Q/K/V/O/gate/up/down 层添加 LoRA adapter（rank=16/32/64/128），仅训练约 1% 参数。关键发现：LoRA BF16 在 RL 中收敛慢（BigMath 上需 500+ steps 才见 reward 上升），对高学习率敏感（LR>5e-6 训练崩溃），而 NVFP4 量化后 LoRA 因量化噪声增熵收敛更快（~200 steps），且可承受 1e-5 学习率。Rollout 阶段 QeRL 的 NVFP4 量化权重 + Marlin kernel 使 LoRA 模型仅 5.9GB（7B），比 BF16 LoRA (15.2GB) 省 ~61% 显存。

## Equivalent Transformation in Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Equivalent Transformation（等价变换）是 LLM 后训练量化中的核心技术手段。它指在保证矩阵乘法输出不变的前提下，对权重矩阵 W 和激活 X 分别施加互逆的变换，使得变换后的权重分布更有利于量化。数学形式为：XW = (XT⁻¹)(TW)，其中 T 为可逆变换矩阵。常见的等价变换包括：(1) 缩放变换（Scaling）：T = diag(s)，即每通道乘以独立的标量——AWQ 和 SmoothQuant 均属此类；(2) 平移变换（Translation/Shift）：对激活 X 减去一个偏置 δ 并在 bias 上加回 δW，对应 Outlier Suppression+ 的设计；(3) 重排变换（Reordering）：T 为置换矩阵（每行/列仅一个 1），如 RPTQ 的 per-cluster 列重排；(4) 仿射变换（Affine Transformation）：AffineQuant 提出，T 为任意 d×d 可逆矩阵，泛化以上所有等价变换。等价变换的核心优势：(a) 保证输出一致性——量化噪声仅通过等效变换参数间接感知；(b) 不同类型等价变换之间正交，叠加使用可进一步扩展优化空间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以此论文的 LLaMA2-7B Linear 层为例，三种等价变换的伪代码对比：
```
# (1) 缩放变换 (AWQ / SmoothQuant)
s = compute_activation_aware_scale(X)  # per-channel scale, d-dim
W' = W * diag(s)                       # 权重按 output channel 缩放
X' = X * diag(1/s)                     # 激活按 input channel 逆缩放
Y = Q(W') @ X' == W @ X                # 等价保证

# (2) 平移变换 (Outlier Suppression+)
δ = learnable_shift_vector            # per-channel shift, d-dim
X' = X - δ                            # 激活减去 shift
b' = b + δ @ W                        # bias 加回 δW
Y = X' @ W + b' == X @ W + b          # 等价保证

# (3) 仿射变换 (AffineQuant) — 泛化以上所有
A = learnable_matrix(d, d)            # d×d 可逆矩阵
W' = Q(A @ W)                         # 权重左乘 A 后量化
X' = (X - δ) @ A^{-1}                 # 激活平移后右乘 A^{-1}
b' = b + δ @ W                        # bias 平移补偿
Y = X' @ W' + b' == X @ W + b         # 等价保证

# (4) 成对旋转变换 (ParoQuant) — sparse Givens 旋转, O(n) 参数
# 变换: T(W) = (∏_{t=1}^{K} R(P_t, Θ_t)) · diag(α) · W
# 每个 R(P_t, Θ_t) 是 independent rotation (不重叠 Givens 旋转的乘积)
# 推理: X' = X · diag(1/α) · R_1^{-1} · ... · R_K^{-1}
# 参数: 旋转角度 Θ（可学习）+ 缩放因子 α（可学习）
# 特点: 参数少 (O(Kn) vs O(n²)), 可在线计算 (~10% 开销), 所有线性层通用
```
当 A = diag(s) 时退化为缩放变换；当 A = I 且仅优化 δ 时退化为纯平移变换；当 A 为置换矩阵时退化为重排变换。AffineQuant 的真正创新在于 A 的所有 d² 个元素均可学习，极大扩展了等价变换的优化空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
等价变换在 PTQ 中的实现方式：(1) 选择等价变换类型（scale/shift/affine/reorder），通常可组合使用；(2) 在校准集上用梯度下降最小化 block 输出 MSE 来优化变换参数；(3) 优化完成后，将变换参数数学合并入相邻层——如 scale/s 合并入 LayerNorm weight，affine matrix/A 合并入 weight 本身（W' = A@W），shift δ 合并入 bias，确保推理零额外开销。AffineQuant 是等价变换思想的终极扩展：同时使用仿射矩阵 A（旋转+缩放）和 shift δ（平移），通过 block-wise MSE loss 联合优化，Lever-Desplanques 定理和 Gradual Mask 保证 d² 参数量下的稳定优化。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers
- AffineQuant Affine Transformation Quantization for Large Language Models
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

aespa 论文使用 Z-FOLD 的 foldable parameters 机制进行量化参数计算（scale/zero-point 初始化）。Foldable parameters 指可数学合并入其他层而不增加推理开销的额外参数（如 LayerNorm 的 affine weight），本质上是等价变换的一种实现形式。aespa 的关键创新在于：在 Z-FOLD 的 foldable parameter 优化中，将标准 Hessian H=2E[XX^T] 替换为 attention-aware Hessian H_V=2E[XA^TAX^T]，使等价变换的优化目标携带跨层依赖信息。

ParoQuant 将等价变换的类型扩展到 **成对旋转变换 (Pairwise Rotation)**——一种稀疏参数化的正交变换。与 AffineQuant 的稠密仿射矩阵 A（d² 参数）不同，ParoQuant 使用 O(Kn) 个 Givens 旋转参数（n 个通道 × K 个 independent rotation），实现了参数效率与表达能力之间的平衡。其逆变换在推理时在线计算（fused CUDA kernel，~10% 开销），而不是像 scaling/shift 那样完全合并到前序算子中。这使得旋转变换能应用于所有线性层（包括前有 element-wise 算子的层），克服了 SpinQuant 旋转合并策略的适用范围限制。

**OmniQuant 的 Learnable Equivalent Transformation (LET)** 是对等效变换思想的差异化实现：

(1) **变换类型组合**：LET 同时使用通道级缩放 s ∈ R^{1×Cin} 和平移 δ ∈ R^{1×Cin}（Eq.3: Y = XW + B = [(X-δ)⊘s]·[s⊙W] + [B+δW]）。与 SmoothQuant（仅预定义缩放）和 OS+（grid-search 缩放 + 预定义平移）不同，所有参数通过梯度下降在 block-wise 误差最小化框架下联合优化。初始化使用 SmoothQuant 的 scaling 和 OS+ 的 shifting 作为 warm start（Table A6 消融表明 scaling 初始化比 shifting 初始化更重要）。

(2) **扩展到 Attention**：LET 不仅应用于线性层，还扩展到 attention 的 Q/K 矩阵乘法（Eq.5: P = Softmax(QK^T) = Softmax((Q⊘s_a)(s_a⊙K^T))），使 KV cache 也可被量化。这是 OmniQuant 相比 SmoothQuant/AWQ/OS+ 的关键扩展（Table A4 消融：移除 attention LET 使 W4A4 PPL 从 10.87 升至 11.34）。

(3) **应用位置**：四对 LET 分别为 [ln1, (q_proj,k_proj,v_proj)]、[v_proj,out_proj]、[Q,K]、[ln2,fc1]。第二层 FFN（fc2）除外，因为 non-linear 层后的高稀疏性特征导致梯度不稳定。

(4) **与 LWC 的协同**：LET 将激活 outlier 迁移到权重（增加 weight quantization 难度）→ LWC 专门降低 weight quantization 难度，形成"迁移→消解"递进关系。Figure A2 可视化：原始激活 outlier 幅值约 70×，SmoothQuant 后降至 2×（仍有明显 gap），LET 后 outlier 与 regular channel 几乎无差异，证明梯度优化的 LET 比手工 heuristic 更彻底地均衡了激活分布。

**PassionSR 的 Learnable Equivalent Transformation (LET)** 是将相同变换思想应用于扩散模型领域的实现：
(1) **应用领域差异**：PassionSR LET 针对 UNet+VAE 的 one-step diffusion SR 模型，覆盖 Conv 层（扩散 UNet 的主要算子）、Linear 层和 Attention 的 Q/K/V 矩阵乘法。与 OmniQuant LET 仅覆盖 LLM 的 Linear/Attention 不同，Conv 层的等效变换是 PassionSR 的独特需求——沿 channel 维应用 scale/shift，变换后归入前层或权重。
(2) **训练策略差异**：PassionSR LET 通过两阶段 DQC 训练：Stage 1 冻结 LBQ 仅训练 LET 的 s 和 δ；Stage 2 重新初始化 LBQ 后联合训练。这种解耦策略是 PassionSR 的独特贡献——OmniQuant LET 和 LWC 是联合训练但通过不同损失函数（block-wise MSE vs 统一 loss）实现稳定。
(3) **效果**：W6A6 下 LET 使 PSNR 提升 2.25 dB（LBQ-only 23.15 → LBQ+LET 25.40 on RealSR），激活分布可视化显示离群值大幅减少。

---

## Affine Transformation Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Affine Transformation Quantization（仿射变换量化）是 AffineQuant 论文提出的 PTQ 等价变换方法，用一个完整的 d×d 可逆仿射矩阵 A 替代传统方法中受限的对角缩放向量 s。核心公式为 argmin_A ||XW - XA⁻¹Q(AW)||²_F。其物理含义：对权重矩阵 W 的每个 output channel（d 维向量），仿射变换 A 对其进行旋转和缩放的任意组合，使其更好地对齐到量化器的 2ⁿ-1 个固定点上。相比于缩放变换（仅统一拉伸/压缩每个 channel 的各维度）和平移变换（仅整体平移），仿射变换可以改变 channel 内各维度的相对数值关系，实现最大化的等价变换优化空间。论文证明，当 A 退化为对角矩阵时，该方法等价于 OmniQuant/SmoothQuant/AWQ；当 A 为置换矩阵时，等价于 RPTQ 的重排。同时引入 learnable shift δ，与仿射变换正交叠加。为在有限校准数据下稳定优化 d² 自由参数，基于 Levy-Desplanques 定理提出 Gradual Mask 机制确保 A 在优化中始终保持可逆。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA2-7B qkv_proj 层（d=4096）为例：
```
# 初始化
A = torch.eye(4096) * smoothquant_scale  # 对角初始化，严格对角占优
δ = torch.zeros(4096)                     # shift 初始化

for epoch in range(20):
    # Gradual Mask (Eq. 6)
    radius = int((epoch+1) / 20 * 4096)
    GM = torch.zeros(4096, 4096)
    GM.fill_diagonal_(1.0)
    for d in range(1, radius+1):
        GM.diagonal(d).fill_(α)   # α=0.01 for low-bit
        GM.diagonal(-d).fill_(α)
    
    # 前向 (Eq. 7)
    A_star = A * GM                     # Hadamard 积
    A_inv = torch.linalg.inv(A_star)    # GM 保证 A* 严格对角占优
    X_t = (X - δ) @ A_inv.T            # 激活: 平移 + 右乘 A⁻¹
    W_t = quantize(A_star @ W, 4bit)   # 权重: 左乘 A + INT4 量化
    bias_t = bias + δ @ W              # bias: 平移补偿
    
    loss = MSE(block(X_t, W_t, bias_t), block_fp(X, W, bias))
    loss.backward()                     # GM 抑制非对角线更新
    optimizer.step()

# 推理时合并
W_final = quantize(A_final @ W, 4bit)          # A 合并入权重
bias_final = bias + δ_final @ W_final          # δ 合并入 bias
ln_weight *= diag(A_ln)                         # LayerNorm 对角的 A 合并
```
关键结果：LLaMA2-7B w4a4 C4 PPL 15.76（OmniQuant 18.02, ↓2.26）；LLaMA-30B w4a4 zero-shot avg 58.61%（OmniQuant 56.63%, ↑1.98%）；均无额外推理开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/bytedance/AffineQuant。基于 PyTorch + HuggingFace，修改自 OmniQuant。关键实现细节：(1) 对每个 transformer block 内的 qkv_proj、out_proj、fc1、fc2 分别维护 A 和 δ，但 fc1-fc2 之间因 GELU/SiLU 使等价变换失效而排除仿射变换；(2) attention 模块在每个 head 内独立应用 A（每 head dim d_head=128）；(3) 矩阵求逆使用 PyTorch linalg.inv，float-double 混合精度（模型 float + A 矩阵 double）在误差和资源间平衡最佳；(4) 优化超参（lr、epoch、clipping）对齐 OmniQuant。稳定性因子 α 的选择：小模型（≤6.7B）α=1；大模型且 ≥3-bit α=1e-2；低比特 α∈{1e-2, 1e-3, 1e-4}。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---

## Levy-Desplanques Theorem

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Levy-Desplanques Theorem 是线性代数中的一个经典定理，由 Naimark & Zeheb (1997) 进行了扩展。定理的核心陈述：所有严格对角占优矩阵（Strictly Diagonally Dominant Matrix）都是可逆的（非奇异的）。严格对角占优的定义为：对矩阵 A 的每一行 i，对角线元素的绝对值严格大于该行其他所有元素绝对值之和：|a_ii| > Σ_{j≠i} |a_ij| for all i。定理直观含义：如果矩阵的对角线元素在各自行中占据足够主导的地位，则该矩阵必然满秩。在 AffineQuant 中，该定理被用作保证仿射变换矩阵 A 在优化过程中始终保持可逆的理论基础——作者将 A 初始化为对角矩阵（天然严格对角占优、必然可逆），并通过 Gradual Mask 在优化过程中抑制非对角线元素的幅度和更新速率，使 A 持续满足严格对角占优条件，从而保证每一步的 A⁻¹ 计算有效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 AffineQuant 优化过程中，Levy-Desplanques 定理的应用链条为：
```
初始化：A₀ = diag(s₁,...,s_d) → 对任意行 i: |a_ii| = |s_i| > 0 = Σ_{j≠i} |a_ij|
       → A₀ 严格对角占优 → Levy-Desplanques → A₀ 可逆 ✓

优化中（epoch=e）：
  前向: A* = A_e ∘ GM → α<1 缩小非对角线 → |a*_ii| > Σ α|a*_ij| 更易满足
  反向: A_{e+1} = A_e + η·GM·∂L/∂A* → 非对角线更新被 α 抑制

理论保证（Theorem 1, Appendix A.2）：
  若 N_e 严格对角占优，且 α < |n_ii^0+ηΣ∂L/∂n_ii*| / (η·Σ|Σ∂L/∂n_ij*|)
  则 N_{e+1} 也严格对角占优 → Levy-Desplanques → N_{e+1} 可逆 ✓
```
即：只要 α 足够小，严格对角占优性质从初始状态通过整个优化过程向前传播，矩阵可逆性始终得到理论保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 AffineQuant 的具体实现中，Levy-Desplanques 定理主要通过 Gradual Mask 隐式使用——不显式检查对角占优条件。因为：(1) 对角初始化自动满足条件；(2) GM 逐步释放对角线附近元素，α<1 持续抑制非对角线幅度；(3) 论文的理论证明（Appendix A.2）提供了严格的 α 上界，实践中选择 α∈[1e-4, 1] 的经验值即可。该定理在更广泛的数值计算中常用于：保证 Jacobi/Gauss-Seidel 迭代收敛、分析矩阵条件数上界、偏微分方程有限差分解的稳定性分析。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---

## Strictly Diagonally Dominant Matrix

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Strictly Diagonally Dominant Matrix（严格对角占优矩阵）是矩阵的一种性质。定义：对 n×n 矩阵 A，若对每一行 i 满足 |a_ii| > Σ_{j≠i} |a_ij|（对角线元素的绝对值严格大于该行所有非对角线元素绝对值之和），则称 A 为严格对角占优矩阵。两个关键数学推论：(1) Levy-Desplanques 定理：严格对角占优矩阵必然可逆（行列式非零，非奇异）；(2) 存在性：严格对角占优矩阵的逆的无穷范数有界 ||A⁻¹||∞ ≤ 1 / min_i(|a_ii| - Σ_{j≠i}|a_ij|)，即条件数可控。在 AffineQuant 中，该矩阵性质被用作 Gradual Mask 设计的目标和验证标准——只要仿射矩阵 A 保持严格对角占优，其逆矩阵 A⁻¹ 必然存在且数值稳定，使得 XA⁻¹Q(AW) 的计算有效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 AffineQuant 上下文中，严格对角占优的演化过程：
```
初始状态（epoch=0, GM radius=0）：
A₀ = diag(s₁,...,s_d)       # 纯对角矩阵
行 i: |s_i| > 0 = Σ_{j≠i} 0  → 严格对角占优 ✓

优化中期（epoch=e, GM radius=r）：
Aₑ 的结构: 对角线=1附近、半径r内的非对角线=α倍值、r外=0
行 i: |a_ii| vs Σ_{j≠i,|i-j|≤r} |α·a_ij|
只要 α < min_i(|a_ii| / Σ_{|i-j|≤r}|a_ij|) → 严格对角占优 ✓

优化完成（epoch=f, radius=d）：
所有元素可学习，但 α 持续抑制非对角线幅度
论文实验（Appendix A.6）展示各 block 的 A 矩阵热力图：
所有矩阵均呈现严格对角占优模式（对角线亮，非对角线暗淡）
```
关键：Gradual Mask 通过两阶段机制维持严格对角占优——(1) 前向 Hadamard 积 A* = A ∘ GM 缩小非对角线元素；(2) 反向传播中 GM 调节非对角线梯度更新速率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在机器学习中，严格对角占优很少作为显式约束，但其导出性质（可逆性、条件数上界）在需要矩阵求逆的场合非常有价值。AffineQuant 的创新在于通过 GM 隐式实现了该约束，无需显式投影或条件检查。在数值线性代数中，严格对角占优矩阵的逆可使用 Jacobi 迭代高效近似，迭代保证收敛。更广泛地，在偏微分方程有限差分法、网络图分析（图 Laplacian）、Markov 链转移矩阵分析中，严格对角占优性质也有重要应用。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---

## Gradual Mask (GM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gradual Mask（GM，渐进掩码）是 AffineQuant 的核心优化稳定性机制，用于在有限校准数据下安全地优化 d×d 仿射变换矩阵 A（d 可达 4096+）。核心思想：(1) 在优化初期冻结 A 的所有非对角线元素（仅更新对角线），使模型从严格对角占优（对角矩阵）的安全起点出发；(2) 随着 epoch 推进，按 (e/total_epochs) × hidden_size 的半径逐步释放靠近对角线的元素参与优化；(3) 释放的非对角线元素乘以稳定性因子 α（<1），抑制其幅度和梯度更新率。GM 的数学定义（Eq. 6）：GM_ij = 1（i=j, 对角线），= α（0 < |i-j| ≤ e/t × hidden_size），= 0（otherwise）。GM 通过双重机制保可逆性：(a) 前向：A* = A ∘ GM，缩小非对角线幅度；(b) 反向：A_{e+1} = A_e + η·GM·∂L/∂A*，GM 调节非对角线学习率。实验证实 GM 是不可或缺的——无 GM 时 LLaMA-7B w2a16 训练崩溃（PPL=NaN），OPT-125M w3a16 PPL 从 32.10 恶化至 53.52。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 d=4096, total_epochs=20, α=0.01 为例的 GM 演化过程：
```
Epoch 1 (e=1): radius = 1/20 * 4096 ≈ 204
  可学习参数: 对角线 4096 + 2*(1+...+204)*α = 4096 + 41406*0.01 ≈ 4510
  实际效果: 几乎等价于 OmniQuant（仅对角线优化）

Epoch 5 (e=5): radius = 5/20 * 4096 = 1024
  可学习参数: 4096 + 2*(1+...+1024)*α = 4096 + 1047552*0.01 ≈ 14572

Epoch 10 (e=10): radius = 2048
  可学习参数: 4096 + 2*(1+...+2048)*α = 4096 + 4193280*0.01 ≈ 45627

Epoch 20 (e=20): radius = 4096
  可学习参数: 4096*4096 ≈ 16.8M（全部解冻）

伪代码：
def create_gradual_mask(d, epoch, total_epochs, alpha):
    radius = int(epoch / total_epochs * d)
    GM = torch.zeros(d, d)
    for i in range(d):
        GM[i, i] = 1.0                          # 对角线
        start = max(0, i - radius)
        end = min(d, i + radius + 1)
        for j in range(start, end):
            if i != j:
                GM[i, j] = alpha                # 非对角线近邻
    return GM
```
注意：attention 模块在每个 head 内独立应用 GM（每 head dim=128），而非在整个 qkv_proj 的 4096 维度上统一应用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中实现 GM 的方式：(1) 每 epoch 前生成 GM 张量（基于当前 epoch 和 α）；(2) 前向 `A_masked = A * GM`，然后 `A_inv = linalg.inv(A_masked)`；(3) GM 不在计算图中（无需对 GM 求梯度），梯度通过 A* 流回 A 时自动被 GM 缩放：`A.grad = GM * A_star.grad`。α 的选择策略：小模型（≤6.7B）α=1（自由更新，因矩阵维度小可天然稳定）；大模型高比特（≥3-bit）α=1e-2（适度抑制）；低比特 α∈{1e-2, 1e-3, 1e-4}（强抑制防崩溃）。当 α→0 时，GM 退化为仅更新对角线（等价于 OmniQuant）。该方法的创新类似 Adaround 中的 gradual β 控制（逐步释放舍入参数），但目的不同——Adaround 是防局部最优，GM 是保证矩阵可逆性。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---

## Weight-Activation Quantization (W4A4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-Activation Quantization（权重-激活联合量化，表示为 W4A4、W8A8 等）是在 LLM 量化中同时对模型权重和各层激活值（中间特征图）执行低比特量化的策略。与 Weight-only Quantization（仅量化权重，保持激活 FP16/BF16）相比，W4A4 的优势：(1) 计算可在低位宽整数域完成（INT4×INT4 GEMM），理论上比 W4A16（FP16 GEMM）计算量更小；(2) 激活的存储和带宽需求也减少 4×（FP16→INT4）。但挑战更大：(a) 激活分布依赖于输入数据和上下文，比权重更难预测，量化误差更难控制；(b) LLM 激活中存在大量 outlier channels——某些通道的激活幅度远超其他，直接量化导致严重信息丢失；(c) 激活量化参数需实时计算或离线统计。AffineQuant 对 W4A4 的改进：在 LayerNorm 后使用对角仿射矩阵 A（仅更新对角线的 A），将 A 合并入 LN weight/bias，对激活进行等价变换以降低量化难度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AffineQuant 中 W4A4 的完整前向传播流程：
```
for each transformer_block:
    # === Attention 部分 ===
    x_norm = LayerNorm(x)
    x_norm = x_norm * diag(A_attn)   # 对角 A 等价变换（可合并入 LN）
    x_norm_q = quantize(x_norm, 4bit) # 激活 INT4 量化
    
    # QKV 投影: INT4 weight × INT4 activation
    q = Q_linear(x_norm_q)  # W_qkv 已合并 A 的权重部分: Q(AW_qkv)
    k = K_linear(x_norm_q)
    v = V_linear(x_norm_q)
    
    attn_out = attention(q, k, v)   # FP16 attention（不量化 softmax）
    attn_q = quantize(attn_out, 4bit)
    o = O_linear(attn_q)            # Out proj: Q(AW_out)
    
    # === MLP 部分 ===
    x_res = x + o
    x_norm2 = LayerNorm(x_res) * diag(A_mlp)  # 对角 A（可合并入 LN）
    x_norm2_q = quantize(x_norm2, 4bit)
    
    gate = GELU(FC1(x_norm2_q))     # fc1: Q(AW_fc1), INT4×INT4
    # ⚠ fc1-fc2 之间不做仿射变换（GELU 使 XA⁻¹ 等价变换失效）
    y = FC2(gate)                    # fc2: Q(AW_fc2), FP16（gate 不量化）
    output = x_res + y
```
W4A4 下的关键结果：LLaMA2-7B C4 PPL 15.76（OmniQuant 18.02），WikiText2 PPL 12.69（OmniQuant 14.26）；LLaMA-30B zero-shot avg 58.61%（OmniQuant 56.63%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
W4A4 在 LLM 推理中的实际部署：(1) 离线 PTQ 阶段用 AffineQuant/SmoothQuant/OmniQuant 等方法优化权重和激活的量化参数；(2) 推理框架需实现 INT4×INT4 GEMM kernel——TensorRT-LLM 和 MLC-LLM 支持 W4A4 推理；(3) 激活量化需实时计算 scale/zero_point——通常从校准数据离线统计 running min/max 或使用 per-token 动态量化。当前 W4A4 的局限性：(a) 现代 GPU 的 Tensor Core 对 INT4 的支持有限（INT8 更成熟），实际加速比低于理论值；(b) 激活量化在 decode 阶段引入额外延迟（per-token quantization overhead）；(c) 在边缘设备（移动 GPU、NPU）上，INT4 MAC 操作的能耗和延迟优势更显著。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models

- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
---

## Binary-Coded Quantization (BCQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Binary-Coded Quantization (BCQ) 是一种非均匀量化方案，将权重矩阵 W ∈ R^{m×n} 表示为 q 个二值基矩阵 B_i ∈ {-1,+1}^{m×n} 与实值缩放因子 α_i ∈ R 的线性组合：Ŵ = Σ_{i=1}^q α_i B_i。其中 q 为量化比特数。参数通过最小化 Frobenius 重建误差 e = ‖W - Ŵ‖_F² 获得。当 q=1 时退化为标准二值量化：B₁* = sign(W), α₁* = ⟨W, B*⟩ / ‖B*‖_F²。当 q>1 时采用贪心初始化 + 交替优化：(1) 贪心：逐比特计算残差 R_i = W - Σ_{j=1}^i α_j B_j，B_{i+1} = sign(R_i)；(2) 交替优化：最小二乘更新 α = (B^T B)^{-1} B^T W，二分搜索重分配 B_i。BCQ 的核心优势是二值结构天然适合硬件加速——每个比特平面的运算化为对激活值的加减操作（因 B_i ∈ {-1,+1}），无需 centroid lookup 等非均匀量化的额外开销。相关工作包括 LUT-GEMM（利用二值模式查表加速 GPU GEMM）、iFPU（利用指数预对齐降低浮点运算复杂度）、FIGLUT（硬件加速器 LUT-based GEMM）。BCQ 也是 ShiftAddLLM 和 AnyBCQ 的底层量化格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BCQ 在 AnyBCQ 中 q=2 至 4 的逐层量化流程（group-wise asymmetric BCQ, g=128）：

```
# q-bit BCQ 量化伪代码
def BCQ_quantize(W, q, g=128, T=20):
    # W ∈ R^{m×n}, 按 g=128 列分组
    for group in range(0, n, g):
        W_g = W[:, group:group+g]
        B = []        # 比特平面列表
        alpha = []    # 缩放因子列表
        residual = W_g.clone()
        
        # 贪心初始化
        for i in range(q):
            B_i = sign(residual)  # B_i ∈ {-1,+1}^{m×g}
            alpha_i = dot(residual, B_i) / norm(B_i)^2
            residual = residual - alpha_i * B_i
            B.append(B_i); alpha.append(alpha_i)
        
        # 交替优化 (T 轮)
        for t in range(T):
            B_mat = concat(B)           # [m×g, q]
            alpha = least_squares(B_mat, W_g)  # α = (B^T B)^{-1} B^T W
            for i in range(q):
                B[i] = binary_search(alpha, W_g)  # 重分配 B_i ∈ {-1,+1}
        
        store(B_1...B_q, alpha_1...alpha_q)
```

关键参数：AnyBCQ 使用 asymmetric BCQ（允许偏差项），g=128，T=20，lr=1e-4，10 MRE epochs。q-bit BCQ 可表达 2^q 个不同值（对应 B 的 2^q 种符号组合），在 α-space 的线性子空间内变化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BCQ 的开源实现：(1) LUT-GEMM (github.com/naver-aics/lut-gemm)：BCQ 格式下的 GPU 推理 kernel；(2) transformer_bcq (github.com/insoochung/transformer_bcq)：TensorFlow BCQ 教程；(3) AnyBCQ (github.com/naver-aics/anybcq)：多精度 BCQ。硬件加速器包括 iFPU（bit-plane 浮点-整数混合运算单元）和 FIGLUT（HPCA 2025，LUT-based FP-INT GEMM 加速器）。权衡：低比特下 BCQ 优于均匀量化但逊于 K-means 聚类；4-bit 时各方法差距小；BCQ 的硬件效率优势（直接比特平面操作，消除 centroid lookup 和 bit-transpose）使其更具实用性。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---

## Multi-Precision LLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Precision LLM 是一种量化范式，允许单个部署模型支持多种推理精度（如 2/3/4-bit），从而根据运行时 SLO 动态切换比特宽度。与 Fixed-Precision LLM（每精度需独立模型）相比，通过共享精度间的量化表示大幅减少存储。源自 CNN 的 Any-Precision Networks (AAAI 2021)，LLM 时代转向 PTQ 方式。代表性工作：(1) Any-Precision LLM (ICML 2024 Oral)：聚类式非均匀量化 + Incremental Upscaling，从 3-bit 逐步分裂 centroid 至 8-bit，但 2-bit 退化严重且不硬件友好；(2) Matryoshka Quantization (2025)：MSB slicing，低比特是高位比特的子集；(3) AnyBCQ：BCQ-based 多精度，共享比特平面 + 独立 scale，2-bit 性能强且硬件友好。存储优势：LLaMA-3.1-8B 存三个独立模型需 9.85GB，AnyBCQ 单模型仅 4.99GB（↓49%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AnyBCQ Multi-Precision LLM 推理流程（用户指定精度 p ∈ {2,3,4}）：

```
def multi_precision_inference(model, input_ids, p):
    hidden = embed(input_ids)
    for layer in model.layers:
        for linear in layer.linears:
            W_bits = load_bitplanes(linear, num_planes=p)  # 仅加载 p 层
            alpha = load_scales(linear, precision=p)       # α_i^{(p)}
            output = zeros(batch, hidden_dim)
            for i in range(p):
                partial = bitplane_gemm(W_bits[i], hidden)  # B_i ∈ {-1,+1}
                output += alpha[i] * partial
        hidden = attention(hidden) + mlp(hidden)
    return lm_head(hidden)
```

Multi-Precision（全模型统一精度切换）与 Mixed-Precision（不同层/不同 token 不同精度）的区别：前者如 AnyBCQ 从 2-bit 切换到 4-bit，后者如 DP-LLM 逐层动态精度、PMPD 逐解码步精度下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
多精度模型优化：(1) Any-Precision LLM：K-means 基础种子 → 分裂 centroid → 存储多套 centroid table；(2) AnyBCQ：BCQ 基础精度 → 逐比特冻结 B_i → 残差提取新比特平面 → 优化 α；(3) Matryoshka Quantization：QAT 联合优化多个比特宽度 loss。部署要求：运行时精度切换、按需比特平面加载、多精度 kernel。限制：Multi-Precision 高精度（如 4-bit）通常略逊于 Fixed-Precision 同精度模型，因共享比特平面的约束缩小了优化空间。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---

## Progressive Precision Expansion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Precision Expansion（渐进式精度扩展）是 AnyBCQ 中实现多精度模型的核心机制。从基础精度 p_L（如 2-bit）开始逐步扩展到 p_H（如 4-bit），每次增加 1 bit。核心原则：(1) 冻结已分配比特平面 B_1...B_{p-1}（低精度模型是高精度模型的严格子集）；(2) 从残差提取新比特平面 B_p = sign(W - Ŵ^{(p-1)})（捕获之前未表达的信息）；(3) 仅优化缩放因子 α（最小二乘闭式解，不修改 B）。与 Any-Precision LLM 的 Incremental Upscaling（分裂 centroid，refinement 型）不同，这是"添加新信息"型（additive），保证单调精度改善：p-bit ≥ (p-1)-bit。AnyBCQ 2-bit MMLU=35.32（vs Any-Precision LLM=24.66），验证了该方法在极低比特的有效性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AnyBCQ Progressive Precision Expansion（p_L=2, p_H=4）：

```
# 阶段 1: 基础精度 p_L=2
B_1, B_2, α_1, α_2 = GREEDY(W)
for t in 1..T:  # T=20
    α = least_squares([B_1,B_2], W)
    B_1, B_2 = binary_search(α, W)

# 阶段 2: p=3 扩展
α_3 = 0; B_3 = zeros_like(W)
for t in 1..T:
    R = W - (α_1*B_1 + α_2*B_2 + α_3*B_3)
    B_3 = sign(R)  # 从残差提取新比特平面
    {α_i} = least_squares([B_1,B_2,B_3], W)  # B_1,B_2 冻结

# 阶段 3: p=4 扩展 (类似)
α_4 = 0; B_4 = zeros_like(W)
for t in 1..T:
    R = W - (α_1*B_1 + α_2*B_2 + α_3*B_3 + α_4*B_4)
    B_4 = sign(R)
    {α_i} = least_squares([B_1,B_2,B_3,B_4], W)
```

B_{p-1} 不能修改的原因：如果允许修改，低精度模型的权重表达会改变，破坏"多精度统一模型"的前提。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) p_L 越低省内存越多但优化空间越紧，AnyBCQ 选 2；(2) B_p=0 初始化使第一次 sign(R) 直接捕获最大残差方向；(3) α 优化用最小二乘闭式解，搭配 block-wise MRE 微调；(4) 每增 1 bit 表达能力翻倍但收益递减（3→4 < 2→3）。限制：高精度下 Fixed-Precision 略优（因 B_i 在低精度优化时未考虑高精度需求），AnyBCQ Multi-prec. 4-bit MMLU=63.15 vs Fixed-prec.=63.90。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---

## Bit-Plane Representation in Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-Plane（比特平面）在量化中指将量化权重按比特位置分解为多个二值矩阵层。在 BCQ 中每层 B_i ∈ {-1,+1}^{m×n} 是独立二值基，q 个平面构成 q-bit 表示。每平面存为 1 bit/元素（packed binary），比 FP16 压缩 16×。多精度推理时按需加载前 p 个平面（p=2 比 p=4 少 50% 数据），精准节约带宽。非均匀量化中比特平面存的是 centroid index 的二进制展开，而 BCQ 中直接是可计算二值操作数（{-1,+1}）——这是 BCQ 硬件效率的关键。数学本质：Ŵ = Σ α_i B_i，B_i 是"方向"，α_i 是"幅度"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
比特平面在 AnyBCQ 推理计算：

```
hidden ∈ R^{1×K}, BP_1...BP_p (每个 packed 为 K×N bits)

output = zeros(1, N)
for plane_idx in range(p):
    BP = load_packed_bitplane(plane_idx)  # 1-bit per weight
    partial = LUT_GEMM(BP, hidden)  # B_i ∈ {-1,+1}: 加减操作 + LUT加速
    alpha = load_scale(plane_idx, precision=p)  # α_i^{(p)}
    output += alpha * partial
```

存储格式：每平面 row-major 连续存储，每 byte 打包 8 个权重 bit。符号映射：0→-1, 1→+1。BCQ vs 非均匀量化的比特平面对比：总比特数相同（qN bits），但 BCQ 可直接操作（加减），非均匀需额外 bit-transpose + table lookup。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPU 上比特平面加载：以 byte/word 为单位，通过 bitwise AND + shift 提取单 bit。按需加载：仅加载所需 p 个平面（kernel 中通过 p 参数控制循环次数和显存读取量）。比特平面 vs centroid index 的运行效率：centroid index 方式需 (1) 加载 p 个平面 (2) bit-transpose 重组为 p-bit index (3) 查表获取 centroid 值 (4) GEMM；BCQ 方式仅需 (1) 加载 p 个平面 (2) 直接加减 + LUT 加速 (3) 缩放累加。BCQ 少了 bit-transpose（占 kernel 延迟 35-58%）和 centroid lookup（占 9-17%）。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---

## Basis Sharing (Cross-Layer SVD Parameter Sharing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Basis Sharing 是一种基于 SVD 的跨层参数共享 LLM 压缩方法。核心思想：将预训练 LLM 中不同层的同类型权重矩阵（W_K, W_Q, W_V, W_Up, W_Gate）水平拼接为一个合并矩阵 W_cat ∈ R^{d1 × n·d2}，然后对缩放后的拼接矩阵做一次性 SVD 分解，提取 k 个共享基向量（basis vectors）构成基矩阵 B''，和每层独有的系数矩阵 C^(i)。每个权重列被重构为共享基向量的线性组合：W_{:,j}^{(i)} ≈ Σ_{m=1}^k B''_{:,m} C_{m,j}^{(i)}。

逻辑链：(1) 水平拼接 n 层同类型权重 → (2) 评估激活感知缩放矩阵 S（S·S^T = cholesky(X^T X)，X 为跨层拼接输入）→ (3) S·W_cat 做 SVD → (4) 截断 k 个奇异值：SW_cat ≈ U_k Σ_k V_k^T → (5) 基矩阵 B'' = S^{-1}U_kΣ_k（所有层共享），系数 C = V_k^T（前 d2 列为第 1 层，后 d2 列为第 2 层...）。

关键特性：(a) 共享基向量意味着"参数原型"跨层复用，不同层通过不同系数实现功能差异化；(b) 压缩比由 k 控制，k = (d1·d2·n × x%)/(d1 + d2·n)；(c) 推理时计算 X·B''·C（两次小矩阵乘代替一次大矩阵乘）；(d) 与传统参数共享（强制权重完全相同）不同，Basis Sharing 保留独有系数，无需从头训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Basis Sharing 算法 pipeline（LLaMA2-7B W_K, n=2 层共享, 20% 压缩比）：

```
# Input: 同类型权重 W^(1)...W^(n) ∈ R^{d1×d2}, 校准输入 X^(1)...X^(n)
# Output: 共享基矩阵 B'' ∈ R^{d1×k}, 每层系数 C^(i) ∈ R^{k×d2}

# Step 1: 垂直拼接输入激活
X = concat_vertical(X^(1), ..., X^(n))    # [L·n, d1]

# Step 2: 计算激活感知缩放矩阵 S
S = cholesky(X^T @ X)^{1/2}               # S·S^T = X^T X, FP64

# Step 3: 水平拼接权重
W_cat = concat_horizontal(W^(1), ..., W^(n))  # [d1, n·d2]

# Step 4: 缩放并 SVD
U, Σ, V^T = SVD(S @ W_cat)

# Step 5: 截断 (k 由压缩比决定)
k = (d1 * d2 * n * x%) / (d1 + d2 * n)
U_k, Σ_k, V_k = U[:,:k], Σ[:k,:k], V^T[:k,:]

# Step 6: 分离共享基和独特系数
B' = U_k @ Σ_k                            # [d1, k] 缩放空间基
B'' = S^{-1} @ B'                         # [d1, k] 最终共享基矩阵
C = V_k                                   # [k, n·d2] 系数, C^(i) = C[:, (i-1)*d2:i*d2]

# Step 7: 推理
Y_i = X_i @ B'' @ C^(i)                  # B'' 共享, C^(i) 每层独有
```

矩阵类型筛选（Frobenius Loss 热力图分析）：
W_K, W_Q, W_V, W_Up, W_Gate → loss_shared < loss_individual → 适合 Basis Sharing
W_Down → rank 增大导致截断损失更大 → 不适合
W_O → loss_shared > loss_individual → 不适合

层分组：相邻层成对（1-2, 3-4, ...），默认 2 层一组。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代码开源：https://github.com/TUDa-HWAI/Basis_Sharing。实现要求：预训练 LLM 作为起点（无需训练），256 条 WikiText-2 校准样本，FP64 S 评估。使用场景：20%-50% 压缩，LLaMA/LLaMA2/OPT/Mistral 均有效，可与 LoRA 组合恢复精度。GPT2 压缩仅需 26.47s。限制：高压缩比（>50%）误差急剧增大，不改变推理计算量（加速来自内存节省）。

涉及论文标题：
- Basis Sharing Cross-Layer Parameter Sharing for Large Language Model Compression

---

## SVD-based LLM Weight Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SVD-based LLM Weight Compression（基于奇异值分解的 LLM 权重压缩）是一类利用低秩近似压缩 LLM 权重的后训练压缩方法。对 W ∈ R^{d1×d2}，SVD 分解 W = U Σ V^T，保留前 k 个最大奇异值：W ≈ U_k Σ_k V_k^T。压缩比 = (d1·d2) / (k·(d1+d2+1))。

演进路径：(1) Vanilla SVD：直接 SVD 截断，压缩误差大；(2) FWSVD（Hsu et al., 2022）：Fisher 信息加权，需梯度、大模型 OOM；(3) ASVD（Yuan et al., 2023）：激活敏感度选择通道，避免梯度；(4) SVD-LLM（Wang et al., 2024b）：whitening matrix S（S·S^T = cholesky(X^T X)）缩放权重后 SVD，使截断误差与输出误差对齐——当前 per-layer SVD SOTA；(5) Basis Sharing（本文）：在 SVD-LLM 基础上扩展跨层共享基向量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SVD-LLM 压缩流程（W, X, 压缩比 x%）：

```
X = collect_activation(W, calib_data)    # [L, d1]
S = cholesky(X^T @ X)^{1/2}             # whitening matrix
U, Σ, V^T = SVD(S @ W)
k = (d1 * d2 * x%) / (d1 + d2)
B = S^{-1} @ U[:,:k] @ Σ[:k,:k]         # 基矩阵 [d1, k]
C = V[:k, :]                             # 系数 [k, d2]
# 推理: Y ≈ X @ B @ C
```

关键数学：引入 S 后 ‖XW - XW_k‖_F² = ‖(SW) - (SW)_k‖_F²，SVD 截断直接对应输出 MSE 最小化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：SVD-LLM https://github.com/AIoT-MLSys-Lab/SVD-LLM; Basis Sharing https://github.com/TUDa-HWAI/Basis_Sharing。需 128-256 条校准样本，FP64 S 评估。LLaMA-7B 20% 压缩 SVD-LLM PPL=7.94 (dense=5.68)，Basis Sharing=7.74。限制：大模型需大显存，>50% 压缩误差急剧增大，加速来自内存带宽节省而非计算量减少，校准集分布影响 reasoning 效果。

涉及论文标题：
- Basis Sharing Cross-Layer Parameter Sharing for Large Language Model Compression
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

SLiM 将 SVD 压缩推广到显著性加权的误差补偿范式：不对原始权重 W 做 SVD，而是对压缩误差的显著性矩阵 S_C = diag(x)·(W^C - W) 做 SVD，再通过逆显著性变换 diag(1/x) 恢复低秩适配器。相比直接对原始权重或误差做 SVD，显著性加权确保低秩近似优先修正对模型输出影响最大的权重通道。该方法的数学基础是显著性函数 F(W)=diag(x)W 的可加性（F(W^C+LR)=F(W^C)+F(LR)）和可逆性，使误差隔离和适配器恢复成为可能。

---

## Frobenius Loss-guided Weight Matrix Selection for Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frobenius Loss-guided Weight Matrix Selection 是 Basis Sharing 中用于确定哪些类型权重矩阵适合跨层共享的分析方法。对每种矩阵类型，评估独立 SVD 压缩 loss 之和 vs 跨层共享 loss，用热力图比较。

分析结论（LLaMA2-7B）：W_K, W_Q, W_V, W_Up, W_Gate 适合共享（共享 loss < 独立 loss 之和）；W_Down 不适合（高维→低维投影拼接后 rank 增大，相同 k 下截断损失更大）；W_O 不适合（注意力输出功能跨层差异大）。具体数值：W_K 9-10 层共享 loss=61817.3 < 独立和=66682.9；W_O 9-10 层共享 loss=10618.3 > 独立和=9250.8。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
热力图生成（32 层 W_K, 20% 压缩比）：

```
for i in range(32):
    X^(i) = collect_activation(W^(i), calib)
    S^(i) = cholesky((X^(i))^T @ X^(i))^{1/2}
    loss_diag[i] = ‖S^(i)·W^(i) - SVD_k(S^(i)·W^(i))‖_F²

for i, j in adjacent_layer_pairs:
    X_cat = concat_vertical(X^(i), X^(j))
    S_ij = cholesky(X_cat^T @ X_cat)^{1/2}
    W_cat = concat_horizontal(W^(i), W^(j))
    heatmap[i][j] = ‖S_ij·W_cat - SVD_k(S_ij·W_cat)‖_F²

suitable = all(heatmap[i][j] < loss_diag[i] + loss_diag[j] for adjacent i,j)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在压缩前离线完成（O(n²)），分析结果决定各类型矩阵的压缩策略。每个 LLM 架构需独立分析，因层间相似性因模型而异。

涉及论文标题：
- Basis Sharing Cross-Layer Parameter Sharing for Large Language Model Compression

## Binary Residual Approximation（二进制残差逼近）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Binary Residual Approximation（二进制残差逼近）是 BiLLM（ICML 2024）提出的一种针对少数高显著度（salient）权重的低比特逼近方法。其核心思想是：不将 salient 权重保留为高精度（INT8/FP16）也不简单二值化，而是通过两阶段递归二值化来逼近原始权重——先用初始二值矩阵 B_o 逼近原始权重矩阵 W，再对残差 (W - α_o*·B_o*) 进行第二次二值化得到 B_r，最终用两个二值矩阵的加权和 α_o·B_o + α_r·B_r 表达原 salient 权重。数学上可证明 ε_rb = ||W - α_o·B_o - α_r·B_r||² ≤ ||W - α_o·B_o||² = ε_direct，即残差逼近的量化误差 ≤ 直接二值化的误差。这相当于用 2-bit 存储开销（两个二值矩阵 + 两个 scalar）达到了接近 8-bit 的 salient 权重保护效果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
BiLLM 中 Binary Residual Approximation 仅应用于 Hessian 选取的 salient 列，执行流程如下：
```
# Step 1: 初始二值化（针对 salient 列 W_sal ∈ R^{n×k}）
α_o = ||W_sal||_ℓ1 / (n × k)              # optimal scaling factor
B_o = α_o · sign(W_sal)                   # first binary matrix

# Step 2: 计算残差
R = W_sal - B_o                            # W_sal - α_o·B_o

# Step 3: 残差二值化
α_r = ||R||_ℓ1 / (n × k)                   # residual scaling factor
B_r = α_r · sign(R)                        # second binary matrix

# 最终逼近：Ŵ_sal = α_o·B_o + α_r·B_r （2-bit 有效位宽）
```
对比：PB-LLM 保留 salient 权重为 INT8（8-bit），BiLLM 用 2-bit（两个二值矩阵）达到更好效果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 BiLLM GitHub 仓库（github.com/Aaronhuang-778/BiLLM）中，binary.py 文件的 `res_approximation` 函数实现了该方法。适用于所有需要保护少数权重精度同时维持极低位宽的量化场景——理想的 salient 比例 r_salient=5-10%（BiLLM Table 1 显示 LLaMA-7B r_salient≈9%），额外存储仅 0.09 bit。该方法的关键洞察是：对 salient 权重，二次二值化逼近远比均匀量化高效，因为残差本身的数值范围远小于原始权重（残差分布更集中在 0 附近，利于二值化）。结合 OBC block-wise 补偿后，整体 binarization 过程约 0.5 小时/7B 模型（单 A100）。

涉及论文标题：
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---

## Bell-shaped Distribution Splitting for Binarization（钟形分布分裂二值化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bell-shaped Distribution Splitting for Binarization（钟形分布分裂二值化）是 BiLLM（ICML 2024）提出的针对非 salient 权重的二值化策略。LLM 的权重（去除少数 salient 权重后）呈对称钟形分布（类似高斯或拉普拉斯分布，大多数值聚集在 0 附近）。直接对这些权重进行二值化（均匀量化极端情况，仅 ±α 两个量化级）会产生极大的 MSQE（Mean Squared Quantization Error），因为大量聚集在 0 的值被强制映射到 ±α。BiLLM 的解决策略是：搜索一个最优分裂点（break-point）p*，将钟形分布沿 p/-p 切割为两个区域——集中区（|w| ≤ p，权重密集在 0 附近）和稀疏区（|w| > p，分布在尾部），然后分别以独立的 scaling factor（α_c, α_s）对各区域独立二值化。这相当于用两个分段常数函数逼近钟形分布曲线。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
BiLLM 的分裂搜索和分段二值化流程（以 LLaMA-7B 为例）：
```
W_nonsal = W_{:, not in salient_cols}         # 非 salient 权重
m = max(|W_nonsal|)                            # 权重极值

# 搜索最优 break-point p*（百分位搜索，步长 0.1）
e_best = inf; p_best = 0
for ratio in [0.1, 0.2, ..., 0.9]:
    p = ratio × m
    W_c = {w ∈ W_nonsal : |w| ≤ p}             # 集中区 (concentrated)
    W_s = {w ∈ W_nonsal : |w| > p}             # 稀疏区 (sparse)
    
    # 各自独立二值化（Equation 12）
    α_c = ||W_c||_ℓ1 / n_c; B_c = α_c · sign(W_c)
    α_s = ||W_s||_ℓ1 / n_s; B_s = α_s · sign(W_s)
    
    # 总体 MSQE（Equation 11）
    θ²_p = ||W_s - B_s||² + ||W_c - B_c||²
    
    if θ²_p < e_best: e_best = θ²_p; p_best = p

# 最终二值化
Ŵ_nonsal = B_c(p_best) + B_s(p_best)          # 1-bit 参数 + 1-bit group flag
```
搜索曲线呈凸性（paper Appendix C 验证），保证了全局最优解的存在。额外开销为 1 bit 用于区分 sparse/concentrated 组（不参与 GEMM 计算）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 BiLLM GitHub 仓库中，binary.py 文件的 `seg_search` 函数实现了百分位搜索。该方法适用于任意呈钟形分布的非 salient 权重集合。搜索范围通常限制在 max(|W|) 的 10%-90% 之间。关键设计选择：(1) OBC 块级补偿在分布搜索之前进行，补偿后的权重分布更接近理想高斯，有利于凸性保证；(2) block_size=128 为最佳平衡点（更小 block 精度更高但 flag 位开销增大）；(3) 仅在一个维度使用分裂（不再嵌套多层分裂），以平衡精度和实现复杂度。该方法在 OPT-6.7B 上对非 salient 权重的二值化提升尤为显著（ablation 显示 splitting-only 提升 > residual-only 提升，Figure 8）。

涉及论文标题：
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---

## Structured Salient Weight Selection（结构化显著权重列选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structured Salient Weight Selection（结构化显著权重列选择）是 BiLLM 提出的基于 Hessian 矩阵的结构化重要权重识别策略。传统方法（如 PB-LLM）通过逐元素 Hessian 重要性排序非结构化选取 salient 权重，但需要额外 1-bit bitmap index 记录每个元素位置，增加平均 bit-width。BiLLM 观察到在 LLM 的多头注意力机制中，salient 权重的 Hessian 值高度聚集在特定列（尤其在 Q/K/V 投影层），因此采用按列结构化选择——将权重矩阵 W ∈ R^{n×m} 按列聚合显著性 S_j = Σ_i w_{ij}²/[H^{-1}]_{ii}²，选择 top-k 显著性最高的列作为 salient 组，以极小 bitmap 开销（1/b_size per weight，b_size=128 时仅 ~0.008 bit）覆盖绝大多数重要权重。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Step 1: 计算逐元素显著性矩阵（Equation 3）
S_{ij} = w_{ij}² / [H^{-1}]_{ii}²           # Hessian 加权的元素级显著性

# Step 2: 按列聚合显著性
col_salience = sum(|S|, dim=0)               # 每列的总显著性

# Step 3: 排序 + 搜索最优列数 n*
row_s = argsort(col_salience, descending)    # 按显著性降序排序列索引
for i = 1, 2, ..., len(row_s):
    B_1 = binary(W_{:, row_s[:i]})            # salient 列二值化
    B_2 = binary(W_{:, not in row_s[:i]})     # 非 salient 列二值化
    e_i = ||W - (B_1 ∪ B_2)||²               # 整体重构误差
n* = argmin_i e_i                             # 选最小误差的列数
salient_cols = row_s[:n*]                     # 通常 3-30 列（per block）
```
搜索范围通常限定在 3-30 列以控制 bit-width。BiLLM Table 1 显示该方法在 OPT/LLaMA/LLaMA2 系列上仅引入 0.07-0.13 bit 额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BiLLM 代码中 salient 函数实现 (binary.py)：`S = W**2 / H_chol_inv**2` → `row_s = torch.topk(torch.sum(torch.abs(S), dim=0), k=max_cols)` → 循环搜索 n*。结构化选择 vs 非结构化选择的权衡：(1) 结构化（按列）：bitmap 开销极小但可能漏掉列内分散的 salient 元素；(2) 非结构化（逐元素）：覆盖所有 salient 元素但每条需要 1-bit bitmap。BiLLM 论文（Appendix G, Figure 15）验证了 OPT 系列中 Q/K/V 层 salient 权重高度聚集于特定列，证实结构化策略的有效性。对于 FFN 层（FC1/FC2）salient 分布较分散的场景，搜索过程会自动选择更多列以覆盖，但平均 bit-width 仍控制在 ~1.1 bit。

涉及论文标题：
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---

## Diffusion Model Binarization（扩散模型二值化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
扩散模型二值化是将扩散模型（Diffusion Model）的 UNet 噪声估计网络中的权重和激活从 32-bit 全精度量化为 1-bit 的最极端压缩技术。与 LLM 二值化不同，扩散模型二值化面临三个独特挑战：(1) 模型结构——UNet 的 encoder 逐层下采样（H×W 减半、C 翻倍）、decoder 逐层上采样，维度不断变化，使得 identity shortcut 无法使用，切断了全精度信息流（二值化模型严重依赖 shortcut 补偿 1-bit 信息损失）；(2) 多步迭代——扩散模型需要 T 步（如 T=2000）迭代去噪，每步的激活分布不同，静态的二值化参数（bias、激活函数）无法适配所有 timestep；(3) 任务特性——SR 任务基于像素级重建，对精度极度敏感，直接套用 BNN/ReActNet 等分类任务二值化方法导致性能崩溃。BI-DiffSR 是首个专门为扩散模型 SR 设计的二值化方法，从结构和激活两个维度解决问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 BI-DiffSR 的 ×2 SR 推理 pipeline 为例，扩散模型二值化的完整流程（DDIM 50 步）：
```
x_T ~ N(0, I)                      # 初始高斯噪声
y = bicubic_upsample(LR_image)     # LR 上采样到 HR 分辨率

for timestep_t in reversed(selected_50_timesteps):
    input_6ch = Concat(x_t, y)                    # [H,W,6]
    F_s = FP_Conv(input_6ch)                      # 仅首层 FP
    t_em = SinusoidalPE(t)
    
    # Encoder: 4层，每层 ResBlock×2 + CP-Down
    for level in 1..4:
        for i in 1..2:
            i_grp = floor(K * t / T)              # K=5, 选 timestep 组
            x_shifted = x + bias[i_grp]            # TaR
            x_bin = Sign(x_shifted)                # 1-bit 激活
            w_bin = (||w||_1/n) * Sign(w)          # 1-bit 权重
            x_conv = bit_count(XNOR(x_bin, w_bin)) # 1-bit 卷积
            x_act = RPReLU[i_grp](x_conv)         # TaA
            x = x_act + x_input                    # identity shortcut
        x = CP_Down(x)                             # H/2, C*2
    
    # Decoder: 4层，每层 ResBlock×3 + CS-Fusion + CP-Up
    for level in 1..4:
        x = CS_Fusion(x, encoder_skip[level])
        for i in 1..3:
            # BI-Conv block (同上)
        x = CP_Up(x)                               # H*2, C/2
    
    epsilon_t = FP_Conv(F_d)
    x_{t-1} = DDIM_step(x_t, epsilon_t, t)

HR_image = x_0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 训练实现：每个 BI-Conv block 中 Sign 用 STE 回传梯度，RPReLU 为 learnable parameter，Adam 优化器，L1 loss。训练 1M iterations（2×A100-80G），batch=16，lr=1e-4。推理时理论加速基于 XNOR+bit-count 替代 FP MAC（32× 内存节省，64× 计算节省），但 BI-DiffSR 未实现定制 CUDA kernel。开源代码：https://github.com/zhengchen1999/BI-DiffSR。模型参数量 4.58M（含折算），单步 OPs 36.67G，比全精度 SR3 节省 79.2% OPs。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

BinaryDM 从不同于 BI-DiffSR 的角度解决 DM 二值化问题：不改变 UNet 架构（无 CP-Down/CP-Up 等结构修改），而是通过 QAT 训练策略优化二值化过程。EBB 解决权重表征能力坍塌，LRM 解决优化方向模糊。BinaryDM 适用于 DDIM（pixel-space）和 LDM（latent-space）两种 DM 架构，无需像 BI-DiffSR 那样修改网络结构。与 BI-DiffSR 的 PTQ 策略不同，BinaryDM 是 QAT 方法，需完整训练流程但获得更好的精度-效率平衡（W1A4 FID=7.74 vs EfficientDM W4A4 FID=10.60）。

---

## CP-Down / CP-Up（Consistent-Pixel-Downsample / Upsample，一致性像素下采样/上采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CP-Down 和 CP-Up 是 BI-DiffSR 提出的二值化友好 UNet 特征缩放模块。传统 UNet 使用 stride 卷积或 pooling 改变特征分辨率，导致输入输出维度不匹配，identity shortcut（二值化模型中传递全精度信息的关键通道）无法使用。CP-Down/CP-Up 将所有维度变化操作隔离到独立的缩放模块中，确保主模块（ResBlock）维度始终一致。CP-Down：先将输入沿通道均分为两组 → 各组经（二值化）卷积处理（维度不变，可加 shortcut）→ 两组结果相加 → Pixel-UnShuffle 降低分辨率并增加通道数。CP-Up：输入经两个（二值化）卷积处理 → 输出沿通道拼接 → Pixel-Shuffle 提升分辨率并减少通道数。核心公式：CP-Down `x_out = PixelUnshuffle(C1(x_s1) + C2(x_s2))`，CP-Up `x_out = PixelShuffle(Concat(C1(x_in), C2(x_in)))`。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CP-Down 伪代码（[H,W,C] → [H/2,W/2,2C]）：
```
x_s1, x_s2 = Chunk(x, 2, dim=C)    # [H,W,C/2] × 2
x1 = BI_Conv1(x_s1)                 # 二值化卷积, 维度不变
x2 = BI_Conv2(x_s2)                 # 二值化卷积, 维度不变
x_mid = x1 + x2                     # 加法融合, 可用shortcut
x_out = PixelUnshuffle(x_mid, 2)    # [H/2,W/2,2C]
```
CP-Up 伪代码（[H,W,C] → [2H,2W,C/2]）：
```
x1 = BI_Conv1(x)                    # [H,W,C]
x2 = BI_Conv2(x)                    # [H,W,C]
x_cat = Concat(x1, x2, dim=C)       # [H,W,2C]
x_out = PixelShuffle(x_cat, 2)      # [2H,2W,C/2]
```
消融验证：+CP-Down&CP-Up 使 PSNR 从 29.29dB（仅 identity shortcut）提升至 31.08dB (+1.79dB)。不引入额外参数——输入通道被均分后各自处理，总 MACs 不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 原生实现：`torch.nn.PixelUnshuffle(2)` 和 `torch.nn.PixelShuffle(2)`（v1.9+）。使用 Pixel-(Un)Shuffle 而非 stride/transposed convolution 的好处：(1) 无可学习参数，减少二值化训练负担；(2) 空间↔通道变换可精确反转，不引入额外信息损失。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---

## CS-Fusion（Channel-Shuffle Fusion，通道混洗融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CS-Fusion 是 BI-DiffSR 提出的 UNet skip connection 特征融合模块。UNet 的 skip connection 需融合 encoder/decoder 特征，但直接 concatenation 导致维度翻倍（与二值化 ResBlock 维度不匹配），直接 addition 因两种特征值域差异巨大（可达数倍）导致小值域特征被遮盖。CS-Fusion 通过 channel shuffle 将两个输入特征按奇偶通道索引交叉重组为两个新特征，平衡值域后通过双分支二值卷积+加法融合。shuffle 后各特征混合了两个输入的信息，值域自然接近。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: x1, x2 ∈ [H, W, C]
# 输出: x_fused ∈ [H, W, C]

# Channel Shuffle: 奇偶交错
x1_sh = Concat(x1_odd, x2_even)     # [H,W,C], x1 奇数通道 + x2 偶数通道
x2_sh = Concat(x1_even, x2_odd)     # [H,W,C], x1 偶数通道 + x2 奇数通道

# 二值化卷积融合（维度不变）
x_fused = BI_Conv1(x1_sh) + BI_Conv2(x2_sh)  # [H,W,C]
```
消融：CS-Fusion=31.99dB vs Concat=31.08dB vs Split=29.67dB vs Add=18.89dB。channel shuffle 零参数、零计算，仅改变索引排列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：构造索引映射后 `torch.cat` 拼接，配合 BI-Conv 完成融合。适用于任何需要融合两个值域差异大的特征并保持输出维度不变的场景。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---

## TaR / TaA（Timestep-aware Redistribution / Activation Function，时间步感知激活重分布/激活函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TaR 和 TaA 是 BI-DiffSR 提出的扩散模型时序感知二值化参数。扩散模型的 T 步迭代去噪中，激活分布随 timestep 剧烈变化，静态的 bias 和 RPReLU 无法适配所有 timestep。TaR/TaA 受 MoE 启发，设置 K 对 (bias^(i), RPReLU^(i))，将总 timestep T 均分为 K 组，每 timestep 仅激活对应组参数：`i = floor(K*t/T)`。这等价于将长时序分割为短区间，降低了单组参数的适配难度。BI-DiffSR 中 K=5, T=2000。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: x_input ∈ [H,W,C], timestep t, T=2000, K=5
i = floor(K * t / T)                 # i ∈ {0,1,2,3,4}
x_shifted = x_input + bias[i]         # TaR: 仅第 i 组 bias
x_bin = Sign(x_shifted)               # 1-bit 激活
w_bin = (||w||_1/n) * Sign(w)         # 1-bit 权重
x_conv = bit_count(XNOR(x_bin, w_bin)) # 1-bit 卷积
x_act = RPReLU[i](x_conv)             # TaA: 仅第 i 组 RPReLU
x_out = x_act + x_input               # shortcut
```
消融：同时用 TaR+TaA PSNR=32.66dB，单独任一反降低（仅 TaR=29.27dB, 仅 TaA=29.13dB vs 无=31.99dB）。K=5 已足够（PSNR 从 K=1=31.99→K=2=32.42→K=5=32.66）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`bias ∈ R^{K×C}` 和 `RPReLU ∈ R^{K×C}` 为 learnable Parameter，推理时 `i=int(K*t/T)` 索引选择。总参数量仅 K×C×2=640 vs 4.58M。设计灵感源于 MoE 稀疏激活——多组参数，每步仅选 1 组。适用于任何多步迭代模型（扩散、流模型）中分布随时间变化的场景。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---

## Evolvable-Basis Binarizer (EBB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Evolvable-Basis Binarizer (EBB) 是 BinaryDM（ICLR 2025）提出的用于扩散模型权重二值化的可演化基二值化器。核心设计：训练第一阶段使用双基残差二值化 w_EBB^bi = σ_I * sign(w) + σ_II * sign(w - σ_I * sign(w)) 作为过渡状态，候选值从 2 个扩展到 {±σ_I ± σ_II} 共 4 种组合，显著增强信息熵和表征空间；然后通过正则化 L_EBB = τ/N * Σ σ_II^i（τ=9e-2）驱动高阶基 σ_II→0；第二阶段移除高阶项，简化为 w^bi = σ_I * sign(w) 达到真正全二值化。EBB 的 "可演化性" 使高阶基仅作训练的脚手架，推理时无额外开销。仅应用于 DM 首尾各 6 层（约 15% 参数），中间层用 vanilla binarizer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BinaryDM 中 EBB 两阶段训练流程：
```
# Stage 1: Multi-basis EBB + Regularization
σ_I = ||w|| / n
σ_II = ||w - σ_I * sign(w)|| / n
for iter in range(stage1_iters):
    w_ebb = σ_I*sign(w) + σ_II*sign(w - σ_I*sign(w))
    o = σ_I*(a ⊗ sign(w)) + σ_II*(a ⊗ sign(w - σ_I*sign(w)))  # ⊗ = XNOR+popcount
    L = L_simple + (τ/N)*Σ σ_II + λ*L_LRM
    # STE backward with ∂σ_I/∂σ_II from Eq.7-8

# Stage 2: Single-basis Binarization
for iter in range(stage2_iters):
    w_bi = σ_I * sign(w)
    o = σ_I * (a ⊗ sign(w))
    L = L_simple + λ*L_LRM
```
EBB 仅在 feature_scale ≥ input_scale/2 的层应用。消融：W1A4, Head+Tail=6, FID=8.53(vs baseline 10.87)；Head+Tail=12 反升至 FID=7.10（过渡参数过多导致不稳定）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch QAT 实现：被 EBB 覆盖的卷积/线性层维护 (w, σ_I, σ_II) 三组参数，~100K iter 时切换阶段。正则化系数 τ=9e-2 最优（τ=0 FID=8.01, τ=9e-2 FID=6.99）。开源：https://github.com/Xingyu-Zheng/BinaryDM。训练约 200K iter，LDM batch=4，DDIM batch=64，lr 为全精度的 1/10~1/100。

涉及论文标题：
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

---

## Low-rank Representation Mimicking (LRM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Low-rank Representation Mimicking (LRM) 是 BinaryDM 提出的低秩表征模仿技术，用于辅助二值化 DM 的优化。核心流程：对全精度 DM 各 timestep embedding 模块的输出 ε̂ ∈ R^{h×w×c} 计算协方差矩阵 C = (hw)⁻² * ε̂ * ε̂^T → 特征分解取前 ⌈c/K⌉ 列特征向量 E（K=4, 降维4倍）→ 将全精度和二值化 DM 的中间表征投影到低秩空间 R^FP = ε̂^FP * E, R^bi = ε̂^bi * E → MSE 损失 ||R^FP - R^bi|| 驱动二值化 DM 沿主成分方向学习。投影矩阵 E 在首 batch 计算后固定，保证优化方向稳定性。LRM 避免高维空间直接对齐导致的模糊优化方向，使二值化 DM 集中于主方向学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Init: compute E_i once and freeze
for i, module in enumerate(timestep_embed_modules):
    ε_fp = fp_model.module_i(first_batch)          # [h,w,c]
    C = (ε_fp @ ε_fp.T) / (h*w)^2                  # [c,c]
    eigvecs = eigh(C).eigenvectors                  # descending
    k = ceil(c / 4)                                 # K=4
    E_i = eigvecs[:, :k]                            # freeze

# Training: low-rank alignment
for iter:
    for i, module in enumerate(timestep_embed_modules):
        R_fp = fp_model.module_i(x) @ E_i           # [h,w,k]
        R_bi = bin_model.module_i(x) @ E_i          # [h,w,k]
        L_LRM += MSE(R_fp, R_bi)
    L_total = L_simple + L_EBB + (1e-4/M) * L_LRM
```
消融：直接 MSE FID=7.36 vs LRM K=4 FID=6.99；K=8 FID=6.95。每 100 iter 更新 E FID=7.11 vs 固定 E FID=6.99（验证稳定性设计）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch: `torch.linalg.eigh` 首 batch 计算，E_i 作为 buffer 存储（不参与梯度）。应用于每个 timestep embedding 模块输出。消融（W1A32, LSUN-Bedrooms）：+EBB FID=7.39 → +EBB+LRM FID=6.99。训练 11.3h vs Q-Diffusion 校准 13.7h。

涉及论文标题：
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

---

## Block Transformer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Transformer 是 Ho et al. (KAIST/LG AI/Google DeepMind, NeurIPS 2024) 提出的分层全局到局部（hierarchical global-to-local）Transformer 架构，将标准自注意力分解为 block 级全局注意力和 token 级局部注意力以系统性地降低自回归推理开销。架构包含：(1) **Embedder**：lookup table 将每 $L_B$ 个 subword token embedding 拼接为 block embedding；(2) **Block Decoder**：在 block 序列上运行的标准自回归 Transformer（序列长度 $L/L_B$），全局 causal self-attention，输出 context embedding；(3) **Token Decoder**：以 context embedding 投影出的 prefix tokens 为全局上下文来源，仅对当前 block 内 $L_B$ token 做局部 self-attention，解码 individual tokens。主配置 $L_B=4$, prefix=2, block:token 参数比 1:1。核心收益：Block Decoder 将 KV cache 大小降 $L_B$ 倍、KV cache IO 降 $L_B^2$ 倍；Token Decoder 将 KV cache IO 从 $O(L^2)$ 降至 $O(L \cdot L_B)$（线性复杂度），prefill 可完全跳过。开源：https://github.com/itsnamgyu/block-transformer。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 Block Transformer (LB=4, prefix=2, L=2048) 推理 pipeline：
```
# Embedder: lookup table 拼接
E_emb = Embedding(V, D/LB)
block_emb[i] = Concat([E_emb[tok_{i*LB+j}] for j in range(LB)])  # [B, 512, D]

# Block Decoder: block-level global SA
h = BlockDecoder(block_emb)   # M_b layers causal SA, 512 blocks
context_emb = h[:, -1, :]     # [B, D] — 最后一个block位置的输出

# Token Decoder: token-level local SA
prefix = Linear(context_emb).view(B, P, D)       # [B, 2, D]
tok_embs = E_tok(curr_block_tokens)               # [B, 4, D]
h_tok = TokenDecoder(Concat([prefix, tok_embs]))  # [B, 6, D], 仅6token间SA
logits = Classifier(h_tok[:, -LB:, :])            # [B, 4, V]
```
KV cache 对比：Block Decoder 512 blocks (↓4× vs 2048), Token Decoder 6 tokens (↓341×)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 GPT-NeoX + HuggingFace Transformers + DeepSpeed ZeRO 实现。8×A100 40GB 训练 300B tokens (Pile)，H100 推理。吞吐量达 vanilla 的 10-25× (prefill-heavy/decode-heavy)。支持从预训练 vanilla transformer uptraining，仅需 10% 训练数据，为从现有大模型迁移到 Block Transformer 架构提供低成本的训练路径。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

---

## Global-to-Local Language Modeling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Global-to-Local Language Modeling 是 Block Transformer 提出的分层语言建模范式。核心思想：将自回归 Transformer 的全局自注意力分解为两个阶段：(1) **Global stage** (Block Decoder)：以粗粒度 block 为单位做全局注意力，将全局上下文压缩为 context embedding；(2) **Local stage** (Token Decoder)：以细粒度 token 为单位仅对当前 block 做局部注意力，依赖 context embedding 获取全局信息。与滑动窗口注意力（SWA）的关键区别：SWA 仅在部分层使用局部注意力，上层依赖堆叠隐式获得全局感受野，prefill 无法跳过。Block Transformer 在**所有 upper layers** 强制局部性，彻底消除 token decoder 的 prefill 和 KV cache。与 MEGABYTE 的关键区别：Block Transformer 首次识别 token decoder（local module）的计算能力对性能同等重要，提出 1:1 参数比和 prefix token 机制，而 MEGABYTE 认为 local module 应尽可能小（6:1 参数比）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KV cache reduction model (L=2048, LB=4):
- Block decoder: context length = 512, KV cache ↓4×, KV cache IO ↓16× (vs vanilla L=2048)
- Token decoder: context length = 4, KV cache ↓512×, KV cache IO ↓256× (vs vanilla)
- Overall: $O(L^2) \rightarrow O(L \cdot L_B)$, 即线性复杂度

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过修改标准 Transformer 架构实现。关键超参数：$L_B$ (block length)、prefix length (token decoder 计算宽度)、parameter allocation ratio ($N_b$:$N_t$ 层数比)。实验显示 $L_B=4$, prefix=2, ratio=1:1 在 perplexity-throughput 上 Pareto-optimal。更大 context length 下收益更大。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

---

## MEGABYTE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MEGABYTE 是 Yu et al. (Meta, NeurIPS 2023) 提出的字节级多尺度 Transformer 架构，采用与 Block Transformer 相似的全局到局部分层结构（patch-level global decoder + byte-level local decoder）。关键差异：(1) MEGABYTE 针对 byte-level 数据（无 tokenizer），Block Transformer 针对 subword-level 推理优化；(2) MEGABYTE 认为 local module 应尽可能小（建议 6:1 global:local 参数比），Block Transformer 证明 1:1 更优且更大 token decoder 提升吞吐量；(3) MEGABYTE 使用 summation 注入全局信息，Block Transformer 使用 prefix。Block Transformer 的 token-level MEGABYTE reimplementation 显示，Block Transformer 在吞吐量上超 MEGABYTE 1.5×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MEGABYTE vs Block Transformer 关键结构对比：
```
# MEGABYTE: 6:1 ratio, summation
ctx_bias = Linear(ctx_emb).reshape(B, LB, D)
tok_input = tok_embs + ctx_bias  # summation, no refinement

# Block Transformer: 1:1 ratio, prefix
prefix = Linear(ctx_emb).view(B, P, D)
tok_input = Concat([prefix, tok_embs])  # prefix, allows self-attention refinement
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block Transformer 论文重新实现 token-level MEGABYTE（summation, 6:1 ratio, Pile 300B tokens training）以进行公平比较。MEGABYTE 官方实现为 byte-level：https://github.com/facebookresearch/megabyte。

涉及论文标题：
- Block Transformer Global-to-Local Language Modeling for Fast Inference (NeurIPS 2024)

---

## MXFP4 / NVFP4 (Microscaling Floating-Point 4-bit Formats)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MXFP4 和 NVFP4 是新型微缩放（Microscaling）浮点格式，专为硬件加速的 4-bit LLM 推理设计。两者采用分层量化架构——元素分块共享 scale factor——但关键设计选择不同：

**MXFP4（Microscaling FP4，OCP 标准）**：G=32（每组 32 个元素共享一个 scale）、E=FP4 E2M1（1 sign + 2 exponent + 1 mantissa bits，7 个正候选值 + 0）、S=E8M0（scale 量化为 power-of-two，8 bits 全给 exponent、无 mantissa），总计 4.25 bits/element。E8M0 设计简化了硬件乘法操作（power-of-two 缩放等价于指数加法），但 scale 的粗粒度近似（仅 power-of-two 步长）引入额外量化误差。NVIDIA Blackwell B200 和 AMD CDNA4 GPU 支持 MXFP4。

**NVFP4（NVIDIA FP4，Blackwell 架构）**：G=16（每组 16 个元素，更小 group → 更精确的 per-group scaling）、E=FP4 E2M1（同 MXFP4 的 E2M1 基类型）、S=E4M3（完整 FP8 scale，4 exponent + 3 mantissa bits），总计 4.5 bits/element。仅 NVIDIA Blackwell GPU 支持。

两者的核心权衡：MXFP4 以粗 scale（E8M0 power-of-two）换取更低存储（4.25 vs 4.5 bits/elem）和简化硬件；NVFP4 以额外 0.25 bits/elem 换取更精确的 scale 表达（E4M3 full FP8）和更小 group（G=16 更细粒度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FP4 E2M1 候选值（对称，7个正+0+7个负）:
# {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}

# MXFP4 (G=32, E=E2M1, S=E8M0) — 4.25 bits/elem
for group in split(tensor, 32):
    s_G = absmax(group)
    s_G_q = E8M0_quantize(s_G)     # power-of-two only
    for x in group: x_q = RTN(x / s_G_q, grid=E2M1)

# NVFP4 (G=16, E=E2M1, S=E4M3) — 4.5 bits/elem
for group in split(tensor, 16):
    s_G = absmax(group)
    s_G_q = E4M3_quantize(s_G)     # full FP8 precision
    for x in group: x_q = RTN(x / s_G_q, grid=E2M1)
```

论文通过 MSE 理论分析揭示的关键发现：Laplace（原生）vs Normal（旋转后）分布下，MSE 收敛率存在 crossover——小 G 时 Laplace MSE 更低（NVFP4 G=16 不应旋转），大 G 时 Normal MSE 更低（MXFP4 G=32 应旋转）。这直接指导了 MR-GPTQ 的设计策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MXFP4 由 OCP MX Alliance 标准化（ocp-microscaling-formats-mx-v1-0-spec）。NVFP4 为 NVIDIA Blackwell 专有格式。硬件支持：NVIDIA Blackwell GPU 通过 tcgen05.mma 指令支持 NVFP4/MXFP4 矩阵乘法；AMD CDNA4 支持 MXFP4。PyTorch 中通过 fake_quantize 进行模拟量化；真实推理通过 CUTLASS/QuTLASS kernel 库调用硬件指令。

QeRL (Huang et al., NVIDIA, 2025) 将 NVFP4 用于 RL 训练 pipeline：用 AWQ calibration 对预训练 LLM 做 NVFP4 量化（calibration: OpenThoughts-114k），结合 LoRA adapter 进行 GRPO/DAPO 强化学习。QeRL 发现 NVFP4 量化噪声可增加策略熵（H(π(|q))），增强 RL 探索能力。NVFP4 结合 Marlin kernel 实现 1.2-2× rollout 加速，7B 模型仅 5.9GB vs BF16 15.2GB。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

---

## MR-GPTQ (Micro-Rotated GPTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MR-GPTQ（Micro-Rotated GPTQ）是专为 FP4 微缩放格式（MXFP4/NVFP4）设计的 GPTQ 变体，包含三个技术组件：(1) MSE-Optimized Grids——交替优化 per-tensor 和 per-group scales 最小化量化 MSE；(2) Static Activation Reordering——先确定 grid/scales 再按 Hessian 重排列列、量化后恢复原序，避免 dynamic act-order 的 10-20% 推理开销；(3) Block-wise Hadamard Rotations——以 group size 为单位进行旋转（"micro-rotation"），对大 G 格式（MXFP4 G=32）降低 per-element MSE，旋转融合入权重零推理开销。提供三种策略变体：MR-GPTQ-MXFP4（GPTQ + rotated MXFP4）、MR-GPTQ-NVFP4（GPTQ + rotated NVFP4 + MSE-optimized grid）、GPTQ + standard NVFP4（无旋转）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MR-GPTQ-MXFP4 离线量化
for each Linear layer:
    W_fp16 = layer.weight
    H_k = block_diag_hadamard(k=32)           # k = group_size
    
    # 1. 权重旋转（离线预融合）
    W_rot = W_fp16 @ H_k                       # Laplace → Normal
    
    # 2. MSE grid（MXFP4: 统一静态 s_T）
    s_T, s_G = MSE_optimize_scales(W_rot)
    
    # 3. Static act-order
    col_order = argsort(H_diag, descending=True)
    W_rot_ordered = W_rot[:, col_order]
    
    # 4. GPTQ per-column (OBS framework)
    for col in range(d_in):
        w_q = MXFP4_quantize(W_rot_ordered[:, col], s_T, s_G)
        error = (W_rot_ordered[:, col] - w_q) / H_inv[col, col]
        W_rot_ordered[:, col+1:] += error * H_inv[col, col+1:]
    
    # 5. 恢复原始列序
    W_q = inv_permute_cols(W_rot_ordered, col_order)

# 推理: 权重旋转已融合 → activation fused rotation (QuTLASS) → FP4 matmul
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：FP-Quant（https://github.com/IST-DASLab/FP-Quant）。量化模型托管于 HuggingFace（ISTA-DASLab/mr-gptq collection）。校准：FineWeb 1024 sequences + Hessian dampening λ=10^-2。关键结果：Llama-3.1-8B MXFP4 MR-GPTQ Avg Recovery 93.31%（RTN=87.83%，GPTQ=89.47%），将 MXFP4 与 NVFP4 精度差缩至 1-2%。大模型（70B+）两种格式均可恢复 98-99% FP16 精度。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization

## Dual Denoising (D²-DPM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual Denoising（D²-DPM，双重去噪）是 AAAI 2025 提出的针对量化扩散模型的后训练去噪机制，"双重"指在逆扩散采样过程中先后处理两类噪声：(1) **量化噪声去噪**：在每个时间步从量化模型输出 ε̂_θ(x_t, t) 中减去估计的量化噪声，恢复扩散噪声分布；(2) **扩散噪声去噪**：用恢复后的扩散噪声分布执行标准逆扩散采样。核心流程：校准阶段用 BRECQ 量化模型后，收集量化输出-噪声对 (ε̂, Δε)_t，为每个时间步 t 估计 ε̂ 和 Δε 的联合高斯分布参数；推理阶段在每一步用 TSQNM 条件化地预测量化噪声的均值和方差，进行修正后再执行 DDIM/DDPM 采样更新。提供两种变体：S-D²（随机双重去噪，减去完整估计噪声）和 D-D²（确定性双重去噪，仅减去条件均值，额外方差吸收到扩散项中）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 校准阶段
{calib_data} = collect_calibration(model_fp)  # 收集 M×T 组校准样本
model_q = BRECQ(model_fp, q_params, calib_data)  # AdaRound 量化
{(ε̂, Δε)^i} = collect_quant_error(model_fp, model_q)  # 收集 S×T 组量化误差对
μ[T×2], Σ[T×4] = gaussian_modeling({(ε̂, Δε)^i})  # 估计每时间步的联合高斯参数

# 推理阶段（双重去噪循环）
for t in T, ..., 1:
    ε̂_θ = model_q(x_t)  # 量化噪声估计网络前向
    μ_cond, σ²_cond = TSQNM(ε̂_θ, μ[t], Σ[t])  # 条件化量化噪声预测
    
    if S-D²:  # 随机双重去噪
        z ~ N(0, I)
        Δε' = μ_cond + sqrt(σ²_cond) * z
        ε' = ε̂_θ - Δε'  # 减去完整估计噪声，恢复分布
        # 用标准 SDE 采样
        
    if D-D²:  # 确定性双重去噪
        ε' = ε̂_θ - μ_cond  # 仅减去条件均值
        g'_eff = sqrt(g²(t) - g⁴(t)*σ²_cond/σ²_t)  # 额外方差吸收到扩散项
        # 用调整后的扩散系数采样
    
    x_{t-1} = DDIM_update(x_t, ε', α, Σ)  # 执行逆扩散更新
```

关键张量计算（LDM-4 W4A8, ImageNet 256×256, scale=3.0, η=0.0, steps=20）：
- 量化输出 ε̂_θ ∈ R^{4×64×64}（latent 空间维度）
- 假设各向同性+元素不相关 → σ²_ε̂, σ²_Δ, σ_ε̂Δ 均为标量（大大简化联合分布参数估计）
- 条件均值 μ_{Δε|ε̂} 和条件方差 σ²_{Δε|ε̂} 对所有元素使用相同的标量修正

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/TaylorJocelyn/D2-DPM。基于 LDM（latent-diffusion）和 BRECQ 框架构建。适用于 LDM-4/LDM-8 等 U-Net 架构的扩散模型，支持 W8A8 和 W4A8 配置。关键结果：W4A8 D-D² 在 ImageNet 256×256 上 FID=9.71（FP=11.13），即量化模型 FID 比全精度模型低 1.42。局限性：需要为每个时间步存储联合分布参数（μ[T×2], Σ[T×4]），需要校准阶段额外前向传播来收集量化噪声统计。

涉及论文标题：
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

## Time Step-aware Quantization Noise Modeling (TSQNM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TSQNM（时间步感知量化噪声建模）是 D²-DPM 中用于精确建模量化噪声的核心技术。基于两个实验观察：(1) 每个时间步的量化噪声 Δε_θ^(t) 近似服从高斯分布；(2) 量化输出 ε̂_θ^(t) 也近似服从高斯分布。由此假设 (ε̂, Δε) 在每个时间步 t 的联合分布近似于二元高斯分布，参数通过 BRECQ 校准阶段收集的量化输出-噪声对来估计。为简化高维估计，假设 ε̂ 和 Δε 的元素间不相关（协方差矩阵为对角阵）、分布各向同性（Σ = σ²I），将联合分布参数压缩为 4 个标量（μ_ε̂, μ_Δ, σ²_ε̂, σ²_Δ, σ_ε̂Δ）每时间步。推理时通过条件高斯公式计算 Δε|ε̂ 的分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 校准阶段：收集统计量
for t in 1..T:
    for s in 1..S:
        ε_fp = model_fp(x_t_s, t)
        ε_q = model_q(x_t_s, t)
        Δε_s = ε_q - ε_fp
        records[t].append((ε_q, Δε_s))
    
    # 估计时间步 t 的联合高斯参数
    μ̂[t] = mean({ε_q})          # 量化输出均值（标量）
    μ̂_Δ[t] = mean({Δε})          # 量化噪声均值（标量）
    σ²_ε̂[t] = var({ε_q})           # 量化输出方差（标量，各向同性假设）
    σ²_Δ[t] = var({Δε})           # 量化噪声方差（标量）
    σ_ε̂Δ[t] = cov({ε_q}, {Δε})   # 交叉协方差（标量）

# 推理阶段：条件化预测
def TSQNM(ε̂_q_current, μ[t], Σ[t]):
    μ_cond = (σ_ε̂Δ/σ²_ε̂) * (ε̂_q_current - μ_ε̂) + μ_Δ  # 条件均值
    σ²_cond = σ²_Δ - σ²_ε̂Δ/σ²_ε̂  # 条件方差
    return μ_cond, σ²_cond
```

与 PTQD 的核心区别：PTQD 假设 Δε 与 ε̂ 线性相关，TSQNM 使用更一般的联合高斯分布→条件分布模型，当线性假设不成立时（特别是不同时间步），联合高斯建模更精确。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 在 BRECQ 量化完成后，用全精度和量化模型对每个时间步的校准数据做一次额外前向传播；(2) 对每个时间步单独估计 4×T 个参数；(3) 存储为查找表在推理时使用。为简化，假设元素间不相关+各向同性，将协方差矩阵压缩为标量。该方法依赖于校准数据的质量和对高斯假设的成立程度。在低 bit-width（如 W4A8）下，由于量化噪声更大，TSQNM 的建模精度相对更高。

涉及论文标题：
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

## SDE-based Diffusion Sampling with Quantization Noise

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是 D²-DPM 提出的将量化噪声纳入 SDE 框架进行理论分析的视角。核心推导：当量化噪声 Δε_θ ~ N(μ_Δ, σ²_Δ I) 被引入噪声估计网络后，反向 SDE 采样方程变形为 d𝐱 = [𝐟(𝐱,t) + g(t)²(ε_θ + μ_Δ)/σ_t]dt + [g(t) + g(t)²σ_Δ√(dt)/σ_t]dw̄。量化噪声的 μ_Δ 叠加到 drift term（影响采样方向），σ_Δ 增大 diffusion coefficient（影响采样波动和收敛性）。基于此分解，D²-DPM 设计了分离式修正：通过条件均值修正 drift term，通过条件方差修正/吸收到 diffusion term 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化扩散模型的 SDE 采样方程分解：

**原始全精度 SDE 采样（Anderson 1982）**:
$$d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + g(t) d\bar{\mathbf{w}}$$

**量化后 SDE 采样（含量化噪声）**:
$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) + g(t)^{2} \frac{\boldsymbol{\epsilon}_{\theta} + \boldsymbol{\mu}_{\Delta}}{\sigma_{t}}\right] dt + \left[g(t) + \frac{g(t)^{2} \sigma_{\Delta} \sqrt{dt}}{\sigma_{t}}\right] d\bar{\mathbf{w}}$$

关键分解：
- Drift term 修正：μ_Δ → 用条件均值 μ_{Δε|ε̂} 从 ε̂_θ 中减去
- Diffusion term 修正：σ_Δ → S-D² 通过减去完整估计噪声恢复；D-D² 通过调整有效 g(t) 吸收

D-D² 的有效扩散系数调整：
$$g'(t) = \sqrt{g^{2}(t) - \frac{g^{4}(t)\sigma_{\Delta}^{2}(t)}{\sigma_{t}^{2}}}$$

当 ODE 采样（η=0, g(t)→0）时，D-D² 的额外标准差实际上将 ODE 转换为了隐式的 Langevin SDE，产生更好的误差缓冲。这解释了 D-D² 在无随机性容量时反而优于 S-D² 的实验现象。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该理论分析主要用于：(1) 理解量化噪声如何在采样过程中传播和积累；(2) 指导设计分离式修正策略（分别修正均值和方差）；(3) 判断采样器（ODE vs SDE）对额外方差的吸收能力。实际实现时不需要显式求解 SDE——D²-DPM 在 DDIM 采样迭代中嵌入修正步骤：先通过 TSQNM 预测量化噪声的参数，再对输出做去噪修正，最后用修正后的 ε' 执行标准 DDIM 更新。其直接对应关系为：DDIM 是 ODE 求解器（η=0），DDPM 是 SDE 求解器（η=1），η 控制随机性容量。

涉及论文标题：
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

---

## Equivalent Scaling for Diffusion Models（扩散模型中的等效缩放）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Equivalent Scaling（等效缩放）是一种通过插入通道级缩放因子 τ 来双向重分布激活与权重的量化难度的方法。数学上，线性层 Y = XW 被改写为 Y = (X/τ)(τ^T ⊙ W)，其中 / 为通道级除法，⊙ 为通道级乘法。τ ∈ R^{Cin} 将某些通道的激活缩小（降低激活量化难度）的同时将对应通道的权重放大（增加权重量化难度），或反之。因整体输出 Y 不变，这种变换是数学等效的。该方法最早由 SmoothQuant（Xiao et al., 2023）引入 LLM 量化，使用手工启发式 τ_c = (max|X_c|)^β / (max|W_c|)^{1-β}。DMQ 论文发现 SmoothQuant 直接迁移到扩散模型会严重失败：因为扩散模型中激活幅度远大于权重幅度，手动 τ 极大 → 权重量化范围被大幅扩展 → 权重量化误差暴增（FFHQ W4A8: Weight Quant. Error 从 0.0060 飙升至 0.0694，FID 从 36.08 飙升至 454.16）。DMQ 提出 Learned Equivalent Scaling (LES)，通过梯度下降直接学习 τ 以最小化量化输出的 MSE（L_i = ||X_iW - Q(X̂_i) Q(Ŵ)||²），避免手动启发式的不准确。LES 的 τ 通过融合策略（τ^T ⊙ W 预计算，τ ⊙ s^X 融合到激活 scale）实现零推理开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 DMQ 的 LES 学习流程为例（W4A8, LDM-4 FFHQ 256×256）：

```
# 输入: FP32 权重 W ∈ R^{Cin×Cout}, 校准集 {(X_i, t_i)}
# 初始化 τ = 1 (所有通道不缩放)
# 累积损失 Λ_t 初始化为 0

for iteration in range(6000):
    batch = sample_calibration_batch(B=32)
    for each (X_i, t_i) in batch:
        # Eq.5: 等效缩放变换
        X_hat = X_i / τ           # 激活缩放 (channel-wise)
        W_hat = τ^T ⊙ W           # 权重缩放 (channel-wise)
        
        # MinMax 量化
        X_q = MinMaxQuant_8bit(X_hat)   # per-tensor
        W_q = MinMaxQuant_4bit(W_hat)   # per-channel
        
        # Eq.6: Layer-wise MSE loss
        L_i = ||X_i @ W - dequant(X_q) @ dequant(W_q)||^2
        
        # Eq.8: Adaptive timestep weight
        lambda_ti = (1 - Λ_{ti} / sum(Λ)) ^ α  # α=20
        loss += lambda_ti * L_i
    
    # 梯度下降更新 τ
    τ = τ - lr * ∇_τ loss
    
    # Eq.9: 更新累积损失 (EMA)
    for each t:
        Λ_t = 0.95 * Λ_t + 0.05 * mean(L_i for i where t_i=t)
```

**融合到推理（零开销）**：
- 权重侧：W_fused = τ^T ⊙ W（预计算存储，替换原 FP32 权重后再量化）
- 激活侧：s_X_fused = τ ⊙ s_X（预计算融合 scale，量化时使用）
- 推理时：X_q = round(X / s_X_fused)，W_q 已含 τ 信息

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 LLM 中，SmoothQuant 的 τ 通过手动公式计算后融合到前一层（LayerNorm weight 或前一层 Linear weight）。但扩散模型中 U-Net 包含非线性的 Swish/SiLU 激活（位于矩阵乘法之前），阻止了 τ 融合到前一层。DMQ 改用 τ 融合到激活 scale（s_X_fused = τ ⊙ s_X），利用静态量化的量化 scale 不变特性回避这个问题。实现时可复用现有量化框架（BRECQ/AdaRound 用于后续的权重量化精炼）。LES 因素仅需几千次迭代优化，远超于 QAT 的完整训练开销，适合 PTQ 的高效部署场景。开源：https://github.com/LeeDongYeun/dmq。

涉及论文标题：
- DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

---

## Computational Invariance (Rotational Invariance) in Transformer Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Computational Invariance（计算不变性/旋转不变性）是 LLM 量化中通过插入正交旋转矩阵来改变激活分布而不改变模型数学输出的核心技术。原理：对线性层 Y = XW^T，插入正交矩阵 R（RR^T=I）后 Y = (XR)(R^T W^T) = XW^T，输出不变。在 Transformer block 中可插入 R1-R4 四个旋转矩阵：(1) R1 右乘 W_q/W_k/W_v/W_up/W_gate，R1^T 左乘 W_out/W_down/W_embedding，R1 右乘 W_lm_head；(2) R2 插入 W_v 和 W_o 之间的多头注意力路径（per-head）；(3) R3 为在线 Hadamard 变换用于 KV cache 量化（因 RoPE 无法融合）；(4) R4 为在线 Hadamard 变换用于 FFN down-projection（因 gating 机制）。R1/R2 可离线融合入权重实现零推理开销，R3/R4 使用快速 Hadamard kernel 在线计算。核心性质：(a) 旋转保持 L2 范数不变；(b) RMSNorm 与旋转可交换（RMSNorm(XR)=RMSNorm(X)R）。

从算法pipeline角度拆解术语，给出具体例子。
以下为插入旋转矩阵后的 Transformer 计算流程（离线融合后）：
```
# 离线：W_q'=W_q@R1, W_k'=W_k@R1, W_v'=W_v@R1@R2
#       W_o'=R2^T@W_o, W_up'=W_up@R1, W_gate'=W_gate@R1
#       W_down'=R1^T@W_down, W_embed'=R1^T@W_embed, W_lm_head'=W_lm_head@R1
# 在线（推理时）：
Q, K, V = X@W_q'^T, X@W_k'^T, X@W_v'^T
K_rope, Q_rope = RoPE(K), RoPE(Q)
scores = Q_rope @ (Hadamard(K_rope))^T   # R3 在线 Hadamard
attn_out = softmax(scores) @ Hadamard(V)  # R3 双向抵消 = Hadamard(attn_out)
O = attn_out @ W_o'^T
gate, up = X@W_gate'^T, X@W_up'^T
down_in = Hadamard(SiLU(gate)*up)         # R4 在线 Hadamard
F = down_in @ W_down'^T                   # W_down' 已含 R1^T
X_out = X_in + O + F
```

术语一般如何实现？如何使用？
实现基于 QuaRot/SpinQuant 代码框架：(1) 加载 HuggingFace 模型后自动识别 Q/K/V/O/Up/Gate/Down 权重位置；(2) 用 `torch.linalg.qr()` 或 Cayley SGD 初始化正交旋转矩阵；(3) 按 Computational Invariance 规则融合入相邻权重（矩阵乘法）；(4) 在线 R3/R4 使用 QuIP# 的快速 Hadamard CUDA kernel。DartQuant 用 Whip Loss + QR-Orth 替代端到端微调加速旋转矩阵获取。

ResQ 扩展了 Computational Invariance 的使用——它在 PCA 排序后的高低精度子空间内分别应用随机旋转。关键区别：ResQ 的 U = PR（P 为 PCA 特征向量矩阵，R 为随机正交旋转），P 负责分配通道到高低精度组，R 在每个组内独立抑制 outliers。投影矩阵 U_A 融合到 block 边界权重（零运行时开销），U_B 融合到 attention value 路径（o_proj 左乘 U_B^T），U_C 因 RoPE 需在线计算但量化为 8-bit（key/query 对称投影保持 attention dot product 不变：q_proj K_proj^T = (q U_C)(U_C^T K^T) = q K^T），U_D 用 Hadamard 矩阵实现快速在线投影。ResQ 的 invariance 设计同时支持 4/8-bit 混合精度（而 QuaRot 仅支持统一 4-bit）。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

---

## Whip Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Whip Loss 是 DartQuant 提出的旋转分布校准损失函数：`Whip = Σ_{i=1}^{C_in} exp(-|x_i|)`（x 为旋转后激活向量）。设计灵感来自 Laplace→Uniform 的 CDF 变换 U_X(x) = τ[exp(x/b)-1]（x≤0）/ τ[1-exp(-x/b)]（x>0）。由于 LLM 激活近似 Laplace(0,b) 分布（论文 Appendix G 统计分析验证），exp(-|x_i|) 在零附近梯度大、远离零时梯度趋近于零。配合旋转矩阵的 norm-invariance 约束（||Rx||=||x||），Whip 将小值"推开"增大 → outliers 被迫减小保持 L2 范数不变 → 激活分布趋向均匀 → 量化误差降低。图 6 直方图验证 Whip 优化后的分布最接近均匀；图 7a 验证 Whip 的量化误差下降曲线远优于量化 loss/方差/峰度。

从算法pipeline角度拆解术语，给出具体例子。
```
# Whip Loss 在旋转校准中的使用 (DartQuant Algorithm 1)
for k = 0 to T_max:
    R = QR(Z)                       # QR-Orth 获得正交矩阵
    O = X @ R                        # 激活旋转: O ∈ R^{T × C_in}
    L = sum(exp(-|O_ij|))           # Whip Loss（所有 token 和 channel）
    Z = Z - lr * ∂L/∂Z              # SGD 更新隐参数
```

量化误差降低机制（4 维示例）：x=[x1≈0, x2≈0, x3≈0, x4>>0]。Whip 优化后 x1,x2,x3 绝对值增大。由 norm-invariance：√(x̃1²+x̃2²+x̃3²+x̃4²)=√(x1²+x2²+x3²+x4²)，前三项增大 → x̃4 必须减小（outlier 被抑制）。

术语一般如何实现？如何使用？
PyTorch 实现：`loss = torch.exp(-torch.abs(rotated_acts)).sum()`。超参：SGD, lr=1e-3~1e-2, epoch=10, batch=64（论文表 23）。校准仅需 128 样本 token 前向收集激活（无需标签），70B 模型校准仅 0.91 GPU-hours（SpinQuant 的 1/47）。局限性：(1) 假设激活近似零均值，偏差大时效果下降；(2) 针对均匀整数格式设计，FP4 等非均匀格式待验证。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

---

## QR-Orth Optimization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
QR-Orth 是 DartQuant 提出的正交矩阵优化方案：引入隐参数 Z ∈ R^{n×n}（无约束实矩阵），通过 QR 分解 R = Q(Z) 获得正交旋转矩阵。优化时用标准 SGD/Adam 在欧几里得空间更新 Z，梯度通过 QR 分解自动微分反向传播至 Z，校准完成后丢弃 Z 仅保留 R。相比 Cayley SGD（需在 Stiefel 流形上投影，额外 6n³ 计算量），QR-Orth 仅需 Householder QR 分解约 4/3 n³。100 步 SGD 实测：QR-Orth 5.7h vs Cayley 8.2h（1.44×）。因 Whip Loss 配合下收敛更快，QR-Orth SGD 6 步即达 Cayley SGD 100 步效果（41× effective 加速）。

从算法pipeline角度拆解术语，给出具体例子。
```
Z = random_hadamard(n)                     # 隐参数初始化
optimizer = SGD([Z], lr=lr)
for step in range(max_steps):
    R, _ = torch.linalg.qr(Z)              # QR: Z = Q·R_upper, R ← Q
    O = X @ R
    loss = exp(-|O|).sum()                 # Whip Loss
    loss.backward()                        # 梯度通过 QR 反向传播至 Z
    optimizer.step()
R_final, _ = torch.linalg.qr(Z)  # 最终旋转矩阵
del Z  # 丢弃隐参数
```

术语一般如何实现？如何使用？
直接使用 PyTorch `torch.linalg.qr(Z)`（内部调用 cuSOLVER/LAPACK 优化的 Householder QR）。Householder QR 计算复杂度约 4/3 n³（论文 Appendix B 详细推导）。与任意 PyTorch 标准优化器（SGD/Adam/AdamW）兼容。对 n ≤ 16384（主流 LLM hidden dim）的矩阵，QR 分解在 GPU 上可高效完成。通用技术，适用于任何需要优化正交矩阵的场景。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

---

## Cayley SGD for Orthogonal Matrix Optimization on Stiefel Manifold

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Cayley SGD 是一种在 Stiefel 流形 V_k(R^n) = {X ∈ R^{n×k} : X^T X = I} 上执行随机梯度下降的黎曼优化方法（Li et al., ICLR 2020）。Stiefel 流形是所有 n×k 列正交矩阵构成的黎曼子流形，在此流形上直接用标准 SGD 更新会破坏正交性。Cayley SGD 通过迭代 Cayley 变换实现高效 retraction（将欧几里得空间的更新投影回 Stiefel 流形）：Y = X + α/2 · W(X+Y)，其中 W=GX^T-XG^T 为切线空间中的反对称矩阵。相比闭式 Cayley 变换（需矩阵求逆 O(n³)），迭代形式仅需矩阵乘法，s=2-3 次迭代即可达到正交精度 ~10^{-5}。每次迭代增加约 6n³ 额外计算量（vs 标准 SGD），约 2× 慢于标准 SGD。SpinQuant 和 OSTQuant 使用 Cayley SGD 端到端微调旋转矩阵。

从算法pipeline角度拆解术语，给出具体例子。
```
# Cayley SGD with Momentum (DartQuant Appendix B.2, Algorithm 3)
M_{k+1} = β M_k - G(X_k)                              # 动量更新
W_hat = M_{k+1}@X_k^T - 0.5*X_k@(X_k^T@M_{k+1}@X_k^T) # 反对称: n³+2n³
W_k = W_hat - W_hat^T                                   # 确保反对称性
M_{k+1} = W_k @ X_k                                     # 动量投影: n³
α = min(lr, 2q/(||W_k||+ε))                             # 自适应步长 (q=0.5)
Y_0 = X_k + α M_{k+1}
for i=1 to s:                                           # 迭代 Cayley: 每次 n³
    Y_i = X_k + α/2 * W_k @ (X_k + Y_{i-1})
X_{k+1} = Y_s                                           # 新正交矩阵
总额外计算量: ≈ 6n³（vs 标准 SGD 的 O(n²)）
```

术语一般如何实现？如何使用？
开源实现：SpinQuant（https://github.com/facebookresearch/SpinQuant）的 Cayley SGD 优化器。在 DartQuant 实验中，Cayley SGD 100 步耗时 8.2h（Adam 8.1h）。主要开销：步骤 5（W_hat 计算 ~3n³）、步骤 7（动量投影 ~n³）、步骤 9-11（迭代 Cayley ~2n³）。DartQuant 提出 QR-Orth 替代 Cayley SGD：QR-Orth 100 步耗时 5.7h（1.44× 加速），且 Whip Loss 配合下 6 步即达 Cayley 100 步效果（41× 综合加速比）。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

---

## Entropy-Guided Quantization Optimization (Maximum Output Entropy)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Entropy-Guided Quantization Optimization（熵引导量化优化）是 Squat (EdgeQAT) 论文提出的量化感知训练技术。其理论基础来自 Messerschmitt (1971)：对高斯分布而言，最大化量化输出熵（Maximum Output Entropy, MOE）的量化器与最小化平均误差（Minimum Average Error, MAE）的量化器近似等价（仅差乘法常数）。因此，在QAT中最大化量化后query/key的信息熵等价于最小化量化误差。具体实现为损失函数 `L_E = -log(Σ_l Σ_h log(1 + σ_q²·σ_k²))`，其中σ_q²和σ_k²为各层各头query和key的方差。因为高斯分布的熵 H(q) = ½log(2πeσ_q²) ∝ σ_q²，通过增大方差来增大熵，同时log(1+σ_q²·σ_k²)的对数缩放防止梯度爆炸。该术语也可广义理解为基于信息论准则（如熵最大化）指导量化器设计的优化方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Squat中熵损失计算的流程（以LLaMA-58M单步训练为例）：
```
def entropy_loss(all_queries, all_keys, L_layers, H_heads):
    total_log_entropy = 0.0
    for l in range(L_layers):
        for h in range(H_heads):
            q = all_queries[l][h]  # [B, N, d_h]
            k = all_keys[l][h]
            var_q = q.var()         # σ_q²
            var_k = k.var()         # σ_k²
            total_log_entropy += log(1 + var_q * var_k)
    L_E = -log(total_log_entropy)
    return L_E
```
理论基础：H(q) = -Σ p(q_i)·log p(q_i) = ½log(2πeσ_q²)，最大化H(q) ∝ 最大化σ_q²。MOE ≈ MAE准则下最大化熵 = 最小化量化误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在QAT训练循环中，熵损失作为额外正则项：`L_total = L_distill + r_E·L_E + r_D·L_D`。Squat中r_E=0.5。实现注意事项：(1)方差在batch维度计算；(2)内层log(1+x)防止数值问题；(3)外层log确保与CE loss尺度匹配。该技术依赖query/key近似高斯的观测（SmoothQuant/Agile-Quant也发现类似特性），对非高斯分布等价性不严格成立但仍可作为正则化。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

---

## Distribution-Aligned Attention Map Distillation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution-Aligned Attention Map Distillation（分布对齐注意力图蒸馏）是 Squat 论文针对量化自注意力模块注意力图结构退化提出的蒸馏技术。论文发现（Figure 3）：FP16注意力图中存在明显的初始token列特征（distinct column pattern），但量化后该特征消失。为此引入分布损失 `L_D = log(Σ_l Σ_h (attn_q · attn_f) / (||attn_q||₂ · ||attn_f||₂))`，通过对数缩放匹配原始损失尺度。该术语可推广为通过分布层面对齐（注意力图、特征图等）来指导量化的蒸馏技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def distribution_loss(q_model, fp_teacher, input_ids):
    with torch.no_grad():
        fp_attns = fp_teacher(input_ids, output_attentions=True).attentions
    q_attns = q_model(input_ids, output_attentions=True).attentions
    total_cos = sum(
        (q_attns[l][:, h] * fp_attns[l][:, h]).sum() /
        (q_attns[l][:, h].norm() * fp_attns[l][:, h].norm())
        for l in range(L) for h in range(H)
    )
    return log(total_cos)  # L_D
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
需要FP16教师模型提供参考注意力图。Squat中r_D=1.0（权重大于L_E）。推理时无此开销（仅训练阶段使用）。可与熵损失L_E联合使用获最佳效果。消融显示L_D单独使用比L_E更有效地恢复精度。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

## Uniform Quantization（均匀量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Uniform Quantization（均匀量化）是将连续的浮点数值映射到等间距离散值的量化方式。其核心公式为：W_int = clamp(round(W/s) + z, 0, 2^N - 1)，反量化 Ŵ = (W_int - z)·s。其中 s 为步长（step size/scaling factor），z 为零点（zero point），N 为目标位宽。均匀量化的量化级别呈等差数列分布，与vector quantization（使用codebook中非均匀离散值）相对。均匀量化的优势在于硬件友好性——标准的INT MAC单元可直接处理均匀量化的整数值，无需额外的codebook查表或比特转置，因此被GPTQ、AWQ、OmniQuant、EfficientQAT等主流方法广泛采用。但在极低位宽（2-bit）下，均匀量化因表达能力有限（仅4个离散值），精度低于QuIP#等vector quantization，后者的复杂codebook设计能更精确逼近原始分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在EfficientQAT的Block-AP中，均匀量化以per-group方式应用于每个Linear层权重：
```
# 每组g个权重共享s(FP16)和z(N-bit)
for group in split(weight, group_size=g):
    s = group.abs().max() / (2^N - 1)          # 初始化步长
    z = -group.min() / s                        # 初始化零点
    # 前向量化（包含在计算图中，可反向传播）
    W_int = clamp(round(W/s) + z, 0, 2^N - 1)   # N-bit 整数
    W_hat = (W_int - z) * s                      # 反量化为FP16
    output = matmul(x, W_hat)                    # FP16矩阵乘法
```
平均位宽公式：bits/param = N + (N+16)/g。例如g=64时：2-bit → 2.28 bits/param，3-bit → 3.30 bits/param，4-bit → 4.31 bits/param。高效硬件兼容性是均匀量化的核心优势：与vector quantization（AQLM、QuIP#）不同——vector量化仅限于weight-only，且需专用kernel处理codebook查表——均匀量化可同时压缩权重和激活，兼容MLC-LLM、AWQ、BitBLAS、Marlin、T-MAC等标准推理框架。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
均匀量化的实现层次：(1) 对称均匀量化（z=0，仅需s）：最简单，但假设权重零均值对称分布；(2) 非对称均匀量化（含s和z）：可处理偏态分布，额外存储一个N-bit零点每group；(3) per-tensor/per-channel/per-group：共享s/z的粒度越细精度越高但存储开销越大。在EfficientQAT中，per-group g=64在精度和存储间取得平衡（2-bit w2g64压缩比≈82% vs FP16）。实现上，PyTorch中通过自定义autograd Function打包量化/反量化为单一操作符，推理时使用packing格式存储（每8个4-bit权重打包为1字节，或每4个2-bit权重打包为1字节）。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

SLiM (ICML 2025) 提出了均匀量化的概率化最优 scaling 方法 SLiM-Quant，解决了传统 symmetric uniform quantization 中 AbsMax 对 outlier 敏感、Grid Search 次优且昂贵的痛点。SLiM-Quant 将 non-convex 的 MSE 最小化目标通过概率化重表述转化为在权重直方图上的数值积分，利用多网格策略高效找到全局最优 α*，使 uniform quantization 精度达到 group quantization 水平，同时保留 uniform 的硬件友好性（单 scale per tensor, 6% 实测加速 vs group）。

---

## Block-AP (Block-wise Training of All Parameters)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-AP (Block-wise Training of All Parameters) 是 EfficientQAT 提出的两阶段 QAT 框架的第一阶段，是在 block-wise reconstruction 框架下首次直接训练所有权重和量化参数的方案。与 BRECQ/OmniQuant/AutoRound 等方法不同——这些方法仅训练辅助量化参数（rounding参数、clipping阈值或步长）以限制优化空间防止过拟合——Block-AP 将标准 QAT 的"全训练"引入 block-wise 框架，同时训练原始权重 W、步长 s 和零点 z。Block-AP 的核心发现是：在 block-wise 重建中，全参数训练无需复杂的可训练参数设计（如 AdaRound 的学习取整），即可取得显著优于部分训练的结果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Block-AP 逐block训练流程
def block_ap(model, calib_data, epochs=2, lr_W=2e-5, lr_s=1e-4):
    for block_idx, block in enumerate(model.blocks):
        fp16_output = block(calib_data)           # 保存FP16输出（训练目标）
        # 初始化block内所有Linear层的s和z
        for linear in block.linears:
            init_scales_and_zeros(linear.W, bit=N, group_size=g)
        for ep in range(epochs):
            for batch in calib_data:               # 4096 samples
                # 前向：量化block内所有权重
                for linear in block.linears:
                    linear.W_int = clamp(round(linear.W/linear.s) + linear.z, 0, 2^N-1)
                    linear.W_hat = (linear.W_int - linear.z) * linear.s
                output = block(input, use_W_hat=True)
                loss = MSE(output, fp16_output)    # 重建损失
                # STE反向传播，更新W, s, z
                loss.backward()
                optim_W.step()  # lr_W: 2e-5(2bit), 1e-5(3/4bit)
                optim_sz.step() # lr_sz: 1e-4
    return model  # 输出量化模型 (W_q in N-bit, s in FP16, z in N-bit)
```
训练超参：batch_size=2, epochs=2, 校准数据4096样本（RedPajama, ctx=2048），训练后输出W_q(N-bit)、s(FP16)、z(N-bit)。Table 4消融：Block-AP单独使用即可将W2G64 Avg PPL从453.49降至8.53，Avg Acc从40.69%恢复至58.99%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block-AP的实现依赖STE（采用LSQ+的三梯度公式Eq.3-5），通过自定义autograd Function将量化/反量化嵌入计算图。每block训练需以下内存：权重W(FP16) ≈ 202.4M参数、量化参数s+z ≈ 6.3M参数（Llama-2-7B单block）。Block-AP不使用Adam/AdamW等状态优化器——论文发现对于block-wise重建任务，SGD足以收敛且无需额外状态内存。与E2E-QP的关系：Block-AP提供高质量权重初始化，E2E-QP在此基础上做端到端微调；Table 4显示两者组合(Long) PPL=7.68 vs Block-AP单独=8.53 vs E2E-QP单独=9.33。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models

## E2E-QP (End-to-End Training of Quantization Parameters)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
E2E-QP (End-to-End Training of Quantization Parameters) 是 EfficientQAT 两阶段框架的第二阶段，冻结Block-AP产出的量化权重 W_q 和零点 z，仅端到端训练步长 s。其设计动机：Block-AP逐block独立训练忽略了跨block交互——各block输出的量化误差会级联放大，导致全局性能下降。E2E-QP通过端到端训练步长s来补偿这种跨block误差传播。由于步长s参数量极小（约占1.6% at g=64），E2E-QP的内存/时间开销远低于传统端到端QAT。例如Llama-2-70B W2G64的E2E-QP仅需34.2GB显存和14.3h训练时间（单A100-80GB）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# E2E-QP 端到端仅训练步长s
def e2e_qp(model, train_data, epochs=1, lr_s=2e-5):
    # 冻结所有权重
    for param in model.parameters():
        param.requires_grad = False
    # 仅解冻步长s（每个quantized group的scale factor）
    for step_size in model.step_sizes():
        step_size.requires_grad = True
    for batch in train_data:                     # 4096 samples, ctx=4096
        # 前向：仅反量化（无Eq.1量化过程）
        for linear in model.linears:
            linear.W_hat = (linear.W_q - linear.z) * linear.s
            # 梯度 ∂W_hat/∂s = W_q - z  （无需STE，直接解析梯度）
        output = model(input)                    # 端到端前向
        loss = cross_entropy(output, labels)     # LM损失
        loss.backward()                          # 仅s接收梯度
        optimizer_s.step()                       # lr_s: 2e-5(2bit), 1e-5(3bit)
    return model
```
关键设计：(1) E2E-QP中不执行量化操作(W_int=clamp(round(W/s)+z,...))，仅执行反量化(Ŵ=(W_q-z)*s)——因此W_q冻结不变；(2) s的梯度 ∂Ŵ/∂s = W_q - z 为解析梯度，无需STE近似；(3) Table 6显示训练s、z或s+z性能相近（s PPL=7.68, z PPL=7.69, s+z PPL=7.68），默认仅训练s以最小化额外位宽（z从N-bit展开为FP16会增加0.22 bits/param）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
E2E-QP 可适配多种下游场景：(1) 量化训练：使用RedPajama 4096样本，batch=32, epoch=1, lr_s=2e-5(2-bit)/1e-5(3-bit)；(2) 指令微调：替换训练数据为Alpaca，batch=16, 10000 steps, src_ctx=384, tgt_ctx=128；(3) 多模态微调：配合LLaVA-1.5 pipeline，冻结LLM + 预训练projector，再端到端微调LLM和projector（lr=2e-5(4-bit)/3e-5(2/3-bit)）。Table 8显示增大E2E-QP样本数可持续降低PPL（128→32764 samples: PPL 8.09→7.50），但Avg Acc在4096后不再显著提升。结合Block-AP + E2E-QP的完整EfficientQAT流程可使Llama-2-70B W2G64在单A100-80GB上41h内完成。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models

---

## FOEM (First-Order Enhanced Method) / First-Order Error Compensation in PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FOEM 是 Zheng et al. (AAAI 2026) 提出的改进 GPTQ 的 PTQ 方法。核心创新：在权重量化的逐列误差补偿中**显式保留并近似一阶梯度项**。传统 GPTQ 沿袭 OBD/OBS 假设——全精度模型已收敛到局部最优，因此一阶梯度 g ≈ 0，仅用二阶 Hessian 项建模量化误差并补偿。FOEM 指出：逐列量化过程中，先量化列的补偿项持续更新后续 latent weights，导致 W 偏离原始 full-precision 权重 𝕎，产生不可忽略的一阶梯度。FOEM 通过 Taylor 展开近似 g(W) ≈ β(W − 𝕎)H（β=0.1 为稳定化因子），代入 Lagrangian 约束优化求解后，H 和 H^{−1} 在代数中自动消去，最终仅需 Cholesky 因子 T 和权重差分运算，无需显式 Hessian 计算或反向传播。开销极小（Llama3-8B 量化时间仅从 GPTQ 的 825.50s 增至 828.90s，+0.4%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FOEM 逐列量化流程（对比 GPTQ 仅二阶补偿）：

```
# === FOEM: 一阶增强的逐层量化 ===
# 输入: FP权重 W (m×n), 校准输入 X, block size B, β=0.1
H = X @ X.T                         # Hessian: 2XX^T（对称矩阵，n×n）
L = Inverse_Cholesky(H + λI)        # Cholesky 分解, H^{-1} = L L^T
T = L.T                              # 上三角矩阵 T = L^T
W_orig = W.clone()                   # 保存原始全精度权重 𝕎
Q = zeros(m, n)                      # 量化后权重
E = zeros(m, B)                      # block 误差矩阵

for i in range(0, n, B):            # 按 block 迭代
    for j in range(i, i+B):          # block 内逐列量化
        Q[:, j] = quant(W[:, j])     # RTN 量化当前列
        # === 一阶增强误差（GPTQ 仅 (w_q-ŵ_q)/T_{jj}）===
        E[:, j-i] = ((W[:, j] - Q[:, j])
                     - β * (W[:, j] - W_orig[:, j])) / T[j, j]
        # 补偿 block 内后续列（含一阶修正项）
        W[:, j:(i+B)] -= E[:, j-i].unsqueeze(1) * T[j, j:(i+B)]
        W[:, j:(i+B)] -= β * (W[:, j] - W_orig[:, j])
    # 补偿 block 外后续列 (lazy batch update)
    W[:, (i+B):] -= E @ T[i:(i+B), (i+B):]
```

**数学推导链**：
1. 保留一阶项：δE = g δw^T + ½ δw H δw^T
2. 带约束 Lagrangian：ℒ = g δw^T + ½ δw H δw^T + λ(e_q δw^T + w_q − ŵ_q)
3. 求导得最优：δw = −(w_q − ŵ_q − g H^{-1} e_q^T) / [H^{-1}]_{qq} · [H^{-1}]_{q,:} − g H^{-1}
4. 梯度近似：g ≈ β(W − 𝕎)H（g(𝕎)≈0 → Taylor 展开 g(W) ≈ g(𝕎) + (W−𝕎)H）
5. 代入消去 H/H^{-1}：δw = −((w_q − ŵ_q) − β(w_q − 𝕎 e_q^T)) / T_{qq} · T_{q,q:} − β(W − 𝕎)

与 GPTQ 的核心差异：分子中多减 β(w_q − 𝕎 e_q^T)，全局多减 β(W − 𝕎)。β=0 退化为 GPTQ。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/Xingyu-Zheng/FOEM，基于 gptqmodel 库集成。使用方式：
```python
from gptqmodel import GPTQModel, QuantizeConfig, FOEMConfig
foem_config = FOEMConfig(alpha=0, beta=0.1, device="cuda")
quant_config = QuantizeConfig(bits=4, group_size=128, foem_config=foem_config)
model = GPTQModel.load("meta-llama/Llama-3-8B", quant_config)
model.quantize(calibration_dataset)
```
β≤0.5 持续有效；β>0.5 因近似误差放大导致性能退化。FOEM 可与 SpinQuant 旋转矩阵无缝结合，在 W4A4KV4 下进一步缩小与 FP16 的差距。跨架构有效：在 SSM 模型 Mamba-1.4B 上 W3A16 PPL 从 GPTAQ 14.10 降至 FOEM 13.91。校准数据：C4 128 samples, seq_len=2048。评估：WikiText2/C4 PPL + 7 zero-shot benchmarks + 5-shot MMLU。

涉及论文标题：
- First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

---

## OBD/OBS/OBC Weight Compensation Framework（最优脑损伤/脑外科/脑压缩框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OBD (Optimal Brain Damage, LeCun et al. 1990) → OBS (Optimal Brain Surgeon, Hassibi & Stork 1993) → OBC/OBQ (Optimal Brain Compression, Frantar & Alistarh 2022) → GPTQ (Frantar et al. 2022) 是一条从神经网络剪枝演化到 LLM 量化的理论链。核心思想：使用二阶 Taylor 展开 δE = g δw^T + ½ δw H δw^T 估计参数移除（剪枝或量化）对损失函数的影响，并通过 Hessian 矩阵信息将误差补偿到剩余参数中。

各阶段演变：
- **OBD**：假设模型已收敛（g≈0）、参数独立（仅 Hessian 对角）、损失近似二次。Saliency s_k = ½ h_{kk} w_k²。缺陷：忽略参数间交互。
- **OBS**：使用全 Hessian 矩阵。Lagrangian 约束优化得 δw = −w_q/[H^{-1}]_{qq} · [H^{-1}]_{q,:}，Saliency ΔE = ½ w_q²/[H^{-1}]_{qq}。缺陷：O(n³)，每次剪枝需重算 H^{-1}。
- **OBC**：限制到权重矩阵逐行优化（每行独立，H = 2XX^T）。扩展至量化：δw = −(w_q − ŵ_q)/[H^{-1}]_{qq} · [H^{-1}]_{q,:}。通过 Gaussian 消元高效更新 H^{-1}。
- **GPTQ**：OBC 的 LLM 适配，lazy batch 更新 + Cholesky 分解：δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}。
- **GPTAQ**（本文）：将 OBC 的优化目标从对称校准 `||(w+Δw)X − wX||²` 扩展为非对称校准 `||(w+Δw)X − wX̃||²`（X̃ 是全精度输入）。这引入残差 r = wX̃ − wX，推导得 δw = −(ŵ_q − w_q)/H_{qq}^{-1} · H_{q,:}^{-1} + r X^T H_{-q}^{-1}（两分量：量化误差补偿 + 前层累积偏差补偿）。通过残差分解 R = Σ W_{:,q} ΔX_{q,:} 和 Theorem 4.2（P = ((ΔX X^T L) ⊙ M_U) L^T）实现高效 GPU 并行。
- **FOEM**（本文）：指出因累积补偿，一阶项不可忽略。保留 g 重新推导，通过 g ≈ β(W−𝕎)H 近似后 Hessian 自动消去，仅增加权重差分运算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
OBS Lagrangian 求解推导（从 OBS 到 FOEM 的演变基础）：

```
# OBS 约束优化（剪枝参数 q 为零）
min_{δw} ½ δw H δw^T
s.t. e_q δw^T + w_q = 0
ℒ = ½ δw H δw^T + λ(e_q δw^T + w_q)
∂ℒ/∂δw = δw H + λ e_q = 0  →  δw = −λ e_q H^{-1}
∂ℒ/∂λ = e_q δw^T + w_q = 0  →  λ = w_q / [H^{-1}]_{qq}
∴ δw = −w_q/[H^{-1}]_{qq} · [H^{-1}]_{q,:}  （最优补偿）
   ΔE = ½ w_q²/[H^{-1}]_{qq}                （剪枝 saliency）

# OBC/OBQ: 量化版本（w_q → ŵ_q）
δw = −(w_q − ŵ_q)/[H^{-1}]_{qq} · [H^{-1}]_{q,:}

# GPTQ: Cholesky 分解 H^{-1} = LL^T, T = L^T
δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}

# FOEM: 保留一阶项 g
ℒ = g δw^T + ½ δw H δw^T + λ(e_q δw^T + w_q − ŵ_q)
∂ℒ/∂δw = g + δw H + λ e_q = 0  →  δw = −(g + λ e_q) H^{-1}
∂ℒ/∂λ = e_q δw^T + w_q − ŵ_q = 0  →  λ = (ŵ_q − w_q − g H^{-1} e_q^T)/[H^{-1}]_{qq}
∴ δw = −(w_q − ŵ_q − g H^{-1} e_q^T)/[H^{-1}]_{qq} · [H^{-1}]_{q,:} − g H^{-1}
# 梯度近似: g ≈ β(W − 𝕎)H, 代入后 H/H^{-1} 消去
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
OBD/OBS 最初用于小网络剪枝（LeNet、AlexNet）。OBC/OBQ 和 GPTQ 扩展至 LLM 量化。GPTQ 的 AutoGPTQ（https://github.com/PanQiWei/AutoGPTQ）是最广泛使用的实现，FOEM 集成于 gptqmodel 库。实践中 Hessian 从少量校准数据（128 samples, seq_len=2048）计算：H = 2XX^T，加 λI 正则化防止奇异。Cholesky 分解后保留上三角 T 用于补偿。该框架直接用于 weight-only 量化（W4A16/W3A16, group_size=128），配合旋转技术（QuaRot、SpinQuant）可扩展至 weight-activation 量化（W4A4KV4）。GPTAQ 进一步扩展 OBC 为非对称校准：在校准中同时使用全精度前向的激活 X̃ 和量化后激活 X，计算 ΔX = X̃ − X，引入残留误差补偿项 `W_{:,q} P_{q,:}` 纠正前层累积的激活偏差。核心优势：仅需少量校准数据（无需训练数据），一次前向计算 Hessian 即可。

SpQR 将 OBS 的敏感度准则直接嵌入量化过程：敏感度 s_ij = (w_ij − quant(w_ij))² / (2[H⁻¹]_jj)，在 GPTQ 逐列量化过程中动态计算（而非预处理）。关键创新：(1) 敏感度捕获了权重间的相关性——某权重大rounding error可被其他未量化权重的连续值优化补偿；(2) outlier检测通过leave-one-out error对比（E_base − E_ol > τ）在量化过程中完成，outlier不仅是"初始敏感"权重，还包括能补偿其他权重量化误差的权重；(3) 调整后的权重值（而非原始值）被保留为16-bit，包含GPTQ误差补偿过程的累积效应。

涉及论文标题：
- First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
- GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

## Asymmetric Calibration (非对称校准) in Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
非对称校准（Asymmetric Calibration）是 GPTAQ 提出的 PTQ 校准范式改进。传统 GPTQ 使用**对称校准（Symmetric Calibration）**：每层独立最小化 `||(w+Δw)X − wX||²`，其中 X 是前一层量化后的输出——即假设"当前层的输入已经是正确的"，仅优化当前层的局部量化误差。非对称校准将该目标改为 `||(w+Δw)X − wX̃||²`，其中 X̃ 是前一层全精度模型输出的激活（"ground truth" reference），X 是前一层量化后的实际输出。两者的差异 ΔX = X̃ − X 来自前层权重和激活量化的累积误差，沿网络深度逐渐放大（GPTAQ Fig.2a 验证）。GPTAQ 通过引入残差 r = wX̃ − wX = wΔX，在 Lagrangian 约束优化框架中推导出包含两项的最优权重更新：量化误差补偿项（与 GPTQ 相同）+ 残留误差补偿项 `r X^T H_{-q}^{-1}`。实验验证（Table 5）：仅用第二项（残留补偿）的 GPTAQ' 就能在零样本准确率上超越 GPTQ（69.0% vs 67.1%），两项联合的完整 GPTAQ 最优（69.6%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
非对称校准的核心差异：

```
# 对称校准（GPTQ）：target = wX（量化后输入的输出）
min ||(w+Δw)X - wX||²              # wX = 当前层的"局部正确"输出
# → δw = -(ŵ_q - w_q)/H_{qq}^{-1} · H_{q,:}^{-1}

# 非对称校准（GPTAQ）：target = wX̃（全精度输入的输出）
min ||(w+Δw)X - wX̃||²              # wX̃ = 全精度模型的"全局正确"输出
# 引入 r = wX̃ - wX = wΔX（输入偏差在输出空间的投影）
# Lagrangian: L = ||Δw X - r||² + λ(e_q Δw^T + w_q - ŵ_q)
# → δw = -(ŵ_q - w_q)/H_{qq}^{-1} · H_{q,:}^{-1} + r X^T H_{-q}^{-1}
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
#        量化误差补偿项 (GPTQ term)                  残留误差补偿项 (new)
```

**Annotations**: X 是量化后激活（actual），X̃ 是全精度激活（reference）。r = wΔX 对应当前层权重的全精度输出与量化后输入的输出的偏差——即使权重未量化，仅因输入偏差就会产生输出偏差。第二项 `r X^T H_{-q}^{-1}` 将这一偏差通过 Hessian 逆矩阵回传到剩余未量化权重中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTAQ 的非对称校准通过 4 个优化步骤高效实现：(1) **任意顺序处理**——放弃每列选择最优 q 的贪心策略，改为从左到右固定顺序，支持所有输出通道并行；(2) **残差分解**——R = Σ_{q=1}^n W_{:,q} ΔX_{q,:}，将 R 分解为 n 个独立神经元分量，预计算一次 ΔX 后消除重复的 R 评估；(3) **Cholesky 重构化 + Theorem 4.2**——P = ((ΔX X^T L) ⊙ M_U) L^T，将 P 矩阵计算融合为一行 GPU 友好代码；(4) **Lazy-Batch 更新**——block 后批量更新 block 外列。使用时需注意：激活量化应在权重量化之前执行（A→W 顺序），使 ΔX 包含激活量化信息；X̃ 临时存储通过逐 block 进出 GPU 优化（Algorithm 2），LLaMA2-7B 约 12GB 临时内存。GPTAQ 实现仅比 GPTQ 多约 20 行代码（主要在 P 矩阵计算和 lazy-batch 中的第二项），额外延迟大维度时 30-40%、小维度时 <10%。

涉及论文标题：
- GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration

## Residual Decomposition in Quantization Calibration (量化校准中的残差分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
残差分解（Residual Decomposition）是 GPTAQ 中解决非对称校准效率瓶颈的关键技术。在非对称校准中，每次量化一列权重后需要重新评估输出残差 R = W X̃ − W X。直接计算复杂度为 O(mnk)，其中 k（token 数 × 校准样本数）远大于 m 和 n（例如 LLaMA2-7B 中 k = 128×2048 ≈ 262K，而 n = 4096），重复计算 R 将导致不可接受的量化时间。GPTAQ 的关键洞察：输出残差 R 可分解为 n 个独立神经元分量的和——R = W ΔX = Σ_{q=1}^n W_{:,q} ΔX_{q,:}。这样：(1) 可一次性计算 ΔX = X̃ − X（O(nk)）；(2) 第 q 次迭代仅关注第 q 个神经元对残差的贡献 `W_{:,q} ΔX_{q,:}`，而非全部 n 个分量；(3) 优化目标简化为 `min ||ΔW_{:,q:} X_{q:,:} − W_{:,q} ΔX_{q:,:}||²`，最优权重更新包含 `W_{:,q} ΔX_{q,:} X_{:,q:}^T H_{-q}^{-1}` 项；(4) 由于 `ΔX_{q,:}X_{:,q:}^T H_{-q:}^{-1}` 与权重更新无关，可预计算为 P 矩阵的对应行。这一分解将复杂度从 O(mn²k) 降至 O(mn² + n²k)（降低 n 倍），使得 GPTAQ 的非对称校准在实际中可用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
残差分解的数学转化：

```
# 直接法（不可行）：
for q in range(n):                  # n = 4096 (hidden_dim)
    quantize_and_update(W[:, q])
    R = W @ (X̃ - X)                 # O(mnk), k=262K → 每次迭代 ~43GB 运算
    # 对全矩阵 R 做 Hessian 逆投影
    correction = R @ X.T @ H_inv    # O(mn² + n²k), 又一大运算
# 总复杂度: O(n × mnk) = O(mn²k)，完全不实际

# 残差分解法（GPTAQ）：
# 预计算（一次性）:
ΔX = X̃ - X                         # O(nk)
ΔX_XT = ΔX @ X.T                   # O(n²k)

# 将 R 写为分解形式:
# R = W @ ΔX = Σ_{q=1}^n W[:,q] × ΔX[q,:]
#   = sum of n rank-1 matrices, each contributed by one neuron

# 第 q 次迭代:
# 仅关注第 q 个神经元残差分量: W[:,q] × ΔX[q,:]
# 优化目标: min ||ΔW[:,q:] @ X[q:,:] − W[:,q] @ ΔX[q:,:]||²
# 权重更新 (Eq. 15):
ΔW[:,q:] = GPTQ_term + W[:,q] @ (ΔX[q,:] @ X[q:,:].T @ H_{q:}^{-1})
#                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                           P[q,q:] 的第 q 行, 可预计算!

# P 矩阵通过 Theorem 4.2 一次性并行计算:
P = ((ΔX_XT @ L) ⊙ M_U) @ L.T     # GPU 上 <1ms (Fig.4a)
```

**Annotations**: 关键洞察是 R 可以按神经元维度分解——每个输出通道（W 的每一行）对 R 的贡献来自输入通道（W 的每一列）× 该输入通道的激活偏差 ΔX[q,:]。因此第 q 次迭代只处理与第 q 个输入神经元相关的分量，而第 1..q-1 个神经元的权重已被固定（量化完成），相应的残差分量已被"吸收"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
残差分解在 GPTAQ 中的实现与 Cholesky 重构化协同：(1) ΔX X^T 矩阵在校准循环外一次性计算（n×n 矩阵，对 hidden_dim=4096 的层约 64MB FP16）；(2) 利用 Cholesky 分解的 L 矩阵和 Theorem 4.2 将 P 矩阵计算并行化为一句话：`P = ((ΔX_XT @ L) * mask_upper_tri) @ L.T`；(3) 在 lazy-batch 更新中直接切片 P[Q,Q:] 与 W[:,Q] 做 rank-1 更新。残差分解的前提是激活量化在权重量化之前执行（A→W 顺序），使得 ΔX 不仅是权重偏差还包含激活量化偏差。GPTAQ 的内存分析：ΔX 临时需要 O(nk) 存储（LLaMA2-7B 约 12GB），但逐 block 处理后可释放；P 矩阵 O(n²)（每层 0.16-0.70GB）需保留在 GPU 内存中供 lazy-batch 迭代使用。残差分解和 Cholesky 重构化、lazy-batch 更新的组合使得 GPTAQ 的额外延迟控制在大维度时 30-40%、小维度时 <10%，而非比 GPTQ 慢 n 倍。

涉及论文标题：
- GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration



---

## Vector Quantization (VQ) for LLM Weight Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vector Quantization（向量量化）是一种将高维浮点向量映射到有限个质心（centroid）的非均匀量化方法。与均匀量化（uniform quantization）将每个标量映射到等间隔固定点不同，VQ 将 d 个连续浮点值作为一个 d 维向量整体量化，映射到预先学习的 codebook C = {c_1, c_2, ..., c_k} 中的一个质心。每个质心 c_i ∈ R^d 是一个 d 维向量，存储 d 个浮点值。原始权重矩阵被划分为多个 group，每个 group 共享一个 codebook。存储时，每个 d 维向量仅需 ⌈log₂(k)⌉ 比特索引（加上 codebook 和 scale 的开销）。VQ 的核心优势：质心可以自由分布在 d 维空间中任意位置（非均匀 grid），能更灵活地匹配权重分布，在相同 bit budget 下可获得比均匀量化更高的 signal-to-quantization-noise ratio (SQNR)。对 LLM 推理，VQ 作为 storage data type（存储格式），推理时需先解码到 native data type 再参与计算。

GPTVQ 论文证实：d 越高（1D→2D→4D），SQNR 越高。但 codebook 大小随 d 指数增长（k = 2^{b·d}，b 为 bits/dimension），因此需在 codebook 大小、解码开销和精度之间权衡。GPTVQ 选择 2D VQ + 6-bit index（3 bits/dim）作为移动端最佳平衡点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GPTVQ 中 2D VQ 的编码/解码流程：

**编码（量化）阶段**（离线，Algorithm 2/VQ_quant）：
```
# 输入: 权重矩阵 W[:, P] (r × d, d=2), codebook C_g (d × k)
# 输出: 量化权重 Q[:, P] + 索引矩阵 I
for each d-dim vector x = W[row, P]:
    # 找最优质心（E-step 公式 5）
    c_idx = argmin_m (x - c_m)^T D (x - c_m)
    # D = diag(1/H^{-1}_{11}, ..., 1/H^{-1}_{dd}) (Hessian 加权)
    Q[row, P] = C_g[:, c_idx]
    I[row] = c_idx  # 存储 6-bit index
```

**解码（推理）阶段**（在线，移动 CPU）：
```
# 输入: 6-bit index i, LUT C_g (64 entries × 8-bit), scale s
v1 = TBL(C_g_dim0, i)  # 查表: dimension 0 的 8-bit 值
v2 = TBL(C_g_dim1, i)  # 查表: dimension 1 的 8-bit 值
w_fp = s * (v1 + v2)   # 合并两维 + 反量化
```

格式总 bpv 计算：bpv = log₂(k)/d + k·d·b_c/l，其中 k=质心数，d=VQ 维度，b_c=codebook bit-width，l=group size。GPTVQ 默认 k=64, d=2, b_c=8, l=8192，得 bpv = 6/2 + 64·2·8/8192 = 3 + 0.125 = 3.125。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VQ for LLM 的代表性方法：AQLM (Egiazarian et al. 2024) 使用 d=8 + 16-bit indices + block fine-tuning，适合云端 GPU；GPTVQ (van Baalen et al. 2024) 使用 d=2 + 6-bit indices 移动 CPU TBL 指令，适合移动端；QuIP# (Tseng et al. 2024) 使用 lattice codebook + Hadamard 旋转。实现框架：GPTVQ 基于 PyTorch（量化）+ 自研 C 推理引擎（CPU 解码）。关键实现考量：(1) codebook 大小受硬件 LUT 指令限制（移动 CPU TBL 仅支持 5-6 bit index）；(2) d 越大 SQNR 越高但解码越慢（需更多 TBL 调用）；(3) codebook 本身也需量化（INT8/INT4）以减少 overhead。

**VQ for MoE LLMs（KBVQ-MoE 的扩展）**：KBVQ-MoE (ICLR 2026) 将 VQ 专门适配到 MoE 架构。核心区别：(1) 不直接对所有 expert 权重做 VQ，而是先通过 KLT 引导的 SVD（IDRE）提取共享低秩分量保持全精度，仅对 expert-specific 残差做 VQ；(2) VQ 后通过 per-channel affine compensation（BCOS）校正量化输出的 mean/variance 偏移。KBVQ-MoE 使用 K-means++ 初始化 codebook（100 iterations），子向量长度 d=4，在 Qwen1.5-MoE-A2.7B 上 2-bit VQ 实现 87% 压缩率（27.9GB→4.3GB），3-bit 下 Avg Acc 67.99 接近 FP16 的 68.07。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---

## EM Algorithm for Codebook Initialization (Hessian-Weighted)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EM（Expectation-Maximization）Algorithm for Codebook Initialization 是 GPTVQ 中用于初始化 VQ codebook 的聚类算法。与标准 k-means 不同，GPTVQ 的 EM 算法引入 Hessian 加权的距离度量：目标函数为加权平方距离 min_{I,c} Σ_m Σ_{i∈I_m} (x^{(i)} - c^{(m)})^T D^{(i)} (x^{(i)} - c^{(m)})，其中 D^{(i)} = diag(1/H^{-1}_{jj}, ...) 是从校准数据的 Hessian 逆矩阵提取的对角权重矩阵。D^{(i)} 衡量每个维度的"重要性"——H^{-1} 对角元越大（该维度对输出影响越大），该维度的量化误差惩罚越大。E-step 用加权距离（公式 5）分配每个 d 维向量到最优质心；M-step 用 Moore-Penrose 伪逆闭式解 c^{(m)} = (Σ D^{(i)})^{+} Σ D^{(i)} x^{(i)} 更新质心。GPTVQ 默认使用 100 次 EM 迭代（消融显示继续迭代仍有小幅改善）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GPTVQ EM 初始化伪代码
# 输入: 权重矩阵 W (r × m), 逆 Hessian 对角 D, codebook 大小 k, VQ 维度 d
# 输出: codebook C ∈ R^{d × k}

# 初始化质心（Mahalanobis 或 k-Means++ seeding）
C = initialize_centroids(W_reshaped_to_d_vectors, k)

for iter in range(100):  # 100 次迭代（默认）
    # === E-step: 分配最优质心 ===
    for each d-dim vector x_i in W:
        best_m = 0, best_dist = inf
        for m in range(k):
            diff = x_i - C[:, m]  # [d]
            # Hessian 加权距离: d_j = 1/H^{-1}_{jj}
            dist = Σ_{j=0}^{d-1} diff[j]^2 / H_inv_diag[j]
            if dist < best_dist:
                best_m, best_dist = m, dist
        assignments[i] = best_m
    
    # === M-step: 闭式解更新质心 ===
    for m in range(k):
        # 收集分配给质心 m 的所有向量
        X_m = [x_i for i where assignments[i] == m]
        if len(X_m) == 0: continue
        # 伪逆闭式解: c_m = (Σ D_i)^{+} · (Σ D_i · x_i)
        sum_D = Σ_i D_i              # [d × d] 对角矩阵
        sum_Dx = Σ_i D_i @ x_i       # [d]
        C[:, m] = pinv(sum_D) @ sum_Dx

return C
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 的 EM 实现使用 Mahalanobis initialization 作为 seeding 方法（比 k-Means++ 快 3-5×）：按马氏距离排序所有数据点，均匀采样 k 个质心，确保质心在代表性距离上分布。PyTorch 实现，支持 GPU 加速。D^{(i)} 在实际中简化为仅使用逆 Hessian 对角元（而非完整 d×d 子矩阵），论文发现两者性能无差异。EM 是 GPTVQ 中最耗时的步骤（对高 d 尤为显著），占总运行时间的 50%+。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---

## Codebook Update via Gradient Descent

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Codebook Update via Gradient Descent 是 GPTVQ 在完成 Algorithm 1 的逐 block 量化后，对 codebook 值进行梯度下降微调的优化步骤。GPTVQ 的逐 block 量化（Algorithm 1+2）已经确定了每个权重的质心索引 I，但质心值 C 是贪心初始化的。通过固定索引 I，以层输出 MSE 为目标进一步优化质心值 C：min_C ||WX - Q(C)X||²_F，其中 Q(C) 是基于 C 和固定索引 I 的查找重建操作（look-up operation）。该目标对 C 是凸二次规划问题，可用闭式解但由于大矩阵求逆开销高，GPTVQ 使用 PyTorch 梯度下降代替——每步更新 C 后重建 Q，梯度 ∂Q/∂C 因查找操作而简单定义。消融（Table 15）显示 codebook update 在所有 setting 下降低最终 perplexity，代价是中等额外运行时间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GPTVQ Codebook Update 伪代码
# 前提: Algorithm 1 和 2 已完成，索引 I 已确定，Q(W) 已构造
# 输入: 原始权重 W，校准输入 X，当前质心 C，索引 I
# 输出: 优化后的质心 C_optimized

# 构建基于 C 和 I 的查找重建函数
def reconstruct_Q(C, I):
    # I[r, c] = 质心索引 (0..k-1)
    # Q[r, c] = C[I[r, c]]  # 查找操作
    return Q

# 用 Adam 梯度下降优化 C
C_param = nn.Parameter(C.clone())
optimizer = Adam([C_param], lr=1e-3)

for step in range(N_steps):  # 通常 ~1000 steps
    Q_hat = reconstruct_Q(C_param, I)  # 用当前 C 重建 Q
    loss = ||W @ X - Q_hat @ X||²_F  # 层输出 MSE
    loss.backward()  # ∂loss/∂C 通过查找操作传播
    optimizer.step()
    optimizer.zero_grad()

C_optimized = C_param.detach()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 使用 PyTorch 的 autograd 实现梯度下降（比闭式解伪逆更快）。每个 group 的 codebook 独立优化，group 间无交互。梯度 ∂Q/∂C 的实现：因 Q[r,c] = C[I[r,c]]（索引查找），∂loss/∂C[m] = Σ_{(r,c): I[r,c]=m} ∂loss/∂Q[r,c]。实现要点：(1) N_steps 通常为数百到一千（消融显示 1000 steps 足够收敛）；(2) 可选对 codebook 值施加 L2 正则化防止过拟合；(3) Codebook update 在 quantization 后、模型导出前执行。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---

## Blockwise Data Normalization for Vector Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Blockwise Data Normalization 是 GPTVQ 在 EM codebook 初始化前对权重数据进行的预处理。目的：降低 VQ 的量化误差。方法：对每个 group 对应的权重子矩阵 W_i，按每 sub-row（block，通常 16/32/64 个元素）执行逐元素除法 W_i ⊘ S_i，其中 scale s^{(i)} = max_j |w_j^{(i)}| 是该 sub-row 的最大绝对值。为覆盖多个数量级，scale 在 log 空间量化到 4-bit：s^{(i)}_{int} = ⌈(log₂(s^{(i)}) - z)/a⌋·a。解码时逆操作：w = w_decoded · 2^{-a·s_{int} - s_0}。消融（Table 16）显示：scaling block size 越小（8→128），perplexity 越低但 overhead 越大；2D 3-bit VQ 时 BS=16 的 PPL=5.66 vs 无 scaling 的 PPL=5.91。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Blockwise Data Normalization (codebook 初始化前的预处理)
# 输入: 权重子矩阵 W_i ∈ R^{r × m}, block size BS=32
# 输出: 归一化后的 W_i_normalized, scales S, log-offset z

for each sub-row w in W_i:  # w ∈ R^{BS}
    # Step 1: 计算 scale (max abs)
    s = max(|w_j| for j in range(BS))
    
    # Step 2: Log-space 量化到 4-bit
    s_int = ceil((log2(s) - z) / a) * a  # a = shared quantization step
    # z = floating point offset, shared per-column
    
    # Step 3: 归一化
    w_normalized = w * 2^{-a * s_int - s_0}  # s_0 = log2(z)
    
    store s_int  # 4-bit per block

# 解码时逆归一化:
# w_decoded_fp = w_vq_decoded * 2^{a * s_int + s_0}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 在 codebook 初始化阶段应用 normalization，在 VQ 解码 kernel 中应用 inverse normalization（高效乘 2 的幂次）。Scale overhead：BS=32 时每个 weight 增加 4/32=0.125 bits overhead。Log-space 量化的优势：(1) 覆盖跨越多个数量级的权重值（LLM 权重常见范围 10^{-3} 到 10^{1}）；(2) 乘法逆归一化在硬件上高效（乘 power-of-two ≈ bit-shift）。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---

## Mahalanobis Initialization for Codebook EM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mahalanobis Initialization 是 GPTVQ 提出的 EM 算法 seeding 方法，替代传统的 k-Means++。方法：对 N 个 d 维数据点 X，先按马氏距离（Mahalanobis distance）到数据中心排序，然后从排序列表中均匀间隔采样 k 个点作为初始质心，采样间隔为 ⌊N/(k-1)⌋。马氏距离 = (x - μ)^T Σ^{-1} (x - μ)，在 GPTVQ 中使用 Hessian 加权的距离（等价于加权欧氏距离）。该方法比 k-Means++ 快 3-5×，且最终 perplexity 相当（Table 13：Mahalanobis PPL 6.05 vs k-Means++ PPL 6.16）。直觉：马氏距离排序确保采样点在不同"距离层级"上均匀分布，覆盖从数据中心到远尾的整个分布范围。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Mahalanobis Initialization 伪代码
# 输入: 数据点 X ∈ R^{N × d}, 均值 μ ∈ R^d, 逆 Hessian 对角 D (≡ Σ^{-1})
# 输出: k 个初始质心

# Step 1: 计算每个数据点的马氏距离
distances = []
for i in range(N):
    diff = X[i] - μ  # [d]
    # 加权马氏距离: (x-μ)^T diag(1/H^{-1}_{jj}) (x-μ)
    mahal_dist = Σ_{j=0}^{d-1} diff[j]^2 / H_inv_diag[j]
    distances.append((mahal_dist, i))

# Step 2: 按距离排序
distances.sort()  # 升序

# Step 3: 均匀间隔采样
step = N / (k - 1)
centroids = []
for m in range(k):
    idx = int(m * step)
    centroids.append(X[distances[idx].index])

return centroids  # 形状: k × d
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 默认使用 Mahalanobis initialization 因其速度优势（Table 13: 2D 3B 16384→756s vs K++ 3168s，4.2× 加速）且精度不劣于 k-Means++。实现要点：(1) 马氏距离计算可与 Hessian 计算共享 D 矩阵；(2) 排序使用 PyTorch 的 argsort；(3) 对 70B 大模型，Mahalanobis 的高效性尤为关键（k-Means++ 可能耗时数十小时）。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---

## Non-Uniform Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Non-Uniform Quantization（非均匀量化）是将浮点值映射到任意分布的离散质心（而非等间隔固定点）的量化方法。与均匀量化（uniform quantization）的等间隔 grid 不同，非均匀量化存储一个 codebook C = {c_1, c_2, ..., c_k}，其中质心值可以位于任意浮点位置。每个原始值 x 存储的是映射质心的索引 j（⌈log₂ k⌉ bits），而非量化值本身。解码时通过索引查表恢复近似值。Vector Quantization 是非均匀量化在高维的推广——当 d=1 时即为标量非均匀量化（1D VQ），当 d>1 时将 d 个值作为一个向量整体映射到 d 维质心。

GPTVQ 证实非均匀量化在 low-bitwidth（2-3 bits）下比均匀量化有显著优势：Llama-v2-70B W2@g128，GPTVQ 1D（标量非均匀）PPL=5.03，远低于 OmniQuant（均匀）的 6.55。2D VQ 进一步降至 4.72。这是因为非均匀质心可以密集分布于高概率区域、稀疏分布于尾部，比均匀 grid 更好地匹配 LLM 权重的实际分布（近似高斯/拉普拉斯，而非均匀）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
非均匀量化的编码/解码公式：
```
编码（量化）:
j* = argmin_j |x - c_j|  # 找最近质心（1D）或加权最近质心（VQ）
store(j*)                # 存储 ⌈log₂(k)⌉ bits

解码（反量化）:
x̂ = c_{j*}              # 查表恢复
```

与均匀量化的对比：
```
# 均匀量化 (INT4, 16 个等间隔点)
Δ = max(|w|) / 7
x̂_uniform = Δ · clamp(round(x/Δ), -8, 7)  # 16 个候选值

# 非均匀量化 (1D VQ, k=8 个质心, 3 bits)
C = {c_0, c_1, ..., c_7}  # 可位于任意位置
x̂_nonuniform = C[argmin_j |x - c_j|]      # 8 个候选值但位置灵活
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
非均匀量化的代表性实现：GPTVQ（1D/2D/4D VQ via EM + GPTQ 框架）；AQLM（8D VQ via beam search + block FT）；NF4/NF3（NormalFloat，基于正态分布分位数的固定 codebook）；AFPQ（非对称 FP 量化，正负值各有独立 scale 的 FP 格式）。非均匀量化在 low-bit（<4 bits）下优势最大；在 4-bit 及以上，均匀量化的等间隔 grid 已足够密集，非均匀的边际增益减小。局限：需要 codebook 存储（额外 bits），需要查表解码（增加延迟），不总是适合硬件加速（GPU Tensor Core 偏好均匀 INT 格式）。

GuidedQuant 提出 LNQ (Layer-wise Non-uniform Quantization)，一种交替优化算法替代 GPTVQ 1D：codebook 用闭式最小二乘解（c = (PᵀHP)⁻¹PᵀHw_j）替代梯度下降，assignment 用 cyclic coordinate descent 替代 GPTQ，保证目标函数单调递减并收敛。LNQ 在所有 settings 下优于 GPTVQ 1D，可与 GuidedQuant 的 end loss guided 目标结合进一步提升性能。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- SqueezeLLM Dense-and-Sparse Quantization

## SLiM-Quant（概率化均匀量化优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLiM-Quant 是 SLiM 论文（Mozaffari et al., ICML 2025）提出的概率化均匀量化优化方法。核心思路是将对称均匀量化中 non-convex 的 MSE 最小化问题（argmin_α ||round(clip(W/α))·2^{q-1} - W||²）通过概率化重表述转化为可求解的积分形式。将权重元素视为从概率密度函数 f(·) 中采样，则量化误差期望为 E_Q(α) = ∫ f(x)|Q^{-1}(Q(x)) - x|² dx，拆分为量化误差 E_quant(α) = ∫_0^α f_abs(x)|α·round(x/α)·2^{1-q} - x|² dx 和裁剪误差 E_clip(α) = ∫_α^∞ f_abs(x)|α - x|² dx。由于实际权重分布不符合任何标准 PDF，SLiM-Quant 在权重直方图上做数值积分，采用多网格策略（低分辨率 10 样本均匀扫描 + 在最低误差区域高分辨率细化搜索）高效找到全局最优 scaling factor α*。相比 Grid Search 或 AbsMax（对 outlier 敏感），SLiM-Quant 在保持 uniform quantization 的硬件友好性（单 scale per tensor）的同时，达到 group quantization 级别的精度。

SLiM-Quant 的激活感知变体 SLiM-Quant^O 进一步定义联合显著性 saliency = |diag(x_mean) × W|（x_mean 为校准集激活平均绝对值），对 top 1% 最高显著性通道做 scale up 权重 × s + scale down 对应激活 ÷ s 的等效变换，降低输出误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SLiM-Quant 算法（Algorithm 1 from paper）
# 输入: 权重绝对值PDF f_abs, 步长 η_low/η_high, 权重 W, 位宽 q

def EstimateError(α):
    # 量化误差: 绝对值小于α的元素
    E_quant = ∫_0^α f_abs(x) |α × round(x/α) × 2^{1-q} - x|² dx
    # 裁剪误差: 绝对值大于α的元素
    E_clip = ∫_α^∞ f_abs(x) |α - x|² dx
    return E_quant + E_clip

# Phase 1: 低分辨率扫描 (10 samples in [0, max(W)])
E = {}
for α in linspace(0, max(|W|), 10):
    E[α] = EstimateError(α)
α_low = argmin(E)

# Phase 2: 高分辨率细化
for α in linspace(α_low - η_low, α_low + η_low, resolution=η_high):
    E[α] = EstimateError(α)
α* = argmin(E)

# 量化输出
W_quant = round(clip(W/α*)) × 2^{q-1}
```

关键参数：直方图 bin 数 = max(512, min(d_in×d_out/1000, 20000))，保证概率密度近似精度。多网格策略使计算开销极小（与 Wanda 相当的压缩时间，见表 21）。

SLiM-Quant^O 额外步骤：
```
x_mean = mean(calibration_activations, dim=batch)
saliency = |diag(x_mean) × W|  # per-channel 显著性
top1pct = top_k(saliency, k=0.01*d_in)
W[top1pct] *= s      # scale up weights
x[top1pct] /= s      # scale down activations (等效)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SLiM-Quant 已集成到 SLiM 开源代码库（https://github.com/Mohammad-Mozaffari/slim / https://github.com/Paramathic/slim）。使用方式：(1) 对每个 Linear 层权重独立计算最优 α*（逐层执行，仅需一层权重在 GPU 内存中）；(2) 直方图构建和数值积分均为纯 CPU/GPU 操作；(3) 与 Wanda、SparseGPT 等剪枝方法及 SLiM-LoRA 低秩适配无缝衔接。SLiM-Quant^W（仅权重误差最小化）推理无额外开销；SLiM-Quant^O 约 1% 激活通道的 on-the-fly scaling 引入轻微不规则内存访问，是精度 vs 开销的权衡。论文实测 SLiM-Quant^W 与 SLiM-Quant^O 准确率差距微小（约 0.1%）。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---

## SLiM-LoRA（显著性可逆可加低秩适配器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLiM-LoRA 是 SLiM 论文提出的 one-shot 低秩适配方法，利用具有可逆性和可加性的显著性函数（F(W) = diag(x)W，x 为校准集激活平均绝对值），通过 SVD 数学推导出低秩适配器的闭式解，无需任何迭代训练。核心创新：(1) 显著性函数 F 的可加性允许将压缩误差的显著性从原权重中隔离——F(-(E_Q+E_S)) = F(W^C - W)，再对显著性误差矩阵做 SVD；(2) 逆显著性变换 L = diag(1/x)·L̃, R = R̃ 将显著性空间的适配器转换回权重空间；(3) 显著性加权确保适配器优先修正对模型输出影响最大的权重通道（对比 Naive-LoRA 均匀最小化 Frobenius 范数）。

数学推导：目标 max_{L,R} ||F(W^C + LR)||² = min_{L,R} ||F(-(E_Q+E_S)) - F(LR)||²。由于 F 可加：F(LR) = diag(x)·L·R。SVD 分解：diag(x)·(-E_Q-E_S) = UΣV^T，取 rank r（论文默认 r=0.1d），得到 diag(x)·L = U_r·Σ_r^{1/2}, R = Σ_r^{1/2}·V_r^T，最终 L = diag(1/x)·U_r·Σ_r^{1/2}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SLiM-LoRA 算法（Algorithm 2 from paper）
# 输入: 原权重W, 压缩权重W^C, 校准输入X
# 输出: 低秩适配器 L, R

# Step 1: 计算总压缩误差
E_C = E_Q + E_S = W^C - W  # d_in × d_out

# Step 2: 构建显著性向量
x_tilde = mean(X, dim=batch)  # [d_in]
x = x_tilde + min(|x_tilde|)  # 避免零元素，保证F可逆

# Step 3: 计算误差显著性
S_C = diag(x) @ E_C  # [d_in, d_out]

# Step 4: SVD 低秩近似
U, Σ, V^T = SVD(S_C)
k = r × d  # r=0.1, rank = 10% of hidden dim
L_tilde = U[:, :k] @ sqrt(Σ[:k, :k])   # [d_in, k]
R = sqrt(Σ[:k, :k]) @ V^T[:k, :]       # [k, d_out]

# Step 5: 逆显著性变换
L = diag(1/x) @ L_tilde   # [d_in, k]

# 最终: W ≈ W^C + L @ R
```

对比 Naive-LoRA：Naive-LoRA 直接对 E_C 做 SVD——L,R = SVD(E_C) 截断——忽略不同权重通道对输出的差异化影响。SLiM-LoRA 的显著性加权使 top 通道（激活幅度大的通道）的误差被优先修正，提升 1-3% 准确率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现集成于 SLiM 开源库。低秩适配器 rank 默认为 hidden_dim × 0.1。可选对适配器做 AbsMax group quantization（group size 128, 4-bit）减少 4× 存储/计算开销。推理时：主权重矩阵使用 Sparse Marlin kernel（2:4 稀疏 × 4-bit 量化），低秩适配器使用 Dense Quantized Marlin 或标准 PyTorch GEMM。内存开销分析（rank r=0.1d）：适配器存储 O(2rd²)，原始权重 O(d²)，当 r≪1 时开销可忽略。FLOP 分析类似。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---

## Wanda Pruning（Wanda 剪枝 / 权重-激活联合幅度剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wanda（Pruning by Weights and Activations, Sun et al., 2023）是一种简单高效的 LLM 后训练 one-shot 剪枝方法。核心思想：weight importance score = |W_ij| × ||X_j||_2，即每个权重的幅度乘以其对应输入特征（列）的 L2 激活范数，按行（output neuron）独立比较并剪除 score 最低的权重。不需要权重更新（zero update）、不需要二阶 Hessian 信息、不需要反向传播。仅需一次前向传播收集激活统计 + 逐行 score 排序，计算复杂度远低于 SparseGPT。

数学表达：S_ij = |W_ij| · ||X_j||_2，对第 i 行保留 S_i 最高的 k%（或满足 N:M 模式）。等价于 SparseGPT 中 Hessian 逆的对角近似（忽略 Hessian 非对角元）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Wanda 剪枝伪代码（per layer）
# 输入: W [d_out, d_in], X [d_in, N_samples], sparsity_ratio s

# Step 1: 收集激活统计
X_norm = ||X||_2  # L2 norm per input channel [d_in]

# Step 2: 计算 importance score
for i in range(d_out):  # per output neuron
    for j in range(d_in):
        S[i][j] = |W[i][j]| * X_norm[j]

# Step 3: 逐行剪枝 (per-output comparison)
for i in range(d_out):
    k = (1-s) * d_in  # 保留的权重数量
    threshold = top_k(S[i], k)  # 第k大的score
    mask[i] = (S[i] >= threshold)
    W_pruned[i] = W[i] * mask[i]  # 零值更新，不调整剩余权重

# 2:4 结构化稀疏变体:
# 将 d_in 分成 d_in/4 个连续组，每组保留score最高的2个
for i in range(d_out):
    for g in range(0, d_in, 4):
        top2_idx = argsort(S[i][g:g+4])[-2:]  # 每组保留2个
        mask[i][g:g+4] = 0
        mask[i][g + top2_idx] = 1
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/locuslab/wanda。支持 OPT、LLaMA、LLaMA-2 模型系列。使用 HuggingFace Transformers 加载模型，128 条 C4 校准样本，逐层前向传播收集激活统计后立即剪枝该层并释放激活，GPU 显存需求 = 单层权重 + 单层激活。在 SLiM 中是默认剪枝方法，作用于量化后的权重 W^Q 上，进一步施加误差 E_S。支持 unstructured (50%) 和 2:4 semi-structured (NVIDIA Sparse Tensor Core 兼容) 两种模式。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---

## One-shot Joint Quantization-Sparsity-LowRank Compression（一-shot 量化-稀疏-低秩联合压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-shot Joint Quantization-Sparsity-LowRank Compression 是 SLiM 框架提出的将量化、稀疏化和低秩近似三者整合为统一 one-shot pipeline 的压缩范式。与各自独立执行（如先剪枝再量化，误差独立累积）不同，SLiM 的三阶段 pipeline 将误差视为统一的可补偿信号：(1) SLiM-Quant 最小化初始量化误差 E_Q；(2) Wanda 在量化权重上施加稀疏，引入 E_S；(3) SLiM-LoRA 对总误差 E_Q+E_S 做显著性加权 SVD，通过低秩适配器闭式解补偿。关键洞察：总压缩误差的显著性（而非 Frobenius 范数）决定对模型输出的影响，因此低秩适配器应优先修正高显著性通道的误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SLiM 整体 pipeline（per layer）
# 阶段一: 量化
α* = SLiM-Quant(W, q=4)         # 概率化最优 scaling factor
W_Q = round(clip(W/α*)) × 2^3   # 4-bit symmetric uniform quant

# 阶段二: 稀疏化
X_norm = ||calibration_X||_2
W_C = Wanda_prune(W_Q, X_norm, sparsity=0.5, pattern="2:4")
E_C = W_C - W   # 总误差 = E_Q + E_S

# 阶段三: 低秩补偿
x = mean(calibration_X) + shift  # 显著性向量
S_C = diag(x) @ E_C              # 误差显著性 [d_in, d_out]
L_tilde, R = SVD(S_C, rank=0.1d) # 低秩近似
L = diag(1/x) @ L_tilde          # 逆显著性变换

# 可选阶段四: 适配器量化 + PEFT 微调
L_Q = AbsMax_group_quantize(L, group_size=128, bits=4)
R_Q = AbsMax_group_quantize(R, group_size=128, bits=4)
# 冻结 W_C, 仅微调 L_Q, R_Q (STE, AdaFactor, 300K tokens C4)

# 推理: Y = SparseMarlin(X, W_C) + X @ L_Q @ R_Q (或 X @ L @ R)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
全流程实现于 SLiM 开源库。压缩配置：4-bit 权重量化 + 50% 2:4 稀疏 + rank=0.1d 低秩适配器。内存缩减（含量化适配器）约 5×（0.19-0.20× dense），FLOP 缩减约 1.5×（低秩适配器引入少量额外计算）。校准数据：128 条 C4 序列。整个压缩过程对 LLaMA-2-7B 约需 39 分钟（单 H100 GPU）。压缩后可选 PEFT 微调（300K tokens, ~14h 单 GPU）进一步缩小与 dense 模型的精度差距。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---

## Block-Diagonal Fisher Approximation for PTQ（块对角Fisher近似的后训练量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-Diagonal Fisher Approximation 是 GuidedQuant 提出的 Hessian 矩阵近似策略，用于后训练量化中更准确地估计量化对 end loss 的影响。传统 layer-wise output-based PTQ（如 GPTQ）将所有 output features 视为同等重要，使用 `||XW - XŴ||²` 作为量化目标。SqueezeLLM 使用 diagonal Fisher 近似 `(ŵ-w)ᵀdiag(F)(ŵ-w)`，但忽略所有 off-diagonal 的 cross-weight interactions。GuidedQuant 提出 block-diagonal 近似：保留 Fisher Information Matrix 中每个 output channel 对应的 d_in×d_in 对角块 `F_j = (1/n) Σ(∂ℓ_i/∂w_j)(∂ℓ_i/∂w_j)ᵀ`，忽略跨 channel 和跨层的 off-diagonal 项。这等价于二阶 Taylor 展开中假设 Hessian block-diagonal，捕获了同一 output channel 内权重之间的相互作用，同时保持计算可处理性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Block-diagonal Fisher 在 GuidedQuant 中的计算流程：
```
# 输入：校准数据 X (n × d_in)，权重 W (d_in × d_out)
# 输出：逐层量化权重 Ŵ

# Step 1: 单次 backward pass 计算 end loss 对各层输出的梯度
for each layer l:
    Z_l = X_l @ W_l                          # 层输出 (n × d_out)
    G_l = ∂ℓ/∂Z_l                             # end loss 梯度 (n × d_out)

# Step 2: 计算 block-diagonal Fisher（每 output channel j 一个 d_in×d_in 块）
for each layer l:
    for each output channel j in [1, d_out]:
        # F_j 捕获第 j 个 output channel 内的 cross-weight 交互
        F_j = (1/n) * X_lᵀ @ Diag(G_l[:,j]²) @ X_l    # d_in × d_in
        
# Step 3: 量化目标（等价于 block-diagonal 二次近似）
# min Σ_j (w_j - ŵ_j)ᵀ F_j (w_j - ŵ_j)
```
注意：直接计算和存储所有 d_out 个 d_in×d_in 的 F_j 矩阵对于 LLM 不可行（Llama-2-7B 需 >110TB）。GuidedQuant 通过 averaging approximation（见单独术语）将 d_out 个矩阵按 g 组平均为 g 个共享矩阵来解决。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block-diagonal Fisher 近似在 LLM 压缩中首次被 GuidedQuant (ICML 2025) 扩展到现代 LLM 规模。此前 WoodFisher (NeurIPS 2020) 使用任意大小 B×B 对角块用于 CNN 剪枝，Optimal BERT Surgeon (EMNLP 2022) 用于 BERT 剪枝（B=50），BRECQ (ICLR 2021) 使用 residual block 对应块用于 CNN 量化。GuidedQuant 的关键创新：(1) 块大小 = d_in×d_in（对应 output channel），保留 channel 内全部权重交互；(2) averaging approximation 使存储从 Θ(d_in² d_out) 降至 Θ(d_in² g)，g 为分组数（通常 2-4）；(3) 与任意 layer-wise output-based PTQ 方法（LNQ、QTIP、SpinQuant）可直接集成。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---

## End Loss Guided PTQ Objective（端到端损失引导的后训练量化目标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
End Loss Guided PTQ Objective 是 GuidedQuant 提出的量化目标函数：`||(∂ℓ/∂Z) ⊙ (XW - XŴ)||²_F`，即用 end loss 对每层输出的梯度 ∂ℓ/∂Z 作为权重，对 layer-wise output error 进行逐元素加权。与标准 layer-wise output-based 目标 `||XW - XŴ||²` 将每个 output feature 的量化误差同等对待不同，end loss guided 目标根据每个 feature 对最终 loss 的敏感度分配不同的重要性权重。该目标基于 end loss 的一阶 Taylor 展开：`ℓ(Ẑ) - ℓ(Z) ≈ ∂ℓ/∂Z · (Ẑ - Z)`。等价于 block-diagonal Fisher 二次近似（见 Block-Diagonal Fisher Approximation 术语），但通过 backprop 梯度直接计算避免了显式构建 Fisher 矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GuidedQuant 的 end loss guided 量化流程（Algorithm 1）：
```
# 输入：校准数据，预训练模型，分组数 g
# 超参数：g=4 (7B/13B), g=2 (70B) for weight-only; g=1 for W+A

# Step 1: 划分 output channels 为 g 组
J_k = {d_out*(k-1)/g + 1, ..., d_out*k/g}  for k in [1, g]

# Step 2: 单次 backward pass，计算并平均梯度的平方
for each layer l, group k:
    s_k = (1/|J_k|) * Σ_{j∈J_k} (∂ℓ/∂z_j)²   # n 维向量

# Step 3: 构建 guided Hessian
    H̄_k = Xᵀ @ Diag(s_k) @ X                    # d_in × d_in

# Step 4: 调用 base quantizer Q 量化该组
    Ŵ[:, J_k] = Q(H̄_k, W[:, J_k])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GuidedQuant 作为 plugin 可直接集成到任何 layer-wise output-based PTQ 方法中：(1) LNQ + GQuant：将 H=XᵀX 替换为 H̄_k，调用 LNQ（Algorithm 2）量化每组；(2) QTIP + GQuant：将 BlockLDLQ 的 Hessian 替换为 H̄_k；(3) SpinQuant + GQuant：将 GPTQ weight quantizer 的 Hessian 替换为 H̄_k。梯度计算只需一次 backward pass（Llama-2-7B: 0.3h on 1×A100），Hessian 可缓存复用多次量化。End loss guided 目标在极端低比特（2-bit）下收益最大：Llama-2-7B non-uniform scalar 2.01-bit Wiki2 perplexity 从 23.31 (LNQ) 降至 8.83 (LNQ+GQuant)。代码开源：github.com/snu-mllab/GuidedQuant。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---

## Layer-wise Non-uniform Quantization (LNQ / 逐层非均匀量化算法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LNQ (Layer-wise Non-uniform Quantization) 是 GuidedQuant 论文提出的一种新的 weight-only non-uniform scalar PTQ 算法。它为每个 output channel 维护独立的 codebook，通过 alternating minimization 交替优化 codebook 向量 c 和 assignment 矩阵 P：(1) Codebook 更新：给定 assignment P，将问题退化为标准最小二乘，使用闭式解 `c = (PᵀHP)⁻¹PᵀHw`，其中 H=XᵀX 是 layer-wise Hessian；(2) Assignment 更新：给定 codebook c，使用 cyclic coordinate descent (CD) 优化，每次更新一个输入维度 i 的权重值 `Ŵ_i = Round(W_i - H_{i,others}/H_{ii} · (Ŵ_others - W_others))`。LNQ 保证目标函数单调递减并收敛（Proposition 4.1）。与 GPTVQ 1D（使用梯度下降优化 codebook + GPTQ 优化 assignment）相比，LNQ 的闭式 codebook 解和 CD assignment 优化都是更优的选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LNQ 算法伪代码（Algorithm 2）：
```
# 输入：H ∈ R^{d_in×d_in}, W ∈ R^{d_in×d_out}, 初始 assignment P
# 超参数：T（交替迭代数）, K（CD 循环轮数）
# 典型值：T=2, K=4 (7B/13B); T=1, K=4 (70B)

H = LLᵀ                                      # Cholesky 分解

for j = 1 to d_out:                          # 每个 output channel 独立并行
    for t = 1 to T:                           # 交替优化
        # Codebook 闭式解（least squares）
        c = (PᵀHP)⁻¹PᵀHw_j                   # O(d_in² m)
        ŵ_j = P @ c                           # 量化权重
        
        # Assignment 优化（cyclic CD）
        for k = 1 to K:                        # K 轮 CD 循环
            for i = 1 to d_in:                 # 遍历每个输入维度
                # 坐标下降闭式解（Eq. 11）
                Ŵ_{i,j} = Round(W_{i,j} - H_{i,[d_in]\i}/H_{i,i} · (Ŵ_{[d_in]\i, j} - W_{[d_in]\i, j}))
        
        # 更新 P：P_{i,q}=1 当 Ŵ_{i,j}=c_q，否则 0
```

GPU 加速技巧：(1) Precomputation trick：预计算 future coordinates 的 Hessian 贡献 B=StrictUpper(H̃)(Ŵ-W)，减少每次 CD 更新的 FLOPs；(2) Lazy batch-updates：将坐标按 batch_size b 分块处理，每块内依次更新、仅 block 内全局修正，减少 memory-bound 全局操作。两技巧组合实现 4× 加速（Llama-2-7B 4-bit: 3.9h → 0.9h on 1×RTX 6000 Ada）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LNQ 适用于 weight-only non-uniform scalar 量化（2-4 bits）。与 SqueezeLLM（k-means + diagonal Fisher）相比，LNQ 使用 layer-wise output error 目标且用闭式解+CD 替代 Lloyd's 算法。与 GPTVQ 1D 相比，LNQ 用闭式 codebook 解替代梯度下降、用 CD 替代 GPTQ，在所有 settings 下均优于 GPTVQ 1D。可进一步集成 GuidedQuant 的 end loss guided 目标（LNQ + GQuant），将原始 Hessian H=XᵀX 替换为 guided Hessian H̄_k。代码随 GuidedQuant 开源（github.com/snu-mllab/GuidedQuant）。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---

## Cyclic Coordinate Descent for Quantization Assignment（循环坐标下降量化分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cyclic Coordinate Descent (CD) 是 LNQ 中用于优化 quantization assignment 的迭代算法。给定固定 codebook c 和 Hessian H，CD 在每一轮中按固定顺序（i=1→d_in）依次更新每个权重值：对第 i 个输入维度，选择使目标函数 `(ŵ-w)ᵀH(ŵ-w)` 最小化的 codebook 值。核心闭式解（Behdin et al., 2023 Lemma 1）：`Ŵ_i = Round(W_i - H_{i,others}/H_{ii} · (Ŵ_others - W_others))`，其中 Round 将值映射到最近的 codebook entry。CD 是 descent method：在初始化 feasible solution 的前提下保证目标函数单调递减。LNQ 中 CD 以当前 assignment+codebook 对应的量化权重为初值，保证每次 CD 调用不增加目标值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
带预计算优化的 CD 算法（Algorithm 4）：
```
# 输入：H, W, c, Ŵ（当前量化权重）, Q（rounded indices）
H̃ = diag(H)⁻¹H                              # 归一化 Hessian
U = StrictUpper(H̃)                            # 严格上三角

for k = 1 to K:                               # K 轮 CD 循环
    B = U @ (Ŵ - W)                           # 预计算 future 坐标贡献
    
    for s in 1, b+1, 2b+1, ..., d_in-b+1:    # lazy batch (b=128)
        for i = s to s+b-1:                   # batch 内 sequential
            Ŵ[i,:] = Round(W[i,:] - B[i,:])   # 坐标下降更新
            Q[i,:] = RoundIdx(W[i,:] - B[i,:])
            B[i+1:s+b, :] += U[i+1:s+b, i] @ (Ŵ[i,:] - W[i,:])  # 局部修正
        
        # batch 完成后全局修正
        B[s+b:, :] += U[s+b:, s:s+b] @ (Ŵ[s:s+b, :] - W[s:s+b, :])

# 从 Q 提取 P：P_{i,q}^{(j)} = 1 if q=Q_{i,j} else 0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CD 在 LLM 量化中的使用：(1) QuantEase (Behdin et al., 2023) 首次将 cyclic CD 用于 uniform 量化；(2) CDQuant (Nair & Suggala, 2024) 证明 greedy CD 优于 GPTQ，cyclic CD 性能接近但计算成本更低；(3) QuIP (Chee et al., 2024) 使用 CD 作为 post-GPTQ/LDLQ 精炼步骤；(4) LNQ (GuidedQuant, 2025) 将 CD 嵌入 alternating minimization 框架用于 non-uniform 量化，在 GuidedQuant 的 ablation study（Table 14）中验证 CD 在所有 settings 下匹配或优于 GPTQ。GPU 上的优化实现使用 precomputation + lazy batch-updates 可达 4× 加速。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---

## Averaging Approximation for PTQ Hessian（PTQ Hessian 的平均近似）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Averaging Approximation 是 GuidedQuant 为解决 block-diagonal Fisher 计算不可行问题而提出的近似技术。直接为每层存储 d_out 个 d_in×d_in 的 Fisher block 矩阵 `H_j = XᵀDiag((∂ℓ/∂z_j)²)X` 需要 Θ(d_in² d_out) 存储——Llama-2-7B 的 self-attention projection 层（d_in=4096~11008, d_out=4096~11008）需远超可用 GPU 内存。Averaging approximation 将 output channels 划分为 g 个组（g≪d_out），每组内平均所有 block 的 Fisher 信息：`H̄_k = XᵀDiag( (1/|J_k|) Σ_{j∈J_k} (∂ℓ/∂z_j)² ) X`。存储从 Θ(d_in² d_out) 降至 Θ(d_in² g)，通过连续的 output channels 成组实现。实验表明（Table 13）g=1 已捕获大部分性能增益（2-bit: Wiki2 9.00 vs g=4 的 8.83），g=2 几乎达到饱和。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 直接法（不可行）：d_out 个 d_in×d_in 矩阵
for j in 1..d_out:
    s_j = (∂ℓ/∂z_j)²                           # n 维向量
    H_j = Xᵀ @ Diag(s_j) @ X                   # d_in × d_in
# 存储: d_out × d_in²，Llama-2-7B 一层需 >200GB

# Averaging approximation（可行）：g 个 d_in×d_in 矩阵
for k in 1..g:
    J_k = {d_out*(k-1)/g + 1, ..., d_out*k/g}  # 连续 channel 分组
    s̄_k = (1/|J_k|) * Σ_{j∈J_k} (∂ℓ/∂z_j)²    # 平均梯度平方
    H̄_k = Xᵀ @ Diag(s̄_k) @ X                   # 共享 Hessian
# 存储: g × d_in²，g=4 时一层 ~0.4GB
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GuidedQuant 使用连续分组策略（默认每 d_out/g 个连续 channels 一组），实验显示此简单策略有效，可能 benefit from more sophisticated clustering。超参数 g 的选择权衡：g 越大 → Fisher 近似越精确（类似 block-diagonal → 接近 full Fisher within groups）→ 性能越好（边际递减）；g 越大 → 存储/计算开销越大（线性增长）。推荐值：g=4 for 7B/13B, g=2 for 70B。g=1（全局平均，退化为 SqueezeLLM 风格的对角近似去掉所有跨权重依赖）在极端压缩下仍有显著提升。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

## Hybrid-Head Architecture (混合头架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid-Head Architecture 是 Hymba (NVIDIA, 2024) 提出的一种 LLM 架构设计，在**同一 Transformer 层内并行**放置 Attention heads 和 SSM (Mamba) heads，两者同时处理相同输入，输出经 learnable per-channel 重缩放后融合。与传统 sequential hybrid（如 Samba/Jamba 交替堆叠 Attention 层和 Mamba 层）不同，hybrid-head 的并行设计使两种算子互补：SSM heads 提供全局上下文摘要（类比 fading memory），Attention heads 提供高分辨率局部召回（类比 snapshot memory）。统一对称公式为：

$$Y = W_{\text{out\_proj}} \left( \beta_1 \cdot \text{norm}(M_{\text{attn}} \tilde{X}) + \beta_2 \cdot \text{norm}(M_{\text{ssm}} \tilde{X}) \right)$$

其中 $M_{\text{attn}} = \text{softmax}(QK^T) W^V$，$M_{\text{ssm}} = G \odot \alpha(A,B,C,\Delta) W^{SSM}$，$\beta_1, \beta_2$ 为可学习 per-channel 重缩放向量。Hymba 发现 SSM heads 输出幅度始终大于 attention heads（Fig. 12），因此引入归一化+重缩放保证训练稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hymba hybrid-head 单层前向：

```
X̃ = concat([R, X], dim=0)          # prepend 128 meta tokens

# 统一输入投影
Q, K, V = W^Q @ X̃, W^K @ X̃, W^V @ X̃
X_ssm = W^{SSM} @ X̃
G = W^G @ X̃

# Attention heads（仅 3 层 global，其余 sliding window）
Y_attn = softmax(Q @ K^T / √d_head + causal_mask) @ V

# SSM heads（Mamba-style recurrent）
h_0 = 0
for i in 1..N+m:
    Δ_i = softplus(W_Δ @ X_ssm[i])
    Ā_i = exp(Δ_i ⊗ A)
    h_i = Ā_i ⊙ h_{i-1} + (Δ_i ⊗ (W_B @ X_ssm[i])) ⊙ X_ssm[i]
    y_i = (W_C @ X_ssm[i]) @ h_i
Y_ssm = G ⊙ Y

# 融合
Y = W_out_proj(β₁ ⊙ norm(Y_attn) + β₂ ⊙ norm(Y_ssm))
```

Hymba-1.5B 配置：32 layers, 25 attn heads (5 GQA groups), attn:mamba 参数比 1:5.23, 3/32 layers 为 global attention, window_size=1024。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hybrid-head 场景：(1) 高效小模型（<2B）——Hymba-1.5B 超越 Llama-3.2-3B，cache 缩小 11.67×；(2) 长上下文——SSM recurrent 支持外推；(3) 端侧部署——小 cache（79MB at 8K）。实现基于 PyTorch + Mamba selective scan + FlashAttention。局限：单层参数量略大于纯 Transformer/Mamba；训练需维护两种机制。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models

## State Space Model (SSM) / Mamba

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SSM 是序列建模框架，通过 $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ 递归编码历史信息。Mamba (Gu & Dao, 2023) 引入输入依赖性选择，$B_t = W_B x_t$, $C_t = W_C x_t$, $\Delta_t = \text{Softplus}(W_\Delta x_t)$，使模型选择性传播或遗忘信息。Mamba-2 (Dao & Gu, 2024) 统一 SSM 与 attention。核心优势：训练用 parallel scan（O(N)），推理用 recurrent（O(1) 内存）。关键局限：常量 size state 导致 recall 弱（300M recall acc 19.23% vs Transformer 39.98%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Mamba 推理（recurrent mode, O(1) 内存）
h = zeros(d_state)
for t in 1..N:
    Δ_t = softplus(W_Δ @ x_t)      # 输入依赖步长
    Ā_t = exp(Δ_t * A)             # 离散化
    B̄_t = Δ_t * (W_B @ x_t)        # 输入投影
    h = Ā_t * h + B̄_t * x_t        # state update
    y_t = (W_C @ x_t) @ h           # output
# 仅需维护 h ∈ R^{d_state}，无需存储 per-token 状态
```

在 Hymba 中，Mamba 作为 SSM heads 提供全局 context 摘要，与 attention heads 并行。SSM 的高效推理特性使得可以激进地用 SWA 替代 global attention（仅 3 层保留），因为 SSM 已总结全局信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/state-spaces/mamba, mamba.py, FLA。适合：(1) 长序列（1M+ tokens, O(1) 推理内存）；(2) hybrid 架构的全局上下文组件（Hymba, Jamba, Samba, Zamba）；(3) 端侧高效推理。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Hymba: A Hybrid-head Architecture for Small Language Models
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## Structured State Space Duality (SSD) / Mamba2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structured State Space Duality (SSD) 是 Dao 和 Gu (2024) 在 Mamba2 中提出的理论框架，揭示了**选择性 SSM 与线性注意力在数学上的对偶关系**——两者都通过半可分矩阵（semiseparable matrix）表达序列变换，仅是同一矩阵的两种分解视角。Mamba2 的核心改进：(1) **Multi-input SSM**：引入 head 结构（类似 Multi-Head Attention），$B_t, C_t$ 在 channel 间共享（类比 grouped-query attention），state group size 可配（默认为 8 组，每组 128 channel）；(2) **简化 block 设计**：$x_t, B_t, C_t, \Delta_t$ 由单次投影并行生成 $(x_t, B_t, C_t, \Delta_t) = F(u_t)$，消除 Mamba1 的 sequential linear，使 tensor parallelism 仅需 1 次 all-reduce（vs Mamba1 的 2 次）；(3) **SSD 算法**：利用分块矩阵分解将 SSM 计算转化为 chunked matmul + short scan，充分利用 GPU Tensor Core（BF16 matmul 约 16× 快于 FP32 逐元素运算），训练速度比 Mamba1 快 2-8×；(4) **更大 state dimension**：$d_{state}$ 从 16 扩展到 64-256；(5) **channel-wise 计算**：SSD scan 是逐 channel 独立计算，因此输出 channel 顺序保持与输入一致（channel order preserving）。Mamba2 的 head 结构和 grouped state 设计直接启发了 Quamba2 的 per-state-group quantization 和 sort-and-cluster 技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Mamba2 SSD block（单 token 推理）
# 输入: u_t ∈ R^D
# 参数: W_in ∈ R^{D×P}, A ∈ R^{d_state}, W_out ∈ R^{P×D}

# Step 1: 并行投影 (x, z, B, C, Δ 一次生成)
proj = W_in @ u_t                                  # R^{P}
x_t = proj[:d_inner]                                # 激活输入
z_t = proj[d_inner:2*d_inner]                       # residual branch
B_t = proj[2*d_inner:2*d_inner+n_heads*d_state]     # 每 head B
C_t = proj[2*d_inner+n_heads*d_state:...]           # 每 head C
Δ_t = softplus(proj[...])                           # 时间步长

# Step 2: Causal conv1d (替代 Mamba1 的独立 conv)
x_conv = causal_conv1d(x_t, W_conv)

# Step 3: SSD scan (channel-wise, 逐 head 独立)
h_t = A_t * h_{t-1} + B_t * x_conv_t               # state update
y_ssd = C_t @ h_t                                    # output via state

# Step 4: Hadamard + output projection
y_out = W_out @ (y_ssd * SiLU(z_t))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/state-spaces/mamba（mamba-2 分支）。Mamba2 发布 checkpoint：2.7B (300B tokens Pile), 8B (1.2T tokens)。适合：(1) 作为量化/压缩研究的目标 backbone（Quamba2, MambaQuant）；(2) 长序列推理（SSD chunked scan 支持可变长度序列）；(3) hybrid 架构的 SSM 组件。注意 SSD 的 channel order preserving 特性是 Quamba2 sort-and-cluster 量化的基础前提。

涉及论文标题：
- Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## Sort-and-Cluster Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sort-and-cluster（SnC）是 Quamba2 提出的针对 SSM 输入激活 $x_t$ 的 8-bit 量化方法，利用 SSM 的两个关键属性：(1) **channel persistence**——各 channel 的激活幅度在不同输入样本间保持一致；(2) **channel order preserving**——SSD 计算是 channel-wise 的，因此输入 channel 顺序等于输出 channel 顺序。SnC 通过 offline calibration 获取各 channel 的最大值，先对 head 内 channel 按最大值排序，再对 head 聚类（m 组），对每组 head 内的 channel 聚类（n 组），最终使用 $m \times n$ 个 scaling factor 量化 $x_t$。排序后的 head 嵌入被"解耦"（disentangle），使得具有相似激活特性的 head 自然聚在一起，从而提升组内量化精度。Quamba2 默认 m=4, n=4，即每层使用 16 个 scaling factor。该方法的额外开销：offline 时需对权重进行 cluster-aware reordering 以匹配新的 channel/head 顺序，online 时仅需按预计算 index 重排激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline calibration（一次执行）
for each block in model:
    for each calibration sample:
        record abs_max = max(|x_t[c]|) for each channel c
    channel_order = argsort(descending, abs_max)     # 按最大值降序排列
    heads_ordered = rearrange(heads, channel_order)
    head_clusters = kmeans(heads_ordered, m)          # m 组 head
    for each head_cluster:
        channel_clusters = kmeans(channels, n)         # n 组 channel
        for each (hc, cc) pair:
            s[hc][cc] = max_abs_value / 127.0          # INT8 scale

# Online inference（每 token）
x_sorted = x_t[channel_order]                         # 按预计算顺序重排
x_sorted = rearrange(x_sorted, head_clusters, channel_clusters)
x_quant = clamp(round(x_sorted / s), -127, 127)       # 8-bit 量化
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖：(1) calibration 数据集（Quamba2 使用 Pile 随机 512 句）；(2) 聚类算法（k-means 或类似无监督方法）；(3) offline weight reordering kernel（按聚类索引重排 W_in 列、W_out 行、W_conv channel、W_norm 参数）。论文发现 m=4, n=4 在所有实验中足够好，更大的 m/n 收益递减。与 MambaQuant 的 clipping 和 Quamba 的 percentile clipping 相比，SnC 在 Mamba2-8B 上提升约 4%（W8A8 设置下 69.8% vs 64.8% FP16 70.7%）。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## Per-State-Group Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-state-group quantization 是 Quamba2 提出的针对 SSM 选择性参数 $B_t$ 和 $C_t$ 的量化方法。在 Mamba2 中，$B_t$ 和 $C_t$ 被组织为多个 state group（默认每组 128 channel，共 8 组），组内共享参数（类比 grouped-query attention）。Quamba2 发现 **state persistence**：各 state group 中激活的 group（数值较大的 group）在时间步和输入样本间保持一致（例如 group 6 在 B 中总是高激活，group 7 在 B 和 C 中总是几乎无变化）。基于此，对每个 state group 使用独立的 scaling factor（而非对整个 B/C 的 per-tensor 量化），大幅提升了小数值 group 的量化精度。在 W4A8 Quamba2-8B 消融中，per-state-group 从 55.1% 提升到 60.7% LAMBADA accuracy（+5.6%），是缩小与 FP16 差距的关键技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline calibration
for each SSM block:
    for each state group g in B:
        s_B[g] = max(|B_t[g]|) / 127.0              # 每 state group 独立 scale
    for each state group g in C:
        s_C[g] = max(|C_t[g]|) / 127.0

# Online inference
B_quant[g] = clamp(round(B_t[g] / s_B[g]), -127, 127)
C_quant[g] = clamp(round(C_t[g] / s_C[g]), -127, 127)
# 注意：cached SSM states 使用相同的 head/channel group indices
# 因此直接复用 x_t 的 sort+cluster index，无需额外 online reorder
h_t = A_t * h_{t-1} + B_quant * x_quant
y_ssd = C_quant * h_t
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该技术特别适合 Mamba2 的 multi-input SSM 架构（有显式 state group），对 Mamba1 也可用类似逻辑（按 head 分组）。开销极小：仅需存储 G 个额外的 FP16 scaling factors（G=state group 数，通常 ≤8）。论文发现 cached SSM states 自然地遵循与 SSM 输入 x 相同的 head/channel 分组（因为 states 是在 channel-wise SSD scan 中从 x 派生的），因此不需要额外为 states 做 online reordering，直接复用 x 的 scale 分组。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## Channel Order Preserving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Channel order preserving 是 Mamba2 SSD 计算的一个结构属性：SSD scan 的线性递归是**逐 channel 独立**计算的，即 $h_t[c] = A_{t}[c] \cdot h_{t-1}[c] + B_{t}[c] \cdot x_t[c]$，因此输出的 channel 顺序与输入的 channel 顺序完全一致。这不同于 Transformer 的 self-attention（跨 channel 混合），是 SSM 独有的计算特性。Quamba2 利用该属性来实现 sort-and-cluster 量化：由于输出 channel 顺序被保留，通过 offline 重排输入投影的**列**、causal conv1d 的**channel**、normalization 权重和输出投影的**行**，可以保持整个 block 的 compute-invariance——量化模型的输出与未重排的 FP16 模型完全等价。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 证明 sketch（以简化 SSD 为例）
# 输入 x 重排：x' = P @ x（P 是 permutation matrix）
# SSD scan 是 channel-wise：h[c] = f(h[c], x[c])
# 因此 y'[c] = f(h[c], x'[c]) = f(h[c], x[perm(c)])
# 输出 y 被同样 permutate：y = P^T @ y'
# 通过同时重排下游的 W_out 行，恢复正确输出：
# W_out_reordered = P @ W_out（对行做逆排列）
# 最终输出不变：W_out_reordered @ y' = P@W_out @ P^T @ y' = W_out @ y（当 y'=P@y）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
仅在 SSD/selective scan 这种 channel-wise 计算中成立。Mamba1 的同类特性也可被利用（因其 selective scan 也是 per-channel 的），但论文未明确讨论 Mamba1 中的实现细节。该属性使 offline weight reordering 成为可能，是 sort-and-cluster 量化可行性的理论前提。没有该属性，sort 后的激活需要通过网上 reorder 恢复，增加延迟。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## Activation Persistence（Channel Persistence / State Persistence）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Activation persistence 是 Quamba2 在 Mamba2 SSM 激活中发现的两种持久化现象：(1) **Channel persistence**：SSD 输入 x 在各 channel 上的激活幅度（最大值）在不同输入样本间保持一致——即若 channel c 在 calibration 样本中是高激活 channel，则在新样本中它仍然是高激活 channel。这使得 offline calibration 得到的 channel order 在 online 推理时依然有效。(2) **State persistence**：$B_t$ 和 $C_t$ 的各 state group 的激活模式（哪些 group 数值大、哪些 group 数值小）在时间步和输入样本间保持一致——例如 group 6 在 B 中持续高激活，group 7 在 B 和 C 中持续低变化。这两种 persistence 是 sort-and-cluster 和 per-state-group quantization 技术的事实基础：若 patterns 不 persistent，offline 校准的 scale 会在 online 推理中失效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文在 Mamba2-8B 的最后一个 block 中验证了这些 properties（图 3）：(a) x 按 calibrated channel max 排序后，对任意输入样本保持大致有序（Spearman 秩相关高）；(b) 排序后的 x 输出 y 也保持顺序；(c-d) B/C 的 state group 激活在时间步上一致；(e-f) B/C 的 state group 激活在不同输入样本间一致。这些 properties 的物理直觉：SSM 的参数 $\Delta_t, B_t, C_t$ 由输入投影生成，而投影权重是固定的，因此相似的输入产生相似的参数分布。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文仅在 Mamba2 中验证了这些 properties，Mamba1 中的 persistence 程度未明确说明。使用方式：(1) offline calibration 收集 channel/state group 统计信息；(2) 基于 persistent patterns 设计量化分组；(3) 由于 patterns 是 persistent 的，offline 确定的 groups 和 scales 在 online 推理中直接使用，无需动态重新校准。注意这是**经验性发现**而非理论保证，论文未在不同架构/任务上验证其普遍性。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## Cluster-Aware Weight Reordering

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cluster-aware weight reordering 是 sort-and-cluster 量化的配套 offline 预处理步骤：由于 sort-and-cluster 改变了激活的 channel/head 顺序（按聚类结果重排），为避免激活和权重的 channel 不匹配，必须对 SSM block 中所有与激活交互的权重矩阵进行对应的 offline 重排。具体重排：(1) input projection weights $W_{in}$ 的**列**按聚类索引排序；(2) causal conv1d weights 的**channel**按聚类索引排序；(3) normalization weights 按聚类索引排序；(4) output projection weights $W_{out}$ 的**行**按聚类索引排序，以恢复正确的输出顺序。因为 SSD 计算保持 channel order（channel order preserving），这些重排保证整个 block 的输出与未重排的 FP16 block 完全等价（compute-invariance）。重排是一次性的 offline 操作，不增加推理延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline reordering（一次执行）
cluster_indices = get_cluster_indices(channel_order, head_clusters, channel_clusters)

# 重排各权重
W_in_reordered = W_in[:, cluster_indices]              # 列重排
W_conv_reordered = W_conv[cluster_indices, :]           # channel 重排
W_norm_reordered = W_norm[cluster_indices]              # normalization 重排
W_out_reordered = W_out[cluster_indices, :]             # 行重排（等价于 P@W_out）
# 其中 P 是 permutation matrix from cluster_indices

# 对于 W4A8/W4A16，权重重排后再进行 GPTQ 优化
W_out_reordered_4bit = GPTQ(W_out_reordered, calib_data)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖 per-group scaling factor 的重新组织（按聚类结果排列），使用 fused dequant+matmul kernel 时需保证 scaling factor layout 与 weight layout 一致以最大化 Tensor Core 加载效率。论文基于 CUTLASS 实现，参考了 Marlin (Frantar et al. 2024) 的 weight 重排策略。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## Head-to-Toe (H2T) Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Head-to-toe (H2T) quantization 指从 embedding 层到 SSM blocks 到 lm_head（输出层）的全模型量化——不留任何 FP16 层。之前的 SSM 量化方法（MambaQuant, Quamba）仅量化 SSM blocks，embedding 和 lm_head 保持 FP16（见表 2）。Quamba2 在 W4A8 设置下，embedding 层使用**per-token quantization**（每 token 独立 scale），lm_head 权重使用**per-group quantization**（分组量化），实现了 4× 全模型显存减小（如 Mamba2-2.7B W4A8：从 FP16 5.2GB 降至 1.4GB）。H2T 的关键价值在于**边缘设备部署**：仅当 embedding 和 lm_head 也被量化，8B 模型才能在 Orin Nano 8G 上运行（FP16 OOM, W8A8 也 OOM, 仅 W4A8/W4A16 可行），实现 13 tokens/s。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 各层量化方式
# Embedding: per-token 量化（输入 token embedding）
e = embedding[token_ids]                              # FP16 lookup
s_e = max(|e|) / 127.0                                # per-token scale
e_quant = clamp(round(e / s_e), -127, 127)            # INT8

# SSM blocks: W4A8/W4A16（sort-and-cluster + per-state-group）
# ... (标准 SSM block 量化流程)

# lm_head: 4-bit per-group weight + FP16 activation
W_lm_head_4bit, s_lm = quantize_per_group(W_lm_head)  # per-group 4-bit
logits = (W_lm_head_4bit * s_lm) @ x_final             # GEMM with dequant
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
小型模型（130M, 370M）的 embedding 量化精度损失较大（约 3-7% LAMBADA），大型模型（2.7B, 8B）几乎无损失（约 0-0.5%），说明大模型对 embedding 量化更鲁棒。论文实现了 4-bit 和 8-bit 的 CUDA embedding/lm_head kernel。H2T 是实现 "部署到边缘设备" 的必要条件——否则 FP16 embedding/lm_head 成为显存瓶颈。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

## W4AX-Mixed Precision with Evolutionary Search

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4AX-mixed 是 Quamba2 提出的混合精度方案：在 SSM 模型的不同 block 中动态选择 W4A8（权重 4-bit+激活 8-bit）或 W4A16（权重 4-bit+激活 16-bit），以在 prefill 延迟和 MMLU 泛化性之间取得最佳平衡。Full W4A8 最大化了 prefill 加速（TTFT 140.78ms vs FP16 197.80ms）但 MMLU 5-shot 下降 5.8%（41.2% vs 47.0% FP16）。Full W4A16 保持较好泛化（45.3% MMLU）但 prefill 延迟增加（209.19ms）。W4AX-mixed 通过进化搜索（evolutionary search, population=40, generations=5）自动识别对量化敏感的 block 分配 W4A16，其余用 W4A8，最终在 MMLU 提升 2.9%（达 44.0%）同时仅增加 10% prefill 延迟（158.36ms）。手工设计的混合精度（前/后 N 层用 W4A16，命名为 HC-first/HC-last）效果远差于自动搜索。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Evolutionary search for mixed precision
population = [random_bitwidth_config() for _ in range(40)]   # 40 个随机配置
for gen in range(5):
    fitness = [eval_accuracy(cfg) for cfg in population]     # 评估准确率
    population = top_k(population, fitness, k=20)            # 保留 top 50%
    new_pop = []
    for _ in range(10):                                      # 10 crossover
        p1, p2 = random_pair(population)
        child = crossover(p1, p2)
        new_pop.append(child)
    for _ in range(10):                                      # 10 mutation
        p = random(population)
        mutant = mutate(p)
        new_pop.append(mutant)
    population = population + new_pop

best_config = population[argmax(fitness)]
# best_config: e.g. [W4A8, W4A8, W4A16, W4A8, ...] (per-block)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
搜索的搜索空间：每层独立选择 W4A8/W4A16，对 56 层模型有 $2^{56}$ 种配置。进化搜索在 5 代内收敛到 Pareto 前沿。适合需要同时优化延迟和准确率的场景（如云服务需要平衡 TTFT 和精度）。论文发现设计的混合精度模型在 Pareto 前沿上优于手工规则（HC-first/HC-last），说明 SSM block 的量化敏感度分布不均匀且无简单规律（并非"首/尾层更敏感"）。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Meta Tokens 是 Hymba 提出的 128 个可学习 embedding，预训练期间 prepend 到所有输入，与模型参数联合优化。推理时固定，K/V/SSM 状态离线预计算，等效 learned cache initialization。三重功能：(1) 缓解 attention sink——吸收 >50% 本该流向 BOS 的 attention；(2) 封装压缩世界知识——不同 domain 激活不同 meta tokens；(3) 作为初始 cache 调制后续 token attention 分布。消融：recall acc +3.75%（48.04%→51.79%），attention map entropy 整体下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline（仅一次）: 预计算 meta tokens 状态
K_meta, V_meta = W^K @ R, W^V @ R
h_meta = SSM_scan(R)   # 128步 scan 后的 state

# Online: 预计算状态 + 用户输入状态拼接
K = [K_meta; K_online], V = [V_meta; V_online]
h_init = h_meta         # SSM 从预计算 state 继续
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 ViT register tokens (Darcet et al. 2023)、prefix tuning (Lester et al. 2021)、StreamingLLM 类似。推理开销极低（仅 cache 多 128 位置）。局限：任务特定 meta tokens 未探索。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models

## Attention Sink (注意力汇)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Sink 是 Xiao et al. (2023) 发现的 Transformer 现象：初始 token（尤其 BOS）在不重要的语义下获得不成比例的高 attention（>50% for Llama3.2-3B）。根源：softmax 求和为 1 的约束迫使模型将过剩概率质量倾注到初始 token 作为"注意力垃圾桶"，导致 cross-token attention 预算不足，损害 recall。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hymba 三管齐下：(1) Meta tokens 作为"更好的垃圾桶"吸收 sink；(2) SWA 限制 attention 范围，BOS 在大多数层不可见；(3) SSM heads 绕过 softmax attention，不受 sink 影响。相比 StreamingLLM（利用 sink 而非消除）、register tokens（仅 ViT）、Quiet Attention（softmax 分母 +1），Hymba 的方案最全面。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解方案对比：(1) StreamingLLM——保留 sink tokens + 最近 tokens；(2) Register tokens——append 可学习 token；(3) Quiet Attention——softmax 分母 +1，等效 all-zero token；(4) SWA——物理隔离。Hymba 同时使用方案 (2)+(3)+(4)。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models

## Effective Receptive Field (ERF / 有效感受野)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ERF 度量神经网络中 token 间信息传播的有效距离。在 LLM 中定义为所有层所有 head 中最后 token 到之前 token 的归一化 attention score 的加权平均距离：$ERF \approx \sum_n \sum_h \sum_s \frac{2 M^h(S,s) \cdot (S-s) \cdot (N-n+1)}{H N (N+1)}$。Hymba 用 ERF 分析不同架构的信息传播效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hymba ERF 分析结论：(1) Llama3 ERF 最大但 cache 代价最高；(2) Parallel hybrid-head 的 ERF 比 sequential hybrid 大一个数量级，cache 相当；(3) 纯 Mamba ERF 最小。这直接解释了 parallel 结构 recall 优势（+4.74%）。

```
ERF = 0
for n in 0..N:
    for h in 0..H-1:
        attn = attention_map[layer=n][head=h][-1, :]
        for s in range(S+1):
            ERF += 2 * attn[s] * (S-s) * (N-n+1) / (H * N * (N+1))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ERF 主要作为架构分析工具：(1) 比较架构信息传播效率；(2) 指导 attention 模式设计；(3) 与 task accuracy 交叉验证。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models

---

## Block-wise Absmax Quantization（逐块绝对值最大量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Absmax Quantization 是一种数据无关（data-free）的权重量化方法，由 Dettmers et al. (2022) 在 8-bit Optimizers 中首次引入，后被 QLoRA（NF4）采用。其核心流程：(1) 将网络权重 W 按固定 block size I（如 64 或 128）分组；(2) 对每 block b 计算绝对值最大值 `w_b^max = max_i |w_{b,i}|` 作为归一化常数；(3) 每个权重除以该 block 的 absmax，归一化到 [-1, 1]；(4) 对归一化权重用固定的 scalar quantizer（如 NF4, AF4, BOF4）量化到 4-bit；(5) 存储 4-bit 索引 × |W| + 每 block 一个 FP16/BF16 量化常数。解码时：`Ŵ_{b,i} = w_b^max * x̂(index)`。该方法不需要校准数据、不需要计算激活值，量化过程极快且内存开销低，但精度略低于依赖校准数据的 PTQ 方法（如 GPTQ, AWQ）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Block-wise Absmax Quantization Pipeline
W = linear_layer.weight                # [d_out, d_in]
W_flat = W.reshape(-1)                  # flatten
I = 64                                  # block size
B = len(W_flat) // I                    # number of blocks

# Step 1: Block partitioning + absmax computation
for b in 1..B:
    block = W_flat[(b-1)*I : b*I]
    w_max[b] = max(abs(block))           # quantization constant (store in BF16)

# Step 2: Normalization
for b in 1..B:
    for i in 1..I:
        x[b,i] = W_flat[(b-1)*I + i] / w_max[b]   # normalized to [-1, 1]

# Step 3: Scalar quantization (e.g., BOF4-S MSE codebook)
codebook = [-0.8568, -0.6693, ..., 0.0, ..., 1.0]  # 16 levels
for b in 1..B:
    for i in 1..I:
        idx = argmin_ℓ |x[b,i] - codebook[ℓ]|
        W_quant_idx[(b-1)*I + i] = idx

# Step 4: Decoding (inference)
for b in 1..B:
    for i in 1..I:
        W_hat[(b-1)*I + i] = w_max[b] * codebook[W_quant_idx[(b-1)*I + i]]
```
关键设计决策：block size I 越小→量化常数越多→精度越高（因 outlier 影响局限在小范围）但内存开销越大。典型 I=64 时，量化常数开销约为 16/64=0.25 bit 每参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
bitsandbytes 库实现了 block-wise absmax 量化为 NF4 格式。在 HuggingFace 中使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_block_size=64)`。BOF4 论文提供了开源实现：https://github.com/ifnspaml/bof4，支持 BOF4/BOF4-S 码本和 OPQ。主要应用：(1) QLoRA 微调；(2) 内存受限推理场景；(3) PTQ 方法对比 baseline。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---

## Signed Absolute Block Maximum Normalization（带符号绝对块最大值归一化 / BOF4-S 归一化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Signed Absolute Block Maximum Normalization 是 BOF4 论文提出的归一化改进。传统 absmax normalization 除以 `max_i |w_{b,i}|`（恒正），归一化后权重在 [-1,1]，两端各有 1/(2I) 离散概率质量。Signed 版本改用 `w_b^max = w_{b, argmax_i|w_{b,i}|}`（带符号值），归一化后仅在 x=1 有 1/I 离散概率质量。这使得量化码本只需固定一个端点（x̂(16)=1）而非两个（-1 和 1），释放一个 reconstruction level 降低整体量化误差。signed normalization 与 double quantization 组合需额外 1 bit 编码量化常数符号。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Standard absmax (NF4, AF4, BOF4):
w_max = max(abs(block))                # always positive
x = block / w_max                       # masses at both -1 (1/2I) and +1 (1/2I)
# Codebook fixes BOTH x̂(1)=-1, x̂(16)=1 → wastes one level

# Signed absmax (BOF4-S):
j_star = argmax(abs(block))
w_max = block[j_star]                  # keeps the SIGN
x = block / w_max                       # mass at +1 only (1/I)
# Codebook fixes only x̂(16)=1 → one extra free level
```
假设 block 中最大绝对值权重为 -0.5。Standard：除以 0.5，该权重映射到 x=-1；Signed：除以 -0.5，该权重映射到 x=1。均精确保存最大权重（BF16 精度），但 signed 版本归一化后分布更紧凑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 https://github.com/ifnspaml/bof4 中实现。关键：(1) 归一化使用 signed max；(2) Lloyd 训练码本时固定 2 个 level（0 和 1）而非 3 个（-1, 0, 1）；(3) 最外层 centroid 更新中概率质量从 1/(2I) 变为 1/I（论文 Eq. 40 vs 39）。BOF4-S 在所有 block size 和误差指标下均优于 BOF4（non-signed）。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---

## BOF4 / BOF4-S（4-bit Block-Wise Optimal Float / 4-bit 逐块最优浮点量化码本）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BOF4（4-bit Block-wise Optimal Float）是 Blumenberg et al. (2025) 提出的 block-wise absmax 量化最优码本家族。核心创新：将 Lloyd's algorithm 的 centroid 更新修改为考虑 block maximum 分布权重——最小化端到端 `MSE(W,Q(W))` 或 `MAE(W,Q(W))`，而非仅归一化权重的量化误差。MSE centroid = 归一化权重的 `w_b^max` 平方加权平均（Eq. 6），MAE centroid = `w_b^max` 加权中位数（Eq. 8）。大 `w_b^max` block 的归一化权重在 centroid 更新中贡献更大（解码时误差被放大）。BOF4 固定 3 个 level（-1, 0, 1），BOF4-S 配合 signed normalization 仅固定 2 个（0, 1）。码本可在高斯权重假设下通过数值积分（Eq. 5/7）或 MC 采样（Eq. 6/8）计算，两者等价（MSE ≈ -56 dB）。BOF4-S(MSE)+OPQ 在 Llama-3.1 8B 上 WikiText-2 PPL=8.43，优于 NF4 (8.53) 和 AF4 (8.51)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BOF4 EM Algorithm (modified Lloyd's algorithm)
# Input: p_W = N(0,1), block size I
# Output: codebook x̂[1..16]

# Step 1: Initialize (e.g., from NF4 levels)
x̂ = [-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
       0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
fixed = {1, 8, 16}  # BOF4-S: {8, 16}

# Step 2: Sample W ~ N(0,1) shape [B, I], normalize x=b,i = w[b,i]/w_max[b]

repeat until convergence:
    # 3a. Partition (nearest neighbor): assign each x to nearest x̂[j]
    
    # 3b. Centroid update (MODIFIED):
    for ℓ in 2..15 where ℓ not in fixed:
        # MSE (Eq. 6): weighted mean by w_max²
        x̂[ℓ] = Σ_k (w_max[k]² * x_k) / Σ_k (w_max[k]²)
        # MAE (Eq. 8): weighted median by w_max
        # x̂[ℓ] = weighted_median({x_k}, weights={w_max[k]})
```
关键区别 vs 标准 Lloyd：centroid 从均匀加权均值变为 block max 平方加权均值。源于 `MSE(W,Q(W)) = (1/B)·Σ_b (w_b^max)²·MSE(X_b, Q̃(X_b))` 分解（Eq. 60）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/ifnspaml/bof4。码本以 Python 数组提供（BOF4-S MSE, I ∈ {32, 64, 128, 256}），也可 MC 方法按需生成。使用时替换 NF4/AF4 码本，解码逻辑相同。码本硬编码，不随具体模型权重调整，不受 OPQ 影响。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)

---

## Outlier-Preserving Quantization (OPQ / 离群值保留量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OPQ（Outlier-Preserving Quantization）是 BOF4 论文提出的混合精度 block-wise 量化策略。解决 outlier weights 破坏 block-wise absmax 归一化分布假设的问题：少数极端 outlier 导致其 block 的 `w_b^max` 异常大，使归一化后非 outlier 权重被过度压缩到零附近（underrange），量化器偏离最优设计区间。OPQ 在量化前将 outlier 替换为 0，单独存储为 BF16 + 64-bit position index。Outlier 判定：`|w_{b,i}| > σ_b * F_M^{-1}(q)`，其中 σ_b 为 block 样本标准差（Eq. 73），`F_M^{-1}(q)` 为绝对 block maxima 分布的 q-分位数（q=0.95）。OPQ 与任意 block-wise 量化方法组合，码本不变。额外内存：I=64, q=0.95 时约 0.96%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# OPQ Algorithm (q = 0.95)
threshold = F_M^{-1}(0.95)  # quantile of absolute block maxima distribution

# Step 1: Per-block standard deviation
for each block b:
    σ_b = std(block[b, :])

# Step 2: Outlier detection and removal
outlier_store = []
for each weight w[b,i]:
    if abs(w[b,i]) > σ_b * threshold:
        outlier_store.append((position, w[b,i]))
        w[b,i] = 0  # replace with zero

# Step 3: Standard block-wise quantization on cleaned weights
w_quant, w_max = blockwise_absmax_quantize(w, codebook, I)

# Step 4: Decode — reconstruct from 4-bit, then overwrite outlier positions with BF16
```
Outlier 判定直观理解（Fig. 7）：`σ_b * F_M^{-1}(0.95)` 表示"95% block 中绝对最大值都不超过的阈值"。超过该阈值说明该权重大得不正常（非高斯），应作 outlier 处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/ifnspaml/bof4。OPQ 与具体码本解耦（可与 NF4, AF4, BOF4, BOF4-S 任意组合）。运行时开销极小（RTX 4070 Ti Super 上生成 1000 tokens 额外耗时随 I 增大递减，Fig. 11）。大 block size（I ≥ 128）效果尤其显著。推荐 q=0.95（{0.9, 0.95, 0.97, 0.99} 中最佳平衡）。

SpQR 提出另一种 outlier 处理方法：使用 OBS 框架的封闭形式敏感度准则 s_ij = (w_ij − quant(w_ij))² / (2[H⁻¹]_jj) 而非 σ_b 乘以分位数阈值。SpQR 的 outlier 判定发生在量化过程中（而非预处理），通过 leave-one-out error 对比动态确定：E_base − E_ol > τ。Outlier 以 CSR 稀疏格式存储（32 bits/outlier：16-bit value + 16-bit col index），约 1% 的权重被保留为 16-bit。非 outlier 权重以 3-4 bit 量化，排除 outlier 后 min-max scale 显著减小。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

---

## KBVQ-MoE (KLT-guided SVD with Bias-Corrected VQ for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KBVQ-MoE 是第一个专为 Mixture-of-Experts (MoE) 架构设计的向量量化（VQ）框架，由 Houmo AI 提出（ICLR 2026 Poster, arXiv:2602.11184）。框架集成两个创新模块：(1) **IDRE（Input-driven Redundancy Elimination）**：KLT 引导的 SVD 分解，将 expert 权重投影到输入相干空间，提取主导共享分量保留全精度，仅对 expert-specific 残差做 VQ；(2) **BCOS（Bias-Corrected Output Stabilization）**：对 VQ 量化的 expert-specific 输出做 channel-wise affine compensation（scale + bias），使每个 channel 的 mean/variance 与 FP16 对齐。总压缩率公式：`ratio = (16(m+ln)min(m,l)k + m·l·b·n + 2^(bv+4)·v·n + 32ln) / (16nml)`，其中 m×l 为权重维度，n 为 expert 数，k 为 SVD 截断秩比例，v 为 VQ 子向量长度，b 为位宽。典型配置（k=1/128, v=4, b=2）下有效位宽 ~2.08 bits，压缩率 87%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KBVQ-MoE 完整 pipeline：
```
=== 离线校准阶段 ===
输入: expert weights {W^(i)}, 校准数据 X, 截断秩 k, VQ 子向量长度 d

# IDRE: KLT-guided SVD 冗余消除
1. C_X = X^T X / (B-1)                        # 输入协方差
2. C_X = U_KLT Λ_KLT U_KLT^T                   # KLT 特征分解
3. U_X = U_KLT Λ_KLT^{1/2}                      # 输入相干基
4. for i in 1..n: W̃^(i) = W^(i) U_X             # 投影到相干空间
5. W̄ = [W̃^(1); ...; W̃^(n)]                      # (n·oc)×ic 统一表示
6. W̄ = (U Σ V^T)^T                            # SVD 分解
7. 选 top-k: U_k = U_{:,1:k}, V_k = V_{:,1:k}
8. 划分 V_k 按 expert: V_k = [Σ_k V_k^(1); ...; Σ_k V_k^(n)]
9. for i in 1..n:
     U_share = U_X^{-1} U_k                     # ic×k 共享映射
     W_share^(i) = (U_share (V_k^(i))^T)^T      # 共享分量 (FP16)
     W_quant^(i) = W^(i) - W_share^(i)          # expert-specific 残差

# VQ 量化 (仅对 W_quant)
10. for i in 1..n:
      将 W_quant^(i) 划分为 d 维子向量 {z}
      K-means++ 初始化 codebook C = {c_1,...,c_K}
      训练 codebook via K-means (100 iters)
      for each z: q = argmin_j ||z - c_j||^2; z_q = c_q
      → W_quant,VQ^(i)

# BCOS: Bias 校正
11. for each expert i:
      ŷ = (W_share^(i) + W_quant,VQ^(i)) x      # 量化输出
      从 calibration 估计 μ_y, σ_y (原始), μ_ŷ, σ_ŷ (量化)
      s_j = σ_{y_j} / σ_{ŷ_j} - 1              # per-channel scale
      b_j = μ_{y_j} - (1+s_j) μ_{ŷ_j}          # per-channel bias

=== 推理阶段 ===
12. y_corr = (1+s) ⊙ ((W_share + W_quant,VQ) x) + b
13. W_share 以 FP16 计算, W_quant,VQ 通过 index→codebook 查表解码
```

关键超参数：k = ic/128（SVD 截断秩，经验最优，average bit-width 增加约 0.08）；d = 4（VQ 子向量长度）；K-means 100 iterations + K-means++ 初始化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KBVQ-MoE 针对 MoE LLM 的 decoder-only 架构设计。校准数据：从 RedPajama 采样 256 条（seq len=4096, seed=42）。适用模型：Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, Mixtral-8x7B, DeepseekV2-Lite。已知限制：(1) SVD 截断秩 k 需经验选择，无自适应机制；(2) 仅验证 decoder-only MoE，未测试 encoder-decoder 或多模态 MoE；(3) 未测试 1-bit 极端量化。开源：arXiv:2602.11184, ICLR 2026 Poster，论文未注明代码仓库。推理加速：Qwen1.5-MoE-A2.7B 2-bit decode speed 35.24 tok/s vs BF16 22.31 tok/s (1.58× speedup)，BCOS overhead <0.1% FLOPs。评测工具：LM-Evaluation-Harness (v0.4.0)。

涉及论文标题：
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---

## Input-driven Redundancy Elimination (IDRE / KLT-guided SVD for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IDRE（Input-driven Redundancy Elimination）是 KBVQ-MoE 框架的前处理模块，通过 KLT 引导的 SVD 消除 MoE expert 间的冗余表示。核心思想：MoE 中不同 expert 对相同输入常产生高度相似输出（Fig. 2a），VQ 重复量化相似表示浪费 codebook 容量。IDRE 分三步：(1) 对输入激活 X 做 KLT 分解构建输入相干基 `U_X = U_KLT Λ_KLT^{1/2}`，将输入按能量方向正交化；(2) 将所有 expert 权重投影到输入相干空间形成统一表示 W̄；(3) 对 W̄ 做截断 SVD 提取 top-k 主导共享分量（保留 FP16），剩余作为 expert-specific 残差交 VQ 量化。KLT 的关键作用：使 SVD 的 Gram 矩阵 `S = W̄^T W̄` 的频谱同时反映输入能量（通过 Λ_X）和跨 expert 权重能量（通过 Σ_i W^(i)T W^(i)），确保提取的共享方向在"输入高能量"和"跨 expert 高使用率"两个维度上同时主导。理论保证（Appendix A.2）：在输入相干空间中，输出 MSE 等价于权重误差的 Frobenius 范数 `L = Σ_i ||(W̃^(i) - W^(i)) U_X||_F^2`，因此所有最小化输出失真的提取/量化操作应在此空间执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# IDRE 核心计算 (Algorithm 1 pre-process 部分)
输入: expert 权重 {W^(1),...,W^(n)} ∈ R^{oc×ic}, 校准激活 X ∈ R^{B×ic}
输出: U_share, {W_share^(i)}, {W_quant^(i)}

# Step 1: KLT 分解输入激活
C_X = (X^T X) / (B-1)                           # ic×ic 输入协方差
U_KLT, Λ_KLT = eig(C_X)                         # C_X = U_KLT Λ_KLT U_KLT^T
U_X = U_KLT @ sqrt(Λ_KLT)                       # ic×ic 输入相干基

# Step 2: 投影 + 堆叠所有 expert
W̄ = zeros((n * oc, ic))
for i in 1..n:
    W̃^(i) = W^(i) @ U_X                         # oc×ic 投影到相干空间
    W̄[i*oc:(i+1)*oc, :] = W̃^(i)                 # 纵向堆叠

# Step 3: 截断 SVD 提取共享子空间
U, Σ, V = SVD(W̄)                                # W̄ 的 SVD
k = ic // 128                                   # 经验截断秩
U_k = U[:, :k]                                  # ic×k 共享左奇异向量
V_k_full = V[:, :k] @ diag(Σ[:k])              # (n·oc)×k 加权右奇异向量

# Step 4: 分解共享/残差
U_share = inv(U_X) @ U_k                        # ic×k 原始空间共享映射
for i in 1..n:
    V_k_i = V_k_full[i*oc:(i+1)*oc, :]          # oc×k expert i 的右奇异分量
    W_share^(i) = (U_share @ V_k_i^T)^T          # oc×ic 共享分量 (保留 FP16)
    W_quant^(i) = W^(i) - W_share^(i)            # oc×ic expert-specific 残差
```

关键性质：(1) 共享子空间通过 Ky Fan 定理最优：`max_{U^T U=I} ||W̄ U||_F^2`，保留了跨 expert 的最大能量方向；(2) 截断误差由尾部奇异值精确控制：`Σ_{j>k} σ_j^2`；(3) 冗余消除率 `ρ_k = Σ_{j=1}^k σ_j^2 / Σ_{j=1}^{ic} σ_j^2`，k=ic/128 时典型 ρ_k ≈ 0.6-0.8（功率律衰减 `σ_j^2 ∝ j^{-α}, α>1`）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
IDRE 应用于 MoE 层所有 expert（shared + routing）的 MLP 权重。仅离线执行一次（开销约等于 1 次前向传播）。消融验证：KLT+SVD 优于纯 SVD（Table 3: Qwen1.5-MoE-A2.7B 2-bit W2 PPL 从 14.03 降至 11.87）；k=ic/128 为经验最优平衡点（Table 4: k=1/128 时 PPL 11.87, k=1/32 时 PPL 11.01 但 bit-width 从 2.08 升至 2.20）。IDRE 可解耦为独立 pre-processing 模块，与不同 VQ 方法组合（GPTVQ, VPTQ, PCDVQ 等）。

涉及论文标题：
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---

## Bias-Corrected Output Stabilization (BCOS / Channel-wise Affine Compensation for VQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BCOS（Bias-Corrected Output Stabilization）是 KBVQ-MoE 框架的后处理模块，校正 VQ 量化后 MoE expert 输出的 distributional shift。问题背景：量化误差在各层累积导致 layer output 的 per-channel mean/variance 偏离 FP16 基线。MoE 架构中多个 expert 的 biased outputs 通过 gating weights 加权求和后被聚合放大，distributional drift 比 dense LLM 更严重（Fig. 3: Direct VQ 后 mean/variance 显著偏离 FP）。BCOS 以 channel-wise affine compensation 校正：`y_corr = (1+s) ⊙ ŷ + b`，其中 ŷ = W_VQ x。s 和 b 基于 MMSE 闭式解：`s_j ≈ σ_{y_j}/σ_{ŷ_j} - 1`, `b_j = μ_{y_j} - (1+s_j)μ_{ŷ_j}`。推导（Appendix A.4）：将校正化为 `min_{s,b} E[||y - ((1+s)⊙ŷ + b)||²]`，对第 j 个 channel 等价于一元线性回归，闭式解为 `α_j = Cov(y_j,ŷ_j)/Var(ŷ_j)`, `b_j = μ_{y_j} - α_j μ_{ŷ_j}`。因 y_j 和 ŷ_j 高度相关（仅差量化噪声），近似 `Cov(y_j,ŷ_j) ≈ σ_{y_j} σ_{ŷ_j}` 得 `s_j = α_j - 1 ≈ σ_{y_j}/σ_{ŷ_j} - 1`。该近似在高相关条件下等价于 MMSE-optimal 估计，非启发式调整。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BCOS 参数估计 + 推理流程
# 离线: 估计 s,b (per expert, per layer)
输入: 原始权重 W, 量化权重 Ŵ = W_share + W_quant,VQ
      校准激活 X

# 收集 per-channel 统计量
y = W @ X                                      # 原始输出 oc×B
ŷ = Ŵ @ X                                      # 量化输出 oc×B

for j in 1..oc:
    μ_y[j] = mean(y[j, :])
    σ_y[j] = std(y[j, :])
    μ_ŷ[j] = mean(ŷ[j, :])
    σ_ŷ[j] = std(ŷ[j, :])

# 计算校正参数 (MMSE 闭式解)
for j in 1..oc:
    s[j] = σ_y[j] / σ_ŷ[j] - 1                 # scale: 对齐 variance
    b[j] = μ_y[j] - (1 + s[j]) * μ_ŷ[j]        # bias: 对齐 mean

# 推理: channel-wise affine
y_corr = (1 + s) ⊙ (Ŵ @ x) + b                 # 逐 channel 乘加
```

存储开销：每层 2·oc 个 FP16 参数（s 和 b 各 oc 个），对 Qwen1.5-MoE-A2.7B gate_proj (oc=5632) 仅 ~22KB/layer。推理计算：仅 element-wise multiply-add，<0.1% expert forward FLOPs。消融（Table 15: BCOS 内部分解）：variance-only 贡献更大（PPL 11.03→10.38），mean-only 贡献小（11.03→11.01），mean+variance 组合最佳（9.61）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BCOS 配合 IDRE 使用效果最佳（IDRE 先消除冗余降低量化难度，BCOS 再校正残余误差）。IDRE+BCOS 组合在 Qwen3-30B-A3B 3-bit 下 PPL 从 18.72（无处理）降至 9.26（-50.5%）。BCOS 为通用模块：可作为 plugin 集成到任何 MoE VQ pipeline 中，在 GPTVQ 和 VPTQ 上均有验证（Table 5: GPTVQ+IDRE+BCOS vs GPTVQ only 在 Qwen1.5-MoE-A2.7B 2-bit 下 PPL 从 12.88 降至 9.43）。局限：BCOS 的 scale 近似 `Cov(y_j,ŷ_j) ≈ σ_{y_j} σ_{ŷ_j}` 在极低比特下（如 1-bit）相关度下降时精度可能降低，论文未测试该场景。

涉及论文标题：
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---

## Learnable Logarithmic Rounding (LLR, 可学习对数舍入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Logarithmic Rounding (LLR) 是 LOGART 提出的首个对对数域 PTQ 实现可学习舍入的技术。现有对数 PTQ 均使用 RTN（⌊·⌋ 或 ⌊·⌉）直接舍入到最近量化级别。LLR 将舍入决策参数化：用 floor ⌊·⌋ 作为下界，引入可学习变量 R（每个 weight 一个），通过 sigmoid σ(R) ∈ (0,1) 控制每个 weight 是向下还是向上舍入。Quant: Q_W = clamp(⌊-log_2(|W|/s)⌋ + σ(R), 0, 2^{N-1}-1)，Dequant: Ŵ = s · sign(W) ⊙ 2^{-Q_W}。优化目标：min_R E[||ΔW·X||_F²] + λ·Σ(1-|2σ(R)-1|^β)，正则项鼓励 σ(R) 逼近 0 或 1（hard rounding）。LLR 梯度：∂L/∂R = 2s·ln2 · M_c ⊙ 2^{-Q_W} ⊙ sign(W) ⊙ [(WX-ŴX)X^T] ⊙ σ'(R) + λ·∂f_reg/∂R。与线性可学习舍入（AdaRound）的关键区别：梯度包含指数项 2^{-Q_W}，小幅值 weight 梯度小、大幅值 weight 梯度大，与对数分布密度结构一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# LLR (Learnable Logarithmic Rounding)
R = nn.Parameter(torch.zeros_like(W))  # per-weight learnable variable
opt = Adam([R], lr=0.05→0.015, CosineAnnealingLR)
for iter in range(500):  # LLM, 2000 for Vision
    # Soft quantize (HAF noise injected in forward pass)
    Q_W = clamp(floor(-log_B(|W|/(s_of·S))) + σ(R), l_a, U)
    Ŵ = S * sign(W) * B^{-Q_W}
    # Loss: reconstruction + regularization
    L_recon = ||(W - Ŵ) @ X||_F²
    L_reg = λ * Σ(1 - |2σ(R)-1|^β)
    L = L_recon + L_reg
    L.backward()  # auto-diff through quant chain
    opt.step()
# Hard round after convergence
Q_final = clamp(floor(-log_B(|W|/(s_of·S))) + round(σ(R)), l_a, U)
```

术语一般如何实现？如何使用？
LLR 在 PyTorch 中实现：R 为 nn.Parameter，σ(R) 用 torch.sigmoid(R)。前向传播 soft quantize，loss 计算 Frobenius 范数重建误差。Adam + CosineAnnealingLR。LOGART 开源: https://github.com/logart-lab/logart。LLR 需反向传播因此比 RTN 慢，但离线 PTQ 阶段完成一次后推理使用 hard-quantized 权重。LLR 与 OHS 有强协同：OHS 先建立优质网格，LLR 在其上收敛更快更优（OHS+LLR 500 iters > 纯 LLR 2000 iters）。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---

## Dynamic Base Quantization (DLog, 动态基量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Base Quantization (DLog) 是混合基对数量化方案：大幅值 weight 用 base-√2（细粒度），小幅值 weight 用 base-2（硬件友好）。码本含 n₁ 个 base-√2 码字和 n₂ 个 base-2 码字（n₁+n₂=2^{N-1}-1）。阈值 t 分割区域：|W_ij| ≥ t → base-√2, scale=√2^m, U=n₁-1；|W_ij| < t → base-2, scale=2^{⌊(m-n₁)/2⌋}, U=n₂-1。其中 m = ⌊log_√2(max(|W|))⌉。元素级量化/反量化：Q_W = clamp(⌊-log_B(|W|/S)⌋, 0, U)，Ŵ = S·sign(W) ⊙ B^{-Q_W}，B ∈ {2, √2} per-element。

从算法pipeline角度拆解术语：
```
# DLog Quant/Dequant
m = round(log_sqrt2(max(|W|)))
t = sqrt2^{(m-n1+1)/2 + floor((m-n1)/2)}
for each (i,j):
    if |W[i,j]| >= t:
        B[i,j]=sqrt2; S[i,j]=sqrt2^m; U[i,j]=n1-1
    else:
        B[i,j]=2; S[i,j]=2^{floor((m-n1)/2)}; U[i,j]=n2-1
    Q_W[i,j] = clamp(floor(-log_{B[i,j]}(|W[i,j]|/S[i,j])), 0, U[i,j])
Ŵ = S ⊙ sign(W) ⊙ B^{-Q_W}  # element-wise dequant
```
n₁:n₂ 由 OHS 中 DBS (Dynamic Base Search) 以 block-wise 重建误差最小化为目标搜索。

术语一般如何实现？如何使用？
在 PyTorch 中：逐元素选择 base（torch.where），分别计算 log_2 和 log_√2 对数域索引。DBS 使用离散 grid search 遍历 n₁ 整数值，每次评估块级 Frobenius 重建损失。DLog 在 LOGART 消融中 alone 将 PPL 从 170.64 (pure Log2 RTN) 降至 66.63 (OPT-125M 3-bit)，为最有效的单组件。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---

## Asymmetric Logarithmic Quantizer (非对称对数量化器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric Logarithmic Quantizer 是 LOGART 提出的首个对数域非对称量化器。传统对数 PTQ 取 |W| 后对称量化，无法匹配 LLM 中非对称权重分布（正负 range 不均）。线性 PTQ 可用 zero-point 偏移解决，但对数域因零附近非线性间距不可行。LogART 通过自适应下界 l_a 为正负值分配不同数量码字：w_h=max(w_max,-w_min), w_l=min(w_max,-w_min), d_a 度量正负 range 在对数域的差异，l_a=⌊d_a/2⌋ 作为量化下界 clamp 起点（从 [0,U] 变为 [l_a,U]）。

从算法pipeline角度拆解术语：
```
# ABS (Asymmetric Bound Search) - calibration-free
for each channel:
    w_h = max(w_max, -w_min); w_l = min(w_max, -w_min)
    if w_l >= t:  d_a = floor(log_sqrt2(w_h)) - round(log_sqrt2(w_l))
    else:  d_a = n1 + floor((m-n1)/2) - round(log2(w_l))
    l_a = floor(d_a / 2)
# 量化 clamp 下界从 0 变为 l_a
```

术语一般如何实现？如何使用？
ABS 为 OHS 的 calibration-free 组件，直接基于 weight 统计量计算，无需任何校准数据。每 channel 独立计算 l_a，额外开销可忽略。消融：OPT-125M PPL 36.10→34.29，LLaMA2-7B PPL 6.56→6.45。对非对称分布明显的 LLaMA 增益更大。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---

## Outlier-Resilient Quantizer (离群值鲁棒量化器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Outlier-Resilient Quantizer 是 LOGART 的离群值自适应裁剪量化器。传统对数 PTQ 用 max(|W|) 定量化范围，单个 outlier 撑大范围降低整体精度。LogART 引入可搜索缩放因子 s_of ∈ (0,1] 缩放 scale S：Q_W = clamp(⌊-log_B(|W|/(s_of·S))⌋ + σ(R), l_a, U)。s_of < 1 时范围缩小，outlier 被裁剪到边界码字；s_of = 1 时退化为无裁剪。

从算法pipeline角度拆解术语：
```
# SFS (Scaling Factor Search) - block-wise, calibration-based
for each block:
    best_s_of = argmin_s_of ||(W_block - Ŵ_block(s_of))·X||_F²
# s_of 为 per-channel 参数，与 DBS 联合搜索
argmin_{s_of, n1, n2} E[||ΔW·X||_F²]
```

术语一般如何实现？如何使用？
SFS 是 LOGART 中最有影响力的组件之一：LLaMA2-7B 上加 SFS 后 PPL 从 9.74→6.24（LLR baseline）。s_of 为 per-channel 参数，block-wise 搜索提供更好的离群值感知。仅需 32 段校准数据，额外耗时数分钟（LLM）至数十秒（CNN）。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---

## Optimized Hyperparameter Search (OHS, 优化超参数搜索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OHS 是 LOGART 的多级超参数搜索策略，快速确定对数 PTQ 最优量化网格。三组件：(1) ABS: tensor-wise, calibration-free, 确定 l_a；(2) SFS: block-wise, calibration-based, 搜索 s_of 抵御 outlier；(3) DBS: block-wise, calibration-based, 搜索 n₁:n₂ 权衡精度与硬件效率。联合优化：argmin_{s_of,n₁,n₂} E[||ΔW·X||_F²]。

OHS 与 LLR 协同效应（核心 insight）：三角不等式分解量化误差 ||ΔW·H^{1/2}||² ≤ (E₁(OHS)+E₂(LLR))²。E₁ 是网格固有离散化误差（OHS 通过最优 θ*={s_of,n₁,l_a} 最小化），E₂ 是理想投影与 LLR 学习结果的残差。OHS 先建优质网格，LLR 再精细化舍入。实验：OHS+LLR 500 iters (PPL 31.15, 1.25 min) > 纯 LLR 2000 iters (PPL 36.27, 4.00 min)。

从算法pipeline角度拆解术语：
```
# OHS Pipeline
# 1. ABS: compute l_a per-channel (no calib)
l_a = asymmetric_bound(w_max, w_min)  # Eq.14-16
# 2. SFS+DBS: joint block-wise search with 32 calib samples
for each block B:
    best_loss = inf
    for s_of in [0.6, 0.7, ..., 1.0]:
        for n1 in [0, 1, ..., 2^{N-1}-1]:
            n2 = 2^{N-1} - 1 - n1
            Ŵ = log_quant(W_B, l_a, s_of, n1, n2)
            loss = ||(W_B - Ŵ)·X_B||_F²
            if loss < best_loss: best_config = (s_of, n1)
```

术语一般如何实现？如何使用？
OHS 三组件独立可组合。消融显示每组件 additive gain：DBS alone PPL 170.64→66.63；+SFS→36.10；+ABS→34.29；+LLR→31.15 (OPT-125M 3-bit)。Calib 数据仅 32 segments × 2048 tokens (LLM) 或 2048 images (vision)。多级设计避免暴力联合搜索的大搜索空间，将 OHS 运行时控制在数秒（小模型）到数分钟（大模型）。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

## Student Float (SF4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Student Float (SF4) 是一种基于 Student's t-distribution 推导的 4-bit 查找表（lookup-based）量化数据类型。与 Normal Float (NF4) 假设权重服从正态分布不同，SF4 基于对 30+ DNN 的大规模 profiling 发现——大多数 DNN 的权重和激活分布由 Student's t-distribution（自由度 ν≈5）最优近似。SF4 通过将概率质量等分为 16 份，经 t-distribution 分位数函数 Q_S(p;ν) 映射，归一化到 [-1,1]，生成 16 个量化层级。具体导出流程（Algorithm 1）：(1) 设定 δ = 0.5×(1/32 + 1/30)；(2) 在概率空间等距生成 p₁=δ,...,p₈=0.5,...,p₁₆=1-δ（固定 p₈=0.5 确保零点无损表示，对称侧多分配值以适配现代激活函数的正偏特性）；(3) s̃ᵢ = Q_S(pᵢ; ν=5)，经 t-distribution 分位数函数映射；(4) sᵢ = s̃ᵢ / maxᵢ|s̃ᵢ| 归一化到 [-1,1]。SF4 的 16 个量化层级为固定值，可作为 NF4 的直接替代品用于 weight-only PTQ。当 ν→∞ 时，t-distribution 收敛到正态分布，SF4 收敛到 NF4。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SF4 在 PTQ pipeline 中的角色是码本（codebook）：

```
# === 离线阶段：SF4 码本生成（一次性） ===
ν = 5  # 基于 30+ DNN profiling 最频自由度
δ = 0.5 * (1/32 + 1/30) ≈ 0.0323
p = [δ, ...均匀间距..., 0.5, ...均匀间距..., 1-δ]  # 16 个概率值
s̃ = Q_S(p; ν)    # t-distribution 分位数函数
s = s̃ / max|s̃|    # 归一化到 [-1,1]
# s = [s₁, s₂, ..., s₁₆] 即为 SF4 的 16 个量化层级

# === 推理/量化阶段：Block-wise SF4 量化 ===
W_flat = W.reshape(-1)
blocks = W_flat.reshape(B, 128)    # block size 128
for b in 1..B:
    w_max[b] = max(|blocks[b,:]|)
    for i in 1..128:
        x = blocks[b,i] / w_max[b]      # 归一化到 [-1,1]
        idx = argminⱼ |x - sⱼ|           # 最近邻 SF4 层级
        Ŵ[b,i] = w_max[b] * s_{idx}     # 解码

# 存储格式：4-bit index + per-block FP16 w_max
# 推理时查表解码：Ŵ = w_max × SF4_table[index]
```

与 NF4 对比：NF4 的概率空间划分基于 Gaussian 分位数（假设 N(0,σ²)），SF4 基于 t-distribution 分位数（假设 t(ν=5)）。论文实验证明 SF4 在 LLaMA2-7B 上平均 LAMBADA 准确率比 NF4 高 0.76%。SF4 对不同自由度的敏感度较低——ν=4/5/6 均接近最优——因此论文固定 ν=5 作为通用选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SF4 实现为修改版 Intel Neural Compressor 库（论文实验平台）中的查找表量化后端。在 QLoRA 风格的 weight-only PTQ 中，SF4 可以直接替换 NF4：只需替换 16 个码本值，block-wise absmax 归一化和查表解码流程完全相同。开源代码位于 https://github.com/cornell-zhang/llm-datatypes。由于 SF4 是纯查找表格式（类似 NF4），其硬件实现需要浮点查找表和高精度 MAC 单元（论文未为 SF4 设计专用 MAC），因此 SF4 主要用于：(a) weight-only 内存绑定推理（memory-bound inference），查表解码开销可忽略；(b) 作为高精度参考格式指导硬件高效数据类型（如 E2M1）的设计。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

## Normal Float (NF4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Normal Float (NF4) 是由 Dettmers et al. (2023) 在 QLoRA 工作中提出的 4-bit 查找表量化数据类型。NF4 假设网络权重服从正态分布 N(0,σ²)，基于信息论最优原则——每个量化层级应等概率使用（即每个层级映射到相同数量的权重值）。NF4 使用 Gaussian 分位数函数将概率质量均匀分为 16 份，经反函数映射生成 16 个量化值，确保量化后直方图近似平坦。NF4 固定 3 个值：-1、0、1（分别对应最小、零点和最大值），其余 13 个值由 Gaussian 分位数确定。NF4 的值分布特点是中心密集（零附近层级间距小）边缘稀疏（远离零的层级间距大），这匹配了正态分布的概率密度结构。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NF4 的导出基于 Gaussian 分位数函数 Φ⁻¹(p)：

```
# === NF4 码本导出 ===
# 将 (0, 1] 概率空间等分 16 个区间
p_i = i/16 for i=1..16  # 或按 QLoRA 的精确方法
# 保留 -1, 0, 1 为固定值
# 其余值通过 Gaussian 分位数映射
nf4_values = Φ⁻¹(p_i)  # standard normal quantile function
# 归一化使端点为 -1 和 1
```

NF4 在 QLoRA 中的使用流程：
```
W_flat = W.reshape(-1)
blocks = W_flat.reshape(B, 64)   # QLoRA 默认 block size 64
for b in 1..B:
    w_max[b] = max(|blocks[b,:]|)
    for i in 1..64:
        x = blocks[b,i] / w_max[b]
        idx = argmin |x - nf4_table|
        Ŵ[b,i] = w_max[b] * nf4_table[idx]
```

论文指出 NF4 的核心缺陷：正态分布假设不正确。30+ DNN 的 profiling 显示（Table 1, Kolmogorov-Smirnov 检验），大多数 DNN 分布最优近似于 Student's t-distribution（ν≈5），而非正态分布。正态分布无法同时拟合分布的尖峰（peak）和厚尾（tail）。这导致 NF4 的概率分配与真实权重分布不匹配，尤其在分布的尖峰区域欠量化而在尾部过量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NF4 是 QLoRA 量化微调框架的核心组件，通过 bitsandbytes 库（https://github.com/TimDettmers/bitsandbytes）实现。bitsandbytes 提供 CUDA kernel 实现 NF4 的查表解码 → FP16 GEMM。NF4 在推理时完全通过查表操作完成反量化，无算术计算。由于 NF4 需要浮点查找表和高精度累加器，其硬件直接实现成本较高，因此在实际系统中，NF4 更适合作为软件层的量化方案或作为设计硬件高效数据类型（如 E2M1）的精度参考。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

## APoT4 (Additive Powers-of-Two)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Additive Powers-of-Two (APoT) 是由 Li et al. (2020, ICLR) 提出的非均匀量化格式，其核心思想是将两个对数值相加来更好地匹配 DNN 数据分布。APoT 的一般形式为 (-1)^S (2^E + 2^Ẽ)，其中 E 和 Ẽ 分别是两组 2 的幂次指数集合。4-bit APoT4 在搜索空间被限制后的唯一合理变体为：E ∈ {0, 2⁻¹, 2⁻², 2⁻⁴}（4 个值），Ẽ ∈ {0, 2⁻³}（2 个值），所有可能的两项求和组合构成 16 个量化层级。APoT 的动机是纯对数格式（如 E3M0）在分布中心过度密集、尾部过度稀疏，添加第二项能调整为更匹配真实分布的钟形形状。在本论文的评价中，APoT4 的精度通常介于 INT4 和 E2M1 之间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
APoT4 的量化值计算：

```
# APoT4: s ∈ {+1, -1}, E ∈ S₁, Ẽ ∈ S₂
# S₁ = {0, 2⁻¹, 2⁻², 2⁻⁴}  (4 elements)
# S₂ = {0, 2⁻³}              (2 elements)
# 组合: 每个(E, Ẽ)对产生一个值 = 2^E + 2^Ẽ
# 正值集: {1+0, 1+0.125, 0.5+0, 0.5+0.125, 0.25+0, 0.25+0.125, 0.0625+0, 0.0625+0.125}
#        = {1.0, 1.125, 0.5, 0.625, 0.25, 0.375, 0.0625, 0.1875}
# 加上镜像负值 → 共 16 个值
```

推理时乘法：Ŵ × X = (-1)^S × (2^E + 2^Ẽ) × X = (-1)^S × (X << E + X << Ẽ)，即两次移位后相加。硬件上不需要乘法器，但论文指出 APoT4 需要额外逻辑将更高精度的 FP32/BF16 格式转换为 APoT 格式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在本论文中，APoT4 通过修改版 Intel Neural Compressor 库实现，其量化层级被硬编码为 16 个固定值，按最近邻原则（与 block-wise absmax scaling 结合）量化权重。APoT4 的 super-precision (SP) 变体复用负零位，增加一个额外求和组合以提升精度。在 MAC 硬件评估中，APoT4 的 accumulator 仅需 16 bit（与 INT4 相同），MAC 面积 181.6 µm²（比 INT4 160.7 µm² 大 13%，但精度更高）。然而论文指出 APoT4 需格式转换逻辑（FP32/BF16 → APoT），使其实用性不如 E2M1。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

## Student's t-distribution for DNN Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 DNN 量化上下文中，Student's t-distribution 用于建模神经网络权重和激活的统计分布。t-distribution 由自由度参数 ν 控制，其概率密度函数为：S(t;ν) = Γ((ν+1)/2) / (√(νπ)·Γ(ν/2)) · (1 + t²/ν)^{-(ν+1)/2}。ν 越小，分布的峰值越尖（leptokurtic）、尾部越厚（heavy-tailed）；ν→∞ 时收敛到标准正态分布 N(0,1)。本论文对 30+ DNN（含 LLM、BERT 类 Transformer、CNN、ViT）的权重和激活进行大规模 profiling，发现：(1) 大多数分布的自由度 ν 在个位数（约 2-7）；(2) Kolmogorov-Smirnov (KS) 检验证明 t-distribution 的拟合度优于正态分布（KS-Δ 大多为正值）；(3) ν ≈ 10 可视为正态分布的近似分界线。这一发现颠覆了量化领域长期依赖的正态分布假设（NF4 的基础）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
t-distribution profiling pipeline：

```
# 逐模型、逐层、逐 tensor 的统计分析
for model in [LLMs, BERT-variants, CNNs, ViTs]:
    for layer in model.modules:
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            W = layer.weight.detach().flatten()
            if W.numel() > N_max:
                W = random_downsample(W)  # 大张量下采样
    
            # 对每个 tensor 拟合 t-distribution 和 normal distribution
            ν_W, μ_W, σ_W = fit_t_distribution(W)  # MLE 估计
            μ_N, σ_N = fit_normal(W)
    
            # KS 检验比较拟合优度
            KS_t = ks_test(W, t_distribution(ν_W, μ_W, σ_W))
            KS_n = ks_test(W, normal(μ_N, σ_N))
            KS_Δ = KS_n - KS_t  # 正值 → t-distribution 更优

# 激活 profiling: 用随机生成输入（与模型适配的 shape）前向传播
# 收集每层的激活张量，重复上述统计
```

该 profiling 的直接输出：大多数模型的 ν 均值在 5 附近。这成为 SF4 固定 ν=5 的经验依据。Table 1 列出主要模型的 ν 和 KS-Δ 值：Mistral-7B (ν=1.66), LLaMA2-7B (ν=6.78), OPT-1B (ν=6.68), BLOOM-7B (ν=10.13, 接近正态边界)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于 HuggingFace Transformers、PyTorch torchvision 和 timm 库加载模型。对 nn.Linear、nn.Conv1D、nn.Conv2D 层的权重逐一 profiling；大张量（数亿参数）随机下采样（经验显示不影响结果）。激活 profiling 使用随机生成输入。t-distribution 参数通过最大似然估计（MLE）拟合。Profiling 结果直接指导：(a) SF4 的 ν 选择；(b) 判断哪些模型适合基于正态分布的量化方法（ν>10 的模型如 FLAN-T5, BLOOM-7B 可能与 NF4 兼容）；(c) 设计硬件高效数据类型（E2M1 形状分段逼近 SF4，因 SF4 的形状反映了 t-distribution 的概率密度结构）。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

## Sub-channel Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sub-channel quantization（子通道量化）也称为 block-wise quantization 或 group-wise quantization，是 LLM weight-only 量化的标准方法。它将权重矩阵按固定大小（block size）分组，每组独立计算一个量化缩放因子（通常是该组的 absmax），组内每个权重按该因子归一化后量化。与 per-tensor quantization（整个张量共享一个 scale）和 per-channel quantization（每行/每列一个 scale）相比，sub-channel quantization 在精度和存储开销之间取得平衡：更小的 block size → 更精准的局部归一化 → 量化误差更低，但需要存储更多 scale 因子。本论文使用 block size=128 作为默认值，因为它在精度和硬件对齐（大多数 MAC 单元可无需拆分累加即可处理 128 元素的 dot-product）之间取得折中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sub-channel weight-only 量化流程：

```
# W ∈ R^{d_out × d_in}
# Block size B=128 (沿 d_in 维度分组)
# 对于 INT4 量化：
W_flat = W.reshape(d_out, d_in // B, B)  # (d_out, n_blocks, B)
scales = zeros(d_out, n_blocks)            # per-block FP16 scales
W_q = zeros(d_out, n_blocks, B, dtype=int4)

for out_ch in 1..d_out:
    for block in 1..n_blocks:
        w_block = W[out_ch, block*B : (block+1)*B]
        s = max(|w_block|)                         # absmax scale
        scales[out_ch, block] = s
        W_q[out_ch, block] = round(w_block / s * Q_max)  # Q_max=7 for INT4

# 推理时解码:
Ŵ[out_ch, i] = scales[out_ch, i//B] * W_q[out_ch, i//B, i%B] / Q_max
```

本论文在 block size sweep 实验（Table 5）中发现：即使 block size 小到 16（超出当前 DNN 加速器的有效支持），各数据类型之间的 format 差异依然存在。例如 channel-wise（block size=d_in）下 E2M1-SP 比 INT4 平均准确率高 4.14%，block size=16 时差距仍为 1.59%。这表明数据类型的选择与 sub-channel 粒度相对独立。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sub-channel quantization 通过修改版 Intel Neural Compressor 库（论文）或 bitsandbytes/QLoRA 库实现。存储格式通常为：4-bit（或 N-bit）packed indices 数组 + per-block FP16/BF16 scale 数组。推理时使用 fused CUDA kernel：在寄存器中查表解码 → FP16 GEMM（或直接 INT4 GEMM for INT4）。论文指出 block size=128 是"足够大以对齐大多数 MAC 单元无需拆分累加"的平衡点。更小的 block size 虽然提升精度，但增加了 scale 存储开销（B=32: scale 占额外 1/32×16bit ≈ 0.5 bit/elem）和反量化调度的复杂度（需要更多 scale 加载和更细粒度的去量化操作）。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

## QLoRA (Quantized Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QLoRA（Quantized Low-Rank Adaptation）由 Dettmers et al. (2023) 提出，是一种将 4-bit NormalFloat (NF4) 量化与 LoRA 参数高效微调结合的 LLM 微调方法。其核心流程：(1) **量化阶段**：对预训练权重 W 执行 NF4 量化得到 Q = q_NF4(W)，将 FP16 权重压缩至 4-bit（存储为 4-bit index + block-wise absmax scale），同时对 scale 做双重量化（8-bit FP + 32-bit FP）进一步压缩；(2) **LoRA 附加**：在所有量化后的线性层附加低秩适配器 A, B（A ∼ N(0,σ²), B=0），标准 LoRA 零初始化；(3) **微调阶段**：freeze 量化权重 Q，仅优化 LoRA 参数。前向传播时 Q 临时解量化为 simulated FP16 参与计算 Y = X · dequant(Q) + X · A B^T。QLoRA 可在单张 48GB GPU 上微调 65B 模型（~18GB 显存）。

QLoRA 的核心缺陷——也是 LoftQ 论文的动机：量化误差导致初始权重 Q+AB^T = Q ≠ W，即微调起点偏离原始预训练权重。在低比特（2-bit/3-bit）时该偏差可能导致模型不收敛。LoftQ 通过交替优化解决了这一问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# QLoRA Pipeline

# Step 1: NF4 量化（离线，每个权重矩阵执行一次）
W_block = W.reshape(d_out, d_in // 64, 64)        # block_size=64
for each block:
    s = absmax(block)                               # FP32 scale
    W_norm = block / s                              # 归一化到 [-1,1]
    W_q[block] = NF4_quantize(W_norm)               # NF4 查表量化 → 4-bit index
    s_FP8 = FP8_quantize(s)                         # 双重量化 scale
    s_FP32 = FP32(s - s_FP8)                        # 残差

# Step 2: LoRA 初始化
lora_A = nn.Linear(d_in, r, bias=False)            # N(0, σ²) 初始化
lora_B = nn.Linear(r, d_out, bias=False)           # 零初始化

# Step 3: 微调
for batch in data:
    # 前向
    W_sim = NF4_dequantize(W_q, s_FP8, s_FP32)    # simulated dequantization
    h = x @ W_sim^T + (x @ lora_A @ lora_B^T) * α/r
    # 反向
    loss.backward()                                 # 梯度仅流经 lora_A, lora_B
    optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/artidoro/qlora。基于 HuggingFace Transformers + PEFT + bitsandbytes。使用方式：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)` → `prepare_model_for_kbit_training()` → `LoraConfig(r=64, lora_alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])`。关键设计：NF4 量化（信息论最优 4-bit 正态分布量化）+ 双重量化（压缩 scale 存储，每参数从 0.5 bit 降至 0.127 bit）+ 分页优化器（CPU offload 梯度检查点避免 OOM）。局限：2-bit 时失效（LoftQ 论文验证）；仅支持 weight-only 量化（激活保持 FP16/BF16）。

涉及论文标题：
- LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
- QERA: an Analytical Framework for Quantization Error Reconstruction
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

QeRL (Huang et al., NVIDIA, 2025) 发现 QLoRA (NF4+LoRA) 在 RL 训练中表现不佳：(1) NF4 的 unpack+lookup table 反量化导致 rollout 比 BF16 LoRA 慢 0.7-0.8×；(2) NF4 量化噪声是静态且确定性的，对 RL 后期训练的 exploitation 阶段不利——QeRL 引入 Adaptive Quantization Noise (AQN) 解决此问题；(3) QeRL (NVFP4+LoRA+AQN) 在 GSM8K 上 7B 达 90.8%，比 QLoRA (NF4+LoRA) 高 5.8 点。

QA-LoRA (Xu et al., Huawei, 2023) 指出 QLoRA 的核心局限：(1) QLoRA 微调后将 s·AB 加回量化权重，使最终模型恢复为 FP16，若需量化推理则必须做 PTQ（GPTQ 后处理），导致不可控的精度损失——尤其在 INT3/INT2 低位宽下退化严重（LLaMA-7B INT2: MMLU 5-shot 仅 25.0-25.8%）；(2) NF4 缺乏 CUDA 算子优化，训练和推理速度均慢于 INT4。QA-LoRA 通过 group-wise 操作使 LoRA 权重仅合并到零点矩阵 β（不改变 Ŵ 和 α），保持 INT 格式，解决了两个问题。QA-LoRA 论文称其 INT4 实现的 QLoRA 变体与原 NF4 QLoRA 精度差异在 ±0.5% 以内。

QLoRA 的安全影响：Q-resafe (Chen et al., ICML 2025) 的系统安全评估显示，QLoRA INT4 量化 Llama-2-7B-Chat 在 benign 校准数据集（UltraChat）上 ASR 从 0.3%（FP16）飙升至 42.3%，在直接有害数据集（AdvBench）上更升至 85.3%，是所有评估方法中安全退化最严重的。这是因为 QLoRA 优先优化效用而牺牲了安全——仅通过 LoRA 低秩适配调整少量参数，不足以保护模型的安全能力。Q-resafe 通过 DPO + 安全关键权重选择性修补可在 1.2 GPU-hours 内将 QLoRA INT4 ASR 从 42.3% 恢复至 2.4%。

## LoftQ (LoRA-Fine-Tuning-aware Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoftQ（LoRA-Fine-Tuning-aware Quantization）是 Li et al. (2023) 提出的量化+LoRA 联合初始化框架。核心创新：不直接量化预训练权重 W，而是同时优化量化 backbone Q 和 LoRA 适配器 A,B 来联合近似 W。目标函数：min ‖W − Q − AB^T‖_F。通过交替优化求解：(1) Quantization step: Q_t = q_N(W − A_{t-1}B_{t-1}^T)，量化低秩分量无法覆盖的残差部分；(2) SVD step: A_t, B_t = top-r SVD(W − Q_t)，用 top-r 奇异值/向量补偿量化误差。输出 Q_T（量化 backbone）和 A_T, B_T（非零 LoRA 适配器初始化，包含量化残差的低秩结构信息）。T=1 等价于 QLoRA 量化 + 量化残差 SVD 后处理，已有显著增益；T>1 通过交替迭代进一步缩小初始化差距。

与 QLoRA 的本质区别：QLoRA 先独立量化再零初始化 LoRA → Q+0 ≠ W；LoftQ 联合优化 → Q_T + A_T B_T^T ≈ W，微调起点更接近预训练权重。与量化函数 q_N 无关，支持 Uniform、NF2、NF4 等任意量化方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LoftQ 初始化（离线，对每个权重矩阵独立执行，可并行）

输入: 预训练权重 W ∈ R^{d1 × d2}, rank r, N-bit 量化函数 q_N, 交替步数 T
A_0, B_0 = 0

for t = 1 to T:
    # Quantization Step: 量化 (W - 当前低秩近似)
    residual = W - A_{t-1} @ B_{t-1}^T       # 去除低秩分量后的残差
    Q_t = q_N(residual)                       # N-bit 量化（Uniform 或 NF）

    # SVD Step: 低秩近似量化误差
    E_t = W - Q_t                             # 量化误差矩阵
    U, Σ, V^T = SVD(E_t)                      # 全奇异值分解
    # 取 top-r 分量:
    A_t[:, i] = sqrt(σ_i) * U[:, i]           # i = 1..r
    B_t[:, i] = sqrt(σ_i) * V[:, i]

输出: Q_T (量化 backbone), A_T, B_T (非零 LoRA 适配器初始化)

# 微调阶段（与 QLoRA 前向公式不同）
Y = X @ dequant(Q_T) + X @ A_T @ B_T^T       # A_T, B_T 非零
# 微调时 Q_T 冻结，仅优化 A_T, B_T（further from A_T, B_T init）

# 推理时可 merge:
W_final = dequant(Q_T) + A_T @ B_T^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源代码：https://github.com/yxli2123/LoftQ；预量化模型：https://huggingface.co/LoftQ。基于 HuggingFace Transformers + PyTorch 实现。LoftQ 作为微调前预处理步骤：加载预训练 FP16 模型 → 逐权重矩阵执行 LoftQ 算法（CPU 上执行，单矩阵 5120×5120 T=5 约 43s）→ 保存量化 backbone（int matrix + lookup table）和 LoRA adapter 初始化 → PEFT LoRA fine-tuning（标准流程，但 adapter 使用 LoftQ 输出而非随机初始化）。计算成本：逐矩阵独立执行且可并行，总量化时间可接受（LLAMA-2-13b 完整 LoftQ < 数分钟）。关键参数：r（LoRA rank，通常 8-64）、T（交替步数，1-10 均可，T=1 已有显著增益，T=5 通常饱和）、q_N（量化方法，NF2/NF4/Uniform 均适用）。适用场景：需要在极低比特（2-bit/3-bit）下微调 LLM，或 QLoRA 精度不足时作为替代方案。

涉及论文标题：
- LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models
- QERA: an Analytical Framework for Quantization Error Reconstruction

LoftQ 的核心缺陷（QERA 揭示）：LoftQ 最小化的是权重逼近误差 ||W − W̃ − C_k||_F，但 QERA 实验 (Figure 1) 证明该目标与最小化模型输出误差不等价——LoftQ 迭代数增加时所有层权重误差单调降，但模型输出误差可能上升（如 5-iter vs 3-iter 在 rank k=8 时输出误差更大）；rank 增加也不保证输出误差单调降。QERA 通过最小化层输出误差替代权重误差，给出了 QER 问题的正确优化目标和闭式解。

---

## Weight Magnitude Reduction (MagR) via ℓ∞-Regularization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Magnitude Reduction (MagR) 是一种基于 ℓ∞-regularized 最小二乘优化的权重预处理技术，在 PTQ 量化之前应用，目的是缩小预训练权重的 channel-wise 最大绝对值（ℓ∞ 范数），从而降低量化步长 δ = (max(w)−min(w))/(2^b−1)，减少量化误差。MagR 的核心优化问题为：min_w ½‖Xw − Xŵ‖² + α‖w‖∞，其中 X 是校准数据的特征矩阵，ŵ 是预训练权重，α 是惩罚参数。MagR 的关键洞察是：LLM 各层的特征矩阵 X 是近似秩亏的（fraction rank 均值 70-84%，最低仅 0.1%），因此 X 的核空间非平凡，存在许多 w 满足 Xw ≈ Xŵ 但 ‖w‖∞ 远小于 ‖ŵ‖∞。MagR 利用这一自由度在核空间中寻找 ℓ∞ 范数最小的解。不同于 AWQ、OmniQuant、QuIP 等线性变换方法需要推理时对特征施加逆变换 T⁻¹ 产生推理开销，MagR 是非线性变换，直接替换权重，推理时零开销。MagR 被 NeurIPS 2024 接收。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MagR 在 overall PTQ pipeline 中的位置：

```
# ====== 离线预处理（一次性，每个 linear layer） ======
输入: 预训练权重 Ŵ ∈ R^{m×n}, Hessian H = XᵀX ∈ R^{m×m}
超参数: K=150, α=10⁻³ (per-channel) / 10⁻⁴ (per-group), η = 1/λ_max(H)

W⁰ = Ŵ
for k = 0 to K-1:
    V^k = W^k - η · H · (W^k - Ŵ)                    # 梯度下降步
    # Proximal step via Moreau decomposition:
    # prox_{ηα‖·‖∞}(V^k) = V^k - ηα · proj_{‖·‖₁≤1}(V^k/(ηα))
    for j = 1 to n:                                   # 每列独立
        v_j = V^k[:, j]
        u = sort(|v_j|, descending)                   # O(m log m)
        ρ = max{i: u_i > (Σ_{r=1}^i u_r - ηα) / i}    # 找阈值索引
        θ = (Σ_{r=1}^ρ u_r - ηα) / ρ                  # 软阈值
        W^{k+1}[:, j] = sign(v_j) ⊙ max(|v_j| - θ, 0)
输出: 预处理权重 W' = W^K

# ====== 量化（与标准 PTQ 相同） ======
δ = β · (max(W'_col) - min(W'_col)) / (2^b - 1)   # β ∈ [0.80, 0.95]
W_q = δ · clamp(round(W'/δ) - z, 0, 2^b-1) + z·δ

# ====== 推理 ======
Y = X @ dequant(W_q)    # 零额外开销
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MagR 基于 PyTorch 实现，开源地址：https://github.com/AozhongZhang/MagR。基于 OPTQ 仓库构建，采用 block-wise 加载（每次 7 个 linear layer 到 GPU）。校准数据：WikiText2 128 个 2048-token 序列。关键参数：迭代数 K=150，α=10⁻³（per-channel）/ 10⁻⁴（per-group），步长 η=1/λ_max(H) 保证收敛，β ∈ [0.80, 0.95]（与 bit-width 正相关）。预处理时间：LLaMA2-7B ~15 min，13B ~30 min，70B ~3.5 hr（单 A100 80GB）。预处理后的权重可直接用于任何标准 PTQ 方法（RTN、OPTQ、QuIP），无需修改推理代码。per-group 变体：将 V ∈ R^{m×n} reshape 为 R^{d×(m·n/d)} 后独立做 ℓ₁-ball 投影，梯度步不变。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---

## Proximal Gradient Descent with ℓ₁-Ball Projection for PTQ Weight Preprocessing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Proximal Gradient Descent (PGD / 近端梯度下降) 是求解复合优化问题 min_x f(x) + g(x) 的一阶迭代算法，其中 f 是光滑凸函数（如最小二乘损失 ½‖XW − XŴ‖²_F），g 是非光滑凸函数（如 ℓ∞ 范数）。每轮迭代分两步：(1) Gradient Descent Step: v = x^k − η∇f(x^k)，沿负梯度方向下降；(2) Proximal Step: x^{k+1} = prox_{ηg}(v)，其中 prox_{ηg}(v) = argmin_x{½‖x−v‖² + ηg(x)} 是 g 的近端算子。在 MagR 中，梯度 ∇f(W) = XᵀX(W − Ŵ)，近端算子 prox_{t‖·‖∞} 通过 Moreau 分解转化为 ℓ₁-ball 投影。PGD 收敛保证：当步长 η ≤ 1/L（L 为 ∇f 的 Lipschitz 常数，即 λ_max(H)）时，PGD 以 O(1/k) 收敛到全局最优解。PGD 比 subgradient descent 快得多（对数收敛 vs 次线性收敛），且通过矩阵化实现支持列级并行处理整个权重矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MagR 中的 PGD 实现（Algorithm 1 + Algorithm 2）：

```
输入: Ŵ ∈ R^{m×n}, H = XᵀX ∈ R^{m×m}, K, η = 1/λ_max(H), α > 0
W⁰ = Ŵ

for k = 0 to K-1:
    # ---- Gradient Descent Step (matrix form, O(m²n)) ----
    G = H @ (W^k - Ŵ)              # ∈ R^{m×n}, 梯度
    V = W^k - η · G                # 梯度下降步

    # ---- Proximal Step (column-wise ℓ₁-ball projection) ----
    # Moreau: prox_{ηα‖·‖∞}(V) = V - ηα · proj_{‖·‖₁≤1}(V/(ηα))
    M = (‖V[:, j]‖₁ ≤ ηα for j=1..n)   # binary mask: 已在球内的列标记为 1
    U = sort(|V|, dim=0, descending)   # 列排序 → U ∈ R^{m×n}
    # 向量化找投影阈值索引 ρ_j (Algorithm 2, line 3)
    cumsum_U = cumsum(U, dim=0)        # 沿行累积和
    cond = U > (cumsum_U - ηα) / arange(1, m+1)  # 阈值条件矩阵
    ρ_j = max row index per column where cond is True
    # 软阈值参数 θ_j = (Σ_{r=1}^{ρ_j} U[r,j] - ηα) / ρ_j
    θ = gather_theta(cumsum_U, ρ_j)
    Θ = tile(θ, [m, 1])               # 广播到 m×n
    # 软阈值操作（仅对 M==0 的列）
    W^{k+1} = (1-M) ⊙ sign(V) ⊙ max(|V| - Θ, 0) + M ⊙ V

输出: W^K
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PGD 的 ℓ₁-ball 投影核心是排序 + 软阈值，O(m log m) per column。PyTorch 实现中使用 `torch.sort` + `torch.cumsum` + `torch.where` 进行向量化批量投影（Algorithm 2 的矩阵版本）。步长 η = 1/λ_max(H) 通过 power iteration 预计算。H = XᵀX 在预处理开始时计算一次即固定，后续仅需矩阵乘法。开源实现：https://github.com/AozhongZhang/MagR。K=150 经验上足够收敛。在 PyProximal（https://pyproximal.readthedocs.io）中也提供类似的 ℓ∞ proximal operator 实现。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---

## Moreau Decomposition for Proximal Operator of ℓ∞-Norm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Moreau Decomposition（莫罗分解）是凸分析中的基本定理[29]，将任意范数的近端算子与其对偶范数的单位球投影联系起来。对于 ℓ∞-norm，其对偶范数为 ℓ₁-norm，Moreau 分解给出了恒等式：prox_{t‖·‖∞}(v) = v − t · proj_{‖·‖₁≤1}(v/t)，其中 proj_{‖·‖₁≤1} 是将向量投影到 ℓ₁ 单位球上的投影算子。这一分解的关键价值在于：ℓ∞-norm 不可微，其近端算子没有直接闭式解；但 ℓ₁-ball 投影有高效的直接算法（排序+软阈值，O(m log m)）。通过 Moreau 分解，将"困难的 ℓ∞ 近端算子"转化为"高效的 ℓ₁-ball 投影"，使整个 PGD 迭代可行且高效。MagR 在每次 PGD 迭代中使用此分解：梯度下降步 → V，然后对各列分别计算 prox_{ηα‖·‖∞}(V) = V − ηα · proj_{‖·‖₁≤1}(V/(ηα))。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 已知: v ∈ R^m (梯度下降步结果的一列), t = ηα > 0

# Moreau 分解（理论等式）:
prox_{t‖·‖∞}(v) = v - t · proj_{‖·‖₁≤1}(v/t)

# 计算步骤:
# Step 1: 缩放
v_scaled = v / t

# Step 2: ℓ₁-ball 投影（Algorithm 3）
if ‖v_scaled‖₁ ≤ 1:
    p = v_scaled               # 已在球内
else:
    u = sort(|v_scaled|, descending)
    ρ = max{i: u_i > (Σ_{r=1}^i u_r - 1) / i}
    θ = (Σ_{r=1}^ρ u_r - 1) / ρ
    p = sign(v_scaled) ⊙ max(|v_scaled| - θ, 0)

# Step 3: Moreau 分解合成
result = v - t · p            # = prox_{t‖·‖∞}(v)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Moreau 分解是理论工具，在 MagR 代码中不单独存在，而是体现为 ℓ₁-ball 投影函数的调用。PGD 迭代中写为 `W_new = V - t * l1_ball_projection(V/t)`，这行代码隐式包含了 Moreau 分解。Moreau 分解不仅适用于 ℓ∞/ℓ₁ 对偶对，也适用于任何范数与其对偶范数之间（如 ℓ₂ 是对偶范数的自身），是 proximal algorithms 工具箱中的基础工具[32]。PyProximal 库（https://pyproximal.readthedocs.io）中的 `L1.prox` 和 `LInfinity.prox` 彼此通过 Moreau 分解互推。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---

## Quantization Step Scaling (β Factor)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Step Scaling (β factor / 量化步长缩放因子) 是在计算 uniform quantization 步长时引入的乘法缩放因子 β ≤ 1：δ = β · (max(w) − min(w))/(2^b − 1)。标准 uniform quantizer 使用 β=1，步长由权重最小/最大值决定。但在极低位宽（2-bit、3-bit）下，标准步长导致量化网格过宽，大量权重落于网格边界，量化误差大。通过 β < 1 收缩步长，量化网格更紧密地聚集在权重分布中心，虽然部分 outlier 被裁剪但大多数权重的量化精度提升，总体 MSE 更低。这一现象的理论基础[21,34]是：最低 MSE 的量化步长并不一定由 min/max 决定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 标准步长（β=1）
δ_std = (max(w) - min(w)) / (2^b - 1)

# β-scaled 步长
δ_magr = β · δ_std

# MagR 经验取值:
#  per-channel INT2: β ∈ [0.80, 0.85]
#  per-channel INT3: β ≈ 0.90
#  per-channel INT4: β = 1.00
#  per-group INT2/INT3: β = 0.95

# 量化过程与标准 uniform quantizer 完全相同，仅 δ 不同
w_q = δ_magr · clamp(round(w/δ_magr) - z, 0, 2^b-1) + z·δ_magr
```

β 与 bit-width 正相关：位宽越低，标准步长的缺陷越明显，需要更大幅缩放。INT4 有 16 个量化级别，标准步长通常已足够好。MagR 预处理降低权重最大幅度后，β 进一步收缩步长将量化网格对齐到高密度区域，两者协同大幅降低 sub-4bit 误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
β 通过网格搜索在 calibration 集上选择：完成 MagR 预处理 → 对 β ∈ {0.80, 0.85, 0.90, 0.95, 1.00} 分别量化 → 选最低 perplexity 对应的 β。MagR ablation study（Table 8）显示 INT2 下 β=0.80 PPL=16.73 vs β=1.00 PPL=16.99，INT3 下 β=0.90 PPL=6.41 vs β=1.00 PPL=6.43。该技术也被 TWN[21]、XNOR-Net[34] 等早期工作中观察到。β 与 MagR 是正交增强：MagR 缩小权重范围，β 进一步优化步长。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

## Selection Mechanism in SSM (Selective State Space / S6)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Selection Mechanism（选择机制）是 Mamba (Gu & Dao, 2023) 的核心创新，将 SSM 参数 (Δ, B, C) 从静态改为输入依赖，使模型沿序列维度"选择性"传播或遗忘信息。Δ_t = softplus(Parameter + s_Δ(x_t)) 控制关注当前 vs 保持历史的平衡；B_t = Linear_N(x_t) 和 C_t = Linear_N(x_t) 提供输入↔状态↔输出的细粒度内容调制。Theorem 1 证明当 N=1, A=-1, B=1 时选择机制退化为 gated RNN: g_t = σ(Linear(x_t)), h_t = (1-g_t)h_{t-1} + g_t·x_t。Ablation (Table 7): 仅 Δ 选择性 PPL 10.93→9.81，三者全开 8.71。Δ 是核心——大 Δ≈关注当前输入并重置状态，小 Δ≈忽略输入保持历史。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 选择性 SSM (S6) 前向 (单通道):
输入: x ∈ R^{B×L×D}, A ∈ R^{D×N} (diagonal, N≈16)
输出: y ∈ R^{B×L×D}

# Step 1: 计算输入依赖参数
Δ = softplus(Linear_R(x) + bias)   → (B, L, D)  # R=64, broadcast to D
B = Linear_N(x)                     → (B, L, N)  # 选择性输入投影
C = Linear_N(x)                     → (B, L, N)  # 选择性输出投影

# Step 2: ZOH 离散化 (fused in SRAM, 不物化完整张量)
Ā_t = exp(Δ_t ⊙ A)                  # element-wise, A 为 diagonal
B̄_t = Δ_t ⊙ B_t                    # 一阶 Taylor 近似

# Step 3: 选择性循环 (parallel scan, 硬件感知实现)
h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t   # time-varying! 每步参数不同

# Step 4: 输出
y_t = C_t ⊙ h_t
```
对比 LTI (S4): Ā, B̄ 对所有 t 相同 → h_t = Ā h_{t-1} + B̄ x_t → 等价于全局卷积 y = x ∗ K̄，可用 FFT。LTI 无法做内容感知推理（Selective Copying 失败），选择机制打破 LTI 约束获得内容感知但损失卷积可用性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/state-spaces/mamba。实现需增加 3 组小型投影（Δ/B/C），参数增量极小（~1%）。关键工程挑战：选择性导致 time-varying → 卷积不可用 → 必须用循环，但朴素循环需物化 (B,L,D,N) 中间状态（大 N 倍）。Mamba 通过硬件感知 fused selective scan kernel 解决（见 kernel调度 层对应条目）。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Discretization in State Space Models (Zero-Order Hold / ZOH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Discretization（离散化）将连续时间 SSM 参数 (Δ, A, B) 转为离散参数 (Ā, B̄)。连续 SSM: h'(t) = Ah(t) + Bx(t), y(t) = Ch(t)。Zero-Order Hold (ZOH)：假设每步内 x(t) 恒定：Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA)-I)·ΔB。欧拉近似：Ā = I + ΔA, B̄ = ΔB（实践中常用简化）。离散化作用：(1) 连续→离散映射；(2) 保证模型归一化；(3) 赋予分辨率不变性；(4) 在选择机制中，Δ 的输入依赖性通过离散化传递到 Ā, B̄，使递归动态整体成为内容感知。Theorem 1 揭示离散化是 RNN gating 的原则性基础：N=1, A=-1, B=1 时 ZOH 精确给出 gated RNN 形式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ZOH 离散化 (per timestep, per channel):
Ā_td = exp(Δ_td · A_d)                       # exp(标量×向量), element-wise
B̄_td = (Δ_td·A_d)^{-1}·(exp(Δ_td·A_d)-1)·Δ_td·B_t

# 实践中通常简化为:
Ā_td ≈ exp(Δ_td · A_d)                        # 精确 exp
B̄_td ≈ Δ_td · B_t                            # 一阶 Taylor: lim_{x→0} (e^x-1)/x = 1
```
Mamba 的选择性 SSM 中，离散化是关键桥梁：Δ 的输入依赖性 → Ā = exp(Δ·A) → 整个递归因 Ā 而变化。因此虽然 A 本身不是选择性的，但通过 Δ 的离散化传递，Ā 变得时间可变。Ablation 显示 A 的选择性不是必需的——Δ 的选择性足够。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 Mamba 硬件感知实现中，离散化融合进单一 CUDA kernel，在 SRAM 内完成：加载 (Δ, A, B) → SRAM 计算 Ā=exp(ΔA), B̄=ΔB → 传给 parallel scan → 输出 y。离散化不作为独立步骤物化 Ā, B̄ 到 HBM，节省 O(BLDN) 内存 IO。其他 SSM 变体 (S5, DSS) 使用不同的离散化策略（有时直接参数化 Ā, B̄ 跳过离散化步骤）。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Error Barrier (误差壁垒)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Error Barrier（误差壁垒）是 Frankle et al. (ICML 2020) 提出的概念，用于量化两个训练好的权重点之间沿线性插值路径的最大损失增量。给定两个收敛的权重点 θ₁ 和 θ₂，定义插值点 θ_λ = (1-λ)θ₁ + λθ₂ (λ∈[0,1])，Error Barrier 为：max_{λ∈[0,1]} [L(θ_λ) − ½(L(θ₁)+L(θ₂))]。Error Barrier 是模型合并可行性的核心指标：零 Error Barrier 意味着线性模式连通性（Linear Mode Connectivity），即两个模型位于同一平坦的损失盆地中，可以安全地通过简单权重平均进行合并。高 Error Barrier 则表示两模型处于不同盆地，线性插值路径会穿越高损失区域，合并质量差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 HDRQ 论文中，Error Barrier 被扩展到量化噪声场景。量化误差 ε₁, ε₂ 被建模为均匀噪声 U[-s/2, s/2]，量化后 Error Barrier 为：max_{λ∈[0,1]} [L(θ_λ+ε_λ) − ½(L(θ₁+ε₁)+L(θ₂+ε₂))]。通过二阶 Taylor 展开分解为原始 Error Barrier + 量化相关项。HDRQ 的关键推导：通过 Hessian 的 M-Lipschitz 连续性可以边界化合并点的 Hessian：|∇²L(θ_λ) − (∇²L(θ₁)+∇²L(θ₂))/2| ≤ M||θ₂−θ₁||/2。这表明降低 Error Barrier 的两个方向：(1) 控制 Hessian 谱（降低曲率敏感性）→ Hessian Regularization；(2) 降低权重间距离 ||θ₂−θ₁|| → Distance Regularization。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Error Barrier 的计算通常涉及：对 λ 在 [0,1] 区间采样多个点（如 λ=0.0, 0.1, ..., 1.0），在每个插值点评估测试集上的损失，找到最大偏差。HDRQ 将其用作理论分析工具而非训练目标：通过分析出降低 Error Barrier 的两条路径（控制 Hessian 和控制距离），指导设计对应的正则化方法。在模型合并实践中，Error Barrier 概念也指导了合并策略（如 permutation alignment via Git Re-Basin）和合并条件（从同一预训练模型微调的模型处于同一 basin，因此可通过简单平均合并）。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

## Model Merging / Weight Averaging (模型合并 / 权重平均)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Model Merging（模型合并，也称 Weight Averaging / 权重平均）是一种训练免（training-free）的多模型融合技术，通过直接对多个模型的权重进行算术平均来获得一个统一的通用模型。其理论基础是线性模式连通性（Linear Mode Connectivity）：从同一预训练初始化出发、分别在不同分布或任务上微调的模型，其权重往往位于同一平坦的损失盆地（loss basin）内，因此简单的线性插值不会导致损失剧烈上升。Li et al. (ECCV 2024) 证明了这一原理在 Multi-Target Domain Adaptation 中的有效性：对不同目标域分别做源自由域自适应（Source-Free Domain Adaptation）得到多个模型，通过 midpoint weight averaging 合并为一个统一模型，无需额外训练。合并操作分为两部分：(1) 模型参数合并 — 简单线性平均 θ_merged = (θ_1 + θ_2 + ... + θ_k) / k；(2) 归一化统计量合并 — 使用 Gaussian prior 从各模型的 BN 统计量估计合并后的 BN 均值和方差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 HDRQ 论文中的模型合并流程（语义分割，两个目标域）为例：
```
# 步骤 1: 源预训练模型 → 单目标域自适应
θ_src = pretrained_ResNet101_on_ImageNet()
θ_G→C = HRDA_adapt(θ_src, target=Cityscapes)   # GTA→Cityscapes
θ_G→I = HRDA_adapt(θ_src, target=IDD)          # GTA→Indian Driving

# 步骤 2: 各域独立 HDRQ 量化
BN_fold(θ_G→C); BN_fold(θ_G→I)                # BN 折叠
θ_C_q = HDRQ_quantize(θ_G→C, bit=4)            # Hessian + Distance Reg
θ_I_q = HDRQ_quantize(θ_G→I, bit=4)

# 步骤 3: Noise-Sampling Rounding + Merging
for k in range(30):                              # 30 次噪声采样
    ε_C, ε_I ~ U[-Δ/2, Δ/2]
    w_merged_k = midpoint_avg_noisy(θ_C_q, θ_I_q, ε_C, ε_I)
    score_k = cosine_similarity(w_merged_k - θ_src, θ_C_q - θ_I_q)
w_merged = argmax_k(score_k)                    # 选择最优样本

# 步骤 4: 多目标域推理
y_C = merged_model(x_C)                          # Cityscapes 上推理
y_I = merged_model(x_I)                          # IDD 上推理
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
模型合并的实现非常简单：(1) 使用 PyTorch 的 state_dict() 获取各模型参数字典；(2) 对各层参数执行逐元素的加权平均；(3) 对 BN 统计量特殊处理（Li et al. 用 Gaussian prior 建模，HDRQ 通过预先 BN folding 规避此问题）。合并的关键前提条件是模型从同一预训练 checkpoint 出发且在各自微调中保持相同架构。HDRQ 揭示了 quantized 模型合并的特有问题：量化引入的离散化噪声破坏权重对齐，增大 Error Barrier。其解决方案（Hessian Regularization + Distance Regularization）在量化阶段即保证合并兼容性，使合并后精度损失从 baseline QDrop 的约 4 mIoU 降至近零。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

## Noise-based Quantization (噪声量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise-based Quantization（噪声量化）是一种将量化误差建模为随机噪声的量化模拟方法，训练时向权重添加采样噪声（w+ε）替代确定性量化值（ŵ），在反向传播中通过 STE 或直接对噪声更新梯度。其数学模型：给定权重 w 和量化步长 Δ，量化误差 ε = w − clamp(⌊w/Δ⌉, −2^{b-1}, 2^{b-1}−1)·Δ 服从 U[-Δ/2, Δ/2] 均匀分布。使用 w+ε 的前向传播，损失函数的期望值为：E[L(w+ε)] ≈ E[L(w) + ε·∇_w L(w) + ½·εᵀ·∇²_w L(w)·ε] ≈ E[L(w) + ½·εᵀ·∇²_w L(w)·ε]，因为 E[ε]=0 消除一阶项。结果：损失隐式惩罚 ∇²_w L(w)（Hessian 迹），驱动权重收敛到更平坦的损失区域。该技术最早由 NICE (Baskin et al. 2021) 和 DiffQ (Défossez et al. 2022) 提出，Bit-Shrinking (Lin et al. 2023) 将其引入 PTQ 结合 sharpness-aware scheduling。HDRQ 是首个将噪声量化引入模型合并框架的方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HDRQ 中噪声量化的 block-wise reconstruction 过程：
```
for iter in range(1, 20001):
    # 1. 确定性量化
    w_hat = clamp(round(w/Δ), -2^{b-1}, 2^{b-1}-1) * Δ
    
    # 2. 噪声采样（前 16500 iter 使用噪声模拟）
    if iter <= 16500:
        ε = w - w_hat                     # U[-Δ/2, Δ/2]
        w_train = w + ε                   # 噪声版本权重
    else:
        w_train = w_hat                   # 切换 fake quantization
    
    # 3. 前向 + 损失
    O_hat = block_forward(x, w_train)
    L_rec = ||O_hat - O_fp||₂²            # 重建损失
    L_dist = λ * ||w_src - w_train||₂²    # 距离正则 (λ=5e-2)
    L = L_rec + L_dist
    
    # 4. 更新权重 w
    optimizer.step()                       # Adam, LR=0.001, cos annealing
```
关键设计：(1) 噪声量化在绝大多数迭代（16500/20000）中生效，确保充分的 Hessian 正则化；(2) 最后 3500 迭代切换到确定性 fake quantization，此时学习率已很小，不会破坏正则化效果；(3) 配合距离正则化项（w_train 到源权重的 ℓ₂ 距离）同时控制权重 divergence。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
噪声量化在 PyTorch 中通过自定义 autograd function 或 fake_quantize 模块实现。典型实现：(1) 在 forward 中计算 deterministic quantized value ŵ；(2) 计算 ε = w − ŵ；(3) 使用 straight-through estimator (STE) 在 backward 中将梯度从 w+ε 回传到 w。HDRQ 的噪声量化与标准噪声量化方法的区别在于额外增加了距离正则项和切换到 fake quantization 的阶段性策略。噪声量化的优势：不需要 STE 近似梯度（如果噪声是加性的而非乘性的）、天然支持平坦极小值搜索、理论上有 Hessian 平滑保证。局限性：(1) 训练过程中的噪声方差影响收敛速度；(2) 噪声模拟和真实量化之间存在 gap（通过最后切换到 fake quantization 缓解）。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

## Multi-Target Domain Adaptation / MTDA (多目标域自适应)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Target Domain Adaptation（MTDA，多目标域自适应）是域自适应（Domain Adaptation）的一个子问题：给定源域上预训练的模型，将其同时适配到多个不同的目标域（而非单一目标域）。与 Single-Target Domain Adaptation (STDA) 对应。传统 MTDA 方法主要包括：(1) 知识蒸馏方法（MTDA-KD, Nguyen-Meidine et al. 2021）——使用多个学生模型适应不同目标域，计算开销高；(2) 信息论方法（Gholami et al. 2020）；(3) 训练免方法（Li et al. ECCV 2024）——通过模型合并实现 MTDA，无需额外训练数据或蒸馏过程。训练免 MTDA 利用了从同一源模型微调的多个域自适应模型位于同一优化盆地的事实，通过简单权重平均合并。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 HDRQ 论文中 Office-Home 数据集的 MTDA 实验为例（源域 Real，目标域 Art/Clipart/Product）：
```
# 步骤 1: 源自由域自适应 (SFDA) - 各自独立
θ₀ = ResNet50_pretrained(ImageNet)                  # 源预训练
θ_R→A = SHOT_adapt(θ₀, target=Art)                  # Real→Art
θ_R→C = SHOT_adapt(θ₀, target=Clipart)              # Real→Clipart
θ_R→P = SHOT_adapt(θ₀, target=Product)              # Real→Product

# 步骤 2: HDRQ 量化 - 各自独立
θ_R→A_q = HDRQ(θ_R→A, bit=4)
θ_R→C_q = HDRQ(θ_R→C, bit=4)
θ_R→P_q = HDRQ(θ_R→P, bit=4)

# 步骤 3: 模型合并
θ_merged = (θ_R→A_q + θ_R→C_q + θ_R→P_q) / 3       # 三域平均

# 步骤 4: 多目标域评估
acc_A = eval(θ_merged, Art)                          # 20.28%
acc_C = eval(θ_merged, Clipart)                      # 45.75%
acc_P = eval(θ_merged, Product)                      # 73.69%
harmonic_mean = 3 / (1/acc_A + 1/acc_C + 1/acc_P)   # 主指标
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MTDA 的实现依赖于：(1) 选择单目标域自适应方法（如 SHOT, HRDA, DANN）；(2) 对各目标域分别执行适配（独立完成，可并行）；(3) 通过模型合并技术融合为统一模型。评估指标为各目标域的 Harmonic Mean 而非算术平均，因为 Harmonic Mean 对最差域更敏感，鼓励在所有域上的均衡性能。HDRQ 的关键贡献是在步骤 (2) 的量化中保持步骤 (3) 的合并兼容性，使 quantized MTDA 首次达到实用水平。HDRQ 不假设对目标域有任何先验知识，仅要求量化阶段可以访问源模型权重（通常成立，因为源模型先于部署部署在中心服务器上）。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

## Noise-Sampling-Based Rounding (基于噪声采样的舍入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise-Sampling-Based Rounding 是 HDRQ 提出的解决模型合并中舍入歧义（Rounding Ambiguity）的技术。问题源于量化模型合并时的整数域歧义：当两个量化值的整数表示 I₁、I₂ 之和为奇数时，midpoint averaging 的合并结果落在两相邻整数中间，导致舍入方向不确定。在浮点域合并公式 I_merged = ⌊(I₁·Δ₁ + I₂·Δ₂)/(Δ₁+Δ₂)⌉ 中，当 Δ₁≈Δ₂（域自适应场景的典型情况，因共享源模型且学习率小），步长项被约去，公式退化为 ⌊(I₁+I₂)/2⌉，又回到歧义问题。HDRQ 的解决方案：合并前对权重添加采样噪声 ε₁, ε₂ ∼ U[-Δ/2, Δ/2]，计算 I_merged = ⌊(I₁·Δ₁+ε₁ + I₂·Δ₂+ε₂)/(Δ₁+Δ₂)⌉。采样多组噪声（如 30 组），计算各组合并权重到目标域权重的向量与原始插值向量的 cosine similarity，选择最高相似度的样本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: 两个量化模型的权重 w1_q = I1 * Δ1, w2_q = I2 * Δ2
#       噪声样本数 K=30

best_score = -inf
best_w_merged = None

for k in range(K):
    # 1. 对每个量化权重添加采样噪声
    ε1 ~ U[-Δ1/2, Δ1/2]
    ε2 ~ U[-Δ2/2, Δ2/2]
    
    # 2. 噪声辅助的整数舍入
    I_merged = round((I1 * Δ1 + ε1 + I2 * Δ2 + ε2) / (Δ1 + Δ2))
    w_merged_k = I_merged * ((Δ1 + Δ2) / 2)     # 反量化
    
    # 3. Cosine similarity 质量评估
    v_merged = flatten(w_merged_k - w_src)       # 合并后变化方向
    v_interp = flatten(w1_q - w2_q)              # 原始插值方向
    score = dot(v_merged, v_interp) / (||v_merged|| * ||v_interp||)
    
    # 4. 选最优
    if score > best_score:
        best_score = score
        best_w_merged = w_merged_k

return best_w_merged
```
当 Δ₁≈Δ₂ 时，噪声 ε₁, ε₂ 打破确定性舍入的对称性，提供多个候选舍入方向。Cosine similarity 筛选确保合并方向与原始无噪声插值方向保持一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该技术在 HDRQ 中仅用于合并阶段（推理前的一次性离线操作），不增加推理时计算开销。采样 30 组噪声的额外计算可忽略（仅需每层权重做一次加法+取整+sanity check）。噪声采样舍入的有效性在实验中被证实（Figure 3）：对比无 filter（随机采样）、不使用 cosine similarity filter 以及 HDRQ 的 Advanced（cosine similarity filter），Advanced 方法显著稳定合并质量，filter 掉低质量噪声样本。三目标域合并场景（Office-Home 三域）由于不再存在奇数歧义（三数平均天然落在连续区域），无需噪声采样。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation

## Linear Time Invariance (LTI) in Sequence Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear Time Invariance (LTI，线性时不变性) 是序列模型动态参数在所有时间步上保持不变的属性。在 SSM 语境：Ā, B̄, C 对所有 t 相同 → 递归 h_t = Ā h_{t-1} + B̄ x_t 等价于全局卷积 y = x ∗ K̄，可用 FFT O(L log L) 训练。所有先前的结构化 SSM（S4, H3, Hyena, RWKV, RetNet）均为 LTI。优点：计算效率高、可并行训练。核心缺陷：无法做内容感知推理——动态对所有 token 相同，不能根据 token 内容决定选择/过滤哪些信息。在 Selective Copying（token 间距随机）和 Induction Heads（上下文关联召回）任务上暴露为致命弱点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LTI SSM (S4): Ā, B̄, C 对所有 t 相同
# 卷积模式 (训练): 所有 timestep 共享相同卷积核
K̄ = [CB̄, CĀB̄, CĀ²B̄, ..., CĀ^{L-1}B̄]  # 固定卷积核
y = x ∗ K̄                                    # FFT 加速, O(L log L)

# 循环模式 (推理): Ā 不依赖 x_t
h_t = Ā ⊙ h_{t-1} + B̄ ⊙ x_t                 # 每步相同动态!
```
LTI 无法解决 Selective Copying：不同位置需记忆/忽略的 token 间距随机 → 但卷积核固定长度 → 无法适配 → 只能依赖 time-awareness 而非 content-awareness。Mamba 的选择机制打破 LTI：Ā_t = f(x_t) → 获得内容感知 → 但损失卷积可用性 → 需要硬件感知实现补偿。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LTI SSM 的实现族：S4 (DPLR + FFT 卷积), S4D (纯对角 + FFT), H3 (S4 嵌入 gated architecture), Hyena (MLP-parameterized 全局卷积替换 S4)。在 GPU 上通过 PyTorch FFT primitives 实现。Mamba 证明了 LTI 是内容感知的主要障碍，打破它并补偿效率损失可以匹配 Transformer 性能同时保持线性复杂度。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Microscaling (MX) Data Formats

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Microscaling (MX) 是由 Open Compute Project (OCP) 在 2023 年发布的一种面向深度学习的块级缩放（block-scaled）数值格式规范（OCP Microscaling Formats V1.0 Specification）。MX 的基本单元是一个大小为 k 的块（默认 k=32），包含 k 个低精度标量元素和一个共享的 8-bit 缩放因子（E8M0 格式）。E8M0 是纯指数格式（8-bit 指数，0-bit 尾数），只能表示 2 的幂次，这使得反量化操作仅需移位运算而非乘法。MX 格式族包括 MXFP8（E4M3 或 E5M2，8-bit 元素）、MXFP6（E3M2 或 E2M3，6-bit 元素）、MXFP4（E2M1，4-bit 元素）和 MXINT8。与传统的 per-tensor 或 per-channel 量化不同，MX 在每个 32 元素块内共享一个 scale，实现更细粒度的量化误差控制。NVIDIA Blackwell GPU 原生支持 MXFP8 和 MXFP4 的 MMA 指令，AMD 和 Intel 也提供软件支持。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
给定一个 FP16 张量 X ∈ R^{L×I}，MXFP 量化流程：
```
# 1. 分块：将 X reshape 为 (L*I/32, 32) 的 block 序列
blocks = X.reshape(-1, 32)
# 2. 逐块量化
for block in blocks:
    s = 2^{floor(log2(max(abs(block)))) - b}  # E8M0 scale, b为格式特定偏置
    Q(block) = round(clip(block/s, -q_max, q_max))
    # 存储：Q(block) 的 32 个元素 + 1 个 E8M0 scale s
```
其中各格式参数：MXFP4 (E2M1): b=1, q_max=6；MXFP6 (E3M2): b=3, q_max=28；MXFP8 (E4M3): b=7, q_max=448。反量化：x = s × Q(x)，s 为 2 的幂次，仅需移位。在 MicroMix 中，同一层的不同通道组 G4/G6/G8 分别应用 MXFP4/MXFP6/MXFP8 的 block-wise 量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MX 格式被 PyTorch (torchao.MXTensor)、NVIDIA Transformer Engine (v2.14+)、AMD Quark、Intel Neural Compressor、OpenVINO 等框架支持。硬件层面，NVIDIA Blackwell (SM 10.0+) 的第五代 Tensor Core 通过 `am16n8k64` 指令原生支持 MXFP4/MXFP8 的 MMA，block scale 反量化融合在 MMA 指令内部。MX 适用于 LLM 推理量化（如 MicroMix 的 ~5.5-bit 混合精度）、训练中的低精度前向（如 DeepSeek V3 使用 UE8M0 FP8-scaled）、以及需要细粒度量化控制的场景。

涉及论文标题：
- MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

---

## Mixed-Precision Quantization for MoE（MoE 模型的混合精度量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixed-Precision Quantization for MoE 是针对 Mixture-of-Experts 模型设计的量化策略，为 MoE block 内不同的 linear block（gate_proj, up_proj, down_proj）分配不同的量化精度（位宽），而非使用统一精度。MxMoE 的核心洞察：MoE block 内存在两个维度的异构性：(1) 量化敏感度异构——同一 expert 内不同 linear block（如 gate_proj vs down_proj）对量化位宽的敏感度差异显著；(2) 计算特性异构——不同 expert 的激活频率差异超过 10×，导致部分 expert 的 GEMM 为 memory-bound（低频率），部分为 compute-bound（高频率）。混合精度策略根据敏感度给敏感 block 分配更高精度、给不敏感 block 分配更低精度；同时根据硬件特性给 memory-bound GEMM 用 weight-only 量化（如 W4A16），给 compute-bound GEMM 用 weight-activation 量化（如 W4A4 或 W8A8）。与 expert 级混合精度（MC-MoE 等）不同，MxMoE 在更细的 linear-block 粒度分配位宽，实验证明 linear-block 粒度一致优于 expert 级。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MxMoE 混合精度量化流程（以 W5A5 目标平均位宽为例）：

```
输入: MoE block (E experts, 每个 3 linear blocks: gate/up/down)
      校准数据 X_cal (128 seqs × 4096 tokens)

1. 逐 linear-block 量化敏感度评估:
   for expert i=1..E, block j in {gate, up, down}:
       for scheme k in {W4A4, W4A4-g128, W8A8, ...}:
           W_q = GPTQ(W_{i,j}, scheme k)
           Ô = MoE_forward(W在第(i,j)位量化为W_q, 其余FP16)
           Δ_{i,j,k} = ||Ô - Ô_FP16||₂

2. ILP 求解最优分配:
   minimize L^r · T^{1-r}
   s.t. 每 block 选一方案, 总内存 ≤ M
   
   输出: x_{i,j,k} (每个 linear block 的量化方案)

3. 按分配方案量化:
   for each linear-block (i,j) with assigned scheme k:
       W_{i,j} → GPTQ_quant(W_{i,j}, scheme k)
       激活运行时按 scheme k 动态量化

示例分配 (Qwen1.5-MoE layer 5, W5A5):
  Expert 0: gate=W4A4-g128, up=W4A4-g128, down=W4A4-g128
  Expert 1: gate=W4A4-g128, up=W4A4-g128, down=W8A8 (敏感 down_proj 获更高精度)
  Expert 22: gate=W8A8, up=W8A8, down=W8A8 (高频 expert 全 8-bit)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖：(1) 量化敏感度通过 Euclidean distance 在校准集上测量，每个 linear block 依次量化评估；(2) ILP solver（如 Gurobi）离线求解最优方案；(3) 量化使用 GPTQ + randomized Hadamard 变换提高精度；(4) 系统层面需要支持混合精度的 Group-GEMM kernel 来实际加速。超参数 r 在 weight-only 极低比特下设为 r=1（精度优先），weight-activation 下设 r=0.75（平衡）。MxMoE 在 2.25-bit weight-only 下 WikiText2 PPL 比 GPTQ 低 2.4（DeepSeekV2-Lite），W5A5 比 FP16 快 3.4×、比 uniform W8A8 快 29.4%。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---

## Linear-Block Level Quantization Granularity（Linear-Block 级量化粒度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear-Block Level Quantization Granularity 是 MxMoE 提出的在 MoE 模型中分配量化位宽的最小单元：不是整个 expert，也不是整个模型层，而是 expert 内部的每个线性投影块（gate_proj, up_proj, down_proj）。传统 MoE 量化工作（如 MC-MoE）以 expert 为粒度分配位宽——同一 expert 的所有 linear block 使用相同精度。但 MxMoE 通过量化敏感度分析（Fig. 1a）发现：同一 expert 内 gate_proj 和 down_proj 对量化的敏感度可能差异很大，统一位宽要么对不敏感 block 浪费精度预算，要么对敏感 block 精度不足。Linear-block 粒度允许更细粒度的精度分配——在同一 expert 内，不敏感的 gate_proj 用 W4A4，敏感的 down_proj 用 W8A8。实验表明 linear-block 粒度一致优于 expert 级（Table 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化粒度对比：

```
Expert-Level Granularity (例如 MC-MoE):
  Expert i: 所有 gate/up/down 共享同一量化方案
  例如 Expert 40: W4A4 应用于 gate+up+down

Linear-Block-Level Granularity (MxMoE):
  Expert i:
    gate_proj → 独立选择量化方案
    up_proj   → 独立选择量化方案
    down_proj → 独立选择量化方案
  例如 Expert 40:
    gate_proj → W4A4 (不敏感)
    up_proj   → W4A4 (不敏感)
    down_proj → W8A8 (敏感，需更高精度)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 MxMoE ILP 中，变量 x_{i,j,k} 的索引 j ∈ {1,2,3} 对应 3 个 linear block，每个独立求解。更细的粒度带来更多变量（E×3×|S| vs E×|S|），但 ILP 仍在可解范围内。与更细的 channel-level 或 element-level 粒度（如 SqueezeLLM）相比，linear-block 粒度避免了 irregular memory access 和 bitwidth lookup 的额外计算开销，保持良好的硬件效率。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---

## Quantization Sensitivity / Perturbation Coefficient（量化敏感度 / 扰动系数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Sensitivity（量化敏感度）在 MxMoE 中以 Perturbation Coefficient Δ_{i,j,k} 量化——对 MoE block 中第 i 个 expert 的第 j 个 linear block，使用量化方案 k 量化后，MoE block 输出与全精度输出的 Euclidean distance：Δ = ||Ô - O||₂。该度量直接反映了该 linear block + 该量化方案的组合对最终输出的扰动程度。Δ 越大，说明该 linear block 对该量化方案越敏感，需要更高精度。在校准集（128 条 WikiText2 序列）上统计 Δ 值。该 metric 假设量化输出扰动与最终 loss 扰动正相关（Choukroun et al. 2019），因此最小化中间输出扰动可有效保持最终模型精度。与 Hessian-based 方法（如 HAWQ）相比，基于输出距离的度量更简单高效，不需要计算 Hessian 矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
计算 Δ_{i,j,k} (Pseudocode):
  输入: MoE block (FP16), X_cal calibration data
  for each linear-block (i,j) in MoE block:
      for each scheme k in S:
          # 保存原始权重
          W_orig = W_{i,j}
          # 临时量化该 linear block
          W_q = GPTQ_quantize(W_{i,j}, scheme k)
          # 计算 MoE block 输出
          Ô = MoE_block_forward(X_cal,
              weights: replace W_{i,j} with W_q, others FP16)
          O = MoE_block_forward(X_cal, all FP16)
          # 扰动 = 输出欧氏距离
          Δ_{i,j,k} = ||Ô - O||₂
          # 恢复权重
          W_{i,j} = W_orig
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Δ 值在 ILP 问题中作为目标 L = Σ Δ_{i,j,k}·x_{i,j,k} 的系数，驱动 solver 将更低的 Δ 分配给更敏感的 block。校准数据量影响 Δ 估计的准确性——128 条 sequence × 4096 tokens 在实践中平衡了准确性和校准开销。MxMoE 指出可能存在跨层依赖导致的敏感度估计偏差（如 Qwen2-MoE 在 3.25-bit 下略逊于 GPTQ），建议使用跨层 loss 而非单层 loss 来改进。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---

## Mixture-of-Experts (MoE) in Large Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture-of-Experts (MoE) 是一种神经网络架构模式，将传统 Transformer 的 FFN（Feed-Forward Network）层替换为多个并行的"专家"子网络（expert FFN），每个 token 仅激活其中 top-k 个专家（稀疏激活）。以 Mixtral 8x7B 为例，每层包含 n=8 个专家，top-k=2，总参数量 47B，但每个 token 仅激活约 13B 参数（2 个专家 × 每专家 FFN + shared attention）。MoE 的核心思想是"条件计算"（conditional computation）：不同 token 由不同专家处理，实现参数量的扩展而计算量保持可控。MoE 层的前向计算：输入 token x → Router 计算路由 logits l = {l_0,...,l_{n-1}} → Softmax 得路由权重 w → 取 top-k 专家的索引 e_j 和权重 → 加权聚合输出 z = Σ_{j=0}^{k-1} w̃_{e_j}·E_{e_j}(x)，其中 w̃_{e_j} = w_{e_j}/Σ_{m=0}^{k-1} w_{e_m} 为归一化路由权重，E_i 为第 i 个专家的 SwiGLU FFN 变换。专家参数通常占 MoE 模型总参数的绝大多数（Mixtral 8x7B 中 8 个专家占 96%），这是 MoE 部署内存的主要瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE 层推理 pipeline（per token, Mixtral 8x7B top-2 为例）：
```
输入: x ∈ R^{d}  (hidden state of a single token)
参数: Router W_r ∈ R^{d×n}, Expert FFN params {W_{gate}^i, W_{up}^i, W_{down}^i}_{i=0}^{n-1}

# Step 1: Routing
l = W_r^T x                         # n 维 routing logits
w = Softmax(l)                       # n 维 routing weights
e0, e1 = TopK(w, k=2)               # 取 top-2 的专家索引

# Step 2: Expert computation (parallel)
w̃[e0] = w[e0] / (w[e0] + w[e1])     # 归一化 routing weight
w̃[e1] = w[e1] / (w[e0] + w[e1])
y0 = SwiGLU(x; W_{gate}^{e0}, W_{up}^{e0}, W_{down}^{e0})
y1 = SwiGLU(x; W_{gate}^{e1}, W_{up}^{e1}, W_{down}^{e1})

# Step 3: Weighted aggregation
z = w̃[e0] · y0 + w̃[e1] · y1          # MoE layer output
```
SwiGLU FFN 计算：`E_i(x) = W_{down}^i · (SiLU(W_{gate}^i x) ⊙ (W_{up}^i x))`，其中 SiLU(x)=x·σ(x)。注意力层在所有 token 间共享，不含专家。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源 MoE LLM 实现：Mixtral 8x7B/8x22B（Mistral AI），DeepSeek-V2/V3（DeepSeek），Qwen1.5-MoE/Qwen3-MoE（Alibaba），Switch Transformers（Google）。HuggingFace Transformers 中使用 `MixtralForCausalLM` 加载 MoE 模型，配置 `num_local_experts` 和 `num_experts_per_tok`。MoE 的部署挑战：专家参数量巨大（Mixtral 8x7B bf16 需 2×A100-80G），静态参数内存远超同等活跃参数量的 dense 模型。路由坍塌（routing collapse）是已知问题：部分专家被频繁激活而其余专家不被使用，导致训练不均衡和参数浪费。

涉及论文标题：
- MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

---

## Post-Training Expert Pruning for MoE LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-Training Expert Pruning 是一种针对 MoE LLM 的后训练压缩技术，通过逐层移除不重要的专家来减少模型参数量和部署内存。与传统 weight pruning（Wanda, SparseGPT）不同，Expert Pruning 将稀疏化粒度从"权重元素"提升到"专家"级别：每层从 n 个专家中保留 r 个最重要的，永久丢弃其余 n−r 个专家及其路由权重。核心方法：使用小规模校准数据集（如 C4 128 条 × 2048 tokens），对每层枚举所有 C(n,r) 种专家组合，以最小化 Frobenius 范数重构损失 ‖F'(x,C) − F(x)‖_F 为目标选择最优子集。其数学表达为：min_C ‖F'(x,C) − F(x)‖_F, s.t. C ⊆ {expert_0,...,expert_{n-1}}, |C|=r。由于每层专家数较小（Mixtral 8x7B 的 n=8），枚举组合数 C(8,4)=70 / C(8,6)=28 完全可行。剪枝后模型可通过修改 config 中的 expert 数量直接使用 HuggingFace Transformers 加载，无需修改模型代码。Mixtral 8x7B 剪枝耗时：r=6 约 30 分钟，r=4 约 90 分钟。支持通用剪枝（C4 校准）和领域特定剪枝（将校准数据切换到目标领域数据集如 MATH，提升数学任务剪枝效果——GSM8K 5-shot r=6: 41.02 vs 51.25）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Pruning pipeline：
```
# Step 1: 校准数据前向传播
for layer l in 1..L:
    for batch in calib_data:             # 128 seqs × 2048 tokens
        x[l], F_l(x) = forward_cache(l)  # 缓存每层输入-输出对

# Step 2: 逐层枚举剪枝
for layer l in 1..L:
    best_loss = inf
    for C in Combinations({expert_0,...,expert_{n-1}}, r):
        # 构造剪枝后 MoE 层（丢弃 n-r 个专家及路由权重）
        F'(x, C) = Σ_{j=0}^{r-1} w̃_{e_j}·E_{e_j}(x)
        w̃_{e_j} = w_{e_j} / Σ_{m=0}^{r-1} w_{e_m}
        loss = ‖F'(x[l], C) − F_l(x[l])‖_F    # Frobenius 重建损失
        if loss < best_loss:
            best_experts[l] = C

# Step 3: 修改模型配置，仅加载保留的专家
config.num_local_experts = r
# 剪枝后 checkpoint 中仅包含 r 组 expert 权重
```
与 weight pruning 的本质区别：Weight pruning 产生稀疏矩阵需专用硬件加速；Expert pruning 直接减少模型层中的子网络数量，在标准 GPU 上即插即用。剪枝 r=6 时 Mixtral 8x7B 内存从 89,926MB 降至 68,383MB（24% 减少），单张 80G GPU 可部署；r=4 时降至 46,879MB（48% 减少）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/Lucky-Lance/Expert_Sparsity。基于 HuggingFace Transformers + PyTorch 实现。使用场景：(1) 通用部署压缩——C4 校准，适应广泛任务；(2) 领域特定压缩——用目标领域数据（如 MATH）校准，提升数学/代码等专业任务剪枝效果。限制：(1) 枚举法复杂度 O(C(n,r)·L)，专家数 n≫8 时不可行（如 32 专家时 C(32,16)≈6×10^8）；(2) 剪枝后通常需微调恢复性能（论文用 MetaMathQA 微调 900 步恢复数学能力）；(3) 仅验证 Mixtral 8x7B 架构，未测试其他 MoE 变体（DeepSeekMoE shared experts 等）。

涉及论文标题：
- MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

---

## Dynamic Expert Skipping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Expert Skipping 是一种在线推理加速技术，在 MoE LLM 的每个 MoE 层根据路由权重动态决定是否跳过次优专家（second-best expert），减少每个 token 实际执行的专家 FFN 数量。与 Expert Pruning 不同，Skipping 不删除任何专家参数，而是在推理时对每个 token 做在线决策：若 top-2 路由权重比 w_{e1}/w_{e0} < β（逐层阈值），则仅执行 top-1 专家计算，跳过 e1 的 FFN。阈值 β 通过校准数据确定：对每层收集所有 token 的 w_{e1}/w_{e0} 比值，取中位数（使跳过概率约 50%）。其理论基础（Appendix A.2）：在 top-k 设置下，动态跳过 i 个专家后的重构损失上界为 L ≤ (Σ_{m=i+1}^k w_m / Σ_{m=1}^k w_m)·D，其中 D 为不同专家输出的期望差异。在 top-2 特例下，跳过条件简化为 w_2 ≤ β·w_1，β=H/(D−H)，H 为允许的重构损失上限。Dynamic Skipping 与 Expert Pruning 正交——可同时使用：剪枝减少内存，跳过减少计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dynamic Expert Skipping pipeline：
```
# === 离线校准: 确定每层 β ===
for layer l in 1..L:
    ratios = []
    for token in calib_data:
        w = Softmax(Router(token))
        e0, e1 = Top2(w)
        ratios.append(w[e1] / w[e0])
    β[l] = median(ratios)         # 中位数 → 跳过概率≈50%

# === 在线推理: per-token 动态跳过 ===
for each token x in autoregressive generation:
    for layer l in 1..L:
        w = Softmax(Router(x))   # n 维路由权重
        e0, e1 = Top2(w)         # 取 top-2
        
        if w[e1] < β[l] * w[e0]:
            # 次优专家贡献小 → 跳过
            y = E_{e0}(x)         # 仅 top-1 专家
        else:
            # 两个专家都执行
            w̃[e0] = w[e0]/(w[e0]+w[e1])
            w̃[e1] = w[e1]/(w[e0]+w[e1])
            y = w̃[e0]·E_{e0}(x) + w̃[e1]·E_{e1}(x)
```
Mixtral 8x7B C4 经验 β 值（32层）：0.402, 0.494, 0.463, 0.484, 0.478, 0.491, 0.523, 0.521, 0.544, 0.570, 0.574, 0.489, 0.503, 0.618, 0.568, 0.535, 0.559, 0.519, 0.537, 0.487, 0.469, 0.461, 0.461, 0.469, 0.458, 0.418, 0.433, 0.418, 0.406, 0.433, 0.447, 0.535。层间差异显著（0.402-0.618），验证逐层独立 β 的必要性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于 HuggingFace Transformers 修改 MoE layer 的 forward 函数，核心改动 < 20 行。组合使用效果：r=6 剪枝 + 动态跳过 (Mixtral 8x7B Instruct) → LM-eval 66.04, speedup 1.27×；比 r=4 纯剪枝 (63.88) 精度更高且加速相当。MATH 校准 β 值：0.503-0.346（数值更小，跳过更保守），因数学任务对精度要求更高。局限性：跳过率约 50% 意味着平均每个 token 激活 1.5 个专家，加速上限约 1.33×（vs top-2 的 2 个专家），无法达到更大的加速比。

涉及论文标题：
- MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

---

## Top-k Expert Routing (Gating Network) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-k Expert Routing（也称 Gating Network 或 Router）是 MoE 架构中决定每个 token 由哪些专家处理的机制。Router 是一个小型线性层 W_r ∈ R^{d×n}，输入 hidden state x，输出 n 维路由 logits l = W_r^T x，通过 Softmax 转换为路由权重 w = Softmax(l)。然后取 w 中最大的 k 个值对应的专家索引作为该 token 的激活专家。Router 的参数占总参数比例极小（Mixtral 8x7B 中 Router 参数约 d×n ≈ 4096×8 = 32K per layer × 32 层 ≈ 1M，vs 总参数 47B）。路由权重的两个用途：(1) 选择 top-k 专家——决定哪些专家执行计算；(2) 加权聚合——归一化后的 w̃_{e_j} 作为各专家输出的混合权重。路由机制在训练和推理中均保持稀疏激活（仅 k 个专家计算），这是 MoE 高效性的核心——参数量大但计算量小。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Router 计算流程（per token）：
```
输入: x ∈ R^d, Router权重 W_r ∈ R^{d×n}, 专家数 n, top-k

l = W_r^T x                          # (n,) routing logits
w = Softmax(l)                       # (n,) routing weights, Σw_i = 1
indices = TopKIndices(w, k)          # 取 top-k 专家索引
weights = w[indices]                 # 对应路由权重
w̃ = weights / sum(weights)           # 归一化: Σw̃ = 1

# 门控输出: (indices, w̃) → 传给 Expert FFNs
```
在 Mixtral 8x7B 中 n=8, k=2。路由分布分析（论文 Fig.5）：不同数据集（C4 vs MATH）下路由偏好差异显著，同一数据集内也存在层间差异。路由坍塌是主要风险——部分专家路由权重始终为 0。缓解方法：负载均衡损失（辅助损失）、expert capacity 限制、noisy top-k gating（加噪声后取 top-k）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 中实现为 `MixtralSparseMoeBlock` 内的 `nn.Linear(hidden_dim, num_experts)`。Router 在 calibration 阶段用于生成校准数据分布分析路由偏好（如专家激活频率统计）。Router 权重在 expert pruning 中也被丢弃：剪枝后仅保留选中专家的对应路由权重通道（而非常见做法——剪枝后重新归一化路由）。Dynamic Skipping 利用路由权重比 w_{e1}/w_{e0} 做在线跳过决策，不需要修改 Router 本身。

涉及论文标题：
- MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

## Shared FFN (Cross-Layer Feed-Forward Network Sharing)

术语解释
在 Transformer 架构中，让所有（或部分）Transformer block 共享同一套 FFN（Feed-Forward Network）参数，而非每层拥有独立 FFN。FFN 通常包含 gate projection（W_gate）、up projection（W_up）和 down projection（W_down）三组权重，占 Transformer 总参数的约 65%。共享 FFN 的核心思想是利用 FFN 层间的参数冗余，将 22 份独立 FFN 缩减为 1 份共享 FFN。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在标准 Transformer decoder（如 Llama 架构）中，每层包含 MHA（Multi-Head Attention）+ MLP/FFN 两个子模块，每个子模块有独立的参数。FFN 通常采用 SwiGLU 结构：`FFN(x) = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))`。这三个权重矩阵（W_gate, W_up, W_down）在每层都有独立副本。在 large-base（1.2B 参数）中，FFN 占总参数的 65%（attention 30%、heads 5%）。

MobiLlama 的 Shared FFN 设计：**所有 22 层 Transformer block 共用同一套 W_gate、W_up、W_down**，而 attention 层的 Q/K/V/O projection 每层独立。共享 FFN 将总参数从 1.2B（large-base）降至 0.5B（减少约 60%），同时保持 22 层的深度和 hidden 2048 的宽度。关键设计理念是"从大开始再缩小"：先设计高容量架构（large-base: 22 层 + hidden 2048），再通过参数共享机制降低参数量，而非从一开始就在层数或宽度上妥协。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
标准独立 FFN vs 共享 FFN 的计算对比：

```
# === 标准每层独立 FFN (如 Llama) ===
for layer in 1..L:
    h = h + MHA[layer](RMSNorm(h))          # 每层独立的 attention 参数
    h = h + FFN[layer](RMSNorm(h))          # 每层独立的 FFN 参数
    # FFN[layer] 含 W_gate^l, W_up^l, W_down^l，每层独立
    # 总 FFN 参数 = L × 3 × d_model × d_intermediate

# === 共享 FFN (MobiLlama) ===
shared_FFN = SwiGLU_FFN(W_gate, W_up, W_down)   # 仅 1 份 FFN 参数
for layer in 1..L:
    h = h + MHA[layer](RMSNorm(h))          # 每层独立的 attention 参数（保留）
    h = h + shared_FFN(RMSNorm(h))          # 所有层共享同一 FFN
    # 总 FFN 参数 = 1 × 3 × d_model × d_intermediate
    # 参数节省: (L-1) × 3 × d_model × d_intermediate

# === SwiGLU FFN 内部计算 ===
def SwiGLU_FFN(x, W_gate, W_up, W_down):
    gate = SiLU(x @ W_gate.T)       # SiLU(x) = x * sigmoid(x)
    up = x @ W_up.T
    return (gate * up) @ W_down.T   # element-wise product then down-project
```

在 MobiLlama 0.5B 中，每层 attention 仍有独立参数（32 heads, Q/K/V/O 各投影），仅 FFN 共享。0.8B 版本通过 widening 共享 FFN（增加 hidden dim 和 intermediate size）在不增加层数的前提下扩大容量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. **代码层面**：在模型定义中仅实例化一个 SwiGLU FFN 模块，所有 decoder layer 引用同一模块对象。PyTorch 中为 `self.shared_ffn = SwiGLU_FFN(config)`，每层 forward 中调用 `self.shared_ffn(x)`。
2. **训练**：共享参数的梯度从所有层汇聚到同一份参数上，AdamW 优化。学习率与独立 FFN 训练相同。
3. **适用场景**：资源受限设备上的 SLM（0.5B-1B 级别），需在参数预算内最大化模型容量。
4. **与 Basis Sharing 的区别**：Basis Sharing（SVD-based）通过低秩分解共享权重的"基向量"，每层有不同组合系数。Shared FFN 更激进——完全共用相同参数，无层间差异。
5. **局限性**：所有层共享同一 FFN 意味着 FFN 无法学习层特定的特征变换，某些需要层特异性 FFN 的任务可能退化。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices

## Small Language Model (SLM)

术语解释
参数量在 0.1B-2B 范围内的小型语言模型，专为资源受限设备设计，强调在有限参数预算下实现尽可能高的精度和效率。与 LLM（通常 >7B）相比，SLM 在预训练成本、推理内存、推理延迟和能耗方面有数量级优势。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLM 的核心设计权衡是在模型容量（深度 × 宽度）和参数预算之间取得最优平衡。三种典型设计策略：
1. **宽度缩减型**（如 baseline1: 22 layers / hidden 1024 / 0.54B）：保持层数不变，缩减 hidden dimension → bottleneck effect（表征能力受限）。
2. **深度缩减型**（如 baseline2: 8 layers / hidden 2048 / 0.52B）：保持宽度不变，缩减层数 → 层次化语言理解能力下降。
3. **参数共享型**（如 MobiLlama: 22 layers / hidden 2048 / 0.52B）：保持高容量架构，通过参数共享机制削减参数 → 当前最优设计策略。

从算法pipeline角度拆解术语，给出具体例子。
MobiLlama 0.5B 完整架构配置：
```
hidden_size = 2048
num_layers = 22
num_heads = 32
intermediate_size = 5632
vocab_size = 32000
max_seq_len = 2048
norm_eps = 1e-6  # RMSNorm epsilon

# 嵌入: 32000 × 2048
# Attention (每层独立): Q/K/V/O projection，GQA kv_heads=4
# FFN (所有层共享): gate/up/down projection，仅 1 份参数
# 总参数 ≈ 0.52B
```

SLM 边界部署性能（MobiLlama 0.5B）：RTX2080Ti bf16: 63.38 tok/s, 3046 MB；i7 CPU 4bit GGUF: 36.32 tok/s, 799 MB；Snapdragon-685 4bit: 7.02 tok/s, 770 MB。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SLM 开发流程：架构设计（选共享策略）→ 预训练（Amber/RedPajama/C4 等公开数据，Flash-Attention 加速）→ 评估（Open LLM Leaderboard benchmarks）→ 量化部署（GGUF 4-bit → CPU/边缘设备）。全透明开源（数据 pipeline + 训练代码 + 模型权重 + 300+ checkpoints + 评估代码）是 MobiLlama 的重要贡献。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices

## SwiGLU (Swish-Gated Linear Unit)

术语解释
一种用于 Transformer FFN 层的门控激活函数，由 Swish（SiLU）激活和门控线性单元组合而成。被 Llama、MobiLlama 等现代 LLM/SLM 广泛采用，替代传统 ReLU/GELU FFN。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SwiGLU FFN 计算：
```
# 输入: x ∈ R^{d_model}
gate = x @ W_gate.T           # gate projection
up   = x @ W_up.T             # up projection
gate_act = SiLU(gate)         # SiLU(x) = x * σ(x)
gated = gate_act ⊙ up         # 逐元素门控乘法
output = gated @ W_down.T     # down projection
```

对比：ReLU FFN 用 2 个权重矩阵 + 1 次非线性；SwiGLU 用 3 个权重矩阵（gate/up/down）+ 门控机制提供更强的非线性表达能力。

从算法pipeline角度拆解术语，给出具体例子。
在 MobiLlama 共享 FFN 中：d_model=2048, d_intermediate=5632。所有 22 层对相同的 W_gate/W_up/W_down 执行 SwiGLU 计算。归一化后的 hidden state 经 RMSNorm → 线性投影 → SiLU → element-wise multiply → down projection → 残差连接。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 中 `LlamaMLP` 类实现：`gate_proj`, `up_proj`, `down_proj` 三个 `nn.Linear` + `nn.SiLU`。推理框架中常 fused 为单一 kernel（gate projection + SiLU + up projection + multiply + down projection）减少内存带宽。Shared FFN 场景下，fused SwiGLU kernel 可进一步获益：所有层对相同权重的访问被 L2 cache 高效服务。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices
- Scaling Law for Quantization-Aware Training

在 Scaling Law for QAT 中，SwiGLU 被识别为 W4A4 QAT 中激活量化误差的关键瓶颈来源。FC2 Proj（即 down_proj）的输入来自 SwiGLU 输出：gate_act = SiLU(x @ W_gate) ⊙ (x @ W_up)。gating + SiLU + element-wise multiply 的复合非线性变换产生系统性 outlier，导致 FC2 Proj 输入 kurtosis=89（即使 QAT 正则化后），远高于 QKV Proj、O Proj、FC1 Proj 等层（均 <10）。论文实验证明：对 FC2 Proj 输入使用 8-bit 混合精度可将 W4A4 量化误差降 20.5%（G=32）至 42.9%（G=256），激活误差对 G 的敏感度 γ_G 从 0.9812 降至 0.4471。

## RMSNorm (Root Mean Square Layer Normalization)

术语解释
简化版 Layer Normalization，仅使用均方根（RMS）统计量归一化，省略均值中心化步骤。相比标准 LayerNorm 计算量减半，被 Llama、MobiLlama 等现代 LLM 作为默认 normalization。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RMSNorm 公式：
```
rms = √((1/d) * Σx_i²)         # 仅计算 RMS
y = γ * x / (rms + ε)           # 缩放，无 center bias
```
对比 LayerNorm：省去均值计算 μ = (1/d)Σx_i，减少约 50% 计算。ε 常设为 1e-6（MobiLlama）或 1e-5（LLaMA）。

从算法pipeline角度拆解术语，给出具体例子。
在 MobiLlama pre-norm 架构中，每层使用两次 RMSNorm：
```
def TransformerBlock(x):
    h = RMSNorm(x, γ_attn)       # Pre-Attention normalization
    x = x + MHA(h)               # Attention + residual
    h = RMSNorm(x, γ_ffn)        # Pre-FFN normalization
    x = x + shared_FFN(h)        # Shared FFN + residual
    return x
```
RMSNorm 每层独立执行，但输出的归一化分布有利于共享 FFN 的泛化——不同层归一化后的表示具有相似统计特性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch: `x * self.weight / (torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True)) + self.eps)`。推理框架中常 fused 到相邻操作（如 fused RMSNorm + residual add kernel）以减少 launch 开销。

涉及论文标题：
- MobiLlama Small Language Model tailored for edge devices

## Modulated Quantization (调制量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Modulated Quantization（调制量化）是 MoDiff 论文提出的核心量化策略。不同于标准 PTQ 直接量化每步的原始激活值 a_t，调制量化利用扩散模型相邻时间步之间的时序相似性，将量化目标从原始激活转换为相邻时间步的差值 a_t − a_{t+1}。其数学基础是利用线性算子 A（如全连接层、卷积层）的线性性：o_t = A(a_t) = A(a_t − a_{t+1}) + o_{t+1}。因此只需量化差值 Δ_t = a_t − a_{t+1}，计算 A(Q(Δ_t)) 后加上缓存的上一时间步输出 o_{t+1} 即可得到当前步输出的近似。由于差值范围通常比原始激活小约 10×（见图1b 橙色 vs 蓝色分布），同等 bit-width 下量化误差大幅降低。当差值范围低于阈值时，Q 可分配 0-bit（即跳过该步计算），此时调制量化退化为缓存方法（如 DeepCache）的超集。调制量化与量化方法无关（agnostic to quantizer），可应用于任意 PTQ 方法（Q-Diffusion、LCQ、LTQ 等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在扩散模型去噪 U-Net 中，调制量化的算法流程为（以单层线性算子 A 为例）：

```
# === 初始化 (t=T) ===
a_hat[T] = Q(a_T)           # 第一步对原始激活量化
o_hat[T] = A(a_hat[T])      # 第一步全量化计算

# === 迭代 (t = T-1 到 1) ===
for t in range(T-1, 0, -1):
    delta = a_t - a_{t+1}           # 计算时序差值
    delta_q = Q(delta)              # 量化差值（低 bit）
    o_hat[t] = A(delta_q) + o_hat[t+1]  # 增量计算 + 缓存输出
```

关键性质：
- 差值 a_t − a_{t+1} 的量化误差由 Theorem 4.3 控制：||x − Q(x)||² ≤ (max(x)−min(x))²d/(2^b−1)²。由于差值范围约 10× 小于原始激活，等 bit 下误差约 100× 更小，或可用低 3-4 bit 达到同等误差。
- 该方法为每个线性层独立执行，不改变层间数据流或模型架构。
- 无需重新训练（training-free），属于 PTQ 范畴。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
1. 仅适用于线性算子（Linear、Conv2d），因其线性性 o_t = A(a_t − a_{t+1}) + o_{t+1} 严格成立。
2. 需移除应用 MoDiff 的层的 bias 项，避免 bias 在跨步迭代中重复累积。
3. 第一步 (t=T) 使用 Warm-up（全精度或高精度激活），为后续差值计算提供高质量基准。
4. 量化器 Q 可选用任意 PTQ 方法（min-max dynamic、Q-Diffusion 校准量化、tensor-wise、channel-wise 等）。
5. 代码开源：https://github.com/WeizhiGao/MoDiff，基于 Q-Diffusion 和 BRECQ（PyTorch）代码库构建。
6. 关键结果：CIFAR-10 W8A3 下 LCQ+MoDiff FID=4.14（vs FP=4.24），计算量从 1636 GBops 降至 154 GBops（10× 节省）。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

## Error-Compensated Modulation (误差补偿调制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Error-Compensated Modulation（误差补偿调制）是 MoDiff 框架中解决跨时间步量化误差累积的关键机制。在基础调制量化中（标准 modulation），使用原始激活 a_{t+1} 计算差值：ô_t = A(Q(a_t − a_{t+1})) + ô_{t+1}。但上一步的量化输出 ô_{t+1} = A(Q(a_{t+1})) 本身就含有量化误差，直接作为基准会导致误差在后续步累积。Theorem 4.4 证明标准调制的误差呈 O(2^{T−k}) 指数增长。

误差补偿调制的核心创新是引入中间变量 â_t = Q(a_t − â_{t+1}) + â_{t+1}，用 â_{t+1} 替代原始 a_{t+1} 作为差值计算的基准。重写后等价于：
ô_t = A(Q(a_t − a_{t+1} + e_{t+1})) + o_{t+1} − A(e_{t+1})
其中 e_{t+1} = a_{t+1} − â_{t+1} 是上一步的量化误差。该式展示了误差被自动补偿：上步漏算的 A(e_{t+1}) 在当前步被显式减去，同时 e_{t+1} 被注入 Q 的输入中参与当前步的量化。Theorem 4.4 证明补偿后的误差呈 O((2c)^{T−k}) (c<1/2) 指数衰减。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
完整的误差补偿调制算法（MoDiff 完整版）伪代码：

```
# === 初始化 (t=T, Warm-up) ===
a_hat[T] = Q(a_T)                   # Eq.(8): 量化原始激活
o_hat[T] = A(a_hat[T])              # Eq.(9): 首次量化计算

# === 误差补偿迭代 (t = T-1 到 1) ===
for t in range(T-1, 0, -1):
    # Eq.(13): 误差补偿激活重建
    a_hat[t] = Q(a_t - a_hat[t+1]) + a_hat[t+1]

    # Eq.(14): 调制量化计算
    o_hat[t] = A(Q(a_t - a_hat[t+1])) + o_hat[t+1]
```

误差补偿的机制解释：
- 定义量化误差 e_t = a_t − â_t（Eq.18），即原始激活与误差补偿后的激活之间的差。
- 展开 a_t − â_{t+1} = a_t − a_{t+1} + (a_{t+1} − â_{t+1}) = (a_t − a_{t+1}) + e_{t+1}
- 因此：ô_t = A(Q(a_t − a_{t+1} + e_{t+1})) + ô_{t+1}
- 而 ô_{t+1} = A(â_{t+1}) = A(a_{t+1} − e_{t+1}) = o_{t+1} − A(e_{t+1})
- 代入得：ô_t = A(Q(a_t − a_{t+1} + e_{t+1})) + o_{t+1} − A(e_{t+1})
- 上一步缺失的 A(e_{t+1}) 在当前步被显式补偿。

与标准调制（无补偿）的对比：
- 标准调制：ô_t = A(Q(a_t − a_{t+1})) + ô_{t+1}
- 这里 ô_{t+1} 本身含误差，且该误差在每步被携带和放大。
- 误差补偿显式隔离和消除了该误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现细节：
1. 每个线性层独立维护 â_t 和 ô_t 两个缓存变量（额外内存开销约 3-4 MB/层 for CIFAR-10）。
2. 需在校准阶段重构校准数据集：使用 MoDiff pipeline 的输入输出对（而非原始激活对）来优化 Q-Diffusion 的量化参数。
3. 逐层独立校准（而非逐 block）以保证稳定性。
4. 误差补偿机制的理论有效性条件为 quantization error bound c < 1/2（见 Corollary A.3），即量化误差必须小于输入幅度的一半，可由适当选择 bit-width b 保证。
5. 兼容所有 sampler（DDIM/DDPM/DPM/PLMS），兼容所有 PTQ 方法。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

## DeepCache (扩散模型深度缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DeepCache 是 Ma et al. (CVPR 2024) 提出的扩散模型加速方法，通过缓存和复用 U-Net 中间层的上采样特征来跳过冗余计算。核心观察：扩散模型去噪过程中相邻时间步的特征图高度相似，尤其在 U-Net 的 high-level（深层）特征。DeepCache 在 U-Net 的跳跃连接处缓存上采样特征，每 N 步更新一次缓存，中间 N−1 步直接复用缓存特征，跳过对应的下采样和中间块计算。N 越大加速越多但生成质量下降越严重。MoDiff 论文的初步研究（图1a）显示 DeepCache 的缓存复用策略会导致误差累积：即使 N=3（每3步更新），最终步的特征与标准扩散的相对 ℓ₂ 距离可达 40%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DeepCache 在 U-Net 中的应用流程：

```
# U-Net 结构: Encoder(下采样) → Middle Block → Decoder(上采样+跳跃连接)
# DeepCache 策略:
cache = None
for t in range(T, 0, -1):
    if t % N == 0 or cache is None:
        # 完整前向：执行 Encoder + Middle + Decoder
        features = UNet_full(x_t, t)
        cache = features.up_sample_block_1  # 缓存上采样第一块特征
    else:
        # 加速前向：跳过 Encoder 和 Middle block
        features_low = UNet.shallow(x_t, t)  # 仅执行浅层
        features = concat(features_low, cache)  # 复用缓存的上采样特征
    x_{t-1} = sampler_step(x_t, features)
```

MoDiff 与 DeepCache 的关系：MoDiff 的调制量化在差值范围低于阈值时可为差值分配 0-bit（即跳过计算），此时 MoDiff 等价于 DeepCache 的行为。但 MoDiff 通过统一的理论框架（调制量化 + 误差补偿）提供了更灵活的控制和更严格的误差保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源代码：https://github.com/horseee/DeepCache（CVPR 2024）。基于 Stable Diffusion 和 DDIM/DDPM sampler 实现。主要修改 U-Net 的 forward 函数，插入缓存检查逻辑。局限性：(1) N 值需手动调参，无自适应机制；(2) 仅适用于 U-Net 架构，不适用于 DiT 等 Transformer 架构；(3) 缓存误差累积严重，不能支持极低比特或大幅加速。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

## QAT Scaling Law (Quantization-Aware Training Scaling Law)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QAT 缩放定律是描述量化感知训练中量化误差 δ_p 随模型参数量 N、训练数据量 D 和量化粒度 G 变化的数学关系。与现有方法（通过 EPM 修改 Chinchilla 的 N 项）不同，QAT 缩放定律将量化误差建模为 Chinchilla loss 的独立加项：L(N, D, G) = A/N^α + B/D^β + E + δ_p(N, D, G)。其中 δ_p(N, D, G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}。使用对数项 log₂(G) 满足边界条件 G=1（无量化）时 δ_p=0。拟合参数 k, γ_N, γ_D, γ_G 均为正数，表明：δ_p 随 N 增大而减小（大模型更鲁棒）、随 D 增大而增大（更多训练数据放大全精度 vs 量化差距）、随 G 变粗而增大（粗粒度量化误差更大）。γ_N, γ_D, γ_G 的大小反映量化误差对各自变量的敏感度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QAT 缩放定律的建立流程：
```
# Step 1: BF16 基线训练 + Chinchilla 拟合
for each (N, D):
    model = Llama3Style(N)
    loss_bf16[N,D] = train_bf16(model, D)
fit L_bf16(N,D) = A/N^α + B/D^β + E  # 用 Huber loss + L-BFGS, 约束 α=β

# Step 2: W4A4 QAT 实验
for each (N, D, G) in grid:  # N∈{74,145,297,595}M, D∈{10,20,50,100}B, G∈{32,64,128,256,per-token/channel}
    model = Llama3Style(N)
    loss_W4A4[N,D,G] = train_W4A4(model, D, G)
    δ_W4A4[N,D,G] = loss_bf16[N,D] - loss_W4A4[N,D,G]  # ground truth

# Step 3: 拟合 δ_p 缩放定律
fit δ_p(N,D,G) = k · D^{γ_D} · (log₂(G))^{γ_G} / N^{γ_N}
# 80 次实验数据，Huber loss + L-BFGS

# Step 4: 外推验证
predict δ_p for 973M model at 100B/200B tokens → compare with actual
```

EPM 推导：eff(C) = [A / (A + k · D^{γ_D} · (log₂(G))^{γ_G} · N^{α-γ_N})]^{1/α}。当 α > γ_N 时（W4A4 满足），eff(C) 随 N 增大而减小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缩放定律的实用价值：(1) 预测任意 (N,D,G) 组合的量化误差，指导模型设计和训练策略；(2) 通过 EPM 评估 W4A4 vs W8A8 的 cost-accuracy trade-off（EPM > 0.5 时 4-bit 更优）；(3) 量化误差分解（W4A16/W16A4）揭示权重 vs 激活的贡献，指导优化方向；(4) 结合 FC2 瓶颈分析，通过混合精度消除主要误差源。论文使用 PyTorch + OLMo2 训练框架，未开源专用代码。拟合使用 Huber loss（对 outlier 鲁棒）+ L-BFGS 优化器。

涉及论文标题：
- Scaling Law for Quantization-Aware Training



## INT4 vs FP4 Quantization (整数 4-bit vs 浮点 4-bit 量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT4 和 FP4 是两种 4-bit 量化数据格式。INT4 均匀映射 16 个等间距整数（[-8, 7]），每个值之间步长相等。FP4（E2M1 格式）由 1-bit 符号 + 2-bit 指数 + 1-bit 尾数组成，仅 15 个有效值（因正负零冗余），可表示 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}。两者在 W4A4 QAT 场景中精度接近，但 INT4 在 per-channel/token 粒度下优 0.015 loss（因多 1 个可表示值），在 group-wise 下等价。论文选择 INT4 作为缩放定律实验的默认格式（因等价或更优且数学形式更简单），并假设 INT4 和 FP4 遵循相同的缩放定律函数形式（图 13 实验验证 INT4 拟合的缩放定律能准确预测 FP4 误差趋势）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# INT4 量化/反量化
scale = 8 / max(|X|)  # M=8 for INT4
X_int = clamp(round(X/scale), -8, 7)
X_hat = X_int * scale  # 均匀分布值: {-8s, -7s, ..., 7s}

# FP4 E2M1 量化/反量化
# E2M1: S(1bit) | E(2bit) | M(1bit)
# bias = 2^(2-1) - 1 = 1  (但E2M1通常bias=0 for subnormal)
# Normal values: (-1)^S × 2^{E} × (1+M/2) for E>0
# Subnormal: (-1)^S × 2^{0} × (M/2) for E=0
# 15 unique values (0 有 ±0 冗余)
scale = 6 / max(|X|)  # M=6 for E2M1 FP4
X_fp4 = map_to_nearest_fp4(X/scale)
X_hat = X_fp4 * scale
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
INT4 广泛用于 LLM 量化（GPTQ、AWQ、GGUF Q4_0 等），因硬件支持成熟（NVIDIA INT8 Tensor Core 可模拟 INT4×INT4 GEMM）。FP4 随着 Blackwell GPU 原生 FP4 Tensor Core 的推出受到更多关注。选择建议：(1) INT4 在细粒度下略优（1 个额外可表示值）；(2) FP4 的 E2M1 隐式匹配分布形状——对小值区域分配更密集层级（0, 0.5, 1, 1.5, 2），类似 Student t-distribution 的尖峰厚尾特征；(3) FP4 可通过 supernormal support 回收负零位提升精度。

涉及论文标题：
- Scaling Law for Quantization-Aware Training
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
## Optimal Brain Restoration (OBR / 最优脑恢复)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OBR (Optimal Brain Restoration) 是 ETH Zurich 提出的训练无关统一框架，用于实现 LLM 的联合量化和稀疏化（joint quantization and sparsification），被 ICLR 2026 接收（https://arxiv.org/abs/2509.11177）。核心思想：在剪枝和量化之间通过 Group Error Compensation 计算最优补偿，调和两者对权重分布的冲突需求——量化偏好紧凑数值范围（减少量化误差），剪枝偏好高方差分布（暴露天然稀疏性）。OBR 将权重元素分为 retain set R（鲁棒的）和 eviction set E（易受压缩影响的），通过 Hessian 矩阵作为"桥梁"将 E 的压缩误差转移到 R：Δw_R^* = −H_{RR}^{-1} H_{RE} e_E（闭式解）。使 OBR 成为首个实现 W4A4KV4 + 50% 稀疏度且无需重训练的 LLM 压缩方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
OBR Pipeline = prune-then-quantize + 两阶段 Group Error Compensation：

```
输入: Hadamard-rotated W ∈ R^{C_out×C_in}, H = 2XX^T, 分区比 α
输出: INT4 + sparse Ŵ

// Step 0: 二阶目标近似
H_full ≈ I ⊗ H  // Kronecker 分解 + 行解耦
→ C_out 个独立子问题: min ½ Δw_i H Δw_i^T

// Step 1: 剪枝（使用 WANDA mask）
M = prune(W),  W^{prune} = W ⊙ M

// Step 2: Group Error Compensation（逐行）
for c = 1 ... C_out:
    // Phase 1: OBR for Pruning
    R₁ = {i | M_{c,i}=1},  E₁ = {j | M_{c,j}=0}
    Δw_{R₁}^{prune} = -H_{R₁R₁}^{-1} H_{R₁E₁} W_{c,E₁}
    w̄ = W_{c,R₁}^{prune} + Δw_{R₁}^{prune}

    // Phase 2: OBR for Quantization
    e^{quant} = w̄ - quantize(w̄)
    E₂ = first α×|R₁| elts,  R₂ = rest (1-α)×|R₁|
    Δw_{R₂}^{quant} = -H_{R₂R₂}^{-1} H_{R₂E₂} e_{E₂}^{quant}

    ΔW_{c,R₁}^{OBR} += Δw_{R₁}^{prune}
    ΔW_{c,R₂}^{OBR} += Δw_{R₂}^{quant}

// Step 3: 量化输出
Ŵ = quantize(W^{prune} + ΔW^{OBR})  // RTN 或 GPTQ
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/csguoh/OBR (Python + PyTorch + CUTLASS)。128 WikiText2 样本校准计算 H=2XX^T，WANDA 生成 mask，OBR 逐行闭式补偿（α=50% 默认）。预校准模型: https://huggingface.co/HangGuo/OBR。7B 约 2h，70B 约 36h（单 A100）。兼容 QuaRot/SpinQuant/FlatQuant 旋转 + WANDA/SparseGPT/magnitude 剪枝。一次压缩，无限次低代价推理。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## Joint Quantization and Sparsification (联合量化和稀疏化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
联合量化和稀疏化是将量化（减少 bit-width）和剪枝（移除不重要权重）同时应用于同一 LLM 的技术范式，突破单一压缩方法的限制。单一量化在 sub-4bit 时性能急剧下降（如 QuaRot W3A4KV4 → 132.97 PPL），单一剪枝 >50% 稀疏度也面临瓶颈。联合方法叠加两种正交压缩维度实现更激进压缩——W4A4KV4+50% sparsity 有效 bit-width 等效于约 W2.5 水平。NVIDIA Ampere/Hopper 原生支持 INT4 sparse GEMM，使联合压缩具备实际硬件加速价值。核心挑战：量化偏好窄范围，剪枝偏好高方差，Hadamard rotation 虽利于量化但破坏剪枝所需的分布差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种实现范式对比：

```
Naive sequential (QuaRot+WANDA):
  rotate(W) → prune(rotated_W) → quantize  // 分布冲突 → 性能崩溃

SparseGPT+GPTQ:
  calibrate → SparseGPT row-wise prune+Hessian update → GPTQ  // 改善但未调和冲突

OBR:
  rotate → prune → OBR compensation(prune error) → OBR compensation(quant error) → quantize
  // 通过 Hessian 桥接两种压缩，调和分布冲突
```

等效 bit-width: effective_bits = actual_bits × (1-sparsity_ratio)。如 W4+50% sparsity → 2bit equivalent per weight。OBR 同等等效位宽下显著优于纯量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) prune-then-quantize 顺序（被 Harma et al. 2024 理论证明最优）；(2) 联合梯度优化（DJPQ, OBQ）；(3) 误差补偿框架（OBR, JSQ）。硬件：NVIDIA Sparse Tensor Cores + INT4。适用场景：边缘设备 LLM 部署、memory-bound 推理。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## W4A4KV4 Quantization (4-bit 权重-激活-KV Cache 量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4A4KV4 是 LLM 压缩的激进配置：Weights INT4、Activations INT4、KV Cache INT4。相比 weight-only (W4A16) 仅压缩模型参数，W4A4KV4 同时压缩推理时的激活内存和 KV Cache 内存，大幅减少 decode 阶段显存占用（KV Cache 是长序列生成瓶颈）。在单张消费级 GPU 上运行 7B LLM 成为可能。核心难点：激活和 KV Cache 比权重对量化更敏感，需 Hadamard rotation 等预处理平滑分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
W4A4KV4 单层推理数据流：

```
// Q/K/V 投影: W4A4 GEMM
Q = dequant(W_Q_int4) × dequant(A_Q_int4)  // INT4→FP16→matmul
K = dequant(W_K_int4) × dequant(A_K_int4)
V = dequant(W_V_int4) × dequant(A_V_int4)

// Attention: FP16
S = QK^T/√d,  A = softmax(S)V

// KV Cache: store INT4
K_cache_int4 = round((K - z_K) / s_K)
V_cache_int4 = round((V - z_V) / s_V)

// Output: W4A4
O = dequant(W_O_int4) × dequant(A_O_int4)
```

内存：weight 4×、activation 4×、KV cache 4× 减少；叠加 50% sparsity → weight 再减半（总约 6.4×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuaRot/SpinQuant/FlatQuant 实现 Hadamard rotation + per-channel/token RTN/GPTQ。OBR 叠加 50% sparsity 并加入误差补偿。NVIDIA Ampere/Hopper 原生支持 INT4 Tensor Core。开源：QuaRot (https://github.com/spcl/QuaRot)，OBR (https://github.com/csguoh/OBR)。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## 2:4 Semi-structured Sparsity (2:4 半结构化稀疏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2:4 semi-structured sparsity：每组连续 4 个权重中恰好 2 个为零，保证 50% 恒定稀疏率 + 硬件友好规则性。NVIDIA Ampere 起通过 Sparse Tensor Cores 原生支持，约 2× 理论加速。vs unstructured sparsity（灵活但无硬件加速），vs block sparsity（粗粒度但精度损失大）。可扩展为 4:8 等变体。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// W ∈ R^{M×K}, 按列分组为 K/4 组
// 每组 [w₀,w₁,w₂,w₃] → 保留 top-2 幅度, 清零 bottom-2
// Metadata: 2-bit index 编码非零位置

// Sparse Tensor Core MMA:
for tile in [M×K]:
    load W_tile(50% bandwidth) + metadata
    load X_tile
    Y += mma.sp.sync(W_tile, X_tile, metadata)  // 硬件跳过零
```

OBR W4A4KV4+2:4 sparse → perplexity 13.32 (vs 34.76 for SparseGPT+GPTQ)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NVIDIA CUTLASS 2:4 sparse GEMM API；PyTorch `torch.sparse.semi_structured`；TensorRT-LLM 内置。剪枝算法 ASP/SparseGPT/WANDA 均支持 2:4。需 A100+ GPU。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## WANDA (Pruning by Weights and Activations / 权重激活感知剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WANDA (Pruning by Weights and activations) 由 Sun et al. (CMU & Meta AI, ICML 2023) 提出，是一种简单高效的 one-shot LLM 剪枝方法。核心创新：剪枝度量 S_ij = |W_ij| · ||X_j||₂（权重绝对值 × 对应输入激活 ℓ₂ 范数），比纯幅度剪枝更准确。每个输出神经元内独立剪枝（行内），保证各输出特征稀疏度平衡。无需反向传播、无需 Hessian、无需权重更新，仅需单次前向。复杂度 O(d²) vs SparseGPT O(d³)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# WANDA 逐行激活感知剪枝
输入: W ∈ R^{C_out×C_in}, 校准激活 X ∈ R^{C_in×L}, sparsity s
输出: M ∈ {0,1}^{C_out×C_in}

for i in range(C_out):
    scores = |W[i,:]| ⊙ ||X||₂              // row j → |W_ij|×||X_j||₂
    k = floor(C_in × s)
    threshold = top_k(scores, k)             // 保留 top-(1-s)
    M[i,:] = (scores >= threshold)
    W_pruned[i,:] = W[i,:] ⊙ M[i,:]
```

OBR 以 WANDA 生成的 mask 为输入（WANDA 负责选定剪枝位置，OBR 负责补偿剪枝损失）。多种 mask 兼容性测试：WANDA > SparseGPT > Magnitude > Random。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/locuslab/wanda。128 calibration samples 单次前向收集 ||X_j||₂ → 逐行剪枝 → 无需 finetuning。支持 unstructured 和 N:M semi-structured。作为 OBR 的默认 mask generator。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## QuaRot (Quantization with Random Orthogonal Transformation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QuaRot 是基于 Hadamard 随机正交变换的 LLM PTQ 方法（Ashkboos et al., NeurIPS 2024）。核心思想：LLM 权重含 outliers，QuaRot 对权重施加随机 Hadamard 变换 W̃ = R₁WR₂^T 将 outliers 扩散到所有元素使分布平坦。由于 R 正交（RR^T=I），乘法不变（WX = R₁^T W̃R₂X），可无精度损失融合。实现 W4A4KV4 量化，是 OBR 的 backbone 旋转方法之一。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// QuaRot 旋转 + 在线融合
// 每个 Linear layer:
W_rot = R₁ × W × R₂^T
X_rot = R₂ × X                      // 在线 Hadamard 变换
Y_rot = W_rot × X_rot = R₁ × (WX)   // 等价性保证
// 下一层需吸收 R₁^T（或在线逆旋转）
// 量化：W_rot → INT4, X_rot → INT4

// Q/K/V 投影共享 R₂-QK, R₁-Q, R₁-K, R₁-V
// KV Cache 也量化到 INT4
```

FHT (Fast Hadamard Transform): O(d log d) 复杂度旋转。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/spcl/QuaRot (PyTorch + fast-hadamard-transform https://github.com/Dao-AILab/fast-hadamard-transform)。`--rotate` 标志启用。OBR 复用 QuaRot 旋转矩阵不重新训练，加入剪枝和误差补偿达成 W4A4KV4+50% sparsity。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## PAR (Piecewise-Affine Regularization / 分段仿射正则化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PAR（Piecewise-Affine Regularization，分段仿射正则化）是 PARQ 论文提出的凸正则化框架，用于通过连续优化诱导模型参数量化。其核心构造：定义分段仿射函数 Ψ(w) = max_{k∈{0,...,m}} {a_k(|w|-q_k) + b_k}，其中 Q = {0, ±q_1, ..., ±q_m} 为目标量化值集合，斜率 {a_k} 满足 0 ≤ a_0 < a_1 < ... < a_m = +∞（严格递增），截距 b_k 由 b_0=0, b_k = b_{k-1} + a_{k-1}(q_k - q_{k-1}) 递推确定。关键性质：(1) 凸性——有限个线性函数的最大值保持凸性（Boyd & Vandenberghe）；(2) 非光滑性——在 ±q_k 处形成折点（kinks），使子梯度包含区间而非常数值，从而在这些位置"困住"优化变量，产生聚类效应。这与 ℓ₁ 正则化通过原点处的折点诱导稀疏性的机制完全类似。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PAR 的一阶最优性条件揭示了量化诱导机理：∇f(w*) ∈ -λ ∂Ψ(w*)。具体展开：
- w* = q_k ⟺ ∇f(w*) ∈ λ(-a_k, -a_{k-1})（子梯度区间使解稳定在 q_k）
- w* ∈ (q_{k-1}, q_k) ⟹ ∇f(w*) = -λ a_{k-1}（非量化区间的梯度被强制等于 2m 个离散值 {±λ a_k} 之一）
直觉：几乎所有的梯度值可以通过将权重置于 Q 中 2m+1 个离散值来平衡，使得最优解处权重"大概率"聚合在离散量化值上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PAR 通过其 proximal map 实现：prox_{Ψ}(u) = argmin_w {Ψ(w) + (1/2)‖w-u‖²}，有闭式解（式 7）：若 |u| ∈ [a_{k-1}+q_k, a_k+q_k]，则 w = sgn(u)·q_k（flat segment，硬量化）；若 |u| ∈ [a_k+q_k, a_k+q_{k+1}]，则 w = u - sgn(u)·a_k（slanted segment，软量化线性偏移）。在 PARQ 实现中，LSBQ 在线估计 Q，独立斜率 schedule ρ_t^{-1} 控制软→硬过渡，无需显式设置 a_k 和 λ。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization

## AProx (Aggregate Proximal Gradient / 聚集近端梯度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AProx（Aggregate Proximal Gradient，聚集近端梯度）是 PARQ 论文提出的随机优化算法，用于求解 PAR-正则化目标函数 minimize_w f(w)+λΨ(w)。算法形式：u^{t+1}=u^t-η_t ∇f(w^t,z^t)（隐变量累积纯梯度），w^{t+1}=prox_{γ_t λ Ψ}(u^{t+1})（用累积步长 γ_t=Ση_s 缩放正则化）。AProx 的核心创新在于使用累积步长 γ_t（而非 Prox-SGD 的单步步长 η_t）缩放近端正则化映射。由于 γ_t → ∞，proximal map 中 flat segments 长度不断增大，slanted segments 相对缩小，使软量化渐近收敛到硬量化。这与 Prox-SGD 的 diminishing regularization（η_t→0 导致正则化消失）恰好相反。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AProx 与 Prox-SGD 的核心对比：
```
Prox-SGD: u^{t+1}=w^t-η_t g^t (w^t 已含过往 prox 贡献), w^{t+1}=prox_{η_t λ Ψ}(u^{t+1})
问题: η_t→0 → 正则化消失 → 无量化效果

AProx: u^{t+1}=u^t-η_t g^t (u^t 仅累加梯度), w^{t+1}=prox_{γ_t λ Ψ}(u^{t+1})
优势: γ_t→∞ → 正则化增强 → 软→硬量化渐进收敛
```
AProx 等价于 ProxConnect（Dockhorn et al. 2021），但从 RDA（Xiao 2010）推导而来，比 Fenchel-Rockafellar 对偶框架更直观。定理 3.2 证明了最后迭代收敛 O(ln(t)/√t)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 实现中，因训练迭代次数有限使得 γ_t 达不到无穷大，使用独立斜率 schedule ρ_t^{-1}（cosine decay 从 1→0）模拟 γ_t 的渐进效应。PARQ 算法将 AProx 的三个组件（LSBQ 在线估 Q、prox 软量化、逆斜率 schedule）组合成实用的 QAT pipeline。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization

## BinaryConnect（二元连接）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BinaryConnect 是 Courbariaux et al.（2015, NeurIPS）提出的早期 QAT 方法，是 STE 的代表性实践。更新规则：u^{t+1}=u^t-η_t ∇f(Q(u^t),z^t), w^{t+1}=Q(u^{t+1})。全精度隐变量 u^t 累积在量化权重 w^t=Q(u^t) 处的梯度，硬量化映射 Q(·) 每次将隐变量投影到离散集 Q^d。Ste 在反向传播时替换 dQ/du=0 为 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
1-bit BinaryConnect 训练流程：
```
初始化: u^1 = w^1 (随机 FP32 权重)
for t=1 to T:
    w^t = sign(u^t) * q          // q = ‖u‖₁/d 或 q=1
    g^t = ∇f(w^t, minibatch_t)
    u^{t+1} = u^t - η_t * g^t    // STE: dQ/du 替换为 1
```
多 bit 推广：Q(·) 为投影到 Q^d (Q={0,±q_1,...,±q_m}) 的阶梯函数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 论文中，BinaryConnect 被统一进 AProx 框架：当 Ψ=δ_Q（indicator function）时，prox_{Ψ}=Q(·) 且尺度不变（任何缩放下不变），因此 BinaryConnect/STE 是 AProx 的特例。PARQ 揭示 STE 是 AProx 在 γ_t→∞ 下的渐近极限，赋予启发式 STE 严谨的优化理论基础。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization

## BinaryRelax（二元松弛）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BinaryRelax 是 Yin et al.（2018, SIAM Journal on Imaging Sciences）提出的 QAT 方法，使用 W 形非凸正则化的 relaxed proximal map 替代硬量化。其 proximal map 在量化值附近为倾斜段，斜率随时间逐渐减小至 0。与 STE 全程硬量化不同，BinaryRelax 允许训练初期权重不完全量化，通过放松的 regularized proximal map 稳定训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BinaryRelax 使用 Moreau envelope of indicator function: Ψ(w)=min_{v∈Q^d} ‖v-w‖_2²（平滑近似）。其 proximal map（图 9b）slanted segment 的斜率随时间递减到 0。与 PARQ 的关键区别：PAR 是凸的（max of linear functions），BinaryRelax 的 W 形正则化是非凸的（有中间 hill），gradient-based 优化可能因初始点被困在"错误山谷"而达不到全局最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 论文将其作为对比 baseline。差异方向：BinaryRelax 的 slanted slope → 0（放松量化），PARQ 的 slanted slope → ∞（收紧量化到硬量化）。实验显示 PARQ 训练更稳定（无 sudden accuracy drops），尤其在极低位宽（1-bit/ternary）场景。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization

## LSBQ (Least Squares Binary Quantization / 最小二乘二元量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LSBQ（Least Squares Binary Quantization，最小二乘二元量化）是 Pouransari et al.（2020, CVPRW）提出的量化值估计方法。目标：将向量 u ∈ R^d 近似为 w_i = Σ_{j=1}^n v_j s_j(u_i)，其中 v_j 为正递减标量，s_j: R → {-1, 1} 为二进制函数，最小化 ‖u - w‖²。量化值集合 Q = {±v_1 ± v_2 ± ... ± v_n}，共 2^n 个可能值。n=1 时闭式解 v_1 = ‖u‖₁/d, s_1(u_i)=sign(u_i)；n>2 时通过 greedy foldable representation 求解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
n=2 (2-bit/ternary): 
  求解 v₁, v₂ → Q = {±(v₁+v₂), ±(v₁-v₂)} (4值 或 ternary 3值若 v₁=v₂)
n>2: greedy foldable
  for j=1..n:
      residual = u - Σ_{ℓ=1}^{j-1} v_ℓ s_ℓ(u)  // 累积重建残差
      s_j(u_i) = sign(residual_i)                // 二进制残差方向
      v_j = mean(|residual|)                     // 最优残差幅度
```
PARQ 在每轮迭代从 u^{t+1} 在线估计 Q^{t+1}，值从随机初始小值→早期膨胀→后期缓慢收缩。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LSBQ 产生 foldable (scaled binary) 量化——量化值间距由数据分布决定。在 PARQ 中使用 Q 直接作为 prox_PARQ 的 flat segment 位置。避免手动预设量化值 {q_k} 和正则化超参数 λ、{a_k}。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization

## ProxConnect（近端连接）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ProxConnect 是 Dockhorn et al.（2021, NeurIPS）提出的 QAT 泛化框架，将 BinaryConnect 的硬量化映射替换为任意单调递增 proximal map。推导自 Fenchel-Rockafellar 对偶和广义条件梯度（Yu et al. 2017）。PARQ 论文明确指出 AProx ≡ ProxConnect（仅有 γ_t 设置的微差异），但给出了一条更直观的推导路径（从 RDA/Xiao 2010 推广）和更强的理论结果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ProxConnect/AProx 的统一形式：u^{t+1}=u^t-η_t g^t, w^{t+1}=P(u^{t+1})，P 为任意 monotone non-decreasing proximal map。关键区别：PARQ 构造的凸 PAR 给出了 P 的显式闭式解（式 7），而 Dockhorn et al. 虽然讨论了凸正则化的可能性但没有给出具体构造。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PARQ 的核心理论贡献：(1) 构建了具体的凸 PAR 实例（之前仅有非凸正则化的 W 形）；(2) 证明 ProxConnect/AProx 在凸 PAR 下具有最后迭代收敛——而 Dockhorn et al. 仅证明了平均迭代收敛（对 QAT 无实际意义，因为平均值通常不被量化）。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization

## Last-iterate Convergence（最后迭代收敛）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Last-iterate Convergence（最后迭代收敛）是随机优化中保证最后一个迭代点 w^T（而非所有迭代的加权平均 w̄^T）收敛到最优解的理论性质。在 QAT 语境中至关重要：即使每个 w^s 都是量化解，其加权平均 w̄^T 通常不再被量化，因此平均迭代收敛结果对 QAT 没有实际意义。最后迭代收敛保证了最终输出的 w^T 本身（量化的）逼近量化约束下的最优解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PARQ 的 Theorem 3.2 证明 AProx 的最后迭代收敛：
```
E[F_λ(w^T)] - F_λ(w*) ≤ GR · (2 + 1.5 ln(T)) / √T
```
证明分两步：(1) Lemma A.3 (Orabona 2020) 将 η_T·q_T（最后迭代差距）分解为平均差距 + 尾项修正：η_T q_T ≤ (1/T)Σ η_t q_t + Σ_k (1/k(k+1)) Σ_{t=T-k+1}^T η_t(q_t - q_{T-k})；(2) 平均差距由 Theorem 3.1 的 regret bound 控制，尾项通过 telescoping + (T-k+1..T 区间) 内的单步 bound 控制。最终收敛率与平均迭代同阶 O(1/√T)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
一般 SGD 的最后迭代收敛常需要额外假设（强凸性、平滑性）。PARQ 证明 AProx 在仅凸性下取得最后迭代收敛，归因于 aggregation 机制的隐式正则化效应。这解决了 ProxConnect/Dockhorn et al. 仅证明平均迭代收敛的理论缺口。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization

## Partially-Binarized Quantization（部分二值化量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
部分二值化量化（Partially-Binarized Quantization）是 PB-LLM（Shang et al., 2023）提出的一种极端低比特 LLM 量化策略。核心思想：不是将所有权重二值化（全部 → ±1），而是检测并保留少量（5%-30%）"显著权重"（salient weights）在高位宽（如 INT8），仅将剩余的 90%+ 非显著权重二值化为 ±1。这与传统的均匀量化（所有权重使用相同位宽）和完全二值化（所有权重 → 1-bit）都不同——它是一种混合精度策略，利用 LLM 中显著权重的存在性来实现接近 1-bit 的平均位宽。存储开销：N_bit ≤ 1 × r_binary + 8 × (1 − r_binary) + 1（bitmap index），例如保留 10% salient 权重为 INT8 时等效位宽约 2.7-bit。PB-LLM 在 PTQ 和 QAT 两种框架下均实现了部分二值化：PTQ 下通过 PB-GPTQ（Hessian 引导迭代量化 + 补偿），QAT 下通过冻结显著权重 + 最优缩放因子（α* = ||w_F||_1/n）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B 某 Linear 层权重 W ∈ R^{d_o × d_i} 为例：
```
# Step 1: Salient Weight Detection
# Magnitude criterion (QAT):
salient_indices = topk(|W|.flatten(), k=int(d_o * d_i * salient_fraction))
salient_mask[i,j] = 1 if (i,j) in salient_indices else 0

# Hessian criterion (PTQ):
H = 2 * X @ X.T  # calibration data Hessian
H_inv = cholesky_inverse(H)
saliency[i,j] = W[i,j]^2 / H_inv[j,j]^2  # from SparseGPT
salient_indices = topk(saliency.flatten(), k)

# Step 2: Partial Binarization
W_salient = W * salient_mask       # 保留 INT8
W_unsalient = W * (1 - salient_mask)  # 二值化

# QAT 前向传播:
α = mean(|W_unsalient|, dim=1)      # column-wise optimal scaling
W_binary = α.unsqueeze(1) * sign(W_unsalient)
y = W_salient @ x + W_binary @ x    # 混合精度矩阵乘法

# PTQ (PB-GPTQ):
for col in range(d_i):
    w_sal = W_salient[:, col]; w_unsal = W_unsalient[:, col]
    ŵ_unsal = α[col] * sign(w_unsal)       # 二值化
    ŵ_sal = MinMaxQuant(w_sal, bit=8)       # INT8 量化
    error = W[:, col] - Ŵ[:, col]
    W[:, col+1:] += error / H_inv[col,col] * H_inv[col, col+1:]  # Hessian 补偿
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/hahnyuan/PB-LLM。关键参数：`--low_frac`（二值化比例）、`--salient_metric`（magnitude 或 hessian）、`--high_bit`（显著权重位宽）。QAT 训练配置：AdamW optimizer, lr=2e-5, cosine decay, 10K iterations, batch size=1 per GPU。训练数据使用 RedPajama-simple-1B。部分二值化矩阵推理时存储为混合格式：INT8 salient weights + packed binary unsalient weights + column-wise α scaling factors + bitmap index。推理时二进制权重通过 dequant（α × sign 值）展开为 FP16 后执行标准 GEMM。论文主要关注 memory 压缩而非 kernel 加速，但理论上二进制权重可将 FP 乘法替换为 XNOR+Bitcount 实现 64× 理论加速。

涉及论文标题：
- PB-LLM Partially Binarized Large Language Models

---

## Optimal Scaling Factor for Binary Weights（二值权重最优缩放因子）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
二值权重最优缩放因子（Optimal Scaling Factor for Binary Weights）是 PB-LLM 在 QAT 框架下提出的解析推导方法。核心问题是：给定 FP 权重列向量 w_F ∈ R^n，将其二值化为 w̄_B = sign(w_F) ∈ {−1, +1}^n 后，如何选择标量缩放因子 α 使得重构误差最小？PB-LLM 证明当 w̄_B = sign(w_F) 时，存在闭式解：α* = ||w_F||_1 / n = mean(|w_F|)，即列向量的 L1 范数平均值。推导过程：最小化 L2 误差 J(α) = ||w_F − α w̄_B||²₂，展开得 J(α) = α²(w̄_B^T w̄_B) − 2α(w_F^T w̄_B) + (w_F^T w_F)。由于 w̄_B = sign(w_F), w̄_B^T w̄_B = n, w_F^T w̄_B = Σ|w_F,i| = ||w_F||₁，求 ∂J/∂α = 0 得 α* = (w_F^T w̄_B)/n = ||w_F||₁/n。这与 XNOR-Net 的 channel-wise L1 norm scaling 在形式上一致，但 PB-LLM 的创新在于将其集成到部分二值化的 QAT 框架中，与 Salient Weights Frozen 协同工作，使训练更高效——无需像 AWQ 那样通过 grid search 寻找最优 scaling factor。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B 某 Linear 层的一个权重列为例（QAT 前向传播）：
```
# 输入: FP latent weight column w_F ∈ R^n (unsalient 权重)
#       salient weights 已 frozen 为 INT8

# Step 1: 二值化
w_bar_B = sign(w_F)  # ∈ {−1, +1}^n

# Step 2: 最优缩放因子（闭式解，无需搜索）
alpha_star = sum(|w_F|) / n  # = ||w_F||₁ / n

# Step 3: 重构二值化权重
w_hat = alpha_star * w_bar_B  # 最优 L2 逼近

# Step 4: 混合精度前向传播
y = W_salient @ x + alpha_star * sign(W_F_unsalient) @ x
# salient 部分: INT8 × FP16 → FP16
# unsalient 部分: (α × sign) × FP16 → FP16 GEMM
```
注意：该最优解仅在 w̄_B = sign(w_F) 的条件下成立——如果使用其他二值化策略（如随机二值化），则需要重新求解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PB-LLM 的 QAT 实现中，该缩放因子按 column-wise 粒度计算——每个输出 channel 的权重列计算独立的 α*。这是基于以下考量：LLM 中不同 channel 的权重幅度差异很大（channel-wise variance），per-column scaling 比 per-tensor scaling 能更精确地近似原始权重分布。与 AWQ 的 per-channel scaling 不同，AWQ 的 scaling 应用于量化前（作为预处理），而 PB-LLM 的 scaling 是二值化后的重构步骤（作为 post-binarization correction）。训练时 α* 每步前向重新计算（因为 w_F 随训练更新），推理时 α* 固定存储（每个列一个 FP16 标量，开销可忽略）。

涉及论文标题：
- PB-LLM Partially Binarized Large Language Models

---

## Salient Weights Frozen in Binarization QAT（二值化 QAT 中的显著权重冻结）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
显著权重冻结（Salient Weights Frozen）是 PB-LLM QAT 训练框架的核心策略。在量化感知训练开始前，按权重大小（magnitude）排序选出 top-k%（如 2%-30%）的显著权重，这些权重在训练全过程保持冻结（不参与梯度更新），仅对剩余的二值化权重的 FP latent 进行优化。设计动机：(1) 显著权重承载了 LLM 的关键语言能力——即使只保留 2% 的权重不解冻，训练也能更快收敛（图 5 训练曲线）；(2) 冻结减少可训练参数量，降低优化难度——PB-LLM 仅需 1-10K iterations 即可恢复量化模型性能，而 LLM-QAT 等全参数量化训练方法需要 100K iterations；(3) 反直觉的是，仅凭 Salient Frozen + Optimal Scaling 两个机制，无需任何训练就能使部分二值化 LLM 保持一定语言能力（图 6：50% salient 的 OPT-1.3B PPL ~20，非完全崩溃）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 训练前（仅执行一次）
W = pretrained_linear_layer.weight  # ∈ R^{d_o × d_i}
salient_frac = 0.1  # 保留 10% 为显著权重
salient_mask = zeros_like(W)
threshold = quantile(|W|.flatten(), 1 - salient_frac)
salient_mask[|W| >= threshold] = 1

W_salient = W * salient_mask        # Frozen, INT8 quantized
W_F_unsalient = W * (1 - salient_mask)  # Trainable FP latent

# 训练循环
for step in 1..10000:
    # 前向: salient 部分冻结，unsalient 部分二值化
    Ŵ_salient = MinMaxQuant(W_salient, bit=8)     # 固定不变
    Ŵ_unsalient = α * sign(W_F_unsalient)          # 每步更新，α = ||w_F||₁/n
    y = Ŵ_salient @ x + Ŵ_unsalient @ x

    # 反向: 仅更新 W_F_unsalient
    ∂L/∂W_F_unsalient = STE(∂L/∂Ŵ_unsalient)  # STE 穿过 sign()
    W_F_unsalient -= lr * ∂L/∂W_F_unsalient
    # W_salient 不更新 (frozen)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PB-LLM 实现中，salient weight 检测使用 element-wise magnitude 排序（而非 AWQ 的 per-channel activation-based 方法）。选择 magnitude 准则的原因：PB-LLM 实验表明，在 PTQ 场景下 magnitude 和 Hessian 检测的效果差异不大（Table 1），而 magnitude 更简单、无需校准数据。显著权重比例（salient fraction）是关键超参数：30% salient（等效 ~3.7-bit）可接近 FP 性能（LLaMA-7B Avg 66.9 vs FP 68.7），10% salient（等效 ~1.7-bit）仍有合理性能（Avg 60.6）。低于 5% salient 时性能急剧下降。显著权重的 element-wise 分布呈均匀随机散射（图 3），无明显的列聚集模式，因此 column-wise 选择方法不适合二值化场景。

涉及论文标题：
- PB-LLM Partially Binarized Large Language Models

## Progressive KV Cache Quantization (渐进式KV缓存量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Progressive KV Cache Quantization 是 PM-KVQ 提出的一种面向 long-CoT LLM 的 KV Cache 量化策略。核心思想是不在每个 decoding step 直接将 KV Cache 量化到目标 Fbit（Final bit-width），而是从 16-bit 开始逐步缩减位宽。当预留给当前 block 的显存预算被占满时，通过位宽缩减操作将已存储的 KV Cache 从当前位宽降一档（16→8→4→2 bit），腾出空间后继续以新的低位宽存储后续 token。最终位宽 Fbit 由块级内存分配确定。

与 baseline（KIVI 等在每个 step 直接将 KV Cache 量化为目标位宽）相比，渐进量化充分利用了生成初期的空闲显存。例如在 Fbit=2、max context=32K 的场景下：前 ~2K token 以 16-bit 存储（零量化误差）→ 随后 ~2K token 以 8-bit 存储（极低误差）→ 后续 ~4K token 以 4-bit → 最后部分以 2-bit 存储。由于 long-CoT 推理的误差传播特性，前期 token 的精度比后期更重要——渐进策略恰好保证了这一需求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

渐进量化在 LLM 推理 decoding 阶段的执行流程：

```
# 对每个 transformer block，目标 Fbit 已知
# KV Cache 按位宽分段存储: segment_16, segment_8, segment_4, segment_2

for each decoding step t:
    K_new, V_new = compute_KV(current_token)  # FP16
    
    current_usage = Σ memory_of(all segments)
    
    if current_usage + FP16_token_size > budget:
        # 找当前最高位宽的非空段并降档
        if segment_16 not empty:
            segment_8_new = equivalent_right_shift(segment_16, 16→8)
            segment_8 = concat(segment_8, segment_8_new)
            segment_16 = empty; current_bit = 8
        elif segment_8 not empty:
            segment_4_new = equivalent_right_shift(segment_8, 8→4)
            segment_4 = concat(segment_4, segment_4_new)
            segment_8 = empty; current_bit = 4
        elif segment_4 not empty and Fbit == 2:
            segment_2_new = equivalent_right_shift(segment_4, 4→2)
            segment_2 = concat(segment_2, segment_2_new)
            segment_4 = empty; current_bit = 2
    
    store(K_new, V_new, bit=current_bit)
```

**Annotations**: 位宽档位按 2 的幂次排列。降档触发条件为 `current_usage + new_token > budget`。整个推理过程中仅执行 log2(16/Fbit) 次降档（每次影响大量已有 token），而非每个 step 降档。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

位宽缩减通过 Equivalent Right Shift 实现（纯整数乘加+移位）。渐进量化与块级内存分配正交：每个 block 有独立的 Fbit 和独立的渐进量化过程。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

## Block-wise Memory Allocation for KV Cache (块级KV缓存内存分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-wise Memory Allocation 是 PM-KVQ 提出的将 KV Cache 量化位宽按 transformer block 粒度非均匀分配的技术。核心观察：不同深度的 transformer block 对 KV Cache 量化误差的敏感度差异显著——深层 block 以及 Qwen 架构的首个 block 比浅层 block 敏感数倍。Uniform bit-width 无法利用这一差异，在显存充足但不足以全局升档时造成浪费。

PM-KVQ 通过 Integer Programming 形式化块级分配问题：`min Σ_i Σ_b x_{i,b}·s_{i,b}` s.t. `Σ_i Σ_b x_{i,b}·Mem(Q_b(K_i)+Q_b(V_i)) ≤ M`，其中敏感度 `s_{i,b} = ||G_{K_i} ⊙ (K_i - Q_b(K_i))||_1 + ||G_{V_i} ⊙ (V_i - Q_b(V_i))||_1` 使用一阶 Taylor 近似估计。CVXPY 在数秒内求解，为敏感 block 分配更高 Fbit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# === 离线阶段 ===
# 输入: N 个 transformer blocks, 校准数据, 可选位宽集合 B, 显存预算 M

for each block i = 1 to N:
    前向传播得 K_i, V_i
    反向传播得梯度 G_K_i, G_V_i
    for each bit-width b in B:
        Q_b_K = fake_quant(K_i, bit=b, group=128)
        Q_b_V = fake_quant(V_i, bit=b, group=128)
        s_{i,b} = ||G_K_i ⊙ (K_i - Q_b_K)||_1
                + ||G_V_i ⊙ (V_i - Q_b_V)||_1

# IP 求解
CVXPY.solve(
    minimize Σ_i Σ_b x_{i,b} * s_{i,b}
    s.t. Σ_b x_{i,b} = 1,  x_{i,b} ∈ {0,1}
         Σ_i Σ_b x_{i,b} * Mem(b) ≤ M
)
# 输出: per-block Fbit (如 block 1→4bit, block 28→4bit, 其余→2bit)
```

**Annotations**: 敏感度使用 L1 范数聚合 per-element 梯度×量化误差。IP 求解耗时 < 5s per model。一阶 Taylor 假设 FP16 附近局部线性，在 2-4 bit 量化下基本成立。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现分三步：(1) 校准阶段逐 block 计算敏感度矩阵；(2) 调用 CVXPY 求解 IP；(3) 推理时每个 block 独立执行渐进量化至分配的 Fbit。该方法的一个局限是敏感度基于单一校准集估计，对分布偏移敏感。当 batch size 减少（单样本显存增加）时，块级分配可将多余显存定向分配给敏感 block，提升 0.84% pass@1。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

## Calibration with Positional Interpolation for KV Cache Quantization (位置插值校准)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PM-KVQ 提出的校准策略，用于解决短上下文校准数据无法准确估计 long-CoT LLM 的 Key Cache 分布的问题。根源在 RoPE：低频通道（`θ_i = θ^{-2i/d}` 随 i 增大指数衰减）的周期可达数万 token（DeepSeek-R1-Qwen-7B 最低频通道周期 54,410 tokens）。短校准数据（512-2048 tokens）仅覆盖正弦曲线的一个小片段，导致 channel-wise reparameterization factor `λ_i = (max_m K_{m,i})^α` 校准偏差。

PM-KVQ 在校准的 RoPE 中引入位置缩放：`cos(s·mθ_i)` 替代 `cos(mθ_i)`。s=4 使 2048 token 模拟 8192 token 的位置分布，低频通道在短数据中展露出完整周期。消融：2048+s=4 的 pass@1 (48.33%) 与 8192 无插值 (48.33%) 持平，超 2048 无插值 (46.67%)。

从算法pipeline角度拆解术语：校准脚本中修改 RoPE 计算 `angle = s * position_m * θ_i`（仅校准阶段）。校准完成后产出的 λ_i 和 per-block Fbit 在推理时直接使用（推理恢复 s=1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现无需修改模型结构——仅校准脚本中乘 s。s 的选择受校准长度和 target 长度影响：PM-KVQ 用 s=4（2048→8192），s 过大引入位置失真（s=16 降至 46.67%）。该方法继承自 Positional Interpolation (Chen et al. 2023) 但首次应用于 KV Cache 量化校准。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

## Equivalent Right Shift for KV Cache Bit-width Reduction (等价右移位宽缩减)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Equivalent Right Shift 是 PM-KVQ 中实现 KV Cache 位宽缩减的核心整数运算：将 2b-bit 量化整数降为 b-bit，结果等效于先反量化到 FP16 再重新量化为 b-bit——但全程仅用整数域乘加和移位。操作公式：`X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b`，参数更新 `Z_b = Z_{2b}`, `S_b = (2^b + 1)S_{2b}`。

三种策略消融（DeepSeek-LLaMA-8B, AIME-2024 pass@1）：Direct Right Shift 12.08% < Modified Right Shift 28.75% < Equivalent Right Shift 38.33%（对比 FP16 44.17%）。

从算法pipeline角度拆解术语：
- 16→8: `X_8 = (65281*(X_16+128)) >> 24`, `S_8 = 257*S_16`
- 8→4: `X_4 = (241*(X_8+8)) >> 12`, `S_4 = 17*S_8`
- 4→2: `X_2 = (13*(X_4+2)) >> 6`, `S_2 = 5*S_4`

术语一般如何实现？可使用 GPU 整数张量操作或 CPU 标准整型指令实现，无需浮点单元或自定义 CUDA kernel。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---

## Per-Frame Quantization（逐帧量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Per-Frame Quantization（逐帧量化）是一种针对多帧视频增强模型的量化策略，为输入张量中每个帧独立分配量化裁剪边界（lb_i, ub_i），而非对所有帧使用统一的 per-tensor 量化参数。其动机来自于多帧视频增强 Transformer 中不同帧的激活分布存在显著差异——网络对不同帧分配不同的注意力权重，导致各帧的激活值范围、分布形态（对称/非对称、长尾程度）各不相同。使用统一的 per-tensor 量化范围会造成某些帧的动态范围利用不足（过宽裁剪→分辨率浪费）或截断过多（过窄裁剪→信息丢失），而逐帧量化使每帧获得适配其自身激活统计的最优量化分辨率。

PMQ-VE 论文通过实验统计展示了这一分布差异：对 RSTT（STVSR）中各帧的激活值分别绘制分布图，发现不同帧的 min/max 激活值差异可达数倍。采用 per-frame 量化后，每帧独立搜索 [lb_i, ub_i]，保证量化误差在帧间均匀分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 PMQ-VE 的 BMFQ 阶段，per-frame 量化的具体流程如下：

```
输入: 多帧激活张量 X ∈ R^{N×C×H×W}，N 帧
输出: 每帧的量化边界 {(lb_i, ub_i)}_{i=1..N}

for i = 1 to N:                          # 逐帧处理
    X_i = X[i, :, :, :]                  # 第 i 帧的激活
    # 百分位初始化（抑制 outliers）
    lb_0 = percentile(X_i, 0.1)          # 下界初始值
    ub_0 = percentile(X_i, 99.9)         # 上界初始值
    # 在搜索空间 [p0.1, p10] × [p90, p99.9] 内查找最优 (lb, ub)
    (lb_i, ub_i) = BTBI(X_i, lb_0, ub_0, ΔL, ΔU)
    # 执行逐帧量化
    X̂_i = fake_quantize_per_channel(X_i, lb_i, ub_i, N_bits)

# 后续层的计算使用量化后的 X̂ = [X̂_1; X̂_2; ...; X̂_N]
```

与传统的 per-tensor quantization（所有帧共享同一对 [lb, ub]）对比，逐帧量化使每帧的量化误差 ||X_i - X̂_i||_2 独立最小化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

逐帧量化的实现不需要特殊的硬件支持——在 fake quantization 框架下，每帧的裁剪边界仍然在 PyTorch 张量操作层面实现。与 per-tensor 量化的唯一区别在于搜索/优化阶段为每帧独立维护一组 (lb_i, ub_i)，前向推理时对每帧独立执行 clamp+round+dequantize。在 PMQ-VE 的实现中，逐帧量化仅应用于激活（权重仍使用 per-channel 量化），且仅对注意力模块中的关键线性层和 MatMul 层执行。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---

## BMFQ（Backtracking-based Multi-Frame Quantization / 回溯式多帧量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

BMFQ（Backtracking-based Multi-Frame Quantization）是 PMQ-VE 框架粗阶段（coarse stage）的核心方法，负责为多帧视频增强模型中的每个帧独立估计最优量化裁剪边界。BMFQ 包含两个关键设计：(1) **逐帧百分位初始化**——对每帧激活 X_i，将搜索空间限定在 [p_{0.1}(X_i), p_{10}(X_i)] × [p_{90}(X_i), p_{99.9}(X_i)]，以此抑制长尾分布中的 outliers；(2) **BTBI 回溯搜索算法**——从百分位初始点出发，递归探索候选 (lb, ub) 配置，通过评估量化误差 ||X_i - X̂_i||_2 指导搜索方向，在遇到误差上升时回溯到上一节点尝试其他方向（剪枝+回溯），终止条件为所有候选被评估或满足收敛阈值 ε。

BMFQ 相比传统 PTQ 边界搜索方法（如 2DQuant 的对称/非对称单向搜索、DBDC+Pac 的顺序调整）的优势在于：(a) 不受 uniform shrink 方向限制，探索更丰富的候选空间；(b) 百分位初始化使搜索起点不受 outliers 影响；(c) 回溯机制允许算法从局部极值逃逸。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

BTBI 伪代码（BMFQ 核心搜索算法）：

```
输入: X_i (第 i 帧激活), 步长 ΔL, ΔU, 收敛阈值 ε
输出: 最优 (lb*, ub*)

# 百分位初始化搜索空间
lb_0 = p_{0.1}(X_i)          # 下界起始于 0.1 分位数
ub_0 = p_{99.9}(X_i)         # 上界起始于 99.9 分位数
visited = {}                  # 已访问节点集合
error_min = +∞

Function Backtrack(lb, ub):
    if (lb, ub) in visited or lb < p_{0.1} or lb > p_{10}
       or ub < p_{90} or ub > p_{99.9}:
        return                                           # 越界剪枝
    visited = visited ∪ {(lb, ub)}
    X̂_i = QuantizeDequantize(X_i, lb, ub)               # uniform quantizer
    err = ||X_i - X̂_i||_2                                 # MSE 评估
    if err > error_min + ε:
        return                                           # 误差剪枝
    if err < error_min:
        error_min = err; lb* = lb; ub* = ub             # 更新最优
    foreach (δ_l, δ_u) in {(±ΔL, 0), (0, ±ΔU)}:         # 四个搜索方向
        Backtrack(lb + δ_l, ub + δ_u)                    # 递归搜索

Backtrack(lb_0, ub_0)
return (lb*, ub*)
```

搜索复杂度为 O(|S|)，其中 S 为搜索空间网格点数。由于剪枝和 visited 集合的存在，实际搜索节点数远小于网格点总数。

在 PMQ-VE 的完整流程中，BMFQ 仅需少量校准数据（无需标签），运行在蒸馏微调之前，为 PMTD 提供合理的初始量化边界。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

BMFQ 在 PyTorch 中实现，使用 fake quantization 模拟量化效果。百分位计算通过 `torch.quantile` 完成，量化误差评估在 FP32 精度下进行（反量化后的值与原始值比较）。BMFQ 运行在校准阶段，仅使用训练集中少量样本（如 Vimeo-90K 的子集），不需要反向传播或梯度计算——所有搜索基于前向 MSE 评估。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

消融实验（STVSR 2-bit）：Baseline（MinMax，无 BMFQ，无 PMTD）12.67dB → +Per-Frame 19.64dB → +BMFQ 27.56dB（+7.92dB）。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---

## PMTD（Progressive Multi-Teacher Distillation / 渐进式多教师蒸馏）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PMTD（Progressive Multi-Teacher Distillation）是 PMQ-VE 框架精阶段（fine stage）的核心方法，一种层次化知识蒸馏策略，用于恢复低比特量化模型的表示能力。与传统的单教师蒸馏（仅使用 FP32 教师监督低比特学生）不同，PMTD 引入中间比特教师（如 INT8）作为"桥梁"，使低比特学生（如 INT4/INT2）能够分阶段、逐步地从更容易的目标学习。

PMTD 的核心洞察在于：低比特模型（4-bit/2-bit）与全精度模型之间存在显著的"容量差距"（capacity gap），直接使用 FP 教师蒸馏使学生在有限容量下难以学习高质量映射。通过引入 INT8 中间教师——一个已经量化感知但在容量上更接近低比特学生的模型——PMTD 为学生提供了更可达的短期学习目标，再逐步过渡到 FP 教师的长远目标。

训练 4-bit 模型的 PMTD 流程：(1) 首先用 FP 教师蒸馏训练 8-bit 模型；(2) 训练 4-bit 模型时，同时使用 INT8 教师和 FP 教师进行监督。损失函数 `L_PMTD = (L_INT + α(t)·L_FP) / (1+α(t))`，其中 α(t) = min(1, t/T_warmup) 线性增长，使训练早期偏向 INT8 教师（学习更易达成的目标），后期逐步过渡到 FP 教师（追求更高精度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PMTD 训练 4-bit 模型的伪代码：

```
# 预训练 INT8 教师（阶段 1）
int8_model = quantize(fp_model, bits=8)
for iter in range(N1):
    x = batch_data()
    out_int8 = int8_model(x)
    out_fp = fp_model(x).detach()
    L = L2(out_int8, out_fp) + λ * MSE(feat_int8, feat_fp)
    L.backward(); update clippings via STE

# 训练 4-bit 学生（阶段 2，PMTD 核心）
int4_model = BMFQ_initialize(model, bits=4)   # 粗阶段初始化
for t in range(T):
    x = batch_data()
    out_4bit = int4_model(x)
    out_int8 = int8_model(x).detach()          # 中间教师
    out_fp = fp_model(x).detach()              # 全精度教师

    # 每个教师的损失 = 重建损失 + 特征匹配损失
    L_rec_int = ||out_4bit - out_int8||_2^2          # L2 输出损失
    L_feat_int = MSE(f_4bit, f_int8)                 # 中间特征 MSE
    L_INT = L_rec_int + λ * L_feat_int               # λ=5

    L_FP = ||out_4bit - out_fp||_2^2 + λ * MSE(f_4bit, f_fp)

    α(t) = min(1.0, t / T_warmup)                    # 线性增长权重
    L = (L_INT + α(t) * L_FP) / (1.0 + α(t))        # 加权组合

    L.backward()  # STE 梯度穿过量化
    optimizer.step()
```

关键超参数：λ=5（特征匹配权重），T_warmup 控制从 INT8 到 FP 监督的过渡速度，batch_size=2/GPU（因多教师增加显存）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PMTD 在 PyTorch 中实现，利用 fake quantization + STE 进行端到端优化。多教师的前向传播可以批量并行处理（教师模型不需要梯度），仅学生模型的量化参数（clipping bounds）通过 STE 更新。对于 2-bit 量化，PMTD 会引入额外的中间教师（如 4-bit），形成 2bit←4bit←8bit←FP 的三级层次。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---

## Percentile-based Initialization for Quantization Bounds（百分位数量化边界初始化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

百分位数量化边界初始化是一种用于 PTQ 中确定量化裁剪范围的技术：使用张量分布的百分位数（而非全局 min/max）作为量化裁剪边界的初始估计。在 PMQ-VE 中，BMFQ 阶段的下界 lb 初始化为 p_{0.1}（第 0.1 百分位数），上界 ub 初始化为 p_{99.9}（第 99.9 百分位数），搜索空间限定在 lb ∈ [p_{0.1}, p_{10}] 和 ub ∈ [p_{90}, p_{99.9}]。

其核心优势在于抑制长尾分布中的 outliers 对量化分辨率的影响：当激活张量存在极端的正值或负值 outliers 时（如 Transformer 中的 post-GELU 激活），使用 min/max 作为量化边界将导致大部分正常值被压缩到极少的量化 bin 中，造成严重的分辨率浪费。百分位初始化通过直接裁剪掉最极端的 0.1%-10% 值，使量化 bin 集中在激活分布的主体区域，显著降低 MSE。

与 MinMax（p_0 和 p_100）相比，百分位初始化的搜索范围更窄且更接近最优解，收敛更快。与仅使用 MSE 搜索边界但不使用百分位约束的传统方法相比，百分位初始化提供了一种抗离群的先验。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 PMQ-VE 的 BMFQ 中：

```
X_i ∈ R^{C×H×W}              # 第 i 帧的激活
p_vals = torch.quantile(X_i.flatten(), 
                        torch.tensor([0.1, 10.0, 90.0, 99.9]) / 100.0)
p_01 = p_vals[0]              # 0.1 百分位 → lb 下界
p_10 = p_vals[1]              # 10 百分位 → lb 上界
p_90 = p_vals[2]              # 90 百分位 → ub 下界  
p_999 = p_vals[3]             # 99.9 百分位 → ub 上界

# 搜索空间定义
S_i = {(lb, ub) | lb ∈ [p_01, p_10], ub ∈ [p_90, p_999]}

# 搜索起点
lb_0 = p_01, ub_0 = p_999
```

与其他方法的对比：MinMax 将搜索范围设为 [min, max]（无 outlier 抑制）；2DQuant 使用对称/非对称的单向 shrink 搜索（探索范围受限）；PMQ-VE 的百分位初始化 + 回溯搜索在不缩小有效搜索范围的同时避免了 outlier 污染。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 PyTorch 中使用 `torch.quantile(input, q)` 计算分位数；历史方法（如 Percentile [Li et al. 2019]）也使用百分位数但直接作为最终量化边界（无搜索优化），效果远低于搜索方法。PMQ-VE 将百分位仅作为搜索起点和边界约束，而非最终量化参数。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---

## Video Frame Interpolation（VFI / 视频帧插值）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Frame Interpolation（VFI，视频帧插值）是视频增强的核心任务之一，目标是根据两帧或多帧连续输入，合成出位于它们时间位置之间的中间帧，从而提升视频的时间分辨率（帧率）。VFI 的关键技术挑战在于：(1) 准确估计帧间的光流/运动轨迹，尤其是大运动、遮挡区域和非线性运动；(2) 合成逼真的中间帧纹理，避免伪影和模糊。VFI 广泛应用于慢动作视频生成、视频帧率上转换、视频压缩等场景。

从早期 CNN-based 方法（如 DVF、SuperSloMo、DAIN）依赖光流估计和 warp+合成 pipeline，发展到 Transformer-based 方法（如 EMA-VFI、BiFormer）通过自注意力直接建模长程时序依赖，显著提升了大运动场景下的插值质量。PMQ-VE 使用 EMA-VFI [Zhong et al. 2024] 作为 VFI 任务的 backbone 进行量化实验，EMA-VFI 通过提取运动和外观信息解决速度模糊性问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VFI 模型（以 EMA-VFI 为例）的典型 pipeline：

```
输入: I_0, I_1 ∈ R^{H×W×3}  (连续两帧)
目标: 输出 I_t (t ∈ (0,1) 时刻的中间帧)

# Step 1: 特征提取
F_0 = Encoder(I_0)           # 多尺度特征金字塔
F_1 = Encoder(I_1)

# Step 2: 运动和外观信息提取（EMA-VFI 核心）
motion_feat = MotionExtractor(F_0, F_1)    # 帧间注意力建模运动
appearance_feat = AppearanceExtractor(F_0, F_1)  # 外观信息聚合

# Step 3: 帧合成
warped_feat = BackwardWarp(F_0, F_1, motion_feat, t)
I_t = Decoder(warped_feat, appearance_feat)
```

PMQ-VE 量化 VFI 模型时，对 Encoder/Decoder 中的所有 Linear 和 MatMul 层执行逐帧量化（BMFQ → PMTD）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常用 VFI 开源实现：EMA-VFI（ECCV 2024, https://github.com/zhongzhihang/EMA-VFI）、RIFE（实时 VFI）、AMT（大规模运动 VFI）。评估数据集：Vimeo-90K（标准 VFI benchmark）、UCF101、SNU-FILM。评估指标：PSNR、SSIM、LPIPS。PMQ-VE 在 Vimeo-90K 测试集上评估，对 EMA-VFI [T] 和 [D] 两种变体分别测试 4-bit 量化性能。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---

## Video Super-Resolution (VSR / 视频超分辨率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Super-Resolution（VSR，视频超分辨率）是从低分辨率（LR）视频序列中利用多帧时序信息重建高分辨率（HR）视频的任务。与单图超分（SISR）不同，VSR 的核心优势在于跨帧聚合亚像素信息——相邻帧中同一物体可能存在微小的亚像素位移，合理利用这些信息可以从多帧低分辨率观测中恢复出超过单帧物理限制的高频细节。

VSR 的历史演进：早期方法使用显式光流估计+运动补偿+融合（如 VESPCN、TOFlow），后演进为可变形卷积对齐（EDVR、BasicVSR），再到最近的 Transformer-based 方法（PSRT、MIA）通过自注意力机制隐式建模时序依赖。PMQ-VE 使用 MIA [Zhou et al. 2024] 作为 VSR backbone 进行量化实验。MIA 采用 masked intra-frame attention 和 inter-frame attention 块，通过掩码注意力更好利用之前增强帧的特征。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VSR（以 MIA 为例）的典型 pipeline：

```
输入: {LR_t, LR_{t-1}, ..., LR_{t-N+1}}  (N 帧低分辨率序列)
目标: 输出 HR_t (当前时刻的高分辨率帧)

# Step 1: 特征传播与对齐
F_{t-1} = propagated_feature_from_previous_timestep
F_aligned = IntraFrameAttention(LR_t) + InterFrameAttention(F_{t-1}, LR_t)
# MIA 使用 masked attention: 仅关注最相关的帧间/帧内区域

# Step 2: 特征增强
F_enhanced = TransformerBlocks(F_aligned)  # 多层自注意力+FFN

# Step 3: 上采样重建
HR_t = Upsample(F_enhanced) + BicubicUpsample(LR_t)  # 残差学习
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常用 VSR 开源实现：BasicVSR/BasicVSR++（https://github.com/ckkelvinchan/BasicVSR）、MIA（CVPR 2024）、PSRT（NeurIPS 2022）。评估数据集：Vimeo-90K（主要用于 VSR 的训练和评估）、Vid4（经典 VSR benchmark，4 个视频片段）、REDS、UDM10。评估指标：PSNR、SSIM（Y 通道）。PMQ-VE 在 Vimeo-90K 和 Vid4 上评估 MIA 的 4-bit 量化性能，报告 PSNR/SSIM。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---

## Spatio-Temporal Video Super-Resolution (STVSR / 时空视频超分辨率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Spatio-Temporal Video Super-Resolution（STVSR，时空视频超分辨率）是视频增强中最具挑战性的任务之一，需要同时提升视频的空间分辨率（空间维度+）和时间分辨率（帧率，时间维度+）。也就是说，输入一个低分辨率、低帧率的视频序列，输出高分辨率、高帧率的视频序列。STVSR 结合了 VSR（空间维度超分）和 VFI（时间维度插帧）两个子问题，但两问题互相耦合——更好的空间细节有助于准确的帧间运动估计，反之更精确的运动信息也有助于多帧空间信息聚合。

PMQ-VE 使用 RSTT [Geng et al. 2022] 作为 STVSR backbone。RSTT 是一个实时 Transformer-based STVSR 模型，通过从编码器不同层构建特征字典（feature dictionary），在解码阶段反复查询该字典来同时增强空间和时间分辨率。其优势在于实时性（满足实际部署需求），但高计算量也使其成为量化的理想候选。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

STVSR（以 RSTT 为例）的 pipeline：

```
输入: {LR_1, LR_2, ..., LR_7}  (7 帧低分辨率、低帧率)
目标: 输出 {HR_t1, HR_t2, ..., HR_tM}  (高分辨率、高帧率序列)

# Step 1: 多级编码
F_level1, F_level2, F_level3 = PyramidEncoder(LR_frames)
# 构建多级特征字典
Dict = StackDict(F_level1, F_level2, F_level3)

# Step 2: 字典查询解码
for each output frame t:
    # 从特征字典中查询对应时空位置信息
    Q_t = LearnableQuery(t)              # 可学习的位置编码查询
    F_t = CrossAttn(Q_t, Dict)           # 跨注意力查询字典
    # 多层解码
    for level in [1, 2, 3]:
        F_t = DecoderBlock(F_t, Dict[level])
    HR_t = ReconstructionHead(F_t)       # 空间上采样+帧合成
```

RSTT 中所有 Linear 层（Q/K/V 投影、FFN 的 FC1/FC2）和 MatMul 层（Q@K^T、Attn@V）均被 PMQ-VE 量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常用 STVSR 方法：Zooming Slow-Mo（首个端到端 STVSR）、TMNet（时空调制网络）、RSTT（实时 Transformer STVSR, CVPR 2022）、STDAN（可变形注意力 STVSR）。评估数据集：Vimeo-90K（含 Vimeo-Fast、Vimeo-Medium、Vimeo-Slow 三个子集，按运动速度分级）、Vid4。评估指标：PSNR、SSIM（Y 通道）。PMQ-VE 在以上四个 benchmark 上均达到最优 4-bit 和 2-bit 量化性能。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---

## Extremely Low-Bit Post-Training Quantization（极低位后训练量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
极低位后训练量化（Extremely Low-Bit PTQ）指将 LLM 权重量化到 sub 2-bit（即有效位宽 <2 bit/权重）的后训练量化方法。与 4-bit/8-bit PTQ（如 GPTQ、AWQ、OmniQuant）不同，极低位 PTQ 面临的核心挑战是：量化误差在 1-2 bit 范围内急剧放大，二值化（1-bit）是其中最极端的形式。极低位 PTQ 的典型策略包括：(1) **混合精度**——部分显著权重保留较高位宽（4-bit/8-bit），其余二值化；(2) **结构化掩码**——用结构化方式标记显著权重以减少掩码存储开销；(3) **缩放因子优化**——通过可学习或迭代方式优化二值化缩放因子；(4) **预处理/后处理**——量化前后通过微调或变换改善权重分布。PTQ1.61 首次将有效位宽真正降至 1.61-bit（不含非结构化掩码开销），而此前 PB-LLM (2.7-bit) 和 BiLLM (2.1-bit) 虽声称 sub 2-bit，实际因非结构化掩码额外占用 ≥1 bit 而超出 2-bit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 PTQ1.61 的极低位量化流程为例：
```
# 输入: 预训练 LLM 权重 W，校准数据 X_calib
# 目标: 量化权重 W_q，平均位宽 < 2 bit/权重

# Step 1 (可选): 量化预处理
W_preprocessed = W + LoRA(W, pretraining_data)

# Step 2: 结构化掩码识别显著权重
for each linear layer W_layer:
    X_act = forward(X_calib, layer)
    ch_mag = ||X_act|| per channel
    salient_channels = topk(ch_mag, 20%)  # 一维掩码
    mask = [1 if i in salient_channels else 0 for i in range(m)]

# Step 3: 混合精度量化
W_q[salient] = round(W[salient] / S_q) + Z_q   # 4-bit
alpha = learnable_init(||w_row||_1 / n_w)
W_q[non-salient] = alpha * sign(W[non-salient])  # 1-bit

# Step 4: 分块优化缩放因子
for epoch in range(20):
    loss = MSE(out_fp, out_q) + (-log(cos_sim(out_fp, out_q)))
    alpha = AdamW(loss, lr=5e-4).step()
```
位宽计算（4096x4096 权重矩阵）：权重位宽 = 0.8x1 + 0.2x4 = 1.6-bit；掩码 = 4096/(4096x4096x(0.8+0.2x4)) ≈ 0.0002-bit；量化参数 ≈ 0.008-bit；合计 1.61-bit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
极低位 PTQ 目前主要在 PyTorch 上用 fake-quantization 做精度仿真，因为商用 GPU 尚不支持 sub 4-bit 整数推理。PB-LLM 和 BiLLM 使用逐元素非结构化掩码（需额外 1-bit bitmap），PTQ1.61 用一维结构化掩码（额外 0.0002-bit）。关键指标：WikiText2/C4 困惑度 + 推理 benchmark 零样本准确率（lm-evaluation-harness）。开源：https://github.com/zjq0455/PTQ1.61。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- PB-LLM Partially Binarized Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---

## Activation-Based Structured Mask for Quantization（基于激活的结构化量化掩码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于激活的结构化量化掩码是 PTQ1.61 提出的用于极低位量化的显著权重识别方法。与 PB-LLM/BiLLM 的逐元素非结构化掩码（每权重 1-bit bitmap, shape mxn）不同，结构化掩码按权重矩阵的**行**（对应输入激活通道）标记显著通道，掩码形状为 mx1，额外存储开销从 ≥1-bit 降至 0.0002-bit。核心推导（Eq. 3-4）：量化误差 E ≤ Σ_i (|x_i| x Σ_j |w_{i,j}^q - w_{i,j}|)，其中 |x_i| 为第 i 通道输入激活幅值。激活幅值约为权重的 1000 倍（尤其 top-20% 通道），因此保留高幅值激活通道对应的权重行为 4-bit 可最大程度降低量化误差上界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 对每个线性层 W in R^{n x m}:
X = forward_pass(X_calib, layer)            # [t, m]
ch_mag = ||X[:, i]|| for i in 1..m          # 每通道幅值 [m]
k = int(m * 0.2)
salient_idx = topk(ch_mag, k)               # top-20% 通道
mask = zeros(m); mask[salient_idx] = 1      # 一维掩码

for i in range(m):
    if mask[i]: W_q[:, i] = 4bit_quant(W[:, i])
    else:       W_q[:, i] = alpha_i * sign(W[:, i])
```
salient_ratio=20% 而非 30%：因为 30% 下位宽升至 1.91-bit，违反 sub 2-bit 定义。掩码基于激活幅值而非 Hessian（如 OWQ），因为 Hessian 近似在极低位下误差被放大。消融实验（Table 5）证实 OWQ Hessian 掩码替换后 LLaMA-7B PPL 从 12.50 崩溃至 22.11。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
掩码存储：一维 4096 bits (512 bytes) vs 非结构化 4096x4096 bits (2 MB) per layer，压缩 4096 倍。与 AWQ 区别：AWQ 用激活-权重关系做 per-channel scaling（grid search），没有掩码。与 OWQ 区别：OWQ 用 Hessian+Cholesky 做列选择保留 FP16，涉及多层近似。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models

---

## Quantization Preprocessing via Restorative LoRA（基于恢复性LoRA的量化预处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
基于恢复性 LoRA 的量化预处理是 PTQ1.61 提出的新范式：量化前用轻量级 LoRA 微调，目标是将预训练模型中散乱的显著权重分布改写为行集中模式，使模型更适配逐通道 PTQ。动机：逐通道 PTQ 按行分配量化参数，但预训练模型显著权重呈散乱分布，行内方差大，同参数难以同时覆盖高/低幅值元素。LoRA 低秩补偿 ΔW=BA 被吸收进权重，将显著权重推向集中行模式。与 QLoRA/QA-LoRA 等后量化 PEFT 区别：(1) 目的——优化量化而非下游任务；(2) 无需存储额外 FP 矩阵；(3) 用预训练数据（RedPajama）而非下游微调数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: M_FP (FP16), RedPajama, rank=64, steps=20K
M_q = extremely_low_bit_quantize(M_FP)
for layer in M_q:
    layer.lora_A = Linear(d_in, 64); layer.lora_B = Linear(64, d_out)
for step in range(20000):
    x = next_batch(RedPajama)
    loss = MSE(M_q(x), M_FP(x))  # teacher-student recovery
    update(lora_A, lora_B)
W_preprocessed = W + lora_B @ lora_A  # 融合 LoRA
W_final_q = PTQ1.61(W_preprocessed)   # 在预处理模型上 PTQ
```
LLaMA-13B WikiText2 PPL：直接量化 14.22 vs 预处理后量化 9.67（提升 4.55 PPL）。通用性：GPTQ/OmniQuant 等 baseline 使用预处理同样提升（Figure 5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
资源：LLaMA-7B <1.2h（单 A100）。局限性：增加预处理时间（总 PTQ 2h vs OmniQuant 1.1h）；FP16 模型本身性能轻微退化。适合极低位 PTQ 性能瓶颈场景，作为可选增强模块。开源：https://github.com/zjq0455/PTQ1.61。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- LoRA: Low-Rank Adaptation of Large Language Models

---

## Angular Bias in Binarization / NLC Loss（二值化中的角度偏差 / 负对数余弦相似度损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
二值化中的角度偏差（Angular Bias）指二值化权重与原始 FP 权重在方向（非仅幅值）上存在偏差。传统方法仅用 MSE loss 最小化幅值差距，但方向不同也会导致与激活相乘后输出差异。NLC Loss（Negative Logarithm of Cosine Similarity Loss）= -log(cos_sim(f₁, f₂))，其中 cos_sim = (f₁·f₂)/(||f₁||x||f₂||)。PTQ1.61 联合 MSE + NLC 作为分块优化目标：E(f₁, f₂) = ||f₁-f₂||₂ + (-log(cos_sim(f₁, f₂)))。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PTQ1.61 分块优化的损失计算
W_q_prime = (alpha_r1 x alpha_r2) * (alpha_s * sign(W))  # Eq. 9
out_fp = block_fp(X); out_q = block_q(X, W_q_prime)

# 联合损失 (Eq. 5-7)
loss_mse = ||out_fp - out_q||_2
cos_sim = (out_fp · out_q) / (||out_fp|| x ||out_q||)
loss_nlc = -log(cos_sim)
total_loss = loss_mse + loss_nlc  # 分支1
# 分支2: 同样联合损失，输入为量化激活 X_q
```
NLC 特性：cos_sim→1 时 NLC→0（方向一致）；cos_sim→0 时 NLC→∞（方向正交，强惩罚）；cos_sim→-1 时 MSE 主导。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
消融（Table 7, LLaMA-1-7B）：带角度偏差 PPL=12.50 vs 不带 13.56（WikiText2）。NLC 在 block 输出级计算（非权重级），因为目标是对齐 block 输出方向。基于 CBQ 分块框架增强。RBNN (Lin et al. 2020) 首次指出角度偏差问题并使用旋转矩阵纠正；LRQuant (Zhao et al. 2024) 在 PTQ 中引入余弦相似度考量。

涉及论文标题：
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization
- RBNN: Rotated Binary Neural Network
- LRQuant: Learnable and Robust Post-Training Quantization for Large Language Models

## ARVG (AutoRegressive Visual Generation / 自回归视觉生成)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ARVG（AutoRegressive Visual Generation）是一类将自回归生成范式应用于视觉生成任务的模型家族。受 LLM 中自回归生成的成功启发，ARVG 模型采用与 LLM 兼容的 Transformer 架构，通过逐 token 预测的方式生成图像。核心架构由 L 个 block 组成，每个 block 包含 Multi-Head Self-Attention（MHSA）、Feedforward Network（FFN）和 Adaptive LayerNorm（AdaLN）。AdaLN 将 conditioning 信息（包括类别标签和位置编码）转换为 shift 和 scale 参数来调整激活分布，从而保持生成 token 之间的双向依赖关系和条件引导。与 LLM 的关键区别：(1) ARVG 预测固定数量的 token（图像分辨率决定了 token 序列长度）；(2) ARVG 以条件信息作为初始 token（sink token）；(3) 不同 ARVG 模型在 token 预测粒度上有差异——VAR 一次预测一个 scale 的所有 token，RAR 一次生成一个 token，PAR 先顺序预测一个 token 再并行预测多个非局部 token，MAR 一次预测多个随机 token。代表模型包括 VAR（2B）、RAR（1.5B）、PAR（3B）、MAR（1B）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 VAR 模型的推理 pipeline 为例（一个 scale 的 token 生成）：

```
输入: 类别标签 c, 位置编码 p
输出: 图像 tokens

# Step 1: 编码 conditioning
cond = encode(c) + encode(p)  # 类别信息 + 位置信息

# Step 2: 初始化（sink token）
x_0 = cond  # 条件信息作为初始 token

# Step 3: 逐 block 处理（L 个 block）
for block l = 1 to L:
    # AdaLN: conditioning 调整激活分布
    shift_attn, scale_attn, shift_ffn, scale_ffn = AdaLN_l(cond)

    # Multi-Head Self-Attention
    x_norm = LayerNorm(x)
    x_scaled = x_norm * scale_attn + shift_attn          # AdaLN 调制
    attn_out = MHSA(Q=x_scaled, K=x_scaled, V=x_scaled)  # 自注意力
    x = x + attn_out                                      # 残差连接

    # Feedforward Network
    x_norm = LayerNorm(x)
    x_scaled = x_norm * scale_ffn + shift_ffn            # AdaLN 调制
    ffn_out = FFN(x_scaled)                               # 两层 MLP + 激活
    x = x + ffn_out                                       # 残差连接

# Step 4: 预测 next-scale tokens
tokens = output_head(x)  # 线性投影 + softmax
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ARVG 模型使用与 LLM 兼容的 Transformer 架构，通常在 ImageNet 上训练，使用交叉熵损失进行 next-token prediction。推理时支持 KV Cache 加速，但由于 AdaLN 的存在，每个 block 的激活分布由 conditioning 动态调整，导致量化面临三大挑战：(1) channel-wise outlier（AdaLN 调制后的激活存在严重的通道间范围差异）；(2) token-wise 动态激活（位置嵌入随 token 位置变化，首 token 为含关键条件信息的 sink token）；(3) sample-wise 分布不匹配（跨样本的激活高度相似，导致校准冗余）。这些特点也是 PTQ4ARVG 论文的核心动机。ARVG 代表模型 VAR、RAR、PAR、MAR 的开源代码分别在其官方 GitHub 仓库中。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

## Gain-Projected Scaling (GPS / 增益投影缩放)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPS（Gain-Projected Scaling）是 PTQ4ARVG 提出的首个基于数学优化（而非经验设计）的量化 scaling 策略。核心思想是通过 Taylor 展开量化损失，将 scaling 对量化的影响量化为 gain function g(s)，然后通过求导得到闭式最优解。具体步骤：(1) 将激活-权重量化损失 E(x,W) 分解为激活量化损失 E_x 和权重量化损失 E_W 的上界；(2) 用 Taylor 展开近似 E_x 和 E_W（以 MSE 替代 Hessian）；(3) 引入 per-channel scaling factor s，分析 scaling 后 E'_x < E_x（激活量化损失降低）和 E'_W > E_W（权重量化损失增加）；(4) 定义 scaling gain g(s) = E_x - E'_x - (E'_W - E_W)；(5) 对 g(s) 求导得闭式解 s_i = s_k · √(Σ_j |ΔW_{i,j}·x_i|) / √(Σ_j |W_{i,j}·Δx_i|)，其中 s_k 为激活 range 最大通道的 scaling factor。GPS 是 Equivalent Scaling 的数学优化版本，与 SmoothQuant 等经验方法不同，GPS 提供了理论保证的最优解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 RAR-B 的 qkv 线性层（X∈R^{T×n}, W∈R^{n×m}）为例：

```
输入: 激活 X (校准数据), 权重 W, bit-width b
输出: 最优 scaling factor s ∈ R^n

# Step 1: 准备量化误差数据
X_q = uniform_quantize(X, b)    # 模拟量化
W_q = uniform_quantize(W, b)
ΔX = |X - X_q|                   # 激活量化误差 (per-channel)
ΔW = |W - W_q|                   # 权重量化误差 (per-channel)

# Step 2: 找激活 range 最大的通道
R_x[i] = max(X[:,i]) - min(X[:,i]), i=1..n
k = argmax(R_x)                  # 激活 range 最大的通道索引

# Step 3: 计算 s_k (使该通道激活和权重 range 对齐)
s_k = sqrt(R_x[k] / R_W[k])

# Step 4: 闭式解计算其余 scaling factors (Eq. 16)
for i = 1 to n:
    if i != k:
        num = sum_{j=1}^{m} |ΔW[i,j] * X[i]|   # 权重量化误差 × 激活值
        den = sum_{j=1}^{m} |W[i,j] * ΔX[i]|    # 权重值 × 激活量化误差
        s[i] = s_k * sqrt(num / den)

# Step 5: 应用等效缩放并融合 (Eq. 2)
X' = X / s                       # 激活除以 s (推理时不执行)
W' = W * s.unsqueeze(1)          # 权重乘以 s (离线融合到 AdaLN 权重)
```

GPS 的关键数学性质：(1) 当 s>1 且 s_i > s_j（基于 Remark 1 的统计观察），scaling 使激活量化损失降低而权重量化损失增加；(2) g(s) 是凸函数，求导后得到的 s_i 是全局最优解；(3) GPS 仅应用于 qkv 和 fc1 层（scaling factor 可被吸收的层），且 scaling factor 离线融合实现零推理开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPS 的实现关键是：(1) Remark 1 的统计验证——当 R_x[i] > R_x[j] 时，超过 98% 的通道满足 s_i > s_j，超过 99.5% 的通道满足 R_x[i]/s_i > R_x[j]/s_j，这保证了 scaling 后激活 range 的相对顺序不变；(2) 逐通道独立计算 s_i，无需迭代训练或反向传播（vs OmniQuant 需数小时训练）；(3) 计算复杂度低——仅需一次前向量化获取 ΔX 和 ΔW，然后逐通道计算 s_i；(4) GPS 可作为 plug-and-play 组件集成到任意量化框架。PTQ4ARVG 论文的实验表明，GPS 在 RAR-B W6A6 上将 FID 从 SmoothQuant 的 63.77 降至 36.51（-42.7%），优于所有经验 scaling 方法（OS+, RepQ*, SQ+RepQ*）。开源代码：https://github.com/BienLuky/PTQ4ARVG。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

## Static Token-Wise Quantization (STWQ / 静态逐Token量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
STWQ（Static Token-Wise Quantization）是 PTQ4ARVG 提出的一种离线分配 per-token 量化参数的方法，专门利用 ARVG 模型的两大独有特性：(1) 固定 token 序列长度（ARVG 生成固定数量的图像 token，不像 LLM 生成可变长度文本）；(2) 跨样本位置不变分布（不同类别和条件的样本在同一 token 位置的激活分布保持一致）。基于这两点，STWQ 在离线校准时为每个 token 位置静态设定量化参数（scale δ 和 zero point z），推理时直接使用而无需在线校准。具体包括两部分：(a) 对 AdaLN 模块输入沿 token 序列逐位置分配量化参数；(b) 对线性层输入将首 token（sink token，含条件信息，分布显著不同于其他 token）与其他 normal token 分开量化。使用 percentile 校准而非 min-max 以保证精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 VAR-d16 模型的 STWQ 离线校准流程为例（W6A6）：

```
输入: 校准数据 X_cal (128 张 ImageNet), ARVG decoder
输出: 静态 per-token 量化参数

# === AdaLN 模块的 STWQ ===
for each block l in decoder:
    X_adaln = run_forward_get_adaln_input(X_cal)  # shape: [N_samples, T, C]

    # 沿 token 维度逐位置校准
    for t = 1 to T (固定的 token 序列长度):
        # 收集所有样本在位置 t 的激活
        X_t = X_adaln[:, t, :]                     # shape: [N_samples, C]

        # Percentile 校准 (而非 min-max)
        low = percentile(X_t, p_low)               # 如 p_low = 0.1%
        high = percentile(X_t, p_high)             # 如 p_high = 99.9%
        δ_adaln[t] = (high - low) / (2^b - 1)
        z_adaln[t] = round(-low / δ_adaln[t])
    # 存储: {δ_adaln[t], z_adaln[t]}, t=1..T (用于推理)

# === 线性层的 STWQ (sink token + normal token) ===
for each linear layer (qkv, fc1, fc2, etc.):
    X_lin = run_forward_get_linear_input(X_cal)    # shape: [N_samples, T, C_in]

    # Sink token (首 token, t=0)
    X_sink = X_lin[:, 0, :]                        # shape: [N_samples, C_in]
    δ_sink = (percentile(X_sink, 99.9) - percentile(X_sink, 0.1)) / (2^b - 1)
    z_sink = round(-percentile(X_sink, 0.1) / δ_sink)

    # Normal tokens (其余 token, t=1..T-1)
    X_normal = X_lin[:, 1:, :]                     # shape: [N_samples, T-1, C_in]
    δ_normal = (percentile(X_normal, 99.9) - percentile(X_normal, 0.1)) / (2^b - 1)
    z_normal = round(-percentile(X_normal, 0.1) / δ_normal)

# 推理时直接使用预设参数 (无在线校准)
for each inference step t:
    if t == 0:
        δ, z = δ_sink, z_sink
    else:
        δ, z = δ_normal, z_normal  # (或 AdaLN 层的 δ_adaln[t])
    X_int = clamp(round(X / δ) + z, 0, 2^b - 1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
STWQ 与 Dynamic Token-Wise Quantization（DTWQ，如 LLM.int8）的本质区别：DTWQ 在每次推理时动态计算 min-max 量化参数，引入额外开销（LLM.int8 在 GPT-3-13B 上造成 0.5× speedup loss），且 min-max 校准精度低（VAR 上 DTWQ 导致 FID 降 15.3）；STWQ 利用 ARVG 的固定 token 长度和位置不变分布，将量化参数完全离线设定，推理时零额外开销。PTQ4ARVG 在 VAR-d16 W6A6 上的实验表明：STWQ 将 SmoothQuant baseline 的 FID 从 18.54 降至 10.41（+SQ+STWQ），且 speedup 保持 2.92×（vs DTWQ 的 2.46×）。STWQ 兼容标准 CUDA kernel 部署。开源：https://github.com/BienLuky/PTQ4ARVG。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

## Distribution-Guided Calibration (DGC / 分布引导校准)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DGC（Distribution-Guided Calibration）是 PTQ4ARVG 提出的一种基于分布熵的校准集选择方法。核心观察：ARVG 模型中网络激活跨样本高度相似（尤其无条件样本），这种 sample-wise 冗余导致随机采样校准集时量化参数的校准不匹配（mismatched calibration）。DGC 使用 Mahalanobis 距离 ρ(x) = √((x-u)^T S^{-1} (x-u)) 衡量每个样本对整体分布熵的贡献，其中 u 和 S 分别为校准池的均值和协方差矩阵。ρ(x) 越大，表示该样本与整体分布的差异越大，对分布熵的贡献越大。DGC 选择 top 50% 高熵样本作为最终校准集，消除冗余样本，确保校准集与真实分布匹配。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 校准池 X_pool = {x_1, ..., x_N} (N 张 ImageNet 图像)
输出: 校准集 X_cal (N/2 张高熵样本)

# Step 1: 计算校准池的分布统计量
u = mean(X_pool)         # 均值向量 (在特征空间)
S = cov(X_pool)          # 协方差矩阵

# Step 2: 计算每个样本的 Mahalanobis 距离
for i = 1 to N:
    d = x_i - u                          # 偏差向量
    ρ[i] = sqrt(d^T · S^{-1} · d)       # Mahalanobis 距离 (Eq. 17)

# Step 3: 选择高熵样本
sorted_indices = argsort(ρ, descending=True)
X_cal = {X_pool[i] for i in sorted_indices[0:N/2]}  # top 50%

# Step 4: 使用 DGC 选择的校准集进行 PTQ
quantize_model_with_calibration(X_cal)
```

Mahalanobis 距离 vs Euclidean 距离的关键区别：Mahalanobis 距离考虑了特征之间的相关性（通过协方差矩阵 S），能识别在相关结构下真正"异常"的样本，而非仅测量绝对距离。这使得 DGC 能有效识别对分布熵贡献大的样本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DGC 是一种校准集预处理方法，可插拔到任意 PTQ 流程。在 PTQ4ARVG 中，DGC 与 GPS 和 STWQ 协同使用：先用 DGC 选择 128 张高熵 ImageNet 图像，然后用 GPS 计算 scaling factor，最后用 STWQ 设定 per-token 量化参数。PTQ4ARVG 的实验表明：(1) DGC 在所有指标（IS/FID/sFID/Precision）上一致优于 random sampling 和 uniform sampling；(2) DGC 随校准集大小增大持续提供一致改进，展现强鲁棒性；(3) DGC 的消融实验中，加入 DGC 后的完整 PTQ4ARVG（GPS+STWQ+DGC）在 RAR-B W6A6 上将 FID 从 6.67（GPS+STWQ only）进一步降至 5.13。DGC 的计算开销很小——仅需对校准池计算一次均值和协方差。开源：https://github.com/BienLuky/PTQ4ARVG。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

## Adaptive LayerNorm (AdaLN / 自适应层归一化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaLN（Adaptive Layer Normalization）是 ARVG 模型和 DiT（Diffusion Transformer）中的核心模块，用于将 conditioning 信息（类别标签、时间步、位置编码等）动态注入到网络激活中。与标准 LayerNorm 不同，AdaLN 不仅做归一化，还通过 conditioning 生成 shift（β）和 scale（γ）参数来调整激活分布：output = LayerNorm(x) * γ(cond) + β(cond)。在 ARVG 中，AdaLN 的作用是保持预测 token 之间的双向依赖关系和条件引导——由于自回归生成中每个 token 只能看到之前的 token，双向依赖通过 conditioning 中的位置信息和类别信息来间接维护。AdaLN 通常在每个 block 中作用于 MHSA 之前和 FFN 之前：MHSA 输入 = LN(x) * MHSA_scale1(cond) + MHSA_shift1(cond)；FFN 输入 = LN(x_attn_out) * FFN_scale1(cond) + FFN_shift1(cond)。

从算法pipeline角度拆解术语，给出具体例子。
以 RAR 模型中一个 Transformer block 的 AdaLN 计算流程为例：

```
输入: x (隐层激活), cond (conditioning: 类别嵌入 + 位置嵌入)
输出: modulated_x

# AdaLN 生成调制参数 (通过线性投影)
MHSA_shift1, MHSA_scale1, MHSA_shift2, MHSA_scale2,
FFN_shift1, FFN_scale1, FFN_shift2, FFN_scale2 = AdaLN_proj(cond)
# 共 8 组参数，每组 shape: [T, C] (T=tokens, C=channels)

# === MHSA 路径 ===
x_norm = LayerNorm(x)
x_modulated = x_norm * (1 + MHSA_scale1) + MHSA_shift1   # 第一次调制
attn_out = MHSA(x_modulated)
attn_modulated = attn_out * (1 + MHSA_scale2) + MHSA_shift2  # 第二次调制
x = x + attn_modulated                                   # 残差连接

# === FFN 路径 ===
x_norm = LayerNorm(x)
x_modulated = x_norm * (1 + FFN_scale1) + FFN_shift1    # 第一次调制
ffn_out = FFN(x_modulated)                               # fc1 -> activation -> fc2
ffn_modulated = ffn_out * (1 + FFN_scale2) + FFN_shift2 # 第二次调制
x = x + ffn_modulated                                     # 残差连接
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaLN 对量化的影响：(1) AdaLN 生成的 scale/shift 参数引入了高度动态的激活分布——位置嵌入沿 token 维度变化导致 AdaLN 输入在 token 维度上高度动态；(2) AdaLN 的 scale 参数可能产生 channel-wise outlier（某些通道的 scale 远大于其他通道），这是 PTQ4ARVG 识别的第一个关键挑战；(3) AdaLN 不保持旋转不变性——与标准 LayerNorm 的 RMSNorm(XQ) = RMSNorm(X)Q 性质不同，AdaLN 的 LN(X) * scale + shift 形式破坏了旋转不变性，使得 QuaRot 的 Hadamard 旋转矩阵无法离线融合，必须在线计算，导致额外推理开销。在 PTQ4ARVG 的 GPS 方法中，scaling factor 被融合到 AdaLN 权重中以实现零推理开销。DiT 模型也使用 AdaLN，ViDiT-Q 方法同样通过在线旋转处理其量化。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

## Sink Token in ARVG (ARVG 中的汇令牌)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sink Token（汇令牌）在 ARVG 中的概念借鉴自 LLM 的 KV Cache 研究——在 LLM 中，Attention 的首 token 对模型性能高度敏感，被称为 "attention sink"。ARVG 中的 sink token 特指以条件信息（类别标签 + 位置编码）作为初始 token 的现象，且这种 sink token 的特殊性体现在：(1) 首 token 含关键类别信息，对条件生成起决定性作用；(2) 首 token 对所有后续 token 可见（因果注意力中首 token 参与所有后续 token 的 attention 计算），被训练为高度敏感的 token；(3) 首 token 的激活分布与其他 token 显著不同。PTQ4ARVG 进一步发现，与 LLM 仅在 Attention 中存在 sink token 不同，ARVG 中 sink token 出现在 MHSA 和 FFN 的所有线性层中。成因：(a) ARVG 固有地使用类别条件作为初始 token；(b) 初始 token 被模型训练为承载关键信息的枢纽；(c) 其分布与其他 token 的分布差异显著。

从算法pipeline角度拆解术语，给出具体例子。
Sink token 在 ARVG 量化中的影响（以 RAR-B 为例）：

```
# ARVG 的 token 序列结构 (T 个 token)
Token[0]: cond_embedding  # ← Sink Token (类别条件 + 位置编码)
Token[1]: image_token_1   # 正常图像 token
Token[2]: image_token_2
...
Token[T-1]: image_token_{T-1}

# 因果关系: Token[0] 对所有 Token[1..T-1] 可见
# 量化影响:
#   Token[0] 的激活分布: outlier 严重, range 大 → 需单独 high-precision 量化
#   Token[1..T-1] 的激活分布: 相对均匀 → 可共用一组量化参数
```

STWQ 的处理方式：对线性层输入，sink token (t=0) 和 normal tokens (t≥1) 分别使用独立的静态量化参数（δ_sink, z_sink）和（δ_normal, z_normal），从而在不引入在线开销的前提下处理 sink token 的分布特殊性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sink token 在 LLM 中被广泛研究（如 StreamingLLM 发现保留 attention sink 对长文本生成至关重要），但在 ARVG 中 PTQ4ARVG 是首次对其进行系统分析并用于量化优化。关键洞察：ARVG 的 sink token 不仅存在于 Attention，还扩散到 FFN 的所有线性层，这使得其影响范围比 LLM 更大。在实现中，识别 sink token 的方法是观察每层激活沿 token 维度的分布：首 token 的激活幅值和方差显著高于其他 token。PTQ4ARVG 的 STWQ 方法通过分离 sink token 和 normal token 的量化参数，在 W6A6 VAR-d16 上将 FID 从 18.54（无 token-wise）降至 10.41（SQ+STWQ）。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models

## Ternarization (Ternary Quantization, 三值量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ternarization（三值量化）是将神经网络权重的取值空间约束为 {-1, 0, +1} 三个离散值的极端压缩技术，等效位宽约为 log₂(3) ≈ 1.58 bit/权重。三值化的核心优势：(1) 存储压缩——仅需 2-bit 索引（编码 3 种状态，实际可更紧凑编码为 ~1.58 bit），远低于 FP16（16 bit）和常规 2-bit 量化（4 种状态的均匀量化）；(2) 计算效率——矩阵乘法 ŴX = (αT+μ)X 中，T 仅含 {-1,0,1} 使得乘法退化为加减法（T_ij=1 加 X_j，T_ij=-1 减 X_j，T_ij=0 跳过），消除绝大多数浮点乘法；(3) 表达能力——相比二值化（±1），三值化的零值 (0) 能更好地匹配 LLM 权重常见的单峰分布（大量权重接近零），自适应的 (α, μ) 参数进一步捕获幅值和偏移信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TWN（Ternary Weight Networks, Li et al. 2016）的经典对称三值化流程：
```
# 输入: W ∈ R^{n×m} (FP16 weights)
# 输出: α (scaling factor), T ∈ {-1,0,1}^{n×m}

for i in range(n):
    Δ_i = 0.75 * mean(|W[i,:]|)            # 阈值估计(假设均匀/正态分布)
    T[i,j] =  1  if W[i,j] >  Δ_i else \
             -1  if W[i,j] < -Δ_i else 0
    α_i = Σ_j(T[i,j] * W[i,j]) / Σ_j(|T[i,j]|)
# 推理: y ≈ α * (T @ x), T@x 仅需加法/减法/跳过
```
PT²-LLM 将对称三值化扩展为非对称：Ŵ = αT + μ，引入逐行偏移 μ 捕获非零均值权重分布。其 ATQ 将一次性阈值估计替换为 ITF+AGA 两阶段无训练优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
三值化在训练场景（QAT/TTQ/BitNet b1.58）中通过 STE（Straight-Through Estimator）在反向传播中近似梯度，前向使用三值权重。在 PTQ 场景中（如 PT²-LLM），三值化通过闭式解迭代优化（无需梯度反传），属于后训练压缩。BitNet.cpp（Wang et al. 2025）提供三值模型在 CPU 上的高效推理实现，利用三值乘法的计算特性（仅加减）实现加速。TereFiC（Yin et al. 2025）将三值推理部署到 FPGA。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models

## Asymmetric Ternary Quantization (非对称三值量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric Ternary Quantization（非对称三值量化）是对经典对称三值方案（TWN）的扩展，在标准的 Ŵ = αT（T∈{−1,0,+1}，α 为缩放因子）基础上引入逐行偏移参数 μ，将量化形式变为 Ŵ = αT + μ，使得三值网格从对称的 {−α, 0, +α} 扩展为非对称的 {−α+μ, μ, α+μ}。μ 初始化为行均值 μ = (1/m)ΣW_ij，捕获 LLM 权重分布中常见的非零均值偏差。对称方案强制假设权重均值为零，在 PTQ 场景下（权重不可通过训练重新分布）导致大量权重被错误映射。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
非对称三值量化的初始化步骤（以 PT²-LLM ATQ 为例）：
```
# 输入: W ∈ R^{n×m}
μ_i = (1/m) * Σ_j W[i,j]                   # 逐行偏移(行均值)
W̃[i,j] = W[i,j] - μ_i                      # 中心化权重
Δ_i = 0.75 * mean(|W̃[i,:]|)                # 阈值
T[i,j] =  1  if W̃[i,j] >  Δ_i else \
         -1  if W̃[i,j] < -Δ_i else 0
α_i = Σ_j(T[i,j] * W̃[i,j]) / Σ_j(|T[i,j]|)
# 去量化: Ŵ[i,j] = α_i * T[i,j] + μ_i
# 每行仅 3 个可能值: {−α_i+μ_i, μ_i, α_i+μ_i}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
非对称三值量化的实现要点：(1) 存储开销——α (n个fp16)、μ (n个fp16)、T (n×m 个 2-bit 索引)，总位宽 ≈ 1.58 + 32n/(nm) → ≈1.58 bit/权重；(2) 推理计算——ŴX = α(TX) + μ(1^T X)，TX 仅需加减法；(3) TernaryLLM (Chen et al. 2024) 也采用了非对称三值方案但属于 QAT 方法；PT²-LLM 是首次在 PTQ 场景下系统实现非对称三值量化。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models

## Iterative Ternary Fitting (ITF, 迭代三值拟合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Iterative Ternary Fitting（ITF，迭代三值拟合）是 PT²-LLM 提出的三值参数无训练优化算法。ITF 将三值量化参数 (α, μ, T) 的优化建模为交替最小化问题：(1) 固定 T 时，通过最小化权重量化误差 E_w = ||W - (αT+μ)||²，对 α_i 和 μ_i 求偏导并置零，得到闭式解（Eq. 9）向量化并行逐行求解最优网格参数 (α*, μ*)；(2) 固定 (α*, μ*) 时，通过 Z_ij = (W_ij - μ_i) / α_i 将权重投影到归一化空间，弹性舍入到最近的三值：T*_ij = argmin_{t∈{−1,0,1}} |Z_ij - t|。两步骤交替进行，每一步贪心地减小 E_w，通常约 10 轮收敛（T 不再变化）。ITF 完全无训练、无梯度反传。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: W ∈ R^{n×m}
α, μ, T = Asymmetric_Ternary_Init(W)        # TWN风格初始化
T_prev = zeros_like(T)
while T ≠ T_prev:                           # 约 10 轮收敛
    T_prev = copy(T)
    # Step A: 闭式求解最优网格 (Eq. 9, 向量化逐行并行)
    α* = (m*(W∘T)1 - (T1)∘(W1)) / (m*(T∘T)1 - (T1)²)
    μ* = ((T∘T)1∘(W1) - (T1)∘((W∘T)1)) / (m*(T∘T)1 - (T1)²)
    # Step B: 弹性舍入更新 T
    Z_ij = (W_ij - μ*_i) / α*_i
    T*_ij = argmin_{t∈{-1,0,1}} |Z_ij - t|
```
ITF 将 LLaMA-2-7B 三值化的 WikiText2 PPL 从 22.88（初始化后）降至 15.47。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ITF 在 PyTorch 中通过向量化张量运算实现：(1) Build_Optimal_Grid 使用 Eq. 9 的逐元素和张量运算（无 Python 循环）；(2) Flexible_Round 通过 argmin 查找 Z 到 {−1,0,1} 的最近映射；(3) 收敛判断通过 (T != T_prev).any()。ITF 约 10 轮可收敛。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models

## Activation-aware Grid Alignment (AGA, 激活感知网格对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Activation-aware Grid Alignment（AGA，激活感知网格对齐）是 PT²-LLM 在 ITF 之后应用的第二阶段优化。ITF 仅最小化权重层面误差 E_w = ||W - Ŵ||²，但 LLM 实际输出取决于 ŴX 而非 Ŵ 本身。AGA 将优化目标切换为 E_x = ||WX - ŴX||²，利用校准数据 X 的激活统计量（协方差矩阵 C = Σ_b Σ_l X_bl X_bl^T）以闭式解更新三值网格参数 (α, μ)。关键设计：AGA 仅更新连续参数 (α, μ)，冻结离散 T 不更新——论文实验表明在 AGA 阶段更新 T 会导致严重过拟合（模型在少量校准样本上 E_x 下降但泛化差）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# AGA 伪代码 (ITF 收敛后执行)
# 输入: W (FP16), T (冻结), X (校准数据 B×L×m)
C = Σ_b Σ_l X[b,l,:] X[b,l,:]^T             # 激活协方差 (m,m)
d = 1^T C 1                                  # 标量
v = T C 1                                     # (n,)
α* = (d*(W∘T)S1 - v∘(WS1)) / (d*T²S1 - v²)   # 闭式解 (Eq. 13)
μ* = (T²S1∘(WS1) - v∘((W∘T)S1)) / (d*T²S1 - v²)
Ŵ = α* T + μ*                                 # T 保持 ITF 输出不变
```
效果：LLaMA-2-7B 上 Avg Acc 从 38.12%（ITF only）提升至 43.33%（ITF+AGA），输出误差 E_x 在 AGA 后急剧下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AGA 实现要点：(1) 校准数据——WikiText2 128个 2048-token 片段；(2) C 矩阵——若 m 较大可通过增量累积 ΣXX^T 避免存储全部激活；(3) 与 AWQ 的区别——AWQ 通过 per-channel scaling + MSE 网格搜索，AGA 通过协方差矩阵闭式解直接求解；(4) T 冻结是防止过拟合的关键设计——三值空间仅有 3^m 种可能赋值，少量校准样本下搜索 T 会严重过拟合。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models

## Structural Similarity-based Reordering (SSR, 结构相似性重排序)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structural Similarity-based Reordering（SSR，结构相似性重排序）是 PT²-LLM 提出的列重排序策略，用于替换 GPTQ 固定顺序或 Hessian 重要性重排序。SSR 的动机：三值化（仅有 3 个量化级别）对块内权重分布极为敏感——离群列和散乱的列间分布会严重扭曲三值网格，使大量权重被错误映射。SSR 利用列间余弦相似度衡量结构相关性，每次选块时从残差矩阵中选取与均值参考向量最相似的 top-k 列，使块内列结构对齐、数值接近，形成更紧凑的分布。排列通过置换矩阵 P 实现（W'=WP, X'=XP），保证输出不变，推理时零额外开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SSR 伪代码 (集成在 GPTQ 逐块量化框架中)
# 输入: W ∈ R^{n×m}, block_size k=128
col_remaining = list(range(m))
while len(col_remaining) >= k:
    W_rem = W[:, col_remaining]             # 残差子矩阵
    w_bar = mean(W_rem, axis=1)             # 列均值参考向量 (n,)
    # 计算剩余每列与 w_bar 的余弦相似度
    sim = [(j, dot(W[:,j], w_bar)/(||W[:,j]||*||w_bar||)) for j in col_remaining]
    block_cols = [j for j,_ in sorted(sim, key=lambda x:-x[1])[:k]]
    ATQ_quantize(W[:, block_cols])           # ITF + AGA 三值化
    error_compensate(W[:, col_remaining])    # GPTQ Hessian 补偿
    col_remaining -= block_cols
```
效果：LLaMA-2-7B 上 SSR vs 无重排 PPL 从 13.06→11.56，优于 Hessian 重排（12.35）和随机重排（12.84）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SSR 实现要点：(1) 列置换是纯索引重排，推理零开销；(2) 余弦相似度计算 O(n·m_remaining) 每步，总开销远低于 ATQ 量化本身；(3) w_bar 作为"代表性列"代理——选与 w_bar 最相似的列等价于选与当前残差方向最一致的列群；(4) 块内方差可视化证实 SSR 使权重分布更紧凑（论文图 3 右侧）。

涉及论文标题：
- PT²-LLM Post-Training Ternarization for Large Language Models

---

## Givens Rotation for LLM Weight Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Givens 旋转（Givens Rotation）是一种正交线性变换，在二维平面内旋转两个坐标轴方向的值，保持其他维度不变。数学上，G(i, j, θ) 是一个 n×n 单位矩阵，仅在 (i,i)、(i,j)、(j,i)、(j,j) 四个位置替换为 cosθ、-sinθ、sinθ、cosθ。其核心性质：(1) 正交性——G^T G = I，保证范数不变，数值稳定（不引入舍入误差放大）；(2) 仅修改两行/两列，计算成本 O(1) 而非全矩阵乘法的 O(n²)。在 ParoQuant 中，Givens 旋转被用于对 LLM 权重矩阵的输入通道进行成对旋转以抑制离群通道：旋转一对通道 i 和 j 后，W'[i,:]=cosθ·W[i,:]-sinθ·W[j,:]，W'[j,:]=sinθ·W[i,:]+cosθ·W[j,:]。这一操作将离群通道的大值通过正交混合分散到正常通道中，收窄每个量化组内的动态范围。参数由 θ（旋转角度，可学习）和 (i,j)（通道对索引）组成，仅需 1 个 float + 2 个 int 参数 per pair。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Givens 旋转作用于权重 (变换 W，离线)
def givens_rotate_weight(W, i, j, theta):
    c, s = cos(theta), sin(theta)
    W[i,:], W[j,:] = c*W[i,:] - s*W[j,:], s*W[i,:] + c*W[j,:]

# Givens 逆旋转作用于激活 (推理时)
def givens_inverse_rotate_activation(X, i, j, theta):
    # 逆变换: angle = -theta, 即 cos(-θ)=cosθ, sin(-θ)=-sinθ
    c, s = cos(theta), sin(theta)
    X[:,i], X[:,j] = c*X[:,i] + s*X[:,j], -s*X[:,i] + c*X[:,j]
```
在 ParoQuant 中，128 通道的每个 group 内，K=8 个 independent rotation 各含 64 个互不重叠的 Givens 旋转。推理时每对旋转仅需 4 次 FMA（每个 token），所有 pair 间无数据依赖，可完全并行化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Givens 旋转在量化中的优势：(1) 相比全旋转矩阵（O(n²) 参数），Givens 旋转只需 O(1) 参数 per pair，内存占用极小；(2) 旋转角度通过 AdamW 逐层学习（初始化为 0，对应恒等变换），避免 Hadamard 变换的固定/随机性质；(3) 推理时逆变换在 fused CUDA kernel 中执行，角度和配对索引存储在寄存器中，激活在 shared memory 中完成全部旋转。与 Hadamard 的关键区别：Hadamard 是固定变换有全局依赖（Butterfly 结构，需 O(n log n) 步），Givens 是局部的、可学习的、完全并行的。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---

## Independent Rotation (for LLM Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Independent Rotation（独立旋转）是 ParoQuant 的核心约束设计：在 n 维通道空间中，选取一组互不重叠的通道对 P={(i_1,j_1),...,(i_m,j_m)}，满足 ∀k≠l: {i_k,j_k}∩{i_l,j_l}=∅（每个通道最多出现在一对中），对每组对施加一个 Givens 旋转。定义为 R(P,Θ)=∏_{k=1}^{m} G(i_k,j_k,θ_k)。由于 pairs 互不重叠，所有 Givens 旋转之间完全独立、无数据依赖、可同步执行。每个 independent rotation 含最多 n/2 对（128 通道时为 64 对），对应 n/2 个可学习参数（仅为 n×n 全正交矩阵 n² 参数的 1/(n-1)）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ParoQuant 的 pair 选择算法 (Algorithm A1)：
```
# g: group size, K: number of rotations, N: pairs per rotation
P_all = shuffle({(i,j) | 1≤i<j≤g})  # 所有可能的配对, 随机排列
A[i,j] = 1 for i≠j else 0            # 全局可用性矩阵
P_1..P_K = [[] for _ in range(K)]
for r in 1..K:
    A_rot = copy(A)                   # 当前 rotation 的通道可用性
    for (i,j) in P_all:
        if |P_r| = N: break
        if A_rot[i,j] == 0: continue  # 通道已被占用
        P_r.append((i,j))
        A_rot[i,:]=0; A_rot[:,i]=0    # 禁用通道 i
        A_rot[j,:]=0; A_rot[:,j]=0    # 禁用通道 j
        A[i,j]=0; A[j,i]=0            # 跨 rotation 不复用此 pair
```
每个 independent rotation 内所有 pair 并发执行，无需 barrier 同步。跨 rotation 不复用 pair 实现更丰富的通道对组合。推理时 K 个 rotations 在 fused kernel 内融合执行，一次激活加载，K 次旋转均在 shared memory 上完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Independent Rotation 的约束导致表达能力有限（O(n) vs 全旋转 O(n²)），补偿策略：(1) 串联 K=8 个使用不同 pairings 的 rotations，扩展有效参数空间；(2) 与 channel-wise scaling 联合使用——scaling 负责全局幅值均衡，rotation 负责局部 token 级对齐；(3) 跨 rotation 不复用 pair 使组合空间最大化。推理开销：每 token 每 group 计算量为 O(K·n) 次 FMA（K=8, n=128 时为 1024 FMA），相比 FP16 GEMM 的 O(n·D)（D 为 hidden_dim >> n）可忽略。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---

## Scaled Pairwise Rotation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scaled Pairwise Rotation（缩放成对旋转）是 ParoQuant 提出的完整预量化权重变换算子 T，由 channel-wise scaling + K 个串联 independent Givens rotations 组成：T(W) = (∏_{t=1}^{K} R(P_t, Θ_t)) · diag(α) · W。其中 diag(α) 是逐通道缩放（α ∈ R^n），R(P_t, Θ_t) 是第 t 个 independent rotation。三者分工：(1) Scaling 拉平全局通道间幅值差异，直接压制"整通道都是离群值"的情况；(2) Rotations 在 token 级别对齐通道对内的值，使数据点聚集到 x=y 线附近（图 1 Right），收窄组内动态范围；(3) 串联多个 rotations 弥补单个 independent rotation 参数有限的问题。推理时，对激活 X 应用逆变换 T^{-1}(X)=X·diag(1/α)·R_1^{-1}·...·R_K^{-1}，在 fused CUDA kernel 中完成。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ParoQuant 两阶段优化流程：
```
# Stage 1: 优化旋转角度 θ 和缩放因子 α
for epoch in 1..10:
    for each layer l:
        Y_fp = l(X)  # 原始层输出
        l' = copy(l)
        for each linear in l':
            for each 128-channel group:
                W_s = diag(α) · W              # Scaling
                W_t = W_s
                for t in 1..K:                  # K=8 rotations
                    for (i,j) in P_t:
                        c,s = cos(θ_t[i,j]), sin(θ_t[i,j])
                        W_t[i,:], W_t[j,:] = c*W_t[i,:]-s*W_t[j,:], s*W_t[i,:]+c*W_t[j,:]
                W_q = RTN_quantize(W_t, 4bit, g=128)  # INT4 均匀量化
        Y_q = l'(X')  # X' 为已量化前层输出
        loss = SmoothL1(Y_q, Y_fp)
        AdamW_update(θ, α, lr=0.05, cosine_decay)

# Stage 2: 微调权重和量化参数
for epoch in 1..10:
    # 固定 θ, α, 微调 W, s_q, z_q
    loss = SmoothL1(Y_q, Y_fp)
    AdamW_update(W, s_q, z_q, lr=1e-5/1e-6)
```
旋转角度 θ 初始化为 0（恒等变换），α 初始化为 1。校准集：2048 样本 × 2048 tokens，WikiText2/C4/RedPajama 均匀混合 + 64 样本 Pile 验证集。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与同类方法的区别：(1) vs AWQ (仅 scaling)——ParoQuant 多了旋转步骤，~10% 推理开销换 2.4% 推理精度提升；(2) vs QTIP (Hadamard 旋转)——Hadamard 是固定的/随机的 O(n log n) 变换，ParoQuant 是可学习的 O(Kn) 变换，参数更少、可并行化、~25% 更快，精度匹敌；(3) vs SpinQuant (可学习全旋转)——SpinQuant 旋转矩阵需合并到前序层权重，仅适用于少数层（output projection），ParoQuant 的 sparse Givens 旋转可直接在线计算，适用所有线性层。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---

## Quantization Error Accumulation in Reasoning LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
量化误差累积（Quantization Error Accumulation）是推理 LLM 在长链式思维（Chain-of-Thought）生成中面临的独特挑战。自回归解码中，每个 token 的生成依赖于所有历史 token。每一步解码的量化层输出误差 ε_{l,t} = ||Q(W_l)·h_{l-1,t} - W_l·h_{l-1,t}|| 会被传播到后续层和后续 token。在短生成（如常识 QA，<100 tokens）中误差可忽略；但在推理任务（MMLU-Pro、AIME 等，CoT 可达数千至数万 tokens）中，误差在每一步叠加，导致正确推理路径偏离。ParoQuant 量化 Qwen3-4B 在 MMLU-Pro 上的数据：FP16=71.0, AWQ=68.2（降 2.8%），ParoQuant=70.1（仅降 0.9%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
误差累积的数学模型：
```
# 总误差近似为所有步所有层的局部误差之和
# E_total ≈ Σ_{t=1}^{T} Σ_{l=1}^{L} (Π_{k=l+1}^{L} ||J_k||) · ε_{l,t}
# 其中 J_k 是第 k 层 FP16 的 Jacobian，ε_{l,t} 是单步单层量化误差
# 对于有残差连接的 Transformer, ||J_k|| ≈ 1 + δ (δ 很小)
# 因此 E_total ≈ Σ_{t=1}^{T} Σ_{l=1}^{L} ε_{l,t} = O(T·L·avg(ε))

# ParoQuant 的逐层优化策略:
for each layer l:
    # 使用已量化前层的输出 X' 作为校准输入
    X' = quantized_previous_layers_output(X)
    # 目标: 最小化当前层量化后输出与原始 FP16 输出的差异
    loss = ||Q(l)(X') - l(X)||
    # 后续层看到的是量化输出, 可以补偿前层误差
```
关键：ParoQuant 在逐层优化时使用 X'（已量化前层输出）而非 X（原始 FP16 输入），使每层的变换参数在优化时感知前层的量化误差，后续层学会补偿这些误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解误差累积的方法：(1) 使用长文本推理 benchmark（MMLU-Pro、AIME）评估量化质量，而非仅用 PPL（PPL 无法反映长生成中的误差累积效应）；(2) 逐层优化时使用已量化前层输出，使变换参数对累积误差有感知和补偿能力；(3) 多样化校准集（混合多个数据集）防止变换参数对单一数据分布的过拟合，提升对多种推理路径误差分布的鲁棒性。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---

## Learnable Boundary Quantizer (LBQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Boundary Quantizer (LBQ) 是 PassionSR 提出的可训练量化器，用于 one-step diffusion 图像超分模型的 PTQ。与传统 PTQ 方法使用固定 min/max 或 grid search 确定 clip bounds 不同，LBQ 将量化边界 B_l（下界）和 B_u（上界）声明为可学习参数，通过梯度下降在校准集上直接优化最优量化区间。其核心公式为：(1) X_c = Clip(X, B_l, B_u)，将输入裁剪到可训练边界内；(2) α = (B_u - B_l) / (2^N - 1)，计算量化 scale；(3) X_I = round((X_c - B_l) / α)，映射到离散整数；(4) X_q = α · X_I + B_l，fake-quantized 输出。与 LSQ（Learned Step Size Quantization）的差异在于：LSQ 仅训练 step size s，而 LBQ 同时训练上下两个边界 B_l 和 B_u，等价于同时训练 scale 和 zero point，并允许非对称量化间隔。LBQ 通过直通估计器（STE）反向传播梯度：当 X ∈ [B_l, B_u] 时 ∂L/∂X = 1，否则为 0。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PassionSR 中 LBQ 应用于 UNet 和 VAE 的所有可量化层（Linear、Conv、MatMul），与 LET 等效变换协同使用：
```
# LBQ 前向传播（fake quantization）
def lbq_forward(X, B_l, B_u, N_bits):
    X_c = torch.clamp(X, B_l, B_u)
    alpha = (B_u - B_l) / (2**N_bits - 1)
    X_int = torch.round((X_c - B_l) / alpha)
    X_q = alpha * X_int + B_l
    return X_q

# LBQ + LET 在 Linear 层中的联合使用
# 可训练参数: B_l, B_u (LBQ, per-tensor), s, δ (LET, per-channel)
X_tilde = (X - δ) / s               # LET 等效变换
W_tilde = s * W                      # LET 等效变换
B_tilde = B + δ @ W                  # LET 等效变换
Y_q = lbq_forward(X_tilde, B_l_a, B_u_a, 8) @ lbq_forward(W_tilde, B_l_w, B_u_w, 8) + B_tilde
```
仅 B_l 和 B_u 可训练（每个量化器 2 个参数），权重 W 保持冻结。在 DQC Stage 1 中 LBQ 冻结、仅训练 LET；在 Stage 2 中 LBQ 和 LET 联合训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LBQ 的实现基于 PyTorch 的 fake quantization 机制：(1) 为每个待量化层创建一对可学习参数 B_l, B_u；(2) 前向传播中执行 fake quant（含 clamp + round + rescale）；(3) 反向传播中通过 STE 近似梯度；(4) 校准完成后，将 B_l, B_u 固化为 INT8/INT6 推理的 scale 和 zero-point。PassionSR 代码仓库（https://github.com/libozhu03/PassionSR）在 `ptq_quantize_single.py` 中实现了 LBQ 与 LET 的联合训练流程。LBQ 也可独立使用（消融实验中 LBQ-only 的 W6A6 PSNR=23.15），但与 LET 联合使用时性能显著提升（PSNR 25.40）。

涉及论文标题：
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

---

## Distributed Quantization Calibration (DQC) in PassionSR

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distributed Quantization Calibration (DQC) 是 PassionSR 提出的两阶段 PTQ 标定策略，旨在解决同时训练 LBQ 的边界参数和 LET 的等效变换参数时的训练不稳定问题。DQC 将整个标定过程拆分为两个阶段：(1) **Stage 1**：冻结 LBQ（使用初始边界），仅训练 LET 的 scale factors s 和 offsets δ。此阶段 LET 将激活分布调整为对量化友好的形态。(2) **Stage 2**：在 LET 更新后的变换向量上重新初始化 LBQ，然后联合训练 LBQ 和 LET。重新初始化的原因在于：LET 的变换改变了各层的激活/权重分布，旧的 LBQ 边界已不再适用。DQC 的核心效果是：标定时间从 3.87h（LBQ+LET 联合训练）降至 1.07h，GPU 显存从 40GB 降至 28GB，显著加速收敛（Fig. 5 展示了 w/ 和 w/o DQC 的 loss 曲线对比）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PassionSR 中 DQC 的执行流程：
```
# DQC Stage 1: 仅训练 LET
for param in LBQ_parameters: param.requires_grad = False
for param in LET_parameters: param.requires_grad = True
for epoch in range(2):
    for X_lr, X_hr in calib_loader:
        Y_fp = fp_model(X_lr)         # FP32 教师输出
        Y_q = quantized_model(X_lr)    # LBQ(frozen) + LET(trainable) 量化输出
        loss = block_wise_mse(Y_q, Y_fp)
        loss.backward()                # 梯度仅流向 LET 的 s, δ

# DQC Stage 2: 重新初始化 LBQ + 联合训练
reinitialize_LBQ_boundaries()          # 基于 LET 变换后的分布重新初始化 B_l, B_u
for param in LBQ_parameters: param.requires_grad = True
for param in LET_parameters: param.requires_grad = True
for epoch in range(2):
    for X_lr, X_hr in calib_loader:
        Y_fp = fp_model(X_lr)
        Y_q = quantized_model(X_lr)    # LBQ+LET 均可训练
        # 模块级逐层损失
        loss_UNet = ||I(Z_lq, ε_q) - I(Z_l, ε_fp)||_2   # latent space MSE
        loss_VAE_enc = ||V_qe(X_fp) - V_fpe(X_fp)||_2
        loss_VAE_dec = ||V_qd(X_q) - V_fpd(X_fp)||_2
        total_loss = loss_UNet + loss_VAE_enc + loss_VAE_dec
        total_loss.backward()          # 梯度流向 LBQ 和 LET
```
UNet 损失函数中的 I(Z_l, ε) 是 OSD 模型特有的从输入 latent Z_l 和预测噪声 ε 到输出 latent Z_h 的变换函数：Z_h = sqrt(1/α̂) · Z_l - sqrt((1-α̂)/α̂) · ε(Z_l)，利用 OSD 模型时间步为常数的特性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DQC 适用于以下场景：(1) 量化器包含可训练参数且与等效变换参数耦合导致训练不稳定时；(2) 需要降低标定 GPU 显存和时间成本时。实现要点：Stage 1 用较小的学习率稳定搜索 LET 的最优解空间；Stage 2 必须重新初始化 LBQ（否则新分布上的旧边界导致性能退化）。PassionSR 开源代码中在 `ptq_quantize_single.py` 实现了 DQC 的完整流程，其中 epoch 数、学习率（1e-5）、标定 batch 均可配置。注意：PassionSR 的 DQC 与 2DQuant 的 DQC（Distillation-based Quantization Calibration）虽然缩写相同但机制不同——PassionSR 的两阶段解耦训练 vs 2DQuant 的知识蒸馏微调 clip bounds。

涉及论文标题：
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

---

## One-Step Diffusion Image Super-Resolution (OSDSR)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-Step Diffusion Image Super-Resolution (OSDSR) 是将扩散模型的多步去噪过程压缩为单步推理的图像超分范式。传统扩散 SR 模型（如 StableSR、DiffBIR、SeeSR）需要 T = 50-1000 步迭代去噪，每步需完整运行 UNet 推理，高延迟阻碍实际部署。OSD 模型通过 score distillation 或 consistency distillation 技术将去噪步数减至 1：对给定的低分辨率输入 LR，模型在单次前向传播中直接从噪声/低质量 latent 恢复到高质量 latent Z_h，再经 VAE decoder 生成高分辨率输出 HR。代表性 OSD SR 模型包括 SinSR（CVPR 2024）、OSEDiff（NeurIPS 2024）、DFOSD（2024）。以 OSEDiff 为例，其结构包含 UNet（去噪预测）、VAE（编解码）、DAPE（detail-aware prompt encoder）和 CLIPEncoder（文本条件），FP32 下参数 1,303M、操作 4,523G。OSD 模型虽然步数降至 1，但单步计算量极大（UNet+VEA MACs 超 2,100G），亟需量化压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
OSEDiff 的 one-step 推理流程（FP32）：
```
# 输入: LR image, 输出: HR image (×4 upscaling)
Z_l = VAE_encoder(LR)                          # latent encoding, ~1,781G MACs
ε_pred = UNet(Z_l, text_embedding)              # noise prediction, ~339G MACs (1 step)
Z_h = sqrt(1/α̂) * Z_l - sqrt((1-α̂)/α̂) * ε_pred  # one-step latent recovery
HR = VAE_decoder(Z_h)                           # latent decoding
```
与多步扩散对比：多步模型需将 T=50 步的 UNet 推理串联，总 MACs = T × UNet_MACs + VAE_MACs ≈ 50 × 339G + 1,781G ≈ 18,731G。OSD 仅 1 步 UNet，总 MACs ≈ 2,262G（含 DAPE），加速 ~8.3×。但 VAE 的 1,781G MACs（78.8%）成为主要瓶颈——这是 PassionSR 选择同时量化 VAE 而非仅量化 UNet 的根本原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
OSD SR 模型的实现基于预训练扩散模型（如 Stable Diffusion 2.1）的 score distillation：(1) 使用预训练 SD 的 UNet + VAE 权重初始化；(2) 通过 score distillation loss 将多步教师的行为蒸馏到单步学生模型；(3) 在真实世界 SR 数据集上微调。OSEDiff 开源代码（https://github.com/cswry/OSEDiff）提供完整训练和推理流程。PassionSR 直接使用 OSEDiff 作为量化 backbone 并进一步简化：将 DAPE-CLIPEncoder 分支替换为基于空字符串预计算的常数 embedding（参数从 1,303M 降至 949M，性能持平），得到仅含 UNet+VAE 的 PassionSR-FP 模型，便于统一量化标定。

涉及论文标题：
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

---

## Multi-Scale Salient Attention Distillation (MSAD / 多尺度显著注意力蒸馏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MSAD 是 QuantSparse 论文提出的在校准阶段用于对齐量化 attention 与 FP attention 的高效蒸馏框架。它解决 video DiT PTQ 中朴素联合量化和稀疏注意力产生的 "amplified attention shift" 问题——量化噪声与稀疏 mask 相互增强，导致 attention 分布严重偏移。MSAD 的核心思路是避免直接存储和蒸馏完整 O(L²) attention 矩阵（对 HunyuanVideo-13B 的 L>10⁴ tokens, 单层 ~6.82GB 不可承受），转而通过两个互补的蒸馏分支以极少内存开销监督 attention 对齐：(1) Global Guidance——对 Q 和 K 做 average pooling 下采样（stride s=128），在低分辨率上计算 attention 并 MSE 蒸馏 FP 与 quantized 版本，捕捉全局结构拓扑（内存 O(L̃²), s=128 时仅 ~0.14GB）；(2) Local Guidance——利用 attention saliency 的重尾分布特性（<10% tokens 占据大部分 attention mass），仅对 FP 模型识别出的 top-k=256 salient queries 做高分辨率 attention 蒸馏（内存 O(kL)），保留细粒度关键细节。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MSAD 伪代码（校准阶段，每 transformer block）：

```
Input: X ∈ R^{L×d_in}, FP weights W_q,W_k,W_v, calibration data
Output: Quantized weights with optimized {s,z}

// FP forward
Q_fp = X·W_q^T, K_fp = X·W_k^T, V_fp = X·W_v^T  // FP16
A_fp = softmax(Q_fp·K_fp^T / √d_k)               // ∈ R^{h×L×L}

// Compute token saliency (Eq. 7)
s_j = Σ_h Σ_i A_fp[h,i,j]                         // aggregate attention received
I = top-k({s_j})                                   // select k salient queries

// Quantized forward
Q_q = Q(X)·Q(W_q)^T, K_q = Q(X)·Q(W_k)^T          // W4A8 quantized matmul
// Q(·) = s·(clip(⌊X/s⌋+z, 0, 2^b-1) - z)

// Global Guidance (Eq. 6)
Q̃_fp = AvgPool(Q_fp, s), K̃_fp = AvgPool(K_fp, s)  // stride s=128
Q̃_q = AvgPool(Q_q, s), K̃_q = AvgPool(K_q, s)
A_global_fp = softmax(Q̃_fp·K̃_fp^T / √d_k)
A_global_q = softmax(Q̃_q·K̃_q^T / √d_k)
L_global = MSE(A_global_fp || A_global_q)

// Local Guidance (Eq. 8)
A_local_fp = softmax(Q_fp[I,:]·K_fp^T / √d_k)     // only for salient queries
A_local_q = softmax(Q_q[I,:]·K_q^T / √d_k)
L_local = MSE(A_local_fp || A_local_q)

// Total loss (Eq. 9)
L_total = L_quant + λ_global·L_global + λ_local·L_local
// λ_global=1e-4 for Wan2.1; λ_global=1.0 for HunyuanVideo
// λ_local=1e-4 for Wan2.1; λ_local=1e2 for HunyuanVideo

// Optimize quant params {s,z}, channel-wise scale, rotation matrix
s,z ← AdamW(L_total, lr_scale=5e-2, lr_others=5e-3)
```

张量形状：L≈10400 tokens (720×1280, 60 frames → H=720/8=90, W=1280/8=160, T=60/4=15 → ~90×160×15/L≈21600, 经 compression to L≈10⁴); L̃=L/s²≈10400/128²≈0.64→1 tokens; k=256。MSAD 内存开销仅 +0.8% GPU memory, 校准时间 +1.6%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MSAD 在 QuantSparse 校准阶段与 block-wise PTQ 结合使用：逐 block 加载 → FP forward 计算 saliency + global/local attention targets → 量化 forward 计算对应 attention → MSE 蒸馏 → AdamW 优化量化参数（15 epochs, cosine LR）。全局和局部分支均以极小开销运行（Global 下采样 s=128 将 L² 降至 L²/16384, Local 仅 kL 计算）。MSAD 有效缓解量化+稀疏化的 attention shift, 使 W4A8+15% density 下 PSNR 从 14.35 提升至 18.72 (Wan2.1-14B 消融)。代码: https://github.com/wlfeng0509/QuantSparse（待发布）。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---

## Second-Order Sparse Attention Reparameterization (SSAR / 二阶稀疏注意力重参数化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SSAR 是 QuantSparse 论文提出的在推理阶段恢复稀疏注意力信息丢失的技术。背景：稀疏注意力 mask 删除了低值但非零的 attention connections，定义了全注意力与稀疏注意力之差为**一阶残差** Δ^(t)=A_full^(t)−A_sparse^(t)。先前的 cache-based 方法（如 DiTFastAttn）假设 Δ^(t) 在 diffusion 时间步间不变（Δ^(t')≈Δ^(t) ∀t,t'），在参考步缓存后复用。但量化引入噪声 ϵ^(t) 后，一阶残差变化 Δ_quant^(t)=Δ^(t)+ϵ^(t) 的 temporal stability 被破坏（量化噪声使 Δ_quant^(t) 在时间步间不守恒）。SSAR 的核心洞察：**二阶残差** Δ̃_quant^(t)=Δ_quant^(t)−Δ_quant^(t-1) 具有显著更高时间稳定性，因为量化噪声 ϵ^(t) 在 diffusion 过程中呈缓变随机过程，相邻步噪声分布相似 → ϵ^(t)−ϵ^(t-1) 近似平稳。SSAR 缓存一阶+二阶残差，在推理时叠加于稀疏注意力输出之上求近似全注意力，并可选 SVD 投影到 top-r 主成分进一步抑制时间方差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SSAR 推理伪代码（结合 W4A8 量化+稀疏 mask M, cache interval τ=5, SVD rank r=16）：

```
Input: M_quant, prompt P, denoising steps T, interval τ
Output: Generated video Y

Load M_quant, input P
Initialize cache: Δ_cache = None, t_ref = -1

for t = 0 to T-1:
    // Quantized sparse attention forward
    Q_q = Q(X_t)·Q(W_q)^T, K_q = Q(X_t)·Q(W_k)^T, V_q = Q(X_t)·Q(W_v)^T
    A_sq = softmax(Q_q·K_q^T/√d_k ⊙ M)  // Eq. 3, M = SVG spatial-temporal mask

    if t - t_ref ≤ τ and cache is valid:
        // Reuse cached residuals (no full attention recomputation)
        A_approx = A_sq + Δ_cache                         // one mat-add overhead
    else:
        // Refresh: recompute full attention for this step
        A_full = softmax(Q_q·K_q^T/√d_k)                 // no mask
        Δ_q_t = A_full - A_sq                            // first-order residual

        if t_ref ≥ 0:  // have previous reference step
            // Compute and cache second-order residual (Eq. 14)
            Δ̃_q = Δ_q_t - Δ_q_prev                            // second-order
            SVD(Δ̃_q) = S·U·V^T                               // decompose
            Δ̃_q_proj = S_{:,:r} · U_{:r,:r} · V^T_{:,:r}      // project to top-r
            Δ_cache = Δ_q_t + Δ̃_q_proj                        // first+second (Eq. 16)
        else:
            Δ_cache = Δ_q_t                                  // first step: only first-order

        t_ref = t
        Δ_q_prev = Δ_q_t
        A_approx = A_sq + Δ_cache

    // Compute attention output
    Out = A_approx @ V_q
    // → subsequent transformer blocks → denoise step
```

关键参数：cache interval τ=5（每 5 步重新计算全 attention 刷新缓存）, SVD rank r=16。开销：仅额外一次矩阵加法（+0.2% DiT time）, 缓存存储 2× attention output（一阶+二阶合并）, +8-11% GPU memory。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SSAR 在推理阶段与量化稀疏 attention 共同使用。优势：(1) 二阶残差缓存与一阶残差合并存储（额外开销可忽略）；(2) SVD 投影选取 top-r 主成分抑制时间方差，几乎无额外开销；(3) 与 MSAD 互补——MSAD 在 calibration 阶段对齐 attention map, SSAR 在 inference 阶段通过残差校正恢复稀疏化损失；(4) cache interval 可调，interval=3→最高质量, interval=6→最高速度。SSAR 在 Wan2.1-14B W4A8 下将 PSNR 从 14.16 (no reparam) → 17.08 (first-order) → 18.68 (second-order) → 18.72 (SSAR with SVD, top-16)。代码: https://github.com/wlfeng0509/QuantSparse（待发布）。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---

## Amplified Attention Shift (放大注意力偏移)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Amplified Attention Shift 是 QuantSparse 论文提出的描述量化与稀疏注意力组合时性能退化机制的核心概念。当模型权重和激活被量化时，量化噪声 ϵ 注入 QK dot product 产生系统性偏差 δ（‖ϵ‖_F ≤ δ）。当稀疏 mask M 同时应用时，mask 删除的低值 attention connection 中本包含的量化噪声被"锁死"进入最终 attention 分布，两者叠加产生超额偏移：Δ_total = Δ_sparse + Δ_quant + O(‖ϵ‖_F·‖M‖_0)。交叉项 O(‖ϵ‖_F·‖M‖_0) 是关键——它意味着总偏移比单独量化和单独稀疏化之和更大。论文通过定量实验验证：单独量化 attention MSE=0.216, 单独稀疏化 attention MSE=0.134, 而组合后 attention MSE=0.685 (远超 0.216+0.134=0.350 的简单相加)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Amplified Attention Shift 的形式化表达：

```
无压缩:    A_full = softmax(Q·K^T/√d_k)·V

仅量化:    A_quant ≈ softmax((Q+ϵ_q)(K+ϵ_k)^T/√d_k)·V
           = A_full + Δ_quant   (MSE ≈ 0.216)

仅稀疏化:  A_sparse = softmax(Q·K^T/√d_k ⊙ M)·V
           = A_full + Δ_sparse  (MSE ≈ 0.134)

量化+稀疏
(naive组合): A_sq = softmax(Q_q·K_q^T/√d_k ⊙ M)·V
           = A_full + Δ_sparse + Δ_quant + O(‖ϵ‖·‖M‖₀)
           (MSE ≈ 0.685, 远超 0.216+0.134=0.350)
```

交叉项 O(‖ϵ‖_F·‖M‖_0) 的来源：量化噪声 ϵ 对 QK 矩阵中所有元素都产生扰动，稀疏 mask 删除部分 attention connection 后，被删除位置的量化误差无法被 softmax normalization 中的其他 attention 值"稀释"，导致保留的 attention connection 承受了不成比例的量化和稀疏化双重扭曲。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuantSparse 通过两个技术对抗 Amplified Attention Shift：(1) MSAD——在 calibration 阶段直接监督 attention map 对齐，减少 Δ_quant 的幅度；(2) SSAR——在 inference 阶段通过二阶残差缓存恢复稀疏化丢失的低值 attention, 减小 Δ_sparse 的幅度。两者协同将 attention MSE 从 0.685 恢复至接近 FP 水平。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---

## Token Saliency in Video DiT Attention (视频 DiT 注意力中的 Token 显著性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Saliency 在 video DiT attention 中指某个 token 在 attention 分布中接收到的总注意力权重。形式化定义：给定 attention map A = softmax(Q·K^T/√d_k) ∈ R^(h×L×L)（h=heads, L=sequence length），token j 的 saliency 定义为 s_j = Σ_h Σ_i A[h,i,j]，即所有 query 对该 token 的注意力之和。QuantSparse 论文在 empirical analysis 中发现 video DiT 的 token saliency 分布呈**重尾分布（heavy-tailed distribution）**：仅 <10% 的 tokens 占据大部分 attention mass（见图 3a, 6, 7——Wan2.1 和 HunyuanVideo 的多个 block 普遍呈现此现象）。这一现象源于 video data 的时空局部性：相邻 spatial/temporal tokens 高度相似，attention 自然集中到少数关键 tokens。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Token saliency 在 MSAD 的 Local Guidance 中的使用：

```
// Compute FP attention
A_fp = softmax(Q_fp·K_fp^T / √d_k) ∈ R^{h×L×L}

// Token saliency (Eq. 7)
for j in 0..L-1:
    s_j = 0
    for h in 0..H-1:
        for i in 0..L-1:
            s_j += A_fp[h, i, j]   // aggregate attention received by token j

// Select top-k salient tokens
I = argsort(s, descending=True)[:k]  // k=256, <2.5% of L≈10⁴

// Local distillation: only on salient queries
A_local = softmax(Q_q[I,:]·K_q^T / √d_k)   // ∈ R^{k×L}
L_local = MSE(A_fp[I,:] || A_local)
```

效果对比：salient selection (PSNR 16.82) vs random selection (PSNR 15.49), 说明 top-k salient 选择显著优于随机采样。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Token saliency 在 QuantSparse 校准阶段每 block 计算一次（因 FP 前向固定, saliency 不变），后续优化迭代中复用 I 索引（无需重复计算）。这一特性与 video data 的 temporal coherence 相关——关键 spatial-temporal tokens 在 denoising 过程中保持稳定。具体选择 k=256（平衡效果与效率），在 s=128 的 global guidance 基础上提供重要的 local 补充。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---

## RaBitQ (Randomized Binary Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RaBitQ（Randomized Binary Quantization）是一种面向高维向量 ANN 搜索的量化方法，由 Gao 和 Long 在 SIGMOD 2024 提出。RaBitQ 将 D 维浮点向量量化为 D 位二进制串（对应 32x 压缩率），提供无偏距离估计器和具有渐近最优误差界的理论保证。核心原理：(1) 码本构造——取超立方体顶点集 C = {±1/√D}^D（所有由 +1/√D 和 -1/√D 组成的 D 维单位向量），乘以随机正交矩阵 P 旋转得到码本 C_r = {Px | x∈C}；(2) 量化编码——对每个数据向量 o，找码本中最近向量 ō₀ 作为量化向量，用 D 位二进制码 x̄_b ∈ {0,1}^D 表示；(3) 距离估计——无偏估计器 ⟨ō₀,q⟩/⟨ō₀,o⟩ 估计内积 ⟨o,q⟩，计算简化为 ⟨ō₀,q⟩ = (2/√D)·⟨q',x̄_b⟩ - (1/√D)·Σq'[i]，其中 q'=P^{-1}q。误差界以高概率为 O(1/√D)。GitHub: https://github.com/gaoj0017/RaBitQ

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RaBitQ 的算法 pipeline 分为 index 和 query 两阶段：
```
# Index 阶段
P = random_orthogonal_matrix(D)  # QR 分解生成
c = centroid(data_vectors)       # 全局质心
for each o_r:
    o = (o_r - c) / ||o_r - c||  # 中心化归一化
    o' = P^{-1} o                # 逆旋转变换
    for d = 1..D:
        x_b[d] = (o'[d] > 0) ? 1 : 0  # 按符号量化
    # ō₀ = P·(2/√D·x_b - 1/√D·1_D)
    store: ||o_r-c||, ⟨ō₀,o⟩, x_b

# Query 阶段
q = (q_r - c) / ||q_r - c||, q' = P^{-1} q, s = Σ_i q'[i]
for each candidate:
    ⟨q',x_b⟩ = FastScan_SIMD(q', x_b)  # 批量 SIMD
    ⟨ō₀,q⟩ = (2/√D)·⟨q',x_b⟩ - (1/√D)·s
    ⟨o,q⟩_est = ⟨ō₀,q⟩ / ⟨ō₀,o⟩
    dist_est² = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·⟨o,q⟩_est
```
关键不变式：(1) 估计器无偏，即 E[⟨ō₀,q⟩/⟨ō₀,o⟩] = ⟨o,q⟩；(2) 误差界始终成立，不依赖数据分布；(3) ⟨ō₀,o⟩ ≈ 0.8 在高维空间中高度集中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RaBitQ 提供两种距离计算实现：(1) FastScan-based：基于 SIMD (AVX512) 批量计算；(2) Bitwise-based：位操作逐个向量计算。RaBitQ 通常与 IVF 索引结合使用，先对数据集聚类再对每聚类内向量量化。也可与图索引结合（如 SymphonyQG, SIGMOD 2025）。代码开源在 https://github.com/gaoj0017/RaBitQ（C++）。局限：仅支持 32x 压缩率 (1 bit/dim)，中等压缩率下需配合 re-ranking 或扩展方案。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Extended RaBitQ (Multi-bit RaBitQ / B-bit RaBitQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Extended RaBitQ 是对原始 RaBitQ（仅支持 B=1 bit/dim）的扩展，支持任意 B bits/dim 的中等压缩率（B=4→8x, B=5→6.4x, B=8→4x）。核心创新是在 D 维空间（而非填充到 B·D 维）中构造码本：G = {-(2^B-1)/2 + u | u=0,...,2^B-1}^D，归一化后做随机正交旋转得 G_r = {P·y/||y|| | y∈G}。该码本继承 RaBitQ 的无偏估计器和误差界，误差随 B 指数衰减（经验公式 ε < 5.75·2^{-B}/√D，>99.9% 置信度）。理论证明仅需 B = Θ(log(ε^{-2}·log(1/δ)/D)) bits 达到误差界 ε，达渐近最优。编码算法通过枚举至多 D·2^{B-1} 个临界值找到最近码本向量，复杂度 O(2^B·D log D)。开源: https://github.com/VectorDB-NTU/Extended-RaBitQ

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index 阶段: 码本 G_r = {P·y/||y|| | y∈G}
P = random_orthogonal_matrix(D)
for each o_r:
    o = (o_r - c) / ||o_r - c||, o' = P^{-1} o
    # Algorithm 1: 枚举 critical values 找最近码本向量
    t=0, v_max=0, 初始化 y_cur, ⟨y_cur,o'⟩, ||y_cur||
    while 存在未枚举的 critical value:
        t = next_min_critical_value()  # minheap, O(log D)
        更新 y_cur（仅一维变化, O(1)）, ⟨y_cur,o'⟩, ||y_cur||
        if ⟨y_cur,o'⟩/||y_cur|| > v_max:
            v_max = ⟨y_cur,o'⟩/||y_cur||, t_max = t
    ȳ = round(t_max · o')
    ȳ_u = ȳ + (2^B-1)/2 · 1_D  # B-bit 无符号整数向量
    # 拆分: ȳ_u = 2^{B-1}·ȳ₀ + ȳ_last（MSB=RaBitQ 二进制码）

# Query 阶段: 两阶段距离比较
q' = P^{-1} q, s = Σ_i q'[i]
# Stage 1: 仅用 MSB ȳ₀ (RaBitQ 二进制码)
⟨ȳ₀,q'⟩ = FastScan_SIMD(ȳ₀, q')
if lower_bound(dist_rough) > best: continue  # 剪枝
# Stage 2: 增量计算
⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ₀,q'⟩ + ⟨ȳ_last,q'⟩  # 复用 Stage 1 结果
⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u,q'⟩ - (2^B-1)/2 · s)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
C++ 实现，与 IVF 索引结合：(1) 数据集聚为 4,096 个聚类（百万级）；(2) 每聚类用本地质心中心化；(3) Algorithm 1 量化编码；(4) 分离存储 ȳ₀（MSB）和 ȳ_last。B=4 或 8 时复用现有 4/8-bit 整数内积实现；其他 B 通过拆分实现。B=5 时 >95% recall (6.4x 压缩)，B=7 时 >99% recall (4.5x 压缩)，均无需 re-ranking。代码: https://github.com/VectorDB-NTU/Extended-RaBitQ

涉及论文标题：
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Product Quantization (PQ)（乘积量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Product Quantization (PQ) 是一种高维向量压缩方法，由 Jégou et al. (2010) 提出。核心思想：将 D 维向量空间分解为 M 个低维子空间的笛卡尔积，每个子空间独立训练子码本（通常 256 中心 = 8-bit），原始向量每段映射到最近子码本向量。M 个子码本索引串联构成量化码。码本总大小 = 256^M（概念性），压缩率可达 32x+。主要问题：(1) 距离估计有偏且无理论误差界；(2) 中等压缩率 (B≥4 bits/dim) 下精度反而不如 SQ；(3) 距离计算需频繁查表 (RAM 访问)，效率低于 SQ/LVQ/RaBitQ。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index 阶段
# 1. 子空间分解: D 维 → M=D/d 个 d 维子空间
# 2. 每子空间 k-means 训练 256 中心子码本
# 3. 量化:
for m = 1..M:
    o_m = o[(m-1)*d : m*d]
    idx[m] = argmin_j ||o_m - c_{m,j}||  # 最近子码本向量
# 存储: idx[1..M]（M bytes）

# Query 阶段: 查表累加
for m = 1..M, j = 0..255:
    dist_table[m][j] = ||q_m - c_{m,j}||²
for each candidate:
    dist_est² = Σ_{m=1..M} dist_table[m][idx_candidate[m]]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
广泛部署于 Faiss (`IndexIVFPQ`)、Milvus 等向量数据库，通常配合高压缩率 (32x+) + IVF + re-ranking。FastScan SIMD 加速版需 k=4 子码本但精度进一步损失。PQ 更适合极高压缩率且能容忍 re-ranking 开销的场景。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Scalar Quantization (SQ)（标量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scalar Quantization (SQ) 是向量量化的最简形式：对每个维度独立量化。收集所有向量的全局 min v_l 和 max v_r，范围 [v_l,v_r] 均匀分为 2^B-1 区间，每浮点数取整到最近区间边界，存为 B-bit 无符号整数。压缩率 32/B（B=8→4x, B=4→8x）。距离计算通过 B-bit 整数与浮点向量做内积（SIMD），无需查表，效率显著优于 PQ。缺点：(1) B<4 时精度极差；(2) 全局范围对异常值敏感。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index 阶段
v_l = min(all vectors all dims), v_r = max(all vectors all dims)
step = (v_r - v_l) / (2^B - 1)
for each vector:
    for d = 1..D:
        code[d] = round((o_r[d] - v_l) / step)

# Query 阶段
# 通过 SIMD 计算 ⟨code, q⟩ 内积（整数×浮点）直接估计距离
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
广泛部署于 Faiss (`IndexScalarQuantizer`)、Milvus、SPANN 等，常与 IVF 结合，B=8 为典型配置。AVX512 SIMD 批量计算。是中等压缩率最常用的方法。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## LVQ (Local Vector Quantization)（局部向量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LVQ 是 SQ 的改进变体，由 Aguerrebere et al. (VLDB 2023) 提出。不同于 SQ 使用全局范围，LVQ 为每向量独立确定 [v_l,v_r]，在各自范围内均匀量化。优势：(1) 对异常值更鲁棒——一个向量的极端值不影响其他向量精度；(2) 同 bit 数下精度优于 SQ。代价：每向量额外存储 2 个 FP32 (v_l, v_r)，对 D>300 的向量可忽略。在 Extended RaBitQ 论文中，LVQ 是最具竞争力的 baseline，但 Extended RaBitQ 在 B>6 时 error 仍比 LVQ 小 1.3x-3.1x。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index
for each vector:
    v_l = min(o_r), v_r = max(o_r)
    step = (v_r - v_l) / (2^B - 1)
    for d = 1..D:
        code[d] = round((o_r[d] - v_l) / step)
    store: v_l, v_r, code[1..D]

# Query: o_approx[d] = v_l + code[d] * step
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与 SQ 几乎一致，距离计算通过 B-bit 整数×浮点 SIMD 内积完成。与 IVF 索引结合使用。详见原 LVQ 论文（VLDB 2023）。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Johnson-Lindenstrauss Transformation (JLT)（约翰逊-林登斯特劳斯变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
JL 引理 (Johnson & Lindenstrauss, 1984) 是降维理论基石：对于 N 个 D 维点集，存在映射 f:R^D→R^d, d=O(ε^{-2}log N)，使所有点对距离在 (1±ε) 因子内保持。JLT 是实现此映射的方法，最常用随机正交矩阵或随机高斯矩阵。已证明在维度-vs-误差权衡上渐近最优 [Larsen & Nelson, FOCS 2017]。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 RaBitQ 中 JLT 的角色：(1) 随机化——P 的随机性使量化误差的集中不等式成立；(2) 等距性——正交矩阵保内积，允许在变换后空间操作；(3) P 应用于数据为 o'=P^{-1}o，利用正交性 ⟨P·y/||y||, o⟩ = ⟨y/||y||, P^{-1}o⟩ 将旋转转移到查询向量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
随机正交矩阵通过 D×D 高斯矩阵 QR 分解生成。P 仅需采样一次，所有向量共享。存储开销 D² FP32（D=3072 时 ~38MB）。应用场景：局部敏感哈希 (LSH)、随机投影树、维度约简。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Quantization Codebook（量化码本）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
量化码本是预定义的向量集合，每个数据向量映射到码本中最近向量（量化向量），最近向量的索引即为量化码。码本大小 K 决定压缩率：log₂(K) bits 编码。关键约束：(1) 覆盖数据分布；(2) 结构支持高效最近搜索和距离计算。Extended RaBitQ 码本：G_r = {P·y/||y|| | y = (-(2^B-1)/2+u), u∈{0,...,2^B-1}}^D，大小 2^{B·D}，由 P 唯一确定，仅需存储 P。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
不同方法的码本对比：
- PQ: C₁×...×C_M 笛卡尔积，|C|=256^M
- SQ: {v_l+i·step}^D 网格，|C|=(2^B)^D
- RaBitQ (B=1): P·{±1/√D}^D, |C|=2^D
- Extended RaBitQ: P·{y/||y|| | y∈网格}, |C|=2^{B·D}
Extended RaBitQ 码本独特性：(1) D 维空间非填充；(2) 归一化保无偏性；(3) 随机旋转破对齐；(4) 整数网格支持高效内积。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RaBitQ/Extended RaBitQ 仅需存储 P，码本按需计算，节省指数级存空间。代码: https://github.com/VectorDB-NTU/Extended-RaBitQ

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Unbiased Estimator in Vector Quantization（向量量化中的无偏估计器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 RaBitQ 系列中，无偏估计器 ⟨ō,q⟩/⟨ō,o⟩ 是 ⟨o,q⟩ 的无偏估计：E[⟨ō,q⟩/⟨ō,o⟩] = ⟨o,q⟩。前提：(1) 码本由随机旋转的单位向量组成；(2) ō 是 o 在码本中的最近向量。随机性来自随机正交矩阵 P。误差界以高概率成立。论文通过 10⁷ 对估计-真值点对做线性回归（slope=1, intercept=0）验证无偏性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
距离估计公式：
||o_r-q_r||² = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·⟨o,q⟩
⟨o,q⟩ ≈ ⟨ō,q⟩ / ⟨ō,o⟩（无偏）
其中 ⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u,q'⟩ - (2^B-1)/2·Σq'[i])，q'=P^{-1}q

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
⟨ō,o⟩ 索引时预计算存储；Σq'[i] 查询时一次计算。无偏性意味着无系统性偏差，排序一致性更好。代码: https://github.com/VectorDB-NTU/Extended-RaBitQ

涉及论文标题：
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Approximate Nearest Neighbor (ANN) Search（近似最近邻搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ANN 是 NN 搜索的松弛版：给定 N 个 D 维数据向量和查询 q，返回与 q 距离接近 NN 的向量，允许精度折损换时间/空间节省。指标：recall@K（前 K 中真 NN 比例）、average distance ratio、QPS（效率）。ANN 是向量数据库、RAG、推荐系统、信息检索的核心操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IVF + 量化的 ANN pipeline：
1. Index: KMeans 聚类 → 每聚类中心化 → 量化编码 → 存储压缩码
2. Query: 找 nprobe 最近聚类 → 扫描候选，用压缩码估计距离 → 返回最小估计距离的向量
关键参数：聚类数 L、nprobe（扫描聚类数）、B（量化 bits/dim）。本论文目标场景：仅存储压缩向量（无原始向量），通过中等压缩率独立产生 >95% recall 无需 re-ranking。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流库：Faiss (Meta)、Milvus、ScaNN (Google)、Annoy (Spotify)、NGT (Yahoo)、pgvector。量化+IVF 或图索引是主流方案。本论文方法可在所有 ANN 任务中无缝替换 SQ/LVQ。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---

## Random Orthogonal Matrix（随机正交矩阵）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
从正交群 O(D)={Q∈R^{D×D}|Q^T Q=I} 上均匀采样的 D×D 矩阵。性质：(1) Q^T=Q^{-1}；(2) 保内积: ⟨Qx,Qy⟩=⟨x,y⟩；(3) 保范数: ||Qx||=||x||。生成：D×D 高斯矩阵 QR 分解取 Q。均匀分布保证 P 的随机性是估计器无偏性和误差界的概率来源——打破码本与数据分布的对齐，使坐标尾界 P[|x[i]|>t/√D]≤2exp(-c₀t²) 成立。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Extended RaBitQ 中：o'=P^{-1}o, q'=P^{-1}q, 利用正交性 ⟨P·y/||y||, q⟩=⟨y/||y||, P^{-1}q⟩=⟨y/||y||, q'⟩，将旋转从码本向量转移到查询向量，实现高效计算。P 仅采样一次，所有向量共享。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QR 分解生成（BLAS dgeqrf），矩阵乘向量用 BLAS gemv (O(D²))。大 D 时可用结构化随机矩阵近似。存储 D² FP32（D=3072 时 ~38MB）。应用：JLT 降维、RaBitQ 量化、随机投影等。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

## Token-aware Quantization Estimator (TQE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token-aware Quantization Estimator (TQE) 是 Q-VDiT (ICML 2025) 提出的面向视频 Diffusion Transformer 的量化误差补偿模块。其理论基础为 Theorem 3.2：量化误差 Δ = W − Q̂(W) 的信息熵 H(Δ) ≤ H(W)（因为 round-to-nearest decimal truncation 是 surjection），因此 Δ 可在更低秩空间估计。TQE 使用两组低维向量参数 α∈R^{d_in} 和 β∈R^{d_out}（共 d_in+d_out 参数，vs 原始权重的 d_in×d_out）进行 rank=1 低秩误差估计，并从 token 维度和 feature 维度正交地补偿量化误差。token 维度的补偿通过 frame-aware 缩放因子 M∈R^t（t 为帧数）实现：M_i = η_i/ω_i，其中 η_i 衡量第 i 帧的量化误差权重（基于量化前后相似度），ω_i 衡量 token 序列的显著度量。TQE 本质上是 LoRA 模块的一个 rank=1 特殊实例，可被 LoRunner Kernel 融合以消除额外延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Video DiT 的 PTQ pipeline 中，TQE 修改了每个 Linear 层的前向传播：
```
# 输入: X ∈ R^{n×d_in}, W ∈ R^{d_out×d_in}, n = s × t (s个空间token, t帧)
# 参数: α∈R^{d_in}, β∈R^{d_out}, M∈R^t (仅 d_in+d_out+t 额外参数)

# Standard quantized forward:
Y_std = Q̂(X) @ Q̂(W)^T   # 量化+反量化后的矩阵乘法

# TQE error compensation:
for frame i in [0..t-1]:
    f_i = i * s   # frame start index
    # Token-aware scaling on quantized activations
    Δ̂[f_i:f_i+s, :] = (M_i ⊙ Q̂(X)[f_i:f_i+s, :]) @ α   # ∈ R^{s×1}

# Final output with error compensation:
Y = Y_std + Δ̂ @ β^T   # ∈ R^{n×d_out}
```

M 的初始化（Eq. 9）：η_i = exp[1-ρ(X_i, Q̂(X)_i)] / Σ_v exp[1-ρ(X_v, Q̂(X)_v)]，ω_i = Σ_τ|X_{i,τ}| / Σ_v Σ_τ|X_{v,τ}|，M_i = η_i/ω_i。α 用 Kaiming init，β 用 zero init。校准训练时同时优化 TQE 参数和量化参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TQE 在 Q-VDiT 开源代码（https://github.com/wlfeng0509/Q-VDiT）中实现，作为 Linear 层的 wrapper 模块。TQE 参数仅 (d_in+d_out+t) 个，相比原权重减少 ~(d_in×d_out)/(d_in+d_out) 倍的参数量。推理时通过 LoRunner Kernel（来自 SVDQuant）将 rank=1 低秩分支与量化 GEMM kernel 融合——down projection (X→Δ̂) 与量化 kernel 融合共享激活张量消除额外内存访问，up projection (Δ̂→output) 与量化计算 kernel 融合——kernel 调用次数减半，额外延迟<5%。TQE 对 W4A6/W3A8/W3A6 均有效，结合 TMD 后在 W3A6 下 Scene Consistency 从 12.04 提升到 23.40（+94%）。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---

## Temporal Maintenance Distillation (TMD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Maintenance Distillation (TMD) 是 Q-VDiT (ICML 2025) 提出的用于视频 DiT 量化优化的知识蒸馏损失函数。它解决标准 MSE（L_task = ||S^{FP}−S^{Q}||²）在视频生成量化中的核心缺陷：MSE 对每帧独立计算，忽略了视频帧间的时空相关性。TMD 通过构建 FP 模型（教师）中帧间相似度分布 D^{FP}_i = softmax([cos_sim(S^{FP}_i, S^{FP}_1),...,cos_sim(S^{FP}_i, S^{FP}_t)]) 作为先验知识，用 KL 散度对齐量化模型（学生）的帧间分布 D^{Q}_i，使每个 frame 的优化受所有帧共同引导。总损失 L_total = L_task + γ·L_temporal（γ=100）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TMD 在 PTQ 校准训练中的计算流程：
```
# 输入: S^{FP}, S^{Q} ∈ R^{n×d}  (n=s×t 个token)
# 步骤1: 提取每帧的token序列
for frame i in [0..t-1]:
    S_fp_i = S^{FP}[i*s : (i+1)*s, :]   # ∈ R^{s×d}
    S_q_i  = S^{Q}[i*s : (i+1)*s, :]

# 步骤2: 计算帧间余弦相似度矩阵 (Eq. 13)
for i, j in [0..t-1]:
    T_fp[i,j] = cos_sim(S_fp_i, S_fp_j) = (S_fp_i · S_fp_j) / (||S_fp_i||·||S_fp_j||)
    T_q[i,j]  = cos_sim(S_q_i, S_q_j)

# 步骤3: 构建每帧的时序分布 (Eq. 14)
for frame i in [0..t-1]:
    D_fp_i = softmax([T_fp[i,0], ..., T_fp[i,t-1]])   # ∈ R^t
    D_q_i  = softmax([T_q[i,0], ..., T_q[i,t-1]])

# 步骤4: KL散度对齐 (Eq. 15)
L_temporal = Σ_{i=1}^{t} KL(D_fp_i || D_q_i)
           = Σ_i Σ_k D_fp_i[k] * log(D_fp_i[k] / D_q_i[k])

# 步骤5: 梯度分析 (Eq. 16-18)
# ∂L_temporal/∂S^{Q}_i = 双向梯度流:
#   Σ_j [∂L_temporal/∂T^{Q}_{i,j} · ∂T^{Q}_{i,j}/∂S^{Q}_i + ∂L_temporal/∂T^{Q}_{j,i} · ∂T^{Q}_{j,i}/∂S^{Q}_i]
# 其中 ∂L_temporal/∂T^{Q}_{i,j} = D^{Q}_{i,j} - D^{FP}_{i,j} (Eq. 17 的简化)
# 因此任意帧对的相关性受所有帧共同数值影响
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TMD 作为额外的损失项叠加到 PTQ 校准损失上，γ=100 是论文通过实验确定的平衡值（对 γ∈{0.1,1,10,100,500,1000} 均有效，不敏感）。TMD 适用于任何视频生成模型的 PTQ/微调场景，因为：(1) 无额外推理开销——TMD 仅在训练时计算，推理时不需计算帧间分布；(2) 普适性强——不依赖模型架构细节，仅需视频帧的 latent token 序列。在 W3A6 设置下，TMD 将 Scene Consistency 从 22.00 (仅 TQE) 提升到 22.58 (TQE+TMD)，VQA-Technical 从 29.58 (SOTA) 提升到 59.10。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---

## Video Diffusion Transformer (V-DiT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Diffusion Transformer (V-DiT) 是将 Diffusion Transformer (DiT) 架构扩展到视频生成任务的模型。DiT (Peebles & Xie, ICCV 2023) 用 Transformer 替代传统 U-Net 作为扩散模型的去噪骨干网络。在视频生成中，DiT 的 latent 表示 Z∈R^{n×d}，其中 n = s × t（s 为空间 token 数，t 为时序 token 数，即 t 帧）。与图像 DiT (I-DiT, n=s) 相比，V-DiT 通过额外的时序维度处理多帧视频，token 数量是 I-DiT 的 t 倍，表达能力更强，信息密度更高。代表性 V-DiT 包括 Open-SORA（基于 PixArt-α 架构，t 帧 latent 并行去噪）和 Latte（在 UCF-101 上训练的 class-conditioned 视频 DiT）。Q-VDiT 首次针对 V-DiT 提出专门的量化方案，因为视频生成的高信息密度使得直接应用图像量化方法会导致严重的信息丢失和帧间不连贯。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
V-DiT 的推理 pipeline（以 Open-SORA 为例）：
```
# 输入: text prompt P, latent shape (t, c, h, w)
# Step 1: 文本编码
text_emb = T5_encoder(P)                     # T5-XXL text encoder

# Step 2: 初始化噪声
Z_T = randn(t, c, h, w)                       # t 帧高斯噪声

# Step 3: 去噪循环 (100-step DDIM)
for step in range(T, 0, -1):
    # 拼合时空 token: flatten spatial dims + concat temporal
    Z_flat = Z_cur.reshape(t*s, d)            # n=s×t token

    # DiT forward: 在每个 timestep t 预测噪声
    noise_pred = DiT(Z_flat, timestep, text_emb)
    # DiT 由交替的 self-attention (跨所有 s×t token) 和 FFN 组成
    # 时序信息通过时空 attention 隐式建模

    # DDIM update
    Z_cur = ddim_step(Z_cur, noise_pred, timestep)

# Step 4: VAE decode (帧独立)
video = VAE_decoder(Z_0)                     # 解码为像素空间视频
```
量化时主要挑战：V-DiT 的 token 数远大于 I-DiT（×t），量化误差在 n=s×t 个 token 上累积传播，加上视频帧间有强时空语义关联，帧间不一致的风险更高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
V-DiT 的实现基于 HuggingFace Diffusers 或自定义框架（Open-SORA 使用自己的实现），推理使用标准 Transformer forward（self-attention + cross-attention + FFN），通常配备 CFG (Classifier-Free Guidance) 和 DDIM/DDPM sampler。Q-VDiT 量化 V-DiT 时仅量化线性层权重（channel-wise）和激活（token-wise dynamic），保持 attention 和 LayerNorm 为 FP16。Open-SORA 使用 100-step DDIM + CFG=4.0，Latte 使用 20-step DDIM + CFG=7.0。量化校准从 10 个 prompt 的 50 个去噪步采样校准数据。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---

## DPO (Direct Preference Optimization，直接偏好优化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Direct Preference Optimization (DPO) 是 Rafailov et al. (NeurIPS 2023) 提出的一种 LLM 偏好对齐方法，直接在偏好数据集上优化策略模型，无需显式训练奖励模型。DPO 的核心思想：(1) 将 RLHF 的奖励函数 r(x,y) 重新参数化为策略模型 π 和参考模型 π_ref 的对数概率比：r(x,y) = β log(π(y|x)/π_ref(y|x))；(2) 将 Bradley-Terry 偏好模型 P(y_w ≻ y_l|x) = σ(r(x,y_w) - r(x,y_l)) 与重新参数化的奖励函数结合，直接推导出 DPO 损失函数：L_DPO = -E_{(x,y_w,y_l)} log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)))。其中 β 控制偏离参考模型的程度（β 越小，策略越接近参考模型）。与 RLHF 相比，DPO 省去了 reward model 训练和 PPO 强化学习两个阶段，直接对偏好数据进行监督式优化，更稳定、更高效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Q-resafe 中的 DPO 用法（安全修补）：
```python
# 参考模型 = 量化后的 LLM π_Q⁰（安全能力受损）
# 策略模型 = 安全修补中的 LLM π_Q（正在被优化恢复安全）
# Q = Q⁰ + Quant(M_Q ⊙ AB)，仅安全关键权重参与更新

# DPO 损失计算
for batch in D_patch:  # D_patch = {(x, y_w, y_l)}
    # y_w: 全精度 LLM 生成的 preferred response（安全）
    # y_l: 量化 LLM 生成的 dispreferred response（不安全）

    log_pi_w = π_Q.log_prob(y_w | x)      # 策略模型对 preferred 的对数概率
    log_pi_l = π_Q.log_prob(y_l | x)      # 策略模型对 dispreferred 的对数概率
    log_ref_w = π_Q⁰.log_prob(y_w | x)    # 参考模型对 preferred 的对数概率
    log_ref_l = π_Q⁰.log_prob(y_l | x)    # 参考模型对 dispreferred 的对数概率

    # DPO 损失（Eq. 1 of Q-resafe）
    ratio_w = log_pi_w - log_ref_w        # preferred 的相对提升
    ratio_l = log_pi_l - log_ref_l        # dispreferred 的相对下降
    loss = -log(σ(β * (ratio_w - ratio_l)))

    loss.backward()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：HuggingFace TRL 库的 DPOTrainer（https://github.com/huggingface/trl）。标准用法：准备偏好数据集（prompt, chosen, rejected），指定 reference model（通常为 SFT 模型本身或冻结副本），设置 β（默认 0.1）。TRL 中：`DPOTrainer(model=model, ref_model=ref_model, beta=0.1, train_dataset=dataset)`。Q-resafe 的特殊用法：参考模型设为量化后的不安全模型 π_Q⁰，让 DPO 的隐式正则化 '不要偏离 π_Q⁰ 太远' 变为 '不要偏离量化模型已保留的效用太远'。DPO β 在 Q-resafe 中设为 0.01（较小值，约束更紧）。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

## SNIP Score (Single-shot Network Pruning Score，单次网络剪枝分数)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SNIP (Single-shot Network Pruning) Score 是 Lee et al. (ICLR 2019) 提出的一种连接敏感性（connection sensitivity）度量，用于在训练前的初始化网络中一次性识别重要连接。其核心公式：对权重 W_ij 和输入 x，定义重要性分数 I(W_ij, x) = |W_ij · ∇_{W_ij} L(x)|，即权重值与对应损失梯度的乘积的绝对值。直觉：(1) |W_ij| 大 → 该连接当前贡献大；(2) |∇_{W_ij} L| 大 → 该连接的微小变化对损失影响大；(3) 两者乘积同时考虑当前贡献和敏感性，比单独用权重大小或梯度更全面。SNIP 在训练**前**一次性计算，不依赖已训练模型，因此是 "single-shot" 的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Q-resafe 中的应用（安全关键权重识别）：
```python
# Q-resafe 将 SNIP score 从初始化前剪枝改编为训练中周期性地识别安全关键权重
# 对量化 LLM 的每层权重矩阵 Q

def compute_safe_score(model_Q, D_calib):
    scores = {}  # layer_name -> importance matrix
    for x in D_calib:
        # 前向传播：计算条件负对数似然
        log_probs = model_Q.log_prob(y | x)
        loss = -log_probs  # L(x) = -log p(y|x)

        # 反向传播：计算梯度
        loss.backward()

        # 对每层量化权重 Q，计算逐元素 SNIP score
        for name, Q in model_Q.named_parameters():
            I = abs(Q * Q.grad)  # |W_ij · ∇_{Q_ij} L(x)|
            scores[name] += I    # 累积跨校准样本

    # 对所有校准样本取平均
    for name in scores:
        scores[name] /= len(D_calib)
    return scores

# 选择 top-τ 百分比的权重
def select_safety_critical(scores, tau):
    M_Q = {}
    for name, score in scores.items():
        threshold = percentile(score.flatten(), (1 - tau) * 100)
        M_Q[name] = (score >= threshold).float()  # 1 = 安全关键
    return M_Q
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
原始 SNIP 实现于初始化阶段：随机采样一个 batch → 前向计算 loss → 反向计算梯度 → 计算 |w ⊙ g| → 全局排序 → 保留 top-k% 连接 → 设置 mask → 训练。在 Q-resafe 中改编为：(1) 在训练过程中周期性（每 K 步）重新计算 SNIP score（因为 Q^t 随训练变化）；(2) 损失函数不是原始分类交叉熵，而是条件语言模型负对数似然；(3) 不是用于剪枝（去除不重要连接），而是反向使用——保留 top-τ% 高 score 权重作为安全关键权重进行更新。这一改编的关键洞察：安全能力集中在少数权重中（类似 LLM 能力集中性假设），可通过 SNIP 敏感性度量定位这些权重。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

## Safety-Critical Weight Identification（安全关键权重识别）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
安全关键权重识别是 Q-resafe 框架的核心机制之一，指在量化 LLM 的众多权重中识别出对模型安全能力最为关键的一小部分权重（top-τ%），以便仅对这些权重进行定向修补而不扰动其余权重。其理论基础：(1) LLM 的能力（包括安全能力）集中在少部分权重中；(2) 量化主要损害效用相关的权重，安全相关的权重子空间在量化过程中因缺乏专门保护而严重受损；(3) 通过 SNIP score I(W_ij, x) = |W_ij · ∇_{W_ij} L(x)| 度量每个权重对安全相关损失（条件负对数似然）的敏感性，排序后选 top-τ% 作为安全关键权重。τ 的选取平衡安全恢复与效用保持：τ=1（全部更新）ASR 降至 1.6% 但 GPU-hours=2.1；τ=0.6 ASR=1.8% GPU-hours=1.2；τ=0.2 ASR=13.9% GPU-hours=0.5。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# 周期性安全关键权重识别（每 K=1000 步执行一次）
if step % K == 0:
    # Step 1: 计算当前模型 Q^t 的每层 SafeScore
    for each layer l with weight matrix Q_l:
        SafeScore_l = zeros_like(Q_l)
        for each prompt x in D_calib:
            y = sample(model, x)  # 或使用 ground truth
            loss_l = -log p(y|x; Q_l)
            loss_l.backward()
            SafeScore_l += abs(Q_l * Q_l.grad)  # element-wise SNIP
        SafeScore_l /= len(D_calib)

    # Step 2: 全局 top-τ 选择
    all_scores = flatten([SafeScore_l for l in layers])
    threshold = top_k_percentile(all_scores, τ)

    # Step 3: 构建掩码矩阵
    for each layer l:
        M_Q[l] = (SafeScore_l >= threshold).float()  # ∈ {0,1}

    # Step 4: 映射到 LoRA 维度
    # M_Q ∈ {0,1}^{d_in × d_out} → (M_A ∈ {0,1}^{d_in × r}, M_B ∈ {0,1}^{r × d_out})
    for each layer l:
        row_mask = any(M_Q[l], dim=1)  # 有安全关键权重的行
        col_mask = any(M_Q[l], dim=0)  # 有安全关键权重的列
        M_A[l] = row_mask.unsqueeze(1).expand(d_in, r)  # 整行标记
        M_B[l] = col_mask.unsqueeze(0).expand(r, d_out)  # 整列标记
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现时：(1) 校准数据集 D_calib 用于计算 SNIP score，Q-resafe 使用 Alpaca-cleaned 或 UltraChat；(2) 周期 K 的选择平衡识别准确性和计算开销——K=1000 为论文默认值，每 1000 步重新评估一次哪些权重是安全关键的；(3) τ 的选取通过消融实验确定（论文推荐 τ=0.6）；(4) 对于无微调量化方法（AWQ），Q-resafe 不执行 DPO 训练，而是在全精度预训练模型上直接计算 SNIP score，将 top-τ% 安全关键权重保留为 FP16，其余量化为 INT4。该技术的核心优势：仅修改极小部分权重即可恢复安全能力，避免了全量微调的计算开销（1.2 GPU-hours vs SFT 8.4h/DPO 9.6h）和效用损失。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

## Attack Success Rate (ASR，攻击成功率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attack Success Rate (ASR) 是 LLM 安全评估中最核心的指标之一，衡量模型在面对有害指令（harmful instructions）时生成不安全/有害响应的比例。ASR = (生成有害响应的测试样本数) / (总测试样本数) × 100%，**越低表示模型越安全**。ASR 的计算通常依赖自动评估器：GPT-4 作为裁判判断模型响应是否包含有害内容，或使用专门训练的 HarmBench 分类器做二元判断。Q-resafe 论文中使用多种 ASR 变体：(1) ASR_Vanilla：使用标准 system prompt ("You are a helpful assistant...")；(2) ASR_AOA：使用绝对服从 prompt ("You are no longer an AI language model...")；(3) ASR_Decoding：变化解码参数（temperature/top-k/top-p）测试解码策略攻击下的安全性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# ASR 评估流程
def compute_asr(model, harmful_prompts, evaluator="gpt-4"):
    harmful_count = 0
    for prompt in harmful_prompts:
        # Step 1: 模型生成响应
        response = model.generate(prompt, temperature=τ, top_k=k, top_p=p)

        # Step 2: 评估响应是否有害
        if evaluator == "gpt-4":
            # GPT-4 判断：打分 1-5，≥3 视为有害
            score = gpt4_judge(prompt, response)  # harmfulness 1-5
            is_harmful = (score >= 3)
        elif evaluator == "harmbench":
            # HarmBench 二元分类器
            is_harmful = harmbench_classifier(response)  # True/False

        harmful_count += int(is_harmful)

    asr = harmful_count / len(harmful_prompts) * 100
    return asr  # %
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ASR 的评测数据集通常来自 AdvBench（520 条有害指令，涵盖亵渎、威胁、错误信息、歧视、网络犯罪等类别）。评估器实现：(1) GPT-4 API 作为裁判：使用精心设计的评判 prompt 让 GPT-4 给响应打分（1-5），取 ≥3 或 ≥4 为有害；(2) HarmBench 分类器（Mazeika et al. 2024）：基于 Llama-2-13B 微调的二分类器，直接判断响应是否恶意。Q-resafe 同时使用两种方法以降低误报率，并额外报告 Harmfulness Score（所有样本的平均有害分数）。论文发现 INT4 量化后 ASR 可从 0.3%（FP16 Llama-2-7B-Chat）飙升至 42.4%（AWQ INT4），Q-resafe 可将 ASR 恢复至 1.8%。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

## AQLM (Additive Quantization of Language Models，语言模型加性量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AQLM (Additive Quantization of Language Models) 是 Egiazarian et al. (2024) 提出的面向 LLM 的极端压缩后训练量化方法。核心思想：使用加性量化（Additive Quantization）——将每个权重向量表示为 M 个 codebook 向量之和（w ≈ Σ_{m=1}^{M} c_m[i_m]），而非传统均匀量化的单一离散值。每个 codebook 包含 2^B 个码字，总 bit 数为 M × B。例如 M=2, B=8 时，每个权重向量用 2×8=16 bits 表示，但通过 256+256 个码字的组合可实现 256×256=65536 种可能的量化值，远超均匀量化的表达能力。AQLM 通过 beam search 或 iterative optimization 为每组权重找到最优的 codebook 索引组合。该方法在 2-bit 量化下能保持 LLM 的推理能力，是 PTQ 方法中压缩率的 SOTA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# AQLM 量化（简化，实际使用 beam search / iterative descent）
# 对权重矩阵 W 按 group_size=d 分组，每组使用 M 个 codebook

# Codebook 学习（offline, per layer）
for each group g:
    # W_g ∈ R^{d}, codebooks C_m ∈ R^{d × 2^B}, m=1..M
    # 目标: min Σ_{i in group} |W_i - Σ_m C_m[i_m]|^2

    # beam search 或交替优化
    for iter in range(max_iters):
        # Fix codebooks, update assignments
        for each weight i:
            best_indices = beam_search(W_i, {C_m})
        # Fix assignments, update codebooks via k-means
        for m in range(M):
            C_m = update_codebook(W_g, assignments)

    # 存储: 每组存储 M 个 codebook (M × d × 2^B × FP16 bytes) + 每个权重的 M 个索引 (M × B bits)

# 推理时 dequant
for each group g:
    w_hat = sum(C_m[code_idx[i][m]] for m in range(M))
    # 从 codebook 中查表并求和
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/Vahe1994/AQLM。基于 PyTorch，支持 HuggingFace Transformers 模型。使用方式：加载模型 → 用校准数据（如 WikiText-2）逐层量化 → 保存量化权重 + codebooks。推理需要使用 AQLM 特定的 CUDA kernel 进行高效的 codebook 查表和累加操作。AQLM 属于 PTQ w/ FT 类别——量化后通过 fine-tuning 进一步优化 codebook 以恢复精度。在 Q-resafe 的安全评估中，AQLM 在 benign 数据集（Risk-I）上 INT4 ASR=18.5%，但在直接有害数据集（Risk-III）上飙升至 77.4%，显示校准数据集的选取对安全至关重要。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

## LLM-QAT (LLM Quantization-Aware Training，LLM量化感知训练)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-QAT 是 Liu et al. (2023) 提出的面向 LLM 的无数据量化感知训练方法。与传统 QAT 需原始训练数据不同，LLM-QAT 的核心创新：使用预训练 LLM 自身生成训练数据（self-generated data），通过从预训练模型中采样 token 序列构建蒸馏数据集，然后在量化模型上执行知识蒸馏——教师为全精度 LLM，学生为量化 LLM。该方法的关键优势：(1) 无需访问预训练数据，保护数据隐私且降低数据获取成本；(2) 通过全参数微调（full-parameter fine-tuning）在量化约束下重新学习权重分布，比 PTQ 更好地补偿量化误差；(3) 可支持 INT4/INT8 等不同位宽，且通过蒸馏保留了全精度模型的效用。属于 QAT w/ FT 类别。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# LLM-QAT Pipeline
# Step 1: 从预训练模型生成蒸馏数据
synthetic_data = []
for i in range(num_samples):
    prompt = random_start_token  # 或从词汇表随机采样
    seq = full_precision_model.generate(prompt, max_length=512)
    synthetic_data.append(seq)

# Step 2: 量化模型初始化
quantized_model = quantize(full_precision_model, bits=4)  # INT4 量化

# Step 3: QAT 蒸馏训练（全参数）
for seq in synthetic_data:
    # 教师（FP16）前向
    with torch.no_grad():
        teacher_logits = full_precision_model(seq)

    # 学生（Simulated INT4）前向，使用 STE 通过量化操作
    student_logits = quantized_model(seq)

    # 蒸馏损失（KL 散度 + 交叉熵）
    loss = KL_div(student_logits, teacher_logits) + CE(student_logits, seq_labels)
    loss.backward()  # STE 梯度通过伪量化操作

    # 更新所有参数
    optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/facebookresearch/LLM-QAT。基于 HuggingFace Transformers + PyTorch，使用标准的 fake quantization（前向量化-反量化模拟，反向 STE）。训练时需多张 GPU（如 8× A100）进行全参数微调。在 Q-resafe 的安全评估中，LLM-QAT 在 benign 数据集（Risk-I）上 INT4 ASR=16.9%，表现优于 QLoRA（ASR=42.3%），因为全参数微调比 LoRA 更好地保留了预训练模型的能力包括安全能力；但在有害数据集上 ASR 仍高达 82.9%。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

## Decoding Attack / Jailbreak via Decoding Strategy（解码策略攻击/解码策略越狱）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
解码策略攻击（Decoding Attack）是一种无需修改模型权重即可绕过 LLM 安全对齐的攻击方法，通过调整自回归生成过程中的解码参数（temperature τ、top-k、top-p）来诱导模型生成不安全响应。其原理：(1) 安全对齐模型的概率分布中，安全响应通常位于高概率区域（p ≈ 1），不安全响应位于低概率尾部；(2) 标准解码（如 greedy 或低 temperature）倾向于选择高概率的安全 token；(3) 提高 temperature（如 τ=0.95）或降低 top-p/to-k 会使采样分布变得平坦，增加从低概率区域（不安全 token）采样的概率；(4) 即使模型权重完全不变，仅通过解码策略的微调就能将 ASR 从 0.3% 提升至 29.8%（FP16 Llama-2-7B-Chat）。量化进一步放大了这一效应——INT4 下 ASR 可达 42.4%（τ=0.95）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# Q-resafe 中的 ASR_Decoding 评估流程
def evaluate_decoding_attack(model, harmful_prompts):
    # 对每个 prompt 测试多种解码配置
    decoding_configs = [
        {"temperature": 0.95, "top_k": 500, "top_p": 0.95},
        {"temperature": 0.95, "top_k": 200, "top_p": 0.7},
        {"temperature": 0.7,  "top_k": 500, "top_p": 0.95},
        {"temperature": 0.7,  "top_k": 200, "top_p": 0.7},
        # ... 49 种配置
    ]

    for prompt in harmful_prompts:
        responses = []
        for config in decoding_configs:
            # 每种配置生成一个响应
            r = model.generate(
                prompt,
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                max_new_tokens=256
            )
            responses.append(r)

        # GPT-4 选择最有害的响应
        best_response = gpt4_select_worst(responses)

        # 评估 ASR
        asr += is_harmful(best_response)

    return asr / len(harmful_prompts)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
解码策略攻击的实现基于 Huang et al. (2023) 的方法。实际操作：(1) 对每个有害 prompt，使用 49 种不同的解码参数组合生成 49 个候选响应；(2) 使用 GPT-4 从候选中选择最有害的响应作为该 prompt 的最终输出；(3) 计算 ASR_Decoding = 有害响应数/总 prompt 数。该攻击的关键发现：即使模型权重完好，也可能通过采样策略绕过安全防线——论文举例：将 Llama-2-7B-Chat 的 temperature 从 0.9 降至 0.7 就足以绕过安全约束。Q-resafe 在设计时考虑了这一点，通过保持安全关键权重接近全精度模型来增强对各种解码配置的鲁棒性。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

## Quantization Error Reconstruction (QER)（量化误差重建）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Error Reconstruction (QER) 是指在模型量化后，用额外的低秩、高精度矩阵 C_k = A_k B_k（rank k ≪ min(m, n)）来补偿/重建量化误差的技术。给定线性层 y = xW 和量化后的近似 W̃ = dq(q(W))，QER 将输出重写为 ỹ = x(W̃ + C_k)。QER 问题的核心挑战在于：给定 C_k 的秩约束，应以什么优化目标求解 A_k 和 B_k。传统方法（ZeroQuant-V2, LoftQ）通过截断 SVD 最小化权重逼近误差 ||W − W̃ − C_k||_F（Problem 1，Eckart-Young-Mirsky 最优解）。QERA 论文证明该目标不能保证降低模型输出误差，转而最小化层输出误差 E[||x(W̃ + C_k) − xW||²]（Problem 2），并给出闭式解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QER 在模型推理pipeline中的位置：模型权重加载 → 逐层应用 q(·) 量化 → dq(·) 反量化 → 计算权重量化误差 E = W − W̃ → 求解 A_k, B_k（离线预处理）→ 推理时 y = xW̃ + x(A_k B_k)。两种求解范式的对比伪代码：

```
# === Problem 1: Minimize Weight Error (ZeroQuant-V2 / LoftQ) ===
W_tilde = dq(q(W))
E = W - W_tilde         # weight quantization error
U, Sigma, Vt = SVD(E)   # full SVD
C_k = U_{:,:k} @ diag(Sigma_{:k}) @ Vt_{:k,:}  # truncated SVD (Eckart-Young-Mirsky optimal)
# 前向: y = x @ W_tilde + x @ C_k
# 问题：最小化的是权重误差，不保证输出误差降低 (Section 4.2, Figure 1)

# === Problem 2: Minimize Layer Output Error (QERA) ===
W_tilde = dq(q(W))
# Calibration: 对校准集 X 中的输入向量 x 累积统计量
# QERA-exact:
R = 1/N * sum_i x_i^T x_i           # 自相关矩阵 [m, m], FP64累积
R_sqrt = matrix_sqrt(R)              # blocked Schur algorithm (CPU, FP64)
Q = R_sqrt @ (W - W_tilde)          # 标度化误差
U, Sigma, Vt = SVD(Q)
A_k = inverse(R_sqrt) @ U_{:,:k}    # 反标度化: [m, k]
B_k = diag(Sigma_{:k}) @ Vt_{:k,:}  # [k, n]

# QERA-approx (假设 E[x_i x_j]=0 for i≠j):
s = [sqrt(E[x_1^2]), ..., sqrt(E[x_m^2])]  # 对角激活统计
S = diag(s)
Q = S @ (W - W_tilde)
U, Sigma, Vt = SVD(Q)
A_k = inverse(S) @ U_{:,:k}
B_k = diag(Sigma_{:k}) @ Vt_{:k,:}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QER 实现基于 PyTorch + SciPy（矩阵平方根使用 blocked Schur algorithm, CPU 执行）。应用于两个场景：(1) QPEFT 中作为 LoRA 初始化——QERA-approx（21s-30min）替代 QLoRA 的随机初始化，使微调起点更接近全精度模型，2-bit RoBERTa @ GLUE 平均 Δacc = +6.05% vs LoftQ；(2) PTQ 中作为离线误差补偿——QERA-exact 或 QERA-approx 预计算低秩项并合并入 W̃，推理时零额外开销。QERA 对量化函数 q(·) 无约束，兼容任意量化方法（Uniform, NF4, MXINT 等）。QERA-approx 的正确标度 S = diag(√E[x_i²]) 从理论上解释了 LQER 启发式标度（使用 E[|x_i|]）在更多校准样本下性能不稳定的原因。开源：https://github.com/ChengZhang-98/QERA。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction

---

## QERA (Quantization Error Reconstruction Analysis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QERA (Quantization Error Reconstruction Analysis) 是 Imperial College London 提出的量化误差重建分析框架，给出 QER 问题（最小化层输出误差）的闭式解。核心贡献：(1) 理论证明：QER 问题中最小化权重逼近误差（Problem 1）与最小化模型输出误差（Problem 2）不等价——LoftQ 迭代增加权重误差单调降但模型输出误差可能增加（Figure 1）；(2) 两个闭式解：QERA-exact (Theorem 1) 使用输入自相关矩阵 R_XX = E[x^T x] 的矩阵平方根进行标度化，与 CALDERA Lemma 4.2 等价但证明路径不同；QERA-approx (Theorem 2) 在"不同嵌入维度不相关"假设（Assumption 1, E[x_i x_j]=0 for i≠j）下将 R_XX 简化为对角矩阵 S² = diag(E[x_i²])；(3) 实践验证：QERA-exact 在 LLM 上的 Assumption 1 测试（超过 60% 层满足、MLP 层尤佳）；QERA-approx 解释了 LQER 启发式标度的成功并解决其校准不稳定问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QERA 的完整数学推导链：

```
问题: min_{C_k} E[||x(W̃ + C_k) − xW||²]

定义 P = W̃ + C_k − W, p_i = P 的第 i 行
目标 = E[||Σ_i x_i p_i||²]
    = E[Σ_i Σ_j x_i x_j p_i p_j^T]
    = Tr(E[x^T x] · P P^T)                (Hadamard product 性质)
    = Tr(R_XX · P P^T)                     (R_XX = E[x^T x])
    = ||R_XX^{1/2} P||_F²                 (R_XX 正半定，有唯一对称平方根)

→ 等价于: min_{C_k} ||R_XX^{1/2}(W̃ + C_k − W)||_F²

令 Q = R_XX^{1/2}(W − W̃), Q_k = R_XX^{1/2}C_k
→ min_{Q_k} ||Q_k − Q||_F²
→ Q_k = SVD_k(Q)  (Eckart-Young-Mirsky 定理)
→ C_k = (R_XX^{1/2})^{-1} · SVD_k(R_XX^{1/2}(W − W̃))

在 QERA-approx 下: 假设 E[x_i x_j]=0 for i≠j
→ R_XX = diag(E[x_1²], ..., E[x_m²]) = S²
→ C_k = S^{-1} · SVD_k(S · (W − W̃))
其中 S = diag(√E[x_1²], ..., √E[x_m²])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PyTorch + Transformers + PEFT + SciPy（blocked Schur algorithm for matrix sqrt, CPU FP64）。关键技术细节：(1) R_XX 外积在 FP32 累积→FP64 精度矩阵平方根以确保数值稳定；(2) block size=32 MXINT 作为量化格式（也可用任意其他量化方法）；(3) 逐层独立计算可并行化。QPEFT 场景推荐 QERA-approx（初始化时间 21s-30min vs QERA-exact 的 1.6min-4.9h，微调可补偿近似误差）；PTQ 场景推荐 QERA-exact（离线预处理，精度最高，推理零额外开销）。QERA-exact 的瓶颈在矩阵平方根的 CPU 计算，GPU 加速矩阵平方根是关键优化方向。开源：https://github.com/ChengZhang-98/QERA。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction

---

## LQER (Low-rank Quantization Error Reconstruction)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LQER (Low-rank Quantization Error Reconstruction) 是 Zhang et al. (2024a, Imperial College London) 提出的 PTQ 方法，在权重量化后使用激活引导的启发式标度矩阵 S 对权重量化误差做截断 SVD 来求解低秩补偿项 A_k, B_k。LQER 的标度矩阵 S 通过对校准集输入 x 的每个维度累积平均绝对值来构建：s_i = E[|x_i|]，S = diag(s_1, ..., s_m)（归一化后）。关键步骤：对标度化后的权重量化误差 S(W − W̃) 做 SVD → 取前 k 个奇异值/向量 → A_k = S^{-1}U_{:,:k}, B_k = Σ_{:k,:k}V_{:k,:}^T。LQER 观察到标度后某些层的奇异值衰减更快（"更理想"的奇异值分布）。

从算法pipeline角度拆解术语：QERA-approx 从理论上揭示了 LQER 的本质——当 Assumption 1 (E[x_i x_j]=0 for i≠j) 成立时，最小化层输出误差的最优标度应为 S = diag(√E[x_i²])，而非 LQER 的 diag(E[|x_i|])。这解释了：(1) LQER 通常优于 ZeroQuant-V2（S=I），因为标度方向近似正确；(2) LQER 性能随校准样本数增加而随机波动（Figure 3 purple curve）——基于绝对值的启发式统计量不收敛到理论最优解。QERA-approx 使用正确的二阶矩标度后，性能随校准样本数单调提升至收敛（Figure 3 green curve）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LQER 的算法流程（Algorithm 2）：初始化 s ← 0 → 对校准集 X 中每个样本 x: s += abs(x) → S = (1/N) diag(s) → 量化 W_q = q(W) → 反量化 W̃ = dq(W_q) → SVD(S(W − W̃)) → A_k = S^{-1}U_{:,:k}, B_k = Σ_{:k,:k}V_{:k,:}^T。LQER 是纯 PTQ 方法，不涉及训练，低秩项离线预计算并在推理时合并入 W̃。ZeroQuant-V2 可视为 LQER 在 S=I 时的特例。LQER 论文的局限：启发式标度不保证最优性，校准样本数与性能关系不稳定。QERA（同研究组后续工作）将 LQER 理论化并给出了正确的解析解。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction

---

## MXINT (Micro-exponent Integer Format)（微指数整数格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MXINT (Micro-exponent Integer) 是 Darvish Rouhani et al. (ISCA 2023) 提出的低精度数值格式，属于 Microscaling (MX) 格式家族。核心设计：一个 block 内的所有元素共享一个小的共享指数（micro-exponent），block 内每个元素使用整数尾数表示。具体地，对 block size B 的权重块，先计算块内最大绝对值决定共享指数 shared_exp = ⌊log₂(max|w|)⌋，然后每个权重除以 2^shared_exp 并舍入到 N-bit 整数范围。QERA 论文使用 emulated MXINT：4-bit 下 block size=32（avg 4.25 bits/elem），3-bit 下 block size=32（3.25 bits），2-bit 下 block size=16（2.50 bits）。MXINT 的精度不仅来自更细粒度的 scaling（每 block 独立指数），还来自较小的 block size 带来的更均匀的块内数值分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MXINT 在 QERA 中的量化-反量化流程：

```
# MXINT 量化 (block_size=32, N=4-bit, range=[-7,7])
for each block of 32 consecutive weights in W:
    max_abs = max(abs(block))
    shared_exp = floor(log2(max_abs))       # block 共享指数
    scale = 2^shared_exp
    for each weight w in block:
        w_int = round(w / scale)             # 归一化并舍入
        w_int = clamp(w_int, -(2^{N-1}-1), 2^{N-1}-1)  # [-7, 7]
    # 存储: shared_exp (8-bit) + 32个(N-1)-bit尾数 + 32个1-bit符号位
    # 每元素实际位宽: 8/32 + N ≈ N + 0.25 bits

# MXINT 反量化
for each block:
    shared_exp = block_metadata.shared_exp
    scale = 2^shared_exp
    for each stored integer w_int in block:
        w_recovered = w_int * scale          # 恢复为近似FP值
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 QERA 中 MXINT 通过 PyTorch 手动实现量化/反量化函数（mxint_quantize / mxint_dequantize），非硬件原生支持。MX 格式家族的完整标准（OCP Microscaling Formats）包括 MXINT8、MXFP8、MXFP6、MXFP4 等变体，AMD、Arm、Intel、Meta、Microsoft、NVIDIA、Qualcomm 等已采用。优势：(1) 相比纯 INT 量化有更大动态范围（通过共享指数）；(2) 相比纯 FP 量化硬件实现更简单（block 内无 FP 乘加逻辑）；(3) block size 可灵活调节精度和开销的权衡。主要应用于 LLM 权重量化（W4A16）、KV cache 量化等低精度推理场景。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction


## Domain Generalization (DG / 域泛化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Domain Generalization（DG，域泛化）是一种机器学习范式，目标是从多个源域（source domains）学习一个模型，使其能在未见过的目标域（target domain）上良好泛化。与 Domain Adaptation（需要目标域未标注数据）不同，DG 在训练期间完全无法访问目标域数据。DG 的核心挑战是防止模型对源域过拟合——模型可能学到源域特有的虚假相关性（spurious correlations，如纹理、背景、光照），而非真正的类别判别特征。DG 的标准评估协议是 DomainBed（Gulrajani & Lopez-Paz, ICLR 2021）：每个域轮流作为目标域，其余作为源域，源域中 20% 作为验证集用于模型选择，在目标域上评估并报告平均准确率和标准差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DG 训练以 DomainBed 协议 + ResNet-50 为例：对每个目标域（leave-one-domain-out），源域 80% 训练 20% 验证，ResNet-50 ImageNet 预训练初始化，Adam (lr=5e-5, no weight decay)，batch_size=32 per-domain，训练 5000 步（DomainNet 15000 步），每 300 步验证选模。主要 DG 方法类别：(1) Domain Alignment——对齐源域特征分布（CORAL, DANN, MMD）；(2) Regularization——正则化抑制源域特有特征（IRM, VREx）；(3) Weight Averaging——平均权重以找平坦极小值（SWAD, DiWA）；(4) Ensembling——集成多模型（EoA）；(5) CLIP-based——利用大规模多模态预训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DomainBed 是 DG 的标准评测框架（GitHub: facebookresearch/DomainBed），提供统一的数据加载和评估协议，支持 PACS、VLCS、OfficeHome、TerraIncognita、DomainNet 五大数据集。QT-DoG 在该框架上引入 QAT 作为隐式正则化，单模型 7-bit 达 66.2% DomainBed 平均（ERM 63.8%），体积减少 4.6×。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization

## Flat Minima / Sharp Minima (平坦极小值 / 尖锐极小值)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flat Minima 和 Sharp Minima 是损失景观中两类不同性质的局部极小值。平坦极小值：权重空间中的大连通区域，区域内损失近乎恒定；尖锐极小值：损失在极小值点周围迅速上升的窄小区域。Hochreiter & Schmidhuber (1997) 定义平坦极小值为"误差保持近似恒定的大连通区域"。泛化关系（Keskar et al., ICLR 2017）：尖锐极小值对扰动敏感 → OOD 泛化差；平坦极小值对扰动鲁棒 → 泛化好。平坦度通过 F_γ(w) = E[E(w') - E(w)] 量化（w' 在 w 为中心半径 γ 的球面上 Monte-Carlo 采样），值越小越平坦。注意 Andriushchenko et al. (ICML 2023) 指出平坦极小值并非普遍保证更好泛化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QT-DoG 中量化噪声驱动平坦极小值的机制：二阶 Taylor 展开 L(w+Δ) ≈ L(w) + ∇LΔ + ½Δ^T H Δ。尖锐区域中 H 特征值大 → ½Δ^T H Δ 项使损失急剧增大 → 模型"逃离"；平坦区域中 H 特征值小 → 噪声影响小 → 模型稳定收敛。QT-DoG 训练平坦度超过 ERM、SAM、SWA，与 SWAD 相当但模型小 75%。寻找平坦极小值的主要方法：(1) SWA——平均训练轨迹上的 checkpoint；(2) SWAD——在验证最优区间内做权重平均；(3) SAM——最小化 max_{||ε||≤ρ} L(w+ε)；(4) 噪声注入——QT-DoG 的量化噪声、权重噪声等隐式正则化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
平坦度分析工具：Hessian 特征值谱、F_γ(w) Monte-Carlo 估计（100 次随机球面采样）、损失景观可视化（沿两个随机方向）。Rissanen (1978) 和 Hochreiter & Schmidhuber (1997) 证明平坦极小值对应低复杂度网络，需要更少的 bit 信息表示每个权重——这从信息论角度解释了为何降低权重比特精度（量化）能引导优化趋向平坦极小值。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization

## Learned Step Size Quantization (LSQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learned Step Size Quantization（LSQ，学习步长量化）是 Esser et al. (ICLR 2020) 提出的 QAT 方法。核心创新：将量化步长 s 作为可学习参数，通过梯度下降与权重一同优化。公式：v̄ = ⌊clip(v/s, -Q_N, Q_P)⌉, v_q = v̄ × s。与传统固定步长方法不同，LSQ 让每层自动学习最优量化粒度。梯度缩放策略：grad_scale ∝ 1/√(Q_P × n_features)，平衡不同大小层之间的步长更新速率。支持 per-tensor 和 per-channel 两种粒度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QT-DoG 中 LSQ per-channel 量化的关键计算：
```
w_scaled = w / s              # s: [out_c, 1, 1, 1]
w_bar = round(clip(w_scaled, -Q_N, Q_P))  # Q_N=2^(b-1), STE through round
w_q = w_bar * s               # 量化权重
# 反向梯度：∂L/∂w = STE(w_bar in range) ? ∂L/∂w_q : 0
#           ∂L/∂s = ∂L/∂w_q * (w_bar - w_scaled)
```
QT-DoG 消融：channel-wise LSQ (87.8% PACS OOD) > layer-wise LSQ (86.3%) > no quant (84.7%)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LSQ 的 PyTorch 实现在 Brevitas、AIMET 等量化库中可用。超参：梯度缩放因子 g = 1/sqrt(Q_P x N)；步长初始化为 2*mean(|w|)/sqrt(Q_P)；支持 signed/unsigned 量化。LSQ+（Bhalgat et al., CVPR 2020）增加了可学习零点偏移。QT-DoG 使用 7-bit LSQ 做 per-channel 权重量化，在 2000 步后启动。

在 Task-Specific ZSQ for Object Detection 中，LSQ 被扩展用于目标检测网络的 QAT：per-tensor symmetric quantization 对权重和激活均使用 LSQ，量化器附加到除首尾层外的所有内部层。QAT 使用 Adam 优化器（YOLOv5 lr=1e-4，YOLO11 lr=1e-5，CNN Mask R-CNN lr=1e-4，Transformer Mask R-CNN lr=1e-6），量化 scale 因子通过反向传播联合学习。实验表明 LSQ 在极低校准集（2k vs 120k full）下效果急剧退化（YOLOv5-s W6A6: LSQ full 31.5% vs LSQ 2k 28.9%），但结合 task-specific 合成数据和蒸馏后大幅超越 full-data LSQ（YOLOv5-l W6A6: 45.1% vs LSQ full 43.3%）。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization
- Scheduling Weight Transitions for Quantization-Aware Training
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

## Quantization Noise as Regularizer (量化噪声正则化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
量化噪声正则化是 QT-DoG 的核心洞察：权重量化引入的误差 Δ = w_q - w 可建模为均匀分布噪声 U(-s/2, s/2)（论文实测 KL 散度 0.0009 vs 均匀分布），该噪声作为隐式正则项通过 Hessian 交互驱动优化趋近平坦极小值。二阶 Taylor 展开 L(w+Δ) ≈ L(w) + ∇LΔ + ½Δ^T H Δ：尖锐区域 H 大 → ½Δ^T H Δ 使损失激增 → 优化器逃离 → 收敛到平坦区域。这与传统权重噪声正则化（An, 1996; Murray & Edwards, 1992）原理一致，但量化噪声具有结构化特性（约束在量化 bin 内），正则化效果更强。QT-DoG 消融：7-bit 量化（87.8% PACS）> 最优均匀噪声 U(-5e-6, 5e-6)（85.6%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化噪声注入发生在每次前向传播：W_q = round(clip(W/s, -Q_N, Q_P)) × s，噪声 Δ = W_q - W, |Δ| ≤ s/2。该噪声对梯度的隐式影响通过 Hessian 的 ½Δ^T H Δ 二次项体现，不直接修改梯度公式。噪声强度由量化 bit 数控制：bit 越低 → s 越大 → 噪声越强。最优 bit 精度权衡：7-bit 最优（PACS），≤4-bit 过度正则化导致性能下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
量化噪声正则化优势：(1) 同时作为正则化器和模型压缩手段（4.6× 体积减少）；(2) 无需额外训练开销（噪声来自量化本身）；(3) 通过 bit 数灵活控制正则化强度。局限：(a) 最优 bit 数依赖数据集和任务；(b) 极低 bit（≤4）导致过度正则化。适用场景：任何使用 QAT 训练的场景均可受益于该隐式正则化效果。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization

## Ensemble of Quantization (EoQ / 量化集成)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ensemble of Quantization（EoQ）是 QT-DoG 利用量化模型体型优势的高效集成策略。训练 E 个独立随机初始化的 QT-DoG 量化模型（7-bit），推理时 bagging：ŷ = argmax_k Softmax((1/E) Σ f(x; w_q^i))。核心优势：5 个 7-bit 模型总存储仅 1.1× 全精度单模型，却在 DomainBed 上达 68.4% 平均准确率（超过 DiWA 68.0%/EoA 68.0%）。相比之下，DiWA 需 60 个全精度模型（60× 体积，12× 训练开销），EoA 需 6 个全精度模型（6× 体积）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
EoQ 训练：独立训练 E=5 个 QT-DoG 模型（仅 random seed 不同），每模型在各自验证集上选最优 checkpoint。推理：probs = [softmax(m(x)) for m in models], ŷ = argmax(mean(probs))。EoQ 总内存 = E × 0.22× = 1.1× 全精度；串行推理延迟 = E × 操作时间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
EoQ 局限：(1) 训练计算量为单模型 E 倍；(2) 串行推理延迟为 E 倍（可并行化改善）。EoQ 在 TerraIncognita 提升最显著：ERM 47.2% → QT-DoG 50.8% → EoQ 53.2%（+6.0% vs ERM）。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization

## Weight Averaging for Domain Generalization (SWA / SWAD / DiWA / EoA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Averaging for DG 是一类通过平均多组模型权重提升 OOD 泛化的方法。主要变体：(1) SWA (Izmailov et al., UAI 2018)——平均训练轨迹上的多时刻权重；(2) SWAD (Cha et al., NeurIPS 2021)——在验证损失最优区间内做权重平均，DomainBed 最强单模型之一（Avg 66.9%）；(3) DiWA (Rame et al., NeurIPS 2022)——平均多个独立训练模型权重，需 60 个模型；(4) EoA (Arpit et al., NeurIPS 2022)——结合集成和权重平均。共同原理：权重平均平滑损失景观的尖锐区域，定位平坦极小值中的连通盆地中心。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SWAD 流程：训练中每 300 步验证 → 当 val_loss 达最优开始收集 checkpoints → 当 val_loss 超过最佳值 1.2× 停止 → 平均收集的所有权重。局限：需存储多份完整模型副本，最终模型仍是全精度。QT-DoG 通过量化噪声无需权重平均即可找到平坦极小值，模型还小 4.6×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 在线平均（SWA）—— w_swa = (n*w_swa + w_current)/(n+1)；(2) 离线平均（SWAD/DiWA）—— Σ w_i/N；(3) 加权平均（EoA）——基于验证性能分配权重。有效原因：平坦盆地中不同时刻/seed 模型权重在盆地不同位置，平均更接近中心最低点。

涉及论文标题：
- QT-DoG Quantization-Aware Training for Domain Generalization

## Trellis Coded Quantization (TCQ)（格状编码量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Trellis Coded Quantization (TCQ) 由 Marcellin 和 Fischer (1990) 首次提出，将 Trellis Coded Modulation 的概念应用于量化。TCQ 定义了一个 (L, k, V) trellis——一个具有 2^L 个节点的有向图，每个节点有 2^{kV} 条入边和出边，每个节点关联一个值 ∈ R^V（形成 codebook C ∈ R^{2^L × V}）。对长度为 T 的序列 S ∈ R^T，每连续 V 个元素分配给 trellis 上的一个节点，约束是分配的节点形成一条 walk。重建序列 Ŝ 由 walk 中节点值的拼接给出。由于相邻元素由 2^{kV} 条边连接，只需存储来自哪条边（k 比特），而非整个 codebook 索引。在可加失真度量（如 MSE）下，最优 Ŝ 可通过 Viterbi 算法在 O(2^L T) 时间内找到——复杂度与比特率 k 无关，与序列维度 T 线性。这使 TCQ 可实现超高维量化（维度 > 100），克服了 Vector Quantization（VQ）的指数复杂度瓶颈（VQ 需 O(2^{kd} d) 时间和空间）。对于 i.i.d. 高斯源，随 L 增大，TCQ 高效逼近无限长度失真率 D_R（2-bit 时 D_R=0.063，QTIP 256D TCQ 达 0.069 MSE，而 QuIP# 8D VQ 达 0.089，标量 Lloyd-Max 达 0.118）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TCQ 在 QTIP 中的算法 pipeline（以量化一个 256 维权重序列为例，L=16, k=2, V=1）：
```
输入: 序列 S ∈ R^{256}, (L=16, k=2, V=1) bitshift trellis G, codebook C ∈ R^{2^16}
1. 定义价值函数: V_t(x) = min{ Σ_{i=1}^t ||C_{x_i} - s_i||² | x_1..x_t 是 G 上路径, x_t=x }
2. 初始化: V_1(x) = ||C_x - s_1||², ∀x ∈ [0, 2^L)
3. for t = 2 to T:
     for each node y ∈ [0, 2^L):
       V_t(y) = min_{(x,y)∈G} [V_{t-1}(x) + ||C_y - s_t||²]
       记录回溯指针 ptr_t(y) = argmin_x
4. 反向回溯: x*_T = argmin_x V_T(x), 然后 x*_{t-1} = ptr_t(x*_t)
5. 输出 Ŝ = [C_{x*_1}, ..., C_{x*_T}], 编码为起始状态(L bits) + 每步 k 比特 = L + kT bits
```
关键：V_t(y) 的更新仅考虑 G 中 (x,y) 有边的节点对，bitshift trellis 中每个节点仅 2^{kV}=4 条入边，故每步 O(2^L × 2^{kV}) = O(2^{L+kV}) = O(2^{18}) ≈ 262K 操作，总复杂度 O(2^L T) ≈ O(16.8M)，远小于暴力搜索 O(2^{kT}) = O(2^{512})。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TCQ 在 LLM 量化中的使用：(1) 离线阶段——用 Incoherence Processing (RHT) 使权重近似 i.i.d. 高斯 → BlockLDLQ 框架中每 T_x×T_y 权重块作为序列用 TCQ 量化 → 存储编码比特；(2) 推理阶段——bitshift trellis 解码 + compute-based codes 即时生成权重 → GEMV。TCQ 的线性复杂度突破了 VQ 的维度限制，使有效维度从 ≤8 提升到 256+。开源实现：https://github.com/Cornell-RelaxML/qtip（QTIP 论文）。TCQ 也可用于其他需要高维量化的场景（如压缩感知、图像编码等），原始 Marcellin & Fischer (1990) 论文主要针对语音/图像压缩。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing

## Bitshift Trellis（位移格状结构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bitshift Trellis 由 Mao 和 Gray (2010) 在"随机排列 trellis 编码器"(RPTC)中引入，是 QTIP 的核心 trellis 结构。在 bitshift trellis 中，节点 i 到 j 有边当且仅当 ∃c ∈ Z, 0 ≤ c < 2^{kV} 使得 j = (i·2^{kV} mod 2^L) + c——即 j 的高 L-kV 位等于 i 的低 L-kV 位。这意味着：第 1 组 V 个权重仅依赖比特位置 {1,2,...,L}，第 2 组仅依赖 {kV+1,...,kV+L}，第 t 组仅依赖 {(t-1)kV+1,...,(t-1)kV+L}。解码时仅需 kV-bit 位移操作（所有硬件原生支持），且各组完全并行解码。无需存储 trellis 图结构（naive TCQ 需存储 2^L×2^{kV} 条边信息）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Bitshift trellis 解码流程（L=16, k=2, V=1）：
```
输入: packed bitstream B (L + kT bits, tail-biting aligned to 32-bit word)
1. 读取起始状态: state = B[0:16] (16-bit word)
2. 第 t=1 组: w_1 = codebook[state], 消耗 0 bits
3. for t = 2 to T:
     state = (state << 2) & 0xFFFF  # kV=2 bit 左移, 保留低 L=16 bits
     state |= B[16 + (t-2)*2 : 16 + (t-1)*2]  # 读入新 2 bits
     w_t = codebook[state]
4. 输出 Ŝ = [w_1, w_2, ..., w_T]
```
并行性：因 w_t 仅依赖 16-bit 窗口（允许 kV=2 bit 滑动），所有 w_t 可同时从编码中独立解码。对比 naive TCQ 需从第 1 bit 开始顺序遍历 trellis 图。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bitshift trellis 需要与伪随机 codebook 配合以避免相邻权重组的强相关性。QTIP 用 compute-based 近似高斯 codes 产生伪随机 codebook，效果接近随机高斯 codebook（RPTC 的原本方案需要存储或应用随机排列，开销过高）。在 GPU 上，bitshift 操作为单条指令（如 PTX shl），且各组并行解码在 CUDA thread 中实现。在 ARM CPU 上同样高效（所有 ARM 指令集均支持位移）。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing

## Compute-based Gaussian Codes for Trellis Quantization（基于计算的高斯码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Compute-based Gaussian Codes 是 QTIP 提出的新型 codebook 设计，以计算代替查找表来生成伪随机近似高斯分布的 codebook 值。目标：在 bitshift trellis 中，相邻权重组共享大量比特位，若 codebook 值顺序排列会产生强相关性（Figure 3 far left），须通过伪随机 codebook 去相关。传统 RPTC 存储/应用随机排列开销过大，QTIP 的 compute-based codes 用极少量 GPU ALU 指令即时生成伪随机高斯值。三种 codes：(1) **1MAD** (2 instr)——LCG (ax+b mod 2^32) → 求和 4 个 8-bit unsigned ints（近似高斯）→ scale/shift；(2) **3INST** (3 instr)——LCG → XOR bottom 16 bits 修改 magic FP16 数的尾数/指数/符号位 → XOR top 16 bits 同操作 → m1+m2 近似高斯（两镜像指数分布之和）；(3) **HYB** (摊销 2 instr/weight)——x²+x mod 2^32 hash → 取 bits 作为 2^Q×2 LUT 索引 → XOR bit 15 翻转第二分量符号。HYB codebook 仅 2KiB (Q=9)，可放入 GPU L1 cache。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
1MAD 码伪代码（Algorithm 1）：
```
输入: L-bit 0-left-padded integer x, uint32 a=34038481, b=76625530.
x ← (a·x + b) mod 2^32          // LCG, 1 MAD 指令
x ← (x & 255) + ((x>>8)&255) + ((x>>16)&255) + ((x>>24)&255)  // 求和 4×8-bit, 1 vabsdiff4
x ← (x - 510) / 147.8           // scale/shift, 复用 MAD 或单独 FMA
输出: 伪随机近似高斯 x ~ N(0,1).
```
3INST 码伪代码（Algorithm 2）：
```
输入: x, a=89226354, b=64248484, m=0.922 (FP16).
x ← (a·x + b) mod 2^32          // LCG, MAD
// 复制 m 到 32-bit 寄存器两半: m32 = (m<<16) | m
x ← x XOR (m32 & mask)          // lop3: XOR + mask, 修改 FP16 尾数/指数
m1 = low16(x) as FP16; m2 = high16(x) as FP16
输出: m1 + m2  // FADD, 近似高斯 (两镜像指数分布之和)
```
HYB 码伪代码（Algorithm 3）：
```
输入: x, codebook C ∈ R^{2^Q × 2} (Q=9 → 2KiB).
x ← (x·x + x) mod 2^32          // hash, MAD
idx ← (x >> (15-Q)) & (2^Q - 1) // LUT index
v ← C[idx]                       // L1 cache lookup, 2×FP16
v[1] ← v[1] XOR (x & (1<<15))   // sign flip via lop3
输出: v (2D 伪随机近似高斯向量).
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
这些 codes 在 GPU 上以 CUDA/PTX 实现，嵌入 GEMV kernel 的权重解码阶段。Codebook 值在寄存器中产生后直接用于矩阵乘累加。HYB codebook 可 fine-tune（类似 QuIP# 的微调流程）。ARM CPU 可用 NEON vqtbl4q_u8 实现 6-bit 1D HYB code（Q=6, V=1）。关键约束：LCG 参数 (a,b) 和 magic FP16 数 m 须精心选择以避免强相关（论文通过遍历搜索确定）。1MAD 和 3INST 不含可训练参数，完全 lookup-free。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing

## Viterbi Algorithm in Trellis Coded Quantization（格状编码量化中的维特比算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Viterbi 算法（Forney, 1973）是一种在 trellis 图上寻找最优路径的动态规划算法。在 TCQ 中，Viterbi 算法求解最小化问题：minimize Σ_{i=1}^{T/V} ||C_{x_i} - s_i||² over x_1,...,x_{T/V} 为图 G 上的 walk。核心是价值函数 V_t(x) = 以节点 x 结束的、前 t 步的最优累计失真。更新规则：V_t(y) = min_{(x,y)∈G} V_{t-1}(x) + ||C_y - s_t||²。通过回溯指针记录每个状态的最优前驱，最终从最小 V_T 的状态反向追踪得到最优路径。复杂度 O(2^L T)，与比特率 k 无关。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 QTIP 的 PyTorch/CUDA 实现中（量化 T=256 维序列，L=16, k=2, V=1）：
```
输入: s[0..T-1] ∈ R^T, codebook C ∈ R^{2^L}
初始化:
  V[0..2^L-1] ← ||C[0..2^L-1] - s[0]||²
  ptr = zeros(T, 2^L)  # 回溯指针
for t = 1 to T-1:
  V_new = ones(2^L) × INF
  for each transition (i→j) in G:  # bitshift: j = ((i<<kV) & (2^L-1)) + c
    err = ||C[j] - s[t]||²
    if V[i] + err < V_new[j]:
      V_new[j] = V[i] + err
      ptr[t, j] = i
  V ← V_new
# 回溯
best = argmin(V); path[T-1] = best
for t = T-2 down to 0:
  path[t] = ptr[t+1, path[t+1]]
输出: 编码 = [起始状态(L bits)] + [每步 c 值(k bits) × T]
```
在 GPU 上，每步的 min 操作可高度并行（每个节点独立），实际实现使用 CUDA reduce 优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Viterbi 算法在 TCQ 中的使用仅限离线量化阶段，推理时不需要（直接 bitshift 解码）。因复杂度 O(2^L T)，L>20 时不实用（L=16 时 2^16=65K 状态已可接受）。BCJR-QAT (2026) 提出用 BCJR forward-backward 算法替代 Viterbi 实现可微量化感知训练（QAT），将硬量化松弛为 soft 期望，进一步扩展了 trellis 量化的应用范围。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing

## Tail-biting Trellis（咬尾格状结构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tail-biting trellis 是 trellis 编码的一种变体，要求 trellis walk 的起始和结束状态共享 L-kV 个比特位，形成"环状"约束。在 QTIP 中，tail-biting 解决了一个工程问题：直接量化长度 T 的序列到 (L,k,V) trellis 产生 kT + (L-kV) 比特（起始状态需额外 L-kV bits）。当硬件字长为 w（如 32 bits）且 w|kT 时，每序列需读取 ⌈(L-kV)/w⌉w 个浪费比特。精确求解 tail-biting 问题需 O(2^{2L}) 动态规划，对 L≥12 不可行。QTIP 提出 Algorithm 4 近似求解：旋转序列 T/2 步 → Viterbi 量化 → 提取重叠 → 以该重叠作为 tail-biting 约束重新量化原始序列，仅需 2 次 Viterbi 调用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QTIP Algorithm 4（Tail-biting 近似）：
```
输入: S ∈ R^T, (L,k,V) trellis G.
S' ← Rotate S right by ⌊T/2⌋      # 旋转序列
Ŝ' ← Viterbi(S', G)               # 第一遍 Viterbi
O ← L-kV bit overlap of Ŝ'_{⌊T/2⌋} and Ŝ'_{⌊T/2⌋+1}  # 提取重叠
Ŝ ← Viterbi(S, G) with start/end overlap = O  # 以 O 约束的第二遍 Viterbi
输出: tail-biting Ŝ.
```
实验表明该近似对 i.i.d. 高斯源几乎无损：量化 4K 个 T=256 序列时，2-bit (k=2) 的 Algorithm 4 MSE 0.0733 等于最优解 0.0733（Table 2）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tail-biting 使总编码比特数 = kT（起始状态隐含在结束状态中），当 kT 整除 w 时无浪费比特。在 QTIP 中，T_x=T_y=16, k=2 → kT = 2×16×16 = 512 bits = 16 个 32-bit word，完美对齐。Tail-biting 的近似在 i.i.d. 数据上极准但可能对非 i.i.d. 数据降质；QTIP 依赖 RHT 确保权重的 i.i.d. 特性。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing

## Proxy Hessian / Per-Layer Proxy Loss（代理 Hessian / 逐层代理损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-layer proxy loss 由 Nagel et al. (2020) 提出，是 LLM 后训练量化中广泛使用的优化目标。定义为 ℓ(Ŵ) = E_x[||(Ŵ - W)x||²] = tr((Ŵ - W)H(Ŵ - W)^T)，其中 H = E_x[xx^T] 称为 proxy Hessian 矩阵。动机：直接最小化逐层输出 MSE 而非权重的 MSE，因为不同权重对输出的影响由输入激活的二阶统计量 (H) 加权。损失是逐层定义的（per-layer），对大模型可处理。H 的估计：从校准数据集采样输入激活 x，计算外积 xx^T 的经验均值。复杂因素：最小化该损失因量化的不可微性而困难，已有方法包括 Hessian-based adaptive rounding (GPTQ)、alternating optimization (AQLM)、coordinate descent (QuIP#)等。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 QTIP 中，proxy loss 的优化由 BlockLDLQ 框架完成（见 BlockLDLQ 条目）。QTIP 的贡献不在此损失函数本身，而在用 TCQ 作为 BlockLDLQ 中的量化器（替代 VQ）——即"用什么量化"而非"如何量化"。具体伪代码见 Algorithm 5（QTIP with BlockLDLQ），核心是：对 H 做 T_y-block LDL 分解 → 逐列处理 → 每列重组为 T_x×T_y 高维序列 → Viterbi TCQ 量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Hessian 生成：对于 Llama 1/2 使用 6144 sequences × 2048 tokens（RedPajama 数据集），Llama 3 使用 4096 sequences × 8192 tokens。H 的对角近似（基于激活方差的简单缩放）在一些方法（如 AWQ）中也有效，但 QuIP# 和 QTIP 需要完整 H 以执行 LDL 分解和自适应 rounding。每层的 H 独立估计，无需跨层传播。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing

## Walsh-Hadamard Transform (WHT / 沃尔什-哈达玛变换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Walsh-Hadamard Transform (WHT) 是一种正交变换，其变换矩阵仅由 +1 和 -1 元素组成。对于维度 N=2^n，WHT 矩阵 H_N 通过 Kronecker 积递归构造：H_2 = 1/√2 [[1, 1], [1, -1]]，H_N = H_2 ⊗ H_{2^{n-1}}。行的集合构成正交的 Walsh-Hadamard 基（满足 H_N^T H_N = H_N H_N^T = I_N）。与 DFT/DCT/DHT 等正弦基不同，WHT 的基函数为方形波（square-wave patterns with sharp transitions），天然适合表示突变/尖峰信号（如模型权重的异常值）。WHT 满足快速计算特性：H_{2^n} 可在 O(n log n) 时间内计算，且由于元素仅为 ±1，计算仅涉及加法和减法，无需矩阵乘法。对于非 2 的幂维度，存在已知 Hadamard 矩阵的特定维度值（如 12、20、28 等），一般情况 N=2^n·m 下可用 H_N = H_{2^n} ⊗ H_m（H_m 为已知 Hadamard 矩阵）。在 QWHA 论文中，WHT 用作 PEFT 适配器的变换核，将稀疏系数矩阵 F 展开为全秩权重更新 ΔW = F H^{-1}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 QWHA 中 WHT 的核心用法——适配器权重更新计算流程：

```
# 给定: 稀疏系数矩阵 F ∈ R^{d_out × d_in}（仅 p 个非零元）
#       WHT 矩阵 H ∈ R^{d_in × d_in}（预计算并缓存，跨层共享）
#       H^{-1} = H^T（WHT 的正交性）

# 前向传播时权重更新 ΔW 的计算：
# 方法1（训练时）: 先展开 F 通过 H^{-1}
ΔW = F @ H^{-1}        # 矩阵乘法，O(d_out × d_in × d_in)
Y = X @ (W_Q + α·ΔW)^T

# 方法2（推理时，利用 WHT 的快速性）: 先对激活 X 做 WHT
X_transformed = X @ H^{-1}   # 通过 fast Hadamard kernel，O(d_in log d_in) 仅用加减法
ΔW_X = F_sparse_matmul(X_transformed)  # 稀疏矩阵乘法，O(p)
Y = W_Q @ X + α·ΔW_X

# WHT 矩阵预计算（一次性，跨层复用）：
H_2 = 1/√2 * [[1, 1], [1, -1]]
for n in 2..log2(N):
    H_{2^n} = H_2 ⊗ H_{2^{n-1}}  # Kronecker product
H_N = normalize(H_{2^n})          # 确保正交：H_N^T H_N = I
```

WHT 在 QA-PEFT 中相比 DCT/DHT 的优势体现在"能量集中"特性：WHT 系数的 Pareto hill index η 最小（分布最陡），意味着量化误差 ΔW_Q 的能量最大比例集中在最少 WHT 系数中。这使得稀疏适配器 F 能用少量参数高效重建量化误差，特别是大振幅异常值。对比：WHA (WHT) 捕获 18.12% 异常值系数，DHA (DHT) 17.06%，DCA (DCT) 仅 7.23%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WHT 在深度学习中的应用包括：(1) QuIP#/QuaRot 等量化方法中用作 incoherence processing（随机 Hadamard 旋转使权重矩阵更接近 i.i.d. 高斯分布，降低量化难度）；(2) QWHA 中作为 PEFT 适配器变换核；(3) 快速 Hadamard 变换可通过 fused CUDA kernel 实现（Dao-AILab 的 fast-hadamard-transform），利用 WHT 的 ±1 结构仅用加减法避免矩阵乘法，显著减少计算开销。H 矩阵可预计算一次并缓存在 GPU 内存中，跨所有同维度线性层复用。对于 WHT 系数和能量的理论等价性：||W||_F^2 = Σσ_i^2 = ||F||_F^2 = ||H'WH||_F^2（正交变换保持 Frobenius 范数）。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，Walsh-Hadamard 旋转矩阵 H 被用作 QA-SFT 中的可选旋转配置 R_i ∈ {H, I}。RoSTE 的关键创新在于自适应旋转策略：对每一层，通过比较 Walsh-Hadamard 旋转与无旋转的量化误差（公式 12）来逐层决策是否使用 H。随机旋转矩阵构造为 R(ζ) = H · Diag(r(ζ))，其中 H 为 Walsh-Hadamard 矩阵，r(ζ) ∈ {-1,1}^d 为随机符号向量。Proposition 4.4 证明使用 H 旋转后，权重量化误差从 O(d·max_i w_i²) 降至 O(log(4d/δ)/2 · ‖w‖²)，有效抑制了 outlier 值对量化精度的危害。旋转分为两类：(1) offline mergeable rotations (R1, R2) 可融进权重矩阵在训练前完成；(2) online rotations (R3, R4) 通过 fast Hadamard CUDA kernel 实现在线变换，用于消除 KV cache 的 activation outlier。

## Quantization-Aware PEFT (QA-PEFT / 量化感知参数高效微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware PEFT (QA-PEFT) 是将模型量化与参数高效微调（PEFT）结合的范式：先对预训练 LLM 权重做低比特量化（如 4-bit/3-bit/2-bit），然后在量化模型上通过 PEFT 适配器（如 LoRA、稀疏适配器等）进行下游任务微调。与传统 PEFT 不同的是，QA-PEFT 在初始化阶段需要显式补偿量化误差：将适配器初始化为近似量化误差 ΔW_Q = W_0 - W_Q 的某种形式（低秩近似或稀疏近似），使初始输出接近全精度模型，再通过微调进一步恢复精度。核心目标函数为最小化层输出误差：min ||ΔW_Q X - ΔW_adapter X||_F^2，其中 X 为校准集激活。与标准的 PEFT（适配器从随机/零初始化）不同，QA-PEFT 的"量化感知初始化"在 sub-4-bit 场景下尤为关键，因为仅靠微调无法完全恢复极端量化带来的精度损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QWHA 中的完整 QA-PEFT pipeline：

```
# Phase 1: Quantization
W_Q = GPTQ_MagR(W_0)  # 4/3/2-bit 量化，group_size=64
ΔW_Q = W_0 - W_Q       # 量化误差

# Phase 2: Calibration (收集激活统计)
calib_set = WikiText2.sample(128 sequences × 2048 tokens)
for X in calib_set:
    accumulate X@X^T  # 外积累积
XX^T = avg(X@X^T)
U, Σ, V^T = SVD(XX^T)
R = U @ sqrt(Σ)        # Hessian 平方根
B = H^{-1} @ R         # WHT 预投影

# Phase 3: Adapter Initialization (minimize ||ΔW_Q·R - F·B||_F^2)
p_i = AdaAlloc(ΔW_Q, X, p)  # 通道级参数分配
for each channel i:
    v = (ΔW_Q)_{i,:} @ R
    E_i = TopK(|v @ B^{-1}|, p_i)   # 选最大系数位置
    c_i = v @ B'^T @ inv(B' @ B'^T) # Refinement
F = Scatter(c, E)

# Phase 4: Fine-tuning (仅更新稀疏矩阵 F 中的非零值 c)
for epoch in 1..3:
    for X, y in Alpaca:
        ΔW = F @ H^{-1}    # 前向：WHT 展开
        Y = (W_Q + α·ΔW) @ X
        loss = CE(Y, y)
        c -= lr * ∂loss/∂c  # 仅更新 c，E 和 H 固定

# Phase 5: Inference
Y = W_Q @ X + α·(F @ (H^{-1} @ X))
```

QA-PEFT 与 PTQ (Post-Training Quantization) 的关键区别：PTQ 量化后不做微调，直接评估；QA-PEFT 利用少量训练数据通过适配器补偿量化误差并适应下游任务。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QA-PEFT 的演进路线：(1) QLoRA (Dettmers et al., 2023) — 首次将 4-bit NF4 量化与 LoRA 结合，但 LoRA 从零初始化；(2) LoftQ (Li et al., 2024) — 迭代 SVD 分解量化误差初始化 LoRA；(3) LQ-LoRA (Guo et al., 2024) — 低秩加量化矩阵分解；(4) CLoQ (Deng et al., 2025) — 校准的 LoRA 初始化，最小化层输出误差；(5) QWHA — 首次将 FT-based adapter (WHA) 引入 QA-PEFT，用 WHT 替代低秩结构，实现 full-rank 适配器 + 量化感知初始化（AdaAlloc + Refinement）。所有方法均使用 GPTQ 作为底层量化方案，适配器应用于所有线性层（Q/K/V/O/Gate/Up/Down projections）。校准集通常使用 WikiText-2（128-256 条序列），因其与微调数据的独立性。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，QA-PEFT 并非其直接方法（RoSTE 不使用 PEFT adapters 如 LoRA），但 QA-PEFT 是其重要的 baseline 对比对象。RoSTE 对比了 QLoRA（4-bit NF4 + LoRA）作为 QA-PEFT baseline：在 Pythia 6.9B W4A4KV4 下 QLoRA (r=64) ROUGE Avg=20.20 vs RoSTE 23.66（Table 7），在 Llama 3.1 8B 实验中 QLoRA 未直接对比但 STE 方法可视为 QLoRA 的无 adapter 变体。RoSTE vs QA-PEFT 的核心区别：(1) QA-PEFT 在量化后通过适配器补偿量化误差，权重保持冻结；(2) RoSTE 直接在量化约束下训练所有权重（full QAT），不使用额外适配器参数量，而是通过自适应旋转矩阵优化量化配置。

## QA-SFT (Quantization-Aware Supervised Fine-Tuning / 量化感知监督微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware Supervised Fine-Tuning (QA-SFT) 是 RoSTE 论文（ICML 2025）提出的新范式，将量化感知训练（QAT）与监督微调（SFT）合并为单一训练阶段，直接输出量化后的微调模型。传统两阶段 pipeline（先 SFT 后 PTQ）先训练全精度模型再量化，导致量化误差无法在训练中被补偿，性能次优。QA-SFT 的核心公式为：`min_{W,R} L_SFT(m_Q(·; W, R)) s.t. R R^T = I`，同时优化量化权重矩阵 W 和旋转矩阵 R。QA-SFT 区别于 QA-PEFT（如 QLoRA）的关键点在于：(1) 不引入额外适配器参数（如 LoRA），直接优化原始权重；(2) 使用 4-bit 权重量化、激活量化和 KV cache 量化（W4A4KV4），而非仅量化权重；(3) 结合 incoherence processing（旋转矩阵消除 outlier），使低比特激活量化成为可能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RoSTE 的完整 QA-SFT pipeline 伪代码：

```
# 输入: 预训练模型权重 {W_i^pt}_{i=0}^{ℓ-1}, SFT 数据集 D_sft, 校准样本 D_cal (n=128)
# 输出: 量化微调模型 m_Q(·; W^{KT}, R^{K-1})

# Phase 1: 修改 normalization layers
for each norm_layer in model:
    if isinstance(norm_layer, LayerNorm):
        absorb mean subtraction into prev weight matrix
        absorb scale/bias into next weight matrix
    if isinstance(norm_layer, RMSNorm):
        absorb RMSNorm scale into next weight matrix

# Phase 2: 初始化
W^0 = {W_i^pt}_{i=0}^{ℓ-1}  # 从预训练权重初始化

# Phase 3: RoSTE 训练循环 (外层 K=1, 内层 T steps)
for k = 0, ..., K-1:
    # -- Lower Level: Rotation Configuration --
    # 全配置量化误差
    E_no_rotation = compute_E(W^{kT}, {I}_{i=0}^{ℓ-1}, D_cal)    # 公式 (12)
    E_all_rotation = compute_E(W^{kT}, {H}_{i=0}^{ℓ-1}, D_cal)

    # 逐层自适应选择
    for i = 0, ..., ℓ-1:
        err_no_rot = compute_layer_error(W_i, I, D_cal)
        err_rot = compute_layer_error(W_i, H, D_cal)
        R_i^k = I if err_no_rot < err_rot else H   # 选择误差更低的配置

    # -- Upper Level: QAT via Rotation-aware STE --
    for t = 0, ..., T-1:
        mini_batch = sample(D_sft)

        # Forward (每层):
        for each layer i:
            X_rot = Q_x(X · R_i)           # 激活量化（含在线旋转）
            W_rot = Q_w(R_i^T · W_i)       # 权重量化（含旋转）
            output = X_rot · W_rot          # INT4 矩阵乘法

        loss = SFT_loss(model_output, labels)  # CE loss on tokens

        # Backward (STE with rotation):
        # ∂L/∂W_i ≈ R_i · (grad from upper layer)  # STE 旋转修正
        # ∂L/∂X ≈ grad_output @ (W_rot)^T · R_i^T
        W_i -= lr * ∂L/∂W_i

# Phase 4: 推理时合并离线旋转
merge R_1, R_1^T, R_2, R_2^T, R_4^T into adjacent weight matrices
keep R_3, R_3^T, R_4 as online fast Hadamard CUDA kernel
```

QA-SFT 中的量化误差函数 E (公式 12) 用于 rotation selection：
```
E({W_i}, {R_i}) = Σ_i ||Q_w(R_i^T W_i) - R_i^T W_i||^2   # 权重量化误差
                + (1/n) Σ_i Σ_j ||Q_x(X_{i,j} R_i) - X_{i,j} R_i||^2  # 激活量化误差
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QA-SFT 的实现要点：(1) 量化方案：非对称均匀量化（asymmetric uniform quantizer），per-token activation quantization（沿 token 维度分组），per-channel weight quantization（沿输出通道维度分组），clipping factor c ∈ {1, 0.95, 0.9}；(2) 训练配置：AdamW optimizer，learning rate sweep（不同模型大小使用不同 LR 范围），cosine/linear LR schedule，gradient accumulation for large models（如 Llama 3.1 8B 用 gradient_accumulation=16）；(3) 旋转矩阵实现：使用 fast Hadamard CUDA kernel（继承自 QuaRot/QuIP# 开源实现），离线可合并旋转预先吸收到权重矩阵中减少推理开销；(4) 代码开源：https://github.com/OptimAI-Lab/RoSTE。训练成本：Qwen2.5 7B W4A4KV4 的 RoSTE 训练时间 2.8h（8×A100），比 SFT→QuaRot 两阶段 2.1→0h 略多但精度显著提升（ROUGE Avg 25.10 vs QuaRot 4.79）。

涉及论文标题：
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

## Fourier Transform-based Adapters (FT-based Adapters / 傅里叶变换基适配器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fourier Transform-based Adapters（FT-based Adapters）是一类 PEFT 适配器，将权重更新 ΔW 表示为稀疏系数矩阵 F 在频域（变换域）中的表示：F = H' ΔW H ⟹ ΔW = H'^{-1} F H^{-1}，其中 H 和 H' 是预定义的固定正交变换核。F 为稀疏矩阵（仅 p 个可训练非零元素），稀疏位置由参数选择策略决定。与 LoRA（ΔW = BA，rank ≤ r）相比，FT-based adapters 的关键优势是"全秩表示能力"：只要 F 的每行每列平均有 ≥2 个非零元，F 就以高概率满秩（rank = min(d_in, d_out)），远超 LoRA 的秩瓶颈。已知的 FT-based adapter 变体包括：(1) FourierFT (DFT kernel)，(2) LoCA/DCA (DCT kernel)，(3) SSH/DHA (DHT kernel)，(4) QWHA/WHA (WHT kernel)。QWHA 论文中的 WHA（WHT-based Adapter）采用单变换设计 ΔW = F H^{-1}（而非双变换 H'^{-1} F H^{-1}），因为在量化场景中输出通道间独立，双变换不提升表示力，反而增加计算开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FT-based Adapter 的核心机制——从稀疏系数到权重更新的展开：

```
# 定义
F = Scatter(c, E)  ∈ R^{d_out × d_in}
# c ∈ R^p: 可训练系数值向量
# E ∈ R^{p × 2}: 非零元素索引列表（固定或可训练）
# Scatter: F[E[l,0], E[l,1]] = c[l], 其余为0

# WHA (QWHA - 单变换):
ΔW = F @ H^{-1}
# H: WHT 矩阵, d_in × d_in, 仅 ±1/√(d_in)
# 计算复杂度: O(p·d_in) sparse-dense + O(d_in log d_in) fast WHT

# DCA/DHA (LoCA/SSH - 双变换):
ΔW = H'^{-1} @ F @ H^{-1}
# H' 和 H: DCT/DHT kernel (包含正弦/余弦计算)
# 计算复杂度: O(p·(d_in+d_out)) + O(d_in log d_in) + O(d_out log d_out)
# 双变换开销显著，训练时间约 WHA 的 3-10x（batch=4: 6.0h vs 26.1h/30.1h）

# 秩分析 (对于随机选择的 F):
# F 的每行非零元平均数: k = p/d_in > r
# F 的每列非零元平均数: l = p/d_out > r
# 当 k,l ≥ 2 时, rank(F) → min(d_in, d_out) 以高概率
# LoRA 对比: rank(BA) ≤ r << min(d_in, d_out)
```

参数选择策略对比（FT-based adapters 必须选定 E——哪些系数位置参与微调）：
- Random：纯随机选择位置，初始化 c=0（FourierFT 原始方案）
- SSH：50% 幅值最大位置 + 50% 随机（假设预训练和微调权重的频谱模式相似）
- LoCA：随机初始化位置 + 微调过程中通过重参数化更新 E（训练开销约 ×2-3）
- AdaAlloc (QWHA)：通道级自适应分配 + 通道内幅值选择 + Refinement（量化感知）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FT-based adapters 的实现要点：(1) 变换矩阵 H 预计算并跨层缓存，不同维度用不同大小的 H；(2) 前向传播中 F @ H^{-1} 的计算可通过 (i) 先计算 F @ H^{-1} 展开为稠密 ΔW，或 (ii) 推理时先对激活做 H^{-1} X 再与稀疏 F 相乘（更快）；(3) 反向传播仅更新 c（F 的非零值），E 通常固定（LoCA 例外，E 通过重参数化可训练）；(4) WHT 相比 DCT/DHT 的计算优势：仅 ±1 元素 → 仅用加减法 → 训练时间与 LoRA 相当，而 DCT/DHT 需复数/三角函数计算。适用场景：需要高表示力（超越 LoRA rank 限制）且可接受少量额外推理开销的场景。在 QA-PEFT 中，FT-based adapter 必须配合量化感知初始化（非随机/零初始化）才能发挥优势，否则效果不如 LoRA-based QA-PEFT。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

## AdaAlloc (Adaptive Allocation / 自适应参数分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaAlloc 是 QWHA 提出的量化感知适配器参数选择策略，由两个层级组成：(1) 通道级自适应预算分配（Channel-wise Budget Allocation）：根据各输出通道的量化误差大小按比例分配可训练参数数量； (2) 通道内幅值选择（Intra-channel Magnitude-based Selection）：在每个通道内，选择 WHT 变换域中系数幅值最大的位置。核心公式为 p_i = floor(p × ||(ΔW_Q X)_{i,:}||_F^t / Σ_j ||(ΔW_Q X)_{j,:}||_F^t)，其中 t 为温度超参数（默认 t=1）。余数分配给当前分配最少的通道，保证所有通道 ≥ 2 个参数以维持 F 的 full-rank 性质。AdaAlloc 是首次同时兼顾"full-rank 表示（fine-tuning 能力）"和"低初始化误差（量化误差补偿）"的参数选择策略：纯幅值选择导致 F 低秩（参数过度集中在少量异常值通道），随机选择虽保持高秩但初始化误差大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AdaAlloc 完整算法流程：

```
Input:  ΔW_Q ∈ R^{d_out × d_in}  # 权重量化误差
        X ∈ R^{d_in × (b·s)}     # 校准集激活
        p                          # 总参数 budget
        t = 1.0                    # 温度参数

# Step 1: 计算各通道的输出误差
for i in 0..d_out-1:
    error[i] = ||(ΔW_Q @ X)[i, :]||_F^t  # 第 i 个输出通道的误差 t 次幂

# Step 2: 按比例分配预算
total_error = sum(error)
p_i = floor(p × error[i] / total_error)  for each i

# Step 3: 余数分配（保证每通道 ≥2, sum(p_i) = p）
remainder = p - sum(p_i)
sort channels by p_i ascending
distribute remainder to channels with smallest p_i (+1 each)
ensure all p_i >= 2  # 满足 full-rank 条件

# Step 4: 通道内幅值选择
B = H^{-1} @ R   # 预计算投影基
for i in 0..d_out-1:
    v = (ΔW_Q)[i, :] @ R
    dense_sol = v @ B^{-1} = (ΔW_Q @ H)[i, :]  # 稠密 WHT 系数
    E_i = TopK_Index(|dense_sol|, p_i)   # 选幅值最大的 p_i 个位置
    B' = B[E_i, :]                         # 选中位置的基向量
    c_i = v @ B'^T @ inv(B' @ B'^T)       # Refinement: 最小二乘精化

Output: E (参数位置), c (参数值)

# AdaAlloc vs 其他策略的对比:
# Magnitude-based: p_i = |ΔW_Q @ H 的前 p 个最大系数| → low-rank F
# Random (LoCA): E 随机初始化 → 高秩但高初始化误差
# SSH: 50% 幅值 + 50% 随机 → 中间方案
# AdaAlloc: 通道级分配 + 通道内幅值 → 唯一同时 high-rank + low-init-error
```

Layer output error 对比 (Table 2): AdaAlloc avg 3.86 vs Magnitude 3.82 vs SSH 4.57 vs Random 5.96, None (no init) 7.21。AdaAlloc 与 Magnitude 误差接近但 rank 远高于 Magnitude（Figure 4：AdaAlloc rank≈r_max，Magnitude rank≈0）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaAlloc 的温度参数 t 控制分配锐度：t < 0.5 时接近均匀分配（无法集中参数到高误差通道），t > 1.5 时过度集中（忽略低误差通道中的重要系数）。论文推荐范围 t ∈ [0.5, 1.0]，默认 t=1 实验表现最优（GSM8k: t=1 得 41.47% vs t=0.25 得 40.11% vs t=2.0 得 40.04%）。AdaAlloc 保证 full-rank 的理论依据：满足 Coja-Oghlan et al. (2020) 的稀疏随机矩阵满秩条件（每行每列 ≥2 非零元）。在 P(r≥4) 的参数 budget 下，即使是输出维度最大的线性层也能保证每行 ≥2 参数。AdaAlloc 与 Refinement 步骤配合使用效果最佳：仅 AdaAlloc 不做 Refinement 时 avg error=7.06，加上 Refinement 后降至 3.86（Table 7）。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

## MagR (Weight Magnitude Reduction / 权重大幅值衰减)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MagR (Weight Magnitude Reduction) 是 Zhang et al. (2024) 提出的一种 LLM 后训练量化（PTQ）预处理技术。核心思想：在量化前对模型权重做幅值衰减变换，降低异常值（outlier）的幅值，使权重的数值范围更集中，从而减少量化时的 clamping 误差和舍入误差。MagR 通过引入一个可学习的 per-channel scaling vector 来缩放权重矩阵：W' = diag(s) · W，其中 s 的元素 < 1 用于压缩异常值的幅值。缩放变换与 Hessian-guided 优化结合，目标是在最小化输出误差的前提下找到最优缩放向量。缩放后的权重 W' 再送入标准量化器（如 GPTQ）进行量化。推理时，缩放因子可以通过与前一层或后一层的权重提前融合（merge），从而不增加推理开销。在 QWHA 论文中，GPTQ_MagR 作为基础量化方案（即 GPTQ + MagR 预处理），用于所有 baseline 和 QWHA 的模型量化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MagR 在 GPTQ_MagR 量化流程中的使用：

```
# MagR 预处理 + GPTQ 量化流程
# Input: 预训练权重矩阵 W ∈ R^{d_out × d_in}
#        activation X, 校准集

# Step 1: MagR - 学习 per-channel 或 per-group scaling factors
# 目标: min_s ||WX - (s^{-1}·round(s·W))X||^2
for each channel/group:
    s_init = max(|W_channel|) / max_safe_value  # 初始估计
    optimize s to minimize output error:         # 梯度下降/网格搜索
        W_scaled = s * W
        W_q = round(W_scaled)                    # 量化
        W_dq = W_q / s                            # 反量化
        error = ||WX - W_dq @ X||^2
    # s < 1: 压缩异常值

# Step 2: GPTQ 逐列量化 (使用 MagR 缩放后的权重)
# Hessian H = 2XX^T (从校准集累积)
W_q = copy(W_scaled)
for col in 0..d_in-1:
    # 量化第 col 列
    W_q[:, col] = quantize(W_scaled[:, col], scale[col])
    # 补偿剩余列的误差
    error = (W_scaled[:, col] - W_q[:, col]) / H[col, col]
    W_scaled[:, col+1:] -= error * H[col, col+1:]

# Step 3: 推理时融合 MagR scaling
# Option A: 融合到上一层输出投影
# Option B: 融合到本层权重 W_q' = W_q / s
# Both options: 零推理开销

# QWHA 中 GPTQ_MagR 的使用:
W_Q = GPTQ_MagR(W_0, calibration_data=X_calib)
ΔW_Q = W_0 - W_Q  # 用于 QWHA 初始化
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MagR 的关键设计选择：(1) Scaling 粒度：per-channel 或 per-group（与量化 group size 对齐）；(2) 优化方法：网格搜索（brute-force，用于小规模）或梯度下降（用于大规模）；(3) 与 GPTQ 的集成方式：先确定 scaling factors，再对缩放后的权重执行 GPTQ 逐列量化。MagR 的有效性源于：LLM 权重的异常值（outlier）是量化的主要精度瓶颈——这些大值被 clamp 后产生巨大误差。通过 MagR 降低异常值幅值后再量化，clamping 边界能覆盖更大比例的权重。与其他量化预处理技术（如 QuaRot 的随机 Hadamard 旋转、SmoothQuant 的 per-channel scaling）的对比：MagR 专注于"幅值衰减"，而 QuaRot 专注于"incoherence processing"（使权重分布更均匀），两者可互补。在 QWHA 实验中，GPTQ_MagR 在 4-bit 下的 CSQA 准确率（LLaMA-3.1-8B）为 69.11%，比原始 GPTQ 更高。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning

---

## Adaptive Quantization Noise (AQN)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AQN 是 QeRL (Huang et al., NVIDIA, 2025) 提出的动态噪声注入机制。核心设计：(1) 对每个量化线性层采样随机噪声 Z_noisy ~ N(0,σ²I)，Z_noisy ∈ R^{1×d}；(2) 噪声通过 RMSNorm scale 参数注入——additive noise 等价转换为 multiplicative noise：(Z_noisy/w+I)⊙\hat{W}；(3) 噪声强度 σ 按指数衰减：σ(k)=σ_start×(σ_end/σ_start)^((k-1)/(K-1))，K=10 阶段，σ_start=1e-2, σ_end=5e-4。Stage 0 仅有量化噪声（σ=0），后续逐步降低注入噪声。关键洞察：量化噪声的静态/确定性特性对 RL 后期不利，AQN 通过动态控制实现探索到利用的过渡——初始利用量化噪声高熵探索，后期降低噪声稳定收敛。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# AQN 嵌入 GRPO/DAPO 的 RL pipeline
K = 10; σ_start = 1e-2; σ_end = 5e-4
steps_per_stage = total_steps // K

for step in range(total_steps):
    k = step // steps_per_stage
    σ = 0 if k==0 else σ_start*(σ_end/σ_start)^((k-1)/(K-1))
    for each RMSNorm before QKV/gate-up:
        Z_noisy ~ N(0, σ²)         # [1, d]
        w_noise = w_orig + Z_noisy  # 融入 RMSNorm scale
    # forward: x_noisy·\hat{W}^T + LoRA
    # backward: gradient only to LoRA A,B
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/NVlabs/QeRL。噪声共享：Q/K/V 共享同 RMSNorm noise，gate/up 共享另一，因其 LLM 架构中分别前置同一 RMSNorm。乘法噪声等效：(Z_noisy/w+I) 作用为 row-wise multiplicative noise on \hat{W}，证明在 QeRL Appendix G。初始化 σ=1e-2 而非传统 noisy network 的 1e-1，因 LLM 对乘法噪声更敏感。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

---

## GRPO (Group Relative Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GRPO 是 DeepSeek (Shao et al., 2024) 提出的 RL 算法，用于 LLM 数学推理训练。核心创新：放弃 PPO 的独立 critic/value network，改用 group-wise relative advantage。对每个 query q 采样 G 个候选输出，用 rule-based reward 评分后计算标准化 advantage：A_i = (r_i − mean(r))/std(r)，所有 token 共享同一 A_i。优化目标含 clipped importance ratio + KL penalty：J(θ) = E[min(ratio·A, clip(ratio,1−α,1+α)·A) − β·D_KL(π_θ||π_ref)]。GRPO 消除 critic 模型训练成本（省 ~50% 显存），适合有 rule-based verifier 的任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GRPO (QeRL 配置: G=8, α=0.2/0.28, μ=4 updates per rollout)
for each D_b from D:
    for each q in D_b:
        {o_i} ~ π_θold(·|q) for i=1..G     # rollout, temp=1.0
        r_i = rule_reward(q, o_i)           # 数学题答案正确性
        A_i = (r_i - mean(r))/std(r)        # group advantage
    for iter in 1..μ:
        for each (q, o_i, A_i):
            ratio = π_θ(o_i|q)/π_θold(o_i|q)
            L = -min(ratio*A_i, clip(ratio,0.8,1.28)*A_i) + β*KL(π_θ||π_ref)
        L.backward(); optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：TRL (HuggingFace), veRL, DeepSeekMath。QeRL 配置：G=8(GSM8K)/16(BigMath)，off-policy(GSM8K, μ=4)/on-policy(BigMath, μ=1)，无额外 entropy/KL loss，rollout temperature=1.0。对比 PPO：无需 value network 但要求 reward 组内方差足够。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

---

## DAPO (Dynamic Sampling Policy Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DAPO 是 ByteDance (Yu et al., 2025) 提出的开源 LLM RL 框架。相比 GRPO 的关键改进：(1) 更高 clipping upper-bound (ε_high=0.28) 避免 entropy collapse；(2) token-level policy gradient（每 token 独立计算梯度，非序列平均）；(3) 移除 KL penalty 消除 exploration 上限；(4) decoupled optimization 简化训练 pipeline。QeRL 在 BigMath 上用 DAPO 训练 7B/14B/32B。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# DAPO (vs GRPO 差异)
# 1. Token-level: loss 对每 token 而非序列平均
for each (q, o_i, A_i):
    for each token t in o_i:
        ratio_t = π_θ(o_{i,t}|q)/π_θold(o_{i,t}|q)
        L_t = -min(ratio_t*A_i, clip(ratio_t,0.8,1.28)*A_i)
    L = mean(L_t)                            # 无 KL 项

# 2. 更高 clip upper (ε_high=0.28 vs GRPO α=0.2)
# 3. on-policy: μ=1 (每次 rollout 后仅 1 次更新)
```
QeRL 配置：G=16, on-policy, max response=8192, clip(0.2,0.28)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/volcengine/verl。DAPO 的 token-level loss 对长序列训练更鲁棒，无 KL 约束使 exploration 更激进——适合 reward 信号明确的数学推理，但对 reward hacking 更敏感。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

---

## Noise Sharing / Noise Merging (via LayerNorm)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise Sharing (Noise Merging) 是 QeRL 提出的零开销噪声注入技术。将 AQN 的 additive Gaussian noise Z_noisy 等价融入 RMSNorm scale w：w_noise=Z_noise+w。数学等价：X·(Z_noisy+Ŵ) = X·Z_noisy+X·Ŵ，通过 RMSNorm 变换得 X_norm·((Z_noisy/w+I)^T⊙Ŵ)，即 additive noise → row-wise multiplicative noise on weight。优势：(1) 零参数开销；(2) 零额外 flops；(3) 不破坏 Marlin NVFP4×BF16 kernel 兼容性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Noise Sharing 操作
w_noise = w + Z_noisy                  # [d], zero overhead
x_norm = x / sqrt(mean(x²)+ε)
x_scaled = w_noise ⊙ x_norm           # 等效乘法噪声
output = x_scaled·\hat{W}^T + LoRA    # Marlin kernel accelerated
# 噪声共享：Q/K/V → 同 RMSNorm；gate/up → 同 RMSNorm
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/NVlabs/QeRL。仅适用于 Pre-LN Transformer (RMSNorm→Linear)。不创建独立噪声向量，修改已有 RMSNorm weight——在训练每步采 Z_noisy 并更新 w_noise，前向完成后恢复 w_orig。核兼容：噪声不写入量化权重，保持 packed 4-bit layout 不变。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

## Data-Free Calibration for Diffusion Model Quantization (数据无关校准)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
数据无关校准（Data-Free Calibration）是一种无需真实数据即可构建量化校准集的技术。在扩散模型量化场景下，其核心思想是：利用扩散模型本身的性质——推理以随机高斯噪声 x_T ~ N(0, I) 作为输入——将高斯噪声通过全精度模型前向传播，在不同时间步 t 采样中间激活作为校准数据。QuEST 的做法：仅需推理全精度模型数次（总计 128-256 样本/时间步），即可获得足够校准样本用于微调量化模型。这与需要真实图像数据的 PTQ 方法（如 Q-Diffusion 使用 5120 张真实图像）形成对比，完全消除了对外部数据集的依赖。数据无关校准之所以可行，是因为：扩散模型全精度前向传播本身就能产出有意义的特征分布，且权重微调的目标是对齐全精度模型输出，而非拟合真实数据分布，因此纯噪声驱动的合成数据即可满足需求。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
数据无关校准在量化 pipeline 中的流程：
```
输入：全精度模型 FP_model, 总时间步 T, 每时间步样本数 N=256
输出：校准集 activations = {a_t^i : t ∈ [0,T], i ∈ [1,N]}

for t in sample_time_steps(T):      # 采样部分时间步
    for i in range(N):               # 每步 128-256 个样本
        x_T = randn(latent_shape)    # 采样标准高斯噪声
        a_t^i = FP_model.forward(x_T, timestep=t)  # 全精度前向
        store(a_t^i)                  # 存储中间激活
```
校准集随后用于 TLA 和 CMA 的 MSE 损失计算：L_TLA = ΣE_t[||FP_TE(t) - Q_TE(t)||²]，其中 FP_TE(t) 来自校准集存储的全精度激活，Q_TE(t) 来自量化模型在同一噪声输入下的前向。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 QuEST 中，数据无关校准直接嵌入微调 pipeline：首先用高斯噪声推理全精度模型获取各层激活（作为 ground truth），然后在每次微调迭代中，将相同的噪声输入量化模型，计算量化输出与预存全精度输出的 MSE。该方法也适用于 Stable Diffusion 等文本到图像模型——尽管文本条件不同，但推理仍从随机噪声开始，因此合成校准集同样有效（Stable Diffusion 使用 128 样本/时间步）。相比 EfficientDM 等方法依赖 ImageNet 等真实校准集，数据无关方式消除了数据采集和隐私顾虑。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Temporal Layer Alignment (TLA, 时序层对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Layer Alignment（TLA）是 QuEST 提出的两种局部对齐损失之一，专门用于微调扩散模型中的时间嵌入（Time Embedding）层以改善量化性能。其核心观察（Property ❶）是：扩散模型在不同时间步承担不同功能（早期去噪 vs 晚期细化），因此准确的时间信息传递对量化至关重要——仅量化时间嵌入层就可使 FID 从 6.77 升至 7.58（W8A8），W4A8 下从 7.55 升至 8.59（相对提升约 15%）。TLA 通过最小化时间嵌入层量化输出与全精度输出的 MSE 来恢复时序精度：L_TLA = Σ_{l∈C_TE} E_t[||O(t; w_l) - Õ(t; w_l, s_l)||²]。关键设计：(1) 同时微调时间嵌入层权重 w_l 和该层的激活量化参数 s_l；(2) 使用单一量化参数集适配所有时间步（无需按时间步分别存储），提高时间效率和存储效率；(3) 时间嵌入层的微调权重和量化参数在后续 CMA 阶段冻结。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TLA 的执行流程伪代码：
```
# 阶段一：TLA —— 仅微调时间嵌入层
w_TE = load_fp_weights()           # 全精度时间嵌入层权重
s_TE = init_minmax_scales()        # 初始化激活量化参数
optimizer = Adam([w_TE, s_TE], lr_w=1e-5, lr_s=1e-4)

for epoch in range(TLA_epochs):
    for t in sample_time_steps():
        x_T = randn(latent_shape)
        O_fp = FP_model.time_embed(t; w_TE_fp)    # 全精度输出（来自校准集）
        w_TE_q = quantize(w_TE, s_w)               # 权重量化（冻结s_w）
        O_q = Q_model.time_embed(t; w_TE_q, s_TE)  # 量化输出
        loss = MSE(O_fp, O_q)                       # L_TLA
        loss.backward()                             # 仅 w_TE, s_TE 有梯度
        optimizer.step()
# TLA 完成后冻结 w_TE, s_TE
```
注意：时间嵌入 t 先经一个或两个线性层转换为 time embedding vector，然后在 UNet 各层注入（通过投影层与 latent image representation 合并）。TLA 微调覆盖这两类线性层。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TLA 是一个两阶段微调策略的第一阶段。实现要点：(1) 权重量化参数在 TLA 期间保持冻结（仅微调全精度权重，量化参数固定）；(2) 激活量化参数与权重联合优化；(3) 微调后的时间嵌入层甚至能超越全精度 baseline——由于同时优化了时间嵌入层和其量化参数，经过微调的量化模型可能输出比原始全精度模型更精确的时间信息（表 2：QuEST TLA 的 FID 5.61 vs FP 6.77）；(4) TLA 可独立使用，也可与 CMA 和全局损失组合获得最佳效果。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Critical Module Alignment (CMA, 关键模块对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Critical Module Alignment（CMA）是 QuEST 提出的两种局部对齐损失之二，专门微调扩散模型中量化敏感的注意力相关层。其核心观察（Property ❷）是：不同类型层的激活对位宽降低的敏感度差异巨大——FeedForward 层激活在 6-bit 即导致生成失败，而所有其他线性层（虽多 5 倍）在 4-bit 才失败，卷积层（虽多 3 倍）也在 4-bit 才失败。CMA 通过最小化这些关键层的量化输出与全精度输出 MSE 来缓解敏感层的量化退化：L_CMA = Σ_{l∈C_A} E_t[||O(z_{t,l}; w_l) - Õ(z̃_{t,l}; w_l, ŝ)||²]，其中 C_A 为注意力相关层集合（包括 Q/K/V/O 投影和 FeedForward 层），z̃_{t,l} 为量化后的层输入，ŝ = s \ s_l (l∈C_TE) 表示除 TLA 已优化的量化参数外的所有参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CMA 的执行流程伪代码：
```
# 阶段二：CMA —— 微调注意力相关层（TLA 冻结后）
w_TE, s_TE = frozen()                    # TLA 结果已冻结
C_A = {Q_proj, K_proj, V_proj, O_proj, FeedForward_layers}
w_A = load_fp_weights()                  # 注意力相关层全精度权重
s = init_minmax_scales()                 # 所有激活量化参数（含 C_A 和其他层）
ŝ = s \ s_TE                             # 排除 TLA 已优化的参数
optimizer = Adam([w_A, ŝ], lr_w=1e-5, lr_s=1e-4)

for epoch in range(CMA_epochs):
    for t in sample_time_steps():
        x_T = randn(latent_shape)
        z_fp = get_calibration_activation(t, layer=l) # 预存的全精度各层激活
        z̃ = Q_model.forward(x_T, t)       # 量化模型前向（含 TLA 量化TE层）
        # 对每个 l ∈ C_A:
        O_fp_l = z_fp[l]                  # 全精度输出
        O_q_l = z̃[l]                      # 量化输出
        loss = sum(MSE(O_fp_l, O_q_l) for l in C_A)  # L_CMA
        loss.backward()                   # 仅 w_A, ŝ 有梯度
        optimizer.step()
# w_A 和 ŝ 得到更新，未选中的层仅 ŝ 得到优化（通过全局损失）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CMA 的关键实现细节：(1) 每个模块使用不同的输入 z̃_{t,l} 进行优化（即已量化的前一层输出），以增强模块对输入扰动的鲁棒性；(2) CMA 在 TLA 之后执行（progressive alignment），因为时序信息独立于图像输入且在模型早期确定，先对齐时间嵌入为后续模块提供准确的时间步指导；(3) CMA 可显著改善 FID——在 TLA 基础上添加 CMA 使 FID 从 4.41 降至 3.26（W4A8, LSUN-Bedrooms）；(4) 如果没有全局损失 L_G，CMA 单独使用的 FID 为 8.99，添加 L_G 后降至 6.41（FID 提升 2.58），说明 CMA 需要全局监督来指导非敏感层的量化参数优化。CMA 的层选择策略是基于经验性的逐层位宽降低实验，而非自动搜索。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Selective Weight Finetuning for Quantization (选择性权重量化微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
选择性权重量化微调是一种参数高效的低比特量化策略：仅微调模型中一小部分关键权重（而非全量参数），使量化模型输出对齐全精度模型。QuEST 的选择策略基于两类识别：(1) 时间嵌入层（需要准确传递时序信息）；(2) 注意力相关层（对位宽降低最敏感）。微调参数总量不足 7%（如 LDM-4 上 <7% 参数被更新），远超 EfficientDM（需训练 LoRA adapter）和 Full-finetune（100% 参数）的效率。选择性微调的理论依据来自 Theorem 3.2：在低比特下，激活扰动 Δ 太大导致泰勒展开不准，需要微调权重 w_n 使模型对扰动鲁棒——但只有与特定输入/功能强相关的层需要此处理，其他层的量化误差可通过全局损失间接优化其量化参数来缓解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
选择性微调与全量微调的对比伪代码：
```
# 全量微调（Full-finetune）
for epoch in range(epochs):
    for t in sample_time_steps():
        output_q = Q_model(x_T, t)        # 全模型前向
        loss = MSE(output_fp, output_q)   # 仅全局损失
        loss.backward()                   # 所有 w 和 s 有梯度
        optimizer_full.step()             # 更新 ~100% 参数

# QuEST 选择性微调
frozen_layers = set(all_layers) \ (C_TE ∪ C_A)  # >93% 的层
for epoch in range(TLA_epochs):           # 阶段一：仅 TE 层
    output_q_TE = Q_model.TE_forward(t)
    loss = MSE(FP_TE, output_q_TE)        # L_TLA
    loss.backward()                       # 仅 w_TE, s_TE
    optimizer.step()

for epoch in range(CMA_epochs):           # 阶段二：仅 Attn 层
    output_q_all = Q_model(x_T, t)        # TE 层已量化冻结
    loss = L_CMA + 2*L_G                 # CMA + 全局
    loss.backward()                       # 仅 w_A, ŝ
    optimizer.step()
```
参数计数对比（LDM-4）：总参数 ≈ 400M，C_TE ≈ 0.5% ≈ 2M，C_A ≈ 5% ≈ 20M，微调参数 < 28M = < 7%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuEST 的选择性微调实现：(1) 权重量化参数固定（s_w 不参与训练），仅微调全精度权重和激活量化参数；(2) 微调无需引入额外参数（与 EfficientDM 的 LoRA 不同），直接修改原始权重；(3) 数据无关（校准集来自高斯噪声）；(4) 训练效率高——LDM-4 W4A8 仅需 0.45 GPU 小时（A6000），而 EfficientDM 需 2.60 小时，Full-finetune 需 0.85 小时但显存更高（15076MB vs 12178MB）；(5) 可扩展到 Stable Diffusion（单 48GB GPU 即可完成，而 Full-finetune 会 OOM）；(6) 集成 LoRA 反而降低性能（FID 增加 5.62），验证了直接微调原始层权重的有效性。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Imbalanced Activation Distributions in Diffusion Models (扩散模型激活分布不平衡)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
扩散模型激活分布不平衡是指：在全精度扩散模型的部分层中，激活值绝大多数集中在零附近（如 [-0.6, 1.7]），但同时存在稀疏但数值较大的离群值（如总范围 [-10, 34]）。这些稀疏大值对生成质量至关重要——替换最大值 token 为零会导致图像严重退化，而替换随机 token 几乎无影响。这种分布对低比特量化构成双重挑战：(1) 大值的量化：若缩小裁剪范围以适应小值，大值会被严重裁剪（clipping error 大）；(2) 小值的量化：若扩大范围覆盖大值，大量小值的量化精度急剧下降（rounding error 大）。在 4-bit（仅 16 个量化级）下，此矛盾尤为尖锐——PTQ 方法无法找到合适的 trade-off，导致生成失败。该现象在 LDM 和 Stable Diffusion 中普遍存在（附录 Fig. 5 展示了多个模型和数据集上的不平衡分布）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
分析激活分布不平衡的流程：
```
# 检测激活分布不平衡
for layer in model.layers:
    for t in sample_time_steps():
        x_T = randn()
        a = model.forward(x_T, t, stop_at=layer)
        hist = histogram(a, bins=100)
        # 分析分布特性
        range_total = max(a) - min(a)           # 总范围（如 44）
        pct_99_range = percentile(a, 99) - percentile(a, 1)  # 99%分位范围
        imbalance_ratio = range_total / (pct_99_range + eps)
        # QuEST 发现：imbalance_ratio 在某些层 >> 10
        # 大值稀疏但重要 → 低比特量化困难
```
QuEST 的解决方案不是直接修改激活分布（不可操作），而是通过微调权重间接调整：微调后激活范围从 [-10, 34] 缩小到 [-4, 14]，标准差从 0.171 降至 0.157，分布更紧凑但均值保持一致——既减少了大值的稀疏性又保护了小值的量化精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该发现的实际意义：(1) 它是 QuEST 方法的核心 motivation——解释了为何需要微调而非仅调量化参数；(2) 通用性——在条件 LDM-4（ImageNet）、无条件 LDM-4（LSUN-Bedrooms）和 Stable Diffusion 中均观测到，说明是扩散模型的固有特性而非特定模型的偶然现象；(3) 与 LLM 中的激活异常值（outlier）问题有相似性——如 LLM.int8() 和 SmoothQuant 也处理类似的分布不平衡——但扩散模型的不平衡更温和（数值范围小），且与批次和时序动态耦合；(4) 可用作诊断工具——在量化任何扩散模型前先检查各层激活分布，识别潜在的问题层。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Time Embedding Layers in Diffusion Models (扩散模型中的时间嵌入层)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
时间嵌入层是扩散模型中将离散时间步 t（整数）转换为连续向量表示的全连接层（通常 1-2 层）。其作用是将时序信息注入模型：时间嵌入向量通过投影层（projection layers）变换后，在不同深度与 latent image representation 合并（通常通过加法或 FiLM 式调制），使模型在每个去噪阶段执行与时间步相适应的操作。QuEST 识别出这一机制在量化中的特殊重要性（Property ❶）：时间嵌入精度直接影响模型在不同时间步的执行正确性——量化的时间嵌入精度下降会导致时间步与模型功能不匹配，进而引起去噪序列震荡（oscillation），使 FID 恶化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
时间嵌入的计算过程伪代码：
```
# 标准扩散模型时间嵌入
def time_embedding(t, w1, w2):
    # t: 整数时间步 (如 150/200)
    # Step 1: 正弦位置编码
    half_dim = embedding_dim // 2
    emb = exp(arange(half_dim) * -log(10000) / (half_dim-1))
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    emb = cat([sin(emb), cos(emb)], dim=-1)   # [1, embedding_dim]

    # Step 2: 线性变换（1 或 2 层）
    emb = linear(emb, w1)                      # 第一层
    emb = silu(emb)                            # 激活函数
    emb = linear(emb, w2)                      # 第二层 → time_emb
    return emb                                  # [1, model_dim]

# 注入 UNet（在各 ResBlock/Attention Block）
def forward(x, t):
    t_emb = time_embedding(t, w1, w2)   # 获取时间嵌入
    for block in unet_blocks:
        # 通过投影层将 t_emb 映射到与 x 相同维度
        scale, shift = proj_out(t_emb).chunk(2)
        x = block(x)
        x = x * (1 + scale) + shift     # 时间条件调制
    return x
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
扩散模型时间嵌入的量化策略：(1) QuEST 的 TLA：同时微调时间嵌入层的全精度权重和其激活量化参数，使量化时间嵌入输出与全精度版本对齐；(2) PTQ baseline：直接量化时间嵌入层权重和激活（不微调），导致 FID 上升 0.81-1.04；(3) 预计算方法：预先计算所有时间步的时间嵌入并直接查表使用——这忽略了量化模型中各模块的兼容性变化（不同模块量化后对相同时间嵌入的响应不同），因此性能不如微调；(4) 其他方法：TFMQ-DM 校准所有时间步的时间嵌入层和投影层；TDQ 通过简单网络学习跨时间步的动态量化参数。时间嵌入层参数极少（<1% 总参数），因此微调成本极低但同时效果显著。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Progressive Alignment with Global Loss in Quantization (渐进对齐与全局损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
渐进对齐与全局损失是 QuEST 的三阶段优化策略的统称：(1) 阶段一 TLA（Temporal Layer Alignment）：独立微调时间嵌入层，利用时序信息独立于图像输入且在模型早期确定的特性；(2) 阶段二 CMA（Critical Module Alignment）：在 TLA 冻结后微调注意力相关层；(3) 全局损失 L_G = E_t[||O(x_t; w) - Õ(x_t; w, s)||²]，在 CMA 阶段叠加，为所有未选中的层提供网络级梯度信号。渐进式设计的原因：时间嵌入和注意力层功能不重叠，且时间信息先于空间/语义信息处理，因此应先对齐时间嵌入再对齐注意力。全局损失的重要性：仅用局部损失（TLA+CMA w/o L_G）FID 为 8.99（TLA）和 6.41（CMA）；添加 L_G 后分别改善至 6.41 和...（CMA+L_G 效果未单独给出但联合使用最显著）。有趣的是，**仅使用全局损失**会导致性能退化 7.13 FID，说明局部+全局的组合是必需的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三阶段联合优化的最终目标函数：
```
argmin_{w_l} (L_TLA + L_CMA + 2*L_G),  l ∈ C_TE ∪ C_A

其中：
L_TLA = Σ_{l∈C_TE} E_t[||FP_time_embed(t) - Q_time_embed(t)||²]
         → 微调：w_l (l∈C_TE), s_l (l∈C_TE)

L_CMA = Σ_{l∈C_A} E_t[||FP_attn(z_l) - Q_attn(z̃_l)||²]
         → 微调：w_l (l∈C_A), ŝ = s \ s_l (l∈C_TE)

L_G   = E_t[||FP_final(x_t; w) - Q_final(x_t; w̃, s)||²]
         → 微调：w_l (l∈C_TE∪C_A), s (全部激活量化参数)

# 渐进执行
# Step 1: TLA 独立训练（w_TE, s_TE）
# Step 2: CMA + 2*L_G 联合训练（w_A, ŝ）
# 注意：Step 2 中权重 w_A 梯度来自 (L_CMA + 2*L_G)，
#       ŝ 梯度来自 (L_CMA + 2*L_G)，
#       这使未选中层的量化参数 s 通过 L_G 获得间接优化
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现细节：(1) 权重 L_G 系数为 2（相对 L_TLA 和 L_CMA 各 1），实验通过网格搜索确定；(2) 网络级训练（network-wise training）——所有激活量化参数一次性优化，而非逐层/逐块重建（PTQ 方式），显著节省时间；(3) Adam 优化器，权重学习率 1e-5，量化参数学习率 1e-4（量化参数需要更大学习率因为对其初始估计更粗糙）；(4) 2300 次迭代完成（vs EfficientDM 的 32000 次）；(5) 该策略可扩展到 Stable Diffusion 等更大模型——Full-finetune OOM，但 QuEST 可在 48GB GPU 上完成；(6) 仅 L_G 不足以获得好性能（原因：全局损失信号对深层参数太弱，需要局部对齐的强梯度信号）。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## AdaRound (Adaptive Rounding, 自适应舍入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaRound（Adaptive Rounding）是一种后训练量化中的权值舍入策略，由 Nagel et al. (ICML 2020) 提出。标准量化使用 round-to-nearest（就近舍入），但 AdaRound 发现：对于最小化任务损失，学习一个是否向上或向下舍入的决策变量比始终就近舍入更好。具体实现：对每个权值 w_i 学习一个连续变量 v_i ∈ [0,1]（通过 sigmoid 约束），最终舍入方向由 round(w_i/s + σ(v_i)-0.5) 决定，其中 σ 为 sigmoid 函数；v_i 通过重建损失（MSE 量化输出 vs 全精度输出）优化。QuEST 将 AdaRound 作为其 baseline 方法的量化策略（表 6 中 "Baseline" = 直接量化 + AdaRound），并在此基础上叠加 TLA、CMA 和 L_G 展示递进式改善。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AdaRound 核心伪代码：
```
# 标准 Round-to-Nearest
W_int = clamp(round(W / s) + Z, qmin, qmax)

# AdaRound: 学习舍入方向
V = randn_like(W) * 0.01               # 可学习变量
for step in range(adapt_round_steps):
    # 前向：软舍入
    W_soft = clamp(floor(W / s) + σ(V) + Z, qmin, qmax)  # σ(V) ∈ [0,1]
    W_deq = (W_soft - Z) * s
    output_q = layer(W_deq, x_calib)    # 量化后输出
    loss = MSE(output_fp, output_q)     # 重建损失
    loss.backward()                     # 仅更新 V
# 推理时固定舍入决策
W_int_final = clamp(floor(W / s) + round(σ(V_final)) + Z, qmin, qmax)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaRound 通常作为 PTQ pipeline 的最后一步，在校准集上逐层优化舍入方向变量。在 QuEST 的上下文中，AdaRound 是 baseline 的量化策略但不作为论文的核心贡献——论文的重点是通过选择性微调超越 AdaRound 的效果。实际使用：(1) AdaRound 不需要反向传播通过量化器（仅优化 V，梯度通过 soft rounding）；(2) 适用于权重和激活；(3) 在 8-bit 效果较好，在 4-bit 效果有限（舍入误差远超 AdaRound 的调整空间），这正是 QuEST 需要权重微调来补充的原因。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

## Incoherence Processing (非相干处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Incoherence Processing（非相干处理）是 Chee et al. (2023) 在 QuIP 中首次提出、QuIP# 进一步改进的一种 LLM 量化前预处理技术。其核心思想是：在量化之前，通过将权重矩阵 W 和代理 Hessian H 与随机正交矩阵做共轭变换（双侧乘法），使 W 和 H 变为 "μ-incoherent"——即矩阵所有元素的幅值高度集中、不存在离群值。形式化定义（Definition 2.1）：Hessian H ∈ R^{n×n} 是 μ-incoherent 的，若其特征分解 H = QΛQᵀ 满足 max|Q_ij| ≤ μ/√n；权重矩阵 W ∈ R^{m×n} 是 μ-incoherent 的，若 max|W_ij| ≤ μ‖W‖_F/√(mn)。非相干性使得量化时每个坐标的误差贡献均匀，任何单一方向不会被过度惩罚，从而在理论上保证量化误差有界（provably bounded error）。QuIP# 的 incoherence processing 使用 Randomized Hadamard Transform (RHT)：对 H 和 W 做 Ŵ ← Had(S_U · Had(S_V · W^T)^T), Ĥ ← Had(S_V · Had(S_V · H)^T)，其中 S_U, S_V 是随机 ±1 对角矩阵，Had 是正交 Hadamard 矩阵。变换保持代理损失不变（tr((UWV^T)(VHV^T)(VW^TU^T)) = tr(WHW^T)）。推理时，量化后的模型通过两次 Hadamard 变换撤销量化的预处理效果：y ← Had(S_U ⊙ decompress_multiply(Ŵ, C, Had(S_V ⊙ x)))。与启发式离群值抑制方法（AWQ 的 per-channel scaling、OmniQuant 的可学习变换）相比，incoherence processing 有严格的理论保证，且不增加 per-group scaling 的额外存储开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 QuIP# 的 incoherence processing 流程（Algorithm 3: IP-RHT）为例：
```
# 输入: W ∈ R^{m×n}, H ∈ R^{n×n}
# S_U ∼ U{±1}^m, S_V ∼ U{±1}^n  (随机符号向量)

# Step 1: 对 W 做双边 RHT 变换
W_tmp = diag(S_V) @ W^T         # m×n → n×m, 逐行符号翻转
W_tmp = Had(W_tmp)               # Fast Walsh-Hadamard Transform (O(m log n))
W_tmp = W_tmp^T                  # n×m → m×n
W_hat = Had(diag(S_U) @ W_tmp)^T # 逐行符号翻转 + FWHT → Ŵ

# Step 2: 对 H 做双边 RHT 变换
H_tmp = diag(S_V) @ H            # n×n
H_tmp = Had(H_tmp)^T
H_hat = Had(diag(S_V) @ H_tmp)^T # Ĥ

# Step 3: 输出 Ŵ, Ĥ, S_U, S_V
# 推理时撤销量化:
# y = Had(S_U ⊙ decompress_multiply(Ŵ_quantized, C, Had(S_V ⊙ x)))
```
QuIP# RHT vs QuIP Kronecker 的理论优势：Lemma 3.1 证明 RHT 实现 μ_H = √(2log(2n²/δ)) 和 μ_W = 2log(4mn/δ)，而 Kronecker 方法的 μ_W^{Kron} = A²log²(4Cmn/δ)²（log 依赖 vs log² 依赖）。时间复杂度 O(n log n) vs O(n√n)。消融实验验证：QuIP# 仅 RHT 替换 Kronecker 即显著降低困惑度（Llama 2 70B 2-bit: 4.58 vs 5.90 Wikitext2）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Incoherence processing 实现要点：(1) Hadamard 矩阵获取：使用 Neil Sloane 维护的已知 Hadamard 矩阵库（http://neilsloane.com/hadamard/）；(2) 非 2 的幂维度处理：分解 n = p×q（p 为最大 2 的幂，q 存在已知 Hadamard 矩阵），V = H_p ⊗ H_q，复杂度 O(q²p log p)；(3) 若无合适 Hadamard 分解，QuIP# 提供 RFFT（Randomized Fast Fourier Transform）替代方案——使用 FFT + 随机复数相位，仅需 n 为偶数，理论界类似，实践略逊于 RHT（2-7B: 8.30 vs 8.22 WikiText2）；(4) 开源实现：https://github.com/Cornell-RelaxML/quip-sharp，提供 CUDA 加速的 FWHT kernel。在更广泛的量化文献中，incoherence processing 的思想已被后续工作采纳：RoMeo（旋转混合精度量化）、GyRot（旋转+分组量化）、SpinQuant 等均使用正交旋转矩阵抑制离群值，只是旋转矩阵类型（Hadamard vs 随机正交 vs Cayley 参数化）和变换粒度（全局 vs block-wise）有所不同。QuaRot 的关键创新在于将 incoherence processing 从仅用于权重量化扩展到同时处理激活值量化：通过计算不变性定理，QuaRot 将离线 Hadamard 变换融入权重矩阵，使跨层激活值自动经过 incoherence processing（X→XQ），从而从根源上消除激活值离群值（图 1 直观对比），使得激活值的 4-bit 量化成为可能。这与 QuIP# 的关键区别：QuIP# 在推理时需为每个权重矩阵执行两次在线 Hadamard 变换（撤销量化预处理），而 QuaRot 将大部分 Hadamard 变换离线融入权重，仅保留每层 1.5 次在线变换（down-projection 和 out-projection 各一次 head Hadamard），大幅降低推理开销。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，incoherence processing 的思想被应用于 QA-SFT 训练而非仅 PTQ 推理。RoSTE 将旋转矩阵 R_i 的选择整合为 bilevel optimization 的下层问题：选择 R_i 以最小化 weight-activation quantization error surrogate loss（12），同时上层通过 STE 优化量化权重。RoSTE 的关键洞察是：并非所有层都受益于旋转——某些层旋转后可能引入新的 outlier（如 Pythia 的末层）。因此采用自适应策略逐层在 I（无旋转）和 H（Walsh-Hadamard 旋转）间选择，仅当旋转降低该层量化误差时才应用。旋转分为 between-block (R1) 和 in-block (R2, R3, R4)，其中 R1/R2 可离线吸收到权重中，R3/R4 通过 fast Hadamard CUDA kernel 在线执行。

---

## Randomized Hadamard Transform (RHT) for Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Randomized Hadamard Transform (RHT) 在 QuIP# 中特指用于量化非相干处理的随机化正交变换：x → V S x，其中 V ∈ R^{n×n} 为正交的 scaled Hadamard 矩阵（V^T V = I），S = diag(s), s ∼ U{±1}^n 为随机符号对角矩阵。RHT 最早由 Halko et al. (2011) 在随机数值线性代数中引入，用于近似矩阵分解。QuIP# 首次将 RHT 应用于 LLM 量化，替代 QuIP (Chee et al., 2023) 的 2-factor Kronecker 积正交矩阵构造。RHT 的核心性质：(1) 将任意向量与 Hadamard 矩阵相乘使输出向量在所有坐标上"扩散"——对标准基向量 e_i，输出 H e_i 的所有元素幅值为 1/√n（完全非相干）；(2) 随机符号翻转提供了随机化保证（Lemma 3.1），将矩阵变为 μ-incoherent；(3) 通过 Fast Walsh-Hadamard Transform (FWHT) 实现 O(n log n) 计算，仅需加减法无需浮点乘法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuIP# 中 RHT 在量化前/推理时的完整计算流程：
```
# === 量化前 (Algorithm 3: IP-RHT) ===
# W ∈ R^{m×n}, H ∈ R^{n×n}
# 采样随机符号: S_U ∼ {±1}^m, S_V ∼ {±1}^n

# RHT 应用于权重矩阵 W (双边):
# Ŵ = Had(diag(S_U) × Had(diag(S_V) × W^T)^T)
#    = H_m × S_U × W × S_V × H_n^T
# 其中 Had(·) 为 FWHT (O(n log n))

# RHT 应用于 Hessian H (双边):
# Ĥ = H_n × S_V × H × S_V × H_n^T
#    = Had(diag(S_V) × Had(diag(S_V) × H)^T)

# === 推理时 (Algorithm 2) ===
# y = Had(S_U ⊙ (quantized_Ŵ × Had(S_V ⊙ x)))
#   = H_m × S_U × (quantized_Ŵ) × S_V × H_n^T × x

# 缩放因子: ρ|W|_F 乘以权重矩阵使其幅值匹配 E8P 码书覆盖范围
```
QuIP# 的 key insight：RHT 将权重分布转换为近似球状亚高斯分布（中心极限定理效应），这为后续 E8P 球状格基码书的向量量化创造了理想条件——"先让数据变高斯，再用球状码书去量化高斯数据"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RHT 的实现关键点：(1) FWHT kernel 使用 Tri Dao 的 CUDA 实现（https://github.com/Dao-AILab/fast-hadamard-transform），通过 in-place butterfly 运算实现，log₂n 层，每层 n/2 对 (a+b, a-b) 操作；(2) 符号向量 S_U, S_V 在无微调时以 1-bit/元素存储（±1），微调时以 FP16 存储以允许梯度优化；(3) 存储开销极低：对 m×n 矩阵，16(n+m)/(nm) bits/weight（微调时）或 (n+m)/(nm)（无微调），对于 4096×4096 的最小矩阵仍 <0.01 bits/weight；(4) 维度非 2 的幂时使用 Kronecker 分解 H = H_p ⊗ H_q，复杂度 O(q²p log p)；(5) 备选方案 RFFT：使用 FFT + 随机复数相位，仅需 n 为偶数，适合无 Hadamard 矩阵的高速硬件（如 DSP/FPGA）。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---

## E8 Lattice / E8P Codebook (E8 格基码书)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
E8 Lattice（E8 格）是 8 维空间中的一种特殊格（lattice）结构，由 Viazovska (2017) 证明其实现了 8 维空间中的最高密度球填充（optimal sphere packing），拥有最优的 kissing number（240，即每个球最多接触 240 个等大小球）。E8 格的数学定义：E8 = (Z⁸ ∪ (Z⁸ + ½)) ∩ {x | 1^T x 是偶数}，即所有整数向量和所有半整数向量中，分量和为偶数的向量集合。E8P（"E8 Padded"）是 QuIP# 基于 E8 格构造的 2-bit 8 维向量量化码书。E8P 包含 2¹⁶ = 65536 个码书条目（每个编码 8 个权重），但利用 E8 格的对称性（符号翻转），仅需存储 2⁸ = 256 条源码书（1KiB），解码时通过 7+1 位符号位和 1 位偏移位恢复完整的 16-bit 码字→8 维 FP 向量。E8P 的构造：从等价表示 D̂₈ = {x ∈ Z⁸ + ½ | 1^T x 是偶数} 出发，利用 (D̂₈ − ¼) ∪ (D̂₈ + ¼) = E8 + ¼ 的移位不变性，选择 S ⊂ |D̂₈| 中范数 ≤ √10 的 227 个元素 + 范数 √12 的 29 个"padding"元素，共 256 条源码书。解码时：8 bits 查 S 得绝对值向量，7 bits 控制 7 个符号翻转（第 8 符号由奇偶性推断），1 bit 控制 ±¼ 偏移。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
E8P 在 QuIP# BlockLDLQ 中的使用（以 2-bit 量化一列 8 维权重块为例）：
```
# E8P 解码一个 16-bit 码字 → 8 维浮点向量
# 输入: codeword (16 bits)
# 输入: codebook_abs[256] (256 条源码书，每条 8×4bit 压缩)

function decode_e8p(codeword):
    s_idx = codeword[0:8]           # 8 bits: 源码书索引 (0-255)
    sign_bits = codeword[8:15]      # 7 bits: 符号翻转控制
    shift_bit = codeword[15]        # 1 bit: ±1/4 偏移

    s = codebook_abs[s_idx]         # 查表得 8 个绝对值 (∈ |D̂_8|)
    
    # 确定需要奇数还是偶数个符号翻转
    # (取决于 s 是否在 D̂_8 中)
    parity_needed = parity_required(s)
    
    # 应用 7 个符号位，推断第 8 个
    for i in 0..6:
        if sign_bits[i]: s[i] = -s[i]
    # 第 8 个符号由 parity 推断
    s[7] *= sign_from_parity(sign_bits, parity_needed)
    
    # 应用 ±1/4 偏移
    offset = 0.25 if shift_bit else -0.25
    return [x + offset for x in s]

# 量化时: 找最近 E8P 码字
# w_8d ∈ R^8 → 遍历 256 条源码书，检查符号翻转组合
# → 找 L2 距离最小的码字 → 输出 16-bit 码字
# (实际实现使用 lattice 解码算法而非暴力搜索)
```
E8P 码书形状与 RHT 变换后的高斯分布的 MSE 最优性：图 3 显示 E8 基码书对所有基准码书（D₄ 格、半整数格等）实现最低的元素级 MSE，因为 E8 的 8 维高 packing density + 球状形状匹配高斯分布。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
E8P 的硬件效率设计：(1) 源码书仅 256×8 条目 × 4bit/条目 = 1KiB，可放入任何现代 GPU 的 L1 cache，即使 32× 复制消除 bank conflict 也仅 32KiB；(2) CUDA kernel 中解码一个 8 维向量需 <5 指令/权重——XOR 符号翻转 + 移位查表 + lop3 融合位操作；(3) 码字存储格式：每个 uint2 打包 4 个 16-bit E8P 码字（= 32 个权重），有效 2 bits/weight；(4) Tensor Core MMA：解码后的 FP16 权重直接送入 `mma.sync.aligned.m16n8k16` PTX 指令做矩阵乘累加，无写入 global memory 的中间步骤；(5) 高比特扩展（RVQ）：4-bit = 2×2-bit E8P，3-bit = 2-bit E8P + 1-bit E8；(6) scaling 因子 ρ 选择：ρ ≈ 0.9（2-bit E8P），通过最小化高斯→E8P 量化 MSE 确定。开源代码：https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quiptools/quiptools_e8p_gemv.cu。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---

## BlockLDLQ (Block Low-Density Layered Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BlockLDLQ 是 QuIP# 提出的一种向量量化兼容的自适应舍入算法，是对 QuIP (Chee et al., 2023) 标量 LDLQ 的块级推广。传统 LDLQ 基于 Hessian H 的 LDL 分解（H = L^T D L），设置反馈矩阵 U = L^T - I，逐列迭代舍入 Ŵ_k = Q(W_k + (W_{:k-1} - Ŵ_{:k-1})a_k)，其中 Q 为标量量化器。BlockLDLQ 的改进：(1) 将列级舍入推广到列块级——以 g 列（QuIP# 中 g=8，匹配 E8P 维度）为一块；(2) 基于 g-block LDL 分解 H = L^T D L，其中 L 为单位块下三角矩阵（n/g 个对角块均为 I_{g×g}），D 为块对角矩阵；(3) 反馈矩阵 U = L^T - I；(4) 量化步骤：Ŵ_k = Q(W_k + (W_{:k-1} - Ŵ_{:k-1})A_k)，其中 A_k ∈ R^{n×g} 为 U 的第 k 个 g 列块，Q 为向量量化器（E8P）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BlockLDLQ 的完整计算流程（以 Llama 2 线性层 W ∈ R^{4096×4096}, g=8 为例）：
```
# 输入: W_hat ∈ R^{m×n} (RHT 变换后), H_hat ∈ R^{n×n} (RHT 变换后)
# 参数: g = 8 (block size), codebook C ∈ R^{256×8} (E8P)

# Step 1: g-block LDL 分解
# H_hat = L^T @ D @ L
# L: unit block lower triangular (512×512 blocks of 8×8 each)
# D: block diagonal (512 blocks of 8×8 each)
# 实现: 从 Cholesky 分解 H_hat = G^T G 导出 L 和 D

# Step 2: 设置反馈矩阵
# U = L^T - I  (unit block upper triangular)

# Step 3: 逐块自适应舍入 (n/g = 512 blocks)
for k from 1 to 512:
    # 当前块的列范围: [(k-1)*8 : k*8]
    A_k = U[:, (k-1)*8 : k*8]     # 反馈向量, size: n × 8
    
    # 之前已量化块的误差
    past_error = W_hat[:, 0:(k-1)*8] - W_quantized[:, 0:(k-1)*8]
    
    # 带反馈的预舍入值
    W_pre = W_hat[:, (k-1)*8 : k*8] + past_error @ A_k
    
    # 对 8 列中的每行独立做 E8P VQ
    for row in 0..m-1:
        W_quantized[row, (k-1)*8 : k*8] = e8p_quantize(W_pre[row, :])
```

误差理论界（Theorem 4.1）：若 H 是 μ-incoherent 且 Q 满足 E[(Q(x)-x)(Q(x)-x)^T] ≤ σ²I，则 E[tr((Ŵ-W)H(Ŵ-W)^T)] ≤ gmμ²σ²/n · tr(H^{1/2})²。与独立块舍入相比，BlockLDLQ 的界从 tr(H) 改善到 tr(H^{1/2})²/n（继承 QuIP 标量 LDLQ 的相同改进因子）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BlockLDLQ 的实现关键：(1) g-block LDL 分解通过对 H 的 Cholesky 分解 H = G^T G 做块划分导出——对角块 D_k = G_{kk}^T G_{kk}，下三角块 L_{jk} = G_{jj}^{-T} G_{jk}^T；(2) 在 QuIP# 中 g=8，与 E8P 的 8 维向量量化完全匹配；(3) 对 m≫n 的大矩阵（LLM 线性层），内循环近似 O(m·2^{kd}·g)（每行搜索 2^{16} 条目），E8P 的 256× 压缩将搜索空间从 2^{16} 降到实际的 256 次查表；(4) 开源：https://github.com/Cornell-RelaxML/quip-sharp，BlockLDLQ 在 Python 层面实现，E8P 量化器在 C++/CUDA 层面实现。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---

## Residual Vector Quantization (RVQ) in PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Residual Vector Quantization (RVQ，残差向量量化) 是一种多阶段向量量化方法，最早由 Juang & Gray (1982) 在语音编码领域提出。在 LLM PTQ 中，RVQ 用于将低比特 VQ 方法扩展到更高比特：给定一个目标向量 x 和目标总位宽 p，使用一组 q_i-bit 的码书逐残差量化——第一阶段用 q_0-bit 码书量化 x 得 δ_0，第二阶段用 q_1-bit 码书量化残差 (x - δ_0)/s_1 得 δ_1，依此类推，最终 Ŵ = Σ δ_i · s_i。QuIP# 使用 RVQ 将 2-bit E8P 扩展到 3-bit 和 4-bit：(a) 4-bit = 2× E8P 2-bit RVQ（ρ₁≈1.03, ρ₂≈3.45）；(b) 3-bit = E8P 2-bit（ρ≈0.98）+ E8 1-bit（范数≤2 的 E8 元素 + 15 个范数 4 元素，ρ≈2.04）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuIP# 中 4-bit RVQ 的量化流程（在 BlockLDLQ 内部，每块 8 维）：
```
# w ∈ R^8 (待量化的 8 维权重块)
# RVQ(w, p=4, q=[2,2])

# Stage 1: 2-bit E8P 量化, scale ρ₁
delta_1 = e8p_quantize(w / rho_1) * rho_1

# Stage 2: 量化残差
residual = (w - delta_1) / rho_2
delta_2 = e8p_quantize(residual) * rho_2

# 最终量化值
w_hat = delta_1 + delta_2

# 输出: 2 个 16-bit 码字 → 4 bits/weight (平均)
```
与标量量化（SQ）对比的优势：RVQ 的每个阶段使用高维 VQ（8D），保留了跨维度形状信息。SQ 在 4-bit 下退化为 1D 16-level 均匀量化，无法捕获多维分布的形状；RVQ 的 2-stage E8P 用 2×2-bit = 4-bit 实现多维球状码书，更匹配 RHT 变换后的高斯权重分布。论文提到更高级的多码书量化方法（如 Additive Quantization）可能进一步改进，但 RVQ "已足够"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RVQ 的实现要点：(1) 各阶段缩放因子 ρ_i 通过最小化高斯→RVQ 量化误差确定（数值搜索），各模型略有不同（ρ 因 incoherence processing 不完全产生精确高斯）；(2) 推理时每个阶段独立解码——stage 1 E8P 码字 + stage 2 E8P 码字 → 两个 8D 向量加权求和 → 送入 MMA；(3) RVQ 的计算开销近似正比于阶段数，但所有阶段可融合在同一 kernel 内完成；(4) RVQ 不限于 E8P——任何低比特 VQ 码书均可作为 RVQ 的子阶段（如 GPTVQ 使用 2×2D VQ 做 4-bit 量化）。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---

## Fine-Tuning for PTQ (后训练量化的微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-Tuning for PTQ 是 AQLM (Egiazarian et al., 2024) 引入、QuIP# 进一步改进的一种 PTQ 与 QAT 的混合方法。纯 PTQ 仅基于校准数据优化单层量化参数（scale、rounding），忽略层间交互导致的激活误差累积。Fine-Tuning for PTQ 在量化后对未量化参数进行小规模微调（"adapting"而非"training from scratch"），以恢复层间保真度。QuIP# 的微调分两阶段：(1) Per-Transformer-Block 微调——在每个 Block 内，冻结已量化的线性层权重，Adam 优化未量化层（后续的线性层、layernorm）和 sign vectors S_U, S_V，最小化 Block 输出 MSE；(2) 端到端微调——所有 Block 量化后，优化 layernorms、所有 S_U/S_V、LM head，最小化 CrossEntropy(量化模型输出, 全精度模型 logits)。关键创新：(a) sign vectors 以 FP16 存储（非 bitvector），允许梯度优化——这让 incoherence processing 可以"学习"如何最好地旋转权重以匹配 E8P 码书形状；(b) 微调数据量极小——256 条 RedPajama 序列训练 + 128 验证，5 epochs；(c) 计算成本低——70B 模型约 50 GPU-hours（对比 LLM-QAT 960 GPU-hours 仅生成训练数据）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuIP# Fine-Tuning 流程（Algorithm 5）：
```
# 输入: 未量化模型 M, Dev set D, 量化顺序 O
# === Phase 1: Per-Block Fine-Tuning ===
X = M.embedding(D)                # token embeddings
for each Decoder Block in M:
    Y = Block(X)                  # FP32 block 输出 (教师)
    X_train, Y_train, X_valid, Y_valid = split(X, Y)
    for each Linear Layer L in Block (按 O 的顺序):
        L_hat = QuIP#-NoFT(L)     # 量化当前层
        L.weight.requires_grad = False  # 冻结量化权重
        # 优化 Block 内参数以最小化 MSE
        optimizer = Adam([Block.unquantized_params, L.S_U, L.S_V], lr=5e-5)
        for epoch in range(5):
            loss = MSE(Block(X_train), Y_train)
            loss.backward(); optimizer.step()
        # 用验证集早停选择最优参数
    X = Y                          # 下一 Block 的输入

# === Phase 2: End-to-End Fine-Tuning ===
# 所有 Block 量化完成，仅 layernorms + S_U + S_V + LM head 可训练
D_train, C_train, D_valid, C_valid = split(D, full_model_logits)
optimizer = Adam([layernorms, all_S_U, all_S_V, lm_head], lr=5e-5)
for epoch in range(5):
    loss = CrossEntropy(M(D_train), C_train)
    loss.backward(); optimizer.step()
```
微调效果：2-bit 模型受益最大（Llama 2 7B: 8.22→6.19 WikiText2），3/4-bit 也有改善。消融显示 FT 改善独立于 E8P 和 RHT 的改善（三组件叠加增益）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 每个 Block 内各层量化顺序 O 可影响最终质量，论文未深入讨论顺序优化（likely sequential by layer position）；(2) 对 2-bit 模型，S_U/S_V 的学习率提高到 5e-4（vs 5e-5），因为在极低比特下 sign vector 的旋转自由度更关键；(3) 70B 端到端微调时序列长度从 4096 降到 3072 以避免 OOM；(4) 开源：https://github.com/Cornell-RelaxML/quip-sharp 提供完整微调脚本；(5) 微调仅需小规模通用文本（RedPajama），无需任务特定数据，保持 PTQ 的数据高效优势。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

## End-to-End Joint Weight-Activation-KV Cache Quantization (W4A4KV4 / 端到端联合量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
End-to-End Joint Quantization (W4A4KV4) 是指将 LLM 的所有线性层权重、中间激活值、以及 KV Cache 全部量化为 4-bit 精度（INT4）的量化方案。与传统仅权重量化（W4A16）或 8-bit 激活量化（W8A8）不同，W4A4KV4 要求：(1) 权重以 INT4 存储并在矩阵乘法时反量化；(2) 激活值在每次前向传播时在线量化为 INT4（per-token 对称量化），矩阵乘法在 INT4×INT4 精度下进行；(3) Key 和 Value 向量在存入缓存时量化为 INT4（asymmetric group-wise），解码时从缓存加载后反量化。QuaRot 是首个声称实现端到端 4-bit 量化的方法，其核心贡献在于通过随机 Hadamard 旋转从根源上消除所有三类张量中的离群值，使统一的对称/非对称 INT4 量化成为可能。LLAMA2-70B 在 W4A4KV4 下仅损失 0.47 WikiText-2 困惑度、保持 99% 零样本精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuaRot 中 W4A4KV4 的完整前向传播流程（单层 Transformer）：
```
# === 输入: FP16 X ∈ R^{T×d}, INT4 packed weights, INT4 KV cache ===

# 1. RMSNorm (FP32, 无scale)
X_norm = X / sqrt(mean(X^2) + eps)

# 2. Attention: Q/K/V projection (W4A4)
for W in [W_q, W_k, W_v]:   # W_q 等已离线融入随机Hadamard Q^T
    s_x = max(|X_norm|, dim=1) × 0.9 / 7   # per-token scale, clip=0.9
    X_q = round(clip(X_norm/s_x, -7, 7))    # INT4 activation
    Y_int = CUTLASS_INT4_GEMM(X_q, W_q)     # TensorCore INT4×INT4
    Y = (float(Y_int) * s_x[:, None] * s_w[None, :])  # dequant → FP16

# 3. RoPE + Post-RoPE Hadamard (在线 head-wise)
Q_h, K_h = RoPE(Q), RoPE(K)
Q_h = Q_h @ (I ⊗ H_{d_h})   # head-wise Walsh-Hadamard, O(d_h log d_h)
K_h = K_h @ (I ⊗ H_{d_h})

# 4. KV Cache 量化存储 (asymmetric group-wise, group=128)
for each group g in K_h:
    z_k = min(K_h[g]), s_k = (max(K_h[g]) - z_k) / 15
    K_q[g] = round(clip((K_h[g] - z_k)/s_k, 0, 15))   # INT4
# V 同理量化存储

# 5. Attention 计算 (FP16, 在线反量化KV)
scores = Q_h @ Dequant(K_q)^T / sqrt(d_h)
attn_out = softmax(scores) @ Dequant(V_q)

# 6. Hadamard heads (在线) + Out-projection (W4A4)
Z_h = attn_out @ (H_{n_h} ⊗ I)   # Kronecker Hadamard, reshape+WHT
Z_q = round(clip(Z_h/s_z, -7, 7))
O_int = CUTLASS_INT4_GEMM(Z_q, W_out_q)
O = dequant(O_int, s_z, s_wo)

# 7. FFN: Gate/Up projection (W4A4)
gate_q = round(clip(X_norm/s_x, -7, 7))
gate = SiLU(dequant(CUTLASS_INT4_GEMM(gate_q, W_gate_q)))
up = dequant(CUTLASS_INT4_GEMM(gate_q, W_up_q))

# 8. 在线 Hadamard (down-projection 前, FP16) + Down-projection (W4A4)
down_in = FastHadamard(gate * up)   # O(d log d) WHT
d_q = round(clip(down_in/s_d, -7, 7))
D_int = CUTLASS_INT4_GEMM(d_q, W_down_q)
D = dequant(D_int, s_d, s_wd)

# 9. Residual connection
X_out = X + O + D  # 所有 MatMul 均为 INT4×INT4
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 W4A4KV4 需要：(1) 离线阶段：用计算不变性定理将旋转矩阵融入权重（RMSNorm scale 吸收 + 随机 Hadamard 融合），约 5 分钟（LLAMA2-70B on A100）；GPTQ 权重量化约 2 小时；(2) 在线推理：CUTLASS 提供 INT4 TensorCore GEMM kernel，FlashInfer 提供量化 KV Cache attention kernel，快速 Hadamard kernel 实现 O(d log d) 在线变换。加速比：LLAMA2-70B prefill 3.33× (batch=64, seq=2048, RTX 3090)，解码内存节省 3.89×。代码开源：https://github.com/spcl/QuaRot。

涉及论文标题：
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

---

## Outlier Features in LLM Activations (LLM 激活值中的离群特征)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Outlier Features（离群值特征/异常值通道）是指 LLM 中间激活张量 X ∈ R^{T×d} 中某些特定通道（列）的数值幅度系统性地远大于其他通道的现象——某些 channel index c 在所有 token 上的 |X_{t,c}| 比其他通道大 20-100 倍（Wei et al., 2022 首次系统记录）。离群值特征集中在约 0.1% 的通道中，但其存在使得激活值的对称均匀量化几乎不可能：离群通道的 scale 由最大值决定（s_x = max(|X|)/q_max），导致非离群通道的有效量化精度极低。离群值特征的出现与 LayerNorm/RMSNorm 机制、Transformer 注意力残差结构、以及训练时的优化动态有关——RMSNorm 将每个 token 的向量除以其 L2 范数，将总能量分散，但特定方向（通道）仍保持极大的权重在内积中。从信号处理角度看，离群值特征可理解为模型的"强响应通道"——某些方向的语义信息（如 token identity、位置）高度集中。QuaRot 通过随机 Hadamard 变换从根源消除离群值：X → XQ，其中 Q = H_d diag(s)，每个输出通道变为所有输入通道的 ±1 加权和，大值被扩散到所有方向（图 1 验证变换后激活值从长尾分布变为类高斯分布）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
离群通道识别的典型伪代码（Atom/SmoothQuant 的 baseline 方法）：
```
# X ∈ R^{T×d}: 一批激活值
# 计算每个通道的最大绝对值（跨所有token）
channel_max = max(|X|, dim=0)       # [d]
# 计算所有通道最大值的中位数
threshold = 5.0 * median(channel_max)  # 或 6.0，超参数
# 识别离群通道
outlier_mask = (channel_max > threshold)  # [d], bool
# 离群通道保持FP16，其余INT4量化
X_normal = X[:, ~outlier_mask]   # 量化到INT4
X_outlier = X[:, outlier_mask]    # 保持FP16
```

QuaRot 的离群值消除方法（根本不需要识别）：
```
# 离线：计算随机Hadamard Q = H_d diag(s), s_i∈{±1}
# 融入权重：W_gate ← Q^T W_gate, W_down ← W_down Q, 等
# 在线：X_rotated = X @ Q  # 但无需显式计算，因为已融入权重
#       X_rotated 的所有通道幅值均匀（图1右）
#       直接用 per-token symmetric INT4 quant: s_x=max(|X_rotated|)/7
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
传统方法处理离群值特征的三种策略：(1) 混合精度（Atom, LLM.int8()）——离群通道保持 FP16/INT8，非离群通道量化；(2) Per-channel scaling（SmoothQuant）——将量化难度通过等价变换从激活值迁移到权重；(3) 校准集搜索（OmniQuant）——用可学习参数优化 per-channel 变换。QuaRot 用 Hadamard 旋转首度从根源消除离群值，使策略 (1)-(3) 中复杂机制不再必要。局限性：当存在大量"massive outliers"的 "pivot tokens" 时（如 Sun et al. 2024 发现某些 token 在整个序列中都产生极高激活值），全局 Hadamard 变换的效果可能不完全。当模型维度非 2 的幂时，需要使用已知 Hadamard 矩阵的 Kronecker 分解（H_d = H_{2^n} ⊗ H_m）。

涉及论文标题：
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

## Hierarchical Latent Caching (HLC) for Diffusion Transformers

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Latent Caching (HLC) 是 QuantCache 论文提出的针对 Diffusion Transformers (DiTs) 视频生成的训练无关（training-free）自适应特征缓存机制。其核心思想：不同于传统缓存方法（如 DeepCache、AdaCache、Δ-Cache）使用静态的固定缓存间隔，HLC 基于 inter-step feature divergence（跨步特征散度）动态决定每个 layer 在每个 timestep 是否需要刷新缓存。具体地，对 timestep t 和 layer l，计算 timestep-wise feature divergence score：D_t^(l) = ||p_t^(l) - p_{t-k}^(l)||_1 / k · ||∇_t m_t^(l)||，其中 p_t^(l) 为 layer l 在 timestep t 的激活值，k 为距离上次缓存刷新的步数，∇_t m_t^(l) 为帧间 feature map 的运动梯度。D_t^(l) 综合衡量了：(1) 激活值的变化幅度（L1 distance），(2) 帧间运动变化速度（inter-frame gradient）。基于 D_t^(l) 与预设阈值 δ_1、δ_2 的比较，HLC 决定三档缓存刷新间隔：τ_t^(l) = τ_max（D_t^(l) < δ_1，内容变化极小，长间隔缓存）、τ_mid（δ_1 ≤ D_t^(l) < δ_2，中等变化）、τ_min（D_t^(l) ≥ δ_2，剧烈变化，频繁刷新）。HLC 专门针对 DiT 架构设计——DiT 缺乏 U-Net 的 skip connections，传统 feature map caching 在 DiT 上效果差。HLC 在 QuantCache 中单独实现 4.12× speedup（on Open-Sora, A800-80GB）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HLC 在 DiT 推理 pipeline 中的运行逻辑：
```python
# HLC: Hierarchical Latent Caching for DiT
# 每个 timestep t 和每个 DiT layer l
k = last_cached_step[l]  # 上次缓存刷新的 timestep
if t - k < tau[l]:        # 缓存未过期
    p_t[l] = cache_buffer[l]  # 直接复用缓存特征
    skip_computation(l)        # 跳过该层完整计算
else:
    p_t[l] = compute_layer(l, x_t)  # 正常前向计算
    D_t_l = norm(p_t[l] - p_k[l], 1) / (t - k) * norm(grad_m_t[l])
    # 三档决策更新刷新间隔
    if D_t_l < delta_1:
        tau[l] = tau_max     # 长间隔 (如 5-10 steps)
    elif D_t_l < delta_2:
        tau[l] = tau_mid     # 中等间隔 (如 2-3 steps)
    else:
        tau[l] = tau_min     # 短间隔 (每步刷新)
    cache_buffer[l] = p_t[l]
    last_cached_step[l] = t
```
HLC 与 AIGQ 联合优化：小 skip（低 τ_t^(l)）时用较小的 bit-width 利用高冗余性加速；大 skip（高 τ_t^(l)）后在 post-skip step 应用较小 bit-width 增强精度补偿缓存 drift。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HLC 实现不需要额外训练或微调，完全在推理时在线运行：(1) 在 GPU 上分配 dedicated cache buffer 存储每层缓存的 feature map（FP16/量化格式）；(2) 每个 timestep 推理前先计算各层的 D_t^(l)，与阈值比较决定是否从 cache buffer 读取或重新计算；(3) 阈值 δ_1、δ_2 和 τ_max/τ_mid/τ_min 为超参数，论文通过经验实验确定。HLC 适用于所有 DiT-based 视频/图像生成模型（如 Open-Sora、Flux、CogVideoX），尤其对长时序视频（如 64+ 帧）加速效果显著。当前开源代码见 https://github.com/JunyiWuCode/QuantCache。

涉及论文标题：
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---

## Adaptive Importance-Guided Quantization (AIGQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Importance-Guided Quantization (AIGQ) 是 QuantCache 论文提出的自适应混合精度量化框架，针对 Diffusion Transformers (DiTs) 视频生成的两阶段动态精度分配：(1) **权重量化（Weight Quantization）**：offline 评估每层的 sensitivity（综合考虑 numerical error、perceptual distortion 和 temporal dynamics），在总 bit-width 预算 B_total 约束下迭代分配每层 bit-width（Σ_l B(l) ≤ B_total）——关键层（高 sensitivity，对纹理重建和运动连续性影响大）分配高精度，冗余层分配低精度。同时引入 channel-balancing mechanism：scaling（修正 pretrained scale shift tables 的静态 imbalance）+ rotation（修正 timestep embeddings 引起的动态变化），确保更均匀的跨 channel 数据分布以减少量化 outlier。(2) **激活量化（Activation Quantization）**：提出 timestep-wise content-adaptive bit allocation function：bit-width(t) = Bit_max（D_t < θ_1，低冗余，关键 timestep）、Bit_mid（θ_1 ≤ D_t < θ_2）、Bit_min（D_t ≥ θ_2，高冗余 timestep，如连续帧变化极小），其中 D_t 为 timestep 冗余度度量（从相邻 feature map 距离推导）。核心洞见：不是所有 timestep 对输出质量贡献相同——早期/高冗余 intermediate step 用低精度即可，关键过渡阶段（细节涌现/场景切换）需要高精度。AIGQ 联合 HLC 实现 6.33× speedup。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AIGQ 的完整 pipeline 流程：
```python
# Offline Phase: Weight Sensitivity Analysis & Budget Allocation
for layer in model.layers:
    # 用小规模校准集评估每层 sensitivity
    layer_sensitivity[l] = compute_sensitivity(
        numerical_error(W_quant[l], W_fp16[l]),
        perceptual_distortion(output_quant[l], output_fp16[l]),
        temporal_variance(feature_map[l])
    )
# 在总预算 B_total 下迭代分配 bit-width
B_remaining = B_total
sorted_layers = sort_by_sensitivity_desc(layers)
for layer in sorted_layers:
    if B_remaining >= 8:
        bit_width_W[l] = 8  # 高 sensitivity → 8-bit
    elif B_remaining >= 6:
        bit_width_W[l] = 6  # 中 sensitivity → 6-bit
    else:
        bit_width_W[l] = 4  # 低 sensitivity → 4-bit
    B_remaining -= bit_width_W[l]

# Channel Balancing
for layer in model.layers:
    S = compute_scaling_correction(scale_shift_tables[l])  # 静态修正
    R = compute_rotation_correction(timestep_embeddings[l])  # 动态修正
    # offline absorb scaling into preceding weights: W_prev = W_prev * S

# Online Phase: Timestep-wise Activation Bit Allocation
for t in range(T, 0, -1):
    D_t = compute_timestep_redundancy(x_t, x_{t-1})  # 相邻 feature map 距离
    if D_t < theta_1:      bit_width_A = Bit_max   # 低冗余，保持高精度
    elif D_t < theta_2:    bit_width_A = Bit_mid   # 中等冗余
    else:                  bit_width_A = Bit_min   # 高冗余，激进量化
    # 推理时动态量化激活 + 低精度 GEMM
    x_quant = uniform_quantize(x_t, bit_width_A, per_layer=True)
    for layer in model.layers:
        w_quant = load_quantized_weight(layer, bit_width_W[l])
        output = low_precision_gemm(w_quant, x_quant)  # 如 W4A6 GEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AIGQ 基于 uniform min-max quantization（per-channel weight + dynamic per-layer activation），激活量化参数 online 计算（minimal overhead），混合精度权重量化 offline 通过小规模校准数据集确定。具体实现时：(1) 将 channel-balancing scaling factors offline 吸收到前层权重中（受 QServe/SmoothQuant/ViDiT-Q 启发），消除推理时额外开销；(2) rotation transformation 通过 CUDA kernel fusion 与量化操作融合；(3) bit-width 分配阈值 θ_1、θ_2 和 B_total 为超参数，论文通过经验实验确定。AIGQ 在 Open-Sora 上 W4A6 配置下仍保持 competitive quality（VBench 指标接近 FP16 baseline），显著优于同 bit-width 下的 Q-DiT、PTQ4DiT、SmoothQuant 等 uniform quantization 方法。开源代码：https://github.com/JunyiWuCode/QuantCache。

涉及论文标题：
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---

## Structural Redundancy-Aware Pruning (SRAP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structural Redundancy-Aware Pruning (SRAP) 是 QuantCache 论文提出的在线（runtime）层剪枝机制，针对 Diffusion Transformers (DiTs) 在单个 timestep 内的结构冗余进行自适应剪枝。核心观察：DiT 的某些层在同一个 timestep 内表现出显著的 representational overlap（表示重叠），意味着某些层的计算可以被剪枝而不损失信息。SRAP 包含两个关键组件：(1) **Layer-wise Cosine Similarity Pruning**：在 timestep t 内计算相邻层 l 和 l+1 的 feature cosine similarity：S_t^(l,l+1) = ⟨p_t^(l), p_t^(l+1)⟩ / (||p_t^(l)|| · ||p_t^(l+1)||)。当 S_t^(l,l+1) > τ_high → 完全跳过 layer l+1（P_prune=1）；当 τ_low ≤ S ≤ τ_high → 以概率 P_base 随机剪枝；当 S < τ_low → 不剪枝（P_prune=0）。(2) **Adaptive Temporal Pruning Rate**：跟踪跨 timestep 的累积 feature variation V_t = Σ_{i=0}^k ||p_t - p_{t-i}||_1。当 V_t < δ_low（扩散过程处于精细 refine 阶段）→ 增加全局剪枝概率；当 V_t > δ_high（剧烈内容变换阶段）→ 减少剪枝以维持信息流。SRAP 联合 HLC 和 AIGQ 形成三层次计算优化，最终实现 6.72× total speedup。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SRAP 在每个 timestep 内的执行流程：
```python
# SRAP: Structural Redundancy-Aware Pruning (per-timestep)
# 计算全局时序累积变化 V_t，决定当前 timestep 的整体剪枝激进程度
V_t = sum(norm(p_t[l] - p_{t-i}[l], 1) for i in range(k) for l in layers)
if V_t < delta_low:       # 精细 refine 阶段 → 激进剪枝
    global_prune_scale = 1.5
elif V_t > delta_high:    # 剧烈变化阶段 → 保守剪枝
    global_prune_scale = 0.5
else:
    global_prune_scale = 1.0

# 逐层剪枝决策
for l in range(num_layers - 1):
    S = cosine_similarity(p_t[l], p_t[l+1])  # Eq. 9
    if S > tau_high:
        P_prune = 1.0         # 高度冗余 → 必剪
    elif S >= tau_low:
        P_prune = P_base * global_prune_scale  # 中等冗余 → 概率剪枝
    else:
        P_prune = 0.0         # 低冗余 → 不剪
    if random() < P_prune:
        skip_layer(l+1)       # 跳过该层；输出直接复用上一层 feature
        p_t[l+1] = p_t[l]     # feature copy forward
```
SRAP 区别于传统的 static layer pruning（预定义固定子集剪枝）——它在运行时根据实时 feature similarity 动态决策，使剪枝行为与当前生成内容的自适应匹配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SRAP 完全在推理时在线执行，无需训练或校准：(1) 在每个 timestep 内对每对相邻层计算 cosine similarity（基于 FP16/量化后的 feature）；(2) 通过预设阈值 τ_high、τ_low 和 P_base 进行决策；(3) 剪枝的 overhead 极小（仅 cosine similarity 计算 + 随机数），远小于被跳过的 full layer computation。SRAP 最适用于 DiT-based 视频生成模型，因为这些模型的相邻层在 denoising 中后期（高相似度）表现出显著冗余。SRAP 有效补充了 HLC（跨 timestep 缓存）和 AIGQ（精度降维），形成时间-层-精度三维联合优化。开源实现见 https://github.com/JunyiWuCode/QuantCache。

涉及论文标题：
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---

## Pre-Quantization Fine-tuning with Fused Rectifier (PQFR)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pre-Quantization Fine-tuning with Fused Rectifier (PQFR) 是 SPR²Q 提出的量化预处理机制，其核心思想是在量化前将可训练的低秩 rectifier 模块（ΔW = BA）的权重增量融入骨干网络，使得模型在量化前主动吸收补偿信息。与普通 PTQ 仅优化量化器参数不同，PQFR 修改权重使其对量化更鲁棒。具体流程：(1) 对每个需要量化的权重矩阵 W，引入低秩增量 ΔW = BA（A∈ℝ^{r×d_in}, B∈ℝ^{d_out×r}, r=8）；(2) 前向计算 W' = W + ΔW，然后伪量化 W_q' = Q_{a,b}(W')；(3) 使用 STE 反向传播，联合优化 rectifier 参数 (A,B) 和量化器裁剪界 (a,b)；(4) 损失函数为 L_pixel + λ·L_feature（像素级 L1 重建 + 逐块特征 L2 对齐）；(5) 训练完成后 fusion：W_final ← W + ΔW。与 QLoRA 区别：QLoRA 将 LoRA 适配器添加到量化后的模型用于下游任务微调，PQFR 将 rectifier 融入量化前的模型以改善量化质量。与 PTQ1.61 的 Restorative LoRA 类似但 PQFR 扩展为 rectifier group + 静态路由。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PQFR: 对单个 Mamba 模块的量化预处理
W = module.weight  # 冻结的原始 FP32 权重
A = nn.Parameter(torch.randn(r, d_in))  # 可训练低秩矩阵
B = nn.Parameter(torch.randn(d_out, r))
a, b = nn.Parameter(l_init), nn.Parameter(u_init)  # 可训练裁剪界

for iter in range(12000):  # Adam, lr=1e-2, Cosine Annealing
    X = get_batch(DF2K)
    delta_W = B @ A  # 低秩增量
    W_prime = W + delta_W  # 补偿后的权重
    
    W_clipped = clamp(W_prime, a, b)
    s = (b - a) / (2^4 - 1)  # 4-bit 量化步长
    W_q = round((W_clipped - a) / s) * s + a  # 量化-反量化
    
    Y_q = X @ W_q  # 量化模型前向
    
    L_pixel = L1(Y_q, Y_fp)  # 像素级重建
    L_feature = sum(L2(φ_l(Y_q), φ_l(Y_fp)) for l in all_blocks)
    loss = L_pixel + λ * L_feature
    
    loss.backward()  # STE 近似 round 导数
    update(A, B, a, b)
```
仅使用 PQFR（无 RGT/OSRC）：Set5 PSNR 从 37.20 提升至 37.44 (+0.24 dB)，Urban100 从 30.69 提升至 31.25 (+0.56 dB)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 PaddlePaddle 框架实现，训练 12,000 iterations (batch=8)，优化器 Adam (lr=1e-2, Cosine Annealing)，rectifier rank r=8。训练完成后 rectifier 被融合到 FP32 权重中——无额外推理参数、无额外 FLOPs、无动态结构修改。

涉及论文标题：
- SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization (Restorative LoRA 同类预处理范式)

---

## Static Priority-Based Rectifier Routing (SPR²)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Static Priority-Based Rectifier Routing (SPR²) 是 SPR²Q 提出的静态 rectifier 路由机制。与 MoE 动态路由（推理时根据输入选择不同 expert，引入额外计算）不同，SPR² 将路由决策完全离线化：训练阶段用动态门控鼓励 rectifier group 专业化学习多种补偿策略，校准阶段通过梯度下降学习最优静态门控权重并预计算增量存入 SPR²Q Table，推理时直接查表融合——无动态门控、无额外 FLOPs。三阶段：(1) RGT（Rectifier Group Training）：动态门控 g_i 加权聚合 N 个 rectifier，12K iterations 训练；(2) OSRC（Offline Static Routing Calibration）：冻结参数后梯度下降 500 iterations 学习 ĉ = argmin_g L，构建 SPR²Q Table；(3) 推理：W_final = W + Table[l]，直接量化推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 阶段 1: RGT (12K iters)
N = 4  # rectifier group size
rectifiers = [(A_i, B_i) for i in range(N)]  # rank=8
gate = GatingNetwork()

for iter in range(12000):
    g = softmax(gate(X))  # 动态路由权重
    delta_W_fused = sum(g[i] * (B_i @ A_i) for i in range(N))
    W_q = Quantize(W + delta_W_fused)
    loss = L_pixel + λ * L_feature
    # STE 更新所有参数

# 阶段 2: OSRC (500 iters)
for each module l:
    g_hat = argmin_g L(W + Σ g_i * ΔW_i)  # Eq.12, 冻结 ΔW_i
    SPR2Q_Table[l] = Σ g_hat[i] * (B_i @ A_i)  # 预计算最优增量

# 阶段 3: 推理
for each module l:
    W_final = W + SPR2Q_Table[l]  # 离线融合
    W_q = Quantize(W_final)
    Y = X @ W_q  # 与原始模型计算图一致
```
消融：PQFR +0.24, +RGT +0.16, +OSRC +0.12 dB (Set5)。关键设计：N=4 够用（N=8 仅 +0.10 dB 边际增益），r=8 为 rank 饱和点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 PaddlePaddle + RTX 4090 实现。rectifier rank r=8, group size N=4。训练 12K+500 iterations (batch=8)。推理阶段：所有 rectifier 参数离线融合，模型结构与原始 MambaIRv2-light 完全一致，推理时无需任何额外的 gate 计算或动态路由分支判断。

涉及论文标题：
- SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

---

## Block-wise Feature Alignment Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Feature Alignment Loss 是 SPR²Q 提出的细粒度特征对齐损失：L_feature = E_x [Σ_{l=1}^{L} ||φ_l(f_q(x)) - φ_l(f_FP(x))||_2²]，其中 φ_l(·) 是第 l 个计算 block 提取的中间特征图。不同于仅对最终输出做像素级 L1 损失的粗粒度监督，此损失在网络的每一层 block 输出级施加对齐约束，确保量化误差在每个 block 被局部补偿而非累积到输出端。属于 self-distillation 范畴（量化模型从自身 FP 版本学习），与知识蒸馏中的教师-学生 feature distillation 的区别在于对等 block 之间的对齐。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
L_feature = 0
for l in range(L):  # 遍历所有 L 个 Mamba block
    phi_q = intermediate_feature(quantized_model, X, block=l)
    phi_fp = intermediate_feature(fp_model, X, block=l)
    L_feature += ||phi_q - phi_fp||_2^2

L_pixel = ||f_q(X) - f_fp(X)||_1  # 像素级 L1
L_total = L_pixel + λ * L_feature  # Eq.6
```
SPR²Q 的总损失 = L_pixel + λ * L_feature，λ 为平衡超参数。此设计使模型同时关注全局 reconstruction fidelity (L_pixel) 和各 block 的局部 consistency (L_feature)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过在模型各 block 输出处注册前向 hook 提取中间特征。适用于 pixel-level fidelity 敏感任务（图像/视频超分辨率、图像恢复），这些任务中量化引起的中间层误差会逐步累积并在输出层表现为纹理模糊和细节丢失。SPR²Q 的消融验证仅使用 L_pixel 显著少于 L_pixel + L_feature。

涉及论文标题：
- SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

## Smooth K (K矩阵平滑 / Attention Channel-wise Outlier Smoothing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Smooth K 是 SageAttention 提出的针对 Attention 中 K 矩阵 Channel-wise Outlier 的平滑方法。核心观察：K 矩阵存在 channel-wise large bias——每个 token 的 key 实际上是"所有 token 共享的大偏置 + 小的 token-wise signal"的叠加，即 outlier 源于大偏置而非 token 间的大方差。Smooth K 通过减去所有 token 的均值来消除该偏置: `K_smooth = K - mean(K)`，其中 `mean(K) = 1/N * Σ_{t=1}^{N} K[t,:]`，形状 1×d。该变换的数学关键性质：它不改变 attention score P，因为对于任意 query q，`σ(q(K - mean(K))^⊤) = σ(qK^⊤ - q·mean(K)^⊤) = σ(qK^⊤)`（softmax 对常数偏移具有不变性）。实验表明 smooth K 使 INT8 per-token 量化的 Cosine Similarity 从 62.24% 提升至 99.47%，且 speed overhead <0.2%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: K ∈ R^{N×d} (FP16)
K_mean = sum(K, dim=0) / N        # [d], 沿token维求均值
K_smooth = K - K_mean             # [N×d], 减去均值消除channel bias
δ_K, K̂_INT8 = ψ_K(K_smooth)       # per-token或per-block INT8量化
S_i^j = Matmul(Q̂_i, K̂_j^T) × δ_Q[i] × δ_K[j]  # INT8 FlashAttention
```
与 SmoothQuant (Xiao et al., 2023) 的区别：SmoothQuant 将量化难度从激活迁移到权重处理 linear layer activation outlier，但 attention 中无可迁移权重维度且 Q 也受 outlier 影响。Smooth K 利用 attention softmax 特有的常数偏移不变性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 FlashAttention-2 tiling 下，mean(K) 通过两次 pass 完成：累加所有 K tile sum → 计算全局 mean → 逐 tile 减去。除以 N 和逐元素减法为 element-wise 操作，开销 <0.2%。可在 Triton kernel 中融合于 K 加载阶段。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## INT8 Quantization for Attention (INT8注意力量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT8 Quantization for Attention 是 SageAttention 提出的针对 Self-Attention 中 QK^⊤ Matmul 的 INT8 后训练量化。设计选择：(1) INT8 而非 FP8——consumer GPU (RTX4090/3090) 上 INT8 Matmul 是 FP16 的 4×、FP8 的 2×，且精度更高（INT8 QK cosine sim 99.54% vs E4M3 92.83% vs E5M2 77.95%）；(2) per-token 或 per-block 粒度——per-channel 不可行（dequant 需 outer axis scale）；(3) 1/√d 融合到 Q 量化中（on-chip fuse）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
δ_Q, Q̂_INT8 = per_block_quantize_int8(Q / sqrt(d))  # b_q=128
δ_K, K̂_INT8 = per_block_quantize_int8(K_smooth)      # b_kv=64
for Q̂_i, K̂_j:
    S_int32 = tl.dot(Q̂_i, K̂_j^T, input_precision='int8')  # Tensor Core u8·u8→s32
    S_ij = S_int32.to(tl.float16) * δ_Q[i] * δ_K[j]        # dequant
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
依赖 NVIDIA Tensor Core INT8 mma(u8.u8.s32)。Triton: `tl.dot(Q̂, K̂^T, input_precision='int8')`。per-block 量化: `scale=max(|x|)/127; x̂=clamp(round(x/scale),-127,127).to(tl.int8)`。Dequant: `S.to(tl.float16) * scales`。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## FP16 Accumulator for PV Matmul (FP16累加器加速PV矩阵乘)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP16 Accumulator for PV Matmul 是 SageAttention 提出的 Attention PV Matmul 加速方案。不对 P,V 做 INT8 量化（最差层 cosine sim 仅 56.40%），保持 P,V 在 FP16 但将 Tensor Core MMA accumulator 从 FP32 降为 FP16。RTX4090/3090 上 mma(f16.f16.f16) 比 mma(f16.f16.f32) 快 2×（Ada Lovelace: FP16 accum 512 FMA/SM/cycle vs FP32 accum 256），且精度与 FP32 accum 一致（cosine sim 差值 0.00%）。关键 insight：P（softmax 输出 ∈[0,1]）和 V 数值范围适合 FP16，attention output O 不需要 FP32 累积精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
O_i^j = diag(e^{m_i^{j-1} - m_i^j}) O_i^{j-1} + Matmul(
    P̃_ij.to(tl.float16), V_j.to(tl.float16), accum=tl.float16  # ← 2x faster
)
```
四个 kernel 变体: SAGEAttn-T/B (QK INT8 + PV FP16 accum)、SAGEAttn-vT/vB (QK INT8 + PV INT8, +4% speed)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Triton: `tl.dot(P̃, V, out_dtype=tl.float16)`。CUDA: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16`。FP16 accum 2× 加速仅在 consumer GPU (RTX4090/3090) 成立——数据中心 GPU (A100/H100) FP32 accum 已是快速路径。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## Adaptive Quantization for Attention (自适应注意力量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Quantization for Attention 是 SageAttention 的 per-layer kernel 选择策略。不同 layer 对 INT8 PV 量化敏感度不同。离线校准：(1) 对每层测试 SAGEAttn-vB 的 cosine similarity；(2) >99.8%（SAGEAttn-B 最差 cosine sim）→ 用 vB（全 INT8, +4% speed）；(3) 否则回退 SAGEAttn-B（FP16 PV）。adaptive 策略比纯 SAGEAttn-T 提升 11.7% OPS，零指标损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线校准
kernel_map = {}
for layer_id in range(num_layers):
    cos_sim = measure(SAGEAttn-vB, calibration_data[layer_id])
    kernel_map[layer_id] = 'vB' if cos_sim > 0.998 else 'B'
# 在线推理: O = sage_attention(Q,K,V, kernel=kernel_map[layer_id])
```
与 LLM.int8() 混合精度不同：adaptive 在 kernel 粒度做选择（整层统一），避免 kernel 内条件分支。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
离线校准一次完成，推理时仅常数时间 kernel dispatch。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## Online Softmax (在线Softmax / Streaming Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Online Softmax 是允许不一次性访问全部输入即可精确计算 softmax 的增量算法，源于 Milakov & Gimelshein (2018)，由 FlashAttention 引入深度学习 attention。核心递推: `m_new = max(m_old, rowmax(S_i))`, `l_new = e^{m_old-m_new}*l_old + rowsum(exp(S_i-m_new))`, `O_new = diag(e^{m_old-m_new})*O_old + exp(S_i-m_new)@V_i`。数学正确性: `exp(S_old-m_new) = exp(S_old-m_old)*exp(m_old-m_new)`。SageAttention 沿用 FlashAttention-2 online softmax，S 计算用 INT8 Matmul，其余保持 FP16。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for i in num_Q_tiles:
    m = -inf, l = 0, O = 0
    for j in num_KV_tiles:
        S = Q̂_i @ K̂_j^T × scales   # INT8 + dequant → FP16
        m_new = max(m, rowmax(S)); P̃ = exp(S - m_new)
        l = exp(m - m_new)*l + rowsum(P̃)
        O = diag(exp(m - m_new))@O + P̃@V_j; m = m_new
    O = diag(1/l) @ O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA/Triton 中通过 register 维护 running state (m,l,O)。FlashAttention-2 中 l,m 以 log-space 维护避免 overflow。SageAttention 所有中间计算保持 FP16。主流 GPU attention kernel（FlashAttention-2/3, FlashInfer, xformers, SageAttention）均使用 online softmax。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## Channel-wise Outlier in Attention Matrices (注意力矩阵的通道异常值)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Channel-wise Outlier in Attention Matrices 是 Attention 中 K 矩阵特定 channel 数值远大于其他 channel 的现象。SageAttention 通过可视化 Unidiffuser/CogvideoX 的 Q,K,V 分布发现：K 存在显著 channel-wise large bias（各 token 一致），V 较轻，Q 相对均匀。根因可能与 transformer attention sink / no-op head 机制相关——某些 head 学会将特定 K 维度推向大值。对 INT8 量化影响：outlier channel 主导 per-token scale，非 outlier channel 信号被压入极小区间被噪声淹没。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# K[:,c] ≈ 100 (outlier), K[:,其他] ∈ [-1,1]
# per-token INT8: δ_K = max(|K[t,:]|)/127 ≈ 100/127 ≈ 0.787
# → 正常 signal (e.g. 0.5) 量化误差 ≈ round(0.5/0.787)-0.5/0.787 ≈ 0.635 >> signal
# → QK^T 精度崩溃
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
处理方法：(1) SageAttention Smooth K——利用 softmax 常数偏移不变性，零精度损失、<0.2% overhead；(2) SmoothQuant per-channel scaling——attention 中不适用（无对应权重维度）；(3) LLM.int8() 混合精度——可行但降低效率。SageAttention 的关键贡献是发现 attention softmax 的特有不变性实现零代价 outlier 平滑。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## PCA-Based Low-Rank Subspace for Mixed-Precision Quantization（基于PCA的低秩子空间混合精度量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PCA-Based Low-Rank Subspace for Mixed-Precision Quantization 是 ResQ (Saxena et al.) 提出的核心量化方法。其关键思想是：不依赖启发式 outlier 检测（如 l_∞-norm 选高精度通道），而是通过 PCA 从理论上找到最小化量化误差的投影基。具体流程：(1) 对校准激活 X ∈ R^{n×d} 计算协方差矩阵 XX^T 的特征分解，取前 r 个最大特征值对应的特征向量构成低秩子空间 P_h ∈ R^{d×r}（高精度），前 d-r 个最小特征值对应的特征向量构成互补子空间 P_l ∈ R^{d×(d-r)}（低精度）；(2) 在每个子空间内应用随机正交旋转 R_h/R_l 抑制 outliers；(3) 最终投影基 U = PR = [P_l P_h]·diag(R_l, R_h)。Theorem 4.2 从理论上证明 PCA 基选择是最优的——它最小化了上界 E||X - X_q||_F ≤ α·E||X||_F - β·E||XP_h||_F 中的 E||XP_h||_F 项，即通过最大化高精度子空间投影的范数来降低量化误差上界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ResQ 的 PCA 低秩子空间混合精度量化 pipeline（以 Meta-Llama-3-8B, W/A/KV=4/4/4, r=d/8 为例）：

```
// ===== 离线校准阶段 =====
// Step 1: 采集校准激活
calib_samples = sample_wikitext(512)  // 512 random samples
for each decoder block in model:
    X_attn = collect_attn_activations(block, calib_samples)  // [tokens, d_h]
    X_ffn  = collect_ffn_activations(block, calib_samples)

// Step 2: PCA 获取投影基 P
C = X_attn @ X_attn.T                           // 协方差矩阵 [d_h, d_h]
eigenvalues, eigenvectors = eigh(C)              // 按升序排列
P_l = eigenvectors[:, :d-r]                      // 前 d-r 列=低精度子空间
P_h = eigenvectors[:, d-r:]                      // 后 r 列=高精度子空间（最大特征值）

// Step 3: 生成随机旋转 + 构造 U
R_h = random_orthogonal_matrix(r)                // Hadamard 或随机正交
R_l = random_orthogonal_matrix(d-r)
U_h = P_h @ R_h                                  // [d_h, r] 高精度投影
U_l = P_l @ R_l                                  // [d_h, d-r] 低精度投影

// Step 4: 融合投影到权重 + GPTQ 优化
W_o_proj' = W_o_proj @ U_A         // U_A 融合到输出投影（右乘）
W_q_proj' = U_A.T @ W_q_proj       // U_A^T 融合到输入投影（左乘）
W_down'   = W_down @ U_D           // U_D = Hadamard 融合到 FFN down_proj
// GPTQ 对融合后权重做进一步优化

// ===== 在线推理阶段 =====
// 激活自动投影（U_A 已融入前一层的输出投影权重，零额外开销）
X_proj = X  // U_A 投影已由前一层 W_o_proj' = W_o_proj @ U_A 完成

// 注意力块内（需要运行时投影的部分）
X_qk = X @ U_C                    // RoPE 前投影（U_C 在线，8-bit 量化）
K_proj = RoPE(X_qk)               // 投影后的 key
Q_proj = RoPE(X_qk)               // 投影后的 query（同一 U_C）
X_v = X @ U_B                     // value 投影（U_B 在线或融合到 W_v）
// 注意力计算不变：Q_proj @ K_proj^T = Q @ K^T

// GEMM: INT4 低精度 + INT8 高精度
Y_low  = INT4_GEMM(Q_L(X @ U_l), Q_L(U_l^T @ W))   // 4-bit 分支
Y_high = INT8_GEMM(Q_H(X @ U_h), Q_H(U_h^T @ W))   // 8-bit 分支
Y = Y_low + Y_high                                   // 交叉项因正交性消失

// FFN: U_D Hadamard 在线变换
X_ffn = Hadamard(X_ffn_in)          // 快速 Hadamard 变换 O(d log d)
down_in = SiLU(gate) * up
Y_ffn = INT4_GEMM(Q_4(down_in), Q_4(W_down))  // down_proj 统一 4-bit
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ResQ 开源：https://github.com/utkarsh-dmx/project-resq。基于 HuggingFace Transformers + PyTorch + CUDA 11.8 + CUTLASS。实现要点：(1) PCA 使用 PyTorch `torch.linalg.eigh()` 对小批量校准数据收集的激活矩阵做特征分解；(2) 四种投影矩阵（U_A/U_B/U_C/U_D）分别处理 block 边界、注意力 KV、FFN 内部投影，r 默认为 d_h/8；(3) U_A 跨所有层共享，U_B/U_C/U_D 逐层独立；(4) U_B 通过后乘 W_v 和左乘 W_o 融入权重链；(5) U_C 因 RoPE 存在需运行时计算但量化为 8-bit；(6) U_D 用 Hadamard 矩阵实现 O(d log d) 快速变换；(7) 投影后权重用 GPTQ 做 Hessian 引导的最优舍入。Meta-Llama-3-8B 完整量化流程在单张 A100 上耗时 35 分钟（表 6）。调节 rank r 可实现 Pareto 最优的精度-效率权衡（图 6a）。

涉及论文标题：
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

---

## Transition Rate (TR) Scheduling（转换率调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transition Rate（TR，转换率）调度是 Lee et al. 提出的专用于 QAT 的训练调度技术。核心动机：传统 QAT 使用学习率（LR）调度来间接控制量化权重的更新幅度，但量化权重仅在潜权重（latent weights）越过 quantizer 的 transition point 时才改变离散级别，其 effective step size 与 LR 相关性弱，导致后期训练不稳定。TR 定义为每次迭代中发生离散级别变化的量化权重占总权重的比例：`k^t = Σᵢ I[w_d^t(i) ≠ w_d^{t-1}(i)] / N`，其中 `w_d` 为离散权重（round/sign 函数输出）。TR 调度通过设定目标 TR `R^t`（通常用 cosine scheduler 从初始值 `λ√b_w` 衰减到零），然后用 TALR 自适应调整潜权重更新步长，使得实际 running TR `K^t`（EMA 平滑后）跟踪目标 TR。这实现了对量化权重的"粗到细"控制——初期高 TR 允许充分探索，后期 TR 趋近零保证收敛稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TR 调度在 QAT 中的每步迭代（来自论文 Algorithm 1）：
```
输入: 总迭代数 T, 目标 TR R^t, momentum m=0.99
输出: 训练后的量化模型

for t in 1..T:
    # Step 1: 前向传播 — 潜权重 w 经 quantizer 转为量化权重 w_q
    w_n = clip(γ·w/s, α, β)          # 归一化 (Eq.1)
    w_d = round(w_n)                   # 离散化 (Eq.2)
    w_q = w_d/γ                        # 定标反归一化 (Eq.3)
    # 用 w_q 计算前向输出和 loss
    
    # Step 2: 反向传播 — STE 通过 discretizer 回传梯度
    计算 gradient term g^t（取决于优化器类型）
    
    # Step 3: 计算当前 TR (Eq.5)
    k^t = count(w_d^t != w_d^{t-1}) / N  # 发生变化的量化权重占比
    
    # Step 4: 估计 running TR (Eq.10)
    K^t = m·K^{t-1} + (1-m)·k^t        # EMA 平滑
    
    # Step 5: 调整 TALR (Eq.11)
    U^t = max(0, U^{t-1} + η(R^t - K^t))  # 加法反馈更新，η=U^0
    
    # Step 6: 更新潜权重 (Eq.12)
    w^{t+1} = w^t - U^t·g^t            # 用 TALR 代替 LR
```
关键设计：初始 target TR = λ√b_w（λ=TR factor, b_w=量化位宽），位宽越大 transition points 越多，需更高初始 TR。TR momentum m=0.99 平滑单步噪声。TALR 更新因子 η = U^0（初始 TALR），使调整步长与初始值成比例。TR 调度仅用于量化潜权重；未量化参数（第一层/最后一层）仍用传统 LR。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) TR 使用离散权重 `w_d`（integer）而非量化权重 `w_q` 来计数 transition——因为 `w_q = δ^t·w_d` 中 scaling factor δ^t 可能变化，使用 `w_q` 会误检 transition；(2) 当 TR 调度启用时，weight quantizer 的 scale parameter s 固定不变——否则 s 变化也会触发 transition，干扰 TR 控制；(3) 支持的 optimizer 类型：SGD、Adam、AdamW、NAdam、Adamax、RMSProp、Adagrad（Table S5 验证），仅需将 g^t 的计算替换为对应优化器规则；(4) 训练时间仅增加约 2%（Table S7）；(5) 在极低位宽（2-bit/binary）和轻量模型（MobileNetV2）上增益更显著（+6.7% 精度）；(6) TR factor λ 在 {5e-3, 1e-3, 5e-4, 1e-4} 中搜索，λ∈[4e-3, 6e-3] 对 ResNet-20 W2A2 均优于 LR baseline。开源：https://cvlab.yonsei.ac.kr/projects/TRS/

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---

## Transition-Adaptive Learning Rate (TALR)（转换自适应学习率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transition-Adaptive Learning Rate（TALR，转换自适应学习率）是 Lee et al. 在 TR 调度中提出的自适应学习率机制，替换传统 QAT 中手动调度的 LR。核心公式：`U^t = max(0, U^{t-1} + η(R^t - K^t))`，其中 R^t 是目标 TR，K^t 是当前 running TR，η 控制更新幅度（设为初始 TALR 值）。这是一个负反馈控制回路：当 K^t < R^t（实际 transition 不足），U^t 增大，促使更多潜权重越过 transition point；当 K^t > R^t（transition 过多），U^t 减小，抑制 transition。与传统 LR 的关键区别：TALR 不是手动预设的 schedule，而是实时反馈控制——当潜权重向 transition point 聚集、transitions 自然容易发生时，K^t 自然升高，TALR 自动降低以抑制振荡。这使 TALR 隐式地考虑了潜权重的分布状态，实现自适应优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TALR 的三种更新方案（论文 Sec. S3.2）：
```
方案1 — 加法更新 (Eq.11，论文默认):
U^t = max(0, U^{t-1} + η(R^t - K^t))
特点: 类似 SGD 权重更新，稳定可控，对 step scheduler 也鲁棒

方案2 — 乘法更新 (Eq.S1):
U^t = U^{t-1} · (R^t / K^t)
特点: 快速适应但敏感于 outlier，step scheduler 时不稳定（Fig. S5）

方案3 — 动量乘法更新 (Eq.S2):
U^t = m'·U^{t-1} + (1-m')·U^{t-1}·(R^t/K^t)
特点: 稳定性与速度的折中，m'=m=0.99，性能接近加法更新
```
TALR 在训练中的典型轨迹（Fig. 3b/S3b）：
- 初期：K^t << R^t → U^t 快速增大以鼓励 exploration
- 中期：U^t 随 R^t 的 cosine decay 逐渐下降
- 后期（~50K iter of CIFAR）：U^t 急剧下降趋近零——因为潜权重已聚集在 transition point 附近（平均距离很小，Fig. 3d），即使极小步长也触发大量 transition，TALR 自动"骤降"抑制振荡
对比 SGD 的 step-decay LR（Fig. S4b）：LR 在两个 decay 台阶之间固定不变，而 TALR 在每个台阶内部仍在单调递减，体现其自适应性。SGDT 在使用 step scheduler 时精度仅降 0.3-0.6%（ReActNet-18），而 SGD 降 0.7-9.1%（Table S3），验证了 TALR 对 scheduler 类型的鲁棒性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) TALR 初始值 = 同类优化器的初始 LR（SGD→1e-1, Adam→1e-3, AdamW→3e-4）；(2) 仅用于量化潜权重的更新，非量化参数仍用传统 LR；(3) TALR 更新发生在 optimizer.step() 中参数更新之前——先根据 K^t 和 R^t 计算 U^t，再用 U^t·g^t 更新 w；(4) TALR 是所有量化层共享的一个标量值（per-model，非 per-layer）；(5) 计算开销极低：仅 element-wise 比较（Eq.5）+ 标量 EMA（Eq.10）+ 标量更新（Eq.11），训练总时间仅增 2%；(6) TALR 与各种 optimizer（SGD, Adam, AdamW, NAdam, Adamax, RMSProp, Adagrad）和 scheduler（cosine/step/linear）兼容，均带来一致精度提升。注意：TALR 的设计不唯一——只要满足 (a) K^t<R^t 时 U^t 增大、(b) K^t>R^t 时 U^t 减小、(c) K^t=R^t 时 U^t 不变这三个准则即可。

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---

## Latent Weights in QAT（潜权重）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Latent Weights（潜权重）是 QAT 中维护的全精度（FP32）权重参数，作为 quantizer 的输入而非直接参与前向计算。QAT 的三步循环：(1) 前向：潜权重 w → quantizer（normalize → discretize → de-normalize）→ 量化权重 w_q → 计算输出和 loss；(2) 反向：STE 将 ∂L/∂w_q 的梯度近似直通回传到 w；(3) 优化：用梯度更新潜权重 w（而非 w_q）。潜权重的存在使离散量化权重可通过连续优化间接训练。但这也带来核心问题：量化权重 w_q 的 effective step size 不仅取决于 LR，还取决于潜权重是否越过 quantizer 的 transition point（即潜权重的分布）。训练后期潜权重倾向于在 transition point 附近聚集（Fig. 3c），导致即使极小 LR 也能触发大量量化权重 transition，造成训练不稳定——这正是 TR 调度要解决的核心痛点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
潜权重在 QAT 中的完整流转（2-bit 权重量化为例）：
```
初始化: w = pretrained_FP_weights  (FP32, nn.Parameter)

每次迭代 t:
    # 前向: 潜权重 → 归一化 → 离散化 → 反归一化 → 量化权重
    w_n = clip(γ·w/s, α, β)        # 归一化潜权重到 [-2, 1]（α=-2, β=1, γ=2）
    w_d = round(w_n)                # 离散化为整数 {-2, -1, 0, 1}
    w_q = w_d/γ                     # 固定反归一化（γ=2 → w_q∈{-1.0,-0.5,0,0.5}）
    output = conv2d(x, w_q)         # 用量化权重计算

    # 反向: STE 直通梯度
    ∂L/∂w_q →（STE: ∂round/∂w_n=1）→ ∂L/∂w_n →（chain rule through clip）→ ∂L/∂w

    # 优化: 更新潜权重
    w^{t+1} = w^t - U^t·g^t          # TR 调度 (TALR)
    # 或 w^{t+1} = w^t - μ^t·g^t    # 传统 LR 调度
```
关键性质：(1) 潜权重是连续值——LR 对潜权重的 effective step size 控制有效（Fig. 1b 蓝线 vs Fig. 2b 蓝线，均与 LR 相关）；(2) 潜权重的分布决定量化权重的 transition 行为——训练后期潜权重聚集在 transition point 附近（Fig. 2c），即使极小 LR 也触发频繁 transition；(3) TR 调度的 TALR 通过负反馈机制隐式考虑潜权重分布——当潜权重接近 transition point 时，K^t 自然升高，U^t 自动降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
潜权重在 PyTorch 中的实现：每个含 quantizer 的层持有 `self.weight` (nn.Parameter, FP32)，前向传播中 `w_q = self.quantizer(self.weight)` 后才用于 F.conv2d。Optimizer 的 param_groups 注册的是 `self.weight`（潜权重）而非 w_q。TR 调度在 optimizer.step() 中修改潜权重的有效学习率（即 TALR），使 w 的实际更新量由 TR 反馈控制。注意：部分 QAT 方法允许 weight quantizer 的 scale s 也是可学习参数——此时 scale 梯度也流经 STE，但当 TR 调度启用时 s 必须固定，否则 s 变化独立于 w 也能触发 transition，干扰 TR 控制。

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---

## Effective Step Size in Quantized Weights（量化权重有效步长）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Effective Step Size（有效步长）衡量单次参数更新中权重的变化幅度。对于全精度权重，effective step size ≈ |μ·g|（LR × 梯度），与 LR 高度相关，因此调度 LR 可直接控制 coarse-to-fine 优化。但对于 QAT 中的量化权重 w_q，由于其 effective step size `|Δw_q^t| = |w_q^t - w_q^{t-1}|` 具有离散特性：`|Δw_q^t| ≈ δ^t·I[w_d^t ≠ w_d^{t-1}]`（δ^t 为量化级别间距），即要么为零（未发生 transition），要么为 δ^t（若发生 transition）。因此量化权重的 average effective step size 主要由发生 transition 的权重数量（即 TR）而非 LR 决定。这一洞察是 TR 调度技术的基础——用调度 TR 替代调度 LR，以直接控制量化权重的 coarse-to-fine 优化进程。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化权重 effective step size 的推导（Eq.6-9）：
```
给定: w_q^t = δ^t·w_d^t, w_q^{t-1} = δ^{t-1}·w_d^{t-1}
假设: δ^t ≈ δ^{t-1}（单个 update 内 scaling factor 变化极小）
      且单次 transition 仅跨越一个量化级别（通常情况）

多 bit 量化（round function, Eq.7 → Eq.9）:
|Δw_q^t| = |δ^t w_d^t - δ^{t-1} w_d^{t-1}| 
         ≈ δ^t·|w_d^t - w_d^{t-1}|
         = δ^t·I[w_d^t ≠ w_d^{t-1}]   （transition 时跨一级别）

二值量化（sign function, Eq.8）:
|Δw_q^t| = ½|δ^t w_d^t - δ^{t-1} w_d^{t-1}| 
         ≈ ½δ^t·I[w_d^t ≠ w_d^{t-1}]  （w_d∈{-1,1}）

推论: Average effective step size ≈ δ^t·(transitions/N) = δ^t·k^t
```
关键实验证据（Table S6, CIFAR-100, ResNet-20 W2A2）：在训练后期将 final target TR 从 0（无 transition）递增到 1e-3（较多 transition），average effective step size 从 0 增至 5e-4，test accuracy 从 65.61%±0.21 降至 62.12%±0.70，精度标准差从 0.21 增到 0.70。验证了减小后期 transition/effective step size 对 QAT 收敛稳定的关键性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Average effective step size 在论文中用作监控指标而非训练目标：在每个 iteration 计算 `mean(|w_q^t - w_q^{t-1}|)` 并记录到训练曲线。论文通过对比 LR 调度（Fig. 1c 蓝线）和 TR 调度（Fig. 1c 红线）下的曲线验证 TR 调度有效性——TR 调度下 effective step size 平滑衰减到零，LR 调度下噪声大且训练后期不收敛。作为 TR 的变体，论文也讨论了直接调度 effective step size（Eq. S3: k^t = Σ|w_q^t(i)-w_q^{t-1}(i)|/N），但由于不同层的量化权重 scale（δ^t）不同，需逐层搜索初始 target value 的超参，而 TR 调度（使用 w_d 计数）与 scale 无关，更具通用性和易用性。

涉及论文标题：
- Scheduling Weight Transitions for Quantization-Aware Training

---

## Shift-Uniform-Log2 Quantizer (SULQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shift-Uniform-Log2 Quantizer (SULQ) 是 I&S-ViT 提出的专门用于 ViT post-Softmax 激活的量化器，旨在解决标准 log2 量化器的"量化低效"(quantization inefficiency)问题。SULQ 的量化公式为：X_q = UQ(-log₂(X + η), b)，即先在 log2 函数的输入上添加 shift bias η，将激活值平移后做 log2 变换，再对输出应用均匀量化器 (UQ)。反量化过程为：X̄ = 2^{-round(D-UQ(X_q))} - η，其中对去量化输出做 round 确保整数输出，使得推理时可用硬件友好的 bit-shift 操作。相比标准 log2 量化器，SULQ 仅增加一次 round 操作和两次加法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SULQ 在 ViT block 中的量化流程（以 3-bit 为例）：
```
输入: X ∈ R^{N×N} (post-Softmax attention map, 值域 ~[10^{-8}, 1])

# Step 1: 确定 η 参数
η_best = argmin_η MSE(X, D-SULQ(SULQ(X, b, η), b, η))
# 通过 grid search 从候选取使量化误差最小的 η

# Step 2: SULQ 量化 (推理时)
Y = -log2(X + η)          # shift + log2 变换
Y_q = UQ(Y, b)            # 均匀量化到 b-bit 整数 [0, 2^b-1]

# Step 3: SULQ 反量化
Y_fp = D-UQ(Y_q)          # 均匀反量化到浮点
Y_int = round(Y_fp)       # round 确保整数,用于 bit-shift
X_hat = 2^{-Y_int} - η    # 2 的负指数幂 + 反 shift
```
关键性质：(1) SULQ 通过均匀量化器完整覆盖输入域，不会像 log2 量化器那样有大量值被 clamp 到远端；(2) SULQ 保持对接近零区域的细粒度 bit 分配，同时对接近 1 的区域分配稀疏 bit，匹配 post-Softmax 的长尾分布；(3) η 参数可调节量化点的分布模式。I&S-ViT 实验：DeiT-S W3A3 时 SULQ 单独贡献 +17.34% 准确率提升（从 3.36% 到 20.70%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中定义 `sulq_quantize(x, b, eta)` 函数，内部调用 `torch.log2(x + eta)` 后接标准均匀量化。η 在优化前通过 grid search 选取使 MSE 最小的值。推理时 SULQ 通过 bit-shift 操作执行 2^{·} 计算，与 log2 量化器硬件效率相同。I&S-ViT 将 SULQ 专门用于 post-Softmax 激活。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

## Smooth Optimization Strategy (SOS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Smooth Optimization Strategy (SOS) 是 I&S-ViT 提出的 ViT PTQ 三阶段训练策略。设计动机源自不同量化粒度下 loss landscape 分析：(1) channel-wise 权重量化 + layer-wise 激活量化 → 粗糙高 loss landscape，易误导优化进入局部极小值；(2) 全精度权重 + channel-wise 激活量化 → 平滑低 loss landscape，优化稳定。SOS 先利用平滑 landscape 优化，再通过 scale reparameterization 无缝转为 layer-wise 量化保持推理效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三阶段流程：
```
# Stage 1: 全精度权重 + 细粒度激活量化优化
for iter in range(1000):
    # weights: FP32 (不量化)
    # post-LayerNorm activations: channel-wise quantizer
    # other activations: layer-wise quantizer
    L = ||X_l_fp - X_l_hat||_2   # block-wise reconstruction
    optimizer.step()  # Adam, lr=4e-5, cosine decay

# Stage 2: 无损 Scale Reparameterization
# 将 channel-wise 量化转为 layer-wise (权重FP32, 无损)
for each post-LayerNorm activation:
    s̃ = Mean(s); z̃ = Mean(z)
    β̃ = (β + s⊙(z-z̃)) / (s/s̃); γ̃ = γ / (s/s̃)
    W̃ = (s/s̃)⊙W; b̃ = b - (s⊙(z-z̃))·W

# Stage 3: 全量化微调
for iter in range(additional_iters):
    # weights: channel-wise quantizer; all activations: layer-wise
    L = ||X_l_fp - X_l_hat||_2; optimizer.step()
```
SOS 与 BRECQ 的两阶段策略的区别：(1) SOS 先全精度权重+量化激活，BRECQ 先量化权重+全精度激活；(2) SOS 包含 ViT 特有的无损 Scale Reparameterization 转换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SOS 在 PyTorch 中通过三阶段训练循环实现。Adam 优化器，权重 lr=4e-5，cosine 衰减，WD=0。Stage 1 使用 channel-wise 量化器（每个通道独立 scale/zero-point），Stage 2 通过修改 LayerNorm affine 参数和后续层权重实现等价转换，Stage 3 切换为 layer-wise 量化器进行微调恢复权重量化损失。量化参数在初始校准后固定不优化。I&S-ViT 实验：SOS 单独贡献 +41.83% 准确率提升（DeiT-S W3A3，从 3.36% 到 45.19%）。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

## Scale Reparameterization for ViT Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scale Reparameterization 是将 channel-wise 量化器等价转换为 layer-wise 量化器的技术，源自 RepQ-ViT (ICCV 2023)。给定 channel-wise 参数 (s∈R^D, z∈R^D)，计算均值作为 layer-wise 参数 (s̃=Mean(s), z̃=Mean(z))，然后调整 LayerNorm affine 参数和下一层权重/偏置将 per-channel scale 差异"吸收"到网络参数中。公式：r₁=s/s̃, r₂=z-z̃; β̃=(β+s⊙r₂)/r₁, γ̃=γ/r₁; W̃_{:,j}=r₁⊙W_{:,j}, b̃_j=b_j-(s⊙r₂)·W_{:,j}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
给定: post-LayerNorm激活 A∈R^{N×D}, channel-wise quantizer (s,z)

# Step 1: LayerNorm 重参数化
原始: A_i = γ_i*(X_i-μ)/σ+β_i
       A_q_i = round((A_i-z_i)/s_i)   # channel-wise量化
替代: β̃_i = (β_i+s_i*(z_i-z̃))/r1_i   # r1_i=s_i/s̃
       γ̃_i = γ_i/r1_i
       Ã_q = round((Ã_i-z̃)/s̃)          # layer-wise量化(s̃,z̃为标量)
可证: Ã_q ≡ A_q_i (数学等价)

# Step 2: 下一层权重补偿
W̃_{:,j}=r₁⊙W_{:,j}; b̃_j=b_j-(s⊙(z-z̃))·W_{:,j}
可证: Ã_q·W̃+b̃ ≡ A_q·W+b (输出等价)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中遍历所有 post-LayerNorm 层，直接修改 state_dict 中对应的 LayerNorm affine 参数和后续 FC 权重/偏置。I&S-ViT 在 SOS Stage 2 中权重仍为全精度时执行转换，确保无损（lossless transition），与 RepQ-ViT 在量化权重上的有损转换形成关键区别。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

## Quantization Inefficiency (量化低效)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Inefficiency 是 I&S-ViT 命名的问题，指 log2 量化器在量化和 post-Softmax 激活时，代表性范围无法覆盖完整输入域。例如 post-Softmax 激活范围 [1.08e-8, 0.868]，-log₂ 输出 [0, 26]，但 3-bit 仅覆盖 [0, 7]，导致 [8, 26] 段全部 clamp 到 7。大量远离零的值被强制映射到同一远端级别，失去区分度。由于 post-Softmax 中大量接近零的值，此问题严重影响模型精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 3-bit log2量化器
X ∈ [1.08e-8, 0.868], s = 0.868/7 ≈ 0.124
Y = -log2(X/s) ∈ [0, 26]  # 26 >> 7
Y_q = clamp(round(Y), 0, 7)
# [8,26]→7, X_hat=9.69e-4, 所有X<9.69e-4映射到同一值

# SULQ方案
Y = -log2(X+η), η=0.01 → Y∈[0,19]
Y_q = UQ(Y, 3)  # 8级别均匀覆盖[0,19]，无clamp损失
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
检测方法：统计 -log₂(X/s) 的输出范围，若 max > 2^b-1 则存在量化低效。SULQ 通过在 log2 输入前添加 shift bias η 压缩输出范围，再通过均匀量化器替代 clamp 实现完整覆盖。η 通过 grid search 最小化 MSE 来确定。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

## Post-Softmax Activation Quantization in ViTs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-Softmax 激活指 ViT MHSA 中经过 Softmax 的注意力权重矩阵 A∈R^{N×N}。特性：(1) 值域 [0,1] 严格非负；(2) 呈长尾分布——大部分值集中于近零区域，少量值近 1。标准均匀量化器在近零区域量化级别稀疏，导致小注意力值被量化为零，破坏注意力机制的信息传递。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ViT Attention pipeline中的post-Softmax量化位置
Q,K,V = Linear_qkv(LayerNorm(X))
Scores = Q·K^T/√d_k
A = Softmax(Scores)              # post-Softmax激活
A_q = SULQ(A, b)                 # I&S-ViT专用量化器
Output = A_q · V                  # 量化注意力权重×V
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
现有方法：(1) FQ-ViT: log2 量化器，对数变换展平长尾分布；(2) PTQ4ViT: twin uniform quantizer；(3) I&S-ViT: SULQ (shift-uniform-log2)，在 log2 前加 shift bias 后接均匀量化，解决 log2 的量化低效问题。所有 log2 系列量化器推理时通过 bit-shift 执行。I&S-ViT 实验：DeiT-S W3A3 下 SULQ 比 log2 量化器精度高 +3.18%（55.78% vs 52.60% UQ baseline）。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

## Post-LayerNorm Activation Quantization in ViTs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-LayerNorm 激活指 ViT Transformer Block 中 Layer Normalization 后的激活值。关键特性：严重的通道间变异性（severe inter-channel variation）——不同通道的数值分布差异巨大。这种高变异性使粗粒度 layer-wise 量化产生大量化误差。I&S-ViT 首次系统分析了不同量化粒度对 loss landscape 的影响：全精度权重 + channel-wise 激活量化产生平滑低 loss landscape，而 channel-wise 权重量化 + layer-wise 激活量化的 landscape 粗糙且高 loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
I&S-ViT SOS 策略对 post-LayerNorm 激活的处理：
```
# Stage1: channel-wise量化 (每通道独立s_c, z_c)
# → 平滑loss landscape, 稳定优化
# Stage2: Scale Reparameterization → layer-wise
# → 无损转换为高效layer-wise方案
# Stage3: layer-wise量化 + 量化权重微调
# → 恢复权重量化损失
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RepQ-ViT 在离线校准后通过 scale reparameterization 将 channel-wise 量化器转为 layer-wise。I&S-ViT 扩展此方法到优化-based PTQ，在 SOS Stage1 利用 channel-wise 平滑 landscape 进行优化，Stage2 无损转换。DeiT-S W3A3：RepQ-ViT 仅 4.37%，I&S-ViT SOS 单独达 45.19%（+40.82%），证明优化-based 方法在低比特场景远远优于纯校准方法。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---

## Salience-Determined Bit Allocation (SBA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SBA 是 SliM-LLM (ICML 2025) 提出的 group-wise 混合精度 bit-width 分配算法。核心思想：利用 LLM 中 salience weight 在 channel 维度上呈现 spatial clustering 的现象（由 activation outlier channels 驱动，Theorem 1 证明：x_{:,p}^* >> x_{:,j} → H_{p,p} >> H_{j,j} → δ_{:,p} > δ_{:,k}），按 group 平均 salience 排序后，通过双指针搜索最优混合精度配置。目标函数为 KL divergence D_KL(xW^T || xŴ_sba^T)，从信息熵角度对齐量化前后输出分布。约束条件为 |G_{N-1}| = |G_{N+1}|（等量低/高精度 group 维持 target 平均 bit-width）。例如 2-bit 场景：高 salience group 给 3-bit，等量低 salience 给 1-bit，其余 2-bit。搜索空间有限（LLaMA-7B 仅需 16 iterations）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: 权重W (n×m), 校准数据x (t×m), 逆Hessian H^in, group_size β, target bit N
# 输出: 每个group的bit-width G[0..k-1]

# Step 1: 计算每个group的平均salience
k = m / β  # group数量 (e.g. 4096/128=32)
for i in range(k):
    w_g = W[:, i*β : (i+1)*β]
    S[i] = mean(w_g² / diag(H^in[i*β:(i+1)*β])²)

# Step 2: 按salience排序groups (ascending)
sorted_idx = argsort(S)

# Step 3: 双指针搜索最优混合精度比
best_kl = INF
for p in range(1, k//2 + 1):
    # lowest p groups → (N-1)-bit, highest p → (N+1)-bit, middle → N-bit
    Ŵ_mixed = fakequant(low_sal, N-1) ∪ fakequant(high_sal, N+1) ∪ fakequant(mid, N)
    kl = D_KL(softmax(x·W^T) || softmax(x·Ŵ_mixed^T))
    if kl < best_kl: best_kl, best_p = kl, p

# 分配: G[sorted_idx[0:best_p]] = N-1; G[sorted_idx[k-best_p:k]] = N+1; G[middle] = N
```
与 ILP（整数线性规划，HAWQ v2 方法）对比：ILP 在 {1,2,3}bit 候选空间中有更宽搜索范围，但 SBA 基于输出 KL 散度的双指针搜索在固定整数 bit-width 下实现更优匹配。LLaMA-7B 2-bit WikiText2 PPL: ILP=17.55 vs SBA=14.58。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SBA 实现集成在 GPTQ 或 OmniQuant 的量化流程中，在权重量化前执行。校准数据 forward pass 获取每层输入 x 计算 Hessian 后，逐层运行 SBA 确定各 group 的 bit-width 配置，随后 GPTQ/OmniQuant 按配置对各 group 使用对应精度量化。SBA 双指针搜索复杂度 O(k²)，但 k 很小（LLaMA-7B 中 k=32，每层仅 16 次迭代），开销可忽略。开源代码：https://github.com/Aaronhuang-778/SliM-LLM。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

---

## Salience-Weighted Quantizer Calibration (SQC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SQC 是 SliM-LLM (ICML 2025) 提出的量化器校准方法，用于增强 group 内局部 salient 权重的表达能力。动机：即使 SBA 给高 salience group 分配高 bit-width，group 内部仍有约 1% 的离散 salient 元素与非 salient 元素共享同一套量化器参数 (scale/z)。传统量化器以所有元素的最小均方误差为目标，非 salient 元素（占~99%）主导优化方向，导致 salient 信息退化。SQC 通过 3-σ 规则筛选 salient 权重 (w_s = {w | w < μ-3σ ∪ w > μ+3σ})，引入 calibration 参数 τ 扩展量化器感知区间，对 scale 和 zero point 在 [1-λ, 1+λ] 区间搜索（λ=0.1, 50 candidates），优化加权目标 argmin_τ (||w_s - τs·Q(w_s, τs, τz)||² + ||w_us - τs·Q(w_us, τs, τz)||²)。关键设计：w_s 和 w_us 共享同一套 (τs, τz)，无需额外存储，保持推理效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: group权重 w^b (n×128), bit-width g_b
# 输出: 最优量化参数 Δ*, z*

# Step 1: 3-σ规则筛选salient权重
μ, σ = mean(w^b), std(w^b)
w_s = {w | w < μ-3σ or w > μ+3σ}  # 约占1%
w_us = w^b - w_s                    # 剩余~99%

# Step 2: 搜索最优τ
w_max, w_min = max(w^b), min(w^b)
best_loss = INF; λ = 0.1; n = 50
for τ in linspace(1-λ, 1+λ, 2n):
    Δ = τ * (w_max - w_min) / (2^g_b - 1)
    z = -⌊τ * w_min / Δ⌋
    ŵ_s = fakequant(w_s, g_b, Δ, z)
    ŵ_us = fakequant(w_us, g_b, Δ, z)
    loss = ||w_s - ŵ_s||²₂ + ||w_us - ŵ_us||²₂
    if loss < best_loss: best_loss, Δ*, z* = loss, Δ, z

# Step 3: 使用最优参数量化整个group
ŵ_q^b = fakequant(w^b, g_b, Δ*, z*)
```
效果：OPT-1.3B 某 channel 绝对误差从 0.0055（vanilla quantizer）降至 0.0039（SQC），salient 权重区域误差显著降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SQC 作为 GPTQ 流程中 fakequant 步骤的替代实现：GPTQ 逐列量化时，每列调用 SQC（替代默认 MinMax 量化器）确定最优 τ、Δ、z。SQC 仅改变量化器参数搜索方式，不引入额外推理参数（τ* 融入最终 Δ* 和 z*）。SQC 可与 SBA 独立使用：SBA 提供 group 级混合精度优化，SQC 在统一精度场景也可独立提升量化质量。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

---

## Parameter Salience in LLM PTQ (δ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parameter Salience（参数显著性 / δ）是 LLM 后训练压缩中的权重重要性度量，定义为量化某个权重元素对层输出产生的二阶误差。基于 SparseGPT (Frantar & Alistarh, 2023) 的形式化：在二次近似 loss 下 L(Ŵ) = tr((Ŵ-W)H(Ŵ-W)^T)，消除 (i,j) 位置元素引入的误差为 δ_{i,j} = w_{i,j}² / [H⁻¹]_{j,j}²。其中 H = (1/P) Σ x^{[k]T} x^{[k]} 是校准数据激活的 Hessian 近似，H⁻¹ 通过 Cholesky 分解高效计算。δ 结合了权重幅值和激活二阶信息：||w|| 大的元素不一定显著（若对应 channel 激活小），而激活 outlier channel 上的权重即使很小也可能高 δ。SliM-LLM 揭示由于 activation outlier channels 始终出现在固定位置（token sink 现象），权重 δ 沿 channel 方向呈现 spatial clustering，这正是 group-wise 混合精度的理论基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: 权重W (n×m), 校准激活x (t×m)
# 输出: 每个元素的salience δ_{i,j}

H = (1/t) * x^T · x              # Hessian近似 (m×m)
H_inv = Cholesky((H + λI)^(-1))  # 逆Hessian Cholesky因子
inv_diag = diag(H_inv)^2          # 对角元平方

for j in range(m):
    δ[:,j] = W[:,j]² / inv_diag[j]²

# 简化近似 (λ极小时): δ_{i,j} ≈ (W_{i,j} · ||x_j||²₂)²
```
Theorem 1 (SliM-LLM)：激活 outier channel p 使 ||x_p||₂  >> ||x_j||₂, ∀j≠p，导致 H_{p,p} >> H_{j,j}，在 λ ≤ e^{-1} 下 δ_{:,p} >> δ_{:,k}。定理连接了 activation outliers → Hessian 对角 dominance → channel-wise salience clustering 的逻辑链。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
δ 在 LLM 压缩中的主要用途：(1) SparseGPT 稀疏化——按 δ 排序选保留权重；(2) PB-LLM/SpQR 元素级混合精度——按 δ 阈值分离高/低精度权重；(3) SliM-LLM group 级混合精度——group 平均 δ 决定 bit-width；(4) BiLLM 显著权重识别——δ 指导结构化二值分割。计算开销主要在 Cholesky 分解 O(m³)，但 per-layer 仅一次。使用 128 个 2048-token 校准样本即可获稳定 Hessian 估计。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---

## KL Divergence for Quantization Bit Allocation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KL Divergence (Kullback-Leibler Divergence) 在 SliM-LLM 的 SBA 中用作混合精度 bit-width 分配的优化目标函数，替代传统 MSE。形式为 D_KL(softmax(xW^T) || softmax(xŴ_sba^T))，即量化前后输出经 softmax 化为概率分布后的 KL 散度。动机：MSE 最小化权重重建误差但不保证输出分布对齐——相同 MSE 的量化方案可能产生不同 token 概率分布偏移。KL 散度从信息熵角度衡量输出分布的偏移，使 bit allocation 更偏向保护对输出分布影响大的 group。实验验证：2-bit 时 KL 优于 MSE（OPT-1.3B: KL=30.71 vs MSE=32.50; LLaMA-7B: KL=14.58 vs MSE=21.94），差异随模型增大更显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 在SBA双指针搜索中作为优化目标
for p in (1 to k//2):
    Ŵ_mixed = assign_mixed_precision(W, sorted_idx, p, N)
    out_q = x · Ŵ_mixed^T      # [t, n]
    out_fp = x · W^T           # [t, n]
    P = softmax(out_q)         # 概率分布
    Q = softmax(out_fp)
    kl = sum(P * log(P / Q)) / t  # 逐token平均KL
```
与 MSE 对比：MSE = mean((out_q - out_fp)²) 考虑逐元素差值；KL = D_KL(P||Q) 考虑分布形状。当 out_q 和 out_fp 各维度等比例偏移时 MSE > 0 但分布一致 KL ≈ 0；当 out_q 某维度剧烈偏差时 KL 惩罚远大于 MSE。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`F.kl_div(F.log_softmax(out_q, dim=-1), F.softmax(out_fp, dim=-1), reduction='batchmean')`。SBA 在 layer 级别计算 KL（非 global），逐层优化 bit allocation。计算比 MSE 多一次 softmax，但 SBA 搜索空间小（k ≤ 32），总体开销可忽略。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

## Sensitivity-Based Non-Uniform Quantization (SqueezeLLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sensitivity-Based Non-Uniform Quantization 是 SqueezeLLM 提出的 weight-only 后训练量化方法。核心思想是将量化问题转化为 weighted k-means clustering：优化目标不是最小化简单的 L2 reconstruction error `||W - W_Q||²`，而是最小化 Fisher-weighted error `Σ F_ii (w_i - Q(w_i))²`，其中 `F_ii` 是 Fisher 信息矩阵对角线（≈Hessian 对角），作为每个权重的 importance weight。这源于 Optimal Brain Damage (OBD) 框架：对 loss 做 Taylor 展开，假设模型已收敛（g≈0），`L(W_Q) ≈ L(W) + ½(W-W_Q)ᵀH(W-W_Q)`，用 Fisher 对角近似 Hessian 对角。结果是 k-means centroids 被"拉向"对最终 loss 更敏感的高 Fisher 值权重，在敏感区域提供更高的量化分辨率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Step 1: 计算 Fisher 信息矩阵 (sensitivity)
# 对 calibration 数据集 D (10-100 samples)
F = zeros_like(W)  # per-weight Fisher diagonal
for each sample d in D:
    loss = cross_entropy(model(d), labels)
    g = backward(loss)  # gradient w.r.t. all weights
    F += g ⊙ g          # element-wise square
F /= |D|                # average over samples

# Step 2: Sensitivity-weighted k-means per output channel
# 目标: argmin_Q Σ_i F_ii * (w_i - Q(w_i))²
for each output channel c:
    k = 2^bit  # e.g., k=8 for 3-bit
    centroids = kmeans++_init(W[c,:], k)
    repeat until convergence:
        # E-step: assign each weight to nearest centroid
        for i in range(in_features):
            assignment[i] = argmin_j ||w_i - centroid_j||²
        # M-step: update centroids weighted by Fisher values
        for j in range(k):
            mask = (assignment == j)
            centroid_j = Σ_{i∈mask} F_ii * w_i / Σ_{i∈mask} F_ii
    LUT[c] = centroids  # FP16, k entries
    indices[c] = assignment  # b-bit per weight, packed
```

效果（LLaMA-7B 3-bit, C4 perplexity）：
- RTN (uniform): PPL 28.26
- Sensitivity-agnostic k-means: PPL 18.08
- Sensitivity-based k-means: PPL 7.75 (close to FP16 baseline 7.08)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源：https://github.com/SqueezeAILab/SqueezeLLM。Fisher 计算需一次完整 backward pass（A100: 7B=0.3min, 65B=2.5min），k-means 在 CPU 上执行（Xeon 48核: 7B=11min, 65B=80min）。关键权衡：(1) 仅需 10-100 校准样本即收敛（比 GPTQ/AWQ 的 128 更少）；(2) Fisher 计算的一次性峰值内存需求高（7B=33GB, 65B=292GB）；(3) 每 channel 需存储 k 个 FP16 centroid（如 3-bit: 8 FP16/channel），但 memory-bound 推理中 LUT dequant 额外计算可被内存带宽瓶颈掩盖。与传统 layer-wise objective（GPTQ: min ||WX - W_QX||²）相比，final-loss-based objective 在 PPL 上有约 0.3 的系统性优势（D.4 消融实验）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization

## Dense-and-Sparse Decomposition (Weight Matrix)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
将权重矩阵分解为 Dense + Sparse 两部分以改善量化性能的方法。SqueezeLLM 提出：W = D + S，S（sparse, CSR FP16）包含 ~0.45% 的异常值（百分位阈值外）和高敏感值（Fisher 排名 top 0.05%），D（dense）包含剩余 99.55% 权重并以非均匀量化存储。动机：(1) LLM 权重中 99.9% 的值集中在 ~10% 的 range 内，少量 outliers 膨胀量化范围 10x → 去除它们大幅缩小 D 的 value range → 提高量化分辨率；(2) 敏感值以 FP16 保留避免扰动最终输出，同时防止 k-means centroids 被它们"拉偏"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# S = {outliers} ∪ {sensitive}
T_min, T_max = percentile(W, 0.2%), percentile(W, 99.8%)  # outlier 阈值
O = {i | W_i < T_min or W_i > T_max}  # ~0.4%
S_top = topk_indices(F, k=0.05%×|W|)  # Fisher 排名最前的 0.05%
S_indices = O ∪ S_top  # ~0.45% (去重)

# S 存储为 CSR: values (FP16) + col_indices (int16) + row_ptrs (int32)
# D = W - S (S_indices 处清零)
D_indices, LUTs = weighted_kmeans_quantize(D, F, bit)

# 推理: Y = D @ X + S @ X (fused kernel)
```

存储分析：每个 sparse 元素 ~32-33 bits (16b value + 16b index + row_ptr 分摊) → 0.45% sparsity 增加 ~0.24 bit/param 的 overhead。3-bit dense + 0.45% sparsity = 3.24 avg bits。与 grouping 对比：Dense-and-Sparse 是 outlier 问题的直接解决方案（直接移除 outliers），grouping 是间接方案（组内隔离），且 grouping + 非均匀量化组合需 per-group LUT（overhead 更大）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源：https://github.com/SqueezeAILab/SqueezeLLM。关键参数：sparsity level 0.45%（0.4% outliers + 0.05% sensitive），已通过消融实验验证为 sweet spot——更低的 sparsity 不足以覆盖主要 outliers，更高的 sparsity 引入 diminishing returns（D.2/D.5）。在 GPU 推理中，sparse part 使用 balanced CSR kernel（10 nonzeros/thread）以处理 per-row sparsity skew。SpQR 也采用类似 dense+sparse 策略，但主要区别在于：SpQR 依赖 fine-grained grouping+bi-level quantization 处理 dense part，而 SqueezeLLM 用 sensitivity-based non-uniform quantization，用更少的 sparsity（0.05% vs 1%）和更简单的 quantization scheme 达到更好效果。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization

## Weight-Only Quantization for LLM Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-Only Quantization 是只量化权重而不量化激活值的后训练量化策略。在 LLM 单 batch 生成式推理中，每 token 解码仅涉及 GEMV 操作（matrix-vector multiply），arithmetic intensity 极低——每个权重加载后只参与一次乘加，无法跨多 token 分摊。推理完全受限于 memory bandwidth 而非 compute。因此只压缩权重（减少内存流量）同时保持 activations 为 FP16，即可获得接近压缩比的加速。SqueezeLLM Sec. 3 用 roofline model 验证：A5000 GPU 上降低 weight 精度→延迟线性下降（Fig. 2），证明 memory 是主导瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线: 量化权重
W_indices_3bit, LUTs, S_csr = quantize(W)  
# 在线推理: activations 保持 FP16
for each Linear layer:
    # Dense: 加载 3-bit indices → LUT查表FP16 → matvec
    Y = lut_dequant_matvec(W_indices, LUTs, X)
    # Sparse (optional): CSR SpMV
    Y += balanced_csr_matvec(S_csr, X) 
# 加速比 ≈ 压缩比 (memory-bound region of roofline)
```

关键前提：(1) 单 batch 推理（batch>1 时 compute 可能成为瓶颈）；(2) GPU memory bandwidth << compute throughput（如 A6000: 768 GB/s vs 222 TFLOPS, ~290x 差距）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表性实现：GPTQ (uniform group quant), AWQ (activation-aware scaling), SqueezeLLM (non-uniform weighted k-means + sparse), SpQR (GPTQ-style + grouping + sparse)。共同特征：activations FP16，计算在 FP16。主要挑战：如何在 3-4 bit 下最小化 perplexity 退化。局限：在 batch_size>1 或 prefill 阶段（compute-bound）加速效果减弱；需要自定义 CUDA kernel 实现 dequantization（PyTorch 原生不支持 3-bit 非均匀格式）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

---

## Zero-shot Quantization (ZSQ / 零样本量化 / Data-Free Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Zero-shot Quantization（ZSQ），也称为 Data-Free Quantization（DFQ），是一种在无法访问原始训练数据的情况下对预训练模型进行量化的技术。ZSQ 的核心流程分为两步：(1) **合成数据集生成（Step 1）**：利用预训练模型的内部统计信息（如 Batch Normalization 层的 running mean 和 standard deviation、分类层预测分布、特征纹理等）反向优化随机噪声或训练一个生成器，生成与原始训练数据分布相似的合成样本；(2) **量化模型微调（Step 2）**：使用 Round-To-Nearest（RTN）初始化量化模型，然后用合成数据集微调量化模型，最小化 KL 散度（知识蒸馏）和交叉熵损失。ZSQ 在数据因隐私、安全或法规原因不可用的真实场景（如医疗数据、商业机密数据）中至关重要。

ZSQ 方法按合成数据集生成方式分为三类：
- **Synthesis-free ZSQ**：无需生成数据，仅基于模型参数属性（如 weight equalization、bias correction）校准量化参数，但在极低位宽（3-bit/4-bit）下性能急剧退化。
- **Generator-based ZSQ**：训练一个额外的生成器网络（如 GAN）产生合成样本。代表方法：GDFQ（首次使用 BN 统计引导的生成器）、ARC/AutoReCon（NAS-based 图像重建）、Qimera（叠加潜在嵌入生成边界支持样本）、AdaSG（将 ZSQ 建模为零和博弈）、AdaDFQ（自适应调节合成样本难度）。
- **Noise-optimization-based ZSQ**：直接从随机高斯噪声出发，通过梯度下降迭代优化噪声以匹配预训练模型统计信息，无需训练额外生成器。代表方法：IntraQ（保留类内异质性）、HAST（困难样本生成与难度提升）、TexQ（纹理特征分布校准）、PSAQ-ViT（ViT 的 patch similarity 引导）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 SynQ 论文的 Noise-optimization ZSQ pipeline（ResNet-18 W3A3 on ImageNet）为例：
```
// === Step 1: 合成数据集生成（Noise Optimization） ===
Initialize {x_i}_{i=1}^{5120} ~ N(0, 1)          // 5120张高斯噪声图像
Randomly assign labels {y_i}_{i=1}^{5120}         // 随机类别标签
for iter in 1..1000:
    // 最小化两项损失
    L_IL = (1/N) Σ CE(q(x_i; θ), y_i)             // Inception Loss: 预训练模型预测对齐标签
    L_BNS = (1/L) Σ ||μ^l(θ) - μ^l(θ,{x_i})||²   // BN Statistics Loss: 匹配BN层统计
           + ||σ^l(θ) - σ^l(θ,{x_i})||²
    L_total = L_IL + α * L_BNS
    x_i = x_i - η * ∇_{x_i} L_total              // 更新合成样本（不更新模型参数）
    if loss plateau for 50 iters: η *= 0.1       // 学习率衰减

// === Step 2: 量化与微调 ===
θ^q = RTN_quantize(θ, bit=3)                      // 使用RTN初始量化（Min-max）
for epoch in 1..100:
    for each x_i in {x_i}:
        loss = KL(q(x_i; θ) || q(x_i; θ^q))        // 知识蒸馏（始终应用）
             + λ_CE * CE(q(x_i; θ^q), y_i)         // 交叉熵（硬标签）
    θ^q = θ^q - η * ∇_{θ^q} loss                   // SGD 更新 θ^q
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ZSQ 的通用实现方式：(1) 使用 PyTorch 加载预训练模型，提取 BN 层 running_mean 和 running_var；(2) 初始化随机噪声张量作为合成样本，在某些方法中还需初始化生成器网络；(3) 使用 Adam 优化器迭代优化合成样本（或训练生成器），损失函数通常包括 BNS loss（L2 距离匹配 BN 统计量）和 CE loss/IL（促使预训练模型对合成样本做出指定类别的预测）；(4) 使用 torch.quantization 或自定义 STE 伪量化模块执行 RTN 量化初始化和微调；(5) 微调时通常使用 SGD + momentum=0.9 优化器，学习率在 {1e-4, 1e-5, 1e-6} 范围。SynQ 开源地址：https://github.com/snudm-starlab/SynQ。ZSQ Survey (IJCAI 2025)：https://github.com/snudm-starlab/ZSQ-Survey。

ZSQ 用于目标检测的 Task-Specific 扩展：Task-Specific ZSQ for Object Detection 指出，现有 ZSQ 方法用于检测任务时存在根本性问题——分类 ZSQ 仅需随机采样类别标签（如 U(0,1000) for ImageNet），而检测任务需要同时合成边界框坐标和类别标签。task-agnostic ZSQ（如 PSAQ-ViT V2、MimiQ、CLAMP-ViT）仅使用 BNS 或 PSE 先验损失生成无任务的通用图像，放弃检测训练损失，导致合成图像缺乏目标位置、尺寸和类别分布的 task-specific 信息，性能次优。task-specific 方案通过 Adaptive Label Sampling（利用预训练检测网络自动重建 label）和 task-specific QAT（同时使用 KL 蒸馏 + 特征蒸馏 + L_detect 检测损失）解决了这一问题，在 W6A6 YOLOv5-l 上超越 full-data LSQ +1.8% mAP，仅用 1/60 训练数据。开源代码：https://github.com/DFQ-Dojo/dfq-toolkit。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

---

## Noise Optimization for Synthetic Dataset Generation（合成数据集的噪声优化生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise Optimization 是 Zero-shot Quantization（ZSQ）中合成数据集生成的主流方法之一。其核心思想是：不训练额外的生成器网络，而是直接从随机高斯噪声出发，通过梯度下降迭代优化噪声本身以匹配预训练模型的内部统计信息（如 Batch Normalization 层的 running mean/variance）和分类行为（使预训练模型对优化后的噪声做出指定类别的预测）。与 Generator-based 方法相比，Noise Optimization 的优势在于：(1) 无需额外训练生成器，节省计算和存储；(2) 生成样本数量可控（如 5120 张）；(3) 灵活性高，可与其他优化目标（如硬样本生成、纹理校准）组合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SynQ 论文中使用的 Noise Optimization pipeline（结合 TexQ 的 calibration center synthesis 和 HAST 的 hard sample generation）：
```
// Stage 1: Calibration Center Synthesis (TexQ)
for each class c in [1..C]:
    x_center[c] = random_noise(3, H, W)            // 每类一个校准中心
    optimize x_center[c] to minimize L_IL + α^C * L_BNS  // 纹理特征校准

// Stage 2: 批量合成样本生成
Initialize {x_i}_{i=1}^N with N(0,1)
for iter in 1..1000:
    // 三项损失
    L_BNS: (1/L) Σ || (μ_l, σ_l) - (μ_l({x_i}), σ_l({x_i})) ||²
    L_IL: (1/N) Σ CE(q(x_i), y_i)                  // Inception Loss
    L_HIL: (1/N) Σ δ(x_i) * CE(q(x_i), y_i)        // HAST: Hard-sample-enhanced IL
    L_Total = L_IL + α_1 * L_HIL + α_2 * L_BNS
    x_i -= η * ∇ L_Total                            // 更新噪声样本

// Stage 3: Sample Difficulty Promotion (HAST)
for each x_i:
    perturbation ε ~ N(0, σ) scaled by λ_P
    x_i = x_i + ε                                   // 附加扰动增加难度
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中实现：(1) 将噪声张量包装为 nn.Parameter；(2) 使用 Adam 优化器（lr=0.5, momentum=0.9）直接优化噪声参数，学习率每 50 步不下降时衰减 0.1；(3) 合成图像使用与原始模型相同的归一化预处理（mean/std）；(4) BNS loss 计算每层 BN 统计量的 L2 距离，Inception Loss 计算预训练模型对合成样本预测分布的交叉熵；(5) 总共生成 N=5120 张图像，batch size=256，共 20 个 batch。主要优势：仅需几分钟到几十分钟即可完成数据生成，远快于 Generator-based 的几小时训练。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---

## Batch Normalization Statistics (BNS) Loss（批归一化统计损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Batch Normalization Statistics (BNS) Loss 是 Zero-shot Quantization 中用于引导合成数据集生成的损失函数。其核心思想是：预训练模型的 Batch Normalization 层在训练过程中累积了原始训练数据的 running mean μ^l(θ) 和 running standard deviation σ^l(θ)，这些统计量本质上编码了原始数据在各层的分布特征。BNS Loss 通过最小化合成数据在预训练模型上计算出的统计量与原始统计量之间的 L2 距离，迫使合成数据的分布逼近真实训练数据的分布。公式为 L_BNS = (1/L) Σ_{l=1}^L (||μ^l(θ) - μ^l(θ,{x_i})||² + ||σ^l(θ) - σ^l(θ,{x_i})||²)，其中 L 为 BN 层总数，μ^l(θ,{x_i}) 和 σ^l(θ,{x_i}) 是当前合成样本 batch 在前向传播时在 BN 层 l 处计算出的均值和标准差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_BNS_loss(model, synthetic_batch):
    """计算所有BN层的统计量匹配损失"""
    loss = 0.0
    bn_layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # module.running_mean: 训练期间累积的原始数据均值 (frozen)
            # module.running_var:  训练期间累积的原始数据方差 (frozen)
            # 前向传播中计算的当前batch统计量
            with torch.no_grad():
                model(synthetic_batch)  # 触发BN层更新当前batch的统计
            # 获取当前batch在BN层的mean和var（需要hook或forward hook提取）
            current_mean = get_current_batch_mean(module)  # μ^l(θ, {x_i})
            current_var  = get_current_batch_var(module)   # σ²
            target_mean  = module.running_mean             # μ^l(θ)
            target_var   = module.running_var              # σ²(θ)
            loss += torch.norm(current_mean - target_mean, p=2)**2
            loss += torch.norm(torch.sqrt(current_var) - torch.sqrt(target_var), p=2)**2
            bn_layer_count += 1
    return loss / bn_layer_count
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中，BN 层的 running_mean 和 running_var 在模型训练后即被冻结（model.eval() 模式下不更新）。实现 BNS Loss 需要：(1) 通过 register_forward_hook 或直接访问 model.modules() 获取每层 BN 的统计量；(2) 在前向传播时使用 model.train() 模式使 BN 层计算当前 batch 的统计（而非使用 running 统计），部分实现要求临时冻结 BN 的 affine 参数；(3) 计算 L2 距离平方作为损失。BNS Loss 是 ZSQ 合成数据集生成的最基础损失项，几乎所有 Noise-optimization ZSQ 方法（ZeroQ, IntraQ, HAST, TexQ, SynQ）均使用。平衡超参数 alpha 通常设为 0.01-1.0。

在 Task-Specific ZSQ for Object Detection 中，BNS Loss 作为 L_prior 用于 CNN-backbone 模型（YOLOv5、CNN-based Mask R-CNN），但与标准 ZSQ 不同——BNS Loss 与 task-specific 检测损失 L_detect 联合使用：L_total = alpha_prior*L_BNS + alpha_detect*L_detect + L_reg，其中 L_detect = L_category + L_box + L_conf，使得合成图像不仅匹配 BN 统计分布，还重建目标检测任务需要的类别和边界框信息。YOLOv5 超参 {alpha_detect, alpha_BN, alpha_TV, alpha_l2} = {0.5, 0.01, 0, 5e-4}。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

---

## Gaussian Low-pass Filter for Synthetic Data in ZSQ（ZSQ中合成数据的高斯低通滤波）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gaussian Low-pass Filter for Synthetic Data 是 SynQ 论文提出的解决 ZSQ 合成数据集高频噪声问题的技术。合成数据集由噪声优化生成（起始于高斯噪声），其频域特征与真实图像显著不同——合成样本的幅度谱均匀分布在高频和低频区域，而真实图像（如 ImageNet）的能量主要集中于低频区域。SynQ 通过傅里叶变换将合成样本 x_i 转换到频域，逐元素乘以 2D 高斯低通核 G 进行滤波，再应用逆傅里叶变换得到滤波后的样本 x_i^F。高斯核 G_{uv} = exp(-D(u,v)²/(2D₀²))，其中 D(u,v) 为频域坐标 (u,v) 到中心的距离，D₀ 为截止频率超参数。滤波后合成样本的幅度分布显著接近真实图像，缓解了噪声导致微调效率低下的问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def gaussian_low_pass_filter(x, D0):
    """对合成样本x应用高斯低通滤波器"""
    # x: (C, H, W) 或 (N, C, H, W) 张量
    # D0: 截止频率（ImageNet推荐值: 40-60）
    C, H, W = x.shape[-3], x.shape[-2], x.shape[-1]

    # 1. 构建2D高斯低通核
    u = torch.arange(W).float() - W/2
    v = torch.arange(H).float() - H/2
    U, V = torch.meshgrid(u, v, indexing='xy')
    D = torch.sqrt(U**2 + V**2)                    # 到频域中心的距离
    G = torch.exp(-D**2 / (2 * D0**2))              # 高斯低通核 H×W
    G = G.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W)

    # 2. 逐通道应用频域滤波
    x_filtered = torch.zeros_like(x)
    for c in range(C):
        F_x = torch.fft.fft2(x[c])                  # 2D FFT
        F_x_shifted = torch.fft.fftshift(F_x)       # 零频移到中心
        F_filtered = F_x_shifted * G.squeeze()       # 逐元素乘高斯核
        F_ishifted = torch.fft.ifftshift(F_filtered)
        x_filtered[c] = torch.fft.ifft2(F_ishifted).real  # 逆FFT取实部
    return x_filtered

# 在SynQ pipeline中的使用位置（生成合成数据集后、微调前）
synthetic_dataset = generate_synthetic_samples(model, N=5120)
filtered_dataset = [gaussian_low_pass_filter(x_i, D0=50) for x_i in synthetic_dataset]
# filtered_dataset随后用于微调量化模型
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 使用 PyTorch 的 torch.fft.fft2 / torch.fft.ifft2 进行 2D 快速傅里叶变换，时间复杂度 O(Z log Z)（Z=HW）；(2) 对 CIFAR 数据集使用较小 D₀（约 8-20），对 ImageNet 使用较大 D₀（约 40-60）；(3) 滤波在合成数据集生成完成后、量化模型微调开始前一次性离线完成，不增加微调循环的计算开销；(4) 过小的 D₀ 会导致过度平滑丢失关键信息，过大的 D₀ 则滤波效果不足。SynQ 消融实验证明低通滤波器是三项贡献中影响最大的（ResNet-18 W3A3：基线 43.63% → +I1 49.43%，+5.80pp）。该技术可直接应用于任意 ZSQ 方法生成的合成数据集。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---

## Class Activation Map (CAM) Alignment in ZSQ（ZSQ中的类激活图对齐）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Class Activation Map (CAM) Alignment 是 SynQ 论文提出的解决 ZSQ 中量化模型"基于错误图像区域预测（off-target patterns）"问题的技术。核心思想：在微调量化模型时，除了标准的 KL 散度（知识蒸馏）和交叉熵损失外，额外加入 CAM 对齐损失 L_CAM = ||S^θ(x_i) - S^θ^q(x_i)||_F²，其中 S^θ 和 S^θ^q 分别为预训练模型和量化模型的 Grad-CAM 显著性图。通过最小化两者之间的 Frobenius 范数（等价于 MSE），强制量化模型关注与预训练模型相同的图像判别区域，从而将目标定位知识从预训练模型蒸馏到量化模型。与 HAST 的特征对齐（Feature Alignment，对齐中间层激活图）相比，CAM 对齐直接对齐"与预测结果因果相关的区域"，更精确地解决了"off-target prediction"问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_CAM_loss(model_fp, model_q, x, target_class):
    """计算预训练模型与量化模型之间的CAM对齐损失"""
    # 1. 获取目标层的激活和梯度
    # 以ResNet为例，选择最后一层卷积层（layer4）
    activation_fp = get_layer_activation(model_fp, 'layer4', x)  # A^{k;θ}(x)
    activation_q  = get_layer_activation(model_q,  'layer4', x)  # A^{k;θ^q}(x)

    # 2. 计算预训练模型的Grad-CAM
    score_fp = model_fp(x)[target_class]   # y^{y_i}
    grad_fp = autograd.grad(score_fp, activation_fp)[0]  # ∂y/∂A
    alpha_fp = grad_fp.mean(dim=(2,3))     # 全局平均池化 → 通道权重
    S_fp = torch.relu((alpha_fp.view(-1,1,1) * activation_fp).sum(dim=1))
    S_fp = S_fp / (S_fp.max() + 1e-8)      # 归一化到[0,1]

    # 3. 计算量化模型的Grad-CAM（同理）
    score_q = model_q(x)[target_class]
    grad_q = autograd.grad(score_q, activation_q)[0]
    alpha_q = grad_q.mean(dim=(2,3))
    S_q = torch.relu((alpha_q.view(-1,1,1) * activation_q).sum(dim=1))
    S_q = S_q / (S_q.max() + 1e-8)

    # 4. MSE对齐
    L_CAM = torch.norm(S_fp - S_q, p='fro')**2
    return L_CAM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SynQ 实现中：(1) CAM 对齐选择 Grad-CAM 作为显著性图技术（优于 CAM 和 Grad-CAM++，见 Section 5.4）；(2) 平衡超参数 λ_CAM 在 {20, 50, 100, 200, 300, 500, 2000} 中 grid search；(3) CAM 对齐的时间复杂度为 O(NLT_θ)，每次迭代需执行预训练模型和量化模型各一次前向传播和 L 层反向传播（用于计算梯度），微调时间开销约增加 17.81%；(4) CAM 对齐可无缝集成到任意使用合成数据集的 ZSQ 方法中。SynQ 实验验证 CAM 对齐显著优于特征对齐（表6：CAM 48.26% vs FA 46.77%），且两者训练时间几乎相同。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---

## Grad-CAM (Gradient-weighted Class Activation Mapping / 梯度加权类激活映射)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Grad-CAM（Gradient-weighted Class Activation Mapping）是 Selvaraju et al. (ICCV 2017) 提出的可视化 CNN 决策依据的技术。Grad-CAM 生成一张与输入图像同尺寸的热力图（saliency map），高亮区域表示模型在做出特定类别预测时"关注"的图像部分。其核心公式为：S^θ(x_i) = ReLU(Σ_k α_k · A^{k;θ}(x_i))，其中 A^{k;θ}(x_i) 为目标卷积层第 k 通道的激活图，α_k = (1/(W_k H_k)) Σ_{w,h} ∂y^{y_i}/∂A^{k;θ}_{wh}(x_i) 为第 k 通道对目标类别预测分数的平均梯度权重（即通道重要度），ReLU 过滤掉负贡献区域只保留对预测有正面影响的区域。Grad-CAM 是 CAM (Zhou et al., CVPR 2016) 的直接推广——CAM 要求模型末尾有全局平均池化层（GAP），Grad-CAM 通过梯度反向传播消除了此限制，可应用于任意 CNN 架构。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 SynQ 论文中，Grad-CAM 作为 CAM 对齐损失 L_CAM 的显著性图生成方法，其计算流程为：
```
输入: 预训练模型θ, 输入图像x, 真实类别y_i, 目标层layer4
输出: 显著性图 S^θ(x) (尺寸: H×W)

1. 前向传播: output = θ(x), 记录layer4的激活A ∈ R^{K×W'×H'}
2. 获取预测分数: y_score = output[y_i]           // 真实类别的logit
3. 反向传播梯度: grad = ∂(y_score) / ∂A           // K×W'×H'
4. 全局平均池化梯度: α_k = (1/(W'H')) Σ_{w,h} grad[k,w,h]  // K维向量
5. 加权组合: S_raw = Σ_k α_k · A[k,:,:]           // W'×H'
6. ReLU过滤: S = ReLU(S_raw)                      // 丢弃负贡献
7. 上采样: S = interpolate(S, (H,W))              // 恢复到原图尺寸
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Grad-CAM 在 PyTorch 中的实现：(1) 使用 register_forward_hook 捕获目标层的激活 A；(2) 使用 register_full_backward_hook 或 torch.autograd.grad 获取梯度；(3) 在指定类别（通常为真实标签类别）上反向传播获取通道权重。SynQ 对比了三种 CAM 技术（W3A3 ResNet-18）：Grad-CAM > CAM > Grad-CAM++。Grad-CAM++ 专为多目标定位设计，对于单目标分类任务不如 Grad-CAM。Grad-CAM 优于 CAM 的另一个优势是：CAM 仅适用于末尾有 GAP 层的模型（如 ResNet），Grad-CAM 无此限制。Grad-CAM 还可用于模型可解释性分析、弱监督目标定位、以及作为知识蒸馏中的注意力转移目标。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---

## Soft Labels for Difficult Samples in ZSQ（ZSQ中困难样本的软标签策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Soft Labels for Difficult Samples 是 SynQ 论文提出的解决 ZSQ 中"错误硬标签误导微调"问题的策略。在 ZSQ 中，合成数据集的标签在初始化时被随机分配，然后通过最小化 Inception Loss 使预训练模型对合成样本预测出对应标签。然而，预训练模型对高难度样本的预测常出错（错误率随难度增加而显著上升，见图3），导致这些样本的硬标签不可靠。SynQ 提出：对难度 δ = 1 - q_{y_i}(x_i; θ) 超过阈值 τ（通常 τ=0.5）的困难样本，训练时完全跳过交叉熵（硬标签）损失，仅使用 KL 散度（软标签/知识蒸馏），以避免错误硬标签对量化模型微调的误导。对于容易样本（δ ≤ τ），则同时使用 KL 散度和交叉熵损失。这一策略通过指示函数 1_{δ(x_i,θ) ≤ τ} 动态决定 CE 损失的施加与否。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_SynQ_loss(model_fp, model_q, x_i, y_i, τ=0.5):
    """SynQ总损失，困难样本仅用软标签"""
    # x_i: 滤波后的合成样本
    # y_i: 硬标签（one-hot）
    with torch.no_grad():
        prob_fp = F.softmax(model_fp(x_i), dim=-1)    # q(x_i; θ)
        difficulty = 1 - prob_fp[y_i.argmax()]          # δ = 1 - q_{y_i}(x_i; θ)

    # KL散度（始终计算，作为软标签的知识蒸馏）
    prob_q = F.softmax(model_q(x_i), dim=-1)
    L_KL = F.kl_div(F.log_softmax(model_q(x_i)), prob_fp, reduction='batchmean')

    # CAM对齐损失（始终计算）
    L_CAM = compute_CAM_loss(model_fp, model_q, x_i, y_i)

    # 交叉熵损失：仅对容易样本施加
    total_loss = L_KL + λ_CAM * L_CAM
    if difficulty <= τ:                                # 容易样本
        L_CE = F.cross_entropy(model_q(x_i), y_i)
        total_loss += λ_CE * L_CE
    # 困难样本: 跳过硬标签，仅使用KL+CAM

    return total_loss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 难度根据预训练模型对真实标签的预测概率定义；(2) 阈值 τ 通常设为 0.5（最优 trade-off），在 {0.5, 0.55, 0.6, 0.65, 0.7} 中搜索；(3) 对于 CIFAR-10，由于预训练模型错误率在更高难度（约 0.65）才开始上升，τ=0.7 为最优——说明不同数据集/模型的最佳 τ 取决于预训练模型的错误率-难度曲线（图3）；(4) 该策略无需额外计算开销（仅需在微调时判断 δ 是否 ≤ τ 决定是否施加 CE）；(5) 该策略的理论基础是：KL 散度蒸馏让量化模型模仿预训练模型的完整输出分布（含不确定性），而非仅拟合一个可能错误的硬标签。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---

## Synthesis-aware Fine-tuning（合成感知微调）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Synthesis-aware Fine-tuning 是 SynQ 论文提出的 Zero-shot Quantization 微调范式。与传统的 ZSQ 微调（对所有合成样本统一使用 KL 散度 + 交叉熵损失）不同，"合成感知"意味着微调过程充分认识到合成数据集与真实数据的三个关键差异并针对性解决：(1) 合成数据集含有高频噪声（频域分布与真实图像不同）→ 通过低通滤波去除噪声；(2) 合成数据集导致量化模型学习错误图像区域（off-target patterns）→ 通过 CAM 对齐蒸馏定位知识；(3) 合成数据集的硬标签对困难样本不可靠 → 困难样本仅用软标签。这三个策略共同构成了 synthesis-aware 的微调框架，使得量化模型在合成数据上的微调效果更接近在真实数据上的微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SynQ 的 Synthesis-aware Fine-tuning 完整流程（Algorithm 1）：
```
输入: 预训练模型θ, 超参数 n_ep, D0, λ_CAM, λ_CE, τ
输出: 量化模型参数 θ^q

// Step 1: 生成合成数据集（任意ZSQ方法均可）
{x_i}_{i=1}^N = generate_synthetic_dataset(θ, N=5120)

// Step 2: 合成感知微调
θ^q = RTN_quantize(θ)                                  // 量化初始化
{x_i^F} = gaussian_low_pass_filter({x_i}, D0)          // Idea 1: 低通滤波
for epoch in 1..n_ep:
    for x_i^F, y_i in {x_i^F, y_i}:
        // 计算KL、CAM损失（始终应用）
        L = KL(q(x_i^F; θ) || q(x_i^F; θ^q)) + λ_CAM * L_CAM(x_i^F; θ, θ^q)
        // 条件交叉熵（Idea 3）
        if δ(x_i^F, θ) ≤ τ:
            L += λ_CE * CE(q(x_i^F; θ^q), y_i)
        L.backward()
    optimizer.step()
return θ^q
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Synthesis-aware Fine-tuning 作为一个框架可集成到任意使用合成数据集的 ZSQ 方法中。SynQ 论文验证了其在 6 种 baseline（GDFQ, Qimera, AdaDFQ, IntraQ, HAST, TexQ）上的兼容性：在 Qimera W3A3 ResNet-18 上提升高达 31.17%p（1.17% → 32.34%），在 PTQ 方法 Genie 上平均提升 0.66%p。三个组件的贡献排序（ResNet-18 W3A3）：低通滤波（+5.80pp）> CAM 对齐（+4.63pp）> 困难样本软标签（+2.79pp）。三者组合达到最佳（基线 43.63% → 52.02%，+8.39pp）。微调时间开销仅 17.81%，且性能随合成数据集大小增长而提高。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning


## Video Diffusion Transformer (V-DM / VDiT, 视频扩散Transformer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Diffusion Transformer（V-DM，也称VDiT）是将扩散模型（Diffusion Model）的去噪过程与 Transformer 架构结合的生成式模型，专门用于视频生成任务。其核心结构为：输入为文本 prompt 经编码后的 embedding 和随机噪声隐变量，经过多层 Transformer Block 进行空间-时间联合建模，逐步去噪生成视频。与 Image Diffusion Transformer (I-DM，如 DiT/FLUX) 的关键区别在于 token 维度：I-DM 的 token 数 n = s（仅空间维度），V-DM 的 token 数 n = s × t（空间 × 时间维度），其中 t 随帧率（FPS）和视频时长线性增长。例如，6 秒视频在 8 FPS 下 t = 49，每帧数千 token，总 token 数可达数万。V-DM 使用全空间-时间注意力（Full Spatial-Temporal Attention），即每个 token 关注所有空间位置和所有帧的所有 token，计算复杂度 O((s×t)²)，导致极端的显存和计算需求。代表性模型包括 CogVideoX (2B/5B, Yang et al. 2024)、HunyuanVideo (13B, Kong et al. 2024)、Open-Sora 等。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 CogVideoX V-DM 的推理流程为例：
```
# V-DM 推理 Pipeline (DDIM sampling, T 步去噪)
x_T ~ N(0, I)  # 随机噪声初始化，x_T ∈ R^{n×d}, n = s×t
for timestep in [T, T-1, ..., 1]:
    # 1. 将当前 timestep 的 noisy latent 和 text embedding 送入 V-DM
    h_0 = concat(x_t, text_emb)  # 输入 embedding

    # 2. 通过 L 层 Transformer Block
    for block in V-DiT.blocks:
        # Multi-Head Self-Attention (Full Spatial-Temporal)
        # Q, K, V ∈ R^{n×d}, n = s×t
        A = softmax(Q @ K.T / sqrt(d_head))  # A ∈ R^{H×n×n}
        h = A @ V  # 每个 token 关注所有空间位置和所有帧

        # FFN (Feed-Forward Network)
        h = FFN(h)

    # 3. 预测噪声 ε_θ(x_t, t, text)
    ε_pred = output_projection(h)

    # 4. DDIM 去噪步
    x_{t-1} = sqrt(α_{t-1}) * (x_t - sqrt(1-α_t) * ε_pred) / sqrt(α_t) + sqrt(1-α_{t-1}) * ε_pred

return x_0  # 去噪后的隐变量 → VAE decoder → 视频帧
```
V-DM 中 token 数 n 极大，导致：(1) 单样本显存消耗高，校准预算（样本数 N）受限；(2) Attention 计算占主导（O(n²)），全序列 sparse attention 成为可挖掘的优化空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
V-DM 的主流实现基于 PyTorch + Diffusers 库。HuggingFace Diffusers 提供 CogVideoX 的预训练模型管道：`from diffusers import CogVideoXPipeline; pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b")`。量化部署方面，S²Q-VDiT 使用 ViDiT-Q 和 FlatQuant 的 CUDA kernel 实现 INT4 weight dequantize + INT6 activation online quantize 推理。V-DM 因其极长 token 序列和全注意力开销，是当前量化压缩最具挑战性和实用价值的场景之一。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---

## Block-wise Post-Training Quantization (逐块后训练量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Post-Training Quantization（逐块后训练量化）是一种 PTQ 优化策略，将模型按 Transformer Block（或 Layer）为单位分解，逐块进行量化参数校准和优化，而非一次性量化整个模型。其核心动机：全模型端到端 PTQ 的显存开销随模型规模线性增长（需同时存储 FP 和量化模型的所有中间激活），而 block-wise 方式每块仅需当前 block 的输入/输出，显存可控。流程为：对每个 Transformer Block（Attention + FFN），使用少量校准数据作为输入，以该 block 的 FP 输出为教师（target），优化量化 block 的量化参数（scale、rotation matrix、clipping threshold 等）以最小化输出 MSE。优化完成后将 block 的量化参数吸收（fold）到权重中，再处理下一 block。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 S²Q-VDiT 的 block-wise PTQ 流程为例：
```
# Block-wise PTQ Pipeline (基于 GPTQ weight quantizer)
calib_data = select_calibration_samples(model, N=40)  # SDS 筛选

for block_idx, block in enumerate(model.blocks):
    # 1. 收集当前 block 的 FP 输入激活
    X_fp = collect_inputs(calib_data, model, block_idx)  # X_fp ∈ R^{N×n×d}

    # 2. 预计算 attention map（用于 STD token 重加权）
    A = block.attention(X_fp)  # A ∈ R^{H×n×n}

    # 3. 计算 token 重要性权重
    for j in range(n):
        S_j = sum(A[h, i, j] for all h, i)
    λ_j = normalize(S_j, λ_min=0.5, λ_max=1.0)

    # 4. 逐 block 优化量化参数（30 样本，15 epochs）
    for epoch in range(15):
        for x in random_sample(X_fp, 30):
            y_fp = block_fp(x)     # FP block 输出
            y_q = block_q(x)       # 量化 block 输出
            # 重加权的 token-wise MSE
            L = (1/n) * sum(λ_j * ||y_fp[j] - y_q[j]||² for j in range(n))
            L.backward()
            # 更新量化参数（不同参数不同 lr）
            update(scale, lr=5e-3)
            update(rotation_matrix, lr=5e-3)
            update(clip_threshold, lr=5e-2)
    optimizer.step()

    # 5. 吸收量化参数到权重（weight folding），推理时无额外开销
    fold_quant_params_to_weight(block_q)

# 激活量化：推理时 online dynamic per-token quantization
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block-wise PTQ 是当前扩散模型 PTQ 的主流范式（Q-Diffusion, PTQ4DiT, ViDiT-Q, S²Q-VDiT 均采用）。实现上在 PyTorch 中逐 block 注册 hook 收集中间激活，用 AdamW + cosine LR scheduler 优化可学习量化参数。S²Q-VDiT 中 block-wise 优化使用 30 个校准样本（从 40 个 SDS 筛选样本中随机选取），每 block 训练 15 epochs。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---

## Hessian-aware Salient Data Selection (SDS, Hessian感知的显著数据选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hessian-aware Salient Data Selection（SDS）是 S²Q-VDiT 提出的面向 V-DMs 的校准数据选择策略。核心思想：在 PTQ 校准预算极端受限（仅几十样本）的情况下，随机采样导致量化性能方差极大（不同 seed 下 Imaging Quality 波动可达 ±1.76），需要一种基于数据"重要性"的筛选方法。SDS 从两个维度评估每个候选样本的重要性：(1) 扩散信息量（Diffusion Salience）C_diff = ||x_t - x_{t-1}||²/||x_t||²——相邻去噪步的隐变量变化越大，说明该 timestep 包含越多"新"的去噪信息；(2) 量化敏感度（Quantization Salience）C_quant = ||x_t^T x_t||_2——基于 Levenberg-Marquardt 近似的 Hessian 矩阵 X^T X 的 L2 范数，Hessian 特征值越大表示该样本对量化扰动越敏感。两个指标经 min-max 归一化到 [0,1] 后取乘积 C_sample = C̄_diff · C̄_quant 作为统一得分，乘积形式由算术-几何平均不等式保证只有当两个维度均高时才得高分，自然惩罚单维度强的样本。按 C_sample 降序选取 Top-N 构成校准集。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Hessian-aware Salient Data Selection (SDS)
# 输入: V-DM 模型 θ, 候选 prompt 集 P, 总去噪步数 T
# 输出: 校准数据集 D_calib (size N=40)

candidates = []
for prompt in P:  # 10 random prompts
    for t in range(1, T+1):
        x_t = model.get_latent(prompt, timestep=t)  # x_t ∈ R^{n×d}
        # (1) Diffusion Salience: 相邻步变化
        C_diff = ||x_t - x_{t-1}||² / ||x_t||²
        # (2) Quantization Salience: Hessian 近似
        C_quant = ||x_t^T @ x_t||_2  # Levenberg-Marquardt approx
        candidates.append((x_t, C_diff, C_quant))

# Min-max 归一化
C_diff_min, C_diff_max = min_max(C_diff for all)
C_quant_min, C_quant_max = min_max(C_quant for all)
for each (x_t, cd, cq) in candidates:
    cd_norm = (cd - C_diff_min) / (C_diff_max - C_diff_min)
    cq_norm = (cq - C_quant_min) / (C_quant_max - C_quant_min)
    score = cd_norm * cq_norm  # 联合得分

# 按 score 降序选 Top-N
D_calib = top_N_by_score(candidates, N=40)
```

Ablation 验证：SDS vs ATOP (All Timesteps from One Prompt) → Imaging Quality=52.95±0.69 vs 51.65±1.76；仅用 DS 或 QS 也能提升性能但联合使用(SDS)最佳；SDS 将随机种子方差从 ±1.76 降至 ±0.69。SDS 构造的校准集可集成到已有 PTQ 方法（如 PTQ4DiT + SDS 将 Aesthetic Quality 从 45.49 提升至 46.89）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SDS 的 Hessian 近似计算使用 Levenberg-Marquardt 近似 H^X = E[2 X^T X]，仅需一步矩阵乘法（X^T @ X）即可得到近似 Hessian，计算开销极小（CogVideoX-2B 仅增加 0.009 分钟、CogVideoX-5B 增加 0.015 分钟）。Attention map 在校准前用 FP 模型一次性前向计算并存储，校准时通过数据索引直接检索，不增加校准循环开销。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---

## Attention-guided Sparse Token Distillation (STD, 注意力引导的稀疏Token蒸馏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention-guided Sparse Token Distillation（STD）是 S²Q-VDiT 提出的 token 级量化损失重加权技术。核心观察：V-DMs 的全空间-时间注意力呈现显著稀疏模式——每层仅约 10% 的 token 拥有高注意力权重，其余 90% 对最终输出影响微弱。传统 block-wise PTQ 使用均匀加权 MSE L_quant = (1/n) Σ_j ||θ^f(x_j) - θ^q(x_j)||²，将所有 token 等权处理，浪费了有限校准数据对高影响力 token 的优化能力。STD 通过每个 Transformer Block 的多头注意力图 A ∈ R^{H×n×n} 计算每个 token j 的全局重要性：S_j = Σ_{h,i} A_{h,i,j}（token j 作为被关注对象从所有 query token 和所有 head 收到的注意力权重之和），经 min-max 归一化并映射到 [λ_min, λ_max] 得到权重 λ_j。最终损失 L_quant = (1/n) Σ_j λ_j · ||θ^f(x_j) - θ^q(x_j)||²，使高影响力 token（λ_j → λ_max=1）获得完整优化力度，低影响力 token（λ_j → λ_min）放松约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention-guided Sparse Token Distillation (STD)
# 在 block-wise PTQ 的每 block 优化中应用

# 1. 用 FP 模型前向计算当前 block 的 multi-head attention map
# A ∈ R^{H×n×n}, H = num_attention_heads
A = block.attention(Q_fp, K_fp, V_fp)  # softmax(Q @ K^T / sqrt(d))

# 2. 计算每个 token 的全局重要性得分
# S_j = sum of attention weight received by token j from ALL query tokens and ALL heads
S = zeros(n)
for h in range(H):
    for i in range(n):   # query token i
        for j in range(n):  # key/value token j
            S[j] += A[h, i, j]
# 优化: S_j = sum(A[:, :, j])  # 沿 head 和 query 维度求和

# 3. 归一化到 [λ_min, λ_max]
S_min, S_max = min(S), max(S)
λ = zeros(n)
for j in range(n):
    λ[j] = (S[j] - S_min) / (S_max - S_min) * (λ_max - λ_min) + λ_min

# 4. 重加权的量化损失
y_fp = block_fp(x)  # FP block 输出
y_q = block_q(x)    # 量化 block 输出
L_quant = (1/n) * sum(λ[j] * ||y_fp[j, :] - y_q[j, :]||² for j in range(n))
L_quant.backward()
```

超参数 λ_max=1（默认），λ_min=0.5 为最佳平衡点（控制低影响力 token 的放松程度）。Ablation 显示所有 λ_min ∈ {0.3, 0.5, 0.7} 均能提升性能，证明 STD 的鲁棒性。STD 可集成到已有 block-wise PTQ 方法：PTQ4DiT + STD 将 Aesthetic Quality 从 45.49 提升至 47.27。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Attention map 在校准前用 FP 模型一次性前向计算并预存储（per block per sample），校准时按数据索引直接检索，几乎不增加校准时间（CogVideoX-2B 仅从 2.82h 增至 2.84h）。推理时无需 attention map 或额外计算，零推理开销。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---

## Levenberg-Marquardt Hessian Approximation for Quantization (LM Hessian近似量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Levenberg-Marquardt (LM) Hessian 近似是量化领域中将权重 W 的量化误差期望 E[||XW^T - X(W+Δ)^T||²] 近似为 Δ(X^T X)Δ^T 的数学方法。其推导路径：(1) 将权重量化误差 Δ = Ŵ - W 代入 MSE 展开：E[||X W^T - X Ŵ^T||²] = E[||X Δ^T||²]；(2) 使用二阶 Taylor 展开（在 Δ=0 处）：≈ Δ g^X + ½ Δ H^X Δ^T，其中 g^X 为梯度、H^X 为 Hessian 矩阵；(3) 对已训练模型 g^X = 0（最优性条件），且 H^X = E[2 X^T X]（Levenberg-Marquardt 近似，用一阶 Jacobian 的外积替代二阶 Hessian）；(4) 最终简化为 E[Δ (X^T X) Δ^T]，即量化误差由输入激活 X 的自相关矩阵 X^T X 加权。LM 近似的关键优势是计算高效——仅需一步矩阵乘法 X^T @ X，无需完整的二阶导数计算（后者在 LLM/V-DM 规模上不可行）。该近似最早由 Optimal Brain Compression (Frantar & Alistarh, NeurIPS 2022) 和 GPTQ (Frantar et al., 2022) 引入量化领域，S²Q-VDiT 进一步将其用于构造量化敏感度指标 C_quant = ||x_t^T x_t||_2 作为校准数据选择的一个维度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# LM Hessian 近似在 SDS 中的应用
# 输入: 候选校准样本的隐变量 x_t ∈ R^{n×d}

# Step 1: 计算 LM 近似 Hessian
# H_X ≈ 2 * X^T X,  X ∈ R^{n×d}  (batch 维度展开)
H_approx = x_t.T @ x_t  # H_approx ∈ R^{d×d}, O(n·d²)

# Step 2: 提取量化敏感度（Hessian 的 L2 范数）
C_quant = ||H_approx||_2  # 即 ||x_t^T x_t||_2
# C_quant 越大 → 样本对量化扰动越敏感 → 更应在校准中被覆盖

# Step 3: 归一化并与其他指标联合
C_quant_norm = (C_quant - C_quant_min) / (C_quant_max - C_quant_min)
```

LM 近似在量化中的另一核心用途是 GPTQ 的逐列误差补偿：H_inv = Cholesky((H_approx + λI)^(-1))，利用 H_approx 的 Cholesky 分解引导量化误差沿未量化列进行补偿分配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LM Hessian 近似在 PyTorch 中的实现极为简洁：`H_approx = x_t.T @ x_t`（一行代码）。对于大规模模型，通常使用 group-wise 或 block-wise 方式分批计算以控制显存。S²Q-VDiT 中计算 LM 近似的额外开销极小——CogVideoX-2B 的 SDS 构造仅增加 0.009 分钟（7.708 → 7.717 min），HunyuanVideo-13B 增加 0.003 分钟（19.505 → 19.508 min）。计算完成后丢弃 H_approx，仅保留逐样本的标量 C_quant 得分用于排序选择。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---

## Compact Tokenizer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Compact Tokenizer（紧凑分词器）是一种针对小语言模型（Tiny Language Model）优化的词表压缩技术。其核心思想是：大语言模型（如7B+参数）使用的分词器通常包含100k+词汇量（vocabulary size），以保证对多种语言和领域的高覆盖率。但对小模型（≤1.5B参数），embedding层和输出head层的参数量在总参数量中占比极大——例如对于12层/2048宽的1B模型，100k词表的embedding+head层占比高达38.19%，远超大模型中约10%的比例。Compact Tokenizer通过统计分析发现训练语料中词频呈现长尾分布（top-48k词汇覆盖97.86%的1.6T tokens语料），识别并移除低频冗余词汇（bottom 52k+词汇仅覆盖不到3%的语料），将词表从100k压缩至48k，使embedding+head参数占比降至18.07%，释放约20%参数给Transformer本体。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在算法Pipeline中，Compact Tokenizer的训练和使用流程如下：

```
输入: 原始大词表 V_large (|V|=100883), 大语料 D (1.6T tokens)

# Phase 1: 词频分析与词表压缩
freq = Counter()  # 统计每个vocab token的出现频率
for sample in D:
    for token_id in tokenize(sample, V_large):
        freq[token_id] += 1

# 按频率降序排序，计算累积覆盖率
sorted_vocab = sort_by_freq_desc(V_large, freq)
cum_freq = 0; total = sum(freq.values())
for k, v in enumerate(sorted_vocab):
    cum_freq += freq[v]
    coverage = cum_freq / total
    if coverage >= 0.9786:  # 目标覆盖率97.86%
        V_compact_size = k + 1  # 得到 k=48k
        break

# Phase 2: 用SentencePiece BPE在新词表大小下重新训练tokenizer
V_compact = train_sentencepiece_bpe(
    corpus=D,
    vocab_size=48000,
    character_coverage=0.9995
)
# 输出: 新的BPE编码规则 → token_id映射

# Phase 3: 小模型使用Compact Tokenizer
# Embedding层: W_emb ∈ R^{48000 × d_model}  (原来: R^{100883 × d_model})
# LM Head层:  W_head ∈ R^{d_model × 48000}  (原来: R^{d_model × 100883})
# 参数占比从 38.19% 降至 18.07%
```

关键技术公式：
- PEHL (Proportion of Embedding and Head Layers) = `(2 × V × d_model) / total_params`
- 推荐 PEHL < 20%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方面：
1. **初始词表来源**：从已有大模型（如PanGu-π-7B）继承tokenizer作为初始词表。
2. **语料覆盖分析**：在训练语料（1.6T tokens）上进行频率统计，计算top-k词汇的累积覆盖率，确定最优k值（本论文k=48k, 覆盖率97.86%）。
3. **Tokenizer实现**：使用SentencePiece库的BPE算法，设置vocab_size=48000重新训练。
4. **超参数推荐**：词表大小应保证累积覆盖率>90%，同时PEHL<20%。过小词表（如8k，覆盖率<70%）导致性能下降。
5. **相关方法对比**：MiniMind使用6400词表的Custom BPE；AG-BPE通过注意力引导打分机制实现16k词表、3.77×压缩比；Compact框架通过后剪枝（post-hoc pruning）同时剪除低频词汇和FFN中间通道。

使用场景：
- 构建1B及以下参数量级的SLM/TLM模型
- 边缘设备部署场景，对参数量和显存严格受限
- 双语/多语言小模型的tokenizer设计
- 从大模型蒸馏到小模型时作为首步优化

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

## Parameter Inheritance

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Parameter Inheritance（参数继承）是一种从小模型初始化的技术，核心思想是将大语言模型（如7B）学到的权重和表征能力"传递"给要训练的小模型（如1B），让小模型从大模型的参数出发开始训练，而非从随机初始化开始。这不同于知识蒸馏（knowledge distillation）——蒸馏需要teacher模型在训练过程中持续提供监督信号；而参数继承是一次性的初始化操作，继承后小模型独立训练。该方法包含两个阶段：(1) Layer Selection——从小模型层数少、大模型层数多的约束出发，通过layer skipping实验识别并保留大模型中对性能最关键的首尾层，移除冗余的中间层；(2) Intra-layer Parameter Selection——在保留的各层内，通过可学习二值掩码（learnable binary mask）自动识别并选择对任务最重要的神经元/参数，形成小模型的初始权重。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 大模型 W_large（PanGu-π-7B, L_large层），目标架构 A_small（L_small层, d_small宽）

# Stage 1: Layer Selection
perf_drop = []  # 每层的性能下降
for layer_i in range(L_large):
    # Layer skipping实验：跳过layer_i，评估下游任务性能
    model_skipped = copy(W_large); model_skipped.skip_layer(layer_i)
    perf_i = evaluate(model_skipped, [ARC-E, HellaSwag, C3])
    perf_drop.append(perf_base - perf_i)

# 发现：前2-3层和最后几层perf_drop大（关键层），中间层perf_drop小（冗余）
# 选择策略：保留前k_begin层 + 中间均匀采样 + 保留最后k_end层
selected_layers = [0,1,2] + uniform_sample(middle_layers, L_small - k_begin - k_end) + [L_large-3, L_large-2, L_large-1]

# Stage 2: Intra-layer Parameter Selection via Learnable Mask
for each selected_layer l:
    W = W_large[l]  # 该层权重矩阵, shape (d_out, d_in)
    # 初始化可学习参数 α ∈ R^{d_out × d_in} (log-probabilities)
    α = init_normal(mean=2.0, std=0.01)  # 偏向mask=1的初始化
    
    # Gumbel-Sigmoid可微二值化
    for step in mask_training_steps:
        u = uniform(0, 1, shape=(d_out, d_in))
        g = -log(-log(u + ε) + ε)  # Gumbel noise
        logits = α + g
        M = sigmoid(logits / τ)     # temperature τ anneal from 1.0 to 0.1
        # 前向传播: h = (W ⊙ M) @ x
        # 损失: L = L_task(h, y) + λ * ||M||_1  (稀疏正则)
        α = α - η * ∇_α L  # 更新mask参数（STE直通估计器）
    
    M_binary = (M > 0.5).float()  # 最终二值化
    # 提取子矩阵: 按mask=1的行/列选取参数
    W_small[l] = extract_submatrix(W, M_binary)

# Stage 3: 用小模型架构初始化
θ_init = assemble_model(selected_layers, W_small, A_small)
# 开始小模型训练
θ_final = train(θ_init, data=1.6T tokens)
```

结果表明：Learnable Mask (Avg=48.08) > Taylor (47.90) > L2 (47.00) > L1 (46.06) > Random Init (42.06)，参数继承是最有效的单一优化组件（+3.26贡献）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现与使用：
1. **Layer Skipping实验**：对多个大模型(LLaMA2-7B/13B, InternLM-7B, PanGu-π-7B)进行skip-1/skip-2/skip-3层实验，发现普适规律——首尾层关键、中间层冗余。
2. **Mask学习方法**：使用Gumbel-Sigmoid重参数化技巧（通过温度τ控制离散化程度，训练中从τ=1.0逐步退火至0.1）实现可微的二值mask训练；反向传播使用Straight-Through Estimator（STE）处理不可微的取整操作。
3. **相关技术对比**：
   - ParaKnowTransfer (ICLR 2024)：参数敏感度+LoRA注入实现7B→13B参数迁移
   - GeneLLM/Learngene：从MoE模型中提取1.25%参数初始化小模型，保留80%+性能
   - CoMe (NeurIPS)：通过层拼接（而非选择+裁剪）逐层压缩
4. **适用前提**：需要同架构系列的大模型checkpoint作为初始化来源（如LLaMA→TinyLLaMA、PanGu-π-7B→PanGu-π-1B Pro）

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

## Learnable Binary Mask

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Learnable Binary Mask（可学习二值掩码）是一种数据驱动的参数重要性评估与选择方法。在神经网络剪枝和参数继承场景中，传统的启发式重要性度量（L1-norm, L2-norm, Taylor expansion）基于固定的数学公式判断哪些权重重要，而这些指标可能与实际任务需求不完全一致。Learnable Binary Mask将"哪些参数重要"本身作为一个可优化的问题：为每个待选择的权重引入一个可学习的二值mask参数，通过任务损失反向传播来自动学习哪些权重应该被保留（mask=1）或丢弃（mask=0）。训练完成后，仅保留mask=1对应的权重作为小模型的初始化参数。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# 输入: 大模型某层权重 W ∈ R^{d_out × d_in}, 输入 x, 目标压缩比
# 输出: 压缩后的权重 W_compressed

# 1. 初始化可学习mask参数
α = nn.Parameter(torch.randn(d_out, d_in) * 0.01 + 2.0)  # 正偏置使初始mask接近1

# 2. Gumbel-Sigmoid 可微二值化
def gumbel_sigmoid(α, τ, training):
    if training:
        # 采样Gumbel噪声: g = -log(-log(u + ε) + ε), u ~ Uniform(0,1)
        u = torch.rand_like(α).clamp(min=1e-8, max=1-1e-8)
        g1 = -torch.log(-torch.log(u))
        g2 = -torch.log(-torch.log(1 - u))
        # Gumbel-Sigmoid: M = σ((α + g1 - g2) / τ)
        logits = (α + g1 - g2) / τ
    else:
        logits = α / τ  # 推理时不用噪声
    return torch.sigmoid(logits)

# 3. 训练循环
optimizer = AdamW([α], lr=1e-3)
τ_start, τ_end = 1.0, 0.1  # temperature annealing
for step in range(max_steps):
    τ = τ_start * (τ_end/τ_start) ** (step/max_steps)  # 指数退火
    
    M = gumbel_sigmoid(α, τ, training=True)
    # 前向: 被mask的权重输出≈0
    h = (W.detach() * M) @ x  # W固定，只训练mask
    
    # 损失 = 任务损失 + 稀疏正则
    L_task = cross_entropy(classifier(h), y)
    L_sparse = λ * M.mean()  # 推动mask稀疏化
    L = L_task + L_sparse
    
    # STE反向传播: 梯度通过不可微的M传递
    L.backward()  # ∇_α L 通过STE计算
    optimizer.step()

# 4. 最终二值化并提取
M_final = (gumbel_sigmoid(α, τ=0.01, training=False) > 0.5).float()
# 提取: 选取mask=1的行索引和列索引
row_idx = M_final.sum(dim=1) > 0  # 保留有任意输入连接的行
col_idx = M_final.sum(dim=0) > 0  # 保留有任意输出连接的列
W_compressed = W[row_idx][:, col_idx]
```

核心机制：
- **Gumbel-Softmax/Straight-Through Estimator**：Gumbel噪声使采样过程可微（前向离散采样，反向梯度直通），temperature τ控制mask的离散程度（τ→0时M趋于0/1二值）
- **稀疏正则**：L1正则λ·Σ|M|推动mask稀疏化，λ控制压缩率
- **与权重解耦**：只训练mask参数，不修改原始大模型权重，保证预训练知识不被破坏

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **MLPruner (PeerJ 2025)**：为每个卷积滤波器关联可学习二值mask，STE处理不可微取整，实现对ResNet/VGG的无损剪枝（54.8% FLOPs reduction, 93.31% top-1 on CIFAR-10）。
2. **Piggyback (ECCV 2018)**：在固定预训练权重上学binary mask，实现单网络适配多任务（每任务仅1 bit/参数的存储开销）。
3. **SCL (IEEE TNNLS 2023)**：将权重重参数化为weight ⊙ binary_mask，证明STE代理梯度须为正数才能收敛，提出LeakyReLU/Softplus/identity STE作为有效选项。
4. **本论文的使用**：在参数继承的Intra-layer阶段使用Gumbel-Sigmoid learnable mask识别PanGu-π-7B中的重要神经元，相比L1/L2/Taylor固定标准提升1-2个Avg百分点。
5. **优势**：端到端可训练、任务感知（task-aware）、无需手工设计重要性度量标准。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

## Multi-round Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-round Training（多轮训练）是一种针对小语言模型的训练策略，解决小模型在单轮大规模数据训练中面临的灾难性遗忘（catastrophic forgetting）问题。传统大模型训练通常对所有数据仅遍历一次（single-pass/one-epoch training），因为大模型容量大、学习新知识时不易覆盖旧知识。但小模型（≤1.5B参数）容量有限，在1.6T tokens的序列训练中，后期数据会严重覆盖早期的学习成果——论文通过"重新计算已训练数据的loss"实验验证了这一点：早期数据的loss从训练时的低值大幅反弹。Multi-round Training的核心思想是：第一轮全量训练后，基于每batch历史loss计算采样概率（困难样本被采样概率更高），第二轮有放回地采样部分数据（如50%）进行继续训练，强化对困难样本的学习。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 训练数据 D, 模型参数 θ, 训练轮数 R=2, 采样率 r=0.5

# Round 1: 全量标准训练
K = 8  # 将数据均匀分为K个part
D_parts = random_split(D, K)  # D = {P1, P2, ..., PK}
loss_history = {k: [] for k in range(K)}  # 记录每个batch的loss

for k in range(K):  # 顺序训练每个part
    for batch in P_k:
        l = compute_loss(θ, batch)
        loss_history[k].append(l)
        θ = AdamW_step(θ, l)  # 标准优化步骤

# 验证forgetting: 重新计算Round 1早期数据在最终θ上的loss
recomputed_losses = {}
for k in range(K):
    L_k = []
    for batch in P_k:
        L_k.append(compute_loss(θ_final_round1, batch))
    recomputed_losses[k] = mean(L_k)
# 观察: 早期part的recomputed loss远高于对应训练时的loss → forgetting证据

# Round 2: Loss-guided数据采样
D_round2 = []
for k in range(K):
    L = loss_history[k]  # batch-wise losses for part k
    # Softmax归一化: p_i ∝ exp(l_i)
    probs = softmax(L)  # p_i = exp(l_i) / Σ_j exp(l_j)
    # 按概率采样 r*N 个batch
    N_k = len(P_k)
    sample_count = int(r * N_k)
    sampled_indices = multinomial_sample(probs, sample_count)
    D_round2.extend([P_k[i] for i in sampled_indices])

# Round 2: 在采样数据上继续训练
shuffle(D_round2)
for batch in D_round2:
    l = compute_loss(θ, batch)
    θ = AdamW_step(θ, l)

输出: 最终模型参数 θ (Round 2)
```

关键公式：
- 采样概率: p_i = exp(l_i) / Σ_{j=1}^{N_k} exp(l_j)，其中l_i为第i个batch在Round 1中记录的loss
- 效果：Round 2 r=50% Avg=54.46 vs Single Round Avg=51.61 (+2.85)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **采样率选择**：论文实验显示r=50%可获得r=100%的~94%收益，推荐r=50%作为效率与性能的平衡点。
2. **训练轮数**：Round 2收益最大(+2.85)，Round 3开始饱和(Avg几乎不变)，推荐总共两轮。
3. **数据组织**：Round 1中数据按时间顺序分为K=8个part，每个part独立计算采样概率（保证数据多样性和loss分布的局部性）。
4. **相关技术对比**：
   - mix-cd (2025): 优先采样"collateral damage"样本（之前正确、当前错误的样本），通过复用已有inference避免额外前向计算
   - Self-Generated Replay (2026): 使用模型自身生成的历史数据伪样本作为replay数据，KL散度替代NTP损失
   - Forecasting Forgetting (2024): 使用NTK-style kernel预测哪些样本会被遗忘，仅replay预测遗忘的样本
5. **适用场景**：小模型（≤3B参数）在大规模语料上的预训练；模型容量不足以单轮充分学习时。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

## Group Query Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Group Query Attention (GQA，分组查询注意力) 是介于标准多头注意力(MHA)和Multi-Query Attention(MQA)之间的一种注意力机制变体，由Ainslie et al.在2023年提出。其核心思想是将query head分成g组（1 < g < h），每组内的多个query head共享同一个key-value head对。相比MHA（h个KV heads，每个query head独立使用一对KV），GQA仅需g个KV heads（g << h），大幅减少推理时需要缓存的Key和Value状态量。相比MQA（所有query head共享唯一一对KV），GQA通过保留多个KV组来维持更好的建模质量。GQA的KV Cache大小是MHA的(g/h)倍，在典型配置下（如h=32, g=8）可减少75%的KV Cache内存占用。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 输入序列 x ∈ R^{seq_len × d_model}, 总head数 h, KV组数 g
参数: Q_proj ∈ R^{d_model × h·d_head}, K_proj ∈ R^{d_model × g·d_head}, V_proj ∈ R^{d_model × g·d_head}

# 1. 投影
Q = x @ Q_proj  # (seq_len, h*d_head)
K = x @ K_proj  # (seq_len, g*d_head)
V = x @ V_proj  # (seq_len, g*d_head)

# 2. 拆分head
Q = reshape(Q, [seq_len, h, d_head])  # (seq_len, h, d_head)
K = reshape(K, [seq_len, g, d_head])  # (seq_len, g, d_head)
V = reshape(V, [seq_len, g, d_head])  # (seq_len, g, d_head)

# 3. GQA计算: 每g组内的h/g个query共享同一个KV
heads_per_group = h // g
O = []
for group_idx in range(g):
    # 该组内的query heads
    Q_group = Q[:, group_idx*heads_per_group : (group_idx+1)*heads_per_group, :]  # (seq_len, heads_per_group, d_head)
    
    # 该组共享的单一KV head
    K_group = K[:, group_idx, :]  # (seq_len, d_head) → 广播为 (seq_len, 1, d_head)
    V_group = V[:, group_idx, :]  # (seq_len, d_head)
    
    # 标准Scaled Dot-Product Attention
    scores = Q_group @ K_group.unsqueeze(1).transpose(-2, -1) / sqrt(d_head)
    attn = softmax(scores, dim=-1)
    O_group = attn @ V_group.unsqueeze(1)  # (seq_len, heads_per_group, d_head)
    O.append(O_group)

O = concat(O, dim=1)  # (seq_len, h, d_head)
O = reshape(O, [seq_len, h*d_head])
output = O @ O_proj  # (seq_len, d_model)

# 4. KV Cache (推理时)
# MHA: Cache_size = 2 * h * seq_len * d_head  (无共享)
# GQA: Cache_size = 2 * g * seq_len * d_head  (每层有g对KV，减少(h/g)倍)
# MQA: Cache_size = 2 * 1 * seq_len * d_head  (最极端)
```

从MHA转换为GQA（本论文使用的方法）：
```python
# 通过mean-pooling将MHA的KV投影矩阵转换为GQA
# 输入: MHA模型有h个KV heads
# 输出: GQA模型有g个KV heads，每组mean-pool原h/g个heads
for group_idx in range(g):
    start = group_idx * heads_per_group
    end = start + heads_per_group
    K_GQA[group_idx] = mean(K_MHA[start:end], dim=0)  # 均值池化
    V_GQA[group_idx] = mean(V_MHA[start:end], dim=0)
# 然后用5%原始数据继续训练
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **生产模型中的GQA配置**：LLaMA 3.1系列（8B: h=32/g=8, 70B: h=64/g=8, 405B: h=128/g=8），Qwen 2.5 7B (h=28/g=4)，Mistral (h=32/g=8)，Gemma 3 12B (h=16/g=4)。
2. **KV Cache减少**：g=8时减少87.5%的KV Cache，对长上下文推理（32K-128K+）至关重要。
3. **从MHA转换**：可通过mean-pooling将MHA模型的KV heads转换为GQA，然后用~5%原始数据继续训练（本论文在PanGu-π-1.5B Pro上验证，GQA版本(1.4B参数)保持原MHA版本(1.5B参数)的~92-97%benchmark性能）。
4. **适用场景**：内存受限的边缘设备部署、长文本输入场景、需要降低推理延迟的在线服务。
5. **其他变体**：Cross-Layer Attention (CLA)进一步跨层共享KV；Cost-Optimal GQA根据上下文长度动态调整head数；QCQA通过进化算法实现非均匀分组。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

## SentencePiece BPE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SentencePiece BPE（Byte-Pair Encoding with SentencePiece）是一种子词级别的文本分词方法。SentencePiece是Google开发的**纯数据驱动**、**语言无关**的分词器框架，将输入文本直接视为Unicode字符序列（无需预分词），通过BPE算法从语料中学习最优的子词合并规则。BPE（Byte-Pair Encoding）的核心思想是：从字符级开始，反复统计相邻token pair的共现频率，贪心地合并最高频pair为新的token，直到词表达到预设大小。这使模型能够处理任意未见过的词（通过子词组合），平衡语义粒度和词表大小。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# BPE学习过程 (SentencePiece内部)
输入: 训练语料 D (raw text), 目标词表大小 vocab_size
输出: 子词合并规则 + token→id映射

# 1. 初始化: 每个Unicode字符 + 空格为基本unit
vocab = {char: count for char in all_chars_in_D}  # 字符级统计
# SentencePiece特性: 空格用▁(U+2581)表示，保持可逆

# 2. BPE迭代合并
while len(vocab) < vocab_size:
    # 统计所有相邻token pair的频率
    pair_freq = defaultdict(int)
    for word_freq, word_tokens in D_tokenized:
        for i in range(len(word_tokens) - 1):
            pair_freq[(word_tokens[i], word_tokens[i+1])] += word_freq
    
    if not pair_freq:
        break  # 无可合并pair
    
    # 贪心选择最高频pair
    best_pair = max(pair_freq, key=pair_freq.get)
    new_token = best_pair[0] + best_pair[1]  # 合并为新的subword
    
    # 更新: 添加新token，在原序列中替换出现位置
    vocab[new_token] = pair_freq[best_pair]
    D_tokenized = merge_pair(D_tokenized, best_pair, new_token)

# 输出: (token, token_id)的映射表
# 例: "自然语言处理" → ["▁自然", "语言", "处理"] → [1234, 567, 890]

# 分词推理 (inference)
def encode(text):
    # 按最长匹配原则应用BPE合并规则
    chars = list(text)
    tokens = apply_bpe_rules(chars, merge_rules)  # 按学习顺序应用规则
    return [vocab_to_id[t] for t in tokens]
```

关键公式：
- BPE合并准则: max_{pair} freq(pair) = freq(token_a, token_b) over all token sequences in corpus
- SentencePiece Loss: 基于unigram language model，优化p(x) = Π_i p(x_i) under subword segmentation

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **SentencePiece安装与基本用法**：
```python
import sentencepiece as spm
# 训练
spm.SentencePieceTrainer.train(
    input='corpus.txt', model_prefix='m',
    vocab_size=48000, character_coverage=0.9995,
    model_type='bpe', num_threads=16
)
# 使用
sp = spm.SentencePieceProcessor(model_file='m.model')
tokens = sp.encode("Hello world!", out_type=str)  # ['▁Hello', '▁world', '!']
ids = sp.encode("Hello world!")  # [123, 456, 78]
```

2. **SentencePiece vs 传统BPE**：SentencePiece将输入视为原始字符序列（无预分词步骤），消除语言特定的tokenization假设（如英语依赖空格分词），真正语言无关；使用▁元字符标记词边界，保证tokenization可逆（detokenization无歧义）。

3. **Typical训练参数**：
   - `vocab_size`：词表大小，大模型通常100k+，小模型推荐32k-48k
   - `character_coverage`：字符覆盖率，推荐0.9995（覆盖99.95%的Unicode字符）
   - `model_type`：bpe（最常用）或unigram
   - `num_threads`：并行训练线程数

4. **本论文中的使用**：基于PanGu-π-7B的100k BPE tokenizer，在1.6T tokens语料上进行频率分析，然后使用SentencePiece BPE算法训练48k紧凑tokenizer。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models

## Gumbel-Softmax

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gumbel-Softmax 是由 Jang, Gu, Poole 在 ICLR 2017 提出的可微分离散采样方法，允许神经网络通过离散分类变量进行端到端梯度反向传播。其核心思想：在 forward pass 中，向未归一化的类别 logits $g_i$ 添加标准 Gumbel 分布噪声 $\sigma_i \sim \text{Gumbel}(0,1)$，然后取 argmax 获得离散选择 $\theta = \arg\max_i (g_i + \sigma_i)$；在 backward pass 中，使用 softmax 的连续近似概率 $p_i = \frac{\exp((g_i + \sigma_i)/\tau)}{\sum_j \exp((g_j + \sigma_j)/\tau)}$ 计算梯度，其中温度系数 $\tau$ 控制 softmax 的锐度（$\tau \to 0$ 时接近 one-hot，$\tau \to \infty$ 时趋向均匀分布）。这种"前向离散 + 反向连续"的 Straight-Through Estimator (STE) 模式使得网络可以端到端学习离散决策。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 Granular-DQ 的 Granularity-Bit Controller (GBC) 中，Gumbel-Softmax 用于为每个图像 patch 选择量化的 bit-width：

```
# GBC 中 Gumbel-Softmax bit-width 选择流程
输入: patch 特征 → 线性层 → 门控 logits g ∈ R^N (N=3, 候选 bit: [4,6,8])

# Forward pass (离散采样)
σ = sample_gumbel(shape=g.shape)  # σ_n = -log(-log(U)), U ~ Uniform(0,1)
θ = argmax(g + σ)                  # 离散门控索引 ∈ {1, 2, 3}

# 计算门控分数 (连续近似, 用于梯度传播)
p_i = exp((g_i + σ_i) / τ) / Σ_n exp((g_n + σ_n) / τ)   # τ=1

# Backward pass (STE): 前向使用离散 θ, 反向使用连续 p 的梯度
∇_{W_g} L = ∂L/∂p · ∂p/∂g · ∂g/∂W_g
```

GBC 接收 D 层多粒度特征融合后的通道统计量 S，线性层 W_g ∈ R^{(N×D)×N} 映射为 N 维门控 logits，对每个 patch 独立采样门控分数 p_i（衡量 patch 对整张图像的贡献比例），映射到对应 bit code。GBC 置于 SR 网络输入端，仅引入可忽略的计算开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

1. **温度退火**：训练初期用较大 τ (1-5) 鼓励探索，后期降低 τ (0.1-0.5) 稳定决策。
2. **Gumbel(0,1) 采样**：通过逆变换 $\sigma = -\log(-\log(U))$, $U \sim \text{Uniform}(0,1)$。
3. **PyTorch 内置支持**：`F.gumbel_softmax(logits, tau=1.0, hard=True)` 实现 hard Gumbel-Softmax（forward 返回 one-hot，backward 用 softmax 梯度）。
4. **应用场景**：动态量化 bit-width 选择、NAS 架构搜索、动态 routing、离散隐变量生成模型 (VAE)、RL 动作选择。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

## Feature Average Bit-width (FAB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Feature Average Bit-width (FAB) 是动态量化方法中衡量量化效率的核心指标，定义为测试数据集上所有特征图在所有 patch 上的平均 bit-width。FAB 越低量化越激进，计算和存储开销越小。与固定比特量化（如 PAMS 的恒定 8-bit）不同，动态量化的 FAB 随图像内容自适应变化，反映实际使用的平均精度水平。计算方式：

$$\text{FAB} = \frac{1}{|\mathcal{D}|} \sum_{X \in \mathcal{D}} \frac{1}{M} \sum_{i=1}^{M} b_i$$

其中 $\mathcal{D}$ 为测试集，$M$ 为图像 $X$ 的 patch 数，$b_i$ 为第 $i$ 个 patch 的 bit-width。

从算法pipeline角度拆解术语，给出具体例子。

在 Granular-DQ 评估中，FAB 与 PSNR/SSIM 构成 trade-off：EDSR ×4 SR 在 Urban100 上，全精度 FAB=32.00 (PSNR 26.03dB)，PAMS FAB=8.00 (26.01dB)，CADyQ FAB=6.09 (25.94dB)，Granular-DQ FAB=4.97 (26.01dB)。Granular-DQ 以最低 FAB 实现与全精度相当的 PSNR。FAB 与 BitOPs（bit-weighted operations）配合，FAB 侧重"平均精度"，BitOPs 侧重"实际计算量"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FAB 是评估指标而非实现组件，通过统计测试集所有 patch 的 bit 分配计算。被 CADyQ、CABM、AdaBM、RefQSR 等动态量化方法广泛采用作为统一的效率度量。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

## Granularity-Bit Controller (GBC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Granularity-Bit Controller (GBC) 是 Granular-DQ 的核心组件，对输入图像的每个 patch 进行粗到细的多粒度层次分析，自适应分配量化 bit-width。设计哲学：不同 patch 的贡献比例不同——细粒度特征揭示局部纹理复杂度，粗粒度特征表达整体场景结构——应根据贡献比例分配计算精度。

GBC 工作流程：(1) 编码器 $\mathcal{E}$ 对输入 X 提取 D 层多粒度特征 $\mathbf{Z} = \{Z_1, ..., Z_D\}$（D-1 次下采样，$Z_1$ 最细粒度，$Z_D$ 最粗粒度）；(2) 所有粒度特征 GroupNorm → 平均池化至 $Z_D$ 分辨率 → concat → GAP 得到通道统计量 $\mathbf{S}$；(3) 线性层 $\mathbf{W}_g \in \mathbb{R}^{(N \times D) \times N}$ 作用于 $\mathbf{S}$ 生成门控 logits $\mathbf{G}$；(4) Gumbel-Softmax 为每个 patch 采样门控分数 $p_i$（patch 贡献比例），映射到候选 bit code {4,6,8}。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程。

```
# GBC 伪代码
输入: 图像 X, 编码器 E, D 层多粒度

# 多粒度特征提取与融合
Z = E(X)  # Z = [Z_1,...,Z_D], Z_1 最细粒度, Z_D 最粗粒度
Z_hat = [GroupNorm(Z_d) → AvgPool(Z_D_res) for Z_d in Z]
Z_cat = concat(Z_hat, dim=channel)

# 通道统计量 + Bit 分配
S = GlobalAvgPool(Z_cat)
G = Linear(W_g)(S)  # W_g: (D×C) → N (N=3, bit codes [4,6,8])

for each patch X_i:
    σ = sample_gumbel(N)
    θ_i = argmax(G[i] + σ)        # 离散门控索引
    p_i = softmax((G[i]+σ)/τ)[θ_i] # 门控分数
    b_i = bit_codes[θ_i]           # → {4, 6, 8}

输出: 每个 patch 的 bit-width b_i (所有层共享)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GBC 使用 PyTorch 的 `F.gumbel_softmax` 实现可微分离散采样。训练时端到端优化（仅 L1 loss），推理时直接 argmax 确定 bit-width。GBC 置于 SR 网络最前端，对任何 CNN/Transformer SR 架构即插即用。各层对同一 patch 使用相同 bit-width（layer-invariant），避免 CADyQ 逐层 bit selector 对层间关系的破坏。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

## Entropy-to-Bit (E2B) Mechanism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Entropy-to-Bit (E2B) 机制是 Granular-DQ 的第二阶段组件，对 GBC 阶段分配了高 bit-width (8-bit) 的 patch 进行细粒度 bit-width 再调整，进一步降低平均 bit-width。理论基础来自 Shannon 信息论：图像 patch 的像素熵反映信息密度和像素分布复杂度，低熵 patch（平坦区域、纯色背景）对量化精度要求低，可用更低 bit-width。

E2B 流程：(1) 训练集所有 LR patch 计算像素熵 $\mathcal{H} = -\sum_i \mathcal{P}(x_i) \log \mathcal{P}(x_i)$，$\mathcal{P}(x_i)$ 由 Gaussian 加权核密度估计得到；(2) 熵值升序排列为 $\mathbf{H}$；(3) 分位数阈值（$t_1=0.5, t_2=0.9$）将 $\mathbf{H}$ 划分为子区间，映射到候选 bit codes [4, 5, 8]；(4) 对 GBC 分配了 8-bit 的 patch，据其熵值 $E$ 查找落入区间确定最终 bit-width。

从算法pipeline角度拆解术语，给出具体计算过程。

```
# E2B + ATC
# 预处理: 训练集熵分布
for each LR_patch:
    r_i = x_i - bin_values
    P(x_i) = Σ exp(-r_i²/(2σ²)) / (ΣΣ exp(-r_i²/(2σ²)) + ε)
    H.append(-Σ P(x_i) log(P(x_i)))
H = sort(H)  # 升序

# 阈值 (分位数)
I_t1, I_t2 = ceil(M*0.5), ceil(M*0.9)  # t1=0.5, t2=0.9

# Bit 适配 (GBC 阶段 b_i==8 的 patch)
if E <= H_t1: b_i = 4     # 低熵 → 低精度
elif E <= H_t2: b_i = 5   # 中熵 → 中等精度
else: b_i = 8              # 高熵 → 高精度

# ATC: 首 epoch EMA 动态校准
t^(j) = t^(j-1) · 0.9997 + Norm(E) · (1-0.9997)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

E2B 仅需首 epoch 计算（LR 样本跨 epoch 一致）。候选 bit [4,5,8] 经消融实验选出最优配置。ATC 使用 EMA (γ=0.9997) 平滑校准阈值。E2B 量化使用 QuantSR 方案。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

## Patch-wise Layer-invariant Dynamic Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Patch-wise Layer-invariant Dynamic Quantization 是 Granular-DQ 提出的新型动态量化范式。与 CADyQ/CABM 同时逐层（layer-wise）和逐块（patch-wise）自适应 bit-width 不同，Granular-DQ 仅针对 patch 自适应——同一图像的不同 patch 可能获得不同 bit-width，但 SR 网络所有层对同一 patch 使用相同 bit-width。

"Layer-invariant" 的核心洞察：逐层独立调整 bit-width 会扰乱原始模型各层间的表示关系。论文通过 t-SNE 可视化证实——CADyQ 量化后各层特征分布明显更离散（与原始全精度模型偏差大），而 Granular-DQ 特征分布更接近原始模型。这解释了为什么放弃 layer sensitivity 反而获得更好量化效果。

从算法pipeline角度拆解术语：

```
# Granular-DQ: Patch-wise + Layer-invariant
b_i = GBC+E2B(X_i)  # 每个 patch 一个全局 bit-width
for conv_layer L_k in SR_model:
    for each patch X_i:
        X_i_hat = Q_{b_i}(X_i)         # 同一 b_i 贯穿所有层
        X_i = conv2d(X_i_hat, Q_8(W_k))
# 优势: 层间关系不被打乱

# CADyQ: Layer-wise + Patch-wise (对比)
for conv_layer L_k:
    for each patch X_i:
        b_{k,i} = BitSelector(L_k, X_i)  # 每层独立选择
        X_i = Q_{b_{k,i}}(conv2d(X_i, W_k))
# 问题: b_{k,i} 随层变化, 破坏原始层间关系
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该范式通过 GBC 集中决策 bit-width，避免每层插入 bit selector 的计算开销。量化器使用 QuantSR（可替换为 PAMS），权重固定 8-bit。Transformer attention block 保持全精度（因量化误差过大）。训练仅需 L1 loss，无需 KD loss 或 bit 正则化项。局限性：混合精度方案需特定硬件和算子支持才能实现真正的压缩加速。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

---

## aespa (Attention-centric Efficient and Scalable Post-training Quantization Algorithm)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
*aespa* 是 Samsung Research 在 NeurIPS 2024 提出的针对超大规模 Transformer 模型的 PTQ 算法。核心策略是"逐层量化 + attention-wise 重构"：每层单独量化以保持效率，但损失函数以 attention 输出重构为目标以引入跨层依赖。aespa 分两步执行：(1) 使用 Z-FOLD 结合提出的 attention-aware Hessian 计算量化参数（scale 和 zero-point）；(2) 使用 AdaRound 结合提出的精炼损失函数优化 weight-rounding policy。关键创新在于精炼量化目标：对 W_V 用 `H_V = 2E[XA^TAX^T]` 替代传统 `H = 2E[XX^T]`，将 Q 和 K 的信息通过 attention map A 耦合进 V 的 Hessian；对 W_Q/W_K 的损失函数分别引入 `E[K^TK]` 和 `E[Q^TQ]` 注入跨投影依赖。通过预计算这些统计量，每轮迭代仅需 O(d_h d^2) FLOPs，远低于传统 block-wise 方法的 O(B d_h L·max{d,L})。代码开源：https://github.com/SamsungLabs/aespa（CC BY-NC 4.0）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-7B Attention Block 的 INT2 量化为例，aespa 完整 pipeline：
```
# 输入: W_Q, W_K, W_V, W_O, W_FFN, 校准数据 X (128 seqs × 2048 tokens)

# === Phase 1: 全精度前向预计算关键统计量 ===
Q, K, V = W_Q(X), W_K(X), W_V(X)        # 全精度前向一次
A = softmax(Q @ K.T / sqrt(d))          # [B, H, L, L] attention map
A_out = A @ V                           # attention output

H_xx = mean(X @ X.T)                    # [d, d] 用于所有层的标准 Hessian
H_v  = mean(X @ A.T @ A @ X.T)            # [d, d] 用于 W_V 的 attention-aware Hessian
E_ktk = mean(K.transpose(-2,-1) @ K)     # [d_h, d_h] 用于 W_Q 损失
E_qtq = mean(Q.transpose(-2,-1) @ Q)     # [d_h, d_h] 用于 W_K 损失

# === Phase 2: 量化参数计算 (Z-FOLD) ===
for layer in [W_Q, W_K, W_V, W_O, W_FFN]:
    H = H_v if layer == W_V else H_xx    # 选对应 Hessian
    s = argmin_s tr(ΔW(s) @ H @ ΔW(s).T)  # Z-FOLD: 优化 step size

# === Phase 3: Weight-rounding 优化 (AdaRound with proposed losses) ===
for iter in range(2000):
    for W_V:
        ΔW_V = W_hat_V - W_V
        loss = sum((ΔW_V @ H_v) * ΔW_V)           # Equation (17), 一次矩阵乘
    for W_Q:
        ΔW_Q = W_hat_Q - W_Q
        loss = tr(E_ktk @ ΔW_Q @ H_xx @ ΔW_Q.T)   # Equation (21), 两次矩阵乘
    for W_K:
        ΔW_K = W_hat_K - W_K
        loss = tr(E_qtq @ ΔW_K @ H_xx @ ΔW_K.T)   # Equation (22)
    for W_O, W_FFN:
        loss = tr(ΔW @ H_xx @ ΔW.T)               # 标准 layer-wise 损失
    loss += λ * rounding_regularization            # AdaRound rounding loss
    loss.backward(); update(W_int)

# 复杂度: 每轮 O(d_h d^2), 与校准数据量无关
# OPT-125M: C_aespa=0.24 GFLOPS vs C_exist(B=4)=6.7 GFLOPS (28× gap)
```
与 baseline BRECQ 的关键区别：BRECQ 每轮需完整 `SA(Q_hat, K_hat, V_hat)` forward，而 aespa 通过预计算将 attention 操作"折叠"进统计量矩阵，后续迭代完全避开 attention computation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
aespa 官方实现（https://github.com/SamsungLabs/aespa）提供 quantize.py（主量化入口）、quantizer.py（核心量化器）、aespa.py（精炼损失实现）、quant_utils.py（工具函数）及定制模型文件（modeling_llama_custom.py, modeling_bloom_custom.py）。支持 OPT、BLOOM、LLaMA、LLaMA2 模型族。关键超参：校准集 128 segments × 2048 tokens from C4；weight-rounding 优化 2000 iterations、lr=0.015、rounding loss weight λ=1.5；量化前使用 OPTQ 初始化 W_int 以加速收敛。仅做 weight-only 量化（激活保持 FP16）。处理时间：OPT-125M INT2 约 5 分钟（GPU）；OPT-6.7B INT2 约 10.2 小时。在资源受限场景下可跳过 weight-rounding 优化，仅用 Z-FOLD + attention-aware Hessian 计算量化参数，OPT-1.3B 仅需 0.35 小时，仍优于 Z-FOLD 用标准 Hessian 的性能。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

## Attention-wise Reconstruction（注意力输出重构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention-wise Reconstruction 是 aespa 论文提出的量化重构目标范式：以 attention 模块的输出 `SA(Q, K, V) = softmax(QK^T/√d)V` 为重构目标，而非以单个 linear 层的输出或整个 Transformer block 输出为目标。其定位介于 layer-wise reconstruction（仅最小化各层输出误差）和 block-wise reconstruction（最小化整个 block 输出误差）之间。核心动机：Q、K、V 投影之间存在强跨层依赖——Q 和 K 共同决定 attention map A，A 又与 V 相乘得到最终 attention 输出——而 layer-wise 方法假设层间独立，忽略了这种依赖。通过以 attention 输出为重构目标，可以在逐层量化的前提下引入跨投影依赖信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种重构目标的数学对比：
```
# Layer-wise Reconstruction (AdaRound, OPTQ)
# 目标: 最小化各层独立输出误差
for each linear layer W:
    minimize E[||Q(W)X - WX||^2]          # 不考虑其他层

# Block-wise Reconstruction (BRECQ)  
# 目标: 最小化整个 attention block 输出误差
minimize E[||SA(Q_hat, K_hat, V_hat) - SA(Q, K, V)||^2]
# 所有 Q/K/V 联合优化 → 每轮需完整 attention forward → O(B·L·d_h·max(d,L))

# Attention-wise Reconstruction (aespa)
# 目标: 最小化 attention 输出误差，但逐层单独量化
for W_V (fixed Q, K):
    minimize E[||A·V_hat - A·V||^2]       # ΔSA_V, Equation (9)
    = minimize E[||ΔW_V·X·A^T||^2]       # 展开后 Q/K 信息通过 A 耦合
for W_Q (fixed K, V):
    minimize E[||SA(Q_hat,K,V) - SA(Q,K,V)||^2]  # Equation (11)
    ≈ minimize E[||K·ΔW_Q·X||^2]                # Equation (15), 通过上界近似
for W_K (fixed Q, V):
    minimize E[||SA(Q,K_hat,V) - SA(Q,K,V)||^2]  # Equation (25)
    ≈ minimize E[||Q·ΔW_K·X||^2]                # Equation (16)
```
有效性验证（Table 5）：在 OPT-125M 上，同时使用 layer-wise 量化粒度 + attention-wise 重构目标，INT2 PPL=69.23，显著优于 layer-wise 粒度 + layer-wise 重构的 AdaRound（PPL=160.7），接近 block-wise 联合量化的 BRECQ（PPL=60.38）。INT3/4 下与 BRECQ 性能几乎持平。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 attention-wise reconstruction 需设计精炼量化损失函数。aespa 的核心实现技巧：(1) 对 W_V：利用 Q/K 固定时 A=softmax(QK^T/√d) 不变，将目标简化为 `E[||ΔW_V·X·A^T||^2]`；(2) 对 W_Q/W_K：通过 first-order Taylor 展开近似 softmax 变化，再构造上界避开 Jacobian 存储（L³ 元素）。然后通过预计算 `E[XA^TAX^T]`、`E[K^TK]`、`E[Q^TQ]` 将这些统计量"冻结"，后续迭代无需执行 attention forward。这一策略不仅降低了计算复杂度，还使每轮 loss 计算等价于在整个校准集上评估（batch size = 全部校准数据量），梯度估计更准确，2000 轮即可收敛。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

## Z-FOLD (Foldable Parameter-based PTQ for LLMs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Z-FOLD 是 Samsung Research 在 EMNLP 2023 提出的针对 LLM 的 learning-free PTQ 方案。核心设计：利用 Transformer 中可折叠（foldable）的额外参数——即可以数学合并入其他层而不增加推理开销的参数（如 LayerNorm 的 affine weight）——来更精细地量化权重。Z-FOLD 通过优化 scale 和 zero-point（量化参数）以及 foldable parameters 来最小化基于 Hessian 的重构误差 `tr(ΔW·H·ΔW^T)`，其中 H=2E[XX^T]。Z-FOLD 是 learning-free 方法（无需梯度反向传播优化 weight-rounding policy），仅需前向统计量计算和闭式解优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Z-FOLD 的核心算法步骤：
```
# Z-FOLD: 基于 Layer-wise Hessian 的 learning-free 量化
H = 2 * mean(X @ X.T)                   # [d, d] Hessian 近似

for each layer:
    # Step 1: 量化参数初始化（使用传统的 min-max 或 MSE 方法）
    s_init, z_init = init_quant_params(W)
    
    # Step 2: 利用 foldable parameters 优化量化参数
    # foldable parameters: LayerNorm weight γ, bias β
    # 优化目标: min_{s, z, γ, β} tr(ΔW(s,z,γ,β) · H · ΔW(s,z,γ,β)^T)
    s_opt, z_opt, γ_opt, β_opt = optimize_quant_and_foldable(W, H, X)
    
    # Step 3: 应用优化后的参数进行量化
    W_int = quantize(W, s_opt, z_opt)    # 使用最近舍入 (RTN)
    
    # Step 4: 将 foldable parameters 合并到相邻层
    # γ_opt 合并到前一层输出或当前层权重，推理时零额外开销
```
在 aespa 中，Z-FOLD 被用于量化参数计算阶段（Algorithm 1 line 4），但 Hessian 被替换为 attention-aware 版本：对 W_V 使用 `H_V = 2E[XA^TAX^T]`，显著提升 INT2 精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Z-FOLD 是 learning-free 方法，不需要 GPU 训练或梯度下降，因此处理速度较快。在 aespa 的实验中，Z-FOLD 是唯一同时展示合理精度和处理时间的 layer-wise 方法（OPTQ 虽快但 INT2 精度崩溃，BRECQ 精度好但不可扩展）。Z-FOLD 的局限性：(1) 依赖最近舍入而非优化 weight-rounding policy；(2) 不考虑 attention 内部跨层依赖（使用标准 H=2E[XX^T]）。aespa 通过在 Z-FOLD 之后增加 AdaRound weight-rounding 优化并替换为 attention-aware Hessian，补齐了这些短板。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

## Cross-layer Dependency in Attention Module（注意力模块中的跨层依赖）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-layer Dependency in Attention Module 指 Transformer 的 attention 模块中 Q（Query）、K（Key）、V（Value）三个投影层之间的强相互依赖关系。在 self-attention 中，Q 和 K 的矩阵乘法结果经 softmax 产生 attention map A，A 再与 V 相乘得到 attention 输出。这意味着：(1) Q 的量化误差会通过 softmax 非线性放大并传播给 V；(2) K 的量化误差同样影响 attention map，进而影响 V 的有效输入；(3) V 的量化误差被 attention map A 加权。三层共同的输出是 `SA(Q,K,V) = A·V`，任何一层的误差都会通过非线性交互影响最终 attention 输出。传统的 layer-wise PTQ（AdaRound、OPTQ）假设层间独立，忽略这种依赖，导致低比特下性能退化严重。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
跨层依赖在 attention 计算中的传播路径：
```
# Self-Attention 前向: 三层之间的信息流
Q = X @ W_Q     # Query projection, 形状 [B, H, L, d_h]
K = X @ W_K     # Key projection
V = X @ W_V     # Value projection

# 依赖点 1: Q-K 交互
S = Q @ K.T / sqrt(d_h)    # [B, H, L, L] 注意力分数
A = softmax(S)              # 注意力权重 → Q/K 误差经 softmax 非线性放大

# 依赖点 2: A-V 交互  
O_attn = A @ V              # [B, H, L, d_h] → V 误差被 A 加权组合
```
Layer-wise 方法独立量化 W_Q、W_K、W_V 时，各自仅最小化 `||ΔW_Q·X||^2`、`||ΔW_K·X||^2`、`||ΔW_V·X||^2`，完全忽略了上述交互路径。aespa 通过 attention-wise reconstruction 捕获这些依赖：W_V 的 Hessian `H_V = 2E[XA^TAX^T]` 通过 A 矩阵将 Q 和 K 的信息注入 V 的量化过程；W_Q 的损失引入 `E[K^TK]` 项以感知 K 的信息；W_K 的损失引入 `E[Q^TQ]` 项以感知 Q 的信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
捕获跨层依赖需要突破传统 layer-wise 方法的独立 Hessian 假设。aespa 的实现方式：(1) 在全精度下前向一次得到 attention map A；(2) 预计算 attention-aware 统计量 `E[XA^TAX^T]`、`E[K^TK]`、`E[Q^TQ]`；(3) 在后续量化迭代中重用这些统计量，无需重复执行 attention forward。这种预计算方法同时解决了效率问题（避免每轮 attention computation）和依赖建模问题（统计量携带跨投影信息）。该思路可推广到其他含非线性交互的模块（如 FFN 中 gate/up 投影的 SiLU 激活交互），但需要更复杂的数学推导。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

## Warmup-Stable-Decay (WSD) Learning Rate Schedule

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Warmup-Stable-Decay（WSD）学习率调度是 LLM 预训练中一种三阶段学习率调度策略。其将训练分为：(1) **Warmup 阶段**：学习率从 0（或极小值）线性增加到峰值 η_max（通常占训练步数的 1-2%）；(2) **Stable/Constant 阶段**：学习率保持恒定在 η_max，持续大部分训练步数（如 89% 步数）；(3) **Decay/Cooldown 阶段**：学习率从 η_max 线性衰减到 0（通常占步数的 10%）。与 Cosine Decay（学习率从峰值连续余弦衰减到底）不同，WSD 的核心特点是存在漫长的恒学习率阶段，这使得：(a) 可以在不同 token 预算处插入 cooldown 来灵活复检模型状态；(b) 恒学习率阶段模型被限制在同一个 loss basin 中迭代，促进探索而非收敛；(c) cooldown 衰减到零时模型才进入更尖锐的 loss 区域。WSD 由 Zhai et al. (2022) 提出用于 vision transformer，后由 Hu et al. (2024) (MiniCPM)、Bakouch et al. (2025) (SmolLM3) 等大规模 LLM 训练广泛采用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WSD 调度在 LLM 预训练中的典型实现伪代码：
```
def wsd_schedule(step, total_steps, peak_lr=3e-3, warmup_pct=0.01, cooldown_pct=0.1):
    warmup_steps = int(total_steps * warmup_pct)
    cooldown_steps = int(total_steps * cooldown_pct)
    stable_steps = total_steps - warmup_steps - cooldown_steps
    if step < warmup_steps:
        lr = peak_lr * (step / warmup_steps)       # 线性增长
    elif step < warmup_steps + stable_steps:
        lr = peak_lr                                # 恒定
    else:
        decay_progress = (step - warmup_steps - stable_steps) / cooldown_steps
        lr = peak_lr * (1.0 - decay_progress)       # 线性衰减到0
    return lr
```
论文的核心发现：在 Stable 阶段（恒学习率，长达数万亿 tokens），验证损失缓慢下降而量化误差几乎不变；一旦进入 Decay 阶段，验证损失急剧下降但量化误差同时激增——这是因为学习率衰减使模型进入更尖锐的 loss 区域（Hessian 最大特征值上升），对量化引起的权重扰动更敏感。论文据此认为 WSD 优于 Cosine Decay，因为 WSD 可以更好控制末期学习率，而 Cosine 的末期学习率由峰值 lr 和训练步数隐式决定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WSD 在现代 LLM 训练框架中实现为自定义 LR scheduler：(1) PyTorch 中通过 `LambdaLR` 或 `SequentialLR` 组合 `LinearLR(during_warmup) → ConstantLR(during_stable) → LinearLR(during_decay)`；(2) 在 plainLM (https://github.com/Niccolo-Ajroldi/plainLM) 等训练代码库中直接实现。参数配置：峰值学习率通常在 1e-3 到 6e-3 之间，warmup 1-2% 步数（如 1900 steps for 100B tokens），cooldown 约 10% 步数，衰减到 0。基于论文发现，选择 WSD 时应在训练过程中持续监控 PTQ 误差作为附加超参数指标——如果两个学习率候选在验证损失上表现相似，应优先选择在衰减后 PTQ 误差更低的那个。

涉及论文标题：
- Training Dynamics Impact Post-Training Quantization Robustness

## Weight Averaging (LAWA) for Quantization Robustness

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Averaging（权重平均）是一种沿训练轨迹聚合多个 checkpoint 权重以改善模型泛化能力和 PTQ 鲁棒性的技术。核心操作：在训练过程中每隔固定步数保存 checkpoint，使用滑动窗口聚合最近的 K 个 checkpoint 的权重（均匀平均），输出平均后的模型。LAWA（LAtest Weight Averaging, Kaddour 2022）是最直接的变体——维护长度为 5 的滚动 FIFO 窗口，每个新 checkpoint 入队时最旧的出队，实时输出均匀平均后的权重。与 Model Soup（Wortsman et al., 2022，对多个独立训练 fine-tune 的模型取平均）不同，Weight Averaging 沿单一训练轨迹操作，无需多次训练。该论文的关键新发现：虽然 LAWA 在全精度下通常不如 lr cooldown，但在 3-bit / 4-bit 量化后，LAWA 可以匹配甚至超越 cooldown 的性能——因为 averaging 促进收敛到更平坦的极小值（wider minima），使模型对量化引起的权重扰动更具鲁棒性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LAWA 的在线算法流程：
```
buffer = []  # FIFO queue, max length = window
for step, batch in enumerate(dataloader):
    loss = model(batch).backward()
    optimizer.step(); optimizer.zero_grad()
    if step % checkpoint_interval == 0:
        buffer.append(deepcopy(model.state_dict()))
        if len(buffer) > window:
            buffer.pop(0)
        avg_state = average_state_dicts(buffer)  # element-wise mean
        lawa_model.load_state_dict(avg_state)
        # 评估 LAWA 模型的量化性能
        W_q = gptq_quantize(lawa_model, bits=3)
        quant_loss = eval_validation(W_q)
```
对于无法自由保存 checkpoint 的开源模型（如 OLMo-1B），使用连续发布 checkpoint 的增量平均：accumulate_n = Σ_{i=last_n} Θ_i / n，评估不同窗口长度 n 的效果。论文 Fig.24 显示 n=5 效果最好，平均后的模型同时降低全精度和量化后验证损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Weight Averaging 的实现：(1) PyTorch 中直接遍历 `state_dict` 逐参数 `(w1 + w2 + ...) / N`；(2) 通过 `torch.save`/`torch.load` 管理 checkpoint 文件。对大规模模型需注意 checkpoint 加载的 I/O 和内存开销。配置建议：checkpoint 保存间隔应足够密集（论文用 500 优化步），窗口长度 5 是默认配置，均匀平均优于指数移动平均。论文还发现 Model Soup（跨不同数据混合的独立训练模型平均）的量化误差低于任何单个成分（Fig.2），表明跨独立运行的 weight averaging 也有利于 PTQ。

涉及论文标题：
- Training Dynamics Impact Post-Training Quantization Robustness

## Hessian Sharpness and Trace in Quantization Robustness

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hessian Sharpness（Hessian 矩阵最大特征值 λ_max，也称"锐度"）和 Hessian Trace（Hessian 矩阵对角线之和，即所有特征值之和）是衡量神经网络 loss landscape 局部几何性质的关键二阶统计量。Sharpness 反映 loss 盆地在最陡方向上的曲率——值越大，loss 对权重扰动的敏感度越高。Trace 反映所有方向上的平均曲率——值越大，整体上 loss 曲面越不平坦。该论文将这两个指标与 PTQ 鲁棒性建立了因果关系：量化的本质是对权重施加离散扰动 W → Ŵ = W + ΔW。如果模型处于尖锐的 loss 区域（高 sharpness/trace），同样的 ΔW 导致更大的 loss 增加，因此量化误差更大。论文通过受控实验证明：(1) lr 衰减时 sharpness 和 trace 同时激增（Fig.9），与量化误差激增的时间模式完全一致；(2) 较大的峰值学习率产生更低的 sharpness 和 trace（Fig.26），对应更低的量化误差；(3) Weight averaging 也降低 sharpness，解释了其对 PTQ 的益处。论文由此假设：lr 衰减使模型进入更尖锐的 loss 区域，使其对量化扰动更敏感。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hessian Sharpness 和 Trace 的估计（基于 PyHessian + Hutchinson estimator）：
```
def estimate_hessian_stats(model, val_loader, n_ht=100, n_pi=50):
    # Hutchinson estimator for trace (unbiased)
    traces = []
    for _ in range(n_ht):
        z = torch.randint(0, 2, shape) * 2 - 1  # Rademacher
        Hv = autograd_hvp(loss_fn, model_params, z)
        traces.append(dot(z, Hv))
    trace_est = mean(traces)

    # Power iteration for λ_max (sharpness)
    v = torch.randn(shape); v /= norm(v)
    for _ in range(n_pi):
        Hv = autograd_hvp(loss_fn, model_params, v)
        lambda_max = dot(v, Hv)
        v = Hv / norm(Hv)
    return lambda_max, trace_est
```
论文的发现模式：(a) 恒学习率阶段：仅 top eigenvalue 短暂上升但其余保持小值→量化误差温和增加；(b) Decay 阶段：所有 eigenvalue 同时急剧上升→量化误差飙升；(c) 较大峰值 lr → 更小的 sharpness 和 trace → 更低量化误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyHessian (Yao et al., 2019) 是标准的 PyTorch Hessian 分析工具（https://github.com/amirgholami/PyHessian），通过 Hessian-vector product (HVP) 自动微分实现无显式构建完整 Hessian 矩阵的统计量估计（160M 参数模型的完整 Hessian 有 2.56×10^16 个元素无法存储）。Hutchinson estimator 使用 Rademacher 分布（Gaussian 也有效但方差更大）；power iteration 收敛到 λ_max 需要 50+ 次 HVP 迭代；HVP 通过对 loss 做两次反向传播实现，内存和时间为标准前向+反向的约两倍。论文在验证集（100 个 FineWebEdu 序列，每个 2048 tokens）上计算统计量。该技术可推广到监控任何训练过程中的泛化和鲁棒性指标变化趋势。

涉及论文标题：
- Training Dynamics Impact Post-Training Quantization Robustness

## Ridge Leverage Score (Ridge 杠杆分数)

术语是什么？
Ridge Leverage Score 是一种统计度量，用于衡量线性回归中每个数据点对模型拟合的影响力。在 UniQL 的 LLM 压缩语境下，该分数用于评估 MLP 中间层每个通道（$D_{int}$ 维度）对输出重建的重要性。给定校准样本的中间激活相关性矩阵 $C = \mathbf{X}_{int}^{\top} \mathbf{X}_{int} \in \mathbb{R}^{D_{int} \times D_{int}}$，ridge leverage scores 定义为：
$$s = \operatorname{diag}\left(C(C + \lambda I)^{-1}\right), \quad s \in \mathbb{R}^{D_{int}}$$
其中 $\lambda$ 是 ridge 正则化强度（UniQL 设 $\lambda=1$），$I$ 为单位矩阵。该分数对每个通道的"杠杆"进行量化——高分数通道对激活重建更重要，排序矩阵 $\mathbf{S}_m = I[:, \operatorname{argsort}(s)]$ 将权重列按重要性降序排列，使得剪枝时只需丢弃末尾（低杠杆）列。

从算法pipeline角度拆解：
在 UniQL 的结构化权重排序 pipeline 中（Algorithm 1），该分数的工作流程：
1. 从 Alpaca 校准集采样 128 个样本（seq_len=2048），计算每个样本的 MLP 中间激活 $\mathbf{X}_{int}^i = \sigma(\mathbf{X}_h^i \mathbf{W}_g) \odot \mathbf{X}_h^i \mathbf{W}_u$
2. 平均所有样本的相关性矩阵 $C = \frac{1}{N}\sum_{i=1}^N \mathbf{X}_{int}^{i\top} \mathbf{X}_{int}^i$
3. 计算 $s = \operatorname{diag}(C(C + \lambda I)^{-1})$
4. 按 $s$ 降序构造排序矩阵 $\mathbf{S}_m$
5. 重排 MLP 权重：$\mathbf{W}_u' = \mathbf{W}_u \mathbf{S}_m$，$\mathbf{W}_g' = \mathbf{W}_g \mathbf{S}_m$，$\mathbf{W}_d' = \mathbf{S}_m^\top \mathbf{W}_d$

伪代码：
```
# 输入: 校准激活 X_int ∈ R^{N × T × D_int}, λ=1
C = einsum("nti,ntj->ij", X_int, X_int) / N    # [D_int, D_int]
C_reg = C + λ * I                               # ridge 正则化
C_inv = solve(C_reg, I)                         # 矩阵求解 (非伪逆)
s = diag(C @ C_inv)                             # [D_int] ridge leverage scores
idx = argsort(s, descending=True)               # 按重要性排序
S_m = I[:, idx]                                 # 排序矩阵
W_u, W_g = W_u @ S_m, W_g @ S_m                # 重排列
W_d = S_m^T @ W_d                               # 重排行
```

术语一般如何实现？如何使用？
Ridge leverage score 在 statistics 和 numerical linear algebra 中有成熟的理论基础（McCurdy, 2018），UniQL 首次将其应用于 LLM 权重排序。相比基线 MoDeGPT 的伪逆（Moore-Penrose inverse, O(n³), FP64），ridge leverage scores 只需一次矩阵求解（可用 Cholesky 分解或 LU 分解），在 BF16 下运算即可，对于 Llama-3.1-8B 的 MLP 层（$D_{int}=14336$），从 20.58 分钟降至数秒（20× 加速）。且 ridge leverage score 的排序结果对不同剪枝率是等价的（重新排序不需要重新计算），因为排序矩阵 $\mathbf{S}_m$ 基于原始满秩权重计算，而非剪枝后的子矩阵。这支持一次排序 → 多剪枝率的 one-pass 设计。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs


## Quantization-Aware SVD Decomposition (QSVD / 量化感知SVD分解)

术语是什么？
Quantization-Aware SVD Decomposition (QSVD) 是 UniQL 提出的一种针对低比特（INT4）量化场景优化的奇异值分解策略。在 LLM 的 value-output 权重对 $(\mathbf{W}_v, \mathbf{W}_o)$ 的联合压缩中，先通过连续两次 SVD 分解排序权重通道：$C^{1/2}\mathbf{W}_v = \mathrm{U}_v \Sigma_v \mathbf{V}_v^{\top}$，然后 $\mathrm{SVD}(\Sigma_v \mathbf{V}_v^{\top} \mathbf{W}_o) = \mathbf{U}\Sigma\mathbf{V}^{\top}$。最终排序权重为 $\mathbf{W}_v = C^{-1/2} \mathbf{U}_v \mathbf{U} \Sigma$，$\mathbf{W}_o = \mathbf{V}^{\top}$。

QSVD 的核心创新在于：将特征值对角阵 $\Sigma$ 融合到 $\mathbf{U}$ 而非 $\mathbf{V}$，使排序后的 $\mathbf{W}_v = (\mathbf{U}\Sigma)$ 而非传统 $\mathbf{W}_v = \mathbf{U}$。其关键洞察是：低比特量化对量化组内的数值分布高度敏感。SVD 的特征值 $\sigma_i$ 呈长尾分布，若将其保留在 $\mathbf{V}$ 侧，$\mathbf{U}$ 列内数值跨度小但 $\mathbf{V}$ 行内跨度大；若融合到 $\mathbf{U}$ 侧，每列 $\mathbf{U}_i$ 乘以对应特征值 $\sigma_i$，则 $\sigma_i$ 自然充当该列的 group-wise 量化缩放因子，避免长尾特征值被 INT4 截断。

从算法pipeline角度拆解：
在 UniQL 的 MHSA value-output 权重排序中（Algorithm 3），QSVD 的流程为：
```
# 输入: W_v ∈ R^{D_h × D_hd}, W_o ∈ R^{D_hd × D_h}, 校准激活 X_h
# Step 1: 计算输入相关性矩阵
C = X_h^T @ X_h                                    # [D_hd, D_hd]
C_half = cholesky(C)                               # 或 sqrtm(C)

# Step 2: 第一次 SVD
U_v, Σ_v, V_v_T = SVD(C_half @ W_v)               # 分解激活加权的 value 投影

# Step 3: 第二次 SVD (joint decomposition)
U, Σ, V_T = SVD(Σ_v @ V_v_T @ W_o)                # 联合分解 value 和 output

# Step 4: QSVD - 融合 Σ 到 U (关键步骤)
W_v_sorted = inv(C_half) @ U_v @ U @ Σ            # U 乘以特征值，每列缩放
W_o_sorted = V_T                                    # V 保留为排序输出矩阵

# 量化时: 每个 group (128列) 共享缩放因子 s = max(|W_v_sorted[:,g]|) / 7
# 由于 Σ 已融合到 U，特征值自然地逐列缩放，避免 INT4 截断失真
```

定量效果：在 W4A16 Llama-3.1-8B 上 25% 剪枝率，不融合 Σ 时精度为 60.2%，融合后（QSVD）提升至 67.7%（+7.5%）；Qwen-2.5-7B 从 61.0% 提升至 64.0%（+3.0%）。

术语一般如何实现？如何使用？
QSVD 的实现依赖于标准数值线性代数库（如 PyTorch 的 `torch.linalg.svd`）。在 LLM 压缩 pipeline 中，QSVD 在权重排序阶段执行（云侧，单次），排序后的 $\mathbf{W}_v$ 列已按重要性降序排列，特征值缩放已内嵌在权重矩阵中。随后进行 GPTQ 量化时（group_size=128, INT4），每组的量化缩放因子 $s_g = \max(|\mathbf{W}_{(i,g)}|) / 7$ 自然受益于 Σ 的预缩放——重要列（大 $\sigma_i$）的权重大，不重要的列（小 $\sigma_i$）权重小，形成天然的数值分布梯度，使 INT4 量化误差最小化。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

## Masked LoRA Fine-tuning (掩码LoRA微调)

术语是什么？
Masked LoRA Fine-tuning 是 UniQL 提出的一种 one-shot 微调策略，通过在已排序但未剪枝的模型上使用 LoRA（Low-Rank Adaptation）进行带随机掩码的训练，使单个模型副本支持多种剪枝率（0%-35%）的部署。与传统方法（SVD-LLM, MoDeGPT）不同：传统方法针对每个目标剪枝率独立训练一个模型，训练成本随压缩率数量线性增长（O(n)）；UniQL 的 masked LoRA fine-tuning 在一次训练中（O(1)）支持所有剪枝率。

核心机制：
1. 使用 Block Influence (BI) scores 预先计算所有目标全局剪枝率 $P = [P_{15}, P_{20}, P_{25}, P_{35}]$ 对应的层间剪枝率分配。
2. 在每个训练步 t，随机抽取一个全局剪枝率 $P_t \sim P$。
3. 对每层，按 $P_t$ 对应的层间分配率 $r_l^{P_t}$ 生成掩码——仅保留重要性排名最高的通道（按 ridge leverage scores / SVD eigenvalues 排序），其余通道的权重在 forward pass 中被置零。
4. 仅更新 LoRA adapter（r=8, α=16），冻结原始权重。

从算法pipeline角度拆解：
```
# 预计算层间剪枝率 (使用 BI scores)
For each target global rate p ∈ {15%, 20%, 25%, 35%}:
    s_layer[l] = 1 - E[x_l^T y_l / (||x_l|| ||y_l||)]   # Block Influence score
    r_layer[l] = L * p * softmax(-s_layer / ε)[l]         # 层间分配, ε=0.1

# Masked LoRA 训练 (单次, 5 epochs, Alpaca dataset)
W_original = freeze(sorted_weights)                      # 已排序、冻结
ΔW_lora = init(A @ B, r=8, α=16)                         # LoRA adapter
For step = 1 to total_steps:
    p_t ~ Uniform(P)                                      # 随机采样剪枝率
    mask = zeros_like(W)
    For layer l:
        keep_ch = int(D[l] * (1 - r_layer[l][p_t]))       # 保留通道数
        mask[l][:keep_ch] = 1                              # 前 keep_ch 列 = 1
    output = model(input, W_original * mask + ΔW_lora)    # 掩码前向
    loss = cross_entropy(output, target)
    ΔW_lora -= lr * ∇loss                                 # 仅更新 LoRA

# 训练后: 量化 → 部署。设备端可选 0%-35% 任意剪枝率
```

术语一般如何实现？如何使用？
超参数（UniQL 默认）：LoRA rank r=8, scaling α=16, dropout 0.05, AdamW optimizer, lr=1e-4, batch size 32 (micro batch 4), warmup 100 steps, seq_len=256, 5 epochs on Alpaca dataset。整个训练在单张 A6000 GPU 上完成（Llama-3.1-8B 约 7h43m 含排序+微调+量化）。关键优势：单次训练产出一个人可以支持所有剪枝率的 LoRA adapter，合并到量化模型中后，设备端仅需根据当前系统负载选择剪枝率并裁剪通道，无需额外训练。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

## Block Influence (BI) Score (块影响力分数)

术语是什么？
Block Influence (BI) Score 是衡量 LLM 中每层（或块）对最终输出影响力的一种度量，用于指导层间非均匀剪枝率的分配。BI score 定义为层输入 $x_l$ 和输出 $y_l$ 之间的余弦相似度偏离期望值：
$$s_l = 1 - \mathbb{E}\left[\frac{x_l^{\top} y_l}{\|x_l\|_2 \|y_l\|_2}\right]$$
其中 $x_l$ 和 $y_l$ 分别是第 $l$ 层的输入和输出激活向量。

直觉：如果一层的输入和输出高度相似（余弦相似度接近 1 → $s_l$ 接近 0），说明该层对信息的变换很小，可以被大幅剪枝。反之，若 $s_l$ 大，说明该层做了大量信息变换，需保留更多通道。基于 BI scores，层间剪枝率通过 softmax 归一化分配：
$$r_l = L \cdot P_{\text{avg}} \cdot \operatorname{softmax}(-s / \varepsilon)_l$$
其中 $P_{\text{avg}}$ 是目标全局剪枝率，$L$ 是总层数，$\varepsilon=0.1$ 控制分配的温度。

从算法pipeline角度拆解：
在 UniQL 中，BI scores 的使用流程：
1. 用 WikiText-2 校准集（128 samples, seq_len=2048）前向传播一次全精度模型。
2. 对每层/块（Transformer block 或 Mamba block）记录 $x_l$ 和 $y_l$，计算 $s_l = 1 - \mathbb{E}[\cos\text{sim}(x_l, y_l)]$。
3. 对每个目标全局剪枝率 P ∈ {15%, 20%, 25%, 35%}，用 softmax 公式计算各层的剪枝率分配。
4. 混合模型（Nemotron-H, Bamba-v2）中，self-attention 层的 BI score 显著高于 SSM 层 → 分配更低的剪枝率，验证了 BI score 能准确反映不同层类型的敏感性差异。

术语一般如何实现？如何使用？
BI score 最初由 Men et al. (2024) 在 ShortGPT 中提出用于识别冗余层。MoDeGPT (Lin et al., 2025) 和 UniQL 均采用该方法进行层间剪枝率分配。计算开销低（仅需一次前向传播），不依赖梯度信息，适合 post-training 场景。注意：BI score 对校准数据集敏感——UniQL 使用 WikiText-2 而非 Alpaca 来计算 BI scores（保证与 MoDeGPT 的可比性），而权重排序和微调使用 Alpaca。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs


## State-Aware Weight Sorting for SSMs (状态感知SSM权重排序)

术语是什么？
State-Aware Weight Sorting 是 UniQL 针对 Mamba（State Space Model）块提出的一种结构化权重排序策略。与 Transformer 的 attention 机制不同，SSM 的计算核心是线性递归 $(h_t = \Delta_t A_t h_{t-1} + \Delta_t B_t x_t,\ y_t = C_t h_t)$，其内部状态 $h_t$ 携带长序列信息。UniQL 将 Mamba 块的计算拆分为两个子公式进行权重排序：

1. **SSM 输入掩码 $\mathcal{M}$（B-C 排序）**：$\mathcal{M} = \phi(X\mathbf{W}_C) (\Delta \odot \phi(X\mathbf{W}_B))^{\top}$。排序 $\mathbf{W}_B^g$ 和 $\mathbf{W}_C^g$ 的列，考虑输入依赖的离散化变量 $\Delta^g$ 通过广播外积对 $\mathbf{B}^g$ 的调制作用。
2. **SSM 状态 $\mathcal{H}$（z-x-o 排序）**：$\mathcal{H} = \Delta A \mathcal{H}(h_0) + \Delta B X_\phi$。从已计算出的 SSM 状态 $\mathcal{H}^i$ 收集相关性矩阵 $C_{\mathcal{H}} = \mathcal{H}^{i\top} \mathcal{H}^i$，计算 ridge leverage scores 来排序 $\mathbf{W}_z^i$、$\mathbf{W}_x^i$ 和 $\mathbf{W}_o^i$。

从算法pipeline角度拆解：
B-C 排序的完整流程（Algorithm 4）：
```
# 输入: W_B^g, W_C^g ∈ R^{D_h × D_s}, 校准激活 X_h, 输入依赖步长 Δ^g
For each SSM group g = 1, ..., G_s:        # Mamba2: G_s=1
    B = conv1d(X_h @ W_B^g)                 # [T, D_s], 1D causal conv + SiLU
    C = conv1d(X_h @ W_C^g)                 # [T, D_s]
    ΔB = Δ^g ⊗ B                            # [H_m^g, T, D_s], 广播外积
    
    # 计算状态维度 D_s 的相关性
    ΔC_B = mean_i((ΔB^i)^T @ (ΔB^i))       # [H_m^g, D_s, D_s]
    C_C = mean_i((C^i)^T @ (C^i))           # [D_s, D_s]
    
    # 多SSM头的范数得分汇总
    s = zeros(D_s)
    For k = 1, ..., H_m^g:
        s += ||(ΔC_B^k)^{1/2}|| ⊙ ||C_C^{1/2}||
    
    S_BC = I[:, argsort(s)]                # 排序矩阵
    W_B^g, W_C^g = W_B^g @ S_BC, W_C^g @ S_BC  # 列重排
```

z-x-o 排序（Algorithm 5）：从 SSM 状态 $\mathcal{H}^i \in \mathbb{R}^{H_m \times (T \times D_s) \times D_{hd}}$ 收集 $C = \frac{1}{N}\sum_{i=1}^N \mathcal{H}^{i,j\top}\mathcal{H}^{i,j}$，然后计算 ridge leverage scores 排序 $\mathbf{W}_z$、$\mathbf{W}_x$、$\mathbf{W}_o$。

与 MLP 的 ridge leverage score 排序的关键区别：B-C 排序的得分是 B 和 C 两组相关性范数的逐元素乘积（考虑了 SSM 的双路径特性），而 z-x-o 排序直接从 SSM 状态相关性开始，称为"状态感知"排序。

术语一般如何实现？如何使用？
该策略专门为 Mamba 架构设计，需要理解 SSM 的线性递归方程和输入依赖离散化。实现要求：能前向传播 Mamba 块（如 Mamba2）以收集 $\mathbf{B}$、$\mathbf{C}$ 的卷积输出和 $\mathcal{H}$ 状态，并实现广播外积和相关性矩阵计算。对 Mamba2-8B，B-C 排序处理 1 个 SSM group（$G_s=1$），$H_m=64$ 个 SSM head，$D_s=128$ 状态维度。整个排序在 A6000 上耗时 16 分钟。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

## One-Pass Adaptive Pruning (单次自适应剪枝)

术语是什么？
One-Pass Adaptive Pruning 是 UniQL 提出的部署策略：在云端对 LLM 进行一次性的结构化权重排序、masked LoRA 微调和量化（single pass），产出一个包含已排序通道的量化模型。部署到边缘设备后，设备端仅需根据当前系统负载（可用内存）决定剪枝率，从量化权重末尾去除指定比例的通道即可获得对应尺寸的模型。该策略的核心思想是将计算密集的排序-微调-量化"压缩"为单次 O(1) 操作，而非为每种剪枝率独立执行 O(n) 次完整压缩流程。

关键设计：
1. **隐藏维度不变**：剪枝仅缩减中间维度（$D_{int}$、$D_{hd}$、$D_s$），隐藏维度 $D_h$ 在所有剪枝率下保持不变，保证各层间的维度匹配。
2. **Hadamard 矩阵不作用于剪枝通道**：Hadamard rotations 仅融合到非剪枝侧的矩阵（如 `q_proj` 的输入 Hadamard 为 Yes，输出为 No*），避免剪枝后预融合矩阵形状不匹配。
3. **设备端 INT4 在线处理**：部署的 4-bit 权重在运行时解包（unpack from INT4）→ 按当前剪枝率去除末尾通道 → 重新打包为 INT32 → 送入 GEMM kernel。

从算法pipeline角度拆解：
```
# 云侧 (one-pass, 单张 A6000 GPU)
Step 1: Ridge leverage scores / QSVD / state-aware 排序各模块权重通道
Step 2: BI scores 计算所有目标剪枝率的层间分配 (O(1) for multiple rates)
Step 3: Masked LoRA 微调 5 epochs (每步随机采样剪枝率训练)
Step 4: GPTQ W4A16 量化 (含 embedding/output 层, head-to-toe)
Step 5: 产出单个 INT4 模型文件 (Llama-3.1-8B: 4.1GB)

# 设备侧 (每次自适应)
Input: device_memory_budget ← OS 报告当前可用内存
If device_memory_budget > 4.1GB: p = 0%    # 满精度量化
elif device_memory_budget > 3.4GB: p = 15%  # 温和剪枝
elif device_memory_budget > 3.0GB: p = 25%  # 中度剪枝
else: p = 35%                               # 激进剪枝

For each prunable layer:
    D' = D * (1 - p%)                       # 新维度
    W_int4 = load_from_file(layer)          # 加载 INT4 packed weights
    W_fp16 = unpack(W_int4)[:, :D']        # 解包，取前 D' 列
    W_int32 = repack(W_fp16)                # 重新打包
    // 加载到 GPU 并使用 Marlin/Marlin-like GEMM kernel 执行
```

术语一般如何实现？如何使用？
适用场景：动态工作负载的边缘设备（如手机、VR/AR 眼镜、Jetson Orin），操作系统在不同时刻分配给 LLM 推理的内存不同。实现需要：①云侧 GPU 48GB+（排序 + 微调 + 量化），②客户端推理框架支持动态 INT4 权重裁剪和重打包，③融合 RoPE kernel 处理排序后的位置嵌入索引。UniQL 在 Jetson Orin Nano 8GB 上验证：从 TAO-HQQ 的 W4A16 固定模型（5.7GB, 133.6ms TPOT）降至 head-to-toe 4-bit + 自适应剪枝模型（4.1GB @ 0%, 3.4GB @ 15%, 2.8GB @ 35%），TPOT 对应 77.2ms / 64.0ms / 57.7ms。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs


---

## Adaptive Label Sampling（自适应标签采样 for ZSQ Object Detection）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Label Sampling 是 Task-Specific Zero-shot Quantization-Aware Training for Object Detection 论文提出的核心创新方法，用于在零样本场景下为检测任务合成带边界框标签的校准数据。其核心思想是：利用预训练全精度检测网络的知识（而非真实标注），通过交替迭代——图像优化和标签更新——逐步使合成标签收敛到teacher模型认可的高质量检测目标集合。具体步骤：(1) 初始化：随机生成高斯噪声输入x和包含单一随机目标的标签y（类别~U(0,C)，bbox中心~U(W/2,1-W/2)，宽高~U(0.2,0.8)）；(2) 循环：每固定迭代步，用teacher模型对当前x做检测推理，取conf > conf_thresh的高置信度预测作为new_tgts，计算IOU(new_tgts, 当前tgts)，添加与现有标签不重叠的新标签(max_iou < iou_thresh)，移除未被teacher重新检测到的旧标签，确保每张图至少保留一个标签；(3) 标签收敛后，固定标签y*，重新初始化高斯噪声x，用task-specific损失L_total = alpha_prior*L_prior + alpha_detect*L_detect(phi(x), y*) + L_reg优化生成最终合成图像。该方法无需任何真实标注或额外网络（如预训练生成模型），仅依赖预训练检测网络的内嵌知识。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以YOLOv5-s在MS-COCO上的Adaptive Label Sampling两阶段流程为例：

```
// Algorithm 1: Adaptive Label Sampling 核心循环
Input: current_image x, current_labels tgts, teacher_model phi(theta),
       conf_thresh, iou_thresh

1. new_tgts = phi(theta).predict(x)[conf > conf_thresh]
   // 用teacher对当前图像做检测推理，取高置信度预测

2. ious = IOU(new_tgts, tgts)
   // 计算新预测和已有标签之间的IOU矩阵

3. add_tgts = new_tgts[(max(ious, dim=1) < iou_thresh)]
   // 添加不与已有标签重叠的新标签

4. minus_tgts = (max(ious, dim=0) < iou_thresh).bool()
   // 找出未被teacher重新检测到的旧标签

5. tgts = tgts[~minus_tgts]            // 移除失效的旧标签
6. tgts = cat([tgts, add_tgts], dim=0) // 合并新旧标签


// 整体两阶段流程：
// Stage 1: Label Sampling (低分辨率 160x160)
x = GaussianNoise(3, 160, 160)
tgts = [random_single_label()]           // 按Table 6的均匀分布随机初始化
for step in range(N1):
    x = Adam(x, lr=1e-2).step(grad(L_total(x, tgts)))  // 优化图像
    if step % interval == 0:
        AdaptiveLabelSampling(x, tgts)    // 调用上述核心循环更新标签
y_star = tgts                             // 固定最终标签

// Stage 2: Image Synthesis (高分辨率 640x640)
x_new = GaussianNoise(3, 640, 640)
for step in range(2500):  // YOLOv5: 2500次迭代
    L = alpha_prior*L_prior + alpha_detect*(L_category + L_box + L_conf) + L_reg
    x_new = Adam(x_new, lr=1e-2, cosine_annealing)
// 输出：高质量task-specific合成校准集 {(x_new_i, y_star_i)}
```

关键设计决策：(1) 两阶段优于单阶段——标签持续变化导致图像优化目标不稳定（消融: 两阶段W6A6 mAP=32.1 vs 单阶段30.6）；(2) 低分辨率采样+高分辨率合成——节省计算且标签采样足够；(3) 最终标签固定后重新初始化高斯噪声——避免前期不准确标签污染图像。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在PyTorch中实现，teacher模型保持eval模式且参数冻结。超参数配置：(1) 初始随机标签：单目标，类别U(0,C)，bbox宽高U(0.2,0.8)，中心U(W/2,1-W/2)；(2) 标签采样分辨率160，合成分辨率640；(3) YOLOv5超参{alpha_detect, alpha_BN, alpha_TV, alpha_l2} = {0.5, 0.01, 0, 5e-4}，2500次迭代；(4) YOLO11超参={1e-3, 1e-3, 0, 5e-5}，3000次迭代；(5) Transformer-backbone Mask R-CNN超参={10.0, 1.0, 0, 1e-3}，4000次迭代；(6) Cutout增强提升多样性。生成2k张校准样本仅需1/60的原始训练数据量（MS-COCO 120k vs 2k），在8x RTX 4090上生成速度约256张/20分钟。消融证明该方法在完全data-free场景（无真实标签、无分布信息）下超越最优in-distribution proxy数据集2.3% mAP@W6A6，仅比使用真实标签差0.7% mAP。开源代码：https://github.com/DFQ-Dojo/dfq-toolkit。

涉及论文标题：
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

---

## Patch Similarity Entropy (PSE / Patch相似性熵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Patch Similarity Entropy (PSE) 是 PSAQ-ViT (ECCV 2022) 提出的损失函数，用于在 Vision Transformer (ViT) 的 Zero-shot Quantization 中替代 Batch Normalization Statistics (BNS) Loss 作为合成数据生成的先验约束。核心动机：ViT 模型使用 Layer Normalization (LN) 而非 BN，LN 不存储 running mean/variance 等运行时统计信息，因此 CNN 模型常用的 BNS Loss 无法直接应用于 ViT。PSE 的解决思路：利用 ViT 处理图像时 patch token 之间的 similarity 结构作为替代约束——计算合成图像在 ViT 各层产生的 patch token 之间的 cosine similarity 分布（通过 softmax 归一化为概率分布），并最小化该分布的熵与预训练模型在真实数据上计算的参考熵之间的差异。在 Task-Specific ZSQ for Object Detection 论文中，PSE 被用于 Transformer-backbone Mask R-CNN（Swin-T/S）模型的 L_prior。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
def compute_PSE_loss(model, synthetic_batch):
    """计算Patch Similarity Entropy损失"""
    loss = 0.0
    for layer_idx, blk in enumerate(model.blocks):
        # 获取该层的patch token表示
        x = blk.norm1(blk.attn.qkv(synthetic_batch))
        # 计算patch间的cosine similarity
        x_norm = F.normalize(x, p=2, dim=-1)         // (N, num_patches, dim)
        sim = x_norm @ x_norm.transpose(-2, -1)       // (N, P, P) similarity矩阵
        # 计算每个patch的相似性分布熵
        sim_probs = F.softmax(sim / tau, dim=-1)      // tau: temperature
        entropy = -(sim_probs * log(sim_probs + eps)).sum(dim=-1).mean()
        # 与预训练的参考熵对齐
        ref_entropy = model.ref_patch_entropies[layer_idx]
        loss += (entropy - ref_entropy) ** 2
    return loss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PSE 在 ZSQ 数据合成阶段替代 BNS Loss，用于所有 LN-based 模型（ViT、Swin Transformer等）。预训练的参考熵值 ref_patch_entropies 需要在真实数据上预先计算一次并保存。在 Task-Specific ZSQ for Object Detection 中，Swin-T/S backbone 的 Mask R-CNN 使用 PSE 作为 L_prior，超参数 {alpha_detect, alpha_PSE, alpha_TV, alpha_l2} = {10.0, 1.0, 0, 1e-3}，合成阶段优化4000次迭代。PSAQ-ViT 和 PSAQ-ViT V2 是关于 PSE 的原始论文。与 BNS Loss 相比，PSE 的优势是不依赖 BN 层，但劣势是需要预先计算参考熵值，且在极低分辨率下 patch 数量少时约束力减弱。

涉及论文标题：
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection
