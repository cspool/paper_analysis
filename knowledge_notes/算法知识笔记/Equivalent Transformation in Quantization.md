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
