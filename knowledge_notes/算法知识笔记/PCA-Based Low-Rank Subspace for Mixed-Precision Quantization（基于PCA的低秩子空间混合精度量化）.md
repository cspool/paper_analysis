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
