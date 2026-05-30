## RoPE-Commutative Codebook (RoPE-可交换码本)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

RoPE-可交换码本是 CommVQ 中针对 Key Cache 设计的特殊码本结构。由于 Key 向量在 self-attention 中会经过 RoPE 旋转，标准 AQ 解码后还需单独应用 RoPE，导致解码开销与 self-attention 叠加。RoPE-可交换码本利用 RoPE 矩阵的 2x2 块对角特性：一个 $2 \times 2$ 矩阵 $C = \begin{pmatrix} x & y \\ -y & x \end{pmatrix}$ 与 RoPE 旋转矩阵 $R_m^i = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$ 满足交换律 $R_m^i C = C R_m^i$。通过在 2D 子空间中设计满足该形式的子码本 $C_K^{jl}$，使 key-query attention 计算 $\alpha_i = q R_t (s_i C_K R_i)^T$ 可改写为 $\alpha_i = \sum_{j,l} (q^j R_t^j) C_K^{jlT} R_i^{jT} [s_i^j=l]^T$，其中 $(q^j R_t^j) C_K^{jlT}$ 对所有 token i 仅需计算一次，解码从 $O(d N_c N)$ 降为与 self-attention 同量级的开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**RoPE-可交换码本的 decoding 流程（单层单头 decoding step）**：

```
// 输入：当前 query q [1, d]，量化 key cache S_K [N, d/2]，2D 子码本 C_K^j_l [2, 2]
// 预计算：对所有 j, l 计算 (q^j R_t^j) C_K^{jlT}，仅需一次
q_proj = q @ W_Q              // [1, d]
q_rope = apply_rope(q_proj)   // 按 2D 子空间旋转

// 预计算复用部分
for j in 0..d/2:              // 遍历 d/2 个 2D 子空间
    q_j = q_rope[2*j:2*j+2]   // [2]
    for l in 0..N_c'-1:       // 遍历每个量化级别
        precomp[j][l] = q_j @ C_K^j_l^T  // [2] @ [2,2]^T -> [2]

// 逐 token 计算 attention score
for i in 0..N-1:
    alpha[i] = 0
    for j in 0..d/2:
        s_val = S_K[i][j]      // 量化索引，每维度 ∈ [0, N_c'-1]
        alpha[i] += dot(precomp[j][s_val[0]], rope_rotate_i(s_val[0]))
                  + dot(precomp[j][s_val[1]], rope_rotate_i(s_val[1]))
```

**聚类中心定义**：码本的 $N_{c'}^2$ 个聚类中心定义为：
$$c_{a,b} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} C_K^j[a] + \begin{bmatrix} 0 \\ 1 \end{bmatrix} C_K^j[b]$$

其中 $C_K^j[a]$ 和 $C_K^j[b]$ 是第 a 和第 b 个 $2 \times 2$ 可交换子码本。每个 2D 子向量被分配到最近聚类中心，用索引对 {a, b} 作为量化表示。

术语一般如何实现？如何使用？

子码本通过 EM 算法在 FineWeb-Edu 校准集上训练（含 soft clustering assignment + temperature annealing 稳定训练）。为提升压缩率，将连续 g 个 2D 子空间分为一组共享量化向量 s，Avg. bit = $R \cdot \log_2(N_{c'}) / g$（R 为残差迭代次数）。CommVQ 配置：$N_{c'}=64$, $g=64$，1-bit 时 R=11，2-bit 时 R=21。Key 码本总大小仅 2.75 MB (1-bit) / 5.25 MB (2-bit)，与 token 数无关。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---
