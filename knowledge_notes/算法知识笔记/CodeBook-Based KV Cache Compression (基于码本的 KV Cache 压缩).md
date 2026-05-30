## CodeBook-Based KV Cache Compression (基于码本的 KV Cache 压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

CodeBook-Based KV Cache Compression 是一种利用向量量化 (Vector Quantization) 方法压缩 KV Cache 的技术。核心思想：将 KV Cache 中高度相似的 Key/Value 向量聚类为少数"码本条目" (codebook entries)，仅存储码本 ($C_K$, $C_V$)、每个 token 到码本条目的索引 ($r_K$, $r_V$，整数类型) 和每个 token 的 L2 magnitude ($m_K$, $m_V$，浮点类型)，推理时通过查表+缩放重建原始向量：$\Gamma_r = C_\Gamma[r_\Gamma] \otimes m_\Gamma$。

SpindleKV 首次在 KV Cache 压缩中提出基于余弦相似度的贪心码本构建方法。与传统 VQ 需要离线训练不同，SpindleKV 的码本是在 prefill 阶段 Just-in-Time (JIT) 在线构建的，无需额外训练。构建过程：(1) 对保留的 KV cache 归一化（除以 L2 magnitude）；(2) 计算 token 间余弦相似度矩阵 $S_\Gamma$；(3) 设定阈值 $\theta_\Gamma$ 构建邻接矩阵 $G_\Gamma = \text{where}(S_\Gamma > \theta_\Gamma, 1, 0)$；(4) 贪心迭代：每次选图中度数最高节点加入码本，将其邻居映射到该码本条目，从图中移除已覆盖节点；(5) 记录每个 token 的 L2 magnitude 用于重建。

这项技术专门针对 KV Cache 在浅层中的"构成性冗余"——浅层 token 之间 KV 向量余弦相似度极高（超过 0.9），但这些 token 各自都获得较高注意力分数，传统 eviction 方法无法有效压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SpindleKV 码本构建伪代码（Algorithm 1）**：

```
Input:  归一化后的 KV Cache Γ_r [l_c, h, d_h]
        Key 阈值 θ_K = 0.98, Value 阈值 θ_V = 0.95
Output: CodeBook C_Γ, 引用索引 r_Γ, Magnitudes m_Γ

C_Γ = []                              # 空码本
r_Γ = [-1, -1, ..., -1]              # 每个 token 的引用，-1 表示未分配
m_Γ = L2_Norm(Γ, dim=-1)              # 记录原始 magnitude
Γ_r = Γ_r / m_Γ                       # 归一化到单位向量

S_Γ = cos_sim(Γ_r, Γ_r)              # [N, N] 余弦相似度矩阵
G_Γ = where(S_Γ > θ_Γ, 1, 0)         # 邻接矩阵：相似度 > 阈值 → 有边

while G_Γ != 0:                      # 直到所有 token 都被覆盖
    s_Γ = sum(G_Γ, dim=1)            # 每个节点的度数
    ι = argmax(s_Γ)                  # 选度数最高的 token
    C_Γ.append(Γ_r[ι])               # 加入码本
    η_ι = argwhere(G_Γ[ι] == 1)      # 邻居
    r_Γ[η_ι] = len(C_Γ) - 1          # 引用指向码本
    mask_Γ = matmul(¬G_Γ[ι]^T, ¬G_Γ[ι])
    G_Γ = G_Γ & mask_Γ               # 移除已覆盖节点

# 推理时重建
Γ_reconstructed = C_Γ[r_Γ] * m_Γ    # 查码本 × 恢复 magnitude
# 对重建后的 K 重新应用 RoPE
```

**KV Cache 最终存储空间计算**：

$$r^\lambda = r_1^\lambda \times r_2^\lambda \times r_3^\lambda$$

其中 $r_1^\lambda$ 是 eviction 保留率，$r_2^\lambda = |C_K^\lambda \cup C_V^\lambda| / \sum(|K_{j,r}^\lambda| + |V_{j,r}^\lambda|)$ 是码本压缩率，$r_3^\lambda$ 是 dtype 转换率（索引用 int、magnitude 用 float 替代完整 FP16 向量）。

术语一般如何实现？如何使用？

SpindleKV 开源实现见 https://github.com/tyxqc/SpindleKV。码本构建仅在 prefill 阶段执行一次。超参数：Key 阈值 $\theta_K = 0.98$，Value 阈值 $\theta_V = 0.95$。实验中仅用码本（无 eviction）即可压缩 50% KV Cache 而准确率无损，验证了浅层构成性冗余假说。该方法与 eviction 方法互补——eviction 处理深层注意力稀疏性，码本处理浅层向量相似性。

GQA 兼容性：对 GQA 模型，SpindleKV 先将 KV head 展开（repeat $h_n$ 次）再构建码本。Expand 引入的重复向量余弦相似度为 1，极容易被码本合并，额外开销被消除。

涉及论文标题：
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

---
