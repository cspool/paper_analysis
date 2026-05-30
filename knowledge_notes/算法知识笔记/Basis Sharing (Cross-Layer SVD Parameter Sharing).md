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
