## Expert Activation Matrix (专家激活矩阵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Activation Matrix（专家激活矩阵）是从 MoE LLM 中提取的、记录每个专家在每个输入样本上激活强度的二维数据矩阵，是分析 MoE 模型内部行为的基础数据结构。给定 m 层、每层 n 个专家的 MoE LLM（总 $N_e = m \times n$ 个专家），在 $N_s$ 个样本前向传播时，对每个 token 的 Router softmax 输出进行记录和聚合，构造矩阵 $X \in \mathbb{R}^{N_e \times N_s}$，其中 $X_{e,i}$ 表示第 e 个专家在第 i 个样本上的句子级总激活强度。

构造过程：对样本 $S_i$ 的第 t 个 token，Router 为第 j 层的第 k 个专家分配权重 $\alpha(i)_{t,j,k}$。句子级激活通过求和聚合：$v_{i,j,k} = \sum_{t=1}^{T} \alpha(i)_{t,j,k}$（式 1）。将每个 $(j,k)$ 展开为一行、每个样本为一列，得到 $X \in \mathbb{R}^{N_e \times N_s}$。矩阵具有非负性（激活值在 [0,T] 范围）和天然稀疏性（每 token 仅激活 top-k 个专家）。

从算法pipeline角度拆解术语：

```
for each sample s_i in S:
    for each token t in s_i:
        for each layer j in 1..m:
            alpha_j = softmax(W_router[j] @ x_t)   # Router 输出
            for each expert k in 1..n:
                record alpha(i)_{t,j,k}

    for each (j,k):
        v_{i,j,k} = sum_{t=1}^{T} alpha(i)_{t,j,k}  # 式(1): 句子级聚合

X[e, i] = v_{i,j,k}   where e = index(j, k)
# X shape: (N_e, N_s), N_e = m * n
```

术语一般如何实现？如何使用？

使用 PyTorch forward hook 在 MoE 层 Router softmax 后捕获激活值。对 shared experts 激活值恒为 1。激活矩阵可用于：(1) HSDL 分解发现协作模式；(2) 专家使用频率统计；(3) 领域偏好分析（计算不同领域激活分布的 cosine similarity）；(4) CAEP 剪枝的贡献评估。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
