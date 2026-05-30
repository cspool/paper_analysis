## BLAST (Block-Level Adaptive Structured Matrices)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BLAST是Lee et al. (2024, NeurIPS)提出的通用化BLR结构。与Monarch每block独立分解不同，BLAST引入三种因子共享：每个输入block l共享左因子V_l∈R^{p×r}，每个输出block k共享右因子U_k∈R^{r×q}，跨block交互由per-block对角矩阵S_{l,k}∈R^{r×r}建模。完整分解：W_{l,k}=V_l·S_{l,k}·U_k。统一表达性：通过设置S_{l,k}可恢复标准低秩（S全部相同）、Monarch（S为特定结构化pattern）、block-diagonal（S_{l,k}=0 for l≠k）等。参数r(p+q+b²)，FLOP nr(p+q+b²)，b²项通常可忽略（b≤16）。

从算法pipeline角度拆解术语：
```
# BLAST权重: V∈R^{16×256×1024}, S∈R^{16×16×1024}, U∈R^{16×1024×256}

X_blocks = X.view(n, 16, 256)
Z_l = batched_bmm(X_blocks, V)             # [16, n, 1024]

for k in range(16):
    Y_k = zeros(n, 256)
    for l in range(16):
        Y_k += (Z_l[l] * S[l,k]) @ U[k]   # Hadamard ⊙ + bmm
    Y.append(Y_k)
return concat(Y, dim=-1)                   # [n, 4096]
```

术语一般如何实现？如何使用？
开源：https://github.com/changwoolee/BLAST；HuggingFace：https://huggingface.co/cwoolee/blast-llama-4B。压缩通过preconditioned gradient descent（300步）分解dense权重→fine-tune。BLAST精度最优：Llama-7B CF=2× WikiText-2 PPL=14.21 vs Monarch 19.54 vs LR 26.33；ViT-B CF=3× ImageNet=79.3%略高于Dense 78.7%。代价是多token推理PyTorch基线性能最差——比dense慢2.63-4.31×（A40），因为两组中间张量(8bnr bytes)+两组permutation。论文Triton kernel优化⑤（permutation-only fusion with tensor core）是实现BLAST实用化的关键。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
