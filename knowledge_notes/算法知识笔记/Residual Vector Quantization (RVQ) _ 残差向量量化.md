## Residual Vector Quantization (RVQ) / 残差向量量化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

残差向量量化（Residual Vector Quantization, RVQ）是一种多级向量量化方法。对输入向量 x，RVQ 使用 K 个级联码本：第一级找到码本 C_1 中最接近 x 的码字 $\hat{x}_1$，计算残差 $r_1 = x - \hat{x}_1$；第二级在 C_2 中找最接近 r_1 的码字 $\hat{r}_2$，累积近似 $\hat{x} = \hat{x}_1 + \hat{r}_2$；重复 K 次。最终 $\hat{x} = \sum_{k=1}^K \hat{c}_k$（K 个码本各选一个码字的和）。RVQ 的思想类似于梯度提升或残差学习。

在 VQLLM（Kumar 2024）中，RVQ 被用于 KV cache 压缩：将 key/value 向量按 channel group（d̂=32）分组，每组用独立的 K=8 级 RVQ 量化，每级 C=2048 个码字。码本通过 EMA（指数移动平均）在线更新。CommVQ 也采用了类似的"残差迭代"思想——通过 R 轮 EM 算法在上轮误差上拟合新码本（R=11 for 1-bit, R=21 for 2-bit）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**RVQ 标准编码/解码流程**：

```
def rvq_encode(x, codebooks):  # codebooks: list of K codebooks
    indices = []
    residual = x
    for k in range(K):
        idx = argmin(||residual - codebooks[k][j]|| for j in 0..C-1)
        indices.append(idx)
        residual = residual - codebooks[k][idx]
    return indices

def rvq_decode(indices, codebooks):
    x_hat = zero_vector
    for k in range(K):
        x_hat += codebooks[k][indices[k]]
    return x_hat
```

术语一般如何实现？如何使用？

RVQ 广泛应用于音频压缩（SoundStream、EnCodec）、图像生成（VQ-VAE-2）和 KV cache 压缩（VQLLM）。VQLLM 的 RVQ 实现使用非连续 channel grouping（对 Key 取间隔 d/d̂ 的通道），并在 Triton kernel 中融合 K 级查找/累加。RVQ 优势是编码质量随 K 增加而单调提升，但计算和存储开销也随 K 线性增长。CommVQ 的"残差迭代"与 RVQ 类似但实现不同：CommVQ 每轮使用完整 EM 聚类（而非单次最近邻查找），且所有 R 轮共享 g=64 的分组结构。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---
