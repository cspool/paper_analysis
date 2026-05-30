## Stride-aware Causal Mask

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Stride-aware Causal Mask 是 MTLA 为实现在 parallel training 下匹配 incremental inference attention pattern 而设计的一种特殊 causal mask。MTLA 在推理时每 s 个 token 共享一组 compressed KV cache vector（temporal compression ratio=s），导致 attention mask 不是标准的三角因果掩码。若训练时简单 pre-downsample KV cache，则 query 在训练时看到的 KV 信息与推理时不一致（覆盖了尚未合并的 incomplete vectors）。Stride-aware mask 通过限制 query 仅能 attend stride 边界上的历史位置，使训练-推理 attention pattern 一致。

Stride-aware causal mask 定义（row m, col n, stride s）：mask[m, n] = 0 iff (n == m) or (n < m and n % s == 0)，否则 mask[m, n] = -∞。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**stride-aware mask 可视化（T=6, s=2）**：

```
# mask ∈ R^{6×6}, 0=允许attend, -∞=masked
#   col: 1  2  3  4  5  6
# row 1:  0 -∞ -∞ -∞ -∞ -∞   (仅自身)
# row 2:  0  0 -∞ -∞ -∞ -∞   (自身 + col 2=stride boundary)
# row 3:  0 -∞  0 -∞ -∞ -∞   (自身 + col 2)
# row 4:  0  0 -∞  0 -∞ -∞   (自身 + cols 2,4)
# row 5:  0 -∞  0 -∞  0 -∞   (自身 + cols 2,4)
# row 6:  0  0 -∞  0 -∞  0   (自身 + cols 2,4,6)
```

构造伪代码：
```
mask = full(T, T, -inf)
for m in 1..T:
    for n in 1..T:
        if n == m or (n < m and n % s == 0):
            mask[m, n] = 0
```

术语一般如何实现？如何使用？

GPU 上通过 `torch.where` 或 attention kernel 内联条件分支实现。开销与标准 causal mask 同阶 O(T²)，可 fused 进 softmax kernel。仅训练时使用；推理时 incremental cache update 自然满足 stride pattern。超参数：temporal compression ratio s。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention

---
