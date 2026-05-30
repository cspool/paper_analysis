## Monarch Matrix Decomposition

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Monarch是Dao et al. (2022, ICML)提出的结构化矩阵分解方法，属于BLR的一种。原始定义：M = P₁ L P₂^T R，L和R是block-diagonal矩阵，P₁和P₂是固定permutation矩阵。Monarch将dense权重划分为b₁×b₂个块，每块独立低秩分解W_{l,k}=V_{l,k}U_{l,k}。关键特征是两次permutation（r'↔b₂→b₂↔b₁）实现跨block信息混合。常用配置b₁=b₂=b=4-16。参数b₁b₂r'(p+q)，FLOP nb₁b₂r'(p+q)。

从算法pipeline角度拆解术语：
```
# Monarch权重: V∈R^{b₁×(r'b₂)×p}, U∈R^{b₂×q×(b₁r')}
# b₁=b₂=16, p=q=256, r'=64

X_blocks = X.view(n, 16, 256)             # [n, b₁, p]
Z = batched_bmm(X_blocks, V^T)             # [b₁, n, r'b₂]

# ↑ 基线性能瓶颈: 两次permutation ↓
Z = Z.reshape(16, n, 16, 64)              # b₁→b₁, n→n, r'b₂→(b₂, r')
Z = Z.transpose(0,2).transpose(1,2)        # → [b₂, n, b₁·r']
# ↑ 需要clone tensor, uncoalesced access

for k in range(16):
    Y_k = Z[k] @ U[k]                     # [n, 1024]@[1024, 256]
Y = final_permute(stack(Y_k))              # [b₂,n,q]→[n,q,b₂]
```

术语一般如何实现？如何使用？
开源：https://github.com/HazyResearch/monarch。训练方式：从头训练Monarch参数化模型或压缩预训练权重。Monarch在ViT-B CF=3×下ImageNet=79.2% vs low-rank 78.9%；GPT2-S CF=1.85×下WikiText-103 PPL=21.1 vs low-rank 21.7。主要问题：多token推理(n=1024)时两次permutation kernel+4bnr bytes中间数据使实际速度比dense慢1.14-1.68×（A40），需通过Triton kernel的permutation fusion和V重排布优化恢复性能。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
