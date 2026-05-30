## Block Low-Rank (BLR) Weight Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Low-Rank (BLR) 压缩是一种结构化权重矩阵压缩技术：将神经网络中一个大的dense权重矩阵W∈R^{i×o}划分为b₁×b₂个块（block），每个块W_{l,k}∈R^{p×q}（其中p=i/b₁, q=o/b₂）分别用低秩分解表示。与全局低秩分解（W=VU，对所有元素统一做rank-r分解）不同，BLR在各block内部独立rank分配，允许不同block捕捉不同程度的局部相关性。这使得BLR比标准low-rank具有更高的表达能力——在相同参数预算（压缩比）下保持更好准确率。

典型BLR方法的参数和计算复杂度：参数b₁b₂r'(p+q)，FLOP nb₁b₂r'(p+q)，n是序列长度。当b₁=b₂=b且r=r'b时渐近复杂度与标准low-rank相同：r(i+o)参数、nr(i+o) FLOP。与dense（i×o参数, n×i×o FLOP）相比，压缩比CF≈1.8-3×。

从算法pipeline角度拆解术语，给出具体例子。
以b₁=16, b₂=16, p=256, q=256, r'=64为例：

```
# 离线阶段：将dense权重W∈R^{4096×4096}分解为BLR参数
for l in range(16):
  for k in range(16):
    W_{l,k} ∈ R^{256×256} ≈ V_{l,k}·U_{l,k}
    # V_{l,k} ∈ R^{256×64}, U_{l,k} ∈ R^{64×256}

# 在线推理 (n=1024 tokens):
Input: X ∈ R^{1024×4096}
X_blocks = X.view(1024, 16, 256)

for k in range(16):  # 每个输出block
    Y_k = zeros(1024, 256)
    for l in range(16):  # accumulate所有输入block的贡献
        Y_k += (X_blocks[:,l,:] @ V_{l,k}) @ U_{l,k}
        # [1024,256]@[256,64]=[1024,64]; [1024,64]@[64,256]=[1024,256]

Y = concat([Y_0,...,Y_15], dim=-1)  # [1024, 4096]
```

术语一般如何实现？如何使用？
BLR训练模式：从头训练（BLR权重参数化初始化）或压缩预训练模型（block-wise SVD for Monarch、preconditioned gradient descent 300步 for BLAST）+fine-tune。BLR的核心tradeoff：更多block→更细粒度的表达能力→更好精度，但也产生更多b×n×r中间张量数据移动（如b=16, n=1024, r=1024时128MB BF16）。在多token推理中，这额外的数据移动可能将compute-bound线性层推入memory-bound→需kernel级优化恢复性能。标准低秩（全局单rank）数据移动最优但高压缩比下精度急剧下降——BLR通过块结构在精度和效率间寻找更优平衡点。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
