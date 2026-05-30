## Learnable Binary Mask

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Learnable Binary Mask（可学习二值掩码）是一种数据驱动的参数重要性评估与选择方法。在神经网络剪枝和参数继承场景中，传统的启发式重要性度量（L1-norm, L2-norm, Taylor expansion）基于固定的数学公式判断哪些权重重要，而这些指标可能与实际任务需求不完全一致。Learnable Binary Mask将"哪些参数重要"本身作为一个可优化的问题：为每个待选择的权重引入一个可学习的二值mask参数，通过任务损失反向传播来自动学习哪些权重应该被保留（mask=1）或丢弃（mask=0）。训练完成后，仅保留mask=1对应的权重作为小模型的初始化参数。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# 输入: 大模型某层权重 W ∈ R^{d_out × d_in}, 输入 x, 目标压缩比
# 输出: 压缩后的权重 W_compressed

# 1. 初始化可学习mask参数
α = nn.Parameter(torch.randn(d_out, d_in) * 0.01 + 2.0)  # 正偏置使初始mask接近1

# 2. Gumbel-Sigmoid 可微二值化
def gumbel_sigmoid(α, τ, training):
    if training:
        # 采样Gumbel噪声: g = -log(-log(u + ε) + ε), u ~ Uniform(0,1)
        u = torch.rand_like(α).clamp(min=1e-8, max=1-1e-8)
        g1 = -torch.log(-torch.log(u))
        g2 = -torch.log(-torch.log(1 - u))
        # Gumbel-Sigmoid: M = σ((α + g1 - g2) / τ)
        logits = (α + g1 - g2) / τ
    else:
        logits = α / τ  # 推理时不用噪声
    return torch.sigmoid(logits)

# 3. 训练循环
optimizer = AdamW([α], lr=1e-3)
τ_start, τ_end = 1.0, 0.1  # temperature annealing
for step in range(max_steps):
    τ = τ_start * (τ_end/τ_start) ** (step/max_steps)  # 指数退火
    
    M = gumbel_sigmoid(α, τ, training=True)
    # 前向: 被mask的权重输出≈0
    h = (W.detach() * M) @ x  # W固定，只训练mask
    
    # 损失 = 任务损失 + 稀疏正则
    L_task = cross_entropy(classifier(h), y)
    L_sparse = λ * M.mean()  # 推动mask稀疏化
    L = L_task + L_sparse
    
    # STE反向传播: 梯度通过不可微的M传递
    L.backward()  # ∇_α L 通过STE计算
    optimizer.step()

# 4. 最终二值化并提取
M_final = (gumbel_sigmoid(α, τ=0.01, training=False) > 0.5).float()
# 提取: 选取mask=1的行索引和列索引
row_idx = M_final.sum(dim=1) > 0  # 保留有任意输入连接的行
col_idx = M_final.sum(dim=0) > 0  # 保留有任意输出连接的列
W_compressed = W[row_idx][:, col_idx]
```

核心机制：
- **Gumbel-Softmax/Straight-Through Estimator**：Gumbel噪声使采样过程可微（前向离散采样，反向梯度直通），temperature τ控制mask的离散程度（τ→0时M趋于0/1二值）
- **稀疏正则**：L1正则λ·Σ|M|推动mask稀疏化，λ控制压缩率
- **与权重解耦**：只训练mask参数，不修改原始大模型权重，保证预训练知识不被破坏

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
1. **MLPruner (PeerJ 2025)**：为每个卷积滤波器关联可学习二值mask，STE处理不可微取整，实现对ResNet/VGG的无损剪枝（54.8% FLOPs reduction, 93.31% top-1 on CIFAR-10）。
2. **Piggyback (ECCV 2018)**：在固定预训练权重上学binary mask，实现单网络适配多任务（每任务仅1 bit/参数的存储开销）。
3. **SCL (IEEE TNNLS 2023)**：将权重重参数化为weight ⊙ binary_mask，证明STE代理梯度须为正数才能收敛，提出LeakyReLU/Softplus/identity STE作为有效选项。
4. **本论文的使用**：在参数继承的Intra-layer阶段使用Gumbel-Sigmoid learnable mask识别PanGu-π-7B中的重要神经元，相比L1/L2/Taylor固定标准提升1-2个Avg百分点。
5. **优势**：端到端可训练、任务感知（task-aware）、无需手工设计重要性度量标准。

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models
