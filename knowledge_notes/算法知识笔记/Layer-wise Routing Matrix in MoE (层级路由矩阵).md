## Layer-wise Routing Matrix in MoE (层级路由矩阵)

术语解释
Layer-wise Routing Matrix 指 MoE 模型中不同 Transformer 层使用**独立的路由参数矩阵**，而非所有层共享同一个路由器。AT-MoE 论证了不同层关注不同抽象级别特征（低层偏基础领域知识，高层偏功能性和风格性特征），因此需要为每层训练独立的路由矩阵。

术语是什么？
在标准 MoE 中，每层虽各有独立的路由权重 W_r^(l) ∈ R^{d×N}（Switch Transformer 等主流架构均如此），但路由逻辑相同：均为 Linear→Softmax→TopK，未针对不同层的特征偏好进行差异化设计。AT-MoE 明确将 layer-wise 路由作为设计原则：对于有 N_T 个 Transformer block 的模型，使用 N_T 组独立的路由矩阵对 (W_G^(l), W_D^(l))，l = 1...N_T。其假设是：
- **低层（1~N_T/3）**：关注基础领域知识特征，路由偏向领域知识类专家组
- **中层（N_T/3~2N_T/3）**：关注综合分析特征，路由在各组间均衡分配
- **高层（2N_T/3~N_T）**：关注功能性和风格性特征，路由偏向功能类和风格类专家组

从算法pipeline角度拆解术语。
```
# Layer-wise Routing: 不同层使用不同路由参数
# Layer 1 (低层): 偏重领域知识
W_G^(1) → 领域组 0.5, 功能组 0.3, 风格组 0.2
# Layer 16 (中层): 均衡
W_G^(16) → 领域组 0.35, 功能组 0.35, 风格组 0.3
# Layer 32 (高层): 偏重功能和风格
W_G^(32) → 领域组 0.15, 功能组 0.5, 风格组 0.35

# 每层的完整 forward:
for l in range(N_T):
    x = Attention_LayerNorm(x) + MHA(x)
    F_G_l = adaptive_grouped_routing(x, W_G^(l), W_D^(l), ...)
    y = (λ * F_G_l + (1-λ) * W_p) @ x + W_0 @ x
    x = x + y  # residual
```

这种设计的理论基础来自 Gao et al. (2024) 的发现：高层学习更抽象和高级的信息，这些特征用于下游任务 ("Higher layers need more LoRA experts")。

术语一般如何实现？如何使用？
- 实现为 N_T 个独立的路由参数矩阵集合 {(W_G^(l), W_D^(l)) | l = 1...N_T}
- 训练时对每组路由矩阵分别训练（所有 LoRA 专家冻结）
- 路由矩阵的总参数量 = N_T × (N_dim × N_G + N_G × N_M)，相比 LLM 总参数量可忽略
- 可与 grouped routing 结合使用，也可独立应用
- 目前无开源实现，论文未提供实验验证
- 相关研究：PathMoE (2026) 共享相邻 block 的路由参数以减少路径空间；UniPool (2026) 共享全局专家池但保留独立路由；Omni-Router (2025) 全层共享路由器

涉及论文标题：
- AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach

---
