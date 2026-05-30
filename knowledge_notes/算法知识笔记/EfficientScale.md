## EfficientScale

术语解释
EfficientScale 是 BAAI 在 AquilaMoE 中提出的两阶段 MoE 高效训练方法，通过 Scale-Up（小模型权重初始化大模型）和 Scale-Out（dense 模型转换为 MoE）两个阶段，用已有预训练权重引导大模型训练，避免从头训练的高昂计算成本。

术语是什么？
EfficientScale 由三个阶段的 pipeline 组成：
1. **Preparation Phase**: 从头训练小 dense 模型（或加载已有预训练权重），准备训练数据
2. **Scale-Up Phase**: 使用小模型的 weights 通过 AKI-Pro 初始化大 dense 模型，大幅降低初始 validation loss，然后连续预训练
3. **Scale-Out Phase**: 使用 Sparse Upcycling 将大 dense 模型转换为 MoE（每个 MLP 层复制为 8 个 expert + 随机初始化 router），再连续预训练 MoE

实际案例：AquilaDense-7B (3.6T tokens) → Scale-Up → AquilaDense-16B (1.2T tokens) → Scale-Out → AquilaMoE 8×16B (545B tokens)。相比从头训练 32B MoE (5345B tokens, 213.8 GPU-days)，EfficientScale 仅需 51.84 GPU-days，时间节省 4.12×，算力节省 3.35×。

从算法pipeline角度拆解术语：
```
# EfficientScale Pipeline
# Phase 1: Preparation
small_dense = train_from_scratch("M(32,4096)", tokens=3.6T)  # AquilaDense-7B

# Phase 2: Scale-Up
# 2a: AKI-Pro 初始化
large_dense = AKI_Pro_init(small_dense, target="M(40,5120)")
# 宽度: AKI 利用相邻层权重打破对称性
# 深度: Interpolation W'_l = floor(l * L_2 / L_1)
# GQA: 将每个 group 视为独立 MHA block 扩展

# 2b: 连续预训练
large_dense = train(large_dense, tokens=1.2T, lr=4.0e-4)  # AquilaDense-16B

# Phase 3: Scale-Out
moe_model = deepcopy(large_dense)
for layer in moe_model.layers:
    experts = [copy(layer.mlp) for _ in range(8)]  # 复制 dense MLP
    layer.moe = MoELayer(experts, router=Linear(hidden_dim, 8, N(0, 0.02)))
moe_model = train(moe_model, tokens=545B, lr=1.5e-4)  # AquilaMoE 8×16B
```

术语一般如何实现？如何使用？
- 前提：有高质量小模型 checkpoint；适用于从零开始训练成本极高的场景
- Scale-Up 阶段验证 loss 显著降低：AKI-Pro initialization loss 7.81 vs random init 12.22 at M(32,4096)
- Scale-Out 使用 Sparse Upcycling，experts 初始化为 dense MLP 复制，router 随机初始化
- 训练期间加 load balancing loss (λ=0.001) 和 max z-loss (λ=0.01) 防止崩溃
- 硬件：Preparation 阶段 480 × ~990 GFLOPS GPU，Scale-Up/Scale-Out 阶段 1024 × 240 GFLOPS accelerators
- 代码开源：https://github.com/FlagAI-Open/Aquila-MoE，模型权重：https://huggingface.co/BAAI/AquilaMoE

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---
