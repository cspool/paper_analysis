## Parameter Inheritance

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Parameter Inheritance（参数继承）是一种从小模型初始化的技术，核心思想是将大语言模型（如7B）学到的权重和表征能力"传递"给要训练的小模型（如1B），让小模型从大模型的参数出发开始训练，而非从随机初始化开始。这不同于知识蒸馏（knowledge distillation）——蒸馏需要teacher模型在训练过程中持续提供监督信号；而参数继承是一次性的初始化操作，继承后小模型独立训练。该方法包含两个阶段：(1) Layer Selection——从小模型层数少、大模型层数多的约束出发，通过layer skipping实验识别并保留大模型中对性能最关键的首尾层，移除冗余的中间层；(2) Intra-layer Parameter Selection——在保留的各层内，通过可学习二值掩码（learnable binary mask）自动识别并选择对任务最重要的神经元/参数，形成小模型的初始权重。

从算法pipeline角度拆解术语，给出术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 大模型 W_large（PanGu-π-7B, L_large层），目标架构 A_small（L_small层, d_small宽）

# Stage 1: Layer Selection
perf_drop = []  # 每层的性能下降
for layer_i in range(L_large):
    # Layer skipping实验：跳过layer_i，评估下游任务性能
    model_skipped = copy(W_large); model_skipped.skip_layer(layer_i)
    perf_i = evaluate(model_skipped, [ARC-E, HellaSwag, C3])
    perf_drop.append(perf_base - perf_i)

# 发现：前2-3层和最后几层perf_drop大（关键层），中间层perf_drop小（冗余）
# 选择策略：保留前k_begin层 + 中间均匀采样 + 保留最后k_end层
selected_layers = [0,1,2] + uniform_sample(middle_layers, L_small - k_begin - k_end) + [L_large-3, L_large-2, L_large-1]

# Stage 2: Intra-layer Parameter Selection via Learnable Mask
for each selected_layer l:
    W = W_large[l]  # 该层权重矩阵, shape (d_out, d_in)
    # 初始化可学习参数 α ∈ R^{d_out × d_in} (log-probabilities)
    α = init_normal(mean=2.0, std=0.01)  # 偏向mask=1的初始化
    
    # Gumbel-Sigmoid可微二值化
    for step in mask_training_steps:
        u = uniform(0, 1, shape=(d_out, d_in))
        g = -log(-log(u + ε) + ε)  # Gumbel noise
        logits = α + g
        M = sigmoid(logits / τ)     # temperature τ anneal from 1.0 to 0.1
        # 前向传播: h = (W ⊙ M) @ x
        # 损失: L = L_task(h, y) + λ * ||M||_1  (稀疏正则)
        α = α - η * ∇_α L  # 更新mask参数（STE直通估计器）
    
    M_binary = (M > 0.5).float()  # 最终二值化
    # 提取子矩阵: 按mask=1的行/列选取参数
    W_small[l] = extract_submatrix(W, M_binary)

# Stage 3: 用小模型架构初始化
θ_init = assemble_model(selected_layers, W_small, A_small)
# 开始小模型训练
θ_final = train(θ_init, data=1.6T tokens)
```

结果表明：Learnable Mask (Avg=48.08) > Taylor (47.90) > L2 (47.00) > L1 (46.06) > Random Init (42.06)，参数继承是最有效的单一优化组件（+3.26贡献）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现与使用：
1. **Layer Skipping实验**：对多个大模型(LLaMA2-7B/13B, InternLM-7B, PanGu-π-7B)进行skip-1/skip-2/skip-3层实验，发现普适规律——首尾层关键、中间层冗余。
2. **Mask学习方法**：使用Gumbel-Sigmoid重参数化技巧（通过温度τ控制离散化程度，训练中从τ=1.0逐步退火至0.1）实现可微的二值mask训练；反向传播使用Straight-Through Estimator（STE）处理不可微的取整操作。
3. **相关技术对比**：
   - ParaKnowTransfer (ICLR 2024)：参数敏感度+LoRA注入实现7B→13B参数迁移
   - GeneLLM/Learngene：从MoE模型中提取1.25%参数初始化小模型，保留80%+性能
   - CoMe (NeurIPS)：通过层拼接（而非选择+裁剪）逐层压缩
4. **适用前提**：需要同架构系列的大模型checkpoint作为初始化来源（如LLaMA→TinyLLaMA、PanGu-π-7B→PanGu-π-1B Pro）

涉及论文标题：
- PanGu-π Pro: Rethinking Optimization and Architecture for Tiny Language Models
