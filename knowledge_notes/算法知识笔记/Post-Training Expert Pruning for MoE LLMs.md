## Post-Training Expert Pruning for MoE LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-Training Expert Pruning 是一种针对 MoE LLM 的后训练压缩技术，通过逐层移除不重要的专家来减少模型参数量和部署内存。与传统 weight pruning（Wanda, SparseGPT）不同，Expert Pruning 将稀疏化粒度从"权重元素"提升到"专家"级别：每层从 n 个专家中保留 r 个最重要的，永久丢弃其余 n−r 个专家及其路由权重。核心方法：使用小规模校准数据集（如 C4 128 条 × 2048 tokens），对每层枚举所有 C(n,r) 种专家组合，以最小化 Frobenius 范数重构损失 ‖F'(x,C) − F(x)‖_F 为目标选择最优子集。其数学表达为：min_C ‖F'(x,C) − F(x)‖_F, s.t. C ⊆ {expert_0,...,expert_{n-1}}, |C|=r。由于每层专家数较小（Mixtral 8x7B 的 n=8），枚举组合数 C(8,4)=70 / C(8,6)=28 完全可行。剪枝后模型可通过修改 config 中的 expert 数量直接使用 HuggingFace Transformers 加载，无需修改模型代码。Mixtral 8x7B 剪枝耗时：r=6 约 30 分钟，r=4 约 90 分钟。支持通用剪枝（C4 校准）和领域特定剪枝（将校准数据切换到目标领域数据集如 MATH，提升数学任务剪枝效果——GSM8K 5-shot r=6: 41.02 vs 51.25）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Pruning pipeline：
```
# Step 1: 校准数据前向传播
for layer l in 1..L:
    for batch in calib_data:             # 128 seqs × 2048 tokens
        x[l], F_l(x) = forward_cache(l)  # 缓存每层输入-输出对

# Step 2: 逐层枚举剪枝
for layer l in 1..L:
    best_loss = inf
    for C in Combinations({expert_0,...,expert_{n-1}}, r):
        # 构造剪枝后 MoE 层（丢弃 n-r 个专家及路由权重）
        F'(x, C) = Σ_{j=0}^{r-1} w̃_{e_j}·E_{e_j}(x)
        w̃_{e_j} = w_{e_j} / Σ_{m=0}^{r-1} w_{e_m}
        loss = ‖F'(x[l], C) − F_l(x[l])‖_F    # Frobenius 重建损失
        if loss < best_loss:
            best_experts[l] = C

# Step 3: 修改模型配置，仅加载保留的专家
config.num_local_experts = r
# 剪枝后 checkpoint 中仅包含 r 组 expert 权重
```
与 weight pruning 的本质区别：Weight pruning 产生稀疏矩阵需专用硬件加速；Expert pruning 直接减少模型层中的子网络数量，在标准 GPU 上即插即用。剪枝 r=6 时 Mixtral 8x7B 内存从 89,926MB 降至 68,383MB（24% 减少），单张 80G GPU 可部署；r=4 时降至 46,879MB（48% 减少）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/Lucky-Lance/Expert_Sparsity。基于 HuggingFace Transformers + PyTorch 实现。使用场景：(1) 通用部署压缩——C4 校准，适应广泛任务；(2) 领域特定压缩——用目标领域数据（如 MATH）校准，提升数学/代码等专业任务剪枝效果。限制：(1) 枚举法复杂度 O(C(n,r)·L)，专家数 n≫8 时不可行（如 32 专家时 C(32,16)≈6×10^8）；(2) 剪枝后通常需微调恢复性能（论文用 MetaMathQA 微调 900 步恢复数学能力）；(3) 仅验证 Mixtral 8x7B 架构，未测试其他 MoE 变体（DeepSeekMoE shared experts 等）。

涉及论文标题：
- MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

---
