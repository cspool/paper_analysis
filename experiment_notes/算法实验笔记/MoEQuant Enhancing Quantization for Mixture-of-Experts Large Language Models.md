## MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出两种后训练（post-training）专家级稀疏化技术——(1) **Expert Pruning（专家剪枝）**：逐层使用小规模校准数据集（C4/MATH），枚举所有保留 r 个专家的组合，以最小化 Frobenius 范数量化重构损失 ‖F'(x, C) − F(x)‖_F 为目标，保留最优专家子集，永久丢弃不重要专家，减少模型参数量和部署内存。(2) **Dynamic Expert Skipping（动态专家跳过）**：在推理时，根据路由权重比值 w_{e1}/w_{e0} 与逐层阈值 β（校准集上该比值的中位数）的比较，动态决定是否跳过次优专家，减少每个 token 激活的专家数，提升推理速度。两者可组合使用。
  - 实验比较：
    - Baseline 1: Wanda（2:4 结构化稀疏，约 50% 参数减少）
    - Baseline 2: Random Expert Pruning（随机丢弃专家）
    - Baseline 3: Frequency-based Expert Pruning（基于激活频率丢弃专家）
    - 比较 r=6（保留6专家/丢弃2专家）和 r=4（保留4专家/丢弃4专家，vs 原始8专家）
    - 评估指标：8 项 EleutherAI LM Harness 零样本任务平均准确率、GSM8K 5-shot 准确率、MATH 零样本准确率、峰值 GPU 内存使用量（MB）、token 生成加速比

- 硬件平台是什么，配置是什么。
  - 推理部署：NVIDIA A100-80G GPU。原始 Mixtral 8x7B (bf16) 需 2 张 A100-80G；剪枝 r=6 或 r=4 后仅需 1 张 A100-80G
  - 微调：16 张 A100-80G GPU（MetaMathQA 微调，900 步，lr=2e-5，cosine scheduler）
  - 推理速度测试基于修改版 AutoGPTQ 脚本

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral 8x7B、Mixtral 8x7B Instruct（MoE 架构，每层 8 个专家，top-2 路由，总参数量 47B，专家占 45B 即 ~96%）
  - 校准数据集：C4（通用任务剪枝，128 条序列 × 2048 tokens）；MATH 训练集（领域特定任务剪枝，128 条序列 × 2048 tokens）
  - Benchmark（通用）：EleutherAI LM Harness 8 项零样本任务——ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OBQA, RTE, WinoGrande
  - Benchmark（领域特定）：GSM8K（5-shot）、MATH（零样本）
  - 微调数据集：MetaMathQA

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/Lucky-Lance/Expert_Sparsity
  - 依赖框架：HuggingFace Transformers
  - 算法 pipeline（Expert Pruning — 逐层枚举剪枝）：
    ```
    # Step 1: 校准数据前向传播，缓存每层输入-输出对
    for layer l in 1..L:
        for batch in calibration_data:
            x[l], F_l(x[l]) = forward_and_cache(layer_l, x)
    
    # Step 2: 逐层枚举剪枝
    for layer l in 1..L:
        best_loss = inf
        for C in combinations({expert_0,...,expert_{n-1}}, r):
            F'(x,C) = Σ_{j:0→r-1} w̃_{e_j} · E_{e_j}(x)
            其中 w̃_{e_j} = w_{e_j} / Σ_{m=0}^{r-1} w_{e_m}
            loss = ‖F'(x[l],C) − F_l(x[l])‖_F
            if loss < best_loss: best_experts[l] = C
    
    # Step 3: 修改 config，加载保留的专家
    # 复杂度: C(n,r) 枚举 × L 层; Mixtral 8x7B r=6 约 30min
    ```
  - 算法 pipeline（Dynamic Expert Skipping — 在线跳过）：
    ```
    # 校准阶段：逐层计算跳过阈值 β
    for layer l in 1..L:
        ratios = []  # 收集路由权重比
        for each token x in calibration_data:
            w[e0], w[e1] = top2 routing weights
            ratios.append(w[e1] / w[e0])
        β[l] = median(ratios)  # 中位数 → 跳过概率约 50%
    
    # 推理阶段：动态跳过
    for each token x in generation:
        e0, e1 = top2 routing indices
        if w[e1] < β[l] * w[e0]:
            y = E_{e0}(x)  # 仅用 top-1 专家
        else:
            y = w̃[e0]·E_{e0}(x) + w̃[e1]·E_{e1}(x)  # 用 top-2
    ```
  - 关键结果：r=6（24% 参数减少）→ 1.19-1.20× 加速，平均性能下降 ~2.9 点；r=4（48% 参数减少）→ 1.27× 加速，平均性能下降 ~7.1 点；组合剪枝+动态跳过 r=4 → 1.33× 加速。r=6 时单张 80G GPU 可部署 bf16 Mixtral 8x7B。领域特定校准（MATH 替代 C4）可大幅提升数学任务剪枝效果（GSM8K 5-shot r=6: 41.02→51.25）。微调后 r=7 的剪枝模型可超越原始 8-expert 模型。
