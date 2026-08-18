## 贪心调色板格式搜索（Greedy Palette Format Search，自动化权重量化格式搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 贪心调色板格式搜索是 UNICORE 的离线权重量化方法：为每个权重张量构造一个大小为 k（如 16）的紧凑 DynFP 格式调色板（palette），再让每个 group 从调色板中选择局部误差最小的格式。动机：单张量各 group 可能需要多种 DynFP 配置，搜索空间巨大（多个 E/M 布局 × gap-insertion 变体 × 众多 Z 候选），穷举或人工探索不可行；全局每 group 自由选 96 个候选会导致元数据与搜索时间过大。该方法数据驱动、无需激活校准、不引入分布偏置，作为一次性离线步骤（Llama-2-7B 单张 RTX 6000 Ada 约 2 分钟），推理时只用存储的格式索引与 scale。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 算法（DynFP-4 例，三阶段）：
    ```
    # 阶段1 候选池生成：枚举所有可行 DynFP-4 配置 → 96 个候选
    #   = 4 个基础 E/M 布局 (E3M0, E2M1, E1M2, E1M2I) × 24 个 Z 赋值
    #   Z 限制在内部 E3M2 正常域内 (Z >= 0.5) 避免 reintroduce subnormal
    # 阶段2 迭代贪心构造 palette P（容量 k）:
    P = {}
    for t in 1..k:
        if t == 1: f_t = argmin_f global_MSE(f)          # 初始化选全局 MSE 最小的格式
        else:      f_t = argmax_f marginal_MSE_reduction(P ∪ {f_t})  # 每轮选使全局 MSE 边际下降最大的候选
        P = P ∪ {f_t}
    # 阶段3 最终分配：每个 group 从 P 中选局部量化误差最小的格式
    for group g: idx_g = argmin_{f in P} local_MSE(g, f)
        # 存 4-bit 格式索引（16-entry palette）+ 8-bit scale；Z 加载进 Unified Format Converter
    ```
  - 行为类似在表示空间对权重聚类；量化 Llama-2-7B 约 2 分钟/checkpoint，推理期无运行时格式搜索。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：软件侧（PyTorch，artifact Software/Accuracy/ 的 quant_utils）实现候选池枚举、全局 MSE 贪心迭代、逐 group 分配；Z 值在计算新权重张量时加载进硬件 Unified Format Converter；元数据随权重存主存。使用：作为 UNICORE-Q（启用分布自适应 DynFP 量化）的权重量化路径，与在线 crest factor K/V 选择互补；评估显示 UNICORE-Q 在 4/4/16 各模型 PPL 最低、zero-shot 平均准确率多数配置最优（DynFP 增益明显）。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
