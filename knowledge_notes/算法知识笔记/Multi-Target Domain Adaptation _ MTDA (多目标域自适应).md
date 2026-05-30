## Multi-Target Domain Adaptation / MTDA (多目标域自适应)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Target Domain Adaptation（MTDA，多目标域自适应）是域自适应（Domain Adaptation）的一个子问题：给定源域上预训练的模型，将其同时适配到多个不同的目标域（而非单一目标域）。与 Single-Target Domain Adaptation (STDA) 对应。传统 MTDA 方法主要包括：(1) 知识蒸馏方法（MTDA-KD, Nguyen-Meidine et al. 2021）——使用多个学生模型适应不同目标域，计算开销高；(2) 信息论方法（Gholami et al. 2020）；(3) 训练免方法（Li et al. ECCV 2024）——通过模型合并实现 MTDA，无需额外训练数据或蒸馏过程。训练免 MTDA 利用了从同一源模型微调的多个域自适应模型位于同一优化盆地的事实，通过简单权重平均合并。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 HDRQ 论文中 Office-Home 数据集的 MTDA 实验为例（源域 Real，目标域 Art/Clipart/Product）：
```
# 步骤 1: 源自由域自适应 (SFDA) - 各自独立
θ₀ = ResNet50_pretrained(ImageNet)                  # 源预训练
θ_R→A = SHOT_adapt(θ₀, target=Art)                  # Real→Art
θ_R→C = SHOT_adapt(θ₀, target=Clipart)              # Real→Clipart
θ_R→P = SHOT_adapt(θ₀, target=Product)              # Real→Product

# 步骤 2: HDRQ 量化 - 各自独立
θ_R→A_q = HDRQ(θ_R→A, bit=4)
θ_R→C_q = HDRQ(θ_R→C, bit=4)
θ_R→P_q = HDRQ(θ_R→P, bit=4)

# 步骤 3: 模型合并
θ_merged = (θ_R→A_q + θ_R→C_q + θ_R→P_q) / 3       # 三域平均

# 步骤 4: 多目标域评估
acc_A = eval(θ_merged, Art)                          # 20.28%
acc_C = eval(θ_merged, Clipart)                      # 45.75%
acc_P = eval(θ_merged, Product)                      # 73.69%
harmonic_mean = 3 / (1/acc_A + 1/acc_C + 1/acc_P)   # 主指标
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MTDA 的实现依赖于：(1) 选择单目标域自适应方法（如 SHOT, HRDA, DANN）；(2) 对各目标域分别执行适配（独立完成，可并行）；(3) 通过模型合并技术融合为统一模型。评估指标为各目标域的 Harmonic Mean 而非算术平均，因为 Harmonic Mean 对最差域更敏感，鼓励在所有域上的均衡性能。HDRQ 的关键贡献是在步骤 (2) 的量化中保持步骤 (3) 的合并兼容性，使 quantized MTDA 首次达到实用水平。HDRQ 不假设对目标域有任何先验知识，仅要求量化阶段可以访问源模型权重（通常成立，因为源模型先于部署部署在中心服务器上）。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
