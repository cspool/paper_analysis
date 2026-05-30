## Slow-Fast Learning Loop (慢-快学习循环)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Slow-Fast Learning Loop 是 SlowFast-VGen 提出的双速学习循环算法，将推理时快速学习（TEMP-LORA）嵌入到慢学习训练过程中，用于需要从多 episode 经验中学习的长时规划任务。其生物学动机来源于认知科学中的互补学习系统（Complementary Learning Systems）：海马体（hippocampus）支持快速编码新经验形成情节记忆，新皮层（neocortex）逐步将记忆抽象整合为通用知识。双循环结构：内层（fast learning loop）在每个 episode 上运行 TEMP-LORA 快速适配并积累数据（input, ground-truth output, TEMP-LORA 参数 Θ）；外层（slow learning loop）固定 TEMP-LORA 参数，利用多 episode 积累的数据更新核心模型权重 Φ，实现从单 episode 记忆到跨 episode 技能泛化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Slow-Fast Learning Loop Algorithm
# Φ: 任务特定慢学习权重, D: 任务特定数据集, β: 慢学习率

while not converged:
    D_s = ∅                       # 准备慢学习数据集
    for each (x, episode) in D:
        # 内层: Fast Learning Loop
        初始化 TEMP-LORA 参数 Θ_0^e
        将 episode 分割为 I 个短序列: {X_i^e}_{i=0}^{I-1}
        for i in 0..I-1:
            # 收集数据点: (输入, ground-truth 输出, 当前 TEMP-LORA 参数)
            D_s = D_s ∪ {X_i^e, X_{i+1}^e, Θ_i^e}
            
            # Fast learning: 固定 Φ, 更新 Θ_i^e
            Y_i = (Φ + Θ_i^e)(X_i^e)                    # 生成
            X_i' = X_i^e ⊕ Y_i                          # 拼接输入输出
            z_t = sqrt(ᾱ_t)·X_i' + sqrt(1-ᾱ_t)·ε       # 加噪
            loss_Θ = ||ε - ε_{Φ+Θ_i^e}(z_t, t)||²      # 去噪 loss
            Θ_{i+1}^e = Θ_i^e - α·∇_Θ loss_Θ            # 更新 TEMP-LORA

    # 外层: Slow Learning Loop
    for {X_i^e, X_{i+1}^e, Θ_i^e} in D_s:
        Φ_i^e = Φ + Θ_i^e                                # 组合慢+快学习权重
        Y_pred = Φ_i^e(X_i^e)                            # 预测输出
        loss_Φ = ||Y_pred - X_{i+1}^e||²                 # 与 ground-truth 比较
        
        # Slow learning: 固定 Θ_i^e, 更新 Φ
        Φ = Φ - β·∇_Φ loss_Φ
```

Annotations:
- Θ_i^e: episode e 中第 i 步的 TEMP-LORA 参数，存储截至该步的情节记忆
- D_s: 收集自所有 episodes 的 (input, output, Θ) 三元组
- β: 慢学习学习率，在 slow-fast loop 中用于更新 Φ
- 外层固定 Θ_i^e 仅更新 Φ：确保逐步巩固泛化知识而非覆盖特定记忆
- 该循环对完整预训练开销大，适用于特定领域/任务的 fine-tuning

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SlowFast-VGen 在两个长时规划任务上验证了 slow-fast learning loop 的效果：(1) RLBench 机器人操作——移动物体后归位，测量到先前位置的距离；(2) Minecraft 游戏导航——沿路径返回起点，测量到预定义路径点的最近距离。实验（Table 2）显示完整的 slow-fast learning loop 优于"无 loop"消融变体：RLBench Dist 0.013 vs 0.055，Minecraft Dist 1.51 vs 2.23。Loop 的关键价值在于：仅 TEMP-LORA（无 loop）能存储单 episode 记忆但无法跨 episode 泛化技能；加入 slow loop 后模型能从多次"移动-归位"或"导航-返回"经验中学习通用策略。论文指出该循环对完整预训练（200k 数据）开销过大，建议用于特定领域 fine-tuning。

涉及论文标题：
- SlowFast-VGen: Slow-Fast Learning for Action-Driven Long Video Generation
