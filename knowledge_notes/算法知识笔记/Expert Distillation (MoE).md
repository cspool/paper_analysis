## Expert Distillation (MoE)

术语解释
MoE知识蒸馏是将大型MoE教师模型的知识迁移到更小的学生模型（可以是更小的MoE或稠密模型）的技术，以在保持性能的同时减少模型大小和推理成本。

术语是什么？
MoE蒸馏的两种主要范式：
1. **MoE→MoE蒸馏**：保持MoE结构但减少专家数/参数
   - DeepSpeed-MoE：使用阶段式知识蒸馏创建PR-MoE的蒸馏版本MoS
   - LLaVA-MoD：结合MoE结构和两阶段蒸馏（mimic distillation + preference distillation）训练小多模态模型
2. **MoE→Dense蒸馏（Sparse to Dense）**：将稀疏MoE转换为稠密模型
   - OneS：两阶段——Knowledge Gathering（求和/平均/top-k/SVD四种聚合方法合并专家）+ Knowledge Distillation
   - Switch Transformers：压缩97%参数后，稠密模型仍保留30%+性能
   - ELSM：稠密学生模型可匹配甚至超越稀疏教师性能
   - MoE-KD：用最频繁使用的expert初始化学生FFN，然后逐层蒸馏

从算法pipeline角度拆解术语。
```
# MoE → Dense Distillation (OneS)
# Step 1: Knowledge Gathering - 合并所有expert
W_merged = aggregate({E_1, E_2, ..., E_N})
# 聚合方法选择：sum, average, top-k, SVD

# Step 2: Knowledge Distillation
for each training sample x:
    y_teacher = MoE_teacher(x)   # 原MoE模型输出
    y_student = Dense_student(x) # 合并后的稠密模型
    
    # KL散度损失
    L_KD = KL(softmax(y_teacher/T), softmax(y_student/T))
    # 也可结合task loss
    L_total = α * L_KD + (1-α) * L_task
```

术语一般如何实现？如何使用？
- 温度系数T控制softmax平滑度，典型值T=2~10
- mimic distillation阶段通常只匹配输出分布，preference distillation进一步优化
- 蒸馏数据集可以是通用语料或任务特定数据
- 可将MoE模型部署到资源受限环境（移动端、边缘设备）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

**Task-MoE vs Distillation 对比（Kudugunta et al., EMNLP 2021）**：
蒸馏 token-MoE (533M) → dense Transformer-Base (142M) 仅保留 **32.25%** BLEU 增益（与 Fedus et al., 2021 Switch Transformer 结果一致：蒸馏 large sparse model 仅保留 small fraction 质量增益）。Task-MoE sub-network extraction 保留 **100%** BLEU 增益（decoder 仅 25M params vs distilled 142M），BLEU 29.0 vs 26.9 (+2.1)。核心原因：蒸馏过程中引入 undesirable artifacts (Freitag et al., 2019)，而 sub-network extraction 直接使用原始 MoE expert 权重。

---
