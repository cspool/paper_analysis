## Knowledge Distillation for MoE Inference (MoE推理中的知识蒸馏 / Mixture-of-Students)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Knowledge Distillation for MoE Inference 是将知识蒸馏技术应用于 MoE 模型推理阶段，以在保持推理精度的前提下显著减少模型大小。MoE 模型虽然在训练时通过稀疏激活降低了计算量，但 inference 时需要存储所有 expert 参数（包括未被激活的 expert），参数量远超同规模的 dense 模型，导致显存压力和部署成本高。MoESys 在 Graph Optimization Pipeline 的 Distillation & Compression 步骤中，将 teacher MoE（含大量 expert）蒸馏为 student MoE（含少量 expert），通过 DeepSpeed 提出的 Mixture-of-Students (MoS) 架构提升 student 模型的精度。蒸馏后，student 模型在推理时仅需激活更少的 expert。

从算法pipeline角度拆解术语：
MoESys 中 MoE distillation 的 pipeline 流程：
```
# Teacher: MoE model with E_t experts
# Student: MoE model with E_s experts (E_s << E_t)

# Step 1: Teacher inference on training data
for batch in distillation_dataset:
    teacher_logits = teacher_moe(batch)  # E_t experts, top-K gating

# Step 2: Student training with distillation loss
for batch in distillation_dataset:
    student_logits = student_moe(batch)  # E_s experts
    teacher_logits = teacher_moe(batch)  # pre-computed or on-the-fly
    loss = CE(student_logits, labels) + λ * KL_div(student_logits, teacher_logits)
    
# Step 3: Deploy student for inference
deploy(student_moe)  # 更少的 expert, 更低的 memory 和更快推理
```

MoS (Mixture-of-Students) 的关键改进：传统 KD 用单一 student 模仿 teacher，MoS 用 multiple students（每个 student 是一个子 expert 组）联合学习，student 间通过 gating 机制分工，提升蒸馏后的模型容量和精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoESys 的蒸馏步骤作为 Graph Optimization Pipeline 的一部分离线执行，在模型部署前完成。
- 类似思路在 GLaM、DeepSpeed-MoE 中也有应用——通过训练阶段的 sparsity 和推理阶段的压缩/蒸馏，在整个 model lifecycle 中保持效率。
- KD 的 trade-off：减少 expert 数会降低模型容量和表达能力，需要通过精心设计的蒸馏策略（如 MoS、task-specific distillation）和量化/剪枝结合来弥补。
- MoESys 论文未详细给出蒸馏的实验结果对比，该方法作为 inference pipeline 中减少模型大小的一个步骤提及。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
