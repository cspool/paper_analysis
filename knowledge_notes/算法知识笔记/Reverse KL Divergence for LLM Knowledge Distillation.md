## Reverse KL Divergence for LLM Knowledge Distillation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reverse KL Divergence定义为 $D_{KL}(p_{student} \parallel p_{teacher}) = \sum_v p_{student}(v) \cdot \log\frac{p_{student}(v)}{p_{teacher}(v)}$。在LLM知识蒸馏中，它与forward KL $D_{KL}(p_{teacher} \parallel p_{student})$形成对比：forward KL是"mean-seeking"——student试图覆盖teacher所有模式（包括低概率token），可能导致student分散概率质量到低概率区域产生hallucination；reverse KL是"mode-seeking"——student聚焦teacher的高概率模式，允许忽略teacher的低概率尾部。M1选择reverse KL进行跨架构蒸馏，因为Mamba student的表达能力有限（相比Transformer teacher），mode-seeking特性使student集中学习teacher的主要推理模式，而非浪费容量覆盖所有低概率token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Forward KL (mean-seeking, standard in many distillation works):
# D_KL(teacher || student) = Σ_v p_t(v) * log(p_t(v) / p_s(v))
#   = -p_t(v) * log(p_s(v)) + const  → standard cross-entropy with soft targets
loss_fwd = -(p_teacher.detach() * log_softmax(student_logits)).sum(dim=-1)

# Reverse KL (mode-seeking, M1's choice):
# D_KL(student || teacher) = Σ_v p_s(v) * log(p_s(v) / p_t(v))
log_p_student = log_softmax(student_logits)
log_p_teacher = log_softmax(teacher_logits)
p_student = softmax(student_logits)
loss_rev = (p_student * (log_p_student - log_p_teacher)).sum(dim=-1)

# Mode-seeking behavior示例 (vocab size=3):
# Teacher: P=[0.7, 0.25, 0.05]
# Forward KL optimal: student ≈ [0.7, 0.25, 0.05] (完全匹配所有模式)
# Reverse KL optimal: student ≈ [0.95, 0.05, 0.0] (聚焦主要模式, 忽略尾部)
# → student更"自信"，输出分布更尖锐
```
M1在蒸馏阶段对每个token位置独立计算token-level reverse KL divergence，仅计算assistant output token的loss（mask user prompt部分）。结合data packing合并多序列至max_len=8192以加速训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在PyTorch中实现为 `F.kl_div(log_p_student, log_p_teacher, reduction='none', log_target=True)` 配合student概率作为权重。M1使用AdamW optimizer, LR=1e-5, β=(0.9,0.95), weight decay=0.1, cosine decay schedule。Reverse KL最适用于：(1) student容量显著小于teacher；(2) student与teacher架构不同（如Transformer→Mamba）；(3) teacher输出分布较flat（有大量低概率token）。在这些场景下reverse KL的mode-seeking特性产生更sharp、更集中的student输出分布。不适用场景：需要student保持与teacher完全相同输出多样性时。开源：https://github.com/jxiw/M1。

涉及论文标题：
- M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

---
