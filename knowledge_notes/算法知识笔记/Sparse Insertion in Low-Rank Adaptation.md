## Sparse Insertion in Low-Rank Adaptation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparse Insertion in Low-Rank Adaptation 是 LoRA 变体中通过选择性插入低秩矩阵来减少可训练参数的技术。核心动机：研究（如 SoRA, Ding et al. 2023）发现并非所有层的低秩适配对下游任务等量贡献——稠密全层插入导致参数浪费。稀疏插入策略通过仅在部分关键位置上激活 LoRA adapter，在维持性能前提下大幅减少参数。SSMLoRA 实现的"交替间隔稀疏插入"具体设计：(1) 仅插入 attention 的 query 和 value 矩阵，key 始终不插；(2) 相邻 encoder 层交替激活——layer l 激活 Q Time Module、layer l+1 激活 V Time Module、layer l+2 激活 Q...；(3) Q 和 V 各自维护独立时间轴（避免跨类型状态干扰）；(4) 非 attention 层（FFN/classifier）使用标准稠密 LoRA（无 SSM 状态）。参数压缩效果：RoBERTa-base 上 1.0M（vs LoRA 1.3M，~77%），LLaMA2-7B 上 15.8M（vs LoRA 20.0M，~79%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# SSMLoRA 中的交替间隔稀疏插入策略
for layer_idx in range(num_layers):
    if layer_idx % 2 == 0:   # 偶数层：激活 query Time Module
        q_output = time_module_q(x, h_q_state)   # SSM 状态增强
        v_output = W_0_v(x)                      # value 跳过（原始 forward）
    else:                     # 奇数层：激活 value Time Module
        q_output = W_0_q(x)                      # query 跳过
        v_output = time_module_v(x, h_v_state)   # SSM 状态增强
    k_output = W_0_k(x)                          # key 始终直接使用原始权重

# 同类矩阵沿独立时间轴累积状态信息
# h_q 只在偶数层的 Q Time Module 间传递
# h_v 只在奇数层的 V Time Module 间传递
```
与 SoRA 的动态 pruning 对比：SoRA 通过重要性评分自适应决定每个 rank 方向的参与度；SSMLoRA 使用固定结构规则（不引入额外计算开销判断重要性），依赖 SSM 状态传递补偿稀疏带来的信息损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两种实现范式：(1) 动态 pruning 类——如 SoRA，在训练中自适应学习 importance score 并逐步剪除低重要性 adapter；(2) 结构化稀疏调度类——如 SSMLoRA 的固定交替规则，实现极简无需额外开销。选择依据：结构化方法适合需要确定性参数预算的场景；动态方法适合需要自适应分配不同层参数容量的场景。SSMLoRA GLUE/SuperGLUE 消融实验验证 rank r=1-16 范围内，稀疏 ~50% 参数的 SSMLoRA 在多数任务上匹配或超越稠密 LoRA。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---
