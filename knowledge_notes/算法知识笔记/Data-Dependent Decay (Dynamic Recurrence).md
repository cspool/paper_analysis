## Data-Dependent Decay (Dynamic Recurrence)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Finch 引入 w_t=exp(-exp(d_t))，d_t 由 LoRA (A_ω∈R^{D×64},B_ω∈R^{64×D}) 基于 ddlerp 后输入生成。每 channel decay 每时间步动态变化，实现选择性记忆：重要 token 降低 decay 延长保留，无关 token 加速遗忘。与 Mamba 的 selective SSM（A_t）精神相似但实现不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
d_t = lora_d(ddlerp_d(x_t,x_{t-1}))  # rank=64
w_t = exp(-exp(d_t))                  # ∈(0,1), contraction
s_t = diag(w_t)·s_{t-1} + k_t^T·v_t   # 动态衰减
```
两次 exp 确保 w_t∈(0,1)，state 不发散。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Finch 每 Time Mixing 层有 LoRA A_ω/B_ω（rank 64）。训练时梯度更新，推理固定。MQAR 长上下文能力的关键驱动。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---
