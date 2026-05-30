## LoRA-augmented Recurrence

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Finch 用 LoRA 生成 data-dependent 递归参数偏移：lora(x)=λ+tanh(xA)B，A∈R^{D×32},B∈R^{32×D}。不同于传统 W'=W+BA fine-tuning，此处 LoRA 使 token-shift μ_□ 和 decay ω 被 data-dependent offset 动态增强。rank-32 低秩矩阵仅 64D 参数（vs D²），约 0.5% per block 参数增量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
lora_r(x) = λ_r + tanh(x@A_r)@B_r   # rank=32, 65D params
ddlerp_r(x_t,x_{t-1}) = x_t + (x_{t-1}-x_t)⊙lora_r(x_t+(x_{t-1}-x_t)⊙μ_x)
# decay LoRA 加倍: A_ω∈R^{D×64}, B_ω∈R^{64×D} (rank=64)
```
初始化 A,B~U(-1e-4,1e-4)，初始 data-dependent 项≈0，从 Eagle 行为逐步学习。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
每 Time Mixing 层 5 组 LoRA（r/k/v/g rank-32, ω rank-64）。消融: 完整 LoRA loss=2.91 < 仅 decay 2.923 < 无 LoRA 2.926。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression
- VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

---
