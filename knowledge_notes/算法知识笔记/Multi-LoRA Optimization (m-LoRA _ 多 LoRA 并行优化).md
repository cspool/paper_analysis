## Multi-LoRA Optimization (m-LoRA / 多 LoRA 并行优化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-LoRA Optimization 是在单 GPU 上同时训练/推理多个 LoRA 变体模型的高吞吐技术，由 m-LoRA / Aspen (Ye et al. 2023) 提出。核心原理：多个 LoRA 变体共享同一 frozen backbone，不同模型的 batch 合并避免重复 backbone 计算。MixLoRA 扩展至 multi-MixLoRA 场景：多模型的 multi-task 输入合并为单 batch，共享 W1/W3 FFN 计算，各模型 router 独立路由执行各自 LoRA adapter。per-model GPU memory 降 ~45%（LLaMA-2 7B: 15.1→8.8GB 训练, 13.7→7.2GB 推理）。

从算法pipeline角度拆解术语（Algorithm 1, Appendix A.7）：
```
for t in {1..M}:                               // M 个 MixLoRA 模型
    T_t = T^{l-1}[t]                            // 模型 t 的输入 [B,N,D]
    r_t' = Top2(Softmax(Linear_t(T_t)))         // 模型 t 独立 router
    h_W1, h_W3 = Shared_W1(T_t), Shared_W3(T_t) // 共享 FFN 计算
    for k in {1..K}:
        h_gate = SiLU(h_W1 + LoRA_k^{W1}) ⊙ (h_W3 + LoRA_k^{W3})
        h_k = Shared_W2(h_gate) + LoRA_k^{W2}(h_gate)
        T_t^l += h_k ⊙ r_t'[:,k]                // router 加权累加
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/TUDB-Labs/m-LoRA
- 适用：多下游任务同时微调、multi-tenant LoRA serving。限制：所有模型需共享相同 backbone。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
