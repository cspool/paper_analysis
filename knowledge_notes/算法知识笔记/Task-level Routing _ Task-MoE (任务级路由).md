## Task-level Routing / Task-MoE (任务级路由)

术语解释
Task-level Routing 是 MoE 路由的一种粒度策略，根据 task identity（如 multilingual NMT 中的语言对或目标语言）而非输入 token 内容来决定 expert 选择，使同一 task 的所有 token 路由到相同的 expert 子集。由 Kudugunta et al. (EMNLP Findings 2021) 在"Beyond Distillation"中提出。

术语是什么？
Task-level routing 将 MoE 的 gating function 从 GATE(x_s)（per-token）改为 GATE(task_id_s)（per-task）。在 MNMT 中，task_id 可以是 target language（French→English 和 German→English 共享 "English" experts）或 language pair（各自独立）。公式：

$$\mathcal{G}_{s,E} = \mathrm{GATE}(\mathrm{task\_id}_s)$$

这与 token-level routing GATE(x_s) 和 sentence-level routing GATE(mean(x_{1:S})) 形成三种路由粒度。核心价值在于**推理效率**：task-level routing 使每个 task 仅需加载 K 个 experts（K=2 for top-2），而非全部 E 个 experts，从而避免了 token-level MoE 的模型并行和 all-to-all 通信开销。

从算法pipeline角度拆解术语。
```
# Task-level MoE Forward (MNMT, decoder)
def task_moe_decoder_forward(x_s, task_id):
    # task_emb 是可学习的 task embedding table
    task_emb = task_embedding_table[task_id]  # e.g., "French"
    logits = router(task_emb)                 # GATE(task_emb), NOT GATE(x_s)
    G = TopK(Softmax(logits), k=2)            # 所有 token 共享相同 G
    y_s = sum(G[e] * FFN_e(x_s) for e in top_k_indices)
    return y_s

# Hybrid strategy (best performer in paper):
# Encoder: token-level routing (flexibility for source language processing)
# Decoder: task-level routing (decoder dominates inference cost, 200x per step)
```

推理时子网络提取：server 仅预加载 task-specific 的 K 个 experts（如 expert 5 + expert 17 for French），不同 task 可在不同设备上独立并行解码。WMT: decoder params 221M→25M (↓88%); large-scale: 6.5B→201M (↓97%)。

术语一般如何实现？如何使用？
- 仅适用于 task boundary 明确的 multi-task 场景（如 multilingual NMT），不适用于通用单任务 LLM
- Hybrid 策略（Token encoder + Task decoder）效果最佳：encoder 保持 per-token 灵活性处理多语言源输入，decoder 用 task-level 路由降低推理成本
- Task boundary 选择：target language（同一目标语言的所有源语言共享 experts，最大化 transfer）vs language pair（各自独立，最大化 specialization）
- 实现基于 GShard 框架（TensorFlow/Lingvo），在 router 输入侧用 task_embedding 替代 token_embedding

涉及论文标题：
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

---
