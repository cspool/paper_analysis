## Relative Token Difficulty

术语解释
Relative Token Difficulty 是 Duo-LLM 提出的概念，定义为 token 从额外计算资源中获益的潜力——通过比较 small model loss 与 oracle/large model loss 的差距来衡量，而非仅看 token 的绝对 loss 值。

术语是什么？
传统观点认为高 loss token 是"困难"token，应路由到大模型处理。Duo-LLM 发现这不足够——某些 token 的 loss 虽高，但切换到 large model 或 oracle 后 loss 几乎不降（如"relationship"在"This can be a"之后，上下文可预测性差导致高 loss 但无法通过更多计算改善）。真正值得更多计算的 token 是那些 small model 与 oracle 之间存在显著 loss gap 的 token。

从算法pipeline角度拆解术语：
```
# Relative Token Difficulty: Loss Gap Method

# For each token in holdout set:
def compute_relative_difficulty(token, context):
    # 1. Small model loss
    loss_small = forward_loss(token, context, all_small=True)
    
    # 2. Big model loss (or oracle)
    loss_big = forward_loss(token, context, all_big=True)
    # or: loss_oracle = oracle_optimal_loss(token, context, budget)
    
    # 3. Absolute difficulty (traditional)
    abs_difficulty = loss_small  # — Higher = harder
    
    # 4. Relative difficulty (Duo-LLM's proposal)
    rel_difficulty = loss_small - loss_big  # Loss gap
    
    return abs_difficulty, rel_difficulty

# 代码示例 (Python):
# "names = [...]" 后紧跟 "iterator = filter(...)"
# loss_small("iterator") = 高 (~5.2)  ← 传统认为是困难 token
# loss_big("iterator")  = 高 (~5.0)  ← gap 仅 0.2, 不值得更多计算
# → 该 token 上下文不确定(新行开头), 计算无法帮助
# 
# "filter(is_big_name, names)" 中的 "len":
# loss_small("len") = 中 (~3.8)
# loss_oracle("len") = 低 (~2.1)  ← gap 1.7, 值得更多计算
# → 上下文有助于推断 "len", 额外计算显著降低 loss
```

术语一般如何实现？如何使用？
- 论文在 C4 validation set 和 Python code holdout set 上计算了所有 token 的 relative difficulty
- 可视化：蓝色=低 difficulty，红色=高 difficulty，发现代码中行首 token 的 small loss 高但 gap 小（不值得额外计算），而 len, filter, None 等关键字 gap 大（值得额外计算）
- 应用：可作为 router 训练的辅助信号——优先将 big 模块分配给 relative difficulty 高的 token，而非仅凭绝对 loss 值
- 该概念仍在早期研究阶段，论文建议进一步探索作为 router 训练的 surrogate metric

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---
