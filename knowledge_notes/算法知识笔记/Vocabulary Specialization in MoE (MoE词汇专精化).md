## Vocabulary Specialization in MoE (MoE词汇专精化)

术语解释
Vocabulary Specialization (VS) 是 Muennighoff et al. (2024) 提出的指标：衡量词汇表中每个 token 被路由到特定 expert 的频率集中度。VS 越高 = 词汇 token 的路由越集中 = expert 的语言学专业化程度越高。本文将其扩展到 CPT 场景，称为 Continual Vocabulary Specialization (CVS)。

术语是什么：
对于 MoE layer j，expert E_i，token x：
$$\text{VS}(j, E_i, x) = \frac{N_{j,x,E_i}^{(k)}}{N_{j,x}}$$

其中 N_{j,x,E_i}^{(k)} 是 token x 在 checkpoint j 下被路由到 expert E_i 的次数，N_{j,x} 是 token x 的总出现次数。

为比较跨 checkpoint 的 specialization 变化：固定 pre-training checkpoint 的 expert-token 映射（将每个 token 分配给最常处理它的 expert），然后在新 checkpoint 上用同一映射计算 VS。如果新 checkpoint 的 VS 显著低于 pre-training checkpoint，说明路由模式发生了变化。

从算法pipeline角度拆解术语：
```python
def continual_vocab_specialization(layer, checkpoint_h, checkpoint_j, test_tokens):
    """CVS: fix mapping from ckpt_h, compute VS using ckpt_j"""
    # Step 1: Create one-to-many mapping from checkpoint_h
    token_to_expert_map = {}
    for token_id in range(vocab_size):
        expert_counts = count_expert_assignments(token_id, checkpoint_h)
        token_to_expert_map[token_id] = argmax(expert_counts)

    # Step 2: Compute CVS using checkpoint_j's routing
    total_vs = 0
    for token in test_tokens:
        assigned_expert = token_to_expert_map[token.id]
        vs = prob_routed_to(token, assigned_expert, checkpoint_j)
        total_vs += vs
    return total_vs / len(test_tokens)
```

术语一般如何实现？如何使用？
- **CPT 分析**：CVS 在 early layers (0-4) 显著降低（路由模式变化大），layers 5-23 几乎不变。0% replay 的 early layers VS 最低 → 与 FineWeb 遗忘相关
- **分布依赖**：同一 token 的 VS 在不同分布上不同（例如 "for" 在 English 和 Code 中的上下文表示不同，路由也不同）
- **架构差异**：Granular MoE 和 Switch MoE 的 VS 模式相似，表明 VS 主要受 CPT 策略（replay）而非架构影响

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
