## Router Saturation (路由饱和率)

术语解释
Router Saturation (RS) 是 Muennighoff et al. (2024) 在 OLMoE 中提出的 MoE 路由稳定性指标：衡量两个训练 checkpoint 之间路由决策的一致性。RS 越高 = 两个 checkpoint 的路由决策越相似 = 路由变化越小。本文将其扩展到持续预训练场景，称为 Continual Router Saturation (CRS)。

术语是什么？
对于两个 checkpoint（分别对应 task h 和 task j 的最终状态），在 N 个 tokens 上的 Router Saturation 定义为：
$$\text{CRS}(h, j) = \frac{1}{N} \sum_{i=1}^{N} \frac{|E_i^{(h)} \cap E_i^{(j)}|}{k}$$

其中 E_i^{(h)} 是 token i 在 checkpoint h 中被选中的 k 个 experts 的集合。RS ∈ [0, 1]，值越低 = 路由决策变化越大。

从算法pipeline角度拆解术语：
```python
def continual_router_saturation(checkpoint_h, checkpoint_j, test_tokens):
    """Compute RS between two checkpoints on test tokens"""
    matches = 0
    for token in test_tokens:
        experts_h = set(route(token, checkpoint_h))  # top-k experts @ ckpt h
        experts_j = set(route(token, checkpoint_j))  # top-k experts @ ckpt j
        matches += len(experts_h & experts_j)
    return matches / (len(test_tokens) * k)
```

术语一般如何实现？如何使用？
- **CPT 分析用途**：RS 帮助识别哪些 MoE layers 的路由在 CPT 中变化最大。本文发现 layers 0-2 和 layers 13-23 的 RS 最低（路由变化最大），且 0% replay 的 checkpoint 在所有层都低于 40% replay
- **与遗忘的关系**：仅有 0% replay checkpoint 出现严重 FineWeb 遗忘，其 RS 在所有层都显著低于 40% replay counterpart，特别是早期层 - 说明早期层的剧烈路由变化与遗忘相关
- **Layer-wise 趋势**：RS 在 layers 2-13 最高（路由最稳定），然后在 layer 13+ 逐渐下降

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
