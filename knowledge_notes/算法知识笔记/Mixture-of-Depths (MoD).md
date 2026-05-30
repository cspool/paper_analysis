## Mixture-of-Depths (MoD)

术语解释
由 Raposo et al. (2024) 提出，将 MoE 条件计算原理应用于 Transformer 深度维度：通过二值门控决定每个 token 是否被某层处理，动态分配 FLOPs 到不同序列位置。

术语是什么？
不同于标准 MoE 在宽度维度的条件计算（选择专家），MoD 在深度维度进行条件计算（选择层）。每层有一个二值 router，若输出为 0 → token 走 skip connection 直通下一层。

```
def mod_layer(x, router, layer_fn):
    selected = router(x) > 0       # [seq] bool
    y = x
    y[selected] = layer_fn(x[selected])
    return y  # 未选中的 tokens 直接跳过
```

术语一般如何实现？如何使用？
- 可与标准 MoE 组合（MoD + MoE FFN）
- FLOPs footprint 低于 vanilla Transformer 或 pure MoE
- 在固定 FLOPs budget 下提升性能

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models

---
