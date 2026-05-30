## Core-Expert Selection in MoE (MoE核心专家选择)

术语解释
Core-Expert Selection 是 C3PO 的优化策略之一：在 pathway 优化时只修改部分"核心专家"的 routing weights。实验发现只优化 top-20 experts 即可覆盖最终 top-8 的 99.8%，性能与全 64 expert 优化持平，但优化变量减少 68.75%。

术语是什么？
稀疏 MoE（如 OLMoE）每层 64 experts 但只激活 top-8。C3PO 策略：
1. 按预训练 router 的初始 routing weights 对 64 experts 排序
2. 只保留 top-n experts 作为可优化变量
3. 被排除的 experts 的 routing weights 保持为 0

覆盖率实验：top-8 覆盖 71.3%, top-12 提高, top-20 覆盖 99.8%

从算法pipeline角度拆解术语：
```
def select_core_experts(gate_logits, n_core=20):
    sorted_indices = argsort_descending(gate_logits)
    core_indices = sorted_indices[:n_core]
    ω_core = gate_logits[core_indices]       # 可优化
    return ω_core, core_indices
```

总体节省（Critical-Layer + Core-Expert）：16×64=1024 → 5×20=100，优化变量减少 90.2%

术语一般如何实现？如何使用？
- n_core 需在目标模型上通过 ablation 确定
- 不同 MoE 架构的 top-k 激活数不同，需要的 n_core 也不同
- 可结合 expert 激活频率统计动态确定每层的 n_core

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
