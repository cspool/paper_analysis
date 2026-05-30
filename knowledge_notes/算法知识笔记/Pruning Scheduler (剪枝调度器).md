## Pruning Scheduler (剪枝调度器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pruning Scheduler 是 AIM 中控制 LLM 各层 visual token 保留率 r^l 的分段线性函数。它决定了 visual token 在哪层开始被剪枝（l₁）、在哪层被完全移除（l₂）、以及中间层如何递减。设计依据：消融实验发现——visual tokens 在早期 LLM 层做跨模态融合时必须全保留（第 8 层剪枝导致 58.0→41.9 的性能崩溃），但在晚期层可全部移除（第 22 层剪枝几乎无影响：58.0→58.1）。

公式（AIM 公式 2）：

$$r^l = \begin{cases} 1, & \text{if } l < l_1 \\ 1 - k(l - l_1), & \text{if } l_1 \le l \le l_2 \\ 0, & \text{if } l > l_2 \end{cases}$$

其中 $k = \frac{1}{l_2 - l_1}$ 为递减斜率。l₁ 和 l₂ 是用户可调的 Scheduler 参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**AIM Scheduler 伪代码**：

```
def pruning_scheduler(l, l1, l2, L):
    """
    l: 当前层索引 (1-indexed)
    l1: 开始剪枝的层
    l2: visual token 完全移除的层
    返回: 当前层 visual token 保留率 r^l
    """
    if l < l1:
        return 1.0                        // 早期层：全保留（跨模态融合阶段）
    elif l1 <= l <= l2:
        return 1.0 - (l - l1) / (l2 - l1)  // 中期层：线性递减
    else:  # l > l2
        return 0.0                        // 晚期层：全部移除（纯文本推理阶段）

// 实际使用：
for l in 1..L:
    r_l = pruning_scheduler(l, l1=14, l2=22, L=28)
    num_keep = int(len(visual_tokens) * r_l)
    visual_tokens = prune_by_pagerank(visual_tokens, num_keep)
```

**不同 (l₁, l₂) 配置的实验结果（VideoMME）**：
| l₁ | l₂ | FLOPs (TB) | VideoMME | 说明 |
|----|----|-----------|----------|------|
| 28 | 29 | 22.90 | 58.0 | 仅 Merging，不剪枝 |
| 14 | 22 | 14.76 | 58.2 | 默认配置（最优 trade-off） |
| 7  | 22 | 12.01 | 56.8 | 更快剪枝，轻微降性能 |
| 14 | 15 | 12.10 | 54.3 | 更快完成剪枝，显著降性能 |
| 7  | 8  | 6.71  | 41.9 | 极早剪枝，性能崩溃 |

**关键发现**：l₂ 比 l₁ 更关键——只要 l₂ ≥ 22（层总数为 28），即使 l₁ 提前到 7，性能仅从 58.2 降至 56.8。但若 l₂ 提前到 15，性能锐降至 54.3。说明晚期层（>22）的 visual tokens 几乎无贡献。

术语一般如何实现？如何使用？

Scheduler 是纯算术函数，无额外计算开销。在 AIM 实现中，Scheduler 参数 (l₁, l₂) 作为超参数在推理前指定。用户可根据目标 FLOP budget 从预标定的配置表中选择合适的 (l₁, l₂) 组合。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---
