## Token Acceptance Rate / TokAR（Token 接受率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Acceptance Rate (TokAR) 是投机解码（包括 SSD）中的核心效率指标，定义为 draft model 生成的候选 token 中被 target model 接受的比例：TokAR = #accepted_tokens / #draft_tokens。TokAR 直接决定了投机解码的加速效果——接受率越高，每轮 draft-verify 迭代的有效产出的 token 越多，加速比越大。理论上，speedup ∝ TokAR × draft_length / (1 + overhead_ratio)。在 TwigVLM 中，TokAR 受 twig block 的训练质量影响：更好的 twig 初始化（从 K 层开始而非 L-T 层）可提升 TokAR 从更低水平到 57.4%，从而提升 RelSpd。TokAR 也受 twig 层数 T 的影响：增加 T 提升 TokAR 但也增加 draft 计算开销，T=3 时 TokAR 饱和。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Token Acceptance Rate 计算
def compute_TokAR(draft_model, target_model, prefix):
    total_draft = 0
    total_accepted = 0
    current = prefix
    while not EOS:
        # Draft阶段
        draft_tokens = draft_model.generate(current, max_len=δ)
        total_draft += len(draft_tokens)
        # Verify阶段
        accepted = target_model.verify(draft_tokens)
        total_accepted += len(accepted)
        # 追加bonus token
        bonus = target_model.generate_next(accepted)
        current += accepted + [bonus]
    return total_accepted / total_draft
```

TwigVLM 消融实验中 TokAR 影响因素：
| 变量 | 配置 | TokAR | RelSpd |
|------|------|-------|--------|
| 初始化策略 | random init | 低 | 120.4% |
| 初始化策略 | VLM layers[L-T:L] | 中 | 131.4% |
| 初始化策略 | VLM layers[K:K+T] | 57.4% | 153.6% |
| Twig 层数 T | 1 | 中 | 154.1% |
| Twig 层数 T | 3 | 饱和 | 153.6% |
| Twig 层数 T | 4 | 饱和(计算增) | 145.4% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TokAR 是衡量 draft model 与 target model 对齐程度的关键指标。更高的 TokAR 需要：(1) draft model 足够强（但也不能太强，否则计算开销大抵消加速）；(2) draft 和 target 的分布一致性高。实现中，TokAR 通过在推理时统计 accepted/total draft tokens 获得，可用于动态调整 draft length 或 early-exit 阈值。TwigVLM 的训练（twig 初始化和层数选择）以最大化 TokAR 为目标间接优化。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
