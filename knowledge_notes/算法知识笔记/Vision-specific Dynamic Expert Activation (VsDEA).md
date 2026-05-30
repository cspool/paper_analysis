## Vision-specific Dynamic Expert Activation (VsDEA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Vision-specific Dynamic Expert Activation (VsDEA) 是 LTDR 的第二个核心模块。基于 RPV 将 vision tokens 分类为 head tokens（RPV ≤ mean RPV, ~87%）和 tail tokens（RPV > mean RPV, ~13%），对 tail tokens 激活更多 experts（Top-a, a > k），采用 renormalized softmax 权重。本质是一种 data-augmentation 策略：tail tokens 数量少但信息密度高，通过让更多 experts 联合处理来降低错误路由的影响，确保 tail tokens 得到充分学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# VsDEA: Vision-specific Dynamic Expert Activation
def vsdea_expert_activation(x, logits, is_vision_tail):
    probs = Softmax(logits)  # [K]

    if is_vision_tail:
        # Tail token: activate a=4 experts (k=2 baseline)
        k_active = a  # a = 4 (MoE-LLaVA) / 12 (Molmo)
    else:
        # Head token or language token: normal activation
        k_active = k  # k = 2 (MoE-LLaVA) / 8 (Molmo)

    indices = TopK(probs, k_active)
    # Renormalize weights over selected experts
    weights = Softmax(logits[indices])
    output = Σ_{j=1..k_active} weights[j] * ExpertFFN_j(x)
    return output

# Tail token identification (before VsDEA)
RPV_v = Variance(Softmax(V @ W_g), dim=1)  # [M]
threshold = Mean(RPV_v)                       # dynamic
is_tail = RPV_v > threshold                   # boolean [M]
```

**Tail Token Selection 消融对比**：
| Selection Method | % Tokens Selected | Avg |
|---|---|---|
| VHTs (Vision Head Tokens, low RPV) | ~87% | 58.0 |
| IATs (Instruction-Aware Tokens, attention-based) | 15% (fixed) | 57.7 |
| VTTs 10% (fixed threshold) | 10% | 57.9 |
| VTTs 15% (fixed threshold) | 15% | 57.9 |
| VTTs 20% (fixed threshold) | 20% | 57.4 |
| VTTs mean-RPV (LTDR, adaptive) | ~13% | **58.8** |

- Selecting head tokens (VHTs) 也有提升但远不如 tail tokens → 验证了 tail tokens 的高信息密度
- Fixed ratio thresholds 不如 adaptive mean-RPV → 动态阈值对不同数据分布更鲁棒
- IATs 效果最差 → 跨模态 attention 的噪声干扰 vision token 重要性判定

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现**：在 MoE layer 的 TopK 选择后添加 conditional branching（if is_tail: k→a）
- **Inference overhead**：几乎为零。All-to-All 通信速度由最慢 expert 决定（all-to-all barrier），VsDEA 不显著增加最慢 expert 负载。实测 V100 avg 1100s vs 1108s，A800 avg 846s vs 917s
- **Memory**：9.44G vs 9.44G baseline（V100-30G）→ 无额外内存开销
- **GPU Utilization**：59.29% vs 59.57% baseline → 几乎无变化
- **a 值选择**：MoE-LLaVA 上 a=4（vs k=2），Molmo 上 a=12（vs k=8）→ 经验性翻倍

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model
