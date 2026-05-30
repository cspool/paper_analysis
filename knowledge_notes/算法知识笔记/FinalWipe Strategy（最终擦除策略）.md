## FinalWipe Strategy（最终擦除策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FinalWipe 是 TwigVLM 提出的一种简单但有效的 token 剪枝补充策略：在 VLM 推理的某个较深层（Kf，如第 24 层）之后，**移除所有剩余的视觉 token**，后续层仅处理文本 token。其动机基于两个观察：(1) 此前研究已表明 VLM 深层（如 20 层之后）的视觉 token 对最终预测贡献极小；(2) 在固定平均保留 token 数 R̄ 下，FinalWipe 允许在前中层保留更多 token R（因为深层 token 数为 0 拉低了平均值），从而提升剪枝后的模型精度。引入 FinalWipe 后，R̄ 的计算从 R̄=[M×K+R×(L-K)]/L 变为 R̄=[M×K+R×(Kf-K)]/L，在 K 和 R̄ 固定时允许更大的 R。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FinalWipe: 在 Kf 层后移除所有 visual tokens
# Kf: FinalWipe 位置 (如 24)
# L: VLM 总层数 (如 32)

def forward_with_final_wipe(X_kept, Kf, L):
    # X_kept: 已被 TTP 剪枝后的序列 [kept_visual_tokens, text_tokens]
    # 前向到 Kf 层
    for layer in range(K+1, Kf+1):
        X_kept = transformer_layer(X_kept)
    
    # FinalWipe: 移除所有 visual tokens
    # X_kept = [visual_R, text_N] → [text_N]
    X_text_only = X_kept[text_positions]
    
    # 剩余层仅处理 text tokens
    for layer in range(Kf+1, L+1):
        X_text_only = transformer_layer(X_text_only)
    
    return X_text_only
```

效果（TwigVLM 消融实验，T=3, K=2, R̄=64）：
| FinalWipe | Kf | R | RelAcc | RelSpd |
|-----------|----|---|--------|--------|
| × | 32 | 30 | 93.1% | 154.6% |
| ✓ | 20 | 50 | 95.8% | 151.3% |
| ✓ | 24 | 41 | 96.0% | 153.6% |
| ✓ | 28 | 37 | 95.1% | 154.1% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FinalWipe 是一种纯推理时策略，无需额外训练。在 TwigVLM 的推理代码中，通过 `forward_high_layers(final_wipe=Kf)` 参数控制。Kf 的选择需要在精度（更大的 R）和速度（更少的 FFN 计算）之间权衡。消融实验表明 Kf=24 是最优平衡点。实现简单：在指定的 Kf 层后，将 attention 计算中的 KV-cache 视觉 token 部分置为不可见即可。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
