## VDGD (Visual Description Grounded Decoding，视觉描述接地解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VDGD 是 ECRD 的前身方法，由 Ghosh 等人于 2024 年提出（arXiv:2405.15683）。VDGD 是一种 training-free 的 LVLM 解码策略，核心思想是：先让模型生成图像的全局文本描述 d=(d1,...,dL)，然后在自回归解码的每一步，对候选 token w 计算其与描述 prefix d_{<j} 之间的 KL 散度（即 -log p_VLM(w | d_{<j})），取所有 prefix 长度 j 上的最小值作为该 token 的"接地得分"，用此得分替换 base logits 后做 softmax。VDGD 的关键洞察是：在描述图像时模型能正确"看到"的视觉细节，在推理时可能被语言先验压制——通过强制解码分布与描述一致，可恢复视觉接地。VDGD 的局限在于：(a) 使用静态单一描述，缺乏自适应能力；(b) min-over-prefix 聚合不稳定，容易受个别 prefix 波动影响；(c) 直接替换 logits 丢弃了 base 模型的校准置信度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# VDGD 解码流程
d = LVLM.generate_description(image)  # 全局描述 (d1,...,dL)
k* = knee_truncation(p_i)             # 候选集大小
C_i = top_k(p_i, k*)                  # 候选 token 集

for w in C_i:
    # min-over-prefix KL 散度
    score(w) = min_{j=1..L} KL(onehot(w) || p_VLM(.|d_{<j}))
            = min_{j=1..L} -log p_VLM(w | d_{<j})

# 用 scores 替换原始 logits
p_VDGD = softmax_{w in C_i}(score(w))
x_i = argmax(p_VDGD)
```

ECRD 对 VDGD 的改进：(a) min→mean：将 min-over-prefix 替换为 mean-over-prefix，更稳定且奖励持续支持；(b) 单证据→多证据：支持证据池中多条证据的平均支持度；(c) 替换→混合：从直接替换 logits 改为与 base 分布通过自适应权重协商混合，保留 base 模型在自信步的行为。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VDGD 作为 training-free decoding wrapper，包裹任意 frozen LVLM，无需 fine-tuning。论文报告跨多个 benchmark 和 LVLM 一致提升 2%-33%。VDGD 在 ECRD 的 ablation 中作为 baseline 对比：Qwen2.5-VL-7B + VDGD 在 TreeBench 上 37.0%→39.5%（+2.5），远低于 ECRD 的 +10.9。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
