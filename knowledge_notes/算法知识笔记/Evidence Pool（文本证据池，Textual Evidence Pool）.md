## Evidence Pool（文本证据池，Textual Evidence Pool）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Evidence Pool 是 ECRD 中维护的一个文本证据集合 E_i = {E_1, ..., E_N}，在解码过程中动态增长。每个证据句 E_j 是一句自然语言描述，由 Visual Decider 在不确定性步生成，描述图像中与当前歧义相关的微观察（micro-observation）。证据池的关键设计特点：(a) 仅存文本——坐标存储在 GRIT 输出中用于可解释性但不参与 scoring，使得后续步骤可直接引用文本证据而无需重新编码图像；(b) 按需增长——初始化为全局图像描述 d_global（提供大范围覆盖），之后仅在 margin≤δ 时追加新证据；(c) 跨步复用——supervisor 在每一步都对证据池中所有证据计算支持度（Eq. 6），早期注入的微观察可在后续步骤中为相关 token 提供概率支撑。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Evidence Pool 生命周期
E_0 = {d_global}  # 初始化：全局图像描述

for step i in decoding:
    # 使用当前证据池评分所有候选 token
    for E_j in E_i:
        q_Ej(w) = mean_{t} p_VLM(w | E_j_prefix[0:t])
    S_i(w) = -log(mean_j q_Ej(w))
    
    # 需要时扩展证据池
    if k* > 1 and margin <= delta:
        w*, new_evidence = VisualDecider(image, prefix, C_i)
        E_{i+1} = E_i ∪ {new_evidence}
    else:
        E_{i+1} = E_i
```

证据示例：`"The first dress from the right-hand side is blue, partially hidden by the tree."`

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
证据池存储在 CPU memory 中（FP16），每步评分 O(k*|E_i|) 可忽略。文本证据的语义性质使其在 token 空间与 decoder 天然兼容——证据句子中的词直接映射到与候选 token 相同的 embedding 空间，无需额外的跨模态对齐。这与 RL-based 方法中反复编码图像裁剪的 pixel-space reasoning 形成对比：文本证据 compact、可组合、可跨步复用。

涉及论文标题：
- See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
