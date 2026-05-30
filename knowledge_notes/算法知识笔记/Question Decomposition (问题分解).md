## Question Decomposition (问题分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Question Decomposition（问题分解）是 D-CoDe 提出的核心方法组件之一，用于解决 Token Overload 问题——即视频输入产生的 visual tokens 数量远超图像预训练 VLM 的处理容量，导致模型无法有效利用全部信息。其核心思路是：使用外部 LLM（GPT-3.5-turbo-0125，temperature t=0.5）将复杂的原始视频问题分解为一组聚焦于视频不同方面的子问题（如角色位置、动作序列、物体交互、场景转换等），每个子问题独立用压缩后的 visual tokens 推理得到子答案，然后将所有子答案拼接为辅助文本，与原始问题和压缩 visual tokens 一起送入 LLM 生成最终答案。关键洞察：子答案（而非子问题本身）提供了多样化的补充语义信息，帮助模型从不同角度"消化"大量 visual tokens——消融实验证实子答案效果远优于子问题（58.0% vs 50.4% on EgoSchema）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Question Decomposition 在 D-CoDe 中的执行流程：
```
# === Question Decomposition ===
# 输入: 原始问题 Q, 压缩 visual tokens F_final
# 参数: t=0.5 (生成温度), model=GPT-3.5-turbo-0125

# Step 1: 子问题生成
system_prompt = """
I am working on a video understanding task. Your job is to break down 
the given question into a series of subquestions that guide the model 
toward solving the problem. The subquestions should focus on temporal 
and dynamic aspects of the video, rather than just static information 
that could be answered from a single frame.
"""
Q_1, Q_2, ..., Q_n = GPT3.5(Q, system_prompt, temperature=t)
# n 不限制, 由 GPT-3.5 自主决定

# Step 2: 逐子问题推理
A_sub = []
for Q_i in [Q_1, ..., Q_n]:
    A_i = LLaVA_NeXT(F_final, Q_i)      # 独立推理
    A_sub.append(A_i)

# Step 3: 融合子答案生成最终答案
# 将子答案拼接为辅助 prompt segment
aux_text = concat(A_sub)
A_final = LLaVA_NeXT(F_final, aux_text, Q)
```

关键设计决策（消融实验验证）：
- **子答案 vs 子问题**：Table 8 显示将子问题（而非子答案）喂入模型反而降低 accuracy（50.4% vs 51.8% w/o decomposition），说明性能提升来自子答案提供的多样化中间信息，而非结构化思考过程
- **Prompt 设计**：移除任务背景解释降低 accuracy（53.2% vs 58.0%），移除"temporal and dynamic aspects"降低 accuracy（54.8% vs 58.0%），但改写措辞不影响（58.4% vs 58.0%），说明语义内容决定性能而非措辞
- **开放域 VideoQA 不适用**：对于简单问题（如 MSVD-QA 的 "What is a man sitting on?"），分解反而过度复杂化并降低 accuracy（72.4% vs 80.0% w/o decomposition），因此 D-CoDe 仅在 Multiple Choice VideoQA 上使用 Question Decomposition

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Question Decomposition 需要调用外部 LLM API（GPT-3.5-turbo-0125），因此引入了显著的推理延迟：+Question Decomposition 使延迟从 6.115 s/sample 增加到 37.395 s/sample（+511%）。轻量变体：使用更小的 CLIP（35% params）→ 35.466 s/sample；限制子问题数 = 5 → 26.273 s/sample, accuracy 56.0%；限制子问题数 = 7 → 33.704 s/sample, accuracy 57.8%。Question Decomposition 的优势场景是复杂、多步推理的问题（如 EgoSchema 的 schema 级理解、NExT-QA 的因果推理），对简单空间查询（如 MSVD-QA）效果负面。实现代码在 `Dcode.py` 的 `generate_subquestions()` 函数中，依赖 `OPENAI_API_KEY`。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition
