## Query Rewriting for Personalized Retrieval（个性化检索的查询重写）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Query Rewriting 是 PEARL Concept-aware Retrieval Algorithm 中的关键预处理步骤。核心问题：通用多模态嵌入模型（Qwen3-VL-Embedding-2B）在训练时从未见过用户动态定义的个性化名称（如 "Adaliz"、"Action A"），无法将含个性化名称的查询有效编码为与视频 clip 嵌入语义对齐的向量。解决方案：在编码查询前，将查询中所有个性化概念名替换为 Concept Memory 中存储的文本描述。例如："What was Adaliz wearing when she was cooking?" → "What was a young female with long black hair and oval face wearing when she was cooking?"。重写后的查询包含嵌入模型可理解的视觉语义特征（性别、发型、面部特征），从而能与 Streaming Memory 中编码了相似视觉内容的 clip 嵌入进行有效余弦相似度匹配。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Query Rewriting 的计算流程：

```
def rewrite_query(original_query, replacement_rules):
    # replacement_rules = {
    #   "Adaliz": "a young female with long black hair and oval face",
    #   "ActionA": "the action of squatting down and then leaping forward"
    # }
    prompt = f"""Rewrite the following question by replacing the 
concept names with their visual descriptions.
Keep the sentence grammatically correct and natural.

Original question: {original_query}
Replacement rules:
"""
    for name, desc in replacement_rules.items():
        prompt += f'- "{name}" → "{desc}"\n'
    prompt += "\nOutput ONLY the rewritten question, nothing else."
    
    rewritten = vlm.generate(prompt)  # 纯文本推理，低延迟
    return rewritten.strip()
```

关键设计：(a) 使用 Concept Memory 中预先存储的描述，无需额外 VLM 推理理解概念语义；(b) 替换保持语法正确性（冠词调整等）；(c) 不改变查询原始语义意图。消融实验（Table 4）证实 Query Rewriting 将 Avg 准确率从 47.96% 提升至 52.24%（+4.28%），对 Real-Time 和 Past-Time 均有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过 VLM 的纯文本推理实现（仅 token 级别的 prompt + replacement_rules → rewritten_query），延迟极低。PEARL 代码库中集成在 `video_qa_inference.py`。重写模板包含 `{query}` 和 `{replacement_instructions}` 两个占位符，输出仅为重写后的问题。适用场景：任何需要将用户自定义名称/标识符映射为嵌入模型可理解的语义描述的检索场景，特别是概念动态定义、嵌入模型训练数据中不包含这些个性化标识符时。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model
