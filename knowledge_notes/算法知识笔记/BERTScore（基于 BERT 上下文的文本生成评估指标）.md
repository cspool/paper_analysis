## BERTScore（基于 BERT 上下文的文本生成评估指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BERTScore（Zhang et al., arXiv:1904.09675）是用预训练 BERT 的上下文嵌入评估生成文本与参考答案语义相似度的指标，范围 [0,1]（越高越相似）。流程：候选文本 y 与参考文本 x 各 token 化并经 BERT 提取上下文向量 → 计算两文本 token 对间的余弦相似度矩阵 → 对每个候选 token 找参考中最相似 token（Precision）、对每个参考 token 找候选中最相似 token（Recall），F1 为调和平均。公式（Web 证据确认）：R_BERT = (1/|x|)·Σ_{x_i∈x} max_{y_j∈y} x_iᵀy_j，P_BERT = (1/|y|)·Σ_{y_j∈y} max_{x_i∈x} x_iᵀy_j，F_BERT = 2·P·R/(P+R)。相比 BLEU/ROUGE 的精确词匹配，BERTScore 语义感知（同义/改写仍高分），适合 QA、摘要、RAG 等生成任务。PRowhammer（ISCA'26）用它量化 LLM 攻击：Llama-2-7B/Mistral-7B/Falcon-7B（4-bit 量化，llama.cpp/GGML）在 Google Natural Questions 100 问上的平均 F1 从 pristine 0.58–0.62 跌到 corrupted 0.25–0.30——论文还验证：即使对任意常量/无关字符串，BERTScore 也落在 0.25–0.30 区间（说明该区间近似"无语义"下限）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# BERTScore 计算（对单个 QA 样本）
def bertscore(x_tokens, y_tokens, model):        # x=参考答案, y=模型生成
    Ex = model.embed(x_tokens)                    # 参考 token 上下文嵌入
    Ey = model.embed(y_tokens)                    # 候选 token 上下文嵌入
    sim = cosine_similarity(Ex, Ey)               # |x|×|y| 相似度矩阵
    P = mean( max_j sim[i][j] for i in cand )     # 候选→参考 对齐
    R = mean( max_i sim[i][j] for j in ref  )     # 参考→候选 对齐
    return 2*P*R/(P+R)                            # F1
```
PRowhammer 的评估管线：对 100 个 NQ 问题，人工标注参考答案 → 用 pristine 模型与 corrupted 模型各生成回答 → 分别算 BERTScore F1 → 报告 100 问平均（表 VI：A6000/4090/5060 上三模型 pristine 0.58–0.62 → corrupt 0.25–0.30）。攻击效果分级：灾难性（输出 # 串或跨语言乱码，BERTScore≈0.25–0.30）与"连贯但错误"（如 "Spike"→"Momo"，表面流畅而事实错误，更难检测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：bert_score Python 库（预训练 BERT/RoBERTa 模型权重）；可调 token 加权（IDF）、层选择（约第 9 层嵌入最佳）、线性重缩放。使用：LLM 生成任务评估（QA/摘要/翻译/RAG）。在 PRowhammer 中作为攻击效果度量而非优化目标；论文用它证明单 bit-flip 足以让 7B 级 LLM 生成无意义或事实错误文本，同时指出 0.25–0.30 是常量/无关字符串的 BERTScore 下限，需配合人工检查区分"乱码"与"连贯但错误"两类失败模式。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU
