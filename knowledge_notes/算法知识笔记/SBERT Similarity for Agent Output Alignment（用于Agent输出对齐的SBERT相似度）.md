## SBERT Similarity for Agent Output Alignment（用于Agent输出对齐的SBERT相似度）

术语是什么？通过联网搜索让回答具体和精准。

SBERT（Sentence-BERT，Reimers & Gurevych, 2019）是BERT的孪生网络变体，通过pooling操作将句子映射到固定维度的embedding空间，使语义相似的句子在embedding空间中有较高的余弦相似度。AIMS使用SBERT embedding的cosine similarity作为衡量SLM和LLM输出对齐程度的核心度量：(1) 请求级：比较整请求在All-SLM和All-LLM下的最终输出embedding相似度；(2) subtask级：比较SP_SLM和SP_LLM预测的next subtask embedding相似度；(3) 收敛检测：比较未来SLM-LLM subtask pair的embedding相似度。阈值默认0.7（empirically determined），低于此阈值认为输出不similar，需走LLM或进入recovery path。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// AIMS中的SBERT相似度计算
function SBERT_similarity(text_a, text_b):
    emb_a = SBERT.encode(text_a)  // shape: [1, 384] (使用all-MiniLM-L6-v2等)
    emb_b = SBERT.encode(text_b)  // shape: [1, 384]
    
    cos_sim = (emb_a · emb_b) / (||emb_a|| * ||emb_b||)
    // 值域: [-1, 1], 0.7+ 认为相似
    
    return cos_sim

// 在SSE中使用
similarity_score = SBERT_similarity(
    SP_SLM.predict(ST_i),   // SLM预测的next subtask
    SP_LLM.predict(ST_i)    // LLM预测的next subtask
)
```

AIMS选择SBERT similarity而非精确匹配（如BLEU/ROUGE）的原因：AI agent的subtask是自然语言描述的动作/决策，语义等价但措辞不同的subtask应视为相似。例如SLM的"Search for James Cameron's mother"和LLM的"Find the director's maternal parent"语义高度相似但文本不同，SBERT能捕捉这种semantic alignment。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SBERT开源实现：https://github.com/UKPLab/sentence-transformers（`pip install sentence-transformers`）。常用模型包括`all-MiniLM-L6-v2`（384维，快速）和`all-mpnet-base-v2`（768维，更准确）。AIMS使用SBERT替代BERTScore作为主要相似度度量（Section 2.2中BERTScore用于WebShop的ground-truth匹配评估），因为SBERT的sentence-level embedding更适合比较不同长度的subtask描述。SBERT similarity同时用于：(1) URC训练标签（All-SLM vs All-LLM输出相似度）；(2) SSE/CD的在线判断；(3) offline profiling数据分析。相似度阈值0.7是论文在多个数据集上empirically确定的，Section 4.6的sensitivity study显示阈值在0.66-0.74范围内提供良好accuracy-SLM usage balance。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

---

