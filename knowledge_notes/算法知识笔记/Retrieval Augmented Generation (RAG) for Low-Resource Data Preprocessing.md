## Retrieval Augmented Generation (RAG) for Low-Resource Data Preprocessing

术语解释
RAG（Retrieval Augmented Generation）是利用检索系统从外部知识库中获取相关信息并作为上下文提供给 LLM 以提升生成质量的方法。MELD（KDD '24）提出增强型 RAG 系统用于跨域检索、自标注（self-annotation）和训练数据扩增。

术语是什么？
MELD 的增强型 RAG 包含三个关键组件：
1. **Entry Alignment**：对结构化/半结构化数据，结构相似性与语义相似性同等重要。为每个 query q 构建正例集 P_q（对齐 entries）和负例集 N_q（未对齐 entries）。
2. **Fine-tuning RAG Model**：使用 sentence-bert（bge-large-en）作 backbone，contrastive loss 微调。负例集为空时做 hard negative mining。
3. **Self-Annotation**：微调后的 RAG 模型自动为未标注 query 检索最相似 instance 并生成伪标签，扩大训练集。还可通过 query transformation 实现跨任务数据增强（如 EM query → DI query）。

从算法pipeline角度拆解术语。
```
Input: few-shot labeled data X_i, unlabeled data X̃_i

// Step 1: Entry Alignment
for each query q:
    Serialize q to dict (tuple + meta: table title, column headers)
    Build P_q (positive), N_q (negative)

// Step 2: Fine-tune RAG
M_RAG = bge-large-en; τ=0.02
loss = -log(exp(cos(emb_q,emb_p)/τ) / Σ_{p'} exp(cos(emb_q,emb_p')/τ))

// Step 3: Self-Annotation
for each unlabeled q_i:
    q_j = argmax cos(M_RAG(q_i), M_RAG(q))
    Annotate q_i with label from q_j
    // Cross-task: EM→DI transformation via masking
```

术语一般如何实现？如何使用？
- MELD 使用 bge-large-en 作为 RAG backbone，temperature τ=0.02
- 与 meta-path 数据增强协同：RAG 负责跨域检索和自标注，meta-path 负责结构化增强
- Ablation 显示 w/o RAG 在所有数据集上性能显著下降（如 Semi-Text-Computer F1 86.46→42.02）

涉及论文标题：
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing
