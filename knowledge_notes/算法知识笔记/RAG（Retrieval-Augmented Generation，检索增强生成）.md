## RAG（Retrieval-Augmented Generation，检索增强生成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RAG 把外部知识检索与 LLM 生成结合：用检索到的文档支撑生成，缓解幻觉与知识过时，支持低成本知识更新。两阶段：(1) 检索阶段——离线用 embedding 模型把知识项编码建索引（通常用 ANNS 如 FAISS/HNSW，也可用 BM25 关键词检索），服务时把 query 编码后检索相似文档；(2) 生成阶段——query 与检索文档拼接成增广上下文输入 decoder-only transformer（每层 self-attention + FFN + 残差 + LN）。与普通 LLM 推理的关键差异：输入序列因拼接长文档而大幅变长（每请求可达上万 token），prefill 长上下文占计算主导并抬升 TTFT；文档静态可复用催生 KV 预计算优化。检索（ANNS/BM25 等）与生成解耦、可独立优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 检索阶段
index = build_index(embed(corpus))        # 离线：embedding + ANNS 索引（HNSW/FAISS）
docs  = ANNS_search(index, embed(query))  # 在线：query 编码 → top-k 检索
# 生成阶段（RAG 推理，MERIDIAN 聚焦此段）
ctx = concat(query, docs)                 # 增广上下文（doc 可达 ~14749 token，query ~10-20 token）
for layer in decoder_layers:              # 自回归生成
    attn(qkv(ctx)); ffn(...)
```
RAG 推理（generation stage）通常主导端到端延迟：HeterRAG 对比中 generation 占端到端延迟 88.84%+；因此 MERIDIAN 聚焦加速 generation，检索优化（IKS/DReX/Pyramid/ANSMET 等）作为正交补充。实测四数据集（2Wiki/HQA/NQ/TQA）doc 平均 856.76–14748.69 token、query 仅 10.28–20.41 token、response 3–5 token。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：主流框架 LlamaIndex、LangChain、Amazon Bedrock Knowledge Bases 等；检索用 FAISS（开源，github.com/facebookresearch/faiss）、HNSW、BM25；LLM 用任意 decoder-only 模型。RAG serving 优化方向：文档 KV 预计算复用（TurboRAG/BlockAttention/MERIDIAN）、缓存 KV 融合（CacheBlend）、检索加速（PIM 近存 ANNS）。MERIDIAN 在其 KV-precomputed 设定上做去中心化 PIM 推理，32 PIM 设备（16 DAC+16 CEC）共 16 TB 容量以容纳 TB 级文档 KV 库。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
