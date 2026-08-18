## 文档 KV 预计算与复用（Document KV Precomputation & Reuse）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV-precomputed RAG 的核心优化：对检索语料中每篇文档，离线预计算其 Key-Value（KV）缓存并存库（如 TurboRAG [44]、BlockAttention [46]），推理时不再重复编码长文档，直接加载缓存 KV 与 query 实时生成的 KV 拼接进注意力，配合轻量微调维持生成质量，可将 TTFT 降低最高 98%。这是"KV 复用"思想在 RAG 语料侧的扩展（区别于 serving 层跨请求共享 KV Cache/prefix caching）：文档 KV 是静态可重用的（同一文档被多请求检索），但规模可达 TB 级（500K 文档约 14 TB），远超设备显存（H100 80 GB），因此必须驻留 host DRAM 并在 query 时经 PCIe/CXL 搬运到设备——这正是集中式 KV-reuse 范式（centralized KV-reuse paradigm）的通信瓶颈来源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线（检索语料构建一次）：
for doc in corpus:
    K_doc, V_doc = encode(doc)          # 文档侧 KV 预计算，轻量微调对齐
    store(K_doc, V_doc)                 # 存 host DRAM（集中式）或 shard 到 PIM（MERIDIAN）
# 在线推理（每 query）：
(q,k,v) = QKVProjection(query)          # 仅 query 需实时编码（~16 token）
K_c, V_c = cache[doc_ids]               # 拉取文档 KV（集中式：整份搬上设备）
attn = softmax(q@[K_c;K_c]^T) @ [V_c;V_c]  # 拼接后集中注意力
```
通信量：集中式每次 query 搬 #Doc tokens×2×d_model×2 bytes（FP16）；MERIDIAN 用文档注意力分解把文档 K/V 分片驻留 PIM，只传 query 向量并回收局部摘要，使"预计算"的红利不被跨设备搬运抵消。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现代表：TurboRAG（chunked 文本预计算 KV cache）、BlockAttention（块级 KV 缓存，prefill 只算 query token）；选择性重算变体：CacheBlend（高偏差 token 重算拼接）、EPIC（chunk 首 token 重算），与预计算正交。MERIDIAN 采用该复用范式但重构 KV 驻留与执行：文档 KV 按 head shard 写进 CXL Type-3 PIM 设备（标准 CXL.mem load/store），文档更新/语料扩展直接写对应 shard、无需系统级重排或重建索引。使用场景：企业 RAG 服务、个性化/隐私敏感部署（小 batch、低延迟 SLO）、长文档 QA。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
