## Generative Recommenders (GRs / 生成式推荐)

术语是什么？通过联网搜索让回答具体和精准。
Generative Recommenders (GRs) 是一种新兴的推荐系统范式，使用Transformer-based生成模型（如LLM或HSTU）替代传统Deep Learning Recommendation Models (DLRMs)，将推荐排序任务建模为sequence-to-sequence生成问题。GR将user profile、candidate items和system instructions编码为token序列，经Transformer self-attention处理后输出每个item的relevance score，最终选出top-k推荐。相比DLRM的embedding table + small dense MLP架构，GR具有更强的表达能力、可捕获复杂高阶user-item交互，且可由Scaling Law驱动——更大模型和更多计算带来更好的推荐效果。Meta的HSTU已在生产环境中部署并实现12.4% topline提升。然而GR的推理计算量比传统DLRM大两个数量级，呈现与LLM prefill阶段相似的compute-bound特性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GR ranking推理pipeline（以LLM-based GR为例）：
```
给定: user profile token set U, N个candidate items I={I1,...,I_N}, instruction tokens Instr
    L层Transformer, 每层含self-attention + FFN

// Step 1: Token序列组装
x^0 = Embed([U, I1, ..., I_N, Instr])  // [T, d] where T = |U|+Σ|I_i|+|Instr|

// Step 2: 逐层Transformer处理
for l = 1 to L:
    // Multi-head self-attention
    q^l, k^l, v^l = Proj(x^{l-1})  // QKV投影
    attn^l = CausalAttention(q^l, k^l, v^l)  // causal mask, 屏蔽跨item attention
    // FFN
    x^l = FFN(attn^l) + x^{l-1}

// Step 3: 判别token → relevance score
z = W_out · x^L[T]  // 最后一层最后一个token投影到vocab logits [V]
for each item i in 1..N:
    s_i = exp(z[v_i]) / Σ_{j=1}^{N} exp(z[v_j])  // softmax normalization

// Step 4: TopK排序
return TopK({s_1, ..., s_N})  // 返回前k个最高分的item
```
关键细节：attention mask屏蔽不同item之间的cross-attention（遵循HSTU设计），保证各item的评分独立性。GR inference的特征是compute-bound（类似LLM prefill），因为长输入序列（up to 8K tokens）需密集矩阵乘法。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流GR实现：
- **HSTU** (Meta, ICML'24)：Hierarchical Sequential Transduction Units，使用item embedding table + causal attention，万亿参数级，已在Meta生产部署。将user-item交互建模为next-token prediction。
- **LLM-based GR**：直接使用预训练LLM（如Qwen2-1.5B、Llama3-1B）fine-tune为推荐排序模型。通过自然语言或结构化token表示user profile和item属性。BAT采用此路线。
- **OneRec** (快手, 2025)：统一检索和排序的生成式推荐框架。
- **GenRank** (淘宝, 2025)：面向大规模工业级生成式排序。
GR可用于推荐系统的排序阶段（输入100-200个candidate items，输出top-k最终推荐），是推荐pipeline中计算最密集的环节。

涉及论文标题：
- BAT: Efficient Generative Recommender Serving with Bipartite Attention

---

