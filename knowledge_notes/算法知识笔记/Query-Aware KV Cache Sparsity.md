## Query-Aware KV Cache Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Query-Aware KV Cache Sparsity 是一种动态稀疏注意力策略，核心思想是：KV cache 中 token 的关键性取决于当前 query token，因此不能预先静态决定哪些 token 重要/不重要，而必须在每步 decode 时根据当前 query 动态选择关键 token。Quest（ICML 2024, MIT HAN Lab）首次系统性地提出并验证了这一概念：传统 KV cache 驱逐方法（H2O、TOVA、StreamingLLM）基于历史 attention 或固定窗口预判 token 重要性（query-agnostic），会丢弃对将来 query 可能关键的 token，导致 passkey retrieval 等长依赖任务准确率近乎 0%。Quest 不驱逐任何 token，而是在每步 decode 评估所有 KV cache page 对当前 query 的关键性，仅加载 Top-K 关键 page 参与 attention。

关键洞察（Quest Fig. 2）：对于 prompt "A is B. C is D. A is"，token "B" 在 query="is" 时 attention score 很高（因为是正确答案），但在之前的 query（"C", "is", "D"）中 attention score 很低。因此同一 token 的关键性随 query 变化而剧烈变化，query-agnostic 方法会错误地丢弃它。Quest Fig. 4 量化了这一效果：H2O（历史注意力累积）的 recall rate 远低于 100%，而 Quest 基于当前 query 的 recall rate 接近 full attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Quest 的 Query-Aware 两阶段稀疏注意力流程**：

```
// === 预处理（Prefill 后，每个 KV cache page） ===
for each page p (含 S=16 tokens):
    for each channel i in 1..d_head:
        M_i^p = max(M_i^p, k_i)  // per-page channel-wise max Key
        m_i^p = min(m_i^p, k_i)  // per-page channel-wise min Key
// 元数据大小: 2 × num_pages × d_head × 2 bytes (FP16)

// === Decode 阶段每步 ===
// Stage 1: Criticality Estimation
Input: Q ∈ R^{d_head}, all {M^p, m^p} for p=1..num_pages
for each page p:
    score_p = 0
    for each channel i in 1..d_head:
        U_i = max(Q_i * m_i^p, Q_i * M_i^p)  // 保证 U_i ≥ Q_i * K_i^(t) ∀t∈p
        score_p += U_i                        // page attention score 上界
top_k_indices = TopK({score_p}, k=K)         // K = token_budget / page_size

// Stage 2: Approximate Attention (仅加载 Top-K pages)
K_selected = load_K_pages(top_k_indices)   // K×S × d_head
V_selected = load_V_pages(top_k_indices)
S = Q @ K_selected^T / sqrt(d_head)        // 仅计算选中 tokens 的 attention
A = softmax(S)
O = A @ V_selected

// 内存加载量: 完整 KV cache 的 (1/PageSize + K/PageNum)
// 例: page_size=16, 64K context (4096 pages), K=256 → ~12.5% of full KV cache
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Quest 基于 FlashInfer kernel 库实现（开源：https://github.com/mit-han-lab/Quest）。核心实现组件：(1) 在 FlashInfer 中新增 per-page metadata buffer（channel-wise min/max Key values），插入新 token 时 O(d_head) 更新；(2) Criticality estimation CUDA kernel：element-wise max(Q_i*m_i^p, Q_i*M_i^p) + reduce-sum，计算 per-page upper-bound score；(3) Top-K filtering 使用 RAFT（RAPIDS）batched Top-K CUDA operator，延迟仅 5-10 µs；(4) Approximate attention 利用 FlashInfer 的 PageAttention 接口，传入 sparse page indices 执行仅选中 page 的 FlashAttention。前两层保持 full attention（因观察到稀疏度 <10%），其余层使用 Quest。支持 Llama-3.1、Mistral-v0.3 等模型家族。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

**Sparse Frontier 论文中的 Quest 实现与评估**：使用 page_size=16（消融实验确定），始终包含当前 token 所在 page。在 Sparse Frontier 的全量评测中，Quest 是 decoding 阶段整体最佳方法。Quest 在 0.95 sparsity (1/20 budget) 下仍可优于更小的 dense 模型。但 Quest 在合成数据（Ruler NIAH）上表现退化——随机符号序列导致 key representations 区分度下降，page-level 粗粒度放大这一效应。相反在自然语言 retrieval（Story Retrieval）上 Quest 优于 Ada-SnapKV。

---
