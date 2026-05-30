## Passkey Retrieval Task (长上下文关键信息检索评估)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Passkey Retrieval 是 Yarn（Peng et al., 2023）提出的评估 LLM 长距离依赖处理能力的 benchmark 任务。任务构造：在一大段无意义文本（"needle in a haystack"）中隐藏一个简单的 passkey（如数字串 "12345"），然后在文本末尾提问 "What is the passkey?"。模型需要在极长上下文中定位并提取该信息。关键变量：(a) passkey 在文本中的深度位置（如 0%, 25%, 50%, 75%, 100%）；(b) 总上下文长度（如 10K、100K tokens）。该任务在 Quest 论文中的特殊价值：对于 query-agnostic KV cache 驱逐方法（H2O/TOVA/StreamingLLM），passkey 在 question 之前出现，可能在 decode 阶段被提前驱逐，导致准确率 0-4%；而 Quest 不驱逐任何 token 且基于 query 动态选择，64-1024 token budget 即达 100% 准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Passkey Retrieval 评估流程 (Quest 论文设置)
// 构造输入:
haystack = "The grass is green. The sky is blue. " × N   // 无意义填充文本
passkey = "The pass key is 12345."
question = "What is the pass key?"

// 测试条件:
depth_ratio ∈ {0%, 25%, 50%, 75%, 100%}  // passkey 插入位置
total_length ∈ {10K, 100K}               // 总 token 数

// Quest 论文的特殊设置（模拟多轮对话）:
// question 被逐 token feed 到 decode 阶段（而非 prefill）
// 因此 H2O/TOVA 可能在 decode 期间驱逐包含 passkey 的 token
// Quest 保留所有 token，靠 query-aware 选择在需要时加载

for each (depth, length) combination:
    prompt = haystack[:pos] + passkey + haystack[pos:]
    prefill(prompt)                       // FlashAttention + full cache
    for each question_token:              // 逐 token decode
        Q = embed(question_token) @ W_Q
        critical_pages = Quest.estimate(Q, page_metadata)
        O = SparseAttention(Q, KV_cache[critical_pages])
        // H2O/TOVA 在这里可能已驱逐 passkey
    accuracy = (generated_answer contains "12345")
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Passkey Retrieval 被广泛用于长上下文 LLM 评估。扩展形式包括 RULER benchmark 中的 Needle In A Haystack (NIAH) 变体（多针、多值、多查询）。Passkey retrieval 与 language modeling perplexity 互补——前者测量长距离精确检索能力，后者测量局部语言建模能力。Quest 论文的实验表明，对 query-agnostic 方法最具挑战性的正是 passkey retrieval（因 token 在 question 之前被驱逐），而对 query-aware 方法恰好是优势所在。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- Yarn: Efficient Context Window Extension of Large Language Models
