## Training-free KV Cache Compression (免训练KV缓存压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Training-free KV Cache Compression 是一类无需模型重训练、微调或校准数据即可应用的 KV Cache 压缩方法。与 training-required 方法（如 MLA、GQA 需要重新训练注意力结构）不同，training-free 方法直接对预训练模型的 KV Cache 运行时的存储和访问进行优化。主要分为两类：(1) Eviction（逐出）——选择性删除不重要的 token，如 H2O、StreamingLLM、SnapKV；(2) Quantization（量化）——降低不重要 token 的数值精度，如 KiVi、LogQuant、QAQ。

Training-free 的核心优势：即插即用，无需访问训练数据和 GPU 训练资源，适用于任何预训练模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Training-free Eviction 流程（H2O/StreamingLLM 模式）**：
```
// Prefill: 正常计算，所有 token 存入 KV Cache
// Decoding: 每步或每 k 步触发 eviction
if len(K_cache) > budget:
    scores = compute_importance(K_cache, V_cache, Q_current)  // H2O用A2S分数
    keep_indices = top_k(scores, budget)                      // StreamingLLM仅保留sink+recent
    K_cache = K_cache[keep_indices]
    V_cache = V_cache[keep_indices]
```

**Training-free Quantization 流程（LogQuant/KiVi 模式）**：
```
// 在选择保留全精度token后
for token in non_selected_tokens:
    K_quant[token] = quantize(K[token], bits=2, per_channel)
    V_quant[token] = quantize(V[token], bits=2, per_token)

// Decoding 时 dequantize
K_deq = dequantize(K_quant)
V_deq = dequantize(V_quant)
attention = softmax(Q @ concat(K_deq, K_fp).T / sqrt(d))
```

术语一般如何实现？如何使用？

Training-free 方法通常通过以下方式集成：(1) HuggingFace transformers 的 Cache 类派生——LogQuant、KiVi 均采用此方式；(2) vLLM/SGLang 等 serving 框架的 KV Cache 管理模块——通过修改 PagedAttention 的内存管理逻辑；(3) monkey-patch 模型的 forward 方法——在 attention 层插入 eviction/quantization 逻辑。

LogQuant 选择方法 (1)：继承 `transformers.Cache`，在 `update()` 方法中调用 `APPENDTOKEN` 算法，利用 Quanto 后端量化非保留 token。优势是与现有 HF 推理 pipeline 无缝兼容。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

**Sparse Frontier 论文对 training-free sparse attention 的系统化**：该论文是迄今最大规模的 training-free 稀疏注意力实证研究（7065 配置，3 模型家族，9 任务，sparsity up to 0.95）。提出四轴分类体系：(1) Unit of Sparsification（blocks/pages vs verticals/slashes），(2) Importance Estimation（fixed vs content-aware），(3) Budget Allocation（uniform vs adaptive/threshold-based），(4) KV Cache Management（eviction vs full cache retention）。基于此选择 6 种代表性方法进行 harmonized 实现并在 vLLM 框架内统一评估。

---
