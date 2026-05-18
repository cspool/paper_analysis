## KV Cache Retrieval（KV缓存检索）

术语是什么？通过联网搜索让回答具体和精准。

KV Cache Retrieval是一种通过将完整KV cache offload到CPU memory或storage，并在推理时选择性fetch相关token来降低GPU memory占用的技术。与pruning、compression、quantization等破坏性方法不同，KV cache retrieval保留全部历史KV cache，仅在每层attention前动态选择最相关token子集取回GPU memory参与计算。典型代表包括FlexGen（offload to CPU/storage）、InfiniGen（generation-stage retrieval）、ReKV（frame-level selection）和V-Rex的ReSV（dynamic per-layer/head retrieval）。核心trade-off是PCIe带宽（4-32 GB/s）远低于GPU memory带宽（1-2 TB/s），因此必须通过高效选择算法最小化fetch volume。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

KV Cache Retrieval在streaming video LLM中的三阶段pipeline（以FlexGen baseline为例）：

```
// Stage 1: Offloading
for each new KV cache entry generated:
    if GPU_memory_used > threshold:
        offload oldest KV entries to CPU memory or storage
        // 完整KV cache永不丢弃，仅迁移到低速存储

// Stage 2: Selection (KV Prediction)
for each decoder layer L:
    Q, K_new, V_new = QKV_Gen(input_hidden)
    // 计算query与历史key的attention score
    scores = Q @ K_history^T           // 在GPU上执行
    top_k_indices = topk(scores, k)    // fixed k selection
    // 固定top-k：所有layer/head用相同k值

// Stage 3: Pre-fetching
    selected_K = fetch_from_cpu(top_k_indices)  // PCIe传输
    selected_V = fetch_from_cpu(top_k_indices)
    // 将选中的K/V prefetch到GPU memory
    attention_out = Attention(Q, selected_K, selected_V)
```

关键特征：(a) Preserve context integrity：完整KV cache始终保留于CPU/storage，未来query可访问任意历史token，支持multi-turn对话。(b) Selective computation：仅对选中的KV子集计算attention，减少计算量。(c) PCIe bottleneck：fetch受PCIe带宽限制（4-32 GB/s vs GPU memory 1-2 TB/s），retrieval latency在streaming video 40K cache length下可达85%总延迟。(d) Selection overhead：KV prediction computation本身也随sequence length增长。

固定top-k的缺陷：token importance在不同layer和head间分布高度不均（有的layer仅需4.2% token，有的需44.0%），固定k导致不重要位置over-fetch浪费PCIe带宽、关键位置under-fetch降低accuracy。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlexGen (https://github.com/FMInference/FlexGen) 将KV cache offload到CPU memory和SSD的三级存储层次（GPU→CPU→SSD），使用线性规划优化offloading schedule。InfiniGen在generation stage做KV cache retrieval，异步prefetch隐藏fetch latency。V-Rex的ReSV通过hash-bit key clustering和WiCSum thresholding实现动态per-layer/head token selection，但未开源。使用流程：模型加载→配置offloading target→设置token budget或selection ratio→每层decoder前计算KV prediction→通过PCIe fetch selected KV→执行attention。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

---

