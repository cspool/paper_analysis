## KV Cache Offloading with Sparse Top-k Retrieval (基于稀疏Top-k检索的KV Cache卸载)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Offloading with Sparse Top-k Retrieval 是一种将完整 KV cache 存储在 CPU 内存中、每次 decoding step 仅将当前 query 最相关的 k 个 KV pair 传输到 GPU 的系统架构策略。与传统的 KV cache offloading（如 FlexGen）有本质区别：(1) 传统 offloading 每 step 将全部层的完整 KV cache 在 CPU↔GPU 间往返搬运（单层 1.6GB × L=32 ≈ 51GB for 8B model @ 100K context），数据搬运本身成为瓶颈；(2) Sparse Top-k Retrieval 利用 attention 的天然稀疏性，仅传输 k 个 value 向量（k ≪ N），数据搬运量从 O(N·D·L) 降至 O(k·D·L)，在 k=2% of N 时降低 50×。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

**Top-k KV Cache Offloading 系统架构流程（Llama-3-8B, 1M context, 16GB commodity GPU）**：

```
┌──────────────────────────────────────────────────────────┐
│                    CPU (Host Memory)                      │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Complete KV Cache: {K_ℓ, V_ℓ} for ℓ=1..32      │     │
│  │  Size: ~520 GB (BF16, 1M tokens, L=32, D=4096) │     │
│  └─────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Faiss ANN Indexes (per head, per layer)         │     │
│  │  - IndexFlatIP(d_k): exact inner product search  │     │
│  │  - Each query → top-k key indices + scores       │     │
│  └─────────────────────────────────────────────────┘     │
│                          │                               │
│         PCIe: only k×D values transferred                │
│         (vs full cache: N×D in traditional offloading)   │
│                          │                               │
└──────────────────────────┼───────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────┐
│              GPU (~16GB VRAM)                             │
│  ┌─────────────────────────────────────────────────┐     │
│  │  Model Weights (e.g., Llama-3-8B: ~16 GB)       │     │
│  └─────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────┐     │
│  │  GPU-side Window Cache (recent generated tokens) │     │
│  │  K_gen, V_gen: O(window_size) ~ constant        │     │
│  └─────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────┐     │
│  │  QKV Projection (cuBLAS matmul)                  │     │
│  │       ↓                                           │     │
│  │  q → CPU for ANN search                          │     │
│  │  V_sel[I], vals ← from CPU (k elements)          │     │
│  │       ↓                                           │     │
│  │  attention_out = softmax(vals/sqrt(d)) @ V_sel   │     │
│  │  + softmax(q@K_gen^T/sqrt(d)) @ V_gen            │     │
│  └─────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

**与传统 KV Cache Offloading (FlexGen) 的对比**：

| 方面 | FlexGen | Top-k Sparse Offloading |
|------|---------|------------------------|
| 数据搬运量 | O(N·D·L) per step | O(k·D·L) per step |
| CPU-GPU 带宽瓶颈 | 严重（51GB×数百 step） | 大幅缓解（k=128 时 ~1-2MB） |
| CPU 计算 | 无/少量（仅做地址管理） | Faiss ANN search (O(log N) with HNSW) |
| GPU 显存 | 仅少量 KV cache 驻留 | 仅 window cache + model weights |
| Context 上限 | ~100K (带宽限) | 1M+ (CPU 内存限) |
| Attention 完整性 | 完整（但有搬运成本） | 近似（top-k sparse） |

术语一般如何实现？如何使用？

实现：(1) Prefill 阶段：在高算力 GPU (H100) 上使用 FlashAttention 一次性构建完整 KV cache 并存至 CPU memory；(2) Index 构建：使用 Faiss IndexFlatIP 或 IndexHNSWFlat 对每层每 head 的 key tensors 构建 ANN index；(3) Decoding 阶段：每 step 对每层执行 q→CPU ANN search→V[I] 搬回 GPU→attention 计算→合并 GPU 本地 window cache attention；(4) k 值选择：基于任务类型和 attention entropy 分析动态调整（NIAH 仅需 k=1，Word Counting 需 ~9% of context）。

适用场景：(a) "Rent cloud for prefill once, decode locally many times"——用户租用 H100 完成一次性 prefill，在本地 16GB GPU 上执行多次 query；(b) 文档分析场景——大量文档作为固定 context，用户反复提问；(c) 代码库 QA——大型代码库作为 context，开发者多次查询。不适用于 context 频繁变化或 streaming 输入场景（每次 context 更新需重新 prefill）。

**MOM-style Static KV Cache Offloading (Prefill-offload, Decode-reload)**：

MOM 采用更简单的 offloading 策略：prefill 阶段每层 attention 计算完后立即将 KV cache offload 到 CPU，全部 prefill 完成后 decode 阶段开始前将所有层 KV cache 统一 reload 回 GPU。与 FlexGen 的 quadratically-scheduled offloading 不同，MOM 的 offloading 不追求在 decode 中反复搬运，而是利用 Mini-sequence 大幅降低 MLP 中间激活后，offloading 进一步释放 KV cache 占用的 GPU 内存来容纳更长序列。因为 decode 阶段 KV cache 已全部回 GPU，decode 速度几乎不受影响（数据仅在 prefill→decode 转换时搬运一次）。

```
// MOM KV Cache Offloading 流程
// Prefill stage:
for layer in 1..L:
    A = Attention(X, causal_mask)
    KV_cache[layer] ← update(A)       // 存储 KV cache
    offload_to_CPU(KV_cache[layer])    // 立即 offload to CPU
    X = MiniSequence_MLP(A)            // Mini-sequence 处理（仅 MLP）

// Decode stage:
reload_all_KV_to_GPU()                 // 一次性全量 reload
for step in 1..max_new_tokens:         // 标准 autoregressive decode
    token = decode_step(KV_cache_on_GPU)
```

MOM 的 offloading 与 Mini-sequence 的协同效应：Mini-sequence 将 prefill 峰值内存从 O(S×I) 降至 O((S/M)×I)，此时 KV cache（O(S×d×L)）成为剩余内存中的最大占用者。Offloading KV cache 在 prefill 阶段到 CPU 能进一步释放 GPU 内存，使最大 context 从 155K 扩展到 455K（~3×）。Offloading alone（不加 Mini-sequence）收益有限，因为 MLP 中间激活仍是瓶颈。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs
- GTA__Grouped-head_latenT_Attention
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU (UVM-based KV cache offloading with LRU eviction on GPU key bank; offloaded attention made graph-capturable to avoid CPU launch overhead; 3M tokens on L40S 48GB; maintains two separate GPU key banks for mask-selection and BSA processes)
- MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models
- TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization (per-layer 分类：quantization-friendly 浅层保留 GPU 量化 KV，sparsity-friendly 深层 CPU offload + critical-channel 驱动的 Top-K 动态检索；仅 1-3% token 需 PCIe 传输；double buffering 重叠 communication 和 computation)

HATA-off是HATA的KVCache offloading扩展变体：使用PCIe 4.0 + 48 CPU threads，将KV cache存储在CPU memory，通过hash-based top-k retrieval仅传输选中的K/V到GPU。相比MagicPIG（LSH-based offloading），HATA-off通过128-bit learned hash codes替代MagicPIG的1500-bit LSH，在Llama2 (36K context)上实现prefill 6.04×、decode 2.54×加速，在Llama3.1 (72K context)上实现prefill 1.32×、decode 2.63×加速。核心优势：消除MagicPIG昂贵的LSH hashing开销，结合GPU优化的attention kernel和KV prefetching。

---
