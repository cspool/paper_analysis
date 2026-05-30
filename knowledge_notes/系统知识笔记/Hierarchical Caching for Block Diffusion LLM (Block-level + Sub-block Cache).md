## Hierarchical Caching for Block Diffusion LLM (Block-level + Sub-block Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Caching for Block Diffusion LLM 是Fast-dLLM v2提出的两级缓存机制，用于加速block diffusion模型的推理。包含两个层级：(1) **Block-level KV Cache**：每个已完成解码的block（block size=32）的所有token的K/V被缓存为read-only prefix context。后续block仅需计算自身attention + 对prefix的cross-attention，无需重复计算已解码block的K/V。由于block diffusion的block间使用causal conditioning（已完成block内容不再改变），此cache是精确（exact）而非近似（approximate）的。(2) **Sub-block DualCache**：块内解码时，将当前block进一步划分为sub-block（size=8），使用Fast-dLLM v1的DualCache机制——同时缓存sub-block的prefix（已解码token）和suffix（仍为[MASK]的token）的K/V，仅需计算sub-block内B_sub×B_sub的自注意力。两层缓存协同：block级cache消除跨block的冗余计算，sub-block cache消除块内的冗余计算。在compute-bound regime（如batch size=32）下显著提升throughput。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Hierarchical Caching在block diffusion推理中的运转：

```
序列结构: [Prompt | Block_1 | Block_2 | ... | Block_k (current)]
         |←─── Block-level KV Cache (精确, read-only) ──→|

当前Block k内部（size=32, sub-block size=8）:
  Sub_1 (decoded) | Sub_2 (decoding) | Sub_3..4 ([MASK])
  |←─ DualCache prefix ─→|              |←─ DualCache suffix ─→|

单次forward的计算量:
  无Cache:        attention over all (prompt + all blocks) → O((|p|+K·B)²)
  Block-level:    attention over prefix + current block       → O((|p|+B)²)
  + Sub-block:    attention over prefix + sub-block only      → O((|p|+S)²)
  (S=sub-block size, B=block size, K=block count)
```

Block-level cache在block完成时更新——与block的最终forward融合（无额外开销）。Sub-block DualCache在每个sub-block内复用（无需块完成后才更新）。Cache全部存储在GPU HBM中，通过索引访问复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) Block-level cache存储为Python dict: key=(layer_id, block_range), value=(K_tensor, V_tensor)；(2) Sub-block DualCache继承自Fast-dLLM v1，在PyTorch eager模式实现，通过attention mask控制sub-block可见范围；(3) batch decoding时所有序列同步逐block推进（通过右填充[MASK]对齐block边界）。参数配置：block_size=32（训练和推理固定），sub_block_size=8（推理时可调），threshold=0.9（并行解码阈值，平衡speed-quality）。Sub-block cache是纯效率优化（不影响accuracy, Figure 6a），在compute-bound regime（batch≥32）下效果显著。

涉及论文标题：
- Fast-dLLM v2: Efficient Block-Diffusion LLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashInfer是一个高效且可定制的attention引擎库，专为LLM推理serving设计。提供一组高性能attention kernel（包括FlashAttention变体、各类sparse attention、KV cache操作等），作为底层kernel library被上层inference framework（如vLLM、SGLang、TensorRT-LLM）调用。核心特征：(1) 模块化设计——每个attention variant作为独立kernel实现，可灵活组合；(2) 支持多种attention变体——MHA、GQA、MQA、MLA；(3) 针对现代GPU（Hopper/Blackwell）优化；(4) 提供C++和Python API。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FlashInfer在LLM serving系统中的位置和运转流程：

```
Inference Serving Framework (vLLM/SGLang/TensorRT-LLM)
│
├── Scheduler: 请求调度、batching、KV cache管理
│
├── Model Executor: 执行单个模型层的计算
│   │
│   └── Attention Module ← 调用FlashInfer kernel
│       │
│       ├── Prefill: FlashInfer prefill kernel (dense或sparse)
│       │   - 输入: Q/K/V tensors + attention mask
│       │   - 输出: attention output tensor
│       │
│       ├── Decode: FlashInfer decode kernel (dense或sparse)
│       │   - 输入: Q tensor + KV cache pointer + page table
│       │   - 输出: attention output (single token)
│       │
│       └── KV Cache Append: FlashInfer append kernel
│           - 将新计算的K/V写入page table管理的cache
```

BLASST已作为FlashInfer的一个attention kernel实现，提供prefill和decode两套kernel，用户通过调整threshold λ即可在标准FlashInfer attention API中使用sparse attention。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer使用CUDA C++编写，每个kernel手写优化（非triton自动生成），支持Hopper（sm90a）和Blackwell（sm100）architecture。API示例（Python binding）：
```
import flashinfer
# Prefill
O = flashinfer.prefill_with_paged_kv_cache(Q, KV_cache, page_table, ...)
# Decode
O = flashinfer.decode_with_paged_kv_cache(Q, KV_cache, page_table, ...)
```
BLASST集成后增加threshold参数：
```
O = flashinfer.decode_with_paged_kv_cache(Q, KV_cache, page_table, 
        blasst_threshold=λ, ...)
```

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding
