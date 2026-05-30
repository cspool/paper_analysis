## Ragged Tensor (Jagged Array)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Ragged Tensor（也称 Jagged Array 或 Variable-Length Tensor）是一种存储变长序列的紧凑数据结构。与 padded tensor（将所有序列 pad 到 max length，浪费存储和计算）不同，ragged tensor 将所有序列的 elements concatenate 为一个 flat 1D tensor（`values`），用额外的 `offsets`（或 `indptr`）数组记录每个序列的起始位置。在 FlashInfer 中，query 和 output 矩阵使用 ragged tensor 存储：不同请求的 tokens 数不同（prefill 阶段 prompt length 可变，decode 阶段各请求开始/结束时间不同），将这些变长 tokens 打包为单个 tensor 消除 padding，提升 memory 和 compute efficiency。

典型 ragged tensor 表示：`values = [token_0_req0, token_1_req0, token_0_req1, token_1_req1, token_2_req1, ...]`，`indptr = [0, 2, 5, ...]`（cumulative lengths）。FlashInfer kernel 通过 `indptr` 定位各请求 boundaries。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashInfer 中 ragged tensor 用于表示可变长度 batch 的 Q / O：

```
// ===== Ragged Tensor 表示 =====
// 3 requests: req_0 (2 tokens), req_1 (3 tokens), req_2 (1 token)
// Padded representation (wasteful):
//   Q_padded = [[q0_0, q0_1, PAD, PAD],     // shape [3, 3, d]
//               [q1_0, q1_1, q1_2, PAD],
//               [q2_0, PAD,  PAD,  PAD]]
// Ragged representation (compact):
//   Q_values = [q0_0, q0_1, q1_0, q1_1, q1_2, q2_0]  // flat, [total_tokens, d]
//   Q_indptr = [0, 2, 5, 6]  // cumulative token counts

// ===== FlashInfer Kernel 使用 ragged tensor =====
__global__ void ragged_attention_kernel(
    float* Q_values,     // [total_tokens, nheads, head_dim]
    int* Q_indptr,       // [batch_size + 1]
    float* KV_cache,     // BSR formatted
    int* kv_indptr,      // BSR row pointers
    int* kv_indices,     // BSR column indices
    float* O_values      // output, same ragged layout as Q
) {
    // CTA 处理某个请求的某个 query tile
    // 通过 Q_indptr 将 flat token index 映射到 request index
    token_start = Q_indptr[request_id];
    token_end = Q_indptr[request_id + 1];
    num_tokens = token_end - token_start;
    
    // 从 Q_values 的 flat layout 中读取该 request 的 Q tile
    q_tile = load_ragged_tile(Q_values, token_start, num_tokens);
    
    // ... attention computation ...
    
    // 写入 O_values 对应位置 (相同 ragged layout)
    store_ragged_tile(O_values, token_start, num_tokens, O_result);
}
```

Ragged tensor 的关键特性：
- **No padding waste**：total memory = Σ actual tokens（而非 batch × max_len padded）
- **Compact compute**：kernel 仅处理实际 tokens（而非 padded zeros），无 wasted FLOPs
- **Indirection overhead**：每次访问需要 `indptr` lookup，但 cost 远低于 padding waste
- FlashInfer 同时支持 ragged Q + BSR KV-cache：Q ragged tensor → per-request boundaries → BSR row mapping → sparse KV-cache access

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Ragged tensor 的实现和使用：
- TensorFlow 原生支持 `tf.RaggedTensor` (2018)
- PyTorch 通过 `torch.nested` 或 `nestedtensor` 支持 nested tensor（PyTorch 1.11+）
- FlashInfer 在 CUDA kernel 层面直接使用 `indptr` + `values` 数组，不依赖高层框架抽象——这是为了 kernel 层面最大性能
- LLM serving 中广泛使用：vLLM、SGLang 等框架在 batch prefill/decode 时自然产生 variable-length Q/O（不同请求到达时间、prompt length 不同）→ ragged tensor 是最优表示

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
