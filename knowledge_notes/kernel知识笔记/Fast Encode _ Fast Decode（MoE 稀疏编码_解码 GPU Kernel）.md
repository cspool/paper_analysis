## Fast Encode / Fast Decode（MoE 稀疏编码/解码 GPU Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fast Encode 和 Fast Decode 是 TUTEL 为 MoE 层 dispatch（编码）和 combine（解码）阶段设计的 SIMT-efficient 稀疏 GPU kernel。传统 MoE 框架（GShard/Fairseq/DeepSpeed）使用稠密 einsum 实现 encode/decode（Figure 20a），时间复杂度 O(T·E·C_g·D)，包含大量零乘加运算和巨大的中间张量分配。TUTEL 将其替换为基于 warp-level 并行的稀疏实现（Figure 20b），时间复杂度降至 O(T·k·D)，其中 T·k = E·C_g，使得稀疏版本的复杂度仅为稠密版本的 1/T。实现基于三个精心设计的 CUDA kernel：K0（门控处理）、K1（稀疏编码/解码）、K2（布局变换）。

从kernel调度角度拆解：

Fast Encode/Decode 的 kernel 设计和调度流程：

```cuda
// === K0: Gate Processing Kernel (SIMT-efficient) ===
// 每个 warp 处理一个 token，沿 M 维度
__global__ void k0_gate_processing(
    float* gate_probs, int64_t* idxs, float* scores,
    const float* logits, int T, int E, int k) {
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int token_id = warp_id;
  
  if (token_id >= T) return;
  
  // Softmax (warp-level shuffle reduction)
  float max_val = -INFINITY;
  for (int e = threadIdx.x; e < E; e += 32)
    max_val = max(max_val, logits[token_id * E + e]);
  max_val = warpReduceMax(max_val);  // __shfl_xor_sync
  
  float sum = 0.0f;
  for (int e = threadIdx.x; e < E; e += 32) {
    float p = expf(logits[token_id * E + e] - max_val);
    gate_probs[token_id * E + e] = p;
    sum += p;
  }
  sum = warpReduceSum(sum);
  
  // Normalize (warp shuffle)
  for (int e = threadIdx.x; e < E; e += 32)
    gate_probs[token_id * E + e] /= sum;
  
  __syncwarp();
  
  // Top-K selection (warp-level, one lane only)
  if (threadIdx.x == 0) {
    // Simple partial sort for k << E
    topk_select(gate_probs + token_id * E, idxs + token_id * k, 
                scores + token_id * k, E, k);
  }
}

// === K1: Sparse Encode Kernel ===
// 每个 warp 处理一个 token，稀疏写入 dispatch_input
__global__ void k1_sparse_encode(
    float* dispatch_input,      // (E, C_g, M) output
    int* expert_counters,        // per-expert counter for capacity tracking
    const float* moe_input,      // (T, M) input features
    const int64_t* idxs,         // (T, k) selected expert indices
    const float* scores,         // (T, k) gating scores
    const int* locations,        // (T, k) 1D location within expert's capacity slot
    int T, int k, int C_g, int M, int E) {
  
  int token_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  if (token_id >= T) return;
  
  for (int ki = 0; ki < k; ki++) {
    int expert = idxs[token_id * k + ki];
    int loc = locations[token_id * k + ki];
    float score = scores[token_id * k + ki];
    
    // Sparse scatter: 仅 top-k 专家非零写入
    for (int m = lane_id; m < M; m += 32) {
      float val = score * moe_input[token_id * M + m];
      dispatch_input[(expert * C_g + loc) * M + m] = val;
    }
  }
}

// === K2: Layout Transform (Flexible A2A) ===
// 将 (E, C_g, D) → (E_g, C, D)，消除对 world_size 的依赖
__global__ void k2_flexible_layout(
    float* output, const float* input,
    int E, int C_g, int C, int W, int D, int E_g) {
  // Stride-copy with index remapping
  // input[e][cg][d] → output[eg][c=eg*C_g+?][d]
  // Inline, no intermediate buffer
}
```

关键优化技术：(1) Warp Shuffling——使用 `__shfl_xor_sync` 和 `__shfl_down_sync` 实现 warp 内 reduction，避免 shared memory 开销；(2) Blelloch Scan——用于 prefix-sum 计算每个 expert 的 capacity slot 偏移；(3) Half2 向量化——利用 `half2` 类型同时处理两个 half-precision 元素，double 内存带宽利用率；(4) 无额外数据拷贝——所有 reshape 操作 inline 完成。

术语一般如何实现？如何使用？

TUTEL 通过 CUDA C++ 实现 K0/K1/K2，在 PyTorch 中通过 `torch.autograd.Function` 封装为可微操作：`tutel.fast_encode(input, idxs, scores, capacity)` 和 `tutel.fast_decode(combined_output, idxs, scores)`。前向/反向均使用自定义 CUDA kernel（Figure 21）。与 Fairseq/DeepSpeed MoE 的 einsum dense 实现相比，GPU 内存节省 20%~90%（Table 5），kernel 时间大幅缩短（Figure 15）。

涉及论文标题：
- Tutel Adaptive Mixture-of-Experts at Scale
