## Split-K Scheme (and its avoidance)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Split-K Scheme是一种GPU kernel内部warp级工作划分策略：将K矩阵沿某个维度拆分（split）到不同warp，每个warp计算其K分片对应的部分结果，最后通过shared memory通信将各部分结果累加（reduce）得到完整输出。在FlashAttention v1的forward pass中采用split-K：K和V被拆分到4个warp，Q对所有warp可见。每个warp独立计算`S_warp = Q @ K_warp^T`得到partial QK^T，然后需要将partial softmax结果写入shared memory，所有warp同步后累加partial outputs。FlashAttention-2识别出这是效率瓶颈——warp间shared memory通信和同步开销大——改为"avoid split-K"：**split Q across warps**，K和V对所有warp共享。每个warp计算其Q slice的完整output，无需warp间通信。这一改变消除了shared memory的read/write和barrier synchronization开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Split-K (FlashAttention v1) 伪代码（4 warps per thread block）：**
```
// Thread block处理一个attention head的forward
// Q: [B_r, d] 所有warp可见（shared memory）
// K, V分4片：K[0:3]各[B_c/4, d], V[0:3]各[B_c/4, d]

// 每个warp (w=0..3):
S_w = Q @ K[w].T              // [B_r, B_c/4] matmul
P_w = softmax_local(S_w, m, ℓ) // 局部softmax（需要全局max/ℓ信息）
// 每个warp将P_w和partial O_w写入shared memory
shared_P[w] = P_w
shared_O_partial[w] = P_w @ V[w]  // [B_r, d]
__syncthreads()                // barrier! 等待所有warp写完

// 只有warp 0做reduce（或其他warp协作reduce）：
O = shared_O_partial[0] + shared_O_partial[1] + shared_O_partial[2] + shared_O_partial[3]
// 额外的shared memory读写！
```

**Avoid Split-K (FlashAttention-2) 伪代码（4 warps per thread block）：**
```
// Thread block处理一个row block的forward
// K: [B_c, d], V: [B_c, d] 所有warp可见（shared memory）
// Q分4片：Q[0:3]各[B_r/4, d]（registers per warp）

// 每个warp (w=0..3) 独立执行：
S_w = Q[w] @ K.T              // [B_r/4, B_c] matmul
m_w_new = max(m_w, rowmax(S_w))
P_w = exp(S_w - m_w_new)       // [B_r/4, B_c]
ℓ_w_new = exp(m_w - m_w_new)*ℓ_w + rowsum(P_w)
O_w = diag(exp(m_w - m_w_new))*O_w + P_w @ V  // [B_r/4, d]
// 无需warp间通信！每个warp独立产出其output slice
// m_w, ℓ_w, O_w在寄存器中，P_w可驻留寄存器
```

关键差异：split-K每次迭代需要O(shared_memory_writes) + barrier + O(reduce)，而avoid split-K无需任何warp间通信。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashAttention-2的CUDA kernel实现中，通过CUTLASS 3.x的TiledMMA和thread block tile迭代器配置warp-level partitioning。在kernel launch配置中，每个thread block使用4或8个warp（128或256 threads）。Q的行在warp间均匀分配——warp 0处理rows 0-31, warp 1处理rows 32-63, etc. (当B_r=128, 4 warps)。K和V通过shared memory对所有warp可见（使用`__shared__`声明或CUTLASS的SharedStorage）。这种方案在A100上使FlashAttention-2 forward达到~210 TFLOPs/s（vs FlashAttention v1 split-K的~105 TFLOPs/s），提升约2×。

涉及论文标题：
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning
