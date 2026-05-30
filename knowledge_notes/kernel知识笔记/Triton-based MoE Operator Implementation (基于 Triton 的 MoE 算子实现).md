## Triton-based MoE Operator Implementation (基于 Triton 的 MoE 算子实现)

术语解释
BrownoutServe 使用 Triton 语言（而非 C++/CUDA）重写 MoE 模块的所有算子，包括 expert FFN 计算、token dispatch/combine、united expert 调用。Triton 在 Python 层面编写 tile-level 计算逻辑，编译为高效 GPU kernel，与 PyTorch 无缝兼容。

术语是什么？
Triton（Tillet et al. 2019）是一个 Python-based 的 GPU 编程语言和编译器。开发者用 Python 语法描述 tile-level 并行计算，Triton 编译器自动生成优化的 GPU kernel（PTX/CUDA），处理 shared memory 管理、thread block 调度、memory coalescing 等底层优化。相比手写 CUDA C++，Triton 开发效率更高且性能接近手写优化水平。

BrownoutServe 选择 Triton 的原因：(1) 与 PyTorch 无缝兼容，简化 BrownoutMoE 与 PyTorch 模型代码的集成；(2) 自动处理 GPU-specific 优化，降低开发维护成本；(3) MoE 的 token dispatch/combine 操作涉及大量 gather/scatter，Triton 的编程模型更友好。

从kernel调度角度拆解术语：
Triton-based BrownoutMoE kernel 执行流程：

```
// 1. Gate Kernel (Triton)
// grid: (num_tokens,), block: expert computation per token
@triton.jit
def gate_kernel(x_ptr, centroid_ptr, scores_ptr, topk_idx_ptr, topk_val_ptr,
                N_TOKENS, N_EXPERTS, HIDDEN_DIM, TOPK):
    pid = tl.program_id(0)  # token index
    if pid >= N_TOKENS: return
    
    # Load token hidden state
    x = tl.load(x_ptr + pid * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM))
    
    # Compute affinity with all experts (x^T · e_i)
    scores = tl.zeros([N_EXPERTS], dtype=tl.float32)
    for e in range(N_EXPERTS):
        centroid = tl.load(centroid_ptr + e * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM))
        scores[e] = tl.sum(x * centroid)  # dot product
    
    # Softmax + Top-K
    topk_vals, topk_idxs = tl.topk(tl.softmax(scores), TOPK)
    tl.store(topk_idx_ptr + pid * TOPK + tl.arange(0, TOPK), topk_idxs)
    tl.store(topk_val_ptr + pid * TOPK + tl.arange(0, TOPK), topk_vals)

// 2. Fused MoE FFN Kernel (Triton) - S1 tokens
// grid: (num_experts_in_S1,), block: per-expert batch computation
@triton.jit
def fused_moe_kernel(tokens_ptr, weights_ptr, outputs_ptr,
                     expert_offsets, N_EXPERTS_S1, HIDDEN, INTERMEDIATE):
    eid = tl.program_id(0)  # expert index
    if eid >= N_EXPERTS_S1: return
    
    # Load expert's token range
    start = tl.load(expert_offsets + eid)
    end = tl.load(expert_offsets + eid + 1)
    n_tokens = end - start
    
    # Gather tokens → 单个大 batch
    tokens = tl.load(tokens_ptr + start * HIDDEN + ...)  # [n_tokens, HIDDEN]
    
    # FFN: gate_proj → silu → up_proj → down_proj
    w_gate = tl.load(weights_ptr + eid * W_SIZE + ...)   # [HIDDEN, INTERMEDIATE]
    w_up = tl.load(weights_ptr + eid * W_SIZE + ...)     # [HIDDEN, INTERMEDIATE]
    w_down = tl.load(weights_ptr + eid * W_SIZE + ...)   # [INTERMEDIATE, HIDDEN]
    
    # GEMM tiles
    gate_out = tl.dot(tokens, w_gate)  # [n_tokens, INTERMEDIATE]
    up_out = tl.dot(tokens, w_up)      # [n_tokens, INTERMEDIATE]
    act_out = tl.silu(gate_out) * up_out
    out = tl.dot(act_out, w_down)       # [n_tokens, HIDDEN]
    
    # Scatter results back
    tl.store(outputs_ptr + start * HIDDEN + ..., out)

// 3. United Expert Kernel (Triton) - S2 tokens
// grid: (num_groups,), block: per-group concat + FFN
@triton.jit
def united_expert_kernel(tokens_ptr, ue_weights_ptr, outputs_ptr,
                         group_offsets, N_GROUPS, HIDDEN, INTERMEDIATE):
    gid = tl.program_id(0)  # group index
    if gid >= N_GROUPS: return
    
    # Concat all tokens from this group's experts
    start = tl.load(group_offsets + gid)
    end = tl.load(group_offsets + gid + 1)
    batch_size = end - start
    
    if batch_size <= 0: return  # empty group
    
    concat_tokens = tl.load(tokens_ptr + start * HIDDEN + ...)  # [batch_size, HIDDEN]
    
    # United Expert FFN (same structure as original expert)
    ue_w_gate = tl.load(ue_weights_ptr + gid * UE_W_SIZE + ...)
    ue_w_up = tl.load(ue_weights_ptr + gid * UE_W_SIZE + ...)
    ue_w_down = tl.load(ue_weights_ptr + gid * UE_W_SIZE + ...)
    
    gate_out = tl.dot(concat_tokens, ue_w_gate)
    up_out = tl.dot(concat_tokens, ue_w_up)
    act_out = tl.silu(gate_out) * up_out
    out = tl.dot(act_out, ue_w_down)
    
    tl.store(outputs_ptr + start * HIDDEN + ..., out)
```

术语一般如何实现？如何使用？
- 开发流程：用 `@triton.jit` 装饰器标注 kernel 函数 → 定义 grid（thread block 分布）→ Python 调用 → Triton JIT 编译为 GPU code
- 与 CUDA 的关系：Triton 生成 PTX 中间表示，经 PTX→SASS 编译在 GPU 上执行；性能通常达到手写 CUDA 的 80-95%
- BrownoutServe 使用 Triton 而非 CUDA 的原因：简化 brownout routing（大量条件分支）的实现，与 PyTorch 无缝集成

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs
