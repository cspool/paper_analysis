## Persistent Kernel with CUDAGraph Compatibility

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Persistent Kernel 是一种 GPU kernel 设计模式：kernel 以固定 grid size 启动后持续运行（而非 one-shot launch-complete），各 CTA 通过循环从 work queue 中消费 task items 直到所有 work 完成。在 FlashInfer 中，persistent kernel 用于解决 CUDAGraph 兼容性问题——CUDAGraph 要求所有 kernel launch parameters（grid size、pointers）在 capture 时确定且不变。标准的 dynamic grid size kernel（如根据不同 batch size 调整 grid）无法被 CUDAGraph capture。

FlashInfer persistent kernel 包含两个 merged stages：(1) attention stage——各 CTA 根据 plan info 处理分配的 KV chunks，输出 partial attention states (O_partial, LSE_partial)；(2) contraction stage——各 CTA 用 ⊕ operator compose 多个 partial states 为 final output。两阶段合并入单一 persistent kernel 消除 inter-kernel overhead。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

CUDAGraph-compatible persistent kernel 执行流程：

```
// ===== CUDAGraph Capture Phase =====
// (done once at init time)

// Step 1: 编译 kernel with fixed grid size
grid_size = compute_max_grid_size(device_SM_count, occupancy)
// grid_size 编译后固定，所有后续 generation steps 使用相同值

// Step 2: Allocate workspace buffer with fixed offsets
workspace = torch.empty(total_workspace_size, device='cuda')
partial_O_offset = 0
partial_LSE_offset = partial_O_size
plan_info_offset = partial_LSE_offset + partial_LSE_size
// offsets 固定，所有 generation steps 使用相同 offsets

// Step 3: Dummy plan (生成 sample plan info, 填充 workspace)
attn.plan(dummy_seqlen_info)
// → plan info written to workspace[plan_info_offset:]

// Step 4: Capture CUDAGraph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for layer in layers:
        attn.run(Q[layer], KV_cache[layer], ...)
        // run() 内部: persistent kernel launch
        // grid_size, workspace pointer → both constant → CUDAGraph OK

// ===== Runtime Generation Loop =====
while not finished:
    seqlen_info.update()             // 读取当前 batch sequence lengths
    attn.plan(seqlen_info)           // CPU: 重新计算 plan info → 写入 workspace
    g.replay()                       // GPU: 重放 CUDAGraph (persistent kernel 执行)

// ===== Persistent Kernel 内部 =====
__global__ void persistent_attention_kernel(
    Q, KV_cache, workspace,  // pointers to fixed offsets
    ...
) {
    // CTA 从 workspace 中读取自己的 work queue
    cta_id = blockIdx.x;
    num_chunks = workspace.plan_info[cta_id].num_chunks;
    
    // Persistent loop: 处理所有分配的 chunks
    for (chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        chunk = workspace.plan_info[cta_id].chunks[chunk_idx];
        
        // Attention computation for this chunk
        O_partial, LSE_partial = compute_attention_chunk(
            Q, KV_cache, chunk.query_range, chunk.kv_range);
        
        // Write partial output to fixed-offset workspace region
        workspace.partial_O[cta_id][chunk_idx] = O_partial;
        workspace.partial_LSE[cta_id][chunk_idx] = LSE_partial;
    }
    
    // Contraction: merge partial states via ⊕ operator
    O_final = 0; LSE_final = -inf;
    for (chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        (O_final, LSE_final) = (O_final, LSE_final) ⊕ 
            (workspace.partial_O[cta_id][chunk_idx],
             workspace.partial_LSE[cta_id][chunk_idx]);
    }
    
    // Write final output (CUDAGraph captures this pointer)
    output[cta_output_range] = O_final;
}
```

CUDAGraph 兼容性的关键约束与 FlashInfer 的解决方案：
| CUDAGraph 约束 | FlashInfer 方案 |
|---|---|
| Grid size 必须固定 | Persistent kernel: 固定 grid size = max occupancy, 循环消费 work queue |
| Kernel arguments (pointers) 必须固定 | Workspace buffer 分配 fixed offsets; partial O, plan info 区域用 absolute offsets |
| 不能有 dynamic memory allocation | 所有内存预分配在 workspace buffer 中 |
| 不能有 CPU-GPU sync | Plan function 在 CUDAGraph capture 外执行; kernel 内仅 device-side 操作 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer persistent kernel 实现：
- Inspector-Executor (IE) model：plan phase (CPU inspector) 分析 workload → execute phase (GPU executor) persistent kernel 按 plan 执行。这是并行计算中处理 irregular workload 的经典模式 (Mirchandaney et al., 1988)。
- CUTLASS persistent kernel 参考实现：CUTLASS 3.x 使用 `cutlass::PersistentKernel` 包装，FlashInfer 类似设计但加入了 attention-specific plan info passing。
- 合并 attention + contraction 为单一 persistent kernel 消除 kernel launch overhead between stages（H100 上每个 kernel launch ~5-10 μs，合并后节省此开销 per layer per step）。
- Plan info 可跨层复用：同一 generation step 内所有 decode attention layers 的 sequence lengths 相同 → 同一 plan info 可复用 → plan overhead 被所有 layers 摊销（all layers × multi-step decoding）。

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
