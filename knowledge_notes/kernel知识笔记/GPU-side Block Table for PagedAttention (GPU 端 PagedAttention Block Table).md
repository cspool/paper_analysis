## GPU-side Block Table for PagedAttention (GPU 端 PagedAttention Block Table)

术语解释
BrownoutServe 对 vLLM 的 PagedAttention 实现进行了优化：将 KV cache 的 block table 从 CPU 移至 GPU 显存，block table 的查询/映射/更新操作实现为 GPU kernel，消除 CPU→GPU 的数据传输延迟。

术语是什么？
PagedAttention（Kwon et al. 2023, vLLM）将 KV cache 管理类比操作系统分页：逻辑 KV cache 被切分为固定大小的 "block"（如 16 tokens/block），通过 block table 将逻辑 block index 映射到物理 GPU memory 中的实际存储位置。原有实现中，block table 存储在 CPU 端，每次 attention kernel 执行前需将 block table 从 CPU 传输到 GPU。

BrownoutServe 的优化：将 block table 直接分配在 GPU 显存中（torch tensor on device），block table 操作（logical→physical 映射查询、新 block 分配、block 驱逐）作为 GPU kernel 执行，消除 CPU-GPU 同步点。

从kernel调度角度拆解术语：
```
// vLLM original: CPU-side block table
// 每 iteration:
//   CPU: block_table[req_id] = [physical_block_0, physical_block_1, ...]
//   CPU → GPU: memcpy(block_table)                    ← 额外数据传输
//   GPU: launch attention_kernel(block_table, Q, K, V)
//   GPU: attention = flash_attn(Q, K_from_blocks, V_from_blocks)

// BrownoutServe optimization: GPU-side block table
// 初始化: block_table = torch.zeros(..., device='cuda')  ← 分配在 GPU
// 每 iteration:
//   GPU: launch block_table_update_kernel(block_table)    ← 无需 CPU 参与
//   GPU: launch attention_kernel(block_table, Q, K, V)   ← 同一 GPU memory
//   GPU: flash_attn 直接读取 GPU block_table

// GPU-side block table lookup kernel (简化)
@triton.jit
def block_table_lookup_kernel(q_ptr, k_cache_ptr, v_cache_ptr,
                               block_table_ptr, output_ptr,
                               SEQ_LEN, BLOCK_SIZE, HEAD_DIM):
    pid = tl.program_id(0)  # query position index
    if pid >= SEQ_LEN: return
    
    q = tl.load(q_ptr + pid * HEAD_DIM + ...)
    
    # GPU-side block table lookup
    block_id = pid // BLOCK_SIZE
    offset = pid % BLOCK_SIZE
    
    # 直接从 GPU memory 读 block table
    physical_block = tl.load(block_table_ptr + block_id)
    
    # 读取 KV cache
    k = tl.load(k_cache_ptr + physical_block * BLOCK_SIZE * HEAD_DIM + offset * HEAD_DIM + ...)
    v = tl.load(v_cache_ptr + physical_block * BLOCK_SIZE * HEAD_DIM + offset * HEAD_DIM + ...)
    
    # Attention compute
    attn_score = tl.sum(q * k) / tl.sqrt(HEAD_DIM)
    out = attn_score * v
    tl.store(output_ptr + pid * HEAD_DIM + ..., out)
```

术语一般如何实现？如何使用？
- 实现依赖：PyTorch tensor on CUDA、Triton kernel、或手写 CUDA kernel
- 关键收益：(1) 消除 CPU→GPU block_table 传输延迟；(2) 减少 kernel launch 次数（block table 更新和 attention 可在同一 kernel 或同一 stream 上执行）；(3) 与 GPU continuous batching 的 scheduling 更紧密集成
- 注意事项：GPU 显存中 block table 占用空间极小（每个 block 仅需一个 int32 index），不影响模型可用的显存空间

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs
