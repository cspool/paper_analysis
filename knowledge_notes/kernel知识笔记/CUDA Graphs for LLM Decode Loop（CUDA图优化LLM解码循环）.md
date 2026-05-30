## CUDA Graphs for LLM Decode Loop（CUDA图优化LLM解码循环）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

CUDA Graphs 是 NVIDIA 提供的 GPU kernel 执行优化机制，将一系列 CUDA kernel launches 预录制为一张"图"，后续通过单次 graph launch 执行整个图，消除 CPU-GPU kernel launch overhead。在 LLM 推理的 decode 阶段，每个 decode step 执行相同的算子序列（QKV projection → attention → FFN → LM head），只是输入数据变化。通过 CUDA Graphs 录制一次 decode step 的计算图，后续 step 仅需更新输入 buffers + graph replay，将数千次独立 kernel launch 减少为一次。

在 MagicDec 论文中，CUDA Graphs 被用于 self-implemented backend 消除 SD decode 循环中的 kernel launch overhead。对于 speculative decoding 场景，draft phase 和 verify phase 各需录制独立的 CUDA graph（因为它们计算模式不同：draft 用压缩 KV + 逐 token 生成，verify 用完整 KV + 并行验证 γ+1 个位置）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
# CUDA Graph 在 MagicDec SD 中的使用

# Step 1: Graph Capture（只执行一次）
graph_draft = cuda.graph()            # draft phase graph
graph_verify = cuda.graph()           # verify phase graph

# 录制 draft graph（γ 次迭代的循环展开）
with cuda.graph_capture(graph_draft):
    for i in range(gamma):
        # 所有 kernel calls 被录制而非真正执行
        q = linear_q(token)            # cuBLAS gemm kernel
        k, v = linear_kv(token)        # cuBLAS gemm kernel  
        o = flashinfer_attention(q, k_sparse, v_sparse)  # custom kernel
        h = layernorm(o)
        h = silu_gate_mlp(h)           # fused MLP kernel
        token = lm_head(h)             # gemm + argmax
        kv_sparse_append(k, v)         # memory copy kernel

# 录制 verify graph（γ+1 个位置的并行验证）
with cuda.graph_capture(graph_verify):
    q_all = linear_q(all_tokens)       # batched gemm
    # ... 完整 KV attention + FFN + LM head

# Step 2: Graph Replay（每个 decode loop iteration）
for iter in range(max_decode_steps):
    # 更新输入 buffers（无需重新录制 graph）
    cuda.memcpy(graph_draft_inputs, new_data)
    
    # 单次 graph launch 执行整个 draft phase
    graph_draft.replay()
    
    # 单次 graph launch 执行整个 verify phase
    graph_verify.replay()
    
    # 更新输出
    cuda.memcpy(outputs, graph_verify_outputs)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PyTorch CUDA Graphs 实现：`torch.cuda.CUDAGraph`。使用方式：(1) 静态 shape warmup → `g = torch.cuda.CUDAGraph()` → `with torch.cuda.graph(g): output = model(input)` → `g.replay()` 重复执行。MagicDec 在 self-implemented backend 中对 draft 和 verify 各使用独立的 CUDA graph。关键限制：输入 tensor shapes 必须固定（要求 batch size 和 sequence length 不变），因此 CUDA graphs 更适合同质 batch（homogeneous batch）场景——这也是 MagicDec 关注同质 batch 的原因之一。torch.compile 可与 CUDA graphs 协同（torch.compile 编译模型为融合 kernel，CUDA graphs 消除 launch overhead）。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
