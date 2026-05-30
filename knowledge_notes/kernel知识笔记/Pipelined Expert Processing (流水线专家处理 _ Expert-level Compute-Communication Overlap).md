## Pipelined Expert Processing (流水线专家处理 / Expert-level Compute-Communication Overlap)

术语是什么？
Pipelined Expert Processing 是 ES-MoE 提出的 MoE 训练优化技术，在 expert-level 粒度上重叠 GPU 计算与 CPU↔GPU 数据传输（expert 参数上传/下载）。当 training iteration 经过 gating network 后，tokens 需要在 GPU 之间交换（token permutation / all-to-all），ES-MoE 利用这个通信窗口异步启动第一个 expert 的 CPU→GPU 上传。后续 experts 顺序处理时，当前 expert 的 GPU kernel（FFN forward/backward）与下一个 expert 的 DMA 传输并行——形成 compute ↔ upload 的细粒度流水线。与传统的 layer-wise pipelining（等整个 layer 所有 experts 完成）不同，ES-MoE 的 expert-level pipelining 在单个 expert 完成后即触发后续操作。

从kernel调度角度拆解：
Pipelined Expert Processing 在一个 MoE layer forward pass 中的调度时序：

```python
# MoE Layer Forward Pass with Pipelined Expert Processing
# K=4 GPUs, num_experts=16 (4 experts per GPU after placement)

def moe_layer_pipelined_forward(tokens, gating_network):
    # Step 1: Gating (GPU kernel)
    gate_output = gating_network(tokens)
    expert_ids = argmax(gate_output)  # top-1
    
    # Step 2: Dynamic Placement (CPU, <2.69us)
    gpu_assignments = greedy_placement(expert_ids)
    # e.g., GPU_0 gets [E3, E7, E12, E1]
    
    # Step 3: Overlapped Permutation + 1st Expert Upload
    # Stream 0: All-to-All token exchange (NVLink, ~few ms)
    # Stream 1: async memcpy E3 (CPU→GPU, PCIe, ~few ms)
    # Both execute concurrently
    cuda_stream_0 = all_to_all_scatter(tokens, gpu_assignments)
    cuda_stream_1 = async_upload_expert(E3_params, CPU→GPU)
    synchronize_both_streams()
    
    # Step 4: PIPELINED Expert Processing
    experts_on_this_gpu = [E3, E7, E12, E1]  # ordered by placement
    for i, expert_e in enumerate(experts_on_this_gpu):
        # Compute expert_e on GPU
        output[expert_e] = expert_e.forward(input_tokens_e)
        # Meanwhile, upload next expert (if not last)
        if i+1 < len(experts_on_this_gpu):
            next_expert = experts_on_this_gpu[i+1]
            async_upload_expert(next_expert, CPU→GPU, stream=upload_stream)
            # upload overlaps with compute of current expert
    
    # Step 5: All-to-All Gather + Weighted Sum
    return combine_expert_outputs_inverse_alltoall(output, gate_output)
```

关键时序约束：expert upload time (TU) 必须 ≤ expert compute time (TC) 才能实现完美 overlap。当 expert 的 token count 很小时，TC 可能 < TU，导致 GPU stall（expert 等待下载完成）。ES-MoE 通过 expert pinning（固定 25% 最热门 experts 在 GPU 上）和 adaptive offloading 来缓解此问题。

术语一般如何实现？如何使用？
- 实现依赖 CUDA streams（计算流 + 拷贝流分离）和 `cudaMemcpyAsync`（非阻塞 DMA）
- 在 PyTorch 中通过 `torch.cuda.Stream` 管理独立的计算和通信流，使用 `stream.record_event()` + `event.wait()` 同步
- Expert 参数需在 CPU 端预加载到 pinned (DMA-able) memory 以实现最高 PCIe 传输带宽
- 对于 backward pass，对称的 pipeline 将 gradient download (GPU→CPU) 与下一 expert 的 backward kernel 重叠

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
