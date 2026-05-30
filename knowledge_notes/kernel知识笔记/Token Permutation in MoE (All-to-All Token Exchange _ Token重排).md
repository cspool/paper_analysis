## Token Permutation in MoE (All-to-All Token Exchange / Token重排)

术语是什么？
Token Permutation 是 expert parallelism 中 token 在 GPU 间重新分配的操作阶段。由于各 GPU 上的 token 需根据 gating network 的路由决策被发送到持有对应 expert 的 GPU 进行计算，token permutation 通过 all-to-all 通信将 token 从"按原始 batch 排列"重排为"按目标 expert 排列"。在 ES-MoE 中，token permutation 阶段被利用为 expert 上传的 overlap 窗口——permutation 通信时间（NVLink all-to-all）通常能覆盖 single expert 的 CPU→GPU 上传时间（PCIe）。

从kernel调度角度拆解：
Token Permutation 的两阶段流程：

```python
# Phase 1: All-to-All Scatter (Dispatch)
# 将 tokens 按 expert routing 发送到目标 GPU
def token_dispatch(tokens, expert_ids, gpu_mapping):
    """
    tokens: (B*S, d_model) on local GPU
    expert_ids: (B*S,), which expert each token goes to
    gpu_mapping: Dict[expert_id -> gpu_id]
    Returns: tokens regrouped by destination GPU
    """
    # Build per-GPU token indices
    gpu_buckets = {gpu_id: [] for gpu_id in range(num_gpus)}
    for token_idx, expert_id in enumerate(expert_ids):
        target_gpu = gpu_mapping[expert_id]
        gpu_buckets[target_gpu].append(token_idx)
    
    # All-to-All: each GPU sends its tokens to target GPUs
    # Uses NCCL alltoallv or similar collective
    recv_tokens = all_to_all_scatter(tokens, gpu_buckets)
    return recv_tokens

# Phase 2: All-to-All Gather (Combine)
# 计算完成后，将 expert outputs 送回 token 原始所在 GPU
def token_combine(expert_outputs, token_origins):
    """
    expert_outputs: per-GPU expert computation results
    token_origins: which GPU each token came from (inverse of dispatch)
    """
    recv_outputs = all_to_all_gather(expert_outputs, token_origins)
    return recv_outputs
```

通信量分析：Token permutation 的通信量与 microbatch size × d_model 成正比，与 expert 数量无关。相比之下，expert 参数的上传/下载通信量与 expert 数量 × expert 参数量成正比。因此当 expert 数量很大时，expert 传输成为主要通信开销——这正是 ES-MoE 将 permutation 时间用于 overlap expert upload 的原因。

术语一般如何实现？如何使用？
- **NCCL alltoall**: NVIDIA 集合通信库，利用 NVLink/NVSwitch 高带宽 GPU-GPU 直连
- **Fairseq/Tutel**: 通过组通信（grouped all-to-all）按 expert parallelism group 交换 tokens
- **FasterMoE**: 提出 expert-centric 通信——交换 experts 而非 tokens（适用于大批量 GPU 场景）
- ES-MoE 将 token permutation 与 expert upload 重叠，利用 permutation 的通信时间窗口进行专家传输

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
