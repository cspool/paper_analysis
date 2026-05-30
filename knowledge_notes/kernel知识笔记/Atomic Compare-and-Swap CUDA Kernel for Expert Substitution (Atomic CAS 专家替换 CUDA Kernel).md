## Atomic Compare-and-Swap CUDA Kernel for Expert Substitution (Atomic CAS 专家替换 CUDA Kernel)

术语解释
Atomic Compare-and-Swap (CAS) CUDA Kernel for Expert Substitution 是 BuddyMoE 实现的 GPU 并行 buddy substitution kernel，使用 CUDA thread block 和 atomic CAS 操作实现无锁的专家并行替换，在 ~0ms 内完成 per-token multi-expert 的 buddy substitution，不引入 noticeable latency overhead。

术语是什么？
Kernel 配置：grid(T, 1, 1) × block(K, 1, 1)——每个 CUDA thread block 处理一个 token 的 K 个 expert 的替换，block 内每个 thread 负责一个 expert。Shared memory U_t[E] 维护当前 token 的已分配 expert set（初始化为 S 中的 expert indices）。每个 thread 检查自己的 expert e_id 是否 GPU-resident（M[e_id]），若非 resident 则遍历 buddy list B[e_id] 寻找 GPU-resident 且不在 U_t 中的 buddy。Atomic CAS（atomicCAS(&U_t[b_id], false, true)）保证无锁的 uniqueness constraint——第一个成功 claim 该 buddy 的 thread 获得它，其他 thread 必须寻找不同 buddy。

从kernel调度角度拆解术语：
```
__global__ void buddy_substitute_kernel(
    int* S, bool* M, int* B, int T, int K, int E, int H
):
    __shared__ bool Ut[E]  # block-level shared memory per token
    
    # Initialize Ut from current expert indices
    for i in range(K):  # parallel init
        e = S[blockIdx.x * K + i]
        Ut[e] = true
    __syncthreads()
    
    e_id = S[blockIdx.x * K + threadIdx.x]
    if M[e_id] == false:  # CPU-resident, need replacement
        for r in range(H):  # search buddy list up to H
            b_id = B[e_id * B_stride + r]
            if M[b_id] and atomicCAS(&Ut[b_id], false, true):
                S[blockIdx.x * K + threadIdx.x] = b_id
                break
    # No suitable buddy found: leave original e_id (fallback to on-demand)
```

Kernel 调度特征：
- 所有操作在 GPU memory 内完成（查 B table、M mask、atomic CAS），无 CPU↔GPU 传输
- grid(T,1,1) 使不同 token 的 thread blocks 在 SM 间并行
- Shared memory Ut[E] 需 E ≤ 64（DeepSeek-V2-Lite）→ shared memory per block = 64 bytes → 可忽略
- Atomic CAS 在 L2 cache 层面操作，延迟 < 100 GPU cycles，远小于 expert FFN GEMM

术语一般如何实现？如何使用？
- 实现为 CUDA __global__ kernel，在 llama.cpp 的 CUDA backend 中集成
- 在 router 输出后、expert FFN 计算前调用
- 若 no suitable GPU-resident buddy → fallback 到 prefetch original expert（pay transfer）或 skip expert per baseline MoE drop policy
- CAS 操作的竞争仅在同一 token 的多个 thread 同时选中同一 buddy 时发生（低概率：buddy lists ≤ 16 且 expert 总数 ≥ 64）

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
