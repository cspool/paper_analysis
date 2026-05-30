## Pipelined Expert Processing（流水线专家处理）

术语是什么？
Pipelined Expert Processing 是 ES-MoE (ICML '24) 提出的 expert-level 计算与通信重叠流水线技术。在 MoE training 中，expert 参数 offload 到 CPU 后需要上传到 GPU 才能计算。传统 layer-wise pipeline 在每个 MoE layer 开始前等待所有 experts 上传完毕，导致 GPU 空转。Pipelined Expert Processing 将 pipeline 粒度从 layer 细化到 individual expert：token permutation 阶段与首个 expert 上传重叠（permutation ~0.05ms 足够上传一个 expert），后续 experts 串行处理时并发上传与计算（expert_N computation || expert_{N+1} upload via PCIe）。

从kernel调度角度拆解术语：
```
# ES-MoE Pipelined Expert Processing per MoE Block (Forward)
# 配置：k GPUs, n experts per layer, PCIe bandwidth B

Gate: x → W_gate·x → softmax → Top-1 expert index per token [GPU]
                      |
Dynamic Expert Placement: greedy schedule expert→GPU mapping [CPU, <2.69us]
                      |
Token Permutation: All-to-All scatter tokens to target GPUs [GPU]
  || (overlapped)
  Expert_0 Upload: cudaMemcpy(W_expert_0, CPU→GPU, stream=copy) [PCIe]
                      |
# Expert Processing Loop (per GPU)
for i in 0..num_local_experts:
  if i > 0:
    Expert_i Upload: cudaMemcpy(W_expert_i, CPU→GPU, stream=copy_i)
  Expert_{i-1} FFN: gate_proj → SiLU⊙up_proj → down_proj  [GPU, stream=compute]
  # Expert_{i-1} computation || Expert_i upload — fully overlapped
  
Token Un-permutation: All-to-All gather expert outputs [GPU]
```

术语一般如何实现？如何使用？
ES-MoE 在 Fairseq 框架上实现。使用 PyTorch CUDA stream 管理：独立的 copy stream 和 compute stream。关键参数：PCIe 4.0 bandwidth ~25 GB/s per direction；expert size 85-170 MB per expert；overlap 效果取决于 per-expert compute time vs upload time。MoE-M 32 experts 下，pipelined processing 使 GPU utilization 从 32% 提升至 39%。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
