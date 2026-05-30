## X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：X-MoE 实现了一套基于 Triton 的跨平台 sparse/irregular kernel，用于支持 PFT padding-free MoE pipeline 在 AMD MI250X GPU 上的高效执行：
    (1) **Triton Gather Kernel**：将 gating 输出 gate_out [S, H] 按 token_ids [B] 索引 gather 到 dispatch_in [B, H]。执行 `dispatch_in[i,:] = gate_out[token_ids[i],:]`。每个 token 分配一个 thread-block（256 threads），block bi 负责复制 gate_out[token_ids[bi],:] → dispatch_in[bi,:]，沿 hidden dimension 循环 H/256 次。通过连续线程处理 hidden dimension 的连续内存位置，确保即使有 `gate_out[token_ids[i],:]` 的不规则索引，内存请求仍是 coalesced。
    (2) **Triton Scatter Kernel**：将 MLP 输出 mlp_out [B, H] 按 token_ids 反向 scatter 回原始序列位置，同时乘以 combine_weights。执行 `combine_in[token_ids[i],:] = mlp_out[i,:] * combine_weights[i]`。不规则写访问通过连续线程沿 hidden dimension 写入保证 coalescing。
    (3) **Sequential GeMM**（非 Triton，Python for-loop 驱动）：在 dispatch_out [Bexp, H] 上，按 tokens_per_expert 数组切片，依次为每个 expert 启动一个标准 GeMM。第 i 个 expert 处理 tokens `dispatch_out[sum(tpi[:i]):sum(tpi[:i+1])]`，共 Elocal 次 GeMM launch。替代传统 padded batched matmul，避免 zero-padding 带来的无效计算。
  - 实验比较：(a) MoE layer forward time breakdown：X-MoE vs DeepSpeed-MoE，在 Small 模型 (EP=8) 和 Large 模型 (EP=64) 上，gating/buffer dispatch/dispatch alltoall/expert compute/combine alltoall/buffer combine 各阶段延迟对比；(b) 激活内存消耗：X-MoE vs DeepSpeed-MoE vs Tutel 在 Large 模型 256 GPU 下的 per-MoE-layer 内存 (GB)；(c) RBD 的 dispatch time breakdown：with/without RBD 的 inter-node alltoall + intra-node alltoall + data transform 时间；(d) Cross-platform：8×NVIDIA A100 上 X-MoE vs DeepSpeed-MoE vs Tutel 的 TFLOPs 和 OOM 情况。

- 后端平台是什么，配置是什么。
  - 主平台：AMD MI250X GPU（Frontier 超级计算机），每 GPU 2 GCD（视为独立 GPU），峰值 191.5 TFLOPs/effective-GPU，Infinity Fabric intra-node（50-200 GB/s），Slingshot 25 GB/s inter-node。
  - 软件栈：ROCm 5.7.1，PyTorch 2.2.0，DeepSpeed 0.15.5，Triton（版本论文未明确说明），AWS-OFI-RCCL plugin + libfabric 1.20.1。
  - 跨平台验证：NVIDIA A100 40GB，CUDA 平台。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 DeepSpeed 0.15.5 + DeepSpeed-Megatron 实现。核心修改：
    (a) **PFT 构造与 ERI-arrays 管理**：在 MoE gating 后实现 PFT construction 例程（flatten + sort top_experts + one-hot + cumsum + token dropping + histogram），替换传统 dispatch_mask 生成。
    (b) **Triton Gather/Scatter Kernel**：替换 einsum-based dispatch，实现 coalesced memory access 的 Triton kernel（gather 读 coalesced，scatter 写 coalesced，均沿 H 维度连续线程分配）。
    (c) **Sequential GeMM**：替换 batched matmul，按 tokens_per_expert 切片为每个 expert 单独 launch GeMM。
    (d) **RBD dispatch/combine 流程**：实现 pilot selection + s1_mapping_indices 构建 + 两级 alltoall（跨节点 uneven + 节点内 uneven）+ local replica 重建 + merge/reorder。
    (e) **SSMB 序列切分**：在 TP→EP 转换处 drop partial tokens，MoE block 结束后 all-gather 恢复。
  - 开源：https://github.com/Supercomputing-System-AI-Lab/X-MoE，集成于 DeepSpeed。

- 基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 以单次 MoE layer forward pass（在 AMD MI250X，Large 模型，EP=64，256 GPU）为例，X-MoE kernel 执行全过程：
    1. **输入**：gating 输出 gate_out [S, H]（S=sequence_length×batch_size, H=7168），top_experts [S, K]（K=8），combine_weights [S, K]，均为 FP16/BF16。
    2. **PFT Construction**（CPU/GPU 混合）：flatten top_experts [S*K] + argsort by combine_weights + one_hot → cumsum（优化为 [E, S*K] layout 使 cumsum 在 outer dimension coalesced，加速 10×）+ weight_mask 过滤 dropped tokens → 生成 ERI-arrays（token_ids [B], expert_ids [B], tokens_per_expert [E], combine_weights [B]），B ≤ max_token_count × E。
    3. **Gather Kernel**（Triton，GPU）：launch B thread-blocks × 256 threads，block bi 执行 `dispatch_in[bi, :H] = gate_out[token_ids[bi], :H]`，H=7168，每个 thread-block 循环 7168/256=28 次。内存访问 coalesced（连续线程→连续 hidden dim 元素）。输出 dispatch_in [B, H]。
    4. **Dispatch Alltoall**（RCCL + libfabric）：先 alltoall tokens_per_expert [E]（metadata，轻量），后 alltoallv dispatch_in [B, H] → dispatch_out [Bexp, H]。RBD 模式下仅 pilot tokens（去重后）走跨节点 alltoall，local replica 走节点内 alltoall。
    5. **Sequential GeMM**（rocBLAS）：for i in 0..Elocal-1: slice = dispatch_out[offset:offset+tpi[i]]; inter = slice @ w1[i]; out = inter @ w2[i]。每 expert 独立 GeMM，无 zero-padding 计算。
    6. **Scatter Kernel**（Triton，GPU）：launch B thread-blocks，执行 `combine_in[token_ids[i], :H] = mlp_out[i, :H] * combine_weights[i]`。写 access coalesced 沿 hidden dimension。输出 final_output [S, H]。
    7. **评估原理**：Throughput (TFLOPs) = 模型单步总 FLOPs / iteration_time。通过 PyTorch profiler 记录各阶段耗时（gating/dispatch/alltoall/expert/combine），内存通过 `torch.cuda.max_memory_allocated()` 峰值。Scalability 通过 weak scaling（固定 per-GPU batch，增加 GPU 数）和 strong scaling（固定 global batch，增加 GPU 数）评估。
