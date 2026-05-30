## Expert Offloading in MoE Inference (MoE 推理中的 Expert 卸载)

术语解释
Expert Offloading 是指将 MoE 模型中不在 GPU 显存活跃使用的 expert 参数存储到 CPU 主存（或更慢的存储层级），然后在推理需要时通过 PCIe 总线按需传输到 GPU 显存的技术。用于在 GPU 内存受限场景下部署大型 MoE 模型。

术语是什么？
Expert offloading 管理两个内存层级：CPU 主存（存储全部 expert 参数和 KV cache states）和 GPU 显存（用于计算和快速数据访问）。当 GPU 需要的 expert 参数不在 resident store 中时，计算 stall 直到 PCIe 传输完成。典型架构组件：
- **GPU Resident Store**：持久驻留在 GPU 显存中的常访问 expert 和 KV cache
- **GPU Staging Buffer**：用于预取动态数据的 GPU 缓冲区
- **CPU Main Memory**：存储全部 expert 参数（包括非活跃 expert）
- **Prefetching**：提前预测并加载即将需要的 expert，与计算重叠以减少 stall

关键性能约束（Compression Error Sensitivity 论文）：
- Mixtral-8x7B FP16 需 ~94 GB VRAM，仅 ~27.5 GB 被活跃使用，~66.6 GB 被非激活 expert 浪费
- PCIe 4.0 带宽 32 GB/s vs GPU 内部带宽 ~300 GB/s —— bottleneck 从 memory-bound 转向 I/O-bound
- 压缩是缓解 offloading I/O 瓶颈的关键：压缩非激活 expert 以减少 PCIe 传输数据量

从系统架构角度拆解术语：
MoE inference with expert offloading 的请求处理流程：
```
输入: batch of tokens
for each MoE layer l:
    # Step 1: Router 计算
    router_weights, expert_ids = Router(token_embeddings)
    
    # Step 2: Expert 加载决策
    for each expert_id in expert_ids:
        if expert_id in GPU_Resident_Store:
            # 命中，直接使用
            expert_weights = GPU_Resident_Store[expert_id]
        else:
            # 未命中，从 CPU 主存加载
            if expert_id in Prefetch_Buffer:
                expert_weights = Prefetch_Buffer[expert_id]
            else:
                # Stall: 等待 PCIe 传输（主要延迟来源）
                expert_weights = PCIe_Transfer(CPU_Memory[expert_id])
                # 可选：压缩传输以减少延迟
                # expert_weights = Decompress(PCIe_Transfer(Compress(CPU_Memory[expert_id])))
    
    # Step 3: Expert FFN 计算
    output = Σ router_weights[i] * Expert_FFN(token, expert_weights_i)
    
    # Step 4: Prefetch 下一层可能需要的 expert（与当前层计算重叠）
    Prefetch_Buffer = Predict_Next_Experts(layer l+1, current_tokens)
```

术语一般如何实现？如何使用？
- 代表性系统：MoE-Infinity（offloading-efficient MoE serving）、Pre-gated MoE（algorithm-system co-design）、SwapMoE（tunable memory budget）、HOBBIT（mixed precision offloading）、FloE（on-the-fly compression offloading）、Klotski（expert-aware multi-batch pipeline）、ExpertFlow（predictive expert caching + token scheduling + routing path predictor）
- 优化方向：
  - (1) Prefetching：预测 expert 激活模式，提前加载
  - (2) Compression：量化（INT4/INT2）或 error-bounded lossy compression 减少传输量
  - (3) Caching：GPU resident store 内存放最热 expert（~15-20% expert 处理 ~80% token）
  - (4) Pipelining：压缩/解压/传输/计算 overlap
  - (5) Hybrid execution：冷 expert 直接在 CPU 执行，避免传输
- 论文（Compression Error Sensitivity Analysis）从压缩误差敏感性角度，为 offloading 中的压缩策略提供了分层指导：浅层 expert 可承受较大 error bound（高压缩比），中层 expert 需保守压缩（小 error bound），深层 expert 可利用噪声增益（中等 error bound）

涉及论文标题：
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference
- ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference
- eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

**eMoE 的 Expert Offloading 方法**：
eMoE 提出了一种与 Per-prompt Prefetching（Pre-gatedMoE/MoEInfinity）不同的 offloading 策略：**周期性预测 + 复用**。核心差异：
1. **Expert Prediction**：使用 BERT-XLNet（0.108B）预测 future expert 序列（而非基于当前 layer 的 gate 输出预取下一层）。预测基于两个 correlation：(a) consecutive layer 间 expert 选择的 cross-correlation 约 0.50，(b) consecutive prompt 间 expert 选择的 cross-correlation 0.75-0.95（Mixtral）或 0.4-0.6（OpenMoE）。
2. **Periodic Invocation**：每 p=40 prompts 调用一次 predictor 并加载新 expert，中间 prompts 复用已加载 expert。Experiment 显示 perplexity 在 ≤60 prompts 复用下基本不变。
3. **Conditioned Loading**：当前 MoE 层的 expert 加载以前一层加载完成为条件（CUDA event sync），防止多路并发 DMA 饱和 PCIe 带宽。
4. **Async Transfer**：`torch.Tensor.copy_(non_blocking=True)` 使 expert 加载与 non-expert layer（self-attention）计算重叠。
5. **Task-aware Filtering**：Task-aware Expert Loading 仅对 routing-sensitive 任务加载预测 expert，insensitive 任务跳过预测以降低加载开销。

Memory 节省 20%-80%，accuracy 保持 93.7%-99.8%。与 baseline Pre-gatedMoE（2.4×-3.5× slower than eMoE）和 MoEInfinity（1.25×-1.5× slower）相比，eMoE 避免了 per-layer continuous prefetching 的 CPU-GPU 带宽争抢。
