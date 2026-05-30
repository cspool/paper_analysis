## LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - LExI 在推理服务层面的核心实现是**修改 vLLM 推理框架中 MoE 层的 top-k 路由参数**，为每一层静态设置不同的 active expert 数量（而非所有层使用相同的 top-k）。具体而言：LExI 的 Stage 1 通过 Monte Carlo 采样计算每层在不同 top-k 下的 Frobenius 范数扰动损失，Stage 2 通过进化搜索在总 active expert budget B 约束下找到最优的逐层 k_j 分配。得到的 k* = (k_1, ..., k_L) 直接应用于 vLLM 的 FusedMoE 模块，替换原有的固定 top-k 路由，使得在推理时每层激活不同数量的 expert。
  - 实验比较：
    - LExI vs Baseline（固定 top-k）vs Inter-Expert Pruning (NAEE) vs Intra-Expert Pruning (MoE-I²)
    - 指标：Throughput（tokens/s，end-to-end latency 换算）、Accuracy/F1/Perplexity
    - 在多个 active expert budget B 下的 Pareto trade-off 对比（B=100, 120, 150, 180 等）

- 硬件平台是什么，配置是什么。
  - **NVIDIA H100 80GB GPUs**
  - 大多数模型 4 GPUs（Mixtral-8x7B, Qwen1.5-MoE, OLMoE, MiniCPM），DeepSeek-V2-Lite 和 DeepSeekVL2-Tiny 使用 2 GPUs
  - Tensor Parallelism 跨 GPU 部署
  - Batch size = 16 推理

- 开源Serving框架是什么。修改了什么。
  - **vLLM**（Kwon et al. 2023）：高性能 LLM 推理框架，原生支持 MoE 模型通过 **FusedMoE** 模块（融合 expert 计算和路由以提升效率）
  - LExI 的修改：
    - 在模型加载后、推理执行前，LExI 根据进化搜索得到的 k* 修改每个 MoE 层的 top-k 参数
    - 具体操作：对每个 MoE layer j，调用 `set_topk(model.moe_layers[j], k_j)` 将路由器的 TopK 选择数量从统一的 k_base 改为 k_j
    - LExI 本身不修改 vLLM 的调度逻辑、内存管理（PagedAttention）或 kernel 实现，仅改变 MoE 层的路由配置参数
    - 推理时的 token 路由流程不变：input token → router(gate) → TopK(k_j) → 激活 k_j 个 expert → 加权求和 → output

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **未开源**。论文未提供代码仓库或任何开源链接。
  - vLLM 框架在 LExI 优化下的推理全过程：
    ```
    1. 模型加载阶段
       model = load_moe_model("Mixtral-8x7B-Instruct")  # 从 HuggingFace 加载
       // 原始所有层 top-k = 2

    2. LExI 离线优化（一次执行）
       k_star = LExI_optimize(model, budget=B)  # Stage 1+2
       // k_star = [2, 1, 2, 2, ..., 1]  — 32层的逐层 top-k
       for layer_j, k_j in enumerate(k_star):
           model.moe_layers[layer_j].topk = k_j  // 修改路由参数

    3. 推理执行（vLLM 在线服务）
       // Batch 请求到达
       prompts = ["What is AI?", "Explain...", ...]  # batch_size=16
       tokens = tokenizer(prompts)

       // Prefill 阶段：并行处理所有 prompt tokens
       for layer_j in range(L):
           hidden_states = attention(layer_j, tokens)  // Self-Attention
           // MoE 层：路由 + 专家计算
           router_logits = gate(hidden_states)  // [B, L, N_experts]
           topk_indices, topk_weights = TopK(router_logits, k=k_j)  // 使用层特定 top-k
           // FusedMoE Kernel (H100 Tensor Cores)
           expert_outputs = []
           for e in topk_indices:
               expert_outputs.append(expert[e](hidden_states))  // FFN: W1→Act→W2
           output = sum(topk_weights[i] * expert_outputs[i])
           tokens = output

       // Decode 阶段：自回归生成
       for step in range(max_new_tokens):
           // 逐 token 经过所有层（每层用其 k_j 个 expert）
           ...

    4. 硬件执行
       - 每个 token 在每个 MoE 层仅激活 k_j 个 expert（而非固定的 2 或 4 个）
       - 减少的总 expert 计算量 ≈ Σ_j (k_base - k_j) / (L × k_base)
       - H100 Tensor Cores 执行 expert FFN GEMM（FusedMoE batch 操作）
       - Communication: 更少的 active experts = 更少的 all-reduce/broadcast 通信量
       - Memory bandwidth: 更少的 expert 参数需要从 HBM 加载到 SM
     ```
  - LExI 在 vLLM 中的作用：通过减少敏感度低的层的 active expert 数量，直接减少那些层的 FFN 计算量、inter-GPU 通信量和 memory bandwidth 使用，从而提升整体推理吞吐量。与 expert pruning 不同，LExI 不删除任何 expert 参数，因此模型在需要时仍可激活更多 expert（通过改变 budget B 快速调整）。
