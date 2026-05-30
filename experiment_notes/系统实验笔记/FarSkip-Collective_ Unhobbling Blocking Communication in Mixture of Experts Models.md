## FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

- 属于Serving调度的实现是什么？实验比较什么？
  - FarSkip-Collective 在推理侧修改 vLLM 和 SGLang 两个开源推理框架，实现 MoE 模型中 all-reduce 通信与计算的重叠。具体修改包括：
    1. **MoE 层 all-reduce 异步化**：vLLM/SGLang 中 MoE 推理使用 all-reduce（而非 all-to-all）来聚合各 EP rank 上的专家输出。FarSkip 将 all-reduce 改为 async_op 模式运行，仅在下一个 MoE 计算前同步，利用 FarSkip 架构中 MLP 输入不依赖当前 attention 输出的特性，将 all-reduce 与 attention 子块计算重叠。
    2. **Attention 层 RowParallelLinear all-reduce 异步化**：修改 attention 输出投影层（RowParallelLinear），将内部 all-reduce 改为 async_op 模式，同步调用仅在下一次 attention 层之前执行。
    3. **HIP/CUDA-graphs 集成**：使用 graph-compatible 通信 API 调用（PyNCCL direct Python binding），使重叠机制与 CUDA graphs 兼容，支持 prefill 和 decode 阶段的 fused kernel。
    4. **MLA（Multi-head Latent Attention）特殊处理**：针对 DeepSeek 模型的 MLA prefill 和 decode 分别使用不同的 fused kernel，每种情况单独实现 async all-reduce 调用。
  - 实验比较 FarSkip-Collective 推理 vs 常规推理在 TTFT（Time-To-First-Token）和解码阶段的加速比。

- 硬件平台是什么，配置是什么。
  - vLLM 推理：1× AMD MI300X 8GPU 机器；FP8 量化 + fused-MoE forward kernel。
  - SGLang 推理单节点：TP=8, EP=8 配置。
  - SGLang 推理多节点：2 节点系统，TP=16, EP=16，8×400Gbs NIC 互联。
  - 推理配置：prefill 阶段 BS=2（per-device），EP=8, TP=8；decode 阶段 BS=1024（多节点 large-batch 设置）。

- 开源Serving框架是什么。修改了什么。
  - **推理框架**：vLLM [19] 和 SGLang [45]——现代 LLM 推理引擎，支持 TP、EP、PP，用于 MoE 模型（如 DeepSeek）的分布式推理。
  - **修改内容**：
    1. **MoE 层**：将 EP 相关的 all-reduce（原本用于聚合各 rank 的专家输出）从同步模式改为 async_op 模式，同步点延迟到下一个 MoE 计算之前。这利用了 FarSkip 架构中 MLP 子块输入使用 outdated activation（$o_{k-1}$）不依赖最新 attention 输出的特性。
    2. **Attention 层**：修改 RowParallelLinear 输出投影层中的 all-reduce 为 async_op 模式，同步延迟到下一个 attention 层之前。对于 MLA 的 prefill 和 decode fused kernel 分别处理。
    3. **CUDA Graphs 兼容**：使用 graph-compatible 通信 API（PyNCCL）替代标准 torch.dist 调用，确保在 CUDA graph capture 场景下异步通信正常工作。
    4. **设计原则**：所有修改在 PyTorch API 层面完成（torch.dist async_op + torch.cuda.Stream），避免 low-level kernel 或 Triton 修改，保持硬件无关性和框架兼容性。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文声明 "We plan to open-source our implementation and modified model checkpoints and provide easy integration with the upstream frameworks"，截至分析时未在 web search 中发现公开代码仓库。
  - **推理框架执行全过程（以 Llama-4-Scout 109B 在 vLLM 上，EP=8, TP=8, MI300X 8GPU 为例）**：

    ```
    ┌─────────────────────────────────────────────────────────┐
    │ 1. vLLM 接收推理请求                                     │
    │    输入：用户 prompt tokens [T₁, T₂, ..., Tₙ]            │
    │    vLLM scheduler 分配请求到 GPU，管理 KV cache           │
    │           ↓                                              │
    │ 2. Attention 子块（TP=8, 列并行 Q/K/V + 行并行 O）       │
    │    ┌─ Q/K/V projection (ColumnParallelLinear) ──────────┐│
    │    │  各 TP rank 独立计算，无通信                         ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ Core Attention (fused MLA kernel) ────────────────┐│
    │    │  各 TP rank 独立计算 attention scores + output       ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ Output projection (RowParallelLinear) ───────────┐ │
    │    │  **FarSkip 修改**: all_reduce(async_op=True)       │ │
    │    │  启动异步 all-reduce，立即返回，不等待完成           │ │
    │    │  返回 partial output (本 rank 计算结果)             │ │
    │    └────────────────────────────────────────────────────┘│
    │           ↓                                              │
    │ 3. MoE 子块（EP=8, 各 rank 持有 E/8 个 expert 权重）     │
    │    ┌─ Gating/Router (各 rank 复制执行) ─────────────────┐│
    │    │  Router(token) → top-k expert indices              ││
    │    │  各 rank 独立计算，无通信                            ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ Expert FFN (各 rank 本地 expert 计算) ────────────┐│
    │    │  输入：replicated activations（vLLM EP 方式）       ││
    │    │  各 rank 仅计算自己持有的 experts                    ││
    │    │  fused-MoE forward kernel, FP8 量化                 ││
    │    └────────────────────────────────────────────────────┘│
    │    ┌─ All-Reduce 聚合 (EP 间) ──────────────────────────┐│
    │    │  **FarSkip 修改**: all_reduce(async_op=True)       ││
    │    │  聚合各 rank 的专家输出                              ││
    │    │  异步启动，返回 partial 结果                         ││
    │    └────────────────────────────────────────────────────┘│
    │           ↓                                              │
    │ 4. 下一层 Attention 子块执行时                             │
    │    ┌─ 同步上一层的 all-reduce ──────────────────────────┐│
    │    │  Wait(all_reduce_handle)  // 此时通信已被重叠       ││
    │    │  获取完整 activation 用于残差加和                    ││
    │    └────────────────────────────────────────────────────┘│
    │    // FarSkip 利用 mlp-in_k = o_{k-1} (outdated)         │
    │    // 使得 MLP 输入不需要等待 attention all-reduce 完成   │
    │    // attention 计算可与上一层的 all-reduce 重叠          │
    │           ↓                                              │
    │ 5. 输出：first token (TTFT) 或 next token (TBT)          │
    └─────────────────────────────────────────────────────────┘
    ```

    **vLLM EP 的 all-reduce 方式说明**：
    不同于训练中使用 all-to-all Dispatch+Combine，vLLM/SGLang 的 MoE EP 实现将 activation 在各 rank 上复制，仅 expert 权重按 EP 分布。各 rank 计算自己的 experts 后，通过 all-reduce 聚合结果（而非 all-to-all）。这种方式消除了 Dispatch/Combine 的 permutation 开销，但 all-reduce 仍然是阻塞的。FarSkip 将此 all-reduce 异步化并与计算重叠。

    **CUDA Stream 层面的执行调度**：
    ```python
    # 伪代码：FarSkip 在 vLLM 中 MoE 层的实现
    # 主计算 stream (default stream)
    with torch.cuda.stream(compute_stream):
        gate_out = router(hidden_states)
        expert_out = fused_moe(hidden_states, gate_out, expert_weights)
    
    # 异步 all-reduce 在独立 stream 上运行
    with torch.cuda.stream(comm_stream):
        # async_op=True: 启动后立即返回 handle
        all_reduce_handle = torch.dist.all_reduce(
            expert_out, async_op=True
        )
    
    # 不等待 all-reduce，立即进入下一层的 attention 计算
    # （FarSkip 架构保证 attention 输入不需要完整的 expert_out）
    attn_out = attention(hidden_states)  # 与 all-reduce 重叠
    
    # 在需要完整 expert_out 之前同步
    all_reduce_handle.wait()
    final_out = hidden_states + attn_out + expert_out
    ```

    **性能结果**：
    - Llama-4-Scout (109B): all-reduce 重叠率 95.3%, TTFT 加速 12.2%-18.5%
    - DeepSeek-V2 (235B): all-reduce 重叠率 97.6%, TTFT 加速 8.2%-16.8%
    - DeepSeek-V3 (671B) SGLang: TTFT 加速 up to 1.34× (TP=8, EP=8)
    - 多节点 decode (TP=16, EP=16, BS=1024): 显著且一致的 TBT 加速（Fig. 7）
