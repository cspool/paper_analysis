## DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是针对 DualSparse-MoE 推理系统的 **Triton kernel 优化**，主要解决 token-expert grouped-GEMM 在计算粒度变化和额外控制操作引入后的效率问题：(1) **Token-Expert Grouped-GEMM Triton Kernel 优化**：由于 2T-Drop 的 dual-threshold 机制导致 token-expert 计算粒度从均匀变为非均匀（有些 expert 被完全跳过、有些仅计算 major half、有些计算完整 expert），标准的 grouped-GEMM kernel 无法高效处理这种变长计算模式。论文使用优化的 Triton kernel 来适配变粒度 grouped-GEMM，在保持 2T-Drop 细粒度计算丢弃的同时，实现与 1T-Drop（粗粒度丢弃）相当的计算效率。(2) **Gating 函数控制逻辑的 Kernel 集成**：将 dual-threshold comparison、expert selection filtering、major/minor expert dispatch 等控制逻辑集成到 Triton kernel 中，减少 host-device 数据传输和 kernel launch 开销。实验比较：在 8×H20 GPU 节点上，1T-Drop vs 2T-Drop 在不同模型（Mixtral TP=8、OLMoE single GPU、DeepSeek EP=8）下的实际 speedup，验证 22%-27% 的 MoE computation drop rate 能有效翻译为 1.17-1.23× MoE module speedup 和 1.07-1.12× end-to-end speedup。

- 后端平台是什么，配置是什么。
  8×NVIDIA H20 GPU 服务器。软件栈：PyTorch + Triton kernel language + SGLang framework + NCCL backend。部署配置：(a) Mixtral-8×7B：TP=8（8×H20 单节点）；(b) OLMoE-Instruct：单 H20 GPU；(c) DeepSeek-V2-Lite-Chat：EP=8（8×H20 单节点）。Speedup 评估：2,000 条随机 prompts（input length=500, output length=100）。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 SGLang inference framework 的 Python/Triton 实现。主要修改：(a) 实现 token-expert computation dropping 的定制 Triton grouped-GEMM kernel，支持变粒度（skip / major-only / full）expert 计算；(b) 将 gating 函数中的 dual-threshold comparison + expert dispatch 逻辑融合到 inference kernel pipeline 中；(c) 确保 2T-Drop 的细粒度计算丢弃不引入额外 kernel launch 开销（与 1T-Drop 保持相同 speedup 水平）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供独立开源代码仓库，基于 SGLang 框架的 Triton kernel 实现。Kernel 原理如下：

  ```
  === Triton Token-Expert Grouped-GEMM Kernel Flow ===

  Input (after gating + dual-threshold decision):
    token_hidden_states: [num_tokens, d_model]
    expert_assignments: [[(expert_id, compute_mode), ...] for each token]
      compute_mode ∈ {SKIP, MAJOR_ONLY, FULL}
    expert_weights: {major_W1, major_W2, major_W3, full_W1, full_W2, full_W3}

  Step 1: Token-Expert Dispatch (Triton kernel)
    Group tokens by compute_mode and target expert:
      For each token t:
        For each activated expert e_j of token t:
          mode = decide(thresholds, s_norm_j)
          if mode == SKIP: continue
          elif mode == MAJOR_ONLY:
            dispatch to group: (expert_e_j, MAJOR)
          else:  # FULL
            dispatch to group: (expert_e_j, FULL)

    Output: per-group token indices + per-group weight pointers

  Step 2: Variable-Length Grouped-GEMM (Triton kernel)
    For each (expert, mode) group:
      tokens_in_group = gathered hidden states
      if mode == MAJOR_ONLY:
        gate_out = Swish(tokens · W_1_major)
        up_out = tokens · W_3_major
        hidden = gate_out ⊙ up_out
        output = hidden · W_2_major
      elif mode == FULL:
        gate_out = Swish(tokens · W_1_full)
        up_out = tokens · W_3_full
        hidden = gate_out ⊙ up_out
        output = hidden · W_2_full

  Step 3: Scatter + Weighted Sum (Triton kernel)
    For each token t:
      y_t = 0
      For each computed expert e_j of token t:
        y_t += s_{e_j} · expert_output[e_j]  // s is original gating score
      token_output[t] = y_t

  === Performance Translation ===
  22%-27% computation drop rate → 1.17-1.23× MoE module speedup
  Key insight: dropping at tensor-level (expert/token granularity)
  enables effective GPU utilization vs fine-grained neuron-level sparsity
  that struggles to convert to real speedup on current hardware
  ```
