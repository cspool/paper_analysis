## BrownoutServe SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 **Triton 语言的 MoE 算子重写 + PagedAttention GPU kernel 优化**：
  1. **MoE 算子 Triton 重写**：BrownoutServe 的 MoE 模块（含 BrownoutMoE 的 expert FFN 计算、token routing、united expert 调用）全部使用 Triton 语言重写，替代传统的 C++/CUDA 实现。Triton 与 PyTorch 无缝兼容，简化了开发和维护。
  2. **PagedAttention GPU kernel 优化**：相比 vLLM 的 PagedAttention 实现，BrownoutServe 将 block table 从 CPU 移至 GPU，block table 操作（查询、映射、更新）全部实现为 GPU kernel 函数。这充分利用了 GPU 并行计算能力，有效减少 PagedAttention 的额外开销。

  实验比较（间接体现在端到端吞吐量/延迟中）：BrownoutServe (with fused MoE) vs vLLM (native, with fused MoE) 在 ShareGPT 和 Alpaca 上的吞吐量比较，其中 BrownoutServe 在使用 fused MoE 后仍能提升 1.07×-1.32×（Fig. 9），部分来源于 kernel 级优化的贡献。

- 后端平台是什么，配置是什么。
  4× NVIDIA A100-PCIE-40GB GPU（每卡 40GB HBM），Intel Xeon Gold 6238 CPU。

- 评估性能的软件/脚本是什么。修改了什么。
  **评估工具**: PyTorch + Triton kernel 语言。论文使用 ShareGPT 和 Alpaca 数据集进行 10 分钟连续推理吞吐量评估，250s burst trace 进行 SLO violation 评估。

  **修改/优化内容**:
  1. **MoE 算子 Triton 重写**：将 BrownoutMoE 中的 expert FFN 计算（矩阵乘法、激活函数）、token dispatch/combine 操作、united expert 调用全部用 Triton 实现。Triton 代码在 Python 层面编写 tile-level 计算逻辑，编译为高效 GPU kernel，替代了传统 hand-written CUDA C++ kernel，降低了开发复杂度同时保持性能。
  2. **PagedAttention block table GPU 化**：原 vLLM 的 block table 管理在 CPU 端进行，每次 attention 计算需 CPU→GPU 数据传输。BrownoutServe 将 block table 直接置于 GPU 显存，block table 的查询（lookup KV cache block index）、映射（map logical→physical block）、更新（eviction/new block allocation）操作全部实现为 GPU kernel。这消除了 CPU-GPU 数据传输延迟，利用 GPU 大规模并行性加速 block table 操作。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。

  **开源**: https://github.com/beyondHJM/BrownoutServe（Apache-2.0）

  **Kernel 调度评估原理（以 1 次 MoE forward pass, Qwen1.5-MoE-A2.7B, 60 experts 为例）**：

  **1. BrownoutMoE kernel 全流程**：
  ```
  输入: token hidden states [batch_size, hidden_dim], shape=(64, 2048)

  Step 1 - Gate Kernel (Triton):
      for each token t (GPU thread block):
          # 计算 token 与所有 60 experts 的 affinity
          s[t, :] = x[t] @ E_centroids.T  # [64, 60]
          top2_idx[t], top2_score[t] = topk(softmax(s[t, :]), k=2)
      → 输出: routing_map (token→expert 映射), routing_weights

  Step 2 - Token Counting & Sorting (GPU):
      A = [(expert_id, token_count, hidden_states), ...]  # 60 个 expert
      sort A by token_count descending (GPU radix sort)
      T = total_tokens * threshold  # S1 阈值
      partition A → S1 (hot experts), S2 (cold experts)

  Step 3 - S1 Original Expert FFN (Triton fused MoE kernel):
      for each expert e in S1:
          # 将 e 对应的 tokens 合并为单 batch
          tokens_e = gather(tokens routed to expert e)
          # FFN: gate_proj → up_proj → activation → down_proj
          h = tokens_e @ W_gate  # gate projection
          u = tokens_e @ W_up    # up projection
          out = (silu(h) * u) @ W_down
          # scatter 回原位置
      → 使用 fused MoE 实现时，多 expert 计算合并为一次 sparse GEMM

  Step 4 - S2 United Expert FFN (Triton kernel):
      对 S2 中 experts 按 way=k 分组
      对每组 group:
          合并所有 group 内 tokens → concat_tokens
          united_expert_out = UE(concat_tokens)  # Triton FFN kernel
          scatter united_expert_out 回原 token 位置

  输出: 所有 token 经过 MoE 后的 hidden states [64, 2048]
  ```

  **2. PagedAttention GPU Block Table Kernel 原理**：
  ```
  原 vLLM (CPU block table):
      CPU 维护 block_table[t] = [physical_block_ids]
      每次 attention: CPU → GPU copy block_table → GPU kernel 查询
      ↑ 通信开销随 seq_len 增长

  BrownoutServe (GPU block table):
      GPU 显存中维护 block_table (torch tensor on device)
      GPU kernel 直接访问:
          for each query token q:
              # 同一 kernel 内完成 block table lookup + attention
              physical_blocks = block_table[request_id]  # GPU-side lookup
              K_cache = gather KV from physical_blocks
              V_cache = gather V from physical_blocks
              out[q] = flash_attention(Q[q], K_cache, V_cache)
      → 消除 CPU→GPU 数据传输，减少 kernel launch 次数
  ```
