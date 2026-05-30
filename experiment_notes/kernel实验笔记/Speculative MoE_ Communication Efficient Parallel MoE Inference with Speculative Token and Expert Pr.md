## Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Sem-MoE 中为 Attention-TP 场景实现的两个融合通信原语（Triton kernel）：(1) **Shuffled-Reduce-Scatter (SRS) kernel**——将 speculative token shuffling 嵌入标准 reduce-scatter 操作，在 ring-based 通信过程中同时完成 token 的预测性重排。核心步骤：查询 token-to-device table T 和 inter-layer expert-sequence-to-device table A → 比较置信度选择预测 → argsort 计算 shuffle indices → 按 indices 重排 token 排列 → reduce-scatter 分发各 device 的 token 分片；(2) **Shuffled-AllGather (SAG) kernel**——MoE 计算完成后，allgather 收集各 device 的 token 分片 → 依据保存的 shuffle indices 进行反向 argsort 恢复原始 token 顺序；(3) **优化 argsort kernel**——自定义 Triton 实现的 argsort，比 PyTorch 原生实现快 25%，是 shuffle 操作的核心性能关键路径；(4) **DeepEP 集成**——在 Sem-MoE 中集成 DeepEP 作为高效 all-to-all 通信后端。shuffling 逻辑嵌入 ring-based communication 的额外开销约 1%。
  - 实验比较：(a) 单 MoE layer 延迟——不同 LAR（Local Activation Rate）下 expert layer latency，Sem-MoE 将 LAR 从 25% 提升至 62%/68%，对应 41.8%/46.6% latency reduction；(b) Attention-TP 端到端 TTFT——不同 input length（256/512/1024）下 vs SGLang/MoETuner；(c) SRS/SAG overhead 测量——shuffling 逻辑 overhead 约 1%，argsort kernel 比 PyTorch 原生快 25%。

- 后端平台是什么，配置是什么。
  - GPU：8-GPU server（96GB HBM/GPU，>400GB/s 专用互联）
  - CPU：2× 44-core Intel CPU，2TB DDR5
  - 软件栈：Triton（OpenAI）+ PyTorch + DeepEP + NCCL/HCCL
  - 评估模型：DeepSeek-V2-Lite（64 routed experts/layer）、Qwen3-30B-A3B（128 experts/layer）、Moonlight-16B

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 SGLang 的 MoE 推理流程。修改内容：(1) 新增 SRS kernel（Triton）——实现 token shuffling + reduce-scatter 融合通信；(2) 新增 SAG kernel（Triton）——实现 allgather + token 顺序恢复；(3) 自定义 argsort Triton kernel——比 PyTorch 原生快 25%；(4) 集成 DeepEP 通信库——替换标准 NCCL all-to-all；(5) mock routing 模块（用于 LAR vs latency 的假设性分析）——通过跳过通信延迟来模拟不同 LAR 下的理想性能上限。
  - 评估方法：通过 CUDA event timing 测量单 MoE layer 及 end-to-end 推理各阶段延迟；通过 mock routing 跳过通信来构建不同 LAR 下的性能参考线。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文基于 SGLang（开源 https://github.com/sgl-project/sglang）构建，Sem-MoE 的自研 kernel 代码（~5000 行 Python + Triton）未提供独立开源仓库链接。
  - Kernel 执行全过程（Attention-TP, SRS + MoE + SAG pipeline）：
    ```
    输入：Attention TP 输出 hidden states X ∈ R^{B×H}

    [SRS Kernel — 融合 Shuffle + Reduce-Scatter]
    Step 1: 查询 scheduling tables
      - 对 batch 中每个 token_id j，查询 T[j] → predicted device d_tok
      - 查询 A[(d_prev_layer1, d_prev_layer2)] → predicted device d_seq
      - 选 C_p 较高的预测源 → final device_ids list D ∈ R^B
    Step 2: Triton argsort kernel
      - input: D ∈ [0, E-1]^B  (per-token target device)
      - 按 device_id 排序 → shuffle_indices ∈ R^B
      - 比 PyTorch 原生快 25%（论文实测）
    Step 3: Token shuffling
      - X_shuffled = X[shuffle_indices]  (GPU tensor indexing)
      - 重排使同一目标 device 的 token 连续排列
    Step 4: Reduce-Scatter
      - ring-based: 每 GPU 获得自己负责的 token 分片 X_local
      - shuffling 逻辑 overhead ≈ 1%（嵌入在 ring communication schedule 中）

    [MoE Computing]
    Step 5: Gate function + Expert FFN
      - 各 GPU 对本地 token 分片 X_local 执行 gate + selected expert computation
      - 由于 SRS 已将 token 预 shuffle 到 expert 所在 device
      - 大部分 expert 计算在本地完成（高 LAR）

    [SAG Kernel — 融合 AllGather + 顺序恢复]
    Step 6: AllGather
      - 收集各 GPU 的 expert 输出，恢复完整 batch Y_shuffled
    Step 7: 反向 argsort
      - reverse_indices = argsort(shuffle_indices)  （恢复原始顺序）
      - Y = Y_shuffled[reverse_indices]
    Step 8: 输出恢复后的 hidden states 进入下一 layer

    性能输出：
    - Per-layer expert latency: Normal w/ all-to-all vs SRS pipeline
    - LAR = (#tokens computed locally) / (#total tokens)
    - Overhead: shuffling ≈ 1% of ring communication time
    - EP communication reduction: 41.8% (DeepSeek) / 46.6% (Qwen3) expert layer latency
    ```
