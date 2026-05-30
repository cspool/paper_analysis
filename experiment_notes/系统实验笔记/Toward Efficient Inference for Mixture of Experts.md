## Toward Efficient Inference for Mixture of Experts

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出两个针对 MoE 推理 serving 的调度优化：
  (1) **Expert Buffering**：利用 expert 激活的时序局部性（temporal locality），在 GPU 显存中仅保留热 expert 参数，其余 expert 参数缓存在 CPU 内存中。当冷 expert 被激活时，通过 PCIe 从 CPU 向 GPU 传输参数（与 token 传输重叠）。采用 LIFO cache eviction 策略，适配 MoE 中 experts 按 ID 顺序执行的特性——evict 最近使用的 expert 以保留复用距离最短的 expert。
  (2) **Load Balancing**：基于运行时 expert 激活数据优化 expert 到 GPU 的放置。(a) Greedy Balancing：按 expert 历史平均负载排序，贪心分配到负载最小的 GPU，约束每个 GPU 等量 experts；(b) Anti-Correlation Balancing：针对 MT-Decoder 中 expert 激活相关的场景，在负载估计中引入 Pearson 相关系数惩罚，避免相关 experts 放置到同一 GPU。
  实验比较：与原始 Fairseq（无 expert buffering、无 load balancing）对比，评估 throughput、memory usage、cache miss rate（vs Belady's MIN）、load distribution（Max load / Avg-Max load）。

- 硬件平台是什么，配置是什么。
  - *Apple* 集群：8×NVIDIA Tesla V100 (32GB)，NVLink 互联，2×Intel Xeon E5-2698 v4，700GB CPU DRAM，16GB/s PCIe 3.0。支持 1/2/4 node。
  - *Pear* 集群：4×NVIDIA RTX A5000 (24GB)，2×Intel Xeon Gold 5317，64GB CPU DRAM，32GB/s PCIe 4.0。仅单节点。

- 开源Serving框架是什么。修改了什么。
  开源框架：**Fairseq**（Meta 开源的序列建模工具包，基于 PyTorch），作为 baseline MoE 实现；代码改进开源在 https://github.com/hyhuang00/moe_inference。
  修改内容：
  1. 在 Fairseq MoE Transformer 的 Expert Parallelism 层中新增 Expert Buffering 模块：每个 GPU 上维护一个 expert cache（大小可配置），在 MoE forward 前检查当前 batch 需要的 experts 是否在 GPU cache 中。若缺失，通过 `torch.cuda.stream` 异步从 CPU 向 GPU 拷贝 expert 参数，与 token 的 all-to-all 传输重叠。
  2. 新增 Load Balancing 模块：在推理前运行 profiling pass（收集 expert activation 数据），然后运行 Greedy 或 Anti-Correlation 算法重新分配 experts 到不同 GPU，生成新的 expert placement 方案。
  3. Dynamic Gating 替换了 Fairseq 中的 static gating 函数实现——用 argsort + bin-count + indexing 替代 batch matmul dispatch mask（详见算法pipeline条目）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。

  MoE 推理全流程（Fairseq + Dynamic Gating + Expert Buffering + Load Balancing，单 node 8×V100）：

  ```
  [预部署阶段 - Load Balancing]
    │
    ├─ Step A: Profiling pass
    │    用少量 batch 执行推理，收集 expert activation 数据 A_mb
    │    (expert m 在 batch b 中处理的 token 比例)
    │
    ├─ Step B: Expert Placement Optimization
    │    ┌─ Greedy: sort experts by avg_load desc →
    │    │         循环分配 each expert to GPU with min load
    │    └─ Anti-Correlation (for MT-Decoder):
    │              在 estimated_load 中加 Pearson corr 惩罚项
    │              load[n] += 0.5 × Σ S_am (correlated experts)
    │    → 输出: expert_to_device 映射，启动时分配
    │
    └─ Step C: GPU Cache 初始化
        每个 GPU 分配固定大小的 expert cache (e.g., 10 experts/GPU)
        初始状态: cache 为空，全部 expert 参数在 CPU memory

  [推理阶段 - Per Batch Forward]
    │
    ├─ Input: token batch S=8 (LM) / S=48 (MT), seq tokens X ∈ R^{S×D}
    ├─ Multi-Head Attention (标准 Transformer, 非 MoE 层)
    │     QKV projection → attention → output X_attn
    │
    ├─ MoE Gating Layer (Dynamic Gating)
    │   │
    │   ├─ gate_logits = W_gate @ X_attn            // O(SD×E)
    │   ├─ assignments = top_k(gate_logits)          // top-2 or top-4
    │   ├─ sorted_idx = argsort(assignments.专家ID)   // O(S log S)
    │   ├─ sorted_X = X_attn[sorted_idx]             // O(SD) indexing
    │   ├─ sizes = bincount(sorted assignments)      // O(S)
    │   │
    │   └─ [All-to-All Round 1]: 通知各 GPU 即将接收的 token 数量
    │       各 GPU 传送 size 整数（~20µs average latency）
    │
    ├─ [Expert Buffering - Fetch & Execute]
    │   For each GPU (hosting subset of experts per Load Balance placement):
    │   │
    │   ├─ Step 1: Check Expert Cache
    │   │    for each expert e needed by this GPU's tokens:
    │   │      if e not in GPU cache:
    │   │        launch async CPU→GPU copy for e's parameters (cudaMemcpyAsync)
    │   │        // 与以下 all-to-all token 传输重叠
    │   │
    │   ├─ [All-to-All Round 2]: Transfer actual tokens
    │   │    tokens per device = split(sorted_X, sizes)
    │   │    → NCCL all-to-all → 各 GPU 收到 assigned tokens
    │   │
    │   ├─ Step 2: Expert Execution (sequential by expert ID)
    │   │    for expert e on this GPU:
    │   │      await expert e params ready (CPU→GPU copy 完成)
    │   │      tokens_e = received_tokens[expert_e_indices]
    │   │      W_up, W_gate_act, W_down = expert_e_parameters
    │   │      out_e = W_down @ (σ(W_gate_act @ tokens_e) ⊙ (W_up @ tokens_e))
    │   │      // 若 e 不在 cache: cache[e] = params (LIFO evict)
    │   │
    │   ├─ Step 3: Cache Eviction (LIFO policy)
    │   │    if cache full:
    │   │      evict most recently accessed inactive expert
    │   │      // 理由: MoE 按 ID 顺序执行 → LIFO 保留复用距离最短的
    │   │
    │   └─ [All-to-All Round 3]: Collect expert outputs back
    │        expert_outputs → NCCL all-to-all → 返回原始 GPU
    │
    ├─ Output Reordering
    │    restore original token order via inverse permutation
    │
    └─ → Next Transformer layer (attention → gating → experts)
  ```

  Expert Buffering 关键数据流：
  ```
  GPU Memory Layout (per GPU, e.g., 10/32 expert slots):
  ┌──────────────────────────────────────┐
  │  Expert Cache (GPU HBM)              │
  │  slot 0: expert_42  [W_up, W_gate, W_down]  │
  │  slot 1: expert_7   [W_up, W_gate, W_down]  │
  │  ...                                  │
  │  slot 9: empty                        │
  ├──────────────────────────────────────┤
  │  Non-Expert Params ( Attention W_QKV,│
  │    Gate Linear, Token Buffers, etc.)  │
  └──────────────────────────────────────┘
  
  CPU Memory (Host DRAM): 全部 128/512 experts 的完整参数
  
  Cache Miss Flow:
    expert_99 needed → not in cache →
      cudaMemcpyAsync(CPU→GPU, expert_99_params, PCIe stream)
      // 与 all-to-all token transfer 并发
      → LIFO evict slot (e.g., evict expert_7, 最近使用但当前 inactive)
      → cache[evicted_slot] = expert_99
  ```

  性能影响（论文数据）：
  - Expert Buffering 减少 static GPU memory 达 1.47×（~2.25GB on V100）。
  - 对 MT-Decoder，cache size=10 experts/GPU（80 across 8 GPUs）时吞吐仍优于 baseline；再小则 cache miss 主导延迟。
  - LIFO cache miss rate 接近理论最优 Belady's MIN。
  - Greedy Load Balancing 额外提升 throughput up to 1.19×（vs dynamic gating alone）；Anti-Correlation balancing 对相关 expert 场景提供 1.02× 增益。
  - Multi-node: Dynamic Gating + Expert Buffering + Load Balancing 总计提升 throughput 2.21×–4.30× (vs Fairseq baseline)。

  关键区别：标准 EP 在每个 expert layer 前后有 all-to-all barrier——所有 GPU 必须等待最慢的 expert 完成。AMoE 每个 GPU 独立决策执行哪个 layer，不等待其他 GPU。当某 expert tokens 不足时不执行（积累中），GPU 转去执行 token 充足的另一个 layer。
