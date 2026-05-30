## FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

- 属于算法pipeline的实现是什么？实验比较什么？
  - FOLDMOE 提出将 token-level 的 all-to-all（A2A）通信与计算的 overlapping 从 MoE 层扩展到整个 Transformer block，通过 attention-MoE pipelining 实现。核心实现包括三部分：
    1. **1A1M 调度（1-Attention-1-MoE schedule）**：将 Transformer block 划分为四个流水线阶段（attention computation → A2A dispatch → expert computation → A2A combine），通过交错执行 attention 和 expert computation，减少 aAaM 调度中因阶段不平衡导致的流水线气泡。
    2. **Token Buffer 与时间均匀微批次（Time-Uniform Micro-Batching）**：在 attention 层和 MoE 层之间引入 token buffer 解耦二者的微批次划分，使 attention 层可按时间均匀（非 token 数量均匀）切片，MoE 层仍保持 token 数量均匀的微批次。使用基于 FLOPs 建模的启发式算法（Algorithm 1: Quick-start time-uniform attention slicing）确定切片方案。
    3. **与 FlashAttention、TP、SP 的兼容**：FOLDMOE 在不改变 attention 因果掩码的前提下与 FlashAttention 兼容；与 TP 正交（TP 切分算子，FOLDMOE 沿序列维度切分数据）；与 SP 兼容（SP 仅作用于 layernorm、dropout 等非 attention/非 MoE 区域）。
  - 实验比较 FOLDMOE 与 Megatron-MoE（无 overlapping baseline）和 Tutel（SOTA token-level MoE-only overlapping baseline）在 GPT-MoE 模型训练上的每迭代延迟（per-iteration latency）加速比。

- 硬件平台是什么，配置是什么。
  - 2 个 AWS g5.48xlarge 节点，每节点 8 张 NVIDIA A10G-24G GPU，共 16 GPU。
  - 节点间通过 100 Gbps 网络互联。
  - 训练配置：2-way cross-node DP + 8-way intra-node TP+SP（attention 层）+ 16-way EP（MoE 层）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：GPT-MoE 系列，基于 GPT-2 的 MoE 变体：
    - GPT-MoE-S: n_layer=6, d_model=512, n_heads=8, expert_hidden_size=1024
    - GPT-MoE-M: n_layer=6, d_model=768, n_heads=8, expert_hidden_size=1536
    - GPT-MoE-L: n_layer=12, d_model=1024, n_heads=8, expert_hidden_size=2048
    - 每隔一个 Transformer block 将 FFN 替换为 MoE 层，使用 top-1 GShard gate。
  - 数据集：Wikipedia dataset。
  - 序列长度：4K 到 32K（均为 2 的幂）。
  - 指标：per-iteration training latency（平均 per-block 延迟），training throughput。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源情况**：论文未提供开源代码链接，在 ACL Anthology 和 web search 中均未找到公开的代码仓库。
  - **算法 pipeline 解释**：

  FOLDMOE 的核心是将每个 Transformer block 内的 attention 层和 MoE 层组成四级流水线：

  ```
  阶段1: Attn(X_{i:j}; K_{1:i-1}, V_{1:i-1}) → Z_{i:j}
  阶段2: A2A dispatch(Z_{i:j})          → Z'_{i:j}  (发送到对应 expert 所在 GPU)
  阶段3: Expert(E_i, Z'_{i:j})          → Y'_{i:j}  (各 expert 独立计算)
  阶段4: A2A combine(Y'_{i:j})          → Y_{i:j}   (收集回原 GPU)
  ```

  **1A1M 调度伪代码**（单 Transformer block 前向）：
  ```
  # 输入: sequence X[0..L-1]，切片方案 S={l1, l2, ..., ld}
  # K/V cache 初始为空
  K_prev, V_prev = [], []
  start = 0

  for mb_idx = 0 to d-1:
      l = S[mb_idx]  # 当前 micro-batch 的 token 数
      X_mb = X[start : start+l]

      # 阶段1: Attention (可与上一 micro-batch 的 A2A combine 重叠)
      K_mb, V_mb = compute_kv(X_mb)
      K_attn = concat(K_prev, K_mb)
      V_attn = concat(V_prev, V_mb)
      Z_mb = flash_attn(X_mb, K_attn, V_attn, causal=True)

      # 存入 token buffer
      buffer.append(Z_mb)

      # 从 buffer 中取 token-uniform 微批次 (MoE 侧固定大小 m = ceil(L/d))
      while len(buffer) >= m:
          Z_moe = buffer.pop(m)  # FIFO 取出 m 个 token
          # 阶段2: A2A dispatch (与下一 attention 微批次重叠)
          Z_disp = all_to_all_dispatch(Z_moe)
          # 阶段3: Expert computation
          Y_expert = moe_experts(Z_disp, gate)
          # 阶段4: A2A combine
          Y_moe = all_to_all_combine(Y_expert)
          Y.concat(Y_moe)

      K_prev = concat(K_prev, K_mb)
      V_prev = concat(V_prev, V_mb)
      start += l

  # drain buffer
  while buffer not empty:
      Z_moe = buffer.pop(min(m, len(buffer)))
      Y_moe = a2a_dispatch → expert → a2a_combine(Z_moe)
      Y.concat(Y_moe)

  return Y
  ```

  **时间均匀切片算法（Algorithm 1）**：
  - 输入：序列总长 L，overlap degree d，理想切片时间 t̂
  - 首先分配 quick-start slice（大小为 ceil(L/d)），最小化启动 A2A 的延迟
  - 然后基于 attention FLOPs 建模 `FLOPs(l, c) = (4H + 3h)lc + 8H²l` 迭代确定后续切片边界，使每个 attention 微批次的计算时间接近 t̂
  - 时间复杂度 O(L)

  **反向传播**：流水线调度按相反顺序执行（A2A combine → expert grad → A2A dispatch → attention grad），保持与正向相同的重叠模式。
