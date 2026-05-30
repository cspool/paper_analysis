## Toward Efficient Inference for Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  论文提出三个 MoE 推理优化技术：(1) **Dynamic Gating**——将静态 gating 的 batch matmul dispatch 替换为 argsort + bin-count + indexing 的动态路由，eliminates dispatch mask 和 placeholder computation；(2) **Expert Buffering**——GPU/CPU 两级缓存机制，仅将热 expert 留在 GPU 显存；(3) **Load Balancing**——Greedy 和 Anti-Correlation 两种算法优化 expert 到 GPU 的分配。
  实验比较：baseline 为 Fairseq static gating，对比方法包括 Tutel（custom kernel）、FasterMoE（communication-computation overlap）、Megablock（block-sparse kernel）。评估指标为 token throughput、memory usage、cache miss rate、load balance（Max load / Avg-Max load）。

- 硬件平台是什么，配置是什么。
  - *Apple* 集群：8×NVIDIA Tesla V100 (32GB HBM2) via NVLink，2×Intel Xeon E5-2698 v4，700GB CPU memory，16GB/s PCIe 3.0 CPU-GPU 带宽，支持单节点和多节点（2/4 nodes）。
  - *Pear* 集群：4×NVIDIA RTX A5000 (24GB)，2×Intel Xeon Gold 5317，64GB CPU memory，32GB/s PCIe 4.0，仅单节点。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - LM-Small: 125M dense / 15B MoE（E=512 experts, top-2 gating, C=0.05, 12 layers, d_model=768, d_ff=3072）
  - LM: 355M dense / 52B MoE（E=512 experts, top-2 gating, C=0.05, 24 layers, d_model=1024, d_ff=4096）
  - MT: 3.3B dense / 54.5B MoE（E=128 experts, top-4 gating, C=1.0, 48 layers, d_model=2048, d_ff=8192）
  数据集：LM 使用 PILE 数据集（Wikipedia, PubMed, Github 三个 domain）；MT 使用 NLLB-200 验证集（English→French, Japanese, Austrian）。
  开源情况。代码开源：https://github.com/hyhuang00/moe_inference，基于 Fairseq [14] 实现。

  算法pipeline 详解（伪代码）：

  ```
  # === Static Gating (Baseline, Fairseq) ===
  # O(S^2 * E * D * C) via batch matmul
  def static_gating(tokens, S, E, C, D):
      # Step 1: gating decisions → expert assignments (size S)
      gate_logits = gate_linear(tokens)           # (S, E)
      assignments = top_k(gate_logits, k)          # (S, k)
      
      # Step 2: create dispatch mask (E, S, S×C) — HIGHLY SPARSE
      mask = zeros(E, S, S*C)
      for i in range(S):
          for e in assignments[i]:
              if expert_capacity_remaining[e] > 0:
                  row = find_first_empty_row(mask[e])
                  mask[e, row, i] = 1
      
      # Step 3: Batch matmul to reorder tokens — O(S^2 E D C)
      dispatched = bmm(mask, tokens)              # (E, S×C, D)
      return dispatched
      
      # 问题: 大量 placeholder (zeros) 填充, mask 内存大

  # === Dynamic Gating (Proposed) ===
  # O(S*D + S*logS) via argsort + indexing
  def dynamic_gating(tokens, S, E, D):
      # Step 1: gating decisions
      gate_logits = gate_linear(tokens)           # (S, E)
      assignments = top_k(gate_logits, k)          # (S, k) → flatten
      
      # Step 2: argsort to group by expert — O(S log S)
      sorted_idx = argsort(assignments[:, 1])     # sort by expert ID
      sorted_tokens = tokens[sorted_idx]           # O(SD) indexing
      
      # Step 3: bin-count for sizes — O(S)
      expert_sizes = bincount(assignments[:, 1], minlength=E)
      
      # Step 4: all-to-all notify sizes (first round, 20µs avg)
      comm.all_to_all(expert_sizes)  # each GPU learns incoming sizes
      
      # Step 5: split + all-to-all transfer tokens (second round)
      token_groups = split(sorted_tokens, expert_sizes)  # by expert
      received = comm.all_to_all(token_groups)
      
      return received  # variable-length list per expert
      
      # 收益: 无 placeholder, 无 mask allocation, 根据实际负载动态容量
  ```

  张量计算对比：
  ```
  Static Gating:
    Tokens X ∈ R^{S×D}  →  Dispatch Mask M ∈ R^{E×S×S×C}
    → Dispatched = M × X  (batch matmul, O(S²EDC))
    → Experts see EXACTLY S×C tokens each, filled with zeros
  
  Dynamic Gating:
    Tokens X ∈ R^{S×D} → Gate → argsort → sorted_X[permutation]
    → bincount → sizes[e] = count of tokens for expert e
    → split sorted_X by sizes → variable-length dispatch
    → Experts see EXACTLY sizes[e] tokens, NO zeros
  ```

  关键：标准 EP 中，block b 的所有 token 必须先完成 attention（barrier），再 all-to-all 分发到 expert GPU，所有 expert GPU 完成计算后再 all-to-all 收集（barrier），才能进入 block b+1。AEP 将这两层 barrier 消除——每个 GPU 在任意时刻执行任意 block 的任意层，cold expert tokens 被自然延迟积累，GPU 永不等 barrier。
