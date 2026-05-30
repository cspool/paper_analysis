## Rectified Sparse Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  ReSA 提出了一种结合 block-sparse attention 和周期性 dense rectification 的稀疏解码方法，通过定期刷新 KV cache 来限制稀疏注意力的误差累积，在不牺牲生成质量的前提下显著加速长序列推理。实现包括三个核心组件：
  (1) **Group Block Sparse Attention (GBSA)**：基于 Quest 算法的 query-aware block-sparse attention，使用 block-wise min/max 描述符进行块级近似匹配，结合 GQA group 内共享注意力模式（来自 NSA 的 shared grouping）以进一步提升效率。Block 大小 b=16，使用 dynamic top-n 策略：永远保留最近 n_local=1 个 block，强制最少 n_min=16 个 block，其余根据活跃 ratio p 动态选择。
  (2) **Dense Rectification**：每 f=32 个 token 后，将最近生成的 token 批量用 dense attention 并行重编码，刷新 KV cache 和 block key cache。这保证稀疏注意力的 KV cache 误差被限制在常数窗口内。
  (3) **Memory Access 模型**：平均每步 memory access 为 mem(KV cache) × (1/b + p + 1/f)，相对于 dense decoding 的理论加速因子由 b、p、f 控制。

  实验比较：
  (a) Math reasoning (test-time scaling)：DeepSeek-R1-Qwen-Distill 1.5B/7B 在 Minerva Math、Gaokao2023En、OlympiadBench、AIME24、AMC23 共 5 个基准上的准确率，对比 Dense、Sparse、Sparse_dense2（前两层 dense）和 ReSA，均在 4K–12K token 平均推理长度。
  (b) Language modeling：Qwen2.5 模型在长序列 book data 下，模拟 sparse decoding pattern，评测不同 rectification frequency x 和 sparsity ratio p 下的 top-3 next-token prediction accuracy。对比 Decode Only（upper bound）和 sparse baseline。
  (c) Retrieval (RULER benchmark)：Qwen2.5 7B 在 RULER 的 8 个子任务（QA、MultiQuery、FWE、VT、MultiKey、MultiValue、CWE、Single）上评测不同 sparsity ratio 下的准确率。
  (d) Inference efficiency：Qwen-2.5 7B 在 NVIDIA A100-80G 上的 kernel-level latency breakdown（16K/64K/256K）和 end-to-end throughput（FP16 和 INT4，4K/16K/64K/256K context）。
  (e) Ablation：f ∈ {16,32,64,128} 和 p ∈ {0.9,0.95,0.98} 网格搜索。
  (f) 与 sparse KV-based self-speculation 的 decoding 速度对比（Table 3），ReSA 平均 1.92× speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-80G GPU。所有实验基于 PyTorch 实现。INT4 实验使用 Marlin kernel 进行 low-bit matmul，group-wise scaling（group size=128）。Custom kernel 参考 Flash Decoding 的 split-execution 策略，使用 TileLang 库实现 group block sparse attention。评测 latency 时仅报告 CUDA kernel 执行时间，排除 CPU-side scheduling overhead。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5 系列（Qwen2.5 7B 为主），DeepSeek-R1-Qwen-Distill 1.5B/7B。DeepSeek-R1-Qwen-Distill 7B 配置：28 层，28 attention heads，4 KV heads (GQA)，hidden size 3584。
  数据集/Benchmark：
  - Math reasoning: Minerva Math, Gaokao2023En, OlympiadBench, AIME24, AMC23
  - Language modeling: long-sequence book data
  - Retrieval: RULER benchmark (8 子任务)
  - Efficiency: 无特定数据集，测量 kernel latency 和 throughput

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源在 https://aka.ms/ReSA-LM。核心算法流程见论文 Algorithm 1（Rectified Sparse Decoding），张量计算如下：

  **Step 1 - Prefill（dense attention）**：
  ```python
  # 标准 dense prefill，构建完整 KV cache K 和 block key cache B
  K, B = DensePrefill(input_ids, model)
  # K: KV cache, B: block key descriptors (min/max per block)
  ```

  **Step 2 - Group Block Sparse Attention (GBSA)**：
  ```python
  # Block representation: 将 key 矩阵按 block size b 分区
  # k_block_min_i = min(k[i*b:(i+1)*b]), k_block_max_i = max(k[i*b:(i+1)*b])
  
  # Block selection per GQA group:
  q_pool = avg_pool(Q_group)  # 组内 query heads 平均池化
  for each block i:
      score_i = sum_j(max(q_j * k_block_max_i[j], q_j * k_block_min_i[j]))
  # 选择 top n blocks (n = max(n_min, ceil(M * p))), n_local 个最近 block 强制保留
  
  # Sparse attention:
  M = create_sparse_mask(selected_blocks)  # M ∈ {0,1}^{h × n × n/b}
  O = softmax(Q @ K^T * extended_mask(M) / sqrt(d)) @ V
  ```

  **Step 3 - Dense Rectification（每 f=32 tokens）**：
  ```python
  if step % f == 0:
      # 将最近 f 个 token batch 用 dense attention 并行重编码
      K, B = DenseForward(tokens[-f:], K, B)
      # 刷新 block key cache B 以匹配更新后的 KV cache K
  ```

  **Step 4 - 循环生成**：交替 sparse decoding → rectification → sparse decoding，直到生成完毕。

  Decode Only 设置（upper bound）：KV cache 全部由 dense attention 构建，仅新 token 使用 sparse attention 解码——代表 ReSA 理论上界。
