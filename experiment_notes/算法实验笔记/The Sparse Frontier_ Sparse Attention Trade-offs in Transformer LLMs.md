## The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  对 training-free 稀疏注意力方法进行最大规模的实证分析。基于四个设计轴（sparsification unit、importance estimation、budget allocation、KV cache management）建立分类体系，选取六种代表性方法并统一实现：**prefilling 阶段**——Vertical-Slash（全局垂直列+对角线斜杠，基于近端 query token 近似估计重要性，均匀 budget）、FlexPrefill（Vertical-Slash 增强版，threshold-based 动态 budget 分配，由 coverage 参数 α 和 min_budget 控制）、Block-Sparse（固定 block 大小 16×16，按 query block 选 top-k key block，均匀 budget）；**decoding 阶段**——SnapKV（token 级重要性估计，1D average pooling kernel=21 平滑，均匀 budget，KV cache eviction）、Ada-SnapKV（SnapKV 增强版，max-aggregation 替代 mean-aggregation 用于跨 head 动态 budget 分配，每 head 最低 budget 20%）、Quest（page 级选择，page size=16，使用 page 的 min/max key 值近似 query-page 相似度，均匀 budget，全 KV cache 保留）。所有方法保留 attention sinks（前 4 个 token）和局部上下文。通过 vLLM 的 FlashAttention 层级拦截实现。

  实验比较：(1) isoCost 分析——相同计算成本下密集小模型 vs 稀疏大模型的 Pareto 前沿对比（Figure 1）；(2) 9 个长期上下文任务（SQuAD/QuALITY/TOEFL QA、RULER NIAH/VT/CWE、Story Retrieval/Multi-hop/Filtering）上的 per-task 性能（Figure 2），分四个信息检索特征组；(3) 不同序列长度 (16k/32k/64k/128k) 下的稀疏容忍度（Figure 3）；(4) 模型大小效应——7B 到 72B 参数量的稀疏容忍度缩放分析（Figure 20/21）；(5) 六种方法之间的横向对比（Vertical-Slash vs FlexPrefill vs Block-Sparse for prefill; SnapKV vs Ada-SnapKV vs Quest for decode）；(6) 多种消融实验——block size (Block-Sparse)、page size (Quest)、近似窗口大小 (Vertical-Slash/FlexPrefill)、kernel size 和近似窗口 (SnapKV/Ada-SnapKV)、min budget (FlexPrefill/Ada-SnapKV)。

- 硬件平台是什么，配置是什么。
  4 个计算节点，每个节点 8 块 NVIDIA H100 GPU，共 ~32 块 H100，运行 21 天。使用 vLLM 推理引擎，全 bf16 精度。总共评估 7065 个配置，每个配置 100 个样本（Qwen）或 50 个样本（Llama/Gemma）。

- 模型是什么。数据集和bench分别是什么。
  **模型**：Qwen 2.5 (7B/14B/32B/72B)，Llama 3.1 (8B/70B)，Gemma 3 (4B/12B/27B)。所有使用 instruction-tuned 变体以支持 chain-of-thought 评估。Gemma 3 采用混合注意力——5/6 层使用 sliding window (1024 tokens)，仅在密集（global attention）层应用稀疏注意力方法。**数据集/Benchmark**：9 个任务——QA 类 (SQuAD/QuALITY/TOEFL)、RULER 合成任务 (NIAH/VT/CWE)、新增 Story 任务 (Story Retrieval/Multi-hop/Filtering) 基于程序化生成的多章叙述。**指标**：Exact Match Accuracy、IoU、F1（范围 0-1），计算成本使用 FLOPs (prefilling) 和 memory transfers (decoding)。稀疏度 0 到 0.95（对应 attention budget 1/1.5 到 1/20）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  **开源**：https://github.com/PiotrNawrot/sparse-frontier（MIT 许可证）。代码在 vLLM 的 FlashAttention 层级拦截 attention 计算，提供 `AbstractAttention` 基类，用户继承并注册即可实现自定义稀疏注意力。

  **算法 Pipeline 伪代码（以 Vertical-Slash prefill 为例）**：
  ```
  Input: Q, K, V ∈ R^{n×d}, sparsity_level, window_size=256
  1. 选择近端 query token: Q_recent = Q[-window_size:, :]  # shape: [w, d]
  2. 近似 attention 分数: S_approx = Q_recent @ K^T / sqrt(d)  # shape: [w, n]
  3. 沿近端 query token 聚合: S_agg = mean(S_approx, dim=0)  # shape: [n]
  4. 分出 prefix + local tokens（始终保留）: prefix=[0:4], local=[n-64:n]
  5. 剩余 token: S_remain = S_agg[4:n-64]
  6. 选择 top-(k_v + k_s) 个最大 S_remain 对应的 token 索引
  7. 将这些索引映射为 vertical columns（全局可见）和 slashes（对角线偏移）
  8. 仅对所选 QK pairs 计算 attention，使用 FlashAttention 的块稀疏模式
  Output: O = attention(Q, K, V) with sparsity = 1 - (selected_pairs / n^2)
  ```

  **Quest decoding 伪代码**：
  ```
  Input: q ∈ R^{d}, KV cache with page_size=16
  1. 将全 KV cache 分为 pages: pages = chunk(KV_cache, page_size)
  2. 对每个 page p 计算: K_min_p = min(p.keys, dim=0), K_max_p = max(p.keys, dim=0)
  3. 计算 page 级近似分数:
     S_approx[p] = max(|q·K_min_p|, |q·K_max_p|) / sqrt(d)
  4. 选择 top-k pages: selected_pages = topk(S_approx, k=token_budget/page_size)
  5. 对 selected_pages 内的所有 token 计算精确 attention
  6. 保留全部 KV cache（不 eviction）
  Output: O with sparsity = 1 - (selected_tokens / total_tokens)
  ```



- 属于算法pipeline的实现是什么？实验比较什么？
  提出 TailorKV，一个混合 KV Cache 压缩框架，核心算法包括：(1) **Offline Identification（离线层分类）**：定义 dense preference score $P = n_q - \sum_{(i,j) \in \hat{\mathcal{I}}} \hat{\mathbf{A}}_{i,j}$，其中 $\hat{\mathbf{A}} = \operatorname{Softmax}(\mathbf{Q}_{\operatorname{last\_q}} \mathbf{K}^{\top} / \sqrt{d_h})$ 使用最近 $n_q$ 个 query 和全部 key 计算 attention，$\hat{\mathcal{I}}$ 为 Top-k attention score 位置集合。若 $P_l > \tau$ 则层 l 为 quantization-friendly（密集注意力分布，适合量化），否则为 sparsity-friendly（稀疏注意力分布，适合动态检索 Top-K tokens）。(2) **Static Quantization（静态量化）**：对 quantization-friendly 层使用 per-channel key 量化 + per-token value 量化，支持极低 1-bit/2-bit 精度，group size=64。(3) **Dynamic Retrieval（动态检索）**：对 sparsity-friendly 层，利用 inter-layer similarity 在当前层 l-1 预估层 l 的 query $\hat{\mathbf{q}}^{(l)} = \mathbf{W}_q^{(l)}(\mathbf{h}^{(l-1)})$，计算 channel 重要性 $s_i = |\hat{\mathbf{q}}_i| \cdot \max(|\mathbf{K}_i|)$，选择 critical channels 对应的 critical key cache 从 CPU 预取到 GPU，在 GPU 上近似 attention scores 后选出 Top-K tokens 从 CPU 获取。

  实验比较：(1) LongBench (13 子任务聚合为 6 类)、InfiniteBench (9 子任务聚合为 5 类)、RULER (13 任务，4K-128K 长度) 上的 task accuracy，与 StreamingLLM、SnapKV、Quest、PQCache 对比；(2) 不同 layer 量化策略的消融（Figure 9a）——仅量化 layer 0（dense）性能最佳，量化 sparsity-friendly 层性能骤降；(3) 动态 vs 静态 channel selection 对比（Figure 9b）——动态选 critical channels 优于离线静态校准；(4) 不同 critical channel 数量 (2/4/8/12) 对性能和延迟的消融（Figure 9c）；(5) 与 SimLayerKV（另一混合方法）的对比（Table 7），TailorKV 在 34.2× 压缩率下与 SimLayerKV 1.53× 压缩率性能相当；(6) Peak GPU memory usage 对比（Figure 7）——128k 序列下相比 Full Cache 降低约 73.8%；(7) 端到端 decoding latency 对比（Table 4, Table 13）。

- 硬件平台是什么，配置是什么。
  两个配置：(1) NVIDIA RTX 3090 (24GB 显存, PCIe 1.0 ×16, 4GB/s) + Intel Xeon Gold 6240 CPU (64GB RAM)。(2) NVIDIA A100 (80GB 显存, PCIe 4.0 ×16, 32GB/s) + Intel Xeon Platinum 8369B CPU。推理精度 FP16/BF16，prefill 阶段结合 4-bit AWQ weight-only 量化。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct (128K context, GQA, 32 layers, Q={0})、Llama-2-7B-32K-Instruct (32K context, MHA, 32 layers, Q={0,1})、Yi-6B-200K (200K context, GQA, 32 layers, Q={0,1})、Yi-9B-200K (200K context, GQA, 48 layers, Q={0,1})。Dataset/Benchmark：LongBench（Qspr/MulFi/HQA/WMQA/GRpt/MulN/TREC/SMSM/TriQA/Repo/LCC/PsgC/PsgR）、InfiniteBench（R.PK/R.Num/En.Dia/Sum/En.MC/En.QA/Zh.QA/Math.F/Code.D）、RULER（N-S1~N-S3/N-MK1~N-MK3/N-MV/N-MQ/VT/CWE/FWE/QA-1/QA-2）。Synthetic Longbench 用于离线确定 τ=0.2。超参：LongBench 用 8 critical channels, 64 local + 128 topk tokens；InfiniteBench/RULER 用 12 critical channels, 128 local + 896 topk tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/ydyhello/TailorKV（ACL 2025 Findings，代码已发布）。依赖：torch==2.4.0, flash-attn==2.6.3, dgl, transformers==4.46.1。MIT License。

  算法流程（伪代码）：
  ```
  # === Offline Identification（离线阶段）===
  # 输入：校准数据集 prompt，模型权重
  # 输出：每层类型 C(l) ∈ {Quantization-Friendly, Sparsity-Friendly}

  for layer l in 0..L-1:
      Q_last_q = recent n_q query vectors at layer l  # shape: (n_q, d_h)
      K = all key vectors at layer l                   # shape: (n, d_h)
      A_hat = Softmax(Q_last_q @ K.T / sqrt(d_h))     # shape: (n_q, n)
      I_hat = {(i, Top_k(A_hat[i,:], k)) for i in 1..n_q}
      P_l = n_q - sum(A_hat[i,j] for (i,j) in I_hat)   # Eq.(8), dense preference score
      if P_l > τ:  # τ=0.2
          C(l) = Quantization-Friendly
      else:
          C(l) = Sparsity-Friendly

  # === 推理阶段 decode step ===

  # Quantization-Friendly 层：静态 1-bit KV cache 量化
  # Key: per-channel quantization, Value: per-token quantization
  # 量化公式 (Eq.4): X_Q = clamp(round((X - z) / s), 0, 2^b - 1)
  # b=1 → 1-bit, group_size=64, zero_point z 和 scaler s 存 FP16

  # Sparsity-Friendly 层：动态检索
  # Stage 1 (at layer l-1): 预估 critical channels → prefetch critical key cache
  q_hat = W_q[l] @ h[l-1]               # inter-layer 预估 query (Eq.13)
  s_i = |q_hat_i| * max(|K_i|)          # channel 重要性 (Eq.10), i=1..d_h
  critical_channels = Top_d_s(s)        # 选 d_s=8 或 12 个 critical channels
  # 从 CPU 异步 prefetch critical key cache (double buffering)

  # Stage 2 (at layer l): 近似 attention → 选择 Top-K tokens
  q = W_q[l] @ h[l]                     # 当前层真实 query
  q_critical = q[critical_channels]     
  K_critical = K[critical_channels]     # 已预取到 GPU
  a_approx = q_critical @ K_critical.T  # 近似 attention scores
  topk_indices = Top_K(a_approx, k=n_topk)
  # 从 CPU 异步 fetch Top-K tokens 的完整 key/value (唯一不可 overlap 的操作)
  
  # 完整 attention: 使用 n_local GPU tokens + n_topk CPU tokens
  ```
