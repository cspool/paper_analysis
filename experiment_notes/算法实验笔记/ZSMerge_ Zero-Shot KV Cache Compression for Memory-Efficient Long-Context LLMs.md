## ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 ZSMerge —— 一种零样本、无参数的动态 KV Cache 压缩框架，包含四个核心组件：(1) **三分区 budget 分配**：将总缓存预算 B 划分为 proximity component (Bp，保留最近 token)、context component (Bc，保留高分 token)、residual component (Br，动态合并历史被驱逐 token)，B = Bp + Bc + Br；(2) **贡献评估**：每个 token t 在解码步 T 的贡献分数 s_t^(T) = λ·s_t^(T-1) + a_t^(T)，其中 a_t^(T) 为 attention 分数，λ=0.98 为指数衰减因子，类似 RL 信用分配；(3) **残差 token 合并**：将被驱逐的候选 token (k_t, v_t) 通过最大内积选择最兼容的 residual slot r̂ = argmax(k_r^T · k_t)，然后用增量均值聚合更新：k_r̂ ← (w_r̂·k_r̂ + k_t)/(w_r̂+1)，v_r̂ ← (w_r̂·v_r̂ + v_t)/(w_r̂+1)，w_r̂ ← w_r̂ + 1；(4) **补偿注意力评分**：修订注意力计算为 â_t^(T) = exp(q_T^T k_t/√d + α·log w_t) / Σ_i exp(...)，其中 w_i 为 token i 的融合计数（未压缩 w_i=1），α∈[0,1] 为 scale factor（论文固定 α=1），log w_t 补偿合并 token 的表示偏差，并保证 Theorem 1：未压缩 token i 的 â_i^(T) ≥ a_i^(T)（原注意力分数），即未压缩 token 保留相对注意力优势。

  实验比较：
  (1) Memory Reduction（§4.2.1）：LLaMA2-7B on A800-80GB，FullKV 在 54K tokens 时 OOM，ZSMerge 用 18K cache budget 保持 VRAM 恒定 43GB，消除 OOM；
  (2) Throughput（§4.2.2）：ZSMerge 维持 9 tokens/sec 解码速率（54K tokens），FullKV 从 9 降至 4 tokens/sec；在 13B model/4096+4096/batch 16 上 ZSMerge 是唯一可运行方法（178.2 tokens/s）；
  (3) Workload-Scalable 吞吐/延迟（Table 1/4）：7B 和 13B 模型，多种 seq length × batch size 组合，ZSMerge 在压缩场景下全面超越 H2O(5%) 和 LESS(5%)，经常超越 FullKV；
  (4) Numerical Error Analysis（§4.3）：残差合并 vs 纯驱逐的注意力输出相对误差，在 ≤20% cache size 下 ZSMerge 减少误差 60.5% (20%), 43.8% (10%), 37.4% (5%)，在 50% 压缩下减少 89.1%；
  (5) XSum ROUGE 质量（Table 2）：LLaMA2-7B 和 Falcon-7B，20%/10%/5% 三种 cache budget，ZSMerge 全面超越 H2O 和 LESS。5% 压缩下 LLaMA2-7B ROUGE-1 30.60 vs FullKV 30.59（基本无损），Falcon-7B 上 ZSMerge 保留超 50% baseline 性能而 LESS 几乎崩溃（ROUGE-1 7.75）；
  (6) LongBench 六类任务（Table 3/5）：LLaMA2-7B 和 Mistral-7B，cache size 512/1024，覆盖 CODE/FSHOT/MDQA/SDQA/SUMM/SYNC，ZSMerge 与 SnapKV 持平，远超 H2O 和 StreamingLLM；
  (7) InfiniteBench 100K+ tokens（Table 6）：LLaMA3-8B，8 个任务，ZSMerge 平均 52.95 vs FullKV 54.69（96.8%），超过 H2O (46.63)、OmniKV (50.33)、InfLLM (42.64)；
  (8) GSM-Infinite-8k（Table 7）：Qwen2.5-7B/Yi-1.5-6B/LLaMA-3.1-8B 三种架构，ZSMerge 整体匹配或超越 SnapKV；
  (9) Hyperparameter Sensitivity（§C.2）：LLaMA2-7B + XSum，sweep Bp/B (0.1-0.9), Br/(B-Bp) (0-0.08), α (0.0-1.0)，推荐 Bp/B=0.5, Br/(B-Bp)=0.02, α=1.0。

- 硬件平台是什么，配置是什么。
  NVIDIA A800-80GB GPU（核心实验平台）。使用 NVIDIA A800-80GB 进行所有吞吐、延迟和 OOM 测试。13B 模型实验也在同一平台完成。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA2-7B (MHA)、Falcon-7B (MQA)、Mistral-7B-Instruct (GQA)、LLaMA-3.1-8B-Instruct (GQA)、Qwen2.5-7B-Instruct (GQA)、Yi-6B/1.5-6B，共六种模型系列，覆盖 MHA/MQA/GQA 三种注意力机制。
  数据集：XSum（16k 新闻文章摘要）、LongBench（21 任务 6 类别，中英文，5K-15K tokens）、InfiniteBench（100K+ tokens，8 任务）、GSM-Infinite-8k（数学推理，3 难度级别）。
  Benchmark/指标：ROUGE-1/2/L（XSum 摘要质量）、LongBench 各任务准确率/得分、InfiniteBench 8 任务准确率、GSM-Infinite-8k symbolic/medium/hard 三级准确率。
  合成数据集：效率测试使用合成数据。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/SusCom-Lab/ZSMerge。基于 Transformers 库实现，仅替换 `scaled_dot_product_attention` 函数。支持 LLaMA、Falcon、Mistral 模型系列。评估使用 KVCache-Factory 框架 (https://github.com/Zefan-Cai/KVCache-Factory)。

  算法 Pipeline 伪代码：

  ```
  # 初始化（每个 attention head）
  B_p = proximity_budget    # 最近 token 保留数
  B_c = context_budget      # 高分 token 保留数
  B_r = residual_budget     # 残差合并 slot 数
  λ = 0.98                  # 衰减因子
  α = 1.0                   # 补偿 scale factor
  s = zeros(T)              # 贡献分数 [T]
  w = zeros(B_r)            # 融合计数 [B_r]
  K_r = zeros(B_r, d)      # 残差 key cache [B_r, d]
  V_r = zeros(B_r, d)      # 残差 value cache [B_r, d]

  # 解码步 T 的前向传播
  def zsmerge_forward(Q, K, V, T):
      # K ∈ R^{T×d}, V ∈ R^{T×d}, Q = q_T ∈ R^d

      # Step 1: 贡献评估（Eq. 5）
      attn_scores = softmax(Q @ K.T / sqrt(d))  # [1, T]
      s = λ * s + attn_scores[0]                 # 指数衰减累积

      # Step 2: Budget 分配与 cache 构建
      # Proximity: 最近 B_p 个 token
      idx_p = [T-B_p+1, ..., T]
      K_p, V_p = K[idx_p], V[idx_p]

      # Context: top-B_c 按 s 排序
      idx_c = topk(s[:T-B_p], B_c)
      K_c, V_c = K[idx_c], V[idx_c]

      # Residual: 将剩余 token 合并到 B_r 个 slot
      idx_evicted = remaining tokens not in idx_p ∪ idx_c
      for (k_t, v_t) in zip(K[idx_evicted], V[idx_evicted]):
          # 选择最兼容的 residual slot（Eq. 6）
          r_hat = argmax(K_r @ k_t)  # maximum dot product
          # 增量均值聚合（Eq. 7）
          K_r[r_hat] = (w[r_hat]*K_r[r_hat] + k_t) / (w[r_hat] + 1)
          V_r[r_hat] = (w[r_hat]*V_r[r_hat] + v_t) / (w[r_hat] + 1)
          w[r_hat] += 1

      # Step 3: 拼接压缩 cache（Eq. 4）
      K_B = concat([K_p, K_c, K_r], dim=0)  # [B, d]
      V_B = concat([V_p, V_c, V_r], dim=0)

      # Step 4: 补偿注意力计算（Eq. 8）
      # 构建权重向量 w_all: uncompressed=1, compressed=w[r_hat]
      scores = Q @ K_B.T / sqrt(d) + α * log(w_all)
      attn = softmax(scores)
      output = attn @ V_B
      return output
  ```

  关键设计要点：
  - 时间复杂度 O(T + B·d)，与纯驱逐方法（O(T)）相近，远优于 full attention 的 O(T²)
  - Br=0 时退化为纯驱逐策略（H2O-like）
  - 残差合并通过 Jensen 不等式保证 attention mass 守恒（Theorem 1）
  - Implementation 细节：prefilling 阶段可用 `window_size` 限制 s 初始化范围（同 SnapKV），大幅加速 prefill
