## Scale-invariant Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Scale-invariant Attention：一种对 attention logits 施加位置依赖的乘性缩放和加性偏置的算法，使注意力机制满足两个性质——scale-invariant total attention（每个 token range 内的总注意力渐进恒定）和 scale-invariant attention sparsity（注意力稀疏性随上下文变长而增长）。变换形式为 $L_t = a_t \bar{L}_t + m_t$，其中 $a_t = \sqrt{2[\log(t/\tau+1) - \log\alpha + \beta/\alpha]}$，$m_t = -a_t^2 + \beta/\alpha$，施加边界条件后 $\alpha = \beta = e^{0.5}$，唯一超参数 $\tau=10$。该变换与 p-RoPE 结合使用（称为 scale-invariant p-RoPE），实现从短上下文（4k）零样本泛化到长上下文（64k）而无需额外长上下文训练。
  
  实验比较：在 GPT-2-style 162M/304M 模型和 Llama 2 7B 上，对比 RoPE、p-RoPE、NoPE、RoPE+NTK、YaRN、LogN+RoPE、LogN+p-RoPE、LogN+NTK、ALiBi、Infini-attention 等基线方法，评估验证 loss（in-distribution 和 zero-shot length generalization）以及 needle-in-a-haystack 长上下文检索准确率。

- 硬件平台是什么，配置是什么。
  162M 模型：单卡 A100 80GB GPU；304M 模型：4×H100 Grace Hopper 节点（DDP）；Llama 2 7B continual pretraining：论文未明确说明 GPU，使用 Torchtune 库训练。

- 模型是什么。数据集和bench分别是什么。
  模型：GPT-2-style（modded-nanogpt 变体，使用 RMSNorm、ReLU² 激活、QK-Norm），162M 参数（12 layers, hidden 768, 6 heads）和 304M 参数（16 layers, hidden 1024, 8 heads）；Llama 2 7B（continual pretraining）。数据集：FineWeb（10B token subset 用于 162M，100B subset 用于 304M，实际使用 ~10B tokens）。Bench：语言建模验证 loss（4k/16k/64k context lengths），needle-in-a-haystack 检索任务（使用 C4 数据集构造）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：论文提供补充材料中的代码，基于 modded-nanogpt（MIT 协议）实现。使用 FlexAttention 实现 scale-invariant attention 和 ALiBi。算法核心：
  ```
  # 给定查询 q (dim d)，键矩阵 K (seq_len T × d)，位置距离 t ∈ [1, T]
  # 标准 attention score: S_t = (1/√d) * Σ_λ q_λ * K_{t,λ}
  
  # 超参数: τ = 10, α = β = e^{0.5}
  # 计算位置依赖的 a_t 和 m_t:
  f_t = log(t/τ + 1) - log(α)
  a_t = sqrt(2 * (f_t + β/α))
  m_t = -a_t^2 + β/α
  
  # 变换 logits:
  L_t = a_t * S_t + m_t      # S_t 是已应用 p-RoPE 后的 score
  
  # 标准 softmax attention:
  A_t = softmax(L_t)         # 在 t=1..T 上归一化
  output = Σ_t A_t * V_t     # 加权和值向量
  ```
  关键特性：当 t 较小时（局部上下文），$a_t^2 \approx 1$, $m_t \approx 0$，近似标准 attention；当 t 增大时，$a_t^2$ 对数增长（使分布更尖锐/sparse），$m_t$ 对数下降（压低远距离 token 的总体权重），实现局部稠密、全局稀疏的 attention 模式。

- 属于算法pipeline的实现是什么？实验比较什么？
  本论文是一篇 revisit/survey 型工作，非提出新算法，而是从实际生产部署角度全面评估四类代表性 KV cache 压缩算法的吞吐、响应长度分布和 negative sample 表现。被评估的实现包括：
  - **KIVI**（量化类）：per-channel key quantization + per-token value quantization。关键参数 group_size=G=32, residual_length=R=128（保留最近 128 tokens 为全精度）。开源：https://github.com/jy-yuan/KIVI
  - **GEAR**（量化类）：用 low-rank matrix 近似量化误差 + sparse matrix 处理 outlier。关键参数 sparsity_ratio=s=2%, rank=r=2%。开源：https://github.com/opengear-project/GEAR
  - **StreamingLLM**（稀疏类）：仅保留 initial tokens (64) + recent tokens (448)，总计 KV cache 大小 = 512。无动态 eviction 计算，结构化计算模式。
  - **H2O**（稀疏类）：基于 accumulated attention scores 动态 evict KV cache。heavy hitter oracle token size=64 + recent size=448，总计 cache size=512。

  实验比较：
  (a) Prefill/Decoding 吞吐 vs FP16 Baseline，在 TRL、TRL+FlashAttention、LMDeploy（含 PagedAttention+FlashAttention）三种框架下，batch size 1~32，prompt length 512~8192。
  (b) 不同 tensor parallelism (TP=1/2/4) 下的相对加速比。
  (c) 响应长度分布差异：比较压缩算法 vs Temperature=0.9/1.1 对输出长度的影响，评估 verbose output 现象。
  (d) Negative sample 分析：使用 LongBench 在 LLaMA-3.1-8B-instruct 和 Mistral-7B 上分析个体样本的精度退化，按 task type（Summarization/QA/Code）分类统计。
  (e) 吞吐预测器（Throughput Predictor）精度：基于 Vidur 框架 profiled attention operator runtime，预测不同 batch×seqlen 的组合。
  (f) 请求路由器（Request Router）：结合吞吐+长度预测器路由请求以最小化端到端延迟。

- 硬件平台是什么，配置是什么。
  主要：4× NVIDIA A6000 (48GB) 通过 NVLink 互联，Intel Xeon Gold 6326 CPU @ 2.90GHz。
  部分实验扩展至：NVIDIA H800 GPU（LLaMA-70B 实验，Figure 2）。
  框架：PyTorch 2.1.2, Transformers 4.43.1, FlashAttention 2.5.6, LMDeploy v6.0.1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-2-7B, LLaMA-2-13B, LLaMA-2-70B, LLaMA-3.1-8B-instruct, Mistral-7B-v0.1。
  数据集/Benchmark：ShareGPT（吞吐分析和长度分布实验，1000 样本子集，max generation tokens=1024），LongBench（negative sample 分析，覆盖 multi-document QA、single-document QA、summarization、few-shot learning、code completion、synthetic tasks）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  论文开源：https://github.com/LLMkvsys/rethink-kv-compression（含 throughput/length predictor、negative sample benchmark、LMDeploy 集成代码）。

  以 KIVI (per-channel key quantization) 为例说明算法 pipeline：
  ```
  # Prefill 阶段：正常计算 KV cache，不量化
  X_K = X @ W_K      # [b, l, d]，全精度存储
  X_V = X @ W_V      # [b, l, d]，全精度存储

  # Decoding 阶段：每步新 token 的 K/V 量化后追加
  for each decode step t:
      x_k = x @ W_K   # [b, 1, d]
      x_v = x @ W_V   # [b, 1, d]
      # Key: per-channel 量化，group_size=G=32
      for c in range(0, d, G):
          x_k_quant[c:c+G] = quantize_per_channel(x_k[c:c+G])  # → INT4
      # Value: per-token 量化
      x_v_quant = quantize_per_token(x_v)                      # → INT4
      # 保留最近 R=128 tokens 为全精度
      append_to_kv_cache(x_k_quant, x_v_quant)
      # Attention 计算时 dequantize
      scores = Q @ dequantize(K_quant)^T / sqrt(d_head)
      output = softmax(scores) @ dequantize(V_quant)
  ```

  以 H2O (accumulated attention score eviction) 为例说明算法 pipeline：
  ```
  # Prefill 阶段：正常计算 attention，累积 attention scores
  scores = Q @ K^T / sqrt(d_head)     # [b, heads, l, l]
  attn_scores_sum = scores.sum(dim=-2) # [b, heads, l]，累积每 token 被 attend 的分数

  # Decoding 阶段：每步动态 evict
  for each decode step:
      # 计算当前 attention scores 并累积
      scores = Q @ K^T / sqrt(d_head)
      attn_scores_sum += scores.sum(dim=-2)

      # 保留 heavy hitter (top 64) + recent (448)
      important_idx = topk(attn_scores_sum, k=64)
      recent_idx = last_n_tokens(448)
      keep_idx = union(important_idx, recent_idx)

      # Evict: 删除不在 keep_idx 中的 KV cache entries
      K = K[keep_idx]
      V = V[keep_idx]
  ```
  H2O 的 eviction 计算需要 multi-pass attention（为计算 importance metric），与 FlashAttention 的单 pass 设计不兼容，导致额外内存访问开销。
