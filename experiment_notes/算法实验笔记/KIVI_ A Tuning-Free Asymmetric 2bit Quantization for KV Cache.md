## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- 属于算法pipeline的实现是什么？实验比较什么？
  - **KIVI**：免调优的 2bit KV Cache 非对称量化算法。核心设计：
    1. **Key Cache per-channel 量化**：分析发现 key cache 中少数固定 channel 具有极大 magnitude outlier，per-channel 量化可以将误差限制在每个 channel 内部，不影响其他正常 channel。实现上，由于 per-channel 量化跨 token，无法直接 append 到流式 KV cache，因此将 key cache 分为 grouped 部分（每 G 个 token 一组做 group-wise per-channel 量化）和 residual 部分（保留 FP16，最多 R 个 token）。
    2. **Value Cache per-token 量化**：value cache 无 outlier pattern，但由于 attention output 是 value cache 的加权求和（权重为稀疏 attention score），per-token 量化将误差限制在每个 token 内部，保证重要 token 不受其他 token 量化影响。实现上同样分为 grouped 和 residual 两部分。
    3. **Full precision sliding window**：residual 部分（最多 R 个 token）保持在 FP16，形成局部全精度滑动窗口。这对 GSM8K 等困难任务至关重要。
    4. 量化方式采用 group-wise round-to-nearest（公式：Q(X) = ⌊(X - z_X)/s_X⌉, X' = Q(X)·s_X + z_X），group size G=32，residual length R=128。
  - 实验比较：
    - **不同量化配置的 fake quantization 对比**：2bit (K per-channel, V per-token) vs 2bit (K per-token, V per-token) vs 2bit (K per-channel, V per-channel) vs 2bit (K per-token, V per-channel) vs 4bit per-token vs 16bit baseline
    - **KIVI-2 / KIVI-4 vs 16bit baseline**：在 Llama-2-7B/13B、Falcon-7B、Mistral-7B 上全面对比
    - **Ablation**：group size G∈{32, 64, 128}、residual length R∈{32, 64, 96, 128}
    - **Efficiency**：KIVI vs FP16 baseline 的峰值内存和吞吐量对比（ShareGPT 真实 workload）
    - **Long context**：LongBench 8 个子任务 + NIAH

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA A100 GPU（80GB）
  - 论文未明确说明具体的 CPU/内存配置

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Llama-2-7B、Llama-2-13B（multi-head attention）
    - Llama-2-7B-Chat、Llama-2-13B-Chat
    - Falcon-7B（multi-query attention，KV cache 仅单头）
    - Mistral-7B（multi-head attention）
    - Llama-3-8B-Instruct（group query attention，KV 8头）
    - Mistral-7B-Instruct-v0.2（group query attention，32K context）
    - LongChat-7B-v1.5-32K（32K context）
  - 数据集/Benchmark：
    - **LM-Eval**（normal context）：CoQA（EM accuracy）、TruthfulQA（BLEU）、GSM8K（EM accuracy）
    - **LongBench**（long context）：Qasper（F1，Single-Doc QA）、QMSum（ROUGE，Summarization）、MultiNews（ROUGE，Summarization）、TREC（Classification，Few-shot）、TriviaQA（F1，Few-shot）、SAMSum（ROUGE，Few-shot）、LCC（Similarity，Code Completion）、RepoBench-P（Similarity，Code Completion）
    - **Needle-in-a-Haystack (NIAH)**：passkey retrieval，使用 Paul Graham Essays 填充背景，7-digit passkey
    - **ShareGPT**：真实 LLM serving workload 效率测试，平均 prompt 长度 161、输出长度 338

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/jy-yuan/KIVI
  - **算法 pipeline 详解**：

  **Prefill 阶段（伪代码）**：
  ```
  输入: X ∈ R^{l_prompt × d}
  1. X_K = X·W_K, X_V = X·W_V  // 计算 key/value
  2. X_V_g = X_V[:l_prompt-R], X_V_r = X_V[l_prompt-R:]  // split value
  3. Q(X_V_g) = GroupQuant(X_V_g, dim=token, G=32)  // per-token group quant
  4. Q(X_K_g), X_K_r = KeyQuant(X_K)  // per-channel group quant + residual
  5. KV cache = {Q(X_K_g), X_K_r, Q(X_V_g), X_V_r}  // 存储量化缓存
  6. return X_K, X_V  // 传给下一层的是全精度
  ```

  **KeyQuant 函数**：
  ```
  procedure KeyQuant(X_K ∈ R^{l×d}):
    r = l % R           // 不能被R整除的余数
    X_K_g = X_K[:l-r]   // grouped 部分
    X_K_r = X_K[l-r:]   // residual 部分（FP16）
    Q(X_K_g) = GroupQuant(X_K_g, dim=channel, numGroup=l//G)  // 沿channel维度分组量化
    return Q(X_K_g), X_K_r
  ```

  **Decoding 阶段（伪代码）**：
  ```
  输入: KV cache, t ∈ R^{1×d}
  1. t_Q = t·W_Q, t_K = t·W_K, t_V = t·W_V
  2. X_K_r = Concat([X_K_r, t_K], dim=token)  // 新token加入residual
  3. X_V_r = Concat([X_V_r, t_V], dim=token)
  4. if len(X_K_r) == R:  // residual满了，量化并移入grouped
       Q(X_K_r) = KeyQuant(X_K_r)
       Q(X_K_g) = Concat([Q(X_K_g), Q(X_K_r)], dim=token)
       X_K_r = empty
  5. if len(X_V_r) > R:
       Q_outdated = GroupQuant(X_V_r[:-R], dim=token, G=32)
       Q(X_V_g) = Concat([Q(X_V_g), Q_outdated], dim=token)
       X_V_r = X_V_r[-R:]
  6. A = Concat([t_Q·Q(X_K_g)^T, t_Q·X_K_r^T], dim=token)  // tiled matmul
  7. A_g = Softmax(A)[:-R], A_r = Softmax(A)[-R:]
  8. t_O = A_g·Q(X_V_g) + A_r·X_V_r  // 混合精度 attention output
  9. return t_O
  ```

  **张量计算关键**：
  - Key cache: X_K ∈ R^{l×d}，沿 channel(dim) 维度分组量化，每 G=32 个 token 一组
  - Value cache: X_V ∈ R^{l×d}，沿 token(dim) 维度分组量化，每 G=32 个 channel 一组
  - Attention score: 使用 tiled matrix multiplication 分别计算 grouped quantized 部分和 residual FP16 部分，Concat 后 Softmax
  - Attention output: 按 residual 划分享 softmax 权重 A_g/A_r，分别与 quantized value 和 FP16 value 做矩阵乘法后求和
