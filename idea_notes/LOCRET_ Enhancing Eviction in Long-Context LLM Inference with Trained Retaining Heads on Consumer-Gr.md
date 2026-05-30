## LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

- baseline方法是什么？
  Baseline 是现有 KV cache 淘汰方法（H2O、SNAPKV、SIRLLM）以及 offloading 方法（InfLLM）、量化方法（HF-2BITS）和 sparse attention 方法（MINFERENCE），其全栈执行例子如下：
  - **算法层**：H2O 基于累积 attention score 识别 heavy-hitter token——在 prefill 阶段计算完整的 attention score 矩阵，累加每 token 被各 query 关注的 attention weight 作为重要性分数，保留 top-k 高分 token。SNAPKV 使用 observation window 内的 query token 计算 attention-based 重要性并通过 voting 机制选择保留 token。SIRLLM 使用 token-level entropy 作为重要性度量进行 eviction。这些方法的**核心缺陷**：(a) H2O 和 SNAPKV 的重要性评分是 non-causal 的——依赖后续 token 的 attention score 来判断当前 token 的重要性，导致在 chunked prefill 中只能看到当前 chunk 时严重低估某些未来会被关注的 token 的重要性（local-global discrepancy），Figure 1 显示 H2O/SNAPKV 的 local-global consistency 远低于 LOCRET；(b) H2O 与 FlashAttention 不兼容——需要 materialize 完整的 attention score 矩阵，无法利用高效 attention kernel；(c) SIRLLM 虽为 causal 评分但准确性不足，在 R.PassKey 和 R.Number 等需要精确检索的任务上表现差（准确率 <4%）；(d) InfLLM 依赖 CPU offloading 存储全量 KV cache，CPU-GPU 通信成为瓶颈；(e) MINFERENCE 虽加速 attention 计算但不压缩 KV cache 大小，内存需求仍为全量 KV cache。
  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline + chunked prefill。FULLATTN 使用 vLLM（含 tensor parallelism）。SIRLLM 使用官方实现（含 CPU 重要性排序操作）。H2O 使用 layer-wise chunked prefill（chunk_size=1024，更大的 chunk 会 OOM）因需全序列 attention scores。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：FlashAttention-2 加速标准 attention（FULLATTN、MINFERENCE）。H2O 使用 PyTorch vanilla attention（不兼容 FlashAttention）。SIRLLM 在 GPU 上执行 TopK + index gather + CPU 排序。
  - **硬件架构层**：NVIDIA A800/H800/4090 GPU，无专用硬件修改。

  Baseline 核心缺陷总结：
  1. **Non-causal 重要性评分导致 local-global discrepancy**：H2O/SNAPKV 需要后续 token 信息才能准确评分，在 chunked prefill 中评分不准确，导致关键 token 被错误 evict。
  2. **与高效 attention kernel 不兼容**：H2O 需要 materialize attention matrix，无法使用 FlashAttention，prefill 极慢（464 tok/s vs LOCRET 5080 tok/s）。
  3. **准确性不足**：SIRLLM 的 token-entropy 评分在精确检索任务上失效；量化方法（HF-2BITS）在所有任务上严重退化。
  4. **不压缩 KV cache 大小**：MINFERENCE 通过 sparse attention 减少计算但不压缩 KV cache 存储，总内存仍为 model weights + full KV cache。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LOCRET 通过训练小型 retaining heads 预测 causal importance score (CIS)，实现 causal 的、准确的、FlashAttention 兼容的 KV cache eviction。

  LOCRET 的全栈执行例子：
  - **算法层（核心创新——Trained Retaining Heads + CIS-based Eviction）**：
    1. **训练式 Causal 重要性评分 → 解决 Baseline 缺陷 1（local-global discrepancy）**：
       在每层注入两层 MLP 作为 retaining head R，输入为 [Q, K, V] 拼接，输出为每 token 在各 head 的预测 CIS Ŝ。Ground truth 定义为 answer token 对该 prefix token 的最大 pre-softmax attention score：S[k]_j = max_p (Q_j K_j^T)_{p,k}。因 CIS 仅依赖当前及之前的 token（causal），一旦计算即不变，在 chunked prefill 中评分始终准确，无需等待后续 token。Figure 1 证明 LOCRET 的 local-global consistency 远高于 H2O/SNAPKV。

    2. **Stabilizers 机制 → 解决上下文不连续性**：
       每次 chunked prefill 后，将最后 n_s 个 token 的 CIS 强制设为 +∞（永不被 evict），保证局部连续上下文。Figure 3 消融实验：n_s=0 时模型完全失败（R.Number 准确率 0%），n_s=2500 时恢复。

    3. **与 FlashAttention 兼容 → 解决 Baseline 缺陷 2**：
       Retaining head 的 CIS 预测在前向 pass 中作为附加输出计算，不需要 materialize attention matrix。推理时使用 FlashAttention 加速全部 attention 计算。

    4. **LOCRET-Q 感知 Query → 解决 query-driven task 失效**：
       训练时将 query token 前置，CIS label 基于包含 query 的 attention。推理时 query 在序列首部确保所有 eviction 感知 query。RULER 上 LOCRET-Q 达 75.54%（vs LOCRET 34.33%）。

    5. **训练开销极小**：仅需 <1 GPU 小时（Phi-3-mini-128K: 0.47h, Llama-3.1-8B: 0.80h），可训练参数仅占 8% 和 2.5%。训练对各种超参数和数据集鲁棒（Figure 6, Table 15-17）。

  - **系统框架层**：基于 PyTorch + HuggingFace Transformers 实现 chunked prefill 推理。推理流程遵循 Algorithm 1：split chunks → per-chunk forward with retaining heads → concat KV cache + scores → Top-b selection → evict low-score units → stabilizers protected → final n_loc tokens → decoding。支持所有 decoder-only LLM（MHA/GQA）。

  - **编译框架层**：论文未明确说明。

  - **kernel调度层**：FlashAttention-2 加速 attention 计算。Retaining head 的小型 MLP 不引入明显延迟（Table 20: w/ R 19153 tok/s vs w/o R 20304 tok/s at 4096 ctx，差距来自系统波动非 overhead）。Cache eviction 的 TopK + gather 为 GPU 标准操作。

  - **硬件架构层**：NVIDIA A800/H800 GPU（训练+主要评估）；NVIDIA 4090 24GB（消费级设备验证）；无专用硬件修改。

  **对比 baseline 的关键差异**：
  - **H2O/SNAPKV (non-causal)** → LOCRET (causal by training)：H2O 在 ∞Bench chunked prefill 时 accuracy 从 FULLATTN 的 48.40 降至 21.18，LOCRET 保持 47.57。特别是 R.Number：H2O 3.39、SNAPKV 2.54 vs LOCRET 97.46。
  - **SIRLLM (token-entropy)** → LOCRET (trained CIS)：SIRLLM R.PassKey 1.69 vs LOCRET 100.00（Phi-3-mini-128K on 4090）。
  - **H2O (vanilla attention)** → LOCRET (FlashAttention)：4090 上 R.PassKey 128K 推理速度 LOCRET 5080 tok/s vs H2O 464 tok/s（~11× speedup）。
  - **MINFERENCE (full KV cache)** → LOCRET (compressed KV cache)：Peak GPU memory LOCRET 17.71 GB vs MINFERENCE 27.63 GB (LongBench, Table 9)。
  - **HF-2BITS (quantization)** → LOCRET (eviction)：∞Bench avg Phi-3-mini-128K: HF-2BITS 2.03 vs LOCRET 34.73。
  - **10M token context**：LOCRET 1747.6× 压缩比，R.PassKey 100% 准确率。
  - **LOCRET 与 pooling 方法正交**：LoCoCo + LOCRET combination 28.70 > LoCoCo 26.01 / LOCRET 27.96 (Table 11)。
  - **LOCRET 与 quantization 兼容**：LOCRET-4bits 平均退化仅 0.85 vs FULLATTN-4bits 退化 0.56 (Table 10)。
