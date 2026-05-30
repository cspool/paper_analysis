## Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  核心实现：(1) per-token magnitude-based unstructured pruning——对每个 token 的 KV cache vector 按元素绝对值排序，移除最低 magnitude 的元素达到目标稀疏度 s；(2) 对 Key cache：使用 per-token magnitude-based 或 output-aware 非结构化剪枝，探索 pruning direction（per-token vs per-channel）和 output-awareness 的影响；(3) 对 Value cache：per-token magnitude-based pruning 即等价于 output-aware per-token pruning（因 V 的每个元素乘以同一个 attention score），per-channel 则需额外计算 output-aware score；(4) 保留 local dense window（最近 32 token 不剪枝）；(5) 利用 bitmap-based 稀疏格式（扩展自 Coruscant）对剪枝后的稀疏 KV cache 进行最大程度压缩，每 tile 为 1×64 列，64-bit bitmap 表示非零位置，tile offset 寻址。

  实验比较：(a) Mustafar per-token magnitude-based unstructured pruning vs ThinK structured pruning，在 Key cache、Value cache、以及 Joint KV cache 上的 LongBench 精度；(b) 不同剪枝度 K_s/V_s=0.5, 0.7 下的精度退化；(c) unstructured vs 2:4 semi-structured sparsity 对比；(d) Mustafar 与正交压缩方法联合：+H2O token eviction, +KIVI (2-bit/4-bit KV cache quantization)；(e) 扩展至 Llama-2-13B-chat 大模型；(f) RULER benchmark 65K context 下 vs ThinK；(g) 80%/90% 极高稀疏度下的精度评估。

- 硬件平台是什么，配置是什么。
  效率评估：NVIDIA RTX 6000 Ada GPU（48GB VRAM）。精度评估：GPU 论文未明确说明型号（使用 HuggingFace Transformers 推理）。性能测量使用 NVIDIA Nsight Profiling Tool。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B（MHA）、Llama-3-8B-Instruct（GQA）、Mistral-7B-Instruct-v0.2（GQA）、Llama-2-13B-chat（MHA）、Llama-3.1-8B-Instruct（RULER 评测）。数据集/benchmark：LongBench（6 类任务：Single-Doc QA, Multi-Doc QA, Summarization, Few-shot Learning, Synthetic, Code）和 RULER（13 个任务含 Needle-in-a-Haystack, context 65,536 tokens）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/dhjoo98/mustafar

  **Mustafar 剪枝算法 pipeline（per-token magnitude-based，以 LLaMA-3-8B 单层 decode step 为例）**：

  ```
  Step 1 - 输入: Q_t ∈ R^{1×d}，当前 token query
         K_cache ∈ R^{T×d}，V_cache ∈ R^{T×d}（T 个已缓存 token 的全精度 KV）

  Step 2 - 剪枝 Key cache（per-token magnitude-based）:
    for each token i = 1..T-W（W=local window size=32）:
        abs_K_i = |K_cache[i, :]|          # element-wise absolute, shape [d]
        threshold = top_k_threshold(abs_K_i, sparsity=K_s)
        mask_K[i, j] = 1 if abs_K_i[j] >= threshold else 0
    # 最近的 W=32 token 的 mask 全部保留（mask[i, :]=1 for i > T-W）

  Step 3 - 剪枝 Value cache（per-token magnitude-based）:
    for each token i = 1..T-W:
        abs_V_i = |V_cache[i, :]|
        threshold = top_k_threshold(abs_V_i, sparsity=V_s)
        mask_V[i, j] = 1 if abs_V_i[j] >= threshold else 0
    # per-token magnitude 等价 output-aware（见 Figure 4 分析）

  Step 4 - 稀疏化 KV Cache:
    K_sparse[i] = K_cache[i] ⊙ mask_K[i]  # element-wise masked
    V_sparse[i] = V_cache[i] ⊙ mask_V[i]

  Step 5 - Bitmap 压缩（per tile = 1×64）:
    for each token i:
        for each tile t（每 64 个连续元素为一组）:
            bitmap = pack_bits(mask[i, t*64 : (t+1)*64])
            nonzeros = gather(K_sparse[i, t*64:(t+1)*64], mask)
            compressed[i].append((tile_offset, bitmap, nonzeros))
  ```

  剪枝公式：
  - Key cache per-token magnitude：S = |K_i|，按 |K_i| 排序保留 top-(1-s) 元素
  - Key cache output-aware：S = |K_i| ⊙ broadcast(Σ_{t} |Q_t|)，Q_t 累加当前和下31个 query
  - Value cache per-token magnitude（等于 output-aware）：S = |V_i|
  - Value cache per-channel output-aware：S = |V| ⊙ broadcast(Σ_{t} |α_t|)，α_t 为 attention score

  MHA 下 GQA 映射：多个 Q head 对应同一 KV pair 时，对每个 KV 的 Q 组求和所有剪枝分数。
