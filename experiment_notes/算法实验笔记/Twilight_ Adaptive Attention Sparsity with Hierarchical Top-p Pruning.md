## Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Twilight，一个hierarchical KV cache pruning框架，通过引入top-p (nucleus) sampling到sparse attention中，为任何现有sparse attention算法赋予自适应budget选择能力。核心算法包含三层：(1) **Token Selector**——将现有top-k sparse attention算法（如Quest、DS）作为黑盒，使用保守的大budget（如1/4 sparsity）预选token子集；(2) **Twilight Pruner**——基于top-p thresholding，将选出的token子集进一步剪枝：计算normalized attention weights，通过binary search找到最小预算B使得累积概率≥阈值p（通常p=0.85-0.95），仅保留top-p token；(3) **Sparse Attention Kernel**——仅对top-p token执行最终精确attention计算。

  实验比较：(a) Accuracy —— Longbench 12 tasks + RULER (long-context) + GSM8K/COQA/PG-19 (medium-context)，比较Quest/DS baselines在不同budget(256-8192)下 vs Quest-Twi/DS-Twi的自适应budget；(b) Efficiency —— self-attention operator speedup (FlashInfer-Twi vs FlashInfer, Quest-Twi vs Quest, vs FlashAttention2) + end-to-end decoding TPOT (batch=32-256)；(c) Ablation —— p threshold sensitivity (perplexity vs speed on PG-19/TrivialQA), time breakdown (TokenSel+Pruner+SparseAttn)。

- 硬件平台是什么，配置是什么。
  单张NVIDIA A100 GPU（40GB/80GB HBM）。Software: PyTorch, CUDA, OpenAI Triton, FlashInfer。Batch inference实验。

- 模型是什么。数据集和bench分别是什么。
  模型：Longchat-7B-v1.5-32k (MHA, 32k context)、LLaMA2-7B-Chat (MHA)、LLaMA-3.1-8B-Instruct (GQA, 128k context)。数据集：Longbench (12 tasks: Qasper, MF-en, HotpotQA, 2WikiMQA, Musique, GovReport, QMSum, MultiNews, TriviaQA, PR-en, LCC, Repobench-P), RULER (16k-96k), GSM8K (8-shot CoT), COQA, PG-19 (perplexity)。Efficiency datasets: Qasper, GovReport, LCC from Longbench (10k-30k prompts)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/tsinghua-ideal/Twilight

  Twilight hierarchical pruning pipeline（以Quest为base algorithm, p=0.85为例）：
  ```
  # 输入: query q ∈ R^{1×d}, FP16 K,V ∈ R^{N×d}, INT4 K_quantized ∈ R^{N×d/2} (额外存储)
  
  # Step 1: Token Selector (base algorithm, e.g., Quest with conservative budget B0=N/4)
  # Quest 使用per-page max-pooling估计token重要性
  K_pooled = max_pool(K, page_size=16)     # R^{N/16 × d}
  scores_approx = q @ K_pooled^T            # R^{1 × N/16}
  top_pages = topk(scores_approx, k=B0/16)  # 选择top pages
  I0 = expand_pages_to_tokens(top_pages)    # |I0| = B0 tokens
  
  # Step 2: Twilight Pruner (top-p sparsity)
  # 2a: SpGEMV with INT4 K cache —— 估计attention weights
  # K_quantized: per-head asymmetric INT4 quantization, paged layout
  q_fp16 @ K_int4^T → W_approx ∈ R^{1×N}   # 使用FlashInfer的SpGEMV kernel
  # Dequantization: K_fp16 ≈ (K_int4 - zero) * scale, per-head动态量化
  # INT4在shared memory中解包，cp.async异步加载，2-stage pipeline隐藏延迟
  
  # 2b: Normalize to get attention weights
  W_norm = softmax(W_approx[I0])            # 仅对I0中的token做softmax
  
  # 2c: Top-p via Binary Search (Algorithm 1, GPU并行)
  l=0, r=max(W_norm), m=(l+r)/2
  repeat:
    W0 = where(W_norm < m, 0.0, W_norm)     # mask below threshold
    if sum(W0) >= p: l = m                  # 累积概率足够，提高阈值
    else: r = m                              # 累积概率不够，降低阈值
  until max(W_norm[W_norm > r]) - min(W_norm[W_norm >= l]) < ε
  # 所有element-wise操作(max/where/sum)融合为单次GPU循环，tensorized执行
  I1 = indices where W_norm >= l            # |I1| = B1 << B0, 自适应budget
  M = mask[I1] = 1                          # 稀疏mask
  
  # Step 3: Sparse Attention Kernel (FlashInfer-based, 仅计算I1中的token)
  # 使用head-wise varlen attention for MHA, group-wise varlen for GQA
  # GQA处理: 同一query group内取各head选择token的union
  S = q @ K[I1]^T / sqrt(d)                # 精确FP16 attention scores
  P = softmax(S)                             # online softmax
  O = P @ V[I1]                              # 精确FP16 attention output
  ```

  关键设计要点：
  - Token Selector使用保守budget B0≈N/4保持高recall，Pruner用top-p做精确筛选
  - INT4 K cache降低SpGEMV的memory access至1/4（2-bit精度不足，8-bit浪费带宽）
  - Top-p binary search将排序O(N log N)降为O(log(range/ε))次并行reduction
  - Head-wise dynamic budget → 使用FlashInfer的load balancing (flatten head dim)处理不平衡
  - Extra memory: INT4 K cache = 1/8 × FP16 KV cache（可复用base algorithm已有的INT4 K cache）
  - p的选择比k更鲁棒：p代表累积概率，对不同分布head/layer/query的敏感度远低于k
