## Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

- 属于算法pipeline的实现是什么？实验比较什么？
  Dynamic-LLaVA 提出一种动态视觉-语言上下文稀疏化框架，通过两个可学习的轻量预测器（image predictor 和 output text predictor）在 MLLM 推理的 prefill 和 decoding 阶段分别稀疏化视觉 token 和语言 token。具体实现分为三部分：
  1. **稀疏化推理**（Sec 3.3.2）：在 prefill 阶段，image predictor 基于第 l 层（l=2）解码器输出的图像 token 特征，通过 `argmax(P^I(S_l^I))` 生成二值 mask M^I，丢弃不重要的图像 token（保留比例 r^I ≈ 20%）。在 decoding without KV cache 模式下，output predictor 类似地稀疏化输出文本 token 集合（保留比例 r^OT ≈ 50%）。在 decoding with KV cache 模式下，output predictor 对每个输出 token 生成一个二值决策 M^{OT}_{N^{OT}_l} ∈ {0,1}，决定是否将其 KV activations 加入 KV cache，实现在线 KV cache 压缩。
  2. **端到端稀疏化训练**（Sec 3.3.3）：训练时使用 MaskedSoftmax（Eq. 7）替代标准 Softmax，通过二值 mask 矩阵 G 隔离非必要 token 对重要 token 的影响，同时保持自回归并行训练。使用 Gumbel-Softmax + Straight-Through Estimator（STE）解决 argmax 不可微问题（Gumbel temperature τ 从 1 指数衰减至 0.1）。加入约束正则项 R（Eq. 10）约束 mask 的保留比例接近预定义的 r^I 和 r^OT，仅对输出长度 ≥ LEN^{OT}=50 的样本进行语言稀疏化训练。
  3. **批量并行稀疏化推理**（Appendix A.1）：通过 Left Padding + TopkArgmax 策略实现 mini-batch 内的并行预测和变长 token 集合的 GPU 批量计算。
  实验比较：(a) 视觉理解 benchmark 上与 SoTA 视觉上下文稀疏化方法（FastV、LLaVA-PruMerge+、VoCo-LLaMA、LLaVA-HiRED、IVTP、TRIM、SparseVLM）和高效视觉投影方法（TokenPacker、LLaVA-Resampler、C-Abstractor、Pixel-Shuffle、LDP-v2）比较准确率；(b) 生成能力 benchmark（LVIS-VQA single/multi-round、ShareGPT4V-VQA）上与 Random/Structure 静态丢弃、H2O KV cache 压缩、FastV+H2O 组合比较 PPL 和 METEOR；(c) 实际推理效率比较 prefill 时间、decoding 时间、GPU 内存（batch size=8）。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A100 (80G)。推理效率测量：1× NVIDIA A100 (80G)，batch size=8。延迟测量：1× A100 (80G)，batch size=1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaVA-1.5-7B（LLM: Vicuna-7B, 32 decoder layers, d=4096）、LLaVA-1.5-13B（LLM: Vicuna-13B, 40 decoder layers, d=5120）。Vision encoder: CLIP ViT-L/14@336px，生成 576 个 image token。
  训练数据集：656K Mixture Dataset（与 LLaVA-1.5 一致），仅使用含图像的数据训练 predictor。
  视觉理解 Benchmark：VQAv2、GQA、VizWiz、SciQA、TextVQA、POPE、MME、MMBench (en)、SEED (image)、MM-Vet、MMVP、RealWorldQA、CVBench-2D。
  生成能力 Benchmark：LVIS-VQA single-round（1000 样本，答案长度 >100 words）、LVIS-VQA multi-round（1000 样本，平均答案 >300 words，>7 轮交互）、ShareGPT4V-VQA single-round（178 样本，caption ≥300 words，平均输出 >1000 tokens）。
  评估指标：准确率（vision understanding）、PPL（生成流畅度）、METEOR（生成相似度）、TFLOPs（计算量）、GPU Memory（KV cache 开销）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Osilly/dynamic_llava。算法 pipeline 如下：

  **预训练阶段（基于 LLaVA-1.5 开源权重，冻结 vision encoder 和 projector，更新 LLM + predictors）**：

  ```
  # 超参数: l=2, r^I=0.2, r^OT=0.5, LEN^{OT}=50, λ=100
  # Gumbel temperature τ: 1.0 → 0.1 (指数衰减)
  # LLM lr=5e-6, Predictors lr=2e-4, batch_size=64

  for each training step:
      # Forward: 前 l 层 full computation
      S_l^I, S_l^T, S_l^OT = LLM_layers_1_to_l(image_tokens, text_tokens, output_tokens)

      # Image predictor (Vision Transformer blocks + MLP 512→256→128→2)
      D^I = P^I(S_l^I)                           # [N_l^I, 2]
      D^{I†} = GumbelSoftmax(D^I, τ)             # 松弛决策
      M^I = argmax(D^{I†})                        # [N_l^I], 二值mask

      # Output text predictor (MLP only)
      D^{OT} = P^{OT}(S_l^{OT})                   # [N_l^{OT}, 2]
      D^{OT†} = GumbelSoftmax(D^{OT}, τ)
      M^{OT} = argmax(D^{OT†})                    # [N_l^{OT}], 二值mask

      # 合并 mask: M = M^I ∪ {1}^{N^T} ∪ M^{OT}
      # 生成 mask 矩阵: G = {M}^{N_l} ∈ R^{N_l×N_l}, diag(G)=1

      # 修改 attention: 替换 Softmax 为 MaskedSoftmax
      # MaskedSoftmax(X_{i,j}, G) = exp(X_{i,j})*G_{i,j} / Σ_k exp(X_{i,k})*G_{i,k}

      # 后续层用 full token set 参与计算但 attention 被 mask 隔离
      loss = LM_loss + λ * R
      # R = ||sum(M^I)/|S_l^I| - r^I|| + ||sum(M^{OT})/|S_l^{OT}| - r^{OT}|| (if |S_l^{OT}| ≥ LEN^{OT})

      # Backward: STE 梯度 ∂L/∂D^{I†} → ∂L/∂M^I 直接传至 D^{I†}
  ```

  **推理阶段（prefill，Eq. 5）**：
  ```
  S_l^I, S_l^T = LLM_layers_1_to_l(image_tokens, text_tokens)
  D^I = P^I(S_l^I)                          # [N_l^I, 2]
  M^I = argmax_j(D^I)                       # j=0 discard, j=1 keep
  S_l^{I*} = {S_{l,i}^I | M_i^I == 1}      # 保留 ~20% image tokens
  S_l^{P*} = S_l^{I*} ∪ S_l^T               # 后续层仅处理缩减后的 token 集
  ```

  **推理阶段（decoding w/ KV cache，Eq. 6）**：
  ```
  Q, K, V = W^Q S_{l,N^{OT}}^{OT}, W^K S_{l,N^{OT}}^{OT}, W^V S_{l,N^{OT}}^{OT}
  O = W^O Attention(Q, S_l^K ∪ K, S_l^V ∪ V)
  M^{OT}_{N^{OT}} = argmax(P^{OT}(S_{l,N^{OT}}^{OT}))
  if M^{OT}_{N^{OT}} == 1: S_l^K ∪= K, S_l^V ∪= V   # 保留 KV
  else:                    S_l^K ∪= ∅, S_l^V ∪= ∅     # 丢弃 KV
  S_{l+1,N^{OT}}^{OT} = FFN(O)
  ```

  **实际效果（LLaVA-1.5-13B，1×A100 80G，batch=8，生成 2K tokens）**：
  Prefill Time: 0.83s (baseline) → 0.37s (Dynamic-LLaVA)，Decoding Time: 4117s → 2382s，GPU Memory (decode 2K): 58G → 42G。Image token 减少约 80%，Decoding TFLOPs 减少约 50%，GPU memory 减少约 50%。

- 属于算法pipeline的实现是什么？实验比较什么？
  提出一种基于加性向量量化（Additive Quantization）的 KV cache 压缩方法 CommVQ，将每个 token 的 key/value 向量作为整体进行量化，而非逐标量量化。核心设计包括三部分：(1) **编码器**：一个轻量级神经网络（线性层 + 激活函数 + 线性层），使用 Gumbel-Softmax 保证端到端可微，将 d 维 key/value 向量编码为长度为 N_c 的二进制序列 s_i ∈ {0,1}^{N_c}；(2) **码本解码**：通过 s_i × C 的矩阵乘法从码本 C ∈ R^{N_c×d} 中重建原始向量 t̂_i = s_i C，编码器和码本通过最小化 MSE loss 联合训练；(3) **RoPE-可交换码本**：利用 RoPE 矩阵的 2×2 块对角结构，设计满足 C = [[x, y], [-y, x]] 形式的 2×2 子码本 C_K^{jl}，使其与 RoPE 旋转矩阵 R_i^j 满足交换律 R_i^j C_K^{jl} = C_K^{jl} R_i^j，从而将 self-attention 中的 key-query 计算改写为 α_i = Σ_j,l (q^j R_t^j) C_K^{jlT} R_i^{jT} [s_i^j = l]^T，使得 (q^j R_t^j) C_K^{jlT} 可跨所有 token 复用，大幅降低解码开销。Key 码本通过 EM 算法在 FineWeb-Edu 校准集上优化（含 soft clustering center assignment 和 temperature annealing 技术）。Value 量化沿用原加法量化方法，但重排矩阵乘法为 Softmax(A) S_V C_V 以降低计算量。实验比较 LongBench、InfiniteBench、GSM8K、Needle-in-a-Haystack benchmark 上与 KIVI、KVQuant、VQLLM 在 2-bit 和 1-bit 量化下的准确率，以及量化误差 MSE。
- 硬件平台是什么，配置是什么。
  NVIDIA H100-80GB GPU（主要实验平台）；NVIDIA RTX 4090（验证单卡推理可行性）。LLaMA-3.1 8B 模型在 RTX 4090 上以 128K context length 运行。
- 模型是什么。数据集和bench分别是什么。
  主要模型：LLaMA-3.1-8B-Instruct（128K context）。额外模型：LLaMA-2-7B（32K context, Together.ai 版本）、Mistral-7B-v0.3（32K context）。校准/训练集：FineWeb-Edu 子集。Benchmark：LongBench（8 个子任务：Qasper, QMSum, MultiNews, TREC, TriviaQA, SAMSum, LCC, RepoBench-P）、InfiniteBench（10 个子任务：R.PK, R.Num, R.KV, En.Sum, En.QA, En.MC, En.Dia, Code.D, Math.F）、GSM8K、Needle-in-a-Haystack。
- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/UMass-Embodied-AGI/CommVQ。算法 pipeline 如下：

  1. **离线训练阶段**：在 FineWeb-Edu 校准集上运行 LLaMA 模型，收集各层的 K/V cache 向量。对 Key cache 使用 EM 算法按 2D 子空间训练 RoPE-可交换码本（Algorithm 1），对 Value cache 使用梯度下降训练编码器和码本（MSE loss + Gumbel-Softmax）。
  
  2. **Prefill 阶段**：正常计算 QKV 投影，将生成的 K/V 向量输入编码器 E，得到量化表示 s_i（Key 的 s_i^j ∈ {0,...,N_c'-1}^2，Value 的 s_i ∈ {0,1}^{N_c}），存储量化后的 KV cache S 替代原始 FP16 KV cache。
  
  3. **Decoding 阶段（Key 解码与 attention 融合）**：
     伪代码：
     ```
     q = x @ W_Q                              # [1, d]
     q_rope = apply_rope(q)                   # [1, d]
     # 预计算，可跨所有 token 复用
     q_pre = (q_rope @ C_K^T)                 # [1, N_c] 等价于 (q^j R_t^j) C_K^{jlT} 对所有 j,l
     # 对每个已缓存的 token i
     for i in range(num_cached_tokens):
         # R_i^T s_i^T 的计算，利用 RoPE 旋转的稀疏性
         alpha[i] = dot(q_pre, rope_rotate(s_i))
     # Softmax 后的 attention reordering
     attn_weights = softmax(alpha / sqrt(d))   # [1, N]
     # Value 解码重排
     output = (attn_weights @ S_V) @ C_V       # [1, N_c] @ [N_c, d] -> [1, d]
     ```
     关键优化：(qR_t) C_K^T 仅计算一次，后续每个 token i 仅需 R_i^T s_i^T 的轻量旋转操作。Value 解码由 O(d N_c N + dN) 降至 O(N_c N + d N_c)。
  
  4. **压缩率计算**：Avg. bit = N_c/d（Value），Avg. bit = R·log₂(N_c')/g（Key），总 KV cache 由 B×N×d×2×16 bits 降至 B×N×N_c×2×1 bits。LLaMA-3.1-8B（d=1024）的配置：
     - 2-bit: N_c=2048, R=21, N_c'=64, g=64, Avg. bit=2.00
     - 1-bit: N_c=1024, R=11, N_c'=64, g=64, Avg. bit=1.03
