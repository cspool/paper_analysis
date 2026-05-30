## SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 SpindleKV，一种平衡浅层和深层 KV cache 压缩的方法。核心算法分为两部分：(1) 深层（deep layers）：基于注意力权重的 token eviction，使用金字塔形（pyramid-shaped）层间 KV cache 分配策略（随层数加深，保留比例线性递减）；(2) 浅层（shallow layers）：基于余弦相似度的 codebook token replacement，利用 KV cache 中 token 向量之间的高余弦相似性（constituent redundancy），对 KV cache 构建 codebook，仅存储 codebook 条目、每个 token 的索引（int 类型）和 L2 magnitude（float 类型），推理时通过 $Γ_r = C_Γ[r_Γ] \otimes m_Γ$ 重建。GQA 处理：将 KV head 展开（repeat $h_n$ 次），展开后使用 eviction + codebook 压缩。计算实际 KV cache 保留率 $r$ 时综合考虑 eviction 保留率 $r_1$、codebook 替换保留率 $r_2$ 和 dtype 转换比率 $r_3$。

  实验比较：(1) LongBench 16 个子任务（Single-Doc QA, Multi-Doc QA, Summarization, Few-shot Learning, Synthetic, Code）在多个 KV cache 保留率（~40%, ~30%, ~25%, ~20%, ~15%）下对比 PyramidInfer 和 PyramidKV；(2) Needle-in-a-Haystack 长上下文检索任务，15% KV cache 下对比 PyramidInfer 和 PyramidKV；(3) 额外 baseline 对比：H2O, SnapKV, StreamingLLM（LongBench on LLaMA3-8B-Instruct）；(4) 消融实验：GQA 集成策略（with/without repeat）、纯 codebook 压缩（无 eviction）、magnitude reconstruction 有效性。

- 硬件平台是什么，配置是什么。
  单卡 NVIDIA RTX 3090 GPU（推理速度测试使用，context length 4096, generation length 1000）。主要实验计算平台论文未明确说明具体配置。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA2-7b-chat（MHA，h=32），LLaMA3-8b-instruct（GQA，h=32, $h_n$=8, $h_g$=4），Mistral-7b-instruct-v0.2（GQA，h=32, $h_n$=8, $h_g$=4）。最大 context length 4K-32K。
  数据集/Benchmark：LongBench（16 个子集：narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_count, passage_retrieval_en, lcc, repobench-p），Needle-in-a-Haystack 检索任务。
  Baselines：PyramidInfer, PyramidKV, H2O, SnapKV, StreamingLLM。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/tyxqc/SpindleKV
  超参数：Key Threshold $\theta_K=0.98$, Value Threshold $\theta_V=0.95$, $\beta=0.05$, $\alpha=0.525$。

  算法流程（伪代码）：
  ```
  # Prefilling 阶段
  # 1. 对所有 prefill token 计算全量 attention
  # 2. 深层 eviction（Section 3.3）
  #    对每层 λ，计算保留率 r_c(λ) = r_c(0) + (r_c(m-1)-r_c(0))/(m-1)·λ
  #    按 accumulated attention score ac_i 选 Top-K token 保留
  #    KV_r = KV[argTopK(ac, k=⌊r_c(λ)×l_c⌋)]
  #
  # 3. GQA 展开（对 GQA 模型）
  #    将 KV head repeat h_n 次：K_expanded = K.repeat(h_n, ...)
  #
  # 4. 浅层 codebook 构建（Section 3.4, Algorithm 1）
  m_Γ = L2_Norm(Γ, dim=-1)           # 记录 magnitude
  Γ_r = Γ_r / m_Γ                    # 归一化
  S_Γ = cos_sim(Γ_r, Γ_r)            # 余弦相似度矩阵
  G_Γ = where(S_Γ > θ_Γ, 1, 0)      # 邻接矩阵
  C_Γ = []                           # CodeBook
  r_Γ = [-1, -1, ..., -1]            # 每个 token 的 codebook 引用
  while G_Γ != 0:
      s_Γ = sum(G_Γ, dim=1)          # 每个节点的度数
      ι = argmax(s_Γ)                # 选度数最高的 token
      C_Γ.append(Γ_r[ι])             # 加入 CodeBook
      η_ι = argwhere(G_Γ[ι] == 1)    # 找到该 token 可合并的邻居
      r_Γ[η_ι] = len(C_Γ) - 1        # 记录引用索引
      mask_Γ = matmul(¬G_Γ[ι]^T, ¬G_Γ[ι])
      G_Γ = G_Γ & mask_Γ             # 从图中移除已处理节点

  # 5. 推理时重建
  Γ_reconstructed = C_Γ[r_Γ] ⊗ m_Γ  # 从 CodeBook + magnitude 恢复
  # 对重建后的 K 重新应用 RoPE
  ```

  推理速度：FullKV 22.16 token/s vs SpindleKV 40% cache 18.39 token/s (LLaMA3-8B)，约 18% 额外开销。

- 硬件平台是什么，配置是什么。
  8 × NVIDIA H200（Tensor Parallelism = 8），CUDA 12.7，总 GPU TFLOPS 428.2，总 RAM 1123.2 GB，单 GPU 内存带宽 4052.8 GB/s，NVLink 带宽 478.1 GB/s，PCIe 5.0 x16。部分 MInference 对比实验在 8 × NVIDIA H100 上进行。vLLM 0.6.3.post1，enforce_eager=True，chunked_prefill=False。

- 模型是什么。数据集和bench分别是什么。
  主模型（base model）：Llama-3.1-70B-Instruct (BF16)、Llama-3.1-405B-Instruct-FP8（neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8）。
  推测器（speculator）：Llama-3.1-8B-Instruct (BF16)。
  数据集：LongBench（含 Single-Doc QA, Multi-Doc QA, Summarization, Few-Shot Learning, Code Completion, Synthetic 六个类别），RULER（含 NIAH variants, Multi-hop Tracking, SQuAD & HotpotQA, CWE & FWE），MMLU（Generative），IFEval，GSM8K（8-shot），HumanEval，MBPP，Arc Challenge，GPQA（8-shot）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/anonymous/speculative_prefill（匿名仓库，ICML 2025 发表时公开）。
  评估框架：LM-EVAL-HARNESS（标准短任务）、EVAL-PLUS（代码任务）。

  算法伪代码（Algorithm 1 from paper）：
  ```
  Require: Base model M, speculator S, look-ahead steps N,
           batch of mixed requests B, base model QKV cache C_b,
           speculator KV cache C_s
  1: B_p, B_d ← split_prefill_decode_requests(B)
  2: for i = 1 to N do                    # Sec 3.2.1 Look-ahead
  3:     B'_p ← model_forward(S, B_p, C_s, store_q=True)
  4:     B_p ← update_requests(B_p, B'_p)
  5:     B_p ← check_for_eos(B_p)
  6: end for
  7: if is_tensor_paralleled() then
  8:     tp_gather_qk(C_s)                # 收集 TP 分片的 Q,K
  9: end if
  10: Q, K ← retrieve_qk(B_p, C_s)
  11: A ← compute_attention_score(Q, K)    # shape: [N, L, S, H]
  12: A ← aggregate_attention_score(A)     # Sec 3.2.2: max over H,L, mean over N → [S]
  13: T ← chunk_select_from_smoothed_attention(A)  # Sec 3.2.3: 1D avg pool + chunk + Top-K
  14: P ← restore_pos_ids(T, B_n)         # Sec 3.2.4: 非连续 position IDs
  15: B ← merge_requests(T, P, B_p, B_d)
  16: Return model_forward(M, B, C_b)
  ```

  张量计算过程：对于 prompt 长度 M、look-ahead N 步、L 层、S 序列长、H 头数，注意力分数 $a_{ij} = \text{Softmax}(Q_{M+j} K^T)_i$（对第 j 个解码 token 的第 i 个 prompt token 的注意力）。聚合策略：$\text{score}(i) = \frac{1}{N}\sum_{j=0}^{N-1} \max_{l \in [0,L), h \in [0,H)} a_{ij}^{lh}$。然后对 score 序列做 1D average pooling 平滑，分 chunk 后取每个 chunk 内平均分数的 Top-K chunks，选中的 token 连同其原始 position IDs 送入主模型 forward。
