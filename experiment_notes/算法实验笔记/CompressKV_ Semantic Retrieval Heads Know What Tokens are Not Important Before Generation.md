## CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 CompressKV，一种针对 GQA-based LLM 的 KV cache 压缩框架，包含三个核心组件：(1) **Semantic Retrieval Head (SRH) 识别**：不依赖传统的 top-1/top-k 精确命中标准，而是通过聚合 attention head 在整个 answer span 上的 attention scores 来评估 head 的语义检索能力，公式为 SemanticRetrievalScore(h) = Σ_{t=1}^N I[y_t ∈ A] Σ_{j∈A} a_{t,j}^h，其中 y_t 是生成 token，A 是 answer span，a_{t,j}^h 是 head h 在 token j 上的 attention weight。得分越高说明 head 越能捕捉语义信息而非仅 copy-paste 行为。(2) **SRH 驱动的 Token 选择**：每层选取 top-k SRH（默认为 top-4），将这些 head 的 attention score 矩阵在 observation window 上求和并在 token 维度上做 1D average pooling（kernel size=5），取平均后选出 top-N 高 attention 的 token 保留其 KV cache，其余 token 的 KV cache 被 evict。同一层内所有 head 共享统一的 token 索引集。(3) **Error-Aware 层级自适应 Cache 分配**：离线阶段在 LongBench 上模拟极端压缩（每层仅 32 tokens，约 0.3% 容量），计算每层的压缩误差 e^(l) = Σ_t ||O_comp,t^l - O_full,t^l||_F / (||O_full,t^l||_F + ε)，跨数据集归一化平均后得到最终重要性分数 ẽ^(l)。在线推理时按 ẽ^(l) 比例分配 cache budget，并设置 per-layer 上下界 [m=32, M=3×B_per-layer] 防止极端分配。实验比较 LongBench（16 个数据集）和 Needle-in-a-Haystack 上不同 KV cache budget（128/256/512/1024/2048）下与 StreamingLLM、SnapKV、PyramidKV、CAKE 的准确率，以及 masking-based ablation 比较 SRH 与传统 Retrieval Head 的重要性，和端到端延迟/峰值内存对比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（单卡），用于延迟和峰值内存评估。FlashAttention-2 默认启用。SRH 识别和误差分数计算可离线完成。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct（GQA，128K context）和 Mistral-7B-Instruct-v0.3（GQA，32K context）。Benchmark：(1) LongBench——16 个数据集，分 6 类：Single-Doc QA (NarrativeQA, Qasper, MultiFieldQA-en)、Multi-Doc QA (HotpotQA, 2WikiMultihopQA, MuSiQue)、Summarization (GovReport, QMSum, MultiNews)、Few-shot Learning (TREC, TriviaQA, SAMSum)、Synthetic (PassageCount, PassageRetrieval-en)、Code (LCC, RepoBench-P)；(2) Needle-in-a-Haystack (NIAH)——评估长上下文中检索隐藏答案的能力，测试 1K-128K 长度。评估使用 greedy decoding 保证公平比较。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/TUDa-HWAI/CompressKV.git。算法 pipeline 如下：

  1. **离线阶段 — SRH 识别**：在 LongBench 验证集上运行完整模型，对每层每个 head h 计算 SemanticRetrievalScore(h)。取每层 top-4 SRH 作为 token 选择的依据。
  
  2. **离线阶段 — 误差分数计算**：
     伪代码（Algorithm 1 核心逻辑）：
     ```
     for each layer l:
         # 模拟极端压缩：每层仅保留 32 tokens 的 KV cache
         O_full^l = Attention(Q^l, K_full^l, V_full^l) @ W_O^l
         O_comp^l = Attention(Q^l, K_comp^l, V_comp^l) @ W_O^l
         e^l = Σ_t ||O_comp,t^l - O_full,t^l||_F / (||O_full,t^l||_F + ε)
         ẽ^l = normalize(average_over_datasets(e^l))
     
     # 分配算法
     B_i = m  for all layers i
     R = B_total - Σ B_i
     B_i = clip(B_i + round(ẽ_i * R), m, M)
     # 贪心调整剩余/超出预算至满足 Σ B_i = B_total
     ```
  
  3. **在线 Prefill 阶段**：
     伪代码（Token 选择）：
     ```
     for each layer l:
         select top-k SRH heads (e.g., k=4)
         for each selected head h:
             # observation window 内的 attention scores
             A_h = attention_scores[h, :, -W:]  # [seq_len, W]
             # sum over observation window
             S_h = sum(A_h, dim=-1)  # [seq_len]
             # 1D average pooling (kernel=5)
             S_h = avg_pool1d(S_h, kernel_size=5)
         # average across selected SRH
         S = mean([S_h for h in selected_heads])
         # select top-N highest-scoring tokens
         keep_indices = topk(S, N)
         # retain KV pairs for keep_indices only
         K_cache = K[keep_indices]
         V_cache = V[keep_indices]
     ```
     参数：observation window = 8 tokens，pooling kernel size = 5，average pooling。
  
  4. **在线 Decoding 阶段**：使用压缩后的 KV cache 进行 attention 计算，新生成 token 的 KV pair 追加到 cache 中。cache 大小受 per-layer budget B_i 限制。

  5. **压缩率计算**：在 128K context 下，256 KV cache entries 仅占全量 KV cache 的 0.07%，仍能达到 NIAH 上 90% 的 full-cache 准确率。LongBench 上 19% KV entries 保持 >99% full-cache 性能。
