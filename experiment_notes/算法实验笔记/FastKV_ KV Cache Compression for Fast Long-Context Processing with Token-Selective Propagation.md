## FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

- 属于算法pipeline的实现是什么？实验比较什么？
  FastKV 提出一种解耦预填充计算与 KV Cache 压缩的两阶段推理框架，核心包含：(1) Token-Selective Propagation (TSP)：在模型中间层（如 LLaMA-3.1-8B 的 layer 15）根据基于注意力权重的 saliency score 选择关键 token 向后层传播，早期层保持完整上下文计算；(2) Layer-wise KV Retention：每个 decoder layer 独立根据注意力重要性分数压缩 KV cache，KV 保留率与预填充计算解耦（两个独立超参数：TSP rate 和 KV retention rate）。实验比较 LongBench（16 个子任务，含单文档 QA、多文档 QA、摘要、少样本学习、合成任务、代码补全）、RULER（检索/聚合/多跳追踪，最长 128K）和 Needle-in-a-Haystack（16K-128K）上的准确率，以及单张 A100 SXM GPU 上端到端时延（预填充 + 256 token 解码）的加速比。Baseline 包括 StreamingLLM、H2O、SnapKV（仅解码加速）和 PyramidInfer、GemFilter（预填充感知加速）。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 SXM GPU（单卡）；FlashAttention-2 kernel。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-3.1-8B-Instruct（32 decoder layers，GQA，支持 128K context window）、Ministral-8B-Instruct（36 decoder layers，GQA，128K context window）、Mistral-Nemo-12B-Instruct（40 decoder layers，GQA，128K context window）。数据集/benchmark：LongBench（英文子集 + 代码，16 个子任务：NrtvQA、Qasper、MF-en、HotpotQA、2WikiMultihopQA、MuSiQue、GovReport、QMSum、MultiNews、TREC、TriviaQA、SAMSum、LCC、RepoBench-P、PassageCount、PassageRetrieval-en）、RULER（11 项子任务，context length: 8K/16K/32K/64K/128K）、Needle-in-a-Haystack（16K-128K，步长 16K）。标定数据集：论文未明确说明具体标定数据，仅说明使用 Equation 3 基于少量标定输入的 hidden state L2 距离自动选择 TSP 层。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/dongwonjo/FastKV。基于 HuggingFace Transformers 的 self-attention 实现 + FlashAttention-2 kernel 集成。

  **算法 pipeline 详解（两阶段预填充 + 独立 KV 压缩）：**

  **阶段一 —— 完整上下文预填充（Layer 0 到 TSP Layer L_TSP）：**
  ```
  for l = 0 to L_TSP:
      X, Att_l, K_X, V_X = layer_l(X)                        # 完整上下文注意力，构建 (N_I, d) 的 K_X, V_X
      K, V = KV_Compress(K_X, V_X, Att_l, R_KV)               # 每层基于 attention score 独立压缩 KV cache
      if l == L_TSP:
          # 计算 saliency score：使用 window tokens (N_obs=8) 作为 query
          S_i^{l,h} = Pooling(Σ_{n=0}^{N_obs} Att_l[h, N_I - n, i + m])  # Eq(1): MaxPooling kernel_size=7
          S_i^{TSP_layer} = (1/H) * Σ_h S_i^{TSP_layer,h}                 # Eq(2): 跨 head 均匀平均
          I_TSP = TopK(S^{TSP_layer}, N_I * R_TSP) ∪ window_indices      # 选取 top-R_TSP 的 token + window tokens
          x = X[I_TSP]                                                     # 仅传播选中的 hidden states (shape: N_I*R_TSP, d)
  ```
  其中 `R_TSP = 0.2`（默认），`N_obs = 8`（观察窗口大小），pooling kernel size = 7。

  **阶段二 —— 压缩上下文预填充（TSP Layer+1 到 Last Layer）：**
  ```
  for l = L_TSP+1 to L-1:
      x, Att_l, K_x, V_x = layer_l(x)                         # 仅在压缩后的 hidden states x 上计算注意力
      K, V = KV_Compress(K_x, V_x, Att_l, R_KV)               # 对压缩上下文同样执行 KV 保留
  ```

  **KV_Compress 核心逻辑：**
  ```
  KV_Compress(K_X, V_X, Att_l, R_KV):
      # 基于 group-wise saliency score（head-wise saliency 在 KV group 内平均）
      for each KV group g:
          S_g = (1/|heads_in_group|) * Σ_{h∈g} S_h           # head-wise 聚合为 group-wise
      I_KV = TopK(S, context_length * R_KV)                    # 按 R_KV 比率选择 top critical tokens
      return K[I_KV], V[I_KV]
  ```

  **TSP Layer 自动选择（Eq 3）：**
  ```
  L_TSP = argmin_{L ≤ L_max} (1/N) Σ_{i=1}^{N} ||H_i - H'_{L,i}||₂²
  ```
  其中 H_i 为完整上下文下最终层 hidden state，H'_{L,i} 为在 layer L 处应用 TSP 后的最终层 hidden state。L_max 约束防止 TSP 层过晚导致预填充节省有限。LLaMA-3.1-8B 选择 layer 15（共 32 层），Ministral-8B 选择 layer 17（共 36 层），Mistral-Nemo-12B 选择 layer 19（共 40 层）。

  **解耦设计核心：** TSP rate（控制预填充计算量）与 KV retention rate（控制解码时 KV cache 大小）完全独立。TSP rate=20% → 预填充计算率 60%，KV retention rate 可独立设为 10% 或 20%。

  **数值结果：**
  - LLaMA-3.1-8B，128K 上下文：预填充加速 1.82×，解码加速 2.87×
  - Ministral-8B，128K 上下文：端到端加速 >2×
  - LongBench 平均准确率（LLaMA，TSP=20%, KV=20%）：49.07 vs Full-context 50.19（下降 1.12 个百分点）
  - Token importance estimation 开销（128K 上下文）：0.15s，仅占预填充总时延 0.88%
  - Needle-in-a-Haystack（LLaMA，KV=10%）：FastKV 99.9 vs Full-context 99.0（TSP 帮助模型聚焦全局关键 token，甚至超越完整上下文）
