## MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  MOM（Memory-efficient Offloaded Mini-sequence Inference）是一种针对长上下文 LLM 推理的内存高效算法，包含两项核心实现：(1) Mini-Sequence Processing——将 MLP 层的输入沿序列维度划分为 M 个 mini-sequences（每个大小 N≈S/M），逐个处理以降低中间激活内存，且仅对最后一个 MLP 层和 LM Head 处理最后一个 token 的表示；(2) KV Cache Offloading——在 prefill 阶段将 KV cache 从 GPU offload 到 CPU，decode 阶段重新加载回 GPU。Mini-sequence 仅作用于 MLP 和 LM Head，attention 层保持不变，因此可与 FlashAttention 和 GQA 等现有优化无缝集成。

  实验比较四种配置：(a) Standard（无优化）、(b) Offload Only（仅 KV cache offloading）、(c) Mini-sequence Only（仅 MLP mini-sequence 分区）、(d) MOM（Mini-sequence + Offloading 组合），以及 Chunked Prefill（chunk size=8192 和 512）。评估维度：峰值 VRAM 使用量、最大可扩展上下文长度、总推理延迟（prefill+decode）、TTFT（Time to First Token）、decode 速度（tokens/s）、准确率（logit equivalence + Needle-in-a-Haystack test）。

- 硬件平台是什么，配置是什么。
  主实验：单张 NVIDIA A100 80GB GPU，bfloat16 精度。额外实验：单张 RTX 4080 mobile 12GB GPU，bitsandbytes 4-bit 量化（context lengths [16000, 20000, 24000]）。

- 模型是什么。数据集和bench分别是什么。
  主要模型：Meta-Llama-3.2-8B（bfloat16）。额外模型：Qwen2.5-7B、Mistral NeMo (12B)、Llama3.2-3B（4-bit 量化）、Qwen2.5-3B（4-bit 量化）。Benchmark：Needle-in-a-Haystack（评估长上下文检索准确率，needle depth × context length 矩阵）、Logit Equivalence Test（随机输入验证输出 logits 完全一致）。Context lengths：A100 上 [48000, 80000, 112000, 144000]，最大可达 455000 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/TianyiZhu877/MOM（基于 HuggingFace Transformers，使用 transformers.cache_utils.OffloadedCache）

  **算法 Pipeline（Algorithm 1）**：
  
  输入：X ∈ R^{B×S×d}，Mini-sequence size C，offloaded KV cache K

  ```
  # 对每个 Transformer Block:
  A = Attention(X)                    # attention 层保持完整，使用 FlashAttention/GQA
  Update and offload KV cache: K ← offload(K, A)
  
  if last MLP layer:
      A_last = A[:, -1, :]            # 仅取最后一个 token 的表示 [B, d]
      O_last = MLP(A_last)            # 仅处理最后一个 token 的 MLP
      L = LM_Head(O_last)             # 仅对最后 token 计算 logits [B, vocab_size]
      Reload KV cache from CPU to GPU for decode stage
      return L
  else:
      M = ceil(S / C)                 # 划分 mini-sequences
      Partition A into {A_i}_{i=1}^M, each A_i ∈ R^{B×N×d}, N ≈ C
      for i = 1 to M:
          O_i = MLP(A_i)              # 逐个处理 mini-sequence
      O = concat([O_1, ..., O_M])     # 拼接输出
      return O
  ```

  **张量计算示例（Llama-3-8B, S=128K, C=8192, d=4096, I=4d=16384）**：
  ```
  # Standard: 中间激活 = S × I = 128K × 16384 ≈ 2.1B floats ≈ 4.2GB (bfloat16)
  # MOM: M = ceil(128K/8192) = 16, 中间激活 ≈ N × I = 8K × 16384 ≈ 131M floats ≈ 262MB
  # 内存节省：4.2GB → 262MB（约 16× reduction per MLP layer）
  ```

  **与 Chunked Prefill 的关键差异**：
  - Chunked Prefill 将整个 prefill（attention + MLP + LM Head）按 chunk 分多次前向，导致 forward-pass 重复开销
  - MOM 仅拆分 MLP 层，所有 mini-sequences 在单次前向 pass 中处理，attention 层保持完整序列计算，仅 MLP 逐 mini-sequence 执行

  **KV Cache Offloading 集成**：
  ```
  # Prefill: 每层 attention 后 offload KV cache to CPU
  K[layer], V[layer] → CPU (via OffloadedCache)
  # Decode: 所有层 KV cache 重新加载到 GPU
  K[all], V[all] → GPU
  # Decode 阶段使用 GPU 上完整 KV cache 进行 autoregressive generation
  ```
  Decode 阶段 KV cache 已全部在 GPU 上，因此 decode 速度几乎无退化（Table 4: MOM decode 25.712 tok/s vs Standard 25.804 tok/s @ 48K context）。
