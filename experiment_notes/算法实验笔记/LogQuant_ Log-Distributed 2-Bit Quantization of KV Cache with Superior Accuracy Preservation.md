## LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是一种 training-free 的 2-bit KV Cache 量化算法，利用对数分布的 token 选择策略保留重要 token 为全精度（BF16）而将其余 token 量化为 INT2，同时基于 position-agnostic 特性重组缓存以提升内存局部性。实验比较：LogQuant vs KiVi（主要 baseline）在 2-bit/4-bit 精度下跨多模型、多任务、多压缩比的准确性（GSM8K Exact Match + LongBench 6 类任务）和吞吐量/内存效率（H100 HuggingFace pipeline）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 48G MIG（单卡，用于效率 benchmark：HuggingFace pipeline，平均 prompt 长度 512，最大输出长度 2000，递增 batch size 记录峰值内存和吞吐量直到 48GB 上限）。准确性评估的 GPU 论文未明确说明具体型号，使用 HuggingFace transformers 推理 pipeline。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama3.1-8B-Instruct、Llama3-8B-Instruct (GQA)、Qwen1.5-7B-Chat (MHA)、Qwen1.5-7B-Chat-AWQ、Qwen1.5-14B-Chat-AWQ (MHA)、Qwen2-7B-Instruct (GQA, 仅保留 1/8 KV heads)、Phi-3-mini-128k-instruct (MHA, 3.8B)。
  数据集/Benchmark：
  - GSM8K（5-shot，输入 token 600-1700，Exact Match 评估）
  - LongBench（全部 21 个数据集，覆盖 6 类任务：Math、Code Completion、Few-shot Learning、Multi-Document QA、Single-Document QA、Summarization、Synthetic Tasks）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/Concyclics/LogQuantKV
  
  **算法 Pipeline**：
  
  1. **Log-distributed token 选择（Algorithm 1）**：维持 2W~3W 个全精度 token。当长度超过 3W 时，对前 2W 个 token 做步长=2 的子采样（A[0:2W:2]），密度减半至 W 个，再追加 W 个新 token，天然形成 log₂ 稀疏性——最新窗口密度 p，次新窗口密度 p/2，再往前 p/4……
  
  ```
  Input: A (list of original precision tokens), a* (new token), W (window length)
  Output: A (updated list of tokens)
  procedure APPENDTOKEN(A, a*, W):
    if length(A) < 3W:
      A ← concat(A, a*)
    else:
      A ← concat(A[0:2W:2], A[2W:3W])  // 将前 2W 个 token 密度减半
      A ← concat(A, a*)                 // 追加新 token
    end if
    return A
  end procedure
  ```
  
  2. **量化策略**：非保留 token 量化为 INT2。每 channel 独立量化（Key-per-channel），group size=64（HuggingFace 默认值）。量化后端使用 Quanto（也可换 HQQ）。
  
  3. **Position-Agnostic 重组**：由于 Attention 输出 O = A·V = softmax(QK^T)·V 对 K、V 中 token 顺序具有置换不变性（即 A_P·V_P = A·V，P 为任意置换），可将全精度 token 与量化 token 连续拼接存储，无需保留原始位置顺序，提升内存局部性和处理效率。
  
  4. **压缩率计算**：对于序列长度 L、保留 2W 个全精度 token 的 BF16 模型 + 2-bit 量化：compression ratio = 16L / (2(L-2W) + 16×2W)。
  
  5. **与 KiVi 的关系**：LogQuant 的 W 受限于 ⌊R/3⌋（KiVi 保留 R 个全精度 token），确保不超过 KiVi 的全精度 token 数量。对于 R=128，LogQuant 使用 3⌊128/3⌋=126 个全精度 token，略少于 KiVi 的 128 个。
  
  6. **集成方式**：继承 HuggingFace transformers 的 Cache 类，通过 derived class 实现。与 HuggingFace 推理 pipeline 无缝兼容。
  
  7. **张量计算流程**（单次解码步骤）：
     - Q ∈ R^(1×d)，K_cache ∈ R^(N×d)，V_cache ∈ R^(N×d)
     - 对 quantized token 的 K/V 做 dequantize：K_deq = dequant(K_quantized, scale, zero_point)
     - K_full = concat([K_deq, K_full_precision])  // position-agnostic 重排后存储
     - A = softmax(Q × K_full^T / √d)  // 标准 scaled dot-product attention
     - O = A × V_full  // 加权求和，token 顺序不影响结果
  
  关键结果：LogQuant 在相同压缩比下，Math/Code 任务准确率比 KiVi 高 40%-200%；吞吐量提升 25%，batch size 增加 60%（H100 48G）。
