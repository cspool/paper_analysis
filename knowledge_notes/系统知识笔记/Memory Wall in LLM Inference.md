## Memory Wall in LLM Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Wall 是计算机架构中长期存在的问题，指内存带宽（memory bandwidth）的增长速度远落后于计算吞吐量（compute throughput）的增长速度，导致计算单元因等待数据而"饥饿"。在 LLM 推理中，Memory Wall 尤为严重——单 batch 生成式推理几乎完全是 matrix-vector operations（GEMV），每个权重加载后仅参与一次乘加运算（无法跨 token 分摊复用），导致 arithmetic intensity（FLOP/byte）极低。SqueezeLLM Sec. 3 量化了此问题：A5000 GPU 的 peak compute 为 222 TFLOPS，但 peak memory bandwidth 仅 768 GB/s（差距 290x）。对于 batch_size=1 的 LLM 推理，roofline model 分析确认整个推理过程位于 memory-bound 区域——降低权重精度（减少内存读取量）直接带来成比例的延迟降低（Fig. 2: bitwidth 线性关联 latency），证明内存带宽是瓶颈而非计算能力。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LLM 单 batch 推理中的 Memory Wall 流程：
```
每个 autoregressive 解码 step:
  1. 从 GPU DRAM 加载所有权重 (如 LLaMA-7B: ~13GB FP16)
     ↓ 瓶颈: 内存带宽限制 (A6000: 768 GB/s)
  2. 加载 KV Cache、input activation (相对很小)
     ↓ 
  3. 计算 GEMV: output[i] = Σ_j W[i,j] × input[j]
     ↓ 计算时间 << 内存加载时间 (memory-bound)
  4. 生成一个 token
     → 总延迟 ≈ 权重加载时间 / 内存带宽
```

Memory Wall 对量化策略的直接启示：(1) 无需对 activation 量化（它们不是瓶颈），只量化权重即可获得接近压缩比的加速；(2) 非均匀量化虽增加少量 dequant 计算开销（LUT lookup），但这些额外计算被内存带宽瓶颈完全掩盖；(3) 不必追求整数算术加速（如 INT4 Tensor Core），因为 compute 不是瓶颈——SqueezeLLM 选择非均匀量化正是因为这个原因。

SqueezeLLM Fig. 2 Roofline 分析（A5000, LLaMA-7B）：
- 16-bit: latency = baseline
- 8-bit: latency ≈ 0.5 × baseline（2x 加速）
- 4-bit: latency ≈ 0.25 × baseline（4x 加速）
- 延迟 ∝ bitwidth（线性关系）→ 确认 memory-bound

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Memory Wall 在 LLM 推理中的应对策略：(1) Weight-only quantization：降低 weight 精度以减少内存读取量（GPTQ, AWQ, SqueezeLLM）；(2) KV Cache 压缩：量化或稀疏化 KV cache（如 KIVI, PQCache）；(3) 投机解码（speculative decoding）：用 draft model 生成多个 token 以摊销权重加载（batch 增大→arithmetic intensity 升高）；(4) Prefill-decode 分离（disaggregation）：将 compute-bound prefill 和 memory-bound decode 解耦到不同硬件。SqueezeLLM 的关键贡献是明确以 Memory Wall 为出发点设计量化方案——与许多 prior work 不同（它们仍假设 uniform/integer 量化的计算加速有意义），SqueezeLLM 选择非均匀量化因为 memory bandwidth 是唯一瓶颈。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
