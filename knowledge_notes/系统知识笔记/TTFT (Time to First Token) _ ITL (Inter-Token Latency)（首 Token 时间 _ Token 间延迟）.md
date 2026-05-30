## TTFT (Time to First Token) / ITL (Inter-Token Latency)（首 Token 时间 / Token 间延迟）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TTFT（Time to First Token）和 ITL（Inter-Token Latency）是 LLM 推理性能评估的两个核心延迟指标，分别衡量推理 pipeline 的不同阶段：(a) **TTFT（首 Token 时间）**：从接收到完整输入 prompt 到生成第一个输出 token 之间的时间，反映 prefill 阶段的效率（包括 prompt tokenization、全部 prompt token 的并行编码、KV-cache 初始化和第一个 token 的自回归生成）。TTFT 决定了用户感知的响应速度（"how fast does the model start responding"），是衡量 LLM serving 交互体验的关键指标。(b) **ITL（Token 间延迟）**：相邻两个输出 token 生成之间的平均时间间隔，反映 decode 阶段的每 token 生成速度。ITL 决定了流式输出的流畅度（"how fast do words appear on screen"）。MoE-Inference-Bench 的 ITL 计算公式为：$ITL = \frac{\text{End-to-End Latency} - \text{TTFT}}{\text{BatchSize} \times \text{OutputTokens} - 1}$。两者的物理含义不同：TTFT 主要由计算量（prefill 对所有 prompt token 做并行 attention）决定，compute-bound 且与 input length 强相关；ITL 主要由显存带宽（decode 每次只处理 1 个新 token，受限于 KV-cache 读取和 expert 参数访问）决定，memory-bound 且与 batch size、model size（活跃参数数量）相关。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MoE 推理中 TTFT 和 ITL 的测量流程（以 vLLM + Mixtral-8x7B on H100，batch_size=64, input_len=2048, output_len=2048 为例）：

```
Timeline of a single request through vLLM serving pipeline:

[Time 0ms]   Client sends POST /v1/completions with prompt (2048 tokens)
             → vLLM API server receives and queues request
             → Scheduler assigns KV-cache blocks (PagedAttention allocation)

[Time 0-T1]  Tokenization + Prefill Phase (TTFT window):
             ├─ Tokenizer: prompt → 2048 token IDs
             ├─ Layer 0..31 (prefill, all 2048 tokens processed in parallel):
             │  ├─ Self-Attention: QKV projection → flash attention (parallel) → O proj
             │  ├─ MoE Router: gate_logits → softmax → topk(expert_1, expert_2)
             │  ├─ Expert FFN: token dispatch → expert GEMM → weighted sum
             │  └─ Store KV-cache for all 2048 positions
             └─ LM Head: hidden_state[last_position] → vocab logits → sample → first_token

[Time T1]    TTFT measured: timestamp(T1) - timestamp(0)
             e.g., TTFT = 450ms for Mixtral-8x7B, BS=64, input=output=2048

[Time T1-T2] Decode Phase (ITL window × output_tokens-1 times):
             For i in 2..2048:  // generate token 2 to 2048
             ├─ Layer 0..31 (decode, single new token only):
             │  ├─ Self-Attention: QKV(new_token) → attend to 2048+i cached K,V
             │  │  (memory-bound: KV-cache read dominates)
             │  ├─ MoE Router + Expert FFN: single token routing + expert compute
             │  │  (memory-bound: loading expert weights dominates)
             │  └─ Append new KV entry for position 2048+i
             └─ LM Head → sample → next_token
             Each iteration: ~8-12ms ITL

[Time T2]    End-to-End Latency = timestamp(T2) - timestamp(0)
             ITL = (End-to-End - TTFT) / (64 × 2048 - 1)
             Throughput = BatchSize × (Input+Output) / End-to-End
                        = 64 × 4096 / (T2 - 0)
```

MoE-Inference-Bench 中 TTFT 和 ITL 的关键测量方法（Section 3.4）：(a) TTFT 通过 `max_output_tokens=1` 限制输出为单 token，记录总生成时间得到 TTFT；(b) ITL 通过完整生成（如 2048 output tokens）的 end-to-end latency 反算（因为 ITL 通常在 decode 阶段保持稳定）。Section 4.1 的关键发现（Figure 3-4）：OLMoE-1B-7B 的 TTFT 比 DeepSeek-V2-Lite 快约 70%；VLM 的延迟差距远大于 LLM（DeepSeek-VL2-Tiny 比 DeepSeek-VL2 的 TTFT 快约 30%，ITL 差距达 240%，end-to-end 差距超 260%），因为视觉编码器的额外计算负载。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **测量实现**：在推理框架（vLLM、TensorRT-LLM 等）中通过 hook/callback 在以下关键节点打时间戳：(a) 请求到达时刻；(b) prefill 完成、第一个 token 生成时刻（TTFT 终点）；(c) 每个 decode step 的 token 生成时刻（ITL 样本点）；(d) 最后一个 token 生成时刻（end-to-end latency 终点）。MoE-Inference-Bench 通过 vLLM 的 Python API 进行测量。
- **性能优化关联**：(a) 降低 TTFT——prefill 阶段增大 batch size 可摊薄 kernel launch 开销，TP（Tensor Parallelism）可并行化 attention 和 FFN 计算（MoE-Inference-Bench 证明 TP 比 PP/EP 更有效）；FP8 量化可加速 prefill 中的 GEMM。(b) 降低 ITL——Fused MoE（减少 kernel launch 和 HBM 往返，Section 7.2 显示 12-18% ITL 改善）；expert pruning（减少每步需加载的参数量）；speculative decoding（单步生成多个 draft token 摊薄验证开销，Section 6.3）。
- **SLO 关联**：TTFT 通常对应于 latency SLO（如 <500ms for interactive chat）；ITL 对应于 throughput 或 streaming speed SLO（如 >20 tokens/sec for smooth reading experience）。MoE-Inference-Bench 在 accuracy-throughput-latency 三维空间中定位各模型（Figure 17-18），帮助选择满足特定 SLO 的最优模型。
- 局限：(a) TTFT 和 ITL 高度依赖于硬件和 batch size（同一模型的 TTFT 在 batch=1 和 batch=64 下可能差数十倍），因此 benchmark 需控制变量（MoE-Inference-Bench 固定 batch size 和 input/output length 进行跨模型比较）；(b) ITL 的简单平均值可能掩盖 decode 过程中的延迟抖动（如某些 expert 被频繁激活导致其所在的 GPU/计算单元成为瓶颈）。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

---
