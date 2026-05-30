## Computation Mapper for NMP-xPU Hybrid Inference (NMP-xPU混合推理的计算映射器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Computation Mapper 是 Stratum 中决定 LLM 推理的各个阶段（prefill vs. decode）在各计算单元（xPU vs. NMP）上如何分配的系统组件。其决策依据是各阶段的算术强度（arithmetic intensity = FLOPs / memory bytes）：prefill phase（大批量 prompt tokens 的并行处理）是 compute-bound——适合在 xPU（H100, 312 TFLOPS）上执行；decode phase（单 token 自回归生成，KV cache 密集型）是 memory-bound——适合在 NMP（19-34 TB/s internal bandwidth）上执行。此分配策略与 AttAcc (ASPLOS '24) 的 prefill-on-GPU + decode-on-PIM 类似。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Stratum Computation Mapper 的执行流程：
```
For each dispatched batch B:
  # Phase 1: Prefill (on xPU)
  xPU executes:
    - Embedding lookup
    - All attention layers (prefill): QKV projection → FlashAttention → output proj
    - All FFN layers (if non-MoE layers exist)
    - MoE gating (lightweight: softmax(W_gate @ X), K_out << K_in)
    - Generate KV cache entries → write to Mono3D DRAM (via interposer)
  
  # Phase 2: Decode (on Stratum NMP)
  xPU sends to NMP:
    - Input token hidden states
    - Expert routing IDs + gating weights
    - Switch Mono3D DRAM to NMP mode
  
  NMP executes (per decode step):
    - Expert FFN (GeMM1/2/3 + activation, using tensor parallelism)
    - Attention (Q@K^T + Softmax + Attn@V, using head-level parallelism)
    - KV cache append (new K,V pairs stored in intermediate tier)
  
  NMP → xPU:
    - Write output hidden states to designated DRAM address space
    - Exit NMP mode
  
  xPU reads output:
    - LM head projection → logits → sampling → next token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Computation Mapper 的关键实现考量：(1) Prefill-to-decode transition——prefill 阶段的 output（last layer hidden states of all prompt tokens）需要通过 interposer 传输到 NMP，传输量 = batch_size × seq_len × hidden_dim × 2 bytes（BF16），这是不可避免的 interposer 带宽开销，但属于一次性 cost；(2) Per-step decode data flow——每 decode step NMP 仅需从 xPU 接收 single token's hidden state（可在 prefill 时提前发送多个 future tokens 的 hidden states 以减少交互次数）；(3) xPU 在 decode phase 的作用——xPU 在 NMP 执行 decode 时并非完全 idle：可并发执行下一个 batch 的 prefill 或 topic classification；(4) 不适用于所有 workload——对于 prefill-heavy workload（如 summarization with long input, short output），compute mapper 可能将更多工作放在 xPU 上。AttAcc 在 decode 阶段将 attention 完全放在 PIM 中，而 Stratum 因其更高的 logic die compute capacity（128 TFLOPS vs. AttAcc 的 much simpler PIM logic），可将 attention + expert 全部 offload 到 NMP。

涉及论文标题：
- Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving
