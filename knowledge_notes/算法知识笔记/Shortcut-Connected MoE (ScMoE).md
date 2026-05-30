## Shortcut-Connected MoE (ScMoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shortcut-Connected MoE (ScMoE) 是 LongCat-Flash / [Cai et al., 2024] 提出的 MoE 架构创新。核心思想：在 Transformer layer 中引入跨层 shortcut 连接——将同一层第一个 Multi-head Latent Attention (MLA) block 的输出直接连接到该层的 MoE block，使前一层的 Dense FFN 计算可以与当前层 MoE 的 dispatch/combine 通信并行执行。

传统 MoE execution paradigm（如 DeepSeek-V3 的 interleaved MoE+Dense FFN）中，Expert Parallelism 要求先完成 all-to-all 通信（token dispatch）才能开始 expert 计算，通信延迟成为串行瓶颈。Shared-expert 架构尝试用单个 expert 的计算时间与通信重叠，但重叠窗口受限于单个 expert 的计算量。ScMoE 将 Dense FFN 从 MoE 之后移到 MoE 之前（通过 shortcut 连接），利用 Dense FFN 较大的 intermediate size（12288 vs expert 2048）创造更大的 computation-communication overlap 窗口。

LongCat-Flash 验证了 ScMoE 在四种模型配置下（2.4B-16B MLA, 3B-20B MHA, 15B-193B GQA）training loss 与 baseline 几乎相同（Figure 4），证明 ScMoE 是 quality-neutral 的架构优化。进一步提升：将 MoE layer 沿 token 维度分为两个 chunk，实现 chunk 间互相重叠 + 与 Dense FFN 重叠。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ScMoE layer 结构（单层包含 2 个 MLA + Dense FFN + MoE）：

```
# ScMoE Layer Forward Pass (per token batch)

输入: hidden_states [batch, seq_len, d_model]

# Stage 1: 第一个 MLA (独立执行)
h1 = MLA_0(hidden_states)  # 产生 attention output + KV cache

# Stage 2: Dense FFN (可与当前层 MoE 通信并行)
dense_out = DenseFFN(h1_chunk_a)  # chunk_a 的 dense FFN
MLA_0_qkv = QKV_Projection(h1_chunk_a)  # chunk_a 的 QKV 投影
# 同时: All-to-All Dispatch(h1_chunk_b → experts)  # chunk_b 的 token dispatch

# Stage 3: MoE GEMM (独立执行)
moe_out_b = MoE_GEMM(dispatched_chunk_b_tokens)  # chunk_b 的 expert 计算

# Stage 4: 第二个 MLA + Dense FFN + All-to-All Combine (并行)
attn_out_a = CoreAttention(MLA_0_qkv_a) + OutputProjection
dense_out_b = DenseFFN(h1_chunk_b)
# 同时: All-to-All Combine(moe_out_b → original GPUs)

输出: attn_out + dense_out + moe_out (残差累加)
```

ScMoE 的关键特征：(1) Shortcut 从 MLA_0 直连到 MoE block，使 Dense FFN 在 MoE 之前执行（而非之后），创造重叠窗口；(2) Token chunking 分两个 chunk 交替执行，chunk_a 的 dense FFN+attention 与 chunk_b 的 all-to-all 通信并行；(3) architecture 与 attention 机制（MLA/MHA/GQA）正交——Figure 4 证明 loss 曲线在三种 attention 下均几乎相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **架构层面**：每层内 MLA_0 → Dense FFN + MoE 平行路径（shortcut 连接），替代传统 MLA_0 → MoE → Dense FFN 串行或 MoE ↔ Dense FFN interleaved。
2. **训练层面**：non-overlapping dispatch/combine 时间从 25.3% 降至 8.4%。ScMoE 与 expert parallelism group (EP=32) 和 V-ZB pipeline 协同使用。
3. **推理层面（SBO）**：ScMoE 是 Single Batch Overlap 的基础——Dense FFN 计算可与 all-to-all dispatch 重叠，Attention Core 可与 all-to-all combine 重叠。TPOT 理论值降低近 50%（vs DeepSeek-V3 TBO）。
4. **通信层面**：Dense FFN 的 intra-node NVLink 通信（TP 的 all-gather/reduce-scatter）可与 MoE 的 inter-node RDMA 通信（EP 的 all-to-all）通过 GPUDirect RDMA 并发执行。

涉及论文标题：
- LongCat-Flash Technical Report
- Shortcut-Connected Expert Parallelism for Accelerating Mixture-of-Experts
