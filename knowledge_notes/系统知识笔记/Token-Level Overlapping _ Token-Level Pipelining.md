## Token-Level Overlapping / Token-Level Pipelining

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token-Level Overlapping（Token级重叠）是一种在 MoE 训练中隐藏 all-to-all (A2A) 通信延迟的技术。将输入 MoE 层的 token 序列沿 sequence 维度切分为多个微批次（micro-batches），在分离的 CUDA stream 上分别执行专家计算（compute stream）和 A2A 通信（communication stream），使不同微批次的通信和计算在时间上重叠。与 sequence-level overlapping（在 batch 维度切分，需要较大的 batch size）相比，token-level overlapping 在长序列训练中优势明显——长序列下 batch size 被内存限制得极小（甚至为 1），但 sequence 维度始终有足够 token 用于微批次划分。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 MoE-only token-level overlapping（如 Tutel）中，流水线仅覆盖 MoE 层内的 3 个阶段：

```
Stream 0 (Compute):  Expert(mb_0)  |  Expert(mb_1)  |  Expert(mb_2)  |  Expert(mb_3)
Stream 1 (Comm):     A2A(mb_0)     |  A2A(mb_1)     |  A2A(mb_2)     |  A2A(mb_3)
```

核心约束：A2A 通信的高斜率（带宽限制）和 expert 计算的低计算量（MoE layer 仅为轻量 FFN）导致通信无法被完全隐藏。32K seqlen 下 expert 仅占 21% 执行时间。

FOLDMOE 将 overlapping 扩展到整个 Transformer block，增加 attention 计算作为 overlap 的计算源：

```
Stream 0 (Compute):  Attn(mb_0) | Attn(mb_1) | Expert(mb_0) | Attn(mb_2) | Expert(mb_1) | Attn(mb_3) | Expert(mb_2) | Expert(mb_3)
Stream 1 (Comm):      idle       | A2A(mb_0)  | A2A(mb_0)   | A2A(mb_1)  | A2A(mb_1)   | A2A(mb_2)  | A2A(mb_2)   | A2A(mb_3)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Token-Level Overlapping 的实现通常涉及：
1. 修改训练框架（如 Megatron-LM）的 layer forward 逻辑，在 token 维度做 micro-batch 循环
2. 使用 PyTorch CUDA stream 分离通信和计算：`torch.cuda.Stream()` 创建独立 stream
3. NCCL 通信在此 stream 上异步执行，与 CUDA kernel 重叠
4. Overlap degree d 的 tuning：FOLDMOE 和 Tutel 都通过搜索 d=2/4/8/16 找最优值——d 越大 bubble 越小但 kernel launch overhead 越大

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
