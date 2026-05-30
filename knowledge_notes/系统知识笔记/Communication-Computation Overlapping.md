## Communication-Computation Overlapping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Communication-Computation Overlapping（通信-计算重叠）是分布式训练中的核心技术——通过在 GPU 的不同 CUDA stream 上并发执行通信（如 NCCL all-reduce、all-to-all）和计算（如矩阵乘法、attention），将通信延迟隐藏在计算时间中，减少 GPU 空闲等待。在 MoE 训练中特指将 A2A 通信（all-to-all dispatch/combine）与 expert 计算重叠。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

在 MoE 训练中的两种 overlapping 粒度：

1. **Sequence-Level Overlapping**（粗粒度）：在 batch 维度切分，不同 sequence 的 A2A 和计算重叠。需要足够大的 batch size，长序列下因内存限制不可行（batch size 可能为 1）。

2. **Token-Level Overlapping**（细粒度）：在 token 维度切分，同一 sequence 内的不同 micro-batch 的 A2A 和计算重叠。不受 batch size 限制，但受限于计算量与通信量的比例。

FOLDMOE 定位的关键问题：

```
MoE-only overlapping (Tutel):  Expert_comp_time / A2A_time → 太小 (21% at 32K)
FOLDMOE solution:             (Attn + Expert)_time / A2A_time → 足够大 (接近或超过 100%)
```

通过将 attention 计算（O(n²)）纳入 overlap 的计算源，随着 seqlen 增长，可用于隐藏通信的计算量也相应增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PyTorch 中的实现模式：
```python
comm_stream = torch.cuda.Stream()
# Main stream: 计算 micro-batch i
with torch.cuda.stream(compute_stream):
    Z_i = attention(X_i, K_cache, V_cache)
# Comm stream:  A2A micro-batch i-1
with torch.cuda.stream(comm_stream):
    Y_i_1 = a2a_dispatch_then_expert_then_combine(Z_i_1)
torch.cuda.synchronize()  # 必要时同步
```
FOLDMOE 在 Megatron-LM 基础上实现此模式，与 FlashAttention、TP、SP 兼容。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

FasterMoE（PPoPP'22）实现了更细粒度的通信-计算重叠：将 all-to-all 拆分为 group-based pairwise exchange 操作序列（S/C/R），在独立的 CUDA comm stream 和 comp stream 上交错执行。与 FOLDMOE 在 sequence 维度切分不同，FasterMoE 在 worker group 维度切分——n 个 group 形成环结构，每 step j 执行 S（send tokens）、C（compute on received tokens）、R（receive results）三个操作，通过将最快操作放在首尾最小化 overhead。当 comp stream 占主导时优化效果最佳（DDL-Roofline 指导）。实测单用 smart scheduling 加速 1.40×。
