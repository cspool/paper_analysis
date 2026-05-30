## FP16 Accumulator for PV Matmul (FP16累加器加速PV矩阵乘)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP16 Accumulator for PV Matmul 是 SageAttention 提出的 Attention PV Matmul 加速方案。不对 P,V 做 INT8 量化（最差层 cosine sim 仅 56.40%），保持 P,V 在 FP16 但将 Tensor Core MMA accumulator 从 FP32 降为 FP16。RTX4090/3090 上 mma(f16.f16.f16) 比 mma(f16.f16.f32) 快 2×（Ada Lovelace: FP16 accum 512 FMA/SM/cycle vs FP32 accum 256），且精度与 FP32 accum 一致（cosine sim 差值 0.00%）。关键 insight：P（softmax 输出 ∈[0,1]）和 V 数值范围适合 FP16，attention output O 不需要 FP32 累积精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
O_i^j = diag(e^{m_i^{j-1} - m_i^j}) O_i^{j-1} + Matmul(
    P̃_ij.to(tl.float16), V_j.to(tl.float16), accum=tl.float16  # ← 2x faster
)
```
四个 kernel 变体: SAGEAttn-T/B (QK INT8 + PV FP16 accum)、SAGEAttn-vT/vB (QK INT8 + PV INT8, +4% speed)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Triton: `tl.dot(P̃, V, out_dtype=tl.float16)`。CUDA: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16`。FP16 accum 2× 加速仅在 consumer GPU (RTX4090/3090) 成立——数据中心 GPU (A100/H100) FP32 accum 已是快速路径。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
