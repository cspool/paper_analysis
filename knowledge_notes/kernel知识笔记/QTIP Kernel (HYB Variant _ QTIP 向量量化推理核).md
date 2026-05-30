## QTIP Kernel (HYB Variant / QTIP 向量量化推理核)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QTIP (Quantization with Trellises and Incoherence Processing, NeurIPS 2024) 是一种 weight-only vector quantization PTQ 方法，其 HYB (hybrid) 变体提供了开源 CUDA kernel 用于 GPU 推理。与 scalar quantization（每权重独立 index）不同，QTIP 将 d 个连续权重编码为一个 lattice codebook vector index，通过 trellis 结构化搜索优化量化。HYB kernel 使用小型的 LUT（适合 GPU L1 cache）进行向量解码，然后将解码后的 FP16 权重与 activation 进行 GEMM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QTIP HYB kernel 的高层次流程：
```
# 输入：packed vector indices I (B bits per d-dim group)
#       activation A, lattice codebook L (shared or per-group)
# 输出：FP16 output

for each vector group g (size d):
    idx = unpack(I, g)                         # 解码 B-bit index
    w_vec = decode_lattice(L, idx)             # d 维向量（比 scalar 查表更复杂）
    for each output m:
        output[m] += dot(A[m, g_start:g_end], w_vec)
```
解码延迟 overhead 使 vector quantization 推理吞吐低于 scalar：RTX 4090 上 Llama-2-7B 2-bit non-uniform scalar 347 tok/s vs vector (QTIP HYB) 200 tok/s，即使 fusing Q/K/V projections 后 vector 仅达 248 tok/s（Table 2, Table 7）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：github.com/Cornell-RelaxML/qtip（HYB CUDA kernel）。QTIP 有三个变体：1MAD、3INST（LUT-free, 论文未开源 kernel）和 HYB（带 L1-cache 友好 LUT, 开源 kernel）。GuidedQuant 使用 QTIP HYB kernel 进行向量量化吞吐评估。推理吞吐结论：non-uniform scalar + Any-Precision-LLM kernel 在 memory-bound（batch=1）场景下比 vector + QTIP HYB kernel 提供更好的 latency-accuracy tradeoff。

在 QTIP 原始论文中的实现：bitshift trellis 解码无需索引解包和格点解码——每个权重通过 compute-based code (1MAD/3INST/HYB) 直接从 L-bit 状态字即时生成。流程为：读取 32-bit word → bitshift 获取 L=16 bit 状态 → 1MAD: (LCG + 4×8bit sum + scale) 2 instr / 3INST: (LCG + XOR FP16 + FADD) 3 instr / HYB: (hash + LUT lookup + sign flip) 摊销 2 instr → 输出 FP16 权重。16×16 tile 映射到 MMA tile 直接矩阵乘。QTIP 在 RTX 6000 Ada 上 Llama 2 7B 2-bit 达 188 tok/s (>3× FP16 55.9 tok/s)，与 QuIP# 吞吐持平（186 tok/s），但有效维度为 256（vs QuIP# 8D），量化质量更高而无额外推理开销。关键优化：HYB codebook 仅 2KiB（比 AQLM 1MiB 小 512×），可 32× 复制消除 bank conflicts；bitshift 解码完全并行化（对比 naive TCQ 的顺序依赖）。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
