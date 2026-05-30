## Roofline Analysis (for GPU Kernel Bottleneck Diagnosis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Roofline Analysis基于roofline模型（Williams et al., 2009）的GPU kernel性能诊断方法。将kernel的arithmetic intensity（α = FLOP/Bytes）在硬件roofline上标注——由peak compute throughput（水平线）和memory bandwidth limit（斜线）组成——判断kernel是compute-bound（α ≥ α̃）还是memory-bound（α < α̃），并估算最优可达性能。论文创新应用：通过roofline分析首次系统性解释为什么BLR方法在多token推理时性能退化——尽管FLOP减半，额外中间数据移动(b×n×r)使α下降10-15×，将compute-bound的dense层推入memory-bound区域。

从kernel调度角度拆解术语：
A40 BF16 roofline参数和BLR分析：

```
// A40 BF16
Peak Compute: ≈ 150 TFLOPS; Peak Bandwidth: ≈ 696 GB/s
Breakpoint: α̃ ≈ 215 FLOP/byte

// Llama-7B Qproj (i=o=4096, r=1024, b=16)
// 单token (n=1): 所有方法α > 215 → compute-bound
//   压缩权重直接加速 (memory traffic受weight读取主导)
//
// 多token (n=1024):
//   Dense:     α ≈ 34G/34MB ≈ 994 → compute-bound ✓
//   Low-Rank:  α ≈ 17G/17MB ≈ 978 → compute-bound ✓
//   Monarch:   α ≈ 17G/138MB ≈ 123 → memory-bound ✗ (α < 215!)
//   BLAST:     α ≈ 17G/266MB ≈ 64  → strongly memory-bound ✗✗
//
// 关键: BLR的b×n×r中间张量是memory bottleneck根源
//   Monarch: +4bnr bytes, BLAST: +8bnr bytes
//   这些中间量dense baseline零开销
```

术语一般如何实现？如何使用？
Roofline分析用于三阶段：(1) profiling——通过NCU/DCGM测量实际FLOP/s和memory bandwidth→计算实测α；(2) diagnosis——对比实测α与α̃判断瓶颈类型→指导优化方向（memory-bound→减少数据移动; compute-bound→优化计算效率）；(3) 验证——优化后重测α确认是否重回compute-bound。论文用此方法不仅诊断BLR退化根因，还验证Triton kernel优化效果。限制：(a)假设零延迟和完美overlap→实际性能低于roofline预测；(b)不考虑cache→实际effective memory traffic可能因L1/L2 hit低于模型假设。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
