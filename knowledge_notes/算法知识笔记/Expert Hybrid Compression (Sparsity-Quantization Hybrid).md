## Expert Hybrid Compression (Sparsity-Quantization Hybrid)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Hybrid Compression 是 FloE 提出的 MoE expert 差异化压缩策略。与传统的 uniform 压缩（所有投影矩阵用同一量化位宽或同一稀疏策略）不同，FloE 利用 expert 内部三组投影矩阵（W_gate, W_down, W_up）对压缩方法的**差异化敏感性**，对不同矩阵采用不同压缩方法：(1) W_up 对 ultra-low-bit 量化最不敏感但在稀疏化上中等敏感——使用 INT2 HQQ 量化；(2) W_gate 和 W_down 对量化极其敏感（INT2 时 perplexity 暴涨 100×+）但对 contextual sparsification 可接受——使用基于 up projection 输出的幅值剪枝（90% 稀疏度下保留 ~10% 通道）。这种混合策略在 Mixtral-8×7B 上实现 9.3× 总体压缩比，同时将精度退化控制在 4.4%~7.6%。

设计依据：(1) 量化敏感度实验（FloE Figure 3b）显示各投影矩阵在 INT2 下的 WikiText-2 perplexity: W_down=14.36, W_gate=6.245, W_up=6.177，即 W_up 对量化最鲁棒；(2) 稀疏化敏感度实验显示各投影矩阵在 90% 稀疏度下 perplexity: W_gate(SiLU)=18.53, W_up=9.13, W_down(input)=6.55，即 W_down 对稀疏化最鲁棒；(3) 理论证明 L_down ≤ L_up < L_gate，支持选择 up 输出剪枝作为误差-效率最佳平衡点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FloE Expert Hybrid Compression 的 DRAM 存储布局
// 每个 expert E_ij 存储:
//   W_up_ij:    INT2 packed [4096, 14336], ~3.6MB (vs FP16 ~117MB)
//   W_gate_ij:  仅保留被 sparsity mask 选中的列, FP16 [4096, 1434] (~10%)
//   W_down_ij^T: 仅保留被 sparsity mask 选中的行(转置为列), FP16 [4096, 1434]
// 总大小: ~3.6MB + 11.7MB + 11.7MB ≈ 27MB vs FP16 全量 ~351MB → 13×

// 实际压缩比:
// 稀疏 gate: 10% × FP16 = 等效 ~1.6 bits/element
// 稀疏 down: 10% × FP16 = 等效 ~1.6 bits/element
// 量化 up:  INT2 = 2 bits/element
// 总体: (1.6+1.6+2)/48 ≈ 10.8% → 9.3× 压缩 (考虑 scale/zero-point overhead)

// 推理时的 decompress + compute 流水线:
// Step 1: CPU 从 DRAM 读取 INT2 W_up + FP16 sparse W_gate[cols], W_down^T[cols]
// Step 2: AVX-512 解量化 W_up (INT2→FP16) + 打包到 pinned memory
// Step 3: 多 CUDA stream 异步传输到 GPU
// Step 4: GPU 执行 sparse GEMV kernel
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 与 Mixed MoE Quantization（attention 高精度 + expert 低精度）不同，Hybrid Compression 是 expert **内部**的差异化压缩
- W_up 选择 INT2 而非 INT1：FloE 未明确说明，但 Table 7 显示 INT1 时 W_up perplexity=520（已不可用），INT2 时仍可控（6.177）
- 稀疏化与量化引入的误差近似独立且可加（FloE Figure 9b），便于分别建模和控制
- 实现依赖：HQQ library（https://github.com/mobiusml/hqq）用于 W_up INT2 量化；Triton kernel 用于 sparse GEMV
- 通用性：FloE 在 Mixtral-8×7B, Phi-3.5-MoE, DeepSeek-V2, DeepSeek-MoE-16B, Qwen1.5-MoE 上均验证了 up projection 对量化和稀疏化的低敏感性

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
