## Joint Quantization and Sparsification (联合量化和稀疏化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
联合量化和稀疏化是将量化（减少 bit-width）和剪枝（移除不重要权重）同时应用于同一 LLM 的技术范式，突破单一压缩方法的限制。单一量化在 sub-4bit 时性能急剧下降（如 QuaRot W3A4KV4 → 132.97 PPL），单一剪枝 >50% 稀疏度也面临瓶颈。联合方法叠加两种正交压缩维度实现更激进压缩——W4A4KV4+50% sparsity 有效 bit-width 等效于约 W2.5 水平。NVIDIA Ampere/Hopper 原生支持 INT4 sparse GEMM，使联合压缩具备实际硬件加速价值。核心挑战：量化偏好窄范围，剪枝偏好高方差，Hadamard rotation 虽利于量化但破坏剪枝所需的分布差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
三种实现范式对比：

```
Naive sequential (QuaRot+WANDA):
  rotate(W) → prune(rotated_W) → quantize  // 分布冲突 → 性能崩溃

SparseGPT+GPTQ:
  calibrate → SparseGPT row-wise prune+Hessian update → GPTQ  // 改善但未调和冲突

OBR:
  rotate → prune → OBR compensation(prune error) → OBR compensation(quant error) → quantize
  // 通过 Hessian 桥接两种压缩，调和分布冲突
```

等效 bit-width: effective_bits = actual_bits × (1-sparsity_ratio)。如 W4+50% sparsity → 2bit equivalent per weight。OBR 同等等效位宽下显著优于纯量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) prune-then-quantize 顺序（被 Harma et al. 2024 理论证明最优）；(2) 联合梯度优化（DJPQ, OBQ）；(3) 误差补偿框架（OBR, JSQ）。硬件：NVIDIA Sparse Tensor Cores + INT4。适用场景：边缘设备 LLM 部署、memory-bound 推理。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
