## Mixed-Precision Quantization for MoE（MoE 模型的混合精度量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixed-Precision Quantization for MoE 是针对 Mixture-of-Experts 模型设计的量化策略，为 MoE block 内不同的 linear block（gate_proj, up_proj, down_proj）分配不同的量化精度（位宽），而非使用统一精度。MxMoE 的核心洞察：MoE block 内存在两个维度的异构性：(1) 量化敏感度异构——同一 expert 内不同 linear block（如 gate_proj vs down_proj）对量化位宽的敏感度差异显著；(2) 计算特性异构——不同 expert 的激活频率差异超过 10×，导致部分 expert 的 GEMM 为 memory-bound（低频率），部分为 compute-bound（高频率）。混合精度策略根据敏感度给敏感 block 分配更高精度、给不敏感 block 分配更低精度；同时根据硬件特性给 memory-bound GEMM 用 weight-only 量化（如 W4A16），给 compute-bound GEMM 用 weight-activation 量化（如 W4A4 或 W8A8）。与 expert 级混合精度（MC-MoE 等）不同，MxMoE 在更细的 linear-block 粒度分配位宽，实验证明 linear-block 粒度一致优于 expert 级。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MxMoE 混合精度量化流程（以 W5A5 目标平均位宽为例）：

```
输入: MoE block (E experts, 每个 3 linear blocks: gate/up/down)
      校准数据 X_cal (128 seqs × 4096 tokens)

1. 逐 linear-block 量化敏感度评估:
   for expert i=1..E, block j in {gate, up, down}:
       for scheme k in {W4A4, W4A4-g128, W8A8, ...}:
           W_q = GPTQ(W_{i,j}, scheme k)
           Ô = MoE_forward(W在第(i,j)位量化为W_q, 其余FP16)
           Δ_{i,j,k} = ||Ô - Ô_FP16||₂

2. ILP 求解最优分配:
   minimize L^r · T^{1-r}
   s.t. 每 block 选一方案, 总内存 ≤ M
   
   输出: x_{i,j,k} (每个 linear block 的量化方案)

3. 按分配方案量化:
   for each linear-block (i,j) with assigned scheme k:
       W_{i,j} → GPTQ_quant(W_{i,j}, scheme k)
       激活运行时按 scheme k 动态量化

示例分配 (Qwen1.5-MoE layer 5, W5A5):
  Expert 0: gate=W4A4-g128, up=W4A4-g128, down=W4A4-g128
  Expert 1: gate=W4A4-g128, up=W4A4-g128, down=W8A8 (敏感 down_proj 获更高精度)
  Expert 22: gate=W8A8, up=W8A8, down=W8A8 (高频 expert 全 8-bit)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖：(1) 量化敏感度通过 Euclidean distance 在校准集上测量，每个 linear block 依次量化评估；(2) ILP solver（如 Gurobi）离线求解最优方案；(3) 量化使用 GPTQ + randomized Hadamard 变换提高精度；(4) 系统层面需要支持混合精度的 Group-GEMM kernel 来实际加速。超参数 r 在 weight-only 极低比特下设为 r=1（精度优先），weight-activation 下设 r=0.75（平衡）。MxMoE 在 2.25-bit weight-only 下 WikiText2 PPL 比 GPTQ 低 2.4（DeepSeekV2-Lite），W5A5 比 FP16 快 3.4×、比 uniform W8A8 快 29.4%。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---
