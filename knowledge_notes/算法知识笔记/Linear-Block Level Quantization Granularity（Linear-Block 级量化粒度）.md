## Linear-Block Level Quantization Granularity（Linear-Block 级量化粒度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Linear-Block Level Quantization Granularity 是 MxMoE 提出的在 MoE 模型中分配量化位宽的最小单元：不是整个 expert，也不是整个模型层，而是 expert 内部的每个线性投影块（gate_proj, up_proj, down_proj）。传统 MoE 量化工作（如 MC-MoE）以 expert 为粒度分配位宽——同一 expert 的所有 linear block 使用相同精度。但 MxMoE 通过量化敏感度分析（Fig. 1a）发现：同一 expert 内 gate_proj 和 down_proj 对量化的敏感度可能差异很大，统一位宽要么对不敏感 block 浪费精度预算，要么对敏感 block 精度不足。Linear-block 粒度允许更细粒度的精度分配——在同一 expert 内，不敏感的 gate_proj 用 W4A4，敏感的 down_proj 用 W8A8。实验表明 linear-block 粒度一致优于 expert 级（Table 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化粒度对比：

```
Expert-Level Granularity (例如 MC-MoE):
  Expert i: 所有 gate/up/down 共享同一量化方案
  例如 Expert 40: W4A4 应用于 gate+up+down

Linear-Block-Level Granularity (MxMoE):
  Expert i:
    gate_proj → 独立选择量化方案
    up_proj   → 独立选择量化方案
    down_proj → 独立选择量化方案
  例如 Expert 40:
    gate_proj → W4A4 (不敏感)
    up_proj   → W4A4 (不敏感)
    down_proj → W8A8 (敏感，需更高精度)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 MxMoE ILP 中，变量 x_{i,j,k} 的索引 j ∈ {1,2,3} 对应 3 个 linear block，每个独立求解。更细的粒度带来更多变量（E×3×|S| vs E×|S|），但 ILP 仍在可解范围内。与更细的 channel-level 或 element-level 粒度（如 SqueezeLLM）相比，linear-block 粒度避免了 irregular memory access 和 bitwidth lookup 的额外计算开销，保持良好的硬件效率。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---
