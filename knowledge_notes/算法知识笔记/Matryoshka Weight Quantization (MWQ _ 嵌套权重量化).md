## Matryoshka Weight Quantization (MWQ / 嵌套权重量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Matryoshka Weight Quantization (MWQ) 是 D2MoE 提出的多步量化技术，通过渐进式嵌套压缩，使不同 bit-width 的量化权重可以共享存储。其核心思想来源于套娃的嵌套结构：高 bit-width 权重在存储上天然包含低 bit-width 权重，无需为每个 bit-width 独立存储一份完整权重。

MWQ 分两步执行：
1. **Asymmetric Quantization 到最低 bit-width b₁**（如 INT2）：以 group-wise（group_size=128）的方式，最小化量化后输出误差 ∥WX - Ŵ_{b₁}X∥₂²，得到量化权重 Q_W_{b₁}、scale factor s_{b₁}、zero-point z_{b₁}
2. **Binary Residual Quantization 渐进增加 bit-width**：对残差 R_{b₁} = W - Ŵ_{b₁}，逐步将其量化为 +1/-1 的 binary 增量权重 Q_W_{b_k}，每步增加 1 bit，最终 b_K = b₁ + (K-1)。反量化时 Ŵ_{b_k} = Ŵ_{b₁} + Σ_{i=2}^{b_k} s_{b_i} · Q_W_{b_i}

这意味着存储 INT2/3/4 时仅需：一份 INT2 base + 两个 1-bit residual + 对应 scale factors，存储量接近 INT4 而非 INT2+INT3+INT4 之和。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 D2MoE-V1 (b₁=2, b_K=4, group_size=128) 为例：

```
=== MWQ 量化 Pipeline ===
输入: FP16 expert weight W ∈ R^{s×h}, calibration data X ∈ R^{h×r},
      Hessian regularizer λ, block size γ (e.g., 128)

Step 1: Cholesky decomposition for error compensation
    H^c = Cholesky((2XX^T + λI)^{-1})  # GPTQ-style correction matrix

Step 2: Asymmetric quantization to b₁=2
    对每组尺寸为 128 的元素:
        z_b1, s_b1 = argmin ||W_group · X_group - Ŵ_b1 · X_group||₂²
        Q_W_{b1} = round(W / s_b1 + z_b1)  # 量化到 INT2
        Ŵ_{b1} = (Q_W_{b1} - z_b1) · s_b1  # 反量化
        R_b1 = W - Ŵ_{b1}                  # 残差
    对后续 block 用 H^c 进行 block-wise error compensation (类似 GPTQ)

Step 3: Binary residual quantization: b₁=2 → b₂=3
    对每组:
        s_b2 = argmin ||R_b1 · X_group - s_b2 · Q_W_{b2} · X_group||₂²
        Q_W_{b2} = round(R_b1 / s_b2)  # 得到 +1/-1 的 binary 权重
        Ŵ_{b2} = (Q_W_{b1} - z_b1) · s_b1 + s_b2 · Q_W_{b2}  # INT3 重构
    
Step 4: Binary residual quantization: b₂=3 → b₃=4
    同上，对 R_b2 = W - Ŵ_{b2} 执行:
        s_b3 = argmin, Q_W_{b3} = round(R_b2 / s_b3)
        Ŵ_{b3} = Ŵ_{b1} + s_b2·Q_W_{b2} + s_b3·Q_W_{b3}  # INT4 重构

输出: {Q_W_{b1}, z_b1, s_b1} ∪ {(Q_W_{b_i}, s_b_i)}_{i=2}^{K}
```

**存储对比**：
- 传统方法（INT2+3+4 独立）：存储 INT2 完整权重 + INT3 完整权重 + INT4 完整权重
- MWQ：存储 1 份 INT2 base + (K-1) 组 1-bit residual + scale factors
- 例如 LLaMA-MoE-3.5B：传统方法 ~9.62GB（INT2/3/4），MWQ ~4.48GB（接近 INT4）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 使用 PyTorch + CUDA 实现。MWQ 属于离线预处理阶段（deployment 前执行一次），使用 C4 calibration dataset（128 random 2048-token segments）。MWQ 借鉴 GPTQ 的 block-wise error compensation，但去掉了 column-level error correction 以降低计算开销。离线阶段在 GPU server (2×A6000) 上执行：LLaMA-MoE-3.5B MWQ 耗时 ~10 min (batch_size=16)，Mixtral 8×7B 耗时 ~20 min (batch_size=4)。

MWQ 的核心使用场景：需要在端侧设备上同时支持多种 bit-width 的 MoE 推理，且内存极度受限（6GB-64GB）时，避免多版本权重存储爆炸。MWQ 的嵌套结构还天然支持低 bit-width 权重的高频复用——多个需要不同 bit-width 的请求可以共享 base 权重，仅额外加载各自需要的 residual bits。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving
