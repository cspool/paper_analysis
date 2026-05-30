## Expert-Specific Operators (ESMM, ESS, ESTMM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Specific Operators 是 HEXA-MoE 提出的替代传统 GeMM/grouped GeMM 的 MoE 计算范式。核心洞察：传统 MoE 使用 GeMM 接口时，由于各 expert 的 workload 动态变化，必须通过 token padding（填充到 capacity）或 discarding（丢弃超出 capacity 的 token）来构造规整的 batch，这产生了冗余 FLOPs 和冗余内存。Expert-Specific Operators 将 MoE 计算从 "GeMM 视角"（先重排为规则 batch 再调 GeMM）重新定义为 "Expert-Specific 视角"（不重排 token，直接做 expert-wise 计算）。三个基本算子：

- **ESMM (Expert-Specific Matrix Multiplication)**：给定输入 x [N, D_i]、权重 W [E, D_i, D_o]、偏置 b [E, D_o] 和 routing choice R(x) [N]，输出 y [N, D_o]，其中 y_i = x_i @ W_{R(x_i)} + b_{R(x_i)}。每个 token 仅与其路由 expert 的权重做矩阵乘法，无需 padding。
- **ESS (Expert-Specific Summation)**：给定输入 x [N, D] 和 routing choice R(x) [N]，输出 y [E, D]，其中 y[e] = Σ_{i: R(x_i)=e} x_i。按 expert 分组累加，用于 backward 中计算 bias 梯度。
- **ESTMM (Expert-Specific Transposed Matrix Multiplication)**：给定两个输入 x1 [N, D1]、x2 [N, D2]（共享 routing choice R(x)），输出 y [E, D1, D2]，其中 y[e, i, j] = Σ_{m: R(x_m)=e} x1[m,i] · x2[m,j]。expert-wise 外积累加，用于 backward 中计算权重梯度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以 top-1 routing 的 MoE 层为例，对比传统 GeMM 方法和 Expert-Specific Operators 方法：

```
# === 传统 GeMM 方法（Tutel）===
# Forward: 需要 dispatch + token padding
for each expert e:
    tokens_e = dispatch(x, R, e)              # 收集路由到 expert e 的 token
    tokens_e_padded = pad(tokens_e, capacity)  # padding 到 expert capacity
    y1_e = GeMM(tokens_e_padded, W1[e])       # [capacity, D_mid]
    y2_e = F(y1_e)                             # 激活
    y_e = GeMM(y2_e, W2[e])                    # [capacity, D_o]
y = combine({y_e})                            # 按原始顺序重组，丢弃 padding 部分

# === Expert-Specific Operators 方法（HEXA-MoE）===
# Forward: in-place 计算，无需 padding/dispatch/combine
y1 = ESMM(x, W1, b1, R(x))                   # [N, D_mid]
y2 = F(y1)                                     # 激活函数（如 GELU）
y  = ESMM(y2, W2, b2, R(x))                   # [N, D_o]

# Backward: auto-diff 提供 ∂ℓ/∂y
∂ℓ/∂b2 = ESS(∂ℓ/∂y, R(x))                     # [E, D_o]
∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x))               # [E, D_mid, D_o]
∂ℓ/∂y2 = ESMM(∂ℓ/∂y, W2^T, null, R(x))        # [N, D_mid]
∂ℓ/∂y1 = ∂ℓ/∂y2 ⊙ F'(y1)                      # element-wise
∂ℓ/∂b1 = ESS(∂ℓ/∂y1, R(x))                    # [E, D_mid]
∂ℓ/∂W1 = ESTMM(x, ∂ℓ/∂y1, R(x))               # [E, D_i, D_mid]
∂ℓ/∂x  = ESMM(∂ℓ/∂y1, W1^T, null, R(x))       # [N, D_i]
```

Top-k routing 扩展：对 k 个 routing choice 分别执行 ESMM，输出为 k 个 ESMM 结果的累加。中间结果 tensor 的内存分配仅扩展为 k 倍。使用 atomicAdd 聚合各 expert 对同一 token 的贡献。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Expert-Specific Operators 在 HEXA-MoE 中通过 CUDA kernel 实现（也提供 Triton 实现）。核心依赖 **re-index vector**（按 routing choice 重排 token indices）作为 I/O 指导，使同 expert 的 token 在内存中逻辑连续，提高 GPU 访存局部性。ESMM kernel 使用 nvcuda::wmma 接口调用 Tensor Core 做 16×16×16 矩阵乘法。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

使用方式：替代 PyTorch 中 MoE 层的标准 nn.Linear + routing 组合。HEXA-MoE 提供 `hexa_moe.moe` 模块，通过 `MoE_Cascaded` 类构建 MoE 层，内部自动使用 ESMM/ESS/ESTMM 替代 GeMM。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
